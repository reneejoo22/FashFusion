"""
Step 6: Stage 1 - Pairwise Fusion MLP training (Polyvore compatibility)

Architecture:
  ProjectionMLP  (shared): 1131 -> 512 -> 256  [LayerNorm + ReLU + Dropout]
  PairwiseFusion          : [A|B||A-B||A*B|cos|meta] = 1029d
                            -> 512 -> 128 -> 1  [ReLU + Dropout + Sigmoid]

Outputs:
  stage1_model.pt     best checkpoint (by val AUC)
  stage1_train_log.csv  epoch-level metrics
"""

import os, pickle, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")

# ── Hyperparameters ──────────────────────────────────────
D_ITEM     = 1131
D_EMB      = 256
D_META     = 5
D_PAIR     = D_EMB * 4 + 1 + D_META   # 1029

BATCH_SIZE = 1024
EPOCHS     = 30
LR         = 3e-4
WD         = 1e-4
PATIENCE   = 5      # early stopping on val AUC
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────
class ProjectionMLP(nn.Module):
    """Shared item projector: D_ITEM -> D_EMB"""
    def __init__(self, d_in=D_ITEM, d_out=D_EMB):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, x):
        return self.net(x)


class FashFusionStage1(nn.Module):
    """
    Two-item pairwise fusion for compatibility prediction.
    p_compatible = P(A and B are stylistically compatible)
    """
    def __init__(self, d_item=D_ITEM, d_emb=D_EMB, d_meta=D_META):
        super().__init__()
        self.proj = ProjectionMLP(d_item, d_emb)
        d_pair = d_emb * 4 + 1 + d_meta  # 1029

        self.fusion = nn.Sequential(
            nn.Linear(d_pair, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, item_A, item_B, meta_pair):
        eA  = self.proj(item_A)                              # (B, 256)
        eB  = self.proj(item_B)                              # (B, 256)
        cos = F.cosine_similarity(eA, eB, dim=1, eps=1e-8)  # (B,)
        cos = cos.unsqueeze(1)                               # (B, 1)

        pair_feat = torch.cat(
            [eA, eB, torch.abs(eA - eB), eA * eB, cos, meta_pair],
            dim=1
        )  # (B, 1029)

        return self.fusion(pair_feat).squeeze(1)  # (B,)

    def embed(self, item):
        """Extract item embedding (inference)."""
        with torch.no_grad():
            return self.proj(item)


# ─────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────
def load_split(name):
    data = torch.load(os.path.join(BASE_DIR, f"poly_pairs_{name}.pt"),
                      weights_only=True)
    ds = TensorDataset(
        data["item_A"].to(DEVICE),
        data["item_B"].to(DEVICE),
        data["meta_pair"].to(DEVICE),
        data["labels"].to(DEVICE),
    )
    shuffle = (name == "train")
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      drop_last=(name == "train"))

print("\nLoading datasets ...")
train_loader = load_split("train")
val_loader   = load_split("val")
test_loader  = load_split("test")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val   batches: {len(val_loader)}")
print(f"  Test  batches: {len(test_loader)}")


# ─────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer=None, criterion=None, train=True):
    model.train(train)
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(train):
        for item_A, item_B, meta, labels in loader:
            preds = model(item_A, item_B, meta)

            if train:
                loss = criterion(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(labels)

            all_preds.append(preds.detach().cpu().float().numpy())
            all_labels.append(labels.cpu().numpy())

    preds_np  = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)
    auc  = roc_auc_score(labels_np, preds_np)
    pr_auc = average_precision_score(labels_np, preds_np)
    f1   = f1_score(labels_np, (preds_np >= 0.5).astype(int))
    avg_loss = total_loss / len(labels_np) if train else 0.0

    return avg_loss, auc, pr_auc, f1


# ─────────────────────────────────────────────────────────
# Initialize model, optimizer, scheduler
# ─────────────────────────────────────────────────────────
model     = FashFusionStage1().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel params: {total_params:,}")
print(f"  ProjectionMLP  : {sum(p.numel() for p in model.proj.parameters()):,}")
print(f"  FusionMLP      : {sum(p.numel() for p in model.fusion.parameters()):,}")


# ─────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────
print(f"\n{'Ep':>3} {'TrainLoss':>10} {'TrainAUC':>9} {'ValAUC':>8} "
      f"{'ValPR-AUC':>10} {'ValF1':>7} {'LR':>8}  {'Time':>6}")
print("-" * 75)

best_val_auc = 0.0
patience_cnt = 0
log_rows     = []
best_ckpt    = os.path.join(BASE_DIR, "stage1_model.pt")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    tr_loss, tr_auc, tr_pr, tr_f1 = run_epoch(
        model, train_loader, optimizer, criterion, train=True)
    vl_loss, vl_auc, vl_pr, vl_f1 = run_epoch(
        model, val_loader, train=False)

    scheduler.step(vl_auc)
    cur_lr = optimizer.param_groups[0]["lr"]
    elapsed = time.time() - t0

    marker = ""
    if vl_auc > best_val_auc:
        best_val_auc = vl_auc
        patience_cnt = 0
        torch.save({
            "epoch":      epoch,
            "model_state": model.state_dict(),
            "val_auc":    vl_auc,
            "val_pr_auc": vl_pr,
            "val_f1":     vl_f1,
            "hparams": {
                "D_ITEM": D_ITEM, "D_EMB": D_EMB, "D_META": D_META,
                "D_PAIR": D_PAIR, "LR": LR, "WD": WD,
            }
        }, best_ckpt)
        marker = " *"
    else:
        patience_cnt += 1

    print(f"{epoch:>3}  {tr_loss:>10.4f}  {tr_auc:>9.4f}  {vl_auc:>8.4f}  "
          f"{vl_pr:>10.4f}  {vl_f1:>7.4f}  {cur_lr:>8.2e}  {elapsed:>5.1f}s{marker}")

    log_rows.append({
        "epoch": epoch, "train_loss": tr_loss, "train_auc": tr_auc,
        "val_auc": vl_auc, "val_pr_auc": vl_pr, "val_f1": vl_f1, "lr": cur_lr,
    })

    if patience_cnt >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
        break


# ─────────────────────────────────────────────────────────
# Test evaluation with best checkpoint
# ─────────────────────────────────────────────────────────
print(f"\nLoading best checkpoint (val AUC={best_val_auc:.4f}) ...")
ckpt = torch.load(best_ckpt, weights_only=True)
model.load_state_dict(ckpt["model_state"])

_, test_auc, test_pr, test_f1 = run_epoch(model, test_loader, train=False)
print(f"\n{'='*50}")
print(f"  TEST RESULTS (official Polyvore benchmark)")
print(f"{'='*50}")
print(f"  AUC       : {test_auc:.4f}")
print(f"  PR-AUC    : {test_pr:.4f}")
print(f"  F1 (@0.5) : {test_f1:.4f}")
print(f"  Best epoch: {ckpt['epoch']}")

# Save train log
log_df = pd.DataFrame(log_rows)
log_path = os.path.join(BASE_DIR, "stage1_train_log.csv")
log_df.to_csv(log_path, index=False)
print(f"\n  Train log saved: {log_path}")
print(f"  Model saved   : {best_ckpt}")
print()
print("  Next -> Step 7: Stage 1 model inference on H&M pairs (p_compatible)")
