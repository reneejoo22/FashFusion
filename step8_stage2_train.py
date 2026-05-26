"""
Step 8: Stage 2 - Co-purchase prediction MLP (H&M)

Transfer: Stage 1 ProjectionMLP weights -> Stage 2 init (then fine-tune)
New feature: p_compatible(1) appended to pair feature vector
Pair dim: [A|B||A-B||A*B|cos|meta(5)|p_compat(1)] = 1030d

Label: 1 = co-purchased, 0 = random negative pair

Outputs:
  stage2_model.pt
  stage2_train_log.csv
"""

import os, time
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

D_ITEM     = 1131
D_EMB      = 256
D_META     = 5
D_PAIR_S2  = D_EMB * 4 + 1 + D_META + 1   # 1031 (adds p_compatible)

BATCH_SIZE = 1024
EPOCHS     = 25
LR         = 1e-4        # lower LR for fine-tuning
LR_PROJ    = 3e-5        # even lower for transferred projection
WD         = 1e-4
PATIENCE   = 5
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────
class ProjectionMLP(nn.Module):
    def __init__(self, d_in=D_ITEM, d_out=D_EMB):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, d_out), nn.LayerNorm(d_out),
        )
    def forward(self, x): return self.net(x)


class FashFusionStage2(nn.Module):
    """
    Stage 2: p_co_purchase prediction.
    Initialized from Stage 1 projection; adds p_compatible as extra signal.
    """
    def __init__(self, d_item=D_ITEM, d_emb=D_EMB, d_meta=D_META):
        super().__init__()
        self.proj = ProjectionMLP(d_item, d_emb)
        d_pair = d_emb * 4 + 1 + d_meta + 1  # +1 for p_compatible

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

    def forward(self, item_A, item_B, meta_pair, p_compatible):
        eA  = self.proj(item_A)
        eB  = self.proj(item_B)
        cos = F.cosine_similarity(eA, eB, dim=1, eps=1e-8).unsqueeze(1)
        p_c = p_compatible.unsqueeze(1) if p_compatible.dim() == 1 else p_compatible

        pair_feat = torch.cat(
            [eA, eB, torch.abs(eA - eB), eA * eB, cos, meta_pair, p_c],
            dim=1
        )
        return self.fusion(pair_feat).squeeze(1)


# ─────────────────────────────────────────────────────────
# Load Stage 1 weights -> init Stage 2 projection
# ─────────────────────────────────────────────────────────
model = FashFusionStage2().to(DEVICE)

ckpt1 = torch.load(os.path.join(BASE_DIR, "stage1_model.pt"), weights_only=True)
model.proj.load_state_dict(
    {k.replace("proj.", ""): v
     for k, v in ckpt1["model_state"].items() if k.startswith("proj.")}
)
print(f"  Stage 1 projection weights loaded (epoch {ckpt1['epoch']})")
print(f"  Stage 2 model params: {sum(p.numel() for p in model.parameters()):,}")


# ─────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────
def load_hnm_split(name):
    data = torch.load(os.path.join(BASE_DIR, f"hnm_pairs_{name}.pt"),
                      weights_only=True)
    ds = TensorDataset(
        data["item_A"].to(DEVICE),
        data["item_B"].to(DEVICE),
        data["meta_pair"].to(DEVICE),
        data["p_compatible"].to(DEVICE),
        data["labels"].to(DEVICE),
    )
    shuffle = (name == "train")
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      drop_last=(name == "train"))

print("\nLoading H&M datasets ...")
train_loader = load_hnm_split("train")
val_loader   = load_hnm_split("val")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val   batches: {len(val_loader)}")


# ─────────────────────────────────────────────────────────
# Optimizer: different LR for projection (transferred) vs fusion (new)
# ─────────────────────────────────────────────────────────
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW([
    {"params": model.proj.parameters(),   "lr": LR_PROJ},
    {"params": model.fusion.parameters(), "lr": LR},
], weight_decay=WD)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)


# ─────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer=None, criterion=None, train=True):
    model.train(train)
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(train):
        for item_A, item_B, meta, p_c, labels in loader:
            preds = model(item_A, item_B, meta, p_c)

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
    auc    = roc_auc_score(labels_np, preds_np)
    pr_auc = average_precision_score(labels_np, preds_np)
    f1     = f1_score(labels_np, (preds_np >= 0.5).astype(int))
    avg_loss = total_loss / len(labels_np) if train else 0.0
    return avg_loss, auc, pr_auc, f1


print(f"\n{'Ep':>3} {'TrainLoss':>10} {'TrainAUC':>9} {'ValAUC':>8} "
      f"{'ValPR-AUC':>10} {'ValF1':>7}  {'Time':>6}")
print("-" * 68)

best_val_auc = 0.0
patience_cnt = 0
log_rows     = []
best_ckpt    = os.path.join(BASE_DIR, "stage2_model.pt")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_auc, tr_pr, tr_f1 = run_epoch(
        model, train_loader, optimizer, criterion, train=True)
    vl_loss, vl_auc, vl_pr, vl_f1 = run_epoch(
        model, val_loader, train=False)

    scheduler.step(vl_auc)
    elapsed = time.time() - t0

    marker = ""
    if vl_auc > best_val_auc:
        best_val_auc = vl_auc
        patience_cnt = 0
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "val_auc":     vl_auc,
            "val_pr_auc":  vl_pr,
            "val_f1":      vl_f1,
        }, best_ckpt)
        marker = " *"
    else:
        patience_cnt += 1

    print(f"{epoch:>3}  {tr_loss:>10.4f}  {tr_auc:>9.4f}  {vl_auc:>8.4f}  "
          f"{vl_pr:>10.4f}  {vl_f1:>7.4f}  {elapsed:>5.1f}s{marker}")

    log_rows.append({
        "epoch": epoch, "train_loss": tr_loss, "train_auc": tr_auc,
        "val_auc": vl_auc, "val_pr_auc": vl_pr, "val_f1": vl_f1,
    })

    if patience_cnt >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch}")
        break


# ─────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STAGE 2 RESULTS")
print("=" * 60)
ckpt2 = torch.load(best_ckpt, weights_only=True)
print(f"  Best epoch  : {ckpt2['epoch']}")
print(f"  Val AUC     : {ckpt2['val_auc']:.4f}")
print(f"  Val PR-AUC  : {ckpt2['val_pr_auc']:.4f}")
print(f"  Val F1      : {ckpt2['val_f1']:.4f}")

log_df = pd.DataFrame(log_rows)
log_path = os.path.join(BASE_DIR, "stage2_train_log.csv")
log_df.to_csv(log_path, index=False)
print(f"  Train log   : {log_path}")
print(f"  Model saved : {best_ckpt}")
print()
print("  Pipeline complete! Text branch end-to-end done.")
print("  Next -> image branch (Step 3) or ablation experiments (E1-E8)")
