"""
Step 6mm: Stage 1 Multimodal Training (Polyvore, text+image, D_ITEM=2411)
  Pair files store indices; feature matrix loaded once into memory.
  Output: stage1_mm_model.pt, stage1_mm_train_log.csv
"""

import os, csv, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D_ITEM  = 2411
D_PROJ  = 256
D_PAIR  = D_PROJ*4 + 1 + 5    # 1030
BATCH   = 2048
EPOCHS  = 30
LR      = 3e-4
WD      = 1e-4
PATIENCE = 5

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Load feature matrix (once) ─────────────────────────
print("\nLoading poly_mm_feats.npy ...")
t0 = time.time()
poly_feats = torch.from_numpy(
    np.load(os.path.join(BASE_DIR, "poly_mm_feats.npy"))
).float()   # (142480, 2411)
print(f"  poly_feats: {poly_feats.shape}  [{time.time()-t0:.1f}s]")


# ── Dataset ────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, path, feats):
        data = torch.load(path, weights_only=False)
        self.idx_a = data['idx_a']
        self.idx_b = data['idx_b']
        self.labels = data['labels'].float()
        self.meta   = data.get('meta_pair', torch.zeros(len(self.labels), 5))
        self.feats  = feats   # shared reference, no copy

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        fa = self.feats[self.idx_a[i]]
        fb = self.feats[self.idx_b[i]]
        return fa, fb, self.labels[i], self.meta[i]


# ── Model ──────────────────────────────────────────────
class ProjectionMLP(nn.Module):
    def __init__(self, d_in=D_ITEM, d_out=D_PROJ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, d_out), nn.LayerNorm(d_out), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class FashFusionStage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = ProjectionMLP()
        self.fusion = nn.Sequential(
            nn.Linear(D_PAIR, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),   nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),     nn.Sigmoid(),
        )

    def forward(self, fa, fb, meta):
        ea = self.proj(fa); eb = self.proj(fb)
        cos = torch.cosine_similarity(ea, eb, dim=1, eps=1e-8).unsqueeze(1)
        pf  = torch.cat([ea, eb, (ea-eb).abs(), ea*eb, cos, meta], dim=1)
        return self.fusion(pf).squeeze(1)


# ── Metrics ────────────────────────────────────────────
def compute_metrics(model, loader, criterion):
    model.eval()
    losses, all_lab, all_pred = [], [], []
    with torch.no_grad():
        for fa, fb, lab, meta in loader:
            fa, fb, lab, meta = fa.to(DEVICE), fb.to(DEVICE), lab.to(DEVICE), meta.to(DEVICE)
            pred = model(fa, fb, meta)
            losses.append(criterion(pred, lab).item())
            all_lab.append(lab.cpu()); all_pred.append(pred.cpu())
    all_lab  = torch.cat(all_lab).numpy()
    all_pred = torch.cat(all_pred).numpy()
    from sklearn.metrics import roc_auc_score, f1_score
    auc = roc_auc_score(all_lab, all_pred)
    f1  = f1_score(all_lab, (all_pred >= 0.5).astype(int))
    return np.mean(losses), auc, f1


# ── Data ───────────────────────────────────────────────
print("Loading pair datasets ...")
train_ds = PairDataset(os.path.join(BASE_DIR, "poly_pairs_mm_train.pt"), poly_feats)
val_ds   = PairDataset(os.path.join(BASE_DIR, "poly_pairs_mm_val.pt"),   poly_feats)
test_ds  = PairDataset(os.path.join(BASE_DIR, "poly_pairs_mm_test.pt"),  poly_feats)
print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)


# ── Train ──────────────────────────────────────────────
model     = FashFusionStage1().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
criterion = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Stage 1 MM training  D_ITEM={D_ITEM} ...")

log_path   = os.path.join(BASE_DIR, "stage1_mm_train_log.csv")
best_auc   = 0.0
best_ep    = 0
no_improve = 0

with open(log_path, 'w', newline='') as f:
    csv.writer(f).writerow(["epoch","train_loss","val_loss","val_auc","val_f1","lr","elapsed"])

for ep in range(1, EPOCHS+1):
    model.train()
    t0 = time.time()
    train_loss = 0.0

    for fa, fb, lab, meta in train_loader:
        fa, fb, lab, meta = fa.to(DEVICE), fb.to(DEVICE), lab.to(DEVICE), meta.to(DEVICE)
        optimizer.zero_grad()
        pred = model(fa, fb, meta)
        loss = criterion(pred, lab)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss, val_auc, val_f1 = compute_metrics(model, val_loader, criterion)
    elapsed = time.time() - t0
    scheduler.step(val_auc)
    lr_cur = optimizer.param_groups[0]['lr']

    print(f"  Ep {ep:02d}/{EPOCHS} | TrLoss={train_loss:.4f} ValLoss={val_loss:.4f} "
          f"ValAUC={val_auc:.4f} ValF1={val_f1:.4f} LR={lr_cur:.2e} [{elapsed:.0f}s]")

    with open(log_path, 'a', newline='') as f:
        csv.writer(f).writerow([ep, f"{train_loss:.4f}", f"{val_loss:.4f}",
                                 f"{val_auc:.4f}", f"{val_f1:.4f}", f"{lr_cur:.2e}", f"{elapsed:.0f}"])

    if val_auc > best_auc:
        best_auc = val_auc; best_ep = ep; no_improve = 0
        torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                    'val_auc': val_auc, 'val_f1': val_f1, 'D_ITEM': D_ITEM, 'D_PROJ': D_PROJ},
                   os.path.join(BASE_DIR, "stage1_mm_model.pt"))
        print(f"    ** Best saved (AUC={best_auc:.4f})")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {ep}"); break

# ── Test eval ──────────────────────────────────────────
print(f"\nLoading best model (epoch {best_ep}) ...")
ckpt = torch.load(os.path.join(BASE_DIR, "stage1_mm_model.pt"), weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
test_loss, test_auc, test_f1 = compute_metrics(model, test_loader, criterion)

print(f"\nStage 1 MM Results:")
print(f"  Best Val AUC : {best_auc:.4f}  (epoch {best_ep})")
print(f"  Test AUC     : {test_auc:.4f}")
print(f"  Test F1      : {test_f1:.4f}")
print("  Next -> step7mm_hnm_pcompat.py")
