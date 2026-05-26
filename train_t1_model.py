"""
T1 모델 단독 훈련 스크립트
  Full MM (2411d) + p_compat, proj 랜덤 초기화
  Output: stage2_t1_model.pt
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

D_ITEM = 2411; D_PROJ = 256; D_META = 5
D_PAIR = D_PROJ*4 + 1 + D_META + 1   # 1031 (p_compat 포함)
EPOCHS = 30; LR = 1e-4; WD = 1e-4; PATIENCE = 5; BATCH = 2048

# ── 데이터 로드 ────────────────────────────────────────
print("\nLoading data ...")
t0 = time.time()
hnm_feats = torch.from_numpy(
    np.load(os.path.join(BASE_DIR, "hnm_mm_feats.npy"))
).float()

trn = torch.load(os.path.join(BASE_DIR, "hnm_pairs_mm_train.pt"), weights_only=False)
val = torch.load(os.path.join(BASE_DIR, "hnm_pairs_mm_val.pt"),   weights_only=False)

trn_idx_a   = trn['idx_a'];  trn_idx_b   = trn['idx_b']
trn_labels  = trn['labels']; trn_meta    = trn.get('meta_pair',    torch.zeros(len(trn['labels']), D_META))
trn_pcompat = trn.get('p_compatible', torch.zeros(len(trn['labels'])))

val_idx_a   = val['idx_a'];  val_idx_b   = val['idx_b']
val_labels  = val['labels']; val_meta    = val.get('meta_pair',    torch.zeros(len(val['labels']), D_META))
val_pcompat = val.get('p_compatible', torch.zeros(len(val['labels'])))

print(f"  Loaded {time.time()-t0:.1f}s  Trn={len(trn_labels):,}  Val={len(val_labels):,}")


# ── Dataset ────────────────────────────────────────────
class PairDS(Dataset):
    def __init__(self, idx_a, idx_b, labels, meta, feats, pcompat):
        self.idx_a = idx_a; self.idx_b = idx_b
        self.labels = labels.float(); self.meta = meta
        self.feats = feats; self.pc = pcompat
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return (self.feats[self.idx_a[i]], self.feats[self.idx_b[i]],
                self.labels[i], self.meta[i], self.pc[i])


# ── 모델 ───────────────────────────────────────────────
class FusionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(D_ITEM, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, D_PROJ), nn.LayerNorm(D_PROJ), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(D_PAIR, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),   nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),     nn.Sigmoid(),
        )
    def forward(self, fa, fb, meta, pc):
        ea = self.proj(fa); eb = self.proj(fb)
        cos = torch.cosine_similarity(ea, eb, dim=1, eps=1e-8).unsqueeze(1)
        pf  = torch.cat([ea, eb, (ea-eb).abs(), ea*eb, cos, meta, pc.unsqueeze(1)], dim=1)
        return self.fusion(pf).squeeze(1)


# ── 훈련 ───────────────────────────────────────────────
t_ds  = PairDS(trn_idx_a, trn_idx_b, trn_labels, trn_meta, hnm_feats, trn_pcompat)
v_ds  = PairDS(val_idx_a,  val_idx_b, val_labels, val_meta, hnm_feats, val_pcompat)
t_ldr = DataLoader(t_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
v_ldr = DataLoader(v_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

model = FusionMLP().to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
crit  = nn.BCELoss()
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)

print(f"\nT1 training  D_ITEM={D_ITEM}  D_PAIR={D_PAIR}  (random proj init)")
save_path  = os.path.join(BASE_DIR, "stage2_t1_model.pt")
best_auc   = 0.0; no_imp = 0

for ep in range(1, EPOCHS+1):
    model.train(); tl = 0
    for fa, fb, lab, meta, pc in t_ldr:
        fa, fb, lab = fa.to(DEVICE), fb.to(DEVICE), lab.to(DEVICE)
        meta, pc = meta.to(DEVICE), pc.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(fa, fb, meta, pc), lab)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); tl += loss.item()

    model.eval(); ap, al = [], []
    with torch.no_grad():
        for fa, fb, lab, meta, pc in v_ldr:
            fa, fb = fa.to(DEVICE), fb.to(DEVICE)
            meta, pc = meta.to(DEVICE), pc.to(DEVICE)
            ap.append(model(fa, fb, meta, pc).cpu().numpy())
            al.append(lab.numpy())

    ep_preds = np.concatenate(ap); ep_labs = np.concatenate(al)
    ep_auc   = roc_auc_score(ep_labs, ep_preds)
    sched.step(ep_auc)
    print(f"  Ep{ep:02d}  TrLoss={tl/len(t_ldr):.4f}  ValAUC={ep_auc:.4f}")

    if ep_auc > best_auc:
        best_auc = ep_auc; no_imp = 0
        best_preds = ep_preds; best_labs = ep_labs
        torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                    'val_auc': ep_auc, 'D_ITEM': D_ITEM, 'D_PROJ': D_PROJ,
                    'use_pcompat': True}, save_path)
        print(f"    ** Best saved (AUC={best_auc:.4f})")
    else:
        no_imp += 1
        if no_imp >= PATIENCE:
            print(f"  Early stop at epoch {ep}"); break

prauc = average_precision_score(best_labs, best_preds)
f1    = f1_score(best_labs, (np.array(best_preds) >= 0.5).astype(int))
print(f"\nT1 Final Results:")
print(f"  AUC={best_auc:.4f}  PR-AUC={prauc:.4f}  F1={f1:.4f}")
print(f"  Saved: {save_path}")
