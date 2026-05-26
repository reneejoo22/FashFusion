"""
Step 7mm: Re-compute H&M p_compatible using multimodal Stage 1 model
  Pair files store indices; hnm_mm_feats.npy loaded once.
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D_ITEM = 2411; D_PROJ = 256; D_PAIR = D_PROJ*4 + 1 + 5

print(f"Device: {DEVICE}")

class ProjectionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_ITEM, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, D_PROJ), nn.LayerNorm(D_PROJ), nn.ReLU(),
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

print("Loading stage1_mm_model.pt ...")
model = FashFusionStage1().to(DEVICE)
ckpt  = torch.load(os.path.join(BASE_DIR, "stage1_mm_model.pt"), weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"  epoch={ckpt['epoch']}  Val AUC={ckpt['val_auc']:.4f}")

print("Loading hnm_mm_feats.npy ...")
hnm_feats = torch.from_numpy(np.load(os.path.join(BASE_DIR, "hnm_mm_feats.npy"))).float()

class PairDataset(Dataset):
    def __init__(self, path, feats):
        data = torch.load(path, weights_only=False)
        self.idx_a  = data['idx_a']
        self.idx_b  = data['idx_b']
        self.labels = data['labels'].float()
        self.meta   = data.get('meta_pair', torch.zeros(len(self.labels), 5))
        self.feats  = feats
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return (self.feats[self.idx_a[i]], self.feats[self.idx_b[i]],
                self.labels[i], self.meta[i], i)

all_scores, all_labels = [], []

for split in ['train', 'val']:
    path = os.path.join(BASE_DIR, f"hnm_pairs_mm_{split}.pt")
    ds   = PairDataset(path, hnm_feats)
    loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0, pin_memory=True)
    print(f"\nScoring {split} ({len(ds):,} pairs) ...")
    t0 = time.time()
    scores, labels = [], []
    with torch.no_grad():
        for fa, fb, lab, meta, _ in loader:
            fa, fb, meta = fa.to(DEVICE), fb.to(DEVICE), meta.to(DEVICE)
            sc = model(fa, fb, meta).cpu().numpy()
            scores.append(sc); labels.append(lab.numpy())
    scores = np.concatenate(scores); labels = np.concatenate(labels)
    all_scores.append(scores); all_labels.append(labels)
    print(f"  pos={scores[labels==1].mean():.4f}  neg={scores[labels==0].mean():.4f}  [{time.time()-t0:.1f}s]")

    # Update pt file with new p_compatible
    data = torch.load(path, weights_only=False)
    data['p_compatible'] = torch.from_numpy(scores).float()
    torch.save(data, path)
    print(f"  Updated p_compatible in {os.path.basename(path)}")

all_scores_np = np.concatenate(all_scores)
all_labels_np = np.concatenate(all_labels)
np.save(os.path.join(BASE_DIR, "hnm_p_compat_mm.npy"), all_scores_np)
print(f"\nSaved hnm_p_compat_mm.npy  ({len(all_scores_np):,} scores)")
print("  Next -> step8mm_stage2_train.py")
