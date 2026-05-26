"""
Ablation & Baseline Experiments for FashFusion/QuadrupleNet
Results saved to ablation_results.csv

Experiments:
  E1: Random baseline
  E2: Popularity baseline (co-purchase freq)
  E3: Jaccard co-occurrence baseline
  E4: Text-only Stage2 (existing stage2_model.pt)
  E5: MM Stage2 no p_compatible (retrain fusion head only)
  E6: Image-only Stage2
  E7: Full MM Stage2 (existing stage2_mm_model.pt)  <- best
  E8: Stage2 MM without cross-domain transfer (random proj init)
"""

import os, csv, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

RESULTS = []

# ── Shared helpers ─────────────────────────────────────
def log(name, auc, prauc, f1, note=""):
    RESULTS.append({'model': name, 'val_auc': auc, 'val_prauc': prauc, 'val_f1': f1, 'note': note})
    print(f"  [{name}]  AUC={auc:.4f}  PR={prauc:.4f}  F1={f1:.4f}  {note}")

def eval_preds(labels, preds):
    auc   = roc_auc_score(labels, preds)
    prauc = average_precision_score(labels, preds)
    f1    = f1_score(labels, (preds >= 0.5).astype(int))
    return auc, prauc, f1

# ── Load shared data ───────────────────────────────────
print("\nLoading data ...")
t0 = time.time()

# Text-only HnM pair data (for E4)
hnm_train_txt = torch.load(os.path.join(BASE_DIR, "hnm_pairs_train.pt"), weights_only=False)
hnm_val_txt   = torch.load(os.path.join(BASE_DIR, "hnm_pairs_val.pt"),   weights_only=False)

# MM HnM pair data (indices + p_compatible)
hnm_val_mm   = torch.load(os.path.join(BASE_DIR, "hnm_pairs_mm_val.pt"),   weights_only=False)
hnm_train_mm = torch.load(os.path.join(BASE_DIR, "hnm_pairs_mm_train.pt"), weights_only=False)

# Feature matrices
hnm_mm_feats = torch.from_numpy(np.load(os.path.join(BASE_DIR, "hnm_mm_feats.npy"))).float()

# Split: text (0:1131) and image (1131:2411) portions
D_TEXT = 1131; D_IMG = 1280; D_ITEM = 2411
hnm_text_feats = hnm_mm_feats[:, :D_TEXT]   # (105542, 1131)
hnm_img_feats  = hnm_mm_feats[:, D_TEXT:]   # (105542, 1280)

val_labels = hnm_val_mm['labels'].numpy()
val_idx_a  = hnm_val_mm['idx_a']
val_idx_b  = hnm_val_mm['idx_b']
val_meta   = hnm_val_mm.get('meta_pair', torch.zeros(len(val_labels), 5))
val_pcomp  = hnm_val_mm.get('p_compatible', torch.zeros(len(val_labels)))

print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Val: {len(val_labels):,} pairs  pos={val_labels.mean()*100:.1f}%")


# ═══════════════════════════════════════════════════════
# E1: Random baseline
# ═══════════════════════════════════════════════════════
print("\n[E1] Random baseline ...")
rng = np.random.RandomState(42)
rand_preds = rng.rand(len(val_labels))
auc, prauc, f1 = eval_preds(val_labels, rand_preds)
log("E1_Random", auc, prauc, f1)


# ═══════════════════════════════════════════════════════
# E2: Popularity baseline (score = avg freq of both items)
# ═══════════════════════════════════════════════════════
print("\n[E2] Popularity baseline ...")
import pickle
with open(os.path.join(BASE_DIR, "article_id_list.pkl"), 'rb') as f:
    article_id_list = pickle.load(f)

# Build popularity from training labels
train_labels_txt = hnm_train_mm['labels'].numpy()
train_idx_a = hnm_train_mm['idx_a'].numpy()
train_idx_b = hnm_train_mm['idx_b'].numpy()

pop = np.zeros(len(article_id_list), dtype=np.float32)
for a, b, lab in zip(train_idx_a, train_idx_b, train_labels_txt):
    if lab == 1:
        pop[a] += 1; pop[b] += 1
pop = pop / (pop.max() + 1e-9)

pop_preds = (pop[val_idx_a.numpy()] + pop[val_idx_b.numpy()]) / 2
auc, prauc, f1 = eval_preds(val_labels, pop_preds)
log("E2_Popularity", auc, prauc, f1)


# ═══════════════════════════════════════════════════════
# E3: Co-occurrence Jaccard baseline
# ═══════════════════════════════════════════════════════
print("\n[E3] Jaccard co-occurrence baseline ...")
from collections import defaultdict

co = defaultdict(int); item_freq = defaultdict(int)
for a, b, lab in zip(train_idx_a, train_idx_b, train_labels_txt):
    if lab == 1:
        key = (min(int(a), int(b)), max(int(a), int(b)))
        co[key] += 1
        item_freq[int(a)] += 1; item_freq[int(b)] += 1

jaccard_preds = np.zeros(len(val_labels), dtype=np.float32)
for i, (a, b) in enumerate(zip(val_idx_a.numpy(), val_idx_b.numpy())):
    key = (min(int(a), int(b)), max(int(a), int(b)))
    c   = co.get(key, 0)
    union = item_freq.get(int(a), 0) + item_freq.get(int(b), 0) - c
    jaccard_preds[i] = c / union if union > 0 else 0.0

auc, prauc, f1 = eval_preds(val_labels, jaccard_preds)
log("E3_Jaccard", auc, prauc, f1)


# ═══════════════════════════════════════════════════════
# E4: Text-only Stage2 (load existing stage2_model.pt)
# ═══════════════════════════════════════════════════════
print("\n[E4] Text-only Stage2 (existing model) ...")

D_TEXT_PAIR = 256*4 + 1 + 5 + 1   # 1030 (text model uses p_compatible too)

class ProjMLPText(nn.Module):
    def __init__(self, d_in=D_TEXT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.LayerNorm(256), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)

class Stage2Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = ProjMLPText()
        self.fusion = nn.Sequential(
            nn.Linear(D_TEXT_PAIR, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),         nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),           nn.Sigmoid(),
        )
    def forward(self, fa, fb, meta, pc):
        ea = self.proj(fa); eb = self.proj(fb)
        cos = torch.cosine_similarity(ea, eb, dim=1, eps=1e-8).unsqueeze(1)
        pf  = torch.cat([ea, eb, (ea-eb).abs(), ea*eb, cos, meta, pc.unsqueeze(1)], dim=1)
        return self.fusion(pf).squeeze(1)

m_txt = Stage2Text().to(DEVICE)
ckpt  = torch.load(os.path.join(BASE_DIR, "stage2_model.pt"), weights_only=False)
m_txt.load_state_dict(ckpt['model_state_dict'])
m_txt.eval()

# Evaluate on val
val_fa_txt = hnm_val_txt['item_A'].to(DEVICE)
val_fb_txt = hnm_val_txt['item_B'].to(DEVICE)
val_m_txt  = hnm_val_txt.get('meta_pair', torch.zeros(len(val_labels), 5)).to(DEVICE)
val_pc_txt = hnm_val_txt.get('p_compatible', torch.zeros(len(val_labels))).to(DEVICE)
val_lab_txt = hnm_val_txt['labels'].numpy()

BSIZE = 4096; all_preds = []
with torch.no_grad():
    for i in range(0, len(val_lab_txt), BSIZE):
        preds = m_txt(val_fa_txt[i:i+BSIZE], val_fb_txt[i:i+BSIZE],
                      val_m_txt[i:i+BSIZE], val_pc_txt[i:i+BSIZE])
        all_preds.append(preds.cpu().numpy())
txt_preds = np.concatenate(all_preds)
auc, prauc, f1 = eval_preds(val_lab_txt, txt_preds)
log("E4_Text_Stage2", auc, prauc, f1, note="tfidf+bert+struct+p_compat (1131d)")


# ═══════════════════════════════════════════════════════
# E5: Image-only Stage2
# ═══════════════════════════════════════════════════════
print("\n[E5] Image-only Stage2 (train from scratch) ...")

D_IMG_PAIR = 256*4 + 1 + 5 + 1   # 1030

class ProjMLPImg(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_IMG, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),   nn.LayerNorm(256), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)

class Stage2Img(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = ProjMLPImg()
        self.fusion = nn.Sequential(
            nn.Linear(D_IMG_PAIR, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),          nn.Sigmoid(),
        )
    def forward(self, fa, fb, meta, pc):
        ea = self.proj(fa); eb = self.proj(fb)
        cos = torch.cosine_similarity(ea, eb, dim=1, eps=1e-8).unsqueeze(1)
        pf  = torch.cat([ea, eb, (ea-eb).abs(), ea*eb, cos, meta, pc.unsqueeze(1)], dim=1)
        return self.fusion(pf).squeeze(1)

class ImgPairDataset(Dataset):
    def __init__(self, mm_data, img_feats):
        self.idx_a   = mm_data['idx_a']
        self.idx_b   = mm_data['idx_b']
        self.labels  = mm_data['labels'].float()
        self.meta    = mm_data.get('meta_pair', torch.zeros(len(self.labels), 5))
        self.pcompat = mm_data.get('p_compatible', torch.zeros(len(self.labels)))
        self.feats   = img_feats
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return (self.feats[self.idx_a[i]], self.feats[self.idx_b[i]],
                self.labels[i], self.meta[i], self.pcompat[i])

img_train_ds = ImgPairDataset(hnm_train_mm, hnm_img_feats)
img_val_ds   = ImgPairDataset(hnm_val_mm,   hnm_img_feats)
img_train_ldr = DataLoader(img_train_ds, batch_size=2048, shuffle=True,  num_workers=0, pin_memory=True)
img_val_ldr   = DataLoader(img_val_ds,   batch_size=4096, shuffle=False, num_workers=0, pin_memory=True)

m_img = Stage2Img().to(DEVICE)
opt   = torch.optim.AdamW(m_img.parameters(), lr=1e-4, weight_decay=1e-4)
crit  = nn.BCELoss()
best_img_auc = 0; patience_count = 0

for ep in range(1, 21):
    m_img.train(); tl = 0
    for fa, fb, lab, meta, pc in img_train_ldr:
        fa, fb, lab = fa.to(DEVICE), fb.to(DEVICE), lab.to(DEVICE)
        meta, pc = meta.to(DEVICE), pc.to(DEVICE)
        opt.zero_grad()
        pred = m_img(fa, fb, meta, pc)
        loss = crit(pred, lab); loss.backward()
        torch.nn.utils.clip_grad_norm_(m_img.parameters(), 1.0)
        opt.step(); tl += loss.item()

    m_img.eval(); all_p, all_l = [], []
    with torch.no_grad():
        for fa, fb, lab, meta, pc in img_val_ldr:
            fa, fb = fa.to(DEVICE), fb.to(DEVICE)
            meta, pc = meta.to(DEVICE), pc.to(DEVICE)
            all_p.append(m_img(fa, fb, meta, pc).cpu().numpy())
            all_l.append(lab.numpy())
    ep_auc = roc_auc_score(np.concatenate(all_l), np.concatenate(all_p))
    print(f"    Img Ep{ep:02d} TrLoss={tl/len(img_train_ldr):.4f} ValAUC={ep_auc:.4f}")
    if ep_auc > best_img_auc:
        best_img_auc = ep_auc; patience_count = 0
        best_img_preds = np.concatenate(all_p); best_img_labels = np.concatenate(all_l)
    else:
        patience_count += 1
        if patience_count >= 4: print(f"    Early stop ep{ep}"); break

auc, prauc, f1 = eval_preds(best_img_labels, best_img_preds)
log("E5_Image_Stage2", auc, prauc, f1, note="img_1280 only")


# ═══════════════════════════════════════════════════════
# E6: MM Stage2 without p_compatible (no cross-domain transfer)
# ═══════════════════════════════════════════════════════
print("\n[E6] MM Stage2 without p_compatible transfer ...")

D_PAIR_NO_PC = 256*4 + 1 + 5   # 1029

class Stage2MM_NoPC(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(D_ITEM, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),    nn.LayerNorm(256), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(D_PAIR_NO_PC, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),          nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),            nn.Sigmoid(),
        )
    def forward(self, fa, fb, meta):
        ea = self.proj(fa); eb = self.proj(fb)
        cos = torch.cosine_similarity(ea, eb, dim=1, eps=1e-8).unsqueeze(1)
        pf  = torch.cat([ea, eb, (ea-eb).abs(), ea*eb, cos, meta], dim=1)
        return self.fusion(pf).squeeze(1)

class NoPCDataset(Dataset):
    def __init__(self, mm_data, feats):
        self.idx_a  = mm_data['idx_a']; self.idx_b = mm_data['idx_b']
        self.labels = mm_data['labels'].float()
        self.meta   = mm_data.get('meta_pair', torch.zeros(len(self.labels), 5))
        self.feats  = feats
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return (self.feats[self.idx_a[i]], self.feats[self.idx_b[i]], self.labels[i], self.meta[i])

nopc_train_ds  = NoPCDataset(hnm_train_mm, hnm_mm_feats)
nopc_val_ds    = NoPCDataset(hnm_val_mm,   hnm_mm_feats)
nopc_train_ldr = DataLoader(nopc_train_ds, batch_size=2048, shuffle=True,  num_workers=0, pin_memory=True)
nopc_val_ldr   = DataLoader(nopc_val_ds,   batch_size=4096, shuffle=False, num_workers=0, pin_memory=True)

m_nopc = Stage2MM_NoPC().to(DEVICE)
opt2   = torch.optim.AdamW(m_nopc.parameters(), lr=1e-4, weight_decay=1e-4)
best_nopc_auc = 0; pc2 = 0

for ep in range(1, 31):
    m_nopc.train(); tl = 0
    for fa, fb, lab, meta in nopc_train_ldr:
        fa, fb, lab, meta = fa.to(DEVICE), fb.to(DEVICE), lab.to(DEVICE), meta.to(DEVICE)
        opt2.zero_grad()
        pred = m_nopc(fa, fb, meta); loss = crit(pred, lab)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m_nopc.parameters(), 1.0)
        opt2.step(); tl += loss.item()

    m_nopc.eval(); all_p, all_l = [], []
    with torch.no_grad():
        for fa, fb, lab, meta in nopc_val_ldr:
            fa, fb, meta = fa.to(DEVICE), fb.to(DEVICE), meta.to(DEVICE)
            all_p.append(m_nopc(fa, fb, meta).cpu().numpy()); all_l.append(lab.numpy())
    ep_auc = roc_auc_score(np.concatenate(all_l), np.concatenate(all_p))
    print(f"    NoPC Ep{ep:02d} TrLoss={tl/len(nopc_train_ldr):.4f} ValAUC={ep_auc:.4f}")
    if ep_auc > best_nopc_auc:
        best_nopc_auc = ep_auc; pc2 = 0
        best_nopc_preds = np.concatenate(all_p); best_nopc_labels = np.concatenate(all_l)
    else:
        pc2 += 1
        if pc2 >= 5: print(f"    Early stop ep{ep}"); break

auc, prauc, f1 = eval_preds(best_nopc_labels, best_nopc_preds)
log("E6_MM_No_pcompat", auc, prauc, f1, note="MM 2411d, no cross-domain transfer")


# ═══════════════════════════════════════════════════════
# E7: Full MM Stage2 (existing stage2_mm_model.pt)
# ═══════════════════════════════════════════════════════
print("\n[E7] Full MM Stage2 (best model) ...")

D_PAIR_S2 = 256*4 + 1 + 5 + 1   # 1030

class Stage2MM(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(D_ITEM, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),    nn.LayerNorm(256), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(D_PAIR_S2, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),         nn.Sigmoid(),
        )
    def forward(self, fa, fb, meta, pc):
        ea = self.proj(fa); eb = self.proj(fb)
        cos = torch.cosine_similarity(ea, eb, dim=1, eps=1e-8).unsqueeze(1)
        pf  = torch.cat([ea, eb, (ea-eb).abs(), ea*eb, cos, meta, pc.unsqueeze(1)], dim=1)
        return self.fusion(pf).squeeze(1)

class FullMMDataset(Dataset):
    def __init__(self, mm_data, feats):
        self.idx_a   = mm_data['idx_a']; self.idx_b = mm_data['idx_b']
        self.labels  = mm_data['labels'].float()
        self.meta    = mm_data.get('meta_pair', torch.zeros(len(self.labels), 5))
        self.pcompat = mm_data.get('p_compatible', torch.zeros(len(self.labels)))
        self.feats   = feats
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return (self.feats[self.idx_a[i]], self.feats[self.idx_b[i]],
                self.labels[i], self.meta[i], self.pcompat[i])

mm_val_ds  = FullMMDataset(hnm_val_mm, hnm_mm_feats)
mm_val_ldr = DataLoader(mm_val_ds, batch_size=4096, shuffle=False, num_workers=0, pin_memory=True)

m_mm = Stage2MM().to(DEVICE)

# Remap keys from stage2_mm_model (proj.net.* -> proj.*)
raw = torch.load(os.path.join(BASE_DIR, "stage2_mm_model.pt"), weights_only=False)
new_state = {}
for k, v in raw['model_state_dict'].items():
    new_key = k.replace('proj.net.', 'proj.')
    new_state[new_key] = v
m_mm.load_state_dict(new_state)
m_mm.eval()

all_p, all_l = [], []
with torch.no_grad():
    for fa, fb, lab, meta, pc in mm_val_ldr:
        fa, fb = fa.to(DEVICE), fb.to(DEVICE)
        meta, pc = meta.to(DEVICE), pc.to(DEVICE)
        all_p.append(m_mm(fa, fb, meta, pc).cpu().numpy()); all_l.append(lab.numpy())
mm_preds = np.concatenate(all_p); mm_labels = np.concatenate(all_l)
auc, prauc, f1 = eval_preds(mm_labels, mm_preds)
log("E7_FullMM_Stage2", auc, prauc, f1, note="text+img+p_compat (2411d) BEST")


# ═══════════════════════════════════════════════════════
# Save results
# ═══════════════════════════════════════════════════════
out_path = os.path.join(BASE_DIR, "ablation_results.csv")
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['model','val_auc','val_prauc','val_f1','note'])
    writer.writeheader()
    writer.writerows(RESULTS)

print(f"\n{'='*65}")
print(f"{'Model':<25} {'AUC':>7} {'PR-AUC':>8} {'F1':>7}")
print(f"{'-'*65}")
for r in RESULTS:
    marker = " <-- BEST" if r['val_auc'] == max(x['val_auc'] for x in RESULTS) else ""
    print(f"  {r['model']:<23} {r['val_auc']:>7.4f} {r['val_prauc']:>8.4f} {r['val_f1']:>7.4f}{marker}")
print(f"{'='*65}")
print(f"\nSaved: {out_path}")
