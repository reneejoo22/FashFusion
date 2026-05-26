"""
FashFusion Full Experiment Suite
=================================
Baselines + Feature Ablations + Architecture Ablations

결과: full_experiment_results.csv

Groups:
  [Baseline]
    B1: Random
    B2: Popularity
    B3: Jaccard

  [Single modality - retrain Stage2]
    A1: TF-IDF only (300d)
    A2: BERT only (768d)
    A3: Struct only (63d)
    A4: Image only (1280d)

  [Multi modality - retrain Stage2]
    A5: Text (tfidf+bert+struct, 1131d)  <- stage2_model.pt
    A6: Text + Image, no p_compat (2411d)
    A7: Full MM: Text+Image+p_compat (2411d)  <- stage2_mm_model.pt

  [Transfer ablation]
    T1: MM + p_compat, no Polyvore proj init (random init)
    T2: Full MM (=A7, best)
"""

import os, csv, time, pickle
from collections import defaultdict
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

RESULTS = []

def log(name, group, auc, prauc, f1, note=""):
    r = dict(model=name, group=group, val_auc=round(auc,4),
             val_prauc=round(prauc,4), val_f1=round(f1,4), note=note)
    RESULTS.append(r)
    print(f"  [{name}]  AUC={auc:.4f}  PR={prauc:.4f}  F1={f1:.4f}  {note}")

def metrics(labels, preds):
    auc   = roc_auc_score(labels, preds)
    prauc = average_precision_score(labels, preds)
    f1    = f1_score(labels, (np.array(preds) >= 0.5).astype(int))
    return auc, prauc, f1

# ── Load shared data ───────────────────────────────────
print("\nLoading shared data ...")
t0 = time.time()

# MM feature matrix (105542 x 2411)
hnm_mm = torch.from_numpy(np.load(os.path.join(BASE_DIR, "hnm_mm_feats.npy"))).float()
D_TFIDF, D_BERT, D_STRUCT, D_IMG = 300, 768, 63, 1280
D_TEXT  = D_TFIDF + D_BERT + D_STRUCT   # 1131
D_ITEM  = D_TEXT + D_IMG                # 2411

# Feature slices (views, no copy)
feat_tfidf  = hnm_mm[:, :300]
feat_bert   = hnm_mm[:, 300:1068]
feat_struct = hnm_mm[:, 1068:1131]
feat_text   = hnm_mm[:, :1131]
feat_img    = hnm_mm[:, 1131:]
feat_mm     = hnm_mm

# Pair datasets (index-based)
val_mm  = torch.load(os.path.join(BASE_DIR, "hnm_pairs_mm_val.pt"),   weights_only=False)
trn_mm  = torch.load(os.path.join(BASE_DIR, "hnm_pairs_mm_train.pt"), weights_only=False)
# Text-only pair data (item_A/B stored directly)
val_txt = torch.load(os.path.join(BASE_DIR, "hnm_pairs_val.pt"),   weights_only=False)
trn_txt = torch.load(os.path.join(BASE_DIR, "hnm_pairs_train.pt"), weights_only=False)

val_labels   = val_mm['labels'].numpy()
val_idx_a    = val_mm['idx_a']
val_idx_b    = val_mm['idx_b']
val_meta     = val_mm.get('meta_pair', torch.zeros(len(val_labels), 5))
val_pcompat  = val_mm.get('p_compatible', torch.zeros(len(val_labels)))

trn_labels   = trn_mm['labels'].numpy()
trn_idx_a    = trn_mm['idx_a']
trn_idx_b    = trn_mm['idx_b']
trn_meta     = trn_mm.get('meta_pair', torch.zeros(len(trn_labels), 5))
trn_pcompat  = trn_mm.get('p_compatible', torch.zeros(len(trn_labels)))

print(f"  Loaded {time.time()-t0:.1f}s  "
      f"Val={len(val_labels):,} Trn={len(trn_labels):,}")


# ── Generic model / dataset / trainer ─────────────────
class PairDS(Dataset):
    """Index-based pair dataset with optional p_compatible."""
    def __init__(self, idx_a, idx_b, labels, meta, feats, pcompat=None):
        self.idx_a  = idx_a; self.idx_b  = idx_b
        self.labels = labels.float()
        self.meta   = meta
        self.feats  = feats
        self.pc     = pcompat if pcompat is not None else torch.zeros(len(labels))
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return (self.feats[self.idx_a[i]], self.feats[self.idx_b[i]],
                self.labels[i], self.meta[i], self.pc[i])


class FusionMLP(nn.Module):
    """Generic Stage2 model for any input dimension."""
    def __init__(self, d_item, use_pcompat=True):
        super().__init__()
        d_proj = 256
        d_pair = d_proj*4 + 1 + 5 + (1 if use_pcompat else 0)
        self.use_pcompat = use_pcompat
        self.proj = nn.Sequential(
            nn.Linear(d_item, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, d_proj), nn.LayerNorm(d_proj), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_pair, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),   nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),     nn.Sigmoid(),
        )
    def forward(self, fa, fb, meta, pc=None):
        ea = self.proj(fa); eb = self.proj(fb)
        cos = torch.cosine_similarity(ea, eb, dim=1, eps=1e-8).unsqueeze(1)
        parts = [ea, eb, (ea-eb).abs(), ea*eb, cos, meta]
        if self.use_pcompat and pc is not None:
            parts.append(pc.unsqueeze(1))
        return self.fusion(torch.cat(parts, dim=1)).squeeze(1)


def train_eval(name, group, d_item, feat_mat, use_pcompat=True,
               init_from_s1=False, epochs=25, lr=1e-4, patience=5, note=""):
    """Train a FusionMLP and evaluate on validation set."""
    print(f"\n[{name}] Training ...")
    pc_trn = trn_pcompat if use_pcompat else None
    pc_val = val_pcompat  if use_pcompat else None

    train_ds = PairDS(trn_idx_a, trn_idx_b, trn_mm['labels'], trn_meta, feat_mat, pc_trn)
    val_ds   = PairDS(val_idx_a,  val_idx_b, val_mm['labels'],  val_meta, feat_mat, pc_val)
    t_ldr = DataLoader(train_ds, batch_size=2048, shuffle=True,  num_workers=0, pin_memory=True)
    v_ldr = DataLoader(val_ds,   batch_size=4096, shuffle=False, num_workers=0, pin_memory=True)

    model = FusionMLP(d_item, use_pcompat).to(DEVICE)

    if init_from_s1 and d_item == D_ITEM:
        # Init proj from stage1_mm_model
        s1 = torch.load(os.path.join(BASE_DIR, "stage1_mm_model.pt"), weights_only=False)
        proj_w = {k.replace('proj.net.', ''): v
                  for k, v in s1['model_state_dict'].items() if k.startswith('proj.net.')}
        model.proj.load_state_dict(proj_w)
        opt = torch.optim.AdamW([
            {'params': model.proj.parameters(),   'lr': lr*0.3},
            {'params': model.fusion.parameters(), 'lr': lr},
        ], weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    crit = nn.BCELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)
    best_auc = 0; no_imp = 0; best_preds = None; best_labs = None

    for ep in range(1, epochs+1):
        model.train(); tl = 0
        for fa, fb, lab, meta, pc in t_ldr:
            fa, fb, lab = fa.to(DEVICE), fb.to(DEVICE), lab.to(DEVICE)
            meta = meta.to(DEVICE)
            pc_d = pc.to(DEVICE) if use_pcompat else None
            opt.zero_grad()
            pred = model(fa, fb, meta, pc_d)
            loss = crit(pred, lab); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item()

        model.eval(); ap, al = [], []
        with torch.no_grad():
            for fa, fb, lab, meta, pc in v_ldr:
                fa, fb = fa.to(DEVICE), fb.to(DEVICE)
                meta = meta.to(DEVICE)
                pc_d = pc.to(DEVICE) if use_pcompat else None
                ap.append(model(fa, fb, meta, pc_d).cpu().numpy()); al.append(lab.numpy())
        ep_preds = np.concatenate(ap); ep_labs = np.concatenate(al)
        ep_auc = roc_auc_score(ep_labs, ep_preds)
        sched.step(ep_auc)
        print(f"    Ep{ep:02d}  TrLoss={tl/len(t_ldr):.4f}  ValAUC={ep_auc:.4f}")
        if ep_auc > best_auc:
            best_auc = ep_auc; no_imp = 0
            best_preds = ep_preds; best_labs = ep_labs
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"    Early stop"); break

    auc, prauc, f1 = metrics(best_labs, best_preds)
    log(name, group, auc, prauc, f1, note)
    return auc


# ═══════════════════════════════════════════════════════
# GROUP 1: Baselines
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("GROUP 1: Baselines")
print("="*60)

print("\n[B1] Random ...")
rng = np.random.RandomState(42)
auc, prauc, f1 = metrics(val_labels, rng.rand(len(val_labels)))
log("B1_Random", "Baseline", auc, prauc, f1)

print("\n[B2] Popularity ...")
pop = np.zeros(len(hnm_mm), dtype=np.float32)
for a, b, lab in zip(trn_idx_a.numpy(), trn_idx_b.numpy(), trn_labels):
    if lab == 1: pop[a] += 1; pop[b] += 1
pop /= (pop.max() + 1e-9)
pop_preds = (pop[val_idx_a.numpy()] + pop[val_idx_b.numpy()]) / 2
auc, prauc, f1 = metrics(val_labels, pop_preds)
log("B2_Popularity", "Baseline", auc, prauc, f1)

print("\n[B3] Jaccard ...")
co = defaultdict(int); ifreq = defaultdict(int)
for a, b, lab in zip(trn_idx_a.numpy(), trn_idx_b.numpy(), trn_labels):
    if lab == 1:
        co[(min(int(a),int(b)), max(int(a),int(b)))] += 1
        ifreq[int(a)] += 1; ifreq[int(b)] += 1
jacc = np.array([co.get((min(int(a),int(b)), max(int(a),int(b))), 0) /
                 max(ifreq[int(a)] + ifreq[int(b)] - co.get((min(int(a),int(b)),
                     max(int(a),int(b))), 0), 1)
                 for a, b in zip(val_idx_a.numpy(), val_idx_b.numpy())], dtype=np.float32)
auc, prauc, f1 = metrics(val_labels, jacc)
log("B3_Jaccard", "Baseline", auc, prauc, f1)

print("\n[B4] Cosine-Text baseline (no training) ...")
cos_preds = torch.cosine_similarity(
    feat_text[val_idx_a], feat_text[val_idx_b], dim=1).numpy()
cos_preds = (cos_preds + 1) / 2   # scale to [0,1]
auc, prauc, f1 = metrics(val_labels, cos_preds)
log("B4_Cosine_Text", "Baseline", auc, prauc, f1)

print("\n[B5] Cosine-Image baseline (no training) ...")
cos_img = torch.cosine_similarity(
    feat_img[val_idx_a], feat_img[val_idx_b], dim=1).numpy()
# Zero-image items give cosine=1 (both zero), filter them
zero_mask = (feat_img[val_idx_a].sum(1) == 0).numpy() | (feat_img[val_idx_b].sum(1) == 0).numpy()
cos_img[zero_mask] = 0.0
cos_img = (cos_img + 1) / 2
auc, prauc, f1 = metrics(val_labels, cos_img)
log("B5_Cosine_Image", "Baseline", auc, prauc, f1)


# ═══════════════════════════════════════════════════════
# GROUP 2: Single-modality ablation
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("GROUP 2: Single-modality ablation")
print("="*60)

train_eval("A1_TFIDF_only",  "SingleModal", D_TFIDF,  feat_tfidf,  use_pcompat=False,
           epochs=20, note="300d")
train_eval("A2_BERT_only",   "SingleModal", D_BERT,   feat_bert,   use_pcompat=False,
           epochs=20, note="768d")
train_eval("A3_Struct_only", "SingleModal", D_STRUCT, feat_struct, use_pcompat=False,
           epochs=20, note="63d")
train_eval("A4_Image_only",  "SingleModal", D_IMG,    feat_img,    use_pcompat=False,
           epochs=20, note="1280d")


# ═══════════════════════════════════════════════════════
# GROUP 3: Multi-modality ablation
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("GROUP 3: Multi-modality ablation")
print("="*60)

# A5: Text-only (load existing stage2_model.pt, eval on mm val set)
print("\n[A5] Text-only Stage2 (existing stage2_model.pt) ...")

class Stage2TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1131, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.LayerNorm(256), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(1030, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),    nn.Sigmoid(),
        )
    def forward(self, fa, fb, meta, pc):
        ea = self.proj(fa); eb = self.proj(fb)
        cos = torch.cosine_similarity(ea, eb, dim=1, eps=1e-8).unsqueeze(1)
        pf  = torch.cat([ea, eb, (ea-eb).abs(), ea*eb, cos, meta, pc.unsqueeze(1)], dim=1)
        return self.fusion(pf).squeeze(1)

m5 = Stage2TextModel().to(DEVICE)
ckpt5 = torch.load(os.path.join(BASE_DIR, "stage2_model.pt"), weights_only=False)
m5.load_state_dict(ckpt5['model_state'])
m5.eval()

# Use text-only pair data for this model
va, vb  = val_txt['item_A'].to(DEVICE), val_txt['item_B'].to(DEVICE)
vm, vpc = val_txt['meta_pair'].to(DEVICE), val_txt['p_compatible'].to(DEVICE)
vlab    = val_txt['labels'].numpy()
BSIZ = 4096; ap = []
with torch.no_grad():
    for i in range(0, len(vlab), BSIZ):
        ap.append(m5(va[i:i+BSIZ], vb[i:i+BSIZ], vm[i:i+BSIZ], vpc[i:i+BSIZ]).cpu().numpy())
auc, prauc, f1 = metrics(vlab, np.concatenate(ap))
log("A5_Text_Stage2", "MultiModal", auc, prauc, f1,
    note="tfidf+bert+struct+p_compat (1131d)")

# A6: Text+Image, no p_compat
train_eval("A6_TextImg_noPc", "MultiModal", D_ITEM, feat_mm, use_pcompat=False,
           init_from_s1=True, epochs=30, note="2411d, no p_compat")

# A7: Full MM (load existing stage2_mm_model.pt)
print("\n[A7] Full MM Stage2 (existing stage2_mm_model.pt) ...")

class Stage2MMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2411, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.LayerNorm(256), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(1030, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),    nn.Sigmoid(),
        )
    def forward(self, fa, fb, meta, pc):
        ea = self.proj(fa); eb = self.proj(fb)
        cos = torch.cosine_similarity(ea, eb, dim=1, eps=1e-8).unsqueeze(1)
        pf  = torch.cat([ea, eb, (ea-eb).abs(), ea*eb, cos, meta, pc.unsqueeze(1)], dim=1)
        return self.fusion(pf).squeeze(1)

m7 = Stage2MMModel().to(DEVICE)
ckpt7 = torch.load(os.path.join(BASE_DIR, "stage2_mm_model.pt"), weights_only=False)
new_state = {k.replace('proj.net.', 'proj.'): v
             for k, v in ckpt7['model_state_dict'].items()}
m7.load_state_dict(new_state)
m7.eval()

val_mm_ds  = PairDS(val_idx_a, val_idx_b, val_mm['labels'], val_meta, feat_mm, val_pcompat)
val_mm_ldr = DataLoader(val_mm_ds, batch_size=4096, shuffle=False, num_workers=0, pin_memory=True)
ap = []
with torch.no_grad():
    for fa, fb, lab, meta, pc in val_mm_ldr:
        fa, fb = fa.to(DEVICE), fb.to(DEVICE)
        meta, pc = meta.to(DEVICE), pc.to(DEVICE)
        ap.append(m7(fa, fb, meta, pc).cpu().numpy())
auc, prauc, f1 = metrics(val_labels, np.concatenate(ap))
log("A7_FullMM_Stage2", "MultiModal", auc, prauc, f1,
    note="tfidf+bert+struct+img+p_compat (2411d) BEST")


# ═══════════════════════════════════════════════════════
# GROUP 4: Transfer ablation
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("GROUP 4: Transfer / Architecture ablation")
print("="*60)

# T1: MM + p_compat, NO Polyvore proj init (random init)
train_eval("T1_MM_NoTransfer", "Transfer", D_ITEM, feat_mm, use_pcompat=True,
           init_from_s1=False, epochs=30,
           note="MM 2411d+p_compat, no Polyvore proj init")

# T2 = A7 (already logged)
print("\n[T2] = A7_FullMM_Stage2 (with Polyvore transfer) - already evaluated")


# ═══════════════════════════════════════════════════════
# Save & Print Summary
# ═══════════════════════════════════════════════════════
out_path = os.path.join(BASE_DIR, "full_experiment_results.csv")
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['model','group','val_auc','val_prauc','val_f1','note'])
    writer.writeheader(); writer.writerows(RESULTS)

print(f"\n{'='*70}")
print(f"{'Model':<26} {'Group':<12} {'AUC':>7} {'PR-AUC':>8} {'F1':>7}")
print(f"{'-'*70}")
best_auc = max(r['val_auc'] for r in RESULTS)
for r in RESULTS:
    mark = " <-BEST" if r['val_auc'] == best_auc else ""
    print(f"  {r['model']:<24} {r['group']:<12} "
          f"{r['val_auc']:>7.4f} {r['val_prauc']:>8.4f} {r['val_f1']:>7.4f}{mark}")
print(f"{'='*70}")
print(f"\nSaved: {out_path}")
