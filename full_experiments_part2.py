"""
Full Experiments Part 2: Groups 3 & 4 + Final Summary
Groups 1 & 2 결과를 하드코딩하고 이어서 실행
"""

import os, csv, time, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Pre-filled results from Part 1 ─────────────────────
RESULTS = [
    {'model':'B1_Random',      'group':'Baseline',    'val_auc':0.5033,'val_prauc':0.4041,'val_f1':0.4480,'note':''},
    {'model':'B2_Popularity',  'group':'Baseline',    'val_auc':0.9250,'val_prauc':0.8816,'val_f1':0.0081,'note':''},
    {'model':'B3_Jaccard',     'group':'Baseline',    'val_auc':0.5303,'val_prauc':0.4378,'val_f1':0.0000,'note':''},
    {'model':'B4_Cosine_Text', 'group':'Baseline',    'val_auc':0.7021,'val_prauc':0.6542,'val_f1':0.5730,'note':'no training'},
    {'model':'B5_Cosine_Image','group':'Baseline',    'val_auc':0.6494,'val_prauc':0.5853,'val_f1':0.5744,'note':'no training'},
    {'model':'A1_TFIDF_only',  'group':'SingleModal', 'val_auc':0.9664,'val_prauc':0.9497,'val_f1':0.8737,'note':'300d'},
    {'model':'A2_BERT_only',   'group':'SingleModal', 'val_auc':0.9485,'val_prauc':0.9275,'val_f1':0.8271,'note':'768d'},
    {'model':'A3_Struct_only', 'group':'SingleModal', 'val_auc':0.7355,'val_prauc':0.6680,'val_f1':0.5440,'note':'63d'},
    {'model':'A4_Image_only',  'group':'SingleModal', 'val_auc':0.9676,'val_prauc':0.9510,'val_f1':0.8795,'note':'1280d'},
]

def log(name, group, auc, prauc, f1, note=""):
    r = dict(model=name, group=group, val_auc=round(auc,4),
             val_prauc=round(prauc,4), val_f1=round(f1,4), note=note)
    RESULTS.append(r)
    print(f"  [{name}]  AUC={auc:.4f}  PR={prauc:.4f}  F1={f1:.4f}  {note}")

def metrics(labels, preds):
    return (roc_auc_score(labels, preds),
            average_precision_score(labels, preds),
            f1_score(labels, (np.array(preds) >= 0.5).astype(int)))

# ── Load data ──────────────────────────────────────────
print("\nLoading data ...")
hnm_mm = torch.from_numpy(np.load(os.path.join(BASE_DIR, "hnm_mm_feats.npy"))).float()
D_TEXT = 1131; D_IMG = 1280; D_ITEM = 2411
feat_text = hnm_mm[:, :D_TEXT]
feat_img  = hnm_mm[:, D_TEXT:]
feat_mm   = hnm_mm

val_mm  = torch.load(os.path.join(BASE_DIR, "hnm_pairs_mm_val.pt"),   weights_only=False)
trn_mm  = torch.load(os.path.join(BASE_DIR, "hnm_pairs_mm_train.pt"), weights_only=False)
val_txt = torch.load(os.path.join(BASE_DIR, "hnm_pairs_val.pt"),      weights_only=False)

val_labels  = val_mm['labels'].numpy()
val_idx_a   = val_mm['idx_a']
val_idx_b   = val_mm['idx_b']
val_meta    = val_mm.get('meta_pair', torch.zeros(len(val_labels), 5))
val_pcompat = val_mm.get('p_compatible', torch.zeros(len(val_labels)))
trn_idx_a   = trn_mm['idx_a']
trn_idx_b   = trn_mm['idx_b']
trn_meta    = trn_mm.get('meta_pair', torch.zeros(len(trn_mm['labels']), 5))
trn_pcompat = trn_mm.get('p_compatible', torch.zeros(len(trn_mm['labels'])))
print(f"  Val={len(val_labels):,}  Trn={len(trn_mm['labels']):,}")


# ── Dataset + Model ────────────────────────────────────
class PairDS(Dataset):
    def __init__(self, idx_a, idx_b, labels, meta, feats, pcompat=None):
        self.idx_a = idx_a; self.idx_b = idx_b
        self.labels = labels.float(); self.meta = meta
        self.feats = feats
        self.pc = pcompat if pcompat is not None else torch.zeros(len(labels))
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return (self.feats[self.idx_a[i]], self.feats[self.idx_b[i]],
                self.labels[i], self.meta[i], self.pc[i])


class FusionMLP(nn.Module):
    def __init__(self, d_item, use_pcompat=True):
        super().__init__()
        D_PROJ = 256
        d_pair = D_PROJ*4 + 1 + 5 + (1 if use_pcompat else 0)
        self.use_pcompat = use_pcompat
        self.proj = nn.Sequential(
            nn.Linear(d_item, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, D_PROJ), nn.LayerNorm(D_PROJ), nn.ReLU(),
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
               init_from_s1=False, epochs=30, lr=1e-4, patience=5, note=""):
    print(f"\n[{name}] Training ...")
    pc_trn = trn_pcompat if use_pcompat else None
    pc_val = val_pcompat  if use_pcompat else None
    t_ds = PairDS(trn_idx_a, trn_idx_b, trn_mm['labels'], trn_meta, feat_mat, pc_trn)
    v_ds = PairDS(val_idx_a,  val_idx_b, val_mm['labels'], val_meta, feat_mat, pc_val)
    t_ldr = DataLoader(t_ds, batch_size=2048, shuffle=True,  num_workers=0, pin_memory=True)
    v_ldr = DataLoader(v_ds, batch_size=4096, shuffle=False, num_workers=0, pin_memory=True)

    model = FusionMLP(d_item, use_pcompat).to(DEVICE)

    if init_from_s1 and d_item == D_ITEM:
        s1 = torch.load(os.path.join(BASE_DIR, "stage1_mm_model.pt"), weights_only=False)
        proj_w = {}
        for k, v in s1['model_state_dict'].items():
            if k.startswith('proj.net.'):
                proj_w[k.replace('proj.net.', '')] = v
            elif k.startswith('proj.') and not k.startswith('proj.net.'):
                proj_w[k.replace('proj.', '')] = v
        try:
            model.proj.load_state_dict(proj_w)
            print(f"    Loaded Stage1 proj weights")
        except Exception as e:
            print(f"    Proj init failed ({e}), using random init")
        opt = torch.optim.AdamW([
            {'params': model.proj.parameters(),   'lr': lr*0.3},
            {'params': model.fusion.parameters(), 'lr': lr},
        ], weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    crit  = nn.BCELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)
    best_auc = 0; no_imp = 0; best_preds = best_labs = None

    for ep in range(1, epochs+1):
        model.train(); tl = 0
        for fa, fb, lab, meta, pc in t_ldr:
            fa, fb, lab = fa.to(DEVICE), fb.to(DEVICE), lab.to(DEVICE)
            meta = meta.to(DEVICE)
            pc_d = pc.to(DEVICE) if use_pcompat else None
            opt.zero_grad()
            loss = crit(model(fa, fb, meta, pc_d), lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item()

        model.eval(); ap, al = [], []
        with torch.no_grad():
            for fa, fb, lab, meta, pc in v_ldr:
                fa, fb = fa.to(DEVICE), fb.to(DEVICE); meta = meta.to(DEVICE)
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
            if no_imp >= patience: print(f"    Early stop"); break

    auc, prauc, f1 = metrics(best_labs, best_preds)
    log(name, group, auc, prauc, f1, note)
    return auc


# ═══════════════════════════════════════════════════════
# GROUP 3: Multi-modality
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("GROUP 3: Multi-modality ablation")
print("="*60)

# A5: Text-only (existing stage2_model.pt, proj.net.* keys, fusion dim=1031)
print("\n[A5] Text-only Stage2 (stage2_model.pt) ...")

class Stage2Text(nn.Module):
    """Matches the exact architecture of stage2_model.pt."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Module()
        self.proj.net = nn.Sequential(
            nn.Linear(1131, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.LayerNorm(256), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(1031, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),    nn.Sigmoid(),
        )
    def _proj(self, x): return self.proj.net(x)
    def forward(self, fa, fb, meta, pc):
        ea = self._proj(fa); eb = self._proj(fb)
        cos = torch.cosine_similarity(ea, eb, dim=1, eps=1e-8).unsqueeze(1)
        pf  = torch.cat([ea, eb, (ea-eb).abs(), ea*eb, cos, meta, pc.unsqueeze(1)], dim=1)
        return self.fusion(pf).squeeze(1)

m5 = Stage2Text().to(DEVICE)
ckpt5 = torch.load(os.path.join(BASE_DIR, "stage2_model.pt"), weights_only=False)
m5.load_state_dict(ckpt5['model_state'])
m5.eval()

va  = val_txt['item_A'].to(DEVICE); vb  = val_txt['item_B'].to(DEVICE)
vm  = val_txt['meta_pair'].to(DEVICE); vpc = val_txt['p_compatible'].to(DEVICE)
vlab = val_txt['labels'].numpy()
ap = []
with torch.no_grad():
    for i in range(0, len(vlab), 4096):
        ap.append(m5(va[i:i+4096], vb[i:i+4096], vm[i:i+4096], vpc[i:i+4096]).cpu().numpy())
auc, prauc, f1 = metrics(vlab, np.concatenate(ap))
log("A5_Text_Stage2", "MultiModal", auc, prauc, f1, note="tfidf+bert+struct+p_compat (1131d)")

# A6: Text+Image, no p_compat
train_eval("A6_TextImg_noPc", "MultiModal", D_ITEM, feat_mm, use_pcompat=False,
           init_from_s1=True, epochs=30, note="2411d, no cross-domain p_compat")

# A7: Full MM (existing stage2_mm_model.pt)
print("\n[A7] Full MM Stage2 (stage2_mm_model.pt) ...")

class Stage2MM(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Module()
        self.proj.net = nn.Sequential(
            nn.Linear(2411, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.LayerNorm(256), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(1031, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),    nn.Sigmoid(),
        )
    def _proj(self, x): return self.proj.net(x)
    def forward(self, fa, fb, meta, pc):
        ea = self._proj(fa); eb = self._proj(fb)
        cos = torch.cosine_similarity(ea, eb, dim=1, eps=1e-8).unsqueeze(1)
        pf  = torch.cat([ea, eb, (ea-eb).abs(), ea*eb, cos, meta, pc.unsqueeze(1)], dim=1)
        return self.fusion(pf).squeeze(1)

m7 = Stage2MM().to(DEVICE)
ckpt7 = torch.load(os.path.join(BASE_DIR, "stage2_mm_model.pt"), weights_only=False)
m7.load_state_dict(ckpt7['model_state_dict'])
m7.eval()

v_ds7  = PairDS(val_idx_a, val_idx_b, val_mm['labels'], val_meta, feat_mm, val_pcompat)
v_ldr7 = DataLoader(v_ds7, batch_size=4096, shuffle=False, num_workers=0, pin_memory=True)
ap = []
with torch.no_grad():
    for fa, fb, lab, meta, pc in v_ldr7:
        fa, fb = fa.to(DEVICE), fb.to(DEVICE); meta, pc = meta.to(DEVICE), pc.to(DEVICE)
        ap.append(m7(fa, fb, meta, pc).cpu().numpy())
auc, prauc, f1 = metrics(val_labels, np.concatenate(ap))
log("A7_FullMM_Stage2", "MultiModal", auc, prauc, f1,
    note="text+img+p_compat (2411d) BEST")


# ═══════════════════════════════════════════════════════
# GROUP 4: Transfer ablation
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("GROUP 4: Transfer ablation")
print("="*60)

# T1: MM + p_compat, NO Polyvore proj init
train_eval("T1_MM_NoTransfer", "Transfer", D_ITEM, feat_mm, use_pcompat=True,
           init_from_s1=False, epochs=30, patience=5,
           note="MM 2411d+p_compat, random proj init")

# T2 = A7 (already done)
auc7 = next(r['val_auc'] for r in RESULTS if r['model'] == 'A7_FullMM_Stage2')
prauc7 = next(r['val_prauc'] for r in RESULTS if r['model'] == 'A7_FullMM_Stage2')
f17 = next(r['val_f1'] for r in RESULTS if r['model'] == 'A7_FullMM_Stage2')
log("T2_MM_WithTransfer", "Transfer", auc7, prauc7, f17,
    note="=A7, MM + Polyvore proj init (BEST)")


# ═══════════════════════════════════════════════════════
# Save & Print Final Summary
# ═══════════════════════════════════════════════════════
out_path = os.path.join(BASE_DIR, "full_experiment_results.csv")
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['model','group','val_auc','val_prauc','val_f1','note'])
    w.writeheader(); w.writerows(RESULTS)

print(f"\n{'='*72}")
print(f"{'Model':<26} {'Group':<12} {'AUC':>7} {'PR-AUC':>8} {'F1':>7}")
print(f"{'-'*72}")
best_auc = max(r['val_auc'] for r in RESULTS)
groups = ['Baseline','SingleModal','MultiModal','Transfer']
for g in groups:
    grp = [r for r in RESULTS if r['group'] == g]
    if grp:
        print(f"  --- {g} ---")
        for r in grp:
            mk = " <-BEST" if r['val_auc'] == best_auc else ""
            print(f"  {r['model']:<24} {r['group']:<12} "
                  f"{r['val_auc']:>7.4f} {r['val_prauc']:>8.4f} {r['val_f1']:>7.4f}{mk}")
print(f"{'='*72}")
print(f"\nSaved: {out_path}")
print("EXPERIMENTS_COMPLETE")
