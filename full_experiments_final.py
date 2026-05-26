"""
FashFusion Full Experiment Suite
=================================
Groups 1~4 통합 실험 코드

[Group 1] Baselines
  B1: Random
  B2: Popularity
  B3: Jaccard
  B4: Cosine-Text (no training)
  B5: Cosine-Image (no training)

[Group 2] Single-modality ablation
  A1: TF-IDF only     (300d)
  A2: BERT only       (768d)
  A3: Struct only     (63d)
  A4: Image only      (1280d)

[Group 3] Multi-modality ablation
  A5: Text only + p_compat     <- stage2_model.pt (1131d)
  A6: Text + Image, no p_compat (2411d, Polyvore proj init)
  A7: Full MM + p_compat       <- stage2_mm_model.pt (2411d)

[Group 4] Transfer ablation
  T1: Full MM + p_compat, random proj init
  T2: Full MM + p_compat, Polyvore proj init  (= A7)

Output: full_experiment_results.csv
"""

import os, csv, time
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

# ── 유틸 ───────────────────────────────────────────────
def log(name, group, auc, prauc, f1, note=""):
    r = dict(model=name, group=group,
             val_auc=round(auc, 4), val_prauc=round(prauc, 4), val_f1=round(f1, 4),
             note=note)
    RESULTS.append(r)
    print(f"  [{name}]  AUC={auc:.4f}  PR={prauc:.4f}  F1={f1:.4f}  {note}")

def metrics(labels, preds):
    auc   = roc_auc_score(labels, preds)
    prauc = average_precision_score(labels, preds)
    f1    = f1_score(labels, (np.array(preds) >= 0.5).astype(int))
    return auc, prauc, f1


# ── 데이터 로드 ────────────────────────────────────────
print("\nLoading shared data ...")
t0 = time.time()

# MM feature matrix  (105542 x 2411)
hnm_mm = torch.from_numpy(
    np.load(os.path.join(BASE_DIR, "hnm_mm_feats.npy"))
).float()

D_TFIDF, D_BERT, D_STRUCT, D_IMG = 300, 768, 63, 1280
D_TEXT = D_TFIDF + D_BERT + D_STRUCT   # 1131
D_ITEM = D_TEXT + D_IMG                # 2411
D_PROJ = 256
D_META = 5

# Feature slice views (no copy)
feat_tfidf  = hnm_mm[:, :300]
feat_bert   = hnm_mm[:, 300:1068]
feat_struct = hnm_mm[:, 1068:1131]
feat_text   = hnm_mm[:, :1131]
feat_img    = hnm_mm[:, 1131:]
feat_mm     = hnm_mm

# Index-based MM pair files
val_mm  = torch.load(os.path.join(BASE_DIR, "hnm_pairs_mm_val.pt"),   weights_only=False)
trn_mm  = torch.load(os.path.join(BASE_DIR, "hnm_pairs_mm_train.pt"), weights_only=False)

# Text-only pair files (item features stored directly, for A5)
val_txt = torch.load(os.path.join(BASE_DIR, "hnm_pairs_val.pt"),   weights_only=False)

val_labels  = val_mm['labels'].numpy()
val_idx_a   = val_mm['idx_a']
val_idx_b   = val_mm['idx_b']
val_meta    = val_mm.get('meta_pair',    torch.zeros(len(val_labels), D_META))
val_pcompat = val_mm.get('p_compatible', torch.zeros(len(val_labels)))

trn_labels  = trn_mm['labels'].numpy()
trn_idx_a   = trn_mm['idx_a']
trn_idx_b   = trn_mm['idx_b']
trn_meta    = trn_mm.get('meta_pair',    torch.zeros(len(trn_labels), D_META))
trn_pcompat = trn_mm.get('p_compatible', torch.zeros(len(trn_labels)))

print(f"  Loaded {time.time()-t0:.1f}s  "
      f"Val={len(val_labels):,}  Trn={len(trn_labels):,}")


# ── Dataset ────────────────────────────────────────────
class PairDS(Dataset):
    """Index-based pair dataset."""
    def __init__(self, idx_a, idx_b, labels, meta, feats, pcompat=None):
        self.idx_a  = idx_a;  self.idx_b  = idx_b
        self.labels = labels.float()
        self.meta   = meta
        self.feats  = feats
        self.pc     = pcompat if pcompat is not None else torch.zeros(len(labels))
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return (self.feats[self.idx_a[i]], self.feats[self.idx_b[i]],
                self.labels[i], self.meta[i], self.pc[i])


# ── 공통 모델 (Groups 2~4 훈련용) ──────────────────────
class FusionMLP(nn.Module):
    """임의 입력 차원에 대한 Stage2 모델."""
    def __init__(self, d_item, use_pcompat=True):
        super().__init__()
        d_pair = D_PROJ*4 + 1 + D_META + (1 if use_pcompat else 0)
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
        parts = [ea, eb, (ea - eb).abs(), ea * eb, cos, meta]
        if self.use_pcompat and pc is not None:
            parts.append(pc.unsqueeze(1))
        return self.fusion(torch.cat(parts, dim=1)).squeeze(1)


# ── 공통 훈련 루프 ─────────────────────────────────────
def train_eval(name, group, d_item, feat_mat,
               use_pcompat=True, init_from_s1=False,
               epochs=30, lr=1e-4, patience=5, note="", save_path=None):
    """FusionMLP를 훈련하고 val 성능을 반환."""
    print(f"\n[{name}] Training ...")
    pc_trn = trn_pcompat if use_pcompat else None
    pc_val = val_pcompat  if use_pcompat else None

    t_ds = PairDS(trn_idx_a, trn_idx_b, trn_mm['labels'], trn_meta, feat_mat, pc_trn)
    v_ds = PairDS(val_idx_a,  val_idx_b, val_mm['labels'], val_meta, feat_mat, pc_val)
    t_ldr = DataLoader(t_ds, batch_size=2048, shuffle=True,  num_workers=0, pin_memory=True)
    v_ldr = DataLoader(v_ds, batch_size=4096, shuffle=False, num_workers=0, pin_memory=True)

    model = FusionMLP(d_item, use_pcompat).to(DEVICE)

    # Stage1 MM proj 가중치 초기화 (A6, T2용)
    if init_from_s1 and d_item == D_ITEM:
        s1 = torch.load(os.path.join(BASE_DIR, "stage1_mm_model.pt"), weights_only=False)
        proj_w = {k.replace('proj.net.', ''): v
                  for k, v in s1['model_state_dict'].items()
                  if k.startswith('proj.net.')}
        try:
            model.proj.load_state_dict(proj_w)
            print(f"    Loaded Stage1 proj weights (epoch={s1['epoch']}, AUC={s1['val_auc']:.4f})")
        except Exception as e:
            print(f"    Proj init failed ({e}), using random init")
        opt = torch.optim.AdamW([
            {'params': model.proj.parameters(),   'lr': lr * 0.3},
            {'params': model.fusion.parameters(), 'lr': lr},
        ], weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    crit  = nn.BCELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)
    best_auc = 0; no_imp = 0; best_preds = best_labs = None

    for ep in range(1, epochs + 1):
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
                fa, fb = fa.to(DEVICE), fb.to(DEVICE)
                meta = meta.to(DEVICE)
                pc_d = pc.to(DEVICE) if use_pcompat else None
                ap.append(model(fa, fb, meta, pc_d).cpu().numpy())
                al.append(lab.numpy())

        ep_preds = np.concatenate(ap); ep_labs = np.concatenate(al)
        ep_auc = roc_auc_score(ep_labs, ep_preds)
        sched.step(ep_auc)
        print(f"    Ep{ep:02d}  TrLoss={tl/len(t_ldr):.4f}  ValAUC={ep_auc:.4f}")

        if ep_auc > best_auc:
            best_auc = ep_auc; no_imp = 0
            best_preds = ep_preds; best_labs = ep_labs
            if save_path:
                torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                            'val_auc': ep_auc, 'D_ITEM': d_item, 'D_PROJ': D_PROJ,
                            'use_pcompat': use_pcompat}, save_path)
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"    Early stop at epoch {ep}"); break

    auc, prauc, f1 = metrics(best_labs, best_preds)
    log(name, group, auc, prauc, f1, note)
    return auc


# ═══════════════════════════════════════════════════════
# GROUP 1: Baselines
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("GROUP 1: Baselines")
print("="*60)

# B1: Random
print("\n[B1] Random ...")
rng = np.random.RandomState(42)
auc, prauc, f1 = metrics(val_labels, rng.rand(len(val_labels)))
log("B1_Random", "Baseline", auc, prauc, f1)

# B2: Popularity
print("\n[B2] Popularity ...")
pop = np.zeros(len(hnm_mm), dtype=np.float32)
for a, b, lab in zip(trn_idx_a.numpy(), trn_idx_b.numpy(), trn_labels):
    if lab == 1:
        pop[a] += 1; pop[b] += 1
pop /= (pop.max() + 1e-9)
pop_preds = (pop[val_idx_a.numpy()] + pop[val_idx_b.numpy()]) / 2
auc, prauc, f1 = metrics(val_labels, pop_preds)
log("B2_Popularity", "Baseline", auc, prauc, f1)

# B3: Jaccard
print("\n[B3] Jaccard ...")
co = defaultdict(int); ifreq = defaultdict(int)
for a, b, lab in zip(trn_idx_a.numpy(), trn_idx_b.numpy(), trn_labels):
    if lab == 1:
        co[(min(int(a), int(b)), max(int(a), int(b)))] += 1
        ifreq[int(a)] += 1; ifreq[int(b)] += 1
jacc = np.array([
    co.get((min(int(a), int(b)), max(int(a), int(b))), 0) /
    max(ifreq[int(a)] + ifreq[int(b)]
        - co.get((min(int(a), int(b)), max(int(a), int(b))), 0), 1)
    for a, b in zip(val_idx_a.numpy(), val_idx_b.numpy())
], dtype=np.float32)
auc, prauc, f1 = metrics(val_labels, jacc)
log("B3_Jaccard", "Baseline", auc, prauc, f1)

# B4: Cosine-Text
print("\n[B4] Cosine-Text baseline (no training) ...")
cos_text = torch.cosine_similarity(
    feat_text[val_idx_a], feat_text[val_idx_b], dim=1).numpy()
cos_text = (cos_text + 1) / 2
auc, prauc, f1 = metrics(val_labels, cos_text)
log("B4_Cosine_Text", "Baseline", auc, prauc, f1)

# B5: Cosine-Image
print("\n[B5] Cosine-Image baseline (no training) ...")
cos_img = torch.cosine_similarity(
    feat_img[val_idx_a], feat_img[val_idx_b], dim=1).numpy()
zero_mask = ((feat_img[val_idx_a].sum(1) == 0) |
             (feat_img[val_idx_b].sum(1) == 0)).numpy()
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

train_eval("A1_TFIDF_only",  "SingleModal", D_TFIDF,  feat_tfidf,
           use_pcompat=False, epochs=20, note="300d")
train_eval("A2_BERT_only",   "SingleModal", D_BERT,   feat_bert,
           use_pcompat=False, epochs=20, note="768d")
train_eval("A3_Struct_only", "SingleModal", D_STRUCT, feat_struct,
           use_pcompat=False, epochs=20, note="63d")
train_eval("A4_Image_only",  "SingleModal", D_IMG,    feat_img,
           use_pcompat=False, epochs=20, note="1280d")


# ═══════════════════════════════════════════════════════
# GROUP 3: Multi-modality ablation
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("GROUP 3: Multi-modality ablation")
print("="*60)

# A5: Text-only (기존 stage2_model.pt 로드, proj.net.* 키 구조)
print("\n[A5] Text-only Stage2 (stage2_model.pt) ...")

class Stage2TextModel(nn.Module):
    """stage2_model.pt 아키텍처와 정확히 일치.
    입력: 1131d text / pair: 256*4+1+5+1 = 1031d
    주의: step8_stage2_train.py(deprecated)의 ProjectionMLP는 LayerNorm 뒤 ReLU 없음."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Module()
        self.proj.net = nn.Sequential(
            nn.Linear(1131, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.LayerNorm(256),  # ReLU 없음: step8 원본과 일치
        )
        self.fusion = nn.Sequential(
            nn.Linear(1031, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),    nn.Sigmoid(),
        )
    def forward(self, fa, fb, meta, pc):
        ea = self.proj.net(fa); eb = self.proj.net(fb)
        cos = torch.cosine_similarity(ea, eb, dim=1, eps=1e-8).unsqueeze(1)
        pf  = torch.cat([ea, eb, (ea - eb).abs(), ea * eb, cos, meta, pc.unsqueeze(1)], dim=1)
        return self.fusion(pf).squeeze(1)

m5 = Stage2TextModel().to(DEVICE)
ckpt5 = torch.load(os.path.join(BASE_DIR, "stage2_model.pt"), weights_only=False)
m5.load_state_dict(ckpt5['model_state'])
m5.eval()

va  = val_txt['item_A'].to(DEVICE);    vb  = val_txt['item_B'].to(DEVICE)
vm  = val_txt['meta_pair'].to(DEVICE); vpc = val_txt['p_compatible'].to(DEVICE)
vlab = val_txt['labels'].numpy()
ap = []
with torch.no_grad():
    for i in range(0, len(vlab), 4096):
        ap.append(m5(va[i:i+4096], vb[i:i+4096],
                     vm[i:i+4096], vpc[i:i+4096]).cpu().numpy())
auc, prauc, f1 = metrics(vlab, np.concatenate(ap))
log("A5_Text_Stage2", "MultiModal", auc, prauc, f1,
    note="tfidf+bert+struct+p_compat (1131d)")

# A6: Text+Image, p_compat 없음 (Stage1 proj 초기화)
train_eval("A6_TextImg_noPc", "MultiModal", D_ITEM, feat_mm,
           use_pcompat=False, init_from_s1=True, epochs=30,
           note="2411d, no p_compat")

# A7: Full MM (기존 stage2_mm_model.pt 로드, proj.net.* 키 구조)
print("\n[A7] Full MM Stage2 (stage2_mm_model.pt) ...")

class Stage2MMModel(nn.Module):
    """stage2_mm_model.pt 아키텍처와 정확히 일치.
    입력: 2411d MM / pair: 256*4+1+5+1 = 1031d"""
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
    def forward(self, fa, fb, meta, pc):
        ea = self.proj.net(fa); eb = self.proj.net(fb)
        cos = torch.cosine_similarity(ea, eb, dim=1, eps=1e-8).unsqueeze(1)
        pf  = torch.cat([ea, eb, (ea - eb).abs(), ea * eb, cos, meta, pc.unsqueeze(1)], dim=1)
        return self.fusion(pf).squeeze(1)

m7 = Stage2MMModel().to(DEVICE)
ckpt7 = torch.load(os.path.join(BASE_DIR, "stage2_mm_model.pt"), weights_only=False)
m7.load_state_dict(ckpt7['model_state_dict'])
m7.eval()

v_ds7  = PairDS(val_idx_a, val_idx_b, val_mm['labels'], val_meta, feat_mm, val_pcompat)
v_ldr7 = DataLoader(v_ds7, batch_size=4096, shuffle=False, num_workers=0, pin_memory=True)
ap = []
with torch.no_grad():
    for fa, fb, lab, meta, pc in v_ldr7:
        fa, fb = fa.to(DEVICE), fb.to(DEVICE)
        meta, pc = meta.to(DEVICE), pc.to(DEVICE)
        ap.append(m7(fa, fb, meta, pc).cpu().numpy())
auc, prauc, f1 = metrics(val_labels, np.concatenate(ap))
log("A7_FullMM_Stage2", "MultiModal", auc, prauc, f1,
    note="tfidf+bert+struct+img+p_compat (2411d)")


# ═══════════════════════════════════════════════════════
# GROUP 4: Transfer ablation
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("GROUP 4: Transfer ablation")
print("="*60)

# T1: Full MM + p_compat, proj 랜덤 초기화
t1_save = os.path.join(BASE_DIR, "stage2_t1_model.pt")
train_eval("T1_MM_NoTransfer", "Transfer", D_ITEM, feat_mm,
           use_pcompat=True, init_from_s1=False, epochs=30, patience=5,
           note="MM 2411d+p_compat, random proj init", save_path=t1_save)
print(f"    T1 checkpoint saved: {t1_save}")

# T2: Full MM + p_compat, Polyvore proj 초기화 (= A7)
a7_res = next(r for r in RESULTS if r['model'] == 'A7_FullMM_Stage2')
log("T2_MM_WithTransfer", "Transfer",
    a7_res['val_auc'], a7_res['val_prauc'], a7_res['val_f1'],
    note="=A7, MM 2411d+p_compat, Polyvore proj init")


# ═══════════════════════════════════════════════════════
# 결과 저장 & 출력
# ═══════════════════════════════════════════════════════
out_path = os.path.join(BASE_DIR, "full_experiment_results.csv")
with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=['model', 'group', 'val_auc', 'val_prauc', 'val_f1', 'note'])
    writer.writeheader()
    writer.writerows(RESULTS)

print(f"\n{'='*72}")
print(f"{'Model':<26} {'Group':<12} {'AUC':>7} {'PR-AUC':>8} {'F1':>7}")
print(f"{'-'*72}")
best_auc = max(r['val_auc'] for r in RESULTS)
for group in ['Baseline', 'SingleModal', 'MultiModal', 'Transfer']:
    grp = [r for r in RESULTS if r['group'] == group]
    if not grp: continue
    print(f"  --- {group} ---")
    for r in grp:
        mark = " <-BEST" if r['val_auc'] == best_auc else ""
        print(f"  {r['model']:<24} {r['group']:<12} "
              f"{r['val_auc']:>7.4f} {r['val_prauc']:>8.4f} {r['val_f1']:>7.4f}{mark}")
print(f"{'='*72}")
print(f"\nSaved: {out_path}")
print("EXPERIMENTS_COMPLETE")
