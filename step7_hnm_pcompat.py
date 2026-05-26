"""
Step 7: Apply Stage 1 model to H&M item pairs -> p_compatible scores

Strategy:
  Stage 1 model input dim = 1131 (tfidf_300 + bert_768 + struct_63)
  H&M struct is 105d (different layout).
  Solution: build H&M-compat struct (63d) matching Polyvore layout:
    [0]    : 0.0          (no price in H&M articles.csv)
    [1]    : 0.0          (no popularity in H&M articles.csv)
    [2:12] : TPO 10d      (text-derived, same as Polyvore)
    [12:63]: garment_group top-50 onehot (51d, closest to Polyvore categoryid)
  -> X_hnm_compat = [hnm_tfidf_300 | hnm_bert_768 | hnm_struct_compat63] = 1131d

Pair generation from transactions_train.csv:
  - Co-purchased items (same basket / same day customer) -> positive
  - Random cross-basket pairs                            -> negative
  - Sample up to MAX_PAIRS pairs for Stage 2 training

Outputs:
  hnm_struct_compat63.npy   H&M items in Polyvore-compatible 63d struct
  hnm_pairs_stage2.pt       {item_A, item_B, meta_pair, p_compatible, label}
  hnm_p_compat.npy          p_compatible score per sampled pair (N,)
"""

import os, pickle, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "hnm")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

SEED       = 42
MAX_PAIRS  = 500_000   # total H&M pairs for Stage 2
NEG_RATIO  = 1.5
BATCH_INFER = 2048

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────
# 1. Build H&M compat-63d struct
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("Building H&M compat-63d struct")
print("=" * 60)

# Load H&M struct (105d) for TPO slice
hnm_struct = np.load(os.path.join(BASE_DIR, "hnm_struct.npy"))  # (105542, 105)
# Layout: [0:10]=TPO, [10:30]=colour_onehot(20), [30:60]=garment_onehot(30), ...

N_HNM = hnm_struct.shape[0]
print(f"  hnm_struct: {hnm_struct.shape}")

# Load H&M encoders to access garment_group top cats
with open(os.path.join(BASE_DIR, "hnm_struct_encoders.pkl"), "rb") as f:
    hnm_enc_info = pickle.load(f)

# Re-build garment_group onehot (top-50+1=51) to match Polyvore cat layout
# H&M garment_group slice in hnm_struct: [30:60] = top-29+1=30 dims
# We need top-50+1=51 dims -> re-encode from raw articles.csv

print("  Re-encoding garment_group_no (top-50) for Polyvore compat ...")

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
SYNONYM = {"tee":"tshirt","trainer":"sneakers","trainers":"sneakers",
           "grey":"gray","burgundy":"red","ivory":"white"}
HNM_STOP = set(ENGLISH_STOP_WORDS) | {
    "hm","article","product","products","item","items",
    "collection","collections","fashion","look","looks","shop","style","styles"
}
def clean_hnm(text):
    if not text or (isinstance(text,float) and np.isnan(text)):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]"," ",text)
    text = re.sub(r"\s+"," ",text).strip()
    tokens = [SYNONYM.get(w,w) for w in text.split() if len(w)>1]
    return " ".join(t for t in tokens if t not in HNM_STOP)

text_columns = [
    "prod_name","product_type_name","product_group_name",
    "graphical_appearance_name","colour_group_name",
    "perceived_colour_value_name","perceived_colour_master_name",
    "department_name","index_name","index_group_name",
    "section_name","garment_group_name","detail_desc"
]
df_art = pd.read_csv(os.path.join(DATA_DIR, "articles.csv"))
existing = [c for c in text_columns if c in df_art.columns]
df_art["cleaned_text"] = (
    df_art[existing].fillna("").astype(str)
    .agg(" ".join, axis=1).apply(clean_hnm)
)
df_art = df_art[df_art["cleaned_text"].str.len() > 0].reset_index(drop=True)

# Top-50 garment_group_no
from collections import Counter
TOP_GAR = 50
gar_counts  = Counter(df_art["garment_group_no"].tolist())
gar_top50   = [c for c,_ in gar_counts.most_common(TOP_GAR)]
gar_to_idx  = {g: i for i,g in enumerate(gar_top50)}

def gar_onehot(val):
    vec = [0] * (TOP_GAR + 1)
    vec[gar_to_idx.get(val, TOP_GAR)] = 1
    return vec

# TPO keywords (same as Polyvore)
TPO_KEYS = ["office","party","wedding","beach","travel",
            "school","sport","outdoor","home","casual"]
TPO_KW   = {
    "office":  ["office","work","business","professional","formal"],
    "party":   ["party","date","night","club","cocktail"],
    "wedding": ["wedding","bridal","prom"],
    "beach":   ["beach","vacation","resort"],
    "travel":  ["travel","trip","airport","pack"],
    "school":  ["school","campus","college"],
    "sport":   ["sport","gym","running","training"],
    "outdoor": ["outdoor","hiking","camping"],
    "home":    ["home","decor","interior"],
    "casual":  ["casual","daily","street"],
}
def tpo_vec(text):
    tokens = set(text.lower().split())
    return [int(any(kw in tokens for kw in TPO_KW[k])) for k in TPO_KEYS]

# Build 63d compat struct
# [0]=0, [1]=0, [2:12]=TPO, [12:63]=garment_onehot(51)
hnm_struct_compat = np.zeros((N_HNM, 63), dtype=np.float32)
for i, row in df_art.iterrows():
    tpo   = tpo_vec(row["cleaned_text"])
    gar   = gar_onehot(row["garment_group_no"])
    hnm_struct_compat[i] = [0.0, 0.0] + tpo + gar

compat63_path = os.path.join(BASE_DIR, "hnm_struct_compat63.npy")
np.save(compat63_path, hnm_struct_compat)
print(f"  Saved: {compat63_path}  {hnm_struct_compat.shape}")

# Build H&M item feature matrix (1131d, matching Stage 1 input)
hnm_tfidf = np.load(os.path.join(BASE_DIR, "hnm_tfidf_300.npy"))  # (105542,300)
hnm_bert  = np.load(os.path.join(BASE_DIR, "hnm_bert_768.npy"))   # (105542,768)
X_hnm = np.concatenate([hnm_tfidf, hnm_bert, hnm_struct_compat], axis=1).astype(np.float32)
print(f"  X_hnm: {X_hnm.shape}")  # (105542, 1131)


# ─────────────────────────────────────────────────────────
# 2. H&M pair generation from transactions
# ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Generating H&M pairs from transactions")
print("=" * 60)

with open(os.path.join(BASE_DIR, "article_id_to_idx.pkl"), "rb") as f:
    art_to_idx = pickle.load(f)

tx_path = os.path.join(DATA_DIR, "transactions_train.csv")
print(f"  Reading transactions (3.3GB) ...")
t0 = time.time()

# Read in chunks to avoid OOM, collect baskets
# Basket = all articles bought by same customer on same day
CHUNK_SIZE = 1_000_000
MAX_TX_ROWS = 5_000_000   # use first 5M rows for speed

baskets = {}  # (customer_id, t_dat) -> [article_idx, ...]
rows_read = 0

for chunk in pd.read_csv(tx_path, chunksize=CHUNK_SIZE,
                          usecols=["t_dat","customer_id","article_id"]):
    for _, row in chunk.iterrows():
        art_idx = art_to_idx.get(int(row["article_id"]))
        if art_idx is None:
            continue
        key = (row["customer_id"], row["t_dat"])
        baskets.setdefault(key, []).append(art_idx)
    rows_read += len(chunk)
    print(f"  Rows read: {rows_read:,}  baskets so far: {len(baskets):,}", end="\r")
    if rows_read >= MAX_TX_ROWS:
        break

print(f"\n  Total baskets: {len(baskets):,}  ({time.time()-t0:.1f}s)")

# Filter baskets with >= 2 items
baskets = {k: list(set(v)) for k, v in baskets.items() if len(set(v)) >= 2}
print(f"  Baskets with >=2 items: {len(baskets):,}")


# ── Positive pairs: co-purchased items ──────────────────
pos_limit = int(MAX_PAIRS / (1 + NEG_RATIO))
pos_pairs = []
basket_list = list(baskets.values())
random.shuffle(basket_list)

for items in basket_list:
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            pos_pairs.append((items[i], items[j]))
            if len(pos_pairs) >= pos_limit:
                break
        if len(pos_pairs) >= pos_limit:
            break
    if len(pos_pairs) >= pos_limit:
        break

print(f"  Positive pairs: {len(pos_pairs):,}")

# ── Negative pairs: random cross-basket ─────────────────
n_neg = int(len(pos_pairs) * NEG_RATIO)
all_basket_items = list({idx for items in basket_list for idx in items})
neg_pairs = []
attempt, max_att = 0, n_neg * 5

# Build item->basket membership set for fast lookup
item_to_baskets = {}
for bid, items in enumerate(basket_list):
    for idx in items:
        item_to_baskets.setdefault(idx, set()).add(bid)

while len(neg_pairs) < n_neg and attempt < max_att:
    attempt += 1
    iA = random.choice(all_basket_items)
    iB = random.choice(all_basket_items)
    if iA == iB:
        continue
    # Ensure they don't share a basket
    if not (item_to_baskets.get(iA, set()) & item_to_baskets.get(iB, set())):
        neg_pairs.append((iA, iB))

print(f"  Negative pairs: {len(neg_pairs):,}")

# ── Metadata pair features (H&M version) ─────────────────
def meta_pair_hnm(iA, iB):
    sA, sB = hnm_struct_compat[iA], hnm_struct_compat[iB]
    tpo_A, tpo_B = sA[2:12], sB[2:12]
    gar_A, gar_B = np.argmax(sA[12:]), np.argmax(sB[12:])

    cat_match = float(gar_A == gar_B and gar_A < TOP_GAR)
    # H&M has colour_group_code -> use slice from full hnm_struct [10:30]
    col_A = np.argmax(hnm_struct[iA, 10:30])
    col_B = np.argmax(hnm_struct[iB, 10:30])
    color_match = float(col_A == col_B)

    nA, nB = np.linalg.norm(tpo_A), np.linalg.norm(tpo_B)
    tpo_overlap = float(np.dot(tpo_A,tpo_B)/(nA*nB)) if nA>0 and nB>0 else 0.0
    return np.array([cat_match, color_match, tpo_overlap, 0.0, 0.0],
                    dtype=np.float32)


# ─────────────────────────────────────────────────────────
# 3. Run Stage 1 inference -> p_compatible
# ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Stage 1 inference on H&M pairs")
print("=" * 60)

# Reload model
D_ITEM, D_EMB, D_META = 1131, 256, 5

class ProjectionMLP(nn.Module):
    def __init__(self, d_in=D_ITEM, d_out=D_EMB):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in,512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,d_out), nn.LayerNorm(d_out),
        )
    def forward(self, x): return self.net(x)

class FashFusionStage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = ProjectionMLP()
        self.fusion = nn.Sequential(
            nn.Linear(D_EMB*4+1+D_META,512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,1), nn.Sigmoid(),
        )
    def forward(self, item_A, item_B, meta_pair):
        eA = self.proj(item_A)
        eB = self.proj(item_B)
        cos = F.cosine_similarity(eA,eB,dim=1,eps=1e-8).unsqueeze(1)
        return self.fusion(torch.cat([eA,eB,torch.abs(eA-eB),eA*eB,cos,meta_pair],dim=1)).squeeze(1)

model = FashFusionStage1().to(DEVICE)
ckpt  = torch.load(os.path.join(BASE_DIR,"stage1_model.pt"), weights_only=True)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"  Stage 1 loaded (best epoch {ckpt['epoch']}, val AUC {ckpt['val_auc']:.4f})")

# Assemble all H&M pairs
all_pairs = [(1, iA, iB) for iA,iB in pos_pairs] + \
            [(0, iA, iB) for iA,iB in neg_pairs]
random.shuffle(all_pairs)

labels_all = np.array([r[0] for r in all_pairs], dtype=np.float32)
iA_all     = np.array([r[1] for r in all_pairs], dtype=np.int32)
iB_all     = np.array([r[2] for r in all_pairs], dtype=np.int32)
meta_all   = np.stack([meta_pair_hnm(r[1],r[2]) for r in all_pairs])

# Inference in batches
p_compat = np.zeros(len(all_pairs), dtype=np.float32)

with torch.no_grad():
    for start in range(0, len(all_pairs), BATCH_INFER):
        end  = min(start + BATCH_INFER, len(all_pairs))
        iA_b = torch.from_numpy(X_hnm[iA_all[start:end]]).to(DEVICE)
        iB_b = torch.from_numpy(X_hnm[iB_all[start:end]]).to(DEVICE)
        me_b = torch.from_numpy(meta_all[start:end]).to(DEVICE)
        p_compat[start:end] = model(iA_b, iB_b, me_b).cpu().numpy()
        if (start // BATCH_INFER) % 20 == 0:
            print(f"  Inferred {end:,}/{len(all_pairs):,}", end="\r")

print(f"\n  p_compatible stats:")
print(f"    mean={p_compat.mean():.4f}  std={p_compat.std():.4f}")
print(f"    pos mean={p_compat[labels_all==1].mean():.4f}  "
      f"neg mean={p_compat[labels_all==0].mean():.4f}")


# ─────────────────────────────────────────────────────────
# 4. Save Stage 2 dataset
# ─────────────────────────────────────────────────────────
print()
print("Saving Stage 2 dataset ...")

item_A_t   = torch.from_numpy(X_hnm[iA_all])
item_B_t   = torch.from_numpy(X_hnm[iB_all])
meta_t     = torch.from_numpy(meta_all)
p_compat_t = torch.from_numpy(p_compat)
labels_t   = torch.from_numpy(labels_all)

# Train/val split (90/10)
n      = len(all_pairs)
n_tr   = int(n * 0.9)
perm   = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))

def save_hnm_split(name, indices):
    idx = perm[indices]
    out = os.path.join(BASE_DIR, f"hnm_pairs_{name}.pt")
    torch.save({
        "item_A":      item_A_t[idx],
        "item_B":      item_B_t[idx],
        "meta_pair":   meta_t[idx],
        "p_compatible": p_compat_t[idx],
        "labels":      labels_t[idx],
    }, out)
    pos = int(labels_t[idx].sum())
    print(f"  {name}: {len(idx):,} pairs  pos={pos:,}  saved -> {out}")

save_hnm_split("train", range(n_tr))
save_hnm_split("val",   range(n_tr, n))

np.save(os.path.join(BASE_DIR,"hnm_p_compat.npy"), p_compat)
print(f"  hnm_p_compat.npy: {p_compat.shape} saved")

print()
print("=" * 60)
print("STEP 7 COMPLETE")
print("=" * 60)
print(f"  H&M pairs total  : {n:,}")
print(f"    Positive (co-purchased) : {len(pos_pairs):,}")
print(f"    Negative (random)       : {len(neg_pairs):,}")
print(f"  p_compatible range: [{p_compat.min():.3f}, {p_compat.max():.3f}]")
print()
print("  Next -> Step 8: Stage 2 co-purchase MLP training")
