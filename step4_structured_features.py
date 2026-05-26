"""
Step 4: Structured / metadata features
  Polyvore items -> poly_struct.npy   (N_items x D_poly)
  H&M articles  -> hnm_struct.npy    (N_art   x D_hnm)
  Encoders saved for pair-level feature computation in Step 5

  Polyvore item feature vector layout:
    [0]       price_log_norm         (1)   log(price+1), then MinMax
    [1]       likes_log_norm         (1)
    [2:12]    tpo_vector             (10)  outfit-level TPO flags
    [12:63]   category_onehot        (51)  top-50 cats + 1 "other"
    total = 63d

  H&M article feature vector layout:
    [0:10]    tpo_vector             (10)  text-derived TPO flags
    [10:30]   colour_group_onehot    (20)  top-19 + other
    [30:60]   garment_group_onehot   (30)  top-29 + other
    [60:75]   index_group_onehot     (15)  top-14 + other
    [75:105]  product_type_onehot    (30)  top-29 + other
    total = 105d
"""

import os, json, pickle, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ─────────────────────────────────────────────────────────
# TPO keyword extractor (same as feature.py)
# ─────────────────────────────────────────────────────────
TPO_KEYWORDS = {
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
TPO_KEYS = list(TPO_KEYWORDS.keys())  # fixed order, length 10

def tpo_vector(text):
    tokens = set(text.lower().split())
    return [int(any(kw in tokens for kw in TPO_KEYWORDS[k])) for k in TPO_KEYS]


def make_onehot_encoder(values_series, top_k):
    """
    Returns (categories_list, encode_fn).
    categories_list has length top_k+1 (top_k real cats + 'other').
    """
    top_cats = (
        values_series.value_counts()
        .head(top_k)
        .index.tolist()
    )
    cat_to_idx = {c: i for i, c in enumerate(top_cats)}

    def encode(val):
        vec = [0] * (top_k + 1)
        idx = cat_to_idx.get(val, top_k)  # unknown -> last 'other' slot
        vec[idx] = 1
        return vec

    return top_cats, encode


# ─────────────────────────────────────────────────────────
# 1. POLYVORE
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("POLYVORE structured features")
print("=" * 60)

POLY_DIR = os.path.join(DATA_DIR, "polyvore")
splits   = ["train_no_dup.json", "valid_no_dup.json", "test_no_dup.json"]

# Load all outfits for TPO (from processed JSON which has outfit-level TPO)
with open(os.path.join(BASE_DIR, "polyvore_text_features_final.json"),
          "r", encoding="utf-8") as f:
    proc_outfits = json.load(f)

outfit_tpo_map = {
    str(o["set_id"]): [o.get(f"tpo_{k}", 0) for k in TPO_KEYS]
    for o in proc_outfits
}

# Load raw JSON for item-level price/likes/category
all_items_raw = {}  # item_id -> dict
for fname in splits:
    path = os.path.join(POLY_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for outfit in data:
        sid = str(outfit.get("set_id",""))
        for item in outfit.get("items", []):
            iid   = f"{sid}_{item.get('index',0)}"
            price = item.get("price", -1)
            likes = item.get("likes", 0)
            catid = item.get("categoryid", -1)
            all_items_raw[iid] = {
                "price":  price if isinstance(price,(int,float)) and price > 0 else 0.0,
                "likes":  likes if isinstance(likes,(int,float)) else 0.0,
                "catid":  str(catid),
                "set_id": sid,
            }

print(f"  Raw item records loaded: {len(all_items_raw):,}")

# Load item_id_list (ordered)
with open(os.path.join(BASE_DIR, "item_id_list.pkl"), "rb") as f:
    item_id_list = pickle.load(f)

# Build raw arrays for scaler fitting (train items only)
with open(os.path.join(BASE_DIR, "outfit_id_to_idx.pkl"), "rb") as f:
    outfit_id_to_idx = pickle.load(f)
train_set_ids = set(outfit_id_to_idx.keys())

# Collect price/likes for train items to fit MinMaxScaler
train_prices = []
train_likes  = []
cat_series   = []

for iid in item_id_list:
    r = all_items_raw.get(iid, {})
    sid = iid.rsplit("_",1)[0]
    is_train = sid in train_set_ids
    if is_train:
        train_prices.append(np.log1p(r.get("price",0)))
        train_likes.append(np.log1p(r.get("likes",0)))
    cat_series.append(r.get("catid","−1"))

# MinMaxScaler on train
price_scaler = MinMaxScaler()
likes_scaler = MinMaxScaler()
price_scaler.fit(np.array(train_prices).reshape(-1,1))
likes_scaler.fit(np.array(train_likes).reshape(-1,1))

# Category one-hot encoder (top-50, fit on train)
train_cats = [
    all_items_raw.get(iid,{}).get("catid","-1")
    for iid in item_id_list
    if iid.rsplit("_",1)[0] in train_set_ids
]
train_cat_series = pd.Series(train_cats)
TOP_CAT = 50
cat_top, cat_encode = make_onehot_encoder(train_cat_series, TOP_CAT)
print(f"  Top-{TOP_CAT} categories cover: "
      f"{train_cat_series.isin(cat_top).mean()*100:.1f}% of train items")

# Build full feature matrix
D_POLY = 1 + 1 + 10 + (TOP_CAT + 1)  # = 63
poly_struct = np.zeros((len(item_id_list), D_POLY), dtype=np.float32)

for i, iid in enumerate(item_id_list):
    r   = all_items_raw.get(iid, {})
    sid = iid.rsplit("_",1)[0]

    price_n = float(price_scaler.transform(
        [[np.log1p(r.get("price",0))]])[0][0])
    likes_n = float(likes_scaler.transform(
        [[np.log1p(r.get("likes",0))]])[0][0])

    tpo  = outfit_tpo_map.get(sid, [0]*10)
    cat  = cat_encode(r.get("catid","-1"))

    row = [price_n, likes_n] + tpo + cat
    poly_struct[i] = row

poly_out = os.path.join(BASE_DIR, "poly_struct.npy")
np.save(poly_out, poly_struct)
print(f"  poly_struct.npy: {poly_struct.shape}  saved -> {poly_out}")

# Save encoders
poly_encoders = {
    "price_scaler": price_scaler,
    "likes_scaler": likes_scaler,
    "cat_top":      cat_top,
    "TOP_CAT":      TOP_CAT,
    "D_POLY":       D_POLY,
    "tpo_keys":     TPO_KEYS,
}
with open(os.path.join(BASE_DIR, "poly_struct_encoders.pkl"), "wb") as f:
    pickle.dump(poly_encoders, f)
print("  poly_struct_encoders.pkl saved")


# ─────────────────────────────────────────────────────────
# 2. H&M ARTICLES
# ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("H&M structured features")
print("=" * 60)

SYNONYM = {
    "tee":"tshirt","trainer":"sneakers","trainers":"sneakers",
    "grey":"gray","burgundy":"red","ivory":"white"
}
HNM_STOP = set(ENGLISH_STOP_WORDS) | {
    "hm","article","product","products","item","items",
    "collection","collections","fashion","look","looks",
    "shop","style","styles"
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

df_art = pd.read_csv(os.path.join(DATA_DIR,"hnm","articles.csv"))
existing = [c for c in text_columns if c in df_art.columns]
df_art["cleaned_text"] = (
    df_art[existing].fillna("").astype(str)
    .agg(" ".join, axis=1).apply(clean_hnm)
)
df_art = df_art[df_art["cleaned_text"].str.len() > 0].reset_index(drop=True)
print(f"  Articles: {len(df_art):,}")

# TPO from text
df_art["tpo"] = df_art["cleaned_text"].apply(tpo_vector)

# Categorical one-hot encoders
CAT_SPECS = [
    ("colour_group_code",        19),
    ("garment_group_no",         29),
    ("index_group_no",           14),
    ("product_type_no",          29),
]

encoders_hnm = {}
D_HNM = 10  # tpo
for col, top_k in CAT_SPECS:
    if col not in df_art.columns:
        print(f"  WARNING: {col} not found, skipping")
        continue
    top_cats, enc_fn = make_onehot_encoder(df_art[col], top_k)
    encoders_hnm[col] = (top_k, top_cats, enc_fn)
    coverage = df_art[col].isin(top_cats).mean() * 100
    print(f"  {col}: top-{top_k} covers {coverage:.1f}%")
    D_HNM += top_k + 1

print(f"  H&M struct dim: {D_HNM}")

hnm_struct = np.zeros((len(df_art), D_HNM), dtype=np.float32)

for i, row in df_art.iterrows():
    vec = list(row["tpo"])
    for col, top_k in CAT_SPECS:
        if col not in encoders_hnm:
            vec += [0] * (top_k + 1)
            continue
        _, _, enc_fn = encoders_hnm[col]
        vec += enc_fn(row[col])
    hnm_struct[i] = vec

hnm_out = os.path.join(BASE_DIR, "hnm_struct.npy")
np.save(hnm_out, hnm_struct)
print(f"  hnm_struct.npy : {hnm_struct.shape}  saved -> {hnm_out}")

hnm_encoder_info = {
    "CAT_SPECS":  CAT_SPECS,
    "encoders":   {k: (v[0], v[1]) for k,v in encoders_hnm.items()},
    "D_HNM":      D_HNM,
    "tpo_keys":   TPO_KEYS,
}
with open(os.path.join(BASE_DIR, "hnm_struct_encoders.pkl"), "wb") as f:
    pickle.dump(hnm_encoder_info, f)
print("  hnm_struct_encoders.pkl saved")


# ─────────────────────────────────────────────────────────
# 3. Quick stats
# ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("FEATURE STATS")
print("=" * 60)
print(f"  poly_struct : {poly_struct.shape}  "
      f"mean={poly_struct.mean():.3f}  std={poly_struct.std():.3f}")
print(f"  hnm_struct  : {hnm_struct.shape}  "
      f"mean={hnm_struct.mean():.3f}  std={hnm_struct.std():.3f}")

# TPO coverage
poly_tpo = poly_struct[:, 2:12]
hnm_tpo  = hnm_struct[:, :10]
print(f"  Poly TPO non-zero: {(poly_tpo.sum(axis=1)>0).mean()*100:.1f}% items")
print(f"  H&M  TPO non-zero: {(hnm_tpo.sum(axis=1)>0).mean()*100:.1f}% items")
print()
print("  Next -> Step 5: Polyvore pair dataset generation")
