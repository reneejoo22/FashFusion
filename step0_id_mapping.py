"""
Step 0: ID Mapping Tables
  - Polyvore: item_id_to_idx.pkl  ('{set_id}_{index}' -> int)
              outfit_id_to_idx.pkl (set_id -> int, for outfit-level TF-IDF lookup)
              poly_item_tfidf.npz  (item-level sparse TF-IDF, shape [N_items, 20000])
  - H&M    : article_id_to_idx.pkl (article_id -> int, matches hnm_tfidf_features_final.npz row order)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ─────────────────────────────────────────────────────────
# 1. POLYVORE
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("POLYVORE: building item-level ID mapping")
print("=" * 60)

POLY_DIR = os.path.join(DATA_DIR, "polyvore")
splits = ["train_no_dup.json", "valid_no_dup.json", "test_no_dup.json"]

# ── 1-1. Load all splits, collect items ──────────────────
all_outfits = []
for fname in splits:
    path = os.path.join(POLY_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_outfits.extend(data)
    print(f"  {fname}: {len(data)} outfits")

print(f"  Total outfits (all splits): {len(all_outfits)}")

# ── 1-2. Build outfit_id_to_idx ──────────────────────────
#    outfit-level TF-IDF from polyvore_text_features_final.json
#    (which was built from train_no_dup.json only = 17316 rows)
with open(os.path.join(BASE_DIR, "polyvore_text_features_final.json"),
          "r", encoding="utf-8") as f:
    proc_outfits = json.load(f)

outfit_id_to_idx = {
    str(o["set_id"]): idx
    for idx, o in enumerate(proc_outfits)
}
print(f"\n  outfit_id_to_idx: {len(outfit_id_to_idx)} train outfits")

outfit_path = os.path.join(BASE_DIR, "outfit_id_to_idx.pkl")
with open(outfit_path, "wb") as f:
    pickle.dump(outfit_id_to_idx, f)
print(f"  Saved: {outfit_path}")

# ── 1-3. Build item_id_to_idx from ALL splits ─────────────
#    Item ID format: "{set_id}_{index}"
#    We also collect item text for per-item TF-IDF

import re
import sys
sys.path.insert(0, BASE_DIR)

# Reuse the same text-cleaning logic from feature.py
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import defaultdict

STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "polyvore", "featuring", "browse", "related", "website",
    "official", "http", "https", "www", "com", "html",
    "utm", "source", "medium", "contest", "contestentry",
    "fashion", "look", "looks", "shop", "set", "style", "styles"
}
SYNONYM = {
    "tee": "tshirt", "trainer": "sneakers", "trainers": "sneakers",
    "grey": "gray", "burgundy": "red", "ivory": "white"
}

def clean(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [SYNONYM.get(w, w) for w in text.split() if len(w) > 1]
    return " ".join(t for t in tokens if t not in STOPWORDS)


# Collect ordered list of (item_id, item_text)
item_records = []   # [(item_id_str, cleaned_text), ...]
seen = set()

for outfit in all_outfits:
    set_id = str(outfit.get("set_id", ""))
    for item in outfit.get("items", []):
        idx_in_outfit = item.get("index", 0)
        item_id = f"{set_id}_{idx_in_outfit}"
        if item_id in seen:
            continue
        seen.add(item_id)
        item_text = clean(item.get("name", ""))
        item_records.append((item_id, item_text))

print(f"\n  Total unique items: {len(item_records)}")

# Build item_id_to_idx
item_id_to_idx = {rec[0]: i for i, rec in enumerate(item_records)}

item_id_path = os.path.join(BASE_DIR, "item_id_to_idx.pkl")
with open(item_id_path, "wb") as f:
    pickle.dump(item_id_to_idx, f)
print(f"  Saved: {item_id_path}")

# Save id order list for reproducibility
item_id_list_path = os.path.join(BASE_DIR, "item_id_list.pkl")
item_id_list = [rec[0] for rec in item_records]
with open(item_id_list_path, "wb") as f:
    pickle.dump(item_id_list, f)
print(f"  Saved: {item_id_list_path}")

# ── 1-4. Item-level TF-IDF (transform with train vectorizer) ──
#    Use the existing trained vectorizer so vocab is consistent
vec_path = os.path.join(BASE_DIR, "polyvore_tfidf_vectorizer_final.pkl")
with open(vec_path, "rb") as f:
    vectorizer = pickle.load(f)

item_texts = [rec[1] for rec in item_records]
X_item = vectorizer.transform(item_texts)
print(f"\n  Item TF-IDF shape: {X_item.shape}")

item_tfidf_path = os.path.join(BASE_DIR, "poly_item_tfidf.npz")
sp.save_npz(item_tfidf_path, X_item)
print(f"  Saved: {item_tfidf_path}")

# ── 1-5. Parse compatibility prediction file ──────────────
compat_path = os.path.join(POLY_DIR, "fashion_compatibility_prediction.txt")
compat_pairs = []
missing_items = set()

with open(compat_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        label = int(parts[0])
        item_ids = parts[1:]
        # Check coverage
        for iid in item_ids:
            if iid not in item_id_to_idx:
                missing_items.add(iid)
        compat_pairs.append((label, item_ids))

print(f"\n  Compatibility pairs: {len(compat_pairs)}")
print(f"  Positive (1): {sum(1 for l,_ in compat_pairs if l==1)}")
print(f"  Negative (0): {sum(1 for l,_ in compat_pairs if l==0)}")
print(f"  Item IDs missing from mapping: {len(missing_items)}")
if missing_items:
    print(f"  Sample missing: {list(missing_items)[:5]}")

# Save compatibility pairs for Step 5
compat_pkl_path = os.path.join(BASE_DIR, "poly_compat_pairs.pkl")
with open(compat_pkl_path, "wb") as f:
    pickle.dump(compat_pairs, f)
print(f"  Saved: {compat_pkl_path}")


# ─────────────────────────────────────────────────────────
# 2. H&M
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("H&M: building article_id_to_idx mapping")
print("=" * 60)

HNM_DIR = os.path.join(DATA_DIR, "hnm")
articles_path = os.path.join(HNM_DIR, "articles.csv")

# Load articles and apply same cleaning to reconstruct row order
# Must replicate feature2.py exactly to match hnm_tfidf_features_final.npz
df_art = pd.read_csv(articles_path)
print(f"  Raw articles: {len(df_art)}")

text_columns = [
    "prod_name", "product_type_name", "product_group_name",
    "graphical_appearance_name", "colour_group_name",
    "perceived_colour_value_name", "perceived_colour_master_name",
    "department_name", "index_name", "index_group_name",
    "section_name", "garment_group_name", "detail_desc"
]
existing_cols = [c for c in text_columns if c in df_art.columns]

HNM_DOMAIN_STOP = {
    "hm", "article", "product", "products", "item", "items",
    "collection", "collections", "fashion", "look", "looks",
    "shop", "style", "styles"
}
HNM_STOPWORDS = set(ENGLISH_STOP_WORDS) | HNM_DOMAIN_STOP

def clean_hnm(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [SYNONYM.get(w, w) for w in text.split() if len(w) > 1]
    return " ".join(t for t in tokens if t not in HNM_STOPWORDS)

df_art["raw_text"] = (
    df_art[existing_cols]
    .fillna("").astype(str)
    .agg(" ".join, axis=1)
)
df_art["cleaned_text"] = df_art["raw_text"].apply(clean_hnm)
df_art = df_art[df_art["cleaned_text"].str.len() > 0].reset_index(drop=True)
print(f"  After cleaning filter: {len(df_art)}")

# Cross-check with saved npz
X_hnm = sp.load_npz(os.path.join(BASE_DIR, "hnm_tfidf_features_final.npz"))
print(f"  hnm_tfidf shape: {X_hnm.shape}")

if len(df_art) != X_hnm.shape[0]:
    print(f"  WARNING: row count mismatch! articles={len(df_art)}, npz={X_hnm.shape[0]}")
    print("  Attempting to load with hnm vectorizer for verification...")
else:
    print(f"  OK Row counts match: {len(df_art)}")

# Build article_id_to_idx
article_id_to_idx = {
    int(row["article_id"]): idx
    for idx, row in df_art.iterrows()
}
print(f"  article_id_to_idx: {len(article_id_to_idx)} articles")

art_path = os.path.join(BASE_DIR, "article_id_to_idx.pkl")
with open(art_path, "wb") as f:
    pickle.dump(article_id_to_idx, f)
print(f"  Saved: {art_path}")

# Save article_id_list for reproducibility
art_list_path = os.path.join(BASE_DIR, "article_id_list.pkl")
article_id_list = df_art["article_id"].tolist()
with open(art_list_path, "wb") as f:
    pickle.dump(article_id_list, f)
print(f"  Saved: {art_list_path}")

# Peek at transactions for pair generation preview
tx_path = os.path.join(HNM_DIR, "transactions_train.csv")
if os.path.exists(tx_path):
    df_tx = pd.read_csv(tx_path, nrows=5)
    print(f"\n  transactions_train.csv columns: {list(df_tx.columns)}")
    print(df_tx.head(3))
else:
    print("\n  transactions_train.csv not found at expected path")


# ─────────────────────────────────────────────────────────
# 3. Summary
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 0 COMPLETE - Summary")
print("=" * 60)
print(f"  item_id_to_idx.pkl      : {len(item_id_to_idx):,} Polyvore items")
print(f"  outfit_id_to_idx.pkl    : {len(outfit_id_to_idx):,} Polyvore train outfits")
print(f"  poly_item_tfidf.npz     : {X_item.shape}")
print(f"  poly_compat_pairs.pkl   : {len(compat_pairs):,} labeled pairs")
print(f"  article_id_to_idx.pkl   : {len(article_id_to_idx):,} H&M articles")
print()
print("  Next → Step 1: TF-IDF → SVD 300d reduction")
