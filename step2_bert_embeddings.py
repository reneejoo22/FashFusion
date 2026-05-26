"""
Step 2: DistilBERT item embeddings (768d, mean pooling, frozen)
  Polyvore: item name text  -> poly_bert_768.npy  (142480 x 768)
  H&M     : article text    -> hnm_bert_768.npy   (105542 x 768)

  Model: distilbert-base-uncased, frozen feature extractor
  Pooling: attention-mask weighted mean pooling
  Device: CUDA (RTX 5070 Ti) if available, else CPU
"""

import os
import re
import json
import pickle
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256   # increase if VRAM allows
MAX_LEN    = 64    # item names are short; articles capped at 64 tokens
MODEL_NAME = "distilbert-base-uncased"

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ─────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────
STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "polyvore","featuring","browse","related","website",
    "official","http","https","www","com","html",
    "utm","source","medium","contest","contestentry",
    "fashion","look","looks","shop","set","style","styles"
}
SYNONYM = {
    "tee":"tshirt","trainer":"sneakers","trainers":"sneakers",
    "grey":"gray","burgundy":"red","ivory":"white"
}

def clean(text):
    if not text or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [SYNONYM.get(w, w) for w in text.split() if len(w) > 1]
    return " ".join(t for t in tokens if t not in STOPWORDS)


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts     = texts
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx] if self.texts[idx] else "[PAD]"
        enc  = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }


def mean_pool(last_hidden, attention_mask):
    """Attention-mask weighted mean pooling."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_texts(texts, tokenizer, model, batch_size, max_len, name):
    """Encode all texts and return L2-normalized float32 array."""
    dataset = TextDataset(texts, tokenizer, max_len)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0, pin_memory=True)

    all_embs = []
    t0 = time.time()
    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)

        out  = model(input_ids=input_ids, attention_mask=attn_mask)
        embs = mean_pool(out.last_hidden_state, attn_mask)  # (B, 768)
        all_embs.append(embs.cpu().float().numpy())

        if (i+1) % 50 == 0:
            done = (i+1) * batch_size
            elapsed = time.time() - t0
            speed = done / elapsed
            eta = (len(texts) - done) / speed if speed > 0 else 0
            print(f"  [{name}] {done:,}/{len(texts):,} "
                  f"({done/len(texts)*100:.1f}%)  "
                  f"{speed:.0f} items/s  ETA {eta:.0f}s")

    embs_np = np.concatenate(all_embs, axis=0).astype(np.float32)

    # L2 normalize
    norms = np.linalg.norm(embs_np, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    embs_np = embs_np / norms

    elapsed = time.time() - t0
    print(f"  [{name}] Done: {embs_np.shape}  "
          f"total {elapsed:.1f}s ({len(texts)/elapsed:.0f} items/s)")
    return embs_np


# ─────────────────────────────────────────────────────────
# Load model once
# ─────────────────────────────────────────────────────────
print()
print("Loading DistilBERT tokenizer + model ...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model     = DistilBertModel.from_pretrained(MODEL_NAME)
model.eval()
model.to(DEVICE)
print(f"Model loaded  |  params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")


# ─────────────────────────────────────────────────────────
# 1. POLYVORE ITEMS
# ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("POLYVORE: DistilBERT item embeddings")
print("=" * 60)

POLY_DIR = os.path.join(DATA_DIR, "polyvore")
splits   = ["train_no_dup.json", "valid_no_dup.json", "test_no_dup.json"]

# Rebuild item order that matches item_id_list.pkl
with open(os.path.join(BASE_DIR, "item_id_list.pkl"), "rb") as f:
    item_id_list = pickle.load(f)

# Build set_id+index -> name lookup from raw JSON
item_name_map = {}
for fname in splits:
    path = os.path.join(POLY_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for outfit in data:
        sid = str(outfit.get("set_id", ""))
        for item in outfit.get("items", []):
            iid = f"{sid}_{item.get('index', 0)}"
            item_name_map[iid] = clean(item.get("name", ""))

print(f"  Item name lookup: {len(item_name_map):,} entries")

# Build text list in item_id_list order
poly_texts = [item_name_map.get(iid, "") for iid in item_id_list]
empty_cnt  = sum(1 for t in poly_texts if not t)
print(f"  Items with empty text: {empty_cnt} / {len(poly_texts)}")

poly_bert = encode_texts(poly_texts, tokenizer, model,
                         BATCH_SIZE, MAX_LEN, "Polyvore")

poly_bert_path = os.path.join(BASE_DIR, "poly_bert_768.npy")
np.save(poly_bert_path, poly_bert)
print(f"  Saved: {poly_bert_path}  ({poly_bert.nbytes/1e6:.1f} MB)")


# ─────────────────────────────────────────────────────────
# 2. H&M ARTICLES
# ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("H&M: DistilBERT article embeddings")
print("=" * 60)

HNM_DOMAIN_STOP = {
    "hm","article","product","products","item","items",
    "collection","collections","fashion","look","looks",
    "shop","style","styles"
}
ALL_HNM_STOP = set(ENGLISH_STOP_WORDS) | HNM_DOMAIN_STOP

def clean_hnm(text):
    if not text or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [SYNONYM.get(w, w) for w in text.split() if len(w) > 1]
    return " ".join(t for t in tokens if t not in ALL_HNM_STOP)

articles_path = os.path.join(DATA_DIR, "hnm", "articles.csv")
df_art = pd.read_csv(articles_path)

text_columns = [
    "prod_name","product_type_name","product_group_name",
    "graphical_appearance_name","colour_group_name",
    "perceived_colour_value_name","perceived_colour_master_name",
    "department_name","index_name","index_group_name",
    "section_name","garment_group_name","detail_desc"
]
existing_cols = [c for c in text_columns if c in df_art.columns]

df_art["cleaned_text"] = (
    df_art[existing_cols].fillna("").astype(str)
    .agg(" ".join, axis=1)
    .apply(clean_hnm)
)
df_art = df_art[df_art["cleaned_text"].str.len() > 0].reset_index(drop=True)
print(f"  Articles after filter: {len(df_art):,}")

# Verify row order matches hnm_tfidf_300.npy
with open(os.path.join(BASE_DIR, "article_id_list.pkl"), "rb") as f:
    article_id_list = pickle.load(f)

assert len(df_art) == len(article_id_list), (
    f"Row count mismatch: df={len(df_art)} vs mapping={len(article_id_list)}"
)
assert list(df_art["article_id"]) == article_id_list, \
    "article_id order mismatch with article_id_list.pkl"
print("  Row order verified vs article_id_list.pkl")

hnm_texts = df_art["cleaned_text"].tolist()

hnm_bert = encode_texts(hnm_texts, tokenizer, model,
                        BATCH_SIZE, MAX_LEN, "H&M")

hnm_bert_path = os.path.join(BASE_DIR, "hnm_bert_768.npy")
np.save(hnm_bert_path, hnm_bert)
print(f"  Saved: {hnm_bert_path}  ({hnm_bert.nbytes/1e6:.1f} MB)")


# ─────────────────────────────────────────────────────────
# 3. Sanity check: repeat Step 1 style check with BERT
# ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SANITY CHECK (BERT cosine sim on compat pairs)")
print("=" * 60)

with open(os.path.join(BASE_DIR, "poly_compat_pairs.pkl"), "rb") as f:
    compat_pairs = pickle.load(f)
with open(os.path.join(BASE_DIR, "item_id_to_idx.pkl"), "rb") as f:
    item_id_to_idx = pickle.load(f)

pos_pairs = [(l,ids) for l,ids in compat_pairs if l==1]
neg_pairs = [(l,ids) for l,ids in compat_pairs if l==0]

pos_sims, neg_sims = [], []
for label, item_ids in pos_pairs[:200] + neg_pairs[:200]:
    idxs = [item_id_to_idx[iid] for iid in item_ids if iid in item_id_to_idx]
    if len(idxs) < 2:
        continue
    vecs = poly_bert[idxs]
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            s = float(np.dot(vecs[i], vecs[j]))
            (pos_sims if label==1 else neg_sims).append(s)

print(f"  BERT Pos sim : {np.mean(pos_sims):.4f} (n={len(pos_sims)})")
print(f"  BERT Neg sim : {np.mean(neg_sims):.4f} (n={len(neg_sims)})")
print(f"  BERT Gap     : {np.mean(pos_sims)-np.mean(neg_sims):.4f}")
print()
tfidf_300 = np.load(os.path.join(BASE_DIR, "poly_tfidf_300.npy"))
pos_t, neg_t = [], []
for label, item_ids in pos_pairs[:200] + neg_pairs[:200]:
    idxs = [item_id_to_idx[iid] for iid in item_ids if iid in item_id_to_idx]
    if len(idxs) < 2:
        continue
    vecs = tfidf_300[idxs]
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            s = float(np.dot(vecs[i], vecs[j]))
            (pos_t if label==1 else neg_t).append(s)

print(f"  TFIDF Pos sim: {np.mean(pos_t):.4f}  Neg: {np.mean(neg_t):.4f}  "
      f"Gap: {np.mean(pos_t)-np.mean(neg_t):.4f}")
print()
print("  Next -> Step 3: MobileNet image embeddings (1280d)")
