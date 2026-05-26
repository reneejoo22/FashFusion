"""
Step 5: Polyvore pair dataset for Stage 1 training

Polyvore compatibility prediction task (standard setup):
  Train pairs: self-generated from train_no_dup.json
    - Positive: item pairs within the same outfit
    - Negative: hard negatives (same category, different outfit)
               + easy negatives (random cross-outfit pairs)
  Val pairs  : self-generated from valid_no_dup.json (same strategy)
  Test pairs : official benchmark -> fashion_compatibility_prediction.txt

Item features (text-only, no image):
  [tfidf_300 | bert_768 | struct_63] = 1131d per item

Pair feature:
  [A(1131), B(1131), |A-B|(1131), A*B(1131), cosine_sim(1), meta_pair(5)] = 4530d
  -> After projection MLP: A_emb(256), B_emb(256), |..|(256), ..(256), cos(1), meta(5) = 1030d

Outputs:
  poly_pairs_train.pt, poly_pairs_val.pt, poly_pairs_test.pt
  pair_split_info.pkl
"""

import os, json, pickle, random
import numpy as np
import torch

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
POLY_DIR  = os.path.join(BASE_DIR, "data", "polyvore")

SEED          = 42
NEG_RATIO     = 1.5   # negatives per positive
HARD_NEG_FRAC = 0.5   # fraction of negatives that are hard (same category)
MAX_PAIRS_PER_OUTFIT = 10  # cap within-outfit pairs to avoid large outfits dominating

random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────
# Load feature arrays
# ─────────────────────────────────────────────────────────
print("Loading feature arrays ...")
tfidf  = np.load(os.path.join(BASE_DIR, "poly_tfidf_300.npy"))
bert   = np.load(os.path.join(BASE_DIR, "poly_bert_768.npy"))
struct = np.load(os.path.join(BASE_DIR, "poly_struct.npy"))

with open(os.path.join(BASE_DIR, "item_id_to_idx.pkl"), "rb") as f:
    item_id_to_idx = pickle.load(f)
with open(os.path.join(BASE_DIR, "item_id_list.pkl"), "rb") as f:
    item_id_list = pickle.load(f)

X_items = np.concatenate([tfidf, bert, struct], axis=1).astype(np.float32)
D_ITEM  = X_items.shape[1]  # 1131
print(f"  X_items: {X_items.shape}")

# item_idx -> category index (from struct: onehot starts at dim 12)
def get_cat_idx(item_idx):
    return int(np.argmax(struct[item_idx, 12:]))  # 0..49 real, 50=other


# ─────────────────────────────────────────────────────────
# Metadata pair features
# ─────────────────────────────────────────────────────────
def meta_pair(iA, iB):
    sA, sB = struct[iA], struct[iB]
    price_A, price_B = float(sA[0]), float(sB[0])
    likes_A, likes_B = float(sA[1]), float(sB[1])
    tpo_A, tpo_B     = sA[2:12], sB[2:12]
    cat_A, cat_B     = np.argmax(sA[12:]), np.argmax(sB[12:])

    cat_match = float(cat_A == cat_B and cat_A < 50)
    color_match = 0.0  # Polyvore has no color code
    nA, nB = np.linalg.norm(tpo_A), np.linalg.norm(tpo_B)
    tpo_overlap = float(np.dot(tpo_A,tpo_B)/(nA*nB)) if nA>0 and nB>0 else 0.0
    return np.array([cat_match, color_match, tpo_overlap,
                     abs(price_A-price_B), abs(likes_A-likes_B)],
                    dtype=np.float32)


# ─────────────────────────────────────────────────────────
# Pair generation for one split
# ─────────────────────────────────────────────────────────
def generate_pairs(split_name, json_fname):
    print(f"\n  [{split_name}] Loading {json_fname} ...")
    with open(os.path.join(POLY_DIR, json_fname), "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect valid item indices per outfit
    outfits = []
    for outfit in data:
        sid   = str(outfit.get("set_id",""))
        idxs  = [item_id_to_idx[f"{sid}_{item['index']}"]
                 for item in outfit.get("items",[])
                 if f"{sid}_{item['index']}" in item_id_to_idx]
        if len(idxs) >= 2:
            outfits.append(idxs)

    print(f"  [{split_name}] Valid outfits: {len(outfits)}")

    # ── Positive pairs ──────────────────────────────────
    pos_pairs = []
    for idxs in outfits:
        pairs = [(idxs[i], idxs[j])
                 for i in range(len(idxs))
                 for j in range(i+1, len(idxs))]
        if len(pairs) > MAX_PAIRS_PER_OUTFIT:
            pairs = random.sample(pairs, MAX_PAIRS_PER_OUTFIT)
        pos_pairs.extend(pairs)

    print(f"  [{split_name}] Positive pairs: {len(pos_pairs):,}")

    # ── Negative pairs ───────────────────────────────────
    n_neg      = int(len(pos_pairs) * NEG_RATIO)
    n_hard     = int(n_neg * HARD_NEG_FRAC)
    n_easy     = n_neg - n_hard

    # Build category -> outfit list for hard negatives
    all_outfit_items = [idx for idxs in outfits for idx in idxs]
    cat_to_items     = {}
    for idx in all_outfit_items:
        c = get_cat_idx(idx)
        cat_to_items.setdefault(c, []).append(idx)

    # Also build outfit membership to prevent same-outfit negatives
    item_to_outfit = {}
    for oid, idxs in enumerate(outfits):
        for idx in idxs:
            item_to_outfit[idx] = oid

    neg_pairs = []

    # Hard negatives: same category, different outfit
    attempt, max_attempt = 0, n_hard * 10
    while len(neg_pairs) < n_hard and attempt < max_attempt:
        attempt += 1
        iA = random.choice(all_outfit_items)
        c  = get_cat_idx(iA)
        candidates = [x for x in cat_to_items.get(c,[]) if x != iA]
        if not candidates:
            continue
        iB = random.choice(candidates)
        if item_to_outfit.get(iA) != item_to_outfit.get(iB):
            neg_pairs.append((iA, iB))

    # Easy negatives: random cross-outfit pairs
    attempt, max_attempt = 0, n_easy * 5
    while len(neg_pairs) < n_neg and attempt < max_attempt:
        attempt += 1
        oid_A = random.randrange(len(outfits))
        oid_B = random.randrange(len(outfits))
        if oid_A == oid_B:
            continue
        iA = random.choice(outfits[oid_A])
        iB = random.choice(outfits[oid_B])
        neg_pairs.append((iA, iB))

    print(f"  [{split_name}] Negative pairs: {len(neg_pairs):,}  "
          f"(hard {sum(1 for _ in neg_pairs[:n_hard]):,} / "
          f"easy {len(neg_pairs)-n_hard:,})")

    # ── Assemble ──────────────────────────────────────────
    records = (
        [(1, iA, iB) for iA, iB in pos_pairs] +
        [(0, iA, iB) for iA, iB in neg_pairs]
    )
    random.shuffle(records)

    labels    = torch.tensor([r[0] for r in records], dtype=torch.float32)
    item_A    = torch.from_numpy(np.stack([X_items[r[1]] for r in records]))
    item_B    = torch.from_numpy(np.stack([X_items[r[2]] for r in records]))
    meta_feat = torch.from_numpy(
        np.stack([meta_pair(r[1], r[2]) for r in records])
    )

    out_path = os.path.join(BASE_DIR, f"poly_pairs_{split_name}.pt")
    torch.save({
        "item_A":    item_A,
        "item_B":    item_B,
        "meta_pair": meta_feat,
        "labels":    labels,
    }, out_path)
    print(f"  [{split_name}] Saved: {out_path}  {item_A.shape}")
    return len(records)


# ─────────────────────────────────────────────────────────
# Generate train / val
# ─────────────────────────────────────────────────────────
n_train = generate_pairs("train", "train_no_dup.json")
n_val   = generate_pairs("val",   "valid_no_dup.json")


# ─────────────────────────────────────────────────────────
# Test set: official fashion_compatibility_prediction.txt
# ─────────────────────────────────────────────────────────
print(f"\n  [test] Building from fashion_compatibility_prediction.txt ...")
with open(os.path.join(BASE_DIR, "poly_compat_pairs.pkl"), "rb") as f:
    compat_pairs = pickle.load(f)

test_records = []
for label, item_ids in compat_pairs:
    idxs = [item_id_to_idx[iid] for iid in item_ids if iid in item_id_to_idx]
    if len(idxs) < 2:
        continue
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            test_records.append((label, idxs[i], idxs[j]))

labels    = torch.tensor([r[0] for r in test_records], dtype=torch.float32)
item_A    = torch.from_numpy(np.stack([X_items[r[1]] for r in test_records]))
item_B    = torch.from_numpy(np.stack([X_items[r[2]] for r in test_records]))
meta_feat = torch.from_numpy(
    np.stack([meta_pair(r[1], r[2]) for r in test_records])
)
out_path = os.path.join(BASE_DIR, "poly_pairs_test.pt")
torch.save({
    "item_A": item_A, "item_B": item_B,
    "meta_pair": meta_feat, "labels": labels,
}, out_path)
pos_t = int(labels.sum())
print(f"  [test] Saved: {out_path}  {item_A.shape}  "
      f"(pos {pos_t:,} / neg {len(test_records)-pos_t:,})")
n_test = len(test_records)


# ─────────────────────────────────────────────────────────
# Save split info
# ─────────────────────────────────────────────────────────
info = {
    "D_ITEM":      D_ITEM,
    "D_META_PAIR": 5,
    "meta_pair_cols": ["category_match","color_match",
                       "tpo_overlap","price_gap","popularity_gap"],
    "split_sizes": {"train": n_train, "val": n_val, "test": n_test},
}
with open(os.path.join(BASE_DIR, "pair_split_info.pkl"), "wb") as f:
    pickle.dump(info, f)

print()
print("=" * 60)
print("STEP 5 COMPLETE")
print("=" * 60)
print(f"  D_ITEM = {D_ITEM}  (tfidf_300 + bert_768 + struct_63)")
print(f"  Train : {n_train:,} pairs")
print(f"  Val   : {n_val:,} pairs")
print(f"  Test  : {n_test:,} pairs (official benchmark)")
print()
print("  Next -> Step 6: Stage 1 Pairwise Fusion MLP training")
