"""
Step 1: TF-IDF -> TruncatedSVD 300d reduction + L2 normalization
  Inputs:
    poly_item_tfidf.npz          (142480 x 20000)
    hnm_tfidf_features_final.npz (105542 x 20000)
  Outputs:
    poly_tfidf_300.npy           (142480 x 300)  float32
    hnm_tfidf_300.npy            (105542 x 300)  float32
    poly_svd_model.pkl           fitted TruncatedSVD (Polyvore)
    hnm_svd_model.pkl            fitted TruncatedSVD (H&M)

  Note: SVD is fit on train-split items for Polyvore, all articles for H&M.
        L2 normalization applied after projection.
"""

import os
import pickle
import time
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SVD_DIM = 300
RANDOM_STATE = 42

def fit_transform_svd(X_sparse, n_components, name, fit_mask=None):
    """
    Fit TruncatedSVD on X_sparse[fit_mask] and transform full X_sparse.
    fit_mask: boolean array or None (fit on all rows).
    Returns (dense_array float32, svd_model).
    """
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)

    if fit_mask is not None:
        X_fit = X_sparse[fit_mask]
        print(f"  [{name}] Fitting SVD on {X_fit.shape[0]:,} train rows ...")
    else:
        X_fit = X_sparse
        print(f"  [{name}] Fitting SVD on all {X_fit.shape[0]:,} rows ...")

    t0 = time.time()
    svd.fit(X_fit)
    print(f"  [{name}] Fit done in {time.time()-t0:.1f}s")
    print(f"  [{name}] Explained variance ratio (top-5): "
          f"{svd.explained_variance_ratio_[:5].round(4)}")
    print(f"  [{name}] Total explained variance ({n_components}d): "
          f"{svd.explained_variance_ratio_.sum():.4f}")

    t0 = time.time()
    X_dense = svd.transform(X_sparse).astype(np.float32)
    print(f"  [{name}] Transform done in {time.time()-t0:.1f}s  shape={X_dense.shape}")

    # L2 normalize each row
    X_dense = normalize(X_dense, norm="l2").astype(np.float32)
    print(f"  [{name}] L2 normalized. Norm sample (first 3): "
          f"{np.linalg.norm(X_dense[:3], axis=1).round(4)}")

    return X_dense, svd


# ─────────────────────────────────────────────────────────
# 1. POLYVORE ITEMS
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("POLYVORE: TF-IDF -> SVD 300d")
print("=" * 60)

X_poly = sp.load_npz(os.path.join(BASE_DIR, "poly_item_tfidf.npz"))
print(f"  Loaded poly_item_tfidf: {X_poly.shape}")

# Identify train items: those whose set_id is in outfit_id_to_idx (train split)
with open(os.path.join(BASE_DIR, "item_id_list.pkl"), "rb") as f:
    item_id_list = pickle.load(f)

with open(os.path.join(BASE_DIR, "outfit_id_to_idx.pkl"), "rb") as f:
    outfit_id_to_idx = pickle.load(f)

train_set_ids = set(outfit_id_to_idx.keys())  # str set_ids from train split

train_mask = np.array([
    iid.rsplit("_", 1)[0] in train_set_ids
    for iid in item_id_list
], dtype=bool)

print(f"  Train items: {train_mask.sum():,} / {len(item_id_list):,}")

poly_dense, poly_svd = fit_transform_svd(
    X_poly, SVD_DIM, "Polyvore", fit_mask=train_mask
)

poly_out = os.path.join(BASE_DIR, "poly_tfidf_300.npy")
np.save(poly_out, poly_dense)
print(f"  Saved: {poly_out}  ({poly_dense.nbytes / 1e6:.1f} MB)")

poly_svd_path = os.path.join(BASE_DIR, "poly_svd_model.pkl")
with open(poly_svd_path, "wb") as f:
    pickle.dump(poly_svd, f)
print(f"  Saved: {poly_svd_path}")


# ─────────────────────────────────────────────────────────
# 2. H&M ARTICLES
# ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("H&M: TF-IDF -> SVD 300d")
print("=" * 60)

X_hnm = sp.load_npz(os.path.join(BASE_DIR, "hnm_tfidf_features_final.npz"))
print(f"  Loaded hnm_tfidf: {X_hnm.shape}")

hnm_dense, hnm_svd = fit_transform_svd(X_hnm, SVD_DIM, "H&M", fit_mask=None)

hnm_out = os.path.join(BASE_DIR, "hnm_tfidf_300.npy")
np.save(hnm_out, hnm_dense)
print(f"  Saved: {hnm_out}  ({hnm_dense.nbytes / 1e6:.1f} MB)")

hnm_svd_path = os.path.join(BASE_DIR, "hnm_svd_model.pkl")
with open(hnm_svd_path, "wb") as f:
    pickle.dump(hnm_svd, f)
print(f"  Saved: {hnm_svd_path}")


# ─────────────────────────────────────────────────────────
# 3. Sanity check
# ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SANITY CHECK")
print("=" * 60)

# Cosine sim between two items from same outfit should be > random pair
compat_path = os.path.join(BASE_DIR, "poly_compat_pairs.pkl")
with open(compat_path, "rb") as f:
    compat_pairs = pickle.load(f)

with open(os.path.join(BASE_DIR, "item_id_to_idx.pkl"), "rb") as f:
    item_id_to_idx = pickle.load(f)

pos_sims, neg_sims = [], []
rng = np.random.default_rng(42)

for label, item_ids in compat_pairs[:500]:
    idxs = [item_id_to_idx[iid] for iid in item_ids if iid in item_id_to_idx]
    if len(idxs) < 2:
        continue
    # pairwise sim within outfit
    vecs = poly_dense[idxs]
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            s = float(np.dot(vecs[i], vecs[j]))
            if label == 1:
                pos_sims.append(s)
            else:
                neg_sims.append(s)

print(f"  Positive pairs avg cosine sim : {np.mean(pos_sims):.4f} (n={len(pos_sims)})")
print(f"  Negative pairs avg cosine sim : {np.mean(neg_sims):.4f} (n={len(neg_sims)})")
print(f"  Gap (pos - neg)               : {np.mean(pos_sims) - np.mean(neg_sims):.4f}")
print()
print("  Next -> Step 2: DistilBERT embeddings (768d)")
