"""
Step 3c: Rebuild pair datasets with multimodal feature indices
  Strategy: store (idx_a, idx_b, labels, meta, p_compat) only.
  Full feature matrices (poly_mm, hnm_mm: 2411d) loaded at training time.
  Avoids ZIP64 issues with >4GB torch files.

  Output (small files, index-based):
    poly_pairs_mm_train.pt  ~ 20 MB each
    poly_pairs_mm_val.pt
    poly_pairs_mm_test.pt
    hnm_pairs_mm_train.pt
    hnm_pairs_mm_val.pt
    poly_mm_feats.npy  (142480 x 2411, 1.37 GB)
    hnm_mm_feats.npy   (105542 x 2411, 1.02 GB)
    mm_pair_split_info.pkl
"""

import os, pickle, time
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
t_start  = time.time()

# ── Load feature matrices ──────────────────────────────
print("Loading feature matrices ...")

poly_tfidf  = np.load(os.path.join(BASE_DIR, "poly_tfidf_300.npy"))
poly_bert   = np.load(os.path.join(BASE_DIR, "poly_bert_768.npy"))
poly_struct = np.load(os.path.join(BASE_DIR, "poly_struct.npy"))
poly_img    = np.load(os.path.join(BASE_DIR, "poly_img_1280.npy"))

hnm_tfidf   = np.load(os.path.join(BASE_DIR, "hnm_tfidf_300.npy"))
hnm_bert    = np.load(os.path.join(BASE_DIR, "hnm_bert_768.npy"))
hnm_struct  = np.load(os.path.join(BASE_DIR, "hnm_struct_compat63.npy"))
hnm_img     = np.load(os.path.join(BASE_DIR, "hnm_img_1280.npy"))

print(f"  Loaded in {time.time()-t_start:.1f}s")

# ── Build full 2411d feature matrices ─────────────────
poly_mm = np.concatenate([poly_tfidf, poly_bert, poly_struct, poly_img], axis=1).astype(np.float32)
hnm_mm  = np.concatenate([hnm_tfidf,  hnm_bert,  hnm_struct,  hnm_img],  axis=1).astype(np.float32)
print(f"  poly_mm: {poly_mm.shape}  hnm_mm: {hnm_mm.shape}")

# Save consolidated feature matrices as .npy (no ZIP64 issue)
poly_mm_path = os.path.join(BASE_DIR, "poly_mm_feats.npy")
hnm_mm_path  = os.path.join(BASE_DIR, "hnm_mm_feats.npy")
# NOTE: 기존 파일이 있으면 덮어쓰지 않음 (1GB+ 파일 재저장 방지).
# step1~3b 소스 피처를 재생성한 경우 아래 파일을 수동으로 삭제 후 재실행 필요.
if not os.path.exists(poly_mm_path):
    np.save(poly_mm_path, poly_mm)
    print(f"  Saved poly_mm_feats.npy  ({poly_mm.nbytes/1e9:.2f} GB)")
else:
    print(f"  poly_mm_feats.npy already exists, skipping save (delete to regenerate)")
if not os.path.exists(hnm_mm_path):
    np.save(hnm_mm_path, hnm_mm)
    print(f"  Saved hnm_mm_feats.npy   ({hnm_mm.nbytes/1e9:.2f} GB)")
else:
    print(f"  hnm_mm_feats.npy already exists, skipping save (delete to regenerate)")

# ── Build reverse index lookup ─────────────────────────
print("\nBuilding reverse lookup tables ...")

# Old text features (1131d) are stored in pair files as item_A/item_B
poly_text = np.concatenate([poly_tfidf, poly_bert, poly_struct], axis=1)  # (142480, 1131)
hnm_text  = np.concatenate([hnm_tfidf,  hnm_bert,  hnm_struct],  axis=1)  # (105542, 1131)

def build_lookup(text_mat):
    lookup = {}
    for i in range(text_mat.shape[0]):
        key = text_mat[i].tobytes()
        lookup[key] = i
    return lookup

poly_lookup = build_lookup(poly_text)
hnm_lookup  = build_lookup(hnm_text)

n_poly_unique = len(poly_lookup)
n_hnm_unique  = len(hnm_lookup)
print(f"  poly_lookup: {n_poly_unique:,}/{poly_text.shape[0]:,}  "
      f"hnm_lookup: {n_hnm_unique:,}/{hnm_text.shape[0]:,}")

def vecs_to_idxs(feat_np, lookup):
    return np.array([lookup.get(feat_np[i].tobytes(), -1)
                     for i in range(feat_np.shape[0])], dtype=np.int64)

# ── Rebuild Polyvore pair index files ─────────────────
print("\nRebuilding Polyvore multimodal pair index files ...")

for split in ['train', 'val', 'test']:
    src  = os.path.join(BASE_DIR, f"poly_pairs_{split}.pt")
    dst  = os.path.join(BASE_DIR, f"poly_pairs_mm_{split}.pt")
    t0   = time.time()
    data = torch.load(src, weights_only=False)

    idx_a = vecs_to_idxs(data['item_A'].numpy(), poly_lookup)
    idx_b = vecs_to_idxs(data['item_B'].numpy(), poly_lookup)

    miss = (idx_a == -1).sum() + (idx_b == -1).sum()
    if miss > 0:
        print(f"  WARNING {split}: {miss} unmatched items (will use idx=0)")
        idx_a[idx_a == -1] = 0
        idx_b[idx_b == -1] = 0

    out = {
        'idx_a'    : torch.from_numpy(idx_a),
        'idx_b'    : torch.from_numpy(idx_b),
        'labels'   : data['labels'],
        'meta_pair': data['meta_pair'],
    }
    torch.save(out, dst)

    n   = len(data['labels'])
    pos = data['labels'].sum().item()
    sz  = os.path.getsize(dst) / 1e6
    print(f"  {split:5s}: {n:>8,} pairs  pos={int(pos):,} ({pos/n*100:.1f}%)  "
          f"file={sz:.1f}MB  [{time.time()-t0:.1f}s]")

# ── Rebuild H&M pair index files ───────────────────────
print("\nRebuilding H&M multimodal pair index files ...")

for split in ['train', 'val']:
    src  = os.path.join(BASE_DIR, f"hnm_pairs_{split}.pt")
    dst  = os.path.join(BASE_DIR, f"hnm_pairs_mm_{split}.pt")
    t0   = time.time()
    data = torch.load(src, weights_only=False)

    idx_a = vecs_to_idxs(data['item_A'].numpy(), hnm_lookup)
    idx_b = vecs_to_idxs(data['item_B'].numpy(), hnm_lookup)

    miss = (idx_a == -1).sum() + (idx_b == -1).sum()
    if miss > 0:
        print(f"  WARNING {split}: {miss} unmatched items (will use idx=0)")
        idx_a[idx_a == -1] = 0
        idx_b[idx_b == -1] = 0

    out = {
        'idx_a'      : torch.from_numpy(idx_a),
        'idx_b'      : torch.from_numpy(idx_b),
        'labels'     : data['labels'],
        'meta_pair'  : data['meta_pair'],
        'p_compatible': data['p_compatible'],
    }
    torch.save(out, dst)

    n   = len(data['labels'])
    pos = data['labels'].sum().item()
    sz  = os.path.getsize(dst) / 1e6
    print(f"  {split:5s}: {n:>8,} pairs  pos={int(pos):,} ({pos/n*100:.1f}%)  "
          f"file={sz:.1f}MB  [{time.time()-t0:.1f}s]")

# ── Save metadata ──────────────────────────────────────
info = {
    'D_ITEM'         : 2411,
    'D_META_PAIR'    : 5,
    'D_PROJ'         : 256,
    'D_PAIR_S1'      : 256*4 + 1 + 5,
    'D_PAIR_S2'      : 256*4 + 1 + 5 + 1,
    'feature_layout' : 'tfidf_300 + bert_768 + struct_63 + img_1280',
    'poly_feats_file': 'poly_mm_feats.npy',
    'hnm_feats_file' : 'hnm_mm_feats.npy',
}
with open(os.path.join(BASE_DIR, "mm_pair_split_info.pkl"), 'wb') as f:
    pickle.dump(info, f)

print(f"\nDone in {time.time()-t_start:.1f}s")
print(f"  Pair files store indices only (small, no ZIP64 issue)")
print(f"  Feature matrices: poly_mm_feats.npy / hnm_mm_feats.npy")
print("  Next -> step6mm_stage1_train.py")
