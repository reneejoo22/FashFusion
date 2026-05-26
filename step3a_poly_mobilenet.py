"""
Step 3a: MobileNet image features - Polyvore
  Input : data/polyvore/images/images/{tid}.jpg  (27,372 / 142,480 items)
  Output: poly_img_1280.npy  (142480 x 1280, float32)
          Items without images -> zero vector (filled later or left as zero)

  MobileNetV3-Large: pretrained ImageNet, global average pool -> 1280d
  Frozen feature extractor (no fine-tuning)
"""

import os, re, json, pickle, time
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR  = os.path.join(BASE_DIR, "data", "polyvore", "images", "images")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256
D_IMG      = 1280

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────
# Build item_id -> image path mapping
# ─────────────────────────────────────────────────────────
print("\nBuilding item -> image path mapping ...")

with open(os.path.join(BASE_DIR, "item_id_list.pkl"), "rb") as f:
    item_id_list = pickle.load(f)

img_files = {os.path.splitext(f)[0]: os.path.join(IMG_DIR, f)
             for f in os.listdir(IMG_DIR) if f.endswith(".jpg")}

# Build item_id -> tid from raw JSON
item_tid_map = {}
for split in ["train_no_dup.json", "valid_no_dup.json", "test_no_dup.json"]:
    with open(os.path.join(BASE_DIR, "data", "polyvore", split),
              "r", encoding="utf-8") as f:
        data = json.load(f)
    for outfit in data:
        sid = str(outfit["set_id"])
        for item in outfit["items"]:
            iid = f"{sid}_{item['index']}"
            m = re.search(r"tid=(\d+)", item.get("image", ""))
            if m:
                item_tid_map[iid] = m.group(1)

# item_id -> img_path (only for items that have an image file)
item_img_path = {}
for iid, tid in item_tid_map.items():
    if tid in img_files:
        item_img_path[iid] = img_files[tid]

print(f"  Total items     : {len(item_id_list):,}")
print(f"  Items with image: {len(item_img_path):,} ({len(item_img_path)/len(item_id_list)*100:.1f}%)")

# Items that need extraction (in item_id_list order)
items_to_extract = [(i, iid, item_img_path[iid])
                    for i, iid in enumerate(item_id_list)
                    if iid in item_img_path]
print(f"  To extract: {len(items_to_extract):,}")


# ─────────────────────────────────────────────────────────
# MobileNetV3-Large feature extractor
# ─────────────────────────────────────────────────────────
print("\nLoading MobileNetV3-Large ...")
backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
# features+avgpool -> 960d, classifier[0]+[1] (Linear 960->1280 + Hardswish) -> 1280d
feature_extractor = torch.nn.Sequential(
    backbone.features,
    backbone.avgpool,
    torch.nn.Flatten(1),
    backbone.classifier[0],   # Linear(960, 1280)
    backbone.classifier[1],   # Hardswish
)
feature_extractor.eval().to(DEVICE)

# Freeze all parameters
for p in feature_extractor.parameters():
    p.requires_grad = False

print(f"  Params: {sum(p.numel() for p in feature_extractor.parameters()):,} (frozen)")

# ImageNet normalization
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────
class PolyImgDataset(Dataset):
    def __init__(self, records, transform):
        self.records   = records    # [(row_idx, iid, img_path), ...]
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row_idx, _, path = self.records[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, 224, 224)
        return row_idx, img


# ─────────────────────────────────────────────────────────
# Extract features
# ─────────────────────────────────────────────────────────
out_path = os.path.join(BASE_DIR, "poly_img_1280.npy")

# Init output array with zeros (items without images stay as zero)
poly_img = np.zeros((len(item_id_list), D_IMG), dtype=np.float32)

dataset = PolyImgDataset(items_to_extract, transform)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=0, pin_memory=False)

print(f"\nExtracting {len(items_to_extract):,} image features ...")
t0 = time.time()

with torch.no_grad():
    for i, (row_idxs, imgs) in enumerate(loader):
        imgs   = imgs.to(DEVICE)
        feats  = feature_extractor(imgs).cpu().float().numpy()

        # L2 normalize
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        feats = feats / norms

        for j, ridx in enumerate(row_idxs.numpy()):
            poly_img[ridx] = feats[j]

        if (i+1) % 20 == 0:
            done  = min((i+1)*BATCH_SIZE, len(items_to_extract))
            speed = done / (time.time()-t0)
            eta   = (len(items_to_extract)-done)/speed if speed>0 else 0
            print(f"  {done:,}/{len(items_to_extract):,}  "
                  f"{speed:.0f} imgs/s  ETA {eta:.0f}s")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s  ({len(items_to_extract)/elapsed:.0f} imgs/s)")

# Save
np.save(out_path, poly_img)
print(f"Saved: {out_path}  shape={poly_img.shape}  "
      f"({poly_img.nbytes/1e6:.1f} MB)")

# Stats
has_img  = (poly_img.sum(axis=1) != 0).sum()
zero_img = (poly_img.sum(axis=1) == 0).sum()
print(f"  Non-zero rows (has image): {has_img:,}")
print(f"  Zero rows (no image)     : {zero_img:,}")
print()
print("  Next -> Step 3b: H&M image features (after Kaggle download)")
