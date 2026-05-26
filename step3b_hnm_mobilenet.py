"""
Step 3b: MobileNet image features - H&M
  Input : data/hnm/images/{subdir}/0{article_id}.jpg  (105,542 articles)
  Output: hnm_img_1280.npy  (105542 x 1280, float32)
          Articles without images -> zero vector

  MobileNetV3-Large: pretrained ImageNet, global average pool -> 1280d
  Frozen feature extractor (no fine-tuning)
  Same pipeline as step3a_poly_mobilenet.py
"""

import os, pickle, time
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR  = os.path.join(BASE_DIR, "data", "hnm", "images")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256
D_IMG      = 1280

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------
# Build article_id -> image path mapping
# ---------------------------------------------------
print("\nBuilding article -> image path mapping ...")

with open(os.path.join(BASE_DIR, "article_id_list.pkl"), "rb") as f:
    article_id_list = pickle.load(f)

print(f"  Total articles: {len(article_id_list):,}")

# Scan downloaded images
img_found = {}
if os.path.exists(IMG_DIR):
    for subdir in os.listdir(IMG_DIR):
        subdir_path = os.path.join(IMG_DIR, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for fname in os.listdir(subdir_path):
            if fname.endswith(".jpg"):
                # fname = "0{article_id}.jpg" -> article_id = int(fname[1:-4])
                try:
                    aid = int(fname[1:-4])
                    img_found[aid] = os.path.join(subdir_path, fname)
                except ValueError:
                    pass

print(f"  Images on disk : {len(img_found):,} ({len(img_found)/len(article_id_list)*100:.1f}%)")

# Build extraction list (in article_id_list order)
items_to_extract = [(i, aid, img_found[aid])
                    for i, aid in enumerate(article_id_list)
                    if aid in img_found]
print(f"  To extract     : {len(items_to_extract):,}")

if not items_to_extract:
    print("\nERROR: No images found in", IMG_DIR)
    print("Run kaggle_download.py first to download H&M images.")
    exit(1)

# ---------------------------------------------------
# MobileNetV3-Large feature extractor
# ---------------------------------------------------
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


# ---------------------------------------------------
# Dataset
# ---------------------------------------------------
class HnmImgDataset(Dataset):
    def __init__(self, records, transform):
        self.records   = records    # [(row_idx, aid, img_path), ...]
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


# ---------------------------------------------------
# Extract features
# ---------------------------------------------------
out_path = os.path.join(BASE_DIR, "hnm_img_1280.npy")

# Init output array with zeros (articles without images stay as zero)
hnm_img = np.zeros((len(article_id_list), D_IMG), dtype=np.float32)

dataset = HnmImgDataset(items_to_extract, transform)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=0, pin_memory=False)

print(f"\nExtracting {len(items_to_extract):,} image features ...")
t0 = time.time()

with torch.no_grad():
    for i, (row_idxs, imgs) in enumerate(loader):
        imgs  = imgs.to(DEVICE)
        feats = feature_extractor(imgs).cpu().float().numpy()

        # L2 normalize
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        feats = feats / norms

        for j, ridx in enumerate(row_idxs.numpy()):
            hnm_img[ridx] = feats[j]

        if (i+1) % 20 == 0:
            done  = min((i+1)*BATCH_SIZE, len(items_to_extract))
            speed = done / (time.time()-t0)
            eta   = (len(items_to_extract)-done)/speed if speed>0 else 0
            print(f"  {done:,}/{len(items_to_extract):,}  "
                  f"{speed:.0f} imgs/s  ETA {eta:.0f}s")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s  ({len(items_to_extract)/elapsed:.0f} imgs/s)")

# Save
np.save(out_path, hnm_img)
print(f"Saved: {out_path}  shape={hnm_img.shape}  "
      f"({hnm_img.nbytes/1e6:.1f} MB)")

# Stats
has_img  = (hnm_img.sum(axis=1) != 0).sum()
zero_img = (hnm_img.sum(axis=1) == 0).sum()
print(f"  Non-zero rows (has image): {has_img:,}")
print(f"  Zero rows (no image)     : {zero_img:,}")
print(f"  Coverage                 : {has_img/len(article_id_list)*100:.1f}%")
print()
print("  Next -> Step 3c: Rebuild pair dataset with image features")
