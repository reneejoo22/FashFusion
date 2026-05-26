"""
H&M full dataset download via Kaggle API (archive.zip)
Downloads the full competition archive and extracts images/ only.
Total: ~34 GB
"""

import os, time, requests, zipfile

TOKEN   = 'KGAT_8218c6f46f6d683fba474cbc43cc97c0'
COMP    = 'h-and-m-personalized-fashion-recommendations'
HEADERS = {'Authorization': f'Bearer {TOKEN}'}

BASE_DIR  = r'C:\Users\hash0\OneDrive\바탕 화면\stt\bigadata_2026'
ZIP_PATH  = os.path.join(BASE_DIR, 'hnm_archive.zip')
IMG_DEST  = os.path.join(BASE_DIR, 'data', 'hnm', 'images')
os.makedirs(IMG_DEST, exist_ok=True)

CHUNK = 1024 * 1024   # 1 MB chunks

# ── Step 1: Get redirect URL ──────────────────────────
print("Getting download URL ...")
r = requests.get(
    f'https://www.kaggle.com/api/v1/competitions/data/download-all/{COMP}',
    headers=HEADERS, allow_redirects=False, timeout=30
)
if r.status_code not in (301, 302, 303, 307, 308):
    print(f"ERROR: Expected redirect, got {r.status_code}: {r.text[:200]}")
    exit(1)

download_url = r.headers['Location']
print(f"  Got GCS URL (length={len(download_url)})")

# ── Step 2: Download zip ──────────────────────────────
print(f"\nDownloading archive to {ZIP_PATH} ...")
t0 = time.time()

# Check if partially downloaded already
start_byte = 0
if os.path.exists(ZIP_PATH):
    start_byte = os.path.getsize(ZIP_PATH)
    print(f"  Resuming from {start_byte/1e9:.2f} GB")

# HEAD to get total size
head = requests.head(download_url, timeout=30)
total_bytes = int(head.headers.get('Content-Length', 0))
print(f"  Total size: {total_bytes/1e9:.2f} GB")

if start_byte >= total_bytes and total_bytes > 0:
    print("  Already fully downloaded!")
else:
    headers = {}
    if start_byte > 0:
        headers['Range'] = f'bytes={start_byte}-'

    dl_r = requests.get(download_url, headers=headers, stream=True, timeout=60)
    mode = 'ab' if start_byte > 0 else 'wb'
    downloaded = start_byte

    with open(ZIP_PATH, mode) as f:
        for chunk in dl_r.iter_content(CHUNK):
            f.write(chunk)
            downloaded += len(chunk)
            elapsed = time.time() - t0
            speed   = (downloaded - start_byte) / elapsed / 1e6  # MB/s
            pct     = downloaded / total_bytes * 100 if total_bytes else 0
            eta_min = (total_bytes - downloaded) / ((downloaded - start_byte) / elapsed) / 60 if downloaded > start_byte else 0
            if int(elapsed) % 30 == 0:
                print(f"  {downloaded/1e9:.2f}/{total_bytes/1e9:.2f} GB  "
                      f"({pct:.1f}%)  {speed:.1f} MB/s  ETA {eta_min:.0f}min", flush=True)

    elapsed = time.time() - t0
    print(f"\nDownload complete in {elapsed/60:.1f} min  "
          f"({os.path.getsize(ZIP_PATH)/1e9:.2f} GB)")

# ── Step 3: Extract images/ only ─────────────────────
print(f"\nExtracting images/ from zip ...")
t1 = time.time()
extracted = 0
skipped   = 0

with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    names = [n for n in zf.namelist() if n.startswith('images/') and n.endswith('.jpg')]
    total = len(names)
    print(f"  Found {total:,} image files in zip")

    for i, name in enumerate(names):
        # name = "images/010/0108775015.jpg"
        parts   = name.split('/')          # ['images', '010', '0108775015.jpg']
        subdir  = parts[1]
        fname   = parts[2]
        dest    = os.path.join(IMG_DEST, subdir, fname)

        if os.path.exists(dest):
            skipped += 1
            continue

        os.makedirs(os.path.join(IMG_DEST, subdir), exist_ok=True)
        with zf.open(name) as src, open(dest, 'wb') as dst:
            dst.write(src.read())
        extracted += 1

        if (i+1) % 5000 == 0:
            elapsed2 = time.time() - t1
            speed2   = (i+1) / elapsed2
            eta2     = (total - i - 1) / speed2 / 60
            print(f"  {i+1:>6,}/{total:,}  extracted={extracted:,}  "
                  f"{speed2:.0f}/s  ETA {eta2:.0f}min")

elapsed2 = time.time() - t1
print(f"\nExtraction done in {elapsed2/60:.1f} min")
print(f"  Extracted: {extracted:,}")
print(f"  Skipped  : {skipped:,} (already existed)")
print(f"\nImages ready at: {IMG_DEST}")
print("  Next -> python step3b_hnm_mobilenet.py")
