"""
H&M image bulk downloader via Kaggle API
URL pattern: /competitions/data/download/{COMP}/images/{dir}/{fname}
  fname = "0" + str(article_id)  (10 digits)
  dir   = fname[:3]
"""

import os, pickle, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

TOKEN   = 'KGAT_8218c6f46f6d683fba474cbc43cc97c0'
COMP    = 'h-and-m-personalized-fashion-recommendations'
HEADERS = {'Authorization': f'Bearer {TOKEN}'}

BASE_DIR   = r'C:\Users\hash0\OneDrive\바탕 화면\stt\bigadata_2026'
DEST_DIR   = os.path.join(BASE_DIR, 'data', 'hnm', 'images')
os.makedirs(DEST_DIR, exist_ok=True)

WORKERS    = 16     # parallel download threads
CHUNK_SIZE = 8192
RETRY      = 3
DELAY_429  = 5      # seconds to wait on rate limit

# ── Load article IDs ──────────────────────────────────────
with open(os.path.join(BASE_DIR, 'article_id_list.pkl'), 'rb') as f:
    article_ids = pickle.load(f)

print(f"Total articles: {len(article_ids):,}")

# Build download tasks
tasks = []
for aid in article_ids:
    fname  = f"0{aid}.jpg"          # e.g. 0108775015.jpg
    subdir = fname[:3]               # e.g. 010
    dest   = os.path.join(DEST_DIR, subdir, fname)
    if os.path.exists(dest):
        continue                     # skip already downloaded
    url = (f'https://www.kaggle.com/api/v1/competitions/data/download'
           f'/{COMP}/images/{subdir}/{fname}')
    tasks.append((dest, url, subdir))

already = len(article_ids) - len(tasks)
print(f"Already downloaded: {already:,}")
print(f"To download: {len(tasks):,}")

if not tasks:
    print("All images already downloaded!")
    exit(0)

# Pre-create subdirectories
subdirs = {t[2] for t in tasks}
for sd in subdirs:
    os.makedirs(os.path.join(DEST_DIR, sd), exist_ok=True)


# ── Download function ─────────────────────────────────────
def download_one(dest, url):
    for attempt in range(RETRY):
        try:
            r = requests.get(url, headers=HEADERS,
                             stream=True, timeout=20)
            if r.status_code == 200:
                with open(dest, 'wb') as f:
                    for chunk in r.iter_content(CHUNK_SIZE):
                        f.write(chunk)
                return 'ok'
            elif r.status_code == 404:
                return 'missing'
            elif r.status_code == 429:
                time.sleep(DELAY_429 * (attempt + 1))
                continue
            else:
                return f'err_{r.status_code}'
        except Exception as e:
            if attempt < RETRY - 1:
                time.sleep(1)
    return 'failed'


# ── Parallel download with progress ──────────────────────
print(f"\nStarting download with {WORKERS} workers ...")
t0      = time.time()
done    = 0
ok      = 0
missing = 0
errors  = 0

with ThreadPoolExecutor(max_workers=WORKERS) as exe:
    futures = {exe.submit(download_one, dest, url): dest
               for dest, url, _ in tasks}

    for fut in as_completed(futures):
        result = fut.result()
        done  += 1
        if result == 'ok':       ok      += 1
        elif result == 'missing': missing += 1
        else:                    errors  += 1

        if done % 500 == 0 or done == len(tasks):
            elapsed = time.time() - t0
            speed   = done / elapsed
            eta     = (len(tasks) - done) / speed if speed > 0 else 0
            pct     = done / len(tasks) * 100
            print(f"  {done:>6,}/{len(tasks):,} ({pct:5.1f}%)  "
                  f"ok={ok:,} miss={missing:,} err={errors:,}  "
                  f"{speed:.1f}/s  ETA {eta/60:.1f}min")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")
print(f"  Downloaded : {ok:,}")
print(f"  Missing    : {missing:,}  (article has no image on Kaggle)")
print(f"  Errors     : {errors:,}")
print(f"  Total imgs : {ok + already:,} / {len(article_ids):,}")
coverage = (ok + already) / len(article_ids) * 100
print(f"  Coverage   : {coverage:.1f}%")
