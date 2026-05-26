import requests, json

TOKEN = 'KGAT_5c5a79307f01b8e7f1fa76c2bf4fc055'
headers = {'Authorization': f'Bearer {TOKEN}'}
COMP = 'h-and-m-personalized-fashion-recommendations'

# List all files (paginate if needed)
all_files = []
page_token = None
while True:
    url = f'https://www.kaggle.com/api/v1/competitions/data/list/{COMP}'
    if page_token:
        url += f'?pageToken={page_token}'
    r = requests.get(url, headers=headers, timeout=30)
    data = r.json()
    all_files.extend(data.get('files', []))
    page_token = data.get('nextPageTokenNullable')
    if not page_token:
        break

print(f"Total files: {len(all_files)}")
for f in all_files:
    name = f['name']
    size_gb = f['totalBytes'] / 1e9
    print(f"  {name:<40}  {size_gb:.2f} GB")

print()
# Test download URL for images.zip
r2 = requests.get(
    f'https://www.kaggle.com/api/v1/competitions/data/download/{COMP}/images.zip',
    headers=headers, allow_redirects=False, timeout=30
)
print(f"images.zip download -> status: {r2.status_code}")
if 'Location' in r2.headers:
    print(f"Redirect: {r2.headers['Location'][:120]}")
