from datasets import load_from_disk
from itertools import combinations
import pandas as pd
import numpy as np
import random, re, torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.dummy import DummyClassifier
import joblib, os

# ================================================
# 1. 데이터 로드
# ================================================
print("[1] 데이터 로드")

dataset = load_from_disk("data/polyvore")
df = dataset["data"].to_pandas()

# outfit_id 추출
df["outfit_id"] = df["item_ID"].str.split("_").str[0]

print(f"전체 아이템 수: {len(df)}")
print(f"전체 outfit 수: {df['outfit_id'].nunique()}")

# 빠른 사이클: outfit 3000개만 샘플
OUTFIT_SAMPLE = 3000
sampled_outfits = df["outfit_id"].unique()[:OUTFIT_SAMPLE]
df = df[df["outfit_id"].isin(sampled_outfits)].reset_index(drop=True)
print(f"샘플 후 아이템 수: {len(df)}")

# ================================================
# 2. 전처리
# ================================================
print("\n[2] 전처리")

df["text"]     = df["text"].fillna("")
df["category"] = df["category"].fillna("")
df = df.drop_duplicates(subset=["item_ID"]).reset_index(drop=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["synthetic_text"] = (
    df["text"] + " " + df["category"]
).apply(clean_text)

print(f"전처리 완료: {len(df)}행")

# ================================================
# 3. 이미지 벡터 추출 (MobileNet)
# ================================================
print("\n[3] 이미지 벡터 추출 (MobileNet)")

mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.classifier = torch.nn.Identity()
for param in mobilenet.parameters():
    param.requires_grad = False
mobilenet.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_image_vector(pil_image):
    try:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        tensor = preprocess(pil_image).unsqueeze(0)
        with torch.no_grad():
            vec = mobilenet(tensor)
        return vec.squeeze().numpy()
    except:
        return np.zeros(1280)

image_vectors = []
for i, row in df.iterrows():
    vec = extract_image_vector(row["image"])
    image_vectors.append(vec)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(df)} 완료")

image_vectors = np.array(image_vectors)
print(f"이미지 벡터 shape: {image_vectors.shape}")

# ================================================
# 4. 텍스트 벡터 추출 (TF-IDF)
# ================================================
print("\n[4] TF-IDF 벡터 추출")

vectorizer = TfidfVectorizer(max_features=512)
tfidf_matrix = vectorizer.fit_transform(df["synthetic_text"]).toarray()
print(f"TF-IDF 벡터 shape: {tfidf_matrix.shape}")

# ================================================
# 5. outfit 기반 레이블 생성
# ================================================
print("\n[5] outfit 기반 레이블 생성")

# 인덱스 매핑 (item_ID → df 행 번호)
id_to_idx = {item_id: idx for idx, item_id in enumerate(df["item_ID"])}

positive_pairs = []
negative_pairs = []

outfit_groups = df.groupby("outfit_id")["item_ID"].apply(list)

for outfit_id, item_ids in outfit_groups.items():
    # Positive: 같은 outfit 내 모든 아이템 쌍 조합
    for id_a, id_b in combinations(item_ids, 2):
        if id_a in id_to_idx and id_b in id_to_idx:
            positive_pairs.append((
                id_to_idx[id_a],
                id_to_idx[id_b],
                1
            ))

# Negative: Positive 수만큼 다른 outfit 아이템 쌍 랜덤 생성
all_ids = df["item_ID"].tolist()
n_neg = len(positive_pairs)

while len(negative_pairs) < n_neg:
    id_a = random.choice(all_ids)
    id_b = random.choice(all_ids)

    # 다른 outfit 아이템인지 확인
    outfit_a = id_a.split("_")[0]
    outfit_b = id_b.split("_")[0]

    if outfit_a != outfit_b and id_a in id_to_idx and id_b in id_to_idx:
        negative_pairs.append((
            id_to_idx[id_a],
            id_to_idx[id_b],
            0
        ))

all_pairs = positive_pairs + negative_pairs
random.shuffle(all_pairs)

print(f"Positive 쌍: {len(positive_pairs)}")
print(f"Negative 쌍: {len(negative_pairs)}")
print(f"총 쌍 수:    {len(all_pairs)}")

# ================================================
# 6. 피처 행렬 구성
# ================================================
print("\n[6] 피처 행렬 구성")

# TF-IDF(A) + TF-IDF(B) + 이미지(A) + 이미지(B)
# 512 + 512 + 1280 + 1280 = 3,584d

X_list, y_list = [], []
for i, j, label in all_pairs:
    feat = np.concatenate([
        tfidf_matrix[i],
        tfidf_matrix[j],
        image_vectors[i],
        image_vectors[j],
    ])
    X_list.append(feat)
    y_list.append(label)

X = np.array(X_list)
y = np.array(y_list)

print(f"피처 shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape} / Test: {X_test.shape}")

# ================================================
# 7. 모델 학습
# ================================================
print("\n[7] 모델 학습")

models_to_try = {
    "Random Classifier":   DummyClassifier(strategy="uniform", random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "MLP Classifier":      MLPClassifier(
                               hidden_layer_sizes=(256, 128),
                               max_iter=100,
                               random_state=42,
                               early_stopping=True,
                           ),
}

results = {}
trained_models = {}

for name, model in models_to_try.items():
    print(f"\n  [{name}] 학습 중...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    results[name] = f1
    trained_models[name] = model
    print(classification_report(y_test, y_pred,
                                target_names=["Not Compatible", "Compatible"]))

# ================================================
# 8. 결과 출력
# ================================================
print("\n" + "="*50)
print("Phase 1 결과 (outfit 기반 스타일 학습)")
print("="*50)
for name, score in sorted(results.items(), key=lambda x: -x[1]):
    bar = "█" * int(score * 30)
    print(f"  {name:22s} | {bar:30s} | F1: {score:.4f}")

best_name  = max(results, key=results.get)
best_model = trained_models[best_name]
print(f"\n최고 성능: {best_name}  (F1: {results[best_name]:.4f})")

# ================================================
# 9. 모델 저장 → Phase 2 (H&M)에서 재사용
# ================================================
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/phase1_compatibility_model.pkl")
joblib.dump(vectorizer, "models/phase1_tfidf_vectorizer.pkl")
np.save("models/phase1_image_vectors.npy", image_vectors)

# item_ID → 인덱스 매핑도 저장
import json
with open("models/phase1_id_to_idx.json", "w") as f:
    json.dump(id_to_idx, f)

print("\n저장 완료:")
print("  models/phase1_compatibility_model.pkl")
print("  models/phase1_tfidf_vectorizer.pkl")
print("  models/phase1_image_vectors.npy")
print("  models/phase1_id_to_idx.json")
print("\n→ H&M Phase 2에서 이 모델로 compatibility score 추출 예정")