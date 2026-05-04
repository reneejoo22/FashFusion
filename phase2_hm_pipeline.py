# ================================================
# FashFusion - Phase 2 (H&M 구매 확률 예측)
# ================================================
# 실행 전 필요:
#   - Phase 1 완료 (models/phase1_*.pkl 파일 존재)
#   - pip install datasets torch torchvision scikit-learn pandas numpy Pillow joblib
# ================================================

import os, re, random, json
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.dummy import DummyClassifier
from itertools import combinations
import joblib

# ================================================
# 1. H&M 데이터 로드 + 샘플 추출
# ================================================
print("\n" + "="*50)
print("[1] H&M 데이터 로드")
print("="*50)

articles     = pd.read_csv("data/hnm/articles.csv")
transactions = pd.read_csv("data/hnm/transactions_train.csv",
                           parse_dates=["t_dat"])

print(f"articles:     {articles.shape}")
print(f"transactions: {transactions.shape}")

# ── 샘플링 전략 ──────────────────────────────
# transactions 3천만개를 다 쓰면 너무 오래 걸림
# → 최근 1개월치 거래만 사용 (빠른 사이클)
RECENT_DAYS = 30
max_date  = transactions["t_dat"].max()
min_date  = max_date - pd.Timedelta(days=RECENT_DAYS)
trans_sample = transactions[transactions["t_dat"] >= min_date].copy()

print(f"\n최근 {RECENT_DAYS}일 거래: {len(trans_sample)}건")
print(f"기간: {min_date.date()} ~ {max_date.date()}")

# 거래에 등장한 article만 articles에서 추출
used_articles = trans_sample["article_id"].unique()
articles_sample = articles[
    articles["article_id"].isin(used_articles)
].reset_index(drop=True)

print(f"사용 상품 수: {len(articles_sample)}")

# ================================================
# 2. 전처리 — Synthetic Metadata Text 생성
# ================================================
print("\n" + "="*50)
print("[2] 전처리")
print("="*50)

# 결측값 처리
text_cols = ["prod_name", "product_type_name", "product_group_name",
             "colour_group_name", "garment_group_name",
             "section_name", "detail_desc"]

for col in text_cols:
    articles_sample[col] = articles_sample[col].fillna("")

# 중복 제거
before = len(articles_sample)
articles_sample = articles_sample.drop_duplicates(
    subset=["article_id"]
).reset_index(drop=True)
print(f"중복 제거: {before} → {len(articles_sample)}행")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Synthetic metadata text 생성
# 상품명 + 타입 + 그룹 + 색상 + 의류군 + 섹션 + 상세설명
articles_sample["synthetic_text"] = (
    articles_sample["prod_name"]          + " " +
    articles_sample["product_type_name"]  + " " +
    articles_sample["product_group_name"] + " " +
    articles_sample["colour_group_name"]  + " " +
    articles_sample["garment_group_name"] + " " +
    articles_sample["section_name"]       + " " +
    articles_sample["detail_desc"]
).apply(clean_text)

print("\n[샘플 synthetic_text 3개]")
for t in articles_sample["synthetic_text"].head(3):
    print(f"  → {t[:100]}")

# article_id → 인덱스 매핑
art_id_to_idx = {
    str(art_id): idx
    for idx, art_id in enumerate(articles_sample["article_id"])
}

# ================================================
# 3. 이미지 벡터 추출 (MobileNet)
# ================================================
print("\n" + "="*50)
print("[3] 이미지 벡터 추출 (MobileNet — Frozen)")
print("="*50)

# H&M 이미지 경로: data/hm/images/{article_id 앞3자리}/{article_id}.jpg
# 예: article_id=108775015 → images/108/0108775015.jpg

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

from PIL import Image

def get_hm_image_path(article_id):
    """H&M 이미지 파일 경로 생성"""
    art_str = str(article_id).zfill(10)   # 10자리로 패딩
    folder  = art_str[:3]                  # 앞 3자리가 폴더명
    return f"data/hm/images/{folder}/0{art_str}.jpg"

def extract_image_vector(article_id):
    try:
        img_path = get_hm_image_path(article_id)
        if not os.path.exists(img_path):
            return np.zeros(1280)
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            vec = mobilenet(tensor)
        return vec.squeeze().numpy()
    except:
        return np.zeros(1280)

image_vectors = []
missing = 0
for i, row in articles_sample.iterrows():
    vec = extract_image_vector(row["article_id"])
    if vec.sum() == 0:
        missing += 1
    image_vectors.append(vec)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(articles_sample)} 완료")

image_vectors = np.array(image_vectors)
print(f"\n이미지 벡터 shape: {image_vectors.shape}")
print(f"이미지 없는 상품: {missing}개 (영벡터로 대체)")

# ================================================
# 4. TF-IDF 텍스트 벡터 추출
# ================================================
print("\n" + "="*50)
print("[4] TF-IDF 텍스트 벡터 추출")
print("="*50)

# Phase 1 vectorizer 재사용 (같은 어휘 공간 유지)
try:
    vectorizer = joblib.load("models/phase1_tfidf_vectorizer.pkl")
    tfidf_matrix = vectorizer.transform(
        articles_sample["synthetic_text"]
    ).toarray()
    print("Phase 1 TF-IDF vectorizer 재사용")
except:
    # Phase 1 모델 없으면 새로 학습
    vectorizer = TfidfVectorizer(max_features=512)
    tfidf_matrix = vectorizer.fit_transform(
        articles_sample["synthetic_text"]
    ).toarray()
    print("새 TF-IDF vectorizer 학습")

print(f"TF-IDF 벡터 shape: {tfidf_matrix.shape}")

# ================================================
# 5. Phase 1 compatibility score 추출
# ================================================
print("\n" + "="*50)
print("[5] Phase 1 compatibility score 로드")
print("="*50)

try:
    phase1_model = joblib.load("models/phase1_compatibility_model.pkl")
    PHASE1_AVAILABLE = True
    print("Phase 1 모델 로드 성공")
except:
    PHASE1_AVAILABLE = False
    print("Phase 1 모델 없음 → compatibility score 미사용")

def get_compatibility_score(idx_a, idx_b):
    """두 아이템의 스타일 호환성 점수 (Phase 1 모델)"""
    if not PHASE1_AVAILABLE:
        return 0.5   # 기본값
    feat = np.concatenate([
        tfidf_matrix[idx_a], tfidf_matrix[idx_b],
        image_vectors[idx_a], image_vectors[idx_b],
    ]).reshape(1, -1)
    return phase1_model.predict_proba(feat)[0][1]

# ================================================
# 6. Co-purchase 레이블 생성
# ================================================
print("\n" + "="*50)
print("[6] Co-purchase 레이블 생성")
print("="*50)

# 같은 고객이 같은 날 구매한 아이템 쌍 → Positive (1)
# 랜덤 쌍                               → Negative (0)

# 고객-날짜별 구매 아이템 목록
co_purchase = trans_sample.groupby(
    ["customer_id", "t_dat"]
)["article_id"].apply(list).reset_index()

# articles_sample에 있는 article_id만 필터링
valid_ids = set(articles_sample["article_id"].astype(str))

positive_pairs = []
for _, row in co_purchase.iterrows():
    items = [str(i) for i in row["article_id"] if str(i) in valid_ids]
    if len(items) >= 2:
        for id_a, id_b in combinations(items, 2):
            if id_a in art_id_to_idx and id_b in art_id_to_idx:
                positive_pairs.append((
                    art_id_to_idx[id_a],
                    art_id_to_idx[id_b],
                    1
                ))

# Positive가 너무 많으면 샘플링
MAX_PAIRS = 20000
if len(positive_pairs) > MAX_PAIRS:
    positive_pairs = random.sample(positive_pairs, MAX_PAIRS)

# Negative: Positive 수만큼 랜덤 쌍 생성
all_ids = list(art_id_to_idx.keys())
negative_pairs = []
while len(negative_pairs) < len(positive_pairs):
    id_a = random.choice(all_ids)
    id_b = random.choice(all_ids)
    if id_a != id_b:
        negative_pairs.append((
            art_id_to_idx[id_a],
            art_id_to_idx[id_b],
            0
        ))

all_pairs = positive_pairs + negative_pairs
random.shuffle(all_pairs)

print(f"Positive 쌍: {len(positive_pairs)}")
print(f"Negative 쌍: {len(negative_pairs)}")
print(f"총 쌍 수:    {len(all_pairs)}")

# ================================================
# 7. 피처 행렬 구성
# ================================================
print("\n" + "="*50)
print("[7] 피처 행렬 구성")
print("="*50)

# 구성:
#   TF-IDF(A)    512d
#   TF-IDF(B)    512d
#   이미지(A)   1280d
#   이미지(B)   1280d
#   compat_score   1d  ← Phase 1에서 온 스타일 점수
# ─────────────────────
#   총             3585d

print("compatibility score 계산 중... (시간이 걸릴 수 있어요)")
X_list, y_list = [], []

for k, (i, j, label) in enumerate(all_pairs):
    compat = get_compatibility_score(i, j)
    feat = np.concatenate([
        tfidf_matrix[i],
        tfidf_matrix[j],
        image_vectors[i],
        image_vectors[j],
        [compat],              # Phase 1 스타일 점수
    ])
    X_list.append(feat)
    y_list.append(label)

    if (k + 1) % 5000 == 0:
        print(f"  {k+1}/{len(all_pairs)} 완료")

X = np.array(X_list)
y = np.array(y_list)

print(f"\n피처 shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape} / Test: {X_test.shape}")

# ================================================
# 8. 모델 학습
# ================================================
print("\n" + "="*50)
print("[8] 모델 학습")
print("="*50)

models_to_try = {
    "Random Classifier":   DummyClassifier(strategy="uniform", random_state=42),
    "Logistic Regression": LogisticRegression(
                               max_iter=1000,
                               random_state=42
                           ),
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
    y_pred  = model.predict(X_test)
    f1      = f1_score(y_test, y_pred)
    results[name] = f1
    trained_models[name] = model
    print(classification_report(y_test, y_pred,
                                target_names=["Not Purchase", "Purchase"]))

# ================================================
# 9. 최종 결과
# ================================================
print("\n" + "="*50)
print("Phase 2 최종 결과 (H&M 구매 확률 예측)")
print("="*50)

for name, score in sorted(results.items(), key=lambda x: -x[1]):
    bar = "█" * int(score * 30)
    print(f"  {name:22s} | {bar:30s} | F1: {score:.4f}")

best_name  = max(results, key=results.get)
best_model = trained_models[best_name]
print(f"\n최고 성능: {best_name}  (F1: {results[best_name]:.4f})")

# ================================================
# 10. 모델 저장 + 예측 함수
# ================================================
os.makedirs("models", exist_ok=True)

joblib.dump(best_model,  "models/phase2_purchase_model.pkl")
joblib.dump(vectorizer,  "models/phase2_tfidf_vectorizer.pkl")
np.save("models/phase2_image_vectors.npy", image_vectors)

print("\n저장 완료:")
print("  models/phase2_purchase_model.pkl")
print("  models/phase2_tfidf_vectorizer.pkl")
print("  models/phase2_image_vectors.npy")

# ── 최종 예측 함수 ────────────────────────────
def predict_purchase_probability(article_id_a, article_id_b):
    """
    두 상품의 구매 확률 예측
    반환값: 0.0 ~ 1.0 (0.85 이상이면 우선 노출)
    """
    if (str(article_id_a) not in art_id_to_idx or
        str(article_id_b) not in art_id_to_idx):
        return None

    i = art_id_to_idx[str(article_id_a)]
    j = art_id_to_idx[str(article_id_b)]
    compat = get_compatibility_score(i, j)

    feat = np.concatenate([
        tfidf_matrix[i], tfidf_matrix[j],
        image_vectors[i], image_vectors[j],
        [compat]
    ]).reshape(1, -1)

    prob = best_model.predict_proba(feat)[0][1]
    return prob

# 샘플 예측 테스트
print("\n=== 샘플 예측 테스트 ===")
sample_pairs = random.sample(all_pairs[:100], 3)
for i, j, true_label in sample_pairs:
    art_a = articles_sample.iloc[i]["article_id"]
    art_b = articles_sample.iloc[j]["article_id"]
    name_a = articles_sample.iloc[i]["prod_name"]
    name_b = articles_sample.iloc[j]["prod_name"]
    prob = predict_purchase_probability(art_a, art_b)
    print(f"\n  상품 A: {name_a}")
    print(f"  상품 B: {name_b}")
    print(f"  실제 레이블: {'함께 구매됨' if true_label == 1 else '구매 안 됨'}")
    print(f"  예측 구매 확률: {prob:.2%}")
    print(f"  우선 노출 여부: {'노출' if prob >= 0.85 else '미노출'}")
