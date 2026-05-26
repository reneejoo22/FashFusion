"""
TF-IDF (+ SVD) vs DistilBERT 임베딩 비교 실험
- 차원: 768 통일
- 데이터: Polyvore (1000 샘플)
- 평가: 코사인 유사도 분포, 클러스터링 품질, Top-K 검색 품질
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import Counter

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_N = 17316   # 전체 사용 (GPU 가능)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("  TF-IDF+SVD vs DistilBERT 임베딩 비교 실험")
print("=" * 60)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1] 데이터 로드 중...")

with open(os.path.join(BASE_DIR, "polyvore_text_features_final.json"), encoding="utf-8") as f:
    json_data = json.load(f)

tpo_df = pd.read_csv(os.path.join(BASE_DIR, "polyvore_tpo_features.csv"))
X_tfidf_full = sp.load_npz(os.path.join(BASE_DIR, "polyvore_tfidf_features_final.npz"))

print(f"  전체 Polyvore outfit 수: {len(json_data)}")
print(f"  TF-IDF 행렬 shape: {X_tfidf_full.shape}")

# TPO 레이블 있는 샘플만 추출 (최소 1개 TPO 활성화)
tpo_cols = [c for c in tpo_df.columns if c.startswith("tpo_")]
tpo_labels_full = tpo_df[tpo_cols].values
has_tpo = tpo_labels_full.sum(axis=1) > 0
valid_idx = np.where(has_tpo)[0]
print(f"  TPO 레이블 있는 샘플: {len(valid_idx)}개")

# 1000개 샘플링
sample_idx = np.random.choice(valid_idx, size=min(SAMPLE_N, len(valid_idx)), replace=False)
sample_idx.sort()

texts = [json_data[i]["text"] for i in sample_idx]
X_tfidf = X_tfidf_full[sample_idx]

# TPO 단일 레이블 (가장 먼저 활성화된 TPO로 단순화)
tpo_names = [c.replace("tpo_", "") for c in tpo_cols]
tpo_matrix = tpo_labels_full[sample_idx]
tpo_single = np.array([
    np.argmax(row) if row.sum() > 0 else -1
    for row in tpo_matrix
])

print(f"  샘플링 완료: {len(texts)}개")
print(f"  TPO 분포: {Counter(tpo_names[i] for i in tpo_single if i >= 0).most_common()}")

# ============================================================
# 2. TF-IDF + SVD → 768차원
# ============================================================
print("\n[2] TF-IDF + Truncated SVD (768차원) 생성 중...")
t0 = time.time()

svd = TruncatedSVD(n_components=768, random_state=RANDOM_SEED, n_iter=5)
X_svd = svd.fit_transform(X_tfidf)
X_svd = Normalizer().fit_transform(X_svd)

t_svd = time.time() - t0
explained = svd.explained_variance_ratio_.sum()
print(f"  완료: {t_svd:.1f}초")
print(f"  SVD shape: {X_svd.shape}")
print(f"  설명된 분산 비율: {explained:.3f} ({explained*100:.1f}%)")

# ============================================================
# 3. DistilBERT → 768차원
# ============================================================
print("\n[3] DistilBERT 임베딩 생성 중 (CPU, ~수 분 소요)...")

from transformers import DistilBertTokenizer, DistilBertModel
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  사용 디바이스: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
model.eval()

def get_bert_embeddings_batch(texts, batch_size=128, max_length=64):
    all_embeddings = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start:start+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # mean pooling
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        emb = (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)
        all_embeddings.append(emb.cpu().numpy())
        if (start // batch_size) % 5 == 0:
            print(f"  진행: {min(start+batch_size, total)}/{total}", flush=True)
    return np.vstack(all_embeddings)

t0 = time.time()
X_bert = get_bert_embeddings_batch(texts, batch_size=32, max_length=64)
X_bert = Normalizer().fit_transform(X_bert)
t_bert = time.time() - t0

print(f"  완료: {t_bert:.1f}초")
print(f"  BERT shape: {X_bert.shape}")

# ============================================================
# 4. 평가 A: 코사인 유사도 분포
# ============================================================
print("\n[4] 평가 A: 코사인 유사도 분포 분석...")

def cosine_similarity_stats(X, tpo_labels, n_pairs=3000):
    """같은 TPO / 다른 TPO 쌍의 코사인 유사도 평균 계산"""
    idx = np.random.choice(len(X), size=min(n_pairs * 2, len(X)), replace=False)
    pairs = [(idx[i], idx[i+1]) for i in range(0, len(idx)-1, 2)]

    same_sims, diff_sims = [], []
    for a, b in pairs:
        if tpo_labels[a] == -1 or tpo_labels[b] == -1:
            continue
        sim = float(np.dot(X[a], X[b]))
        if tpo_labels[a] == tpo_labels[b]:
            same_sims.append(sim)
        else:
            diff_sims.append(sim)
    return same_sims, diff_sims

same_svd, diff_svd = cosine_similarity_stats(X_svd, tpo_single)
same_bert, diff_bert = cosine_similarity_stats(X_bert, tpo_single)

print(f"\n  {'방법':<20} {'같은TPO 평균':>14} {'다른TPO 평균':>14} {'분리도(gap)':>12}")
print(f"  {'-'*62}")
gap_svd = np.mean(same_svd) - np.mean(diff_svd)
gap_bert = np.mean(same_bert) - np.mean(diff_bert)
print(f"  {'TF-IDF + SVD':<20} {np.mean(same_svd):>14.4f} {np.mean(diff_svd):>14.4f} {gap_svd:>12.4f}")
print(f"  {'DistilBERT':<20} {np.mean(same_bert):>14.4f} {np.mean(diff_bert):>14.4f} {gap_bert:>12.4f}")

# ============================================================
# 5. 평가 B: 클러스터링 품질
# ============================================================
print("\n[5] 평가 B: 클러스터링 품질 (KMeans, k=10)...")

K = 10

t0 = time.time()
km_svd = KMeans(n_clusters=K, random_state=RANDOM_SEED, n_init=5, max_iter=100)
labels_svd = km_svd.fit_predict(X_svd)
t_km_svd = time.time() - t0

t0 = time.time()
km_bert = KMeans(n_clusters=K, random_state=RANDOM_SEED, n_init=5, max_iter=100)
labels_bert = km_bert.fit_predict(X_bert)
t_km_bert = time.time() - t0

valid_mask = tpo_single >= 0

sil_svd  = silhouette_score(X_svd, labels_svd, sample_size=500)
sil_bert = silhouette_score(X_bert, labels_bert, sample_size=500)
db_svd   = davies_bouldin_score(X_svd, labels_svd)
db_bert  = davies_bouldin_score(X_bert, labels_bert)
nmi_svd  = normalized_mutual_info_score(tpo_single[valid_mask], labels_svd[valid_mask])
nmi_bert = normalized_mutual_info_score(tpo_single[valid_mask], labels_bert[valid_mask])

print(f"\n  {'방법':<20} {'Silhouette↑':>12} {'Davies-Bouldin↓':>16} {'NMI↑':>8} {'시간(s)':>8}")
print(f"  {'-'*68}")
print(f"  {'TF-IDF + SVD':<20} {sil_svd:>12.4f} {db_svd:>16.4f} {nmi_svd:>8.4f} {t_km_svd:>8.1f}")
print(f"  {'DistilBERT':<20} {sil_bert:>12.4f} {db_bert:>16.4f} {nmi_bert:>8.4f} {t_km_bert:>8.1f}")

# ============================================================
# 6. 평가 C: Top-K 검색 품질
# ============================================================
print("\n[6] 평가 C: Top-K 유사 아이템 검색 품질 (K=10)...")

def topk_precision(X, tpo_labels, K=10, n_queries=200):
    """쿼리 아이템의 Top-K 이웃 중 같은 TPO 비율"""
    query_idx = np.random.choice(
        np.where(tpo_labels >= 0)[0],
        size=min(n_queries, (tpo_labels >= 0).sum()),
        replace=False
    )
    sim_matrix = cosine_similarity(X[query_idx], X)
    precisions = []
    for i, q in enumerate(query_idx):
        sims = sim_matrix[i].copy()
        sims[q] = -1  # 자기 자신 제외
        top_k = np.argsort(sims)[-K:]
        valid_top = [j for j in top_k if tpo_labels[j] >= 0]
        if not valid_top:
            continue
        prec = sum(1 for j in valid_top if tpo_labels[j] == tpo_labels[q]) / len(valid_top)
        precisions.append(prec)
    return np.mean(precisions), np.std(precisions)

prec_svd, std_svd   = topk_precision(X_svd, tpo_single, K=10, n_queries=200)
prec_bert, std_bert = topk_precision(X_bert, tpo_single, K=10, n_queries=200)

print(f"\n  {'방법':<20} {'Precision@10 평균':>18} {'std':>8}")
print(f"  {'-'*50}")
print(f"  {'TF-IDF + SVD':<20} {prec_svd:>18.4f} {std_svd:>8.4f}")
print(f"  {'DistilBERT':<20} {prec_bert:>18.4f} {std_bert:>8.4f}")

# ============================================================
# 7. 최종 요약
# ============================================================
print("\n" + "=" * 60)
print("  최종 실험 요약")
print("=" * 60)

summary = {
    "방법": ["TF-IDF+SVD(768)", "DistilBERT(768)"],
    "임베딩 생성 시간(s)": [round(t_svd, 1), round(t_bert, 1)],
    "SVD 분산 설명(%)": [round(explained*100, 1), "-"],
    "같은TPO 유사도": [round(np.mean(same_svd), 4), round(np.mean(same_bert), 4)],
    "다른TPO 유사도": [round(np.mean(diff_svd), 4), round(np.mean(diff_bert), 4)],
    "유사도 분리도(gap)": [round(gap_svd, 4), round(gap_bert, 4)],
    "Silhouette(↑)": [round(sil_svd, 4), round(sil_bert, 4)],
    "Davies-Bouldin(↓)": [round(db_svd, 4), round(db_bert, 4)],
    "NMI(↑)": [round(nmi_svd, 4), round(nmi_bert, 4)],
    "Precision@10(↑)": [round(prec_svd, 4), round(prec_bert, 4)],
}

df_summary = pd.DataFrame(summary)
print(df_summary.to_string(index=False))

# CSV 저장
out_path = os.path.join(BASE_DIR, "experiment_results.csv")
df_summary.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n결과 저장: {out_path}")
print("실험 완료.")
