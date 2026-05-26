# 텍스트 표현 방법 선택 근거 보고서
## TF-IDF vs DistilBERT — 패션 메타데이터 텍스트 임베딩 비교

---

## 1. 개요

본 프로젝트는 Polyvore 및 H&M 데이터셋의 상품 메타데이터(상품명, 카테고리, 색상, 의류군, 상세설명)를
텍스트 피처 벡터로 변환하는 과정에서 **TF-IDF + TruncatedSVD** 방식을 채택하였다.
본 보고서는 DistilBERT 대비 TF-IDF를 선택한 근거를 실험 결과 및 관련 문헌을 통해 기술한다.

---

## 2. 비교 대상 및 실험 설계

### 2.1 비교 방법

| 방법 | 설명 |
|------|------|
| **TF-IDF + TruncatedSVD** | sklearn TfidfVectorizer (20,000차원) → TruncatedSVD로 768차원 축소 → L2 정규화 |
| **DistilBERT (mean pooling)** | `distilbert-base-uncased` 사전학습 모델, attention mask 기반 mean pooling → 768차원, L2 정규화 |

### 2.2 실험 데이터

- **데이터셋**: Polyvore `polyvore_text_features_final.json`
- **샘플 수**: 6,728개 (TPO 레이블 보유 outfit)
- **텍스트 구성**: outfit 이름 + 설명 + 아이템 이름 합산 후 전처리
- **Ground Truth**: TPO(Time/Place/Occasion) 레이블 10개 카테고리
  (casual, office, party, school, travel, beach, wedding, home, sport, outdoor)
- **실험 환경**: CPU(SVD), GPU — NVIDIA GeForce RTX 5070 Ti (DistilBERT)

### 2.3 평가 지표

| 지표 | 설명 |
|------|------|
| 코사인 유사도 분리도 (Gap) | 같은 TPO 쌍 평균 유사도 − 다른 TPO 쌍 평균 유사도 |
| Silhouette Score ↑ | 클러스터 내부 응집도 (높을수록 좋음, KMeans k=10) |
| Davies-Bouldin Index ↓ | 클러스터 간 분리도 (낮을수록 좋음) |
| NMI ↑ | TPO 레이블 대비 클러스터 일치도 (0~1) |
| Precision@10 ↑ | Top-10 유사 아이템 중 같은 TPO 비율 |

---

## 3. 실험 결과

### 3.1 전체 비교표

| 방법 | 임베딩 생성 시간 | 같은 TPO 유사도 | 다른 TPO 유사도 | Gap (분리도) | Silhouette ↑ | Davies-Bouldin ↓ | NMI ↑ | Precision@10 ↑ |
|------|:-----------:|:----------:|:----------:|:-------:|:--------:|:------------:|:----:|:-----------:|
| TF-IDF + SVD | 7.8s | 0.0480 | 0.0310 | **0.0170** | 0.0239 | 6.1588 | **0.0893** | **0.3575** |
| DistilBERT | 3.3s (GPU) | 0.8943 | 0.8880 | 0.0062 | **0.0397** | **3.5483** | 0.0696 | 0.2815 |

### 3.2 핵심 관찰

**① 유사도 분리도 (Gap): TF-IDF 2.7배 우위**

DistilBERT는 같은 TPO(0.8943)와 다른 TPO(0.8880) 간 유사도 차이가 0.0062에 불과하다.
즉 모든 패션 아이템이 거의 동일하게 보이는 **구별력 붕괴(discrimination collapse)** 현상이 발생한다.
TF-IDF는 Gap이 0.0170으로 2.7배 더 뚜렷하게 카테고리를 분리한다.

**② Precision@10: TF-IDF +7.6%p 우위**

Top-10 유사 아이템 검색에서 같은 TPO 카테고리 아이템을 찾는 정밀도가
TF-IDF(0.3575) > DistilBERT(0.2815)로, 실질적인 검색 품질에서도 TF-IDF가 우위이다.
10개 카테고리 랜덤 기준(~0.10) 대비 TF-IDF는 3.5배, DistilBERT는 2.8배 수준이다.

**③ NMI: TF-IDF +28% 우위**

TPO ground truth 레이블 대비 클러스터 일치도에서 TF-IDF(0.0893)가 DistilBERT(0.0696) 대비
28% 높다. 패션 메타데이터의 카테고리 의미를 TF-IDF가 더 잘 포착함을 시사한다.

**④ Silhouette·Davies-Bouldin: DistilBERT 우위 (단, 해석 주의)**

클러스터 내부 응집도(Silhouette)와 분리도(DB Index)에서는 DistilBERT가 더 좋은 수치를 보인다.
그러나 이는 DistilBERT 임베딩 공간 자체가 밀집되어 있어 어떤 기준으로 나눠도
내부 편차가 작아 보이는 효과이며, 실제 TPO 의미 구조(NMI)와는 일치하지 않는다.

---

## 4. 이론적 근거 (관련 논문)

### 4.1 BERT 임베딩의 Anisotropy 문제

> **Li et al. (2020). "On the Sentence Embeddings from Pre-trained Language Models." EMNLP 2020.**

사전학습 BERT 임베딩은 전체 벡터 공간을 균등하게 사용하지 않고
**좁은 원뿔(cone) 형태**로 집중되는 비등방성(anisotropy) 특성을 가진다.
이로 인해 두 임베딩 간 코사인 유사도가 항상 높게 측정되어 의미적 구별력이 감소한다.
본 실험에서 DistilBERT의 유사도가 TPO 관계와 무관하게 0.88~0.89로 수렴한 현상이 이를 뒷받침한다.

### 4.2 짧은 텍스트에서 BERT Pooling의 한계

> **Kong et al. (2022). "PromptBERT: Improving BERT Sentence Embeddings with Prompts." EMNLP 2022.**

BERT의 mean pooling은 모든 토큰을 동등하게 취급하여,
패션 메타데이터처럼 **핵심 키워드(예: black, leather, office)의 변별력이 중요한 짧은 텍스트**에서
정보 손실이 발생한다. 반면 TF-IDF의 IDF 가중치는 도메인 핵심 키워드를 자동으로 upweight한다.

### 4.3 도메인 Fine-tuning 없는 BERT의 한계

> **Garg et al. (2020). "Comparing BERT against Traditional Machine Learning Text Classification." arXiv 2005.13012.**

Fine-tuning 없는 generic BERT는 단문 분류에서 TF-IDF와 유사한 정확도를 내면서
연산 비용은 1,000배 이상 소요된다.
패션 도메인 전용 어휘(TPO 카테고리, 색상 용어 등)는 general corpus로 학습된 BERT가
충분히 표현하지 못한다.

### 4.4 이커머스 상품 메타데이터에서 TF-IDF의 강세

> **Biswas et al. (2017). "MRNet-Product2Vec: A Multi-task Recurrent Neural Network for Product Embeddings." Amazon Science.**

이커머스 상품 메타데이터에서 dense 임베딩이 TF-IDF를 능가하려면
**대규모 도메인 fine-tuning이 필수**임을 실증하였다.
상품명 + 카테고리 + 색상 등 구조화된 텍스트에서는 TF-IDF의 키워드 매칭이
fine-tuning 없는 신경망 임베딩을 능가하는 강력한 베이스라인임을 보인다.

---

## 5. 실험의 한계 및 유의사항

본 실험은 TF-IDF의 우위를 보이기 위한 **intrinsic evaluation**이며, 다음 한계를 가진다.

| 한계 | 설명 |
|------|------|
| TPO 레이블 순환성 | TPO 레이블이 텍스트 키워드 기반으로 파생되어 TF-IDF에 유리한 평가 환경일 수 있음 |
| 절대 수치 낮음 | NMI 0.09, Silhouette 0.02 등 절대 성능은 낮으며, 이는 두 방법 모두에 해당 |
| TPO 커버리지 부족 | 전체 17,316개 중 6,728개(38.8%)만 TPO 레이블 보유 |
| Intrinsic 평가 한계 | 클러스터링·유사도 결과가 하위 Classifier 성능을 직접 보장하지 않음 |

**따라서 본 실험 결과는 TF-IDF 선택의 정당화 근거로 사용되며,
실제 프로젝트 유효성은 Stage 1 Compatibility Classifier의 AUC/F1 결과로 최종 검증한다.**

---

## 6. 결론

패션 상품 메타데이터(상품명 + 카테고리 + 색상 + 설명)는 짧고 키워드 중심적인 구조화 텍스트로,
Generic DistilBERT(fine-tuning 없음)보다 **TF-IDF + TruncatedSVD**가 더 적합한 표현 방법이다.

실험에서 TF-IDF는 TPO 카테고리 분리도(Gap) 2.7배, Precision@10 +7.6%p, NMI +28%의 우위를 보였으며,
이는 BERT 임베딩의 anisotropy, pooling 한계, 도메인 어휘 미적응이라는
이론적 원인으로 설명된다.

단, 하위 분류기 성능 검증이 추가로 필요하며, 도메인 fine-tuned BERT 도입 시 재비교를 권장한다.

---

*실험 환경: Python 3.13.2 / sklearn 1.8.0 / PyTorch 2.11.0+cu128 / NVIDIA RTX 5070 Ti*
*실험 일자: 2026-05-03*
