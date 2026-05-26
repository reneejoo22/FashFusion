# FashFusion 모델 아키텍처 상세 설명

---

## 1. 전체 구조 개요


1. **멀티모달 표현 학습**: 텍스트(TF-IDF, BERT, 구조적 속성)와 이미지(MobileNetV3) 특징을 결합하여 아이템당 2,411차원의 풍부한 표현을 구성한다.
2. **크로스 도메인 전이**: Polyvore(패션 전문가 큐레이션) 도메인에서 학습한 스타일 호환성 지식을 H&M(실제 구매 이력) 도메인으로 전이한다. 단, 가중치 직접 전이 대신 **호환성 점수를 추가 피처로 주입**하는 방식을 채택한다.

전체 파이프라인은 아래와 같다.

```
[Polyvore Dataset]                    [H&M Dataset]
      │                                     │
      ▼                                     ▼
 Feature Extraction                  Feature Extraction
 (tfidf+bert+struct+img)             (tfidf+bert+struct+img)
      │                                     │
      ▼                                     │
 Stage 1 Training                           │
 (FashFusionStage1)                         │
      │                                     │
      ▼                                     ▼
 p_compat 계산 ──────────────────► Stage 2 Training
 (H&M 페어에 적용)                   (FashFusionStage2)
                                          │
                                          ▼
                                  구매 호환성 예측
```

---

## 2. 특징 추출 (Feature Extraction)

아이템 하나에 대해 이질적인 4가지 모달리티를 추출하고 연접(concatenation)하여 **2,411차원 벡터** $\mathbf{x} \in \mathbb{R}^{2411}$를 구성한다.

$$\mathbf{x} = [\mathbf{x}_{\text{tfidf}} \| \mathbf{x}_{\text{bert}} \| \mathbf{x}_{\text{struct}} \| \mathbf{x}_{\text{img}}]$$

### 2.1 TF-IDF (300d)

상품명, 카테고리명 등 텍스트 필드에 TF-IDF를 적용하고 SVD를 통해 300차원으로 축소한다.

$$\text{TF-IDF}(t, d) = \text{tf}(t,d) \cdot \log\frac{N}{df(t)}$$

- **선택 이유**: 도메인 특화 어휘(브랜드명, 소재명, 스타일 용어)에서 단어 빈도 기반 표현이 BERT보다 효과적임을 실험적으로 확인 (A1: 0.9668 > A2_BERT: 0.9490).

### 2.2 BERT (768d)

`bert-base-uncased`의 [CLS] 토큰 출력을 768차원 의미론적 임베딩으로 사용한다.

$$\mathbf{x}_{\text{bert}} = \text{BERT}_{\text{[CLS]}}(\text{text})$$

- **모델**: `distilbert-base-uncased` (DistilBERT). Attention mask 기반 mean pooling 적용
- **동결(Frozen) 사용**: fine-tuning 없이 사전학습 임베딩을 그대로 활용. 패션 도메인 특화가 부족하나, TF-IDF와 상호 보완적 역할을 한다.

### 2.3 구조적 특징 (63d)

카테고리 코드, 색상 그룹, 가격대, 부서 등 범주형/수치형 메타데이터를 원-핫 인코딩 및 정규화하여 결합한다.

- 단독으로는 AUC=0.7354로 성능이 낮으나, 멀티모달 결합 시 보완적 신호를 제공한다.

### 2.4 이미지 특징 (1280d) — MobileNetV3-Large

ImageNet 사전학습 MobileNetV3-Large에서 특징을 추출한다. **주의**: 일반적으로 `features + avgpool`까지만 사용하면 960차원이 출력되나, 본 모델은 `classifier[0]` (Linear 960→1280)과 `classifier[1]` (Hardswish)까지 포함하여 **1,280차원**을 추출한다.

```python
feature_extractor = nn.Sequential(
    backbone.features,      # Conv layers
    backbone.avgpool,       # Adaptive avg pool → (B, 960, 1, 1)
    nn.Flatten(1),          # → (B, 960)
    backbone.classifier[0], # Linear(960, 1280)
    backbone.classifier[1], # Hardswish activation
)                           # → (B, 1280)
```

MobileNetV3의 Hardswish는 $h(x) = x \cdot \frac{\text{ReLU6}(x+3)}{6}$로 정의되며, ReLU보다 매끄러운 비선형성을 제공한다. 단독 AUC=0.9672로 TF-IDF(0.9668)와 동등한 수준의 강력한 시각적 표현력을 확인했다.

---

## 3. ProjectionMLP

각 아이템 벡터 $\mathbf{x} \in \mathbb{R}^{d_{\text{item}}}$를 공통 임베딩 공간 $\mathbb{R}^{256}$으로 사영한다.

$$\mathbf{e} = \text{ProjectionMLP}(\mathbf{x}) \in \mathbb{R}^{256}$$

### 구조

```
x (d_item) → Linear(d_item, 512) → LayerNorm(512) → ReLU → Dropout(0.3)
           → Linear(512, 256)   → LayerNorm(256)  → ReLU
           → e (256d)
```

### 설계 선택

- **LayerNorm**: BatchNorm 대비 배치 크기에 독립적이며, 추론 시 일관성을 보장한다. 특히 이질적인 모달리티 특징이 혼합된 고차원 입력에서 학습 안정성에 기여한다.
- **병목(Bottleneck) 구조**: 2411d → 512d → 256d의 점진적 축소를 통해 모달리티 간 불균형한 차원을 정규화하고 일반화 성능을 향상시킨다.
- **Stage 1→2 가중치 전이**: Stage 2 초기화 시 Stage 1에서 학습한 ProjectionMLP 가중치를 재사용하고, 낮은 학습률(LR=3e-5, fusion LR의 1/3)로 fine-tuning한다.

---

## 4. 페어 특징 구성 (Pair Feature Construction)

두 아이템 A, B의 임베딩 $\mathbf{e}_A, \mathbf{e}_B \in \mathbb{R}^{256}$으로부터 **1,031차원 페어 특징**을 구성한다.

$$\mathbf{p} = [\mathbf{e}_A \| \mathbf{e}_B \| |\mathbf{e}_A - \mathbf{e}_B| \| \mathbf{e}_A \odot \mathbf{e}_B \| \cos(\mathbf{e}_A, \mathbf{e}_B) \| \mathbf{m} \| s_{\text{pc}}]$$

| 구성요소 | 차원 | 수식 | 역할 |
|---|---|---|---|
| $\mathbf{e}_A$ | 256 | — | 아이템 A 임베딩 |
| $\mathbf{e}_B$ | 256 | — | 아이템 B 임베딩 |
| $\|\mathbf{e}_A - \mathbf{e}_B\|$ | 256 | 절댓값 차이 | 비대칭 차이 포착 |
| $\mathbf{e}_A \odot \mathbf{e}_B$ | 256 | 요소별 곱 | 차원별 상호작용 |
| $\cos(\mathbf{e}_A, \mathbf{e}_B)$ | 1 | $\frac{\mathbf{e}_A \cdot \mathbf{e}_B}{\|\mathbf{e}_A\|\|\mathbf{e}_B\|}$ | 전역 유사도 |
| $\mathbf{m}$ | 5 | 카테고리 일치 등 | 명시적 메타 신호 |
| $s_{\text{pc}}$ | 1 | Stage 1 출력 | 크로스도메인 신호 |
| **합계** | **1,031** | | |

### 설계 근거

$[\mathbf{e}_A \| \mathbf{e}_B \| \mathbf{e}_A - \mathbf{e}_B \| \mathbf{e}_A \odot \mathbf{e}_B]$ 구조는 자연어 처리의 문장 쌍 비교 모델(InferSent, BiMPM 등)에서 효과가 검증된 방식으로, 두 표현 간의 방향성, 크기, 요소별 상호작용을 동시에 포착한다.

코사인 유사도를 별도로 추가한 이유는, $\mathbf{e}_A \odot \mathbf{e}_B$가 크기 정보에 민감한 반면, 코사인은 방향(각도) 정보만을 순수하게 담기 때문이다.

---

## 5. FusionMLP

페어 특징 $\mathbf{p} \in \mathbb{R}^{1031}$을 입력받아 호환성 점수 $\hat{y} \in [0,1]$을 출력한다.

$$\hat{y} = \sigma(W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \mathbf{p} + b_1) + b_2) + b_3)$$

### 구조

```
p (1031d) → Linear(1031, 512) → ReLU → Dropout(0.3)
          → Linear(512, 128)  → ReLU → Dropout(0.2)
          → Linear(128, 1)    → Sigmoid
          → ŷ ∈ [0,1]
```

### 손실 함수 및 최적화

- **Binary Cross-Entropy**: $\mathcal{L} = -\frac{1}{N}\sum_{i}[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$
- **AdamW**: L2 정규화를 weight decay(1e-4)로 분리하여 적용, 일반화 성능 향상
- **Gradient Clipping** (norm=1.0): 학습 초기 불안정한 그래디언트 폭발 방지
- **ReduceLROnPlateau**: val AUC 기준으로 patience=3 이후 LR을 0.5배 감소

---

## 6. 2단계 학습 전략

### Stage 1: Polyvore 패션 호환성 학습

Polyvore 데이터셋의 전문가 큐레이션 호환성 레이블로 FashFusionStage1을 학습한다. 이 단계에서 p_compat 피처가 없으므로 페어 차원은 1,030d($256 \times 4 + 1 + 5$)이다.

- **목적**: 패션 전문가의 스타일 호환성 지식을 ProjectionMLP 가중치에 내재화
- **결과**: 폴리보어 Val AUC=0.8763 (epoch 7)

### Stage 2: H&M 구매 이력 예측

Stage 1이 완료되면, Stage 1 모델을 H&M 전체 페어에 적용하여 **p_compatible 점수**를 계산한다. 이를 추가 피처로 삽입(1,031d)하여 Stage 2를 학습한다.

- **Differential LR (T2)**: ProjectionMLP는 lr=3e-5로 미세 조정, FusionMLP는 lr=1e-4로 학습. Polyvore에서 학습한 표현을 보존하면서 H&M 도메인에 점진적으로 적응
- **Random Init (T1)**: ProjectionMLP를 랜덤 초기화하고 단일 lr=1e-4로 학습. H&M 도메인에 완전 자유 최적화

### p_compatible의 역할

Stage 1이 내뱉는 점수 $s_{\text{pc}} = f_{\text{S1}}(\mathbf{e}_A, \mathbf{e}_B) \in [0,1]$은 Polyvore에서 학습한 패션 스타일 호환성 신호다. H&M 페어에서 실측 실험 결과:

- 양성 페어(함께 구매됨): $\bar{s}_{\text{pc}} = 0.7213$
- 음성 페어(함께 구매되지 않음): $\bar{s}_{\text{pc}} = 0.6587$

통계적으로 유의미한 분리(Δ=0.0626)가 확인되어, 크로스 도메인 전이의 유효성을 검증했다. A6→A7 비교에서 p_compat 추가로 AUC +0.003 향상을 확인했다.

---

## 7. Transfer Learning 실험 결과 해석

| 모델 | proj 초기화 | p_compat | AUC |
|---|---|---|---|
| A6_TextImg_noPc | Polyvore | ✗ | 0.9763 |
| A7_FullMM (T2) | Polyvore | ✓ | 0.9793 |
| T1_MM_NoTransfer | 랜덤 | ✓ | 0.9826~0.9834 |

**T1 > T2**의 해석: Polyvore(패션 전문가 스타일 큐레이션)와 H&M(실제 소비자 구매 이력) 간에는 비자명한 도메인 갭이 존재한다. Polyvore 기반 가중치로 초기화된 ProjectionMLP를 낮은 LR로 fine-tuning하면, H&M 도메인에 맞는 표현 공간으로의 이동이 제약된다. 반면 p_compatible 피처(크로스도메인 점수 자체)는 Stage 2의 FusionMLP가 도메인 갭을 명시적으로 보정할 수 있는 신호로 유효하게 활용된다.

즉, **크로스도메인 지식의 최적 전이 방식은 가중치 초기화(implicit)가 아닌 피처 주입(explicit)**임을 실험적으로 시사한다.

---

## 8. 모델 파라미터 요약

| 컴포넌트 | 구조 | 파라미터 수 |
|---|---|---|
| ProjectionMLP | 2411→512→256 | 약 1.44M |
| FusionMLP | 1031→512→128→1 | 약 593K |
| **전체** | | **약 2.03M** |

---

## 9. 구현 상의 핵심 고려사항

### 9.1 ZIP64 문제 해결

PyTorch의 내장 ZIP 라이브러리(miniz)는 4GB 초과 파일(ZIP64 포맷)을 지원하지 않는다. 2,411차원 × 500,000 페어 × float32 = 약 4.8GB로 임계를 초과한다. 이를 해결하기 위해 페어 파일에는 **인덱스만 저장**하고, 특징 행렬은 별도의 `.npy` 파일로 분리 저장한다.

```
hnm_pairs_mm_train.pt  (~20MB): idx_a, idx_b, labels, meta, p_compat
hnm_mm_feats.npy      (~1.02GB): (105542, 2411) float32
```

학습 시 `feats[idx_a[i]]`로 즉시 조회하므로 메모리 효율과 I/O 성능을 동시에 확보한다.

### 9.2 Windows DataLoader 데드락

Windows 환경에서 PyTorch DataLoader의 `num_workers > 0` 설정은 멀티프로세싱 시작 방식(spawn)으로 인해 `if __name__ == '__main__'` 가드 없이 사용 시 데드락이 발생한다. 전 모듈에서 `num_workers=0`으로 고정하여 이를 회피했다.

### 9.3 MobileNetV3 특징 차원 오류 방지

`torchvision`의 MobileNetV3-Large에서 `features + avgpool`만 사용하면 960차원이 출력된다. `classifier[0]`(Linear 960→1280)과 `classifier[1]`(Hardswish)를 반드시 포함해야 사전학습 시 설계된 1,280차원 표현을 올바르게 추출할 수 있다.

---

## 10. 코드 파일 구성 및 설명

전체 파이프라인은 단계별로 독립된 스크립트로 구성된다. 각 스크립트는 이전 단계의 출력 파일을 입력으로 받아 처리하는 직렬 파이프라인 구조를 따른다.

```
step3a → step3b → step3c → step6mm → step7mm → step8mm → full_experiments_final
```

---

### `step3a_poly_mobilenet.py` — Polyvore 이미지 특징 추출

Polyvore 데이터셋의 아이템 이미지를 MobileNetV3-Large로 처리하여 1,280차원 특징 벡터를 추출한다.

- **입력**: Polyvore 이미지 디렉토리
- **출력**: `poly_img_1280.npy` (142,480 × 1,280)
- **커버리지**: 142,480개 아이템 중 27,372개 이미지 존재 (19.2%). 이미지가 없는 아이템은 영벡터(zero vector)로 처리
- **핵심 처리**: `features → avgpool → Flatten → classifier[0] → classifier[1]` 순으로 특징 추출하여 960d가 아닌 1,280d 확보
- **Windows 대응**: `num_workers=0`, `pin_memory=False`로 DataLoader 데드락 방지

---

### `step3b_hnm_mobilenet.py` — H&M 이미지 특징 추출

H&M 데이터셋(Kaggle, 30.81GB) 아이템 이미지를 동일한 MobileNetV3-Large로 처리한다.

- **입력**: H&M 이미지 디렉토리 (Kaggle archive.zip에서 추출)
- **출력**: `hnm_img_1280.npy` (105,542 × 1,280)
- **커버리지**: 105,542개 아이템 중 105,100개 이미지 존재 (99.6%). step3a 대비 압도적으로 높은 커버리지
- **step3a와 동일한 아키텍처** 사용하여 두 도메인 간 이미지 표현 공간 통일

---

### `step3c_rebuild_pairs.py` — 멀티모달 페어 인덱스 파일 생성

기존의 텍스트 특징 기반 페어 파일(.pt)을 인덱스 기반으로 재구성하고, 멀티모달 특징 행렬을 통합 저장한다.

- **입력**: `poly_pairs_{split}.pt`, `hnm_pairs_{split}.pt` (텍스트 특징 직접 저장 방식), 각 모달리티별 `.npy` 파일
- **출력**:
  - `poly_mm_feats.npy` (142,480 × 2,411) — Polyvore 통합 특징 행렬
  - `hnm_mm_feats.npy` (105,542 × 2,411) — H&M 통합 특징 행렬
  - `poly_pairs_mm_{train|val|test}.pt`, `hnm_pairs_mm_{train|val}.pt` — 인덱스 기반 페어 파일 (~20MB)
- **핵심 설계**: 4GB 초과 파일의 ZIP64 문제를 회피하기 위해 페어 파일에는 `(idx_a, idx_b, labels, meta, p_compat)` 인덱스만 저장. 특징 행렬은 `.npy`로 분리
- **역방향 조회(Reverse Lookup)**: 기존 파일의 1,131d 텍스트 벡터를 키로 사용해 행렬 내 인덱스를 역산

---

### `step6mm_stage1_train.py` — Stage 1 멀티모달 모델 학습 (Polyvore)

Polyvore 패션 호환성 데이터로 FashFusionStage1을 학습한다. 이 단계는 크로스도메인 전이의 출발점이다.

- **입력**: `poly_mm_feats.npy`, `poly_pairs_mm_{train|val|test}.pt`
- **출력**: `stage1_mm_model.pt` (best val AUC 기준 저장)
- **모델**: ProjectionMLP(2411→256) + FusionMLP(1030→1) — p_compat 피처 없음
- **학습 설정**: AdamW(lr=3e-4), ReduceLROnPlateau(patience=2), Gradient Clip(1.0), Early Stopping(patience=5)
- **결과**: Val AUC=0.8763 (epoch 7), Test AUC=0.7384

---

### `step7mm_hnm_pcompat.py` — H&M p_compatible 점수 계산

학습된 Stage 1 모델로 H&M 전체 페어에 대한 패션 호환성 점수(p_compatible)를 계산하고 페어 파일에 주입한다.

- **입력**: `stage1_mm_model.pt`, `hnm_pairs_mm_{train|val}.pt`, `hnm_mm_feats.npy`
- **출력**: `hnm_p_compat_mm.npy`, 업데이트된 `hnm_pairs_mm_{train|val}.pt` (p_compatible 필드 추가)
- **검증**: 양성 페어 평균 0.7213 vs 음성 페어 0.6587 (Δ=0.0626). 크로스도메인 신호의 유효성 확인
- **처리 방식**: 모델을 eval 모드로 고정 후 배치 단위 추론. 그래디언트 계산 없음(no_grad)

---

### `step8mm_stage2_train.py` — Stage 2 멀티모달 모델 학습 (H&M)

p_compatible 피처가 포함된 H&M 데이터로 FashFusionStage2(=T2 모델)를 학습한다.

- **입력**: `hnm_mm_feats.npy`, `hnm_pairs_mm_{train|val}.pt` (p_compat 포함), `stage1_mm_model.pt`
- **출력**: `stage2_mm_model.pt` (best val AUC 기준 저장)
- **모델**: ProjectionMLP(2411→256, Stage1 가중치로 초기화) + FusionMLP(1031→1)
- **Differential LR**: ProjectionMLP lr=3e-5 (fine-tune), FusionMLP lr=1e-4 (full train)
- **결과**: Val AUC=0.9793, PR-AUC=0.9675, F1=0.9104 (epoch 39)

---

### `train_t1_model.py` — T1 모델 단독 학습

T1 모델(랜덤 proj 초기화)을 독립적으로 학습하고 체크포인트를 저장하는 스크립트다. `full_experiments_final.py`에서 T1 훈련 부분만 분리한 것이다.

- **입력**: `hnm_mm_feats.npy`, `hnm_pairs_mm_{train|val}.pt`
- **출력**: `stage2_t1_model.pt` (best val AUC 기준 저장)
- **모델**: ProjectionMLP(2411→256, **랜덤 초기화**) + FusionMLP(1031→1)
- **학습 설정**: 단일 AdamW(lr=1e-4), Early Stopping(patience=5)
- **T2와의 차이**: proj 초기화 방식만 다르며, 나머지 구조/데이터/설정 동일
- **결과**: Val AUC=0.9826~0.9834 (실행마다 랜덤 초기화로 미세 변동)

---

### `full_experiments_final.py` — 전체 비교 실험 통합 스크립트

14개 모델(Baseline 5개, SingleModal 4개, MultiModal 3개, Transfer 2개)을 동일한 val set에서 일괄 평가하여 공정한 비교를 수행한다.

- **입력**: 모든 특징 파일(.npy), 페어 파일(.pt), 기존 체크포인트(stage2_model.pt, stage2_mm_model.pt)
- **출력**: `full_experiment_results.csv` (14개 모델 결과)
- **구조**:
  - B1~B5: 별도 학습 없이 통계/유사도 기반 점수 계산
  - A1~A4: `FusionMLP`를 단일 모달리티 피처로 학습 (20 epoch)
  - A5: `stage2_model.pt` 로드 후 평가만 (텍스트 전용 H&M 모델)
  - A6: 멀티모달 + Stage1 proj 초기화, p_compat 없이 학습 (30 epoch)
  - A7: `stage2_mm_model.pt` 로드 후 평가만 (=T2)
  - T1: 멀티모달 + 랜덤 초기화, p_compat 포함 학습 (30 epoch)
  - T2: A7 결과 재참조 (동일 모델, 중복 학습 없음)
- **평가 지표**: AUC-ROC, PR-AUC, F1-score (threshold=0.5)

---

### 데이터셋 분할 요약

| 데이터셋 | Train | Val | Test |
|---|---|---|---|
| Polyvore (Stage 1) | ~180,000쌍 | 별도 | 별도 |
| H&M (Stage 2) | **450,000쌍 (90%)** | **50,000쌍 (10%)** | 없음 |

H&M 최종 모델 평가는 **Val set(50,000쌍)** 기준이며, Train/Val 분할은 시간 순서가 아닌 랜덤 분할이다.
