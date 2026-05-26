# FashFusion: 멀티모달 패션 추천 시스템

**빅데이터프로그래밍 | 한성대학교**

---

## 1. 프로젝트 개요

본 프로젝트는 패션 아이템 간 구매 호환성(co-purchase compatibility)을 예측하는 멀티모달 2단계 추천 시스템 **FashFusion**을 제안한다. 텍스트(TF-IDF, BERT, 구조적 속성)와 이미지(MobileNetV3) 특징을 결합하고, 크로스 도메인 전이 학습을 통해 예측 성능을 향상시킨다.

### 1.1 문제 정의

H&M 고객 구매 데이터에서 두 패션 아이템이 함께 구매될 가능성을 예측하는 문제이다.

- **입력**: 두 패션 아이템 (텍스트 + 이미지 특징)
- **출력**: 구매 호환성 점수 ∈ [0, 1]
- **평가 지표**: AUC-ROC, PR-AUC, F1-score

---

## 2. 데이터셋

### 2.1 Polyvore (Stage 1 학습)

| 항목 | 값 |
|---|---|
| 출처 | Polyvore 패션 코디 데이터셋 |
| 아이템 수 | 142,480개 |
| 이미지 커버리지 | 19.2% (27,372개) |
| 페어 수 (train) | 약 180,000쌍 |
| 레이블 | 패션 전문가 큐레이션 호환성 |

### 2.2 H&M (Stage 2 학습 및 평가)

| 항목 | 값 |
|---|---|
| 출처 | H&M Personalized Fashion Recommendations (Kaggle) |
| 아이템 수 | 105,542개 |
| 이미지 커버리지 | 99.6% (105,100개) |
| Train 페어 수 | 450,000쌍 |
| Val 페어 수 | 50,000쌍 |
| 레이블 | 실제 고객 구매 이력 기반 |

---

## 3. 모델 아키텍처

### 3.1 특징 추출 (Feature Extraction)

아이템당 총 **2,411차원** 멀티모달 특징 벡터를 구성한다.

| 모달리티 | 방법 | 차원 |
|---|---|---|
| TF-IDF | TfidfVectorizer (상품명, 카테고리) | 300d |
| BERT | bert-base-uncased CLS 임베딩 | 768d |
| 구조적 특징 | 카테고리, 색상, 가격 등 원-핫/정규화 | 63d |
| 이미지 | MobileNetV3-Large (classifier[0]+[1] 포함) | 1280d |
| **합계** | | **2,411d** |

> **이미지 특징 추출 주의사항**: MobileNetV3-Large의 `features + avgpool`만 사용하면 960d가 추출된다. `classifier[0]` (Linear 960→1280)과 `classifier[1]` (Hardswish)까지 포함해야 올바른 1,280d 특징을 얻는다.

### 3.2 2단계 학습 구조

```
[Stage 1 - Polyvore]
  Polyvore 패션 호환성 데이터
      → ProjectionMLP (2411d → 256d)
      → FusionMLP (1030d → 1 / sigmoid)
      → stage1_mm_model.pt 저장

[Stage 2 - H&M]
  H&M 구매 이력 데이터
  + p_compatible (Stage1 모델로 계산한 크로스도메인 신호)
      → ProjectionMLP (Stage1 가중치로 초기화, fine-tune)
      → FusionMLP (1031d → 1 / sigmoid)
      → stage2_mm_model.pt 저장
```

### 3.3 페어 특징 구성

두 아이템 A, B의 임베딩 eA, eB (각 256d)로부터 페어 특징을 구성한다.

```
pair_feat = [eA | eB | |eA-eB| | eA⊙eB | cos(eA,eB) | meta | p_compat]
           = [256 | 256 | 256 | 256 | 1 | 5 | 1] = 1,031d
```

| 구성요소 | 차원 | 설명 |
|---|---|---|
| eA, eB | 256+256 | 두 아이템 임베딩 |
| \|eA-eB\| | 256 | 차이 벡터 (절댓값) |
| eA⊙eB | 256 | 요소별 곱 |
| cos(eA, eB) | 1 | 코사인 유사도 |
| meta | 5 | 카테고리 일치 여부 등 메타 특징 |
| p_compat | 1 | Stage1 크로스도메인 호환성 점수 |

### 3.4 FusionMLP

```
1031d → Linear(512) → ReLU → Dropout(0.3)
      → Linear(128) → ReLU → Dropout(0.2)
      → Linear(1)   → Sigmoid
```

---

## 4. 실험 설계

총 4개 그룹, 14개 모델을 비교 평가한다.

| 그룹 | 목적 | 실험 |
|---|---|---|
| Baseline | 비학습 기반 비교군 | B1~B5 |
| Single-modal | 모달리티별 단독 기여도 | A1~A4 |
| Multi-modal | 멀티모달 조합 효과 | A5~A7 |
| Transfer | 전이 학습 효과 | T1~T2 |

### 4.1 학습 설정

| 하이퍼파라미터 | Stage 1 | Stage 2 |
|---|---|---|
| Optimizer | AdamW | AdamW (differential LR) |
| Learning Rate | 3e-4 | proj: 3e-5 / fusion: 1e-4 |
| Batch Size | 2,048 | 2,048 |
| Max Epochs | 30 | 40 |
| Early Stopping | patience=5 | patience=6 |
| LR Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| Grad Clip | 1.0 | 1.0 |
| Weight Decay | 1e-4 | 1e-4 |

---

## 5. 실험 결과

### 5.1 전체 결과 요약

| 모델 | 그룹 | AUC-ROC | PR-AUC | F1 |
|---|---|---|---|---|
| B1_Random | Baseline | 0.5033 | 0.4041 | 0.4480 |
| B2_Popularity | Baseline | 0.9250 | 0.8816 | 0.0081 |
| B3_Jaccard | Baseline | 0.5303 | 0.4378 | 0.0000 |
| B4_Cosine_Text | Baseline | 0.7021 | 0.6542 | 0.5730 |
| B5_Cosine_Image | Baseline | 0.6494 | 0.5853 | 0.5744 |
| A1_TFIDF_only | SingleModal | 0.9668 | 0.9504 | 0.8805 |
| A2_BERT_only | SingleModal | 0.9490 | 0.9286 | 0.8512 |
| A3_Struct_only | SingleModal | 0.7354 | 0.6677 | 0.5418 |
| A4_Image_only | SingleModal | 0.9672 | 0.9505 | 0.8759 |
| A5_Text_Stage2 | MultiModal | 0.9591 | 0.9412 | 0.8695 |
| A6_TextImg_noPc | MultiModal | 0.9763 | 0.9637 | 0.9040 |
| A7_FullMM_Stage2 | MultiModal | 0.9793 | 0.9675 | 0.9104 |
| **T1_MM_NoTransfer** | **Transfer** | **0.9834** | **0.9731** | **0.9115** |
| T2_MM_WithTransfer | Transfer | 0.9793 | 0.9675 | 0.9104 |

### 5.2 그룹별 분석

#### Group 1: Baseline 분석

- **B2_Popularity** (AUC=0.9250): 인기도 기반 베이스라인이 예상 외로 높은 AUC를 기록했으나 F1=0.0081로 극히 낮다. 이는 인기도 점수가 연속값으로 분포하여 0.5 임계값 기준 이진 분류가 작동하지 않기 때문이다.
- **B3_Jaccard** (F1=0.0000): 훈련 데이터에 등장한 적 없는 페어는 점수=0이 되므로 사실상 분류 불가.
- **B4, B5**: 학습 없이 코사인 유사도만으로 AUC 0.65~0.70 수준으로 의미 있는 신호가 존재함을 확인.

#### Group 2: 단일 모달리티 분석

- **TF-IDF** (0.9664)와 **Image** (0.9676)가 가장 높은 성능으로 두 모달리티 모두 강한 단독 예측력을 보인다.
- **BERT** (0.9485)는 TF-IDF보다 낮다. 사전학습 임베딩이 패션 도메인에 특화되지 않아 TF-IDF의 도메인 특화 어휘 표현에 비해 불리한 것으로 분석된다.
- **Struct** (0.7355): 카테고리/색상 등 구조적 특징만으로는 한계가 있으나, 다른 모달리티와 결합 시 보완적 역할을 한다.

#### Group 3: 멀티모달 조합 분석

- **A5 → A6**: 텍스트(1131d)에 이미지(1280d)를 추가하면 AUC 0.9591 → 0.9763으로 **+0.017** 향상. 이미지 모달리티의 명확한 기여를 확인.
- **A6 → A7**: p_compatible 피처(크로스도메인 신호) 추가로 AUC 0.9762 → 0.9793으로 **+0.003** 추가 향상. Polyvore 패션 호환성 지식이 H&M 예측에 유효함을 입증.

#### Group 4: 전이 학습 분석

| | T1 (랜덤 초기화) | T2 (Polyvore 초기화) |
|---|---|---|
| proj 초기화 | 랜덤 | Stage1 가중치 |
| proj LR | 1e-4 | 3e-5 (소폭 fine-tune) |
| p_compat 피처 | ✅ 사용 | ✅ 사용 |
| AUC | **0.9829** | 0.9793 |

- **T1 > T2**: Polyvore(패션 전문가 호환성)와 H&M(실제 구매 이력) 간의 도메인 갭으로 인해, proj 가중치를 Polyvore로 초기화하고 낮은 LR로 제한적 fine-tune하는 것이 오히려 H&M 도메인에서 제약이 됨.
- **핵심 발견**: Polyvore 지식은 **p_compatible 피처**로 활용하는 것이 효과적이며, 가중치 직접 전이는 도메인 갭으로 인해 성능 저하를 유발한다.

---

## 6. 주요 발견 및 결론

### 6.1 멀티모달 통합의 효과

단일 모달리티 최고 성능(A4_Image: 0.9672) 대비 멀티모달 최고(T1: 0.9834)는 **+0.016 AUC** 향상을 달성했다. 텍스트와 이미지 정보가 상호 보완적임을 확인했다.

### 6.2 크로스도메인 전이의 한계와 가능성

- **가중치 전이**: Polyvore → H&M 직접 가중치 전이는 도메인 갭으로 인해 효과 없음
- **피처 전이(p_compat)**: Polyvore 모델의 예측 점수를 추가 입력 피처로 사용하는 방식은 유효 (+0.003 AUC)
- **결론**: 크로스도메인 지식은 가중치가 아닌 **피처로 전달**하는 것이 더 효과적

### 6.3 최종 모델 (T1_MM_NoTransfer)

**AUC=0.9834, PR-AUC=0.9731, F1=0.9115**

- 2411d 멀티모달 특징 (TF-IDF + BERT + Struct + MobileNetV3)
- Polyvore Stage1 모델로 계산한 p_compatible 피처 포함
- proj 랜덤 초기화, 단일 LR=1e-4로 H&M 데이터에 완전 최적화
- 랜덤 베이스라인(B1: 0.5033) 대비 **+0.4796 AUC** 향상

