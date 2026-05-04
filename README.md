# FashFusion
> 이미지와 텍스트 멀티모달 퓨전을 활용한 아이템 적합도 분석 및 구매 전환 확률 예측 시스템

**팀명** : QuadrupleNet | **2026학년도 1학기** | 빅데이터프로그래밍[A]

---

## 프로젝트 개요

장바구니에 담긴 의류 조합과 잡화 간의 스타일 적합도를 분석하여,  
사용자의 실제 구매 전환 확률을 예측하는 멀티모달 AI 모델입니다.

---

## 모델 아키텍처

```
[이미지]  →  MobileNet (Frozen)  →  이미지 벡터 [1280d]  ──┐
                                                             ├──→  MLP  →  구매 확률
[텍스트]  →  TF-IDF              →  텍스트 벡터 [512d]   ──┘
```

| 구성 요소 | 역할 | 방식 |
|-----------|------|------|
| MobileNet V2 | 이미지 벡터 추출 | Frozen (파인튜닝 없음) |
| TF-IDF | 텍스트 벡터 추출 | Synthetic Metadata Text |
| MLP Classifier | 구매 확률 예측 | From Scratch (직접 설계) |

---

## 학습 단계

### Phase 1 — Polyvore 스타일 사전학습
- 실제 outfit 내 아이템 쌍 → Positive
- 랜덤 조합 쌍 → Negative
- **결과: MLP F1-score 0.73**

### Phase 2 — H&M 구매 확률 예측
- 같은 날 함께 구매한 아이템 쌍 → Positive
- 랜덤 쌍 → Negative
- Phase 1 compatibility score 피처로 활용

---

## 성능 비교

| 모델 | F1-score |
|------|----------|
| Random Classifier (베이스라인) | 0.50 |
| Logistic Regression | 0.68 |
| **MLP Classifier (최종)** | **0.73** |

---

## 파일 구조

```
FashFusion/
  polyvore_phase1_outfit.py   # Phase 1: 스타일 사전학습
  phase2_hm_pipeline.py       # Phase 2: 구매 확률 예측
  eda_analysis.py             # EDA 분석 및 시각화
  README.md
  data/                       # ← .gitignore 제외 (직접 다운로드 필요)
    polyvore/
    hnm/
```

---

## 데이터셋 다운로드

### Polyvore Outfits Dataset

```bash
pip install datasets
```

```python
from datasets import load_dataset

dataset = load_dataset("Marqo/polyvore")
dataset.save_to_disk("data/polyvore")
```

### H&M Personalized Fashion Recommendations

1. [kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) 접속
2. Kaggle 계정 로그인
3. **"Join Competition"** 버튼 클릭 (규정 동의)
4. Data 탭에서 다운로드
5. 압축 해제 후 `data/hnm/` 폴더에 저장

---

## 실행 방법

```bash
# 패키지 설치
pip install torch torchvision transformers scikit-learn pandas numpy Pillow datasets matplotlib seaborn joblib

# EDA 분석
python eda_analysis.py

# Phase 1: 스타일 사전학습
python polyvore_phase1_outfit.py

# Phase 2: 구매 확률 예측
python phase2_hm_pipeline.py
```

---

## 팀원

| 이름 | 역할 |
|------|------|
| 하승한 | 실험 및 성능 분석 |
| 주희연 | 데이터 수집 및 전처리 |
| 최동렬 | 모델 설계 및 구현 |
| 송지원 | 문서 및 발표 |
