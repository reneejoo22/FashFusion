# FashFusion — 코드 파일 가이드

## 전체 실행 순서

```
[데이터 수집]
kaggle_download_zip.py

[전처리 파이프라인]
step0 → step1 → step2 → step3a/3b → step3c → step4 → step5

[멀티모달 모델 학습]
step6mm → step7mm → step8mm

[실험 및 평가]
full_experiments_final.py
```

---

## 📁 데이터 수집

### `kaggle_download_zip.py`
H&M 데이터셋을 Kaggle API를 통해 다운로드한다.

- **역할**: Kaggle 대회 전체 아카이브(~34GB)를 스트리밍 다운로드 후 `images/` 디렉토리만 추출
- **출력**: `data/hnm/images/` (H&M 상품 이미지)
- **비고**: Bearer 토큰 인증 방식 사용. 대용량 파일의 Resume 다운로드 지원

---

## 📁 전처리 파이프라인

단계별로 순서대로 실행해야 하며, 각 단계의 출력이 다음 단계의 입력이 된다.

### `step0_id_mapping.py`
아이템 ID를 정수 인덱스로 매핑하고, TF-IDF 희소 행렬을 생성한다.

- **역할**: 원시 JSON/CSV의 문자열 ID → 정수 인덱스 변환 테이블 생성 + 원본 20,000차원 TF-IDF 구성
- **입력**: Polyvore JSON 파일들, H&M `articles.csv`
- **출력**:
  - `item_id_to_idx.pkl` — Polyvore 아이템 ID 매핑
  - `article_id_to_idx.pkl` — H&M 아이템 ID 매핑
  - `poly_item_tfidf.npz` (142,480 × 20,000) — Polyvore TF-IDF 희소 행렬
  - `hnm_tfidf_features_final.npz` (105,542 × 20,000) — H&M TF-IDF 희소 행렬

---

### `step1_svd_reduction.py`
20,000차원 TF-IDF를 Truncated SVD로 300차원으로 축소한다.

- **역할**: 고차원 희소 TF-IDF → 밀집(dense) 300차원 벡터. L2 정규화 적용
- **입력**: `poly_item_tfidf.npz`, `hnm_tfidf_features_final.npz`
- **출력**:
  - `poly_tfidf_300.npy` (142,480 × 300)
  - `hnm_tfidf_300.npy` (105,542 × 300)
  - `poly_svd_model.pkl`, `hnm_svd_model.pkl` — 학습된 SVD 모델

---

### `step2_bert_embeddings.py`
DistilBERT를 사용해 아이템 텍스트를 768차원 임베딩으로 변환한다.

- **역할**: 상품명, 설명 등 텍스트 필드를 `distilbert-base-uncased` CLS 토큰으로 임베딩. Attention mask 기반 mean pooling 적용. 가중치 동결(frozen) 사용
- **입력**: Polyvore/H&M 텍스트 데이터
- **출력**:
  - `poly_bert_768.npy` (142,480 × 768)
  - `hnm_bert_768.npy` (105,542 × 768)

---

### `step3a_poly_mobilenet.py`
Polyvore 상품 이미지에서 MobileNetV3-Large로 1,280차원 이미지 특징을 추출한다.

- **역할**: `features → avgpool → classifier[0](Linear 960→1280) → classifier[1](Hardswish)` 순으로 처리하여 1,280차원 추출. 이미지가 없는 아이템은 영벡터 처리
- **입력**: Polyvore 이미지 디렉토리
- **출력**: `poly_img_1280.npy` (142,480 × 1,280)
- **커버리지**: 142,480개 중 27,372개 이미지 보유 (19.2%)
- **주의**: `num_workers=0` 필수 (Windows DataLoader 데드락 방지)

---

### `step3b_hnm_mobilenet.py`
H&M 상품 이미지에서 동일한 MobileNetV3-Large로 1,280차원 이미지 특징을 추출한다.

- **역할**: step3a와 동일한 아키텍처로 H&M 이미지 처리. 두 도메인의 이미지 표현 공간 통일
- **입력**: `data/hnm/images/`
- **출력**: `hnm_img_1280.npy` (105,542 × 1,280)
- **커버리지**: 105,542개 중 105,100개 이미지 보유 (99.6%)

---

### `step3c_rebuild_pairs.py`
4가지 모달리티를 결합한 2,411차원 통합 특징 행렬을 생성하고, 페어 파일을 인덱스 기반으로 재구성한다.

- **역할**: `[tfidf_300 | bert_768 | struct_63 | img_1280]` 연접 → 2,411차원 통합 특징 행렬 저장. 기존 텍스트 특징 직접 저장 방식의 페어 파일을 인덱스 기반으로 재설계 (ZIP64 4GB 한계 우회)
- **입력**: `poly_tfidf_300.npy`, `poly_bert_768.npy`, `poly_struct.npy`, `poly_img_1280.npy`, 같은 구조의 H&M 파일들, 기존 페어 파일들
- **출력**:
  - `poly_mm_feats.npy` (142,480 × 2,411) — Polyvore 통합 특징 행렬
  - `hnm_mm_feats.npy` (105,542 × 2,411) — H&M 통합 특징 행렬
  - `poly_pairs_mm_{train|val|test}.pt` — 인덱스 기반 Polyvore 페어 (~20MB)
  - `hnm_pairs_mm_{train|val}.pt` — 인덱스 기반 H&M 페어 (~20MB)

---

### `step4_structured_features.py`
카테고리, 색상, 가격 등 메타데이터를 수치화하여 구조적 특징 벡터를 생성한다.

- **역할**: 범주형 변수 원-핫 인코딩, 수치형 변수 정규화. 두 도메인의 구조적 특징을 각각 63차원으로 통일
- **입력**: Polyvore/H&M 메타데이터
- **출력**:
  - `poly_struct.npy` (142,480 × 63)
  - `hnm_struct_compat63.npy` (105,542 × 63)

---

### `step5_pair_dataset.py`
Polyvore 데이터에서 Stage 1 학습용 텍스트 기반 페어 데이터셋을 생성한다.

- **역할**: 동일 코디 내 아이템 → 양성 페어, 동일 카테고리 다른 코디 → 하드 네거티브, 랜덤 크로스 코디 → 이지 네거티브로 구성. 1,131차원(텍스트 전용) 특징 직접 저장
- **입력**: Polyvore JSON 파일들, `poly_tfidf_300.npy`, `poly_bert_768.npy`, `poly_struct.npy`
- **출력**: `poly_pairs_{train|val|test}.pt`, `hnm_pairs_{train|val}.pt`

---

## 📁 멀티모달 모델 학습

### `step6mm_stage1_train.py`
Polyvore 데이터로 **Stage 1 멀티모달 모델**을 학습한다.

- **역할**: 패션 전문가 큐레이션 호환성 레이블로 ProjectionMLP(2411→256) + FusionMLP(1030→1) 학습. 학습된 가중치를 Stage 2 초기화에 활용
- **입력**: `poly_mm_feats.npy`, `poly_pairs_mm_{train|val|test}.pt`
- **출력**: `stage1_mm_model.pt`
- **결과**: Val AUC=0.8763

---

### `step7mm_hnm_pcompat.py`
Stage 1 모델로 H&M 전체 페어의 **p_compatible 점수**를 계산한다.

- **역할**: 학습된 Stage 1 모델을 H&M 페어에 적용하여 크로스도메인 패션 호환성 점수 계산 후 페어 파일에 주입
- **입력**: `stage1_mm_model.pt`, `hnm_mm_feats.npy`, `hnm_pairs_mm_{train|val}.pt`
- **출력**: `hnm_p_compat_mm.npy`, 업데이트된 `hnm_pairs_mm_{train|val}.pt`
- **검증**: 양성 페어 평균 0.7213 vs 음성 페어 0.6587 (크로스도메인 신호 유효성 확인)

---

### `step8mm_stage2_train.py`
H&M 구매 이력으로 **Stage 2 멀티모달 모델(T2)** 을 학습한다.

- **역할**: Stage 1 proj 가중치로 초기화 후 Differential LR(proj=3e-5, fusion=1e-4)로 fine-tuning. p_compat 포함 1,031차원 페어 특징 사용
- **입력**: `hnm_mm_feats.npy`, `hnm_pairs_mm_{train|val}.pt` (p_compat 포함), `stage1_mm_model.pt`
- **출력**: `stage2_mm_model.pt` **(공식 T2 모델)**
- **결과**: Val AUC=0.9793, PR-AUC=0.9675, F1=0.9104

---

### `train_t1_model.py`
p_compat은 사용하되 Polyvore 가중치 전이 없이 **T1 모델**을 학습한다.

- **역할**: ProjectionMLP를 랜덤 초기화하고 단일 LR=1e-4로 H&M 데이터에 완전 자유 최적화. T2와의 차이는 proj 초기화 방식뿐
- **입력**: `hnm_mm_feats.npy`, `hnm_pairs_mm_{train|val}.pt`
- **출력**: `stage2_t1_model.pt` **(공식 T1 모델)**
- **결과**: Val AUC=0.9826~0.9834 (랜덤 초기화로 실행마다 미세 변동)

---

## 📁 실험 및 평가

### `full_experiments_final.py` ⭐ (메인 실험 파일)
14개 모델을 동일 val set에서 일괄 평가하는 **통합 실험 스크립트**다.

- **역할**: Baseline(5개) + Single-modal(4개) + Multi-modal(3개) + Transfer(2개) 모델을 공정하게 비교
- **입력**: 모든 특징 파일(.npy), 페어 파일(.pt), 기존 체크포인트
- **출력**: `full_experiment_results.csv`
- **평가 지표**: AUC-ROC, PR-AUC, F1-score

---

---

## 📁 구버전 / 미사용 파일 (Deprecated)

현재 파이프라인에서는 사용하지 않으며, 개발 과정에서 생성된 파일들이다.

| 파일 | 이전 역할 | 대체 파일 |
|---|---|---|
| `step6_stage1_train.py` | Stage 1 학습 (텍스트 전용) | `step6mm_stage1_train.py` |
| `step7_hnm_pcompat.py` | p_compat 계산 (텍스트 전용) | `step7mm_hnm_pcompat.py` |
| `step8_stage2_train.py` | Stage 2 학습 (텍스트 전용) | `step8mm_stage2_train.py` |
| `full_experiments.py` | 통합 실험 (A5에서 크래시) | `full_experiments_final.py` |
| `full_experiments_part2.py` | Groups 3~4 실험 (임시) | `full_experiments_final.py` |
| `ablation_experiments.py` | 초기 ablation 실험 | `full_experiments_final.py` |
| `feature.py`, `feature2.py` | 초기 특징 추출 시도 | step0~step4 |
| `experiment.py` | 초기 실험 스크립트 | `full_experiments_final.py` |
| `kaggle_download.py` | Kaggle 파일별 다운로드 (실패) | `kaggle_download_zip.py` |
| `kaggle_check.py` | API 토큰 확인용 | — |

---

## 📁 주요 산출 파일 (체크포인트 & 데이터)

| 파일 | 설명 | 크기 |
|---|---|---|
| `poly_mm_feats.npy` | Polyvore 통합 특징 행렬 (142,480 × 2,411) | ~1.37 GB |
| `hnm_mm_feats.npy` | H&M 통합 특징 행렬 (105,542 × 2,411) | ~1.02 GB |
| `stage1_mm_model.pt` | Stage 1 학습 체크포인트 | 수 MB |
| `stage2_mm_model.pt` | **공식 T2 모델** 체크포인트 | 수 MB |
| `stage2_t1_model.pt` | **공식 T1 모델** 체크포인트 | 수 MB |
| `full_experiment_results.csv` | 14개 모델 최종 평가 결과 | — |
