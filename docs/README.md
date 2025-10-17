# LAIDD - Learning-based AI Drug Discovery

딥러닝 기반 신약 개발 통합 파이프라인

## 📋 목차

- [개요](#개요)
- [설치](#설치)
- [데이터 준비](#데이터-준비)
- [사용법](#사용법)
- [프로젝트 구조](#프로젝트-구조)
- [각 모듈 설명](#각-모듈-설명)

## 🎯 개요

LAIDD는 다음 3단계로 구성된 신약 개발 파이프라인입니다:

1. **Pretraining**: BindingDB 데이터를 이용한 Multi-Affinity 학습
2. **Fine-tuning**: GSK3β 타겟 특화 모델 학습
3. **Generative Model**: LSTM 기반 신규 분자 생성

## 🔧 설치

### 필수 요구사항

- Python 3.8 이상
- CUDA (GPU 사용 시)

### 의존성 패키지 설치

```bash
# RDKit 설치
pip install rdkit-pypi

# PyTorch 설치 (CUDA 버전에 맞게)
pip install torch torchvision torchaudio

# 기타 패키지
pip install pandas numpy tqdm matplotlib scikit-learn
```

## 📁 데이터 준비

프로젝트 루트에 `data` 폴더를 생성하고 다음 파일들을 배치하세요:

```
LAIDD/
├── data/
│   ├── generative_data.csv      # 생성 모델 학습 데이터
│   ├── pretraining_data.tsv     # Pretraining 데이터 (BindingDB)
│   └── finetuning_data.tsv      # Fine-tuning 데이터 (GSK3β)
```

### 데이터 파일 형식

**generative_data.csv**:
- 컬럼: `SMILES` 또는 `smiles`, `ID` 또는 `Molecule_ID`

**pretraining_data.tsv**:
- 필수 컬럼: `Standardized_SMILES`, `Standard Type`, `pChEMBL_Value`, `Target_Sequence`
- 지원하는 affinity 타입: IC50, Ki, EC50, Kd

**finetuning_data.tsv**:
- 필수 컬럼: `Ligand SMILES`, `BindingDB Target Chain Sequence`, `Standard Type`, `pChEMBL`

## 🚀 사용법

### 방법 1: 통합 실행 스크립트 (권장)

```bash
# 데이터 파일 확인
python run_all.py --check-only

# 전체 파이프라인 실행
python run_all.py --all

# 개별 단계 실행
python run_all.py --pretraining    # Pretraining만
python run_all.py --finetuning     # Fine-tuning만
python run_all.py --generative     # Generative model만

# 기존 결과 무시하고 재실행
python run_all.py --all --no-skip
```

### 방법 2: 개별 스크립트 실행

```bash
# 1. Pretraining
python pretraining_code.py

# 2. Fine-tuning
python finetuning_code.py

# 3. Generative Model
python generativemodel.py
```

### 커스텀 경로 지정

```bash
# Generative model에 커스텀 데이터 사용
python generativemodel.py --data /path/to/data.csv --output /path/to/output

# 개별 실행 시 경로는 코드 내에서 수정 가능
```

## 📂 프로젝트 구조

```
LAIDD/
├── README.md                    # 이 문서
├── run_all.py                   # 통합 실행 스크립트
├── pretraining_code.py          # Pretraining 모듈
├── finetuning_code.py           # Fine-tuning 모듈
├── generativemodel.py           # Generative model 모듈
│
├── bs_denovo/                   # BioGen 라이브러리
│   ├── vocab.py
│   ├── lang_data.py
│   ├── lang_lstm.py
│   └── gen_eval.py
│
├── data/                        # 데이터 폴더 (사용자가 준비)
│   ├── generative_data.csv
│   ├── pretraining_data.tsv
│   └── finetuning_data.tsv
│
└── results/                     # 실행 결과 (자동 생성)
    ├── pretraining/
    │   ├── best_model.pt
    │   ├── train_split.tsv
    │   ├── val_split.tsv
    │   ├── test_split.tsv
    │   └── final_results.json
    │
    ├── finetuning/
    │   ├── best_gsk3b_model.pt
    │   ├── train_split.tsv
    │   ├── val_split.tsv
    │   ├── test_split.tsv
    │   └── gsk3b_finetuning_results.json
    │
    └── generativemodel/
        ├── models/
        │   └── lstm_e*.ckpt
        ├── generated_molecules.csv
        ├── generation_metrics.json
        ├── molecules_split.csv
        ├── my_tokens.txt
        └── training_curves.png
```

## 🔍 각 모듈 설명

### 1. Pretraining (pretraining_code.py)

**목적**: BindingDB 데이터를 사용한 Multi-Affinity 학습

**주요 기능**:
- 4가지 affinity 타입 동시 학습 (IC50, Ki, EC50, Kd)
- Scaffold-based 데이터 분할
- Multi-task learning을 통한 일반화 성능 향상

**출력**:
- `results/pretraining/best_model.pt`: 최고 성능 모델
- `results/pretraining/final_results.json`: 학습 결과 메트릭
- 데이터 분할 파일 (train/val/test)

**주요 하이퍼파라미터**:
- Batch size: 32
- Learning rate: 1e-3
- Epochs: 30
- Model: Conv-based encoder + Multi-head predictor

### 2. Fine-tuning (finetuning_code.py)

**목적**: GSK3β 타겟에 특화된 모델 학습

**주요 기능**:
- Pretrained 모델 기반 전이 학습
- 데이터 증강을 통한 과적합 방지
- Single model 또는 5-fold cross-validation 지원

**출력**:
- `results/finetuning/best_gsk3b_model.pt`: 최고 성능 모델
- `results/finetuning/gsk3b_finetuning_results.json`: 평가 결과
- 데이터 분할 파일

**주요 하이퍼파라미터**:
- Batch size: 16 (작은 데이터셋)
- Learning rate: 1e-4 (낮은 학습률)
- Epochs: 50
- Early stopping patience: 10

### 3. Generative Model (generativemodel.py)

**목적**: LSTM 기반 신규 분자 생성

**주요 기능**:
- SMILES 기반 분자 생성
- Scaffold-based 데이터 분할
- 생성 분자의 validity, uniqueness, novelty 평가

**출력**:
- `results/generativemodel/generated_molecules.csv`: 생성된 분자 (20,000개)
- `results/generativemodel/generation_metrics.json`: 생성 품질 메트릭
- `results/generativemodel/training_curves.png`: 학습 곡선
- `results/generativemodel/models/`: 학습된 모델 체크포인트

**주요 하이퍼파라미터**:
- LSTM hidden units: [512, 512, 512] (3-layer)
- Embedding size: 256
- Batch size: 256
- Max epochs: 80
- Target loss: 22.0

**평가 메트릭**:
- **Validity**: 생성된 SMILES가 유효한 분자인지
- **Uniqueness**: 중복 없이 다양한 분자 생성
- **Novelty**: 학습 데이터에 없는 새로운 분자
- **Internal Diversity**: 생성된 분자 간 구조적 다양성

## 📊 결과 확인

### Pretraining 결과

```python
import json

with open('results/pretraining/final_results.json', 'r') as f:
    results = json.load(f)

print("Test RMSE:", results['test_overall']['RMSE'])
print("Test R²:", results['test_overall']['R2'])
```

### Fine-tuning 결과

```python
with open('results/finetuning/gsk3b_finetuning_results.json', 'r') as f:
    results = json.load(f)

print("Test RMSE:", results['test_metrics']['RMSE'])
print("Test R²:", results['test_metrics']['R2'])
```

### Generative Model 결과

```python
import pandas as pd

# 생성된 분자 로드
df = pd.read_csv('results/generativemodel/generated_molecules.csv')

# 유효한 분자만 필터링
valid_df = df[df['valid'] == 1]
print(f"Valid molecules: {len(valid_df)} / {len(df)}")

# 메트릭 확인
with open('results/generativemodel/generation_metrics.json', 'r') as f:
    metrics = json.load(f)

print("Validity:", metrics['validity_overall'])
print("Uniqueness:", metrics['uniqueness'])
print("Novelty:", metrics['novelty'])
```

## 🛠️ 트러블슈팅

### 1. Import 오류

```python
ModuleNotFoundError: No module named 'bs_denovo'
```

**해결**: `bs_denovo` 폴더가 프로젝트 루트에 있는지 확인

### 2. CUDA out of memory

**해결**: Batch size를 줄이거나 GPU 메모리가 큰 환경에서 실행

```python
# pretraining_code.py 또는 finetuning_code.py에서
batch_size=16  # 32에서 16으로 감소
```

### 3. RDKit 설치 오류

**해결**: conda 환경 사용 권장

```bash
conda install -c conda-forge rdkit
```

### 4. 데이터 파일 형식 오류

**해결**:
- TSV 파일은 탭으로 구분되어야 함
- 필수 컬럼명이 정확한지 확인
- 인코딩은 UTF-8 사용

## 📝 커스터마이징

### 하이퍼파라미터 수정

각 스크립트의 `main()` 또는 `if __name__ == "__main__":` 섹션에서 하이퍼파라미터를 수정할 수 있습니다.

```python
# pretraining_code.py
model, results = main_multi_affinity_bindingdb(
    tsv_path=TSV_PATH,
    batch_size=64,      # 수정 가능
    epochs=50,          # 수정 가능
    lr=5e-4,            # 수정 가능
    ...
)
```

### 새로운 affinity 타입 추가

```python
# pretraining_code.py
AFFINITY_TYPES = ['IC50', 'Ki', 'EC50', 'Kd', 'NEW_TYPE']  # 새로운 타입 추가
```

### 모델 아키텍처 수정

```python
# pretraining_code.py의 MultiAffinityNet 클래스
model = MultiAffinityNet(
    d_model=512,        # 256에서 512로 증가
    n_layers=3,         # 레이어 수 증가
    dropout=0.2         # Dropout 비율 조정
)
```

## 📈 성능 최적화 팁

1. **GPU 활용**: CUDA 사용 시 학습 속도 10배 이상 향상
2. **Batch size 조정**: GPU 메모리에 맞게 최대한 크게 설정
3. **Gradient accumulation**: 메모리 부족 시 사용
4. **Early stopping**: 불필요한 학습 시간 절약
5. **Mixed precision training**: AMP 사용으로 메모리 절약

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

## 📄 라이센스

이 프로젝트는 교육 및 연구 목적으로 사용됩니다.

## 👥 개발자

LAIDD Team

---

**마지막 업데이트**: 2025-10-17
