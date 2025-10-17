# LAIDD - Learning-based AI Drug Discovery

딥러닝 기반 신약 개발 통합 파이프라인

## 📋 목차

- [개요](#개요)
- [프로젝트 구조](#프로젝트-구조)
- [설치](#설치)
- [빠른 시작](#빠른-시작)
- [사용법](#사용법)
- [각 모듈 설명](#각-모듈-설명)
- [결과 확인](#결과-확인)
- [트러블슈팅](#트러블슈팅)

## 🎯 개요

LAIDD는 다음 3단계로 구성된 신약 개발 파이프라인입니다:

1. **Pretraining**: BindingDB 데이터를 이용한 Multi-Affinity 학습
2. **Fine-tuning**: GSK3β 타겟 특화 모델 학습
3. **Generative Model**: LSTM 기반 신규 분자 생성 (**20,000개**)

## 📂 프로젝트 구조

```
LAIDD/
├── README.md                          # 이 문서
├── QUICKSTART.md                      # 빠른 시작 가이드
├── PROJECT_STRUCTURE.md               # 상세 구조 설명
│
├── data/                              # 데이터 폴더 (사용자 준비)
│   ├── generative_data.csv
│   ├── pretraining_data.tsv
│   └── finetuning_data.tsv
│
├── src/                               # 소스 코드
│   ├── pretraining/
│   │   └── pretraining_code.py        # Multi-Affinity Pretraining
│   ├── finetuning/
│   │   └── finetuning_code.py         # GSK3β Fine-tuning
│   ├── generative/
│   │   └── generativemodel.py         # LSTM 분자 생성
│   └── utils/
│       └── bs_denovo/                 # BioGen 라이브러리
│
├── scripts/                           # 실행 스크립트
│   ├── run_all.py                     # 통합 실행 스크립트 (메인)
│   └── ...
│
└── results/                           # 결과 (자동 생성)
    ├── pretraining/                   # Pretraining 결과
    ├── finetuning/                    # Fine-tuning 결과
    └── generativemodel/               # 생성된 신규 화합물
```

## 🔧 설치

### 필수 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (권장)

### 의존성 설치

```bash
# RDKit
pip install rdkit-pypi

# PyTorch (CUDA 버전에 맞게)
pip install torch torchvision torchaudio

# 기타 패키지
pip install pandas numpy tqdm matplotlib scikit-learn
```

## 🚀 빠른 시작

### 1단계: 데이터 준비

```bash
cd /LAIDD

# data 폴더에 다음 파일들 배치:
# - generative_data.csv
# - pretraining_data.tsv
# - finetuning_data.tsv
```

### 2단계: 데이터 확인

```bash
python scripts/run_all.py --check-only
```

### 3단계: 실행

```bash
# 전체 파이프라인 실행
python scripts/run_all.py --all
```

## 📖 사용법

### 방법 1: 통합 실행 (권장)

```bash
# 전체 파이프라인
python scripts/run_all.py --all

# 개별 단계
python scripts/run_all.py --pretraining    # Pretraining만
python scripts/run_all.py --finetuning     # Fine-tuning만
python scripts/run_all.py --generative     # Generative model만

# 기존 결과 무시하고 재실행
python scripts/run_all.py --all --no-skip
```

### 방법 2: 개별 스크립트 실행

```bash
# 프로젝트 루트에서 실행
cd /LAIDD

python src/pretraining/pretraining_code.py
python src/finetuning/finetuning_code.py
python src/generative/generativemodel.py
```

## 🔍 각 모듈 설명

### 1. Pretraining (src/pretraining/)

**목적**: BindingDB 데이터로 Multi-Affinity 학습

**기능**:
- 4가지 affinity 타입 동시 학습 (IC50, Ki, EC50, Kd)
- Scaffold-based 데이터 분할
- Multi-task learning

**출력**:
- `results/pretraining/best_model.pt`
- `results/pretraining/final_results.json`

**하이퍼파라미터**:
- Batch size: 32
- Learning rate: 1e-3
- Epochs: 30

### 2. Fine-tuning (src/finetuning/)

**목적**: GSK3β 타겟 특화 모델

**기능**:
- Pretrained 모델 기반 전이 학습
- 데이터 증강으로 과적합 방지
- Single model / 5-fold CV 지원

**출력**:
- `results/finetuning/best_gsk3b_model.pt`
- `results/finetuning/gsk3b_finetuning_results.json`

**하이퍼파라미터**:
- Batch size: 16
- Learning rate: 1e-4
- Epochs: 50

### 3. Generative Model (src/generative/)

**목적**: LSTM 기반 신규 분자 생성

**기능**:
- SMILES 기반 분자 생성
- Validity, Uniqueness, Novelty 평가

**출력**:
- `results/generativemodel/generated_molecules.csv` (**20,000개 분자**)
- `results/generativemodel/generation_metrics.json`
- `results/generativemodel/training_curves.png`

**하이퍼파라미터**:
- LSTM layers: 3 × 512 units
- Embedding size: 256
- Batch size: 256
- Max epochs: 80

## 📊 결과 확인

### 생성된 분자 확인

```python
import pandas as pd

# 생성된 분자 로드
df = pd.read_csv('results/generativemodel/generated_molecules.csv')

# 유효한 분자만
valid_df = df[df['valid'] == 1]
print(f"Valid molecules: {len(valid_df)} / {len(df)}")

# 상위 10개
print(valid_df[['smiles_raw', 'smiles_can']].head(10))
```

### 메트릭 확인

```python
import json

# Pretraining
with open('results/pretraining/final_results.json') as f:
    pre = json.load(f)
print("Pretraining RMSE:", pre['test_overall']['RMSE'])

# Fine-tuning
with open('results/finetuning/gsk3b_finetuning_results.json') as f:
    fine = json.load(f)
print("Fine-tuning RMSE:", fine['test_metrics']['RMSE'])

# Generative
with open('results/generativemodel/generation_metrics.json') as f:
    gen = json.load(f)
print("Validity:", gen['validity_overall'])
print("Uniqueness:", gen['uniqueness'])
print("Novelty:", gen['novelty'])
```

## 🛠️ 트러블슈팅

### Import 오류

```bash
ModuleNotFoundError: No module named 'utils.bs_denovo'
```

**해결**: 프로젝트 루트에서 실행

```bash
cd /LAIDD
python scripts/run_all.py --all
```

### CUDA Out of Memory

**해결**: Batch size 줄이기

각 스크립트의 메인 함수에서:
```python
batch_size=16  # 32 → 16
```

### RDKit 설치 오류

**해결**: conda 사용

```bash
conda install -c conda-forge rdkit
```

### 데이터 파일 형식

- **CSV**: 쉼표 구분, UTF-8
- **TSV**: 탭 구분, UTF-8
- 필수 컬럼명 확인 필요

## 📝 추가 문서

- **[QUICKSTART.md](QUICKSTART.md)**: 빠른 시작 가이드
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: 상세 구조 설명
- **[docs/](docs/)**: 추가 문서

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

## 📄 라이센스

교육 및 연구 목적으로 사용됩니다.

---

**버전**: 2.0 (폴더 구조 개선)
**업데이트**: 2025-10-17

**LAIDD Team**

