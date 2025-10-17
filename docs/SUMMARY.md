# LAIDD 프로젝트 요약

## 🎯 완료된 작업

### 1. 코드 표준화 ✅
- **경로 표준화**: 모든 하드코딩된 경로를 상대 경로로 변경
  - `PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))` 사용
  - 사용자 환경에 독립적으로 실행 가능

### 2. 결과 폴더 구조화 ✅
모든 결과가 체계적으로 저장됩니다:

```
results/
├── pretraining/          # Pretraining 결과
│   ├── best_model.pt
│   ├── train_split.tsv
│   ├── val_split.tsv
│   ├── test_split.tsv
│   └── final_results.json
│
├── finetuning/           # Fine-tuning 결과
│   ├── best_gsk3b_model.pt
│   ├── train_split.tsv
│   ├── val_split.tsv
│   ├── test_split.tsv
│   └── gsk3b_finetuning_results.json
│
└── generativemodel/      # Generative model 결과
    ├── models/
    │   └── lstm_e*.ckpt
    ├── generated_molecules.csv  # 생성된 신규 화합물
    ├── generation_metrics.json
    ├── molecules_split.csv
    ├── my_tokens.txt
    └── training_curves.png
```

### 3. 통합 실행 스크립트 ✅
[run_all.py](run_all.py)를 통한 원클릭 실행:

```bash
# 전체 파이프라인
python run_all.py --all

# 개별 단계
python run_all.py --pretraining
python run_all.py --finetuning
python run_all.py --generative

# 데이터 확인
python run_all.py --check-only
```

### 4. Import 문제 해결 ✅
- `finetuning_code.py`의 import를 `pretraining_code`에서 가져오도록 수정
- 모든 모듈 간 의존성 해결
- 프로젝트 루트 기준 상대 import 사용

### 5. 문서화 ✅
- [README.md](README.md): 전체 프로젝트 문서
- [QUICKSTART.md](QUICKSTART.md): 빠른 시작 가이드
- [SUMMARY.md](SUMMARY.md): 이 문서

## 📋 주요 변경사항

### generativemodel.py
```python
# Before
DATA_CSV = '/home/kaist/projects/BioGen/molecules_unified.csv'
OUT_SAMPLES = '/home/kaist/projects/BioGen/out/samples_single.csv'

# After
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(PROJECT_ROOT, 'data', 'generative_data.csv')
OUT_SAMPLES = os.path.join(output_dir, 'generated_molecules.csv')
```

### pretraining_code.py
```python
# Before
TSV_PATH = "./data/pretraining_data.tsv"
out_dir="./results"

# After
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TSV_PATH = os.path.join(PROJECT_ROOT, "data", "pretraining_data.tsv")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "pretraining")
```

### finetuning_code.py
```python
# Before
from multi_affinity_bindingdb import (...)
PRETRAINED_MODEL_PATH = "best_model.pt"

# After
from pretraining_code import (...)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_MODEL_PATH = os.path.join(PROJECT_ROOT, "results", "pretraining", "best_model.pt")
```

## 🚀 사용 방법

### 단계별 실행

#### 1. 데이터 준비
```bash
cd /home/eastj/LAIDD
mkdir -p data

# 데이터 파일 복사
cp /path/to/generative_data.csv data/
cp /path/to/pretraining_data.tsv data/
cp /path/to/finetuning_data.tsv data/
```

#### 2. 데이터 확인
```bash
python run_all.py --check-only
```

#### 3. 실행
```bash
# 전체 파이프라인 (권장)
python run_all.py --all

# 또는 개별 실행
python run_all.py --pretraining   # Step 1
python run_all.py --finetuning    # Step 2
python run_all.py --generative    # Step 3
```

## 📊 예상 결과

### Pretraining 완료 후
- `results/pretraining/best_model.pt`: 약 100-500MB
- 학습된 모델이 4가지 affinity 타입 예측 가능

### Fine-tuning 완료 후
- `results/finetuning/best_gsk3b_model.pt`
- GSK3β 타겟에 특화된 모델

### Generative Model 완료 후
- `results/generativemodel/generated_molecules.csv`
- **20,000개의 신규 화합물 생성**
- Validity, Uniqueness, Novelty 메트릭 포함

## 🔧 트러블슈팅

### 1. ImportError
```bash
# bs_denovo 모듈 확인
ls bs_denovo/

# 없다면 해당 디렉토리 확인 필요
```

### 2. CUDA Out of Memory
각 스크립트에서 batch_size 조정:
```python
# pretraining_code.py
batch_size=16  # 32에서 16으로

# finetuning_code.py
batch_size=8   # 16에서 8로
```

### 3. 데이터 파일 형식
- CSV: 쉼표로 구분, UTF-8 인코딩
- TSV: 탭으로 구분, UTF-8 인코딩
- 필수 컬럼명 확인 (README 참조)

## 📈 성능 벤치마크

### 환경
- GPU: NVIDIA A100 (40GB) 권장
- CPU: 16+ cores 권장
- RAM: 32GB+ 권장

### 예상 실행 시간
- Pretraining: 2-4시간 (데이터 크기 의존)
- Fine-tuning: 30분-1시간
- Generative Model: 2-3시간

## 🎁 추가 기능

### 커스텀 데이터 사용
```bash
python generativemodel.py --data /path/to/custom.csv --output /path/to/output
```

### 5-fold Cross-validation
```python
# finetuning_code.py 실행 시 옵션 2 선택
python finetuning_code.py
# > 2 (5-fold 선택)
```

## 📦 프로젝트 배포

### 필요 파일
```
LAIDD/
├── run_all.py                 # 필수
├── pretraining_code.py        # 필수
├── finetuning_code.py         # 필수
├── generativemodel.py         # 필수
├── bs_denovo/                 # 필수
├── README.md                  # 권장
├── QUICKSTART.md              # 권장
└── data/                      # 사용자 준비
```

### 배포 체크리스트
- [ ] Python 3.8+ 설치
- [ ] 필수 패키지 설치 (requirements.txt)
- [ ] 데이터 파일 준비
- [ ] GPU 환경 확인 (선택)
- [ ] `run_all.py --check-only` 실행
- [ ] `run_all.py --all` 실행

## 🎯 다음 단계

1. **생성된 분자 분석**
   ```python
   import pandas as pd
   df = pd.read_csv('results/generativemodel/generated_molecules.csv')
   valid_df = df[df['valid'] == 1]
   # 추가 분석...
   ```

2. **Fine-tuned 모델로 예측**
   ```python
   import torch
   from pretraining_code import MultiAffinityNet

   model = MultiAffinityNet(...)
   ckpt = torch.load('results/finetuning/best_gsk3b_model.pt')
   model.load_state_dict(ckpt['model'])
   # 예측...
   ```

3. **결과 시각화**
   - 학습 곡선: `results/generativemodel/training_curves.png`
   - 커스텀 시각화 스크립트 작성

## 📞 지원

문제가 발생하면:
1. README.md의 트러블슈팅 섹션 확인
2. 에러 메시지 확인
3. 데이터 파일 형식 재확인

---

**프로젝트 완료일**: 2025-10-17
**버전**: 1.0
