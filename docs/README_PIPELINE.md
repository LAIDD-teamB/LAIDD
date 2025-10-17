# LAIDD Unified Pipeline

이 문서는 LAIDD 프로젝트의 통합 파이프라인 사용법을 설명합니다.

## 개요

LAIDD 통합 파이프라인은 다음 3단계를 자동으로 실행합니다:

1. **Pretraining**: Multi-affinity 모델 사전학습 (BindingDB 데이터)
2. **Finetuning**: GSK3β 특화 fine-tuning
3. **Generative Model**: LSTM 기반 분자 생성 모델 학습 (선택사항)

## 파일 구조

```
LAIDD/
├── run_all_pipeline.py          # 통합 파이프라인 실행 스크립트
├── pipeline_config.json          # 파이프라인 설정 파일
├── pretraining_code.py           # Pretraining 코드
├── finetuning_code.py            # Finetuning 코드
├── multi_affinity_bindingdb.py   # 공통 모델 정의
├── bs_denovo/                    # Generative model 모듈
│   ├── lang_lstm.py
│   ├── gen_eval.py
│   └── ...
└── data/
    ├── pretraining_data.tsv      # Pretraining 데이터
    └── finetuning_data.tsv       # Finetuning 데이터
```

## 사용법

### 1. 기본 사용 (모든 단계 실행)

```bash
python run_all_pipeline.py
```

기본 설정으로 pretraining과 finetuning을 순차적으로 실행합니다.

### 2. 설정 파일을 사용한 실행

```bash
python run_all_pipeline.py --config pipeline_config.json
```

### 3. 기본 설정 파일 생성

```bash
python run_all_pipeline.py --create-config
```

이 명령어는 `pipeline_config.json` 파일을 생성합니다. 이 파일을 수정하여 파이프라인을 커스터마이즈할 수 있습니다.

### 4. 특정 단계 건너뛰기

```bash
# Pretraining 건너뛰기 (기존 모델 사용)
python run_all_pipeline.py --skip-pretraining

# Finetuning 건너뛰기
python run_all_pipeline.py --skip-finetuning

# Generative model 활성화
python run_all_pipeline.py --enable-generative
```

## 설정 파일 (pipeline_config.json)

### 전체 구조

```json
{
  "output_root": "./pipeline_results",
  "run_pretraining": true,
  "run_finetuning": true,
  "run_generative": false,

  "pretraining": { ... },
  "finetuning": { ... },
  "generative": { ... }
}
```

### Pretraining 설정

```json
"pretraining": {
  "data_path": "./data/pretraining_data.tsv",
  "batch_size": 32,
  "max_len_smiles": 256,
  "max_len_seq": 1500,
  "epochs": 30,
  "lr": 0.001,
  "weight_decay": 1e-05,
  "use_scaffold": true,
  "grad_accum": 4,
  "warmup_steps": 1000,
  "save_every": 5,
  "target_seq_column": "BindingDB Target Chain Sequence"
}
```

주요 파라미터:
- `data_path`: Pretraining 데이터 경로
- `epochs`: 학습 에포크 수
- `batch_size`: 배치 크기
- `use_scaffold`: Scaffold split 사용 여부

### Finetuning 설정

```json
"finetuning": {
  "data_path": "./data/finetuning_data.tsv",
  "pretrained_model_path": null,
  "use_5fold": false,
  "epochs": 50,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "batch_size": 16,
  "early_stopping_patience": 15,
  "validation_split": 0.2,
  "test_split": 0.1,
  "freeze_encoder": false,
  "data_augmentation": true
}
```

주요 파라미터:
- `pretrained_model_path`: 사전학습 모델 경로 (null이면 pretraining 단계 출력 사용)
- `use_5fold`: 5-fold cross-validation 사용 여부
- `freeze_encoder`: Encoder 레이어 동결 여부
- `data_augmentation`: 데이터 증강 사용 여부

### Generative Model 설정

```json
"generative": {
  "training_script": null,
  "timeout": 3600
}
```

현재는 비활성화되어 있으며, 별도의 training script를 지정하여 사용할 수 있습니다.

## 출력 결과

파이프라인 실행 후 다음과 같은 디렉토리 구조가 생성됩니다:

```
pipeline_results/
├── pipeline.log                  # 실행 로그
├── pipeline_config.json          # 사용된 설정
├── pipeline_results.json         # 최종 결과 요약
├── 1_pretraining/
│   ├── best_model.pt            # Best pretraining model
│   ├── final_results.json       # Pretraining 결과
│   ├── train_split.tsv
│   ├── val_split.tsv
│   └── test_split.tsv
├── 2_finetuning/
│   ├── best_gsk3b_model.pt      # Best finetuning model
│   ├── gsk3b_finetuning_results.json
│   ├── train_split.tsv
│   ├── val_split.tsv
│   └── test_split.tsv
└── 3_generative/
    └── (generative model 출력)
```

## 데이터 준비

### Pretraining 데이터 형식

TSV 파일이며 다음 컬럼이 필요합니다:

- `Standardized_SMILES`: SMILES 문자열
- `BindingDB Target Chain Sequence`: 단백질 서열
- `Standard Type`: Affinity 타입 (IC50, Ki, EC50, Kd)
- `pChEMBL_Value`: pChEMBL 값 (음수 로그 변환된 affinity)

### Finetuning 데이터 형식

TSV 파일이며 다음 컬럼이 필요합니다:

- `Ligand SMILES`: SMILES 문자열
- `BindingDB Target Chain Sequence`: GSK3β 단백질 서열
- `Standard Type`: Affinity 타입
- `pChEMBL`: pChEMBL 값

## 예제

### 예제 1: 전체 파이프라인 실행

```bash
# 1. 데이터 준비
mkdir -p data
# (데이터 파일을 data/ 폴더에 배치)

# 2. 파이프라인 실행
python run_all_pipeline.py

# 3. 결과 확인
ls pipeline_results/
cat pipeline_results/pipeline_results.json
```

### 예제 2: 커스터마이즈된 설정으로 실행

```bash
# 1. 기본 설정 파일 생성
python run_all_pipeline.py --create-config

# 2. pipeline_config.json 파일 수정
# (에디터로 열어서 파라미터 변경)

# 3. 수정된 설정으로 실행
python run_all_pipeline.py --config pipeline_config.json
```

### 예제 3: Finetuning만 실행 (기존 pretraining 모델 사용)

```bash
# pipeline_config.json에서 설정 변경:
# "run_pretraining": false
# "finetuning.pretrained_model_path": "./path/to/pretrained_model.pt"

python run_all_pipeline.py --skip-pretraining
```

### 예제 4: 5-fold cross-validation으로 finetuning

```bash
# pipeline_config.json에서 설정 변경:
# "finetuning.use_5fold": true

python run_all_pipeline.py
```

## 트러블슈팅

### 문제 1: 데이터 파일을 찾을 수 없음

**증상**: `FileNotFoundError: Pretraining data not found`

**해결**:
```bash
# 데이터 경로 확인
ls data/pretraining_data.tsv
ls data/finetuning_data.tsv

# 또는 pipeline_config.json에서 경로 수정
```

### 문제 2: GPU 메모리 부족

**증상**: `CUDA out of memory`

**해결**: `pipeline_config.json`에서 batch size 줄이기
```json
"pretraining": {
  "batch_size": 16  // 32에서 16으로 감소
}
```

### 문제 3: Pretraining 모델 로딩 실패

**증상**: `Pretrained model not found`

**해결**:
```bash
# pretraining을 먼저 실행하거나
python run_all_pipeline.py --skip-finetuning

# 또는 기존 모델 경로를 명시
# pipeline_config.json:
# "finetuning.pretrained_model_path": "./path/to/model.pt"
```

## 고급 사용법

### 개별 스크립트 직접 실행

각 단계를 개별적으로 실행할 수도 있습니다:

```bash
# Pretraining만
python pretraining_code.py

# Finetuning만
python finetuning_code.py
```

### 파이프라인 결과 분석

```python
import json

# 결과 로드
with open('pipeline_results/pipeline_results.json') as f:
    results = json.load(f)

# Pretraining 성능
print(f"Pretraining Test RMSE: {results['pretraining']['test_metrics']['RMSE']:.3f}")

# Finetuning 성능
print(f"Finetuning Test RMSE: {results['finetuning']['test_metrics']['RMSE']:.3f}")
```

## 참고사항

1. **GPU 사용 권장**: 대규모 데이터셋의 경우 GPU 사용을 권장합니다.

2. **데이터 크기**: Pretraining은 대량의 데이터(수만~수십만 샘플), Finetuning은 소량의 특화 데이터를 사용합니다.

3. **실행 시간**:
   - Pretraining: 수 시간 ~ 수일 (데이터 크기에 따라)
   - Finetuning: 30분 ~ 수 시간

4. **체크포인트**: 각 단계의 best model이 자동으로 저장되므로 중간에 중단되어도 재시작 가능합니다.

## 라이선스 및 인용

이 코드를 사용하는 경우 적절한 인용을 부탁드립니다.

## 문의

문제가 발생하거나 질문이 있는 경우 이슈를 등록해주세요.
