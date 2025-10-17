# LAIDD 프로젝트 구조 상세 설명

## 📁 전체 구조

```
LAIDD/
├── README.md                           # 메인 문서
├── QUICKSTART.md                       # 빠른 시작 가이드
├── PROJECT_STRUCTURE.md                # 이 문서
├── .gitignore                          # Git 설정
│
├── data/                               # 데이터 (사용자 준비)
│   ├── generative_data.csv
│   ├── pretraining_data.tsv
│   └── finetuning_data.tsv
│
├── src/                                # 소스 코드
│   ├── pretraining/
│   │   └── pretraining_code.py         # Multi-Affinity 학습
│   │
│   ├── finetuning/
│   │   └── finetuning_code.py          # GSK3β 특화 학습
│   │
│   ├── generative/
│   │   └── generativemodel.py          # LSTM 분자 생성
│   │
│   └── utils/
│       └── bs_denovo/                  # BioGen 라이브러리
│           ├── vocab.py
│           ├── lang_data.py
│           ├── lang_lstm.py
│           ├── gen_eval.py
│           └── ...
│
├── scripts/                            # 실행 스크립트
│   ├── run_all.py                      # 통합 실행 (메인)
│   ├── run_all_pipeline.py             # 대체 실행
│   ├── pipeline_config.json            # 설정 파일
│   └── quick_start.sh
│
├── docs/                               # 추가 문서
│   ├── README_PIPELINE.md
│   └── ...
│
└── results/                            # 결과 (자동 생성)
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
        ├── generated_molecules.csv     # ★ 20,000개 신규 분자
        ├── generation_metrics.json
        ├── molecules_split.csv
        ├── my_tokens.txt
        └── training_curves.png
```

## 🎯 주요 파일 설명

### 📄 실행 스크립트

| 파일 | 위치 | 설명 |
|------|------|------|
| **run_all.py** | scripts/ | **메인 실행 스크립트** - 통합 파이프라인 |
| run_all_pipeline.py | scripts/ | 대체 실행 스크립트 (config 파일 기반) |
| quick_start.sh | scripts/ | Bash 스크립트 버전 |

### 💻 소스 코드

| 파일 | 위치 | 설명 |
|------|------|------|
| **pretraining_code.py** | src/pretraining/ | Multi-Affinity Pretraining |
| **finetuning_code.py** | src/finetuning/ | GSK3β Fine-tuning |
| **generativemodel.py** | src/generative/ | LSTM 분자 생성 |
| bs_denovo/ | src/utils/ | BioGen 라이브러리 |

### 📚 문서

| 파일 | 위치 | 설명 |
|------|------|------|
| **README.md** | 루트 | **메인 문서** - 전체 프로젝트 설명 |
| **QUICKSTART.md** | 루트 | 빠른 시작 가이드 |
| **PROJECT_STRUCTURE.md** | 루트 | 이 문서 - 구조 상세 설명 |

## 🔄 실행 흐름

```
1. 데이터 준비
   data/ 폴더에 CSV/TSV 파일 배치
   ↓
2. 데이터 확인
   python scripts/run_all.py --check-only
   ↓
3. Pretraining
   python scripts/run_all.py --pretraining
   → results/pretraining/best_model.pt 생성
   ↓
4. Fine-tuning
   python scripts/run_all.py --finetuning
   → results/finetuning/best_gsk3b_model.pt 생성
   ↓
5. Generative Model
   python scripts/run_all.py --generative
   → results/generativemodel/generated_molecules.csv 생성
```

## 📦 모듈 간 의존성

```
src/finetuning/finetuning_code.py
    ↓ import
src/pretraining/pretraining_code.py
    (MultiAffinityNet, Dataset, metrics 등)

src/generative/generativemodel.py
    ↓ import
src/utils/bs_denovo/
    (SmilesVocabulary, LSTMGenerator 등)
```

## 🛤️ Import 경로

### Pretraining
```python
# src/pretraining/pretraining_code.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# → /LAIDD
```

### Fine-tuning
```python
# src/finetuning/finetuning_code.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pretraining'))
from pretraining_code import MultiAffinityNet, ...
```

### Generative Model
```python
# src/generative/generativemodel.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from utils.bs_denovo.vocab import SmilesVocabulary
```

## 🚀 실행 방법

### 방법 1: 통합 실행 (권장)

```bash
cd /LAIDD
python scripts/run_all.py --all
```

### 방법 2: 개별 실행

```bash
cd /LAIDD
python src/pretraining/pretraining_code.py
python src/finetuning/finetuning_code.py
python src/generative/generativemodel.py
```

## 📊 결과 파일 크기 (예상)

| 파일 | 크기 |
|------|------|
| pretraining/best_model.pt | ~100-500 MB |
| finetuning/best_gsk3b_model.pt | ~100-500 MB |
| generativemodel/generated_molecules.csv | ~2-5 MB |
| generativemodel/models/*.ckpt | ~100-300 MB (각) |

## 🎨 폴더 구조 개선 사항

### ✅ 개선 전 → 후

| 항목 | 개선 전 | 개선 후 |
|------|---------|---------|
| 소스 코드 | 루트에 산재 | `src/`에 모듈별 정리 |
| 실행 스크립트 | 루트 | `scripts/`에 집중 |
| 유틸리티 | `bs_denovo/` | `src/utils/bs_denovo/` |
| 문서 | 혼재 | 루트 + `docs/` |
| 결과 | `results/` | 세부 구조화 |

### 💡 장점

1. **명확한 책임 분리**: 코드/스크립트/문서/결과 분리
2. **쉬운 탐색**: 구조가 직관적
3. **확장 가능**: 새 모듈 추가 용이
4. **버전 관리**: Git에서 관리하기 쉬움

## 🔧 개발 가이드

### 새 모듈 추가

```bash
# 1. 새 폴더 생성
mkdir -p src/new_module

# 2. 코드 작성
cat > src/new_module/new_code.py << 'EOF'
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# 코드 작성...
EOF

# 3. 실행 스크립트에 추가
# scripts/run_all.py에 새 함수 추가
```

### 경로 설정 패턴

```python
# 모든 모듈에서 사용하는 표준 패턴
import os

# 프로젝트 루트 계산
PROJECT_ROOT = os.path.dirname(  # LAIDD/
    os.path.dirname(              # src/
        os.path.dirname(          # module/
            __file__              # script.py
        )
    )
)

# 데이터 경로
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "file.csv")

# 결과 경로
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "module_name")
os.makedirs(OUT_DIR, exist_ok=True)
```

## 📝 데이터 파일 형식

### generative_data.csv
```csv
ID,SMILES
mol_001,CCO
mol_002,CCN
...
```

### pretraining_data.tsv
```tsv
Standardized_SMILES\tStandard Type\tpChEMBL_Value\tTarget_Sequence
CCO\tIC50\t7.5\tMKVLWA...
...
```

### finetuning_data.tsv
```tsv
Ligand SMILES\tBindingDB Target Chain Sequence\tStandard Type\tpChEMBL
CCO\tMKVLWA...\tIC50\t7.5
...
```

## 🎯 모범 사례

### ✅ DO

```bash
# 항상 프로젝트 루트에서 실행
cd /LAIDD
python scripts/run_all.py --all
```

### ❌ DON'T

```bash
# 서브폴더에서 직접 실행하지 않기
cd /LAIDD/src/pretraining
python pretraining_code.py  # ← Import 오류 발생 가능
```

## 💻 IDE 설정

### VSCode

```json
{
  "python.analysis.extraPaths": [
    "${workspaceFolder}/src"
  ]
}
```

### PyCharm

- Mark `src/` as Sources Root

## 📞 지원

문제 발생 시:
1. [README.md](README.md)의 트러블슈팅 확인
2. 프로젝트 루트에서 실행 확인
3. 경로 설정 확인

---

**버전**: 2.0 (폴더 구조 개선)
**업데이트**: 2025-10-17
