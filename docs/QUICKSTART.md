# 🚀 빠른 시작 가이드

## 1단계: 데이터 준비

```bash
cd /home/eastj/LAIDD

# data 폴더 생성 (이미 있다면 생략)
mkdir -p data

# 데이터 파일을 data 폴더에 복사
# - generative_data.csv
# - pretraining_data.tsv
# - finetuning_data.tsv
```

## 2단계: 데이터 확인

```bash
python run_all.py --check-only
```

예상 출력:
```
✓ generative_data.csv: XX.XX MB - Generative model training data
✓ pretraining_data.tsv: XX.XX MB - Pretraining data (BindingDB)
✓ finetuning_data.tsv: XX.XX MB - Fine-tuning data (GSK3β)

✓ 모든 데이터 파일이 준비되었습니다.
```

## 3단계: 전체 실행

```bash
# 전체 파이프라인 실행 (Pretraining -> Fine-tuning -> Generative)
python run_all.py --all
```

이 명령어는 다음을 순차적으로 실행합니다:
1. **Pretraining** (~수 시간, GPU 권장)
2. **Fine-tuning** (~수십 분)
3. **Generative Model** (~수 시간)

## 개별 실행

각 단계를 개별적으로 실행할 수도 있습니다:

```bash
# 1. Pretraining만
python run_all.py --pretraining

# 2. Fine-tuning만 (Pretraining 완료 후)
python run_all.py --finetuning

# 3. Generative model만
python run_all.py --generative
```

## 결과 확인

```bash
# 결과 폴더 구조 확인
tree results/

# 또는
ls -R results/
```

예상 출력:
```
results/
├── pretraining/
│   ├── best_model.pt
│   └── final_results.json
├── finetuning/
│   ├── best_gsk3b_model.pt
│   └── gsk3b_finetuning_results.json
└── generativemodel/
    ├── generated_molecules.csv
    └── generation_metrics.json
```

## 생성된 분자 확인

```python
import pandas as pd

# 생성된 분자 로드
df = pd.read_csv('results/generativemodel/generated_molecules.csv')

# 유효한 분자만
valid_df = df[df['valid'] == 1]

# 첫 10개 분자 출력
print(valid_df.head(10))
```

## 재실행

기존 결과를 무시하고 처음부터 다시 실행:

```bash
python run_all.py --all --no-skip
```

## 도움말

```bash
python run_all.py --help
```

## 문제 해결

### CUDA out of memory
→ Batch size 줄이기 (각 스크립트 내 수정)

### Import 오류
→ 필요한 패키지 설치:
```bash
pip install rdkit-pypi torch pandas numpy tqdm matplotlib scikit-learn
```

### 데이터 파일 없음
→ data/ 폴더에 필요한 CSV/TSV 파일 배치

---

**더 자세한 정보는 [README.md](README.md) 참고**
