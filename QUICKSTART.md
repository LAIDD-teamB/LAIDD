# 🚀 LAIDD 빠른 시작 가이드

3단계로 LAIDD 파이프라인을 실행하세요!

## 📝 사전 준비

- Python 3.8+
- 필수 패키지 설치
- GPU (권장)

## 1️⃣ 데이터 준비

```bash
cd /LAIDD

# data 폴더에 다음 파일 배치:
data/
├── generative_data.csv       # 생성 모델 데이터
├── pretraining_data.tsv      # Pretraining 데이터
└── finetuning_data.tsv       # Fine-tuning 데이터
```

## 2️⃣ 데이터 확인

```bash
python scripts/run_all.py --check-only
```

**예상 출력:**
```
✓ generative_data.csv: XX MB - Generative model training data
✓ pretraining_data.tsv: XX MB - Pretraining data (BindingDB)
✓ finetuning_data.tsv: XX MB - Fine-tuning data (GSK3β)

✓ 모든 데이터 파일이 준비되었습니다.
```

## 3️⃣ 실행

### 전체 파이프라인 (한번에)

```bash
python scripts/run_all.py --all
```

이 명령어는 다음을 순차 실행합니다:
1. **Pretraining** (~2-4시간) → Multi-Affinity 학습
2. **Fine-tuning** (~30분-1시간) → GSK3β 특화
3. **Generative Model** (~2-3시간) → 신규 분자 생성

### 개별 실행

```bash
# 단계별 실행
python scripts/run_all.py --pretraining
python scripts/run_all.py --finetuning
python scripts/run_all.py --generative
```

## 📊 결과 확인

### 폴더 구조

```bash
tree results/

# 또는
ls -R results/
```

**결과:**
```
results/
├── pretraining/
│   ├── best_model.pt                    # Pretraining 모델
│   └── final_results.json
├── finetuning/
│   ├── best_gsk3b_model.pt              # Fine-tuned 모델
│   └── gsk3b_finetuning_results.json
└── generativemodel/
    ├── generated_molecules.csv          # ★ 20,000개 신규 분자
    ├── generation_metrics.json
    └── training_curves.png
```

### 생성된 분자 확인

```python
import pandas as pd

# 생성된 분자 로드
df = pd.read_csv('results/generativemodel/generated_molecules.csv')

# 유효한 분자만
valid = df[df['valid'] == 1]
print(f"생성된 유효 분자: {len(valid)}/{len(df)}")

# 상위 10개 확인
print(valid[['smiles_can']].head(10))
```

## 🔄 재실행

기존 결과 무시하고 처음부터:

```bash
python scripts/run_all.py --all --no-skip
```

## 💡 도움말

```bash
python scripts/run_all.py --help
```

## ⚠️ 문제 해결

### Import 오류
```bash
# 프로젝트 루트에서 실행
cd /LAIDD
python scripts/run_all.py --all
```

### CUDA Out of Memory
→ 각 스크립트에서 `batch_size` 줄이기

### 패키지 설치
```bash
pip install rdkit-pypi torch pandas numpy tqdm matplotlib scikit-learn
```

## 📚 더 알아보기

- **[README.md](README.md)**: 전체 문서
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: 구조 설명

---

**소요 시간**: 전체 약 5-8시간 (GPU 환경)
**생성 결과**: 20,000개 신규 화합물

Happy Drug Discovery! 🎉
