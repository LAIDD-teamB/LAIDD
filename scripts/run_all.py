#!/usr/bin/env python3
"""
통합 실행 스크립트 - LAIDD Pipeline
Pretraining -> Fine-tuning -> Generative Model을 순차적으로 실행합니다.
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

# Project root (상위 디렉토리로 이동)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ASCII Art Logo
LOGO = """
╔═══════════════════════════════════════════════════════════════╗
║                     LAIDD Pipeline v1.0                       ║
║          Drug Discovery with Deep Learning Models             ║
╚═══════════════════════════════════════════════════════════════╝
"""

def print_section(title):
    """섹션 제목 출력"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def check_data_files():
    """필요한 데이터 파일 확인"""
    print_section("데이터 파일 확인")

    data_dir = os.path.join(PROJECT_ROOT, "data")
    required_files = {
        "generative_data.csv": "Generative model training data",
        "pretraining_data.tsv": "Pretraining data (BindingDB)",
        "finetuning_data.tsv": "Fine-tuning data (GSK3β)"
    }

    missing_files = []
    for filename, description in required_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"✓ {filename}: {size_mb:.2f} MB - {description}")
        else:
            print(f"✗ {filename}: MISSING - {description}")
            missing_files.append(filename)

    if missing_files:
        print(f"\n⚠ 경고: {len(missing_files)}개의 데이터 파일이 누락되었습니다.")
        print(f"누락된 파일: {', '.join(missing_files)}")
        print(f"데이터 파일을 {data_dir}에 배치한 후 다시 실행해주세요.")
        return False

    print("\n✓ 모든 데이터 파일이 준비되었습니다.")
    return True

def run_pretraining(skip_if_exists=True):
    """Pretraining 실행"""
    print_section("STEP 1: Pretraining (Multi-Affinity Learning)")

    output_dir = os.path.join(PROJECT_ROOT, "results", "pretraining")
    best_model = os.path.join(output_dir, "best_model.pt")

    if skip_if_exists and os.path.exists(best_model):
        print(f"✓ Pretrained model이 이미 존재합니다: {best_model}")
        print("  건너뛰기... (강제 재실행: --no-skip 옵션 사용)")
        return True

    print("Pretraining 시작...")
    script_path = os.path.join(PROJECT_ROOT, "src", "pretraining", "pretraining_code.py")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=False
        )
        print("\n✓ Pretraining 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Pretraining 실패: {e}")
        return False

def run_finetuning(skip_if_exists=True):
    """Fine-tuning 실행"""
    print_section("STEP 2: Fine-tuning (GSK3β-specific)")

    pretrained_model = os.path.join(PROJECT_ROOT, "results", "pretraining", "best_model.pt")
    if not os.path.exists(pretrained_model):
        print(f"✗ Pretrained model을 찾을 수 없습니다: {pretrained_model}")
        print("  먼저 pretraining을 실행해주세요.")
        return False

    output_dir = os.path.join(PROJECT_ROOT, "results", "finetuning")
    best_model = os.path.join(output_dir, "best_gsk3b_model.pt")

    if skip_if_exists and os.path.exists(best_model):
        print(f"✓ Fine-tuned model이 이미 존재합니다: {best_model}")
        print("  건너뛰기... (강제 재실행: --no-skip 옵션 사용)")
        return True

    print("Fine-tuning 시작...")
    script_path = os.path.join(PROJECT_ROOT, "src", "finetuning", "finetuning_code.py")

    # 사용자 입력 자동화 (choice 1: single model fine-tuning)
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,
            input="1\n",  # Single model fine-tuning 선택
            text=True,
            check=True,
            capture_output=False
        )
        print("\n✓ Fine-tuning 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Fine-tuning 실패: {e}")
        return False

def run_generative_model(skip_if_exists=True):
    """Generative Model 실행"""
    print_section("STEP 3: Generative Model (Molecular Design)")

    output_dir = os.path.join(PROJECT_ROOT, "results", "generativemodel")
    generated_file = os.path.join(output_dir, "generated_molecules.csv")

    if skip_if_exists and os.path.exists(generated_file):
        print(f"✓ Generated molecules이 이미 존재합니다: {generated_file}")
        print("  건너뛰기... (강제 재실행: --no-skip 옵션 사용)")
        return True

    print("Generative Model 학습 및 생성 시작...")
    script_path = os.path.join(PROJECT_ROOT, "src", "generative", "generativemodel.py")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=False
        )
        print("\n✓ Generative Model 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Generative Model 실패: {e}")
        return False

def print_summary():
    """결과 요약 출력"""
    print_section("실행 결과 요약")

    results_dir = os.path.join(PROJECT_ROOT, "results")

    # Pretraining 결과
    print("📁 Pretraining 결과:")
    pretraining_dir = os.path.join(results_dir, "pretraining")
    if os.path.exists(pretraining_dir):
        files = os.listdir(pretraining_dir)
        print(f"   위치: {pretraining_dir}")
        print(f"   파일 개수: {len(files)}")
        if "best_model.pt" in files:
            model_size = os.path.getsize(os.path.join(pretraining_dir, "best_model.pt")) / 1024 / 1024
            print(f"   Best model: {model_size:.2f} MB")

    # Fine-tuning 결과
    print("\n📁 Fine-tuning 결과:")
    finetuning_dir = os.path.join(results_dir, "finetuning")
    if os.path.exists(finetuning_dir):
        files = os.listdir(finetuning_dir)
        print(f"   위치: {finetuning_dir}")
        print(f"   파일 개수: {len(files)}")
        if "best_gsk3b_model.pt" in files:
            model_size = os.path.getsize(os.path.join(finetuning_dir, "best_gsk3b_model.pt")) / 1024 / 1024
            print(f"   Best model: {model_size:.2f} MB")

    # Generative Model 결과
    print("\n📁 Generative Model 결과:")
    generative_dir = os.path.join(results_dir, "generativemodel")
    if os.path.exists(generative_dir):
        files = os.listdir(generative_dir)
        print(f"   위치: {generative_dir}")
        print(f"   파일 개수: {len(files)}")
        if "generated_molecules.csv" in files:
            import pandas as pd
            df = pd.read_csv(os.path.join(generative_dir, "generated_molecules.csv"))
            print(f"   생성된 분자 수: {len(df)}")
            if 'valid' in df.columns:
                valid_count = df['valid'].sum()
                print(f"   유효한 분자 수: {valid_count} ({valid_count/len(df)*100:.1f}%)")

    print("\n" + "="*80)
    print("✓ 전체 파이프라인 실행 완료!")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="LAIDD 통합 실행 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 파이프라인 실행
  python run_all.py --all

  # Pretraining만 실행
  python run_all.py --pretraining

  # Fine-tuning만 실행
  python run_all.py --finetuning

  # Generative model만 실행
  python run_all.py --generative

  # 기존 결과 무시하고 강제 재실행
  python run_all.py --all --no-skip
        """
    )

    parser.add_argument('--all', action='store_true', help='전체 파이프라인 실행')
    parser.add_argument('--pretraining', action='store_true', help='Pretraining만 실행')
    parser.add_argument('--finetuning', action='store_true', help='Fine-tuning만 실행')
    parser.add_argument('--generative', action='store_true', help='Generative model만 실행')
    parser.add_argument('--no-skip', action='store_true', help='기존 결과 무시하고 재실행')
    parser.add_argument('--check-only', action='store_true', help='데이터 파일만 확인')

    args = parser.parse_args()

    # 로고 출력
    print(LOGO)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 옵션이 하나도 선택되지 않은 경우
    if not any([args.all, args.pretraining, args.finetuning, args.generative, args.check_only]):
        print("⚠ 실행할 작업을 선택해주세요.")
        print("   예: python run_all.py --all")
        print("   도움말: python run_all.py --help")
        return

    # 데이터 파일 확인
    if not check_data_files():
        if not args.check_only:
            print("\n⚠ 데이터 파일이 준비되지 않아 실행을 중단합니다.")
        return

    if args.check_only:
        print("\n데이터 파일 확인이 완료되었습니다.")
        return

    skip_existing = not args.no_skip

    # 실행할 작업 결정
    run_pre = args.all or args.pretraining
    run_fine = args.all or args.finetuning
    run_gen = args.all or args.generative

    success = True

    # 실행
    if run_pre:
        if not run_pretraining(skip_if_exists=skip_existing):
            success = False
            if args.all:
                print("\n⚠ Pretraining 실패로 인해 나머지 단계를 건너뜁니다.")
                return

    if run_fine:
        if not run_finetuning(skip_if_exists=skip_existing):
            success = False
            if args.all:
                print("\n⚠ Fine-tuning 실패로 인해 나머지 단계를 건너뜁니다.")
                return

    if run_gen:
        if not run_generative_model(skip_if_exists=skip_existing):
            success = False

    # 결과 요약
    if success:
        print_summary()

    print(f"\n종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
