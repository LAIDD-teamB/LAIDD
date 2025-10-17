##### GSK3β Fine-tuning Code for Multi-Affinity Model (Fixed)
# ===========================
# PyTorch 버전 호환성 문제 해결
# ReduceLROnPlateau verbose 매개변수 제거
# ===========================

import os, re, math, random, subprocess, sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import json
from copy import deepcopy

# ---------------------------
# 기존 모델 구조 import (pretraining_code.py에서)
# ---------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pretraining'))

from pretraining_code import (
    MultiAffinityNet, MultiAffinityDataset, collate_fn_multi,
    smiles_vocab, aa_vocab, AFFINITY_TYPES, affinity_to_id,
    compute_metrics, compute_metrics_by_affinity, EarlyStopper,
    run_epoch_multi_amp, SEED
)

# ---------------------------
# GSK3β Fine-tuning Dataset Class
# ---------------------------
class GSK3BDataset(MultiAffinityDataset):
    """
    GSK3β 특화 데이터셋 클래스
    실습노트의 접근방식을 따라 작은 데이터셋에 대한 과적합 방지 고려
    """
    def __init__(self, df, max_len_smiles=256, max_len_seq=1500, 
                 augment_data=True, noise_level=0.01):
        # 기본 초기화
        super().__init__(df, max_len_smiles, max_len_seq, drop_missing=True)
        self.augment_data = augment_data
        self.noise_level = noise_level
        
        # GSK3β 특화 정보 출력
        print(f"[GSK3B INFO] 파인튜닝 데이터: {len(self.df)}개")
        print(f"[GSK3B INFO] 데이터 증강: {augment_data}")
        print(f"[GSK3B INFO] Target Sequence 길이: {self.df['Target_Sequence'].str.len().mean():.0f} ± {self.df['Target_Sequence'].str.len().std():.0f}")
        
    def __getitem__(self, idx):
        # 기본 데이터 가져오기
        item = super().__getitem__(idx)
        
        # 실습노트: 과적합 방지를 위한 데이터 증강
        if self.augment_data and random.random() < 0.3:  # 30% 확률로 노이즈 추가
            # pValue에 작은 노이즈 추가 (실제 실험 오차 모사)
            noise = torch.normal(0, self.noise_level, size=(1,)).item()
            item['y'] = torch.clamp(item['y'] + noise, 0, 15)  # pIC50 범위 제한
            
        return item

# ---------------------------
# Fine-tuning 전용 학습 함수
# ---------------------------
def fine_tune_gsk3b(pretrained_model_path,
                    gsk3b_data_path,
                    output_dir="./finetuning_results",
                    epochs=50,
                    lr=1e-4,  # 실습노트: 파인튜닝에서는 낮은 학습률 사용
                    weight_decay=1e-4,
                    batch_size=16,  # 작은 데이터셋이므로 작은 배치 크기
                    early_stopping_patience=10,
                    validation_split=0.2,
                    test_split=0.1,
                    freeze_encoder=False,  # 인코더 동결 여부
                    data_augmentation=True):
    
    print("=" * 80)
    print("GSK3β Fine-tuning for Multi-Affinity Prediction")
    print("=" * 80)
    print(f"Pre-trained model: {pretrained_model_path}")
    print(f"GSK3β data: {gsk3b_data_path}")
    print(f"Output directory: {output_dir}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------
    # 1. 데이터 로딩 및 전처리
    # ---------------------------
    print("\n[STEP 1] 데이터 로딩 및 전처리")
    
    # TSV 파일 읽기
    df = pd.read_csv(gsk3b_data_path, sep='\t')
    print(f"원본 데이터: {len(df)} rows")
    
    # 필수 컬럼 확인 및 매핑
    column_mapping = {
        'Ligand SMILES': 'Standardized_SMILES',
        'BindingDB Target Chain Sequence': 'Target_Sequence',
        'Standard Type': 'Standard Type',
        'pChEMBL': 'pChEMBL_Value'
    }
    
    # 컬럼 이름 변경
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # 지원하는 affinity 타입만 필터링
    df_filtered = df[df['Standard Type'].isin(AFFINITY_TYPES)].copy()
    print(f"지원하는 affinity 타입 필터링 후: {len(df_filtered)} rows")
    
    # Affinity 타입별 분포 확인
    affinity_counts = df_filtered['Standard Type'].value_counts()
    print("Affinity 타입 분포:")
    for aff_type, count in affinity_counts.items():
        print(f"  {aff_type}: {count}개")
    
    # 결측값 제거
    df_clean = df_filtered.dropna(subset=['Standardized_SMILES', 'Target_Sequence', 'pChEMBL_Value']).reset_index(drop=True)
    print(f"결측값 제거 후: {len(df_clean)} rows")
    
    # ---------------------------
    # 2. 데이터 분할 (실습노트: 작은 데이터셋에 대한 신중한 분할)
    # ---------------------------
    print(f"\n[STEP 2] 데이터 분할")
    
    # Stratified split (affinity type 기준)
    from sklearn.model_selection import train_test_split
    
    # 먼저 train+val과 test 분할
    train_val_df, test_df = train_test_split(
        df_clean, 
        test_size=test_split, 
        random_state=SEED,
        stratify=df_clean['Standard Type']
    )
    
    # train과 validation 분할
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=validation_split/(1-test_split),
        random_state=SEED,
        stratify=train_val_df['Standard Type']
    )
    
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # 데이터셋 생성
    train_dataset = GSK3BDataset(train_df, augment_data=data_augmentation)
    val_dataset = GSK3BDataset(val_df, augment_data=False)
    test_dataset = GSK3BDataset(test_df, augment_data=False)
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn_multi, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn_multi, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn_multi, num_workers=2)
    
    # ---------------------------
    # 3. Pre-trained 모델 로딩
    # ---------------------------
    print(f"\n[STEP 3] Pre-trained 모델 로딩")
    
    # 모델 초기화
    model = MultiAffinityNet(d_model=256, d_hidden=256, n_layers=2, dropout=0.1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Pre-trained weights 로딩
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print("Pre-trained model state dict 로딩 완료")
    else:
        model.load_state_dict(checkpoint)
        print("Pre-trained model weights 로딩 완료")
    
    model.to(device)
    
    # ---------------------------
    # 4. Fine-tuning 설정
    # ---------------------------
    print(f"\n[STEP 4] Fine-tuning 설정")
    
    # 실습노트: 인코더 동결 옵션
    if freeze_encoder:
        print("인코더 부분 동결")
        for name, param in model.named_parameters():
            if any(layer in name for layer in ['emb_smi', 'emb_seq', 'conv_smi', 'conv_seq']):
                param.requires_grad = False
    
    # 학습 가능한 파라미터 수 확인
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"전체 파라미터: {total_params:,}")
    print(f"학습 가능한 파라미터: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # 옵티마이저 설정
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    # 스케줄러 설정 (verbose 매개변수 제거)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # AMP 스케일러
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    # Early stopping
    early_stopper = EarlyStopper(patience=early_stopping_patience, min_delta=1e-4)
    
    # ---------------------------
    # 5. Fine-tuning 학습
    # ---------------------------
    print(f"\n[STEP 5] Fine-tuning 학습 시작")
    print(f"Epochs: {epochs}, Learning Rate: {lr}, Batch Size: {batch_size}")
    print("-" * 60)
    
    best_val_rmse = float('inf')
    best_model_path = os.path.join(output_dir, "best_gsk3b_model.pt")
    history = []
    
    for epoch in range(1, epochs + 1):
        # 학습
        train_loss, train_metrics, train_by_type = run_epoch_multi_amp(
            model, train_loader, optimizer=optimizer, scaler=scaler, 
            device=device, scheduler=None, grad_accum=1
        )
        
        # 검증
        val_loss, val_metrics, val_by_type = run_epoch_multi_amp(
            model, val_loader, optimizer=None, scaler=None, device=device
        )
        
        # 스케줄러 업데이트 (verbose 출력을 수동으로 처리)
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"    Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        # 진행상황 출력
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} RMSE: {train_metrics['RMSE']:.3f} | "
              f"Val Loss: {val_loss:.4f} RMSE: {val_metrics['RMSE']:.3f} R²: {val_metrics['R2']:.3f}")
        
        # Affinity별 성능 출력 (매 10 에포크마다)
        if epoch % 10 == 0:
            print("  Validation by affinity type:")
            for aff_type in AFFINITY_TYPES:
                if val_by_type[aff_type]['count'] > 0:
                    metrics = val_by_type[aff_type]
                    print(f"    {aff_type}: RMSE={metrics['RMSE']:.3f}, R²={metrics['R2']:.3f} (n={metrics['count']})")
        
        # Best model 저장
        if val_metrics['RMSE'] < best_val_rmse:
            best_val_rmse = val_metrics['RMSE']
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_rmse': best_val_rmse,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'val_by_type': val_by_type,
                'fine_tuning_config': {
                    'lr': lr,
                    'batch_size': batch_size,
                    'freeze_encoder': freeze_encoder,
                    'data_augmentation': data_augmentation
                }
            }, best_model_path)
            print(f"    → Best model saved (RMSE: {best_val_rmse:.3f})")
        
        # 히스토리 저장
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_rmse': train_metrics['RMSE'],
            'train_r2': train_metrics['R2'],
            'val_loss': val_loss,
            'val_rmse': val_metrics['RMSE'],
            'val_r2': val_metrics['R2'],
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Early stopping 체크
        if early_stopper.step(val_metrics['RMSE']):
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # ---------------------------
    # 6. 최종 테스트 평가
    # ---------------------------
    print(f"\n[STEP 6] 최종 테스트 평가")
    
    # Best model 로딩
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    # 테스트 평가
    test_loss, test_metrics, test_by_type = run_epoch_multi_amp(
        model, test_loader, optimizer=None, scaler=None, device=device
    )
    
    print(f"\n=== GSK3β Fine-tuning Results ===")
    print(f"Overall Test Performance:")
    print(f"  RMSE: {test_metrics['RMSE']:.3f}")
    print(f"  MAE:  {test_metrics['MAE']:.3f}")
    print(f"  R²:   {test_metrics['R2']:.3f}")
    
    print(f"\nPer-Affinity Performance:")
    for aff_type in AFFINITY_TYPES:
        if test_by_type[aff_type]['count'] > 0:
            metrics = test_by_type[aff_type]
            print(f"  {aff_type:>4}: RMSE={metrics['RMSE']:.3f}, R²={metrics['R2']:.3f} (n={metrics['count']})")
    
    # ---------------------------
    # 7. 결과 저장
    # ---------------------------
    results = {
        'test_metrics': test_metrics,
        'test_by_type': test_by_type,
        'best_val_rmse': best_val_rmse,
        'training_history': history,
        'fine_tuning_config': {
            'pretrained_model_path': pretrained_model_path,
            'epochs': epoch,  # 실제 실행된 에포크
            'lr': lr,
            'batch_size': batch_size,
            'freeze_encoder': freeze_encoder,
            'data_augmentation': data_augmentation,
            'early_stopping_patience': early_stopping_patience
        },
        'dataset_info': {
            'total_samples': len(df_clean),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'affinity_distribution': affinity_counts.to_dict()
        }
    }
    
    # 결과 저장
    results_path = os.path.join(output_dir, "gsk3b_finetuning_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 데이터 분할 저장
    train_df.to_csv(os.path.join(output_dir, "train_split.tsv"), sep='\t', index=False)
    val_df.to_csv(os.path.join(output_dir, "val_split.tsv"), sep='\t', index=False)
    test_df.to_csv(os.path.join(output_dir, "test_split.tsv"), sep='\t', index=False)
    
    print(f"\n모든 결과가 저장되었습니다: {output_dir}")
    print(f"  - Best model: {best_model_path}")
    print(f"  - Results: {results_path}")
    
    return model, results

# ---------------------------
# 5-fold Fine-tuning 함수 추가
# ---------------------------
def fine_tune_gsk3b_5fold(pretrained_model_path,
                          gsk3b_data_path,
                          output_dir="./fientuning_results",
                          epochs=50,
                          lr=1e-4,
                          weight_decay=1e-4,
                          batch_size=16,
                          early_stopping_patience=10,
                          test_split=0.1,
                          freeze_encoder=False,
                          data_augmentation=True):
    """
    5-fold cross-validation을 사용한 GSK3β Fine-tuning
    """
    from sklearn.model_selection import StratifiedKFold
    
    print("=" * 80)
    print("GSK3β 5-Fold Cross-Validation Fine-tuning")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터 로딩
    df = pd.read_csv(gsk3b_data_path, sep='\t')
    
    # 컬럼 매핑
    column_mapping = {
        'Ligand SMILES': 'Standardized_SMILES',
        'BindingDB Target Chain Sequence': 'Target_Sequence',
        'Standard Type': 'Standard Type',
        'pChEMBL': 'pChEMBL_Value'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # 필터링 및 정리
    df_filtered = df[df['Standard Type'].isin(AFFINITY_TYPES)].copy()
    df_clean = df_filtered.dropna(subset=['Standardized_SMILES', 'Target_Sequence', 'pChEMBL_Value']).reset_index(drop=True)
    
    print(f"Total samples: {len(df_clean)}")
    
    # Test set 분리
    from sklearn.model_selection import train_test_split
    train_val_df, test_df = train_test_split(
        df_clean, 
        test_size=test_split, 
        random_state=SEED,
        stratify=df_clean['Standard Type']
    )
    
    print(f"Train+Val: {len(train_val_df)}, Test: {len(test_df)}")
    
    # 5-fold CV 설정
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    fold_results = []
    best_models = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_df, train_val_df['Standard Type'])):
        print(f"\n{'='*20} Fold {fold+1}/5 {'='*20}")
        
        # 폴드별 데이터 분할
        fold_train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Fold {fold+1} - Train: {len(fold_train_df)}, Val: {len(fold_val_df)}")
        
        # 데이터셋 생성
        train_dataset = GSK3BDataset(fold_train_df, augment_data=data_augmentation)
        val_dataset = GSK3BDataset(fold_val_df, augment_data=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 collate_fn=collate_fn_multi, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               collate_fn=collate_fn_multi, num_workers=2)
        
        # 모델 초기화 및 Pre-trained weights 로딩 (호환성 문제 해결)
        model = MultiAffinityNet(d_model=256, d_hidden=256, n_layers=2, dropout=0.1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint
        
        # 현재 모델과 호환되는 파라미터만 로딩
        model_dict = model.state_dict()
        filtered_dict = {}
        skipped_params = []
        
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                skipped_params.append(k)
        
        # 호환되는 파라미터만 업데이트
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        
        print(f"Pre-trained model 로딩 완료")
        print(f"  로딩된 파라미터: {len(filtered_dict)}/{len(pretrained_dict)}")
        if skipped_params:
            print(f"  건너뛴 파라미터: {skipped_params}")
        
        model.to(device)
        
        # 인코더 동결 설정
        if freeze_encoder:
            for name, param in model.named_parameters():
                if any(layer in name for layer in ['emb_smi', 'emb_seq', 'conv_smi', 'conv_seq']):
                    param.requires_grad = False
        
        # 옵티마이저 및 스케줄러
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        early_stopper = EarlyStopper(patience=early_stopping_patience, min_delta=1e-4)
        
        # 학습
        best_val_rmse = float('inf')
        fold_model_path = os.path.join(output_dir, f"best_gsk3b_model_fold{fold}.pt")
        
        for epoch in range(1, epochs + 1):
            # 학습
            train_loss, train_metrics, _ = run_epoch_multi_amp(
                model, train_loader, optimizer=optimizer, scaler=scaler, 
                device=device, scheduler=None, grad_accum=1
            )
            
            # 검증
            val_loss, val_metrics, val_by_type = run_epoch_multi_amp(
                model, val_loader, optimizer=None, scaler=None, device=device
            )
            
            scheduler.step(val_loss)
            
            # 진행상황 출력 (매 10 에포크)
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}/{epochs} | Val RMSE: {val_metrics['RMSE']:.3f} R²: {val_metrics['R2']:.3f}")
            
            # Best model 저장
            if val_metrics['RMSE'] < best_val_rmse:
                best_val_rmse = val_metrics['RMSE']
                torch.save({
                    'model': model.state_dict(),
                    'fold': fold,
                    'epoch': epoch,
                    'val_rmse': best_val_rmse,
                    'val_metrics': val_metrics,
                    'val_by_type': val_by_type
                }, fold_model_path)
            
            # Early stopping
            if early_stopper.step(val_metrics['RMSE']):
                print(f"  Early stopping at epoch {epoch}")
                break
        
        # 폴드 결과 저장
        fold_results.append({
            'fold': fold,
            'best_val_rmse': best_val_rmse,
            'val_metrics': val_metrics,
            'val_by_type': val_by_type,
            'model_path': fold_model_path
        })
        
        best_models.append(fold_model_path)
        
        print(f"Fold {fold+1} completed - Best Val RMSE: {best_val_rmse:.3f}")
    
    # 전체 폴드 결과 요약
    print(f"\n{'='*20} 5-Fold Results Summary {'='*20}")
    
    fold_rmses = [result['best_val_rmse'] for result in fold_results]
    print(f"Fold RMSEs: {[f'{rmse:.3f}' for rmse in fold_rmses]}")
    print(f"Mean RMSE: {np.mean(fold_rmses):.3f} ± {np.std(fold_rmses):.3f}")
    
    # 테스트 셋에서 앙상블 평가
    print(f"\n{'='*20} Ensemble Test Evaluation {'='*20}")
    
    test_dataset = GSK3BDataset(test_df, augment_data=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn_multi, num_workers=2)
    
    # 앙상블 예측 (5개 모델 평균)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_predictions = []
    all_targets = []
    
    for batch in test_loader:
        X_smi, L_smi = batch['X_smi'].to(device), batch['L_smi'].to(device)
        X_seq, L_seq = batch['X_seq'].to(device), batch['L_seq'].to(device)
        affinity_types = batch['affinity_types'].to(device)
        targets = batch['y'].cpu().numpy()
        
        batch_predictions = []
        
        # 각 폴드 모델로 예측
        for model_path in best_models:
            model = MultiAffinityNet(d_model=256, d_hidden=256, n_layers=2, dropout=0.1)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            model.to(device)
            model.eval()
            
            with torch.no_grad():
                pred = model(X_smi, L_smi, X_seq, L_seq, affinity_types)
                batch_predictions.append(pred.cpu().numpy())
        
        # 앙상블 평균
        ensemble_pred = np.mean(batch_predictions, axis=0)
        all_predictions.append(ensemble_pred)
        all_targets.append(targets)
    
    # 최종 앙상블 성능 계산
    final_predictions = np.concatenate(all_predictions)
    final_targets = np.concatenate(all_targets)
    
    ensemble_metrics = compute_metrics(final_targets, final_predictions)
    
    print(f"Ensemble Test Performance:")
    print(f"  RMSE: {ensemble_metrics['RMSE']:.3f}")
    print(f"  MAE:  {ensemble_metrics['MAE']:.3f}")
    print(f"  R²:   {ensemble_metrics['R2']:.3f}")
    
    # 결과 저장
    ensemble_results = {
        'fold_results': fold_results,
        'ensemble_test_metrics': ensemble_metrics,
        'fold_rmses': fold_rmses,
        'mean_rmse': float(np.mean(fold_rmses)),
        'std_rmse': float(np.std(fold_rmses)),
        'model_paths': best_models,
        'config': {
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
            'freeze_encoder': freeze_encoder,
            'data_augmentation': data_augmentation
        }
    }
    
    # 결과 파일 저장
    with open(os.path.join(output_dir, "5fold_results.json"), 'w') as f:
        json.dump(ensemble_results, f, indent=2)
    
    # 테스트 데이터 저장
    test_df.to_csv(os.path.join(output_dir, "test_split.tsv"), sep='\t', index=False)
    
    print(f"\n5-fold 결과가 저장되었습니다: {output_dir}")
    print(f"모델 파일들: {best_models}")
    
    return best_models, ensemble_results

# ---------------------------
# 실행 함수
# ---------------------------
if __name__ == "__main__":
    # Get project root directory (LAIDD/)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 경로 설정 (현재 디렉토리 기준)
    PRETRAINED_MODEL_PATH = os.path.join(PROJECT_ROOT, "results", "pretraining", "best_model.pt")
    GSK3B_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "finetuning_data.tsv")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "finetuning")
    
    print("GSK3β Fine-tuning 시작")
    print(f"Pre-trained model: {PRETRAINED_MODEL_PATH}")
    print(f"GSK3β data: {GSK3B_DATA_PATH}")
    
    # 파일 존재 확인
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"ERROR: Pre-trained model not found: {PRETRAINED_MODEL_PATH}")
        sys.exit(1)
    
    if not os.path.exists(GSK3B_DATA_PATH):
        print(f"ERROR: GSK3B data not found: {GSK3B_DATA_PATH}")
        sys.exit(1)
    
    # 사용자에게 선택 옵션 제공
    print("\nFine-tuning 방법을 선택하세요:")
    print("1. Single model fine-tuning")
    print("2. 5-fold cross-validation fine-tuning")
    
    choice = input("선택 (1 또는 2): ").strip()
    
    try:
        if choice == "1":
            # 단일 모델 파인튜닝
            model, results = fine_tune_gsk3b(
                pretrained_model_path=PRETRAINED_MODEL_PATH,
                gsk3b_data_path=GSK3B_DATA_PATH,
                output_dir=OUTPUT_DIR,
                epochs=50,
                lr=1e-4,  # 실습노트: 낮은 학습률
                weight_decay=1e-4,
                batch_size=16,
                early_stopping_patience=15,
                validation_split=0.2,
                test_split=0.1,
                freeze_encoder=False,  # 모든 레이어 학습
                data_augmentation=True  # 과적합 방지
            )
            
            print("\n🎉 GSK3β Fine-tuning이 성공적으로 완료되었습니다!")
            
        elif choice == "2":
            # 5-fold 파인튜닝
            FOLD_OUTPUT_DIR = "./gsk3b_5fold_results"
            
            model_paths, ensemble_results = fine_tune_gsk3b_5fold(
                pretrained_model_path=PRETRAINED_MODEL_PATH,
                gsk3b_data_path=GSK3B_DATA_PATH,
                output_dir=FOLD_OUTPUT_DIR,
                epochs=50,
                lr=1e-4,
                weight_decay=1e-4,
                batch_size=16,
                early_stopping_patience=10,
                test_split=0.1,
                freeze_encoder=False,
                data_augmentation=True
            )
            
            print("\n🎉 GSK3β 5-fold Fine-tuning이 성공적으로 완료되었습니다!")
            print(f"앙상블 모델들: {len(model_paths)}개")
            print(f"앙상블 테스트 RMSE: {ensemble_results['ensemble_test_metrics']['RMSE']:.3f}")
            
        else:
            print("잘못된 선택입니다. 1 또는 2를 입력해주세요.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Fine-tuning 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()