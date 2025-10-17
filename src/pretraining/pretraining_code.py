##### 20250903 Multi-Affinity Learning 버전 (BindingDB 적용)
# ===========================
# Multi-task Learning으로 4가지 affinity 동시 학습
# BindingDB_clean_processed.tsv 파일 형식에 맞게 수정
# IC50, Ki, EC50, Kd 등을 하나의 모델에서 처리
# ===========================

import os, re, math, random, subprocess, sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# 재현성
# ---------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------------------
# RDKit 가드
# ---------------------------
def ensure_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        return True
    except Exception:
        pass
    print("[INFO] RDKit 미탑재 -> pip로 설치 시도 ...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rdkit-pypi"])
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        print("[INFO] RDKit 설치 및 활성화 성공")
        return True
    except Exception as e:
        print(f"[WARN] RDKit 설치 실패 -> scaffold split 비활성화: {e}")
        return False

HAS_RDKIT = ensure_rdkit()
if HAS_RDKIT:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

# ---------------------------
# Vocab
# ---------------------------
class SimpleVocab:
    def __init__(self, tokens):
        self.pad = "<PAD>"; self.eos = "<EOS>"; self.unk = "<UNK>"
        self.id2tok = [self.pad, self.eos, self.unk] + list(tokens)
        self.tok2id = {t:i for i,t in enumerate(self.id2tok)}
    def encode(self, text, add_eos=True, max_len=None):
        s = str(text)
        ids = [self.tok2id.get(ch, self.tok2id[self.unk]) for ch in s]
        if add_eos: ids.append(self.tok2id[self.eos])
        if max_len is not None: ids = ids[:max_len]
        return ids
    def pad_id(self): return self.tok2id[self.pad]

SMILES_CHARS = list("#%()+-./0123456789:=@ABCDEFGHIKLMNOPRSTVXYZ[]abcdefgilmnoprstuy\\")
AA_CHARS     = list("ACDEFGHIKLMNPQRSTVWY")
smiles_vocab = SimpleVocab(tokens=SMILES_CHARS)
aa_vocab     = SimpleVocab(tokens=AA_CHARS)

# ---------------------------
# Affinity Type Mapping
# ---------------------------
AFFINITY_TYPES = ['IC50', 'Ki', 'EC50', 'Kd']  # 4가지 affinity 타입
affinity_to_id = {aff: i for i, aff in enumerate(AFFINITY_TYPES)}

# ---------------------------
# Multi-Affinity Dataset (BindingDB용 수정)
# ---------------------------
class MultiAffinityDataset(Dataset):
    """
    BindingDB_clean_processed.tsv 파일 형식에 맞게 수정
    필수 컬럼: 'Standardized_SMILES', 'Target_Sequence', 'pChEMBL_Value', 'Standard Type'
    """
    def __init__(self, df, max_len_smiles=256, max_len_seq=1500,
                 drop_missing=True, use_target_column='Target_Sequence'):
        self.df = df.copy()
        
        # BindingDB 파일의 컬럼명에 맞게 수정
        required_cols = ["Standardized_SMILES", "Standard Type", "pChEMBL_Value"]
        
        # Target sequence 컬럼 확인 (여러 가능한 이름)
        target_seq_candidates = [use_target_column, "Target_Sequence", "Target Sequence", 
                               "Protein_Sequence", "Sequence"]
        target_seq_col = None
        for col in target_seq_candidates:
            if col in self.df.columns:
                target_seq_col = col
                break
        
        if target_seq_col is None:
            raise ValueError(f"타겟 서열 컬럼을 찾을 수 없습니다. 가능한 컬럼명: {target_seq_candidates}")
        
        # 필수 컬럼 확인
        for c in required_cols:
            if c not in self.df.columns:
                raise ValueError(f"필수 컬럼 누락: {c}")
        
        # 타겟 서열 컬럼 이름 통일
        if target_seq_col != "Target_Sequence":
            self.df["Target_Sequence"] = self.df[target_seq_col]
        
        # 결측값 제거
        if drop_missing:
            initial_len = len(self.df)
            self.df = self.df[~self.df["pChEMBL_Value"].isna()].reset_index(drop=True)
            self.df = self.df[~self.df["Standard Type"].isna()].reset_index(drop=True)
            self.df = self.df[~self.df["Standardized_SMILES"].isna()].reset_index(drop=True)
            self.df = self.df[~self.df["Target_Sequence"].isna()].reset_index(drop=True)
            print(f"[INFO] 결측값 제거: {initial_len} -> {len(self.df)} ({initial_len - len(self.df)} 제거)")
        
        # 지원하는 affinity 타입만 필터링
        before_filter = len(self.df)
        self.df = self.df[self.df["Standard Type"].isin(AFFINITY_TYPES)].reset_index(drop=True)
        after_filter = len(self.df)
        if before_filter != after_filter:
            print(f"[INFO] 지원하지 않는 affinity 타입 제거: {before_filter} -> {after_filter}")
        
        # pChEMBL_Value가 이미 계산되어 있으므로 그대로 사용
        self.df["pValue"] = self.df["pChEMBL_Value"].astype(float)
        
        self.max_len_smiles = max_len_smiles
        self.max_len_seq = max_len_seq
        
        print(f"[INFO] 로드된 데이터: {len(self.df)}개")
        print(f"[INFO] Affinity 타입 분포:")
        for aff_type in AFFINITY_TYPES:
            count = (self.df["Standard Type"] == aff_type).sum()
            print(f"  - {aff_type}: {count}개")
        
        # pChEMBL 값 범위 확인
        pchembl_stats = self.df["pValue"].describe()
        print(f"[INFO] pChEMBL 값 범위: {pchembl_stats['min']:.2f} ~ {pchembl_stats['max']:.2f}")
        print(f"[INFO] pChEMBL 평균: {pchembl_stats['mean']:.2f} ± {pchembl_stats['std']:.2f}")

    def __len__(self): return len(self.df)

    def _encode_smiles(self, s):
        return smiles_vocab.encode(s, add_eos=True, max_len=self.max_len_smiles)

    def _encode_seq(self, s):
        s = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", str(s)) if pd.notna(s) else ""
        return aa_vocab.encode(s, add_eos=True, max_len=self.max_len_seq)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smi_ids = torch.tensor(self._encode_smiles(row["Standardized_SMILES"]), dtype=torch.long)
        seq_ids = torch.tensor(self._encode_seq(row["Target_Sequence"]), dtype=torch.long)
        
        # pValue와 affinity type
        y = torch.tensor(float(row["pValue"]), dtype=torch.float32)
        affinity_type = row["Standard Type"]
        affinity_id = torch.tensor(affinity_to_id[affinity_type], dtype=torch.long)
        
        # 메타데이터 (BindingDB 파일에 맞게 수정)
        meta = {
            "UniProt_ID": row.get("UniProt_ID", row.get("UniProt ID", "")),
            "Target_Name": row.get("Target_Name", row.get("Target Name", "")),
            "Standard_Type": affinity_type,
            "Standard_Value": row.get("Standard Value", ""),
            "Standard_Units": row.get("Standard Units", ""),
            "pChEMBL_Value": row["pChEMBL_Value"],
            "is_active": row.get("is_active", False)
        }
        return {
            "smiles_ids": smi_ids, 
            "seq_ids": seq_ids, 
            "y": y, 
            "affinity_id": affinity_id,
            "meta": meta
        }

def pad_1d(seqs, pad_id):
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out, lens

def collate_fn_multi(batch):
    smiles_ids = [b["smiles_ids"] for b in batch]
    seq_ids = [b["seq_ids"] for b in batch]
    y_list = [b["y"] for b in batch]
    affinity_ids = [b["affinity_id"] for b in batch]
    
    X_smi, L_smi = pad_1d(smiles_ids, smiles_vocab.pad_id())
    X_seq, L_seq = pad_1d(seq_ids, aa_vocab.pad_id())
    y = torch.stack(y_list, dim=0)
    affinity_types = torch.stack(affinity_ids, dim=0)
    meta = [b["meta"] for b in batch]
    
    return {
        "X_smi": X_smi, "L_smi": L_smi, 
        "X_seq": X_seq, "L_seq": L_seq,
        "y": y, "affinity_types": affinity_types, 
        "meta": meta
    }

# ---------------------------
# Multi-Affinity Model
# ---------------------------
class MaskedMeanPool(nn.Module):
    def forward(self, x, lengths):
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
        mask = mask.float().unsqueeze(-1)
        x = x * mask
        s = x.sum(dim=1)
        l = mask.sum(dim=1).clamp(min=1.0)
        return s / l

class ConvBlock(nn.Module):
    def __init__(self, d_model, k=5, d_hidden=None, p=0.1):
        super().__init__()
        d_hidden = d_hidden or d_model
        self.conv = nn.Conv1d(d_model, d_hidden, kernel_size=k, padding=k//2)
        self.act = nn.GELU()
        self.proj = nn.Linear(d_hidden, d_model)
        self.drop = nn.Dropout(p)
    def forward(self, x, lengths):
        x = x.transpose(1, 2)
        h = self.conv(x).transpose(1, 2)
        h = self.act(h)
        h = self.proj(h)
        return self.drop(h)

class MultiAffinityNet(nn.Module):
    def __init__(self,
                 vocab_smi=len(smiles_vocab.id2tok),
                 vocab_seq=len(aa_vocab.id2tok),
                 n_affinity_types=len(AFFINITY_TYPES),
                 d_model=256, d_hidden=256, n_layers=2, dropout=0.1):
        super().__init__()
        
        # 공통 인코더
        self.emb_smi = nn.Embedding(vocab_smi, d_model, padding_idx=smiles_vocab.pad_id())
        self.emb_seq = nn.Embedding(vocab_seq, d_model, padding_idx=aa_vocab.pad_id())
        self.conv_smi = ConvBlock(d_model, k=5, d_hidden=d_hidden, p=dropout)
        self.conv_seq = ConvBlock(d_model, k=7, d_hidden=d_hidden, p=dropout)
        self.pool = MaskedMeanPool()
        
        # Affinity type embedding
        self.affinity_emb = nn.Embedding(n_affinity_types, d_model//4)
        
        # 공통 특징 추출기
        in_dim = d_model * 2 + d_model//4  # compound + protein + affinity_type
        layers = []
        dim = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(dim, d_hidden), nn.GELU(), nn.Dropout(dropout)]
            dim = d_hidden
        self.shared_mlp = nn.Sequential(*layers)
        
        # 각 affinity type별 예측 헤드
        self.heads = nn.ModuleDict({
            aff_type: nn.Linear(d_hidden, 1) for aff_type in AFFINITY_TYPES
        })
        
        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X_smi, L_smi, X_seq, L_seq, affinity_types):
        # 공통 인코딩
        es = self.emb_smi(X_smi)
        ep = self.emb_seq(X_seq)
        es = self.conv_smi(es, L_smi)
        ep = self.conv_seq(ep, L_seq)
        ps = self.pool(es, L_smi)
        pp = self.pool(ep, L_seq)
        
        # Affinity type embedding
        aff_emb = self.affinity_emb(affinity_types)
        
        # 결합된 특징
        h = torch.cat([ps, pp, aff_emb], dim=-1)
        h = self.shared_mlp(h)
        
        # 각 샘플에 대해 해당하는 affinity head 사용
        predictions = []
        for i, aff_id in enumerate(affinity_types):
            aff_type = AFFINITY_TYPES[aff_id.item()]
            pred = self.heads[aff_type](h[i:i+1])
            predictions.append(pred)
        
        y_pred = torch.cat(predictions, dim=0).squeeze(-1)
        return y_pred

# ---------------------------
# Split 함수들
# ---------------------------
def murcko_scaffold(smiles):
    if not HAS_RDKIT:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None

def scaffold_split(df, ratios=(0.7, 0.15, 0.15), seed=SEED):
    if not HAS_RDKIT:
        return random_split(df, ratios, seed)
    df = df.copy()
    df["_scaf"] = df["Standardized_SMILES"].astype(str).apply(murcko_scaffold)
    df["_scaf"] = df["_scaf"].fillna("NA_" + df["Standardized_SMILES"].astype(str))
    scaf_to_idxs = {}
    for idx, scaf in enumerate(df["_scaf"]):
        scaf_to_idxs.setdefault(scaf, []).append(idx)
    scafs = list(scaf_to_idxs.keys())
    random.Random(seed).shuffle(scafs)
    n = len(df); n_train = int(n * ratios[0]); n_val = int(n * ratios[1])
    train_idx, val_idx, test_idx = [], [], []
    count = 0
    for scaf in scafs:
        idxs = scaf_to_idxs[scaf]
        if count < n_train: train_idx.extend(idxs)
        elif count < n_train + n_val: val_idx.extend(idxs)
        else: test_idx.extend(idxs)
        count += len(idxs)
    tr = df.iloc[train_idx].drop(columns=["_scaf"]).reset_index(drop=True)
    va = df.iloc[val_idx].drop(columns=["_scaf"]).reset_index(drop=True)
    te = df.iloc[test_idx].drop(columns=["_scaf"]).reset_index(drop=True)
    return tr, va, te

def random_split(df, ratios=(0.7, 0.15, 0.15), seed=SEED):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df); n_train = int(n * ratios[0]); n_val = int(n * ratios[1])
    tr = df.iloc[:n_train].reset_index(drop=True)
    va = df.iloc[n_train:n_train+n_val].reset_index(drop=True)
    te = df.iloc[n_train+n_val:].reset_index(drop=True)
    return tr, va, te

# ---------------------------
# Metrics
# ---------------------------
@torch.no_grad()
def compute_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    mse = float(np.mean((y_true - y_pred)**2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    var = float(np.var(y_true))
    r2 = 1.0 - (mse / var) if var > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

@torch.no_grad()
def compute_metrics_by_affinity(y_true, y_pred, affinity_types):
    """Affinity type별로 메트릭 계산"""
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    affinity_types = affinity_types.detach().cpu().numpy()
    
    overall = compute_metrics(torch.tensor(y_true), torch.tensor(y_pred))
    by_type = {}
    
    for i, aff_type in enumerate(AFFINITY_TYPES):
        mask = affinity_types == i
        if mask.sum() > 0:
            y_t = y_true[mask]
            y_p = y_pred[mask]
            by_type[aff_type] = compute_metrics(torch.tensor(y_t), torch.tensor(y_p))
            by_type[aff_type]["count"] = int(mask.sum())
        else:
            by_type[aff_type] = {"RMSE": 0, "MAE": 0, "R2": 0, "count": 0}
    
    return overall, by_type

# ---------------------------
# Training Loop
# ---------------------------
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience; self.min_delta = min_delta
        self.best = float("inf"); self.counter = 0
    def step(self, value):
        if value < self.best - self.min_delta:
            self.best = value; self.counter = 0; return False
        self.counter += 1
        return self.counter >= self.patience

def run_epoch_multi_amp(model, loader, optimizer=None, scaler=None, device=None, scheduler=None, grad_accum=1):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_mode = optimizer is not None
    model.to(device); model.train() if train_mode else model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0; y_all=[]; p_all=[]; aff_all=[]
    step = 0
    
    for batch in loader:
        X_smi = batch["X_smi"].to(device, non_blocking=True)
        L_smi = batch["L_smi"].to(device, non_blocking=True)
        X_seq = batch["X_seq"].to(device, non_blocking=True)
        L_seq = batch["L_seq"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        affinity_types = batch["affinity_types"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            y_pred = model(X_smi, L_smi, X_seq, L_seq, affinity_types)
            loss = criterion(y_pred, y) / (grad_accum if train_mode else 1)

        if train_mode:
            scaler.scale(loss).backward()
            if (step+1) % grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
        
        total_loss += loss.item() * y.size(0) * (grad_accum if train_mode else 1)
        y_all.append(y.detach()); p_all.append(y_pred.detach()); aff_all.append(affinity_types.detach())
        step += 1

    y_all = torch.cat(y_all, dim=0); p_all = torch.cat(p_all, dim=0); aff_all = torch.cat(aff_all, dim=0)
    avg_loss = total_loss / len(y_all)
    overall_metrics, by_type_metrics = compute_metrics_by_affinity(y_all, p_all, aff_all)
    return avg_loss, overall_metrics, by_type_metrics

# ---------------------------
# 메인 학습 함수 (BindingDB용 수정)
# ---------------------------
def main_multi_affinity_bindingdb(tsv_path,
                                  batch_size=64,
                                  max_len_smiles=256,
                                  max_len_seq=1500,
                                  epochs=20,
                                  lr=1e-3,
                                  weight_decay=1e-5,
                                  out_dir="./results",
                                  use_scaffold=True,
                                  grad_accum=2,
                                  warmup_steps=500,
                                  save_every=5,
                                  resume_path=None,
                                  target_seq_column='Target_Sequence'):
    
    print(f"\n[INFO] 결과 저장 폴더 생성: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    # --- split cache ---
    split_cache = os.path.join(out_dir, "split_cached.flag")
    if os.path.exists(split_cache):
        print("[INFO] 기존 데이터 분할 사용")
        tr_df = pd.read_csv(os.path.join(out_dir, "train_split.tsv"), sep="\t")
        va_df = pd.read_csv(os.path.join(out_dir, "val_split.tsv"), sep="\t")
        te_df = pd.read_csv(os.path.join(out_dir, "test_split.tsv"), sep="\t")
    else:
        print("[INFO] 데이터 로딩 및 분할 중...")
        df = pd.read_csv(tsv_path, sep="\t")
        print(f"   원본 데이터: {len(df):,} 행")
        
        # BindingDB 파일의 필수 컬럼 확인
        required_cols = ["Standardized_SMILES", "Standard Type", "pChEMBL_Value"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"필수 컬럼 누락: {col}. 사용 가능한 컬럼: {list(df.columns)}")
        
        # 활성 데이터만 사용하거나 전체 데이터 사용 선택
        if 'is_active' in df.columns:
            active_count = df['is_active'].sum()
            total_count = len(df)
            print(f"   활성 데이터: {active_count}/{total_count} ({active_count/total_count*100:.1f}%)")
            
            # 선택적으로 활성 데이터만 사용
            use_active_only = input("활성 데이터만 사용하시겠습니까? (y/n): ").lower().startswith('y')
            if use_active_only:
                df = df[df['is_active'] == True].reset_index(drop=True)
                print(f"   활성 데이터만 선택: {len(df):,} 행")
        
        if use_scaffold and HAS_RDKIT:
            print("   [INFO] Scaffold split 수행...")
            tr_df, va_df, te_df = scaffold_split(df, ratios=(0.7, 0.15, 0.15), seed=SEED)
        else:
            print("   [INFO] Random split 수행...")
            tr_df, va_df, te_df = random_split(df, ratios=(0.7, 0.15, 0.15), seed=SEED)
        
        tr_df.to_csv(os.path.join(out_dir, "train_split.tsv"), sep="\t", index=False)
        va_df.to_csv(os.path.join(out_dir, "val_split.tsv"), sep="\t", index=False)
        te_df.to_csv(os.path.join(out_dir, "test_split.tsv"), sep="\t", index=False)
        open(split_cache, "w").close()
        print("   [INFO] 분할 결과 저장 완료")

    # --- datasets ---
    print("\n[INFO] 데이터셋 생성 중...")
    train_ds = MultiAffinityDataset(tr_df, max_len_smiles=max_len_smiles, max_len_seq=max_len_seq,
                                   drop_missing=True, use_target_column=target_seq_column)
    val_ds = MultiAffinityDataset(va_df, max_len_smiles=max_len_smiles, max_len_seq=max_len_seq,
                                 drop_missing=True, use_target_column=target_seq_column)
    test_ds = MultiAffinityDataset(te_df, max_len_smiles=max_len_smiles, max_len_seq=max_len_seq,
                                  drop_missing=True, use_target_column=target_seq_column)

    print("\n[DATASET INFO]")
    print(f"Train: {len(train_ds):,} samples")
    print(f"Val:   {len(val_ds):,} samples") 
    print(f"Test:  {len(test_ds):,} samples")
    print(f"Total: {len(train_ds) + len(val_ds) + len(test_ds):,} samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, collate_fn=collate_fn_multi)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, collate_fn=collate_fn_multi)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, collate_fn=collate_fn_multi)

    # --- model/opt/scheduler/AMP ---
    print("\n[INFO] 모델 초기화 중...")
    model = MultiAffinityNet(d_model=256, d_hidden=256, n_layers=2, dropout=0.1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL INFO]")
    print(f"Device: {device}")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"Affinity types: {', '.join(AFFINITY_TYPES)}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = (len(train_loader) // max(1, grad_accum)) * epochs
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # --- resume ---
    start_epoch = 1
    best_val_rmse = float("inf")
    best_path = os.path.join(out_dir, "best_model.pt")
    if resume_path and os.path.exists(resume_path):
        print(f"[INFO] 체크포인트에서 재시작: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_rmse = ckpt.get("best_val_rmse", best_val_rmse)

    stopper = EarlyStopper(patience=7, min_delta=1e-3)
    history = []

    print(f"\n[TRAINING] Epoch {start_epoch} -> {epochs}")
    print("=" * 80)

    for ep in range(start_epoch, epochs+1):
        tr_loss, tr_met, tr_by_type = run_epoch_multi_amp(model, train_loader, optimizer=optimizer,
                                                          scaler=scaler, device=device, scheduler=scheduler,
                                                          grad_accum=grad_accum)
        va_loss, va_met, va_by_type = run_epoch_multi_amp(model, val_loader, optimizer=None,
                                                          scaler=None, device=device)

        # 진행상황 출력
        progress = ep / epochs * 100
        bar_length = 30
        filled_length = int(bar_length * ep // epochs)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        print(f"\n[{progress:5.1f}%] |{bar}| Epoch {ep:02d}/{epochs}")
        print(f"  Train: Loss={tr_loss:.4f}, RMSE={tr_met['RMSE']:.3f}, R2={tr_met['R2']:.3f}")
        print(f"  Val:   Loss={va_loss:.4f}, RMSE={va_met['RMSE']:.3f}, R2={va_met['R2']:.3f}")
        
        # Affinity type별 성능 출력
        print("  Affinity-specific performance:")
        for aff_type in AFFINITY_TYPES:
            val_info = va_by_type[aff_type]
            if val_info['count'] > 0:
                print(f"    {aff_type:>4}: RMSE={val_info['RMSE']:.3f}, R2={val_info['R2']:.3f} (n={val_info['count']})")

        # Best model 저장
        if va_met["RMSE"] < best_val_rmse:
            best_val_rmse = va_met["RMSE"]
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": ep,
                "best_val_rmse": best_val_rmse,
                "train_metrics": tr_met,
                "val_metrics": va_met,
                "val_by_type": va_by_type
            }, best_path)
            print(f"  -> Best model saved (RMSE: {best_val_rmse:.3f})")

        # 주기적 저장
        if ep % save_every == 0:
            save_path = os.path.join(out_dir, f"model_epoch_{ep}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": ep,
                "best_val_rmse": best_val_rmse,
                "train_metrics": tr_met,
                "val_metrics": va_met,
                "val_by_type": va_by_type
            }, save_path)

        history.append({
            "epoch": ep,
            "train_loss": tr_loss, "train_rmse": tr_met["RMSE"], "train_r2": tr_met["R2"],
            "val_loss": va_loss, "val_rmse": va_met["RMSE"], "val_r2": va_met["R2"],
            "val_by_type": va_by_type
        })

        # Early stopping
        if stopper.step(va_met["RMSE"]):
            print(f"\n[INFO] Early stopping at epoch {ep}")
            break

    # --- 최종 테스트 평가 ---
    print("\n" + "=" * 80)
    print("[FINAL EVALUATION]")
    
    # Best model 로드
    if os.path.exists(best_path):
        print(f"[INFO] Loading best model for test evaluation...")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    
    test_loss, test_met, test_by_type = run_epoch_multi_amp(model, test_loader, optimizer=None,
                                                           scaler=None, device=device)
    
    print(f"Test Results:")
    print(f"  Overall: RMSE={test_met['RMSE']:.3f}, MAE={test_met['MAE']:.3f}, R2={test_met['R2']:.3f}")
    print(f"  By Affinity Type:")
    for aff_type in AFFINITY_TYPES:
        test_info = test_by_type[aff_type]
        if test_info['count'] > 0:
            print(f"    {aff_type:>4}: RMSE={test_info['RMSE']:.3f}, R2={test_info['R2']:.3f} (n={test_info['count']})")

    # 결과 저장
    results_path = os.path.join(out_dir, "final_results.json")
    import json
    final_results = {
        "test_overall": test_met,
        "test_by_type": test_by_type,
        "best_val_rmse": best_val_rmse,
        "training_history": history,
        "model_info": {
            "total_params": total_params,
            "affinity_types": AFFINITY_TYPES,
            "dataset_sizes": {
                "train": len(train_ds),
                "val": len(val_ds),
                "test": len(test_ds)
            }
        }
    }
    
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n[INFO] 훈련 완료! 결과 저장: {out_dir}")
    print(f"  - Best model: {best_path}")
    print(f"  - Results: {results_path}")
    
    return model, final_results

# ---------------------------
# 실행 예시 및 메인 함수
# ---------------------------
if __name__ == "__main__":
    # Get project root directory (LAIDD/)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # BindingDB_clean_processed.tsv 파일 사용
    TSV_PATH = os.path.join(PROJECT_ROOT, "data", "pretraining_data.tsv")
    
    print("=" * 100)
    print("Multi-Affinity Learning for BindingDB")
    print("=" * 100)
    print(f"Input file: {TSV_PATH}")
    print(f"Supported affinity types: {', '.join(AFFINITY_TYPES)}")
    print("=" * 100)
    
    # 파일 존재 확인
    if not os.path.exists(TSV_PATH):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {TSV_PATH}")
        print("올바른 경로를 확인해주세요.")
        sys.exit(1)
    
    # 간단한 데이터 미리보기
    try:
        df_preview = pd.read_csv(TSV_PATH, sep="\t", nrows=5)
        print(f"\n[PREVIEW] 파일 구조:")
        print(f"  컬럼: {list(df_preview.columns)}")
        
        # 백슬래시 문제 해결
        full_df = pd.read_csv(TSV_PATH, sep="\t")
        num_rows = len(full_df)
        print(f"  샘플 수: {num_rows} rows")
        
        # Standard Type 분포 확인
        if 'Standard Type' in full_df.columns:
            type_counts = full_df['Standard Type'].value_counts()
            print(f"  Standard Type 분포:")
            for stype, count in type_counts.head(10).items():
                print(f"    {stype}: {count:,}")
        
        # 활성 데이터 비율 확인
        if 'is_active' in full_df.columns:
            active_ratio = full_df['is_active'].mean() * 100
            print(f"  활성 데이터 비율: {active_ratio:.1f}%")
        
    except Exception as e:
        print(f"[WARN] 파일 미리보기 실패: {e}")
    
    # Output directory
    OUT_DIR = os.path.join(PROJECT_ROOT, "results", "pretraining")

    # 학습 실행
    try:
        model, results = main_multi_affinity_bindingdb(
            tsv_path=TSV_PATH,
            batch_size=32,  # BindingDB는 큰 데이터셋이므로 batch size 조정
            max_len_smiles=256,
            max_len_seq=1500,
            epochs=30,
            lr=1e-3,
            weight_decay=1e-5,
            out_dir=OUT_DIR,
            use_scaffold=True,  # Scaffold split 사용
            grad_accum=4,
            warmup_steps=1000,
            save_every=5,
            target_seq_column='BindingDB Target Chain Sequence 1'  # 실제 BindingDB 컬럼명
        )
        
        print("\n🎉 훈련이 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"\n❌ 훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()