"""
Generative Model for Molecular Design using LSTM
Converted from generativemodel.ipynb
"""

import sys
import os
import time
import random
import json
import re

import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Get project root directory (LAIDD/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from utils import bs_denovo
from utils.bs_denovo.vocab import SmilesVocabulary
from utils.bs_denovo.lang_data import StringDataset
from utils.bs_denovo.lang_lstm import LSTMGeneratorConfig, LSTMGenerator
from utils.bs_denovo.gen_eval import standard_metrics

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', DEVICE)


# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_clean_data(data_csv):
    """Load CSV and canonicalize SMILES"""
    raw = pd.read_csv(data_csv)

    # Standardize column names
    colmap = {c.lower(): c for c in raw.columns}
    id_col = colmap.get('molecule_id') or colmap.get('cid') or 'ID'
    sm_col = colmap.get('smiles') or 'SMILES'
    raw = raw.rename(columns={id_col: 'id', sm_col: 'smiles'})[['id', 'smiles']]

    def canon(s: str):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True)

    raw['smiles_can'] = raw['smiles'].map(canon)
    clean = raw.dropna(subset=['smiles_can']).drop_duplicates(subset=['smiles_can']).reset_index(drop=True)
    clean = clean.rename(columns={'smiles_can': 'smiles'})[['id', 'smiles']]

    print(f'Original rows: {len(raw)} | Valid/canonicalized: {len(clean)}')
    return clean


def create_scaffold_split(clean, seed=42, sizes={'trn': 0.8, 'vld': 0.1, 'tst': 0.1}):
    """Create scaffold-based train/val/test split"""
    # Fix duplicate columns
    clean = clean.loc[:, ~clean.columns.duplicated()].copy()

    if isinstance(clean['smiles'], pd.DataFrame):
        clean['smiles'] = clean['smiles'].iloc[:, 0]

    def get_scaffold(smi: str) -> str:
        m = Chem.MolFromSmiles(str(smi))
        if not m:
            return ''
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(scaf, canonical=True) if scaf else ''

    # Calculate scaffolds with progress bar
    tqdm.pandas(desc="(1/3) Calculating Murcko scaffolds")
    clean['scaffold'] = clean['smiles'].astype(str).progress_apply(get_scaffold)

    # Collect unique scaffolds
    print("(2/3) Collecting unique scaffolds...")
    scaffolds = list(clean['scaffold'].unique())
    print(f"   -> Found {len(scaffolds)} unique scaffolds from {len(clean)} molecules")

    rng = np.random.default_rng(seed)
    rng.shuffle(scaffolds)

    # Assign splits by scaffold
    N = len(clean)
    cut_trn = int(N * sizes['trn'])
    cut_vld = int(N * (sizes['trn'] + sizes['vld']))

    assign = {}
    cum = 0

    pbar = tqdm(scaffolds, desc="(3/3) Assigning splits")
    for sc in pbar:
        idxs = clean.index[clean['scaffold'] == sc].tolist()
        n = len(idxs)
        if cum < cut_trn:
            assign_split = 'trn'
        elif cum < cut_vld:
            assign_split = 'vld'
        else:
            assign_split = 'tst'
        for i in idxs:
            assign[i] = assign_split
        cum += n
        pbar.set_postfix(cum=cum, split=assign_split)

    clean['split'] = clean.index.map(assign)
    print(clean['split'].value_counts())

    return clean


# ============================================================================
# 2. VOCABULARY BUILDING
# ============================================================================

def build_vocabulary(trn_smiles, tok_path):
    """Build vocabulary from training SMILES with robust tokenization"""
    os.makedirs(os.path.dirname(tok_path), exist_ok=True)

    # Regex tokenizer
    REGEX = re.compile(r"""
    (
      \[[^\[\]]+\]                                   # bracket atom, e.g. [NH3+], [Li+]
     |%\d{2}                                         # ring numbers like %10
     |Br|Cl|Si|Se|Na|Li|Mg|Ca|Al|Ag|Zn|Fe|Cu|Mn|Pt|Au|Hg|Sn|Pb|Bi|Sb|Ge|As|Kr|Xe|Rn
     |Ni|Co|Ti|Cr|Ga|In|Pd|Cd|Zr|V|Y|Nb|Mo|Ru|Rh|Hf|Ta|W|Re|Os|Ir
     |@@|@                                           # stereochemistry
     |=|#|-|:|/|\\                                   # bond symbols
     |\(|\)                                          # branches
     |\d                                             # ring closure digits
     |[A-Z][a-z]?                                    # element symbols
     |se|te|[cnopsb]                                 # aromatic (incl. 'se','te')
    )
    """, re.VERBOSE)

    def tokenize_regex(s: str):
        s = s.strip()
        return REGEX.findall(s) if s else []

    # Collect tokens from regex
    tokset = set()
    for s in trn_smiles:
        tokset.update(tokenize_regex(s))

    # Also collect tokens from bs_denovo tokenizer
    _probe = SmilesVocabulary(list_tokens=['<PAD>'])
    for s in trn_smiles:
        for t in _probe.tokenize(s):
            tokset.add(t)

    # Add special tokens and save
    specials = ['<CLS>', '<BOS>', '<EOS>', '<PAD>', '<MSK>', '<UNK>']
    tokens = sorted(tokset) + specials
    with open(tok_path, 'w') as f:
        f.write('\n'.join(tokens))
    print(f'[Vocab] saved → {tok_path} | size={len(tokens)}')

    # Reload and check for OOV
    smivoc_inst = SmilesVocabulary(file_name=tok_path)
    oov = set()
    for s in trn_smiles:
        for t in smivoc_inst.tokenize(s):
            if t not in smivoc_inst.tok2id:
                oov.add(t)

    if oov:
        print('[Vocab] Remaining OOV detected, auto-extending:', sorted(oov)[:30], '...' if len(oov) > 30 else '')
        base = tokens[:-len(specials)]
        tokens = sorted(set(base).union(oov)) + specials
        with open(tok_path, 'w') as f:
            f.write('\n'.join(tokens))
        smivoc_inst = SmilesVocabulary(file_name=tok_path)
        print(f'[Vocab] re-saved with OOV merged → size={len(tokens)}')
    else:
        print('[Vocab] No OOV tokens found on train split.')

    print('Vocab size:', smivoc_inst.vocab_size)
    return smivoc_inst


# ============================================================================
# 3. DATASET PREPARATION
# ============================================================================

def prepare_datasets(clean, smivoc_inst, len_cap=200):
    """Prepare train/val/test datasets with length filtering"""
    trn_smiles = clean.loc[clean['split'] == 'trn', 'smiles'].astype(str).tolist()
    vld_smiles = clean.loc[clean['split'] == 'vld', 'smiles'].astype(str).tolist()
    tst_smiles = clean.loc[clean['split'] == 'tst', 'smiles'].astype(str).tolist()

    def tok_len(s):
        return len(smivoc_inst.tokenize(str(s)))

    # Apply length cap
    trn_smiles = [s for s in trn_smiles if tok_len(s) <= len_cap]
    vld_smiles = [s for s in vld_smiles if tok_len(s) <= len_cap]
    tst_smiles = [s for s in tst_smiles if tok_len(s) <= len_cap]
    print(f"len caps → trn:{len(trn_smiles)} vld:{len(vld_smiles)} tst:{len(tst_smiles)}")

    # Create datasets
    trn_ds = StringDataset(smivoc_inst, trn_smiles)
    vld_ds = StringDataset(smivoc_inst, vld_smiles)
    tst_ds = StringDataset(smivoc_inst, tst_smiles)

    # Smoke test
    smoke_dl = DataLoader(trn_ds, batch_size=8, shuffle=True, num_workers=0, collate_fn=trn_ds.collate_fn)
    first_batch = next(iter(smoke_dl))
    if isinstance(first_batch, (list, tuple)):
        print(f"collate returned a tuple of {len(first_batch)} items")
    else:
        print("Smoke OK — batch tensor shape:", first_batch.shape)

    return trn_ds, vld_ds, tst_ds, trn_smiles


# ============================================================================
# 4. MODEL TRAINING
# ============================================================================

def train_model(trn_ds, smivoc_inst, device, ckpt_tmpl, max_epochs=80, target_loss=22.0):
    """Train LSTM generator with periodic validity checks"""
    gen_conf = LSTMGeneratorConfig(
        device=device,
        voc=smivoc_inst,
        emb_size=256,
        hidden_layer_units=[512, 512, 512],  # 3-layer
        batch_size=256,
        init_lr=1e-4,
        lr_mult=0.92,
        lr_decay_interval=3,
        ckpt_path=ckpt_tmpl,
    )

    gen = LSTMGenerator(gen_conf)
    gen.save()  # Save epoch 0

    losses, valid_curve = [], []
    sample_every = 2
    sample_n = 1000
    sample_max_len = 200

    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        ep_losses = gen.train(trn_ds, epochs=1, save_period=1, prog_save=True, dl_njobs=2, debug=None)
        loss = float(ep_losses[-1] if isinstance(ep_losses, (list, tuple)) else ep_losses)
        losses.append(loss)

        # Periodic validity check
        val_ratio = np.nan
        if ep % sample_every == 0:
            samples = gen.sample_decode(sample_n, max_len=sample_max_len)
            val_ratio = float(np.mean([
                1.0 if (isinstance(s, str) and Chem.MolFromSmiles(s) is not None) else 0.0
                for s in samples
            ]))
        valid_curve.append(val_ratio)

        # Early stopping
        if loss < target_loss:
            print(f"Reached target loss < {target_loss}. Early stop at epoch {ep}.")
            break

    epochs = len(losses)
    print(f"epochs={epochs}, last_loss={losses[-1]:.3f}, time={time.time()-t0:.1f}s")

    return gen, losses, valid_curve


def plot_training_curves(losses, valid_curve, output_dir):
    """Plot training loss and validity curves"""
    ep = np.arange(1, len(losses) + 1)

    # Extract non-NaN validity values
    val_ep = [i + 1 for i, v in enumerate(valid_curve) if not (isinstance(v, float) and np.isnan(v))]
    val_vals = [v for v in valid_curve if not (isinstance(v, float) and np.isnan(v))]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ep, losses, marker='o', label='train loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train loss (NLL)')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(val_ep, val_vals, marker='s', linestyle='--', color='tab:orange', label='validity (1k, every 2ep)')
    ax2.set_ylabel('Validity')
    ax2.set_ylim(0.0, 1.05)

    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    print(f"best loss = {min(losses):.3f} @ epoch {int(np.argmin(losses)+1)} | last = {losses[-1]:.3f}")
    if val_vals:
        print(f"best validity(1k) = {max(val_vals):.3f} @ epoch {val_ep[int(np.argmax(val_vals))]}")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved training curves to: {plot_path}")
    plt.close()


# ============================================================================
# 5. SAMPLING AND EVALUATION
# ============================================================================

def sample_molecules(ckpt_path, smivoc_inst, n_samples=20000, max_len=200):
    """Load checkpoint and generate molecules"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    gen_loaded = LSTMGenerator.construct_by_ckpt_dict(ckpt, smivoc_inst)

    print(f"Sampling {n_samples} molecules (max_len={max_len}) from: {ckpt_path}")
    samples = gen_loaded.sample_decode(n_samples, max_len=max_len)

    # Validate and canonicalize
    rows = []
    valid_cnt = 0
    for s in samples:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is not None:
            s_can = Chem.MolToSmiles(m, canonical=True)
            rows.append((s, s_can, 1))
            valid_cnt += 1
        else:
            rows.append((s if isinstance(s, str) else '', '', 0))

    out_df = pd.DataFrame(rows, columns=['smiles_raw', 'smiles_can', 'valid'])
    validity_ratio = valid_cnt / len(samples)
    print(f"Valid ratio: {validity_ratio:.3f}")

    return out_df, validity_ratio


def evaluate_molecules(out_df, trn_set, metrics_json):
    """Evaluate generated molecules and save metrics"""
    # Overall validity
    valid_mask = (out_df['valid'] == 1)
    validity_overall = float(valid_mask.mean())

    # Valid-only standard metrics
    valid_can = out_df.loc[valid_mask, 'smiles_can'].astype(str).tolist()
    std = standard_metrics(valid_can, trn_set=trn_set, subs_size=1000, n_jobs=2)

    # Compile metrics
    metrics = {
        "validity_overall": validity_overall,
        "uniqueness": float(std.get("uniqueness", np.nan)),
        "novelty": float(std.get("novelty", np.nan)),
        "intdiv": float(std.get("intdiv", np.nan)),
    }
    print(metrics)

    # Save metrics
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics:", metrics_json)

    return metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(data_csv=None, output_dir=None):
    # Configuration with flexible paths
    if data_csv is None:
        data_csv = os.path.join(PROJECT_ROOT, 'data', 'generative_data.csv')

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, 'results', 'generativemodel')

    # Create all necessary directories
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Configuration paths
    DATA_CSV = data_csv
    SPLIT_CSV = os.path.join(output_dir, 'molecules_split.csv')
    TOK_PATH = os.path.join(output_dir, 'my_tokens.txt')
    CKPT_TMPL = os.path.join(model_dir, 'lstm_e{}.ckpt')
    OUT_SAMPLES = os.path.join(output_dir, 'generated_molecules.csv')
    MET_JSON = os.path.join(output_dir, 'generation_metrics.json')

    # 1. Load and preprocess data
    print("\n" + "="*80)
    print("1. LOADING AND PREPROCESSING DATA")
    print("="*80)
    clean = load_and_clean_data(DATA_CSV)

    # 2. Create scaffold split
    print("\n" + "="*80)
    print("2. CREATING SCAFFOLD SPLIT")
    print("="*80)
    clean = create_scaffold_split(clean, seed=SEED)
    os.makedirs(os.path.dirname(SPLIT_CSV), exist_ok=True)
    clean[['id', 'smiles', 'split']].to_csv(SPLIT_CSV, index=False)
    print(f"Saved split to: {SPLIT_CSV}")

    # 3. Build vocabulary
    print("\n" + "="*80)
    print("3. BUILDING VOCABULARY")
    print("="*80)
    trn_smiles_all = clean.loc[clean['split'] == 'trn', 'smiles'].astype(str).tolist()
    smivoc_inst = build_vocabulary(trn_smiles_all, TOK_PATH)

    # 4. Prepare datasets
    print("\n" + "="*80)
    print("4. PREPARING DATASETS")
    print("="*80)
    trn_ds, vld_ds, tst_ds, trn_smiles = prepare_datasets(clean, smivoc_inst, len_cap=200)

    # 5. Train model
    print("\n" + "="*80)
    print("5. TRAINING MODEL")
    print("="*80)
    os.makedirs(os.path.dirname(CKPT_TMPL.format(0)), exist_ok=True)
    gen, losses, valid_curve = train_model(trn_ds, smivoc_inst, DEVICE, CKPT_TMPL)

    # 6. Plot training curves
    print("\n" + "="*80)
    print("6. PLOTTING TRAINING CURVES")
    print("="*80)
    plot_training_curves(losses, valid_curve, output_dir)

    # 7. Sample molecules (use last epoch)
    print("\n" + "="*80)
    print("7. SAMPLING MOLECULES")
    print("="*80)
    last_epoch = len(losses)
    ckpt_path = CKPT_TMPL.format(last_epoch)
    out_df, _ = sample_molecules(ckpt_path, smivoc_inst, n_samples=20000, max_len=200)
    out_df.to_csv(OUT_SAMPLES, index=False)
    print(f"Saved samples to: {OUT_SAMPLES}")

    # 8. Evaluate
    print("\n" + "="*80)
    print("8. EVALUATING MOLECULES")
    print("="*80)
    trn_set = set([s for s in trn_smiles if isinstance(s, str) and len(s) > 0])
    metrics = evaluate_molecules(out_df, trn_set, MET_JSON)

    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Samples: {OUT_SAMPLES}")
    print(f"Metrics: {MET_JSON}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generative Model for Molecular Design')
    parser.add_argument('--data', type=str, help='Path to input CSV file')
    parser.add_argument('--output', type=str, help='Output directory for results')
    args = parser.parse_args()

    main(data_csv=args.data, output_dir=args.output)
