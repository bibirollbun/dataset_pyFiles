import os
DATA_DIR = ''
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None:
    print("현재 코드는 Kaggle 노트북에서 실행 중입니다.")
    DATA_DIR = '/kaggle/input/playground-series-s5e11/'
else:
    print("현재 코드는 로컬 환경에서 실행 중입니다.")
    DATA_DIR = '/home/epikskool/workspace/workdir/datasets/ps5e11/'

# 또는 기본값을 지정하여 더 간단하게 사용할 수 있습니다.
env = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'localhost')
print(f"현재 실행 환경: {env}")


import os
import sys
import math
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

# ----------------------------- Paths & Logging -----------------------------
BASE_DIR = Path(f"{DATA_DIR}")
OUTPUT_DIR = Path(".")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "code_10_7_v9.txt"
SUB_PATH = OUTPUT_DIR / "submission.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
print("Purpose: Train FT-Transformer with 5-fold Stratified CV (multi-seed), leak-safe encoders, and dual TE+raw for QD numerics; emit submission CSV.")
print(f"Logs -> {LOG_FILE}")
print(f"Planned FULL-mode submission path -> {SUB_PATH}")

# ----------------------------- Config -----------------------------
DEBUG = True  # two-pass: DEBUG then FULL
SEED = 42

# Number of seeds (default 3 per user request; override via env N_SEEDS)
try:
    N_SEEDS = int(os.environ.get("N_SEEDS", "1"))
except Exception:
    N_SEEDS = 1
SEEDS = [SEED + i for i in range(N_SEEDS)]
print(f"Seed ensemble: N_SEEDS={N_SEEDS}, seeds={SEEDS}")

# FULL hyperparameters (24GB VRAM)
HP_FULL = dict(
    d_token=128,
    n_blocks=2,
    n_heads=4,
    lr=8e-4,
    weight_decay=1e-5,
    betas=(0.9, 0.98),
    eps=1e-6,
    attn_dropout=0.05,
    ffn_dropout=0.05,
    residual_dropout=0.05,
    d_ffn_factor=2.0,
    max_epochs=20,
    patience=3,
    batch_size=4096,
    grad_clip=0.99999999,
    warmup_ratio=0.03,
    min_lr=1e-5,
    # ---- MoE 추가 ----
    n_moe_experts=8,
    moe_capacity_factor=0.99999999,
    moe_aux_loss_alpha=1e-3,
)

# DEBUG hyperparameters
HP_DEBUG = dict(
    d_token=64,
    n_blocks=1,
    n_heads=4,
    lr=1e-3,
    weight_decay=1e-5,
    betas=(0.9, 0.98),
    eps=1e-8,
    attn_dropout=0.05,
    ffn_dropout=0.05,
    residual_dropout=0.05,
    d_ffn_factor=2.0,
    max_epochs=1,
    patience=1,
    batch_size=512,
    grad_clip=1.0,
    warmup_ratio=0.06,
    min_lr=1e-5,
    # ---- MoE DEBUG 축소 ----
    n_moe_experts=2,
    moe_capacity_factor=1.2,
    moe_aux_loss_alpha=0.0,
)

TARGET_COL = "loan_paid_back"
ID_COL = "id"

# Quasi-discrete numerics and canonical categorical (if present)
QD_NUMERICS_CANONICAL = [
    "debt_to_income_ratio", 
    "credit_score", 
    "interest_rate",
    # Add high-value engineered features for TE
    "financial_health_score",
    "payment_stress_index",
]

ENGINEERED_NUMERIC_COLS = [
    'income_debt_interest_ratio', 'income_debt_interest_score', 'interest_cost',
    'income_to_loan_ratio', 'credit_debt_ratio', 'loan_burden_ratio',
    'financial_health_score', 'payment_stress_index',
    'cross_loan_burden_ratio_payment_stress_index',
    'cross_credit_debt_ratio_financial_health_score',
]

CAT_CANONICAL = ["gender", "marital_status", "education_level", "employment_status", "loan_purpose", "grade_subgrade"]

# ----------------------------- Reproducibility -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ----------------------------- Device & AMP -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.cuda.is_available()
print(f"Device check -> {DEVICE.type.upper()}. Mixed precision: {'ON' if AMP else 'OFF'}. Purpose: Use CUDA if available; proceed conservatively otherwise.")

# ----------------------------- Data Loading -----------------------------
def read_competition_data(base_dir: Path):
    print("Purpose: Load train/test/sample CSV. Inputs: task/playground-series-s5e11/*.csv")
    train = pl.read_csv(base_dir / "train.csv")
    test = pl.read_csv(base_dir / "test.csv")
    sample = pl.read_csv(base_dir / "sample_submission.csv")
    print(f"Validation: train shape={train.shape}, test shape={test.shape}, sample shape={sample.shape}")
    return train, test, sample


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Apply feature engineering based on EDA insights"""
    
    # Basic interaction features
    df = df.with_columns([
        (pl.col('annual_income') * pl.col('debt_to_income_ratio') / pl.col('interest_rate')).alias('income_debt_interest_ratio'),
        ((pl.col('annual_income') * pl.col('debt_to_income_ratio') / pl.col('interest_rate')) * pl.col('credit_score')).alias('income_debt_interest_score'),
        (pl.col('loan_amount') * pl.col('interest_rate') / 100).alias('interest_cost'),
        (pl.col('annual_income') / pl.col('loan_amount')).alias('income_to_loan_ratio'),
        (pl.col('credit_score') / pl.col('debt_to_income_ratio')).alias('credit_debt_ratio'),
        (pl.col('loan_amount') / pl.col('annual_income')).alias('loan_burden_ratio'),
        (pl.col('credit_score') * (1 - pl.col('debt_to_income_ratio') / 100) * 
         (pl.col('annual_income') / pl.col('loan_amount'))).alias('financial_health_score'),
        (pl.col('loan_amount') * pl.col('interest_rate') / pl.col('annual_income') * 100).alias('payment_stress_index'),
    ])
    
    # Correlation-based features (from EDA)
    # Calculate correlation strength features for highly correlated columns
    high_corr_cols = ['loan_amount', 'interest_cost', 'credit_score', 'debt_to_income_ratio']
    
    for col in high_corr_cols:
        if col in df.columns:
            df = df.with_columns([
                (pl.col('annual_income') * pl.col(col)).alias(f'income_weighted_{col}'),
                (pl.col('annual_income') / (pl.col(col) + 1)).alias(f'income_ratio_{col}')
            ])
    
    # Cross-correlation features
    df = df.with_columns([
        (pl.col('loan_burden_ratio') * pl.col('payment_stress_index')).alias('cross_loan_burden_ratio_payment_stress_index'),
        (pl.col('credit_debt_ratio') * pl.col('financial_health_score')).alias('cross_credit_debt_ratio_financial_health_score'),
    ])
    
    return df

train_df, test_df, sample_df = read_competition_data(BASE_DIR)
train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

assert TARGET_COL in train_df.columns, f"Target column '{TARGET_COL}' not found in train.csv"
assert ID_COL in train_df.columns and ID_COL in test_df.columns, "ID column missing"

y_full = train_df[TARGET_COL].cast(pl.Int32).to_numpy() # train_df[TARGET_COL].astype(int).values
if not set(np.unique(y_full)).issubset({0, 1}):
    print("Target not strictly {0,1}; binarizing at 0.5.")
    y_full = (train_df[TARGET_COL].cast(pl.Float64).to_numpy() >= 0.5).astype(int) # y_full = (train_df[TARGET_COL].astype(float).values >= 0.5).astype(int)

# ----------------------------- Schema Inference (global preview only) -----------------------------
present_qd = [c for c in QD_NUMERICS_CANONICAL if c in train_df.columns]
present_cat = [c for c in CAT_CANONICAL if c in train_df.columns]

# Get categorical columns by dtype
dtype_cats = [c for c in train_df.columns 
              if train_df[c].dtype in [pl.Utf8, pl.Categorical] 
              and c not in [TARGET_COL, ID_COL]]
cat_cols_global = list(dict.fromkeys(present_cat + dtype_cats))

# Get numeric columns
numeric_candidates_global = [c for c in train_df.columns 
                             if train_df[c].dtype.is_numeric() 
                             and c not in [TARGET_COL, ID_COL]]

# dtype_cats = [c for c in train_df.columns if (train_df[c].dtype == "object" or str(train_df[c].dtype).startswith("category")) and c not in [TARGET_COL, ID_COL]]
# cat_cols_global = list(dict.fromkeys(present_cat + dtype_cats))
# numeric_candidates_global = [c for c in train_df.columns if (np.issubdtype(train_df[c].dtype, np.number)) and c not in [TARGET_COL, ID_COL]]
print(f"Global schema preview: numeric≈{len(numeric_candidates_global)}, cats≈{len(cat_cols_global)}, present_qd={present_qd}")

# ----------------------------- Encoders & Transforms -----------------------------
NA_CAT_TOKEN = "__NA__"
UNK_CAT_TOKEN = "__UNK__"
NA_STR_SET = {"", "nan", "none", "null", "na", "n/a"}

def canonicalize_cat_series(s: pl.Series) -> pl.Series:
    """Convert series to string, handle nulls and NA-like values"""
    ser = s.cast(pl.Utf8).str.strip_chars()
    # Replace nulls and NA-like strings with NA_CAT_TOKEN
    ser = ser.fill_null(NA_CAT_TOKEN)
    ser = pl.when(ser.str.to_lowercase().is_in(list(NA_STR_SET))).then(pl.lit(NA_CAT_TOKEN)).otherwise(ser)
    return ser

def build_cat_vocabs(df: pl.DataFrame, cat_cols: List[str]) -> Dict[str, Dict[str, int]]:
    """Build vocabulary mappings for categorical columns.
    
    Maps unique values to integer indices, with UNK_CAT_TOKEN=0 and sorted unique values starting at 1.
    
    Args:
        df: Polars DataFrame containing the categorical columns
        cat_cols: List of categorical column names to process
        
    Returns:
        Dictionary mapping column names to {value: index} dictionaries
    """
    vocabs: Dict[str, Dict[str, int]] = {}
    for c in cat_cols:
        # Get canonicalized series
        vals = canonicalize_cat_series(df[c])
        
        # Get unique values and sort - need to materialize the result
        # Method 1: Use select to get a series
        uniq_series = df.select(canonicalize_cat_series(df[c]).unique().sort()).to_series()
        
        # Or Method 2: Work directly with the series
        # uniq_list = sorted(vals.unique().to_list())
        
        mapping = {UNK_CAT_TOKEN: 0}
        for i, v in enumerate(uniq_series.to_list(), start=1):
            mapping[str(v)] = i
        
        if NA_CAT_TOKEN not in mapping:
            mapping[NA_CAT_TOKEN] = len(mapping)
        
        vocabs[c] = mapping
    
    return vocabs

def apply_cat_vocabs(df: pl.DataFrame, cat_cols: List[str], vocabs: Dict[str, Dict[str, int]]) -> np.ndarray:
    """Apply vocabulary mappings to convert categorical columns to integer indices."""
    if len(cat_cols) == 0:
        return np.zeros((len(df), 0), dtype=np.int64)
    
    # Materialize all canonicalized columns at once
    canon_df = df.select([
        canonicalize_cat_series(df[c]).alias(c) for c in cat_cols
    ])
    
    mats = []
    for c in cat_cols:
        mapping = vocabs[c]
        arr = canon_df[c].to_numpy()
        idx = np.array([mapping.get(v, mapping[UNK_CAT_TOKEN]) for v in arr], dtype=np.int64)
        mats.append(idx.reshape(-1, 1))
    
    return np.concatenate(mats, axis=1)

def quantize_interest_rate_series(s: pl.Series, step: float = 0.25) -> pl.Series:
    """Quantize numeric series to step intervals, format as string"""
    x = s.cast(pl.Float64, strict=False)
    q = ((x / step).round() * step)
    return pl.when(q.is_null()).then(pl.lit("NaNLevel")).otherwise(q.cast(pl.Utf8).str.slice(0, 5))  # Format as string



def m_estimate_mapping(count: int, pos: int, prior: float, m: float = 5.0) -> float:
    return (pos + m * prior) / (count + m)

def cross_fit_m_estimate_oof(
    df: pl.DataFrame,
    y: np.ndarray,
    col: str,
    n_splits: int = 5,
    m: float = 5.0,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]], float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    prior = float(np.mean(y))
    
    # Create aggregation for full map
    tmp_all = pl.DataFrame({
        col: df[col].cast(pl.Utf8).fill_null("NaNLevel"),
        "_y": y
    })
    agg_all = tmp_all.group_by(col).agg([
        pl.col("_y").count().alias("count"),
        pl.col("_y").sum().alias("sum")
    ])
    
    # Convert to dict
    full_map: Dict[str, Tuple[int, int]] = {}
    for row in agg_all.iter_rows(named=True):
        key = str(row[col])
        full_map[key] = (int(row["count"]), int(row["sum"]))
    
    # OOF encoding
    oof = np.zeros(len(df), dtype=np.float32)
    df_np = df.to_numpy()  # Convert once for indexing
    
    for tr_idx, va_idx in skf.split(df, y):
        # Create train aggregation
        tmp_tr = pl.DataFrame({
            col: df[col].cast(pl.Utf8).fill_null("NaNLevel").gather(tr_idx),
            "_y": y[tr_idx]
        })
        agg_tr = tmp_tr.group_by(col).agg([
            pl.col("_y").count().alias("count"),
            pl.col("_y").sum().alias("sum")
        ])
        
        tr_map: Dict[str, Tuple[int, int]] = {}
        for row in agg_tr.iter_rows(named=True):
            key = str(row[col])
            tr_map[key] = (int(row["count"]), int(row["sum"]))
        
        vals_va = df[col].cast(pl.Utf8).fill_null("NaNLevel").gather(va_idx).to_numpy()
        enc = np.array(
            [m_estimate_mapping(tr_map[v][0], tr_map[v][1], prior, m) 
             if v in tr_map else prior for v in vals_va],
            dtype=np.float32,
        )
        oof[va_idx] = enc
    
    return oof, full_map, prior

def apply_m_estimate_map(df: pl.DataFrame, col: str, full_map: Dict[str, Tuple[int, int]], 
                         prior: float, m: float = 5.0) -> np.ndarray:
    vals = df[col].cast(pl.Utf8).fill_null("NaNLevel").to_numpy()
    out = np.empty(len(vals), dtype=np.float32)
    for i, v in enumerate(vals):
        if v in full_map:
            cnt, pos = full_map[v]
            out[i] = m_estimate_mapping(cnt, pos, prior, m)
        else:
            out[i] = prior
    return out

def standardize_train_valid_test(
    Xtr: pl.DataFrame, Xva: pl.DataFrame, Xte: pl.DataFrame, cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Tuple[float, float]]]:
    stats = {}
    def z(x, mu, sd):
        return (x - mu) / sd if sd > 0 else x - mu
    
    Xtr_out, Xva_out, Xte_out = [], [], []
    for c in cols:
        # Use polars for statistics
        tr_vals = Xtr[c].to_numpy()
        mu = float(np.nanmean(tr_vals))
        sd = float(np.nanstd(tr_vals))
        sd = sd if sd > 1e-12 else 1.0
        stats[c] = (mu, sd)
        
        Xtr_out.append(z(tr_vals, mu, sd).reshape(-1, 1))
        Xva_out.append(z(Xva[c].to_numpy(), mu, sd).reshape(-1, 1))
        Xte_out.append(z(Xte[c].to_numpy(), mu, sd).reshape(-1, 1))
    
    if len(cols) == 0:
        return (np.zeros((len(Xtr), 0), np.float32), 
                np.zeros((len(Xva), 0), np.float32), 
                np.zeros((len(Xte), 0), np.float32), 
                stats)
    return (
        np.concatenate(Xtr_out, axis=1).astype(np.float32),
        np.concatenate(Xva_out, axis=1).astype(np.float32),
        np.concatenate(Xte_out, axis=1).astype(np.float32),
        stats,
    )

# ----------------------------- Dataset -----------------------------
class TabDataset(Dataset):
    def __init__(self, X_num: np.ndarray, X_cat: np.ndarray, y: Optional[np.ndarray] = None):
        n = len(y) if y is not None else (X_num.shape[0] if X_num is not None else X_cat.shape[0])
        self.X_num = (X_num.astype(np.float32) if X_num is not None else np.zeros((n, 0), dtype=np.float32))
        self.X_cat = (X_cat.astype(np.int64) if X_cat is not None else np.zeros((n, 0), dtype=np.int64))
        self.y = None if y is None else y.astype(np.float32).reshape(-1, 1)
    def __len__(self): return self.X_num.shape[0]
    def __getitem__(self, idx):
        if self.y is None: return self.X_num[idx], self.X_cat[idx]
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

# ----------------------------- Minimal FT-Transformer -----------------------------
class FeatureTokenizer(nn.Module):
    def __init__(self, n_num_features: int, cat_cardinalities: Optional[List[int]], d_token: int):
        super().__init__()
        self.n_num = int(n_num_features)
        self.d_token = int(d_token)
        self.has_cat = cat_cardinalities is not None and len(cat_cardinalities) > 0
        if self.n_num > 0:
            self.num_weight = nn.Parameter(torch.empty(self.n_num, self.d_token))
            self.num_bias = nn.Parameter(torch.empty(self.n_num, self.d_token))
            nn.init.kaiming_uniform_(self.num_weight, a=math.sqrt(5))
            nn.init.uniform_(self.num_bias, -1e-3, 1e-3)
        else:
            self.register_parameter("num_weight", None)
            self.register_parameter("num_bias", None)
        if self.has_cat:
            self.embeddings = nn.ModuleList([nn.Embedding(int(c), self.d_token) for c in cat_cardinalities])
            for emb in self.embeddings:
                nn.init.kaiming_uniform_(emb.weight, a=math.sqrt(5))
        else:
            self.embeddings = nn.ModuleList()

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        tokens = []
        if self.n_num > 0 and x_num is not None and x_num.numel() > 0:
            x = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
            tokens.append(x)
        if self.has_cat and x_cat is not None and x_cat.shape[1] > 0:
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            if len(embs) > 0:
                tokens.append(torch.stack(embs, dim=1))
        if len(tokens) == 0:
            B = x_num.shape[0] if x_num is not None else x_cat.shape[0]
            return torch.zeros(B, 0, self.d_token, device=x_num.device if x_num is not None else x_cat.device)
        return torch.cat(tokens, dim=1)

class ReGLU_FFN(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 2 * d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_in)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        nn.init.uniform_(self.fc1.bias, -1e-3, 1e-3)
        nn.init.uniform_(self.fc2.bias, -1e-3, 1e-3)
    def forward(self, x):
        u, v = self.fc1(x).chunk(2, dim=-1)
        x = F.relu(u) * v
        x = self.dropout(self.fc2(x))
        return x

class SwitchFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, n_experts: int, dropout: float, capacity_factor: float):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(d_model, n_experts)
        nn.init.kaiming_uniform_(self.router.weight, a=math.sqrt(5))
        nn.init.uniform_(self.router.bias, -1e-3, 1e-3)
        self.experts = nn.ModuleList([
            ReGLU_FFN(d_model, d_hidden, dropout) for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,T,d)
        B, T, D = x.shape
        logits = self.router(x)                # (B,T,E)
        gates = F.softmax(logits, dim=-1)      # (B,T,E)
        expert_idx = gates.argmax(dim=-1)      # (B,T)
        # Capacity 계산
        total_tokens = B * T
        cap = max(1, int(self.capacity_factor * total_tokens / self.n_experts))
        # 마스크 및 카운트
        outputs = torch.zeros_like(x)
        aux_loss = x.new_tensor(0.0)
        # Load balance 손실 계산 (미니배치 평균)
        probs_mean = gates.mean(dim=(0,1))     # (E,)
        assign_frac = torch.zeros(self.n_experts, device=x.device)
        for e in range(self.n_experts):
            mask_e = (expert_idx == e)         # (B,T)
            idx_e = mask_e.nonzero(as_tuple=False)  # (N_e,2)
            N_e = idx_e.shape[0]
            assign_frac[e] = float(N_e) / total_tokens
            if N_e == 0:
                continue
            if N_e > cap:
                idx_e = idx_e[:cap]            # 드롭 초과 토큰
                N_e = cap
            # Gather
            tokens_e = x[idx_e[:,0], idx_e[:,1], :]  # (N_e,d)
            out_e = self.experts[e](tokens_e)        # (N_e,d)
            if out_e.dtype != outputs.dtype:
                out_e = out_e.to(outputs.dtype)
            # Scatter back
            outputs[idx_e[:,0], idx_e[:,1], :] = out_e
        # Switch balance loss: E * sum_e p_e f_e
        balance_loss = self.n_experts * torch.sum(probs_mean * assign_frac)
        aux_loss = balance_loss
        return outputs, aux_loss

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, attn_dropout: float,
                 ffn_dropout: float, residual_dropout: float,
                 n_moe_experts: int = 0, moe_capacity_factor: float = 1.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                          dropout=attn_dropout, batch_first=True)
        self.drop_res1 = nn.Dropout(residual_dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.use_moe = n_moe_experts and n_moe_experts > 1
        if self.use_moe:
            self.ffn = SwitchFFN(d_model, d_ffn, n_moe_experts, ffn_dropout, moe_capacity_factor)
        else:
            self.ffn = ReGLU_FFN(d_model, d_ffn, ffn_dropout)
        self.drop_res2 = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        aux_loss = x.new_tensor(0.0)
        x_attn = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
        x = x + self.drop_res1(x_attn)
        x_norm = self.ln2(x)
        if self.use_moe:
            ffn_out, aux = self.ffn(x_norm)
            aux_loss = aux
        else:
            ffn_out = self.ffn(x_norm)
        x = x + self.drop_res2(ffn_out)
        return x, aux_loss

# class TransformerBlock(nn.Module):
#     def __init__(self, d_model: int, n_heads: int, d_ffn: int, attn_dropout: float, ffn_dropout: float, residual_dropout: float):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(d_model)
#         self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=attn_dropout, batch_first=True)
#         self.drop_res1 = nn.Dropout(residual_dropout)
#         self.ln2 = nn.LayerNorm(d_model)
#         self.ffn = ReGLU_FFN(d_model, d_ffn, ffn_dropout)
#         self.drop_res2 = nn.Dropout(residual_dropout)
#     def forward(self, x):
#         x_attn = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
#         x = x + self.drop_res1(x_attn)
#         x_ffn = self.ffn(self.ln2(x))
#         x = x + self.drop_res2(x_ffn)
#         return x

# class FTTransformer(nn.Module):
#     def __init__(self, n_num_features: int, cat_cardinalities: Optional[List[int]], d_token: int, n_blocks: int, n_heads: int, d_ffn: int,
#                  attn_dropout: float, ffn_dropout: float, residual_dropout: float, d_out: int = 1):
#         super().__init__()
#         assert d_token % n_heads == 0, f"d_token must be divisible by n_heads; got {d_token} % {n_heads}"
#         self.tokenizer = FeatureTokenizer(n_num_features, cat_cardinalities, d_token)
#         self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
#         nn.init.uniform_(self.cls, -1e-3, 1e-3)
#         self.blocks = nn.Sequential(*[
#             TransformerBlock(d_token, n_heads, d_ffn, attn_dropout, ffn_dropout, residual_dropout)
#             for _ in range(n_blocks)
#         ])
#         self.head_norm = nn.LayerNorm(d_token)
#         self.head = nn.Linear(d_token, d_out)
#         nn.init.kaiming_uniform_(self.head.weight, a=math.sqrt(5))
#         nn.init.uniform_(self.head.bias, -1e-3, 1e-3)
#     def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
#         x_tokens = self.tokenizer(x_num, x_cat)  # (B, T, d)
#         B = x_tokens.shape[0]
#         cls = self.cls.expand(B, -1, -1)
#         x = torch.cat([cls, x_tokens], dim=1)
#         x = self.blocks(x)
#         x = self.head_norm(x[:, 0, :])
#         return self.head(x)
    
class FTTransformer(nn.Module):
    def __init__(self, n_num_features: int, cat_cardinalities: Optional[List[int]], d_token: int,
                 n_blocks: int, n_heads: int, d_ffn: int,
                 attn_dropout: float, ffn_dropout: float, residual_dropout: float,
                 d_out: int = 1, n_moe_experts: int = 0, moe_capacity_factor: float = 1.0):
        super().__init__()
        assert d_token % n_heads == 0
        self.tokenizer = FeatureTokenizer(n_num_features, cat_cardinalities, d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.uniform_(self.cls, -1e-3, 1e-3)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ffn, attn_dropout, ffn_dropout,
                             residual_dropout, n_moe_experts=n_moe_experts,
                             moe_capacity_factor=moe_capacity_factor)
            for _ in range(n_blocks)
        ])
        self.head_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, d_out)
        nn.init.kaiming_uniform_(self.head.weight, a=math.sqrt(5))
        nn.init.uniform_(self.head.bias, -1e-3, 1e-3)

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_tokens = self.tokenizer(x_num, x_cat)
        B = x_tokens.shape[0]
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x_tokens], dim=1)
        aux_total = x.new_tensor(0.0)
        for blk in self.blocks:
            x, aux = blk(x)
            aux_total = aux_total + aux
        x = self.head_norm(x[:, 0, :])
        return self.head(x), aux_total

# ----------------------------- Scheduler -----------------------------
def make_warmup_cosine(total_steps: int, warmup_ratio: float = 0.06, min_lr_ratio: float = 0.01):
    warmup = max(1, int(total_steps * warmup_ratio))
    def lr_mult(step: int):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return lr_mult

# ----------------------------- Training helpers -----------------------------
def train_one_fold(
    tr_df: pl.DataFrame,  # Changed from pd.DataFrame
    va_df: pl.DataFrame,  # Changed from pd.DataFrame
    te_df: pl.DataFrame,  # Changed from pd.DataFrame
    y_tr: np.ndarray,
    y_va: np.ndarray,
    hp: dict,
    fold_idx: int,
    seed: int,
    debug_mode: bool,
    is_full_mode_first_fold: bool,
) -> Tuple[np.ndarray, np.ndarray, float, int, bool]:
    # Schema detection
    present_qd = [c for c in QD_NUMERICS_CANONICAL if c in tr_df.columns]
    present_cat = [c for c in CAT_CANONICAL if c in tr_df.columns]
    
    dtype_cats = [c for c in tr_df.columns 
                  if tr_df[c].dtype in [pl.Utf8, pl.Categorical] 
                  and c not in [TARGET_COL, ID_COL]]
    cat_cols = list(dict.fromkeys(present_cat + dtype_cats))

    numeric_candidates = [c for c in tr_df.columns 
                         if tr_df[c].dtype.is_numeric() 
                         and c not in [TARGET_COL, ID_COL]]
    num_cols = list(numeric_candidates)

    # Quantize interest_rate
    ir_q_col = None
    if "interest_rate" in present_qd:
        ir_q_col = "interest_rate_q"
        tr_df = tr_df.with_columns(quantize_interest_rate_series(tr_df["interest_rate"]).alias(ir_q_col))
        va_df = va_df.with_columns(quantize_interest_rate_series(va_df["interest_rate"]).alias(ir_q_col))
        te_df = te_df.with_columns(quantize_interest_rate_series(te_df["interest_rate"]).alias(ir_q_col))

    # Columns to TE (cross-fitted): the two native QD numerics and the quantized interest rate (if present)
    qd_te_cols: List[str] = []
    for c in ["debt_to_income_ratio", "credit_score"]:
        if c in present_qd:
            qd_te_cols.append(c)
    if ir_q_col is not None:
        qd_te_cols.append(ir_q_col)

    # Fit categorical vocabs on train-fold only
    vocabs = build_cat_vocabs(tr_df, cat_cols)
    cat_cardinalities = [max(vocabs[c].values()) + 1 for c in cat_cols]
    print(f"[fold{fold_idx}|seed{seed}] Categorical: {len(cat_cols)} cols; first few cardinalities={cat_cardinalities[:6]}")

    # Cross-fitted TE on train-fold; apply to valid/test
    te_maps = {}; te_prior = {}; m_value = 5.0
    te_tr_feats = []; te_va_feats = []; te_te_feats = []
    for c in qd_te_cols:
        oof, full_map, prior = cross_fit_m_estimate_oof(tr_df, y_tr, c, n_splits=5, m=m_value, seed=seed)
        te_maps[c] = full_map; te_prior[c] = prior
        te_tr_feats.append(oof.reshape(-1, 1))
        te_va_feats.append(apply_m_estimate_map(va_df, c, full_map, prior, m=m_value).reshape(-1, 1))
        te_te_feats.append(apply_m_estimate_map(te_df, c, full_map, prior, m=m_value).reshape(-1, 1))
    Xtr_te = np.concatenate(te_tr_feats, axis=1) if te_tr_feats else np.zeros((len(tr_df), 0), np.float32)
    Xva_te = np.concatenate(te_va_feats, axis=1) if te_va_feats else np.zeros((len(va_df), 0), np.float32)
    Xte_te = np.concatenate(te_te_feats, axis=1) if te_te_feats else np.zeros((len(te_df), 0), np.float32)
    if len(qd_te_cols) > 0:
        print(f"[fold{fold_idx}|seed{seed}] TE m={m_value} on {qd_te_cols}; priors={[round(te_prior[c],4) for c in qd_te_cols]}")

    # Standardize numerics (includes raw QD numerics for dual representation)
    Xtr_num, Xva_num, Xte_num, zstats = standardize_train_valid_test(tr_df, va_df, te_df, num_cols)
    print(f"[fold{fold_idx}|seed{seed}] Standardized numerics={len(num_cols)}; example stats={list(zstats.items())[:3]}")

    # Combine numeric features: raw standardized + TE for QD numerics
    Xtr_num_all = np.concatenate([Xtr_num, Xtr_te], axis=1) if Xtr_te.shape[1] else Xtr_num
    Xva_num_all = np.concatenate([Xva_num, Xva_te], axis=1) if Xva_te.shape[1] else Xva_num
    Xte_num_all = np.concatenate([Xte_num, Xte_te], axis=1) if Xte_te.shape[1] else Xte_num

    # Categorical indices (canonicalized NA handling)
    Xtr_cat = apply_cat_vocabs(tr_df, cat_cols, vocabs)
    Xva_cat = apply_cat_vocabs(va_df, cat_cols, vocabs)
    Xte_cat = apply_cat_vocabs(te_df, cat_cols, vocabs)

    n_num_features = Xtr_num_all.shape[1]
    print(f"[fold{fold_idx}|seed{seed}] Final tokens -> numeric={n_num_features}, categorical={len(cat_cols)}")

    # DataLoaders
    class TabDS(torch.utils.data.Dataset):
        def __init__(self, Xn, Xc, y=None):
            self.Xn = torch.from_numpy(Xn).float()
            self.Xc = torch.from_numpy(Xc).long() if Xc.shape[1] > 0 else torch.zeros((Xn.shape[0], 0), dtype=torch.long)
            self.y = None if y is None else torch.from_numpy(y.astype(np.float32)).view(-1, 1)
        def __len__(self): return self.Xn.shape[0]
        def __getitem__(self, i):
            if self.y is None: return self.Xn[i], self.Xc[i]
            return self.Xn[i], self.Xc[i], self.y[i]

    dl_tr = DataLoader(TabDS(Xtr_num_all, Xtr_cat, y_tr), batch_size=hp["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(TabDS(Xva_num_all, Xva_cat, y_va), batch_size=hp["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    dl_te = DataLoader(TabDS(Xte_num_all, Xte_cat, None), batch_size=hp["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # Build model
    model = FTTransformer(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_token=hp["d_token"],
        n_blocks=hp["n_blocks"],
        n_heads=hp["n_heads"],
        d_ffn=int(hp["d_token"] * hp["d_ffn_factor"]),
        attn_dropout=hp["attn_dropout"],
        ffn_dropout=hp["ffn_dropout"],
        residual_dropout=hp["residual_dropout"],
        d_out=1,
        n_moe_experts=hp.get("n_moe_experts", 0),
        moe_capacity_factor=hp.get("moe_capacity_factor", 1.0),
    ).to(DEVICE)

    # Optimizer, loss, scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], betas=hp["betas"], eps=hp["eps"], weight_decay=hp["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    total_steps = max(1, len(dl_tr) * hp["max_epochs"])
    lr_lambda = make_warmup_cosine(total_steps, warmup_ratio=hp["warmup_ratio"], min_lr_ratio=hp["min_lr"] / hp["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    def eval_auc(dloader):
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for xb_num, xb_cat, yb in dloader:
                xb_num = xb_num.to(DEVICE, non_blocking=True)
                xb_cat = xb_cat.to(DEVICE, non_blocking=True) if xb_cat.shape[1] > 0 else None
                yb = yb.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=AMP):
                    logits, aux_loss = model(xb_num, xb_cat)
                    # Ensure same dtype under autocast
                    if yb.dtype != logits.dtype:
                        yb = yb.to(dtype=logits.dtype)
                    bce_loss = loss_fn(logits, yb)
                    loss = bce_loss + hp.get("moe_aux_loss_alpha", 0.0) * aux_loss
                preds.append(torch.sigmoid(logits).detach().cpu().numpy().ravel())
                ys.append(yb.cpu().numpy().ravel())
        p = np.concatenate(preds); y = np.concatenate(ys)
        try:
            return roc_auc_score(y, p), p
        except ValueError:
            return float("nan"), p

    best_auc, best_epoch, best_state = -1.0, -1, None
    epochs_no_improve = 0
    nan_loss_flag = False
    t0 = time.time()

    for epoch in range(1, hp["max_epochs"] + 1):
        model.train()
        epoch_loss, n_seen = 0.0, 0
        for xb_num, xb_cat, yb in dl_tr:
            xb_num = xb_num.to(DEVICE, non_blocking=True)
            xb_cat = xb_cat.to(DEVICE, non_blocking=True) if xb_cat.shape[1] > 0 else None
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=AMP):
                logits, aux_loss = model(xb_num, xb_cat)
                bce_loss = loss_fn(logits, yb)
                loss = bce_loss + hp.get("moe_aux_loss_alpha", 0.0) * aux_loss
            if torch.isnan(loss):
                nan_loss_flag = True
            scaler.scale(loss).backward()
            if hp["grad_clip"] is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            bs = yb.shape[0]
            epoch_loss += loss.item() * bs
            n_seen += bs

        train_loss = epoch_loss / max(1, n_seen)
        val_auc, _ = eval_auc(dl_va)
        print(f"[fold{fold_idx}|seed{seed}] Epoch {epoch}/{hp['max_epochs']} - train_loss={train_loss:.5f} | val_auc={val_auc:.6f} | lr={optimizer.param_groups[0]['lr']:.6f}")

        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= hp["patience"]:
                print(f"[fold{fold_idx}|seed{seed}] Early stopping at epoch {epoch}. Best AUC={best_auc:.6f} @ epoch {best_epoch}")
                break

        if (not debug_mode) and is_full_mode_first_fold and epoch == 1 and nan_loss_flag:
            logging.warning("[FULL] NaN loss detected after 1st epoch on fold 0. Aborting remaining training and proceeding to inference.")
            break

    t1 = time.time()
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[fold{fold_idx}|seed{seed}] Best val AUC={best_auc:.6f}; best_epoch={best_epoch}; fold_train_time_sec={(t1 - t0):.1f}")

    # Final valid preds
    model.eval()
    va_preds = []
    with torch.no_grad():
        for xb_num, xb_cat, yb in dl_va:
            xb_num = xb_num.to(DEVICE, non_blocking=True)
            xb_cat = xb_cat.to(DEVICE, non_blocking=True) if xb_cat.shape[1] > 0 else None
            with torch.cuda.amp.autocast(enabled=AMP):
                logits, _ = model(xb_num, xb_cat)
            va_preds.append(torch.sigmoid(logits).detach().cpu().numpy().ravel())
    va_probs = np.concatenate(va_preds)

    # Test preds
    te_preds = []
    with torch.no_grad():
        for xb_num, xb_cat in dl_te:
            xb_num = xb_num.to(DEVICE, non_blocking=True)
            xb_cat = xb_cat.to(DEVICE, non_blocking=True) if xb_cat.shape[1] > 0 else None
            with torch.cuda.amp.autocast(enabled=AMP):
                logits, _ = model(xb_num, xb_cat)
            te_preds.append(torch.sigmoid(logits).detach().cpu().numpy().ravel())
    te_probs = np.concatenate(te_preds)

    abort_all = (not debug_mode) and is_full_mode_first_fold and nan_loss_flag
    return va_probs, te_probs, best_auc, best_epoch, abort_all

# ----------------------------- CV runner (DEBUG/FULL) -----------------------------
def run_one_mode(debug: bool):
    mode = "DEBUG" if debug else "FULL"
    print(f"====================== Running mode: {mode} ======================")
    hp = HP_DEBUG if debug else HP_FULL

    n = len(train_df)
    global_oof_sum = np.zeros(n, dtype=np.float64)
    global_test_sum = np.zeros(len(test_df), dtype=np.float64)
    per_seed_oof_aucs = []

    for si, seed in enumerate(SEEDS):
        set_seed(seed)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(n, dtype=np.float64)
        test_sum_folds = np.zeros(len(test_df), dtype=np.float64)
        fold_aucs = []
        abort_all = False

        # Convert to pandas temporarily for sklearn split (or use indices directly)
        train_indices = np.arange(len(train_df))
        
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_indices, y_full)):
            # Use polars indexing (gather)
            tr_df = train_df[tr_idx]
            va_df = train_df[va_idx]
            y_tr = y_full[tr_idx]
            y_va = y_full[va_idx]
            
            print(f"[seed{seed}] Fold {fold_idx}: train={len(tr_df)}, valid={len(va_df)}")

            # DEBUG sampling
            if debug:
                n_debug = min(1000, len(tr_df))
                if n_debug <= 0.5 * len(tr_df):
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=len(tr_df) - n_debug, random_state=seed)
                    keep_idx, _ = next(sss.split(np.zeros(len(y_tr)), y_tr))
                    tr_df = tr_df[keep_idx]
                    y_tr = y_tr[keep_idx]
                    print(f"[seed{seed}] Fold {fold_idx}: DEBUG sampling -> train size={len(tr_df)}")

            va_probs, te_probs, best_auc, best_epoch, abort_all = train_one_fold(
                tr_df=tr_df, va_df=va_df, te_df=test_df, y_tr=y_tr, y_va=y_va, hp=hp,
                fold_idx=fold_idx, seed=seed, debug_mode=debug, is_full_mode_first_fold=(fold_idx == 0)
            )

            oof[va_idx] = va_probs
            test_sum_folds += te_probs
            fold_aucs.append(best_auc)
            print(f"[seed{seed}] Fold {fold_idx} complete: val_auc={best_auc:.6f}, best_epoch={best_epoch}")

            if abort_all:
                logging.warning(f"[seed{seed}] Aborting remaining folds due to NaN-loss guard (FULL mode, fold 0 epoch 1).")
                break

        # Per-seed aggregation
        seed_oof_auc = roc_auc_score(y_full[:len(oof)], oof)
        per_seed_oof_aucs.append(seed_oof_auc)
        global_oof_sum += oof
        n_folds_executed = len(fold_aucs)
        test_avg_folds = test_sum_folds / max(1, n_folds_executed)
        global_test_sum += test_avg_folds

        print(f"[seed{seed}] OOF AUC={seed_oof_auc:.6f} over {n_folds_executed} folds; per-fold AUCs={['{:.6f}'.format(a) for a in fold_aucs]}")

    # Seed-averaged aggregates
    oof_mean = global_oof_sum / len(SEEDS)
    overall_oof_auc = roc_auc_score(y_full, oof_mean)
    test_mean = global_test_sum / len(SEEDS)
    print(f"[{mode}] Overall OOF AUC (seed-averaged)={overall_oof_auc:.6f}; per-seed OOF AUCs={['{:.6f}'.format(a) for a in per_seed_oof_aucs]}")

    # Output handling
    if debug:
        print("[DEBUG] Skipping submission write as per guidelines.")
    else:
        sub = pl.DataFrame({
            ID_COL: test_df[ID_COL],
            TARGET_COL: np.clip(test_mean, 1e-6, 1 - 1e-6)
        })
        sub.write_csv(SUB_PATH)
        print(f"[FULL] Submission written: {SUB_PATH}")
        pct = np.percentile(test_mean, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        print(f"[FULL] Prediction summary: min={pct[0]:.6f}, p1={pct[1]:.6f}, p5={pct[2]:.6f}, "
                     f"p25={pct[3]:.6f}, median={pct[4]:.6f}, p75={pct[5]:.6f}, p95={pct[6]:.6f}, p99={pct[7]:.6f}, max={pct[8]:.6f}")

# ----------------------------- Execute -----------------------------
print("Purpose: Execute two passes: (1) DEBUG sanity-check (no submission), (2) FULL CV training + inference with N_SEEDS ensembling. Metric: ROC-AUC.")
hf_token = os.environ.get("HF_TOKEN", "")
if hf_token:
    print("HF_TOKEN detected in environment (not used).")

# Pass 1: DEBUG
# run_one_mode(debug=True)
# Pass 2: FULL
run_one_mode(debug=False)

