%%writefile grouped_query_attention.py

"""
Grouped Query Attention (GQA) - Pure PyTorch Implementation

GQA is a memory-efficient attention mechanism that reduces the number of key-value
heads while maintaining multiple query heads. This balances the trade-off between
Multi-Head Attention (MHA) and Multi-Query Attention (MQA).

Architecture:
- MHA: num_query_heads == num_kv_heads (e.g., 8 Q heads, 8 KV heads)
- GQA: num_query_heads > num_kv_heads (e.g., 8 Q heads, 2 KV heads)
- MQA: num_kv_heads == 1 (e.g., 8 Q heads, 1 KV head)

Reference:
- "GQA: Training Generalized Multi-Query Transformer Models" (Ainslie et al., 2023)
- https://arxiv.org/abs/2305.13245

Usage:
    >>> attn = GroupedQueryAttention(embed_dim=512, num_heads=8, num_kv_heads=2)
    >>> x = torch.randn(32, 10, 512)  # (batch, seq_len, embed_dim)
    >>> out, attn_weights = attn(x, x, x)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) layer.

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of query heads (must be divisible by num_kv_heads)
        num_kv_heads: Number of key/value heads (default: same as num_heads for MHA)
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in projections (default: True)
        batch_first: If True, input/output shape is (batch, seq, feature) (default: True)

    Shape:
        - Input: (batch, seq_len, embed_dim) if batch_first=True
                 (seq_len, batch, embed_dim) if batch_first=False
        - Output: Same shape as input
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        # Validate configuration
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        assert num_heads % self.num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"

        self.head_dim = embed_dim // num_heads
        self.num_groups = num_heads // self.num_kv_heads  # How many query heads share each KV head

        # Query projection: full dimension (all query heads)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Key/Value projections: reduced dimension (fewer KV heads)
        kv_embed_dim = self.head_dim * self.num_kv_heads
        self.k_proj = nn.Linear(embed_dim, kv_embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, kv_embed_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Grouped Query Attention.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            key_padding_mask: Mask for padded keys (True = ignore)
            need_weights: Return attention weights
            attn_mask: Additive attention mask
            average_attn_weights: Average attention weights across heads

        Returns:
            - attn_output: Attention output
            - attn_weights: Attention weights (if need_weights=True)
        """

        # Handle batch_first
        if self.batch_first:
            # Input: (batch, seq, embed_dim)
            B, T_q, _ = query.shape
            _, T_k, _ = key.shape
        else:
            # Input: (seq, batch, embed_dim)
            T_q, B, _ = query.shape
            T_k, _, _ = key.shape
            # Convert to batch_first for processing
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Project Q, K, V
        Q = self.q_proj(query)  # (B, T_q, embed_dim)
        K = self.k_proj(key)    # (B, T_k, kv_embed_dim)
        V = self.v_proj(value)  # (B, T_k, kv_embed_dim)

        # Reshape for multi-head attention
        # Q: (B, T_q, num_heads, head_dim)
        Q = Q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T_q, head_dim)

        # K, V: (B, T_k, num_kv_heads, head_dim)
        K = K.view(B, T_k, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, num_kv_heads, T_k, head_dim)
        V = V.view(B, T_k, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, num_kv_heads, T_k, head_dim)

        # Expand K and V to match number of query heads
        # Each KV head is shared by (num_heads // num_kv_heads) query heads
        if self.num_kv_heads != self.num_heads:
            # Repeat each KV head `num_groups` times
            # (B, num_kv_heads, T_k, head_dim) -> (B, num_heads, T_k, head_dim)
            K = K.repeat_interleave(self.num_groups, dim=1)
            V = V.repeat_interleave(self.num_groups, dim=1)

        # Scaled dot-product attention
        # Q: (B, num_heads, T_q, head_dim)
        # K: (B, num_heads, T_k, head_dim)
        # V: (B, num_heads, T_k, head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, num_heads, T_q, T_k)

        # Apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: (B, T_k), True means ignore
            # Reshape to (B, 1, 1, T_k) for broadcasting
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, num_heads, T_q, T_k)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, T_q, head_dim)

        # Reshape back to (B, T_q, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)

        # Output projection
        attn_output = self.out_proj(attn_output)  # (B, T_q, embed_dim)

        # Handle batch_first
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)  # (T_q, B, embed_dim)

        # Prepare attention weights for return
        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)  # (B, T_q, T_k)
            return attn_output, attn_weights
        else:
            return attn_output, None


# Convenience function for drop-in replacement
def MultiheadGQA(
    embed_dim: int,
    num_heads: int,
    num_kv_heads: Optional[int] = None,
    dropout: float = 0.0,
    bias: bool = True,
    batch_first: bool = True,
) -> GroupedQueryAttention:
    """
    Factory function to create a Grouped Query Attention layer.
    Compatible with PyTorch's MultiheadAttention signature.

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads (default: num_heads for standard MHA)
        dropout: Dropout probability
        bias: Use bias in projections
        batch_first: Input/output is (batch, seq, feature)

    Returns:
        GroupedQueryAttention module

    Examples:
        >>> # Standard MHA (8 Q heads, 8 KV heads)
        >>> mha = MultiheadGQA(embed_dim=512, num_heads=8)

        >>> # GQA (8 Q heads, 2 KV heads) - 4x memory reduction
        >>> gqa = MultiheadGQA(embed_dim=512, num_heads=8, num_kv_heads=2)

        >>> # MQA (8 Q heads, 1 KV head) - 8x memory reduction
        >>> mqa = MultiheadGQA(embed_dim=512, num_heads=8, num_kv_heads=1)
    """
    return GroupedQueryAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dropout=dropout,
        bias=bias,
        batch_first=batch_first,
    )


if __name__ == "__main__":
    # Quick test
    print("Testing Grouped Query Attention implementation...\n")

    # Configuration
    batch_size = 4
    seq_len = 12
    embed_dim = 128
    num_heads = 8
    num_kv_heads = 2

    # Create input
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Test different configurations
    configs = [
        ("Standard MHA", num_heads, num_heads),
        ("GQA (4:1 ratio)", num_heads, 2),
        ("MQA (8:1 ratio)", num_heads, 1),
    ]

    for name, n_heads, n_kv_heads in configs:
        print(f"{name}: {n_heads} Q heads, {n_kv_heads} KV heads")
        attn = MultiheadGQA(
            embed_dim=embed_dim,
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Forward pass
        out, weights = attn(x, x, x, need_weights=True)

        # Count parameters
        n_params = sum(p.numel() for p in attn.parameters())

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Attention weights shape: {weights.shape}")
        print(f"  Parameters: {n_params:,}")
        print(f"  Memory reduction vs MHA: {1 - (n_kv_heads / n_heads):.1%}\n")

    print("✓ All tests passed!")



# coding: utf-8
# FT-Transformer single-file Kaggle script for "task/playground-series-s5e11"
# v10: Added Grouped Query Attention (GQA) support for efficient inference
# v9: Categorical NA fix + interest_rate quantized TE + dual representation (raw+TE) for all QD numerics
#     5-fold Stratified CV with fold-averaged test predictions and optional multi-seed ensembling.
#
# Outputs:
# - Logs: task/playground-series-s5e11/outputs/10_7/code_10_7_v9.txt
# - Submission: task/playground-series-s5e11/outputs/10_7/submission_9.csv
#
# Task/metric: Binary classification; evaluation by ROC-AUC (per competition).
# Loss: BCEWithLogitsLoss (stable, proper for probability estimation).
#
# GQA Configuration:
# - USE_GQA=1 (default: 0)       → Enable Grouped Query Attention
# - GQA_KV_HEADS=2 (default: 2)  → Number of KV heads for GQA

import os
import sys
import math
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

# Import GQA implementation
try:
    from grouped_query_attention import MultiheadGQA
    GQA_AVAILABLE = True
except ImportError:
    GQA_AVAILABLE = False
    print("WARNING: grouped_query_attention.py not found. GQA disabled.")

# ----------------------------- GQA Configuration -----------------------------
USE_GQA = int(os.environ.get("USE_GQA", "1"))  # Default: disabled (use standard MHA)
GQA_KV_HEADS = int(os.environ.get("GQA_KV_HEADS", "2"))  # Default: 2 KV heads (4:1 ratio with 8 Q heads)

if USE_GQA and not GQA_AVAILABLE:
    print("ERROR: USE_GQA=1 but grouped_query_attention.py not found. Falling back to MHA.")
    USE_GQA = 0

# ----------------------------- Paths & Logging -----------------------------
BASE_DIR = Path("/kaggle/input/playground-series-s5e11")
OUTPUT_DIR = Path(".")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "code_10_7_v9.txt"
SUB_PATH = OUTPUT_DIR / "submission_9.csv"

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
print(f"Attention mechanism: {'GQA' if USE_GQA else 'MHA'}" + (f" (Q heads: 8, KV heads: {GQA_KV_HEADS}, ratio: {8//GQA_KV_HEADS}:1)" if USE_GQA else ""))

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
    n_blocks=4,
    n_heads=8,
    lr=8e-4,
    weight_decay=1e-5,
    betas=(0.9, 0.98),
    eps=1e-8,
    attn_dropout=0.10,
    ffn_dropout=0.10,
    residual_dropout=0.10,
    d_ffn_factor=2.0,
    max_epochs=25,
    patience=3,
    batch_size=4096,
    grad_clip=1.0,
    warmup_ratio=0.06,
    min_lr=1e-5,
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
    attn_dropout=0.10,
    ffn_dropout=0.10,
    residual_dropout=0.10,
    d_ffn_factor=2.0,
    max_epochs=1,
    patience=1,
    batch_size=512,
    grad_clip=1.0,
    warmup_ratio=0.06,
    min_lr=1e-5,
)

TARGET_COL = "loan_paid_back"
ID_COL = "id"

# Quasi-discrete numerics and canonical categorical (if present)
QD_NUMERICS_CANONICAL = ["debt_to_income_ratio", "credit_score", "interest_rate", "annual_income", "loan_amount"]
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
    train = pd.read_csv(base_dir / "train.csv")
    test = pd.read_csv(base_dir / "test.csv")
    sample = pd.read_csv(base_dir / "sample_submission.csv")
    print(f"Validation: train shape={train.shape}, test shape={test.shape}, sample shape={sample.shape}")
    return train, test, sample

train_df, test_df, sample_df = read_competition_data(BASE_DIR)
assert TARGET_COL in train_df.columns, f"Target column '{TARGET_COL}' not found in train.csv"
assert ID_COL in train_df.columns and ID_COL in test_df.columns, "ID column missing"

y_full = train_df[TARGET_COL].astype(int).values
if not set(np.unique(y_full)).issubset({0, 1}):
    print("Target not strictly {0,1}; binarizing at 0.5.")
    y_full = (train_df[TARGET_COL].astype(float).values >= 0.5).astype(int)

# ----------------------------- Schema Inference (global preview only) -----------------------------
present_qd = [c for c in QD_NUMERICS_CANONICAL if c in train_df.columns]
present_cat = [c for c in CAT_CANONICAL if c in train_df.columns]
dtype_cats = [c for c in train_df.columns if (train_df[c].dtype == "object" or str(train_df[c].dtype).startswith("category")) and c not in [TARGET_COL, ID_COL]]
cat_cols_global = list(dict.fromkeys(present_cat + dtype_cats))
numeric_candidates_global = [c for c in train_df.columns if (np.issubdtype(train_df[c].dtype, np.number)) and c not in [TARGET_COL, ID_COL]]
print(f"Global schema preview: numeric≈{len(numeric_candidates_global)}, cats≈{len(cat_cols_global)}, present_qd={present_qd}")

# ----------------------------- Encoders & Transforms -----------------------------
NA_CAT_TOKEN = "__NA__"
UNK_CAT_TOKEN = "__UNK__"
NA_STR_SET = {"", "nan", "none", "null", "na", "n/a"}

def canonicalize_cat_series(s: pd.Series) -> pd.Series:
    # Robust NA handling: strip, lower, unify to NA token for empties and textual NAs
    ser = s.copy()
    mask_null = ser.isna()
    ser = ser.astype(str).str.strip()
    mask_text_na = ser.str.lower().isin(NA_STR_SET)
    ser = ser.mask(mask_null | mask_text_na, NA_CAT_TOKEN)
    return ser

def build_cat_vocabs(df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Dict[str, int]]:
    vocabs: Dict[str, Dict[str, int]] = {}
    for c in cat_cols:
        vals = canonicalize_cat_series(df[c])
        uniq = pd.unique(vals)
        mapping = {UNK_CAT_TOKEN: 0}
        for i, v in enumerate(sorted(map(str, uniq)), start=1):
            mapping[v] = i
        if NA_CAT_TOKEN not in mapping:
            mapping[NA_CAT_TOKEN] = len(mapping)
        vocabs[c] = mapping
    return vocabs

def apply_cat_vocabs(df: pd.DataFrame, cat_cols: List[str], vocabs: Dict[str, Dict[str, int]]) -> np.ndarray:
    mats = []
    for c in cat_cols:
        mapping = vocabs[c]
        arr = canonicalize_cat_series(df[c]).values
        idx = np.array([mapping.get(v, mapping[UNK_CAT_TOKEN]) for v in arr], dtype=np.int64)
        mats.append(idx.reshape(-1, 1))
    if len(mats) == 0:
        return np.zeros((len(df), 0), dtype=np.int64)
    return np.concatenate(mats, axis=1)

def quantize_interest_rate_series(s: pd.Series, step: float = 0.25) -> pd.Series:
    # Convert to numeric, quantize to 'step' percent, format as string token; NA -> "NaNLevel"
    x = pd.to_numeric(s, errors="coerce")
    q = (x / step).round() * step
    return q.map(lambda v: f"{v:.2f}" if pd.notnull(v) else "NaNLevel")

def m_estimate_mapping(count: int, pos: int, prior: float, m: float = 5.0) -> float:
    return (pos + m * prior) / (count + m)

def _agg_to_full_map(agg_df: pd.DataFrame, count_col: str = "count", pos_col: str = "sum") -> Dict[str, Tuple[int, int]]:
    full_map: Dict[str, Tuple[int, int]] = {}
    if count_col not in agg_df.columns and "cnt" in agg_df.columns:
        count_col = "cnt"
    if pos_col not in agg_df.columns and "pos" in agg_df.columns:
        pos_col = "pos"
    for key, row in agg_df.iterrows():
        cnt = int(float(row[count_col])) if pd.notnull(row[count_col]) else 0
        pos = int(float(row[pos_col])) if pd.notnull(row[pos_col]) else 0
        full_map[str(key)] = (cnt, pos)
    return full_map

def cross_fit_m_estimate_oof(
    df: pd.DataFrame,
    y: np.ndarray,
    col: str,
    n_splits: int = 5,
    m: float = 5.0,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]], float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    prior = float(np.mean(y))
    # full map for inference
    tmp_all = pd.DataFrame({col: df[col].astype(str).fillna("NaNLevel"), "_y": y})
    agg_all = tmp_all.groupby(col)["_y"].agg(["count", "sum"])
    full_map = _agg_to_full_map(agg_all, "count", "sum")
    # OOF
    oof = np.zeros(len(df), dtype=np.float32)
    for tr_idx, va_idx in skf.split(df, y):
        tmp_tr = pd.DataFrame({col: df.iloc[tr_idx][col].astype(str).fillna("NaNLevel").values, "_y": y[tr_idx]})
        agg_tr = tmp_tr.groupby(col)["_y"].agg(["count", "sum"])
        tr_map = _agg_to_full_map(agg_tr, "count", "sum")
        vals_va = df.iloc[va_idx][col].astype(str).fillna("NaNLevel").values
        enc = np.array(
            [m_estimate_mapping(tr_map[v][0], tr_map[v][1], prior, m) if v in tr_map else prior for v in vals_va],
            dtype=np.float32,
        )
        oof[va_idx] = enc
    return oof, full_map, prior

def apply_m_estimate_map(df: pd.DataFrame, col: str, full_map: Dict[str, Tuple[int, int]], prior: float, m: float = 5.0) -> np.ndarray:
    vals = df[col].astype(str).fillna("NaNLevel").values
    out = np.empty(len(vals), dtype=np.float32)
    for i, v in enumerate(vals):
        if v in full_map:
            cnt, pos = full_map[v]
            out[i] = m_estimate_mapping(cnt, pos, prior, m)
        else:
            out[i] = prior
    return out

def standardize_train_valid_test(
    Xtr: pd.DataFrame, Xva: pd.DataFrame, Xte: pd.DataFrame, cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Tuple[float, float]]]:
    stats = {}
    def z(x, mu, sd):
        return (x - mu) / sd if sd > 0 else x - mu
    Xtr_out, Xva_out, Xte_out = [], [], []
    for c in cols:
        mu = float(np.nanmean(Xtr[c].values))
        sd = float(np.nanstd(Xtr[c].values))
        sd = sd if sd > 1e-12 else 1.0
        stats[c] = (mu, sd)
        Xtr_out.append(z(Xtr[c].values, mu, sd).reshape(-1, 1))
        Xva_out.append(z(Xva[c].values, mu, sd).reshape(-1, 1))
        Xte_out.append(z(Xte[c].values, mu, sd).reshape(-1, 1))
    if len(cols) == 0:
        return np.zeros((len(Xtr), 0), np.float32), np.zeros((len(Xva), 0), np.float32), np.zeros((len(Xte), 0), np.float32), stats
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

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, attn_dropout: float, ffn_dropout: float, residual_dropout: float, use_gqa: bool = False, gqa_kv_heads: int = 2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)

        # Use GQA if enabled, otherwise standard MHA
        if use_gqa:
            self.attn = MultiheadGQA(embed_dim=d_model, num_heads=n_heads, num_kv_heads=gqa_kv_heads, dropout=attn_dropout, batch_first=True)
        else:
            self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=attn_dropout, batch_first=True)

        self.drop_res1 = nn.Dropout(residual_dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = ReGLU_FFN(d_model, d_ffn, ffn_dropout)
        self.drop_res2 = nn.Dropout(residual_dropout)
    def forward(self, x):
        x_attn = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
        x = x + self.drop_res1(x_attn)
        x_ffn = self.ffn(self.ln2(x))
        x = x + self.drop_res2(x_ffn)
        return x

class FTTransformer(nn.Module):
    def __init__(self, n_num_features: int, cat_cardinalities: Optional[List[int]], d_token: int, n_blocks: int, n_heads: int, d_ffn: int,
                 attn_dropout: float, ffn_dropout: float, residual_dropout: float, d_out: int = 1, use_gqa: bool = False, gqa_kv_heads: int = 2):
        super().__init__()
        assert d_token % n_heads == 0, f"d_token must be divisible by n_heads; got {d_token} % {n_heads}"
        self.tokenizer = FeatureTokenizer(n_num_features, cat_cardinalities, d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.uniform_(self.cls, -1e-3, 1e-3)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_token, n_heads, d_ffn, attn_dropout, ffn_dropout, residual_dropout, use_gqa, gqa_kv_heads)
            for _ in range(n_blocks)
        ])
        self.head_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, d_out)
        nn.init.kaiming_uniform_(self.head.weight, a=math.sqrt(5))
        nn.init.uniform_(self.head.bias, -1e-3, 1e-3)
    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        x_tokens = self.tokenizer(x_num, x_cat)  # (B, T, d)
        B = x_tokens.shape[0]
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x_tokens], dim=1)
        x = self.blocks(x)
        x = self.head_norm(x[:, 0, :])
        return self.head(x)

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
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    te_df: pd.DataFrame,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    hp: dict,
    fold_idx: int,
    seed: int,
    debug_mode: bool,
    is_full_mode_first_fold: bool,
) -> Tuple[np.ndarray, np.ndarray, float, int, bool]:
    # Per-fold roles
    present_qd = [c for c in QD_NUMERICS_CANONICAL if c in tr_df.columns]
    present_cat = [c for c in CAT_CANONICAL if c in tr_df.columns]
    dtype_cats = [c for c in tr_df.columns if (tr_df[c].dtype == "object" or str(tr_df[c].dtype).startswith("category")) and c not in [TARGET_COL, ID_COL]]
    cat_cols = list(dict.fromkeys(present_cat + dtype_cats))

    numeric_candidates = [c for c in tr_df.columns if (np.issubdtype(tr_df[c].dtype, np.number)) and c not in [TARGET_COL, ID_COL]]
    # Dual representation: keep raw numerics for all QD (do NOT remove them)
    num_cols = list(numeric_candidates)  # includes raw debt_to_income_ratio, credit_score, interest_rate if numeric
    
    # Columns to TE (cross-fitted): the two native QD numerics and the quantized interest rate (if present)
    qd_te_cols: List[str] = []
    for c in present_qd:
        qd_te_cols.append(c)

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
        use_gqa=USE_GQA,
        gqa_kv_heads=GQA_KV_HEADS,
    ).to(DEVICE)

    # Print model parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[fold{fold_idx}|seed{seed}] Model parameters: {n_params:,}")

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
                with torch.cuda.amp.autocast(enabled=AMP):
                    logits = model(xb_num, xb_cat)
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
                logits = model(xb_num, xb_cat)
                loss = loss_fn(logits, yb)
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
                logits = model(xb_num, xb_cat)
            va_preds.append(torch.sigmoid(logits).detach().cpu().numpy().ravel())
    va_probs = np.concatenate(va_preds)

    # Test preds
    te_preds = []
    with torch.no_grad():
        for xb_num, xb_cat in dl_te:
            xb_num = xb_num.to(DEVICE, non_blocking=True)
            xb_cat = xb_cat.to(DEVICE, non_blocking=True) if xb_cat.shape[1] > 0 else None
            with torch.cuda.amp.autocast(enabled=AMP):
                logits = model(xb_num, xb_cat)
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

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_full)):
            tr_df = train_df.iloc[tr_idx].reset_index(drop=True).copy()
            va_df = train_df.iloc[va_idx].reset_index(drop=True).copy()
            y_tr = y_full[tr_idx]; y_va = y_full[va_idx]
            print(f"[seed{seed}] Fold {fold_idx}: train={len(tr_df)}, valid={len(va_df)}")

            # DEBUG sampling
            if debug:
                n_debug = min(1000, len(tr_df))
                if n_debug <= 0.5 * len(tr_df):
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=len(tr_df) - n_debug, random_state=seed)
                    keep_idx, _ = next(sss.split(np.zeros(len(y_tr)), y_tr))
                    tr_df = tr_df.iloc[keep_idx].reset_index(drop=True)
                    y_tr = y_tr[keep_idx]
                    print(f"[seed{seed}] Fold {fold_idx}: DEBUG sampling -> train size={len(tr_df)}")
                else:
                    logging.warning(f"[seed{seed}] Fold {fold_idx}: DEBUG sample would exceed 50% of fold; using full train fold.")

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
        sub = pd.DataFrame({ID_COL: test_df[ID_COL].values, TARGET_COL: np.clip(test_mean, 1e-6, 1 - 1e-6)})
        sub.to_csv(SUB_PATH, index=False)
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
run_one_mode(debug=True)
# Pass 2: FULL
run_one_mode(debug=False)

