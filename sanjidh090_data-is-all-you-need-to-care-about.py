# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# =========================================================
# Kaggle-Ready TabTransformer++ (Regression or Classification)
# - Quantile binning -> token IDs per feature (+ special [CLS])
# - Value tower with gated fusion (per-token MLP on raw value)
# - Transformer encoder blocks with token dropout + stochastic depth
# - AdamW + linear warmup + cosine decay
# - EMA (exponential moving average) weights
# - KFold CV with strict leak-free preprocessing per fold
# - Optional isotonic calibration (regression)
# Author: You (and a friendly assistant)
# =========================================================

import os
import math
import random
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Config (EDIT THESE)
# -----------------------------
DATA_PATH = "/kaggle/input/playground-series-s5e9/train.csv"   # <-- change this
TEST_PATH = '/kaggle/input/playground-series-s5e9/test.csv'                                   # optional
TARGET_COL = "BeatsPerMinute"                                # <-- change this
ID_COL = 'id'                                        # e.g., "id"

TASK_TYPE = "regression"                             # "regression" | "classification"
NUM_CLASSES = 1                                      # if classification, set number of classes

N_FOLDS = 5
RANDOM_STATE = 42
EPOCHS = 30
BATCH_SIZE = 1024
LR = 3e-4
WEIGHT_DECAY = 1e-2
WARMUP_EPOCHS = 2

D_MODEL = 192
N_HEADS = 8
N_LAYERS = 4
MLP_RATIO = 2.0
DROPOUT = 0.1
TOKEN_DROPOUT = 0.05            # randomly drop tokens during training
STOCHASTIC_DEPTH = 0.1           # layer-wise drop path
NBINS = 64                       # quantile bins per numeric feature
USE_EMA = True
EMA_DECAY = 0.999
USE_ISOTONIC = True if TASK_TYPE == "regression" else False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Utils
# -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(RANDOM_STATE)

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * float(self.last_epoch + 1) / float(self.warmup_epochs)
                    for base_lr in self.base_lrs]
        # Cosine decay
        progress = (self.last_epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
        return [base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]

import copy
import torch

class ModelEMA:
    def __init__(self, model, decay=0.999):
        # clone the fully-built model instead of calling type(model)()
        self.ema = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v_ema in self.ema.state_dict().items():
            v = msd[k]
            if v_ema.dtype.is_floating_point:
                v_ema.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                # non-float buffers (e.g., ints) get copied directly
                v_ema.copy_(v)

# -----------------------------
# Data prep: quantile binning (per fold), z-scoring for values
# -----------------------------
def compute_bin_edges(x: np.ndarray, nbins: int) -> np.ndarray:
    qt = QuantileTransformer(n_quantiles=nbins, output_distribution='uniform', random_state=RANDOM_STATE)
    # qt outputs uniform [0,1]; we collect the quantile thresholds to reverse-map to bins
    # We'll approximate bin edges via quantiles
    quantiles = np.linspace(0, 1, nbins + 1)
    # fit the transformer and then use np.quantile on original x for edges
    qt.fit(x.reshape(-1, 1))
    edges = np.quantile(x, quantiles)
    # Ensure unique, handle degeneracy
    edges = np.unique(edges)
    # If duplicates reduce bins
    return edges

def digitize_with_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # returns bin indices in [0, len(edges)-2]
    bins = np.clip(np.digitize(x, edges[1:-1], right=True), 0, len(edges)-2)
    return bins

# -----------------------------
# Dataset
# -----------------------------
class TabDataset(Dataset):
    def __init__(self, X_values: np.ndarray, token_ids: np.ndarray, y=None):
        self.X_values = X_values.astype(np.float32)
        self.token_ids = token_ids.astype(np.int64)
        self.y = None if y is None else (y.astype(np.float32) if TASK_TYPE=="regression" else y.astype(np.int64))

    def __len__(self):
        return self.X_values.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X_values[idx], self.token_ids[idx]
        return self.X_values[idx], self.token_ids[idx], self.y[idx]

# -----------------------------
# Model: TabTransformer++ with value tower + gating
# -----------------------------
class DropPath(nn.Module):
    # Stochastic depth
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=2.0, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TabTransformerPP(nn.Module):
    def __init__(self, n_features, nbins, d_model, n_heads, n_layers, mlp_ratio, dropout,
                 task_type="regression", num_classes=1, token_dropout=0.0):
        super().__init__()
        self.task_type = task_type
        self.n_features = n_features
        self.token_dropout = token_dropout

        # +1 for [CLS] token
        self.total_tokens = n_features + 1

        # Embedding for binned tokens per feature (each feature shares the same nbins space but
        # we add per-feature offsets through a feature embedding)
        self.value_proj = nn.Linear(1, d_model)  # value tower input 1D -> d_model
        self.value_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.bin_embed = nn.Embedding(nbins, d_model)
        self.feature_embed = nn.Embedding(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.pos_embed = nn.Parameter(torch.randn(1, self.total_tokens, d_model) * 0.02)

        drop_rates = np.linspace(0, STOCHASTIC_DEPTH, n_layers)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio=mlp_ratio, dropout=dropout, drop_path=drop_rates[i])
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        head_out = num_classes if task_type == "classification" else 1
        self.head = nn.Linear(d_model, head_out)

    def forward(self, values, token_ids):
        """
        values: (B, F) float32 z-scored raw values
        token_ids: (B, F) int64 binned tokens per feature [0..NBINS-1]
        """
        B, F = values.shape

        # value tower + gating
        v = self.value_proj(values.unsqueeze(-1))         # (B, F, d)
        g = self.value_gate(v)                            # (B, F, d)

        # token embeddings (bin + per-feature embedding)
        # Create feature indices [0..F-1] repeated for batch
        feat_idx = torch.arange(F, device=values.device).unsqueeze(0).expand(B, F)
        t = self.bin_embed(token_ids) + self.feature_embed(feat_idx)  # (B, F, d)

        # gated fusion
        x = t * g + v * (1 - g)                           # (B, F, d)

        # prepend CLS
        cls = self.cls_token.expand(B, -1, -1)            # (B,1,d)
        x = torch.cat([cls, x], dim=1)                    # (B, 1+F, d)

        # token dropout (except CLS)
        if self.training and self.token_dropout > 0.0:
            mask = (torch.rand(B, F, device=x.device) > self.token_dropout).float().unsqueeze(-1)
            x[:, 1:, :] = x[:, 1:, :] * mask + 0.0  # dropped tokens become zeros (pos embed still adds info)

        # add positional embeddings
        x = x + self.pos_embed[:, :x.size(1), :]

        # transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # CLS pooling
        logits = self.head(cls_out)
        return logits

# -----------------------------
# Training / Eval
# -----------------------------
def train_one_epoch(model, ema, loader, optimizer, scaler, epoch, scheduler=None):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        if TASK_TYPE == "regression":
            values, token_ids, y = batch
            y = y.unsqueeze(1)
        else:
            values, token_ids, y = batch

        values = values.to(DEVICE)
        token_ids = token_ids.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            preds = model(values, token_ids)
            if TASK_TYPE == "regression":
                loss = F.smooth_l1_loss(preds, y)
            else:
                loss = F.cross_entropy(preds, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * values.size(0)
        n += values.size(0)

    if scheduler is not None:
        scheduler.step()
    return total_loss / max(1, n)

@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds = []
    gts = []
    for batch in loader:
        if TASK_TYPE == "regression":
            values, token_ids, y = batch
            y = y.numpy() if isinstance(y, np.ndarray) else y.cpu().numpy()
        else:
            values, token_ids, y = batch
            y = y.cpu().numpy()
        values = values.to(DEVICE)
        token_ids = token_ids.to(DEVICE)
        out = model(values, token_ids)
        out = out.detach().cpu().numpy()
        if TASK_TYPE == "classification":
            preds.append(F.softmax(torch.tensor(out), dim=1).numpy())
        else:
            preds.append(out.reshape(-1))
        gts.append(y)
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    return preds, gts

# -----------------------------
# Main CV
# -----------------------------
def run_cv(df: pd.DataFrame, features: List[str]):
    X = df[features].copy()
    y = df[TARGET_COL].values

    if TASK_TYPE == "classification":
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splitter = skf.split(X, y)
    else:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splitter = kf.split(X, y)

    oof = np.zeros(len(df), dtype=float if TASK_TYPE=="regression" else np.float32)
    oof_logits = None
    if TASK_TYPE == "classification":
        oof_logits = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)

    fold_scores = []
    for fold, (trn_idx, val_idx) in enumerate(splitter):
        print(f"\n===== Fold {fold+1}/{N_FOLDS} =====")
        X_trn, X_val = X.iloc[trn_idx].copy(), X.iloc[val_idx].copy()
        y_trn, y_val = y[trn_idx], y[val_idx]

        # Per-fold scalers and bin edges (leak-free)
        scaler = StandardScaler()
        scaler.fit(X_trn.values)
        Z_trn = scaler.transform(X_trn.values)
        Z_val = scaler.transform(X_val.values)

        # Per-feature quantile edges + tokenization
        edges_list = []
        for j in range(Z_trn.shape[1]):
            e = compute_bin_edges(Z_trn[:, j], NBINS)
            # ensure at least 2 edges
            if len(e) < 2:
                e = np.array([-np.inf, np.inf], dtype=float)
            edges_list.append(e)

        def tokenize(Z):
            T = np.zeros_like(Z, dtype=np.int64)
            for j, e in enumerate(edges_list):
                T[:, j] = digitize_with_edges(Z[:, j], e)
                # clip if edges collapsed
                T[:, j] = np.clip(T[:, j], 0, max(0, len(e)-2))
            return T

        tok_trn = tokenize(Z_trn)
        tok_val = tokenize(Z_val)

        # Datasets/Loaders
        trn_ds = TabDataset(Z_trn, tok_trn, y_trn)
        val_ds = TabDataset(Z_val, tok_val, y_val)
        trn_dl = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        # Model
        model = TabTransformerPP(
            n_features=Z_trn.shape[1],
            nbins=NBINS,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            mlp_ratio=MLP_RATIO,
            dropout=DROPOUT,
            task_type=TASK_TYPE,
            num_classes=NUM_CLASSES,
            token_dropout=TOKEN_DROPOUT
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scheduler = WarmupCosineLR(optimizer, warmup_epochs=WARMUP_EPOCHS, max_epochs=EPOCHS)
        ema = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None

        best_score = float("inf") if TASK_TYPE=="regression" else -float("inf")
        best_state = None

        for epoch in range(EPOCHS):
            t0 = time.time()
            train_loss = train_one_epoch(model, ema, trn_dl, optimizer, scaler, epoch, scheduler)
            # Eval (EMA if available)
            eval_model = ema.ema if (ema is not None) else model
            preds, gts = predict(eval_model, val_dl)
            if TASK_TYPE == "regression":
                score = rmse(gts, preds)
                improved = score < best_score
                metric_name = "RMSE"
            else:
                # classification
                score = accuracy_score(gts, preds.argmax(axis=1))
                improved = score > best_score
                metric_name = "Acc"

            if improved:
                best_score = score
                best_state = eval_model.state_dict()

            dt = time.time() - t0
            print(f"Epoch {epoch+1:02d}/{EPOCHS} - train_loss={train_loss:.4f}  val_{metric_name}={score:.5f}  ({dt:.1f}s)")

        # Load best
        model.load_state_dict(best_state)

        # Final fold preds
        preds, gts = predict(model, val_dl)
        if TASK_TYPE == "classification":
            oof_logits[val_idx] = preds
            pred_labels = preds.argmax(axis=1)
            fold_metric = accuracy_score(gts, pred_labels)
            oof[val_idx] = pred_labels
        else:
            # Optional isotonic calibration in validation space
            if USE_ISOTONIC:
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(preds, gts)
                preds = ir.predict(preds)
            oof[val_idx] = preds
            fold_metric = rmse(gts, preds)

        fold_scores.append(fold_metric)
        print(f"Fold {fold+1} {('Acc' if TASK_TYPE=='classification' else 'RMSE')} = {fold_metric:.5f}")

    # Overall
    if TASK_TYPE == "classification":
        overall = accuracy_score(y, oof.astype(int))
        print(f"\nOOF Accuracy = {overall:.5f}")
    else:
        overall = rmse(y, oof)
        print(f"\nOOF RMSE = {overall:.5f}")

    print("Per-fold:", np.array(fold_scores))
    return oof

# -----------------------------
# Entry
# -----------------------------
def main():
    if not os.path.exists(DATA_PATH):
        # Fallback: synth dataset to verify script runs
        print("DATA_PATH not found. Generating synthetic regression dataset for demo.")
        n = 50000
        f = 20
        rng = np.random.RandomState(RANDOM_STATE)
        X = rng.randn(n, f)
        coefs = rng.randn(f)
        y = X @ coefs + rng.randn(n) * 0.5
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(f)])
        df[TARGET_COL] = y
        features = [c for c in df.columns if c != TARGET_COL]
    else:
        df = pd.read_csv(DATA_PATH)
        if ID_COL and ID_COL in df.columns:
            df = df.drop(columns=[ID_COL])
        # Infer numeric features
        features = [c for c in df.columns if c != TARGET_COL and np.issubdtype(df[c].dtype, np.number)]
        if len(features) == 0:
            # try to coerce
            for c in df.columns:
                if c == TARGET_COL: 
                    continue
                df[c] = pd.to_numeric(df[c], errors="coerce")
            features = [c for c in df.columns if c != TARGET_COL and np.issubdtype(df[c].dtype, np.number)]
        # Drop rows with missing target
        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        # Fill missing numeric with median (leak-free imputation would be per-fold; we keep it simple here)
        df[features] = df[features].fillna(df[features].median())

    print(f"Using {len(features)} numeric features.")
    _ = run_cv(df, features)

if __name__ == "__main__":
    main()



# -----------------------------
# Full train + predict on test
# -----------------------------
def train_full_and_predict(df_train, df_test, features):
    X_train = df_train[features].values
    y_train = df_train[TARGET_COL].values
    X_test = df_test[features].values

    # Fit scaler + bin edges on full train
    scaler = StandardScaler()
    scaler.fit(X_train)
    Z_train = scaler.transform(X_train)
    Z_test = scaler.transform(X_test)

    edges_list = []
    for j in range(Z_train.shape[1]):
        e = compute_bin_edges(Z_train[:, j], NBINS)
        if len(e) < 2:
            e = np.array([-np.inf, np.inf], dtype=float)
        edges_list.append(e)

    def tokenize(Z):
        T = np.zeros_like(Z, dtype=np.int64)
        for j, e in enumerate(edges_list):
            T[:, j] = digitize_with_edges(Z[:, j], e)
        return T

    tok_train = tokenize(Z_train)
    tok_test = tokenize(Z_test)

    trn_ds = TabDataset(Z_train, tok_train, y_train)
    test_ds = TabDataset(Z_test, tok_test)
    trn_dl = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = TabTransformerPP(
        n_features=Z_train.shape[1],
        nbins=NBINS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        task_type=TASK_TYPE,
        num_classes=NUM_CLASSES,
        token_dropout=TOKEN_DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=True)
    scheduler = WarmupCosineLR(optimizer, warmup_epochs=WARMUP_EPOCHS, max_epochs=EPOCHS)
    ema = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None

    for epoch in range(EPOCHS):
        train_one_epoch(model, ema, trn_dl, optimizer, scaler_amp, epoch, scheduler)
    eval_model = ema.ema if ema else model

    preds, _ = predict(eval_model, test_dl)

    # For classification, take argmax
    if TASK_TYPE == "classification":
        preds = preds.argmax(axis=1)

    return preds


def main():
    # ... (same as before, but now after run_cv:)
    if not os.path.exists(DATA_PATH):
        print("Demo mode, no submission.")
        return

    df = pd.read_csv(DATA_PATH)
    if ID_COL and ID_COL in df.columns:
        ids = df[ID_COL].values
        df = df.drop(columns=[ID_COL])
    else:
        ids = np.arange(len(df))

    features = [c for c in df.columns if c != TARGET_COL and np.issubdtype(df[c].dtype, np.number)]
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df[features] = df[features].fillna(df[features].median())

    _ = run_cv(df, features)

    if TEST_PATH:
        df_test = pd.read_csv(TEST_PATH)
        test_ids = df_test[ID_COL].values if ID_COL else np.arange(len(df_test))
        df_test[features] = df_test[features].fillna(df[features].median())

        preds = train_full_and_predict(df, df_test, features)
        sub = pd.DataFrame({"id": test_ids, "target": preds})
        sub.to_csv("submission_tab.csv", index=False)
        print("Saved submission_tab.csv")






# ================================
# Train-only CV + Fold Distributions
# ================================
import numpy as np, pandas as pd, warnings, matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
warnings.filterwarnings("ignore")

# --- 1) Load
DATA_DIR   = "/kaggle/input/playground-series-s5e9"
ID_COL     = "id"
TARGET_COL = "BeatsPerMinute"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")  # not used for scoring, here only to keep parity if needed

# --- 2) Minimal, stable feature set (all are in dataset)
def engineer_features(df):
    df = df.copy()
    df["mood_energy_interaction"]   = df["MoodScore"] * df["Energy"]
    df["loudness_vocal_interaction"]= df["AudioLoudness"] * df["VocalContent"]
    df["acoustic_instrumental_ratio"]= df["AcousticQuality"] / (df["InstrumentalScore"] + 1e-6)
    return df

train_fe = engineer_features(train)
test_fe  = engineer_features(test)

FEATURES = [
    'RhythmScore','AudioLoudness','VocalContent','AcousticQuality',
    'InstrumentalScore','LivePerformanceLikelihood','MoodScore',
    'TrackDurationMs','Energy',
    'mood_energy_interaction','loudness_vocal_interaction','acoustic_instrumental_ratio'
]

X = train_fe[FEATURES].values
y = train[TARGET_COL].values

# --- 3) CV setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# --- 4) Models
lgb_params = dict(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

USE_RIDGE = True     # set False to use pure LGBM predictions

oof_pred_lgb = np.zeros(len(train))
oof_pred_blend = np.zeros(len(train))
fold_scores = []

# store fold-wise y and preds for distribution plots
fold_true, fold_pred = [], []

print("Training with KFold =", n_splits)
for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_tr, X_va = X[trn_idx], X[val_idx]
    y_tr, y_va = y[trn_idx], y[val_idx]

    # LightGBM
    lgbm = lgb.LGBMRegressor(**lgb_params)
    lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    p_lgb = lgbm.predict(X_va)

    # Optional tiny ridge on top of LGB (helps variance sometimes)
    if USE_RIDGE:
        ridge = Ridge(alpha=1e-3, fit_intercept=True, random_state=42)
        ridge.fit(p_lgb.reshape(-1,1), y_va)
        p_blend = ridge.predict(p_lgb.reshape(-1,1))
    else:
        p_blend = p_lgb

    # store
    oof_pred_lgb[val_idx]   = p_lgb
    oof_pred_blend[val_idx] = p_blend
    fold_true.append(y_va)
    fold_pred.append(p_blend)  # plot the final per-fold prediction

    rmse = mean_squared_error(y_va, p_blend, squared=False)
    fold_scores.append(rmse)
    print(f"Fold {fold}: RMSE = {rmse:.5f}")

# overall OOF score
oof_rmse = mean_squared_error(y, oof_pred_blend, squared=False)
print(f"\nOOF RMSE (blend) = {oof_rmse:.5f}")
print("Per-fold RMSEs:", [f"{s:.5f}" for s in fold_scores])

# --- 5) Visualization: per-fold distributions of Actual vs Predicted
# We'll draw 2 rows: top row = KDE hist of Actual vs Pred; bottom row = boxplots per fold
cols = n_splits
plt.figure(figsize=(4*cols, 8))

# (A) KDE-style histograms per fold
for i in range(n_splits):
    ax = plt.subplot(2, cols, i+1)
    ax.hist(fold_true[i], bins=40, alpha=0.5, label="Actual", density=True)
    ax.hist(fold_pred[i], bins=40, alpha=0.5, label="Predicted", density=True)
    ax.set_title(f"Fold {i+1} â€” Dist.")
    ax.set_xlabel("BeatsPerMinute")
    ax.set_ylabel("Density")
    if i == 0:
        ax.legend(frameon=False)

# (B) Boxplots comparing Actual vs Predicted per fold
# Prepare data aligned for boxplot
box_actual = [np.asarray(t) for t in fold_true]
box_pred   = [np.asarray(p) for p in fold_pred]

for i in range(n_splits):
    ax = plt.subplot(2, cols, cols + i + 1)
    bp = ax.boxplot([box_actual[i], box_pred[i]],
                    labels=["Actual","Pred"], showfliers=False)
    ax.set_title(f"Fold {i+1} â€” Boxplot")
    ax.set_ylabel("BeatsPerMinute")

plt.tight_layout()
plt.show()

# --- 6) Overall scatter & residuals (like your previous figure, but using train-only OOF)
pred_all   = oof_pred_blend
residuals  = y - pred_all

plt.figure(figsize=(14,5))

# Scatter: Actual vs Predicted
ax1 = plt.subplot(1,2,1)
ax1.scatter(y, pred_all, s=6, alpha=0.5)
lims = [min(y.min(), pred_all.min())-5, max(y.max(), pred_all.max())+5]
ax1.plot(lims, lims, 'k--', lw=1)
ax1.set_title("Actual vs. Predicted (OOF)")
ax1.set_xlabel("Actual")
ax1.set_ylabel("Predicted")
ax1.set_xlim(lims); ax1.set_ylim(lims)

# Residual plot
ax2 = plt.subplot(1,2,2)
ax2.scatter(pred_all, residuals, s=6, alpha=0.5)
ax2.axhline(0, color='k', ls='--', lw=1)
ax2.set_title("Residuals vs. Predicted (OOF)")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Residuals")

plt.tight_layout()
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

corrs = train_fe[FEATURES + [TARGET_COL]].corr()[TARGET_COL].sort_values()
plt.figure(figsize=(6,8))
sns.barplot(y=corrs.index, x=corrs.values, palette="coolwarm")
plt.title("Correlation of Features with BeatsPerMinute")
plt.show()



# ================================
# Residual Boosting (train-only CV)
# Stage-1: LGBM  â†’ yÌ‚1
# Stage-2: XGBoost on residuals (y - yÌ‚1) â†’ rÌ‚
# Final: yÌ‚ = yÌ‚1 + rÌ‚
# ================================
import numpy as np, pandas as pd, warnings, matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
warnings.filterwarnings("ignore")

# ---- 1) Load
DATA_DIR   = "/kaggle/input/playground-series-s5e9"
ID_COL     = "id"
TARGET_COL = "BeatsPerMinute"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")  # not scored; kept for parity if you want to train on full later

# ---- 2) Features
def engineer_features(df):
    df = df.copy()
    df["mood_energy_interaction"]     = df["MoodScore"] * df["Energy"]
    df["loudness_vocal_interaction"]  = df["AudioLoudness"] * df["VocalContent"]
    df["acoustic_instrumental_ratio"] = df["AcousticQuality"] / (df["InstrumentalScore"] + 1e-6)
    # a few extra interactions to increase variance learning
    df["energy_over_vocal"]  = df["Energy"] / (df["VocalContent"] + 1e-6)
    df["loud_over_duration"] = df["AudioLoudness"] / (df["TrackDurationMs"] + 1e-6)
    df["mood_times_rhythm"]  = df["MoodScore"] * df["RhythmScore"]
    return df

train_fe = engineer_features(train)

FEATURES = [
    'RhythmScore','AudioLoudness','VocalContent','AcousticQuality',
    'InstrumentalScore','LivePerformanceLikelihood','MoodScore',
    'TrackDurationMs','Energy',
    'mood_energy_interaction','loudness_vocal_interaction','acoustic_instrumental_ratio',
    'energy_over_vocal','loud_over_duration','mood_times_rhythm'
]

X = train_fe[FEATURES].values
y = train[TARGET_COL].values

# ---- 3) CV
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# ---- 4) Models (give them enough capacity to avoid mean-collapse)
lgb_params = dict(
    n_estimators=4000,
    learning_rate=0.02,
    num_leaves=256,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    min_data_in_leaf=20,
    reg_alpha=0.0,
    reg_lambda=0.2,
    random_state=42,
    n_jobs=-1
)

xgb_params = dict(
    n_estimators=2500,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.5,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

# ---- 5) OOF containers
oof_stage1 = np.zeros(len(train))
oof_stage2 = np.zeros(len(train))
oof_final  = np.zeros(len(train))
fold_scores_s1, fold_scores_s2, fold_scores_final = [], [], []

print("Training residual boosting with KFold =", n_splits)
for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_tr, X_va = X[trn_idx], X[val_idx]
    y_tr, y_va = y[trn_idx], y[val_idx]

    # ---- Stage 1: base model on y
    m1 = lgb.LGBMRegressor(**lgb_params)
    m1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(200, verbose=False)])
    p1 = m1.predict(X_va)

    # ---- Stage 2: residual model on (y - p1)
    res_tr = y_tr - m1.predict(X_tr)       # residuals on training chunk
    m2 = xgb.XGBRegressor(**xgb_params)
    m2.fit(X_tr, res_tr, eval_set=[(X_va, y_va - p1)], verbose=False)  # monitor residual validation

    p2 = m2.predict(X_va)

    # ---- Combine
    p_final = p1 + p2

    # ---- Store OOF
    oof_stage1[val_idx] = p1
    oof_stage2[val_idx] = p2
    oof_final [val_idx] = p_final

    # ---- Scores
    rmse_s1    = mean_squared_error(y_va, p1, squared=False)
    rmse_s2    = mean_squared_error(y_va - p1, p2, squared=False)
    rmse_final = mean_squared_error(y_va, p_final, squared=False)
    fold_scores_s1.append(rmse_s1)
    fold_scores_s2.append(rmse_s2)
    fold_scores_final.append(rmse_final)

    print(f"Fold {fold}:  Stage1 RMSE={rmse_s1:.5f} | Residual RMSE={rmse_s2:.5f} | Final RMSE={rmse_final:.5f}")

# ---- 6) Overall OOF scores
rmse_s1_all    = mean_squared_error(y, oof_stage1, squared=False)
rmse_s2_all    = mean_squared_error(y - oof_stage1, oof_stage2, squared=False)
rmse_final_all = mean_squared_error(y, oof_final, squared=False)

print("\n================ Residual Boosting Summary (OOF) ================")
print("Per-fold Stage1 :", [f"{s:.5f}" for s in fold_scores_s1])
print("Per-fold Resid  :", [f"{s:.5f}" for s in fold_scores_s2])
print("Per-fold Final  :", [f"{s:.5f}" for s in fold_scores_final])
print(f"OOF Stage1 RMSE    = {rmse_s1_all:.5f}")
print(f"OOF Residual RMSE  = {rmse_s2_all:.5f}")
print(f"OOF Final  RMSE    = {rmse_final_all:.5f}")

# ---- 7) Diagnostics/plots (OOF only)
pred_all  = oof_final
residuals = y - pred_all

plt.figure(figsize=(14,5))
ax1 = plt.subplot(1,2,1)
ax1.scatter(y, pred_all, s=6, alpha=0.5)
lims = [min(y.min(), pred_all.min())-5, max(y.max(), pred_all.max())+5]
ax1.plot(lims, lims, 'k--', lw=1)
ax1.set_title("Actual vs. Predicted (OOF) â€” Residual Boosting")
ax1.set_xlabel("Actual")
ax1.set_ylabel("Predicted")
ax1.set_xlim(lims); ax1.set_ylim(lims)

ax2 = plt.subplot(1,2,2)
ax2.scatter(pred_all, residuals, s=6, alpha=0.5)
ax2.axhline(0, color='k', ls='--', lw=1)
ax2.set_title("Residuals vs. Predicted (OOF)")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Residuals")
plt.tight_layout()
plt.show()

# ---- 8) Fold-wise distribution comparison
plt.figure(figsize=(4*n_splits, 8))
for i, (trn_idx, val_idx) in enumerate(kf.split(X, y), 1):
    y_va = y[val_idx]
    p_va = oof_final[val_idx]
    ax = plt.subplot(2, n_splits, i)
    ax.hist(y_va, bins=40, alpha=0.5, density=True, label="Actual")
    ax.hist(p_va, bins=40, alpha=0.5, density=True, label="Pred")
    ax.set_title(f"Fold {i} â€” Dist.")
    if i == 1: ax.legend(frameon=False)
    ax2 = plt.subplot(2, n_splits, n_splits + i)
    ax2.boxplot([y_va, p_va], labels=["Actual","Pred"], showfliers=False)
    ax2.set_title(f"Fold {i} â€” Boxplot")
plt.tight_layout(); plt.show()



# ==========================================
# Train-only CV with LOG-TRANSFORMED target
# (learn on log1p(BPM), invert with expm1)
# ==========================================
import numpy as np, pandas as pd, warnings, matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
warnings.filterwarnings("ignore")

# ---- 1) Load
DATA_DIR   = "/kaggle/input/playground-series-s5e9"
ID_COL     = "id"
TARGET_COL = "BeatsPerMinute"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")  # not used for scoring

# ---- 2) Features
def engineer_features(df):
    df = df.copy()
    df["mood_energy_interaction"]     = df["MoodScore"] * df["Energy"]
    df["loudness_vocal_interaction"]  = df["AudioLoudness"] * df["VocalContent"]
    df["acoustic_instrumental_ratio"] = df["AcousticQuality"] / (df["InstrumentalScore"] + 1e-6)
    # a few extra interactions
    df["energy_over_vocal"]  = df["Energy"] / (df["VocalContent"] + 1e-6)
    df["loud_over_duration"] = df["AudioLoudness"] / (df["TrackDurationMs"] + 1e-6)
    df["mood_times_rhythm"]  = df["MoodScore"] * df["RhythmScore"]
    return df

train_fe = engineer_features(train)

FEATURES = [
    'RhythmScore','AudioLoudness','VocalContent','AcousticQuality',
    'InstrumentalScore','LivePerformanceLikelihood','MoodScore',
    'TrackDurationMs','Energy',
    'mood_energy_interaction','loudness_vocal_interaction','acoustic_instrumental_ratio',
    'energy_over_vocal','loud_over_duration','mood_times_rhythm'
]

X = train_fe[FEATURES].values
y = train[TARGET_COL].values

# ---- 3) Transform target (log1p)
y_log = np.log1p(y)

# ---- 4) CV
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# ---- 5) Model (a bit more capacity)
lgb_params = dict(
    n_estimators=4000,
    learning_rate=0.02,
    num_leaves=256,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    min_data_in_leaf=20,
    reg_alpha=0.0,
    reg_lambda=0.2,
    random_state=42,
    n_jobs=-1
)

# ---- 6) OOF containers (in original BPM space)
oof_pred_log = np.zeros(len(train))  # predictions on log-scale
oof_pred     = np.zeros(len(train))  # expm1-inverted predictions
fold_scores  = []

fold_true, fold_pred = [], []

print("Training with log-transformed targetâ€¦")
for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y_log), 1):
    X_tr, X_va = X[trn_idx], X[val_idx]
    y_tr_log, y_va_log = y_log[trn_idx], y_log[val_idx]
    y_va = y[val_idx]  # for RMSE in original space

    mdl = lgb.LGBMRegressor(**lgb_params)
    mdl.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], callbacks=[lgb.early_stopping(200, verbose=False)])

    p_va_log = mdl.predict(X_va)
    p_va     = np.expm1(p_va_log)  # back to BPM

    oof_pred_log[val_idx] = p_va_log
    oof_pred    [val_idx] = p_va
    fold_true.append(y_va)
    fold_pred.append(p_va)

    rmse = mean_squared_error(y_va, p_va, squared=False)
    fold_scores.append(rmse)
    print(f"Fold {fold}: RMSE (orig space) = {rmse:.5f}")

oof_rmse = mean_squared_error(y, oof_pred, squared=False)
print(f"\nOOF RMSE (orig space) = {oof_rmse:.5f}")
print("Per-fold RMSEs:", [f"{s:.5f}" for s in fold_scores])

# ---- 7) Diagnostics: overall scatter & residuals (original space)
pred_all  = oof_pred
residuals = y - pred_all

plt.figure(figsize=(14,5))
ax1 = plt.subplot(1,2,1)
ax1.scatter(y, pred_all, s=6, alpha=0.5)
lims = [min(y.min(), pred_all.min())-5, max(y.max(), pred_all.max())+5]
ax1.plot(lims, lims, 'k--', lw=1)
ax1.set_title("Actual vs. Predicted (OOF) â€” Log-target")
ax1.set_xlabel("Actual BPM")
ax1.set_ylabel("Predicted BPM")
ax1.set_xlim(lims); ax1.set_ylim(lims)

ax2 = plt.subplot(1,2,2)
ax2.scatter(pred_all, residuals, s=6, alpha=0.5)
ax2.axhline(0, color='k', ls='--', lw=1)
ax2.set_title("Residuals vs. Predicted (OOF)")
ax2.set_xlabel("Predicted BPM")
ax2.set_ylabel("Residuals")
plt.tight_layout(); plt.show()

# ---- 8) Fold-wise distribution comparison (original space)
plt.figure(figsize=(4*n_splits, 8))
for i, (trn_idx, val_idx) in enumerate(kf.split(X, y_log), 1):
    y_va = y[val_idx]
    p_va = oof_pred[val_idx]
    ax = plt.subplot(2, n_splits, i)
    ax.hist(y_va, bins=40, alpha=0.5, density=True, label="Actual")
    ax.hist(p_va, bins=40, alpha=0.5, density=True, label="Pred")
    ax.set_title(f"Fold {i} â€” Dist.")
    if i == 1: ax.legend(frameon=False)
    ax2 = plt.subplot(2, n_splits, n_splits + i)
    ax2.boxplot([y_va, p_va], labels=["Actual","Pred"], showfliers=False)
    ax2.set_title(f"Fold {i} â€” Boxplot")
plt.tight_layout(); plt.show()



# import tensorflow as tf

# # Ensure that TensorFlow is using the GPU
# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     print(f"Using GPU: {physical_devices[0]}")
#     tf.config.set_visible_devices(physical_devices[0], 'GPU')
# else:
#     print("No GPU available, using CPU.")



# # --- 0. IMPORTS ---
# !pip install category_encoders
# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# import xgboost as xgb
# import tensorflow as tf
# import itertools
# import warnings
# import category_encoders as ce

# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import HistGradientBoostingRegressor
# from sklearn.isotonic import IsotonicRegression

# # Keras specific imports for Neural Network
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# # Configure warnings
# warnings.filterwarnings('ignore')

# # --- Constants ---
# ID_COL = 'id'
# TARGET_COL = 'BeatsPerMinute'
# N_SPLITS = 10        # Number of folds for cross-validation
# RANDOM_STATE = 42
# TOP_N_FEATURES = 250 # Number of features to select after engineering

# # --- 1. DATA LOADING ---
# print("ðŸŽµ Loading Data...")
# try:
#     train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
#     test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
#     # Placeholder for external data loading from Notebook 3 logic
#     # external_df = pd.read_csv("/kaggle/input/song-data/song_data.csv") 
#     # train_df = pd.concat([train_df, external_df], ignore_index=True) # Combine external data
# except FileNotFoundError:
#     print("Warning: Data files not found. Proceeding with placeholder logic if possible.")
#     # In a real run, you would handle file paths appropriately.

# # --- 2. FEATURE ENGINEERING (Combined Strategy) ---
# print("ðŸ§® Engineering features by combining all strategies...")

# def feature_engineer(train_data, test_data):
#     """Applies combined feature engineering from all notebooks."""
#     combined_df = pd.concat([train_data.drop(TARGET_COL, axis=1), test_data], ignore_index=True)

#     # --- Strategy A: Manual Interactions (from Script 2) ---
#     combined_df['mood_energy_interaction'] = combined_df['MoodScore'] * combined_df['Energy']
#     combined_df['rhythm_energy_interaction'] = combined_df['RhythmScore'] * combined_df['Energy']
#     combined_df['acoustic_instrumental_ratio'] = combined_df['AcousticQuality'] / (combined_df['InstrumentalScore'] + 1e-6)
    
#     # --- Strategy B: Statistical Aggregates (from Script 3) ---
#     # First, create bins to group by
#     combined_df['EnergyBin'] = pd.cut(combined_df['Energy'], bins=10, labels=False)
#     combined_df['MoodBin'] = pd.cut(combined_df['MoodScore'], bins=10, labels=False)
    
#     aggregation_features = ['RhythmScore', 'VocalContent', 'AcousticQuality']
#     group_by_col = 'EnergyBin'
#     for feature in aggregation_features:
#         stats = combined_df.groupby(group_by_col)[feature].agg(['mean', 'std'])
#         stats.columns = [f'{feature}_by_{group_by_col}_mean', f'{feature}_by_{group_by_col}_std']
#         combined_df = combined_df.merge(stats, on=group_by_col, how='left')

#     # --- Strategy C: Combinatorial Features (from Script 1) ---
#     combinatorial_features_to_create = []
#     base_combinatorial_cols = ['EnergyBin', 'MoodBin', 'VocalContent', 'AcousticQuality']
    
#     for combo in itertools.combinations(base_combinatorial_cols, 2): # Size 2 combinations
#         new_feature_name = '||'.join(combo)
#         combinatorial_features_to_create.append(new_feature_name)
#         combined_df[new_feature_name] = combined_df[list(combo)].astype(str).agg('_'.join, axis=1)

#     return combined_df, combinatorial_features_to_create

# # Apply feature engineering
# df_processed, combinatorial_features = feature_engineer(train_df, test_df)

# # --- 3. PREPROCESSING & FEATURE SELECTION ---
# print("ðŸ”’ Encoding features and selecting top performers...")

# # Separate train and test again
# X = df_processed.iloc[:len(train_df)]
# X_test = df_processed.iloc[len(train_df):]
# y = train_df[TARGET_COL]

# # Target Encoding for combinatorial features
# encoder = ce.TargetEncoder(cols=combinatorial_features)
# X[combinatorial_features] = encoder.fit_transform(X[combinatorial_features], y)
# X_test[combinatorial_features] = encoder.transform(X_test[combinatorial_features])

# # Identify all features for selection process
# original_features = [col for col in test_df.columns if col not in [ID_COL]]
# engineered_features = [col for col in df_processed.columns if col not in train_df.columns and col not in combinatorial_features]
# all_features = original_features + engineered_features + combinatorial_features

# # Remove duplicates from feature list while preserving order
# all_features = list(dict.fromkeys(all_features)) 

# # Clean data before selection
# X[all_features] = X[all_features].fillna(X[all_features].mean())
# X_test[all_features] = X_test[all_features].fillna(X[all_features].mean())

# # Feature Selection
# print(f"Selecting top {TOP_N_FEATURES} features from {len(all_features)} total candidates...")
# selector_model = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1)
# selector_model.fit(X[all_features], y)
# importances = pd.Series(selector_model.feature_importances_, index=all_features)
# selected_features = importances.sort_values(ascending=False).index[:TOP_N_FEATURES].tolist()

# # Final data for modeling
# X_final = X[selected_features]
# X_test_final = X_test[selected_features]

# print(f"Data ready for modeling with {len(selected_features)} features.")

# # --- 4. LEVEL 1 MODELING: NEURAL NETWORK DEFINITION ---
# def build_nn_model(input_shape):
#     inputs = Input(shape=(input_shape,))
#     x = BatchNormalization()(inputs)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.3)(x)
#     x = BatchNormalization()(x)
#     x = Dense(64, activation='relu')(x)
#     x = Dropout(0.2)(x)
#     outputs = Dense(1, activation='linear')(x)
    
#     model = Model(inputs=inputs, outputs=outputs)
#     # Using Huber loss for robustness to outliers (from Notebook 3)
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='huber')
#     return model

# # --- 5. LEVEL 1 MODELING: CROSS-VALIDATION ENSEMBLE ---
# print("ðŸ¤– Training Level 1 models (LGBM, XGB, HGB, Ridge, NN)...")

# kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# # OOF and test predictions for 5 models
# oof_preds = np.zeros((len(X_final), 5)) 
# test_preds = np.zeros((len(X_test_final), 5))

# for fold, (train_idx, val_idx) in enumerate(kf.split(X_final, y)):
#     print(f"\n===== Fold {fold+1}/{N_SPLITS} =====")
#     X_train, X_val = X_final.iloc[train_idx], X_final.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     # --- Scaling (for Ridge and NN) ---
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
#     X_test_scaled = scaler.transform(X_test_final)

#     # --- Model 1: LightGBM ---
#     print("Training LGBM...")
#     model_lgbm = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, n_estimators=1000)
#     model_lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
#     oof_preds[val_idx, 0] = model_lgbm.predict(X_val)
#     test_preds[:, 0] += model_lgbm.predict(X_test_final) / N_SPLITS

#     # --- Model 2: XGBoost ---
#     print("Training XGBoost...")
#     model_xgb = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, n_estimators=1000, tree_method='hist')
#     model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
#     oof_preds[val_idx, 1] = model_xgb.predict(X_val)
#     test_preds[:, 1] += model_xgb.predict(X_test_final) / N_SPLITS

#     # --- Model 3: Ridge Regression ---
#     print("Training Ridge...")
#     model_ridge = Ridge(random_state=RANDOM_STATE, alpha=20)
#     model_ridge.fit(X_train_scaled, y_train)
#     oof_preds[val_idx, 2] = model_ridge.predict(X_val_scaled)
#     test_preds[:, 2] += model_ridge.predict(X_test_scaled) / N_SPLITS

#     # --- Model 4: HistGradientBoostingRegressor ---
#     print("Training HGBoost...")
#     model_hgb = HistGradientBoostingRegressor(random_state=RANDOM_STATE, max_iter=500)
#     model_hgb.fit(X_train, y_train)
#     oof_preds[val_idx, 3] = model_hgb.predict(X_val)
#     test_preds[:, 3] += model_hgb.predict(X_test_final) / N_SPLITS

#     # --- Model 5: Neural Network ---
#     print("Training Neural Network...")
#     model_nn = build_nn_model(X_train_scaled.shape[1])
#     callbacks = [
#         EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
#         ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
#     ]
#     model_nn.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
#                  epochs=100, batch_size=64, callbacks=callbacks, verbose=0)
#     oof_preds[val_idx, 4] = model_nn.predict(X_val_scaled).flatten()
#     test_preds[:, 4] += model_nn.predict(X_test_scaled).flatten() / N_SPLITS

# # --- 6. LEVEL 2 STACKING & CALIBRATION ---
# print("\nðŸ§  Training Level 2 Meta-Model (Stacking)...")

# # Level 2 training data uses OOF predictions from Level 1 models
# X_meta_train = oof_preds
# y_meta_train = y
# X_meta_test = test_preds

# # Train a simple linear meta-model for robust blending
# meta_model = Ridge(alpha=1.0, fit_intercept=True)
# meta_model.fit(X_meta_train, y_meta_train)

# blended_oof_preds = meta_model.predict(X_meta_train)
# final_oof_rmse = np.sqrt(mean_squared_error(y_meta_train, blended_oof_preds))
# print(f"Final Stacked OOF RMSE: {final_oof_rmse:.5f}")

# # Generate final predictions on test data
# stacked_test_preds = meta_model.predict(X_meta_test)

# # --- 7. POST-PROCESSING CALIBRATION ---
# print("calibrating final predictions...")
# ir = IsotonicRegression(y_min=y.min(), y_max=y.max(), out_of_bounds="clip")
# ir.fit(blended_oof_preds, y_meta_train)
# calibrated_preds = ir.transform(stacked_test_preds)

# # --- 8. SUBMISSION ---
# print("\nðŸ“„ Creating final submission file...")
# submission_df = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: calibrated_preds})
# submission_df.to_csv('submission_ultimate_ensemble.csv', index=False)

# print("--- Prediction Statistics ---")
# print(f"Target Mean: {y.mean():.2f} | Final Prediction Mean: {calibrated_preds.mean():.2f}")
# print(f"Target StdDev: {y.std():.2f} | Final Prediction StdDev: {calibrated_preds.std():.2f}")
# print("\nâœ… Submission file created successfully!")


dp = pd.read_csv('/kaggle/input/submissions/submission_final_calibrated.csv')
import matplotlib.pyplot as plt
plt.hist(dp['BeatsPerMinute'], bins=50) # The number of bins can be adjusted
plt.title('Distribution of BeatsPerMinute')
plt.xlabel('BeatsPerMinute')
plt.ylabel('Frequency')
plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from your two files
# NOTE: Replace 'your_first_file.csv' and 'your_second_file.csv'
# with the actual names of your files.
df1 = pd.read_csv('/kaggle/input/submit/submission_26.38444.csv')
df2 = pd.read_csv('/kaggle/input/submissions/submission_diverse_ensemble.csv')

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the density for the first file
sns.kdeplot(df1['BeatsPerMinute'], label='File 1', fill=True, alpha=0.8)

# Plot the density for the second file with a different color
sns.kdeplot(df2['BeatsPerMinute'], label='File 2', fill=True, alpha=1)

# Add a title, labels, and a legend
plt.title('Comparison of BeatsPerMinute Distributions')
plt.xlabel('BeatsPerMinute')
plt.ylabel('Density')
plt.legend()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Set the style to 'ggplot'
plt.style.use('ggplot')

# Load your data
df1 = pd.read_csv('/kaggle/input/submit/submission_26.38444.csv')
df2 = pd.read_csv('/kaggle/input/submissions/submission_diverse_ensemble.csv')

# Create an overlaid histogram with the new style
plt.figure(figsize=(10, 6))
plt.hist(df1['BeatsPerMinute'], bins=10, alpha=0.6, label='File 1')
plt.hist(df2['BeatsPerMinute'], bins=10, alpha=0.6, label='File 2')

plt.title('Comparison of BeatsPerMinute Distributions (ggplot Style)')
plt.xlabel('BeatsPerMinute')
plt.ylabel('Frequency')
plt.legend()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the seaborn style to 'whitegrid'
sns.set_style('whitegrid')

# Load your data
df1 = pd.read_csv('/kaggle/input/submit/submission_26.38444.csv')
df2 = pd.read_csv('/kaggle/input/submissions/submission_diverse_ensemble.csv')

# Create an overlaid density plot with the new style
plt.figure(figsize=(10, 6))
sns.kdeplot(df1['BeatsPerMinute'], label='File 1', fill=True, alpha=0.6)
sns.kdeplot(df2['BeatsPerMinute'], label='File 2', fill=True, alpha=0.6)

plt.title('Comparison of BeatsPerMinute Distributions (whitegrid Style)')
plt.xlabel('BeatsPerMinute')
plt.ylabel('Density')
plt.legend()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: List your filenames
# NOTE: Replace these with the actual names of your five files.
filenames = [
    '/kaggle/input/submissions/submission_final_calibrated.csv',
    '/kaggle/input/submissions/submission_diverse_ensemble.csv',
    '/kaggle/input/submissions/submission_weighted_blend.csv',
    '/kaggle/input/submissions/submission_stacked_final.csv',
    '/kaggle/input/submit/submission_26.38444.csv'
]

# Step 2: Load the data from all files into a dictionary of DataFrames
data_dfs = {}
for filename in filenames:
    try:
        data_dfs[filename] = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        continue

# Step 3: Create the plot
plt.figure(figsize=(10, 6))
sns.set_style('whitegrid')

# Use a color palette to ensure each line is distinct
colors = sns.color_palette("husl", len(data_dfs))

# Plot each distribution in a loop
for i, (filename, df) in enumerate(data_dfs.items()):
    sns.kdeplot(df['BeatsPerMinute'], label=f"File {i+1}", color=colors[i], fill=True, alpha=0.5)

# Add titles, labels, and a legend
plt.title('Comparison of BeatsPerMinute Distributions Across Five Files', fontsize=16)
plt.xlabel('BeatsPerMinute', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Submissions', loc='upper right')
plt.show()


# import pandas as pd

# # Load predictions of all models
# model_predictions = {
#     'final_calibrated': pd.read_csv('/kaggle/input/submissions/submission_final_calibrated.csv'),
#     'diverse_ensemble': pd.read_csv('/kaggle/input/submissions/submission_diverse_ensemble.csv'),
#     'weighted_blend': pd.read_csv('/kaggle/input/submissions/submission_weighted_blend.csv'),
#     'stacked_final': pd.read_csv('/kaggle/input/submissions/submission_stacked_final.csv'),
#     'submission_26_38444': pd.read_csv('/kaggle/input/submit/submission_26.38444.csv')
# }

# # Select models based on performance metrics (e.g., accuracy, F1-score) on validation data.
# # Assuming you've already evaluated models' performance and selected top ones, say top_model_1 and top_model_2.

# top_models = ['submission_26_38444', 'diverse_ensemble','final_calibrated']

# # Combine predictions from top models
# final_predictions = (model_predictions['final_calibrated'] + model_predictions['stacked_final']) / 3

# # Save the final ensemble predictions
# final_predictions.to_csv('final_ensemble_predictions.csv', index=False)



# import pandas as pd

# # Load predictions of all models
# model_predictions = {
#     'final_calibrated': pd.read_csv('/kaggle/input/submissions/submission_final_calibrated.csv'),
#     'diverse_ensemble': pd.read_csv('/kaggle/input/submissions/submission_diverse_ensemble.csv'),
#     'weighted_blend': pd.read_csv('/kaggle/input/submissions/submission_weighted_blend.csv'),
#     'stacked_final': pd.read_csv('/kaggle/input/submissions/submission_stacked_final.csv'),
#     'submission_26_38444': pd.read_csv('/kaggle/input/submit/submission_26.38444.csv')
# }

# # Select models to combine (based on your preference)
# top_models = ['submission_26_38444', 'diverse_ensemble', 'final_calibrated', 'stacked_final']

# # Initialize an empty DataFrame to hold final predictions
# final_predictions = pd.DataFrame()

# # Loop through the selected models to select the highest value for each row (ID)
# for model in top_models:
#     if final_predictions.empty:
#         final_predictions = model_predictions[model]
#     else:
#         # Select the highest prediction across all models for each row (ID)
#         final_predictions = final_predictions.apply(
#             lambda row: row.combine(model_predictions[model].iloc[row.name], max),
#             axis=1
#         )

# # Save the final highest predictions
# final_predictions.to_csv('final_highest_predictions.csv', index=False)



import pandas as pd

# Load predictions of all models
model_predictions = {
    'final_calibrated': pd.read_csv('/kaggle/input/submissions/submission_final_calibrated.csv'),
    'diverse_ensemble': pd.read_csv('/kaggle/input/submissions/submission_diverse_ensemble.csv'),
    'weighted_blend': pd.read_csv('/kaggle/input/submissions/submission_weighted_blend.csv'),
    'stacked_final': pd.read_csv('/kaggle/input/submissions/submission_stacked_final.csv'),
    'submission_26_38444': pd.read_csv('/kaggle/input/submit/submission_26.38444.csv')
}

# Weights assigned to models (based on performance)
model_weights = {
    'final_calibrated': 0.2,
    'diverse_ensemble': 0.25,
    'weighted_blend': 0.15,
    'stacked_final': 0.10,
    'submission_26_38444': 0.3
}

# Use ID column from one of the submissions (all must align)
final_predictions = pd.DataFrame()
final_predictions['id'] = model_predictions['submission_26_38444'].iloc[:, 0]

# Blend prediction columns
blended = sum(
    model_predictions[model].iloc[:, 1] * weight
    for model, weight in model_weights.items()
)

final_predictions['prediction'] = blended

# Save the final blended predictions to CSV
final_predictions.to_csv('final_weighted_blended_predictions.csv', index=False)





