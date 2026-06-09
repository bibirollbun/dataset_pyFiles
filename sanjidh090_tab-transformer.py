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
# - Auto-detects /kaggle/input/*/train.csv, test.csv, sample_submission.csv
# - Quantile binning -> token IDs per numeric feature (+ [CLS])
# - Value tower + gating, Transformer blocks, token dropout, stochastic depth
# - AdamW + warmup + cosine decay, AMP, EMA (via deepcopy)
# - KFold CV (leak-free scaling/binning per fold), prints OOF metrics
# - Full-train on all data + test prediction -> submission.csv
# =========================================================

import os
import math
import copy
import time
import random
import numpy as np
import pandas as pd
from typing import List

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Auto paths (Kaggle) + minimal config
# -----------------------------
def _autodetect_kaggle_paths():
    base = "/kaggle/input"
    best = None  # (root, has_test, has_sample)
    for root, _, files in os.walk(base):
        s = {f.lower() for f in files}
        if "train.csv" in s:
            has_test = "test.csv" in s
            has_sample = "sample_submission.csv" in s
            score = (1 if has_test else 0) + (1 if has_sample else 0)
            if best is None or score > ((1 if best[1] else 0) + (1 if best[2] else 0)):
                best = (root, has_test, has_sample)
    if best is None:
        return {"train": None, "test": None, "sample": None}
    root, has_test, has_sample = best
    return {
        "train": os.path.join(root, "train.csv"),
        "test": os.path.join(root, "test.csv") if has_test else None,
        "sample": os.path.join(root, "sample_submission.csv") if has_sample else None,
    }

_auto = _autodetect_kaggle_paths()
DATA_PATH        = _auto["train"] or "/kaggle/input/playground-series-s5e9/train.csv"
TEST_PATH        = _auto["test"]   or None
SAMPLE_SUB_PATH  = _auto["sample"] or None

# â–¶ï¸� Set these according to your competition
TARGET_COL       = "BeatsPerMinute"   # change if different
ID_COL           = "id"       # change or set to None if absent

# Problem type + submission preferences
TASK_TYPE        = "regression"      # "regression" | "classification"
NUM_CLASSES      = 1                 # set when classification
SUBMISSION_MODE  = "labels"          # "labels" | "prob" | "probs_per_class"
PROB_CLASS_IDX   = 1                 # used when SUBMISSION_MODE == "prob"

# Training hyperparams
N_FOLDS       = 5
RANDOM_STATE  = 42
EPOCHS        = 35
BATCH_SIZE    = 1024
LR            = 3e-4
WEIGHT_DECAY  = 1e-2
WARMUP_EPOCHS = 2

# Model hyperparams
D_MODEL          = 192
N_HEADS          = 8
N_LAYERS         = 4
MLP_RATIO        = 2.0
DROPOUT          = 0.1
TOKEN_DROPOUT    = 0.05
STOCHASTIC_DEPTH = 0.10
NBINS            = 64
USE_EMA          = True
EMA_DECAY        = 0.999
USE_ISOTONIC     = (TASK_TYPE == "regression")

NUM_WORKERS      = 2
PIN_MEMORY       = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP    = (DEVICE == "cuda")

print("[paths]")
print("  train :", DATA_PATH)
print("  test  :", TEST_PATH)
print("  sample:", SAMPLE_SUB_PATH)
print("[columns]")
print("  id    :", ID_COL)
print("  target:", TARGET_COL)
print("[mode]")
print("  task  :", TASK_TYPE)

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
            return [base_lr * float(self.last_epoch + 1) / float(self.warmup_epochs)
                    for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
        return [base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]

class ModelEMA:
    """EMA via deepcopy (no re-instantiation)."""
    def __init__(self, model, decay=0.999):
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
                v_ema.copy_(v)

# -----------------------------
# Quantile binning helpers
# -----------------------------
def compute_bin_edges(x: np.ndarray, nbins: int) -> np.ndarray:
    q = np.linspace(0, 1, nbins + 1)
    edges = np.quantile(x, q)
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([-np.inf, np.inf], dtype=float)
    return edges

def digitize_with_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size < 2:
        return np.zeros_like(x, dtype=np.int64)
    bins = np.clip(np.digitize(x, edges[1:-1], right=True), 0, len(edges)-2)
    return bins.astype(np.int64)

# -----------------------------
# Dataset
# -----------------------------
class TabDataset(Dataset):
    def __init__(self, X_values: np.ndarray, token_ids: np.ndarray, y=None):
        self.X_values = X_values.astype(np.float32)
        self.token_ids = token_ids.astype(np.int64)
        if y is None:
            self.y = None
        else:
            if TASK_TYPE == "regression":
                self.y = y.astype(np.float32)
            else:
                self.y = y.astype(np.int64)
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
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
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

        self.total_tokens = n_features + 1  # + [CLS]

        # Value tower + gate
        self.value_proj = nn.Linear(1, d_model)
        self.value_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        # Token embeddings
        self.bin_embed     = nn.Embedding(nbins, d_model)
        self.feature_embed = nn.Embedding(n_features, d_model)
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed     = nn.Parameter(torch.randn(1, self.total_tokens, d_model) * 0.02)

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
        v = self.value_proj(values.unsqueeze(-1))  # (B,F,d)
        g = self.value_gate(v)                    # (B,F,d)

        # token embeddings (bin + per-feature)
        feat_idx = torch.arange(F, device=values.device).unsqueeze(0).expand(B, F)
        t = self.bin_embed(token_ids) + self.feature_embed(feat_idx)  # (B,F,d)

        # gated fusion
        x = t * g + v * (1 - g)  # (B,F,d)

        # prepend CLS
        cls = self.cls_token.expand(B, -1, -1)    # (B,1,d)
        x = torch.cat([cls, x], dim=1)            # (B,1+F,d)

        # token dropout (except CLS)
        if self.training and self.token_dropout > 0.0:
            mask = (torch.rand(B, F, device=x.device) > self.token_dropout).float().unsqueeze(-1)
            x[:, 1:, :] = x[:, 1:, :] * mask

        # positional embeddings
        x = x + self.pos_embed[:, :x.size(1), :]

        # transformer
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # CLS pooling
        logits = self.head(cls_out)
        return logits

# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(model, ema, loader, optimizer, scaler, epoch, scheduler=None):
    model.train()
    total_loss = 0.0
    n = 0
    for values, token_ids, y in loader:
        if TASK_TYPE == "regression":
            y = y.unsqueeze(1)
        values = values.to(DEVICE)
        token_ids = token_ids.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=AMP):
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
def predict_with_y(model, loader):
    model.eval()
    preds, gts = [], []
    for values, token_ids, y in loader:
        values = values.to(DEVICE)
        token_ids = token_ids.to(DEVICE)
        out = model(values, token_ids).detach().cpu()
        if TASK_TYPE == "classification":
            out = F.softmax(out, dim=1).numpy()
            preds.append(out); gts.append(y.cpu().numpy())
        else:
            preds.append(out.view(-1).numpy()); gts.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    return preds, gts

@torch.no_grad()
def predict_no_y(model, loader):
    model.eval()
    preds = []
    for values, token_ids in loader:
        values = values.to(DEVICE)
        token_ids = token_ids.to(DEVICE)
        out = model(values, token_ids).detach().cpu()
        if TASK_TYPE == "classification":
            preds.append(F.softmax(out, dim=1).numpy())
        else:
            preds.append(out.view(-1).numpy())
    return np.concatenate(preds, axis=0)

# -----------------------------
# CV
# -----------------------------
def run_cv(df: pd.DataFrame, features: List[str]):
    X = df[features].copy()
    y = df[TARGET_COL].values

    label_encoder = None
    if TASK_TYPE == "classification":
        if not np.issubdtype(df[TARGET_COL].dtype, np.number):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(df[TARGET_COL].values)
            print(f"Classes: {list(label_encoder.classes_)}")
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y)
    else:
        splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y)

    oof = np.zeros(len(df), dtype=float if TASK_TYPE=="regression" else np.int64)
    oof_logits = None
    if TASK_TYPE == "classification":
        oof_logits = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)

    fold_scores = []
    for fold, (trn_idx, val_idx) in enumerate(splitter):
        print(f"\n===== Fold {fold+1}/{N_FOLDS} =====")
        X_trn, X_val = X.iloc[trn_idx].copy(), X.iloc[val_idx].copy()
        y_trn, y_val = y[trn_idx], y[val_idx]

        # Per-fold scaler and bin edges (leak-free)
        scaler = StandardScaler().fit(X_trn.values)
        Z_trn = scaler.transform(X_trn.values)
        Z_val = scaler.transform(X_val.values)

        edges_list = [compute_bin_edges(Z_trn[:, j], NBINS) for j in range(Z_trn.shape[1])]

        def tokenize(Z):
            T = np.zeros_like(Z, dtype=np.int64)
            for j, e in enumerate(edges_list):
                T[:, j] = digitize_with_edges(Z[:, j], e)
            return T

        tok_trn = tokenize(Z_trn)
        tok_val = tokenize(Z_val)

        trn_ds = TabDataset(Z_trn, tok_trn, y_trn)
        val_ds = TabDataset(Z_val, tok_val, y_val)
        trn_dl = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

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

        optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scaler_amp = torch.cuda.amp.GradScaler(enabled=AMP)
        scheduler  = WarmupCosineLR(optimizer, warmup_epochs=WARMUP_EPOCHS, max_epochs=EPOCHS)
        ema        = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None

        best_metric = float("inf") if TASK_TYPE=="regression" else -float("inf")
        best_state = None

        for epoch in range(EPOCHS):
            t0 = time.time()
            train_loss = train_one_epoch(model, ema, trn_dl, optimizer, scaler_amp, epoch, scheduler)
            eval_model = ema.ema if (ema is not None) else model
            preds, gts = predict_with_y(eval_model, val_dl)

            if TASK_TYPE == "regression":
                score = rmse(gts, preds); improved = score < best_metric; tag = "RMSE"
            else:
                score = accuracy_score(gts, preds.argmax(axis=1)); improved = score > best_metric; tag = "Acc"

            if improved:
                best_metric = score
                best_state = eval_model.state_dict()

            dt = time.time() - t0
            print(f"Epoch {epoch+1:02d}/{EPOCHS} - train_loss={train_loss:.4f}  val_{tag}={score:.5f}  ({dt:.1f}s)")

        model.load_state_dict(best_state)

        preds, gts = predict_with_y(model, val_dl)
        if TASK_TYPE == "classification":
            oof_logits[val_idx] = preds
            pred_labels = preds.argmax(axis=1)
            oof[val_idx] = pred_labels
            fold_metric = accuracy_score(gts, pred_labels)
        else:
            if USE_ISOTONIC:
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(preds, gts)
                preds = ir.predict(preds)
            oof[val_idx] = preds
            fold_metric = rmse(gts, preds)

        fold_scores.append(fold_metric)
        print(f"Fold {fold+1} {'Acc' if TASK_TYPE=='classification' else 'RMSE'} = {fold_metric:.5f}")

    if TASK_TYPE == "classification":
        overall = accuracy_score(y, oof.astype(int))
        print(f"\nOOF Accuracy = {overall:.5f}")
    else:
        overall = rmse(y, oof)
        print(f"\nOOF RMSE = {overall:.5f}")

    print("Per-fold metrics:", np.array(fold_scores))
    return oof, (oof_logits if TASK_TYPE=="classification" else None)

# -----------------------------
# Full-train + Test Predict
# -----------------------------
def fit_full_and_predict(df_train, df_test, features):
    y = df_train[TARGET_COL].values
    label_encoder = None
    if TASK_TYPE == "classification":
        if not np.issubdtype(df_train[TARGET_COL].dtype, np.number):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(df_train[TARGET_COL].values)
            print(f"[FULL] Classes: {list(label_encoder.classes_)}")

    X_train = df_train[features].values
    X_test  = df_test[features].values

    scaler = StandardScaler().fit(X_train)
    Z_train = scaler.transform(X_train)
    Z_test  = scaler.transform(X_test)

    edges_list = [compute_bin_edges(Z_train[:, j], NBINS) for j in range(Z_train.shape[1])]

    def tokenize(Z):
        T = np.zeros_like(Z, dtype=np.int64)
        for j, e in enumerate(edges_list):
            T[:, j] = digitize_with_edges(Z[:, j], e)
        return T

    tok_train = tokenize(Z_train)
    tok_test  = tokenize(Z_test)

    trn_ds  = TabDataset(Z_train, tok_train, y)
    test_ds = TabDataset(Z_test, tok_test)
    trn_dl  = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

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

    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=AMP)
    scheduler  = WarmupCosineLR(optimizer, warmup_epochs=WARMUP_EPOCHS, max_epochs=EPOCHS)
    ema        = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None

    for epoch in range(EPOCHS):
        _ = train_one_epoch(model, ema, trn_dl, optimizer, scaler_amp, epoch, scheduler)

    eval_model = ema.ema if ema else model
    test_preds = predict_no_y(eval_model, test_dl)  # ndarray

    if TASK_TYPE == "classification":
        if SUBMISSION_MODE == "labels":
            test_preds = test_preds.argmax(axis=1)
            if label_encoder is not None:
                test_preds = label_encoder.inverse_transform(test_preds.astype(int))
        elif SUBMISSION_MODE == "prob":
            test_preds = test_preds[:, PROB_CLASS_IDX]
        elif SUBMISSION_MODE == "probs_per_class":
            pass
        else:
            raise ValueError(f"Unknown SUBMISSION_MODE: {SUBMISSION_MODE}")

    return test_preds

# -----------------------------
# Entry
# -----------------------------
def main():
    if not os.path.exists(DATA_PATH):
        print("DATA_PATH not found. Exiting.")
        return

    df = pd.read_csv(DATA_PATH)

    # ID handling
    if ID_COL and ID_COL in df.columns:
        train_ids = df[ID_COL].values
    else:
        train_ids = np.arange(len(df))

    # Collect candidate features (exclude id/target)
    feature_candidates = [c for c in df.columns if c != TARGET_COL and (ID_COL is None or c != ID_COL)]

    # Coerce to numeric when possible
    for c in feature_candidates:
        if not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    features = [c for c in feature_candidates if np.issubdtype(df[c].dtype, np.number)]

    # Basic imputations
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df[features] = df[features].fillna(df[features].median())

    print(f"Using {len(features)} numeric features.")

    # CV
    _oof, _ = run_cv(df, features)

    # Submission
    if TEST_PATH and os.path.exists(TEST_PATH):
        df_test = pd.read_csv(TEST_PATH)
        if ID_COL and ID_COL in df_test.columns:
            test_ids = df_test[ID_COL].values
        else:
            test_ids = np.arange(len(df_test))

        # Align test features with train
        keep = [c for c in features if c in df_test.columns]
        missing = list(set(features) - set(keep))
        if missing:
            print(f"[Warn] {len(missing)} feature(s) missing in test, dropping them: {sorted(missing)}")
        features_final = keep

        # Coerce + impute test using train medians
        for c in features_final:
            if not np.issubdtype(df_test[c].dtype, np.number):
                df_test[c] = pd.to_numeric(df_test[c], errors="coerce")
        medians = df[features_final].median()
        df_test[features_final] = df_test[features_final].fillna(medians)

        preds = fit_full_and_predict(df[[*features_final, TARGET_COL]], df_test[features_final].copy(), features_final)

        # Build submission in either generic or sample schema
        if SAMPLE_SUB_PATH and os.path.exists(SAMPLE_SUB_PATH):
            sub = pd.read_csv(SAMPLE_SUB_PATH)
            id_col = sub.columns[0]
            sub[id_col] = test_ids
            non_id_cols = [c for c in sub.columns if c != id_col]

            if TASK_TYPE == "classification" and SUBMISSION_MODE == "probs_per_class" and preds.ndim == 2:
                if preds.shape[1] != len(non_id_cols):
                    raise ValueError("Preds shape does not match sample_submission columns.")
                sub[non_id_cols] = preds
            else:
                # single target column
                if len(non_id_cols) != 1:
                    print("[Warn] sample_submission has multiple target cols but SUBMISSION_MODE != 'probs_per_class'. Using first.")
                sub[non_id_cols[0]] = preds
        else:
            # Generic two-column: id + target
            sub = pd.DataFrame({
                (ID_COL or "id"): test_ids,
                "target": preds
            })

        sub.to_csv("submission_csla.csv", index=False)
        print("Saved submission.csv")
    else:
        print("TEST_PATH not set/found. Skipping submission.")

if __name__ == "__main__":
    main()


