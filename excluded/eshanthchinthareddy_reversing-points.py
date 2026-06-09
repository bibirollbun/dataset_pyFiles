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


# ============================================================
# Reversal Points Detection - Leakage-Free, Time-Aware, 2-Stage
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

import gc

# -------------------
# CONFIG
# -------------------
DATA_DIR = "/kaggle/input/detecting-reversal-points-in-us-equities/competition_data"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH  = f"{DATA_DIR}/test.csv"
OUTPUT_PATH = "/kaggle/working/submission.csv"

SEED = 42
N_SPLITS = 5
EMBARGO_STEPS = 8             # exclude this many steps around fold boundaries
TOP_K_MI_BOOL = 200           # per fold: # boolean features to engineer rolling counts for
K_SELECT = 1500               # per fold: SelectKBest(MI) keep top-K overall
np.random.seed(SEED)

print("="*70)
print("REVERSAL DETECTION - Leakage-Free, Time-Aware, 2-Stage Pipeline")
print("="*70)

# -------------------
# LOAD
# -------------------
print("\n[1/8] Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print(f"✓ Train: {train.shape} | Test: {test.shape}")
print(f"✓ Train dtypes: {train.dtypes.value_counts().to_dict()}")

# -------------------
# TARGET MAPPING
# -------------------
print("\n[2/8] Preparing target...")
mapping = {"HH": "H", "LH": "H", "HL": "L", "LL": "L"}
train["class_label"] = train["class_label"].map(mapping).fillna("None")

classes = ["H","L","None"]
label_map = {c:i for i,c in enumerate(classes)}
y = train["class_label"].map(label_map).values

for c in classes:
    cnt = (train["class_label"]==c).sum()
    print(f"  {c}: {cnt} ({100.0*cnt/len(train):.1f}%)")

# -------------------
# BASIC PREP
# -------------------
print("\n[3/8] Feature identification & type fixes...")

meta_cols = ["train_id","id","ticker_id","t","class_label"]
feature_cols = [c for c in train.columns if c not in meta_cols and c in test.columns]

# ensure datetime & sorting
train["t"] = pd.to_datetime(train["t"])
test["t"]  = pd.to_datetime(test["t"])
train = train.sort_values(["ticker_id","t"]).reset_index(drop=True)
test  = test.sort_values(["ticker_id","t"]).reset_index(drop=True)

# unify dtypes (booleans -> int8; numerics stay numeric)
bool_cols = [c for c in feature_cols if train[c].dtype==bool]
num_cols  = [c for c in feature_cols if c not in bool_cols]

for df in (train, test):
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(np.int8)
    # coerce numerics safely
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# fill/clean
for df in (train, test):
    df[feature_cols] = df[feature_cols].fillna(0).replace([np.inf,-np.inf], 0)

print(f"✓ Features total: {len(feature_cols)} | bool: {len(bool_cols)} | numeric: {len(num_cols)}")

# -------------------
# TIME-AWARE CV SPLITS (per ticker, chronological, with embargo)
# -------------------
print("\n[4/8] Building time-aware folds with embargo...")

def time_folds_with_embargo(df, n_splits=5, embargo=8):
    """
    Returns: list of (train_idx, val_idx) for global dataframe df (train only).
    Splits are made per ticker chronologically then concatenated.
    """
    idx_by_fold_tr = [list() for _ in range(n_splits)]
    idx_by_fold_val = [list() for _ in range(n_splits)]

    # process per ticker
    for tid, g in df.groupby("ticker_id", sort=False):
        g = g.sort_values("t")
        n = len(g)
        # boundaries for contiguous folds by time
        fold_sizes = [n // n_splits + (1 if x < n % n_splits else 0) for x in range(n_splits)]
        starts, ends = [], []
        start = 0
        for fs in fold_sizes:
            end = start + fs
            starts.append(start)
            ends.append(end)
            start = end

        # produce masks per fold with embargo
        for k in range(n_splits):
            val_start, val_end = starts[k], ends[k]
            val_idx_local = np.arange(val_start, val_end)

            # embargoed train: exclude [val_start-embargo, val_end+embargo)
            left_cut  = max(0, val_start - embargo)
            right_cut = min(n, val_end + embargo)
            train_left  = np.arange(0, left_cut)
            train_right = np.arange(right_cut, n)
            tr_idx_local = np.concatenate([train_left, train_right]) if len(train_left)+len(train_right) > 0 else np.array([], dtype=int)

            # map local to global
            global_idx = g.index.to_numpy()
            idx_by_fold_tr[k].extend(global_idx[tr_idx_local].tolist())
            idx_by_fold_val[k].extend(global_idx[val_idx_local].tolist())

    # convert to arrays
    folds = []
    for k in range(n_splits):
        tr = np.array(sorted(idx_by_fold_tr[k]), dtype=int)
        vl = np.array(sorted(idx_by_fold_val[k]), dtype=int)
        folds.append((tr, vl))
        print(f"  Fold {k+1}: train={len(tr)} val={len(vl)}")
    return folds

folds = time_folds_with_embargo(train, n_splits=N_SPLITS, embargo=EMBARGO_STEPS)

# -------------------
# Utility: rolling boolean counts (past-only)
# -------------------
def add_bool_rolling_counts(df, bool_list, windows=(3,5,10)):
    """
    For each bool col in bool_list, create past-only rolling sums:
      col_sum_w = sum over last w steps (excluding current row).
    """
    if not bool_list:
        return pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    # operate per ticker to respect chronology
    for tid, g in df.groupby("ticker_id", sort=False):
        order = g.sort_values("t")
        for col in bool_list:
            s = order[col].astype(np.int16)
            # exclude current -> shift(1)
            s_shift = s.shift(1)
            for w in windows:
                out.loc[order.index, f"{col}_sum{w}"] = (
                    s_shift.rolling(window=w, min_periods=1).sum().astype(np.float32)
                )
    # fillna 0 for early periods
    return out.fillna(0.0)

# -------------------
# Two-stage training across folds
# -------------------
print("\n[5/8] Training (two-stage) with fold-local selection & engineering...")

n_train = len(train)
n_test  = len(test)

# OOF composed probabilities: columns [H, L, None]
oof_probs = np.zeros((n_train, 3), dtype=np.float32)
test_probs_accum = np.zeros((n_test, 3), dtype=np.float32)

fold_scores = []

for fold_id, (tr_idx, vl_idx) in enumerate(folds, 1):
    print(f"\n  === Fold {fold_id}/{N_SPLITS} ===")
    X_tr_df = train.iloc[tr_idx]
    X_vl_df = train.iloc[vl_idx]
    y_tr = y[tr_idx]
    y_vl = y[vl_idx]

    # ---- fold-local MI to pick top-K boolean cols for rolling counts
    top_bool = []
    if bool_cols:
        # MI on booleans only (cheap) — sample if needed
        Xb = X_tr_df[bool_cols].values
        top_k = min(TOP_K_MI_BOOL, Xb.shape[1])
        mi = mutual_info_classif(Xb, y_tr, discrete_features=True, random_state=SEED)
        order = np.argsort(mi)[::-1]
        sel = order[:top_k]
        top_bool = [bool_cols[i] for i in sel]
        print(f"    Selected {len(top_bool)} boolean cols for rolling counts")

    # ---- engineer rolling counts (safe: past-only, per ticker)
    roll_tr = add_bool_rolling_counts(X_tr_df, top_bool)
    roll_vl = add_bool_rolling_counts(X_vl_df, top_bool)
    roll_te = add_bool_rolling_counts(test,   top_bool)

    # assemble candidate feature frame (base + rolling counts)
    base_cols = feature_cols  # already cleaned
    Xtr_base = X_tr_df[base_cols].copy()
    Xvl_base = X_vl_df[base_cols].copy()
    Xte_base = test[base_cols].copy()

    # concat engineered rolling features
    X_tr_full = pd.concat([Xtr_base.reset_index(drop=True), roll_tr.reset_index(drop=True)], axis=1)
    X_vl_full = pd.concat([Xvl_base.reset_index(drop=True), roll_vl.reset_index(drop=True)], axis=1)
    X_te_full = pd.concat([Xte_base.reset_index(drop=True), roll_te.reset_index(drop=True)], axis=1)

    # garbage collect intermediates we won't reuse
    del Xtr_base, Xvl_base, Xte_base, roll_tr, roll_vl, roll_te
    gc.collect()

    # ---- fold-local filter: drop near-constant via VarianceThreshold
    # For booleans, variance = p*(1-p). threshold=0.005 ~ drops p in [0,0.005] U [0.995,1]
    vt = VarianceThreshold(threshold=0.005)
    X_tr_vt = vt.fit_transform(X_tr_full)
    X_vl_vt = vt.transform(X_vl_full)
    X_te_vt = vt.transform(X_te_full)

    # ---- fold-local SelectKBest by Mutual Information (works with mixed dtypes as numeric)
    k_keep = min(K_SELECT, X_tr_vt.shape[1])
    skb = SelectKBest(mutual_info_classif, k=k_keep)
    X_tr_sel = skb.fit_transform(X_tr_vt, y_tr)
    X_vl_sel = skb.transform(X_vl_vt)
    X_te_sel = skb.transform(X_te_vt)

    print(f"    Feature dims: VT={X_tr_vt.shape[1]}  SKB={X_tr_sel.shape[1]}")

    # ---- Stage A: None (0) vs HL (1)
    y_tr_A = (y_tr != label_map["None"]).astype(int)
    clf_A = LGBMClassifier(
        objective="binary",
        class_weight={0:1, 1:45},     # emphasize recall for HL
        num_leaves=31, max_depth=6,
        feature_fraction=0.45, bagging_fraction=0.85, bagging_freq=1,
        min_data_in_leaf=120, min_gain_to_split=0.05,
        lambda_l1=2.0, lambda_l2=6.0,
        learning_rate=0.05, n_estimators=1200,
        random_state=SEED+fold_id, n_jobs=-1, verbosity=-1
    )
    clf_A.fit(X_tr_sel, y_tr_A)

    pA_vl = clf_A.predict_proba(X_vl_sel)[:,1]
    pA_te = clf_A.predict_proba(X_te_sel)[:,1]

    # ---- Stage B: H (1) vs L (0), trained only on HL rows
    mask_HL_tr = y_tr != label_map["None"]
    y_tr_B = (y_tr[mask_HL_tr] == label_map["H"]).astype(int)

    clf_B = LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        num_leaves=31, max_depth=6,
        feature_fraction=0.45, bagging_fraction=0.85, bagging_freq=1,
        min_data_in_leaf=80, min_gain_to_split=0.02,
        lambda_l1=1.0, lambda_l2=3.0,
        learning_rate=0.05, n_estimators=900,
        random_state=SEED+fold_id, n_jobs=-1, verbosity=-1
    )
    clf_B.fit(X_tr_sel[mask_HL_tr], y_tr_B)

    pH_vl = clf_B.predict_proba(X_vl_sel)[:,1]   # P(H | HL)
    pH_te = clf_B.predict_proba(X_te_sel)[:,1]

    # ---- Compose probs: P(None)=1-P(HL), P(H)=P(HL)*P(H|HL), P(L)=P(HL)*(1-P(H|HL))
    vl_probs = np.column_stack([pA_vl*pH_vl, pA_vl*(1.0-pH_vl), 1.0-pA_vl]).astype(np.float32)
    te_probs = np.column_stack([pA_te*pH_te, pA_te*(1.0-pH_te), 1.0-pA_te]).astype(np.float32)

    # store OOF & accumulate test
    oof_probs[vl_idx] = vl_probs
    test_probs_accum += te_probs / N_SPLITS

    # quick fold score with naïve argmax (pre-thresholding)
    pred_vl_argmax = vl_probs.argmax(axis=1)
    fold_f1 = f1_score(y_vl, pred_vl_argmax, average="macro")
    fold_scores.append(fold_f1)
    print(f"    Fold Macro-F1 (argmax): {fold_f1:.4f}")

    # cleanup
    del X_tr_full, X_vl_full, X_te_full, X_tr_vt, X_vl_vt, X_te_vt, X_tr_sel, X_vl_sel, X_te_sel
    gc.collect()

print("\n[6/8] Cross-validated performance (pre-threshold):")
print(f"✓ CV Macro-F1 (mean±std): {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

# -------------------
# Threshold Optimization on OOF (2-parameter grid)
# -------------------
print("\n[7/8] Optimizing thresholds on OOF...")

# We optimize two thresholds:
#  - tau_A : HL gate (if P(HL) >= tau_A -> not None)
#  - tau_H : H vs L split (if P(H|HL) >= tau_H -> H else L)
P_HL = oof_probs[:,0] + oof_probs[:,1]
P_H_given = np.divide(oof_probs[:,0], np.maximum(P_HL, 1e-8))

best = {"f1": -1, "tau_A": 0.5, "tau_H": 0.5}
y_true = y

def apply_thresholds(p_hl, p_h, tau_A, tau_H):
    # start as None
    pred = np.full(len(p_hl), label_map["None"], dtype=int)
    mask_hl = p_hl >= tau_A
    # among HL, choose H/L by tau_H
    pred[mask_hl] = np.where(p_h[mask_hl] >= tau_H, label_map["H"], label_map["L"])
    return pred

# grid (focused but adjustable)
tau_A_grid = np.linspace(0.30, 0.75, 19)   # HL gate
tau_H_grid = np.linspace(0.35, 0.65, 13)   # H vs L split

for ta in tau_A_grid:
    for th in tau_H_grid:
        pred = apply_thresholds(P_HL, P_H_given, ta, th)
        f1m = f1_score(y_true, pred, average="macro")
        if f1m > best["f1"]:
            best.update({"f1": f1m, "tau_A": ta, "tau_H": th})

print(f"✓ Optimal thresholds: tau_A={best['tau_A']:.3f}, tau_H={best['tau_H']:.3f}")
print(f"✓ OOF Macro-F1 (thresholded): {best['f1']:.4f}")

# diagnostics
pred_oof = apply_thresholds(P_HL, P_H_given, best["tau_A"], best["tau_H"])
print("\nOOF classification report (thresholded):")
print(classification_report(y_true, pred_oof, target_names=classes, digits=3))

# -------------------
# Submission
# -------------------
print("\n[8/8] Generating submission...")

# Apply same thresholds to averaged test probs
P_HL_te = test_probs_accum[:,0] + test_probs_accum[:,1]
P_H_given_te = np.divide(test_probs_accum[:,0], np.maximum(P_HL_te, 1e-8))
pred_test_idx = apply_thresholds(P_HL_te, P_H_given_te, best["tau_A"], best["tau_H"])
pred_test_lbl = np.array(classes, dtype=object)[pred_test_idx]

submission = pd.DataFrame({
    "id": test["id"] if "id" in test.columns else np.arange(len(test)),
    "class_label": pred_test_lbl
})

submission.to_csv(OUTPUT_PATH, index=False)
print(f"✓ Saved: {OUTPUT_PATH}")

# Distribution
dist = submission["class_label"].value_counts()
print("\nPrediction distribution:")
for c in classes:
    cnt = int(dist.get(c, 0))
    print(f"  {c}: {cnt} ({100.0*cnt/len(submission):.1f}%)")

print("\n" + "="*70)
print("DONE")
print("="*70)


