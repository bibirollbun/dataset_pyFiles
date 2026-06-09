%%time
# === Cell 1: imports / paths / seed / utils ===
import os, gc, sys, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from sklearn.model_selection import GroupKFold, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer

warnings.filterwarnings("ignore")

# ---- paths (adjust base_dir if needed) ----
# Works on Kaggle & Paperspace; edit base_dir if your data elsewhere
base_dir = Path("/kaggle/input/drw-crypto-market-prediction")
if not base_dir.exists():
    base_dir = Path("/notebooks/Kaggle Competitions/data/drw-crypto-market-prediction")
if not base_dir.exists():
    base_dir = Path("data/drw-crypto-market-prediction")  # last fallback
DATA_DIR = base_dir
WORK_DIR = Path(os.getenv("KAGGLE_WORKING_DIR", "/kaggle/working"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

print("DATA_DIR:", DATA_DIR)
print("WORK_DIR:", WORK_DIR)

# ---- seed everywhere ----
SEED = 42
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
set_seed()

# ---- Pearson (numpy) + scorer for sklearn ----
def pearson_r_np(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.std() == 0 or y_pred.std() == 0: 
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])

pearson_scorer = make_scorer(pearson_r_np, greater_is_better=True)



# === Reload/clean with Polars & align columns ===

train_pl = pl.read_parquet(DATA_DIR / "train.parquet")
test_pl  = pl.read_parquet(DATA_DIR / "test.parquet")

assert "label" in train_pl.columns, "train must contain 'label'"

# strip index-like columns that can appear when parquet was saved from pandas
def clean_cols(cols):
    return [c for c in cols if c != "label" and not c.startswith("__index_level_") and c != "index"]

train_feats = clean_cols(train_pl.columns)
test_feats  = clean_cols(test_pl.columns)

# keep only features present in BOTH train and test
feat_cols = [c for c in train_feats if c in set(test_feats)]

# cast to float32; leave NAs (Ridge can work after scaling)
train_pl = train_pl.select(feat_cols + ["label"]).with_columns([
    pl.col(feat_cols).cast(pl.Float32),
    pl.col("label").cast(pl.Float32),
])
test_pl = test_pl.select(feat_cols).with_columns([
    pl.col(feat_cols).cast(pl.Float32),
])

# hand off to sklearn as pandas/NumPy
X_df    = train_pl.select(feat_cols).to_pandas()
y_vec   = train_pl.select("label").to_numpy().ravel()
test_df = test_pl.select(feat_cols).to_pandas()

print("n_features:", len(feat_cols))
print("Shapes:", X_df.shape, y_vec.shape, test_df.shape)
print("Example features:", feat_cols[:8])



# === Cell 3: GroupKFold via Polars row-hash (fallback to pandas if needed) ===
try:
    # Polars row-hash (UInt64), stable and fast
    groups = train_pl.select(pl.struct(feat_cols).hash_rows()).to_numpy().ravel().astype("uint64")
except Exception as e:
    print("Polars hash_rows() failed; falling back to pandas hashing. Reason:", e)
    groups = pd.util.hash_pandas_object(X_df, index=False).astype("uint64").to_numpy()

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

folds = []
for tr_idx, va_idx in gkf.split(X_df, y_vec, groups=groups):
    folds.append((tr_idx, va_idx))

# diagnostics
vals, counts = np.unique(groups, return_counts=True)
dup_groups = int((counts > 1).sum())
dup_rows   = int(counts[counts > 1].sum())
dup_rate   = 100.0 * dup_rows / len(groups)
print(f"duplicate groups (>=2 rows): {dup_groups} | rows in duplicates: {dup_rows} ({dup_rate:.2f}%)")
print("fold sizes (train/valid):", [(len(a), len(b)) for a,b in folds])



# === Cell 4: ridge fold trainer ===
RIDGE_ALPHAS = np.logspace(-4, 3, 20)  # 1e-4 .. 1e3 (tweak as needed)

def train_ridge_one_fold(tr_idx, va_idx, alphas=RIDGE_ALPHAS, seed=SEED):
    X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
    y_tr, y_va = y_vec[tr_idx], y_vec[va_idx]

    # scaler + ridge in one pipeline (correct scaling inside CV)
    pipe = Pipeline([
        ("sc", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(random_state=seed))
    ])

    # inner CV for hyperparam search
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=seed)
    search = GridSearchCV(
        estimator=pipe,
        param_grid={"ridge__alpha": alphas},
        scoring=pearson_scorer,
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    search.fit(X_tr, y_tr)

    best = search.best_estimator_
    # (best already refit on inner train folds; refit on full outer train for stability)
    best.fit(X_tr, y_tr)

    va_pred = best.predict(X_va).astype("float32")
    r = pearson_r_np(y_va, va_pred)

    te_pred = best.predict(test_df).astype("float32")

    return va_idx, va_pred, te_pred, r, search.best_params_



# === Cell 5: run CV ===
oof = np.zeros(len(y_vec), dtype=np.float32)
test_preds, fold_scores, best_params_per_fold = [], [], []

for f, (tr_idx, va_idx) in enumerate(folds, 1):
    print(f"\n=== Ridge Fold {f}/{len(folds)} | train={len(tr_idx)} valid={len(va_idx)} ===")
    va_i, va_p, te_p, r, bp = train_ridge_one_fold(tr_idx, va_idx)
    oof[va_i] = va_p
    test_preds.append(te_p)
    fold_scores.append(r)
    best_params_per_fold.append(bp)
    print(f"Fold {f} Pearson r: {r:.6f} | best params: {bp}")
    gc.collect();

oof_r = pearson_r_np(y_vec, oof)
print(f"\nRidge OOF Pearson r: {oof_r:.6f} | per-fold: {np.round(fold_scores, 6)}")



# === Cell 6: save OOF + submission ===
oof_path = WORK_DIR / "oof_ridge.csv"
pd.DataFrame({"oof_pred": oof}).to_csv(oof_path, index=False)

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub["prediction"] = np.mean(np.stack(test_preds, axis=0), axis=0).astype("float32")
sub_path = WORK_DIR / "submission_ridge.csv"
sub.to_csv(sub_path, index=False)

print("Saved:")
print("  OOF ->", oof_path)
print("  SUB ->", sub_path)


