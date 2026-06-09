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



import os, gc, warnings, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb
from scipy.optimize import nnls


SEED      = 42
random.seed(SEED); np.random.seed(SEED)

N_SPLITS  = 5          
LRATE     = 0.03
EARLY     = 300
N_EST     = 20000

TARGET    = "BeatsPerMinute"
IDCOL     = "id"       
INPUT_DIR = "/kaggle/input/playground-series-s5e9"

WINSOR    = True      
WQ        = 0.005      


# 1) Load
# ---------------------------
train = pd.read_csv(f"{INPUT_DIR}/train.csv")
test  = pd.read_csv(f"{INPUT_DIR}/test.csv")
sub   = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")

assert TARGET in train.columns, f"Target '{TARGET}' not found."
if IDCOL not in train.columns:
    if "ID" in train.columns: IDCOL = "ID"
    else: raise ValueError("ID column not found (expected 'id' or 'ID').")

y = train[TARGET].astype(np.float32).values
feature_cols = [c for c in train.columns if c not in [TARGET, IDCOL]]

X_raw     = train[feature_cols].copy()
Xtest_raw = test[feature_cols].copy()

cat_cols = X_raw.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
print(f"Detected -> numeric: {len(num_cols)} | categorical: {len(cat_cols)}")


# 2) Winsorize numeric 
# ---------------------------
if WINSOR and len(num_cols) > 0:
    ql = X_raw[num_cols].quantile(WQ)
    qh = X_raw[num_cols].quantile(1 - WQ)
    X_raw[num_cols]     = X_raw[num_cols].clip(ql, qh, axis=1)
    Xtest_raw[num_cols] = Xtest_raw[num_cols].clip(ql, qh, axis=1)


# 3) Frequency + OOF Target Encoding 
# ---------------------------
GLOBAL_MEAN = float(y.mean())

def add_cat_encodings(train_df, test_df, y_vec, cat_cols, n_splits=N_SPLITS, seed=SEED):
    if not cat_cols:
        return pd.DataFrame(index=train_df.index), pd.DataFrame(index=test_df.index)

    # Frequency (count) encoding
    freq_tr = pd.DataFrame(index=train_df.index)
    freq_te = pd.DataFrame(index=test_df.index)
    for c in cat_cols:
        vc = train_df[c].value_counts(dropna=False)
        freq_tr[f"{c}_freq"] = train_df[c].map(vc).astype("float32")
        freq_te[f"{c}_freq"] = test_df[c].map(vc).fillna(0).astype("float32")

    # OOF target mean encoding
    te_tr = pd.DataFrame(index=train_df.index)
    te_te = pd.DataFrame(index=test_df.index)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for c in cat_cols:
        oof_col = np.zeros(len(train_df), dtype=np.float32)
        for tr_idx, va_idx in kf.split(train_df):
            m = (
                train_df.iloc[tr_idx][c]
                .to_frame()
                .join(pd.Series(y_vec[tr_idx], index=train_df.index[tr_idx], name="y"))
                .groupby(c)["y"].mean()
            )
            oof_col[va_idx] = train_df.iloc[va_idx][c].map(m).fillna(GLOBAL_MEAN).astype("float32").values
        te_map_full = (
            train_df[c]
            .to_frame()
            .join(pd.Series(y_vec, index=train_df.index, name="y"))
            .groupby(c)["y"].mean()
        )
        te_tr[f"{c}_te"] = oof_col
        te_te[f"{c}_te"] = test_df[c].map(te_map_full).fillna(GLOBAL_MEAN).astype("float32")

    enc_tr = pd.concat([freq_tr, te_tr], axis=1)
    enc_te = pd.concat([freq_te, te_te], axis=1)
    return enc_tr, enc_te

enc_tr, enc_te = add_cat_encodings(X_raw, Xtest_raw, y, cat_cols, n_splits=N_SPLITS, seed=SEED)


# 4) Build final design matrix 
# ---------------------------
from sklearn.impute import SimpleImputer
if len(num_cols):
    imp = SimpleImputer(strategy="median")
    X_num      = imp.fit_transform(X_raw[num_cols]).astype(np.float32)
    Xtest_num  = imp.transform(Xtest_raw[num_cols]).astype(np.float32)
else:
    X_num      = np.zeros((len(X_raw), 0), dtype=np.float32)
    Xtest_num  = np.zeros((len(Xtest_raw), 0), dtype=np.float32)

X = np.concatenate([X_num, enc_tr.values.astype(np.float32)], axis=1)
X_test = np.concatenate([Xtest_num, enc_te.values.astype(np.float32)], axis=1)
print("Final design matrix:", X.shape, X_test.shape)

def rmse(a,b): return mean_squared_error(a,b,squared=False)


# 5) StratifiedKFold on binned target
# ---------------------------
bins = pd.qcut(y, q=min(10, max(2, len(np.unique(y))//2)),
               labels=False, duplicates="drop")
skf  = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


# 6) LightGBM (seed-bag), memory-lean
# ---------------------------
SEEDS = [42, 2025]   # add a 3rd (e.g., 7) for final if time permits
lgb_oof  = np.zeros(len(y), dtype=np.float32)
lgb_test = np.zeros(X_test.shape[0], dtype=np.float32)

for s in SEEDS:
    tmp_oof = np.zeros(len(y), dtype=np.float32)
    skf_s = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=s)
    for fold, (tr, va) in enumerate(skf_s.split(X, bins), 1):
        X_tr, X_va, y_tr, y_va = X[tr], X[va], y[tr], y[va]
        lgbm = LGBMRegressor(
            n_estimators=N_EST, learning_rate=LRATE,
            num_leaves=48, max_depth=-1,
            subsample=0.8, colsample_bytree=0.75,
            min_child_samples=60, reg_lambda=1.5, reg_alpha=0.1,
            metric="rmse", random_state=s + fold
        )
        lgbm.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(EARLY), lgb.log_evaluation(0)]
        )
        p = lgbm.predict(X_va).astype(np.float32)
        tmp_oof[va] = p
        lgb_test += lgbm.predict(X_test).astype(np.float32) / (N_SPLITS * len(SEEDS))
        del X_tr, X_va, y_tr, y_va, lgbm, p; gc.collect()
    lgb_oof += tmp_oof / len(SEEDS)

lgb_cv = rmse(y, lgb_oof)
print(f"LGBM (bagged) CV RMSE: {lgb_cv:.5f}")


# 7) XGBoost (hist), regularized
# ---------------------------
xgb_oof  = np.zeros(len(y), dtype=np.float32)
xgb_test = np.zeros(X_test.shape[0], dtype=np.float32)

for fold, (tr, va) in enumerate(skf.split(X, bins), 1):
    X_tr, X_va, y_tr, y_va = X[tr], X[va], y[tr], y[va]
    xgbm = xgb.XGBRegressor(
        n_estimators=N_EST, learning_rate=LRATE,
        max_depth=6, min_child_weight=8.0,
        subsample=0.8, colsample_bytree=0.7,
        reg_lambda=3.0, reg_alpha=0.0, gamma=0.15,
        max_bin=256, tree_method="hist", predictor="auto",
        random_state=SEED + fold, nthread=-1
    )
    xgbm.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        early_stopping_rounds=EARLY,
        verbose=False
    )
    p = xgbm.predict(X_va).astype(np.float32)
    xgb_oof[va] = p
    xgb_test += xgbm.predict(X_test).astype(np.float32) / N_SPLITS
    del X_tr, X_va, y_tr, y_va, xgbm, p; gc.collect()

xgb_cv = rmse(y, xgb_oof)
print(f"XGB  CV RMSE: {xgb_cv:.5f}")


# 8) NNLS blend + calibration
# ---------------------------
oof_mat  = np.vstack([lgb_oof, xgb_oof]).T.astype(np.float64)
test_mat = np.vstack([lgb_test, xgb_test]).T.astype(np.float64)
w, _     = nnls(oof_mat, y.astype(np.float64))
w        = w / (w.sum() if w.sum()!=0 else 1.0)
ens_oof  = oof_mat @ w
ens_test = test_mat @ w
print("NNLS weights [LGBM, XGB]:", np.round(w,4))
print("Ensemble OOF RMSE:", rmse(y, ens_oof))

# linear mean-variance calibration on OOF, apply to test
mu_y, mu_o = float(y.mean()), float(ens_oof.mean())
var_o = float(ens_oof.var()) + 1e-12
cov_  = float(((ens_oof - mu_o)*(y - mu_y)).mean())
b = cov_ / var_o; a = mu_y - b*mu_o
ens_oof_cal  = a + b*ens_oof
ens_test_cal = a + b*ens_test
use_cal = rmse(y, ens_oof_cal) <= rmse(y, ens_oof)
final_pred = ens_test_cal if use_cal else ens_test
print("Calibration used:", use_cal)


# 9) Exact-duplicate row signature override
# ---------------------------
def row_sig(df):
    tmp = df.copy()
    for c in tmp.columns:
        tmp[c] = tmp[c].astype(str).fillna("NA")
    return pd.util.hash_pandas_object(tmp, index=False)

sig_train = row_sig(train[feature_cols])
sig_test  = row_sig(test[feature_cols])

sig_to_mean = pd.DataFrame({"sig": sig_train, "y": y}).groupby("sig")["y"].mean()
override_idx = sig_test.isin(sig_to_mean.index)
if override_idx.any():
    overrides = sig_test[override_idx].map(sig_to_mean).values.astype(np.float32)
    final_pred[override_idx.values] = overrides
    print(f"Signature overrides applied to {override_idx.sum()} test rows.")


# 10) Clip + tiny debias & save files
# ---------------------------
gmean = float(train[TARGET].mean())
final_pred = np.clip(final_pred, 40, 220)
final_pred = 0.98*final_pred + 0.02*gmean

# Ensemble submission
sub1 = sub.copy()
sub1["BeatsPerMinute"] = final_pred.astype(np.float32)
sub1.to_csv("submission.csv", index=False)
print("Wrote submission.csv (ensemble)")

# Single-model hedge (LGBM)
single = np.clip(lgb_test, 40, 220).astype(np.float32)
sub2 = sub.copy()
sub2["BeatsPerMinute"] = 0.98*single + 0.02*gmean
sub2.to_csv("submission_lgbm.csv", index=False)
print("Wrote submission_lgbm.csv (single)")


