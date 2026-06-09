# =====================================================
# IEEE-CIS Fraud Detection - LightGBM baseline (offline)
# =====================================================

import os, gc, random
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from pandas.api.types import is_numeric_dtype

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ---- Find dataset directory ----
def find_ieee_dir():
    base = Path("/kaggle/input")
    for d in base.iterdir():
        if (d/"train_transaction.csv").exists() and (d/"test_transaction.csv").exists():
            return d
    raise FileNotFoundError("Attach IEEE-CIS Fraud Detection dataset in Input panel.")
DATA_DIR = find_ieee_dir()
print("Data dir:", DATA_DIR)

# ---- Load & merge ----
train_tr = pd.read_csv(DATA_DIR/"train_transaction.csv")
train_id = pd.read_csv(DATA_DIR/"train_identity.csv")
test_tr  = pd.read_csv(DATA_DIR/"test_transaction.csv")
test_id  = pd.read_csv(DATA_DIR/"test_identity.csv")
sub_tpl  = pd.read_csv(DATA_DIR/"sample_submission.csv")  # TransactionID,isFraud

train = train_tr.merge(train_id, on="TransactionID", how="left")
test  = test_tr.merge(test_id,   on="TransactionID", how="left")
del train_tr, train_id, test_tr, test_id; gc.collect()
print("Shapes:", train.shape, test.shape)

TARGET = "isFraud"

# ---- Feature engineering ----
# 1) Frequency encode some categoricals
def freq_encode(df_tr, df_te, cols):
    for c in cols:
        freqs = df_tr[c].value_counts(dropna=False)
        df_tr[c+"_fe"] = df_tr[c].map(freqs)
        df_te[c+"_fe"]  = df_te[c].map(freqs)
    return df_tr, df_te

base_cats = [c for c in ["card1","card2","card3","card4",
                         "addr1","P_emaildomain","R_emaildomain",
                         "DeviceType","DeviceInfo"] if c in train.columns]
train, test = freq_encode(train, test, base_cats)

# 2) Time features
if "TransactionDT" in train.columns:
    for df in (train, test):
        df["DT_day"]       = (df["TransactionDT"] // (24*60*60)).astype("float32")
        df["DT_hour"]      = ((df["TransactionDT"] // (60*60)) % 24).astype("float32")
        df["DT_dayofweek"] = (df["DT_day"] % 7).astype("float32")

# 3) Encode ALL object columns
obj_cols = sorted(
    set([c for c in train.columns if train[c].dtype == "object"]) |
    set([c for c in test.columns  if test[c].dtype  == "object"])
)
print("Object columns to encode:", len(obj_cols))
for c in obj_cols:
    if c not in train.columns: train[c] = "__NA__"
    if c not in test.columns:  test[c]  = "__NA__"
    both = pd.concat([train[c].astype(str), test[c].astype(str)], axis=0).fillna("__NA__")
    codes, _ = pd.factorize(both, sort=True)
    train[c] = codes[:len(train)].astype("int32")
    test[c]  = codes[len(train):].astype("int32")

# ---- Align features between train and test ----
FEATURES = [c for c in train.columns if c != TARGET]
missing_in_test = [c for c in FEATURES if c not in test.columns]
if missing_in_test:
    add = {}
    for c in missing_in_test:
        if c in train.columns and is_numeric_dtype(train[c]):
            add[c] = np.float32(train[c].median())
        else:
            add[c] = 0
    test = pd.concat([test,
                      pd.DataFrame({k: np.repeat(v, len(test)) for k, v in add.items()},
                                   index=test.index)], axis=1)

# ---- Fill numeric NaNs ----
for c in FEATURES:
    if is_numeric_dtype(train[c]):
        med = train[c].median()
        train[c] = train[c].fillna(med)
        test[c]  = test[c].fillna(med)

# ---- Train/valid split (time-based) ----
if "TransactionDT" in train.columns:
    thr = np.quantile(train["TransactionDT"], 0.80)
    trn_idx = train.index[train["TransactionDT"] <= thr]
    val_idx = train.index[train["TransactionDT"] >  thr]
else:
    msk = np.random.rand(len(train)) < 0.8
    trn_idx = train.index[msk]; val_idx = train.index[~msk]

X_tr, y_tr = train.loc[trn_idx, FEATURES], train.loc[trn_idx, TARGET].astype(int)
X_val, y_val = train.loc[val_idx, FEATURES], train.loc[val_idx, TARGET].astype(int)

# ---- LightGBM ----
lgb_train = lgb.Dataset(X_tr, y_tr)
lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train)

params = dict(
    objective="binary", metric="auc", boosting_type="gbdt",
    learning_rate=0.03, num_leaves=96, max_depth=-1,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.10, reg_lambda=0.20,
    min_child_samples=80, verbose=-1, seed=SEED
)

model = lgb.train(
    params, lgb_train, num_boost_round=8000,
    valid_sets=[lgb_train, lgb_val], valid_names=["train","valid"],
    callbacks=[lgb.early_stopping(300), lgb.log_evaluation(100)]
)

val_pred = model.predict(X_val, num_iteration=model.best_iteration)
print("Validation AUC:", roc_auc_score(y_val, val_pred))

# ---- Predict test & save submission ----
feat_names = list(model.feature_name())

# add missing columns in test if any
for c in feat_names:
    if c not in test.columns:
        test[c] = 0

# ensure numeric
for c in feat_names:
    test[c] = pd.to_numeric(test[c], errors="coerce")
    if test[c].isna().any():
        fill = train[c].median() if c in train.columns else 0
        test[c] = test[c].fillna(fill)

X_test = test[feat_names]
pred = model.predict(X_test, num_iteration=model.best_iteration)

sub = sub_tpl.copy()
sub["isFraud"] = pred
out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Saved:", out_path, "| shape:", sub.shape)
print(sub.head())





