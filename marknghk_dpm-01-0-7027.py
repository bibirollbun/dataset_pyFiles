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


import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import combinations





# -------------------
# Config
# -------------------
TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e12/test.csv"
ORG_PATH   = "/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv"
SUB_PATH   = "/kaggle/input/playground-series-s5e12/sample_submission.csv"

INDEX  = "id"
TARGET = "diagnosed_diabetes"

FOLDS = 10
RANDOM_STATE = 42
WIN_SIZE = 1000
THRESHOLD = 88
TEST_COL = "physical_activity_minutes_per_week"
VAL_WEIGHT_RATIO = 15

xgb_params = {
    "objective": "binary:logistic",
    "learning_rate": 0.01,
    "max_depth": 6,
    "scale_pos_weight": 1.2,
    "subsample": 0.8,
    "reg_lambda": 3.0,
    "colsample_bytree": 0.8,
    "min_child_weight": 8,
    "n_jobs": -1,
    "eval_metric": "auc",
    # FIX: Add tree_method="hist" for categorical support
    "tree_method": "hist",
    # FIX: Hardcode CPU if you are not enabling GPU accelerator
    "device": "cpu", 
    # "enable_categorical": True, # Remove from here, keep in DMatrix
}



# -------------------
# Utilities
# -------------------
def find_cutoff_id(train: pd.DataFrame) -> int:
    rm = train[TEST_COL].rolling(window=WIN_SIZE).mean()
    m = rm > THRESHOLD
    cutoff = rm[m].index.min()
    if pd.isna(cutoff):
        return int(train[INDEX].max() + 1)
    return int(cutoff)

def clip_outliers(train, test, num_cols, q_lo=0.01, q_hi=0.99):
    for c in num_cols:
        lo = train[c].quantile(q_lo)
        hi = train[c].quantile(q_hi)
        train[c] = train[c].clip(lo, hi)
        test[c]  = test[c].clip(lo, hi)
    return train, test

def add_orig_stats(train, test, org, base_cols):
    global_mean = float(org[TARGET].mean())
    created = []

    for c in base_cols:
        mean_map = org.groupby(c)[TARGET].mean().rename(f"orig_mean_{c}")
        cnt_map  = org.groupby(c).size().rename(f"orig_count_{c}")

        train = train.merge(mean_map, on=c, how="left")
        test  = test.merge(mean_map, on=c, how="left")
        train = train.merge(cnt_map,  on=c, how="left")
        test  = test.merge(cnt_map,  on=c, how="left")

        train[f"orig_mean_{c}"]  = train[f"orig_mean_{c}"].fillna(global_mean).astype("float32")
        test[f"orig_mean_{c}"]   = test[f"orig_mean_{c}"].fillna(global_mean).astype("float32")
        train[f"orig_count_{c}"] = train[f"orig_count_{c}"].fillna(0).astype("float32")
        test[f"orig_count_{c}"]  = test[f"orig_count_{c}"].fillna(0).astype("float32")

        created += [f"orig_mean_{c}", f"orig_count_{c}"]

    return train, test, created

def add_round_feats(train, test, cols=("triglycerides", "cholesterol_total", "systolic_bp")):
    out = []
    for c in cols:
        for suffix, nd in [("1s", 0), ("10s", -1), ("100s", -2)]:
            newc = f"{c}_rnd_{suffix}"
            for df in (train, test):
                df[newc] = df[c].round(nd).astype(int)
            out.append(newc)
    return train, test, out

def add_cat_interactions(train, test, cat_cols):
    inter = []
    for k in (2, 3):
        for cols in combinations(cat_cols, k):
            newc = "_".join(cols)
            for df in (train, test):
                df[newc] = df[list(cols)].astype(str).agg("_".join, axis=1).astype("category")
            inter.append(newc)
    return train, test, inter

def enforce_category(dfs, cols):
    for df in dfs:
        for c in cols:
            # convert any object/category mix to stable category dtype
            df[c] = df[c].astype("category")



# -------------------
# Target Encoder (OOF + smoothing, fixed dtypes)
# -------------------
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, cv=5, smooth=1.0, drop_original=True, random_state=42):
        self.cols = cols
        self.cv = cv
        self.smooth = float(smooth)
        self.drop_original = drop_original
        self.random_state = random_state

    def fit(self, X, y):
        self.global_mean_ = float(pd.Series(y).mean())
        self.maps_ = {}

        # store per-column mean/count tables
        y_ser = pd.Series(y).reset_index(drop=True)
        X_df = X.reset_index(drop=True)

        for c in self.cols:
            key = X_df[c].astype(str)
            stats = pd.DataFrame({"key": key, "y": y_ser}).groupby("key")["y"].agg(["mean", "count"])
            self.maps_[c] = stats
        return self

    def transform(self, X):
        Xo = X.copy()
        for c in self.cols:
            stats = self.maps_[c]
            key = Xo[c].astype(str)

            enc = key.map(stats["mean"]).astype("float32")
            cnt = key.map(stats["count"]).astype("float32")

            m = float(self.smooth)
            enc = (cnt * enc + m * self.global_mean_) / (cnt + m)
            Xo[f"TE_{c}"] = enc.fillna(self.global_mean_).astype("float32")

        if self.drop_original:
            Xo = Xo.drop(columns=self.cols)

        return Xo

    def fit_transform(self, X, y):
        X_df = X.reset_index(drop=True)
        y_ser = pd.Series(y).reset_index(drop=True)

        oof_te = pd.DataFrame(index=X_df.index)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        for tr, va in kf.split(X_df, y_ser):
            self.fit(X_df.iloc[tr], y_ser.iloc[tr])
            Xt = self.transform(X_df.iloc[va])
            # only keep the new TE_* columns in the OOF container
            te_cols = [c for c in Xt.columns if c.startswith("TE_")]
            oof_te.loc[va, te_cols] = Xt[te_cols].values

        # fit on full data so later .transform(test) uses full mapping
        self.fit(X_df, y_ser)

        # return X with OOF TE columns inserted, and (optionally) original cols dropped
        X_out = X_df.copy()
        for c in oof_te.columns:
            X_out[c] = oof_te[c].astype("float32")

        if self.drop_original:
            X_out = X_out.drop(columns=self.cols)

        return X_out




# ============================================================
# Main: load -> feature engineering -> CV train -> submission
# ============================================================
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
org   = pd.read_csv(ORG_PATH)

cutoff_id = find_cutoff_id(train)

cat_cols = [c for c in test.columns if test[c].dtype == "O"]  # 6 cols in your notebook
num_cols = [c for c in test.columns if test[c].dtype in ["float64", "int64"] and c != INDEX]
base_cols = [c for c in train.columns if c not in [INDEX, TARGET]]

# numeric clipping
train, test = clip_outliers(train, test, num_cols)

# orig stats features (from org)
train, test, ORIG = add_orig_stats(train, test, org, base_cols)

# rounding features
train, test, ROUND = add_round_feats(train, test)

# ensure raw categoricals are category before building interactions
enforce_category([train, test], cat_cols)

# interaction features (2-way and 3-way on categoricals)
train, test, INTER = add_cat_interactions(train, test, cat_cols)

# enforce categoricals again (important)
enforce_category([train, test], cat_cols + INTER)

FEATURES = ORIG + num_cols + cat_cols + ROUND + INTER

# build weighted full training matrix using cutoff-based split
train_df = train[train[INDEX] < cutoff_id].copy()
val_df   = train[train[INDEX] >= cutoff_id].copy()

X = pd.concat([train_df[FEATURES], val_df[FEATURES]], axis=0).reset_index(drop=True)
y = pd.concat([train_df[TARGET],  val_df[TARGET]], axis=0).reset_index(drop=True)

w = np.concatenate([
    np.ones(len(train_df), dtype="float32"),
    np.ones(len(val_df), dtype="float32") * VAL_WEIGHT_RATIO
])

# CV
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof   = np.zeros(len(X), dtype="float32")
preds = np.zeros(len(test), dtype="float32")

fold_artifacts = []  # (te_inter, te_round, booster) for reuse

for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    X_tr = X.iloc[tr].copy()
    y_tr = y.iloc[tr].copy()
    w_tr = w[tr]

    X_va = X.iloc[va].copy()
    y_va = y.iloc[va].copy()
    w_va = w[va]

    X_te = test[FEATURES].copy()

    # IMPORTANT: preserve category dtype before encoding
    enforce_category([X_tr, X_va, X_te], cat_cols + INTER)

    # TE for interaction + rounding (drop originals)
    te_inter = TargetEncoder(cols=INTER, cv=5, smooth=1.0, drop_original=True, random_state=RANDOM_STATE)
    X_tr = te_inter.fit_transform(X_tr, y_tr)
    X_va = te_inter.transform(X_va)
    X_te = te_inter.transform(X_te)

    te_round = TargetEncoder(cols=ROUND, cv=5, smooth=1.0, drop_original=True, random_state=RANDOM_STATE)
    X_tr = te_round.fit_transform(X_tr, y_tr)
    X_va = te_round.transform(X_va)
    X_te = te_round.transform(X_te)

    # CRITICAL: after TE, the original 6 categoricals remain; ensure they are 'category' not 'object'
    enforce_category([X_tr, X_va, X_te], cat_cols)

    # Guardrail: no object columns allowed in DMatrix
    obj_cols = X_tr.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        raise ValueError(f"Object columns still present: {obj_cols}")

    dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, enable_categorical=True)
    dva = xgb.DMatrix(X_va, label=y_va, weight=w_va, enable_categorical=True)
    dte = xgb.DMatrix(X_te, enable_categorical=True)

    
    params = dict(xgb_params)
    params["random_state"] = RANDOM_STATE + fold

    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=10_000,
        evals=[(dtr, "train"), (dva, "valid")],
        early_stopping_rounds=200,
        verbose_eval=500
    )

    oof[va] = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
    preds  += booster.predict(dte, iteration_range=(0, booster.best_iteration + 1))

    fold_artifacts.append((te_inter, te_round, booster))

preds /= FOLDS

# submission
sub = pd.read_csv(SUB_PATH)
sub[TARGET] = preds
sub.to_csv("submission.csv", index=False)

# optional saves for ensembling
np.save("oof.npy", oof)
np.save("preds.npy", preds)

