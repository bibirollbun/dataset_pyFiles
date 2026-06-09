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


# Kaggle-ready solution for "Predicting the Beats-per-Minute of Songs" (Playground S5E9)
# This script:
#  - basic preprocessing + simple feature engineering
#  - trains LightGBM + CatBoost with KFold
#  - averages predictions for final submission

import os
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

# -------- config --------
SEED = 2025
NFOLDS = 5
TARGET = "BeatsPerMinute"
ID_COL = "id"
DATA_DIR = "/kaggle/input/playground-series-s5e9"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUTPUT_PATH = "/kaggle/working/submission.csv"

np.random.seed(SEED)

# -------- helpers --------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def reduce_mem(df):
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df

# -------- load data --------
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_SUB_PATH)

train = reduce_mem(train)
test = reduce_mem(test)

# -------- preprocessing --------
train.drop_duplicates(ID_COL, inplace=True)

features = [c for c in train.columns if c not in [ID_COL, TARGET]]

nunique = train[features].nunique()
const_feats = nunique[nunique == 1].index.tolist()
if const_feats:
    for c in const_feats:
        features.remove(c)
    train.drop(columns=const_feats, inplace=True)
    test.drop(columns=const_feats, inplace=True)

num_cols = train[features].select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    if train[col].isna().any() or test[col].isna().any():
        med = train[col].median()
        train[col].fillna(med, inplace=True)
        test[col].fillna(med, inplace=True)

# simple row-level stats
train["r_mean"] = train[num_cols].mean(axis=1)
train["r_std"]  = train[num_cols].std(axis=1).fillna(0)
train["r_sum"]  = train[num_cols].sum(axis=1)
test["r_mean"]  = test[num_cols].mean(axis=1)
test["r_std"]   = test[num_cols].std(axis=1).fillna(0)
test["r_sum"]   = test[num_cols].sum(axis=1)

extra = ["r_mean", "r_std", "r_sum"]
for c in extra:
    if c not in features:
        features.append(c)

# encode categoricals
cat_cols = train[features].select_dtypes(include=["object"]).columns.tolist()
if cat_cols:
    from sklearn.preprocessing import LabelEncoder
    for c in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[c].astype(str), test[c].astype(str)], axis=0)
        le.fit(combined)
        train[c] = le.transform(train[c].astype(str))
        test[c]  = le.transform(test[c].astype(str))

# -------- model training --------
oof_preds_lgb = np.zeros(len(train))
oof_preds_cat = np.zeros(len(train))
test_preds_lgb = np.zeros(len(test))
test_preds_cat = np.zeros(len(test))

kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
fold = 0

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 128,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.4,
    "n_jobs": -1,
    "verbosity": -1,
    "seed": SEED,
}

cat_params = {
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 8,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": SEED,
    "early_stopping_rounds": 200,
    "verbose": 100,
}

start = time.time()
for train_idx, val_idx in kf.split(train):
    fold += 1
    X_tr = train.iloc[train_idx][features]
    y_tr = train.iloc[train_idx][TARGET]
    X_val = train.iloc[val_idx][features]
    y_val = train.iloc[val_idx][TARGET]

    # ---- LightGBM ----
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_val, label=y_val)
    clf = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=10000,
        valid_sets=[dtrain, dvalid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=200)
        ]
    )
    oof_preds_lgb[val_idx] = clf.predict(X_val, num_iteration=clf.best_iteration)
    test_preds_lgb += clf.predict(test[features], num_iteration=clf.best_iteration) / NFOLDS

    # ---- CatBoost ----
    cat_features_idx = [X_tr.columns.get_loc(c) for c in cat_cols] if cat_cols else []
    cat = CatBoostRegressor(**cat_params)
    cat.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        cat_features=cat_features_idx,
        use_best_model=True,
        verbose=False
    )
    oof_preds_cat[val_idx] = cat.predict(X_val)
    test_preds_cat += cat.predict(test[features]) / NFOLDS

    rmse_lgb = rmse(y_val, oof_preds_lgb[val_idx])
    rmse_cat = rmse(y_val, oof_preds_cat[val_idx])
    rmse_ens = rmse(y_val, 0.5 * (oof_preds_lgb[val_idx] + oof_preds_cat[val_idx]))
    print(f"Fold {fold} LGB RMSE: {rmse_lgb:.6f} | CAT RMSE: {rmse_cat:.6f} | ENS RMSE: {rmse_ens:.6f}")

    del dtrain, dvalid, clf, cat, X_tr, X_val, y_tr, y_val
    gc.collect()

print("Overall OOF LGB RMSE:", rmse(train[TARGET], oof_preds_lgb))
print("Overall OOF CAT RMSE:", rmse(train[TARGET], oof_preds_cat))
oof_ensemble = 0.5 * (oof_preds_lgb + oof_preds_cat)
print("Overall OOF Ensemble RMSE:", rmse(train[TARGET], oof_ensemble))

# final blended test predictions
test_preds = 0.5 * (test_preds_lgb + test_preds_cat)

# -------- submission --------
submission = pd.DataFrame({ID_COL: test[ID_COL].astype(int), TARGET: test_preds})
submission[TARGET] = submission[TARGET].round(6)
submission.to_csv(OUTPUT_PATH, index=False)
print(f"Saved submission to: {OUTPUT_PATH}")
print("Elapsed time (s):", time.time() - start)
print(submission.head())


