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


# Cell 1 — setup & imports
import os
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

RANDOM_SEED = 42
N_FOLDS = 5
TARGET = "loan_paid_back"
ID_COL = "id"

print("Python", sys.version if 'sys' in globals() else "ok")
print("LightGBM version:", lgb.__version__)



# Cell 2 — load data (correct for this competition)
DATA_DIR = "/kaggle/input/playground-series-s5e11"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)
print("Sample submission:", sample_submission.head())
train.head()



# Cell 3 — quick EDA
print("Target distribution (train):")
print(train[TARGET].value_counts(dropna=False))
print("\nMissing values (train):")
print(train.isnull().sum().sort_values(ascending=False).head(20))

print("\nSample types:")
print(train.dtypes.value_counts())

# basic numeric summary
display(train.describe().T)



# Cell 4 — preprocessing
features = [c for c in train.columns if c not in [ID_COL, TARGET]]
cat_cols = [c for c in features if train[c].dtype == "object" or train[c].dtype.name == "category"]
num_cols = [c for c in features if c not in cat_cols]

print("Num features:", len(features))
print("Categorical cols:", len(cat_cols))
print("Numeric cols:", len(num_cols))

# Fill numeric missing with median (simple)
for c in num_cols:
    if train[c].isnull().any() or test[c].isnull().any():
        med = train[c].median()
        train[c].fillna(med, inplace=True)
        test[c].fillna(med, inplace=True)

# Fill categorical missing with special token
for c in cat_cols:
    train[c].fillna("__MISSING__", inplace=True)
    test[c].fillna("__MISSING__", inplace=True)

# Label encode categoricals (LightGBM can handle integers)
for c in cat_cols:
    le = LabelEncoder()
    le.fit(list(train[c].astype(str).values) + list(test[c].astype(str).values))
    train[c] = le.transform(train[c].astype(str))
    test[c]  = le.transform(test[c].astype(str))

# refresh feature list in case new features added later
features = [c for c in train.columns if c not in [ID_COL, TARGET]]
print("Features count after preprocessing:", len(features))



# Cell 5 — simple feature engineering
# Frequency encoding for categorical columns
for c in cat_cols:
    freq = train[c].value_counts()
    train[f"{c}_freq"] = train[c].map(freq).fillna(0).astype(np.int32)
    test[f"{c}_freq"]  = test[c].map(freq).fillna(0).astype(np.int32)

# missing_count (useful even if we filled above — gives signal of columns originally missing)
train["missing_count"] = (train[features] == "__MISSING__").sum(axis=1)
test["missing_count"]  = (test[features] == "__MISSING__").sum(axis=1)

# update features list
features = [c for c in train.columns if c not in [ID_COL, TARGET]]
print("Features after FE:", len(features))



# Cell 6 — training (OOF + test predictions) — FIXED FOR LGB VERSION
from lightgbm import early_stopping, log_evaluation

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "n_jobs": -1,
    "seed": RANDOM_SEED,
    "verbosity": -1
}

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
feature_importance_df = []

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

for fold, (tr_idx, val_idx) in enumerate(skf.split(train, train[TARGET])):
    print(f"\n>>> Fold {fold+1}/{N_FOLDS}")
    X_tr, X_val = train.iloc[tr_idx][features], train.iloc[val_idx][features]
    y_tr, y_val = train.iloc[tr_idx][TARGET], train.iloc[val_idx][TARGET]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_val, label=y_val)

    clf = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dtrain, dvalid],
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(200)
        ]
    )

    oof_preds[val_idx] = clf.predict(X_val, num_iteration=clf.best_iteration)
    test_preds += clf.predict(test[features], num_iteration=clf.best_iteration) / N_FOLDS

    # feature importance
    imp_df = pd.DataFrame({
        "feature": features,
        "importance": clf.feature_importance(importance_type="gain"),
        "fold": fold+1
    })
    feature_importance_df.append(imp_df)

    del clf, dtrain, dvalid, X_tr, X_val
    gc.collect()

# OOF AUC
oof_auc = roc_auc_score(train[TARGET], oof_preds)
print("\nOOF AUC:", oof_auc)



# Cell 7 — Save outputs
# Save OOF predictions for stacking and analysis
train_oof = train[[ID_COL, TARGET]].copy()
train_oof["oof_pred"] = oof_preds
train_oof.to_csv("train_oof.csv", index=False)
print("Saved train_oof.csv")

# Create final submission (probabilities)
submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_preds})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv — show head:")
display(submission.head())


