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
# Diabetes Prediction Challenge - Kaggle Playground S5E12
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# ============================================================
# Load Data
# ============================================================

train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

target = "diagnosed_diabetes"
id_col = "id"

X = train.drop(columns=[target])
y = train[target]
X_test = test.copy()

print("Train shape:", X.shape)
print("Test shape:", X_test.shape)

# ============================================================
# Preprocessing
# ============================================================

# Fill missing values
for c in X.columns:
    if X[c].dtype.kind in "biufc":  # numeric
        med = X[c].median()
        X[c] = X[c].fillna(med)
        X_test[c] = X_test[c].fillna(med)
    else:  # categorical
        X[c] = X[c].astype(str).fillna("NA")
        X_test[c] = X_test[c].astype(str).fillna("NA")

# Factorize categorical features consistently
for c in X.columns:
    if X[c].dtype == object:
        combined = pd.concat([X[c], X_test[c]], axis=0).astype(str)
        codes, uniques = pd.factorize(combined)
        X[c] = codes[:len(X)]
        X_test[c] = codes[len(X):]

# ============================================================
# LightGBM Parameters
# ============================================================

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
    "verbosity": -1
}

# ============================================================
# Stratified K-Fold Training
# ============================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n========== FOLD {fold} ==========")

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    valid_data = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=200),
        ]
    )

    # Predictions
    oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits

# ============================================================
# OOF Score
# ============================================================

print("\nOOF ROC-AUC:", roc_auc_score(y, oof))

# ============================================================
# Create Submission
# ============================================================

submission = sample.copy()
submission[target] = test_preds
submission.to_csv("submission.csv", index=False)

print("\nSaved submission.csv")
submission.head()


