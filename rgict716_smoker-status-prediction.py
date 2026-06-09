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


# ================== Imports ==================
import os
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# ================== Load Data ==================
train = pd.read_csv("/kaggle/input/playground-series-s3e24/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s3e24/test.csv")

# Save test IDs
test_ids = test["id"]

# ================== Combine Train & Test for Feature Engineering ==================
train["is_train"] = 1
test["is_train"] = 0
test["smoking"] = -1  # Dummy target for concatenation

full_df = pd.concat([train, test], axis=0)

# ================== Feature Engineering ==================
full_df['dental_caries_sq'] = full_df['dental caries'] ** 2
full_df['weight(kg)_sq'] = full_df['weight(kg)'] ** 2
full_df['weightxheight'] = full_df['weight(kg)'] * full_df['height(cm)']
full_df['ALT_sq'] = full_df['ALT'] ** 2

# ================== Re-Split Data ==================
train_df = full_df[full_df["is_train"] == 1].drop(["id", "is_train"], axis=1)
test_df = full_df[full_df["is_train"] == 0].drop(["id", "is_train", "smoking"], axis=1)

X = train_df.drop("smoking", axis=1)
y = train_df["smoking"]

# ================== Optuna Objective Function ==================
def objective(trial):
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.2),
        'random_state': 43,
        'n_jobs': -1,
    }

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, valid_idx in kf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100)]
        )

        preds = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, preds)
        aucs.append(auc)

    return np.mean(aucs)

# ========== Run Optuna Optimization ==========
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# Best hyperparameters
best_params = study.best_params
print("Best parameters:", best_params)

# ================== Train Final Model with Best Params ==================
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test_df))
auc_scores = []

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**best_params, random_state=fold, n_jobs=-1)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100)]
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(test_df)[:, 1] / kf.n_splits

    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    auc_scores.append(fold_auc)
    print(f"Fold {fold + 1} AUC: {fold_auc:.5f}")

print(f"\nMean AUC: {np.mean(auc_scores):.5f}")

# ================== Save Submission ==================
submission = pd.DataFrame({
    "id": test_ids,
    "smoking": test_preds
})

submission.to_csv("smoking_prediction_submission_optuna.csv", index=False)
print("Submission file saved successfully!")


