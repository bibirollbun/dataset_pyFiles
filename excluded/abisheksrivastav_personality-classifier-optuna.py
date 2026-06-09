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


# Personality Classification - XGBoost (Optuna Tuned) + LightGBM + CatBoost (GPU + StratifiedKFold Ensemble + cuDF + Pseudo Labeling)

import pandas as pd
import numpy as np
import cudf
import cupy as cp
from cuml.preprocessing import LabelEncoder as cuLabelEncoder
from cuml.model_selection import train_test_split
from cuml import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb
import optuna

import warnings
warnings.filterwarnings("ignore")

# Load data with cuDF for GPU processing
train = cudf.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = cudf.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

# Map target to 0/1
train['Personality'] = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})

# Save test IDs
test_ids = test['id'].to_pandas()

# Pseudo-labeling: add test to train with soft targets from XGB ensemble
combined = cudf.concat([train.assign(is_test=0), test.assign(Personality=np.nan, is_test=1)], axis=0)

# Fill missing values
combined = combined.fillna(combined.mode().iloc[0])

# Convert all categorical features to strings (treat as categories)
exclude_cols = ['id', 'Personality', 'is_test']
features = [col for col in combined.columns if col not in exclude_cols]

# Convert all to strings
for col in features:
    combined[col] = combined[col].astype(str)

# Label encode using cuML
for col in features:
    le = cuLabelEncoder()
    combined[col] = le.fit_transform(combined[col])

# Split back
train = combined[combined['is_test'] == 0].drop(columns='is_test')
test = combined[combined['is_test'] == 1].drop(columns=['is_test', 'Personality'])
X = train.drop(columns=['id', 'Personality'])
y = train['Personality'].astype(np.int32)
X_test = test.drop(columns=['id'])

# Optuna tuning for XGBoost
X_np, y_np = X.to_pandas(), y.to_pandas()

def objective(trial):
    params = {
        "objective": "binary:logistic",
        "tree_method": "gpu_hist",
        "eval_metric": "logloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "lambda": trial.suggest_float("lambda", 0, 2.0),
        "alpha": trial.suggest_float("alpha", 0, 2.0)
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X_np, y_np):
        dtrain = xgb.DMatrix(X_np.iloc[train_idx], label=y_np.iloc[train_idx])
        dval = xgb.DMatrix(X_np.iloc[val_idx], label=y_np.iloc[val_idx])
        model = xgb.train(params, dtrain, num_boost_round=300,
                          evals=[(dval, "eval")],
                          early_stopping_rounds=20, verbose_eval=False)
        preds = model.predict(dval)
        preds_binary = (preds > 0.5).astype(int)
        scores.append(accuracy_score(y_np.iloc[val_idx], preds_binary))
    return np.mean(scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
params_xgb = study.best_params
params_xgb.update({"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "gpu_hist"})

# Final models with pseudo-labeling
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np)):
    X_train, X_val = X_np.iloc[train_idx], X_np.iloc[val_idx]
    y_train, y_val = y_np.iloc[train_idx], y_np.iloc[val_idx]

    # XGB
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test.to_pandas())
    model_xgb = xgb.train(params_xgb, dtrain, num_boost_round=500,
                          evals=[(dval, "eval")], early_stopping_rounds=20, verbose_eval=False)
    pred_xgb_val = model_xgb.predict(dval)
    pred_xgb_test = model_xgb.predict(dtest)

    # LGB
    model_lgb = lgb.LGBMClassifier(device="gpu", n_estimators=500)
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20, verbose=False)])
    pred_lgb_val = model_lgb.predict_proba(X_val)[:, 1]
    pred_lgb_test = model_lgb.predict_proba(X_test.to_pandas())[:, 1]

    # CatBoost
    cat_model = CatBoostClassifier(task_type="GPU", iterations=500, verbose=0, auto_class_weights="Balanced")
    train_pool = Pool(X_train, label=y_train)
    val_pool = Pool(X_val, label=y_val)
    cat_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=20)
    pred_cat_val = cat_model.predict_proba(X_val)[:, 1]
    pred_cat_test = cat_model.predict_proba(X_test.to_pandas())[:, 1]

    # Weighted ensemble
    acc_xgb = accuracy_score(y_val, (pred_xgb_val > 0.5).astype(int))
    acc_lgb = accuracy_score(y_val, (pred_lgb_val > 0.5).astype(int))
    acc_cat = accuracy_score(y_val, (pred_cat_val > 0.5).astype(int))
    total = acc_xgb + acc_lgb + acc_cat
    w_xgb, w_lgb, w_cat = acc_xgb / total, acc_lgb / total, acc_cat / total

    oof_preds[val_idx] = pred_xgb_val * w_xgb + pred_lgb_val * w_lgb + pred_cat_val * w_cat
    test_preds += (pred_xgb_test * w_xgb + pred_lgb_test * w_lgb + pred_cat_test * w_cat) / skf.n_splits

# Threshold optimization
best_thresh = 0.5
best_acc = 0
for t in np.arange(0.4, 0.61, 0.01):
    acc = accuracy_score(y_np, (oof_preds > t).astype(int))
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print(f"Best Threshold: {best_thresh}, Accuracy: {best_acc:.5f}")

# Submission
final_preds = (test_preds > best_thresh).astype(int)
sample_submission["Personality"] = ["Extrovert" if p == 1 else "Introvert" for p in final_preds]
sample_submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")


