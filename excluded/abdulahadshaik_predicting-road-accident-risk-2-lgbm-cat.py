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


# ====================================================
# Road Accident Risk Prediction - LightGBM vs CatBoost
# ====================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from catboost import CatBoostRegressor, Pool

# ====================================================
# 1. Load Data
# ====================================================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

TARGET = "accident_risk"
ID = "id"

X = train.drop(columns=[TARGET])
y = train[TARGET]
X_test = test.copy()

# ====================================================
# 2. Preprocessing
# ====================================================
# Identify categorical and numerical features
categorical_features = ["road_type", "lighting", "weather", "time_of_day"]
numerical_features = [col for col in X.columns if col not in categorical_features + [ID]]

# One-hot encode categoricals for LightGBM
X_lgb = pd.get_dummies(X, columns=categorical_features)
X_test_lgb = pd.get_dummies(X_test, columns=categorical_features)

# Align columns between train/test
X_lgb, X_test_lgb = X_lgb.align(X_test_lgb, join="left", axis=1)
X_test_lgb = X_test_lgb.fillna(0)

# ====================================================
# 3. Cross-validation Setup
# ====================================================
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

# ====================================================
# 4. LightGBM Training
# ====================================================
lgb_oof = np.zeros(len(X))
lgb_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.01788944408051965,
    "num_leaves": 100,
    "feature_fraction": 0.7914285733151603,
    "bagging_fraction": 0.7815983258073242,
    "bagging_freq": 2,
    "min_data_in_leaf": 81,
    "random_state": 42,
    "device": "gpu" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu"
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_lgb, y)):
    print(f"=== LightGBM Fold {fold+1} ===")
    X_train, X_val = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=1000,
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(100)
        ]
    )
    
    lgb_oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    lgb_preds += model.predict(X_test_lgb, num_iteration=model.best_iteration) / FOLDS

lgb_score = rmse(y, lgb_oof)
print(f"LightGBM CV RMSE: {lgb_score:.5f}")

# ====================================================
# 5. CatBoost Training
# ====================================================
cat_oof = np.zeros(len(X))
cat_preds = np.zeros(len(X_test))

cat_params = {
    "iterations": 1000,
    "learning_rate": 0.0222439870839864,
    "depth": 9,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "l2_leaf_reg": 0.19213176122762995,
    "random_seed": 42,
    "task_type": "GPU" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "CPU",
    "early_stopping_rounds": 100,
    "verbose": 100
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"=== CatBoost Fold {fold+1} ===")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    val_pool = Pool(X_val, y_val, cat_features=categorical_features)
    test_pool = Pool(X_test, cat_features=categorical_features)
    
    model = CatBoostRegressor(**cat_params)
    model.fit(train_pool, eval_set=val_pool)
    
    cat_oof[val_idx] = model.predict(X_val)
    cat_preds += model.predict(test_pool) / FOLDS

cat_score = rmse(y, cat_oof)
print(f"CatBoost CV RMSE: {cat_score:.5f}")

# ====================================================
# 6. Save Submissions
# ====================================================
# LightGBM submission
sub_lgb = sample_submission.copy()
sub_lgb[TARGET] = lgb_preds.clip(0, 1)  # ensure [0,1] range
sub_lgb.to_csv("submission_lightgbm.csv", index=False)

# CatBoost submission
sub_cat = sample_submission.copy()
sub_cat[TARGET] = cat_preds.clip(0, 1)
sub_cat.to_csv("submission_catboost.csv", index=False)

print("Submissions saved: submission_lightgbm.csv & submission_catboost.csv")





import pandas as pd
import numpy as np

LGB_SUBMISSION_FILE = "/kaggle/working/submission_lightgbm.csv"
CAT_SUBMISSION_FILE = "/kaggle/working/submission_catboost.csv"
FINAL_SUBMISSION_FILE = "submission.csv"
TARGET = "accident_risk"

print("--- Starting Ensemble Process ---")

df_lgb = pd.read_csv(LGB_SUBMISSION_FILE)
df_cat = pd.read_csv(CAT_SUBMISSION_FILE)

print(f"Loaded LightGBM predictions: {len(df_lgb)} rows.")
print(f"Loaded CatBoost predictions: {len(df_cat)} rows.")

final_predictions = (df_lgb[TARGET] + df_cat[TARGET]) / 2

final_predictions = final_predictions.clip(0, 1)

print(f"\nCalculated mean ensemble predictions (50/50 average).")
print(f"Final prediction range: [{final_predictions.min():.5f}, {final_predictions.max():.5f}]")

df_final = pd.DataFrame({
    'id': df_lgb['id'],
    TARGET: final_predictions
})

df_final.to_csv(FINAL_SUBMISSION_FILE, index=False)

print(f"\nSuccessfully created the final ensembled submission file: {FINAL_SUBMISSION_FILE}")
print("--- Ensemble Process Complete ---")

