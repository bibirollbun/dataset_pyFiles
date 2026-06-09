"""
S5E10 - Advanced Model Ensemble with K-Fold, Optuna Weight Tuning, and Ridge Meta-Stacking.
Author: Rudra Prasad Bhuyan

Version: 
    - V2 - 27 Oct 2025 08:15 IST
    - V1 - https://www.kaggle.com/code/rudraprasadbhuyan/s5e10-p8-simple-ensemble-xgb-catb-lgbm


Goal:
    â€¢ Train LGBM, XGBoost, and CatBoost models
    â€¢ Generate OOF predictions via StratifiedKFold
    â€¢ Optimize blending weights using Optuna
    â€¢ Stack with Ridge meta-model
    â€¢ Output final submission (0â€“1 clipped)
"""
print("")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import numpy as np
import pandas as pd
import optuna
import warnings
import random
import os

import lightgbm as lgb
import xgboost as xgb
import catboost as catb

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
np.random.seed(51)
random.seed(51)


TRAIN_PATH = "/kaggle/input/playground-series-s5e10/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e10/test.csv"
SUB_PATH = "/kaggle/input/playground-series-s5e10/sample_submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sub_df = pd.read_csv(SUB_PATH)

print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")


categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
binary_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
numeric_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

# Label encode categorical columns
le = LabelEncoder()
for col in categorical_cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# Convert binary columns
for col in binary_cols:
    train_df[col] = train_df[col].astype(int)
    test_df[col] = test_df[col].astype(int)

# Fill numeric missing values
for col in numeric_cols:
    median_val = train_df[col].median()
    train_df[col].fillna(median_val, inplace=True)
    test_df[col].fillna(median_val, inplace=True)

print('Preprocessing Done.')


target = "accident_risk"
features = [c for c in train_df.columns if c not in ["id", target]]

X = train_df[features]
y = train_df[target]
X_test = test_df[features]

print('X, y Ready')


# LightGBM
lgb_params = {
    'boosting_type': 'gbdt', 
    'learning_rate': 0.0360269510015689, 
    'subsample': 0.8059018900516028, 
    'colsample_bytree': 0.9625693024050926,
    'n_estimators': 2000,
    'random_state': 51,
    'n_jobs': -1
}

# XGBoost
xgb_params = {
    'learning_rate': 0.018095111403323844, 
    'subsample': 0.8849524851971824, 
    'colsample_bytree': 0.9645096790114126,
    'n_estimators': 5000,
    'random_state': 51,
    'n_jobs': -1,
    'enable_categorical': True,
    'eval_metric': 'rmse',
    'tree_method': 'hist'
}

# CatBoost
catb_params = {
    'subsample': 0.931753361976819,
    'learning_rate': 0.07951639588772055,
    'iterations': 5000,
    'random_state': 51,
    'eval_metric': 'RMSE',
    'task_type': 'CPU',
    'verbose': False
}

print('Parameters Fixed')


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=51)
y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')

oof_preds = np.zeros((len(X), 3))
test_preds = np.zeros((len(X_test), 3))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_binned), 1):
    print(f"\n{'='*40} \n Fold {fold}\n{'='*40}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train models
    model_lgbm = lgb.LGBMRegressor(**lgb_params)
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_catb = catb.CatBoostRegressor(**catb_params)
    
    model_lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=200, verbose=False)
    model_catb.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    # OOF predictions
    oof_preds[val_idx, 0] = model_lgbm.predict(X_val)
    oof_preds[val_idx, 1] = model_xgb.predict(X_val)
    oof_preds[val_idx, 2] = model_catb.predict(X_val)
    
    # Test predictions (averaged)
    test_preds[:, 0] += model_lgbm.predict(X_test) / kf.n_splits
    test_preds[:, 1] += model_xgb.predict(X_test) / kf.n_splits
    test_preds[:, 2] += model_catb.predict(X_test) / kf.n_splits

fold_rmse = np.sqrt(mean_squared_error(y, oof_preds.mean(axis=1)))
print(f"\nBase Models OOF Mean RMSE: {fold_rmse:.6f}")


def objective(trial):
    w = np.array([
        trial.suggest_float("w_lgbm", 0.0, 1.0),
        trial.suggest_float("w_xgb", 0.0, 1.0),
        trial.suggest_float("w_catb", 0.0, 1.0)
    ])
    w /= np.sum(w)
    blended = np.dot(oof_preds, w)
    rmse = np.sqrt(mean_squared_error(y, blended))  
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40, timeout=300)
best_w = study.best_params
weights = np.array([best_w['w_lgbm'], best_w['w_xgb'], best_w['w_catb']])
weights /= weights.sum()

print(f'\nBest Weights: {best_w} \n')
print(f"\nBest Optuna Weights: {weights}")


oof_blend = np.dot(oof_preds, weights)
test_blend = np.dot(test_preds, weights)

optuna_rmse = np.sqrt(mean_squared_error(y, oof_blend))
print(f"Optuna Weighted Blend RMSE: {optuna_rmse:.6f}")


meta_model = Ridge(alpha=1.0, random_state=51)
meta_model.fit(oof_preds, y)
stack_oof = meta_model.predict(oof_preds)
stack_rmse = np.sqrt(mean_squared_error(y, stack_oof))
print(f"Ridge Stacking RMSE: {stack_rmse:.6f}")

stack_test = meta_model.predict(test_preds)


# Weighted average of Ridge and Optuna blends (tuned ratio)
final_pred = 0.5 * stack_test + 0.5 * test_blend
final_pred = np.clip(final_pred, 0, 1)

sub_df["accident_risk"] = final_pred
sub_df.to_csv("/kaggle/working/submission.csv", index=False)

display(pd.read_csv('submission.csv').head(12))
print(f"\n\n Optuna RMSE: {optuna_rmse:.6f} | Ridge Stack RMSE: {stack_rmse:.6f}")

