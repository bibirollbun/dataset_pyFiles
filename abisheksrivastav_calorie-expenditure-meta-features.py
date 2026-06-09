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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import xgboost as xgb
import catboost as cb
import optuna
from itertools import product

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

# Feature Engineering
def add_features(df):
    df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
    df['Temp_Diff'] = df['Body_Temp'] - 36.5
    df['Weight_Duration'] = df['Weight'] * df['Duration']
    df['HR_Duration'] = df['Heart_Rate'] * df['Duration']
    df['Age_Duration'] = df['Age'] * df['Duration']
    df['Interaction'] = df['Heart_Rate'] * df['Age'] * df['Duration']
    return df

train = add_features(train)
test = add_features(test)

# One-hot encoding
train = pd.get_dummies(train, columns=["Sex"], drop_first=True)
test = pd.get_dummies(test, columns=["Sex"], drop_first=True)

# Align columns
test = test[train.drop(columns=["Calories"]).columns]

# Target transform
y = np.log1p(train["Calories"])
X = train.drop(columns=["Calories"])
X_test = test.copy()

# Quantile normalization
qt = QuantileTransformer(output_distribution='normal')
X = qt.fit_transform(X)
X_test = qt.transform(X_test)

# KFold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_oof = np.zeros(len(X))
xgb_oof = np.zeros(len(X))
cat_oof = np.zeros(len(X))
lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}")
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[early_stopping(stopping_rounds=20),log_evaluation(0)])
    lgb_oof[val_idx] = lgb_model.predict(X_val)
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits

    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method='gpu_hist', predictor='gpu_predictor', verbosity=0)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    xgb_oof[val_idx] = xgb_model.predict(X_val)
    xgb_preds += xgb_model.predict(X_test) / kf.n_splits

    # CatBoost
    cat_model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, task_type='GPU', early_stopping_rounds=100, verbose=0)
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    cat_oof[val_idx] = cat_model.predict(X_val)
    cat_preds += cat_model.predict(X_test) / kf.n_splits

# Meta-features
meta_X = np.vstack([lgb_oof, xgb_oof, cat_oof]).T
meta_test = np.vstack([lgb_preds, xgb_preds, cat_preds]).T

# Optuna tuning for ensemble weights
def objective(trial):
    w1 = trial.suggest_float("w1", 0.0, 1.0)
    w2 = trial.suggest_float("w2", 0.0, 1.0 - w1)
    w3 = 1.0 - w1 - w2
    blended = w1 * lgb_oof + w2 * xgb_oof + w3 * cat_oof
    score = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(blended)))
    return score

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
weights = study.best_params
print(f"Best RMSLE: {study.best_value:.5f} | Weights: {weights}")

# Final predictions
final_test_preds = weights['w1'] * lgb_preds + weights['w2'] * xgb_preds + (1 - weights['w1'] - weights['w2']) * cat_preds

# Submission
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
submission["Calories"] = np.expm1(final_test_preds).clip(0, 2000)
submission.to_csv("submission.csv", index=False)
print("Submission saved.")



