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


!pip install catboost lightgbm xgboost optuna --quiet


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


# File: calories_prediction_optimized.py

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_df = train
test_df = test

TARGET = 'Calories'
ID = 'id'

# Combine for preprocessing
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df[TARGET] = np.nan

data = pd.concat([train_df, test_df], ignore_index=True)

# Feature engineering
data['BMI'] = data['Weight'] / ((data['Height'] / 100) ** 2)
data['HR_Duration'] = data['Heart_Rate'] * data['Duration']
data['Age_Duration'] = data['Age'] * data['Duration']
data['BMI_HR'] = data['BMI'] * data['Heart_Rate']

# Encode 'Sex'
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Feature columns (top 10 by importance only)
features = [
    'Sex', 'Age', 'Weight', 'Heart_Rate', 'BMI',
    'Age_Duration', 'BMI_HR', 'HR_Duration', 'Height', 'Body_Temp'
]

# Scaling
scaler = RobustScaler()
data[features] = scaler.fit_transform(data[features])

# Split back
train_data = data[data['is_train'] == 1].drop(columns=['is_train'])
test_data = data[data['is_train'] == 0].drop(columns=['is_train', TARGET])

X = train_data[features]
y = np.log1p(train_data[TARGET])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_lgb = np.zeros(len(X))
oof_preds_xgb = np.zeros(len(X))
oof_preds_cat = np.zeros(len(X))
test_preds_lgb = np.zeros(len(test_data))
test_preds_xgb = np.zeros(len(test_data))
test_preds_cat = np.zeros(len(test_data))
feature_importance_df = pd.DataFrame()

# LightGBM parameters
params_lgb = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.015,
    'verbosity': -1,
    'seed': 42,
    'num_leaves': 64,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0
}

# XGBoost parameters
params_xgb = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.015,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 1000,
    'random_state': 42
}

# CatBoost parameters
params_cat = {
    'iterations': 1000,
    'learning_rate': 0.015,
    'depth': 6,
    'loss_function': 'RMSE',
    'verbose': False,
    'random_seed': 42
}

for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # LightGBM
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid)
    model_lgb = lgb.train(
        params_lgb,
        train_set,
        num_boost_round=10000,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=0)
        ]
    )
    oof_preds_lgb[valid_idx] = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
    test_preds_lgb += model_lgb.predict(test_data[features], num_iteration=model_lgb.best_iteration) / kf.n_splits

    # XGBoost
    model_xgb = xgb.XGBRegressor(**params_xgb)
    model_xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=100, verbose=False)
    oof_preds_xgb[valid_idx] = model_xgb.predict(X_valid)
    test_preds_xgb += model_xgb.predict(test_data[features]) / kf.n_splits

    # CatBoost
    model_cat = CatBoostRegressor(**params_cat)
    model_cat.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100)
    oof_preds_cat[valid_idx] = model_cat.predict(X_valid)
    test_preds_cat += model_cat.predict(test_data[features]) / kf.n_splits

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = model_lgb.feature_importance()
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

# Blend predictions
oof_preds = 0.4 * oof_preds_lgb + 0.3 * oof_preds_xgb + 0.3 * oof_preds_cat
test_preds = 0.4 * test_preds_lgb + 0.3 * test_preds_xgb + 0.3 * test_preds_cat

rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof_preds)))
print(f'OOF RMSLE: {rmsle:.5f}')

avg_importance = feature_importance_df.groupby("feature")["importance"].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x=avg_importance.values, y=avg_importance.index)
plt.title("Average Feature Importance (Pruned)")
plt.tight_layout()
plt.show()

submission = pd.DataFrame({
    ID: test_data[ID].values,
    TARGET: np.expm1(test_preds)
})

submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")


