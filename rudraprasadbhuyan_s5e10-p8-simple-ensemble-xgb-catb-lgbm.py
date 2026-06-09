"""
Goal: Understand how to apply the Ensemble in simple way w/o feature engineering.

Author: Rudra Prasad Bhuyan
V1: 26-10-2025 22:53 IST
"""
print("")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb
import catboost as catb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import optuna

import warnings
warnings.filterwarnings('ignore')


sub_path = '/kaggle/input/playground-series-s5e10/sample_submission.csv'
train_path = '/kaggle/input/playground-series-s5e10/train.csv'
test_path = '/kaggle/input/playground-series-s5e10/test.csv'

sub_df = pd.read_csv(sub_path)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


# Data Preparation

# Label Encoding for categorical columns
categorical_columns = ['road_type', 'lighting', 'weather', 'time_of_day']
label_encoder = LabelEncoder()

for col in categorical_columns:
    train_df[col] = label_encoder.fit_transform(train_df[col])
    test_df[col] = label_encoder.transform(test_df[col])

# Binary columns (convert to 0/1)
binary_columns = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for col in binary_columns:
    train_df[col] = train_df[col].astype(int)
    test_df[col] = test_df[col].astype(int)

# Handle missing values for numeric columns and ensure correct types
numeric_columns = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
train_df[numeric_columns] = train_df[numeric_columns].fillna(train_df[numeric_columns].median())
test_df[numeric_columns] = test_df[numeric_columns].fillna(test_df[numeric_columns].median())

# Check data types to ensure everything is numeric (int, float, or bool)
print(train_df.dtypes)  # Should show int, float, or bool for all columns
print(test_df.dtypes)


# Define target and features
target = 'accident_risk'
features = [col for col in train_df.columns if col not in ['id', target]]

X = train_df[features]
y = train_df[target]
X_test = test_df[features]

# Split train data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=51)


# Define Models with Parameters

# LGBM Model
lgb_params = {
    'boosting_type': 'gbdt', 
    'learning_rate': 0.0360269510015689, 
    'subsample': 0.8059018900516028, 
    'colsample_bytree': 0.9625693024050926
}
model_lgbm = lgb.LGBMRegressor(
    **lgb_params,
    random_state=51,
    n_jobs=-1,
    n_estimators=2000
)

# XGB Model
xgb_params = {
    'learning_rate': 0.018095111403323844, 
    'subsample': 0.8849524851971824, 
    'colsample_bytree': 0.9645096790114126
}
model_xgb = xgb.XGBRegressor(
    **xgb_params,
    random_state=51,
    n_jobs=-1,
    enable_categorical=True,
    n_estimators=5000,
    eval_metric='rmse',
    tree_method='hist'
)

# CatBoost Model
catb_params = {
    'subsample': 0.931753361976819,
    'learning_rate': 0.07951639588772055
}
model_catb = catb.CatBoostRegressor(
    **catb_params,
    random_state=51,
    iterations=5000,
    eval_metric='RMSE',
    task_type='CPU',
    verbose=False
)


# Fit the models

# LGBM
model_lgbm.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric='rmse',
)

# XGBoost
model_xgb.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False,
    early_stopping_rounds=100
)

# CatBoost
model_catb.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    verbose=False
)


# Predict with all models
pred_lgbm = model_lgbm.predict(X_test)
pred_xgb = model_xgb.predict(X_test)
pred_catb = model_catb.predict(X_test)

# Ensemble prediction (weighted average)
final_pred = (0.2 * pred_lgbm + 0.6 * pred_xgb + 0.1 * pred_catb)


# Create the final submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'accident_risk': final_pred
})

# Save submission file
submission.to_csv('submission.csv', index=False)
print(submission.head())

