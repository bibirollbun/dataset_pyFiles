"""
Goal: This code applies model blending with LightGBM, XGBoost, and CatBoost using 
      a weighted average approach. It also uses Optuna to optimize the blending weights.

Author: Rudra Prasad Bhuyan
V1: 26-10-2025 22:44 IST
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
from sklearn.linear_model import LinearRegression

import optuna

import warnings
warnings.filterwarnings('ignore')


sub_path = '/kaggle/input/playground-series-s5e10/sample_submission.csv'
train_path = '/kaggle/input/playground-series-s5e10/train.csv'
test_path = '/kaggle/input/playground-series-s5e10/test.csv'

sub_df = pd.read_csv(sub_path)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


categorical_columns = ['road_type', 'lighting', 'weather', 'time_of_day']
label_encoder = LabelEncoder()

for col in categorical_columns:
    train_df[col] = label_encoder.fit_transform(train_df[col])
    test_df[col] = label_encoder.transform(test_df[col])

binary_columns = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for col in binary_columns:
    train_df[col] = train_df[col].astype(int)
    test_df[col] = test_df[col].astype(int)

numeric_columns = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
train_df[numeric_columns] = train_df[numeric_columns].fillna(train_df[numeric_columns].median())
test_df[numeric_columns] = test_df[numeric_columns].fillna(test_df[numeric_columns].median())

display(train_df.dtypes)  
print(test_df.dtypes)


target = 'accident_risk'
features = [col for col in train_df.columns if col not in ['id', target]]

X = train_df[features]
y = train_df[target]
X_test = test_df[features]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=51)


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


oof_lgbm = model_lgbm.predict(X_valid)
oof_xgb = model_xgb.predict(X_valid)
oof_catb = model_catb.predict(X_valid)


def objective_blending(trial):
    wt_lgbm = trial.suggest_uniform("weight_lgbm", 0, 1)
    wt_xgb = trial.suggest_uniform("weight_xgb", 0, 1)
    wt_catb = trial.suggest_uniform("weight_catb", 0, 1)

    total_weight = wt_lgbm + wt_xgb + wt_catb
    wt_lgbm /= total_weight
    wt_xgb /= total_weight
    wt_catb /= total_weight

    final_pred = (wt_lgbm*oof_lgbm + wt_xgb*oof_xgb + wt_catb*oof_catb)
    
    rmse = np.sqrt(mean_squared_error(y_valid, final_pred))
    return rmse


study_blend = optuna.create_study(direction='minimize')
study_blend.optimize(objective_blending, n_trials=50)


best_weights = study_blend.best_params
best_weights


weight_lgbm = best_weights['weight_lgbm']
weight_xgb = best_weights['weight_xgb']
weight_catb = best_weights['weight_catb']


final_pred = (weight_lgbm * oof_lgbm + weight_xgb * oof_xgb + weight_catb * oof_catb)


pred_lgbm_test = model_lgbm.predict(X_test)
pred_xgb_test = model_xgb.predict(X_test)
pred_catb_test = model_catb.predict(X_test)


final_pred_test = (weight_lgbm * pred_lgbm_test + weight_xgb * pred_xgb_test + weight_catb * pred_catb_test)


sub_df['accident_risk'] = final_pred_test
sub_df.to_csv('/kaggle/working/blended_submission.csv', index=False)
pd.read_csv('blended_submission.csv')


# Stack the OOF predictions as features for the meta-model
stacked_train = np.column_stack([oof_lgbm, oof_xgb, oof_catb])


# Train the meta-model (Linear Regression)
meta_model = LinearRegression()
meta_model.fit(stacked_train, y_valid)


# Make predictions for the test set
pred_lgbm_test = model_lgbm.predict(X_test)
pred_xgb_test = model_xgb.predict(X_test)
pred_catb_test = model_catb.predict(X_test)


stacked_test = np.column_stack([pred_lgbm_test, pred_xgb_test, pred_catb_test])


final_meta_pred = meta_model.predict(stacked_test)


sub_df['accident_risk'] = final_meta_pred
sub_df.to_csv('/kaggle/working/submission.csv', index=False)
pd.read_csv('submission.csv')

