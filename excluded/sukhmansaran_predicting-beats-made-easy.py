import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import cudf
# import cupy as cp


# ignoring all the warnings
import warnings
import os

warnings.filterwarnings("ignore")

os.environ["PYTHONWARNINGS"] = "ignore"


raw_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
raw2_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


raw_df.head()


raw_df.describe(include = 'all')


raw_df.isna().sum()


raw_df.duplicated().sum()


raw_df.info()


raw2_df.head()


raw2_df.describe(include = 'all')


raw2_df.isna().sum()


raw2_df.duplicated().sum()


raw2_df.info()


train_df = raw_df.copy()
test_df = raw2_df.copy()


train_df = train_df.drop(columns = ['id'])
test_df = test_df.drop(columns = ['id'])


train_corr_matrix = train_df.corr()
plt.figure(figsize = (12, 8))
sns.heatmap(train_corr_matrix, fmt = ".2f", cmap = "coolwarm")
plt.title('Correlation Matrix - Train Dataset')
plt.show()


train_df_xg = train_df
test_df_xg = test_df

non_numeric_columns_train = train_df_xg.select_dtypes(include = ['object']).columns
non_numeric_columns_test = test_df_xg.select_dtypes(include = ['object']).columns

for col in non_numeric_columns_train:
    train_df_xg[col] = train_df_xg[col].astype('category')
    print(f"Converted {col}: {len(train_df_xg[col].cat.categories)} categories")

for col in non_numeric_columns_test:
    test_df_xg[col] = test_df_xg[col].astype('category')
    print(f"Converted {col}: {len(test_df_xg[col].cat.categories)} categories")


# shuffling the data
from sklearn.utils import shuffle
train_df = shuffle(train_df).reset_index(drop=True)

x_train = train_df_xg.drop(columns='BeatsPerMinute')
y_train = train_df_xg['BeatsPerMinute']

# splitting the train dataset into train/validation
from sklearn.model_selection import train_test_split, GridSearchCV
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)


import xgboost as xgb
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, KFold
from sklearn import metrics

# parameter grid for XGBoost regression
param_distributions = {
    'n_estimators': [300, 500, 700],
    'max_depth': [6, 8, 10, 12],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# XGBoost for regression
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    tree_method='hist',
    device='cuda',
    enable_categorical=True
)

# KFold for regression CV
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# Halving Random Search
random_search_xgb = HalvingRandomSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    min_resources=10000,
    scoring='neg_root_mean_squared_error',
    cv=cv_strategy,
    verbose=2,
    n_jobs=-1
)

print("Starting XGBoost HalvingRandomSearchCV...")
random_search_xgb.fit(x_train, y_train)

# getting best model and parameters
best_model_xgb = random_search_xgb.best_estimator_
best_params_xgb = random_search_xgb.best_params_
best_cv_score_xgb = random_search_xgb.best_score_

print(f"\n=== XGBOOST RESULTS ===")
print(f"Best CV RMSE: {-best_cv_score_xgb:.4f}")
print(f"Best Parameters: {best_params_xgb}")

# predictions
predictions_xgb = best_model_xgb.predict(x_val)



from sklearn import metrics

rmse_xgb = metrics.mean_squared_error(y_val, predictions_xgb, squared=False)
mae_xgb = metrics.mean_absolute_error(y_val, predictions_xgb)
r2_xgb = metrics.r2_score(y_val, predictions_xgb)

print(f"RMSE of XGBoost: {rmse_xgb:.4f}")
print(f"MAE of XGBoost: {mae_xgb:.4f}")
print(f"R² of XGBoost: {r2_xgb:.4f}")


import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

# parameter grid for LightGBM (regression)
param_distributions = {
    'n_estimators': [300, 500, 700, 1000],
    'max_depth': [10, 15, -1],
    'learning_rate': [0.05, 0.03, 0.01],
    'num_leaves': [127, 255, 511],
    'min_gain_to_split': [0.0],
    'min_sum_hessian_in_leaf': [0.001, 0.01]
}

# LightGBM for regression
lgb_model = lgb.LGBMRegressor(
    boosting_type='gbdt',
    objective='regression',
    metric='rmse',
    device='gpu'
)

# KFold for regression CV
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# Halving Random Search for regression
random_search_lgb = HalvingRandomSearchCV(
    estimator=lgb_model,
    param_distributions=param_distributions,
    min_resources=30000,
    scoring='neg_root_mean_squared_error',
    cv=cv_strategy,
    n_jobs=-1,
    verbose=0,
)

print("Starting LightGBM HalvingRandomSearchCV...")

random_search_lgb.fit(x_train, y_train)

# getting best model and parameters
best_model_lgb = random_search_lgb.best_estimator_
best_params_lgb = random_search_lgb.best_params_
best_cv_score_lgb = random_search_lgb.best_score_

print(f"\n=== BEST MODEL RESULTS ===")
print(f"Best CV RMSE: {-best_cv_score_lgb:.4f}")
print(f"Best Parameters: {best_params_lgb}")

# predictions
predictions_lgb = best_model_lgb.predict(x_val)



from sklearn import metrics

# Regression evaluation metrics
rmse_lgb = metrics.mean_squared_error(y_val, predictions_lgb, squared=False)
mae_lgb = metrics.mean_absolute_error(y_val, predictions_lgb)
r2_lgb = metrics.r2_score(y_val, predictions_lgb)

print(f"RMSE of LightGBM: {rmse_lgb:.4f}")
print(f"MAE of LightGBM: {mae_lgb:.4f}")
print(f"R² of LightGBM: {r2_lgb:.4f}")


from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

# parameter grid for CatBoost
param_distributions = {
    'iterations': [100, 200],
    'depth': [6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.15],
    'l2_leaf_reg': [1, 3, 5],
    'border_count': [128, 254],
    'bagging_temperature': [0]
}

# CatBoost for regression
cat_model = CatBoostRegressor(
    objective='RMSE',
    eval_metric='RMSE',
    task_type='GPU',
    verbose=0
)

# KFold for regression CV
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# Halving Random Search for regression
random_search_cat = HalvingRandomSearchCV(
    estimator=cat_model,
    param_distributions=param_distributions,
    min_resources=5000,
    max_resources=600000,
    scoring='neg_root_mean_squared_error',
    cv=cv_strategy,
    verbose=1,
)

print("Starting CatBoost training...")

random_search_cat.fit(x_train, y_train)

# getting best model and parameters
best_model_cat = random_search_cat.best_estimator_
best_params_cat = random_search_cat.best_params_
best_cv_score_cat = random_search_cat.best_score_

print(f"\n=== CATBOOST RESULTS ===")
print(f"Best CV RMSE: {-best_cv_score_cat:.4f}")
print(f"Best Parameters: {best_params_cat}")

# predictions
predictions_cat = best_model_cat.predict(x_val)



from sklearn import metrics

# Regression evaluation metrics
rmse_cat = metrics.mean_squared_error(y_val, predictions_cat, squared=False)
mae_cat = metrics.mean_absolute_error(y_val, predictions_cat)
r2_cat = metrics.r2_score(y_val, predictions_cat)

print(f"RMSE of CatBoost: {rmse_cat:.4f}")
print(f"MAE of CatBoost: {mae_cat:.4f}")
print(f"R² of CatBoost: {r2_cat:.4f}")


x_test = test_df_xg

predictions = best_model_xgb.predict(x_test).tolist()
ids = raw2_df['id'].values

pred_df = pd.DataFrame()
pred_df['id'] = ids


pred_df['BeatsPerMinute'] = predictions
pred_df.head()


pred_df.to_csv('predicted_xgb.csv', index = False)


predictions = best_model_lgb.predict(x_test).tolist()
ids = raw2_df['id'].values

pred_df['BeatsPerMinute'] = predictions
print(pred_df.head())


pred_df.to_csv('predicted_lgb.csv', index = False)


predictions = best_model_cat.predict(x_test).tolist()
ids = raw2_df['id'].values

pred_df['BeatsPerMinute'] = predictions
print(pred_df.head())


pred_df.to_csv('predicted_cat.csv', index = False)

