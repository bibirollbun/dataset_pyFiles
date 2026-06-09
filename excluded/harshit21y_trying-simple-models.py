# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the" read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression


train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

print('done read')


train = train.replace([np.inf, -np.inf], np.nan)
test = test.replace([np.inf, -np.inf], np.nan)


# Impute missing values with median for numerical columns
for col in train.select_dtypes(include=[np.number]).columns:
    train[col] = train[col].fillna(train[col].median())
    if col in test.columns:
        test[col] = test[col].fillna(train[col].median())


# Drop columns with all NaNs and align test set
train = train.dropna(axis=1, how='all')
test = test[train.drop(columns=["label"]).columns]

# Split features and target
X_all = train.drop(columns=["label"])
y = train["label"]



rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=5,
    n_jobs=-1,
    random_state=42
)

# Fit the model
rf.fit(X_all.sample(frac=0.2, random_state=42), y.sample(frac=0.2, random_state=42))
# Get feature importances
importances = pd.Series(rf.feature_importances_, index=X_all.columns)

# Sort and get top 200 features
top_features = importances.sort_values(ascending=False).head(100).index



# Final feature set
X = X_all[top_features]
X_test = test[top_features]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

print('Done scaling')


# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# #decision tree

# model = DecisionTreeRegressor(max_depth=9, random_state=42)
# model.fit(X_train, y_train)


# #evaluating decision tree

# y_train_pred = model.predict(X_train)
# y_val_pred = model.predict(X_val)

# # Train metrics
# train_mse = mean_squared_error(y_train, y_train_pred)
# train_corr, _ = pearsonr(y_train, y_train_pred)

# # Val metrics
# val_mse = mean_squared_error(y_val, y_val_pred)
# val_corr, _ = pearsonr(y_val, y_val_pred)

# print(f"Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")
# print(f"Train Pearson: {train_corr:.4f}, Val Pearson: {val_corr:.4f}")


# from sklearn.ensemble import RandomForestRegressor

# rf_model = RandomForestRegressor(
#     n_estimators=200,
#     max_depth=7,
#     min_samples_leaf=30,
#     max_features='sqrt',
#     n_jobs=-1,
#     random_state=42
# )

# rf_model.fit(X_train, y_train)


# #random forest evaluate

# y_train_pred = rf_model.predict(X_train)
# y_val_pred = rf_model.predict(X_val)

# # Train metrics
# train_mse = mean_squared_error(y_train, y_train_pred)
# train_corr, _ = pearsonr(y_train, y_train_pred)

# # Val metrics
# val_mse = mean_squared_error(y_val, y_val_pred)
# val_corr, _ = pearsonr(y_val, y_val_pred)

# print(f"Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")
# print(f"Train Pearson: {train_corr:.4f}, Val Pearson: {val_corr:.4f}")


#hist 

from sklearn.ensemble import HistGradientBoostingRegressor

hgb_model = HistGradientBoostingRegressor(
    max_iter=300,             # same as n_estimators
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=30,
    random_state=42
)

hgb_model.fit(X_train, y_train)


#evaluate hist

y_train_pred = hgb_model.predict(X_train)
y_val_pred = hgb_model.predict(X_val)

# Train metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_corr, _ = pearsonr(y_train_pred, y_train )

# Val metrics
val_mse = mean_squared_error(y_val, y_val_pred)
val_corr, _ = pearsonr(y_train_pred, y_train )

print(f"Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")
print(f"Train Pearson: {train_corr:.4f}, Val Pearson: {val_corr:.4f}")


val_corr



test_pred_hgb = hgb_model.predict(X_test_scaled)


submission = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")
submission["prediction"] = test_pred_hgb
submission.to_csv("submission.csv", index=False)
print("ğŸ“� Submission file saved as 'submission.csv'")

submission.head()



#XG boost

from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    verbosity=0
)

xgb_model.fit(X_train, y_train)


# light gbm

from lightgbm import LGBMRegressor

lgb_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=6,
    num_leaves=40,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgb_model.fit(X_train, y_train)


# evaluate xg boost

y_train_pred = xgb_model.predict(X_train)
y_val_pred = xgb_model.predict(X_val)

# Train metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_corr, _ = pearsonr(y_train, y_train_pred)

# Val metrics
val_mse = mean_squared_error(y_val, y_val_pred)
val_corr, _ = pearsonr(y_val, y_val_pred)

print(f"Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")
print(f"Train Pearson: {train_corr:.4f}, Val Pearson: {val_corr:.4f}")


#evaluate light gbm

y_train_pred = lgb_model.predict(X_train)
y_val_pred = lgb_model.predict(X_val)

# Train metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_corr, _ = pearsonr(y_train, y_train_pred)

# Val metrics
val_mse = mean_squared_error(y_val, y_val_pred)
val_corr, _ = pearsonr(y_val, y_val_pred)

print(f"Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")
print(f"Train Pearson: {train_corr:.4f}, Val Pearson: {val_corr:.4f}")













