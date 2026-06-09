# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Thesis Annotation:
# This section imports all required Python libraries for data processing, modeling,
# visualization, and machine learning. Unused libraries have been removed to keep the code clean
# and reproducible. These tools are essential for implementing the hybrid forecasting model.

import numpy as np
import pandas as pd
import os
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
from prophet import Prophet
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# ğŸ“‚ Load Data
train = pd.read_csv("/kaggle/input/rossmann-store-sales/train.csv")
test = pd.read_csv("/kaggle/input/rossmann-store-sales/test.csv")
store = pd.read_csv("/kaggle/input/rossmann-store-sales/store.csv")
sample_submission = pd.read_csv("/kaggle/input/rossmann-store-sales/sample_submission.csv")

print("âœ… Files loaded successfully")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Store shape: {store.shape}")
print(f"Sample submission shape: {sample_submission.shape}")


# Inspect Structures & Nulls

# Display first few rows
print("\nğŸ”� train.csv preview:")
print(train.head())

print("\nğŸ”� test.csv preview:")
print(test.head())

print("\nğŸ”� store.csv preview:")
print(store.head())

# Data types
print("\nğŸ”� train dtypes:")
print(train.dtypes)

print("\nğŸ”� test dtypes:")
print(test.dtypes)

print("\nğŸ”� store dtypes:")
print(store.dtypes)

# Null value counts
print("\nğŸš¨ Missing values in train:")
print(train.isnull().sum())

print("\nğŸš¨ Missing values in test:")
print(test.isnull().sum())

print("\nğŸš¨ Missing values in store:")
print(store.isnull().sum())


# ğŸ“Š Merge and Clean
train = pd.merge(train, store, on='Store', how='left')
test = pd.merge(test, store, on='Store', how='left')
train = train[(train['Open'] != 0) & (train['Sales'] > 0)].copy()
test['ZeroPrediction'] = (test['Open'] == 0).astype(int)


# ğŸ“† Calendar Features
def add_date_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    return df

train = add_date_features(train)
test = add_date_features(test)


# ğŸ�¯ Encode + Promo + Competition
assortment_map = {'a': 1, 'b': 2, 'c': 3}
storetype_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
stateholiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3}
train['Assortment'] = train['Assortment'].map(assortment_map)
test['Assortment'] = test['Assortment'].map(assortment_map)
train['StoreType'] = train['StoreType'].map(storetype_map)
test['StoreType'] = test['StoreType'].map(storetype_map)
train['StateHoliday'] = train['StateHoliday'].astype(str).map(stateholiday_map)
test['StateHoliday'] = test['StateHoliday'].astype(str).map(stateholiday_map)

def promo_month_feature(df):
    df['PromoInterval'] = df['PromoInterval'].fillna('')
    df['IsPromoMonth'] = 0
    for i in df.index:
        interval = df.at[i, 'PromoInterval']
        if interval:
            month_str = datetime(2000, df.at[i, 'Month'], 1).strftime('%b')
            if month_str in interval.split(','):
                df.at[i, 'IsPromoMonth'] = 1
    return df

def compute_competition_open(df):
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0).astype(int)
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0).astype(int)
    df['CompetitionOpenSince'] = pd.to_datetime(dict(year=df['CompetitionOpenSinceYear'], month=df['CompetitionOpenSinceMonth'], day=15), errors='coerce')
    df['CompetitionOpenMonths'] = ((df['Date'].dt.year - df['CompetitionOpenSince'].dt.year) * 12 + (df['Date'].dt.month - df['CompetitionOpenSince'].dt.month)).clip(lower=0)
    return df

train = promo_month_feature(train)
test = promo_month_feature(test)
train = compute_competition_open(train)
test = compute_competition_open(test)
train['Promo2Since'] = pd.to_datetime((train['Promo2SinceYear'].fillna(0).astype(int) * 100 + train['Promo2SinceWeek'].fillna(0).astype(int)).astype(str) + '0', format='%Y%W%w', errors='coerce')
test['Promo2Since'] = pd.to_datetime((test['Promo2SinceYear'].fillna(0).astype(int) * 100 + test['Promo2SinceWeek'].fillna(0).astype(int)).astype(str) + '0', format='%Y%W%w', errors='coerce')
train['Promo2OpenWeeks'] = ((train['Date'] - train['Promo2Since']) / np.timedelta64(1, 'W')).clip(lower=0).fillna(0).astype(int)
test['Promo2OpenWeeks'] = ((test['Date'] - test['Promo2Since']) / np.timedelta64(1, 'W')).clip(lower=0).fillna(0).astype(int)
train['Promo2Active'] = ((train['Promo2'] == 1) & (train['Promo2OpenWeeks'] > 0)).astype(int)
test['Promo2Active'] = ((test['Promo2'] == 1) & (test['Promo2OpenWeeks'] > 0)).astype(int)


# ğŸ”® Prophet Trend Fix: Cover full train+test range
prophet_df = train[['Date', 'Sales']].groupby('Date').sum().reset_index()
prophet_df.columns = ['ds', 'y']
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
prophet_model.fit(prophet_df)
future = pd.DataFrame({'ds': pd.date_range(start=train['Date'].min(), end=test['Date'].max())})
forecast = prophet_model.predict(future)
trend_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'prophet_trend'})
train = pd.merge(train, trend_df, on='Date', how='left')
test = pd.merge(test, trend_df, on='Date', how='left')
train['residual'] = train['Sales'] - train['prophet_trend']


# ğŸ§  SHAP + LAG-ENHANCED Features
shap_top_features = ['Store', 'DayOfWeek', 'Promo', 'Year', 'Month', 'Day', 'SchoolHoliday', 'Promo2Active', 'CompetitionOpenMonths', 'IsPromoMonth']
lag_features = ['Sales_lag1', 'Sales_lag7', 'Sales_roll_mean_7', 'Sales_roll_std_7']
feature_cols = list(dict.fromkeys(shap_top_features + lag_features + ['prophet_trend']))


# ğŸ§¾ Add Lags to Full Train for Modeling
train = train.sort_values(['Store', 'Date'])
train['Sales_lag1'] = train.groupby('Store')['Sales'].shift(1)
train['Sales_lag7'] = train.groupby('Store')['Sales'].shift(7)
train['Sales_roll_mean_7'] = train.groupby('Store')['Sales'].shift(1).rolling(7).mean().reset_index(0, drop=True)
train['Sales_roll_std_7'] = train.groupby('Store')['Sales'].shift(1).rolling(7).std().reset_index(0, drop=True)
train = train.dropna()


# ğŸ”� Split + Train
cutoff_date = train['Date'].max() - pd.Timedelta(weeks=6)
train_set = train[train['Date'] < cutoff_date].copy()
valid_set = train[train['Date'] >= cutoff_date].copy()
X_train = train_set[feature_cols].fillna(0)
y_train = train_set['residual']
X_valid = valid_set[feature_cols].fillna(0)
y_valid = valid_set['residual']

xgb_model = XGBRegressor(
    n_estimators=900,
    learning_rate=0.08,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0,
    min_child_weight=3,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42
)
xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=30, verbose=100)

# ğŸ”® Predict on Test (Generate Lag Features)
recent_train = train[train['Date'] >= train['Date'].max() - pd.Timedelta(days=42)].copy()
recent_train['Dataset'] = 'train'
test['Dataset'] = 'test'
test_lag_input = pd.concat([recent_train, test], sort=False).sort_values(['Store', 'Date'])

def add_lag_rolling(df, group_col, value_col):
    df[value_col + '_lag1'] = df.groupby(group_col)[value_col].shift(1)
    df[value_col + '_lag7'] = df.groupby(group_col)[value_col].shift(7)
    df[value_col + '_roll_mean_7'] = df.groupby(group_col)[value_col].shift(1).rolling(7).mean().reset_index(0, drop=True)
    df[value_col + '_roll_std_7'] = df.groupby(group_col)[value_col].shift(1).rolling(7).std().reset_index(0, drop=True)
    return df

test_lagged = add_lag_rolling(test_lag_input, 'Store', 'Sales')
test_lagged = test_lagged[test_lagged['Dataset'] == 'test'].copy()
test_lagged = test_lagged.fillna(0)

X_test = test_lagged[feature_cols].fillna(0)
y_test_pred_residual = xgb_model.predict(X_test)
test_lagged['Sales'] = y_test_pred_residual + test_lagged['prophet_trend']
test_lagged.loc[test_lagged['ZeroPrediction'] == 1, 'Sales'] = 0
test_lagged['Sales'] = test_lagged['Sales'].clip(lower=0)


# ğŸ’¾ Export
submission = test_lagged[['Id', 'Sales']].copy()
submission['Id'] = submission['Id'].astype(int)
submission['Sales'] = submission['Sales'].round(2)
submission.sort_values('Id', inplace=True)
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv with lag features exported")

