# Rossmann Store Sales - Improved & Simplified
import numpy as np
import pandas as pd
import lightgbm as lgb
import os

# Load Data
ROOT = '/kaggle/input/rossmann-store-sales'
df_train = pd.read_csv(
    f'{ROOT}/train.csv',
    parse_dates=['Date'],
    dtype={'StateHoliday': 'category'}
)
df_test = pd.read_csv(
    f'{ROOT}/test.csv',
    parse_dates=['Date'],
    dtype={'StateHoliday': 'category'}
)
df_store = pd.read_csv(f'{ROOT}/store.csv')


# Add test label column for uniform processing
df_test['Sales'] = np.nan
df_all = pd.concat([df_train, df_test], sort=False)

# Merge store info
df_all = df_all.merge(df_store, on='Store', how='left')


# Fill NA
na_fill = {
    'Open': 1,
    'CompetitionDistance': df_all['CompetitionDistance'].median(),
    'CompetitionOpenSinceMonth': df_all['CompetitionOpenSinceMonth'].mode()[0],
    'CompetitionOpenSinceYear': df_all['CompetitionOpenSinceYear'].mode()[0],
    'Promo2': 0,
    'Promo2SinceWeek': 0,
    'Promo2SinceYear': 0,
    'PromoInterval': ''
}
df_all.fillna(na_fill, inplace=True)

# Date Features
df_all['Year'] = df_all['Date'].dt.year
df_all['Month'] = df_all['Date'].dt.month
df_all['Day'] = df_all['Date'].dt.day
df_all['DayOfWeek'] = df_all['Date'].dt.dayofweek

df_all['WeekOfYear'] = df_all['Date'].dt.isocalendar().week.astype(int)
df_all['IsWeekend'] = df_all['DayOfWeek'].isin([5,6]).astype(int)

# Competition Open Time
comp_open = (12 * (df_all['Year'] - df_all['CompetitionOpenSinceYear']) + 
             (df_all['Month'] - df_all['CompetitionOpenSinceMonth'])).clip(lower=0)
df_all['CompetitionOpenMonth'] = comp_open


# Promo2 running
promo2 = ((df_all['Promo2'] == 1) &
          ((df_all['Promo2SinceYear'] < df_all['Year']) |
           ((df_all['Promo2SinceYear'] == df_all['Year']) & (df_all['Promo2SinceWeek'] <= df_all['WeekOfYear']))))
df_all['IsPromo2Running'] = promo2.astype(int)

# Encode categoricals
df_all['StoreType'] = df_all['StoreType'].astype('category').cat.codes
df_all['Assortment'] = df_all['Assortment'].astype('category').cat.codes
df_all['StateHoliday'] = df_all['StateHoliday'].astype(str).astype('category').cat.codes

# After all features are created, split again
train = df_all[~df_all['Sales'].isna()].copy()
test = df_all[df_all['Sales'].isna()].copy()

# Drop zero sales (often closed)
train = train[train['Sales'] > 0]

# Features to use
features = [
    'Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'StoreType', 'Assortment', 'Open',
    'Year', 'Month', 'Day', 'WeekOfYear', 'IsWeekend',
    'CompetitionDistance', 'CompetitionOpenMonth', 'IsPromo2Running'
]

X_train = train[features]
y_train = train['Sales']
X_test = test[features]


# --- セル5の既存コード（LightGBMトレーニングと予測部分）を完全に置き換え ---

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# KFoldの設定
kf = KFold(n_splits=5, shuffle=True, random_state=42) # 5分割交差検証

oof_preds = np.zeros(len(X_train)) # Out-Of-Fold予測値格納用
test_preds = np.zeros(len(X_test)) # テストセットの予測値格納用 (各フォールドの平均を取る)

# LightGBMパラメータ (上記の調整済みparamsを使用)
params = {
    'objective': 'regression_l1',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 2000, # early_stoppingを使うので大きめに設定
    'learning_rate': 0.01,
    'num_leaves': 64,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_jobs': -1,
    'seed': 42,
    'verbose': -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"--- Fold {fold+1} ---")
    X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]

    train_set_fold = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_set_fold = lgb.Dataset(X_val_fold, label=y_val_fold)

    model_fold = lgb.train(
        params,
        train_set_fold,
        valid_sets=[val_set_fold],
        callbacks=[lgb.early_stopping(100, verbose=False)] # 100回改善がなければ停止
    )

    oof_preds[val_idx] = model_fold.predict(X_val_fold) # OOF予測を格納
    test_preds += model_fold.predict(X_test) / kf.n_splits # テスト予測を平均

# 最終的なテスト予測をtestデータフレームに格納
test['Sales'] = test_preds

# Openが0の店舗の売上は0に設定
test.loc[test['Open'] == 0, 'Sales'] = 0

# OOF予測のRMSE評価
rmse_oof = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"Overall OOF RMSE: {rmse_oof}")


# Submission
submission = test[['Id', 'Sales']].copy()
submission["Id"] = submission["Id"].astype(int)
submission.to_csv('submission.csv', index=False)
print("✅ Submission saved.")

