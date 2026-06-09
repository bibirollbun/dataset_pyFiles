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
import lightgbm as lgb

# --- 0. データ読み込みと結合 ---
# ※ Kaggle環境での一般的な設定をシミュレーション
dfs = {}

for name in ['train','test']:
    df = pd.read_csv(f'/kaggle/input/bike-sharing-demand/{name}.csv')
    df['_data'] = name  # train/test識別用
    dfs[name] = df


# train/test をまとめる

df_all = pd.concat([dfs['train'], dfs['test']], ignore_index=True)

# train/test をまとめる
df_all = pd.concat([dfs['train'], dfs['test']], ignore_index=True)
train_len = len(dfs['train'])
print("✅ データの読み込みと結合完了")


# =================================================================
# 1. 特徴量エンジニアリング
# =================================================================

# 1-1. datetimeからの時間情報抽出
df_all['datetime'] = pd.to_datetime(df_all['datetime'])
df_all['year'] = df_all['datetime'].dt.year
df_all['month'] = df_all['datetime'].dt.month
df_all['hour'] = df_all['datetime'].dt.hour
df_all['weekday'] = df_all['datetime'].dt.weekday # 月曜: 0, 日曜: 6

# 1-2. 多重共線性の回避と不要な列の削除
df_all = df_all.drop(['atemp', 'datetime'], axis=1)

# 1-3. カテゴリ変数のダミー変数化 (One-Hot Encoding)
categorical_features = ['year', 'month', 'hour', 'weekday', 'season', 'weather', 'holiday', 'workingday']

# One-Hot Encoding の前にカテゴリ変数として型を変換
for col in categorical_features:
    df_all[col] = df_all[col].astype('category')

df_all = pd.get_dummies(df_all, columns=categorical_features, drop_first=True)

# 1-4. 目的変数と識別子（_data）を一旦退避
df_target = df_all[['_data', 'casual', 'registered', 'count']].copy()
df_all = df_all.drop(['_data', 'casual', 'registered', 'count'], axis=1)


print("✅ 特徴量エンジニアリング完了")
print(f"最終的な特徴量数: {df_all.shape[1]}")


# =================================================================
# 2. モデル構築のためのデータ再分離と目的変数変換
# =================================================================

# 2-1. データの再分離
X_train = df_all.iloc[:train_len]
X_test = df_all.iloc[train_len:]

# 2-2. 目的変数の抽出と Log(x+1) 変換
y_train = df_target.loc[df_target['_data'] == 'train', 'count']
y_train_log = np.log1p(y_train)

print(f"学習データ (X_train) 形状: {X_train.shape}")


# =================================================================
# 3. LightGBMモデルの学習と予測
# =================================================================

# 3-1. モデルの定義と学習
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'n_jobs': -1,
    'seed': 42,
    'verbose': -1
}

model = lgb.LGBMRegressor(**lgb_params)
model.fit(X_train, y_train_log)

print("✅ LightGBMモデルの学習完了")

# 3-2. テストデータに対する予測
predictions_log = model.predict(X_test)

# 3-3. 逆変換とクリッピング (利用者は非負)
predictions = np.expm1(predictions_log)
predictions = np.maximum(0, predictions).astype(int)

print("✅ 予測と逆変換完了")


# =================================================================
# 4. 提出ファイルの作成
# =================================================================

# 提出ファイルのフォーマットに合わせてデータフレームを作成
submission = pd.DataFrame({
    'datetime': dfs['test']['datetime'],  # 元のテストデータからdatetimeを取得
    'count': predictions
})

# 提出ファイル（submission.csv）として保存
submission.to_csv('submission.csv', index=False)

print("\n--- 最終予測結果（一部） ---")
print(submission.head())

