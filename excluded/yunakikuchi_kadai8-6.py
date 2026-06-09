# -----------------------------------------------------------
# 学習データ，テストデータの読み込みと準備（このセルは変更不可）
# -----------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 乱数シードがある関数を呼ぶときはこの値で統一する
rseed = 71

# 学習データ、テストデータの読み込み
train = pd.read_csv('../input/demand-forecasting-kernels-only/train.csv')
test = pd.read_csv('../input/demand-forecasting-kernels-only/test.csv')

# 学習データから目的変数を分離
train_x = train.drop(['sales'], axis=1)
train_y = train['sales']

# テストデータからid列を分離
test_x = test.drop(['id'], axis=1)
test_id = test['id'] # submit用に温存

# 学習データにおける各特徴量の要約統計量・ヒストグラム・欠損(True: 欠損値あり)
print(train_x.describe())
print(train_x.isnull().any())
train_x.hist(bins=100, color="blue", grid=True, label='pandas')
plt.show()

# テストデータにおける各特徴量の要約統計量・ヒストグラム・欠損(True: 欠損値あり)
print(test_x.describe())
print(test_x.isnull().any())
test_x.hist(bins=100, color="blue", grid=True, label='pandas')
plt.show()


# -----------------------------------------------------------
# (A) 変数 store の前処理（改良版）
# -----------------------------------------------------------
col = 'store'

# ラベルエンコーディング
le = LabelEncoder()
le.fit(train_x[col].fillna('NA'))
train_x['store'] = le.transform(train_x[col].fillna('NA'))
test_x['store'] = le.transform(test_x[col].fillna('NA'))


# -----------------------------------------------------------
# (B) 変数 item の前処理（改良版）
# -----------------------------------------------------------
col = 'item'

# ラベルエンコーディング
le = LabelEncoder()
le.fit(train_x[col].fillna('NA'))
train_x['item'] = le.transform(train_x[col].fillna('NA'))
test_x['item'] = le.transform(test_x[col].fillna('NA'))

# store × item の組合せ特徴
train_x['store_item'] = train_x['store'] * 1000 + train_x['item']
test_x['store_item'] = test_x['store'] * 1000 + test_x['item']

# ラベルエンコード（組合せ用）
le = LabelEncoder()
le.fit(train_x['store_item'])
train_x['store_item'] = le.transform(train_x['store_item'])
test_x['store_item'] = le.transform(test_x['store_item'])


# -----------------------------------------------------------
# (C) 変数 date の前処理（改良版）
# -----------------------------------------------------------
col = 'date'

# 日付をdatetime型に変換
train[col] = pd.to_datetime(train[col])
test[col] = pd.to_datetime(test[col])

# 基本特徴量
train_x['dow'] = train[col].dt.dayofweek
test_x['dow'] = test[col].dt.dayofweek

train_x['year'] = train[col].dt.year
test_x['year'] = test[col].dt.year

train_x['month'] = train[col].dt.month
test_x['month'] = test[col].dt.month

train_x['day'] = train[col].dt.day
test_x['day'] = test[col].dt.day

train_x['dayofyear'] = train[col].dt.dayofyear
test_x['dayofyear'] = test[col].dt.dayofyear

train_x['quarter'] = train[col].dt.quarter
test_x['quarter'] = test[col].dt.quarter

train_x['is_month_start'] = train[col].dt.is_month_start.astype(int)
test_x['is_month_start'] = test[col].dt.is_month_start.astype(int)

train_x['is_month_end'] = train[col].dt.is_month_end.astype(int)
test_x['is_month_end'] = test[col].dt.is_month_end.astype(int)

train_x['is_weekend'] = train[col].dt.dayofweek.isin([5, 6]).astype(int)
test_x['is_weekend'] = test[col].dt.dayofweek.isin([5, 6]).astype(int)

train_x['weekofyear'] = train[col].dt.isocalendar().week.astype(int)
test_x['weekofyear'] = test[col].dt.isocalendar().week.astype(int)

# store × dow の組合せ特徴
train_x['store_dow'] = train_x['store'] * 10 + train_x['dow']
test_x['store_dow'] = test_x['store'] * 10 + test_x['dow']

le = LabelEncoder()
le.fit(train_x['store_dow'])
train_x['store_dow'] = le.transform(train_x['store_dow'])
test_x['store_dow'] = le.transform(test_x['store_dow'])

# 時系列の周期性（sin/cos） → 月
train_x['month_sin'] = np.sin(2 * np.pi * train_x['month'] / 12)
test_x['month_sin'] = np.sin(2 * np.pi * test_x['month'] / 12)

# 曜日の周期性（sin）
train_x['dow_sin'] = np.sin(2 * np.pi * train_x['dow'] / 7)
test_x['dow_sin'] = np.sin(2 * np.pi * test_x['dow'] / 7)

# 年×月を結合して1つの時間的連続特徴に（年 * 12 + 月）
train_x['year_month'] = train_x['year'] * 12 + train_x['month']
test_x['year_month'] = test_x['year'] * 12 + test_x['month']

# date列削除
train_x = train_x.drop([col], axis=1)
test_x = test_x.drop([col], axis=1)


# -----------------------------------------------------------
# XGBRegressorの学習・推論（このセルは変更不可）
# -----------------------------------------------------------
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# 回帰用GBDTの学習
model = XGBRegressor(objective="reg:squarederror", random_state=rseed)
model.fit(train_x, train_y)

# 学習データ予測 & RMSE算出（参考値）
train_pred = model.predict(train_x)
rmse = np.sqrt(mean_squared_error(train_y, train_pred))
print(f"Train RMSE: {rmse:.5f}")

# テスト予測 & 提出ファイル作成
test_pred = model.predict(test_x)
submission = pd.DataFrame({'id': test_id, 'sales': test_pred})
submission.to_csv("submission.csv", index=False)

