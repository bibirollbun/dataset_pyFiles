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
# (A) store の前処理
# -----------------------------------------------------------
col = 'store'

le = LabelEncoder()
le.fit(train_x[col])
train_x[col] = le.transform(train_x[col])
test_x[col] = le.transform(test_x[col])



# -----------------------------------------------------------
# (B) item の前処理
# -----------------------------------------------------------
col = 'item'

le = LabelEncoder()
le.fit(train_x[col])
train_x[col] = le.transform(train_x[col])
test_x[col] = le.transform(test_x[col])




# -----------------------------------------------------------
# (C) date の前処理（Version 1 + 微调）
# -----------------------------------------------------------
col = 'date'

train[col] = pd.to_datetime(train[col])
test[col] = pd.to_datetime(test[col])

# 日期分解
train_x['day'] = train[col].dt.day
test_x['day'] = test[col].dt.day

train_x['month'] = train[col].dt.month
test_x['month'] = test[col].dt.month

train_x['dow'] = train[col].dt.dayofweek
test_x['dow'] = test[col].dt.dayofweek

train_x['week'] = train[col].dt.isocalendar().week.astype(int)
test_x['week'] = test[col].dt.isocalendar().week.astype(int)

train_x['dayofyear'] = train[col].dt.dayofyear
test_x['dayofyear'] = test[col].dt.dayofyear

train_x['is_weekend'] = train_x['dow'].isin([5,6]).astype(int)
test_x['is_weekend'] = test_x['dow'].isin([5,6]).astype(int)

train_x['is_start_month'] = (train_x['day'] <= 3).astype(int)
test_x['is_start_month'] = (test_x['day'] <= 3).astype(int)

train_x['is_end_month'] = (train_x['day'] >= 28).astype(int)
test_x['is_end_month'] = (test_x['day'] >= 28).astype(int)

train_x['is_middle_month'] = ((train_x['day'] >= 13) & (train_x['day'] <= 18)).astype(int)
test_x['is_middle_month'] = ((test_x['day'] >= 13) & (test_x['day'] <= 18)).astype(int)

# 周期性特征
train_x['sin_doy'] = np.sin(2 * np.pi * train_x['dayofyear'] / 365)
test_x['sin_doy'] = np.sin(2 * np.pi * test_x['dayofyear'] / 365)

train_x['cos_doy'] = np.cos(2 * np.pi * train_x['dayofyear'] / 365)
test_x['cos_doy'] = np.cos(2 * np.pi * test_x['dayofyear'] / 365)

train_x['sin_dow'] = np.sin(2 * np.pi * train_x['dow'] / 7)
test_x['sin_dow'] = np.sin(2 * np.pi * test_x['dow'] / 7)

train_x['cos_dow'] = np.cos(2 * np.pi * train_x['dow'] / 7)
test_x['cos_dow'] = np.cos(2 * np.pi * test_x['dow'] / 7)

# store × month
train_x['store_month'] = train_x['store'] * 100 + train_x['month']
test_x['store_month'] = test_x['store'] * 100 + test_x['month']

le_store_month = LabelEncoder()
le_store_month.fit(train_x['store_month'])
train_x['store_month_enc'] = le_store_month.transform(train_x['store_month'])
test_x['store_month_enc'] = le_store_month.transform(test_x['store_month'])

# ➤ item × week
train_x['item_week'] = train_x['item'] * 100 + train_x['week']
test_x['item_week'] = test_x['item'] * 100 + test_x['week']

train_x = train_x.drop(columns=[col, 'store_month'])  # store_month 已转为编码版本
test_x = test_x.drop(columns=[col, 'store_month'])



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

