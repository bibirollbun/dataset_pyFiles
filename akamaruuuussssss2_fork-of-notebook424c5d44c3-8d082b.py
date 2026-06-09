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
# (A) 変数 store の前処理
# -----------------------------------------------------------
col = 'store'
le = LabelEncoder()
le.fit(train_x[col].fillna('NA'))
train_x[col] = le.transform(train_x[col].fillna('NA'))
test_x[col] = le.transform(test_x[col].fillna('NA'))

# storeの平方根変換・二次特徴量を追加（非線形性の抽出）
train_x['store_sqrt'] = np.sqrt(train_x[col])
test_x['store_sqrt'] = np.sqrt(test_x[col])

train_x['store_square'] = train_x[col] ** 2
test_x['store_square'] = test_x[col] ** 2


# -----------------------------------------------------------
# (B) 変数 item の前処理
# -----------------------------------------------------------
col = 'item'
le = LabelEncoder()
le.fit(train_x[col].fillna('NA'))
train_x[col] = le.transform(train_x[col].fillna('NA'))
test_x[col] = le.transform(test_x[col].fillna('NA'))

# itemのカテゴリーグループ（10で割るのはベースラインなので5で区切るなど変えてみる）
train_x['item_cat5'] = train_x[col] // 5
test_x['item_cat5'] = test_x[col] // 5

# storeとitemの組み合わせ特徴（ベースラインよりバリエーション追加）
train_x['store_item'] = train_x['store'].astype(str) + '_' + train_x['item'].astype(str)
test_x['store_item']  = test_x['store'].astype(str) + '_' + test_x['item'].astype(str)

le_combo = LabelEncoder()
le_combo.fit(train_x['store_item'])
train_x['store_item'] = le_combo.transform(train_x['store_item'])
test_x['store_item']  = le_combo.transform(test_x['store_item'])


# -----------------------------------------------------------
# (C) 変数 date の前処理
# -----------------------------------------------------------
col = 'date'

train[col] = pd.to_datetime(train[col])
test[col]  = pd.to_datetime(test[col])

train_x['dow'] = train[col].dt.dayofweek
test_x['dow']  = test[col].dt.dayofweek

train_x['weekofyear'] = train[col].dt.isocalendar().week.astype(int)
test_x['weekofyear']  = test[col].dt.isocalendar().week.astype(int)

train_x['is_month_end'] = train[col].dt.is_month_end.astype(int)
test_x['is_month_end']  = test[col].dt.is_month_end.astype(int)

# 新規特徴量：月の初めかどうか
train_x['is_month_start'] = train[col].dt.is_month_start.astype(int)
test_x['is_month_start']  = test[col].dt.is_month_start.astype(int)

# 新規特徴量：四半期（1〜4）
train_x['quarter'] = train[col].dt.quarter
test_x['quarter']  = test[col].dt.quarter

# 新規特徴量：日（1〜31）
train_x['day'] = train[col].dt.day
test_x['day']  = test[col].dt.day

# store × 曜日 などの交互作用特徴量
train_x['store_dow'] = train_x['store'].astype(str) + '_' + train_x['dow'].astype(str)
test_x['store_dow']  = test_x['store'].astype(str) + '_' + test_x['dow'].astype(str)

le_interact = LabelEncoder()
le_interact.fit(train_x['store_dow'])
train_x['store_dow'] = le_interact.transform(train_x['store_dow'])
test_x['store_dow'] = le_interact.transform(test_x['store_dow'])

# date列は削除
train_x = train_x.drop([col], axis=1, errors='ignore')
test_x = test_x.drop([col], axis=1, errors='ignore')


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

