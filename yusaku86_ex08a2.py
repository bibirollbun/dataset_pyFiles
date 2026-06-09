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
# (A) 変数 store の前処理（LabelEncoding + storeごとの平均売上を特徴量に追加）
# -----------------------------------------------------------
col = 'store'

# Label Encoding
le = LabelEncoder()
le.fit(train_x[col].fillna('NA'))
train_x[col] = le.transform(train_x[col].fillna('NA'))
test_x[col] = le.transform(test_x[col].fillna('NA'))

# storeごとの平均売上を特徴量として追加（目的変数salesの直接利用ではないためOK）
store_sales_mean = train.groupby('store')['sales'].mean()
train_x['store_sales_mean'] = train_x['store'].map(store_sales_mean)
test_x['store_sales_mean'] = test_x['store'].map(store_sales_mean)



# -----------------------------------------------------------
# (B) 変数 item の前処理（LabelEncoding + itemごとの売上の標準偏差を特徴量に追加）
# -----------------------------------------------------------
col = 'item'

# Label Encoding
le = LabelEncoder()
le.fit(train_x[col].fillna('NA'))
train_x[col] = le.transform(train_x[col].fillna('NA'))
test_x[col] = le.transform(test_x[col].fillna('NA'))

# itemごとの sales の標準偏差を追加
item_sales_std = train.groupby('item')['sales'].std()
train_x['item_sales_std'] = train_x['item'].map(item_sales_std)
test_x['item_sales_std'] = test_x['item'].map(item_sales_std)



# -----------------------------------------------------------
# (C) 変数 date の前処理
# -----------------------------------------------------------
col = 'date'

train[col] = pd.to_datetime(train[col])
test[col]  = pd.to_datetime(test[col])

train_x['year'] = train[col].dt.year
test_x['year']  = test[col].dt.year

train_x['month'] = train[col].dt.month
test_x['month']  = test[col].dt.month

train_x['day'] = train[col].dt.day
test_x['day']  = test[col].dt.day

train_x['dow'] = train[col].dt.dayofweek
test_x['dow']  = test[col].dt.dayofweek

train_x['weekofyear'] = train[col].dt.isocalendar().week.astype(int)
test_x['weekofyear']  = test[col].dt.isocalendar().week.astype(int)

train_x['dayofyear'] = train[col].dt.dayofyear
test_x['dayofyear']  = test[col].dt.dayofyear

train_x['is_weekend'] = (train[col].dt.dayofweek >= 5).astype(int)
test_x['is_weekend']  = (test[col].dt.dayofweek >= 5).astype(int)

# ✅ 追加: 四半期情報
train_x['quarter'] = train[col].dt.quarter
test_x['quarter']  = test[col].dt.quarter

# ✅ 追加: 月初/月末フラグ
train_x['is_month_start'] = train[col].dt.is_month_start.astype(int)
test_x['is_month_start']  = test[col].dt.is_month_start.astype(int)

train_x['is_month_end'] = train[col].dt.is_month_end.astype(int)
test_x['is_month_end']  = test[col].dt.is_month_end.astype(int)

# ラベルエンコーディング（train + test 合体で fit する）
from sklearn.preprocessing import LabelEncoder
for new_col in train_x.columns.difference(['store', 'item']):  # 変換対象を明示
    le = LabelEncoder()
    all_vals = pd.concat([train_x[new_col], test_x[new_col]])
    le.fit(all_vals)
    train_x[new_col] = le.transform(train_x[new_col])
    test_x[new_col]  = le.transform(test_x[new_col])

# 元の 'date' は不要
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

