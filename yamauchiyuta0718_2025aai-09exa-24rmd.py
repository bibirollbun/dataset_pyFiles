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

# ベースラインモデル：ラベルエンコーディング
le = LabelEncoder()
le.fit(train_x[col].fillna('NA'))
train_x[col] = le.transform(train_x[col].fillna('NA'))
test_x[col] = le.transform(test_x[col].fillna('NA'))


# -----------------------------------------------------------
# (B) 変数 item の前処理
# -----------------------------------------------------------
col = 'item'

# ベースライン：ラベルエンコーディング
le = LabelEncoder()
le.fit(train_x[col].fillna('NA'))
train_x[col] = le.transform(train_x[col].fillna('NA'))
test_x[col] = le.transform(test_x[col].fillna('NA'))


# -----------------------------------------------------------
# (C) 変数 date の前処理
# -----------------------------------------------------------
col = 'date'

# 日付文字列をPandasの日時型(datetime: dt)に変換
train[col] = pd.to_datetime(train[col])
test[col]  = pd.to_datetime(test[col])

# 新特徴「年」を作成
new_col_year = 'year'
train_x[new_col_year] = train[col].dt.year
test_x[new_col_year]  = test[col].dt.year

# 新特徴「月」を作成
new_col_month = 'month'
train_x[new_col_month] = train[col].dt.month
test_x[new_col_month]  = test[col].dt.month

# 新特徴「日」を作成
new_col_day = 'day'
train_x[new_col_day] = train[col].dt.day
test_x[new_col_day]  = test[col].dt.day

# 新特徴「週」を作成
train_x['week'] = train[col].dt.isocalendar().week.astype(int)
test_x['week']  = test[col].dt.isocalendar().week.astype(int)

# 新特徴「曜日」を作成
train_x['dow'] = train[col].dt.dayofweek  # 曜日（0=月曜日, ..., 6=日曜日）
test_x['dow']  = test[col].dt.dayofweek

# 新特徴「四半期情報」を作成
train_x['quarter'] = train[col].dt.quarter
test_x['quarter']  = test[col].dt.quarter

train_x['is_quarter_start'] = train[col].dt.is_quarter_start.astype(int)
test_x['is_quarter_start']  = test[col].dt.is_quarter_start.astype(int)

train_x['is_quarter_end'] = train[col].dt.is_quarter_end.astype(int)
test_x['is_quarter_end']  = test[col].dt.is_quarter_end.astype(int)

# print(train_x.head(21)) # 最初の3週間分を表示(dow = 0 ⇔ 月曜日）

# 未加工の 'date' は使用しない（※テスト側が全て未知カテゴリになるためそのままの使用は困難）
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

