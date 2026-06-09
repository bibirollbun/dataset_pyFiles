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
# ベースラインモデル：ラベルエンコーディング
le = LabelEncoder()
le.fit(train_x['store'].fillna('NA'))
train_x['store'] = le.transform(train_x['store'].fillna('NA'))
test_x['store'] = le.transform(test_x['store'].fillna('NA'))


# -----------------------------------------------------------
# (B) 変数 item の前処理
# -----------------------------------------------------------
# ベースライン：ラベルエンコーディング
le = LabelEncoder()
le.fit(train_x['item'].fillna('NA'))
train_x['item'] = le.transform(train_x['item'].fillna('NA'))
test_x['item'] = le.transform(test_x['item'].fillna('NA'))


# -----------------------------------------------------------
# (C) 変数 date の前処理
# -----------------------------------------------------------
# 日付文字列をPandasの日時型(datetime: dt)に変換
train['date'] = pd.to_datetime(train['date'])
test['date']  = pd.to_datetime(test['date'])

#------ 問３ -----------------
# 新特徴「曜日」を作成（※Pandasの日時型(dt)の機能を使用；詳細は各自で調べること）
#train_x['dayofweek'] = train['date'].dt.dayofweek
#test_x['dayofweek']  = test['date'].dt.dayofweek
#print(train_x.head(21)) # 最初の3週間分を表示(dow = 0 ⇔ 月曜日）
#----------------------------

#------ 問４ -----------------
# 新特徴「年」を作成
train_x['year'] = train['date'].dt.year
test_x['year']  = test['date'].dt.year
# 新特徴「月」を作成 (周期変数化)
train_x['month'] = train['date'].dt.month
test_x['month']  = test['date'].dt.month
train_x['month_sin'] = np.sin(2 * np.pi * train_x['month'] / 12)
train_x['month_cos'] = np.cos(2 * np.pi * train_x['month'] / 12)
test_x['month_sin']  = np.sin(2 * np.pi * test_x['month'] / 12)
test_x['month_cos']  = np.cos(2 * np.pi * test_x['month'] / 12)
# 新特徴「日」を作成
train_x['day'] = train['date'].dt.day
test_x['day']  = test['date'].dt.day
# 新特徴「曜日」を作成 (周期変数化)
train_x['weekday'] = train['date'].dt.dayofweek #(weekday : 0=月, 1=火,..., 6=日）
test_x['weekday']  = test['date'].dt.dayofweek
train_x['weekday_sin'] = np.sin(2 * np.pi * train_x['weekday'] / 7)
train_x['weekday_cos'] = np.cos(2 * np.pi * train_x['weekday'] / 7)
test_x['weekday_sin']  = np.sin(2 * np.pi * test_x['weekday'] / 7)
test_x['weekday_cos']  = np.cos(2 * np.pi * test_x['weekday'] / 7)
# 新特徴量「月末・月初」追加
#train_x['is_month_end']   = train['date'].dt.is_month_end.astype(int)
#train_x['is_month_start'] = train['date'].dt.is_month_start.astype(int)
#test_x['is_month_end']    = test['date'].dt.is_month_end.astype(int)
#test_x['is_month_start']  = test['date'].dt.is_month_start.astype(int)
#----------------------------

# 未加工の 'date' は使用しない
train_x = train_x.drop(['date'], axis=1)
test_x = test_x.drop(['date'], axis=1)
# 'month' も使用しない
train_x = train_x.drop(['month'], axis=1)
test_x = test_x.drop(['month'], axis=1)
# 'weekday' も使用しない
train_x = train_x.drop(['weekday'], axis=1)
test_x = test_x.drop(['weekday'], axis=1)

print(train_x.head(11)) # 最初の3週間分を表示(dow = 0 ⇔ 月曜日）


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

