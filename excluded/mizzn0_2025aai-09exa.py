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

# print("trainの変数：", set(train.columns))
# print("testの変数：", set(test.columns))


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
# col = 'date'

# # ベースライン：使用しない（※テスト側が全て未知カテゴリになるためそのままの使用は困難）
# train_x = train_x.drop([col], axis=1)
# test_x = test_x.drop([col], axis=1)

# -----------------------------------------------------------
# (C) 変数 date の前処理
# -----------------------------------------------------------

col = 'date'

# 日付文字列をPandasの日時型(datetime: dt)に変換
train[col] = pd.to_datetime(train[col])
test[col]  = pd.to_datetime(test[col])

# 新特徴「曜日」を作成（※Pandasの日時型(dt)の機能を使用；詳細は各自で調べること）
new_col = 'dow' # ⇔ day of the week
train_x[new_col] = train[col].dt.dayofweek
test_x[new_col]  = test[col].dt.dayofweek
# print(train_x.head(21)) # 最初の3週間分を表示(dow = 0 ⇔ 月曜日）

# 曜日を周期的に
train_x['dow_sin'] = np.sin(2 * np.pi * train['date'].dt.dayofweek / 7)
train_x['dow_cos'] = np.cos(2 * np.pi * train['date'].dt.dayofweek / 7)

test_x['dow_sin'] = np.sin(2 * np.pi * test['date'].dt.dayofweek / 7)
test_x['dow_cos'] = np.cos(2 * np.pi * test['date'].dt.dayofweek / 7)

# 土日(5, 6)なら
train_x['weekend'] = train_x['dow'].isin([5, 6]).astype(int)
test_x['weekend']  = test_x['dow'].isin([5, 6]).astype(int)

# 金曜日(4)なら
train_x['friday'] = (train_x['dow'] == 4).astype(int)
test_x['friday']  = (test_x['dow'] == 4).astype(int)

# 月曜日なら
train_x['monday'] = (train_x['dow'] == 0).astype(int)
test_x['monday']  = (test_x['dow'] == 0).astype(int)

# それ以外
train_x['others'] =  train_x['dow'].isin([1, 2, 3]).astype(int)
test_x['others']  = train_x['dow'].isin([1, 2, 3]).astype(int)

# 月をいれる
train_x['month'] = train['date'].dt.month
test_x['month']  = test['date'].dt.month

# 周期的にする
train_x['month_sin'] = np.sin(2 * np.pi * train['date'].dt.month / 12)
train_x['month_cos'] = np.cos(2 * np.pi * train['date'].dt.month / 12)

test_x['month_sin'] = np.sin(2 * np.pi * test['date'].dt.month / 12)
test_x['month_cos'] = np.cos(2 * np.pi * test['date'].dt.month / 12)

# 上旬中旬下旬で分割
def map_ten_days(day):
    if day <= 10:
        return 0  # 上旬
    elif day <= 20:
        return 1  # 中旬
    else:
        return 2  # 下旬
        
train_x['ten_days'] = train['date'].dt.day.apply(map_ten_days)
test_x['ten_days']  = test['date'].dt.day.apply(map_ten_days)

train_x['ten_sin'] = np.sin(2 * np.pi * train_x['ten_days'] / 3)
train_x['ten_cos'] = np.cos(2 * np.pi * train_x['ten_days'] / 3)

test_x['ten_sin'] = np.sin(2 * np.pi * test_x['ten_days'] / 3)
test_x['ten_cos'] = np.cos(2 * np.pi * test_x['ten_days'] / 3)

# ダブりを落とす
train_x = train_x.drop([new_col], axis=1)
test_x = test_x.drop([new_col], axis=1)

train_x = train_x.drop(['month'], axis=1)
test_x = test_x.drop(['month'], axis=1)

train_x = train_x.drop(['ten_days'], axis=1)
test_x = test_x.drop(['ten_days'], axis=1)


print(train_x.head(21))
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

