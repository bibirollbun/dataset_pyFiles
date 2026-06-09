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
le_store = LabelEncoder()
le_store.fit(train_x[col].astype(str))
train_x[col] = le_store.transform(train_x[col].astype(str))
test_x[col]  = le_store.transform(test_x[col].astype(str))


# -----------------------------------------------------------
# (B) 変数 item の前処理
# -----------------------------------------------------------
col = 'item'
le_item = LabelEncoder()
le_item.fit(train_x[col].astype(str))
train_x[col] = le_item.transform(train_x[col].astype(str))
test_x[col]  = le_item.transform(test_x[col].astype(str))


# -----------------------------------------------------------
# (C) 変数 date の前処理
# -----------------------------------------------------------
col = 'date'
np.random.seed(71)

# 1) datetime 変換
train[col] = pd.to_datetime(train[col])
test[col]  = pd.to_datetime(test[col])

for df_src, df_feat in [(train, train_x), (test, test_x)]:
    # 基本カレンダー情報
    df_feat['year']         = df_src[col].dt.year
    df_feat['month']        = df_src[col].dt.month
    df_feat['day']          = df_src[col].dt.day
    df_feat['dow']          = df_src[col].dt.dayofweek
    # 追加クロス特徴：store×month, item×dow
    df_feat['store_month']  = df_feat['store'].astype(str) + '_' + df_feat['month'].astype(str)
    df_feat['item_dow']     = df_feat['item'].astype(str)  + '_' + df_feat['dow'].astype(str)
    # 追加周期特徴
    radians = 2 * np.pi
    df_feat['dow_sin']      = np.sin(radians * df_feat['dow'] / 7)
    df_feat['dow_cos']      = np.cos(radians * df_feat['dow'] / 7)
    df_feat['month_sin']    = np.sin(radians * (df_feat['month']-1) / 12)
    df_feat['month_cos']    = np.cos(radians * (df_feat['month']-1) / 12)

# 2) クロス特徴を Label Encoding
le_sm = LabelEncoder()
le_sm.fit(train_x['store_month'])
train_x['store_month'] = le_sm.transform(train_x['store_month'])
test_x['store_month']  = le_sm.transform(test_x['store_month'])

le_id = LabelEncoder()
le_id.fit(train_x['item_dow'])
train_x['item_dow'] = le_id.transform(train_x['item_dow'])
test_x['item_dow']  = le_id.transform(test_x['item_dow'])

# 3) テストデータ列揃え
test_x = test_x.reindex(columns=train_x.columns, fill_value=0)

# 4) 不要列 drop
train_x.drop([col], axis=1, inplace=True)
test_x .drop([col], axis=1, inplace=True)


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

