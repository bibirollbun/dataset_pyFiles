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
# (A) 変数 store の前処理（修正版）
# -----------------------------------------------------------
col = 'store'

# 店舗を3グループに分割する関数
def store_group(store):
    if store in [1, 2, 3]:
        return 11
    elif store in [4, 5, 6]:
        return 12
    else:
        return 13

new_col = 'group'

#applyで各行に関数を適用
train_x[new_col] = train_x[col].apply(store_group)
test_x[new_col] = test_x[col].apply(store_group)

print(train_x.head(10))

# ラベルエンコーディング
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


#store × item の組合せ特徴量（2種類の特徴量を使用するため、処理を変えている）
train_x['store_item'] = train_x['store'].astype(str) + '_' + train_x['item'].astype(str)
test_x['store_item']  = test_x['store'].astype(str) + '_' + test_x['item'].astype(str)

le_si = LabelEncoder()
le_si.fit(train_x['store_item'])
train_x['store_item'] = le_si.transform(train_x['store_item'])
test_x['store_item']  = le_si.transform(test_x['store_item'])


# -----------------------------------------------------------
# (C) 変数 date の前処理
# -----------------------------------------------------------
col = 'date'

# 日付文字列をPandasの日時型(datetime: dt)に変換
train[col] = pd.to_datetime(train[col])
test[col]  = pd.to_datetime(test[col])

#新特徴「日付」
new_col = 'day' 
train_x[new_col] = train[col].dt.day
test_x[new_col]  = test[col].dt.day

#新特徴「年」
#new_col = 'year' 
#train_x[new_col] = train[col].dt.year
#test_x[new_col]  = test[col].dt.year

#「平日/休日」用関数(土日なら1,平日なら2)
def get_weekend(dayofweek):
    if dayofweek in [5, 6]:
        return 1
    else:
        return 2
        
# 新特徴「平日/休日」
#new_col = 'wd/we' # ⇔ weekday(平日)/weekend(週末)
#train_x[new_col] = train[col].dt.dayofweek.apply(get_weekend)
#test_x[new_col]  = test[col].dt.dayofweek.apply(get_weekend)


#「季節」用の関数(7=春 8=夏 9=秋 10=冬)
def get_season(month):
    if month in [3, 4, 5]:
        return 7
    elif month in [6, 7, 8]:
        return 8
    elif month in [9, 10, 11]:
        return 9
    else:
        return 10

#新特徴「季節」
new_col = 'season'
train_x[new_col] = train[col].dt.month.apply(get_season)
test_x[new_col]  = test[col].dt.month.apply(get_season)
#print(train_x.head(60)) # 季節が変わるまで（3月のデータに入るまで）のデータを表示(確認用)

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

