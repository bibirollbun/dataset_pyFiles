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
#col = 'date'

# ベースライン：使用しない（※テスト側が全て未知カテゴリになるためそのままの使用は困難）
#train_x = train_x.drop([col], axis=1)
#test_x = test_x.drop([col], axis=1)


# -----------------------------------------------------------
# (C) 変数 date の前処理　★ここだけを書き替える★
# -----------------------------------------------------------
col = "date"

# 1. 日付を datetime へ
train[col] = pd.to_datetime(train[col])
test[col]  = pd.to_datetime(test[col])

# 2. 基本派生
for df in (train_x, test_x):
    base = train if df is train_x else test           # 対応する元 DataFrame
    df["year"]        = base[col].dt.year
    df["month"]       = base[col].dt.month
    df["day"]         = base[col].dt.day
    df["dow"]         = base[col].dt.dayofweek        # 0=Mon
    df["weekofyear"]  = base[col].dt.isocalendar().week.astype(int)
    df["dayofyear"]   = base[col].dt.dayofyear
    df["quarter"]     = base[col].dt.quarter
    
    # フラグ系
    df["is_weekend"]      = (df["dow"] >= 5).astype(int)
    df["is_month_start"]  = base[col].dt.is_month_start.astype(int)
    df["is_month_end"]    = base[col].dt.is_month_end.astype(int)
    df["is_quarter_start"]= base[col].dt.is_quarter_start.astype(int)
    df["is_quarter_end"]  = base[col].dt.is_quarter_end.astype(int)
    df["is_year_start"]   = base[col].dt.is_year_start.astype(int)
    df["is_year_end"]     = base[col].dt.is_year_end.astype(int)

    # 周期（サイクリック）エンコーディング
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"]   / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"]   / 7)

# 3. 未加工の date 列は不要なので削除
train_x = train_x.drop([col], axis=1)
test_x  = test_x.drop([col], axis=1)



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







