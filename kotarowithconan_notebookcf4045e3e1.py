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
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

col = 'date'

# --- 1. 日付変換 ---
# 日付文字列をPandasの日時型(datetime: dt)に変換
# この時点で train_x, test_x には 'date' 列が存在します
train[col] = pd.to_datetime(train[col])
test[col]  = pd.to_datetime(test[col])

# --- 2. 時間に関する特徴量を追加 ---
# 年、月、日、週番号を新しい特徴量として追加
# これにより、モデルがトレンドや季節性を学習できるようになります
for df in [train_x, test_x]:
    original_date_series = pd.to_datetime(df[col]) # df[col] を日付型に変換
    df['year'] = original_date_series.dt.year
    df['month'] = original_date_series.dt.month
    df['day'] = original_date_series.dt.day
    df['weekofyear'] = original_date_series.dt.isocalendar().week.astype(int)

# --- 3. 曜日の特徴量作成とOne-Hot Encoding ---
new_col = 'dow' # ⇔ day of the week
train_x[new_col] = pd.to_datetime(train_x[col]).dt.dayofweek
test_x[new_col]  = pd.to_datetime(test_x[col]).dt.dayofweek

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(train_x[[new_col]])

# 変換後のDataFrameを作成し、元のDataFrameに連結
encoded_cols = encoder.get_feature_names_out([new_col])
train_encoded = pd.DataFrame(encoder.transform(train_x[[new_col]]), columns=encoded_cols)
test_encoded  = pd.DataFrame(encoder.transform(test_x[[new_col]]), columns=encoded_cols)
train_encoded.index = train_x.index
test_encoded.index = test_x.index
train_x = pd.concat([train_x, train_encoded], axis=1)
test_x  = pd.concat([test_x, test_encoded], axis=1)

# --- 5. 結果の確認 ---
print("--- 前処理後の特徴量 (先頭5行) ---")
print(train_x.head())

# --- 4. 不要な列の削除 ---
# 元の 'date' 列と、エンコードに使った 'dow' 列を削除
train_x = train_x.drop(columns=[col, new_col])
test_x  = test_x.drop(columns=[col, new_col])





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

