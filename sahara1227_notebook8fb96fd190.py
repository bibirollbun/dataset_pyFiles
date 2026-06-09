import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# データ読み込み
train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")
sample = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv")

# データの形状を表示
print("Train shape:", train.shape)
print("Test shape:", test.shape)


# datetimeの型変換と新規特徴量の作成
train["datetime"] = pd.to_datetime(train["datetime"])
test["datetime"] = pd.to_datetime(test["datetime"])

# 年、月、日、時刻、曜日の特徴量を追加
train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour
train["weekday"] = train["datetime"].dt.weekday  # 0=月曜日,6=日曜日

test["year"] = test["datetime"].dt.year
test["month"] = test["datetime"].dt.month
test["day"] = test["datetime"].dt.day
test["hour"] = test["datetime"].dt.hour
test["weekday"] = test["datetime"].dt.weekday


# 欠損値の確認
print(train.isnull().sum())
print(test.isnull().sum())


 # ライブラリ
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# 特徴量リスト（←ここが超重要！）
feature_columns = ['workingday',
                   'temp', 'humidity',
                   'year', 'month', 'hour', 'hour_group','weekday','temp_humid','atemp','temp_diff']  # 追加！

# 特徴量を追加する（例：temp_diff, is_weekend, hour_group）
train['hour_group'] = train['hour'] // 4
test['hour_group'] = test['hour'] // 4

#時間情報の細分化
for df in [train, test]:
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday  # 月:0〜日:6
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['weekday'] = df['datetime'].dt.weekday  # 月曜0〜日曜6
    df['temp_humid'] = df['temp'] * df['humidity']  # 蒸し暑さ
    df["temp_diff"] = df["temp"] - df["atemp"]  # 体感とのズレ


# datetime から特徴量を抽出（事前に datetime を datetime型に変換しておく）
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

# 例：カテゴリ変数として扱うカラムを指定
categorical_cols = ['season', 'holiday', 'workingday', 'weather', 'hour', 'weekday']

# カテゴリ変換
for col in categorical_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

# 交差特徴量例
train['hour_workingday'] = train['hour'].astype(str) + '_' + train['workingday'].astype(str)
test['hour_workingday'] = test['hour'].astype(str) + '_' + test['workingday'].astype(str)

# カテゴリとして扱う
train['hour_workingday'] = train['hour_workingday'].astype('category')
test['hour_workingday'] = test['hour_workingday'].astype('category')


X_train = train[feature_columns]
X_test = test[feature_columns]

# 目的変数を log1p で変換して学習
y = np.log1p(train['count'].values)

#y = np.log1p(train['count'])  # log変換

#  モデルの定義と学習
model = RandomForestRegressor(random_state=0)
model.fit(X_train, y)

#  予測
y_pred = model.predict(X_test)

# float64に変換
y_pred = y_pred.astype(np.float64)

# 異常値処理
y_pred = np.where(np.isinf(y_pred), 0, y_pred)
y_pred = np.where(np.isnan(y_pred), 0, y_pred)
y_pred = np.where(y_pred < 0, 0, y_pred)

# clipして指数変換
y_pred = np.clip(y_pred, a_min=-20, a_max=20)
y_pred = np.expm1(y_pred)
y_pred = np.clip(y_pred, a_min=0, a_max=1000)

# 整数化
y_pred = np.round(y_pred).astype(int)

import matplotlib.pyplot as plt

importances = model.feature_importances_
features = X_train.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.title("Feature Importances")
plt.show()


# trainから不要な列を削除
train.drop(["casual", "registered", "day"], axis=1, inplace=True)
# testから不要な列を削除（casual, registeredはもともと無く、dayを削除）
test.drop(["day"], axis=1, inplace=True)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split  # ← これを追加！
from sklearn.metrics import mean_squared_log_error


# 特徴量と目的変数の設定
X = train.drop(['datetime', 'count'], axis=1)
y = train['count'].values

# 学習・検証に分割
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# モデル作成・学習
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# X_test も X_train のカラムと順番を揃える
X_test = test[X_train.columns]  # ← この順番に注意！

# テストデータに対する予測
y_pred = model.predict(X_test)




#特徴エンジニアリングの追加
train['temp_diff'] = train['atemp'] - train['temp']
test['temp_diff'] = test['atemp'] - test['temp']



import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

model = lgb.LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(0)
    ]
)

y_pred = model.predict(X_valid)
y_pred = np.clip(y_pred, 0, None)
score = np.sqrt(mean_squared_log_error(y_valid, y_pred))
print("LightGBM RMSLE:", score)






# 予測
y_pred = model.predict(X_test)
# 負値があれば0に置き換え、四捨五入して整数化
y_pred = np.where(y_pred < 0, 0, y_pred)
y_pred = np.round(y_pred).astype(int)

# submission用データ作成
submission = test[['datetime']].copy()
submission['count'] = y_pred.astype(int)
# ファイル保存
submission.to_csv("submission.csv", index=False)


# count列の整形（すでに済んでいれば省略）
submission['count'] = y_pred.astype(int)

# datetime列の整形（小数秒を削除して正規フォーマットへ）
submission['datetime'] = pd.to_datetime(submission['datetime'])
submission['datetime'] = submission['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

# 保存
submission.to_csv('submission.csv', index=False)

# 提出用データフレームの作成
# 提出用ファイル作成
submission = sample.copy()
submission['count'] = y_pred  # y_pred は整数かつ非負であること
submission.to_csv('submission.csv', index=False)

print(submission.shape)
print(submission.head())

# 最終確認: 提出ファイルの中身と形式チェック
print(submission.columns)     # ['datetime', 'count'] であること（スペルも大事）
print(submission.dtypes)      # datetime: object or datetime64, count: int
print(submission.shape)       # (6493, 2)
print(submission.isnull().sum())  # 欠損値がないこと
print(submission.head())      # 見た目で問題がないか

# ファイルを再保存
submission.to_csv("submission.csv", index=False)

# ファイル内容確認（先頭数行）
!head submission.csv

# 行数確認（ヘッダー含めて6494行であること）
!wc -l submission.csv

y_valid_pred = model.predict(X_valid)

# 予測値を0以上にクリップ
y_valid_pred_clipped = np.clip(y_valid_pred, a_min=0, a_max=None)

# RMSLEを計算（log変換なし版）
from sklearn.metrics import mean_squared_log_error
rmsle = np.sqrt(mean_squared_log_error(y_valid, y_valid_pred_clipped))

print("RMSLE（検証データ）:", rmsle)



