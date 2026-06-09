# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# データ読み込み（適宜ファイル名と行数を調整）
df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", parse_dates=["pickup_datetime"], nrows=500000)

# 2009年以降に絞る
df = df[df['pickup_datetime'] >= '2009-01-01']

# 年ごとの列を作成
df['year'] = df['pickup_datetime'].dt.year

# 年ごとに平均運賃を集計
yearly_avg_fare = df.groupby('year')['fare_amount'].mean()

# 結果表示
print(yearly_avg_fare)


yearly_stats = df.groupby('year')['fare_amount'].agg(['mean', 'sum', 'count'])
print(yearly_stats)


# 外れ値を消す
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 200)]


yearly_stats = df.groupby('year')['fare_amount'].agg(['mean', 'sum', 'count'])
print(yearly_stats)


# グラフを表示する
import matplotlib.pyplot as plt

yearly_avg_fare.plot(kind='bar', title='Yearly Average Taxi Fare (from 2009)')
plt.xlabel('Year')
plt.ylabel('Average Fare ($)')
plt.grid(True)
plt.tight_layout()
plt.show()


# データのクリーニング（乗車人数が0以下、異常な場合を除く）
df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]  # 一般的なタクシーの定員範囲

# 乗車人数ごとの平均運賃を求める
fare_by_passenger = df.groupby('passenger_count')['fare_amount'].mean()

# 結果を表示
print(fare_by_passenger)


# 平均、中央値、標準偏差などをまとめて表示
fare_stats = df.groupby('passenger_count')['fare_amount'].agg(['mean', 'median', 'std', 'count'])
print(fare_stats)


#グラフを表示する
fare_by_passenger.plot(kind='bar', title='Average Fare by Passenger Count', color='skyblue')
plt.xlabel('Passenger Count')
plt.ylabel('Average Fare ($)')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()



# 日時をdatetime型に変換
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

# 曜日を抽出（0=月曜日, 6=日曜日）
df['weekday'] = df['pickup_datetime'].dt.dayofweek

# 曜日ごとの平均料金を計算
avg_fare_by_weekday = df.groupby('weekday')['fare_amount'].mean()

# 曜日を文字ラベルに変換（オプション）
weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
avg_fare_by_weekday.index = [weekday_labels[i] for i in avg_fare_by_weekday.index]

# 結果表示
print(avg_fare_by_weekday)


avg_fare_by_weekday.plot(kind='bar', title='Average Taxi Fare by Weekday')
plt.ylabel('Average Fare ($)')
plt.xlabel('Weekday')
plt.show()


# データ読み込み（適宜ファイル名と行数を調整）
df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", parse_dates=["pickup_datetime"], nrows=500000)

# 1. ハバーサイン関数（緯度・経度から距離計算）
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 地球の半径（km）
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# 2. 距離計算
df["distance_km"] = haversine(
    df["pickup_latitude"], df["pickup_longitude"],
    df["dropoff_latitude"], df["dropoff_longitude"]
)

df = df[
    (df["distance_km"] > 0) & (df["distance_km"] <= 100) &
    (df["fare_amount"] > 0) & (df["fare_amount"] <= 200)
]

# 3. 距離を1km単位に丸めてグループ化
df["distance_bin"] = df["distance_km"].round().astype(int)

# 4. 各距離ごとの平均料金を計算
fare_by_distance = df.groupby("distance_bin")["fare_amount"].mean().reset_index()

# 5. 結果表示
print(fare_by_distance)

# 6. グラフ表示（任意）
plt.figure(figsize=(10,6))
plt.plot(fare_by_distance["distance_bin"], fare_by_distance["fare_amount"], marker='o')
plt.title("Average Fare by Distance (0-35 km)")
plt.xlabel("Distance (km)")
plt.ylabel("Average Fare ($)")
plt.grid(True)
plt.show()



# 2. ハバーサイン関数（緯度・経度から距離計算）
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 地球の半径（km）
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# 3. 距離計算
df["distance_km"] = haversine(
    df["pickup_latitude"], df["pickup_longitude"],
    df["dropoff_latitude"], df["dropoff_longitude"]
)

# 4. 27km以降外れ値が目立つため、除去（距離0〜27km、料金0〜200ドル）
df = df[
    (df["distance_km"] > 0) & (df["distance_km"] <= 27) &
    (df["fare_amount"] > 0) & (df["fare_amount"] <= 200)
]

# 5. 距離を1km単位に丸めてグループ化
df["distance_bin"] = df["distance_km"].round().astype(int)

# 6. 各距離ごとの平均料金を計算
fare_by_distance = df.groupby("distance_bin")["fare_amount"].mean().reset_index()

# 7. 結果表示
print(fare_by_distance)

# 8. グラフ表示（任意）
plt.figure(figsize=(10,6))
plt.plot(fare_by_distance["distance_bin"], fare_by_distance["fare_amount"], marker='o')
plt.title("Average Fare by Distance (0-27 km)")
plt.xlabel("Distance (km)")
plt.ylabel("Average Fare ($)")
plt.grid(True)
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


train_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=1000000)  # 1M行だけ使用（メモリ節約）
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")


# 日時処理
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
train_df["hour"] = train_df["pickup_datetime"].dt.hour
train_df["day"] = train_df["pickup_datetime"].dt.dayofweek

# 欠損削除・異常除外
train_df = train_df.dropna()
train_df = train_df[(train_df["fare_amount"] > 0) & (train_df["fare_amount"] < 200)]

# 特徴量選択
features = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count", "hour", "day"]
X = train_df[features]
y = train_df["fare_amount"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 検証スコア
val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Validation RMSE: {rmse:.2f}")



# テストデータに同様の特徴追加
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])
test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["day"] = test_df["pickup_datetime"].dt.dayofweek
X_test = test_df[features]

# 予測と保存
test_df["fare_amount"] = model.predict(X_test)
submission = test_df[["key", "fare_amount"]]
submission.to_csv("submission.csv", index=False)

