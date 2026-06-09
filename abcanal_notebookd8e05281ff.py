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


df = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv')
df.head()


df.describe()


# データの読み込み（先頭から50万件のみ使用）
df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    parse_dates=["pickup_datetime"],
    nrows=500000
)

# 2009年以降のデータに絞る（古い記録を除外）
df = df[df['pickup_datetime'] >= '2009-01-01']

# 年ごとの情報を抽出（新たに "year" 列を作成）
df['year'] = df['pickup_datetime'].dt.year

# 年ごとの平均運賃を集計
yearly_avg_fare = df.groupby('year')['fare_amount'].mean()

# 集計結果を表示
print(yearly_avg_fare)


# データをグループ化
yearly_stats = df.groupby('year')['fare_amount'].agg(['mean', 'sum', 'count'])
#表示
print(yearly_stats)


# 外れ値を削除
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 200)]


#再度グループ化
yearly_stats = df.groupby('year')['fare_amount'].agg(['mean', 'sum', 'count'])
#表示
print(yearly_stats)


# matplotlibで仮説２に関するグラフを表示する
import matplotlib.pyplot as plt

yearly_avg_fare.plot(kind='bar', title='Yearly Average Taxi Fare (2009~2015)')
plt.xlabel('Year')
plt.ylabel('Average Fare [$]')
plt.grid(True)
plt.tight_layout()
plt.show()


# データのクリーニング：乗車人数が0以下、または6人超の異常値を除外
# 一般的なタクシーの定員（1〜6人）を考慮
df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]

# 乗車人数ごとの平均運賃を算出
fare_by_passenger = df.groupby('passenger_count')['fare_amount'].mean()

# 結果を表示（各乗車人数における平均運賃）
print(fare_by_passenger)


# データをグループ化
fare_stats = df.groupby('passenger_count')['fare_amount'].agg(['mean', 'median', 'std', 'count'])
#表示
print(fare_stats)


#グラフを表示する
fare_by_passenger.plot(kind='bar', title='Average Fare by Passenger Count', color='skyblue')
plt.xlabel('Passenger Count')
plt.ylabel('Average Fare [$]')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()



# 日時情報を datetime 型に変換（文字列は扱うことができない）
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

# 曜日(0=月曜 ,..., 6=日曜とする)を新しい列 'weekday' として抽出
df['weekday'] = df['pickup_datetime'].dt.dayofweek

# 曜日ごとの平均運賃を算出
avg_fare_by_weekday = df.groupby('weekday')['fare_amount'].mean()

# 曜日の数値(0〜6)を文字ラベル(Mon〜Sun)に置き換え(表示時にはわかりにくいから)
weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
avg_fare_by_weekday.index = [weekday_labels[i] for i in avg_fare_by_weekday.index]

# 結果を表示(曜日ごとの平均タクシー運賃)
print(avg_fare_by_weekday)


# 曜日ごとの平均タクシー運賃を棒グラフで表示
avg_fare_by_weekday.plot(kind='bar', title='Average Taxi Fare by Weekday')
plt.ylabel('Average Fare [$]')
plt.xlabel('Weekday')
plt.show()


# データ読み込み（最初の50万件のみを読み込む）
# pickup_datetime列はdatetime型として読み込み
df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", parse_dates=["pickup_datetime"], nrows=500000)

#  ハバーサイン関数（緯度・経度から地球面上での最短距離を計算）
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 地球の半径（km）
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1) # 緯度差（ラジアン）
    dlambda = np.radians(lon2 - lon1) # 経度差（ラジアン）

    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

#  出発地点(pickup)と到着地点(dropoff)から距離[km]を計算し、新しい列 'distance_km' を作成
df["distance_km"] = haversine(
    df["pickup_latitude"], df["pickup_longitude"],
    df["dropoff_latitude"], df["dropoff_longitude"]
)

# 外れ値(距離が0以下、または100km超,運賃が0以下、または200ドル超)を除外
df = df[
    (df["distance_km"] > 0) & (df["distance_km"] <= 100) &
    (df["fare_amount"] > 0) & (df["fare_amount"] <= 200)
]

# 距離を1km単位で丸めてビン(区間)を作成
df["distance_bin"] = df["distance_km"].round().astype(int)

# 各距離[km]ごとの平均運賃を集計
fare_by_distance = df.groupby("distance_bin")["fare_amount"].mean().reset_index()

# 結果表示(距離ごとの結果を確認)
print(fare_by_distance)

# グラフ表示
plt.figure(figsize=(10,6))
plt.plot(fare_by_distance["distance_bin"], fare_by_distance["fare_amount"], marker='o')
plt.title("Average Fare by Distance")
plt.xlabel("Distance (km)")
plt.ylabel("Average Fare ($)")
plt.grid(True)
plt.show()



#  ハバーサイン関数（緯度・経度から地球面上での最短距離を計算）
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 地球の半径（km）
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1) # 緯度差（ラジアン）
    dlambda = np.radians(lon2 - lon1) # 経度差（ラジアン）

    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

#  出発地点(pickup)と到着地点(dropoff)から距離[km]を計算し、新しい列 'distance_km' を作成
df["distance_km"] = haversine(
    df["pickup_latitude"], df["pickup_longitude"],
    df["dropoff_latitude"], df["dropoff_longitude"]
)
# 外れ値(27km以降を追加)を除外
df = df[
    (df["distance_km"] > 0) & (df["distance_km"] <= 27) &
    (df["fare_amount"] > 0) & (df["fare_amount"] <= 200)
]

# 距離を1km単位で丸めてビン(区間)を作成
df["distance_bin"] = df["distance_km"].round().astype(int)

# 各距離[km]ごとの平均運賃を集計
fare_by_distance = df.groupby("distance_bin")["fare_amount"].mean().reset_index()

# 結果表示(距離ごとの結果を確認)
print(fare_by_distance)

# グラフ表示
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


# 訓練データ読み込み（最初の10万件のみを読み込む）
train_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=1000000)  
# テストデータの読み込み
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")


# pickup_datetime列をdatetime型に変換
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
# 時間、曜日を特徴量として追加
train_df["hour"] = train_df["pickup_datetime"].dt.hour
train_df["day"] = train_df["pickup_datetime"].dt.dayofweek

# Na(欠損地)の削除
train_df = train_df.dropna()
# 外れ値の除外(上カラム参照)
train_df = train_df[(df["distance_km"] > 0) & (df["distance_km"] <= 27) & (train_df["fare_amount"] > 0) & (train_df["fare_amount"] < 200)]

# モデルに使う特徴量の選択(緯度経度、乗客数、時間帯、曜日)
features = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count", "hour", "day"]
X = train_df[features]
y = train_df["fare_amount"]

# 学習用・検証用データを8:2で分割
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# モデル:ランダムフォレスト
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 検証スコアの計算、表示
val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Validation RMSE: {rmse:.2f}")


# pickup_datetime を datetime 型に変換
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])
# 時間、曜日を特徴量として追加
test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["day"] = test_df["pickup_datetime"].dt.dayofweek
# 学習時と同じ特徴量セットでテストデータを整形
X_test = test_df[features]

# 学習済みモデルを使って、テストデータの運賃を予測
test_df["fare_amount"] = model.predict(X_test)
submission = test_df[["key", "fare_amount"]]
# 保存する
submission.to_csv("submission.csv", index=False)

