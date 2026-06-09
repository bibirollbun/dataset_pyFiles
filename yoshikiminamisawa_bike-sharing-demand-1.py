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


# 全期間におけるレンタル数
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=["datetime"])

plt.plot(df["datetime"], df["count"])
plt.xlabel("datetime")
plt.ylabel("count")
plt.title("Bike Rentals Over Time")
plt.show()


# 「月」「時間」「曜日」ごとの平均レンタル数
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=["datetime"])

# datetimeから「月」「時間」「曜日」を取り出す
df["month"] = df["datetime"].dt.month
df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.day_name()

# 月ごとの平均レンタル数のヒストグラム作成
monthly_mean = df.groupby("month")["count"].mean()
plt.bar(monthly_mean.index, monthly_mean.values)
plt.xlabel("month")
plt.ylabel("count")
plt.title("Average Count by Month")
plt.show()

# 時間ごとの平均レンタル数のヒストグラム作成
hourly_mean = df.groupby("hour")["count"].mean()
plt.bar(hourly_mean.index, hourly_mean.values)
plt.xlabel("hour")
plt.ylabel("count")
plt.title("Average Count by Hour")
plt.show()

# 曜日ごとの平均レンタル数のヒストグラム作成
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_mean = df.groupby("weekday")["count"].mean().reindex(weekday_order)
plt.bar(weekday_mean.index, weekday_mean.values)
plt.xlabel("weekday")
plt.ylabel("count")
plt.title("Average Count by Weekday")
plt.show()



# 天気ごとの平均レンタル数
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=["datetime"])

# 番号を天気の説明に変換
weather_map = {
    1: "Clear / Few clouds",
    2: "Mist / Cloudy",
    3: "Light Rain / Snow",
    4: "Heavy Rain / Fog"
}
df["weather_desc"] = df["weather"].map(weather_map)

# ヒストグラム作成
weather_mean = df.groupby("weather_desc")["count"].mean()
plt.bar(weather_mean.index, weather_mean.values)
plt.xlabel("Weather")
plt.ylabel("Count")
plt.title("Average Count by Weather")
plt.show()



# temp, atempごとの平均レンタル数
import pandas as pd
import matplotlib.pyplot as plt

# 1℃間隔で表示
df["temp_bin"] = df["temp"].round()
df["atemp_bin"] = df["atemp"].round()

# tempごとの平均レンタル数のヒストグラム作成
plt.bar(df.groupby("temp_bin")["count"].mean().index,
        df.groupby("temp_bin")["count"].mean().values)
plt.xlabel("temp (°C)")
plt.ylabel("count")
plt.title("Average Count by Temperature")
plt.show()

# atempごとの平均レンタル数のヒストグラム作成
plt.bar(df.groupby("atemp_bin")["count"].mean().index,
        df.groupby("atemp_bin")["count"].mean().values)
plt.xlabel("atemp (°C)")
plt.ylabel("count")
plt.title("Average Count by Feels-like Temperature")
plt.show()


# humidity, windspeedごとの平均レンタル数
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=["datetime"])

# 1％，1km/h間隔で表示
df["humidity_bin"] = df["humidity"]
df["windspeed_bin"] = df["windspeed"]

# humidityごとの平均レンタル数のヒストグラム作成
plt.bar(df.groupby("humidity_bin")["count"].mean().index,
        df.groupby("humidity_bin")["count"].mean().values)
plt.xlabel("humidity (%)")
plt.ylabel("count")
plt.title("Average Count by Humidity")
plt.show()

# windspeedごとの平均レンタル数のヒストグラム作成
plt.bar(df.groupby("windspeed_bin")["count"].mean().index,
        df.groupby("windspeed_bin")["count"].mean().values)
plt.xlabel("windspeed")
plt.ylabel("count")
plt.title("Average Count by Wind Speed")
plt.show()



# holidayとworkingdayでの平均レンタル数
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=["datetime"])

# 0:holiday, 1:workingday
df["day_type"] = df["workingday"].map({0: "Holiday/Weekend", 1: "Working Day"})

# holidayとworkingdayでの平均レンタル数を計算
mean_counts = df.groupby("day_type")["count"].mean()

# 棒グラフ作成
plt.bar(mean_counts.index, mean_counts.values)
plt.xlabel("Day Type")
plt.ylabel("Average Count")
plt.title("Working Day vs Holiday/Weekend")
plt.show()



# casualとregisteredでの平均レンタル数
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=["datetime"])

# casualとregisteredでの平均レンタル数を計算
mean_counts = {
    "Casual Users": df["casual"].mean(),
    "Registered Users": df["registered"].mean()
}

# 棒グラフ作成
plt.bar(mean_counts.keys(), mean_counts.values())
plt.xlabel("User Type")
plt.ylabel("Average Count")
plt.title("Average Bike Rentals by User Type")
plt.show()



# 会員×休日, 会員×平日，非会員×休日, 非会員×平日の４グループにおける「月」ごとの平均レンタル数
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=["datetime"])

# 月を抽出
df["month"] = df["datetime"].dt.month

# グラフ描画設定
plt.figure(figsize=(10,6))

# 各条件をループして平均を描く
for user, col in [("registered", "registered"), ("casual", "casual")]:
    for cond, label in [(df["workingday"] == 0, "holiday"), (df["workingday"] == 1, "workingday")]:
        monthly = df[cond].groupby("month")[col].mean()
        plt.plot(monthly.index, monthly.values, marker="o", label=f"{user} + {label}")

# 軸・凡例など
plt.xlabel("Month")
plt.ylabel("Average Rentals")
plt.title("Monthly Count by User Type and Day Type")
plt.legend()
plt.grid(True)
plt.show()






# 会員×休日, 会員×平日，非会員×休日, 非会員×平日の４グループにおける「時間」ごとの平均レンタル数
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=["datetime"])

# 時間を抽出
df["hour"] = df["datetime"].dt.hour

# グラフ描画設定
plt.figure(figsize=(10,6))

# 各条件をループして平均を描く
for user, col in [("registered", "registered"), ("casual", "casual")]:
    for cond, label in [(df["workingday"] == 0, "holiday"), (df["workingday"] == 1, "workingday")]:
        hour = df[cond].groupby("hour")[col].mean()
        plt.plot(hour.index, hour.values, marker="o", label=f"{user} + {label}")

# 軸・凡例など
plt.xlabel("Hour")
plt.ylabel("Average Rentals")
plt.title("Hour Count by User Type and Day Type")
plt.legend()
plt.grid(True)
plt.show()


# 休日と平日における「時間」ごとの平均レンタル数
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=["datetime"])

# 時間を抽出
df["hour"] = df["datetime"].dt.hour

# グラフ描画設定
plt.figure(figsize=(10,6))

# holidayとworkingdayで2分割して平均を描く
for label, cond in [("Holiday", df["workingday"] == 0),
                    ("Workingday", df["workingday"] == 1)]:
    hour_avg = df[cond].groupby("hour")["count"].mean()
    plt.plot(hour_avg.index, hour_avg.values, marker="o", label=label)

# 軸・凡例など
plt.xlabel("Hour")
plt.ylabel("Average Rentals")
plt.title("Hourly Average Rentals: Holiday vs Workingday")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 「時間」ごとの平均レンタル数の曜日での比較
import pandas as pd
import matplotlib.pyplot as plt
import calendar

# データ読み込み
df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=["datetime"])

# 曜日を抽出（0=Monday, 6=Sunday）
df["weekday"] = df["datetime"].dt.weekday
df["weekday_name"] = df["weekday"].apply(lambda x: calendar.day_name[x])

# 時間を抽出
df["hour"] = df["datetime"].dt.hour

# グラフ描画設定
plt.figure(figsize=(12,6))

# 曜日順（Monday→Sunday）
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# 各曜日ごとに平均レンタル数を描画
for day in order:
    hour_avg = df[df["weekday_name"] == day].groupby("hour")["count"].mean()
    plt.plot(hour_avg.index, hour_avg.values, marker="o", label=day)

# 軸・凡例など
plt.xlabel("Hour")
plt.ylabel("Average Rentals")
plt.title("Hourly Average Rentals by Day of Week")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 会員と非会員における「時間」ごとの平均レンタル数
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=["datetime"])

# 時間を抽出
df["hour"] = df["datetime"].dt.hour

# 各ユーザータイプごとに平均を計算
hourly_casual = df.groupby("hour")["casual"].mean()
hourly_registered = df.groupby("hour")["registered"].mean()

# グラフ描画
plt.figure(figsize=(10,6))
plt.plot(hourly_casual.index, hourly_casual.values, marker="o", label="Casual Users", color="orange")
plt.plot(hourly_registered.index, hourly_registered.values, marker="o", label="Registered Users", color="blue")

# 軸・凡例など
plt.xlabel("Hour of Day")
plt.ylabel("Average Rentals")
plt.title("Hourly Average Rentals: Casual vs Registered Users")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# 会員と非会員における「時間」ごとの平均レンタル数の曜日での比較
import pandas as pd
import matplotlib.pyplot as plt
import calendar

# データ読み込み
df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=["datetime"])

# 時間・曜日を抽出
df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.weekday
df["weekday_name"] = df["weekday"].apply(lambda x: calendar.day_name[x])

# 曜日順（Monday→Sunday）
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# グラフを曜日ごとに並べて表示
fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()

for i, day in enumerate(order):
    ax = axes[i]
    day_data = df[df["weekday_name"] == day]
    
    for user in ["registered", "casual"]:
        hour_avg = day_data.groupby("hour")[user].mean()
        ax.plot(hour_avg.index, hour_avg.values, marker="o", label=user)
    
    ax.set_title(day)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Rentals")
    ax.grid(True)
    ax.legend()

# 余分なプロット枠を非表示（曜日は7日なので）
fig.delaxes(axes[-1])

plt.suptitle("Hourly Average Rentals by Day of Week and User Type", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()



# レンタル数の分布の可視化
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")

train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=["datetime"])

train["hour"] = train["datetime"].dt.hour
train["weekday"] = train["datetime"].dt.weekday

# countの分布の可視化
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.histplot(train["count"], bins=50, kde=True)
plt.title("Histogram of count (raw)")

# log(1+count)の分布の可視化
plt.subplot(1,2,2)
sns.histplot(np.log1p(train["count"]), bins=50, kde=True)
plt.title("Histogram of log1p(count)")
plt.show()

# countの時間ごとの分布の可視化
plt.figure(figsize=(12,6))
sns.boxplot(x=train["hour"], y=train["count"])
plt.title("Count distribution by hour")
plt.show()

# countの曜日ごとの分布の可視化
plt.figure(figsize=(12,6))
sns.boxplot(x=train["weekday"], y=train["count"])
plt.title("Count distribution by weekday")
plt.show()


# 会員×休日, 会員×平日，非会員×休日, 非会員×平日の４グループにおけるcountの時間ごとの分布の可視化
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=False)

# --- Casual × Workingday ---
sns.boxplot(
    x=train[train["workingday"] == 1]["hour"],
    y=train[train["workingday"] == 1]["casual"],
    palette="viridis",
    ax=axes[0, 0]
)
axes[0, 0].set_title("Casual users by hour (Workingday)")
axes[0, 0].set_xlabel("Hour")
axes[0, 0].set_ylabel("Casual count")

# --- Casual × Holiday ---
sns.boxplot(
    x=train[train["holiday"] == 1]["hour"],
    y=train[train["holiday"] == 1]["casual"],
    palette="viridis",
    ax=axes[0, 1]
)
axes[0, 1].set_title("Casual users by hour (Holiday)")
axes[0, 1].set_xlabel("Hour")
axes[0, 1].set_ylabel("Casual count")

# --- Registered × Workingday ---
sns.boxplot(
    x=train[train["workingday"] == 1]["hour"],
    y=train[train["workingday"] == 1]["registered"],
    palette="viridis",
    ax=axes[1, 0]
)
axes[1, 0].set_title("Registered users by hour (Workingday)")
axes[1, 0].set_xlabel("Hour")
axes[1, 0].set_ylabel("Registered count")

# --- Registered × Holiday ---
sns.boxplot(
    x=train[train["holiday"] == 1]["hour"],
    y=train[train["holiday"] == 1]["registered"],
    palette="viridis",
    ax=axes[1, 1]
)
axes[1, 1].set_title("Registered users by hour (Holiday)")
axes[1, 1].set_xlabel("Hour")
axes[1, 1].set_ylabel("Registered count")

plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit

# 1. データ読み込み
train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv",
                    parse_dates=["datetime"])

train = train.sort_values("datetime").reset_index(drop=True)

# 2. 特徴量作成
for df in [train]:
    df["hour"]    = df["datetime"].dt.hour
    df["day"]     = df["datetime"].dt.day
    df["month"]   = df["datetime"].dt.month
    df["year"]    = df["datetime"].dt.year
    df["weekday"] = df["datetime"].dt.weekday

    # peak_reg（registered用ピーク）
    df["peak_reg"] = 0
    mask_work = (df["workingday"] == 1)
    df.loc[mask_work & df["hour"].isin([7, 8, 9, 16, 17, 18]), "peak_reg"] = 1

    # peak_cas（casual用ピーク）
    df["peak_cas"] = 0
    df.loc[df["hour"].between(11, 17), "peak_cas"] = 1


# 3. windspeed == 0 の補正
from sklearn.ensemble import RandomForestRegressor

def fill_windspeed(df):
    df_w0    = df[df["windspeed"] == 0]
    df_wnot0 = df[df["windspeed"] != 0]

    if len(df_w0) == 0:
        return df

    model_wind = RandomForestRegressor(
        random_state=42, n_estimators=100, n_jobs=-1
    )
    cols = ["season", "weather", "humidity", "temp", "atemp", "month", "year"]
    model_wind.fit(df_wnot0[cols], df_wnot0["windspeed"])
    df_w0.loc[:, "windspeed"] = model_wind.predict(df_w0[cols])

    df = pd.concat([df_wnot0, df_w0]).sort_index()
    return df

train = fill_windspeed(train)

# 4. 特徴量
base_features = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "year", "hour", "day", "month", "weekday"
]

features_casual = base_features + ["peak_cas"]
features_reg    = base_features + ["peak_reg"]

X_casual = train[features_casual]
X_reg    = train[features_reg]

y_casual_log = np.log1p(train["casual"])
y_reg_log    = np.log1p(train["registered"])

# TimeSeriesSplit（時系列を保った5分割）
tscv = TimeSeriesSplit(n_splits=5)

# RMSLE
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# casual / registered を別々に学習してから合算して RMSLE を計算
def evaluate_pair(model_casual, model_reg, name):
    rmsles = []

    for train_idx, val_idx in tscv.split(train):
        # ====== 学習・検証分割 ======
        Xc_tr, Xc_val = X_casual.iloc[train_idx], X_casual.iloc[val_idx]
        Xr_tr, Xr_val = X_reg.iloc[train_idx],    X_reg.iloc[val_idx]

        yc_tr, yc_val = y_casual_log.iloc[train_idx], y_casual_log.iloc[val_idx]
        yr_tr, yr_val = y_reg_log.iloc[train_idx],    y_reg_log.iloc[val_idx]

        # ====== casual モデル学習 ======
        model_casual.fit(Xc_tr, yc_tr)

        # ====== registered モデル学習 ======
        model_reg.fit(Xr_tr, yr_tr)

        # ====== 予測 ======
        pred_casual    = np.expm1(model_casual.predict(Xc_val))
        pred_registered = np.expm1(model_reg.predict(Xr_val))

        y_true = np.expm1(yc_val) + np.expm1(yr_val)
        y_pred = np.clip(pred_casual + pred_registered, 0, None)

        rmsles.append(rmsle(y_true, y_pred))

    print(f"{name:20s}  RMSLE = {np.mean(rmsles):.5f}")



# RandomForest
from sklearn.ensemble import RandomForestRegressor

rf_casual = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf_reg = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

evaluate_pair(rf_casual, rf_reg, "RandomForest")


# GradientBoosting
from sklearn.ensemble import GradientBoostingRegressor

gb_casual = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_reg = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

evaluate_pair(gb_casual, gb_reg, "GradientBoosting")


# ExtraTrees
from sklearn.ensemble import ExtraTreesRegressor

et_casual = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

et_reg = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

evaluate_pair(et_casual, et_reg, "ExtraTrees")



# KNN
from sklearn.neighbors import KNeighborsRegressor

knn_casual = KNeighborsRegressor(
    n_neighbors=10,
    weights="distance",
    n_jobs=-1
)

knn_reg = KNeighborsRegressor(
    n_neighbors=10,
    weights="distance",
    n_jobs=-1
)

evaluate_pair(knn_casual, knn_reg, "KNN")



# LinearRegression
from sklearn.linear_model import LinearRegression

lr_casual = LinearRegression()
lr_reg    = LinearRegression()

evaluate_pair(lr_casual, lr_reg, "LinearRegression")



# LightGBM
import lightgbm as lgb

lgb_casual = lgb.LGBMRegressor(
    objective = "rmse",
    learning_rate = 0.04,
    num_leaves = 20,
    n_estimators = 1000,
    subsample = 1.0,
    colsample_bytree = 0.8,
    min_child_samples = 20,
    reg_alpha = 0.0,
    reg_lambda = 0.0,
    random_state = 42,
    verbose = -1
)

lgb_reg = lgb.LGBMRegressor(
    objective = "rmse",
    learning_rate = 0.04,
    num_leaves = 20,
    n_estimators = 1000,
    subsample = 1.0,
    colsample_bytree = 0.8,
    min_child_samples = 20,
    reg_alpha = 0.0,
    reg_lambda = 0.0,
    random_state = 42,
    verbose = -1
)

evaluate_pair(lgb_casual, lgb_reg, "LightGBM")



# CatBoost
from catboost import CatBoostRegressor

cb_casual = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    random_state=42,
    verbose=0
)

cb_reg = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    random_state=42,
    verbose=0
)

evaluate_pair(cb_casual, cb_reg, "CatBoost")



import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
import lightgbm as lgb


# データ読み込み
train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=["datetime"])
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv", parse_dates=["datetime"])

# 特別日の補正
def apply_special_days(df):
    df['datetime'] = pd.to_datetime(df['datetime'])

    dates = [
        (pd.Timestamp(2011, 4, 15), 1, 0),  # Tax day
        (pd.Timestamp(2012, 4, 16), 1, 0),  # Tax day
        (pd.Timestamp(2011, 11, 25), 0, 1),  # Thanksgiving Friday
        (pd.Timestamp(2012, 11, 23), 0, 1),  # Thanksgiving Friday
        (pd.Timestamp(2011, 12, 24), 0, 1),  # Christmas
        (pd.Timestamp(2012, 12, 24), 0, 1),  # Christmas
        (pd.Timestamp(2011, 12, 26), 0, 1),  # Christmas
        (pd.Timestamp(2012, 12, 26), 0, 1),  # Christmas
        (pd.Timestamp(2011, 12, 31), 0, 1),  # New Year’s Eve
        (pd.Timestamp(2012, 12, 31), 0, 1),  # New Year’s Eve
    ]

    for date, workingday_value, holiday_value in dates:
        mask = df['datetime'].dt.date == date.date()
        df.loc[mask, 'workingday'] = workingday_value
        df.loc[mask, 'holiday'] = holiday_value

    return df

# 特別日の補正をtrainとtestに適用
train = apply_special_days(train)
test = apply_special_days(test)

# 特徴量作成
for df in [train, test]:
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["year"]  = df["datetime"].dt.year
    df["weekday"] = df["datetime"].dt.weekday

    # peak_reg（registered用ピーク）
    df["peak_reg"] = 0
    mask_work = (df["workingday"] == 1)
    df.loc[mask_work & df["hour"].isin([7,8,9,16,17,18]), "peak_reg"] = 1

    # peak_cas（casual用ピーク）
    df["peak_cas"] = 0
    df.loc[df["hour"].between(11,17), "peak_cas"] = 1

    # 時間の周期（24時間）
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # 月の周期（12ヶ月）
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# windspeedの補正を追加
def fill_windspeed_with_source(source_df, target_df):
    df_wnot0 = source_df[source_df["windspeed"] != 0]
    if len(df_wnot0) == 0:
        return target_df

    model_wind = RandomForestRegressor(random_state=42, n_estimators=100)
    cols = ["season", "weather", "humidity", "temp", "atemp", "month", "year"]
    model_wind.fit(df_wnot0[cols], df_wnot0["windspeed"])

    # windspeed == 0 または windspeed >= 55 を補完対象に
    df_w0 = target_df[(target_df["windspeed"] == 0) | (target_df["windspeed"] >= 55)]

    if len(df_w0) > 0:
        target_df.loc[df_w0.index, "windspeed"] = model_wind.predict(df_w0[cols])

    return target_df

# ベースとなる特徴量
base_features = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "year", "hour", "day", "month", "weekday",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos"
]
# casual用の特徴量
features_casual = base_features + ["peak_cas"]
# registered用の特徴量
features_reg     = base_features + ["peak_reg"]




# モデル定義（LightGBM）
lgb_params = dict(
    objective = "rmse",
    learning_rate = 0.04,
    num_leaves = 20,
    n_estimators = 1000,
    subsample = 1.0,
    colsample_bytree = 0.8,
    min_child_samples = 20,
    reg_alpha = 0.0,
    reg_lambda = 0.0,
    random_state = 42,
    verbose = -1
)

model_casual_lgb = lgb.LGBMRegressor(**lgb_params)
model_reg_lgb    = lgb.LGBMRegressor(**lgb_params)

# 予測結果を入れていく
pred_frames = []

# 擬似ラベルを段階的に蓄積（最初は元のtrainのみ）
aug_train = train.copy()

# テストに含まれる (year, month) を古い順に処理
for (y, m) in sorted(test[["year","month"]].drop_duplicates().itertuples(index=False, name=None)):
    # この月のテスト対象（20日〜末日）
    test_sub  = test[(test["year"] == y) & (test["month"] == m)].copy()
    cutoff_ts = pd.Timestamp(y, m, 20, 0, 0)

    # 学習に使えるのは「cutoffより前」のaug_trainのみ
    train_sub = aug_train[aug_train["datetime"] < cutoff_ts].copy()

    # windspeed を時系列順に安全に補完
    train_sub = fill_windspeed_with_source(train_sub, train_sub)
    test_sub  = fill_windspeed_with_source(train_sub, test_sub)

    # 学習データ（casual）
    X_train_casual = train_sub[features_casual]
    y_train_casual = np.log1p(train_sub["casual"])

    # 学習データ（registered）
    X_train_reg = train_sub[features_reg]
    y_train_reg = np.log1p(train_sub["registered"])

    # モデル学習 (casual)
    model_casual_lgb.fit(X_train_casual, y_train_casual)
    
    # モデル学習 (registered)
    model_reg_lgb.fit(X_train_reg, y_train_reg)

    # テスト (casual)
    X_test_casual = test_sub[features_casual]
    # テスト (registered)
    X_test_reg = test_sub[features_reg]

    # 予測 (casual)
    pred_casual = np.expm1(model_casual_lgb.predict(X_test_casual))
    
    # 予測 (registered)
    pred_registered = np.expm1(model_reg_lgb.predict(X_test_reg))

    pred_total = np.clip(pred_casual + pred_registered, 0, None)
    test_sub["count"] = pred_total

    # 予測を保存（提出用）
    pred_frames.append(test_sub[["datetime","count"]])

    # この月の予測（20–末）を擬似ラベルとして将来の学習に使うために，aug_trainへ追加
    pseudo_rows = test_sub[["datetime","count"] + base_features + ["peak_cas"] + ["peak_reg"]].copy()
    pseudo_rows["casual"] = pred_casual
    pseudo_rows["registered"] = pred_registered

    # 特別日の補正をpseudo_rowsに適用
    pseudo_rows = apply_special_days(pseudo_rows)
    
    aug_train = pd.concat([aug_train, pseudo_rows], ignore_index=True)

# 提出ファイル作成
submission = pd.concat(pred_frames, ignore_index=True).sort_values("datetime")
submission.to_csv("submission.csv", index=False)
print("Saved: submission.csv ; shape:", submission.shape)


import matplotlib.pyplot as plt
import pandas as pd

# ============ 1. casual の Feature Importance 取得 ============
fi_casual = pd.DataFrame({
    "feature": model_casual_lgb.feature_name_,
    "importance": model_casual_lgb.feature_importances_
})

# ============ 2. registered の Feature Importance 取得 ============
fi_reg = pd.DataFrame({
    "feature": model_reg_lgb.feature_name_,
    "importance": model_reg_lgb.feature_importances_
})

# ============ 3. 正規化（合計を100に） ============
fi_casual["importance"] = fi_casual["importance"] / fi_casual["importance"].sum() * 100
fi_reg["importance"]    = fi_reg["importance"]    / fi_reg["importance"].sum() * 100


# ============ 4. 可視化関数 ============
def plot_fi(df, title):
    df_sorted = df.sort_values("importance", ascending=False)
    plt.figure(figsize=(10,6))
    plt.barh(df_sorted["feature"], df_sorted["importance"], color="skyblue")
    plt.xlabel("Importance (%)")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

# ============ 5. グラフ描画 ============
plot_fi(fi_casual, "LightGBM Feature Importance (Casual)")
plot_fi(fi_reg, "LightGBM Feature Importance (Registered)")



corr_cols = base_features + ["peak_cas", "peak_reg", "casual", "registered", "count"]

# 相関行列
corr = train[corr_cols].corr()

# マスク
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False

# 描画
fig, ax = plt.subplots()
fig.set_size_inches(20, 15)

sns.heatmap(corr, mask=mask, vmax=1.0, square=True, annot=True)

plt.show()


