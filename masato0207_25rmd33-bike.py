import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# データ読み込み
DATA_DIR = "/kaggle/input/bike-sharing-demand"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

    
#print("▼ train.csv")
#print(train.shape)      # 行数・列数
#print(train.columns)    # 列名一覧
#print(train.dtypes)     # データ型の確認
#print(train.head())     # 先頭5行

#print("\n▼ test.csv")
#print(test.shape)
#print(test.columns)
#print(test.head())
#print("\n▼ train 欠損値")
#print(train.isnull().sum())

#nprint(train["temp"].describe())

#print("\n▼ test 欠損値")
#print(test.isnull().sum())


# datetime を分解して特徴量を作成
def preprocess(df):
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["working_hour"] = df["hour"] * df["workingday"]
    
    # peak（利用ピーク時間帯の特徴量）
    df['peak'] = df[['hour', 'workingday']].apply(
        lambda x: int(
            (x['workingday'] == 1 and (x['hour'] == 8 or 12 <= x['hour'] <= 13 or 17 <= x['hour'] <= 18)) or
            (x['workingday'] == 0 and 10 <= x['hour'] <= 19)
        ), axis=1)

   # sticky（平日かつ湿度が高い = 不快）
    df['sticky'] = df[['humidity', 'workingday']].apply(lambda x: int(x['workingday'] == 1 and x['humidity'] >= 60), axis=1)

    
    return df

def apply_special_day_adjustments(df):
    df["datetime"] = pd.to_datetime(df["datetime"])
    dates = [
        (pd.Timestamp(2011, 4, 15), 1, 0),  # Tax day
        (pd.Timestamp(2012, 4, 16), 1, 0),  # Tax day
        (pd.Timestamp(2011, 11, 25), 0, 1),  # Thanksgiving Friday
        (pd.Timestamp(2012, 11, 23), 0, 1),  # Thanksgiving Friday
        (pd.Timestamp(2011, 12, 24), 0, 1),  # Christmas
        (pd.Timestamp(2012, 12, 24), 0, 1),
        (pd.Timestamp(2011, 12, 26), 0, 1),
        (pd.Timestamp(2012, 12, 26), 0, 1),
        (pd.Timestamp(2011, 12, 31), 0, 1),  # New Year’s Eve
        (pd.Timestamp(2012, 12, 31), 0, 1),
        (pd.Timestamp(2012, 5, 21), 0, 1),  # Storms
        (pd.Timestamp(2012, 6, 1), 0, 1),  # Tornado
        (pd.Timestamp(2012, 10, 30), 0, 1),  # Sandy
    ]
    for date, workingday_value, holiday_value in dates:
        df.loc[df['datetime'].dt.date == date.date(), 'workingday'] = workingday_value
        df.loc[df['datetime'].dt.date == date.date(), 'holiday'] = holiday_value
    return df

train = apply_special_day_adjustments(train)
test = apply_special_day_adjustments(test)

train = preprocess(train)
test = preprocess(test)


# 使用する特徴量を選択（簡易的なもの）
features = ["season", "year", "month", "hour", "weekday", "holiday", "workingday", 
            "weather", "temp", "atemp", "humidity", "windspeed",
            "peak","sticky", "working_hour"]  

X = train[features]
# X = train[['hour']]LB:0.79714
# X = train[['temp']]
y = np.log1p(train['count'])
X_test = test[features]
# X_test = test[["temp"]]


# 学習と予測
model = XGBRegressor(random_state=71)
model.fit(X, y)
preds = np.expm1(model.predict(X_test))

# マイナスになる予測値を防ぐ（提出形式上、負の値は不可）
preds = [max(0, round(x)) for x in preds]


# 提出ファイル作成
submission = pd.DataFrame({
    "datetime": test["datetime"],
    "count": preds
})
submission.to_csv("submission.csv", index=False)

print("submission.csv を出力しました。")

