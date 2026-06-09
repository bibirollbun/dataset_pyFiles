import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import MiniBatchKMeans


train = pd.read_csv("/kaggle/input/data-real/train.csv")
test = pd.read_csv("/kaggle/input/data-real/test.csv")


test.shape, train.shape


train.info(), test.info()


train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"])
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])


plt.figure(figsize=(10,6))
sns.histplot(train["trip_duration"], bins=150)

plt.xlabel("trip_duration")
plt.ylabel("count")
plt.show()


train["trip_duration"].quantile([0.95, 0.99, 0.995, 0.999])


train = train[train["trip_duration"].between(10, 3*3600)].copy()


plt.figure(figsize=(8,8))
sns.countplot(
    x="passenger_count",
    data = train
)

plt.show()


train["passenger_count"].isin([7,8,9]).sum(), train["passenger_count"].value_counts().loc[[7,8,9]]


train = train[train["passenger_count"].between(1, 6)].copy()


cols = ["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.hist(train["pickup_latitude"], bins=100)
plt.title("pickup_latitude")
plt.subplot(1,2,2)
plt.hist(train["pickup_longitude"], bins=100)
plt.title("pickup_longitude")
plt.tight_layout()
plt.show()


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return 6371.0 * c

def add_features(df):
    # hour, weekday
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # 緯度経度差分
    df["dlat"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["dlon"] = df["dropoff_longitude"] - df["pickup_longitude"]

    # 直線距離(km)
    df["haversine_distance"] = haversine(
        df["pickup_longitude"], df["pickup_latitude"],
        df["dropoff_longitude"], df["dropoff_latitude"]
    )

    # store_and_fwd_flag を 0/1
    df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)

    return df


def add_place_clusters(train, test, n_clusters=200, random_state=42):
    # train+testのpickup/dropoff全部をまとめて学習（境界ズレ防止）
    coords = np.vstack([
        train[["pickup_latitude","pickup_longitude"]].values.astype(np.float64),
        train[["dropoff_latitude","dropoff_longitude"]].values.astype(np.float64),
        test[["pickup_latitude","pickup_longitude"]].values.astype(np.float64),
        test[["dropoff_latitude","dropoff_longitude"]].values.astype(np.float64),
    ])

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=10000,
        random_state=random_state,
        n_init="auto"
    )
    kmeans.fit(coords)

    train["pickup_cluster"]  = kmeans.predict(train[["pickup_latitude","pickup_longitude"]].values.astype(np.float64))
    train["dropoff_cluster"] = kmeans.predict(train[["dropoff_latitude","dropoff_longitude"]].values.astype(np.float64))
    test["pickup_cluster"]   = kmeans.predict(test[["pickup_latitude","pickup_longitude"]].values.astype(np.float64))
    test["dropoff_cluster"]  = kmeans.predict(test[["dropoff_latitude","dropoff_longitude"]].values.astype(np.float64))

    # LightGBMに「カテゴリ」として渡すのがコツ
    for c in ["pickup_cluster","dropoff_cluster"]:
        train[c] = train[c].astype("category")
        test[c]  = test[c].astype("category")

    return train, test


def manhattan_distance_km(df):
    lat_km = 111.0
    lon_km = 111.0 * np.cos(np.radians(df["pickup_latitude"]))
    

train, test = add_place_clusters(train, test, n_clusters=100)

train = add_features(train)
test  = add_features(test)


feature_cols = [
    "hour", "weekday", "month", "day", "is_weekend",
    "haversine_distance",
    "dlat", "dlon",
    "passenger_count",
    "store_and_fwd_flag",
    "pickup_cluster",
    "dropoff_cluster" 
    
]

X = train[feature_cols].copy()
X_test = test[feature_cols].copy()

y=np.log1p(train["trip_duration"].values)


n_splits = 5
kf = KFold(n_splits = n_splits, shuffle = True, random_state=42)

oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model = lgb.LGBMRegressor(
        n_estimators = 3000,
        learning_rate = 0.05,
        num_leaves=63,
        subsample = 0.8,
        colsample_bytree = 0.8,
        random_state=42
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric = "rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    va_pred = model.predict(X_va)
    oof[va_idx] = va_pred

    fold_rmse = np.sqrt(mean_squared_error(y_va, va_pred))
    print(f"fold{fold}: RMSE(log) = {fold_rmse:.5f}")

    test_pred += model.predict(X_test) / n_splits


cv_rmse = np.sqrt(mean_squared_error(y, oof))
print("CV RMSE(log space) =", cv_rmse )


# %% [code] {"execution":{"iopub.status.busy":"2025-12-17T07:34:06.842658Z","iopub.execute_input":"2025-12-17T07:34:06.842912Z","iopub.status.idle":"2025-12-17T07:34:07.658451Z","shell.execute_reply.started":"2025-12-17T07:34:06.842889Z","shell.execute_reply":"2025-12-17T07:34:07.657644Z"},"jupyter":{"outputs_hidden":false}}
test_pred_sec = np.expm1(test_pred)
test_pred_sec = np.clip(test_pred_sec, 0, None)

sub = pd.DataFrame({"id": test["id"], "trip_duration": test_pred_sec.astype(np.float32)})
sub.to_csv("submission.csv", index=False)
print("saved: submission.csv")

