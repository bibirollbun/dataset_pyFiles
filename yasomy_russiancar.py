!pip install pytorch-tabnet


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = "/kaggle/input/russian-car-plates-prices-prediction/"
train_df = pd.read_csv(DATA_PATH + "train.csv")
train_df.dropna(subset=["plate", "price"], inplace=True)

Q1 = train_df["price"].quantile(0.25)
Q3 = train_df["price"].quantile(0.75)
IQR = Q3 - Q1
train_df = train_df[(train_df["price"] > Q1 - 1.5 * IQR) & (train_df["price"] < Q3 + 1.5 * IQR)]


def extract_plate_features(plate):
    match = re.search(r'([A-ZА-Я]+)(\d+)([A-ZА-Я]+)', plate)
    return match.groups() if match else ("", "0", "")


train_df[["letters1", "digits", "letters2"]] = train_df["plate"].apply(lambda x: pd.Series(extract_plate_features(x)))
train_df["digits"] = pd.to_numeric(train_df["digits"], errors='coerce').fillna(0)
train_df["letters1"] = train_df["letters1"].astype("category").cat.codes
train_df["letters2"] = train_df["letters2"].astype("category").cat.codes
train_df["digit_count"] = train_df["digits"].astype(str).apply(len)
train_df["unique_digits"] = train_df["digits"].astype(str).apply(lambda x: len(set(x)))
train_df["unique_letters"] = train_df["plate"].apply(lambda x: len(set(x)))
train_df["double_letters"] = train_df["plate"].apply(lambda x: sum(x[i] == x[i+1] for i in range(len(x)-1)))
features = ["letters1", "digits", "letters2", "digit_count", "unique_letters", "unique_digits", "double_letters"]
X = train_df[features]
y = np.log1p(train_df["price"])
scaler = StandardScaler()
X[["digits", "unique_digits", "digit_count"]] = scaler.fit_transform(X[["digits", "unique_digits", "digit_count"]])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
xgb.fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
rf.fit(X_train, y_train)
lgb = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
lgb.fit(X_train, y_train)


y_pred = (xgb.predict(X_val) + rf.predict(X_val) + lgb.predict(X_val)) / 3
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")


test_df = pd.read_csv(DATA_PATH + "test.csv")
test_df[["letters1", "digits", "letters2"]] = test_df["plate"].apply(lambda x: pd.Series(extract_plate_features(x)))
test_df["digits"] = pd.to_numeric(test_df["digits"], errors='coerce').fillna(0)
test_df["letters1"] = test_df["letters1"].astype("category").cat.codes
test_df["letters2"] = test_df["letters2"].astype("category").cat.codes
test_df["digit_count"] = test_df["digits"].astype(str).apply(len)
test_df["unique_digits"] = test_df["digits"].astype(str).apply(lambda x: len(set(x)))
test_df["unique_letters"] = test_df["plate"].apply(lambda x: len(set(x)))
test_df["double_letters"] = test_df["plate"].apply(lambda x: sum(x[i] == x[i+1] for i in range(len(x)-1)))

X_test = test_df[features]
X_test[["digits", "unique_digits", "digit_count"]] = scaler.transform(X_test[["digits", "unique_digits", "digit_count"]])
test_df["price"] = np.expm1((xgb.predict(X_test) + rf.predict(X_test) + lgb.predict(X_test)) / 3)


train_df["true_price"] = np.expm1(y)

top_10_expensive = train_df[["plate", "true_price"]].sort_values(by="true_price", ascending=False).head(10)
print(top_10_expensive)


test_df[["id", "price"]].to_csv("Submission.csv", index=False)
print("Submission готов!")

