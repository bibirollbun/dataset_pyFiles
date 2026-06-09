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


data = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')
data


test = pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')
test


store = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')
store


sub = pd.read_csv('/kaggle/input/rossmann-store-sales/sample_submission.csv')
sub


df = data.merge(store, on="Store", how="left")
df


df = df[df["Open"] != 0].copy()
df = df.sort_values(["Store", "Date"])


df["Date"] = pd.to_datetime(df["Date"], errors="coerce")


df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day
df["dayofweek"] = df["Date"].dt.dayofweek
df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
df["StateHoliday_flag"] = (df["StateHoliday"].astype(str) != "0").astype(int)


df["sales_lag_1"] = df.groupby("Store")["Sales"].shift(1)
df["sales_lag_1"].fillna(df["Sales"].mean(), inplace=True)

df["CompetitionDistance"].fillna(df["CompetitionDistance"].median(), inplace=True)


df = df.sort_values("Date")


df.info()


df.describe()


import matplotlib.pyplot as plt
import seaborn as sns

df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["dayofweek"] = df["Date"].dt.dayofweek

daily_sales = df.groupby("Date")["Sales"].sum().reset_index()

plt.figure(figsize=(12,5))
plt.plot(daily_sales["Date"], daily_sales["Sales"])
plt.title("Daily Total Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()


sales_dow = df.groupby("dayofweek")["Sales"].mean().reset_index()

plt.figure(figsize=(8,4))
sns.barplot(x="dayofweek", y="Sales", data=sales_dow, palette="Blues_d")
plt.title("Average Sales by Day of Week")
plt.xlabel("Day of Week (0=Mon,6=Sun)")
plt.ylabel("Average Sales")
plt.show()



sales_month = df.groupby("month")["Sales"].mean().reset_index()

plt.figure(figsize=(8,4))
sns.barplot(x="month", y="Sales", data=sales_month, palette="Oranges_d")
plt.title("Average Sales by Month")
plt.xlabel("Month")
plt.ylabel("Average Sales")
plt.show()


train_size = int(len(df)*0.8)
train_df = df.iloc[:train_size]
test_df  = df.iloc[train_size:]


X_train = train_df[["Store","Promo","CompetitionDistance","year","month","day","dayofweek","sales_lag_1"]]
y_train = train_df["Sales"]

X_test  = test_df[["Store","Promo","CompetitionDistance","year","month","day","dayofweek","sales_lag_1"]]
y_test  = test_df["Sales"]



plt.figure(figsize=(8,4))
sns.histplot(df["Sales"], bins=50, kde=True, color="green")
plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Count")
plt.show()


numeric_cols = ["Sales", "CompetitionDistance", "Promo", "year", "month", "dayofweek"]
corr = df[numeric_cols].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Between Features")
plt.show()


import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.1,
    "max_depth": 6
}

model = xgb.train(params, dtrain, num_boost_round=200)


y_pred = model.predict(dtest)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.4f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') 
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Predicted vs Actual Sales")
plt.show()

