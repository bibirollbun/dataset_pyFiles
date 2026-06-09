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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/rossmann-store-sales/train.csv", parse_dates=["Date"])
test = pd.read_csv("/kaggle/input/rossmann-store-sales/test.csv", parse_dates=["Date"])
store = pd.read_csv("/kaggle/input/rossmann-store-sales/store.csv")
submission = pd.read_csv("/kaggle/input/rossmann-store-sales/sample_submission.csv")


train.head()



test.head()



store.head()


submission.head()


#  Merge Store Info
train = train.merge(store, on="Store", how="left")
test = test.merge(store, on="Store", how="left")



# Data Cleaning
# Remove closed stores from train
train = train[train["Open"] == 1]

# Fill NA in test
test["Open"].fillna(1, inplace=True)  

# Drop rows with zero sales (optional)
train = train[train["Sales"] > 0]



train .head()


# Feature Engineering
def add_features(df):
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    return df

train = add_features(train)
test = add_features(test)


train.head()


#  Visualize Seasonality
plt.figure(figsize=(12, 5))
sns.lineplot(data=train.groupby("Month")["Sales"].mean(), label="Average Sales by Month")
plt.title("Monthly Sales Trend")
plt.ylabel("Sales")
plt.grid()
plt.show()


#  Time Series Forecasting (LSTM-like Feature + XGBoost)

from xgboost import XGBRegressor


features = ["Store", "DayOfWeek", "Promo", "SchoolHoliday", "Year", "Month", "Day", "WeekOfYear"]
target = "Sales"

# Prepare training data
X_train = train[features]
y_train = train[target]

# Train model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)




X_test = test[features]
test["Sales"] = model.predict(X_test)


test["Sales"] = test["Sales"].clip(lower=0)





final_submission = test[["Id", "Sales"]]
final_submission.to_csv("submission.csv", index=False)
final_submission.head()



from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
model.fit(X_tr, y_tr)
preds = model.predict(X_val)

mape = np.mean(np.abs((y_val - preds) / y_val)) * 100
rmse = np.sqrt(mean_squared_error(y_val, preds))

print(f" MAPE: {mape:.2f}%")
print(f"RMSE: {rmse:.2f}")

