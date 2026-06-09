import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
train.drop(columns=["id", "date"], inplace=True)
train.head()


train.info()


train.isnull().sum()


train["country"].unique()


train["product"].unique()


train = train.dropna(subset=["num_sold"])
train.head()


train.info()


train = pd.get_dummies(train, drop_first=True)
train.head()


corr = train.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.show()


x = train.drop(columns="num_sold")
y = train["num_sold"]


y


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


reg = RandomForestRegressor(n_jobs = -1)
reg.fit(x_train, y_train)


y_pred = reg.predict(x_test)
y_pred


sns.distplot(y_test - y_pred)
plt.show()


mape = mean_absolute_percentage_error(y_test, y_pred)
mape


reg = XGBRegressor(n_jobs = -1)
reg.fit(x_train, y_train)


y_pred = reg.predict(x_test)
y_pred


mape = mean_absolute_percentage_error(y_test, y_pred)
mape


sns.distplot(y_test - y_pred)
plt.show()


reg = LGBMRegressor(n_jobs = -1)
reg.fit(x_train, y_train)


y_pred = reg.predict(x_test)
y_pred


mape = mean_absolute_percentage_error(y_test, y_pred)
mape


sns.distplot(y_test - y_pred)
plt.show()


reg = RandomForestRegressor(n_jobs = -1)
reg.fit(x, y)


test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")
test.head()


test.drop(columns=["id", "date"], inplace=True)
test = pd.get_dummies(test, drop_first=True)
test.head()


pred = reg.predict(test)
pred


sub["num_sold"] = pred
sub.head()


sub.to_csv("submission.csv", index=False)

