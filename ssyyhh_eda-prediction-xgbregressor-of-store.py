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


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_columns", None)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
train


train.info()


train.describe()


train.describe(include = "object")


# Dorp useless feature("id")
train.drop("id", axis = 1, inplace = True)
train.head()


# Change type of "date" from int to datetime
train["date"] = pd.to_datetime(train["date"])


sns.lineplot(x = train["date"], y = train["num_sold"], linestyle = "-", color = "b", marker = "o")
plt.show()


# Change type of "date" from datetime to str
train["date"] = train["date"].astype("str")

# Split "date" with "Year"/"Month"/"Day"
train[["Year", "Month", "Day"]] = train["date"].str.split("-", expand = True)
train["Year"] = train["Year"].astype("int")
train["Month"] = train["Month"].astype("int")
train["Day"] = train["Day"].astype("int")

train.drop(columns = ["date"], axis = 1, inplace = True)
train.head()


# How many values in categorical features
cat_cols = []
for col in train.columns:
    if train[col].dtype == "object":
        cat_cols.append(col)

for i in cat_cols:
    print(train[i].value_counts())
    print("-"*30)


# Relocate target("store") to end column
train1 = train.copy()
train.drop("store", axis = 1, inplace = True)
train = pd.concat([train, train1["store"]], axis = 1)
train.head()


# Histogram of this dataset
plt.figure(figsize = (15, 25))
for i, col in enumerate(train.columns, 1):
    plt.subplot(4, 2, i)
    sns.histplot(x = train[col])
    plt.title(f"Histogram of {col} Data")
    plt.xticks(rotation = 45)
    plt.plot()


train.pivot_table(index = "store", columns = "country", values = "num_sold", aggfunc = "mean").plot.bar(rot = 45)


train.pivot_table(index = "store", columns = "product", values = "num_sold", aggfunc = "mean").plot.bar(rot = 45)


train.pivot_table(index = "store", columns = "Year", values = "num_sold", aggfunc = "mean").plot.bar(rot = 45)


# Change categorical features to numerical ones
le = LabelEncoder()
for col in cat_cols:
    train[col] = le.fit_transform(train[col])

train.head()


# Drop first row in this dataset
train = train.drop(index = 0).reset_index(drop = True)
train.head()


# Correlation of this dataset
plt.figure(figsize = (7, 7))
train_corr = train.corr()
sns.heatmap(train_corr, fmt = ".3f", annot = True, cmap = "YlGnBu")
plt.show()


# Split dataset with train & test
X = train.iloc[:, :-1]
y = train.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# Standardization of X_train & X_test
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Find best parameters for XGBRegressor model
xgbr = XGBRegressor()
param_grid = {"n_estimators" : [40,80,120,160,200],
             "max_depth" : [2,3,4,5,6],
             "max_features" : [01.,0.2,0.3,0.4,0.5]}
gscv = GridSearchCV(xgbr, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
gscv.fit(X_train, y_train)
print("Best Parameters :", gscv.best_params_)


# Adopt Best Parameters
xgbr = XGBRegressor(n_estimators = 200, max_depth = 6, max_features = 1)
xgbr.fit(X_train, y_train)


# Predict XGBRegressor model
y_xgbr = xgbr.predict(X_test).round()
y_xgbr


# Mean Absolute Error btw y_xgbr & y_test
print("MAE Score :", mean_absolute_error(y_xgbr , y_test))




