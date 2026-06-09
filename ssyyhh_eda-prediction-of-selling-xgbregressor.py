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
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
pd.set_option("display.max_columns", None)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost.sklearn import XGBRegressor


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
train


# Drop useless feature("id")
train.drop("id", axis = 1, inplace = True)


# Drop row which has NaN
train.dropna(axis = 0, inplace = True)


train.info()


train.describe(include = "object")


train.describe()


# Change type of "date" from categorical ones to numerical ones
train["date"] = pd.to_datetime(train["date"])


# Selling amounts by Date
sns.lineplot(x = train["date"], y = train["num_sold"], color = "b", linestyle = "-")
plt.show()


# How many values in categorical features
cat_cols = []
for col in train.columns:
    if train[col].dtypes == "object":
        cat_cols.append(col)

for i in cat_cols:
    print(train[i].value_counts())
    print("-"*30)


plt.figure(figsize = (12, 8))
for i, col in enumerate(train.columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(x = train[col])
    plt.title(f"Histogram of {col} Data")
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.plot()


# Selling amounts of product in several countries
train.pivot_table(index = "country", columns = "product", values = "num_sold", aggfunc = "mean").plot.bar(rot = 0)


# Store's selling amount in several countries
train.pivot_table(index = "country", columns = "store", values = "num_sold", aggfunc = "mean").plot.bar(rot = 0)


# Store's selling amount in several countries
train.pivot_table(index = "store", columns = "product", values = "num_sold", aggfunc = "mean").plot.bar(rot = 0)


# split date with "Year"/"Month"/"Day"
train["date"] = train["date"].astype("str")

train[["Year", "Month", "Day"]] = train["date"].str.split("-", expand = True)
train["Year"] = train["Year"].astype("int")
train["Month"] = train["Month"].astype("int")
train["Day"] = train["Day"].astype("int")

train.drop("date", axis = 1, inplace = True)
train.head()


# Relocate target("num_sold") to end column
train1 = train.copy()
train.drop("num_sold", axis = 1, inplace = True)
train = pd.concat([train, train1["num_sold"]], axis = 1)
train.head()


# Change categorical features to numerical ones
le = LabelEncoder()
for col in cat_cols:
    train[col] = le.fit_transform(train[col])
train.head()


# Correlation of this dataset
plt.figure(figsize = (7, 7))
train_corr = train.corr()
sns.heatmap(train_corr, fmt = ".3f", annot = True, cmap = "YlGnBu")
plt.show()


# Split dataset with train & test
X = train.iloc[:, :-1]
y = train.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Find best parameters for XGBRegressor
xgbr = XGBRegressor()
param_grid = {"ne_stimators" : [40, 80, 120, 160, 200],
              "max_depth" : [2,3,4,5],
             "max_features" : [0.1, 0.2, 0.3, 0.4, 0.5]}
gscv = GridSearchCV(xgbr, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
gscv.fit(X_train, y_train)
print("Best parameters :", gscv.best_params_)


# Prepare Test data
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
test


# Drop Useless feature("id")
test.drop("id", axis = 1, inplace = True)


# Split "date" with "Year"/"Month"/"Day"
test[["Year", "Month", "Day"]] = test["date"].str.split("-", expand = True)
test["Year"] = test["Year"].astype("int")
test["Month"] = test["Month"].astype("int")
test["Day"] = test["Day"].astype("int")

test.drop("date", axis = 1, inplace = True)
test.head()


# Change categorical features with numerical ones
le = LabelEncoder()
for col in cat_cols:
    test[col] = le.fit_transform(test[col])
test


# Set up with best parameters
xgbr = XGBRegressor(n_estimators = 40, max_depth = 5, max_features = 0.1)
xgbr.fit(X_train, y_train)


# Predict XGBRegressor model with best parameters
y_xgbr = xgbr.predict(test)
y_xgbr


# Prepare for Submission Data
submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")
submission


# Mean Absolute Percentage Error btw y_xgbr & submission
def MAPE(submission, yxgbr):
	return np.mean(np.abs((submission - y_xgbr) / submission)) * 100 
    
print("MAPE :", MAPE(submission.iloc[:, 1], y_xgbr))




