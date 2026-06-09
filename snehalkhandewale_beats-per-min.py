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


import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


train.head()


train.shape


test.shape


train.isnull().sum()


train.duplicated().sum()


train.info()


train = train.drop(columns=["id"])
test = test.drop(columns=["id"])


for col in train:
    plt.figure(figsize=(12,4))

    # Histogram
    plt.subplot(1,2,1)
    sns.histplot(train[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')

    # Boxplot
    plt.subplot(1,2,2)
    sns.boxplot(x=train[col])
    plt.title(f'Boxplot of {col}')

    plt.tight_layout()
    plt.show()
    


X = train.drop(columns=["BeatsPerMinute"])
y = train["BeatsPerMinute"]


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state = 42)


LR = LinearRegression()
LR.fit(X_train,y_train)


y_pred = LR.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse


LR.fit(X,y)
y_pred_LR = LR.predict(test)


rf = RandomForestRegressor()
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse


rf.fit(X,y)
y_pred_rf = rf.predict(test)


xgb = XGBRegressor()
xgb.fit(X_train, y_train)


y_pred = xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse


xgb.fit(X,y)
y_pred_xgb = xgb.predict(test)

