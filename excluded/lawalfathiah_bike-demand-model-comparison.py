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


df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")


df.info()


df.head()


def time_converter(dataset):
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    dataset['day'] = dataset['datetime'].dt.day
    dataset['month'] = dataset['datetime'].dt.month
    dataset['hour'] = dataset['datetime'].dt.hour
    dataset.drop(columns = 'datetime', inplace = True)

time_converter(df)


df


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split


X = df.drop(columns = ['count', 'casual', 'registered'])
y = df['count']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 42)


clf1 = DecisionTreeRegressor()
param_grid1 = {"criterion": ["squared_error", "friedman_mse", "absolute_error"], "splitter" : ["best", "random"], "max_features" : ["auto", "sqrt", "log2"]} 


Grid_model1 = GridSearchCV(clf1, param_grid1, cv = 5)


Grid_model1.fit(X_train, y_train)


Grid_model1.best_params_


model1 = DecisionTreeRegressor(
criterion = 'squared_error', max_features = 'auto', splitter = 'best'
)


model1.fit(X_train, y_train)


model1.score(X_train, y_train)


model1.score(X_valid, y_valid)


import xgboost as xgb


clf2 = xgb.XGBRegressor()
param_grid2 = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 9]
}


Grid_model2 = GridSearchCV(clf2, param_grid2, cv = 5)


Grid_model2.fit(X_train, y_train)


Grid_model2.best_params_


model2 = xgb.XGBRegressor(learning_rate = 0.1, max_depth = 7, n_estimators = 1000)
model2.fit(X_train, y_train)


model2.score(X_train, y_train)


model2.score(X_valid, y_valid)


from sklearn.ensemble import RandomForestRegressor


## This part took so long to run, I had to put it into markdown

clf3 = RandomForestRegressor()
param_grid3 = {
    'max_features': ['sqrt', 'log2'], 
    'criterion': ["squared_error", "absolute_error", "poisson"] 
}

Grid_model3 = GridSearchCV(clf3, param_grid3, cv = 5)

Grid_model3.fit(X_train, y_train)

Grid_model3.best_params_


model3 = RandomForestRegressor()


model3.fit(X_train, y_train)


model3.score(X_train, y_train)


model3.score(X_valid, y_valid)


test_df = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")


test_df.head()


time_converter(test_df)


test_df.info()


test_df.head()


X_test = test_df


y_pred = model2.predict(X_test)
y_pred


test_df['count'] = y_pred.astype(int)


test_df
















