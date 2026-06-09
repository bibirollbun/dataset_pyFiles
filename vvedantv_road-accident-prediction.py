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


import sklearn
import seaborn as sns
import matplotlib.pyplot as plt


df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


df_train.head()


df_test.head()


df_train.info()


df_train.isnull().sum()


X = df_train.drop(['accident_risk', 'id'], axis=1)
y = df_train['accident_risk']


df_test.head()
df_test = df_test.drop('id', axis=1)


X = pd.get_dummies(X, drop_first=True)


X.head()


df_test = pd.get_dummies(df_test, drop_first=True)


X.info()


X['num_lanes'].unique()


sns.boxplot(X['num_reported_accidents'])


from sklearn.preprocessing import StandardScaler


sd = StandardScaler()
X[['speed_limit']] = sd.fit_transform(X[['speed_limit']])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

knr = KNeighborsRegressor()
param_grid = {'n_neighbors': [3,5,7],
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan']}
Grid = GridSearchCV(estimator=knr, 
                    param_grid=param_grid,
                    scoring='neg_mean_squared_error',
                    cv=3, n_jobs=-1)



X.dtypes
X = pd.get_dummies(X, drop_first=True)
X.dtypes


Grid.fit(X_train, y_train)


best_knr = Grid.best_estimator_
y_pred = best_knr.predict(X_test)

