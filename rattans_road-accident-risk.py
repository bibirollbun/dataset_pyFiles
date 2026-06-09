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


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


X_full = pd.read_csv('../input/playground-series-s5e10/train.csv')
X_test = pd.read_csv('../input/playground-series-s5e10/test.csv')

X_full.describe()


X_full.nunique()


X_test.nunique()


# Adding a base_risk feature

X_full['base_risk'] = (
        0.3 * X_full["curvature"] + 
        0.2 * (X_full["lighting"] == "night").astype(int) + 
        0.1 * (X_full["weather"] != "clear").astype(int) + 
        0.2 * (X_full["speed_limit"] >= 60).astype(int) + 
        0.1 * (np.array(X_full["num_reported_accidents"]) > 2).astype(int)
    )
X_test['base_risk'] = (
        0.3 * X_test["curvature"] + 
        0.2 * (X_test["lighting"] == "night").astype(int) + 
        0.1 * (X_test["weather"] != "clear").astype(int) + 
        0.2 * (X_test["speed_limit"] >= 60).astype(int) + 
        0.1 * (np.array(X_test["num_reported_accidents"]) > 2).astype(int)
)


X_full.info()


#converting boolean values to int

bool_cols = ['road_signs_present', 'public_road', 'road_signs_present', 'holiday', 'school_season']
for col in bool_cols:
    X_full[f'{col}'] = X_full[f'{col}'].astype(int)
    X_test[f'{col}'] = X_test[f'{col}'].astype(int)


# categorical encoding for categorical features

str_cols = ['road_type', 'lighting','weather', 'time_of_day']
OH = OneHotEncoder(sparse_output = False)
train_oh = pd.DataFrame(OH.fit_transform(X_full[str_cols]))
test_oh = pd.DataFrame(OH.transform(X_test[str_cols]))

X_full = pd.concat([X_full.drop(str_cols, axis = 1), train_oh], axis = 1)
X_test = pd.concat([X_test.drop(str_cols, axis = 1), test_oh], axis = 1)


len(X_full)


X_full = X_full.drop_duplicates()
len(X_full)


# An important step that beginners forget and get error
# All the column name should be of same type before passing to the model

X_full.columns = X_full.columns.astype(str)
X_test.columns = X_test.columns.astype(str)


# Splitting the dataset
X_train, X_val, y_train, y_val = train_test_split(X_full.drop(['id', 'accident_risk'], axis = 1), X_full.accident_risk, test_size = 0.2, random_state = 6)


modelx = XGBRegressor(n_estimators = 80, n_jobs = -1, gamma = 0.00001, learning_rate = 0.2, booster = 'gbtree', device = 'cuda', eval_metric = 'rmse', num_parallel_tree = 10,  random_state = 6)
modelx.fit(X_train, y_train)


pred = modelx.predict(X_val)


mean_squared_error(pred, y_val, squared = False)


X_full.drop('id', axis = 1, inplace = True)
test_id = X_test.pop('id')
y = X_full.pop('accident_risk')


X_full.head()


score = -1 * cross_val_score(modelx, X_full, y, cv = 5, scoring = 'neg_mean_squared_error')
(score.mean()) ** 0.5


modelx.fit(X_full, y)

final_pred = modelx.predict(X_test)


output = pd.DataFrame({'id':test_id, 'accident_risk':final_pred})
output.head()


output.to_csv('submission.csv', index = False)

