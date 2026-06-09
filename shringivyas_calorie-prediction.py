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


data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')


data.isnull().sum()


data.info()


data.nunique()


# encoding categorical col 
data = pd.get_dummies(data, drop_first = True)


data.head()


X = data.drop(columns = ['Calories'], axis = 1)
y = data['Calories']


X.head()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_log_error
from sklearn import linear_model


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


random_forest_model = RandomForestRegressor(n_estimators = 100, random_state = 42)
random_forest_model.fit(X_train, y_train)
y_pred = random_forest_model.predict(X_test)


mean_abs_error = mean_absolute_error(y_test, y_pred)
r2_s = r2_score(y_test, y_pred)
print(mean_abs_error, r2_s)


#mean squared log error
y_pred = np.maximum(0, y_pred)
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
rmsle


linear_reg_model = linear_model.LinearRegression()
linear_reg_model.fit(X_train, y_train)
y_pred_lr = linear_reg_model.predict(X_test)

mean_abs_error_lr = mean_absolute_error(y_test, y_pred_lr)
r2_s_lr = r2_score(y_test, y_pred_lr)

y_pred_lr = np.maximum(0, y_pred_lr)
rmsle_lr = np.sqrt(mean_squared_log_error(y_test, y_pred_lr))
print(f'mean abs error = {mean_abs_error_lr}, r square = {r2_s_lr}, root mean square log error = {rmsle_lr} ')


test_file = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test_file = pd.get_dummies(test_file, drop_first = True)


test_predictions = random_forest_model.predict(test_file)


test_file['Calories'] = test_predictions


test_file.head()


submission_file = pd.DataFrame()
submission_file['id'] = test_file['id']
submission_file['Calories'] = test_file['Calories']


submission_file.head()


submission_file.to_csv('submission_file.csv', index = False)

