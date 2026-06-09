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
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv") 
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


y = train['Calories']
x = train.drop(columns = ['id', 'Calories'])


x_test = test.drop(columns = ['id'])


x_train,x_val,y_train,y_val = train_test_split(x, y, train_size = 0.7, test_size = 0.3, random_state = 0)


numerical_cols = [cname for cname in x_train.columns if 
                x_train[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in x_train.columns if 
                    x_train[cname].dtype == "object"]


numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
model_1 = XGBRegressor(random_state = 10, n_estimators = 1000, learning_rate = 0.05)
Model_1 = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model_1)
                     ])
Model_1.fit(x_train, y_train)

pred_1 = Model_1.predict(x_val)


model_2 = RandomForestRegressor(random_state = 10, n_estimators = 100, max_depth = 15)
Model_2 = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model_2)
                     ])
Model_2.fit(x_train, y_train)

pred_2 = Model_2.predict(x_val)


model_3 = AdaBoostRegressor(random_state = 10, n_estimators = 50, learning_rate = 0.05)
Model_3 = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model_3)
                     ])
Model_3.fit(x_train, y_train)

pred_3 = Model_3.predict(x_val)


print(np.sqrt(mean_squared_log_error(y_val, np.where(pred_3 < 0, np.median(pred_3), pred_3))))


pred = (0.5 * pred_1) + (0.5 * pred_2)


print(np.sum(pred<0))


pred = np.where(pred < 0, np.mean(pred), pred)
print(pred)


rmsle = np.sqrt(mean_squared_log_error(y_val, pred))


rmsle


predictions_1 = Model_1.predict(x_test)
predictions_2 = Model_2.predict(x_test)
predictions = (0.5 * predictions_1) + (0.5 * predictions_2)
predictions = np.maximum(0, predictions)


id_columns = test['id']


result = pd.DataFrame(
    {
        'id':id_columns,
        'Calories':predictions
    })


result.to_csv('predictions.csv',index = False)




