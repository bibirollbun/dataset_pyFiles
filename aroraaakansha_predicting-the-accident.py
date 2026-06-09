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


import pandas as pd
import numpy as np



train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
train.shape


test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test.head()



sample = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
sample.head()


x=train.drop(columns=['id','accident_risk'])
x.head()


y=train['accident_risk']
y


x = train.drop(['accident_risk', 'id'], axis=1)
x


cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
x = pd.get_dummies(x, columns=cat_cols, drop_first=True)
x.dtypes


bool_cols = x.select_dtypes('bool').columns
x[bool_cols] = x[bool_cols].astype(int)


x.dtypes



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


x_train,x_val,y_train,y_val = train_test_split(x, y, test_size=0.2, random_state=42)
x_train


x_train.dtypes



rf=RandomForestRegressor(random_state=42)
rf


rf.fit(x_train, y_train)


preds = rf.predict(x_val)
preds 


rmse = mean_squared_error(y_val, preds, squared=False)
rmse


test_features = test.drop(columns=['id'])
test_features


test_features = pd.get_dummies(test_features, columns=cat_cols, drop_first=True)
test_features


bool_cols = test_features.select_dtypes('bool').columns
test_features[bool_cols] = test_features[bool_cols].astype(int)
test_features


test_features = test_features.reindex(columns=x.columns, fill_value=0)
test_features.head



test_preds = rf.predict(test_features)
submission = pd.DataFrame({'id': test['id'], 'accident_risk': test_preds})
submission



submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': test_preds
})


submission.to_csv('submission.csv', index=False)





submission.shape

