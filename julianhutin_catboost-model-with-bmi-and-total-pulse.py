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


exemple = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
exemple


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
train


train['Sex'] = train['Sex'].astype('category')
test['Sex'] = test['Sex'].astype('category')


train['imc'] = train['Weight'] / (train['Height']/100)**2
train['total_heart_pulse'] = train['Heart_Rate'] * train['Duration']

test['imc'] = test['Weight'] / (test['Height']/100)**2
test['total_heart_pulse'] = test['Heart_Rate'] * test['Duration']


X,y = train.drop(['id','Calories'], axis = 1),train['Calories']


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

xgb = XGBRegressor(enable_categorical = True)
xgb.fit(X, y)

mean_squared_error(y, xgb.predict(X))


from catboost import CatBoostRegressor

catboost = CatBoostRegressor(depth=10, learning_rate=0.1, iterations=1000)
catboost.fit(X, y, cat_features=['Sex'])

mean_squared_error(y, catboost.predict(X))


preds = catboost.predict(test.drop(['id'], axis = 1))

submission = pd.DataFrame(list(zip(test['id'].values, preds)), columns=['id', 'Calories'])
submission = submission.clip(lower=1)
submission


submission.to_csv('submission.csv', index=False)

