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
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
# from tabpfn import TabPFNRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
# read data
train_data = pd.read_csv('/kaggle/input/predicting-the-price-of-diamond/train.csv')
test_data = pd.read_csv('/kaggle/input/predicting-the-price-of-diamond/test.csv')
# filter data 
categorical_col = list(train_data.dtypes[train_data.dtypes == 'object'].index)
train_data = pd.get_dummies(train_data,columns=categorical_col,drop_first=True)
test_data = pd.get_dummies(test_data,columns=categorical_col,drop_first=True)
# model tranning 
X = train_data.drop(columns=['price','id'])
y = train_data['price']
test_id = test_data['id']
test_data = test_data.reindex(columns=X.columns, fill_value=0)
train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=42)
# model = RandomForestRegressor(n_estimators=100,random_state=42)   #score == 0.62
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42) #score = 0.6435
# model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42); #score = 0.64
# model = CatBoostRegressor(     #0.6398027708730789
#     iterations=2000,
#     learning_rate=0.03,
#     depth=8,
#     loss_function='RMSE',
#     early_stopping_rounds=100,
#     verbose=100,
#     random_state=42
# )
# model = TabPFNClassifier(device='cuda' or 'cpu')
model.fit(train_X,train_y)
# cheking model 
pred_y = model.predict(test_X)
mae = mean_absolute_error(test_y,pred_y)
r2 = r2_score(test_y,pred_y)
print(f"MAE: {mae}")
print(f"R^2 Score: {r2}")
# pred the test 
test_pred = model.predict(test_data)
print(test_pred)
# submission 
submission = pd.DataFrame({
    'id':test_id,
    'price':test_pred
}) 
submission.to_csv('submission.csv', index=False)




