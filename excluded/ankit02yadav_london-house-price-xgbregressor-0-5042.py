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
# from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score,accuracy_score
# read data
train_data = pd.read_csv('/kaggle/input/london-house-price-prediction-advanced-techniques/train.csv')
test_data = pd.read_csv('/kaggle/input/london-house-price-prediction-advanced-techniques/test.csv')
train_data = train_data.drop(columns=['fullAddress','postcode','ID'])
test_id = test_data['ID']
test_data =  test_data.drop(columns=['fullAddress','postcode','ID'])
# filter
categrical_col = ['country','outcode','tenure','propertyType','currentEnergyRating']
train_data = pd.get_dummies(train_data,columns=categrical_col,drop_first=True)
test_data = pd.get_dummies(test_data,columns=categrical_col,drop_first=True)
# train 
X = train_data.drop(columns=['price']);
y = train_data['price']
test_data = test_data.reindex(columns=X.columns, fill_value=0)
train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=42)
# model = RandomForestRegressor(n_estimators=100,random_state=42)
# model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42) 0.47 == 249954.49
# model = XGBRegressor(n_estimators=5,random_state=42) #score = 0.502
model = XGBRegressor(  #0.5314
    n_estimators=300,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.7,
    random_state=42
)
print(X.dtypes[X.dtypes == 'object'])
model.fit(train_X,train_y)
print("trained")
# check
pred_y = model.predict(test_X)
mae = mean_absolute_error(test_y,pred_y)
r2 = r2_score(test_y,pred_y)
print("MAE : ",mae)
print("r^2 : ",r2)
# pred
test_pred = model.predict(test_data)
# submission 
submission = pd.DataFrame({
    'ID':test_id,
    'price':test_pred
}) 
submission.to_csv('submission.csv', index=False)




