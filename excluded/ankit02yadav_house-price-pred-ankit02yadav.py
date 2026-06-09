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
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv('/kaggle/input/house-price-pred/train.csv')
categorical_columns=['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
X = data.drop(columns=["price"])
y = data['price'] 
train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=42)
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(train_X,train_y)
pred_y = model.predict(test_X)
mae = mean_absolute_error(test_y,pred_y)
r2 = r2_score(test_y,pred_y)
print("MEAN ABSOLUTE ERROR : ",mae)
print("R^2 SCORE : ",r2)
# model trainned
test_data = pd.read_csv('/kaggle/input/house-price-pred/test.csv')
# filter the testdata 
test_data = pd.get_dummies(test_data, columns=categorical_columns, drop_first=True)
pred_test_y = model.predict(test_data)
# model predicted the test prices
# makeing into csv file 
print(pred_test_y)
submission = pd.DataFrame({
    'Id': test_data['Id'], 
    'SalePrice': pred_test_y
})

submission.to_csv('submission.csv', index=False)





