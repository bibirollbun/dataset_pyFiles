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
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv('/kaggle/input/house-price-predict-mega/TehranHouse.csv')
categrical_column = ['Address']
data = pd.get_dummies(data, columns=categrical_column, drop_first=True)
data['Area'] = data['Area'].astype(str).str.replace(',', '').astype(int)
X = data.drop(columns=['Price(USD)'])
y = data['Price(USD)']
train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=42)
model = RandomForestRegressor(n_estimators=100,random_state=42)
print(X.dtypes[X.dtypes == 'object'])
model.fit(train_X,train_y)
# model ready
# checking 
pred_y = model.predict(test_X)
mae = mean_absolute_error(test_y,pred_y)
r2 = r2_score(test_y,pred_y)
print("MAE : ",mae)
print("R^2 : ",r2) #0.9984068133231365
# model is perfecto
# now test 
test_data = pd.read_csv('/kaggle/input/house-price-predict-mega/TehranHouse_test.csv')
test_data = pd.get_dummies(test_data,columns=categrical_column,drop_first=True)
test_ids = test_data['ID']
test_data = test_data.reindex(columns=train_X.columns, fill_value=0)
test_data['Area'] = test_data['Area'].astype(str).str.replace(',', '').astype(int)
test_pred = model.predict(test_data)
print(test_data)
# submission file
submission = pd.DataFrame({
    'id':test_ids,
    'Price(USD)':test_pred
})
submission.to_csv('submission.csv', index=False)




