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


from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


train_data=pd.read_csv("/kaggle/input/kagglehack2/train.csv")
test_data=pd.read_csv("/kaggle/input/kagglehack2/test.csv")


X_train = train_data.drop(columns=['target'])
y_train = train_data['target']
X_test = test_data.drop(columns=['id'])
X_id=test_data['id']


X_test.head()


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)


model=LinearRegression()   #Ridge()
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
r2 = r2_score(y_val, y_pred)
print(f"Validation R2Score: {r2}")


y_test_pred =model.predict(X_test)


test_data_prediction=pd.DataFrame({
    'id': X_id,
    'target': y_test_pred })


test_data_prediction.to_csv("prediction2_6.csv", index=False)

