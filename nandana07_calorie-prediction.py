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


train= pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test= pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
train.head()


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

train['Sex']= train['Sex'].map({'male':0,'female':1})
test['Sex']= test['Sex'].map({'male':0,'female':1})


x=train.drop(['id','Calories'],axis=1)
y=train['Calories']
x_test= test.drop(['id'], axis=1)


x_train, x_val, y_train, y_val= train_test_split(x,y,test_size=0.2, random_state=42)


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


model=RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(x_train, y_train)


val_preds= model.predict(x_val)
print("RMSLE:",rmsle(y_val,val_preds))


test_preds= model.predict(x_test)


subm= pd.DataFrame({
    'id': test['id'],
    'Calries': test_preds
})
subm.to_csv('subm.csv', index=False)








