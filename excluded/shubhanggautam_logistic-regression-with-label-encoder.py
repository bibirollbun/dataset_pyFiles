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


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train


# encoding categorical data
le = LabelEncoder()
train_en = train.copy()
for col in train_en[['job','marital','education','default','housing','loan','contact','month','poutcome']]:
    train_en[col]= le.fit_transform(train_en[col])
train_en


#test validation split
from sklearn.model_selection import train_test_split
x_train = train_en.drop(columns = ['id','y'])
y_train = train_en['y']
xt,xv,yt,yv = train_test_split(x_train,y_train,test_size = 0.2,random_state = 42)


model = LogisticRegression()
model.fit(xt, yt)
prob = model.predict(xv)


prob


from sklearn.metrics import mean_squared_log_error
# --- RMSLE function
def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


print(rmsle(yv,prob))


test_en = test.copy()
test_en = test_en.drop(columns = ['id'])
for col in test_en[['job','marital','education','default','housing','loan','contact','month','poutcome']]:
    test_en[col]= le.fit_transform(test_en[col])


result = model.predict_proba(test_en)


result


submission = pd.DataFrame()
submission['id'] = test['id']
submission['y'] = result[:,1]


submission.to_csv("submission.csv", index=False)

