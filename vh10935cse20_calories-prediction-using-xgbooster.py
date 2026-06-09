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
train=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
sample=pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


print(test.shape)
print(train.shape)
train.head(3)


test.head(3)


print(train.isnull().sum())
print(train.describe())
print(train.info())


print(test.isnull().sum())
print(test.describe())
print(test.info())


from sklearn.preprocessing import LabelEncoder,StandardScaler
le=LabelEncoder()
sc=StandardScaler()


train['Sex']=le.fit_transform(train['Sex'])
test['Sex']=le.transform(test['Sex'])


features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
train['Calories_log'] = np.log1p(train['Calories'])
X = train[features]
y = train['Calories_log']
X_test = test[features]
X_scaled=sc.fit_transform(train[features])
X_test_scaled=sc.transform(test[features])


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error


xgb_model=XGBRegressor(max_depth=6,n_estimators=1000,learning_rate=0.05,random_state=42)


xgb_model.fit(X_scaled,y)


y_pred_log = xgb_model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)


submission=pd.DataFrame({'id':test['id'] ,'Calories':y_pred})
submission.to_csv('/kaggle/working/submission.csv',index=False)


y_pred_train_log = xgb_model.predict(X_scaled)
y_pred_train = np.expm1(y_pred_train_log)
rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), y_pred_train))
print(f'RMSLE: {rmsle:.5f}')




