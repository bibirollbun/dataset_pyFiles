# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split   # For splitting datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder       # For scaling data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # For evaluation metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

import tensorflow as tf         # For deep learning (TensorFlow)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import datetime as dt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train= pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', index_col= 'id')
print(train.head())

test= pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', index_col= 'id')
print(test.head())


print(train.isnull().sum(), test.isnull().sum())


def hottime(data):
    data['date'] = pd.to_datetime(data['date'])
    
    # Extract year, month, and day into separate columns
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    
    data= data.drop('date', axis=1)

    return data

train= hottime(train)
test= hottime(test)
test.head()


def encoding(data):
    cats= []
    for column in data.select_dtypes(include=['object', 'int']).columns:
        cats.append(column)
    
    encoder= LabelEncoder()
    for cat in cats:
        data[cat]= encoder.fit_transform(data[cat])
    
    return data

train= encoding(train)
test= encoding(test)
test.head()


from sklearn.impute import KNNImputer
def null_filler(data):
    imputer = KNNImputer(n_neighbors=3)
    data_imputed_knn = imputer.fit_transform(data)
    data_imputed_knn = pd.DataFrame(data_imputed_knn, columns=data.columns)
    return data_imputed_knn

train_imp= null_filler(train)
test_imp= null_filler(test)
print(train_imp.head(), test_imp.head())


train_imp.isna().sum()
test_imp.isna().sum()


X= train_imp[['year', 'month', 'day', 'country', 'store', 'product']].to_numpy()
y= train_imp[['num_sold']].to_numpy()
X_new= test_imp


X_train, X_cv, y_train, y_cv= train_test_split(X, y, test_size= 0.05)
X_cv, X_test, y_cv, y_test= train_test_split(X, y, test_size= 0.50)

print(X_train.shape, X_cv.shape, X_test.shape, y_cv.shape)


def mape(y, yhat):
    r=0
    for n in y:
        r= (abs(y-yhat)/y)*100
    score= r/len(y)
    return score


model = LGBMRegressor(
    boosting_type='gbdt',
    learning_rate=0.1,
    n_estimators=1000,
    early_stopping_rounds=10
)


model.fit(
    X_train, y_train,
    eval_set=[(X_cv, y_cv)],
    eval_metric='mae'
)



y_pred= model.predict(X_test)


y_test[1051], y_pred[1051]


yhat= model.predict(X_new)
yhat[0:10]


submission = pd.DataFrame({'id': pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')['id'], 'num_sold': yhat})
print(submission.head())


submission.to_csv('submission.csv', index=False)


!ls




