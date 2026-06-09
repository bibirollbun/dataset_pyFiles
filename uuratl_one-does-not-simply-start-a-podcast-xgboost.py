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


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np 


train_data = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
train = train_data.copy()
test = test_data.copy()


train.head()


train.isna().sum()


test.isna().sum()


train.describe().T


train['Episode_Sentiment'] = train['Episode_Sentiment'].map({'Neutral': 0, 'Positive': 1, 'Negative': -1})
train['Publication_Day'] = train['Publication_Day'].map({'Thursday':4, 'Saturday':6, 'Tuesday':2, 'Monday':1, 'Sunday':7, 'Wednesday':3, 'Friday':5})
train["Episode"] = train["Episode_Title"].astype(str).str.replace("Episode ", "").astype('int')
train.drop(columns="Episode_Title", axis=1, inplace=True)


def fill_nans(x):
    data = x.copy()
    """
    data, categorical_vars, numerical_vars"""
    categorical_vars = []
    numerical_vars = []
    for col in data.columns:
        if data[col].dtype == 'object':
            categorical_vars.append(col)
            data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            numerical_vars.append(col)
            data[col].fillna(data[col].median(), inplace=True)
    return data, categorical_vars, numerical_vars

def fix_outliers(dataset, numeric_cols, drop=False, treshold=1.5):
    data = dataset.copy()
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
    
        lower_bound = Q1 - treshold * IQR
        upper_bound = Q3 + treshold * IQR
        if drop:
            return data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        else:
            data.loc[data[col] < lower_bound, col] = lower_bound
            data.loc[data[col] > upper_bound, col] = upper_bound
            return data


train, cat_vars, num_vars = fill_nans(train)


num_vars.remove('id')
num_vars


train = fix_outliers(train, list(set(num_vars) - set(['success_rate', 'Episode'])), drop=True)


train.describe()


train[train['Episode_Length_minutes'] == 0]


train[train['Episode_Length_minutes'] <= train['Listening_Time_minutes']]


train = train[train['Episode_Length_minutes'] >= train['Listening_Time_minutes']]
train.describe().T


train['success_rate'] = train['Listening_Time_minutes']/train['Episode_Length_minutes']


cat_vars


lb = LabelEncoder()
train[cat_vars] = train[cat_vars].apply(lb.fit_transform)


train


x = train.drop(['id','Listening_Time_minutes', 'success_rate'],axis=1)
y = train['success_rate']


x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=.2, random_state=42)


model = xgb.XGBRegressor()
model.fit(x_train, y_train)


pred = model.predict(x_test)
score = mean_squared_error(y_test.values, pred)**(1/2)
print(f'Score: {score}')



test.head()


test['Episode_Sentiment'] = test['Episode_Sentiment'].map({'Neutral': 0, 'Positive': 1, 'Negative': -1})
test['Publication_Day'] = test['Publication_Day'].map({'Thursday':4, 'Saturday':6, 'Tuesday':2, 'Monday':1, 'Sunday':7, 'Wednesday':3, 'Friday':5})
test["Episode"] = test["Episode_Title"].astype(str).str.replace("Episode ", "").astype('int')
test.drop(columns="Episode_Title", axis=1, inplace=True)


test, cat_vars_, num_vars_ = fill_nans(test)


test


cat_vars_, num_vars_


test[cat_vars] = test[cat_vars].apply(lb.fit_transform)


pred_ = model.predict(test.drop(columns='id', axis=1))


pred_


submission = pd.DataFrame()
submission[['id', 'Episode_Length_minutes']] = test[['id', 'Episode_Length_minutes']]
submission['success_rate'] = pred_


submission['Listening_Time_minutes'] = submission['success_rate']*submission['Episode_Length_minutes']


submission = submission.drop(columns=['Episode_Length_minutes', 'success_rate'])
submission.to_csv('submission.csv', index=False)

