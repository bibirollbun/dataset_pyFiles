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


train = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')


train['Previous Claims'] = train['Previous Claims'].fillna(0)
train['Marital Status'] = train['Marital Status'].fillna(0)
train['Customer Feedback'] = train['Customer Feedback'].fillna(0)
test['Previous Claims'] = test['Previous Claims'].fillna(0)
test['Marital Status'] = test['Marital Status'].fillna(0)
test['Customer Feedback'] = test['Customer Feedback'].fillna(0)


float_feature = ['Age','Annual Income','Number of Dependents','Health Score','Vehicle Age','Credit Score','Insurance Duration']
for feature in float_feature:
    train[feature] = train[feature].fillna(train[feature].mean())
    test[feature] = test[feature].fillna(train[feature].mean())


train.isnull().sum()


train['Marital Status'] = train['Marital Status'].map({'Married':1,'Divorced':-2,'Single':-1})
test['Marital Status'] = test['Marital Status'].map({'Married':1,'Divorced':-2,'Single':-1})
train['Gender'] = train['Gender'].map({'Female':1,'Male':-1})
test['Gender'] = test['Gender'].map({'Female':1,'Male':-1})
train['Smoking Status'] = train['Smoking Status'].map({'Yes':1,'No':-1})
test['Smoking Status'] = test['Smoking Status'].map({'Yes':1,'No':-1})
train['Customer Feedback'] = train['Customer Feedback'].map({'Good':1,'Average':0,'Poor':-1})
test['Customer Feedback'] = test['Customer Feedback'].map({'Good':1,'Average':0,'Poor':-1})
train['Policy Type'] = train['Policy Type'].map({'Premium':3, 'Comprehensive':2, 'Basic':1})
test['Policy Type'] = test['Policy Type'].map({'Premium':3, 'Comprehensive':2, 'Basic':1})


test['Policy Start Date'] = pd.to_datetime(test['Policy Start Date']).dt.strftime('%Y')
train['Policy Start Date'] = pd.to_datetime(train['Policy Start Date']).dt.strftime('%Y')


dm1 = pd.get_dummies(train['Education Level'])
dm2 = pd.get_dummies(train['Occupation'])
dm3 = pd.get_dummies(train['Location'])
dm4 = pd.get_dummies(train['Exercise Frequency'])
dm5 = pd.get_dummies(train['Property Type'])
tdm1 = pd.get_dummies(test['Education Level'])
tdm2 = pd.get_dummies(test['Occupation'])
tdm3 = pd.get_dummies(test['Location'])
tdm4 = pd.get_dummies(test['Exercise Frequency'])
tdm5 = pd.get_dummies(test['Property Type'])


train = pd.concat([train,dm1,dm2,dm3,dm4,dm5],axis=1)
test = pd.concat([test,tdm1,tdm2,tdm3,tdm4,tdm5],axis=1)


train = train.drop(['Education Level','Occupation','Location','id','Exercise Frequency','Property Type'],axis = 1)
test = test.drop(['Education Level','Occupation','Location','id','Exercise Frequency','Property Type'],axis = 1)
x = train.drop(['Premium Amount'],axis = 1)
y = train['Premium Amount']


x["Policy Start Date"] = pd.to_numeric(x["Policy Start Date"],errors='coerce')
test["Policy Start Date"] = pd.to_numeric(test["Policy Start Date"],errors='coerce')



x.info()


from xgboost import XGBRegressor
params = {
    'objective':'reg:squarederror',
    'n_estimators':200,
    'learning_rate':0.3,
    'eval_metric':'mae',
    'enable_categorical' : True
}
model = XGBRegressor(**params)


from sklearn.model_selection import train_test_split


x_t,x_v,y_t,y_v = train_test_split(x,y,test_size = 0.2)


model.fit(x_t,y_t)


y_pred = model.predict(x_v)


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_pred,y_v)


r2_score(y_pred,y_v)


test_pred = model.predict(test)


tmp = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')


submission = pd.DataFrame({'id':tmp['id'],'Premium Amount':test_pred})
submission.to_csv('submission.csv',index = False)

