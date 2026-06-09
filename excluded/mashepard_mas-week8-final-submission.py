# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

from lightgbm import early_stopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from matplotlib import pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


### Train data set

train_data = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")

### Convert Nan (Not a Number) values to 0
train_data.AMT_ANNUITY = np.nan_to_num(train_data.AMT_ANNUITY,0)
train_data.AMT_GOODS_PRICE = np.nan_to_num(train_data.AMT_GOODS_PRICE,0)
train_data.DAYS_EMPLOYED = np.nan_to_num(train_data.DAYS_EMPLOYED,0)
train_data.AMT_REQ_CREDIT_BUREAU_HOUR = np.nan_to_num(train_data.AMT_REQ_CREDIT_BUREAU_HOUR,0)
train_data.AMT_REQ_CREDIT_BUREAU_DAY = np.nan_to_num(train_data.AMT_REQ_CREDIT_BUREAU_DAY,0)
train_data.AMT_REQ_CREDIT_BUREAU_WEEK = np.nan_to_num(train_data.AMT_REQ_CREDIT_BUREAU_WEEK,0)
train_data.AMT_REQ_CREDIT_BUREAU_MON = np.nan_to_num(train_data.AMT_REQ_CREDIT_BUREAU_MON,0)
train_data.AMT_REQ_CREDIT_BUREAU_QRT = np.nan_to_num(train_data.AMT_REQ_CREDIT_BUREAU_QRT,0)
train_data.AMT_REQ_CREDIT_BUREAU_YEAR = np.nan_to_num(train_data.AMT_REQ_CREDIT_BUREAU_YEAR,0)
train_data.CODE_GENDER = train_data.CODE_GENDER.replace("XNA", "M")

train_data.head(5)


### Total number of payments/non-payments amongst applicants
### 0: Did NOT miss a loan payment
### 1: Missed loan payment

train_data['TARGET'].value_counts(normalize=True) * 100


### Convert categorical columnar values to binary values

### CODE_GENDER 
CODE_GENDER  = ['F','M']
enc = OrdinalEncoder(categories = [CODE_GENDER])
train_data['CODE_GENDER'] = enc.fit_transform(train_data[['CODE_GENDER']])
train_data.head(15)


### Convert categorical columnar values to binary values

### NAME_CONTRACT_TYPE 
CONTRACT_TYPE = ['Cash loans','Revolving loans']
enc = OrdinalEncoder(categories = [CONTRACT_TYPE])
train_data['NAME_CONTRACT_TYPE'] = enc.fit_transform(train_data[['NAME_CONTRACT_TYPE']])
train_data.head(15)


### Test data set

test_data = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")

### Convert Nan (Not a Number) values to 0
test_data.AMT_ANNUITY = np.nan_to_num(test_data.AMT_ANNUITY,0)
test_data.AMT_GOODS_PRICE = np.nan_to_num(test_data.AMT_GOODS_PRICE,0)
test_data.DAYS_EMPLOYED = np.nan_to_num(test_data.DAYS_EMPLOYED,0)
test_data.AMT_REQ_CREDIT_BUREAU_HOUR = np.nan_to_num(test_data.AMT_REQ_CREDIT_BUREAU_HOUR,0)
test_data.AMT_REQ_CREDIT_BUREAU_DAY = np.nan_to_num(test_data.AMT_REQ_CREDIT_BUREAU_DAY,0)
test_data.AMT_REQ_CREDIT_BUREAU_WEEK = np.nan_to_num(test_data.AMT_REQ_CREDIT_BUREAU_WEEK,0)
test_data.AMT_REQ_CREDIT_BUREAU_MON = np.nan_to_num(test_data.AMT_REQ_CREDIT_BUREAU_MON,0)
test_data.AMT_REQ_CREDIT_BUREAU_QRT = np.nan_to_num(test_data.AMT_REQ_CREDIT_BUREAU_QRT,0)
test_data.AMT_REQ_CREDIT_BUREAU_YEAR = np.nan_to_num(test_data.AMT_REQ_CREDIT_BUREAU_YEAR,0)
test_data.CODE_GENDER = test_data.CODE_GENDER.replace("XNA", "M")

test_data.head(5)


### Convert categorical columnar values to binary values

### CODE_GENDER 
CODE_GENDER  = ['F','M']
enc = OrdinalEncoder(categories = [CODE_GENDER])
test_data['CODE_GENDER'] = enc.fit_transform(test_data[['CODE_GENDER']])
test_data.head(15)


### Convert categorical columnar values to binary values

### NAME_CONTRACT_TYPE 
CONTRACT_TYPE = ['Cash loans','Revolving loans']
enc = OrdinalEncoder(categories = [CONTRACT_TYPE])
test_data['NAME_CONTRACT_TYPE'] = enc.fit_transform(test_data[['NAME_CONTRACT_TYPE']])
test_data.head(15)


y = train_data["TARGET"]

X = train_data[["AMT_INCOME_TOTAL","AMT_CREDIT", "AMT_ANNUITY", "CODE_GENDER", "NAME_CONTRACT_TYPE", "DAYS_BIRTH", "DAYS_EMPLOYED"]]
X_train,X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8)
X_train.shape, X_valid.shape, y_train.shape,y_valid.shape


### Create a Light GBM model and evaluate performance 
import lightgbm
lgbm_train_data = lightgbm.Dataset(X_train, label=y_train)
lgbm_valid_data = lightgbm.Dataset(X_valid, label=y_valid)


# Specify the parameters for LightGBM

parameters = {'objective': 'binary',
              'metric': 'auc',
              'is_unbalance': 'true',
              'boosting': 'gbdt',
              'num-leaves': 20,
              'feature_fraction': 0.8,
              'bagging_faction': 0.5,
              'bagging_freq': 20,
              'learning_rate': 0.01,
              'verbose': -1    
            }


# Train the LightGBM model for maximum 5000 rounds. Early stopping criteria is 50 iterations
from lightgbm import early_stopping

model_lgbm = lightgbm.train(parameters,
                           lgbm_train_data,
                           valid_sets=lgbm_valid_data,
                           num_boost_round=5000,
                           callbacks=[early_stopping(stopping_rounds=50)]
                           )


from sklearn.metrics import roc_auc_score

y_train_pred = model_lgbm.predict(X_train)
y_valid_pred = model_lgbm.predict(X_valid)

print("AUC Train: {:4f}\nAUC Valid: {:.4f}".format(roc_auc_score(y_train, y_train_pred),
                                                   roc_auc_score(y_valid, y_valid_pred)))


# Find the predictions for test data
###X_test = test_data[["AMT_INCOME_TOTAL","AMT_CREDIT", "AMT_ANNUITY", "CODE_GENDER", "NAME_CONTRACT_TYPE, "DAYS_BIRTH", "DAYS_EMPLOYED"]]
###X_test = test_data[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "CODE_GENDER", "NAME_CONTRACT_TYPE", "DAYS_BIRTH", "DAYS_EMPLOYED"]]


X_test = test_data[["AMT_INCOME_TOTAL","AMT_CREDIT", "AMT_ANNUITY", "CODE_GENDER", "NAME_CONTRACT_TYPE", "DAYS_BIRTH", "DAYS_EMPLOYED"]]
Y_test = test_data[['SK_ID_CURR']]
y_test_pred = model_lgbm.predict(X_test)


output = pd.DataFrame({'SK_ID_CURR': Y_test.SK_ID_CURR, 'TARGET':y_test_pred})

output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


# Print accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_test, y_test_pred)
print(f"accuracy: {acc}")


# Confusion matrix

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test, y_test_pred, labels=[0,1])
print("Confusion Matrix:")
print(cm)

