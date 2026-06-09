# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

from matplotlib import pyplot as plt



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




test_data = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")

### Convert Nan (Not a Number) values to 0
test_data.AMT_ANNUITY = np.nan_to_num(test_data.AMT_ANNUITY,0)
test_data.AMT_GOODS_PRICE = np.nan_to_num(test_data.AMT_GOODS_PRICE,0)
test_data.DAYS_EMPLOYED = np.nan_to_num(test_data.DAYS_EMPLOYED,0)
test_data.AMT_REQ_CREDIT_BUREAU_HOUR = np.nan_to_num(test_data.AMT_REQ_CREDIT_BUREAU_HOUR,0)
test_data.AMT_REQ_CREDIT_BUREAU_DAY = np.nan_to_num(test_data.AMT_REQ_CREDIT_BUREAU_DAY,0)
test_data.AMT_REQ_CREDIT_BUREAU_MON = np.nan_to_num(test_data.AMT_REQ_CREDIT_BUREAU_MON,0)
test_data.AMT_REQ_CREDIT_BUREAU_QRT = np.nan_to_num(test_data.AMT_REQ_CREDIT_BUREAU_QRT,0)
test_data.AMT_REQ_CREDIT_BUREAU_YEAR = np.nan_to_num(test_data.AMT_REQ_CREDIT_BUREAU_YEAR,0)



### Convert categorical columnar values to binary values


### NAME_CONTRACT_TYPE 
CONTRACT_TYPE = ['Cash loans','Revolving loans']
enc = OrdinalEncoder(categories = [CONTRACT_TYPE])
train_data['NAME_CONTRACT_TYPE'] = enc.fit_transform(train_data[['NAME_CONTRACT_TYPE']])
train_data.head(15)


### FLAG_OWN_CAR 
OWN_CAR = ['N','Y']
enc = OrdinalEncoder(categories = [OWN_CAR])
train_data['FLAG_OWN_CAR'] = enc.fit_transform(train_data[['FLAG_OWN_CAR']])
train_data.head(15)


### FLAG_OWN_REALTY 
REALTY = ['N','Y']
enc = OrdinalEncoder(categories = [REALTY])
train_data['FLAG_OWN_REALTY'] = enc.fit_transform(train_data[['FLAG_OWN_REALTY']])
train_data.head(15)


### create a subset of the application_train data
### this new data model will contain the binary version of the above 
### categorical columns plus the target column
subset_train_data = train_data[['SK_ID_CURR','NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'TARGET']]
subset_train_data.head(15)


plt.scatter(train_data.AMT_INCOME_TOTAL, train_data.TARGET)


import seaborn as sns
sns.countplot(x=train_data.TARGET, data=train_data)


### Further subdivide the original training set in X_train (80%) and X_test (20%)
X = subset_train_data.iloc[:, 1:4]
y = subset_train_data.iloc[:, 4]
X_train,X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)



### Get a corresponding set of records for SK_ID_CURR as above
U = subset_train_data.iloc[:, 0]
v = subset_train_data.iloc[:, 4]
U_train,U_test, v_train, v_test = train_test_split(U, v, train_size=0.8)


### Build the logistic regression model

from sklearn.linear_model import LogisticRegression 

model = LogisticRegression(class_weight='balanced')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions = (model.predict(X_test) / 100)


model.score(X_test, y_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
### print(confusion_matrix(y_test, y_pred))
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = ["payment", "non-payment"])
cm_disp.plot()


print(classification_report(y_test, y_pred))


output = pd.DataFrame({'SK_ID_CURR': U_test, 'TARGET': (predictions * 100)})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

