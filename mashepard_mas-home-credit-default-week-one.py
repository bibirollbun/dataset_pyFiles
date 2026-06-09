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

train_data.head(5)


### Test Data set

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

test_data.head(5)


### Total number of payments/non-payments amongst applicants
### 0: Did NOT miss a loan payment
### 1: Missed loan payment

train_data['TARGET'].value_counts(normalize=True) * 100


### Selected group aggregations
qry_result = pysqldf("Select sum(CASE WHEN CODE_GENDER = 'M' and TARGET = 0 and AMT_INCOME_TOTAL <=100000 THEN 1 ELSE 0 END) Male_100000_Paid, sum(CASE WHEN CODE_GENDER = 'M' and TARGET = 1 and AMT_INCOME_TOTAL <=100000 THEN 1 ELSE 0 END) Male_100000_UnPaid from train_data")
print(qry_result)


### Group aggregations

### 0: Did NOT miss a loan payment
### 1: Missed loan payment

### Male, paid loans (TARGET - 0), by total income
qry = ("Select sum(CASE WHEN CODE_GENDER = 'M' and TARGET = 0 and AMT_INCOME_TOTAL <=100000 THEN 1 ELSE 0 END) Male_100000_Paid,\
        sum(CASE WHEN CODE_GENDER = 'M' and TARGET = 0 and AMT_INCOME_TOTAL BETWEEN 100001 AND 200000 THEN 1 ELSE 0 END) Male_200000_Paid,\
        sum(CASE WHEN CODE_GENDER = 'M' and TARGET = 0 and AMT_INCOME_TOTAL BETWEEN 200001 AND 300000 THEN 1 ELSE 0 END) Male_300000_Paid, \
        sum(CASE WHEN CODE_GENDER = 'M' and TARGET = 0 and AMT_INCOME_TOTAL > 300001 THEN 1 ELSE 0 END) Male_300001_Up_Paid \
        from train_data")

qry_result = pysqldf(qry)
print("Male, paid loans (TARGET - 0), by total income")
print(qry_result)


### Male, missed loan payment (TARGET = 1), by total income
qry = ("Select sum(CASE WHEN CODE_GENDER = 'M' and TARGET = 1 and AMT_INCOME_TOTAL <=100000 THEN 1 ELSE 0 END) Male_100000_UnPaid,\
        sum(CASE WHEN CODE_GENDER = 'M' and TARGET = 1 and AMT_INCOME_TOTAL BETWEEN 100001 AND 200000 THEN 1 ELSE 0 END) Male_200000_UnPaid,\
        sum(CASE WHEN CODE_GENDER = 'M' and TARGET = 1 and AMT_INCOME_TOTAL BETWEEN 200001 AND 300000 THEN 1 ELSE 0 END) Male_300000_UnPaid, \
        sum(CASE WHEN CODE_GENDER = 'M' and TARGET = 1 and AMT_INCOME_TOTAL > 300001 THEN 1 ELSE 0 END) Male_300001_Up_UnPaid from train_data")
        

qry_result = pysqldf(qry)
print("Male, missed loan payment (TARGET = 1), by total income")
print(qry_result)




### Female, paid loans (TARGET - 0), by total income
qry = ("Select sum(CASE WHEN CODE_GENDER = 'F' and TARGET = 0 and AMT_INCOME_TOTAL <=100000 THEN 1 ELSE 0 END) Female_100000_Paid,\
        sum(CASE WHEN CODE_GENDER = 'F' and TARGET = 0 and AMT_INCOME_TOTAL BETWEEN 100001 AND 200000 THEN 1 ELSE 0 END) Female_200000_Paid,\
        sum(CASE WHEN CODE_GENDER = 'F' and TARGET = 0 and AMT_INCOME_TOTAL BETWEEN 200001 AND 300000 THEN 1 ELSE 0 END) Female_300000_Paid, \
        sum(CASE WHEN CODE_GENDER = 'F' and TARGET = 0 and AMT_INCOME_TOTAL > 300001 THEN 1 ELSE 0 END) Female_300001_Up_Paid \
        from train_data")

qry_result = pysqldf(qry)
print("Female, paid loans (TARGET - 0), by total income")
print(qry_result)

### Female, missed loan payment (TARGET = 1) , by total income
qry = ("Select sum(CASE WHEN CODE_GENDER = 'F' and TARGET = 1 and AMT_INCOME_TOTAL <=100000 THEN 1 ELSE 0 END) Female_100000_Unpaid,\
        sum(CASE WHEN CODE_GENDER = 'F' and TARGET = 1 and AMT_INCOME_TOTAL BETWEEN 100001 AND 200000 THEN 1 ELSE 0 END) Female_200000_Unpaid,\
        sum(CASE WHEN CODE_GENDER = 'F' and TARGET = 1 and AMT_INCOME_TOTAL BETWEEN 200001 AND 300000 THEN 1 ELSE 0 END) Female_300000_Unpaid, \
        sum(CASE WHEN CODE_GENDER = 'F' and TARGET = 1 and AMT_INCOME_TOTAL > 300001 THEN 1 ELSE 0 END) Female_300001_Up_Unpaid \
        from train_data")

qry_result = pysqldf(qry)
print("Female, missed loans payments (TARGET - 1), by total income")
print(qry_result)



# Data visualizations

### Missed payments by contract type 
qry = ("Select Count(TARGET) as Nbr_Missed_Payments, NAME_CONTRACT_TYPE \
       from train_data \
       Where TARGET = 1 \
       Group By TARGET, NAME_CONTRACT_TYPE")
qry_result = pysqldf(qry)
qry_result.plot(kind='bar', x='NAME_CONTRACT_TYPE', y='Nbr_Missed_Payments')
plt.title("Missed Payments by Contract Type")
plt.show()


### Missed payments by education type 
qry = ("Select Count(TARGET) as Nbr_Missed_Payments, NAME_EDUCATION_TYPE \
       from train_data \
       Where TARGET = 1 \
       Group By TARGET, NAME_EDUCATION_TYPE")
qry_result = pysqldf(qry)
qry_result.plot(kind='bar', x='NAME_EDUCATION_TYPE', y='Nbr_Missed_Payments')
plt.title("Missed Payments by Education Type")
plt.show()


### Build a baseline model
from sklearn.ensemble import RandomForestClassifier

y = train_data["TARGET"] 

features = ["NAME_EDUCATION_TYPE","OCCUPATION_TYPE","NAME_HOUSING_TYPE","NAME_CONTRACT_TYPE"]

X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = (model.predict(X_test) / 100)


### rf.score(X_test, y_test)

### from sklearn.metrics import classification_report
### print(classification_report(y_test, y_pred))



output = pd.DataFrame({'SK_ID_CURR': test_data.SK_ID_CURR, 'TARGET': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

