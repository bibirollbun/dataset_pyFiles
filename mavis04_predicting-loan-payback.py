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


import pandas as pd


test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')


train.head(10)


train.isnull().sum()


train.duplicated().sum()


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


X = train[['annual_income','debt_to_income_ratio','credit_score','loan_amount','interest_rate','gender','marital_status','education_level','employment_status','loan_purpose','grade_subgrade']]
y = train[['loan_paid_back']]


print(X)


categorical_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
display(X.head())


model = RandomForestClassifier(n_estimators=160,random_state=42)


model.fit(X,y.values.ravel())


test.head(10)


X_test = test[['annual_income','debt_to_income_ratio','credit_score','loan_amount','interest_rate','gender','marital_status','education_level','employment_status','loan_purpose','grade_subgrade']]


categorical_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
display(X_test.head())


prediction = model.predict(X_test)


prediction.shape


print(prediction)


probabilities = model.predict_proba(X_test)


print(probabilities[:,1])


ID = test[['id']]



print(ID)


result = pd.DataFrame({'id':ID.values.flatten(),'Prediction':probabilities[:,1]})


print(result)


result.to_csv('prediction.csv',index=False)

