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


#importing useful libraries
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


#Loading Datasets

X_full = pd.read_csv('../input/playground-series-s5e7/train.csv')
X_test = pd.read_csv('../input/playground-series-s5e7/test.csv')
X_test_og = X_test
"""X_full = X_full.dropna()"""


X_test.isna().sum()


#updating categorical data into binary terms

def turn_to_numeric(Dataframe, column, string):
    new = Dataframe[column] == string
    new += 0
    Dataframe[column] = new

# 1 means Yes, 0 implies No
"""turn_to_numeric(X_full, 'Stage_fear', 'No')
turn_to_numeric(X_full, 'Drained_after_socializing', "Yes")
turn_to_numeric(X_test, 'Stage_fear', 'No')
turn_to_numeric(X_test, 'Drained_after_socializing', 'Yes')
"""

X_full = X_full.sort_values(by = 'Stage_fear')
X_test = X_test.sort_values(by = 'Stage_fear')
X_full['Stage_fear'] = pd.Categorical(X_full['Stage_fear']).codes
X_test['Stage_fear'] = pd.Categorical(X_test['Stage_fear']).codes

X_full = X_full.sort_values(by = 'Drained_after_socializing')
X_test = X_test.sort_values(by = 'Drained_after_socializing')
X_full['Drained_after_socializing'] = pd.Categorical(X_full['Drained_after_socializing']).codes
X_test['Drained_after_socializing'] = pd.Categorical(X_test['Drained_after_socializing']).codes

X_full = X_full.sort_values(by = 'id')
X_test = X_test.sort_values(by = 'id')

# 1 means Exrovert and 0 refers to Introvert
turn_to_numeric(X_full, 'Personality', 'Extrovert')




X_full['new1'] = X_full.Drained_after_socializing * X_full.Friends_circle_size
X_test['new1'] = X_test.Drained_after_socializing * X_test.Friends_circle_size

X_full['new2'] = X_full.Time_spent_Alone + X_full.Going_outside
X_test['new2'] = X_test.Time_spent_Alone + X_test.Going_outside

X_full['new3'] = X_full.Drained_after_socializing * X_full.Stage_fear
X_test['new3'] = X_test.Drained_after_socializing * X_test.Stage_fear




"""X_full = X_full.drop(['Post_frequency'], axis = 1)
X_test = X_test.drop(['Post_frequency'], axis = 1)"""


X_full = X_full.fillna(500)
X_test = X_test.fillna(500)
X_full.sum()



# Removing not needed columns before splitting the data

y = X_full["Personality"]
X = X_full.drop(["id", "Personality"], axis = 1)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=3)


my_imputer = SimpleImputer()


cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

for col in cols_with_missing:
    X_train[col + '_was_missing'] = X_train[col].isnull()
    X_valid[col + '_was_missing'] = X_valid[col].isnull()

# Imputation

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns





##### model creation
model = RandomForestClassifier(n_estimators = 350, random_state = 3)
model.fit(imputed_X_train, y_train)


test_pred = model.predict(imputed_X_valid)
test_pred


test_pred.mean()


# In case of regression models such thing would have been needed.
# Since it is a classification problem.
for i in range(len(test_pred)):
    test_pred[i] = 1 if test_pred[i]>= 0.7 else 0


accuracy_score(y_valid, test_pred)


n = sum((y_valid - test_pred) !=0)
100 - (n / len(y_valid) * 100)


imputed_X_full = pd.DataFrame(my_imputer.fit_transform(X_full))
imputed_X_full.columns = X_full.columns
imputed_X_full = imputed_X_full.drop(['id', 'Personality'], axis = 1)

model.fit(imputed_X_full, y)


# taking a glance at the test data
X_test.head()


imputed_X_test = pd.DataFrame(my_imputer.fit_transform(X_test))
imputed_X_test.columns = X_test.columns
imputed_X_test = imputed_X_test.drop(['id'], axis = 1)
final_prediction = model.predict(imputed_X_test)

final_prediction


list = []
for i in range(len(final_prediction)):
    if final_prediction[i]>=0.7:
        list.append("Extrovert")
    else :
        list.append("Introvert")

output = pd.DataFrame({'id': X_test_og.id,
                       'Personality': list})
output.to_csv('submission.csv', index=False)


len(X_test)
output.groupby("Personality").count()

