# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import log_loss
import lightgbm as lgb
from sklearn.metrics import log_loss




# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data = pd.read_csv('/kaggle/input/multiclassificationtask/train.csv')  
print(train_data.head())  

print("Train Data Info:")
print(train_data.info())  

print("\nTrain Data Description:")
print(train_data.describe())  



null_percentage = (train_data.isnull().sum() / len(train_data)) * 100

print(null_percentage)


train_data = train_data.drop(columns=['Drug', 'Cholesterol', 'Tryglicerides', 'Ascites' , 'Hepatomegaly' , 'Spiders' , 'Copper' , 'Alk_Phos' , 'SGOT'])

numerical_cols = train_data.select_dtypes(include=['float64']).columns
train_data[numerical_cols] = train_data[numerical_cols].fillna(train_data[numerical_cols].median())

categorical_cols = train_data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
    
print(train_data.isnull().sum())



print(train_data['Edema'].value_counts())
print(train_data['Status'].value_counts())
print(train_data['Sex'].value_counts())


train_data = pd.get_dummies(train_data, columns=['Sex', 'Edema'], drop_first=False)

scaler = StandardScaler()

numerical_columns = ['N_Days', 'Age', 'Bilirubin', 'Albumin', 'Platelets', 'Prothrombin', 'Stage']

train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])

print(train_data.head())


X = train_data.drop(columns=['Status', 'id'])
le = LabelEncoder()
y = le.fit_transform(train_data['Status'])  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train size: {X_train.shape[0]} samples")
print(f"Validation size: {X_val.shape[0]} samples")

print("Statuslar va ularning raqamlari:", dict(zip(le.classes_, range(len(le.classes_)))))


model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  

y_pred_prob = model.predict_proba(X_val)

logloss = log_loss(y_val, y_pred_prob, labels=[0, 1, 2])  # 0, 1, 2 sinflari bilan log loss hisoblash

print(f"Log Loss: {logloss}")



model_xgb = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=42)
model_xgb.fit(X_train, y_train)

y_pred_prob_xgb = model_xgb.predict_proba(X_val)

logloss_xgb = log_loss(y_val, y_pred_prob_xgb, labels=[0, 1, 2])
print(f"Log Loss (XGBoost): {logloss_xgb}")


model_lgb = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42)
model_lgb.fit(X_train, y_train)

y_pred_prob_lgb = model_lgb.predict_proba(X_val)


logloss_lgb = log_loss(y_val, y_pred_prob_lgb, labels=[0, 1, 2])
print(f"Log Loss (LightGBM): {logloss_lgb}")


test_data = pd.read_csv('/kaggle/input/multiclassificationtask/test.csv')


print("Test Data Info:")
print(test_data.info())  

print("\nTest Data Description:")
print(test_data.describe())  


null_percentage1 = (test_data.isnull().sum() / len(test_data)) * 100

print(null_percentage1)


test_data = test_data.drop(columns=['Drug', 'Cholesterol', 'Tryglicerides', 'Ascites' , 'Hepatomegaly' , 'Spiders' , 'Copper' , 'Alk_Phos' , 'SGOT'])

numerical_cols = test_data.select_dtypes(include=['float64']).columns
test_data[numerical_cols] = test_data[numerical_cols].fillna(test_data[numerical_cols].median())

categorical_cols = test_data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    test_data[col] = test_data[col].fillna(test_data[col].mode()[0])
    
print(test_data.isnull().sum())


test_data = pd.get_dummies(test_data, columns=['Sex', 'Edema'], drop_first=False)

test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])

X_test = test_data.drop(columns=['id'])


y_test_prob = model.predict_proba(X_test)

submission = pd.DataFrame(y_test_prob, columns=['Status_C', 'Status_CL', 'Status_D'])

submission['id'] = test_data['id']

submission = submission[['id', 'Status_C', 'Status_CL', 'Status_D']]

submission.to_csv('submission.csv', index=False)


