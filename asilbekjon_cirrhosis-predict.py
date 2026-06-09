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
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from xgboost import XGBClassifier


train=pd.read_csv('/kaggle/input/multiclassificationtask/train.csv')


test=pd.read_csv("/kaggle/input/multiclassificationtask/test.csv")


train.info()


train.describe()


train.isnull().sum()


train['Age'] = train['Age']//365


train.Status.value_counts()


train=train[train['Status'] !='Y']


#encode
encode=LabelEncoder()
train['Status']=encode.fit_transform(train.Status.values)
train.Status.value_counts()
# 0- C
# 1-CL
# 2-D


train.drop(['id'],axis=1)


#categirical and numeric columns
categorical=['Drug','Sex','Ascites','Hepatomegaly','Spiders','Edema']
numeric=['N_Days','Age','Bilirubin','Cholesterol','Albumin','Copper','Alk_Phos','SGOT','Tryglicerides','Platelets','Prothrombin','Stage']


#numeric pipeline
num_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaler',StandardScaler())
])
#categorical pipeline
cat_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder())
])
#full pipeline
full_pipeline=ColumnTransformer([
    ('num',num_pipeline,numeric),
    ('cat',cat_pipeline,categorical)
])


X=train.drop('Status',axis=1)
y=train['Status'].copy()


y.value_counts()


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)


X_train_prepared=full_pipeline.fit_transform(X_train)
X_test_prepared=full_pipeline.transform(X_test)


RF=RandomForestClassifier()
RF.fit(X_train_prepared,y_train)
y_predict=RF.predict(X_test_prepared)
y_proba = RF.predict_proba(X_test_prepared)
y_proba_clipped=np.clip(y_proba, 1e-15, 1 - 1e-15)
logloss = log_loss(y_test,y_proba_clipped)
print(classification_report(y_test,y_predict))
print('Accuracy',accuracy_score(y_test,y_predict))
print('Log Loss:', logloss)



conf_mat=confusion_matrix(y_test,y_predict)
sns.heatmap(conf_mat,annot=True,fmt='g')
plt.show()


DT=DecisionTreeClassifier()
DT.fit(X_train_prepared,y_train)
y_predict_1=DT.predict(X_test_prepared)
y_proba_1 = DT.predict_proba(X_test_prepared)
y_proba_clipped_1=np.clip(y_proba_1, 1e-15, 1 - 1e-15)
logloss_1 = log_loss(y_test, y_proba_clipped_1)
print(classification_report(y_test,y_predict_1))
print('Accuracy',accuracy_score(y_test,y_predict_1))
print('Log Loss:', logloss_1)


conf_mat=confusion_matrix(y_test,y_predict_1)
sns.heatmap(conf_mat,annot=True,fmt='g')
plt.show()


GB=GradientBoostingClassifier()
GB.fit(X_train_prepared,y_train)
y_predict_2=GB.predict(X_test_prepared)
y_proba_2= GB.predict_proba(X_test_prepared)
y_proba_clipped_2=np.clip(y_proba_2, 1e-15, 1 - 1e-15)
logloss_2= log_loss(y_test, y_proba_clipped_2)
print(classification_report(y_test,y_predict_2))
print('Accuracy',accuracy_score(y_test,y_predict_2))
print('Log Loss:', logloss_2)


XGB=XGBClassifier()
XGB.fit(X_train_prepared,y_train)
y_predict_3 = XGB.predict(X_test_prepared)
y_proba_3= XGB.predict_proba(X_test_prepared)
y_proba_clipped_3=np.clip(y_proba_3, 1e-15, 1 - 1e-15)
logloss_3= log_loss(y_test, y_proba_clipped_3)
print(classification_report(y_test, y_predict_3))
print("Accuracy:", accuracy_score(y_test,y_predict_3))
print('Log Loss:', logloss_3)


test


test_id=test['id'].copy()


test['Age']=test['Age']//365


test.drop('id',axis=1)


test_prepared=full_pipeline.transform(test)


y_test_predict=GB.predict(test_prepared)
y_test_proba=GB.predict_proba(test_prepared)
prepared_y=np.clip(y_test_proba, 1e-15, 1 - 1e-15)


# 0- C
# 1-CL
# 2-D
submission_cirrhosis= pd.DataFrame({
    'id':test_id,
    'Status_C':prepared_y[:,0],
    'Status_CL':prepared_y[:,1],
    'Status_D':prepared_y[:,2]
})
submission_cirrhosis.to_csv('submission_cirrhosis1.csv',index=False)


# Saving Model
import pickle

filename = 'GradientBoostingClassifier.pkl'
with open(filename, 'wb') as file:
    pickle.dump(GB, file)


submission_cirrhosis

