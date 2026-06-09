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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  RobustScaler,OrdinalEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,recall_score,f1_score,classification_report


train=pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
train.head()


train.isnull().sum()



Onc=OrdinalEncoder()
train[['gender','marital_status','education_level','employment_status','loan_purpose','grade_subgrade']]=Onc.fit_transform(train[['gender','marital_status','education_level','employment_status','loan_purpose','grade_subgrade']])
train.head()


X=train.drop(columns=['id','loan_paid_back'])
Y=train['loan_paid_back']


sc=RobustScaler()
X_scal=sc.fit_transform(X)


 X_train,X_val,Y_train,Y_val=train_test_split(X_scal,Y,test_size=0.3,random_state=42,stratify=Y)
# svm=SVC(kernel='rbf',probability=True,C=1.0,random_state=42)
# svm.fit(X_train,Y_train)
# y_pred=svm.predict(X_val)
# print(accuracy_score(Y_val,Y_pred))


xgb=XGBClassifier(n_estimators=100,max_depth=6,subsample=0.6,learning_rate=0.06,gamma=0,colsample_bytree=0.7,random_state=42,eval_metric='logloss',reg_alpha=1,reg_lambda=0)
xgb.fit(X_train,Y_train)
Y_pred=xgb.predict(X_val)
print(accuracy_score(Y_val,Y_pred))
print(classification_report(Y_val,Y_pred))


lgbm=LGBMClassifier(max_depth=6,subsample=0.8,n_estimators=100,random_state=42,colsample_bytree=0.6,learning_rate=0.05,reg_alpha=0.1,reg_lambda=1,scale_pos_weight=83650/332145)
lgbm.fit(X_train,Y_train)
Y_pred=lgbm.predict(X_val)
print(accuracy_score(Y_val,Y_pred))
print(classification_report(Y_val,Y_pred))


test=pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
test_ids=test['id']
test.head()


test[['gender','marital_status','education_level','employment_status','loan_purpose','grade_subgrade']]=Onc.transform(test[['gender','marital_status','education_level','employment_status','loan_purpose','grade_subgrade']])


X_test=test.drop(columns=['id'])
X_test_scal=sc.transform(X_test)


y_pred=xgb.predict(X_test)
y_pred=lgbm.predict(X_test)
submission=pd.DataFrame({
    "id":test_ids,
    "loan_paid_back":y_pred
    
})
submission.to_csv("submission.csv",index=False)




