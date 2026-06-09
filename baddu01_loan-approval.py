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


df=pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')


df.head()


df.tail()


df.info()


df.isnull().sum()


df=df.drop('id',axis=1)


from ydata_profiling import ProfileReport


profile=ProfileReport(df,title='Profile Report')
profile


num_cols=df.select_dtypes(include=['number','float']).columns.tolist()
num_cols.remove('loan_status')
cat_cols=df.select_dtypes(exclude=['number','float']).columns.tolist()


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


procsr=ColumnTransformer(transformers=[
    ('ohe',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat_cols)
])


X=df.drop('loan_status',axis=1)
y=df['loan_status']


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


X_train_trans=procsr.fit_transform(X_train)
X_test_trans=procsr.transform(X_test)


from sklearn.linear_model import LogisticRegression


lr=LogisticRegression()


lr.fit(X_train_trans,y_train)


from sklearn.metrics import roc_auc_score


roc_auc_score(y_test,lr.predict(X_test_trans))


from sklearn.tree import DecisionTreeClassifier


dt=DecisionTreeClassifier()


dt.fit(X_train_trans,y_train)


roc_auc_score(y_test,dt.predict(X_test_trans))


from sklearn.ensemble import RandomForestClassifier


rf=RandomForestClassifier(n_estimators=500,n_jobs=-1)


rf.fit(X_train_trans,y_train)


roc_auc_score(y_test,rf.predict(X_test_trans))


from xgboost import XGBClassifier


xgb=XGBClassifier(n_jobs=-1,eta=0.1)


xgb.fit(X_train_trans,y_train)


roc_auc_score(y_test,xgb.predict(X_test_trans))


from sklearn.svm import SVC


svc=SVC()


svc.fit(X_train_trans,y_train)


roc_auc_score(y_test,svc.predict(X_test_trans))


from lightgbm import LGBMClassifier


lgb=LGBMClassifier(n_jobs=-1,learning_rate=0.1)


lgb.fit(X_train_trans,y_train)


roc_auc_score(y_test,lgb.predict(X_test_trans))


from sklearn.pipeline import Pipeline


final_pipeline=Pipeline(steps=[
    ('procsr',procsr),
    ('model',lgb)
])


test=pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')


test.drop('id',axis=1)


final_pipeline.fit(X,y)


preds=final_pipeline.predict(test)


output = pd.DataFrame({
    'id': test['id'],
    'loan_status': preds
})

output.to_csv('submission.csv', index=False)




