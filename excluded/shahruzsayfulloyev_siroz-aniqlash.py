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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import log_loss


df_train = pd.read_csv('/kaggle/input/multiclassificationtask/train.csv')
df_test = pd.read_csv('/kaggle/input/multiclassificationtask/test.csv')
df_submission = pd.read_csv('/kaggle/input/multiclassificationtask/sample_submission.csv')
df_submission


df_train


df_train['Age_years'] = (df_train['Age']/365).astype(float)
df_train.drop(['Age', 'id'], axis=1, inplace=True)
df_train = df_train[df_train['Status']!='Y']


df_train.isnull().sum()


df_train.Drug.value_counts()


df_train.Sex.value_counts()


df_train.Ascites.value_counts()


df_train.Hepatomegaly.value_counts()


df_train.Spiders.value_counts()


df_train.Edema.value_counts()


df_train.Stage.value_counts()


X = df_train.drop('Status', axis=1)
y = df_train['Status']


enc = LabelEncoder()
y = enc.fit_transform(y)


Statis_label = enc.classes_
Statis_label


objects = X.select_dtypes(include=['object']).columns.to_list()
nums = X.select_dtypes(include=['float64']).columns.to_list()


objects_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


preprocessor = ColumnTransformer([
    ('objects', objects_pipeline, objects),
    ('nums', num_pipeline, nums)
])


X_prepared = preprocessor.fit_transform(X)
X_prepared


X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.2,stratify = y, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


XGB_model = XGBClassifier(n_estimators=100, random_state=42)
XGB_model.fit(X_train_scaled, y_train)
XGB_proba = XGB_model.predict_proba(X_test_scaled)
XGB_loss = log_loss(y_test, XGB_proba, labels = [0,1,2])
XGB_loss


df_test


df_test_id=pd.DataFrame(df_test['id'])


df_test['Age_years'] = (df_test['Age']/365).astype(float)
df_test.drop(['Age', 'id'], axis=1, inplace=True)


df_test_prepared = preprocessor.transform(df_test)
df_test_scaled = scaler.transform(df_test_prepared)
df_test_proba = XGB_model.predict_proba(df_test_scaled)


submission = pd.DataFrame(df_test_proba, columns=['Status_C','Status_CL','Status_D'])
submission['id'] = df_test_id['id']
submission = submission[['id','Status_C','Status_CL','Status_D']]
submission


submission.to_csv('submission.csv', index=False)

