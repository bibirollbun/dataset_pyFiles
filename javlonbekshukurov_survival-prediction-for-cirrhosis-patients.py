# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline 
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import log_loss
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train=pd.read_csv('/kaggle/input/multiclassificationtask/train.csv')
train.head()


test=pd.read_csv('/kaggle/input/multiclassificationtask/test.csv')
test.head()


train.shape


train.isnull().sum()


train.info()


train.Drug.value_counts()


train.Ascites .value_counts()


train.Hepatomegaly.value_counts()


train.Spiders .value_counts()


train.drop_duplicates(inplace=True)


train['age_in_years']=(train['Age']/365).astype(int)
train.drop('Age', axis=1, inplace=True)
train.head()


train.head()


train=train[train['Status'] != 'Y']
X=train.drop(columns=['id','Status'])
y=train['Status']


test['age_in_years']=(test['Age']/365).astype(int)
test.drop(['Age','id'], axis=1, inplace=True)
test.head()


test.isnull().sum()


most_frequent_cols=['Ascites','Spiders','Sex','Edema']
ffill_cols = ['Drug', 'Hepatomegaly']
num_cols = X.select_dtypes(include=['number']).columns.tolist()


class ForwardFillImputer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(X).ffill().values


most_freq_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
ffill_pipeline = Pipeline([
    ('ffill', ForwardFillImputer()),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


full_pipeline = ColumnTransformer([
    ('most_freq_cat', most_freq_pipeline, most_frequent_cols),
    ('ffill_cat', ffill_pipeline, ffill_cols),
    ('num', num_pipeline, num_cols)
])


X_transformed=full_pipeline.fit_transform(X)
X_transformed[0]


X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)


LR_model=LogisticRegression(multi_class='multinomial', solver='lbfgs')
LR_model.fit(X_train, y_train)


# Model evaluation
y_pred_proba = LR_model.predict_proba(X_test)
eps = 1e-15
y_pred_clipped = np.clip(y_pred_proba, eps, 1 - eps)
loss = log_loss(y_test, y_pred_clipped)
print("Log Loss: " ,loss)


RF_model=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
RF_model.fit(X_train, y_train)


y_pred_proba = RF_model.predict_proba(X_test)
eps = 1e-15
y_pred_clipped = np.clip(y_pred_proba, eps, 1 - eps)
loss = log_loss(y_test, y_pred_clipped)
print("Log Loss: " ,loss)


GB_model=GradientBoostingClassifier(n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42)
GB_model.fit(X_train, y_train)
y_preda_proba = GB_model.predict_proba(X_test)
eps = 1e-15
y_pred_clipped = np.clip(y_preda_proba, eps, 1 - eps)


loss = log_loss(y_test, y_pred_clipped)
print("Log Loss: " ,loss)


le=LabelEncoder()
y_train_encoded=le.fit_transform(y_train)
y_test_encoded=le.fit_transform(y_test)


XGB_model = xgb.XGBClassifier(use_label_encoder=False,learning_rate=0.11,max_depth=5,n_estimators=180,eval_metric='logloss',random_state=42,subsample=0.7,colsample_bytree=0.2)
XGB_model.fit(X_train, y_train_encoded)
y_pred=XGB_model.predict_proba(X_test)
eps = 1e-15
y_pred_clipped = np.clip(y_pred, eps, 1 - eps)


loss = log_loss(y_test_encoded, y_pred_clipped)
print("Log Loss: " ,loss)


test_transformed=full_pipeline.transform(test)


test_predictions = XGB_model.predict_proba(test_transformed)

if 'id' in test.columns:
    submission = pd.DataFrame({'id': test['id']})
else:
    submission = pd.DataFrame({'id': range(15000, 15000+len(test))})

submission[['Status_C', 'Status_CL', 'Status_D']] = test_predictions

submission.to_csv('submission.csv', index=False)
print(submission)

