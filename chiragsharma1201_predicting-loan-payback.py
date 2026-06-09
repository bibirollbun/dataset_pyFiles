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


d=pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")


d.head()


d.info()


d.dtypes


d.index


d.describe()


d.isnull().sum()


d.shape


d1=pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


d1.head()


d1.info()


d1.describe()


d1.shape


d1.dtypes


d1.isnull().sum()


test_id = d1['id'].copy()


d=d.drop('id', axis=1)
d1=d1.drop('id', axis=1)


numerical_cols = [
    'annual_income',
    'debt_to_income_ratio',
    'credit_score',
    'loan_amount',
    'interest_rate'
]
categorical_cols = [
    'gender',
    'marital_status',
    'education_level',
    'employment_status',
    'loan_purpose',
    'grade_subgrade'
]


for col in numerical_cols:
    if d[col].isnull().any():
        median_val = d[col].median()
        d[col] = d[col].fillna(median_val)      
        d1[col]  = d1[col].fillna(median_val)


for col in categorical_cols:
    if d[col].isnull().any():
        mode_val = d[col].mode()[0]
        d[col] = d[col].fillna(mode_val)        
        d1[col]  = d1[col].fillna(mode_val)


from sklearn.preprocessing import LabelEncoder


label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([d[col], d1[col]], axis=0).astype(str)
    le.fit(combined)
    d[col]  = le.transform(d[col].astype(str))
    d1[col] = le.transform(d1[col].astype(str))
    label_encoders[col] = le


X = d.drop('loan_paid_back', axis=1)
y = d['loan_paid_back']


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
)


X_train.shape


X_test.shape


import lightgbm as lgb


l= lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=8,
    num_leaves=64,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=30,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)


l.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)


l.best_iteration_


from sklearn.metrics import roc_auc_score


l_pred = l.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, l_pred)
print(f'Validation AUC: {auc_score:.5f}')



test_pred = l.predict_proba(d1)[:, 1]


test_pred


submission = pd.DataFrame({
    'id': test_id,
    'loan_paid_back': test_pred
})


submission.to_csv('submission.csv', index=False)


submission.head()


submission.hist()


submission.shape


submission.corr()

