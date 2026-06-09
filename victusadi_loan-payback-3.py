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
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')

test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

sample = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


train.columns


y = train.loan_paid_back

features = [ 'annual_income','debt_to_income_ratio','credit_score', 'loan_amount', 
            'interest_rate', 'employment_status','loan_purpose']

X = train[features]

X_test = test[features]


numeric_feats = ['annual_income','debt_to_income_ratio','credit_score', 'loan_amount', 'interest_rate']

categ_feats = ['employment_status','loan_purpose']


num_pipe = Pipeline(steps=[
    ('impute',SimpleImputer(strategy = 'median')),
    ('scale', StandardScaler())
])

categ_pipe = Pipeline(steps=[
    ('imp',SimpleImputer(strategy = 'most_frequent')),
    ('one',OneHotEncoder(handle_unknown='ignore',sparse=False))
])

preprocessor = ColumnTransformer([
    ('num',num_pipe,numeric_feats),
    ('cat',categ_pipe,categ_feats)
])


pipe = Pipeline(steps=[
    ('pre',preprocessor),
    ('model',LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    ))
])

pipe.fit(X,y)

preds = pipe.predict_proba(X_test)[:,1]

print("Prediction made successfully!")


scores = cross_val_score(pipe,X,y, cv=5, scoring = 'roc_auc')
print(scores.mean())


submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back' : preds
})

submission.to_csv('submission.csv',index=False)

print("Submission made successfully!!")

