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


import warnings
warnings.filterwarnings("ignore")


import seaborn as sns


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


train.info()


l = train.columns.tolist()
for _ in l:
    print(train[_].value_counts())
    print("-"*50)


list(train.columns)


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
4


num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])

cat_pipeline = Pipeline([
    ('ordinal_encoder',OrdinalEncoder()),
    ('imputer',SimpleImputer(strategy = 'most_frequent')),
    ('cat_encoder',OneHotEncoder(sparse_output=False))
])


from sklearn.compose import ColumnTransformer



num_attribs = ['id',
 'annual_income',
 'debt_to_income_ratio',
 'credit_score',
 'loan_amount',
 'interest_rate']
cat_attribs = ['gender',
 'marital_status',
 'education_level',
 'employment_status',
 'loan_purpose',
 'grade_subgrade']

preprocessPipeline = ColumnTransformer([
    ('num',num_pipeline,num_attribs),
    ('cat',cat_pipeline,cat_attribs)
])


X_train = preprocessPipeline.fit_transform(train)
X_train


y_train = train["loan_paid_back"]


from sklearn.svm import SVC


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform


param_distrib = {
    "gamma": loguniform(0.001, 0.1),
    "C": uniform(1, 10)
}
rnd_search_cv = RandomizedSearchCV(SVC(random_state=2025), param_distrib, n_iter=100, cv=5,
                                   random_state=42)
rnd_search_cv.fit(X_train, y_train)
lin_clf = rnd_search_cv.best_estimator_


X_test = preprocessPipeline.transform(test)
y_pred = lin_clf.predict(X_test)


submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': y_pred.flatten() 
})
submission.to_csv('submission.csv', index=False)

