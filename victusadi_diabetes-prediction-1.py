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
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt


df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")

df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


print("Training Columns:\n",df_train.columns)
print("Test Columns:\n",df_test.columns)


print(df_train.shape)
print(df_test.shape)


print(df_train.describe())
print(df_test.head())


y = df_train.diagnosed_diabetes


features = ['age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides', 'gender', 'ethnicity','smoking_status',
       'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history']

X = df_train[features]

X_test = df_test[features]

num_feat = ['age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides']

cat_feat = ['gender', 'ethnicity','smoking_status']

num_pipe = Pipeline(steps=[
    ('impute',SimpleImputer(strategy='median')),
    ('scale',StandardScaler())
])

cat_pipe = Pipeline(steps=[
    ('imp',SimpleImputer(strategy='most_frequent')),
    ('hot',OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer([
    ('num',num_pipe,num_feat),
    ('cat',cat_pipe,cat_feat)
])

pipe = Pipeline(steps=[
    ('pre',preprocessor),
    ('model',XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    ))
])


pipe.fit(X,y)

preds = pipe.predict_proba(X_test)[:,1]

print("Prediction made successfully")


submission = pd.DataFrame({
    'id': df_test['id'],
    'diagnosed_diabetes': preds
})

submission.to_csv('submission.csv',index=False)
print("XGBoost Submission saved to submission.csv")

