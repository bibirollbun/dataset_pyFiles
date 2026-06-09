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


from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import cupy as cp

import math
import logging
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger().setLevel(logging.ERROR)
%matplotlib inline


df_sample = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col='id')
df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col='id')


df_train.head()


X = df_train.drop(['y'], axis=1)
y = df_train['y']


num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']


num_pipeline = Pipeline(steps=[
    ('scaler', RobustScaler())
])

cat_pipeline = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine all preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num_pre', num_pipeline, num_cols),
    ('cat_pre', cat_pipeline, cat_cols)
])


xgb_clf = XGBClassifier(tree_method='hist', device="cuda", objective="binary:logistic", use_label_encoder=False, eval_metric='logloss', random_state=42)


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_clf)
])


# Define hyperparameter grid for tuning
param_grid = {
    'classifier__n_estimators': [200, 500, 1000],
    'classifier__max_depth': [3, 5, 7],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__colsample_bytree': [0.6, 0.8, 1.0],
    'classifier__learning_rate': [0.05, 0.1, 0.2]
}


skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Grid Search with Cross-Validation
grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring='roc_auc', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)


best_pipeline = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)


# Evaluation
y_pred = best_pipeline.predict(X_val)
print("Accuracy:", roc_auc_score(y_val, y_pred))
print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))


test_predictions = best_pipeline.predict(df_test)


submission = pd.DataFrame({'y': test_predictions}, index=df_test.index)
submission.to_csv('submission.csv')
submission.head()


# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', new_xg)
# ])


# pipeline.fit(X_train, y_train)


# a = pipeline.predict(X_val)
# print("Accuracy:", accuracy_score(y_val, a))
# print("Accuracy:", roc_auc_score(y_val, a))
# print("\nClassification Report:\n", classification_report(y_val, a))




