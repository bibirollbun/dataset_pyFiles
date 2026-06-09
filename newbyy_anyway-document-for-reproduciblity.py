%%capture
!pip install duckdb


import sys
import os
import gc

import warnings
warnings.filterwarnings('ignore')

import random
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import duckdb

import lightgbm as lgb
import xgboost as xgb
import catboost as cat

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import ks_2samp

from plotly import * 


import duckdb
duckdb.query('PRAGMA disable_progress_bar;')


RANDOM_STATE = 1966
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
seed_everything(RANDOM_STATE)

class Shhh:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        
%matplotlib inline
%config InlineBackend.figure_format='retina'

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

BASEPATH = '/kaggle/input/optimizingdefaultmodelbyfirstpaymentdefault/'


df = duckdb.sql(f"""
    CREATE OR REPLACE TEMPORARY TABLE eng_table AS
    SELECT
        *
    FROM read_csv_auto('{BASEPATH}kaggle_dataset.csv');

    SELECT * FROM eng_table;
""").df()

df.head(10)


df.info()


for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

X = df.drop(['target', 'ID'], axis=1)
y = df['target']


param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_leaf_nodes': [None, 10, 20, 30]
}

dtree = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dtree, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X, y)

print("best hyperparameter:", grid_search.best_params_)

fit_model = DecisionTreeClassifier(
    max_depth=None,
    min_samples_split=2,
    random_state=42
)
fit_model.fit(X, y)


y_proba = fit_model.predict_proba(X)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)




roc_auc = roc_auc_score(y, y_proba)
accuracy = accuracy_score(y, y_pred)
ks_stat, _ = ks_2samp(y_proba[y == 1], y_proba[y == 0])

print("ROC AUC Score:", roc_auc)
print("Accuracy:", accuracy)
print("Kolmogorov-Smirnov (KS) Score:", ks_stat)



df_output = df[['ID']].copy()
df_output['target_predicted'] = y_pred
df_output.to_csv('output_newbyy.csv', index=False)

