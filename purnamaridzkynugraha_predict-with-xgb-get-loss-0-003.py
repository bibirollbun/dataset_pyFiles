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


!pip install jcopml


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from jcopml.pipeline import num_pipe, cat_pipe
from jcopml.plot import plot_missing_value
from jcopml.feature_importance import mean_score_decrease
import glob


df = pd.read_csv('/kaggle/input/solana-skill-sprint-memcoin-graduation/train.csv',index_col='Unnamed: 0')
df_test = pd.read_csv('/kaggle/input/solana-skill-sprint-memcoin-graduation/test_unlabeled.csv',index_col='Unnamed: 0')
chunk_files = glob.glob('/kaggle/input/pump-fun-graduation-february-2025/chunk*.csv')
df_chunk = pd.concat([pd.read_csv(file) for file in chunk_files], ignore_index=True)


df.head()


df_test.head()


df_chunk.head()


plot_missing_value(df)


df = df.drop(columns='slot_graduated',axis =1)


df_chunk['mint'] = df_chunk['base_coin']

df_features = df_chunk.groupby('mint').agg({
    'slot': ['min', 'max', 'count'],
    'direction': lambda x: (x == 'buy').sum(),  
    'base_coin_amount': ['sum', 'mean'],
    'quote_coin_amount': ['sum', 'mean'],
    'virtual_token_balance_after': 'last',
    'virtual_sol_balance_after': 'last',
}).reset_index()

df_features.columns = ['mint'] + [f"{i}_{j}" if j != '' else i for i, j in df_features.columns[1:]]

df_train_merged = df.merge(df_features, on='mint', how='left')
df_test_merge = df_test.merge(df_features, on='mint', how='left')


df_train_merged.head()


df_test_merge.head()


X = df_train_merged.select_dtypes(include='number')
y = df_train_merged['has_graduated'].astype(int)  


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV


pipeline = make_pipeline( StandardScaler(), LassoCV(cv=5, max_iter=5000, random_state=42))
pipeline.fit(X, y)
coef = pipeline.named_steps['lassocv'].coef_
selected_features = X.columns[coef != 0]
selected_features


X = X[selected_features]
X.rename(columns={'direction_<lambda>': 'direction'}, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from jcopml.tuning import random_search_params as rsp
from sklearn.metrics import log_loss
xgb = XGBClassifier()

model = RandomizedSearchCV(
    xgb, rsp.xgb_poly_params,
    n_iter=30, cv=3,
    n_jobs=-1, verbose=1,
    error_score='raise'
)

model.fit(X_train, y_train)

y_pred_train = model.predict_proba(X_train)
y_pred_test = model.predict_proba(X_test)

print("Log Loss Train:", log_loss(y_train, y_pred_train))
print("Log Loss Test:", log_loss(y_test, y_pred_test))
print("Best CV Score (dari tuning):", model.best_score_)



df_test_merge = df_test_merge[selected_features]


df_test_merge.rename(columns={'direction_<lambda>': 'direction'}, inplace=True)


predict =  model.predict_proba(df_test_merge)[:, 1]



predictions_df = pd.DataFrame({
    'mint': df_test.mint,
    'has_graduated': predict
})

predictions_df.to_csv('/kaggle/working/predictions.csv', index=False)

