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


import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from IPython.display import clear_output

import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from lightgbm import early_stopping  
from sklearn.model_selection import *
from sklearn.metrics import *

from sklearn.preprocessing import StandardScaler

train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'])
train = train.drop(columns=['id'])

train = train.drop_duplicates()
train = train.dropna(subset=['num_sold'])

test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'])
test_id = test['id']
test = test.drop(columns=['id'])


import requests

def get_gdp_per_capita(alpha3, year):
    url='https://api.worldbank.org/v2/country/{0}/indicator/NY.GDP.PCAP.CD?date={1}&format=json'
    response = requests.get(url.format(alpha3,year)).json()
    return response[1][0]['value']

df = train[['date', 'country']].copy()
alpha3s = ['CAN', 'FIN', 'ITA', 'KEN', 'NOR', 'SGP']
df['alpha3'] = df['country'].map(dict(zip(
    np.sort(df['country'].unique()), alpha3s)))
years = np.sort(df['date'].dt.year.unique())
df['year'] = df['date'].dt.year
gdp = np.array([
    [get_gdp_per_capita(alpha3, year) for year in years]
    for alpha3 in alpha3s
])
gdp = pd.DataFrame(gdp/gdp.sum(axis=0), index=alpha3s, columns=years)
df['GDP'] = df.apply(lambda s: gdp.loc[s['alpha3'], s['year']], axis=1)


train = train.merge(df[['country', 'date', 'GDP']], on=['country', 'date'], how='left')
test = test.merge(df[['country', 'date', 'GDP']], on=['country', 'date'], how='left')


def eda(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['month_name'] = df['date'].dt.month_name()
    df['day_of_week'] = df['date'].dt.day_name()
    df['week'] = df['date'].dt.isocalendar().week
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12) 
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)  
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['group'] = (df['year'] - 2020) * 48 + df['month'] * 4 + df['day'] // 7
    df.drop('date', axis=1, inplace=True)
    df['cos_year'] = np.cos(df['year'] * (2 * np.pi) / 100)
    df['sin_year'] = np.sin(df['year'] * (2 * np.pi) / 100)
    # why using sin/cos? to tell model that after Dec we have Jan, of we dont do this it will
    # consider 1 to 12 and then 12 to 1 wont be considered. same applies on week day also
    # this is universal funnction whenever we have date.
    dummy_prefixes = ['country', 'store', 'product','month_name','day_of_week']
    df = pd.get_dummies(df, columns=dummy_prefixes, drop_first=True)

    return df

train = eda(train)
test = eda(test)


X = train.drop(['num_sold'], axis=1)
y = train['num_sold']
X_test = test


import lightgbm as lgb
import numpy as np
from sklearn.model_selection import RepeatedKFold
from catboost import CatBoostRegressor
from tqdm import tqdm
from IPython.display import clear_output
import shap

# Define RMSLE function
def rmsle(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.maximum(np.array(y_pred), 0)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))


params2 = {'n_estimators': 1327, 'max_depth': 7, 'colsample_bytree': 0.6932974324289563, 
          'subsample': 0.10409176504249848, 'learning_rate': 0.03556720266195535, 'min_child_samples': 78}

params1 = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.08,
    'max_depth': 13,
    'reg_alpha': 0.01,
    'lambda_l2': 0.01,  
    'min_child_samples' : 32,
    'colsample_bytree': 0.93,
    'subsample': 0.7, 
    'seed': 42,
    'verbose': -1,
    'device' : 'cpu' 
}
def training(params, modelName, X, y, X_test):
    kfold = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
    fold_test_preds = []  # List to store test predictions from each fold
    rmsle_values = []  # List to store RMSLE for each fold
    modelM = None
    for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(X, y), desc="Training Folds", total=10)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Apply log transformation to target variables
        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)
        
        # Initialize model based on modelName
        if modelName == "lgbm":
            model = lgb.LGBMRegressor(**params)
        elif modelName == "catboost":
            model = CatBoostRegressor(**params, random_state=42)  # Silent=True for less verbose
        
    
        model.fit(X_train, y_train_log,
                      eval_set=[(X_val, y_val_log)],
                      callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        modelM = model
        
        y_val_pred_log = model.predict(X_val)
        y_val_pred = np.expm1(y_val_pred_log)
        
        #RMSLE
        rmsle_valid = rmsle(y_val, y_val_pred)
        rmsle_values.append(rmsle_valid)
        
        # Predict on the test set
        test_log_pred = model.predict(X_test)
        test_pred = np.expm1(test_log_pred)
        
    
        fold_test_preds.append(test_pred)
        clear_output(wait=True)
        

    return fold_test_preds, np.mean(rmsle_values)

ans = training(params1,'lgbm', X, y, X_test)


final_test_pred = np.mean(ans[0], axis = 0)
rmsle_values = ans[1]
print(f'LGBM {rmsle_values}')


preds = final_test_pred


predictions = pd.DataFrame({
    'id': test_id,
    'num_sold': preds
})
predictions.to_csv('ans.csv', index=False)

