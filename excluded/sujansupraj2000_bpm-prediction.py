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


#importing libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#importing data
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv', index_col ='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv', index_col='id')
original = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')


print(train.shape)
print(original.shape)


original.head()


#combining both
df = pd.concat([train,original])
print(df.shape)


df['AudioLoudness'] = df['AudioLoudness'] * -1


#check for nulls
df.isnull().sum()


#check for duplicates
df.duplicated().sum()


df.info()


df.describe()


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
plt.show()


#XGB Regressor
"""from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

y = df['BeatsPerMinute']
X = df.drop(columns=['BeatsPerMinute'])

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold = 1

for train_idx, val_idx in kf.split(X,y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]


    best_xgb_params={
                 'n_estimators': 630,
                 'learning_rate': 0.01,
                 'colsample_bytree': 1.0,
                 'reg_alpha': 0.020682970136481817,
                 'reg_lambda': 68.19931550257023,
                 'tree_method':'hist',
                 'random_state': 42
                } 

    xgb = XGBRegressor(objective = 'reg:squarederror', **best_xgb_params)
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    print(f"Fold {fold} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    fold += 1
"""
    


#LGB Regressor
#import optuna
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
"""def objective(trial):
    params = {
        'n_estimators' : trial.suggest_int('n_estimators', 50, 100),
        'max_depth' : trial.suggest_int('max_depth', 3, 12),
        'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves' : trial.suggest_int('num_leaves', 1, 250),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.05, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0001, 0.1),
        'reg_lambda' : trial.suggest_float('reg_lambda', 1, 300),
        'max_bin' : trial.suggest_int('max_bin', 20, 255),
        'random_state': 42,
        'n_jobs': -1
    } """

y = df['BeatsPerMinute']
X = df.drop(columns=['BeatsPerMinute'])
best_params = {'n_estimators': 85, 
                          'max_depth': 9,
                          'learning_rate': 0.05825572456474483,
                          'num_leaves': 7, 
                          'colsample_bytree': 0.9460569126939476, 
                          'reg_alpha': 0.049781967381528035,
                          'reg_lambda': 112.01878062355247,
                          'max_bin': 138}

    #rmse_scores=[]
fold =1
n_splits=5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(X,y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    
    lgb = LGBMRegressor(**best_params)
    lgb.fit(X_train, y_train)
    
    y_pred = lgb.predict(X_val)
    
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    print(f"Fold {fold} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    #rmse_scores.append(rmse)
    fold += 1
    #return np.mean(rmse_scores)
#study_lgb = optuna.create_study(direction='minimize')
#study_lgb.optimize(objective, n_trials = 15)
#print("best lgbm parameters",study_lgb.best_trial.params)
    


test['AudioLoudness'] = test['AudioLoudness'] * -1
BeatsPerMinute = lgb.predict(test)
prediction_df = pd.DataFrame({
        'id': test.index, 
        'BeatsPerMinute': BeatsPerMinute})
prediction_df.to_csv('Submissions.csv', index=False)
print('Submission File created')

