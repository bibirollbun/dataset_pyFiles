# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.base import clone
import re

import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import *
from sklearn.preprocessing import *

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import log_evaluation, early_stopping

from sklearn.metrics import *

pd.set_option('display.max_columns', None)
from IPython.display import clear_output
from tqdm import tqdm, trange
from tabulate import tabulate
import random
import time
import logging
from IPython.display import display
from IPython.display import display, HTML
from colorama import Fore
from datetime import datetime
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s4e9/sample_submission.csv')


test_idx = test['id']


train.drop('id',inplace=True,axis=1)
test.drop('id',inplace=True,axis=1)


train.head()


train .info()


train.isnull().sum()


target = 'price'


all_features = [i for i in train.columns if i not in ['id',target]]


cat_features = [i for i in all_features if train[i].dtype in ['object']]



num_features = train[all_features].select_dtypes(include=['int64','float64']).columns.tolist()


num_features


def update(df):
    
    t = 100
    
    df['accident'] = df['accident'].map({
        'None reported': 'not_reported',
        'At least 1 accident or damage reported': 'reported'
    })
    df['transmission'] = df['transmission'].str.replace('/', '').str.replace('-', '')
    df['transmission'] = df['transmission'].str.replace(' ', '_')
    
    cat_c = ['brand','model','fuel_type','engine','transmission','ext_col','int_col','accident','clean_title']
    re_ = ['model','engine','transmission','ext_col','int_col']
    
    for col in re_:
        df.loc[df[col].value_counts(dropna=False)[df[col]].values < t, col] = "noise"
        
    for col in cat_c:
        df[col] = df[col].fillna('missing')
        df[col] = df[col].astype('category')
        
    return df


train  = update(train)
test   = update(test)


def feature(df):
    current_year = datetime.now().year

    df['age_year'] = current_year - df['model_year']
    df['milage_per_year'] = df['milage']/df['age_year']

    def extract_horsepower(engine):
        try:
            return float(engine.split('HP')[0])
        except:
            return None

    def extract_engine_size(engine):
        try:
            return float(engine.split(' ')[1].replace('L', ''))
        except:
            return None

    df['horsepower'] = df['engine'].apply(extract_horsepower)
    df['engine_size'] = df['engine'].apply(extract_engine_size)
    df['power_to_weight_ratio'] = df['horsepower']/df['engine_size']

    luxury_brands =  ['Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Land', 
                    'Lexus', 'Jaguar', 'Bentley', 'Maserati', 'Lamborghini', 
                    'Rolls-Royce', 'Ferrari', 'McLaren', 'Aston', 'Maybach']
    df['Is_Luxury_Brand'] = df['brand'].apply(lambda x: 1 if x in luxury_brands else 0)

    df['Accident_Impact'] = df.apply(lambda x: 1 if x['accident'] == 1 and x['clean_title'] == 0 else 0, axis=1)
    
    return df



train = feature(train)
test = feature(test)


%%time

X = train.drop(['price'], axis=1)
y = train['price']
callbacks = [early_stopping(stopping_rounds=100)]

SEED = 601
n_splits = 5

def Train_ML(model, model_name, test):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(test.shape[0])
    val_rmse_list = []
    train_rmse_list = []

    for fold_idx, (train_index, val_index) in tqdm(enumerate(kf.split(X)), desc=f"Model: {model_name}", total=n_splits):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model_clone = clone(model)
        model_clone.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)

        val_preds = model_clone.predict(X_val)
        oof_preds[val_index] = val_preds
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        val_rmse_list.append(val_rmse)

        train_preds = model_clone.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        train_rmse_list.append(train_rmse)

        test_preds += model_clone.predict(test)
        clear_output(wait=True)

    mean_test_preds = test_preds / n_splits
    mean_val_rmse = np.mean(val_rmse_list)
    mean_train_rmse = np.mean(train_rmse_list)

    results = {
        'model_name': model_name,
        'mean_train_rmse': mean_train_rmse,
        'mean_val_rmse': mean_val_rmse,
        'per_fold_train_rmse': train_rmse_list,
        'per_fold_val_rmse': val_rmse_list
    }

    print(f"Model: {model_name}")
    print(f"Mean Train RMSE: {mean_train_rmse:.5f}")
    print(f"Mean Validation RMSE: {mean_val_rmse:.5f}\n")

    return oof_preds, mean_test_preds



Light2 = {'objective': 'regression','metric': 'rmse','num_boost_round':10_000,'learning_rate': 0.023395755673174177, 'max_depth': 4, 'num_leaves': 159, 'min_child_weight': 6.64512679143092, 'min_split_gain': 1.6984507610468915e-07, 'subsample': 0.5598176343183838, 'colsample_bytree': 0.510945164298283, 'lambda_l1': 0.1368118399550561, 'lambda_l2': 4.590879971301159}

Light1 = LGBMRegressor(**Light2, random_state=SEED, verbose=-1)
of_p2, mpL1 = Train_ML(Light1,'LGB_Tunned_2',test)




submission = pd.DataFrame({
    "id": test_idx,
    "price": mpL1  
})

submission.to_csv("submission.csv", index=False)



submission.head()


# X = train.drop(['priece'],axis=1)
# y = train['priece']
# seed = 41
# n_splits = 5



# def train_ml(model, model_name, test):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     oof_preds = np,zeros(X.shape[0])
#     test_preds = np.zeros(test.shape[0])
#     val_rmse_list = []
#     train_rmse_list = []

#     for fold_idx, (train_index, val_index) in tqdm(enumerate(kf.split(X)), desc=f"Model: {model_name}", total=n_splits):
#         X_train,x_val = X.iloc[train_index], X.iloc[val_index]
#         y_train,y_val = y.iloc[train_index] , y.iloc[val_index]

#         model_clone = clone(model)
#         model_clone.fit(X_train,y_train, eval_set =[(X_val,y_val)],early_stopping_rounds=100,
#             verbose=False)


#         val_preds = model_clone.predict(X_val)
#         oof_preds[val_index] = val_preds
#         val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
#         val_rmse_list.append(val_rmse)

#         train_preds = model_clone.predict(X_train)
#         train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
#         train_rmse_list.append(train_rmse)

#         test_preds += model_clone.predict(test)
    
#     mean_test_preds = test_preds / n_splits
#     mean_val_rmse = np.mean(val_rmse_list)
#     mean_train_rmse = np.mean(train_rmse_list)

#     print(f"Model: {model_name}")
#     print(f"Mean Train RMSE: {mean_train_rmse:.5f}")
#     print(f"Mean Validation RMSE: {mean_val_rmse:.5f}\n")
    
#     return oof_preds, mean_test_preds



# def objective(trial):
#     params = {
#         'n_estimators': 10000,
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#         'max_depth': trial.suggest_int('max_depth', 3, 8),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
#         'random_state': SEED
#     }

#     model = XGBRegressor(**params)
#     kf = KFold(n_splits = 3, shuffle = True, random_state = seed)
#     rmse_scores = []


#     for train_idx, val_idx in kf.split(X):
#         X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
#         y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

#         model.fit(
#             X_train_fold, y_train_fold,
#             eval_set=[(X_val_fold, y_val_fold)],
#             early_stopping_rounds=100,
#             verbose=False
#         )
#         val_preds = model.predict(X_val_fold)
#         rmse = np.sqrt(mean_squared_error(y_val_fold, val_preds))
#         rmse_scores.append(rmse)
#     return np.mean(rmse_scores)



# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials =N_TRIALS)
# best_params = study.best_params

# xgb_model = XGBRegressor(**best_params)
# oof_preds, test_preds = Train_ML(xgb_model, 'XGBost_Optuna', test)
# print("Training completed!")


# SEED = 601
# n_splits = 5
# N_TRIALS = 50


# def train_ml(model, model_name,test):
#     kf =KFold(n_splits = n_splits, shuffle = shuffle, random_state=SEED)
#     oof_preds = np.zeros(X.shape[0])
#     test_preds = np.zeros(test.shape[0])

#     val_rmse_list = []
#     train_rmse_list = []

#     for fold_idx, (train_index, val_index) in tqdm(enumerate(kf.split(X)), desc=f"Model: {model_name}", total=n_splits):
#         X_train , X_pred = X.iloc[train_index], X.iloc[val_index]
#         y_train, y_pred = y.iloc[train_index], y.iloc[val_index]

#         model_clone =colone(model)
#         model_clone.fit(X_train, y_train,eval_set=[(X_val, y_val)], 
#                 early_stopping_rounds=100,
#                 verbose=False)

#         train_preds = model_clone.predict(X_val)
#         oof_preds[val_index] = val_preds
#         val_rmse = np.sqrt(mean_squeared_error(_val, val_preds))
#         val_rmse_list.append(val_rmse)

#     mean_test_preds = test_preds/n_splits
#     mean)val_rmse = np.mean(val_rmse_list)
#     mean_train)rmse=np,mean(train_rmse_list)

#     print(f"Model: {model_name}")
#     print(f"Mean Train RMSE: {mean_train_rmse:.5f}")
#     print(f"Mean Validation RMSE: {mean_val_rmse:.5f}\n")
    
#     return oof_preds, mean_test_preds



# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 2000, 15000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
#         'random_state': SEED,
#     }
#     model = XGBRegressor(**params)
#     kf = KFold(n_splits=3,shuffle=True, random_state=SEED)
#     rmse_scores = []

#     for train_idx, val_idx in kf.split(X):
#         X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
#         y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
#         model.fit(
#             X_train_fold, y_train_fold,
#             eval_set=[(X_val_fold, y_val_fold)],
#             early_stopping_rounds=100,
#             verbose=False
#         )
#         val_preds = model.predict(X_val_fold)
#         rmse = np.sqrt(mean_squared_error(y_val_fold, val_preds))
#         rmse_scores.append(rmse)
#     return np.mean(rmse_scores)



# study = optuna.create_study(direction='minimalize')
# study.opimize(objectie, n_trials = N_TRIALS)
# best_params = study.best_params


# xgb_model =XGBRegressor(**best_params)
# oof_preds, test_preds = trial_ml(xgb_model, 'xgbboost optuna', test)
# print('succesfull')


# submission = pd.DataFrame({
#     "id": test["id"],
#     "price": test_preds
# })

# submission.to_csv("submission.csv", index=False)


