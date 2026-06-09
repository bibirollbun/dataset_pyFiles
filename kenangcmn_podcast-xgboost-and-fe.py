# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings("ignore")
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train_df.head()


train_df.info()


train_df.describe()


train_df.columns


for col in train_df.select_dtypes(include='float64').columns:
    train_df[col] = train_df[col].astype('float32')
for col in train_df.select_dtypes(include='int64').columns:
    train_df[col] = train_df[col].astype('int32')


rmv = ["Listening_Time_minutes"]
features = [c for c in train_df.columns if c not in rmv]
cats = [c for c in features if train_df[c].dtype == "object"]

print(f"Features: {len(features)} (Categorical: {len(cats)})")


train_df = train_df.copy()
test_df = test_df.copy()

print("Missing Values in train data:")
print(train_df.isnull().sum())

print("\nMissing Values in test data:")
print(test_df.isnull().sum())


train_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].median(), inplace=True)

train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median(), inplace=True)
test_df['Episode_Length_minutes'].fillna(test_df['Episode_Length_minutes'].median(), inplace=True)

train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median(), inplace=True)
test_df['Guest_Popularity_percentage'].fillna(test_df['Guest_Popularity_percentage'].median(), inplace=True)


rmv = ["Listening_Time_minutes"]
features = [c for c in train_df.columns if c not in rmv]
cats = [c for c in features if train_df[c].dtype == "object"]

print(f"Features: {len(features)} (Categorical: {len(cats)})")


print(cats)


%%time
from itertools import combinations
from tqdm import tqdm
encoded_columns = []
encode_columns = ['Episode_Length_minutes', 'Genre', 'Episode_Title', 'Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'Publication_Day', 'Publication_Time']
pair_size = [2,3,4,5]

for r in pair_size:
    for cols in tqdm(list(combinations(encode_columns, r))):
        new_col_name = '_'.join(cols)
        
        train_df[new_col_name] = train_df[list(cols)].astype(str).agg('_'.join, axis=1)
        train_df[new_col_name] = train_df[new_col_name].astype('category')
        
        test_df[new_col_name] = test_df[list(cols)].astype(str).agg('_'.join, axis=1)
        test_df[new_col_name] = test_df[new_col_name].astype('category')

        encoded_columns.append(new_col_name)


rmv = ["Listening_Time_minutes"]
features = [c for c in train_df.columns if c not in rmv]
cats = [c for c in features if train_df[c].dtype == "category"]



for col in features:
    if train_df[col].dtype == "object":
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')

rmv = ["Listening_Time_minutes"]
features = [c for c in train_df.columns if c not in rmv]
cats = [c for c in features if train_df[c].dtype == "category"]

print(f"Features: {len(features)} (Categorical: {len(cats)})")


import xgboost as xgb
from xgboost import XGBRegressor, callback

from category_encoders import TargetEncoder

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


import optuna
import logging
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score

""" 
def optimize_xgboost(train_df,features,n_trials = 30):
    def objective(trial):
        params = {"n_estimators": trial.suggest_int("n_estimators", 1000, 3000),
                  "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.4, log=True),
                  "max_depth": trial.suggest_int("max_depth", 3, 12),}
                
     # 5-fold cross-validation
        FOLDS = 5
        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
        rmse_scores = []

        for train_idx, valid_idx in kf.split(train_df.iloc[:20000]):
            X_train, X_valid = train_df.iloc[train_idx][features], train_df.iloc[valid_idx][features]
            y_train, y_valid = train_df.iloc[train_idx][rmv], train_df.iloc[valid_idx][rmv]
        
            model = XGBRegressor(**params, 
                                 objective="reg:squarederror",
                                 eval_metric="rmse",
                                 early_stopping_rounds=100,
                                 random_state=42,
                                 enable_categorical=True, 
                                 verbosity=2)
        
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=200)
        
            preds = model.predict(X_valid)
            rmse = np.sqrt(mean_squared_error(y_valid, preds))
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(direction="minimize") 
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

best_params = optimize_xgboost(train_df, features, n_trials=30)
print("Best hyperparameters:", best_params)
"""


def target_encode(train_df, test_df, col, target, stats='mean', prefix='TE'):
    col_name = f"{prefix}_{col}"
    
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

    agg = train_df.groupby(col)[target].agg(stats)

    if isinstance(agg, pd.DataFrame):
        agg = agg.iloc[:, 0]

    test_df[col_name] = test_df[col].map(agg)

    test_df[col_name].fillna(agg.mean(), inplace=True)

    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')

    return test_df


%%time

from colorama import Fore, Style, Back

import gc
gc.collect()

class TQDMCallback(callback.TrainingCallback):
    def __init__(self, total, fold_num=None):
        self.pbar = tqdm(total=total, desc=f"Training Fold {fold_num}", leave=True)

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        if epoch + 1 == self.pbar.total:
            self.pbar.close()
        return False

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds_xgb = np.zeros(len(train_df))
test_preds_xgb = np.zeros(len(test_df))

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 3000,
    'learning_rate': 0.01,
    'max_depth': 10,
    'subsample': 1.0,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.8,
    'reg_lambda': 4,
    'seed': 42,
    'enable_categorical': True,
    'tree_method': 'hist',  
    'device': 'gpu'
}

for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
    print(Fore.GREEN +f"### Fold {fold+1} ###"+Style.RESET_ALL)

    X_train = train_df.loc[train_idx, features + rmv].reset_index(drop=True)
    y_train = X_train[rmv]
    X_valid, y_valid  = train_df.loc[valid_idx, features].reset_index(drop=True), train_df.loc[valid_idx, rmv].reset_index(drop=True)
    X_test = test_df[features].reset_index(drop=True)
    

    kf2 = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

    for fold2, (train_idx2, valid_idx2) in enumerate(kf2.split(X_train)):
        train2 = X_train.iloc[train_idx2].copy()
        valid2 = X_train.iloc[valid_idx2][features].copy()

        
        for col in tqdm(encoded_columns, total=len(encoded_columns), desc=f"Second KFold's {fold2+1} / {FOLDS} columns"):
            te_col = f'TE_{col}'
            valid2 = target_encode(train2, valid2, col, rmv, stats='mean', prefix="TE")
            X_train.loc[valid_idx2, te_col] = valid2[te_col].values

        del train2, valid2

    gc.collect()

    for col in encoded_columns:
        X_valid = target_encode(X_train, X_valid, col, rmv, stats='mean', prefix="TE")
        X_test = target_encode(X_train, X_test, col, rmv, stats='mean', prefix="TE")

    te_cols = [f'TE_{col}' for col in encoded_columns]
    X_train.drop(rmv + encoded_columns, axis=1, inplace=True)
    X_valid.drop(encoded_columns, axis=1, inplace=True)
    X_test.drop(encoded_columns, axis=1, inplace=True)

    model = XGBRegressor(**xgb_params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=150,
        verbose=0,
        callbacks=[TQDMCallback(total=xgb_params['n_estimators'], fold_num=fold+1)])

    oof_preds_xgb[valid_idx] = model.predict(X_valid)
    test_preds_xgb += model.predict(X_test) / FOLDS

    gc.collect()

rmse = np.sqrt(mean_squared_error(train_df[rmv], oof_preds_xgb))
print(Fore.GREEN + f"Validation RMSE: {rmse}"+ Style.RESET_ALL)


sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
sub["Listening_Time_minutes"] = test_preds_xgb
sub.to_csv("submission.csv", index=False)

