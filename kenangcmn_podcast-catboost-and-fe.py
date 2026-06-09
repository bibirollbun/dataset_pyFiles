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

train_df[cats] = train_df[cats].astype(str)
test_df[cats] = test_df[cats].astype(str)


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

print(f"Features: {len(features)} (Categorical: {len(cats)})")



import catboost as cb
from sklearn.metrics import mean_squared_error
import optuna
from sklearn.model_selection import KFold, train_test_split
"""
def objective(trial):
    params = {
        "iterations": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.2, log=True),
        "depth": trial.suggest_int("depth", 2, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.01, 10, log=True),
    }
    train_df2 = pd.get_dummies(train_df[:20000], columns=cats)
    y_train = train_df2["Listening_Time_minutes"]
    X_train = train_df2.drop(labels = "Listening_Time_minutes", axis = 1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)                          
    model = cb.CatBoostRegressor(**params, silent=True)
    model.fit(X_train, y_train, verbose = 0)
    predictions = model.predict(X_val)
    rmse = mean_squared_error(y_val, predictions, squared=False)
    return rmse
 """  


""" 
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, show_progress_bar=True)
"""


""" 
print('Best hyperparameters:', study.best_params)
print('Best RMSE:', study.best_value)
"""


#best_params = study.best_params


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

    cats.append(col_name)

    return test_df


%%time

import gc
gc.collect()

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds_cat = np.zeros(len(train_df))
test_preds_cat = np.zeros(len(test_df))

catboost_params = {"iterations": 3000,
                   "learning_rate": 0.02,
                   "depth": 6,
                   "loss_function": "RMSE",
                   "eval_metric": "RMSE",
                   "random_seed": 42,
                   "early_stopping_rounds": 200,
                   "verbose": 500
                    }

for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
    print(f"### Fold {fold+1} ###")

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

    cats = [c for c in X_train.columns if X_train[c].dtype in ["object", "category"]]
    
    model = cb.CatBoostRegressor(**catboost_params,
                                cat_features = cats)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

    oof_preds_cat[valid_idx] = model.predict(X_valid)
    test_preds_cat += model.predict(X_test) / FOLDS

rmse = np.sqrt(mean_squared_error(train_df[rmv], oof_preds_cat))
print(f"Validation RMSE: {rmse}")


sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
sub["Listening_Time_minutes"] = test_preds_cat
sub.to_csv("submission.csv", index=False)

