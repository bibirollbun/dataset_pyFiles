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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train_df.head()


test_df.head()


train_df.info()


train_df.describe()


for col in train_df.select_dtypes(include='float64').columns:
    train_df[col] = train_df[col].astype('float32')
for col in train_df.select_dtypes(include='int64').columns:
    train_df[col] = train_df[col].astype('int32')


print("Missing Values in train data:")
print(train_df.isnull().sum())

print("\nMissing Values in test data:")
print(test_df.isnull().sum())


train_df["Sex"] = train_df["Sex"].map({"male": 1, "female": 0}).astype("category")
test_df["Sex"] = test_df["Sex"].map({"male": 1, "female": 0}).astype("category")


rmv = ["Calories"]
features = [c for c in train_df.columns if c not in rmv]
cats = ["Sex"]


nums = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]


def add_feature_crosses(df, nums):
    df_new = df.copy()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            f1 = nums[i]
            f2 = nums[j]
            df_new[f"{f1}_x_{f2}"] = df_new[f1] * df_new[f2]
    return df_new

train_df = add_feature_crosses(train_df, nums)
test_df = add_feature_crosses(test_df, nums)


X = train_df.drop(columns=["id", "Calories"])
y = np.log1p(train_df["Calories"])
X_test = test_df.drop(columns=["id"])


import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from xgboost.callback import TrainingCallback
from colorama import Fore, Style, Back
import gc
import optuna
import logging
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import catboost as cb


class TQDMCallback(TrainingCallback):
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Training", leave=False)
    
    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False  # return True to stop training early

    def after_training(self, model):
        self.pbar.close()
        return model

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds_xgb = np.zeros(len(train_df))
test_preds_xgb = np.zeros(len(test_df))

xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 2000,
            'learning_rate': 0.02,
            'max_depth': 10,
            'gamma': 0.01,
            'subsample': 0.9,
            'colsample_bytree': 0.75,
            'max_delta_step': 2,
            'reg_alpha': 0.8,
            'reg_lambda': 4,
            'seed': 42,
            'enable_categorical': True,
            'tree_method': 'hist',  
            'device': 'gpu'
            }

for fold, (train_idx, valid_idx) in enumerate(kf.split(X,y)):
    print(f"### Fold {fold+1} ###")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    model = XGBRegressor(**xgb_params)

    callbacks = [TQDMCallback(total=xgb_params.get("n_estimators", 1000))]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=100,
        verbose=0,
        callbacks=callbacks
    )

    oof_preds_xgb[valid_idx] = model.predict(X_valid)
    test_preds_xgb += model.predict(X_test) / FOLDS

rmsle = np.sqrt(mean_squared_log_error(train_df[rmv], np.expm1(oof_preds_xgb)))
print(f"\nValidation RMSLE: {rmsle}")


FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds_cat = np.zeros(len(train_df))
test_preds_cat = np.zeros(len(test_df))

catboost_params = {"learning_rate": 0.02,
                   "iterations": 2000,
                   "depth" :10,
                   "l2_leaf_reg": 3,
                   "loss_function": "RMSE",
                   "eval_metric": "RMSE",
                   "random_seed": 42,
                   "early_stopping_rounds": 100,
                   "verbose": 500,
                   "cat_features": cats
                    }

for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
    print(f"### Fold {fold+1} ###")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    model = cb.CatBoostRegressor(**catboost_params)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

    oof_preds_cat[valid_idx] = model.predict(X_valid)
    test_preds_cat += model.predict(X_test) / FOLDS

rmsle = np.sqrt(mean_squared_log_error(train_df[rmv], np.expm1(oof_preds_cat) ))
print(f"Validation RMSE: {rmsle}")


final_preds = 0.3 * test_preds_cat + 0.7 * test_preds_xgb
final_preds = np.expm1(final_preds)
final_preds = np.clip(final_preds, 1, 314)
total_rmsle = np.sqrt(mean_squared_log_error(train_df[rmv],(0.3*np.expm1(oof_preds_cat)+0.7*np.expm1(oof_preds_xgb) )))
print(f"Total RMSLE: {total_rmsle}")


sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
sub["Calories"] = final_preds
sub.to_csv("submission.csv", index=False)


sub.head()

