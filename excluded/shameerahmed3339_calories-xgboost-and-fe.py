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


train_df["Sex"] = train_df["Sex"].astype("category")
test_df["Sex"] = test_df["Sex"].astype("category")


train_df["BMI"] = train_df["Weight"] / np.square(train_df["Height"]/100).astype("float32")
test_df["BMI"] = test_df["Weight"] / np.square(test_df["Height"]/100).astype("float32")


def bmi_to_weighttype(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi <= 24.9:
        return "NormalWeight"
    elif bmi <= 29.9:
        return "Overweight"
    else:
        return "Obesity"

train_df["WeightType"] = train_df["BMI"].apply(bmi_to_weighttype).astype("category")
test_df["WeightType"] = test_df["BMI"].apply(bmi_to_weighttype).astype("category")


def age_to_group(age):
    if age <= 18:
        return "Child"
    elif age <= 30:
        return "Young Adult"
    elif age <= 50:
        return "Adult"
    else:
        return "Senior"

train_df["AgeGroup"] = train_df["Age"].apply(age_to_group).astype("category")
test_df["AgeGroup"] = test_df["Age"].apply(age_to_group).astype("category")


train_df = train_df.drop(columns=["id"])
test_df = test_df.drop(columns=["id"])


rmv = ["Calories"]
features = [c for c in train_df.columns if c not in rmv]
cats = [c for c in features if train_df[c].dtype == "category"]
nums = [c for c in features if c not in cats]

print(f"Features: {len(features)}\nCategorical: {len(cats)}\nNumerical: {len(nums)}")


print(cats)
print(nums)


def add_feature_crosses(df):
    df = df.copy()
    df["HeartLoad"] = df["Heart_Rate"] * df["Duration"]
    df["TempHeartInteraction"] = df["Body_Temp"] * df["Heart_Rate"]
    df["BMI_Duration"] = df["BMI"] * df["Duration"]
    return df


def add_feature_ratios(df):
    df = df.copy()
    df["Weight_per_Height"] = df["Weight"] / df["Height"]
    df["HeartRate_per_Age"] = df["Heart_Rate"] / df["Age"]
    df['Intensity'] = df['Heart_Rate'] / df['Duration']   
    df['log_Duration'] = np.log1p(df['Duration'])
    df['log_Intensity'] = np.log1p(df['Intensity'])
    return df


train_df = add_feature_crosses(train_df)
train_df = add_feature_ratios(train_df)

test_df = add_feature_crosses(test_df)
test_df = add_feature_ratios(test_df)


from itertools import combinations
from tqdm.auto import tqdm

encoded_columns = []
encode_columns = ['Sex', 'WeightType', 'AgeGroup']
pair_size = [2,3]

for r in pair_size:
    for cols in tqdm(list(combinations(encode_columns, r))):
        new_col_name = '_'.join(cols)
        
        train_df[new_col_name] = train_df[list(cols)].astype(str).agg('_'.join, axis=1)
        train_df[new_col_name] = train_df[new_col_name].astype('category')
        
        test_df[new_col_name] = test_df[list(cols)].astype(str).agg('_'.join, axis=1)
        test_df[new_col_name] = test_df[new_col_name].astype('category')

        encoded_columns.append(new_col_name)


train_df["Calories"] = np.log1p(train_df["Calories"])


rmv = ["Calories"]
features = [c for c in train_df.columns if c not in rmv]
cats = [c for c in features if train_df[c].dtype == "category"]


train_df.head()


test_df.head()


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
"""
def optimize_xgboost(train_df,features,n_trials = 30):
    def objective(trial):
        params = {"n_estimators": trial.suggest_int("n_estimators", 1000, 3000),
                  "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.4, log=True),
                  "max_depth": trial.suggest_int("max_depth", 3, 12),}
                
     # 5-fold cross-validation
        FOLDS = 5
        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
        rmsle_scores = []

        for train_idx, valid_idx in kf.split(train_df.iloc[:20000]):
            X_train, X_valid = train_df.iloc[train_idx][features], train_df.iloc[valid_idx][features]
            y_train, y_valid = train_df.iloc[train_idx][rmv], train_df.iloc[valid_idx][rmv]
        
            model = XGBRegressor(**params, 
                                 objective="reg:squarederror",
                                 eval_metric="rmse",
                                 early_stopping_rounds=100,
                                 random_state=42,
                                 enable_categorical=False, 
                                 verbosity=0)
        
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=0)
        
            preds = model.predict(X_valid)
            preds = np.clip(preds, 0, None)
            rmsle = np.sqrt(mean_squared_log_error(y_valid, preds))
            rmsle_scores.append(rmsle)

        return np.mean(rmsle_scores)

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(direction="minimize") 
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
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


gc.collect()

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
                'max_depth':10,
                'colsample_bytree':0.7,
                'subsample':0.9,
                'n_estimators':2000,
                'learning_rate':0.02,
                'gamma':0.01,
                'max_delta_step':2,
                'eval_metric':'rmse',
                'enable_categorical':True,
                'random_state':42
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

    callbacks = [TQDMCallback(total=xgb_params.get("n_estimators", 1000))]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=150,
        verbose=0,
        callbacks=callbacks
    )

    oof_preds_xgb[valid_idx] = model.predict(X_valid)
    test_preds_xgb += model.predict(X_test) / FOLDS

rmsle = np.sqrt(mean_squared_log_error(np.expm1(train_df[rmv]), np.expm1(oof_preds_xgb)))
rmsle_w_clip = np.sqrt(mean_squared_log_error(np.expm1(train_df[rmv]), np.clip(np.expm1(oof_preds_xgb),1,314)))
print(Fore.GREEN + f"\nValidation RMSLE: {rmsle} "+ Style.RESET_ALL)
print(Fore.GREEN + f"\nValidation with clipped RMSLE: {rmsle_w_clip} "+ Style.RESET_ALL)


sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
sub["Calories"] = np.clip(np.expm1(test_preds_xgb),1,314)
sub.to_csv("submission.csv", index=False)

