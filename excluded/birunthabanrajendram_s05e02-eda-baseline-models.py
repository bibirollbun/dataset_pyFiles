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


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

import optuna
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
org = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")


train.head(10)


train.isnull().sum()


train.dtypes


test.sample(5)


test.dtypes


test.isnull().sum()


train = pd.concat([train, org], axis = 0, ignore_index = True)


combined = pd.concat([train, test], axis = 0)

num_cols = test.select_dtypes(include = ['number']).columns
object_cols = train.select_dtypes(include = ['object']).columns

impute_value = combined[num_cols].median()

combined[num_cols] = combined[num_cols].fillna('impute_value')
combined[object_cols] = combined[object_cols].fillna('None')


train = combined.iloc[:len(train)]
test = combined.iloc[len(train):]


from sklearn.model_selection import KFold
from category_encoders import TargetEncoder

TE = TargetEncoder(smoothing = 20)

cat = train.select_dtypes(include=['object','category']).columns

train[cat] = TE.fit_transform(train[cat], train['Price'])
test[cat] = TE.transform(test[cat])


X = train.drop(['Price'], axis = 1)
y = train['Price']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state = 42)


# def objective(trial):
#     params = {
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log = True),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
#         'gamma': trial.suggest_float('gamma', 0, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
#         'random_state': 42,
#         'n_jobs': -1,
#         'eval_metric': 'rmse',
#     }

#     model = XGBRegressor(**params)

#     model.fit(
#         X_train, y_train,
#         eval_set=[(X_valid, y_valid)],
#         early_stopping_rounds = 100,
#         verbose = False
#     )
#     val_predictions = model.predict(X_valid)
#     rmse = mean_squared_error(y_valid, val_predictions, squared=False)
#     return rmse

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)

# print('Best Hyperparameters: ', study.best_params)
# print('Beat RMSE: ', study.best_value)


params = {'max_depth': 3, 'n_estimators': 1228, 'learning_rate': 0.007545367144920574, 'subsample': 0.6664167687024933, 'colsample_bytree': 0.6373895785838637, 'min_child_weight': 6.508506116427075, 'gamma': 0.2250895433766551, 'reg_alpha': 0.19600669706195117, 'reg_lambda': 0.7086268511778124}


final_model = XGBRegressor(**params, random_state=42, n_jobs = -1)

final_model.fit(
    X_train, y_train,
    eval_set = [(X_valid, y_valid)],
    early_stopping_rounds = 100,
    verbose = 200
)


val_predictions = final_model.predict(X_valid)

val_rmse = mean_squared_error(y_valid, val_predictions, squared=False)
print(f"Final RMSE: ", {val_rmse})


y_test_pred = final_model.predict(test)

submission = pd.DataFrame({'id': text.index, 'Price': y_test_pred})
submission.to_csv('submission.csv', index=False)

display(submission)


# model = XGBRegresoor(
#     max_depth = 5,
#     n_estimators = 2000
# )

