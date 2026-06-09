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


import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score


import warnings
warnings.filterwarnings('ignore')


# read the training and testing dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


# check the first 5 rows of training dataset
train.head()


# count the no of NaN entries in training dataset
train.isna().sum()


# count the no of NaN entries in testing dataset
test.isna().sum()


# display the info of all columns
train.info()


# selecting the categorical and numerical objects required to train the model from test dataset
cat_cols = test.select_dtypes(include=['object']).columns
num_cols = test.select_dtypes(include=['int64', 'float64']).columns
target_col = ['Personality'] # this is the target the model will be predicting
print("Categorical columns:")
for i in cat_cols:
    print(i)
print("\nNumerical columns:")
for i in num_cols:
    print(i)


# filling the NaN of numerical columns with the median value
for col in num_cols:
    median_val = train[col].median()
    train.fillna({col: median_val}, inplace=True)
    test.fillna({col: median_val}, inplace=True)


# filling the NaN of categorical columns with the mode value
for col in cat_cols:
    mode_val = train[col].mode()[0]
    train.fillna({col: mode_val}, inplace=True)
    test.fillna({col: mode_val}, inplace=True)


# encoding the categorical columns
labels = {} 
for i in cat_cols.append(pd.Index(target_col)):
    labels[i]= LabelEncoder()
    train[i] = labels[i].fit_transform(train[i])


for i in cat_cols:
    test[i] = labels[i].transform(test[i]) 


# these are the features that will be used for training and prediction
features=['Time_spent_Alone','Stage_fear','Social_event_attendance','Going_outside',
          'Drained_after_socializing', 'Friends_circle_size','Post_frequency']


# just separating the features and target
X = train[features]
y = train[target_col]


# splitting the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22, stratify=y)


def objective_xgb(trial):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'logloss',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'random_state': 22,
        'use_label_encoder': False
    }

    model = XGBClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
    return score.mean()

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=30)

best_xgb = XGBClassifier(**study_xgb.best_params, objective='binary:logistic', 
                         tree_method='hist', device='cuda', use_label_encoder=False, random_state=22)


print(f"Best XGBoost parameters:{study_xgb.best_params}")


def objective_lgbm(trial):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    params = {
        'objective': 'binary',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int("num_leaves", 31, 200),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'subsample': trial.suggest_float("subsample", 0.6, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0),
        'device': 'gpu',
        'random_state': 22,
        'verbosity': -1
    }

    model = LGBMClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
    return score.mean()

study_lgbm = optuna.create_study(direction="maximize")
study_lgbm.optimize(objective_lgbm, n_trials=30)

best_lgbm = LGBMClassifier(**study_lgbm.best_params, objective='binary', 
                           device='gpu', random_state=22, verbosity=-1)


print(f"Best LightGBM parameters:{study_lgbm.best_params}")


def objective_cat(trial):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    params = {
        'iterations': 1000,
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.2),
        'depth': trial.suggest_int("depth", 4, 10),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        'random_seed': 22,
        'verbose': 0,
        'task_type': 'CPU',
        'loss_function': 'Logloss'
    }

    model = CatBoostClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
    return score.mean()

study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(objective_cat, n_trials=30)

best_cat = CatBoostClassifier(**study_cat.best_params, iterations=1000, random_seed=22, 
                              task_type='CPU', verbose=0, loss_function='Logloss')


print(f"Best CatBoost parameters:{study_cat.best_params}")


# stack model
meta_model = XGBClassifier(
    objective='binary:logistic',
    tree_method='hist',
    device='cuda',
    # random_state=42,
    # learning_rate=0.05,
    # n_estimators=300,
    # max_depth=4
)

stack_model = StackingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('lgbm', best_lgbm),
        ('catboost', best_cat)
    ],
    final_estimator=meta_model,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    passthrough=True,
    stack_method='predict_proba'
)

stack_model.fit(X_train, y_train)


# calculate the accuray on test split
y_pred = stack_model.predict(X_test)
print(f"Accuracy_score: {accuracy_score(y_test, y_pred)}")


# make final submission
test_pred = stack_model.predict(test[features])
test_pred_labels = labels['Personality'].inverse_transform(test_pred)
submission = pd.DataFrame({'id': test['id'], 'Personality': test_pred_labels})
submission.to_csv('submission.csv', index=False)




