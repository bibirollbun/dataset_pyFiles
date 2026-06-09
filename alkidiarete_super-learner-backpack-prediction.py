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


import pandas as pd
import numpy as np
import seaborn as sns
import optuna
import matplotlib.pyplot as plt
from math import sqrt
from numpy import hstack
from numpy import vstack
from numpy import asarray
from cuml.preprocessing import TargetEncoder
from cuml.linear_model import LinearRegression as cuLinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from cuml.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')



train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col='id')
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col='id')

train = pd.concat([train, train_extra], axis=0, ignore_index=True)


def feature_engineering(df):
    size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
    df['Size_Num'] = df['Size'].map(size_mapping)
    df['Compartments_per_Size'] = df['Compartments'] / df['Size_Num']    
    df['Weight_per_Compartment'] = df['Weight Capacity (kg)'] / df['Compartments'] 
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Waterproof_Laptop'] = df['Waterproof'] * df['Laptop Compartment']
    df['Is_Durable_Material'] = df['Material'].apply(lambda x: 1 if x in ['Leather', 'Nylon'] else 0)
    df['Is_Lightweight_Material'] = df['Material'].apply(lambda x: 1 if x in ['Canvas', 'Nylon'] else 0)
    df['Luxury_Material'] = df['Material'].apply(lambda x: 1 if x == 'Leather' else 0)
    df['Professional_Style'] = df['Style'].apply(lambda x: 1 if x in ['Messenger', 'Tote'] else 0)
    df['Casual_Style'] = df['Style'].apply(lambda x: 1 if x in ['Backpack', 'Duffle'] else 0)
    df['Is_Premium_Brand'] = df['Brand'].apply(lambda x: 1 if x in ['Nike', 'Under Armour', 'Adidas'] else 0)
    df['Is_Budget_Brand'] = df['Brand'].apply(lambda x: 1 if x == 'Jansport' else 0)
    df['Is_Small'] = df['Size'].apply(lambda x: 1 if x == 'Small' else 0)
    df['Is_Medium'] = df['Size'].apply(lambda x: 1 if x == 'Medium' else 0)
    df['Is_Large'] = df['Size'].apply(lambda x: 1 if x == 'Large' else 0)

    return df

train = feature_engineering(train)
test = feature_engineering(test)


target = "Price"
features = [col for col in train.columns if col != target]
CATS = [col for col in train.columns if col not in [target, "Weight Capacity (kg)"]]

for col in CATS:
    train[col] = train[col].fillna('Missing').astype(str)
    test[col] = test[col].fillna('Missing').astype(str)


TE = TargetEncoder(n_folds=5, smooth=20, split_method='random', stat='mean')

for col in CATS:
    train[f"TE_{col}"] = TE.fit_transform(train[col], train["Price"])
    test[f"TE_{col}"] = TE.transform(test[col])

all_features = features + [f"TE_{col}" for col in CATS]

train = train.drop(columns=CATS)
test = test.drop(columns=CATS)


X = train.drop('Price', axis=1)  
y = train['Price']  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)


def get_models():
    return [
        XGBRegressor(tree_method='hist', device='cuda'),
        LGBMRegressor(device_type='gpu', verbose=-1),
        CatBoostRegressor(task_type='GPU', verbose=False),
    ]

def get_out_of_fold_predictions(X, y, models):
    meta_X, meta_y = list(), list()
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    for train_ix, test_ix in kfold.split(X):
        fold_yhats = list()
        train_X, test_X = X.iloc[train_ix], X.iloc[test_ix]
        train_y, test_y = y.iloc[train_ix], y.iloc[test_ix]
        meta_y.extend(test_y)
        for model in models:
            model.fit(train_X, train_y)
            yhat = model.predict(test_X)
            fold_yhats.append(yhat.reshape(-1, 1))
        meta_X.append(np.hstack(fold_yhats))
    return np.vstack(meta_X), np.array(meta_y)

def fit_base_models(X, y, models):
    for model in models:
        model.fit(X, y)

def fit_meta_model(X, y):
    model = cuLinearRegression(fit_intercept=True, normalize=True, algorithm='eig')
    model.fit(X, y)
    return model

def evaluate_models(X, y, models):
    for model in models:
        yhat = model.predict(X)
        yhat = np.maximum(yhat, 0)
        rmse = np.sqrt(mean_squared_error(y, yhat))
        print(f'{model.__class__.__name__}: RMSE {rmse:.3f}')

def super_learner_predictions(X, models, meta_model):
    meta_X = np.hstack([model.predict(X).reshape(-1, 1) for model in models])
    return meta_model.predict(meta_X)

models = get_models()

meta_X, meta_y = get_out_of_fold_predictions(X_train, y_train, models)
print('Meta Data Shape:', meta_X.shape, meta_y.shape)

fit_base_models(X_train, y_train, models)

meta_model = fit_meta_model(meta_X, meta_y)

evaluate_models(X_val, y_val, models)

yhat = super_learner_predictions(X_val, models, meta_model)
yhat = np.maximum(yhat, 0)  
rmse = np.sqrt(mean_squared_error(y_val, yhat))
print(f'Super Learner: RMSE {rmse:.3f}')


test_pred = super_learner_predictions(test, models, meta_model)


sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

output = pd.DataFrame({"id": sub.id, "Price": test_pred})
output.to_csv('submission_ensemble.csv', index=False)

output.head()

