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


## import libraries
import os
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
# import xgboost

# from matplotlib import pyplot as plt

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)


## check available data
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv(r'/kaggle/input/innovative-ai-challenge-2024/train.csv')
df_test = pd.read_csv(r'/kaggle/input/innovative-ai-challenge-2024/test.csv')


df_train.head()


df_train.shape, df_train.isna().sum()


df_train.describe()


non_num_col = df_train.select_dtypes('object').columns.tolist()
for col in non_num_col:
    print(df_train[col].value_counts(), '\n')


irrelevant_fields = set(['State', 'id'])
cat_features = set(non_num_col) - irrelevant_fields
target = set(['Crop_Yield (kg/ha)'])

num_features = set(df_train.columns) - target - irrelevant_fields

features = list(num_features.union(cat_features))
target = list(target)

print(features, target)


temp_dict = dict(zip(list(cat_features), ['category']*len(cat_features)))
df_train = df_train.astype(temp_dict) 

df_train[features].dtypes


X_train = df_train[features].copy()
y_train = df_train[target].copy()


%%time

# Randomized Search
fit_params = {"eval_metric": 'rmse',
              'tree_method': 'hist',
              'use_label_encoder': False,
              'enable_categorical': True,
              'early_stopping_rounds': 50
             }

xgb = XGBRegressor(**fit_params)
kfolds = 5

params = {
    'min_child_weight': [1, 2, 3, 4, 5],  ## stop splitting after you reach 'x' degree of purity
    'gamma': [0.5, 1, 1.5, 2],  ## min loss reduction to make the split
    'max_depth': [2,3],  ## depth of a tree
    'learning_rate': [0.001, 0.01, 0.1, 0.5],
    'n_estimators': [50, 150, 250, 350, 500],
    'reg_alpha': [0.0001, 0.001, 0.1, 1],  ## lasso reg
    'reg_lambda': [0.0001, 0.001, 0.1, 1],  ## ridge reg
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'random_state': [54],
    'seed': [10]
}

random_search = RandomizedSearchCV(xgb,
                                   cv = kfolds,
                                   param_distributions = params,
                                   n_iter = 500,
                                   scoring='neg_root_mean_squared_error',
                                   verbose=2,
                                   return_train_score = True,
                                   random_state = 54,
                                   refit=True,
                                   # n_jobs = 2
                                  )

evaluation = [(X_train, y_train)]
random_search.fit(X_train, y_train, eval_set=evaluation, verbose=False)


all_results = pd.DataFrame(random_search.cv_results_)
all_results


print(random_search.best_estimator_)


best_hyperparams = random_search.best_params_
best_hyperparams


all_results[(all_results.param_n_estimators==250) & (all_results.param_min_child_weight==3) & 
            (all_results.param_max_depth==3) & (all_results.param_learning_rate==0.1) & 
            (all_results.param_gamma==1.5) & (all_results.param_colsample_bytree==0.6)]


all_results[(all_results.mean_train_score>-25) & (all_results.std_train_score<10)]


# selected_params = all_results['params'].loc[367]
selected_params = {
                     'seed': 10,
                     'reg_lambda': 0.001,
                     'reg_alpha': 0.0001,
                     'random_state': 54,
                     'n_estimators': 250,
                     'min_child_weight': 1,
                     'max_depth': 3,
                     'learning_rate': 0.5,
                     'gamma': 2,
                     'colsample_bytree': 0.5}


%%time 
model = XGBRegressor(
        n_estimators = int(selected_params['n_estimators']),
        max_depth = int(selected_params['max_depth']),
        gamma = selected_params['gamma'],
        reg_alpha = selected_params['reg_alpha'],
        reg_lambda = selected_params['reg_lambda'],
        min_child_weight = selected_params['min_child_weight'],
        learning_rate = selected_params['learning_rate'],
        colsample_bytree = selected_params['colsample_bytree'],
        random_state = selected_params['random_state'],
        seed = selected_params['seed'], **fit_params)

evaluation = [(X_train, y_train), (X_train, y_train)]
model.fit(X_train, y_train, eval_set=evaluation, verbose=True)


X_test = df_test[features].copy()
X_test = X_test.astype(temp_dict)   # convert to categorical dtypes 

X_test.dtypes


y_pred = model.predict(X_test)
y_pred


# Save the model for future use
import pickle
pickle.dump(model, open('crop_yield_model_v1.pkl', 'wb'))


df_submission = df_test[['id']].copy()
df_submission['Target'] = y_pred
df_submission.to_csv('submission_crop_yield_model_v1.csv', index=False)
print(df_submission.head(10))


for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

