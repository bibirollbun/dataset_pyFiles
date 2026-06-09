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


train_df = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')
test_df = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')

sub = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/sample_submission.csv')


target = train_df['HOMELESS_RATE']


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,cross_val_score, KFold

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer



# base model
models = [LinearRegression, Ridge, Lasso, ElasticNet,
          RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor,
          XGBRegressor, LGBMRegressor, CatBoostRegressor, SVR]

X_train, X_test, y_train, y_test = train_test_split(train_df.drop(['ID', 'HOMELESS_RATE'], axis = 1), target, test_size=0.2, random_state=42)

results = {}

# make_scorerを使って、scoringに入力するためのroot_mean_squred_errorを作成します

rmse_scorer = make_scorer(mean_squared_error, squared=False)

for model in models:
    model_name = model.__name__
    print(f"Performing cross-validation for {model_name}...")
    scores = cross_val_score(model(), X_train, y_train, cv=5, scoring=rmse_scorer)
    results[model_name] = scores.mean()
    print(f"{model_name} - Mean RMSE: {scores.mean():.4f}")
    print(f"{model_name} - RMSE scores: {scores}\n")

best_model_name = min(results, key=results.get)
print(f"Best model based on cross-validation RMSE: {best_model_name}")

df_model_score = pd.DataFrame.from_dict(results, orient='index', columns=['RMSE'])
df_model_score


df_model_score.sort_values(by='RMSE')


"""
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor

param_grid_catboost = {
    'iterations': [100, 250, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5],
    # 'border_count': [32, 64, 128],
    # 'random_strength': [0, 1, 2],
    # 'bagging_temperature': [0, 1, 2],
    # 'od_type': ['IncToDec', 'Iter'],
    # 'od_wait': [10, 20, 30],
    'verbose': [0] # Suppress verbose output during grid search
}

# Initialize CatBoostRegressor
catboost = CatBoostRegressor(random_state=42)

# Initialize GridSearchCV with the model and parameter grid
# Using the same RMSE scorer as before
grid_search_catboost = GridSearchCV(estimator=catboost, param_grid=param_grid_catboost, cv=5, scoring=rmse_scorer, n_jobs=-1)

print("Performing Grid Search for CatBoostRegressor...")
# Fit the grid search to the training data
grid_search_catboost.fit(X_train, y_train)

# Print the best parameters and the corresponding best score
print("\nBest parameters found: ", grid_search_catboost.best_params_)
print("Best RMSE score found: {:.4f}".format(grid_search_catboost.best_score_))

# Get the best model from the grid search
best_catboost_model = grid_search_catboost.best_estimator_
best_catboost_model
"""


"""
best_catboost_model.fit(X_train, y_train)

prediction = best_catboost_model.predict(test_df.drop('ID', axis = 1))
sub['HOMELESS_RATE'] = prediction

sub.to_csv('submission.csv', index = False)
sub
"""



from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

# Define the parameter grid for Ridge
param_grid_ridge = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'],
    'fit_intercept': [True, False],
    'max_iter': [100, 200, 300],
    'tol': [1e-4, 1e-5, 1e-6],
}

# Initialize Ridge
ridge = Ridge(random_state=42)

# Initialize GridSearchCV with the model and parameter grid
# Using the same RMSE scorer as before
grid_search_ridge = GridSearchCV(estimator=ridge, param_grid=param_grid_ridge, cv=5, scoring=rmse_scorer, n_jobs=-1)

print("Performing Grid Search for Ridge...")
# Fit the grid search to the training data
grid_search_ridge.fit(X_train, y_train)

# Print the best parameters and the corresponding best score
print("\nBest parameters found: ", grid_search_ridge.best_params_)
print("Best RMSE score found: {:.4f}".format(grid_search_ridge.best_score_))

# Get the best model from the grid search
best_ridge_model = grid_search_ridge.best_estimator_





prediction = best_ridge_model.predict(test_df.drop('ID', axis = 1))

pred_plus = []

for pred in prediction:
    if pred >=0:
        pred = pred
    elif pred <0:
        pred = 0
    pred_plus.append(float(pred))

print(pred_plus)


sub['HOMELESS_RATE'] = pred_plus
sub.to_csv('submission.csv')
sub





