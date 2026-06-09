import os
import sys
import optuna
import scipy

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")


optuna.logging.set_verbosity(optuna.logging.WARNING)


def f(X):
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
df_orig = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')


categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day']
numerical_features = ['curvature', 'num_lanes', 'speed_limit', 'num_reported_accidents']
boolean_features = ['road_signs_present', 'public_road', 'holiday', 'school_season']

predictors = categorical_features + numerical_features + boolean_features
target = 'accident_risk'


df_train = df_train.drop(columns = ['id'])
df_train.drop_duplicates(keep = 'first', inplace = True, ignore_index = True)


df_train['orig_target'] = f(df_train)
df_test['orig_target'] = f(df_test)


original_columns = []
global_orig_mean = df_orig[target].mean()
for col in predictors:
    new_col_name = f"orig_{col}"
    original_columns.append(new_col_name)
    
    df_tmp = df_orig.groupby(col)[target].mean()
    df_tmp.name = new_col_name
    df_train = df_train.merge(df_tmp, on = col, how = 'left')
    df_test = df_test.merge(df_tmp, on = col, how = 'left')

    df_train[new_col_name] = df_train[new_col_name].fillna(global_orig_mean)
    df_test[new_col_name] = df_test[new_col_name].fillna(global_orig_mean)


road_type_map = {'rural': 1, 'highway': 2, 'urban': 3}
lighting_map = {'daylight': 1, 'dim': 2, 'night': 3}
weather_map = {'clear': 1, 'rainy': 2, 'foggy': 3}
time_of_day_map = {'morning': 1, 'afternoon': 2, 'evening': 3}


df_train['road_type'] = df_train['road_type'].replace(road_type_map)
df_test['road_type'] = df_test['road_type'].replace(road_type_map)

df_train['lighting'] = df_train['lighting'].replace(lighting_map)
df_test['lighting'] = df_test['lighting'].replace(lighting_map)

df_train['weather'] = df_train['weather'].replace(weather_map)
df_test['weather'] = df_test['weather'].replace(weather_map)

df_train['time_of_day'] = df_train['time_of_day'].replace(time_of_day_map)
df_test['time_of_day'] = df_test['time_of_day'].replace(time_of_day_map)


for feature in boolean_features:
    df_train[feature] = df_train[feature].astype('int')
    df_test[feature] = df_test[feature].astype('int')


lr_model = LinearRegression().fit(X = df_train[['curvature']], 
                                  y = df_train[['accident_risk']])
df_train['curvature_reg'] = lr_model.predict(df_train[['curvature']])
df_test['curvature_reg'] = lr_model.predict(df_test[['curvature']])


predictors += original_columns + ['orig_target', 'curvature_reg']


def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "max_features": trial.suggest_float("max_features", 0.25, 1.0, step = 0.01),
        "max_samples": trial.suggest_float("max_samples", 0.25, 1.0, step = 0.01),
        "min_samples_split": trial.suggest_int("min_samples_split", 10, 1000, step = 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100)}
    
        
    alg = RandomForestRegressor(**params,
                                n_estimators = 250,
                                criterion = 'squared_error',
                                bootstrap = True,
                                oob_score = True,
                                random_state = 42,
                                verbose = 0,
                                n_jobs = 2)
    alg.fit(df_train[predictors], df_train[target])
    oob_pred = alg.oob_prediction_

    
    return mean_squared_error(df_train[target], oob_pred, squared = False)


study = optuna.create_study(direction = 'minimize', study_name = 'random forest')
study.optimize(func = objective, 
               n_trials = 100,
               n_jobs = 2,
               gc_after_trial = False,
               show_progress_bar = False)


print('Best hyperparameters:', study.best_params)
print("----------")
print('Best RMSE:', study.best_value)


params = study.best_params


alg = RandomForestRegressor(**params,
                            n_estimators = 250,
                            criterion = 'squared_error',
                            bootstrap = True,
                            oob_score = True,
                            random_state = 42,
                            verbose = 0,
                            n_jobs = 2)
alg.fit(df_train[predictors], df_train[target])
oob_pred = alg.oob_prediction_


rmse = mean_squared_error(df_train[target], oob_pred, squared = False)
print("OOB RMSE: ", rmse)


test_pred = alg.predict(df_test[predictors])


submission = pd.DataFrame({'id': df_test['id'], 'accident_risk': test_pred})
submission.to_csv('/kaggle/working/submission.csv', index = False)

