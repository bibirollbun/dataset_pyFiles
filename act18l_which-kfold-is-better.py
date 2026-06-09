import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold,TimeSeriesSplit,GroupKFold
from sklearn.metrics import mean_absolute_percentage_error
import logging
import optuna
import warnings
warnings.filterwarnings('ignore')
import gc


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
train = train.dropna(subset=['num_sold'])
train['date'] = pd.to_datetime(train['date'])
test = train[train['date'].dt.year == 2016]
train = train[train['date'].dt.year != 2016]




def process_date_features(df):
    df['date'] = pd.to_datetime(df['date'])

    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    df['month_name'] = df['date'].dt.month_name()
    df['day_of_week'] = df['date'].dt.day_name()

    df['week'] = df['date'].dt.isocalendar().week

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    df['group'] = (df['year'] - 2020) * 48 + df['month'] * 4 + df['day'] // 7

    df.drop('date', axis=1, inplace=True)

    df['cos_year'] = np.cos(df['year'] * (2 * np.pi) / 100)
    df['sin_year'] = np.sin(df['year'] * (2 * np.pi) / 100)

    df['Season'] = df['month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else
                                              'Spring' if x in [3, 4, 5] else
                                              'Summer' if x in [6, 7, 8] else
                                              'Autumn')

    dummy_prefixes = ['country', 'store', 'product', 'month_name', 'day_of_week', 'Season']
    df = pd.get_dummies(df, columns=dummy_prefixes, drop_first=True)

    return df

train = process_date_features(train)
test = process_date_features(test)


X = train.drop(columns=['num_sold'])
y = np.log1p(train['num_sold'])
test_x = test[X.columns]
test_y = np.log1p(test['num_sold'])


def lgbm_objective(trial,kf,group=None):
    lgbm_params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 4, 25),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 0.001, 1.0),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 0.001, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.3, 1.0),
        "objective": "mse",
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-4, 1.0),
        "random_state": 42,
        "verbose": -1,
    }
    mape_scores = []
    if isinstance(kf, KFold):
        split_tmp = kf.split(X,y)
    elif isinstance(kf, GroupKFold):
        groups = (X['year'] - 2010) * 48 + X['month'] * 4 + X['day'] // 7
        split_tmp = kf.split(X, y, groups)
    elif  isinstance(kf, TimeSeriesSplit):
        split_tmp = kf.split(X, y)
    for train_index, valid_index in split_tmp:
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        lgbm_model = LGBMRegressor(**lgbm_params)
        lgbm_model.fit(X_train, y_train)
        y_pred = np.expm1(lgbm_model.predict(X_valid))
        score = mean_absolute_percentage_error(np.expm1(y_valid), y_pred)
        mape_scores.append(score)
    return  np.mean(mape_scores)


optuna_times = 100


kf = KFold(n_splits=5, shuffle=True, random_state=42)
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: lgbm_objective(trial, kf), n_trials=optuna_times)


best_params = study.best_params
best_params["objective"]="mse"
final_model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
final_model.fit(X, y)
final_preds = np.expm1(final_model.predict(test_x))
final_mape = mean_absolute_percentage_error(np.expm1(test_y), final_preds)
print(f"Final Model MAPE: {final_mape}")


kf = KFold(n_splits=5)
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: lgbm_objective(trial, kf), n_trials=optuna_times)


best_params = study.best_params
best_params["objective"]="mse"
final_model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
final_model.fit(X, y)
final_preds = np.expm1(final_model.predict(test_x))
final_mape = mean_absolute_percentage_error(np.expm1(test_y), final_preds)
print(f"Final Model MAPE: {final_mape}")


kf = GroupKFold(n_splits=5)
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: lgbm_objective(trial, kf), n_trials=optuna_times)


best_params = study.best_params
best_params["objective"]="mse"
final_model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
final_model.fit(X, y)
final_preds = np.expm1(final_model.predict(test_x))
final_mape = mean_absolute_percentage_error(np.expm1(test_y), final_preds)
print(f"Final Model MAPE: {final_mape}")


kf = TimeSeriesSplit(n_splits=5)
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: lgbm_objective(trial, kf), n_trials=optuna_times)


best_params = study.best_params
best_params["objective"]="mse"
final_model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
final_model.fit(X, y)
final_preds = np.expm1(final_model.predict(test_x))
final_mape = mean_absolute_percentage_error(np.expm1(test_y), final_preds)
print(f"Final Model MAPE: {final_mape}")




