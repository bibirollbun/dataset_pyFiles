import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error


def validate(prepare, get_model, metrics, num_folds=5, return_metrics=False, verbose=True):
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
    
    train_df['test'] = 0
    test_df['test'] = 1
    
    train_df['fold'] = np.random.randint(num_folds, size=len(train_df))
    test_df['fold'] = -1
    
    df = pd.concat((train_df, test_df))
    df_p = prepare(df)

    models = []
    results = []
    
    for fold in range(num_folds):
        train = df_p[(df_p.test == 0) & (df_p.fold != fold)]
        train_X = train.drop(['id', 'test', 'fold', 'accident_risk'], axis=1)
        train_y = train['accident_risk']
        
        val = df_p[(df_p.test == 0) & (df_p.fold == fold)]
        val_X = val.drop(['id', 'test', 'fold', 'accident_risk'], axis=1)
        val_y = val['accident_risk']

        model = get_model()
        model.fit(train_X, train_y)
        
        train_pred = model.predict(train_X)
        val_pred = model.predict(val_X)

        train_score = metrics(train_y, train_pred)
        val_score = metrics(val_y, val_pred)

        models.append(model)
        results.append(val_score)

        if verbose:
            print(f'Fold {fold}: train_score={train_score}, val_score={val_score}')

    if verbose:
        print(f'Mean score={np.mean(results)}')

    if return_metrics:
        return models, np.mean(results)
        
    return models


def predict(prepare, models):
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
    
    train_df['test'] = 0
    test_df['test'] = 1
    
    df = pd.concat((train_df, test_df))
    df_p = prepare(df)

    test = df_p[(df_p.test == 1)].copy()
    test_X = test.drop(['id', 'test', 'accident_risk'], axis=1)

    columns = []
    for n, model in enumerate(models):
        col = f'pred_{n}'
        columns.append(col)
        test[col] = model.predict(test_X)

    test['accident_risk'] = np.mean(test[columns], axis=1)
    
    return test


def onehot_encoding(df, cols, drop=False):
    for col in cols:
        values = df[col].unique()
        for val in values:
            df[f'{col}_{val}'] = (df[col] == val).astype(int)
        if drop:
            df = df.drop(col, axis=1)
    return df


def target_encoding(df, cols):
    for col in cols:
        values = df[col].unique()
        replace = {}
        for val in values:
            replace[val] = df[df[col] == val]['accident_risk'].mean()
        df[col] = df[col].replace(replace)
    return df


def binary_encoding(df, cols):
    for col in cols:
        df[col] = df[col].astype(int)
    return df


from sklearn.linear_model import Ridge

def prepare(df):
    categoric = ['road_type', 'lighting', 'weather', 'time_of_day', 'speed_limit', 'num_lanes', 'num_reported_accidents']
    numeric = ['curvature']
    binary = ['road_signs_present', 'public_road', 'holiday', 'school_season']
    
    df = df.copy()
    df = onehot_encoding(df, categoric, drop=True)
    #df = target_encoding(df, categoric)
    df = binary_encoding(df, binary)
    return df

def metrics(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def get_model():
    return Ridge(alpha=1.0)

ridge_models = validate(prepare, get_model, metrics)


from sklearn.tree import DecisionTreeRegressor

def get_model():
    return DecisionTreeRegressor()

tree_models = validate(prepare, get_model, metrics)


import math

def get_ensemble_proba(alpha, n):
    s = 0
    n = int(n)
    for k in range(math.ceil(n / 2), n + 1):
        a = alpha ** k * (1 - alpha) ** (n - k)
        b = math.comb(n, k)
        s += a * b * (0.5 if k == n/2 else 1)
    return s

get_ensemble_proba(alpha=0.5, n=10)


alpha = 0.51

ps = []
ns = np.logspace(0, 3, 20)
for n in ns:
    ps.append(get_ensemble_proba(alpha=alpha, n=n))

plt.figure()
plt.plot(ns, ps)
plt.show()


from sklearn.ensemble import RandomForestRegressor

def get_model():
    return RandomForestRegressor(n_estimators=10)

rf_models = validate(prepare, get_model, metrics)


from sklearn.ensemble import GradientBoostingRegressor

def get_model():
    return GradientBoostingRegressor(n_estimators=10, verbose=0)

gb_models = validate(prepare, get_model, metrics)


from xgboost import XGBRegressor

def get_model():
    return XGBRegressor(n_estimators=100)

xgb_models = validate(prepare, get_model, metrics)


from catboost import CatBoostRegressor

def get_model():
    return CatBoostRegressor(n_estimators=100, verbose=0)

cb_models = validate(prepare, get_model, metrics)


feature_importance = cb_models[0].get_feature_importance()
feature_importance


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df = prepare(train_df.drop(['id', 'accident_risk'], axis=1))
columns = df.columns
columns


data = [df.columns] + [model.get_feature_importance() for model in cb_models]

data = pd.DataFrame(data).T
data['feature_importance'] = data.iloc[:, 1:].mean(axis=1)
data


from sklearn.base import BaseEstimator, RegressorMixin

class FunctionRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, func):
        self.func = func
        
    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        return self.func(X)

def no_prepare(df):
    return df

def f(X):
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)
    
def get_model():
    return FunctionRegressor(f)

func_models = validate(no_prepare, get_model, metrics)


import optuna

def objective(trial):
    #suggest_categorical(name, choice)
    #suggest_float(name, low, high, *, step=None, log=False)
    #suggest_int(name, low, high, step=1, log=False)
    
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

study.best_params  # E.g. {'x': 2.002108042}


def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 1000, step=1, log=True)
    
    get_model = lambda:CatBoostRegressor(n_estimators=n_estimators, verbose=0)
    cb_models, result = validate(prepare, get_model, metrics, return_metrics=True, verbose=False)

    return result

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=2)
study.best_params


from catboost import CatBoostRegressor

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 1000, step=1, log=True)
    
    get_model = lambda:CatBoostRegressor(n_estimators=n_estimators, task_type='GPU', devices='0', verbose=0)
    cb_models, result = validate(prepare, get_model, metrics, return_metrics=True, verbose=False)

    return result

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)
study.best_params




