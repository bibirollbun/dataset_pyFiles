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


from typing import Tuple, Optional

import optuna
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def data_process(fname: str = 'train') -> Tuple[np.ndarray, Optional[np.ndarray]]:
    df = pd.read_csv(f'/kaggle/input/playground-series-s5e3/{fname}.csv', index_col=0)
    
    y = None
    if 'rainfall' in df.columns:
        X = StandardScaler().fit_transform(df.drop(columns=['rainfall']))
        y = df['rainfall'].to_numpy()
    else:
        X = StandardScaler().fit_transform(df)

    return X, y


def objective(trial:optuna.trial.Trial):
    X, y = data_process()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    
    params = {
        'n_estimators': trial.suggest_int(name='n_estimators', low=50, high=100),
        'max_depth': trial.suggest_int(name='max_depth', low=2, high=10),
        'max_leaves': trial.suggest_int(name='max_leaves', low=3, high=20),
        'grow_policy': trial.suggest_categorical(name='grow_policy', choices=['depthwise', 'lossguide']),
        'learning_rate': trial.suggest_float(name='learning_rate', low=.001, high=.1),
        'subsample': trial.suggest_float(name='subsample', low=.1, high=1),
        'reg_alpha': trial.suggest_float(name='reg_alpha', low=.01, high=1),
        'reg_lambda': trial.suggest_float(name='reg_lambda', low=.01, high=1.0),
    }

    cls = xgb.XGBRegressor(**params)
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    return float(mean_squared_error(y_test, y_pred))


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20, show_progress_bar=True)


params = study.best_params
params


data,_ = data_process('test')
data


X, y = data_process()

cls = xgb.XGBRegressor(**params).fit(X, y)
# cls.predict(data)


df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
df = df.assign(rainfall=cls.predict(data))
df.head()


df[['id', 'rainfall']].to_csv('submission.csv', index=False)




