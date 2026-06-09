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
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from xgboost import XGBRegressor


data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col=0)
target = data['Price']

test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col=0)

encoder = TargetEncoder(verbose=1, drop_invariant=True, return_df=True, min_samples_leaf=20, smoothing=10)
encoder = encoder.fit(data.drop(columns='Price'), target)
data = encoder.transform(data.drop(columns='Price'))
test = encoder.transform(test)


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)


def objective_xg(trial):

    params = {
        "n_estimators": 70,
        "eval_metric": "rmse",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
        "min_child_weight": trial.suggest_int("min_child_weight", 0.01, 1),
        "subsample": trial.suggest_loguniform("subsample", 0.1, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1)
    }

    model =  XGBRegressor(**params, enable_categorical = False)

    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return mean_squared_error(y_test, y_pred, squared = False)


trial = optuna.create_study(direction="minimize")
optuna.logging.set_verbosity(optuna.logging.WARNING)
trial.optimize(objective_xg, n_trials=5, show_progress_bar=True)


trial.best_params


cls = XGBRegressor(**trial.best_params, n_estimators = 100,
                   eval_metric = "rmse", enable_categorical = False)

cls.fit(X_train, y_train)


mean_squared_error(cls.predict(X_test), y_test)


test = test.assign(Price=cls.predict(test))
test.head()


test['Price'].reset_index().to_csv('submission.csv', index=False)




