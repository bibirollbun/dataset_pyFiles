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


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
ids = test['id']
test = test.drop(columns='id')
y_train = train['rainfall']
X_train = train.drop(columns=['rainfall', 'id'])



test = test.fillna(value=test.mean())





from lightgbm import LGBMRegressor






'''import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from optuna.integration import XGBoostPruningCallback
import numpy as np

# Разделение данных
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': trial.suggest_float('eta', 0.001, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'verbosity': 0,
    }
    
    train_data = xgb.DMatrix(X_train, label=y_train)
    val_data = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params,
        train_data,
        num_boost_round=1000,
        evals=[(val_data, 'eval')],
        verbose_eval=False,
        callbacks=[XGBoostPruningCallback(trial, 'eval-rmse')]
    )
    
    preds = model.predict(val_data)
    return mean_squared_error(y_val, preds)

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.HyperbandPruner()
)
study.optimize(objective, n_trials=100)

print('Лучшие параметры:', study.best_params)
print('Лучшая MSE:', study.best_value)

# Создание финальной модели
best_params = study.best_params.copy()
best_params.update({
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'verbosity': 0
})

best_model = xgb.train(
    best_params,
    xgb.DMatrix(X_train, y_train),
    num_boost_round=1000
)'''


from catboost import CatBoostClassifier


model = CatBoostClassifier()
model.fit(X_train, y_train)


y_test = model.predict(test)


y_test = np.where(y_test > 1, 1, y_test)
y_test = np.where(y_test < 0, 0, y_test)


ss = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


ss


output = pd.DataFrame({'id': ids, 'rainfall': y_test})


output.to_csv("submission_baseline.csv", index=False)

