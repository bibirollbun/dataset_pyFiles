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


pip install optuna-integration[xgboost]


import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import optuna



















TRAINPATH = r"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv"
TESTPATH  = r"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv"
dftrainraw = pd.read_csv(TRAINPATH)
dftestraw = pd.read_csv(TESTPATH)




dftrainclean = dftrainraw.dropna(subset=['CORRUCYSTIC_DENSITY']).copy()

X = dftrainclean.drop(['CORRUCYSTIC_DENSITY', 'LOCAL_IDENTIFIER'], axis=1)
y = dftrainclean['CORRUCYSTIC_DENSITY']
Xtest = dftestraw.drop('LOCAL_IDENTIFIER', axis=1)


numericalfeatures = X.select_dtypes(include=np.number).columns.tolist()
categoricalfeatures = X.select_dtypes(include='object').columns.tolist()

numericaltransformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

categoricaltransformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numericaltransformer, numericalfeatures),
        ('cat', categoricaltransformer, categoricalfeatures)
    ],
    remainder='passthrough'
)


Xtrainopt, Xvalopt, ytrainopt, yvalopt = train_test_split(X, y, test_size=0.2, random_state=42)
Xtrainoptprocessed = preprocessor.fit_transform(Xtrainopt)
Xvaloptprocessed = preprocessor.transform(Xvalopt)



def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.3),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'tree_method': 'hist',
        'random_state': 42
    }
    model = xgb.XGBRegressor(**params)
    pruningcallback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-rmse")

    model.fit(Xtrainoptprocessed, ytrainopt,
              eval_set=[(Xvaloptprocessed, yvalopt)],
              early_stopping_rounds=50,
              callbacks=[pruningcallback],
              verbose=False)
    preds = model.predict(Xvaloptprocessed)
    rmse = np.sqrt(mean_squared_error(yvalopt, preds))

    return rmse



study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(objective, n_trials=100)
bestxgbparams = study.best_params
print(f"Best XGBoost params found: {bestxgbparams}")


xgbpipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(random_state=42, **bestxgbparams))
])


xgbpipeline.fit(Xtrainopt, ytrainopt)
xgbpreds = xgbpipeline.predict(Xtest)




submission = pd.DataFrame({
    'LOCAL_IDENTIFIER': dftestraw['LOCAL_IDENTIFIER'],
    'CORRUCYSTIC_DENSITY': xgbpreds
})

submission.to_csv('submission.csv', index=False)
print(submission.head(20))




