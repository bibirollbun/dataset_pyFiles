# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# (Gerekirse) paket kurulumu
# !pip -q install optuna xgboost

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_log_error
from sklearn.utils import check_random_state

import optuna
from xgboost import XGBRegressor

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s4e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e4/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s4e4/sample_submission.csv')


train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)


train.head()


train.info()


# mapping = {'F':0,'M':1,'I':2}
# train['Sex'] = train['Sex'].map(mapping)
# test['Sex'] = test['Sex'].map(mapping)


train.isnull().sum()


train['Sex'].unique()


target = 'Rings'

number_columns = train.select_dtypes(include=['int64','float64']).columns.tolist()
if target in number_columns:
    number_columns.remove(target)

cat_columns = [i for i in train.columns if i not in number_columns +[target]]

log = []
for j in number_columns:
    if (train[j] >= 0).all():
        if train[j].skew() > 0:
            log.append(j)
print(log)


def stratify_bins(y,n_bins = 10, random_state=42):
    rng = check_random_state(random_state)
    y = pd.Series(y)

    if y.nunique() == 1:
        return np.zeros(len(y), dtype=int)

    bins = pd.cut(y, bins=n_bins, labels = False)

    if bins.isna().any():
        y_noisy = y + rng.normal(0, 1e-4, size=len(y))
        bins = pd.cut(y_noisy, bins=n_bins, labels=False)
    return bins.astype(int)


preprocessor = ColumnTransformer(
    transformers=[
        ('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one'),log),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_columns),
        ],
    remainder = 'passthrough',
    verbose_feature_names_out=False
)

pipeline_template = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('xgb',XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    ))
])


def rmsle_skore(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return mean_squared_log_error(y_true, y_pred, squared=False)


def objective_function(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 50.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 50.0),
        'max_bin': trial.suggest_int('max_bin', 256, 4096, log=True),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
    }

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('xgb', XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',
            n_jobs=-1,
            random_state=42,
            **params
        ))
    ])

    X = train.drop(columns=[target])
    y = train[target].values

    y_bins = stratify_bins(y, n_bins=10, random_state=42)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_scores = []
    for fold,(tr_idx, va_idx) in enumerate(kf.split(X,y_bins)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        pipeline.fit(X_tr, y_tr)
        va_pred = pipeline.predict(X_va)

        score = rmsle_skore(y_va, va_pred)
        fold_scores.append(score)
    
    return float(np.mean(fold_scores))




optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='minimize', study_name ='xgb_rmsle_abalone')
study.optimize(objective_function,n_trials=25, show_progress_bar=True)

print("En iyi RMSLE:", study.best_value)
print("En iyi parametreler:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")



best_param = study.best_params.copy()

finaly_pipeline = Pipeline(steps=[
    ('ct', preprocessor),
    ('xgb', XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        n_jobs=-1,
        random_state=42,
        **best_param
    ))
])

X_full = train.drop(columns=[target])
y_full = train[target].values
finaly_pipeline.fit(X_full, y_full)

test_pred = finaly_pipeline.predict(test)
test_pred = np.clip(test_pred, 0, None)


if 'Rings' in sub.columns:
    sub['Rings'] = test_pred
elif 'rings' in sub.columns:
    sub['rings'] = test_pred
else:
    sub['Rings'] = test_pred

sub.to_csv('submission.csv', index=False)
sub.head()





