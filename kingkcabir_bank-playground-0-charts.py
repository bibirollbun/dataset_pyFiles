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


bank_tr = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
bank_ts = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
lent = '*'*40
class get_summary:
    def __init__(self, x):
        self.x = x if isinstance(x, pd.DataFrame) else pd.DataFrame()
    def data_set(self):
        #checks for duplicate
        duplicate = self.x.duplicated().any()
        #drop duplicates 
        if duplicate == True:
            self.x.drop_duplicates(inplace=True)
            self.x.reset_index(drop=True)
             #checks for empty values
        null = self.x.isna().sum().any()
        #missing values
        total_missing = self.x.isnull().sum().sum()
        #data types
        data_type = self.x.dtypes
        #shape
        shapes = self.x.shape
        return f"Duplicate: {duplicate}\nNull: {null}\nMissing_value: {total_missing}\nTypes:\n{data_type}\nShape: {shapes}"
    #missing values
    def total_missing(self):
        missing_vals = self.x.isnull().sum()
        cols_with_missing = missing_vals[missing_vals > 0]
        if not cols_with_missing.empty:
            return cols_with_missing.to_dict()
        else:
            return f"{'......No missing values detected......'}"
print(f"Training dataset:\n{get_summary(bank_tr).data_set()}\n{lent}\nTest dataset:\n{get_summary(bank_ts).data_set()}")
print(f"{lent}\ncolumns with missing values train\n{lent}\n{get_summary(bank_tr).total_missing()}\n{lent}\ncolumns with missing values test\n{lent}\n{get_summary(bank_ts).total_missing()}")



from sklearn.preprocessing import LabelEncoder, MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import optuna


tr_features = bank_tr.drop(['id', 'y'], axis=1)
ts_features = bank_ts.drop('id', axis=1)


enc = LabelEncoder()
def encode_catFeatures(x):
    for vals in x.columns:
        if x[vals].dtype != 'int64':
            x[vals] = pd.DataFrame(enc.fit_transform(x[vals]))
            
    return x

X = encode_catFeatures(tr_features)


scala = MinMaxScaler(feature_range=(0,1))
X_scaled = pd.DataFrame(scala.fit_transform(X))
X_scaled.head(2)


X_tst = encode_catFeatures(ts_features)
X_test = pd.DataFrame(scala.fit_transform(X_tst))
X_test.head(2)


target = bank_tr['y']


X_scaled_train, X_scaled_val, target_train,  target_val = train_test_split(
    X_scaled, target, random_state=12, test_size=0.2
)


def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 16),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.01, 1.0)
        }
    model = xgb.XGBRegressor(**params, n_jobs=-1, random_state=12)
    model.fit(X_scaled_train, target_train)

    pred = model.predict(X_scaled_val)
    score = roc_auc_score(target_val, pred)
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)


best_params = study.best_params
xgb_model = xgb.XGBRegressor(**best_params, n_jobs=-1, random_state=12)
xgb_model.fit(X_scaled_train, target_train)

xgb_pred = xgb_model.predict(X_scaled_val)
auc_score = roc_auc_score(target_val, xgb_pred)
print(f"ROC_CURVE: {auc_score:.4f}")


prediction = xgb_model.predict(X_test)
submission = pd.DataFrame({'id': bank_ts['id'],
                           'y': prediction })
submission.head(3)


submission.to_csv("submission.csv", index=False) 

