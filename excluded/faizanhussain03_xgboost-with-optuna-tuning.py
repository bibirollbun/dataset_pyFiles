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


train=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv').set_index('id')
test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv').set_index('id')
sample_submission=pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


train.y.value_counts()


import matplotlib.pyplot as plt
import seaborn as sns

import optuna
import warnings

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings('ignore')


# lets see how many missing values in data, types of columns in Data
combined=pd.concat([train.drop(columns=['y']),test],axis=0)
print(f"There are total {combined.isnull().sum().sum()} missing values")

# numerical columns and categorical columns

cat_col=[]
num_col=[]

for col in train.drop(columns=['y']).columns:
    if(train[col].dtypes!='object'):
        num_col.append(col)
    else:    
        cat_col.append(col)
print("cat_col:",cat_col) 
print("num_col:",num_col) 
        


X=train.drop(columns=['y'])
y=train.y

X_test=test

for col in X[cat_col]:
    le = LabelEncoder()
    le.fit(list(X[col].astype(str)) + list(X_test[col].astype(str)))
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))



def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 10000, step=100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True)
    }
    
    model = XGBClassifier(**params, objective='binary:logistic', random_state=42, device='cuda', n_jobs=-1,
                          enable_categorical=True, tree_method='hist')
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    return cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()


study = optuna.create_study(direction='maximize',study_name="xgb_roc",storage="sqlite:///xgb_roc.db",load_if_exists=True)
study.optimize(objective, n_trials=50, timeout=5400, show_progress_bar=True)
best_params = study.best_trial.params
print('Best Parameters:', best_params)
print('Best Trial:', study.best_trial)


!ls


!rm xgb_roc.db




