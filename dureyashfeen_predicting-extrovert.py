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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train.head(10)


# ğŸ”� EDA
print("Shape of training data:", train.shape)
print("Missing values:\n", train.isnull().sum())


# Separate target and ID
X = train.drop(columns=['id', 'Personality'])
y = train['Personality']
X_test = test.drop(columns=['id'])

# Identify columns
cat_cols = ['Stage_fear', 'Drained_after_socializing']
num_cols = [col for col in X.columns if col not in cat_cols]

# Imputation
for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0])
    X_test[col] = X_test[col].fillna(X[col].mode()[0])

for col in num_cols:
    X[col] = X[col].fillna(X[col].median())
    X_test[col] = X_test[col].fillna(X[col].median())

# Encode categoricals
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])

# Encode target (Introvert/Extrovert â†’ 0/1)
y = le.fit_transform(y)


def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'accuracy',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500)
    }
    model = lgb.LGBMClassifier(**params)
    return cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
best_params_lgb = study.best_params


model_lgb = lgb.LGBMClassifier(**best_params_lgb)
model_xgb = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
model_cb = cb.CatBoostClassifier(verbose=0)
model_rf = RandomForestClassifier(n_estimators=150, random_state=42)
model_et = ExtraTreesClassifier(n_estimators=150, random_state=42)


stack_model = StackingClassifier(
    estimators=[
        ('lgb', model_lgb),
        ('xgb', model_xgb),
        ('cb', model_cb),
        ('rf', model_rf),
        ('et', model_et),
    ],
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

stack_model.fit(X, y)


cv_scores = cross_val_score(stack_model, X, y, cv=5, scoring='accuracy')
print("âœ… Stacked Model CV Accuracy:", round(cv_scores.mean(), 4))


preds = stack_model.predict(X_test)
submission['Personality'] = le.inverse_transform(preds)
submission.to_csv('submission.csv', index=False)




