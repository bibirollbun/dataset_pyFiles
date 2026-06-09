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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from functools import partial
import optuna
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")



df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv').set_index('id')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv').set_index('id')


df.info()


num_cols = df.select_dtypes(include='number').columns
cat_cols = df.drop(columns='Personality', axis=1).select_dtypes(include='object').columns


for i in num_cols:
    sns.displot(df[i], kde=True)
    plt.show()


for i in cat_cols:
    sns.displot(df[i])
    plt.show()


sns.displot(df['Personality'])


# Feature Engineering
def feature_engineer(df):
    df['Extrovert_score'] = df['Social_event_attendance'] + df['Going_outside'] + df['Friends_circle_size'] + df['Post_frequency']
    df['Introvert_score'] = df['Time_spent_Alone'] - df['Extrovert_score']
    df['Inp']=df['Introvert_score']-df['Post_frequency']
    df['set']=df['Social_event_attendance']-df['Time_spent_Alone']
    return df


df[cat_cols] = df[cat_cols].fillna('missing')
df[num_cols] = df[num_cols].fillna(-1)


df = feature_engineer(df)
print(df.head())


X = df.drop(columns=['Personality'], axis=1)
y = df['Personality']


le = LabelEncoder()
scaler = MinMaxScaler()
ohe = OneHotEncoder()


y = le.fit_transform(y)


X.info()


def objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'n_jobs': -1,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    num_cols = X.select_dtypes(exclude='object').columns
    cat_cols = X.select_dtypes(include='object').columns

    preprocess = ColumnTransformer([
        ('cat', ohe, cat_cols),
        ('num', scaler, num_cols)
    ])

    model = Pipeline([
        ('process', preprocess),
        ('rf', XGBClassifier(
            **params,
            )
        )
    ])

    # Cross-validation score (you can use accuracy, f1, etc.)
    fold = StratifiedKFold(n_splits=5, shuffle=True)
    score = cross_val_score(model, X, y, cv=fold, scoring='accuracy')
    return score.mean()


objective_func = partial(objective, X=X, y=y)
study = optuna.create_study(direction='maximize')
study.optimize(objective_func, n_trials=50)


print("\nBest Parameters:", study.best_trial.params)
print("Best Cross-Validated Accuracy:", f"{study.best_value:.2%}")


preprocess = ColumnTransformer([
    ('cat', ohe, cat_cols),
    ('num', scaler, num_cols)
])


train_pipe = Pipeline([
    ('process', preprocess),
    ('model', XGBClassifier(**study.best_params))
]).fit(X, y)


test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
test[cat_cols] = test[cat_cols].fillna('missing')
test[num_cols] = test[num_cols].fillna(-1)
test = feature_engineer(test)


final_pred = train_pipe.predict(test)
final_pred = le.inverse_transform(final_pred)
sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv').set_index('id')
sub['Personality'] = final_pred
sub.to_csv('submission.csv')




