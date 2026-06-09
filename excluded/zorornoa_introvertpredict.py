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


df1 = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df2 = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


print(df1.columns)
print(df2.columns)


print(df1.isna().sum())
print(df2.isna().sum())


%pip install optuna
%pip install xgboost

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier
import optuna


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


binary_cols = ['Stage_fear', 'Going_outside', 'Drained_after_socializing']

for col in binary_cols:
    train[col] = train[col].map({'Yes': 1, 'No': 0})
    test[col] = test[col].map({'Yes': 1, 'No': 0})


features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
            'Going_outside', 'Drained_after_socializing', 
            'Friends_circle_size', 'Post_frequency']


train_null_cols = train[features].isnull().all()
test_null_cols = test[features].isnull().all()


drop_cols = train_null_cols[train_null_cols].index.tolist() + test_null_cols[test_null_cols].index.tolist()
drop_cols = list(set(drop_cols))  # remove duplicates
print("Dropping columns with all missing values:", drop_cols)

features = [f for f in features if f not in drop_cols]



imputer = SimpleImputer(strategy='mean')

train_imputed = pd.DataFrame(imputer.fit_transform(train[features]), columns=features)
test_imputed = pd.DataFrame(imputer.transform(test[features]), columns=features)


train[features] = train_imputed
test[features] = test_imputed



le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])


scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])


X = train[features]
y = train['Personality']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    }
    
    model = XGBClassifier(**params)
    score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
    return score


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)



best_params = study.best_params
print("Best Hyperparameters:", best_params)

best_model = XGBClassifier(**best_params)
best_model.fit(X_train, y_train)


y_pred = best_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred, target_names=le.classes_))



test_preds = best_model.predict(test[features])
test['Personality'] = le.inverse_transform(test_preds)


submission = test[['id', 'Personality']]


submission.to_csv('submission.csv', index=False)

