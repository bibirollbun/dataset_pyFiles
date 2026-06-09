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


df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")


df_train.columns


df_train['Stage_fear'].value_counts()


df_train['Drained_after_socializing'].value_counts()


df_train['Personality'].value_counts()


df_train.describe()


X_train = df_train[['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
       'Post_frequency']].reset_index()
X_train['Stage_fear'] = np.where(df_train['Stage_fear']=='Yes', 1, 0)
X_train['Drained_after_socializing'] = np.where(df_train['Drained_after_socializing']=='Yes', 1, 0)
y_train = np.where(df_train['Personality']=="Introvert", 1, 0)


y_train.mean()


from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import optuna
from optuna.pruners import HyperbandPruner
import numpy as np

def objective_cv(trial):
    param_dict = {
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
        'gamma': trial.suggest_float('gamma', 0.1, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for (train_idx, val_idx) in cv.split(X_train, y_train):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train[train_idx], y_train[val_idx]

        model = XGBClassifier(**param_dict)
        model.fit(X_t, y_t)

        # y_train_pred = model.predict_proba(X_t)[:, 1]
        # roc_auc_train = roc_auc_score(y_t, y_train_pred)
        # print(roc_auc_train)
        
        y_pred = model.predict_proba(X_v)[:, 1]
        roc_auc = roc_auc_score(y_v, y_pred)
        cv_scores.append(roc_auc)

    return np.mean(cv_scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective_cv, n_trials=100)


print('Best trial:')
trial = study.best_trial
print(' Value: ', trial.value)
print(' Params: ')
for key, value in trial.params.items():
    print(f'  {key}: {value}')


final_model = XGBClassifier(**trial.params, random_state=42)
final_model.fit(X_train, y_train)


y_pred = final_model.predict_proba(X_train)[:, 1]


fpr, tpr, thresholds = roc_curve(y_train, y_pred)


plt.plot(fpr, tpr)


threshold = np.arange(0, 1, 1e-3)
accuracy = [accuracy_score(y_train, np.where(y_pred>t, 1, 0)) for t in threshold]
plt.plot(threshold, accuracy)


df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


X_test = df_test[['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
       'Post_frequency']].reset_index()
X_test['Stage_fear'] = np.where(df_test['Stage_fear']=='Yes', 1, 0)
X_test['Drained_after_socializing'] = np.where(df_test['Drained_after_socializing']=='Yes', 1, 0)


y_test_pred = final_model.predict(X_test)


y_test_pred


df_submission = pd.DataFrame({'id':df_test['id'], 'Personality':np.where(y_test_pred==1, "Introvert", "Extrovert")}).set_index('id')


df_submission.to_csv("/kaggle/working/submission1.csv")


df_submission




