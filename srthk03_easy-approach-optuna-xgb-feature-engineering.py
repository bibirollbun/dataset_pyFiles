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


df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


df['Sex'] = df['Sex'].map({'male' : 1, 'female' : 0})


df.head()


df['BMI'] = df['Weight']/((df['Height']/100)**2)
df['WxDuration'] = df['Weight']*df['Duration']
# df['AxHR'] = df['Age']*df['Heart_Rate']
# df['AxBT'] = df['Age']*df['Body_Temp']
df['WxBT'] = df['Weight']*df['Body_Temp']
df.head()


import seaborn as sns
df.corr()
sns.heatmap(df.corr(), annot=True)


X = df.drop(['id', 'Height', 'Weight', 'Calories'], axis = 1)
y = df['Calories']
X.head()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error as msle 


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 3.0),
        'random_state': 42,
        'tree_method': 'hist',
        'device': 'cuda',
        'n_jobs': -1
        
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds = np.maximum(0, preds)

    rmsle = np.sqrt(msle(y_test, preds))
    return rmsle


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials = 50)

print("Best RMSLE", study.best_value)
print("Best params", study.best_params)


best_params = study.best_params
final_model = XGBRegressor(**best_params)
final_model.fit(X_train, y_train)


y_pred = final_model.predict(X_test)


from sklearn.metrics import mean_squared_log_error as msle 

rmsle = np.sqrt(msle(y_test, y_pred))
print(rmsle)


test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

test['BMI'] = test['Weight'] / ((test['Height'] /100)**2)
test['WxDuration'] = test['Weight']*test['Duration']
test['WxBT'] = test['Weight']*test['Body_Temp']

test = test.drop(['id', 'Height', 'Weight'], axis = 1)
test['Sex'] = test['Sex'].map({'male' : 1, 'female' : 0})
test.head()


submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")



y_pred =final_model.predict(test)
y_pred = np.maximum(0,y_pred)


submission['Calories'] = y_pred
submission.to_csv("submission1.csv", index=False)
print("\nâœ… Submission file saved.")




