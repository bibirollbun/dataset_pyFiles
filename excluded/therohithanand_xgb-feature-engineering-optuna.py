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


# Import necessary libraries
import numpy as np
import pandas as pd
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer,mean_squared_log_error as msle 
from sklearn.model_selection import train_test_split,cross_val_score


# Read datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


print(train.head())
print(train.info())
print(train.describe())


# Feature Engineering 
def feature_engineering(data):
    # Body Mass Index
    data['BMI'] = data['Weight']/(data['Height']/100)**2

    # Body Surface Area (DuBois formula)
    data['BSA'] = 0.007184 * (data['Height'])**0.725 * data['Weight']**0.425

    # Fat Free Mass
    data['FFM'] = np.where(data['Sex'] == 'male',
                           data['Weight'] * 0.80,
                           data['Weight'] * 0.70)

    # Body Fat Mass
    data['Fat_Mass'] = data['Weight'] - data['FFM']

    # Body Fat Percentage
    data['Body_Fat_Percentage'] = (data['Fat_Mass'] / data['Weight']) * 100

    # Intensity of exercise
    X['Intensity'] = X['Heart_Rate'] / X['Duration']

    # Age group category
    X['Age_Group'] = pd.cut(X['Age'], 
                              bins=[19, 30, 40, 50, 65, 80], 
                              labels=[0, 1, 2, 3, 4])

    # Gender to integer type
    data['Sex'] = data['Sex'].map({'male' : 1, 'female' : 0})
    
    return data

# Performing feature engineering on Train and Test Data
train = feature_engineering(train)
test = feature_engineering(test)


# Drop unwanted columns

X = train.drop(columns=['id','Calories'])
y = train['Calories']

ids = test['id']
test = test.drop(columns=['id'])


# Split Dataset 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# Define the custom RMSLE scorer
def rmsle(y_true, y_pred):
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


# Using Optuna to find best values of hyperparameters
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }

    model = XGBRegressor(
        **params,
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        verbosity=0
    )

    score = cross_val_score(
        model,
        X_train,
        y_train,
        scoring=rmsle_scorer,
        cv=3,
        n_jobs=-1
    )

    return -score.mean()

# Create and run the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, show_progress_bar=True)
print("Best parameters:", study.best_params)


# Best parameters after Optuna
best_params = study.best_params

# Create the model
model = XGBRegressor(**best_params)

# Fit the model
model.fit(X_train, y_train)


# Make predictions

y_pred = model.predict(X_test)
y_pred = np.maximum(0, preds)
error = rmsle(y_test,y_pred)
print("RMSLE: ",error)


# Prediction on test dataset

preds = model.predict(test)

ans = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
ans['id'] = ids
ans['Calories'] = preds
ans.to_csv('submission.csv',index=False)
print("Predictions saved successfully!")

