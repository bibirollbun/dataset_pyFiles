# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, make_scorer


import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv').set_index('id')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv').set_index('id')


# Print Distribution Target
target = 'Calories'
sns.displot(train[target], bins=40)


for i in train.drop(columns='Calories', axis=1).columns:
    sns.displot(train[i], bins=40)


# Preprocessing
train.drop_duplicates(inplace=True)
train['BMI'] = train['Weight'] / (train['Height'] ** 2)
test['BMI'] = test['Weight'] / (test['Height'] ** 2)

train['BMR'] = 10 * train['Weight'] + 6.25 * train['Height'] - 5 * train['Age'] + np.where(train['Sex'] == 1, 5, -161)
test['BMR'] = 10 * test['Weight'] + 6.25 * test['Height'] - 5 * test['Age'] + np.where(test['Sex'] == 1, 5, -161)

train['Activity_Level'] = train['Heart_Rate'] * train['Duration']
test['Activity_Level'] = test['Heart_Rate'] * test['Duration']


# Feature separation
X = train.drop(columns='Calories')
y = np.log1p(train['Calories'])


# Feature Columns
cat_cols = [col for col in X.columns if X[col].dtype == 'object']
num_cols = [col for col in X.columns if X[col].dtype != 'object']


# Create Pipeline
numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

category_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', category_transformer, cat_cols)
])


# Define Optuna objective function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'verbosity': 0,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'gpu_id': 0,
    }

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(**params))
    ])

    score = cross_val_score(model, X, y, scoring=make_scorer(mean_squared_error, greater_is_better=False), cv=5)
    return -np.mean(score)  # minimize RMSE


# Run Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)


# Best params and model training
print("Best trial:")
print(f"  Score (neg RMSE): {study.best_trial.value}")
print("  Params:")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")



# Train final model on full training set
final_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(**study.best_trial.params))
])


final_model.fit(X, y)

y_pred = final_model.predict(test)
y_pred = np.expm1(y_pred)  # undo log1p transform


submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv').set_index('id')
submission['Calories'] = y_pred
submission.to_csv('./submission.csv')

