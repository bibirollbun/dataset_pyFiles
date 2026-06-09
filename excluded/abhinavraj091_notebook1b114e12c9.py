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
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.impute import SimpleImputer
import optuna
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Load the data
train_file_path = '/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/train.csv'
test_file_path = '/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/test.csv'
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Separate features and target
X = train_data.drop(columns=["target"])
y = train_data["target"]
X_test = test_data.drop(columns=["id"])

# Imputation for missing values
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Feature Engineering: Creating a sum of features
X['sum_features'] = X.sum(axis=1)
X_test['sum_features'] = X_test.sum(axis=1)

# Log transformation of target variable if skewed
if y.skew() > 1 or y.skew() < -1:
    y = np.log1p(y)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Objective function for Optuna (XGBoost)
def xgb_objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 2),
    }

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('transformer', PowerTransformer()),
        ('model', XGBRegressor(**xgb_params, random_state=42, n_jobs=-1, objective='reg:squarederror'))
    ])

    cv_scores = []
    for train_index, val_index in kf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        cv_scores.append(mean_squared_error(y_val, y_pred))
    return np.mean(cv_scores)

# Optimize XGBoost
study = optuna.create_study(direction='minimize')  # Minimize MSE
study.optimize(xgb_objective, n_trials=50)
best_xgb_params = study.best_params

# Final Model: Stacking Regressor with XGB, LGBM, and CatBoost
xgb_model = XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1)
lgbm_model = LGBMRegressor()
cat_model = CatBoostRegressor(learning_rate=0.05, iterations=1000, depth=7, random_state=42, verbose=0)

estimators = [
    ('xgb', xgb_model),
    ('lgbm', lgbm_model),
    ('cat', cat_model)
]
stack_model = StackingRegressor(estimators=estimators, final_estimator=XGBRegressor())

# Cross-validation for the stacked model
cv_scores = []
for train_index, val_index in kf.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    stack_model.fit(X_train, y_train)
    y_pred = stack_model.predict(X_val)
    cv_scores.append(mean_squared_error(y_val, y_pred))

print(f'Mean CV MSE: {np.mean(cv_scores)}')

# Final model fitting on the whole dataset
stack_model.fit(X, y)

# Predict on test set
predictions = stack_model.predict(X_test)

# Reverse the log transformation if applied
if y.skew() > 1 or y.skew() < -1:
    predictions = np.expm1(predictions)

# Save submission file
submission = pd.DataFrame({'id': test_data['id'], 'target': predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")


