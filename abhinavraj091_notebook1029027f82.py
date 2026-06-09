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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import optuna
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

# Feature engineering with PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = pd.DataFrame(poly.fit_transform(X), columns=poly.get_feature_names_out(X.columns))
X_test_poly = pd.DataFrame(poly.transform(X_test), columns=poly.get_feature_names_out(X.columns))

# Log-transform the target if it's highly skewed
if y.skew() > 1 or y.skew() < -1:
    y = np.log1p(y)

# Initialize KFold for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# XGBoost objective for Optuna
def xgb_objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 600, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 3),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
    model = XGBRegressor(**xgb_params, random_state=42, n_jobs=-1, objective='reg:squarederror')
    scores = []
    for train_index, val_index in kf.split(X_poly, y):
        X_train, X_val = X_poly.iloc[train_index], X_poly.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        model.fit(X_train, y_train)
        scores.append(model.score(X_val, y_val))
    return np.mean(scores)

# Optuna optimization for XGBoost
xgb_study = optuna.create_study(direction='maximize')
xgb_study.optimize(xgb_objective, n_trials=50)
best_xgb_params = xgb_study.best_params

# Train the final XGBoost model with best parameters
final_xgb_model = XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1, objective='reg:squarederror')
final_xgb_model.fit(X_poly, y)

# Make predictions on the test set
predictions = final_xgb_model.predict(X_test_poly)

# Reverse the log transformation if applied
if y.skew() > 1 or y.skew() < -1:
    predictions = np.expm1(predictions)

# Save predictions to a CSV file
submission = pd.DataFrame({'id': test_data['id'], 'target': predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")


