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


train_data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col=False)
test_data = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col=False)


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


X = train_data.drop(["Listening_Time_minutes","id"],axis=1).copy()
y = train_data["Listening_Time_minutes"].copy()

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder()),
    ('scaler', StandardScaler())
])

# Full preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Apply transformation
X_transformed = preprocessor.fit_transform(X)


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import cupy as cp
import cudf



target = y.astype(float)  # Ensure target is numerical

xgb_regressor = XGBRegressor()


X_train, X_test, y_train, y_test = train_test_split(X_transformed, target, test_size=0.2, random_state=42)

xgb_regressor.fit(X_train, y_train)
y_pred = xgb_regressor.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"scored: {rmse}")


import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Example data split (replace X, y with your data)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Objective function for Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42,
        'objective': 'reg:squarederror',
        'tree_method': 'auto'
    }

    model = XGBRegressor(**params, device="cuda")
    # Use cross-validation to evaluate
    score = cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
    return -score.mean()
"""
# Create study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, timeout=600)  # adjust n_trials or timeout as needed

# Print best results
print("Best trial:")
print("  Value (RMSE):", study.best_value)
print("  Params:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

# Train best model on full training data
best_model = XGBRegressor(**study.best_params)
best_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = best_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE:", test_rmse)
"""


best_params = {
    "n_estimators": 428,
    "learning_rate": 0.014335242549004461,
    "max_depth": 15,
    "subsample": 0.9833839501959647,
    "colsample_bytree": 0.7635524031080269,
    "gamma": 3.567510172896715,
    "reg_alpha": 1.5380128462052438,
    "reg_lambda": 4.270056375391337,
}

best_model = XGBRegressor(**best_params)


import xgboost as xgb

submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv", index_col=None)
X_test_transformed = preprocessor.transform(test_data.drop("id",axis=1))

best_model.fit(X_transformed, y)
preds_xgb = best_model.predict(X_test_transformed)
preds_pub = pd.read_csv('/kaggle/input/12-636-xgboost-bayes/ensemble.csv')['Listening_Time_minutes'].values


submission["Listening_Time_minutes"] = 0.3 * preds_xgb + 0.7 * preds_pub
submission.to_csv("submission.csv", index = False)


