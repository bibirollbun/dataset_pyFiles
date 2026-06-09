# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/playground-series-s5e2'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import optuna

# Set random seed for reproducibility
np.random.seed(42)


# Load datasets
print("Loading datasets...")
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

# Combine train and train_extra to create a larger training set
print("Combining training datasets...")
train_combined = pd.concat([train, train_extra], ignore_index=True)


# Separate features and target
X = train_combined.drop(columns=['id', 'Price'])
y = train_combined['Price']
X_test = test.drop(columns=['id'])

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
print(f"Categorical columns: {cat_cols}")
print(f"Numerical columns: {num_cols}")

# Handle missing values
print("Handling missing values...")
for col in num_cols:
    X[col].fillna(X[col].median(), inplace=True)
    X_test[col].fillna(X[col].median(), inplace=True)

for col in cat_cols:
    X[col].fillna('missing', inplace=True)
    X_test[col].fillna('missing', inplace=True)


# Encode categorical features for XGBoost
print("Encoding categorical features for XGBoost...")
label_encoders = {}
X_encoded = X.copy()
X_test_encoded = X_test.copy()
for col in cat_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    # Handle unseen categories in test set by mapping them to 'missing'
    X_test_encoded[col] = X_test[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else le.transform(['missing'])[0])
    label_encoders[col] = le

# Feature engineering (example: Capacity per Weight)
print("Performing feature engineering...")
if 'Capacity' in X.columns and 'Weight' in X.columns:
    X_encoded['Capacity_per_Weight'] = X_encoded['Capacity'] / (X_encoded['Weight'] + 1e-6)
    X_test_encoded['Capacity_per_Weight'] = X_test_encoded['Capacity'] / (X_test_encoded['Weight'] + 1e-6)
    X['Capacity_per_Weight'] = X['Capacity'] / (X['Weight'] + 1e-6)
    X_test['Capacity_per_Weight'] = X_test['Capacity'] / (X_test['Weight'] + 1e-6)
    print("Added feature: Capacity_per_Weight")


# Split data into training and validation sets
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
X_train_cb, X_val_cb, y_train_cb, y_val_cb = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare data for XGBoost
print("Preparing data for XGBoost...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test_encoded)

# Prepare data for CatBoost
print("Preparing data for CatBoost...")
train_pool = Pool(X_train_cb, y_train_cb, cat_features=cat_cols)
val_pool = Pool(X_val_cb, y_val_cb, cat_features=cat_cols)
test_pool = Pool(X_test, cat_features=cat_cols)


# Hyperparameter tuning for XGBoost using Optuna
print("Tuning XGBoost hyperparameters...")
def xgb_objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': trial.suggest_float('eta', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True)
    }
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dval, 'val')],
                      early_stopping_rounds=10, verbose_eval=False)
    preds = model.predict(dval)
    rmse = mean_squared_error(y_val, preds, squared=False)
    return rmse


study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(xgb_objective, n_trials=25)
best_xgb_params = study_xgb.best_params
print(f"Best XGBoost params: {best_xgb_params}")

# Train final XGBoost model
print("Training final XGBoost model...")
model_xgb = xgb.train(best_xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')],
                      early_stopping_rounds=10, verbose_eval=100)



# Hyperparameter tuning for CatBoost using Optuna
print("Tuning CatBoost hyperparameters...")
def cb_objective(trial):
    params = {
        'iterations': 50,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42
    }
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=10, verbose=0)
    preds = model.predict(val_pool)
    rmse = mean_squared_error(y_val, preds, squared=False)
    return rmse

study_cb = optuna.create_study(direction='minimize')
study_cb.optimize(cb_objective, n_trials=25)
best_cb_params = study_cb.best_params
print(f"Best CatBoost params: {best_cb_params}")


# Train final CatBoost model
print("Training final CatBoost model...")
model_cb = CatBoostRegressor(**best_cb_params)
model_cb.fit(train_pool, eval_set=val_pool, early_stopping_rounds=10, verbose=25)

# Predictions on validation set
print("Evaluating models on validation set...")
preds_xgb_val = model_xgb.predict(dval)
preds_cb_val = model_cb.predict(val_pool)

# RMSE for individual models
rmse_xgb = mean_squared_error(y_val, preds_xgb_val, squared=False)
rmse_cb = mean_squared_error(y_val, preds_cb_val, squared=False)
print(f"XGBoost RMSE: {rmse_xgb}")
print(f"CatBoost RMSE: {rmse_cb}")

# Ensemble predictions (simple averaging)
print("Creating ensemble predictions...")
preds_ensemble_val = (preds_xgb_val + preds_cb_val) / 2
rmse_ensemble = mean_squared_error(y_val, preds_ensemble_val, squared=False)
print(f"Ensemble RMSE: {rmse_ensemble}")


# Test set predictions
print("Generating test set predictions...")
preds_xgb_test = model_xgb.predict(dtest)
preds_cb_test = model_cb.predict(test_pool)
preds_test = (preds_xgb_test + preds_cb_test) / 2  # Ensemble

# Create submission file
print("Creating submission file...")
submission = pd.DataFrame({
    'id': test['id'],
    'Price': preds_test
})
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")







