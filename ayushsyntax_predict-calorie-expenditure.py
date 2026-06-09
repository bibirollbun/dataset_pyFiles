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
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_log_error, make_scorer



# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


# Add BMI and interaction features
if 'Weight' in train.columns and 'Height' in train.columns:
    train['BMI'] = train['Weight'] / (train['Height'] ** 2)
    test['BMI'] = test['Weight'] / (test['Height'] ** 2)

if 'Duration' in train.columns and 'HeartRate' in train.columns:
    train['Duration_HR'] = train['Duration'] * train['HeartRate']
    test['Duration_HR'] = test['Duration'] * test['HeartRate']

# Prepare features and target
X = train.drop(['id', 'Calories'], axis=1)
y = train['Calories']
test_ids = test['id']
X_test = test.drop('id', axis=1)

# One-hot encode and align
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
X, X_test = X.align(X_test, join='inner', axis=1)


# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Log-transform target to avoid negative predictions
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

# Train
lr = LinearRegression()
lr.fit(X_train_scaled, y_train_log)

# Predict & invert log transform
preds_log = lr.predict(X_val_scaled)
preds = np.expm1(preds_log)
preds = np.clip(preds, a_min=0, a_max=None)  # Safety clip

# Evaluate
rmsle = np.sqrt(mean_squared_log_error(y_val, preds))
print(f"Linear Regression RMSLE: {rmsle:.4f}")


# Generate polynomial features
poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_val_poly = poly.transform(X_val_scaled)
X_test_poly = poly.transform(X_test_scaled)


# Custom RMSLE scorer for hyperparameter tuning
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


# Define parameter grid
param_dist = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

# Initialize XGBoost with GPU
xgb = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmsle',
    tree_method='gpu_hist',        # GPU acceleration
    predictor='gpu_predictor',     # GPU inference
    n_estimators=200,
    random_state=42
)

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=30,                     # More iterations for better tuning
    scoring=rmsle_scorer,          # Custom RMSLE scorer
    cv=5,
    verbose=1,
    random_state=42
)

# Log-transform target for safe predictions
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

random_search.fit(X_train_poly, y_train_log)
best_params = random_search.best_params_
best_score = -random_search.best_score_  # Convert back to positive RMSLE
print(f"Best RMSLE from RandomizedSearchCV: {best_score:.4f}")


# Train with early stopping
xgb_final = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmsle',
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    early_stopping_rounds=10,
    n_estimators=1000,
    **best_params,
    random_state=42
)

xgb_final.fit(
    X_train_poly, y_train_log,
    eval_set=[(X_val_poly, y_val_log)],
    verbose=False
)

# Predict & invert log transform
preds_xgb_log = xgb_final.predict(X_val_poly)
preds_xgb = np.expm1(preds_xgb_log)
preds_xgb = np.clip(preds_xgb, a_min=0, a_max=None)

# Evaluate
rmsle_xgb = np.sqrt(mean_squared_log_error(y_val, preds_xgb))
print(f"XGBoost RMSLE: {rmsle_xgb:.4f}")


from sklearn.model_selection import train_test_split

# Split full dataset into train + validation (for early stopping)
X_full_poly_train, X_full_poly_val, y_full_train, y_full_val = train_test_split(
    X_full_poly, y_log, test_size=0.1, random_state=42
)

# Final model with GPU and early stopping
xgb_final = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmsle',
    tree_method='hist',          # Updated for XGBoost ≥2.0
    device='cuda',               # GPU acceleration
    early_stopping_rounds=10,
    n_estimators=1000,
    **best_params,
    random_state=42
)

# Train with early stopping
xgb_final.fit(
    X_full_poly_train, y_full_train,
    eval_set=[(X_full_poly_val, y_full_val)],
    verbose=False
)


# Predict & invert log transform
test_preds_log = xgb_final.predict(X_test_poly)
test_preds = np.expm1(test_preds_log)
test_preds = np.clip(test_preds, a_min=0, a_max=None)

# Save submission
submission = pd.DataFrame({'id': test_ids, 'Calories': test_preds})
submission.to_csv('submission.csv', index=False)


from IPython.display import FileLink
print("Click on the link below to download your submission file:")
display(FileLink('submission.csv'))




