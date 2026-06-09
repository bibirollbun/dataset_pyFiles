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
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)


# Load the datasets
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
    sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
except FileNotFoundError:
    print("Ensure train.csv, test.csv, and sample_submission.csv are in the same directory.")
    exit()

# Store the test IDs for the submission file
test_ids = test_df['id']

# Drop the 'id' column as it's not a feature
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

# Identify features and target variable
TARGET = 'accident_risk'
FEATURES = [col for col in train_df.columns if col != TARGET]

# Handle categorical features using Label Encoding
# This is efficient for tree-based models like LightGBM
categorical_features = train_df.select_dtypes(include=['object']).columns

for col in categorical_features:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col]) # Use transform for the test set


# Set up K-fold cross-validation
NUM_FOLDS = 5
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# Prepare arrays to store predictions
oof_preds = np.zeros(train_df.shape[0])
test_preds = np.zeros(test_df.shape[0])

# LightGBM model parameters
# These are a good starting point and can be further tuned
lgbm_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

# Iterate through each fold
for fold, (train_index, val_index) in enumerate(kf.split(train_df)):
    print(f"--- Fold {fold+1}/{NUM_FOLDS} ---")
    X_train, X_val = train_df.loc[train_index, FEATURES], train_df.loc[val_index, FEATURES]
    y_train, y_val = train_df.loc[train_index, TARGET], train_df.loc[val_index, TARGET]

    model = lgb.LGBMRegressor(**lgbm_params)
    
    # Train the model with early stopping
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(stopping_rounds=100)])
              
    # Predict on the validation and test sets
    oof_preds[val_index] = model.predict(X_val)
    test_preds += model.predict(test_df[FEATURES]) / NUM_FOLDS

# Clip predictions to the valid range [0, 1] as per competition rules
test_preds = np.clip(test_preds, 0, 1)


# Create the final submission DataFrame
submission_df = pd.DataFrame({
    'id': test_ids,
    'accident_risk': test_preds
})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' generated successfully!")


import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression # Meta-model for stacking

# Set seeds for reproducibility
np.random.seed(42)

# --- Data Loading and Preprocessing ---
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    exit()

test_ids = test_df['id']
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

TARGET = 'accident_risk'
FEATURES = [col for col in train_df.columns if col != TARGET]

categorical_features = train_df.select_dtypes(include=['object']).columns

for col in categorical_features:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

# --- Hyperparameter Tuning (using a hardcoded set for demonstration) ---
best_lgbm_params = {
    'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 2000,
    'learning_rate': 0.02, 'num_leaves': 64, 'max_depth': 8,
    'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 1,
    'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_child_samples': 40,
    'verbose': -1, 'n_jobs': -1, 'seed': 42
}

best_xgb_params = {
    'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'n_estimators': 2000,
    'learning_rate': 0.02, 'max_depth': 8, 'subsample': 0.7,
    'colsample_bytree': 0.7, 'min_child_weight': 1, 'lambda': 1,
    'alpha': 0.1, 'seed': 42
}

best_cat_params = {
    'iterations': 2000, 'learning_rate': 0.02, 'depth': 8,
    'l2_leaf_reg': 3, 'loss_function': 'RMSE', 'eval_metric': 'RMSE',
    'early_stopping_rounds': 100, 'verbose': 0, 'random_seed': 42
}

# --- Corrected Stacking Function and Execution ---
def get_oof_preds(model_class, params, X, y, X_test, folds):
    """
    Trains a model using K-fold cross-validation and returns 
    out-of-fold (OOF) predictions and test set predictions.
    """
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_index, val_index) in enumerate(folds.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Instantiate the model within the loop to get a fresh model for each fold
        model = model_class(**params)
        
        # Fit the model with early stopping based on model type
        if model_class == lgb.LGBMRegressor:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[lgb.early_stopping(stopping_rounds=100)])
        elif model_class == xgb.XGBRegressor:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
        elif model_class == cb.CatBoostRegressor:
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=0)
        
        oof_preds[val_index] = model.predict(X_val)
        test_preds += model.predict(X_test) / folds.n_splits
        
    return oof_preds, test_preds

# Prepare data for stacking
X_train_stack = train_df[FEATURES]
y_train_stack = train_df[TARGET]
X_test_stack = test_df[FEATURES]

NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Get OOF and test predictions for each base model
# Pass the class and parameters separately
lgbm_oof, lgbm_test = get_oof_preds(lgb.LGBMRegressor, best_lgbm_params, X_train_stack, y_train_stack, X_test_stack, folds)
xgb_oof, xgb_test = get_oof_preds(xgb.XGBRegressor, best_xgb_params, X_train_stack, y_train_stack, X_test_stack, folds)
cat_oof, cat_test = get_oof_preds(cb.CatBoostRegressor, best_cat_params, X_train_stack, y_train_stack, X_test_stack, folds)

# Create the meta-features for stacking
X_meta_train = np.column_stack((lgbm_oof, xgb_oof, cat_oof))
X_meta_test = np.column_stack((lgbm_test, xgb_test, cat_test))

# Train the meta-model
meta_model = LinearRegression()
meta_model.fit(X_meta_train, y_train_stack)

# Generate final predictions
final_predictions = meta_model.predict(X_meta_test)
final_predictions = np.clip(final_predictions, 0, 1)

print("Final predictions generated. Meta-model R²:", meta_model.score(X_meta_train, y_train_stack))

# --- Submission File Generation ---
submission_df = pd.DataFrame({
    'id': test_ids,
    'accident_risk': final_predictions
})

submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' with stacking ensemble generated successfully!")

