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
import seaborn as sns
from sklearn.model_selection import train_test_split ,KFold , GridSearchCV
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor


df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


print("Missing values in training data:")
print(df.isnull().sum())
print("\nMissing values in test data:")
print(df_test.isnull().sum())

# Handle missing values (example: fill with median for numerical columns)
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].fillna(df[col].median())

for col in df_test.columns:
    if df_test[col].dtype in ['int64', 'float64']:
        df_test[col] = df_test[col].fillna(df_test[col].median())

print("\nMissing values after imputation in training data:")
print(df.isnull().sum())
print("\nMissing values after imputation in test data:")
print(df_test.isnull().sum())


# Identify categorical features (assuming none based on describe output, but good practice to check)
categorical_features = [col for col in df.columns if df[col].dtype == 'object']
print(f"Categorical features in training data: {categorical_features}")

categorical_features_test = [col for col in df_test.columns if df_test[col].dtype == 'object']
print(f"Categorical features in test data: {categorical_features_test}")

# Separate target variable
X = df.drop(['BeatsPerMinute', 'id'], axis=1)
y = df['BeatsPerMinute']

# Store test ids for submission
test_ids = df_test['id']
X_test = df_test.drop('id', axis=1)


# Split training data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining data shapes:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}")


# Due to the large dataset, we'll use a smaller subset for demonstration
X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, test_size=0.9, random_state=42)

# Define parameter grid for LightGBM (example)
lgbm_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 63]
}

# Instantiate GridSearchCV for LightGBM with GPU
lgbm_grid_search = GridSearchCV(LGBMRegressor(random_state=42), lgbm_param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)

# Fit GridSearchCV for LightGBM
print("\nRunning GridSearchCV for LightGBM...")
lgbm_grid_search.fit(X_train_subset, y_train_subset)

print("\nBest parameters for LightGBM:", lgbm_grid_search.best_params_)
print("Best RMSE for LightGBM:", -lgbm_grid_search.best_score_)


# Define the number of folds
n_splits = 10

# Instantiate KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Lists to store RMSE scores for each fold
lgbm_rmse_scores = []

print(f"Performing {n_splits}-fold cross-validation...")

# Cross-validation for LightGBM
print("\nLightGBM Cross-Validation:")
for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"  Fold {fold+1}/{n_splits}")
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    model = LGBMRegressor(random_state=42, **lgbm_grid_search.best_params_)
    model.fit(X_train_fold, y_train_fold)
    preds = model.predict(X_val_fold)
    rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
    lgbm_rmse_scores.append(rmse)
    print(f"    Fold {fold+1} RMSE: {rmse}")

print(f"\nAverage LightGBM RMSE: {np.mean(lgbm_rmse_scores)}")
print(f"LightGBM RMSE scores: {lgbm_rmse_scores}")


# Retrain LightGBM on the full training data
final_lgbm_model = LGBMRegressor(random_state=42, **lgbm_grid_search.best_params_)
final_lgbm_model.fit(X, y)

# Generate predictions on the test set using the final LightGBM model
lgbm_test_predictions_cv = final_lgbm_model.predict(X_test)

# Create submission DataFrame for LightGBM (from CV)
lgbm_submission_df_cv = pd.DataFrame({'id': test_ids, 'BeatsPerMinute': lgbm_test_predictions_cv})

# Save the LightGBM submission file (from CV)
lgbm_submission_df_cv.to_csv('submission.csv', index=False)

print("LightGBM submission file (from CV) created successfully: submission_lgbm_cv.csv")

