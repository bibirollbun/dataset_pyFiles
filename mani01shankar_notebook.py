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
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# --- 1. Load Data using Kaggle File Paths ---
# Use the correct paths for the Kaggle environment
try:
    print("Loading data from Kaggle paths...")
    train_path = '/kaggle/input/playground-series-s5e9/train.csv'
    test_path = '/kaggle/input/playground-series-s5e9/test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: Make sure the dataset is correctly attached to your Kaggle notebook.")
    print(f"Attempted to load from: {train_path} and {test_path}")
    exit()

# --- 2. Prepare Data for Training ---
# Features are all columns except 'id' and the target 'BeatsPerMinute'
features = [col for col in train_df.columns if col not in ['id', 'BeatsPerMinute']]
X = train_df[features]
y = train_df['BeatsPerMinute']

# The test set has the same features
X_test = test_df[features]

# --- 3. Create a Validation Set ---
# This is crucial for early stopping to prevent the model from overfitting.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Test data shape: {X_test.shape}")

# --- 4. Initialize and Train XGBoost Model ---
# We use a set of optimized hyperparameters for better performance.
print("\nTraining the XGBoost model...")
xgb_regressor = xgb.XGBRegressor(
    objective='reg:squarederror', # Explicitly set the objective for regression
    n_estimators=2000,            # High number of trees, will be optimized by early stopping
    learning_rate=0.01,           # Low learning rate is more robust
    max_depth=7,                  # Depth of each tree
    subsample=0.8,                # Fraction of samples used per tree
    colsample_bytree=0.8,         # Fraction of features used per tree
    random_state=42,
    n_jobs=-1,                    # Use all available CPU cores
    tree_method='hist'            # Use the fast histogram-based algorithm
)

# Train the model with early stopping.
# It will monitor the RMSE on the validation set and stop if it doesn't improve for 50 rounds.
xgb_regressor.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=100  # Prints evaluation results every 100 rounds
)

# --- 5. Evaluate on Validation Set ---
# Check the final RMSE on the held-out validation data
val_preds = xgb_regressor.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"\nFinal Validation RMSE: {rmse:.4f}")

# --- 6. Make Predictions and Create Submission File ---
print("\nMaking predictions on the test data...")
predictions = xgb_regressor.predict(X_test)
print("Predictions are ready.")

# Create the submission file in the specified format
submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': predictions})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print("Here's a preview of the submission file:")
print(submission_df.head())

