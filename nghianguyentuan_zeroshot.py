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


# =============================================================================
# 1. SETUP & IMPORT LIBRARIES
# =============================================================================
# Standard libraries for data handling and analysis
import pandas as pd
import numpy as np

# Machine Learning library for the model
import lightgbm as lgb

# Utility for displaying progress
from tqdm import tqdm
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

print("Libraries imported successfully.")

# =============================================================================
# 2. LOAD DATA
# =============================================================================
# Load the training, testing, and sample submission files into pandas DataFrames.
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s4e3/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s4e3/test.csv')
    sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s4e3/sample_submission.csv')
    print("Data loaded successfully.")
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the data files are in the correct directory.")
    exit() # Exit the script if data is not found

# =============================================================================
# 3. DATA PREPARATION & FEATURE ENGINEERING
# =============================================================================
# In this problem, the provided features are already generated from another model.
# Therefore, extensive feature engineering may not be necessary. We will focus on
# preparing the data for the model.

# Define the target columns (the 7 defect types)
TARGET_COLUMNS = [
    'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
    'Dirtiness', 'Bumps', 'Other_Faults'
]

# The 'id' column is an identifier and not a feature for the model.
# We identify feature columns as all columns that are not 'id' or a target.
FEATURE_COLUMNS = [col for col in train_df.columns if col not in TARGET_COLUMNS + ['id']]

print("\nData Preparation:")
print(f"Identified {len(TARGET_COLUMNS)} target columns.")
print(f"Identified {len(FEATURE_COLUMNS)} feature columns.")

# Separate features (X) and targets (y) for training and testing
X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMNS]
X_test = test_df[FEATURE_COLUMNS]

# Check for data consistency between train and test sets
if not all(X_train.columns == X_test.columns):
    raise ValueError("Train and test columns do not match!")

# Let's inspect the first few rows of our feature sets
print("\nFirst 5 rows of training features (X_train):")
print(X_train.head())

# =============================================================================
# 4. MODEL TRAINING (BINARY RELEVANCE with LightGBM)
# =============================================================================
# We will train a separate LightGBM classifier for each of the 7 defect types.
# This approach is called "Binary Relevance".

print("\nStarting model training...")

# Dictionary to store the predictions for each target
predictions = {}

# Define LightGBM model parameters
# These are some well-performing, general-purpose parameters.
# For better results, these can be tuned using techniques like cross-validation.
lgb_params = {
    'objective': 'binary',        # Objective for binary classification
    'metric': 'auc',              # Evaluation metric: Area Under ROC Curve
    'boosting_type': 'gbdt',      # Traditional Gradient Boosting Decision Tree
    'n_estimators': 500,          # Number of boosting rounds
    'learning_rate': 0.02,        # Step size shrinkage
    'num_leaves': 31,             # Max number of leaves in one tree
    'max_depth': -1,              # No limit on tree depth
    'seed': 42,                   # Random seed for reproducibility
    'n_jobs': -1,                 # Use all available CPU cores
    'verbose': -1,                # Suppress verbose output
    'colsample_bytree': 0.8,      # Subsample ratio of columns when constructing each tree
    'subsample': 0.8,             # Subsample ratio of the training instance
}


# Loop through each target column, train a model, and make predictions
for target in tqdm(TARGET_COLUMNS, desc="Training models for each defect"):
    print(f"\nTraining model for: {target}")

    # Initialize a new LightGBM classifier with the defined parameters
    model = lgb.LGBMClassifier(**lgb_params)

    # Train the model on the full training data for the current target
    model.fit(X_train, y_train[target])

    # Predict probabilities on the test set for the positive class (class 1)
    # predict_proba returns a 2D array: [prob_class_0, prob_class_1]
    # We need the probability of the defect being present, which is the second column [:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]

    # Store the predictions in our dictionary
    predictions[target] = test_preds

print("\nModel training and prediction completed.")

# =============================================================================
# 5. CREATE SUBMISSION FILE
# =============================================================================
# Now, we will format our predictions into the required submission file format.

print("\nCreating submission file...")

# Create a new DataFrame for the submission using the 'id' from the test set
submission_df = pd.DataFrame({'id': test_df['id']})

# Add the prediction columns to the submission DataFrame
for target, preds in predictions.items():
    submission_df[target] = preds

# Save the DataFrame to a CSV file, without the index column
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print("First 5 rows of the submission file:")
print(submission_df.head())

# =============================================================================
# 6. CONCLUSION & NEXT STEPS
# =============================================================================
print("\n--- Process Finished ---")
print("This script has successfully:")
print("1. Loaded and prepared the data.")
print("2. Trained 7 independent LightGBM models.")
print("3. Predicted defect probabilities on the test set.")
print("4. Generated a 'submission.csv' file in the correct format.")
print("\nPossible Improvements for higher scores:")
print("- Hyperparameter Tuning: Use Optuna or GridSearchCV with cross-validation.")
print("- Cross-Validation: Train models using K-Fold cross-validation for more robust predictions.")
print("- Feature Engineering: Although features are pre-generated, exploring interactions or polynomial features might help.")
print("- Model Ensembling: Combine predictions from different models (e.g., XGBoost, CatBoost) to improve generalization.")




