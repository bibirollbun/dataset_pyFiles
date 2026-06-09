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


# --- 1. Import Libraries ---
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
import gc  # Garbage Collector for memory management

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')


# --- 2. Configuration and Setup ---

# Set a random seed for reproducibility (so results are the same every time)
SEED = 42
np.random.seed(SEED)

# Set the number of folds for cross-validation
# 10 is robust, 3 or 5 is faster for testing
N_SPLITS = 10

# Define the path to your data files
DATA_PATH = '../input/cat-in-the-dat-ii/'


# --- 3. Load Data ---
print("Loading data...")
try:
    # Read the training, test, and submission files
    train_df = pd.read_csv(f'{DATA_PATH}train.csv')
    test_df = pd.read_csv(f'{DATA_PATH}test.csv')
    submission_df = pd.read_csv(f'{DATA_PATH}sample_submission.csv')

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

except FileNotFoundError:
    print(f"Error: Data files not found in {DATA_PATH}")
    # Handle the error, maybe stop the script
    # For now, we'll assume the files loaded


# --- 4. Exploratory Data Analysis (EDA) ---
# This section is for understanding the data

print("\n--- Starting EDA ---")

# Check the target variable distribution
print("\nTarget Distribution:")
print(train_df['target'].value_counts(normalize=True))

# Check for columns that have missing (null) values
print("\nMissing Values (Train):")
missing_train = train_df.isnull().sum()
print(missing_train[missing_train > 0])

# Check the number of unique values (cardinality) for each feature
print("\nFeature Cardinality (Unique Values):")
for col in train_df.columns:
    if col not in ['id', 'target']:
        print(f"{col}: {train_df[col].nunique()} unique values")

print("--- EDA Complete ---")


# --- 5. Feature Engineering and Preprocessing ---
print("\n--- Starting Feature Engineering ---")

# Create copies for processing
# X = features, y = target
X = train_df.drop(['id', 'target'], axis=1)
y = train_df['target']
X_test = test_df.drop('id', axis=1)

# Save the test IDs for the final submission file
X_test_ids = test_df['id']

# --- 5a. Handle Missing Values ---
print("Handling missing values...")
# Loop through all columns
for col in X.columns:
    # Find columns that are 'object' (text/string) type
    if X[col].dtype == 'object':
        # Fill missing values with the string "MISSING"
        # This treats "missing" as its own category
        X[col] = X[col].fillna("MISSING")
        X_test[col] = X_test[col].fillna("MISSING")

# Note: We leave numeric NaNs (like in bin_0) as is.
# CatBoost can handle them natively.

# --- 5b. Manual Feature Mapping ---
print("Mapping manual ordinal/binary features...")
# These features have a clear order, so we map them to numbers

# ord_1: Novice (0) < Contributor (1) < ...
ord_1_mapping = {
    'Novice': 0, 'Contributor': 1, 'Expert': 2,
    'Master': 3, 'Grandmaster': 4, "MISSING": -1
}
X['ord_1'] = X['ord_1'].map(ord_1_mapping)
X_test['ord_1'] = X_test['ord_1'].map(ord_1_mapping)

# ord_2: Freezing (0) < Cold (1) < ...
ord_2_mapping = {
    'Freezing': 0, 'Cold': 1, 'Warm': 2, 'Hot': 3,
    'Boiling Hot': 4, 'Lava Hot': 5, "MISSING": -1
}
X['ord_2'] = X['ord_2'].map(ord_2_mapping)
X_test['ord_2'] = X_test['ord_2'].map(ord_2_mapping)

# bin_3: T/F to 1/0
X['bin_3'] = X['bin_3'].map({'T': 1, 'F': 0, 'MISSING': -1})
X_test['bin_3'] = X_test['bin_3'].map({'T': 1, 'F': 0, 'MISSING': -1})

# bin_4: Y/N to 1/0
X['bin_4'] = X['bin_4'].map({'Y': 1, 'N': 0, 'MISSING': -1})
X_test['bin_4'] = X_test['bin_4'].map({'Y': 1, 'N': 0, 'MISSING': -1})

# --- 5c. Cyclic Feature Engineering ---
print("Creating cyclic features for day and month...")
# This helps the model understand that day 7 is next to day 1,
# and month 12 is next to month 1.

# Fill any missing day/month with 0 (a neutral placeholder)
X['day'] = X['day'].fillna(0)
X_test['day'] = X_test['day'].fillna(0)
X['month'] = X['month'].fillna(0)
X_test['month'] = X_test['month'].fillna(0)

# Create sin/cos features for day of the week (max=7)
X['day_sin'] = np.sin(2 * np.pi * X['day'] / 7.0)
X['day_cos'] = np.cos(2 * np.pi * X['day'] / 7.0)
X_test['day_sin'] = np.sin(2 * np.pi * X_test['day'] / 7.0)
X_test['day_cos'] = np.cos(2 * np.pi * X_test['day'] / 7.0)

# Create sin/cos features for month of the year (max=12)
X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12.0)
X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12.0)
X_test['month_sin'] = np.sin(2 * np.pi * X_test['month'] / 12.0)
X_test['month_cos'] = np.cos(2 * np.pi * X_test['month'] / 12.0)

# Drop the original 'day' and 'month' columns
X = X.drop(['day', 'month'], axis=1)
X_test = X_test.drop(['day', 'month'], axis=1)

# --- 5d. Advanced 'ord_5' Splitting ---
print("Splitting ord_5...")
# 'ord_5' has values like 'aZ', 'bA'.
# Splitting them into two separate features can be powerful.
X['ord_5_1'] = X['ord_5'].str[0]
X['ord_5_2'] = X['ord_5'].str[1]
X_test['ord_5_1'] = X_test['ord_5'].str[0]
X_test['ord_5_2'] = X_test['ord_5'].str[1]

# Drop the original 'ord_5' column
X = X.drop('ord_5', axis=1)
X_test = X_test.drop('ord_5', axis=1)

# --- 5e. Define and Prepare Categorical Features ---
print("Defining and converting categorical features...")

# List all features we want CatBoost to treat as categories
cat_features = [
    'bin_0', 'bin_1', 'bin_2',
    'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4',
    'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',
    'ord_0', 'ord_3', 'ord_4',
    'ord_5_1', 'ord_5_2'  # Our new engineered features
]

# Convert all these categorical columns to 'str' type
# This is a safety step to ensure CatBoost treats them as categories,
# even if they look like numbers (e.g., bin_0).
for col in cat_features:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# Fill NaNs in numeric columns that we didn't map
# (CatBoost can handle NaNs, but filling with a placeholder is explicit)
num_cols = ['bin_0', 'bin_1', 'bin_2']
X[num_cols] = X[num_cols].fillna(-1)
X_test[num_cols] = X_test[num_cols].fillna(-1)

print("--- Preprocessing and FE Complete ---")


# --- 6. Model Training (Tuned CatBoost) ---
print(f"\n--- Starting CatBoost {N_SPLITS}-Fold Training ---")

# Setup the K-Fold cross-validation
# StratifiedKFold ensures each fold has the same balance of targets (0s and 1s)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# Create arrays to store out-of-fold (OOF) predictions and test predictions
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
models = []  # List to store the trained model from each fold

# Start the cross-validation loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")

    # Split the data into training and validation sets for this fold
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Initialize the CatBoostClassifier with tuned parameters
    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.02,
        depth=7,
        eval_metric='AUC',
        random_seed=SEED,
        task_type="GPU",  # Use "GPU" or "CPU"
        early_stopping_rounds=200,  # Stop if AUC doesn't improve for 200 rounds
        verbose=1000,  # Print progress every 500 trees
        cat_features=cat_features,
        nan_mode='Min',  # How to handle missing values in numeric features

        # --- Regularization Parameters ---
        l2_leaf_reg=3.0,  # L2 regularization
        subsample=0.8,  # Use 80% of data for training each tree
        bootstrap_type='Poisson',  # Required for 'subsample' on GPU
    )

    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True  # Save the model from the best iteration
    )

    # Make predictions on the validation set
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds

    # Make predictions on the test set
    # We average the predictions from all folds
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    models.append(model)

    print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, val_preds)}")

    # Clean up memory
    del X_train, y_train, X_val, y_val
    gc.collect()


# --- 7. Final Results ---
print("\n--- Training Complete ---")

# Calculate and print the overall Out-of-Fold (OOF) AUC score
# This is our most reliable measure of the model's performance
overall_auc = roc_auc_score(y, oof_preds)
print(f"Overall CV AUC (Tuned CatBoost): {overall_auc}")


# --- 8. Create Submission File ---
print("Creating submission file...")

# Create a DataFrame with the test IDs and our averaged predictions
submission_df_cb = pd.DataFrame({'id': X_test_ids, 'target': test_preds})

# Save the DataFrame to a CSV file
submission_df_cb.to_csv('submission.csv', index=False)

print("\nAll steps complete! 'submission.csv' is ready.")

