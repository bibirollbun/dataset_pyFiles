import pandas as pd
import numpy as np
import glob
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold # Import KFold
import xgboost as xgb # Import the XGBoost library
import warnings

warnings.simplefilter('ignore')

# --- 1. Configuration ---
# Path to the original training data to get the true target values.
# Make sure this path is correct.
try:
    TRAIN_CSV_PATH = '/kaggle/input/playground-series-s4e12/train.csv'
    train_df = pd.read_csv(TRAIN_CSV_PATH)
except FileNotFoundError:
    print("train.csv not found. Creating a dummy train file for demonstration.")
    # Create a dummy file if not found, for demonstration purposes.
    # In a real scenario, you would need the actual train.csv.
    oof_files = glob.glob('oof_*.csv')
    if oof_files:
        num_samples = pd.read_csv(oof_files[0]).shape[0]
        train_df = pd.DataFrame({
            'id': range(num_samples),
            'Premium Amount': np.random.rand(num_samples) * 1000 + 500
        })
    else:
        print("Error: No OOF files found to determine sample size. Exiting.")
        exit()


# --- 2. Load Prediction Files ---

print("Searching for prediction files...")
# Find all Out-of-Fold (OOF) and test prediction files.
oof_files = sorted(glob.glob('/kaggle/input/**/oof_*.csv', recursive=True))
test_files = sorted(glob.glob('/kaggle/input/**/test_*.csv', recursive=True))

if not oof_files or not test_files or len(oof_files) != len(test_files):
    print("Error: Mismatch in OOF and Test files or no files found.")
    print(f"Found {len(oof_files)} OOF files and {len(test_files)} Test files.")
    exit()

print(f"Found {len(oof_files)} pairs of prediction files.")
for i, (o_file, t_file) in enumerate(zip(oof_files, test_files)):
    print(f"  Model {i+1}: {o_file} | {t_file}")

# Load the true target and apply log1p for RMSLE calculation.
y_true_log = np.log1p(train_df['Premium Amount'])

# Load all OOF predictions and apply log1p.
oof_preds_log = []
for file in oof_files:
    preds = pd.read_csv(file)['Premium Amount']
    oof_preds_log.append(np.log1p(preds))

# Load all test predictions.
test_preds = []
for file in test_files:
    preds = pd.read_csv(file)['Premium Amount']
    test_preds.append(preds)

# Stack the OOF predictions as columns to create the training data for the meta-model.
X_train_meta = np.column_stack(oof_preds_log)

# Stack the test predictions similarly to create the prediction data for the meta-model.
test_preds_log = [np.log1p(p) for p in test_preds]
X_test_meta = np.column_stack(test_preds_log)

print(f"\nMeta-model training data shape: {X_train_meta.shape}")
print(f"Meta-model test data shape: {X_test_meta.shape}")


# --- 3. XGBoost Meta-Model Training with Cross-Validation ---

# Define KFold with the same conditions as when creating the L1 model's OOF.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize arrays to store the meta-model's OOF predictions and test predictions.
oof_meta_preds = np.zeros(X_train_meta.shape[0])
test_meta_preds_list = []

print("\nStarting training of XGBoost meta-model with KFold CV...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_meta, y_true_log)):
    print(f"--- Fold {fold + 1}/{kf.get_n_splits()} ---")

    # Split the data into training and validation sets.
    X_train, X_val = X_train_meta[train_idx], X_train_meta[val_idx]
    y_train, y_val = y_true_log.iloc[train_idx], y_true_log.iloc[val_idx]

    # Define the XGBoost meta-model.
    # Parameters can be tuned as needed.
    meta_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=2000, # Increase n_estimators and optimize with early stopping
        learning_rate=0.02,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
        tree_method='hist' # Setting for faster training
    )

    # Train the meta-model using OOF predictions as features and the true values as the target.
    meta_model.fit(X_train, y_train,
                   eval_set=[(X_val, y_val)],
                   early_stopping_rounds=100, # Set early stopping rounds
                   verbose=200)

    # Save the predictions on the validation data as OOF predictions.
    oof_meta_preds[val_idx] = meta_model.predict(X_val)

    # Make predictions on the test data and store them in a list.
    test_meta_preds_list.append(meta_model.predict(X_test_meta))

# --- 4. Final Prediction and Submission ---

print("\n--- Meta-Model Training Finished ---")

# Calculate the overall OOF score for the meta-model.
rmsle_score = np.sqrt(mean_squared_error(y_true_log, oof_meta_preds))
print(f"Overall OOF RMSLE for Meta-Model: {rmsle_score:.6f}")

# Average the test predictions from each fold to get the final prediction.
final_preds_log = np.mean(test_meta_preds_list, axis=0)

# The predictions are on a log scale, so convert them back to the original scale.
final_test_preds = np.expm1(final_preds_log)

# Clip negative predictions to 0 (as insurance premiums cannot be negative).
final_test_preds[final_test_preds < 0] = 0

# Create the submission file.
submission_df = pd.read_csv(test_files[0])[['id']]
submission_df['Premium Amount'] = final_test_preds
submission_df.to_csv('submission_xgb_meta_cv.csv', index=False)

print("\nSubmission file 'submission_xgb_meta_cv.csv' created successfully.")




