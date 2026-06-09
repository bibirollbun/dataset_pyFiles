import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# --- 1. Data Loading ---
# Define paths for the dataset files.
TRAIN_FILE = 'train.parquet'
TEST_FILE = 'test.parquet'
SAMPLE_SUBMISSION_FILE = 'sample_submission.csv'
SUBMISSION_OUTPUT_FILE = 'submission.csv' # Output file name


print(f"Loading sample submission file from {SAMPLE_SUBMISSION_FILE} to determine required rows...")
try:
    sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_FILE)
    required_test_rows = len(sample_submission_df)
    print(f"Sample submission loaded. Required rows for submission: {required_test_rows}")
except FileNotFoundError as e:
    print(f"Error loading sample_submission.csv: {e}. Cannot determine exact required rows.")
    # Fallback for dummy data if sample_submission.csv is also not found
    required_test_rows = 538150 # Use the specific number from the error message as a fallback
    print(f"Using fallback for required rows: {required_test_rows}")



print(f"Loading data from {TRAIN_FILE} and {TEST_FILE}...")
try:
    train_df = pd.read_parquet(TRAIN_FILE)
    test_df = pd.read_parquet(TEST_FILE)
    print("Train and Test data loaded successfully from actual files.")
except FileNotFoundError as e:
    print(f"Error loading train/test files: {e}. Generating dummy data for demonstration purposes.")
    # --- Generate Dummy Data if files are not found (for demonstration) ---
    num_train_rows = 10000 # Keep dummy train size reasonable
    num_x_features = 890 # As specified in the description
    x_features_names = [f'X_{i}' for i in range(1, num_x_features + 1)]
    base_market_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']


# Dummy train_df
train_data = {
        'timestamp': pd.to_datetime(pd.date_range(start='2024-01-01', periods=num_train_rows, freq='min')),
        'bid_qty': np.random.rand(num_train_rows) * 100 + 50,
        'ask_qty': np.random.rand(num_train_rows) * 100 + 51, # ask_qty > bid_qty generally
        'buy_qty': np.random.rand(num_train_rows) * 20,
        'sell_qty': np.random.rand(num_train_rows) * 20,
        'volume': np.random.rand(num_train_rows) * 100,
        'label': np.random.rand(num_train_rows) * 0.1 - 0.05 # Anonymized price movement (e.g., -0.05 to +0.05)
}
for feature in x_features_names:
    train_data[feature] = np.random.rand(num_train_rows) * 10 # Example random X features
train_df = pd.DataFrame(train_data)



# Dummy test_df - Use 'required_test_rows' for the number of rows
test_data = {
        'timestamp': np.arange(required_test_rows), # Unique ID, now matching required length
        'bid_qty': np.random.rand(required_test_rows) * 100 + 50,
        'ask_qty': np.random.rand(required_test_rows) * 100 + 51,
        'buy_qty': np.random.rand(required_test_rows) * 20,
        'sell_qty': np.random.rand(required_test_rows) * 20,
        'volume': np.random.rand(required_test_rows) * 100,
        'label': np.zeros(required_test_rows) # All labels are 0 in the test set
}
for feature in x_features_names:
    test_data[feature] = np.random.rand(required_test_rows) * 10
test_df = pd.DataFrame(test_data)



# Re-create dummy sample_submission_df based on the determined required_test_rows
# This ensures consistency for the dummy data path.
# IMPORTANT: When generating dummy sample_submission_df, ensure it has 'ID' column
# with the correct indexing (e.g., 1-indexed if Kaggle expects it).
sample_submission_df = pd.DataFrame({
        'ID': np.arange(required_test_rows) + 1, # THIS IS THE LINE THAT WAS CHANGED
        'prediction': np.zeros(required_test_rows)
})
print(f"Dummy train/test data generated successfully. Dummy test_df has {len(test_df)} rows.")

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Sample Submission data shape (used for row count check): {sample_submission_df.shape}")


# Ensure the test_df has the correct number of rows if it was loaded from file.
# This check is more critical when actual files are loaded.
if len(test_df) != required_test_rows:
    print(f"Warning: test_df has {len(test_df)} rows, but sample_submission.csv requires {required_test_rows} rows.")
    print("This might indicate an issue with the test.parquet file or a mismatch in expected data.")
    # For a competition, you'd likely stop here or investigate.
    # For now, we will proceed, but it's important to flag.


# --- 2. Feature Engineering ---
features = [col for col in train_df.columns if col.startswith('X_')]
features.extend(['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume'])
target = 'label'

train_df.dropna(subset=[target], inplace=True)

common_features_in_both = list(set(train_df.columns) & set(test_df.columns))
model_features = [f for f in features if f in common_features_in_both and f != target]

X_train = train_df[model_features]
y_train = train_df[target]
X_test = test_df[model_features]

print(f"Number of features used for training: {len(model_features)}")
print(f"First 5 feature names used: {model_features[:5]}")


# --- 3. Model Training ---
split_point = int(len(X_train) * 0.8)
X_train_model, X_val_model = X_train.iloc[:split_point], X_train.iloc[split_point:]
y_train_model, y_val_model = y_train.iloc[:split_point], y_train.iloc[split_point:]

print(f"Training data size for model: {len(X_train_model)}")
print(f"Validation data size for model: {len(X_val_model)}")

lgb_params = {
    'objective': 'regression_l1',
    'metric': 'rmse',
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 64,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}

model = lgb.LGBMRegressor(**lgb_params)

print("\nStarting model training (with early stopping)...")
model.fit(X_train_model, y_train_model,
          eval_set=[(X_val_model, y_val_model)],
          eval_metric='rmse',
          callbacks=[lgb.early_stopping(100, verbose=False)],
          )

print("Model training complete.")

val_predictions = model.predict(X_val_model)
rmse = np.sqrt(mean_squared_error(y_val_model, val_predictions))
print(f"Validation RMSE: {rmse:.4f}")



# --- 4. Prediction ---
print("Making predictions on the test set...")
test_predictions = model.predict(X_test)
print("Predictions complete.")


# --- 5. Submission File Generation ---
# Ensure the prediction array has the same number of rows as required
if len(test_predictions) != required_test_rows:
    print(f"Adjusting prediction array length from {len(test_predictions)} to {required_test_rows}.")
    # This could happen if test_df was malformed or shorter than expected.
    # For robust competitions, you might want to raise an error if this happens.
    # For now, we'll pad/truncate if necessary, but it's not ideal.
    if len(test_predictions) > required_test_rows:
        test_predictions = test_predictions[:required_test_rows]
    else:
        # Pad with zeros or a sensible default if predictions are too short
        test_predictions = np.pad(test_predictions, (0, required_test_rows - len(test_predictions)), 'constant', constant_values=0)



# Create a DataFrame for submission with 'ID' and 'prediction' columns
submission_df = pd.DataFrame({
    'ID': sample_submission_df['ID'],  # Use the 'ID' column directly from sample_submission_df
    'prediction': test_predictions
})


# Save the submission file
submission_df.to_csv(SUBMISSION_OUTPUT_FILE, index=False)


print(f"\nSubmission file '{SUBMISSION_OUTPUT_FILE}' created successfully with {len(submission_df)} rows.")
print("First 5 rows of the submission file:")
print(submission_df.head())


# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# # --- 1. Data Loading ---
# # Define paths for the dataset files.
# TRAIN_FILE = 'train.parquet'
# TEST_FILE = 'test.parquet'
# SAMPLE_SUBMISSION_FILE = 'sample_submission.csv'
# SUBMISSION_OUTPUT_FILE = 'submission.csv' # Output file name

# print(f"Loading sample submission file from {SAMPLE_SUBMISSION_FILE} to determine required rows...")
# try:
#     sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_FILE)
#     required_test_rows = len(sample_submission_df)
#     print(f"Sample submission loaded. Required rows for submission: {required_test_rows}")
# except FileNotFoundError as e:
#     print(f"Error loading sample_submission.csv: {e}. Cannot determine exact required rows.")
#     # Fallback for dummy data if sample_submission.csv is also not found
#     required_test_rows = 538150 # Use the specific number from the error message as a fallback
#     print(f"Using fallback for required rows: {required_test_rows}")


# print(f"Loading data from {TRAIN_FILE} and {TEST_FILE}...")
# try:
#     train_df = pd.read_parquet(TRAIN_FILE)
#     test_df = pd.read_parquet(TEST_FILE)
#     print("Train and Test data loaded successfully from actual files.")
# except FileNotFoundError as e:
#     print(f"Error loading train/test files: {e}. Generating dummy data for demonstration purposes.")
#     # --- Generate Dummy Data if files are not found (for demonstration) ---
#     num_train_rows = 10000 # Keep dummy train size reasonable
#     num_x_features = 890 # As specified in the description
#     x_features_names = [f'X_{i}' for i in range(1, num_x_features + 1)]
#     base_market_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']

#     # Dummy train_df
#     train_data = {
#         'timestamp': pd.to_datetime(pd.date_range(start='2024-01-01', periods=num_train_rows, freq='min')),
#         'bid_qty': np.random.rand(num_train_rows) * 100 + 50,
#         'ask_qty': np.random.rand(num_train_rows) * 100 + 51, # ask_qty > bid_qty generally
#         'buy_qty': np.random.rand(num_train_rows) * 20,
#         'sell_qty': np.random.rand(num_train_rows) * 20,
#         'volume': np.random.rand(num_train_rows) * 100,
#         'label': np.random.rand(num_train_rows) * 0.1 - 0.05 # Anonymized price movement (e.g., -0.05 to +0.05)
#     }
#     for feature in x_features_names:
#         train_data[feature] = np.random.rand(num_train_rows) * 10 # Example random X features
#     train_df = pd.DataFrame(train_data)

#     # Dummy test_df - Use 'required_test_rows' for the number of rows
#     test_data = {
#         'timestamp': np.arange(required_test_rows), # Unique ID, now matching required length
#         'bid_qty': np.random.rand(required_test_rows) * 100 + 50,
#         'ask_qty': np.random.rand(required_test_rows) * 100 + 51,
#         'buy_qty': np.random.rand(required_test_rows) * 20,
#         'sell_qty': np.random.rand(required_test_rows) * 20,
#         'volume': np.random.rand(required_test_rows) * 100,
#         'label': np.zeros(required_test_rows) # All labels are 0 in the test set
#     }
#     for feature in x_features_names:
#         test_data[feature] = np.random.rand(required_test_rows) * 10
#     test_df = pd.DataFrame(test_data)

#     # Re-create dummy sample_submission_df based on the determined required_test_rows
#     # This ensures consistency for the dummy data path.
#     # IMPORTANT: When generating dummy sample_submission_df, ensure it has 'ID' column
#     # with the correct indexing (e.g., 1-indexed if Kaggle expects it).
#     sample_submission_df = pd.DataFrame({
#         'ID': np.arange(required_test_rows) + 1, # THIS IS THE LINE THAT WAS CHANGED
#         'prediction': np.zeros(required_test_rows)
#     })
#     print(f"Dummy train/test data generated successfully. Dummy test_df has {len(test_df)} rows.")

# print(f"Train data shape: {train_df.shape}")
# print(f"Test data shape: {test_df.shape}")
# print(f"Sample Submission data shape (used for row count check): {sample_submission_df.shape}")

# # Ensure the test_df has the correct number of rows if it was loaded from file.
# # This check is more critical when actual files are loaded.
# if len(test_df) != required_test_rows:
#     print(f"Warning: test_df has {len(test_df)} rows, but sample_submission.csv requires {required_test_rows} rows.")
#     print("This might indicate an issue with the test.parquet file or a mismatch in expected data.")
#     # For a competition, you'd likely stop here or investigate.
#     # For now, we will proceed, but it's important to flag.

# # --- 2. Feature Engineering ---
# features = [col for col in train_df.columns if col.startswith('X_')]
# features.extend(['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume'])
# target = 'label'

# train_df.dropna(subset=[target], inplace=True)

# common_features_in_both = list(set(train_df.columns) & set(test_df.columns))
# model_features = [f for f in features if f in common_features_in_both and f != target]

# X_train = train_df[model_features]
# y_train = train_df[target]
# X_test = test_df[model_features]

# print(f"Number of features used for training: {len(model_features)}")
# print(f"First 5 feature names used: {model_features[:5]}")

# # --- 3. Model Training ---
# split_point = int(len(X_train) * 0.8)
# X_train_model, X_val_model = X_train.iloc[:split_point], X_train.iloc[split_point:]
# y_train_model, y_val_model = y_train.iloc[:split_point], y_train.iloc[split_point:]

# print(f"Training data size for model: {len(X_train_model)}")
# print(f"Validation data size for model: {len(X_val_model)}")

# lgb_params = {
#     'objective': 'regression_l1',
#     'metric': 'rmse',
#     'n_estimators': 2000,
#     'learning_rate': 0.02,
#     'feature_fraction': 0.7,
#     'bagging_fraction': 0.7,
#     'bagging_freq': 1,
#     'lambda_l1': 0.1,
#     'lambda_l2': 0.1,
#     'num_leaves': 64,
#     'verbose': -1,
#     'n_jobs': -1,
#     'seed': 42,
#     'boosting_type': 'gbdt',
# }

# model = lgb.LGBMRegressor(**lgb_params)

# print("\nStarting model training (with early stopping)...")
# model.fit(X_train_model, y_train_model,
#           eval_set=[(X_val_model, y_val_model)],
#           eval_metric='rmse',
#           callbacks=[lgb.early_stopping(100, verbose=False)],
#           )

# print("Model training complete.")

# val_predictions = model.predict(X_val_model)
# rmse = np.sqrt(mean_squared_error(y_val_model, val_predictions))
# print(f"Validation RMSE: {rmse:.4f}")

# # --- 4. Prediction ---
# print("Making predictions on the test set...")
# test_predictions = model.predict(X_test)
# print("Predictions complete.")

# # --- 5. Submission File Generation ---
# # Ensure the prediction array has the same number of rows as required
# if len(test_predictions) != required_test_rows:
#     print(f"Adjusting prediction array length from {len(test_predictions)} to {required_test_rows}.")
#     # This could happen if test_df was malformed or shorter than expected.
#     # For robust competitions, you might want to raise an error if this happens.
#     # For now, we'll pad/truncate if necessary, but it's not ideal.
#     if len(test_predictions) > required_test_rows:
#         test_predictions = test_predictions[:required_test_rows]
#     else:
#         # Pad with zeros or a sensible default if predictions are too short
#         test_predictions = np.pad(test_predictions, (0, required_test_rows - len(test_predictions)), 'constant', constant_values=0)


# # Create a DataFrame for submission with 'ID' and 'prediction' columns
# submission_df = pd.DataFrame({
#     'ID': sample_submission_df['ID'],  # Use the 'ID' column directly from sample_submission_df
#     'prediction': test_predictions
# })

# # Save the submission file
# submission_df.to_csv(SUBMISSION_OUTPUT_FILE, index=False)

# print(f"\nSubmission file '{SUBMISSION_OUTPUT_FILE}' created successfully with {len(submission_df)} rows.")
# print("First 5 rows of the submission file:")
# print(submission_df.head())


