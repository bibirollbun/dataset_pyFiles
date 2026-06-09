import pandas as pd
import lightgbm as lgb
import re
import os # To check for file existence

print("Libraries imported.")


# Define the base path for Kaggle inputs
# If running locally, change this path to where your train.csv and test.csv are located
try:
    # Running in Kaggle environment
    input_path = "/kaggle/input/health-signal-analytics-classifying-smoker-status"
    train_csv_path = os.path.join(input_path, "train.csv")
    test_csv_path = os.path.join(input_path, "test.csv")
    sample_submission_path = os.path.join(input_path, "sample_submission.csv")
    print("Running in Kaggle environment.")
except FileNotFoundError:
    # Running locally (adjust path as needed)
    print("Running locally. Assuming 'train.csv' and 'test.csv' are in the current directory.")
    train_csv_path = "train.csv"
    test_csv_path = "test.csv"
    sample_submission_path = "sample_submission.csv" # Or remove if not needed locally

# Load the datasets
print(f"Loading train data from: {train_csv_path}")
train_df = pd.read_csv(train_csv_path)

print(f"Loading test data from: {test_csv_path}")
test_df = pd.read_csv(test_csv_path)

# Optionally load sample submission to check format later
# print(f"Loading sample submission from: {sample_submission_path}")
# sample_submission_df = pd.read_csv(sample_submission_path)

print("Data loaded successfully.")


print("--- Training Data ---")
print(f"Shape: {train_df.shape}")
print(train_df.info())
print("\nFirst 5 rows:")
print(train_df.head())
print("\nDescribe:")
print(train_df.describe())


print("\n\n--- Test Data ---")
print(f"Shape: {test_df.shape}")
print(test_df.info())
print("\nFirst 5 rows:")
print(test_df.head())
print("\nDescribe:")
print(test_df.describe())

# print("\n\n--- Sample Submission ---")
# print(f"Shape: {sample_submission_df.shape}")
# print(sample_submission_df.head())


def clean_col_names(df):
    """Cleans column names by removing special characters and replacing spaces."""
    cols = df.columns
    new_cols = []
    for col in cols:
        # Replace special chars like '()','/' with '_'
        new_col = re.sub(r'[()/\-]+', '_', col)
        # Replace spaces with '_'
        new_col = new_col.replace(' ', '_')
        # Reduce multiple underscores to one
        new_col = re.sub(r'_+', '_', new_col)
        # Remove leading/trailing underscores
        new_col = new_col.strip('_')
        # Optional: convert to lowercase
        new_col = new_col.lower()
        new_cols.append(new_col)
    df.columns = new_cols
    return df

print("Cleaning column names for train_df...")
train_df = clean_col_names(train_df)
print("Train columns:", train_df.columns.tolist())

print("\nCleaning column names for test_df...")
test_df = clean_col_names(test_df)
print("Test columns:", test_df.columns.tolist())



train_df['bmi'] = train_df['weight_kg'] / (train_df['height_cm'] / 100)**2
test_df['bmi'] = test_df['weight_kg'] / (test_df['height_cm'] / 100)**2

# Other ideas:
# - Ratios between features (e.g., waist/height)
# - Polynomial features
# - Binning continuous features

print("Feature engineering step ")


target = 'smoking'
# Identify features (all columns except 'id' and the target 'smoking')
# Ensure column names used here match the cleaned names
features = [col for col in train_df.columns if col not in ['id', target]]

print(f"Target variable: {target}")
print(f"Number of features: {len(features)}")
# print(f"Features: {features}") # Uncomment to see all features

# Prepare data for LightGBM
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features] # Use the same features from the test set
test_ids = test_df['id']   # Keep track of IDs for submission

print("\nTrain features shape:", X_train.shape)
print("Train target shape:", y_train.shape)
print("Test features shape:", X_test.shape)
print("Test IDs shape:", test_ids.shape)


# Configure LightGBM Classifier
# Parameters can be tuned using techniques like GridSearchCV or RandomizedSearchCV
lgbm = lgb.LGBMClassifier(objective='binary',
                           metric='auc',      # Evaluation metric for the competition
                           random_state=42,   # For reproducibility
                           n_estimators=1000, # Start with a reasonable number, tune if needed
                           learning_rate=0.05, # Common starting point, tune if needed
                           num_leaves=31,      # Default, tune if needed
                           # Add other hyperparameters for tuning as needed:
                           # colsample_bytree=0.8,
                           # subsample=0.8,
                           # reg_alpha=0.1,
                           # reg_lambda=0.1,
                         )

print("Starting LightGBM model training...")
lgbm.fit(X_train, y_train)
print("Model training complete.")

# Optional: Check feature importance
# feature_importance_df = pd.DataFrame({'feature': features, 'importance': lgbm.feature_importances_})
# feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
# print("\nTop 10 Feature Importances:")
# print(feature_importance_df.head(10))


print("Making predictions on the test set...")
# Predict probabilities for the positive class (smoking = 1)
predictions_proba = lgbm.predict_proba(X_test)[:, 1]
print("Predictions generated.")
print("First 10 predicted probabilities:", predictions_proba[:10])


print("Creating submission DataFrame...")
submission_df = pd.DataFrame({'id': test_ids, 'smoking': predictions_proba})

print("Submission DataFrame head:")
print(submission_df.head())

# Define the output file path
submission_path = "submission.csv" # Saves in the default output directory in Kaggle

print(f"\nSaving submission file to: {submission_path}")
submission_df.to_csv(submission_path, index=False)

print("Submission file created successfully!")




