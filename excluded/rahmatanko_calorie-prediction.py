import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder


def rmsle(ytrue, ypred):
    # Enforce non-negativity for predictions.
    ypred[ypred < 0] = 0
    # Compute the square root of the mean squared logarithmic error.
    return np.sqrt(mean_squared_log_error(ytrue, ypred))



# Verify the presence of 'train.csv' and 'test.csv' in the current working directory.
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
    print("Dataset loading successful.")
    print(f"Training data dimensions: {train_df.shape[0]} rows, {train_df.shape[1]} columns.")
    print(f"Test data dimensions: {test_df.shape[0]} rows, {test_df.shape[1]} columns.")
except FileNotFoundError:
    print("Error: Required data files ('train.csv' or 'test.csv') not found.")
    print("Please ensure these files are located in the same directory as this notebook.")
    exit() # Terminate execution if data files are inaccessible.

# Preserve the 'id' column from the test set; it is essential for the submission format.
test_ids = test_df['id']


def preprocess_and_engineer_features(df):
    """
    Applies a series of preprocessing steps and generates engineered features
    to enhance the predictive power of the dataset.
    """
    # Initialize LabelEncoder for categorical feature transformation.
    le = LabelEncoder()
    # Apply Label Encoding to the 'Sex' column and then drop the original column.
    df['Sex_encoded'] = le.fit_transform(df['Sex'])
    df = df.drop('Sex', axis=1)

    # Calculate Body Mass Index (BMI). Note: Height is converted from cm to meters.
    df['BMI'] = df['Weight'] / ((df['Height'] / 100)**2)

    # Generate interaction terms, which often reveal non-linear relationships.
    df['Duration_HeartRate'] = df['Duration'] * df['Heart_Rate']
    df['Duration_BodyTemp'] = df['Duration'] * df['Body_Temp']
    df['HeartRate_BodyTemp'] = df['Heart_Rate'] * df['Body_Temp']

    # Handle any infinite values (e.g., from division by zero) by converting them to NaN.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Impute any remaining NaN values with the median of their respective columns.
    # This list should include all features that might contain NaNs after engineering.
    for col in ['BMI', 'Duration_HeartRate', 'Duration_BodyTemp', 'HeartRate_BodyTemp']:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df

print("\nInitiating data preprocessing and feature engineering pipeline...")
train_df_processed = preprocess_and_engineer_features(train_df.copy())
test_df_processed = preprocess_and_engineer_features(test_df.copy())
print("Preprocessing and feature engineering completed successfully.")

# Identify the features (independent variables) and the target (dependent variable).
features = [col for col in train_df_processed.columns if col not in ['id', 'Calories']]
X_train = train_df_processed[features]
y_train = train_df_processed['Calories']
X_test = test_df_processed[features]

# Ensure column consistency between training and test sets. This is crucial for model inference.
common_cols = list(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_cols]
X_test = X_test[common_cols]
print(f"\nFeatures selected for model training ({len(common_cols)}): {common_cols}")



print("\nCommencing K-Fold Cross-Validation for XGBoost model training...")

# Define XGBoost Regressor hyperparameters. These parameters are chosen as a strong baseline;
# further tuning can lead to incremental performance gains.
xgb_params = {
    'objective': 'reg:squarederror', # Specifies a regression task (predicting continuous values).
    'eval_metric': 'rmse',           # Metric used for evaluation during training (Root Mean Squared Error).
    'eta': 0.01,                     # Learning rate: controls the step size at each iteration.
    'max_depth': 8,                  # Maximum depth of a tree: limits model complexity.
    'subsample': 0.7,                # Subsample ratio of the training instance: reduces variance.
    'colsample_bytree': 0.7,         # Subsample ratio of columns when constructing each tree: prevents overfitting.
    'min_child_weight': 1,           # Minimum sum of instance weight needed in a child: controls tree splitting.
    'seed': 42,                      # Random seed for reproducibility across runs.
    'n_estimators': 2000,            # Number of boosting rounds (trees) to build.
    'n_jobs': -1,                    # Utilizes all available CPU cores for parallel processing.
    'tree_method': 'hist',           # Specifies the histogram-based tree construction algorithm for GPU.
    'device': 'cuda',
    'early_stopping_rounds': 100     # Stops training if validation metric does not improve for 100 consecutive rounds.
}

# Initialize K-Fold Cross-Validation with 5 splits, shuffling data for randomness.
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train)) # Array to store Out-Of-Fold predictions for training data.
test_preds = np.zeros(len(X_test)) # Array to accumulate predictions for the test set.
fold_rmsle_scores = [] # List to store RMSLE score for each fold.

# Iterate through each fold for training and validation.
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"--- Processing Fold {fold+1}/{kf.n_splits} ---")
    # Split data into training and validation sets for the current fold.
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Instantiate the XGBoost Regressor model with defined parameters.
    model = xgb.XGBRegressor(**xgb_params)
    # Train the model on the log-transformed target variable.
    model.fit(X_train_fold, np.log1p(y_train_fold),
              eval_set=[(X_val_fold, np.log1p(y_val_fold))], # Evaluate on log-transformed validation target.
              verbose=False) # Suppress verbose output during training for cleaner logs.

    # Generate predictions on the validation set and inverse-transform them.
    val_preds = np.expm1(model.predict(X_val_fold))
    oof_preds[val_idx] = val_preds # Store OOF predictions.

    # Calculate and record the RMSLE for the current fold.
    fold_rmsle = rmsle(y_val_fold, val_preds)
    fold_rmsle_scores.append(fold_rmsle)
    print(f"Fold {fold+1} RMSLE: {fold_rmsle:.6f}")

    # Accumulate predictions for the final test set by averaging across folds.
    test_preds += np.expm1(model.predict(X_test)) / kf.n_splits

# Calculate the overall RMSLE based on all Out-Of-Fold predictions.
overall_rmsle = rmsle(y_train, oof_preds)
print(f"\n--- Cross-Validation Training Complete ---")
print(f"Average Cross-Validation RMSLE across all folds: {np.mean(fold_rmsle_scores):.6f}")
print(f"Overall Out-Of-Fold (OOF) RMSLE: {overall_rmsle:.6f}")



# Apply a maximum threshold of 0 to ensure all calorie predictions are non-negative.
final_predictions = np.maximum(0, test_preds)
# Construct the submission DataFrame with 'id' and 'Calories' columns.
submission_df = pd.DataFrame({'id': test_ids, 'Calories': final_predictions})
# Save the DataFrame to a CSV file, excluding the index.
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' successfully generated!")

