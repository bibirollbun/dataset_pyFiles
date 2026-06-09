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


# Part 1: Imports and Data Loading

# Essential libraries for data manipulation and numerical operations
import pandas as pd
import numpy as np

# Libraries for machine learning modeling
from sklearn.model_selection import KFold, cross_val_score # For cross-validation splitting and evaluation
from sklearn.preprocessing import StandardScaler          # For feature scaling
from sklearn.linear_model import LinearRegression         # Linear model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # Ensemble models
from xgboost import XGBRegressor                          # Extreme Gradient Boosting
from lightgbm import LGBMRegressor                        # Light Gradient Boosting Machine

# Libraries for evaluating model performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

# Optional: For saving and loading models (useful for deploying later)
import joblib

# --- Configuration for Kaggle Environment ---
# Define the base path for your input files in Kaggle.
# This path is where 'train.csv', 'test.csv', and 'sample_submission.csv' are located.
KAGGLE_INPUT_PATH = "/kaggle/input/playground-series-s4e5/"

# --- Load Data ---
print("--- Part 1: Loading Data ---")
try:
    # Load the training dataset (contains features and the target variable 'FloodProbability')
    train_df = pd.read_csv(KAGGLE_INPUT_PATH + 'train.csv')
    # Load the test dataset (contains only features, you need to predict 'FloodProbability' for these)
    test_df = pd.read_csv(KAGGLE_INPUT_PATH + 'test.csv')
    # Load the sample submission file (shows the required format for your final submission)
    sample_submission_df = pd.read_csv(KAGGLE_INPUT_PATH + 'sample_submission.csv')

    print("All data files loaded successfully.")
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Sample submission shape: {sample_submission_df.shape}")

except FileNotFoundError as e:
    # If any file is not found, print an error and exit to prevent further issues.
    print(f"ERROR: One or more data files not found. Please check the path: {e}")
    print(f"Expected path: {KAGGLE_INPUT_PATH}")
    exit() # Exit the script if data loading fails

print("\n--- Part 1 Complete ---")


# Part 2: Initial Data Preprocessing

print("\n--- Part 2: Initial Data Preprocessing ---")

# --- Process Training Data ---
# Kaggle datasets often have an 'id' column that is not a feature and should be removed.
if 'id' in train_df.columns:
    train_df = train_df.drop('id', axis=1) # axis=1 means drop a column
    print("Dropped 'id' column from 'train_df'.")
else:
    print("'id' column not found in 'train_df', skipping drop.")

# Handle duplicate rows in the training data.
# Duplicate rows can bias the model and lead to overfitting.
initial_train_rows = train_df.shape[0] # Get the number of rows before removing duplicates
train_df.drop_duplicates(inplace=True) # Remove duplicate rows directly in the DataFrame
duplicates_removed = initial_train_rows - train_df.shape[0]
if duplicates_removed > 0:
    print(f"Removed {duplicates_removed} duplicate rows from 'train_df'.")
else:
    print("No duplicate rows found in 'train_df'.")

# --- Process Test Data ---
# Store the 'id' column from the test data separately.
# This 'id' column is crucial for creating the submission file later.
# We will drop it from the test_df_features to use for prediction, but keep it in 'test_ids'.
if 'id' in test_df.columns:
    test_ids = test_df['id'] # Store the 'id' column
    # Create a copy of the test DataFrame without the 'id' column, which will be used for features.
    test_df_features = test_df.drop('id', axis=1)
    print("Stored 'id' column from 'test_df' for submission and created 'test_df_features'.")
else:
    test_ids = None # In case 'test_df' unexpectedly lacks an 'id' column
    test_df_features = test_df.copy() # Make a copy without dropping 'id' if it's not there.
    print("'id' column not found in 'test_df'. No 'id' stored for submission.")

# --- Define Features (X) and Target (y) ---
# X will contain all features (independent variables) for training.
# We drop 'FloodProbability' because it is our target variable.
X = train_df.drop('FloodProbability', axis=1)
# y will contain the target variable (dependent variable).
y = train_df['FloodProbability']

# Ensure that the feature columns in the test set match the training set.
# This is a good practice to prevent errors if columns differ between datasets.
common_features = list(X.columns) # Get the list of feature column names from the training data
test_df_features = test_df_features[common_features] # Select only these columns in the test features

print(f"\nTraining features (X) shape: {X.shape}")
print(f"Training target (y) shape: {y.shape}")
print(f"Test features for final prediction ('test_df_features') shape: {test_df_features.shape}")

print("\n--- Part 2 Complete ---")


# Part 3: Feature Scaling (Revised for Pipeline)

print("\n--- Part 3 (Revised): Feature Scaling Setup for Pipelines ---")

# Initialize the StandardScaler, but DO NOT fit it on X here.
# It will be fitted *within* each cross-validation fold by the Pipeline.
scaler = StandardScaler()
print("StandardScaler initialized (will be fitted within cross-validation pipeline).")

# The test data still needs to be scaled by a scaler fitted on the full *training data*.
# So, we'll fit a scaler on 'X' here, just for the final prediction on 'test_df_features'.
# This scaler is distinct from the one used inside the CV pipeline for evaluation realism.
final_scaler_for_test = StandardScaler()
final_scaler_for_test.fit(X) # Fit this specific scaler on the ENTIRE training data X
test_df_features_scaled = final_scaler_for_test.transform(test_df_features) # Transform test features
print("Test features scaled using a scaler fitted on the full training dataset.")
print(f"Shape of scaled test features: {test_df_features_scaled.shape}")

print("\n--- Part 3 Complete (Revised) ---")


print("\n--- Part 4 (Modified): Linear Regression Model Definition and CV Setup ---")

from sklearn.pipeline import Pipeline # Import Pipeline

# Define ONLY the Linear Regression model within a Pipeline
models = {
    "Linear Regression": Pipeline([('scaler', StandardScaler()), ('regressor', LinearRegression())])
}
print(f"Defined {len(models)} model: Linear Regression.")

# --- K-Fold Cross-Validation Strategy ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print(f"K-Fold Cross-Validation set up with {kf.n_splits} splits.")

# --- Define Scorers for Cross-Validation ---
scorers = {
    'neg_mae': make_scorer(mean_absolute_error, greater_is_better=False),
    'neg_root_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False, squared=False),
    'r2': make_scorer(r2_score, greater_is_better=True)
}
print("Evaluation scorers for cross-validation defined.")

print("\n--- Part 4 Complete (Modified) ---")



print("\n--- Part 5 (Modified): Performing Cross-Validation and Training Final Linear Regression Model ---")

results_cv = {}
predictions_for_submission = {}

# Since we only have one model, we can directly access it from the 'models' dictionary
model_name = "Linear Regression"
model_pipeline = models[model_name] # Get the Linear Regression Pipeline object

print(f"\n--- Processing {model_name} ---")

# --- Cross-Validation Evaluation ---
print(f"Performing {kf.n_splits}-Fold Cross-Validation for {model_name}...")
mae_scores = cross_val_score(model_pipeline, X, y, cv=kf, scoring=scorers['neg_mae'], n_jobs=-1)
rmse_scores = cross_val_score(model_pipeline, X, y, cv=kf, scoring=scorers['neg_root_mean_squared_error'], n_jobs=-1)
r2_scores = cross_val_score(model_pipeline, X, y, cv=kf, scoring=scorers['r2'], n_jobs=-1)

avg_mae = -mae_scores.mean()
avg_rmse = -rmse_scores.mean()
avg_r2 = r2_scores.mean()

results_cv[model_name] = {"MAE": avg_mae, "RMSE": avg_rmse, "R2": avg_r2}

print(f"{model_name} - Cross-Validation Metrics (Mean over {kf.n_splits} folds):")
print(f"  MAE: {avg_mae:.4f}")
print(f"  RMSE: {avg_rmse:.4f}")
print(f"  R2 Score: {avg_r2:.4f}")

# --- Train Final Model on FULL Training Data ---
print(f"Training final {model_name} model on full training dataset for submission...")
model_pipeline.fit(X, y) # Fit the pipeline on the full original training data

# Make predictions on the pre-scaled actual test.csv data
final_test_predictions = model_pipeline.predict(test_df_features) # Pass UNscaled test features here

predictions_for_submission[model_name] = final_test_predictions
print(f"Predictions for {model_name} on test data generated.")

print("\n--- Part 5 Complete (Modified): Linear Regression Cross-Validated & Final Model Trained ---")



# Part 6: Summarize Results and Create Submission File

print("\n--- Part 6: Summarizing Results and Creating Submission File ---")

# Convert the cross-validation results dictionary into a Pandas DataFrame for easy viewing.
results_cv_df = pd.DataFrame(results_cv).T # .T transposes the DataFrame for better readability
print("\n--- Model Performance Summary (Cross-Validation) ---")
print(results_cv_df.round(4)) # Display results rounded to 4 decimal places

# --- Create Submission File for the Best Model ---
# Determine the best model based on the lowest average RMSE from cross-validation.
# 'idxmin()' finds the index (model name) of the minimum value in the 'RMSE' column.
best_model_name_cv = results_cv_df['RMSE'].idxmin()
# Retrieve the predictions generated by this best model on the original test_df_features.
best_model_predictions_cv = predictions_for_submission[best_model_name_cv]

print(f"\nBased on Cross-Validation RMSE, the best model is: {best_model_name_cv}")

# Create the final submission DataFrame as required by Kaggle.
# It needs 'id' from the original test_df and 'FloodProbability' (your predictions).
submission_df_best_cv = pd.DataFrame({
    'id': test_ids,
    'FloodProbability': best_model_predictions_cv
})

# Ensure the predicted probabilities are within the valid range of 0 to 1.
# Some regression models can predict values slightly outside this range.
submission_df_best_cv['FloodProbability'] = np.clip(submission_df_best_cv['FloodProbability'], 0, 1)

# Save the submission DataFrame to a CSV file named 'submission.csv'.
# 'index=False' prevents Pandas from writing the DataFrame index as a column in the CSV.
submission_df_best_cv.to_csv('submission.csv', index=False)
print(f"Final submission file 'submission.csv' (from {best_model_name_cv}) created successfully.")

# Display the first few rows of the generated submission file for verification.
print("\n--- Head of the generated 'submission.csv' file: ---")
print(submission_df_best_cv.head())

# --- Optional: Save the Best Model and Scaler ---
# It's good practice to save your trained model and the scaler.
# This allows you to load them later to make predictions on new data without retraining.
best_model_for_saving = models[best_model_name_cv] # Get the actual model object itself
joblib.dump(best_model_for_saving, f'best_model_cv_{best_model_name_cv.replace(" ", "_").lower()}.pkl')
joblib.dump(scaler, 'scaler.pkl') # Save the scaler that was fitted on your training data
print(f"\nBest CV model ({best_model_name_cv}) and scaler saved for future use.")

print("\n--- Part 6 Complete & Project Finished ---")

