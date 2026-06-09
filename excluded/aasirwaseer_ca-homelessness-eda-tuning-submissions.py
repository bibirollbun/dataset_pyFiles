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


# Set display options for better viewing
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load the datasets
# Make sure the CSV files are in the same directory as your script/notebook.
try:
    train_df = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')
    test_df = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')
    sample_submission_df = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/sample_submission.csv')
except FileNotFoundError as e:
    print(e)
    print("\nPlease ensure 'train.csv', 'test.csv', and 'sample_submission.csv' are downloaded and in the correct directory.")
    # In a real script, you might exit here. For collaboration, we'll assume the user will fix the path.
    train_df = pd.DataFrame() # Create empty dataframe to avoid further errors


if not train_df.empty:
    # --- Inspect the Training Data ---
    print("--- Training Data Info ---")
    train_df.info()

    print("\n--- First 5 Rows of Training Data ---")
    print(train_df.head())

    print("\n--- Summary Statistics for Training Data ---")
    print(train_df.describe())

    print("\n--- Missing Values in Training Data ---")
    print(train_df.isnull().sum())

    print("\n--- Test Data Shape ---")
    print(test_df.shape)

    print("\n--- Sample Submission Format ---")
    print(sample_submission_df.head())


import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Visualize the Target Variable Distribution ---
print("--- Target Variable Analysis ---")
plt.figure(figsize=(14, 6))

# Plot original distribution
plt.subplot(1, 2, 1)
sns.histplot(train_df['HOMELESS_RATE'], kde=True, bins=30)
plt.title('Original Distribution of HOMELESS_RATE')

# Plot log-transformed distribution
# We use np.log1p which calculates log(1+x) to handle cases where the rate is 0.
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(train_df['HOMELESS_RATE']), kde=True, bins=30)
plt.title('Log-Transformed Distribution of HOMELESS_RATE')

plt.suptitle('Target Variable Distribution Analysis', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# --- 2. Check for Redundant Columns ---
print("\n--- Redundancy Checks ---")
# Check 1: Veteran vs. Non-Veteran
# We expect the sum to be constant if they are perfectly complementary.
# The small variance is due to floating point precision.
vet_sum_std = (train_df['VETERAN_POP_PCT'] + train_df['NONVETERAN_POP_PCT']).std()
print(f"Standard deviation of (Veteran + Non-Veteran): {vet_sum_std:.6f}")

# Check 2: Disability vs. No-Disability
dis_sum_std = (train_df['DISABILITY_POP_PCT'] + train_df['NODISABILITY_POP_PCT']).std()
print(f"Standard deviation of (Disability + No-Disability): {dis_sum_std:.6f}")

# Check 3: Age U18 vs Family Members U18
age_family_diff = (train_df['AGE_U18_PCT'] - train_df['FAMILY_MEMBERS_UNDER_18_PCT']).abs().sum()
print(f"Absolute difference between AGE_U18_PCT and FAMILY_MEMBERS_UNDER_18_PCT: {age_family_diff:.6f}")
print("Based on these checks, we can likely drop 'NONVETERAN_POP_PCT', 'NODISABILITY_POP_PCT', and 'FAMILY_MEMBERS_UNDER_18_PCT'.")


# --- 3. Visualize Feature Correlation ---
print("\n--- Correlation Analysis ---")
# Drop the ID column for correlation calculation
corr_matrix = train_df.drop('ID', axis=1).corr()

plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title('Feature Correlation Matrix', fontsize=16)
plt.show()

# Display correlations with the target variable
print("\n--- Top 10 Features Most Correlated with HOMELESS_RATE ---")
print(corr_matrix['HOMELESS_RATE'].abs().sort_values(ascending=False).head(11))


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# --- 1. Preprocessing ---

# Apply the log transformation to the target variable
# This creates a more normal distribution for the model to learn from.
train_df['HOMELESS_RATE_LOG'] = np.log1p(train_df['HOMELESS_RATE'])

# Define features (X) and target (y)
features = train_df.drop([
    'ID',
    'HOMELESS_RATE',
    'HOMELESS_RATE_LOG',
    # Drop redundant/collinear columns identified in Step 2
    'FAMILY_MEMBERS_UNDER_18_PCT', # Identical to AGE_U18_PCT
    'NONVETERAN_POP_PCT',          # Highly correlated with VETERAN_POP_PCT
    'NODISABILITY_POP_PCT'         # Highly correlated with DISABILITY_POP_PCT
], axis=1)

target = train_df['HOMELESS_RATE_LOG']

# --- 2. Data Splitting ---

# Split data into training and validation sets (80/20 split)
# This lets us evaluate how well our model generalizes to new, unseen data.
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")


# --- 3. Baseline Model (Ridge Regression) ---

# Initialize and train the model. Ridge is a good baseline because it's
# robust to features that are still somewhat correlated.
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train, y_train)


# --- 4. Evaluation ---

# Make predictions on the validation set
log_predictions = model.predict(X_val)

# IMPORTANT: Inverse transform the predictions to get back to the original scale
# np.expm1 is the inverse of np.log1p
original_scale_predictions = np.expm1(log_predictions)
original_scale_y_val = np.expm1(y_val)

# Calculate the Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(original_scale_y_val, original_scale_predictions))

print(f"\nBaseline Model Validation RMSE: {rmse:.6f}")


import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Train the LightGBM Model ---

# We use the same data splits (X_train, X_val, etc.) from Step 3 for a fair comparison.
lgbm = lgb.LGBMRegressor(random_state=42)
lgbm.fit(X_train, y_train)

# --- 2. Evaluation ---

# Make predictions and inverse transform them
lgbm_log_predictions = lgbm.predict(X_val)
lgbm_original_scale_predictions = np.expm1(lgbm_log_predictions)

# Calculate the new RMSE
lgbm_rmse = np.sqrt(mean_squared_error(original_scale_y_val, lgbm_original_scale_predictions))

print(f"Baseline Model Validation RMSE (from Step 3): {rmse:.6f}")
print(f"LightGBM Model Validation RMSE: {lgbm_rmse:.6f}")

if lgbm_rmse < rmse:
    print("\nSuccess! The LightGBM model performed better than the baseline.")
else:
    print("\nThe LightGBM model did not outperform the baseline. This can happen with small datasets.")

# --- 3. Feature Importance ---

# Create a DataFrame for feature importances
feature_importances = pd.DataFrame({
    'feature': features.columns,
    'importance': lgbm.feature_importances_
}).sort_values('importance', ascending=False)

# Plot the top 15 features
plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importances.head(15))
plt.title('Top 15 Feature Importances (LightGBM)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# --- 1. Final Model Selection ---
# Based on our validation scores, the Ridge model performed best.
final_model = Ridge(alpha=1.0, random_state=42)

# --- 2. Retrain on Full Data ---
# We use the 'features' and 'target' DataFrames defined in Step 3,
# which contain all 130 rows of the preprocessed training data.
final_model.fit(features, target)
print("Final model trained on the full training dataset.")

# --- 3. Prepare Test Data ---
# It is CRITICAL to process the test data exactly like the training data.
test_ids = test_df['ID']
test_features = test_df.drop([
    'ID',
    # Drop the same columns we dropped from the training set
    'NONVETERAN_POP_PCT',
    'NODISABILITY_POP_PCT',
    'FAMILY_MEMBERS_UNDER_18_PCT'
], axis=1)

# Ensure the column order is the same
test_features = test_features[features.columns]


# --- 4. Make Final Predictions ---
final_log_predictions = final_model.predict(test_features)

# Inverse transform to get the original scale
final_predictions = np.expm1(final_log_predictions)

# Handle any potential negative predictions (rare, but good practice)
final_predictions[final_predictions < 0] = 0


# --- 5. Create Submission File ---
submission_df = pd.DataFrame({
    'ID': test_ids,
    'HOMELESS_RATE': final_predictions
})

submission_df.to_csv('submission_1.csv', index=False)

print("\n'submission.csv' has been created successfully!")
print("\n--- First 5 Rows of Submission File ---")
print(submission_df.head())


from sklearn.model_selection import KFold, cross_val_score

# --- 1. Setup ---
# We'll use the same 'features' and 'target' (log-transformed) from before.
# The model is our best one so far: Ridge Regression.
model = Ridge(alpha=1.0, random_state=42)

# Configure the cross-validation procedure.
# We'll split the data into 5 folds and shuffle it for randomness.
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# --- 2. Run Cross-Validation ---
# 'cross_val_score' automates the process of splitting, training, and evaluating.
# We use 'neg_root_mean_squared_error' because the function expects a score where higher is better.
# It will return negative RMSE values, so we'll flip the sign back to positive.
scores = cross_val_score(model, features, target, cv=cv, scoring='neg_root_mean_squared_error')

# Convert scores to positive RMSE values
rmse_scores = -scores

# --- 3. Display Results ---
print(f"Scores for each of the 5 folds: {rmse_scores}")
print(f"Average CV RMSE: {rmse_scores.mean():.6f}")
print(f"Standard Deviation of CV RMSE: {rmse_scores.std():.6f}")


from sklearn.model_selection import GridSearchCV
import numpy as np

# --- 1. Setup the Grid Search ---
model = Ridge(random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the range of alpha values to test.
# np.logspace creates a range of numbers spaced evenly on a log scale.
param_grid = {
    'alpha': np.logspace(-3, 3, 100) # Test 100 values from 0.001 to 1000
}

# --- 2. Run the Grid Search ---
# GridSearchCV will test every alpha in our param_grid using 5-fold cross-validation.
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1 # Use all available CPU cores to speed up the process
)

# Fit the grid search to the entire dataset.
grid_search.fit(features, target)


# --- 3. Display Results ---
best_alpha = grid_search.best_params_['alpha']
best_rmse = -grid_search.best_score_

print(f"The best alpha found was: {best_alpha:.4f}")
print(f"The best cross-validated RMSE with this alpha is: {best_rmse:.6f}")


# --- 1. Initialize Final Tuned Model ---
# We use the best alpha found by GridSearchCV.
best_alpha = 0.9326
final_tuned_model = Ridge(alpha=best_alpha, random_state=42)

# --- 2. Retrain on the Full Dataset ---
# We train the optimized model on all available training data.
final_tuned_model.fit(features, target)
print("Final tuned model trained on the full training dataset.")

# --- 3. Prepare Test Data ---
# The test data preparation is the same as in Step 5.
test_ids = test_df['ID']
test_features = test_df.drop([
    'ID',
    'NONVETERAN_POP_PCT',
    'NODISABILITY_POP_PCT',
    'FAMILY_MEMBERS_UNDER_18_PCT'
], axis=1)
test_features = test_features[features.columns]

# --- 4. Make Final Predictions ---
final_log_predictions = final_tuned_model.predict(test_features)
final_predictions = np.expm1(final_log_predictions)
final_predictions[final_predictions < 0] = 0

# --- 5. Create New Submission File ---
submission_df_tuned = pd.DataFrame({
    'ID': test_ids,
    'HOMELESS_RATE': final_predictions
})

# Save to a new file to avoid overwriting our first one.
submission_df_tuned.to_csv('submission.csv', index=False)

print("\n'submission_tuned.csv' has been created successfully!")
print("\n--- First 5 Rows of the New Submission File ---")
print(submission_df_tuned.head())


from sklearn.preprocessing import PolynomialFeatures

# --- 1. Create Polynomial Features ---
# We will generate features up to the 2nd degree (e.g., x^2, x*y).
# include_bias=False prevents adding a constant column of ones.
poly = PolynomialFeatures(degree=2, include_bias=False)

# Fit and transform our original features
features_poly = poly.fit_transform(features)

print(f"Original number of features: {features.shape[1]}")
print(f"Number of features after PolynomialFeatures: {features_poly.shape[1]}")


# --- 2. Re-evaluate Model with New Features ---
# We use our best tuned model from the previous step.
best_alpha = 0.9326
tuned_model = Ridge(alpha=best_alpha, random_state=42)

# Run the same 5-fold cross-validation on the new feature set
cv = KFold(n_splits=5, shuffle=True, random_state=42)
new_scores = cross_val_score(
    tuned_model,
    features_poly, # Use the new polynomial features
    target,
    cv=cv,
    scoring='neg_root_mean_squared_error'
)

# Convert scores to positive RMSE
new_rmse_scores = -new_scores
new_avg_rmse = new_rmse_scores.mean()

print(f"\nPrevious best CV RMSE: {best_rmse:.6f}")
print(f"New CV RMSE with Polynomial Features: {new_avg_rmse:.6f}")

if new_avg_rmse < best_rmse:
    print("\nSuccess! The new features improved our model's score.")
else:
    print("\nThe new features did not improve the model's score.")


from sklearn.model_selection import cross_val_predict

# --- 1. Get Cross-Validated Predictions from Both Models ---
# We will use the original feature set for this.

# Model 1: Our best tuned Ridge model
tuned_ridge = Ridge(alpha=0.9326, random_state=42)

# Model 2: Our LightGBM model
lgbm = lgb.LGBMRegressor(random_state=42)

# cross_val_predict returns the predictions for each data point when it was in the hold-out set.
# This prevents data leakage.
ridge_preds_cv = cross_val_predict(tuned_ridge, features, target, cv=cv)
lgbm_preds_cv = cross_val_predict(lgbm, features, target, cv=cv)


# --- 2. Average the Predictions ---
ensemble_preds_cv = (ridge_preds_cv + lgbm_preds_cv) / 2


# --- 3. Evaluate the Ensemble ---
# We calculate the RMSE of these averaged predictions against the true target.
# Remember, the target is still on the log scale.
ensemble_rmse = np.sqrt(mean_squared_error(target, ensemble_preds_cv))

print(f"Previous best CV RMSE (Ridge): {best_rmse:.6f}")
print(f"Ensemble Model CV RMSE: {ensemble_rmse:.6f}")

if ensemble_rmse < best_rmse:
    print("\nSuccess! The ensemble model performed better than the single best model.")
else:
    print("\nThe ensemble did not outperform the single best model.")




