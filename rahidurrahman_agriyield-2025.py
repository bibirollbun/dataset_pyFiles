#check the dataset
import pandas as pd

# Load the datasets from Kaggle input directory
train_df = pd.read_csv('/kaggle/input/agriyield-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/agriyield-2025/test.csv')

# Check the first few rows of the training dataset to understand its structure
print("Training Data Preview:")
print(train_df.head())

# Check for missing values in both train and test datasets
print("\nMissing Values in Training Data:")
print(train_df.isnull().sum())

print("\nMissing Values in Test Data:")
print(test_df.isnull().sum())

# Check data types of each column
print("\nData Types in Training Data:")
print(train_df.dtypes)

print("\nData Types in Test Data:")
print(test_df.dtypes)

# Check for duplicate rows
print("\nDuplicate Rows in Training Data:")
print(train_df.duplicated().sum())

print("\nDuplicate Rows in Test Data:")
print(test_df.duplicated().sum())

# Statistical summary of the numerical features
print("\nStatistical Summary of Training Data:")
print(train_df.describe())

# Check if any columns have unexpected data types or incorrect formats
# For example, 'field_id' should be categorical, and 'yield' should be numeric
print("\nChecking Data Integrity:")
if not pd.api.types.is_numeric_dtype(train_df['yield']):
    print("Warning: 'yield' column is not numeric!")

if not pd.api.types.is_numeric_dtype(train_df['soil_ph']):
    print("Warning: 'soil_ph' column is not numeric!")


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting histograms to visualize the distribution of key features
features = ['soil_ph', 'organic_matter', 'sand_pct', 'temperature', 'humidity', 'rainfall', 'ndvi', 'yield']
train_df[features].hist(bins=20, figsize=(15, 10))
plt.tight_layout()
plt.show()



# Step 1: Check for outliers in 'organic_matter' using IQR
Q1 = train_df['organic_matter'].quantile(0.25)
Q3 = train_df['organic_matter'].quantile(0.75)
IQR = Q3 - Q1

# Calculate the lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify the outliers in 'organic_matter'
organic_matter_outliers = train_df[(train_df['organic_matter'] < lower_bound) | (train_df['organic_matter'] > upper_bound)]
print(f"Organic Matter Outliers:")
print(organic_matter_outliers)

# Step 2: Check for outliers in 'ndvi' (should be between 0 and 1)
ndvi_outliers = train_df[(train_df['ndvi'] < 0) | (train_df['ndvi'] > 1)]
print(f"NDVI Outliers:")
print(ndvi_outliers)

# Optional: Remove outliers from 'organic_matter' and 'ndvi' if required
# Removing outliers
train_df_cleaned = train_df[(train_df['organic_matter'] >= lower_bound) & (train_df['organic_matter'] <= upper_bound)]
train_df_cleaned = train_df_cleaned[(train_df_cleaned['ndvi'] >= 0) & (train_df_cleaned['ndvi'] <= 1)]

# Checking the cleaned dataset
print(f"Cleaned Training Data:")
print(train_df_cleaned.head())


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, loguniform



# Prepare the feature matrix (X) and target vector (y)
X = train_df.drop(columns=['field_id', 'yield'])
y = train_df['yield']

# Scale the features for better performance of the SVM model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test_df.drop(columns=['field_id']))



# Split the data into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Check the shape of the resulting splits
print(f"Training Data Shape: {X_train.shape}")
print(f"Validation Data Shape: {X_val.shape}")



# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100, 1000, 2000],  # Regularization strength
    'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5],  # Epsilon for margin of error in regression
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]  # Kernel coefficient
}



# Initialize the SVM regression model
svm_model = SVR(kernel='rbf')

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, 
                           cv=5, scoring='neg_mean_squared_error', 
                           verbose=2, n_jobs=-1)

# Fit GridSearchCV on the training data
grid_search.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")



# Train the model with the best parameters
best_svm_model = grid_search.best_estimator_

# Make predictions on the validation set
y_pred_best = best_svm_model.predict(X_val)

# Calculate RMSE (Root Mean Squared Error) for model evaluation
rmse_best = mean_squared_error(y_val, y_pred_best, squared=False)
print(f"RMSE for Optimized SVM: {rmse_best}")



# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'C': loguniform(1e-2, 1e2),  # Log-uniform distribution for C
    'epsilon': uniform(0.01, 0.5),  # Uniform distribution for epsilon
    'gamma': ['scale', 'auto']  # Discrete values for gamma
}



# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=svm_model, param_distributions=param_dist, 
                                   n_iter=50, cv=5, scoring='neg_mean_squared_error', 
                                   verbose=2, n_jobs=-1, random_state=42)

# Fit RandomizedSearchCV on the training data
random_search.fit(X_train, y_train)

# Get the best parameters from the random search
best_random_params = random_search.best_params_
print(f"Best Hyperparameters from Random Search: {best_random_params}")



# Train the model with the best parameters found by RandomizedSearchCV
best_random_svm_model = random_search.best_estimator_

# Make predictions on the validation set
y_pred_random = best_random_svm_model.predict(X_val)

# Calculate RMSE for the optimized Random Search model
rmse_random = mean_squared_error(y_val, y_pred_random, squared=False)
print(f"RMSE for Random Search Optimized SVM: {rmse_random}")



# Train the best model (either from GridSearchCV or RandomizedSearchCV) on the entire dataset
best_final_svm_model = grid_search.best_estimator_  # Or random_search.best_estimator_

# Train on the full dataset
best_final_svm_model.fit(X_scaled, y)

# Predict on the test set
y_test_pred_final = best_final_svm_model.predict(X_test_scaled)

# Prepare the final submission
submission = pd.DataFrame({'field_id': test_df['field_id'], 'yield': y_test_pred_final})

# Save the submission file
submission.to_csv('final_submission_svm_optimized.csv', index=False)
print(f"Submission file created: submission.csv")


