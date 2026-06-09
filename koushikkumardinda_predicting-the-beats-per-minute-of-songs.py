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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


print("Loading datasets...")
try:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
    sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
except FileNotFoundError:
    print("Error: Ensure 'train.csv', 'test.csv', and 'sample_submission.csv' are in the same directory.")
    exit()

print("--- Data Overview ---")
print("Training data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("\nFirst 5 rows of the training data:")
print(train_df.head())
print("\nTraining data info:")
train_df.info()

print("\nMissing values in training data:")
print(train_df.isnull().sum().sum())
print("\nMissing values in test data:")
print(test_df.isnull().sum().sum())

# Analyze the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(train_df['BeatsPerMinute'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of BeatsPerMinute')
plt.xlabel('BeatsPerMinute')
plt.ylabel('Frequency')
plt.show()

# Display a correlation matrix to understand feature relationships
plt.figure(figsize=(12, 10))
corr_matrix = train_df.corr(numeric_only=True)
sns.heatmap(corr_matrix, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Features and Target')
plt.show()


# Separate features and target variable
X = train_df.drop(['id', 'BeatsPerMinute'], axis=1)
y = train_df['BeatsPerMinute']
X_test = test_df.drop('id', axis=1)
test_ids = test_df['id']

print("\n--- Preprocessing & Feature Engineering ---")

# A simple example of creating a new feature: ratio of two features
# This demonstrates the concept; feel free to add more complex features.
if 'feature_0' in X.columns and 'feature_1' in X.columns:
    X['feature_ratio'] = X['feature_0'] / (X['feature_1'] + 1e-6) # Add small epsilon to avoid division by zero
    X_test['feature_ratio'] = X_test['feature_0'] / (X_test['feature_1'] + 1e-6)
    print("Added a new feature 'feature_ratio'.")
else:
    print("Skipping 'feature_ratio' creation as required features were not found.")

# Scaling features can sometimes help models, especially linear ones.
# While tree-based models like XGBoost are not sensitive to scaling, it's a good practice.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
print("Features have been scaled using StandardScaler.")


# Using XGBoost Regressor as the primary model.
# Start with default parameters to get a baseline.
print("\n--- Model Training ---")
model = xgb.XGBRegressor(objective='reg:squarederror',
                         eval_metric='mae', # MAE is the competition metric
                         n_estimators=100,
                         random_state=42,
                         n_jobs=-1)

print("Training baseline XGBoost model...")
model.fit(X_scaled, y)
print("Baseline model training complete.")


print("\n--- Hyperparameter Tuning with GridSearchCV ---")
# Define a smaller, manageable parameter grid for demonstration.
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}

# Use KFold for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Set up GridSearchCV to find the best parameters.
# The scoring metric is negative mean absolute error because GridSearchCV maximizes the score.
grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring='neg_mean_absolute_error',
                           cv=kf,
                           n_jobs=-1,
                           verbose=1)

print("Running GridSearchCV...")
grid_search.fit(X_scaled, y)

print("\nBest hyperparameters found by GridSearchCV:")
print(grid_search.best_params_)
print("\nBest MAE score (negated):", -grid_search.best_score_)

# Use the best model from the grid search for final predictions
best_model = grid_search.best_estimator_
print("\nFinal model is the best estimator from the grid search.")


print("\n--- Making Predictions and Creating Submission File ---")
predictions = best_model.predict(X_test_scaled)

# Ensure predictions are non-negative, as BPM cannot be negative.
predictions[predictions < 0] = 0

submission_df = pd.DataFrame({'id': test_ids, 'BeatsPerMinute': predictions})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully.")
print("\nFirst 5 rows of the submission file:")
print(submission_df.head())

