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


# Import libraries
import numpy as np
import pandas as pd

# Load datasets
filePath = "/kaggle/input/playground-series-s5e9"
train = pd.read_csv(f'{filePath}/train.csv')
test = pd.read_csv(f'{filePath}/test.csv')
sample_submission = pd.read_csv(f'{filePath}/sample_submission.csv')

# Show the shape of the datasets
print('Train shape:', train.shape)
print('Test shape:', test.shape)
print('Sample submission shape:', sample_submission.shape)

# Display the first few rows of the training data
train.head()


# Show columns and data types
print(train.dtypes)

# Check for missing values
print('\nMissing values per column:')
print(train.isnull().sum())


# Summary statistics for numeric columns
train.describe()


import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of the target variable (assuming 'BeatsPerMinute' is the target column)
plt.figure(figsize=(8, 4))
sns.histplot(train['BeatsPerMinute'], kde=True, bins=30)
plt.title('Distribution of BPM (Target Variable)')
plt.xlabel('BPM')
plt.ylabel('Frequency')
plt.show()


# Plot distributions for a few numeric features and their relationship with BPM
numeric_features = train.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [f for f in numeric_features if f != 'BeatsPerMinute']  # Exclude target
sample_features = numeric_features[:3]  # Plot first 3 features as example

fig, axes = plt.subplots(len(sample_features), 2, figsize=(12, 4 * len(sample_features)))
for i, feature in enumerate(sample_features):
    # Distribution
    sns.histplot(train[feature], kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Distribution of {feature}')
    # Relationship with BeatsPerMinute
    sns.scatterplot(x=train[feature], y=train['BeatsPerMinute'], ax=axes[i, 1], alpha=0.3)
    axes[i, 1].set_title(f'{feature} vs BeatsPerMinute')
plt.tight_layout()
plt.show()


# Handle missing values (example: fill numeric with median, categorical with mode)
for col in train.columns:
    if train[col].isnull().sum() > 0:
        if train[col].dtype == 'object':
            mode = train[col].mode()[0]
            train[col].fillna(mode, inplace=True)
            test[col].fillna(mode, inplace=True)
        else:
            median = train[col].median()
            train[col].fillna(median, inplace=True)
            test[col].fillna(median, inplace=True)


# One-hot encode categorical variables
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
train_encoded = pd.get_dummies(train, columns=categorical_cols)
test_encoded = pd.get_dummies(test, columns=categorical_cols)

# Align train and test dataframes to have the same columns
train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)


from sklearn.preprocessing import StandardScaler

# Identify feature columns (exclude target and any ID columns)
target_col = 'BeatsPerMinute'  # Update if your target column is named differently
feature_cols = [col for col in train_encoded.columns if col != target_col]

scaler = StandardScaler()
train_encoded[feature_cols] = scaler.fit_transform(train_encoded[feature_cols])
test_encoded[feature_cols] = scaler.transform(test_encoded[feature_cols])


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


# Split features and target
X = train_encoded[feature_cols]
y = train_encoded[target_col]




# # Faster Random Forest for testing
# rf = RandomForestRegressor(n_estimators=10, random_state=42)

# # Faster cross-validation
# scores = cross_val_score(rf, X, y, cv=3, scoring='neg_root_mean_squared_error')
# print('Cross-validated RMSE (fast version):', -scores.mean())

# # Fit the model on the full training data
# rf.fit(X, y)

# # Predict on the test set
# test_preds = rf.predict(test_encoded[feature_cols])

# # Prepare the submission DataFrame (make sure the column name matches sample_submission)
# submission = sample_submission.copy()
# submission['BeatsPerMinute'] = test_preds  # Update column name if needed

# # Save to CSV for Kaggle submission
# submission.to_csv('submission.csv', index=False)
# print("Submission file 'submission.csv' created!")


# !pip install lightgbm


import lightgbm as lgb

# Create the LightGBM regressor
lgbm = lgb.LGBMRegressor(n_estimators=100, random_state=42)

# Cross-validation (same as before, 3 folds, negative RMSE)
lgbm_scores = cross_val_score(lgbm, X, y, cv=3, scoring='neg_root_mean_squared_error')
print('LightGBM Cross-validated RMSE:', -lgbm_scores.mean())


from sklearn.model_selection import RandomizedSearchCV

# Define parameter grid for LightGBM
param_dist = {
    'num_leaves': [15, 31, 50, 70, 100],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10, -1],
    'min_child_samples': [5, 10, 20, 30],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Set up the RandomizedSearchCV
lgbm_search = RandomizedSearchCV(
    estimator=lgb.LGBMRegressor(n_estimators=100, random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
 )

# Run the search
lgbm_search.fit(X, y)

print('Best parameters:', lgbm_search.best_params_)
print('Best cross-validated RMSE:', -lgbm_search.best_score_)


# Fit LightGBM on the full training data
lgbm.fit(X, y)

# Predict on the test set
lgbm_test_preds = lgbm.predict(test_encoded[feature_cols])

# Prepare the submission DataFrame (make sure the column name matches sample_submission)
lgbm_submission = sample_submission.copy()
lgbm_submission['BeatsPerMinute'] = lgbm_test_preds  # Update column name if needed

# Save to CSV for Kaggle submission
lgbm_submission.to_csv('submission.csv', index=False)
print("Submission file from lgbm model 'submission.csv' created!")




