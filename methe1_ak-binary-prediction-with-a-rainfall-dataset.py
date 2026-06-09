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


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)


# Load the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')  # Replace with your file path if needed
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

# Display basic info
print("Train Data Info:")
print(train_df.info())
print("\nTest Data Info:")
print(test_df.info())
print("\nSample Submission Head:")
print(sample_submission.head())


# Check for missing values
print("Missing values in train:", train_df.isnull().sum())
print("Missing values in test:", test_df.isnull().sum())

# Impute missing winddirection in test (id 2707)
test_df['winddirection'].fillna(train_df['winddirection'].median(), inplace=True)

# Combine train and test for consistent preprocessing (exclude rainfall)
all_data = pd.concat([train_df.drop('rainfall', axis=1), test_df], axis=0)

# Feature Engineering
# Cyclical encoding for day and winddirection
all_data['day_sin'] = np.sin(2 * np.pi * all_data['day'] / 365)
all_data['day_cos'] = np.cos(2 * np.pi * all_data['day'] / 365)
all_data['winddir_sin'] = np.sin(2 * np.pi * all_data['winddirection'] / 360)
all_data['winddir_cos'] = np.cos(2 * np.pi * all_data['winddirection'] / 360)

# Derived features
all_data['temp_range'] = all_data['maxtemp'] - all_data['mintemp']
all_data['humid_dew_diff'] = all_data['humidity'] - all_data['dewpoint']

# Split back into train and test
train_processed = all_data.iloc[:len(train_df)].copy()
test_processed = all_data.iloc[len(train_df):].copy()
train_processed['rainfall'] = train_df['rainfall']

# Features to scale (original features + derived, no duplicates)
features_to_scale = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
                     'humidity', 'cloud', 'sunshine', 'windspeed', 'temp_range', 'humid_dew_diff']

scaler = StandardScaler()
train_processed[features_to_scale] = scaler.fit_transform(train_processed[features_to_scale])
test_processed[features_to_scale] = scaler.transform(test_processed[features_to_scale])

# Final feature set (cyclical features + scaled features, no duplicates)
features = ['day_sin', 'day_cos', 'winddir_sin', 'winddir_cos'] + features_to_scale

print("Processed Train Head:")
print(train_processed[features + ['rainfall']].head())
print("Processed Test Head:")
print(test_processed[features].head())

# Verify uniqueness of feature names
print("Feature names:", features)
print("Number of unique features:", len(set(features)))


# Rainfall distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='rainfall', data=train_processed)
plt.title('Rainfall Distribution')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(train_processed[features + ['rainfall']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Example: Humidity vs Rainfall
plt.figure(figsize=(6, 4))
sns.boxplot(x='rainfall', y='humidity', data=train_processed)
plt.title('Humidity vs Rainfall')
plt.show()


# Prepare training data
X = train_processed[features]
y = train_processed['rainfall']

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)

# Validate
y_pred_proba = xgb_model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_pred_proba)
print(f"Validation ROC AUC: {roc_auc:.4f}")

# Cross-validation
cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='roc_auc')
print(f"5-Fold CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


# Plot feature importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model, max_num_features=10)
plt.title('Feature Importance')
plt.show()


# Predict probabilities for test set
X_test = test_processed[features]
test_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Prepare submission
submission = sample_submission.copy()
submission['rainfall'] = test_pred_proba

# Save to CSV
submission.to_csv('xg_submission.csv', index=False)
print("Submission file saved as 'submission.csv'")
print(submission.head())


from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200]
}

# Grid search
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X, y)

# Best model
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best ROC AUC: {grid_search.best_score_:.4f}")

# Update model with best parameters
xgb_model = grid_search.best_estimator_


from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,  # Number of trees
    max_depth=10,      # Limit depth to prevent overfitting
    random_state=42,
    n_jobs=-1          # Use all available cores
)

# Prepare training data (same as XGBoost)
X = train_processed[features]
y = train_processed['rainfall']

# Split for validation (same split as XGBoost for fair comparison)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Validate
rf_pred_proba = rf_model.predict_proba(X_val)[:, 1]
rf_roc_auc = roc_auc_score(y_val, rf_pred_proba)
print(f"Random Forest Validation ROC AUC: {rf_roc_auc:.4f}")

# Cross-validation
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"Random Forest 5-Fold CV ROC AUC: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")

# Compare with XGBoost
print(f"\nXGBoost Best CV ROC AUC (from tuning): {grid_search.best_score_:.4f}")


# Plot feature importance for Random Forest
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(rf_model.feature_importances_, index=features)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# Predict probabilities for test set with Random Forest
X_test = test_processed[features]
rf_test_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Prepare submission
rf_submission = sample_submission.copy()
rf_submission['rainfall'] = rf_test_pred_proba

# Save to CSV
rf_submission.to_csv('submission.csv', index=False)
print("Random Forest submission saved as 'rf_submission.csv'")
print(rf_submission.head())


# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5]
}

# Grid search
rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
rf_grid_search.fit(X, y)

# Best model
print(f"Best Random Forest Parameters: {rf_grid_search.best_params_}")
print(f"Best Random Forest ROC AUC: {rf_grid_search.best_score_:.4f}")

# Update model with best parameters
rf_model = rf_grid_search.best_estimator_

