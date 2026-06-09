# Install required packages
!pip install kaggle pandas numpy matplotlib seaborn plotly


# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Set style for plots
plt.style.use('seaborn')
sns.set_palette('husl')


%cd /kaggle/input/playground-series-s5e4/



# Error handling for file operations
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    print('Training set shape:', train_df.shape)
    print('Test set shape:', test_df.shape)
except FileNotFoundError as e:
    print(f'Error: {e}')
    print('Please ensure the CSV files are in the correct directory.')


# Display basic information about the training dataset
print("\nTraining Dataset Info:")
train_df.info()

print("\nFirst few rows of training data:")
train_df.head()


# Basic statistical description
print("\nStatistical Description of Training Data:")
train_df.describe()


# Check for missing values
print("\nMissing Values in Training Data:")
train_df.isnull().sum()


# Function to plot numerical features distribution
def plot_numerical_distributions(df, numerical_cols):
    n_cols = 2
    n_rows = (len(numerical_cols) + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        sns.histplot(data=df, x=col, ax=axes[idx], kde=True)
        axes[idx].set_title(f'Distribution of {col}')
    
    plt.tight_layout()
    plt.show()


train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import optuna
import warnings
from xgboost import XGBRegressor


# Drop unnecessary columns
columns_to_drop = ['Podcast_Name', 'Episode_Title', 'id']
columns_to_drop1 = ['Podcast_Name', 'Episode_Title']
train_df = train_df.drop(columns_to_drop, axis=1)
test_df = test_df.drop(columns_to_drop1, axis=1)

# Handle missing values first
# Fill missing values in Episode_Length_minutes with median by Genre
length_medians = train_df.groupby('Genre')['Episode_Length_minutes'].transform('median')
train_df['Episode_Length_minutes'] = train_df['Episode_Length_minutes'].fillna(length_medians)
test_df['Episode_Length_minutes'] = test_df.groupby('Genre')['Episode_Length_minutes'].transform(lambda x: x.fillna(x.median()))

# Fill missing values in Guest_Popularity_percentage with 0 (assuming no guest means 0 popularity)
train_df['Guest_Popularity_percentage'] = train_df['Guest_Popularity_percentage'].fillna(0)
test_df['Guest_Popularity_percentage'] = test_df['Guest_Popularity_percentage'].fillna(0)

# Fill missing values in Number_of_Ads with mode
train_df['Number_of_Ads'] = train_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].mode()[0])
test_df['Number_of_Ads'] = test_df['Number_of_Ads'].fillna(test_df['Number_of_Ads'].mode()[0])

# Create binary features for popular hosts and guests
train_df['is_popular_host'] = (train_df['Host_Popularity_percentage'] > 70).astype(int)
train_df['is_popular_guest'] = (train_df['Guest_Popularity_percentage'] > 70).astype(int)

test_df['is_popular_host'] = (test_df['Host_Popularity_percentage'] > 70).astype(int)
test_df['is_popular_guest'] = (test_df['Guest_Popularity_percentage'] > 70).astype(int)

# One-hot encode categorical features
categorical_features = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# Get dummy variables for each categorical feature
for feature in categorical_features:
    # Fit on both train and test to ensure all categories are captured
    all_categories = pd.concat([train_df[feature], test_df[feature]]).unique()
    
    # Create dummy variables
    train_dummies = pd.get_dummies(train_df[feature], prefix=feature)
    test_dummies = pd.get_dummies(test_df[feature], prefix=feature)
    
    # Add missing columns to test set
    for col in train_dummies.columns:
        if col not in test_dummies.columns:
            test_dummies[col] = 0
    
    # Add missing columns to train set
    for col in test_dummies.columns:
        if col not in train_dummies.columns:
            train_dummies[col] = 0
    
    # Ensure columns are in the same order
    test_dummies = test_dummies[train_dummies.columns]
    
    # Add to dataframes
    train_df = pd.concat([train_df, train_dummies], axis=1)
    test_df = pd.concat([test_df, test_dummies], axis=1)
    
    # Drop original categorical column
    train_df = train_df.drop(feature, axis=1)
    test_df = test_df.drop(feature, axis=1)

# Scale numerical features
numerical_features = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                     'Guest_Popularity_percentage', 'Number_of_Ads']

scaler = StandardScaler()
train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
test_df[numerical_features] = scaler.transform(test_df[numerical_features])

print('Final feature set shape:', train_df.shape)
print('\nFeature names:')
print(train_df.columns.tolist())


# Install required ML packages
!pip install scikit-learn lightgbm xgboost optuna


# Separate features and target
X = train_df.drop(['Listening_Time_minutes'], axis=1)
y = train_df['Listening_Time_minutes']

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print('Training set shape:', X_train.shape)
print('Validation set shape:', X_val.shape)


train_df.head()


# Updated evaluation function to include MAE
from sklearn.metrics import mean_absolute_error

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f'{model_name} Results:')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R2 Score: {r2:.4f}\n')


# # LightGBM model with Optuna optimization and GPU support
# def objective(trial):
#     params = {
#         'objective': 'regression',
#         'metric': 'rmse',
#         'verbosity': -1,
#         'boosting_type': 'gbdt',
#         'device': 'gpu',  # Enable GPU
#         'gpu_platform_id': 0,
#         'gpu_device_id': 0,
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 100),
#         'max_depth': trial.suggest_int('max_depth', 3, 12),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
#     }
    
#     model = lgb.LGBMRegressor(**params)
#     scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
#     return -scores.mean()

# # Run Optuna optimization
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=2)

# # Get best parameters
# best_params = study.best_params
# best_params.update({
#     'device': 'gpu',
#     'gpu_platform_id': 0,
#     'gpu_device_id': 0
# })
# # print('Best parameters:', best_params)


# Best parameters from Optuna optimization (Trial 22)
best_params = {
    'n_estimators': 916,
    'learning_rate': 0.08366680842136756,
    'num_leaves': 99,
    'max_depth': 11,
    'min_child_samples': 58,
    'subsample': 0.8939146369158769,
    'colsample_bytree': 0.9111777868466708,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt'
}


# Train final model with best parameters
lgb_model = lgb.LGBMRegressor(**best_params)
lgb_model.fit(X_train, y_train)

# Make predictions
lgb_train_pred = lgb_model.predict(X_train)
lgb_val_pred = lgb_model.predict(X_val)

# Evaluate model
print('Training Results:')
evaluate_model(y_train, lgb_train_pred, 'LightGBM')
print('Validation Results:')
evaluate_model(y_val, lgb_val_pred, 'LightGBM')


from xgboost import XGBRegressor
# Define the objective function for Optuna
def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42,
        'objective': 'reg:squarederror',  # Standard regression objective
        'eval_metric': 'rmse'             # Root mean squared error metric
    }
    
    xgb_model = XGBRegressor(**params)
    scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    return -scores.mean()

# Run Optuna optimization
xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(xgb_objective, n_trials=30)

# Get the best parameters
xgb_best_params = xgb_study.best_params
print('Best parameters for XGBoost:', xgb_best_params)


# Train XGBoost model with best parameters
xgb_model = XGBRegressor(**xgb_best_params)
xgb_model.fit(X_train, y_train)

# Make predictions
xgb_train_pred = xgb_model.predict(X_train)
xgb_val_pred = xgb_model.predict(X_val)

# Evaluate XGBoost model
print('XGBoost Training Results:')
evaluate_model(y_train, xgb_train_pred, 'XGBoost')
print('XGBoost Validation Results:')
evaluate_model(y_val, xgb_val_pred, 'XGBoost')


# Evaluate both LightGBM and XGBoost models
# LightGBM Evaluation
print('LightGBM Training Results:')
evaluate_model(y_train, lgb_train_pred, 'LightGBM')
print('LightGBM Validation Results:')
evaluate_model(y_val, lgb_val_pred, 'LightGBM')

# XGBoost Evaluation
print('XGBoost Training Results:')
evaluate_model(y_train, xgb_train_pred, 'XGBoost')
print('XGBoost Validation Results:')
evaluate_model(y_val, xgb_val_pred, 'XGBoost')

# Combined Residuals Analysis
plt.figure(figsize=(15, 10))

# LightGBM Residuals
plt.subplot(221)
sns.histplot(lgb_train_pred - y_train, kde=True)
plt.title('LightGBM Training Residuals Distribution')
plt.xlabel('Residuals')

plt.subplot(222)
sns.histplot(lgb_val_pred - y_val, kde=True)
plt.title('LightGBM Validation Residuals Distribution')
plt.xlabel('Residuals')

# XGBoost Residuals
plt.subplot(223)
sns.histplot(xgb_train_pred - y_train, kde=True)
plt.title('XGBoost Training Residuals Distribution')
plt.xlabel('Residuals')

plt.subplot(224)
sns.histplot(xgb_val_pred - y_val, kde=True)
plt.title('XGBoost Validation Residuals Distribution')
plt.xlabel('Residuals')

plt.tight_layout()
plt.show()

# Combined Scatter Plots
plt.figure(figsize=(15, 10))

# LightGBM Scatter Plots
plt.subplot(221)
plt.scatter(lgb_train_pred, y_train, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.title('LightGBM Training: Predicted vs Actual')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')

plt.subplot(222)
plt.scatter(lgb_val_pred, y_val, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.title('LightGBM Validation: Predicted vs Actual')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')

# XGBoost Scatter Plots
plt.subplot(223)
plt.scatter(xgb_train_pred, y_train, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.title('XGBoost Training: Predicted vs Actual')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')

plt.subplot(224)
plt.scatter(xgb_val_pred, y_val, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.title('XGBoost Validation: Predicted vs Actual')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')

plt.tight_layout()
plt.show()


%cd /kaggle/working


# Prepare test data
test_features = test_df.drop(['id'], axis=1)

# Make predictions
test_predictions = lgb_model.predict(test_features)

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': test_predictions
})

submission.to_csv('lgb_submission.csv', index=False)
print('lightGBM Submission file created!')


# Make predictions on test set for XGBoost
xgb_test_predictions = xgb_model.predict(test_features)

# Create submission file for XGBoost
xgb_submission = pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': xgb_test_predictions
})

xgb_submission.to_csv('xgb_submission.csv', index=False)
print('XGBoost submission file created!')

