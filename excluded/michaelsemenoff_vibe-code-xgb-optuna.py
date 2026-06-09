import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history
import warnings
warnings.filterwarnings('ignore')

# Load the data
train_df = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/train_df.csv')
test_df = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/test_df.csv')

# Convert datetime column to datetime type
train_df['datetime'] = pd.to_datetime(train_df['datetime'])
test_df['datetime'] = pd.to_datetime(test_df['datetime'])


# Basic data examination
print("Training Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)
print("\nTraining Data Info:")
print(train_df.info())
print("\nTraining Data Summary Statistics:")
print(train_df.describe())

# Check for missing values
print("\nMissing Values in Training Data:")
print(train_df.isnull().sum())
print("\nMissing Values in Test Data:")
print(test_df.isnull().sum())

# Plot the target variable
plt.figure(figsize=(15, 6))
plt.plot(train_df['datetime'], train_df['e_users'])
plt.title('E-commerce Users Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Users')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot distributions of features
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.histplot(train_df['promotion_1'], kde=True, ax=axes[0])
axes[0].set_title('Distribution of Promotion 1')
sns.histplot(train_df['promotion_2'], kde=True, ax=axes[1])
axes[1].set_title('Distribution of Promotion 2')
sns.histplot(train_df['promotion_3'], kde=True, ax=axes[2])
axes[2].set_title('Distribution of Promotion 3')
plt.tight_layout()
plt.show()

# Correlation analysis
plt.figure(figsize=(10, 8))
correlation_matrix = train_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


# Extract time-based features
for df in [train_df, test_df]:
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['week'] = df['datetime'].dt.isocalendar().week
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['quarter'] = df['datetime'].dt.quarter
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['week_of_year'] = df['datetime'].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    # Add more features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)

# Create lag features (for all promotion columns)
for lag in range(1, 24):  # Expand lags to 24 hours
    for col in ['promotion_1', 'promotion_2', 'promotion_3']:
        train_df[f'{col}_lag_{lag}'] = train_df[col].shift(lag)
        test_df[f'{col}_lag_{lag}'] = test_df[col].shift(lag)
        
# Create lag features at different levels: 1, 2, 3 days
for lag in [24, 48, 72]:
    for col in ['promotion_1', 'promotion_2', 'promotion_3']:
        train_df[f'{col}_lag_{lag}h'] = train_df[col].shift(lag)
        test_df[f'{col}_lag_{lag}h'] = test_df[col].shift(lag)

# Fill NaN values created from lag features with the mean of each column
for col in train_df.columns:
    if 'lag' in col:
        train_df[col] = train_df[col].fillna(train_df[col].mean())
        test_df[col] = test_df[col].fillna(test_df[col].mean())

# Create rolling window features for promotion columns with various windows
for window in [6, 12, 24, 48, 168]:  # Adding more window sizes including weekly (168)
    for col in ['promotion_1', 'promotion_2', 'promotion_3']:
        train_df[f'{col}_rolling_mean_{window}'] = train_df[col].rolling(window=window).mean()
        train_df[f'{col}_rolling_std_{window}'] = train_df[col].rolling(window=window).std()
        train_df[f'{col}_rolling_min_{window}'] = train_df[col].rolling(window=window).min()
        train_df[f'{col}_rolling_max_{window}'] = train_df[col].rolling(window=window).max()
        
        test_df[f'{col}_rolling_mean_{window}'] = test_df[col].rolling(window=window).mean()
        test_df[f'{col}_rolling_std_{window}'] = test_df[col].rolling(window=window).std()
        test_df[f'{col}_rolling_min_{window}'] = test_df[col].rolling(window=window).min()
        test_df[f'{col}_rolling_max_{window}'] = test_df[col].rolling(window=window).max()

# Fill NaN values for rolling features
for col in train_df.columns:
    if 'rolling' in col:
        train_df[col] = train_df[col].fillna(train_df[col].mean())
        test_df[col] = test_df[col].fillna(test_df[col].mean())

# Get all feature columns
feature_columns = [col for col in train_df.columns if col not in ['datetime', 'e_users']]

print(f"Number of features: {len(feature_columns)}")
print("Sample feature names:", feature_columns[:10])

# Prepare feature matrix X and target vector y
X = train_df[feature_columns]
y = train_df['e_users']


def objective(trial):
    # Define the hyperparameters to optimize - simplified set to avoid instabilities
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': trial.suggest_categorical('booster', ['gbtree']),
        'lambda': trial.suggest_float('lambda', 1e-4, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-4, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-4, 0.5, log=True)
    }
    
    # Initialize time series cross-validator - use only 3 splits for faster trials
    tscv = TimeSeriesSplit(n_splits=3)
    rmse_scores = []
    
    # For early stopping
    early_stopping_rounds = 50
    n_estimators = 500  # Reduced from 3000
    
    # Perform time series cross-validation with error handling
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        try:
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            # Train the model with early stopping
            model = xgb.train(
                param,
                dtrain,
                num_boost_round=n_estimators,
                evals=[(dval, 'validation')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False
            )
            
            # Predict on validation set
            y_pred = model.predict(dval)
            
            # Calculate RMSE with error handling for NaN values
            if np.isnan(y_pred).any():
                return float('inf')  # Return a large value if predictions contain NaN
                
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            rmse_scores.append(rmse)
        except Exception as e:
            print(f"Error in objective function: {e}")
            return float('inf')  # Return a large value on error
    
    # Return the average RMSE with error handling
    if not rmse_scores:
        return float('inf')  # Return a large value if no valid scores
        
    return np.mean(rmse_scores)
    
# Create an Optuna study
print("Starting hyperparameter optimization with Optuna...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10, show_progress_bar=True)

# Ensure best_params is defined for the cells below
best_params = study.best_params
best_value = study.best_value
print("Starting hyperparameter optimization with Optuna...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10, show_progress_bar=True)

# Get the best parameters
best_params = study.best_params
best_value = study.best_value

print(f"\nBest RMSE: {best_value:.4f}")
print("\nBest hyperparameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Visualize the optimization results
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
optuna.visualization.matplotlib.plot_param_importances(study)
plt.tight_layout()
plt.show()


# Set up the XGBoost model with the best parameters
best_params['objective'] = 'reg:squarederror'
best_params['eval_metric'] = 'rmse'

# Initialize time series cross-validator
tscv = TimeSeriesSplit(n_splits=5)
xgb_rmse_scores = []

# For tracking performance across folds
all_true_values = []
all_predicted_values = []
all_fold_indices = []

print("\nTraining XGBoost with optimal parameters using time series cross-validation...")
for fold, (train_index, val_index) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train the model with early stopping
    xgb_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=3000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Predict on validation set
    y_pred = xgb_model.predict(dval)
    
    # Store true and predicted values for later analysis
    all_true_values.extend(y_val.tolist())
    all_predicted_values.extend(y_pred.tolist())
    all_fold_indices.extend([fold] * len(y_val))
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    xgb_rmse_scores.append(rmse)
    print(f'Fold {fold + 1} RMSE: {rmse:.4f}')

# Display average RMSE score
print(f'\nAverage RMSE across all folds: {np.mean(xgb_rmse_scores):.4f}')

# Plot predicted vs actual values for each fold
plt.figure(figsize=(15, 8))
fold_colors = ['blue', 'green', 'red', 'purple', 'orange']
for fold in range(5):
    fold_indices = [i for i, x in enumerate(all_fold_indices) if x == fold]
    fold_true = [all_true_values[i] for i in fold_indices]
    fold_pred = [all_predicted_values[i] for i in fold_indices]
    plt.scatter(fold_true, fold_pred, alpha=0.5, color=fold_colors[fold], label=f'Fold {fold+1}')

plt.plot([min(all_true_values), max(all_true_values)], [min(all_true_values), max(all_true_values)], 'k--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values Across All Folds')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Train the final model on the entire dataset
print("\nTraining final model on entire dataset...")
dtrain_full = xgb.DMatrix(X, label=y)
final_xgb_model = xgb.train(
    best_params,
    dtrain_full,
    num_boost_round=3000,
    verbose_eval=100
)

# Prepare the test feature matrix
X_test = test_df[feature_columns]
dtest = xgb.DMatrix(X_test)

# Predict on the test set
xgb_predictions = final_xgb_model.predict(dtest)

# Create the submission dataframe
submission_df = pd.DataFrame({
    'datetime': test_df['datetime'],
    'e_users': xgb_predictions
})

# Save the submission file
submission_df.to_csv('submission_xgboost_optuna.csv', index=False)
print("\nSubmission file 'submission_xgboost_optuna.csv' created successfully.")


# Feature importance analysis
feature_importance = final_xgb_model.get_score(importance_type='gain')
sorted_idx = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
sorted_feature_names = [x[0] for x in sorted_idx]
sorted_feature_values = [x[1] for x in sorted_idx]

# Plot top 30 features
plt.figure(figsize=(14, 10))
plt.barh(range(min(30, len(sorted_feature_names))), sorted_feature_values[:30])
plt.yticks(range(min(30, len(sorted_feature_names))), sorted_feature_names[:30])
plt.xlabel('Feature Importance (gain)')
plt.title('XGBoost Feature Importance (Top 30)')
plt.tight_layout()
plt.show()

# Temporal Patterns Analysis
# Hourly pattern
hourly_pattern = train_df.groupby('hour')['e_users'].mean().reset_index()
plt.figure(figsize=(10, 5))
plt.bar(hourly_pattern['hour'], hourly_pattern['e_users'])
plt.xlabel('Hour of Day')
plt.ylabel('Average e_users')
plt.title('Average e_users by Hour of Day')
plt.xticks(range(24))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Day of week pattern
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_of_week_pattern = train_df.groupby('day_of_week')['e_users'].mean().reset_index()
plt.figure(figsize=(10, 5))
plt.bar(day_of_week_pattern['day_of_week'], day_of_week_pattern['e_users'])
plt.xlabel('Day of Week')
plt.ylabel('Average e_users')
plt.title('Average e_users by Day of Week')
plt.xticks(range(7), day_names)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Save the model for future use
import pickle

# Save the XGBoost model
with open('xgb_model_optuna.pkl', 'wb') as f:
    pickle.dump(final_xgb_model, f)

# Save the best parameters
with open('xgb_best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print("\nModel and best parameters saved for future use.")
print("\nSummary:")
print(f"Best RMSE from Optuna optimization: {best_value:.4f}")
print(f"Average RMSE from time series cross-validation: {np.mean(xgb_rmse_scores):.4f}")
print("Submission file created: submission_xgboost_optuna.csv")

