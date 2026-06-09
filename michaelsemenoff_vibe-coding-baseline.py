import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler

# Load the data
train_df = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/train_df.csv')
test_df = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/test_df.csv')

# Convert datetime column to datetime type
train_df['datetime'] = pd.to_datetime(train_df['datetime'])
test_df['datetime'] = pd.to_datetime(test_df['datetime'])


# Exploratory Data Analysis
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

# Create lag features (for all promotion columns)
for lag in range(1, 8):  # Create lags from 1 to 7
    for col in ['promotion_1', 'promotion_2', 'promotion_3']:
        train_df[f'{col}_lag_{lag}'] = train_df[col].shift(lag)
        test_df[f'{col}_lag_{lag}'] = test_df[col].shift(lag)

# Fill NaN values created from lag features with the mean of each column
for col in train_df.columns:
    if 'lag' in col:
        train_df[col] = train_df[col].fillna(train_df[col].mean())
        test_df[col] = test_df[col].fillna(test_df[col].mean())

# Create rolling window features for promotion columns
for col in ['promotion_1', 'promotion_2', 'promotion_3']:
    train_df[f'{col}_rolling_mean'] = train_df[col].rolling(window=24).mean()
    train_df[f'{col}_rolling_std'] = train_df[col].rolling(window=24).std()
    
    test_df[f'{col}_rolling_mean'] = test_df[col].rolling(window=24).mean()
    test_df[f'{col}_rolling_std'] = test_df[col].rolling(window=24).std()

# Fill NaN values for rolling features
for col in train_df.columns:
    if 'rolling' in col:
        train_df[col] = train_df[col].fillna(train_df[col].mean())
        test_df[col] = test_df[col].fillna(test_df[col].mean())


# Define the feature columns
feature_columns = [
    'promotion_1', 'promotion_2', 'promotion_3',
    'hour', 'day', 'week', 'month', 'year', 'day_of_week', 'quarter', 'day_of_year', 'week_of_year',
    'promotion_1_lag_1', 'promotion_1_lag_2', 'promotion_1_lag_3', 'promotion_1_lag_4', 'promotion_1_lag_5', 'promotion_1_lag_6', 'promotion_1_lag_7',
    'promotion_2_lag_1', 'promotion_2_lag_2', 'promotion_2_lag_3', 'promotion_2_lag_4', 'promotion_2_lag_5', 'promotion_2_lag_6', 'promotion_2_lag_7',
    'promotion_3_lag_1', 'promotion_3_lag_2', 'promotion_3_lag_3', 'promotion_3_lag_4', 'promotion_3_lag_5', 'promotion_3_lag_6', 'promotion_3_lag_7',
    'promotion_1_rolling_mean', 'promotion_1_rolling_std',
    'promotion_2_rolling_mean', 'promotion_2_rolling_std',
    'promotion_3_rolling_mean', 'promotion_3_rolling_std',
    'is_weekend'
]

# Prepare the feature matrix X and target vector y
X = train_df[feature_columns]
y = train_df['e_users']


# Initialize the linear regression model
lr_model = LinearRegression()

# Initialize time series cross-validator
tscv = TimeSeriesSplit(n_splits=5)

# Store RMSE scores for each fold
lr_rmse_scores = []

# Perform time series cross-validation
for train_index, val_index in tscv.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Train the model
    lr_model.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred = lr_model.predict(X_val)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    lr_rmse_scores.append(rmse)
    print(f'Linear Regression Fold RMSE: {rmse}')

# Display average RMSE score
print(f'Linear Regression Average RMSE: {np.mean(lr_rmse_scores)}')

# Train the model on the entire training set
lr_model.fit(X, y)

# Prepare the test feature matrix
X_test = test_df[feature_columns]

# Predict on the test set
lr_predictions = lr_model.predict(X_test)

# Create the submission dataframe
lr_submission_df = pd.DataFrame({
    'datetime': test_df['datetime'],
    'e_users': lr_predictions
})

# Save the submission file
lr_submission_df.to_csv('submission_lr.csv', index=False)

# Display the first few rows of the submission file
print("Linear Regression Submission File:")
print(lr_submission_df.head())


# Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=1000, 
    learning_rate=0.01, 
    max_depth=6, 
    subsample=0.8, 
    colsample_bytree=0.8,
    early_stopping_rounds=50
)

# Initialize time series cross-validator
tscv = TimeSeriesSplit(n_splits=5)

# Store RMSE scores for each fold
xgb_rmse_scores = []

# Perform time series cross-validation
for fold, (train_index, val_index) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Train the model with early stopping
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric='rmse',
        verbose=100
    )
    
    # Get best score
    best_score = xgb_model.best_score
    
    # Predict on validation set
    y_pred = xgb_model.predict(X_val)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    xgb_rmse_scores.append(rmse)
    print(f'XGBoost Fold {fold + 1} RMSE: {rmse:.4f} (Best Score: {best_score:.4f})')

# Display average RMSE score
print(f'\nXGBoost Average RMSE: {np.mean(xgb_rmse_scores):.4f}')

# Train the final model on the entire training set
# Use a small portion of the data (5%) as a validation set for early stopping
train_size = int(len(X) * 0.95)
X_train_final, X_val_final = X.iloc[:train_size], X.iloc[train_size:]
y_train_final, y_val_final = y.iloc[:train_size], y.iloc[train_size:]

xgb_model.set_params(early_stopping_rounds=50)
xgb_model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_val_final, y_val_final)],
    eval_metric='rmse',
    verbose=False
)

# Predict on the test set
xgb_predictions = xgb_model.predict(X_test)

# Create the submission dataframe
xgb_submission_df = pd.DataFrame({
    'datetime': test_df['datetime'],
    'e_users': xgb_predictions
})

# Save the submission file
xgb_submission_df.to_csv('submission_xgboost.csv', index=False)

# Display the first few rows of the submission file
print("XGBoost Submission File:")
print(xgb_submission_df.head())


# Initialize the LightGBM model
lgb_model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50
)

# Initialize time series cross-validator
tscv = TimeSeriesSplit(n_splits=5)

# Store RMSE scores for each fold
lgb_rmse_scores = []

# Perform time series cross-validation
for fold, (train_index, val_index) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Train the model with early stopping using only the validation set
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse'
    )
    
    # Get the best validation score (now stored under 'valid_0')
    best_score = lgb_model.best_score_['valid_0']['rmse']  # Extract the actual metric value
    
    # Predict on validation set
    y_pred = lgb_model.predict(X_val)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    lgb_rmse_scores.append(rmse)
    print(f'LightGBM Fold {fold + 1} RMSE: {rmse:.4f} (Best Score: {best_score:.4f})')

# Display average RMSE score
print(f'\nLightGBM Average RMSE: {np.mean(lgb_rmse_scores):.4f}')

# Train the final model on the entire dataset
# Split a small portion for validation to enable early stopping
train_size = int(len(X) * 0.95)
X_train_final, X_val_final = X.iloc[:train_size], X.iloc[train_size:]
y_train_final, y_val_final = y.iloc[:train_size], y.iloc[train_size:]

lgb_model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_val_final, y_val_final)],
    eval_metric='rmse'
)

# Predict on the test set
lgb_predictions = lgb_model.predict(X_test)

# Create the submission dataframe
lgb_submission_df = pd.DataFrame({
    'datetime': test_df['datetime'],
    'e_users': lgb_predictions
})

# Save the submission file
lgb_submission_df.to_csv('submission_lightgbm.csv', index=False)

# Display the first few rows of the submission file
print("LightGBM Submission File:")
print(lgb_submission_df.head())


# Ensemble approach - average predictions from different models
ensemble_predictions = (lr_predictions + xgb_predictions + lgb_predictions) / 3

# Create the submission dataframe for ensemble predictions
submission_ensemble_df = pd.DataFrame({
    'datetime': test_df['datetime'],
    'e_users': ensemble_predictions
})

# Save the ensemble submission file
submission_ensemble_df.to_csv('submission_ensemble.csv', index=False)

# Display the first few rows of the ensemble submission file
print("Ensemble Submission File:")
print(submission_ensemble_df.head())


# Feature Importance Analysis (using LightGBM model)
feature_importance = lgb_model.feature_importances_
feature_names = feature_columns

# Sort feature importance in descending order
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_feature_importance = feature_importance[sorted_idx]
sorted_feature_names = [feature_names[i] for i in sorted_idx]

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.barh(range(len(sorted_feature_importance)), sorted_feature_importance)
plt.yticks(range(len(sorted_feature_importance)), sorted_feature_names)
plt.xlabel('Feature Importance')
plt.title('LightGBM Feature Importance')
plt.tight_layout()
plt.show()


# Model Comparison
models = ['Linear Regression', 'XGBoost', 'LightGBM', 'Ensemble']
rmse_values = [
    np.mean(lr_rmse_scores), 
    np.mean(xgb_rmse_scores), 
    np.mean(lgb_rmse_scores), 
    None  # Ensemble doesn't have a validation RMSE
]

plt.figure(figsize=(10, 6))
plt.bar(models[:-1], rmse_values[:-1])
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('Model Comparison by RMSE (Lower is Better)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Visualize predictions (using a small subset of data for clarity)
sample_size = 100  # Number of data points to visualize

# Get a subset of the true values and predictions
sample_true = train_df['e_users'].iloc[-sample_size:].values
sample_pred_lr = lr_model.predict(X.iloc[-sample_size:])
sample_pred_xgb = xgb_model.predict(X.iloc[-sample_size:])
sample_pred_lgb = lgb_model.predict(X.iloc[-sample_size:])

# Create a DataFrame for plotting
sample_df = pd.DataFrame({
    'Date': train_df['datetime'].iloc[-sample_size:],
    'Actual': sample_true,
    'Linear Regression': sample_pred_lr,
    'XGBoost': sample_pred_xgb,
    'LightGBM': sample_pred_lgb
})

# Plot the predictions vs actual values
plt.figure(figsize=(14, 7))
plt.plot(sample_df['Date'], sample_df['Actual'], label='Actual', marker='o', markersize=3)
plt.plot(sample_df['Date'], sample_df['Linear Regression'], label='Linear Regression', marker='s', markersize=3)
plt.plot(sample_df['Date'], sample_df['XGBoost'], label='XGBoost', marker='^', markersize=3)
plt.plot(sample_df['Date'], sample_df['LightGBM'], label='LightGBM', marker='x', markersize=3)

plt.xlabel('Date')
plt.ylabel('e_users')
plt.title('Model Predictions vs Actual Values')
plt.legend()
plt.xticks(rotation=45)
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

# Monthly pattern
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_pattern = train_df.groupby('month')['e_users'].mean().reset_index()
plt.figure(figsize=(10, 5))
plt.bar(monthly_pattern['month'], monthly_pattern['e_users'])
plt.xlabel('Month')
plt.ylabel('Average e_users')
plt.title('Average e_users by Month')
plt.xticks(range(1, 13), month_names)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Time Series Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

# Select a continuous period of data
continuous_period = train_df.sort_values('datetime')
continuous_period = continuous_period.set_index('datetime')
ts_data = continuous_period['e_users'].resample('D').mean()  # Resample to daily data for decomposition

# Perform time series decomposition
decomposition = seasonal_decompose(ts_data.dropna(), model='additive', period=7)  # assuming weekly seasonality

# Plot decomposition
plt.figure(figsize=(14, 12))

plt.subplot(4, 1, 1)
plt.plot(decomposition.observed)
plt.title('Observed')
plt.grid(linestyle='--', alpha=0.7)

plt.subplot(4, 1, 2)
plt.plot(decomposition.trend)
plt.title('Trend')
plt.grid(linestyle='--', alpha=0.7)

plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal)
plt.title('Seasonality')
plt.grid(linestyle='--', alpha=0.7)

plt.subplot(4, 1, 4)
plt.plot(decomposition.resid)
plt.title('Residuals')
plt.grid(linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# Conclusion and Final Model Selection
print("Model Comparison Summary:")
print("-" * 50)
print(f"Linear Regression RMSE: {np.mean(lr_rmse_scores):.4f}")
print(f"XGBoost RMSE: {np.mean(xgb_rmse_scores):.4f}")
print(f"LightGBM RMSE: {np.mean(lgb_rmse_scores):.4f}")
print("-" * 50)

# Determine best model based on RMSE
model_rmse_dict = {
    "Linear Regression": np.mean(lr_rmse_scores),
    "XGBoost": np.mean(xgb_rmse_scores),
    "LightGBM": np.mean(lgb_rmse_scores)
}
best_model = min(model_rmse_dict, key=model_rmse_dict.get)

print(f"Based on the RMSE values, the best performing individual model is: {best_model}")
print("However, the ensemble approach combining all models may provide more robust predictions.")
print("-" * 50)
print("Final Submission: submission_ensemble.csv")

# Create final submission file (using the ensemble predictions)
final_submission = pd.DataFrame({
    'datetime': test_df['datetime'],
    'e_users': ensemble_predictions
})

# Save the final submission file
final_submission.to_csv('submission.csv', index=False)
print("Final submission file 'submission.csv' created successfully.")


# Save models for future use
import pickle

# Save the LightGBM model
with open('lgb_model.pkl', 'wb') as f:
    pickle.dump(lgb_model, f)

# Save the XGBoost model
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Save the Linear Regression model
with open('lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

# Save the feature scaler
scaler = MinMaxScaler()
scaler.fit(X)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModels saved for future use.")


# Function for making predictions with the ensemble model
def predict_with_ensemble(new_data):
    """
    Make predictions using the ensemble of models
    
    Parameters:
        new_data (DataFrame): New data with the same features as the training data
    
    Returns:
        dict: Dictionary containing predictions from each model and the ensemble
    """
    # Prepare data
    new_data = new_data.copy()
    
    # Extract temporal features if datetime is provided
    if 'datetime' in new_data.columns:
        new_data['datetime'] = pd.to_datetime(new_data['datetime'])
        new_data['hour'] = new_data['datetime'].dt.hour
        new_data['day'] = new_data['datetime'].dt.day
        new_data['week'] = new_data['datetime'].dt.isocalendar().week
        new_data['month'] = new_data['datetime'].dt.month
        new_data['year'] = new_data['datetime'].dt.year
        new_data['day_of_week'] = new_data['datetime'].dt.dayofweek
        new_data['quarter'] = new_data['datetime'].dt.quarter
        new_data['day_of_year'] = new_data['datetime'].dt.dayofyear
        new_data['week_of_year'] = new_data['datetime'].dt.isocalendar().week
        new_data['is_weekend'] = new_data['day_of_week'].isin([5, 6]).astype(int)
    
    # Linear Regression prediction
    lr_pred = lr_model.predict(new_data[feature_columns])
    
    # XGBoost prediction
    xgb_pred = xgb_model.predict(new_data[feature_columns])
    
    # LightGBM prediction
    lgb_pred = lgb_model.predict(new_data[feature_columns])
    
    # Ensemble prediction
    ensemble_pred = (lr_pred + xgb_pred + lgb_pred) / 3
    
    return {
        'linear_regression': lr_pred,
        'xgboost': xgb_pred,
        'lightgbm': lgb_pred,
        'ensemble': ensemble_pred
    }

print("\nEnsemble prediction function created.")
print("Example usage: predictions = predict_with_ensemble(new_data)")


# Potential future improvements
print("\nPotential Future Improvements:")
print("-" * 50)
print("1. More advanced time series models like Prophet or DeepAR")
print("2. Incorporate external data sources like events, holidays, or social media trends")
print("3. Implement more sophisticated ensemble techniques like stacking")
print("4. Automated hyperparameter tuning for all models")
print("5. More extensive feature engineering including lag features and rolling statistics")
print("6. Implement anomaly detection to handle outliers")
print("7. Consider hierarchical forecasting if data can be segmented by user groups")
print("8. Experiment with different LightGBM parameters for optimization")
print("9. Try different window sizes for rolling statistics")
print("10. Implement cross-validation with different time periods")
print("-" * 50)
print("End of analysis.")

