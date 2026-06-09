import os
import numpy as np
import pandas as pd

import optuna
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from prophet import Prophet

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


# Load the dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


# Print the headers of train_df and test_df
print("Train DataFrame Header:")
print(train_df.columns)

print("\nTest DataFrame Header:")
print(test_df.columns)

print("\nSample Submission Header:")
print(sample_submission_df.columns)


# Check for missing values
missing_data = train_df.isnull().sum()
print("Missing values in each column:")
print(missing_data)


# Check rows with missing num_sold
missing_rows = train_df[train_df['num_sold'].isnull()]
print("Sample rows with missing num_sold:")
print(missing_rows.head())

# Analyze missing data distribution
print("Missing data distribution by store and product:")
print(missing_rows.groupby(['store', 'product']).size())


# Calculate grouped means for 'Holographic Goose'
holographic_means = train_df[
    (train_df['product'] == 'Holographic Goose') & (train_df['num_sold'].notnull())
].groupby(['country', 'store', 'date'])['num_sold'].mean()

# Define a function to fill missing values using grouped means
def impute_holographic(row):
    if pd.isnull(row['num_sold']) and row['product'] == 'Holographic Goose':
        return holographic_means.get((row['country'], row['store'], row['date']), None)
    return row['num_sold']

# Apply the function
train_df['num_sold'] = train_df.apply(impute_holographic, axis=1)


# Fill remaining missing values with store-level means
store_means = train_df[
    (train_df['product'] == 'Holographic Goose') & (train_df['num_sold'].notnull())
].groupby('store')['num_sold'].mean()

train_df['num_sold'] = train_df.apply(
    lambda row: store_means[row['store']] if pd.isnull(row['num_sold']) and row['product'] == 'Holographic Goose' else row['num_sold'],
    axis=1
)


# Group by store and product to compute means
grouped_means = train_df[train_df['num_sold'].notnull()].groupby(['store', 'product'])['num_sold'].mean()

# Fill missing values for other products
train_df['num_sold'] = train_df.apply(
    lambda row: grouped_means[row['store'], row['product']] if pd.isnull(row['num_sold']) else row['num_sold'],
    axis=1
)


# Check for remaining missing values
print("Remaining missing values:")
print(train_df.isnull().sum())

# Validate the distribution of num_sold after imputation
print("Updated distribution of num_sold:")
print(train_df['num_sold'].describe())


# Prepare data for LightGBM
X = train_df[['country', 'store', 'product', 'date']]
y = train_df['num_sold']

# Label encoding for categorical columns
for col in ['country', 'store', 'product', 'date']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Convert date to numerical format
X.loc[:, 'date'] = pd.to_datetime(X['date']).astype(int) / 10**9

print(X.dtypes)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Train LightGBM baseline model
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbosity': -1
}

# Train with early stopping using callbacks
callbacks = [
    early_stopping(stopping_rounds=50),  # Early stopping callback
    log_evaluation(period=10)           # Log every 10 rounds
]

# Train LightGBM model
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,  # Maximum number of boosting rounds
    valid_sets=[train_data, val_data],  # Validation set for early stopping
    callbacks=callbacks
)

# Evaluate the model
y_pred = model.predict(X_val, num_iteration=model.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"LightGBM Baseline RMSE: {rmse}")


# Prepare data for Prophet
prophet_df = train_df[['date', 'num_sold']].rename(columns={'date': 'ds', 'num_sold': 'y'})

# Train Prophet model
prophet_model = Prophet()
prophet_model.fit(prophet_df)

# Forecast
future = prophet_model.make_future_dataframe(periods=365)
forecast = prophet_model.predict(future)

# Plot forecast
prophet_model.plot(forecast)


# Define parameter grid
param_grid = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 500]
}

# Perform grid search
model = lgb.LGBMRegressor(objective='regression')
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

# Best parameters and score
print(f"Best Parameters: {grid.best_params_}")
print(f"Best RMSE: {np.sqrt(-grid.best_score_)}")


def objective(trial):
    # Define the hyperparameter space
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }
    
    # Train with early stopping using callbacks
    callbacks = [
        early_stopping(stopping_rounds=50),  # Early stopping callback
        log_evaluation(period=10)           # Log every 10 rounds
    ]
    
    # Train model
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        callbacks=callbacks
    )
    
    # Predict and calculate RMSE
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Best parameters
print(f"Best Parameters: {study.best_params}")
print(f"Best RMSE: {study.best_value}")


# Evaluate the model
print(f"LightGBM Baseline RMSE: {rmse}")

# Best parameters and score
print(f"Best Parameters: {grid.best_params_}")
print(f"Best RMSE: {np.sqrt(-grid.best_score_)}")

# Best parameters
print(f"Best Parameters: {study.best_params}")
print(f"Best RMSE: {study.best_value}")


# Label encode categorical columns
for col in ['country', 'store', 'product', 'date']:
    le = LabelEncoder()
    test_df[col] = le.fit_transform(test_df[col])

# Convert 'date' to numeric timestamp
test_df['date'] = pd.to_datetime(test_df['date']).astype(int) / 10**9


# Train data preparation
X_full = X[['country', 'store', 'product', 'date']]
y_full = y

# Train the final model with Grid Search parameters (example)
final_model = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=50,
    learning_rate=0.1,
    n_estimators=500
)
final_model.fit(X_full, y_full)


# Make predictions
test_df['num_sold'] = final_model.predict(test_df[['country', 'store', 'product', 'date']])


# Fill in the predicted 'num_sold' values
sample_submission_df['num_sold'] = test_df['num_sold']

# Save the submission file
sample_submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")

