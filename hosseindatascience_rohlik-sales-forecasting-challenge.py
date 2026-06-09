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


# 1. Data Loading, Exploration, and Missing Value Handling
# Objective:

# Load all datasets, explore the data structure, and check for missing values before further processing.
# Datasets:

#     sales_train.csv (Training dataset)
#     sales_test.csv (Test dataset)
#     calendar.csv (Holidays, school holidays, and other time-based features)
#     inventory.csv (Product-specific inventory details)
#     test_weights.csv (Weights for evaluation, may be used for weighted scoring or modeling)

# Actions:

#     Load all datasets into pandas DataFrames.
#     Perform basic data exploration:
#         Check the shape and column types.
#         Preview the first few rows of each dataset.
#         Check for missing values in all datasets.
#     Handle missing values before feature engineering:
#         Fill missing values in numerical features (e.g., median or mean for sales, total_orders, sell_price_main).
#         Fill categorical missing values with "Unknown" if applicable (e.g., inventory dataset).
#         Ensure test dataset does not contain any missing values.

# Output:

# Cleaned datasets with missing values handled, ready for merging.
# 2. Data Preprocessing and Merging
# Objective:

# Merge all datasets into a unified dataset for training.
# Actions:

#     Convert date columns to datetime format for consistency.
#     Merge sales_train_df with calendar_df on date.
#     Merge sales_train_df with inventory_df on unique_id to include product inventory details.
#     Ensure that test dataset (sales_test_df) does not include target variables (sales, availability) and only retains features needed for predictions.

# Output:

# Unified and clean datasets for both training and testing.
# 3. Feature Engineering
# Objective:

# Create new features to improve model performance.
# Actions:

#     Date-based features:
#         Extract year, month, day_of_week, week_of_year, and is_weekend from the date column.
#     Lag features (for training dataset only):
#         Create sales_last_7_days, sales_last_30_days, total_orders_last_7_days, and total_orders_last_30_days.
#     Rolling aggregate features:
#         Compute rolling averages for sales and total_orders over 7-day and 30-day windows.
#     Price-related features:
#         Compute price_change as a percentage change in sell_price_main.
#     Ensure lag and rolling features do not introduce missing values:
#         Fill NaN values resulting from shifting with 0 or use forward-fill.

# Output:

# Feature-engineered datasets ready for model training.
# 4. Splitting the Data
# Objective:

# Prepare training and validation sets.
# Actions:

#     Time-based split:
#         Use the first portion of the dataset for training and the later portion for validation.
#     Rolling-window cross-validation (optional):
#         Train on past data, validate on future data.

# Output:

# Prepared training and validation sets for model fitting.
# 5. Model Development and Evaluation
# Objective:

# Train and evaluate different models.
# Actions:

#     Baseline Model:
#         Implement a simple forecasting method (e.g., naive approach).
#     Advanced Models:
#         Train models such as:
#             Linear Regression
#             Random Forest / Gradient Boosting
#             XGBoost / LightGBM
#             Neural Networks (if applicable)
#     Evaluation Metrics:
#         Use metrics like RMSE, MAE, MAPE, and RÂ² to assess model performance.

# Output:

# Trained models with evaluation results.
# 6. Hyperparameter Tuning and Model Refinement
# Objective:

# Optimize model performance.
# Actions:

#     Perform grid search or random search for hyperparameter tuning.
#     Use feature selection techniques to remove unimportant features.
#     Experiment with different feature sets, models, and tuning strategies.

# Output:

# Optimized models with improved performance.
# 7. Prediction on Test Data
# Objective:

# Generate predictions for the test dataset.
# Actions:

#     Use the best-performing model to predict sales for sales_test_df.
#     Ensure predictions follow the required format.

# Output:

# Predictions for submission.
# 8. Submission
# Objective:

# Format predictions and submit.
# Actions:

#     Format predictions according to competition requirements.
#     Submit the final predictions.

# Output:

# Submission-ready predictions.


# Stage 01

import pandas as pd
import numpy as np

# Load datasets
sales_train_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
sales_test_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv')
calendar_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv')
inventory_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
test_weights_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')

# 1. Check dataset shapes
print(f"Sales Train Data Shape: {sales_train_df.shape}")
print(f"Sales Test Data Shape: {sales_test_df.shape}")
print(f"Calendar Data Shape: {calendar_df.shape}")
print(f"Inventory Data Shape: {inventory_df.shape}")
print(f"Test Weights Data Shape: {test_weights_df.shape}")

# 2. Preview first rows of each dataset
datasets = {
    "Sales Train": sales_train_df,
    "Sales Test": sales_test_df,
    "Calendar": calendar_df,
    "Inventory": inventory_df,
    "Test Weights": test_weights_df
}

for name, df in datasets.items():
    print(f"\n{name} Data Preview:")
    print(df.head())

# 3. Check for missing values in all datasets
print("\nMissing Data Summary (Before Handling):")
for name, df in datasets.items():
    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        print(f"\nMissing values in {name}:")
        print(missing_values)
    else:
        print(f"\nNo missing values in {name} dataset.")

# Handle missing values in Sales Train dataset
sales_train_df['sales'] = sales_train_df['sales'].fillna(sales_train_df['sales'].median())
sales_train_df['total_orders'] = sales_train_df['total_orders'].fillna(sales_train_df['total_orders'].median())
sales_train_df['sell_price_main'] = sales_train_df['sell_price_main'].fillna(sales_train_df['sell_price_main'].median())
sales_train_df['availability'] = sales_train_df['availability'].fillna(1)  # Assuming 1 means available

# Handle missing values in Sales Test dataset
sales_test_df = sales_test_df.fillna(0)

# Handle missing values in Calendar dataset
calendar_df['holiday_name'] = calendar_df['holiday_name'].fillna("No Holiday")


# Handle missing values in Inventory dataset
inventory_df = inventory_df.fillna("Unknown")

# Handle missing values in Test Weights dataset
test_weights_df['weight'] = test_weights_df['weight'].fillna(test_weights_df['weight'].median())


# 5. Verify after handling missing values
print("\nMissing Data Summary (After Handling):")
for name, df in datasets.items():
    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        print(f"\nStill missing values in {name}:")
        print(missing_values)
    else:
        print(f"\nNo missing values in {name} dataset after handling.")

print("\nStage 01 Completed! Ready for Feature Engineering.")




# Check the number of NaN values before and after merging
print("Missing values before merging (Stage 01):")
print(sales_train_df.isna().sum())

# Check for missing values specifically in columns related to merging
print("Missing values in Calendar before merging:")
print(calendar_df.isna().sum())

print("Missing values in Inventory before merging:")
print(inventory_df.isna().sum())



# Stage 02

import pandas as pd

# Define chunk size
chunk_size = 500000  # You can adjust this based on memory limitations

# Ensure the 'date' column in calendar_df is in datetime format
calendar_df['date'] = pd.to_datetime(calendar_df['date'])

# Load sales_train_df in chunks
sales_train_chunks = []
for chunk in pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv', chunksize=chunk_size):
    # Convert 'date' column to datetime format
    chunk['date'] = pd.to_datetime(chunk['date'])
    
    # Merge with calendar_df and inventory_df for each chunk
    calendar_chunk = calendar_df.drop(columns=['warehouse'])  # Drop 'warehouse' before merging
    chunk = pd.merge(chunk, calendar_chunk, on='date', how='left', suffixes=('_train', '_calendar'))
    chunk = pd.merge(chunk, inventory_df, on='unique_id', how='left', suffixes=('_train', '_inventory'))
    
    # Append processed chunk
    sales_train_chunks.append(chunk)

# Concatenate all chunks into one DataFrame
sales_train_df = pd.concat(sales_train_chunks, ignore_index=True)

# Load sales_test_df in chunks
sales_test_chunks = []
for chunk in pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', chunksize=chunk_size):
    # Convert 'date' column to datetime format
    chunk['date'] = pd.to_datetime(chunk['date'])
    
    # Create time-related features for sales_test_df chunk
    chunk['year'] = chunk['date'].dt.year
    chunk['month'] = chunk['date'].dt.month
    chunk['day_of_week'] = chunk['date'].dt.dayofweek
    chunk['week_of_year'] = chunk['date'].dt.isocalendar().week
    chunk['is_weekend'] = (chunk['day_of_week'] >= 5).astype(int)
    
    # Retain only necessary columns
    chunk = chunk[['unique_id', 'date', 'warehouse', 'total_orders', 'sell_price_main', 
                   'type_0_discount', 'type_1_discount', 'type_2_discount', 'type_3_discount', 
                   'type_4_discount', 'type_5_discount', 'type_6_discount', 'year', 'month', 
                   'day_of_week', 'week_of_year', 'is_weekend']]
    
    # Merge with calendar_df
    chunk = pd.merge(chunk, calendar_df, on='date', how='left', suffixes=('_test', '_calendar'))

    # Merge with inventory_df
    chunk = pd.merge(chunk, inventory_df, on='unique_id', how='left', suffixes=('_test', '_inventory'))
    
    # Append processed chunk
    sales_test_chunks.append(chunk)

# Concatenate all chunks into one DataFrame
sales_test_df = pd.concat(sales_test_chunks, ignore_index=True)

# 1. Handle missing values in Sales Train dataset
sales_train_df = sales_train_df.copy()
sales_train_df['sales'] = sales_train_df['sales'].fillna(sales_train_df['sales'].median())
sales_train_df['total_orders'] = sales_train_df['total_orders'].fillna(sales_train_df['total_orders'].median())
sales_train_df['sell_price_main'] = sales_train_df['sell_price_main'].fillna(sales_train_df['sell_price_main'].median())
sales_train_df['availability'] = sales_train_df['availability'].fillna(1)  # Assuming 1 means available
sales_train_df = sales_train_df.fillna("Unknown")  # Fill remaining categorical NaNs

# 2. Handle missing values in Sales Test dataset
sales_test_df = sales_test_df.fillna(0)  # No target column in test set, fill missing values with 0

# 3. Verify missing values after handling
print("\nMissing Values After Handling:")
print(sales_train_df.isna().sum())
print(sales_test_df.isna().sum())

# Check if merging was successful
print(f"\nMerged Sales Train Data Shape: {sales_train_df.shape}")
print(f"Merged Sales Test Data Shape: {sales_test_df.shape}")

# Display preview of merged datasets
print("\nMerged Sales Train Data Preview:")
print(sales_train_df.head())

print("\nMerged Sales Test Data Preview:")
print(sales_test_df.head())

print("\nStage 02 Completed! Ready for Feature Engineering.")



import pandas as pd
import numpy as np

# Define chunk size
chunk_size = 500000  # Adjust based on available memory

# Process Sales Train Data in Chunks
sales_train_chunks = []
for chunk in pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv', chunksize=chunk_size):
    # Convert 'date' column to datetime format
    chunk['date'] = pd.to_datetime(chunk['date'])
    
    # 1. Date-Based Features
    chunk['year'] = chunk['date'].dt.year
    chunk['month'] = chunk['date'].dt.month
    chunk['day_of_week'] = chunk['date'].dt.dayofweek
    chunk['week_of_year'] = chunk['date'].dt.isocalendar().week
    chunk['is_weekend'] = (chunk['day_of_week'] >= 5).astype(int)

    # Append processed chunk
    sales_train_chunks.append(chunk)

# Concatenate processed chunks
sales_train_df = pd.concat(sales_train_chunks, ignore_index=True)

# Sort data to ensure correct lag calculation
sales_train_df = sales_train_df.sort_values(['unique_id', 'date'])

# 2. Lag Features
sales_train_df['sales_last_7_days'] = sales_train_df.groupby('unique_id')['sales'].shift(7)
sales_train_df['sales_last_30_days'] = sales_train_df.groupby('unique_id')['sales'].shift(30)
sales_train_df['total_orders_last_7_days'] = sales_train_df.groupby('unique_id')['total_orders'].shift(7)
sales_train_df['total_orders_last_30_days'] = sales_train_df.groupby('unique_id')['total_orders'].shift(30)

# Fill missing values in lag features with 0
lag_features = ['sales_last_7_days', 'sales_last_30_days', 'total_orders_last_7_days', 'total_orders_last_30_days']
sales_train_df[lag_features] = sales_train_df[lag_features].fillna(0)

# Fill missing values caused by lagging
sales_train_df[['sales', 'total_orders']] = sales_train_df[['sales', 'total_orders']].fillna(0)

# 3. Rolling Aggregate Features
sales_train_df['sales_rolling_7'] = sales_train_df.groupby('unique_id')['sales'].transform(lambda x: x.rolling(7, min_periods=1).mean())
sales_train_df['sales_rolling_30'] = sales_train_df.groupby('unique_id')['sales'].transform(lambda x: x.rolling(30, min_periods=1).mean())

sales_train_df['total_orders_rolling_7'] = sales_train_df.groupby('unique_id')['total_orders'].transform(lambda x: x.rolling(7, min_periods=1).mean())
sales_train_df['total_orders_rolling_30'] = sales_train_df.groupby('unique_id')['total_orders'].transform(lambda x: x.rolling(30, min_periods=1).mean())

# 4. Price-Related Features
sales_train_df['price_change'] = sales_train_df.groupby('unique_id')['sell_price_main'].pct_change().fillna(0)

# 5. Handle Missing Values (Ensuring No NaNs Due to Rolling or Lag Features)
rolling_features = ['sales_rolling_7', 'sales_rolling_30', 'total_orders_rolling_7', 'total_orders_rolling_30']
sales_train_df[rolling_features] = sales_train_df[rolling_features].ffill()

# Process Sales Test Data in Chunks
sales_test_chunks = []
for chunk in pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', chunksize=chunk_size):
    # Convert 'date' column to datetime format
    chunk['date'] = pd.to_datetime(chunk['date'])

    # Date-Based Features
    chunk['year'] = chunk['date'].dt.year
    chunk['month'] = chunk['date'].dt.month
    chunk['day_of_week'] = chunk['date'].dt.dayofweek
    chunk['week_of_year'] = chunk['date'].dt.isocalendar().week
    chunk['is_weekend'] = (chunk['day_of_week'] >= 5).astype(int)

    # Append processed chunk
    sales_test_chunks.append(chunk)

# Concatenate processed chunks
sales_test_df = pd.concat(sales_test_chunks, ignore_index=True)

# Verify missing values after feature engineering
print("\nMissing Values After Fixing:")
print(sales_train_df.isna().sum())

print("\nFeature Engineering Completed! Ready for Splitting the Data.")



# Stage 04: Splitting the Data

import pandas as pd
import numpy as np

# Ensure Data is Sorted by 'date' for Time-Based Splitting
sales_train_df = sales_train_df.sort_values(['unique_id', 'date']).reset_index(drop=True)

# 1. **Time-Based Split (80-20)**
split_date = sales_train_df['date'].quantile(0.8)  # Use 80% of data for training

train_df = sales_train_df[sales_train_df['date'] <= split_date].copy()
val_df = sales_train_df[sales_train_df['date'] > split_date].copy()

print(f"Training Set Shape: {train_df.shape}")
print(f"Validation Set Shape: {val_df.shape}")

# 2. **Rolling-Window Cross-Validation (Optional)**
# Define window sizes (e.g., 3 splits)
window_splits = []
num_splits = 3  # Number of rolling windows

total_days = (sales_train_df['date'].max() - sales_train_df['date'].min()).days
window_size = total_days // (num_splits + 1)

for i in range(num_splits):
    start_date = sales_train_df['date'].min() + pd.Timedelta(days=i * window_size)
    end_date = start_date + pd.Timedelta(days=window_size)

    rolling_train = sales_train_df[sales_train_df['date'] < end_date].copy()
    rolling_val = sales_train_df[(sales_train_df['date'] >= end_date) & (sales_train_df['date'] < end_date + pd.Timedelta(days=window_size))].copy()

    window_splits.append((rolling_train, rolling_val))
    print(f"Rolling Window {i+1}: Train {rolling_train.shape}, Validation {rolling_val.shape}")

# **Final Verification**
print("\nFinal Data Splits:")
print(f"Training Data Shape: {train_df.shape}")
print(f"Validation Data Shape: {val_df.shape}")

print("\nSplitting Completed! Ready for Model Development.")



import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make a copy of val_df to avoid chained assignment warning
val_df_copy = val_df.copy()

# NaÃ¯ve forecast: Use last week's sales as the prediction
val_df_copy["sales_pred"] = val_df_copy.groupby("unique_id")["sales"].shift(7)

# Fill NaN values with median sales
val_df_copy["sales_pred"] = val_df_copy["sales_pred"].fillna(val_df_copy["sales"].median())

# Calculate evaluation metrics
mae_naive = mean_absolute_error(val_df_copy["sales"], val_df_copy["sales_pred"])
rmse_naive = np.sqrt(mean_squared_error(val_df_copy["sales"], val_df_copy["sales_pred"]))
r2_naive = r2_score(val_df_copy["sales"], val_df_copy["sales_pred"])

print(f"NaÃ¯ve Model Performance:")
print(f"MAE: {mae_naive:.4f}")
print(f"RMSE: {rmse_naive:.4f}")
print(f"RÂ² Score: {r2_naive:.4f}")



from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Select relevant features (excluding non-numeric ones)
features = [
    "total_orders", "sell_price_main", "type_0_discount", "type_1_discount",
    "type_2_discount", "type_3_discount", "type_4_discount", "type_5_discount",
    "type_6_discount", "year", "month", "day_of_week", "week_of_year", "is_weekend",
    "sales_last_7_days", "sales_last_30_days", "total_orders_last_7_days",
    "total_orders_last_30_days", "sales_rolling_7", "sales_rolling_30",
    "total_orders_rolling_7", "total_orders_rolling_30", "price_change"
]

X_train = train_df[features]
y_train = train_df["sales"]
X_valid = val_df[features]
y_valid = val_df["sales"]

# Train linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on validation set
y_pred = lr_model.predict(X_valid)

# Evaluate Linear Regression
mae = mean_absolute_error(y_valid, y_pred)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
r2 = r2_score(y_valid, y_pred)

print(f"Linear Regression Model Performance:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"RÂ² Score: {r2:.4f}")



from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_valid)

# Evaluate Random Forest Model
mae_rf = mean_absolute_error(y_valid, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_valid, y_pred_rf))
r2_rf = r2_score(y_valid, y_pred_rf)

print(f"Random Forest Model Performance:")
print(f"MAE: {mae_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}")
print(f"RÂ² Score: {r2_rf:.4f}")



import xgboost as xgb

xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_valid)

# Evaluate XGBoost Model
mae_xgb = mean_absolute_error(y_valid, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_valid, y_pred_xgb))
r2_xgb = r2_score(y_valid, y_pred_xgb)

print(f"XGBoost Model Performance:")
print(f"MAE: {mae_xgb:.4f}")
print(f"RMSE: {rmse_xgb:.4f}")
print(f"RÂ² Score: {r2_xgb:.4f}")



import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
lgb_model.fit(X_train, y_train)

y_pred_lgb = lgb_model.predict(X_valid)

# Evaluate LightGBM Model
mae_lgb = mean_absolute_error(y_valid, y_pred_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_valid, y_pred_lgb))
r2_lgb = r2_score(y_valid, y_pred_lgb)

print(f"LightGBM Model Performance:")
print(f"MAE: {mae_lgb:.4f}")
print(f"RMSE: {rmse_lgb:.4f}")
print(f"RÂ² Score: {r2_lgb:.4f}")



results = pd.DataFrame({
    "Model": ["NaÃ¯ve", "Linear Regression", "Random Forest", "XGBoost", "LightGBM"],
    "MAE": [mae_naive, mae, mae_rf, mae_xgb, mae_lgb],
    "RMSE": [rmse_naive, rmse, rmse_rf, rmse_xgb, rmse_lgb],
    "RÂ² Score": [r2_naive, r2, r2_rf, r2_xgb, r2_lgb]
})

print("\nModel Performance Comparison:")
print(results.sort_values(by="RMSE"))




import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
from tqdm import tqdm

# Keep only the top features to reduce memory
important_features = [
    "total_orders", "sell_price_main", "sales_last_7_days", "sales_last_30_days",
    "total_orders_last_7_days", "total_orders_last_30_days", "sales_rolling_7",
    "sales_rolling_30", "total_orders_rolling_7", "total_orders_rolling_30"
]

X_train_reduced = X_train[important_features]
X_valid_reduced = X_valid[important_features]

# Reduce dataset size for tuning (use 500,000 samples instead of full dataset)
sample_size = 500000
X_train_sample = X_train_reduced.sample(sample_size, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]

# Define a reduced hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [6, 10],  
    'learning_rate': [0.01, 0.05],  
    'subsample': [0.7, 1.0],  
    'colsample_bytree': [0.7, 1.0],  
    'gamma': [0, 5],  
    'reg_lambda': [1, 5],  
    'reg_alpha': [0, 5]
}

# Count total combinations for progress tracking
total_combinations = 30  # RandomizedSearchCV is set to run 30 iterations

print(f"Starting Hyperparameter Tuning with {total_combinations} random combinations...\n")

# Initialize XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Custom class to track progress
class TqdmSearchCV(RandomizedSearchCV):
    def _run_search(self, evaluate_candidates):
        with tqdm(total=total_combinations, desc="Hyperparameter Tuning Progress") as pbar:
            def wrapped_evaluate_candidates(candidate_params):
                pbar.update(len(candidate_params))  # Update progress bar
                return evaluate_candidates(candidate_params)
            return super()._run_search(wrapped_evaluate_candidates)

# Perform Randomized Search with progress tracking
random_search = TqdmSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=total_combinations,  
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=1,  
    random_state=42
)

# Fit the model on the smaller dataset
random_search.fit(X_train_sample, y_train_sample)

# Display the best hyperparameters
print("\nBest Hyperparameters Found:", random_search.best_params_)



import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
from tqdm import tqdm

# Keep only the top features to reduce memory
important_features = [
    "total_orders", "sell_price_main", "sales_last_7_days", "sales_last_30_days",
    "total_orders_last_7_days", "total_orders_last_30_days", "sales_rolling_7",
    "sales_rolling_30", "total_orders_rolling_7", "total_orders_rolling_30"
]

X_train_reduced = X_train[important_features]
X_valid_reduced = X_valid[important_features]

# Reduce dataset size for tuning (use 500,000 samples instead of full dataset)
sample_size = 1000000
X_train_sample = X_train_reduced.sample(sample_size, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]

# Define a reduced hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [6, 10],  
    'learning_rate': [0.01, 0.05],  
    'subsample': [0.7, 1.0],  
    'colsample_bytree': [0.7, 1.0],  
    'gamma': [0, 5],  
    'reg_lambda': [1, 5],  
    'reg_alpha': [0, 5]
}

# Count total combinations for progress tracking
total_combinations = 30  # RandomizedSearchCV is set to run 30 iterations

print(f"Starting Hyperparameter Tuning with {total_combinations} random combinations...\n")

# Initialize XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Custom class to track progress
class TqdmSearchCV(RandomizedSearchCV):
    def _run_search(self, evaluate_candidates):
        with tqdm(total=total_combinations, desc="Hyperparameter Tuning Progress") as pbar:
            def wrapped_evaluate_candidates(candidate_params):
                pbar.update(len(candidate_params))  # Update progress bar
                return evaluate_candidates(candidate_params)
            return super()._run_search(wrapped_evaluate_candidates)

# Perform Randomized Search with progress tracking
random_search = TqdmSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=total_combinations,  
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=1,  
    random_state=42
)

# Fit the model on the smaller dataset
random_search.fit(X_train_sample, y_train_sample)

# Display the best hyperparameters
print("\nBest Hyperparameters Found:", random_search.best_params_)



import xgboost as xgb
import pandas as pd
import numpy as np
import joblib  # For saving the trained model

# Keep only the top features for efficiency
important_features = [
    "total_orders", "sell_price_main", "sales_last_7_days", "sales_last_30_days",
    "total_orders_last_7_days", "total_orders_last_30_days", "sales_rolling_7",
    "sales_rolling_30", "total_orders_rolling_7", "total_orders_rolling_30"
]

X_train_reduced = X_train[important_features]
X_valid_reduced = X_valid[important_features]

# Use the best hyperparameters found
best_params = {
    'subsample': 0.7,
    'reg_lambda': 1,
    'reg_alpha': 5,
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'gamma': 5,
    'colsample_bytree': 1.0,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'tree_method': 'hist',  # Optimized for large datasets
    'n_jobs': 1  # Avoid memory issues
}

# Initialize the final XGBoost model
final_xgb_model = xgb.XGBRegressor(**best_params)

# Train the model on the full dataset
print("\nTraining the Final XGBoost Model...")
final_xgb_model.fit(X_train_reduced, y_train)

# Save the trained model (for future use)
joblib.dump(final_xgb_model, "final_xgb_model.pkl")

print("\nXGBoost Model Training Complete & Model Saved!")



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Make predictions on the validation set
y_pred_xgb = final_xgb_model.predict(X_valid_reduced)

# Compute evaluation metrics
mae_xgb = mean_absolute_error(y_valid, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_valid, y_pred_xgb))
r2_xgb = r2_score(y_valid, y_pred_xgb)

# Display results
print("\nXGBoost Model Performance on Validation Set:")
print(f"Mean Absolute Error (MAE): {mae_xgb:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_xgb:.4f}")
print(f"RÂ² Score: {r2_xgb:.4f}")



import lightgbm as lgb
import joblib  # For saving the model

# Keep only the top features for efficiency
important_features = [
    "total_orders", "sell_price_main", "sales_last_7_days", "sales_last_30_days",
    "total_orders_last_7_days", "total_orders_last_30_days", "sales_rolling_7",
    "sales_rolling_30", "total_orders_rolling_7", "total_orders_rolling_30"
]

X_train_reduced = X_train[important_features]
X_valid_reduced = X_valid[important_features]

# Define LightGBM parameters (optimized for large datasets)
best_lgb_params = {
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 1.0,
    'reg_alpha': 5,
    'reg_lambda': 1,
    'random_state': 42,
    'n_jobs': -1  # Use all available CPU cores efficiently
}

# Initialize and train the LightGBM model
print("\nTraining the Final LightGBM Model...")
final_lgb_model = lgb.LGBMRegressor(**best_lgb_params)
final_lgb_model.fit(X_train_reduced, y_train)

# Save the trained model
joblib.dump(final_lgb_model, "final_lgb_model.pkl")

print("\nLightGBM Model Training Complete & Model Saved!")



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Make predictions
y_pred_lgb = final_lgb_model.predict(X_valid_reduced)

# Compute evaluation metrics
mae_lgb = mean_absolute_error(y_valid, y_pred_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_valid, y_pred_lgb))
r2_lgb = r2_score(y_valid, y_pred_lgb)

# Display results
print("\nLightGBM Model Performance on Validation Set:")
print(f"Mean Absolute Error (MAE): {mae_lgb:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_lgb:.4f}")
print(f"RÂ² Score: {r2_lgb:.4f}")



import pandas as pd

# Create a comparison table
results = pd.DataFrame({
    "Model": ["XGBoost", "LightGBM"],
    "MAE": [mae_xgb, mae_lgb],
    "RMSE": [rmse_xgb, rmse_lgb],
    "RÂ² Score": [r2_xgb, r2_lgb]
})

# Sort results by RMSE
print("\nModel Performance Comparison:")
print(results.sort_values(by="RMSE"))



import pandas as pd
import numpy as np
import joblib

# âœ… Load the trained XGBoost model
final_xgb_model = joblib.load("final_xgb_model.pkl")

# âœ… Load the test dataset
test_df = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv")
calendar_df = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv")
inventory_df = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv")
train_df = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv")  # Needed for historical data

# âœ… Ensure the date format is correct
calendar_df["date"] = pd.to_datetime(calendar_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])
train_df["date"] = pd.to_datetime(train_df["date"])

# âœ… Merge test_df with calendar_df & inventory_df
calendar_chunk = calendar_df.drop(columns=['warehouse'])  # Remove 'warehouse' to prevent duplication
test_df = pd.merge(test_df, calendar_chunk, on="date", how="left", suffixes=("_test", "_calendar"))
test_df = pd.merge(test_df, inventory_df, on="unique_id", how="left", suffixes=("_test", "_inventory"))

# âœ… Create time-based features for the test dataset
test_df["year"] = test_df["date"].dt.year
test_df["month"] = test_df["date"].dt.month
test_df["day_of_week"] = test_df["date"].dt.dayofweek
test_df["week_of_year"] = test_df["date"].dt.isocalendar().week
test_df["is_weekend"] = (test_df["day_of_week"] >= 5).astype(int)

# âœ… Combine train and test data for correct lag/rolling calculations
combined_df = pd.concat([train_df, test_df], ignore_index=True).sort_values(["unique_id", "date"])

# âœ… Calculate lag-based features
combined_df["sales_last_7_days"] = combined_df.groupby("unique_id")["sales"].shift(7)
combined_df["sales_last_30_days"] = combined_df.groupby("unique_id")["sales"].shift(30)
combined_df["total_orders_last_7_days"] = combined_df.groupby("unique_id")["total_orders"].shift(7)
combined_df["total_orders_last_30_days"] = combined_df.groupby("unique_id")["total_orders"].shift(30)

# âœ… Calculate rolling window features
combined_df["sales_rolling_7"] = combined_df.groupby("unique_id")["sales"].transform(lambda x: x.rolling(7, min_periods=1).mean())
combined_df["sales_rolling_30"] = combined_df.groupby("unique_id")["sales"].transform(lambda x: x.rolling(30, min_periods=1).mean())
combined_df["total_orders_rolling_7"] = combined_df.groupby("unique_id")["total_orders"].transform(lambda x: x.rolling(7, min_periods=1).mean())
combined_df["total_orders_rolling_30"] = combined_df.groupby("unique_id")["total_orders"].transform(lambda x: x.rolling(30, min_periods=1).mean())

# âœ… Extract only the test rows
test_df = combined_df[combined_df["date"].isin(test_df["date"])]

# âœ… Keep only the top features used in training
important_features = [
    "total_orders", "sell_price_main", "sales_last_7_days", "sales_last_30_days",
    "total_orders_last_7_days", "total_orders_last_30_days", "sales_rolling_7",
    "sales_rolling_30", "total_orders_rolling_7", "total_orders_rolling_30"
]

# âœ… Select features from test set
X_test_reduced = test_df[important_features]

# âœ… Handle missing values in test set (use median imputation)
X_test_reduced.fillna(X_test_reduced.median(), inplace=True)

# âœ… Generate predictions using the trained model
test_df["sales_pred"] = final_xgb_model.predict(X_test_reduced)

# âœ… Load test weights and apply them to predictions
test_weights = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv")
test_df = test_df.merge(test_weights, on="unique_id", how="left")

# âœ… Apply weighting to predictions
test_df["weighted_sales_pred"] = test_df["sales_pred"] * test_df["weight"]

# âœ… Prepare final submission file
submission = test_df[["unique_id", "weighted_sales_pred"]].rename(columns={"weighted_sales_pred": "sales"})

# âœ… Save submission file
submission.to_csv("submission.csv", index=False)

print("\nâœ… Submission File Created: 'submission.csv' | Ready for Kaggle Submission! ðŸš€")



import os
print(os.listdir("/kaggle/working/"))



from IPython.display import FileLink
FileLink("submission.csv")





