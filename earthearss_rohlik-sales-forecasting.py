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


# Import packages
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning tools
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# # Advanced models (optional for later)
# import xgboost as xgb
# import lightgbm as lgb


# File paths
sales_train_path = "/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv"
sales_test_path = "/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv"
inventory_path = "/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv"
calendar_path = "/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv"
test_weights_path = "/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv"

# Load the datasets
sales_train = pd.read_csv(sales_train_path)
sales_test = pd.read_csv(sales_test_path)
inventory = pd.read_csv(inventory_path)
calendar = pd.read_csv(calendar_path)
test_weights = pd.read_csv(test_weights_path)


from sklearn.metrics import mean_absolute_error

# Compute WMAE
def calculate_wmae(actual, predicted, weights):
    return np.sum(weights * np.abs(actual - predicted)) / np.sum(weights)


sales_train.info()


# Convert date columns to datetime format
sales_train['date'] = pd.to_datetime(sales_train['date'])
sales_test['date'] = pd.to_datetime(sales_test['date'])
calendar['date'] = pd.to_datetime(calendar['date'])


sales_train.info()


sales_train.head()


# Check for missing values
def check_missing_values(df):
    return df.isnull().sum()

print("Missing values in Sales Train:")
print(check_missing_values(sales_train))

print("Missing values in Sales Test:")
print(check_missing_values(sales_test))


# Drop non-numeric columns
numeric_sales_train = sales_train.select_dtypes(include=['float64', 'int64'])

# Compute correlation matrix
plt.figure(figsize=(10, 8))
corr = numeric_sales_train.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap")
plt.show()


# Aggregate sales by date
sales_trend = sales_train.groupby('date')['sales'].sum()

# Plot the trend
plt.figure(figsize=(14, 6))
plt.plot(sales_trend.index, sales_trend.values, label="Total Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Trend Over Time")
plt.legend()
plt.grid()
plt.show()



# Histogram for total_orders
plt.figure(figsize=(10, 6))
sns.histplot(data=sales_train, x='total_orders', bins=30, kde=False)
plt.title("Histogram of Total Orders")
plt.xlabel("Total Orders")
plt.ylabel("Frequency")
plt.grid()
plt.show()



# Box plot of type_0_discount
plt.figure(figsize=(10, 6))
sns.boxplot(data=sales_train, x='type_0_discount', color='skyblue')
plt.title("Box Plot of Type 0 Discount")
plt.xlabel("Type 0 Discount")
plt.grid()
plt.show()



sales_train['type_0_discount'].max()


sales_train['type_0_discount'].min()


# Count of type_0_discount values less than 0
negative_discounts = (sales_train['type_0_discount'] < 0).sum()
print(f"Number of type_0_discount values less than 0: {negative_discounts}")


print(sales_train.groupby('warehouse')['sell_price_main'].mean())
print(sales_train.groupby('warehouse')['sell_price_main'].sum())


from sklearn.preprocessing import StandardScaler

# Standardise sell_price_main
scaler = StandardScaler()
sales_train['sell_price_main_scaled'] = scaler.fit_transform(sales_train[['sell_price_main']])
sales_test['sell_price_main_scaled'] = scaler.fit_transform(sales_test[['sell_price_main']])
sales_train.head()


from sklearn.linear_model import LinearRegression

# Define features and target
features = ['total_orders', 'sell_price_main_scaled', 'type_0_discount', 'type_2_discount', 'type_6_discount']  # Add more features if needed
target = 'sales'

# Drop rows with missing values in the features and target
sales_train_cleaned = sales_train.dropna(subset=features + [target])
sales_test_cleaned = sales_test.dropna(subset=features)

# Train-test split
X_train = sales_train_cleaned[features]
y_train = sales_train_cleaned[target]
X_test = sales_test_cleaned[features]


# Initialise and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict sales on the test set
sales_test_cleaned['sales_hat'] = model.predict(X_test)


# Merge weights with the test data
sales_test_cleaned = sales_test_cleaned.merge(test_weights, on='unique_id', how='left')

# Example: Replace `sales_test_cleaned['sales']` with actual test targets when available
sales_test_cleaned['sales'] = 0  # Placeholder, replace with actual sales if available
wmae = calculate_wmae(
    actual=sales_test_cleaned['sales'], 
    predicted=sales_test_cleaned['sales_hat'], 
    weights=sales_test_cleaned['weight']
)
print(f"WMAE: {wmae}")


sales_test_cleaned.head()


from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge

# Drop rows with missing values
sales_train_cleaned = sales_train.dropna()

# Prepare data for training
X = sales_train_cleaned.drop(columns=['sales', 'unique_id', 'warehouse', 'date', 'availability'])
y = sales_train_cleaned['sales']

# TimeSeriesSplit setup
tscv = TimeSeriesSplit(n_splits=10)  # Adjust n_splits based on your data size


# Ridge regression with grid search for alpha
ridge = Ridge()
param_grid = {'alpha': np.logspace(-1.3, -0.3, 10)}  # Log scale for alpha between 0.05 and 0.5
grid_search = GridSearchCV(ridge, param_grid, cv=tscv, scoring='neg_mean_absolute_error')

# Perform grid search
grid_search.fit(X, y)

# Best model and alpha
best_model = grid_search.best_estimator_
best_alpha = grid_search.best_params_['alpha']

print(f"Best alpha: {best_alpha}")
print(f"Best Validation MAE: {-grid_search.best_score_}")


# Predict on test set
X_test = sales_test.drop(columns=['unique_id', 'warehouse', 'date'], errors='ignore')
X_test = X_test.fillna(0)  # Handle missing values in test set
sales_test['sales_hat'] = best_model.predict(X_test)

# Merge test weights and calculate WMAE
sales_test = sales_test.merge(test_weights, on='unique_id', how='left')

# Define WMAE calculation
def calculate_wmae(actual, predicted, weights):
    return np.sum(weights * np.abs(actual - predicted)) / np.sum(weights)

# Placeholder for actual sales (replace if actual values are available)
if 'sales' not in sales_test.columns:
    sales_test['sales'] = 0  # Placeholder for actual test sales

# Calculate WMAE
wmae = calculate_wmae(
    actual=sales_test['sales'], 
    predicted=sales_test['sales_hat'], 
    weights=sales_test['weight']
)
print(f"WMAE: {wmae}")


sales_train


sales_test


# # Add a data_split column to differentiate datasets
# sales_train['data_split'] = 'train'
# sales_test['data_split'] = 'test'

# # Add sales_hat to sales_train (set as NaN since we don't have predictions for train data)
# sales_train['sales_hat'] = np.nan

# # Combine both datasets
# combined_data = pd.concat([sales_train, sales_test], ignore_index=True)

# # Filter for 2024 data only
# combined_data_2024 = combined_data[combined_data['date'].dt.year == 2024]

# # Group by date to calculate total sales and predictions
# trend_data = combined_data_2024.groupby(['date', 'data_split'])[['sales', 'sales_hat']].sum().reset_index()

# # Plotting the trend
# plt.figure(figsize=(15, 8))
# sns.lineplot(data=trend_data, x='date', y='sales', hue='data_split', label='Actual Sales')
# sns.lineplot(data=trend_data, x='date', y='sales_hat', hue='data_split', linestyle='--', label='Predicted Sales')
# plt.title('Sales and Sales Predictions Over Time (2024)')
# plt.xlabel('Date')
# plt.ylabel('Sales')
# plt.legend()
# plt.grid()
# plt.show()


from prophet import Prophet

# Example using a subset of IDs (for quicker testing)
unique_ids = sales_train['unique_id'].unique()[:3]  # Adjust range as needed
results = []

for unique_id in unique_ids:
    # Filter training data for the current unique_id
    df_id = sales_train[sales_train['unique_id'] == unique_id][['date', 'sales']]
    df_id = df_id.rename(columns={'date': 'ds', 'sales': 'y'})
    
    # Check for sufficient data
    if len(df_id) < 20:  # Skip IDs with very few data points
        print(f"Skipping unique_id {unique_id} due to insufficient data.")
        continue
    
    # Fit Prophet model
    model = Prophet()
    model.fit(df_id)
    
    # Prepare test data for the current unique_id
    test_id = sales_test[sales_test['unique_id'] == unique_id][['date']]
    test_id = test_id.rename(columns={'date': 'ds'})
    test_id = test_id.drop_duplicates()  # Ensure no duplicates
    
    # Predict on test data
    forecast = model.predict(test_id)
    forecast['unique_id'] = unique_id
    
    # Store results
    results.append(forecast[['ds', 'yhat', 'unique_id']])

# Combine all forecasts
final_forecast = pd.concat(results, ignore_index=True)



final_forecast


# Ensure no redundant merge with test_weights
sales_test_eval = sales_test[['date', 'unique_id', 'sales', 'weight']]

# Debugging: Verify columns after selecting necessary ones
print("Columns in sales_test_eval:", sales_test_eval.columns)

# Merge with final_forecast
merged = pd.merge(
    sales_test_eval,
    final_forecast,
    left_on=['date', 'unique_id'],
    right_on=['ds', 'unique_id'],
    how='inner'
)

# Define WMAE calculation
def calculate_wmae(actual, predicted, weights):
    return np.sum(weights * np.abs(actual - predicted)) / np.sum(weights)

# Calculate WMAE
wmae = calculate_wmae(
    actual=merged['sales'], 
    predicted=merged['yhat'], 
    weights=merged['weight']
)
print(f"WMAE: {wmae}")



sales_train.info()


sales_train['unique_id'].nunique()


from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# Convert unique_id to categorical or correct its values
sales_train['unique_id'] = sales_train['unique_id'].astype('object')

# Identify categorical features
categorical_features = ['warehouse', 'unique_id']  # Add other categorical column names if needed

# Define features and target
X = sales_train.drop(columns=['date', 'sales', 'availability'], errors='ignore')  # Features
y = sales_train['sales']  # Target

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Remove rows with NaN values in target
X_train = X_train[y_train.notnull()]
y_train = y_train[y_train.notnull()]
X_val = X_val[y_val.notnull()]
y_val = y_val[y_val.notnull()]

# Initialize CatBoost regressor
cat_model = CatBoostRegressor(
    iterations=500,         # Maximum boosting rounds
    learning_rate=1,      # Step size for boosting
    depth=8,                # Tree depth
    cat_features=categorical_features,  # Specify categorical features
    loss_function='MAE',    # Loss function (Mean Absolute Error for sales prediction)
    random_seed=42,
    verbose=50,             # Log progress every 50 iterations
    early_stopping_rounds=50  # Stop if no improvement for 50 iterations
)

# Train the model with early stopping
cat_model.fit(
    X_train, 
    y_train, 
    eval_set=(X_val, y_val), 
    use_best_model=True      # Retain the best iteration during training
)

# Evaluate on validation set
y_val_pred = cat_model.predict(X_val)
mae = mean_absolute_error(y_val, y_val_pred)
print(f"Validation MAE: {mae}")



# Predict on test data
X_test = sales_test.drop(columns=['unique_id', 'date', 'sales', 'availability'], errors='ignore')  # Drop unnecessary columns
X_test = X_test.fillna(0)  # Handle missing values
sales_test['sales_hat'] = cat_model.predict(X_test)




