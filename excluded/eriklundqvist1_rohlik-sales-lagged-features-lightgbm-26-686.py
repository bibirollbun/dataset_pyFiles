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


# Import some more libraries
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# Do not truncate the display of DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# Read the data
train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
test = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv")
inventory = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv")
calendar = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv")
test_weights = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv")
solution = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv")


# Merge data
train = pd.merge(train, inventory, how='left', on=['unique_id', 'warehouse'])
train = pd.merge(train, calendar, how='left', on=['date', 'warehouse'])
test = pd.merge(test, inventory, how='left', on=['unique_id', 'warehouse'])
test = pd.merge(test, calendar, how='left', on=['date', 'warehouse'])
print(f"Shape of train: {train.shape}")  
print(f"Shape of test: {test.shape}")


# Clean up duplicate columns
y_columns = [col for col in train.columns if col.endswith('_y')]
train = train.drop(columns=y_columns)
test = test.drop(columns=y_columns)
train = train.rename(columns={col: col.replace('_x', '') for col in train.columns if col.endswith('_x')})
test = test.rename(columns={col: col.replace('_x', '') for col in test.columns if col.endswith('_x')})

print(f"Shape of train: {train.shape}")  
print(f"Shape of test: {test.shape}")



# Check for any NaN values after merge
print("NaN values in train:", train.isnull().sum())



# total_orders and sales only missing for 52 out of 
# 4 million rows is negligible, drop them
train = train.dropna(subset=['total_orders', 'sales'])


print(train.shape)


print("NaN values in test:", test.isnull().sum())


# Assign holiday_name = NaN a value ("No Holiday") in both train and test
train['holiday_name'] = train['holiday_name'].fillna('No Holiday')
test['holiday_name'] = test['holiday_name'].fillna('No Holiday')


# Convert date columns to datetime
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
calendar["date"] = pd.to_datetime(calendar["date"])


# Let's look at basic info about train data
print("Train data shape: ", train.shape)
print("\nFirst few rows of training data:")
train.head()


# Train data types:
print(train.dtypes)


print(test.dtypes)


# Date range in train:
print(f"Train dates: {train['date'].min()} to {train['date'].max()}")


# Test date range
print(f"Test dates: {test['date'].min()} to {test['date'].max()}")
# Two weeks of data that ends right after train data ends
# Could use last two weeks of train data as validation


##### EDA #####


# Variable to plot
daily_sales = train.groupby("date")["sales"].sum().reset_index()


# Plot sales distribution
plt.figure(figsize=(12,6))
sns.histplot(train["sales"], bins=50, kde=True)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.show()


daily_sales["sales_30d_avg"] = daily_sales["sales"].rolling(30).mean()  # Changed from 7 to 30 days
plt.figure(figsize=(15,6))
plt.plot(daily_sales["date"], daily_sales["sales"], label="Daily Sales")
plt.plot(daily_sales["date"], daily_sales["sales_30d_avg"], label="30-Day Avg", color="red")  # Updated label
plt.title("Daily Sales with 30-Day Moving Average")
plt.legend()
plt.show()


# Daily sales per warehouse to see the relative volumes
warehouse_daily = train.groupby(["warehouse", "date"])["sales"].sum().reset_index()

# Remove warnings to avoid clutter
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Plotting
plt.figure(figsize=(15,6))
sns.lineplot(data=warehouse_daily, x="date", y="sales", hue="warehouse")
plt.title("Daily Sales by Warehouse")
plt.show()


# Get unique warehouses
warehouses = warehouse_daily["warehouse"].unique()

# Calculate number of rows and columns for subplot grid
n_warehouses = len(warehouses)
n_cols = 2  
n_rows = (n_warehouses + n_cols - 1) // n_cols  # Ceiling division to ensure enough rows

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
fig.suptitle("Daily Sales and 30-Day Moving Average for Each Warehouse", fontsize=16, y=1.02)  # Add main title
axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

for idx, (ax, warehouse) in enumerate(zip(axes, warehouses)):
    subset = warehouse_daily[warehouse_daily["warehouse"] == warehouse]
    # Plot daily sales
    sns.lineplot(data=subset, x="date", y="sales", ax=ax, alpha=0.5, label='Daily Sales')
    # Add 30-day moving average
    moving_avg = subset["sales"].rolling(30).mean()
    sns.lineplot(data=subset, x="date", y=moving_avg, ax=ax, color='red', label='30-Day Avg')
    
    ax.set_title(f"Sales for {warehouse}")
    ax.set_ylabel("Sales")
    ax.legend()

# Remove any empty subplots
for idx in range(len(warehouses), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()


# Relatively little seasonal variation in Budapest_1 and Frankfurt_1
# Warehouses mostly show a positive sales trend


# Extract year from date in train and test
train['year'] = train['date'].dt.year
test['year'] = test['date'].dt.year

# Calculate total sales per warehouse per year
warehouse_yearly = train.groupby(['warehouse', 'year'])['sales'].sum().reset_index()

# Calculate company-wide yearly totals
company_yearly = train.groupby('year')['sales'].sum().reset_index()


plt.figure(figsize=(12, 6))
sns.barplot(x='year', y='sales', hue='warehouse', data=warehouse_yearly)
plt.title('Yearly Sales by Warehouse')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.legend(title='Warehouse')
plt.show()


plt.figure(figsize=(10, 5))
sns.barplot(x='year', y='sales', data=company_yearly, color='purple')
plt.title('Company-Wide Yearly Sales')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.show()

# Growth hasn't been that big since 2021,
# could indicate a declining interest since Covid


# Create day_of_week column for train and test
train['day_of_week'] = train['date'].dt.dayofweek  # 0=Monday, 6=Sunday
test['day_of_week'] = test['date'].dt.dayofweek  # 0=Monday, 6=Sunday


# Then calculate total sales per day
daily_sales = train.groupby('day_of_week')['sales'].sum().reset_index()

# Create bar plot
plt.figure(figsize=(10,6))
sns.barplot(x='day_of_week', y='sales', data=daily_sales, color='skyblue')
plt.title('Total Sales by Day of Week (0 = Monday)')
plt.xlabel('Day of Week')
plt.ylabel('Total Sales')
plt.show()

# Friday is the day with the most volume, weekends the least


# Total Sales Per Month

# Extract month from date for train and test
train['month'] = train['date'].dt.month
test['month'] = test['date'].dt.month

# Calculate total sales per month
monthly_sales = train.groupby('month')['sales'].sum().reset_index()

# Map month numbers to names
month_names = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
monthly_sales['month_name'] = monthly_sales['month'].map(month_names)

# Create plot
plt.figure(figsize=(12, 6))
sns.barplot(x='month_name', y='sales', data=monthly_sales, color='teal')
plt.title('Total Sales by Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()

# June and July have by far the least volume
# The dates to predict sales for are two weeks in June,
# which might make it more difficult


# Quick check of discount impact on sales
train['has_any_discount'] = (train[['type_0_discount', 'type_1_discount', 'type_2_discount', 
                                  'type_3_discount', 'type_4_discount', 'type_5_discount', 
                                  'type_6_discount']] > 0).any(axis=1)

print("Average sales with/without discounts:")
print(train.groupby('has_any_discount')['sales'].mean())
# Discounts seem to have a big impact on sales


# Quick check of holiday impact
print("\nAverage sales on holidays vs non-holidays:")
print(train.groupby('holiday')['sales'].mean())



##### END OF EDA #####


train.head()


test.head()


for df in [train, test]:
    df['has_any_discount'] = (df[['type_0_discount', 'type_1_discount', 'type_2_discount', 
                                'type_3_discount', 'type_4_discount', 'type_5_discount', 
                                'type_6_discount']] > 0).any(axis=1).astype(int)



# Sort data by ID and date 
train = train.sort_values(['unique_id', 'date'])
test = test.sort_values(['unique_id', 'date'])


# Shapes before lag feaures:
print(f"Train shape after lag features: {train.shape}")
print(f"Test shape after lag features: {test.shape}")


# Combine train and test
test['sales'] = np.nan  # Add empty sales column to test
combined_data = pd.concat([train, test]).sort_values(['unique_id', 'date'])

# Create lag features 
combined_data['sales_7days_ago'] = combined_data.groupby('unique_id')['sales'].shift(7)
combined_data['sales_14days_ago'] = combined_data.groupby('unique_id')['sales'].shift(14)
combined_data['sales_28days_ago'] = combined_data.groupby('unique_id')['sales'].shift(28)
combined_data['sales_365days_ago'] = combined_data.groupby('unique_id')['sales'].shift(365)

# 2. Fill NaN values in lag features
for lag_col in ['sales_7days_ago', 'sales_14days_ago', 'sales_28days_ago', 'sales_365days_ago']:
    combined_data[lag_col] = combined_data.groupby('unique_id')[lag_col].fillna(method='ffill')
    combined_data[lag_col] = combined_data[lag_col].fillna(0)  # Fill remaining NaNs with 0


# Split back into train and test
train = combined_data[combined_data['sales'].notna()].copy()
test = combined_data[combined_data['sales'].isna()].copy()

# Drop rows with missing lag features
train = train.dropna()


# Fill missing lag values in test with mean of recent values
for lag_col in ['sales_7days_ago', 'sales_14days_ago', 'sales_28days_ago', 'sales_365days_ago']:
    combined_data[lag_col] = combined_data.groupby('unique_id')[lag_col].fillna(method='ffill')
    # If still any NaNs, fill with 0
    combined_data[lag_col] = combined_data[lag_col].fillna(0)



# Shapes after lag feaures:
print(f"Train shape after lag features: {train.shape}")


# Use last 14 days as validation (same length as test period)
# Could use 2 weeks in June 2023 as sanity check
# Use last 14 days in train as validation
val_start_date = '2024-05-17'
val_mask = train['date'] >= val_start_date
train_data = train[~val_mask]  # Exclude validation period
val_data = train[val_mask]

# Verify the dates
print(f"Train data: {train_data['date'].min()} to {train_data['date'].max()}")
print(f"Validation: {val_data['date'].min()} to {val_data['date'].max()}")
print(f"Test: {test['date'].min()} to {test['date'].max()}")


def prepare_features(df):
    # Prepare features for modeling
    
    # Discount feature if it doesn't exist
    if 'has_any_discount' not in df.columns:
        df['has_any_discount'] = (df[['type_0_discount', 'type_1_discount', 'type_2_discount', 
                                    'type_3_discount', 'type_4_discount', 'type_5_discount', 
                                    'type_6_discount']] > 0).any(axis=1).astype(int)
    
    feature_cols = [
        'day_of_week', 'month', 'year', 
        'has_any_discount', 'holiday', 'shops_closed',
        'winter_school_holidays', 'school_holidays',
        'warehouse', 'total_orders', 'sell_price_main',
        'L1_category_name_en', 'L2_category_name_en', 'L3_category_name_en', 'L4_category_name_en',  
        'sales_7days_ago', 'sales_14days_ago', 'sales_28days_ago', 'sales_365days_ago'  
    ]
    
    return df[feature_cols]



# Prepare features for each dataset
X_train = prepare_features(train_data)
y_train = train_data['sales']
X_val = prepare_features(val_data)
y_val = val_data['sales']
X_test = prepare_features(test)


# Merge weights before modeling
train_with_weights = pd.merge(train_data, test_weights, on='unique_id', how='left')
val_with_weights = pd.merge(val_data, test_weights, on='unique_id', how='left')


# Check that data looks good before modeling:
# Check shapes
print("Shapes:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}")

# Check all have same columns
print("\nAll datasets have same columns?")
print(f"X_train columns: {X_train.columns.tolist()}")
print(f"X_val columns: {X_val.columns.tolist()}")
print(f"X_test columns: {X_test.columns.tolist()}")

# Check for any missing values
print("\nMissing values:")
print("X_train:", X_train.isnull().sum().sum())
print("X_val:", X_val.isnull().sum().sum())
print("X_test:", X_test.isnull().sum().sum())

# Quick look at feature ranges
print("\nFeature ranges in train vs test:")
for col in X_train.select_dtypes(include=['int64', 'float64']).columns:
    print(f"\n{col}:")
    print(f"Train range: {X_train[col].min():.2f} to {X_train[col].max():.2f}")
    print(f"Test range: {X_test[col].min():.2f} to {X_test[col].max():.2f}")


# Note: No holidays in the test set


test.head()


train.head()


print("Missing values in test features:")
print(X_test.isnull().sum())


# Check data types before modeling
print(f"X_train types\n: {X_train.dtypes}\n")
print(f"y_train types\n: {y_train.dtypes}\n")
print(f"X_val types\n: {X_val.dtypes}\n")
print(f"y_val types\n: {y_val.dtypes}\n")
print(f"X_test types\n: {X_test.dtypes}\n")


# TIME TO START MODELING!


import lightgbm as lgb


# Define categorical features
categorical_features = ['warehouse', 'L1_category_name_en', 'L2_category_name_en', 'L3_category_name_en', 'L4_category_name_en']


X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()

for col in categorical_features:
    X_train[col] = X_train[col].astype('category')
    X_val[col] = X_val[col].astype('category')
    X_test[col] = X_test[col].astype('category')



# Train with validation
params = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8
}



# Create datasets for LightGBM model
train_dataset = lgb.Dataset(
    X_train, 
    label=y_train,
    categorical_feature=categorical_features,
    weight=train_with_weights['weight']
)

val_dataset = lgb.Dataset(
    X_val, 
    label=y_val,
    categorical_feature=categorical_features,
    weight=val_with_weights['weight']
)



# Train model
model = lgb.train(
    params,
    train_dataset,
    num_boost_round=100,
    valid_sets=[train_dataset, val_dataset],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=10),
        lgb.log_evaluation(period=10)
    ]
)


# WMAE using only L1_category_name as name of internal category,
# was 26.3358, using all 4 showed improvement


# Check feature importances
importance = pd.DataFrame({
    'feature': model.feature_name(),
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)


import matplotlib.pyplot as plt

# Create bar plot of feature importances
plt.figure(figsize=(12, 6))
importance.plot(kind='bar', x='feature', y='importance')
plt.title('Feature Importance in LightGBM Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Make predictions and create submission
test_predictions = model.predict(X_test)


# Create submission file
test_ids = test['unique_id'].astype(str) + '_' + test['date'].dt.strftime('%Y-%m-%d')
submission = pd.DataFrame({
    'id': test_ids,
    'sales_hat': test_predictions
})


# Quick sanity check
print("Submission sample:")
print(submission.head())
print("\nShape:", submission.shape)
print("\nCheck for any negative predictions:")
print((submission['sales_hat'] < 0).sum())



# Save submission
submission.to_csv('submission.csv', index=False)

