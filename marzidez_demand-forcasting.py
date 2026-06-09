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


import matplotlib.pyplot as plt
import seaborn as sns
import os
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit


train_df = pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/train.csv", parse_dates  = ['date'])
test_df = pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/test.csv", parse_dates = ['date'])

# parse_date = [] converts the strings its find there into real datetime64[ns]


print("train Shape:", train_df.shape)
print("Tese Shape:", test_df.shape)


print("\nTrain Head:")
print(train_df.head)
print("\nTest Head:")
print(test_df.head())


print("\nTrain Info:")
train_df.info()
print("\nTest Info:")
test_df.info()


print("\Missing Values in Train:")
print(train_df.isnull().sum())
print("\nMissing Values in Test:")
print(test_df.isnull().sum())

# There are no NaN values.


print("Training Date Range: {} to {}".format(train_df['date'].min(), train_df['date'].max()))
print("\nTest Date Range: {} to {}".format(test_df['date'].min(), test_df['date'].max()))

# Train Data : 5 years and 1 day
# Test Data: 90 days


print("\nUnique Stores (Train_df): ", train_df['store'].nunique())
print("\nUnique Items (Train-df): ", train_df['item'].nunique())
print("\nUnique Stores (Tese_df): ", test_df['store'].unique())
print("\nUnique Items (Test_df): ",test_df['item'].nunique())

# Cause im not seeing any NaN values and our date is no datetime type and store and item are integers.
# I'll move on to the next step and more analysis maybe.


# Fining out the overall sales trends:

# Having date as index so that we can do time series plotting easier

train_df_indexed = train_df.set_index('date').sort_index() # Moving one column more into the index and setting the order ascending

plt.figure(figsize=(15, 6))
train_df_indexed['sales'].resample('D').sum().plot(title='Total Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid(True)
plt.tight_layout()
plt.show()

# This simple line chart shows a near 3x increase in five years.


plt.figure(figsize=(15, 6))
train_df_indexed['sales'].resample('W').sum().plot(title='Total Weekly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid(True)
plt.tight_layout()
plt.show()

# This line is smooth and nearly linear


yearly = train_df_indexed['sales'].resample('Y').sum()
year_over_year = yearly.pct_change()*100
plt.figure(figsize=(8, 4))
sns.barplot(x=year_over_year.index.year[1:], y=year_over_year.values[1:])
plt.title('Year-over-Year % Growth')
plt.ylabel('Sales % Change')
plt.tight_layout()
plt.show()


train_df_indexed['month'] = train_df_indexed.index.month
plt.figure(figsize=(12,4))
sns.boxplot(x='month', y='sales', data=train_df_indexed)
plt.title('Monthly Sales Distribution (all years)')
plt.ylabel('Sales')
plt.tight_layout()
plt.show()

# median sales peak in summer (7-8) and a drop in winter (12)
# Dec(12) we have the lowest median
# There are dots above the whiskers but cluster above July(6) and August(7) whcih shows a demand spike


daily = train_df_indexed['sales'].resample('D').sum()
rolling = daily.rolling(window=30).agg(['mean','std'])
plt.figure(figsize=(15,5))
rolling['mean'].plot(label='30-day rolling mean')
rolling['std'].plot(label='30-day rolling std', alpha=0.7)
plt.title('Overall Trend & Volatility')
plt.legend()
plt.tight_layout()
plt.show()

# Trend (blue) â€“ 30-day rolling mean climbs steadily from ~15 k (2013) to â‰ˆ35 k (2017) â€“ a 2.3Ã— growth with no major reversals.
# Volatility (orange) â€“ 30-day rolling standard deviation tracks the same upward path, telling us bigger swings appear as the absolute level rises; peaks around Â±5 k by 2017.
# the mean/std ratio stays roughly constant, so relative volatility is stableâ€”the series is purely multiplicative (good to log-transform before modelling)


train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day_of_week'] = train_df['date'].dt.dayofweek # Monday=0, Sunday=6
train_df['day_of_year'] = train_df['date'].dt.dayofyear
train_df['week_of_year'] = train_df['date'].dt.isocalendar().week.astype(int)


plt.figure(figsize=(10, 5))
sns.lineplot(data=train_df.groupby('month')['sales'].mean().reset_index(), x='month', y='sales')
plt.title('Average Sales by Month')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.grid(True)
plt.tight_layout()
plt.show()

# The line chart shows average sales by month, peaking in July (month 7) and dipping to the lowest in January (month 1) and December (month 12). 


plt.figure(figsize=(10, 5))
sns.lineplot(data=train_df.groupby('day_of_week')['sales'].mean().reset_index(), x='day_of_week', y='sales')
plt.title('Average Sales by Day of Week (0=Monday, 6=Sunday)')
plt.xlabel('Day of Week')
plt.ylabel('Average Sales')
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 6))
sns.lineplot(data=train_df.groupby('day_of_year')['sales'].mean().reset_index(), x='day_of_year', y='sales')
plt.title('Average Sales by Day of Year')
plt.xlabel('Day of Year')
plt.ylabel('Average Sales')
plt.grid(True)
plt.tight_layout()
plt.show()

#A gradual increase from the start of the year, with sales peaking around mid-year (roughly day 180â€“210).
#A weekly oscillation pattern is visible, indicating recurring short-term fluctuations.
#After the peak, sales decline gradually, with a sharp drop near the end of the year (around day 330).
#There's a slight recovery right at the year's end.


plt.figure(figsize=(10, 5))
sns.barplot(x=train_df.groupby('item')['sales'].sum().index, y=train_df.groupby('item')['sales'].sum().values)
plt.title('Total Sales per Item')
plt.xlabel('Item ID')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.tight_layout()
plt.show()

# Item 14 and 28 are top ones near 1.6 million each
# Item 4 and 33 are much lower near 0.4 million each.


plt.figure(figsize=(10, 5))
sns.barplot(x=train_df.groupby('store')['sales'].sum().index, y=train_df.groupby('store')['sales'].sum().values)
plt.title('Total Sales per Store')
plt.xlabel('Store ID')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.show()
# Store 2 is the highest one around 6.2 million
# Store 8 is the second one around 5.9 milion
# Store 7 is the lowest one around 3.3 million


plt.figure(figsize=(10, 5))
sns.histplot(train_df['sales'], bins=50, kde=True)
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

print(f"Number of zero sales: {(train_df['sales'] == 0).sum()}")

# A sale value of 30 to 4o$ shows the most frequent amount(over 7000 times)
# A right skewed distribution
# most sales are for smaller amounts, while high-value sales are much less common


def create_time_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int) # 1 if weekend, 0 otherwise
    return df


train_df = create_time_features(train_df)
test_df = create_time_features(test_df)

print("Time features created for train and test sets.")
train_df.head()
# now we have a cleaner and more analyzable table


# Lag features are past values of a time series. 
# For this problem, knowing the sales for an item in 
# a specific store 90 days ago or a year ago is highly predictive of the sales today. 
# We will create lags that correspond to the 90-day test period and yearly seasonality.

combined_df = pd.concat([train_df, test_df], sort=False, ignore_index=True) #This ensures that we can calculate lags for the start of the test set using values from the end of the train set.

# Sort by store, item, and date to ensure correct sequential order
combined_df.sort_values(by=['store', 'item', 'date'], inplace=True)


lag_periods = [90, 91, 98, 180, 364, 365]

for lag in lag_periods:
    combined_df[f'sales_lag_{lag}'] = combined_df.groupby(['store', 'item'])['sales'].transform(lambda x: x.shift(lag))

combined_df[combined_df['item']==1].tail()
# Lag: Using past data to predict the future.


rolling_windows = [7, 14, 28, 90]
shift_period = 90

for window in rolling_windows:
    combined_df[f'sales_rolling_mean_{window}'] = combined_df.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.shift(shift_period).rolling(window).mean()
    )
    combined_df[f'sales_rolling_std_{window}'] = combined_df.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.shift(shift_period).rolling(window).std()
    )

# Rolling-window features = summary stats (mean, std, min, max, â€¦) calculated over the last 90 observations as the window â€œrollsâ€� through time.
# They capture recent trend & volatility without looking into the future.


print(f"Shape of combined_df before split: {combined_df.shape}")
print(f"Number of NaN sales in combined_df: {combined_df['sales'].isna().sum()}")


# Spliting the data using .copy() to be safe
train_final_df = combined_df[combined_df['sales'].notna()].copy()
test_final_df = combined_df[combined_df['sales'].isna()].copy()

print("Shape of train_final_df immediately after split:", train_final_df.shape)
print("Shape of test_final_df immediately after split:", test_final_df.shape)


# Filling NaN values created by feature engineering
train_final_df.fillna(-1, inplace=True)
test_final_df.fillna(-1, inplace=True)

print("Shape of train_final_df after filling NaNs:", train_final_df.shape)


# The 'sales' column in test_final_df is all -1 now
if 'sales' in test_final_df.columns:
    test_final_df.drop('sales', axis=1, inplace=True)

print("Final Train Shape:", train_final_df.shape)
print("Final Test Shape:", test_final_df.shape)


# Features to apply target encoding to
target_encode_features = ['store', 'item', 'month', 'day_of_week', 'is_weekend', 'quarter']

# Calculating global mean sales to fill any potential missing values in the test set
global_mean_sales = train_final_df['sales'].mean()
print(f"Global mean sales: {global_mean_sales:.2f}")


for feature in target_encode_features:
    # Calculating the mean sales for each category in the training set
    mean_map = train_final_df.groupby(feature)['sales'].mean()
    
    # Creating the new feature name
    new_feature_name = f'{feature}_target_enc'
    
    # Maping the mean values to the training and test sets
    train_final_df[new_feature_name] = train_final_df[feature].map(mean_map)
    test_final_df[new_feature_name] = test_final_df[feature].map(mean_map)
    
    # Filling any potential NaN values in the test set with the global mean
    # cause there might be a category in test but not in train
    test_final_df[new_feature_name].fillna(global_mean_sales, inplace=True)


print("\nTarget encoding complete. New features added.")
print("Train set with new features:")
print(train_final_df[['date', 'store', 'item', 'sales', 'store_target_enc', 'item_target_enc']].head())

print("\nTest set with new features:")
print(test_final_df[['date', 'store', 'item', 'id', 'store_target_enc', 'item_target_enc']].head())


print("\nFinal shapes after target encoding:")
print("Final Train Shape:", train_final_df.shape)
print("Final Test Shape:", test_final_df.shape)


# Categorical Feature Encoding
print("Data types of key categorical columns:")
print(train_final_df[['store', 'item']].dtypes)


# Harmonic Features for Seasonality
def create_harmonic_features(df):
    # For yearly seasonality
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # For weekly seasonality
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # For monthly seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

train_final_df = create_harmonic_features(train_final_df)
test_final_df = create_harmonic_features(test_final_df)

print("Harmonic features created for both train and test sets.")
print(train_final_df[['date', 'day_of_week', 'day_of_week_sin', 'day_of_week_cos']].head())


# Final Data Preparation & Cleanup
print(f"Shape of train_final_df before dropping NaNs: {train_final_df.shape}")
initial_rows = train_final_df.shape[0]
train_final_df.dropna(inplace=True)
final_rows = train_final_df.shape[0]

print(f"Dropped {initial_rows - final_rows} rows from the training set.")
print(f"Shape of train_final_df after dropping NaNs: {train_final_df.shape}")


print(f"Any NaNs in final training data: {train_final_df.isnull().sum().sum()}")
print(f"Any NaNs in final test data: {test_final_df.isnull().sum().sum()}")

print("\nFinal Model-Ready Shapes:")
print(f"Train: {train_final_df.shape}")
print(f"Test:  {test_final_df.shape}")


# Creating a validation set from the last 90 days of the training data
max_date = train_final_df['date'].max()
split_date = max_date - pd.DateOffset(days=89)

val_df = train_final_df[train_final_df['date'] >= split_date].copy()
train_part_df = train_final_df[train_final_df['date'] < split_date].copy()

print(f"Training data from {train_part_df['date'].min().date()} to {train_part_df['date'].max().date()}")
print(f"Validation data from {val_df['date'].min().date()} to {val_df['date'].max().date()}")
print(f"Train part shape: {train_part_df.shape}")
print(f"Validation part shape: {val_df.shape}")


# Evaluation Metric: SMAPE (Symmetric Mean Absolute Percentage Error)

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100



# Baseline 1: Last Observed Value (90-Day Lag)

# This model predicts that the sales for a given day will be the same as the sales from 90 days ago for that same store-item combination. We already created the `sales_lag_90` feature for this.

y_true_val = val_df['sales']
y_pred_lag90 = val_df['sales_lag_90']

smape_lag90 = smape(y_true_val, y_pred_lag90)

print(f"SMAPE for 90-Day Lag Baseline: {smape_lag90:.4f}")


# Baseline 2: Simple Average per Store-Item
# This model predicts that the sales for a given store-item combination will be its historical average, calculated from the training portion of the data.

# Calculating the average sales for each store-item group using the training part
store_item_avg = train_part_df.groupby(['store', 'item'])['sales'].mean().reset_index()
store_item_avg.rename(columns={'sales': 'avg_sales'}, inplace=True)

# Merging this average onto the validation set
val_merged = pd.merge(val_df, store_item_avg, on=['store', 'item'], how='left')

# Fallbacking for any new store-item pairs (unlikely with this split, but good practice)
global_avg = train_part_df['sales'].mean()
val_merged['avg_sales'].fillna(global_avg, inplace=True)

# Calculating SMAPE
y_true_val = val_merged['sales'] 
y_pred_avg = val_merged['avg_sales']

smape_avg = smape(y_true_val, y_pred_avg)

print(f"SMAPE for Simple Average Baseline: {smape_avg:.4f}")
# our predictions will be wrong by about 19.8% on average.


# Model Training and Evaluation
# We will now train and evaluate several models in a structured loop. This allows us to compare their performance on our validation set and select the best one for the final prediction.

# Additional imports for this section
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Defining Features (X) and Target (y) for validation
features = [col for col in train_final_df.columns if col not in ['date', 'id', 'sales']]
categorical_features = ['store', 'item', 'day_of_week', 'month', 'year', 'is_weekend', 'quarter']

X_train = train_part_df[features]
y_train = train_part_df['sales']
X_val = val_df[features]
y_val = val_df['sales']

# For the final model submission
X_test = test_final_df[features]

print(f"Training with {len(features)} features.")
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

# Defining Models to Train
# We define our models in a dictionary for easy iteration.
models = {
    "LightGBM": lgb.LGBMRegressor(random_state=42, objective='regression_l1', metric='mae', n_jobs=-1, verbose=-1),
    "XGBoost": xgb.XGBRegressor(random_state=42, objective='reg:squarederror', eval_metric='mae', n_jobs=-1),
    "Ridge": Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(random_state=42))
    ])
}

# Dictionary to store results
results = {}

# Training and Evaluation Loop
for name, model in models.items():
    print(f"--- Training {name} ---")
    
    # Training the model
    model.fit(X_train, y_train)
    
    # Making the predictions on the validation set
    y_pred_val = model.predict(X_val)
    
    # Ensuring the predictions are non-negative
    y_pred_val[y_pred_val < 0] = 0
    
    # Calculating SMAPE score
    score = smape(y_val, y_pred_val)
    results[name] = score
    
    print(f"SMAPE for {name}: {score:.4f}\n")

# Comparing Results and Select Best Model
print("--- Model Comparison ---")
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['SMAPE_Score']).sort_values(by='SMAPE_Score')
print(results_df)

best_model_name = results_df.index[0]
best_model = models[best_model_name]
print(f"\nBest performing model: {best_model_name} with SMAPE: {results_df.iloc[0,0]:.4f}")

# Creating the Submission File with the Best Model
# Retraining the best model on the full training data
print(f"\nRetraining best model ({best_model_name}) on full training data...")
X_full_train = train_final_df[features]
y_full_train = train_final_df['sales']

best_model.fit(X_full_train, y_full_train)

print("Predicting on test data...")
predictions = best_model.predict(X_test)

# Ensuring predictions are non-negative
predictions[predictions < 0] = 0

# Creating the submission DataFrame
submission_df = pd.DataFrame({'id': test_final_df['id'], 'sales': predictions})


# Saving to csv
submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())


submission_df


feature_importances = pd.DataFrame({
    'feature': X_full_train.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 12))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('XGBoost Feature Importance')
plt.show()

# Past Sales are the most critical predictors.
# Monthly/Yearly Seasonality (month_cos, month_target_enc) and Item/Store Averages (item_target_enc, store_target_enc) are also highly influential.


# Managerial Report: 30-Day Demand Forecast

# Combining predictions with test set details
report_df = pd.merge(test_df[['id', 'date', 'store', 'item']], submission_df, on='id')

# Defining the 30-day forecast horizon
start_date = report_df['date'].min()
end_date = start_date + pd.DateOffset(days=29)
print(f"Generating report for forecast horizon: {start_date.date()} to {end_date.date()}")

# Filtering for the 30-day horizon
thirty_day_forecast = report_df[(report_df['date'] >= start_date) & (report_df['date'] <= end_date)]

# Aggregating total demand per item
# We sum the predicted sales across all stores for each item
item_demand_30_days = thirty_day_forecast.groupby('item')['sales'].sum().reset_index()
item_demand_30_days.rename(columns={'sales': 'predicted_demand_30_days'}, inplace=True)

# Sorting to find top-demand items
top_items_to_purchase = item_demand_30_days.sort_values(by='predicted_demand_30_days', ascending=False)

# Round the demand for cleaner presentation
top_items_to_purchase['predicted_demand_30_days'] = top_items_to_purchase['predicted_demand_30_days'].round(0).astype(int)

# Displaying the report
print("\nThis table shows the total predicted sales demand for each item across all stores. For the top items.")
print(f"\n{top_items_to_purchase.head(15).to_string(index=False)}")




