import pandas as pd
import numpy as np


# --- Step 1: Read in the Data Files ---
# Update file paths as needed for your local environment.
sales_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
calendar_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
sell_prices_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')


# --- Step 2: Convert Sales Data from Wide to Long Format ---
# The sales data is stored with one row per item and columns 'd_1' to 'd_N' representing dates.
id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
value_vars = [col for col in sales_df.columns if col.startswith('d_')]
sales_long = pd.melt(sales_df, id_vars=id_vars, value_vars=value_vars, 
                     var_name='d', value_name='sales')


# --- Step 3: Merge with Calendar Data ---
# The calendar file has a column 'd' that aligns with our melted sales data.
sales_long = sales_long.merge(calendar_df, on='d', how='left')


# --- Step 4: Merge with Sell Prices ---
# Merge pricing information based on store, item, and the week (wm_yr_wk).
sales_long = sales_long.merge(sell_prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')


# --- Step 5: Preprocess Dates and Sort Data ---
# Convert the 'date' column to datetime and sort by 'id' then date.
sales_long['date'] = pd.to_datetime(sales_long['date'])
sales_long.sort_values(by=['id', 'date'], inplace=True)


# --- Step 6: Create Lag Features ---
# These features capture previous days' sales, e.g., 7-day and 28-day lags.
lag_days = [7, 28]
for lag in lag_days:
    sales_long[f'sales_lag_{lag}'] = sales_long.groupby('id')['sales'].shift(lag)


# --- Step 7: Create Rolling Statistics ---
# Calculate rolling mean and standard deviation for given window sizes.
rolling_windows = [7, 28]
for window in rolling_windows:
    # Use shift(1) to avoid using current day's sales in the rolling calculation.
    sales_long[f'rolling_mean_{window}'] = sales_long.groupby('id')['sales'].transform(
        lambda x: x.shift(1).rolling(window).mean())
    sales_long[f'rolling_std_{window}'] = sales_long.groupby('id')['sales'].transform(
        lambda x: x.shift(1).rolling(window).std())


# --- Step 8: Generate Calendar-Based Features ---
# Extract date parts that can capture seasonality: day of week, month, year, and weekend indicator.
sales_long['dayofweek'] = sales_long['date'].dt.dayofweek
sales_long['month'] = sales_long['date'].dt.month
sales_long['year'] = sales_long['date'].dt.year
sales_long['is_weekend'] = sales_long['dayofweek'].isin([5, 6]).astype(int)


# Optionally, flag special events using the calendar data (e.g., holidays or events)
# For example, you could create a binary feature from an event column:
if 'event_name_1' in sales_long.columns:
    sales_long['is_event'] = sales_long['event_name_1'].notnull().astype(int)


# --- Step 9: Create Price-Based and Interaction Features ---
# Example: Calculate the price change rate over time for each item.
sales_long['price_change_rate'] = sales_long.groupby('id')['sell_price'].pct_change()


# You could also create an interaction feature between promotional events and price
sales_long['price_event_interaction'] = sales_long['sell_price'] * sales_long.get('is_event', 0)


# Preview the enriched dataset
print(sales_long.head())


sales_long.columns


# !pip install lightgbm


import lightgbm as lgb
print(lgb.__version__)


# Remove initial rows with NaN due to lag/rolling calculations
df_model = sales_long.dropna(subset=['sales_lag_28'])

# For this example, we choose a date cutoff.
# Adjust the date thresholds based on your available data.
train_df = df_model[df_model['date'] < '2016-04-25']
valid_df = df_model[(df_model['date'] >= '2016-04-25') & (df_model['date'] <= '2016-05-22')]

# Define feature columns (you can expand this list as you refine your features)
features = [col for col in df_model.columns if col.startswith('sales_lag') or 
            col.startswith('rolling_') or col in ['dayofweek', 'month', 'is_weekend', 'price_change_rate']]
target = 'sales'

# Split into X and y
X_train = train_df[features]
y_train = train_df[target]
X_valid = valid_df[features]
y_valid = valid_df[target]


# Debug prints: Verify that the data is non-empty and correctly shaped.
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_valid shape:", X_valid.shape)
print("y_valid shape:", y_valid.shape)


# Debug: Print available date range in df_model
print("Min date in df_model:", df_model['date'].min())
print("Max date in df_model:", df_model['date'].max())

# For example, you might set the validation period as the last 28 days of available data.
max_date = df_model['date'].max()
train_df = df_model[df_model['date'] < (max_date - pd.Timedelta(days=28))]
valid_df = df_model[df_model['date'] >= (max_date - pd.Timedelta(days=28))]

# Confirm that both training and validation sets are non-empty.
print("Train set date range:", train_df['date'].min(), "-", train_df['date'].max())
print("Validation set date range:", valid_df['date'].min(), "-", valid_df['date'].max())

print("X_train shape:", train_df[features].shape)
print("y_train shape:", train_df[target].shape)
print("X_valid shape:", valid_df[features].shape)
print("y_valid shape:", valid_df[target].shape)


train_df[features].head()


# Create LightGBM Datasets from your training and validation data.
lgb_train = lgb.Dataset(train_df[features], label=train_df[target])
lgb_valid = lgb.Dataset(valid_df[features], label=valid_df[target])

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1  # to suppress internal logging
}

# Train the LightGBM model using callbacks for early stopping and logging.
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_valid],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

print("Best iteration:", model.best_iteration)


# Save the model to a file.
model.save_model('m5_trained_model.txt')


# Extract feature importances.
importances = model.feature_importance()
feature_names = features  # The feature list used during training.

# Combine and display in a DataFrame
feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(feat_imp)











# Choose a sample time series 'id'
sample_id = df_model['id'].unique()[0]
sample_series = df_model[df_model['id'] == sample_id].copy().sort_values('date')
last_known_date = sample_series['date'].max()
forecast_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=28)

# Create a temporary DataFrame for recursive forecasting.
temp_series = sample_series.copy()
forecast_preds = []

for forecast_date in forecast_dates:
    new_features = {}
    new_features['date'] = forecast_date

    # Generate lag features for forecast_date.
    for lag in [7, 28]:
        lag_date = forecast_date - pd.Timedelta(days=lag)
        lag_value = temp_series.loc[temp_series['date'] == lag_date, 'sales']
        new_features[f'sales_lag_{lag}'] = lag_value.values[0] if not lag_value.empty else np.nan

    # Compute rolling statistics.
    for window in [7, 28]:
        window_start = forecast_date - pd.Timedelta(days=window)
        window_end = forecast_date - pd.Timedelta(days=1)
        window_sales = temp_series[(temp_series['date'] >= window_start) & (temp_series['date'] <= window_end)]['sales']
        new_features[f'rolling_mean_{window}'] = window_sales.mean() if len(window_sales) > 0 else np.nan
        new_features[f'rolling_std_{window}'] = window_sales.std() if len(window_sales) > 0 else np.nan

    # Calendar features.
    new_features['dayofweek'] = forecast_date.dayofweek
    new_features['month'] = forecast_date.month
    new_features['is_weekend'] = int(forecast_date.dayofweek in [5, 6])
    
    # Set default for price_change_rate (customize as needed).
    new_features['price_change_rate'] = 0

    # Create a DataFrame row for prediction.
    new_row_df = pd.DataFrame([new_features])
    X_new = new_row_df[features]
    pred_sales = model.predict(X_new)[0]
    forecast_preds.append(pred_sales)
    
    # Append the new forecast to update features recursively.
    new_features['sales'] = pred_sales
    new_features['id'] = sample_id
    temp_series = pd.concat([temp_series, pd.DataFrame([new_features])], ignore_index=True)

# Compile forecasts for display.
forecast_df = pd.DataFrame({
    'date': forecast_dates,
    'predicted_sales': forecast_preds
})
print("Forecasted Sales for id:", sample_id)
print(forecast_df)


# Define parameters for the Tweedie model.
params_tweedie = {
    'objective': 'tweedie',
    'tweedie_variance_power': 1.1,  # Adjust between 1.0 (Poisson) and 2.0 (Gamma) as needed.
    'metric': 'rmse',               # We use RMSE here for monitoring (LightGBM does not natively evaluate WRMSSE).
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}

# Create LightGBM Datasets from the training and validation sets.
lgb_train = lgb.Dataset(train_df[features], label=train_df[target])
lgb_valid = lgb.Dataset(valid_df[features], label=valid_df[target])

# Train the model using callbacks for early stopping and logging.
model_tweedie = lgb.train(
    params_tweedie,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_valid],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

print("Best iteration for Tweedie model:", model_tweedie.best_iteration)


# Get predictions from the Tweedie model on the validation set.
val_preds_tweedie = model_tweedie.predict(valid_df[features])

# Attach the predictions to the validation DataFrame.
valid_df_with_preds = valid_df.copy()
valid_df_with_preds['predicted'] = val_preds_tweedie


def simple_wrmsse(train_df, valid_df):
    """
    Computes a simplified WRMSSE score.
    
    Parameters:
        train_df : DataFrame
            Training data including 'id', 'date', and 'sales'.
        valid_df : DataFrame
            Validation data including 'id', 'date', 'sales', and 'predicted'.
    
    Returns:
        overall_wrmsse : float
            The weighted RMSSE computed across series.
    """
    series_ids = valid_df['id'].unique()
    weighted_errors = []
    weights = []
    
    for sid in series_ids:
        # Get training sales history for this series.
        train_sales = train_df[train_df['id'] == sid].sort_values('date')['sales'].values
        # Get actual validation sales.
        valid_sales = valid_df[valid_df['id'] == sid].sort_values('date')['sales'].values
        # Get predictions for this series.
        pred_sales = valid_df[valid_df['id'] == sid].sort_values('date')['predicted'].values
        
        # Compute the scaling factor: RMS of the differences in training sales.
        if len(train_sales) > 1:
            scale = np.sqrt(np.mean(np.diff(train_sales) ** 2))
        else:
            scale = 1.0
        
        # Compute RMSE for the validation period for this series.
        rmse = np.sqrt(np.mean((valid_sales - pred_sales) ** 2))
        
        # Calculate the series-specific RMSSE.
        series_error = rmse / scale if scale > 0 else rmse
        
        # Weight for the series: Here we use the sum of training sales.
        series_weight = train_sales.sum()
        
        weighted_errors.append(series_error ** 2 * series_weight)
        weights.append(series_weight)
    
    overall_wrmsse = np.sqrt(np.sum(weighted_errors) / np.sum(weights))
    return overall_wrmsse

# Compute the simplified WRMSSE score.
wrmsse_score = simple_wrmsse(train_df, valid_df_with_preds)
print("Simplified WRMSSE score for Tweedie model:", wrmsse_score)




