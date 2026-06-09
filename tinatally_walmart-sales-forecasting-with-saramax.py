# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
%matplotlib inline 
plt.style.use('seaborn-v0_8-deep')
import seaborn as sns
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from scipy.signal import find_peaks 
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller

from typing import Union, List
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
stores_df = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')
features_df = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip')



def file_info(df):
    print('HEAD')
    display(df.head())
    print()
    print('INFO')
    display(df.info())
    print()
    print('Summary Statistic')
    display(df.describe().T)
    print()
    print('CHECK FOR NULL VALUES')
    null_values = df.isnull().sum()/df.shape[0]
    display(null_values[null_values >0])
    print()
    print('Duplicated values')
    display(df.duplicated().sum())


file_info(train_df)


file_info(stores_df)


file_info(features_df)


# convert date object to datetime format
train_df['Date'] = pd.to_datetime(train_df['Date'])
features_df['Date'] = pd.to_datetime(features_df['Date'])


# fix missing values using simpleimputer 
missing_col = ['MarkDown1',
               'MarkDown2',
               'MarkDown3',
               'MarkDown4',
               'MarkDown5',
               'CPI',
               'Unemployment']
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
features_df[missing_col] = imputer.fit_transform(features_df[missing_col])


# drop isholiday in the features file 
features_df.drop(columns=['IsHoliday'], inplace=True)


# merge files on store and date
# Merge stores_df and features_df on Store and Date
stores_and_features = stores_df.merge(features_df, on=['Store'], how='outer')

# Merge result with train_df
full_df = stores_and_features.merge(train_df, on=['Store','Date'], how='right')


file_info(full_df)


# sns.pairplot(full_df,  hue='IsHoliday');


numeric_col = full_df.select_dtypes(include=['number']).columns

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16,12))
axes = axes.flatten()

for i, col in enumerate(numeric_col):
    sns.histplot(full_df[col], kde=True, ax=axes[i], bins=40)
    axes[i].set_title(col)

plt.tight_layout()
plt.show()


# check for outliers
cols = ['MarkDown1',
        'MarkDown2',
        'MarkDown3',
        'MarkDown4',
        'MarkDown5',
        'Weekly_Sales'
       ]

for col in cols:
    Q1 = full_df[col].quantile(0.25)
    Q3 = full_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    plt.figure(figsize=(6,4))
    sns.boxplot(x=full_df[col])
    plt.title(f"{col} Boxplot with IQR")
    plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
    plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Bound')
    plt.legend()
    plt.show()

    outliers = full_df[(full_df[col] < lower_bound) | (full_df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers detected")


#log scaled values 
for col in cols:
    log_col = np.log1p(full_df[col])  # log-transform the column
    Q1 = log_col.quantile(0.25)
    Q3 = log_col.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Plot boxplot with log scale
    plt.figure(figsize=(6,4))
    sns.boxplot(x=log_col)
    plt.title(f"{col} Boxplot with IQR (log scale)")
    plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
    plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Bound')
    plt.legend()
    plt.show()

    outliers = full_df[(log_col < lower_bound) | (log_col > upper_bound)]
    print(f"{col}: {len(outliers)} outliers detected")


plt.figure(figsize =(12,12))
sns.heatmap(round(full_df.select_dtypes(include=np.number).corr(),2),annot=True);


full_df['year'] = full_df['Date'].dt.year
full_df['month'] = full_df['Date'].dt.month
full_df['day'] = full_df['Date'].dt.day
full_df['is_weekend'] = full_df['Date'].dt.dayofweek >4


plt.figure(figsize=(12,6))
plt.plot(full_df['Weekly_Sales'], label='Original', alpha=0.5)
plt.plot(full_df['Weekly_Sales'].rolling(7).mean(), label='7-week Rolling Mean')
plt.plot(full_df['Weekly_Sales'].rolling(13).mean(), label='13-week Rolling Mean')
plt.title("Weekly Sales Rolling Means")
plt.legend()
plt.show()


#Check for autocorrelation and partial autocorrelation 
plot_acf(full_df['Weekly_Sales'])
plot_pacf(full_df['Weekly_Sales'])
plt.show()


# Perform stationarity test: Use ADF test before and after differencing
def adfuller_stats(values):
    results = adfuller(values)
    results_output = pd.Series(results[0:4],
                          index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in results[4].items():
      results_output['Critical Value (%s) '%key] = value
    return results_output
adfuller_stats(full_df['Weekly_Sales'].values)


sns.countplot(x='Type', data= full_df)
plt.title('Distribution of Store Types');


print(full_df['Store'].nunique(), "stores")
print(full_df['Dept'].nunique(),'Deptments')
print(full_df['IsHoliday'].value_counts(normalize=True))


# Aggregate weekly sales by date
sales_over_time = full_df.groupby('Date')['Weekly_Sales'].sum()

plt.figure(figsize=(15,5))
plt.plot(sales_over_time)
plt.title("Total Weekly Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.show()


# Holiday vs Non-Holiday
plt.figure(figsize=(8,5))
sns.boxplot(x='IsHoliday', y='Weekly_Sales', data=full_df)
plt.title("Weekly Sales: Holiday vs Non-Holiday")
plt.show()


# Store/Dept Level Analysis
# Average sales by store
store_sales = full_df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)
store_sales.plot(kind='bar', figsize=(15,5), title="Average Weekly Sales per Store");




# Average sales by department
dept_sales = full_df.groupby('Dept')['Weekly_Sales'].mean().sort_values(ascending=False)
dept_sales.plot(kind='bar', figsize=(15,5), title="Average Weekly Sales per Department");


# Promotions & Markdown Effects
markdown_cols = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
plt.figure(figsize=(15,5))
sns.heatmap(full_df[markdown_cols + ['Weekly_Sales']].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation: Markdown vs Weekly Sales")
plt.show()


# Economic Indicators
econ_cols = ['CPI','Unemployment']
for col in econ_cols:
    plt.figure(figsize=(10,5))
    sns.scatterplot(x=full_df[col], y=full_df['Weekly_Sales'])
    plt.title(f"Weekly Sales vs {col}")
    plt.show()


sales_over_time = full_df.groupby('Date')['Weekly_Sales'].sum()
mean_sales = sales_over_time.mean()

holiday_dates = full_df[full_df['IsHoliday'] == True]['Date'].unique()

plt.figure(figsize=(15,5))
plt.plot(sales_over_time, label='Weekly Sales')
plt.axhline(mean_sales, color='blue', linestyle='--', label=f'Mean Sales: {mean_sales:,.0f}')

# Shade holiday weeks
for date in holiday_dates:
    plt.axvspan(date, date + pd.Timedelta(days=6), color='green', alpha=0.3)

plt.title("Total Weekly Sales Over Time (Holidays Highlighted)")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend()
plt.show()


sales_by_day = full_df.groupby('day')['Weekly_Sales'].mean()

#map numbers to day names
day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
sales_by_day.index = sales_by_day.index.map(day_names)
sales_by_day = sales_by_day.sort_values(ascending=False)
# Plot
plt.figure(figsize=(8,5))
sns.lineplot(x=sales_by_day.index, y=sales_by_day.values )
plt.title("Average Weekly Sales by Day of the Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Weekly Sales")
plt.show()


# Weekday/Weekend Effects
plt.figure(figsize=(8,5))
sns.boxplot(x='is_weekend', y='Weekly_Sales', data=full_df)
plt.title("Weekly Sales: Weekday vs Weekend")
plt.show()


def compute_weighted_MAE(y_true: pd.Series, y_pred: Union[pd.Series, np.ndarray], df_source: pd.DataFrame) -> float:
    """
    Computes the Weighted Mean Absolute Error (WMAE).
    
    This function is unified to work with both SARIMAX and XGBoost outputs.
    It determines the holiday weights based on the dates present in the y_true index
    and the source DataFrame (df_source).

    Args:
        y_true (pd.Series): The true sales values (must have a datetime index).
        y_pred (Union[pd.Series, np.ndarray]): The predicted sales values.
        df_source (pd.DataFrame): The original, unaggregated source DataFrame
                                  containing the 'Date' and 'IsHoliday' columns.

    Returns:
        float: The Weighted Mean Absolute Error (WMAE).
    """
    y_true_values = y_true.values
    y_pred_values = y_pred.values if hasattr(y_pred, 'values') else np.array(y_pred)

    if len(y_true_values) != len(y_pred_values):
        raise ValueError("y_true and y_pred must have the same length.")

    # Get the dates for the test set from the y_true index
    test_dates = y_true.index

    # Aggregate holiday status from the full source data
    # True if ANY store had a holiday on that Date
    holiday_weekly = df_source.groupby('Date')['IsHoliday'].max() 

    # Align holiday status to the test set dates
    holiday_status = holiday_weekly.reindex(test_dates).fillna(False)
    
    # Assign weights (5 for Holiday, 1 for non-Holiday)
    weekly_weights = holiday_status.map({True: 5, False: 1}).values
    
    # Compute weighted absolute errors
    weighted_errors = weekly_weights * np.abs(y_true_values - y_pred_values)
    wmae = weighted_errors.sum() / weekly_weights.sum()
    
    print(f"Weighted MAE (WMAE): {wmae:,.2f}")
    return wmae


# Ensure Date is datetime and data is sorted
sales = full_df.groupby('Date')['Weekly_Sales'].sum().sort_index()

# Split data into train test split 
test_weeks = 26
train = sales.iloc[:-test_weeks]
test = sales.iloc[-test_weeks:]

print(f"Train length: {len(train)}, Test length: {len(test)}")
# non-seasonal (p,d,q) = (1,1,1); seasonal (P,D,Q,s) = (1,1,1,52)
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 52)

def train_sarima_model(train, exog=None):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,exog=exog,
                enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    
    print(results.summary())
    return results
results = train_sarima_model(train, exog=None)


sarima_forecast = results.get_forecast(steps=test_weeks)
sarima_forecast_mean = sarima_forecast.predicted_mean
sarima_forecast_ci = sarima_forecast.conf_int()
wmae = compute_weighted_MAE(
    y_true=test, 
    y_pred=sarima_forecast_mean, 
    df_source=full_df
)


def plot_results_with_ci(train, test, title, mean, ci):
    
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train, label="Train", alpha=0.6)
    plt.plot(test.index, test, label="Actual", color="red")
    plt.plot(mean.index, mean, label="Forecast", color="green")
    
    plt.fill_between(ci.index,
                     ci.iloc[:, 0],
                     ci.iloc[:, 1], color="green", alpha=0.2)
    
    plt.axvline(test.index[0], color='black', linestyle='--', alpha=0.5)
    plt.title(title)
    plt.ylabel("Weekly Sales")
    plt.legend()
    plt.show()

plot_results_with_ci(train, test, 
                     "SARIMA Forecast vs Actual Weekly Sales",
                     sarima_forecast_mean,sarima_forecast_ci )


order = (0, 1, 1)
seasonal_order = (0, 1, 0, 52)
exog_vars = ['Unemployment', 'IsHoliday']
# Use mean for unemployment (or max, as the rate should be the same)
exog_unemp = full_df.groupby('Date')['Unemployment'].mean()
# Use max for IsHoliday (True if any store is a holiday)
exog_holiday = full_df.groupby('Date')['IsHoliday'].max().astype(int)

# Combine into the correct exogenous dataframe
exog = pd.DataFrame({'Unemployment': exog_unemp, 'IsHoliday': exog_holiday})

# Train/test split 
test_weeks = 26

# Then proceed with your train/test split and modeling
exog_train = exog.iloc[:-test_weeks]
exog_test = exog.iloc[-test_weeks:]

results_exog = train_sarima_model(train, exog=exog_train)


# Forecast with exogenous variables
sarimax_forecast = results_exog.get_forecast(steps=test_weeks, exog=exog_test)
sarimax_forecast_mean = sarimax_forecast.predicted_mean
sarimax_forecast_ci = sarimax_forecast.conf_int()
wmae_exog = compute_weighted_MAE(
    y_true=test, 
    y_pred=sarimax_forecast_mean, 
    df_source=full_df
)


plot_results_with_ci(train, test, 
                     "SARIMAX-X Forecast vs Actual Weekly Sales",
                     sarimax_forecast_mean, sarimax_forecast_ci)


#Data Preparation and Feature Engineering
def prepare_data_for_xgboost(df):
    """Aggregates data to weekly level and creates time-based features."""
    
    # Create Time-Based Features (from Date)
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Year'] = df['Date'].dt.isocalendar().year.astype(int)
    df['Month'] = df['Date'].dt.month
    
    # Define aggregation rules
    agg_rules = {
        'Weekly_Sales': 'sum',
        'Temperature': 'mean',
        'Fuel_Price': 'mean',
        'MarkDown1': 'sum',
        'MarkDown2': 'sum',
        'MarkDown3': 'sum',
        'MarkDown4': 'sum',
        'MarkDown5': 'sum',
        'CPI': 'mean',
        'Unemployment': 'mean',
        'IsHoliday': 'max',  # True if any store had a holiday
        # Use a non-null column to keep the date
        'Week': 'first',
        'Year': 'first',
        'Month': 'first'
    }
    
    # Aggregate to Weekly Level 
    weekly_data = df.groupby('Date').agg(agg_rules).reset_index()
    
    # Target variable (Y)
    y = weekly_data['Weekly_Sales']
    dates = weekly_data['Date']
    # Feature set (X)
    X = weekly_data.drop(columns=['Date', 'Weekly_Sales', 'Size', 'Week', 'Year', 'Month'], errors='ignore')
    
    return X, y, dates

X, y, dates = prepare_data_for_xgboost(full_df)

# Add back time features if desired for XGBoost to capture seasonality
X['Week'] = dates.dt.isocalendar().week.astype(int).values
X['Year'] = dates.dt.isocalendar().year.astype(int).values

## Train/Test Split
test_weeks = 26
train_end_index = len(y) - test_weeks

X_train = X.iloc[:train_end_index]
X_test = X.iloc[train_end_index:]
y_train = y.iloc[:train_end_index]
y_test = y.iloc[train_end_index:]
dates_test = y_test.index 

print(f"Train length: {len(X_train)}, Test length: {len(X_test)}")

## XGBoost Model Training
xgb_model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    n_jobs=-1
)

# Train the model
xgb_model.fit(
    X_train, y_train,
    # Use early stopping to prevent overfitting
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,
    verbose=False
)

## Forecasting and Evaluation
train_pred = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)


## Weighted Evaluation Metric
def compute_weighted_MAE_xgb(test_dates, y_true, y_pred, df_source):
    """
    Computes the Weighted Mean Absolute Error (WMAE).
    """
    holiday_weekly = df_source.groupby('Date')['IsHoliday'].max() 
    holiday_status = holiday_weekly.reindex(test_dates).fillna(False)
    weekly_weights = holiday_status.map({True: 5, False: 1}).values
    weighted_errors = weekly_weights * np.abs(y_true.values - y_pred)
    wmae = weighted_errors.sum() / weekly_weights.sum()
    print(f"\nWeighted MAE (WMAE): {wmae:,.2f}")
    return wmae

# Calculate WMAE
wmae_xgb = compute_weighted_MAE_xgb(
    test_dates=dates_test,  
    y_true=y_test,  
    y_pred=y_pred_test,  
    df_source=full_df 
)

# Calculate residual standard deviation for CI
residual_std = np.std(y_train - train_pred)

# Get feature importance scores
importance = xgb_model.get_booster().get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'Feature': list(importance.keys()),
    'Importance': list(importance.values())
}).sort_values(by='Importance', ascending=False)

# Plotting Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('XGBoost Feature Importance (Split Weight)')
plt.show() 


def plot_xgboost_forecast_with_ci(train: pd.Series, test: pd.Series, 
                                  forecast: Union[np.ndarray, pd.Series],
                                  title: str, residual_std: float=None, ci_multiplier: float=1.96):
    """
    Plots training data, actual test data, forecast, and optional confidence interval (CI).
    
    The function uses the DateTimeIndex of the data for accurate X-axis plotting
    and formats the dates for readability.

    Args:
        train (pd.Series): Historical training sales data (must have DateTimeIndex).
        test (pd.Series): Actual sales data for the test period (must have DateTimeIndex).
        forecast (np.array or pd.Series): Forecasted values (y_pred).
        title (str): Plot title.
        residual_std (float, optional): Standard deviation of training residuals 
                                        to simulate CI (e.g., from XGBoost).
        ci_multiplier (float, optional): Multiplier for CI (default 1.96 ~ 95% CI).
    """
    # Align forecast to test index 
    if isinstance(forecast, np.ndarray):
        if len(forecast) != len(test):
            print("Warning: Forecast length does not match test length. Check data split.")
        forecast = pd.Series(forecast, index=test.index)

    plt.figure(figsize=(16, 8)) 
    
    # Plot training data
    plt.plot(train.index, train, label="Training Data", alpha=0.6, color='#1f77b4')
    
    # Plot actual test data
    plt.plot(test.index, test, label="Actual Sales (Test Period)", color="red", linewidth=2)
    
    # Plot forecast
    plt.plot(forecast.index, forecast, label="Forecast", color="green", linewidth=2)
    
    # Plot confidence interval if residual_std is provided
    if residual_std is not None:
        ci_lower = forecast - ci_multiplier * residual_std
        ci_upper = forecast + ci_multiplier * residual_std
        
        # Label the CI based on the multiplier (e.g., 95% if multiplier is 1.96)
        ci_level = int(round(ci_multiplier / 1.96 * 95))
        
        plt.fill_between(forecast.index, ci_lower, ci_upper, 
                         color="green", alpha=0.2, label=f'{ci_level}% Confidence Interval (Simulated)')
    
    # Vertical line for forecast start
    if not test.empty:
        plt.axvline(test.index[0], color='black', linestyle=':', alpha=0.7, label='Forecast Start')
    
    # Date Formatting and Tick Control for X-Axis 
    ax = plt.gca()
    
    # Combine the index of both train and test data for full X-axis range
    full_index = train.index.append(test.index)
    
    # Sample the index every 26 weeks (half a year) for tick positions
    tick_indices = np.arange(0, len(full_index), 26) 
    
    # Ensure the last date is always included
    if len(full_index) > 0 and (len(full_index) - 1) not in tick_indices:
        tick_indices = np.append(tick_indices, len(full_index) - 1)
        
    actual_ticks = full_index[tick_indices]

    # Set the ticks to fall exactly on the sampled data points
    ax.set_xticks(actual_ticks)
    
    # Format the dates (Year and Month)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right') 

    plt.title(title, fontsize=18, fontweight='bold')
    plt.ylabel("Weekly Sales", fontsize=14)
    plt.xlabel("Date", fontsize=14)
    plt.ticklabel_format(style='plain', axis='y') 
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

plot_xgboost_forecast_with_ci(
    train=y_train,
    test=y_test,
    forecast=y_pred_test,
    title=f"XGBoost Forecast with 95% CI (WMAE: {wmae_xgb:,.0f})", 
    residual_std=None,  
    ci_multiplier=1.96
)




#  Data Preparation and Feature Engineering (with Lag 52) 
def prepare_data_for_xgboost(df):
    """Aggregates data to weekly level and creates time-based features."""
    
    # Create Time-Based Features (from Date)
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Year'] = df['Date'].dt.isocalendar().year.astype(int)
    df['Month'] = df['Date'].dt.month
    
    # Define aggregation rules
    agg_rules = {
        'Weekly_Sales': 'sum',
        'Temperature': 'mean',
        'Fuel_Price': 'mean',
        'MarkDown1': 'sum',
        'MarkDown2': 'sum',
        'MarkDown3': 'sum',
        'MarkDown4': 'sum',
        'MarkDown5': 'sum',
        'CPI': 'mean',
        'Unemployment': 'mean',
        'IsHoliday': 'max',  
    }
    
    # Aggregate to Weekly Level
    weekly_data = df.groupby('Date').agg(agg_rules)
    weekly_data = weekly_data.reset_index()

    # Time Series Features 
    weekly_data['Sales_Lag_1'] = weekly_data['Weekly_Sales'].shift(1)
    weekly_data['Sales_Lag_52'] = weekly_data['Weekly_Sales'].shift(52) 
    
    weekly_data['Rolling_Mean_4'] = weekly_data['Weekly_Sales'].shift(1).rolling(window=4).mean()
    weekly_data['Rolling_Std_12'] = weekly_data['Weekly_Sales'].shift(1).rolling(window=12).std()
    
    # Drop NaNs created by lag/rolling features (first 52 weeks are dropped)
    weekly_data = weekly_data.dropna(subset=['Sales_Lag_52'])
    
    # Set Date as the index for plotting and time-series feature access
    dates = weekly_data['Date']
    weekly_data = weekly_data.set_index('Date')
    
    # Target variable (Y) is a Series with the Date index
    y = weekly_data['Weekly_Sales']
    
    # Feature set (X)
    X = weekly_data.drop(columns=['Weekly_Sales'], errors='ignore')
    
    return X, y, dates

# Execute data preparation
X, y, dates = prepare_data_for_xgboost(full_df)

# Add back time features as separate columns for XGBoost
X['Week'] = X.index.isocalendar().week.astype(int).values
X['Year'] = X.index.isocalendar().year.astype(int).values

# Train/Test Split 
test_weeks = 26
train_end_index = len(y) - test_weeks

X_train = X.iloc[:train_end_index]
X_test = X.iloc[train_end_index:]
y_train = y.iloc[:train_end_index]
y_test = y.iloc[train_end_index:]
dates_test = y_test.index 

print(f"Train length (after lag drop): {len(X_train)}, Test length: {len(X_test)}")


#  XGBoost Model Training and Forecasting 
xgb_model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1500,        
    learning_rate=0.03,       
    max_depth=7,              
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0.1,                
    min_child_weight=1,       
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=75, 
    verbose=False
)

# Make predictions
train_pred = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# Calculate WMAE
wmae_xgb = compute_weighted_MAE_xgb(
    test_dates=dates_test,  
    y_true=y_test,  
    y_pred=y_pred_test,  
    df_source=full_df 
)

# Calculate residual standard deviation for CI
residual_std = np.std(y_train - train_pred)
print(f"Calculated Training Residual Standard Deviation for CI: {residual_std:,.2f}")

plot_xgboost_forecast_with_ci(
    train=y_train,
    test=y_test,
    forecast=y_pred_test,
    # Updated title to reflect WMAE score
    title=f"XGBoost Forecast with 95% CI (WMAE: {wmae_xgb:,.0f})", 
    residual_std=residual_std,  
    ci_multiplier=1.96
)

#  Feature Importance Plot
importance = xgb_model.get_booster().get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'Feature': list(importance.keys()),
    'Importance': list(importance.values())
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), color='darkred')
plt.title('XGBoost Feature Importance (Split Weight) - Improved')
plt.tight_layout()
plt.show()


def prepare_data_for_sarimax(df, exog_vars):
    """
    Aggregates data to weekly level and prepares target (y) and exogenous (exog) variables.
    SARIMAX does not need the complex lag features created for XGBoost.
    """
    # Aggregation rules for all features used in XGBoost and SARIMAX
    agg_rules = {
        'Weekly_Sales': 'sum',
        'Temperature': 'mean',
        'Fuel_Price': 'mean',
        'MarkDown1': 'sum',
        'MarkDown2': 'sum',
        'MarkDown3': 'sum',
        'MarkDown4': 'sum',
        'MarkDown5': 'sum',
        'CPI': 'mean',
        'Unemployment': 'mean',
        'IsHoliday': 'max',  
    }
    
    # Aggregate to Weekly Level
    weekly_data = df.groupby('Date').agg(agg_rules)
    weekly_data = weekly_data.reset_index()

    # Set Date as the index
    weekly_data['Date'] = pd.to_datetime(weekly_data['Date'])
    weekly_data = weekly_data.set_index('Date')
    
    # Ensure 'IsHoliday' is binary (0 or 1) for the model
    weekly_data['IsHoliday'] = weekly_data['IsHoliday'].astype(int)

    # Target variable (Y) is a Series
    y = weekly_data['Weekly_Sales']
    
    # Exogenous features (X) selected based on XGBoost importance
    exog = weekly_data[exog_vars].copy()
    
    return y, exog

def plot_sarimax_forecast(train, test, forecast_mean, forecast_ci, title, wmae):
    """Plots training data, test data, and SARIMAX forecast with CI."""
    
    plt.figure(figsize=(16, 8))
    
    # Plot training data
    plt.plot(train.index, train, label="Training Data", alpha=0.6, color='#1f77b4')
    
    # Plot actual test data
    plt.plot(test.index, test, label="Actual Sales (Test Period)", color="red", linewidth=2)
    
    # Plot forecast
    plt.plot(forecast_mean.index, forecast_mean, label=f"SARIMAX Forecast (WMAE: {wmae:,.0f})", color="orange", linewidth=2)
    
    # Plot confidence interval
    plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], 
                     color="orange", alpha=0.2, label='95% Confidence Interval')
    
    # Vertical line for forecast start
    if not test.empty:
        plt.axvline(test.index[0], color='black', linestyle=':', alpha=0.7, label='Forecast Start')
    
    # Date Formatting and Tick Control
    ax = plt.gca()
    full_index = train.index.append(test.index)
    tick_indices = np.arange(0, len(full_index), 26) 
    if len(full_index) - 1 not in tick_indices:
        tick_indices = np.append(tick_indices, len(full_index) - 1)
        
    actual_ticks = full_index[tick_indices]
    ax.set_xticks(actual_ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right') 

    plt.title(title, fontsize=18, fontweight='bold')
    plt.ylabel("Weekly Sales", fontsize=14)
    plt.xlabel("Date", fontsize=14)
    plt.ticklabel_format(style='plain', axis='y')
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def train_and_forecast_sarimax(y, exog, df_source, test_weeks=26, order=(0, 1, 1), seasonal_order=(0, 1, 0, 52)):
    """Trains and forecasts using SARIMAX with XGBoost-derived exogenous variables."""
    
    train_end_index = len(y) - test_weeks
    
    y_train = y.iloc[:train_end_index]
    y_test = y.iloc[train_end_index:]
    
    exog_train = exog.iloc[:train_end_index]
    exog_test = exog.iloc[train_end_index:]
    
    print(f"\n--- SARIMAX Model Setup ---")
    print(f"Train length: {len(y_train)}, Test length: {len(y_test)}")
    print(f"SARIMAX Order: {order}, Seasonal Order: {seasonal_order}")
    print(f"Exogenous Variables Used: {list(exog.columns)}")

    # Train the SARIMAX model
    sarimax_model = SARIMAX(
        y_train, 
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    results = sarimax_model.fit(disp=False)
    
    # Make forecast
    forecast_obj = results.get_forecast(steps=test_weeks, exog=exog_test)
    forecast_mean = forecast_obj.predicted_mean
    forecast_ci = forecast_obj.conf_int()
    
    # Evaluate
    wmae = compute_weighted_MAE(y_test, forecast_mean, df_source=df_source)
    
    # Plot
    plot_sarimax_forecast(
        train=y_train,
        test=y_test,
        forecast_mean=forecast_mean,
        forecast_ci=forecast_ci,
        title=f"SARIMAX Forecast (XGBoost Features) - WMAE: {wmae:,.0f}",
        wmae=wmae
    )
    
    return wmae

# Exogenous Variables based on XGBoost Importance
XGBOOST_TOP_EXOG_VARS = ['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'CPI', 'IsHoliday']

# Prepare data
y_sarimax, exog_sarimax = prepare_data_for_sarimax(full_df, exog_vars=XGBOOST_TOP_EXOG_VARS)

# Run the improved SARIMAX model
final_wmae = train_and_forecast_sarimax(
    y=y_sarimax,
    exog=exog_sarimax,
    df_source=full_df,
    test_weeks=26,
    order=(0, 1, 1),
    seasonal_order=(0, 1, 0, 52)
)


def prepare_data_for_sarimax(df):
    """Aggregates data to weekly level and prepares target (y) and exogenous (all possible) variables."""
    
    agg_rules = {
        'Weekly_Sales': 'sum',
        'Temperature': 'mean',
        'Fuel_Price': 'mean',
        'MarkDown1': 'sum',
        'MarkDown2': 'sum',
        'MarkDown3': 'sum',
        'MarkDown4': 'sum',
        'MarkDown5': 'sum',
        'CPI': 'mean',
        'Unemployment': 'mean',
        'IsHoliday': 'max',
    }
    
    weekly_data = df.groupby('Date').agg(agg_rules).reset_index()
    weekly_data['Date'] = pd.to_datetime(weekly_data['Date'])
    weekly_data = weekly_data.set_index('Date')
    weekly_data['IsHoliday'] = weekly_data['IsHoliday'].astype(int)

    y = weekly_data['Weekly_Sales']
    
    # All possible exogenous variables for selection
    all_exog = weekly_data.drop(columns=['Weekly_Sales'], errors='ignore').copy()
    
    # Ensure all NaNs (from MarkDowns at start) are filled for SARIMAX
    all_exog = all_exog.fillna(0)
    
    return y, all_exog

def train_and_forecast_sarimax(y, all_exog, exog_vars, df_source, test_weeks=26, order=(0, 1, 1), seasonal_order=(0, 1, 0, 52), plot=False):
    """
    Trains, evaluates, and optionally plots a single SARIMAX model.
    It selects the specific features using 'exog_vars' from the full set 'all_exog'.
    """
    
    # Select the specific exogenous features for this run
    exog = all_exog[exog_vars]
    
    train_end_index = len(y) - test_weeks
    
    y_train = y.iloc[:train_end_index]
    y_test = y.iloc[train_end_index:]
    
    # Select exog variables for train and test sets
    exog_train = exog.iloc[:train_end_index]
    exog_test = exog.iloc[train_end_index:]
    
    # Train the SARIMAX model
    try:
        sarimax_model = SARIMAX(
            y_train, 
            exog=exog_train, # Use the selected exogenous data
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        results = sarimax_model.fit(disp=False)
        
        # Make forecast
        forecast_obj = results.get_forecast(steps=test_weeks, exog=exog_test)
        forecast_mean = forecast_obj.predicted_mean
        forecast_ci = forecast_obj.conf_int()
        
        #  Evaluate
        wmae = compute_weighted_MAE(y_test, forecast_mean, df_source=df_source)
        
        # Include AIC for internal tracking, though WMAE is the final metric
        aic = results.aic 
        
        # Plot (only if plot=True)
        if plot:
            plot_sarimax_forecast(
                train=y_train,
                test=y_test,
                forecast_mean=forecast_mean,
                forecast_ci=forecast_ci,
                title=f"SARIMAX Forecast ({order}{seasonal_order}, Features: {len(exog_vars)}) - WMAE: {wmae:,.0f}",
                wmae=wmae
            )
        
        return wmae, aic # Return both WMAE and AIC for the grid search functionality
    
    except Exception:
        # Return infinity WMAE and AIC for failed models
        return np.inf, np.inf
    
    except Exception:
        # Catch errors like convergence failure
        return np.inf, np.inf

# Combined Feature and Parameter Grid Search
def sarimax_feature_grid_search(y, all_exog, df_source, test_weeks=26, s=52):
    """Performs nested grid search: features (outer) and SARIMAX orders (inner)."""
    
    # feature sets to test (Addressing Multicollinearity)
    FEATURE_SETS = {
        'Set_1_Core': ['Unemployment', 'IsHoliday'],
        'Set_2_Core+Macro': ['Unemployment', 'IsHoliday', 'Fuel_Price', 'CPI'],
        'Set_3_Promo_Only': ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'IsHoliday'],
        'Set_4_Full_Exog': list(all_exog.columns), # All 10 features
    }
    
    # SARIMAX parameter grid
    p_values = [0, 1]
    d_values = [1]
    q_values = [0, 1]
    P_values = [0, 1]
    D_values = [0, 1] 
    Q_values = [0, 1]
    
    pdq = list(itertools.product(p_values, d_values, q_values))
    PDQs = list(itertools.product(P_values, D_values, Q_values, [s]))
    
    order_grid = list(itertools.product(pdq, PDQs))
    
    print(f"\n--- SARIMAX Nested Grid Search Initiated ---")
    print(f"Total feature sets: {len(FEATURE_SETS)}, Total orders per set: {len(order_grid)}")
    print(f"Total models to test: {len(FEATURE_SETS) * len(order_grid)}")
    
    best_wmae = np.inf
    best_result = {}
    
    # Outer Loop: Feature Sets
    for set_name, exog_vars in FEATURE_SETS.items():
        print(f"\nTesting Feature Set: {set_name} ({len(exog_vars)} features)")
        
        # Inner Loop: SARIMAX Orders
        for param in order_grid:
            order, seasonal_order = param
            
            wmae, aic = train_and_forecast_sarimax(
                y, all_exog, exog_vars, df_source, 
                test_weeks=test_weeks, order=order, seasonal_order=seasonal_order, plot=False
            )
            
            if wmae < best_wmae:
                best_wmae = wmae
                best_result = {
                    'WMAE': wmae,
                    'AIC': aic,
                    'Order': order,
                    'Seasonal_Order': seasonal_order,
                    'Feature_Set': set_name,
                    'Exog_Vars': exog_vars
                }
                print(f"--> NEW BEST: {set_name} {order}{seasonal_order} WMAE={wmae:,.2f}")
    
    print(f"\n--- Nested Grid Search Complete ---")
    
    # Print and Plot the Optimal Model
    print("----------------------------------------------------------------")
    print(f"   Optimal Model Found: {best_result['Feature_Set']}")
    print(f"   Order: {best_result['Order']}{best_result['Seasonal_Order']}")
    print(f"   Features: {best_result['Exog_Vars']}")
    print(f"   Final WMAE: {best_result['WMAE']:,.2f}")
    print("----------------------------------------------------------------")

    # Final run of the optimal model for plotting
    final_wmae = train_and_forecast_sarimax(
        y, all_exog, best_result['Exog_Vars'], df_source, 
        test_weeks=test_weeks, order=best_result['Order'], 
        seasonal_order=best_result['Seasonal_Order'], plot=True
    )
    
    return best_result

# Prepare data
y_sarimax, all_exog = prepare_data_for_sarimax(full_df)

# Run the Grid Search to find optimal parameters
final_best_model_result = sarimax_feature_grid_search(
    y=y_sarimax,
    all_exog=all_exog,
    df_source=full_df,
    test_weeks=26,
    s=52
)

optimal_order = final_best_model_result['Order']
optimal_seasonal_order = final_best_model_result['Seasonal_Order']
optimal_exog_vars = final_best_model_result['Exog_Vars']
optimal_wmae = final_best_model_result['WMAE']

# Final Model Run to plot best result
print(f"\n Final Model Run (Optimal Order: {optimal_order}{optimal_seasonal_order} | WMAE: {optimal_wmae:,.2f}) ")
print(f"Using Optimal Exogenous Features: {optimal_exog_vars} ")






