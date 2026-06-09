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


# Load the training data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')

# Load the test data
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

# Load the sample submission data
sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


# Inspect the first few rows of each dataset
print("Train Dataset:\n", train_df.head(), "\n")
print("Test Dataset:\n", test_df.head(), "\n")
print("Sample Submission:\n", sample_submission_df.head())


# Check for missing values in the train dataset
print("Missing values before handling:\n", train_df.isnull().sum())

# Fill missing values in 'num_sold' with 0 (assuming missing values mean no sales)
train_df['num_sold'] = train_df['num_sold'].fillna(0)

# Verify that there are no missing values
print("Missing values after handling:\n", train_df.isnull().sum())


# Get column names for train dataset
print("Train Dataset Columns:", train_df.columns)

# Get column names for test dataset
print("Test Dataset Columns:", test_df.columns)

# Get column names for sample submission dataset
print("Sample Submission Dataset Columns:", sample_submission_df.columns)


# Convert the 'date' column to datetime format
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

# Extract year, month, day, and day of the week
train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['day_of_week'] = train_df['date'].dt.day_name()

test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['day_of_week'] = test_df['date'].dt.day_name()

# Display the new columns for verification
print("Train Dataset with date features:\n", train_df.head())
print("Test Dataset with date features:\n", test_df.head())


import matplotlib.pyplot as plt
import seaborn as sns

# Aggregate sales by month
monthly_sales = train_df.groupby('month')['num_sold'].sum().reset_index()

# Plot monthly sales trend
plt.figure(figsize=(12, 6))
sns.lineplot(x='month', y='num_sold', data=monthly_sales, marker='o', color='blue')
plt.title('Monthly Sales Trend', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.grid(True, linestyle='--')
plt.xticks(range(1, 13))
plt.show()


# Suppress the FutureWarning by explicitly setting observed=True
day_of_week_sales = day_of_week_sales.groupby('day_of_week', observed=True)['num_sold'].sum().reset_index()

# Reorder day of the week for proper sequence
day_of_week_sales['day_of_week'] = pd.Categorical(day_of_week_sales['day_of_week'],
                                                  categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                                  ordered=True)

# Aggregate sales by day of the week
day_of_week_sales = day_of_week_sales.sort_values('day_of_week')

# Plot day of the week sales trend
plt.figure(figsize=(12, 6))
sns.barplot(x='day_of_week', y='num_sold', data=day_of_week_sales, palette='viridis')
plt.title('Day of the Week Sales Trend', fontsize=16, weight='bold')
plt.xlabel('Day of the Week', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  # Adjust layout for better spacing
plt.show()


# Grouping sales by date and aggregating the sales
time_series_data = train_df.groupby('date')['num_sold'].sum().reset_index()

# Set the index to 'date' for time series operations
time_series_data.set_index('date', inplace=True)

# Display time series data
print("Time Series Data Head:\n", time_series_data.head())



from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Decompose the time series
result = seasonal_decompose(time_series_data['num_sold'], model='additive', period=365)  # assuming yearly seasonality

# Plot the decomposition with better layout and styling
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
plt.suptitle('Seasonal Decomposition of Sales Data', fontsize=18, weight='bold')

# Plot trend
axes[0].plot(result.trend, color='blue')
axes[0].set_title('Trend Component', fontsize=16)
axes[0].grid(True, linestyle='--')

# Plot seasonal
axes[1].plot(result.seasonal, color='green')
axes[1].set_title('Seasonal Component', fontsize=16)
axes[1].grid(True, linestyle='--')

# Plot residuals
axes[2].plot(result.resid, color='red')
axes[2].set_title('Residuals', fontsize=16)
axes[2].grid(True, linestyle='--')

# Adjust spacing between subplots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Fit ARIMA model
model = ARIMA(time_series_data['num_sold'], order=(5, 1, 0))  # (p, d, q) parameters for ARIMA
model_fit = model.fit()

# Forecasting for the next 365 days
forecast = model_fit.forecast(steps=365)

# Plotting the forecast with enhanced visuals
plt.figure(figsize=(12, 6))
time_series_data['num_sold'].plot(label='Observed', color='blue', alpha=0.7, linewidth=2)
forecast.plot(label='Forecast', color='red', linewidth=2)
plt.title('Sales Forecast with ARIMA Model', fontsize=18, weight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


