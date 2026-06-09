# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv')
test_weights = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')
inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
sales_train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
sales_test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv')
solution = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')


sales_train.info()


sales_test.info()


inventory.info()


calendar.info()


def combine_sales_data(target_df, inventory_df, calendar_df):
    """
    Combines sales data (target_df) with inventory and calendar data, modifying the target_df in place.
    
    Parameters:
    target_df (pd.DataFrame): The sales data (either from sales_train or sales_test), which will be modified.
    inventory_df (pd.DataFrame): The inventory data containing product details.
    calendar_df (pd.DataFrame): The calendar data containing information about holidays and events.
    """

    
    # Merge target_df with inventory_df on 'unique_id'
    target_df = target_df.merge(inventory_df[['unique_id', 'product_unique_id', 'name', 
                                  'L1_category_name_en', 'L2_category_name_en', 
                                  'L3_category_name_en', 'L4_category_name_en']], 
                    on='unique_id', how='left')
    
    # Ensure the 'date' column in target_df is in datetime format
    #target_df['date'] = pd.to_datetime(target_df['date'])  # Ensure the date is in datetime format
    
    # Merge target_df with calendar_df on 'date' and 'warehouse'
    target_df = target_df.merge(calendar_df[['date', 'warehouse', 'holiday', 'holiday_name', 
                                 'shops_closed', 'winter_school_holidays', 'school_holidays']], 
                    on=['date', 'warehouse'], how='left')

    return target_df




sales_train = combine_sales_data(sales_train, inventory, calendar)
sales_test = combine_sales_data(sales_test, inventory, calendar)


sales_train['date'] = pd.to_datetime(sales_train['date'])
sales_test['date'] = pd.to_datetime(sales_test['date'])


# Check the structure of the dataset
sales_train.info()


# Check the structure of the dataset
sales_test.info()


# Summary statistics for numerical features
sales_train.describe()


# Distribution of sales
plt.figure(figsize=(10, 6))
sns.histplot(sales_train['sales'], kde=True, bins=50)
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()


# Distribution of total orders
plt.figure(figsize=(10, 6))
sns.histplot(sales_train['total_orders'], kde=True, bins=50)
plt.title('Distribution of Total Orders')
plt.xlabel('Total Orders')
plt.ylabel('Frequency')
plt.show()


# Distribution of selling price
plt.figure(figsize=(10, 6))
sns.histplot(sales_train['sell_price_main'], kde=True, bins=50)
plt.title('Distribution of Selling Price')
plt.xlabel('Sell Price')
plt.ylabel('Frequency')
plt.show()


# Distribution of availability
plt.figure(figsize=(10, 6))
sns.histplot(sales_train['availability'], kde=True, bins=50)
plt.title('Distribution of Availability')
plt.xlabel('Availability')
plt.ylabel('Frequency')
plt.show()


# Boxplot to detect outliers for sales
plt.figure(figsize=(10, 6))
sns.boxplot(x=sales_train['sales'])
plt.title('Boxplot of Sales')
plt.show()

# Boxplot to detect outliers for total orders
plt.figure(figsize=(10, 6))
sns.boxplot(x=sales_train['total_orders'])
plt.title('Boxplot of Total Orders')
plt.show()


#Sales over Time
# Set 'date' as index for time series analysis
sales_train.set_index('date', inplace=True)

# Plot sales over time (aggregate by month)
monthly_sales = sales_train['sales'].resample('M').sum()
plt.figure(figsize=(14, 7))
monthly_sales.plot()
plt.title('Monthly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()


#Total Orders over Time
# Plot total orders over time (aggregate by month)
monthly_orders = sales_train['total_orders'].resample('M').sum()
plt.figure(figsize=(14, 7))
monthly_orders.plot()
plt.title('Monthly Total Orders Over Time')
plt.xlabel('Date')
plt.ylabel('Total Orders')
plt.show()


#Heatmap of Sales by Month and Day of Week
# Extract month and weekday from the date
sales_train['month'] = sales_train.index.month
sales_train['weekday'] = sales_train.index.weekday

# Pivot table for heatmap: sum of sales by month and weekday
sales_heatmap = sales_train.pivot_table(values='sales', index='weekday', columns='month', aggfunc='sum')

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(sales_heatmap, cmap='Blues', annot=True, fmt='.0f', cbar=True)
plt.title('Heatmap of Sales by Month and Weekday')
plt.xlabel('Month')
plt.ylabel('Weekday')
plt.show()



# Compute correlations between numerical features
correlation_matrix = sales_train[['sales', 'total_orders', 'sell_price_main', 'availability', 'type_0_discount', 
                                  'type_1_discount', 'type_2_discount', 'type_3_discount', 'type_4_discount', 
                                  'type_5_discount', 'type_6_discount']].corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Correlation Matrix')
plt.show()


# Sales vs Sell Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x=sales_train['sell_price_main'], y=sales_train['sales'])
plt.title('Sales vs Sell Price')
plt.xlabel('Sell Price')
plt.ylabel('Sales')
plt.show()

# Sales vs Total Orders
plt.figure(figsize=(10, 6))
sns.scatterplot(x=sales_train['total_orders'], y=sales_train['sales'])
plt.title('Sales vs Total Orders')
plt.xlabel('Total Orders')
plt.ylabel('Sales')
plt.show()

# Sales vs Availability
plt.figure(figsize=(10, 6))
sns.scatterplot(x=sales_train['availability'], y=sales_train['sales'])
plt.title('Sales vs Availability')
plt.xlabel('Availability')
plt.ylabel('Sales')
plt.show()



# Scatter plot of each discount type vs Sales
discount_columns = ['type_0_discount', 'type_1_discount', 'type_2_discount', 'type_3_discount', 
                    'type_4_discount', 'type_5_discount', 'type_6_discount']

for col in discount_columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=sales_train[col], y=sales_train['sales'])
    plt.title(f'Sales vs {col}')
    plt.xlabel(col)
    plt.ylabel('Sales')
    plt.show()



# Sales across warehouse types
plt.figure(figsize=(10, 6))
sns.boxplot(x=sales_train['warehouse'], y=sales_train['sales'])
plt.title('Sales Distribution Across Warehouses')
plt.xlabel('Warehouse')
plt.ylabel('Sales')
plt.show()

# Sales across L1 category
plt.figure(figsize=(10, 6))
sns.boxplot(x=sales_train['L1_category_name_en'], y=sales_train['sales'])
plt.title('Sales Distribution Across L1 Category')
plt.xlabel('L1 Category')
plt.ylabel('Sales')
plt.xticks(rotation=90)
plt.show()

# Sales across L2 category
plt.figure(figsize=(10, 6))
sns.boxplot(x=sales_train['L2_category_name_en'], y=sales_train['sales'])
plt.title('Sales Distribution Across L2 Category')
plt.xlabel('L2 Category')
plt.ylabel('Sales')
plt.xticks(rotation=90)
plt.show()



# Sales by holiday presence
plt.figure(figsize=(10, 6))
sns.boxplot(x=sales_train['holiday'], y=sales_train['sales'])
plt.title('Sales Distribution by Holiday')
plt.xlabel('Holiday')
plt.ylabel('Sales')
plt.show()

# Sales by school holidays
plt.figure(figsize=(10, 6))
sns.boxplot(x=sales_train['school_holidays'], y=sales_train['sales'])
plt.title('Sales Distribution by School Holidays')
plt.xlabel('School Holidays')
plt.ylabel('Sales')
plt.show()



# Aggregate sales by month
sales_train['month'] = sales_train.index.month
monthly_sales = sales_train.groupby('month')['sales'].sum()

# Plot the monthly sales trend
plt.figure(figsize=(10, 6))
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker='o')
plt.title('Total Sales by Month (Seasonality Analysis)')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(ticks=np.arange(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.show()



# Extract weekday from the date (0 = Monday, 6 = Sunday)
sales_train['weekday'] = sales_train.index.weekday

# Aggregate sales by weekday
weekday_sales = sales_train.groupby('weekday')['sales'].sum()

# Plot sales by weekday
plt.figure(figsize=(10, 6))
sns.lineplot(x=weekday_sales.index, y=weekday_sales.values, marker='o')
plt.title('Total Sales by Weekday')
plt.xlabel('Weekday')
plt.ylabel('Total Sales')
plt.xticks(ticks=np.arange(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.grid(True)
plt.show()



# Extract week of the year
sales_train['week_of_year'] = sales_train.index.isocalendar().week

# Aggregate sales by week of the year
weekly_sales = sales_train.groupby('week_of_year')['sales'].sum()

# Plot weekly sales trend
plt.figure(figsize=(12, 6))
sns.lineplot(x=weekly_sales.index, y=weekly_sales.values, marker='o')
plt.title('Total Sales by Week of the Year')
plt.xlabel('Week of the Year')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()



# Aggregate sales based on holiday status (1 = holiday, 0 = no holiday)
holiday_sales = sales_train.groupby('holiday')['sales'].sum()

# Plot holiday sales
plt.figure(figsize=(10, 6))
sns.barplot(x=holiday_sales.index, y=holiday_sales.values)
plt.title('Total Sales by Holiday')
plt.xlabel('Holiday (0 = No, 1 = Yes)')
plt.ylabel('Total Sales')
plt.show()



# Aggregate sales based on school holidays
school_holiday_sales = sales_train.groupby('school_holidays')['sales'].sum()

# Plot school holiday sales
plt.figure(figsize=(10, 6))
sns.barplot(x=school_holiday_sales.index, y=school_holiday_sales.values)
plt.title('Total Sales by School Holidays')
plt.xlabel('School Holidays (0 = No, 1 = Yes)')
plt.ylabel('Total Sales')
plt.show()



# Pivot table for heatmap: sum of sales by week of the year and month
sales_heatmap_seasonal = sales_train.pivot_table(values='sales', index='week_of_year', columns='month', aggfunc='sum')

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(sales_heatmap_seasonal, cmap='coolwarm', annot=True, fmt='.0f', cbar=True)
plt.title('Heatmap of Sales by Week of the Year and Month')
plt.xlabel('Month')
plt.ylabel('Week of the Year')
plt.show()



# Extract the year
sales_train['year'] = sales_train.index.year

# Aggregate sales by year
annual_sales = sales_train.groupby('year')['sales'].sum()

# Plot annual sales trends
plt.figure(figsize=(10, 6))
sns.lineplot(x=annual_sales.index, y=annual_sales.values, marker='o')
plt.title('Annual Sales Trend')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()



from statsmodels.tsa.seasonal import seasonal_decompose

# Resample sales data to monthly for decomposition
monthly_sales = sales_train['sales'].resample('M').sum()

# Perform seasonal decomposition
decomposition = seasonal_decompose(monthly_sales, model='additive', period=12) #model='multiplicative'

# Plot decomposition
plt.figure(figsize=(14, 8))
decomposition.plot()
plt.show()



#from statsmodels.tsa.stattools import adfuller

# Perform Augmented Dickey-Fuller test
#result = adfuller(sales_train['sales'].dropna())  # Drop NaNs before testing
#print(f"ADF Statistic: {result[0]}")
#print(f"p-value: {result[1]}")

# If p-value > 0.05, the series is likely non-stationary and may require differencing




#!pip install ruptures  # Install ruptures if not already installed
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from ruptures import Pelt

# Ensure 'sales' column is numeric and drop missing values
#sales_data = sales_train['sales'].dropna().values  # Convert to 1D NumPy array

# Apply change point detection
#model = "l2"  # Model type for change point detection
#algo = Pelt(model=model).fit(sales_data)  # Fit the model
#result = algo.predict(pen=10)  # Penalty parameter to control the number of change points

# Plot the change points
#plt.figure(figsize=(12, 6))
#plt.plot(sales_data, label='Sales Data')
#for cp in result:
#    plt.axvline(x=cp, color='r', linestyle='--', label='Change Point' if cp == result[0] else "")
#plt.title("Change Points in Sales Data")
#plt.xlabel("Time Index")
#plt.ylabel("Sales")
#plt.legend()
#plt.show()

