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


import pandas as pd

file_path = '/kaggle/input/demand-forecasting-kernels-only/train.csv'

df = pd.read_csv(file_path)

df.head()



df.info() # Get a concise summary of the DataFrame
df.describe()  # Generate descriptive statistics for numerical columns
df.duplicated().sum() # Check for duplicate rows in the dataset


# Check the shape of the dataset (rows, columns)
print("Dataset shape:", df.shape)


# Display the column names for quick reference
print("Columns:", df.columns.tolist()) 


# Check for missing values in each column
print("Missing values per column:")
print(df.isnull().sum())


# Check unique values in each column
for col in df.columns:
    print(f"Column: {col} → Unique values: {df[col].nunique()}")


# Display data types for all columns
print("Data types:")
print(df.dtypes)


# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])
# Sort by date just in case
df = df.sort_values('date')

# Create time-based features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek

# Drop if needed (مثلاً لو فيه قيم مكررة)
df = df.drop_duplicates()


# Identify missing values per column before further processing
print("Missing values before preprocessing:\n", df.isnull().sum())


# Reset the index to keep the DataFrame tidy
df.reset_index(drop=True, inplace=True)


# Display first few rows to verify new columns and transformations
print(df.head())



# Display column data types after transformation
print("\nData types after preprocessing:\n", df.dtypes)


# Display shape to ensure no rows were unintentionally dropped
print("\nDataset shape after preprocessing:", df.shape)


# Total order size / Get overall statistics for the 'sales' column to understand its central tendency and spread
print("Sales descriptive statistics:\n", df['sales'].describe())




# Calculate the average sales for each store
avg_sales_per_store = df.groupby('store')['sales'].mean().sort_values(ascending=False)
print("\nAverage sales per store:\n", avg_sales_per_store)


# Calculate the average sales for each item
avg_sales_per_item = df.groupby('item')['sales'].mean().sort_values(ascending=False)
print("\nAverage sales per item:\n", avg_sales_per_item)


# Aggregate sales by date to observe the demand trend across the full timeline
daily_sales = df.groupby('date')['sales'].sum()
print("\nTotal sales over time (first few rows):\n", daily_sales.head())



# Get the date with the maximum total sales
max_sales_date = daily_sales.idxmax()
max_sales_value = daily_sales.max()
print(f"\nHighest sales day: {max_sales_date} with total sales: {max_sales_value}")


# Get the date with the minimum total sales
min_sales_date = daily_sales.idxmin()
min_sales_value = daily_sales.min()
print(f"Lowest sales day: {min_sales_date} with total sales: {min_sales_value}")


# Calculate monthly total sales to identify seasonality (e.g., peaks in certain months)
monthly_sales = df.groupby('month')['sales'].sum().sort_index()
print("\nMonthly total sales:\n", monthly_sales)


# Calculate average sales for each day of the week 
weekly_sales_pattern = df.groupby('dayofweek')['sales'].mean()
print("\nAverage sales by day of the week:\n", weekly_sales_pattern)



# Check the correlation between numeric columns (e.g., sales vs other numeric variables)
correlation_matrix = df.corr()
print("\nCorrelation matrix:\n", correlation_matrix)



# Quick check using interquartile range (IQR) method
Q1 = df['sales'].quantile(0.25)  # First quartile (25%)
Q3 = df['sales'].quantile(0.75)  # Third quartile (75%)
IQR = Q3 - Q1                     # Interquartile range
lower_bound = Q1 - 1.5 * IQR      # Lower threshold for outliers
upper_bound = Q3 + 1.5 * IQR      # Upper threshold for outliers
outliers = df[(df['sales'] < lower_bound) | (df['sales'] > upper_bound)]
print(f"\nNumber of potential outliers in sales: {outliers.shape[0]}")



import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

# Distribution of sales
plt.figure(figsize=(10, 5))  # Set figure size
sns.histplot(df['sales'], bins=50, kde=True)  # Histogram with Kernel Density Estimate (KDE) curve
plt.title("Distribution of Sales")  # Add a title
plt.xlabel("Sales")  # X-axis label
plt.ylabel("Frequency")  # Y-axis label
plt.show()





# Sales over time
plt.figure(figsize=(14, 5))  # Wider figure for time-series
daily_sales = df.groupby('date')['sales'].sum()  # Aggregate sales by date
plt.plot(daily_sales.index, daily_sales.values, color='blue', linewidth=2)
plt.title("Total Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.grid(True)  # Add grid lines for better readability
plt.show()




# Sales by store
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='store', y='sales')
plt.title("Sales Distribution per Store")
plt.xlabel("Store")
plt.ylabel("Sales")
plt.show()


#Average sales per month
plt.figure(figsize=(10, 5))
monthly_sales = df.groupby('month')['sales'].mean()
sns.barplot(x=monthly_sales.index, y=monthly_sales.values, palette="viridis")
plt.title("Average Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Average Sales")
plt.show()


# Average sales by day of week -----
plt.figure(figsize=(10, 5))
weekly_sales = df.groupby('dayofweek')['sales'].mean()
sns.barplot(x=weekly_sales.index, y=weekly_sales.values, palette="coolwarm")
plt.title("Average Sales by Day of the Week")
plt.xlabel("Day of Week (0=Monday, 6=Sunday)")
plt.ylabel("Average Sales")
plt.show()


#Correlation heatmap -----
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()  # Calculate correlation coefficients
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# Check for missing values in each column to see if any data is incomplete
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)


# Check for duplicate rows to identify if the dataset contains redundant information
duplicate_rows = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_rows}")


# Check the data types of each column to ensure they are appropriate for analysis
print("\nData types before conversion:\n", df.dtypes)


# Example: Convert 'date' column to datetime if it is not already in datetime format
if df['date'].dtype == 'object':
    df['date'] = pd.to_datetime(df['date'])
    print("\n' date' column converted to datetime.")


# Check for outliers using basic statistics (e.g., for sales column)
Q1 = df['sales'].quantile(0.25)  # First quartile (25%)
Q3 = df['sales'].quantile(0.75)  # Third quartile (75%)
IQR = Q3 - Q1  # Interquartile range
lower_bound = Q1 - 1.5 * IQR  # Lower threshold for outliers
upper_bound = Q3 + 1.5 * IQR  # Upper threshold for outliers

# Count how many outliers exist based on the defined thresholds
outliers = ((df['sales'] < lower_bound) | (df['sales'] > upper_bound)).sum()
print(f"\nNumber of outliers in 'sales' column: {outliers}")


# Final check for dataset shape after handling
print(f"\nFinal dataset shape after handling: {df.shape}")

