# Kaggle Playground Series - Sticker Sales Forecasting
# Exploratory Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Set styling
plt.style.use('seaborn')
sns.set_palette("husl")

# Display all columns
pd.set_option('display.max_columns', None)


# Read the data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


# Display basic information about the datasets
print("Training Dataset Info:")
print(train_df.info())
print("\nFirst few rows of training data:")
print(train_df.head())


# Convert date column to datetime
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])


# Basic statistics of numerical columns
print("\nBasic statistics of numerical columns:")
print(train_df.describe())


# Check for missing values
print("\nMissing values in training data:")
print(train_df.isnull().sum())


# Unique values in categorical columns
print("\nUnique values in categorical columns:")
for col in ['country', 'store', 'product']:
    print(f"\n{col}:")
    print(train_df[col].value_counts())


# Time series analysis
# Add time-based features
train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['dayofweek'] = train_df['date'].dt.dayofweek


# Plot overall sales trend
plt.figure(figsize=(15, 6))
daily_sales = train_df.groupby('date')['num_sold'].sum().reset_index()
plt.plot(daily_sales['date'], daily_sales['num_sold'])
plt.title('Overall Daily Sales Trend')
plt.xlabel('Date')
plt.ylabel('Number of Products Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Sales by country
plt.figure(figsize=(12, 6))
country_sales = train_df.groupby('country')['num_sold'].sum().sort_values(ascending=False)
sns.barplot(x=country_sales.index, y=country_sales.values)
plt.title('Total Sales by Country')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Sales by product
plt.figure(figsize=(12, 6))
product_sales = train_df.groupby('product')['num_sold'].sum().sort_values(ascending=False)
sns.barplot(x=product_sales.index, y=product_sales.values)
plt.title('Total Sales by Product')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Monthly seasonality
plt.figure(figsize=(12, 6))
monthly_sales = train_df.groupby('month')['num_sold'].mean()
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values)
plt.title('Average Sales by Month')
plt.xlabel('Month')
plt.ylabel('Average Number of Products Sold')
plt.tight_layout()
plt.show()


# Day of week patterns
plt.figure(figsize=(12, 6))
dow_sales = train_df.groupby('dayofweek')['num_sold'].mean()
sns.lineplot(x=dow_sales.index, y=dow_sales.values)
plt.title('Average Sales by Day of Week')
plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
plt.ylabel('Average Number of Products Sold')
plt.tight_layout()
plt.show()


# Distribution of sales
plt.figure(figsize=(12, 6))
sns.histplot(data=train_df, x='num_sold', bins=50)
plt.title('Distribution of Number of Products Sold')
plt.xlabel('Number of Products Sold')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# Create a heatmap of sales by country and product
pivot_table = train_df.pivot_table(
    values='num_sold',
    index='country',
    columns='product',
    aggfunc='sum'
)
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='.0f')
plt.title('Sales Heatmap: Country vs Product')
plt.tight_layout()
plt.show()


# Box plot of sales by country
plt.figure(figsize=(15, 6))
sns.boxplot(data=train_df, x='country', y='num_sold')
plt.title('Sales Distribution by Country')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Statistical summary by country
print("\nSales statistics by country:")
print(train_df.groupby('country')['num_sold'].describe())


# Correlation analysis of numerical features
correlation_matrix = train_df[['num_sold', 'year', 'month', 'day', 'dayofweek']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()


# Time series decomposition for the top selling country
from statsmodels.tsa.seasonal import seasonal_decompose

top_country = country_sales.index[0]
country_data = train_df[train_df['country'] == top_country].groupby('date')['num_sold'].sum()
decomposition = seasonal_decompose(country_data, period=7)

plt.figure(figsize=(15, 12))
plt.subplot(411)
plt.plot(country_data)
plt.title(f'Time Series Decomposition for {top_country}')
plt.subplot(412)
plt.plot(decomposition.trend)
plt.title('Trend')
plt.subplot(413)
plt.plot(decomposition.seasonal)
plt.title('Seasonal')
plt.subplot(414)
plt.plot(decomposition.resid)
plt.title('Residual')
plt.tight_layout()
plt.show()

