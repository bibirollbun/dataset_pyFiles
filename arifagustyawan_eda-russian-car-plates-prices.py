import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Load the training data
train = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')


# Display basic information
print("Training data shape:", train.shape)
print("Test data shape:", test.shape)


# Check the first few rows
print("\nFirst few rows of training data:")
print(train.head())


# Check the data types and missing values
print("\nTraining data info:")
print(train.info())


# Check for missing values
print("\nMissing values in training data:")
print(train.isnull().sum())


# Basic statistics of the target variable
print("\nStatistics of the price:")
print(train['price'].describe())


# Plot distribution of prices
plt.figure(figsize=(10, 6))
sns.histplot(train['price'], kde=True)
plt.title('Distribution of Car Plate Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.xscale('log')  # Using log scale due to potential price skew
plt.show()


# Parse plate information
def parse_plate(plate):
    if isinstance(plate, str):
        # Extract all letters (series) and digits (number)
        letters = ''.join(re.findall(r'[A-Z]', plate))
        digits = ''.join(re.findall(r'\d', plate))
        
        # Get region code (last numbers)
        region_code = None
        if digits:
            region_code = digits[-3:] if len(digits) >= 3 else digits
            number = digits[:-len(region_code)] if len(digits) > len(region_code) else ''
        else:
            number = ''
        
        # Get series (letters)
        series = letters
        
        return {
            'series': series,
            'number': number,
            'region_code': region_code,
            'series_length': len(series),
            'number_length': len(number)
        }
    return {'series': None, 'number': None, 'region_code': None, 'series_length': 0, 'number_length': 0}


# Apply the parse_plate function to both train and test datasets
train_plates = pd.DataFrame(train['plate'].apply(parse_plate).tolist())
test_plates = pd.DataFrame(test['plate'].apply(parse_plate).tolist())


# Add the parsed information to the original dataframes
train = pd.concat([train, train_plates], axis=1)
test = pd.concat([test, test_plates], axis=1)


# Convert date to datetime and extract features
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])


# Extract date features
for df in [train, test]:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter


# Check for special patterns in numbers (like repetitions, sequential, etc.)
def extract_number_patterns(number):
    if not isinstance(number, str) or not number:
        return {
            'is_sequential': 0,
            'is_repeated': 0,
            'has_repeated_digits': 0,
            'unique_digits': 0
        }
    
    # Check for sequential numbers (like 123, 456, 789)
    sequential_patterns = ['123', '234', '345', '456', '567', '678', '789', '987', '876', '765', '654', '543', '432', '321']
    is_sequential = any(pattern in number for pattern in sequential_patterns)
    
    # Check for repeated numbers (like 111, 222, 333)
    repeated_patterns = [f"{i}{i}{i}" for i in range(10)]
    is_repeated = any(pattern in number for pattern in repeated_patterns)
    
    # Check for repeating digits (like 001, 100, 220)
    unique_digits = len(set(number))
    has_repeated_digits = unique_digits < len(number)
    
    return {
        'is_sequential': int(is_sequential),
        'is_repeated': int(is_repeated),
        'has_repeated_digits': int(has_repeated_digits),
        'unique_digits': unique_digits
    }


# Apply number pattern extraction to both datasets
train_number_patterns = pd.DataFrame(train['number'].apply(extract_number_patterns).tolist())
test_number_patterns = pd.DataFrame(test['number'].apply(extract_number_patterns).tolist())


# Add the pattern information to the dataframes
train = pd.concat([train, train_number_patterns], axis=1)
test = pd.concat([test, test_number_patterns], axis=1)


# Load government and region code information
import sys
sys.path.append("/kaggle/input/russian-car-plates-prices-prediction")

from supplemental_english import GOVERNMENT_CODES, REGION_CODES


def is_government_plate(series, number, region_code):
    for x in GOVERNMENT_CODES:
        if series == x[0] and number in range(x[1][0], x[1][1]+1) and region_code == x[2]:
            return 1
    return 0


# Apply government plate check
train['is_government'] = train.apply(lambda row: is_government_plate(row['series'], row['number'], row['region_code']), axis=1)
test['is_government'] = test.apply(lambda row: is_government_plate(row['series'], row['number'], row['region_code']), axis=1)


# Exploration of region codes
plt.figure(figsize=(12, 6))
top_regions = train['region_code'].value_counts().head(20)
sns.barplot(x=top_regions.index, y=top_regions.values)
plt.title('Top 20 Region Codes')
plt.xticks(rotation=90)
plt.xlabel('Region Code')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# Analyze price by region code
plt.figure(figsize=(12, 6))
region_avg_price = train.groupby('region_code')['price'].mean().sort_values(ascending=False).head(20)
sns.barplot(x=region_avg_price.index, y=region_avg_price.values)
plt.title('Average Price by Region Code (Top 20)')
plt.xticks(rotation=90)
plt.xlabel('Region Code')
plt.ylabel('Average Price')
plt.tight_layout()
plt.show()


# Analyze time trends
plt.figure(figsize=(12, 6))
time_trend = train.groupby('year')['price'].mean()
plt.plot(time_trend.index, time_trend.values, 'o-')
plt.title('Average Price by Year')
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.grid(True)
plt.show()


# Analyze impact of special patterns on price
plt.figure(figsize=(10, 6))
pattern_comparison = train.groupby('is_repeated')['price'].mean()
pattern_comparison = pd.DataFrame({'Pattern': ['Regular', 'Repeated'], 'Price': pattern_comparison.values})
sns.barplot(x='Pattern', y='Price', data=pattern_comparison)
plt.title('Impact of Repeated Numbers on Price')
plt.xlabel('Number Pattern')
plt.ylabel('Average Price')
plt.show()


# Check correlation matrix of numerical features
numerical_cols = ['price', 'year', 'month', 'day', 'day_of_week', 'is_sequential', 
                 'is_repeated', 'has_repeated_digits', 'unique_digits', 'is_government']
correlation_matrix = train[numerical_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.show()


# Check if government plates have higher prices
govt_comparison = train.groupby('is_government')['price'].agg(['mean', 'median', 'count'])
print("\nPrice comparison between government and regular plates:")
print(govt_comparison)


# Calculate how many unique plates appear multiple times
duplicate_plates = train['plate'].value_counts()
print(f"\nNumber of plates that appear multiple times: {sum(duplicate_plates > 1)}")
print(f"Max appearances of a single plate: {duplicate_plates.max()}")


# Summary of findings
print("\n=== Summary of Key Findings ===")
print(f"1. Price range: {train['price'].min()} to {train['price'].max()}, with a mean of {train['price'].mean():.2f}")
print(f"2. Number of unique plates: {train['plate'].nunique()} out of {len(train)} records")
print(f"3. Top region code: {train['region_code'].value_counts().idxmax()} with {train['region_code'].value_counts().max()} occurrences")
print(f"4. Date range: {train['date'].min().date()} to {train['date'].max().date()}")

