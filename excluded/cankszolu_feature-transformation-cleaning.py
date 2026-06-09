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


import requests
import holidays  # Import the holidays library for country-specific public holiday data
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


# Load Datasets as dataframe
train_data = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_data = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Understand general schema of Train Dataset
train_data.info()


# Understand general schema of Test Dataset
test_data.info()


# Let's take a look at the first few rows of the training dataset to analyze its structure and 
# understand the data distribution
train_data.head()


# Null Data per Future
train_data.isnull().sum()


# Null Data per Future
test_data.isnull().sum()


#Null values in the 'num_sold' column are filled with the mean to prevent missing data from affecting the model.
train_data['num_sold'] = train_data['num_sold'].fillna(train_data['num_sold'].mean())


train_data.isnull().sum()


# Convert the 'date' column to datetime format for easier date-related operations
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])


# Extract features
train_data['year'] = train_data['date'].dt.year
train_data['month'] = train_data['date'].dt.month
train_data['day'] = train_data['date'].dt.day
train_data['day_of_week'] = train_data['date'].dt.dayofweek  # Monday = 0, Sunday = 6
train_data['quarter'] = train_data['date'].dt.quarter
train_data['is_weekend'] = train_data['day_of_week'].apply(lambda x: x in [5, 6])
train_data['day_of_year'] = train_data['date'].dt.dayofyear


test_data['year'] = test_data['date'].dt.year
test_data['month'] = test_data['date'].dt.month
test_data['day'] = test_data['date'].dt.day
test_data['day_of_week'] = test_data['date'].dt.dayofweek
test_data['quarter'] = test_data['date'].dt.quarter
test_data['is_weekend'] = test_data['day_of_week'].apply(lambda x: x in [5, 6])
test_data['day_of_year'] = test_data['date'].dt.dayofyear


# Function to determine the season based on the month
def get_season(month):
    if month in [12, 1, 2]:  # December, January, February -> Winter
        return 'Winter'
    elif month in [3, 4, 5]:  # March, April, May -> Spring
        return 'Spring'
    elif month in [6, 7, 8]:  # June, July, August -> Summer
        return 'Summer'
    else:  # September, October, November -> Autumn
        return 'Autumn'

# Applying the 'get_season' function to create a 'season' column based on the 'month' column
train_data['season'] = train_data['month'].apply(get_season)  # Add season information to the train dataset
test_data['season'] = test_data['month'].apply(get_season)  # Add season information to the test dataset


# Ratio Calculations for train dataset
ratios = {}


# 1. Day of Year Ratio
ratios['day_of_year_ratio'] = train_data.groupby('day_of_year')['num_sold'].mean() / train_data['num_sold'].mean()

# 2. Week of Year Ratio
train_data['week_of_year'] = train_data['date'].dt.isocalendar().week
test_data['week_of_year'] = test_data['date'].dt.isocalendar().week  # Test datasına ekleme yapıldı
ratios['week_of_year_ratio'] = train_data.groupby('week_of_year')['num_sold'].mean() / train_data['num_sold'].mean()

# 3. Season Ratio
ratios['season_ratio'] = train_data.groupby('season')['num_sold'].mean() / train_data['num_sold'].mean()

# 4. Month Ratio
ratios['month_ratio'] = train_data.groupby('month')['num_sold'].mean() / train_data['num_sold'].mean()

# 5. Product Ratio
ratios['product_ratio'] = train_data.groupby('product')['num_sold'].mean() / train_data['num_sold'].mean()

# 6. Country Ratio
ratios['country_ratio'] = train_data.groupby('country')['num_sold'].mean() / train_data['num_sold'].mean()

# 7. Quarter Ratio
ratios['quarter_ratio'] = train_data.groupby('quarter')['num_sold'].mean() / train_data['num_sold'].mean()

# 8. Day of Week Ratio
ratios['day_of_week_ratio'] = train_data.groupby('day_of_week')['num_sold'].mean() / train_data['num_sold'].mean()

# Apply Ratios to Train and Test Datasets
for ratio_name, ratio_values in ratios.items():
    train_data[ratio_name] = train_data[ratio_name.split('_ratio')[0]].map(ratio_values).fillna(1)
    test_data[ratio_name] = test_data[ratio_name.split('_ratio')[0]].map(ratio_values).fillna(1)


# Apply Ratios to Train and Test Datasets
for ratio_name, ratio_values in ratios.items():
    train_data[ratio_name] = train_data[ratio_name.split('_ratio')[0]].map(ratio_values).fillna(1)
    test_data[ratio_name] = test_data[ratio_name.split('_ratio')[0]].map(ratio_values).fillna(1)


# Drop the temporary 'week_of_year' column from both datasets
train_data.drop(columns=['week_of_year'], inplace=True, errors='ignore')
test_data.drop(columns=['week_of_year'], inplace=True, errors='ignore')


# Mapping countries to their alpha-2 country codes
alpha2 = dict(zip(np.sort(train_data.country.unique()), ['CA', 'FI', 'IT', 'KE', 'NO', 'SG']))

# Create a dictionary containing public holiday lists for each country (from 2010 to 2019)
h = {c: holidays.country_holidays(a, years=range(2010, 2020)) for c, a in alpha2.items()}

# Initialize the is_holiday, is_pre_holiday, and last_day_is_holiday columns to 0
train_data['is_holiday'] = 0
test_data['is_holiday'] = 0

train_data['is_pre_holiday'] = 0
test_data['is_pre_holiday'] = 0

train_data['last_day_is_holiday'] = 0
test_data['last_day_is_holiday'] = 0

# Mark holidays, pre-holiday, and last-day-as-holiday as 1 (True) for each country
for c in alpha2:
    holiday_dates = list(h[c].keys())  # Extract only the holiday dates (as datetime)
    
    # For training data
    train_data.loc[train_data.country == c, 'is_holiday'] = train_data.date.isin(holiday_dates).astype(int)
    train_data.loc[train_data.country == c, 'is_pre_holiday'] = train_data.date.isin(
        (pd.to_datetime(holiday_dates) - pd.Timedelta(days=1)).date
    ).astype(int)
    train_data.loc[train_data.country == c, 'last_day_is_holiday'] = train_data.date.isin(
        (pd.to_datetime(holiday_dates) + pd.Timedelta(days=1)).date
    ).astype(int)
    
    # For test data
    test_data.loc[test_data.country == c, 'is_holiday'] = test_data.date.isin(holiday_dates).astype(int)
    test_data.loc[test_data.country == c, 'is_pre_holiday'] = test_data.date.isin(
        (pd.to_datetime(holiday_dates) - pd.Timedelta(days=1)).date
    ).astype(int)
    test_data.loc[test_data.country == c, 'last_day_is_holiday'] = test_data.date.isin(
        (pd.to_datetime(holiday_dates) + pd.Timedelta(days=1)).date
    ).astype(int)


# Convert the is_holiday, is_pre_holiday, and last_day_is_holiday features to boolean type
train_data['is_holiday'] = train_data['is_holiday'].astype(bool)
test_data['is_holiday'] = test_data['is_holiday'].astype(bool)

train_data['is_pre_holiday'] = train_data['is_pre_holiday'].astype(bool)
test_data['is_pre_holiday'] = test_data['is_pre_holiday'].astype(bool)

train_data['last_day_is_holiday'] = train_data['last_day_is_holiday'].astype(bool)
test_data['last_day_is_holiday'] = test_data['last_day_is_holiday'].astype(bool)

# Verification: Check the data types after conversion
print(train_data[['is_holiday', 'is_pre_holiday', 'last_day_is_holiday']].dtypes)
print(test_data[['is_holiday', 'is_pre_holiday', 'last_day_is_holiday']].dtypes)


# Function to get GDP per capita for a given country and year
def get_gdp_per_capita(country, year):
    alpha3 = {'Canada': 'CAN', 'Finland': 'FIN', 'Italy': 'ITA',
              'Kenya': 'KEN', 'Norway': 'NOR', 'Singapore': 'SGP'}
    
    url = "https://api.worldbank.org/v2/country/{0}/indicator/NY.GDP.PCAP.CD?date={1}&format=json".format(
        alpha3[country], year)
    
    response = requests.get(url).json()
    return response[1][0]['value']

# Create lists to store country, year, and GDP data
countrys = []
years = []
gdps = []

# Get GDP per capita for each country and year (2010-2019)
for country in ['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore']:
    for year in range(2010, 2020):
        countrys.append(country)
        years.append(year)
        gdps.append(get_gdp_per_capita(country, year))

# Create a DataFrame with the GDP data
gdp_df = pd.DataFrame({"country": countrys, "year": years, "gdp": gdps})

# Merge GDP data with train and test data based on 'country' and 'year'
train_data = pd.merge(train_data, gdp_df, on=['country', 'year'], how='left')
test_data = pd.merge(test_data, gdp_df, on=['country', 'year'], how='left')


# Check the first few rows to confirm the merge
print(train_data.head()[['country', 'year', 'gdp']])
print(test_data.head()[['country', 'year', 'gdp']])


scaler = StandardScaler()
train_data["gdp"] = scaler.fit_transform(train_data[["gdp"]])
test_data["gdp"] = scaler.fit_transform(test_data[["gdp"]])


# month_country: Combination of month and country
train_data['month_country'] = train_data['month'].astype(str) + "_" + train_data['country']
test_data['month_country'] = test_data['month'].astype(str) + "_" + test_data['country']

# month_store: Combination of month and store
train_data['month_store'] = train_data['month'].astype(str) + "_" + train_data['store']
test_data['month_store'] = test_data['month'].astype(str) + "_" + test_data['store']

# month_product: Combination of month and product
train_data['month_product'] = train_data['month'].astype(str) + "_" + train_data['product']
test_data['month_product'] = test_data['month'].astype(str) + "_" + test_data['product']

# month_country_store: Combination of month, country, and store
train_data['month_country_store'] = train_data['month'].astype(str) + "_" + train_data['country'] + "_" + train_data['store']
test_data['month_country_store'] = test_data['month'].astype(str) + "_" + test_data['country'] + "_" + test_data['store']

# month_country_product: Combination of month, country, and product
train_data['month_country_product'] = train_data['month'].astype(str) + "_" + train_data['country'] + "_" + train_data['product']
test_data['month_country_product'] = test_data['month'].astype(str) + "_" + test_data['country'] + "_" + test_data['product']

# month_store_product: Combination of month, store, and product
train_data['month_store_product'] = train_data['month'].astype(str) + "_" + train_data['store'] + "_" + train_data['product']
test_data['month_store_product'] = test_data['month'].astype(str) + "_" + test_data['store'] + "_" + test_data['product']


categorical_columns = ['month_country', 'month_store', 'month_product', 
                       'month_country_store', 'month_country_product', 'month_store_product']


for col in categorical_columns:
    train_unique = set(train_data[col].unique())
    test_unique = set(test_data[col].unique())
    
    # Train setinde olmayan test kategorileri için "unknown"
    train_data[col] = train_data[col].astype(str)
    test_data[col] = test_data[col].astype(str)
    train_data[col] = train_data[col].apply(lambda x: x if x in test_unique else 'unknown')
    test_data[col] = test_data[col].apply(lambda x: x if x in train_unique else 'unknown')

# Tüm kategorik sütunları encode etme
for col in categorical_columns:
    combined = pd.concat([train_data[col], test_data[col]], axis=0)
    combined_encoded, _ = pd.factorize(combined)
    train_data[col] = combined_encoded[:len(train_data)]
    test_data[col] = combined_encoded[len(train_data):]


# Circular encoding for month
train_data['month_sin'] = np.sin(2 * np.pi * train_data['month'] / 12)
train_data['month_cos'] = np.cos(2 * np.pi * train_data['month'] / 12)
test_data['month_sin'] = np.sin(2 * np.pi * test_data['month'] / 12)
test_data['month_cos'] = np.cos(2 * np.pi * test_data['month'] / 12)

# Normalize day
train_data['day_normalized'] = train_data['day'] / 31
test_data['day_normalized'] = test_data['day'] / 31

# Circular encoding for day_of_week
train_data['dow_sin'] = np.sin(2 * np.pi * train_data['day_of_week'] / 7)
train_data['dow_cos'] = np.cos(2 * np.pi * train_data['day_of_week'] / 7)
test_data['dow_sin'] = np.sin(2 * np.pi * test_data['day_of_week'] / 7)
test_data['dow_cos'] = np.cos(2 * np.pi * test_data['day_of_week'] / 7)

# Applying sine and cosine transformations to the 'day_of_year' feature to capture its cyclical nature
train_data['day_of_year_sin'] = np.sin(2 * np.pi * train_data['day_of_year'] / 365)
train_data['day_of_year_cos'] = np.cos(2 * np.pi * train_data['day_of_year'] / 365)
test_data['day_of_year_sin'] = np.sin(2 * np.pi * test_data['day_of_year'] / 365)
test_data['day_of_year_cos'] = np.cos(2 * np.pi * test_data['day_of_year'] / 365)

# Diff of year start
train_data['time_since_start'] = train_data['year'] - train_data['year'].min()
test_data['time_since_start'] = test_data['year'] - train_data['year'].min()


train_data.info()


# Test: Unique values for each column to determine encoding needs
for col in train_data.columns:
    unique_vals = train_data[col].nunique()
    print(f"Column '{col}': {unique_vals} unique values")


# Identifying and Dropping Unnecessary Columns
columns_to_drop = ['date', 'month', 'day','day_of_week','day_of_year']

train_data = train_data.drop(columns=columns_to_drop, errors='ignore')
test_data = test_data.drop(columns=columns_to_drop, errors='ignore')

train_data = train_data.drop(columns='id', errors='ignore')


# Test: Unique values for each column to determine encoding needs
for col in train_data.columns:
    unique_vals = train_data[col].nunique()
    print(f"Column '{col}': {unique_vals} unique values")


# Categorical columns to be one-hot encoded
one_hot_columns = ['country', 'store', 'product', 'quarter', 'season']

# Boolean columns to be binary encoded
binary_columns = ['is_weekend', 'is_holiday', 'is_pre_holiday', 'last_day_is_holiday']


# One-Hot Encoding
train_data = pd.get_dummies(train_data, columns=one_hot_columns)
test_data = pd.get_dummies(test_data, columns=one_hot_columns)


for col in binary_columns:
    train_data[col] = train_data[col].astype(bool)
    test_data[col] = test_data[col].astype(bool)


# dataset CSV export
train_data.to_csv('train_data_4.csv', index=False)

# Test dataset CSV export
test_data.to_csv('test_data_4.csv', index=False)


# Test: Unique values for each column to determine encoding needs
for col in train_data.columns:
    unique_vals = train_data[col].nunique()
    print(f"Column '{col}': {unique_vals} unique values")


# Test: Unique values for each column to determine encoding needs
for col in test_data.columns:
    unique_vals = test_data[col].nunique()
    print(f"Column '{col}': {unique_vals} unique values")


train_data.info()


test_data.info()

