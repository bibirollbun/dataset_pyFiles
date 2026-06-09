# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np 
import pandas as pd 


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import gc
import warnings

# PACF - ACF
# ------------------------------------------------------
import statsmodels.api as sm

# DATA VISUALIZATION
# ------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# CONFIGURATIONS
# ------------------------------------------------------
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
train.iloc[:10]


train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])


train.describe()


train['num_sold'].value_counts()


train.isna().sum().sort_values(ascending=False)


# Filter rows where num_sold is missing
missing_num_sold = train[train['num_sold'].isnull()]

# Group by store and country to count missing values
missing_summary = missing_num_sold.groupby(['store', 'country'])['num_sold'].size().reset_index(name='missing_count')
missing_summary


test.isna().sum().sort_values(ascending=False)


train['date'].describe()


test['date'].describe()


# 1. Total sales over time
plt.figure(figsize=(28, 6))
train.groupby('date')['num_sold'].sum().plot(title='Total Sales Over Time', xlabel='Date', ylabel='Number of Products Sold')
plt.grid()
plt.show()


sales_by_product = train.groupby(['date', 'product'])['num_sold'].sum().reset_index()

# Pivot the data to have products as columns and dates as rows
sales_pivot = sales_by_product.pivot(index='date', columns='product', values='num_sold')

# Plot the data
plt.figure(figsize=(12, 6))
for product in sales_pivot.columns:
    plt.plot(sales_pivot.index, sales_pivot[product], label=product)

# Customize the plot
plt.title('Total Sales Over Time by Product')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend(title='Product', loc='upper left', fontsize='small')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()



sales_by_country = train.groupby(['date', 'country'])['num_sold'].sum().reset_index()

# Pivot the data to have products as columns and dates as rows
sales_pivot = sales_by_country.pivot(index='date', columns='country', values='num_sold')

# Plot the data
plt.figure(figsize=(12, 6))
for country in sales_pivot.columns:
    plt.plot(sales_pivot.index, sales_pivot[country], label=country)

# Customize the plot
plt.title('Total Sales Over Time by Country')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend(title='Country', loc='upper left', fontsize='small')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()



sales_by_store = train.groupby(['date', 'store'])['num_sold'].sum().reset_index()

# Pivot the data to have products as columns and dates as rows
sales_pivot = sales_by_store.pivot(index='date', columns='store', values='num_sold')

# Plot the data
plt.figure(figsize=(12, 6))
for store in sales_pivot.columns:
    plt.plot(sales_pivot.index, sales_pivot[store], label=store)

# Customize the plot
plt.title('Total Sales Over Time by store')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend(title='store', loc='upper left', fontsize='small')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()


# agg_train = train.groupby("date", as_index = False)['num_sold'].sum()

# agg_train.head()


# agg_train["date"] = pd.to_datetime(agg_train["date"])  # Ensure 'date' is in datetime format
# agg_train = agg_train.set_index("date").to_period(freq="D")  # Replace "D" with appropriate frequency

# moving_average = agg_train.rolling(
#     window=365,       # 365-day window
#     center=True,      # puts the average at the center of the window
#     min_periods=183,  # choose about half the window size
# ).mean()              # compute the mean (could also do median, std, min, max, ...)

# ax = agg_train.plot(style=".", color="0.5")
# moving_average.plot(
#     ax=ax, linewidth=3, title="Sticker Sales - 365-Days Moving Average", legend=False,
# );


# moving_average = agg_train.rolling(
#     window=120,       # 365-day window
#     center=True,      # puts the average at the center of the window
#     min_periods=60,  # choose about half the window size
# ).mean()              # compute the mean (could also do median, std, min, max, ...)

# ax = agg_train.plot(style=".", color="0.5")
# moving_average.plot(
#     ax=ax, linewidth=3, title="Sticker Sales - 120-Days Moving Average", legend=False,
# );


# moving_average = agg_train.rolling(
#     window=30,       # 365-day window
#     center=True,      # puts the average at the center of the window
#     min_periods=15,  # choose about half the window size
# ).mean()              # compute the mean (could also do median, std, min, max, ...)

# ax = agg_train.plot(style=".", color="0.5")
# moving_average.plot(
#     ax=ax, linewidth=3, title="Sticker Sales - 30-Day Moving Average", legend=False,
# );


# train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
# train['date'] = pd.to_datetime(train['date'])


# train['country'].value_counts()


# train['store'].value_counts()


train[['country','store']].value_counts()


# Filter the data for Canada
canada_data = train[train['country'] == 'Canada']

# Add a new column to indicate whether `num_sold` is filled or missing
canada_data['status'] = canada_data['num_sold'].notnull().replace({True: 'Filled', False: 'Missing'})

# Get the unique stores in Canada
stores = canada_data['store'].unique()

# Create a 1x3 grid for plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, constrained_layout=True)

# Loop through each store and plot the data
for idx, store in enumerate(stores):
    # Filter data for the current store
    store_data = canada_data[canada_data['store'] == store]
    
    # Group data by date and status
    store_grouped = store_data.groupby(['date', 'status'])['num_sold'].sum().reset_index()

    # Pivot to prepare for plotting
    pivot_table = store_grouped.pivot(index='date', columns='status', values='num_sold')
    
    # Plot on the respective subplot
    ax = axes[idx]
    for status in pivot_table.columns:
        ax.plot(pivot_table.index, pivot_table[status], marker='o', label=status)
    
    # Set titles and labels
    ax.set_title(f'Sales for Store: {store}')
    ax.set_xlabel('Date')
    if idx == 0:  # Only add ylabel to the first subplot
        ax.set_ylabel('Number of Products Sold')
    ax.legend(title='Status', loc='upper left', fontsize='small')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add a main title
fig.suptitle('Canada Sales by Store with Filled or Missing num_sold', fontsize=16)

# Show the plot
plt.show()



train['product'].value_counts()


# Filter the data for Canada
canada_premium_store = train[(train['country'] == 'Canada') & (train['store'] == 'Premium Sticker Mart')]

# Add a new column to indicate whether `num_sold` is filled or missing
canada_premium_store['status'] = canada_premium_store['num_sold'].notnull().replace({True: 'Filled', False: 'Missing'})

# Get the unique products in Canada
products = canada_premium_store['product'].unique()

# Create a 1x3 grid for plotting
fig, axes = plt.subplots(1,5, figsize=(18, 6), sharey=True, constrained_layout=True)

# Loop through each store and plot the data
for idx, product in enumerate(products):
    # Filter data for the current store
    product_data = canada_premium_store[canada_premium_store['product'] == product]
    
    # Group data by date and status
    product_grouped = product_data.groupby(['date', 'status'])['num_sold'].sum().reset_index()

    # Pivot to prepare for plotting
    pivot_table = product_grouped.pivot(index='date', columns='status', values='num_sold')
    
    # Plot on the respective subplot
    ax = axes[idx]
    for status in pivot_table.columns:
        ax.plot(pivot_table.index, pivot_table[status], marker='o', label=status)
    
    # Set titles and labels
    ax.set_title(f'Sales for Store: {product}')
    ax.set_xlabel('Date')
    if idx == 0:  # Only add ylabel to the first subplot
        ax.set_ylabel('Number of Products Sold')
    ax.legend(title='Status', loc='upper left', fontsize='small')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add a main title
fig.suptitle('Canada Sales by Products in Premium Sticker Mart Store with Filled or Missing num_sold', fontsize=16)

# Show the plot
plt.show()


filtered_data = train[
    (train['country'] == 'Canada') & 
    (train['store'] == 'Premium Sticker Mart') & 
    (train['product'] == 'Holographic Goose')
]

filtered_data['status'] = filtered_data['num_sold'].notnull().replace({True: 'Filled', False: 'Missing'})

# Extract the month from the date column
filtered_data['Month'] = filtered_data['date'].dt.month

# Group data by Month and Status, summing num_sold
grouped_data = filtered_data.groupby(['Month','status'])['id'].nunique().reset_index()

# Pivot the data for plotting
pivot_table = grouped_data.pivot(index='Month', columns='status', values='id')

# Plot the data
plt.figure(figsize=(10, 6))
for status in pivot_table.columns:
    plt.plot(pivot_table.index, pivot_table[status], marker='o', label=status)

# Customize the plot
plt.title('Monthly Sales for Holographic Goose (Canada, Premium Sticker Mart)')
plt.xlabel('Month')
plt.ylabel('Total Number Sold')
plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(title='Status')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()



# Filter the data for Kenya
kenya_data = train[train['country'] == 'Kenya']

# Add a new column to indicate whether `num_sold` is filled or missing
kenya_data['status'] = kenya_data['num_sold'].notnull().replace({True: 'Filled', False: 'Missing'})

# Get the unique stores in Kenya
stores = kenya_data['store'].unique()

# Create a 1x3 grid for plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, constrained_layout=True)

# Loop through each store and plot the data
for idx, store in enumerate(stores):
    # Filter data for the current store
    store_data = kenya_data[kenya_data['store'] == store]
    
    # Group data by date and status
    store_grouped = store_data.groupby(['date', 'status'])['num_sold'].sum().reset_index()

    # Pivot to prepare for plotting
    pivot_table = store_grouped.pivot(index='date', columns='status', values='num_sold')
    
    # Plot on the respective subplot
    ax = axes[idx]
    for status in pivot_table.columns:
        ax.plot(pivot_table.index, pivot_table[status], marker='o', label=status)
    
    # Set titles and labels
    ax.set_title(f'Sales for Store: {store}')
    ax.set_xlabel('Date')
    if idx == 0:  # Only add ylabel to the first subplot
        ax.set_ylabel('Number of Products Sold')
    ax.legend(title='Status', loc='upper left', fontsize='small')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add a main title
fig.suptitle('Kenya Sales by Store with Filled or Missing num_sold', fontsize=16)

# Show the plot
plt.show()



train.shape


kenya_size = train[train['country'] == 'Kenya'].shape
canada_size = train[train['country'] == 'Canada'].shape

kenya_size, canada_size


kenya_missing = train['num_sold'][train['country'] == 'Kenya'].isna().sum()
canada_missing = train['num_sold'][train['country'] == 'Canada'].isna().sum()

kenya_missing, canada_missing


kenya_missing = train['num_sold'][(train['country'] == 'Kenya') & (train['store'] == 'Discount Stickers')].isna().sum()
canada_missing = train['num_sold'][(train['country'] == 'Canada') & (train['store'] == 'Discount Stickers')].isna().sum()

kenya_missing, canada_missing


def feature_engineering(df):
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_name'] = df['date'].dt.day_name()   # Full name of the day
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int) # True for Saturday/Sunday
    df['is_month_start'] = df['date'].dt.is_month_start
    df['is_month_end'] = df['date'].dt.is_month_end
    df['is_year_start'] = df['date'].dt.is_year_start
    df['is_year_end'] = df['date'].dt.is_year_end
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['days_since_year_start'] = df['date'] - pd.to_datetime(df['Year'].astype(str) + '-01-01')
    df['days_since_year_start'] = df['days_since_year_start'].dt.days
    df['days_until_weekend'] = (5 - df['day_of_week']).clip(lower=0)
    df['is_midweek'] = df['day_of_week'].isin([1, 2, 3]).astype(int)
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['days_since_year_start'] = df['date'] - pd.to_datetime(df['Year'].astype(str) + '-01-01')
    df['days_since_year_start'] = df['days_since_year_start'].dt.days


feature_engineering(train), feature_engineering(test)

train.head()


# Define a function to assign seasons
def assign_season(row):
    if row['country'] in ['Singapore', 'Kenya']:
        return 'Summer all-year'
    else:
        month = row['Month']  
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'


# Create the 'season' column
train['season'] = train.apply(assign_season, axis=1)
test['season'] = test.apply(assign_season, axis=1)

# Display the first few rows to check the new feature
print(train[['date', 'country', 'Month', 'season']].head())



import holidays
# Define a function to check if a date is a public holiday based on the country
def is_public_holiday(date, country):
    try:
        # Initialize the appropriate country's holiday list
        if country == 'Canada':
            country_holidays = holidays.Canada()
        elif country == 'Finland':
            country_holidays = holidays.Finland()
        elif country == 'Italy':
            country_holidays = holidays.Italy()
        elif country == 'Kenya':
            country_holidays = holidays.Kenya()
        elif country == 'Norway':
            country_holidays = holidays.Norway()
        elif country == 'Singapore':
            country_holidays = holidays.Singapore()
        else:
            return 0  # If country is not in the list, return 0
        
        # Return 1 if the date is a holiday, else return 0
        return 1 if date in country_holidays else 0
    except Exception as e:
        print(f"Error processing country {country}: {e}")
        return 0


# Apply the function to add the 'is_holiday' column
train['is_holiday'] = train.apply(lambda row: is_public_holiday(row['date'], row['country']), axis=1)
test['is_holiday'] = test.apply(lambda row: is_public_holiday(row['date'], row['country']), axis=1)
print(train['is_holiday'].value_counts())


def create_grouped_lag_features(df, group_cols, target_col, lags):
    """
    Create lag features for a specified column grouped by other columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a datetime index.
    group_cols (list): List of columns to group by (e.g., ['country', 'store', 'product']).
    target_col (str): The name of the column for which to create lag features.
    lags (list): A list of lag periods (e.g., [1, 30, 60, 90, 120, 365]).

    Returns:
    pd.DataFrame: DataFrame with lag features added.
    """
    for lag in lags:
        df[f'lag_{lag}'] = (
            df.groupby(group_cols)[target_col]
            .shift(lag)
        )
    return df


# Ensure 'date' column is in datetime format and set as index
train['date'] = pd.to_datetime(train['date'])

# Sort the data by group columns and date to ensure proper lagging
train = train.sort_values(['country', 'store', 'product', 'date'])

# List of lag periods
lag_periods = [1, 30, 60, 90, 120, 365]

# Create lag features for 'num_sold' grouped by 'country', 'store', and 'product'
train_with_lags = create_grouped_lag_features(
    train, 
    group_cols=['country', 'store', 'product'], 
    target_col='num_sold', 
    lags=lag_periods
)

# Display the first few rows with lag features
train_with_lags.iloc[20:40,]


train_with_lags.iloc[5210:5224,]


train.columns


test.columns


def create_grouped_lag_features_for_test(train, test, group_cols, target_col, lags):
    """
    Create lag features for the test dataset using train dataset values.

    Parameters:
    train (pd.DataFrame): The training dataset containing the target column.
    test (pd.DataFrame): The test dataset (without the target column).
    group_cols (list): The columns to group by (e.g., ['country', 'store', 'product']).
    target_col (str): The name of the column for which to create lag features (e.g., 'num_sold').
    lags (list): A list of lag periods (e.g., [1, 30, 60, 90, 120, 365]).

    Returns:
    pd.DataFrame: Test dataset with lag features added.
    """
    # Concatenate train and test datasets
    test[target_col] = None  # Add a placeholder for the target column in the test set
    combined = pd.concat([train, test], axis=0, sort=False)

    # Sort combined data by group columns and date
    combined = combined.sort_values(group_cols + ['date'])

    # Create lag features
    for lag in lags:
        combined[f'lag_{lag}'] = (
            combined.groupby(group_cols)[target_col]
            .shift(lag)
        )
    
    # Extract the test dataset with lag features
    test_with_lags = combined[combined['date'] >= test['date'].min()].drop(columns=[target_col])
    
    return test_with_lags

# Parameters for lag feature generation
group_columns = ['country', 'store', 'product']
lag_periods = [1, 30, 60, 90, 120, 365]

# Ensure the date column is in datetime format
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

# Generate lag features for the test dataset
test = create_grouped_lag_features_for_test(
    train=train,
    test=test,
    group_cols=group_columns,
    target_col='num_sold',
    lags=lag_periods
)

# Display the first few rows of the test dataset with lag features
print(test.head())



train[(train['country'] == 'Canada') & (train['date'] == '2016-11-30') & (train['store'] == 'Discount Stickers') & (train['product'] == 'Holographic Goose')]


test[(test['country'] == 'Norway')].head()


# Define the legend types you want to loop over
legend_types = ['store', 'country', 'product']

# Loop through each legend type
for legend in legend_types:
    # Aggregate the data by Year and the current legend type
    yearly_sales = train.groupby(['Year', legend])['num_sold'].sum().reset_index()

    # Pivot the data to get the current legend type as columns and Years as rows
    pivot_table = yearly_sales.pivot(index='Year', columns=legend, values='num_sold')

    # Plot the data
    plt.figure(figsize=(8, 4))
    for col in pivot_table.columns:
        plt.plot(pivot_table.index, pivot_table[col], marker='o', label=col)

    # Add labels, legend, and title
    plt.title(f'Yearly Sales with {legend.capitalize()} as Legend')
    plt.xlabel('Year')
    plt.ylabel('Number of Products Sold')
    plt.legend(title=legend.capitalize(), loc='upper left', fontsize='small')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()



# Get the list of unique countries in the dataset
unique_countries = train['country'].unique()

# Create a 2x3 grid for plotting
fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Loop through each country and generate a chart
for idx, country in enumerate(unique_countries):
    # Filter the data for the current country
    country_data = train[train['country'] == country]
    
    # Aggregate the data by Year and Store for the current country
    yearly_sales_country = country_data.groupby(['Year', 'store'])['num_sold'].sum().reset_index()

    # Pivot the data to get Stores as columns and Years as rows
    pivot_table = yearly_sales_country.pivot(index='Year', columns='store', values='num_sold')

    # Plot on the respective subplot
    ax = axes[idx]
    for store in pivot_table.columns:
        ax.plot(pivot_table.index, pivot_table[store], marker='o', label=store)

    # Add labels, legend, and title for each subplot
    ax.set_title(f'{country}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Products Sold')
    ax.legend(title='Store', loc='upper left', fontsize='small')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Remove any empty subplots if there are less than 6 countries
for i in range(len(unique_countries), len(axes)):
    fig.delaxes(axes[i])

# Set a main title for the entire figure
fig.suptitle('Yearly Sales with Store as Legend by Country', fontsize=16)

# Show the plot
plt.show()



# Define the legend types you want to loop over
legend_types = ['Year', 'store', 'country', 'product']

# Loop through each legend type
for legend in legend_types:
    # Aggregate the data by Year and the current legend type
    monthly_sales = train.groupby(['Month', legend])['num_sold'].sum().reset_index()

    # Pivot the data to get the current legend type as columns and Years as rows
    pivot_table = monthly_sales.pivot(index='Month', columns=legend, values='num_sold')

    # Plot the data
    plt.figure(figsize=(12, 5))
    for col in pivot_table.columns:
        plt.plot(pivot_table.index, pivot_table[col], marker='o', label=col)

    # Add labels, legend, and title
    plt.title(f'Monthly Sales with {legend.capitalize()} as Legend')
    plt.xlabel('Month')
    plt.ylabel('Number of Products Sold')
    plt.legend(title=legend.capitalize(), loc='upper left', fontsize='small')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()



import math

# Define the list of features
features = [
    'country', 'store', 'product', 'Year', 'Month', 'Day', 'day_of_week', 'day_name', 
    'quarter', 'is_weekend', 'is_month_start', 'is_month_end', 'is_year_start', 
    'is_year_end', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos', 
    'days_since_year_start', 'days_until_weekend', 'is_midweek', 'day_of_year', 
    'day_of_year_sin', 'day_of_year_cos', 'season'
]

# Number of rows and columns for subplots
cols = 3
rows = math.ceil(len(features) / cols)

# Create a figure with subplots
fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))
axes = axes.flatten()  # Flatten the axes for easy iteration

# Loop through each feature and plot
for i, feature in enumerate(features):
    ax = axes[i]
    if train[feature].dtype in ['object', 'category']:  # Categorical features
        sns.boxplot(x=train[feature], y=train['num_sold'], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    else:  # Numerical features
        sns.scatterplot(x=train[feature], y=train['num_sold'], ax=ax)
    
    ax.set_title(f'Relationship between {feature} and num_sold')
    ax.set_xlabel(feature)
    ax.set_ylabel('num_sold')

# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()



train[train['num_sold'] > 3000].shape


# List of columns to check
columns_to_check = ['country', 'store', 'product', 'Year', 'Month', 'Day', 'day_of_week', 
                    'day_name', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end', 
    'is_year_start', 'is_year_end', 'month_sin', 'month_cos', 
    'day_of_week_sin', 'day_of_week_cos', 'days_since_year_start', 
    'days_until_weekend', 'is_midweek', 'day_of_year', 'day_of_year_sin', 
    'day_of_year_cos', 'season'
]

# Filter rows where num_sold > 3000
filtered_data = train[train['num_sold'] > 3000]

# Loop through each column and calculate value_counts
for column in columns_to_check:
    print(f"Value Counts for '{column}' when num_sold > 3000:")
    print(filtered_data[column].value_counts())
    print("-" * 50) 



train[train['num_sold'] > 3000]['country'].value_counts()


train[train['num_sold'] > 3000]['store'].value_counts()


train['is_midweek'].value_counts()


# Define the list of lag columns to check for null values
lag_columns = ['lag_1','lag_30', 'lag_60', 'lag_90', 'lag_120', 'lag_365']

# Drop rows where any of the specified lag columns have NaN
train = train.dropna(subset=lag_columns)
test = test.fillna(0)


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# Define numerical and categorical features
numerical_features = ['Month', 'Day', 'day_of_week', 'quarter',  'is_weekend', 'is_month_start', 'is_month_end',
                      'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos', 'days_since_year_start', 'days_until_weekend', 
                      'is_midweek','day_of_year', 'day_of_year_sin', 'day_of_year_cos','is_holiday','lag_30', 'lag_60', 'lag_90', 'lag_120', 'lag_365']

categorical_features = ['country', 'store', 'product',  'season']

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Normalize numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Encode categorical features
    ]
)

# Apply the preprocessor 
X = train[numerical_features + categorical_features]  # Features to normalize & encode
X_normalized = preprocessor.fit_transform(X)

# Get feature names from preprocessor
feature_names = (
    numerical_features + 
    list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
)
X_normalized_df = pd.DataFrame(X_normalized, columns=feature_names)

# Display the first few rows of the normalized dataset
print(X_normalized_df.head())



from sklearn.cluster import KMeans

# Perform K-Means Clustering
kmeans = KMeans(n_clusters=6, random_state=42)  # Example with 5 clusters
kmeans.fit(X_normalized)

# Add cluster labels to the original data
train['cluster'] = kmeans.labels_

# Display the first few rows with cluster labels
print(train[['country', 'store', 'product', 'cluster']].head())



train['cluster'].value_counts()


train['date'].max()


test['date'].min()


X_test = test[numerical_features + categorical_features]  # Features to normalize & encode
X_test_normalized = preprocessor.fit_transform(X_test)
kmeans.fit(X_test_normalized)
test['cluster'] = kmeans.labels_


# cluster_summary = train.groupby('cluster').mean()
# print(cluster_summary)


from sklearn.decomposition import PCA

# Reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)

# Plot clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=train['cluster'], cmap='viridis', s=50)
plt.colorbar(label='Cluster')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clusters')
plt.show()


# # Drop rows where 'num_sold' is missing
# cleaned_data = train[train['num_sold'].notnull()]

# # Check the shape of the cleaned dataset
# print(f"Original dataset shape: {train.shape}")
# print(f"Cleaned dataset shape: {cleaned_data.shape}")



# Replace missing values in 'num_sold' with 0
cleaned_data = train.copy()  # Create a copy of the original DataFrame
cleaned_data['num_sold'] = cleaned_data['num_sold'].fillna(0)


cleaned_data.columns


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


categorical_features = ['country', 'store', 'product', 'day_name', 'season']
numerical_features = ['Year', 'Month', 'Day', 'day_of_week', 'days_since_year_start', 
                      'days_until_weekend', 'day_of_year', 'month_sin', 'month_cos', 
                      'day_of_week_sin', 'day_of_week_cos', 'is_holiday', 'lag_1', 'lag_30', 'lag_60', 'lag_90', 'lag_120', 'lag_365']

# Separate features and target
X = cleaned_data.drop(columns=['num_sold']) 
y = cleaned_data['num_sold']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# List of models
models = {
    "XGBRegressor": XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1),
    "LGBMRegressor": LGBMRegressor(random_state=42, n_estimators=100, learning_rate=0.1),
    "CatBoostRegressor": CatBoostRegressor(random_state=42, n_estimators=100, learning_rate=0.1, verbose=0)
}

# Initialise a result dictionary
results = {}

# Loop through each model and pipeline
for model_name, model in models.items():
    print(f"Training {model_name}...")
    
    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Store the results
    results[model_name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    
    # Print results for the current model
    print(f"{model_name} Results:")
    print(f" - MAE: {mae:.2f}")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - R2: {r2:.2f}")
    print("-" * 30)

# Compare results
results_df = pd.DataFrame(results).T
print("\nComparison of Model Performance:")
print(results_df)




# Extract feature importance for each model
for model_name, model in models.items():
    print(f"Feature Importance for {model_name}:")
    
    # Fit model with preprocessing pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    
    # Extract feature importance from the model
    importances = model.feature_importances_
    feature_names = pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)
    all_features = numerical_features + list(feature_names)
    
    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    # Plot feature importance as a bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance for {model_name}')
    plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
    plt.tight_layout()
    plt.show()



# # Extract feature importance for each model
# for model_name, model in models.items():
#     print(f"Feature Importance for {model_name}:")
#     # Fit model with preprocessing pipeline
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', model)
#     ])
#     pipeline.fit(X_train, y_train)
    
#     # Extract feature importance from the model
#     importances = model.feature_importances_
#     feature_names = pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)
#     all_features = numerical_features + list(feature_names)
#     importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
#     print(importance_df)



# Residuals for the best model (e.g., LGBMRegressor)
best_model = models['LGBMRegressor']
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', best_model)
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for LGBM Regressor')
plt.show()

# Check residual distribution
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, alpha=0.7, color='blue')
plt.axvline(0, color='red', linestyle='--')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution for LGBM Regressor')
plt.show()



from sklearn.model_selection import learning_curve

# Learning curve for XGBRegressor
train_sizes, train_scores, test_scores = learning_curve(
    Pipeline(steps=[('preprocessor', preprocessor), ('regressor', XGBRegressor(random_state=42))]),
    X_train, y_train, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
)

# Calculate mean and standard deviation
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', label='Training Error')
plt.plot(train_sizes, test_scores_mean, 'o-', label='Validation Error')
plt.xlabel('Training Set Size')
plt.ylabel('RMSE')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.show()


# Align test dataframe with X_train 
test = test[X_train.columns]

# Choose the best model (e.g., LGBMRegressor based on performance)
best_model_name = "LGBMRegressor"
best_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', models[best_model_name])
])

# Train the pipeline on the full training data
best_model_pipeline.fit(X_train, y_train)

# Transform and predict on the test dataframe
test_transformed = best_model_pipeline.named_steps['preprocessor'].transform(test)
test_predictions = best_model_pipeline.predict(test)

# Add predictions to the test dataframe
test['num_sold'] = test_predictions

# Display the test dataframe with predictions
print(test.head())


# agg_test = test.groupby("date", as_index = False)['num_sold'].sum()

# agg_test["date"] = pd.to_datetime(agg_test["date"])  # Ensure 'date' is in datetime format
# agg_test = agg_test.set_index("date").to_period(freq="D")  # Replace "D" with appropriate frequency

# moving_average = agg_test.rolling(
#     window=365,       # 365-day window
#     center=True,      # puts the average at the center of the window
#     min_periods=183,  # choose about half the window size
# ).mean()              # compute the mean (could also do median, std, min, max, ...)

# ax = agg_test.plot(style=".", color="0.5")
# moving_average.plot(
#     ax=ax, linewidth=3, title="Sticker Sales - 365-Days Moving Average", legend=False,
# );


# read_sample = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
# read_sample.head()


wanted_col = ['id', 'num_sold']
test = test[wanted_col]


# Optionally, save the predictions to a CSV file
test.to_csv('test_with_predictions.csv', index=False)
















