import pandas as pd 
import matplotlib.pyplot as plt
import warnings
import math
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")


# Check for missing values
missing_values = train.isnull().sum()
print("Missing values:\n", missing_values)

# Percentage of missing values
missing_percentage = (missing_values / len(train)) * 100
print("Percentage of missing values:\n", missing_percentage)



# Fill missing values in 'num_sold' with 0 (or use df['num_sold'].fillna(df['num_sold'].mean(), inplace=True))
train['num_sold'].fillna(0, inplace=True)

# Verify missing values are handled
print(train.isnull().sum())



# Check for duplicates
duplicates = train.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Drop duplicates
train.drop_duplicates(inplace=True)



# Convert 'date' column to datetime format
train['date'] = pd.to_datetime(train['date'])

# Verify the data type
print(train.info())



# Summary for numerical columns
print(train.describe())

# Summary for categorical columns
print(train[['country', 'store', 'product']].describe())



# Generate a complete date range
date_range = pd.date_range(start=train['date'].min(), end=train['date'].max())

# Check missing dates
missing_dates = set(date_range) - set(train['date'])
print("Missing dates in the time series:", missing_dates)





# Aggregate sales by date
sales_trend = train.groupby('date')['num_sold'].sum()

# Plot sales trend
plt.figure(figsize=(12, 6))
sales_trend.plot(title='Total Sales Over Time', color='blue')
plt.xlabel('Date')
plt.ylabel('Number of Products Sold')
plt.show()





# Assuming 'train' is your DataFrame with 'date', 'country', and 'num_sold'

# Aggregate sales by date and country
sales_by_country = train.groupby(['date', 'country'])['num_sold'].sum().unstack()

# Set up the number of subplots (one for each country)
num_countries = len(sales_by_country.columns)
fig, axes = plt.subplots(num_countries, 1, figsize=(12, 5 * num_countries))

# Plot each country's sales trend in a separate subplot
for i, country in enumerate(sales_by_country.columns):
    sales_by_country[country].plot(ax=axes[i], title=f'Sales Trend for {country}', color='blue')
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel('Number of Products Sold')

# Adjust layout to make sure there's enough space between the plots
plt.tight_layout()
plt.show()





# Assuming 'train' is your DataFrame with 'date', 'country', 'product', and 'num_sold'

# Convert 'date' column to datetime (if it's not already)
train['date'] = pd.to_datetime(train['date'])

# Aggregate sales by date, country, and product
sales_by_country_product = train.groupby(['date', 'country', 'product'])['num_sold'].sum().unstack(fill_value=0)

# Set up the number of subplots (one for each country)
num_countries = len(sales_by_country_product.index.levels[1])  # Get number of countries
fig, axes = plt.subplots(num_countries, 1, figsize=(12, 5 * num_countries))

# Plot each country's sales trend for each product in a separate subplot
for i, country in enumerate(sales_by_country_product.index.levels[1]):
    # Select the sales data for the current country
    country_sales = sales_by_country_product.xs(country, level='country')

    # Plot each product's sales as a separate line
    for product in country_sales.columns:
        country_sales[product].plot(ax=axes[i], label=product)

    # Set title and labels for each subplot
    axes[i].set_title(f'Sales Trend for {country}')
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel('Number of Products Sold')
    axes[i].legend(title='Product')

# Adjust layout to make sure there's enough space between the plots
plt.tight_layout()
plt.show()





# Assuming 'train' is your DataFrame with 'date', 'country', 'store', and 'num_sold'

# Convert 'date' column to datetime (if it's not already)
train['date'] = pd.to_datetime(train['date'])

# Aggregate sales by date, country, and store
sales_by_country_store = train.groupby(['date', 'country', 'store'])['num_sold'].sum().unstack(fill_value=0)

# Set up the number of subplots (one for each country)
num_countries = len(sales_by_country_store.index.levels[1])  # Get number of countries
fig, axes = plt.subplots(num_countries, 1, figsize=(12, 5 * num_countries))

# Plot each country's sales trend for each store in a separate subplot
for i, country in enumerate(sales_by_country_store.index.levels[1]):
    # Select the sales data for the current country
    country_sales = sales_by_country_store.xs(country, level='country')

    # Plot each store's sales as a separate line
    for store in country_sales.columns:
        country_sales[store].plot(ax=axes[i], label=store)

    # Set title and labels for each subplot
    axes[i].set_title(f'Sales Trend for {country}')
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel('Number of Products Sold')
    axes[i].legend(title='Store')

# Adjust layout to make sure there's enough space between the plots
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import pandas as pd


# Suppress all warnings
warnings.filterwarnings("ignore")

# Assuming 'train' is your DataFrame with 'date', 'country', 'store', 'product', and 'num_sold'

# Convert 'date' column to datetime (if it's not already)
#train['date'] = pd.to_datetime(train['date'])

# Filter data for Canada
canada_sales = train[train['country'] == 'Canada']

# Aggregate sales by date, store, and product
sales_by_store_product = canada_sales.groupby(['date', 'store', 'product'])['num_sold'].sum().reset_index()

# Get the number of stores
num_stores = sales_by_store_product['store'].nunique()

# Calculate the number of rows and columns for 3 rows of subplots
num_rows = 3
num_columns = math.ceil(num_stores / num_rows)

# Set up the plot with subplots
fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))

# Flatten axes for easier indexing (in case of more than one row)
axes = axes.flatten()

# Loop through each store and plot the sales trend
for i, (store, data) in enumerate(sales_by_store_product.groupby('store')):
    ax = axes[i]
    for product in data['product'].unique():
        product_data = data[data['product'] == product]
        ax.plot(product_data['date'], product_data['num_sold'], label=product, marker='o')
    
    ax.set_title(f'Sales Trend for {store}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Products Sold')
    ax.legend(title='Product')
    
# Adjust layout to make room for all subplots
plt.tight_layout()
plt.show()



# Suppress all warnings
warnings.filterwarnings("ignore")

# Assuming 'train' is your DataFrame with 'date', 'country', 'store', 'product', and 'num_sold'

# Convert 'date' column to datetime (if it's not already)
# train['date'] = pd.to_datetime(train['date'])

# List of countries to plot
countries = ['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore']

# Loop through each country
for idx, country in enumerate(countries):
    # Filter data for the current country
    country_sales = train[train['country'] == country]

    # Aggregate sales by date, store, and product
    sales_by_store_product = country_sales.groupby(['date', 'store', 'product'])['num_sold'].sum().reset_index()

    # Get the number of stores
    num_stores = sales_by_store_product['store'].nunique()

    # Calculate the number of columns for the subplots
    num_columns = math.ceil(num_stores / 3)

    # Set up the subplots for each country
    fig, axes_country = plt.subplots(3, num_columns, figsize=(15, 5 * 3))

    # Flatten axes for easier indexing
    axes_country = axes_country.flatten()

    # Loop through each store and plot the sales trend for the current country
    for i, (store, data) in enumerate(sales_by_store_product.groupby('store')):
        ax = axes_country[i]
        for product in data['product'].unique():
            product_data = data[data['product'] == product]
            ax.plot(product_data['date'], product_data['num_sold'], label=product, marker='o')

        ax.set_title(f'Sales Trend for {store}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Products Sold')
        ax.legend(title='Product')

    # Add title for the country figure (bold)
    fig.suptitle(f'Sales Trends for {country}', fontsize=16, fontweight='bold')

    # Adjust layout to make room for all subplots
    plt.tight_layout()

    # Add some space between the subplots and the country title
    plt.subplots_adjust(top=0.9)

    # Show plots for the current country
    plt.show()


