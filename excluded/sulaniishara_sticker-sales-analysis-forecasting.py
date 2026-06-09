# Importing necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy.signal import find_peaks
import requests
import plotly.express as px
import plotly.io as pio

import warnings
warnings.filterwarnings("ignore")


# Load the datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv", parse_dates=["date"])
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv", parse_dates=["date"])
# sample_df = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")

# Verify shapes
print("Train Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)


# Display sample data
print("Training Dataset: \n")
display(train_df.head(10))
print('\n')
print("Test Dataset: \n")
display(test_df.head(10))


# Display information for the training dataset
print("Training Dataset Information: \n")
train_info = train_df.info()
display(train_info)
print('\n')
# Display information for the test dataset
print("Test Dataset Information: \n")
test_info = test_df.info()
display(test_info)


# Display the number of unique values in each column for train_df
print("\nUnique Values in Each Column (Train Data):")
print(train_df.nunique())

# Display the lists of numerical and categorical columns in train_df
non_numerical_columns_train = train_df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical Columns (Train Data):", non_numerical_columns_train)

# Display unique values for each categorical column in train_df
for col in non_numerical_columns_train:
    print(f"\nColumn: {col}")
    print(f"Unique Values: {train_df[col].unique()}")

# Display the number of unique values in each column for test_df
print("\nUnique Values in Each Column (Test Data):")
print(test_df.nunique())

# Display the lists of numerical and categorical columns in test_df
non_numerical_columns_test = test_df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical Columns (Test Data):", non_numerical_columns_test)

# Display unique values for each categorical column in test_df
for col in non_numerical_columns_test:
    print(f"\nColumn: {col}")
    print(f"Unique Values: {test_df[col].unique()}")


# Function to fetch GDP per capita for a given country and year
def get_gdp_per_capita(alpha3, year):
    """
    Fetch GDP per capita for a specific country and year from the World Bank API.
    
    """
    url = f'https://api.worldbank.org/v2/country/{alpha3}/indicator/NY.GDP.PCAP.CD?date={year}&format=json'
    try:
        response = requests.get(url)
        response.raise_for_status()  
        data = response.json()
        return data[1][0]['value'] if data[1] else None  
    except (requests.RequestException, KeyError, IndexError) as e:
        print(f"Error fetching data for {alpha3} in {year}: {e}")
        return None



# Function to create a DataFrame of GDP ratios
def create_gdp_dataframe(alpha3s, years, country_names):
    """
    Create a DataFrame of normalized GDP per capita ratios for multiple countries and years.
    
    """
    # Fetch GDP data for all countries and years
    gdp_data = [
        [get_gdp_per_capita(alpha3, year) for year in years]
        for alpha3 in alpha3s
    ]
    
    # Create a DataFrame with countries as rows and years as columns
    gdp_df = pd.DataFrame(gdp_data, index=country_names, columns=years)
    
    # Normalize GDP values by dividing by the column sum (yearly total)
    gdp_df = gdp_df / gdp_df.sum(axis=0)
    
    # Reshape the DataFrame into long format
    gdp_df = gdp_df.reset_index().rename(columns={'index': 'country'})
    gdp_df = gdp_df.melt(id_vars=['country'], var_name='year', value_name='ratio')
    
    return gdp_df



# Function to adjust ratios for specific countries
def adjust_ratios(gdp_df, adjustments):
    """
    Adjust GDP ratios for specific countries based on custom rules.
    
    """
    adjusted_df = gdp_df.copy()
    for country, adjustment in adjustments.items():
        adjusted_df.loc[adjusted_df['country'] == country, 'ratio'] -= adjustment
    return adjusted_df

if __name__ == "__main__":
    # Define input parameters
    alpha3s = ['CAN', 'FIN', 'ITA', 'KEN', 'NOR', 'SGP']
    years = range(2010, 2020)
    country_names = np.sort(['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore'])  # Example list
    
    # Create the GDP DataFrame
    gdp_ratios_df = create_gdp_dataframe(alpha3s, years, country_names)
    
    # Adjust Kenya's ratio by subtracting 0.0007
    adjustments = {'Kenya': 0.0007}
    gdp_per_capita_filtered_ratios_df = adjust_ratios(gdp_ratios_df, adjustments)
    
    print(gdp_per_capita_filtered_ratios_df.head(6))



pio.renderers.default = 'iframe'

# Filter for a specific year, e.g., 2010
gdp_2010 = gdp_per_capita_filtered_ratios_df[gdp_per_capita_filtered_ratios_df['year'] == 2010]

# Plot choropleth map
fig = px.choropleth(
    gdp_2010,
    locations='country',
    locationmode='country names',
    color='ratio',
    hover_name='country',
    title='Normalized GDP Per Capita Ratios (2010)',
    color_continuous_scale=px.colors.sequential.Plasma
)
fig.show()



# Visualize missing values with a heatmap
plt.figure(figsize=(10, 4))
sns.heatmap(train_df.isnull(), cbar=False, cmap="plasma")
plt.title("Heatmap of Missing Values in Training Dataset")
plt.show()


# Count of missing and non-missing values
missing_count = train_df['num_sold'].isnull().sum()
non_missing_count = len(train_df) - missing_count

# Define colors using the Plasma color palette
colors = [plt.cm.plasma(0.1), plt.cm.plasma(0.7)]

# Create subplots: one for the bar chart and one for the donut chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.bar(['Non-missing', 'Missing'], [non_missing_count, missing_count], color=colors)
ax1.set_title('Distribution of Missing Values in num_sold')
ax1.set_ylabel('Count')
ax1.grid(True, linestyle='--', alpha=0.6)

for i, count in enumerate([non_missing_count, missing_count]):
    ax1.text(i, count + 0.5, str(count), ha='center', va='bottom')

labels = ['Non-missing', 'Missing']
sizes = [non_missing_count, missing_count]
ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90,
        wedgeprops=dict(width=0.3))  
ax2.axis('equal')  
ax2.set_title('Proportion of Missing Values in num_sold')

plt.tight_layout()
plt.show()



# Create a new column to indicate missingness
train_df['num_sold_missing'] = train_df['num_sold'].isnull()

# Group by date and calculate the percentage of missing values
missing_by_date = train_df.groupby('date')['num_sold_missing'].mean()

plt.figure(figsize=(12, 4))
missing_by_date.plot(kind='line', color=plt.cm.plasma(0.2), alpha=0.7)
plt.title("Trend of Missing 'num_sold' Values Over Time")
plt.xlabel('Date')
plt.ylabel('Missing Value Percentage')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print(missing_by_date.describe())


# Count missing values by country
missing_by_country = train_df[train_df['num_sold_missing']]['country'].value_counts()

plt.figure(figsize=(12, 4))
missing_by_country.plot(kind='bar', color=plt.cm.plasma(0.8))
plt.title("Missing 'num_sold' Values by Country")
plt.xlabel("Country")
plt.ylabel("Count of Missing Values")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print(missing_by_country.describe())



# Count missing values by store
missing_by_store = train_df[train_df['num_sold_missing']]['store'].value_counts()

plt.figure(figsize=(12, 4))
missing_by_store.plot(kind='bar', color=plt.cm.plasma(0.4))
plt.title("Missing 'num_sold' Values by Store")
plt.xlabel("Store")
plt.ylabel("Count of Missing Values")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print(missing_by_store.describe())



# Count missing values by product
missing_by_product = train_df[train_df['num_sold_missing']]['product'].value_counts()

plt.figure(figsize=(12, 4))
missing_by_product.plot(kind='bar', color=plt.cm.plasma(0.5))
plt.title("Missing 'num_sold' Values by Product")
plt.xlabel("Product")
plt.ylabel("Count of Missing Values")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print(missing_by_product.describe())



# Count missing values grouped by both country and store
missing_by_country_store = (
    train_df[train_df['num_sold_missing']]
    .groupby(['country', 'store'])['num_sold']
    .size()
    .reset_index(name='missing_count')
)

plt.figure(figsize=(14, 6))

# Create a bar plot for missing values by store within each country
sns.barplot(
    data=missing_by_country_store,
    x='store',
    y='missing_count',
    hue='country',
    palette='plasma'
)

plt.title("Missing 'num_sold' Values by Store for Each Country", fontsize=16)
plt.xlabel("Store", fontsize=12)
plt.ylabel("Count of Missing Values", fontsize=12)
plt.xticks(rotation=45, fontsize=12)
plt.legend(title='Country', fontsize=12, title_fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# Count missing values grouped by both country and product
missing_by_country_product = (
    train_df[train_df['num_sold_missing']]
    .groupby(['country', 'product'])['num_sold']
    .size()
    .reset_index(name='missing_count')
)

plt.figure(figsize=(14, 6))

# Create a bar plot for missing values by product within each country
sns.barplot(
    data=missing_by_country_product,
    x='product',
    y='missing_count',
    hue='country',
    palette='plasma'
)

plt.title("Missing 'num_sold' Values by Product for Each Country", fontsize=16)
plt.xlabel("Product", fontsize=12)
plt.ylabel("Count of Missing Values", fontsize=12)
plt.xticks(rotation=45, fontsize=12)
plt.legend(title='Country', fontsize=12, title_fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# Count missing values grouped by both store and product
missing_by_store_product = (
    train_df[train_df['num_sold_missing']]
    .groupby(['store', 'product'])['num_sold']
    .size()
    .reset_index(name='missing_count')
)

plt.figure(figsize=(14, 6))

# Create a bar plot for missing values by product within each store
sns.barplot(
    data=missing_by_store_product,
    x='product',
    y='missing_count',
    hue='store',
    palette='plasma'
)

plt.title("Missing 'num_sold' Values by Product for Each Store", fontsize=16)
plt.xlabel("Product", fontsize=12)
plt.ylabel("Count of Missing Values", fontsize=12)
plt.xticks(rotation=45, fontsize=12)
plt.legend(title='Store', fontsize=12, title_fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# Dropping specified column from train_data
train_df = train_df.drop('num_sold_missing', axis=1)

# Display the updated DataFrame to confirm the columns have been dropped
print("Updated train_data after dropping specified columns:")
print(train_df.head())



# Create a copy of the DataFrame
train_df_imputed = train_df.copy()
print(f"Missing values remaining: {train_df_imputed['num_sold'].isna().sum()}")

# Extract the year from the date
train_df_imputed["year"] = train_df_imputed["date"].dt.year

# Loop through each year to perform imputation
for year in train_df_imputed["year"].unique():
    # Target ratio (Norway)
    target_ratio = gdp_per_capita_filtered_ratios_df.loc[
        (gdp_per_capita_filtered_ratios_df["year"] == year) & 
        (gdp_per_capita_filtered_ratios_df["country"] == "Norway"), "ratio"
    ].values[0]

    # Impute Time Series 1: Canada, Discount Stickers, Holographic Goose
    current_ratio_can = gdp_per_capita_filtered_ratios_df.loc[
        (gdp_per_capita_filtered_ratios_df["year"] == year) & 
        (gdp_per_capita_filtered_ratios_df["country"] == "Canada"), "ratio"
    ].values[0]
    ratio_can = current_ratio_can / target_ratio
    train_df_imputed.loc[
        (train_df_imputed["country"] == "Canada") & 
        (train_df_imputed["store"] == "Discount Stickers") & 
        (train_df_imputed["product"] == "Holographic Goose") & 
        (train_df_imputed["year"] == year), 
        "num_sold"
    ] = (
        train_df_imputed.loc[
            (train_df_imputed["country"] == "Norway") & 
            (train_df_imputed["store"] == "Discount Stickers") & 
            (train_df_imputed["product"] == "Holographic Goose") & 
            (train_df_imputed["year"] == year), 
            "num_sold"
        ] * ratio_can
    ).values

    # Impute Time Series 2-3: Canada, Premium Sticker Mart / Stickers for Less
    for store in ["Premium Sticker Mart", "Stickers for Less"]:
        current_ts = train_df_imputed.loc[
            (train_df_imputed["country"] == "Canada") & 
            (train_df_imputed["store"] == store) & 
            (train_df_imputed["product"] == "Holographic Goose") & 
            (train_df_imputed["year"] == year)
        ]
        missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
        train_df_imputed.loc[
            (train_df_imputed["country"] == "Canada") & 
            (train_df_imputed["store"] == store) & 
            (train_df_imputed["product"] == "Holographic Goose") & 
            (train_df_imputed["year"] == year) & 
            (train_df_imputed["date"].isin(missing_ts_dates)), 
            "num_sold"
        ] = (
            train_df_imputed.loc[
                (train_df_imputed["country"] == "Norway") & 
                (train_df_imputed["store"] == store) & 
                (train_df_imputed["product"] == "Holographic Goose") & 
                (train_df_imputed["year"] == year) & 
                (train_df_imputed["date"].isin(missing_ts_dates)), 
                "num_sold"
            ] * ratio_can
        ).values

    # Impute Time Series 4: Kenya, Discount Stickers, Holographic Goose
    current_ratio_ken = gdp_per_capita_filtered_ratios_df.loc[
        (gdp_per_capita_filtered_ratios_df["year"] == year) & 
        (gdp_per_capita_filtered_ratios_df["country"] == "Kenya"), "ratio"
    ].values[0]
    ratio_ken = current_ratio_ken / target_ratio
    train_df_imputed.loc[
        (train_df_imputed["country"] == "Kenya") & 
        (train_df_imputed["store"] == "Discount Stickers") & 
        (train_df_imputed["product"] == "Holographic Goose") & 
        (train_df_imputed["year"] == year), 
        "num_sold"
    ] = (
        train_df_imputed.loc[
            (train_df_imputed["country"] == "Norway") & 
            (train_df_imputed["store"] == "Discount Stickers") & 
            (train_df_imputed["product"] == "Holographic Goose") & 
            (train_df_imputed["year"] == year), 
            "num_sold"
        ] * ratio_ken
    ).values

    # Impute Time Series 5-6: Kenya, Premium Sticker Mart / Stickers for Less
    for store in ["Premium Sticker Mart", "Stickers for Less"]:
        current_ts = train_df_imputed.loc[
            (train_df_imputed["country"] == "Kenya") & 
            (train_df_imputed["store"] == store) & 
            (train_df_imputed["product"] == "Holographic Goose") & 
            (train_df_imputed["year"] == year)
        ]
        missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
        train_df_imputed.loc[
            (train_df_imputed["country"] == "Kenya") & 
            (train_df_imputed["store"] == store) & 
            (train_df_imputed["product"] == "Holographic Goose") & 
            (train_df_imputed["year"] == year) & 
            (train_df_imputed["date"].isin(missing_ts_dates)), 
            "num_sold"
        ] = (
            train_df_imputed.loc[
                (train_df_imputed["country"] == "Norway") & 
                (train_df_imputed["store"] == store) & 
                (train_df_imputed["product"] == "Holographic Goose") & 
                (train_df_imputed["year"] == year) & 
                (train_df_imputed["date"].isin(missing_ts_dates)), 
                "num_sold"
            ] * ratio_ken
        ).values

    # Impute Time Series 7: Kenya, Discount Stickers, Kerneler
    current_ts = train_df_imputed.loc[
        (train_df_imputed["country"] == "Kenya") & 
        (train_df_imputed["store"] == "Discount Stickers") & 
        (train_df_imputed["product"] == "Kerneler") & 
        (train_df_imputed["year"] == year)
    ]
    missing_ts_dates = current_ts.loc[current_ts["num_sold"].isna(), "date"]
    train_df_imputed.loc[
        (train_df_imputed["country"] == "Kenya") & 
        (train_df_imputed["store"] == "Discount Stickers") & 
        (train_df_imputed["product"] == "Kerneler") & 
        (train_df_imputed["year"] == year) & 
        (train_df_imputed["date"].isin(missing_ts_dates)), 
        "num_sold"
    ] = (
        train_df_imputed.loc[
            (train_df_imputed["country"] == "Norway") & 
            (train_df_imputed["store"] == "Discount Stickers") & 
            (train_df_imputed["product"] == "Kerneler") & 
            (train_df_imputed["year"] == year) & 
            (train_df_imputed["date"].isin(missing_ts_dates)), 
            "num_sold"
        ] * ratio_ken
    ).values

# Check for remaining missing values
print(f"Missing values remaining after imputation: {train_df_imputed['num_sold'].isna().sum()}")

# Manual imputation for specific IDs
train_df_imputed.loc[train_df_imputed["id"] == 23719, "num_sold"] = 4
train_df_imputed.loc[train_df_imputed["id"] == 207003, "num_sold"] = 195

# Final check for missing values
print(f"Final missing values remaining: {train_df_imputed['num_sold'].isna().sum()}")



# Check for duplicate rows in the training dataset
train_duplicates = train_df_imputed.duplicated().sum()
print(f"\nNumber of duplicate rows in the training dataset: {train_duplicates}")

# Check for duplicate rows in the test dataset
test_duplicates = test_df.duplicated().sum()
print(f"Number of duplicate rows in the test dataset: {test_duplicates}")



# Custom colormap using Plasma
plasma_cmap = cm.get_cmap("plasma")

def visualize_num_sold_with_peaks(data, feature='num_sold'):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    ax = sns.histplot(data[feature], bins=30, kde=True, color=plasma_cmap(0.5))
    plt.title(f'Histogram of {feature} with KDE', fontsize=12)
    plt.xlabel(feature, fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    kde = sns.kdeplot(data[feature], ax=ax, color=plasma_cmap(0.7)).lines[0].get_data()
    kde_x, kde_y = kde[0], kde[1]
    peaks, _ = find_peaks(kde_y)

    for peak_idx in peaks:
        plt.plot(kde_x[peak_idx], kde_y[peak_idx], "ro")  

    plt.subplot(1, 2, 2)
    sns.boxplot(x=data[feature], color=plasma_cmap(0.5))
    plt.title(f'Box Plot of {feature}', fontsize=12)
    plt.xlabel(feature, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

visualize_num_sold_with_peaks(train_df_imputed, feature='num_sold')


# Function to display bar plot and pie chart for categorical columns
def plot_categorical_distribution(data, column_name):
    plasma_colors = sns.color_palette("plasma", data[column_name].nunique())
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    sns.countplot(y=column_name, data=data, palette=plasma_colors)
    plt.title(f'Distribution of {column_name}', fontsize=12)
    plt.xlabel('Count', fontsize=10)
    plt.ylabel(column_name, fontsize=10)

    ax = plt.gca()
    for p in ax.patches:
        count = int(p.get_width())
        ax.annotate(f'{count}', 
                    (p.get_width() + 0.1, p.get_y() + p.get_height() / 2), 
                    ha='left', va='center', fontsize=10, color='black')
    
    sns.despine(left=True, bottom=True)
    
    # Pie chart for percentage distribution
    plt.subplot(1, 2, 2)
    data[column_name].value_counts().plot.pie(
        autopct='%1.1f%%', 
        colors=plasma_colors, 
        startangle=90, 
        explode=[0.05] * data[column_name].nunique(), 
        shadow=True
    )
    plt.title(f'Percentage Distribution of {column_name}', fontsize=12)
    plt.ylabel('')  

    plt.tight_layout()
    plt.show()

categorical_columns = ['country', 'store', 'product']
for column in categorical_columns:
    plot_categorical_distribution(train_df_imputed, column)


def facetgrid_and_boxplot(data, categorical_column, target_column):
    g = sns.FacetGrid(data, col=categorical_column, col_wrap=3, height=4, sharex=False, sharey=False, palette="plasma")
    g.map(sns.histplot, target_column, kde=False, bins=30, color=sns.color_palette("plasma")[0])
    g.set_titles("{col_name}")
    g.set_axis_labels(target_column, "Frequency")
    g.fig.suptitle(f"Distribution of {target_column} across unique values of {categorical_column}", y=1.02, fontsize=16)
    g.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    sns.boxplot(x=categorical_column, y=target_column, data=data, palette="plasma")
    plt.title(f"Boxplot of {target_column} by {categorical_column}", fontsize=12)
    plt.xlabel(categorical_column, fontsize=10)
    plt.ylabel(target_column, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

target_column = "num_sold"
categorical_column = "country"

facetgrid_and_boxplot(train_df_imputed, categorical_column, target_column)


def facetgrid_and_boxplot(data, categorical_column, target_column):
    g = sns.FacetGrid(data, col=categorical_column, col_wrap=3, height=4, sharex=False, sharey=False, palette="plasma")
    g.map(sns.histplot, target_column, kde=False, bins=30, color=sns.color_palette("plasma")[1])
    g.set_titles("{col_name}")
    g.set_axis_labels(target_column, "Frequency")
    g.fig.suptitle(f"Distribution of {target_column} across unique values of {categorical_column}", y=1.02, fontsize=16)
    g.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    sns.boxplot(x=categorical_column, y=target_column, data=data, palette="plasma")
    plt.title(f"Boxplot of {target_column} by {categorical_column}", fontsize=12)
    plt.xlabel(categorical_column, fontsize=10)
    plt.ylabel(target_column, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

target_column = "num_sold"
categorical_column = "store"

facetgrid_and_boxplot(train_df_imputed, categorical_column, target_column)


def facetgrid_and_boxplot(data, categorical_column, target_column):
    g = sns.FacetGrid(data, col=categorical_column, col_wrap=3, height=4, sharex=False, sharey=False, palette="plasma")
    g.map(sns.histplot, target_column, kde=False, bins=30, color=sns.color_palette("plasma")[2])
    g.set_titles("{col_name}")
    g.set_axis_labels(target_column, "Frequency")
    g.fig.suptitle(f"Distribution of {target_column} across unique values of {categorical_column}", y=1.02, fontsize=16)
    g.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    sns.boxplot(x=categorical_column, y=target_column, data=data, palette="plasma")
    plt.title(f"Boxplot of {target_column} by {categorical_column}", fontsize=12)
    plt.xlabel(categorical_column, fontsize=10)
    plt.ylabel(target_column, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

target_column = "num_sold"
categorical_column = "product"

facetgrid_and_boxplot(train_df_imputed, categorical_column, target_column)


# Aggregate sales by date
daily_sales = train_df_imputed.groupby('date')['num_sold'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=daily_sales, x='date', y='num_sold', color=sns.color_palette("plasma")[0])
plt.title('Daily Sales Trend', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


categorical_columns = ['country', 'store', 'product']

# Set up the subplots for daily sales trends
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8), sharey=True)

# Loop through each category and create a subplot for daily trends
for i, category in enumerate(categorical_columns):
    # Aggregate sales by date and category
    category_sales_daily = train_df_imputed.groupby(['date', category])['num_sold'].sum().reset_index()
    
    sns.lineplot(
        data=category_sales_daily,
        x='date',
        y='num_sold',
        hue=category,
        palette='plasma',
        linewidth=1,
        ax=axes[i]
    )
    
    axes[i].set_title(f'Daily Sales Trend by {category.capitalize()}', fontsize=14)
    axes[i].set_xlabel('Date', fontsize=12)
    axes[i].set_ylabel('Total Sales', fontsize=12 if i == 0 else 0)  
    axes[i].legend(title=category.capitalize(), fontsize=10)
    axes[i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


# Function to calculate weights of each country based on total sales
def calculate_country_weights(df):
    """
    Calculate the weights of each country based on total sales.
    """
    total_sales = df["num_sold"].sum()
    country_weights = df.groupby("country")["num_sold"].sum() / total_sales
    return country_weights

# Function to calculate weights of each store based on total sales
def calculate_store_weights(df):
    """
    Calculate the weights of each store based on total sales.
    """
    total_sales = df["num_sold"].sum()
    store_weights = df.groupby("store")["num_sold"].sum() / total_sales
    return store_weights

# Function to calculate weights of each product based on total sales
def calculate_product_weights(df):
    """
    Calculate the weights of each product based on total sales.
    """
    total_sales = df["num_sold"].sum()
    product_weights = df.groupby("product")["num_sold"].sum() / total_sales
    return product_weights

# Calculate weights
country_weights = calculate_country_weights(train_df_imputed)
store_weights = calculate_store_weights(train_df_imputed)
product_weights = calculate_product_weights(train_df_imputed)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot Country Weights
sns.barplot(x=country_weights.index, y=country_weights.values, palette='plasma', ax=axes[0])
axes[0].set_title('Country Contribution to Total Sales', fontsize=12)
axes[0].set_xlabel('Country', fontsize=12)
axes[0].set_ylabel('Weight (Proportion of Total Sales)', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', linestyle='--', alpha=0.6)

# Plot Store Weights
sns.barplot(x=store_weights.index, y=store_weights.values, palette='plasma', ax=axes[1])
axes[1].set_title('Store Contribution to Total Sales', fontsize=12)
axes[1].set_xlabel('Store', fontsize=12)
axes[1].set_ylabel('Weight (Proportion of Total Sales)', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', linestyle='--', alpha=0.6)

# Plot Product Weights
sns.barplot(x=product_weights.index, y=product_weights.values, palette='plasma', ax=axes[2])
axes[2].set_title('Product Contribution to Total Sales', fontsize=12)
axes[2].set_xlabel('Product', fontsize=12)
axes[2].set_ylabel('Weight (Proportion of Total Sales)', fontsize=12)
axes[2].tick_params(axis='x', rotation=45)
axes[2].grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()



def decompose(data, group_col, ax, colormap='plasma'):
    """
    Decomposes and plots the fraction of sales for each group (e.g., store, product) over time
    
    """
    # Group data by date and group_col, then calculate total sales
    grouped = data.groupby(['date', group_col])['num_sold'].sum().reset_index()
    
    # Calculate global totals by date
    global_totals = data.groupby('date')['num_sold'].sum().reset_index()
    global_totals.rename(columns={'num_sold': 'num_sold_global'}, inplace=True)
    
    # Merge grouped data with global totals
    merged = grouped.merge(global_totals, on='date')
    merged['fractions'] = merged['num_sold'] / merged['num_sold_global']
    
    unique_groups = np.sort(merged[group_col].unique())
    colors = cm.get_cmap(colormap, len(unique_groups))
    
    for i, group in enumerate(unique_groups):
        mask = merged[group_col] == group
        ax.plot(
            merged[mask]['date'], 
            merged[mask]['fractions'], 
            label=group, 
            color=colors(i)  
        )
    
    ax.legend(bbox_to_anchor=(1, 1), title=group_col.capitalize())
    ax.set_xlabel('Date')
    ax.set_ylabel('Fraction of Total Sales')
    ax.grid(True, linestyle='--', alpha=0.6)



if __name__ == "__main__":
    # Display country weights
    print("Country Weights:")
    print(country_weights)

    # Decompose and plot country fractions
    fig, ax = plt.subplots(figsize=(10, 6))
    decompose(train_df_imputed, 'country', ax, colormap='plasma')
    ax.set_title("Country Fractions Over Time")
    plt.show()

    # Display store weights
    print("Store Weights:")
    print(store_weights)

    # Decompose and plot store fractions
    fig, ax = plt.subplots(figsize=(10, 6))
    decompose(train_df_imputed, 'store', ax, colormap='plasma')
    ax.set_title("Store Fractions Over Time")
    plt.show()

    # Display product weights
    print("Product Weights:")
    print(product_weights)

    # Decompose and plot product fractions
    fig, ax = plt.subplots(figsize=(10, 6))
    decompose(train_df_imputed, 'product', ax, colormap='plasma')
    ax.set_title("Product Fractions Over Time")
    plt.show()



def forecast_product_ratios(train_df, forecast_years):
    """
    Forecast product ratios for specific years based on historical data.

    """
    # Calculate product fractions by day
    product_df = train_df.groupby(["date", "product"])["num_sold"].sum().reset_index()
    
    # Pivot to get a DataFrame where each column represents a product
    product_ratio_df = product_df.pivot(index="date", columns="product", values="num_sold")
    
    # Normalize each row to calculate product ratios
    product_ratio_df = product_ratio_df.div(product_ratio_df.sum(axis=1), axis=0)
    product_ratio_df = product_ratio_df.stack().rename("ratios").reset_index()
    
    # Forecast product ratios for the given years
    forecasted_ratios = []
    for base_year, target_year, year_shift in forecast_years:
        # Filter data for the base year
        forecast_df = product_ratio_df[product_ratio_df["date"].dt.year == base_year].copy()
        # Shift the date to the target year
        forecast_df["date"] += pd.DateOffset(years=year_shift)
        forecasted_ratios.append(forecast_df)
    
    forecasted_ratios_df = pd.concat(forecasted_ratios, ignore_index=True)
    
    return forecasted_ratios_df



# Define the input DataFrame (train_df_imputed) and forecast years
forecast_years = [(2015, 2017, 2), (2016, 2018, 2), (2015, 2019, 4)]

forecasted_ratios_df = forecast_product_ratios(train_df_imputed, forecast_years)

# Display a sample of the forecasted ratios
print("Forecasted Product Ratios (Sample):")
print(forecasted_ratios_df.head(5))


# Create a copy of the DataFrame
original_train_df_imputed = train_df_imputed.copy()

# Aggregate total sales by date
train_df_imputed = train_df_imputed.groupby(["date"])["num_sold"].sum().reset_index()

# Extract year, month, day, and day of the week from the date
train_df_imputed["year"] = train_df_imputed["date"].dt.year
train_df_imputed["month"] = train_df_imputed["date"].dt.month
train_df_imputed["day"] = train_df_imputed["date"].dt.day
train_df_imputed["day_of_week"] = train_df_imputed["date"].dt.dayofweek



correlation_features = [
    'year', 
    'month', 
    'day', 
    'day_of_week', 
    'num_sold'
]

# Compute the correlation matrix
correlation_matrix = train_df_imputed[correlation_features].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="plasma", cbar=True, square=True, linewidths=0.5)
plt.title("Correlation Heatmap of Features", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


# Aggregate total sales by year and month
monthly_sales = train_df_imputed.groupby(['year', 'month'])['num_sold'].sum().reset_index()

# Create a 'date' column for monthly aggregation
monthly_sales['date'] = pd.to_datetime(monthly_sales[['year', 'month']].assign(day=1))



# Set up the line plot for monthly sales trends
plt.figure(figsize=(14, 7))
sns.lineplot(
    data=monthly_sales,
    x='date',
    y='num_sold',
    marker='o',
    color=sns.color_palette("plasma", as_cmap=True)(0.5) 
)

plt.title("Monthly Sales Trends", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Total Sales", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Pivot the data for heatmap
heatmap_data = monthly_sales.pivot(index="year", columns="month", values="num_sold")

plt.figure(figsize=(12, 8))
sns.heatmap(
    heatmap_data,
    annot=True, fmt=".0f", cmap="plasma", linewidths=0.5, cbar=True
)

plt.title("Sales Trends by Month and Year", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Year", fontsize=12)
plt.xticks(ticks=range(1, 13), labels=[
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", 
    "Aug", "Sep", "Oct", "Nov", "Dec"
], rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()



test_total_sales_df = test_df.groupby(["date"])["id"].first().reset_index().drop(columns="id")
test_total_sales_df["month"] = test_total_sales_df["date"].dt.month
test_total_sales_df["day"] = test_total_sales_df["date"].dt.day
test_total_sales_df["day_of_week"] = test_total_sales_df["date"].dt.dayofweek



df = train_df_imputed.copy()
df['iso_year'] = df['date'].dt.isocalendar().year 
df['iso_week'] = df['date'].dt.isocalendar().week 
df['week_id'] = df['iso_year'].astype(str) + '-W' + df['iso_week'].astype(str).str.zfill(2)
df['day_of_week'] = df['date'].dt.dayofweek

# Aggregate weekly total sales
weekly_total = df.groupby('week_id')['num_sold'].sum().reset_index()
weekly_total.rename(columns={'num_sold': 'weekly_total_sold'}, inplace=True)

# Merge weekly totals back to the original DataFrame
df = pd.merge(df, weekly_total, on='week_id')

# Calculate daily sales ratio as a percentage of weekly total sales
df['daily_sales_ratio'] = df['num_sold'] / df['weekly_total_sold']

# Aggregate weekly ratios by day of the week
weekly_ratio = df.groupby(['week_id', 'day_of_week'])['daily_sales_ratio'].sum().reset_index()

# Get the start date for each week (Monday)
weekly_ratio['week_start'] = pd.to_datetime(weekly_ratio['week_id'] + '-1', format='%Y-W%W-%w')
first_monday = weekly_ratio['week_start'].min()

# Define days of the week for labeling in plots
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']



palette = sns.color_palette("plasma_r", 7)  

plt.figure(figsize=(14, 7))
for day in range(7):
    day_data = weekly_ratio[weekly_ratio['day_of_week'] == day]
    plt.plot(
        day_data['week_start'], 
        day_data['daily_sales_ratio'], 
        label=days[day], 
        marker='o', 
        color=palette[day]  
    )

plt.xlim(left=first_monday)
plt.ylim(top=0.3)
plt.xlabel('Week Start Date (Monday)', fontsize=12)
plt.ylabel('Daily Sales Ratio (Percentage of Weekly Total)', fontsize=12)
plt.title('Weekly Sales Ratio by Day of Week', fontsize=14)
plt.legend(title='Day of Week', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)  
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.show()



# Aggregate sales by day of the week
day_of_week_sales = train_df_imputed.groupby("day_of_week")["num_sold"].sum()
day_of_week_sales.index = days  

plt.figure(figsize=(10, 5))
sns.barplot(x=day_of_week_sales.index, y=day_of_week_sales.values, palette="plasma_r")
plt.title("Total Sales by Day of the Week", fontsize=14)
plt.xlabel("Day of the Week", fontsize=12)
plt.ylabel("Total Sales", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()



# Group by month and day of the week, summing the sales
monthly_day_sales = df.groupby(['month', 'day_of_week'])['num_sold'].sum().reset_index()

# Define days of the week for labeling in plots
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Pivot the data for heatmap: rows = days, columns = months, values = sales
heatmap_data = monthly_day_sales.pivot(index='day_of_week', columns='month', values='num_sold')

# Set day names for rows; ensure index aligns with expected day order
heatmap_data.index = days[:len(heatmap_data)]  

plt.figure(figsize=(12, 8))
sns.heatmap(
    heatmap_data, 
    annot=True, fmt=".0f", cmap="plasma", linewidths=0.5, cbar=True
)

plt.title("Aggregate Sales by Month and Day of Week", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Day of the Week", fontsize=12)
plt.xticks(ticks=range(1, 13), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()



# Calculate Day of Week Ratios and Adjust Sales

# Calculate the average sales for each day of the week and normalize it
day_of_week_ratio = (
    train_df_imputed.groupby("day_of_week")["num_sold"].mean() / 
    train_df_imputed.groupby("day_of_week")["num_sold"].mean().mean()
).rename("day_of_week_ratios")

# Display day of week ratios
display(day_of_week_ratio)

# Merge day of week ratios back into the main DataFrame
train_df_imputed = pd.merge(train_df_imputed, day_of_week_ratio, how="left", on="day_of_week")

# Adjust num_sold based on day of week ratios
train_df_imputed["adjusted_num_sold"] = train_df_imputed["num_sold"] / train_df_imputed["day_of_week_ratios"]

# Check the difference for Adjusted Sales
difference_check = (train_df_imputed["num_sold"].sum() - train_df_imputed["adjusted_num_sold"].sum()) / train_df_imputed["num_sold"].sum()
print(f"The difference between original and adjusted total sales as a proportion is: {difference_check:.6f}")

# Print adjusted_num_sold values
print("\nAdjusted Sales (Adjusted num_sold):")
print(train_df_imputed[["date", "num_sold", "adjusted_num_sold"]].head())  



# Function to Prepare Test Data for Forecasting
def prepare_test_data(train_df_imputed, test_total_sales_df, day_of_week_ratio):
    """
    Prepare the test data by calculating daily mean sales and incorporating day-of-week ratios.
    
    """
    # Filter training data for the last X years (from 2010 onwards)
    train_last_x_years_df = train_df_imputed.loc[train_df_imputed["year"] >= 2010]
    
    # Calculate daily mean of adjusted sales for each month and day
    train_day_mean_df = train_last_x_years_df.groupby(["month", "day"])["adjusted_num_sold"].mean().reset_index()
    
    # Merge average daily sales into the test DataFrame
    test_total_sales_df = pd.merge(test_total_sales_df, train_day_mean_df, how="left", on=["month", "day"])
    
    # Merge with day-of-week ratios
    test_total_sales_df = pd.merge(test_total_sales_df, day_of_week_ratio.reset_index(), how="left", on="day_of_week")
    
    # Calculate forecasted daily sales
    test_total_sales_df["num_sold"] = test_total_sales_df["adjusted_num_sold"] * test_total_sales_df["day_of_week_ratios"]
    
    return test_total_sales_df



# Function to Disaggregate Total Sales Forecast
def disaggregate_forecast(test_df, test_total_sales_df, store_weights, gdp_per_capita_filtered_ratios_df, forecasted_ratios_df):
    """
    Disaggregate total sales forecast by incorporating store, country, and product ratios.
    
    """
    # Add store ratios
    store_weights_df = store_weights.reset_index()
    test_sub_df = pd.merge(test_df, test_total_sales_df, how="left", on="date")
    test_sub_df.rename(columns={"num_sold": "day_num_sold"}, inplace=True)
    
    # Add product ratios
    test_sub_df = pd.merge(test_sub_df, store_weights_df, how="left", on="store")
    test_sub_df.rename(columns={"num_sold": "store_ratio"}, inplace=True)
    
    # Add country ratios
    test_sub_df["year"] = test_sub_df["date"].dt.year
    test_sub_df = pd.merge(test_sub_df, gdp_per_capita_filtered_ratios_df, how="left", on=["year", "country"])
    test_sub_df.rename(columns={"ratio": "country_ratio"}, inplace=True)
    
    # Add product ratios
    test_sub_df = pd.merge(test_sub_df, forecasted_ratios_df, how="left", on=["date", "product"])
    test_sub_df.rename(columns={"ratios": "product_ratio"}, inplace=True)
    
    # Adjust for bias for Kenya's GDP ratio
    test_sub_df.loc[test_sub_df['country'] == 'Kenya', 'country_ratio'] += 0.00249144564 * 1 / 10
    
    # Calculate final forecasted `num_sold`
    test_sub_df["num_sold"] = (
        test_sub_df["day_num_sold"] * 
        test_sub_df["store_ratio"] * 
        test_sub_df["country_ratio"] * 
        test_sub_df["product_ratio"]
    )
    
    # Round `num_sold` to nearest integer
    test_sub_df["num_sold"] = test_sub_df["num_sold"].round()
    
    return test_sub_df



# Function to Plot Individual Time Series
def plot_individual_ts(df):
    """
    Plot individual time series for each combination of country, store, and product.
    
    """
    # Generate a color palette using Plasma
    unique_countries = df["country"].unique()
    colour_map = sns.color_palette("plasma", len(unique_countries))
    
    country_color_map = {country: colour_map[i] for i, country in enumerate(unique_countries)}
    
    for country in unique_countries:
        f, axes = plt.subplots(df["store"].nunique() * df["product"].nunique(), figsize=(20, 70))
        count = 0
        
        for store in df["store"].unique():
            for product in df["product"].unique():
                plot_data = df.loc[
                    (df["product"] == product) & 
                    (df["country"] == country) & 
                    (df["store"] == store)
                ]
                sns.lineplot(data=plot_data, x="date", y="num_sold", linewidth=0.5,
                             ax=axes[count], color=country_color_map[country])
                axes[count].set_title(f"{country} - {store} - {product}")
                axes[count].axvline(pd.to_datetime("2017-01-01"), color='black', linestyle='--')
                axes[count].grid(True, linestyle='--', alpha=0.6)
                count += 1



# Prepare the forecasted data
test_total_sales_forecasted = prepare_test_data(train_df_imputed, test_total_sales_df, day_of_week_ratio)

# Disaggregate the forecasted data
test_forecast_disaggregated = disaggregate_forecast(
    test_df,
    test_total_sales_forecasted,
    store_weights,
    gdp_per_capita_filtered_ratios_df,
    forecasted_ratios_df
)

# Plot individual time series using both original and forecasted data
plot_individual_ts(pd.concat([original_train_df_imputed, test_forecast_disaggregated]).reset_index(drop=True))



# Create submission file
sample_df = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")
sample_df["num_sold"] = test_forecast_disaggregated["num_sold"]

display(sample_df.head(5))

sample_df.to_csv("submission.csv", index=False)
print("Submission file created.")

