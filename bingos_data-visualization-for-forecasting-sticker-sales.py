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
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
data.head()


data.drop('id',axis=1,inplace=True)


data.shape


data.info()


data.isnull().sum()


data.dropna(inplace=True)


data.duplicated().sum()


data['country'].value_counts().plot(kind='bar')


data['store'].value_counts().plot(kind='bar')


data['product'].value_counts().plot(kind='bar')


data['date'] = pd.to_datetime(data['date'])


data.info()


import matplotlib.pyplot as plt
import seaborn as sns


# Univariate Analysis for Categorical Variables
def plot_categorical(column):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=column, order=data[column].value_counts().index)
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)
    plt.show()

print("Categorical Variables Analysis:")
for col in ['country', 'store', 'product']:
    print(f"\nFrequency counts for {col}:")
    print(data[col].value_counts())
    plot_categorical(col)


# Univariate Analysis for Numerical Variable
print("\nNumerical Variable Analysis:")
print("Summary Statistics for num_sold:")
print(data['num_sold'].describe())

# Histogram for num_sold
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='num_sold', bins=30, kde=True)
plt.title('Distribution of num_sold')
plt.xlabel('Number Sold')
plt.ylabel('Frequency')
plt.show()


# Box Plot for num_sold
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, y='num_sold')
plt.title('Box Plot of num_sold')
plt.ylabel('Number Sold')
plt.show()


# Univariate Analysis for Datetime Variable
print("\nDatetime Variable Analysis:")
df_date_agg = data.groupby('date')['num_sold'].sum().reset_index()

plt.figure(figsize=(15, 10))
sns.lineplot(data=df_date_agg, x='date', y='num_sold')
plt.title('Total num_sold Over Time')
plt.xlabel('Date')
plt.ylabel('Total Number Sold')
plt.xticks(rotation=45)
plt.show()


# List of categorical columns
categorical_cols = ['country', 'store', 'product']

# Generate countplots for each pair of categorical variables
for i, col1 in enumerate(categorical_cols):
    for col2 in categorical_cols[i+1:]:
        plt.figure(figsize=(12, 6))
        sns.countplot(data=data, x=col1, hue=col2)
        plt.title(f'Countplot of {col1} by {col2}')
        plt.xticks(rotation=45)
        plt.legend(title=col2)
        plt.tight_layout()
        plt.show()


# Generate boxplots for num_sold against each categorical variable
for col in categorical_cols:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x=col, y='num_sold')
    plt.title(f'Boxplot of num_sold by {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Aggregate num_sold by date and country
df_date_country = data.groupby(['date', 'country'])['num_sold'].sum().reset_index()

# Plot using lineplot with hue
plt.figure(figsize=(14, 7))
sns.lineplot(data=df_date_country, x='date', y='num_sold', hue='country')
plt.title('Total num_sold Over Time by Country')
plt.xlabel('Date')
plt.ylabel('Total Number Sold')
plt.xticks(rotation=45)
plt.legend(title='Country')
plt.tight_layout()
plt.show()


# Loop through store and product for time-based analysis
for col in ['store', 'product']:
    df_date_col = data.groupby(['date', col])['num_sold'].sum().reset_index()
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_date_col, x='date', y='num_sold', hue=col)
    plt.title(f'Total num_sold Over Time by {col}')
    plt.xlabel('Date')
    plt.ylabel('Total Number Sold')
    plt.xticks(rotation=45)
    plt.legend(title=col)
    plt.tight_layout()
    plt.show()




# Assuming the dataset is loaded as 'df'
df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'])

# Set up the FacetGrid
g = sns.FacetGrid(df, col='country', row='store', height=4, aspect=2)
g.map_dataframe(sns.lineplot, x='date', y='num_sold')
g.set_axis_labels('Date', 'Number Sold')
g.set_titles(col_template='{col_name}', row_template='{row_name}')
g.fig.suptitle('num_sold Trends by Country and Store', y=1.02)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Define bins using quantiles
bins = [0, df['num_sold'].quantile(0.33), df['num_sold'].quantile(0.66), df['num_sold'].max()]
labels = ['Low', 'Medium', 'High']

# Bin num_sold and add as a new column
df['num_sold_binned'] = pd.cut(df['num_sold'], bins=bins, labels=labels, include_lowest=True)

# Check the distribution of bins
print(df['num_sold_binned'].value_counts())

# Create a countplot
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='country', hue='num_sold_binned')
plt.title('Distribution of num_sold Bins by Country')
plt.xlabel('Country')
plt.ylabel('Count')
plt.legend(title='num_sold Bin')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Cross-tabulation: country vs. store
country_store_ct = pd.crosstab(df['country'], df['store'])
plt.figure(figsize=(10, 8))
sns.heatmap(country_store_ct, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Cross-Tabulation: Country vs. Store')
plt.xlabel('Store')
plt.ylabel('Country')
plt.tight_layout()
plt.show()


# Cross-tabulation: country vs. num_sold_binned
country_num_sold_ct = pd.crosstab(df['country'], df['num_sold_binned'])
plt.figure(figsize=(10, 8))
sns.heatmap(country_num_sold_ct, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Cross-Tabulation: Country vs. num_sold Bins')
plt.xlabel('num_sold Bin')
plt.ylabel('Country')
plt.tight_layout()
plt.show()

