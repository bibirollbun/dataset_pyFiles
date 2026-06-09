import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'])
train_og_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'])

test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'])


display(train_df.head(5))
display(test_df.head(5))


country_count = train_df['country'].value_counts().reset_index()
store_count = train_df['store'].value_counts().reset_index()
product_count = train_df['product'].value_counts().reset_index()

fig, axs = plt.subplots(3,1, figsize=(10,15))
sns.barplot(data=country_count, x='country', y='count', ax=axs[0])
axs[0].set_title('Data entries for countries')
axs[0].set(xlabel=None)
sns.barplot(data=store_count, x='store', y='count', ax=axs[1])
axs[1].set_title('Data entries for stores')
axs[1].set(xlabel=None)
sns.barplot(data=product_count, x='product', y='count', ax=axs[2])
axs[2].set_title('Data entries for products')
axs[2].set(xlabel=None)
plt.show()


entries_counts = train_df.groupby(['country','store','product'])['id'].count().rename('num_rows').reset_index()
unique_series = entries_counts['num_rows'].value_counts().reset_index()

display(entries_counts.head(5))
display(unique_series.head(5))


missing_values = train_df.isna().sum()

print(f"Missing values in the dataset: \n\n{missing_values}")


sales_counts = train_df.groupby(['country', 'store', 'product'])['num_sold'].count().rename('num_rows')

missing = sales_counts.loc[sales_counts != 2557]
missing_df = missing.reset_index()
missing_df['num_missing'] = 2557 - missing_df['num_rows']

display(missing_df)


f, axs = plt.subplots(9,1, figsize=(20,50))

for i, (country, store, product) in enumerate(missing.index):
    plot_df = train_df.loc[
        (train_df['country'] == country) & 
        (train_df['store'] == store) & 
        (train_df['product'] == product)
    ]
    missing_entries = plot_df.loc[plot_df['num_sold'].isna()]
    plot_df_clean = plot_df.dropna(subset=["num_sold"])

    sns.lineplot(data=plot_df_clean, x='date', y='num_sold', ax=axs[i])
    for missing_date in missing_entries['date']:
        axs[i].axvline(missing_date, color='red',  linestyle='-', linewidth=1.0, alpha=0.2)
    axs[i].set_title(f"{country} - {store} - {product}")


print(f"Train data, start date: {train_df['date'].min()}")
print(f"Train data, end date: {train_df['date'].max()}")
print(f"Test data, start date: {test_df['date'].min()}")
print(f"Test data, end date: {test_df['date'].max()}")


weekly_countries = train_df.groupby(['country', 'product', pd.Grouper(key='date', freq='W')])['num_sold'].sum().reset_index()
monthly_countries = train_df.groupby(['country', 'product', pd.Grouper(key='date', freq='MS')])['num_sold'].sum().reset_index()

weekly_stores = train_df.groupby(['store', 'product', pd.Grouper(key='date', freq='W')])['num_sold'].sum().reset_index()
monthly_stores = train_df.groupby(['store', 'product', pd.Grouper(key='date', freq='MS')])['num_sold'].sum().reset_index()


f,axs = plt.subplots(5,1, figsize=(20,50))
f.tight_layout()

for i, prod in enumerate(weekly_countries['product'].unique()):
    sns.lineplot(data=weekly_countries[weekly_countries['product'] == prod], x='date', y='num_sold', hue='country', ax=axs[i])
    axs[i].set_title(f'Weekly sales of {prod} per country')
    axs[i].set(xlabel=None)
plt.show()


f,axs = plt.subplots(5,1, figsize=(20,50))
f.tight_layout()

for i, prod in enumerate(monthly_countries['product'].unique()):
    sns.lineplot(data=monthly_countries[monthly_countries['product'] == prod], x='date', y='num_sold', hue='country', ax=axs[i])
    axs[i].set_title(f'Monthly sales of {prod} per country')
    axs[i].set(xlabel=None)
plt.show()


f,axs = plt.subplots(5,1, figsize=(20,50))
f.tight_layout()

for i, prod in enumerate(weekly_stores['product'].unique()):
    sns.lineplot(data=weekly_stores[weekly_stores['product'] == prod], x='date', y='num_sold', hue='store', ax=axs[i])
    axs[i].set_title(f'Weekly sales of {prod} per store')
    axs[i].set(xlabel=None)
plt.show()


f,axs = plt.subplots(5,1, figsize=(20,50))
f.tight_layout()

for i, prod in enumerate(monthly_stores['product'].unique()):
    sns.lineplot(data=monthly_stores[monthly_stores['product'] == prod], x='date', y='num_sold', hue='store', ax=axs[i])
    axs[i].set_title(f'Monthly sales of {prod} per store')
    axs[i].set(xlabel=None)
plt.show()

