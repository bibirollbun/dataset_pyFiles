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
import numpy as np
import dateutil.easter as easter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
import math
import warnings

warnings.filterwarnings(action="ignore")
sns.set(style="whitegrid")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
gdp = pd.read_csv('/kaggle/input/gdp-per-capita-2010-2019/filtered_gdp_per_capita (2).csv')

for df in [train_df, test_df]:
    df['date'] = pd.to_datetime(df.date)
    df.set_index('date', inplace=True, drop=False)


display(train_df.head())
display(test_df.head())
display(gdp.head())


train_df['missing_num_sold'] = train_df['num_sold'].isnull()

train_df.head()


missing_by_country = train_df.groupby('country')['missing_num_sold'].sum()
missing_by_store = train_df.groupby('store')['missing_num_sold'].sum()
missing_by_product = train_df.groupby('product')['missing_num_sold'].sum()

fig, axes = plt.subplots(3, 1, figsize=(8, 16), sharex=False)

def annotate_bars(ax):
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,.0f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom')

# Missingness by country
ax0 = missing_by_country.sort_values().plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Proportion of Missing num_sold by Country', fontsize=14)
axes[0].set_ylabel('Proportion Missing')
annotate_bars(axes[0])
axes[0].tick_params(axis='x', rotation=0)

# Missingness by store
ax1 = missing_by_store.sort_values().plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('Proportion of Missing num_sold by Store', fontsize=14)
axes[1].set_ylabel('Proportion Missing')
annotate_bars(axes[1])
axes[1].tick_params(axis='x', rotation=0)

# Missingness by product
ax2 = missing_by_product.sort_values().plot(kind='bar', ax=axes[2], color='salmon')
axes[2].set_title('Proportion of Missing num_sold by Product', fontsize=14)
axes[2].set_ylabel('Proportion Missing')
annotate_bars(axes[2])
axes[2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('missing.png')  
plt.show()
plt.close()


train_df['num_sold'].fillna(0, inplace=True)
train_df.isnull().sum()


print("Train Dataset:\nFirst day:", train_df.date.min(), "   Last day:", train_df.date.max())
print("Test Dataset:\nFirst day:", test_df.date.min(), "   Last day:", test_df.date.max())


# Extract seasonal attributes
train_df['day'] = train_df['date'].dt.day
train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['quarter'] = train_df['date'].dt.quarter
train_df['weekday'] = train_df['date'].dt.weekday 


country_sales = train_df.groupby('country')['num_sold'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=country_sales.index, y=country_sales.values, palette='magma')
plt.title("Average Sales by Country")
plt.xlabel("Country")
plt.ylabel("Average Sales")

for p in ax.patches:
    ax.annotate(f'{p.get_height():,.0f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


yearly_sales = train_df.groupby(['year', 'country'])['num_sold'].sum().reset_index()
gdp_long = gdp.melt(id_vars='year', var_name='country', value_name='GDP_per_capita')
gdp_long['country'] = gdp_long['country'].str.replace('GDP_', '')
gdp_sales = pd.merge(yearly_sales, gdp_long, on=['year', 'country'], how='inner')


display(yearly_sales.head())
display(gdp_long.head())
display(gdp_sales.head())


agg_gdp=gdp_long.groupby('year')['GDP_per_capita'].sum()

plt.figure(figsize=(10, 6))
sns.lineplot(x=agg_gdp.index, y=agg_gdp.values, marker='o', color='b')
plt.title('Aggreggate GDP per Capita Trends')
plt.xlabel('Year')
plt.ylabel('Total GDP')
plt.grid(True)
plt.show()


agg_gdp=gdp_sales.groupby('year')['GDP_per_capita'].sum()
yearly_sales = gdp_sales.groupby('year')['num_sold'].sum()

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

sns.lineplot(x=agg_gdp.index, y=agg_gdp.values, marker='o', color='b', ax=axes[0])
axes[0].set_title("GDP_per_capita Trends Over Time by Country")
axes[0].set_ylabel("GDP per Capita")
axes[0].legend(title="Country")

sns.lineplot(x=yearly_sales.index, y=yearly_sales.values, marker='o', color='b', ax=axes[1])
axes[1].set_title("Sales Trends Over Time by Country")
axes[1].set_ylabel("Total Sales")
axes[1].set_xlabel("Year")
axes[1].legend(title="Country")

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

sns.lineplot(data=gdp_sales, x="year", y="GDP_per_capita", hue="country", ax=axes[0])
axes[0].set_title("GDP per Capita Trends Over Time by Country")
axes[0].set_ylabel("GDP per Capita")
axes[0].legend(title="Country")

sns.lineplot(data=gdp_sales, x="year", y="num_sold", hue="country", ax=axes[1])
axes[1].set_title("Sales Trends Over Time by Country")
axes[1].set_ylabel("Total Sales")
axes[1].set_xlabel("Year")
axes[1].legend(title="Country")

plt.tight_layout()
plt.savefig('gdp_and_sales.png')  
plt.show() 
plt.close()


y = gdp_sales['GDP_per_capita']
x = gdp_sales['num_sold']
correlation = y.corr(x)
correlation 


y = gdp_sales['GDP_per_capita']
x = gdp_sales['num_sold']
correlation = y.corr(x)
print(f'Correlation between GDP per Capita and num_sold:\n{correlation}')

plt.title('Correlation')
plt.scatter(x, y)
plt.plot(np.unique(x), 
         np.poly1d(np.polyfit(x, y, 1))
         (np.unique(x)), color='red')

plt.xlabel('GDP per Capita')
plt.ylabel('num_sold')


seasonal_sales = train_df.groupby(['year', 'month'])['num_sold'].sum().reset_index()

palette = sns.color_palette("tab10", n_colors=len(seasonal_sales['year'].unique()))

plt.figure(figsize=(14, 6))
sns.lineplot(data=seasonal_sales, x='month', y='num_sold', hue='year', marker="o", palette=palette)
plt.title('Monthly Sales Trends by Year', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Products Sold', fontsize=12)
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('monthly.png')  
plt.show()
plt.close()


weekday_sales = train_df.groupby('weekday')['num_sold'].mean()

plt.figure(figsize=(10, 6))
sns.barplot(x=weekday_sales.index, y=weekday_sales.values, palette="coolwarm")
plt.title('Weekday Sales Trend')
plt.xlabel('Weekday')
plt.ylabel('Average Sales')
plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.savefig('weekly.png')  
plt.show()
plt.close()


num_groups = 90
cols = 3
rows = math.ceil(num_groups / cols)

plt.figure(figsize=(18, rows * 3.5))

for i, (combi, df) in enumerate(train_df.groupby(['country', 'store', 'product'])):
    ax = plt.subplot(rows, cols, i + 1)

    december_sales = (
        df[df['date'].dt.month == 12]
        .groupby(df['date'].dt.day)['num_sold']
        .mean()
    )

    ax.bar(
        december_sales.index,
        december_sales,
        color=['b'] * 25 + ['orange'] * 6    )

    ax.set_title(combi)
    ax.set_xticks(ticks=range(5, 31, 5))

plt.tight_layout(h_pad=1.5)
plt.suptitle('Daily Sales for December', y=1.005)
plt.savefig('december.png')  
plt.show()
plt.close()

