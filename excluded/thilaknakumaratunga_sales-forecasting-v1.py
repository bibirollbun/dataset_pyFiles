# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
paths = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        print(path)
        paths.append(path)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


print("hello")


!pip install py7zr


import py7zr
for filenames in paths:
    with py7zr.SevenZipFile(filenames, mode='r') as z_ref:
        z_ref.extractall(path='/kaggle/working')


train = pd.read_csv("/kaggle/working/train.csv")
stores = pd.read_csv("/kaggle/working/stores.csv")
items = pd.read_csv("/kaggle/working/items.csv")
holiday_events = pd.read_csv("/kaggle/working/holidays_events.csv")
transactions = pd.read_csv("/kaggle/working/transactions.csv")
oil = pd.read_csv("/kaggle/working/oil.csv")
test = pd.read_csv('/kaggle/working/test.csv')


print("train: ",train.columns)
print("stores: ",stores.columns)
print("items: ",items.columns)
print("holidays_events: ",holiday_events.columns)
print("transactions: ",transactions.columns)
print("oil: ",oil.columns)
print("test: ", test.columns)


train


import pandas as pd
import matplotlib.pyplot as plt

# Convert date column to datetime
train["date"] = pd.to_datetime(train["date"])

# Aggregate unit sales by date
daily_sales = train.groupby("date")["unit_sales"].sum().reset_index()

# Plot line graph
plt.figure(figsize=(12, 6))
plt.plot(daily_sales["date"], daily_sales["unit_sales"], label="Total Daily Sales", color="blue")

# Formatting
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.title("Daily Sales Over Time")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.show()



stats = []
for col in train.columns:  # Iterate over each column in the DataFrame
    stats.append((
        col,  # Column name
        train[col].nunique(),  # Count of unique values in the column
        round(train[col].isnull().sum() * 100 / train.shape[0], 3),  # Percentage of missing values
        round(train[col].value_counts(normalize=True, dropna=False).values[0] * 100, 3),  # Frequency of the most common value (as %)
        train[col].dtype  # Data type of the column
    ))


stats_df = pd.DataFrame(stats, columns=[
    'features ',  # Column name
    'the number of unique attributes ',  # Number of unique values in the column
    'the proportion of missing values ',  # Percentage of missing values
    'the proportion of maximum attributes ',  # Frequency of the most common value
    'feature type '  # Data type of the column
])



stats_df.drop(columns = ['the proportion of maximum attributes '])


stores


stores['type'].unique()


## cluster is a grouping of similar stores.
stores['cluster'].unique()


perishable_counts = stores['cluster'].value_counts()

# Plotting the bar graph
plt.figure(figsize=(6, 4))
perishable_counts.plot(kind='bar', color='green')
plt.title('Count of Clusters')
plt.xlabel('Clusters')
plt.ylabel('Count')
plt.show()


print("hello")


import matplotlib.pyplot as plt
import seaborn as sns
# Count the number of stores in each city-state combination

city_state_counts = stores.groupby(["state", "city"]).size().unstack()

# Plot stacked bar chart with states on X-axis
city_state_counts.plot(kind="bar", stacked=True, figsize=(9, 6), colormap="tab20")

plt.xlabel("State")
plt.ylabel("Number of Stores")
plt.title("Distribution of Stores by State and City")
plt.legend(title="City", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xticks(rotation=45, ha="right")

plt.show()


stores_stats = []
for col in stores.columns:  # Iterate over each column in the DataFrame
    stores_stats.append((
        col,  # Column name
        stores[col].nunique(),  # Count of unique values in the column
        round(stores[col].isnull().sum() * 100 / train.shape[0], 3),  # Percentage of missing values
        round(stores[col].value_counts(normalize=True, dropna=False).values[0] * 100, 3),  # Frequency of the most common value (as %)
        stores[col].dtype  # Data type of the column
    ))

stores_stats_df = pd.DataFrame(stores_stats, columns=[
    'features ',  # Column name
    'the number of unique attributes ',  # Number of unique values in the column
    'the proportion of missing values ',  # Percentage of missing values
    'the proportion of maximum attributes ',  # Frequency of the most common value
    'feature type '  # Data type of the column
])

stores_stats_df.drop(columns = ['the proportion of maximum attributes '])


oil["date"] = pd.to_datetime(oil["date"])

# Drop rows with NaN values
oil_ = oil.dropna()

# Plot bar graph
plt.figure(figsize=(12, 6))
plt.plot(oil_["date"], oil_["dcoilwtico"],label="Oil Price")

# Formatting
plt.xlabel("Date")
plt.ylabel("Oil Price (USD)")
plt.title("Oil Price Over Time")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.show()


transactions


transactions["date"] = pd.to_datetime(transactions["date"])


# Aggregate transactions by date
daily_transactions = transactions.groupby("date")["transactions"].sum().reset_index()

# Plot line graph
plt.figure(figsize=(12, 6))
plt.plot(daily_transactions["date"], daily_transactions["transactions"], label="Total Transactions", color="blue")

# Formatting
plt.xlabel("Date")
plt.ylabel("Total Transactions")
plt.title("Daily Transactions Over Time")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.show()


daily_transactions 


holiday_events


len(train['store_nbr'].unique())


oil


items


items['family'].unique()


tems_stats = []
for col in items.columns:  # Iterate over each column in the DataFrame
    tems_stats.append((
        col,  # Column name
        items[col].nunique(),  # Count of unique values in the column
        round(items[col].isnull().sum() * 100 / train.shape[0], 3),  # Percentage of missing values
        round(items[col].value_counts(normalize=True, dropna=False).values[0] * 100, 3),  # Frequency of the most common value (as %)
        items[col].dtype  # Data type of the column
    ))

items_stats_df = pd.DataFrame(tems_stats, columns=[
    'features ',  # Column name
    'the number of unique attributes ',  # Number of unique values in the column
    'the proportion of missing values ',  # Percentage of missing values
    'the proportion of maximum attributes ',  # Frequency of the most common value
    'feature type '  # Data type of the column
])

items_stats_df.drop(columns = ['the proportion of maximum attributes '])


perishable_counts = items['perishable'].value_counts()

# Plotting the bar graph
plt.figure(figsize=(6, 4))
perishable_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Count of Perishable vs Non-Perishable Items')
plt.xlabel('Perishable (0: Non-Perishable, 1: Perishable)')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Non-Perishable', 'Perishable'], rotation=0)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is 'items'
# Count the occurrences of each family type
family_counts = items['family'].value_counts()

# Generate a list of colors (one for each family)
colors = plt.cm.get_cmap('tab20', len(family_counts))

# Plot the horizontal bar graph with different colors
plt.figure(figsize=(10, 6))
family_counts.plot(kind='barh', color=colors(range(len(family_counts))))
plt.xlabel('Number of Items')
plt.ylabel('Family')
plt.title('Distribution of Items Across Families')
plt.show()






holiday_events


holiday_events['type'].unique()

