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


!pip install py7zr



import py7zr
from subprocess import check_output

for dirname, _, filenames in os.walk('/kaggle/input/favorita-grocery-sales-forecasting'):
    for filename in filenames:
        archive = py7zr.SevenZipFile(os.path.join(dirname, filename), mode='r')
        archive.extractall(path="/kaggle/working")
        archive.close()



import pandas as pd

# Load datasets
data_paths = {
    "train": "/kaggle/working/train.csv",
    "test": "/kaggle/working/test.csv",
    "stores": "/kaggle/working/stores.csv",
    "items": "/kaggle/working/items.csv",
    "transactions": "/kaggle/working/transactions.csv",
    "holidays": "/kaggle/working/holidays_events.csv",
    "oil": "/kaggle/working/oil.csv"
}

# Load all datasets into a dictionary
datasets = {name: pd.read_csv(path) for name, path in data_paths.items()}

# Display basic information for each dataset
for name, df in datasets.items():
    print(f"Dataset: {name}")
    display(df.head())
    print(df.info(), "\n")



for name, df in datasets.items():
    print(f"Dataset: {name}")
    display(df.describe())
    print(df.info(), "\n")






for name, df in datasets.items():
    print(f"{name}: {df.isnull().sum()}")


# Convert date columns to datetime format
date_columns = ["train", "test", "transactions", "holidays", "oil"]

for dataset in date_columns:
    datasets[dataset]["date"] = pd.to_datetime(datasets[dataset]["date"])

# Convert categorical columns
category_columns = {
    "stores": ["city", "state", "type", "cluster"],
    "items": ["family"],
    "holidays": ["type", "locale", "locale_name", "description", "transferred"],
}

for dataset, cols in category_columns.items():
    for col in cols:
        datasets[dataset][col] = datasets[dataset][col].astype("category")



datasets["oil"]['dcoilwtico'] = datasets["oil"]['dcoilwtico'].interpolate()



datasets["oil"]['dcoilwtico'] = datasets["oil"]['dcoilwtico'].fillna(method='bfill')



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.histplot(datasets["train"]["unit_sales"], bins=50, kde=True)
plt.title("Distribution of Sales (Unit Sales)")
plt.xlabel("Unit Sales")
plt.ylabel("Frequency")
plt.show()




for name, df in datasets.items():
    datasets[name] = df.replace([np.inf, -np.inf], np.nan)



for name, df in datasets.items():
    print(f"{name}: {df.isnull().sum()}")





# Aggregate daily sales
daily_sales = datasets["train"].groupby("date")["unit_sales"].sum()

plt.figure(figsize=(14, 6))
plt.plot(daily_sales, marker="o", linestyle="-", label="Total Sales")
plt.title("Daily Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Total Unit Sales")
plt.legend()
plt.grid(True)
plt.show()



# Merge train dataset with stores
train_stores = datasets["train"].merge(datasets["stores"], on="store_nbr", how="left")

# Aggregate sales by store type
sales_by_store_type = train_stores.groupby("type")["unit_sales"].sum()

# Plot sales by store type
plt.figure(figsize=(10, 6))
sns.barplot(x=sales_by_store_type.index, y=sales_by_store_type.values)
plt.title("Total Sales by Store Type")
plt.xlabel("Store Type")
plt.ylabel("Total Unit Sales")
plt.show()



# Merge train dataset with items
train_items = datasets["train"].merge(datasets["items"], on="item_nbr", how="left")

# Aggregate sales by product family
sales_by_family = train_items.groupby("family")["unit_sales"].sum().sort_values(ascending=False)

# Plot top product families
plt.figure(figsize=(14, 6))
sns.barplot(x=sales_by_family.index[:15], y=sales_by_family.values[:15])  # Top 15
plt.xticks(rotation=45)
plt.title("Top 15 Product Families by Sales")
plt.xlabel("Product Family")
plt.ylabel("Total Unit Sales")
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot to detect outliers
plt.figure(figsize=(12, 6))
sns.boxplot(x=datasets["train"]["unit_sales"])
plt.title("Boxplot of Sales Data (unit_sales)")
plt.show()




# Ensure 'date' is in datetime format
datasets["train"]["date"] = pd.to_datetime(datasets["train"]["date"])

# Extract date features
datasets["train"]["day"] = datasets["train"]["date"].dt.day
datasets["train"]["month"] = datasets["train"]["date"].dt.month
datasets["train"]["year"] = datasets["train"]["date"].dt.year
datasets["train"]["day_of_week"] = datasets["train"]["date"].dt.dayofweek
datasets["train"]["is_weekend"] = datasets["train"]["day_of_week"].isin([5, 6]).astype(int)














