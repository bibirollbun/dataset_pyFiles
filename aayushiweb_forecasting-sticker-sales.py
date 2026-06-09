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


train1=pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


train1.head()


test.head()


train= train1.drop(columns=["num_sold"])  # Features
Y = train1["num_sold"]  # Target



train.head()



Y.head()


final=pd.concat([train,test],axis=0)
final.info()


import datetime as dt
final["date"]=pd.to_datetime(final["date"])


final['Year']=(final["date"]).dt.year
final["month"]=(final['date']).dt.month
final["day"]=final["date"].dt.day


final.info()


train1['num_sold'].fillna(0, inplace=True)





import matplotlib.pyplot as plt


plt.figure(figsize=(12, 6))
train1.groupby('date')['num_sold'].sum().plot()
plt.title('Total Products Sold Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Products Sold')
plt.grid()
plt.show()


plt.figure(figsize=(12, 6))
train1.groupby('country')['num_sold'].sum().plot(kind='bar')
plt.title('Total Products Sold by Country')
plt.xlabel('Country')
plt.ylabel('Number of Products Sold')
plt.grid()
plt.show()


plt.figure(figsize=(12, 6))
train1.groupby('store')['num_sold'].sum().plot(kind='bar')
plt.title('Total Products Sold by Store Type')
plt.xlabel('Store Type')
plt.ylabel('Number of Products Sold')
plt.grid()
plt.show()




plt.figure(figsize=(12, 6))
train1.groupby('product')['num_sold'].sum().sort_values(ascending=False).plot(kind='bar', figsize=(15, 7))
plt.title('Total Products Sold by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Number of Products Sold')
plt.grid()
plt.show()




train1['date'] = pd.to_datetime(train1['date'], errors='coerce')
# Seasonal trends (e.g., monthly sales) visualization
train1['month'] = train1['date'].dt.month
train1['month'] = train1['date'].dt.month
plt.figure(figsize=(12, 6))
train1.groupby('month')['num_sold'].sum().plot()
plt.title('Seasonal Trends: Products Sold by Month')
plt.xlabel('Month')
plt.ylabel('Number of Products Sold')
plt.grid()
plt.show()


plt.figure(figsize=(12, 6))
(train1.groupby(['date', 'store'])['num_sold'].sum().groupby('store').mean()).plot(kind='bar')
plt.title('Average Daily Sales per Store Type')
plt.xlabel('Store Type')
plt.ylabel('Average Daily Products Sold')
plt.grid()
plt.show()


train1['year'] = train1['date'].dt.year
train1['weekday'] = train1['date'].dt.day_name()

# 1. Monthly sales trends (detailed)
monthly_sales = train1.groupby('month')['num_sold'].sum()

plt.figure(figsize=(12, 6))
monthly_sales.plot(kind='bar')
plt.title('Total Products Sold by Month')
plt.xlabel('Month')
plt.ylabel('Number of Products Sold')
plt.xticks(range(12), [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])
plt.grid()
plt.show()


# Scatter plot visualization: Number of products sold over time, categorized by country

plt.figure(figsize=(12, 6))
for country in train1['country'].unique():
    country_data = train1[train1['country'] == country]
    plt.scatter(
        country_data['date'], 
        country_data['num_sold'], 
        label=country, 
        alpha=0.6
    )

plt.title('Number of Products Sold Over Time by Country')
plt.xlabel('Date')
plt.ylabel('Number of Products Sold')
plt.legend(title='Country', loc='upper left')
plt.grid()
plt.show()



import seaborn as sns


train1['month'] = train1['date'].dt.month
heatmap_data = train1.pivot_table(
    values='num_sold',
    index='country',
    columns='month',
    aggfunc=np.sum,
    fill_value=0
)

# Plotting the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=.5)
plt.title('Heatmap of Products Sold: Country vs Month')
plt.xlabel('Month')
plt.ylabel('Country')
plt.show()





final.info()


cat_ftrs = list(final.select_dtypes(include=['object']).columns)
cat_ftrs 


cat_cou=pd.get_dummies(final["country"])
cat_cou=cat_pro.astype(int)
cat_cou.info()


cat_str=pd.get_dummies(final["store"])
cat_str=cat_str.astype(int)
cat_str.info()


cat_pro=pd.get_dummies(train["product"])
cat_pro=cat_pro.astype(int)
cat_pro.info()




