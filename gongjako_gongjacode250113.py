import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


df_train=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


df_train.head()


df_train.tail()


df_test.head()


df_test.tail()


df_train.nunique()


df_train.isnull().sum()


missing_values=df_train.groupby('country')['num_sold'].apply(lambda x:x.isnull().sum())
missing_values


import matplotlib.pyplot as plt

missing_values.plot(kind='bar',figsize=(10,6))
plt.title('Missing_values in num_sold by Country')
plt.xlabel('country')
plt.ylabel('Number of Missing Values')
plt.xticks(rotation=45)
plt.show()


# 전체 결측치 계산
total_missing = sum(missing_values)
percentages = {country: (count / total_missing) * 100 for country, count in missing_values.items() if count>0}
print(percentages)


import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.pie(
    percentages.values(),
    labels=percentages.keys(),
    autopct='%1.1f%%',
    startangle=140
)
plt.title('Missing_percentage in num_sold by Country')
plt.show()


df_train['date']=pd.to_datetime(df_train['date'])
daily_sales=df_train.groupby('date')['num_sold'].sum()


df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)



df.isnull().sum()


df['date'] = pd.to_datetime(df['date'])


# Extracting time-based features (month, day of the week, quarter, etc.)
# Applying cyclic transformations to handle periodicity in features

df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek
# df['dayofyear'] = df['date'].dt.dayofyear
df['quarter'] = df['date'].dt.quarter
df['weekofyear'] = df['date'].dt.isocalendar().week #Week of the year
df['year'] = df['date'].dt.year
df['day'] = df['date'].dt.day

# Apply sine and cosine transformations to cyclical features
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)  # Assuming 31 days max
df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

df['day_of_week_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

df['week_of_year_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 52)
df['week_of_year_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 52)

df['Group'] = (df['year'] - 2010) * 48 + df['month'] * 4 + df['day'] // 7



# Generating summary statistics for the 'num_sold' column

df['num_sold'].describe()


# Checking unique values for categorical columns like 'country', 'store', and 'product'

print(df['country'].unique())
print(df['store'].unique())
print(df['product'].unique())


# Plotting total sales by date to observe trends and seasonality

sales_by_date = df.groupby('date')['num_sold'].sum()
sales_by_date.plot(title='Total Sales Over Time')


# Creating bar plots for sales by country, store, and product to identify high-performing categories

sales_by_country = df.groupby('country')['num_sold'].sum()
sales_by_country.plot(kind='bar', title='Sales by Country')


sales_by_store = df.groupby('store')['num_sold'].sum()
sales_by_store.plot(kind='bar', title='Sales by store')


sales_by_product = df.groupby('product')['num_sold'].sum()
sales_by_product.plot(kind='bar', title='Sales by product')


# Calculate Outlier Bounds Using IQR
# Calculating Interquartile Range (IQR) to identify potential outliers
# Filtering rows where 'num_sold' is outside the lower and upper bounds

# Recalculate Q1, Q3, and IQR
Q1 = df['num_sold'].quantile(0.25)
Q3 = df['num_sold'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter outliers
outliers = df[(df['num_sold'] < lower_bound) | (df['num_sold'] > upper_bound)]
print(f"Number of Outliers: {len(outliers)}")


# Visualizing outliers using box plots grouped by different categorical features

plt.figure(figsize=(10, 6))
df.boxplot(column='num_sold', by='country', grid=False)
plt.title("Outliers in 'num_sold' by Country")
plt.suptitle("")  # Removes the default matplotlib title
plt.ylabel('Number of Units Sold')
plt.xlabel('Country')
plt.show()



plt.figure(figsize=(12, 6))
df.boxplot(column='num_sold', by='store', grid=False)
plt.title("Outliers in 'num_sold' by Store")
plt.suptitle("")
plt.ylabel('Number of Units Sold')
plt.xlabel('Store')
plt.show()


plt.figure(figsize=(12, 6))
df.boxplot(column='num_sold', by='product', grid=False)
plt.title("Outliers in 'num_sold' by Product")
plt.suptitle("")
plt.ylabel('Number of Units Sold')
plt.xlabel('Product')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=df['num_sold'])
plt.title("Overall Outliers in 'num_sold'")
plt.xlabel('Number of Units Sold')
plt.show()


# Group outliers by country
outliers_by_country = outliers.groupby('country').size()
print("Outliers by Country:")
print(outliers_by_country)

# Group outliers by store
outliers_by_store = outliers.groupby('store').size()
print("Outliers by Store:")
print(outliers_by_store)

# Group outliers by product
outliers_by_product = outliers.groupby('product').size()
print("Outliers by Product:")
print(outliers_by_product)


outliers_by_month = outliers.groupby('month').size()
print("Outliers by Month:")
print(outliers_by_month)


outliers_by_day = outliers.groupby('dayofweek').size()
print("Outliers by Day of the Week:")
print(outliers_by_day)


outliers_by_year = outliers.groupby('year').size()
print("Outliers by Year:")
print(outliers_by_year)


# # replace with zero 
# # Drop rows where 'dataset' is 'train' and 'num_sold' is NaN
# df = df[~((df_train) & (df_train['num_sold'].isna()))]


# # df.loc[df['dataset'] == 'train', 'num_sold'] = df.loc[df['dataset'] == 'train', 'num_sold'].fillna(0)




