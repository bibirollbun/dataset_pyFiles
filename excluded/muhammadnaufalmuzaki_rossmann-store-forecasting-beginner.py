import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train=pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')
df_store=pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')
df_test=pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')


pd.concat([df_train.head(), df_train.tail()])


pd.concat([df_store.head(), df_store.tail()])


pd.concat([df_test.head(), df_test.tail()])


print(f"df_train shape: {df_train.shape}")
print(f"df_store shape: {df_store.shape}")
print(f'df_test shape: {df_test.shape}')


print(df_train.info())
print('---------------------------------------------')
print(df_store.info())
print('---------------------------------------------')


test_ids=df_test['Id']

target_column='Sales'

cat_col_train=df_train.select_dtypes(include=['object']).columns
num_col_train=df_train.select_dtypes(exclude=['object']).columns
cat_col_store=df_store.select_dtypes(include=['object']).columns
num_col_store=df_store.select_dtypes(exclude=['object']).columns

print("Target Column:", target_column)
print("\nCategorical Columns: ")
for col in cat_col_train:
    print(col)
print("----------")
for col in cat_col_store:
    print(col)
print("\nNumerical Columns: ")
for col in num_col_train:
    print(col)
print("----------")
for col in num_col_store:
    print(col)


df_train.describe().round(2)


df_store.describe().round(2)


cat_col_train_check=['StateHoliday']
cat_col_store_check=['StoreType', 'Assortment']
for col in cat_col_train_check:
    print(f"'{col}' has {df_train[col].nunique()} unique categories")
for col in cat_col_store_check:
    print(f"'{col}' has {df_store[col].nunique()} unique categories")



plt.figure(figsize=(15, 9))
plt.title("Visualizing Missing Values (df_train)")
sns.heatmap(df_train.isnull(), cbar=False, cmap=sns.color_palette('magma'), yticklabels=False)
plt.show()

plt.figure(figsize=(15, 9))
plt.title("Visualizing Missing Values (df_store)")
sns.heatmap(df_store.isnull(), cbar=False, cmap=sns.color_palette('magma'), yticklabels=False)
plt.show()


# number of week
fig, ax1=plt.subplots(figsize=(15, 4))
sns.countplot(x='Open', hue='DayOfWeek', data=df_train, palette="husl", ax=ax1)
plt.show()


# Sales by date
df_train['Year']=df_train['Date'].apply(lambda x: int(str(x)[:4]))
df_train['Month']=df_train['Date'].apply(lambda x: int(str(x)[5:7]))

df_test['Year']=df_test['Date'].apply(lambda x: int(str(x)[:4]))
df_test['Month']=df_test['Date'].apply(lambda x: int(str(x)[5:7]))

df_train['Date']=df_train['Date'].apply(lambda x: (str(x)[:7]))
df_test['Date']=df_test['Date'].apply(lambda x: (str(x)[:7]))

avr_sales=df_train.groupby('Date')['Sales'].mean()
pct_change_sales=df_train.groupby('Date')['Sales'].sum().pct_change()

fig, (axis1, axis2)=plt.subplots(2, 1, sharex=True, figsize=(15, 8))

ax1=avr_sales.plot(legend=True, ax=axis1, marker='o', title="Average Sales")
ax1.set_xticks(range(len(avr_sales)))
ax1.set_xticklabels(avr_sales.index.tolist(), rotation=90)

ax2 = pct_change_sales.plot(legend=True ,ax=axis2, marker='o', rot=90, colormap="summer", title="Sales Percent Change")

ax1.grid()
ax2.grid()
plt.show()


# Sales by year
fig, (axis1, axis2)=plt.subplots(1, 2, figsize=(15, 4))

sns.barplot(x='Year', y='Sales', data=df_train, ax=axis1)
sns.barplot(x='Year', y='Sales', data=df_train, ax=axis2)


# Customers
fig, (axis1, axis2)=plt.subplots(2, 1, figsize=(15, 8))

sns.boxplot(df_train['Customers'], whis=np.inf, ax=axis1, orient='h')

avr_customers=df_train.groupby('Date')['Customers'].mean()

ax=avr_customers.plot(legend=True, ax=axis2, marker='o', title="Average Customers")
ax.set_xticks(range(len(avr_customers)))
xlabels = ax.set_xticklabels(avr_customers.index.tolist(), rotation=90)
ax.grid()


# Sales and Customers by Week
fig, (axis1, axis2)=plt.subplots(1, 2, figsize=(15, 7))

ax1=sns.barplot(x='DayOfWeek', y='Sales', data=df_train, ax=axis1)
ax2=sns.barplot(x='DayOfWeek', y='Customers', data=df_train, ax=axis2)
ax1.grid(axis='y')
ax2.grid(axis='y')


# Promo
fig, (axis1, axis2)=plt.subplots(1, 2, figsize=(15, 7))

ax1=sns.barplot(x='Promo', y='Sales', data=df_train, ax=axis1)
ax2=sns.barplot(x='Promo', y='Customers', data=df_train, ax=axis2)
ax1.grid(axis='y')
ax2.grid(axis='y')


#StateHoliday

# merge data '0' and 0 in StateHoliday
df_train['StateHoliday'].loc[df_train['StateHoliday']==0]='0'

# Plotting
fig, axis=plt.subplots(1, 1, figsize=(8, 5))
ax=sns.countplot(x='StateHoliday', data=df_train, ax=axis)

for col in ['Sales', 'Customers']:
    fig, (axis1, axis2)=plt.subplots(1, 2, figsize=(15, 6))
    ax1=sns.barplot(x='StateHoliday', y=col, data=df_train, ax=axis1)
    mask=(df_train['StateHoliday']!='0')&(df_train['Sales']>0)
    ax2=sns.barplot(x='StateHoliday', y=col, data=df_train[mask], ax=axis2)
    ax1.grid(axis='y')
    ax2.grid(axis='y')
    


# merge State Holiday to holiday or not
df_train['StateHoliday']=df_train['StateHoliday'].map({0:0, "0":0, "a":1, "b":1, "c":1})
df_test['StateHoliday']=df_test['StateHoliday'].map({0:0, "0":0, "a":1, "b":1, "c":1})

fig, (axis1, axis2)=plt.subplots(1, 2, figsize=(15, 7))
ax1=sns.barplot(x='StateHoliday', y='Sales', data=df_train, ax=axis1)
ax2=sns.barplot(x='StateHoliday', y='Customers', data=df_train, ax=axis2)
ax1.grid(axis='y')
ax2.grid(axis='y')


#SchoolHoliday
sns.set_style("whitegrid")
sns.countplot(x='SchoolHoliday', data=df_train)

fig, (axis1, axis2)=plt.subplots(1, 2, figsize=(15, 4))
ax1=sns.barplot(x='SchoolHoliday', y='Sales', data=df_train, ax=axis1)
ax2=sns.barplot(x='SchoolHoliday', y='Customers', data=df_train, ax=axis2)


# Sales
fig, (axis1, axis2)=plt.subplots(2, 1, figsize=(15, 8))
sns.boxplot([df_train['Customers']], whis=np.inf, ax=axis1, orient='h')
df_train['Sales'].plot(kind='hist', bins=70, xlim=(0,15000), ax=axis2)


#Using df_store

avr_sales_customers=df_train.groupby('Store')[['Sales', 'Customers']].mean()

df_sales_customers=pd.DataFrame({
                             'Sales':avr_sales_customers['Sales'],
                             'Customers':avr_sales_customers['Customers']})
df_store=pd.merge(df_sales_customers, df_store, on='Store')
df_store.head()


#StoreType
sns.countplot(x='StoreType', data=df_store, order=['a', 'b', 'c', 'd'])

fig, (axis1, axis2)=plt.subplots(1, 2, figsize=(15, 6))
sns.barplot(x='StoreType', y='Sales', data=df_store, ax=axis1, order=['a', 'b', 'c', 'd'])
sns.barplot(x='StoreType', y='Customers', data=df_store, ax=axis2, order=['a', 'b', 'c', 'd'])
plt.show()


#Assortment
sns.countplot(x='Assortment', data=df_store, order=['a', 'b', 'c'])

fig, (axis1, axis2)=plt.subplots(1, 2, figsize=(15, 6))
sns.barplot(x='Assortment', y='Sales', data=df_store, ax=axis1, order=['a', 'b', 'c'])
sns.barplot(x='Assortment', y='Customers', data=df_store, ax=axis2, order=['a', 'b', 'c'])
plt.show()


#Promo2
sns.countplot(x='Promo2', data=df_store)

fig, (axis1, axis2)=plt.subplots(1, 2, figsize=(15, 6))
sns.barplot(x='Promo2', y='Sales', data=df_store, ax=axis1)
sns.barplot(x='Promo2', y='Customers', data=df_store, ax=axis2)
plt.show()


#CompetitionDistance
df_store['CompetitionDistance'].fillna(df_store['CompetitionDistance'].median())

df_store.plot(kind='scatter', x='CompetitionDistance', y='Sales', figsize=(15, 4))
df_store.plot(kind='kde', x='CompetitionDistance', y='Sales', figsize=(15, 4))
plt.show()


# Correlation
store_piv=pd.pivot_table(df_train, values='Sales', index='Date', columns=['Store'], aggfunc='sum')
store_pct_change=store_piv.pct_change().dropna()
store_piv.head()


# .... continue Correlation

start_store=1
end_store=5

# using summation of sales values for each store
fig, (axis)=plt.subplots(figsize=(12, 5))
sns.heatmap(store_piv[list(range(start_store, end_store+1))].corr(), annot=True, linewidth=1)

# using percent change for each store
fig, (axis)=plt.subplots(figsize=(12, 5))
sns.heatmap(store_pct_change[list(range(start_store, end_store+1))].corr(), annot=True, linewidth=1)
plt.show()

