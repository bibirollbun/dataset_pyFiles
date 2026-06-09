import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm 

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")


import os
for dirname, _, filenames in os.walk('/kaggle/playground-series-s5e1'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train_data.sample(n=6)


test_data.sample(n=6)


test_data['date'] = pd.to_datetime(test_data['date'])


test_data.info()


train_data.shape


train_data.info()


train_data['date'] = pd.to_datetime(train_data['date'])


train_data.info()


train_data.describe(include='all')


print("      Stores")
print(" ")
print(train_data['store'].value_counts())
print('_'*30)
print("      Countries")
print(" ")
print(train_data['country'].value_counts())
print('_'*30)
print("      Products")
print(" ")
print(train_data['product'].value_counts())


train_data.isnull().any()


train_data['num_sold'].isnull().sum()


train_data.duplicated().any()


train_data[['num_sold']].boxplot()
plt.show()


sns.histplot(train_data['num_sold'], kde=True) 
plt.show()


median=train_data['num_sold'].median()
print("Median Value :",median)


mean=train_data['num_sold'].mean()
print("Mean Value :",mean)


na_indices = train_data['num_sold'].isnull()
fill_values = []
for i in range(len(train_data)):
    
    if na_indices[i]:
        if i % 2 == 0: # Even/odd rows
            fill_values.append(mean)
        else: fill_values.append(median)
    else:   
        fill_values.append(train_data['num_sold'][i]) # Original value

train_data['num_sold'] = fill_values

train_data[['num_sold']].astype('int64')


train_data = train_data[(train_data['num_sold'] > 0) & (train_data['num_sold'] < 2200)]



sns.histplot(train_data['num_sold'], kde=True,bins=30) 
plt.show()


train_data[['num_sold']].boxplot()
plt.show()


train_data.isnull().any()


train_data.duplicated().any()


train_data['num_sold'].describe()


train_data['month'] = train_data['date'].dt.month_name()
train_data['year'] = train_data['date'].dt.year


train_data.sample(n=6)


sales_by_product = train_data.groupby('product')['num_sold'].sum()

sales_by_product_sorted = sales_by_product.sort_values(ascending=False) 

norm = plt.Normalize(sales_by_product_sorted.min(), sales_by_product_sorted.max())

cmap = cm.plasma  # You can change to other colormaps like 'plasma', 'magma', 'inferno', 'cividis', etc.

plt.figure(figsize=(6, 5))

for i, (product, sales) in enumerate(sales_by_product_sorted.items()):
    color = cmap(norm(sales))  # Get color based on normalized sales
    plt.bar(x=product, height=sales, color=color)

plt.xlabel("Product")
plt.ylabel("Number Sold")
plt.title("Total Number Sold | Product")

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


total_sales_by_country = train_data.groupby('country')['num_sold'].sum()
plt.figure(figsize=(8,6))  
plt.pie(total_sales_by_country, labels=total_sales_by_country.index,explode=(0.05,0.05,0.05,0.5,0.05,0.05) ,
        shadow=True,autopct='%1.1f%%', startangle=90)
plt.title('Total Number Sold | Country')
plt.axis('equal')  
plt.show()


sales_by_country_store = train_data.groupby(['country', 'store'])['num_sold'].sum().reset_index()
plt.figure(figsize=(14, 7))
sns.barplot(x='country', y='num_sold', hue='store', data=sales_by_country_store, palette='viridis')
plt.title('Total Sales by Country and Store')
plt.xlabel('Country Name')
plt.ylabel('Total Number of Items Sold')
plt.legend(title='Store Name')
plt.tight_layout()
plt.show()


total_sales_by_store = train_data.groupby('store')['num_sold'].sum()
plt.figure(figsize=(6,6))  
plt.pie(total_sales_by_store, labels=total_sales_by_store.index,explode=(0.05,0.05,0.05) ,
        shadow=True,autopct='%1.1f%%', startangle=0)
plt.title('Total Number Sold | Store')
plt.axis('equal')  
plt.show()


total_sales_for_every_product = pd.pivot_table(train_data, values='num_sold', index='product', columns='year',
                            margins=True,margins_name='Total' ,aggfunc=np.sum)
total_sales_for_every_product


transposed=total_sales_for_every_product.transpose()
transposed.plot(kind='line', figsize=(14, 5),cmap='plasma')
plt.title('Total Number of Products Sold | years')
plt.xlabel('Year')
plt.ylabel('Total Number Sold')
plt.grid(True)
plt.legend(title='Product Name')
plt.show()


total_sales_for_every_store = pd.pivot_table(train_data, values='num_sold', index='store', columns='year',
                            margins=True,margins_name='Total' ,aggfunc=np.sum)
total_sales_for_every_store


transposed=total_sales_for_every_store.transpose()
transposed.plot(kind='line', figsize=(14, 4),cmap='viridis')
plt.title('Total Number sold | Stores | Years')
plt.xlabel('Year')
plt.ylabel('Total Number Sold')
plt.grid(True)
plt.legend(title='Store Name')
plt.show()


total_sales_in_countries = pd.pivot_table(train_data, values='num_sold', index='country', columns='year',
                            margins=True,margins_name='Total' ,aggfunc=np.sum)
total_sales_in_countries


yearly_sales = train_data.groupby(['year','month'])['num_sold'].sum()

yearly_sales.plot(kind='line', figsize=(19, 5))
plt.title('Total number sold | Years ')
plt.xlabel('Year')
plt.ylabel('Total Number Sold')
plt.grid(True)
plt.tight_layout()
plt.show()


month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

train_data['month_num'] = train_data['month'].map(month_map)

train_data = train_data.sort_values(by=['year', 'month_num'])

sales = train_data.groupby(['year', 'month_num'])['num_sold'].sum()

sales.index = pd.to_datetime(sales.index.map(lambda x: f'{x[0]}-{x[1]:02}-01'))

plt.figure(figsize=(20, 6))
plt.plot(sales)
plt.title('Detailed Overall Sales Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Total Number Sold')
plt.grid(True)
plt.tight_layout()
plt.show()


train_data


train_data2=train_data.copy()
train_data2


country_mapping = {country: idx + 1 for idx, country in enumerate(sorted(train_data2['country'].unique()))}
train_data2['country_maping'] = train_data2['country'].map(country_mapping)


product_mapping = {product: idx + 1 for idx, product in enumerate(sorted(train_data2['product'].unique()))}
train_data2['product_maping'] = train_data2['product'].map(product_mapping)


store_mapping = {store: idx + 1 for idx, store in enumerate(sorted(train_data2['store'].unique()))}

train_data2['store_mapping'] = train_data2['store'].map(store_mapping)


train_data2['country_maping'].unique()


train_data2.sample(n=5)


X = train_data2[['country_maping', 'store_mapping', 'product_maping', 'year','month_num']]
y = train_data2['num_sold']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} - MSE: {mse:.2f}")


rf_model = RandomForestRegressor(random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')


test_data.sample(n=5)


test_data['month_num'] = test_data['date'].dt.month
test_data['year'] = test_data['date'].dt.year


test_data2=test_data.copy()
test_data2


country_mapping = {country: idx + 1 for idx, country in enumerate(sorted(test_data2['country'].unique()))}
test_data2['country_maping'] = test_data2['country'].map(country_mapping)


product_mapping = {product: idx + 1 for idx, product in enumerate(sorted(test_data2['product'].unique()))}
test_data2['product_maping'] = test_data2['product'].map(product_mapping)


store_mapping = {store: idx + 1 for idx, store in enumerate(sorted(test_data2['store'].unique()))}
test_data2['store_mapping'] = test_data2['store'].map(store_mapping)


test_data2.sample(n=5)


X_test = test_data2[['country_maping', 'store_mapping', 'product_maping', 'year','month_num']]


y_pred = rf_model.predict(X_test)

test_data2["num_sold"] = y_pred


test_data2.sample(n=5)


test_data2["num_sold"] = y_pred.round(0).astype(int)


submit_df = test_data2[["id", "num_sold"]]


submit_df


submit_df.to_csv("submission.csv", index=False)

print("Submission file saved as 'submission.csv'.")




