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


import numpy as np
import pandas as pd
import joblib
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Data preprocessing and machine learning tools
from sklearn.model_selection import train_test_split  # Importing train_test_split
from sklearn.ensemble import RandomForestRegressor  # Random Forest Regressor model
from sklearn.preprocessing import OneHotEncoder  # For encoding categorical features
from sklearn.compose import ColumnTransformer  # For applying transformations to specific columns
from sklearn.metrics import mean_squared_error, r2_score  # Metrics for model evaluation



# Load the training and test datasets
sales_train_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')  # Change the path accordingly
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')  # Change the path accordingly


# Display the first few rows of the train dataset
print("Training Data:")
print(sales_train_data.head())


sales_train_data.shape


# Display the first few rows of the test dataset
test_df.head()


sales_train_data.info()


sales_train_data.isnull().sum()


sales_train_data.shape


test_df.info()


test_df.shape


test_df.isnull().sum()


sales_train_data = sales_train_data.dropna().reset_index(drop=True)


print("Unique Country Counts:")
print(sales_train_data['country'].value_counts())


print("\nUnique Product Counts:")
print(sales_train_data['product'].value_counts())


print("\nUnique Store Counts:")
print(sales_train_data['store'].value_counts())


print("\nUnique Date Counts:")
print(sales_train_data['date'].nunique())


print("\nNumerical Summary of 'num_sold':")
print(sales_train_data['num_sold'].describe())


print("\nTotal Sales by Date:")
total_sales_by_date = sales_train_data.groupby('date')['num_sold'].sum()
print(total_sales_by_date.head())


print("\nTotal Sales by Product:")
total_sales_by_product = sales_train_data.groupby('product')['num_sold'].sum()
print(total_sales_by_product.head())


# Date features extraction
sales_train_data['date'] = pd.to_datetime(sales_train_data['date'])
sales_train_data['year'] = sales_train_data['date'].dt.year
sales_train_data['month'] = sales_train_data['date'].dt.month
sales_train_data['day'] = sales_train_data['date'].dt.day
sales_train_data['weekday'] = sales_train_data['date'].dt.weekday
sales_train_data['is_weekend'] = sales_train_data['weekday'].isin([5, 6]).astype(int)


# Sales Lag Feature (1 day lag for sales)
sales_train_data['sales_lag_1'] = sales_train_data.groupby(['product', 'store'])['num_sold'].shift(1)


# Categorical Encodings: OneHot Encoding for 'country', 'store', and 'product'
encoder = ColumnTransformer(
    transformers=[
        ('country', OneHotEncoder(), ['country']),
        ('store', OneHotEncoder(), ['store']),
        ('product', OneHotEncoder(), ['product'])
    ], remainder='passthrough')


# Fit and transform the data
encoded_data = encoder.fit_transform(sales_train_data)


# Convert 'date' to datetime
test_df['date'] = pd.to_datetime(test_df['date'])

# Extracting features from the date
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day

# Optional: Add weekday and weekend features
test_df['weekday'] = test_df['date'].dt.weekday
test_df['is_weekend'] = test_df['weekday'].isin([5, 6]).astype(int)

# Check the updated DataFrame
test_df.head()



# 'sales_train_data' üzerinde toplam satışları tarih bazında görselleştirme
plt.figure(figsize=(12, 6))
sales_train_data.groupby('date')['num_sold'].sum().plot(kind='line', color='b', linewidth=2)
plt.title('Toplam Satışlar Zaman İçinde', fontsize=16)
plt.xlabel('Tarih', fontsize=12)
plt.ylabel('Toplam Satışlar', fontsize=12)
plt.grid(True)
plt.show()


# 'sales_train_data' üzerinde ürün bazında toplam satışları görselleştirme
plt.figure(figsize=(12, 6))
sales_train_data.groupby('product')['num_sold'].sum().sort_values(ascending=False).plot(kind='bar', color='c')
plt.title('Ürün Bazında Toplam Satışlar', fontsize=16)
plt.xlabel('Ürün', fontsize=12)
plt.ylabel('Toplam Satışlar', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.show()


# 'sales_train_data' üzerinde ülke bazında satış dağılımı
country_sales = sales_train_data.groupby('country')['num_sold'].sum()
plt.figure(figsize=(8, 8))
country_sales.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3", len(country_sales)))
plt.title('Ülkelere Göre Satış Dağılımı', fontsize=16)
plt.ylabel('')
plt.show()


# 'sales_train_data' üzerinde aylar bazında toplam satışları görselleştirme
sales_train_data['month'] = sales_train_data['date'].dt.month
plt.figure(figsize=(12, 6))
sales_train_data.groupby('month')['num_sold'].sum().plot(kind='bar', color='g')
plt.title('Aylar Bazında Toplam Satışlar', fontsize=16)
plt.xlabel('Ay', fontsize=12)
plt.ylabel('Toplam Satışlar', fontsize=12)
plt.xticks(rotation=0)
plt.show()


# 'sales_train_data' üzerinde haftalık satışları görselleştirme (Isı Haritası)
sales_train_data['weekday'] = sales_train_data['date'].dt.weekday
sales_train_data['week'] = sales_train_data['date'].dt.isocalendar().week

sales_weekly = sales_train_data.pivot_table(values='num_sold', index='weekday', columns='week', aggfunc='sum')

plt.figure(figsize=(16, 8))
sns.heatmap(sales_weekly, cmap='YlGnBu', annot=True, fmt='.1f', linewidths=0.5)
plt.title('Haftalık Satışlar (Isı Haritası)', fontsize=16)
plt.xlabel('Hafta', fontsize=12)
plt.ylabel('Haftanın Günü', fontsize=12)
plt.show()


# 'sales_train_data' üzerinde ürünlere göre satışların dağılımını görselleştirme
plt.figure(figsize=(12, 6))
sns.boxplot(data=sales_train_data, x='product', y='num_sold', palette="Set2")
plt.title('Ürün Bazında Satış Dağılımı', fontsize=16)
plt.xlabel('Ürün', fontsize=12)
plt.ylabel('Satış Sayısı', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.show()


# 'sales_train_data' üzerinde ülkelere göre zaman içindeki satışları görselleştirme
plt.figure(figsize=(12, 6))
for country in sales_train_data['country'].unique():
    country_data = sales_train_data[sales_train_data['country'] == country]
    country_data.groupby('date')['num_sold'].sum().plot(label=country)

plt.title('Zaman İçinde Ülkelere Göre Satışlar', fontsize=16)
plt.xlabel('Tarih', fontsize=12)
plt.ylabel('Toplam Satışlar', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# 'sales_train_data' üzerinde ürün ve ülke bazında satışları görselleştirme
plt.figure(figsize=(12, 6))
sales_by_product_country = sales_train_data.groupby(['product', 'country'])['num_sold'].sum().unstack()
sales_by_product_country.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set2')
plt.title('Ürün ve Ülkeye Göre Satışlar', fontsize=16)
plt.xlabel('Ürün', fontsize=12)
plt.ylabel('Toplam Satışlar', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.show()



# 'sales_train_data' üzerinde haftalık ortalama satışları görselleştirme
sales_train_data['week'] = sales_train_data['date'].dt.isocalendar().week
avg_weekly_sales = sales_train_data.groupby('week')['num_sold'].mean()

plt.figure(figsize=(12, 6))
avg_weekly_sales.plot(kind='line', color='r', linewidth=2)
plt.title('Haftalık Ortalama Satışlar', fontsize=16)
plt.xlabel('Hafta', fontsize=12)
plt.ylabel('Ortalama Satışlar', fontsize=12)
plt.grid(True)
plt.show()


# 'sales_train_data' üzerinde yıllık ürün satışlarını yığılmış bar grafiği ile görselleştirme
sales_train_data['year'] = sales_train_data['date'].dt.year
sales_by_product_year = sales_train_data.groupby(['year', 'product'])['num_sold'].sum().unstack()

plt.figure(figsize=(12, 6))
sales_by_product_year.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
plt.title('Yıllık Ürün Satışları', fontsize=16)
plt.xlabel('Yıl', fontsize=12)
plt.ylabel('Toplam Satışlar', fontsize=12)
plt.xticks(rotation=0)
plt.show()



# İlk 5 satırı görmek için
sales_train_data.head()


# Set 'date' as the index after converting it to datetime
sales_train_data['date'] = pd.to_datetime(sales_train_data['date'])  # Convert the 'date' column to datetime if not already in that format
sales_train_data.set_index('date', inplace=True)

# Extracting features from the date
sales_train_data['dayofweek'] = sales_train_data.index.dayofweek  # Day of the week (0 = Monday, 6 = Sunday)
sales_train_data['quarter'] = sales_train_data.index.quarter  # Quarter of the year
sales_train_data['week_of_year'] = sales_train_data.index.isocalendar().week  # Week number of the year
sales_train_data['day_of_month'] = sales_train_data.index.day  # Day of the month
sales_train_data['is_weekend'] = sales_train_data.index.dayofweek >= 5  # Weekend (Saturday and Sunday)


# Convert 'date' to datetime if not already in datetime format
test_df['date'] = pd.to_datetime(test_df['date'])

# Set 'date' as the index
test_df.set_index('date', inplace=True)

# Extracting features from the date
test_df['dayofweek'] = test_df.index.dayofweek  # Day of the week (0 = Monday, 6 = Sunday)
test_df['quarter'] = test_df.index.quarter  # Quarter of the year
test_df['week_of_year'] = test_df.index.isocalendar().week  # Week number of the year
test_df['day_of_month'] = test_df.index.day  # Day of the month
test_df['is_weekend'] = test_df.index.dayofweek >= 5  # Weekend (Saturday and Sunday)



# Dropping 'id' and 'num_sold' columns to separate features and target
X = sales_train_data.drop(columns=['id', 'num_sold'], axis=1)  # Features
y = sales_train_data['num_sold']  # Target


# Check for missing values in the features
X.isnull().sum()


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Check for any missing values in the split data
X_train.isnull().sum(), X_test.isnull().sum()


# Define the encoder (for one-hot encoding categorical columns)
encoder = ColumnTransformer(
    transformers=[
        ('country', OneHotEncoder(), ['country']),  # One-hot encoding for 'country'
        ('store', OneHotEncoder(), ['store']),      # One-hot encoding for 'store'
        ('product', OneHotEncoder(), ['product'])   # One-hot encoding for 'product'
    ],
    remainder='passthrough'  # Keep other columns as they are
)

# Fit and transform on the training data
X_train_encoded = encoder.fit_transform(X_train)

# Now, transform the test data using the same encoder
X_test_encoded = encoder.transform(X_test)


from sklearn.impute import SimpleImputer

# Define an imputer to fill missing values with the median of each column
imputer = SimpleImputer(strategy='median')

# Apply the imputer to the training and test sets
X_train_encoded = imputer.fit_transform(X_train_encoded)
X_test_encoded = imputer.transform(X_test_encoded)


from sklearn.ensemble import RandomForestRegressor

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_encoded, y_train)



# Make predictions on the test set
y_pred = model.predict(X_test_encoded)


# Evaluate the model

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")




