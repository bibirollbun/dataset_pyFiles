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


# File paths
train_file = "/kaggle/input/santander-product-recommendation/train_ver2.csv.zip"
test_file = "/kaggle/input/santander-product-recommendation/test_ver2.csv.zip"

# Load the datasets
train_data = pd.read_csv(train_file, compression='zip')
test_data = pd.read_csv(test_file, compression='zip')

# Display the first few rows of the training dataset
train_data.head(), train_data.info()


# Display the first few rows and information about the dataset
print(train_data.head())
print(train_data.info())



# Check missing values and unique values in the 'age' column
print("Missing values in 'age':", train_data['age'].isna().sum())
print("Unique values in 'age':", train_data['age'].unique())



# Clean the 'age' column
train_data['age'] = pd.to_numeric(train_data['age'], errors='coerce')  # Convert to numeric, setting invalid to NaN
median_age = train_data['age'].median(skipna=True)  # Calculate median excluding NaN
train_data['age'] = train_data['age'].fillna(median_age)  # Fill NaN with median
train_data['age'] = train_data['age'].clip(lower=18, upper=100)  # Cap ages to a reasonable range

# Verify the cleaned 'age' column
print(train_data['age'].describe())
print(train_data['age'].unique())



# Check for missing values across all columns
missing_values = train_data.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
print(missing_values)



# Drop columns with near-complete missing values
columns_to_drop = ['conyuemp', 'ult_fec_cli_1t']
train_data = train_data.drop(columns=columns_to_drop)




# Optimize memory usage by downcasting numeric types
for col in train_data.select_dtypes(include=['float64']).columns:
    train_data[col] = pd.to_numeric(train_data[col], downcast='float')
for col in train_data.select_dtypes(include=['int64']).columns:
    train_data[col] = pd.to_numeric(train_data[col], downcast='integer')

# Impute 'renta' with median
train_data['renta'] = train_data['renta'].fillna(train_data['renta'].median())

# Impute categorical columns with mode, one by one to manage memory
categorical_columns = ['segmento', 'canal_entrada', 'sexo']
for col in categorical_columns:
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])

# Impute numeric columns with 0
numeric_columns = ['ind_nomina_ult1', 'ind_nom_pens_ult1']
for col in numeric_columns:
    train_data[col] = train_data[col].fillna(0)

# Save intermediate progress
train_data.to_csv('cleaned_train_data.csv', index=False)

# Verify if there are any missing values left
print("Total missing values after cleaning:", train_data.isnull().sum().sum())
print("Dataset memory usage (MB):", train_data.memory_usage(deep=True).sum() / 1024**2)



# Identify columns with remaining missing values
remaining_missing = train_data.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]
print(remaining_missing)



# Convert 'fecha_dato' to datetime and extract 'year' and 'month'
train_data['fecha_dato'] = pd.to_datetime(train_data['fecha_dato'], format='%Y-%m-%d')
train_data['year'] = train_data['fecha_dato'].dt.year
train_data['month'] = train_data['fecha_dato'].dt.month

# Verify the new columns
print(train_data[['fecha_dato', 'year', 'month']].head())



# Convert 'fecha_alta' to datetime and calculate customer seniority in months
train_data['fecha_alta'] = pd.to_datetime(train_data['fecha_alta'], errors='coerce')
train_data['customer_seniority'] = ((train_data['fecha_dato'] - train_data['fecha_alta']).dt.days // 30).clip(lower=0)

# Verify the new 'customer_seniority' column
print(train_data[['fecha_alta', 'fecha_dato', 'customer_seniority']].head())



# Identify product columns (columns ending with '_ult1')
product_columns = [col for col in train_data.columns if col.endswith('_ult1')]

# Sort data by customer ID and date to ensure chronological order
train_data = train_data.sort_values(['ncodpers', 'fecha_dato'])

# Create lag features for product columns
for col in product_columns:
    train_data[f'prev_{col}'] = train_data.groupby('ncodpers')[col].shift(1).fillna(0).astype(int)

# Verify a few lagged features
print(train_data[[product_columns[0], f'prev_{product_columns[0]}']].head())





