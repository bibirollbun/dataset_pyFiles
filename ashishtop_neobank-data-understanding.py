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


folder_path = '/kaggle/input/neo-bank-non-sub-churn-prediction/'
# List all files in the folder
all_files = os.listdir(folder_path)

##Filter out train files
train_files = [file for file in all_files if file.startswith('train') and file.endswith('.parquet')]
test_files = [file for file in all_files if file.startswith('test') and file.endswith('.parquet')]

# Read and concatenate train dataframes
train_dfs = [pd.read_parquet(os.path.join(folder_path, file)) for file in train_files]
train_df = pd.concat(train_dfs, ignore_index=True)

# Read and concatenate test dataframes
test_dfs = [pd.read_parquet(os.path.join(folder_path, file)) for file in test_files]
test_df = pd.concat(test_dfs, ignore_index=True)


train_df.describe().T


print('Number of records in train : ', train_df.shape[0])
print('Number of records in test : ', test_df.shape[0])


### Check various data types 
def check_datatypes(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    return categorical_cols, numerical_cols, date_cols


categorical_cols, numerical_cols, date_cols = check_datatypes(train_df)
print('categorical columns are as below' ,categorical_cols )
print('----------------------------------------------------')
print('Numerical columns are as below' ,numerical_cols )
print('----------------------------------------------------')
print('date columns are as below' , date_cols)


# Convert 'date_of_birth' column to datetime
train_df['date_of_birth'] = pd.to_datetime(train_df['date_of_birth'])
test_df['date_of_birth'] = pd.to_datetime(test_df['date_of_birth'])


## Add new column Age 
from datetime import datetime
# Calculate age and create a new column 'Age'
current_date = datetime.now()
train_df['age'] = train_df['date_of_birth'].apply(lambda dob: current_date.year - dob.year - ((current_date.month, current_date.day) < (dob.month, dob.day)))
train_df[['date_of_birth', 'age']].head()


categorical_cols, numerical_cols, date_cols = check_datatypes(train_df)


print('categorical columns are as below' ,categorical_cols )
print('----------------------------------------------------')
print('Numerical columns are as below' ,numerical_cols )
print('----------------------------------------------------')
print('date columns are as below' , date_cols)


def check_missing_values(df,categorical_cols, numerical_cols):
    missing_values = {
        'categorical': df[categorical_cols].isnull().sum().to_dict(),
        'numerical': df[numerical_cols].isnull().sum().to_dict()
    }
    return missing_values

# Check for missing values
missing_values = check_missing_values(train_df,categorical_cols, numerical_cols)

print("Missing values in categorical columns:", missing_values['categorical'])
print("Missing values in numerical columns:", missing_values['numerical'])


train_df[categorical_cols].head()


import matplotlib.pyplot as plt
import seaborn as sns


def plot_top_10_pie(df, column):
    top_10 = df[column].value_counts().nlargest(10)
    plt.figure(figsize=(4, 4))
    top_10.plot(kind='pie', autopct='%1.1f%%')
    plt.title(f'Top 10 {column.capitalize()} Distribution')
    plt.ylabel('')
    plt.show()


plot_top_10_pie(train_df, 'country')


plot_top_10_pie(train_df, 'job')


from ast import literal_eval
from datetime import datetime, timedelta


# Function to safely evaluate touchpoints
def safe_literal_eval(val):
    try:
        return literal_eval(val)
    except (ValueError, SyntaxError):
        return []

# Convert touchpoints from string representation of list to actual list
train_df['touchpoints'] = train_df['touchpoints'].apply(safe_literal_eval)

# Convert date column to datetime
train_df['date'] = pd.to_datetime(train_df['date'], format='%d-%m-%Y')

# Filter data for the last 24 months
current_date = datetime.now()
start_date = current_date - timedelta(days=24*30)
filtered_df = train_df[train_df['date'] >= start_date]

# Initialize column for touchpoint count
filtered_df['touchpoint_count'] = 0

# Count number of times touchpoints is not empty for each customer_id
for index, row in filtered_df.iterrows():
    if row['touchpoints']:
        filtered_df.at[index, 'touchpoint_count'] = 1

# Aggregate counts at customer_id level
result_df = filtered_df.groupby('customer_id')['touchpoint_count'].sum().reset_index()


train_df[numerical_cols].head()


num_col = ['interest_rate', 'atm_transfer_in', 'atm_transfer_out', 'bank_transfer_in', 'bank_transfer_out', 'crypto_in', 'crypto_out', 'bank_transfer_in_volume', 'bank_transfer_out_volume', 'crypto_in_volume', 'crypto_out_volume', 'complaints', 'tenure', 'age']
# Calculate correlation matrix
corr_matrix = train_df[num_col].corr()

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()


plot_top_10_pie(train_df, 'from_competitor')


# Calculate max tenure at customer_id level
max_tenure_df = train_df.groupby('customer_id')['tenure'].max().reset_index()

# Merge the max tenure back to the original DataFrame
train_df = train_df.merge(max_tenure_df, on='customer_id', suffixes=('', '_max'))

# Rename the new column to 'max_tenure'
train_df.rename(columns={'tenure_max': 'max_tenure'}, inplace=True)


# Filter data for the last 24 months
current_date = datetime.now()
start_date = current_date - timedelta(days=24*30)
filtered_df = train_df[train_df['date'] >= start_date]

# Initialize column for complaint count in the last 24 months
train_df['complaint_count_24month'] = 0

# Count number of complaints in the last 24 months for each customer_id
complaint_counts = filtered_df.groupby('customer_id')['complaints'].sum().reset_index()
complaint_counts.rename(columns={'complaints': 'complaint_count_24month'}, inplace=True)

# Merge the complaint counts back to the original DataFrame
train_df = train_df.merge(complaint_counts, on='customer_id', how='left', suffixes=('', '_24month'))

# Fill NaN values with 0 (for customers with no complaints in the last 24 months)
train_df['complaint_count_24month'].fillna(0, inplace=True)


train_df.drop(columns=['complaint_count_24month_24month'], inplace=True)


train_df.head()


plot_top_10_pie(train_df, 'complaint_count_24month')


# Calculate max and min interest rates at customer_id level
interest_stats = train_df.groupby('customer_id')['interest_rate'].agg(['max', 'min']).reset_index()
interest_stats.rename(columns={'max': 'max_interest', 'min': 'min_interest'}, inplace=True)

# Merge the max and min interest rates back to the original DataFrame
train_df = train_df.merge(interest_stats, on='customer_id', how='left')


train_df.head()




