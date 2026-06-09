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
import os


!pip install py7zr --quiet


import py7zr

base_path = "/kaggle/input/favorita-grocery-sales-forecasting"

files = [
    "train.csv.7z",
    "test.csv.7z",
    "stores.csv.7z",
    "items.csv.7z",
    "holidays_events.csv.7z",
    "transactions.csv.7z",
    "oil.csv.7z",
    "sample_submission.csv.7z"
]

for f in files:
    with py7zr.SevenZipFile(f"{base_path}/{f}", mode='r') as z:
        z.extractall("/kaggle/working")



train = pd.read_csv("/kaggle/working/train.csv")
test = pd.read_csv("/kaggle/working/test.csv")
stores = pd.read_csv("/kaggle/working/stores.csv")
items = pd.read_csv("/kaggle/working/items.csv")
holidays = pd.read_csv("/kaggle/working/holidays_events.csv")
transactions = pd.read_csv("/kaggle/working/transactions.csv")
oil = pd.read_csv("/kaggle/working/oil.csv")
sample_sub = pd.read_csv("/kaggle/working/sample_submission.csv")


train.head()


train.info()
train.describe()


test.head()


test.info()
test.describe()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc  # Garbage collection to manage RAM

base_path = '/kaggle/working/'

# ---------------------------------------------------------
# 1. INSPECT AUXILIARY FILES (Small & Easy to Load)
# ---------------------------------------------------------
print("--- Loading Auxiliary Files ---")

# Stores: crucial for picking a specific store or cluster
stores = pd.read_csv(base_path + 'stores.csv')
print(f"Stores shape: {stores.shape}")
print(stores.head())

# Items: contains 'family' and 'perishable' info - good for Multimodal/Embeddings
items = pd.read_csv(base_path + 'items.csv')
print(f"\nItems shape: {items.shape}")
print(items.head())

# Oil: Exogenous variable (economic health)
oil = pd.read_csv(base_path + 'oil.csv')
print(f"\nOil shape: {oil.shape}")

# Transactions: Footfall data
transactions = pd.read_csv(base_path + 'transactions.csv')
print(f"\nTransactions shape: {transactions.shape}")

# ---------------------------------------------------------
# 2. INTELLIGENT INSPECTION OF TRAIN.CSV (The Giant)
# ---------------------------------------------------------
print("\n--- analyzing train.csv (Chunking Method) ---")
print("Reading the first 5 rows to understand structure...")
train_head = pd.read_csv(base_path + 'train.csv', nrows=5)
print(train_head)

# We need to know which store has the most consistent data without loading 125M rows.
# We will iterate through the file in chunks.

chunksize = 5_000_000  # Process 5 million rows at a time
store_counts = {}
date_min = None
date_max = None
total_rows = 0

# Define types to save memory
dtypes = {
    'id': 'int32',
    'store_nbr': 'int8',
    'item_nbr': 'int32',
    'unit_sales': 'float32',
    'onpromotion': 'object' # Boolean but often contains NaNs or distinct values
}

print(f"\nStreaming train.csv to gather statistics (this may take a few minutes)...")

try:
    for chunk in pd.read_csv(base_path + 'train.csv', dtype=dtypes, chunksize=chunksize, parse_dates=['date']):
        # Update date range
        chunk_min = chunk['date'].min()
        chunk_max = chunk['date'].max()
        
        if date_min is None or chunk_min < date_min:
            date_min = chunk_min
        if date_max is None or chunk_max > date_max:
            date_max = chunk_max
            
        # Count rows per store in this chunk
        counts = chunk['store_nbr'].value_counts().to_dict()
        
        for store, count in counts.items():
            store_counts[store] = store_counts.get(store, 0) + count
            
        total_rows += len(chunk)
        print(f"Processed {total_rows} rows...", end='\r')

    print(f"\n\nTotal Rows: {total_rows}")
    print(f"Date Range: {date_min} to {date_max}")

except Exception as e:
    print(f"Error reading file (ensure it is unzipped): {e}")

# ---------------------------------------------------------
# 3. DECISION SUPPORT: WHICH STORE TO CHOOSE?
# ---------------------------------------------------------

# Convert store counts to DataFrame
store_stats = pd.DataFrame(list(store_counts.items()), columns=['store_nbr', 'total_records'])
store_stats = store_stats.sort_values('total_records', ascending=False)

# Merge with store metadata to see City/Type/Cluster
rich_store_stats = pd.merge(store_stats, stores, on='store_nbr')

print("\n--- Top 10 Stores by Data Volume ---")
print(rich_store_stats.head(10))

# Check Transactions volume for the top store to ensure it's active
top_store = rich_store_stats.iloc[0]['store_nbr']
print(f"\nChecking transaction history for Store {top_store}...")
top_store_trans = transactions[transactions['store_nbr'] == top_store]

plt.figure(figsize=(12, 4))
plt.plot(pd.to_datetime(top_store_trans['date']), top_store_trans['transactions'])
plt.title(f'Daily Transactions for Store {top_store}')
plt.xlabel('Date')
plt.ylabel('Transactions')
plt.show()


holidays.head(10)


import pandas as pd
import numpy as np
import gc
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
BASE_PATH = '/kaggle/working/'
OUTPUT_TRAIN_FILE = 'all_stores_train.csv'
OUTPUT_TEST_FILE = 'all_stores_test.csv'

# ---------------------------------------------------------
# 1. LOAD & PREP AUXILIARY DATA (Common to Train & Test)
# ---------------------------------------------------------
print("--- 1. Preparing Auxiliary Data ---")

# ITEMS
items = pd.read_csv(BASE_PATH + 'items.csv')

# OIL
oil = pd.read_csv(BASE_PATH + 'oil.csv')
oil['date'] = pd.to_datetime(oil['date'])
full_date_range = pd.date_range(start='2013-01-01', end='2017-08-31')
oil = oil.set_index('date').reindex(full_date_range)
oil['dcoilwtico'] = oil['dcoilwtico'].ffill().bfill()
oil = oil.reset_index().rename(columns={'index': 'date'})

# HOLIDAYS
holidays = pd.read_csv(BASE_PATH + 'holidays_events.csv')
holidays['date'] = pd.to_datetime(holidays['date'])
# Keep National + Local (Quito)
mask_national = holidays['locale'] == 'National'
mask_local = (holidays['locale'] == 'Local') & (holidays['locale_name'] == 'Quito')
holidays = holidays[mask_national | mask_local]
holidays = holidays[holidays['transferred'] == False]  # Remove transferred
holidays = holidays[['date', 'type', 'description']]
holidays.rename(columns={'type': 'holiday_type'}, inplace=True)
holidays = holidays.drop_duplicates(subset='date')  # Keep first if multiple

# TRANSACTIONS
transactions = pd.read_csv(BASE_PATH + 'transactions.csv')
transactions['date'] = pd.to_datetime(transactions['date'])

print("Auxiliary data ready.")

# ---------------------------------------------------------
# 2. HELPER FUNCTION: MERGE FEATURES
# ---------------------------------------------------------
def merge_features(df, is_train=True):
    """Merges Items, Oil, Holidays, and creates date features"""
    
    # Merge Items
    df = pd.merge(df, items, on='item_nbr', how='left')
    
    # Merge Oil
    df = pd.merge(df, oil, on='date', how='left')
    
    # Merge Holidays
    df = pd.merge(df, holidays, on='date', how='left')
    df['holiday_type'] = df['holiday_type'].fillna('Work Day')
    df['description'] = df['description'].fillna("No Holiday")
    
    # Date Features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    return df

# ---------------------------------------------------------
# 3. PROCESS TRAIN DATA (ALL STORES)
# ---------------------------------------------------------
print("\n--- 2. Processing Train Data for All Stores ---")

dtypes_train = {
    'id': 'int32',
    'store_nbr': 'int8',
    'item_nbr': 'int32',
    'unit_sales': 'float32',
    'onpromotion': 'object'
}

chunksize = 5_000_000
train_chunks = []

for chunk in pd.read_csv(BASE_PATH + 'train.csv', dtype=dtypes_train, chunksize=chunksize, parse_dates=['date']):
    # Filter 2016-2017 only
    chunk = chunk[(chunk['date'].dt.year >= 2016) & (chunk['date'].dt.year <= 2017)]
    
    if len(chunk) > 0:
        # Clip negative sales
        chunk['unit_sales'] = chunk['unit_sales'].clip(lower=0)
        
        # Onpromotion preprocessing
        chunk['onpromotion'] = chunk['onpromotion'].fillna(False)
        chunk['onpromotion'] = (chunk['onpromotion'].astype(str) == 'True').astype(int)
        
        train_chunks.append(chunk)
    print(f"Processing chunks...", end='\r')

# Combine all
train_df = pd.concat(train_chunks, axis=0)
del train_chunks
gc.collect()

# Merge features
train_df = merge_features(train_df, is_train=True)

# Merge transactions
train_df = pd.merge(train_df, transactions[['store_nbr', 'date', 'transactions']], 
                    on=['store_nbr','date'], how='left')
train_df['transactions'] = train_df['transactions'].fillna(0)

# Sort
train_df = train_df.sort_values(['store_nbr','item_nbr','date'])
print(f"\nTrain Shape: {train_df.shape}")
print(train_df.head(3))

# Save
train_df.to_csv(OUTPUT_TRAIN_FILE, index=False)
print(f"Saved {OUTPUT_TRAIN_FILE}")

# ---------------------------------------------------------
# 4. PROCESS TEST DATA (ALL STORES)
# ---------------------------------------------------------
print("\n--- 3. Processing Test Data for All Stores ---")

test_df = pd.read_csv(BASE_PATH + 'test.csv', parse_dates=['date'])
# Filter 2016-2017 if needed, though test is usually Aug 2017
test_df['onpromotion'] = test_df['onpromotion'].fillna(False).astype(int)

# Merge features
test_df = merge_features(test_df, is_train=False)

# Sort
test_df = test_df.sort_values(['store_nbr','item_nbr','date'])
print(f"Test Shape: {test_df.shape}")
print(test_df.head(3))

# Save
test_df.to_csv(OUTPUT_TEST_FILE, index=False)
print(f"Saved {OUTPUT_TEST_FILE}")


import pandas as pd
#, 48, 8, 49, 50, 11
TOP_STORES = [44, 47, 45, 46, 3]

train_df = pd.read_csv('/kaggle/working/all_stores_train.csv', parse_dates=['date'])

train_df = train_df[train_df['store_nbr'].isin(TOP_STORES)]

train_df = train_df.sort_values(['store_nbr', 'item_nbr', 'date'])

print("Filtered Train Shape:", train_df.shape)

train_df.to_csv('top5_stores_train.csv', index=False)
print("Saved top5_stores_train.csv")

test_df = pd.read_csv('/kaggle/working/all_stores_test.csv', parse_dates=['date'])

test_df = test_df[test_df['store_nbr'].isin(TOP_STORES)]

test_df = test_df.sort_values(['store_nbr', 'item_nbr', 'date'])

print("Filtered Test Shape:", test_df.shape)

test_df.to_csv('top5_stores_test.csv', index=False)
print("Saved top5_stores_test.csv")



train_df.info


import shutil
import os

input_file = '/kaggle/working/all_stores_train.csv'
zip_file = '/kaggle/working/all_stores_train.zip'

print("Zipping file...")
shutil.make_archive(
    base_name=zip_file.replace('.zip', ''),
    format='zip',
    root_dir='/kaggle/working',
    base_dir='all_stores_train.csv'
)

print(f"Zip created: {zip_file}")
print("Original size:", os.path.getsize(input_file) / 1e9, "GB")
print("Zip size:", os.path.getsize(zip_file) / 1e9, "GB")


train_df.tail(20)


test_df.head()




