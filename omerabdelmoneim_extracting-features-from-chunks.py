# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


DIR = '/kaggle/input/pump-fun-graduation-february-2025/'


train = pd.read_csv(DIR + 'train.csv', index_col=0)
train.head(5)


test = pd.read_csv(DIR + 'test_unlabeled.csv', index_col=0)
test.head(5)


train['slots_to_graduation'] = train['slot_graduated'] - train['slot_min']
train.drop(columns=['slot_graduated'],inplace=True)
train.head(5)


chunk_names = []
for _, _, filenames in os.walk(DIR):
    for filename in filenames:
        if 'chunk' in filename:
            chunk_names.append(filename)
print(chunk_names)



from concurrent.futures import ProcessPoolExecutor
def read_and_process_chunk(file_path):
    # Read the CSV chunk
    chunk = pd.read_csv(file_path, dtype=dtype_dict, parse_dates=['block_time'])
    return chunk
dtype_dict = {
    'slot': 'int32',
    'tx_idx': 'int32',
    'base_coin_amount': 'int32',
    'quote_coin_amount': 'int32',
    'virtual_token_balance_after': 'int32',
    'virtual_sol_balance_after': 'int32',
    'provided_gas_fee': 'int32',
    'provided_gas_limit': 'int32',
    'fee': 'int32',
    'consumed_gas': 'int32',
    'direction':'category'
}
file_paths = [os.path.join(DIR, chunk_name) for chunk_name in chunk_names]
results = []
with ProcessPoolExecutor() as executor:
    # Map each file to the read_and_process_chunk function
    results = list(executor.map(read_and_process_chunk, file_paths))
data = pd.concat(results, ignore_index=True)
data.head(5)


data['buy'] = data['direction'] == 'buy'
data.info()


data.dtypes


train_test = pd.concat([train, test])
print("shape before: ", train_test.shape)
train_test =  train_test.merge(
        data[['slot', 'base_coin','block_time']],
        how='left',
        left_on=['slot_min', 'mint'],
        right_on=['slot', 'base_coin']
    ).drop(columns=['slot','base_coin']).drop_duplicates()
train_test.rename(columns={'block_time': 'creation_time'}, inplace=True)
print("shape after: ", train_test.shape)
train_test.head(5)


data['consumed_gas_percentage'] = data['consumed_gas'] / data['provided_gas_limit']
data['token_sol_after_balance_ratio'] = data['virtual_token_balance_after'] / data['virtual_sol_balance_after']
data['quote_to_balance_ratio'] = data['quote_coin_amount'] / data['virtual_token_balance_after']


exclude_columns = ['slot']
numeric_columns = data.select_dtypes(include=['number', 'bool']).columns.difference(exclude_columns)
agg_dict = {col: ['mean','sum','max'] for col in numeric_columns}
agg_dict['tx_idx'] = 'count'
agg_dict['slot'] = 'nunique'
agg_dict['block_time'] = 'nunique'
agg_dict['consumed_gas_percentage'] = 'mean'
agg_dict['token_sol_after_balance_ratio'] = 'mean'
agg_dict['quote_to_balance_ratio']  = 'mean'
agg_dict['buy'] = ['mean','sum']
coin_agg = data.groupby('base_coin').agg(agg_dict)
coin_agg.columns = ['_'.join(col).strip() for col in coin_agg.columns.values]
coin_agg.rename(columns={'buy_mean':'buy_percentage','buy_sum':'buy_count',
                         'tx_idx_count':'transaction_count',
                         'slot_nunique':'slot_count','block_time_nunique':'block_count'}, inplace=True)
coin_agg.reset_index(inplace=True)
coin_agg.head()


data_dated = data.merge(
    right=train_test[['mint', 'creation_time']],
    how='left',
    left_on='base_coin',
    right_on='mint'
).drop(columns=['mint']).rename(columns={'creation_time': 'coin_creation_time'})
data_dated['time_since_creation'] = data_dated['block_time'] - data_dated['coin_creation_time']
data_dated.head()


first_15_seconds = data_dated[data_dated['time_since_creation'].dt.total_seconds() <= 15]
first_15_seconds.head()


first_15_seconds_agg = first_15_seconds.groupby('base_coin').agg(agg_dict)
first_15_seconds_agg.columns = ['_'.join(col).strip() for col in first_15_seconds_agg.columns.values]
first_15_seconds_agg.rename(columns={'buy_mean':'buy_percentage','buy_sum':'buy_count',
                         'tx_idx_count':'transaction_count',
                         'slot_nunique':'slot_count','block_time_nunique':'block_count'}, inplace=True)
first_15_seconds_agg.columns = ["first_15_sec_" + col for col in first_15_seconds_agg.columns.values]
first_15_seconds_agg.reset_index(inplace=True)
first_15_seconds_agg.head()


all_data = train_test.merge(coin_agg,left_on='mint',right_on='base_coin',
                            how='left').drop(columns=['base_coin'])
print("shape: ", all_data.shape)
all_data.head(5)


all_data = all_data.merge(first_15_seconds_agg,left_on='mint',right_on='base_coin',
                            how='left').drop(columns=['base_coin'])
print("shape: ", all_data.shape)
all_data.head(5)


new_train=all_data[all_data['has_graduated'].notna()]

# Reorder columns
cols = new_train.columns.tolist()
cols.remove('slots_to_graduation')
cols.append('slots_to_graduation')
cols.remove('has_graduated')
cols.append('has_graduated')
new_train = new_train[cols]

print("train shape: ", train.shape)
print("new_train shape: ", new_train.shape)
new_train.head(5)


new_test=all_data[all_data['has_graduated'].isna()].drop(columns=['has_graduated','slots_to_graduation'])
print("test shape: ", test.shape)
print("new_test shape: ", new_test.shape)
new_test.head(5)


# Check for duplicate 'mint' values in new_train
train_duplicates_mint = new_train[new_train.duplicated(subset=['mint'])]
print("Duplicate rows in new_train based on 'mint':")
print(train_duplicates_mint.shape)

# Check for duplicate 'mint' values in new_test
test_duplicates_mint = new_test[new_test.duplicated(subset=['mint'])]
print("Duplicate rows in new_test based on 'mint':")
print(test_duplicates_mint.shape)



new_train.to_csv('train.csv',index=False)
print("train frame saved")
new_test.to_csv('test.csv',index=False)
print("test frame saved")

