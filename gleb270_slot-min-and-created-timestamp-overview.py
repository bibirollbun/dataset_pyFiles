import polars as pl
import pandas as pd


train = pl.read_csv('/kaggle/input/solana-skill-sprint-memcoin-graduation/train.csv')
train = train.filter(pl.col('is_valid'))

test = pl.read_csv('/kaggle/input/solana-skill-sprint-memcoin-graduation/test_unlabeled.csv')
test = test.filter(pl.col('is_valid'))


pump_fun_info = pl.read_parquet('/kaggle/input/pump-fun-api-solana-tokens-info/pump_fun_api_info.parquet')


train_joined = train.join(pump_fun_info, how='left', on='mint')
test_joined = test.join(pump_fun_info, how='left', on='mint')


pd.to_datetime(train_joined['created_timestamp'].min() * 1_000_000), pd.to_datetime(train_joined['created_timestamp'].max() * 1_000_000)


pd.to_datetime(test_joined['created_timestamp'].min() * 1_000_000), pd.to_datetime(test_joined['created_timestamp'].max() * 1_000_000)


train_joined['slot_min'].min(), train_joined['slot_min'].max()


test_joined['slot_min'].min(), test_joined['slot_min'].max()


import matplotlib.pyplot as plt


plt.scatter(train_joined['created_timestamp'], train_joined['slot_min']);


plt.scatter(test_joined['created_timestamp'], test_joined['slot_min']);

