import pandas as pd
import numpy as np
import os
import gc
import sys
import copy
from pathlib import Path
from datetime import datetime, timedelta, date
import time
from dateutil.relativedelta import relativedelta 

import pyarrow.parquet as pq
import pyarrow as pa

from tqdm import tqdm


df = pd.read_parquet(r"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id=0/part-0.parquet")


df.info()


df.columns


target_cols = []
base_cols = []
feat_cols = []
for c in df.columns:
    if c.startswith('responder_'):
        target_cols.append(c)
    elif c.startswith('feature_'):
        feat_cols.append(c)
    else:
        base_cols.append(c)
print(f"target: {target_cols}\nfeatures: {feat_cols}\nbase: {base_cols}")


df['date_id'].nunique(), df['time_id'].nunique()


df['symbol_id'].value_counts().sort_index()


na_cnt = df.isna().sum()
na_cnt.sort_values(inplace=True, ascending=False)


keep_feats = na_cnt[(na_cnt.index.str.contains('feature_', regex=False)) & (na_cnt<100)].index.values.tolist()
print(keep_feats)


# responder_6
df[base_cols + keep_feats + target_cols]['symbol_id'].value_counts()


df.loc[df['symbol_id']==30, keep_feats]


sid= 30
keep_cols = base_cols + keep_feats + target_cols


train_folder = r"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet"
# next(os.walk(train_folder))[1]
all_train_files = []

if Path(train_folder).exists():
    all_train_files = list(Path(train_folder).rglob("*.parquet"))


train_dfs = []

for train_file in tqdm(all_train_files):
    df = pd.read_parquet(train_file)
    train_dfs.append(df.loc[df['symbol_id']==30, keep_cols].copy(deep=True))


df_all = pd.concat(train_dfs)
df_all.shape


del train_dfs
gc.collect()


df_all.info()


pq.write_table(pa.Table.from_pandas(df_all), f'{sid}.parquet', compression = 'GZIP')





