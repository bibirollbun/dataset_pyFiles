import os
import re
import gc
import torch
import random
import ctypes

import numpy as np
import pandas as pd
import polars as pl

from psutil import Process
from os import path, walk, getpid


def CleanMemory():
    "This method cleans the memory off unused objects and displays the cleaned state RAM usage";

    gc.collect();
    libc.malloc_trim(0);
    pid        = getpid();
    py         = Process(pid);
    memory_use = py.memory_info()[0] / 2. ** 30;
    return f"\nRAM usage = {memory_use :.4} GB";


def reduce_memory_usage_pl(df):
    """ Reduce memory usage by polars dataframe {df} with name {name} by changing its data types.
        Original pandas version of this function: https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65 """
    print(f"Memory usage of dataframe is {round(df.estimated_size('mb'), 2)} MB")
    Numeric_Int_types = [pl.Int8,pl.Int16,pl.Int32,pl.Int64]
    Numeric_Float_types = [pl.Float32,pl.Float64]    
    for col in df.columns:
        try:
            col_type = df[col].dtype
            if col_type == pl.Categorical or col == 'case_id':
                continue
            c_min = df[col].min()
            c_max = df[col].max()
            if col_type in Numeric_Int_types:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df = df.with_columns(df[col].cast(pl.Int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df = df.with_columns(df[col].cast(pl.Int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df = df.with_columns(df[col].cast(pl.Int32))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df = df.with_columns(df[col].cast(pl.Int64))
            elif col_type in Numeric_Float_types:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df = df.with_columns(df[col].cast(pl.Float32))
                else:
                    pass
            # elif col_type == pl.Utf8:
            #     df = df.with_columns(df[col].cast(pl.Categorical))
            else:
                pass
        except:
            pass
    print(f"Memory usage of dataframe became {round(df.estimated_size('mb'), 2)} MB")
    _ = CleanMemory()
    return df 


path = "/kaggle/input/user-retention-prediction/"
libc = ctypes.CDLL("libc.so.6")


train = pl.read_csv(f"{path}train.csv").with_columns([
    pl.from_epoch(pl.col("Timestamp"), time_unit="s").dt.offset_by("8h").alias("oper_time")
]).sort(['ID', 'oper_time']).with_columns([
    pl.col("oper_time").dt.date().cast(pl.Datetime).alias("oper_date"),
    pl.col("oper_time").dt.hour().cast(pl.Int8).alias("oper_hour"),
]).with_columns([
    (pl.col("oper_time") - pl.col("oper_time").shift(1)).dt.total_seconds().fill_null(0).clip(0, 1e9).over(["ID", "oper_date"]).alias("oper_time_diff"),
    (pl.col("oper_time").shift(-1) - pl.col("oper_time")).dt.total_seconds().fill_null(0).clip(0, 1e9).over(["ID", "oper_date"]).alias("oper_time_diff2"),
    pl.col('ActionId').shift(1).fill_null('None').over(["ID", "oper_date"]).alias("last_ActionId"),
    pl.col('ActionId').shift(-1).fill_null('None').over(["ID", "oper_date"]).alias("next_ActionId")
])

train = reduce_memory_usage_pl(train)


train = train.with_columns(
    [
        pl.col('oper_hour').is_in([1, 2, 3, 4, 5, 6]).cast(pl.Int8).alias('if_night'),
        pl.concat_str([pl.col('last_ActionId'), pl.col('ActionId')], separator="-").alias('last_cur_id'),
        pl.concat_str([pl.col('last_ActionId'), pl.col('ActionId'), pl.col('next_ActionId')], separator="-").alias('last_cur_next_id'),
    ]
)


def get_label(df):
    label = df.group_by(['ID', 'oper_date']).agg(
         (pl.col("oper_time").max() - pl.col("oper_time").min()).dt.total_seconds().fill_null(0).clip(0, 1e9).alias('use_time'),
         pl.col("oper_time").count().alias('use_times')
    )
    label = label.with_columns(
        (pl.col("oper_date") + pl.duration(days=7)).alias("end_date")
    )
    result = (
        label.lazy()
        .join(
            label.lazy(),
            on="ID",
            how="inner",
            suffix="_future"
        )
        .filter(
            pl.col("oper_date_future").is_between(
                pl.col("oper_date"), 
                pl.col("end_date"),
                closed="right"
            )
        )
        .group_by(["ID", "oper_date"])
        .agg(
            pl.col('use_time').count().alias("label"), 
            pl.col('use_time').sum().alias("use_time"), 
             pl.col('use_times').sum().alias("use_times"), 
            ).sort(["ID", "oper_date"])
        .collect()
    )
    label = label.join(result, on=['ID', 'oper_date'], how='left')[['ID', 'oper_date', 'label', 'use_time', 'use_times']].with_columns([
        pl.col('label').fill_null(0),
        pl.col('use_time').fill_null(0),
        pl.col('use_times').fill_null(0)
    ])
    label = reduce_memory_usage_pl(label)

    return label


train_label = get_label(train)


train = train.join(
            train_label,
            on=["ID", 'oper_date'],
            how="inner",
            suffix="_future"
        )


def get_id_count(train, col):
    res = None
    for i in range(8):
        cur = train.filter(pl.col('label') == (7 - i)).group_by(col).agg(
            pl.col('label').count().alias(f'{col}_label_{7 - i}')
        )
        # cur = cur.with_columns(
        #     pl.col(f'{col}_label_{i}').rank()
        # )
        if i == 0:
            res = cur
        else:
            res = res.join(
            cur,
            on=col,
            how="left",
            suffix="_future"
        )

    res = res.fill_null(0).with_columns(
        *[(pl.col(f'{col}_label_{i}') / pl.col(f'{col}_label_{i}').sum()).alias(f'tf_{i}') for i in range(8)],
        pl.sum_horizontal(pl.col([f'{col}_label_{i}' for i in range(8)])).alias('sum')
    ).with_columns(
        
        *[(pl.col(f'tf_{i}') * pl.col(f'{col}_label_{i}') / pl.col('sum')).alias(f'tf_idf_{i}') for i in range(8)]
    )
    
    return res.select(pl.col([col] + [f'tf_idf_{i}' for i in range(8)]))


def get_id_count(train, col):
    res = train.filter(pl.col('label') == 0).group_by(col).agg(
            pl.col('label').count().alias(f'{col}_label_0')
        )
    res2 = train.filter(pl.col('label') > 0).group_by(col).agg(
            pl.col('label').count().alias(f'{col}_label_1')
        )
    res = res.join(
            res2,
            on=col,
            how="left",
            suffix="_future"
        )

    res = res.fill_null(0).with_columns(
        *[(pl.col(f'{col}_label_{i}') / pl.col(f'{col}_label_{i}').sum()).alias(f'tf_{i}') for i in range(2)],
        pl.sum_horizontal(pl.col([f'{col}_label_{i}' for i in range(2)])).alias('sum')
    ).with_columns(
        *[(pl.col(f'tf_{i}') * pl.col(f'{col}_label_{i}') / pl.col('sum')).alias(f'tf_idf_{i}') for i in range(2)]
    )
    
    return res.select(pl.col([col] + [f'tf_idf_{i}' for i in range(2)]))


train_data_selected = pl.col("oper_date").is_between(
                pl.col('oper_date').min(), 
                pl.col('oper_date').max() - pl.duration(days=7),
                closed="both"
            )


last_cur_id = get_id_count(train.filter(train_data_selected), 'last_cur_id')
last_cur_next_id = get_id_count(train.filter(train_data_selected), 'last_cur_next_id')


last_cur_id_most = []
for i in range(2):
    tmp = last_cur_id.sort(f'tf_idf_{i}', descending=True).head(10)['last_cur_id'].to_list()
    last_cur_id_most.append(tmp)


last_cur_next_id_most = []
for i in range(2):
    tmp = last_cur_next_id.sort(f'tf_idf_{i}', descending=True).head(10)['last_cur_next_id'].to_list()
    last_cur_next_id_most.append(tmp)


train = train.with_columns(
    *[pl.col('last_cur_id').is_in(last_cur_id_most[i]).cast(pl.Int8).alias(f'last_cur_id_tag_{i}') for i in range(2)],
    *[pl.col('last_cur_next_id').is_in(last_cur_next_id_most[i]).cast(pl.Int8).alias(f'last_cur_next_id_{i}') for i in range(2)],
)


train = train.with_columns(
    pl.col('oper_time').rank(method="ordinal", descending=True).alias("head").over(['ID', 'oper_date']),
    pl.col('oper_time').rank(method="ordinal", descending=True).alias("tail").over(['ID', 'oper_date'])
)


drop_cols = ['last_ActionId', 'next_ActionId', 'last_cur_id', 'last_cur_next_id']
selected_cols = [i for i in train.columns if i not in drop_cols]


train = reduce_memory_usage_pl(train)


train.select(pl.col(selected_cols)).write_parquet("train.parquet")




