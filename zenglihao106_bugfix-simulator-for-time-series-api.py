import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import polars as pl
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import lightgbm
import torch


valid_from = 1577 # for private you should change to 1455 (1 year)


alltraindata = pl.scan_parquet("/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet")
valid_df = alltraindata.filter(pl.col("date_id")>=valid_from).collect()


valid_df = valid_df.with_columns(pl.Series(range(len(valid_df))).alias("row_id"),
                                pl.lit(True).alias("is_scored"))
len(valid_df)


valid_df.write_parquet("valid_df.parquet")


test_sample = pl.read_parquet("/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet/date_id=0/part-0.parquet")
test_sample.head(3)


valid_df = valid_df.select(test_sample.columns)


valid_df.head(3)


lag_sample = pl.read_parquet("/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet/date_id=0/part-0.parquet")
print(lag_sample.head(3))
train_sample = pl.read_parquet("/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id=0/part-0.parquet",n_rows=1)
responder_cols = [s for s in train_sample.columns if "responder" in s]

def makelag(date_id):
    """
    Making lag at the previout day

    Args:
    date_id (int): date_id at the previout day
    
    Returns:
    pl.dataframe
    """
    
    lag = alltraindata.filter(pl.col("date_id")==date_id).select(["date_id","time_id","symbol_id"] + responder_cols).collect()
    lag.columns = lag_sample.columns
    
    return lag    


os.makedirs("./debug/test.parquet",exist_ok=True)
os.makedirs("./debug/lags.parquet",exist_ok=True)


total_iterations = len(valid_df["date_id"].unique())
total_iterations


for num_days, df_per_day in tqdm(valid_df.group_by("date_id",maintain_order=True),total=total_iterations,desc="Processing"):
    
       
    day = num_days[0] - valid_from # date_id must start from 0.
    
    os.makedirs(f"./debug/test.parquet/date_id={day}",exist_ok=True)
    os.makedirs(f"./debug/lags.parquet/date_id={day}",exist_ok=True)
    
    lag = makelag(num_days[0] - 1)

    # lines to add
    df_per_day = df_per_day.with_columns(pl.lit(day).alias("date_id")).cast(test_sample.schema)
    lag = lag.with_columns(pl.lit(day).alias("date_id")).cast(lag_sample.schema)
    
    df_per_day.write_parquet(f"./debug/test.parquet/date_id={day}/part-0.parquet")
    lag.write_parquet(f"./debug/lags.parquet/date_id={day}/part-0.parquet")


# TODO
# def predict(...):...


import kaggle_evaluation.jane_street_inference_server


%%time

inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            './debug/test.parquet',
            './debug/lags.parquet',
        )
    )




