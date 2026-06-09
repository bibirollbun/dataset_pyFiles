import gc
import polars as pl

train = pl.read_parquet(f"/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet/date_id=0/part-0.parquet")
display(train.head(20))

lag_cols_original = ["date_id", "symbol_id"] + [f"responder_{idx}" for idx in range(9)]
lag_cols_rename = { f"responder_{idx}" : f"responder_{idx}_lag_1" for idx in range(9)}

def add_lags(df):
    lags = df.select(pl.col(lag_cols_original))
    lags = lags.rename(lag_cols_rename)
    lags = lags.with_columns(date_id = pl.col('date_id') + 1,)  # lagged by 1 day
    lags = lags.group_by(["date_id", "symbol_id"], maintain_order=True).last()  # pick up last record of previous date
    return df.join(lags, on=["date_id", "symbol_id"], how="left")

df_train, df_valid = None, None
for i in range(0,10):
    df = pl.read_parquet(f"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}/part-0.parquet")
    df = df.with_columns(
        (pl.col("responder_6") * 2).cast(pl.Int32).alias("label"),        
        pl.col('date_id').cast(pl.Int64),
        pl.col('time_id').cast(pl.Int64),
    )
    time_id_max = df['time_id'].max()
    print(df['date_id'].min(), df['date_id'].max(), df['time_id'].min(), time_id_max,)
    training_cutoff = time_id_max - int(time_id_max // 5)
    print(df.shape, training_cutoff)

    train = df.filter(pl.col('time_id') < training_cutoff)
    valid = df.filter(pl.col('time_id') >= training_cutoff)
    if df_train is None:
        df_train = train
        df_valid = valid
    else:
        df_train = df_train.vstack(train)
        df_valid = df_valid.vstack(valid)
    del train, valid
    _ = gc.collect()

df_train = add_lags(df_train)
df_valid = add_lags(df_valid)

print(df_train.shape, df_valid.shape)
print(df_train.columns)
df_train.write_parquet("./train.parquet")
df_valid.write_parquet("./valid.parquet")


