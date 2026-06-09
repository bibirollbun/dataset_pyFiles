import os
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

from catboost import CatBoostRegressor, EShapCalcType, EFeaturesSelectionAlgorithm, Pool
import kaggle_evaluation.jane_street_inference_server
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

DATA_DIR = Path("/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet")


feature_09_dict = {
    2: 0,
    4: 1,
    9: 2,
    11: 3,
    12: 4,
    14: 5,
    15: 6,
    25: 7,
    26: 8,
    30: 9,
    34: 10,
    42: 11,
    44: 12,
    46: 13,
    49: 14,
    50: 15,
    57: 16,
    64: 17,
    68: 18,
    70: 19,
    81: 20,
    82: 21
}
feature_10_dict = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 10: 7, 12: 8}
feature_11_dict = {
    9: 0,
    11: 1,
    13: 2,
    16: 3,
    24: 4,
    25: 5,
    34: 6,
    40: 7,
    48: 8,
    50: 9,
    59: 10,
    62: 11,
    63: 12,
    66: 13,
    76: 14,
    150: 15,
    158: 16,
    159: 17,
    171: 18,
    195: 19,
    214: 20,
    230: 21,
    261: 22,
    297: 23,
    336: 24,
    376: 25,
    388: 26,
    410: 27,
    522: 28,
    534: 29,
    539: 30
}


def reduce_memory_usage() -> pl.Expr:
    expressions = [
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col("date_id", "time_id").cast(pl.Int16),
        pl.col("symbol_id").cast(pl.Int8),
        ]
    return expressions

def map_category() -> pl.Expr:
    expressions = [
        pl.col('symbol_id').replace({i: i for i in range(39)}).fill_null(99),
        pl.col('feature_09').replace(feature_09_dict).fill_null(99),
        pl.col('feature_10').replace(feature_10_dict).fill_null(99),
        pl.col('feature_11').replace(feature_11_dict).fill_null(99),
    ]
    return expressions
    
def get_temporal_features() -> pl.Expr:
    expressions = [
        (pl.col('date_id') % 170).alias('day'),
        (pl.col('date_id') * 2 * np.pi / 170).sin().cast(pl.Float32).alias('date_sin'),
        (pl.col('date_id') * 2 * np.pi / 170).cos().cast(pl.Float32).alias('date_cos'),
        #(pl.col('time_id') * 2 * np.pi / 967).sin().cast(pl.Float32).alias('time_id_sin'),
        #(pl.col('time_id') * 2 * np.pi / 967).cos().cast(pl.Float32).alias('time_id_cos')
    ]
    return expressions

def get_lag_stats_per_day() -> pl.Expr:
    group = ["date_id", "symbol_id"]
    # cols = [f"responder_{i}_lag_1" for i in range(9)]
    cols = ['responder_6_lag_1']
    expressions = []
    for col in cols:
        exprs = [
            # pl.col(col).min().over(group).alias(f"{col}_min"),
            pl.col(col).max().over(group).alias(f"{col}_max"),
            # pl.col(col).std().over(group).alias(f"{col}_std")
            # pl.col(col).median().over(group).alias(f"{col}_median_per_day")
        ]
        expressions.extend(exprs)
    return expressions

def get_lag_features() -> pl.Expr:
    group = ['symbol_id', 'date_id']
    expressions = [
        pl.col('feature_07').shift(2).over(group).alias('feature_07_lag'),
        pl.col('feature_06').shift(1).over(group).alias('feature_06_lag'),
        pl.col('feature_60').shift(2).over(group).alias('feature_60_lag'),
        # pl.col('feature_04').shift(4).over(group).alias('feature_04_lag'),
        # pl.col('feature_05').shift(5).over(group).alias('feature_05_lag'),
        # pl.col('feature_36').shift(4).over(group).alias('feature_36_lag'),
        # pl.col('feature_58').shift(4).over(group).alias('feature_58_lag'),
        # pl.col('feature_59').shift(1).over(group).alias('feature_59_lag'),
        # pl.col('feature_38').shift(8).over(group).alias('feature_38_lag'),
        # pl.col('feature_52').shift(3).over(group).alias('feature_52_lag'),
    ]
    return expressions

def generate_features(df, df_lags):
    exprs_1 = [
        map_category(),
        get_temporal_features(),
        get_lag_features(),
    ]
    expressions = [e for sublist in exprs_1 for e in sublist]
    df = df.with_columns(reduce_memory_usage())
    df = df.with_columns(
        pl.col('feature_09').cast(pl.Int8), 
        pl.col('feature_10').cast(pl.Int8), 
        pl.col('feature_11').cast(pl.Int16), 
    )
    df = df.with_columns(expressions)
    
    df_lags = df_lags.with_columns(reduce_memory_usage())
    df_lags = df_lags.with_columns(get_lag_stats_per_day())
    df = df.join(
        df_lags,
        on=["date_id", "time_id", "symbol_id"], how="left"
    )
    ign_cols = [f"responder_{i}_lag_1" for i in range(9)] + ['date_id']
    return df.select(pl.all().exclude(ign_cols))


def get_lag_responders() -> pl.Expr:
    cols = [f"responder_{i}" for i in range(9)]
    expressions = [
        pl.col(col)
        .shift(i)
        .over('symbol_id', 'time_id')
        .alias(f"{col}_lag_{i}")
        for col in cols for i in [1] # lags
        ]
    return expressions

train = pl.scan_parquet(DATA_DIR).filter(pl.col('partition_id')>5)
train = train.with_columns(get_lag_responders()).collect()
# remove nulls resulting from day 1 lags
start_date = train.select('date_id')[0]
train = train.filter(pl.col('date_id') > start_date)
train.head()


data = train.filter(
    pl.col('date_id')>1576,
    pl.col('symbol_id')==0
    ).select('feature_07').to_numpy().reshape(-1)
plot_acf(data, lags = 10, title='Autocorrelation'); print()
plot_pacf(data, lags = 10, title='Partial Autocorrelation'); print()


features = [f"feature_{i:02}" for i in range(79)]
lag_responders = [f"responder_{i}_lag_1" for i in range(9)]

params = {
    'iterations': 1200,
    'learning_rate': 0.02,
    'depth': 8,
    'l2_leaf_reg': 5,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.9,
    'loss_function': 'RMSE',
    'eval_metric': 'MAE',
    'metric_period': 100,
    'od_type': 'Iter',
    'od_wait': 30,
    'task_type': 'GPU',
    'allow_writing_files': False,
    'use_best_model': False
}

model = CatBoostRegressor(**params)

x_train = generate_features(
    train.filter(pl.col('date_id')<1576).select(['date_id', 'time_id', 'symbol_id', 'weight'] + features),
    train.filter(pl.col('date_id')<1576).select(['date_id', 'time_id', 'symbol_id'] + lag_responders)
    ).to_numpy()
y_train = train.filter(pl.col('date_id')<1576).select('responder_6').to_numpy()

x_valid = generate_features(
    train.filter(pl.col('date_id')>1576).select(['date_id', 'time_id', 'symbol_id', 'weight'] + features),
    train.filter(pl.col('date_id')>1576).select(['date_id', 'time_id', 'symbol_id'] + lag_responders)
    ).to_numpy()
y_valid = train.filter(pl.col('date_id')>1576).select('responder_6').to_numpy()

model.fit(
    Pool(x_train, y_train),
    # verbose=False,
    eval_set=[(x_valid, y_valid)]
)


def weighted_r2(y_true, y_pred, sample_weight):
    r2 = (1 - np.average((y_true - y_pred)**2, weights=sample_weight) /
          (np.average(y_true**2, weights=sample_weight) + 1e-38)
    )
    return r2

r2_score = weighted_r2(
    y_valid.reshape(-1),
    model.predict(x_valid),
    train.filter(pl.col('date_id')>1576).select('weight').to_numpy().reshape(-1)
)
print("r2_score:",  r2_score.round(5))




