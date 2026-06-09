import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
import numpy as np
import pandas as pd
import polars as pl

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

import tensorflow as tf

from keras import layers
from keras import Model
from keras import ops
from keras import regularizers

from sklearn.metrics import r2_score

from pathlib import Path
import gc

DATA_DIR = Path("/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet")


class DataProcessor:
    ALL_FEATURES = [f"feature_{i:02}" for i in range(79)]
    FEATURES = [x for x in ALL_FEATURES if x not in
     ["feature_09", "feature_10", "feature_11"]]
    # selected features from
    # https://github.com/evgeniavolkova/kagglejanestreet/blob/master/janestreet/data_processor.py
    SELECT_FEATURES = [
        'feature_06',
        'feature_04',
        'feature_07',
        'feature_36',
        'feature_60',
        'feature_45',
        'feature_56',
        'feature_05',
        'feature_51',
        'feature_19',
        'feature_66',
        'feature_59',
        'feature_54',
        'feature_70',
        'feature_71',
        'feature_72',
    ]
    CAT_FEATURES = ["feature_09", "feature_10", "feature_11"]
    RESPONDERS = [f"responder_{i}" for i in range(9)]
    FEATURE_09_DICT = {
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
    FEATURE_10_DICT = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 10: 7, 12: 8}
    FEATURE_11_DICT = {
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

    def __init__(self, df: pl.LazyFrame):
        self.df = df

    def reduce_memory_usage(self) -> list[pl.Expr]:
        expressions = [
            pl.col(pl.Float64).cast(pl.Float32),
            pl.col("date_id").cast(pl.Int16),
            pl.col("time_id").cast(pl.Int16),
            pl.col("symbol_id").cast(pl.Int8),
        ]
        return expressions

    def map_category(self) -> list[pl.Expr]:
        expressions = [
            pl.col('symbol_id').replace({i: i for i in range(39)}).fill_null(99),
            pl.col('feature_09').replace(DataProcessor.FEATURE_09_DICT).fill_null(99),
            pl.col('feature_10').replace(DataProcessor.FEATURE_10_DICT).fill_null(99),
            pl.col('feature_11').replace(DataProcessor.FEATURE_11_DICT).fill_null(99),
        ]
        return expressions

    def get_lag_responders(self) -> list[pl.Expr]:
        cols = [f"responder_{i}" for i in range(9)]
        expressions = [
            pl.col(col)
            .shift(i)
            .fill_null(0)
            .over('symbol_id', 'time_id')
            .alias(f"{col}_lag_{i}")
            for col in cols for i in [1]  # lags
        ]
        return expressions

    def get_temporal_features(self) -> list[pl.Expr]:
        expressions = [
            (pl.col('date_id') % 170).alias('day'),
            (pl.col('date_id') * 2 * np.pi / 170).sin()\
            .cast(pl.Float32).alias('date_sin'),
            (pl.col('date_id') * 2 * np.pi / 170).cos()\
            .cast(pl.Float32).alias('date_cos'),
        ]
        return expressions

    def get_tag_average(self) -> list[pl.Expr]:
        expressions = [
            pl.mean_horizontal(['feature_67', 'feature_68', 'feature_69'])\
            .alias('tag_11_mean'),
        ]
        return expressions

    def get_select_lag_features(self) -> list[pl.Expr]:
        group = ['symbol_id', 'date_id']
        expressions = [
            pl.col('feature_07').shift(2).fill_null(0)\
            .over(group).alias('feature_07_lag'),
            pl.col('feature_06').shift(1).fill_null(0)\
            .over(group).alias('feature_06_lag'),
            pl.col('feature_60').shift(2).fill_null(0)\
            .over(group).alias('feature_60_lag'),
        ]
        return expressions

    def get_lag_responder_stats_per_day(self) -> list[pl.Expr]:
        group = ["date_id", "symbol_id"]
        cols = ['responder_6_lag_1']
        expressions = []
        for col in cols:
            exprs = [
                pl.col(col).max().over(group).alias(f"{col}_max"),
                # pl.col(col).min().over(group).alias(f"{col}_min"),
                # pl.col(col).mean().over(group).alias(f"{col}_mean"),
                # pl.col(col).std().over(group).alias(f"{col}_std")
            ]
            expressions.extend(exprs)
        return expressions

    def get_stats_per_date_time(self) -> list[pl.Expr]:
        group = ["date_id", "time_id"]
        expressions = []
        for col in DataProcessor.SELECT_FEATURES:
            exprs =[
                pl.col(col).mean().over(group).alias(f"{col}_mean"),
                pl.col(col).std().over(group).alias(f"{col}_std"),
                # pl.col(col).skew().over(group).alias(f"{col}_skew"),
                # pl.col(col).kurtosis().over(group).alias(f"{col}_kurtosis")
            ]
            expressions.extend(exprs)
        return expressions

    def get_diff_means(self) -> list[pl.Expr]:
        expressions = [
            (pl.col(col) - pl.col(f"{col}_mean")).alias(f"{col}_diff_mean")
            for col in DataProcessor.SELECT_FEATURES
        ]
        return expressions

    def min_max_scaler(self, df: pl.DataFrame, columns: List) -> pl.DataFrame:
        for col in columns:
            col_min = df.select(col).min()
            col_max = df.select(col).max()
            df = df.with_columns(
                ((pl.col(col) - col_min) / ((col_max - col_min)+1e-10))
                .alias(col)
                )
        return df

    def generate_features(self) -> pl.DataFrame:
        exprs_1 = [
            self.reduce_memory_usage(),
            self.get_temporal_features(),
            self.get_tag_average(),
            self.get_select_lag_features(),
            self.get_stats_per_date_time(),
            # self.get_lag_responders(),
        ]
        expressions_1 = [e for sublist in exprs_1 for e in sublist]  # Flatten the list
        df = self.df.with_columns(expressions_1)

        exprs_2 = [
            self.map_category(),
            # self.get_lag_responder_stats_per_day(),
            [pl.col('time_id') / 968, pl.col('day') / 170]
        ]
        expressions_2 = [e for sublist in exprs_2 for e in sublist]
        df = df.with_columns(expressions_2)
        df = df.with_columns(self.map_category())

        ign_cols = [x for x in DataProcessor.RESPONDERS if x != 'responder_6']\
                  + ['partition_id'] \
                  # + [f"responder_{i}_lag_1" for i in range(9)] \
                  # + [f"{col}_mean" for col in DataProcessor.SELECT_FEATURES] \
                  # + [f"{col}_rmean" for col in DataProcessor.SELECT_FEATURES] \
        df = df.drop(ign_cols).fill_null(0)

        return df


# pl.scan_parquet(DATA_DIR).filter(pl.col('partition_id')>4).select('date_id').first().collect()


def get_initial_lags(df: pl.LazyFrame, date_id: int) -> pl.DataFrame:
    df_lags = (
        df.filter(pl.col('date_id')==date_id)
        .select(['time_id', 'symbol_id'] + [f"responder_{i}" for i in range(9)])
        .collect()
        )
    df_lags = df_lags.rename(dict(zip([f"responder_{i}" for i in range(9)],
                            [f"responder_{i}_lag_1" for i in range(9)])))
    return df_lags


df = pl.scan_parquet(DATA_DIR).filter(pl.col('date_id')>=700)
df_train = df.filter(pl.col('date_id') < 1572, pl.col('date_id')!=700)
df_valid = df.filter(pl.col('date_id') > 1576)

train_lags = get_initial_lags(df, 700)
valid_lags = get_initial_lags(df, 1576)


def get_embeddings(df: pl.LazyFrame, col: str, embd_dim: int):
    df_col = df.filter(pl.col('time_id')==0).select(col).collect()
    num_unique = df_col.n_unique()
    model = keras.Sequential() # keras embedding model
    model.add(keras.layers.Embedding(num_unique, embd_dim))
    model.compile('rmsprop', 'mse')
    embeddings = model.predict(df_col).reshape(-1, embd_dim)
    embeddings = pl.DataFrame(
        np.repeat(embeddings, 968, axis=0),
        schema=[f"{col}_embd_{i}" for i in range(embd_dim)]
        )
    return embeddings

df_symbol = get_embeddings(df, 'symbol_id', 20)
df_feat_09 = get_embeddings(df, 'feature_09', 11)
df_feat_10 = get_embeddings(df, 'feature_10', 9)
df_feat_11 = get_embeddings(df, 'feature_11', 15)

#  # one-hot encoding
# df_symbol = df.select('symbol_id').collect().to_dummies()
# df_feat_09 = df.select('feature_09').collect().to_dummies()
# df_feat_10 = df.select('feature_10').collect().to_dummies()
# df_feat_11 = df.select('feature_11').collect().to_dummies()

df_embeddings = pl.concat([df_symbol, df_feat_09, df_feat_10, df_feat_11],
                          how='horizontal')
df_embeddings.columns = [f"embd_{i}" for i in range(df_embeddings.shape[1])]
df_embeddings = pl.concat(
    [df.select('date_id').collect(), df_embeddings], how='horizontal'
    )

del df_symbol, df_feat_09, df_feat_10, df_feat_11
gc.collect()

print(df_embeddings.shape)


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(
        self, df: pl.LazyFrame, lags: pl.DataFrame, embeddings: pl.DataFrame,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.df = df
        self.lags = lags
        self.embeddings = embeddings
        self.date_ids = df.select('date_id').unique().sort('date_id').collect()
        self.targets = ['responder_6']

    def __len__(self):
        return len(self.date_ids) - 1

    def __getitem__(self, idx):
        RESPONDERS = [f"responder_{i}" for i in range(9)]
        LAG_RESPONDERS = [f"responder_{i}_lag_1" for i in range(9)]
        EMBEDDING_COLS = self.embeddings.columns
        TEMPORAL_COLS = ['time_id', 'day', 'date_sin', 'date_cos']
        IGNORED_COLS = ['date_id', 'symbol_id', 'feature_09', 'feature_10',
                       'feature_11', 'partition_id']

        date_id = self.date_ids[idx]
        df_cur = self.df.filter(pl.col('date_id') == date_id).collect()
        df_embeddings_cur = self.embeddings.filter(pl.col('date_id') == date_id)

        if idx > 0:
            date_id_prev = self.date_ids[idx - 1]
            df_prev_responders = (
                self.df.filter(pl.col('date_id') == date_id_prev)
                .select(['time_id', 'symbol_id'] + RESPONDERS)
                .collect()
                )
            df_prev_responders = df_prev_responders.rename(
                dict(zip(RESPONDERS, LAG_RESPONDERS)))
        else:
            df_prev_responders = self.lags

        df_cur = df_cur.join(
            df_prev_responders, on=['symbol_id', 'time_id'], how='left'
            )
        df_cur = pl.concat([df_cur, df_embeddings_cur.drop('date_id')], how='horizontal')

        processor = DataProcessor(df_cur)
        df_cur = processor.generate_features()

        df_cur = df_cur.fill_null(0)
        df_cur = df_cur.fill_nan(0)

        trend_data = df_cur.select(pl.all().exclude(IGNORED_COLS + RESPONDERS))
        scaled_cols = [col for col in trend_data.columns if col not in
                       EMBEDDING_COLS + TEMPORAL_COLS]
        trend_data = processor.min_max_scaler(trend_data, scaled_cols)
        num_feats = trend_data.shape[1]
        trend_data = trend_data.to_numpy().reshape(-1, 968, num_feats)
        target_data = df_cur.select(self.targets).to_numpy().reshape(-1, 968, 1)

        return trend_data, target_data

    @property
    def num_batches(self):
        return len(self)

train_generator = DataGenerator(df_train, train_lags, df_embeddings)
valid_generator = DataGenerator(df_valid, valid_lags, df_embeddings)


for x, y in train_generator:
    print(x.shape, y.shape)
    break


keras.config.set_dtype_policy("mixed_float16")

def build_model(input_sequence_length=968):
    inputs = layers.Input(shape=(input_sequence_length, 181))
    print("Expected Model Input Shape:", inputs.shape)

    x = layers.GRU(128, return_sequences=True)(inputs)
    # x = layers.Dropout(0.1)(x)
    # x = layers.GRU(128, return_sequences=True)(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.GRU(128)(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(input_sequence_length, activation='linear')(x)

    model = Model(inputs, outputs)
    # optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    return model

model = build_model()
model.summary()


%%time
callback = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2, restore_best_weights=True
        )

history = model.fit(
        train_generator,
        epochs=5,
        validation_data=(valid_generator),
        callbacks=[callback],
        )


preds = model.predict(valid_generator).reshape(-1)
print(preds.min())
print(preds.max())


def weighted_r2(y_true, y_pred, sample_weight):
    r2 = (1 - np.average((y_true - y_pred)**2, weights=sample_weight) /
          (np.average(y_true**2, weights=sample_weight) + 1e-38)
    )
    return r2


targets = []
for i in range(len(valid_generator)):
    targets.append(valid_generator[i][1].reshape(-1))
targets = np.concatenate(targets)

weights = []
for i in range(len(valid_generator)):
    weights.append(valid_generator[i][0].reshape(-1, 181)[:, 1])
weights = np.concatenate(weights)

weighted_r2(targets, preds, weights)




