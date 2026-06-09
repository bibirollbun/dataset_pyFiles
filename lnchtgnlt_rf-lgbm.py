import os
import numpy as np
import pandas as pd
import polars as pl
from typing import List
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

for dirname, _, filenames in os.walk("/kaggle/"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

KAGGLE = True  # define paths accordingly
SUBMISSION = True  # use smaller datasets during dev

if KAGGLE:
    crypto_folder = Path("/kaggle/input/drw-crypto-market-prediction")
else:
    crypto_folder = Path("../raw_data/crypto")


def get_clean_crypto_data(train: bool = True) -> pl.LazyFrame:
    """
    Load and clean crypto data, returning either train or test set.

    Args:
        train: If True, return training set. If False, return test set.

    Returns:
        Cleaned lazy frame with columns that have variance and no infinite values.
    """

    filename = "train.parquet" if train else "test.parquet"

    # load data
    crypto_lazy = pl.scan_parquet(crypto_folder / filename)
    n_cols = len(crypto_lazy.collect_schema().names())

    if train and KAGGLE:
        # rename timestamp column
        crypto_lazy = crypto_lazy.with_columns(
            pl.col("__index_level_0__").alias("timestamp")
        ).drop(["__index_level_0__"])

    # Remove columns with zero variance in the training set
    train_lazy = pl.scan_parquet(crypto_folder / "train.parquet")
    if KAGGLE:
        train_lazy = train_lazy.with_columns(
            pl.col("__index_level_0__").alias("timestamp")
        ).drop(["__index_level_0__"])

    # Get column names and calculate variance on training set (for consistency)
    crypto_var = train_lazy.select(pl.exclude(["timestamp"]).var())

    crypto_var_cols = (
        crypto_var.select(pl.all() == 0.0)
        .first()
        .collect()
        .to_pandas()
        .T.rename(columns={0: "is_variance_null"})
        .reset_index()
        .rename(columns={"index": "column_name"})
        .groupby("is_variance_null")["column_name"]
        .unique()
    )

    crypto_cols_with_var = crypto_var_cols[False]

    try:
        cols_no_var = crypto_var_cols[True]
        print(f"Columns with no variance : {cols_no_var}")
    except KeyError:
        print("All columns have variance in the train set")

    # remove columns that have no variance in the training set
    train_lazy = train_lazy.select(
        ["timestamp"] + [pl.col(c) for c in crypto_cols_with_var]
    )

    # Remove columns with infinite values (check on training set)
    current_columns = train_lazy.collect_schema().names()
    contains_infinite_cols = (
        train_lazy.select(pl.exclude("timestamp").abs().max().is_infinite())
        .collect()
        .to_pandas()
        .T.rename(columns={0: "contains_infinite"})
        .reset_index()
        .rename(columns={"index": "column_name"})
        .groupby("contains_infinite")["column_name"]
        .unique()
    )

    try:
        cols_with_inf_vals = contains_infinite_cols[True]
        print(f"Columns with infinite values : {cols_with_inf_vals}")
    except KeyError:
        print("No columns with infinite values")

    if not train:
        # add dummy timestamps
        crypto_lazy = crypto_lazy.with_columns(
            ID=range(1, crypto_lazy.select(pl.len()).collect().item() + 1)
        )
    # Filter clean columns based on what's available in the current dataset
    clean_columns = [
        c for c in current_columns if c in contains_infinite_cols[False]
    ] + ["timestamp", "ID"]
    available_columns = crypto_lazy.collect_schema().names()
    final_columns = [c for c in clean_columns if c in available_columns]
    print(f"Eventually {len(final_columns)}, removed {n_cols - len(final_columns)}")

    return crypto_lazy.select(final_columns)


def get_diff_features(df: pl.LazyFrame, stats_columns: List[str]):
    return (
        df.with_columns(pl.exclude(stats_columns).diff())
        .with_row_index()
        .fill_null(strategy="backward")
        .select(pl.exclude("index"))
    )

def get_ma_features(df: pl.LazyFrame, cols: List[str], ws:int=100):
    return (
        df.with_columns(
            pl.col(cols)
            .rolling_mean(
                window_size=ws, 
                min_samples=1
                )
        )
    )

def get_rolling_var(df: pl.LazyFrame, cols: List[str], ws:int=100):
    return (
        df.with_columns(
            pl.col(cols)
            .rolling_var(
                window_size=ws,
                min_samples=1
            ).backward_fill()
        )
    )


stats_columns = [
    "timestamp",
    "bid_qty",
    "ask_qty",
    "buy_qty",
    "sell_qty",
    "volume",
    "label",
]
stats_columns_test = [
    "ID",
    "bid_qty",
    "ask_qty",
    "buy_qty",
    "sell_qty",
    "volume",
    "label",
]
X_exclude = ["timestamp", "label"]
X_test_exclude = ["ID", "label"]


crypto_lazy_clean = get_clean_crypto_data(train=True)
cols = crypto_lazy_clean.collect_schema().names()




# join level with diff values
# crypto_lazy_clean = crypto_lazy_clean.join(
#     get_diff_features(crypto_lazy_clean, stats_columns),
#     on=stats_columns,
#     how="inner",
#     suffix="_diff",
# )

crypto_lazy_clean = crypto_lazy_clean.join(
    get_ma_features(crypto_lazy_clean, [c for c in cols if c not in stats_columns]),
    on=stats_columns,
    how="inner",
    suffix="_ma23",
)
print(len(crypto_lazy_clean.collect_schema().names()))
crypto_lazy_clean = crypto_lazy_clean.join(
    get_rolling_var(
        crypto_lazy_clean.select(cols), 
        [c for c in cols if c not in stats_columns]
        ).fill_null(0.),
    on=stats_columns,
    how="inner",
    suffix="_var23",
)
print(len(crypto_lazy_clean.collect_schema().names()))




X = crypto_lazy_clean.select(pl.exclude(X_exclude)).collect().to_numpy()
y = crypto_lazy_clean.select(pl.col("label")).collect().to_numpy().T[0]

if not SUBMISSION:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,  # TODO : question this, whether timestamps are independant draws
        random_state=42,
    )
else:
    X_train, y_train = X, y
del X
del y




lr = 1.

lin = RandomForestRegressor(
    # fit_intercept=True,
    n_estimators=80,
    n_jobs=-1,
    max_depth=10,
    min_samples_split=100,
    min_samples_leaf=50,
    max_features="sqrt",
    max_samples=0.5,
    random_state=41,
)
# n_samples = 80_000
lin.fit(
    X_train,
    y_train,
    # sample_weight=np.flip(1.0 / np.sqrt(np.arange(1, n_samples+1)))
)

y_train_lin = lin.predict(X_train)

print(f"R2 train lin: {r2_score(y_train, y_train_lin)}")
print(f"Pearson train lin : {pearsonr(y_train, y_train_lin)}")

y_train_res = y_train - lr * y_train_lin


lgb_model = lgb.LGBMRegressor(
    random_state=42,
    # weight=np.flip(1.0 / np.sqrt(np.arange(1, len(X_train)+1))),
    # n_estimators=80,
    # max_depth=10,
    n_jobs=-1,
)
lgb_model.fit(X_train, y_train_res)

y_train_hat = lgb_model.predict(X_train)

print(f"R2 train : {r2_score(y_train, y_train_hat + lr * y_train_lin)}")
print(f"Pearson train : {pearsonr(y_train, y_train_hat + lr * y_train_lin)}")

if not SUBMISSION:
    y_test_lin = lin.predict(X_test)

    print(f"R2 test lin : {r2_score(y_test, y_test_lin)}")
    print(f"Pearson test lin : {pearsonr(y_test, y_test_lin)}")

    y_test_hat = lgb_model.predict(X_test)

    print(f"R2 test : {r2_score(y_test, y_test_hat + lr * y_test_lin)}")
    print(f"Pearson test : {pearsonr(y_test, y_test_hat + lr * y_test_lin)}")

del y_train_lin
del y_train_res
del y_train_hat
del X_train

    


crypto_lazy_test = get_clean_crypto_data(train=False)
# create unique row identifier
# n = crypto_lazy_test.select(pl.len()).collect().item()
# crypto_lazy_test = crypto_lazy_test.with_columns(ID=range(1, n + 1))

# print(crypto_lazy_test.select(pl.len()).collect().item())

# crypto_lazy_test = crypto_lazy_test.join(
#     get_diff_features(crypto_lazy_test, stats_columns_test),
#     on=stats_columns_test,
#     how="inner",
#     suffix="_diff",
# )
cols_test = crypto_lazy_test.collect_schema().names()
crypto_lazy_test = crypto_lazy_test.join(
    get_ma_features(crypto_lazy_test, [c for c in cols_test if c not in stats_columns_test]),
    on=stats_columns_test,
    how="inner",
    suffix="_ma23",
)
crypto_lazy_test = crypto_lazy_test.join(
    get_rolling_var(
        crypto_lazy_test.select(cols_test), 
        [c for c in cols_test if c not in stats_columns_test]
        ),
    on=stats_columns_test,
    how="inner",
    suffix="_var23",
)

# crypto_lazy_test = get_diff_features(crypto_lazy_test, stats_columns_test)
# assert n == crypto_lazy_test.select(pl.len()).collect().item()


X_test = crypto_lazy_test.select(pl.exclude(X_test_exclude)).collect().to_numpy()
y_lin_test = lin.predict(X_test)
y_hat_lgb_test = lgb_model.predict(X_test)

del X_test


crypto_lazy_test = crypto_lazy_test.with_columns(
    prediction=y_hat_lgb_test + lr * y_lin_test
)
# crypto_lazy_test.head(5).collect()
crypto_lazy_test.select([pl.col("ID"), pl.col("prediction")]).collect().write_csv(
    Path("submission.csv")
)




