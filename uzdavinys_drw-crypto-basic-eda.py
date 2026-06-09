# data processing libraries
import numpy as np
import pandas as pd
import polars as pl

from datetime import datetime
import os

# for monitoring progress
from tqdm import tqdm

import seaborn as sns # plots for statistical analysis
import matplotlib.pyplot as plt # for data visualization

# define default colors for plots in notebook
from matplotlib import cycler
from matplotlib.colors import LinearSegmentedColormap
colors = ["#068D9D", "#53599A", "#607BB0", "#6D9DC5", "#77BECF", "#80DED9", "#AEECEF"]

plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=True, prop_cycle=cycler('color', colors))

SEED = 42


def reduce_mem_usage(dataframe, dataset):
    """
    Function taken from: https://www.kaggle.com/code/ravaghi/drw-crypto-market-prediction-ensemble
    """
    print('Reducing memory usage for:', dataset)
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    
    for col in dataframe.columns:
        col_type = dataframe[col].dtype

        c_min = dataframe[col].min()
        c_max = dataframe[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                dataframe[col] = dataframe[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                dataframe[col] = dataframe[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                dataframe[col] = dataframe[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                dataframe[col] = dataframe[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                dataframe[col] = dataframe[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)
            else:
                dataframe[col] = dataframe[col].astype(np.float64)

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print('--- Memory usage before: {:.2f} MB'.format(initial_mem_usage))
    print('--- Memory usage after: {:.2f} MB'.format(final_mem_usage))
    print('--- Decreased memory usage by {:.1f}%\n'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))

    return dataframe


%%time
df_train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
df_test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

df_train = reduce_mem_usage(df_train, "train")
df_test = reduce_mem_usage(df_test, "test")

df_train = df_train.reset_index()

proprietary_features = [col for col in df_train.columns if col.startswith('X')]
print(f"There are {len(proprietary_features)} anonymized market proprietary features.")

basic_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
print(f"There are {len(basic_features)} basic features.\n")

all_features = proprietary_features + basic_features
target = 'label'

# convert from pandas to polars
df_train = pl.from_pandas(df_train)
df_test = pl.from_pandas(df_test)

print(f"Train dataset contains {df_train.shape[0]} rows and {df_train.shape[1]} columns." )
print(f"Test dataset contains {df_test.shape[0]} rows and {df_test.shape[1]} columns." )


fig, ax = plt.subplots(2, 1, figsize=(18, 6), sharex=True)

ax[0].plot(df_train["timestamp"], df_train[target])
ax[0].set_ylabel(target)

ax[0].set_title("Train dataset")

ax[1].plot(df_train["timestamp"], np.cumsum(df_train[target]), color=colors[1])
ax[1].set_xlabel("timestamp")
ax[1].set_ylabel(f"{target} cumulative sum")

plt.tight_layout()


%%time
# import BTC historical data 
btc_data = pd.read_csv('../input/bitcoin-historical-data/btcusd_1-min_data.csv')
btc_data['Timestamp'] = [datetime.fromtimestamp(x) for x in btc_data['Timestamp']]

# filter out range
btc_data = btc_data.loc[btc_data['Timestamp'] >= "2023-03-01"]
btc_data = btc_data.loc[btc_data['Timestamp'] < "2024-03-01"].reset_index(drop=True)

btc_data["close_norm"] = btc_data["Close"] - btc_data["Close"].values[0]
btc_data["close_change"] = btc_data["Close"].pct_change()
btc_data["close_change"] = btc_data["close_change"].fillna(0)
btc_data["close_change"] *= 1000
btc_data.head()


fig, ax = plt.subplots(2, 1, figsize=(18, 6), sharex=True)

ax[0].plot(btc_data["Timestamp"], btc_data["close_change"])
ax[0].set_ylabel("pct change of close price")

ax[0].set_title("BTC historical price")

ax[1].plot(btc_data["Timestamp"], btc_data["close_norm"], color=colors[1])
ax[1].set_xlabel("timestamp")
ax[1].set_ylabel("norm. close price")

plt.tight_layout()


# compare charts
fig, ax = plt.subplots(2, 2, figsize=(18, 6), sharex=True)
ax = ax.flatten()

ax[0].set_title("Train dataset")
ax[0].plot(df_train["timestamp"], np.cumsum(df_train[target]), color=colors[0])
ax[0].set_xlabel("timestamp")
ax[0].set_ylabel(f"{target} cumulative sum")

ax[1].set_title("BTC historical price")
ax[1].plot(btc_data["Timestamp"], btc_data["close_norm"], color=colors[1])
ax[1].set_xlabel("timestamp")
ax[1].set_ylabel("norm. close price")

ax[2].plot(df_train["timestamp"], df_train[target])
ax[2].set_ylabel(target)

ax[3].plot(btc_data["Timestamp"], btc_data["close_change"], color=colors[1])
ax[3].set_ylabel("pct change of close price")

plt.tight_layout()


fig, ax = plt.subplots(2, 2, figsize=(18, 6))
ax = ax.flatten()

sns.boxplot(x=df_train[target], ax=ax[0])

sns.boxplot(x=btc_data["close_change"], ax=ax[1], color=colors[1])

sns.histplot(data=df_train, x=target, kde=True, ax=ax[2])
sns.histplot(data=btc_data, x="close_change", kde=True, ax=ax[3])

plt.tight_layout()


# Get count of null values per column
null_counts  = df_train.select(proprietary_features).null_count()

# Filter to show only columns with missing values
columns_with_nulls = null_counts.select([
    col for col in null_counts.columns 
    if null_counts[col].item() > 0
])

if columns_with_nulls.is_empty():
    print("There are no columns with NA values.")
else:
    print(columns_with_nulls)


%%time
single_unique_value = []
print(f"feature | unique value count")
for col in all_features:
    _cnt = df_train[col].n_unique()
    if _cnt < 10:
        single_unique_value.append(col)
        print(f"{col} | {_cnt}")

print(f"There are {len(single_unique_value)} features with single unique value.")


%%time
corr_data = df_train.select(all_features).corr()


%%time
columns = corr_data.columns
corr_long = []

for i, col1 in enumerate(columns):
    for j, col2 in enumerate(columns):
        # Only upper triangle to avoid duplicates
        if i < j:  
            corr_value = corr_data[col1][j]
            corr_long.append({
                'feature_1': col1,
                'feature_2': col2,
                'correlation': corr_value,
                'abs_correlation': abs(corr_value)
            })

corr_long = pd.DataFrame(corr_long).dropna().sort_values("correlation").reset_index(drop=True)
corr_long['abs_correlation_rounded'] = corr_long['abs_correlation'].round(2)
corr_long.head()


corr_long.tail()


_no_prairs = len(corr_long)
_df = corr_long.groupby("abs_correlation_rounded")['correlation'].count().tail(11).reset_index()
_df.columns = ["abs_correlation_rounded", "feature_pair_cnt"]
_df['share, %'] = (_df['feature_pair_cnt'] / _no_prairs * 100).round(3)
_df['cum. share, %'] = _df['share, %'].cumsum()
_df


_df = corr_long[corr_long["abs_correlation"] == 1]
print(f"{_df.shape[0]} feature pairs have perfect correlation.")


fig, ax = plt.subplots(6, 5, figsize=(18, 18))
ax = ax.flatten()

for i, idx in enumerate(_df.index):
    _col_x = _df.loc[idx, "feature_1"]
    _col_y = _df.loc[idx, "feature_2"]
    _corr = _df.loc[idx, "correlation"]

    _df_plot = df_train.select([_col_x, _col_y]).sample(1_000)

    ax[i].scatter(_df_plot.select(_col_x), _df_plot.select(_col_y))

    ax[i].set_title(f"Corr: {_corr:.3f}")
    ax[i].set_xlabel(_col_x)
    ax[i].set_ylabel(_col_x)
    
plt.tight_layout()


# Scater plot of TOP 20 most positive correlated features

# select feature pairs
_df = corr_long[corr_long["abs_correlation"].round(2) < 1]
_df = _df.tail(20)

fig, ax = plt.subplots(5, 4, figsize=(18, 16))
ax = ax.flatten()

for i, idx in enumerate(_df.index):
    _col_x = _df.loc[idx, "feature_1"]
    _col_y = _df.loc[idx, "feature_2"]
    _corr = _df.loc[idx, "correlation"]

    _df_plot = df_train.select([_col_x, _col_y]).sample(1_000)

    ax[i].scatter(_df_plot.select(_col_x), _df_plot.select(_col_y))

    ax[i].set_title(f"Corr: {_corr:.3f}")
    ax[i].set_xlabel(_col_x)
    ax[i].set_ylabel(_col_x)
    
plt.tight_layout()


# Scater plot of TOP 20 most negative correlated features

# select feature pairs
_df = corr_long[corr_long["abs_correlation"].round(2) < 1]
_df = _df.head(20)

fig, ax = plt.subplots(5, 4, figsize=(18, 16))
ax = ax.flatten()

for i, idx in enumerate(_df.index):
    _col_x = _df.loc[idx, "feature_1"]
    _col_y = _df.loc[idx, "feature_2"]
    _corr = _df.loc[idx, "correlation"]

    _df_plot = df_train.select([_col_x, _col_y]).sample(1_000)

    ax[i].scatter(_df_plot.select(_col_x), _df_plot.select(_col_y))

    ax[i].set_title(f"Corr: {_corr:.3f}")
    ax[i].set_xlabel(_col_x)
    ax[i].set_ylabel(_col_x)
    
plt.tight_layout()


# select feature pairs
_df = corr_long[corr_long["abs_correlation"].round(2) < 0.05]
_df = _df.sample(20)

fig, ax = plt.subplots(5, 4, figsize=(18, 16))
ax = ax.flatten()

for i, idx in enumerate(_df.index):
    _col_x = _df.loc[idx, "feature_1"]
    _col_y = _df.loc[idx, "feature_2"]
    _corr = _df.loc[idx, "correlation"]

    _df_plot = df_train.select([_col_x, _col_y]).sample(1_000)

    ax[i].scatter(_df_plot.select(_col_x), _df_plot.select(_col_y))

    ax[i].set_title(f"Corr: {_corr:.3f}")
    ax[i].set_xlabel(_col_x)
    ax[i].set_ylabel(_col_x)
    
plt.tight_layout()


%%time
corr_data = df_train.select(all_features + [target]).corr()
df_target_corr = pd.DataFrame({"feature": corr_data.columns, "corr": np.array(corr_data.select("label")).reshape(-1)})
df_target_corr = df_target_corr.dropna()
df_target_corr = df_target_corr.loc[df_target_corr["feature"] != "label"].reset_index(drop=True)
df_target_corr.tail()


# select feature pairs
_df = df_target_corr.sort_values("corr")
_df = _df.tail(20)
_df


fig, ax = plt.subplots(5, 4, figsize=(18, 16))
ax = ax.flatten()

_df_sample = df_train.sample(1_000)

for i, idx in enumerate(_df.index):
    _feat = _df.loc[idx, "feature"]
    _corr = _df.loc[idx, "corr"]

    ax[i].scatter(_df_sample.select(_feat), _df_sample.select(target))

    ax[i].set_title(f"Corr: {_corr:.3f}")
    ax[i].set_xlabel(_feat)
    ax[i].set_ylabel(target)
    
plt.tight_layout()


# select feature pairs
_df = df_target_corr.sort_values("corr")
_df = _df.head(20)
_df


fig, ax = plt.subplots(5, 4, figsize=(18, 16))
ax = ax.flatten()

_df_sample = df_train.sample(1_000)

for i, idx in enumerate(_df.index):
    _feat = _df.loc[idx, "feature"]
    _corr = _df.loc[idx, "corr"]

    ax[i].scatter(_df_sample.select(_feat), _df_sample.select(target))

    ax[i].set_title(f"Corr: {_corr:.3f}")
    ax[i].set_xlabel(_feat)
    ax[i].set_ylabel(target)
    
plt.tight_layout()


%%time
data = list()

for col in df_train.columns[6:]:
    _cnt = df_train.select(pl.col(col).round(4).n_unique()).item()
    data.append([col, _cnt])

df_unique_cnts = pd.DataFrame(data, columns = ["Feature", "n_unique"])
df_unique_cnts = df_unique_cnts.sort_values("n_unique").reset_index(drop=True)
df_unique_cnts.tail()


top_features = df_unique_cnts["Feature"][-10:]

fig, ax = plt.subplots(5, 2, figsize=(18, 16), sharex=True)
ax = ax.flatten()

for i, _feat in enumerate(top_features):

    ax[i].plot(df_train["timestamp"], df_train[_feat])
    ax[i].set_xlabel("Timestamp")
    ax[i].set_ylabel(_feat)
    
plt.tight_layout()


fig, ax = plt.subplots(5, 2, figsize=(18, 16))
ax = ax.flatten()

for i, _feat in enumerate(top_features):

    ax[i].plot(df_train["timestamp"], df_train[_feat].cum_sum())
    ax[i].set_xlabel("Timestamp")
    ax[i].set_ylabel(_feat)
    
plt.tight_layout()


folder_path = "/kaggle/input/crypto-currencies-daily-prices"
filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

df_crypto = pd.DataFrame()

for filename in tqdm(filenames):
    _df = pd.read_csv(f"{folder_path}/{filename}")
    _df['date'] = pd.to_datetime(_df['date'])

    # filter out train date range
    _df = _df.loc[_df['date'] >= "2023-03-01"]
    _df = _df.loc[_df['date'] < "2024-03-01"].reset_index(drop=True)

    df_crypto = pd.concat([df_crypto, _df]).reset_index(drop=True)

df_crypto.head()


_tickers = df_crypto["ticker"].sample(20)


fig, ax = plt.subplots(5, 4, figsize=(18, 16))
ax = ax.flatten()

for i, _ticker in enumerate(_tickers):

    _df_plot = df_crypto.loc[df_crypto["ticker"] == _ticker]

    ax[i].plot(pd.to_datetime(_df_plot.date), _df_plot["close"])
    ax[i].set_xlabel("date")
    ax[i].set_ylabel("close")
    ax[i].set_title(_ticker)
    
plt.tight_layout()


df_benckmark = df_train.select(["timestamp", "X363"])
df_benckmark = df_benckmark.to_pandas()
df_benckmark = df_benckmark.set_index('timestamp')
df_benckmark = df_benckmark.resample('D').agg({
    'X363': ['min', 'max', 'first', 'last']})
df_benckmark = df_benckmark.reset_index()
df_benckmark.columns = ['timestamp', 'low', 'high', 'open', 'close']
df_benckmark['cum_sum'] = df_benckmark['close'].cumsum()
df_benckmark.head()


data = list()

_tickers = df_crypto["ticker"].unique()

for i, _ticker in enumerate(_tickers):
    _df = df_crypto.loc[df_crypto["ticker"] == _ticker].copy()
    
    # join tables
    _df = pd.merge(
        df_benckmark[["timestamp", "cum_sum"]],
        _df[["date", "close"]],
        left_on='timestamp',
        right_on='date',
        how='left').dropna()
    
    # normalize
    # _df['cum_sum'] /= _df['cum_sum'].values[0]
    # _df['close'] /= _df['close'].values[0]
    
    _df['ratio'] = _df['cum_sum'] / _df['close']
    _corr = _df[["cum_sum", "close"]].corr()

    data.append([_ticker, _df['ratio'].std(), _corr.values[0][1]])

df_ratios = pd.DataFrame(data, columns=["ticker", 'ratio_std', 'corr']).sort_values(by="corr")
df_ratios.head()


df_ratios.tail()


# plot most matching prices

fig, ax = plt.subplots(3, 2, figsize=(18, 10))
ax = ax.flatten()

for i, _ticker in enumerate(df_ratios["ticker"].values[-5:]):

    _df_plot = df_crypto.loc[df_crypto["ticker"] == _ticker]

    ax[i].plot(pd.to_datetime(_df_plot.date), _df_plot["close"].values, label=f"{_ticker}")
    
    ax[i].set_xlabel("date")
    ax[i].set_ylabel("close")
    ax[i].set_title(_ticker)

ax[5].plot(pd.to_datetime(df_benckmark.timestamp), df_benckmark["cum_sum"], label="X363", color=colors[1])
ax[5].set_xlabel("date")
ax[5].set_ylabel("cum. sum of X363")
ax[5].set_title("Benchmark: X363 feature")

plt.tight_layout()




