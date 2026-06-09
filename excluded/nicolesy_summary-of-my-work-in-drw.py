import os
import random
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import re


from time import time
from tqdm import tqdm

import optuna

from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import make_scorer

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor

import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import math
from scipy.signal import savgol_filter
from sklearn.mixture import GaussianMixture


train = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
test = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")


def compress_memory(df: pd.DataFrame) -> pd.DataFrame:
    
    mem_before = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        if col == 'label':
            continue
        col_type = df[col].dtype
        if col_type == 'float64':
            df[col] = df[col].astype('float32')
        elif col_type == 'int64':
            df[col] = df[col].astype('int32')

    mem_after = df.memory_usage(deep=True).sum() / 1024**2

    print(f"Memory size before compression: {mem_before:.2f} MB")
    print(f"Memory size after compression: {mem_after:.2f} MB")
    print(f"Compression ratio: {mem_after / mem_before:.2%}")

    return df

train = compress_memory(train)
test = compress_memory(test)
print(train.shape)
print(test.shape)


def daily_row_count_summary(df):
    daily_counts = df.resample('D').size()
    print(len(daily_counts))
    count_of_days = daily_counts.value_counts().sort_index()
    summary_df = pd.DataFrame([count_of_days.values], columns=count_of_days.index)
    summary_df.index = ['Day Count']
    print(summary_df)
    # return summary_df

daily_row_count_summary(train)
# print(len(test)) # 538150 -> 538150/1440 = 373.71527777777777

def align_train_fill_missing(train):
    train = train.sort_index()

    full_idx = pd.date_range(start=train.index.min(), end=train.index.max(), freq='T')

    train_aligned = train.reindex(full_idx)

    train_aligned = train_aligned.ffill().bfill()
    
    return train_aligned

# print("Before alignment", len(train))
# train = align_train_fill_missing(train)
# print("After alignment", len(train)) # 527040 = 366*1440


def global_weighted_daily_sampling(train: pd.DataFrame,
                                   day_rows: int = 1440,
                                   total_days: int = 60,
                                   decay_rate: float = 3):

    n = len(train)
    total_full_days = n // day_rows
    if total_full_days < total_days:
        raise ValueError(f"The data is insufficient for sampling {total_days} day, only {total_full_days}day")

    day_start_indices = np.arange(total_full_days) * day_rows

    distance_from_end = total_full_days - 1 - np.arange(total_full_days)

    weights = np.exp(-decay_rate * distance_from_end)
    weights /= weights.sum()

    sampled_days = np.random.choice(total_full_days, size=total_days, replace=False, p=weights)

    sampled_days_sorted = np.sort(sampled_days)

    sampled_chunks = [train.iloc[start:start + day_rows].copy() for start in day_start_indices[sampled_days_sorted]]

    sampled_df = pd.concat(sampled_chunks, axis=0).reset_index(drop=True)

    return sampled_df

# recently_train = global_weighted_daily_sampling(train, day_rows=1440, total_days=60, decay_rate=3)
# print(recently_train.shape)  # (60 * 1440)


# Try to load precomputed timestamp reconstruction data
timestamp_recon_path = '/kaggle/input/the-order-of-the-test-rows-2/closest_rows.csv'
use_timestamp_reconstruction = os.path.exists(timestamp_recon_path)

if use_timestamp_reconstruction:
    print("Found timestamp reconstruction file, loading...")
    
    # Load precomputed timestamp reconstruction data
    t = pd.Series(pd.read_csv(timestamp_recon_path)['0'].to_numpy())
    assert t.shape == (test.shape[0],)
    print('Reconstructed timestamps share:', len(t[t >= 0]) / len(t))


    # Process timestamp reconstruction
    t -= 10080
    t[t < 0] = 538149

    t = t.sort_values()
    t[t <= len(t)] = np.arange(t[t <= len(t)].shape[0])
    t = t.sort_index()

    t = pd.Series(np.arange(538150), index=t.to_numpy()).sort_index()


    # Sort test dataset by reconstructed time order
    test = test.iloc[t.to_numpy()]

else:
    print("WARNING: Timestamp reconstruction file not found!")
    print(f"Expected path: {timestamp_recon_path}")
    print("Proceeding without timestamp reconstruction...")
    print("This may significantly impact model performance since lagged features assume temporal order.")
    
    t = pd.Series(np.arange(len(test)))

print(train.shape)
print(test.shape)


test_label = pd.DataFrame(index=test.index)
test_label["label"] = np.nan
test_label


def feature_engineering(df):
    cols_to_normalize = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    scaler = StandardScaler()

    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    
    # === Basic interaction features ===
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']
    df['buy_sell_interaction'] = df['buy_qty'] * df['sell_qty']

    # === Spread indicators ===
    df['spread_indicator'] = (df['ask_qty'] - df['bid_qty']) / (df['ask_qty'] + df['bid_qty'] + 1e-10)
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)

    # === Volume-weighted features ===
    df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['volume_weighted_bid'] = df['bid_qty'] * df['volume']
    df['volume_weighted_ask'] = df['ask_qty'] * df['volume']

    # === Ratio features ===
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)
    df['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-10)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-10)

    # === Buying and selling pressure indicators ===
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-10)

    # === Liquidity measurements ===
    df['total_liquidity'] = df['bid_qty'] + df['ask_qty']
    df['liquidity_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_liquidity'] + 1e-10)
    df['depth_imbalance'] = df['liquidity_imbalance']  # reused name
    df['total_depth'] = df['total_liquidity']  # reused name
    df['log_depth'] = np.log1p(df['total_depth'])

    # === Trade intensity and market activity ===
    df['trade_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-10)
    df['avg_trade_size'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['volume_participation'] = (df['buy_qty'] + df['sell_qty']) / (df['total_liquidity'] + 1e-10)
    df['market_activity'] = df['volume'] * df['total_liquidity']
    df['activity_intensity'] = df['trade_intensity']  # reused name

    # === Execution quality proxy indicators ===
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['realized_volatility_proxy'] = np.abs(df['order_flow_imbalance']) * df['volume']
    df['realized_spread_proxy'] = 2 * np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['price_impact_proxy'] = (df['buy_qty'] - df['sell_qty']) / (df['total_depth'] + 1e-10)

    # === Normalized volume features ===
    df['normalized_buy_volume'] = df['buy_qty'] / (df['bid_qty'] + 1e-10)
    df['normalized_sell_volume'] = df['sell_qty'] / (df['ask_qty'] + 1e-10)

    # === Complex interaction features ===
    df['liquidity_adjusted_imbalance'] = df['order_flow_imbalance'] * df['depth_ratio'] if 'depth_ratio' in df else df['order_flow_imbalance'] * (df['total_depth'] / (df['volume'] + 1e-10))
    df['pressure_spread_interaction'] = df['buying_pressure'] * df['spread_indicator']
    df['flow_depth_interaction'] = (df['buy_qty'] - df['sell_qty']) * df['total_depth']
    df['imbalance_volume_interaction'] = df['order_flow_imbalance'] * df['volume']
    df['depth_volume_interaction'] = df['total_depth'] * df['volume']

    # === Information asymmetry and market efficiency indicators ===
    df['trade_informativeness'] = (df['buy_qty'] - df['sell_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['execution_shortfall_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['adverse_selection_proxy'] = ((df['buy_qty'] - df['sell_qty']) / (df['total_depth'] + 1e-10)) * df['volume']

    df['fill_probability'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['execution_rate'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_efficiency'] = df['volume'] / (np.abs(df['bid_qty'] - df['ask_qty']) + 1e-10)

    # === Nonlinear transformations ===
    df['log_volume'] = np.log1p(df['volume'])
    df['log_buy_qty'] = np.log1p(df['buy_qty'])
    df['log_sell_qty'] = np.log1p(df['sell_qty'])
    df['log_bid_qty'] = np.log1p(df['bid_qty'])
    df['log_ask_qty'] = np.log1p(df['ask_qty'])

    df['sqrt_volume'] = np.sqrt(df['volume'])
    df['sqrt_depth'] = np.sqrt(df['total_depth'])
    df['volume_squared'] = df['volume'] ** 2
    df['imbalance_squared'] = df['order_flow_imbalance'] ** 2

    # === Relative ratio indicators ===
    df['bid_ratio'] = df['bid_qty'] / (df['total_depth'] + 1e-10)
    df['ask_ratio'] = df['ask_qty'] / (df['total_depth'] + 1e-10)
    df['buy_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['sell_ratio'] = df['sell_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)

    # === Market pressure and stress indicators ===
    df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_stress'] = df['volume'] / (df['total_depth'] + 1e-10) * np.abs(df['order_flow_imbalance'])
    df['depth_depletion'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)

    # === Directional indicators ===
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['net_buying_ratio'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['directional_volume'] = df['net_order_flow'] * np.log1p(df['volume'])
    df['signed_volume'] = np.sign(df['net_order_flow']) * df['volume']

    # --- Add advanced complex features ---
    required_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    if all(f in df.columns for f in required_cols):
        bid = df['bid_qty'].values
        ask = df['ask_qty'].values
        buy = df['buy_qty'].values
        sell = df['sell_qty'].values
        vol = df['volume'].values

        if 'kyle_lambda_complex' not in df.columns:
            order_imbalance = (bid - ask) / (bid + ask + 1e-6)
            flow_imbalance = (buy - sell) / (buy + sell + 1e-6)
            kyle_lambda = flow_imbalance * np.sqrt(np.abs(order_imbalance)) / (np.log1p(vol) + 1e-6)
            df['kyle_lambda_complex'] = kyle_lambda

        if 'vol_adjusted_pressure' not in df.columns:
            total_pressure = bid + ask
            vol_adj_pressure = np.log1p(total_pressure) * np.exp(-vol / (vol.mean() + 1e-6))
            df['vol_adjusted_pressure'] = vol_adj_pressure

        if 'trade_intensity_asymmetry' not in df.columns:
            buy_intensity = buy / (vol + 1e-6)
            sell_intensity = sell / (vol + 1e-6)
            intensity_asymmetry = np.sign(buy_intensity - sell_intensity) * np.log1p(np.abs(buy_intensity - sell_intensity))
            df['trade_intensity_asymmetry'] = intensity_asymmetry

        if 'bid_minus_ask' not in df.columns:
            bid_ask_diff = bid - ask
            df['bid_minus_ask'] = bid_ask_diff

        if 'volume_gaussian_kernel' not in df.columns:
            vol_kernel = np.exp(-((vol - vol.mean()) ** 2) / (2 * (vol.std() + 1e-6) ** 2))
            df['volume_gaussian_kernel'] = vol_kernel

    # === Replace infinite and missing values ===
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


train = feature_engineering(train)
test = feature_engineering(test)
print(train.shape)
print(test.shape)


base_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']

non_anonymous_features = [
    col for col in train.columns
    if not col.startswith('X') and col != 'label' and col not in base_cols
]

correlations = train[non_anonymous_features].corrwith(train['label'])
top_5_features = correlations.abs().sort_values(ascending=False).head(5)

print("Top 5 features with the highest absolute correlation with label:")
for feature, corr in top_5_features.items():
    print(f"{feature}: {corr:.4f}")


def replace_extreme_outliers_with_quantiles(df, columns, top_n=30, lower_q=0.2, upper_q=0.8):
    df_replaced = df.copy()
    for col in columns:

        lower_val = df[col].quantile(lower_q)
        upper_val = df[col].quantile(upper_q)

        min_indices = df[col].nsmallest(top_n).index
        max_indices = df[col].nlargest(top_n).index

        df_replaced.loc[min_indices, col] = lower_val

        df_replaced.loc[max_indices, col] = upper_val

    return df_replaced

train_df = replace_extreme_outliers_with_quantiles(train, non_anonymous_features, top_n=20, lower_q=0.1, upper_q=0.9)
test_df = replace_extreme_outliers_with_quantiles(test, non_anonymous_features, top_n=20, lower_q=0.1, upper_q=0.9)
print(train_df.shape)
print(test_df.shape)


def plot_two_features_for_four_dfs(df1, df2, df3, df4):

    dfs = [df1, df2, df3, df4]
    titles = ['train_original', 'train_after', 'test_original', 'test_after']
    features = ['normalized_buy_volume', 'bid_ask_ratio']
    
    for feature in features:
        print(feature)
        plt.figure(figsize=(16, 3))
        for i, df in enumerate(dfs):
            plt.subplot(1, 4, i + 1)
            plt.plot(df[feature].values, color='steelblue', linewidth=0.5)
            plt.title(titles[i], fontsize=10)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()

plot_two_features_for_four_dfs(train, train_df, test, test_df)
# del train, test


correlations = train_df[non_anonymous_features].corrwith(train_df['label'])
top_5_features = correlations.abs().sort_values(ascending=False).head(5)

print("Top 5 features with the highest absolute correlation with label:")
for feature, corr in top_5_features.items():
    print(f"{feature}: {corr:.4f}")


def apply_savgol_filter(df, columns_to_invert):
    
    window_length = 721
    polyorder = 3
    for feature in columns_to_invert:
        df[feature] = savgol_filter(df[feature], window_length=window_length, polyorder=polyorder)
    window_length = 121
    polyorder = 6
    for feature in columns_to_invert:
        df[feature] = savgol_filter(df[feature], window_length=window_length, polyorder=polyorder)
    
    return df

train_df = apply_savgol_filter(train_df, non_anonymous_features)
test_df = apply_savgol_filter(test_df, non_anonymous_features)

correlations = train_df[non_anonymous_features].corrwith(train_df['label'])
top_5_features = correlations.abs().sort_values(ascending=False).head(5)

print(train_df.shape)
print(test_df.shape)

print("Top 5 features with the highest absolute correlation with label:")
for feature, corr in top_5_features.items():
    print(f"{feature}: {corr:.4f}")


def select_top_features_by_rf(df: pd.DataFrame, label_col: str = 'label', top_n: int = 10, n_estimators: int = 30) -> list:

    X = df.drop(columns=[label_col])
    y = df[label_col]

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = rf.feature_importances_
    feature_names = X.columns

    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    imp_df = imp_df[imp_df['importance'] > 0]

    imp_df = imp_df.sort_values(by='importance', ascending=False).head(top_n)

    return imp_df['feature'].tolist()

# recently_train = train_df[non_anonymous_features + ["label"]].tail(50000)
# top_features = select_top_features_by_rf(recently_train, label_col='label', top_n=10, n_estimators=50)
# print("Top features:", top_features)
# del recently_train
top_features = ['imbalance_volume_interaction', 'net_order_flow', 'bid_ask_interaction', 'normalized_sell_volume', 'buy_sell_ratio', 'sqrt_depth', 'signed_volume', 'imbalance_squared', 'bid_sell_interaction', 'relative_spread']


gc.collect()


def process_anonymous_variables(
    df,                         
    label_col: str = "label",
    anon_regex: str = r"^X\d+$",
    norm_eps: float = 1e-9,
    uniq_thresh: int = 10,
    collinear_thresh: float = 0.9,
    var_quantile: float = 0.2,
    verbose: bool = True,
):
    # --- 0. Ready Polars DataFrame ------------------------------------------
    if isinstance(df, pl.DataFrame):
        pl_df = df.clone()
    else:
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df Must be pandas.DataFrame or polars.DataFrame")
        pl_df = pl.from_pandas(df)

    if label_col not in pl_df.columns:
        raise ValueError(f"column '{label_col}' doesn't in DataFrame")

    anon_cols = [c for c in pl_df.columns if re.match(anon_regex, c)]
    if not anon_cols:
        raise ValueError("No anonymous variables found")
    if verbose:
        print(f"The total number of original anonymous variables: {len(anon_cols)} col")

    # --- 1. Normalization ----------------------------------------------------------
    pl_df = pl_df.with_columns([
        ((pl.col(c) - pl.col(c).min()) /
         (pl.col(c).max() - pl.col(c).min() + norm_eps)).alias(c)
        for c in anon_cols
    ])

    # --- 2. Remove columns containing any infinity values ------------------
    cols_to_keep = []
    for c in anon_cols:
        col_series = pl_df[c]
        if col_series.is_infinite().any():
            if verbose:
                print(f"Drop column with infinite values: {c}")
            continue
        cols_to_keep.append(c)
    anon_cols = cols_to_keep
    if verbose:
        print(f"Step-1 After dropping infinite value columns, keep: {len(anon_cols)} col")

    # --- 3. Unique Filter -----------------------------------------------------
    nuniques_raw = (
        pl_df.select([pl.col(c).n_unique().alias(c) for c in anon_cols])
        .to_dict(as_series=False)
    )
    nuniques = {k: (v[0] if isinstance(v, list) else v) for k, v in nuniques_raw.items()}
    anon_cols = [c for c in anon_cols if nuniques[c] >= uniq_thresh]
    if verbose:
        print(f"Stepâ€‘2 Drop unique < {uniq_thresh}, keep: {len(anon_cols)} col")

    # --- 4. Variance Filter --------------------------------------------------------
    vars_raw = (
        pl_df.select([pl.col(c).var().alias(c) for c in anon_cols])
        .to_dict(as_series=False)
    )
    vars_dict = {k: (v[0] if isinstance(v, list) else v) for k, v in vars_raw.items()}
    var_values = np.array([vars_dict[c] for c in anon_cols])
    cutoff = np.quantile(var_values, var_quantile)
    anon_cols = [c for c in anon_cols if vars_dict[c] > cutoff]
    if verbose:
        print(f"Stepâ€‘3 Drop var lowest {var_quantile:.0%}, keep: {len(anon_cols)} col")

    # --- 5. NumPy --------------------------------------------------------
    anon_mat = pl_df.select(anon_cols).to_numpy()
    label_vals = pl_df.select(label_col).to_numpy().ravel()

    # --- 6. Spearman Multicollinearity Filter -------------------------------------------
    kept, dropped = [], set()
    for i, col_i in tqdm(
        enumerate(anon_cols),
        total=len(anon_cols),
        disable=not verbose,
        desc="Stepâ€‘4 Spearman Multicollinearity Filter",
        dynamic_ncols=True,
    ):
        if col_i in dropped:
            continue
        kept.append(col_i)
        xi = anon_mat[:, i]
        for j in range(i + 1, len(anon_cols)):
            col_j = anon_cols[j]
            if col_j in dropped:
                continue
            rho, _ = spearmanr(xi, anon_mat[:, j], nan_policy="omit")
            if abs(rho) > collinear_thresh:
                dropped.add(col_j)
    anon_cols = kept
    if verbose:
        print(f"Stepâ€‘4 Drop |Ï�_s|>{collinear_thresh}, keep: {len(anon_cols)} col")

    # --- result -------------------------------------------------------
    final_df = pl_df.to_pandas(use_pyarrow_extension_array=True)
    kept_cols = anon_cols
    other_cols = [c for c in final_df.columns if c not in kept_cols]
    final_df = final_df[other_cols + kept_cols]

    if verbose:
        print("âœ… The final retained anonymous variable columns:", kept_cols)

    return final_df

# recently_train = train.tail(50000)
# recently_train = process_anonymous_variables(recently_train, label_col="label")


def calc_mi_with_label(df, label_col='label', feature_prefix='X', eps=1e-9):
    feature_cols = [col for col in df.columns if col.startswith(feature_prefix)]
    
    X = df[feature_cols].copy()
    y = df[label_col]
    
    # Convert to float
    try:
        X_float = X.astype(float)
    except Exception as e:
        print("Failed to convert X to float:", e)
        raise
    
    # Replace positive and negative infinity with NaN
    X_float.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaNs forward then backward
    X_float.fillna(method='ffill', inplace=True)
    X_float.fillna(method='bfill', inplace=True)
    
    # Drop columns that still have NaNs
    cols_with_nan = X_float.columns[X_float.isna().any()].tolist()
    if cols_with_nan:
        X_float.drop(columns=cols_with_nan, inplace=True)
    else:
        print("No NaNs remain after filling.")

    # Convert y to float and check NaNs
    y_float = y.astype(float)
    if y_float.isna().any():
        raise ValueError("Input y contains NaN.")

    # Check for infinity values again
    if np.isinf(X_float.to_numpy()).any():
        raise ValueError("Input X contains infinite values.")
    if np.isinf(y_float.to_numpy()).any():
        raise ValueError("Input y contains infinite values.")

    # Compute mutual information
    mi_scores = mutual_info_regression(X_float.to_numpy(), y_float.to_numpy())
    mi_series = pd.Series(mi_scores, index=X_float.columns).sort_values(ascending=False)

    # Select columns with MI > 0.1
    high_mi_cols = mi_series[mi_series > 0.1].index.tolist()
    print(f"Number of columns with mutual information > 0.1: {len(high_mi_cols)}")
    print("Columns with mutual information > 0.1:", high_mi_cols)
    
    # Create final DataFrame with selected features + label
    final_df = pd.concat([X_float[high_mi_cols], y_float], axis=1)
    
    return final_df, high_mi_cols


# mi_recently_train, mi_list = calc_mi_with_label(recently_train, label_col='label')
# mi_recently_train.head(3)
# del recently_train, mi_recently_train
mi_list = ['X613', 'X140', 'X182', 'X98', 'X345', 'X429', 'X387', 'X590', 
           'X584', 'X758', 'X769', 'X385', 'X610', 'X427', 'X428', 'X179', 
           'X780', 'X95', 'X137', 'X612', 'X178', 'X611', 'X344', 'X779', 
           'X386', 'X181', 'X608', 'X138', 'X587', 'X342', 'X134', 'X176', 
           'X92', 'X572', 'X578', 'X581', 'X609', 'X339', 'X381', 'X423', 
           'X384', 'X96', 'X180', 'X768', 'X301', 'X302', 'X300', 'X426', 
           'X422', 'X296', 'X379', 'X298', 'X303', 'X131', 'X94', 'X299', 
           'X380', 'X136', 'X219', 'X294', 'X767', 'X466', 'X343', 'X292', 
           'X295', 'X421', 'X569', 'X605', 'X416', 'X761', 'X297', 'X173', 
           'X293', 'X172', 'X606', 'X89', 'X336', 'X290', 'X175', 'X566', 
           'X575', 'X338', 'X88', 'X560', 'X90', 'X139', 'X132', 'X420', 
           'X378', 'X778', 'X291', 'X465', 'X288', 'X174', 'X86', 'X170', 
           'X128', 'X425', 'X628', 'X375', 'X417', 'X333', 'X289', 'X373', 
           'X383', 'X629', 'X130', 'X682', 'X445', 'X374', 'X683', 'X772', 
           'X585', 'X286', 'X218', 'X332', 'X762', 'X169', 'X563', 'X125', 
           'X586', 'X337', 'X607', 'X415', 'X557', 'X588', 'X655', 'X166', 
           'X627', 'X614', 'X82', 'X626', 'X592', 'X287', 'X766', 'X97', 
           'X372', 'X738', 'X126', 'X654', 'X341', 'X83', 'X133', 'X508', 
           'X410', 'X330', 'X583', 'X591', 'X589', 'X777', 'X739', 'X167', 
           'X377', 'X284', 'X419', 'X124', 'X678', 'X679', 'X414', 'X84', 
           'X444', 'X757', 'X127', 'X217', 'X464', 'X711', 'X335', 'X573', 
           'X163', 'X168', 'X582', 'X38', 'X501', 'X710', 'X580', 'X285', 
           'X684', 'X35', 'X91', 'X577', 'X574', 'X571', 'X331', 'X651', 
           'X282', 'X603', 'X656', 'X198', 'X602', 'X674', 'X283', 'X54', 
           'X650', 'X42', 'X368', 'X36', 'X56', 'X371', 'X624', 'X675', 
           'X41', 'X39', 'X554', 'X44', 'X37', 'X548', 'X40', 'X46', 'X750', 
           'X45', 'X765', 'X53', 'X121', 'X85', 'X625', 'X760', 'X740', 
           'X50', 'X247', 'X413', 'X773', 'X52', 'X47', 'X25', 'X48', 'X55', 
           'X226', 'X404', 'X49', 'X157', 'X367', 'X706', 'X240', 'X507', 
           'X576', 'X443', 'X452', 'X43', 'X707', 'X205', 'X272', 'X51', 
           'X33', 'X326', 'X570', 'X562', 'X451', 'X735', 'X369', 'X712', 
           'X411', 'X500', 'X561', 'X327', 'X450', 'X680', 'X329', 'X122', 
           'X164', 'X80', 'X734', 'X494', 'X559', 'X487', 'X646', 'X197', 
           'X764', 'X565', 'X473', 'X685', 'X160', 'X463']


train_cols = base_cols + top_features + mi_list + ['label']
test_cols = base_cols + top_features + mi_list

train_cols = [col for col in train_cols if col in train_df.columns]
test_cols = [col for col in test_cols if col in test_df.columns]

train_df = train_df[train_cols].copy()
test_df = test_df[test_cols].copy()

print(train_df.shape)
print(test_df.shape)


def pca_on_anonymous_vars(train_data, test_data, feature_prefix='X', variance_threshold=0.95):
    anon_cols = [col for col in train_data.columns if col.startswith(feature_prefix)]
    non_anon_cols_train = [col for col in train_data.columns if not col.startswith(feature_prefix)]
    non_anon_cols_test = [col for col in non_anon_cols_train if col in test_data.columns]
    
    X_train_anon = train_data[anon_cols].astype(float)
    X_test_anon = test_data[anon_cols].astype(float)
    
    pca = PCA(n_components=variance_threshold)
    X_train_pca = pca.fit_transform(X_train_anon)
    
    n_components = pca.n_components_
    print(f"Number of PCA components to retain {variance_threshold*100}% variance: {n_components}")
    
    X_test_pca = pca.transform(X_test_anon)
    
    pca_cols = [f'PC{i+1}' for i in range(n_components)]
    train_pca_data = pd.DataFrame(X_train_pca, columns=pca_cols, index=train_data.index)
    test_pca_data = pd.DataFrame(X_test_pca, columns=pca_cols, index=test_data.index)
    
    train_final = pd.concat([train_data[non_anon_cols_train], train_pca_data], axis=1)
    test_final = pd.concat([test_data[non_anon_cols_test], test_pca_data], axis=1)
    
    return n_components, train_final, test_final


n_components, train_df, test_df = pca_on_anonymous_vars(train_df, test_df)
print("Components:", n_components)
print(train_df.shape)
print(test_df.shape)


trans_Anony = [col for col in train_df.columns if col.startswith("PC")]
correlations = train_df[trans_Anony].corrwith(train_df['label'])
top_5_features = correlations.abs().sort_values(ascending=False).head(5)

print("Top 5 features with the highest absolute correlation with label:")
for feature, corr in top_5_features.items():
    print(f"{feature}: {corr:.4f}")


# By lgbm importance:
pc_list = ["PC2", "PC13", "PC5", "PC46", "PC27", "PC43", "PC22", "PC40", "PC48", "PC6", "PC28", "PC7", "PC20", "PC23", "PC33", "PC3"]


# print(train_df.columns)


nn_input = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume',
       'imbalance_volume_interaction', 'net_order_flow', 'bid_ask_interaction',
       'normalized_sell_volume', 'buy_sell_ratio', 'sqrt_depth',
       'signed_volume', 'imbalance_squared', 'bid_sell_interaction',
       'relative_spread', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6',
       'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15',
       'PC16', 'PC17', 'PC18', 'PC19', 'PC20', 'PC21', 'PC22', 'PC23', 'PC24',
       'PC25', 'PC26', 'PC27', 'PC28', 'PC29', 'PC30', 'PC31', 'PC32', 'PC33',
       'PC34', 'PC35', 'PC36', 'PC37', 'PC38', 'PC39', 'PC40', 'PC41', 'PC42',
       'PC43', 'PC44', 'PC45', 'PC46', 'PC47', 'PC48', 'PC49', 'PC50']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train_full = train_df[nn_input].tail(420000).values.astype(np.float32)
y_train_full = train_df["label"].tail(420000).values.astype(np.float32).reshape(-1, 1)
X_test = test_df[nn_input].values.astype(np.float32)

split_idx = int(len(X_train_full) * 0.8)
X_train = X_train_full[:split_idx]
y_train = y_train_full[:split_idx]
X_val = X_train_full[split_idx:]
y_val = y_train_full[split_idx:]

# DataLoader
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Pearson Loss
def pearson_loss(pred, target):
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)
    pred_cent = pred - pred_mean
    target_cent = target - target_mean
    cov = torch.mean(pred_cent * target_cent)
    pred_var = torch.mean(pred_cent ** 2)
    target_var = torch.mean(target_cent ** 2)
    corr = cov / (torch.sqrt(pred_var) * torch.sqrt(target_var) + 1e-8)
    return 1 - corr

# MLP Model
class FeedForwardBottleneck(nn.Module):
    def __init__(self, input_dim=X_train.shape[1], dropout=0.3):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.bottleneck = nn.Linear(32, 16)
        self.regressor = nn.Linear(16, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.silu(self.layer2(x))
        x = self.dropout(x)
        x = F.silu(self.layer3(x))
        x = self.dropout(x)
        x = F.silu(self.layer4(x))
        x = self.dropout(x)
        z = F.silu(self.bottleneck(x))
        out = self.regressor(z)
        return out, z

# EarlyStopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"âš ï¸� EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# # Training
# model = FeedForwardBottleneck(input_dim=X_train.shape[1]).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
# epochs = 500
# early_stopper = EarlyStopping(patience=10)

# for epoch in range(epochs):
#     model.train()
#     total_loss = 0

#     for xb, yb in train_loader:
#         xb, yb = xb.to(device), yb.to(device)
#         optimizer.zero_grad()
#         preds, _ = model(xb)
#         mse = F.mse_loss(preds, yb)
#         p_loss = pearson_loss(preds, yb)
#         loss = 0.7 * p_loss + 0.3 * mse
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     avg_train_loss = total_loss / len(train_loader)

#     # val
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for xb, yb in val_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             preds, _ = model(xb)
#             mse = F.mse_loss(preds, yb)
#             p_loss = pearson_loss(preds, yb)
#             loss = 0.7 * p_loss + 0.3 * mse
#             val_loss += loss.item()
#     avg_val_loss = val_loss / len(val_loader)

#     print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

#     if early_stopper.best_loss is None or avg_val_loss < early_stopper.best_loss:
#         torch.save(model.state_dict(), "best_ff_model.pth")
#         print("âœ… Model improved. Saved to best_ff_model.pth")

#     early_stopper(avg_val_loss)
#     if early_stopper.early_stop:
#         print("â�¹ï¸� Early stopping triggered. Training stopped.")
#         break

# print("âœ… Training finished.")


model = FeedForwardBottleneck(input_dim=X_train.shape[1]).to(device)
model.load_state_dict(torch.load("/kaggle/input/nn-model/best_ff_model.pth", map_location=device))
model.eval()
print("âœ… Model loaded from best_ff_model.pth")


# result
@torch.no_grad()
def extract_bottleneck_features(model, data_tensor, batch_size=512):
    model.eval()
    preds_list = []
    z_list = []
    for i in range(0, len(data_tensor), batch_size):
        batch = data_tensor[i:i+batch_size].to(device)
        preds, z = model(batch)
        preds_list.append(preds.cpu())
        z_list.append(z.cpu())
    preds_all = torch.cat(preds_list).numpy()
    z_all = torch.cat(z_list).numpy()
    return preds_all, z_all

train_tensor = torch.from_numpy(train_df[nn_input].values.astype(np.float32))
test_tensor = torch.from_numpy(test_df[nn_input].values.astype(np.float32))

NN_train_preds, train_feats_np = extract_bottleneck_features(model, train_tensor)
NN_test_preds, test_feats_np = extract_bottleneck_features(model, test_tensor)


test_label_nn = test_label.copy()
test_label_nn["label"] = NN_test_preds
print(test_label_nn.head(5))
test_label_nn = test_label_nn.reset_index()
submit = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
submit['prediction'] = test_label_nn.set_index('ID').loc[submit['ID'], 'label'].values
submit = submit.sort_values(by='ID').reset_index(drop=True)
print(submit.head(5))
submit.to_csv("submission_nn.csv", index=False)


# Create a new feature column for bottleneck features (N0, N1, ...)
new_feat_cols = [f"N{i}" for i in range(train_feats_np.shape[1])]

# Directly add the new bottleneck features to the original DataFrames
for i, col in enumerate(new_feat_cols):
    train_df[col] = train_feats_np[:, i]  # Add bottleneck features to train_df
    test_df[col] = test_feats_np[:, i]    # Add bottleneck features to test_df

# Print the final shapes to check if the features are added correctly
print(train_df.shape)
print(test_df.shape)


# cols = [f'N{i}' for i in range(16)]
# zero_ratio = (train_df[cols] == 0).sum() / len(train_df)
# print("Train zero_ratio in NN features:", zero_ratio)

# cols = [f'N{i}' for i in range(16)]
# zero_ratio = (test_df[cols] == 0).sum() / len(test_df)
# print("Test zero_ratio in NN features:", zero_ratio)


new_trans_Anony = [col for col in train_df.columns if col.startswith("N")]
correlations = train_df[new_trans_Anony].corrwith(train_df['label'])
top_5_features = correlations.abs().sort_values(ascending=False).head(5)

print("Top 5 features with the highest absolute correlation with label:")
for feature, corr in top_5_features.items():
    print(f"{feature}: {corr:.4f}")


n_list = [f"N{i}" for i in range(16)]


train_cols = base_cols + top_features + pc_list + n_list + ['label']
test_cols = base_cols + top_features + pc_list + n_list

train_cols = [col for col in train_cols if col in train_df.columns]
test_cols = [col for col in test_cols if col in test_df.columns]

train_df = train_df[train_cols].copy()
test_df = test_df[test_cols].copy()

print(train_df.shape)
print(test_df.shape)


gc.collect()


def add_required_lag_features(train, test, train_df, test_df):
    
    # Original + new list of required features
    features_to_create = [
        'X198_lead_30', 'X179_lead_120', 'X197_lead_30',
        'X173_lead_120', 'X179_lead_150', 'X198', 'X40',
        'X445_lead_30', 'X175_lead_365', 'X181_lead_365',
        'X240_lead_30', 'X239_lead_40', 'X119_lead_30', 'X239_lead_80',
        'X113_lead_70', 'X239_lead_50', 'X623_lead_30', 'X623_lead_40',
        'X119_lead_80', 'X157_lead_30', 'X119_lead_50', 'X173_lead_80',
        'X215_lead_40', 'X239_lead_70', 'X623_lead_80', 'X743_lead_30',
        'X239_lead_60', 'X173_lead_40', 'X324_lead_30', 'X240_lead_60',
        'X113_lead_50', 'X743_lead_60', 'X743_lead_70', 'X198_lead_70',
        'X743_lead_50', 'X240_lead_70', 'X198_lead_50', 'X173_lead_60',
        'X173_lead_70', 'X157_lead_40', 'X198_lead_80', 'X198_lead_40',
        'X743_lead_80', 'X240_lead_50', 'X113_lead_30', 'X119_lead_70',
        'X240_lead_80', 'X240_lead_40', 'X239_lead_30'
    ]

    # Step 1: Separate lagged and direct (non-lagged) features
    lag_map = {}        # Format: {lag: [column1, column2, ...]}
    direct_cols = []    # Columns without lag

    for feat in features_to_create:
        if "_lead_" in feat:
            col, lag = feat.split("_lead_")
            lag = int(lag)
            lag_map.setdefault(lag, []).append(col)
        else:
            direct_cols.append(feat)

    # Step 2: Combine train and test for consistent lag computation
    full_df = pd.concat([train, test], axis=0, ignore_index=True)

    lagged_features = []

    # Step 3: Generate lagged features
    for lag, cols in lag_map.items():
        print(f"ğŸ“¦ Creating lagged features: lag={lag}, columns={cols}")
        lagged = full_df[cols].shift(-lag)
        lagged.columns = [f"{col}_lead_{lag}" for col in cols]
        lagged = lagged.fillna(0.0).astype(np.float32)
        lagged_features.append(lagged)

    # Step 4: Combine all lagged features
    lagged_df = pd.concat(lagged_features, axis=1)

    # Step 5: Split lagged features back to train/test
    train_lag = lagged_df.iloc[:len(train)].reset_index(drop=True)
    test_lag = lagged_df.iloc[len(train):].reset_index(drop=True)

    # Step 6: Extract direct features
    direct_train = train[direct_cols].reset_index(drop=True)
    direct_test = test[direct_cols].reset_index(drop=True)

    # Step 7: Concatenate everything
    train_df_final = pd.concat([train_df.reset_index(drop=True), direct_train, train_lag], axis=1)
    test_df_final = pd.concat([test_df.reset_index(drop=True), direct_test, test_lag], axis=1)

    # Step 8: Memory cleanup
    gc.collect()

    print("âœ… Feature construction completed.")
    print(f"train_df_final.shape = {train_df_final.shape}")
    print(f"test_df_final.shape = {test_df_final.shape}")

    return train_df_final, test_df_final

train_df, test_df = add_required_lag_features(train, test, train_df, test_df)


def append_lag_features(train, test, train_df, test_df, mi_list, lag_list, chunk_size=10):

    full_df = pd.concat([train[mi_list], test[mi_list]], axis=0, ignore_index=True)
    lag_features_all = []

    total_chunks = len(range(0, len(lag_list), chunk_size))
    print(f"total {total_chunks} chunks")

    for i in tqdm(range(0, len(lag_list), chunk_size), desc="Lag chunck process"):
        lag_chunk = lag_list[i:i+chunk_size]
        print(f"  â†’ creating lag feature: {lag_chunk}")
        
        for lag in tqdm(lag_chunk, leave=False, desc="  sub process"):
            lagged = full_df.shift(-lag)
            lagged.columns = [f"{col}_lead_{lag}" for col in lagged.columns]
            lagged = lagged.fillna(0.0).astype(np.float32)
            lag_features_all.append(lagged)

        gc.collect()

    all_lag_features = pd.concat(lag_features_all, axis=1)

    train_lagged = all_lag_features.iloc[:len(train)].reset_index(drop=True)
    test_lagged = all_lag_features.iloc[len(train):].reset_index(drop=True)

    train_df_aug = pd.concat([train_df.reset_index(drop=True), train_lagged], axis=1)
    test_df_aug = pd.concat([test_df.reset_index(drop=True), test_lagged], axis=1)

    return train_df_aug, test_df_aug


# train = train[mi_list]
# test = test[mi_list]

# lag_list = [1, 5, 15, 20, 30, 60, 120, 150, 300]
# chunk_size = 3

# train_df_lagged, test_df_lagged = append_lag_features(
#     train, test, train_df, test_df, mi_list, lag_list, chunk_size
# )


# train_cols = base_cols + top_features + pc_list + n_list + ['label']
# test_cols = base_cols + top_features + pc_list + n_list

# train_cols = [col for col in train_cols if col in train_df.columns]
# test_cols = [col for col in test_cols if col in test_df.columns]

# train_df_all = train_df[train_cols].copy()
# test_df_all = test_df[test_cols].copy()

# print(train_df_all.shape)
# print(test_df_all.shape)


X = train_df.drop(columns=['label'])
y = train_df['label']

split_index = int(len(X) * 0.8)
X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

def pearson_metric(y_true, y_pred):
    score, _ = pearsonr(y_true, y_pred)
    return 'pearson', score, True


X = train_df.drop(columns=['label'])
y = train_df['label']

split_index = int(len(X) * 0.8)
X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

def pearson_metric(y_true, y_pred):
    score, _ = pearsonr(y_true, y_pred)
    return 'pearson', score, True

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'None',  
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'n_estimators': 1000,
        'random_state': 42,
        'n_jobs': 1,
        'device': 'gpu',
    }

    try:
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=pearson_metric,
            callbacks=[lgb.early_stopping(50, verbose=False)]
            
        )

        preds = model.predict(X_val)
        score, _ = pearsonr(y_val, preds)

        print(f"Trial {trial.number}, Pearson score: {score:.4f}")

        del model, preds
        gc.collect()

        return float(score)

    except Exception as e:
        print(f"Trial {trial.number} failed with exception: {e}")
        gc.collect()
        return -1.0

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=20)

# print("Best parameters:", study.best_params)

# best_params = study.best_params

# best_params.update({
#     'objective': 'regression',
#     'metric': 'None',
#     'boosting_type': 'gbdt',
#     'n_estimators': 1000,
#     'n_jobs': 1,
#     'device': 'gpu',
#     'verbosity': -1
# })

best_params = {
    'objective': 'regression',
    'metric': 'None',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'n_jobs': 1,
    'device': 'gpu',
    'verbosity': -1,
    'learning_rate': 0.015865508582763952,
    'num_leaves': 67,
    'max_depth': 10,
    'min_child_samples': 66,
    'subsample': 0.6655741689043483,
    'colsample_bytree': 0.5974224259462245,
    'reg_alpha': 9.090859081454806e-06,
    'reg_lambda': 2.3603142329797598e-07
}


def custom_time_series_split(X, folds=5, val_ratio=0.2):
    n = len(X)
    fold_sizes = [n // folds] * folds
    for i in range(n % folds):
        fold_sizes[i] += 1
    
    fold_boundaries = []
    start = 0
    for size in fold_sizes:
        fold_boundaries.append((start, start + size))
        start += size

    splits = []
    for (start, end) in fold_boundaries:
        fold_len = end - start
        val_start = start + int(fold_len * (1 - val_ratio))

        train_idx = np.arange(0, val_start)
        val_idx = np.arange(val_start, end)

        splits.append((train_idx, val_idx))

    return splits


def cross_val_lgb_time_series_weighted(X_tr, y_tr, X_test, params, folds=5, val_ratio=0.2):
    splits = custom_time_series_split(X_tr, folds=folds, val_ratio=val_ratio)

    cv_scores, fold_preds = [], []
    feat_imp = np.zeros(X_tr.shape[1])
    feature_names = X_tr.columns.tolist()

    top30_nonanon_per_fold = []

    X_test = X_test.reindex(columns=X_tr.columns, fill_value=0.0)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_tr_fold, X_va_fold = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
        y_tr_fold, y_va_fold = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]

        print(f"[Fold {fold+1}] Train idx: {tr_idx[0]}-{tr_idx[-1]}, Val idx: {va_idx[0]}-{va_idx[-1]}")

        model = lgb.LGBMRegressor(**params, random_state=42 + fold)
        model.fit(
            X_tr_fold, y_tr_fold,
            eval_set=[(X_va_fold, y_va_fold)],
            eval_metric=pearson_metric,
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        fold_feat_imp = model.booster_.feature_importance(importance_type="gain")
        feat_imp += fold_feat_imp

        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': fold_feat_imp
        }).sort_values('importance', ascending=False)

        # âœ… not "PC"
        non_anon_imp_df = imp_df[~imp_df['feature'].str.startswith("Q")].copy()
        
        print(f"[Fold {fold+1}] Top 10 non-anonymous features:")
        print(non_anon_imp_df.head(10).to_string(index=False))

        top30 = non_anon_imp_df.head(30).reset_index(drop=True)
        top30_nonanon_per_fold.append(top30)

        val_pred = model.predict(X_va_fold)
        score = pearsonr(y_va_fold, val_pred)[0]
        cv_scores.append(score)
        print(f"[Fold {fold+1}] Pearson Score: {score:.4f}")

        fold_preds.append(model.predict(X_test))

        del model
        gc.collect()

    # ---------- Weighted embedding ----------
    weights = np.array([0, 0.4, 0, 0.4, 0.2], dtype=float)
    weights = weights[:len(fold_preds)]
    weights /= weights.sum()
    preds_test = np.average(fold_preds, axis=0, weights=weights)

    mean_cv = np.mean(cv_scores)
    avg_feat_imp = feat_imp / len(fold_preds)

    # ---------- The intersection of the last two folds is not an anonymous feature ----------
    if len(top30_nonanon_per_fold) >= 2:
        last_two = top30_nonanon_per_fold[-2:]
        set1 = set(last_two[0]['feature'])
        set2 = set(last_two[1]['feature'])
        common_features = list(set1.intersection(set2))

        avg_importance_dict = {}
        for feat in common_features:
            imp1 = last_two[0].loc[last_two[0]['feature'] == feat, 'importance'].values[0]
            imp2 = last_two[1].loc[last_two[1]['feature'] == feat, 'importance'].values[0]
            avg_importance_dict[feat] = (imp1 + imp2) / 2

        common_imp_df = pd.DataFrame({
            'feature': list(avg_importance_dict.keys()),
            'avg_importance': list(avg_importance_dict.values())
        }).sort_values('avg_importance', ascending=False)

        print("\nâœ… Top common non-anonymous features in last two folds (intersection):")
        print(common_imp_df.to_string(index=False))

    return mean_cv, preds_test, avg_feat_imp

mean_cv, LGB_NN_5f_preds, imp_gain = cross_val_lgb_time_series_weighted(
    X, y, test_df, best_params, folds=5, val_ratio=0.2
)

print(f"Mean CV Pearson: {mean_cv:.4f}")


test_label_lgb_nn_5f = test_label.copy()
test_label_lgb_nn_5f["label"] = LGB_NN_5f_preds
print(test_label_lgb_nn_5f.head(5))
test_label_lgb_nn_5f = test_label_lgb_nn_5f.reset_index()
submit = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
submit['prediction'] = test_label_lgb_nn_5f.set_index('ID').loc[submit['ID'], 'label'].values
submit = submit.sort_values(by='ID').reset_index(drop=True)
print(submit.head(5))
submit.to_csv("submission_lgb_nn_5f.csv", index=False)


def custom_time_series_split(X, folds=3, val_ratio=0.2):
    """
    Sliding window time series split:
    - Each fold is a fixed-size chunk of data (e.g. 1/3 of total)
    - In each fold, use the first (1 - val_ratio) as training and the rest as validation
    """
    import numpy as np

    n = len(X)
    fold_size = n // folds

    splits = []
    for i in range(folds):
        fold_start = i * fold_size
        fold_end = (i + 1) * fold_size if i < folds - 1 else n

        fold_indices = np.arange(fold_start, fold_end)
        fold_len = fold_end - fold_start

        val_len = int(fold_len * val_ratio)
        train_len = fold_len - val_len

        train_idx = fold_indices[:train_len]
        val_idx = fold_indices[train_len:]

        splits.append((train_idx, val_idx))

    return splits


def cross_val_lgb_time_series_weighted(X_tr, y_tr, X_test, params, folds=3):
    """
    Perform 3-fold sliding window cross-validation with weighted ensemble.
    """
    splits = custom_time_series_split(X_tr, folds=folds)

    cv_scores, fold_preds = [], []
    feat_imp = np.zeros(X_tr.shape[1])
    feature_names = X_tr.columns.tolist()

    top30_nonanon_per_fold = []

    # Align test set columns
    X_test = X_test.reindex(columns=X_tr.columns, fill_value=0.0)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_tr_fold, X_va_fold = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
        y_tr_fold, y_va_fold = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]

        print(f"[Fold {fold+1}] Train idx: {tr_idx[0]}-{tr_idx[-1]}, Val idx: {va_idx[0]}-{va_idx[-1]}")

        model = lgb.LGBMRegressor(**params, random_state=42 + fold)
        model.fit(
            X_tr_fold, y_tr_fold,
            eval_set=[(X_va_fold, y_va_fold)],
            eval_metric=pearson_metric,
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        fold_feat_imp = model.booster_.feature_importance(importance_type="gain")
        feat_imp += fold_feat_imp

        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': fold_feat_imp
        }).sort_values('importance', ascending=False)

        # Filter non-anonymous features
        non_anon_imp_df = imp_df[~imp_df['feature'].str.startswith("Q")].copy()
        
        print(f"[Fold {fold+1}] Top 10 non-anonymous features:")
        print(non_anon_imp_df.head(10).to_string(index=False))

        top30 = non_anon_imp_df.head(30).reset_index(drop=True)
        top30_nonanon_per_fold.append(top30)

        val_pred = model.predict(X_va_fold)
        score = pearsonr(y_va_fold, val_pred)[0]
        cv_scores.append(score)
        print(f"[Fold {fold+1}] Pearson Score: {score:.4f}")

        fold_preds.append(model.predict(X_test))

        del model
        gc.collect()

    # ---------- Weighted ensemble ----------
    # Adjust the weights for the last two folds (0, 0.5, 0.5)
    weights = np.array([0, 0, 1], dtype=float)  # 3 folds with 0, 0.5, 0.5 weights
    weights = weights[:len(fold_preds)]  # Ensure that we only use the folds we have
    weights /= weights.sum()  # Normalize the weights
    preds_test = np.average(fold_preds, axis=0, weights=weights)

    mean_cv = np.mean(cv_scores)
    avg_feat_imp = feat_imp / len(fold_preds)

    # ---------- Intersection of the last two folds (non-anonymous features) ----------
    if len(top30_nonanon_per_fold) >= 2:
        last_two = top30_nonanon_per_fold[-2:]
        set1 = set(last_two[0]['feature'])
        set2 = set(last_two[1]['feature'])
        common_features = list(set1.intersection(set2))

        avg_importance_dict = {}
        for feat in common_features:
            imp1 = last_two[0].loc[last_two[0]['feature'] == feat, 'importance'].values[0]
            imp2 = last_two[1].loc[last_two[1]['feature'] == feat, 'importance'].values[0]
            avg_importance_dict[feat] = (imp1 + imp2) / 2

        common_imp_df = pd.DataFrame({
            'feature': list(avg_importance_dict.keys()),
            'avg_importance': list(avg_importance_dict.values())
        }).sort_values('avg_importance', ascending=False)

        print("\nâœ… Top common non-anonymous features in last two folds (intersection):")
        print(common_imp_df.to_string(index=False))

    return mean_cv, preds_test, avg_feat_imp

# Example usage
mean_cv, LGB_NN_3f_preds, imp_gain = cross_val_lgb_time_series_weighted(
    X, y, test_df, best_params, folds=3
)

print(f"Mean CV Pearson: {mean_cv:.4f}")


test_label_lgb_nn_3f = test_label.copy()
test_label_lgb_nn_3f["label"] = LGB_NN_3f_preds
print(test_label_lgb_nn_3f.head(5))
test_label_lgb_nn_3f = test_label_lgb_nn_3f.reset_index()
submit = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
submit['prediction'] = test_label_lgb_nn_3f.set_index('ID').loc[submit['ID'], 'label'].values
submit = submit.sort_values(by='ID').reset_index(drop=True)
print(submit.head(5))
submit.to_csv("submission_lgb_nn_3f.csv", index=False)


test_lgb_nn_combo = test_label.copy()
combo = (LGB_NN_3f_preds + LGB_NN_5f_preds)/2
test_lgb_nn_combo["label"] = combo
print(test_lgb_nn_combo.head(5))
test_lgb_nn_combo = test_lgb_nn_combo.reset_index()
submit = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
submit['prediction'] = test_lgb_nn_combo.set_index('ID').loc[submit['ID'], 'label'].values
submit = submit.sort_values(by='ID').reset_index(drop=True)
print(submit.head(5))
submit.to_csv("submission_lgb_nn_combo.csv", index=False)


X = train_df.drop(columns=['label'])
y = train_df['label']


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

test_scaled = scaler.transform(test_df)
test_scaled = pd.DataFrame(test_scaled, columns=test_df.columns, index=test_df.index)

model = Ridge()
model.fit(X_scaled, y)

coef_series = pd.Series(model.coef_, index=X.columns).sort_values()

threshold = 0.05
low_coef_features = coef_series[coef_series.abs() < threshold].index

X_cleaned = X_scaled.drop(columns=low_coef_features)
test_cleaned = test_scaled.drop(columns=low_coef_features)

model_cleaned = Ridge()
model_cleaned.fit(X_cleaned, y)

coef_series_cleaned = pd.Series(model_cleaned.coef_, index=X_cleaned.columns).sort_values()

plt.figure(figsize=(15, 4))
coef_series_cleaned.plot(kind='bar')
plt.title("Feature Importance (Ridge Coefficients) After Filtering")
plt.xlabel("Feature")
plt.ylabel("Coefficient Value")
plt.tight_layout()
plt.show()

print(f"Removed features with abs(coef) < {threshold}:\n{low_coef_features.tolist()}")

ridge_preds = model_cleaned.predict(test_cleaned)


test_ridge = test_label.copy()
test_ridge["label"] = ridge_preds
print(test_ridge.head(5))
test_ridge = test_ridge.reset_index()
submit = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
submit['prediction'] = test_ridge.set_index('ID').loc[submit['ID'], 'label'].values
submit = submit.sort_values(by='ID').reset_index(drop=True)
print(submit.head(5))
submit.to_csv("submission_ridge.csv", index=False)


X = train_df.drop(columns=['label'])
y = train_df['label']


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

test_scaled = scaler.transform(test_df)
test_scaled = pd.DataFrame(test_scaled, columns=test_df.columns, index=test_df.index)

model = SGDRegressor(random_state=42)
model.fit(X_scaled, y)

coef_series = pd.Series(model.coef_, index=X.columns).sort_values()

threshold = 0.05
low_coef_features = coef_series[coef_series.abs() < threshold].index

X_cleaned = X_scaled.drop(columns=low_coef_features)
test_cleaned = test_scaled.drop(columns=low_coef_features)

model_cleaned = SGDRegressor(random_state=42)
model_cleaned.fit(X_cleaned, y)

coef_series_cleaned = pd.Series(model_cleaned.coef_, index=X_cleaned.columns).sort_values()

plt.figure(figsize=(15, 4))
coef_series_cleaned.plot(kind='bar')
plt.title("Feature Importance (Coefficients) After Filtering")
plt.xlabel("Feature")
plt.ylabel("Coefficient Value")
plt.tight_layout()
plt.show()

print(f"Removed features with abs(coef) < {threshold}:\n{low_coef_features.tolist()}")

SGD_preds = model_cleaned.predict(test_cleaned)


test_SGD = test_label.copy()
test_SGD["label"] = SGD_preds
print(test_SGD.head(5))
test_SGD = test_SGD.reset_index()
submit = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
submit['prediction'] = test_SGD.set_index('ID').loc[submit['ID'], 'label'].values
submit = submit.sort_values(by='ID').reset_index(drop=True)
print(submit.head(5))
submit.to_csv("submission_SGD.csv", index=False)


linear_combo = 0*ridge_preds + 1*SGD_preds


test_linear_combo = test_label.copy()
test_linear_combo["label"] = linear_combo
print(test_linear_combo.head(5))
test_linear_combo = test_linear_combo.reset_index()
submit = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
submit['prediction'] = test_linear_combo.set_index('ID').loc[submit['ID'], 'label'].values
submit = submit.sort_values(by='ID').reset_index(drop=True)
print(submit.head(5))
submit.to_csv("submission_linear_combo.csv", index=False)


def analyze_series(data, title_prefix=""):
    data = np.asarray(data)
    cumsum = np.cumsum(data)

    fig, axs = plt.subplots(1, 3, figsize=(15, 3))

    axs[0].plot(data, color='blue')
    axs[0].set_title(f'{title_prefix} Line Plot')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('Value')

    axs[1].hist(data, bins=100, color='orange', alpha=0.7)
    axs[1].set_title(f'{title_prefix} Histogram')
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('Frequency')

    axs[2].plot(cumsum, color='green')
    axs[2].set_title(f'{title_prefix} Cumulative Sum Plot')
    axs[2].set_xlabel('Index')
    axs[2].set_ylabel('Cumulative Sum')

    plt.tight_layout()
    plt.show()


analyze_series(NN_test_preds, title_prefix="NN Test Predictions")
analyze_series(LGB_NN_5f_preds, title_prefix="LGB NN 5 folds Test Predictions")
analyze_series(LGB_NN_3f_preds, title_prefix="LGB NN 3 folds Test Predictions")
analyze_series(combo, title_prefix="LGB NN combo Test Predictions")
analyze_series(ridge_preds, title_prefix="Ridge Test Predictions")
analyze_series(SGD_preds, title_prefix="SGD Test Predictions")
analyze_series(linear_combo, title_prefix="Linear combo Test Predictions")


last_pred1 = pd.read_csv("/kaggle/input/preds-0-95/submission (24).csv")
test_label_last1 = test_label.copy()
last_pred1 = last_pred1.set_index('ID')
last_pred1.index = last_pred1.index.astype(test_label_last1.index.dtype)
test_label_last1['label'] = last_pred1['prediction']
test_label_last1 = test_label_last1.reset_index()
print(test_label_last1.head())

last_pred2 = pd.read_csv("/kaggle/input/preds-0-90/submission.csv")
test_label_last2 = test_label.copy()
last_pred2 = last_pred2.set_index('ID')
last_pred2.index = last_pred2.index.astype(test_label_last2.index.dtype)
test_label_last2['label'] = last_pred2['prediction']
test_label_last2 = test_label_last2.reset_index()
print(test_label_last2.head())

last_pred3 = pd.read_csv("/kaggle/input/preds-0-9516/submission (25).csv")
test_label_last3 = test_label.copy()
last_pred3 = last_pred3.set_index('ID')
last_pred3.index = last_pred3.index.astype(test_label_last3.index.dtype)
test_label_last3['label'] = last_pred3['prediction']
test_label_last3 = test_label_last3.reset_index()
print(test_label_last3.head())

analyze_series(test_label_last1['label'], title_prefix="0.95_preds")
analyze_series(test_label_last2['label'], title_prefix="0.90_preds")
analyze_series(test_label_last3['label'], title_prefix="0.9516_preds")

correlation = test_label_last1['label'].corr(test_label_last2['label'])
print("preds corrï¼š",correlation)
cumsum1 = test_label_last1['label'].cumsum()
cumsum2 = test_label_last2['label'].cumsum()
correlation = cumsum1.corr(cumsum2)
print("cum_preds corrï¼š", correlation)


del NN_test_preds, LGB_NN_5f_preds, LGB_NN_3f_preds, combo, ridge_preds, SGD_preds, linear_combo


lgb_nn = pd.read_csv("/kaggle/working/submission_lgb_nn_combo.csv")
ridge_sgd = pd.read_csv("/kaggle/working/submission_linear_combo.csv")
public1 = pd.read_csv("/kaggle/input/preds-0-95/submission (24).csv")
public2 = pd.read_csv("/kaggle/input/preds-0-90/submission.csv")
public3 = pd.read_csv("/kaggle/input/preds-0-9516/submission (25).csv")
public4 = pd.read_csv("/kaggle/input/preds-0-95169/submission (26).csv")

weights = {
    'lgb_nn': 0.1,
    'ridge_sgd': 0.1,
    'public1': 0.1,
    'public2': 0.1,
    'public3': 0.2, 
    'public4': 0.4,}


assert (lgb_nn['ID'] == ridge_sgd['ID']).all()
assert (lgb_nn['ID'] == public1['ID']).all()
assert (lgb_nn['ID'] == public2['ID']).all()
assert (lgb_nn['ID'] == public3['ID']).all()
assert (lgb_nn['ID'] == public4['ID']).all()

final_prediction = (
    weights['lgb_nn'] * lgb_nn['prediction'] +
    weights['ridge_sgd'] * ridge_sgd['prediction'] +
    weights['public1'] * public1['prediction'] +
    weights['public2'] * public2['prediction'] +
    weights['public3'] * public3['prediction'] +
    weights['public4'] * public4['prediction']
)


final_submission = pd.DataFrame({
    'ID': lgb_nn['ID'],
    'prediction': final_prediction
})

print(final_submission.head(5))

final_submission.to_csv("/kaggle/working/final_weighted_submission.csv", index=False)

