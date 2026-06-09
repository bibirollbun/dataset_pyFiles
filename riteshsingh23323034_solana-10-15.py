# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import pandas as pd

# List of folders where CSVs are stored
folders = [
   "/kaggle/input/alpha-radar-solana-sprint"
]

all_csvs = []
for folder in folders:
    for file in os.listdir(folder):
        if file.endswith(".csv") and file != "Sample_Dataset.csv":
            all_csvs.append(os.path.join(folder, file))

# Concatenate
df = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)

print("Final shape:", df.shape)
df.head()



# faithful_polars_port_improved.py
# Faithful Polars front-end with per-token pandas-based group computation
# Produces token-level features identical to the original pandas pipeline.
#
# Requirements:
#   pip install polars pandas numpy scipy

import polars as pl
import pandas as pd
import numpy as np
from math import sqrt
from scipy.stats import linregress
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# --------------------------
# Utility helpers (same semantics as original)
# --------------------------
# import pandas as pd
# import numpy as np

def parse_timestamp_to_seconds(ts, reference=None):
    """
    Converts a timestamp or time string into total seconds.
    - Supports datetime strings like '2025-10-06 05:19:11.492000+00:00'
    - Supports MM:SS(.ms) or HH:MM:SS formats
    - If `reference` is provided, subtracts it (to get relative seconds)
    """
    if pd.isna(ts):
        return np.nan

    # Numeric input
    if isinstance(ts, (int, float, np.number)):
        return float(ts)

    s = str(ts).strip()

    # Handle datetime-like strings
    try:
        dt = pd.to_datetime(s, errors="raise", utc=True)
        if reference is not None:
            dt0 = pd.to_datetime(reference, utc=True)
            return (dt - dt0).total_seconds()
        else:
            return dt.timestamp()   # absolute seconds since epoch
    except Exception:
        pass

    # Handle MM:SS or HH:MM:SS style
    if ":" in s:
        parts = s.split(":")
        try:
            parts = [float(p) for p in parts]
        except:
            return np.nan
        sec = parts[-1]
        total = sec
        mul = 60.0
        for p in parts[-2::-1]:
            total += p * mul
            mul *= 60.0
        return float(total)

    # Fallback: try numeric
    try:
        return float(s)
    except:
        return np.nan


def rms(series: pd.Series):
    arr = series.dropna().values
    if arr.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(arr))))


def safe_first(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) else np.nan


def safe_last(series: pd.Series):
    s = series.dropna()
    return s.iloc[-1] if len(s) else np.nan


def linear_slope(x: np.ndarray, y: np.ndarray):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan
    if np.allclose(x, x[0]):
        return 0.0
    if np.allclose(y, y[0]):
        return 0.0
    try:
        slope, _, _, _, _ = linregress(x, y)
        return float(slope)
    except Exception:
        return np.nan


def series_stats(s: pd.Series) -> dict:
    out = {}
    arr = s.dropna().values
    if arr.size == 0:
        for k in ['min', 'max', 'mean', 'median', 'sum', 'std', 'var', 'skew', 'kurt', 'first', 'last', 'rms']:
            out[k] = np.nan
        return out
    out['min'] = float(np.nanmin(arr))
    out['max'] = float(np.nanmax(arr))
    out['mean'] = float(np.nanmean(arr))
    out['median'] = float(np.nanmedian(arr))
    out['sum'] = float(np.nansum(arr))
    out['std'] = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
    out['var'] = float(np.nanvar(arr, ddof=1)) if arr.size > 1 else 0.0
    out['skew'] = float(pd.Series(arr).skew()) if arr.size > 2 else 0.0
    out['kurt'] = float(pd.Series(arr).kurt()) if arr.size > 3 else 0.0
    nz = s.dropna()
    out['first'] = float(nz.iloc[0]) if len(nz) > 0 else np.nan
    out['last'] = float(nz.iloc[-1]) if len(nz) > 0 else np.nan
    out['rms'] = float(rms(pd.Series(arr)))
    return out


# --------------------------
# The group function (runs per token) using pandas to replicate original behavior
# --------------------------
def compute_token_row(pdf: pd.DataFrame, token_col: str, holder_col: str, present_numeric_cols: list):
    """
    pdf: pandas DataFrame for a single token group (already filtered to _rel_t <= 30)
    returns: dict row with token-level features
    """
    row = {}
    token = pdf[token_col].iloc[0] if token_col in pdf.columns and len(pdf) > 0 else None
    row[token_col] = token

    sub = pdf.sort_values('_rel_t', kind='mergesort').copy()
    tx_count = len(sub)
    row['tx_count_30s'] = int(tx_count)
    row['txs_per_sec'] = float(tx_count) / 30.0

    if tx_count > 0:
        row['time_span'] = float(sub['_rel_t'].max() - sub['_rel_t'].min())
        row['first_trade_rel_t'] = float(sub['_rel_t'].min())
        row['last_trade_rel_t'] = float(sub['_rel_t'].max())
    else:
        row['time_span'] = np.nan
        row['first_trade_rel_t'] = np.nan
        row['last_trade_rel_t'] = np.nan

    # holders
    if holder_col in sub.columns:
        row['unique_holders_count'] = int(sub[holder_col].nunique())
        if tx_count > 0:
            vc = sub[holder_col].value_counts()
            top_addr_count = int(vc.iloc[0]) if len(vc) > 0 else 0
        else:
            top_addr_count = 0
        row['max_trades_by_one_holder'] = top_addr_count
    else:
        row['unique_holders_count'] = np.nan
        row['max_trades_by_one_holder'] = np.nan

    # per-numeric stats
    for c in present_numeric_cols:
        if c in sub.columns:
            stats_out = series_stats(sub[c])
        else:
            stats_out = {k: np.nan for k in ['min', 'max', 'mean', 'median', 'sum', 'std', 'var', 'skew', 'kurt', 'first', 'last', 'rms']}
        for k, v in stats_out.items():
            row[f'{c}_{k}'] = v
        # delta
        try:
            row[f'{c}_delta'] = (stats_out['last'] - stats_out['first']) if (not np.isnan(stats_out['last']) and not np.isnan(stats_out['first'])) else np.nan
        except:
            row[f'{c}_delta'] = np.nan

        # slope over time for this feature
        try:
            valid_mask = sub[c].notna() & sub['_rel_t'].notna()
            if valid_mask.sum() >= 2:
                xs = sub.loc[valid_mask, '_rel_t'].values.astype(float)
                ys = sub.loc[valid_mask, c].values.astype(float)
                row[f'{c}_slope'] = linear_slope(xs, ys)
            else:
                row[f'{c}_slope'] = np.nan
        except Exception:
            row[f'{c}_slope'] = np.nan

    # price_proxy
    if ('sol_delta' in sub.columns) and ('token_quantity' in sub.columns):
        def safe_div(a, b):
            try:
                return float(a) / float(b) if (pd.notna(a) and pd.notna(b) and b != 0) else np.nan
            except:
                return np.nan

        price_proxy = sub.apply(lambda r: safe_div(r.get('sol_delta', np.nan), r.get('token_quantity', np.nan)), axis=1)
        pp_stats = series_stats(price_proxy)
        for k, v in pp_stats.items():
            row[f'price_proxy_{k}'] = v
        row['price_proxy_delta'] = (pp_stats['last'] - pp_stats['first']) if (not np.isnan(pp_stats['last']) and not np.isnan(pp_stats['first'])) else np.nan
        row['price_proxy_rms'] = pp_stats['rms']
        # slope
        try:
            valid_mask = (~pd.isna(price_proxy)) & sub['_rel_t'].notna()
            if valid_mask.sum() >= 2:
                xs = sub.loc[valid_mask, '_rel_t'].values.astype(float)
                ys = price_proxy[valid_mask].values.astype(float)
                row['price_proxy_slope'] = linear_slope(xs, ys)
            else:
                row['price_proxy_slope'] = np.nan
        except:
            row['price_proxy_slope'] = np.nan
    else:
        for k in ['min', 'max', 'mean', 'median', 'sum', 'std', 'var', 'skew', 'kurt', 'first', 'last', 'rms', 'delta', 'slope']:
            row[f'price_proxy_{k}'] = np.nan

    # bucketed features (0-10,10-20,20-30)
    buckets = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    for i, (start, end) in enumerate(buckets):
        bmask = (sub['_rel_t'] >= start) & (sub['_rel_t'] < end)
        bucket_df = sub[bmask]
        row[f'bucket_{i + 1}_tx_count'] = int(len(bucket_df))
        if 'token_volume' in bucket_df.columns:
            row[f'bucket_{i + 1}_token_volume_sum'] = float(bucket_df['token_volume'].sum()) if len(bucket_df) > 0 else 0.0
        else:
            row[f'bucket_{i + 1}_token_volume_sum'] = np.nan
        if 'sol_volume' in bucket_df.columns:
            row[f'bucket_{i + 1}_sol_volume_sum'] = float(bucket_df['sol_volume'].sum()) if len(bucket_df) > 0 else 0.0
        else:
            row[f'bucket_{i + 1}_sol_volume_sum'] = np.nan
        if 'buy_count' in bucket_df.columns:
            row[f'bucket_{i + 1}_buy_sum'] = int(bucket_df['buy_count'].sum()) if len(bucket_df) > 0 else 0
        else:
            row[f'bucket_{i + 1}_buy_sum'] = np.nan
        if 'sell_count' in bucket_df.columns:
            row[f'bucket_{i + 1}_sell_sum'] = int(bucket_df['sell_count'].sum()) if len(bucket_df) > 0 else 0
        else:
            row[f'bucket_{i + 1}_sell_sum'] = np.nan

    # time-to-50% volume and fraction in first 10s
    if 'token_volume' in sub.columns and sub['token_volume'].notna().any():
        sorted_sub = sub.sort_values('_rel_t', kind='mergesort')
        if 'token_volume' in sorted_sub.columns:
            cum = sorted_sub['token_volume'].cumsum()
            total_vol = float(cum.iloc[-1]) if len(cum) > 0 else 0.0
            if total_vol > 0:
                half_idx = int(np.searchsorted(cum.values, total_vol * 0.5))
                t_half = float(sorted_sub['_rel_t'].iloc[half_idx]) if half_idx < len(sorted_sub) else float(sorted_sub['_rel_t'].iloc[-1])
                row['time_to_50pct_volume'] = t_half
                v_first10 = float(sorted_sub[sorted_sub['_rel_t'] < 10]['token_volume'].sum())
                row['fraction_volume_first10s'] = float(v_first10 / total_vol)
            else:
                row['time_to_50pct_volume'] = np.nan
                row['fraction_volume_first10s'] = np.nan
        else:
            row['time_to_50pct_volume'] = np.nan
            row['fraction_volume_first10s'] = np.nan
    else:
        row['time_to_50pct_volume'] = np.nan
        row['fraction_volume_first10s'] = np.nan

    # Derived ratios & interactions
    buy_last = row.get('buy_count_last', np.nan)
    sell_last = row.get('sell_count_last', np.nan)
    tot_count_last = row.get('total_count_last', np.nan)

    try:
        row['buy_minus_sell'] = (buy_last - sell_last) if (not pd.isna(buy_last) and not pd.isna(sell_last)) else np.nan
    except:
        row['buy_minus_sell'] = np.nan
    try:
        if not pd.isna(tot_count_last):
            row['buy_frac_of_total'] = (buy_last / tot_count_last) if (not pd.isna(buy_last) and tot_count_last > 0) else np.nan
        else:
            row['buy_frac_of_total'] = np.nan
    except:
        row['buy_frac_of_total'] = np.nan

    mv_std = row.get('market_cap_usd_std', np.nan)
    tv_sum = row.get('token_volume_sum', np.nan)
    if (not pd.isna(mv_std)) and (not pd.isna(tv_sum)):
        row['volatility_x_volume'] = mv_std * tv_sum
    else:
        row['volatility_x_volume'] = np.nan

    lq_first = row.get('liquidity_ratio_first', np.nan)
    lq_last = row.get('liquidity_ratio_last', np.nan)
    if (not pd.isna(lq_first)) and (not pd.isna(lq_last)):
        row['liquidity_ratio_change'] = lq_last - lq_first
        row['liquidity_ratio_pct_change'] = (lq_last - lq_first) / (abs(lq_first) + 1e-9)
    else:
        row['liquidity_ratio_change'] = np.nan
        row['liquidity_ratio_pct_change'] = np.nan

    cr_sold = row.get('creator_sold_last', np.nan)
    cr_bal = row.get('creator_balance_last', np.nan)
    token_vol_sum = row.get('token_volume_sum', np.nan)
    try:
        row['creator_sold_to_balance'] = (cr_sold / (cr_bal + 1e-9)) if (not pd.isna(cr_sold) and not pd.isna(cr_bal)) else np.nan
    except:
        row['creator_sold_to_balance'] = np.nan
    try:
        row['creator_sold_frac_volume'] = (cr_sold / token_vol_sum) if (not pd.isna(cr_sold) and not pd.isna(token_vol_sum) and token_vol_sum > 0) else np.nan
    except:
        row['creator_sold_frac_volume'] = np.nan

    top10_last = row.get('top10_percent_total_last', np.nan)
    top10_first = row.get('top10_percent_total_first', np.nan)
    if (not pd.isna(top10_last)) and (not pd.isna(top10_first)):
        row['top10_pct_change'] = top10_last - top10_first
    else:
        row['top10_pct_change'] = np.nan

    row['is_sparse'] = 1 if tx_count <= 2 else 0
    row['sparse_tx_count'] = tx_count

    bsum = row.get('buy_count_sum', np.nan)
    ssum = row.get('sell_count_sum', np.nan)
    if (not pd.isna(bsum)) and (not pd.isna(ssum)) and (not pd.isna(row['unique_holders_count'])):
        try:
            row['wash_trade_proxy'] = ((bsum + ssum) / (row['unique_holders_count'] + 1e-9)) * row['max_trades_by_one_holder']
        except:
            row['wash_trade_proxy'] = np.nan
    else:
        row['wash_trade_proxy'] = np.nan

    return row


# --------------------------
# Main pipeline using Polars for IO + grouping
# --------------------------
def fe_token_level_full_parity(polars_df: pl.DataFrame,
                               time_col: str = 'timestamp',
                               token_col: str = 'mint_token_id',
                               holder_col: str = 'holder'):
    # ensure in-memory polars DataFrame
    df = polars_df.lazy().collect()

    # parse timestamp into seconds with same semantics
    df = df.with_columns(
        pl.col(time_col).map_elements(lambda x: parse_timestamp_to_seconds(x), return_dtype=pl.Float64).alias('_ts_seconds')
    )

    # _rel_t relative to first per token (window)
    df = df.with_columns(
        (pl.col('_ts_seconds') - pl.col('_ts_seconds').min().over(token_col)).alias('_rel_t')
    )

    # version-compatible clip: replace negative _rel_t with 0.0
    df = df.with_columns(
        pl.when(pl.col('_rel_t') < 0.0)
          .then(0.0)
          .otherwise(pl.col('_rel_t'))
          .alias('_rel_t')
    )

    # keep only <= 30s window
    df = df.filter(pl.col('_rel_t') <= 30.0)

    # numeric columns to consider (same list as original)
    numeric_cols = [
        'token_quantity', 'token_delta', 'sol_delta', 'token_volume', 'sol_volume',
        'liquidity_ratio', 'virtual_sol_reserves', 'virtual_token_reserves',
        'consumed_gas', 'fee', 'relative_strength_index', 'bollinger_relative_position',
        'volume_oscillator', 'rate_of_change', 'money_flow_index',
        'market_cap_usd', 'creator_fee', 'creator_fee_pump',
        'creator_balance', 'creator_sold', 'total_holders', 'current_holders',
        'top10_percent_total', 'holder_ratio', 'buy_sell_ratio',
        # also support some count columns if present
        'buy_count', 'sell_count', 'total_count'
    ]
    present_numeric_cols = [c for c in numeric_cols if c in df.columns]

    # build a list of token groups
    tokens = df.select(pl.col(token_col)).unique().to_series().to_list()
    print(f"Found {len(tokens)} tokens. Aggregating per-token groups...")

    token_rows = []
    # convert entire polars df to pandas once for fast per-token slicing in pandas
    pdf_all = df.to_pandas()

    for i, token in enumerate(tokens):
        sub_pdf = pdf_all[pdf_all[token_col] == token].copy()
        # ensure sorted by _rel_t (mergesort stable)
        sub_pdf = sub_pdf.sort_values('_rel_t', kind='mergesort')
        row = compute_token_row(sub_pdf, token_col, holder_col, present_numeric_cols)
        token_rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1} / {len(tokens)} tokens...")

    # build pandas DataFrame from rows then convert to polars
    token_feats_pdf = pd.DataFrame(token_rows)

    # Replace infinite with NaN
    token_feats_pdf.replace([np.inf, -np.inf], np.nan, inplace=True)

    # add log transforms for specific bases if present
    for base in ['token_volume_sum', 'sol_volume_sum', 'market_cap_usd_sum', 'token_quantity_sum']:
        if base in token_feats_pdf.columns:
            token_feats_pdf[f'log_{base}'] = token_feats_pdf[base].apply(lambda x: np.log1p(x) if pd.notna(x) and x > 0 else np.nan)

    # reorder columns to put counts/sparse first
    cols = list(token_feats_pdf.columns)
    pref = ['tx_count_30s', 'txs_per_sec', 'is_sparse', 'sparse_tx_count', 'unique_holders_count', 'max_trades_by_one_holder']
    for p in pref[::-1]:
        if p in cols:
            cols.insert(0, cols.pop(cols.index(p)))
    token_feats_pdf = token_feats_pdf[cols]

    # convert to polars for output (keep token_col as a regular column)
    token_features_pl = pl.from_pandas(token_feats_pdf.reset_index(drop=False))

    return token_features_pl


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    RAW_PATH = "/kaggle/working/converted_data4.csv"  # change to your filename
    OUT_PATH = "token_features_full_parity4.csv"

    print("Reading CSV with Polars (explicit schema where useful) ...")

    # you can customize this map if you have a printed schema; otherwise we force common numeric columns to float
    numeric_cols = [
        'token_quantity', 'token_delta', 'sol_delta', 'token_volume', 'sol_volume',
        'liquidity_ratio', 'virtual_sol_reserves', 'virtual_token_reserves',
        'consumed_gas', 'fee', 'relative_strength_index', 'bollinger_relative_position',
        'volume_oscillator', 'rate_of_change', 'money_flow_index',
        'market_cap_usd', 'creator_fee', 'creator_fee_pump',
        'creator_balance', 'creator_sold', 'total_holders', 'current_holders',
        'top10_percent_total', 'holder_ratio', 'buy_sell_ratio',
        'buy_count', 'sell_count', 'total_count'
    ]
    # build schema_override map (use pl.Float64 for numeric suspects)
    schema_overrides = {c: pl.Float64 for c in numeric_cols}

    # try to read with schema_overrides; fallback to more defensive read if necessary
    try:
        d1=df
        for col in d1.select_dtypes(include=["int64", "float64", "bool"]).columns:
            d1[col] = d1[col].astype(np.float32)
        d1.insert(0, "index", range(1,len(d1)+1))  
        d1.to_csv("converted_data.csv", index=False)
            
            # Try primary read with schema overrides (fast + typed)
        df_raw = pl.read_csv("/kaggle/working/converted_data.csv", schema_overrides=schema_overrides,
                                 infer_schema_length=20000, try_parse_dates=False)
    except Exception as e:
        # fallback: read with relaxed inference then coerce suspicious columns
        print("Primary read failed or raised warning, retrying as utf8 then coercing numeric columns. Error:", e)
        df_raw = pl.read_csv(RAW_PATH, dtypes="utf8", infer_schema_length=2000)
        for c in numeric_cols:
            if c in df_raw.columns:
                df_raw = df_raw.with_columns(
                    pl.col(c)
                    .str.replace_all(",", "")
                    .str.replace_all(r"[^\d\.\-eE\+]", "")
                    .cast(pl.Float64, strict=False)
                    .alias(c)
                )

    print("CSV loaded: rows =", df_raw.height, "cols =", df_raw.width)

    # Defensive: if creator_sold is boolean, convert to int for numeric ops
    if "creator_sold" in df_raw.columns and df_raw["creator_sold"].dtype == pl.Boolean:
        df_raw = df_raw.with_columns(pl.col("creator_sold").cast(pl.Int64).alias("creator_sold"))

    print("Computing features (parity with pandas). This will iterate per-token and may take time ...")
    token_feats = fe_token_level_full_parity(df_raw, time_col='timestamp', token_col='mint_token_id', holder_col='holder')

    print("Finished. Shape:", token_feats.shape)
    token_feats.write_csv(OUT_PATH)
    print("Saved to:", OUT_PATH)

    # quick peek
    try:
        print(token_feats.head(3).to_pandas().T)
    except Exception:
        print(token_feats.head(3))



# if __name__ == "__main__":
#     import os
#     import re
#     from pathlib import Path

#     INPUT_FOLDER = Path("/kaggle/input/alpha-radar-solana-sprint")
#     # We'll write one output file per input chunk for easier discovery:
#     # token_features_full_parity_september_chunk_010.csv ... _015.csv
#     OUT_DIR = Path.cwd()  # change if you want a different output folder

#     # chunk range you requested (inclusive)
#     CHUNK_START = 1
#     CHUNK_END = 5

#     # filename pattern base (expects: september_2025_first30s_chunk_002.csv)
#     pattern = re.compile(r"evaluation_set_30s_chunk_(\d{3})\.csv$", re.IGNORECASE)

#     # numeric columns / schema override you had
#     numeric_cols = [
#         'token_quantity', 'token_delta', 'sol_delta', 'token_volume', 'sol_volume',
#         'liquidity_ratio', 'virtual_sol_reserves', 'virtual_token_reserves',
#         'consumed_gas', 'fee', 'relative_strength_index', 'bollinger_relative_position',
#         'volume_oscillator', 'rate_of_change', 'money_flow_index',
#         'market_cap_usd', 'creator_fee', 'creator_fee_pump',
#         'creator_balance', 'creator_sold', 'total_holders', 'current_holders',
#         'top10_percent_total', 'holder_ratio', 'buy_sell_ratio',
#         'buy_count', 'sell_count', 'total_count'
#     ]
#     schema_overrides = {c: pl.Float64 for c in numeric_cols}

#     # collect candidate files
#     all_files = []
#     for f in INPUT_FOLDER.iterdir():
#         if f.is_file():
#             m = pattern.search(f.name)
#             if m:
#                 idx = int(m.group(1))
#                 if CHUNK_START <= idx <= CHUNK_END:
#                     all_files.append((idx, f))
#     all_files.sort()  # sort by chunk number

#     if len(all_files) == 0:
#         raise SystemExit(f"No matching files found in {INPUT_FOLDER} for chunks {CHUNK_START:03d}..{CHUNK_END:03d}")

#     print(f"Found {len(all_files)} files to process: {[p.name for (_, p) in all_files]}")

#     for idx, file_path in all_files:
#         out_filename = f"token_features_full_parity_september_chunk_{idx:03d}.csv"
#         out_path = OUT_DIR / out_filename

#         print(f"\n--- Processing chunk {idx:03d}: {file_path.name} -> {out_filename} ---")
#         try:
#             d1=df
#             for col in d1.select_dtypes(include=["int64", "float64", "bool"]).columns:
#                 d1[col] = d1[col].astype(np.float32)
#             d1.insert(0, "index", range(1,len(d1)+1))  
#             d1.to_csv("converted_data.csv", index=False)
            
#             # Try primary read with schema overrides (fast + typed)
#             df_raw = pl.read_csv("/kaggle/working/converted_data.csv", schema_overrides=schema_overrides,
#                                  infer_schema_length=20000, try_parse_dates=False)
#         except Exception as e:
#             print("Primary read failed or raised warning, retrying as utf8 then coercing numeric columns. Error:", e)
#             d1=pd.read_csv(str(file_path))
#             for col in d1.select_dtypes(include=["int64", "float64", "bool"]).columns:
#                 d1[col] = d1[col].astype(np.float32)

#             d1.to_csv("converted_data.csv", index=False)
            
#             df_raw = pl.read_csv("/kaggle/working/converted_data.csv", dtypes="utf8", infer_schema_length=2000)
#             for c in numeric_cols:
#                 if c in df_raw.columns:
#                     df_raw = df_raw.with_columns(
#                         pl.col(c)
#                           .str.replace_all(",", "")
#                           .str.replace_all(r"[^\d\.\-eE\+]", "")
#                           .cast(pl.Float64, strict=False)
#                           .alias(c)
#                     )

#         print("CSV loaded: rows =", df_raw.height, "cols =", df_raw.width)

#         # Defensive: if creator_sold is boolean, convert to int for numeric ops
#         if "creator_sold" in df_raw.columns and df_raw["creator_sold"].dtype == pl.Boolean:
#             df_raw = df_raw.with_columns(pl.col("creator_sold").cast(pl.Int64).alias("creator_sold"))

#         # Compute token features for this chunk
#         print("Computing features for this chunk ...")
#         token_feats = fe_token_level_full_parity(df_raw, time_col='timestamp', token_col='mint_token_id', holder_col='holder')

#         # Convert to pandas and write per-chunk CSV (single file per chunk)
#         tf_pd = token_feats.to_pandas()
#         tf_pd.replace([np.inf, -np.inf], np.nan, inplace=True)

#         tf_pd.to_csv(out_path, index=False)
#         print(f"Wrote {len(tf_pd)} rows to {out_path}")

#         # cleanup references
#         del df_raw, token_feats, tf_pd

#     print("\nAll done. Outputs saved to:", str(OUT_DIR))



# d1=pd.read_csv("/kaggle/input/alpha-radar-solana-sprint/evaluation_set_30s_chunk_001.csv")


# d1.head()


# d1.dtypes


# for col in d1.select_dtypes(include=["int64", "float64", "bool"]).columns:
#                 d1[col] = d1[col].astype(np.float32)


# d1.dtypes


# d1.shape


# d1.insert(0, "index", range(1,len(d1)+1))  


# d1.head()

