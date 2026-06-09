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


# Verify GPU availability
import torch
print("GPU Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


# Import Libraries
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import torch
import gc

%matplotlib inline


# ## 1. Load Data
# Load and merge the raw data files with relational structure, using lazy loading to manage memory.
def load_raw_data(data_dir: Path, train=True) -> pl.DataFrame:
    """Load and merge raw data files with relational structure, optimized for memory."""
    base_path = data_dir / "parquet_files" / ("train" if train else "test")
    
    # Load base table lazily
    base = pl.scan_parquet(base_path / f"{'train' if train else 'test'}_base.parquet")
    
    # Load and merge static tables (limit to first file to reduce memory usage)
    static_files = list(base_path.glob(f"{'train' if train else 'test'}_static_0_*.parquet"))[:1] + \
                   [base_path / f"{'train' if train else 'test'}_static_cb_0.parquet"]
    df = base
    for static_file in static_files:
        static_df = pl.scan_parquet(static_file)
        print(f"Merging with {static_file.name}")
        df = df.join(static_df, on="case_id", how="left")
        gc.collect()  # Free memory after each merge

    # Load and aggregate credit bureau data (limit to first file to reduce memory usage)
    cb_files = list(base_path.glob(f"{'train' if train else 'test'}_credit_bureau_a_1_*.parquet"))[:1]
    cb_dfs = [pl.scan_parquet(f) for f in cb_files]
    cb_data = pl.concat(cb_dfs, how="vertical_relaxed")
    # Collect a small sample to inspect columns (minimal memory usage)
    cb_sample = cb_data.head(10).collect()
    print(f"Credit Bureau columns: {cb_sample.columns}")
    numeric_cols = [col for col in cb_sample.columns if cb_sample[col].dtype in [pl.Float64, pl.Int64]]
    agg_col = numeric_cols[0] if numeric_cols else None
    cb_agg = cb_data.group_by("case_id").agg(
        num_records=pl.len().alias("cb_num_records"),
        **({f"{agg_col}_mean": pl.col(agg_col).mean().alias(f"{agg_col}_mean")} if agg_col else {})
    )
    df = df.join(cb_agg, on="case_id", how="left")
    gc.collect()

    # Load and aggregate previous application data with schema alignment (limit to first file to reduce memory usage)
    applprev_files = list(base_path.glob(f"{'train' if train else 'test'}_applprev_*.parquet"))[:1]
    applprev_dfs = [pl.scan_parquet(f) for f in applprev_files]
    # Align schemas
    all_columns = set().union(*[set(df.collect_schema().names()) for df in applprev_dfs])
    aligned_applprev_dfs = []
    for df in applprev_dfs:
        missing_cols = all_columns - set(df.collect_schema().names())
        if missing_cols:
            for col in missing_cols:
                df = df.with_columns(pl.lit(None).alias(col))
        # Reorder columns to match the first DataFrame's schema
        df = df.select(sorted(all_columns, key=lambda x: list(applprev_dfs[0].collect_schema().names()).index(x) if x in applprev_dfs[0].collect_schema().names() else len(applprev_dfs[0].collect_schema().names())))
        aligned_applprev_dfs.append(df)
    applprev_data = pl.concat(aligned_applprev_dfs, how="vertical_relaxed")
    # Collect a small sample to inspect columns
    applprev_sample = applprev_data.head(10).collect()
    print(f"Previous Application columns: {applprev_sample.columns}")
    applprev_agg = applprev_data.group_by("case_id").agg(
        num_prev_apps=pl.len().alias("num_prev_apps"),
        **({"prev_credamount_mean": pl.col("credamount_590A").mean().alias("prev_credamount_mean")} if "credamount_590A" in applprev_sample.columns else {})
    )
    df = df.join(applprev_agg, on="case_id", how="left")
    gc.collect()

    # Collect the lazy DataFrame
    df = df.collect()
    return df

# Load train and test data
data_path = Path("/kaggle/input/home-credit-credit-risk-model-stability")
train_df_full = load_raw_data(data_path, train=True)
print(f"Shape of train_df_full before sampling: {train_df_full.shape}")
test_df = load_raw_data(data_path, train=False)

# Sample the training data to reduce memory usage
sample_size = 0.3  # Increase to 30% of data
train_df = train_df_full.sample(fraction=sample_size, seed=42)
print(f"Shape of train_df after sampling: {train_df.shape}")
gc.collect()


train_df.write_csv("train_df.csv")



train_df.head(30)


print("Columns in train_df:", train_df.columns)


train_df.describe()


import polars as pl
import gc

# Assume train_df and test_df are already loaded as Polars DataFrames

def preprocess_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the data by handling missing values, outliers, and encoding."""
    df_processed = df.clone() # Work on a clone

    # Impute numerical columns with median and handle outliers
    numerical_cols = [col for col, dtype in df_processed.schema.items() if dtype.is_numeric()]
    
    for col in numerical_cols:
        # Impute with median
        col_median = df_processed.select(pl.col(col).median()).item()
        
        if col_median is None:
            fill_value_for_num = 0.0
            df_processed = df_processed.with_columns(pl.col(col).fill_null(fill_value_for_num))
        else:
            df_processed = df_processed.with_columns(pl.col(col).fill_null(col_median))
            
        # Cap outliers at 99th percentile (only for specified columns)
        if col in ["actualdpd_943P", "childnum_21L", "credamount_590A", "mainoccupationinc_437A"]:
            if df_processed[col].null_count() < len(df_processed): # Check if column is not all nulls
                percentile_99_series = df_processed.select(pl.col(col).quantile(0.99, "midpoint")).get_column(col)
                if not percentile_99_series.is_empty() and percentile_99_series[0] is not None:
                    percentile_99 = percentile_99_series[0]
                    
                    # --- THIS IS THE CORRECTED SYNTAX ---
                    # Ensure upper_bound is not less than lower_bound.
                    # For these columns, percentile_99 should be >= 0.
                    actual_upper_bound = max(0.0, percentile_99) 
                    
                    df_processed = df_processed.with_columns(
                        pl.col(col).clip(lower_bound=0.0, upper_bound=actual_upper_bound) # Use 0.0 for float consistency
                    )
                # else: percentile_99 could not be computed, consider skipping clipping or using a default.

        # Log-transform for skewed columns
        if col in ["credamount_590A", "mainoccupationinc_437A", "annuity_853A"]:
            df_processed = df_processed.with_columns(
                (pl.when(pl.col(col) >= 0).then(pl.col(col)).otherwise(0) + 1).log().alias(f"log_{col}")
            )

    # Impute existing categorical columns with mode
    categorical_cols = [col for col, dtype in df_processed.schema.items() if dtype == pl.Categorical]
    for col in categorical_cols:
        mode_series = df_processed.select(pl.col(col).mode().first()).get_column(col)
        
        if mode_series.is_empty() or mode_series[0] is None:
            actual_mode_to_fill = "Unknown_Mode_Cat"
        else:
            actual_mode_to_fill = mode_series[0]
            
        df_processed = df_processed.with_columns(pl.col(col).fill_null(actual_mode_to_fill))

    # Handle string columns (cast to categorical after imputation)
    string_cols = ["education_1138M", "familystate_726L", "credtype_587L", "profession_152M", 
                   "cancelreason_3545846M", "district_544M", "postype_4733339M", 
                   "rejectreason_755M", "rejectreasonclient_4145042M", "status_219L"]
    for col in string_cols:
        if col in df_processed.columns:
            if df_processed.schema[col] == pl.Categorical:
                continue

            mode_series = df_processed.select(pl.col(col).mode().first()).get_column(col)

            if mode_series.is_empty() or mode_series[0] is None:
                actual_mode_to_fill = "Unknown_Mode_Str"
            else:
                actual_mode_to_fill = mode_series[0]
            
            df_processed = df_processed.with_columns(
                pl.col(col).fill_null(actual_mode_to_fill).cast(pl.Categorical)
            )

    # Convert time columns
    time_cols = ["employedfrom_700D", "dtlastpmt_581D", "creationdate_885D", "dtlastpmtallstes_3545839D", 
                 "firstnonzeroinstldate_307D", "dateactivated_425D", "approvaldate_319D"]
    for col in time_cols:
        if col in df_processed.columns:
            if df_processed.schema[col] == pl.Utf8: # If it's a string that should be a number
                 df_processed = df_processed.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False) # Try to cast to float, nullify if fails
                 )

            # Now assume it's a numeric type (or became one)
            if df_processed.schema[col].is_numeric():
                df_processed = df_processed.with_columns(
                    (pl.col(col).fill_null(0) / -365.25).alias(f"{col}_years")
                )

    return df_processed

# # Example Usage:
# # Assuming train_df and test_df are Polars DataFrames loaded elsewhere
if 'train_df' in locals() and isinstance(train_df, pl.DataFrame):
    train_processed = preprocess_data(train_df) # Pass the DataFrame directly
    print("Train data processed.")
    print(train_processed.head())
else:
    print("train_df is not a Polars DataFrame or not defined.")

if 'test_df' in locals() and isinstance(test_df, pl.DataFrame):
     test_processed = preprocess_data(test_df) # Pass the DataFrame directly
     print("Test data processed.")
     # print(test_processed.head())
else:
     print("test_df is not a Polars DataFrame or not defined.")

gc.collect()


train_processed.describe()













