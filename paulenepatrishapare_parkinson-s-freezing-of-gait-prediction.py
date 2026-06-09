import numpy as np                   
import pandas as pd                 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
import tsfresh
import os
from sklearn.preprocessing import StandardScaler
import polars as pl
import dask.dataframe as dd
from pathlib import Path            


# File paths for training datasets
defog   = Path('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/train/defog')
notype  = Path('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/train/notype')
tdcsfog = Path('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog')


# List all files, folders, and subfolders
all_files = os.listdir('../input/tlvmc-parkinsons-freezing-gait-prediction')
print('All competition datasets:')
print(all_files)

print()

train_files = os.listdir('../input/tlvmc-parkinsons-freezing-gait-prediction/train')
print('Folders in train:')
print(train_files)

print() 

defog_files = os.listdir(defog)
print('First 10 files in defog:')
print(defog_files[:10])
print(f'Files remaining: {len(defog_files)-10}')

print() 

tdcsfog_files = os.listdir(tdcsfog)
print('First 10 files in tdcsfog:')
print(tdcsfog_files[:10])
print(f'Files remaining: {len(tdcsfog_files)-10}')

print() 

notype_files = os.listdir(notype)
print('First 10 files in notype:')
print(notype_files[:10])
print(f'Files remaining: {len(notype_files)-10}')


def load_files(folder_path: Path) -> pl.DataFrame:
    """
    Loads all CSVs from a folder into a single Polars DataFrame.
    Adds patient_id from file filename. 
    """
    
    df_list = []
    for path in folder_path.glob("*.csv"): 
        patient_id = path.stem
        df = pl.read_csv(path)
        df = df.with_columns(pl.lit(patient_id).alias("patient_id"))
        df_list.append(df)
    return pl.concat(df_list) if df_list else pl.DataFrame()
    

def convert_valid_and_task(df: pl.DataFrame) -> pl.DataFrame:
    """
    Cast Valid and Task columns to Int8 if present.
    """
    out = df
    if "Valid" in out.columns:
        out = out.with_columns(pl.col("Valid").cast(pl.Int8).alias("Valid"))
    if "Task" in out.columns:
        out = out.with_columns(pl.col("Task").cast(pl.Int8).alias("Task"))
    return out
    

def add_acc_magnitude(df: pl.DataFrame) -> pl.DataFrame:
    """
    Acc magnitude = sqrt(AccV^2 + AccML^2 + AccAP^2).
    """
    
    return df.with_columns(
        ((pl.col("AccV")**2 + pl.col("AccML")**2 + pl.col("AccAP")**2).sqrt())
        .alias("Acc_MAGNITUDE")
    )


def standardize_acc_by_patient(df: pl.DataFrame) -> pl.DataFrame:
    """
    Z-score AccV/AccML/AccAP per patient_id.
    """
    
    acc_cols = ["AccV", "AccML", "AccAP"]
    out = df
    for col in acc_cols:
        if col in out.columns:
            out = out.with_columns(
                ((pl.col(col) - pl.col(col).mean().over("patient_id")) /
                 pl.col(col).std().over("patient_id")).alias(col)
            )
    return out

def detect_outliers(df: pl.DataFrame, z_thresh: float = 3.0) -> pl.DataFrame:
    """
    Return rows where any accel channel has |z| > z_thresh (after standardization).
    """
    
    acc_cols = ["AccV", "AccML", "AccAP"]
    # Build a boolean mask across columns that exist
    masks = []
    for c in acc_cols:
        if c in df.columns:
            masks.append(pl.col(c).abs() > z_thresh)
    return df.filter(pl.any_horizontal(masks)) if masks else df.head(0)


def infer_fs(time_series: np.ndarray) -> float:
    """
    Infer sampling frequency (Hz) from a time array (seconds) using median Δt.
    """
    
    # Avoid zeros / NaNs
    dt = np.diff(time_series.astype(float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Cannot infer sampling frequency from Time column.")
    return 1.0 / np.median(dt)


def butter_bandpass(low_hz: float, high_hz: float, fs: float, order: int = 4):
    """
    Design a Butterworth band-pass filter.
    """
    
    nyq = fs / 2.0
    low = max(1e-6, low_hz / nyq)
    high = min(0.999999, high_hz / nyq)
    if high <= low:
        raise ValueError(f"Invalid band: low={low_hz}Hz, high={high_hz}Hz for fs={fs}Hz")
    b, a = butter(order, [low, high], btype="band")
    return b, a



def apply_bandpass_to_polars(df: pl.DataFrame, low_hz: float = 0.1, high_hz: float = 30.0, order: int = 4) -> pl.DataFrame:
    """
    Apply band-pass filter to AccV/AccML/AccAP per patient using scipy.signal.filtfilt.
    We convert each patient's slice to pandas to interpolate small gaps, filter, then write back.
    Sampling frequency is inferred from Time (seconds).
    """
    needed = ["Time", "patient_id", "AccV", "AccML", "AccAP"]
    for col in ["Time", "patient_id"]:
        if col not in df.columns:
            raise ValueError(f"'{col}' must be present to filter.")
    # Work columns that actually exist
    acc_cols = [c for c in ["AccV", "AccML", "AccAP"] if c in df.columns]
    if not acc_cols:
        return df

    # Process per patient to keep borders clean for filtfilt
    out_slices = []
    for pid, pdf in df.select(needed).to_pandas().groupby("patient_id", sort=False):
        # Ensure sorted by time
        pdf = pdf.sort_values("Time", kind="mergesort")
        fs = infer_fs(pdf["Time"].to_numpy())
        b, a = butter_bandpass(low_hz, high_hz, fs, order)

        # Interpolate small gaps and filter each channel
        for col in acc_cols:
            # to_numeric + interpolate to avoid NaNs breaking filtfilt
            sig = pd.to_numeric(pdf[col], errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)
            # If the segment is too short, skip filtering
            if sig.size > max(3 * max(len(b), len(a)), 20):
                sig_f = filtfilt(b, a, sig, method="pad")
            else:
                sig_f = sig
            pdf[col] = sig_f

        out_slices.append(pdf)

    # Merge all filtered patient slices back
    filtered_pd = pd.concat(out_slices, ignore_index=True)

    # Join the filtered columns back onto original Polars df (keeping any extra columns intact)
    filtered_pl = pl.from_pandas(filtered_pd[["Time", "patient_id"] + acc_cols])
    out = df.join(filtered_pl, on=["Time", "patient_id"], how="left", suffix="_f")
    # Overwrite originals with filtered where available
    for col in acc_cols:
        fcol = f"{col}_f"
        out = out.with_columns(
            pl.when(pl.col(fcol).is_not_null()).then(pl.col(fcol)).otherwise(pl.col(col)).alias(col)
        ).drop(fcol)
    return out



defog_df   = load_files(defog)
notype_df  = load_files(notype)
tdcsfog_df = load_files(tdcsfog)


print(defog_df.head())
print(defog_df.describe())
print(defog_df.shape)
print(defog_df.columns)
print(defog_df.dtypes)

print(notype_df.head())
print(notype_df.shape)
print(notype_df.columns)
print(notype_df.dtypes)
print(notype_df.describe())

print(tdcsfog_df.head())
print(tdcsfog_df.shape)
print(tdcsfog_df.columns)
print(tdcsfog_df.dtypes)
print(tdcsfog_df.describe())

events_df = pd.read_csv('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/events.csv')
print(events_df.head())
print(events_df.shape)
print(events_df.columns)
print(events_df.dtypes)
print(events_df.describe())

unique_defog_patients = defog_df["patient_id"].unique()
print(unique_defog_patients)



G_CONVERSION = 9.80665
for df_name in ["defog_df", "notype_df", "tdcsfog_df"]:
    df_tmp = locals()[df_name]
    acc_cols = [c for c in ["AccV", "AccML", "AccAP"] if c in df_tmp.columns]
    if acc_cols:
        # Multiply in-place using with_columns to avoid SettingWithCopy issues
        locals()[df_name] = df_tmp.with_columns([ (pl.col(c) * G_CONVERSION).alias(c) for c in acc_cols ])

print(defog_df)
print(notype_df)
print(tdcsfog_df)


defog_df   = convert_valid_and_task(defog_df)
notype_df  = convert_valid_and_task(notype_df)
tdcsfog_df = convert_valid_and_task(tdcsfog_df)
print(defog_df)



defog_df   = apply_bandpass_to_polars(defog_df,   low_hz=0.1, high_hz=30.0, order=4)
notype_df  = apply_bandpass_to_polars(notype_df,  low_hz=0.1, high_hz=30.0, order=4)
tdcsfog_df = apply_bandpass_to_polars(tdcsfog_df, low_hz=0.1, high_hz=30.0, order=4)


defog_df   = add_acc_magnitude(defog_df)
notype_df  = add_acc_magnitude(notype_df)
tdcsfog_df = add_acc_magnitude(tdcsfog_df)


def outlier_acc_magnitude(df: pl.DataFrame) -> pl.DataFrame:
    df_outlier = df.with_columns((pl.col('Acc_MAGNITUDE') > 30).alias('is_outlier_mag'))
    return df_outlier.filter(pl.col("is_outlier_mag") == True)

print(outlier_acc_magnitude(defog_df))
print(outlier_acc_magnitude(notype_df))
print(outlier_acc_magnitude(tdcsfog_df))


defog_df   = standardize_acc_by_patient(defog_df)
notype_df  = standardize_acc_by_patient(notype_df)
tdcsfog_df = standardize_acc_by_patient(tdcsfog_df)

# 10b) Check z-score outliers (|z| > 3)
print(detect_outliers(defog_df))
print(detect_outliers(notype_df))
print(detect_outliers(tdcsfog_df))


print(f'DEFOG DATA TYPES:\n{defog_df.dtypes}\n')
print(f'TDCSFOG DATA TYPES:\n{tdcsfog_df.dtypes}\n')
print(f'NOTYPE DATA TYPES:\n{notype_df.dtypes}\n')

# Null counts
print(defog_df.null_count())
print(notype_df.null_count())
print(tdcsfog_df.null_count())

