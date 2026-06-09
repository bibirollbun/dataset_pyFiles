# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns   
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
# import seglearn as sglearn        # For windowing and sequence modeling
import tsfresh     
import os
from sklearn.preprocessing import StandardScaler


 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import polars as pl
import dask.dataframe as dd
from pathlib import Path


# File paths for three training datasets
defog = Path('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/train/defog')
notype = Path('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/train/notype')
tdcsfog = Path('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog')


defog_files = [f for f in os.listdir(defog) if f.endswith('.csv')]

# List to store individual DataFrames
defog_list = []

for path in defog.glob("*.csv"):
    patient_id = path.stem  # removes .csv

    df = pl.read_csv(path)
    df = df.with_columns([
        pl.lit(patient_id).alias("patient_id")
    ])
    
    defog_list.append(df)

defog_df = pl.concat(defog_list)
# for f in defog_files:
#     file_path = os.path.join(defog, f)
#     df = pl.read_csv(file_path)
#     df = df.with_columns([
#         pl.lit(f).alias('file')  # Add filename as identifier
#     ])
#     defog_list.append(df)

# # Concatenate into one large DataFrame
# defog_df = pl.concat(defog_list)


defog_df.head()


tdcsfog_files = [f for f in os.listdir(tdcsfog) if f.endswith('.csv')]

# List to store individual DataFrames
tdcsfog_list = []

for path in tdcsfog.glob("*.csv"):
    patient_id = path.stem  # removes .csv

    df = pl.read_csv(path)
    df = df.with_columns([
        pl.lit(patient_id).alias("patient_id")
    ])
    
    tdcsfog_list.append(df)

tdcsfog_df = pl.concat(tdcsfog_list)


tdcsfog_df.head()


print(defog_df.head())
# print(defog_df.info())
print(defog_df.describe())
print(defog_df.shape)     # (rows, columns)
print(defog_df.columns)   # list of column names
print(defog_df.dtypes)    # list of column types


print(tdcsfog_df.head())
# print(tdcsfog_df.info())
print(tdcsfog_df.shape)     # (rows, columns)
print(tdcsfog_df.columns)   # list of column names
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


# 1. Filter your Polars DF for a single patient and convert to pandas
df = defog_df.filter(pl.col("patient_id") == 'be9d33541d').to_pandas()

# 2. Plot
plt.figure(figsize=(15, 6))

# Plot acceleration
plt.plot(df['Time'], df['AccV'], label='AccV', alpha=0.7)
plt.plot(df['Time'], df['AccML'], label='AccML', alpha=0.7)
plt.plot(df['Time'], df['AccAP'], label='AccAP', alpha=0.7)

# 3. Plot events
plt.plot(df['Time'], df['StartHesitation'], label='StartHesitation', alpha=0.7)
plt.plot(df['Time'], df['Turn'], label='Turn', alpha=0.7)
plt.plot(df['Time'], df['Walking'], label='Walking', alpha=0.7)


# 4. Final touches
plt.xlabel("Time")
plt.ylabel("Acceleration (g)")
plt.title(f"Patient: {patient_id} - Acceleration + FOG Events")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()


# Data types of features 
print(f'DEFOG DATA TYPES:\n{defog_df.dtypes}\n')
print(f'TDCSFOG DATA TYPES:\n{tdcsfog_df.dtypes}\n')


print(tdcsfog_df.null_count())


# Convert accerlations in defog to m/s^2
G_CONVERSION = 9.80665
defog_df[["AccV", "AccML", "AccAP"]] *= G_CONVERSION
print(defog_df)


# Convert the Valid and Task Columns to Integer Columns
def convert_valid_and_t(df):
    df = df.with_columns(
        pl.col("Valid").cast(pl.Int8).alias("Valid")
    )
    
    df = df.with_columns(
        pl.col("Task").cast(pl.Int8).alias("Task")
    )
    return df
defog_df = convert_valid_and_t(defog_df)
# tdcsfog_df = convert_valid_and_t(tdcsfog_df)


print(defog_df)


# Create a new column that contains the acceleration magnitude
def acc_magnitude(df):
    df = df.with_columns(
        (
            (pl.col("AccV") ** 2 + pl.col("AccML") ** 2 + pl.col("AccAP") ** 2).sqrt()
        ).alias("Acc_MAGNITUDE")
    )

    return df

tdcsfog_df = acc_magnitude(tdcsfog_df)
defog_df = acc_magnitude(defog_df)
defog_df


# Standardize acceleration per patient for each training dataframe
def standardize_acc_by_patient(df: pl.DataFrame):
    acc_cols = ['AccV', 'AccML', 'AccAP']
    for col in acc_cols:
        df = df.with_columns(
            (
                (pl.col(col) - pl.col(col).mean().over("patient_id")) /
                pl.col(col).std().over("patient_id")
            ).alias(col)  # overwrite original column
        )
    return df

tdcsfog_df = standardize_acc_by_patient(tdcsfog_df)
defog_df = standardize_acc_by_patient(defog_df)
defog_df


# Band-pass Filter 
def infer_fs(time_seconds: np.ndarray) -> float:
    dt = np.diff(np.asarray(time_seconds, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Cannot infer sampling frequency from Time column.")
    return 1.0 / np.median(dt)

def design_bandpass(low_hz: float, high_hz: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    low = max(1e-6, low_hz / nyq)
    high = min(0.999999, high_hz / nyq)
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid band for fs={fs:.3f}Hz: low={low_hz}Hz, high={high_hz}Hz")
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_series(y: pd.Series, b, a) -> np.ndarray:
    sig = pd.to_numeric(y, errors="coerce").interpolate(limit_direction="both").to_numpy(float)
    return filtfilt(b, a, sig, method="pad")

def bandpass_dataframe(df: pd.DataFrame, cols=('AccV','AccML','AccAP'),
                       low_hz=0.1, high_hz=30.0, order=4) -> pd.DataFrame:
    out = df.copy()
    # Only keep columns that exist
    cols = tuple([c for c in cols if c in out.columns])
    if len(cols) == 0:
        return out

    fs = infer_fs(out['Time'].to_numpy())
    b, a = design_bandpass(low_hz, high_hz, fs, order)
    for col in cols:
        out[f"{col}_bp"] = bandpass_series(out[col], b, a)
    return out




# Apply Band-pass to all patients 
def add_bandpass_to_all_patients(pl_df: pl.DataFrame,
                                 cols=('AccV','AccML','AccAP'),
                                 low_hz=0.1, high_hz=30.0, order=4) -> pl.DataFrame:
    if "patient_id" not in pl_df.columns:
        raise ValueError("Expected a 'patient_id' column.")

    out_chunks = []
    # Unique patient list
    patient_ids = pl_df.select("patient_id").unique().to_series().to_list()

    for pid in patient_ids:
        g = pl_df.filter(pl.col("patient_id") == pid).to_pandas()
        # Skip tiny or malformed groups
        if "Time" not in g.columns or len(g) < 5:
            out_chunks.append(pl.from_pandas(g))  # just pass through
            continue

        try:
            g_bp = bandpass_dataframe(g, cols=cols, low_hz=low_hz, high_hz=high_hz, order=order)
        except Exception as e:
            print(f"[WARN] Skipping bandpass for patient {pid}: {e}")
            g_bp = g  # pass through raw if something fails

        out_chunks.append(pl.from_pandas(g_bp))

    return pl.concat(out_chunks, how="vertical_relaxed")

defog_df_bp   = add_bandpass_to_all_patients(defog_df,   cols=('AccV','AccML','AccAP'),
                                             low_hz=0.1, high_hz=30.0, order=4)
tdcsfog_df_bp = add_bandpass_to_all_patients(tdcsfog_df, cols=('AccV','AccML','AccAP'),
                                             low_hz=0.1, high_hz=30.0, order=4)

print("DEFOG with band-pass columns:", [c for c in defog_df_bp.columns if c.endswith("_bp")][:6], "...")
print("TDCSFOG with band-pass columns:", [c for c in tdcsfog_df_bp.columns if c.endswith("_bp")][:6], "...")


def add_magnitude_cols(pl_df: pl.DataFrame) -> pl.DataFrame: 
    out = pl_df.with_columns(
        ((pl.col("AccV")**2 + pl.col("AccML")**2 + pl.col("AccAP")**2).sqrt()).alias("AccMag")
    )
    bp_cols = {"AccV_bp", "AccML_bp", "AccAP_bp"}
    if bp_cols.issubset(set(out.columns)):
        out = out.with_columns(
            ((pl.col("AccV_bp")**2 + pl.col("AccML_bp")**2 + pl.col("AccAP_bp")**2).sqrt()).alias("AccMag_bp")
        )
    return out 
    
def plot_patient_mag(pl_df: pl.DataFrame, patient_id: str,
                     time_col: str = "Time",
                     show_events: bool = True,
                     title_suffix: str = ""):
    dfp = pl_df.filter(pl.col("patient_id") == patient_id).to_pandas()
    if not {"AccMag", time_col}.issubset(dfp.columns):
        raise ValueError("Missing magnitude or time column")
    plt.figure(figsize=(16,6))
    plt.plot(dfp[time_col], dfp["AccMag"], label="|a| (raw)", alpha=0.7)
    if "AccMag_bp" in dfp.columns:
        plt.plot(dfp[time_col], dfp["AccMag_bp"], label="|a| (0.1–30 Hz)", linewidth=1.6)
    if show_events:
        for ev in ["StartHesitation", "Turn", "Walking"]:
            if ev in dfp.columns:
                plt.plot(dfp[time_col], dfp[ev], label=ev, alpha=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration magnitude")
    plt.title(f"Patient {patient_id} – Acc Magnitude {title_suffix}")
    plt.legend(ncol=3)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Add magnitude columns to dataframes
defog_df_bp = add_magnitude_cols(defog_df_bp)
tdcsfog_df_bp = add_magnitude_cols(tdcsfog_df_bp)
defog_df = add_magnitude_cols(defog_df)
tdcsfog_df = add_magnitude_cols(tdcsfog_df)

# Plot magnitude for one patient
plot_patient_mag(defog_df_bp, patient_id="4c3aa8ea6e", title_suffix="(raw vs band-pass)")



# Create a new column that contains Time as seconds
def time_to_seconds(df, hertz):
    df = df.with_columns(
        (
            (pl.col("Time") / hertz)
        ).alias("Time (seconds)")
    )

    return df

tdcsfog_df = time_to_seconds(tdcsfog_df, 128)
defog_df = time_to_seconds(tdcsfog_df, 100)
defog_df


# Check for outliers from acceleration
def detect_outliers(df: pl.DataFrame):
    acc_cols = ['AccV', 'AccML', 'AccAP']
    for col in acc_cols: 
        z_col = col
        outlier_df = df.filter(pl.col(z_col).abs() > 3.0)
    return outlier_df
print(detect_outliers(defog_df))
print(detect_outliers(tdcsfog_df))


# Get unique patient IDs with a StartHesitation, Turn, and Walking event
# Take a subset of 3 patients for each event
StartHesPatients = (
    defog_df.filter(pl.col("StartHesitation") == 1)
            .select("patient_id")
            .unique()
            .to_series()[:3]  # take first 3
)
print(f"Patients with Start Hesitation: {StartHesPatients.to_list()}")

TurnPatients = (
    defog_df.filter(pl.col("Turn") == 1)
            .select("patient_id")
            .unique()
            .to_series()[:3]
)
print(f"Patients with Turn: {TurnPatients.to_list()}")

WalkingPatients = (
    defog_df.filter(pl.col("Walking") == 1)
            .select("patient_id")
            .unique()
            .to_series()[:3]
)
print(f"Patients with Walking: {WalkingPatients.to_list()}")


# Get unique patient IDs with a StartHesitation, Turn, and Walking event 
# (including band-pass)
if {"StartHesitation","Turn","Walking"}.issubset(set(defog_df_bp.columns)):
    StartHesPatients = (
        defog_df_bp.filter(pl.col("StartHesitation") == 1)
                   .select("patient_id").unique().to_series()[:3]
    )
    TurnPatients = (
        defog_df_bp.filter(pl.col("Turn") == 1)
                   .select("patient_id").unique().to_series()[:3]
    )
    WalkingPatients = (
        defog_df_bp.filter(pl.col("Walking") == 1)
                   .select("patient_id").unique().to_series()[:3]
    )
    print(f"Patients with Start Hesitation: {StartHesPatients.to_list()}")
    print(f"Patients with Turn: {TurnPatients.to_list()}")
    print(f"Patients with Walking: {WalkingPatients.to_list()}")


# Start Hestitation
# 1. Filter your Polars DF for a single patient and convert to pandas
df = defog_df.filter(pl.col("patient_id") == '81262644e7').to_pandas()

# 2. Plot
plt.figure(figsize=(15, 6))

# Plot acceleration
plt.plot(df['Time'], df['AccV'], label='AccV', alpha=0.7)
plt.plot(df['Time'], df['AccML'], label='AccML', alpha=0.7)
plt.plot(df['Time'], df['AccAP'], label='AccAP', alpha=0.7)

# 3. Plot events
plt.plot(df['Time'], df['StartHesitation'], label='StartHesitation', alpha=0.7)


# 4. Final touches
plt.xlabel("Time")
plt.ylabel("Acceleration (g)")
plt.title(f"Patient: {patient_id} - Acceleration + FOG Events")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()


# Start Hestitation
# 1. Filter your Polars DF for a single patient and convert to pandas
df = defog_df.filter(pl.col("patient_id") == '3ba3590a08').to_pandas()

# 2. Plot
plt.figure(figsize=(15, 6))

# Plot acceleration
plt.plot(df['Time'], df['AccV'], label='AccV', alpha=0.7)
plt.plot(df['Time'], df['AccML'], label='AccML', alpha=0.7)
plt.plot(df['Time'], df['AccAP'], label='AccAP', alpha=0.7)

# 3. Plot events
plt.plot(df['Time'], df['StartHesitation'], label='StartHesitation', alpha=0.7)


# 4. Final touches
plt.xlabel("Time")
plt.ylabel("Acceleration (g)")
plt.title(f"Patient: {patient_id} - Acceleration + FOG Events")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()



# Start Hestitation
# 1. Filter your Polars DF for a single patient and convert to pandas
df = defog_df.filter(pl.col("patient_id") == 'd98358a75f').to_pandas()

# 2. Plot
plt.figure(figsize=(15, 6))

# Plot acceleration
plt.plot(df['Time'], df['AccV'], label='AccV', alpha=0.7)
plt.plot(df['Time'], df['AccML'], label='AccML', alpha=0.7)
plt.plot(df['Time'], df['AccAP'], label='AccAP', alpha=0.7)

# 3. Plot events
plt.plot(df['Time'], df['StartHesitation'], label='StartHesitation', alpha=0.7)


# 4. Final touches
plt.xlabel("Time")
plt.ylabel("Acceleration (g)")
plt.title(f"Patient: {patient_id} - Acceleration + FOG Events")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()


def extract_time_features_pd(
    df: pd.DataFrame,
    fs: float,
    win_s: float,
    hop_s: float,
    signal_cols=('AccX','AccY','AccZ','GyroX','GyroY','GyroZ'),
    label_cols=None,                 # e.g. ['StartHesitation','Turn','Walking','Event']
    id_cols=None,                    # e.g. ['subject_id','series_id','id'] to carry through
):
    df = df.copy()
    n = len(df)
    win = int(round(fs*win_s))
    hop = int(round(fs*hop_s))
    if win <= 0 or hop <= 0:
        raise ValueError("win_s and hop_s must be > 0")

    # Pre-pull arrays for speed
    X = df.loc[:, signal_cols].to_numpy(dtype=float)

    # Helper feature fns (safe on NaNs/empties)
    def feats_one(w):
        f = {}
        # basic stats
        f.update({f'{c}_mean': np.nanmean(w[:,i]) for i,c in enumerate(signal_cols)})
        f.update({f'{c}_var' : np.nanvar (w[:,i]) for i,c in enumerate(signal_cols)})
        f.update({f'{c}_std' : np.nanstd (w[:,i]) for i,c in enumerate(signal_cols)})
        f.update({f'{c}_min' : np.nanmin (w[:,i]) for i,c in enumerate(signal_cols)})
        f.update({f'{c}_max' : np.nanmax (w[:,i]) for i,c in enumerate(signal_cols)})
        f.update({f'{c}_median': np.nanmedian(w[:,i]) for i,c in enumerate(signal_cols)})
        f.update({f'{c}_iqr': np.nanpercentile(w[:,i],75)-np.nanpercentile(w[:,i],25) for i,c in enumerate(signal_cols)})

        # energy & rms
        f.update({f'{c}_energy': np.nansum(np.square(w[:,i]))/len(w) for i,c in enumerate(signal_cols)})
        f.update({f'{c}_rms'   : np.sqrt(np.nanmean(np.square(w[:,i]))) for i,c in enumerate(signal_cols)})

        # skew & kurt
        for i,c in enumerate(signal_cols):
            col = w[:,i]
            f[f'{c}_skew'] = skew(col, nan_policy='omit', bias=False)
            f[f'{c}_kurt'] = kurtosis(col, nan_policy='omit', fisher=True, bias=False)

        # vector features for tri-axial groups
        if set(['AccX','AccY','AccZ']).issubset(signal_cols):
            ax = [signal_cols.index('AccX'), signal_cols.index('AccY'), signal_cols.index('AccZ')]
            acc = w[:,ax]
            mag = np.sqrt(np.sum(acc**2, axis=1))
            f['Acc_mag_mean'] = np.nanmean(mag)
            f['Acc_mag_std']  = np.nanstd(mag)
            # Signal Magnitude Area (SMA)
            f['Acc_sma'] = (np.nansum(np.abs(acc), axis=0).sum()) / len(mag)

            # correlations
            for (a,b) in [('AccX','AccY'),('AccX','AccZ'),('AccY','AccZ')]:
                i1, i2 = signal_cols.index(a), signal_cols.index(b)
                col1, col2 = w[:,i1], w[:,i2]
                if np.all(np.isfinite(col1)) and np.all(np.isfinite(col2)) and len(col1) > 1:
                    f[f'corr_{a}_{b}'] = np.corrcoef(col1, col2)[0,1]
                else:
                    f[f'corr_{a}_{b}'] = np.nan

        if set(['GyroX','GyroY','GyroZ']).issubset(signal_cols):
            gx = [signal_cols.index('GyroX'), signal_cols.index('GyroY'), signal_cols.index('GyroZ')]
            gyro = w[:,gx]
            mag = np.sqrt(np.sum(gyro**2, axis=1))
            f['Gyro_mag_mean'] = np.nanmean(mag)
            f['Gyro_mag_std']  = np.nanstd(mag)
            f['Gyro_sma'] = (np.nansum(np.abs(gyro), axis=0).sum()) / len(mag)
            for (a,b) in [('GyroX','GyroY'),('GyroX','GyroZ'),('GyroY','GyroZ')]:
                i1, i2 = signal_cols.index(a), signal_cols.index(b)
                col1, col2 = w[:,i1], w[:,i2]
                if np.all(np.isfinite(col1)) and np.all(np.isfinite(col2)) and len(col1) > 1:
                    f[f'corr_{a}_{b}'] = np.corrcoef(col1, col2)[0,1]
                else:
                    f[f'corr_{a}_{b}'] = np.nan

        return f

    rows = []
    start_idx = 0
    win_id = 0
    while start_idx + win <= n:
        end_idx = start_idx + win
        w = X[start_idx:end_idx, :]
        feat = feats_one(w)
        # add time/window metadata
        feat['win_id'] = win_id
        feat['t_start_s'] = start_idx / fs
        feat['t_end_s']   = (end_idx-1) / fs

        # bring-through IDs from the *end* row of the window (common in DEFoG baselines)
        if id_cols:
            for c in id_cols:
                feat[c] = df.iloc[end_idx-1][c]

        # aggregate labels if provided (ANY>0 inside window)
        if label_cols:
            sub = df.iloc[start_idx:end_idx]
            feat['label_any'] = bool((sub[label_cols].fillna(0).to_numpy() > 0).any())
            for c in label_cols:
                feat[f'label_{c}'] = bool((sub[c].fillna(0).to_numpy() > 0).any())

        rows.append(feat)
        win_id += 1
        start_idx += hop

    return pd.DataFrame(rows)


DATA_DIR = Path("/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/train")

print("Available training folders:", [p.name for p in DATA_DIR.iterdir()])


tdcsfog_file = list((DATA_DIR / "tdcsfog").glob("*.csv"))[0]
defog_file   = list((DATA_DIR / "defog").glob("*.csv"))[0]
notype_file  = list((DATA_DIR / "notype").glob("*.csv"))[0]

print("Sample files chosen:")
print("tdcsfog:", tdcsfog_file.name)
print("defog:", defog_file.name)
print("notype:", notype_file.name)



tdcsfog_df = pd.read_csv(tdcsfog_file)
defog_df   = pd.read_csv(defog_file)
notype_df  = pd.read_csv(notype_file)


import re

def tdcsfog_time_features(df, fs=128.0, win_s=2.0, hop_s=0.5):
    """
    Extract time-domain features from accelerometer data in tdcsfog_df.
    Works purely in pandas.
    """

    # --- auto-detect accelerometer columns ---
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    candidates = [
        ['AccX','AccY','AccZ'],
        ['AccelX','AccelY','AccelZ'],
        ['acc_x','acc_y','acc_z'],
        ['AccV','AccML','AccAP'],
        ['accv','accml','accap'],
    ]
    acc_cols = None
    for trio in candidates:
        found = [lower.get(c.lower()) for c in trio]
        if all(found):
            acc_cols = found
            break
    if acc_cols is None:
        acc_cols = [c for c in cols if re.search('acc', c, re.I)][:3]
    if len(acc_cols) < 3:
        raise KeyError(f"Could not find 3 accelerometer columns. Found: {acc_cols}")

    # --- label columns ---
    label_cols = [c for c in ['StartHesitation','Turn','Walking'] if c in df.columns]

    # --- window setup ---
    win = int(round(fs * win_s))
    hop = int(round(fs * hop_s))
    n = len(df)

    X = df[acc_cols].to_numpy(dtype=float)
    rows = []
    start = 0
    win_id = 0

    while start + win <= n:
        end = start + win
        W = X[start:end, :]
        f = {}

        # per-axis features
        for i, c in enumerate(acc_cols):
            w = W[:, i]
            f[f'{c}_mean']   = np.nanmean(w)
            f[f'{c}_std']    = np.nanstd(w)
            f[f'{c}_var']    = np.nanvar(w)
            f[f'{c}_min']    = np.nanmin(w)
            f[f'{c}_max']    = np.nanmax(w)
            f[f'{c}_median'] = np.nanmedian(w)
            q75, q25 = np.nanpercentile(w, [75, 25])
            f[f'{c}_iqr']    = q75 - q25
            f[f'{c}_energy'] = np.nansum(w**2) / len(w)
            f[f'{c}_rms']    = np.sqrt(np.nanmean(w**2))
            f[f'{c}_skew']   = skew(w, nan_policy='omit', bias=False)
            f[f'{c}_kurt']   = kurtosis(w, nan_policy='omit', fisher=True, bias=False)

        # vector magnitude features
        mag = np.sqrt(np.sum(W**2, axis=1))
        f['Acc_mag_mean'] = np.nanmean(mag)
        f['Acc_mag_std']  = np.nanstd(mag)
        f['Acc_sma']      = np.nansum(np.abs(W)) / len(W)

        # correlations
        def corr_safe(a,b):
            if len(a) > 1 and np.isfinite(a).all() and np.isfinite(b).all():
                return np.corrcoef(a,b)[0,1]
            return np.nan
        f[f'corr_{acc_cols[0]}_{acc_cols[1]}'] = corr_safe(W[:,0], W[:,1])
        f[f'corr_{acc_cols[0]}_{acc_cols[2]}'] = corr_safe(W[:,0], W[:,2])
        f[f'corr_{acc_cols[1]}_{acc_cols[2]}'] = corr_safe(W[:,1], W[:,2])

        # labels (if exist)
        if label_cols:
            sub = df.iloc[start:end][label_cols].fillna(0).to_numpy()
            f['label_any'] = bool((sub > 0).any())
            for j, c in enumerate(label_cols):
                f[f'label_{c}'] = bool((sub[:, j] > 0).any())

        # window metadata
        f['win_id'] = win_id
        f['t_start_s'] = start / fs
        f['t_end_s']   = (end - 1) / fs

        rows.append(f)
        win_id += 1
        start += hop

    return pd.DataFrame(rows)


# parameters
FS = 128.0
WIN = 2.0   # 2 seconds per window
HOP = 0.5   # 0.5-second step

tdcsfog_feats = tdcsfog_time_features(tdcsfog_df, fs=FS, win_s=WIN, hop_s=HOP)

print("Detected accelerometer columns:", [c for c in tdcsfog_feats.columns if 'mean' in c][:3])
print("Shape:", tdcsfog_feats.shape)
tdcsfog_feats.head()


from scipy.fft import rfft, rfftfreq      # Fast Fourier Transform functions + corresponding frequency bins for real-valued signals 
from scipy.signal import welch            # Power Spectral Density function (shows how signal's power varies across frequencies)

def extract_frequency_features(df, fs=128.0, win_s=2.0, hop_s=0.5, signal_cols=(('AccX','AccY','AccZ'))):
    """
    Extracts frequency domain features from accelerometer data
    Inputs: 
    - df: pandas DataFrame that has time-series sensor data 
    - fs: sampling frequency in Hz (how many readings per second)
    - win_s: window size in seconds 
    - hop_s: step size in seconds (how far we slide each window)
    - signal_cols: columns to use (like accelerometer axes)
    """

    df = df.copy()                 # make a copy so we don't modify the original datafram e
    n = len(df)                    # number of total samples (rows)
    win = int(round(fs* win_s))    # window size in samples 
    hop = int(round(fs * hop_s))   # hop size in samples
    rows = []                      # to store feature dictionaries for each window 
    start = 0                      # start index for the first window 
    win_id = 0                     # window counter 

    # Loop through the data using sliding windows 
    while start + win <= n:
        end = start + win
        # get window segment (subset of rows)
        segment = df.iloc[start:end][list(signal_cols)].to_numpy(dtype=float)

        # prepare frquency bins (x-axis for FFT)
        freqs = rfftfreq(win, d=1/fs)      # rfft only returns positive frequencies 

        fdict = {}   # dictionary to store the feature for this window 

        # Loop through each sensor column (AccX, AccY, AccZ)
        for i, col in enumerate(signal_cols): 
            signal = segment [:, i]               # extract one axis 
            signal = signal - np.nanmean(signal)  # center around 0 
            signal = np.nan_to_num(signal)        # replace NaNs with 0s (avoid FFT errors)

            # FFT: convert from time to frequency 
            fft_vals = np.abs(rfft(signal))           # absolute value of FFT (magnitude)
            fft_power = fft_vals ** 2                 # power = magnitude squared 

            # PSD - Power Spectral Density 
            f_psd, psd = welch(signal, fs=fs, nperseg=min(win, len(signal)))

            # Basic FFT & PSD feature summaries to help distinguish steady vs irregular movements 
            fdict[f'{col}_fft_mean'] = np.mean(fft_power)   # average FFT power
            fdict[f'{col}_fft_max']  = np.max(fft_power)    # peak FFT power
            fdict[f'{col}_fft_std']  = np.std(fft_power)    # variation in FFT power

            fdict[f'{col}_psd_mean'] = np.mean(psd)         # average power across frequencies
            fdict[f'{col}_psd_max']  = np.max(psd)          # maximum power (dominant peak)
            fdict[f'{col}_psd_std']  = np.std(psd)          # variability in PSD

            # Dominant Frequency 
            # frequency (in Hz) where PSD is largest
            dom_freq = f_psd[np.argmax(psd)]
            fdict[f'{col}_dominant_freq'] = dom_freq

            # Band Power (energy in specific frequency ranges) 
            # Integrate (area under curve) power in low, mid, and high frequency ranges
            fdict[f'{col}_bandpower_low']  = np.trapz(psd[(f_psd>=0)  & (f_psd<3)],  f_psd[(f_psd>=0)  & (f_psd<3)])   # 0–3 Hz (large body movements)
            fdict[f'{col}_bandpower_mid']  = np.trapz(psd[(f_psd>=3)  & (f_psd<10)], f_psd[(f_psd>=3)  & (f_psd<10)])  # 3–10 Hz (normal walking frequency)
            fdict[f'{col}_bandpower_high'] = np.trapz(psd[(f_psd>=10) & (f_psd<30)], f_psd[(f_psd>=10) & (f_psd<30)])  # 10–30 Hz (fine tremors/sudden changes)
            
        fdict['win_id'] = win_id                 # window number
        fdict['t_start_s'] = start / fs          # start time (seconds)
        fdict['t_end_s']   = (end - 1) / fs      # end time (seconds)

        rows.append(fdict)    # store results
        win_id += 1           # move to next window
        start += hop          # slide window by hop length

    # return a new DataFrame with all extracted frequency features
    return pd.DataFrame(rows)


tdcsfog_df['patient_id'] = 'patient_1' 
print(tdcsfog_df.columns)


from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters


def extract_tsfresh_features_windowed(
    df, id_col='patient_id', time_col='Time',
    signal_cols=('AccV','AccML','AccAP'),
    fs=128.0, win_s=2.0
):
    """
    Extract tsfresh features per short window (fast and memory-safe).
    Each patient signal is divided into small windows before feature extraction.
    """
    win = int(fs * win_s)
    df = df.copy()
    all_feats = []

    for pid in df[id_col].unique():
        sub = df[df[id_col] == pid].copy().reset_index(drop=True)
        n = len(sub)
        print(f"Extracting tsfresh features for {pid} ({n} samples)")

        # Create window IDs (one per window)
        sub['window_id'] = (sub.index // win).astype(int)

        # Melt into long format
        df_long = sub.melt(
            id_vars=[id_col, 'window_id', time_col],
            value_vars=list(signal_cols),
            var_name='sensor_axis',
            value_name='value'
        ).rename(columns={id_col: 'id', time_col: 'time'})

        # Combine patient_id and window_id into one unique series id
        df_long['id'] = df_long['id'].astype(str) + "_" + df_long['window_id'].astype(str)

        # Clean numeric values
        df_long['time']  = pd.to_numeric(df_long['time'], errors='coerce')
        df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
        df_long = df_long.dropna(subset=['time', 'value']).sort_values(['id', 'time'])

        # Extract light feature set
        feats = extract_features(
            df_long,
            column_id='id',
            column_sort='time',
            column_value='value',
            default_fc_parameters=EfficientFCParameters(),
            n_jobs=0,
            disable_progressbar=False
        )

        feats.reset_index(inplace=True)
        feats[id_col] = pid
        all_feats.append(feats)
        print(f"✓ Done {pid}")

    return pd.concat(all_feats, ignore_index=True) if all_feats else pd.DataFrame()


def combine_features(manual_df, tsfresh_df):
    """
    Merge manual and tsfresh features automatically, even if key columns differ.
    """
    # Auto-detect join column
    common_keys = set(manual_df.columns) & set(tsfresh_df.columns)
    possible_keys = {'id', 'patient_id', 'window_id'}
    join_key = list(common_keys & possible_keys)
    
    if not join_key:
        print("[WARN] No shared key found. Using 'id' by default.")
        manual_df['id'] = manual_df.get('id', 'unknown')
        tsfresh_df['id'] = tsfresh_df.get('id', 'unknown')
        join_key = ['id']
    else:
        join_key = [join_key[0]]  # pick the first match
    
    print(f"Merging on key: {join_key[0]}")
    merged = pd.merge(tsfresh_df, manual_df, how='inner', on=join_key)
    return merged


# Sample testing

# sample_df = tdcsfog_df.head(500).copy()
# sample_df['patient_id'] = 'p_test'

# tsfresh_feats = extract_tsfresh_features_windowed(
    # sample_df,
    # id_col='patient_id',
    # time_col='Time',
    # signal_cols=('AccV','AccML','AccAP'),
    # fs=128.0,
    # win_s=2.0
# )




FS = 128.0
WIN = 2.0
HOP = 0.5

# Manual frequency-domain features
freq_feats = extract_frequency_features(
    tdcsfog_df,
    fs=FS,
    win_s=WIN,
    hop_s=HOP,
    signal_cols=('AccV','AccML','AccAP')
)

# Add patient_id to match tsfresh later
freq_feats['id'] = tdcsfog_df['patient_id'].iloc[0] if 'patient_id' in tdcsfog_df.columns else 'unknown'

print("Manual frequency features shape:", freq_feats.shape)

# tsfresh automatic features
tsfresh_feats = extract_tsfresh_features_windowed(
    tdcsfog_df,
    id_col='patient_id',
    time_col='Time',
    signal_cols=('AccV','AccML','AccAP')
)
print("TSFresh features shape:", tsfresh_feats.shape)

# Ensure both have lowercase 'id' column for merging
tsfresh_feats = tsfresh_feats.rename(columns={'patient_id': 'id'})  # keep one id column
freq_feats = freq_feats.rename(columns={'patient_id': 'id'})        # unify naming just in case

# Drop duplicates if needed
tsfresh_feats = tsfresh_feats.loc[:, ~tsfresh_feats.columns.duplicated()]

# Confirm alignment before merge
print("freq_feats ids:", freq_feats['id'].head().tolist())
print("tsfresh_feats ids:", tsfresh_feats['id'].head().tolist())

# Merge again
features_df = combine_features(freq_feats, tsfresh_feats)
print("✅ Combined feature set shape:", features_df.shape)


import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

plt.rcParams["figure.figsize"] = (6, 4)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("PCA/LDA environment ready.")


# Iris dataset
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target
class_names_iris = iris.target_names

# 1) Standardize
scaler = StandardScaler()
X_iris_std = scaler.fit_transform(X_iris)

# 2) PCA to 2D
pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
X_iris_pca2 = pca2.fit_transform(X_iris_std)

print("Explained variance ratio (2 comps):", np.round(pca2.explained_variance_ratio_, 4))
print("Cumulative (2 comps):", np.round(np.sum(pca2.explained_variance_ratio_), 4))

# 3) Scatter (distinct markers; no explicit colors)
markers = ["o", "s", "^"]
plt.figure()
for i, m in enumerate(markers):
    plt.scatter(X_iris_pca2[y_iris == i, 0], X_iris_pca2[y_iris == i, 1], marker=m, label=class_names_iris[i])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Iris — PCA 2D Projection")
plt.legend()
plt.show()

# 4) Explained variance per PC (k=2)
plt.figure()
plt.bar([1, 2], pca2.explained_variance_ratio_)
plt.xticks([1, 2])
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance Ratio — Iris PCA (k=2)")
plt.show()

# 5) Loadings (components x features)
print("PCA component loadings:\n", np.round(pca2.components_, 4))


# Reconstruction MSE vs. k (Iris)
max_k = X_iris.shape[1]
ks = list(range(1, max_k + 1))
mse_by_k = []

for k in ks:
    p = PCA(n_components=k, random_state=RANDOM_STATE).fit(X_iris_std)
    X_proj = p.transform(X_iris_std)
    X_rec = p.inverse_transform(X_proj)
    mse = np.mean((X_iris_std - X_rec) ** 2)
    mse_by_k.append(mse)

plt.figure()
plt.plot(ks, mse_by_k, marker="o")
plt.xlabel("Number of Components (k)")
plt.ylabel("Mean Squared Reconstruction Error")
plt.title("PCA Reconstruction Error vs. k — Iris")
plt.show()

# Scree & cumulative
pca_full = PCA().fit(X_iris_std)
evar = pca_full.explained_variance_ratio_
cum_evar = np.cumsum(evar)

plt.figure()
plt.plot(range(1, len(evar)+1), evar, marker="o")
plt.xlabel("Component index")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot — Iris")
plt.show()

plt.figure()
plt.plot(range(1, len(cum_evar)+1), cum_evar, marker="o")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance — Iris")
plt.show()

print("Explained variance ratio:", np.round(evar, 4))
print("Cumulative variance:", np.round(cum_evar, 4))


lda2 = LDA(n_components=2)
X_iris_lda2 = lda2.fit_transform(X_iris_std, y_iris)

plt.figure()
for i, m in enumerate(["o", "s", "^"]):
    plt.scatter(X_iris_lda2[y_iris == i, 0], X_iris_lda2[y_iris == i, 1], marker=m, label=class_names_iris[i])
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.title("Iris — LDA 2D Projection")
plt.legend()
plt.show()

print("LDA explained_variance_ratio_:", np.round(lda2.explained_variance_ratio_, 4))


# Wine comparison
wine = datasets.load_wine()
X_wine, y_wine = wine.data, wine.target

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

pipe_lr = Pipeline([("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])

pipe_pca_lr = Pipeline([("scaler", StandardScaler()),
                        ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])

pipe_lda_lr = Pipeline([("scaler", StandardScaler()),
                        ("lda", LDA(n_components=min(len(np.unique(y_wine)) - 1, X_wine.shape[1]))),
                        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])

scores_lr  = cross_val_score(pipe_lr,     X_wine, y_wine, cv=cv, scoring="accuracy")
scores_pca = cross_val_score(pipe_pca_lr, X_wine, y_wine, cv=cv, scoring="accuracy")
scores_lda = cross_val_score(pipe_lda_lr, X_wine, y_wine, cv=cv, scoring="accuracy")

print("Wine accuracy (mean ± std)")
print("  LR only :", np.mean(scores_lr),  "±", np.std(scores_lr))
print("  PCA+LR  :", np.mean(scores_pca), "±", np.std(scores_pca))
print("  LDA+LR  :", np.mean(scores_lda), "±", np.std(scores_lda))


# Digits comparison (64 features)
digits = datasets.load_digits()
X_digits, y_digits = digits.data, digits.target

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

pipe_lr_d = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])

pipe_pca_lr_d = Pipeline([("scaler", StandardScaler()),
                          ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                          ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])

pipe_lda_lr_d = Pipeline([("scaler", StandardScaler()),
                          ("lda", LDA(n_components=min(len(np.unique(y_digits)) - 1, X_digits.shape[1]))),
                          ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])

scores_lr_d  = cross_val_score(pipe_lr_d,     X_digits, y_digits, cv=cv, scoring="accuracy")
scores_pca_d = cross_val_score(pipe_pca_lr_d, X_digits, y_digits, cv=cv, scoring="accuracy")
scores_lda_d = cross_val_score(pipe_lda_lr_d, X_digits, y_digits, cv=cv, scoring="accuracy")

print("Digits accuracy (mean ± std)")
print("  LR only :", np.mean(scores_lr_d),  "±", np.std(scores_lr_d))
print("  PCA+LR  :", np.mean(scores_pca_d), "±", np.std(scores_pca_d))
print("  LDA+LR  :", np.mean(scores_lda_d), "±", np.std(scores_lda_d))


X_tr, X_te, y_tr, y_te = train_test_split(
    X_wine, y_wine, test_size=0.3, random_state=RANDOM_STATE, stratify=y_wine
)

def plot_cm(cm, title):
    plt.figure()
    plt.imshow(cm, aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.show()

# Baseline
pipe_lr.fit(X_tr, y_tr)
y_pred_base = pipe_lr.predict(X_te)
cm_base = confusion_matrix(y_te, y_pred_base)
print("Accuracy (LR only):", accuracy_score(y_te, y_pred_base))
plot_cm(cm_base, "Confusion Matrix — LR only")
print(classification_report(y_te, y_pred_base))

# PCA pipeline
pipe_pca_lr.fit(X_tr, y_tr)
y_pred_pca = pipe_pca_lr.predict(X_te)
cm_pca = confusion_matrix(y_te, y_pred_pca)
print("Accuracy (PCA + LR):", accuracy_score(y_te, y_pred_pca))
plot_cm(cm_pca, "Confusion Matrix — PCA + LR")
print(classification_report(y_te, y_pred_pca))

# LDA pipeline
pipe_lda_lr.fit(X_tr, y_tr)
y_pred_lda = pipe_lda_lr.predict(X_te)
cm_lda = confusion_matrix(y_te, y_pred_lda)
print("Accuracy (LDA + LR):", accuracy_score(y_te, y_pred_lda))
plot_cm(cm_lda, "Confusion Matrix — LDA + LR")
print(classification_report(y_te, y_pred_lda))

