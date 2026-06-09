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


# ============================================================
# NYC Taxi Fare Prediction — Small & Fast (with Graphs)
# ~300k sample, Ridge + 3-fold CV (RMSLE), submission.csv
# Handles train.csv/.zip, test.csv/.zip; silences FutureWarnings
# ============================================================

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)   # silence seaborn/pandas future warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set(style="whitegrid")
    HAS_SNS = True
except Exception:
    HAS_SNS = False

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge

# ----------------------------
# Settings (fast)
# ----------------------------
TARGET_ROWS = 300_000   # ~300k for quick run
CHUNK_SIZE  = 1_000_000
N_FOLDS     = 3

def info(msg): print(f"[INFO] {msg}")

# ----------------------------
# Helpers
# ----------------------------
def haversine_vec(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2*np.arcsin(np.sqrt(a))

def bearing_vec(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon)*np.cos(lat2)
    x = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

def manhattan_distance_approx(lat1, lon1, lat2, lon2):
    lat_km, lon_km = 111.32, 85.39
    return np.abs(lat2-lat1)*lat_km + np.abs(lon2-lon1)*lon_km

def rmsle(y_true, y_pred):
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

# ----------------------------
# Locate files under /kaggle/input
# ----------------------------
def find_files(base="/kaggle/input"):
    train_path = test_path = sample_path = None
    for root, _, files in os.walk(base):
        fl = {f.lower(): f for f in files}
        if "train.csv" in fl: train_path = os.path.join(root, fl["train.csv"])
        elif "train.csv.zip" in fl: train_path = os.path.join(root, fl["train.csv.zip"])
        if "test.csv" in fl: test_path = os.path.join(root, fl["test.csv"])
        elif "test.csv.zip" in fl: test_path = os.path.join(root, fl["test.csv.zip"])
        if "sample_submission.csv" in fl: sample_path = os.path.join(root, fl["sample_submission.csv"])
        if train_path and test_path and sample_path:
            break
    if not (train_path and test_path and sample_path):
        info("Could not find all files. Attach the competition data via 'Add data'.")
        raise FileNotFoundError("Need train.csv(.zip), test.csv(.zip), sample_submission.csv")
    return train_path, test_path, sample_path

TRAIN_PATH, TEST_PATH, SAMPLE_PATH = find_files()
info(f"Found:\n  TRAIN:  {TRAIN_PATH}\n  TEST:   {TEST_PATH}\n  SAMPLE: {SAMPLE_PATH}")

# ----------------------------
# Read ~300k train rows (chunked) and test
# ----------------------------
usecols = ["key","fare_amount","pickup_datetime",
           "pickup_longitude","pickup_latitude",
           "dropoff_longitude","dropoff_latitude",
           "passenger_count"]

def read_train_sample(train_path, n_target=TARGET_ROWS, chunk_size=CHUNK_SIZE, seed=42):
    rng = np.random.default_rng(seed)
    samples, total = [], 0
    if train_path.lower().endswith(".zip"):
        with zipfile.ZipFile(train_path) as zf:
            csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            with zf.open(csvs[0]) as f:
                for chunk in pd.read_csv(f, usecols=usecols, chunksize=chunk_size):
                    if total >= n_target: break
                    frac = min(1.0, (n_target-total)/len(chunk))
                    samp = chunk.sample(frac=frac, random_state=rng.integers(1e9))
                    samples.append(samp); total += len(samp)
    else:
        for chunk in pd.read_csv(train_path, usecols=usecols, chunksize=chunk_size):
            if total >= n_target: break
            frac = min(1.0, (n_target-total)/len(chunk))
            samp = chunk.sample(frac=frac, random_state=seed+total)
            samples.append(samp); total += len(samp)
    return pd.concat(samples, ignore_index=True)

info(f"Sampling ~{TARGET_ROWS:,} rows from train...")
train = read_train_sample(TRAIN_PATH, TARGET_ROWS, CHUNK_SIZE)
info(f"Train sample shape: {train.shape}")

def read_test(test_path):
    cols = ["key","pickup_datetime","pickup_longitude","pickup_latitude",
            "dropoff_longitude","dropoff_latitude","passenger_count"]
    if test_path.lower().endswith(".zip"):
        with zipfile.ZipFile(test_path) as zf:
            csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            with zf.open(csvs[0]) as f:
                return pd.read_csv(f, usecols=cols)
    else:
        return pd.read_csv(test_path, usecols=cols)

test = read_test(TEST_PATH)
info(f"Test shape: {test.shape}")

# ----------------------------
# Light cleaning / filters
# ----------------------------
train = train[(train["fare_amount"] >= 0) & (train["fare_amount"] <= 500)]
train = train[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 8)]
lat_min, lat_max = 40.5, 41.0
lon_min, lon_max = -74.3, -72.9
for c in ["pickup_latitude","dropoff_latitude"]:
    train = train[(train[c] >= lat_min) & (train[c] <= lat_max)]
for c in ["pickup_longitude","dropoff_longitude"]:
    train = train[(train[c] >= lon_min) & (train[c] <= lon_max)]
info(f"After filters: {train.shape}")

# ----------------------------
# Feature engineering
# ----------------------------
def add_features(df):
    df = df.copy()
    dt = pd.to_datetime(df["pickup_datetime"], utc=True, errors="coerce")
    df["pickup_year"]  = dt.dt.year
    df["pickup_month"] = dt.dt.month
    df["pickup_day"]   = dt.dt.day
    df["pickup_hour"]  = dt.dt.hour
    df["pickup_dow"]   = dt.dt.dayofweek
    df["is_weekend"]   = (df["pickup_dow"] >= 5).astype(int)
    df["rush_hour"]    = ((df["pickup_hour"].between(7,9)) | (df["pickup_hour"].between(16,19))).astype(int)
    df["haversine_km"] = haversine_vec(df["pickup_latitude"], df["pickup_longitude"],
                                       df["dropoff_latitude"], df["dropoff_longitude"])
    df["manhattan_km"] = manhattan_distance_approx(df["pickup_latitude"], df["pickup_longitude"],
                                                   df["dropoff_latitude"], df["dropoff_longitude"])
    df["bearing"]      = bearing_vec(df["pickup_latitude"], df["pickup_longitude"],
                                     df["dropoff_latitude"], df["dropoff_longitude"])
    df = df.drop(columns=["pickup_datetime"])
    return df

train_fe = add_features(train)
test_fe  = add_features(test)

# Replace any +/-inf with NaN before plotting/learning (prevents seaborn warnings)
for df_ in (train_fe, test_fe, train, test):
    df_.replace([np.inf, -np.inf], np.nan, inplace=True)

y = train["fare_amount"].values
X = train_fe.drop(columns=["fare_amount","key"], errors="ignore")
X_test = test_fe.drop(columns=["key"], errors="ignore")

num_cols = [c for c in X.columns if c not in ["pickup_year","pickup_month","pickup_dow","pickup_hour","is_weekend","rush_hour"]]
cat_cols = [c for c in ["pickup_year","pickup_month","pickup_dow","pickup_hour","is_weekend","rush_hour"] if c in X.columns]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
    ],
    remainder="drop"
)

# Log-target Ridge (fast, solid)
class LogTargetWrapper:
    def __init__(self, base_model): self.base = base_model
    def fit(self, X, y): self.base.fit(X, np.log1p(np.maximum(y,0))); return self
    def predict(self, X): return np.clip(np.expm1(self.base.predict(X)), 0, None)
    def get_params(self, deep=True): return {"base_model": self.base}
    def set_params(self, **p):
        if "base_model" in p: self.base = p.pop("base_model")
        for k,v in p.items(): setattr(self.base,k,v); return self

ridge = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", LogTargetWrapper(Ridge(alpha=2.0, random_state=42)))
])

# ----------------------------
# EDA (quick graphs)
# ----------------------------
# 1) Fare distribution
plt.figure(figsize=(6,4))
if HAS_SNS:
    sns.histplot(train["fare_amount"], bins=60, kde=False)
else:
    plt.hist(train["fare_amount"], bins=60)
plt.title("Fare Amount Distribution (0–100)"); plt.xlim(0,100)
plt.xlabel("fare_amount"); plt.ylabel("freq"); plt.tight_layout(); plt.show()

# 2) Fare vs Haversine (sample)
N = min(120_000, len(train))
plt.figure(figsize=(6,4))
plt.scatter(train_fe["haversine_km"][:N], train["fare_amount"][:N], s=2, alpha=0.2)
plt.title("Fare vs Haversine Distance (sample)")
plt.xlabel("haversine_km"); plt.ylabel("fare_amount"); plt.tight_layout(); plt.show()

# 3) Mean fare by hour
hour_mean = pd.concat([train_fe["pickup_hour"], train["fare_amount"]], axis=1).groupby("pickup_hour")["fare_amount"].mean()
plt.figure(figsize=(6,4))
if HAS_SNS:
    sns.barplot(x=hour_mean.index, y=hour_mean.values)
else:
    plt.bar(hour_mean.index, hour_mean.values)
plt.title("Mean Fare by Pickup Hour"); plt.xlabel("hour"); plt.ylabel("mean fare")
plt.tight_layout(); plt.show()

# ----------------------------
# Train & CV (Ridge, 3-fold)
# ----------------------------
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
scores = cross_val_score(ridge, X, y, cv=kf, scoring=rmsle_scorer, n_jobs=-1)
print(f"Ridge RMSLE ({N_FOLDS}-fold): {-scores.mean():.5f} (± {scores.std():.5f})")

# Fit on all & predict test
ridge.fit(X, y)
test_pred = ridge.predict(X_test)
test_pred = np.clip(test_pred, 0, 5000)

# Submission
sample = pd.read_csv(SAMPLE_PATH)
assert {"key","fare_amount"} <= set(sample.columns)
sub = sample.copy()
sub["fare_amount"] = test_pred
sub.to_csv("submission.csv", index=False)
print("\n✅ Saved submission.csv")
print(sub.head())


