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


# Check what folders exist under /kaggle/input
!ls /kaggle/input

# Extract using 7-Zip from the correct competition folder
!mkdir -p ./data

print("Extracting train.zip ...")
!7z x /kaggle/input/nyc-taxi-trip-duration/train.zip -o./data -y

print("Extracting test.zip ...")
!7z x /kaggle/input/nyc-taxi-trip-duration/test.zip -o./data -y

print("Extracting sample_submission.zip ...")
!7z x /kaggle/input/nyc-taxi-trip-duration/sample_submission.zip -o./data -y

print("Files in ./data:")
!ls -lh ./data



import os
import gc
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============== CONFIG ===============

RANDOM_SEED = 42
N_FOLDS = 5
BATCH_SIZE = 4096
N_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.25
N_CLUSTERS = 20  # for KMeans on locations

DATA_DIR = Path("./data")

# --- Optional weather features ---
USE_WEATHER = True  # set to False if you don't want weather
# TODO: change this path to match your attached weather dataset
WEATHER_PATH = Path("/kaggle/input/nyc-taxi-weather-data/weather_data.csv")

# =============== SEEDING ===============

def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =============== WEATHER LOADER ===============

def load_weather():
    if not USE_WEATHER:
        print("Weather features disabled (USE_WEATHER = False).")
        return None

    if not WEATHER_PATH.exists():
        print(f"Weather file not found at {WEATHER_PATH}. Continuing without weather features.")
        return None

    w = pd.read_csv(WEATHER_PATH)
    if 'date' not in w.columns:
        print("Weather file found, but no 'date' column. "
              "Please adjust WEATHER_PATH or column name. Continuing without weather.")
        return None

    # Convert to Python date objects for easy merge
    w['date'] = pd.to_datetime(w['date']).dt.date

    # Keep only numeric + date
    keep_cols = ['date'] + [c for c in w.columns if c != 'date']
    w = w[keep_cols]

    print("Weather data loaded with columns:", list(w.columns))
    return w

weather_df = load_weather()

# =============== FEATURE ENGINEERING ===============

def haversine_array(lat1, lng1, lat2, lng2):
    """
    Vectorized haversine distance (km) between two points.
    """
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["pickup_year"] = df["pickup_datetime"].dt.year
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_day"] = df["pickup_datetime"].dt.day
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_minute"] = df["pickup_datetime"].dt.minute
    df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df["pickup_weekofyear"] = df["pickup_datetime"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["pickup_dayofweek"] >= 5).astype(int)

    # keep a pure date column for merging weather
    df["pickup_date"] = df["pickup_datetime"].dt.date

    # simple US 2016 federal holidays list
    holidays_2016 = [
        "2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30",
        "2016-07-04", "2016-09-05", "2016-10-10", "2016-11-11",
        "2016-11-24", "2016-12-26"
    ]
    df["is_holiday"] = df["pickup_date"].astype(str).isin(holidays_2016).astype(int)

    return df


def add_geo_features(df):
    lat1 = df["pickup_latitude"].values
    lng1 = df["pickup_longitude"].values
    lat2 = df["dropoff_latitude"].values
    lng2 = df["dropoff_longitude"].values

    df["haversine_km"] = haversine_array(lat1, lng1, lat2, lng2)
    df["manhattan_km"] = (np.abs(lat1 - lat2) + np.abs(lng1 - lng2)) * 111  # rough
    df["delta_lat"] = lat2 - lat1
    df["delta_lng"] = lng2 - lng1
    df["bearing"] = np.degrees(np.arctan2(df["delta_lng"], df["delta_lat"] + 1e-6))
    return df


def add_basic_flags(df):
    # store_and_fwd_flag: Y/N -> 1/0
    df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)
    return df


def cluster_locations(full_df):
    """
    Fit KMeans on all pickup/dropoff points (train+test) and
    add cluster labels as features.
    """
    coord_cols = ["pickup_latitude", "pickup_longitude",
                  "dropoff_latitude", "dropoff_longitude"]

    coords = full_df[coord_cols].values
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    full_df["loc_cluster"] = kmeans.fit_predict(coords)
    return full_df


def engineer_features(train_df, test_df, weather_df=None):
    # Combine for consistent processing
    train_df["is_train"] = 1
    test_df["is_train"] = 0
    full = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    full = add_time_features(full)
    full = add_geo_features(full)
    full = add_basic_flags(full)

    # Merge weather on pickup_date if available
    if weather_df is not None:
        full = full.merge(
            weather_df,
            how="left",
            left_on="pickup_date",
            right_on="date"
        )
        # drop the weather 'date' column after merge
        if "date" in full.columns:
            full = full.drop(columns=["date"])

    # Cluster-based feature
    full = cluster_locations(full)

    # Drop raw datetime & helper date (non-numeric)
    drop_cols = []
    if "pickup_datetime" in full.columns:
        drop_cols.append("pickup_datetime")
    if "pickup_date" in full.columns:
        drop_cols.append("pickup_date")
    full = full.drop(columns=drop_cols)

    # Split back
    train_proc = full[full["is_train"] == 1].copy()
    test_proc = full[full["is_train"] == 0].copy()

    train_proc = train_proc.drop(columns=["is_train"])
    test_proc = test_proc.drop(columns=["is_train"])

    return train_proc, test_proc

# =============== LOAD DATA ===============

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

print("Train shape (raw):", train.shape)
print("Test shape (raw):", test.shape)

# Target: use log1p to stabilize
train["log_trip_duration"] = np.log1p(train["trip_duration"].clip(lower=1))

# Feature engineering
train_fe, test_fe = engineer_features(train, test, weather_df)

# ---- Make sure we only use numeric features ----

TARGET_COL = "log_trip_duration"

# Numeric columns only (this automatically drops any remaining non-numerics)
numeric_cols = train_fe.select_dtypes(include=[np.number]).columns.tolist()

# Features = numeric columns except target and raw trip_duration
feature_cols = [
    c for c in numeric_cols
    if c not in ["trip_duration", TARGET_COL]
]

print("Feature columns:", feature_cols)
print("Number of features:", len(feature_cols))

# Build matrices
X = train_fe[feature_cols].values
y = train_fe[TARGET_COL].values.astype("float32")
X_test = test_fe[feature_cols].values
test_ids = test_fe["id"].values  # id is numeric, we still read it from test_fe

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


# =============== DATASET / DATALOADER ===============

class TaxiDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X.astype("float32")
        self.y = y.astype("float32") if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.y is not None:
            return x, self.y[idx]
        else:
            return x


# =============== MODEL ===============

class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT * 0.5),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def rmse_loss(pred, target):
    return torch.sqrt(nn.functional.mse_loss(pred, target))

# =============== TRAINING / EVAL ===============

def train_one_fold(X_train, y_train, X_val, y_val, fold_idx):
    print(f"\n========== Fold {fold_idx + 1} ==========")

    train_ds = TaxiDataset(X_train, y_train)
    val_ds = TaxiDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MLPRegressor(input_dim=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, N_EPOCHS + 1):
        # ----- train -----
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb)
            loss = rmse_loss(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ----- validate -----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                preds = model(xb)
                loss = rmse_loss(preds, yb)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        scheduler.step(avg_val)

        print(
            f"Epoch {epoch:02d} | "
            f"train RMSE: {avg_train:.4f} | "
            f"val RMSE: {avg_val:.4f}"
        )

        # Early stopping
        if avg_val < best_val_loss - 1e-4:
            best_val_loss = avg_val
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss


def predict_model(model, X):
    ds = TaxiDataset(X, None)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(DEVICE)
            pb = model(xb).cpu().numpy()
            preds.append(pb)
    preds = np.concatenate(preds, axis=0)
    return preds


# =============== K-FOLD TRAINING ===============

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

oof_preds = np.zeros(len(X), dtype="float32")
test_preds_folds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model, best_val = train_one_fold(X_tr, y_tr, X_val, y_val, fold)
    print(f"Best val RMSE for fold {fold+1}: {best_val:.4f}")

    # OOF predictions
    oof_preds[val_idx] = predict_model(model, X_val)

    # Test predictions
    test_pred = predict_model(model, X_test)
    test_preds_folds.append(test_pred)

    # Free GPU memory between folds
    del model
    torch.cuda.empty_cache()
    gc.collect()

# Overall OOF score
oof_rmse = math.sqrt(np.mean((oof_preds - y) ** 2))
print(f"\nOOF RMSE (log duration): {oof_rmse:.4f}")

# =============== CREATE SUBMISSION ===============

# Average test predictions over folds
test_preds_mean = np.mean(test_preds_folds, axis=0)

# Convert from log1p back to seconds
test_trip_duration = np.expm1(test_preds_mean)
test_trip_duration = np.clip(test_trip_duration, 1, None)

submission = pd.DataFrame({
    "id": test_ids,
    "trip_duration": test_trip_duration
})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Saved submission to:", submission_path)



import os
import pandas as pd

# Path where we saved the submission
sub_path = "submission.csv"

# Check it exists
print("File exists:", os.path.isfile(sub_path))

# Show basic info
df_sub = pd.read_csv(sub_path)
print("Shape:", df_sub.shape)
print(df_sub.head())

# Confirm it will be saved in the working directory
!ls -lh


