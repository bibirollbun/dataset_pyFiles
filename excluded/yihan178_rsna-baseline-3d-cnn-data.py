from glob import glob
import os, json, random
import numpy as np
import pandas as pd
import polars as pl
import pandas as pd
import shutil
import pydicom
import matplotlib.pyplot as plt

from ipywidgets import interact, widgets, fixed
from pathlib import Path
import os, random, numpy as np, pandas as pd, pydicom, cv2, gc

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import kaggle_evaluation.rsna_inference_server


# Kaggle dataset root (mounted read-only in Kaggle notebooks)
BASE = Path("/kaggle/input/rsna-intracranial-aneurysm-detection")

# Where each DICOM series lives: series/<SeriesInstanceUID>/*.dcm
SERIES_DIR = BASE / "series" 

# CSV files provided by the competition
TRAIN_CSV  = BASE / "train.csv"             # main labels per SeriesInstanceUID
# LOCAL_CSV  = BASE / "train_localizers.csv"  # per-slice point annotations

# Load the main training table
train_df = pd.read_csv(TRAIN_CSV)

# Quick peek (optional in class)
print("Rows in train.csv:", len(train_df))
train_df.head()


# Small fixed voxel size for memory & speed (D, H, W)
DEPTH, HEIGHT, WIDTH = 48, 96, 96
BATCH_SIZE = 4
EPOCHS = 2
N_SERIES = 60
SEED = 1337
random.seed(SEED); np.random.seed(SEED)

df = pd.read_csv(TRAIN_CSV)[["SeriesInstanceUID","Aneurysm Present"]].dropna()
df["Aneurysm Present"] = df["Aneurysm Present"].astype(int)
df = df[df["SeriesInstanceUID"].apply(lambda s: (SERIES_DIR/s).exists())].reset_index(drop=True)

# Try to balance classes within the small sample
pos = df[df["Aneurysm Present"]==1].sample(min(N_SERIES//2, df["Aneurysm Present"].sum()), 
                                           random_state=SEED)
neg = df[df["Aneurysm Present"]==0].sample(N_SERIES-len(pos), random_state=SEED)
df_small = pd.concat([pos,neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)

# Simple split 
val_n = max(8, int(0.2*len(df_small)))
df_val = df_small.iloc[:val_n].reset_index(drop=True)
df_tr  = df_small.iloc[val_n:].reset_index(drop=True)

len(df_tr), len(df_val)  # helps sanity-check: e.g., 48 train / 12 val


def _sort_key(d):
    """Sort slices. Prefer z-position; fall back to InstanceNumber."""
    z = None
    if hasattr(d, "ImagePositionPatient") and len(d.ImagePositionPatient)>=3:
        try:
            z = float(d.ImagePositionPatient[2])
        except:
            pass
    if z is not None:
        return (0, z)
    return (1, float(getattr(d, "InstanceNumber", 0)))

def load_series(series_id, target_shape=(DEPTH,HEIGHT,WIDTH), clip=(1,99)):
    """Read a series and turn it into a fixed-size, normalized 3D volume (D,H,W,1)."""
    sdir = SERIES_DIR/series_id
    files = [p for p in sdir.glob("*.dcm")]
    if not files:
        return np.zeros((*target_shape,1), np.float32)

    # Read headers, ensure pixels exist
    headers = []
    for fp in files:
        try:
            d = pydicom.dcmread(str(fp), force=True, stop_before_pixels=False)
            if hasattr(d, "PixelData"):
                headers.append(d)
        except:
            pass
    if not headers:
        return np.zeros((*target_shape,1), np.float32)

    # Sort slices (z if available, otherwise instance number)
    headers.sort(key=_sort_key)

    # Extract pixels; handle multi-frame DICOMs too
    slices = []
    for d in headers:
        try:
            arr = d.pixel_array.astype(np.float32)
            slope = float(getattr(d,"RescaleSlope",1.0))
            inter = float(getattr(d,"RescaleIntercept",0.0))
            arr = arr*slope + inter
            if arr.ndim==2:
                slices.append(arr)
            elif arr.ndim==3:
                # handle (F,H,W) or (H,W,F)
                if arr.shape[0] < 8 and arr.shape[-1] > arr.shape[0]:
                    arr = np.moveaxis(arr, -1, 0)
                for k in range(arr.shape[0]):
                    slices.append(arr[k])
        except:
            continue
    if not slices:
        return np.zeros((*target_shape,1), np.float32)

    vol = np.stack(slices,0)  # (D0,H0,W0)

    # Robust intensity clamp (1–99th percentile), then normalize to [0,1]
    lo, hi = np.percentile(vol, clip)
    vol = np.clip(vol, lo, hi)
    D0,H0,W0 = vol.shape

    # Resize each slice to target H×W
    resized = np.zeros((D0, target_shape[1], target_shape[2]), np.float32)
    for i in range(D0):
        resized[i] = cv2.resize(vol[i], (target_shape[2], target_shape[1]), interpolation=cv2.INTER_LINEAR)

    # Depth-uniform sampling to target D
    idx = np.linspace(0, D0-1, target_shape[0]).astype(int)
    vol = resized[idx]

    # Normalize to [0,1]
    mn, mx = vol.min(), vol.max()
    if mx>mn:
        vol = (vol-mn)/(mx-mn)

    return vol[...,None].astype(np.float32)  # (D,H,W,1)



# Make GPU memory growth "on demand" (avoid grabbing all VRAM at once)
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass
# print("GPUs visible to TF:", gpus)

# Optional mixed precision for speed (safe because output Dense is float32)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

def make_ds_cache(df_part, shuffle):
    """With memory caching: first epoch is slow, later epochs much faster"""
    sids = df_part["SeriesInstanceUID"].astype(str).values
    ys   = df_part["Aneurysm Present"].astype(np.float32).values

    def py_load(sid_bytes):
        sid = sid_bytes.decode()
        x = load_series(sid)  # (D,H,W,1) float32 in [0,1]
        return x.astype(np.float32)

    def wrapper(sid, y):
        vol = tf.numpy_function(py_load, [sid], tf.float32)
        vol.set_shape((DEPTH,HEIGHT,WIDTH,1))
        return vol, y

    ds = tf.data.Dataset.from_tensor_slices((sids, ys))
    if shuffle:
        ds = ds.shuffle(min(len(df_part),256), seed=SEED, reshuffle_each_iteration=True)

    # key: map -> cache -> batch -> prefetch
    ds = ds.map(wrapper, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    ds = ds.cache()   # store mapped volumes in memory; reuse instead of reloading
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# Final train/val datasets
train_ds_cache = make_ds_cache(df_tr, shuffle=True)
val_ds_cache   = make_ds_cache(df_val, shuffle=False)


def bigger_3dcnn():
    inp = keras.Input((DEPTH, HEIGHT, WIDTH, 1))
    x = layers.Conv3D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPool3D()(x)
    x = layers.Conv3D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool3D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = bigger_3dcnn()

model.summary()


# Optional: touch the datasets once to fill the cache
for _ in train_ds_cache.take(1): pass
for _ in val_ds_cache.take(1):   pass

history = model.fit(
    train_ds_cache,
    validation_data=val_ds_cache,
    epochs=4,
    verbose=1,
)


val_metrics = model.evaluate(val_ds_cache , return_dict=True, verbose=0)
print(val_metrics)


# Save the trained model
MODEL_PATH = "/kaggle/working/tiny3dcnn_model.keras"
model.save(MODEL_PATH)
print("Model saved to:", MODEL_PATH)


# Load the saved model only once (outside predict)
MODEL_PATH = "/kaggle/working/tiny3dcnn_model.keras"
inference_model = keras.models.load_model(MODEL_PATH, compile=False)


# Define label columns and constants (same as demo submission)

ID_COL = "SeriesInstanceUID"

LABEL_COLS = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
    "Aneurysm Present",
]


# Load the saved model only once (outside predict)

def predict(series_path: str) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction for one series folder."""

    # Get the series ID (the folder name)
    series_id = os.path.basename(series_path)

    # ---- load and preprocess the DICOM series ----
    # Use the same loader we defined before
    vol = load_series(series_id)          # (D,H,W,1)
    vol = np.expand_dims(vol, 0)          # (1,D,H,W,1) -> batch dimension

    # ---- model prediction ----
    pred_prob = inference_model.predict(vol, verbose=0)[0][0]  # single float between 0-1

    # ---- build final prediction table ----
    # 13 other columns = 0.5, last one from model
    row = [series_id] + [0.5] * 13 + [float(pred_prob)]
    predictions = pl.DataFrame(
        data=[row],
        schema=[ID_COL, *LABEL_COLS],
        orient="row",
    )

    # ---- sanity checks (keep these) ----
    if isinstance(predictions, pl.DataFrame):
        assert predictions.columns == [ID_COL, *LABEL_COLS]
    elif isinstance(predictions, pd.DataFrame):
        assert (predictions.columns == [ID_COL, *LABEL_COLS]).all()
    else:
        raise TypeError("The predict function must return a DataFrame")

    # ---- cleanup to avoid disk overflow ----
    shutil.rmtree("/kaggle/shared", ignore_errors=True)

    # Kaggle expects the ID column to be dropped before returning
    return predictions.drop(ID_COL)


inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))

