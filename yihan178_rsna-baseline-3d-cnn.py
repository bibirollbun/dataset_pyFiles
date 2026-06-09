from glob import glob
import os, json, random
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt

from ipywidgets import interact, widgets, fixed
from pathlib import Path
import os, random, numpy as np, pandas as pd, pydicom, cv2, gc



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


# Basic counts
print("Total samples:", len(train_df))
print("Aneurysm Present positive rate:", round(train_df['Aneurysm Present'].mean()*100, 2), "%")
print("\nModality counts:")
print(train_df['Modality'].value_counts())

# Age distribution
ages = pd.to_numeric(train_df['PatientAge'], errors='coerce').dropna()
plt.figure(figsize=(6,4))
plt.hist(ages, bins=20)
plt.xlabel('Age'); plt.ylabel('Count'); plt.title('Patient Age Distribution')
plt.show()

# Aneurysm presence (0/1)
pos = train_df['Aneurysm Present'].value_counts().sort_index()
plt.figure(figsize=(5,4))
plt.bar(['No (0)', 'Yes (1)'], pos.values)
plt.title('Aneurysm Present Count'); plt.show()


# --- Minimal slider viewer: one dropdown + one slider ---
from ipywidgets import Dropdown, IntSlider, VBox
from IPython.display import display, clear_output
import functools, pydicom, matplotlib.pyplot as plt

def list_series_uids(limit=50):
    uids = [uid for uid in train_df['SeriesInstanceUID'] if (SERIES_DIR/str(uid)).exists()]
    uids = sorted(uids)[:limit]    # remove shuffle, keep deterministic order
    return uids

# Read a series and cache the result to avoid repeated disk reads
@functools.lru_cache(maxsize=256)
def read_series_stack(series_uid: str):
    series_path = SERIES_DIR / series_uid
    files = list(series_path.glob("*.dcm"))

    # Sort by InstanceNumber (good enough for classroom viewing)
    def inst_no(p):
        try:
            d = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            return int(getattr(d, 'InstanceNumber', 0))
        except:
            return 0
    files = sorted(files, key=inst_no)

    # Load pixel arrays into a list of 2D slices
    imgs = []
    for p in files:
        try:
            d = pydicom.dcmread(str(p), force=True)
            imgs.append(d.pixel_array)
        except:
            pass
    return imgs  # list of 2D numpy arrays

# UI controls: dropdown for series, slider for slice index
uids = list_series_uids(60)
uid_dd = Dropdown(options=uids, description='Series')
sl     = IntSlider(description='Slice', min=1, max=1, step=1, value=1)

out = widgets.Output()

def refresh_plot(*_):
    """Redraw the selected slice for the selected series."""
    with out:
        out.clear_output(wait=True)
        imgs = read_series_stack(str(uid_dd.value))
        if not imgs:
            print("No readable slices for this series.")
            return

        # Keep slider range in sync with number of slices
        sl.max = max(1, len(imgs))
        idx = min(sl.value - 1, len(imgs) - 1)  # convert to 0-based index

        # Display the current slice
        plt.figure(figsize=(5,5))
        plt.imshow(imgs[idx], cmap='gray')
        plt.title(f'{uid_dd.value}\nSlice {idx+1}/{len(imgs)}')
        plt.axis('off')
        plt.show()

# Redraw when series or slice changes
uid_dd.observe(lambda ch: refresh_plot(), names='value')
sl.observe(lambda ch: refresh_plot(), names='value')

# Initial draw
refresh_plot()
display(VBox([uid_dd, sl, out]))



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



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

def make_ds(df_part, shuffle):
    """Build a tf.data dataset that yields (volume, label)."""
    sids = df_part["SeriesInstanceUID"].astype(str).values
    ys   = df_part["Aneurysm Present"].astype(np.float32).values

    def py_load(sid_bytes):
        sid = sid_bytes.decode()
        x = load_series(sid)                 # (D,H,W,1) float32 in [0,1]
        return x.astype(np.float32)

    def wrapper(sid, y):
        # Bridge Python/Numpy to TF tensors (shape is known)
        vol = tf.numpy_function(py_load, [sid], tf.float32)
        vol.set_shape((DEPTH,HEIGHT,WIDTH,1))
        return vol, y

    ds = tf.data.Dataset.from_tensor_slices((sids, ys))
    if shuffle:
        ds = ds.shuffle(min(len(df_part),256), seed=SEED, reshuffle_each_iteration=True)

    # Map (parallel), batch, prefetch; deterministic=False allows better throughput
    ds = ds.map(wrapper, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)\
           .batch(BATCH_SIZE)\
           .prefetch(tf.data.AUTOTUNE)
    return ds

# Final train/val datasets
train_ds = make_ds(df_tr, shuffle=True)
val_ds   = make_ds(df_val, shuffle=False)


import math

n_train = len(df_tr)
n_val   = len(df_val)
steps_train = math.ceil(n_train / BATCH_SIZE)
steps_val   = math.ceil(n_val / BATCH_SIZE)

print(f"Train series: {n_train}  |  steps/epoch ≈ {steps_train}")
print(f"Val   series: {n_val}    |  val_steps   ≈ {steps_val}")



# Grab one training batch
train_batch = next(iter(train_ds))
x_tr, y_tr = train_batch  # x: (B, D, H, W, 1), y: (B,)
print("Train batch X shape:", x_tr.shape, "dtype:", x_tr.dtype)
print("Train batch y shape:", y_tr.shape, "dtype:", y_tr.dtype)
print("First few train labels:", y_tr.numpy()[:8])

# Grab one validation batch
val_batch = next(iter(val_ds))
x_va, y_va = val_batch
print("Val batch X shape:", x_va.shape, "dtype:", x_va.dtype)
print("Val batch y shape:", y_va.shape, "dtype:", y_va.dtype)
print("First few val labels:", y_va.numpy()[:8])


import matplotlib.pyplot as plt
import numpy as np

# pick the first example of the training batch
vol = x_tr[0].numpy()  # (D,H,W,1) in [0,1]  ## try to change to 1, 2, ... observe some thing?
vol = np.squeeze(vol, axis=-1)  # (D,H,W)
D,H,W = vol.shape
mid = D // 2

# 1) show the middle slice
plt.figure(figsize=(4,4))
plt.imshow(vol[mid], cmap='gray')
plt.title(f"One training example\nmiddle slice {mid+1}/{D}")
plt.axis('off')
plt.show()

# 2) show a small montage (e.g., 12 evenly spaced slices)
k = 12
idxs = np.linspace(0, D-1, k).astype(int)
cols = 6
rows = int(np.ceil(k/cols))
plt.figure(figsize=(10, 3.5))
for i, z in enumerate(idxs):
    ax = plt.subplot(rows, cols, i+1)
    ax.imshow(vol[z], cmap='gray')
    ax.set_title(f"z={z+1}", fontsize=8)
    ax.axis('off')
plt.suptitle("Evenly spaced slices (overview)", y=1.02)
plt.tight_layout()
plt.show()



def build_tiny_3dcnn():
    inp = keras.Input((DEPTH,HEIGHT,WIDTH,1))
    x = layers.Conv3D(16,3,padding="same",activation="relu")(inp)  # 16 filters, 3 by 3 by 3
    x = layers.MaxPool3D()(x)  # by default, 2 by 2 by 2.
    x = layers.Conv3D(32,3,padding="same",activation="relu")(x) # 32 filters, 3 by 3 by 3
    x = layers.GlobalAveragePooling3D()(x)
    
    x = layers.Dropout(0.2)(x) # set the rate to 0.2
    
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x) 

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],          
        steps_per_execution=32       
    )
    return model

model = build_tiny_3dcnn()

model.summary()



# Don't run this cell
# callbacks = [
#     keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=1, factor=0.5, verbose=1),
#     keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=2, restore_best_weights=True),
# ]
 
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=1,
#     callbacks=callbacks,
#     verbose=1,
# )


# Don't run this cell
# val_metrics = model.evaluate(val_ds, return_dict=True, verbose=0)
# print(val_metrics)


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



train_ds_nocache = make_ds(df_tr, shuffle=True)
val_ds_nocache   = make_ds(df_val, shuffle=False)

train_ds_cache = make_ds_cache(df_tr, shuffle=True)
val_ds_cache   = make_ds_cache(df_val, shuffle=False)



import time

# First, run the version without caching
model = build_tiny_3dcnn()
t0 = time.perf_counter()
history = model.fit(
    train_ds_nocache,
    validation_data=val_ds_nocache,
    epochs=2,
    verbose=1,
)
t1 = time.perf_counter()
print(f"[NoCache] Total time: {t1 - t0:.1f}s")

# Now, run the cached version (use a new model for a fair comparison)
model2 = build_tiny_3dcnn()

# Optional: touch the datasets once to fill the cache
for _ in train_ds_cache.take(1): pass
for _ in val_ds_cache.take(1):   pass

t2 = time.perf_counter()
history2 = model2.fit(
    train_ds_cache,
    validation_data=val_ds_cache,
    epochs=4,
    verbose=1,
)
t3 = time.perf_counter()
print(f"[Cache]   Total time: {t3 - t2:.1f}s")



# def bigger_3dcnn():
#     inp = keras.Input((DEPTH, HEIGHT, WIDTH, 1))
#     x = layers.Conv3D(16, 3, padding="same", activation="relu")(inp)
#     x = layers.MaxPool3D()(x)
#     x = layers.Conv3D(32, 3, padding="same", activation="relu")(x)
#     ... # Another max pooling layer

#     ... # Add flatten layer
#     ... # Add a dense layer with 128 neurons.
#     x = layers.Dropout(...)(x) # increase the dropout rate to 0.4

#     out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

#     model = keras.Model(inp, out)
#     model.compile(
#         optimizer=keras.optimizers.Adam(1e-3),
#         loss="binary_crossentropy",
#         metrics=["accuracy"]
#     )
#     return model


# def bigger_3dcnn():
#     inp = keras.Input((DEPTH, HEIGHT, WIDTH, 1))
#     x = layers.Conv3D(16, 3, padding="same", activation="relu")(inp)
#     x = layers.MaxPool3D()(x)
#     x = layers.Conv3D(32, 3, padding="same", activation="relu")(x)
#     x = layers.MaxPool3D()(x)

#     x = layers.Flatten()(x)
#     x = layers.Dense(128, activation="relu")(x)
#     x = layers.Dropout(0.4)(x)

#     out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

#     model = keras.Model(inp, out)
#     model.compile(
#         optimizer=keras.optimizers.Adam(1e-3),
#         loss="binary_crossentropy",
#         metrics=["accuracy"]
#     )
#     return model


# train_ds_cache = make_ds_cache(df_tr, shuffle=True)
# val_ds_cache   = make_ds_cache(df_val, shuffle=False)


# # Now, run the cached version (use a new model for a fair comparison)
# model3 = bigger_3dcnn()

# # Optional: touch the datasets once to fill the cache
# for _ in train_ds_cache.take(1): pass
# for _ in val_ds_cache.take(1):   pass

# t4 = time.perf_counter()
# history3 = model3.fit(
#     train_ds_cache,
#     validation_data=val_ds_cache,
#     epochs=4,
#     verbose=1,
# )
# t5 = time.perf_counter()
# print(f"[Cache]   Total time: {t3 - t2:.1f}s")


# val_metrics3 = model3.evaluate(val_ds_cache , return_dict=True, verbose=0)
# print(val_metrics3)

