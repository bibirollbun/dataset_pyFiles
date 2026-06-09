import pandas as pd
# Load the CSV files
train_df = pd.read_csv("/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/train.csv")
test_df = pd.read_csv("/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/test.csv")



# Display dataset info
print("Train shape :", train_df.shape)
print("Test shape : ",test_df.shape)



train_df.head()


train_df['label'].hist(bins=100)


import numpy as np
import matplotlib.pyplot as plt
import os

npy_dir = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/ot/ot"

sample_id = train_df['id'].iloc[0]
sample_path = os.path.join(npy_dir,sample_id)

data = np.load(sample_path)
print("Shape of the sample is : ",data.shape)


# View a few bands
plt.figure(figsize=(12, 4))
for i, band in enumerate([0, 10, 30, 50, 70, 90]):
    plt.subplot(1, 6, i+1)
    plt.imshow(data[:, :, band], cmap='viridis')
    plt.title(f'Band {band}')
    plt.axis('off')
plt.tight_layout()
plt.show()


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# --- Configuration ---
BASE_DIR    = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025"
NPY_DIR     = os.path.join(BASE_DIR, "ot/ot")
TRAIN_CSV   = os.path.join(BASE_DIR, "train.csv")
TEST_CSV    = os.path.join(BASE_DIR, "test.csv")
SUBMISSION  = "submission_xgb2.csv"

IMG_SHAPE   = (128, 128, 125)
TARGET_SIZE = np.prod(IMG_SHAPE)


def load_and_flatten(path):
    """
    Load a .npy file (or raw .npy if header corrupt), flatten to 1D,
    pad by repeating last value if too short, or truncate if too long.
    """
    try:
        arr = np.load(path)
        flat = arr.ravel()
    except Exception:
        flat = np.fromfile(path, dtype=np.float32)

    # pad/truncate to TARGET_SIZE
    if flat.size < TARGET_SIZE:
        if flat.size == 0:
            flat = np.zeros(TARGET_SIZE, dtype=np.float32)
        else:
            pad_vals = np.full(TARGET_SIZE - flat.size, flat[-1], dtype=np.float32)
            flat = np.concatenate([flat, pad_vals])
    else:
        flat = flat[:TARGET_SIZE]

    return flat


def extract_features(df):
    """
    For each row in df (with 'id'), load the patch, fix shape, and compute
    mean reflectance for each of the 125 bands.
    Returns an (n_samples, 125) array.
    """
    features = []
    for fn in df['id']:
        path = os.path.join(NPY_DIR, fn)
        flat = load_and_flatten(path)
        # reshape and compute band means
        cube = flat.reshape(IMG_SHAPE)
        band_means = cube.mean(axis=(0, 1))
        features.append(band_means)
    return np.vstack(features)


# --- 1) Load CSVs ---
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# --- 2) Build feature matrices ---
print("Extracting features for training set...")
X = extract_features(train_df)    # shape (n_train, 125)
y = train_df['label'].values

print("Extracting features for test set...")
X_test = extract_features(test_df) # shape (n_test, 125)

# --- 3) Train‐validation split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# --- 4) XGBoost Regressor setup ---
xgb_model = xgb.XGBRegressor(
    n_estimators=800,
    learning_rate=0.08,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='gpu_hist'  # or 'hist' if no GPU
)

# --- 5) Train with early stopping ---
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric='mae',
    early_stopping_rounds=20,
    verbose=True
)

# --- 6) Validation performance ---
y_pred_val = xgb_model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_pred_val)
print(f"Validation MAE: {val_mae:.4f}")

# --- 7) Predict on test set & save submission ---
y_pred_test = xgb_model.predict(X_test)
y_pred_test = np.clip(np.round(y_pred_test), 1, 100).astype(int)

submission_df = pd.DataFrame({
    "id": test_df["id"],
    "label": y_pred_test
})
submission_df.to_csv(SUBMISSION, index=False)
print(f"✅ Submission saved to {SUBMISSION}")



import os
import numpy as np
import pandas as pd
import tensorflow as tf


# --- Configuration ---
BASE_DIR = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025"
NPY_DIR = os.path.join(BASE_DIR, "ot/ot")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
SUBMISSION_CSV = "submission.csv"

IMG_SHAPE = (128, 128, 125)
BATCH_SIZE = 16
EPOCHS = 20
AUTOTUNE = tf.data.AUTOTUNE


# --- Utility: pad or crop to target shape ---
def fix_shape(arr):
    target = IMG_SHAPE
    fixed = np.zeros(target, dtype=np.float32)
    # compute minimal overlap
    mins = np.minimum(arr.shape, target)
    fixed[:mins[0], :mins[1], :mins[2]] = arr[:mins[0], :mins[1], :mins[2]]
    return fixed


# --- Data loading functions for tf.data ---
def _load_npy(path):
    arr = np.load(path.decode())
    arr = fix_shape(arr)
    arr = arr.astype(np.float32) / 255.0
    # add channel
    return arr[..., np.newaxis]


def parse_train(path, label):
    x = _load_npy(path)
    return x, label


def parse_test(path):
    x = _load_npy(path)
    return x



# --- Build tf.data pipelines ---
# Load metadata
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# File paths
train_paths = train_df['id'].apply(lambda x: os.path.join(NPY_DIR, x)).values
train_labels = train_df['label'].values.astype(np.float32)

test_paths = test_df['id'].apply(lambda x: os.path.join(NPY_DIR, x)).values

# Training dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = (train_ds
            .shuffle(len(train_paths))
            .map(lambda p, y: tf.py_function(parse_train, [p, y], [tf.float32, tf.float32]), num_parallel_calls=AUTOTUNE)
            .map(lambda x, y: (tf.ensure_shape(x, (*IMG_SHAPE, 1)), tf.ensure_shape(y, [])))
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))


# Validation split
val_size = int(0.1 * len(train_paths))
val_ds = train_ds.take(val_size)
train_ds = train_ds.skip(val_size)

# Test dataset
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = (test_ds
           .map(lambda p: tf.py_function(parse_test, [p], tf.float32), num_parallel_calls=AUTOTUNE)
           .map(lambda x: tf.ensure_shape(x, (*IMG_SHAPE, 1)))
           .batch(BATCH_SIZE)
           .prefetch(AUTOTUNE))


# --- Transformer Model Definition ---
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Multi-head Self-Attention
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = tf.keras.layers.Add()([x, inputs])
    # Feed-forward
    y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = tf.keras.layers.Dense(ff_dim, activation="relu")(y)
    y = tf.keras.layers.Dense(inputs.shape[-1])(y)
    return tf.keras.layers.Add()([y, x])


def build_transformer_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    # flatten spatial dims into sequence
    seq = tf.keras.layers.Reshape((input_shape[0] * input_shape[1], input_shape[2]))(inputs)
    # transformer blocks
    x = transformer_encoder(seq, head_size=32, num_heads=4, ff_dim=128)
    x = transformer_encoder(x, head_size=32, num_heads=4, ff_dim=128)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # regression head
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs)


model = build_transformer_model((*IMG_SHAPE, 1))
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()



# --- Train ---
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)


# WORKING GAVE GOOD RESULTS

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from sklearn.model_selection import train_test_split

# --- Configuration ---
BASE_DIR    = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025"
NPY_DIR     = os.path.join(BASE_DIR, "ot/ot")
TRAIN_CSV   = os.path.join(BASE_DIR, "train.csv")
TEST_CSV    = os.path.join(BASE_DIR, "test.csv")
SUBMISSION  = "submission.csv"

IMG_SHAPE   = (128, 128, 125)
TARGET_SIZE = np.prod(IMG_SHAPE)
BATCH_SIZE  = 4    # adjust to your GPU memory
EPOCHS      = 20
AUTOTUNE    = tf.data.AUTOTUNE


# --- 1) Robust loader + NaN handling + per-sample min-max ---
def load_and_process(path_bytes):
    path = path_bytes.numpy().decode()
    # try np.load; else fallback to raw read
    try:
        arr = np.load(path)
    except Exception:
        arr = np.fromfile(path, dtype=np.float32)
    else:
        arr = arr.ravel()
    # pad / truncate
    if arr.size < TARGET_SIZE:
        arr = np.pad(arr, (0, TARGET_SIZE - arr.size), mode="constant")
    else:
        arr = arr[:TARGET_SIZE]
    # reshape
    cube = arr.reshape(IMG_SHAPE).astype(np.float32)
    # replace NaNs & infs
    cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)
    # per-sample min-max
    mn, mx = cube.min(), cube.max()
    cube = (cube - mn) / ( (mx - mn) + 1e-6 )
    # add channel axis
    return cube[..., np.newaxis]


def tf_parse_train(path, label):
    x = tf.py_function(load_and_process, [path], tf.float32)
    x.set_shape((*IMG_SHAPE, 1))
    return x, tf.cast(label, tf.float32)


def tf_parse_test(path):
    x = tf.py_function(load_and_process, [path], tf.float32)
    x.set_shape((*IMG_SHAPE, 1))
    return x


# --- 2) Prepare tf.data pipelines ---
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

train_paths  = train_df["id"].apply(lambda f: os.path.join(NPY_DIR, f)).values
train_labels = train_df["label"].values.astype(np.float32)
test_paths   = test_df["id"].apply(lambda f: os.path.join(NPY_DIR, f)).values

# full dataset
full_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
full_ds = full_ds.shuffle(len(train_paths), reshuffle_each_iteration=True)
full_ds = full_ds.map(tf_parse_train, num_parallel_calls=AUTOTUNE)

# split 10% for validation
val_count = int(0.1 * len(train_paths))
val_ds   = full_ds.take(val_count).batch(BATCH_SIZE).prefetch(AUTOTUNE)
train_ds = full_ds.skip(val_count).batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
      .map(tf_parse_test, num_parallel_calls=AUTOTUNE)
      .batch(BATCH_SIZE)
      .prefetch(AUTOTUNE)
)


# --- 3) Model: 3D-Conv ↓ → Transformer → Regression head ---
def transformer_block(x, head_size, num_heads, ff_dim, dropout=0.1):
    attn = layers.MultiHeadAttention(key_dim=head_size,
                                     num_heads=num_heads,
                                     dropout=dropout)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(x.shape[-1])(ff)
    return layers.Add()([x, ff])

def build_model():
    inp = layers.Input((*IMG_SHAPE, 1))
    # downsample via 3D conv
    x = layers.Conv3D(16, 3, strides=2, padding="same", activation="relu")(inp)  # 64×64×63×16
    x = layers.Conv3D(32, 3, strides=2, padding="same", activation="relu")(x)    # 32×32×32×32
    # flatten spatial dims into sequence
    b,h,w,d,c = x.shape
    x = layers.Reshape((h*w, d*c))(x)
    # two transformer blocks
    x = transformer_block(x, head_size=16, num_heads=2, ff_dim=64)
    x = transformer_block(x, head_size=16, num_heads=2, ff_dim=64)
    # regression head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="linear")(x)
    return Model(inp, out)

model = build_model()
opt = optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
model.compile(optimizer=opt, loss="mse", metrics=["mae"])
model.summary()


# --- 4) Train without NaNs ---
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)


# --- 5) Predict & Save Submission ---
preds = model.predict(test_ds).flatten()
preds = np.clip(np.round(preds), 1, 100).astype(int)

pd.DataFrame({
    "ID": test_df["id"],
    "label": preds
}).to_csv(SUBMISSION, index=False)

print("✅ Done — submission.csv created.")



import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from sklearn.model_selection import train_test_split

# --- Configuration ---
BASE_DIR    = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025"
NPY_DIR     = os.path.join(BASE_DIR, "ot/ot")
TRAIN_CSV   = os.path.join(BASE_DIR, "train.csv")
TEST_CSV    = os.path.join(BASE_DIR, "test.csv")
SUBMISSION  = "submission.csv"

IMG_SHAPE   = (128, 128, 125)
TARGET_SIZE = np.prod(IMG_SHAPE)
BATCH_SIZE  = 4
EPOCHS      = 20
AUTOTUNE    = tf.data.AUTOTUNE


# --- 1) Robust loader + NaN handling + per-sample min-max ---
def load_and_process(path_bytes):
    path = path_bytes.numpy().decode()
    try:
        arr = np.load(path)
    except Exception:
        arr = np.fromfile(path, dtype=np.float32)
    else:
        arr = arr.ravel()
    if arr.size < TARGET_SIZE:
        arr = np.pad(arr, (0, TARGET_SIZE - arr.size), mode="constant")
    else:
        arr = arr[:TARGET_SIZE]
    cube = arr.reshape(IMG_SHAPE).astype(np.float32)
    cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = cube.min(), cube.max()
    cube = (cube - mn) / ((mx - mn) + 1e-6)
    return cube[..., np.newaxis]


def tf_parse_train(path, label):
    x = tf.py_function(load_and_process, [path], tf.float32)
    x.set_shape((*IMG_SHAPE, 1))
    return x, tf.cast(label, tf.float32)


def tf_parse_test(path):
    x = tf.py_function(load_and_process, [path], tf.float32)
    x.set_shape((*IMG_SHAPE, 1))
    return x


# --- Data pipelines ---
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

train_paths  = train_df["id"].apply(lambda f: os.path.join(NPY_DIR, f)).values
train_labels = train_df["label"].values.astype(np.float32)
test_paths   = test_df["id"].apply(lambda f: os.path.join(NPY_DIR, f)).values

full_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
full_ds = full_ds.shuffle(len(train_paths), reshuffle_each_iteration=True)
full_ds = full_ds.map(tf_parse_train, num_parallel_calls=AUTOTUNE)

val_count = int(0.1 * len(train_paths))
val_ds   = full_ds.take(val_count).batch(BATCH_SIZE).prefetch(AUTOTUNE)
train_ds = full_ds.skip(val_count).batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
      .map(tf_parse_test, num_parallel_calls=AUTOTUNE)
      .batch(BATCH_SIZE)
      .prefetch(AUTOTUNE)
)


# --- Model: 3D-Conv ↓ → 2D-Conv layers → Transformer → Regression ---
def transformer_block(x, head_size, num_heads, ff_dim, dropout=0.1):
    attn = layers.MultiHeadAttention(key_dim=head_size,
                                     num_heads=num_heads,
                                     dropout=dropout)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(x.shape[-1])(ff)
    return layers.Add()([x, ff])

def build_model():
    inp = layers.Input((*IMG_SHAPE, 1))
    # 3D downsampling
    x = layers.Conv3D(16, 3, strides=2, padding="same", activation="relu")(inp)
    x = layers.Conv3D(32, 3, strides=2, padding="same", activation="relu")(x)
    # 2D conv: collapse spectral into channels
    shape = tf.keras.backend.int_shape(x)
    # shape: (batch, h, w, d, c)
    h, w, d, c = shape[1], shape[2], shape[3], shape[4]
    x2d = layers.Reshape((h, w, d * c))(x)
    # add deep 2D conv layers
    x2d = layers.Conv2D(64, 3, padding="same", activation="relu")(x2d)
    x2d = layers.Conv2D(64, 3, padding="same", activation="relu")(x2d)
    # prepare sequence for transformer
    seq_len = h * w
    feat_dim = 64
    x_seq = layers.Reshape((seq_len, feat_dim))(x2d)
    # transformer blocks
    x_seq = transformer_block(x_seq, head_size=16, num_heads=2, ff_dim=64)
    x_seq = transformer_block(x_seq, head_size=16, num_heads=2, ff_dim=64)
    # regression head
    x_out = layers.GlobalAveragePooling1D()(x_seq)
    x_out = layers.Dense(128, activation="relu")(x_out)
    x_out = layers.Dropout(0.3)(x_out)
    out = layers.Dense(1, activation="linear")(x_out)
    return Model(inp, out)

model = build_model()
opt = optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
model.compile(optimizer=opt, loss="mse", metrics=["mae"])
model.summary()

# --- Train & Submit ---
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
preds = model.predict(test_ds).flatten()
preds = np.clip(np.round(preds), 1, 100).astype(int)

pd.DataFrame({"ID": test_df["id"], "label": preds}).to_csv(SUBMISSION, index=False)
print("Saved submission:", SUBMISSION)



import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# --- Configuration ---
BASE_DIR    = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025"
NPY_DIR     = os.path.join(BASE_DIR, "ot/ot")
TRAIN_CSV   = os.path.join(BASE_DIR, "train.csv")
TEST_CSV    = os.path.join(BASE_DIR, "test.csv")
SUBMISSION  = "submission_xgb.csv"

IMG_SHAPE   = (128, 128, 125)
TARGET_SIZE = np.prod(IMG_SHAPE)


def load_and_flatten(path):
    """
    Load a .npy file (or raw .npy if header corrupt), flatten to 1D,
    pad by repeating last value if too short, or truncate if too long.
    """
    try:
        arr = np.load(path)
        flat = arr.ravel()
    except Exception:
        flat = np.fromfile(path, dtype=np.float32)

    # pad/truncate to TARGET_SIZE
    if flat.size < TARGET_SIZE:
        if flat.size == 0:
            # completely missing? fill with zeros
            flat = np.zeros(TARGET_SIZE, dtype=np.float32)
        else:
            # repeat last value
            pad_vals = np.full(TARGET_SIZE - flat.size, flat[-1], dtype=np.float32)
            flat = np.concatenate([flat, pad_vals])
    else:
        flat = flat[:TARGET_SIZE]

    return flat


def extract_features(df):
    """
    For each row in df (with 'id'), load the patch, fix shape, and compute
    mean reflectance for each of the 125 bands.
    Returns an (n_samples, 125) array.
    """
    features = []
    for fn in df['id']:
        path = os.path.join(NPY_DIR, fn)
        flat = load_and_flatten(path)
        # reshape and compute band means
        cube = flat.reshape(IMG_SHAPE)
        band_means = cube.mean(axis=(0, 1))
        features.append(band_means)
    return np.vstack(features)


# --- 1) Load CSVs ---
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# --- 2) Build feature matrices ---
print("Extracting features for training set...")
X = extract_features(train_df)    # shape (n_train, 125)
y = train_df['label'].values

print("Extracting features for test set...")
X_test = extract_features(test_df) # shape (n_test, 125)

# --- 3) Train‐validation split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# --- 4) XGBoost Regressor setup ---
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='gpu_hist'  # or 'hist' if no GPU
)

# --- 5) Train with early stopping ---
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric='mae',
    early_stopping_rounds=20,
    verbose=True
)

# --- 6) Validation performance ---
y_pred_val = xgb_model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_pred_val)
print(f"Validation MAE: {val_mae:.4f}")

# --- 7) Predict on test set & save submission ---
y_pred_test = xgb_model.predict(X_test)
y_pred_test = np.clip(np.round(y_pred_test), 1, 100).astype(int)

submission_df = pd.DataFrame({
    "id": test_df["id"],
    "label": y_pred_test
})
submission_df.to_csv(SUBMISSION, index=False)
print(f"✅ Submission saved to {SUBMISSION}")



import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 0) Mixed Precision & Strategy
# -----------------------------------------------------------------------------
mixed_precision.set_global_policy('mixed_float16')
strategy = tf.distribute.MirroredStrategy()
print(f"Using {strategy.num_replicas_in_sync} GPUs")

# -----------------------------------------------------------------------------
# 1) Configuration
# -----------------------------------------------------------------------------
BASE_DIR   = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025"
NPY_DIR    = os.path.join(BASE_DIR, "ot/ot")
TRAIN_CSV  = os.path.join(BASE_DIR, "train.csv")
TEST_CSV   = os.path.join(BASE_DIR, "test.csv")
SUBMISSION = "submission_ensemble.csv"

IMG_SHAPE   = (128, 128, 125)
TARGET_SIZE = np.prod(IMG_SHAPE)
BATCH_SIZE  = 4       # per GPU
EPOCHS      = 20
LR          = 1e-4
AUTOTUNE    = tf.data.AUTOTUNE

# -----------------------------------------------------------------------------
# 2) NaN-safe loader + normalization
# -----------------------------------------------------------------------------
def load_and_normalize(path_bytes):
    path = path_bytes.numpy().decode('utf-8')
    try:
        arr = np.load(path)
        flat = arr.ravel()
    except Exception:
        flat = np.fromfile(path, dtype=np.float32)
    # pad/truncate
    if flat.size < TARGET_SIZE:
        if flat.size == 0:
            flat = np.zeros(TARGET_SIZE, dtype=np.float32)
        else:
            pad = np.full(TARGET_SIZE - flat.size, flat[-1], dtype=np.float32)
            flat = np.concatenate([flat, pad])
    else:
        flat = flat[:TARGET_SIZE]
    cube = flat.reshape(IMG_SHAPE).astype(np.float32)
    cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = cube.min(), cube.max()
    cube = (cube - mn) / ((mx - mn) + 1e-6)
    return cube

def tf_process(path, label=None):
    cube = tf.py_function(load_and_normalize, [path], tf.float32)
    cube.set_shape(IMG_SHAPE)
    cnn_in = cube[..., tf.newaxis]
    seq_in = tf.reshape(cube, (IMG_SHAPE[0]*IMG_SHAPE[1], IMG_SHAPE[2]))
    if label is None:
        return (cnn_in, seq_in)
    return (cnn_in, seq_in), tf.cast(label, tf.float32)

# -----------------------------------------------------------------------------
# 3) Load CSV & Split
# -----------------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

paths = train_df['id'].apply(lambda f: os.path.join(NPY_DIR, f)).values
labels = train_df['label'].values.astype(np.float32)

p_train, p_val, y_train, y_val = train_test_split(
    paths, labels, test_size=0.1, random_state=42
)

# Build datasets
train_ds = (
    tf.data.Dataset
      .from_tensor_slices((p_train, y_train))
      .shuffle(len(p_train))
      .map(lambda p,y: tf_process(p,y), num_parallel_calls=AUTOTUNE)
      .batch(BATCH_SIZE)
      .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset
      .from_tensor_slices((p_val, y_val))
      .map(lambda p,y: tf_process(p,y), num_parallel_calls=AUTOTUNE)
      .batch(BATCH_SIZE)
      .prefetch(AUTOTUNE)
)

test_paths = test_df['id'].apply(lambda f: os.path.join(NPY_DIR, f)).values
test_ds = (
    tf.data.Dataset
      .from_tensor_slices(test_paths)
      .map(lambda p: tf_process(p), num_parallel_calls=AUTOTUNE)
      .batch(BATCH_SIZE)
      .prefetch(AUTOTUNE)
)

# -----------------------------------------------------------------------------
# 4) Model Definition (unchanged Big Model)
# -----------------------------------------------------------------------------
with strategy.scope():
    def inception3d(x, f1, f3, f5, proj):
        p1 = layers.Conv3D(f1,1,padding='same',activation='relu')(x)
        p2 = layers.Conv3D(proj,1,padding='same',activation='relu')(x)
        p2 = layers.Conv3D(f3,3,padding='same',activation='relu')(p2)
        p3 = layers.Conv3D(proj,1,padding='same',activation='relu')(x)
        p3 = layers.Conv3D(f5,5,padding='same',activation='relu')(p3)
        p4 = layers.MaxPool3D(3,strides=1,padding='same')(x)
        p4 = layers.Conv3D(f5,1,padding='same',activation='relu')(p4)
        return layers.Concatenate()([p1,p2,p3,p4])

    def transformer_block(seq, head_size, num_heads, ff_dim, dropout=0.1):
        attn = layers.MultiHeadAttention(key_dim=head_size,
                                         num_heads=num_heads,
                                         dropout=dropout)(seq, seq)
        x = layers.Add()([seq, attn])
        x = layers.LayerNormalization()(x)
        ff = layers.Dense(ff_dim, activation='relu')(x)
        ff = layers.Dense(x.shape[-1], dtype='float32')(ff)
        return layers.Add()([x, ff])

    inp_cnn = layers.Input((*IMG_SHAPE,1))
    inp_seq = layers.Input((IMG_SHAPE[0]*IMG_SHAPE[1], IMG_SHAPE[2]))

    # CNN branch
    x = layers.Conv3D(64,7,strides=2,padding='same',activation='relu')(inp_cnn)
    x = layers.MaxPool3D(3,strides=2,padding='same')(x)
    x = inception3d(x,32,64,16,32)
    x = inception3d(x,64,128,32,64)
    x = layers.GlobalAveragePooling3D()(x)
    out_cnn = layers.Dense(128, activation='relu')(x)

    # Transformer branch
    y = transformer_block(inp_seq,32,4,128)
    y = transformer_block(y,32,4,128)
    y = layers.GlobalAveragePooling1D()(y)
    out_trans = layers.Dense(128, activation='relu')(y)

    # Merge + head
    merged = layers.Concatenate()([out_cnn, out_trans])
    z = layers.Dense(256, activation='relu')(merged)
    z = layers.Dropout(0.4)(z)
    z = layers.Dense(64, activation='relu')(z)
    out = layers.Dense(1, activation='linear', dtype='float32')(z)

    model = Model([inp_cnn, inp_seq], out)
    model.compile(
        optimizer=optimizers.Adam(LR, clipnorm=1.0),
        loss='mse', metrics=['mae']
    )

model.summary()

# -----------------------------------------------------------------------------
# 5) Train & Predict
# -----------------------------------------------------------------------------
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

preds = model.predict(test_ds).flatten()
preds = np.clip(np.round(preds), 1, 100).astype(int)

pd.DataFrame({'id': test_df['id'], 'label': preds}) \
  .to_csv(SUBMISSION, index=False)

print("✅ Saved:", SUBMISSION)





