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


# Environment + logging + reproducibility

import os, random, json
from pathlib import Path

# Quiet TensorFlow logs (0=all, 1=INFO, 2=WARNING, 3=ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SEED = 42
def set_py_seed(seed=SEED):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
set_py_seed()

# Paths
DATA_DIR = Path("/kaggle/input/leaf-classification")
TRAIN_ZIP = DATA_DIR / "train.csv.zip"
TEST_ZIP = DATA_DIR / "test.csv.zip"
SAMPLE_ZIP = DATA_DIR / "sample_submission.csv.zip"
IMAGES_ZIP = DATA_DIR / "images.zip"  # optional
WORK_DIR = Path("/kaggle/working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

print("Setup OK")



# Non-TF imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix
)

print("Imported sklearn, numpy, pandas, matplotlib")



# Import TensorFlow and configure device(s)

import tensorflow as tf

# Optional: use GPU(s) with memory growth
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Memory growth not set:", e)

tf.random.set_seed(SEED)

# Multi-GPU strategy (works with 0, 1, or many GPUs)
strategy = tf.distribute.MirroredStrategy()
NUM_REPLICAS = strategy.num_replicas_in_sync
print("TensorFlow:", tf.__version__)
print("GPUs:", gpus)
print("Replicas in sync:", NUM_REPLICAS)

# Training hyperparameters (batch scales with replicas, capped)
EPOCHS = 50
PATIENCE = 8
BASE_BATCH = 64
BATCH = min(BASE_BATCH * max(1, NUM_REPLICAS), 256)
print("Batch size:", BATCH)



# Load CSVs directly from .zip files

train_df = pd.read_csv(TRAIN_ZIP, compression="zip")
test_df  = pd.read_csv(TEST_ZIP,  compression="zip")
sample_sub = pd.read_csv(SAMPLE_ZIP, compression="zip")

id_col = "id"
target_col = "species"
feature_cols = [c for c in train_df.columns if c not in [id_col, target_col]]

X = train_df[feature_cols].values
y_labels = train_df[target_col].values
X_test = test_df[feature_cols].values
test_ids = test_df[id_col].values

le = LabelEncoder()
y_int = le.fit_transform(y_labels)
num_classes = len(le.classes_)

print(f"Rows={len(train_df)}  Features={len(feature_cols)}  Classes={num_classes}")
train_df.head(3)



# EDA artifacts for the report

# Class balance (top 20)
cls_counts = pd.Series(y_labels).value_counts().sort_values(ascending=False)
plt.figure(figsize=(10,4))
cls_counts.head(20).plot(kind="bar")
plt.title("Top 20 species counts")
plt.tight_layout()
plt.savefig(WORK_DIR / "eda_class_balance_top20.png")
plt.close()

# PCA scatter (2D)
X_std_for_pca = StandardScaler().fit_transform(X)
pc = PCA(n_components=2, random_state=SEED).fit_transform(X_std_for_pca)
plt.figure(figsize=(6,5))
plt.scatter(pc[:,0], pc[:,1], s=6, c=y_int, cmap="tab20")
plt.title("PCA on standardized features")
plt.tight_layout()
plt.savefig(WORK_DIR / "eda_pca.png")
plt.close()

print("Saved EDA plots:", (WORK_DIR / 'eda_class_balance_top20.png').name, (WORK_DIR / 'eda_pca.png').name)



# EDA artifacts for the report

# --- Class balance (top 20) ---
cls_counts = pd.Series(y_labels).value_counts().sort_values(ascending=False)
plt.figure(figsize=(10,4))
cls_counts.head(20).plot(kind="bar")
plt.title("Top 20 species counts")
plt.tight_layout()
plt.savefig(WORK_DIR / "eda_class_balance_top20.png")
plt.show()   # show inline

# --- PCA scatter (2D) ---
X_std_for_pca = StandardScaler().fit_transform(X)
pc = PCA(n_components=2, random_state=SEED).fit_transform(X_std_for_pca)
plt.figure(figsize=(6,5))
plt.scatter(pc[:,0], pc[:,1], s=6, c=y_int, cmap="tab20")
plt.title("PCA on standardized features")
plt.tight_layout()
plt.savefig(WORK_DIR / "eda_pca.png")
plt.show()   # show inline




# Standardize and reshape to (timesteps, channels) for Conv1D

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_test_std = scaler.transform(X_test)

X_1d = X_std[..., None]       # (n_samples, n_features, 1)
X_test_1d = X_test_std[..., None]
input_shape = (X_1d.shape[1], 1)

print("Train shape:", X_1d.shape, " Test shape:", X_test_1d.shape)



import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.random.set_seed(SEED)

# Make TF play nice with GPU memory (if any)
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Memory growth not set:", e)

strategy = tf.distribute.MirroredStrategy()
NUM_REPLICAS = strategy.num_replicas_in_sync
print("TF:", tf.__version__, "| Replicas:", NUM_REPLICAS)

EPOCHS = 50
PATIENCE = 8
BASE_BATCH = 64
BATCH = min(BASE_BATCH * max(1, NUM_REPLICAS), 256)
print("BATCH =", BATCH)



from tensorflow.keras import layers, models, optimizers

def build_cnn_baseline(input_shape, num_classes):
    m = models.Sequential(name="cnn_baseline")
    m.add(layers.Conv1D(32, 5, activation="relu", input_shape=input_shape))
    m.add(layers.MaxPooling1D(2))
    m.add(layers.Flatten())
    m.add(layers.Dense(64, activation="relu"))
    m.add(layers.Dense(num_classes, activation="softmax"))
    m.compile(optimizer=optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m

from tensorflow.keras import layers, models, optimizers, initializers

def build_cnn_deep(input_shape, num_classes):
    he = initializers.HeNormal()

    inp = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv1D(64, 7, padding="same", kernel_initializer=he)(inp)
    x = layers.BatchNormalization()(x)           # BN before activation is fine
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.15)(x)                  # smaller dropout

    # Block 2
    x = layers.Conv1D(128, 5, padding="same", kernel_initializer=he)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.20)(x)

    # Block 3
    x = layers.Conv1D(128, 3, padding="same", kernel_initializer=he)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Use both GAP and Flattened features (richer than GAP alone)
    gap = layers.GlobalAveragePooling1D()(x)
    flat = layers.Flatten()(x)
    x = layers.Concatenate()([gap, flat])

    x = layers.Dense(128, activation="relu", kernel_initializer=he)(x)
    x = layers.Dropout(0.25)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    m = models.Model(inp, out, name="cnn_deep_fixed")
    m.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),  # lower LR stabilizes BN
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def build_cnn_tunable(input_shape, num_classes, filters=64, ksize=5, dense_units=128, dr=0.25, lr=1e-3):
    he = tf.keras.initializers.HeNormal()
    inp = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv1D(filters, ksize, padding="same", kernel_initializer=he)(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling1D()(x); x = layers.Dropout(dr)(x)

    # Block 2
    x = layers.Conv1D(filters*2, ksize, padding="same", kernel_initializer=he)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling1D()(x); x = layers.Dropout(dr)(x)

    # Block 3
    x = layers.Conv1D(filters*2, 3, padding="same", kernel_initializer=he)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    # Use both GAP and Flatten
    gap = layers.GlobalAveragePooling1D()(x)
    flat = layers.Flatten()(x)
    x = layers.Concatenate()([gap, flat])

    x = layers.Dense(dense_units, activation="relu", kernel_initializer=he)(x)
    x = layers.Dropout(dr)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    m = models.Model(inp, out, name="cnn_tunable_fixed")
    m.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return m




from tensorflow.keras import callbacks
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf

# Clear any aborted collectives state before re-running CV
tf.keras.backend.clear_session()

# Use 1 device for CV only (does NOT affect your MirroredStrategy used elsewhere)
_cv_device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
cv_strategy = tf.distribute.OneDeviceStrategy(device=_cv_device)

def with_strategy(builder):
    # Rebind to use the single-device strategy just for CV
    def _wrapped(_shape, _classes, *args, **kwargs):
        with cv_strategy.scope():
            return builder(_shape, _classes, *args, **kwargs)
    return _wrapped


def train_with_val(model, X_tr, y_tr, X_va, y_va):
    cbs = [
        callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_accuracy"),
        callbacks.ReduceLROnPlateau(patience=max(PATIENCE//2, 3), factor=0.5, min_lr=1e-5, monitor="val_loss")
    ]
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS, batch_size=BATCH, verbose=0, callbacks=cbs
    )
    proba = model.predict(X_va, verbose=0)
    acc = accuracy_score(y_va, np.argmax(proba, axis=1))
    return proba, acc, hist.history

def run_cv(model_builder, X_data, y_data, folds=5, name="model"):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y_data), num_classes), dtype=np.float32)
    accs = []
    for fold, (tr, va) in enumerate(skf.split(X_data, y_data), start=1):
        print(f"[{name}] Fold {fold}/{folds}")
        X_tr, X_va = X_data[tr], X_data[va]
        y_tr, y_va = y_data[tr], y_data[va]
        model = model_builder(input_shape, num_classes)
        proba, acc, _ = train_with_val(model, X_tr, y_tr, X_va, y_va)
        oof[va] = proba
        accs.append(acc)
        print(f"  val_acc={acc:.4f}")
    oof_acc = accuracy_score(y_data, np.argmax(oof, axis=1))
    print(f"[{name}] mean_acc={np.mean(accs):.4f}  std={np.std(accs):.4f}  oof_acc={oof_acc:.4f}")
    return oof, accs



oof_baseline, accs_baseline = run_cv(
    with_strategy(build_cnn_baseline), X_1d, y_int, folds=5, name="CNN_baseline"
)

oof_deep, accs_deep = run_cv(
    with_strategy(build_cnn_deep), X_1d, y_int, folds=5, name="CNN_deep"
)



# Lightweight but smart hyperparameter search for the 3rd CNN model

import random
from sklearn.metrics import accuracy_score

cfg = {"filters": 64, "ksize": 5, "dense_units": 192, "dr": 0.25, "lr": 1e-3}
oof_tuned, accs_tuned = run_cv(
    with_strategy(lambda s, c: build_cnn_tunable(s, c, **cfg)),
    X_1d, y_int, folds=5, name="CNN_tuned_fixed"
)




# Micro-averaged ROC/PR with macro & weighted summaries (readable for 99 classes)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import tensorflow as tf

def plot_multiclass_roc_pr(y_true_int, y_proba, title_prefix):
    y_true = tf.keras.utils.to_categorical(y_true_int, num_classes=num_classes)

    # ROC summaries
    try:
        auc_macro = roc_auc_score(y_true, y_proba, average="macro", multi_class="ovr")
        auc_weighted = roc_auc_score(y_true, y_proba, average="weighted", multi_class="ovr")
    except Exception:
        auc_macro = float("nan"); auc_weighted = float("nan")

    fpr, tpr, _ = roc_curve(y_true.ravel(), y_proba.ravel())
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label="micro ROC")
    plt.plot([0,1],[0,1], linestyle="--", label="chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} — ROC (micro)\nmacro AUC={auc_macro:.3f} | weighted AUC={auc_weighted:.3f}")
    plt.legend(); plt.tight_layout(); plt.show()

    # PR summaries
    precision, recall, _ = precision_recall_curve(y_true.ravel(), y_proba.ravel())
    ap_macro = average_precision_score(y_true, y_proba, average="macro")
    ap_weighted = average_precision_score(y_true, y_proba, average="weighted")
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label="micro PR")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{title_prefix} — PR (micro)\nmacro AP={ap_macro:.3f} | weighted AP={ap_weighted:.3f}")
    plt.legend(); plt.tight_layout(); plt.show()

# Plot for each
plot_multiclass_roc_pr(y_int, oof_baseline, "CNN Baseline")




plot_multiclass_roc_pr(y_int, oof_deep, "CNN Deep")



plot_multiclass_roc_pr(y_int, oof_tuned,    "CNN Tuned (fixed)")


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Use OOF predictions (already computed in CV) instead of tuned_holdout
va_pred = np.argmax(oof_tuned, axis=1)
cm = confusion_matrix(y_int, va_pred)

plt.figure(figsize=(14,12))
sns.heatmap(cm, cmap="Blues", cbar=True, square=True, 
            xticklabels=False, yticklabels=False)
plt.title("Confusion Matrix (Tuned CNN, 99 classes)", fontsize=16)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()



# === Deep CNN training with dynamic-safe batch size (multi-GPU proof) ===
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import callbacks

# Assumes you already defined: strategy, build_cnn_deep, input_shape, num_classes,
# X_1d, y_int, X_test_1d, sample_sub, le, test_ids, SEED, EPOCHS, PATIENCE

SEED     = globals().get("SEED", 42)
EPOCHS   = globals().get("EPOCHS", 50)
PATIENCE = globals().get("PATIENCE", 8)
strategy = globals().get("strategy", tf.distribute.MirroredStrategy())
NUM_REPLICAS = strategy.num_replicas_in_sync

# Split once (we'll size batch based on actual val size)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_1d, y_int, test_size=0.10, stratify=y_int, random_state=SEED
)

def choose_safe_batch(val_size, num_replicas, target=64):
    """Largest batch ≤ target and ≤ val_size that is divisible by replicas."""
    b = min(target, val_size)
    b = (b // num_replicas) * num_replicas
    if b < num_replicas:  # ensure at least one example per replica
        b = num_replicas
    return int(b)

# Pick a safe global batch size (works for both train and val)
BATCH = choose_safe_batch(len(y_va), NUM_REPLICAS, target=64)  # e.g., with val≈99 and 2 GPUs -> 64
print(f"Replicas: {NUM_REPLICAS} | Val size: {len(y_va)} | Using BATCH={BATCH}")

def make_ds(X, y=None, batch_size=32, training=False, drop=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y)) if y is not None else tf.data.Dataset.from_tensor_slices(X)
    if training:
        ds = ds.shuffle(min(len(X), 2048), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=drop)  # force equal shapes for all steps
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_ds(X_tr, y_tr, batch_size=BATCH, training=True,  drop=True)
val_ds   = make_ds(X_va, y_va, batch_size=BATCH, training=False, drop=True)  # drop remainder to keep per-replica equal

with strategy.scope():
    model = build_cnn_deep(input_shape, num_classes)

cbs = [
    callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_accuracy"),
    callbacks.ReduceLROnPlateau(patience=max(PATIENCE//2, 3), factor=0.5, min_lr=1e-5, monitor="val_loss"),
]

print("Steps/epoch:", tf.data.experimental.cardinality(train_ds).numpy(),
      "| Val steps:", tf.data.experimental.cardinality(val_ds).numpy())

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1, callbacks=cbs)

# Exact val accuracy (predict on full val set without dropping)
val_proba = model.predict(tf.data.Dataset.from_tensor_slices(X_va).batch(BATCH), verbose=0)
val_acc = accuracy_score(y_va, val_proba.argmax(axis=1))
print(f"Validation accuracy (full val): {val_acc:.4f}")

# Test predictions (no dropping needed)
test_proba = model.predict(tf.data.Dataset.from_tensor_slices(X_test_1d).batch(BATCH), verbose=0)
print("Test proba shape:", test_proba.shape)

# Build submission
id_col = "id"
class_cols = list(sample_sub.columns); class_cols.remove(id_col)
sub_probs = pd.DataFrame(0.0, index=np.arange(len(test_ids)), columns=class_cols)
sub_probs[le.classes_] = test_proba
submission = pd.concat([pd.Series(test_ids, name=id_col), sub_probs], axis=1)

out_path = "/kaggle/working/submission_deep.csv"
submission.to_csv(out_path, index=False)
print("Wrote submission to:", out_path)
display(submission.head(3))





