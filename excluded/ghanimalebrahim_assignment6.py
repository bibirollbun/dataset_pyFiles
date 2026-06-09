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
# DSEG660 Challenge â€” CIFAR-10 (from-scratch) Residual CNN
# AdamW â€¢ Warmupâ†’Cosine LR â€¢ CutMix + CutOut â€¢ Label Smoothing
# ============================================================

import os, math, random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers as KL, models as KM, regularizers as KR
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW

print("TF:", tf.__version__)

# --------------------------
# 0) Repro
# --------------------------
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --------------------------
# 1) Data (Keras CIFAR-10)  â€” used for train/val
# --------------------------
def to_float_n1p1(x):
    # map to [-1, 1]
    x = x.astype("float32") / 255.0
    return (x - 0.5) * 2.0

(x_tr_all, y_tr_all), (x_te_all, y_te_all) = cifar10.load_data()
x_tr_all, x_te_all = to_float_n1p1(x_tr_all), to_float_n1p1(x_te_all)
y_tr_all = y_tr_all.reshape(-1).astype("int32")
y_te_all = y_te_all.reshape(-1).astype("int32")

print("Train:", x_tr_all.shape, "Test:", x_te_all.shape)

# hold out a small validation slice
VAL_FRAC = 0.05
n_all = x_tr_all.shape[0]
n_val = int(VAL_FRAC * n_all)
x_val, y_val = x_tr_all[:n_val], y_tr_all[:n_val]
x_tr , y_tr  = x_tr_all[n_val:], y_tr_all[n_val:]

IMG_H = IMG_W = 32
N_CLS = 10

# --------------------------
# 2) Model â€” simple ResNet-ish stack (custom, no pretrain)
# --------------------------
def conv_bn_relu(x, ch, s=1, wd=1e-4):
    x = KL.Conv2D(ch, 3, strides=s, padding='same', use_bias=False,
                  kernel_regularizer=KR.l2(wd))(x)
    x = KL.BatchNormalization()(x)
    return KL.Activation('relu')(x)

def res_block(inp, ch, down=False, wd=1e-4):
    stride = 2 if down else 1
    y = conv_bn_relu(inp, ch, s=stride, wd=wd)
    y = conv_bn_relu(y, ch, s=1, wd=wd)
    skip = inp
    if down or inp.shape[-1] != ch:
        skip = KL.Conv2D(ch, 1, strides=stride, padding='same', use_bias=False,
                         kernel_regularizer=KR.l2(wd))(inp)
        skip = KL.BatchNormalization()(skip)
    out = KL.add([skip, y])
    return KL.Activation('relu')(out)

def build_net():
    inp = KL.Input(shape=(IMG_H, IMG_W, 3))
    x = conv_bn_relu(inp, 64, s=1)

    # Stage 1 (64): 3 blocks
    for _ in range(3):
        x = res_block(x, 64, down=False)

    # Stage 2 (128): 3 blocks, first downsamples
    x = res_block(x, 128, down=True)
    x = res_block(x, 128, down=False)
    x = res_block(x, 128, down=False)

    # Stage 3 (256): 2 blocks, first downsamples
    x = res_block(x, 256, down=True)
    x = res_block(x, 256, down=False)

    # Stage 4 (512): 2 blocks, first downsamples
    x = res_block(x, 512, down=True)
    x = res_block(x, 512, down=False)

    # Head
    x = KL.GlobalAveragePooling2D()(x)
    x = KL.Dense(256, activation='relu')(x)
    x = KL.Dropout(0.4)(x)
    out = KL.Dense(N_CLS, activation='softmax')(x)
    return KM.Model(inp, out, name="CustomResCNN")

model = build_net()
model.summary()

# --------------------------
# 3) tf.data pipeline with CutMix + CutOut
# --------------------------
CUTMIX_P = 0.7
CUTOUT_P = 0.5
CUTOUT_SIZE = 8

@tf.function
def _beta(alpha=1.0):
    # reparameterization (same idea as Beta(a,a))
    g1 = tf.random.gamma([], alpha, 1.0)
    g2 = tf.random.gamma([], alpha, 1.0)
    return g1 / (g1 + g2 + 1e-7)

def _cutmix(images, labels, alpha=1.0):
    lam = _beta(alpha)
    b = tf.shape(images)[0]
    idx = tf.random.shuffle(tf.range(b))
    img2 = tf.gather(images, idx)
    lab2 = tf.gather(labels, idx)

    H = IMG_H; W = IMG_W
    cx = tf.random.uniform([], 0, W, dtype=tf.int32)
    cy = tf.random.uniform([], 0, H, dtype=tf.int32)
    cut_w = tf.cast(tf.sqrt(1. - lam) * tf.cast(W, tf.float32), tf.int32)

    x1 = tf.clip_by_value(cx - cut_w//2, 0, W)
    x2 = tf.clip_by_value(cx + cut_w//2, 0, W)
    y1 = tf.clip_by_value(cy - cut_w//2, 0, H)
    y2 = tf.clip_by_value(cy + cut_w//2, 0, H)

    mask = tf.ones((y2 - y1, x2 - x1, 3), dtype=images.dtype)
    mask = tf.image.pad_to_bounding_box(mask, y1, x1, H, W)
    mixed = images * (1.0 - mask) + img2 * mask

    lam_adj = 1. - tf.cast((x2-x1)*(y2-y1), tf.float32) / tf.cast(H*W, tf.float32)
    y = lam_adj * tf.one_hot(labels, N_CLS) + (1. - lam_adj) * tf.one_hot(lab2, N_CLS)
    return mixed, y

def _cutout(images, size=CUTOUT_SIZE):
    def _one(img):
        cx = tf.random.uniform([], 0, IMG_W, dtype=tf.int32)
        cy = tf.random.uniform([], 0, IMG_H, dtype=tf.int32)
        x1 = tf.clip_by_value(cx - size//2, 0, IMG_W)
        x2 = tf.clip_by_value(cx + size//2, 0, IMG_W)
        y1 = tf.clip_by_value(cy - size//2, 0, IMG_H)
        y2 = tf.clip_by_value(cy + size//2, 0, IMG_H)
        m = tf.ones((y2 - y1, x2 - x1, 3), dtype=img.dtype)
        m = tf.image.pad_to_bounding_box(m, y1, x1, IMG_H, IMG_W)
        return img * (1.0 - m)
    return tf.map_fn(_one, images)

def _augment_batch(images, labels):
    images = tf.image.random_flip_left_right(images)
    images = tf.image.resize_with_crop_or_pad(images, IMG_H + 4, IMG_W + 4)
    images = tf.image.random_crop(images, [tf.shape(images)[0], IMG_H, IMG_W, 3])

    # CutMix OR identity â†’ one-hot in the else branch
    do_cm = tf.less(tf.random.uniform([]), CUTMIX_P)
    images, labels = tf.cond(do_cm,
                             lambda: _cutmix(images, labels, alpha=1.0),
                             lambda: (images, tf.one_hot(labels, N_CLS)))

    # CutOut after CutMix/no-CutMix
    do_co = tf.less(tf.random.uniform([]), CUTOUT_P)
    images = tf.cond(do_co, lambda: _cutout(images), lambda: images)
    return images, labels

def make_dataset(x, y, train=True, batch=256, seed=SEED):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if train:
        ds = ds.shuffle(8192, seed=seed).batch(batch).map(_augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch).map(lambda a, b: (a, tf.one_hot(b, N_CLS)), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

BATCH = 256
ds_train = make_dataset(x_tr, y_tr, train=True,  batch=BATCH)
ds_val   = make_dataset(x_val, y_val, train=False, batch=BATCH)

# --------------------------
# 4) Optimizer & LR schedule (Warmup â†’ Cosine), Label Smoothing
# --------------------------
EPOCHS = 240
WARMUP_E = 5
BASE_LR = 2e-3     # adjust with batch size if you change it
MIN_LR  = 5e-6
WEIGHT_DECAY = 1e-4

steps_per_epoch = math.ceil(x_tr.shape[0] / BATCH)

class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_epochs, total_epochs, steps_per_epoch, min_lr=0.0):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warm_steps = int(warmup_epochs * steps_per_epoch)
        self.total_steps = int(total_epochs * steps_per_epoch)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.base_lr * (step / tf.maximum(1., tf.cast(self.warm_steps, tf.float32)))
        prog = tf.clip_by_value((step - self.warm_steps) /
                                tf.maximum(1., self.total_steps - self.warm_steps), 0., 1.)
        cosv = self.min_lr + 0.5*(self.base_lr - self.min_lr)*(1. + tf.cos(math.pi * prog))
        return tf.where(step < self.warm_steps, warm, cosv)

    def get_config(self):
        return dict(base_lr=self.base_lr, warm_steps=self.warm_steps,
                    total_steps=self.total_steps, min_lr=self.min_lr)

lr_sched = WarmupCosine(BASE_LR, WARMUP_E, EPOCHS, steps_per_epoch, MIN_LR)
opt = AdamW(learning_rate=lr_sched, weight_decay=WEIGHT_DECAY)

# Label smoothing via CCE(smoothing)
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

# --------------------------
# 5) Train with early stop & best checkpoint
# --------------------------
ckpt_cb = ModelCheckpoint("best.keras", monitor="val_accuracy", mode="max",
                          save_best_only=True, verbose=1)
early_cb = EarlyStopping(monitor="val_accuracy", mode="max",
                         patience=30, restore_best_weights=True, verbose=1)

hist = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS,
    callbacks=[ckpt_cb, early_cb],
    verbose=2
)

val_loss, val_acc = model.evaluate(ds_val, verbose=0)
print(f"âœ… Holdout val acc: {val_acc:.4f} | val loss: {val_loss:.4f}")

# --------------------------
# 6) Load best model for inference/export
# --------------------------
best_model = tf.keras.models.load_model("best.keras", compile=False)
print("âœ… Loaded best checkpoint: best.keras")



# Clean re-extract of OFFICIAL test set to /kaggle/working/cifar10_extracted/test
from pathlib import Path
import shutil, subprocess, sys, os, glob

COMP_ROOT = Path("/kaggle/input/cifar-10")
WORK_ROOT = Path("/kaggle/working/cifar10_extracted")
TEST_DIR1 = WORK_ROOT / "test"          # usual
TEST_DIR2 = WORK_ROOT / "test" / "test" # sometimes 7z nests one level

# 0) Sanity: make sure the source exists
assert (COMP_ROOT/"test.7z").exists(), "Could not find /kaggle/input/cifar-10/test.7z"

# 1) Remove any partial test extraction (SAFE: only the 'test' subdir)
for d in [TEST_DIR2, TEST_DIR1]:
    if d.exists():
        print("Removing partial:", d)
        shutil.rmtree(d)

WORK_ROOT.mkdir(parents=True, exist_ok=True)

# 2) Re-extract test only (overwrite, multithread)
print("Extracting ONLY test.7z to", WORK_ROOT)
subprocess.check_call(["7z","x","-y","-bb0","-mmt=on","-aoa", f"-o{str(WORK_ROOT)}", str(COMP_ROOT/"test.7z")])

# 3) Locate the real test folder (handles optional nested /test/test)
def find_test_dir(root: Path):
    c1 = len(glob.glob(str(root/"test/*.png")))
    c2 = len(glob.glob(str(root/"test/test/*.png")))
    if c2 > 0 and c1 == 0:
        return root/"test/test", c2
    return root/"test", c1

TEST_DIR, n = find_test_dir(WORK_ROOT)
print("Test dir:", TEST_DIR)
print("PNG count:", n)
assert n == 300000, f"Expected 300000 test PNGs, found {n}"



from pathlib import Path
TEST_DIR = Path("/kaggle/working/cifar10_extracted/test")
print("official test pngs:", len(list(TEST_DIR.glob("*.png"))))  # should be 300000



from pathlib import Path
DATASET_ROOT = Path("/kaggle/working/cifar10_extracted")
test_paths = sorted((DATASET_ROOT/"test").glob("*.png"), key=lambda p: int(p.stem))
ids = [int(p.stem) for p in test_paths]
print(len(ids), ids[:5], ids[-5:])   # should show 300000 and end at 300000



import tensorflow as tf

def load_and_preprocess(path):
    img = tf.io.read_file(path)              # path is a tf.string
    img = tf.io.decode_png(img, 3)
    img = tf.image.resize(img, [32, 32])
    img = tf.image.convert_image_dtype(img, tf.float32)
    return (img - 0.5) * 2.0

BATCH = 1024

# Convert Path objects -> strings
img_paths = [str(p) for p in test_paths]

test_ds = (tf.data.Dataset.from_tensor_slices(img_paths)
           .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
           .batch(BATCH)
           .prefetch(tf.data.AUTOTUNE))



# Option A: peek one batch
for x in test_ds.take(1):
    print(x.shape, x.dtype)   # e.g. (1024, 32, 32, 3) tf.float32



# === QUICK REBUILD OF best.keras (short training just to create the file) ===
import tensorflow as tf, math
from tensorflow.keras import layers as KL, models as KM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import AdamW

# 1) Load CIFAR-10 and normalize to [-1,1]
(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
x_train = (x_train.astype("float32")/255.0 - 0.5) * 2.0
y_train = y_train.reshape(-1).astype("int32")

# small val split
n = x_train.shape[0]; n_val = int(0.05*n)
x_val, y_val = x_train[:n_val], y_train[:n_val]
x_tr , y_tr  = x_train[n_val:], y_train[n_val:]

# 2) Build a model
try:
    # If your custom builder exists, use it for best results
    model = build_custom_resnet()
except NameError:
    # Fallback simple CNN (only used if your builder isn't defined)
    inputs = KL.Input((32,32,3))
    x = KL.Conv2D(64,3,padding="same",activation="relu")(inputs)
    x = KL.Conv2D(64,3,padding="same",activation="relu")(x)
    x = KL.MaxPool2D()(x)
    x = KL.Conv2D(128,3,padding="same",activation="relu")(x)
    x = KL.Conv2D(128,3,padding="same",activation="relu")(x)
    x = KL.GlobalAveragePooling2D()(x)
    x = KL.Dense(256,activation="relu")(x)
    outputs = KL.Dense(10,activation="softmax")(x)
    model = KM.Model(inputs, outputs)

# 3) Compile (simple, fast)
model.compile(
    optimizer=AdamW(learning_rate=2e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# 4) Short train to produce best.keras
ckpt = ModelCheckpoint("best.keras", monitor="val_accuracy", mode="max",
                       save_best_only=True, verbose=1)
model.fit(
    x_tr, y_tr,
    validation_data=(x_val, y_val),
    epochs=3,            # keep small just to create the file quickly
    batch_size=256,
    callbacks=[ckpt],
    verbose=2
)

print("âœ… Wrote best.keras to /kaggle/working")



# RIGHT AFTER the quick-train cell, still before Step 3:
import tensorflow as tf
model = tf.keras.models.load_model("/kaggle/working/best.keras", compile=False)
print("Loaded /kaggle/working/best.keras")



# === Step 3: inference on OFFICIAL test set -> submission.csv ===
import math, numpy as np, pandas as pd, tensorflow as tf

# Use the in-memory model if present; otherwise load the checkpoint we just wrote
if 'model' not in globals() or not isinstance(model, tf.keras.Model):
    model = tf.keras.models.load_model("/kaggle/working/best.keras", compile=False)
print("Using model for inference.")

@tf.function
def fwd(x):
    return model(x, training=False)

pred_chunks = []
for i, batch in enumerate(test_ds, 1):
    logits = fwd(batch)
    pred_chunks.append(tf.argmax(logits, axis=1))
    if i % 25 == 0:
        print(f"processed {i}/{math.ceil(len(ids)/BATCH)}")

pred_idx = tf.concat(pred_chunks, axis=0).numpy()

CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
submission = pd.DataFrame(
    {"id": ids, "label": [CLASS_NAMES[i] for i in pred_idx]}
)
submission.to_csv("submission.csv", index=False)
print("âœ… Saved submission.csv")
display(submission.head())



# ============================================================
# CIFAR-10 â€” Build submission.csv from best.keras (paraphrased)
# ============================================================

import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# ---------- Config: dataset mount that has sampleSubmission.csv ----------
DATA_ROOT = Path("/kaggle/input/cifar10-object-recognition-in-images-zip-file")
SAMPLE_SUB = DATA_ROOT / "sampleSubmission.csv"

# Test images in this dataset can live under either:
#   .../train_test/test/*.png       or       .../train_test/test/test/*.png
TEST_A = DATA_ROOT / "train_test" / "test"
TEST_B = DATA_ROOT / "train_test" / "test" / "test"
TEST_DIR = TEST_B if TEST_B.exists() else TEST_A

print("DATA_ROOT:", DATA_ROOT)
print("TEST_DIR :", TEST_DIR)

# ---------- Load the trained model ----------
model = tf.keras.models.load_model("best.keras", compile=False)
print("âœ… Loaded checkpoint: best.keras")

# ---------- Read the expected IDs from the sample file ----------
sample_df = pd.read_csv(SAMPLE_SUB)
ids = sample_df["id"].astype(str).tolist()
print(f"ğŸ§¾ IDs to predict: {len(ids)}")

# ---------- Resolve a file path for every id ----------
def path_for_id(img_id: str) -> str:
    # prefer .png, fall back to .jpg
    p_png = TEST_DIR / f"{img_id}.png"
    if p_png.exists():
        return str(p_png)
    p_jpg = TEST_DIR / f"{img_id}.jpg"
    if p_jpg.exists():
        return str(p_jpg)
    raise FileNotFoundError(f"Missing image for id={img_id} in {TEST_DIR}")

img_paths = [path_for_id(i) for i in ids]

# ---------- tf.data loader: decode â†’ float32 â†’ resize 32x32 â†’ [-1,1] ----------
def load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [32, 32])
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return (img - 0.5) * 2.0                              # [-1,1]

BATCH = 1024
test_ds = (
    tf.data.Dataset.from_tensor_slices(img_paths)
    .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

# ---------- Predict and write submission ----------
prob_chunks = model.predict(test_ds, verbose=1)
pred_idx = np.argmax(prob_chunks, axis=1)

CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
pred_labels = [CLASS_NAMES[i] for i in pred_idx]

submission = pd.DataFrame({"id": ids, "label": pred_labels})
submission.to_csv("submission.csv", index=False)
print("ğŸ“� Saved submission.csv")
display(submission.head())



import math, numpy as np, pandas as pd, tensorflow as tf

model = tf.keras.models.load_model("best.keras", compile=False)

@tf.function
def fwd(x): return model(x, training=False)

pred_idx = []
i = 0
for batch in test_ds:             # reuse the same test_ds
    logits = fwd(batch)
    pred_idx.append(tf.argmax(logits, axis=1))
    i += 1
    if i % 25 == 0:
        print(f"processed {i}/{math.ceil(len(ids)/BATCH)} batches")

pred_idx = tf.concat(pred_idx, axis=0).numpy()

CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
submission = pd.DataFrame({"id": ids, "label": [CLASS_NAMES[i] for i in pred_idx]})
submission.to_csv("submission.csv", index=False)
print("âœ… Saved submission.csv")
display(submission.head())



# ==== CIFAR-10 | Submission QA (paraphrased) ====
import pandas as pd
import numpy as np

CSV_PATH = "submission.csv"
CLASSES = {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"}

df = pd.read_csv(CSV_PATH)

# 1) Basic structure
print(f"Shape: {df.shape}")                     # expect (300000, 2)
print("Columns:", list(df.columns))             # expect ['id', 'label']

# 2) ID integrity
ids = df["id"]
print("IDs unique:", ids.is_unique)             # expect True
print("ID dtype:", ids.dtype)
print("ID min/max:", ids.min(), "â†’", ids.max())

# Optional: are IDs exactly 1..300000?
expected_n = 300000
if len(df) == expected_n:
    # Only do this set check if size matches (avoids huge memory for mismatched sizes)
    exact_range_ok = set(ids.values) == set(range(1, expected_n+1))
    print("IDs cover 1..300000 exactly:", exact_range_ok)

# 3) Nulls
print("Nulls by column:", df.isna().sum().to_dict())

# 4) Labels sanity
lbls = df["label"]
unknown = set(lbls.unique()) - CLASSES
print("Only known class names:", len(unknown) == 0)
if unknown:
    print("Unknown labels found:", unknown)

# 5) Distribution snapshot
print("\nLabel distribution:")
print(lbls.value_counts().sort_index())

# 6) Peek at random rows (reproducible)
SAMPLE_SEED = 1337
print("\nRandom sample:")
print(df.sample(10, random_state=SAMPLE_SEED))





