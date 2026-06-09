# ==== Cell 1: setup ====
import os, math, random, json, pathlib, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

print("TensorFlow:", tf.__version__)
assert tf.__version__.startswith("2."), "TF 2.x required"

# Seed (note: we do not force deterministic ops; Kaggle speed matters)
SEED       = 42
IMG_SIZE   = 32
NUM_CLASSES= 10
BATCH_SIZE = 128
AUTO       = tf.data.AUTOTUNE

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# CIFAR-10 class names (label order used by Keras)
CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Small helpers
def to_one_hot(y, num_classes=NUM_CLASSES):
    return tf.one_hot(tf.cast(y, tf.int32), num_classes)

print("Setup OK.")



# ==== Cell 2: data arrays (no external weights) ====
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train_full = y_train_full.squeeze().astype("int32")
y_test       = y_test.squeeze().astype("int32")

# scale to [0,1]
x_train_full = x_train_full.astype("float32")/255.0
x_test       = x_test.astype("float32")/255.0

# 45k train / 5k val
x_train, y_train = x_train_full[:45000], y_train_full[:45000]
x_val,   y_val   = x_train_full[45000:], y_train_full[45000:]

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)



# ==== Cell 3: augmentations ====

@tf.function
def random_crop_pad_flip(img):
    # classic CIFAR trick: pad 4 -> random crop -> flip
    img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE + 4, IMG_SIZE + 4)
    img = tf.image.random_crop(img, size=[IMG_SIZE, IMG_SIZE, 3])
    img = tf.image.random_flip_left_right(img)
    return img

@tf.function
def cutout_image(img, size=8):
    # per-image CutOut (no control-flow pitfalls)
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    cy = tf.random.uniform([], 0, h, dtype=tf.int32)
    cx = tf.random.uniform([], 0, w, dtype=tf.int32)
    half = size // 2
    y1 = tf.clip_by_value(cy - half, 0, h)
    y2 = tf.clip_by_value(cy + half, 0, h)
    x1 = tf.clip_by_value(cx - half, 0, w)
    x2 = tf.clip_by_value(cx + half, 0, w)

    # build rectangular mask (0 in the box, 1 elsewhere)
    rect = tf.pad(
        tf.zeros([y2 - y1, x2 - x1, 3], dtype=img.dtype),
        paddings=[[y1, h - y2], [x1, w - x2], [0, 0]],
        constant_values=0.0
    )
    mask = 1.0 - rect
    return img * mask

@tf.function
def apply_cutout_batch(images, p=0.5, size=8):
    # map per-image cutout, then randomly choose to keep/appply per sample
    cut_imgs = tf.map_fn(lambda im: cutout_image(im, size=size),
                         images, fn_output_signature=tf.float32)
    b = tf.less(tf.random.uniform([tf.shape(images)[0], 1, 1, 1]), p)
    return tf.where(b, cut_imgs, images)

@tf.function
def cutmix_batch(images, labels, alpha=1.0, p=0.5):
    # CutMix with one-hot labels; kept unchanged, robust path
    if tf.random.uniform(()) > p:
        return images, labels
    b = tf.shape(images)[0]
    idx = tf.random.shuffle(tf.range(b))
    imgs2 = tf.gather(images, idx)
    labs2 = tf.gather(labels, idx)

    # sample area via lambda -> box size
    lam = tf.random.uniform([], 0.3, 0.7)  # stable range
    H = tf.shape(images)[1]; W = tf.shape(images)[2]
    r  = tf.sqrt(1.0 - lam)
    rw = tf.cast(r * tf.cast(W, tf.float32), tf.int32)
    rh = tf.cast(r * tf.cast(H, tf.float32), tf.int32)
    rx = tf.random.uniform([], 0, W, dtype=tf.int32)
    ry = tf.random.uniform([], 0, H, dtype=tf.int32)

    x1 = tf.clip_by_value(rx - rw // 2, 0, W)
    y1 = tf.clip_by_value(ry - rh // 2, 0, H)
    x2 = tf.clip_by_value(rx + rw // 2, 0, W)
    y2 = tf.clip_by_value(ry + rh // 2, 0, H)

    rect = tf.pad(
        tf.ones([y2 - y1, x2 - x1, 3], dtype=images.dtype),
        paddings=[[y1, H - y2], [x1, W - x2], [0, 0]],
        constant_values=0.0
    )
    mixed = images * (1.0 - rect) + imgs2 * rect

    area = tf.cast((x2 - x1) * (y2 - y1), tf.float32)
    lam_adj = 1.0 - area / tf.cast(H * W, tf.float32)
    labs = lam_adj * labels + (1.0 - lam_adj) * labs2
    return mixed, labs

print("Augmentations ready (per-image CutOut; robust CutMix).")



# ==== Cell 4: tf.data pipelines ====
# One-hot labels up front (required for CutMix / categorical loss)

y_train_oh = tf.one_hot(y_train, NUM_CLASSES)
y_val_oh   = tf.one_hot(y_val,   NUM_CLASSES)
y_test_oh  = tf.one_hot(y_test,  NUM_CLASSES)

USE_CUTMIX  = True   # set False if you want simpler run
CUTMIX_P    = 0.5
USE_CUTOUT  = True
CUTOUT_P    = 0.5
CUTOUT_SIZE = 8

def aug_batch(images, labels):
    # per-image crop+flip
    images = tf.map_fn(random_crop_pad_flip, images, fn_output_signature=tf.float32)
    # optional CutOut
    if USE_CUTOUT:
        images = apply_cutout_batch(images, p=CUTOUT_P, size=CUTOUT_SIZE)
    # optional CutMix
    if USE_CUTMIX:
        images, labels = cutmix_batch(images, labels, alpha=1.0, p=CUTMIX_P)
    return images, labels

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train_oh))
train_ds = train_ds.shuffle(50000, seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.map(aug_batch, num_parallel_calls=AUTO).prefetch(AUTO)

val_ds   = tf.data.Dataset.from_tensor_slices((x_val, y_val_oh))  \
           .batch(BATCH_SIZE).prefetch(AUTO)
test_ds_official = tf.data.Dataset.from_tensor_slices((x_test, y_test_oh)) \
           .batch(BATCH_SIZE).prefetch(AUTO)

train_steps = int(tf.data.experimental.cardinality(train_ds).numpy())
val_steps   = int(tf.data.experimental.cardinality(val_ds).numpy())
print("train_steps:", train_steps, "| val_steps:", val_steps)



# ==== Cell 5: model (custom residual CNN, no pretrained weights) ====
from tensorflow.keras import layers, regularizers, Model

WEIGHT_DECAY = 5e-4  # classic CIFAR-10 L2

def conv3x3(x, filters, stride=1):
    return layers.Conv2D(
        filters, 3, strides=stride, padding="same",
        use_bias=False, kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )(x)

def basic_block(x, filters, stride):
    # Pre-activation basic block
    shortcut = x
    y = layers.BatchNormalization()(x)
    y = layers.ReLU()(y)
    y = conv3x3(y, filters, stride=stride)

    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = conv3x3(y, filters, stride=1)

    if (shortcut.shape[-1] != filters) or (stride != 1):
        shortcut = layers.Conv2D(
            filters, 1, strides=stride, padding="same", use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
        )(shortcut)
    out = layers.Add()([shortcut, y])
    return out

def make_stage(x, filters, blocks, first_stride):
    x = basic_block(x, filters, first_stride)
    for _ in range(blocks - 1):
        x = basic_block(x, filters, 1)
    return x

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(
        64, 3, padding="same", use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )(inp)

    # Stages (compact but strong)
    x = make_stage(x,  72, blocks=2, first_stride=1)  # 32x32
    x = make_stage(x, 112, blocks=2, first_stride=2)  # 16x16
    x = make_stage(x, 160, blocks=3, first_stride=2)  # 8x8
    x = make_stage(x, 256, blocks=2, first_stride=2)  # 4x4

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(num_classes, activation="softmax",
                       kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    return Model(inp, out, name="MiniResNet_A6")

model = build_model()
model.summary()



# ==== Cell 6: optimizer & compile (AdamW + warmup-cosine, serializable) ====

import numpy as np
from tensorflow import keras

# steps/epoch from your tf.data pipelines
train_steps = int(tf.data.experimental.cardinality(train_ds).numpy())

EPOCHS       = 200
TOTAL_STEPS  = train_steps * EPOCHS
WARMUP_STEPS = int(0.10 * TOTAL_STEPS)   # 10% warmup

class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps, min_lr=1e-5):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # linear warmup
        warm = self.base_lr * (step / tf.maximum(1.0, self.warmup_steps))
        # cosine decay to min_lr
        progress = (step - self.warmup_steps) / tf.maximum(1.0, self.total_steps - self.warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cos = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + tf.cos(np.pi * progress))
        return tf.where(step < self.warmup_steps, warm, cos)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }

# rebuild the schedule & optimizer
lr_schedule = WarmupCosine(
    base_lr=3e-4,
    warmup_steps=WARMUP_STEPS,
    total_steps=TOTAL_STEPS,
    min_lr=1e-5,
)
optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=5e-4)

# one-hot labels â†’ categorical loss with smoothing (stable with CutMix)
loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
print("Recompiled with serializable WarmupCosine. Ready to train.")



# ==== Cell 7: train & evaluate ====
ckpt_path = "/kaggle/working/mini_resnet_best.keras"

callbacks = [
    keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=20, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.CSVLogger("/kaggle/working/train_log.csv", append=False),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# quick sanity on official CIFAR-10 test split
test_loss, test_acc = model.evaluate(test_ds_official, verbose=0)
print(f"[Official CIFAR-10 test] accuracy={test_acc:.4f}  loss={test_loss:.4f}")
print("Best model saved at:", ckpt_path)



# ==== STEP 8: Build submission.csv for CIFAR-10 Challenge ====
import os, numpy as np, pandas as pd, tensorflow as tf

# Paths
DATA_DIR = "/kaggle/input/cifar-10"
WORK_DIR = "/kaggle/working"
MODEL_PATH = "/kaggle/input/rashid/mini_resnet_best.keras"
SAMPLE_SUB = os.path.join(DATA_DIR, "sampleSubmission.csv")
TEST_7Z = os.path.join(DATA_DIR, "test.7z")
TEST_DIR = os.path.join(WORK_DIR, "test")

# --- Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"â�Œ Model not found at {MODEL_PATH}. Please attach it under 'Add input'.")
print(f"âœ… Model found at: {MODEL_PATH}")

# --- Load trained model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("âœ… Model loaded successfully.")

# --- Extract test.7z only if not extracted
if not os.path.exists(TEST_DIR) or len(os.listdir(TEST_DIR)) == 0:
    print("ğŸ“¦ Extracting test.7z ...")
    os.system(f"7z x {TEST_7Z} -o{WORK_DIR}/test -y > /dev/null")
else:
    print("âœ… Test images already extracted.")

# --- Gather all .png files
import glob
pngs = sorted(glob.glob(os.path.join(TEST_DIR, "**", "*.png"), recursive=True))
print(f"ğŸ“¸ Found {len(pngs)} test images.")

if len(pngs) == 0:
    raise FileNotFoundError("No test images found. Please verify test.7z extraction.")

# --- Define loading and preprocessing
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [32, 32])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = (img - 0.5) * 2.0  # normalize [-1,1]
    return img

BATCH = 512
test_ds = (tf.data.Dataset.from_tensor_slices(pngs)
            .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH)
            .prefetch(tf.data.AUTOTUNE))

# --- Predict
probs = model.predict(test_ds, verbose=1)
pred_idx = np.argmax(probs, axis=1)

# --- Create submission
class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
pred_labels = [class_names[i] for i in pred_idx]

ids = [int(os.path.basename(p).split(".")[0]) for p in pngs]
submission = pd.DataFrame({"id": ids, "label": pred_labels})
submission = submission.sort_values("id")
save_path = os.path.join(WORK_DIR, "submission.csv")
submission.to_csv(save_path, index=False)

print(f"\nâœ… Submission file saved to: {save_path}")
print(submission.head())


