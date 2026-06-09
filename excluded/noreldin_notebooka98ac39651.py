!pip install py7zr
import os, glob
import numpy as np
import pandas as pd
from PIL import Image
import py7zr

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Path to the uploaded CIFAR10 7z dataset
DATA_PATH = "/kaggle/input/cifar-10/train.7z"   # adjust if different filename
TEST_PATH = "/kaggle/input/cifar-10/test.7z"    # adjust if different filename



# Extract train and test archives
os.makedirs("/kaggle/working/cifar10/train", exist_ok=True)
os.makedirs("/kaggle/working/cifar10/test", exist_ok=True)

print("Extracting train...")
with py7zr.SevenZipFile(DATA_PATH, mode='r') as archive:
    archive.extractall(path="/kaggle/working/cifar10/train")

print("Extracting test...")
with py7zr.SevenZipFile(TEST_PATH, mode='r') as archive:
    archive.extractall(path="/kaggle/working/cifar10/test")

print("Extraction complete!")



# Training data: usually in /train/{class_name}/{images}
train_dir = "/kaggle/working/cifar10/train"
test_dir  = "/kaggle/working/cifar10/test/test"

# Build class map
classes = sorted(os.listdir(train_dir))
class_to_idx = {c: i for i, c in enumerate(classes)}
print("Classes:", classes)

X_train, y_train = [], []
for c in classes:
    folder = os.path.join(train_dir, c)
    for img_file in glob.glob(os.path.join(folder, "*.png")):
        img = Image.open(img_file).convert("RGB").resize((32,32))
        X_train.append(np.array(img))
        y_train.append(class_to_idx[c])

X_train = np.array(X_train, dtype="float32") / 255.0
y_train = np.array(y_train, dtype="int32")

print("Train shape:", X_train.shape, y_train.shape)

# Test data
X_test, test_ids = [], []
for img_file in sorted(glob.glob(os.path.join(test_dir, "*.png"))):
    img_id = os.path.basename(img_file).split(".")[0]
    img = Image.open(img_file).convert("RGB").resize((32,32))
    X_test.append(np.array(img))
    test_ids.append(img_id)

X_test = np.array(X_test, dtype="float32") / 255.0
print("Test shape:", X_test.shape)



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
)

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)



import os, math, time, sys, json, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# Reproducibility (still allows GPU nondeterminism in some ops)
SEED = 1337
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)  # keep as-is for CIFAR-10
BATCH_SIZE = 256
EPOCHS = 100

# Safety: make sure memory growth is enabled (prevents OOM on some Kaggle GPUs)
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass




SEED = 1337
AUTO = tf.data.AUTOTUNE
NUM_CLASSES_TF = tf.constant(10, dtype=tf.int32)  # adjust if needed

basic_augment = keras.Sequential([
    layers.RandomFlip("horizontal", seed=SEED),
    layers.RandomTranslation(0.1, 0.1, seed=SEED),
    layers.RandomZoom(0.1, seed=SEED),
], name="basic_augment")

def ensure_one_hot_float(labels):
    """Accepts [B] ints or [B,C] one-hots (int/float). Returns float32 one-hot [B,C]."""
    rank = tf.rank(labels)
    def _from_indices():
        idx = tf.cast(labels, tf.int32)              # [B]
        return tf.one_hot(idx, NUM_CLASSES_TF)       # float32 [B,C]
    def _from_onehot():
        return tf.cast(labels, tf.float32)           # float32 [B,C]
    return tf.cond(tf.equal(rank, 1), _from_indices, _from_onehot)

def _rand_bbox(height, width, lam):
    height = tf.cast(height, tf.float32)
    width  = tf.cast(width,  tf.float32)
    cut_rat = tf.sqrt(1.0 - lam)                     # float32
    cut_h = tf.cast(height * cut_rat, tf.int32)
    cut_w = tf.cast(width  * cut_rat, tf.int32)
    cy = tf.random.uniform([], 0, tf.cast(height, tf.int32), dtype=tf.int32)
    cx = tf.random.uniform([], 0, tf.cast(width,  tf.int32), dtype=tf.int32)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, tf.cast(height, tf.int32))
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, tf.cast(height, tf.int32))
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, tf.cast(width,  tf.int32))
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, tf.cast(width,  tf.int32))
    return y1, x1, y2, x2

@tf.function
def cutmix(images, labels, alpha=1.0):
    """Graph-safe CutMix (expects images float32 [B,H,W,C], labels one-hot float32 [B,C])."""
    images = tf.cast(images, tf.float32)
    labels = ensure_one_hot_float(labels)

    B = tf.shape(images)[0]
    H = tf.shape(images)[1]
    W = tf.shape(images)[2]

    idx = tf.random.shuffle(tf.range(B))
    images_shuf = tf.gather(images, idx)
    labels_shuf = tf.gather(labels, idx)

    g1 = tf.random.gamma([1], alpha=alpha, dtype=tf.float32)
    g2 = tf.random.gamma([1], alpha=alpha, dtype=tf.float32)
    lam = tf.squeeze(g1 / (g1 + g2))
    lam = tf.clip_by_value(lam, 0.0, 1.0)

    y1, x1, y2, x2 = _rand_bbox(H, W, lam)
    box_h = tf.maximum(y2 - y1, 0)
    box_w = tf.maximum(x2 - x1, 0)

    def mix_with_box():
        M_core = tf.ones([B, box_h, box_w, 1], dtype=images.dtype)
        M = tf.pad(M_core, [[0,0],[y1, H - y2],[x1, W - x2],[0,0]])
        mixed = images * (1.0 - M) + images_shuf * M
        box_area = tf.cast(box_h * box_w, tf.float32)
        lam_adj = 1.0 - box_area / tf.cast(H * W, tf.float32)
        new_labels = lam_adj * labels + (1.0 - lam_adj) * labels_shuf
        return mixed, new_labels

    return tf.cond(tf.logical_and(box_h > 0, box_w > 0), mix_with_box, lambda: (images, labels))

@tf.function
def cutout_single(image, mask_size=12):
    image = tf.cast(image, tf.float32)
    H = tf.shape(image)[0]; W = tf.shape(image)[1]
    cy = tf.random.uniform([], 0, H, dtype=tf.int32)
    cx = tf.random.uniform([], 0, W, dtype=tf.int32)
    y1 = tf.clip_by_value(cy - mask_size // 2, 0, H)
    y2 = tf.clip_by_value(cy + mask_size // 2, 0, H)
    x1 = tf.clip_by_value(cx - mask_size // 2, 0, W)
    x2 = tf.clip_by_value(cx + mask_size // 2, 0, W)
    box_h = tf.maximum(y2 - y1, 0); box_w = tf.maximum(x2 - x1, 0)

    def erase():
        rect = tf.ones([box_h, box_w, tf.shape(image)[-1]], dtype=image.dtype)
        rect = tf.pad(rect, [[y1, H - y2],[x1, W - x2],[0,0]])
        return image * (1.0 - rect)

    return tf.cond(tf.logical_and(box_h > 0, box_w > 0), erase, lambda: image)

@tf.function
def apply_strong_aug(images, labels, p_cutmix=0.5, p_cutout=0.5):
    images = tf.cast(images, tf.float32)
    labels = ensure_one_hot_float(labels)

    images = basic_augment(images, training=True)

    rnd = tf.random.uniform([], 0, 1)
    def _do_cutmix():
        return cutmix(images, labels)

    def _maybe_cutout():
        B = tf.shape(images)[0]
        flags = tf.random.uniform([B], 0, 1) < p_cutout
        def _one(img, flag):
            return tf.cond(flag, lambda: cutout_single(img), lambda: tf.cast(img, tf.float32))
        imgs = tf.map_fn(lambda x: _one(x[0], x[1]),
                         (images, flags),
                         fn_output_signature=tf.float32)
        return imgs, labels

    return tf.cond(rnd < p_cutmix, _do_cutmix, _maybe_cutout)



def make_ds(X, y, batch_size, training=True):
    # Convert labels to one-hot encoding first
    y_one_hot = tf.one_hot(y, depth=NUM_CLASSES)
    
    ds = tf.data.Dataset.from_tensor_slices((X, y_one_hot))
    if training:
        ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=training)
    if training:
        # Apply augmentation and SET THE OUTPUT SIGNATURE explicitly
        ds = ds.map(
            lambda x, y: apply_strong_aug(x, y), 
            num_parallel_calls=AUTO
        )
        # Fix: Set the shape explicitly after augmentation
        def set_shapes(images, labels):
            images.set_shape([batch_size, 32, 32, 3])
            labels.set_shape([batch_size, NUM_CLASSES])
            return images, labels
        ds = ds.map(set_shapes, num_parallel_calls=AUTO)
    else:
        # For validation, also set shapes
        def set_shapes_val(images, labels):
            images.set_shape([None, 32, 32, 3])
            labels.set_shape([None, NUM_CLASSES])
            return images, labels
        ds = ds.map(set_shapes_val, num_parallel_calls=AUTO)
    
    ds = ds.prefetch(AUTO)
    return ds

# Rebuild datasets with the fixed pipeline
train_ds = make_ds(X_train, y_train, BATCH_SIZE, training=True)
val_ds   = make_ds(X_val,   y_val,   BATCH_SIZE, training=False)


def se_block(x, se_ratio=0.25):
    in_ch = x.shape[-1]
    squeeze = layers.GlobalAveragePooling2D()(x)
    squeeze = layers.Dense(int(in_ch * se_ratio), activation="relu")(squeeze)
    excite  = layers.Dense(in_ch, activation="sigmoid")(squeeze)
    excite  = layers.Reshape((1,1,in_ch))(excite)
    return layers.Multiply()([x, excite])

def DSConv(x, filters, kernel_size=3, stride=1, drop=0.0):
    # Depthwise-Separable Conv block with BN+GELU+SE
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)

    x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)

    x = se_block(x, se_ratio=0.25)
    if drop > 0:
        x = layers.Dropout(drop)(x)
    return x

def Stem(x, filters=64):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    return x

def downsample(x, filters):
    x = layers.Conv2D(filters, 3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    return x

def build_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, drop=0.1):
    inp = layers.Input(shape=input_shape)

    # Optional per-pixel normalization layer (leave if your inputs are [0,1])
    x = layers.Rescaling(1.0, offset=0.0)(inp)  # no change; acts as identity

    x = Stem(x, 64)             # 32x32
    x = DSConv(x,  96, 3, 1, drop=drop)
    x = DSConv(x,  96, 3, 1, drop=drop)

    x = downsample(x, 160)      # 16x16
    x = DSConv(x, 160, 3, 1, drop=drop)
    x = DSConv(x, 160, 3, 1, drop=drop)

    x = downsample(x, 256)      # 8x8
    x = DSConv(x, 256, 3, 1, drop=drop)
    x = DSConv(x, 256, 3, 1, drop=drop)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inp, out, name="CustomCIFAR10_DSConv_SE")
    return model

model = build_cnn()
model.summary()



'''class WarmupCosineSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps=0, min_lr=1e-5):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Warmup
        if self.warmup_steps > 0:
            warmup_lr = self.base_lr * (step / tf.maximum(1.0, self.warmup_steps))
        else:
            warmup_lr = self.base_lr

        # Cosine decay (after warmup)
        progress = tf.minimum(1.0, tf.maximum(0.0, (step - self.warmup_steps) / tf.maximum(1.0, self.total_steps - self.warmup_steps)))
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + tf.cos(math.pi * progress))

        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: cosine_lr)

# Steps/epoch based on training dataset
steps_per_epoch = int(np.ceil(len(X_train) / BATCH_SIZE))
total_steps = steps_per_epoch * EPOCHS
warmup_ratio = 0.1  # 10% warmup
warmup_steps = int(total_steps * warmup_ratio)

BASE_LR = 3e-3     # good starting point for AdamW on CIFAR-10 with batch 256
MIN_LR  = 3e-5

lr_schedule = WarmupCosineSchedule(base_lr=BASE_LR, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=MIN_LR)
optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)'''



import tensorflow as tf
from keras import ops
from keras.optimizers.schedules import LearningRateSchedule
from keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
class WarmupCosineSchedule(LearningRateSchedule):
    """
    Linear warmup to base_lr over `warmup_steps`, then cosine decay to `min_lr`
    by `total_steps`. All args are Python scalars (ints/floats).
    """

    def __init__(self, base_lr, total_steps, warmup_steps=0, min_lr=0.0, name=None):
        super().__init__()
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if warmup_steps < 0 or warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps)")
        if min_lr < 0:
            raise ValueError("min_lr must be >= 0")

        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.name = name or "WarmupCosineSchedule"

        # Precompute the cosine portion length
        self._decay_steps = self.total_steps - self.warmup_steps

    def __call__(self, step):
        step = ops.cast(step, "float32")
        warmup_steps = ops.cast(self.warmup_steps, "float32")
        decay_steps  = ops.cast(self._decay_steps, "float32")

        # Linear warmup: 0 -> base_lr
        def warmup():
            # avoid div by zero if warmup_steps == 0
            denom = ops.maximum(warmup_steps, 1.0)
            return (step / denom) * self.base_lr

        # Cosine decay: base_lr -> min_lr
        def cosine():
            # progress from 0..1 over decay steps
            t = ops.minimum(step - warmup_steps, decay_steps) / ops.maximum(decay_steps, 1.0)
            cosine_decay = 0.5 * (1.0 + ops.cos(ops.constant(3.141592653589793) * t))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

        return ops.where(step < warmup_steps, warmup(), cosine())

    def get_config(self):
        # Must return JSON-serializable types
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)



import tensorflow as tf
import keras

# --- Training config ---
EPOCHS = 100                         # adjust if needed
BASE_LR = 3e-4                       # peak LR during/after warmup
MIN_LR  = 1e-6                       # floor LR at end of cosine
WARMUP_FRACTION = 0.10               # first 10% of steps are warmup
WEIGHT_DECAY = 1e-4

# --- Infer steps/total_steps from the dataset ---
card = tf.data.experimental.cardinality(train_ds).numpy()
steps_per_epoch = int(card) if card > 0 else None
if steps_per_epoch is None:
    raise ValueError(
        "train_ds must have a finite cardinality. "
        "Make sure it's batched and not infinite/repeat()."
    )
total_steps  = steps_per_epoch * EPOCHS
warmup_steps = max(1, int(WARMUP_FRACTION * total_steps))

# --- Learning-rate schedule (your custom class must be defined above) ---
lr_schedule = WarmupCosineSchedule(
    base_lr=BASE_LR,
    total_steps=total_steps,
    warmup_steps=warmup_steps,
    min_lr=MIN_LR,
)

# --- Optimizer & compile ---
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=WEIGHT_DECAY,
)
# Use 'categorical_crossentropy' if your labels are one-hot; switch to 'sparse_categorical_crossentropy' if labels are int-encoded.
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# --- Callbacks ---
ckpt = keras.callbacks.ModelCheckpoint(
    filepath="best_weights.h5",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=True,  # faster & avoids full optimizer serialization in experiments
    verbose=1,
)
early = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=10,
    restore_best_weights=True,
    verbose=1,
)

# (Optional) quick LR logger each epoch
def _lr_for_epoch(epoch, logs):
    # current global step = (seen samples / batch), but we can query the optimizer variable directly:
    opt_step = optimizer.iterations.numpy()
    # Beware: querying inside the graph may differ; this is for a rough log.
    # Evaluate the schedule at the current step:
    current_lr = float(lr_schedule(opt_step).numpy())
    print(f"\n[LR] step {opt_step} -> lr={current_lr:.6g}")

lr_logger = keras.callbacks.LambdaCallback(on_epoch_end=_lr_for_epoch)

# --- Train ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[ckpt, early, lr_logger],
)


