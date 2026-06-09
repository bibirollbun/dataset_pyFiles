import os, glob, numpy as np, pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import Callback

# ========= Repro & GPU info =========
np.random.seed(42)
tf.random.set_seed(42)
print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


# ============================================================================
# LOAD & PREPARE DATA â€” simple /255 normalization (no standard deviation)
# ============================================================================
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to [0,1]
x_train = x_train.astype("float32") / 255.0
x_val   = x_val.astype("float32") / 255.0

# One-hot encode labels for categorical crossentropy
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_val   = tf.keras.utils.to_categorical(y_val, 10)

print("x_train:", x_train.shape, "y_train:", y_train.shape)
print("x_val:", x_val.shape, "y_val:", y_val.shape)


# ============================================================================
# CIFAR-10 AUGMENTATION + MIXUP (robust & dtype-safe)
# ============================================================================

import tensorflow as tf
from tensorflow.keras import layers

AUTO  = tf.data.AUTOTUNE
BATCH = 128

# ----------------------------
# MixUp
# ----------------------------
def mixup_batch(x, y, alpha=0.2):
    """Batch-wise MixUp (uniform-symmetric lambda; dtype-safe)."""
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    # Symmetric lambda in [0.5, 1.0] for balanced mixing
    lam = tf.random.uniform([], 0.0, 1.0, dtype=x.dtype)
    lam = tf.maximum(lam, 1.0 - lam)

    idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))
    x2 = tf.gather(x, idx)
    y2 = tf.gather(y, idx)

    x_m = lam * x + (1.0 - lam) * x2
    y_m = lam * y + (1.0 - lam) * y2
    return x_m, y_m

def cutmix_batch(x, y, alpha=1.0):
    """Applies CutMix to a batch (dtype-safe)."""
    batch_size = tf.shape(x)[0]
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]

    lam = tf.random.uniform([], 0, 1, dtype=tf.float32)
    rx = tf.random.uniform([], 0, tf.cast(W, tf.float32))
    ry = tf.random.uniform([], 0, tf.cast(H, tf.float32))
    rw = tf.cast(tf.cast(W, tf.float32) * tf.math.sqrt(1. - lam), tf.int32)
    rh = tf.cast(tf.cast(H, tf.float32) * tf.math.sqrt(1. - lam), tf.int32)

    x1 = tf.clip_by_value(tf.cast(rx, tf.int32) - rw // 2, 0, W)
    y1 = tf.clip_by_value(tf.cast(ry, tf.int32) - rh // 2, 0, H)
    x2 = tf.clip_by_value(tf.cast(rx, tf.int32) + rw // 2, 0, W)
    y2 = tf.clip_by_value(tf.cast(ry, tf.int32) + rh // 2, 0, H)

    indices = tf.random.shuffle(tf.range(batch_size))
    x2_batch = tf.gather(x, indices)
    y2_batch = tf.gather(y, indices)

    mask = tf.ones((y2 - y1, x2 - x1, 3), dtype=tf.float32)
    pad = [[y1, H - y2], [x1, W - x2], [0, 0]]
    mask = tf.pad(mask, pad, constant_values=0.)
    inv_mask = 1. - mask

    mixed_x = x * inv_mask + x2_batch * mask
    lam = 1. - tf.cast((x2 - x1) * (y2 - y1), tf.float32) / tf.cast(W * H, tf.float32)
    mixed_y = lam * y + (1. - lam) * y2_batch
    return mixed_x, mixed_y


# ----------------------------
# Advanced Augmentations (+Cutout via tf.pad)
# ----------------------------
class AdvancedAugmentation(layers.Layer):
    def __init__(self, mask_size=8, **kwargs):
        super().__init__(**kwargs)
        self.mask_size = mask_size
        self.flip        = layers.RandomFlip("horizontal")
        self.rotation    = layers.RandomRotation(0.10)
        self.zoom        = layers.RandomZoom(0.10)
        self.translation = layers.RandomTranslation(0.10, 0.10)
        self.contrast    = layers.RandomContrast(0.15)

    def cutout_one(self, image):
        """
        Create a (H,W,1) mask with a zeroed square hole and multiply.
        Works with dynamic shapes and broadcasts over channels.
        """
        image = tf.convert_to_tensor(image)
        image = tf.cast(image, tf.float32)

        H = tf.shape(image)[0]
        W = tf.shape(image)[1]
        # Use 1-channel mask and broadcast to 3
        ms = tf.minimum(tf.minimum(self.mask_size, H), W)

        # random top-left for the hole (inclusive end -> +1)
        top  = tf.random.uniform([], 0, H - ms + 1, dtype=tf.int32)
        left = tf.random.uniform([], 0, W - ms + 1, dtype=tf.int32)

        # hole of zeros, pad with ones around
        hole = tf.zeros([ms, ms, 1], dtype=image.dtype)
        pad  = [
            [top,  H - top  - ms],
            [left, W - left - ms],
            [0, 0]
        ]
        mask = tf.pad(hole, pad, constant_values=1.0)   # (H,W,1), ones with a zero hole
        return image * mask                              # broadcast over channels

    def call(self, images, training=None):
        images = tf.cast(images, tf.float32)
        if training:
            images = self.flip(images)
            images = self.rotation(images)
            images = self.zoom(images)
            images = self.translation(images)
            images = self.contrast(images)

            # 50% of batches get Cutout; map_fn keeps shapes intact
            def apply_cutout():
                return tf.map_fn(self.cutout_one, images, fn_output_signature=tf.float32)
            images = tf.cond(tf.random.uniform([]) < 0.5, apply_cutout, lambda: images)
        return images

# ----------------------------
# Datasets
# ----------------------------
augmenter = AdvancedAugmentation(mask_size=8)

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(50_000, seed=42, reshuffle_each_iteration=True)
    .batch(BATCH, drop_remainder=True)
    .map(lambda xb, yb: (augmenter(xb, training=True), yb), num_parallel_calls=AUTO)
    .map(lambda xb, yb: mixup_batch(xb, yb, alpha=0.2), num_parallel_calls=AUTO)
    .map(
        # 50% of batches use CutMix instead of MixUp
        lambda xb, yb: tf.cond(
            tf.random.uniform([]) < 0.5,
            lambda: cutmix_batch(xb, yb, alpha=1.0),
            lambda: (xb, yb),
        ),
        num_parallel_calls=AUTO,
    )
    .prefetch(AUTO)
)


# train_ds = (
#     tf.data.Dataset.from_tensor_slices((x_train, y_train))
#     .shuffle(50_000, seed=42, reshuffle_each_iteration=True)
#     .batch(BATCH, drop_remainder=True)
#     .map(lambda xb, yb: (augmenter(xb, training=True), yb), num_parallel_calls=AUTO)
#     .map(lambda xb, yb: mixup_batch(xb, yb, alpha=0.2), num_parallel_calls=AUTO)
#     .prefetch(AUTO)
# )

val_ds = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .batch(BATCH)
    .prefetch(AUTO)
)

print("âœ… Augmentation + MixUp pipelines are ready.")



def channel_attention(x, reduction=8):
    c = x.shape[-1]
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    d1 = layers.Dense(c // reduction, activation="relu")
    d2 = layers.Dense(c, activation="sigmoid")
    a1 = d2(d1(gap))
    a2 = d2(d1(gmp))
    attn = layers.Add()([a1, a2])
    attn = layers.Reshape((1,1,c))(attn)
    return layers.Multiply()([x, attn])

def spatial_attention(x):
    avg = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    mx  = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate()([avg, mx])
    attn = layers.Conv2D(1, 7, padding="same", activation="sigmoid")(concat)
    return layers.Multiply()([x, attn])

class DropPath(layers.Layer):
    def __init__(self, survival_prob=0.9):
        super().__init__()
        self.survival_prob = survival_prob
    def call(self, x, training=None):
        if (not training) or self.survival_prob == 1.0:
            return x
        keep = tf.random.uniform([tf.shape(x)[0],1,1,1]) < self.survival_prob
        return tf.where(keep, x / self.survival_prob, tf.zeros_like(x))

def hybrid_residual_block(x, filters, stride=1, use_proj=False, use_attention=True, dp_survival=1.0):
    shortcut = x
    bneck = max(filters // 4, 16)

    # 1x1
    x = layers.Conv2D(bneck, 1, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    # 3x3
    x = layers.Conv2D(bneck, 3, strides=stride, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    # 1x1
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)

    if use_attention:
        x = channel_attention(x)
        x = spatial_attention(x)

    if use_proj or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False, kernel_initializer="he_normal")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = DropPath(dp_survival)(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

def se_block(x, ratio=16):
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(channels // ratio, activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, channels))(se)
    return layers.Multiply()([x, se])
    
print("residual blocks and other model addition")


def build_model(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = AdvancedAugmentation()(inputs)

    # ---- Stem ----
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)

    # ---- Stage 1: residual blocks, 32Ã—32 ----
    for _ in range(3):
        shortcut = x
        x = layers.Conv2D(64, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(64, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
        x = se_block(x)

    # ---- Stage 2: two blocks, 16Ã—16 ----
    # Downsample first
    shortcut = layers.Conv2D(128, 1, strides=2, padding="same", use_bias=False)(x)
    shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Conv2D(128, 3, strides=2, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    x = se_block(x)

    # One more residual block (same size)
    shortcut = x
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    x = se_block(x)

    # ---- Stage 3: **Added block here â†’ makes it "ResNet-10"**, 8Ã—8 ----
    shortcut = layers.Conv2D(256, 1, strides=2, padding="same", use_bias=False)(x)
    shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Conv2D(256, 3, strides=2, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    x = se_block(x)

    # âœ… Extra residual block (the â€œ+1â€� youâ€™re adding)
    for _ in range(3):
        shortcut = x
        x = layers.Conv2D(256, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(256, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
        x = se_block(x)

    # ---- Stage 4: 512 filters, 4Ã—4 ----
    shortcut = layers.Conv2D(512, 1, strides=2, padding="same", use_bias=False)(x)
    shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Conv2D(512, 3, strides=2, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(512, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    x = se_block(x)

    # After the first Stage 4 block, add:
    for _ in range(2):
        shortcut = x
        x = layers.Conv2D(512, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(512, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
        x = se_block(x)


    # ---- Head ----
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", kernel_initializer="he_normal")(x)

    model = Model(inputs, outputs, name="ResNet10_Simple")
    return model


class CosineAnnealingWarmupRestarts(tf.keras.callbacks.Callback):
    def __init__(self, max_lr=1e-3, min_lr=1e-6, warmup_epochs=10, T_0=50, T_mult=2):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.cycle_length = T_0
        self.cycle = 0
        self.epoch_in_cycle = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.max_lr * (epoch + 1) / self.warmup_epochs
        else:
            # After warmup, use cosine with restarts
            self.epoch_in_cycle = (epoch - self.warmup_epochs) % self.cycle_length
            progress = self.epoch_in_cycle / self.cycle_length
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
            
            # Handle cycle restarts
            if self.epoch_in_cycle + 1 == self.cycle_length:
                self.cycle += 1
                self.cycle_length = int(self.T_0 * (self.T_mult ** self.cycle))
        
        # Fix: Use assign method for newer Keras/TensorFlow
        self.model.optimizer.learning_rate.assign(lr)
        print(f"Epoch {epoch+1}: LR = {lr:.6e}")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Fix: Use numpy() to get the value
        logs["lr"] = float(self.model.optimizer.learning_rate.numpy())


model = build_model()   # <-- safe to call outside any strategy
model.summary()

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=1e-3,
    weight_decay=5e-4,
    beta_1=0.9,
    beta_2=0.999
)

# labels are one-hot, so use CategoricalCrossentropy with smoothing
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])



EPOCHS = 300

# Learning rate scheduler
cosine = CosineAnnealingWarmupRestarts(
    max_lr=1e-3,
    min_lr=1e-6,
    warmup_epochs=10,
    T_0=50,
    T_mult=2
)

# Callbacks without EarlyStopping - will train full 300 epochs
callbacks = [
    cosine,
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.keras",  # Changed to .keras format (recommended)
        monitor="val_accuracy", 
        save_best_only=True, 
        verbose=1,
        mode='max'
    ),
]

# Train for full 300 epochs
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=128,
    callbacks=callbacks,
    verbose=1,
)

# ============================================
# âœ… Load and save best model weights for Kaggle
# ============================================
# Load the best weights that were saved during training
model.load_weights("best_model.keras")

# Save in different formats for compatibility
model.save_weights("best_model.weights.h5")
model.save("best_model.keras")  # Full model save

print(f"âœ… Training completed: {len(history.history['loss'])} epochs")
print(f"âœ… Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print("âœ… Saved best_model.keras and best_model.weights.h5 â€” ready for Kaggle submission")


# ================================================================
# âœ… Extract CIFAR-10 test images (from test.7z)
# ================================================================
!apt install -y libarchive-dev
!pip install -q libarchive-c py7zr   # âœ… fixed typo: "libarchiv" â†’ "libarchive-c"

import os, glob, py7zr

test_7z_path = "/kaggle/input/cifar-10/test.7z"   # adjust if dataset path differs

# Verify file existence
if not os.path.exists(test_7z_path):
    raise FileNotFoundError(f"âš ï¸� File not found: {test_7z_path}")

# Extraction directory
extract_dir = "./test"
os.makedirs(extract_dir, exist_ok=True)

print("ğŸ“¦ Extracting test.7z with py7zr...")

with py7zr.SevenZipFile(test_7z_path, mode='r') as archive:
    archive.extractall(path=extract_dir)

print("âœ… Extraction completed.")

# Quick check
png_files = glob.glob(os.path.join(extract_dir, "*.png"))
print(f"ğŸ–¼ï¸� Found {len(png_files)} PNG test images in '{extract_dir}'")



import os, shutil, glob

src = "./test/test"
dst = "./test"

if os.path.exists(src):
    files = glob.glob(os.path.join(src, "*.png"))
    print(f"ğŸ“¦ Moving {len(files)} files from nested folder to top-level test/ ...")

    os.makedirs(dst, exist_ok=True)
    moved = 0
    for f in files:
        shutil.move(f, dst)
        moved += 1
        if moved % 5000 == 0:
            print(f"  Moved {moved} files...")

    print(f"âœ… Done! Total moved: {moved}")

    # Try removing now-empty subfolder
    try:
        os.rmdir(src)
        print("ğŸ§¹ Removed empty ./test/test/ folder.")
    except Exception as e:
        print(f"(info) Could not remove ./test/test/: {e}")
else:
    print("No nested folder found.")



import glob
print("âœ… Files now in ./test:", len(glob.glob("./test/*.png")))



# ============================================================================
# FINAL KAGGLE SUBMISSION CELL â€” /255 normalization only (no mean/std)
# ============================================================================
import os, glob, numpy as np, pandas as pd
from PIL import Image
import tensorflow as tf

# Load best weights (Keras 3 naming convention)
if os.path.exists("best_model.h5"):
    model.load_weights("best_model.weights.h5")
    print("âœ… Loaded best_model.weights.h5")
else:
    raise FileNotFoundError("â�Œ best_model.weights.h5 not found!")

model.trainable = False

# ============================================================================
# Load test images and preprocess (normalize only)
# ============================================================================
test_dir = "test"
test_files = sorted(
    glob.glob(os.path.join(test_dir, "*.png")),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
)
print(f"âœ… Found {len(test_files)} test images")

def load_batch(paths):
    """Load and normalize test images"""
    arr = np.zeros((len(paths), 32, 32, 3), dtype="float32")
    ids = np.zeros((len(paths),), dtype="int32")
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        arr[i] = np.asarray(img, dtype="float32") / 255.0   # âœ… Normalize only
        ids[i] = int(os.path.splitext(os.path.basename(p))[0])
    return arr, ids

# ============================================================================
# Predict in batches for efficiency
# ============================================================================
all_preds, all_ids = [], []
B = 5000  # process 5000 images per batch

for s in range(0, len(test_files), B):
    batch_files = test_files[s:s+B]
    xb, ib = load_batch(batch_files)
    pb = model.predict(xb, batch_size=256, verbose=1)
    all_preds.append(pb)
    all_ids.append(ib)

preds = np.concatenate(all_preds, axis=0)
test_ids = np.concatenate(all_ids, axis=0)
pred_classes = np.argmax(preds, axis=1)

# ============================================================================
# Sanity check: prediction distribution
# ============================================================================
vals, cnts = np.unique(pred_classes, return_counts=True)
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

print("\nğŸ“Š Prediction class histogram:")
for v, c in zip(vals, cnts):
    print(f"  {class_names[v]:12s}: {c:6d} ({c/len(pred_classes)*100:.1f}%)")

if len(vals) == 1:
    print("\nâš ï¸� WARNING: All predictions collapsed to ONE class!")
elif max(cnts) > len(pred_classes) * 0.4:
    print(f"\nâš ï¸� WARNING: Class '{class_names[vals[np.argmax(cnts)]]}' "
          f"has {max(cnts)/len(pred_classes)*100:.1f}% of predictions")

# ============================================================================
# Create submission.csv
# ============================================================================
# ============================================================================
# âœ… Create submission.csv with numeric labels
# ============================================================================
submission = pd.DataFrame({
    "id": test_ids,
    "label": [class_names[i] for i in pred_classes]
}).sort_values("id").reset_index(drop=True)

submission.to_csv("submission.csv", index=False)
print("\nâœ… submission.csv created successfully!")
print(submission.head(10))
print(f"\nTotal predictions: {len(submission)}")






