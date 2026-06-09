# DSEG660 Challenge â€” CIFAR-10 Custom Residual CNN 
# AdamW + Warmup-Cosine LR + CutMix & CutOut + Label Smoothing


import os, random, math, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW

print("TensorFlow version:", tf.__version__)

# --------------------------
# Reproducibility
# --------------------------
SEED = 1337
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# --------------------------
# Kaggle paths (for later submission step)
# --------------------------
DATA_DIR = "/kaggle/input/cifar10-object-recognition-in-images-zip-file"
TEST_DIR = os.path.join(DATA_DIR, "train_test")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sampleSubmission.csv")

# ============================================================
# 1) Load CIFAR-10 dataset (official Keras) for TRAIN/VAL
# ============================================================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

def normalize_img(x):
    # normalize to [-1, 1]
    return (x.astype("float32") / 255.0 - 0.5) * 2.0

x_train = normalize_img(x_train)
x_test  = normalize_img(x_test)
y_train = y_train.reshape(-1).astype("int32")
y_test  = y_test.reshape(-1).astype("int32")
print("Train:", x_train.shape, " Test:", x_test.shape)

# 5% validation split
n = x_train.shape[0]; n_val = int(0.05*n)
x_val, y_val = x_train[:n_val], y_train[:n_val]
x_tr , y_tr  = x_train[n_val:], y_train[n_val:]

# ============================================================
# 2) Model: Custom Residual CNN (modified ResNet-style)
# ============================================================
def conv_block(x, filters, stride=1):
    y = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4), use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    return y

def residual_block(x, filters, downsample=False):
    stride = 2 if downsample else 1
    y = conv_block(x, filters, stride)
    y = conv_block(y, filters)
    if downsample or x.shape[-1] != filters:
        x = layers.Conv2D(filters, 1, strides=stride, padding='same',
                          kernel_regularizer=regularizers.l2(1e-4), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
    out = layers.add([x, y])
    return layers.Activation('relu')(out)

def build_custom_resnet():
    inputs = layers.Input(shape=(32, 32, 3))
    x = conv_block(inputs, 64)

    # Stage 1: 3 blocks of 64
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # Stage 2: 3 blocks of 128
    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    # Stage 3: 2 blocks of 256
    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 256)

    # Stage 4: 2 blocks of 512
    x = residual_block(x, 512, downsample=True)
    x = residual_block(x, 512)

    # Head: GAP + custom dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)   # ğŸ”¹ extra dense layer
    x = layers.Dropout(0.4)(x)                    # ğŸ”¹ extra dropout
    outputs = layers.Dense(10, activation='softmax')(x)

    return models.Model(inputs, outputs)

model = build_custom_resnet()
model.summary()

# ============================================================
# 3) Advanced Augmentations: CutMix + CutOut (tf.data)
# ============================================================
IMG_SIZE = 32
NUM_CLASSES = 10
CUTMIX_PROB = 0.7
CUTOUT_PROB = 0.5
CUTOUT_SIZE = 8

@tf.function
def _beta(alpha=1.0):
    g1 = tf.random.gamma([], alpha, 1.0)
    g2 = tf.random.gamma([], alpha, 1.0)
    return g1/(g1+g2+1e-7)

def cutmix(images, labels, alpha=1.0):
    lam = _beta(alpha)
    bs = tf.shape(images)[0]
    idx = tf.random.shuffle(tf.range(bs))
    x2, y2 = tf.gather(images, idx), tf.gather(labels, idx)

    H, W = IMG_SIZE, IMG_SIZE
    rx = tf.random.uniform([], 0, W, dtype=tf.int32)
    ry = tf.random.uniform([], 0, H, dtype=tf.int32)
    cut = tf.cast(tf.sqrt(1. - lam) * tf.cast(W, tf.float32), tf.int32)
    x1 = tf.clip_by_value(rx - cut//2, 0, W); x2b = tf.clip_by_value(rx + cut//2, 0, W)
    y1 = tf.clip_by_value(ry - cut//2, 0, H); y2b = tf.clip_by_value(ry + cut//2, 0, H)

    mask = tf.ones((y2b-y1, x2b-x1, 3), images.dtype)
    mask = tf.image.pad_to_bounding_box(mask, y1, x1, H, W)
    mixed = images*(1.0 - mask) + x2*mask

    lam_adj = 1. - tf.cast((x2b-x1)*(y2b-y1), tf.float32)/tf.cast(H*W, tf.float32)
    y = lam_adj*tf.one_hot(labels, NUM_CLASSES) + (1. - lam_adj)*tf.one_hot(y2, NUM_CLASSES)
    return mixed, y

def cutout(images, size=CUTOUT_SIZE):
    def _one(img):
        cx = tf.random.uniform([], 0, IMG_SIZE, dtype=tf.int32)
        cy = tf.random.uniform([], 0, IMG_SIZE, dtype=tf.int32)
        x1 = tf.clip_by_value(cx - size//2, 0, IMG_SIZE); x2 = tf.clip_by_value(cx + size//2, 0, IMG_SIZE)
        y1 = tf.clip_by_value(cy - size//2, 0, IMG_SIZE); y2 = tf.clip_by_value(cy + size//2, 0, IMG_SIZE)
        m = tf.ones((y2-y1, x2-x1, 3), img.dtype)
        m = tf.image.pad_to_bounding_box(m, y1, x1, IMG_SIZE, IMG_SIZE)
        return img*(1.0 - m)
    return tf.map_fn(_one, images)

def augment_batch(images, labels):
    images = tf.image.random_flip_left_right(images)
    images = tf.image.resize_with_crop_or_pad(images, IMG_SIZE+4, IMG_SIZE+4)
    images = tf.image.random_crop(images, [tf.shape(images)[0], IMG_SIZE, IMG_SIZE, 3])
    do_cm = tf.less(tf.random.uniform([]), CUTMIX_PROB)
    images, labels = tf.cond(do_cm,
                             lambda: cutmix(images, labels),
                             lambda: (images, tf.one_hot(labels, NUM_CLASSES)))
    do_co = tf.less(tf.random.uniform([]), CUTOUT_PROB)
    images = tf.cond(do_co, lambda: cutout(images), lambda: images)
    return images, labels

def make_ds(xx, yy, train=True, batch=256, seed=SEED):
    ds = tf.data.Dataset.from_tensor_slices((xx, yy))
    if train:
        ds = ds.shuffle(8192, seed=seed).batch(batch).map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch).map(lambda a,b: (a, tf.one_hot(b, NUM_CLASSES)), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

# Build datasets
BATCH_SIZE = 256
train_ds = make_ds(x_tr, y_tr, True,  batch=BATCH_SIZE)
val_ds   = make_ds(x_val, y_val, False, batch=BATCH_SIZE)

# ============================================================
# 4) Optimizer + Warmup-Cosine LR + Label Smoothing (fixed)
# ============================================================
EPOCHS = 200
WARMUP_EPOCHS = 5
BASE_LR = 2e-3       # adjust if you change batch size
MIN_LR  = 5e-6
WEIGHT_DECAY = 1e-4

steps_per_epoch = math.ceil(x_tr.shape[0] / BATCH_SIZE)

class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_epochs, total_epochs, steps_per_epoch, min_lr=0.0):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps  = total_epochs  * steps_per_epoch
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.base_lr * (step / tf.maximum(1., tf.cast(self.warmup_steps, tf.float32)))
        prog = tf.clip_by_value((step - self.warmup_steps) /
                                tf.maximum(1., self.total_steps - self.warmup_steps), 0., 1.)
        cos  = self.min_lr + 0.5*(self.base_lr - self.min_lr)*(1. + tf.cos(math.pi*prog))
        return tf.where(step < self.warmup_steps, warm, cos)

    def get_config(self):  # ğŸ”¹ needed for TF 2.18+ serialization
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }

# Instantiate schedule + optimizer
lr_schedule = WarmupCosine(BASE_LR, WARMUP_EPOCHS, EPOCHS, steps_per_epoch, MIN_LR)
optimizer   = AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)

# Compile model with label smoothing
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

# ============================================================
# 5) Train (checkpoint best val_accuracy)
# ============================================================
ckpt = ModelCheckpoint("best.keras", monitor="val_accuracy", mode="max",
                       save_best_only=True, verbose=1)
early = EarlyStopping(monitor="val_accuracy", mode="max",
                      patience=30, restore_best_weights=True, verbose=1)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[ckpt, early],
    verbose=2
)

# Evaluate validation accuracy
val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"âœ… Val accuracy (held-out 5%): {val_acc:.4f} | Val loss: {val_loss:.4f}")

# ============================================================
# 6) Load BEST model for inference
# ============================================================
best_model = tf.keras.models.load_model("best.keras", compile=False)
print("âœ… Loaded best checkpoint.")



# === CIFAR-10 | Build /kaggle/working/submission.csv from best.keras ===
import os, glob, numpy as np, pandas as pd, tensorflow as tf, subprocess, shutil

# --- Paths (competition mount) ---
DATA_DIR   = "/kaggle/input/cifar-10"
TEST_7Z    = os.path.join(DATA_DIR, "test.7z")
WORK_DIR   = "/kaggle/working"
EXTRACT_TO = os.path.join(WORK_DIR, "test")          # we will extract here
MODEL_PATH = os.path.join(WORK_DIR, "best.keras")    # created by your training cell
SAMPLE_SUB = os.path.join(DATA_DIR, "sampleSubmission.csv")

print("probe ok")
print("DATA_DIR :", DATA_DIR)
print("TEST_DIR  :", EXTRACT_TO)
print("sampleSubmission exists:", os.path.exists(SAMPLE_SUB))

# --- Safety checks ---
assert os.path.exists(MODEL_PATH), "best.keras not found in /kaggle/working. Run the training cell first."

# --- Ensure test images are extracted (handles both test/ and test/test layouts) ---
if not os.path.exists(EXTRACT_TO):
    os.makedirs(EXTRACT_TO, exist_ok=True)

# If there are clearly no images yet, extract
need_extract = True
for pat in (os.path.join(EXTRACT_TO, "*.png"),
            os.path.join(EXTRACT_TO, "test", "*.png")):
    if glob.glob(pat):
        need_extract = False
        break

if need_extract:
    print("ğŸ“¦ Extracting test.7z ...")
    # -y overwrite, quiet the console noise
    ret = os.system(f"7z x '{TEST_7Z}' -o'{EXTRACT_TO}' -y > /dev/null")
    if ret != 0:
        raise RuntimeError("7z extraction failed. Is test.7z attached under 'Add Input'?")

# Normalise path in case archive created test/test
CANDIDATES = [EXTRACT_TO, os.path.join(EXTRACT_TO, "test")]
TEST_DIR = next((p for p in CANDIDATES if os.path.exists(p)), EXTRACT_TO)

# --- Discover images ---
pngs = sorted(glob.glob(os.path.join(TEST_DIR, "**", "*.png"), recursive=True))
print(f"ğŸ“¸ Found {len(pngs)} images under {TEST_DIR}")
assert len(pngs) == 300000, f"Expected 300000 test images, found {len(pngs)}"

# --- tf.data pipeline (resize to 32x32 and scale to [-1,1]) ---
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [32, 32])
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = (img - 0.5) * 2.0                              # [-1,1]
    return img

BATCH = 1024
ds = (tf.data.Dataset.from_tensor_slices(pngs)
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH)
        .prefetch(tf.data.AUTOTUNE))

# --- Load model & predict ---
best_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("âœ… Loaded best.keras")

probs = best_model.predict(ds, verbose=1)
pred_idx = np.argmax(probs, axis=1)

# --- Build submission (id comes from filename) ---
label_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
pred_labels = [label_names[i] for i in pred_idx]
ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in pngs]

sub = pd.DataFrame({"id": ids, "label": pred_labels}).sort_values("id")
assert len(sub) == 300000, f"Submission length is {len(sub)} not 300000"

# --- Save into /kaggle/working and expose a duplicate name to refresh sidebar ---
SAVE_PATH = os.path.join(WORK_DIR, "submission.csv")
sub.to_csv(SAVE_PATH, index=False)

EXPOSED = os.path.join(WORK_DIR, "submission_rashid.csv")  # duplicate helps sidebar refresh
try:
    shutil.copy(SAVE_PATH, EXPOSED)
except Exception:
    pass

print(f"\nğŸ“� Saved: {SAVE_PATH}")
print(sub.head())

# Show what's in /kaggle/working so you can click-download from the sidebar
print("\n/kaggle/working contents:")
subprocess.run(["ls","-lh","/kaggle/working"])



import os, subprocess, pandas as pd
print("Contents of /kaggle/working:")
subprocess.run(["ls","-lh","/kaggle/working"])

p = "/kaggle/working/submission.csv"
print("\nFound submission.csv?", os.path.exists(p))
if os.path.exists(p):
    print("Size (bytes):", os.path.getsize(p))
    print(pd.read_csv(p, nrows=5))


