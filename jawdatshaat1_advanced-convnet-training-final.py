# ============================================================
# CIFAR-10 (Kaggle) — Custom Residual CNN (from scratch)
# AdamW (+Native EMA), Cosine LR w/ Warmup, MixUp-or-CutMix, 10-crop TTA
# Mixed precision, cached tf.data pipeline, class-labeled previews
# ============================================================

# ------------------------- FLAGS ----------------------------
USE_EMA             = True
EMA_DECAY           = 0.999
USE_CUTOUT          = False       # off (MixUp/CutMix already strong, and this is slow)
USE_MIXUP           = True
EPOCHS              = 260         
WARMUP_EPOCHS       = 8
BATCH               = 128
BASE_LR             = 3e-3
WEIGHT_DECAY        = 1e-4
IMG_SIZE            = 32
SEED                = 1337
JIT_COMPILE         = False       # set True if your session is stable (can speed up)

REFIT_EPOCHS        = 8
SHOW_FIGS           = True
# ------------------------------------------------------------

import os, math, random, numpy as np, pandas as pd, matplotlib.pyplot as plt, zipfile, shutil, time, sys
from IPython.display import display, FileLink
import tensorflow as tf
from tensorflow.keras import mixed_precision

# ---- Setup & reproducibility
mixed_precision.set_global_policy("mixed_float16")
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# ---- Kaggle paths
INPUT_DIR = "/kaggle/input/cifar-10"
WORK_DIR  = "/kaggle/working"
os.makedirs(WORK_DIR, exist_ok=True)
print("INPUT_DIR contents:", os.listdir(INPUT_DIR))

# ---- Extract train.7z / test.7z (dataset PNGs)
try:
    import py7zr  # noqa
except Exception:
    print("Installing py7zr …")
    import subprocess, sys as _sys
    subprocess.check_call([_sys.executable, "-m", "pip", "install", "-q", "py7zr"])
    import py7zr

def extract_7z(src_7z, dst_dir):
    import py7zr
    with py7zr.SevenZipFile(src_7z, mode='r') as z:
        z.extractall(path=dst_dir)

if not os.path.exists(os.path.join(WORK_DIR, "train")):
    extract_7z(os.path.join(INPUT_DIR, "train.7z"), WORK_DIR)
if not os.path.exists(os.path.join(WORK_DIR, "test")):
    extract_7z(os.path.join(INPUT_DIR, "test.7z"), WORK_DIR)
print("Extracted:", [p for p in os.listdir(WORK_DIR) if p in ["train","test"]])

# ---- Labels & split
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
class_to_id = {c:i for i,c in enumerate(CLASSES)}

df = pd.read_csv(os.path.join(INPUT_DIR, "trainLabels.csv"))
df["path"] = df["id"].astype(str).radd(os.path.join(WORK_DIR, "train") + "/") + ".png"
df["label_id"] = df["label"].map(class_to_id).astype(int)

# Stratified 45k/5k
try:
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=5000, random_state=SEED)
    idx_train, idx_valid = next(sss.split(df["id"].values, df["label_id"].values))
    train_df = df.iloc[idx_train].reset_index(drop=True)
    valid_df = df.iloc[idx_valid].reset_index(drop=True)
    print("Using STRATIFIED 45k/5k split")
except Exception as e:
    print("Stratified split unavailable, falling back to last-5k:", e)
    valid_ids = set(df.tail(5000)["id"].tolist())
    train_df = df[~df["id"].isin(valid_ids)].reset_index(drop=True)
    valid_df = df[df["id"].isin(valid_ids)].reset_index(drop=True)

print(f"Train: {len(train_df)}  Valid: {len(valid_df)}")

# ---- tf.data + MixUp/CutMix
AUTO = tf.data.AUTOTUNE
NUM_CLASSES = 10
MEAN = tf.constant([0.4914, 0.4822, 0.4465], tf.float32)
STD  = tf.constant([0.2470, 0.2435, 0.2616], tf.float32)

def decode_image(path, label_id):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)     # [0,1]
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = (img - MEAN) / STD
    y = tf.one_hot(label_id, NUM_CLASSES, dtype=tf.float32)
    return img, y

def color_jitter(img, y):
    if tf.random.uniform([]) < 0.5:
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.image.random_saturation(img, 0.85, 1.15)
    return img, y

def pad_random_crop_flip(img, y):
    img = tf.pad(img, [[4,4],[4,4],[0,0]], mode="REFLECT")
    img = tf.image.random_crop(img, [IMG_SIZE, IMG_SIZE, 3])
    img = tf.image.random_flip_left_right(img)
    return img, y

def cutout(img, mask_size=16):
    h = tf.shape(img)[0]; w = tf.shape(img)[1]
    m = tf.cast(mask_size, tf.int32)
    cy = tf.random.uniform([], 0, h, dtype=tf.int32)
    cx = tf.random.uniform([], 0, w, dtype=tf.int32)
    y1 = tf.clip_by_value(cy - m//2, 0, h); y2 = tf.clip_by_value(cy + m//2, 0, h)
    x1 = tf.clip_by_value(cx - m//2, 0, w); x2 = tf.clip_by_value(cx + m//2, 0, w)
    mask = tf.ones([y2-y1, x2-x1, 3], dtype=img.dtype)
    mask = tf.pad(mask, [[y1, h-y2],[x1, w-x2],[0,0]])
    return img * (1.0 - mask)

def sample_beta_tf(alpha=0.8):
    u1 = tf.random.gamma([], alpha, dtype=tf.float32)
    u2 = tf.random.gamma([], alpha, dtype=tf.float32)
    lam = u1 / (u1 + u2)
    return tf.maximum(lam, 1.0 - lam)

def rand_bbox(h, w, lam):
    h = tf.cast(h, tf.int32); w = tf.cast(w, tf.int32)
    cut_ratio = tf.sqrt(1.0 - lam)
    ch = tf.cast(tf.cast(h, tf.float32) * cut_ratio, tf.int32)
    cw = tf.cast(tf.cast(w, tf.float32) * cut_ratio, tf.int32)
    cy = tf.random.uniform([], 0, h, dtype=tf.int32)
    cx = tf.random.uniform([], 0, w, dtype=tf.int32)
    y1 = tf.clip_by_value(cy - ch//2, 0, h); y2 = tf.clip_by_value(cy + ch//2, 0, h)
    x1 = tf.clip_by_value(cx - cw//2, 0, w); x2 = tf.clip_by_value(cx + cw//2, 0, w)
    return y1, y2, x1, x2

@tf.function
def mixup_or_cutmix(images, labels, p_cutmix=0.5, alpha=0.8):
    r = tf.random.uniform([])
    bs = tf.shape(images)[0]
    idx = tf.random.shuffle(tf.range(bs))
    images2 = tf.gather(images, idx)
    labels2 = tf.gather(labels, idx)

    def cutmix_branch():
        lam = sample_beta_tf(alpha)
        h = tf.shape(images)[1]; w = tf.shape(images)[2]
        y1, y2, x1, x2 = rand_bbox(h, w, lam)
        mask = tf.ones([bs, y2-y1, x2-x1, 3], dtype=images.dtype)
        mask = tf.pad(mask, [[0,0],[y1, h-y2],[x1, w-x2],[0,0]])
        mixed = images * (1.0 - mask) + images2 * mask
        lam_eff = 1.0 - tf.cast((y2-y1)*(x2-x1), tf.float32) / tf.cast(h*w, tf.float32)
        out_lbl = lam_eff * labels + (1.0 - lam_eff) * labels2
        return mixed, out_lbl

    def mixup_branch():
        lam = sample_beta_tf(alpha)
        mixed = lam * images + (1.0 - lam) * images2
        out_lbl = lam * labels + (1.0 - lam) * labels2
        return mixed, out_lbl

    return tf.cond(r < p_cutmix, cutmix_branch, mixup_branch)

MIX_PROB = tf.Variable(0.6 if USE_MIXUP else 0.0, trainable=False, dtype=tf.float32)

def after_batch_aug(images, labels):
    do_mix = tf.logical_and(tf.constant(USE_MIXUP), tf.less(tf.random.uniform([]), MIX_PROB))
    def mix_branch():    return mixup_or_cutmix(images, labels, p_cutmix=0.5, alpha=0.8)
    def no_mix_branch():
        imgs = tf.map_fn(lambda im: cutout(im), images) if USE_CUTOUT else images
        return imgs, labels
    return tf.cond(do_mix, mix_branch, no_mix_branch)

def apply_ds_opts(ds, deterministic=False):
    opts = tf.data.Options()
    opts.deterministic = deterministic
    exp = opts.experimental_optimization
    for name, val in [
        ("autotune_buffers", True),
        ("apply_default_optimizations", True),
        ("map_parallelization", True),
    ]:
        if hasattr(exp, name):
            try: setattr(exp, name, val)
            except Exception: pass
    return ds.with_options(opts)

def make_base_ds(frame, shuffle=False, cache=True):
    ds = tf.data.Dataset.from_tensor_slices((frame["path"].values, frame["label_id"].values))
    ds = ds.map(decode_image, num_parallel_calls=AUTO)
    if cache: ds = ds.cache()
    if shuffle: ds = ds.shuffle(8192, reshuffle_each_iteration=True)
    return ds

def make_train_ds(df_train):
    ds = make_base_ds(df_train, shuffle=True, cache=True)
    ds = ds.map(pad_random_crop_flip, num_parallel_calls=AUTO, deterministic=False)
    ds = ds.map(color_jitter, num_parallel_calls=AUTO, deterministic=False)
    ds = ds.batch(BATCH, drop_remainder=True)
    ds = ds.map(after_batch_aug, num_parallel_calls=AUTO, deterministic=False)
    ds = apply_ds_opts(ds, deterministic=False).prefetch(AUTO)
    return ds

def make_valid_ds(df_valid):
    ds = make_base_ds(df_valid, shuffle=False, cache=True)
    ds = ds.batch(BATCH)
    ds = apply_ds_opts(ds, deterministic=True).prefetch(AUTO)
    return ds

train_ds = make_train_ds(train_df)
valid_ds = make_valid_ds(valid_df)

# ---- Augmented sample preview
if SHOW_FIGS:
    xb, yb = next(iter(train_ds.take(1)))
    grid = min(16, xb.shape[0])
    plt.figure(figsize=(8,8))
    for i in range(grid):
        ax = plt.subplot(4,4,i+1)
        img = xb[i].numpy()
        img_vis = (img * STD.numpy()) + MEAN.numpy()
        img_vis = np.clip(img_vis, 0, 1)
        top = int(np.argmax(yb[i].numpy()))
        conf = float(np.max(yb[i].numpy()))
        ax.imshow(img_vis); ax.axis('off')
        ax.set_title(f"{CLASSES[top]} ({conf:.2f})", fontsize=9)
    plt.suptitle("Augmented samples (MixUp/CutMix)"); plt.tight_layout(); plt.show(); plt.close()

# ---- Model (ResNet-ish + SE)
from tensorflow.keras import layers, models, regularizers

def conv_bn_relu(x, f, k=3, s=1, wd=WEIGHT_DECAY):
    x = layers.Conv2D(f, k, s, padding="same", use_bias=False,
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)

def se_block(x, r=16):
    c = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(max(c//r, 8), activation="relu", kernel_initializer="he_normal")(s)
    s = layers.Dense(c, activation="sigmoid", kernel_initializer="he_normal")(s)
    s = layers.Reshape((1,1,c))(s)
    return layers.Multiply()([x, s])

def basic_block(x, f, down=False, wd=WEIGHT_DECAY, use_se=True, drop_path=0.0):
    s = 2 if down else 1
    shortcut = x
    y = conv_bn_relu(x, f, 3, s, wd)
    y = layers.Conv2D(f, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(wd))(y)
    y = layers.BatchNormalization()(y)
    if use_se: y = se_block(y)
    if drop_path > 0: y = layers.Dropout(drop_path)(y)
    if down or x.shape[-1] != f:
        shortcut = layers.Conv2D(f, 1, s, padding="same", use_bias=False,
                                 kernel_initializer="he_normal",
                                 kernel_regularizer=regularizers.l2(wd))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    out = layers.Add()([shortcut, y])
    return layers.ReLU()(out)

def build_resnet_cifar(num_classes=10, wd=WEIGHT_DECAY, head_drop=0.15):
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = layers.Activation("linear", dtype="float32", name="cast_to_f32")(inp)
    def stage(x, f, n, down_first, dp):
        for i in range(n):
            x = basic_block(x, f, down=(down_first and i==0), wd=wd, use_se=True, drop_path=dp)
        return x
    x = conv_bn_relu(x, 64, 3, 1, wd)
    x = stage(x, 64,  3, False, 0.0)
    x = stage(x, 128, 3, True,  0.0)
    x = stage(x, 256, 3, True,  0.10)  # mild stochastic depth late
    x = stage(x, 512, 3, True,  0.10)
    x = layers.GlobalAveragePooling2D()(x)
    if head_drop > 0: x = layers.Dropout(head_drop)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32",
                       kernel_initializer="he_normal")(x)
    return models.Model(inp, out, name="CIFAR10_CustomResNet_SE")

model = build_resnet_cifar()
model.summary()

# ---- Optimizer + cosine LR with warmup (AdamW + native EMA)
from tensorflow.keras.optimizers import AdamW
optimizer = AdamW(learning_rate=BASE_LR, weight_decay=WEIGHT_DECAY,
                  use_ema=USE_EMA, ema_momentum=EMA_DECAY, clipnorm=1.0)

def lr_schedule(epoch):
    if epoch < WARMUP_EPOCHS:
        return BASE_LR * (epoch + 1) / WARMUP_EPOCHS
    t = (epoch - WARMUP_EPOCHS) / max(1, (EPOCHS - WARMUP_EPOCHS))
    min_lr = BASE_LR * 0.01
    return min_lr + 0.5 * (BASE_LR - min_lr) * (1 + math.cos(math.pi * t))

class MixProbScheduler(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # taper mixing sooner for sharper late training
        if epoch < 150:   MIX_PROB.assign(0.6)
        elif epoch < 200: MIX_PROB.assign(0.3)
        else:             MIX_PROB.assign(0.1)

lr_cb = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
ckpt_cb = tf.keras.callbacks.ModelCheckpoint("best.keras", monitor="val_accuracy",
                                             save_best_only=True, mode="max", verbose=1)
early_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=35,
                                            restore_best_weights=True, verbose=1, start_from_epoch=120)
nan_cb = tf.keras.callbacks.TerminateOnNaN()
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"], jit_compile=JIT_COMPILE)

# ---- Train (verbose=1 shows batch progress so it never looks frozen)
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[MixProbScheduler(), lr_cb, ckpt_cb, early_cb, nan_cb],
    verbose=1
)

# ---- Curves
if SHOW_FIGS:
    hist = history.history
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(hist["loss"]); plt.plot(hist["val_loss"]); plt.title("Loss"); plt.legend(["train","val"]); plt.grid(True, ls="--", alpha=.4)
    plt.subplot(1,2,2); plt.plot(hist["accuracy"]); plt.plot(hist["val_accuracy"]); plt.title("Accuracy"); plt.legend(["train","val"]); plt.grid(True, ls="--", alpha=.4)
    plt.tight_layout(); plt.show(); plt.close()

val_loss, val_acc = model.evaluate(valid_ds, verbose=0)
print(f"Validation accuracy: {val_acc:.4f}")

# --- Ensure best.keras in /kaggle/working BEFORE next steps
ckpt_path = os.path.join(WORK_DIR, "best.keras")
if os.path.exists("best.keras") and (not os.path.samefile("best.keras", ckpt_path)):
    os.replace("best.keras", ckpt_path)
elif not os.path.exists(ckpt_path) and os.path.exists("best.keras"):
    os.replace("best.keras", ckpt_path)
elif not os.path.exists(ckpt_path):
    model.save(ckpt_path)
print("Checkpoint at:", ckpt_path, " | Exists:", os.path.exists(ckpt_path))

# =========================
# Refit on ALL 50k (train+valid) with tiny LR, then 10-crop TTA
# =========================

# 1) Refit on all labels
all_df = pd.concat([train_df, valid_df], ignore_index=True)
def make_all_ds(df_all):
    ds = make_base_ds(df_all, shuffle=True, cache=True)
    ds = ds.map(pad_random_crop_flip, num_parallel_calls=AUTO, deterministic=False)
    ds = ds.map(color_jitter, num_parallel_calls=AUTO, deterministic=False)
    ds = ds.batch(BATCH, drop_remainder=True)
    ds = ds.map(after_batch_aug, num_parallel_calls=AUTO, deterministic=False)
    ds = apply_ds_opts(ds, deterministic=False).prefetch(AUTO)
    return ds
all_ds = make_all_ds(all_df)

from tensorflow.keras.models import load_model
best = load_model(ckpt_path, compile=False)
best.compile(
    optimizer=AdamW(learning_rate=1e-4, weight_decay=WEIGHT_DECAY, use_ema=USE_EMA, ema_momentum=EMA_DECAY),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.01),
    metrics=["accuracy"],
    jit_compile=JIT_COMPILE
)
print("\n[Refit] Training on all 50k with tiny LR…")
best.fit(all_ds, epochs=REFIT_EPOCHS, verbose=1)
best.save(ckpt_path)
del best

# 2) Deterministic 10-crop TTA (5 crops × {no-flip, flip})
print("\n[TTA] 10-crop (5 crops × flip) inference…")

def decode_test_raw(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = (img - MEAN) / STD
    return img

def pad_and_crop(img, top, left, size=IMG_SIZE, pad=4):
    im = tf.pad(img, [[pad,pad],[pad,pad],[0,0]], mode="REFLECT")
    return tf.image.crop_to_bounding_box(im, top, left, size, size)

CROPS = [(0,0), (0,8), (8,0), (8,8), (4,4)]

sub_template = pd.read_csv(os.path.join(INPUT_DIR, "sampleSubmission.csv"))
test_paths = [os.path.join(WORK_DIR, "test", (str(fid) if str(fid).endswith(".png") else f"{fid}.png"))
              for fid in sub_template["id"].values]
path_ds = tf.data.Dataset.from_tensor_slices(test_paths)

def make_variant_ds(flip=False, crop_idx=0):
    def _map(p):
        img = decode_test_raw(p)
        top,left = CROPS[crop_idx]
        img = pad_and_crop(img, top, left)
        if flip: img = tf.image.flip_left_right(img)
        return img
    return (path_ds.map(_map, num_parallel_calls=AUTO)
            .batch(BATCH).prefetch(AUTO))

best = tf.keras.models.load_model(ckpt_path, compile=False)

probs_sum = None
variants = []
for ci in range(5):
    variants.append(make_variant_ds(flip=False, crop_idx=ci))
    variants.append(make_variant_ds(flip=True,  crop_idx=ci))

for i, ds in enumerate(variants):
    print(f"Variant {i+1}/10")
    p = best.predict(ds, verbose=1)
    probs_sum = p if probs_sum is None else probs_sum + p

probs = probs_sum / 10.0
pred_ids = np.argmax(probs, axis=1)
pred_labels = [CLASSES[i] for i in pred_ids]

# ---- Write submissions visible in Output pane
sub = sub_template.copy()
sub["label"] = pred_labels

out_csv   = os.path.join(WORK_DIR, "submission_tta.csv")
alias_csv = os.path.join(WORK_DIR, "submission.csv")
zip_path  = os.path.join(WORK_DIR, "submission_tta.zip")

sub.to_csv(out_csv, index=False)
if os.path.exists(alias_csv): os.remove(alias_csv)
shutil.copy2(out_csv, alias_csv)

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(out_csv,   arcname=os.path.basename(out_csv))
    zf.write(alias_csv, arcname=os.path.basename(alias_csv))

# ---- Sanity checks & sidebar-friendly links
def _size(p):
    try: return os.path.getsize(p)
    except: return 0
print("\n=== OUTPUT CHECKS (post-refit + 10-crop TTA) ===")
print("Dir:", WORK_DIR)
print("CSV exists:", os.path.exists(out_csv), "  size:", _size(out_csv))
print("Alias exists:", os.path.exists(alias_csv), "  size:", _size(alias_csv))
print("ZIP exists:", os.path.exists(zip_path), "  size:", _size(zip_path))
try:
    display(pd.read_csv(out_csv).head())
except Exception as e:
    print("Preview failed:", e)

for p in [out_csv, alias_csv, zip_path]:
    if os.path.exists(p):
        display(FileLink(p))

sys.stdout.flush()
print("\nReady to download from the Output panel:")
print("  -", out_csv)
print("  -", alias_csv)
print("  -", zip_path)


