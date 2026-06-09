# Step 1: sanity check + EDA for Plant Seedlings Classification
import os, sys, glob, random
from collections import Counter
from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt

data_root = "/kaggle/input/plant-seedlings-classification"
train_dir = f"{data_root}/train"

assert os.path.isdir(train_dir), f"Train dir not found at {train_dir}"

# list classes
classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
print(f"Classes ({len(classes)}): {classes}")

# count images per class and check for corrupt files
class_counts = {}
corrupt = []
sizes_w, sizes_h = [], []

for c in classes:
    paths = glob.glob(os.path.join(train_dir, c, "*"))
    class_counts[c] = 0
    for p in paths:
        try:
            with Image.open(p) as im:
                im.verify()              # quick corruption check
            with Image.open(p) as im2:    # reopen to get size
                w, h = im2.size
                sizes_w.append(w); sizes_h.append(h)
            class_counts[c] += 1
        except UnidentifiedImageError:
            corrupt.append(p)

print("\nImage count per class:")
for k, v in class_counts.items():
    print(f"{k:20s} {v:5d}")

print(f"\nTotal images: {sum(class_counts.values())}")
print(f"Corrupt images detected: {len(corrupt)}")
if corrupt:
    print("Examples of corrupt:", corrupt[:3])

# size stats
print(f"\nWidth  min={np.min(sizes_w)}, max={np.max(sizes_w)}, mean={int(np.mean(sizes_w))}")
print(f"Height min={np.min(sizes_h)}, max={np.max(sizes_h)}, mean={int(np.mean(sizes_h))}")

# bar chart of class distribution
plt.figure(figsize=(10,4))
plt.bar(class_counts.keys(), class_counts.values())
plt.xticks(rotation=45, ha="right")
plt.title("Images per class")
plt.tight_layout()
plt.show()

# show one sample per class
plt.figure(figsize=(14,8))
for i, c in enumerate(classes[:12]):  # dataset has 12 classes, show up to 12
    path = glob.glob(os.path.join(train_dir, c, "*"))[0]
    img = Image.open(path).convert("RGB")
    plt.subplot(3, 4, i+1)
    plt.imshow(img)
    plt.title(c, fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.show()



# Step 2: datasets and class weights

import os, math, numpy as np, tensorflow as tf
from collections import Counter

data_root = "/kaggle/input/plant-seedlings-classification"
train_dir = f"{data_root}/train"

IMG_SIZE  = 224
BATCH     = 32
VAL_SPLIT = 0.15
SEED      = 1337

# 2.1 stratified split using image_dataset_from_directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical"
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names, "  ->", num_classes)

# 2.2 data augmentation and preprocessing
data_augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

normalizer = tf.keras.layers.Rescaling(1./255)

def augment_batch(x, y):
    return data_augment(x, training=True), y

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (train_ds
            .shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
            .map(lambda x, y: (normalizer(x), y), num_parallel_calls=AUTOTUNE)
            .map(augment_batch, num_parallel_calls=AUTOTUNE)
            .prefetch(AUTOTUNE))

val_ds = (val_ds
          .map(lambda x, y: (normalizer(x), y), num_parallel_calls=AUTOTUNE)
          .prefetch(AUTOTUNE))

# 2.3 class weights from folder counts (slight imbalance helps with weighting)
counts = {cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in class_names}
total = sum(counts.values())
class_weights = {i: total / (num_classes * counts[class_names[i]]) for i in range(num_classes)}
print("Per-class image counts:", counts)
print("Class weights:", class_weights)



import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
print(tf.__version__)



# Step 3A: EfficientNet baseline, train classifier head first

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 224
num_classes = len(class_names)

# mixed precision helps if you attach a GPU. Harmless on CPU.
try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
except Exception as e:
    print("Mixed precision policy not set:", e)

# 3A.1 backbone
base = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg"
)
base.trainable = False  # freeze for head training

# 3A.2 head
inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
x = inputs
x = layers.Rescaling(1.0)(x)   # no-op because we already normalized, but keeps graph stable
x = base(x, training=False)
x = layers.Dropout(0.3)(x)
# label smoothing will be applied via loss, final activation stays softmax
outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
model = models.Model(inputs, outputs)

# 3A.3 optimizer + loss + metrics
# cosine decay schedule for stable convergence
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
total_epochs_head = 8
lr_start = 3e-4
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=lr_start,
    decay_steps=steps_per_epoch * total_epochs_head
)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

model.compile(optimizer=opt,
              loss=loss,
              metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])

# 3A.4 callbacks
ckpt = tf.keras.callbacks.ModelCheckpoint(
    "best_head.keras", monitor="val_acc", mode="max",
    save_best_only=True, verbose=1
)
early = tf.keras.callbacks.EarlyStopping(
    monitor="val_acc", mode="max",
    patience=4, restore_best_weights=True
)
reduce = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, verbose=1
)

# 3A.5 train head
history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs_head,
    class_weight=class_weights,
    callbacks=[ckpt, early, reduce],
    verbose=1
)

print("Head training done.")



from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input
from tensorflow.keras import layers, models
import tensorflow as tf

IMG_SIZE = 224
num_classes = len(class_names)

# rebuild the backbone and head with correct preprocessing
base = EfficientNetV2S(include_top=False, weights="imagenet",
                       input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg")
base.trainable = False

inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
x = preprocess_input(inputs)         # this is the only normalization
x = base(x, training=False)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
model = models.Model(inputs, outputs)



for xb, yb in train_ds.take(1):
    print("y shape:", yb.shape, "dtype:", yb.dtype, "sample:", yb[0])



import tensorflow as tf

steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
total_epochs_head = 8

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=3e-4,
    decay_steps=steps_per_epoch * total_epochs_head
)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

model.compile(optimizer=opt,
              loss=loss,
              metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])



ckpt = tf.keras.callbacks.ModelCheckpoint("best_head.keras",
                                          monitor="val_acc", mode="max",
                                          save_best_only=True, verbose=1)
early = tf.keras.callbacks.EarlyStopping(monitor="val_acc",
                                         mode="max", patience=4,
                                         restore_best_weights=True)
reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                              factor=0.5, patience=2, verbose=1)

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    class_weight=class_weights,
    callbacks=[ckpt, early, reduce],
    verbose=1
)



import tensorflow as tf
from tensorflow import data as tfdata

IMG_SIZE = 224
BATCH = 32
SEED = 1337

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,                # your /kaggle/input/plant-seedlings-classification/train
    label_mode="categorical",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode="categorical",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
)

class_names = train_ds.class_names
print("classes:", class_names)

AUTOTUNE = tfdata.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)



from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input

num_classes = len(class_names)

base = EfficientNetV2S(
    include_top=False, weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg"
)
base.trainable = False

inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
x = preprocess_input(inputs)     # this is the key change
x = base(x, training=False)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
model = models.Model(inputs, outputs)

import tensorflow as tf
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
total_epochs_head = 8
lr = tf.keras.optimizers.schedules.CosineDecay(3e-4, steps_per_epoch * total_epochs_head)

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="val_acc")]
)

ckpt   = tf.keras.callbacks.ModelCheckpoint("best_head.keras", monitor="val_acc", mode="max", save_best_only=True, verbose=1)
early  = tf.keras.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=4, restore_best_weights=True)
reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=2, verbose=1)



history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    callbacks=[ckpt, early, reduce],
    verbose=1
)

print("best val_acc:", round(max(history_head.history["val_acc"]), 4))
model.load_weights("best_head.keras")
print("eval:", [round(x, 4) for x in model.evaluate(val_ds, verbose=0)])



# Step 4A: unfreeze top layers and recompile with a tiny LR + AdamW

import math
import tensorflow as tf
from tensorflow.keras import optimizers

# unfreeze last ~100 layers of the EfficientNet backbone
base.trainable = True
trainable_from = len(base.layers) - 100
for i, layer in enumerate(base.layers):
    layer.trainable = (i >= trainable_from)

# optional: keep BatchNorms in inference mode to avoid instability
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# very small LR for fine tuning
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
total_epochs_ft = 15
lr_ft = 3e-5

try:
    opt = tf.keras.optimizers.AdamW(learning_rate=lr_ft, weight_decay=1e-5)
except Exception:
    # older TF without AdamW
    opt = tf.keras.optimizers.Adam(learning_rate=lr_ft)

loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

model.compile(optimizer=opt,
              loss=loss,
              metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])

ckpt_ft = tf.keras.callbacks.ModelCheckpoint(
    "best_ft.keras", monitor="val_acc", mode="max",
    save_best_only=True, verbose=1
)
early_ft = tf.keras.callbacks.EarlyStopping(
    monitor="val_acc", mode="max",
    patience=5, restore_best_weights=True
)
reduce_ft = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, verbose=1
)



# Step 4B: fine tune
history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs_ft,
    class_weight=class_weights,
    callbacks=[ckpt_ft, early_ft, reduce_ft],
    verbose=1
)

print("best val_acc:", round(max(history_ft.history["val_acc"]), 4))
model.load_weights("best_ft.keras")
print("eval:", [round(x, 4) for x in model.evaluate(val_ds, verbose=0)])



# Step 5A: fine-tune the top of EfficientNetV2S

import tensorflow as tf
from tensorflow.keras import optimizers, callbacks

# unfreeze last N blocks of the backbone
for layer in base.layers:
    layer.trainable = False
unfreeze = 60                      # tweak 40–120 if you have time
for layer in base.layers[-unfreeze:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

# lower LR schedule for fine-tuning
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
fine_epochs = 15
base_lr = 3e-5
wd = 1e-5
opt_fine = optimizers.AdamW(learning_rate=optimizers.schedules.CosineDecay(
                               initial_learning_rate=base_lr,
                               decay_steps=steps_per_epoch * fine_epochs),
                            weight_decay=wd)

model.compile(optimizer=opt_fine,
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
              metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])

ckpt2 = callbacks.ModelCheckpoint("best_finetune.keras",
                                  monitor="val_acc", mode="max",
                                  save_best_only=True, verbose=1)
early2 = callbacks.EarlyStopping(monitor="val_acc", mode="max",
                                 patience=5, restore_best_weights=True)
reduce2 = callbacks.ReduceLROnPlateau(monitor="val_loss",
                                      factor=0.5, patience=2, verbose=1)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=fine_epochs,
    class_weight=class_weights,
    callbacks=[ckpt2, early2, reduce2],
    verbose=1
)

print("Fine-tune done.")
print("Eval:", [round(x, 4) for x in model.evaluate(val_ds, verbose=0)])



history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=fine_epochs,           # same as before
    class_weight=class_weights,   # same
    callbacks=[ckpt2, early2],    # <= removed reduce2
    verbose=1
)

print("Fine-tune done.")
print("Eval:", [round(x, 4) for x in model.evaluate(val_ds, verbose=0)])



# reload the best finetuned weights, just to be safe
from tensorflow import keras
model.load_weights("best_finetune.keras")

# save whole model for reproducibility
model.save("plant_effnetv2s_finetune.keras")
print("Saved.")



import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# collect ground truth and predictions on validation set
y_true = []
y_pred = []

for xb, yb in val_ds:
    prob = model.predict(xb, verbose=0)
    y_pred.append(np.argmax(prob, axis=1))
    y_true.append(np.argmax(yb.numpy(), axis=1))

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
classes = class_names

# pretty plot
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(cm, interpolation="nearest")
ax.figure.colorbar(im, ax=ax)
ax.set_xticks(np.arange(len(classes))); ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, rotation=90); ax.set_yticklabels(classes)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix")
plt.tight_layout(); plt.show()

# normalized confusion matrix for class balance reading
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, interpolation="nearest")
ax.figure.colorbar(im, ax=ax)
ax.set_xticks(np.arange(len(classes))); ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, rotation=90); ax.set_yticklabels(classes)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Normalized Confusion Matrix")
plt.tight_layout(); plt.show()

# text report
print(classification_report(y_true, y_pred, target_names=classes, digits=4))



import numpy as np, matplotlib.pyplot as plt, itertools
from PIL import Image
# collect one batch of predictions from val_ds
y_true = []
y_pred = []
paths = []
for xb, yb in val_ds.unbatch().take(2000):  # enough for full val
    y_true.append(np.argmax(yb.numpy()))
    pr = model.predict(xb[None, ...], verbose=0)
    y_pred.append(np.argmax(pr))
y_true = np.array(y_true); y_pred = np.array(y_pred)

cls_to_id = {c:i for i,c in enumerate(class_names)}
target = cls_to_id['Black-grass']
idx = np.where((y_true == target) & (y_pred != target))[0]
print("Black-grass misclassified:", len(idx))

# show 16 mistakes
subset = idx[:16]
plt.figure(figsize=(10,10))
for i,k in enumerate(subset):
    plt.subplot(4,4,i+1)
    img = next(itertools.islice(val_ds.unbatch().as_numpy_iterator(), k, None))[0]
    plt.imshow((img*255).astype("uint8"))
    plt.title(f"pred {class_names[y_pred[k]]}")
    plt.axis("off")
plt.tight_layout(); plt.show()



import numpy as np, matplotlib.pyplot as plt, itertools

# collect one batch of predictions
y_true, y_prob = [], []
for xb, yb in val_ds.unbatch().take(2000):
    y_true.append(yb.numpy())
    y_prob.append(model.predict(xb[None, ...], verbose=0)[0])
y_true = np.array(y_true)
y_pred = y_prob = np.array(y_prob)
cls_to_id = {c:i for i, c in enumerate(class_names)}
target = cls_to_id['Black-grass']
idx = np.where((y_true.argmax(1) == target) & (y_pred.argmax(1) != target))[0]
print("Black-grass misclassified:", len(idx))

# show up to 16 mistakes
subset = idx[:16]
plt.figure(figsize=(10,10))
for k, i in enumerate(subset):
    plt.subplot(4,4,k+1)
    # images from val_ds are 0..255 float32 already, so just clip and cast
    img = next(itertools.islice(val_ds.unbatch().as_numpy_iterator(), i, None))[0]
    plt.imshow(np.clip(img, 0, 255).astype("uint8"))
    plt.title(f"pred {class_names[y_pred[i].argmax()]}")
    plt.axis("off")
plt.tight_layout(); plt.show()



from tensorflow.keras import layers
import tensorflow as tf

cls_to_id = {c:i for i, c in enumerate(class_names)}
weak_id = cls_to_id['Black-grass']

aug_extra = tf.keras.Sequential([
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.2),
    layers.RandomZoom(0.12),
    layers.GaussianNoise(5.0),
], name="boost_aug")

def boost_weak(x, y):
    # y is one-hot, pick weak samples in batch
    mask = tf.equal(tf.argmax(y, axis=1), weak_id)
    mask = tf.cast(mask, x.dtype)
    x_boost = aug_extra(x, training=True)
    # apply heavy aug only to weak samples
    x = x_boost * mask[:, None, None, None] + x * (1 - mask)[:, None, None, None]
    return x, y

AUTOTUNE = tf.data.AUTOTUNE
train_ds_boost = train_ds.map(boost_weak, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)



from tensorflow.keras import layers
import tensorflow as tf

cls_to_id = {c:i for i, c in enumerate(class_names)}
weak_id = cls_to_id['Black-grass']

aug_extra = tf.keras.Sequential([
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.2),
    layers.RandomZoom(0.12),
    layers.GaussianNoise(5.0),
], name="boost_aug")

def boost_weak(x, y):
    # y is one-hot, pick weak samples in batch
    mask = tf.equal(tf.argmax(y, axis=1), weak_id)
    mask = tf.cast(mask, x.dtype)
    x_boost = aug_extra(x, training=True)
    # apply heavy aug only to weak samples
    x = x_boost * mask[:, None, None, None] + x * (1 - mask)[:, None, None, None]
    return x, y

AUTOTUNE = tf.data.AUTOTUNE
train_ds_boost = train_ds.map(boost_weak, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)



from tensorflow.keras import layers
import tensorflow as tf

cls_to_id = {c:i for i, c in enumerate(class_names)}
weak_id = cls_to_id['Black-grass']

# do aug in 0..1 space, then scale back to 0..255
aug_extra = tf.keras.Sequential([
    layers.Rescaling(1/255.0),
    layers.RandomContrast(0.30),
    layers.RandomBrightness(0.20),
    layers.RandomZoom(0.12),
    layers.GaussianNoise(0.08),          # valid in 0..1 space
    layers.Lambda(lambda z: tf.clip_by_value(z, 0., 1.)),
    layers.Rescaling(255.0),
], name="boost_aug")

def boost_weak(x, y):
    # y is one-hot, pick weak samples in batch
    mask = tf.equal(tf.argmax(y, axis=1), weak_id)
    mask = tf.cast(mask, x.dtype)
    x_boost = aug_extra(x, training=True)
    # apply heavy aug only to weak samples
    x = x_boost * mask[:, None, None, None] + x * (1 - mask)[:, None, None, None]
    return x, y

AUTOTUNE = tf.data.AUTOTUNE
train_ds_boost = train_ds.map(boost_weak, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)



from tensorflow.keras import layers
import tensorflow as tf

cls_to_id = {c:i for i, c in enumerate(class_names)}
weak_id = cls_to_id["Black-grass"]

# 0..255 in, do aug in 0..1, then go back to 0..255, keep float32 throughout
aug_extra = tf.keras.Sequential([
    layers.Rescaling(1.0/255.0, dtype="float32"),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.2),
    layers.RandomZoom(0.12),
    layers.GaussianNoise(0.08),                 # stddev must be 0..1, 0.08 is fine
    layers.Lambda(lambda z: tf.clip_by_value(z, 0., 1.)),
    layers.Rescaling(255.0, dtype="float32"),
], name="boost_aug")

def boost_weak(x, y):
    # pick weak-class samples in the batch
    mask = tf.equal(tf.argmax(y, axis=1), weak_id)          # [B]
    # make sure dtypes match for the blend
    x = tf.cast(x, tf.float32)
    m = tf.cast(mask, tf.float32)
    m = tf.reshape(m, (-1, 1, 1, 1))                        # broadcast to image shape
    x_boost = tf.cast(aug_extra(x, training=True), tf.float32)
    x = x_boost * m + x * (1.0 - m)
    return x, y

AUTOTUNE = tf.data.AUTOTUNE
train_ds_boost = train_ds.map(boost_weak, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)



xb, yb = next(iter(train_ds_boost.take(1)))
print(xb.dtype, xb.numpy().min(), xb.numpy().max(), yb.dtype)



fine2_epochs = 5

ckpt3  = tf.keras.callbacks.ModelCheckpoint(
    "best_boost.keras", monitor="val_acc", mode="max",
    save_best_only=True, verbose=1
)
early3 = tf.keras.callbacks.EarlyStopping(
    monitor="val_acc", mode="max",
    patience=3, restore_best_weights=True
)

history_boost = model.fit(
    train_ds_boost,
    validation_data=val_ds,
    epochs=fine2_epochs,
    class_weight=class_weights,
    callbacks=[ckpt3, early3],
    verbose=1
)

print("Eval:", [round(x, 4) for x in model.evaluate(val_ds, verbose=0)])



# full validation predictions
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

y_true, y_prob = [], []
for xb, yb in val_ds:
    y_true.append(yb.numpy())
    y_prob.append(model.predict(xb, verbose=0))
y_true = np.concatenate(y_true, 0)
y_prob = np.concatenate(y_prob, 0)
y_pred = y_prob.argmax(1)

print(classification_report(y_true.argmax(1), y_pred, target_names=class_names, digits=4))



# Step: make the weak-class boost stronger
from tensorflow.keras import layers
import tensorflow as tf

weak_id = {c:i for i,c in enumerate(class_names)}['Black-grass']

aug_extra = tf.keras.Sequential([
    layers.Rescaling(1/255.0, dtype="float32"),
    layers.RandomContrast(0.45),
    layers.RandomBrightness(0.25),
    layers.RandomZoom(0.15),
    layers.GaussianNoise(0.12),
    layers.Lambda(lambda z: tf.clip_by_value(z, 0., 1.)),
    layers.Rescaling(255.0, dtype="float32"),
], name="boost_aug_v2")

def boost_weak(x, y):
    # mask batch items that are Black-grass
    m = tf.cast(tf.equal(tf.argmax(y, axis=1), weak_id), tf.float32)   # shape [B]
    m = tf.reshape(m, (-1, 1, 1, 1))                                   # broadcast to HxWxC
    # apply heavy aug then blend only those items
    x_boost = tf.cast(aug_extra(tf.cast(x, tf.float32), training=True), tf.float32)
    x = x_boost * m + x * (1.0 - m)
    return x, y

AUTOTUNE = tf.data.AUTOTUNE
train_ds_boost2 = train_ds.map(boost_weak, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
print("boosted dataset v2 ready")



# Step: short boost training on the weak class and quick check

import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# very small LR to avoid wrecking the already good weights
opt2 = tf.keras.optimizers.Adam(1e-5)
model.compile(
    optimizer=opt2,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")]
)

ckptb = tf.keras.callbacks.ModelCheckpoint(
    "best_boost2.keras", monitor="val_acc", mode="max",
    save_best_only=True, verbose=1
)
earlyb = tf.keras.callbacks.EarlyStopping(
    monitor="val_acc", mode="max",
    patience=2, restore_best_weights=True
)

history = model.fit(
    train_ds_boost2,
    validation_data=val_ds,
    epochs=3,
    class_weight=class_weights,
    callbacks=[ckptb, earlyb],
    verbose=1
)

print("Eval:", [round(x, 4) for x in model.evaluate(val_ds, verbose=0)])

# quick Black-grass metrics after boost
y_true, y_prob = [], []
for xb, yb in val_ds:
    y_true.append(yb.numpy())
    y_prob.append(model.predict(xb, verbose=0))

y_true = np.concatenate(y_true, 0)
y_pred = np.argmax(np.concatenate(y_prob, 0), 1)

cls_to_id = {c:i for i, c in enumerate(class_names)}
k = cls_to_id['Black-grass']
p, r, f1, _ = precision_recall_fscore_support(y_true.argmax(1), y_pred, labels=[k], average=None)
print({'Black-grass': {'precision': round(float(p[0]),4),
                       'recall': round(float(r[0]),4),
                       'f1': round(float(f1[0]),4)}})



# Mix original training data with boosted stream and upweight Black-grass
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, classification_report

AUTOTUNE = tf.data.AUTOTUNE

# 1) mix datasets
train_ds_mix = train_ds.concatenate(train_ds_boost2).shuffle(4096, reshuffle_each_iteration=True).prefetch(AUTOTUNE)

# 2) bump class weight for Black-grass
cls_to_id = {c:i for i, c in enumerate(class_names)}
cw2 = class_weights.copy()
cw2[cls_to_id['Black-grass']] = float(cw2[cls_to_id['Black-grass']]) * 2.0  # try 2x first

# 3) very small LR, keep same loss and metrics
opt3 = tf.keras.optimizers.Adam(8e-6)
model.compile(
    optimizer=opt3,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")]
)

earlyb = tf.keras.callbacks.EarlyStopping(monitor="val_acc", mode="max",
                                          patience=2, restore_best_weights=True)

history = model.fit(
    train_ds_mix,
    validation_data=val_ds,
    epochs=2,
    class_weight=cw2,
    callbacks=[earlyb],
    verbose=1
)

print("Eval:", [round(x, 4) for x in model.evaluate(val_ds, verbose=0)])

# 4) quick per-class check with focus on Black-grass
y_true, y_prob = [], []
for xb, yb in val_ds:
    y_true.append(yb.numpy())
    y_prob.append(model.predict(xb, verbose=0))

y_true = np.concatenate(y_true, 0)
y_pred = np.argmax(np.concatenate(y_prob, 0), 1)

k = cls_to_id['Black-grass']
p, r, f1, _ = precision_recall_fscore_support(y_true.argmax(1), y_pred, labels=[k], average=None)
print({'Black-grass': {'precision': round(float(p[0]),4),
                       'recall': round(float(r[0]),4),
                       'f1': round(float(f1[0]),4)}})



# Slightly stronger push for Black-grass and lower label smoothing
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

# 1) bump class weight to 3x for Black-grass
cls_to_id = {c:i for i, c in enumerate(class_names)}
cw3 = class_weights.copy()
cw3[cls_to_id['Black-grass']] = float(cw3[cls_to_id['Black-grass']]) * 3.0

# 2) tiny LR, lower label smoothing
opt4 = tf.keras.optimizers.Adam(8e-6)
model.compile(
    optimizer=opt4,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")]
)

history = model.fit(
    train_ds_mix,                 # the mixed dataset you just built
    validation_data=val_ds,
    epochs=1,                     # single epoch nudge
    class_weight=cw3,
    verbose=1
)

print("Eval:", [round(x, 4) for x in model.evaluate(val_ds, verbose=0)])

# quick Black-grass check
y_true, y_prob = [], []
for xb, yb in val_ds:
    y_true.append(yb.numpy())
    y_prob.append(model.predict(xb, verbose=0))
y_true = np.concatenate(y_true, 0)
y_pred = np.argmax(np.concatenate(y_prob, 0), 1)

k = cls_to_id['Black-grass']
p, r, f1, _ = precision_recall_fscore_support(y_true.argmax(1), y_pred, labels=[k], average=None)
print({'Black-grass': {'precision': round(float(p[0]),4),
                       'recall': round(float(r[0]),4),
                       'f1': round(float(f1[0]),4)}})



# Post-processing: only predict "Black-grass" if prob >= p_min and margin >= m_min
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report

cls_to_id = {c:i for i, c in enumerate(class_names)}
k_bg = cls_to_id['Black-grass']

# 1) collect probs on val set
y_true, y_prob = [], []
for xb, yb in val_ds:
    y_true.append(yb.numpy())
    y_prob.append(model.predict(xb, verbose=0))
y_true = np.concatenate(y_true, 0)
y_prob = np.concatenate(y_prob, 0)

# 2) grid search small set of thresholds
p_grid = [0.50, 0.55, 0.60, 0.65, 0.70]
m_grid = [0.05, 0.10, 0.15, 0.20, 0.25]

def apply_rule(probs, p_min, m_min):
    top1 = probs.argmax(1)
    top1p = probs.max(1)
    # second-best probs
    part = probs.copy()
    part[np.arange(part.shape[0]), top1] = -1.0
    top2p = part.max(1)
    # start from argmax predictions
    y_hat = top1.copy()
    # where top1 is Black-grass but not confident enough, switch to second best
    bad = (top1 == k_bg) & ((top1p < p_min) | ((top1p - top2p) < m_min))
    # switch to second best class for those
    y_hat[bad] = np.where(bad, part.argmax(1), y_hat)
    return y_hat

best = None
for p_min in p_grid:
    for m_min in m_grid:
        y_hat = apply_rule(y_prob, p_min, m_min)
        p, r, f1, _ = precision_recall_fscore_support(y_true.argmax(1), y_hat, labels=[k_bg], average=None)
        f1_bg = float(f1[0])
        acc = (y_hat == y_true.argmax(1)).mean()
        if best is None or f1_bg > best['f1_bg']:
            best = {'p_min': p_min, 'm_min': m_min, 'f1_bg': f1_bg, 'acc': acc}

print("Best BG rule:", best)

# 3) report full metrics with the chosen rule
y_hat = apply_rule(y_prob, best['p_min'], best['m_min'])
print("Accuracy with rule:", round((y_hat == y_true.argmax(1)).mean(), 4))

rep = classification_report(y_true.argmax(1), y_hat, target_names=class_names, digits=4)
print(rep)

# Also print Black-grass line clearly
p, r, f1, _ = precision_recall_fscore_support(y_true.argmax(1), y_hat, labels=[k_bg], average=None)
print({'Black-grass': {'precision': round(float(p[0]),4),
                       'recall': round(float(r[0]),4),
                       'f1': round(float(f1[0]),4)}})



# 6) Post-processing: only predict 'Black-grass' if confident, else switch to 2nd best

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report

# 6.1 collect probs and labels from val_ds
y_true, y_prob = [], []
for xb, yb in val_ds:
    y_true.append(yb.numpy())
    y_prob.append(model.predict(xb, verbose=0))

y_true = np.concatenate(y_true, axis=0)          # shape [N, C], one-hot
y_prob = np.concatenate(y_prob, axis=0)          # shape [N, C], softmax probs

k_bg = class_names.index('Black-grass')          # target class index

def second_best(probs):
    top1  = probs.argmax(axis=1)                 # [N]
    top1p = probs.max(axis=1)                    # [N]
    tmp   = probs.copy()
    tmp[np.arange(tmp.shape[0]), top1] = -1.0    # remove top1
    top2  = tmp.argmax(axis=1)                   # [N]
    top2p = tmp.max(axis=1)                      # [N]
    return top1, top1p, top2, top2p

def apply_rule(probs, p_min, m_min):
    """Return y_hat labels [N] after applying confidence rule for Black-grass."""
    top1, top1p, top2, top2p = second_best(probs)
    y_hat = top1.copy()
    # low confidence if prob below p_min OR margin below m_min
    low_conf = (top1p < p_min) | ((top1p - top2p) < m_min)
    # only intervene on Black-grass predictions that are low confidence
    bad = (top1 == k_bg) & low_conf               # [N] boolean mask
    y_hat[bad] = top2[bad]                        # assign elementwise, shapes match
    return y_hat

# 6.2 small grid search for thresholds
p_grid = np.array([0.50, 0.55, 0.60, 0.65, 0.70])
m_grid = np.array([0.05, 0.10, 0.15, 0.20, 0.25])

best = None
for p in p_grid:
    for m in m_grid:
        y_hat = apply_rule(y_prob, p, m)
        acc = (y_hat == y_true.argmax(1)).mean()
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_true.argmax(1), y_hat, labels=[k_bg], average=None
        )
        f1_bg = float(f1[0])
        if best is None or f1_bg > best['f1_bg']:
            best = {'p_min': p, 'm_min': m, 'f1_bg': f1_bg, 'acc': acc}

print('Best rule:', best)

# 6.3 final report with chosen thresholds
y_hat = apply_rule(y_prob, best['p_min'], best['m_min'])
print('Accuracy with rule:', round((y_hat == y_true.argmax(1)).mean(), 4))
print(classification_report(y_true.argmax(1), y_hat, target_names=class_names, digits=4))



# === Inspect Black-grass errors (one cell) ===
import numpy as np, itertools
import matplotlib.pyplot as plt
from PIL import Image

# 1) Get predictions again to be explicit
y_true, y_prob = [], []
for xb, yb in val_ds:
    y_true.append(yb.numpy())
    y_prob.append(model.predict(xb, verbose=0))

y_true = np.concatenate(y_true, axis=0)      # [N, C]
y_prob = np.concatenate(y_prob, axis=0)      # [N, C]
true_lbl = y_true.argmax(1)                  # [N]

# 2) Reuse your best thresholds
k_bg = class_names.index('Black-grass')
def second_best(probs):
    top1  = probs.argmax(1)
    top1p = probs.max(1)
    tmp   = probs.copy()
    tmp[np.arange(tmp.shape[0]), top1] = -1.0
    top2  = tmp.argmax(1)
    top2p = tmp.max(1)
    return top1, top1p, top2, top2p

def apply_rule(probs, p_min, m_min):
    top1, top1p, top2, top2p = second_best(probs)
    y_hat = top1.copy()
    low_conf = (top1p < p_min) | ((top1p - top2p) < m_min)
    bad = (top1 == k_bg) & low_conf
    y_hat[bad] = top2[bad]
    return y_hat

y_hat = apply_rule(y_prob, p_min=0.50, m_min=0.25)

# 3) Indices for FP and FN of Black-grass
fp_idx = np.where((y_hat == k_bg) & (true_lbl != k_bg))[0]     # predicted BG, actually not BG
fn_idx = np.where((true_lbl == k_bg) & (y_hat != k_bg))[0]     # actually BG, predicted other

# 4) Print counts by class for both FP and FN
from collections import Counter
fp_true = Counter(class_names[c] for c in true_lbl[fp_idx])
fn_pred = Counter(class_names[c] for c in y_hat[fn_idx])

print("FP (predicted Black-grass but actually):")
for k,v in sorted(fp_true.items(), key=lambda x: -x[1]): print(f"  {k}: {v}")
print("\nFN (actually Black-grass but predicted as):")
for k,v in sorted(fn_pred.items(), key=lambda x: -x[1]): print(f"  {k}: {v}")

# 5) Visualize a few FP and FN to see patterns
def show_subset(idxs, title, rows=2, cols=6):
    take = idxs[:rows*cols]
    if len(take) == 0:
        print(f"No {title.lower()}.")
        return
    # pull raw images from val_ds deterministically by index
    batch_imgs = list(itertools.islice(val_ds.unbatch().as_numpy_iterator(), max(take)+1))
    imgs = [batch_imgs[i][0] for i in take]  # [H,W,3], float32 0..255
    plt.figure(figsize=(cols*2.2, rows*2.2))
    for j,i in enumerate(take):
        plt.subplot(rows, cols, j+1)
        plt.imshow(np.clip(imgs[j], 0, 255).astype("uint8"))
        plt.title(f"true {class_names[true_lbl[i]]}\npred {class_names[y_hat[i]]}", fontsize=9)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout(); plt.show()

show_subset(fp_idx, "Black-grass False Positives")
show_subset(fn_idx, "Black-grass False Negatives")



# === Adaptive post-rule focused on Loose Silky-bent conflicts ===
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report

# 1) Collect predictions again to be explicit
y_true, y_prob = [], []
for xb, yb in val_ds:
    y_true.append(yb.numpy())
    y_prob.append(model.predict(xb, verbose=0))
y_true = np.concatenate(y_true, axis=0)      # [N, C]
y_prob = np.concatenate(y_prob, axis=0)      # [N, C]
true_lbl = y_true.argmax(1)

# 2) Index helpers
k_bg  = class_names.index('Black-grass')
k_lsb = class_names.index('Loose Silky-bent')

def second_best(probs):
    top1  = probs.argmax(1)
    top1p = probs.max(1)
    tmp   = probs.copy()
    tmp[np.arange(tmp.shape[0]), top1] = -1.0
    top2  = tmp.argmax(1)
    top2p = tmp.max(1)
    return top1, top1p, top2, top2p

def apply_rule_adaptive(probs, p_bg=0.50, m_base=0.25, m_add_lsb=0.15):
    """If pred is Black-grass and second best is LSB, need margin >= m_base + m_add_lsb,
       else need margin >= m_base. Also require prob >= p_bg."""
    top1, top1p, top2, top2p = second_best(probs)
    margin = top1p - top2p
    need_margin = np.where(top2 == k_lsb, m_base + m_add_lsb, m_base)
    y_hat = top1.copy()
    low_conf = (top1p < p_bg) | (margin < need_margin)
    bad = (top1 == k_bg) & low_conf
    y_hat[bad] = top2[bad]
    return y_hat

# 3) Small grid search with constraints
best = None
grid_p     = [0.50, 0.52, 0.55]
grid_mbase = [0.22, 0.25]
grid_madd  = [0.10, 0.15, 0.20]

for p in grid_p:
    for mb in grid_mbase:
        for ma in grid_madd:
            y_hat = apply_rule_adaptive(y_prob, p_bg=p, m_base=mb, m_add_lsb=ma)
            acc = (y_hat == true_lbl).mean()
            pr, rc, f1, _ = precision_recall_fscore_support(true_lbl, y_hat, labels=[k_bg], average=None)
            pr, rc, f1 = float(pr[0]), float(rc[0]), float(f1[0])

            # constraints to protect recall and overall accuracy
            if rc >= 0.80 and acc >= 0.982:
                score = pr  # maximize precision subject to constraints
                cand = {'p': p, 'm_base': mb, 'm_add_lsb': ma, 'acc': acc, 'pr_bg': pr, 'rc_bg': rc, 'f1_bg': f1}
                if best is None or score > best['pr_bg']:
                    best = cand

print("Best adaptive thresholds:", best)

# 4) Report with chosen thresholds
y_hat = apply_rule_adaptive(y_prob, p_bg=best['p'], m_base=best['m_base'], m_add_lsb=best['m_add_lsb'])
from sklearn.metrics import accuracy_score
print("Accuracy:", round(accuracy_score(true_lbl, y_hat), 4))

# Black-grass line
from sklearn.metrics import precision_recall_fscore_support
pr, rc, f1, _ = precision_recall_fscore_support(true_lbl, y_hat, labels=[k_bg], average=None)
print(f"Black-grass  precision={pr[0]:.4f}  recall={rc[0]:.4f}  f1={f1[0]:.4f}")

# Optional short report to ensure other classes did not collapse
from sklearn.metrics import classification_report
print(classification_report(true_lbl, y_hat, target_names=class_names, digits=4))



# Robust adaptive search with fallback
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score

# re-use y_prob, true_lbl, class_names, and helper from the previous cell
k_bg  = class_names.index('Black-grass')
k_lsb = class_names.index('Loose Silky-bent')

def second_best(probs):
    top1  = probs.argmax(1)
    top1p = probs.max(1)
    tmp   = probs.copy()
    tmp[np.arange(tmp.shape[0]), top1] = -1.0
    top2  = tmp.argmax(1)
    top2p = tmp.max(1)
    return top1, top1p, top2, top2p

def apply_rule_adaptive(probs, p_bg=0.50, m_base=0.25, m_add_lsb=0.15):
    top1, top1p, top2, top2p = second_best(probs)
    margin = top1p - top2p
    need_margin = np.where(top2 == k_lsb, m_base + m_add_lsb, m_base)
    y_hat = top1.copy()
    low_conf = (top1p < p_bg) | (margin < need_margin)
    bad = (top1 == k_bg) & low_conf
    y_hat[bad] = top2[bad]
    return y_hat

# broadened grids
grid_p     = [0.48, 0.50, 0.52, 0.55, 0.58]
grid_mbase = [0.20, 0.22, 0.25, 0.28]
grid_madd  = [0.08, 0.12, 0.15, 0.18, 0.22]

candidates = []
best = None

for p in grid_p:
    for mb in grid_mbase:
        for ma in grid_madd:
            y_hat = apply_rule_adaptive(y_prob, p_bg=p, m_base=mb, m_add_lsb=ma)
            acc = (y_hat == true_lbl).mean()
            pr, rc, f1, _ = precision_recall_fscore_support(true_lbl, y_hat, labels=[k_bg], average=None)
            pr, rc, f1 = float(pr[0]), float(rc[0]), float(f1[0])
            cand = {'p': p, 'm_base': mb, 'm_add_lsb': ma,
                    'acc': acc, 'pr_bg': pr, 'rc_bg': rc, 'f1_bg': f1}
            candidates.append(cand)
            # main constraint target
            if rc >= 0.80 and acc >= 0.982:
                if best is None or pr > best['pr_bg']:
                    best = cand

# fallback if constraints never met
fallback_used = False
if best is None:
    fallback_used = True
    # pick the highest precision, then highest recall, then highest accuracy
    candidates.sort(key=lambda d: (d['pr_bg'], d['rc_bg'], d['acc']), reverse=True)
    best = candidates[0]

print("Best adaptive thresholds:", best, "(fallback)" if fallback_used else "")

# apply and report
y_hat = apply_rule_adaptive(y_prob, p_bg=best['p'], m_base=best['m_base'], m_add_lsb=best['m_add_lsb'])
print("Accuracy:", round(accuracy_score(true_lbl, y_hat), 4))
pr, rc, f1, _ = precision_recall_fscore_support(true_lbl, y_hat, labels=[k_bg], average=None)
print(f"Black-grass  precision={pr[0]:.4f}  recall={rc[0]:.4f}  f1={f1[0]:.4f}")
print(classification_report(true_lbl, y_hat, target_names=class_names, digits=4))



# 1) restore best boosted weights, eval cleanly, and save the model

import json, os, tensorflow as tf
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
os.makedirs("/kaggle/working/export_v1", exist_ok=True)

# restore
model.load_weights("best_boost2.keras")  # if your filename differs, use that one

# eval on val_ds
y_true, y_prob = [], []
for xb, yb in val_ds:
    y_true.append(yb.numpy())
    y_prob.append(model.predict(xb, verbose=0))
import numpy as np
y_true = np.concatenate(y_true, 0).argmax(1)
y_prob = np.concatenate(y_prob, 0)
y_hat  = y_prob.argmax(1)

# report
acc = accuracy_score(y_true, y_hat)
print("Accuracy:", round(acc, 4))
print(classification_report(y_true, y_hat, target_names=class_names, digits=4))
k_bg = class_names.index("Black-grass")
pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_hat, labels=[k_bg], average=None)
print(f"Black-grass  precision={pr[0]:.4f}  recall={rc[0]:.4f}  f1={f1[0]:.4f}")

# save model
model.save("/kaggle/working/export_v1/seedlings_model.keras")
tf.saved_model.save(model, "/kaggle/working/export_v1/saved_model")

# save label map
with open("/kaggle/working/export_v1/labels.json", "w") as f:
    json.dump({"classes": class_names}, f)

print("Saved to /kaggle/working/export_v1")



# === targeted fine-tune to lift Black-grass recall ===
import tensorflow as tf, numpy as np, json, os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# use boosted set if it exists, otherwise fall back
try:
    _ = train_ds_boost  # defined earlier
    ds_train = train_ds_boost
    print("Using boosted dataset")
except NameError:
    ds_train = train_ds
    print("Using regular train_ds")

# focal loss for one-hot labels with per-class alpha
@tf.function
def focal_loss_onehot(y_true, y_pred, alpha, gamma=1.5, eps=1e-7):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    ce = -y_true * tf.math.log(y_pred)
    pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
    alpha_vec = tf.reshape(alpha, [1, 1, tf.shape(alpha)[0]])  # broadcast to [B,1,C]
    alpha_w = tf.reduce_sum(alpha_vec * y_true, axis=-1)       # [B,1]
    fl = tf.pow(1.0 - pt, gamma) * tf.reduce_sum(ce, axis=-1, keepdims=False)
    return tf.reduce_mean(alpha_w * fl)

# build alpha vector, give extra weight to 'Black-grass'
k_bg = class_names.index("Black-grass")
alpha = np.ones(len(class_names), dtype="float32")
alpha[k_bg] = 2.5  # try 2.5x
alpha_tf = tf.constant(alpha, dtype=tf.float32)

# compile with tiny LR
opt = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=opt,
              loss=lambda y_true, y_pred: focal_loss_onehot(y_true, y_pred, alpha_tf, gamma=1.5),
              metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])

# early stop guard
cb = tf.keras.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=1, restore_best_weights=True)

history = model.fit(
    ds_train,
    validation_data=val_ds,
    epochs=2,
    callbacks=[cb],
    verbose=1,
)

# evaluate
y_true, y_prob = [], []
for xb, yb in val_ds:
    y_true.append(yb.numpy()); y_prob.append(model.predict(xb, verbose=0))
y_true = np.concatenate(y_true, 0).argmax(1)
y_prob = np.concatenate(y_prob, 0)
y_hat  = y_prob.argmax(1)

acc = accuracy_score(y_true, y_hat)
pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_hat, labels=[k_bg], average=None)
print("Accuracy:", round(acc, 4))
print(f"Black-grass precision={pr[0]:.4f}  recall={rc[0]:.4f}  f1={f1[0]:.4f}")
print(classification_report(y_true, y_hat, target_names=class_names, digits=4))

# save v2
os.makedirs("/kaggle/working/export_v2", exist_ok=True)
model.save("/kaggle/working/export_v2/seedlings_model_v2.keras")
with open("/kaggle/working/export_v2/labels.json", "w") as f:
    json.dump({"classes": class_names}, f)
print("Saved to /kaggle/working/export_v2")



import tensorflow as tf
import numpy as np

# pick boosted ds if you already created it, else fallback
try:
    _ = train_ds_boost
    ds_train = train_ds_boost
    print("Using boosted dataset")
except NameError:
    ds_train = train_ds
    print("Using regular train_ds")

# compile with standard CCE + small LR
opt = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=opt,
              loss="categorical_crossentropy",
              metrics=[tf.keras.metrics.CategoricalAccuracy(name="val_acc")])

# up-weight Black-grass
cls_to_id = {c:i for i,c in enumerate(class_names)}
w = {i: 1.0 for i in range(len(class_names))}
w[cls_to_id["Black-grass"]] = 3.0   # start with 3x, tune if needed

early = tf.keras.callbacks.EarlyStopping(monitor="val_acc", mode="max",
                                         patience=1, restore_best_weights=True)

history = model.fit(
    ds_train,
    validation_data=val_ds,
    epochs=2,
    class_weight=w,
    callbacks=[early],
    verbose=1
)



import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# vectorized predictions
probs = model.predict(val_ds, verbose=0)
y_hat = probs.argmax(1)
y_true = np.concatenate([yb.numpy() for _, yb in val_ds], axis=0).argmax(1)

k_bg = class_names.index("Black-grass")

acc = accuracy_score(y_true, y_hat)
pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_hat, labels=[k_bg], average=None)

print("Acc:", round(acc, 4))
print(f"Black-grass  precision={pr[0]:.4f}  recall={rc[0]:.4f}  f1={f1[0]:.4f}")
print(classification_report(y_true, y_hat, target_names=class_names, digits=4))



import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# 1) Keras sanity check, should be ~0.9463 like the fit log
loss, k_acc = model.evaluate(val_ds, verbose=0)
print("keras val_acc:", round(k_acc, 4))

# 2) Vectorized, order-aligned predictions and labels
probs = model.predict(val_ds, verbose=0)                 # shape [N, C]
y_true = np.concatenate([np.argmax(y.numpy(), 1) for _, y in val_ds], axis=0)  # shape [N]
y_hat  = probs.argmax(1)

print("len(probs):", len(probs), "len(y_true):", len(y_true))  # must match

# 3) Full report
acc = accuracy_score(y_true, y_hat)
print("Acc:", round(acc, 4))

k_bg = class_names.index("Black-grass")
pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_hat, labels=[k_bg], average=None)
print(f"Black-grass  precision={pr[0]:.4f}  recall={rc[0]:.4f}  f1={f1[0]:.4f}")

print(classification_report(y_true, y_hat, target_names=class_names, digits=4))



import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

y_true_chunks, y_hat_chunks = [], []

for xb, yb in val_ds:                       # one pass, aligned
    probs_b = model.predict(xb, verbose=0)  # [B, C]
    y_hat_chunks.append(probs_b.argmax(1))
    y_true_chunks.append(yb.numpy().argmax(1))

y_true = np.concatenate(y_true_chunks)
y_hat  = np.concatenate(y_hat_chunks)

print("lens:", len(y_true), len(y_hat))
print("pred dist:", np.bincount(y_hat, minlength=len(class_names)))

acc = accuracy_score(y_true, y_hat)
print("Acc:", round(acc, 4))

k_bg = class_names.index("Black-grass")
pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_hat, labels=[k_bg], average=None)
print(f"Black-grass precision={pr[0]:.4f} recall={rc[0]:.4f} f1={f1[0]:.4f}")

print(classification_report(y_true, y_hat, target_names=class_names, digits=4))



import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# collect probs and labels in one aligned pass
y_true_chunks, y_prob_chunks = [], []
for xb, yb in val_ds:
    y_prob_chunks.append(model.predict(xb, verbose=0))
    y_true_chunks.append(yb.numpy().argmax(1))

y_true = np.concatenate(y_true_chunks)
y_prob = np.concatenate(y_prob_chunks)
N, C = y_prob.shape

# top1 and top2 indices and probabilities
order = np.argsort(y_prob, axis=1)
top2_idx = order[:, -2]
top1_idx = order[:, -1]
top1_p   = y_prob[np.arange(N), top1_idx]
top2_p   = y_prob[np.arange(N), top2_idx]
margin   = top1_p - top2_p

# thresholds chosen from earlier grid that gave strong BG f1
p_min      = 0.48
m_base     = 0.80
m_add_lsb  = 0.20
k_bg       = class_names.index('Black-grass')

# start with standard argmax prediction
y_hat = top1_idx.copy()

# rule 1, if predicted BG but not confident, fall back to second best
mask_bg_uncertain = (y_hat == k_bg) & ((top1_p < p_min) | (margin < m_base))
y_hat[mask_bg_uncertain] = top2_idx[mask_bg_uncertain]

# rule 2, if BG is runner-up and the gap is tiny, promote to BG
mask_bg_promote = (y_hat != k_bg) & (top2_idx == k_bg) & (margin < m_add_lsb)
y_hat[mask_bg_promote] = k_bg

# report
acc = accuracy_score(y_true, y_hat)
pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_hat, labels=[k_bg], average=None)
print("Acc:", round(acc, 4))
print(f"Black-grass precision={pr[0]:.4f} recall={rc[0]:.4f} f1={f1[0]:.4f}")
print(classification_report(y_true, y_hat, target_names=class_names, digits=4))





