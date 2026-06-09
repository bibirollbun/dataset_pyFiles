import os
print(os.listdir("/kaggle/input"))
print(os.listdir("/kaggle/input/cifar-10"))



# Extract Kaggle competition test set into ./test/
!apt -yqq install libarchive-dev >/dev/null
!pip -q install libarchive >/dev/null

import os, glob, libarchive.public

seven_z = "/kaggle/input/cifar-10/test.7z"
assert os.path.exists(seven_z), "Add the 'cifar-10' competition as an Input first."

if not os.path.exists("test"):
    os.makedirs("test", exist_ok=True)
    cnt = 0
    for _ in libarchive.public.file_pour(seven_z):
        cnt += 1
        if cnt % 10000 == 0:
            print("Extracted:", cnt)
    print("Extraction finished.")
else:
    print("Folder 'test' already exists; skipping.")

# Verify count ~300,000 files
print("Files in test/:", len(glob.glob("test/*")))



# Step 2 — load official CIFAR-10 and make a 5k validation split
import numpy as np
from tensorflow.keras.datasets import cifar10

# 50k train / 10k test from the official dataset
(x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

# 5,000 images for validation
VAL = 5000
x_val   = x_train_full[-VAL:]
y_val   = y_train_full[-VAL:]
x_train = x_train_full[:-VAL]
y_train = y_train_full[:-VAL]

# Flatten labels to shape (N,)
y_train = y_train.squeeze().astype(np.int32)
y_val   = y_val.squeeze().astype(np.int32)
y_test  = y_test.squeeze().astype(np.int32)

CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

print("Train:", x_train.shape, y_train.shape)
print("Val:  ", x_val.shape,   y_val.shape)
print("Test: ", x_test.shape,  y_test.shape)
print("Classes:", CLASS_NAMES)



# Step 3 — tf.data pipeline (NO rotations)
import tensorflow as tf

AUTO       = tf.data.AUTOTUNE
IMG_SIZE   = 32
BATCH_SIZE = 128
SEED       = 42

def augment(img, label):
    # to float in [0,1]
    img = tf.cast(img, tf.float32) / 255.0

    # classic CIFAR trick: pad + random crop
    img = tf.pad(img, [[4,4],[4,4],[0,0]], mode="REFLECT")
    img = tf.image.random_crop(img, [IMG_SIZE, IMG_SIZE, 3], seed=SEED)

    # horizontal flip only (NO rotations)
    img = tf.image.random_flip_left_right(img, seed=SEED)

    # mild color jitter
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

    # small cutout (random erase)
    size = tf.random.uniform([], 6, 12, dtype=tf.int32, seed=SEED)
    cx   = tf.random.uniform([], 0, IMG_SIZE, dtype=tf.int32, seed=SEED)
    cy   = tf.random.uniform([], 0, IMG_SIZE, dtype=tf.int32, seed=SEED)
    y1 = tf.clip_by_value(cy - size // 2, 0, IMG_SIZE)
    x1 = tf.clip_by_value(cx - size // 2, 0, IMG_SIZE)
    y2 = tf.clip_by_value(y1 + size, 0, IMG_SIZE)
    x2 = tf.clip_by_value(x1 + size, 0, IMG_SIZE)
    mask = tf.ones([y2 - y1, x2 - x1, 3], tf.float32)
    pad  = tf.pad(mask, [[y1, IMG_SIZE - y2], [x1, IMG_SIZE - x2], [0,0]])
    img  = img * (1. - pad)

    return img, label

def normalize(img, label):
    return tf.cast(img, tf.float32) / 255.0, label

def make_ds(images, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(10_000, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(augment, num_parallel_calls=AUTO)
    else:
        ds = ds.map(normalize, num_parallel_calls=AUTO)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTO)
    return ds

train_ds = make_ds(x_train, y_train, training=True)
val_ds   = make_ds(x_val,   y_val,   training=False)
test_ds  = make_ds(x_test,  y_test,  training=False)

print("Pipelines ready (NO rotations). One training batch:",
      next(iter(train_ds.take(1)))[0].shape)




# Step 4 — ResNet-18 for CIFAR-10 (random init)
from tensorflow import keras
from tensorflow.keras import layers, regularizers

WEIGHT_DECAY = 5e-4  # standard for CIFAR training

def conv3x3(filters, stride=1):
    return layers.Conv2D(
        filters, 3, strides=stride, padding="same", use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )

def basic_block(x, filters, stride):
    y = conv3x3(filters, stride)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = conv3x3(filters, 1)(y)
    y = layers.BatchNormalization()(y)

    # projection if shape changes
    if x.shape[-1] != filters or stride != 1:
        x = layers.Conv2D(
            filters, 1, strides=stride, use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
        )(x)
        x = layers.BatchNormalization()(x)

    out = layers.Add()([x, y])
    out = layers.ReLU()(out)
    return out

def make_layer(x, filters, blocks, stride):
    x = basic_block(x, filters, stride)
    for _ in range(1, blocks):
        x = basic_block(x, filters, 1)
    return x

def build_resnet18(input_shape=(32,32,3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # CIFAR stem (3x3 conv)
    x = layers.Conv2D(
        64, 3, padding="same", use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 4 stages
    x = make_layer(x,  64, 2, stride=1)  # 32x32
    x = make_layer(x, 128, 2, stride=2)  # 16x16
    x = make_layer(x, 256, 2, stride=2)  # 8x8
    x = make_layer(x, 512, 2, stride=2)  # 4x4

    x = layers.GlobalAveragePooling2D()(x)
    logits = layers.Dense(num_classes, activation=None,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    outputs = layers.Softmax(dtype="float32")(logits)  # numerically stable
    return keras.Model(inputs, outputs, name="ResNet18_CIFAR")

model = build_resnet18()
model.summary()



# ==== Compile & Train ResNet18 on CIFAR-10 ====
from tensorflow import keras
from tensorflow.keras import callbacks

# 1) Optimizer / loss / metric
opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
model.compile(
    optimizer=opt,
    loss=keras.losses.SparseCategoricalCrossentropy(),  # no label smoothing
    metrics=["accuracy"],
)

# 2) Callbacks (save the best model by val_accuracy)
ckpt_path = "/kaggle/working/resnet18_best.keras"
cbs = [
    callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_accuracy", mode="max",
        save_best_only=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5,
        min_lr=1e-5, verbose=1
    ),
    callbacks.EarlyStopping(
        monitor="val_loss", patience=12,
        restore_best_weights=True, verbose=1
    ),
]

# 3) Train
EPOCHS = 60  # we can increase later if needed
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cbs,
    verbose=1,
)

# 4) Quick test-set estimate (should be close to Kaggle score)
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\n[Official CIFAR-10 test] accuracy={test_acc:.4f}  loss={test_loss:.4f}")
print(f"Best model saved to: {ckpt_path}")



# --- Build ordered test set, run inference, and write submission.csv ---

import os, pandas as pd, tensorflow as tf

# 1) Paths (we extracted test images to /kaggle/working/test)
TEST_DIR = "/kaggle/working/test"
assert os.path.isdir(TEST_DIR), f"Test folder not found: {TEST_DIR}"

# 2) Read the sample submission to get the exact required order
sub = pd.read_csv("/kaggle/input/cifar-10/sampleSubmission.csv")

# Make sure ids are ints, then build string filepaths in the right order
sub["id"] = sub["id"].astype(int)
test_paths = [f"{TEST_DIR}/{i}.png" for i in sub["id"]]

# Quick sanity check
missing = sum(not os.path.exists(p) for p in test_paths)
print(f"Found test images (existing paths): {len(test_paths) - missing}/{len(test_paths)}")
assert missing == 0, "Some test image paths were not found. Check TEST_DIR."

# 3) TF Dataset for test (NO augmentation; just decode + normalize to [0,1])
AUTO = tf.data.AUTOTUNE
IMG_SIZE = 32

def load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return img

test_ds = (tf.data.Dataset
           .from_tensor_slices(tf.constant(test_paths))   # <-- STRING tensors
           .map(load_and_preprocess, num_parallel_calls=AUTO)
           .batch(256)
           .prefetch(AUTO))

# 4) Predict
CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
probs = model.predict(test_ds, verbose=1)
pred_ids = tf.argmax(probs, axis=1).numpy()
sub["label"] = [CLASS_NAMES[i] for i in pred_ids]

# 5) Save submission
out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print(sub.head())
print(f"\nSubmission saved to: {out_path}")



# 1) Where things are
COMP_DIR   = "/kaggle/input/cifar-10"              # competition input
TEST_7Z    = f"{COMP_DIR}/test.7z"
TEST_DIR   = "/kaggle/working/test"                 # we will create this
SAMPLE_CSV = f"{COMP_DIR}/sampleSubmission.csv"

# 2) Install extractor (safe to rerun)
!apt -yqq install libarchive-dev >/dev/null
!pip -q install libarchive >/dev/null

# 3) Extract test.7z -> /kaggle/working/test  (only if missing)
import os, glob, libarchive.public

if not os.path.isdir(TEST_DIR):
    cnt = 0
    for _ in libarchive.public.file_pour(TEST_7Z):
        cnt += 1
        if cnt % 10000 == 0:
            print("Extracted:", cnt)
    print("Extraction finished.")
else:
    print("Folder 'test' already exists; skipping.")

print("Files in test/:", len(glob.glob(TEST_DIR + "/*")))
print("Paths OK:", os.path.isdir(TEST_DIR), "->", TEST_DIR)
print("Sample CSV:", SAMPLE_CSV)



!ls -lh /kaggle/working

