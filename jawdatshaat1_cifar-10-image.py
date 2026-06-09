!pip install -q py7zr
import py7zr



# =========================================================
# CIFAR-10 — ResNet18 (from scratch) + Kaggle Submission
# - Uses py7zr to extract /kaggle/input/cifar-10/*.7z to /kaggle/working/*
# - Trains on official Keras CIFAR-10 (allowed; no pretrained weights)
# - Rotation augmentation included (±15 deg)
# - Predicts on extracted Kaggle test images and saves submission.csv
# =========================================================
import os, glob, math, random, numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import AdamW

print("TensorFlow:", tf.__version__)

# --------------------------
# Reproducibility
# --------------------------
SEED = 1337
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# =========================================================
# Cell 1 — Paths & competition inputs
# =========================================================
# Main competition mount commonly named 'cifar-10'
INPUT_DIR = "/kaggle/input/cifar-10"
assert os.path.exists(INPUT_DIR), f"Expected {INPUT_DIR} to exist as an input on Kaggle."

TRAIN_7Z = os.path.join(INPUT_DIR, "train.7z")
TEST_7Z  = os.path.join(INPUT_DIR, "test.7z")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sampleSubmission.csv")
assert os.path.exists(SAMPLE_SUB), "sampleSubmission.csv not found in the competition input."

# Extraction targets
WK_TRAIN_DIR = "/kaggle/working/train_images"
WK_TEST_DIR  = "/kaggle/working/test_images"
os.makedirs(WK_TRAIN_DIR, exist_ok=True)
os.makedirs(WK_TEST_DIR,  exist_ok=True)

# =========================================================
# Cell 2 — Extract .7z archives with py7zr
# =========================================================
import py7zr

def extract_7z(file_path, out_dir):
    try:
        with py7zr.SevenZipFile(file_path, mode='r') as z:
            z.extractall(out_dir)
        print(f" Extracted: {file_path} -> {out_dir}")
        return True
    except Exception as e:
        print(f" Failed to extract {file_path}: {e}")
        return False

print("=== EXTRACTING COMPETITION DATA ===")
if os.path.exists(TRAIN_7Z):
    extract_7z(TRAIN_7Z, WK_TRAIN_DIR)
if os.path.exists(TEST_7Z):
    extract_7z(TEST_7Z, WK_TEST_DIR)

# After extraction the structure is typically:
# /kaggle/working/train_images/train/*.png   (50,000 files)
# /kaggle/working/test_images/test/*.png     (300,000 files)
# Resolve the inner folders robustly:
def find_inner_dir(root, expected_leaf):
    # Try common layouts: root/<expected_leaf> or root/*/<expected_leaf>
    d1 = os.path.join(root, expected_leaf)
    if os.path.isdir(d1):
        return d1
    # look one level deep
    for d in glob.glob(os.path.join(root, "*")):
        cand = os.path.join(d, expected_leaf)
        if os.path.isdir(cand):
            return cand
    # fall back to root if files are directly inside
    return root

TRAIN_IMG_DIR = find_inner_dir(WK_TRAIN_DIR, "train")
TEST_IMG_DIR  = find_inner_dir(WK_TEST_DIR, "test")
print("Resolved TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
print("Resolved TEST_IMG_DIR :", TEST_IMG_DIR)

# (Optional) quick count
n_train_found = len(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png"))) + len(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg")))
n_test_found  = len(glob.glob(os.path.join(TEST_IMG_DIR, "*.png")))  + len(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
print("Found train imgs:", n_train_found, "| test imgs:", n_test_found)

# =========================================================
# Cell 3 — Load official CIFAR-10 for training/validation
#        (Allowed; same data distribution as competition)
# =========================================================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)
print("CIFAR-10 train:", x_train.shape, y_train.shape, "| test:", x_test.shape, y_test.shape)

# =========================================================
# Cell 4 — Model: ResNet18-like (from scratch, no pretrained)
# =========================================================
def conv_block(x, filters, stride=1, wd=1e-4):
    y = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      kernel_regularizer=regularizers.l2(wd),
                      use_bias=False, kernel_initializer="he_normal")(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    return y

def residual_block(x, filters, downsample=False, wd=1e-4):
    stride = 2 if downsample else 1
    y = conv_block(x, filters, stride=stride, wd=wd)
    y = layers.Conv2D(filters, 3, padding='same',
                      kernel_regularizer=regularizers.l2(wd),
                      use_bias=False, kernel_initializer="he_normal")(y)
    y = layers.BatchNormalization()(y)
    if downsample or x.shape[-1] != filters:
        x = layers.Conv2D(filters, 1, strides=stride, padding='same',
                          kernel_regularizer=regularizers.l2(wd),
                          use_bias=False, kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
    out = layers.Add()([x, y])
    out = layers.ReLU()(out)
    return out

def build_resnet18(input_shape=(32,32,3), num_classes=10, wd=1e-4, drop=0.5):
    inputs = layers.Input(shape=input_shape)
    x = conv_block(inputs, 64, wd=wd)
    x = residual_block(x, 64, wd=wd)
    x = residual_block(x, 64, wd=wd)

    x = residual_block(x, 128, downsample=True, wd=wd)
    x = residual_block(x, 128, wd=wd)

    x = residual_block(x, 256, downsample=True, wd=wd)
    x = residual_block(x, 256, wd=wd)

    x = residual_block(x, 512, downsample=True, wd=wd)
    x = residual_block(x, 512, wd=wd)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(drop)(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           kernel_initializer="he_normal")(x)
    return models.Model(inputs, outputs, name="ResNet18_CIFAR10")

model = build_resnet18()
optimizer = AdamW(learning_rate=3e-4, weight_decay=1e-5, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# =========================================================
# Cell 5 — Augmentation + training callbacks
# (Rotation added here: ±15 degrees)
# =========================================================
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

checkpoint = ModelCheckpoint("best_model_loss.h5", monitor="val_loss",
                             save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=12,
                           restore_best_weights=True, verbose=1)

def cosine_annealing(epoch):
    max_epochs = 60
    initial_lr = 3e-4
    return 0.5 * initial_lr * (1 + math.cos(math.pi * epoch / max_epochs))
lr_scheduler = LearningRateScheduler(cosine_annealing)

# =========================================================
# Cell 6 — Train
# (Let Keras infer steps_per_epoch from generator length)
# =========================================================
BATCH_SIZE = 128
EPOCHS = 60

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop, lr_scheduler],
    verbose=2
)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f" Local CIFAR-10 test accuracy: {acc:.4f}")
print(f" Local CIFAR-10 test loss    : {loss:.4f}")

# =========================================================
# Cell 7 — Load best checkpoint
# =========================================================
best_model = tf.keras.models.load_model("best_model_loss.h5")
print(" Loaded best_model_loss.h5")

# =========================================================
# Cell 8 — Build Kaggle submission from extracted test images
# =========================================================
# Load IDs from sampleSubmission.csv (1..300000)
sample = pd.read_csv(SAMPLE_SUB)
id_list = sample["id"].tolist()
print("Total test IDs:", len(id_list))

# Resolve test directory (png/jpg)
def resolve_img_path(base_dir, img_id):
    p_png = os.path.join(base_dir, f"{img_id}.png")
    if os.path.exists(p_png): return p_png
    p_jpg = os.path.join(base_dir, f"{img_id}.jpg")
    if os.path.exists(p_jpg): return p_jpg
    raise FileNotFoundError(f"Missing image id={img_id} in {base_dir}")

test_paths = [resolve_img_path(TEST_IMG_DIR, i) for i in id_list]

# tf.data pipeline for fast batched inference
def load_img_tf(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [32, 32])  # safety
    return img

batch_size = 1024
ds = tf.data.Dataset.from_tensor_slices(test_paths)
ds = ds.map(load_img_tf, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Predict
probs = best_model.predict(ds, verbose=1)
pred_idx = np.argmax(probs, axis=1)
label_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
pred_labels = [label_names[i] for i in pred_idx]

# Save submission
submission = pd.DataFrame({"id": id_list, "label": pred_labels})
submission.to_csv("submission.csv", index=False)
print(" Saved submission.csv")
print(submission.head())

# Quick checks
print("Shape:", submission.shape)
print("Unique IDs:", submission['id'].is_unique)
print("ID range:", submission['id'].min(), "to", submission['id'].max())
print(submission['label'].value_counts().head())



import os
print("CWD:", os.getcwd())         # should be /kaggle/working
assert os.path.exists("submission.csv")
from IPython.display import FileLink
FileLink("submission.csv")


