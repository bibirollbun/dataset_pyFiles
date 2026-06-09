# ====================================================
# CIFAR-10 | ResNet18 (safe) + Robust Kaggle Submission
# Works with dataset: /kaggle/input/cifar10-object-recognition-in-images-zip-file
# ====================================================
import os, math, random, glob, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import AdamW

warnings.filterwarnings("ignore")

print("TF version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# --------------------------
# Reproducibility
# --------------------------
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --------------------------
# Paths (Kaggle)
# --------------------------
DATA_DIR = "/kaggle/input/cifar10-object-recognition-in-images-zip-file"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sampleSubmission.csv")

# The zip-file dataset sometimes nests test images differently.
# We'll resolve robustly later, but these are the *candidate* folders:
TEST_DIR_CANDIDATES = [
    os.path.join(DATA_DIR, "train_test"),
    os.path.join(DATA_DIR, "train_test", "test"),
    os.path.join(DATA_DIR, "train_test", "test", "test"),
]

CKPT_PATH = "/kaggle/working/best_model_loss.h5"   # where we save/load the trained model

# ====================================================
# 1) Load CIFAR-10 (official Keras) for TRAIN/VAL
# ====================================================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)
print("Train:", x_train.shape, y_train.shape, "  Val/Test:", x_test.shape, y_test.shape)

# ====================================================
# 2) Model: ResNet18-like (from scratch)
# ====================================================
def conv_block(x, filters, stride=1):
    y = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    return y

def residual_block(x, filters, downsample=False):
    stride = 2 if downsample else 1
    y = conv_block(x, filters, stride)
    y = conv_block(y, filters)
    if downsample or x.shape[-1] != filters:
        x = layers.Conv2D(filters, 1, strides=stride, padding='same',
                          kernel_regularizer=regularizers.l2(1e-4),
                          use_bias=False)(x)
        x = layers.BatchNormalization()(x)
    out = layers.add([x, y])
    out = layers.Activation('relu')(out)
    return out

def build_resnet18():
    inputs = layers.Input(shape=(32, 32, 3))
    x = conv_block(inputs, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)

    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 256)

    x = residual_block(x, 512, downsample=True)
    x = residual_block(x, 512)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    return models.Model(inputs, outputs)

# Build/compile model
model = build_resnet18()
optimizer = AdamW(learning_rate=3e-4, weight_decay=1e-5, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# ====================================================
# 3) Data Augmentation (gentle, no rotation)
# ====================================================
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

checkpoint_cb = ModelCheckpoint(
    CKPT_PATH, monitor="val_loss", save_best_only=True, verbose=1
)
early_stop_cb = EarlyStopping(
    monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
)

def cosine_annealing(epoch):
    max_epochs = 60
    initial_lr = 3e-4
    return 0.5 * initial_lr * (1 + math.cos(math.pi * epoch / max_epochs))
lr_scheduler_cb = LearningRateScheduler(cosine_annealing)

# ====================================================
# 4) TRAIN (only if checkpoint does not exist)
# ====================================================
BATCH_SIZE = 128
EPOCHS = 60

if not os.path.exists(CKPT_PATH):
    print("No checkpoint found -> training model...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(x_test, y_test),
        steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, early_stop_cb, lr_scheduler_cb],
        verbose=1
    )
else:
    print("Checkpoint already exists -> skipping training.")

# ====================================================
# 5) Load BEST model
# ====================================================
def robust_load_model():
    # try common candidates and also SavedModel directories if needed
    import glob
    candidates = [CKPT_PATH, "best_model_loss.h5", "/kaggle/working/best_model.h5"]
    for p in candidates:
        if os.path.exists(p):
            print("Loading model from:", p)
            return tf.keras.models.load_model(p)

    savedmodel = glob.glob("/kaggle/working/**/saved_model.pb", recursive=True)
    if savedmodel:
        mdir = os.path.dirname(savedmodel[0])
        print("Loading SavedModel from:", mdir)
        return tf.keras.models.load_model(mdir)

    raise FileNotFoundError(
        "No checkpoint found. Re-run the training cell so best_model_loss.h5 is created."
    )

best_model = robust_load_model()
print("âœ… Model ready.")

# ====================================================
# 6) Build Submission
# ====================================================
assert os.path.exists(SAMPLE_SUB_PATH), "sampleSubmission.csv not found in dataset!"
sample = pd.read_csv(SAMPLE_SUB_PATH)
id_list = sample["id"].tolist()
print("IDs to predict:", len(id_list))

# Resolve a valid test directory and extension per id
def resolve_img_path(img_id):
    # Try each candidate folder and .png/.jpg
    for base in TEST_DIR_CANDIDATES:
        p_png = os.path.join(base, f"{img_id}.png")
        if os.path.exists(p_png):
            return p_png
        p_jpg = os.path.join(base, f"{img_id}.jpg")
        if os.path.exists(p_jpg):
            return p_jpg
    return None

# Try to locate the first few to assert structure is correct early
probes = [resolve_img_path(i) for i in id_list[:10]]
if not all(probes):
    print("âš ï¸�  Some probe paths were not found. Candidate folders tried:")
    for c in TEST_DIR_CANDIDATES:
        print("   -", c)
    # don't crash; maybe deeper IDs existâ€”full pass below will assert

paths = []
missing = 0
for i in id_list:
    p = resolve_img_path(i)
    if p is None:
        missing += 1
    else:
        paths.append(p)

if missing > 0:
    raise FileNotFoundError(
        f"{missing} test images could not be resolved. "
        "Please open the dataset in the right panel and confirm the test folder structure. "
        "We look for: train_test/, train_test/test/, or train_test/test/test/ with .png/.jpg."
    )

print("Resolved test paths:", len(paths))

# tf.data input pipeline: decode -> resize -> [0,1]
def load_tf(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [32, 32])
    return img

BATCH_PRED = 1024
ds = tf.data.Dataset.from_tensor_slices(paths)
ds = ds.map(load_tf, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(BATCH_PRED).prefetch(tf.data.AUTOTUNE)

# Predict (shows progress)
probs = best_model.predict(ds, verbose=1)
pred_idx = np.argmax(probs, axis=1)

# Map idx -> class string that Kaggle expects
label_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
pred_labels = [label_names[i] for i in pred_idx]

submission = pd.DataFrame({"id": id_list, "label": pred_labels})
submission.to_csv("submission.csv", index=False)
print("ğŸ“� Saved submission.csv")
print(submission.head())

# Basic sanity
print("Shape:", submission.shape)
print("Columns:", list(submission.columns))
print("IDs unique?:", submission['id'].is_unique)
print("NaNs per column:", submission.isna().sum().to_dict())
print("Label distribution (top 10):")
print(submission['label'].value_counts().head(10))


