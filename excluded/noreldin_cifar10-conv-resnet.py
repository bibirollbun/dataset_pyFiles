!pip install py7zr


import shutil, os

# ðŸš¨ CAREFUL: This deletes everything in /kaggle/working
shutil.rmtree("/kaggle/working", ignore_errors=True)
os.makedirs("/kaggle/working", exist_ok=True)




import os, glob
import numpy as np
import pandas as pd
from PIL import Image

import py7zr 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Paths ---
WORK_DIR = "/kaggle/working"
INPUT_DIR = "/kaggle/input/cifar-10"
TRAIN_7Z = f"{INPUT_DIR}/train.7z"
TEST_7Z  = f"{INPUT_DIR}/test.7z"
TRAIN_CSV = f"{INPUT_DIR}/trainLabels.csv"

TRAIN_DIR = f"{WORK_DIR}/train"
TEST_DIR  = f"{WORK_DIR}/test"

# --- Extract train/test ---
print("Extracting train...")
with py7zr.SevenZipFile(TRAIN_7Z, mode='r') as z:
    z.extractall(path=TRAIN_DIR)

print("Extracting test...")
with py7zr.SevenZipFile(TEST_7Z, mode='r') as z:
    z.extractall(path=TEST_DIR)

print("Train files:", len(glob.glob(f"{TRAIN_DIR}/train/*.png")))
print("Test files:", len(glob.glob(f"{TEST_DIR}/test/*.png")))

# --- Load labels ---
TRAIN_DIR = f"{TRAIN_DIR}/train"
TEST_DIR = f"{TEST_DIR}/test"
df_train = pd.read_csv(TRAIN_CSV)
CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
label2id = {c:i for i,c in enumerate(CLASS_NAMES)}
id2label = {i:c for c,i in label2id.items()}
df_train['label_id'] = df_train['label'].map(label2id)


# --- Load train images into memory ---
def load_images(df, img_dir):
    X = []
    y = []
    for _, row in df.iterrows():
        img_id = row['id']
        label = row['label_id']
        path = os.path.join(img_dir, f"{img_id}.png")
        img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        X.append(img)
        y.append(label)
    return np.array(X), np.array(y)

X, y = load_images(df_train, TRAIN_DIR)
print("Train data:", X.shape, y.shape)

# --- Train/Val split ---
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# --- Preprocessing & Augmentation ---
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow(X_train, y_train, batch_size=152)
val_gen   = val_datagen.flow(X_val, y_val, batch_size=152)


# --- ResNet Model (with residual connections) ---
from tensorflow.keras import layers, models

def residual_block(x, filters, downsample=False):
    shortcut = x
    stride = 2 if downsample else 1

    x = layers.Conv2D(filters, (3,3), strides=stride, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, (3,3), strides=1, padding='same')(x)

    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), strides=stride, padding='same')(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet_small(input_shape=(32,32,3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3,3), strides=1, padding='same', activation='relu')(inputs)

    # Stage 1
    x = residual_block(x, 32)
    x = residual_block(x, 32)

    # Stage 2
    x = residual_block(x, 64, downsample=True)
    x = residual_block(x, 64)

    # Stage 3
    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

model = build_resnet_small()
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


# --- Train ---
EPOCHS = 50
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# --- Evaluate on val set (proxy for Kaggle score) ---
val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print("Validation Accuracy:", val_acc)

# --- Predict on Kaggle test set (300k images) ---
test_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in glob.glob(f"{TEST_DIR}/*.png")]
test_ids.sort()

def load_test_image(img_id):
    path = os.path.join(TEST_DIR, f"{img_id}.png")
    img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    return img

X_test = np.array([load_test_image(i) for i in test_ids])
X_test = X_test.astype('float32') / 255.0

y_pred = model.predict(X_test, batch_size=256)
labels = [id2label[np.argmax(p)] for p in y_pred]

sub = pd.DataFrame({"id": test_ids, "label": labels})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv:", sub.shape)



# ======================================================
# CIFAR-10 Kaggle Competition (Keras ConvNet, random init)
# ======================================================
# Dataset files:
#   /kaggle/input/cifar-10/train.7z
#   /kaggle/input/cifar-10/test.7z
#   /kaggle/input/cifar-10/trainLabels.csv
#
# Output:
#   /kaggle/working/submission.csv
# ======================================================

import os, glob
import numpy as np
import pandas as pd
from PIL import Image

import py7zr 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Paths ---
WORK_DIR = "/kaggle/working"
INPUT_DIR = "/kaggle/input/cifar-10"
TRAIN_7Z = f"{INPUT_DIR}/train.7z"
TEST_7Z  = f"{INPUT_DIR}/test.7z"
TRAIN_CSV = f"{INPUT_DIR}/trainLabels.csv"

TRAIN_DIR = f"{WORK_DIR}/train"
TEST_DIR  = f"{WORK_DIR}/test"

# --- Extract train/test ---
print("Extracting train...")
with py7zr.SevenZipFile(TRAIN_7Z, mode='r') as z:
    z.extractall(path=TRAIN_DIR)

print("Extracting test...")
with py7zr.SevenZipFile(TEST_7Z, mode='r') as z:
    z.extractall(path=TEST_DIR)

print("Train files:", len(glob.glob(f"{TRAIN_DIR}/train/*.png")))
print("Test files:", len(glob.glob(f"{TEST_DIR}/test/*.png")))

# --- Load labels ---
TRAIN_DIR = f"{TRAIN_DIR}/train"
TEST_DIR = f"{TEST_DIR}/test"
df_train = pd.read_csv(TRAIN_CSV)
CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
label2id = {c:i for i,c in enumerate(CLASS_NAMES)}
id2label = {i:c for c,i in label2id.items()}
df_train['label_id'] = df_train['label'].map(label2id)

# --- Load train images into memory ---
def load_images(df, img_dir):
    X = []
    y = []
    for _, row in df.iterrows():
        img_id = row['id']
        label = row['label_id']
        path = os.path.join(img_dir, f"{img_id}.png")
        img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        X.append(img)
        y.append(label)
    return np.array(X), np.array(y)

X, y = load_images(df_train, TRAIN_DIR)
print("Train data:", X.shape, y.shape)

# --- Train/Val split ---
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# --- Preprocessing & Augmentation ---
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow(X_train, y_train, batch_size=128)
val_gen   = val_datagen.flow(X_val, y_val, batch_size=128)

# --- ConvNet Model ---
def build_convnet():
    model = keras.Sequential([
        layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    return model

model = build_convnet()
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# --- Train ---
EPOCHS = 50
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# --- Evaluate on val set (proxy for Kaggle score) ---
val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print("Validation Accuracy:", val_acc)

# --- Predict on Kaggle test set (300k images) ---
test_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in glob.glob(f"{TEST_DIR}/*.png")]
test_ids.sort()

def load_test_image(img_id):
    path = os.path.join(TEST_DIR, f"{img_id}.png")
    img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    return img

X_test = np.array([load_test_image(i) for i in test_ids])
X_test = X_test.astype('float32') / 255.0

y_pred = model.predict(X_test, batch_size=256)
labels = [id2label[np.argmax(p)] for p in y_pred]

sub = pd.DataFrame({"id": test_ids, "label": labels})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv:", sub.shape)




model.summary




