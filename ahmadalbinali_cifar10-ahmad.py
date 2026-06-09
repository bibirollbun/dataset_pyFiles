# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install py7zr -q

import os
import math
import random
import glob
import warnings

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



# Reproducibility
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths (Kaggle)
DATA_DIR = "/kaggle/input/cifar10-object-recognition-in-images-zip-file"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sampleSubmission.csv")

# Candidate test directories inside the zip
TEST_DIR_CANDIDATES = [
    os.path.join(DATA_DIR, "train_test"),
    os.path.join(DATA_DIR, "train_test", "test"),
    os.path.join(DATA_DIR, "train_test", "test", "test"),
]

CKPT_PATH = "/kaggle/working/best_model_loss.h5"


# Load CIFAR-10 from Keras
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize images
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Convert labels to one-hot
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

print("Train:", x_train.shape, y_train.shape)
print("Val/Test:", x_test.shape, y_test.shape)



def conv_block(x, filters, stride=1):
    y = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    return layers.Activation('relu')(y)

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
    return layers.Activation('relu')(out)

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

model = build_resnet18()
optimizer = AdamW(learning_rate=3e-4, weight_decay=1e-5, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()



# Augmentation: shifts + horizontal flip
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Callbacks
checkpoint_cb = ModelCheckpoint(CKPT_PATH, monitor="val_loss", save_best_only=True, verbose=1)
early_stop_cb  = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1)

def cosine_annealing(epoch):
    max_epochs = 60
    initial_lr = 3e-4
    return 0.5 * initial_lr * (1 + math.cos(math.pi * epoch / max_epochs))

lr_scheduler_cb = LearningRateScheduler(cosine_annealing)



BATCH_SIZE = 128
EPOCHS = 60

if not os.path.exists(CKPT_PATH):
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(x_test, y_test),
        steps_per_epoch=len(x_train)//BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, early_stop_cb, lr_scheduler_cb],
        verbose=1
    )
else:
    print("Checkpoint found; skipping training.")



def robust_load_model():
    candidates = [CKPT_PATH, "best_model_loss.h5", "/kaggle/working/best_model.h5"]
    for p in candidates:
        if os.path.exists(p):
            return tf.keras.models.load_model(p)
    savedmodels = glob.glob("/kaggle/working/**/saved_model.pb", recursive=True)
    if savedmodels:
        return tf.keras.models.load_model(os.path.dirname(savedmodels[0]))
    raise FileNotFoundError("Best model checkpoint not found.")

best_model = robust_load_model()
print("Loaded best model.")

# Numeric evaluation on CIFAR-10 test split
loss, accuracy = best_model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")




# Read sample submission
sample = pd.read_csv(SAMPLE_SUB_PATH)
id_list = sample["id"].tolist()

# Resolve image path for each ID
def resolve_img_path(img_id):
    for base in TEST_DIR_CANDIDATES:
        for ext in (".png", ".jpg"):
            p = os.path.join(base, f"{img_id}{ext}")
            if os.path.exists(p):
                return p
    return None

paths = [resolve_img_path(i) for i in id_list]
if any(p is None for p in paths):
    missing = sum(1 for p in paths if p is None)
    raise FileNotFoundError(f"{missing} test images missing.")

# Create tf.data pipeline
def load_tf(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [32, 32])

ds = tf.data.Dataset.from_tensor_slices(paths)
ds = ds.map(load_tf, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(1024).prefetch(tf.data.AUTOTUNE)

# Predict and assemble submission
probs = best_model.predict(ds, verbose=1)
pred_idx = np.argmax(probs, axis=1)
label_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
pred_labels = [label_names[i] for i in pred_idx]

submission = pd.DataFrame({"id": id_list, "label": pred_labels})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved:", submission.shape)


