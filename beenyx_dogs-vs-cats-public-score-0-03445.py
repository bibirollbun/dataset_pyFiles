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


import zipfile

with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')


import os

train_dir = "/kaggle/working/train"
files = os.listdir(train_dir)
print("Total images:", len(files))
print("Example filenames:", files[:5])



from sklearn.model_selection import train_test_split
import shutil
import os

# Step 1: Create target folders
base_dir = "/kaggle/working/dataset"
for folder in ["train/dog", "train/cat", "val/dog", "val/cat"]:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

# Step 2: Get filenames
image_files = os.listdir("/kaggle/working/train")
dog_imgs = [f for f in image_files if "dog" in f]
cat_imgs = [f for f in image_files if "cat" in f]

# Step 3: Train/val split
train_dogs, val_dogs = train_test_split(dog_imgs, test_size=0.1, random_state=42)
train_cats, val_cats = train_test_split(cat_imgs, test_size=0.1, random_state=42)

# Step 4: Move to correct folders
def move_images(images, label, split):
    for img in images:
        src = f"/kaggle/working/train/{img}"
        dst = f"{base_dir}/{split}/{label}/{img}"
        shutil.copy2(src, dst)

move_images(train_dogs, "dog", "train")
move_images(val_dogs, "dog", "val")
move_images(train_cats, "cat", "train")
move_images(val_cats, "cat", "val")


!ls /kaggle/working/dataset/train
!ls /kaggle/working/dataset/val


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Train data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)

# Validation generator — no augmentation!
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow images from directory
train_generator = train_datagen.flow_from_directory(
    '/kaggle/working/dataset/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'  # dog=1, cat=0
)

val_generator = val_datagen.flow_from_directory(
    '/kaggle/working/dataset/val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)


import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # freeze base

# Add custom classifier head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x) 

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)


from tensorflow.keras.applications import (
    EfficientNetB5, EfficientNetB3,
    DenseNet201, Xception,
    ResNet152, MobileNetV2, InceptionV3
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pandas as pd

# Model configurations
models_to_test = [
    {"name": "MobileNetV2", "model": MobileNetV2, "input_size": (160, 160)},
    {"name": "EfficientNetB5", "model": EfficientNetB5, "input_size": (456, 456)},
    # {"name": "EfficientNetB3", "model": EfficientNetB3, "input_size": (300, 300)},
    # {"name": "DenseNet201", "model": DenseNet201, "input_size": (224, 224)},
    # {"name": "Xception", "model": Xception, "input_size": (299, 299)},
    # {"name": "ResNet152", "model": ResNet152, "input_size": (224, 224)},
    # {"name": "InceptionV3", "model": InceptionV3, "input_size": (299, 299)}
]

# Feature settings and learning rates to explore
feature_settings = [
    {"dropout": 0.3, "rotation": 20},
    {"dropout": 0.5, "rotation": 30}
]
learning_rates = [0.001, 0.0005]

results = []

# Experiment loop
for model_cfg in models_to_test:
    for feature_cfg in feature_settings:
        for lr in learning_rates:
            print(f"Testing: {model_cfg['name']} | Dropout: {feature_cfg['dropout']} | Rotation: {feature_cfg['rotation']} | LR: {lr}")

            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=feature_cfg["rotation"],
                zoom_range=0.2,
                horizontal_flip=True
            )
            val_datagen = ImageDataGenerator(rescale=1./255)

            train_generator = train_datagen.flow_from_directory(
                '/kaggle/working/dataset/train',
                target_size=model_cfg["input_size"],
                batch_size=32,
                class_mode='binary'
            )
            val_generator = val_datagen.flow_from_directory(
                '/kaggle/working/dataset/val',
                target_size=model_cfg["input_size"],
                batch_size=32,
                class_mode='binary'
            )

            try:
                base_model = model_cfg["model"](weights='imagenet', include_top=False,
                                                input_shape=(*model_cfg["input_size"], 3))
                base_model.trainable = False

                x = base_model.output
                x = GlobalAveragePooling2D()(x)
                x = Dropout(feature_cfg["dropout"])(x)
                output = Dense(1, activation='sigmoid')(x)

                model = Model(inputs=base_model.input, outputs=output)
                model.compile(optimizer=Adam(learning_rate=lr),
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

                history = model.fit(train_generator, epochs=2,
                                    validation_data=val_generator, verbose=1)

                val_acc = history.history['val_accuracy'][-1]
                val_loss = history.history['val_loss'][-1]

                results.append({
                    "model": model_cfg["name"],
                    "img_size": model_cfg["input_size"],
                    "dropout": feature_cfg["dropout"],
                    "rotation": feature_cfg["rotation"],
                    "lr": lr,
                    "val_acc": round(val_acc, 4),
                    "val_loss": round(val_loss, 4)
                })

            except Exception as e:
                print(f"Failed on {model_cfg['name']} — {e}")
                results.append({
                    "model": model_cfg["name"],
                    "img_size": model_cfg["input_size"],
                    "dropout": feature_cfg["dropout"],
                    "rotation": feature_cfg["rotation"],
                    "lr": lr,
                    "val_acc": None,
                    "val_loss": None,
                    "error": str(e)
                })

# Summary table
df_results1 = pd.DataFrame(results)
df_results1.sort_values("val_acc", ascending=False, inplace=True)
print(df_results1)


# Model configurations
models_to_test = [
    # {"name": "MobileNetV2", "model": MobileNetV2, "input_size": (160, 160)},
    # {"name": "EfficientNetB5", "model": EfficientNetB5, "input_size": (456, 456)},
    {"name": "EfficientNetB3", "model": EfficientNetB3, "input_size": (300, 300)},
    {"name": "DenseNet201", "model": DenseNet201, "input_size": (224, 224)},
    # {"name": "Xception", "model": Xception, "input_size": (299, 299)},
    # {"name": "ResNet152", "model": ResNet152, "input_size": (224, 224)},
    # {"name": "InceptionV3", "model": InceptionV3, "input_size": (299, 299)}
]

# Feature settings and learning rates to explore
feature_settings = [
    {"dropout": 0.3, "rotation": 20},
    {"dropout": 0.5, "rotation": 30}
]
learning_rates = [0.001, 0.0005]

results = []

# Experiment loop
for model_cfg in models_to_test:
    for feature_cfg in feature_settings:
        for lr in learning_rates:
            print(f"Testing: {model_cfg['name']} | Dropout: {feature_cfg['dropout']} | Rotation: {feature_cfg['rotation']} | LR: {lr}")

            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=feature_cfg["rotation"],
                zoom_range=0.2,
                horizontal_flip=True
            )
            val_datagen = ImageDataGenerator(rescale=1./255)

            train_generator = train_datagen.flow_from_directory(
                '/kaggle/working/dataset/train',
                target_size=model_cfg["input_size"],
                batch_size=32,
                class_mode='binary'
            )
            val_generator = val_datagen.flow_from_directory(
                '/kaggle/working/dataset/val',
                target_size=model_cfg["input_size"],
                batch_size=32,
                class_mode='binary'
            )

            try:
                base_model = model_cfg["model"](weights='imagenet', include_top=False,
                                                input_shape=(*model_cfg["input_size"], 3))
                base_model.trainable = False

                x = base_model.output
                x = GlobalAveragePooling2D()(x)
                x = Dropout(feature_cfg["dropout"])(x)
                output = Dense(1, activation='sigmoid')(x)

                model = Model(inputs=base_model.input, outputs=output)
                model.compile(optimizer=Adam(learning_rate=lr),
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

                history = model.fit(train_generator, epochs=2,
                                    validation_data=val_generator, verbose=1)

                val_acc = history.history['val_accuracy'][-1]
                val_loss = history.history['val_loss'][-1]

                results.append({
                    "model": model_cfg["name"],
                    "img_size": model_cfg["input_size"],
                    "dropout": feature_cfg["dropout"],
                    "rotation": feature_cfg["rotation"],
                    "lr": lr,
                    "val_acc": round(val_acc, 4),
                    "val_loss": round(val_loss, 4)
                })

            except Exception as e:
                print(f"Failed on {model_cfg['name']} — {e}")
                results.append({
                    "model": model_cfg["name"],
                    "img_size": model_cfg["input_size"],
                    "dropout": feature_cfg["dropout"],
                    "rotation": feature_cfg["rotation"],
                    "lr": lr,
                    "val_acc": None,
                    "val_loss": None,
                    "error": str(e)
                })

# Summary table
df_results2 = pd.DataFrame(results)
df_results2.sort_values("val_acc", ascending=False, inplace=True)
print(df_results2)


# Model configurations
models_to_test = [
    # {"name": "MobileNetV2", "model": MobileNetV2, "input_size": (160, 160)},
    # {"name": "EfficientNetB5", "model": EfficientNetB5, "input_size": (456, 456)},
    # {"name": "EfficientNetB3", "model": EfficientNetB3, "input_size": (300, 300)},
    # {"name": "DenseNet201", "model": DenseNet201, "input_size": (224, 224)},
    {"name": "Xception", "model": Xception, "input_size": (299, 299)},
    {"name": "ResNet152", "model": ResNet152, "input_size": (224, 224)},
    # {"name": "InceptionV3", "model": InceptionV3, "input_size": (299, 299)}
]

# Feature settings and learning rates to explore
feature_settings = [
    {"dropout": 0.3, "rotation": 20},
    {"dropout": 0.5, "rotation": 30}
]
learning_rates = [0.001, 0.0005]

results = []

# Experiment loop
for model_cfg in models_to_test:
    for feature_cfg in feature_settings:
        for lr in learning_rates:
            print(f"Testing: {model_cfg['name']} | Dropout: {feature_cfg['dropout']} | Rotation: {feature_cfg['rotation']} | LR: {lr}")

            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=feature_cfg["rotation"],
                zoom_range=0.2,
                horizontal_flip=True
            )
            val_datagen = ImageDataGenerator(rescale=1./255)

            train_generator = train_datagen.flow_from_directory(
                '/kaggle/working/dataset/train',
                target_size=model_cfg["input_size"],
                batch_size=32,
                class_mode='binary'
            )
            val_generator = val_datagen.flow_from_directory(
                '/kaggle/working/dataset/val',
                target_size=model_cfg["input_size"],
                batch_size=32,
                class_mode='binary'
            )

            try:
                base_model = model_cfg["model"](weights='imagenet', include_top=False,
                                                input_shape=(*model_cfg["input_size"], 3))
                base_model.trainable = False

                x = base_model.output
                x = GlobalAveragePooling2D()(x)
                x = Dropout(feature_cfg["dropout"])(x)
                output = Dense(1, activation='sigmoid')(x)

                model = Model(inputs=base_model.input, outputs=output)
                model.compile(optimizer=Adam(learning_rate=lr),
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

                history = model.fit(train_generator, epochs=2,
                                    validation_data=val_generator, verbose=1)

                val_acc = history.history['val_accuracy'][-1]
                val_loss = history.history['val_loss'][-1]

                results.append({
                    "model": model_cfg["name"],
                    "img_size": model_cfg["input_size"],
                    "dropout": feature_cfg["dropout"],
                    "rotation": feature_cfg["rotation"],
                    "lr": lr,
                    "val_acc": round(val_acc, 4),
                    "val_loss": round(val_loss, 4)
                })

            except Exception as e:
                print(f"Failed on {model_cfg['name']} — {e}")
                results.append({
                    "model": model_cfg["name"],
                    "img_size": model_cfg["input_size"],
                    "dropout": feature_cfg["dropout"],
                    "rotation": feature_cfg["rotation"],
                    "lr": lr,
                    "val_acc": None,
                    "val_loss": None,
                    "error": str(e)
                })

# Summary table
df_results3 = pd.DataFrame(results)
df_results3.sort_values("val_acc", ascending=False, inplace=True)
print(df_results3)


# Model configurations
models_to_test = [
    # {"name": "MobileNetV2", "model": MobileNetV2, "input_size": (160, 160)},
    # {"name": "EfficientNetB5", "model": EfficientNetB5, "input_size": (456, 456)},
    # {"name": "EfficientNetB3", "model": EfficientNetB3, "input_size": (300, 300)},
    # {"name": "DenseNet201", "model": DenseNet201, "input_size": (224, 224)},
    # {"name": "Xception", "model": Xception, "input_size": (299, 299)},
    # {"name": "ResNet152", "model": ResNet152, "input_size": (224, 224)},
    {"name": "InceptionV3", "model": InceptionV3, "input_size": (299, 299)}
]

# Feature settings and learning rates to explore
feature_settings = [
    {"dropout": 0.3, "rotation": 20},
    {"dropout": 0.5, "rotation": 30}
]
learning_rates = [0.001, 0.0005]

results = []

# Experiment loop
for model_cfg in models_to_test:
    for feature_cfg in feature_settings:
        for lr in learning_rates:
            print(f"Testing: {model_cfg['name']} | Dropout: {feature_cfg['dropout']} | Rotation: {feature_cfg['rotation']} | LR: {lr}")

            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=feature_cfg["rotation"],
                zoom_range=0.2,
                horizontal_flip=True
            )
            val_datagen = ImageDataGenerator(rescale=1./255)

            train_generator = train_datagen.flow_from_directory(
                '/kaggle/working/dataset/train',
                target_size=model_cfg["input_size"],
                batch_size=32,
                class_mode='binary'
            )
            val_generator = val_datagen.flow_from_directory(
                '/kaggle/working/dataset/val',
                target_size=model_cfg["input_size"],
                batch_size=32,
                class_mode='binary'
            )

            try:
                base_model = model_cfg["model"](weights='imagenet', include_top=False,
                                                input_shape=(*model_cfg["input_size"], 3))
                base_model.trainable = False

                x = base_model.output
                x = GlobalAveragePooling2D()(x)
                x = Dropout(feature_cfg["dropout"])(x)
                output = Dense(1, activation='sigmoid')(x)

                model = Model(inputs=base_model.input, outputs=output)
                model.compile(optimizer=Adam(learning_rate=lr),
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

                history = model.fit(train_generator, epochs=2,
                                    validation_data=val_generator, verbose=1)

                val_acc = history.history['val_accuracy'][-1]
                val_loss = history.history['val_loss'][-1]

                results.append({
                    "model": model_cfg["name"],
                    "img_size": model_cfg["input_size"],
                    "dropout": feature_cfg["dropout"],
                    "rotation": feature_cfg["rotation"],
                    "lr": lr,
                    "val_acc": round(val_acc, 4),
                    "val_loss": round(val_loss, 4)
                })

            except Exception as e:
                print(f"Failed on {model_cfg['name']} — {e}")
                results.append({
                    "model": model_cfg["name"],
                    "img_size": model_cfg["input_size"],
                    "dropout": feature_cfg["dropout"],
                    "rotation": feature_cfg["rotation"],
                    "lr": lr,
                    "val_acc": None,
                    "val_loss": None,
                    "error": str(e)
                })

# Summary table
df_results4 = pd.DataFrame(results)
df_results4.sort_values("val_acc", ascending=False, inplace=True)
print(df_results4)


# === STEP 1: UNZIP FILES ===
import zipfile
import os

with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')

with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')

# === STEP 2: SPLIT CAT/DOGS INTO FOLDERS FOR FULL TRAINING ===
import shutil

full_train_dir = "/kaggle/working/full_data"
os.makedirs(full_train_dir + "/cat", exist_ok=True)
os.makedirs(full_train_dir + "/dog", exist_ok=True)

all_images = os.listdir('/kaggle/working/train')

for img in all_images:
    label = "dog" if "dog" in img else "cat"
    shutil.move(f"/kaggle/working/train/{img}", f"{full_train_dir}/{label}/{img}")

# === STEP 3: SETUP IMAGE GENERATOR ===
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (299, 299)
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    full_train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# === STEP 4: BUILD & TRAIN XCEPTION MODEL ===
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Initial training
model.fit(train_generator, epochs=5)

# Optional: Fine-tune
base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=5)

# === STEP 5: MAKE TEST PREDICTIONS ===
import numpy as np
from tensorflow.keras.preprocessing import image
import pandas as pd

test_dir = '/kaggle/working/test'
test_images = sorted(os.listdir(test_dir))  # maintain order

def load_and_prep(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)

preds = []
ids = []

for fname in test_images:
    img_path = os.path.join(test_dir, fname)
    prob = model.predict(load_and_prep(img_path), verbose=0)[0][0]
    preds.append(prob)
    ids.append(fname.split(".")[0])  # remove .jpg



# === STEP 6: CREATE SUBMISSION CSV ===
submission1 = pd.DataFrame({"id": ids, "label": preds})
submission1.to_csv("MSBA.Session01.MayankSingh.Xception.csv", index=False)


# === STEP 1: UNZIP FILES ===
import zipfile, os, shutil

with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')

# === STEP 2: ORGANIZE FULL TRAIN DATA ===
full_train_dir = "/kaggle/working/full_data"
os.makedirs(full_train_dir + "/cat", exist_ok=True)
os.makedirs(full_train_dir + "/dog", exist_ok=True)

for img in os.listdir('/kaggle/working/train'):
    label = "dog" if "dog" in img else "cat"
    shutil.move(f"/kaggle/working/train/{img}", f"{full_train_dir}/{label}/{img}")

# === STEP 3: DATA GENERATOR ===
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (299, 299)
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    full_train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# === STEP 4: INCEPTIONV3 MODEL ===
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Initial training: 5 epochs (frozen)
model.fit(train_generator, epochs=5)

# Fine-tuning: unfreeze & 3 more epochs
base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=3)

# === STEP 5: PREDICT ON TEST SET ===
import numpy as np
from tensorflow.keras.preprocessing import image
import pandas as pd

test_dir = '/kaggle/working/test'
test_images = sorted(os.listdir(test_dir))

def load_and_prep(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)

preds = []
ids = []

for fname in test_images:
    prob = model.predict(load_and_prep(os.path.join(test_dir, fname)), verbose=0)[0][0]
    preds.append(prob)
    ids.append(fname.split(".")[0])


# === STEP 6: GENERATE SUBMISSION FILE ===
submission2 = pd.DataFrame({"id": ids, "label": preds})
submission2.to_csv("MSBA.Session01.MayankSingh.Inception.csv", index=False)


# === SETUP ===
print("Starting setup...")
import zipfile, os, glob, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# === STEP 1: UNZIP DATA ===
print("Unzipping data...")
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')

# === STEP 2: LOAD EB7 FOR FEATURE EXTRACTION ===
print("Loading EfficientNetB7 model...")
IMG_SIZE = (600, 600)
base_model = EfficientNetB7(weights='imagenet', include_top=False, pooling='avg', input_shape=(600, 600, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(image_paths):
    print(f"Extracting features from {len(image_paths)} images...")
    features = []
    for idx, path in enumerate(image_paths):
        if idx % 500 == 0:
            print(f"Processed {idx}/{len(image_paths)} images...")
        img = load_img(path, target_size=IMG_SIZE)
        x = img_to_array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        feat = model.predict(x, verbose=0)
        features.append(feat.squeeze())
    print("Feature extraction complete.")
    return np.array(features)

# === STEP 3: EXTRACT FEATURES FROM TRAIN IMAGES ===
print("Loading and labeling training data...")
train_files = glob.glob("/kaggle/working/train/*.jpg")
train_labels = [1 if 'dog' in fname else 0 for fname in train_files]
X = extract_features(train_files)
y = np.array(train_labels)

# === STEP 4: SPLIT OFF TEST SET FOR FINAL EVALUATION ===
print("Splitting data into train and holdout test sets...")
X_train_cv, X_test_final, y_train_cv, y_test_final = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# === STEP 5: NESTED CV WITH GRID SEARCH FOR LOGISTIC REGRESSION ===
print("Starting GridSearchCV with nested cross-validation...")
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
logreg = LogisticRegression(max_iter=1000)
grid = GridSearchCV(logreg, param_grid, cv=inner_cv, scoring='neg_log_loss', n_jobs=-1, verbose=1)
grid.fit(X_train_cv, y_train_cv)

# === STEP 6: EVALUATE BEST MODEL ON FINAL TEST SET ===
print("Evaluating best model on test set...")
best_lr = grid.best_estimator_
y_pred_proba = best_lr.predict_proba(X_test_final)[:, 1]
y_pred = best_lr.predict(X_test_final)

print("Best Parameters:", grid.best_params_)
print("Test Log Loss:", round(log_loss(y_test_final, y_pred_proba), 4))
print("Classification Report:\n", classification_report(y_test_final, y_pred))

# === CONFUSION MATRIX ===
print("Displaying confusion matrix...")
cm = confusion_matrix(y_test_final, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()



from sklearn.metrics import roc_curve, auc, precision_recall_curve

# === ROC CURVE ===
fpr, tpr, _ = roc_curve(y_test_final, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# === PRECISION-RECALL CURVE ===
precision, recall, _ = precision_recall_curve(y_test_final, y_pred_proba)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Logistic Regression')
plt.legend()
plt.grid()
plt.show()


# === STEP 7: PREDICT ON KAGGLE TEST SET ===
print("Extracting test features and predicting...")

# Get test image paths
test_dir = "/kaggle/working/test"
test_files = sorted(glob.glob(f"{test_dir}/*.jpg"))
test_ids = [os.path.basename(f).split('.')[0] for f in test_files]

# Extract features from test images
X_test_submission = extract_features(test_files)

# Predict probabilities
test_preds = best_lr.predict_proba(X_test_submission)[:, 1]

# Create submission DataFrame
submission = pd.DataFrame({
    "id": test_ids,
    "label": test_preds
})

# Save to CSV
submission_file = "MSBA.SessionX.MayankSingh.EB7_LR.csv"
submission.to_csv(submission_file, index=False)

print(f"Submission file saved as: {submission_file}")


from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
import numpy as np

# === 0. SPLIT DATA ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# === 1. LOGISTIC REGRESSION with GRID SEARCH ===
print("Training Logistic Regression...")
lr_param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}
lr_model = LogisticRegression(max_iter=1000)
lr_cv = GridSearchCV(lr_model, lr_param_grid, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                     scoring='neg_log_loss', verbose=1, n_jobs=-1)
lr_cv.fit(X_train, y_train)
best_lr = lr_cv.best_estimator_
lr_val_pred = best_lr.predict_proba(X_val)[:, 1]
lr_logloss = log_loss(y_val, lr_val_pred)
print(f"Logistic Regression Log Loss: {lr_logloss:.5f}")

# === 2. XGBOOST with GRID SEARCH ===
print("\nTraining XGBoost...")
xgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_cv = GridSearchCV(xgb_model, xgb_param_grid, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                      scoring='neg_log_loss', verbose=1, n_jobs=-1)
xgb_cv.fit(X_train, y_train)
best_xgb = xgb_cv.best_estimator_
xgb_val_pred = best_xgb.predict_proba(X_val)[:, 1]
xgb_logloss = log_loss(y_val, xgb_val_pred)
print(f"XGBoost Log Loss: {xgb_logloss:.5f}")

# === 3. ENSEMBLE BLENDING (GRID SEARCH OVER WEIGHTS) ===
print("\nSearching for best ensemble weights...")
best_score = float('inf')
best_weight = None

for w in np.linspace(0, 1, 21):
    blended_pred = w * lr_val_pred + (1 - w) * xgb_val_pred
    blended_logloss = log_loss(y_val, blended_pred)
    print(f"Weight LR: {w:.2f}, XGB: {1 - w:.2f} => Log Loss: {blended_logloss:.5f}")
    
    if blended_logloss < best_score:
        best_score = blended_logloss
        best_weight = w

print(f"\nBest Ensemble Weight\n LR: {best_weight:.2f}, XGB: {1 - best_weight:.2f} | Log Loss: {best_score:.5f}")


# === SETUP ===
import zipfile, os, glob, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# === UNZIP ===
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')

# === EB7 FEATURE EXTRACTOR ===
IMG_SIZE = (600, 600)
base_model = EfficientNetB7(weights='imagenet', include_top=False, pooling='avg', input_shape=(600, 600, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(image_paths):
    features = []
    for idx, path in enumerate(image_paths):
        if idx % 500 == 0:
            print(f"Processed {idx}/{len(image_paths)}")
        img = load_img(path, target_size=IMG_SIZE)
        x = img_to_array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        feat = model.predict(x, verbose=0)
        features.append(feat.squeeze())
    return np.array(features)

# === EXTRACT FEATURES FOR TRAIN ===
train_files = glob.glob("/kaggle/working/train/*.jpg")
train_labels = [1 if 'dog' in fname else 0 for fname in train_files]
X = extract_features(train_files)
y = np.array(train_labels)

# === TRAIN/VALIDATION SPLIT ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# === LOGISTIC REGRESSION TUNING ===
print("\nTraining Logistic Regression...")
lr_param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']}
lr_model = LogisticRegression(max_iter=1000)
lr_cv = GridSearchCV(lr_model, lr_param_grid, cv=StratifiedKFold(3, shuffle=True, random_state=42),
                     scoring='neg_log_loss', n_jobs=-1, verbose=1)
lr_cv.fit(X_train, y_train)
best_lr = lr_cv.best_estimator_
lr_val_pred = best_lr.predict_proba(X_val)[:, 1]
lr_loss = log_loss(y_val, lr_val_pred)
print(f"Log Loss (LR): {lr_loss:.5f}")

# === XGBOOST TUNING ===
print("\nTraining XGBoost...")
xgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_cv = GridSearchCV(xgb_model, xgb_param_grid, cv=StratifiedKFold(3, shuffle=True, random_state=42),
                      scoring='neg_log_loss', n_jobs=-1, verbose=1)
xgb_cv.fit(X_train, y_train)
best_xgb = xgb_cv.best_estimator_
xgb_val_pred = best_xgb.predict_proba(X_val)[:, 1]
xgb_loss = log_loss(y_val, xgb_val_pred)
print(f"Log Loss (XGB): {xgb_loss:.5f}")

# === ENSEMBLE GRID SEARCH ===
print("\nGrid search for best ensemble weights...")
best_score = float('inf')
best_weight = None
for w in np.linspace(0, 1, 21):
    blended = w * lr_val_pred + (1 - w) * xgb_val_pred
    loss = log_loss(y_val, blended)
    print(f"LR: {w:.2f} | XGB: {1-w:.2f} => Log Loss: {loss:.5f}")
    if loss < best_score:
        best_score = loss
        best_weight = w

print(f"\nBest Blending Weight LR: {best_weight:.2f} | XGB: {1-best_weight:.2f} | Log Loss: {best_score:.5f}")

# === PREDICT ON TEST SET & SUBMIT ===
print("\nExtracting test features and generating submission...")
test_files = sorted(glob.glob("/kaggle/working/test/*.jpg"))
test_ids = [os.path.basename(f).split('.')[0] for f in test_files]
X_test_submission = extract_features(test_files)

lr_test_pred = best_lr.predict_proba(X_test_submission)[:, 1]
xgb_test_pred = best_xgb.predict_proba(X_test_submission)[:, 1]
ensemble_test_pred = best_weight * lr_test_pred + (1 - best_weight) * xgb_test_pred

submission = pd.DataFrame({"id": test_ids, "label": ensemble_test_pred})
submission_file = "MSBA.Session01.MayankSingh_ensemble.csv"
submission.to_csv(submission_file, index=False)
print(f"Submission saved: {submission_file}")


from sklearn.ensemble import StackingClassifier

base_learners = [
    ('lr', LogisticRegression(C=0.02, solver='lbfgs', max_iter=10000)),
    ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4, subsample=0.85, use_label_encoder=False, eval_metric='logloss'))
]

meta_learner = LogisticRegression(max_iter=1000)
stacked_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=3, passthrough=True)
stacked_model.fit(X_train, y_train)

# === VALIDATION PERFORMANCE ===
val_pred_proba = stacked_model.predict_proba(X_val)[:, 1]
val_logloss = log_loss(y_val, val_pred_proba)
print(f"Stacked Model Log Loss on Validation Set: {val_logloss:.5f}")


# === PREDICT ON TEST DATA ===
test_files = sorted(glob.glob("/kaggle/working/test/*.jpg"))
test_ids = [os.path.basename(f).split('.')[0] for f in test_files]
X_test_submission = extract_features(test_files)
test_preds = stacked_model.predict_proba(X_test_submission)[:, 1]

# === GENERATE SUBMISSION FILE ===
submission = pd.DataFrame({"id": test_ids, "label": test_preds})
submission_file = "MSBA.Session01.MayankSingh_stacked.csv"
submission.to_csv(submission_file, index=False)
print(f"Submission saved: {submission_file}")


# === SETUP ===
import zipfile, os, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# === UNZIP ===
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')

# === EB3 MODEL ===
IMG_SIZE = (300, 300)
base_model = EfficientNetB3(weights='imagenet', include_top=False, pooling='avg', input_shape=(300, 300, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(image_paths):
    features = []
    for idx, path in enumerate(image_paths):
        if idx % 500 == 0:
            print(f"Processed {idx}/{len(image_paths)}")
        img = load_img(path, target_size=IMG_SIZE)
        x = img_to_array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        feat = model.predict(x, verbose=0)
        features.append(feat.squeeze())
    return np.array(features)

# === EXTRACT TRAIN FEATURES ===
train_files = glob.glob("/kaggle/working/train/*.jpg")
train_labels = [1 if 'dog' in fname else 0 for fname in train_files]
X = extract_features(train_files)
y = np.array(train_labels)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# === LOGISTIC REGRESSION (EXTENSIVE GRID) ===
print("Training Logistic Regression...")
lr_param_grid = {
    'C': np.logspace(-4, 2, 10),  # C from 0.0001 to 100
    'penalty': ['l2'],
    'solver': ['lbfgs']
}
lr_model = LogisticRegression(max_iter=10000)
lr_cv = GridSearchCV(lr_model, lr_param_grid, cv=StratifiedKFold(3, shuffle=True, random_state=42),
                     scoring='neg_log_loss', n_jobs=-1, verbose=1)
lr_cv.fit(X_train, y_train)
best_lr = lr_cv.best_estimator_
lr_val_pred = best_lr.predict_proba(X_val)[:, 1]
print(f"Log Loss (LR): {log_loss(y_val, lr_val_pred):.5f}")

# === XGBOOST (NORMAL GRID) ===
print("Training XGBoost...")
xgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8,1.0]
}
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_cv = GridSearchCV(xgb_model, xgb_param_grid, cv=StratifiedKFold(3, shuffle=True, random_state=42),
                      scoring='neg_log_loss', n_jobs=-1, verbose=1)
xgb_cv.fit(X_train, y_train)
best_xgb = xgb_cv.best_estimator_
xgb_val_pred = best_xgb.predict_proba(X_val)[:, 1]
print(f"Log Loss (XGB): {log_loss(y_val, xgb_val_pred):.5f}")

# === GRID SEARCH OVER BLENDING WEIGHTS ===
print("Blending models...")
best_score = float('inf')
best_wt = None

for w in np.linspace(0, 1, 21):
    blended = w * lr_val_pred + (1 - w) * xgb_val_pred
    loss = log_loss(y_val, blended)
    print(f"LR: {w:.2f} | XGB: {1-w:.2f} => Log Loss: {loss:.5f}")
    if loss < best_score:
        best_score = loss
        best_wt = w

print(f"\nBest Blend → LR: {best_wt:.2f} | XGB: {1-best_wt:.2f} | Log Loss: {best_score:.5f}")



# === TEST PREDICTION AND SUBMISSION ===
print("Extracting test features...")
test_files = sorted(glob.glob("/kaggle/working/test/*.jpg"))
test_ids = [os.path.basename(f).split('.')[0] for f in test_files]
X_test_submission = extract_features(test_files)

lr_test_pred = best_lr.predict_proba(X_test_submission)[:, 1]
xgb_test_pred = best_xgb.predict_proba(X_test_submission)[:, 1]
final_test_pred = best_wt * lr_test_pred + (1 - best_wt) * xgb_test_pred

submission = pd.DataFrame({"id": test_ids, "label": final_test_pred})
submission.to_csv("MSBA.SessionX.MayankSingh_EB3_ensemble.csv", index=False)
print("Submission file saved.")


import zipfile, os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ConvNeXtBase
from tensorflow.keras.applications.convnext import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# === UNZIP ===
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')

# === ORGANIZE FOLDERS ===
import shutil
train_dir = '/kaggle/working/train'
base_dir = '/kaggle/working/dataset'
for label in ['dog', 'cat']:
    os.makedirs(f"{base_dir}/train/{label}", exist_ok=True)
    os.makedirs(f"{base_dir}/val/{label}", exist_ok=True)

from sklearn.model_selection import train_test_split
all_files = os.listdir(train_dir)
dog_files = [f for f in all_files if f.startswith("dog")]
cat_files = [f for f in all_files if f.startswith("cat")]
train_dog, val_dog = train_test_split(dog_files, test_size=0.1, random_state=42)
train_cat, val_cat = train_test_split(cat_files, test_size=0.1, random_state=42)

for f in train_dog: shutil.copy2(f"{train_dir}/{f}", f"{base_dir}/train/dog/{f}")
for f in val_dog: shutil.copy2(f"{train_dir}/{f}", f"{base_dir}/val/dog/{f}")
for f in train_cat: shutil.copy2(f"{train_dir}/{f}", f"{base_dir}/train/cat/{f}")
for f in val_cat: shutil.copy2(f"{train_dir}/{f}", f"{base_dir}/val/cat/{f}")

# === DATA AUGMENTATION ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    f"{base_dir}/train", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)
val_gen = val_datagen.flow_from_directory(
    f"{base_dir}/val", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

# === ConvNeXtV2 MODEL ===
base_model = ConvNeXtBase(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze for now

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# === TRAIN ===
history = model.fit(train_gen, validation_data=val_gen, epochs=5)

# Optionally unfreeze and fine-tune:
# base_model.trainable = True
# model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(train_gen, validation_data=val_gen, epochs=3)






import zipfile
import os

# Force re-extraction to /kaggle/working/
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')

# Check the result
print("Total files in /kaggle/working/test:", len(os.listdir("/kaggle/working/test")) if os.path.exists("/kaggle/working/test") else "Folder missing")
print("Total JPGs in /kaggle/working:", len([f for f in os.listdir("/kaggle/working") if f.endswith('.jpg')]))



import glob
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image

# === LOAD TEST IMAGES ===
test_dir = '/kaggle/working/test'
test_files = sorted(glob.glob(f"{test_dir}/*.jpg"))
test_ids = [os.path.basename(f).split('.')[0] for f in test_files]

# === Predict for each test image ===
preds = []

for idx, filepath in enumerate(test_files):
    if idx % 500 == 0:
        print(f"Processing {idx}/{len(test_files)}...")
    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    prob = model.predict(img_array, verbose=0)[0][0]
    preds.append(prob)

# === CREATE SUBMISSION ===
submission = pd.DataFrame({
    "id": test_ids,
    "label": preds
})

submission_file = "MSBA.Session01.MayankSingh_ConvNeXt.csv"
submission.to_csv(submission_file, index=False)
print(f"Submission saved: {submission_file}")


IMG_SIZE = (224, 224)
base_model = ConvNeXtBase(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

test_dir = '/kaggle/working/test'
test_files = sorted(glob.glob(f"{test_dir}/*.jpg"))
test_ids = [os.path.basename(f).split('.')[0] for f in test_files]

print(f"Total test images found: {len(test_files)}")

preds = []
for idx, fpath in enumerate(test_files):
    if idx % 500 == 0:
        print(f"Predicting {idx}/{len(test_files)}...")
    img = image.load_img(fpath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    prob = model.predict(img_array, verbose=0)[0][0]
    preds.append(prob)

# === CREATE SUBMISSION ===
submission = pd.DataFrame({
    "id": test_ids,
    "label": preds
})

submission_file = "MSBA.Session01.MayankSingh_ConvNeXt_raw.csv"
submission.to_csv(submission_file, index=False)
print(f"Submission saved: {submission_file}")

