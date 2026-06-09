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


import os
import zipfile
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

# Step 1: Unzip dataset
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/train.zip", 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/train_data")

with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/test1.zip", 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/test_data")

# Step 2: Prepare directories
base_dir = '/kaggle/working/cats_vs_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

for category in ['Cat', 'Dog']:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, category), exist_ok=True)

# Step 3: Split into train and val
train_data_path = "/kaggle/working/train_data/train"
all_images = os.listdir(train_data_path)
random.shuffle(all_images)

split_index = int(0.8 * len(all_images))
train_images = all_images[:split_index]
val_images = all_images[split_index:]

for fname in train_images:
    label = 'Dog' if 'dog' in fname else 'Cat'
    shutil.copy(os.path.join(train_data_path, fname),
                os.path.join(train_dir, label, fname))

for fname in val_images:
    label = 'Dog' if 'dog' in fname else 'Cat'
    shutil.copy(os.path.join(train_data_path, fname),
                os.path.join(validation_dir, label, fname))

# Step 4: Data generators
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    validation_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Step 5: VGG16 Model
def build_model(base_model):
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
vgg16_model = build_model(vgg16_base)

vgg16_model.compile(optimizer=Adam(learning_rate=0.0001),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

print("\nğŸ”§ Training VGG16...")
vgg16_model.fit(train_gen, validation_data=val_gen, epochs=3)

# Step 6: VGG19 Model
vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
vgg19_model = build_model(vgg19_base)

vgg19_model.compile(optimizer=Adam(learning_rate=0.0001),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

print("\nğŸ”§ Training VGG19...")
vgg19_model.fit(train_gen, validation_data=val_gen, epochs=3)

# Step 7: Prediction on test images
test_dir = "/kaggle/working/test_data/test1"
test_images = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir)[:10]]

def predict_images(model, image_paths, name="Model"):
    print(f"\nğŸ”� Predictions by {name}:")
    class_names = ['Cat', 'Dog']
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        label = class_names[int(prediction > 0.5)]
        print(f"{os.path.basename(img_path)} -> Predicted: {label} (Score: {prediction:.4f})")

predict_images(vgg16_model, test_images, "VGG16")
predict_images(vgg19_model, test_images, "VGG19")


