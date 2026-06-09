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

print(os.listdir("/kaggle/input"))



import os
import matplotlib.pyplot as plt
import cv2

# Corrected path based on newly added dataset
base_path = "/kaggle/input/plantdisease/PlantVillage"

# Get disease categories
categories = os.listdir(base_path)
print(f"Total classes: {len(categories)}")

# Preview sample images from first 9 categories
plt.figure(figsize=(15, 15))
for i, category in enumerate(categories[:9]):
    img_dir = os.path.join(base_path, category)
    img_path = os.path.join(img_dir, os.listdir(img_dir)[0])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.title(category)
    plt.axis('off')
plt.tight_layout()
plt.show()



import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



data_dir = "/kaggle/input/plantdisease/PlantVillage"
img_height, img_width = 224, 224  # MobileNet input size
batch_size = 32



# Data generators for training & validation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 80% train, 20% validation
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)



base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()



history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5  # You can increase if needed
)



# Save the Keras model
model.save("crop_disease_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open("crop_disease_model.tflite", "wb") as f:
    f.write(tflite_model)



pip install gtts


