import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hanya tampilkan error, abaikan warning



# Supress warning log dari TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hanya tampilkan error, abaikan warning

import zipfile
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Ekstrak dataset dari ZIP (gunakan dataset dari kompetisi Kaggle: Dogs vs Cats)
with zipfile.ZipFile('../input/dogs-vs-cats/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/tmp/dogs-vs-cats')

base_dir = '/tmp/dogs-vs-cats/train'

# Buat folder training dan validation
train_dir = '/tmp/cats-vs-dogs/training'
val_dir = '/tmp/cats-vs-dogs/validation'
os.makedirs(train_dir + '/cats', exist_ok=True)
os.makedirs(train_dir + '/dogs', exist_ok=True)
os.makedirs(val_dir + '/cats', exist_ok=True)
os.makedirs(val_dir + '/dogs', exist_ok=True)

# Fungsi membagi gambar ke folder training & validation
def split_data(source_dir, train_dir, val_dir, split_size):
    files = [f for f in os.listdir(source_dir) if os.path.getsize(os.path.join(source_dir, f)) > 0]
    random.shuffle(files)
    split_index = int(len(files) * split_size)
    train_files = files[:split_index]
    val_files = files[split_index:]
    
    for f in train_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(val_dir, f))

# Pisahkan file berdasarkan nama
cat_source = [f for f in os.listdir(base_dir) if f.startswith('cat')]
dog_source = [f for f in os.listdir(base_dir) if f.startswith('dog')]

os.makedirs('/tmp/cats_temp', exist_ok=True)
os.makedirs('/tmp/dogs_temp', exist_ok=True)

for f in cat_source:
    shutil.copy(os.path.join(base_dir, f), os.path.join('/tmp/cats_temp', f))
for f in dog_source:
    shutil.copy(os.path.join(base_dir, f), os.path.join('/tmp/dogs_temp', f))

# Split data: 90% train, 10% validation
split_data('/tmp/cats_temp', train_dir + '/cats', val_dir + '/cats', 0.9)
split_data('/tmp/dogs_temp', train_dir + '/dogs', val_dir + '/dogs', 0.9)

# ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Model CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Visualisasi Akurasi & Loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()


