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
import os

# Giải nén train.zip
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/train_raw')

# Giải nén test1.zip
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/test1.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/test')



import shutil
import random

# Đường dẫn nguồn
source_dir = '/kaggle/working/train_raw/train'
split_dir = '/kaggle/working/data_split'

# Tạo thư mục mới
for split in ['train', 'val']:
    for cls in ['cats', 'dogs']:
        os.makedirs(os.path.join(split_dir, split, cls), exist_ok=True)

# Danh sách ảnh
all_files = os.listdir(source_dir)
cat_files = [f for f in all_files if f.startswith("cat")]
dog_files = [f for f in all_files if f.startswith("dog")]

# Shuffle và split
random.seed(42)
random.shuffle(cat_files)
random.shuffle(dog_files)

split_cat = int(0.8 * len(cat_files))
split_dog = int(0.8 * len(dog_files))

train_cats, val_cats = cat_files[:split_cat], cat_files[split_cat:]
train_dogs, val_dogs = dog_files[:split_dog], dog_files[split_dog:]

# Copy file
def copy_files(files, src, dst):
    for f in files:
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))

copy_files(train_cats, source_dir, os.path.join(split_dir, 'train', 'cats'))
copy_files(val_cats, source_dir, os.path.join(split_dir, 'val', 'cats'))
copy_files(train_dogs, source_dir, os.path.join(split_dir, 'train', 'dogs'))
copy_files(val_dogs, source_dir, os.path.join(split_dir, 'val', 'dogs'))



import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(split_dir, 'train'),
    image_size=(180, 180),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(split_dir, 'val'),
    image_size=(180, 180),
    batch_size=32
)



from tensorflow.keras.preprocessing import image
import numpy as np

test_dir = '/kaggle/working/test/test1'
test_filenames = sorted(os.listdir(test_dir))  # Đảm bảo thứ tự

def load_test_images(image_dir, image_size=(150, 150)):
    images = []
    for fname in test_filenames:
        img_path = os.path.join(image_dir, fname)
        img = image.load_img(img_path, target_size=image_size)
        img_array = image.img_to_array(img)
        images.append(img_array)
    return np.array(images) / 255.0  # Chuẩn hóa

test_ds = load_test_images(test_dir)
print("Shape of test images:", test_ds.shape)



#data processing

#Resize image to 150x150
img_size = (150,150)

#One-hot encoding label
num_classes = 2

def preprocess(image,label):
    #resize theo yeu cau
    image = tf.image.resize(image, img_size)
    #dua ve [0,1]
    image = image / 255.0

    #one-hot label 
    label = tf.one_hot(label, depth=num_classes)
    
    return image, label

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(preprocess).shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds  = val_ds.map(preprocess).prefetch(buffer_size=AUTOTUNE)


for image, label in train_ds.take(1):
    print("Image shape:", image.shape)
    print("Label (one-hot):", label.numpy())


from tensorflow import keras
from tensorflow.keras import layers

# Tạo layer data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),          # Lật ngang ngẫu nhiên
    layers.RandomRotation(0.1),               # Xoay ±10%
    layers.RandomZoom(0.1),                   # Zoom ±10%
    layers.RandomContrast(0.1)                # Tăng/giảm độ tương phản
])


#model CNN

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([

    data_augmentation,

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu',padding='same'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu',padding='same'),
    layers.MaxPooling2D(),

    layers.Conv2D(256, (3,3), activation='relu',padding='same'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(2, activation='softmax')
])

model.summary()


#compile

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  #da one_hot
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs=30
)


#plot accurancy and loss
import matplotlib.pyplot as plt

# Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()



# Loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()



from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Base model: MobileNetV2 (Transfer Learning)
base_model = MobileNetV2(input_shape=(150, 150, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False  

model = tf.keras.Sequential([
    base_model,    
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

model.summary()

#Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)



history_tl = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
)



import matplotlib.pyplot as plt

# Accuracy
plt.figure()
plt.plot(history_tl.history['accuracy'], label='Train Accuracy')
plt.plot(history_tl.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Loss
plt.figure()
plt.plot(history_tl.history['loss'], label='Train Loss')
plt.plot(history_tl.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


#fine tune

base_model.trainable = True

fine_tune_at = 100 
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False



model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)



early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                             patience=3, min_lr=1e-3)

# Train tiếp với fine-tuning
history_ftune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,  # hoặc nhiều hơn nếu cần
    callbacks=[early_stopping,reduce_lr]
)



# Accuracy
plt.figure()
plt.plot(history_ftune.history['accuracy'], label='Train Accuracy')
plt.plot(history_ftune.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Loss
plt.figure()
plt.plot(history_ftune.history['loss'], label='Train Loss')
plt.plot(history_ftune.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()




