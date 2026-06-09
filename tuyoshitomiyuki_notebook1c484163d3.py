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

# train.zipの解凍
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/train')

# test1.zipの解凍
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/test')


import shutil
import os
from sklearn.model_selection import train_test_split

original_dir = '/kaggle/working/train/train'
base_dir = '/kaggle/working/data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

for category in ['dog', 'cat']:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)

all_filenames = os.listdir(original_dir)
train_filenames, val_filenames = train_test_split(all_filenames, test_size=0.2, random_state=42)

for filename in train_filenames:
    if 'dog' in filename:
        shutil.copy(os.path.join(original_dir, filename), os.path.join(train_dir, 'dog', filename))
    else:
        shutil.copy(os.path.join(original_dir, filename), os.path.join(train_dir, 'cat', filename))

for filename in val_filenames:
    if 'dog' in filename:
        shutil.copy(os.path.join(original_dir, filename), os.path.join(val_dir, 'dog', filename))
    else:
        shutil.copy(os.path.join(original_dir, filename), os.path.join(val_dir, 'cat', filename))


import tensorflow as tf
from tensorflow.keras import layers, models

img_size = 128
batch_size = 517

# Resize + Padding 関数
def resize_and_pad(image, target_size=128):
    shape = tf.shape(image)[:2]
    h, w = shape[0], shape[1]

    # 長辺に合わせてスケーリング
    scale = tf.cast(target_size, tf.float32) / tf.cast(tf.reduce_max([h, w]), tf.float32)
    new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)

    image = tf.image.resize(image, [new_h, new_w])

    # 中央パディング
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    image = tf.image.pad_to_bounding_box(image,
                                         offset_height=pad_h // 2,
                                         offset_width=pad_w // 2,
                                         target_height=target_size,
                                         target_width=target_size)
    return image

#  拡張＋前処理
def preprocess(image, label):
    image = resize_and_pad(image, target_size=img_size)
    image = tf.cast(image, tf.float32) / 255.0

    # 拡張
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)

    return image, label

# バリデーション用（拡張なし）
def preprocess_val(image, label):
    image = resize_and_pad(image, target_size=img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# データセット作成関数
def get_dataset(dir_path, batch_size=32, is_training=True):
    dataset = tf.keras.utils.image_dataset_from_directory(
        dir_path,
        image_size=(256, 256),  # 仮設定、あとで resize_and_pad される
        batch_size=batch_size,
        label_mode='binary'
    )

    if is_training:
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000)
    else:
        dataset = dataset.map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.prefetch(tf.data.AUTOTUNE)

# データセットを取得
train_dataset = get_dataset(train_dir, batch_size=batch_size, is_training=True)
val_dataset = get_dataset(val_dir, batch_size=batch_size, is_training=False)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = 150
batch_size = 1024

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_size, img_size),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

val_generator = val_datagen.flow_from_directory(val_dir,
                                                target_size=(img_size, img_size),
                                                batch_size=batch_size,
                                                class_mode='binary')


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D

model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    GlobalAveragePooling2D(),     # ここをFlatten()の代わりに
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


from tensorflow.keras.callbacks import *

f_log = '/kaggle/working/weight'
es_cb = EarlyStopping(monitor='val_loss', patience=1000, verbose=1, mode="auto")
f_path = os.path.join(f_log, 'weights.' '{epoch:03d}-{loss:.4f}-{val_loss:.4f}.h5')
cp_cb = ModelCheckpoint(filepath=f_path, monitor='val_loss', verbose=1,
                            save_best_only=True, mode="auto", save_weights_only=False)


log = model.fit(train_dataset,
                    epochs=30,
                    validation_data=val_dataset,callbacks=[es_cb, cp_cb], shuffle=True)


import matplotlib.pyplot as plt

# モデルの学習履歴 log（例：model.fit(...) の戻り値）
loss = log.history['loss']
val_loss = log.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo-', label='Training Loss')      # 青い点線
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss') # 赤い点線
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


test_dir = '/kaggle/working/test/test'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    '/kaggle/working/',
    classes=['test'],
    target_size=(img_size, img_size),
    batch_size=1,
    class_mode=None,
    shuffle=False)

predictions = model.predict(test_generator, verbose=1)
predicted_classes = predictions.ravel()

import pandas as pd

filenames = test_generator.filenames
ids = [int(os.path.basename(fname).split('.')[0]) for fname in filenames]

submission = pd.DataFrame({'id': ids, 'label': predicted_classes})
submission.sort_values('id', inplace=True)
submission.to_csv('/kaggle/working/submission.csv', index=False)

