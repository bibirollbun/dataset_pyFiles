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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# GPUの有無確認（CPUでも問題なし）
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# =========================
# 1. データの解凍と整理
# =========================

base_dir = '/kaggle/working/dogs-vs-cats-data'
train_zip = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# 解凍
with zipfile.ZipFile(train_zip, 'r') as zip_ref:
    zip_ref.extractall(base_dir)

original_train_dir = os.path.join(base_dir, 'train')

# 画像ファイルをクラス別フォルダに分ける（cats, dogs）
split_train_dir = os.path.join(base_dir, 'split')
train_cats_dir = os.path.join(split_train_dir, 'train/cats')
train_dogs_dir = os.path.join(split_train_dir, 'train/dogs')
val_cats_dir = os.path.join(split_train_dir, 'val/cats')
val_dogs_dir = os.path.join(split_train_dir, 'val/dogs')

for path in [train_cats_dir, train_dogs_dir, val_cats_dir, val_dogs_dir]:
    os.makedirs(path, exist_ok=True)

# データ分割（train:val = 90:10）
all_filenames = os.listdir(original_train_dir)
cats = [f for f in all_filenames if f.startswith('cat')]
dogs = [f for f in all_filenames if f.startswith('dog')]

random.seed(42)
random.shuffle(cats)
random.shuffle(dogs)

def split_and_copy(images, train_dir, val_dir, split_ratio=0.9):
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    for f in train_images:
        shutil.copy(os.path.join(original_train_dir, f), os.path.join(train_dir, f))
    for f in val_images:
        shutil.copy(os.path.join(original_train_dir, f), os.path.join(val_dir, f))

split_and_copy(cats, train_cats_dir, val_cats_dir)
split_and_copy(dogs, train_dogs_dir, val_dogs_dir)

# =========================
# 2. モデルと学習設定
# =========================

img_height = 150
img_width = 150
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(split_train_dir, 'train'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(split_train_dir, 'val'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# CNNモデル構築
model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 学習
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)

# =========================
# 3. モデル保存（任意）
# =========================
model.save("/kaggle/working/dogs_vs_cats_model.h5")





# =========================
# 4. テスト画像の準備
# =========================

test_zip = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'
test_dir = '/kaggle/working/test'

# 解凍
with zipfile.ZipFile(test_zip, 'r') as zip_ref:
    zip_ref.extractall(test_dir)

test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    directory='/kaggle/working',  # test/ ディレクトリの親
    classes=['test'],             # test/ ディレクトリ内を指定
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode=None,
    shuffle=False
)

# 推論
pred = model.predict(test_generator, steps=len(test_generator), verbose=1)

import pandas as pd
import numpy as np
import os

# ファイル名からIDを取得
filenames = test_generator.filenames
ids = [int(os.path.basename(name).split('.')[0]) for name in filenames]
labels = (pred > 0.5).astype(int).flatten()

submission = pd.DataFrame({'id': ids, 'label': labels})
submission = submission.sort_values('id')  # id順に並べる（重要）

submission.to_csv('/kaggle/working/submission.csv', index=False)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()



