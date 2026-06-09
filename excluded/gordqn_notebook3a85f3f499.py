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
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
import shutil
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping
import random


with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')


with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')


file_list = os.listdir('/kaggle/working')
print(file_list)


# 解凍済みの画像フォルダ
img_dir = '/kaggle/working/train'

# 画像ファイル名一覧（JPEGのみ）
all_images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

# 猫と犬のファイルを分ける
cat_images = [f for f in all_images if f.startswith('cat')]
dog_images = [f for f in all_images if f.startswith('dog')]

# 猫・犬からそれぞれランダムに3枚ずつ
sample_cats = random.sample(cat_images, 3)
sample_dogs = random.sample(dog_images, 3)

# 画像表示（3枚 × 2行）
plt.figure(figsize=(12, 6))

# 猫画像（1行目）
for i, fname in enumerate(sample_cats):
    img_path = os.path.join(img_dir, fname)
    img = Image.open(img_path)
    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(f"Cat: {fname}")
    plt.axis('off')

# 犬画像（2行目）
for i, fname in enumerate(sample_dogs):
    img_path = os.path.join(img_dir, fname)
    img = Image.open(img_path)
    plt.subplot(2, 3, i + 4)
    plt.imshow(img)
    plt.title(f"Dog: {fname}")
    plt.axis('off')

plt.tight_layout()
plt.show()


IMG_SIZE = 128  # EfficientNetB3に合わせる
BATCH_SIZE = 32


all_files = [(os.path.join(img_dir, fname), 0 if fname.startswith('cat') else 1) for fname in all_images]
random.shuffle(all_files)

# train/valid分割
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

# 画像読み込みと加工
def process_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return img, label

# Dataset生成関数
def create_dataset(file_label_list):
    paths, labels = zip(*file_label_list)
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    ds = ds.map(process_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataset(train_files)
val_ds = create_dataset(val_files)


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # 2クラス分類
])


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


early_stop = EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop]
)


loss, acc = model.evaluate(val_ds)
print(f"✅ 検証データでの精度: {acc:.4f}")


test_dir = '/kaggle/working/test'
test_images = sorted([os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.endswith('.jpg')])


def process_test_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return img

# Dataset作成
test_ds = tf.data.Dataset.from_tensor_slices(test_images)
test_ds = test_ds.map(process_test_image).batch(BATCH_SIZE)


preds = model.predict(test_ds)


# 画像IDを取得（ファイル名 '1234.jpg' → 1234）
image_ids = [int(os.path.basename(path).split('.')[0]) for path in test_images]

# 確率（そのまま出力）
probs = preds.reshape(-1)

# 提出用DataFrame作成
submission_df = pd.DataFrame({'id': image_ids, 'label': probs})

# ID順に並び替え（Kaggleの仕様に合わせる）
submission_df = submission_df.sort_values('id')

# CSV出力（小数点はfloatのままでOK）
submission_df.to_csv('/kaggle/working/submission.csv', index=False)

print("✅ 犬である確率を含む submission.csv を出力しました！")

