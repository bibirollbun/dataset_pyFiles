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


original_train_dir = '/kaggle/working/train/train'


base_dir = '/kaggle/working/data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')


for dir_path in [train_dir, val_dir, test_dir]:
    for category in ['dog', 'cat']:
        os.makedirs(os.path.join(dir_path, category), exist_ok=True)

all_filenames = os.listdir(original_train_dir)

# 1回目の分割 → train（90%）と test候補（10%）
train_val_filenames, test_filenames = train_test_split(all_filenames, test_size=0.1, random_state=42)

# 2回目の分割 → train_val をさらに train（90%）と val（10%）に
train_filenames, val_filenames = train_test_split(train_val_filenames, test_size=0.1, random_state=42)

def copy_files(filenames, dst_dir):
    for filename in filenames:
        src = os.path.join(original_train_dir, filename)
        if 'dog' in filename:
            dst = os.path.join(dst_dir, 'dog', filename)
        else:
            dst = os.path.join(dst_dir, 'cat', filename)
        shutil.copy(src, dst)

copy_files(train_filenames, train_dir)
copy_files(val_filenames, val_dir)
copy_files(test_filenames, test_dir)


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


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
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
                    epochs=16,
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


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

# テストデータのディレクトリ
test_dir = '/kaggle/working/data/test'
img_size = 128  # 適宜定義

# Generator の設定
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# 正解ラベル（generatorから取得）
y_true = test_generator.classes

# モデルから予測（generatorを使う）
predicted_probs = model.predict(test_generator, verbose=1).ravel()

# 0.4を閾値にして2クラス分類
y_pred = (predicted_probs >= 0.4).astype(int)

# 指標を計算
acc = accuracy_score(y_true, y_pred)

print(f'Accuracy : {acc:.4f}')


# test_generator からファイル名一覧取得
filenames = test_generator.filenames

# ファイル名の先頭の数字をIDとして抽出（例: '123.jpg' → 123）
ids = []
for fname in filenames:
    base = os.path.basename(fname)
    id_str = base.split('.')[0]
    try:
        ids.append(int(id_str))
    except ValueError:
        ids.append(0)  # 変換できない場合の処理

# 予測ラベルは y_pred に合わせる（0/1 の分類結果）
submission = pd.DataFrame({'id': ids, 'label': y_pred})

# id で昇順ソート
submission.sort_values('id', inplace=True)

# CSVファイルに書き出し（ヘッダー付き、indexなし）
submission.to_csv('/kaggle/working/submission.csv', index=False)

# 最初の10行を表示
print(submission.head(10))

import os
print(os.listdir()) 

