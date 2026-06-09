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


import zipfile # zipファイルの解凍に必要
import tensorflow as tf # 機械学習に必要
from tensorflow.keras.preprocessing.image import ImageDataGenerator # データの正規化に必要
from tensorflow.keras import layers, models, regularizers # レイヤークラス, 学習モデル
from tensorflow.keras.preprocessing import image # 画像の前処理に必要
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.applications import EfficientNetV2S # 転移学習
from tensorflow.keras.losses import BinaryCrossentropy
import shutil # クラス分けに必要


IMG_SIZE = 384
BATCH_SIZE = 32
EPOCHS_0 = 10
EPOCHS_1 = 20
VALIDATION_SPLIT = 0.2


zip_file = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'
out_dir = '/kaggle/working'

with zipfile.ZipFile(zip_file, 'r') as file:
    file.extractall(out_dir)


train_dir = '/kaggle/working/train'

cat_dir = os.path.join(train_dir, 'cats')
dog_dir = os.path.join(train_dir, 'dogs')

os.makedirs(cat_dir, exist_ok=True)
os.makedirs(dog_dir, exist_ok=True)

for file_name in os.listdir(train_dir):
    file = os.path.join(train_dir, file_name)
    if not os.path.isfile(file):
        continue

    if file_name.lower().startswith('cat'):
        shutil.move(os.path.join(train_dir, file_name), os.path.join(cat_dir, file_name))
    elif file_name.lower().startswith('dog'):
        shutil.move(os.path.join(train_dir, file_name), os.path.join(dog_dir, file_name))


train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input,
    validation_split=VALIDATION_SPLIT,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    fill_mode='reflect'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input,
    validation_split=VALIDATION_SPLIT
)

train_generator = train_datagen.flow_from_directory(
    '/kaggle/working/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    '/kaggle/working/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)


# モデル
base_model = EfficientNetV2S(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling='avg'
)

x = base_model.output
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs=base_model.input, outputs=outputs)

base_model.trainable = False

# スケジューラー
lr_schedule = CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=len(train_generator) * EPOCHS_0,
    alpha=1e-5
)

# オプティマイザー
optimizer = AdamW(
    learning_rate=lr_schedule,
    weight_decay=1e-5
)

# コンパイル
model.compile(
    loss=BinaryCrossentropy(label_smoothing=0.05),
    optimizer=optimizer,
    metrics=['accuracy']
)


callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
]

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_0,
    callbacks=callbacks
)


# モデル
for layer in base_model.layers[-20:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

# スケジューラー
lr_schedule = CosineDecay(
    initial_learning_rate=1e-5,
    decay_steps=len(train_generator) * EPOCHS_1,
    alpha=1e-6
)

# オプティマイザー
optimizer = AdamW(
    learning_rate=lr_schedule,
    weight_decay=1e-5
)

# コンパイル
model.compile(
    loss=BinaryCrossentropy(label_smoothing=0.05),
    optimizer=optimizer,
    metrics=['accuracy']
)


callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
]

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_1,
    callbacks=callbacks
)


# テストデータファイルの解凍
zip_file = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'
out_dir = '/kaggle/working'

with zipfile.ZipFile(zip_file, 'r') as file:
    file.extractall(out_dir)

# テストデータの前処理
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input)
test_generator = train_datagen.flow_from_directory(
    '/kaggle/working',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False,
    classes=['test']
)

# 推論
pred = model.predict(test_generator, verbose=1)

# 出力データの整理
ids = [int(os.path.splitext(os.path.basename(path))[0]) for path in test_generator.filenames]
labels = np.clip(pred, 1e-6, 1 - 1e-6).ravel()
view = list(zip(ids, labels))
view.sort(key=lambda x: x[0])

# CSV提出
df = pd.DataFrame(view, columns=['id', 'label'])
df.to_csv('submission.csv', index=False)

