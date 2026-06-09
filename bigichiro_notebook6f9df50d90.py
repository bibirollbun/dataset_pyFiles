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
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop

# =======================
# 1. zipファイルを解凍
# =======================
input_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/"
work_dir = "/kaggle/working/"

with zipfile.ZipFile(input_dir + "train.zip", "r") as zip_ref:
    zip_ref.extractall(work_dir + "train/")

with zipfile.ZipFile(input_dir + "test.zip", "r") as zip_ref:
    zip_ref.extractall(work_dir + "test/")

# =======================
# 2. データセットの整理
# =======================
# trainフォルダ構成: train/cat.123.jpg, dog.456.jpg
original_train_dir = os.path.join(work_dir, "train", "train")  # zipの中に 'train/' サブフォルダがある
base_dir = os.path.join(work_dir, "data_split")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

for split in [train_dir, val_dir]:
    os.makedirs(os.path.join(split, "cats"), exist_ok=True)
    os.makedirs(os.path.join(split, "dogs"), exist_ok=True)

# 画像ファイルをラベルごとに分割
filenames = os.listdir(original_train_dir)
cat_files = [f for f in filenames if f.startswith('cat')]
dog_files = [f for f in filenames if f.startswith('dog')]

random.seed(42)
random.shuffle(cat_files)
random.shuffle(dog_files)

# 2000 train / 1000 val for each class
for i, fname in enumerate(cat_files):
    src = os.path.join(original_train_dir, fname)
    if i < 2000:
        dst = os.path.join(train_dir, "cats", fname)
    elif i < 3000:
        dst = os.path.join(val_dir, "cats", fname)
    shutil.copy(src, dst)

for i, fname in enumerate(dog_files):
    src = os.path.join(original_train_dir, fname)
    if i < 2000:
        dst = os.path.join(train_dir, "dogs", fname)
    elif i < 3000:
        dst = os.path.join(val_dir, "dogs", fname)
    shutil.copy(src, dst)

# =======================
# 3. データジェネレータの定義
# =======================
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# =======================
# 4. DenseNetモデル構築
# =======================
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(150,150,3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=RMSprop(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# =======================
# 5. 学習
# =======================
history = model.fit(
    train_gen,
    epochs=3,
    validation_data=val_gen
)

# =======================
# 6. 学習過程を可視化
# =======================
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.title('Accuracy')
plt.legend()
plt.show()

# =======================
# 7. テスト画像を予測
# =======================
test_dir = os.path.join(work_dir, "test", "test")  # 解凍後のパス
test_files = os.listdir(test_dir)
test_df = pd.DataFrame({'filename': test_files})

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    test_dir,
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=(150,150),
    batch_size=32,
    shuffle=False
)

preds = model.predict(test_generator)
test_df['label'] = preds
test_df['label'] = test_df['label'].apply(lambda x: 1 if x > 0.5 else 0)

# =======================
# 8. 提出用ファイル生成
# =======================
submission = pd.DataFrame({
    'id': test_df['filename'].str.extract('(\d+)')[0].astype(int),
    'label': test_df['label']
})
submission = submission.sort_values('id')
submission.to_csv('/kaggle/working/submission.csv', index=False)


