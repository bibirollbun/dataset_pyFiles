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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import shutil # クラス分けに必要
import glob # ファイルのソートに必要
import re
from sklearn.metrics import accuracy_score # 評価


IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
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
    rescale=1./255,
    validation_split=VALIDATION_SPLIT,
    rotation_range=30,            # 回転
    width_shift_range=0.2,        # 横移動
    height_shift_range=0.2,       # 縦移動
    shear_range=0.2,              # シアー変形
    zoom_range=0.3,               # ズーム
    horizontal_flip=True,         # 水平方向の反転
    brightness_range=[0.6, 1.4],
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    '/kaggle/working/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    '/kaggle/working/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)


def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)), # 画像化から特徴を抽出する
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2), # 重要な情報を残しつつ、画像サイズを圧縮
            
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
            
        layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
            
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)), # 中間特徴を学習し、分析精度を向上させる
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid') # sigmoidにより0-1の確率を算出
    ])
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

model = build_model()
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)


pred = model.predict(val_generator)
pred = (pred > 0.5).astype(int).flatten()
label = val_generator.classes
acc = accuracy_score(pred, label)

df = pd.DataFrame({'acc': [acc]})
df.to_csv('accuracy.csv', index=False)


# テストデータファイルの解凍
zip_file = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'
out_dir = '/kaggle/working'

with zipfile.ZipFile(zip_file, 'r') as file:
    file.extractall(out_dir)

# ファイル一覧を所得
test_dir = '/kaggle/working/test'
test_files = os.listdir(test_dir)

# 数値順にソートする関数
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

# 数値順にソート
sorted_files = sorted(test_files, key=extract_number)

# 画像ファイルのフルパスリスト
image_path_list = [os.path.join(test_dir, image) for image in sorted_files]

# 前処理
images = [image.img_to_array(image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))) / 255.0 for path in image_path_list]
batch = np.array(images)

# 推論
pred = model.predict(batch)

# IDの取得
ids = [os.path.splitext(os.path.basename(path))[0] for path in image_path_list]

# ラベルの取得
labels = np.clip(pred, 1e-6, 1 - 1e-6).ravel()

# CSVで保存
df = pd.DataFrame({'id': ids, 'label': labels})
df.to_csv('submission.csv', index=False)

