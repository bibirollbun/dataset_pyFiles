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
#from tensorflow.keras.applications import InceptionV3
#from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
#from tensorflow.keras.models import Model
import shutil # クラス分けに必要
import random
import glob # ファイルのソートに必要
import re


hyper_epochs = 10
hyper_split_rate = 0.8


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)), # 画像化から特徴を抽出する
    layers.MaxPooling2D(2, 2), # 重要な情報を残しつつ、画像サイズを圧縮
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(), # 畳み込み層の出力を、1次元のベクトルに変換
    layers.Dropout(0.5), #追加
    layers.Dense(512, activation='relu'), # 中間特徴を学習し、分析精度を向上させる
    layers.Dense(1, activation='sigmoid') # sigmoidにより0-1の確率を算出
])


"""base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(299, 299, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)"""


model.compile(
    loss='binary_crossentropy', # 損失関数
    optimizer='adam', # 最適化アルゴリズム
    metrics=['accuracy'] # 評価指標 accuracy：予測が正解だった割合
)


"""model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)"""


zip_file = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'
out_dir = '/kaggle/working'

with zipfile.ZipFile(zip_file, 'r') as file:
    file.extractall(out_dir)


# 画像ファイル一覧を取得し、ランダム化
learn_dir = '/kaggle/working/train'
learn_file_names = [file for file in os.listdir(learn_dir) if os.path.isfile(os.path.join(learn_dir, file))]
random.shuffle(learn_file_names)

# 分岐点を計算
split_point = int(len(learn_file_names) * hyper_split_rate)
train_file_names = learn_file_names[:split_point]
val_file_names = learn_file_names[split_point:]

# フォルダ作成
train_dir = '/kaggle/working/train/train'
val_dir = '/kaggle/working/train/val'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# ファイルの移動
for file_name in train_file_names:
    shutil.move(os.path.join(learn_dir, file_name), os.path.join(train_dir, file_name))

for file_name in val_file_names:
    shutil.move(os.path.join(learn_dir, file_name), os.path.join(val_dir, file_name))


# クラスを分別する関数
def devide_class(any_dir):
    cat_dir = os.path.join(any_dir, 'cats')
    dog_dir = os.path.join(any_dir, 'dogs')

    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(dog_dir, exist_ok=True)

    for file_name in os.listdir(any_dir):
        file = os.path.join(any_dir, file_name)
        if not os.path.isfile(file):
            continue

        if file_name.lower().startswith('cat'):
            shutil.move(os.path.join(any_dir, file_name), os.path.join(cat_dir, file_name))
        elif file_name.lower().startswith('dog'):
            shutil.move(os.path.join(any_dir, file_name), os.path.join(dog_dir, file_name))

devide_class('/kaggle/working/train/train')
devide_class('/kaggle/working/train/val')


train_datagen = ImageDataGenerator(rescale=1./255)
    #rotation_range=20,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    '/kaggle/working/train/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

val_generator = train_datagen.flow_from_directory(
    '/kaggle/working/train/val',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)


history = model.fit(
    train_generator,
    epochs=hyper_epochs, # 訓練データをn回学習する
    validation_data=val_generator
)


"""base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(train_generator, epochs=10, validation_data=val_generator)"""


model_file = '/kaggle/working/cnn.h5'

model.save(model_file)
# model = load_model(model_file)


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
images = [image.img_to_array(image.load_img(path, target_size=(150, 150))) / 255.0 for path in image_path_list]
batch = np.array(images)

# 推論
preds = model.predict(batch)


# IDの取得
ids = [os.path.splitext(os.path.basename(path))[0] for path in image_path_list]

# ラベルの取得
labels = [1 if p[0] > 0.5 else 0 for p in preds]

# CSVで保存
df = pd.DataFrame({'id': ids, 'label': labels})
df.to_csv('submission.csv', index=False)




