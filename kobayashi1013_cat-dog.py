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
from tensorflow.keras import layers, models # レイヤークラス, 学習モデル
from tensorflow.keras.preprocessing import image # 画像の前処理に必要
import shutil # クラス分けに必要
import glob # ファイルのソートに必要
import re


file_path = '/kaggle/input/dogs-vs-cats/train.zip'
out_path = '/kaggle/working'

with zipfile.ZipFile(file_path, 'r') as file:
    file.extractall(out_path)


train_dir = '/kaggle/working/train'
cat_dir = os.path.join(train_dir, 'cats')
dog_dir = os.path.join(train_dir, 'dogs')

os.makedirs(cat_dir, exist_ok=True)
os.makedirs(dog_dir, exist_ok=True)

for filename in os.listdir(train_dir):
    
    file_path = os.path.join(train_dir, filename)
    if not os.path.isfile(file_path):
        continue
    
    if filename.lower().startswith('cat'):
        shutil.move(os.path.join(train_dir, filename), os.path.join(cat_dir, filename))
    elif filename.lower().startswith('dog'):
        shutil.move(os.path.join(train_dir, filename), os.path.join(dog_dir, filename))


file_path = '/kaggle/input/dogs-vs-cats/test1.zip'
out_path = '/kaggle/working'

with zipfile.ZipFile(file_path, 'r') as file:
    file.extractall(out_path)


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    '/kaggle/working/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)


test_generator = test_datagen.flow_from_directory(
    '/kaggle/working/test1',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)), # 画像化から特徴を抽出する
    layers.MaxPooling2D(2, 2), # 重要な情報を残しつつ、画像サイズを圧縮
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(), # 畳み込み層の出力を、1次元のベクトルに変換
    layers.Dense(512, activation='relu'), # 中間特徴を学習し、分析精度を向上させる
    layers.Dense(1, activation='sigmoid') # sigmoidにより0-1の確率を算出
])


model.compile(
    loss='binary_crossentropy', # 損失関数
    optimizer='adam', # 最適化アルゴリズム
    metrics=['accuracy'] # 評価指標 accuracy：予測が正解だった割合
)


history = model.fit(
    train_generator,
    epochs=10, # 訓練データを10回学習する
)


file_path = '/kaggle/working/cnn.h5'

model.save(file_path)
# model = load_model(file_path)


# ファイル一覧を所得
test_dir = '/kaggle/working/test1'
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

