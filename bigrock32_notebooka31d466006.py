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
import glob
import shutil
import random


#解凍先パス
base_dir = "/kaggle/working"
train_zip_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
test_zip_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"


#zipファイルの展開
with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(base_dir, 'train_raw'))

with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(base_dir, 'test'))


#ディレクトリ構成を作成
folders = [
    'images/train/dog', 'images/train/cat',
    'images/val/dog', 'images/val/cat',
    'images/test/test'
]
for folder in folders:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)


#ファイルパス取得 & シャッフル
all_dogs = glob.glob(os.path.join(base_dir, 'train_raw/train/dog.*.jpg'))
all_cats = glob.glob(os.path.join(base_dir, 'train_raw/train/cat.*.jpg'))

random.seed(42)
random.shuffle(all_dogs)
random.shuffle(all_cats)


# 4. train : val = 8 : 2 に分割
split_idx_dog = int(0.8 * len(all_dogs))
split_idx_cat = int(0.8 * len(all_cats))

train_dogs = all_dogs[:split_idx_dog]
val_dogs = all_dogs[split_idx_dog:]

train_cats = all_cats[:split_idx_cat]
val_cats = all_cats[split_idx_cat:]


# 画像をフォルダに移動
def move_images(file_list, target_dir):
    for file in file_list:
        shutil.copy(file, os.path.join(base_dir, target_dir))

# train
move_images(train_dogs, 'images/train/dog')
move_images(train_cats, 'images/train/cat')

# val
move_images(val_dogs, 'images/val/dog')
move_images(val_cats, 'images/val/cat')

# test
test_images = glob.glob(os.path.join(base_dir, 'test/test/*.jpg'))
move_images(test_images, 'images/test/test')



# ファイル一覧を取得 (犬と猫の画像をそれぞれ取得し、結合)
train_dog_images = glob.glob('/kaggle/working/images/train/dog/*.jpg')
train_cat_images = glob.glob('/kaggle/working/images/train/cat/*.jpg')
images = train_dog_images + train_cat_images # 両方のリストを結合

# ファイル名を確認(先頭3枚)
print(images[:3])


# 画像表示に必要なモジュールをインポート
import matplotlib.pyplot as plt
from PIL import Image

# 画像の表示
if images: # リストが空でないことを確認
    image_sample = images[0]
    img = Image.open(image_sample)
    print('filename:{}, size:{}'.format(image_sample, img.size))
    plt.imshow(img)
    plt.show()
else:
    print("画像ファイルが見つかりませんでした。")


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 共通パラメータ
img_width, img_height = 224, 224
batch_size = 32
classes = ['cat', 'dog']


# 学習用：データ拡張あり + 正規化
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #shear_range=0.2,
    #zoom_range=0.2,
    horizontal_flip=True
)


# 検証用：正規化のみ
val_datagen = ImageDataGenerator(rescale=1.0 / 255)


# テスト用：正規化のみ（ラベルなし、shuffle=False）
test_datagen = ImageDataGenerator(rescale=1.0 / 255)


# ジェネレーターの作成
train_generator = train_datagen.flow_from_directory(
    directory='/kaggle/working/images/train',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    classes=classes
)

val_generator = val_datagen.flow_from_directory(
    directory='/kaggle/working/images/val',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    classes=classes
)

test_generator = test_datagen.flow_from_directory(
    directory='/kaggle/working/images/test',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam


# 入力サイズ（224x224x3）
input_tensor = Input(shape=(224, 224, 3))


# ResNet50のベースモデル（全結合層は除く）
base_model = ResNet50(
    include_top=False,          # 出力層は含めない
    weights='imagenet',         # ImageNetの学習済み重みを使用
    input_tensor=input_tensor
)


# パラメータを全て固定（学習させない）
for layer in base_model.layers:
    layer.trainable = True


# 転移学習用に新しい層を追加
from tensorflow.keras.layers import Dropout
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)  # 中間層
x = Dropout(0.5)(x)
output = Dense(2, activation='softmax')(x)  # 最終出力（2クラス）


# モデルの構築
model = Model(inputs=base_model.input, outputs=output)


# モデルのコンパイル
model.compile(
    optimizer=Adam(learning_rate=1e-6),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# エポック数とステップ数の設定
epochs = 20
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = val_generator.samples // val_generator.batch_size


# モデルの学習
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=validation_steps
)


# テスト用 ImageDataGenerator（ラベルなしの推論用）
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    directory=os.path.join(base_dir, 'images/test'),
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,
    shuffle=False  # ファイル名順に固定
)


# 推論
predictions = model.predict(test_generator, verbose=1)

# 犬 = 1, 猫 = 0 として扱う前提で、犬の確率を抽出
dog_probs = predictions[:, 1]  # Softmaxの出力2列目（犬の確率）

# ファイル名からidを抽出
import os
image_ids = [int(os.path.basename(fname).split('.')[0]) for fname in test_generator.filenames]

# submission DataFrame
submission_df = pd.DataFrame({
    'id': image_ids,
    'label': dog_probs
})

# id順にソート（Kaggleの仕様に合わせる）
submission_df = submission_df.sort_values('id')

# CSVとして保存
submission_df.to_csv('submission.csv', index=False)


# 保存されたファイル一覧を表示して確認
!ls -lh submission.csv

