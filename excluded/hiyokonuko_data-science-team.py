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

# train.zipの展開
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working')

# test1.zipの展開
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/test1.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working')


import shutil
from pathlib import Path

# pathの定義
train_dir = "/kaggle/working/train"
dog_dir = os.path.join(train_dir, "dog")
cat_dir = os.path.join(train_dir, "cat")

# 学習データのフォルダを作成（犬・猫）
os.makedirs(dog_dir, exist_ok=True)
os.makedirs(cat_dir, exist_ok=True)

# 最初にすべてのファイル名を取得（固定リスト）
file_list = os.listdir(train_dir)

# 移動処理
for fname in file_list:
    src = os.path.join(train_dir, fname)

    # ファイルかつ存在しているかを確認
    if os.path.isfile(src):
        if fname.startswith("dog"):
            shutil.move(src, os.path.join(dog_dir, fname))
        elif fname.startswith("cat"):
            shutil.move(src, os.path.join(cat_dir, fname))

# ファイル数の確認
dog_count = len(list(Path(dog_dir).glob("*.jpg")))
cat_count = len(list(Path(cat_dir).glob("*.jpg")))

print("dog:", dog_count)
print("cat:", cat_count)


# 画像表示に必要なモジュールをインポート
import glob
import matplotlib.pyplot as plt
from PIL import Image
import random

dog_images = glob.glob('/kaggle/working/train/dog/*.jpg')
cat_images = glob.glob('/kaggle/working/train/cat/*.jpg')

# 画像の表示（犬）
index = random.randint(0, dog_count)
image_sample = dog_images[index]
img = Image.open(image_sample)
print('filename:{}, size:{}'.format(image_sample, img.size))
plt.imshow(img)
plt.show()

# 画像の表示（猫）
index = random.randint(0, cat_count)
image_sample = cat_images[index]
img = Image.open(image_sample)
print('filename:{}, size:{}'.format(image_sample, img.size))
plt.imshow(img)
plt.show()


from sklearn.model_selection import train_test_split

# クラス作成
animal_class = ['cat', 'dog'] # overviewによるとcat=0,dog=1

# 検証用データを格納するフォルダを作成
os.makedirs("/kaggle/working/val/dog", exist_ok=True)
os.makedirs("/kaggle/working/val/cat", exist_ok=True)

# 実際に仕分けていく
for cls in animal_class:
    files = list(Path("/kaggle/working/train", cls).glob("*.jpg"))
    train_f, val_f = train_test_split(files, test_size=0.2, random_state=42)
    for f in val_f:
        dest = Path("/kaggle/working/val", cls)
        shutil.move(f, os.path.join(dest,f.name))

#ファイル数の確認
# 元がそれぞれ12500なので訓練用 = 10000,検証用 = 2500になっているはず
print("dog(train):", len(list(Path("/kaggle/working/train/dog").glob("*.jpg"))))
print("dog(val):", len(list(Path("/kaggle/working/val/dog").glob("*.jpg"))))
print("cat(train):", len(list(Path("/kaggle/working/train/cat").glob("*.jpg"))))
print("cat(val):", len(list(Path("/kaggle/working/val/cat").glob("*.jpg"))))


#testフォルダの中にサブフォルダを作成（flow_from_directory使うにはサブフォルダじゃなきゃだめらしい）
os.makedirs("/kaggle/working/test1/unknows", exist_ok=True)

# 最初にすべてのファイル名を取得（固定リスト）
file_list = os.listdir("/kaggle/working/test1")

#test1フォルダ内の画像データをサブフォルダに移動
for fname in file_list:
    src = os.path.join("/kaggle/working/test1", fname)

    # ファイルかつ存在しているかを確認
    if os.path.isfile(src):
        shutil.move(src, os.path.join("/kaggle/working/test1/unknows", fname))


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 各データのディレクトリのpathを定義
train_data_dir = '/kaggle/working/train'
validation_data_dir = '/kaggle/working/val'
test_data_dir = '/kaggle/working/test1'

#flow_from_directory用の変数を定義
img_width, img_height = 224, 224  # 画像サイズ
batch_size = 1024                 # バッチサイズ

# 学習データのImageDataGenerator作成
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,          # 各画素のスケールを0～255から0～1に変換して正規化
    rotation_range=20,          # 画像をランダムに ±20度まで回転（被写体の位置ズレに強くなる）
    width_shift_range=0.1,      # 横方向にランダムで最大10%（被写体の位置ズレに強くなる）
    height_shift_range=0.1,     # 縦方向にランダムで最大10%（被写体の位置ズレに強くなる）
    zoom_range=0.1,             # ランダム(±10%)にズームイン or ズームアウト （被写体のサイズ変化に強くなる）
    horizontal_flip=True        # ランダムで画像を左右反転させる (右向き・左向き両方学習できる)
)

# データ数増加
train_generator = train_datagen.flow_from_directory(
    directory=train_data_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    classes=animal_class,
    class_mode='categorical',
    batch_size=batch_size
)


# 検証,テスト用のデータのImageDataGenerator作成
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

#検証データの正規化
val_generator = val_datagen.flow_from_directory(
    directory=validation_data_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    classes=animal_class,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)

#テストデータの正規化（こっちは犬と猫でclass分けしていないのでclasses指定なし）
test_generator = test_datagen.flow_from_directory(
    directory=test_data_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    class_mode=None,
    batch_size=batch_size,
    shuffle=False
)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# モデル構築
model = Sequential([
    Input(shape=(img_width, img_height, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# コンパイル
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# 学習回数を設定
epochs = 10

history = model.fit(
    train_generator,
    epochs = epochs,
    validation_data = val_generator
)


loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# 推論結果の取得
pred_probs = model.predict(test_generator)

# クラスのインデックスに変換（0 or 1）
pred_classes = np.argmax(pred_probs, axis=1)
pred_classes


## 結果を表に作成
pred_pd = pd.DataFrame(pred_probs)
# id列を作成
pred_pd.insert(0, 'id', range(1, len(pred_pd) + 1))
pred_pd.columns
pred_pd = pred_pd.drop([0], axis=1)
pred_pd = pred_pd.rename(columns={1: 'label'})
pred_pd


sub = pd.read_csv('/kaggle/input/dogs-vs-cats/sampleSubmission.csv') # ひな形ファイルを読み込む
print(sub.shape)
# subのlabel列をtest_predictedのlabel列で上書きする
sub['label'] = pred_pd['label']
# 結果を表示
sub
sub.to_csv('submission.csv', index=False) # 変換したファイルを保存

