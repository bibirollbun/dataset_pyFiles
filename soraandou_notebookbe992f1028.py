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


# 環境準備と必要なライブラリのインポート

import os
import zipfile
import shutil
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# GPUの有無確認
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

print("--- 0. 環境準備とライブラリのインポートが完了しました ---")


print("データの解凍と整理を開始")

# Kaggleデータセットのベースパス
base_data_path = '/kaggle/input/dogs-vs-cats-redux-kernels-edition'

# 解凍先および作業ディレクトリの設定
# /kaggle/working はノートブックが作業できる永続的なストレージ
working_dir = '/kaggle/working/dogs-vs-cats-data'
if not os.path.exists(working_dir):
    os.makedirs(working_dir)

# 訓練データのZIPファイルパス
train_zip_path = os.path.join(base_data_path, 'train.zip')
# テストデータのZIPファイルパス
test_zip_path = os.path.join(base_data_path, 'test.zip')


# 訓練データの解凍
print(f"'{os.path.basename(train_zip_path)}' を解凍中...")
with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(working_dir)
print("訓練データの解凍が完了しました。")

# テストデータの解凍 (予測フェーズで必要)
print(f"'{os.path.basename(test_zip_path)}' を解凍中...")
with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(working_dir)
print("テストデータの解凍が完了しました。")

# 解凍された元の訓練画像があるディレクトリ
original_train_dir = os.path.join(working_dir, 'train')
original_test_dir = os.path.join(working_dir, 'test')

# 訓練/検証データ分割用の新しいディレクトリ構造を作成
# structure: split_data/train/cats, split_data/train/dogs, split_data/val/cats, split_data/val/dogs
split_data_dir = os.path.join(working_dir, 'split_data')

train_cats_dir = os.path.join(split_data_dir, 'train/cats')
train_dogs_dir = os.path.join(split_data_dir, 'train/dogs')
val_cats_dir = os.path.join(split_data_dir, 'val/cats')
val_dogs_dir = os.path.join(split_data_dir, 'val/dogs')

# 必要なディレクトリを全て作成
for path in [train_cats_dir, train_dogs_dir, val_cats_dir, val_dogs_dir]:
    os.makedirs(path, exist_ok=True)

# 画像を犬と猫に分類し、訓練セットと検証セットに分割して移動
all_fnames = os.listdir(original_train_dir)
random.shuffle(all_fnames) # ランダムにシャッフルして分割を公平にする

train_split_ratio = 0.9 # 90%を訓練、10%を検証

num_train_dogs = 0
num_train_cats = 0
num_val_dogs = 0
num_val_cats = 0

for i, fname in enumerate(all_fnames):
    src_path = os.path.join(original_train_dir, fname)
    if 'cat' in fname:
        if i < len(all_fnames) * train_split_ratio:
            dst_path = os.path.join(train_cats_dir, fname)
            num_train_cats += 1
        else:
            dst_path = os.path.join(val_cats_dir, fname)
            num_val_cats += 1
    elif 'dog' in fname:
        if i < len(all_fnames) * train_split_ratio:
            dst_path = os.path.join(train_dogs_dir, fname)
            num_train_dogs += 1
        else:
            dst_path = os.path.join(val_dogs_dir, fname)
            num_val_dogs += 1
    
    shutil.copyfile(src_path, dst_path)

print(f"訓練セット: 犬 {num_train_dogs}枚, 猫 {num_train_cats}枚")
print(f"検証セット: 犬 {num_val_dogs}枚, 猫 {num_val_cats}枚")
print("--- 1. データの解凍と整理が完了しました ---")


print("データ拡張と画像データのジェネレータ作成を開始します")

# 画像のリサイズサイズとバッチサイズを設定
IMAGE_WIDTH, IMAGE_HEIGHT = 150, 150 # モデルの入力サイズ
BATCH_SIZE = 32

# 訓練データ用ImageDataGenerator: データ拡張と正規化
train_datagen = ImageDataGenerator(
    rescale=1./255,          # ピクセル値を0-1に正規化
    rotation_range=40,       # ランダムな回転 (0-40度)
    width_shift_range=0.2,   # 水平方向のシフト (画像の幅の20%以内)
    height_shift_range=0.2,  # 垂直方向のシフト (画像の高さの20%以内)
    shear_range=0.2,         # シアー変換
    zoom_range=0.2,          # ランダムなズーム
    horizontal_flip=True,    # 水平方向の反転
    fill_mode='nearest'      # 新しく生成されたピクセルの埋め方
)

# 検証データ用ImageDataGenerator: 正規化のみ (データ拡張は適用しない)
validation_datagen = ImageDataGenerator(rescale=1./255)

# テストデータ用ImageDataGenerator: 正規化のみ
test_datagen = ImageDataGenerator(rescale=1./255)

# 訓練データジェネレータの作成
train_generator = train_datagen.flow_from_directory(
    os.path.join(split_data_dir, 'train'), # 訓練データディレクトリ
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary' # 犬/猫の二値分類のため
)

# 検証データジェネレータの作成
validation_generator = validation_datagen.flow_from_directory(
    os.path.join(split_data_dir, 'val'), # 検証データディレクトリ
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# テストデータジェネレータの作成
# テストデータはラベルを持たないため、class_mode=None、予測順序を保つためshuffle=False
test_generator = test_datagen.flow_from_directory(
    original_test_dir, # 解凍したテストデータディレクトリ
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=1, # 1枚ずつ予測するため
    class_mode=None, # ラベルがないため
    shuffle=False # 予測順序を保持するため
)

print("データ拡張と画像データのジェネレータ作成が完了しました")


print(" モデルの構築を開始します")

model = Sequential([
    Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)), # Input層を明示的に定義
    # 第1畳み込みブロック
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    # 第2畳み込みブロック
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    # 第3畳み込みブロック
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    # 第4畳み込みブロック
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    # 特徴マップを平坦化
    Flatten(),
    # 過学習を防ぐためのドロップアウト層
    Dropout(0.5), # 50%のニューロンをランダムに無効化
    # 全結合層
    Dense(512, activation='relu'),
    # 出力層: 二値分類のためsigmoid活性化関数
    Dense(1, activation='sigmoid') # 犬である確率 (0-1) を出力
])

# モデルのコンパイル: オプティマイザ、損失関数、評価指標を設定
# 学習率を小さめに設定し、安定した学習を目指す
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',       # 二値分類の標準的な損失関数
              metrics=['accuracy'])             # 評価指標として精度を使用

# モデルの概要を表示
model.summary()

print("モデルの構築が完了しました")


print("--- 4. モデルの訓練を開始します ---")

# 訓練エポック数を設定
EPOCHS = 50 # 適切な値は試行錯誤により決定。GPU環境ならもう少し多く試せる

# モデルの訓練を実行
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE, # 1エポックあたりのステップ数
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE # 検証データでのステップ数
)

print("--- 4. モデルの訓練が完了しました ---")


print("---テスト画像の準備と予測、提出ファイルの生成を開始")

# テストデータのZIPファイルパス
test_zip_path = os.path.join(base_data_path, 'test.zip')

# テストデータ解凍先のディレクトリを設定
test_extract_dir = '/kaggle/working/test_images_for_prediction'
if os.path.exists(test_extract_dir):
    shutil.rmtree(test_extract_dir) # 既存のディレクトリがあれば削除
os.makedirs(test_extract_dir)

# テストデータの解凍
print(f"'{os.path.basename(test_zip_path)}' を '{test_extract_dir}' に解凍中...")
with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(test_extract_dir) # test_images_for_prediction/test/ の中に画像がある状態
print("テストデータの解凍が完了しました。")

# ImageDataGenerator for test data (正規化のみ)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# テストデータジェネレータの作成
test_generator = test_datagen.flow_from_directory(
    directory=test_extract_dir, # 親ディレクトリを指定
    classes=['test'],           # このサブディレクトリを読み込む
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=1,               # 1枚ずつ予測するため
    class_mode=None,            # ラベルがないため
    shuffle=False               # 予測順序を保持するため
)

print(f"Test Generator found {test_generator.samples} images.")
if test_generator.samples == 0:
    print("CRITICAL ERROR: Test generator found 0 images. Please check the 'test_extract_dir' path and its contents.")
    print("Expected structure: test_images_for_prediction/test/1.jpg, 2.jpg, ...")


# モデルによる推論 (予測)
print("テスト画像に対する予測を開始")
# len(test_generator) は test_generator.samples // batch_size と同等 (batch_size=1なので samples と同じ)
pred = model.predict(test_generator, steps=len(test_generator), verbose=1)
print("予測が完了")

# 提出ファイルの作成
print("提出ファイルを生成中")
# ファイル名からIDを取得
filenames = test_generator.filenames # 例: ['test/1.jpg', 'test/10.jpg', ...]
ids = [int(os.path.basename(name).split('.')[0]) for name in filenames]

labels = pred.flatten() # 確率をそのまま提出

submission = pd.DataFrame({'id': ids, 'label': labels})

# id順に並べる
submission = submission.sort_values('id')
submission.to_csv('/kaggle/working/submission.csv', index=False)

print(f"提出ファイル '/kaggle/working/submission.csv' が作成")

print("---テスト画像の準備と予測、提出ファイルの生成が完了 ---")


print(" 学習曲線の描画を開始")

# 精度 (Accuracy) の学習曲線
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# 損失 (Loss) の学習曲線
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

print("---学習曲線の描画が完了 ---")

