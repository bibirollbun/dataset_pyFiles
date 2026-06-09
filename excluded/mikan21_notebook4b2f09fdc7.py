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
# ===== 1. zipファイルの解凍 =====
input_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/"
work_dir = "/kaggle/working/"

with zipfile.ZipFile(input_dir + "train.zip", "r") as zip_ref:
    zip_ref.extractall(work_dir + "train/")

with zipfile.ZipFile(input_dir + "test.zip", "r") as zip_ref:
    zip_ref.extractall(work_dir + "test/")



# InceptionV3による犬猫分類（Kaggle Redux用）
import os, shutil
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# 1. Redux形式で展開された画像を cat/dog に振り分け
src = "/kaggle/working/train/train"
dst = "/kaggle/working/train_split"
os.makedirs(f"{dst}/cat", exist_ok=True)
os.makedirs(f"{dst}/dog", exist_ok=True)

for fname in os.listdir(src):
    if fname.startswith("cat"):
        shutil.move(os.path.join(src, fname), os.path.join(dst, "cat", fname))
    elif fname.startswith("dog"):
        shutil.move(os.path.join(src, fname), os.path.join(dst, "dog", fname))

# 2. ImageDataGeneratorによる読み込み
train_dir = dst
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# 3. InceptionV3 モデル構築（転移学習）
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. 学習（3エポック）
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=3
)

# 5. 精度グラフの表示
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# テスト画像ディレクトリ
test_dir = "/kaggle/working/test/test"

# テストデータ用のジェネレータ
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    directory="/kaggle/working/",
    classes=["test"],
    target_size=(150, 150),
    batch_size=1,
    class_mode=None,
    shuffle=False
)

# 予測の実行
predictions = model.predict(test_generator, verbose=1)
labels = predictions.ravel()  # sigmoid出力なのでそのまま確率値

# ファイル名からid抽出（test/1234.jpg → 1234）
file_paths = test_generator.filenames
ids = [int(p.split("/")[-1].split(".")[0]) for p in file_paths]

# DataFrameの作成と保存
submission_df = pd.DataFrame({"id": ids, "label": labels})
submission_df.to_csv("submission.csv", index=False)
print("submission.csv を保存しました。")


