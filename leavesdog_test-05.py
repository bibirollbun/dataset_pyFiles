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


import os, zipfile, shutil, pathlib

# 解凍先ディレクトリ
WORK_DIR = '/kaggle/working'

# train.zip の解凍
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', 'r') as zip_ref:
    zip_ref.extractall(f'{WORK_DIR}')

# 犬・猫を分類するためのフォルダ作成
base_dir = pathlib.Path(f'{WORK_DIR}/dataset')
cat_dir = base_dir / 'cats'
dog_dir = base_dir / 'dogs'
cat_dir.mkdir(parents=True, exist_ok=True)
dog_dir.mkdir(parents=True, exist_ok=True)

# trainフォルダの画像をcats/dogsに仕分け
train_dir = pathlib.Path(f'{WORK_DIR}/train')
for file in train_dir.iterdir():
    if 'cat' in file.name:
        shutil.move(str(file), cat_dir / file.name)
    elif 'dog' in file.name:
        shutil.move(str(file), dog_dir / file.name)


import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.convnext import preprocess_input

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

import cv2
import numpy as np
from tensorflow.keras.applications.convnext import preprocess_input

# 画像の中心以外をぼかす
def focus_blur_preprocessing(img):
    # BGR変換
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 中心にフォーカス
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) * 0.4), 255, -1)  # 半径は40%程度

    blurred = cv2.GaussianBlur(img, (25, 25), 0)
    focused = np.where(mask[..., None] == 255, img, blurred)

    # BGR → RGB
    focused = cv2.cvtColor(focused, cv2.COLOR_BGR2RGB)

    # ConvNeXt用の前処理
    return preprocess_input(focused.astype(np.float32))

# ImageDataGeneratorの設定
datagen = ImageDataGenerator(
    preprocessing_function=focus_blur_preprocessing,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.2
)

train_gen = datagen.flow_from_directory(
    directory='/kaggle/working/dataset',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    directory='/kaggle/working/dataset',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)


from tensorflow.keras.applications import ConvNeXtSmall
from tensorflow.keras import layers, models, optimizers

base_model = ConvNeXtSmall(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # 転移学習初期は凍結

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint('convnext_small_best.h5', save_best_only=True)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=18,
    callbacks=callbacks
)


# ConvNeXt の全層を再学習させる
base_model.trainable = True  # 凍結解除

for layer in base_model.layers[:-20]:
    layer.trainable = False

# 再コンパイル
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-6),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# エポック数を定義
initial_epochs = len(history.epoch)
fine_tune_epochs = 4
total_epochs = initial_epochs + fine_tune_epochs

# ファインチューニング実行
history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=total_epochs,
    initial_epoch=initial_epochs,
    callbacks=callbacks
)


# 学習済みモデルを保存
model.save("/kaggle/working/best_model.h5")


# test.zip の解凍
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/test')

test_datagen = ImageDataGenerator(
    preprocessing_function=focus_blur_preprocessing  # 学習と同じ前処理関数
)

# ImageDataGeneratorでテスト画像を読み込み
test_gen = ImageDataGenerator(preprocessing_function=focus_blur_preprocessing).flow_from_directory(
    directory='/kaggle/working',
    classes=['test'],
    target_size=IMG_SIZE,
    batch_size=1,
    shuffle=False,
    class_mode=None
)

# 推論（0〜1の確率）
preds = model.predict(test_gen, verbose=1)


import pandas as pd

# sampleSubmission.csv 読み込み
submission = pd.read_csv('/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')

# ファイル名からIDを抽出
test_ids = [int(fname.split('/')[-1].split('.')[0]) for fname in test_gen.filenames]

# 推論結果をDataFrameにする
submission = pd.DataFrame({
    'id': test_ids,
    'label': np.clip(preds.flatten(), 1e-6, 1 - 1e-6)
})

# id順にソート
submission = submission.sort_values('id')
submission.to_csv('submission.csv', index=False)

# 表示
submission.head(10)

