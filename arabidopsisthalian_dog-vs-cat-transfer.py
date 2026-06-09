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
from tensorflow.keras.applications import EfficientNetB5 # 転移学習
from tensorflow.keras.optimizers import Adam
import shutil # クラス分けに必要
import glob # ファイルのソートに必要
import re
from sklearn.metrics import accuracy_score # 評価


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')  # TPUの検出
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError as e:
    strategy = tf.distribute.get_strategy()  # CPU or single GPU fallback
    print("TPU not found, using default strategy")


IMG_SIZE = 456
BATCH_SIZE = 32
EPOCHS_0 = 10
EPOCHS_1 = 5
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
    rotation_range=20,            # 回転
    zoom_range=0.2,               # ズーム
    horizontal_flip=True,         # 水平方向の反転
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


with strategy.scope():
    base_model = EfficientNetB5(weights='imagenet', include_top=False)
    base_model.trainable = False
    
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid') # sigmoidにより0-1の確率を算出
    ])
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )


print(1)
# Create batches
train_dataset = train_generator.batch(batch_size)

# Training loop
epochs = 5
for epoch in range(EPOCHS_1):
    print(f"\nStart of epoch {epoch+1}")
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss_value = loss_fn(y_batch, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        if step % 1 == 0:
            print(f"Step {step}: loss = {loss_value:.4f}")


print('end')


callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_0,
    callbacks=callbacks
)


with strategy.scope():
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=Adam(learning_rate=1e-5),
        metrics=['accuracy']
    )


model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_1,
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

