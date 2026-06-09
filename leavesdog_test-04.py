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


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.convnext import preprocess_input

# 画像サイズなどの設定
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# データ拡張＆正規化（validation_splitで分割）
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,  # 20%を検証用
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.2
)

train_gen = datagen.flow_from_directory(
    directory='/kaggle/working/dataset',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    directory='/kaggle/working/dataset',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
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
    epochs=10,
    callbacks=callbacks
)


import matplotlib.pyplot as plt

# 総エポック数
total_epochs = len(history.history['loss'])
if 'history_ft' in locals():
    total_epochs += len(history_ft.history['loss'])

# グラフ描画（2画面）
plt.figure(figsize=(12, 5))

# ========== 1. lossグラフ ==========
plt.subplot(1, 2, 1)

# 前半の学習（凍結中）
plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'], '-o')
plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], '-o')

# ========== 2. accuracyグラフ ==========
plt.subplot(1, 2, 2)

plt.plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'], '-o')
plt.plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], '-o')

if 'history_ft' in locals():
    start = len(history.history['accuracy']) + 1
    plt.plot(range(start, start + len(history_ft.history['accuracy'])), history_ft.history['accuracy'], '-o')
    plt.plot(range(start, start + len(history_ft.history['val_accuracy'])), history_ft.history['val_accuracy'], '-o')

plt.title('Accuracy Transition')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()
plt.legend(['train_acc_1', 'val_acc_1', 'train_acc_2', 'val_acc_2'] if 'history_ft' in locals() else ['accuracy', 'val_accuracy'], loc='best')

# 表示
plt.tight_layout()
plt.show()


import tensorflow as tf
import numpy as np

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.input],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # 2値分類の1クラス目の出力を使う

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# 予測結果取得
preds = model.predict(val_gen)
pred_labels = (preds > 0.5).astype(int).flatten()

# 正解ラベル、ファイル名
true_labels = val_gen.classes
filenames = val_gen.filenames

# 誤分類画像リスト
misclassified = []
for i in range(len(filenames)):
    if pred_labels[i] != true_labels[i]:
        misclassified.append((filenames[i], true_labels[i], pred_labels[i]))


import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.convnext import preprocess_input
import os

# Grad-CAM heatmap を生成する関数
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.input],  # convnext_smallの入力
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]  # 勾配
    conv_outputs = conv_outputs[0]  # 出力特徴マップ

    # 勾配をグローバル平均プーリング
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # 重み付き合計
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    # ReLU適用＆正規化
    cam = np.maximum(cam, 0)
    cam /= (np.max(cam) + 1e-8)
    return cam

# 前処理関数（画像読み込みとConvNeXt用の整形）
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    return np.expand_dims(preprocess_input(img_array), axis=0), img

# Grad-CAM 実行
results = []

# convnext_small を model から取得
convnext_model = model.get_layer('convnext_small')

# 最後のConv層名
last_conv_layer_name = "convnext_small_stage_3_block_2_depthwise_conv"

# 可視化実行（最初の10件）
for img_rel_path, true, pred in misclassified[:10]:
    img_path = os.path.join(val_gen.directory, img_rel_path)
    img_array, original_img = load_and_preprocess_image(img_path)

    heatmap = make_gradcam_heatmap(img_array, convnext_model, last_conv_layer_name)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    original = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    results.append((superimposed, img_rel_path, true, pred))


plt.figure(figsize=(12, 5 * ((len(results)+1)//2)))
for i, (img, path, true, pred) in enumerate(results):
    plt.subplot((len(results)+1)//2, 2, i+1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"True:{true} Pred:{pred}\n{os.path.basename(path)}")
    plt.axis('off')
plt.tight_layout()
plt.show()


# 学習済みモデルを保存
model.save("/kaggle/working/best_model.h5")


!pip install -q albumentations


import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

IMG_SIZE = (224, 224)
height, width = IMG_SIZE

train_transform = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)), 
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.OneOf([
        A.GaussianBlur(p=0.3),
        A.GaussNoise(p=0.3),
        A.ISONoise(p=0.3),
    ], p=0.3),
    A.CoarseDropout(max_holes=1, max_height=40, max_width=40, p=0.5),
    A.Normalize()
])


from tensorflow.keras.utils import Sequence
import numpy as np
import os
from PIL import Image

class AlbumentationsDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size, transform, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        batch_labels = []

        for i in batch_indices:
            img = np.array(Image.open(self.image_paths[i]).convert("RGB"))
            img = self.transform(image=img)['image']
            batch_images.append(img)
            batch_labels.append(self.labels[i])

        return np.stack(batch_images), np.array(batch_labels)


from glob import glob

cat_images = glob('/kaggle/working/dataset/cats/*.jpg')
dog_images = glob('/kaggle/working/dataset/dogs/*.jpg')

image_paths = cat_images + dog_images
labels = [0] * len(cat_images) + [1] * len(dog_images)

# データ分割
from sklearn.model_selection import train_test_split

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)


train_gen = AlbumentationsDataGenerator(
    image_paths=train_paths,
    labels=train_labels,
    batch_size=32,
    transform=train_transform
)

val_transform = A.Compose([
    A.Resize(height=height, width=width),
    A.Normalize()
])

val_gen = AlbumentationsDataGenerator(
    image_paths=val_paths,
    labels=val_labels,
    batch_size=32,
    transform=val_transform,
    shuffle=False
)


callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint('convnext_small_best.h5', save_best_only=True)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks
)


import matplotlib.pyplot as plt

epochs = len(history.history['loss'])

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), history.history['loss'], label='Train Loss')
plt.plot(range(1, epochs + 1), history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend(); plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), history.history['accuracy'], label='Train Acc')
plt.plot(range(1, epochs + 1), history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend(); plt.grid()

plt.tight_layout()
plt.show()


# test.zip の解凍
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/test')

# ImageDataGeneratorでテスト画像を読み込み
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
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

# 推論結果を代入
submission['label'] = preds
submission.to_csv('submission.csv', index=False)

# 提出ファイルの読み込み
submission = pd.read_csv('submission.csv')

# 表示
submission.head(10)

