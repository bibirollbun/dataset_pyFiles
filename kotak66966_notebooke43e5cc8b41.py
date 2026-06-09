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
import shutil
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# GPUの有無確認（CPUでも問題なし）
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# =========================
# 1. データの解凍と整理
# =========================

base_dir = '/kaggle/working/dogs-vs-cats-data'
train_zip = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# 解凍
with zipfile.ZipFile(train_zip, 'r') as zip_ref:
    zip_ref.extractall(base_dir)

original_train_dir = os.path.join(base_dir, 'train')

# 画像ファイルをクラス別フォルダに分ける（cats, dogs）
split_train_dir = os.path.join(base_dir, 'split')
train_cats_dir = os.path.join(split_train_dir, 'train/cats')
train_dogs_dir = os.path.join(split_train_dir, 'train/dogs')
val_cats_dir = os.path.join(split_train_dir, 'val/cats')
val_dogs_dir = os.path.join(split_train_dir, 'val/dogs')

for path in [train_cats_dir, train_dogs_dir, val_cats_dir, val_dogs_dir]:
    os.makedirs(path, exist_ok=True)

# データ分割（train:val = 90:10）
all_filenames = os.listdir(original_train_dir)
cats = [f for f in all_filenames if f.startswith('cat')]
dogs = [f for f in all_filenames if f.startswith('dog')]

random.seed(42)
random.shuffle(cats)
random.shuffle(dogs)

def split_and_copy(images, train_dir, val_dir, split_ratio=0.9):
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    for f in train_images:
        shutil.copy(os.path.join(original_train_dir, f), os.path.join(train_dir, f))
    for f in val_images:
        shutil.copy(os.path.join(original_train_dir, f), os.path.join(val_dir, f))

split_and_copy(cats, train_cats_dir, val_cats_dir)
split_and_copy(dogs, train_dogs_dir, val_dogs_dir)

# =========================
# 2. モデルと学習設定
# =========================

img_height = 150
img_width = 150
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(split_train_dir, 'train'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(split_train_dir, 'val'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# CNNモデル構築
model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 学習
history = model.fit(
    train_generator,
    epochs=2,
    validation_data=val_generator
)

# =========================
# 3. モデル保存（任意）
# =========================
model.save("/kaggle/working/dogs_vs_cats_model.h5")


# =========================
# 4. テスト画像の準備
# =========================

test_zip = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'
test_dir = '/kaggle/working/test'

# 解凍
with zipfile.ZipFile(test_zip, 'r') as zip_ref:
    zip_ref.extractall(test_dir)

test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    directory='/kaggle/working',  # test/ ディレクトリの親
    classes=['test'],             # test/ ディレクトリ内を指定
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode=None,
    shuffle=False
)

# 推論
pred = model.predict(test_generator, steps=len(test_generator), verbose=1)

import pandas as pd
import numpy as np
import os

# ファイル名からIDを取得
filenames = test_generator.filenames
ids = [int(os.path.basename(name).split('.')[0]) for name in filenames]
labels = (pred > 0.5).astype(int).flatten()

submission = pd.DataFrame({'id': ids, 'label': labels})
submission = submission.sort_values('id')  # id順に並べる（重要）

submission.to_csv('/kaggle/working/submission.csv', index=False)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


from sklearn.metrics import accuracy_score, recall_score, f1_score

# 検証データの予測値
val_gen.reset()
val_pred = model.predict(val_gen, steps=len(val_gen), verbose=1)
val_pred_label = (val_pred > 0.5).astype(int).flatten()

# 正解ラベル（flow_from_dataframeのgeneratorが内部で持っている）
val_true = val_gen.classes

# 再現率とF1スコアを計算
recall = recall_score(val_true, val_pred_label)
f1 = f1_score(val_true, val_pred_label)
acc = accuracy_score(val_true, val_pred_label)

print(f"Recall (検証データ): {recall:.4f}")
print(f"F1-score (検証データ): {f1:.4f}")
print(f"Accuracy (検証データ): {acc:.4f}")


import pandas as pd
import os, glob, time, copy, zipfile
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import models, transforms

# データを初期化
!rm -rf ../data

import os
import zipfile
import glob
from sklearn.model_selection import train_test_split

# ディレクトリ定義
base_dir = '/kaggle/input/dogs-vs-cats-redux-kernels-edition'
data_dir = '/kaggle/working/data/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# 書き込み先ディレクトリ作成
os.makedirs(data_dir, exist_ok=True)

# zipファイル解凍
with zipfile.ZipFile(os.path.join(base_dir, 'train.zip')) as train_zip:
    train_zip.extractall(data_dir)
with zipfile.ZipFile(os.path.join(base_dir, 'test.zip')) as test_zip:
    test_zip.extractall(data_dir)

# ファイル一覧取得
train_list_all = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

# train/val分割（再現性あり）
train_list, val_list = train_test_split(train_list_all, test_size=0.2, random_state=42)

import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_label_from_path(path):
    filename = os.path.basename(path)
    return 'dog' if filename.startswith('dog') else 'cat'

train_labels = [get_label_from_path(p) for p in train_list]
val_labels = [get_label_from_path(p) for p in val_list]

df_train = pd.DataFrame({'filename': train_list, 'label': train_labels})
df_val = pd.DataFrame({'filename': val_list, 'label': val_labels})

img_size = 128

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    df_train, 
    x_col='filename',
    y_col='label',
    target_size=(img_size, img_size),
    class_mode='binary',
    batch_size=32,
    shuffle=True
)
val_gen = val_datagen.flow_from_dataframe(
    df_val,
    x_col='filename',
    y_col='label',
    target_size=(img_size, img_size),
    class_mode='binary',
    batch_size=32,
    shuffle=False
)

#ラベル作成データフレイム化
def get_label_from_path(path):
    filename = os.path.basename(path)
    return 'dog' if filename.startswith('dog') else 'cat'

train_labels = [get_label_from_path(p) for p in train_list]
val_labels = [get_label_from_path(p) for p in val_list]

df_train = pd.DataFrame({'filename': train_list, 'label': train_labels})
df_val = pd.DataFrame({'filename': val_list, 'label': val_labels})

#データローダーの作成
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    df_train, 
    x_col='filename',
    y_col='label',
    target_size=(img_size, img_size),
    class_mode='binary',
    batch_size=8,
    shuffle=True
)
val_gen = val_datagen.flow_from_dataframe(
    df_val,
    x_col='filename',
    y_col='label',
    target_size=(img_size, img_size),
    class_mode='binary',
    batch_size=8,
    shuffle=False
)

#EfficientNetB0でモデル構築
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(img_size, img_size, 3)
)
base_model.trainable = False  # 転移学習なのでまずは特徴抽出層を凍結

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#学習
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=2
)

#推論
df_test = pd.DataFrame({'filename': test_list})

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_dataframe(
    df_test,
    x_col='filename',
    target_size=(img_size, img_size),
    class_mode=None,
    batch_size=1,
    shuffle=False
)

pred = model.predict(test_gen, steps=len(test_gen), verbose=1)
labels = (pred > 0.5).astype(int).flatten()

ids = [int(os.path.basename(x).split('.')[0]) for x in test_list]
submission = pd.DataFrame({'id': ids, 'label': labels})
submission = submission.sort_values('id')

submission.to_csv('submission.csv', index=False)

import matplotlib.pyplot as plt

# 精度の推移
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 損失の推移
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show

from sklearn.metrics import accuracy_score, recall_score, f1_score

# 検証データの予測値
val_gen.reset()
val_pred = model.predict(val_gen, steps=len(val_gen), verbose=1)
val_pred_label = (val_pred > 0.5).astype(int).flatten()

# 正解ラベル（flow_from_dataframeのgeneratorが内部で持っている）
val_true = val_gen.classes

# 再現率とF1スコアを計算
recall = recall_score(val_true, val_pred_label)
f1 = f1_score(val_true, val_pred_label)
acc = accuracy_score(val_true, val_pred_label)

print(f"Recall (検証データ): {recall:.4f}")
print(f"F1-score (検証データ): {f1:.4f}")
print(f"Accuracy (検証データ): {acc:.4f}")


import os
import zipfile
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score

# ================================
# 0. GPU設定
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# 1. データ展開とパス準備
# ================================
train_zip = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
test_zip = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"
base_dir = "/kaggle/working"

with zipfile.ZipFile(train_zip, 'r') as zip_ref:
    zip_ref.extractall(base_dir)
with zipfile.ZipFile(test_zip, 'r') as zip_ref:
    zip_ref.extractall(base_dir)

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# ================================
# 2. データセット定義
# ================================
class CatsDogsDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = 1 if "dog" in os.path.basename(path).lower() else 0
        return image, label

# ファイルリスト作成・分割（画像のみ抽出）
all_files = [os.path.join(train_dir, fname) for fname in os.listdir(train_dir)
             if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

# データ拡張と前処理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = CatsDogsDataset(train_files, transform=train_transform)
val_dataset = CatsDogsDataset(val_files, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

# ================================
# 3. モデル定義
# ================================
weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
model = shufflenet_v2_x1_0(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ================================
# 4. 訓練ループ＋評価記録
# ================================
train_losses = []
train_accuracies = []
val_accuracies = []
val_recalls = []
val_f1s = []

for epoch in range(2):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = correct / total
    train_losses.append(running_loss)
    train_accuracies.append(train_acc)

    print(f"[Train] Epoch {epoch+1}, Loss: {running_loss:.4f}, Accuracy: {train_acc:.4f}")

    # === 検証 ===
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    val_accuracies.append(acc)
    val_recalls.append(recall)
    val_f1s.append(f1)

    print(f"[Val]   Epoch {epoch+1}, Accuracy: {acc:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# ================================
# 5. 推論・提出ファイル生成
# ================================
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_files = sorted(
            [os.path.join(test_dir, fname) for fname in os.listdir(test_dir)
             if fname.lower().endswith('.jpg')],
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )
        self.transform = transform

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, idx):
        path = self.test_files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        id = int(os.path.basename(path).split('.')[0])
        return img, id

test_dataset = TestDataset(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()
predictions = []
ids = []

with torch.no_grad():
    for imgs, img_ids in tqdm(test_loader, desc="Inferencing"):
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # class 1: dog
        predictions.extend(probs.cpu().numpy())
        ids.extend(img_ids)

submission = pd.DataFrame({'id': ids, 'label': predictions})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("✅ Submission file saved to /kaggle/working/submission.csv")

# ================================
# 6. グラフ描画
# ================================
epochs = range(1, len(train_losses) + 1)

# Loss
plt.figure(figsize=(10, 4))
plt.plot(epochs, train_losses, marker='o', label='Train Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

# Accuracy
plt.figure(figsize=(10, 4))
plt.plot(epochs, train_accuracies, marker='o', label='Train Accuracy')
plt.plot(epochs, val_accuracies, marker='s', label='Val Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

# F1-score
plt.figure(figsize=(10, 4))
plt.plot(epochs, val_f1s, marker='^', label='Val F1-score')
plt.plot(epochs, val_recalls, marker='x', label='Val Recall')
plt.title('Validation F1/Recall per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.grid(True)
plt.legend()
plt.show()



# ===============================
# 0. インストールとインポート
# ===============================
!pip install -q timm

import os
import zipfile
import random
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import matplotlib.pyplot as plt

# ===============================
# 1. データ展開と準備
# ===============================
base_path = '/kaggle/input/dogs-vs-cats-redux-kernels-edition'
os.makedirs("/kaggle/working/train", exist_ok=True)
os.makedirs("/kaggle/working/test", exist_ok=True)

# ZIPファイル展開
with zipfile.ZipFile(os.path.join(base_path, "train.zip"), 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/train")
with zipfile.ZipFile(os.path.join(base_path, "test.zip"), 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/test")

# 全画像パス取得とラベル付け
train_dir = "/kaggle/working/train/train"
all_images = os.listdir(train_dir)
labels = [1 if 'dog' in fname else 0 for fname in all_images]
image_paths = [os.path.join(train_dir, fname) for fname in all_images]

# ===============================
# 2. Train / Val 分割
# ===============================
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

# ===============================
# 3. Datasetクラス定義
# ===============================
class DogCatDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image

# ===============================
# 4. DataLoader準備
# ===============================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = DogCatDataset(train_paths, train_labels, transform_train)
val_dataset = DogCatDataset(val_paths, val_labels, transform_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ===============================
# 5. モデル定義（ViT）
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)
model = model.to(device)

# ===============================
# 6. 学習ループ
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 5
train_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_loss_list.append(avg_loss)

    # validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    val_acc_list.append(acc)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

# ===============================
# 7. 学習曲線
# ===============================
plt.plot(train_loss_list, label="Loss")
plt.plot(val_acc_list, label="Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()

# ===============================
# 8. 推論と提出ファイル作成
# ===============================
test_dir = "/kaggle/working/test/test"
test_images = sorted(os.listdir(test_dir), key=lambda x: int(x.split('.')[0]))
test_paths = [os.path.join(test_dir, fname) for fname in test_images]
test_dataset = DogCatDataset(test_paths, transform=transform_val)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
preds = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds.extend(probs[:, 1].cpu().numpy())  # 犬の確率

submission = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))
submission["label"] = preds
submission.to_csv("submission.csv", index=False)



# ===============================
# 0. 必要なライブラリのインストール
# ===============================
!pip install timm -q

# ===============================
# 1. インポート
# ===============================
import os
import zipfile
import random
import shutil
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

import timm

# ===============================
# 2. データセットの準備
# ===============================
# 解凍
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip", 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/train")

# フォルダ構成変更：犬猫ごとに分ける
train_dir = "/kaggle/working/train/train"
output_dir = "/kaggle/working/split_data"
os.makedirs(f"{output_dir}/cat", exist_ok=True)
os.makedirs(f"{output_dir}/dog", exist_ok=True)

for fname in os.listdir(train_dir):
    label = fname.split('.')[0]
    src = os.path.join(train_dir, fname)
    dst = os.path.join(output_dir, label, fname)
    shutil.move(src, dst)

# ===============================
# 3. 前処理とデータローダ
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

full_dataset = ImageFolder(output_dir, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# ===============================
# 4. モデル定義：ViT Tiny
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=2)
model = model.to(device)

# ===============================
# 5. 学習準備
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 5

train_losses = []
val_losses = []
val_accuracies = []

# ===============================
# 6. 学習ループ
# ===============================
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    total_vloss = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_vloss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    avg_val_loss = total_vloss / len(val_loader)
    val_losses.append(avg_val_loss)
    accuracy = correct / len(val_dataset)
    val_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Acc: {accuracy:.4f}")

# ===============================
# 7. グラフ描画
# ===============================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Val Acc')
plt.legend()
plt.title("Accuracy")
plt.show()


