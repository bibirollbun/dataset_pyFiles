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


img_size = 224
batch_size = 16
epochs = 30


# データを初期化
!rm -rf ../data


# データの解凍
base_dir = '/kaggle/input/dogs-vs-cats-redux-kernels-edition'
data_dir = "/kaggle/working/data/"
train_dir = '/kaggle/working/data/train'
test_dir = '/kaggle/working/data/test'

os.makedirs('./data', exist_ok=True)

# 学習データの解凍
with zipfile.ZipFile(os.path.join(base_dir, 'train.zip')) as train_zip:
    train_zip.extractall(data_dir)
    
#推論データの解凍
with zipfile.ZipFile(os.path.join(base_dir, 'test.zip')) as test_zip:
    test_zip.extractall(data_dir)


train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

# 学習データから検証データを分割(8:2)
train_list, val_list = train_test_split(train_list, test_size=0.2)


class ImageTransform():
    
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize((resize, resize)),  # 画像サイズを224にリサイズ
                transforms.RandomHorizontalFlip(),    # ランダム水平反転
                transforms.RandomVerticalFlip(),      # ランダム垂直反転
                transforms.ToTensor(),                # テンソルに変換
                transforms.Normalize(mean, std)       # 正規化
            ]),
            'val': transforms.Compose([
                transforms.Resize((resize, resize)),  # 画像サイズを224にリサイズ
                transforms.ToTensor(),                # テンソルに変換
                transforms.Normalize(mean, std)       # 正規化
            ])
        }
        
    def __call__(self, img, phase):
        return self.data_transform[phase](img)


size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
class ImageDataset(data.Dataset):
    
    def __init__(self, file_list, transform=None, phase='train'):    
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        
        img_transformed = self.transform(img, self.phase)
        
        # ファイル名からラベルを取得
        label = img_path.split('/')[-1].split('.')[0]
        if label == 'dog': #ファイル名がdogであれば 1
            label = 1
        elif label == 'cat':#ファイル名がcatであれば 0
            label = 0

        return img_transformed, label


# データの前処理
train_dataset = ImageDataset(train_list, transform=ImageTransform(size, mean, std), phase='train')
val_dataset = ImageDataset(val_list, transform=ImageTransform(size, mean, std), phase='val')

# データの読み込み
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}


import numpy as np

def get_label_from_path(path):
    filename = os.path.basename(path)
    if filename.startswith('dog'):
        return 1
    else:
        return 0

train_labels = np.array([get_label_from_path(p) for p in train_list])
val_labels = np.array([get_label_from_path(p) for p in val_list])


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


base_dir = '/kaggle/input/dogs-vs-cats-redux-kernels-edition'
data_dir = "/kaggle/working/data/"
train_dir = '/kaggle/working/data/train'
test_dir = '/kaggle/working/data/test'

os.makedirs(data_dir, exist_ok=True)

# zip解凍
import zipfile
with zipfile.ZipFile(os.path.join(base_dir, 'train.zip')) as train_zip:
    train_zip.extractall(data_dir)
with zipfile.ZipFile(os.path.join(base_dir, 'test.zip')) as test_zip:
    test_zip.extractall(data_dir)

import glob
# 画像ファイル一覧
train_list_all = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

# train/val分割
from sklearn.model_selection import train_test_split
train_list, val_list = train_test_split(train_list_all, test_size=0.2, random_state=42)


#ラベル作成データフレイム化
def get_label_from_path(path):
    filename = os.path.basename(path)
    return 'dog' if filename.startswith('dog') else 'cat'

train_labels = [get_label_from_path(p) for p in train_list]
val_labels = [get_label_from_path(p) for p in val_list]

df_train = pd.DataFrame({'filename': train_list, 'label': train_labels})
df_val = pd.DataFrame({'filename': val_list, 'label': val_labels})


from tensorflow.keras.preprocessing.image import ImageDataGenerator

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


!pip install -q keras-cv


from tensorflow import keras


from tensorflow.keras.preprocessing.image import ImageDataGenerator

df_test = pd.DataFrame({'filename': test_list})

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_dataframe(
    df_test,
    x_col='filename',
    class_mode=None,
    target_size=(img_size, img_size),
    batch_size=1,
    shuffle=False
)


!pip install -q transformers timm


from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from PIL import Image
import numpy as np

# 事前学習済みViTのロード（ImageNetで学習済み2値分類用ヘッドに切り替えが必要な場合あり）
model_name = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name, num_labels=2)
processor = ViTImageProcessor.from_pretrained(model_name)


# 画像のパスからnumpy配列への変換例
def preprocess_images(image_paths):
    images = [Image.open(p).convert('RGB') for p in image_paths]
    return processor(images=images, return_tensors="pt")['pixel_values']

# 例: val_listやtest_listを使って推論データ作成
pixel_values = preprocess_images(test_list[:8])  # 8枚だけ例


model.eval()
with torch.no_grad():
    outputs = model(pixel_values)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).numpy()
print(preds)  # 0, 1で出る（ラベルに合わせてマッピングを工夫）


import glob
import os
from sklearn.model_selection import train_test_split

train_dir = '/kaggle/working/data/train'
test_dir = '/kaggle/working/data/test'

train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

def get_label_from_path(path):
    filename = os.path.basename(path)
    return 1 if filename.startswith('dog') else 0

labels = [get_label_from_path(p) for p in train_list]
train_imgs, val_imgs, train_labels, val_labels = train_test_split(train_list, labels, test_size=0.2, random_state=42)


from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

class CatsDogsDataset(Dataset):
    def __init__(self, file_list, labels=None, processor=None):
        self.file_list = file_list
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        if self.labels is not None:
            label = self.labels[idx]
            return inputs, label
        else:
            return inputs

batch_size = 8  # ViTは重いので小さめ
train_ds = CatsDogsDataset(train_imgs, train_labels, processor)
val_ds   = CatsDogsDataset(val_imgs,   val_labels,   processor)
test_ds  = CatsDogsDataset(test_list,  None,         processor)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size)
test_loader  = DataLoader(test_ds,  batch_size=1)


import torch
from transformers import ViTForImageClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2  # 犬猫
)
model.to(device)



from PIL import Image
from torch.utils.data import Dataset
from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

class CatsDogsDataset(Dataset):
    def __init__(self, file_list, labels=None, processor=None):
        self.file_list = file_list
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        if self.labels is not None:
            label = self.labels[idx]
            return inputs, label
        else:
            return inputs


batch_size = 8  # ViTは重いので小さめ
train_ds = CatsDogsDataset(train_imgs, train_labels, processor)
val_ds   = CatsDogsDataset(val_imgs,   val_labels,   processor)
test_ds  = CatsDogsDataset(test_list,  None,         processor)

from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size)
test_loader  = DataLoader(test_ds,  batch_size=1)


import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

epochs = 4  # Kaggleなら3〜5くらいで様子見
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        inputs, labels = batch
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        labels = torch.tensor(labels).to(device)
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} Train Loss: {avg_loss:.4f}")
    
    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            labels = torch.tensor(labels).to(device)
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    acc = correct / total
    print(f"Validation Accuracy: {acc:.4f}")


model.eval()
all_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        inputs = batch
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=1).item()
        all_preds.append(pred)

ids = [int(os.path.basename(x).split('.')[0]) for x in test_list]
import pandas as pd
submission = pd.DataFrame({'id': ids, 'label': all_preds})
submission = submission.sort_values('id')
submission.to_csv('submission.csv', index=False)


# すでにtest_genがある前提
pred = model.predict(test_gen, steps=len(test_gen), verbose=1)
labels = (pred > 0.5).astype(int).flatten()

ids = [int(os.path.basename(x).split('.')[0]) for x in test_list]
submission = pd.DataFrame({'id': ids, 'label': labels})
submission = submission.sort_values('id')
submission.to_csv('submission.csv', index=False)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# val_labels が "dog"/"cat" の場合は 0/1 に変換
val_labels_int = [1 if label == 'dog' else 0 for label in val_labels]

# 検証データで推論（val_genを使って予測）
val_pred_probs = model.predict(val_gen)
val_pred_labels = (val_pred_probs > 0.5).astype(int).flatten()

# 指標を計算
acc = accuracy_score(val_labels_int, val_pred_labels)
prec = precision_score(val_labels_int, val_pred_labels)
rec = recall_score(val_labels_int, val_pred_labels)
f1 = f1_score(val_labels_int, val_pred_labels)

print('--- Validation Data ---')
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

