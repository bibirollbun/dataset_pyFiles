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

train_zip_path = '/kaggle/input/dogs-vs-cats/train.zip'
extract_path_train = '/kaggle/working/train/'

with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path_train)

test_zip_path = '/kaggle/input/dogs-vs-cats/test1.zip'
extract_path_test = '/kaggle/working/test/'

with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path_test)

print("解凍が完了しました。")
print(os.listdir('/kaggle/working/train/train')[:5]) # 中身を一部確認
print(os.listdir('/kaggle/working/test/test1')[:5])  # 中身を一部確認


import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

TRAIN_DIR = '/kaggle/working/train/train/'

all_filenames = os.listdir(TRAIN_DIR)

df_data = []
for filename in all_filenames:
    filepath = os.path.join(TRAIN_DIR, filename)
    label = 1 if 'dog' in filename else 0  # dogなら1, catなら0
    df_data.append({'filepath': filepath, 'label': label})

df = pd.DataFrame(df_data)

print(f"総画像数: {len(df)}")
print(df.head())

train_df, val_df = train_test_split(
    df,
    test_size=0.2,       # 20%を検証用にする
    random_state=42,     # 再現性のための乱数シード
    stratify=df['label'] # ラベルの比率を保つ
)

print(f"訓練データ数: {len(train_df)}")
print(f"検証データ数: {len(val_df)}")


IMG_SIZE = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])


class CatsDogsDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.filepaths = df['filepath'].values
        self.labels = df['label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

BATCH_SIZE = 32

train_dataset = CatsDogsDataset(train_df, transform=train_transforms)
val_dataset = CatsDogsDataset(val_df, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


images, labels = next(iter(train_loader))
print("\n--- 動作確認 ---")
print(f"1バッチの画像テンソルの形状: {images.shape}") # [バッチサイズ, 色チャネル, 高さ, 幅]
print(f"1バッチのラベルテンソルの形状: {labels.shape}")


import torch
import torch.nn as nn
import torch.optim as optim
import timm
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = timm.create_model('vit_base_patch16_224', pretrained=True)

num_classes = 2
model.head = nn.Linear(model.head.in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

num_epochs = 3 # エポック数（全訓練データを何周学習するか）

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        
        loss = criterion(outputs, labels)

        loss.backward()
        
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        train_bar.set_postfix(loss=(running_loss / (train_bar.n + 1)))

    train_accuracy = 100 * correct_train / total_train
    train_loss = running_loss / len(train_loader)

    model.eval() # モデルを評価モードに設定
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val
    val_loss /= len(val_loader)

    print(f"\nEpoch {epoch+1}/{num_epochs}:")
    print(f"  Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
    print(f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")

print("学習が完了しました。")


import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

TEST_DIR = '/kaggle/working/test/test1/'

test_filenames = os.listdir(TEST_DIR)
test_filenames.sort(key=lambda x: int(x.split('.')[0]))
test_filepaths = [os.path.join(TEST_DIR, f) for f in test_filenames]

class TestDataset(Dataset):
    def __init__(self, filepaths, transform=None):
        self.filepaths = filepaths
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

test_dataset = TestDataset(test_filepaths, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.eval() # モデルを評価モードに
predictions = []

with torch.no_grad():
    test_bar = tqdm(test_loader, desc="Predicting")
    for images in test_bar:
        images = images.to(device)
        outputs = model(images)
        
        probs = F.softmax(outputs.data, dim=1)[:, 1].cpu().numpy()
        predictions.extend(probs)

ids = [int(f.split('.')[0]) for f in test_filenames]

submission_df = pd.DataFrame({
    'id': ids,
    'label': predictions
})

submission_df.to_csv('submission.csv', index=False)

print("\nsubmission.csvの作成が完了しました。")
print("ファイルの中身:")
print(submission_df.head())

