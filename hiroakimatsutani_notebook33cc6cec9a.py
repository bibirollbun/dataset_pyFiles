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


import numpy as np
import pandas as pd
import os
import zipfile
from glob import glob
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

from sklearn.model_selection import train_test_split
from PIL import Image

%config Completer.use_jedi = False


INPUT_DIR = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/'
OUTPUT_DIR = '/kaggle/working/'

train_zip_path = os.path.join(INPUT_DIR, 'train.zip')
test_zip_path = os.path.join(INPUT_DIR, 'test.zip')

print("訓練データ(train.zip)を解凍しています...")
with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(OUTPUT_DIR)
print("解凍が完了しました。")

print("テストデータ(test.zip)を解凍しています...")
with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(OUTPUT_DIR)
print("解凍が完了しました。")

TRAIN_IMAGE_DIR = os.path.join(OUTPUT_DIR, 'train')
TEST_IMAGE_DIR = os.path.join(OUTPUT_DIR, 'test')

print(f"\nTrain images path: {TRAIN_IMAGE_DIR}")
print(f"Test images path: {TEST_IMAGE_DIR}")


weights = ViT_B_16_Weights.DEFAULT
preprocess = weights.transforms()

print("ViTモデル用の前処理パイプライン:")
print(preprocess)

class DogsVsCatsDataset(Dataset):
    def __init__(self, file_paths, transform=None, is_test=False):
        self.file_paths = file_paths
        self.transform = transform
        self.is_test = is_test
        self.label_map = {"cat": 0, "dog": 1}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, os.path.basename(img_path)
        else:
            label_name = os.path.basename(img_path).split('.')[0]
            label = self.label_map[label_name]
            return image, label

all_files = glob(os.path.join(TRAIN_IMAGE_DIR, '*.jpg'))
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42, stratify=[fn.split('.')[0] for fn in os.listdir(TRAIN_IMAGE_DIR)])

train_dataset = DogsVsCatsDataset(file_paths=train_files, transform=preprocess)
val_dataset = DogsVsCatsDataset(file_paths=val_files, transform=preprocess)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"訓練データの数: {len(train_dataset)}")
print(f"検証データの数: {len(val_dataset)}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = vit_b_16(weights=weights)

num_features = model.heads.head.in_features
model.heads.head = nn.Linear(num_features, 2)

model = model.to(device)

print("モデルの最終層（ヘッド）が変更されました:")
print(model.heads)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)
        
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()
    
def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()


NUM_EPOCHS = 3
LEARNING_RATE = 1e-5

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

best_val_acc = 0.0
MODEL_SAVE_PATH = '/kaggle/working/best_model.pth'

for epoch in range(NUM_EPOCHS):
    print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
    
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")
    
    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Best model saved with validation accuracy: {best_val_acc:.4f}")

print("\n--- 学習完了 ---")
print(f"最終的なベスト検証精度: {best_val_acc:.4f}")


print("提出用ファイルの作成を開始します...")

model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

test_files = sorted(glob(os.path.join(TEST_IMAGE_DIR, '*.jpg')), key=lambda x: int(os.path.basename(x).split('.')[0]))
test_dataset = DogsVsCatsDataset(file_paths=test_files, transform=preprocess, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_preds = []
all_ids = []

with torch.no_grad():
    for images, filenames in tqdm(test_loader, desc="Predicting"):
        images = images.to(device)
        outputs = model(images)
        
        # 確率に変換 (dogクラスである確率)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        all_preds.extend(probs)
        
        ids = [int(fn.split('.')[0]) for fn in filenames]
        all_ids.extend(ids)
        
submission_df = pd.DataFrame({'id': all_ids, 'label': all_preds})
# idでソートして提出フォーマットを保証
submission_df = submission_df.sort_values(by='id').reset_index(drop=True)

# ラベルの値をクリッピング (0に近い/1に近い極端な値を避ける)
submission_df['label'] = np.clip(submission_df['label'], 0.005, 0.995)

submission_df.to_csv('submission.csv', index=False)

print("\nsubmission.csv が正常に作成されました。")
print(submission_df.head())


