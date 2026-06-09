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
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim


class PaddyDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, label_encoder=None, variety_encoder=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        
        self.label_encoder = label_encoder or {label: i for i, label in enumerate(self.df["label"].unique())}
        self.variety_encoder = variety_encoder or {v: i for i, v in enumerate(self.df["variety"].unique())}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image_id"]
        label_str = row["label"]
        variety_str = row["variety"]
        age = row["age"]  # float or int

        label = self.label_encoder[label_str]
        variety = self.variety_encoder[variety_str]

        img_path = os.path.join(self.image_dir, label_str, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        variety_tensor = torch.tensor(variety, dtype=torch.long)
        age_tensor = torch.tensor([age], dtype=torch.float32)  # 1次元Tensorに

        return image, label, variety_tensor, age_tensor

class PaddyTestDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe                      # 画像名などが含まれるDataFrame
        self.image_dir = image_dir               # 画像が保存されているディレクトリ
        self.transform = transform               # 画像に施す前処理

    def __len__(self):
        return len(self.df)                      # サンプル数

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image_id"] # ファイル名
        img_path = os.path.join(self.image_dir, img_name)  # フルパス
        image = Image.open(img_path).convert("RGB")        # 画像をRGBで読み込む
        if self.transform:
            image = self.transform(image)        # 前処理を適用
        return image, img_name


class PaddyDiseaseModel(nn.Module):
    def __init__(self, num_classes=10, num_varieties=5, meta_features=1, embedding_dim=4):
        super().__init__()
        
        self.cnn = models.efficientnet_b0(pretrained=True)
        num_img_features = self.cnn.classifier[1].in_features
        self.cnn.classifier = nn.Identity()
        
        self.variety_embedding = nn.Embedding(num_varieties, embedding_dim)
        self.meta_fc = nn.Sequential(
            nn.Linear(embedding_dim + meta_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(num_img_features + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, images, varieties, metas):
        img_feat = self.cnn(images)
        var_embed = self.variety_embedding(varieties)
        meta_input = torch.cat([var_embed, metas], dim=1)
        meta_feat = self.meta_fc(meta_input)
        combined = torch.cat([img_feat, meta_feat], dim=1)
        output = self.classifier(combined)
        return output


# CSV読み込み
df = pd.read_csv("/kaggle/input/paddy-disease-classification/train.csv")

test_img_dir = "/kaggle/input/paddy-disease-classification/test_images"

test_image_names = sorted(os.listdir(test_img_dir))

test_df = pd.DataFrame({"image_id": test_image_names})

# ラベル・品種エンコーダを作成
label_encoder = {label: i for i, label in enumerate(df["label"].unique())}
variety_encoder = {v: i for i, v in enumerate(df["variety"].unique())}

# 年齢を正規化（平均0、標準偏差1）
age_mean = df["age"].mean()
age_std = df["age"].std()
df["age"] = (df["age"] - age_mean) / age_std

# 学習・検証分割
from sklearn.model_selection import train_test_split
df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# 画像前処理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# テスト画像の前処理（学習時と同じサイズ・正規化を使用）
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),                         # サイズ調整
    transforms.ToTensor(),                                 # テンソルへ変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406],       # 正規化
                         std=[0.229, 0.224, 0.225])
])

# Dataset作成
train_dataset = PaddyDataset(df_train, "/kaggle/input/paddy-disease-classification/train_images", 
                            transform=train_transform,
                            label_encoder=label_encoder,
                            variety_encoder=variety_encoder)

val_dataset = PaddyDataset(df_val, "/kaggle/input/paddy-disease-classification/train_images", 
                          transform=val_transform,
                          label_encoder=label_encoder,
                          variety_encoder=variety_encoder)

# テスト用DatasetとDataLoaderの作成
test_dataset = PaddyTestDataset(pd.DataFrame({"image_id": test_image_names}), test_img_dir, transform=test_transform)

# DataLoader作成
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


num_classes = len(label_encoder)
num_varieties = len(variety_encoder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PaddyDiseaseModel(num_classes=num_classes, num_varieties=num_varieties, meta_features=1)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, varieties, ages in dataloader:
        images, labels = images.to(device), labels.to(device)
        varieties, ages = varieties.to(device), ages.to(device)

        optimizer.zero_grad()
        outputs = model(images, varieties, ages)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, varieties, ages in dataloader:
            images, labels = images.to(device), labels.to(device)
            varieties, ages = varieties.to(device), ages.to(device)

            outputs = model(images, varieties, ages)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


num_epochs = 10

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs} "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


import matplotlib.pyplot as plt

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1,2,2)
plt.plot(epochs, train_accuracies, label="Train Acc")
plt.plot(epochs, val_accuracies, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")

plt.tight_layout()
plt.show()


# モデルを評価モードに（Dropoutなどを無効に）
model.eval()

# ラベルインデックス → ラベル文字列への辞書（例：0 → 'blast'）
idx_to_label = {v: k for k, v in train_dataset.label_encoder.items()}

# 予測ラベル格納用リスト
predictions = []

# 勾配計算をオフに（高速・省メモリ）
with torch.no_grad():
    for images, image_names in test_loader:
        images = images.to(device)                # GPUへ送る

        # ダミーの variety（全て0）と age（全て0.0）
        dummy_varieties = torch.zeros(images.size(0), dtype=torch.long).to(device)
        dummy_ages = torch.zeros(images.size(0), 1, dtype=torch.float32).to(device)

        outputs = model(images, dummy_varieties, dummy_ages)
        # outputs = model(images)                   # 順伝播で予測
        _, predicted = torch.max(outputs, 1)      # スコア最大のクラスを選択
        preds = [idx_to_label[int(i)] for i in predicted.cpu()]  # ラベル名に変換
        # predictions.extend(preds)                 # 結果を蓄積

        # image_idとlabelをセットにして記録
        for img_name, label in zip(image_names, preds):
            predictions.append((img_name, label))


submission_df = pd.DataFrame(predictions, columns=["image_id", "label"])
submission_df.to_csv("/kaggle/working/submission.csv", index=False)

print("submission.csv was created")

