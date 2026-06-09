#　モジュールのインポート
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim


#データセット作成
class Dataset(Dataset):
    def __init__(self, filepaths, labels=None, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            label = self.labels[idx]
            return img, label
        else:
            return img, os.path.basename(img_path)


# zipファイル展開
import zipfile

# zipファイルのパス
zip_dir = "../input/dogs-vs-cats-redux-kernels-edition/"
output_dir = "/kaggle/working/"  # 展開先

# 解凍対象のzipファイル一覧
zip_files = ["train.zip", "test.zip"]

# 各zipファイルを展開
for zip_file in zip_files:
    zip_path = os.path.join(zip_dir, zip_file)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

# 展開後のパスを設定
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

# 画像ファイルを取得
train_files = glob.glob(os.path.join(train_dir, "*.jpg"))
test_files = glob.glob(os.path.join(test_dir, "*.jpg"))
print(f"ファイル数: {len(train_files)}")
print(f"ファイル数: {len(test_files)}")
labels = [1 if "dog" in os.path.basename(f) else 0 for f in train_files]

print(f"train_files: {len(train_files)}")
print(f"labels: {len(labels)}")

# train/val分割
train_files, val_files, train_labels, val_labels = train_test_split(
    train_files, labels, test_size=0.2, random_state=42, stratify=labels
)


# 前処理
transform = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

#DataLoaderの作成
batch_size = 32

train_dataset = Dataset(train_files, train_labels, transform)
val_dataset = Dataset(val_files, val_labels, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# モデル構築・学習準備
from torchvision.models import resnet18, ResNet18_Weights
#from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights=ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)

# conv1とbn1を固定
#for param in model.conv1.parameters():
#    param.requires_grad = False

#for param in model.bn1.parameters():
#    param.requires_grad = False
# layer1とlayer2も固定
#for param in model.layer1.parameters():
#    param.requires_grad = False

#for param in model.layer2.parameters():
#    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)

# transformを定義
transform = weights.transforms()

model = model.to(device)

criterion = nn.CrossEntropyLoss()

#optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

# 残りの層だけoptimizerに渡す(学習率を調整)
optimizer = torch.optim.AdamW([
    #{"params": model.layer1.parameters(), "lr": 1e-6},
    #{"params": model.layer2.parameters(), "lr": 1e-5},
    #{"params": model.layer3.parameters(), "lr": 1e-7},
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(),     "lr": 1e-4}
], weight_decay=1e-4)

#scheduler = StepLR(optimizer, step_size=5, gamma=0.1)# version6



import torch
from tqdm import tqdm

#num_epochs = 5
#num_epochs = 10

# version9
num_epochs = 30
#patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0
delta = 1e-4

#version10
#patience = 3
#version11
#patience = 5
#version12
patience = 3

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()  # 学習モード
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        
        # 精度計算
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
    train_epoch_loss = running_loss / total
    train_epoch_acc = correct / total
    
    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_acc)
    
    print(f"[{epoch+1}/{num_epochs}] Loss: {train_epoch_loss:.4f} | Accuracy: {train_epoch_acc:.4f}")
    
    # モデルを評価モードに切り替え
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    # バリデーションデータで性能評価
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 予測ラベルを取得
            _, preds = torch.max(outputs, 1)

            val_loss += loss.item() * images.size(0)  # 合計loss
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
    val_epoch_loss = val_loss / val_total
    val_epoch_acc = val_correct / val_total
    
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc)
    
    print(f"→ Val Loss: {val_epoch_loss:.4f} | Val Accuracy: {val_epoch_acc:.4f}")

    # 学習率の調整
    #scheduler.step()

    # アーリーストッピングの導入
    if val_epoch_loss < best_val_loss- delta:
        best_val_loss = val_epoch_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


# 学習の損失と精度の可視化
import matplotlib.pyplot as plt

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()

plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.show()


# transforms は学習と同じものを使用
test_transform = transform  

# testデータセット作成
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

# testファイル一覧
test_dataset = TestDataset(test_files, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 推論モード
model.eval()
all_preds = []
all_ids = []

with torch.no_grad():
    for images, paths in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = probs[:, 1].cpu().numpy() # 犬クラスの確率
        all_preds.extend(preds)
        all_ids.extend([int(os.path.basename(p).split(".")[0]) for p in paths])


#CSVファイル作成
# idとlabelをDataFrameに
submission_df = pd.DataFrame({
    "id": all_ids,
    "label": all_preds
})

submission_df = submission_df.sort_values("id").reset_index(drop=True)
submission_df.to_csv("/kaggle/working/submission.csv", index=False)
# チェック
print(submission_df.dtypes)
print(submission_df.head())
print(submission_df["label"].min(), submission_df["label"].max())


