import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd


class NasConfig():
    def __init__(self, dir):
        self.root_dir = dir
        self.train_dir = os.path.join(self.root_dir, "train")
        self.test_dir = os.path.join(self.root_dir, "samples")
        self.number_cls = 2
        self.cls_map = {"dog": 0, "cat": 1}#dogやcatのラベルを0と1
        self.index_map = {0: "dog", 1: "cat"}#0,1のデータをdog,catに戻す
        self.read_train()
        self.read_test()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def read_train(self):
        self.train_datas = []
        for file in os.listdir(self.train_dir):
            if file.endswith(".jpg"):
                img = os.path.join(self.train_dir, file)
                label = self.cls_map[file.split(".")[0]]
                self.train_datas.append([img, label])

    #トレインデータと同様の処理を追加
    def read_test(self):
        self.test_datas = []
        for file in os.listdir(self.test_dir):
            if file.endswith(".jpg"):
                img = os.path.join(self.test_dir, file)
                idStr = file.split(".")[0]
                self.test_datas.append([img, int(idStr)])


data_root = "/kaggle/input/vc-master-24-2-dogs-vs-cats"
config = NasConfig(data_root)

print("train datas sample:")
print(config.train_datas[:5])

print("test datas sample:")
print(config.test_datas[:5])


import cv2

def edge_extract(image_pil):
    # PIL画像 → numpyに変換
    img = np.array(image_pil.convert("L"))  # グレースケールに変換
    
    # Sobelフィルターでx,y方向のエッジ検出
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # エッジ強度を計算
    edge_img = np.sqrt(sobelx**2 + sobely**2)

    # 正規化
    edge_img = np.uint8(255 * edge_img / np.max(edge_img))
    
    # numpy → PILに戻す
    edge_pil = Image.fromarray(edge_img)

    # CNN用に3チャンネルに変換
    edge_pil = edge_pil.convert("RGB")

    return edge_pil


class EdgeDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        img = Image.open(img_path)

        # エッジ抽出フィルターをかける
        img = edge_extract(img)

        if self.transform:
            img = self.transform(img)

        return img, label


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225])
])
train_dataset = EdgeDataset(config.train_datas, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

for epoch in range(7):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_acc = calculate_accuracy(model, train_loader, device)

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.4f}")


import pandas as pd
from torch.utils.data import Dataset, DataLoader

# テスト用Datasetクラス
class EdgeTestDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, img_id = self.data_list[idx]
        img = Image.open(img_path)
        img = edge_extract(img)

        if self.transform:
            img = self.transform(img)

        return img, img_id

# テストデータセットとデータローダーの作成
test_dataset = EdgeTestDataset(config.test_datas, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# モデル
model.eval()
all_preds = []
all_ids = []

with torch.no_grad():
    for images, ids in test_loader:
        images = images.to(config.device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_ids.extend(ids.numpy() if torch.is_tensor(ids) else ids)

# 整合性チェック
assert len(all_ids) == len(all_preds), f"IDと予測数が合いません: {len(all_ids)} vs {len(all_preds)}"

# 予測ラベルを文字列に変換
pred_labels = [config.index_map[p] for p in all_preds]

# 提出用CSV作成
submission = pd.DataFrame({
    "id": all_ids,
    "label": pred_labels
})

submission.to_csv("submission.csv", index=False)

print("submission.csv を出力しました")


with open("submission.csv", "r") as f:
    print(f.read()[:500])

