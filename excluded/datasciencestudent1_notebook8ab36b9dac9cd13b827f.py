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
import torch
#インポートを調査



class NasConfig():
    def __init__(self, dir):
        self.root_dir = dir
        self.train_dir = os.path.join(self.root_dir, "train")
        self.test_dir = os.path.join(self.root_dir, "samples")
        self.number_cls = 2
        self.cls_map = {"dog": 0, "cat": 1}
        self.index_map = {0: "dog", 1: "cat"}
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

    def read_test(self):
        self.test_datas = []
        for file in os.listdir(self.test_dir):
            if file.endswith(".jpg"):
                img = os.path.join(self.test_dir, file)
                # 例：'1234.jpg' → 1234 を取得
                id_str = os.path.splitext(file)[0]
                self.test_datas.append([img, int(id_str)])



data_root = "/kaggle/input/vc-master-24-2-dogs-vs-cats"
config = NasConfig(data_root)


print("train datas sample:")
print(config.train_datas[:5])


print("test datas sample:")
print(config.test_datas[:5])


import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


import cv2  # OpenCVが使える環境なら


def edge_extract(image_pil):
    # PIL画像 → numpyに変換
    img = np.array(image_pil.convert("L"))  # グレースケールに変換

#画像処理ソーベル処理以外も使える。画像の認識方法
    # Sobelフィルターでx,y方向のエッジ検出
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # エッジ強度を計算
    edge_img = np.sqrt(sobelx**2 + sobely**2)
    # 正規化して0~255に戻す
    edge_img = np.uint8(255 * edge_img / np.max(edge_img))
    # numpy → PILに戻す
    edge_pil = Image.fromarray(edge_img)
    # CNN用に3チャンネルに変換（1チャンネルだとCNNの最初の層を変える必要があるので）
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
        # 変換（リサイズ、テンソル化、正規化など）を適用
        if self.transform:
            img = self.transform(img)
        return img, label


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNetの平均・標準偏差を例に
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
        x = self.pool(F.relu(self.conv1(x)))  # 128->64
        x = self.pool(F.relu(self.conv2(x)))  # 64->32
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)#学習率変更する


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


for epoch in range(7):#数値変える。整数値max7ぐらい
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


# --- 1. テスト用Datasetクラス ---
class EdgeTestDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        img_path, img_id = self.data_list[idx]
        img = Image.open(img_path)
        img = edge_extract(img)  # エッジ抽出は訓練と同じ処理
        if self.transform:
            img = self.transform(img)
        return img, img_id


# --- 2. テストデータセットとデータローダーの作成 ---
test_dataset = EdgeTestDataset(config.test_datas, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# --- 3. モデル推論 ---
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


# --- 4. 整合性チェック ---
assert len(all_ids) == len(all_preds), f"IDと予測数が合いません: {len(all_ids)} vs {len(all_preds)}"


# --- 5. 予測ラベルを文字列に変換 ---
pred_labels = [config.index_map[p] for p in all_preds]


# --- 6. 提出用CSV作成 ---
submission = pd.DataFrame({
    "id": all_ids,
    "label": pred_labels
})
submission.to_csv("submission.csv", index=False)
print("submission.csv を出力しました")

