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


import os, zipfile, glob, time
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.models import vit_b_16
import timm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F


# デバイスの設定（GPUを指定）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize = 224 # 画像の入力サイズ
mean = (0.485, 0.456, 0.406) # ImageNetの平均値
std = (0.229, 0.224, 0.225) # ImageNetの標準偏差
batch_size = 32 # バッチサイズ
num_epochs = 10 # エポック数


# データディレクトリのパス設定
base_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"

# 解凍先ディレクトリのパス設定
data_dir = "/kaggle/working/data"
train_dir = "/kaggle/working/data/train"
test_dir = "/kaggle/working/data/test"

# dataディレクトリを作成
os.makedirs(data_dir, exist_ok=True)

# データの解凍
with zipfile.ZipFile(os.path.join(base_dir, "train.zip")) as train_zip:
    train_zip.extractall(data_dir)
with zipfile.ZipFile(os.path.join(base_dir, "test.zip")) as test_zip:
    test_zip.extractall(data_dir)

# 画像ファイル全てのパスを取得してリスト化
train_list = glob.glob(os.path.join(train_dir, "*.jpg"))
test_list = glob.glob(os.path.join(test_dir, "*.jpg"))

# 学習データのリストをシャッフルして、学習:検証 = 8:2に分割する
train_list, val_list = train_test_split(train_list, test_size=0.2)


class ImageTransform():
    # 画像をリサイズ、クロップ、Tensor化、正規化するクラス
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(resize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            "val": transforms.Compose([
                transforms.Resize(int(resize * 1.14)),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)


class ImageDataset(Dataset):
    def __init__(self, file_list, transform=None, phase="train"):
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
        label_str = os.path.basename(img_path).split(".")[0]
        if label_str == "dog":
            label = 1
        elif label_str == "cat":
            label = 0

        return img_transformed, label


# データの前処理
# 前処理インスタンスの作成
transform = ImageTransform(resize, mean, std)

# DatasetとDataLoaderの生成
train_dataset = ImageDataset(train_list, transform, phase="train")
val_dataset = ImageDataset(val_list, transform, phase="val")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

dataloader_dict = {"train": train_dataloader, "val": val_dataloader}


# 事前学習済みのViTを読み込み
net = vit_b_16(weights=True)

for name, param in net.named_parameters():
    if "heads" not in name:
        param.requires_grad = False

# 2クラス分類用にヘッドを置換
in_features = net.heads.head.in_features
net.heads.head = nn.Linear(in_features, 2)

net = net.to(device)


criterion = nn.CrossEntropyLoss() # クロスエントロピー誤差
optimizer = torch.optim.Adam(net.heads.head.parameters(), lr=0.0001) # ヘッドのみ学習


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    # 損失と精度の履歴
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-'*20)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else :
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # ミニバッチを順次取得
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad() # 勾配を初期化
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # 統計量の更新
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # エポックごとの損失・精度を計算して表示
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 履歴に保存
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.cpu().numpy())

            # ベストモデル更新
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    # 学習時間の表示
    time_elapsed = time.time() - since #学習の総時間を計算
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # ベストモデルの重みを返す
    model.load_state_dict(best_model_wts)
    return model, history


net, history = train_model(net, dataloader_dict, criterion, optimizer, num_epochs)


# 損失のプロット
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.legend()

# 精度のプロット
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
plt.legend()

plt.show()


# 変数の初期化
id_list = []
pred_list = []

with torch.no_grad():
    for test_path in tqdm(test_list):
        img = Image.open(test_path) #画像読み込み
        _id = int(test_path.split('/')[-1].split('.')[0]) #ID抽出

        transform = ImageTransform(resize, mean, std)
        img = transform(img, phase='val') #画像変換
        img = img.unsqueeze(0) #バッチ次元の追加
        img = img.to(device) #画像をGPUに転送

        net.eval() #モデルを評価モードに設定

        outputs = net(img) #モデルによる推論
        preds = F.softmax(outputs, dim=1)[:, 1].tolist() #クラス1(dog)の確率を算出
        
        id_list.append(_id) #IDをリストに追加
        pred_list.append(preds[0]) #予測値をリストに追加


res = pd.DataFrame({'id': id_list, 'label': pred_list}) #IDと予測値からデータフレームを作成


res.sort_values(by='id', inplace=True) #ID順に並び替え
res.reset_index(drop=True, inplace=True)

res.to_csv('/kaggle/working/submission.csv', index=False) #データフレームをCSVファイルに保存

