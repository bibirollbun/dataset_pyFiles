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


import glob
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.model_selection import train_test_split
import time
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

# シード固定（再現性確保）
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()



size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 8



# データを初期化
!rm -rf ../data


import zipfile

# データの解凍
base_dir = '../input/dogs-vs-cats-redux-kernels-edition'
train_dir = '../data/train'
test_dir = '../data/test'

os.makedirs('../data', exist_ok=True)

# 学習データの解凍
with zipfile.ZipFile(os.path.join(base_dir, 'train.zip')) as train_zip:
    train_zip.extractall('../data')
    
#推論データの解凍
with zipfile.ZipFile(os.path.join(base_dir, 'test.zip')) as test_zip:
    test_zip.extractall('../data')


train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

# 学習データから検証データを分割(8:2)
train_list, val_list = train_test_split(train_list, test_size=0.2)
# ★画像数を制限して高速化したい場合（例：学習800枚、検証200枚）
train_list = train_list[:800]
val_list = val_list[:200]



class ImageTransform:
    def __init__(self, resize, mean, std):
        self.transforms_dict = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=10),
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            'val': transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        }

    def __call__(self, img, phase):
        return self.transforms_dict[phase](img)



class ImageDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')  # RGB変換を追加
        img = self.transform(img, self.phase)

        label_str = img_path.split('/')[-1].split('.')[0]
        label = 1 if label_str == 'dog' else 0

        return img, label



# データの前処理
train_dataset = ImageDataset(train_list, transform=ImageTransform(size, mean, std), phase='train')
val_dataset = ImageDataset(val_list, transform=ImageTransform(size, mean, std), phase='val')

# データの読み込み
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}


use_pretrained = True
# 事前学習済みのvgg16
net = models.vgg16(pretrained=True)

# 全パラメータ更新停止
for param in net.parameters():
    param.requires_grad = False

# classifierの5層目以降は微調整可能に
for param in net.classifier[5:].parameters():
    param.requires_grad = True

# 出力層のクラス数変更（2クラス）
net.classifier[6] = nn.Linear(4096, 2)

params_to_update = [p for p in net.parameters() if p.requires_grad]
update_params_name = ['classifier.6.weight', 'classifier.6.bias']

for name, param in net.named_parameters():
    if name in update_params_name: #更新するパラメータの場合
        param.requires_grad = True #更新する
        params_to_update.append(param) #更新するパラメータの追加
    else: #更新しないパラメータの場合
        param.requires_grad = False #更新しない


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params_to_update, lr=1e-4)

# 検証精度が改善しなければ学習率を下げる
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

def train_model(net, dataloader_dict, criterion, optimizer, num_epoch):
    
    since = time.time() #学習開始時刻を記録
    best_model_wts = copy.deepcopy(net.state_dict()) #最良モデルの初期化
    best_acc = 0.0 #最良精度の初期化
    
     # 損失と精度の履歴
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    net = net.to(device) #モデルをGPUに転送
    print('training started')
    for epoch in range(num_epoch): #各エポックの進捗
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-'*20)
        
        for phase in ['train', 'val']:
            #学習と検証を交互に行う
            if phase == 'train':
                net.train() #学習フェーズ
            else:
                net.eval() #検証フェーズ
                
            epoch_loss = 0.0
            epoch_corrects = 0
            
            for inputs, labels in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs) #出力を算出
                    _, preds = torch.max(outputs, 1) #予測値の取得
                    loss = criterion(outputs, labels) #損失の計算
                    
                    if phase == 'train':
                        loss.backward() #逆伝播で学習
                        optimizer.step()
                        
                    #バッチごとの損失と精度を累積    
                    epoch_loss += loss.item() * inputs.size(0) 
                    epoch_corrects += torch.sum(preds == labels.data)
            
            #エポックごとの損失と精度の計算
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # 履歴に保存
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.cpu().numpy())
            
            # 最良モデルの保存
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())
                
    time_elapsed = time.time() - since #学習の総時間を計算
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 最良の重みを読み込む
    net.load_state_dict(best_model_wts)
    return net, history


transform = ImageTransform(size, mean, std)

# train_list, val_list は事前に用意されている想定
train_dataset = ImageDataset(train_list, transform=transform, phase='train')
val_dataset = ImageDataset(val_list, transform=transform, phase='val')

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

dataloaders = {'train': train_loader, 'val': val_loader}



def train_model(net, dataloaders, criterion, optimizer, scheduler, num_epochs, device):
    best_acc = 0.0
    best_weights = copy.deepcopy(net.state_dict())
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    net.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for phase in ['train', 'val']:
            net.train() if phase == 'train' else net.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.cpu().numpy())

            if phase == 'val':
                scheduler.step(epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(net.state_dict())

        # 5エポック以降でval_accが改善しなければ早期終了
        if epoch > 5 and all(history['val_acc'][-1] <= acc for acc in history['val_acc'][-4:-1]):
            print(f"Early stopping at epoch {epoch+1}")
            break

    net.load_state_dict(best_weights)
    return net, history



net, history = train_model(net, dataloaders, criterion, optimizer, scheduler, num_epochs, device)

torch.save(net.state_dict(), 'best_model_v2.pth')


import matplotlib.pyplot as plt
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


# ハイパーパラメータの定義（推論・学習共通）
size = 224  # 入力画像のサイズ
mean = (0.485, 0.456, 0.406)  # ImageNetの平均
std = (0.229, 0.224, 0.225)   # ImageNetの標準偏差



class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)



print(len(test_list))       # 画像の数を確認
print(test_list[:5])        # 最初の5件を表示（空なら [] になる）



import os

print(os.listdir('../data/test'))  # 画像ファイル名が出るか？



import glob
test_list = glob.glob('/kaggle/working/data/test/*.jpg')  # 実際に確認できたパスに合わせる
print(f"画像数: {len(test_list)} 枚")



import os

folder = "../data/test"
filenames = os.listdir(folder)

# 拡張子が .jpg のものだけに絞ってパスを作成
test_list = [os.path.join(folder, f) for f in filenames if f.endswith('.jpg')]

print(f"画像数: {len(test_list)} 枚")
print(test_list[:5])  # 最初の5件だけ表示



import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from tqdm import tqdm

# モデルを評価モードにする（1回だけでOK）
net.eval()

# 前処理（transform）もループ外で定義
transform = ImageTransform(size, mean, std)

id_list = []
pred_list = []

with torch.no_grad():
    for test_path in tqdm(test_list):
        try:
            img = Image.open(test_path).convert('RGB')  # カラーモード統一
            _id = int(os.path.basename(test_path).split('.')[0])  # ファイル名からIDを取得

            img = transform(img, phase='val')  # 前処理
            img = img.unsqueeze(0).to(device)  # バッチ次元追加＋GPU転送

            outputs = net(img)
            preds = F.softmax(outputs, dim=1)[:, 1].item()  # クラス1(dog)の確率

            id_list.append(_id)
            pred_list.append(preds)

        except Exception as e:
            print(f"❌ Error on {test_path}: {e}")

# DataFrameにしてCSV出力
res = pd.DataFrame({'id': id_list, 'label': pred_list})
res = res.sort_values('id').reset_index(drop=True)

# 確認（Kaggleでは12500件必要）
print(f"✅ 予測件数: {len(res)} 件")

res.to_csv('/kaggle/working/submission.csv', index=False)




if len(res) != 12500:
    print(f"⚠️ 行数が不足しています: {len(res)} 行")
else:
    print("✅ 正しい提出形式です（12500行）")



res.to_csv('/kaggle/working/submission.csv', index=False)



res.to_csv('/kaggle/working/submission.csv', index=False)



res.to_csv('/kaggle/working/submission.csv', index=False)



res.to_csv('submission.csv', index=False)



import pandas as pd

# テスト用（resが消えてる場合は仮のデータでも可）
# res = pd.DataFrame({'id': [1, 2], 'label': [0.9, 0.1]})

res.to_csv('submission.csv', index=False)

import os
print("カレントディレクトリのファイル一覧:")
print(os.listdir('.'))




from google.colab import files
files.download("submission.csv")


from google.colab import files
files.download('submission.csv')



from google.colab import drive
drive.mount('/content/drive')



from IPython.display import HTML
import base64

with open("submission.csv") as f:
    b64 = base64.b64encode(f.read().encode()).decode()

payload = f'data:text/csv;base64,{b64}'
HTML(f'<a download="submission.csv" href="{payload}" target="_blank">Download submission.csv</a>')


