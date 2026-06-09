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


import zipfile
import os

# エラーの原因となっていたディレクトリパスを、ご提示の正しいパスに修正します。
input_dir = '/kaggle/input/dogs-vs-cats-redux-kernels-edition'

# train.zipのパス
train_zip_path = os.path.join(input_dir, 'train.zip')
# test.zipのパス (test1.zipから修正)
test_zip_path = os.path.join(input_dir, 'test.zip')

# --- train.zipを解凍 ---
print(f"Extracting {train_zip_path}...")
try:
    with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
        zip_ref.extractall('./') # カレントディレクトリに'train'フォルダが作成される
    print("train.zip extracted successfully.")
except FileNotFoundError:
    print(f"ERROR: train.zip not found at the specified path: {train_zip_path}")
    print("Please make sure the 'input_dir' variable is correct.")

# --- test.zipを解凍 ---
print(f"Extracting {test_zip_path}...")
try:
    with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
        zip_ref.extractall('./') # カレントディレクトリに'test'フォルダが作成される
    print("test.zip extracted successfully.")
except FileNotFoundError:
    print(f"ERROR: test.zip not found at the specified path: {test_zip_path}")
    print("Please make sure the 'input_dir' variable is correct.")

print("\nUnzipping process finished.")

# 解凍後のファイルを確認（オプション）
print("\nFiles in current directory:")
# ls -l コマンドで詳細を表示してみます
!ls -l ./


import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import os

# --- 1. ファイルリストとラベルの作成 ---
# 解凍した'train'フォルダのパス
train_dir = './train'
# os.listdirでフォルダ内の全ファイル名を取得
train_files = os.listdir(train_dir)
# 'dog.xxxx.jpg'なら1, 'cat.xxxx.jpg'なら0というラベルを作成
# ファイル名に'dog'が含まれているかで判定
labels = [1 if 'dog' in file else 0 for file in train_files]

# フルパスのリストを作成
train_filepaths = [os.path.join(train_dir, f) for f in train_files]


# --- 2. 学習データと検証データに分割 ---
# 80%を学習用、20%を検証用に分割する
train_filepaths, val_filepaths, train_labels, val_labels = train_test_split(
    train_filepaths,
    labels,
    test_size=0.2,       # 20%を検証用にする
    random_state=42,     # 実行結果を再現可能にするための乱数シード
    stratify=labels      # 元のデータセットの犬と猫の比率を保ったまま分割する
)

print(f"Number of training samples: {len(train_filepaths)}")
print(f"Number of validation samples: {len(val_filepaths)}")


# --- 3. 画像の前処理とデータ拡張を定義 ---
# ResNetなどの事前学習モデルを使う場合、ImageNetの学習に使われた平均と標準偏差で正規化するのが一般的です
image_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),      # 画像サイズを224x224に統一
        transforms.RandomHorizontalFlip(),  # 50%の確率で水平反転させる
        transforms.RandomRotation(10),      # -10度から+10度の範囲でランダムに回転させる
        transforms.ToTensor(),              # PIL ImageをPyTorchテンソルに変換
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNetの平均と標準偏差で正規化
    ]),
    # 検証データにはデータ拡張（ランダムな変換）は行わない
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# --- 4. DatasetとDataLoaderの作成 ---
# PyTorchのDatasetクラスを継承して、自作のデータセットクラスを作成
class CatsDogsDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # 画像を読み込む
        img_path = self.filepaths[idx]
        image = Image.open(img_path).convert('RGB') # JPGなのでRGBに変換
        # ラベルを取得
        label = self.labels[idx]
        
        # もしtransformが定義されていれば、画像に適用する
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Datasetオブジェクトを作成
train_dataset = CatsDogsDataset(train_filepaths, train_labels, transform=image_transforms['train'])
val_dataset = CatsDogsDataset(val_filepaths, val_labels, transform=image_transforms['val'])

# DataLoaderを作成
# BATCH_SIZEは、一度にモデルに渡す画像の枚数。CPU環境なので大きすぎない値（例: 16や32）が良い
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("\nDataset and DataLoader have been created successfully.")
print("Now you are ready to build and train the model.")


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time # 時間を計測するためインポート

# --- 0. デバイスの設定 ---
# GPUが利用可能ならGPUを、そうでなければCPUを使用します
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 1. 事前学習済みモデル(ResNet)のロード ---
# ここではResNet50を使います。'weights'引数で学習済みの重みを指定します。
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)


# --- 2. モデルの改造（転移学習のキモ） ---
# (A) まずは、ダウンロードしたモデルの重みを更新しないように凍結します
for param in model.parameters():
    param.requires_grad = False

# (B) ResNetの最後の全結合層(fc)を、新しい層に差し替えます
#     - model.fc.in_featuresで、最後の層への入力ユニット数を取得できます
#     - 出力は「犬」と「猫」の2つなので、2にします
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# モデル全体を、設定したデバイス（CPUまたはGPU）に送ります
model = model.to(device)

print("Pre-trained model loaded and final layer replaced for 2-class classification.")


# --- 3. 損失関数とオプティマイザの定義 ---
# 損失関数：分類問題で一般的に使われるクロスエントロピー損失
criterion = nn.CrossEntropyLoss()

# オプティマイザ：差し替えた層(model.fc)のパラメータだけを学習対象とします
# Adamという効率的なアルゴリズムを使用します
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


# --- 4. 学習ループの実行 ---
NUM_EPOCHS = 3 # CPUなのでエポック数は少なめに設定（時間はかかります）
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time() # エポック開始時間
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 10)

    # --- 学習フェーズ ---
    model.train() # モデルを訓練モードに
    running_loss = 0.0
    running_corrects = 0

    # train_loaderからミニバッチ（32枚ずつのデータ）を取り出す
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 勾配をリセット
        optimizer.zero_grad()

        # 順伝播（モデルに画像を入力して予測を出力）
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1) # 最も確率の高いクラスを予測結果とする
        loss = criterion(outputs, labels) # 損失を計算

        # 逆伝播（損失を元に勾配を計算）
        loss.backward()
        # パラメータを更新
        optimizer.step()

        # 統計情報を記録
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    train_epoch_loss = running_loss / len(train_dataset)
    train_epoch_acc = running_corrects.double() / len(train_dataset)

    print(f'Train Loss: {train_epoch_loss:.4f} Acc: {train_epoch_acc:.4f}')

    # --- 検証フェーズ ---
    model.eval() # モデルを評価モードに
    val_loss = 0.0
    val_corrects = 0

    # 勾配計算は不要なので、torch.no_grad()で囲む
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            
    val_epoch_loss = val_loss / len(val_dataset)
    val_epoch_acc = val_corrects.double() / len(val_dataset)
    
    epoch_end_time = time.time() # エポック終了時間
    epoch_duration = epoch_end_time - epoch_start_time #かかった時間

    print(f'Validation Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
    print(f"Epoch time: {epoch_duration:.2f} seconds") # 時間を表示
    
    # 最高の検証精度が出たモデルの重みを保存
    if val_epoch_acc > best_val_acc:
        best_val_acc = val_epoch_acc
        # 'best_model.pth'という名前でモデルのパラメータを保存
        torch.save(model.state_dict(), 'best_model.pth')
        print("Best model weights saved.")
        
print(f"\nTraining complete. Best validation accuracy was: {best_val_acc:.4f}")


import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np

# --- 0. デバイスの設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} for inference")


# --- 1. モデルの準備 ---
# まずは学習時と全く同じ構造のモデルを準備します
model = models.resnet50(weights=None) # 学習済みの重みは使わないのでNone
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# 保存しておいた最高のモデルの重み(state_dict)を読み込みます
model.load_state_dict(torch.load('best_model.pth'))

# モデルをデバイスに送り、評価モードに切り替えます
model = model.to(device)
model.eval()
print("Best model loaded for inference.")


# --- 2. テストデータの準備 ---
# テストデータ用の前処理を定義します（検証データと全く同じでOK）
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# テストデータ用のDatasetクラス
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
        return image, os.path.basename(img_path)

# テストファイルのパスを取得
test_dir = './test'
test_files = os.listdir(test_dir)
test_filepaths = [os.path.join(test_dir, f) for f in test_files]

# DatasetとDataLoaderを作成
test_dataset = TestDataset(test_filepaths, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# --- 3. 推論の実行 ---
predictions = []
filenames = []

with torch.no_grad(): # 勾配計算は不要
    for inputs, fnames in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1) # 最も確率の高いクラスを取得 (0=cat, 1=dog)
        
        predictions.extend(preds.cpu().numpy())
        filenames.extend(fnames)

print("Inference complete.")


# --- 4. 提出ファイルの作成 ---
# ファイル名からidを取得 (例: '123.jpg' -> 123)
ids = [int(os.path.splitext(fname)[0]) for fname in filenames]

# idと予測結果を紐づけてDataFrameを作成
submission_df = pd.DataFrame({'id': ids, 'label': predictions})

# idの昇順に並べ替える (Kaggleの要件)
submission_df = submission_df.sort_values(by='id')

# CSVファイルとして出力
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' has been created successfully!")
print("You can now submit this file to the Kaggle competition.")
# 最初の5行を表示して確認
print(submission_df.head())

