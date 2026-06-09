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


!pip install -q timm

import os, zipfile, random, gc, warnings
import numpy as np
import pandas as pd
import torch
import timm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import torchvision.transforms as transforms

warnings.filterwarnings("ignore")# 警告の無効化


# 再現性を確保するために乱数シードを固定
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

# ハイパーパラメータ設定(画像サイズやバッチサイズなど)
IMG_SIZE = 256            
BATCH_SIZE = 32           
EPOCHS = 5                
FOLDS = 3                     
LR = 2e-4                  
N_TTA = 3               
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# zipファイルの解凍
DATA_DIR = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"
for z in ["train.zip", "test.zip"]:
    with zipfile.ZipFile(f"{DATA_DIR}/{z}") as f:
        f.extractall("/kaggle/working")

TRN_DIR = "/kaggle/working/train"
TST_DIR = "/kaggle/working/test"

# データフレーム作成
train_files = os.listdir(TRN_DIR)
df = pd.DataFrame({
    "filepath": [f"{TRN_DIR}/{f}" for f in train_files],
    "label": [1 if f.startswith("dog") else 0 for f in train_files]
})


# データ拡張の定義
def get_transforms(mode):
    if mode == "train":
        # 学習時のデータ拡張を定義
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # 検証・推論時のデータ拡張を定義
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# 犬・猫データセットクラスの定義
class DogCatDataset(Dataset):
    # コンストラクタ(データ拡張など)
    def __init__(self, df, transform, is_test=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test

    # データ総数を返す
    def __len__(self):
        return len(self.df)

    # 指定のデータを取得
    def __getitem__(self, idx):
        img_path = self.df.filepath[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        if self.is_test:
            img_id = os.path.basename(img_path).split(".")[0]
            return img, img_id
        else:
            label = self.df.label[idx]
            return img, torch.tensor([label], dtype=torch.float32)


class DogCatModel(nn.Module):
    def __init__(self, model_name="resnet34", pretrained=True):# 事前学習済みresnet34を使用
        super().__init__()
        
        # 事前学習済みモデルの読み込み
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0  # 分類層は使用しない
        )
        
        # 出力される特徴ベクトルの次元数を取得
        in_features = self.backbone.num_features
        
        # 分類用の全結合層を定義
        self.head = nn.Sequential(
            nn.Dropout(0.3),             
            nn.Linear(in_features, 1)
        )

    # 入力→特徴量→スコア
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def train_model(model, train_loader, valid_loader):

    # 損失関数と最適化手法を定義
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler() 

    best_acc = 0
    best_weights = None

    for epoch in range(1, EPOCHS + 1):
        model.train()  # 学習モードに切り替え
        train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            # 順伝播と損失計算
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 逆伝播とパラメータ更新
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        scheduler.step()  # 学習率更新

        # 検証
        model.eval()
        correct = 0
        total = 0
        valid_loss = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()

                # 出力にシグモイドをかける。0.5以上を犬とする
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        # 精度と損失の平均を計算
        accuracy = correct / total
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        # 結果を表示
        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.4f}")

        # 現時点で最も精度が高いモデルの重みを保存
        if accuracy > best_acc:
            best_acc = accuracy
            best_weights = model.state_dict().copy()

    # 最も性能の良かった重みに戻す
    model.load_state_dict(best_weights)
    return model


def train_with_folds():
    
    # 学習済みモデルのパスを保存するリスト
    trained_models = []
    
    # ラベルの比率を保ったまま分割(Stratified K-Fold)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    
    # 各Foldごとに学習・検証を実行
    for fold, (train_idx, valid_idx) in enumerate(skf.split(df, df.label)):
        print(f"\n=== Fold {fold+1}/{FOLDS} ===")
        
        # 学習用データローダの作成
        train_loader = DataLoader(
            DogCatDataset(df.iloc[train_idx], get_transforms("train")),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        # 検証用データローダの作成
        valid_loader = DataLoader(
            DogCatDataset(df.iloc[valid_idx], get_transforms("valid")),
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # モデルの作成と、学習の実行
        model = DogCatModel().to(DEVICE)
        model = train_model(model, train_loader, valid_loader)
        
        # 学習済みモデルを保存
        model_path = f"dogcat_model_fold{fold}.pt"
        torch.save(model.state_dict(), model_path)
        trained_models.append(model_path)
        
        # メモリを解放
        del model, train_loader, valid_loader
        gc.collect()
        torch.cuda.empty_cache()
    
    # 全Foldの学習済みモデルのパスを返す
    return trained_models


def predict(model_paths):
    """テストデータに対して予測を行い、提出ファイルを作成する関数"""
    
    # ID順にソートしてデータフレームを作成
    test_files = sorted(os.listdir(TST_DIR), key=lambda x: int(x.split('.')[0]))
    test_df = pd.DataFrame({"filepath": [f"{TST_DIR}/{f}" for f in test_files]})
    
    # 全モデルの予測結果を格納するリスト
    all_predictions = []
    
    for model_path in model_paths:
        print(f"Predicting with model: {model_path}")
        
        # モデルの読み込み
        model = DogCatModel().to(DEVICE)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # テスト時のTTAごとの予測を格納
        tta_predictions = []
        
        for _ in range(N_TTA):
            # データローダを作成
            test_loader = DataLoader(
                DogCatDataset(test_df, get_transforms("valid"), is_test=True),
                batch_size=BATCH_SIZE * 2,
                shuffle=False,
                num_workers=2
            )
            
            # 予測結果を保存するリスト
            predictions = []
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(DEVICE)
                    with autocast():
                        outputs = model(images)
                        outputs = torch.sigmoid(outputs).squeeze().cpu().numpy()  # 確率に変換
                    predictions.extend(outputs)
            
            # 1回分のTTA結果を保存
            tta_predictions.append(np.array(predictions))
        
        # TTA結果の平均を計算
        avg_predictions = np.mean(tta_predictions, axis=0)
        all_predictions.append(avg_predictions)
        
        # メモリ解放
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    # アンサンブル学習(全モデルの平均予測値を計算)
    final_predictions = np.mean(all_predictions, axis=0)
    
    # 予測値を0.01〜0.99の範囲にする
    final_predictions = np.clip(final_predictions, 0.01, 0.99)
    
    # 提出用のデータフレームを作成
    submission = pd.DataFrame({
        "id": [f.split('.')[0] for f in test_files],
        "label": final_predictions
    })
    
    # CSVファイルとして保存
    submission.to_csv("submission.csv", index=False)
    print("Submission file created!")
    
    return submission


if __name__ == "__main__":
    # モデルの学習
    trained_model_paths = train_with_folds()
    
    # テストデータに対して予測を行い、提出用ファイルを作成
    submission = predict(trained_model_paths)
    
    # 先頭5行だけ表示して確認
    print("\nSubmission Preview:")
    print(submission.head())

