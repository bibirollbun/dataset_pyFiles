import pandas as pd
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms, datasets
from sklearn.metrics import f1_score


# パスの定義
BASE_DIR = '/kaggle/input/2025-iyatomi-lab-competition' 
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
SAMPLE_CSV_PATH = os.path.join(BASE_DIR, 'sample_submission.csv')

# 前処理に関するハイパラ
SEED_VALUE = 42
VALID_SIZE = 0.1
INPUT_SIZE = 224

# 学習に関するハイパラ
LEARNING_RATE = 1e-5
EPOCHS = 10
BATCH_SIZE = 256

# シード値の固定
def set_seed(seed=SEED_VALUE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed()


class TrainTransforms:
    ''' 学習データをモデルへ入力する際に適応する変換'''
    def __init__(self, input_size):
        self.transforms = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]
        )
    def __call__(self, image):
        return self.transforms(image)

class TestTransforms:
    ''' 検証・テストデータをモデルへ入力する際に適応する変換'''
    def __init__(self, input_size):
        self.transforms = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]
        )
    def __call__(self, image):
        return self.transforms(image)


# ImageFolderrで 'train/' 全体を読み込む
full_train_dataset = datasets.ImageFolder(
    TRAIN_DIR, 
    transform=TestTransforms(input_size=INPUT_SIZE)
)
class_names = full_train_dataset.classes
num_classes = len(class_names)
print(f"クラス数: {num_classes}")
print(f"クラス名: {class_names}")

# 学習データと検証データに分割
n_samples = len(full_train_dataset)
n_val = int(n_samples * VALID_SIZE)
n_train = n_samples - n_val

train_dataset_subset, valid_dataset_subset = random_split(
    full_train_dataset, 
    [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED_VALUE)
)
print(f"Trainデータ: {len(train_dataset_subset)}枚, Validデータ: {len(valid_dataset_subset)}枚")

train_dataset_subset.dataset.transform = TrainTransforms(input_size=INPUT_SIZE)


# TestDataset
test_df = pd.read_csv(SAMPLE_CSV_PATH, dtype={'id': str})
test_df['path'] = test_df['id'].apply(lambda x: os.path.join(TEST_DIR, x))

class TestDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image

test_dataset = TestDataset(test_df['path'], TestTransforms(input_size=INPUT_SIZE))


# データローダーの作成
train_dataloader = DataLoader(
    train_dataset_subset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    pin_memory=True, 
    num_workers=os.cpu_count()
)

valid_dataloader = DataLoader(
    valid_dataset_subset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    pin_memory=True, 
    num_workers=os.cpu_count()
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    pin_memory=True, 
    num_workers=os.cpu_count()
)


images, labels = next(iter(train_dataloader))
plt.figure(figsize=(16, 8))
for i in range(min(8, BATCH_SIZE)):
    plt.subplot(2, 4, i + 1)
    img = images[i].permute(1, 2, 0).numpy()
    img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(f"Class: {class_names[labels[i]]}")
    plt.axis('off')
plt.show()


class CNN(torch.nn.Module):
    ''' 5×5の畳み込みカーネルを持つ3層のCNN'''
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
        ) # -> (B, 8, 112, 112)
        
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
        ) # -> (B, 16, 56, 56)
        
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
        ) # -> (B, 32, 28, 28)
        
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # -> (B, 32, 1, 1)
        
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = torch.nn.Linear(32, output_size)
            
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.adaptive_pool(x) # (B, 32, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 32)
        x = self.dropout(x)
        x = self.linear(x)
        return x

# モデルの定義
model = CNN(input_size=INPUT_SIZE, output_size=num_classes)


def train(model, train_dataloader, valid_dataloader, optimizer, criterion, epochs):
    ''' 学習及び検証を行う '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'使用デバイス：{device}')

    # 高速化設定
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()

    print('Start training...')
    best_model_state = model.state_dict()
    best_f1 = 0.0
    loss_history, f1_history = {'train': [], 'valid': []}, {'train': [], 'valid': []}

    for epoch in range(epochs):
        print(f'\nepoch [{epoch+1}/{epochs}]')

        # 学習フェーズ
        model.train()
        epoch_preds, epoch_labels = [], []
        epoch_losses = 0.0

        for images, labels in tqdm(train_dataloader, desc="Training"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # 混合精度 (autocast) で forward
            with torch.autocast(device_type=device.type):
                preds_proba = model(images)
                loss = criterion(preds_proba, labels)

            # backward(重みの最適化)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses += loss.item()
            preds = torch.argmax(preds_proba, dim=1)
            epoch_preds.append(preds.cpu().detach())
            epoch_labels.append(labels.cpu().detach())
            del preds_proba, preds, loss, images, labels

        epoch_preds = torch.cat(epoch_preds, 0)
        epoch_labels = torch.cat(epoch_labels, 0)
        epoch_losses /= len(train_dataloader)
        epoch_f1 = f1_score(epoch_labels, epoch_preds, average='macro', zero_division=0) # zero_division追加
        loss_history['train'].append(epoch_losses)
        f1_history['train'].append(epoch_f1)

        # 検証フェーズ
        model.eval()
        epoch_preds, epoch_labels = [], []
        epoch_losses = 0.0

        with torch.inference_mode():
            for images, labels in tqdm(valid_dataloader, desc="Validation"):
                images = images.to(device)
                labels = labels.to(device)

                with torch.autocast(device_type=device.type):
                    preds_proba = model(images)
                    loss = criterion(preds_proba, labels)

                epoch_losses += loss.item()
                preds = torch.argmax(preds_proba, dim=1)
                epoch_preds.append(preds.cpu().detach())
                epoch_labels.append(labels.cpu().detach())
                del preds_proba, preds, loss, images, labels

            epoch_preds = torch.cat(epoch_preds, 0)
            epoch_labels = torch.cat(epoch_labels, 0)
            epoch_losses /= len(valid_dataloader)
            epoch_f1 = f1_score(epoch_labels, epoch_preds, average='macro', zero_division=0) # zero_division追加
            loss_history['valid'].append(epoch_losses)
            f1_history['valid'].append(epoch_f1)

        # 最良モデルの保存
        if best_f1 < f1_history["valid"][-1]:
            best_model_state = model.state_dict()
            best_f1 = f1_history["valid"][-1]
            print(f"✨ New Best F1: {best_f1:.4f} (Model saved)")

        print(f'train_loss:{loss_history["train"][-1]:.4f}, train_f1:{f1_history["train"][-1]:.4f}')
        print(f'valid_loss:{loss_history["valid"][-1]:.4f}, valid_f1:{f1_history["valid"][-1]:.4f}')

    model.load_state_dict(best_model_state)
    print(f'\nFinish training! Best Valid F1: {best_f1:.4f}')
    return model, loss_history, f1_history


# 最適化器・損失関数の定義
optimizer = torch.optim.RAdam(params=model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# 学習
model, loss_history, f1_history = train(
    model, 
    train_dataloader, 
    valid_dataloader, 
    optimizer, 
    criterion, 
    EPOCHS
)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history['train'], label='train loss', c='b')
plt.plot(loss_history['valid'], label='val loss', c='r')
plt.title('Cross Entropy Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(f1_history['train'], label='train F1', c='b')
plt.plot(f1_history['valid'], label='val F1', c='r')
plt.title('Macro-F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# モデルを保存
torch.save(model.cpu().state_dict(), 'model.pth')


def test(model, test_dataloader, class_names):
    ''' テストを行う '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'使用デバイス：{device}')

    print('Start testing...')
    model.eval()
    pred_labels = [] # 予測した「ラベル名」を格納

    with torch.inference_mode():
        for images in tqdm(test_dataloader, desc="Testing"):
            images = images.to(device, non_blocking=True)
            
            with torch.autocast(device_type=device.type):
                preds_proba = model(images)

            preds_idx = torch.argmax(preds_proba, dim=1)
            
            # インデックスをクラス名に変換
            for idx in preds_idx.cpu().numpy():
                pred_labels.append(class_names[idx])
                
            del preds_proba, preds_idx, images

    print('Finish testing!')
    return pred_labels


# テストデータの予測
predicted_class_names = test(model, test_dataloader, class_names)

# submission.csvの書き出し
sample_df = pd.read_csv(SAMPLE_CSV_PATH, dtype={'id': str})

sample_df['label'] = predicted_class_names

sample_df.to_csv('submission.csv', index=False, header=True)
sample_df.head()

