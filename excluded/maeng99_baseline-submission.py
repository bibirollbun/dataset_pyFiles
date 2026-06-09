!pip install efficientnet_pytorch


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import random
import joblib
from efficientnet_pytorch import EfficientNet

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #GPU 할당

#하이퍼 파라미터 튜닝
CFG = {
    'IMG_SIZE':128, #이미지 사이즈
    'EPOCHS':20, #에포크
    'LEARNING_RATE':1e-4, #학습률
    'BATCH_SIZE':32, #배치사이즈
    'NUM_WORKERS': 1,
    'SEED':42
}


# Seed 고정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED'])


train = pd.read_csv("/kaggle/input/2025-bamboo-summer-competiton-dl-pr/train.csv")
test = pd.read_csv("/kaggle/input/2025-bamboo-summer-competiton-dl-pr/test.csv")

# 라벨 인코딩
le = LabelEncoder()
train["label_encoded"] = le.fit_transform(train["label"])

train.head()


class ButterflyDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, train=True):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = default_loader(img_path)
        if self.transform:
            image = self.transform(image)

        if self.train:
            label = row["label_encoded"]
            return image, label
        else:
            return image, row["filename"]


train_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor(),
])


# train/val 8:2로 stratify 분할
train_df, val_df = train_test_split(
    train,
    test_size=0.2,
    stratify=train["label_encoded"]
)


num_classes = len(le.classes_)
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
model = model.to(device)

train_dataset = ButterflyDataset(train_df, image_dir=os.path.join("/kaggle/input/2025-bamboo-summer-competiton-dl-pr/train"), transform=train_transform)
val_dataset = ButterflyDataset(val_df, image_dir=os.path.join("/kaggle/input/2025-bamboo-summer-competiton-dl-pr/train"), transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=CFG['NUM_WORKERS'])
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=CFG['NUM_WORKERS'])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(CFG['EPOCHS']):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{CFG['EPOCHS']}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}/{CFG['EPOCHS']}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")


test_dataset = ButterflyDataset(test, image_dir=os.path.join("/kaggle/input/2025-bamboo-summer-competiton-dl-pr/test"), transform=test_transform, train=False)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=CFG['NUM_WORKERS'])


model.eval()
predictions = []
filenames = []

with torch.no_grad():
    for images, fnames in tqdm(test_loader, desc="Predicting"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        labels = le.inverse_transform(preds)
        predictions.extend(labels)
        filenames.extend(fnames)


submission = pd.read_csv('/kaggle/input/2025-bamboo-summer-competiton-dl-pr/sample_submission.csv')
submission['label'] = predictions
submission.to_csv(os.path.join("baseline_submission.csv"), index=False)
print("✅ submission.csv 생성 완료!")




