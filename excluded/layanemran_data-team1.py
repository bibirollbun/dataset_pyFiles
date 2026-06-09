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


import pandas as pd

# Ù…Ø³Ø§Ø± Ù…Ù„Ù� Ø§Ù„ØªØµÙ†ÙŠÙ�Ø§Øª
csv_path = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv'

# Ù‚Ø±Ø§Ø¡Ø© Ù…Ù„Ù� CSV
df = pd.read_csv(csv_path)

# Ù†Ø¸Ù� Ø£Ø³Ù…Ø§Ø¡ Ø§Ù„Ø£Ø¹Ù…Ø¯Ø© (Ù„Ùˆ Ù�ÙŠÙ‡Ø§ Ù�Ø±Ø§ØºØ§Øª Ø£Ùˆ Ø£Ø­Ø±Ù� ÙƒØ¨ÙŠØ±Ø©)
df.columns = df.columns.str.strip().str.lower()

# Ø·Ø¨Ø§Ø¹Ø© Ø§Ù„Ø£Ø¹Ù…Ø¯Ø© Ø¹Ø´Ø§Ù† Ù†ØªØ£ÙƒØ¯ Ø£Ø³Ù…Ø§Ø¤Ù‡Ø§
print("Ø£Ø¹Ù…Ø¯Ø© Ø§Ù„Ù…Ù„Ù�:", df.columns)

# ØªÙ†Ø¸ÙŠÙ� Ø§Ù„Ù‚ÙŠÙ… Ø§Ù„Ù…Ù‡Ù…Ø©
if 'image' in df.columns:
    df['image'] = df['image'].str.strip()
elif 'image_name' in df.columns:
    df['image_name'] = df['image_name'].str.strip()
else:
    print("Ù„Ø§ ÙŠÙˆØ¬Ø¯ Ø¹Ù…ÙˆØ¯ Ø¨Ø§Ø³Ù… 'image' Ø£Ùˆ 'image_name' Ù�ÙŠ Ø§Ù„Ù…Ù„Ù�")

if 'label' in df.columns:
    df['label'] = df['label'].str.strip().str.lower()
else:
    print("Ù„Ø§ ÙŠÙˆØ¬Ø¯ Ø¹Ù…ÙˆØ¯ Ø¨Ø§Ø³Ù… 'label' Ù�ÙŠ Ø§Ù„Ù…Ù„Ù�")

# Ø¹Ø±Ø¶ Ø£ÙˆÙ„ 5 ØµÙ�ÙˆÙ�
print(df.head())

# Ø·Ø¨Ø§Ø¹Ø© Ø§Ù„Ù�Ø¦Ø§Øª ÙˆØ¹Ø¯Ø¯Ù‡Ø§
if 'label' in df.columns:
    unique_labels = sorted(df['label'].unique())
    print(f"Ø¹Ø¯Ø¯ Ø§Ù„Ù�Ø¦Ø§Øª: {len(unique_labels)}")
    print("Ø§Ù„Ù�Ø¦Ø§Øª:", unique_labels)
    print("\nØ¹Ø¯Ø¯ Ø§Ù„Ø¹ÙŠÙ†Ø§Øª Ù„ÙƒÙ„ Ù�Ø¦Ø©:")
    print(df['label'].value_counts())



import pandas as pd

csv_path = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv'
df = pd.read_csv(csv_path)

# ØªÙ†Ø¸ÙŠÙ� Ø§Ù„Ø£Ø¹Ù…Ø¯Ø© Ø§Ù„Ù…Ù‡Ù…Ø©
df['filename'] = df['filename'].str.strip()
df['label'] = df['label'].str.strip().str.lower()

# ØªØ£ÙƒÙŠØ¯ Ø§Ù„Ù†Ø¸Ø§Ù�Ø©
print(df.head())
print(df['label'].value_counts())



# ØªØ±ØªÙŠØ¨ Ø§Ù„Ù�Ø¦Ø§Øª Ø£Ø¨Ø¬Ø¯ÙŠØ§Ù‹ (Ø§Ø®ØªÙŠØ§Ø±ÙŠ Ø¨Ø³ Ù…Ù�ÙŠØ¯ Ù„Ù„Ø«Ø¨Ø§Øª)
classes = sorted(df['label'].unique())

label_to_idx = {label: idx for idx, label in enumerate(classes)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

print("label_to_idx:", label_to_idx)



import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class SheepDataset(Dataset):
    def __init__(self, dataframe, img_dir, label_to_idx, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'filename']
        label_name = self.df.loc[idx, 'label']
        label_idx = self.label_to_idx[label_name]

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label_idx



from sklearn.model_selection import KFold

# Ø¥Ø¶Ø§Ù�Ø© Ø¹Ù…ÙˆØ¯ Ø¬Ø¯ÙŠØ¯ Ù�Ø§Ø±Øº Ø§Ø³Ù…Ù‡ 'fold'
df['fold'] = -1

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    df.loc[val_idx, 'fold'] = fold

print(df['fold'].value_counts())



from torchvision import transforms

# Ù†Ø¬Ù‡Ø² Ø§Ù„ØªØ­ÙˆÙŠÙ„Ø§Øª (transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ø­Ø¬Ù… Ø«Ø§Ø¨Øª Ù„Ù„ØµÙˆØ±
    transforms.ToTensor(),          # ØªØ­ÙˆÙŠÙ„ Ø§Ù„ØµÙˆØ±Ø© Ø¥Ù„Ù‰ Tensor
    transforms.Normalize([0.5]*3, [0.5]*3)  # ØªØ·Ø¨ÙŠØ¹ ØªÙ‚Ø±ÙŠØ¨ÙŠ
])

# Ù…Ø³Ø§Ø± Ø§Ù„ØµÙˆØ±
img_dir = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train/'

# ØªÙ‚Ø³ÙŠÙ… Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ø¨Ù†Ø§Ø¡Ù‹ Ø¹Ù„Ù‰ fold
train_df = df[df['fold'] != 0].reset_index(drop=True)
val_df = df[df['fold'] == 0].reset_index(drop=True)

# Ø¥Ù†Ø´Ø§Ø¡ Ø§Ù„Ù€ Dataset
train_dataset = SheepDataset(train_df, img_dir, label_to_idx, transform=transform)
val_dataset = SheepDataset(val_df, img_dir, label_to_idx, transform=transform)

# Ø§Ù„ØªØ­Ù‚Ù‚ Ù…Ù† Ø£ÙˆÙ„ Ø¹Ù†ØµØ±
img, label = train_dataset[0]
print("Ø´ÙƒÙ„ Ø§Ù„ØµÙˆØ±Ø©:", img.shape)
print("Ø§Ù„Ù�Ø¦Ø© (Ø±Ù‚Ù…):", label)



from torch.utils.data import DataLoader

# Ø¥Ø¹Ø¯Ø§Ø¯Ø§Øª Ø§Ù„ØªØ­Ù…ÙŠÙ„
BATCH_SIZE = 32  # Ø­Ø³Ø¨ Ø§Ù„Ù…ØªØ·Ù„Ø¨Ø§Øª

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ØªØ¬Ø±Ø¨Ø©: Ù†Ø´ÙˆÙ� Ø£ÙˆÙ„ batch
images, labels = next(iter(train_loader))
print("Batch shape:", images.shape)  # Ù†ØªÙˆÙ‚Ø¹: [32, 3, 224, 224]
print("Batch labels:", labels[:10])



import torch
import torchvision.models as models
import torch.nn as nn

# Ù†Ø­Ø¯Ø¯ Ø¹Ø¯Ø¯ Ø§Ù„Ù�Ø¦Ø§Øª
num_classes = len(label_to_idx)

# ØªØ­Ù…ÙŠÙ„ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬ Ø§Ù„Ø£Ø³Ø§Ø³ÙŠ Ù…Ø¹ Ø§Ù„ÙˆØ²Ù†Ø§Øª Ø§Ù„Ù…Ø³Ø¨Ù‚Ø©
model = models.convnext_tiny(pretrained=True)

# ØªØ¹Ø¯ÙŠÙ„ Ø§Ù„Ø·Ø¨Ù‚Ø© Ø§Ù„Ù†Ù‡Ø§Ø¦ÙŠØ© Ù„ØªÙ†Ø§Ø³Ø¨ Ø¹Ø¯Ø¯ Ø§Ù„Ù�Ø¦Ø§Øª
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

# Ù†Ù‚Ù„ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬ Ø¥Ù„Ù‰ Ø§Ù„Ø¬Ù‡Ø§Ø² Ø§Ù„Ù…Ù†Ø§Ø³Ø¨ (GPU Ø£Ùˆ CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Ø·Ø¨Ø§Ø¹Ø© Ù„Ù„ØªØ£ÙƒØ¯
print(model.classifier)



criterion = nn.CrossEntropyLoss()



from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)



from torch.optim.lr_scheduler import CosineAnnealingLR

# Ù†Ø¨Ø¯Ø£ Ø¨Ø¹Ø¯Ø¯ epochs Ù…Ø­Ø¯Ø¯
EPOCHS = 30
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)



import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold
from torchvision import transforms
from torchvision.models import convnext_tiny
from PIL import Image
import pandas as pd

# -- Ø¥Ø¹Ø¯Ø§Ø¯ Dataset --
class SheepDataset(Dataset):
    def __init__(self, df, root_dir, label_to_idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'filename']
        label_name = self.df.loc[idx, 'label']
        label_idx = self.label_to_idx[label_name]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label_idx

# -- ØªØ­Ù…ÙŠÙ„ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª --
csv_file = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv'
img_dir = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train/'

df = pd.read_csv(csv_file)

# ØªØ­ÙˆÙŠÙ„ Ø£Ø³Ù…Ø§Ø¡ Ø§Ù„Ù�Ø¦Ø§Øª Ø¥Ù„Ù‰ Ø£Ø±Ù‚Ø§Ù…
labels = df['label'].unique()
label_to_idx = {label: idx for idx, label in enumerate(labels)}
df['label_idx'] = df['label'].map(label_to_idx)

# Ø§Ø³ØªØ®Ø±Ø§Ø¬ Ø§Ù„Ù…Ø¬Ù…ÙˆØ¹Ø© (Group) Ù„ØªÙ‚Ø³ÙŠÙ… Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª
df['group'] = df['filename'].str.extract(r'(^\w+)', expand=False)

# -- Ø¥Ø¹Ø¯Ø§Ø¯ Ø§Ù„ØªØ­ÙˆÙŠÙ„Ø§Øª --
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = SheepDataset(df, img_dir, label_to_idx, transform=transform)

# -- Ø¥Ø¹Ø¯Ø§Ø¯ ØªÙ‚Ø³ÙŠÙ… Ø§Ù„Ù�ÙˆÙ„Ø¯ --
gkf = GroupKFold(n_splits=5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

start_fold = 0
start_epoch = 0
model = None
optimizer = None
scheduler = None

for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df['label_idx'], groups=df['group'])):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_fold{fold}.pt')

    if fold < start_fold:
        continue

    print(f'\n--- Fold {fold + 1}/5 ---')

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    # Ø¥Ø¹Ø¯Ø§Ø¯ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬ (Ø£Ùˆ Ø¥Ø¹Ø§Ø¯Ø© Ø§Ù„ØªØ­Ù…ÙŠÙ„)
    if fold > start_fold or model is None:
        model = convnext_tiny(weights='IMAGENET1K_V1')
        model.classifier[2] = nn.Linear(in_features=768, out_features=len(label_to_idx))
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        start_epoch = 0

    if fold == start_fold and os.path.exists(checkpoint_path):
        print(f'ğŸ”„ Ø§Ø³ØªØ¹Ø§Ø¯Ø© checkpoint Ù„Ù„Ù�ÙˆÙ„Ø¯ {fold}...')
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])

    for epoch in range(start_epoch, 30):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels_batch in train_loader:
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            running_corrects += (preds == labels_batch).sum().item()

        epoch_loss = running_loss / len(train_subset)
        epoch_acc = running_corrects / len(train_subset)

        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels_batch in val_loader:
                inputs = inputs.to(device)
                labels_batch = labels_batch.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                val_corrects += (preds == labels_batch).sum().item()

        val_acc = val_corrects / len(val_subset)

        print(f'Epoch [{epoch + 1}/30] - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f} - Val Acc: {val_acc:.4f}')

        scheduler.step()

        # Ø­Ù�Ø¸ checkpoint Ø¨Ø¹Ø¯ ÙƒÙ„ Ø¥Ø¨ÙˆÙƒ
        checkpoint = {
            'fold': fold,
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

    start_epoch = 0



import matplotlib.pyplot as plt
import numpy as np


folds_data = {
    "fold_1": {
  "loss": [
    1.3265, 0.6677, 0.3955, 0.3577, 0.2742, 0.2626, 0.2617, 0.1961, 0.1968, 0.2829,
    0.2084, 0.1494, 0.1269, 0.1944, 0.1602, 0.1103, 0.1787, 0.1159, 0.1363, 0.0902,
    0.1270, 0.1198, 0.1153, 0.1052, 0.0928, 0.0789, 0.0663, 0.0992, 0.0766, 0.1241
  ],
  "train_acc": [
    0.5450, 0.8110, 0.9028, 0.8954, 0.9248, 0.9064, 0.9193, 0.9450, 0.9450, 0.9009,
    0.9303, 0.9596, 0.9633, 0.9450, 0.9541, 0.9670, 0.9376, 0.9688, 0.9651, 0.9743,
    0.9578, 0.9670, 0.9688, 0.9670, 0.9780, 0.9798, 0.9817, 0.9651, 0.9798, 0.9633
  ],
  "val_acc": [
    0.6788, 0.7883, 0.7810, 0.8613, 0.8759, 0.8540, 0.8832, 0.8905, 0.8759, 0.8905,
    0.9051, 0.9270, 0.9197, 0.8832, 0.8832, 0.8613, 0.8978, 0.8905, 0.9051, 0.8905,
    0.8978, 0.8832, 0.9343, 0.8686, 0.9343, 0.9197, 0.9416, 0.9051, 0.9051, 0.9124
  ]
},

    "fold_2": {
  "loss": [
    1.3976, 0.6967, 0.4432, 0.3304, 0.2845, 0.2429, 0.2360, 0.1477, 0.1727, 0.1712,
    0.1616, 0.1523, 0.1529, 0.1243, 0.1124, 0.1155, 0.1132, 0.1015, 0.0943, 0.1084,
    0.0821, 0.0858, 0.0561, 0.0786, 0.1044, 0.0900, 0.0747, 0.0593, 0.0729, 0.0593
  ],
  "train_acc": [
    0.5138, 0.8073, 0.8899, 0.8991, 0.9138, 0.9266, 0.9266, 0.9578, 0.9450, 0.9394,
    0.9560, 0.9578, 0.9541, 0.9596, 0.9743, 0.9578, 0.9688, 0.9651, 0.9670, 0.9651,
    0.9743, 0.9780, 0.9853, 0.9725, 0.9633, 0.9743, 0.9761, 0.9872, 0.9798, 0.9798
  ],
  "val_acc": [
    0.7153, 0.8394, 0.8613, 0.8978, 0.8540, 0.8613, 0.8905, 0.8978, 0.9051, 0.8978,
    0.9416, 0.9051, 0.9124, 0.9124, 0.9197, 0.9124, 0.8759, 0.9197, 0.9343, 0.9124,
    0.9051, 0.9197, 0.9197, 0.9124, 0.9051, 0.8978, 0.9416, 0.8759, 0.8905, 0.9197
  ]
},
    "fold_3": {
  "loss": [
    1.2681, 0.6539, 0.4637, 0.3291, 0.2619, 0.2405, 0.2026, 0.2168, 0.1679, 0.1431,
    0.1619, 0.1952, 0.1271, 0.1207, 0.1610, 0.1227, 0.1084, 0.1065, 0.0902, 0.0747,
    0.0863, 0.0819, 0.0878, 0.0910, 0.0830, 0.0523, 0.0721, 0.0668, 0.0850, 0.0674
  ],
  "train_acc": [
    0.5678, 0.7985, 0.8608, 0.8993, 0.9286, 0.9304, 0.9396, 0.9286, 0.9432, 0.9579,
    0.9487, 0.9396, 0.9707, 0.9725, 0.9524, 0.9634, 0.9707, 0.9597, 0.9707, 0.9835,
    0.9799, 0.9707, 0.9762, 0.9725, 0.9762, 0.9835, 0.9780, 0.9780, 0.9725, 0.9835
  ],
  "val_acc": [
    0.7206, 0.8382, 0.8309, 0.8603, 0.8897, 0.8603, 0.8676, 0.8971, 0.8971, 0.8603,
    0.8603, 0.9044, 0.8529, 0.8603, 0.9191, 0.8971, 0.9044, 0.8897, 0.8971, 0.8971,
    0.8897, 0.8971, 0.9118, 0.9044, 0.9044, 0.8971, 0.9044, 0.8603, 0.8824, 0.9118
  ]
},
    "fold_4": {
  "loss": [
    1.3977, 0.6819, 0.4877, 0.3749, 0.2767, 0.2341, 0.2103, 0.1819, 0.1803, 0.1506,
    0.1567, 0.1233, 0.1053, 0.1070, 0.1669, 0.1457, 0.1276, 0.0856, 0.1131, 0.1204,
    0.0603, 0.1037, 0.0877, 0.0648, 0.0847, 0.0731, 0.0840, 0.0817, 0.0713, 0.0673
  ],
  "train_acc": [
    0.4817, 0.8059, 0.8645, 0.8810, 0.9176, 0.9322, 0.9377, 0.9542, 0.9487, 0.9487,
    0.9487, 0.9615, 0.9689, 0.9652, 0.9469, 0.9560, 0.9524, 0.9689, 0.9652, 0.9652,
    0.9890, 0.9670, 0.9707, 0.9817, 0.9799, 0.9817, 0.9744, 0.9707, 0.9799, 0.9835
  ],
  "val_acc": [
    0.7279, 0.8235, 0.7941, 0.8382, 0.8676, 0.8529, 0.8750, 0.8235, 0.8897, 0.8897,
    0.8971, 0.8750, 0.8824, 0.8897, 0.8971, 0.8824, 0.8971, 0.8971, 0.9118, 0.9191,
    0.8971, 0.8897, 0.9265, 0.9338, 0.9338, 0.8971, 0.8971, 0.9191, 0.9118, 0.8897
  ]
},
    "fold_5": {
  "loss": [
    1.3796, 0.7272, 0.4602, 0.3741, 0.2856, 0.2890, 0.3954, 0.2200, 0.1931, 0.1829,
    0.1421, 0.1603, 0.1338, 0.1185, 0.1170, 0.1073, 0.1024, 0.0774, 0.0807, 0.0962,
    0.0901, 0.0778, 0.0621, 0.0648, 0.0910, 0.0682, 0.0640, 0.0894, 0.0740, 0.0546
  ],
  "train_acc": [
    0.5037, 0.7821, 0.8736, 0.8956, 0.9231, 0.8993, 0.8810, 0.9377, 0.9359, 0.9432,
    0.9579, 0.9432, 0.9542, 0.9707, 0.9634, 0.9707, 0.9689, 0.9835, 0.9762, 0.9670,
    0.9707, 0.9799, 0.9872, 0.9835, 0.9744, 0.9817, 0.9890, 0.9725, 0.9780, 0.9853
  ],
  "val_acc": [
    0.7500, 0.8529, 0.8750, 0.8603, 0.8162, 0.8603, 0.8897, 0.8676, 0.8824, 0.9044,
    0.9265, 0.9265, 0.8897, 0.9338, 0.9559, 0.9338, 0.9118, 0.9706, 0.8676, 0.9044,
    0.9338, 0.8750, 0.9412, 0.9191, 0.9191, 0.9265, 0.9265, 0.9412, 0.9118, 0.9559
  ]
}
}

epochs = range(1, 31)

# Ø±Ø³Ù… Loss Ù„ÙƒÙ„ Ù�ÙˆÙ„Ø¯
plt.figure(figsize=(14, 5))
for fold in folds_data:
    plt.plot(epochs, folds_data[fold]["loss"], label=f"{fold} Loss")
plt.title("Training Loss per Epoch for Each Fold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Ø±Ø³Ù… Train Accuracy Ù„ÙƒÙ„ Ù�ÙˆÙ„Ø¯
plt.figure(figsize=(14, 5))
for fold in folds_data:
    plt.plot(epochs, folds_data[fold]["train_acc"], label=f"{fold} Train Accuracy")
plt.title("Training Accuracy per Epoch for Each Fold")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Ø±Ø³Ù… Validation Accuracy Ù„ÙƒÙ„ Ù�ÙˆÙ„Ø¯
plt.figure(figsize=(14, 5))
for fold in folds_data:
    plt.plot(epochs, folds_data[fold]["val_acc"], label=f"{fold} Validation Accuracy")
plt.title("Validation Accuracy per Epoch for Each Fold")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Ø­Ø³Ø§Ø¨ Ø§Ù„Ù…ØªÙˆØ³Ø· Ù„ÙƒÙ„ Ø¥Ø¨ÙˆÙƒ Ø¹Ø¨Ø± Ø§Ù„Ù�ÙˆÙ„Ø¯Ø²
avg_loss = np.mean([folds_data[fold]["loss"] for fold in folds_data], axis=0)
avg_train_acc = np.mean([folds_data[fold]["train_acc"] for fold in folds_data], axis=0)
avg_val_acc = np.mean([folds_data[fold]["val_acc"] for fold in folds_data], axis=0)

# Ø±Ø³Ù… Ø§Ù„Ù…ØªÙˆØ³Ø·
plt.figure(figsize=(14, 5))
plt.plot(epochs, avg_loss, label="Average Loss", color="red", linewidth=2)
plt.title("Average Training Loss per Epoch Across All Folds")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(epochs, avg_train_acc, label="Average Train Accuracy", color="green", linewidth=2)
plt.title("Average Training Accuracy per Epoch Across All Folds")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(epochs, avg_val_acc, label="Average Validation Accuracy", color="blue", linewidth=2)
plt.title("Average Validation Accuracy per Epoch Across All Folds")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()



import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder

# Ø¥Ø¹Ø¯Ø§Ø¯Ø§Øª
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 1e-4

# Ø§Ù„Ù…Ø³Ø§Ø±Ø§Øª
img_dir = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train/'
csv_path = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv'

# ØªØ­Ù…ÙŠÙ„ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª
df = pd.read_csv(csv_path)

# ØªØ±ØªÙŠØ¨ Ø§Ù„ØªØµÙ†ÙŠÙ�Ø§Øª ÙŠØ¯ÙˆÙŠÙ‹Ø§
label_to_idx = {'Barbari': 0, 'Goat': 1, 'Harri': 2, 'Naeimi': 3, 'Najdi': 4, 'Roman': 5, 'Sawakni': 6}
idx_to_label = {v: k for k, v in label_to_idx.items()}
df['label'] = df['label'].map(label_to_idx)
num_classes = len(label_to_idx)

# Ø§Ù„ØªØ­ÙˆÙŠÙ„Ø§Øª (Ù†Ù�Ø³ Ø¥Ø¹Ø¯Ø§Ø¯Ø§Øª Fold 5)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.loc[idx, 'filename']
        label = int(self.dataframe.loc[idx, 'label'])
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ØªØ¬Ù‡ÙŠØ² DataLoader Ø¹Ù„Ù‰ ÙƒØ§Ù…Ù„ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª
train_dataset = CustomDataset(df, root_dir=img_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ØªØ­Ù…ÙŠÙ„ Ù†Ù…ÙˆØ°Ø¬ ConvNeXt ÙˆØªØ¹Ø¯ÙŠÙ„Ù‡
model = models.convnext_tiny(pretrained=True)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Ø§Ù„ØªØ¯Ø±ÙŠØ¨
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_preds = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        correct_preds += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_preds.double() / len(train_dataset)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

# Ø­Ù�Ø¸ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬ Ø§Ù„Ù†Ù‡Ø§Ø¦ÙŠ
torch.save(model.state_dict(), "/kaggle/working/convnext_fold5_full.pt")
import shutil
shutil.copy("/kaggle/working/convnext_fold5_full.pt", "/kaggle/working/convnext_fold5_full_download.pt")



import os

print("Ù…Ù„Ù� Ø§Ù„Ù†Ù…ÙˆØ°Ø¬ Ù…ÙˆØ¬ÙˆØ¯ØŸ", os.path.exists("/kaggle/working/convnext_fold5_full.pt"))



import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# Ø¥Ø¹Ø¯Ø§Ø¯Ø§Øª
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dir = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test/"
model_path = "/kaggle/working/convnext_fold5_full.pt"

# Ù†Ù�Ø³ ØªØ­ÙˆÙŠÙ„Ø§Øª Ø§Ù„ØªØ¯Ø±ÙŠØ¨
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset Ù„Ù„ØªØ³Øª
class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name

# ØªØ­Ù…ÙŠÙ„ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª
test_dataset = TestDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ØªØ­Ù…ÙŠÙ„ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬
model = models.convnext_tiny(pretrained=False)
model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 7)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Ø§Ù„ØªÙ†Ø¨Ø¤
predictions = []
filenames = []

with torch.no_grad():
    for images, names in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        filenames.extend(names)

print("âœ… Ø§Ù„ØªÙ†Ø¨Ø¤Ø§Øª Ø¬Ø§Ù‡Ø²Ø©!")



import matplotlib.pyplot as plt
from PIL import Image
import os

# Ø¹Ø¯Ø¯ Ø§Ù„ØµÙˆØ± Ø§Ù„Ù„ÙŠ Ø±Ø§Ø­ Ù†Ø¹Ø±Ø¶Ù‡Ø§
num_images = 12
rows, cols = 3, 4

# ØªØ¹Ø±ÙŠÙ� Ù‚Ø§Ù…ÙˆØ³ ØªØ­ÙˆÙŠÙ„ Ø§Ù„Ø£Ø±Ù‚Ø§Ù… Ø¥Ù„Ù‰ Ø£Ø³Ù…Ø§Ø¡ Ø§Ù„Ø³Ù„Ø§Ù„Ø§Øª
idx_to_label = {
    0: 'barbari',
    1: 'goat',
    2: 'harri',
    3: 'naeimi',
    4: 'najdi',
    5: 'roman',
    6: 'sawakni'
}

# Ø¥Ù†Ø´Ø§Ø¡ Ø§Ù„Ø´ÙƒÙ„
plt.figure(figsize=(16, 10))

for i in range(num_images):
    img_path = os.path.join(test_dir, filenames[i])
    img = Image.open(img_path)

    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)

    # Ø¹Ø±Ø¶ Ø§Ø³Ù… Ø§Ù„Ø³Ù„Ø§Ù„Ø© Ø¨Ø¯Ù„Ø§Ù‹ Ù…Ù† Ø§Ù„Ø±Ù‚Ù…
    label_name = idx_to_label[predictions[i]]
    plt.title(f"Predicted: {label_name}")
    plt.axis('off')

plt.tight_layout()
plt.show()



import pandas as pd

# Ø¥Ù†Ø´Ø§Ø¡ Ù…Ù„Ù� Ø§Ù„ØªØ³Ù„ÙŠÙ… Ù…Ù† Ø§Ù„Ù†ØªØ§Ø¦Ø¬ Ø§Ù„Ù„ÙŠ Ø·Ù„Ø¹Ù†Ø§Ù‡Ø§
submission = pd.DataFrame({
    'filename': filenames,
    'label': predictions
})

submission.to_csv("submission.csv", index=False)
print("âœ… ØªÙ… Ø¥Ù†Ø´Ø§Ø¡ submission.csv Ø¨Ù†Ø¬Ø§Ø­!")


import os

print(os.listdir('.'))



# Ø£ÙˆÙ„Ù‹Ø§: ØªØ£ÙƒØ¯ÙŠ Ø¥Ù† Ø¹Ù†Ø¯Ùƒ Ø§Ù„Ù‚Ø§Ù…ÙˆØ³
idx_to_label = {
    0: 'barbari',
    1: 'goat',
    2: 'harri',
    3: 'naeimi',
    4: 'najdi',
    5: 'roman',
    6: 'sawakni'
}

# Ù†Ø­ÙˆÙ„ Ø§Ù„Ø£Ø±Ù‚Ø§Ù… Ø¥Ù„Ù‰ Ø£Ø³Ù…Ø§Ø¡ Ø§Ù„Ø³Ù„Ø§Ù„Ø§Øª
predicted_labels = [idx_to_label[p] for p in predictions]

# Ø«Ù… Ù†Ù†Ø´Ø¦ Ù…Ù„Ù� Ø§Ù„ØªØ³Ù„ÙŠÙ…
import pandas as pd

submission = pd.DataFrame({
    'filename': filenames,
    'label': predicted_labels
})

submission.to_csv("submission.csv", index=False)
print("âœ… ØªÙ… Ø­Ù�Ø¸ submission.csv Ø¨Ø£Ø³Ù…Ø§Ø¡ Ø§Ù„Ø³Ù„Ø§Ù„Ø§Øª Ø¨Ø¯Ù„ Ø§Ù„Ø£Ø±Ù‚Ø§Ù…!")



from IPython.display import FileLink
display(FileLink('submission.csv'))


