import numpy as np 
import pandas as pd 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# import tensorflow as tf
# import tensorflow_hub as hub
# from tensorflow.keras import layers, models


#no transfer learning
#limit model
#only black and white

#!pip install -q tensorflow==2.16 tensorflow-hub tensorflow-addons



class ImageCSVDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, label_map=None, is_test=False):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir  
        self.transform = transform
        self.label_map = label_map
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = self.df.iloc[idx]['path']  
        img_path = os.path.join(self.root_dir, rel_path)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            image_id = int(row['ID'])
            print(image_id)
            return image, image_id  # useful for mapping back to submission
        else:
            label = self.label_map.get(row.get("biome_type", -1), -1)
            return image, label if label != -1 else row['path']



train_df = pd.read_csv("/kaggle/input/acm-ai-hack/train.csv")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.Grayscale(num_output_channels=1)
])

class_names = sorted(train_df['biome_type'].unique())
print(f"Number of classes: {len(class_names)}")
label2idx = {label: idx for idx, label in enumerate(class_names)}

train_dataset = ImageCSVDataset(
    csv_file="/kaggle/input/acm-ai-hack/train.csv",
    root_dir="/kaggle/input/acm-ai-hack/train/",
    transform=transform,
    label_map=label2idx
)


train_dataset = ImageCSVDataset(
    csv_file="/kaggle/input/acm-ai-hack/train.csv",
    root_dir="/kaggle/input/acm-ai-hack/train/",
    transform=transform,
    label_map=label2idx
   
)


# import tensorflow as tf
# import tensorflow_hub as hub
# from tensorflow.keras import layers, models
# from pathlib import Path

# DATA_DIR   = Path("minecraft_images")          # /class_x/xxx.png
# IMG_SIZE   = (224, 224)
# BATCH_SIZE = 32

# train_ds = tf.keras.utils.image_dataset_from_directory(
#     DATA_DIR, validation_split=0.2, subset="training",
#     seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
#     label_mode="categorical")

# val_ds   = tf.keras.utils.image_dataset_from_directory(
#     DATA_DIR, validation_split=0.2, subset="validation",
#     seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
#     label_mode="categorical")

# NUM_CLASSES = train_ds.element_spec[1].shape[-1]

# # Prefetch for speed
# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
# val_ds   = val_ds.cache().prefetch(AUTOTUNE)

# train_ds = train_ds.map(lambda x, y: (tf.tile(x, [1,1,1,3]), y))
# val_ds   = val_ds.map(  lambda x, y: (tf.tile(x, [1,1,1,3]), y))




# data_aug = tf.keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.05),
#     layers.RandomZoom(0.1),
# ], name="augment")


import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.act   = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # match shapes for the skip-connection if depth or stride changes
        self.skip  = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.skip(x)                 # ① shortcut path

        out = self.act(self.bn1(self.conv1(x))) # ② main path
        out = self.bn2(self.conv2(out))

        out = self.act(out + identity)          # ③ combine & activate
        return out

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # 224×224 → 112×112
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2)
        )

        # Block-1: 112×112 → 56×56, channels 32 → 64
        self.block1 = ResidualBlock(32,  64, stride=2)
        # Block-2: 56×56  → 28×28, channels 64 → 128
        self.block2 = ResidualBlock(64, 128, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))   # 28×28 → 1×1
        self.dropout     = nn.Dropout(0.5) #delete this if needed
        self.fc          = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)             # (B, 32, 112, 112)
        x = self.block1(x)           # (B, 64, 56, 56)
        x = self.block2(x)           # (B, 128, 28, 28)

        x = self.global_pool(x)      # (B, 128, 1, 1)
        x = x.flatten(1)             # (B, 128)
        x = self.dropout(x) #delete this if needed
        return self.fc(x)



import torch
from torch.utils.data import random_split
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(label2idx)
model = SimpleCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


import torch
from torch.utils.data import random_split
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
def train_one_epoch(model, dataloader):
    model.train()
    total_loss = 0
    for batch in dataloader:
        imgs, labels = batch  
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            imgs, labels = batch  
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct/total


from tqdm.auto import tqdm
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
EPOCHS = 10
for epoch in tqdm(range(EPOCHS)):
    train_loss = train_one_epoch(model, train_loader)
    val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.2%}")

