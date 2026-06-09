# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.v2 as transforms
import torchvision.models as models


# Configuration class
class Config:
    SEED = 42
    VAL_SPLIT = 0.2
    IMAGE_SIZE = (64, 64)
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 5

config = Config()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(config.SEED)



# Dataset class
class CassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = plt.imread(row['img_path'])
        label = row['label']

        if self.transform:
            image = self.transform(image)

        return image, label


# Define transforms
train_transforms = transforms.Compose([
    transforms.Resize(config.IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize(config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Data loading
input_path = '/kaggle/input/cassava-leaf-disease-classification/'
train_df = pd.read_csv(os.path.join(input_path, 'train.csv'))
train_df['img_path'] = train_df['image_id'].apply(lambda x: os.path.join(input_path, 'train_images', x))

train_data, val_data = train_test_split(train_df, test_size=config.VAL_SPLIT, stratify=train_df['label'], random_state=config.SEED)

train_dataset = CassavaDataset(train_data, transform=train_transforms)
val_dataset = CassavaDataset(val_data, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)


# Model definition
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
model = model.cuda() if torch.cuda.is_available() else model

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
scheduler = ExponentialLR(optimizer, gamma=0.9)

# Mixed precision setup
scaler = torch.cuda.amp.GradScaler()


# Training function
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


# Validation function
def validate_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


# Training loop
for epoch in range(config.EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)
    scheduler.step()

    print(f"Epoch {epoch+1}/{config.EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


# Save the model
torch.save(model.state_dict(), 'cassava_model.pth')

