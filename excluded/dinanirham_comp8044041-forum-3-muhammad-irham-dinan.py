import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms, models

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import copy
import os
import requests
import random
import zipfile
import shutil


TRAIN_PATH = '/kaggle/working/train'
TEST_PATH  = '/kaggle/working/test1'
WORK_DIR  = '/kaggle/working/dogs-vs-cats'

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
NUM_WORKERS = 2
MOMENTUM = 0.9
STEP_SIZE = 7
GAMMA = 0.1
RANDOM_STATE = 25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

!unzip -q "/kaggle/input/dogs-vs-cats/train.zip"
!unzip -q "/kaggle/input/dogs-vs-cats/test1.zip"


train_files = os.listdir(TRAIN_PATH)

cats = [f for f in train_files if 'cat' in f]
dogs = [f for f in train_files if 'dog' in f]

print(f"Total files of train set: {len(train_files)}")

print(f"Cats: {len(cats)}")
print(f"Dogs: {len(dogs)}")


def show_images(image_list, n=10):
    fig, axes = plt.subplots(2, 5, figsize=(16, 10))
    axes = axes.ravel()
    
    for i, img_name in enumerate(random.sample(image_list, n)):
        img_path = os.path.join(TRAIN_PATH, img_name)
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].set_title(img_name.split('.')[0].upper())
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

show_images(train_files)


class DogsVsCatsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        label = 0 if 'cat' in img_name else 1
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

all_files = os.listdir(TRAIN_PATH)
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=RANDOM_STATE)

os.makedirs('data/train', exist_ok=True)
os.makedirs('data/val', exist_ok=True)


for f in tqdm(train_files, desc='Copying train files'):
    shutil.copy(os.path.join(TRAIN_PATH, f), os.path.join('data/train', f))

for f in tqdm(val_files, desc='Copying val files'):
    shutil.copy(os.path.join(TRAIN_PATH, f), os.path.join('data/val', f))

print(f"Train set samples: {len(train_files)}")
print(f"Val set samples: {len(val_files)}")



train_dataset = DogsVsCatsDataset('data/train', transform=train_transform)
val_dataset = DogsVsCatsDataset('data/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"Train set batches: {len(train_loader)}")
print(f"Valid set batches: {len(val_loader)}")


class DogCatClassifier(nn.Module):
    def __init__(self, num_classes=2, fine_tune_last_block=True):
        super(DogCatClassifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        if fine_tune_last_block:
            for param in self.model.layer4.parameters():
                param.requires_grad = True
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

model = DogCatClassifier(num_classes=2).to(DEVICE)
print(model)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

best_val_acc = 0.0
device = DEVICE

def train_epoch(model, loader, criterion, optimizer, device=DEVICE):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/len(loader), 'acc': 100.*correct/total})
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device=DEVICE):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/len(loader), 'acc': 100.*correct/total})
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

for epoch in range(EPOCHS):
    print(f'\nEpoch {epoch+1}/{EPOCHS}')
    print('-' * 50)
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    scheduler.step(val_loss)
    
    print(f'\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, 'best_model.pth')
        print(f'\nSaved best model with accuracy: {val_acc:.2f}%')

print('\n' + '='*50)
print(f'Training completed! Best validation accuracy: {best_val_acc:.2f}%')
print('='*50)




