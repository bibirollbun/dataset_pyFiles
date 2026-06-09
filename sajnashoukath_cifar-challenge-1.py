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


# -----------------------------
# Install Required Package
# -----------------------------
!pip install py7zr

# -----------------------------
# Imports
# -----------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import os
import py7zr



# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



class ResNeXtBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=4):
        super(ResNeXtBottleneck, self).__init__()
        D = int(math.floor(out_channels * (base_width / 64.0))) * cardinality
        self.conv1 = nn.Conv2d(in_channels, D, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D)
        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1,
                               groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(D)
        self.conv3 = nn.Conv2d(D, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, block, num_blocks, cardinality=32, num_classes=10):
        super(ResNeXt, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1, cardinality=cardinality)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2, cardinality=cardinality)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2, cardinality=cardinality)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride, cardinality):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, cardinality))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNeXt50_32x4d():
    return ResNeXt(ResNeXtBottleneck, [3, 4, 6], cardinality=32)

# Instantiate model
model = ResNeXt50_32x4d().to(device)
print("ResNeXt50 (32x4d) model instantiated successfully.")


# -----------------------------
# Data Augmentation
# -----------------------------

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w, dtype=img.dtype)
        y = torch.randint(h, (1,)).item()
        x = torch.randint(w, (1,)).item()
        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)
        mask[y1:y2, x1:x2] = 0
        img = img * mask.unsqueeze(0)
        return img


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    Cutout(length=16),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)


# -----------------------------
# Loss, Optimizer, Scheduler
# -----------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)

class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.001, T_up=5, gamma=0.5, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.gamma = gamma
        self.cycle = 0
        self.T_i = T_0
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_up:
            return [(self.base_lrs[i] + (self.eta_max - self.base_lrs[i]) * self.last_epoch / self.T_up)
                    for i in range(len(self.base_lrs))]
        else:
            cos_inner = (self.last_epoch - self.T_up) / (self.T_i - self.T_up)
            return [self.base_lrs[i] + 0.5 * (self.eta_max - self.base_lrs[i]) *
                    (1 + math.cos(math.pi * cos_inner))
                    for i in range(len(self.base_lrs))]

scheduler = CosineAnnealingWarmupRestarts(optimizer, T_0=60, eta_max=0.001, T_up=10)


# -----------------------------
# Training and Validation
# -----------------------------

def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, desc=f"Training Epoch {epoch+1}")):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print(f'[Epoch {epoch + 1}, Batch {batch_idx + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

def validate(epoch):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc=f"Validation Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(f'Validation Accuracy after Epoch {epoch + 1}: {acc:.2f}%')
    return acc

# -----------------------------
# Training Loop
# -----------------------------
best_acc = 0.0
num_epochs = 60
for epoch in range(num_epochs):
    train(epoch)
    acc = validate(epoch)
    scheduler.step()
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'resnext50_cifar10_best.pt')
        print(f"Best model saved with accuracy: {best_acc:.2f}%")

print("Initial Training completed.")



# -----------------------------
# Inference on Test Set
# -----------------------------

# Decompressing Test Data (Assuming you have a 7z file)
# Note: Adjust the paths based on your environment
test_archive_path = '/kaggle/input/cifar-10/test.7z'
extracted_path = '/kaggle/working/test'

if not os.path.exists(extracted_path):
    os.makedirs(extracted_path, exist_ok=True)
    with py7zr.SevenZipFile(test_archive_path, mode='r') as z:
        z.extractall(extracted_path)
    print("Test data decompressed successfully.")
else:
    print("Test data already decompressed.")


import os
import py7zr

test_archive = '/kaggle/input/cifar-10/test.7z'
extracted_path = '/kaggle/working/test'

if not os.path.exists(extracted_path):
    os.makedirs(extracted_path, exist_ok=True)

with py7zr.SevenZipFile(test_archive, mode='r') as archive:
    archive.extractall(path=extracted_path)



# List all files and folders in the extracted directory
for root, dirs, files in os.walk(extracted_path):
    print("Folder:", root)
    print("Files:", files[:5])  # Show first 5 files only
    print("----")



# -----------------------------
# Required Packages
# -----------------------------
!pip install py7zr

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
from tqdm import tqdm
import py7zr
import zipfile

# -----------------------------
# Device
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# -----------------------------
# ResNeXt Model Definition
# -----------------------------
class ResNeXtBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=4):
        super().__init__()
        D = int(out_channels * (base_width / 64.0)) * cardinality
        self.conv1 = nn.Conv2d(in_channels, D, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(D)
        self.conv2 = nn.Conv2d(D, D, 3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(D)
        self.conv3 = nn.Conv2d(D, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNeXt(nn.Module):
    def __init__(self, block, num_blocks, cardinality=32, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 128, num_blocks[0], 1, cardinality)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], 2, cardinality)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], 2, cardinality)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    def _make_layer(self, block, out_channels, num_blocks, stride, cardinality):
        layers = []
        strides = [stride] + [1]*(num_blocks-1)
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, cardinality))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.linear(out)

def ResNeXt50_32x4d():
    return ResNeXt(ResNeXtBottleneck, [3,4,6], cardinality=32)

# -----------------------------
# CIFAR-10 Classes
# -----------------------------
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# -----------------------------
# Custom Dataset
# -----------------------------
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
    def __len__(self):
        return len(self.image_filenames)
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        digits = ''.join(filter(str.isdigit, self.image_filenames[idx]))
        img_id = int(digits) if digits else idx
        return image, img_id

# -----------------------------
# Test Transforms
# -----------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
test_transforms = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

# -----------------------------
# Decompress Test Data
# -----------------------------
test_archive = '/kaggle/input/cifar-10/test.7z'
test_dir = '/kaggle/working/test/test'  # ✅ Correct folder with images

if not os.path.exists(test_dir):
    os.makedirs(test_dir, exist_ok=True)
    with py7zr.SevenZipFile(test_archive, mode='r') as archive:
        archive.extractall(test_dir)
    print("Test data decompressed ✅")
else:
    print("Test data already exists ✅")

# -----------------------------
# Load Model
# -----------------------------
model_path = '/kaggle/working/resnext50_cifar10_best.pt'
model = ResNeXt50_32x4d().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded ✅")

# -----------------------------
# Test DataLoader
# -----------------------------
test_dataset = CustomImageDataset(test_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# -----------------------------
# Inference
# -----------------------------
all_predictions = []
with torch.no_grad():
    for images, img_ids in tqdm(test_loader, desc="Inference"):
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs,1)
        preds = preds.cpu().numpy()
        for img_id, pred in zip(img_ids, preds):
            all_predictions.append((img_id, classes[pred]))

# -----------------------------
# Save CSV
# -----------------------------
submission_df = pd.DataFrame(all_predictions, columns=['id','label'])
submission_df = submission_df.sort_values('id').reset_index(drop=True)
submission_csv = '/kaggle/working/submission.csv'
submission_df.to_csv(submission_csv, index=False)
print("CSV saved ✅", submission_csv)

# -----------------------------
# Zip CSV for Kaggle Submission
# -----------------------------
zip_path = '/kaggle/working/submission.zip'
with zipfile.ZipFile(zip_path,'w') as zipf:
    zipf.write(submission_csv, arcname='submission.csv')
print("ZIP created ✅", zip_path)



import pandas as pd

submission_csv = '/kaggle/working/submission.csv'
df = pd.read_csv(submission_csv)

# Check basic info
print("Number of rows:", len(df))
print("Columns:", df.columns)
print("First 10 rows:\n", df.head(10))

# Check the unique labels predicted
print("Unique predicted labels:", df['label'].unique())



import pandas as pd

# Load your submission CSV
df = pd.read_csv('/kaggle/working/submission.csv')

# Convert 'tensor(1)' -> 1
df['id'] = df['id'].apply(lambda x: int(''.join(filter(str.isdigit, x))))

# Save cleaned CSV
submission_clean = '/kaggle/working/submission_clean.csv'
df.to_csv(submission_clean, index=False)

print("Cleaned submission CSV saved ✅", submission_clean)
print(df.head())





