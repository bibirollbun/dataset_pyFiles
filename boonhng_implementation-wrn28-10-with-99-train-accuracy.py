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


pip install py7zr


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import math
import py7zr
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


archive_path = '../input/cifar-10/train.7z'
extract_folder = '/kaggle/temp'
with py7zr.SevenZipFile(archive_path, mode='r') as archive:
    archive.extractall(path=extract_folder)

archive_path = '../input/cifar-10/test.7z'
with py7zr.SevenZipFile(archive_path, mode='r') as archive:
    archive.extractall(path=extract_folder)


# Data augmentation and normalization
class CIFAR10CustomDataset(Dataset):
    def __init__(self, img_dir, labels_df=None, transform=None):
        self.img_dir = img_dir
        self.labels_df = labels_df
        self.transform = transform
        self.ids = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = Image.open(os.path.join(self.img_dir, img_id)).convert('RGB')
        if self.transform:
            img = self.transform(img)

        if self.labels_df is not None:
            label = int(self.labels_df.loc[self.labels_df['id'] == int(img_id.split('.')[0]), 'label_idx'].values[0])
            return img, label
        else:
            return img, int(img_id.split('.')[0])


class CutOut(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = random.randint(0, h)
            x = random.randint(0, w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask).expand_as(img)
        img = img * mask
        return img

# Transformation function using PyTorch's transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),  
    transforms.RandomCrop(32, padding=4),  
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    CutOut(n_holes=1, length=8), 
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
])
train_csv = pd.read_csv('/kaggle/input/cifar-10/trainLabels.csv')
label_names = sorted(train_csv['label'].unique())
label_to_index = {label: idx for idx, label in enumerate(label_names)}
train_csv['label_idx'] = train_csv['label'].map(label_to_index) # qua nhieu buggg ~~
train_dataset = CIFAR10CustomDataset(img_dir='/kaggle/temp/train',labels_df=train_csv, transform=train_transform)
test_dataset = CIFAR10CustomDataset(img_dir='/kaggle/temp/test', transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out += identity
        out = self.relu2(out)
        return out

class WRN(nn.Module):
    def __init__(self, depth, wide, num_classes=10, dropout_rate=0.0):
        super(WRN, self).__init__()
        self.in_channels = 16
        n = (depth - 4) // 6
        k = wide
        temp = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = self.wide_block(Block, temp[1], n, stride=1, dropout_rate=dropout_rate)
        self.block2 = self.wide_block(Block, temp[2], n, stride=2, dropout_rate=dropout_rate)
        self.block3 = self.wide_block(Block, temp[3], n, stride=2, dropout_rate=dropout_rate)
        self.bn = nn.BatchNorm2d(temp[3])
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(temp[3], num_classes)

    def wide_block(self, block, outputs, num_blocks, stride, dropout_rate=0.0):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, outputs, stride, dropout_rate))
            self.in_channels = outputs
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def create_wrn(depth=28, wide=10, num_classes=10, dropout_rate=0.3):
    return WRN(depth, wide, num_classes, dropout_rate)


model = create_wrn().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=80,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=1e4,  # giảm dần mạnh ở cuối
)
def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print(f"Epoch {epoch} | Loss: {running_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.2f}%")


def test(model, test_loader):
    model.eval()
    result = []
    with torch.no_grad():
        for inputs, ids in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs,1)
            predicted_names = [label_names[idx] for idx in predicted.cpu().numpy()]
            result += list(zip(ids,predicted_names))
    submission_df = pd.DataFrame(result, columns=['id', 'label'])
    submission_df['id'] = submission_df['id'].astype(int)  
    submission_df = submission_df.sort_values(by='id')  
    submission_df.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    num_epochs = 80
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train(epoch)
    test(model, test_loader)
    print("Predictions saved to submission.csv")

