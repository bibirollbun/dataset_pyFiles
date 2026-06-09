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
        os.path.join(dirname, filename)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install git+https://github.com/WildlifeDatasets/wildlife-datasets@develop
!pip install git+https://github.com/WildlifeDatasets/wildlife-tools


import os
import numpy as np
import pandas as pd
import timm
import torchvision.transforms as T
from wildlife_datasets.datasets import AnimalCLEF2025
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity


root = '/kaggle/input/animal-clef-2025'
transform_display = T.Compose([
    T.Resize([384, 384]),
    ])
transform = T.Compose([
    *transform_display.transforms,
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


dataset = AnimalCLEF2025(root, transform=transform_display, load_label=True)


# Let's look at one example from the dataset
dataset[0]


import matplotlib.pyplot as plt
import random

# Choose 6 random indices
indices = random.sample(range(len(dataset)), 6)

# Set up the plot
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for ax, idx in zip(axes.flatten(), indices):
    img, label = dataset[idx]
    ax.imshow(img)
    ax.set_title(label)
    ax.axis('off')

plt.tight_layout()
plt.show()


dataset.plot_grid()
dataset.metadata


dataset.metadata[['species']].value_counts()


idx = dataset.metadata['identity'].str.startswith('SeaTurtleID2022')
idx[idx.isnull()] = False
dataset.plot_grid(idx=idx);


# Loading the dataset
dataset = AnimalCLEF2025(root, transform=transform, load_label=True)
dataset_database = dataset.get_subset(dataset.metadata['split'] == 'database')


dataset_database[0]


new_database = dataset_database


updated_database = []

for item in new_database:
    new_item = list(item)
    if item[1].startswith('Lynx'):
        new_item[1] = 'Lynx'
    elif item[1].startswith('Sea'):
        new_item[1] = 'SeaTurtle'
    elif item[1].startswith('Sal'):
        new_item[1] = 'Salamander'
    updated_database.append(tuple(new_item))


len(updated_database)


updated_database[13000]


from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # Extract unique labels and create a mapping to integers
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(set(label for _, label in data)))}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tensor, label = self.data[idx]
        # Convert label string to integer
        label_idx = self.label_to_index[label]
        return tensor, label_idx



# Assuming updated_database is your list of (tensor, label) tuples
mydataset = CustomDataset(updated_database)


mydataset[0]


len(mydataset)


# Let's seperate the cross validation set aside

from torch.utils.data import random_split

train_size = int(0.8 * len(mydataset))
_size=len(mydataset) - train_size
train_data, _data = random_split(mydataset, [train_size, _size])

cv_size = int(0.5 * len(_data))  # 50% for CV
test_size = len(_data) - cv_size  # Remaining 50% for Test

cv_data, test_data = random_split(_data, [cv_size, test_size])

print(f"Train data size: {train_size}\n" + f"CV data size: {cv_size}\n" + f"Test data size: {test_size}\n")


import torch


from torch.nn import Sequential
from torch.nn import ZeroPad2d
from torch.nn import Flatten
from torch.nn import Conv2d
from torch.nn import AvgPool2d
from torch.nn import MaxPool2d
from torch.nn import AdaptiveMaxPool2d
from torch.nn import BatchNorm2d
import torch.nn.functional as F


import torch.nn.init as init


np.random.seed(1)
torch.manual_seed(0)


import torch.nn as nn


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, f, filters):
        super().__init__()
        F1, F2, F3 = filters

        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.conv2 = nn.Conv2d(F1, F2, kernel_size=f, stride=1, padding=f//2, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)

        self.conv3 = nn.Conv2d(F2, F3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(F3)

    def forward(self, X):
        X_shortcut = X

        X = self.conv1(X)
        X = self.bn1(X)
        X = F.relu(X, inplace=False)

        X = self.conv2(X)
        X = self.bn2(X)
        X = F.relu(X, inplace=False)

        X = self.conv3(X)
        X = self.bn3(X)

        X = X + X_shortcut
        X = F.relu(X, inplace=False)

        return X



def test_identity_block():
    X = torch.randn(2, 64, 32, 32)
    block = IdentityBlock(in_channels=64, f=3, filters=[64, 64, 64])
    out = block(X)
    assert out.shape == X.shape, f"Expected output shape {X.shape}, but got {out.shape}"
    print("✅ Test passed: output shape matches input shape.")

test_identity_block()


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, f, filters, s):
        super().__init__()
        
        F1, F2, F3 = filters

        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=1, stride=s, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.conv2 = nn.Conv2d(F1, F2, kernel_size=f, stride=1, padding=f//2, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)

        self.conv3 = nn.Conv2d(F2, F3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(F3)

        self.shortcut_conv = nn.Conv2d(in_channels, F3, kernel_size=1, stride=s, padding=0, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(F3)

    def forward(self, X):
        X_shortcut = X

        X = self.conv1(X)
        X = self.bn1(X)
        X = F.relu(X, inplace=False)

        X = self.conv2(X)
        X = self.bn2(X)
        X = F.relu(X, inplace=False)

        X = self.conv3(X)
        X = self.bn3(X)

        X_shortcut = self.shortcut_conv(X_shortcut)
        X_shortcut = self.shortcut_bn(X_shortcut)

        X = X + X_shortcut
        X = F.relu(X, inplace=False)

        return X



def test_conv_block():
    X = torch.randn(2, 64, 32, 32)
    block = ConvolutionalBlock(in_channels=64, f=3, filters=[64, 64, 64], s=2)
    out = block(X)
    assert out.shape == X.shape, f"Expected output shape {X.shape}, but got {out.shape}"
    print("✅ Test passed: output shape matches input shape.")

test_identity_block()


class ResNet50(nn.Module):
    
    def __init__(self, in_channels, classes, in_features):
        
        super().__init__()
        
        self.in_channels = in_channels
        # Define weight parameters and make initialisations for stage 1
        self.zero_pad1 = nn.ZeroPad2d(3)
        self.W1 = nn.Parameter(torch.randn(64, in_channels, 7, 7))
        self.bn1 = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        # stage - 2
        self.cnv_b2 = ConvolutionalBlock(in_channels=64, f=3, filters=[64, 64, 256], s=1)
        self.idn_b2 = IdentityBlock(in_channels=256, f=3, filters=[64, 64, 256])
        self.idn_b22 = IdentityBlock(in_channels=256, f=3, filters=[64, 64, 256])

        # stage - 3
        self.cnv_b3 = ConvolutionalBlock(in_channels=256, f=3, filters=[128, 128, 512], s=2)
        self.idn_b31 = IdentityBlock(in_channels=512, f=3, filters=[128, 128, 512])
        self.idn_b32 = IdentityBlock(in_channels=512, f=3, filters=[128, 128, 512])
        self.idn_b33 = IdentityBlock(in_channels=512, f=3, filters=[128, 128, 512])

        # stage - 4
        self.cnv_b4 = ConvolutionalBlock(in_channels=512, f=3, filters=[256, 256, 1024], s=2)
        self.idn_b41 = IdentityBlock(in_channels=1024, f=3, filters=[256, 256, 1024])
        self.idn_b42 = IdentityBlock(in_channels=1024, f=3, filters=[256, 256, 1024])
        self.idn_b43 = IdentityBlock(in_channels=1024, f=3, filters=[256, 256, 1024])
        self.idn_b44 = IdentityBlock(in_channels=1024, f=3, filters=[256, 256, 1024])
        self.idn_b45 = IdentityBlock(in_channels=1024, f=3, filters=[256, 256, 1024])

        # stage - 5
        self.cnv_b5 = ConvolutionalBlock(in_channels=1024, f=3, filters=[512, 512, 2048], s=2)
        self.idn_b51 = IdentityBlock(in_channels=2048, f=3, filters=[512, 512, 2048])
        self.idn_b52 = IdentityBlock(in_channels=2048, f=3, filters=[512, 512, 2048])

    def forward(self, X):

        # stage - 1
        X = self.zero_pad1(X)
        X = F.conv2d(X, self.W1, bias=None, stride=(2, 2), padding=0)
        X = self.bn1(X)
        X = F.relu(X, inplace=False)
        X = self.max_pool1(X)

        # stage - 2
        X = self.cnv_b2(X)
        X = self.idn_b2(X)
        X = self.idn_b22(X)

        # stage - 3
        X = self.cnv_b3(X)
        X = self.idn_b31(X)
        X = self.idn_b32(X)
        X = self.idn_b33(X)

        # stage - 4
        X = self.cnv_b4(X)
        X = self.idn_b41(X)
        X = self.idn_b42(X)
        X = self.idn_b43(X)
        X = self.idn_b44(X)
        X = self.idn_b45(X)

        # stage - 5
        X = self.cnv_b5(X)
        X = self.idn_b51(X)
        X = self.idn_b52(X)

        # AvgPool
        X = self.avg_pool(X)

        # output layer
        X = torch.flatten(X, start_dim=1)
        X = self.fc(X)
        
        return X


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def test_resnet50():

    # Instantiate model
    model = ResNet50(in_channels=3, classes=10, in_features=2048).to(device)
    
    # Input image tensor: batch size = 2, channels = 3 , height = 224, width = 224
    X = torch.randn(2, 3, 224, 224).to(device)

    # Forward pass
    out = model(X)

    # Expected output shape: [batch_size, num_classes]
    expected_shape = (2, 10)

    # Assertion
    assert out.shape == expected_shape, f"Expected output shape {expected_shape}, but got {out.shape}"

    print("✅ ResNet50 test passed: output shape is correct.")

# Run test
test_resnet50()


# Let's check the model summary 
!pip install torchsummary -q
from torchsummary import summary


example = ResNet50(in_channels=3, classes=3, in_features=2048).to(device)
summary(example, tuple(mydataset[0][0].shape)) # The shape of our input example is (3, 384, 384)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate your model
model = ResNet50(in_channels=3, classes=3, in_features=2048).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Number of epochs
num_epochs = 1

# Sample DataLoaders (replace these with your actual loaders)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(cv_data, batch_size=32, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    
    loop = tqdm(train_loader, leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Stats
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # TQDM live metrics
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item(), acc=100. * correct / total)
    
    # Epoch summary
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    print(f"\nEpoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Optionally add validation loop here too 👇
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = 100. * val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.2f}%\n")

print("✅ Training complete.")





