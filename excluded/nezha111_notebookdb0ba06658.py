import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, InterpolationMode, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from torchsummary import summary
from PIL import Image

import os, copy, zipfile, time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


baseDir = '/kaggle/working/'
inDir = '/kaggle/input/street-view-getting-started-with-julia/'


def dataExtraction(extractionDir, filePath):
    # Create the target directory if it does not exist
    if not os.path.exists(extractionDir):
        os.makedirs(extractionDir)

    try:
        with zipfile.ZipFile(filePath, "r") as zfile:
            zfile.extractall(extractionDir)
        print(f"Extraction complete. Files have been extracted to: {extractionDir}")

    except zipfile.BadZipFile:
        print(f"Error: '{filePath}' is not a valid ZIP file or may be corrupted.")

    except FileNotFoundError:
        print(f"Error: The file '{filePath}' does not exist.")


dataExtraction(baseDir, inDir + 'train.zip')
dataExtraction(baseDir, inDir + 'test.zip')


# 标签转换
labels = pd.read_csv(inDir+'trainLabels.csv')
unq = sorted(labels['Class'].unique())
toLabel = {char: i for i, char in enumerate(unq)}
labels['Class'] = labels['Class'].map(toLabel)
labels = labels.values.tolist()


m_samples = len(labels)
transforms = v2.Compose([
    v2.Resize((32, 32), antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])
images = []
for idx in range(1, m_samples + 1):
    image = Image.open(os.path.join(baseDir,"train", str(idx) + ".Bmp")).convert("RGB")
    img_tensor = transforms(image)
    min_val = torch.min(img_tensor)
    max_val = torch.max(img_tensor)
    img_tensor = (img_tensor - min_val) / (max_val - min_val)
    images.append(img_tensor)


class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx][1]
        return image, label


train_dat = ImageDataset(images, labels)
train_dataloader = DataLoader(train_dat, batch_size=128, shuffle=True, drop_last=True)

for X, y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


net = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(576, 120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(84, 62),
)
summary(net, (3, 32, 32))


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)


epochs = 100
for epoch in range(epochs):
    train_loss = 0.0
    correct_train = 0.0
    total_size = 0

    train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
    for batch, (X, y) in enumerate(train_bar):
        pred = net(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            # 计算当前 batch 的指标
            batch_loss = loss.item()
            batch_correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            batch_size = y.size(0)
            # 更新累积指标
            train_loss += batch_loss * batch_size
            correct_train += batch_correct
            total_size += batch_size
            
            # avg_loss = train_loss / total_size
            # scheduler.step(avg_loss)
            # 实时更新进度条描述（显示当前 batch 的 loss 和 acc）
            if batch % 20 == 0:
                train_bar.set_postfix(
                    {
                        "loss": f"{train_loss/total_size:.4f}",
                        "acc": f"{correct_train/total_size:.4f}",
                    }
                )


class ImageDatasetTest(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return image


test_images = []
for idx in range(6284, 12504):
    image = Image.open(os.path.join("test", str(idx) + ".Bmp")).convert("RGB")
    img_tensor = transforms(image)
    min_val = torch.min(img_tensor)
    max_val = torch.max(img_tensor)
    img_tensor = (img_tensor - min_val) / (max_val - min_val)
    test_images.append(img_tensor)
    
test_dat = ImageDatasetTest(test_images)
test_dataloader = DataLoader(test_dat, batch_size=18)


net.eval()
pred = []
with torch.no_grad():
    for X in test_dataloader:
        a = net(X).argmax(1)
        pred.append(a)

pred = torch.cat(pred)
pred1 = [unq[val] for val in pred.tolist()]
df = pd.DataFrame(pred1, columns=['Class'])
df.index = np.arange(6284, len(df) + 6284)
df.index.name = 'ID'
df.head()


df.to_csv(baseDir + 'submission.csv', index = True)

