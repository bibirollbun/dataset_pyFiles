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


from sklearn.model_selection import train_test_split
from zipfile import ZipFile
import torch

sample_submission = pd.read_csv("/kaggle/input/fashion-mnist-itba/submission_sample.csv")

with ZipFile("/kaggle/input/fashion-mnist-itba/fashion-mnist-itba-lab-ml-2018b.zip", 'r') as zip:
    zip.extractall()


np.sort(pd.read_csv("/kaggle/working/train_labels.csv")["label"].unique())


X = np.load("/kaggle/working/train_images.npy")
y = np.array(pd.read_csv("/kaggle/working/train_labels.csv"))


X_train, X_test, y_train, y_test = train_test_split(X, y)


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class FMNIST(Dataset):
    def __init__(self, x, y):
        self.imgs = x
        self.labels = y

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = torch.tensor(self.imgs[idx]).float()
        image = image.unsqueeze(0)
        label = torch.tensor(self.labels[idx]).long()
        label = label.unsqueeze(0)
        
        return image, label

batch_size = 32

train_dataset = FMNIST(X_train, y_train)
test_dataset = FMNIST(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 1),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 6, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 1),
            nn.BatchNorm2d(6),
            nn.Conv2d(6, 12, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 1),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7500, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = CNN().to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

size = len(train_dataloader.dataset)

epochs = 10

model.train()

for epoch in range(epochs):
    total = 0
    correct = 0
    print(f"Epoch: {epoch}\n_______________")
    for batch, (X, y) in enumerate(train_dataloader):
        output = model(X)
        y = y.view(-1)
        loss = loss_fn(output, y)
        pred = torch.argmax(output, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    print(f"Accuracy: {(correct / total) * 100}%")


model.eval()

test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for X, y in test_dataloader:
        X = X.to(device)
        y = y.view(-1).long().to(device)  # ensure shape [B] and correct type

        output = model(X)
        test_loss += loss_fn(output, y).item()

        preds = torch.argmax(output, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

test_loss /= len(test_dataloader)
accuracy = (correct / total) * 100
print(f"Test Error: \n Accuracy: {accuracy:.1f}%, Avg loss: {test_loss:.6f}")


X_sub = torch.tensor(np.load("/kaggle/working/test_images.npy")).float().unsqueeze(0).unsqueeze(0).permute(1, 2, 0, 3, 4)

predictions = []

for img in X_sub:
    output = model(img)
    ans = torch.argmax(output, dim=1)
    
    predictions += ans


preds = []
for p in predictions:
    preds.append(int(p))


preds[:10]


sample_submission


sample_submission = sample_submission.drop(["label"], axis=1)
sample_submission


sample_submission["Category"] = preds

sample_submission.to_csv("submission.csv", index=False)




