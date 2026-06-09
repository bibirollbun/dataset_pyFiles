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


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader


train_ds = pd.read_csv("/kaggle/input/try-mnist/train.csv")
train_ds


labels = train_ds['label'].values                
pixels = train_ds.drop('label', axis=1).values  
images = pixels.astype(np.float32) / 255.0
images = images.reshape(-1, 1, 28, 28)  


from torch.utils.data import Dataset
class DigitCsvDataset(Dataset):
    def __init__(self, images_tensor, labels_tensor, transform=None):
        self.images = images_tensor
        self.labels = labels_tensor
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]      
        lbl = self.labels[idx]    
        if self.transform:
            img = self.transform(img)
        return img, lbl



from torch.utils.data import random_split
n = len(images)
n_train = int(0.8 * n)
n_val   = n - n_train
train_imgs, val_imgs = torch.utils.data.random_split(
    DigitCsvDataset(images, labels),
    [n_train, n_val]
)

train_loader = DataLoader(train_imgs, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_imgs,   batch_size=64, shuffle=False)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                      
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch:2d} | Train Loss: {epoch_loss:.4f}")



model.eval()
val_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for imgs, lbls in val_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        outputs = model(imgs)
        loss    = criterion(outputs, lbls)
        val_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == lbls).sum().item()
        total   += lbls.size(0)
avg_loss = val_loss / len(val_loader.dataset)
accuracy = correct / total
print(avg_loss)
print(accuracy)


test_ds = pd.read_csv("/kaggle/input/try-mnist/test.csv")
test_ds


from torch.utils.data import TensorDataset
test_pixels = test_ds.values
test_images = test_pixels.astype(np.float32) / 255.0
test_images = test_images.reshape(-1, 1, 28, 28) 
test_tensor = torch.from_numpy(test_images)
test_dataset = TensorDataset(test_tensor)         
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)


model.eval()
all_preds = []

with torch.no_grad():
    for (imgs,) in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())


len(all_preds)


submission = pd.DataFrame({"ImageId": np.arange(1, len(all_preds) + 1),"Label": all_preds})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")




