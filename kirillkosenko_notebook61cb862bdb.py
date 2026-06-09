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


!unzip -q "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
!unzip -q "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"


from sklearn.model_selection import train_test_split

def prepare_data(train_path, val_size=0.2, random_state=42):
    train_filenames = os.listdir(train_path)
    train_categories = ['dog' if filename.split(".")[0] == 'dog' else 'cat' for filename in train_filenames]

    df = pd.DataFrame({
        'filename': train_filenames,
        'category': train_categories
    })

    train_df, val_df = train_test_split(df, test_size=val_size, stratify=df["category"], random_state=random_state)
    return train_df, val_df


train_path = "/kaggle/working/train"
train_df, val_df = prepare_data(train_path)
print(f"Total Training Images: {len(train_df)}")
print(f"Total Validation Images: {len(val_df)}")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class TrainDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'filename']
        label_str = self.df.loc[idx, 'category']
        label = 1 if label_str == 'dog' else 0

        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class TestDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.images = os.listdir(folder)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.folder, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        img_id = int(img_name.split('.')[0])
        return image, img_id


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 64 * 64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),
])

train_dataset = TrainDataset(train_df, train_path, transform)
val_dataset = TrainDataset(val_df, train_path, transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)


    print(f"Epoch [{epoch+1}/5] | Loss: {total_loss/len(train_loader)} | Val Acc: {100*correct/total}%")


test_path = "/kaggle/working/test"
test_dataset = TestDataset(test_path, transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()
predictions = []

with torch.no_grad():
    for imgs, img_ids in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        for img_id, pred in zip(img_ids, preds.cpu().numpy()):
            predictions.append([int(img_id.item()), pred])

submission = pd.DataFrame(predictions, columns=["id", "label"])
submission = submission.sort_values("id")  
submission.to_csv("/kaggle/working/submission1.csv", index=False)

print(submission.head())


test_path = "/kaggle/working/test"
test_dataset = TestDataset(test_path, transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()
predictions = []

with torch.no_grad():
    for imgs, img_ids in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)[:, 1] 
        for img_id, prob in zip(img_ids, probs.cpu().numpy()):
            predictions.append([int(img_id.item()), float(prob)])

submission = pd.DataFrame(predictions, columns=["id", "label"])
submission = submission.sort_values("id")
submission.to_csv("/kaggle/working/submission.csv", index=False)

print(submission.head())

