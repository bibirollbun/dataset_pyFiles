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


from zipfile import ZipFile
import torch

with ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip", 'r') as zip:
    zip.extractall()

with ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip", 'r') as zip:
    zip.extractall()


device = "cuda" if torch.cuda.is_available() else "cpu"
device


from PIL import Image
import os, glob

train_folder_path = "/kaggle/working/train"
test_folder_path = "/kaggle/working/test"

train_files = [f for f in os.listdir(train_folder_path) if f.endswith('.jpg')]
test_files = [f for f in os.listdir(test_folder_path) if f.endswith('.jpg')]

train_images = []

for image_path in train_files:
    image = Image.open(os.path.join(train_folder_path, image_path))
    image = image.resize((100, 100))
    train_images.append(torch.tensor(np.array(image)).float().to(device).permute(2, 0, 1) / 255)

all_images_train = glob.glob(os.path.join(train_folder_path, "*.jpg"))
train_labels = [1.0 if "dog" in os.path.basename(p) else 0.0 for p in all_images_train]

test_images = []

for image_path in test_files:
    image = Image.open(os.path.join(test_folder_path, image_path))
    image = image.resize((100, 100))
    test_images.append(torch.tensor(np.array(image)).float().to(device).permute(2, 0, 1) / 255)


from sklearn.model_selection import train_test_split

X, y = train_images, torch.tensor(train_labels).long().to(device)

X_train, X_test, y_train, y_test = train_test_split(X, y)

print("Train class counts:", torch.bincount(y_train.long()))


import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision

class CatsAndDogs(Dataset):
    def __init__(self, x, y):
        self.imgs = x
        self.labels = y
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # transform = torchvision.transforms.Compose([
        #     torchvision.transforms.ToTensor()
        # ])
        img = self.imgs[idx]
        label = self.labels[idx]
        # label = torch.tensor(label).float()
        # img = transform(img)
        return img, label

training_data = CatsAndDogs(X_train, y_train)
test_data = CatsAndDogs(X_test, y_test)

batch_size = 16
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


cnt = 0

for idx, (inputs, labels) in enumerate(train_dataloader):
    if cnt < 1:
        print(labels.view(-1, 1) )
    cnt += 1




