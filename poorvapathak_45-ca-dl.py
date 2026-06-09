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


# imports
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from tqdm import tqdm
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger


# function to get just the non-hidden files in a director
def ListDirectory(dir):
    files = []
    for item in os.listdir(dir):
        # if the item doesnt start with . then it non-hidden
        if not item.startswith('.'):
            files.append(item)
    return (files)


# EDA on the Dataset
# image standardization
def StandardizeImage(input_tensor):
    mean = input_tensor.mean(axis = (1,2))
    std = input_tensor.std(axis = (1,2))
    transformation = transforms.Normalize(mean, std)
    return transformation(input_tensor)


def ReadAndProcessImages(location, dim_reducer):
    directories = ListDirectory(location)
    label_to_int = {label:idx for idx, label in enumerate(directories)}
    img_tensors = [] 
    ys = [] 
    for directory in tqdm(directories):
        for file_name in ListDirectory(f'{location}/{directory}'):            
            img = Image.open(f'{location}/{directory}/{file_name}')
            img_np = np.array(img)
            img_tensor = torch.tensor(img_np, dtype = torch.float).permute(2,0,1)
            standardized_img_tensor = StandardizeImage(img_tensor)
            final_img_tensor = dim_reducer(standardized_img_tensor)
            img_tensors.append(final_img_tensor)
            ys.append(label_to_int[directory])

   # concatinate all img tensors
    img_tensors_stacked = torch.stack(img_tensors, dim = 0)
    return img_tensors_stacked, ys


# setting up data
dim_reducer = nn.AdaptiveAvgPool2d(output_size=(20,50))

train_img_tensors, train_ys = ReadAndProcessImages(location=f'{wd}/train', dim_reducer=dim_reducer)
val_img_tensors, val_ys = ReadAndProcessImages(location=f'{wd}/val', dim_reducer=dim_reducer)

train_data = []
for i in range(train_img_tensors.shape[0]):
    train_data.append([train_img_tensors[i], train_ys[i]])
val_data = []
for i in range(val_img_tensors.shape[0]):
    val_data.append([val_img_tensors[i], val_ys[i]])

class InstrumentsDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

train_dataset = InstrumentsDataset(train_data)
train_loader = DataLoader(dataset = train_dataset, batch_size = 25, shuffle = True)
val_dataset = InstrumentsDataset(val_data)
val_loader = DataLoader(dataset = val_dataset, batch_size = 25, shuffle = True)


# model experimenting to understand parameters

class InstrumentsModule(pl.LightningModule):
    def __init__(self, model_object, loss_module):
        super().__init__()
        self.model = model_object
        self.loss_module = loss_module

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params = self.model.parameters(), lr = 0.01)
        return optimizer

    def training_step(self, batch, batch_index):
        X, y = batch
        preds = self(X)
        loss = self.loss_module(preds, y)
        acc = torch.sum(preds.argmax(dim = 1) == y)/len(y)
        self.log('training_accuracy', acc)
        self.log('training_loss', loss)
        return loss

    def validation_step(self, batch, batch_index):
        X, y = batch
        preds = self(X)
        acc = torch.sum(preds.argmax(dim = 1) == y)/len(y)
        self.log('validation_accuracy', acc)

    def test_step(self, batch, batch_index):
        X, y = batch
        preds = self(X)
        acc = torch.sum(preds.argmax(dim = 1) == y)/len(y)
        self.log('test_accuracy', acc)

# basic model architecture
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels = 10, kernel_size=3, padding = 1), # a first convolution to scale up the channel size
            nn.BatchNorm2d(num_features=10),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels = 10, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features=10),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(10, 6)
            )

    def forward(self, x):
        return (self.model(x))


# make predictions

test_img_tensors, test_ys = ReadAndProcessImages(location=f'{wd}/test', dim_reducer=dim_reducer) 

test_data = []
for i in range(test_img_tensors.shape[0]):
    test_data.append([test_img_tensors[i], test_ys[i]])

test_dataset = InstrumentsDataset(test_data)
test_loader = DataLoader(dataset = test_dataset)

trainer.test(instruments_module, dataloaders=test_loader)

