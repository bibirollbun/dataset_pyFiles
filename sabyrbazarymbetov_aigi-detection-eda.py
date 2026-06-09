import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from PIL import Image
import torchvision.models as models
from torchvision.transforms import v2
from torchvision import datasets


from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from transformers import pipeline 

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score


train = pd.read_csv('/kaggle/input/detect-ai-vs-human-generated-images/train.csv')
test = pd.read_csv('/kaggle/input/detect-ai-vs-human-generated-images/test.csv')

root = '/kaggle/input/ai-vs-human-generated-dataset/'


train.head()


train = train.drop(columns=['Unnamed: 0'])

train.head()


test.head()


# Set device
if torch.cuda.is_available():
    device='cuda'
elif torch.backends.xpu.is_available():
    device = 'xpu'
else:
    device = 'cpu'

device = torch.device(device)

print(f'Using device: {device}')


class ImageDataset(Dataset):
    def __init__(self, root, df, transform=None, is_train=True):
        self.root = root
        self.df = df
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        img_path = os.path.join(self.root, img_path)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        image = image.to(device)
        
        if self.is_train:
            label = self.df.iloc[idx, 1]            
            return image, label
        else:
            return image


def stats(dataloader):
    mean = torch.zeros(3).to(device)
    square = torch.zeros(3).to(device)
    num_pixels = 0
    
    for images, _ in tqdm(dataloader, desc='Mean & STD'):
        images = images.to(device)
        b, c, h, w = images.shape
        num_pixels += b * h * w
        mean += images.sum(dim=[0, 2, 3])  # Sum over batch, height, width
        square += (images ** 2).sum(dim=[0, 2, 3])  # Sum of x**2

    if num_pixels == 0:
        raise ValueError("Error: No pixels found! Check your dataloader.")

    mean /= num_pixels
    variance = (square / num_pixels) - (mean ** 2)
    variance = torch.clamp(variance, min=1e-6)  # Prevent negative values
    std = torch.sqrt(variance)

    mean = mean.detach().cpu()
    std = std.detach().cpu()

    print(f'Mean: {mean.tolist()}')
    print(f'STD: {std.tolist()}')

    return mean.tolist(), std.tolist()



class DivideBy255(object):
    def __call__(self, tensor):
        return tensor / 255.0
        
# Transform
transform = v2.Compose([
    v2.ToImage(),
    DivideBy255(), # Normalize
])


train_dataset = ImageDataset(root, train, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

stats(train_dataloader)


real = train[train['label'] == 0]

real_dataset = ImageDataset(root, real, transform=transform)
real_dataloader = DataLoader(real_dataset, batch_size=1, shuffle=False, num_workers=0)

stats(real_dataloader)


fake = train[train['label'] == 1]
fake_dataset = ImageDataset(root, fake, transform=transform)
fake_loader = DataLoader(fake_dataset, batch_size=1, shuffle=False, num_workers=0)


stats(fake_loader)

