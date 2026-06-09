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


import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
from torch import nn, optim
from PIL import Image
import pandas as pd
from tqdm import tqdm

# --- 基本設定 ---
base = './data'
raw_train = os.path.join(base, 'train')
raw_test = os.path.join(base, 'test')

# ① フォルダ構造の整備 (ImageFolder 用)
for split in ['train', 'val']:
    for cls in ['cats', 'dogs']:
        os.makedirs(os.path.join(base, split, cls), exist_ok=True)
for split in ['train', 'val']:
    split_dir = os.path.join(base, split)
    count = len(os.listdir(os.path.join(base, 'train', 'cats'))) + len(os.listdir(os.path.join(base, 'train', 'dogs')))
    # move images if original raw_train contains jpg files
files = os.listdir(raw_train)
cats = [f for f in files if f.startswith('cat.')]
dogs = [f for f in files if f.startswith('dog.')]

split_idx = int(0.2 * len(cats))
for f in cats[:split_idx]:
    shutil.move(os.path.join(raw_train, f), os.path.join(base, 'val/cats', f))
for f in cats[split_idx:]:
    shutil.move(os.path.join(raw_train, f), os.path.join(base, 'train/cats', f))
for f in dogs[:split_idx]:
    shutil.move(os.path.join(raw_train, f), os.path.join(base, 'val/dogs', f))
for f in dogs[split_idx:]:
    shutil.move(os.path.join(raw_train, f), os.path.join(base, 'train/dogs', f))

# テストデータ用フォルダ準備
test_unknown = os.path.join(raw_test, 'unknown')
os.makedirs(test_unknown, exist_ok=True)
for f in os.listdir(raw_test):
    if f.endswith('.jpg'):
        shutil.move(os.path.join(raw_test, f), os.path.join(test_unknown, f))

# --- データローダ設定 ---
train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder(os.path.join(base, 'train'), transform=train_tf)
val_ds = datasets.ImageFolder(os.path.join(base, 'val'), transform=val_tf)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

test_ds = datasets.ImageFolder(raw_test, transform=val_tf)  # 'unknown' フォルダあり
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

print("Loaded datasets:", len(train_ds), len(val_ds), len(test_ds))


