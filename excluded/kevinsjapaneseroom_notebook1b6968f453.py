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


import pandas as pd
import os, glob, time, copy, zipfile
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import models, transforms


size = 224 #画像サイズ
mean = (0.485, 0.456, 0.406) #ImageNetデータセットの平均値
std = (0.229, 0.224, 0.225)  #ImageNetデータセットの標準偏差
batch_size = 32 #バッチサイズ
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epoch = 10 #学習回数


!rm -rf ./data


base_dir = '../input/dogs-vs-cats-redux-kernels-edition'
train_dir = './data/train'
test_dir = './data/test'

# 解凍先のフォルダを作成（すでにあってもOK）
os.makedirs('./data', exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)



# 学習データの解凍
with zipfile.ZipFile(os.path.join(base_dir, 'train.zip')) as train_zip:
    train_zip.extractall('./data')
    
#推論データの解凍
with zipfile.ZipFile(os.path.join(base_dir, 'test.zip')) as test_zip:
    test_zip.extractall('./data')


train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

# 学習データから検証データを分割(8:2)
train_list, val_list = train_test_split(train_list, test_size=0.2)
print("clear")




