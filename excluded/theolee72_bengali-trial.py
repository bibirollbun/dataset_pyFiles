# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!pip install albumentations
!pip install pretrainedmodels
!pip install iterative-stratification
!pip install pyarrow
import matplotlib.pyplot as plt
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import joblib
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import albumentations as A
import pretrainedmodels
import albumentations.pytorch

from torch.utils.data import Dataset
from sklearn.metrics import recall_score


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install torchvision


data_dir = "/kaggle/input/bengaliai-cv19/"
df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))


df_train.head()


os.makedirs('/kaggle/temp')


files_train = [f'train_image_data_{fid}.feather' for fid in range(4)]
for fname in files_train:
    F = os.path.join("/kaggle/input/bengaliaicv19feather", fname)
    df_save = pd.read_feather(F)
    img_ids = df_save['image_id'].values
    img_array = df_save.iloc[:, 1:].values
    for idx in tqdm(range(len(df_save))):
        img_id = img_ids[idx]
        img = img_array[idx]
        joblib.dump(img, f"/kaggle/temp/{img_id}.pkl")


plt.figure(figsize=(10, 20))
df_train["grapheme_root"].value_counts().sort_index().plot.barh()


df_train["vowel_diacritic"].value_counts().sort_index().plot.barh()


df_train["consonant_diacritic"].value_counts().sort_index().plot.barh()


df_train['id']=df_train["image_id"].apply(lambda x : int(x.split('_')[1]))


X = df_train[['id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values[:, 0]
y = df_train[['id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values[:, 1:]


mskf = MultilabelStratifiedKFold(n_splits=6, random_state=42, shuffle=True)


df_train["fold"] = -1


for i, (trn_idx, vld_idx) in enumerate(mskf.split(X, y)):
    print(i, trn_idx, vld_idx)
    


for i, (trn_idx, vld_idx) in enumerate(mskf.split(X, y)):
    df_train.loc[vld_idx, 'fold'] = i


df_train['fold'].value_counts()


df_train


trn_fold = [i for i in range(5) if i not in [5]]
vld_fold = [5]
trn_idx = df_train.loc[df_train['fold'].isin(trn_fold)].index
vld_idx = df_train.loc[df_train['fold'].isin(vld_fold)].index


class BengaliDataset(Dataset):
    def __init__(self, csv, img_height, img_width, transform):
        self.csv = csv.reset_index()
        self.img_ids = csv['image_id'].values
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform

    def __len__(self):
        return len(self.csv)
    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img = joblib.load(f'/kaggle/temp/{img_id}.pkl').reshape(self.img_height, self.img_width).astype(np.uint8)
        img = 255 - img
        
        img = img[:, :, np.newaxis]
        img = np.repeat(img, 3, 2)
        
        if self.transform is not None:
            img = self.transform(image=img)['image']

        label_1 = self.csv.iloc[index]['grapheme_root']
        label_2 = self.csv.iloc[index]['vowel_diacritic']
        label_3 = self.csv.iloc[index]['consonant_diacritic']

        return img, np.array([label_1, label_2, label_3])
        


train_augmentation = A.Compose([
    A.Rotate(20),
    A.Normalize(normalization="min_max"),
    A.pytorch.transforms.ToTensorV2()
])
valid_augmentation = A.Compose([
    A.Normalize(normalization="min_max"),
    A.pytorch.transforms.ToTensorV2()
])


from torch.utils.data import Dataset, DataLoader


trn_dataset = BengaliDataset(csv = df_train.loc[trn_idx], 
                            img_height = 137,
                            img_width = 236,
                            transform = train_augmentation)
vld_dataset = BengaliDataset(csv = df_train.loc[vld_idx], 
                            img_height = 137,
                            img_width = 236,
                            transform = valid_augmentation)


trn_dataset[0][0].max()


trn_loader = DataLoader(trn_dataset, 
                       shuffle=True,
                       num_workers=4,
                       batch_size=256)
vld_loader = DataLoader(vld_dataset, 
                       shuffle=False,
                       num_workers=4,
                       batch_size=256)


for inputs, target in trn_loader:
    break
inputs.shape


target.shape


import torchvision
from torchvision import models


model = models.resnet34()


model


# model.fc1 = nn.Linear(512, 168)  # For grapheme
# model.fc2 = nn.Linear(512, 11)   # vowel
# model.fc3 = nn.Linear(512, 7)    # consonant


model


n_grapheme=168
n_vowel=11
n_consonant=7
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, n_grapheme + n_vowel + n_consonant)
model


model = model.cuda()


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


loss_fn = nn.CrossEntropyLoss()
schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     mode='max',
                                                     verbose=True,
                                                     patience = 7,
                                                     factor=0.5)


from tqdm import tqdm_notebook


best_score = -1


for e in range(2):
    train_loss = []
    model.train()


    for inputs, targets in tqdm_notebook(trn_loader):
        
    
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        logits = model(inputs)
        logits_split = torch.split(logits, [n_grapheme, n_vowel, n_consonant], dim=1)
        
        loss = loss_fn(logits_split[0], targets[:, 0]) + loss_fn(logits_split[1], targets[:, 1]) + loss_fn(logits_split[2], targets[:, 2])
        
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(loss.item())
    
    val_loss = []
    val_true = []
    val_pred = []
    
    model.eval()
    
    with torch.no_grad():
        for inputs, targets in tqdm_notebook(vld_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
    
            logits = model(inputs)
            
            logits_split = torch.split(logits, [n_grapheme, n_vowel, n_consonant], dim=1)
        
            loss = loss_fn(logits_split[0], targets[:, 0]) + loss_fn(logits_split[1], targets[:, 1]) + loss_fn(logits_split[2], targets[:, 2])
    
            val_loss.append(loss.item())
    
            grapheme = logits_split[0].cpu().argmax(dim=1).data.numpy()
            vowel = logits_split[1].cpu().argmax(dim=1).data.numpy()
            cons = logits_split[2].cpu().argmax(dim=1).data.numpy()
    
            val_true.append(targets.cpu().numpy())
            val_pred.append(np.stack([grapheme, vowel, cons], axis=1))
    
    
    val_true = np.concatenate(val_true)
    val_pred = np.concatenate(val_pred)
    
    print(val_true.shape, val_pred.shape)
    
    val_loss = np.mean(val_loss)
    train_loss = np.mean(train_loss)
    
    print(val_loss, train_loss)
    
    score_g = recall_score(val_true[:, 0], val_pred[:, 0], average='macro')
    score_v = recall_score(val_true[:, 1], val_pred[:, 1], average='macro')
    score_c = recall_score(val_true[:, 2], val_pred[:, 2], average='macro')
    
    final_score = np.average([score_g, score_v, score_c], weights=[2, 1, 1])
    
    print(f'train_loss: {train_loss: .5f}; val_loss: {val_loss: .5f}; score: {final_score: .5f}')
    print(f'score_g: {score_g: .5f}; score_c: {score_v: .5f}; score: {score_c: .5f}')

    if final_score > best_score:
        best_score = final_score

        torch.save(model, "model.pth")




