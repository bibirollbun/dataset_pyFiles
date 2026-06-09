import pandas as pd

data_path = '/kaggle/input/aerial-cactus-identification/'

labels = pd.read_csv(data_path + 'train.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')


import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

mpl.rc('font', size=15)
plt.figure(figsize=(7,7))
label = ['Has Cactus', 'No cactus']
plt.pie(labels['has_cactus'].value_counts(), labels=label, autopct='%.1f%%')


from zipfile import ZipFile

with ZipFile(data_path + 'train.zip') as zipper:
    zipper.extractall()

with ZipFile(data_path + 'test.zip') as zipper:
    zipper.extractall()


import os

num_train = len(os.listdir('train/'))
num_test = len(os.listdir('test/'))

print(f'훈련 데이터 개수: {num_train}')
print(f'테스트 데이터 개수: {num_test}')


import matplotlib.gridspec as gridspec
import cv2

mpl.rc('font', size=7)
plt.figure(figsize=(15,6))
grid = gridspec.GridSpec(2,6)

last_has_cactus_img_name = labels[labels['has_cactus']==1]['id'][-12:]

for idx, img_name in enumerate(last_has_cactus_img_name):
    img_path = 'train/' + img_name
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax = plt.subplot(grid[idx])
    ax.imshow(image)


image.shape


# GPU 할당하기
import torch
import random
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from sklearn.model_selection import train_test_split

train, valid = train_test_split(labels, test_size=0.1, stratify=labels['has_cactus'], random_state=42)


import cv2 # OpenCV Library
from torch.utils.data import Dataset
import torch

class ImageDataset(Dataset):
  #초기화 매서드(생성자)
  def __init__(self, df, img_dir='./', transform=None):
    super().__init__()
    self.df = df
    self.img_dir = img_dir
    self.transform = transform

  #데이터셋 크기 반환
  def __len__(self):
    return len(self.df)
  
  #idx 해당 데이터 반환 메서드
  def __getitem__(self, idx):
    img_id = self.df.iloc[idx, 0]
    img_path = self.img_dir + img_id
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = self.df.iloc[idx, 1]

    if self.transform is not None:
      image = self.transform(image)
    return image, label


from torchvision import transforms

transform_train = transforms.Compose([transforms.ToTensor(),
                                      transforms.Pad(32, padding_mode='symmetric'),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                      ])

transform_test = transforms.Compose([transforms.ToTensor(),
                                    transforms.Pad(32, padding_mode='symmetric'),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])





dataset_train = ImageDataset(df=train, img_dir='train/', transform=transform_train)
dataset_valid = ImageDataset(df=valid, img_dir='train/', transform=transform_test)


from torch.utils.data import DataLoader

loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
loader_valid = DataLoader(dataset_valid, batch_size=32, shuffle=False)


# Model생성

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self):
    super().__init__()

    self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2),
                               nn.BatchNorm2d(32),
                               nn.LeakyReLU(),
                               nn.MaxPool2d(kernel_size=2))

    self.layer2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2),
                               nn.BatchNorm2d(64),
                               nn.LeakyReLU(),
                               nn.MaxPool2d(kernel_size=2))

    self.layer3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2),
                               nn.BatchNorm2d(128),
                               nn.LeakyReLU(),
                               nn.MaxPool2d(kernel_size=2))

    self.layer4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2),
                               nn.BatchNorm2d(256),
                               nn.LeakyReLU(),
                               nn.MaxPool2d(kernel_size=2))

    self.layer5 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=2),
                               nn.BatchNorm2d(512),
                               nn.LeakyReLU(),
                               nn.MaxPool2d(kernel_size=2))

    self.avg_pool = nn.AvgPool2d(kernel_size=4)

    self.fc1 = nn.Linear(in_features=512*1*1, out_features=64)
    self.fc2 = nn.Linear(in_features=64, out_features=2)



  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.avg_pool(x)
    x = x.view(-1, 512*1*1) #평탄화
    x = self.fc1(x)
    x = self.fc2(x)
    return x

model = Model().to(device)

#손실함수 정의
criterion = nn.CrossEntropyLoss()
#옵티마이저
optimizer = torch.optim.Adamax(model.parameters(), lr=0.00006)

epochs = 70
for epoch in range(epochs):
  epoch_loss = 0

  for images, labels in loader_train:
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    epoch_loss += loss.item() #역전파
    loss.backward()
    optimizer.step() #가중치갱신 
  
  print(f'Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(loader_train):.4f}')


from sklearn.metrics import roc_auc_score

true_list = []
preds_list = []

model.eval()

with torch.no_grad():
  for images, labels in loader_valid:
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    preds = torch.softmax(outputs.cpu(), dim=1)[:, 1]# 사이킷런 함수는 CPU에 있기에 옮겨주어야함.
    true = labels.cpu()

    preds_list.extend(preds)
    true_list.extend(true)

print(f'ROC AUC Score: {roc_auc_score(true_list, preds_list):.4f}')


dataset_test = ImageDataset(df=submission, img_dir='test/', transform=transform_test)
loader_test = DataLoader(dataset=dataset_test, batch_size=32, shuffle=False)


# Evaluation + submission

model.eval()

preds=[]

with torch.no_grad():
    for images, _ in loader_test:
        images = images.to(device)
        outputs = model(images)
        preds_part = torch.softmax(outputs.cpu(), dim=1)[:,1].tolist()#to.list() needed for submission, else Tensor
        preds.extend(preds_part)

submission['has_cactus'] = preds
submission.to_csv('submission.csv', index=False)


import shutil

shutil.rmtree('./train')
shutil.rmtree('./test')

