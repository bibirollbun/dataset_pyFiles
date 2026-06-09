import zipfile
path_to_zip_file = '/kaggle/input/fashion-mnist-itba/fashion-mnist-itba-lab-ml-2018b.zip'
with zipfile.ZipFile(path_to_zip_file,'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')



import numpy as np
import pandas as pd


import torch.nn as nn


X_train = np.load('/kaggle/working/train_images.npy')
X_test = np.load('/kaggle/working/test_images.npy')
y_train = pd.read_csv('/kaggle/working/train_labels.csv')['label'].values


X_train.shape


import torch


images_x = torch.tensor(X_train, dtype = torch.float32).unsqueeze(1)/255

images_y = torch.tensor(y_train,dtype = torch.float32)
images_y = images_y.type(torch.LongTensor)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,20,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.edges1 = nn.Linear(14*14*20,64)
        self.edges2 = nn.Linear(64,10)
        self.activation = nn.Sigmoid()

    def forward(self,x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.edges1(x)
        x = self.activation(x)
        x = self.edges2(x)
        return x


images_x.size()


device = torch.device('cpu')


import os
from torch.utils.data import Dataset
from PIL import Image


from torch.utils.data import DataLoader
from torchvision import transforms


import torch.nn as nn
import torch.optim as optim


class ComfortableData(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __getitem__(self,idd):
        return self.x[idd],self.y[idd]
    def __len__(self):
        return len(self.x)

comfort_data = ComfortableData(images_x,images_y)
data_for_train = DataLoader(comfort_data, batch_size = 128, shuffle=True)
model = CNN().to(device)

cr = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)
epochs = 10

corrects = 0
for i in range(epochs):
    cost = 0
    for image,answer in data_for_train:
        image, answer = image.to(device), answer.to(device)
        optimizer.zero_grad()
        predictions = model(image)
        error = cr(predictions,answer)
        error.backward()
        optimizer.step()
        cost += error.item()
        if i == epochs - 1:
            for j in range(len(answer)):
                if torch.argmax(predictions[j]).item() == answer[j].item():
                    corrects += 1
    if i % 1 == 0:
        print(cost/len(data_for_train))




class ComfortableData1(Dataset):
    def __init__(self,x):
        self.x = x
    def __getitem__(self,idd):
        return self.x[idd]
    def __len__(self):
        return len(self.x)


sample = pd.read_csv('/kaggle/input/fashion-mnist-itba/submission_sample.csv')
sample


images_x = torch.tensor(X_test, dtype = torch.float32).unsqueeze(1)/255
comfort_data = ComfortableData1(images_x)
data_for_test = DataLoader(comfort_data, batch_size = 128, shuffle=False)


model.eval()
ans=[]
for image in data_for_test:
    image = image.to(device)
    optimizer.zero_grad()
    predictions = model(image)
    for j in range(len(predictions)):
        ans.append(torch.argmax(predictions[j]).item())


submission = pd.DataFrame({
    'Category': ans,
    'Id': sample['Id']
})
submission.to_csv('submission.csv',index = False)

