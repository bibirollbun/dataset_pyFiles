import pandas as pd
sample_submission = pd.read_csv('/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')
sample_submission


import zipfile
path_to_zip_file = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'
with zipfile.ZipFile(path_to_zip_file,'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')



path_to_zip_file = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'
with zipfile.ZipFile(path_to_zip_file,'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')


import os 
from PIL import Image


import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import numpy as np 
import pandas as pd 
from PIL import Image
from matplotlib import pyplot as plt


a = os.listdir('/kaggle/working/train')
img1 = Image.open(os.path.join('/kaggle/working/train',a[0]))
print(img1.size)


b=[]
for line in os.listdir('/kaggle/working/train'):
    b.append(line.split('.')[0])
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(b)



class CustomDataset(Dataset):
    def __init__(self,directorie,transform):
        label1=[]
        filename_image=[]
        self.directory = directorie
        for line in os.listdir(directorie):
            k = line
            k = k.split('.')
            label1.append(k[0])
            filename_image.append(line)
        self.label = label1
        self.filename = filename_image
        self.transform = transform
    def __len__(self):
        return len(self.label)
    def __getitem__(self,idx):
        img = Image.open(os.path.join(self.directory,self.filename[idx]))
        if self.transform:
            img = self.transform(img)
        val = le.transform([self.label[idx]])[0]
        return img,val


transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])
train_dataset = CustomDataset('/kaggle/working/train',transform=transform)
train_dataloader = DataLoader(train_dataset,batch_size = 32,shuffle = True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,20,3,padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.edges1 = nn.Linear(16*16*20,64)
        self.edges2 = nn.Linear(64,2)
        self.activation = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.edges1(x)
        x = self.activation(x)
        x = self.edges2(x)
        return x


device = 'cuda'


model = CNN().to(device)

cr = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)
epochs = 10

corrects = 0
for i in range(epochs):
    print(i)
    cost = 0
    for image,answer in train_dataloader:
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



sample_submission = pd.read_csv('/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')


class CustomDataset1(Dataset):
    def __init__(self,directorie,transform):
        filename_image=[]
        self.directory = directorie
        for line in os.listdir(directorie):
            filename_image.append(line)
        self.filename = filename_image
        self.transform = transform
    def __len__(self):
        return len(self.filename)
    def __getitem__(self,idx):
        img = Image.open(os.path.join(self.directory,self.filename[idx]))
        if self.transform:
            img = self.transform(img)
        f = self.filename[idx]
        f = str(f)
        f = f[0:len(f)-4]
        return img,f


test_dataset = CustomDataset1('/kaggle/working/test',transform=transform)
test_dataloader = DataLoader(test_dataset,batch_size = 32,shuffle = False)


for image,filename in test_dataloader:
    image, filename = image.to(device), filename
    optimizer.zero_grad()
    predictions = model(image)
    for j in range(len(filename)):
        sample_submission.loc[int(filename[j])-1,'label'] = torch.softmax(predictions[j], dim=0)[1].item()


sample_submission.to_csv('submission.csv',index=False)


sample_submission

