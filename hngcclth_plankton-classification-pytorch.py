# Manipulation Libraries

from termcolor import cprint
import os
from glob import glob
import random
from warnings import filterwarnings
filterwarnings('ignore')


# supporting libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import random

#importing pytorch and associated libraries

import torch
import torch.nn as nn
from torchvision import transforms as transforms
from torch.utils.data import Dataset, DataLoader


# unzipping train folder
!unzip -q ../input/datasciencebowl/train.zip


# unzipping test folder
!unzip -q ../input/datasciencebowl/test.zip


#unzipping sample submission file
!unzip ../input/datasciencebowl/sampleSubmission.csv.zip


# Class distribution in PIE_CHART 

class_names = []
class_count = []

for name in os.listdir('./train/'):
    class_names.append(name)
    class_count.append(len(os.listdir(f'./train/{name}')))
plt.figure(figsize=(10, 10))
plt.pie(class_count, labels = class_names)
plt.title('Class Distribution (Train)')
plt.show()


# Class distribution in BAR-PLOT

plt.figure(figsize=(20, 8))
sns.barplot(class_names, class_count, palette = 'Blues')
plt.title('Class Distribution (Train)')
plt.xticks(rotation=90)
plt.show()


# fixing the seeds

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# Neural Network blocks and models

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kerel_size = 3, stride = 1, padding = 0):
        super(Conv, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kerel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x):
        return self.seq(x)

class PlankNet(nn.Module):
    def __init__(self, in_channels, num_classes, H = 128, W = 128):
        super(PlankNet, self).__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 16, 4), #125
            nn.MaxPool2d(2), #62
            Conv(16, 32, 3), #60
            Conv(32, 64, 3), #58
            nn.Dropout(0.1),
            nn.MaxPool2d(2), # 29
            Conv(64, 128), # 27
            nn.Dropout(0.2),
            Conv(128, 64, 3), # 25
            Conv(64, 32, 3), # 23
            nn.Flatten(),
            nn.Linear(32*23*23 , 4096),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        return self.model(x)
num_classes = len(class_names)
model = PlankNet(3, num_classes, 128, 128)


rand_data = torch.rand(1, 3, 128, 128)
print(model(rand_data).shape)


cprint(model, "blue")


# Model layers overview

for name, param in model.named_parameters():
    print(f"{name} : {param.shape}")


#Generating csv file to gather data of images 

def generate_csv(root,train = True, img_ext = 'jpg'):
    df = pd.DataFrame(columns = ['path', 'class'])
    if train:
        for index,label in enumerate(os.listdir(root)):
            links = glob(f"{root}/{label}/*{img_ext}")
            temp_df = pd.DataFrame({'path': links, 'class': np.ones(len(links), dtype='float32')*index})
            df = pd.concat([df, temp_df], axis = 0)
    else:
        links = glob(f"{root}/*{img_ext}")
        temp_df = pd.DataFrame({'path': links, 'class': np.ones(len(links), dtype = 'float32')})
        df = pd.concat([df, temp_df], axis = 0)
        
    return df
        


train_csv = generate_csv('./train')
train_csv.head()


test_csv = generate_csv('./test',train = False)
test_csv.head()


# sorting the image by their names

test_csv.sort_values('path', inplace = True)
test_csv.head()


# Genrating dataset classes

def load_image(path , H, W):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (H,W))
    return img

def transform():
    return transforms.Compose([
        transforms.RandomRotation(90),
        transforms.ToTensor()
    ])


class PlanktonDataset(Dataset):
    def __init__(self, df, H = 128, W = 128, transform = None):
        super(PlanktonDataset, self).__init__()
        self.df = df
        self.H = H
        self.W = W
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        path = self.df.iloc[index, 0]
        img = load_image(path, H = self.H, W = self.W)
        label = self.df.iloc[index, 1]
        img = Image.fromarray(img)
        if self.transform != None:
            img = self.transform()(img)
        else:
            img = transforms.ToTensor()(img)
        return (img, (label, path))
    
# Creating the train and test datasets.   
train_ds = PlanktonDataset(train_csv, 128, 128, transform = transform)
test_ds = PlanktonDataset(test_csv, 128, 128)


# data access format

image, label = test_ds.__getitem__(3)
image.shape, label[0], label[1]


# Train-Validation Split :

spl_idx = int(train_ds.__len__() * 0.75)
print(f"Splitting index : {spl_idx}")
train_ds, val_ds = torch.utils.data.random_split(train_ds,[spl_idx, train_ds.__len__() - spl_idx] )


print(f"Size of train dataset : {train_ds.__len__()}")
print(f"Size of validation dataset : {val_ds.__len__()}")
print(f"Size of test dataset : {test_ds.__len__()}")


# Creating data loaders specifying the batch size

BATCH_SIZE = 64
train_dl = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True)
val_dl = DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle = True)
test_dl = DataLoader(test_ds, batch_size = BATCH_SIZE, shuffle = False)


# Chossing training hyperparameters and also the optimizer and loss

EPOCHS = 30
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(params = model.parameters(), lr = 1e-4)


train_loss = []
val_loss = []


# Model Training...

model = model.cuda()  # Putting the model inside GPU
best_loss = np.inf
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1} : \n")
    TR_LOSS = 0.0
    VAL_LOSS = 0.0
    model.train()
    
    # Train Data Forward & Backward Pass
    
    for index, (train_patch, (labels, _)) in enumerate(train_dl):
        optim.zero_grad()
        train_patch = train_patch.cuda()
        labels = labels.long().cuda()
        op = model(train_patch)
        tloss = criterion(op, labels)
        TR_LOSS += tloss.item()
        train_loss.append(tloss.item())
        tloss.backward()
        optim.step()
        
        if index % 100 == 99:
            print(f"         Step {index + 1} Loss : {'%.4f'%(tloss.item())}")   
    model.eval()
    
    # Validation Checking ( Only Forward Pass )
    
    with torch.no_grad():
        for index, (val_patch, (labels, _)) in enumerate(val_dl):
            val_patch = val_patch.cuda()
            labels = labels.long().cuda()
            op = model(val_patch)
            vloss = criterion(op, labels)
            VAL_LOSS += vloss.item()
            val_loss.append(vloss.item())
    print(f"\n     Training Loss : {'%.4f'%(TR_LOSS)}  ||  Validation Loss : {'%.4f'%(VAL_LOSS)}\n")
    
    if VAL_LOSS < best_loss :      # Model Updationg
        cprint("Model Updation : Success!\n", 'green')
        torch.save(model, 'best_model.pth')
        best_loss = VAL_LOSS
    else:
        cprint("Model Updation : Failed!\n", 'red')
cprint('Training completed...', 'green')


plt.figure(figsize=(20,8))
plt.plot(train_loss)
plt.title('Train Loss', size = 20)
plt.xlabel('STEPS')
plt.ylabel('LOSS')
plt.show()


plt.figure(figsize=(20,8))
plt.plot(val_loss)
plt.title('Validation Loss', size = 20)
plt.xlabel('STEPS')
plt.ylabel('LOSS')
plt.show()


best_model = torch.load('./best_model.pth')
cprint(best_model, 'blue')


# to filter only the names of the images

def preprocess_names(names_list):
    name_list = []
    for name in names_list:
        name_list.append(name.split('/')[-1])
    return name_list


# Test data generating function

def create_submission_file(test_dataloader, model, class_names):
    df = pd.DataFrame()
    model.eval()
    sf_layer = nn.Softmax(dim = 1)
    with torch.no_grad():
        for index, (test_patch, attr) in enumerate(test_dataloader):  # loading test data
            paths = preprocess_names(attr[1])
            test_patch = test_patch.cuda()
            op = model(test_patch)
            op = sf_layer(op)
            paths = np.array(list(paths))   #adding the filenames in a list
            op = op.cpu().detach().numpy()
            patch_df = pd.DataFrame(op, columns = class_names)  #creating the patch dataframe
            patch_df.insert(0, 'image', paths)
            df = pd.concat([df, patch_df], axis = 0)    # joining the patch dataframe with the main one.
            
            if index % 100 == 99:
                print(f"{index + 1} Steps Completed...\n")    
    print('Test Dataframe Generated...\n')
    
    return df


best_model = best_model.cuda()
pred_df = create_submission_file(test_dl, best_model, class_names)
pred_df.head()


# Checking if all the imae names are different
assert pred_df['image'].nunique() == pred_df.shape[0], " Submission format not correct!"
cprint('Submission correcty created !', 'green')


pred_df.to_csv('submission.csv', index = False)

