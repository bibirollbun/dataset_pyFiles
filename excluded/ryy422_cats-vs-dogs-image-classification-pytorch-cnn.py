# Importing all necessary libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import SubsetRandomSampler

from PIL import Image
import cv2

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torchvision.transforms import Resize
from torchvision.transforms import Compose


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Unzipping all training images
!unzip /kaggle/input/dogs-vs-cats/train.zip


# Checking total Images in training data
files = os.listdir("./train")
len(files)


# Reading image using PIL
im_PIL = np.array(Image.open("./train/cat.0.jpg"))
im_PIL


# Transforming Image np array into tensor using ToTensor
tra = Compose([ToTensor()])
tra_im_PIL = tra(im_PIL)
print(f"Mean of Image channels: {tra_im_PIL.mean([1,2])}")
print(f"Std of Image channels: {tra_im_PIL.std([1,2])}")
tra_im_PIL


# Plotting Image 

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(im_PIL)
plt.title("Image before transformation")
plt.subplot(1,2,2)
plt.imshow(tra_im_PIL.permute(1,2,0))
plt.title("Image after transformation")


# Creating labels for images using dataframes

Labels = pd.DataFrame(columns=["File_name","Label"])
Labels["Label"] = Labels["Label"].astype(dtype=float)


# Assigning labels to the images
i=0
for  dirname, _, filenames in os.walk("./train"):
    for filename in filenames:
        if filename.find("cat") != -1:
            Labels.loc[i,"File_name"] = filename
            Labels.loc[i,"Label"] = 0
        else:
            Labels.loc[i,"File_name"] = filename
            Labels.loc[i,"Label"] = 1
        i += 1


# checking label file
Labels.head()


# getting channels, Width and Height from the Images

ch = torch.zeros(Labels["File_name"].shape[0])
W = torch.zeros(Labels["File_name"].shape[0])
H = torch.zeros(Labels["File_name"].shape[0])
i=0

for fname in Labels["File_name"]:
    im = read_image("./train/"+fname)
    ch[i] = im.shape[0]
    W[i] = im.shape[1]
    H[i] = im.shape[2]
    i += 1 


# checking max/min of channels, width and height

print(f"max channels: {torch.max(ch)}")
print(f"min channels: {torch.min(ch)}")
print(f"max Width: {torch.max(W)}")
print(f"min Width: {torch.min(W)}")
print(f"max Height: {torch.max(H)}")
print(f"min Height: {torch.min(H)}\n")

print(f"min idx Width: {torch.argmin(W)}")
print(f"max idx Width: {torch.argmax(W)}")
print(f"min idx Height: {torch.argmin(H)}")
print(f"max idx Height: {torch.argmax(H)}")


# getting indexes of images under width and height of 100

minW = W<100
minW_idx = minW.nonzero()

minH = H<100
minH_idx = minH.nonzero()


# Plotting Images with width less than 100

resize = Resize((240,240))
plt.figure(figsize = (15,15))
for i in range(15):
    plt.subplot(5,3,i+1)
    plt.imshow(resize(read_image("./train/"+Labels.iloc[int(minW_idx[i]),0])).permute(1,2,0))


# checking mean and median of height and width of images

print(f"Width mean: {W.mean()} and Height: {W.median()}")
print(f"Width mean: {H.mean()} and Height: {H.median()}")


# plotting height and Width of images
plt.figure(figsize = (20,10))
plt.scatter(W,H)


# Data Traformation for Images using Compose
Data_transform = Compose([ToTensor(), Resize(size = (240,240))])


# Creating Custom Dataset class for image and labels

class CvD_Data(Dataset):
    def __init__(self, Label_file, Imgs_dir, transform_Img=None, transform_label=None):
        self.Labels = Label_file
        self.Imgs_dir = Imgs_dir
        self.transform_Img = transform_Img
        self.transform_label = transform_label
    
    def __len__(self):
        return len(self.Labels)
    
    def __getitem__(self,idx):
        
        img_path = os.path.join(self.Imgs_dir, self.Labels.iloc[idx,0])
        image = np.array(Image.open(img_path))
        
        label = self.Labels.iloc[idx,1]
    
        if self.transform_Img:
            image = self.transform_Img(image)
            
        if self.transform_label:
            label = torch.tensor(label, dtype=torch.float32)

        return image, label
            


# creating Dataset object from training data 
DatasetCD = CvD_Data(Labels, "./train", Data_transform, ToTensor())


# Splitting data into training and validation sets

validation_split = 0.2
random_seed= 42

dataset_size = len(DatasetCD)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

np.random.seed(random_seed)
np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(DatasetCD, batch_size=100, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(DatasetCD, batch_size=100,
                                                sampler=valid_sampler)


# Creating a sample iterable for train_loader
ex_iter = iter(train_loader)
demo_imgs, demo_lbls = ex_iter.next()


print(f"Images shape: {demo_imgs.shape}\nLabels shape: {demo_lbls.shape}")
print(f"Images dtype: {demo_imgs.dtype}\nLabels dtype: {demo_lbls.dtype}")


# plotting sample training Images
plt.figure(figsize=(10,10))
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.imshow(demo_imgs[i].permute(1,2,0))
  plt.title(demo_lbls[i].item())


# setting device to cuda if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# Creating our custom convolutional model

class CatsVDogs(nn.Module):
    def __init__(self):
        super().__init__()
        self.convL1 = nn.Conv2d(3,6,5)
        self.MaxPool = nn.MaxPool2d(2,2)
        self.convL2 = nn.Conv2d(6,6,5)
        self.convL3 = nn.Conv2d(6,12,3,2)
        self.convL4 = nn.Conv2d(12,12,2,2)
        self.fc1 = nn.Linear(12*7*7,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,1)
    
    def forward(self, X):
        x = self.MaxPool(F.relu(self.convL1(X)))
        x = self.MaxPool(F.relu(self.convL2(x)))
        x = F.relu(self.convL3(x))
        x = self.MaxPool(F.relu(self.convL4(x)))
        x = x.view(-1,12*7*7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
    
        return x
               


model_custom = CatsVDogs().to(device)
print(model_custom)


# Total Parameters in custom model

total_param = 0
for i in model_custom.parameters():
    total_param = total_param + i.flatten().shape[0]
print(f"Total Parameters in pretrained model: {total_param}")


# setting loss and optimizer for custom model
Loss_Custom = nn.BCELoss()
Optim_Custom = torch.optim.Adam(model_custom.parameters(), lr=0.001)

n_steps = len(train_loader)
epochs = 3


# Training Custom model 

for epoch in range(epochs):
    for i, (Imgs, Labels) in enumerate(train_loader):
        
        Imgs = Imgs.float()
        Labels = Labels.view(-1,1)
        
        Imgs = Imgs.to(device)
        Labels = Labels.to(device)
        
        Predicts = model_custom(Imgs)
        loss = Loss_Custom(Predicts, Labels)
        
        
        Optim_Custom.zero_grad()
        loss.backward()
        Optim_Custom.step()
        
        if (i+1) % 50 == 0 :
            with torch.no_grad():
                count = 0
                for j in range(len(Labels)):
                    if Predicts[j]>=0.5:
                        Predicts[j] = 1
                    else:
                        Predicts[j] = 0
                    if Predicts[j] == Labels[j]:
                        count += 1    
                acc = count/len(Labels)
                acc = acc * 100
                
            print(f"epoch: {epoch+1}/{epochs}, step: {i+1}/{n_steps}, loss: {loss.item():.4f}, Accuracy: {acc:.2f}")


# Testing Custom model on validation set

correct = 0
for i, (Imgs, Labels) in enumerate(validation_loader):
            
    Imgs = Imgs.float()
    Labels = Labels.view(-1,1)
        
    Imgs = Imgs.to(device)
    Labels = Labels.to(device)
    
    model_custom.eval() 
        
    with torch.no_grad():
        Predicts = model_custom(Imgs)
        
        for j in range(len(Labels)):
            if Predicts[j] >=0.5:
                Predicts[j] = 1
            else:
                Predicts[j] = 0
            
            if Predicts[j] == Labels[j]:
                correct += 1    
                
Acc_cust = correct/(len(validation_loader)*100)
Acc_cust = Acc_cust*100
print(f"Accuracy of Custom model over Validation set is: {Acc_cust:.2f}")


# Since our Custom model is doing slightly better than random guessing we will use pretrained model 
# using Shufflenet v2 x1.0 pretrained model

shuffle_net_v2x1 = torchvision.models.shufflenet_v2_x1_0(pretrained=True, progress=True)
print(shuffle_net_v2x1)


# changing final fc layer to 1 since our problem is binary classification

shuffle_net_v2x1.fc = nn.Linear(shuffle_net_v2x1.fc.in_features, 1)
print(shuffle_net_v2x1)


# creating shufflenet model 
model = shuffle_net_v2x1.to(device)


# Total Parameters in Shuffle net pretrained model

total_param = 0
for i in model.parameters():
    total_param = total_param + i.flatten().shape[0]
print(f"Total Parameters in pretrained model: {total_param}")


# setting loss and optimizer for model
Loss = nn.BCEWithLogitsLoss()
Optim = torch.optim.Adam(model.parameters(), lr=0.001)

n_steps = len(train_loader)
epochs = 5


# Training Shuffle net Pretrained model

for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        
        imgs = imgs.float()
        labels = labels.view(-1,1)
        
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        Predicts = model(imgs)
        loss = Loss(Predicts, labels)
        
        Optim.zero_grad()
        loss.backward()
        Optim.step()
        
        if (i+1) % 50 == 0 :
            with torch.no_grad():
                pred = torch.sigmoid(Predicts).clone()   # applying sigmoid because final layer of shuffle net is linear
                correct = 0
                for j in range(len(labels)):
                    if pred[j]>=0.5:                     # predicting output 0 or 1 if value is greater or less than 0.5
                        pred[j] = 1
                    else:
                        pred[j] = 0
                    if pred[j] == labels[j]:
                        correct += 1    
                acc = correct/len(labels)
                acc = acc*100
                
            print(f"epoch: {epoch+1}/{epochs}, step: {i+1}/{n_steps}, loss: {loss.item():.4f}, Accuracy: {acc:.2f}")


# Testing Shuffle net Pretrained model on Validation set

correct = 0
for i, (Imgs, Labels) in enumerate(validation_loader):
            
    Imgs = Imgs.float()
    Labels = Labels.view(-1,1)
        
    Imgs = Imgs.to(device)
    Labels = Labels.to(device)
    
    model.eval() 
        
    with torch.no_grad():
        Predicts = model(Imgs)
        pred = torch.sigmoid(Predicts).clone()
        
        for j in range(len(Labels)):
            if pred[j] >=0.5:
                pred[j] = 1
            else:
                pred[j] = 0
            
            if pred[j] == Labels[j]:
                correct += 1  

Acc_shuffle = correct/(len(validation_loader)*100)
Acc_shuffle = Acc_shuffle*100

print(f"Accuracy of shuffle net model over Validation set is: {Acc_shuffle:.2f}")


# saving custom and shufflenet model
torch.save(model_custom,"./Custom_Cnn")
torch.save(model, "./ShuffleNet_v2")


!unzip /kaggle/input/dogs-vs-cats/test1.zip


# Checking total Images in training data
test_files = os.listdir("./test1")
len(test_files)


# Classifying Images from test set using shuffle net

sampleSubmission = pd.DataFrame(columns=["label"])

i=0
for  dirname, _, filenames in os.walk("./test1"):
    for filename in filenames:
        image = np.array(Image.open("./test1/"+filename))  # Image Loading
        image = Data_transform(image)                      # data transformation
        
        image = image.float()
        image = image.view(1,3,240,240)
        image = image.to(device)
        model.eval()
        with torch.no_grad():
            Predict = model(image)
            pred = torch.sigmoid(Predict).clone()
            if pred >= 0.5:
                pred = 1
            else:
                pred = 0
            
            sampleSubmission.loc[i] = pred
        
        i += 1


sampleSubmission["id"] = sampleSubmission.index


sampleSubmission = sampleSubmission[["id","label"]]
sampleSubmission.head()


sampleSubmission.to_csv("sampleSubmission.csv", index=False)

