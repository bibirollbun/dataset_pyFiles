!unzip /kaggle/input/dogs-vs-cats/train.zip


from PIL import Image
import os
import torch 
import torchvision
from torchvision.transforms import transforms,v2
from torch.utils.data import Dataset,DataLoader


train_path="/kaggle/working/train"
train_dir=os.listdir(train_path)
print(len(train_dir))

cats_paths=[os.path.join(train_path,path) for path in train_dir if "cat" in path]
dogs_paths=[os.path.join(train_path,path) for path in train_dir if "dog" in path]
print(len(cats_paths),len(dogs_paths))


val_cats=cats_paths[:2500]
val_dogs=dogs_paths[:2500]

train_cats=cats_paths[2500:]
train_dogs=dogs_paths[2500:]

all_train=[]
all_valid=[]

all_train.extend(train_cats)
all_train.extend(train_dogs)

all_valid.extend(val_cats)
all_valid.extend(val_dogs)

print(f"length of training data: {len(all_train)} , length of validation data: {len(all_valid)}")


img=Image.open(cats_paths[0])
img


img=Image.open(dogs_paths[0])
img


train_transforms=v2.Compose(
    [
        v2.PILToTensor(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomResizedCrop(size=[227,227],antialias=True ),
        v2.ToDtype(torch.float32,scale=True)
    ]
)

valid_trasforms=v2.Compose(
    [
        v2.PILToTensor(),
        v2.ToDtype(torch.float32,scale=True)

    ]
)

trasform={
    "train_transform":train_transforms,
    "valid_transform":valid_trasforms
    
}


class CatsDogs(Dataset):
    def __init__(self,paths,transforms=None,is_train=True):
        self.paths=paths
        self.transforms=transforms
        self.is_train=is_train

    def __getitem__(self,index):
        img=Image.open(self.paths[index])
        img=img.resize((227,227))
        if self.transforms:
            if self.is_train:
                img=self.transforms["train_transform"](img)
            else:
                img=self.transforms["valid_transform"](img)
        label=''
        if 'cat' in self.paths[index]:
            label=0
        else:
            label=1
        return img,label
        
    def __len__(self):
        return len(self.paths)


train_dataset=CatsDogs(all_train,trasform)
valid_dataset=CatsDogs(all_valid,trasform,False)


train_loader=DataLoader(train_dataset,shuffle=True,batch_size=50)
valid_loader=DataLoader(valid_dataset,shuffle=True,batch_size=50)


import matplotlib.pyplot as plt
img=train_dataset[1500][0]
plt.imshow(img.permute(1,2,0))


import matplotlib.pyplot as plt
img=valid_dataset[1500][0]
plt.imshow(img.permute(1,2,0))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv1=nn.Conv2d(3,96,kernel_size=11,stride=4)
        self.pool1=nn.MaxPool2d(kernel_size=3,stride=2)

        self.conv2=nn.Conv2d(96,256,kernel_size=5,padding="same")
        self.pool2=nn.MaxPool2d(kernel_size=3,stride=2)

        self.conv3=nn.Conv2d(256,384,kernel_size=3,padding="same")
        self.conv4=nn.Conv2d(384,384,kernel_size=3,padding="same")
        self.conv5=nn.Conv2d(384,256,kernel_size=3,padding="same")
        self.pool3=nn.MaxPool2d(kernel_size=3,stride=2)

        self.l1=nn.Linear(9216,4096)
        self.l2=nn.Linear(4096,4096)
        self.l3=nn.Linear(4096,2)
        
    def forward(self,data):
        out=nn.ReLU()(self.conv1(data))
        out=self.pool1(out)
        
        out=nn.ReLU()(self.conv2(out))
        out=self.pool2(out)

        out=nn.ReLU()(self.conv3(out))
        out=nn.ReLU()(self.conv4(out))
        out=nn.ReLU()(self.conv5(out))
        out=self.pool3(out)
        
        out=nn.Flatten()(out)
        out=nn.ReLU()(self.l1(out))
        out=nn.ReLU()(self.l2(out))
        out=self.l3(out)
        return out


model=AlexNet()
model.to(device)


criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)



train_losses,valid_losses=[],[]


def train_epoch():
    epoch_loss=0
    for i, (imgs,labels) in enumerate(train_loader):
        imgs=imgs.to(device)
        labels=labels.to(device)

        out=model(imgs)
        loss=criterion(out,labels)
        epoch_loss+=loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss=epoch_loss/len(train_loader)
    return avg_loss
        


def valid_epoch():
    valid_loss=0
    for i, (imgs,labels) in enumerate(valid_loader):
        imgs=imgs.to(device)
        labels=labels.to(device)

        out=model(imgs)
        loss=criterion(out,labels)
        valid_loss +=loss
    avg_loss=valid_loss/len(valid_loader)
    valid_losses.append(avg_loss)
    return avg_loss


for epoch in range(5):
    train_error=train_epoch()
    val_error=valid_epoch
    print(f"epoch {epoch+1} /epochs---training error={train_error} ---validation error={val_error}")





from torchvision import models


model=models.resnet50(weights="IMAGENET1K_V1")


model


for param in model.parameters():
    param.requires_grad=False


model.fc=nn.Linear(2048,2)


model


for param in model.fc.parameters():
    print(param.requires_grad)


model.to(device)


for epoch in range(5):
    train_error=train_epoch()
    val_error=valid_epoch
    print(f"epoch {epoch+1} /epochs---training error={train_error} ---validation error={val_error}")




