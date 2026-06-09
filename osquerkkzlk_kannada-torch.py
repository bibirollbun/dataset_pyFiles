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


import torch
from torch import nn
import pandas as pd
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision.transforms import v2
import os


dir="/kaggle/input/Kannada-MNIST"
original_train=pd.read_csv(os.path.join(dir,"train.csv"))
original_test=pd.read_csv(os.path.join(dir,"test.csv"))
all_data=pd.concat((original_train.iloc[:,1:],original_test.iloc[:,1:]),axis=0)
all_data=all_data.values.astype(np.float32).reshape(-1,1,28,28)

train=original_train.shape[0]
data_train=all_data[:train]
data_test=all_data[train:]
data_target=original_train.iloc[:,0].values.astype(np.int64)


# 处理数据
transforms_train=v2.Compose([
    v2.Resize(40),
    v2.RandomResizedCrop(28,scale=(0.8,1.1),ratio=(1,1)),
    v2.RandomAffine(degrees=15,translate=(0.1,0.1),scale=(0.9,1.1),shear=10),]
)
class dataset(Dataset):
    def __init__(self,x,y=None,transform=None):
        self.x=x
        self.y=y
        self.transform=transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        x  , y=self.x[idx],self.y[idx] if self.y is not None else None
        if self.transform is not None:
            x=self.transform(x)
        return x , y


# 显示 10张图像
def display (data,target):
    _,axes=plt.subplots(nrows=2,ncols=5,figsize=(8,6))
    for i in range(2):
        for j in range(5):
            axes[i,j].imshow(data[i*5+j].squeeze(0),cmap="viridis")
            axes[i,j].set_title(f"Num_{target[i*5+j]}")
    plt.tight_layout()
display(data_train,data_target)


# 基本参数设置
batch_size=64
epochs=10
num_classes=10
device="cuda"


# 划分数据集
from sklearn.model_selection import train_test_split

data_train=torch.tensor(data_train,device=device,dtype=torch.float32)
data_target=torch.tensor(data_target,device=device,dtype=torch.int64)
data_test=torch.tensor(data_test,device=device,dtype=torch.float32)
x_train,x_val,y_train,y_val=train_test_split(data_train,data_target)
x_train.shape,x_val.shape


train_iter=DataLoader(dataset(x_train,y_train,transform=transforms_train),batch_size,shuffle=True)
val_iter=DataLoader(dataset(x_val,y_val),batch_size,shuffle=False)
test_iter=DataLoader(dataset(data_test),batch_size,shuffle=False)


# 定义模型

model=nn.Sequential(
    nn.Conv2d(1,16,3,padding=1),
    nn.Conv2d(16,32,3,padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.AvgPool2d((4,4)),

    nn.Flatten(),
    nn.Linear(7*7*32,128),
    nn.ReLU(),
    nn.Dropout(0.2),
    
    nn.Linear(128,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Dropout(0.4),

    nn.Linear(16,num_classes)
)

def init_weight(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
        
model.apply(init_weight)


optim=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode="min",factor=0.5,patience=1)
loss=nn.CrossEntropyLoss()


def accuracy(net,data_iter):
    net.eval()
    with torch.no_grad():
        accurate=all_num=0
        for x,y in data_iter:
            x,y=x.to(device),y.to(device)
            pred=net(x).argmax(dim=1)
            accurate+=(pred==y).sum().item()
            all_num+=len(y)
    net.train()
    return accurate/all_num


def compute_loss(net,data_iter):
    net.eval()
    with torch.no_grad():
        l=all_num=0
        for x,y in data_iter:
            x,y=x.to(device),y.to(device)
            l_epoch=loss(net(x),y)
            l+=l_epoch.item()*len(y)
            all_num+=len(y)
    net.train()
    return l/all_num


# 开始训练
from sklearn.metrics import accuracy_score
model.train()
model.to(device)
train_loss_record=[]
val_loss_record=[]
train_acc=[]

for epoch in range(epochs):
    l_epoch=0
    train_num=0
    for x,y in train_iter:
        x,y=x.to(device),y.to(device)
        optim.zero_grad()
        pred=model(x)
        l=loss(pred,y)
        l.backward()
        l_epoch+=l.item()*len(y)
        train_num+=len(y)
        optim.step()

    l_epoch/=train_num
    train_epoch_acc=accuracy(model,train_iter)
    val_loss=compute_loss(model,val_iter)
    scheduler.step(val_loss)
    
    train_loss_record.append(l_epoch)
    val_loss_record.append(val_loss)
    train_acc.append(train_epoch_acc)
    
    print(f"<{epoch+1}>  正在训练")
    print(f"<训练损失>：{l_epoch}")
    print(f"<验证损失>：{val_loss}")
    print(f"<训练精度>：{train_epoch_acc}")
    


sample=pd.read_csv(os.path.join(dir,"Dig-MNIST.csv")) 
sample_train=sample.iloc[:,1:].values.astype(np.float32)
sample_label=sample.iloc[:,0].values.astype(np.int64) 

sample_train=torch.tensor(sample_train,dtype=torch.float32).reshape(-1,1,28,28)
sample_label=torch.tensor(sample_label,dtype=torch.long)
sample_iter=DataLoader(dataset(sample_train,sample_label),batch_size,shuffle=False)
accuracy(model,sample_iter)



# 开始预测
from torch.utils.data import TensorDataset
model.eval()
with torch.no_grad():
    temp=[]
    for x in DataLoader(TensorDataset(data_test.reshape(-1,1,28,28)),batch_size,shuffle=False):
        pred=model(x[0])
        pred=pred.argmax(dim=1)
        temp.append(pred.cpu().numpy())
    temp=np.concatenate(temp,axis=0)
    sub_=pd.read_csv(os.path.join(dir,"sample_submission.csv"))
    sub_["label"]=temp
    sub_.to_csv("/kaggle/working/submission.csv",index=False)


# os.remove(os.path.join(dir,"/kaggle/working/submission1.csv"))




