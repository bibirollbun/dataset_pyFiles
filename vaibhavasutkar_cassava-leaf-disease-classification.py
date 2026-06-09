import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os




import torch
import torchvision
from torch.utils.data import DataLoader,random_split
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.transforms import ToTensor
import torchvision.transforms as T



data_dir='/kaggle/input/cassava-leaf-disease-classification/train_images'
label_dir='/kaggle/input/cassava-leaf-disease-classification/train.csv'


from torch.utils.data import dataset
import os

files=os.listdir(data_dir)


len(files)


files[1]





df=pd.read_csv(label_dir)
df.head()


def open_image(path):
    with open(path,'rb') as f:
        img=Image.open(f)
        return img.convert('RGB')


from torch.utils.data import Dataset
class CassavaLeaf(Dataset):
    def __init__(self,root,transform):
        super().__init__()
        self.root=root
        self.files=[fname for fname in os.listdir(root) if fname.endswith('jpg')]
        self.classes=df.iloc[:,1]
        self.transform=transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self,i):
        fname=self.files[i]
        fpath=os.path.join(self.root,fname)
        img=self.transform(open_image(fpath))
        class_idx=self.classes[i]
        return img,class_idx


imagenet_stats=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

image_size=120
dataset=CassavaLeaf(data_dir,T.Compose([T.Resize(image_size),
                                       T.Pad(8,padding_mode='reflect'),
                                       T.RandomCrop(image_size),
                                       T.ToTensor(),
                                       T.Normalize(*imagenet_stats)]))


len(dataset)


train_ds,val_ds=random_split(dataset,[20000,1397])



train_dl=DataLoader(train_ds,batch_size=20,shuffle=True,num_workers=3,pin_memory=True)
val_dl=DataLoader(val_ds,batch_size=20,num_workers=3,pin_memory=True)


import json

json_file = open('/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json')
classes_name = json.load(json_file)
classes_name


import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

def denormalize(image,mean,std):
    # if len(image.shape)==3:
        image=image.unsqueeze(0)
        mean=torch.tensor(mean).reshape(1,3,1,1)
        std=torch.tensor(std).reshape(1,3,1,1)
        return image *std+mean

def show_image(img_tensor,label):
  print('label : ',classes_name[str(label)],'('+str(label)+')')
  img_tensor=denormalize(img_tensor,*imagenet_stats)[0].permute((1,2,0))
  plt.imshow(img_tensor)


show_image(*dataset[0])


from torchvision.utils import make_grid

def show_batch(dl):
  for image,label in dl:
    print(image.shape)
      
    fig,ax=plt.subplots(figsize=(16,16))
    ax.set_xticks([]);ax.set_yticks([])
    image=denormalize(image[:64],*imagenet_stats)
    # print(image[0].shape)
    ax.imshow(make_grid(image[0],nrow=8).permute(1,2,0))
    break

show_batch(train_dl)


def accuracy(output,label):
    _,pred=torch.max(output,dim=1)
    return torch.tensor(torch.sum(pred==label).item()/len(pred))

class ImageClassifierBase(nn.Module):
    def training_step(self,batch):
        image,label=batch
        out=self(image)
        loss=F.cross_entropy(out,label)
        return loss

    def validation_step(self,batch):
        image,label=batch
        out=self(image)
        val_loss=F.cross_entropy(out,label)
        val_acc=accuracy(out,label)
        return {'val_loss':val_loss,'val_acc':val_acc}

    def validation_epoch_end(self,output):
         batch_loss=[x['val_loss'] for x in output]
         epoch_loss=torch.stack(batch_loss).mean()
         batch_acc=[x['val_acc'] for x in output]
         epoch_acc=torch.stack(batch_acc).mean()
         return {'val_loss':epoch_loss.item(),'val_acc':epoch_acc.item()}

    def epoch_end(self,epoch,result):
        print("Epoch [{}],{} train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, "last_lr: {:.5f},".format(result['lrs'][-1]) if 'lrs' in result else '',
            result['train_loss'], result['val_loss'], result['val_acc']))

    
    @torch.no_grad()
    def evaluate(self,val_loader):
         self.eval()
         output=[self.validation_step(batch) for batch in val_loader]
         return self.validation_epoch_end(output)


from torchvision import models

#models.list_models()

class CassavaModel(ImageClassifierBase):
    def __init__(self,num_classes,pretrained=True):
        super().__init__()
        self.network=models.resnet50(pretrained=pretrained)
        self.network.fc=nn.Linear(self.network.fc.in_features,num_classes)

    def forward(self,xb):
        return self.network(xb)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)


class DeviceDataLoader():
    def __init__(self,dl,device):
        self.dl=dl
        self.device=device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b,self.device)

    def __len__(self):
        return len(self.dl)


from tqdm.notebook import tqdm

def fit(epochs,lr,model,train_dl,val_dl,opt_fn=torch.optim.SGD):
    history=[]
    opt=opt_fn(model.parameter(),lr)

    for epoch in range(epochs):
        model.train()
        train_losses=[]

        for batch in tqdm(train_dl):
            loss=model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            opt.step()
            opt.zero_grad()
        result=model.evaluate(val_dl)
        result['train_loss']=torch.stack(train_losses).mean().item()
        model.epoch(epoch,result)
        history.append(result)
    return history


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs,mox_lr,model,train_dl,val_dl,weight_decay=0,
                  grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history=[]
    optimizer=opt_func(model.parameters(),max_lr,weight_decay=weight_decay)

    sched=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr,epochs=epochs,
                                             steps_per_epoch=len(train_dl))

    for epoch in range(epochs):
        model.train()
        train_losses=[]
        lrs=[]
        for batch in tqdm(train_dl):
            loss=model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(),grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            lrs.append(get_lr(optimizer))
            sched.step()
        result=model.evaluate(val_dl)
        result['train_loss']=torch.stack(train_losses).mean().item()
        result['lrs']=lrs
        model.epoch_end(epoch,result)
        history.append(result)
    return history
            


device=get_default_device()
device


train_dl=DeviceDataLoader(train_dl,device)
val_dl=DeviceDataLoader(val_dl,device)


model=CassavaModel(len(dataset.classes))


to_device(model, device);


history=[model.evaluate(val_dl)]



history


epochs = 6
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam


history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
                         grad_clip=grad_clip,
                         weight_decay=weight_decay,
                         opt_func=opt_func)


































