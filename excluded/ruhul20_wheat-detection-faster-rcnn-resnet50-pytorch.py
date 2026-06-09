!pip install torch torchvision


import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random
from PIL import Image , ImageDraw 
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torchvision import transforms as T

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor




train = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")
train.head()


coord = pd.DataFrame(list(train.bbox.apply(lambda x : x[1:-1].split(",")).values),columns=["x1","y1","w","h"])


df = pd.concat([train,coord],axis=1)


df['x1']=pd.to_numeric(df['x1'])
df['y1']=pd.to_numeric(df['y1'])
df['w']=pd.to_numeric(df['w'])
df['h']=pd.to_numeric(df['h'])



df['x2']=df['x1']+df['w']
df['y2']=df['y1']+df['h']


df


df.drop(['bbox','width','height','w','h','source'],axis=1,inplace=True)


df.head()


unique_imgs=df.image_id.unique()
unique_imgs


class custDat(torch.utils.data.Dataset):
    def __init__(self, df, unique_imgs, indices):
        self.df = df
        self.unique_imgs = unique_imgs
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        image_name = self.unique_imgs[self.indices[idx]]
        boxes = self.df[self.df.image_id == image_name].values[:, 1:].astype("float")
        img = Image.open("../input/global-wheat-detection/train/" + image_name + ".jpg").convert('RGB')
        labels = torch.ones(boxes.shape[0], dtype=torch.int64)
        target = {}
        target["boxes"] = torch.tensor(boxes)
        target["label"] = labels
        return T.ToTensor()(img), target



train_inds , val_inds =train_test_split(range(unique_imgs.shape[0]),test_size=0.1)


def custom_collate(data):
    return data


train_dl = torch.utils.data.DataLoader(custDat(df,unique_imgs,train_inds),batch_size=16,shuffle=True,collate_fn=custom_collate,pin_memory=True if torch.cuda.is_available() else False)
val_dl = torch.utils.data.DataLoader(custDat(df,unique_imgs,val_inds),batch_size=8,shuffle=True,collate_fn=custom_collate,pin_memory=True if torch.cuda.is_available() else False)



model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes=2
in_features=model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor =  FastRCNNPredictor(in_features,num_classes)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


device


optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9,weight_decay=0.005)
num_epochs=10


model.to(device)
for epochs in range (num_epochs):
    epoch_loss = 0
    for data in tqdm(train_dl):
        imgs=[]
        targets = []
        for d in data :
            imgs.append(d[0].to(device))
            targ = {}
            targ['boxes']=d[1]['boxes'].to(device)
            targ['labels']=d[1]['label'].to(device)
            targets.append(targ)
        loss_dict = model(imgs,targets)
        loss= sum(v for v in loss_dict.values())
        epoch_loss += loss.cpu().detach().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(epoch_loss) 
            
            


model.eval()
data = iter(val_dl).__next__()


img= data[0][0]
boxes=data[0][1]['boxes']
labels=data[0][1]['label']


output = model([img.to(device)])


output


out_bbox=output[0]['boxes']
out_scores=output[0]['scores']


keep = torchvision.ops.nms(out_bbox,out_scores,0.45)


out_bbox.shape , keep.shape


im = (img.permute(1,2,0).cpu().detach().numpy()*255).astype('uint8')
im


vsample = Image.fromarray(im)
draw = ImageDraw.Draw(vsample)
for box in boxes :
    draw.rectangle(list(box),fill=None,outline='red')
vsample


# Save the model state
torch.save(model.state_dict(), 'modified_fasterrcnn_resnet50_fpn.pth')


