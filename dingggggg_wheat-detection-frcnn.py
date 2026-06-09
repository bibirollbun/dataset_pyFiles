import json
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


df = pd.read_csv('../input/global-wheat-detection/train.csv')
df.head()


df.info()


df['bbox'] = df['bbox'].apply(json.loads)
fig, ax = plt.subplots(1,2,figsize=(10,10))
ax = ax.flatten()

image = plt.imread('../input/global-wheat-detection/train/{}.jpg'.format(df.image_id.iloc[0]))
ax[0].imshow(image)

for xmin,ymin,width,height in df.bbox[df.image_id == df.image_id.iloc[0]]:
    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax[1].add_patch(rect)
    ax[1].imshow(image)


%%writefile create_label.py

import os
import json
import pandas as pd

import torch

def create_label():
    df = pd.read_csv("../input/global-wheat-detection/train.csv")
    df['bbox'] = df['bbox'].apply(json.loads)
    labels = []
    image_id = list(df['image_id'].unique())
    for name in image_id:
        bbox = df[df.image_id==name]['bbox']
        bbox_list = [[xmin,ymin,xmin+w,ymin+h] for xmin,ymin,w,h in bbox]
        labels.append({
            'boxes': bbox_list,
            'labels': [1 for _ in range(len(bbox_list))],
            'image_id': name})
    #print(len(labels))
    return labels,image_id


%%writefile dataset.py

import os
import cv2
from PIL import Image
import numpy as np
import torch
from create_label import create_label
from torch.utils.data import Dataset,DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)

class MyDataset(Dataset):
    def __init__(self,root,transform=None):
        dataset,image_id = create_label()
        self.labels = dataset
        self.transform = transform
        self.image_id = image_id
        self.images_path = [os.path.join(root,i)+'.jpg' for i in self.image_id]
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self,item):
        image = cv2.imread(self.images_path[item])
        #print(self.images_path[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        labels = self.labels[item]

        if self.transform:
            # augmented = self.transform(image=image, bboxes=labels['boxes'], labels=labels['labels'])
            # image = augmented['image']

            # labels = {
            #     'boxes' : torch.tensor(augmented['bboxes'], dtype=torch.float32),
            #     'labels' : torch.tensor(augmented['labels'], dtype=torch.int64),
            #     'image_id' : labels['image_id']
            # }
            image = self.transform(image)
            labels = {
                'boxes': torch.tensor(labels['boxes'], dtype = torch.float32),
                'labels': torch.tensor(labels['labels'], dtype = torch.int64)
            }
            
        return image, labels

if __name__ == '__main__':
    train_root = '../input/global-wheat-detection/train'
    
    # train_transform = A.Compose([
    #     A.HorizontalFlip(p=0.5),  
    #     A.RandomBrightnessContrast(p=0.3),
    #     A.MotionBlur(p=0.2),
    #     A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    #     ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    train_transform = Compose([
        ToTensor()
    ])
    
    train_data = MyDataset(root=train_root,transform = train_transform)    
    train_loader = DataLoader(dataset=train_data,batch_size=16,shuffle=True,collate_fn=collate_fn)

    # for i,j in train_loader:
    #     print(j[2])
    #     break


pip install -q pprintpp


import os
import shutil
import numpy as np
import torch
from dataset import MyDataset
from torch.utils.data import DataLoader,Subset
from torchvision.transforms import Compose,ToTensor,RandomAffine,ColorJitter,RandomHorizontalFlip
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn,fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprintpp import pprint
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_arg():
    parser = ArgumentParser(description='Faster RCNN')
    parser.add_argument('--epochs','-e',type=int,default=10)
    parser.add_argument('--ratio','-r',type=float,default=0.8)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--log', '-l', type=str, default='tensorboard')
    parser.add_argument('--checkpoint', '-c', type=str, default=None)
    parser.add_argument('--trained', '-t', type=str, default='trained')
    args, unknown = parser.parse_known_args()  
    return args

def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)

if __name__ == '__main__':
    arg = get_arg()

    if not os.path.isdir(arg.trained):
        os.makedirs(arg.trained)
    if os.path.isdir(arg.log):
        shutil.rmtree(arg.log)

    writer = SummaryWriter(arg.log)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_root = '../input/global-wheat-detection/train' 

    # train_transform = A.Compose([
    #     A.HorizontalFlip(p=0.5),  
    #     A.RandomBrightnessContrast(p=0.2),
    #     A.MotionBlur(p=0.2),
    #     A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    #     ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    train_transform = Compose([
        # ColorJitter(
        #     brightness=0.125,
        #     contrast=0.5,
        #     saturation=0.5,
        #     hue=0.05
        # ),
        ToTensor()
    ])

    train_data = MyDataset(root=train_root,transform = train_transform)
    
    ratio = arg.ratio
    random.seed(42)
    index_data = [i for i in range(0,len(train_data))]
    random.shuffle(index_data)

    train_size = int(ratio * len(index_data))
    train_index = index_data[:train_size]
    val_index = index_data[train_size:]
    train = Subset(train_data,train_index)
    val = Subset(train_data,val_index)
    
    train_loader = DataLoader(dataset=train,batch_size=16,shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val,batch_size=16,shuffle=False,collate_fn=collate_fn)

    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights='DEFAULT',
                                              trainable_backbone_layers=6)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features,num_classes=2)
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001)

    if arg.checkpoint:
        checkpoint = torch.load(arg.checkpoint,map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_map = checkpoint['best_map']
    else:
        start_epoch = 0
        best_map = 0.0

    train_step = len(train)/arg.batch_size
    val_step = len(val)/arg.batch_size

    for epoch in range(start_epoch,arg.epochs):
        model.train()
        train_loss = []
        progress_bar = tqdm(train_loader)
        for iteration,(image,label) in enumerate(progress_bar):
            image = [i.to(device) for i in image]
            target = [{'boxes':i['boxes'].to(device),'labels':i['labels'].to(device)} for i in label]
            losses = model(image,target)
            final_loss = sum([loss for loss in losses.values()])

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            train_loss.append(final_loss.item())
            mean_loss = np.mean(train_loss)
            writer.add_scalar('train_loss',mean_loss,global_step=epoch*train_step+iteration)
            progress_bar.set_description('Epoch: {}/{} | Loss: {:4f}'.format(epoch+1,arg.epochs,mean_loss))

        model.eval()
        val_loss = []
        progress_bar = tqdm(val_loader)
        metric = MeanAveragePrecision(iou_type="bbox")
        for iteration,(image,label) in enumerate(progress_bar):
            image = [i.to(device) for i in image]
            pred = []
            act = []
            with torch.no_grad():
                outputs = model(image)
                for i in outputs:
                    pred.append({
                        'boxes':i['boxes'].to('cpu'),
                        'scores':i['scores'].to('cpu'),
                        'labels':i['labels'].to('cpu')
                    })
                for i in label:
                    act.append({
                        'boxes':i['boxes'],
                        'labels':i['labels']
                    })
            metric.update(pred, act)
        result = metric.compute()
        pprint(result)
        writer.add_scalar('val_map',result['map'],epoch)
        progress_bar.set_description('val_map: {:4f}'.format(result['map']))

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch+1,
            'best_map': result['map']
        }
        torch.save(checkpoint,'{}/last.pt'.format(arg.trained))
        if result['map'] > best_map:
            best_map = result['map']
            torch.save(checkpoint,'{}/best.pt'.format(arg.trained))


import cv2
import matplotlib.pyplot as plt

test_root = '../input/global-wheat-detection/test'
path = os.listdir(test_root)
images_path = []
for i in path:
    image_path = os.path.join(test_root,i)
    images_path.append(image_path)
        
model = fasterrcnn_mobilenet_v3_large_320_fpn(num_classes=2)

optimizer = torch.optim.SGD(params=model.parameters(),lr=0.001,momentum=0.9)

checkpoint = torch.load('trained/best.pt',map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()),weights_only=True)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
    
model.double().to(device)
model.eval()

#submit = []

count = 0
threshold = 0.5
for index,image in enumerate(images_path):
    text = path[index].replace('.jpg',' ')
    ori_image = cv2.imread(image)
    image = cv2.cvtColor(ori_image,cv2.COLOR_BGR2RGB)
    image = np.transpose(image,(2,0,1))/255.
    image = [torch.from_numpy(image).to(device)]
    with torch.no_grad():
        output = model(image)
        count +=1
        for i in output:
            boxes = i['boxes']
            scores = i['scores']
            labels = i['labels']
            for b,s,l in zip(boxes,scores,labels):
                xmin,ymin,xmax,ymax = b
                if s > threshold:
                    cv2.rectangle(ori_image,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(255,0,0),3)
                    cv2.putText(ori_image,str('{:.2f}'.format(s.item())),(int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX ,
                                1, (0, 255, 0), 3, cv2.LINE_AA)

                    #text += '{} {} {} {}'.format(s,xmin,ymin,xmax-xmin,ymax-ymin)
            #submit.append(text)
            
            cv2.imwrite("prediction_{}.jpg".format(count), ori_image)

fig ,ax = plt.subplots(3,3,figsize=(12,12))
ax = ax.flatten()

for index,ax_i in enumerate(ax):
    image = plt.imread('../working/prediction_{}.jpg'.format(index+1))
    ax_i.imshow(image)
    ax_i.axis('off')

