import pandas as pd
import numpy as np 
import os  
import re 
from PIL import Image 



train_df = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')
train_df


train_df.info()





train_df


train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1 
train_df['h'] = -1


def expand_bbox(x):
    r = np.array(re.findall('([0-9]+[.]?[0-9]*)',x)) 
    if len(r) == 0:        
        r = [-1,-1,-1,-1] 
    return r
train_df[['x','y','w','h']] = np.stack(train_df['bbox'].apply(lambda x:expand_bbox(x)))


train_df['x'] = train_df['x'].astype(np.float64) 
train_df['y'] = train_df['y'].astype(np.float64) 
train_df['w'] = train_df['w'].astype(np.float64)
train_df['h'] = train_df['h'].astype(np.float64)


train_df = train_df.drop('bbox',axis = 1)


from sklearn.model_selection import train_test_split


train_ids  , valid_ids = train_test_split(train_df , test_size  = 0.08,random_state = 42 , shuffle = True)


train_ids.shape , valid_ids.shape


from torch.utils.data import DataLoader , Dataset
import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision
import cv2
import torch








class sDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()
        
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

#-------------------------------------------------------------------------
    
    def __len__(self):
      
        return self.image_ids.shape[0]

#-------------------------------------------------------------------------
    
    def __getitem__(self, index: int):
        
        
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]
        
        # 2. قراءة الصورة
        image_path = os.path.join(self.image_dir, f'{image_id}.jpg') 
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) 
        image /= 255.0 

       
       
        boxes = records[['x', 'y', 'w', 'h']].values 
        
        
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2] # x_max = x + w
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3] # y_max = y + h
        
        
        
       
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones(len(records), dtype=torch.int64)
        iscrowd = torch.zeros(len(records), dtype=torch.int64)
        
        # 5. تجميع بيانات Target
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index]) 
        target['iscrowd'] = iscrowd
       
        if self.transforms:
   
             sample = {
                'image': image,
                'boxes': target['boxes'].tolist(),  
                'labels': labels.tolist(),         
                'shape': image.shape               
        }
    
       
    
             sample = self.transforms(**sample) 
    
    
             image = sample['image']
            
            
              
            
            
       
        return image, target, image_id


def get_train():
    return A.Compose([
        A.HorizontalFlip(0.5),
        ToTensorV2(p = 1.0)
    ],bbox_params = {'format':'pascal_voc','label_fields':['labels']}) 
def get_valid():
    return A.Compose([
        ToTensorV2(p =1.0)
        
    ],bbox_params={'format':'pascal_voc','label_fields':['labels']})


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained  = True)


num_classes = 2 

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features , num_classes)



DIR_TRAIN ='/kaggle/input/global-wheat-detection/train' 
DIR_TEST = '/kaggle/input/global-wheat-detection/test'
train_dataset = sDataset(train_df, DIR_TRAIN, get_train())
valid_dataset = sDataset(valid_ids, DIR_TRAIN, get_valid())



def collate_fn(batch):
    return tuple(zip(*batch))




train_data_loader  = DataLoader(
    train_dataset,
    batch_size = 8,
    shuffle = False,
    num_workers =4,
    collate_fn = collate_fn
)
valid_data_loader = DataLoader(
    valid_dataset,
    batch_size = 16,
    shuffle = False,
    num_workers = 4,
    collate_fn = collate_fn
)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


images ,image_ids  , target= next(iter(train_data_loader))



images = list(image.to(device) for image in images)


targets = [{k: v.to(device) for k, v in t.items()} for t in image_ids]


boxes = targets[2]['boxes'].cpu().numpy().astype(np.int32)


sample = images[2].permute(1,2,0).cpu().numpy()


import matplotlib.pyplot as plt


fig , ax=plt.subplots(1, 1, figsize=(16, 8))

for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 3)
    

ax.imshow(sample)


model.to(device)
params = [p for p in model.parameters() if p.requires_grad]


optimizer = torch.optim.SGD(params , lr = 0.005 , momentum = 0.9 , weight_decay = 0.0005) 
lr_scheduler = None
num_epochs = 2


print(images[0].size) 
print(targets[0])


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0



loss_hist = Averager()
itr = 1    

for epoch in range(num_epochs):
    loss_hist.reset() 

    for images , targets , images_ids in train_data_loader:

        
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        loss_hist.send(loss_value) 

        optimizer.zero_grad() 
        losses.backward() 
        optimizer.step() 
        if itr % 50 == 0:
            print(f'iteration #(itr) loss:{loss_value}') 
        itr += 1

    if lr_scheduler is not None:
        lr_scheduler.step()
    print(f'ecpoch {epoch} loss {loss_hist.value}')



images , targets , image_ids = next(iter(valid_data_loader))


images = list(img.to(device) for img in images) 
targets = [{k: v.to(device)for k , v in t.items()} for t in targets] 




boxes = targets[1]['boxes'].cpu().numpy().astype(np.int32) 



sample = images[1].permute(1,2,0).cpu().numpy()


fig , ax = plt.subplots(1,1,figsize = (16,8)) 
for box in boxes:
    cv2.rectangle(sample,
                 (box[0],box[1]),
                  (box[2],box[3]),
                  (50,0,0),5)
ax.imshow(sample)


model.eval() 
cpu_device = torch.device('cpu')
outputs = model(images)
outputs = [{k: v.to(cpu_device) for k , v in t.items()} for t in outputs]


torch.save(model.state_dict(), 'chacek_plants_for_Wheat.pth')




