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


import os

# Create a list to store file paths
all_input_files = []

# Walk through the directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        full_path = os.path.join(dirname, filename)
        all_input_files.append(full_path)

# Now `all_input_files` contains paths to all files in /kaggle/input


!pip install -q segmentation-models-pytorch


import os
import cv2
import math
import numpy as np
import pandas as pd
from PIL import Image
import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torchvision
from torchvision.transforms import v2
from torchvision.transforms import functional as TF
from torchvision import tv_tensors

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn import functional as F

import segmentation_models_pytorch as smp
import torchmetrics as tm
from torchmetrics.aggregation import MeanMetric

from scipy.spatial.distance import directed_hausdorff


normalizer = lambda x: (x - x.min()) / (x.max() - x.min())   #to make MRI images visible


def num_trainable_params(model):   #Count the model parameters (weights+biases)
  nums = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
  return nums






def set_seed(seed):     #to avoid changing data everytime it runs
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)



    



def segmented_image_show(image, mask):   #to show MRI image with its mask
    
    plt.subplot(1, 2, 1)            #View the raw photo
    plt.imshow(image, cmap='gray')
    plt.axis(False)
    plt.title('MRI image')     
#-------------------------------------------------------------------------------------------------
    plt.subplot(1, 2, 2)           #View segmented photo
    plt.imshow(image, cmap='gray');
    for i, color in zip(range(3), ['#ffd670', '#f15bb5', '#00f5d4']):#Making masks for each class with designated colors
        cmap = mcolors.ListedColormap(['none', color])
        plt.imshow(mask[i, :, :], cmap=cmap, alpha=0.7)
        
    plt.axis(False);
    plt.title('Segmented image');   
#--------------------------------------------------------------------------------------------------
    plt.figure(figsize=(2, 1))     #View colorbar and class names
    plt.scatter([], [], c='#ffd670', label='large_bowel', s=100)
    plt.scatter([], [], c='#f15bb5', label='small_bowel', s=100)
    plt.scatter([], [], c='#00f5d4', label='stomach', s=100)
    plt.axis(False)
    plt.legend()
    plt.show()



def model_output_show(image, target, output):   #to show MRI image with its mask
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)            #View the raw photo
    plt.imshow(image, cmap='gray')
    plt.axis(False)
    plt.title('MRI image')     
#-------------------------------------------------------------------------------------------------
    plt.subplot(1, 3, 2)           #View segmented photo
    plt.imshow(image, cmap='gray');
    for i, color in zip(range(3), ['#ffd670', '#f15bb5', '#00f5d4']):#Making masks for each class with designated colors
        cmap = mcolors.ListedColormap(['none', color])
        plt.imshow(target[i, :, :], cmap=cmap, alpha=0.7)
        
    plt.axis(False);
    plt.title('target');   
#--------------------------------------------------------------------------------------------------
    plt.subplot(1, 3, 3)           #View segmented photo
    plt.imshow(image, cmap='gray');
    for i, color in zip(range(3), ['#ffd670', '#f15bb5', '#00f5d4']):#Making masks for each class with designated colors
        cmap = mcolors.ListedColormap(['none', color])
        plt.imshow(output[i, :, :], cmap=cmap, alpha=0.7)
        
    plt.axis(False);
    plt.title('output'); 
    
#---------------------------------------------------------------------------------------------------
    plt.figure(figsize=(2, 1))     #View colorbar and class names
    plt.scatter([], [], c='#ffd670', label='large_bowel', s=100)
    plt.scatter([], [], c='#f15bb5', label='small_bowel', s=100)
    plt.scatter([], [], c='#00f5d4', label='stomach', s=100)
    plt.axis(False)
    plt.legend()
    plt.show()


def train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch=None):
    model.train()
    loss_train = MeanMetric()
    metric.reset()
    
    with tqdm.tqdm(train_loader, unit='batch') as tepoch:
        #use "tqdm" to show the progressbar
        for inputs, targets in tepoch:
            if epoch is not None:
                tepoch.set_description(f'Epoch {epoch}')

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, targets)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_train.update(loss.item(), weight=len(targets))
            metric.update(outputs, targets)
            
            tepoch.set_postfix(loss=loss_train.compute().item(),
                               metric=metric.compute().item())

    return model, loss_train.compute().item(), metric.compute().item()



def evaluate(model, test_loader, loss_fn, metric):
    model.eval()
    loss_eval = MeanMetric()
    metric.reset()
    
    with torch.inference_mode():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            loss_eval.update(loss.item(), weight=len(targets))
            
            metric(outputs, targets)


    return loss_eval.compute().item(), metric.compute().item()



def segment(image, model):
    with torch.inference_mode():
        prediction = model(image)
    return torch.sigmoid(prediction).cpu()


class MedicalDatasetCreator(Dataset):
    
    def __init__(self, phase, transforms):
        
        self.transforms = transforms
        self.phase = phase
        self.df = self.create_dataset(self.phase)
        self.classes = ['large_bowel', 'small_bowel', 'stomach']
     
    
    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        image = Image.open(sample['path'])
        mask = self.rle_decode(image.size[::-1], sample[self.classes])
        
        #use tv_tensors for augment picture with their masks correctly
        image = tv_tensors.Image(image)  
        mask = tv_tensors.Mask(mask)

        image, mask = self.transforms(image, mask)
        mask = mask.int()
        return image, mask
    
    
    def __len__(self):
        return len(self.df)
    
    
    @classmethod
    def rle_decode(self, im_size, segments):
        mask = torch.zeros(3, im_size[0]*im_size[1], dtype=torch.int32)
        #if image size would be 266*266 then mask size will be 3*(266*266)
        
        for i, segment in enumerate(segments):
            if str(segment)!='nan':
                #change segments format to visible and trainable format  "start length-->start:end" like "1 4 --> 1 2 3 4"
                segment = segment.split()
                starts = np.array(segment[::2], dtype=np.int32) - 1     #starts are starting from 1 so we have to do "- 1"
                ends = starts + np.array(segment[1::2], dtype=np.int32)
                for start, end in zip(starts, ends):   #dim 0 has got 3 array ,one array for each class
                    mask[i, start:end] = 1
        return mask.reshape((3, im_size[0], im_size[1]))
    
    
    @classmethod
    def create_dataset(self, phase):
        ids = []
        large_bowels = []
        small_bowels = []
        stomachs =[]
        path = []
        
        if phase=='train':
            pathes = train_imgs
        elif phase=='valid':
            pathes = valid_imgs
        elif phase=='test':
            pathes = test_imgs

        for idx in range(len(pathes)):  #make dataset for each phase
            name = '{}_{}_{}'.format(pathes[idx].split('/')[-3],
                                    pathes[idx].split('/')[-1].split('_')[0],
                                    pathes[idx].split('/')[-1].split('_')[1]
                                )
            if len(pivot[pivot['id']==name])!=0:
                path.append(pathes[idx])
                ids.append(pivot[pivot['id']==name]['id'].values[0])
                large_bowels.append(pivot[pivot['id']==name]['large_bowel'].values[0])
                small_bowels.append(pivot[pivot['id']==name]['small_bowel'].values[0])
                stomachs.append(pivot[pivot['id']==name]['stomach'].values[0])

        dataset = pd.DataFrame({'id':ids, 'path':path, 'large_bowel':large_bowels, 'small_bowel':small_bowels, 'stomach':stomachs})
        return dataset


seed = 8

wandb_enable = False

backbone = 'efficientnet-b1'

num_classes = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'


df = pd.read_csv('/kaggle/input/uw-madison-gi-tract-image-segmentation/train.csv')
df.head()


train_cases = ['case129','case2','case131','case134','case6','case135','case9','case7','case139',
               'case140','case11','case142','case136','case141','case145','case18','case19','case148',
               'case149','case22','case146','case24','case144','case154','case156','case29','case30',
               'case32','case34','case36','case40','case41','case43','case44','case47','case49','case53',
               'case54','case55','case138','case63','case65','case66','case80','case81','case88','case90',
               'case91','case101','case102','case107','case108','case111','case113','case114','case115',
               'case116','case117','case118','case121','case125']
valid_cases = ['case33','case130','case133','case122','case16','case84','case20','case58','case92']
test_cases = ['case67','case35','case42','case74','case77','case110','case15','case143','case78',
              'case147','case85','case119','case89','case123','case124']

train_imgs, valid_imgs, test_imgs = [], [], []

for dirname, _, filenames in os.walk('/kaggle/input/uw-madison-gi-tract-image-segmentation/train'):
    for filename in filenames:
        if dirname.split('/')[5] in train_cases:
            train_imgs.append(os.path.join(dirname, filename))
        elif dirname.split('/')[5] in valid_cases:
            valid_imgs.append(os.path.join(dirname, filename))
        elif dirname.split('/')[5] in test_cases:
            test_imgs.append(os.path.join(dirname, filename))



idx = 77
image = Image.open(train_imgs[idx])
im_size = image.size[::-1]
image = TF.to_tensor(image)
image = normalizer(image[0])


name = '{}_{}_{}'.format(train_imgs[idx].split('/')[-3],
                        train_imgs[idx].split('/')[-1].split('_')[0],
                        train_imgs[idx].split('/')[-1].split('_')[1]
                        )
related_df = df[df['id']==name]

segments = related_df['segmentation'].values
mask = torch.zeros(3, im_size[0]*im_size[1], dtype=torch.int32)

for i, segment in enumerate(segments):
    if str(segment)!='nan':
        segment = segment.split()
        starts = np.array(segment[::2], dtype=np.int32) - 1
        ends = starts + np.array(segment[1::2], dtype=np.int32)
        for start, end in zip(starts, ends):
            mask[i, start:end] = 1
mask = mask.reshape((3, im_size[0], im_size[1]))


#sort each sample in one line and delete the all NaN samples
pivot = df.pivot_table(index='id', columns='class', values='segmentation', aggfunc='first').reset_index()
pivot


test_ids = [] #Images with no segment should be out final test ids
for i in df['id'].unique():
    if i not in pivot['id'].unique():
        test_ids.append(i)


len(test_ids)


train_transform = v2.Compose([
    v2.Resize(size=(234,), antialias=True),
    v2.RandomCrop(size=(224, 224)),
    v2.RandomPhotometricDistort(p=0.5),
    v2.RandomHorizontalFlip(p=0.5),
    v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
    v2.Normalize(mean=(0.5,), std=(0.5,)),
    v2.Lambda(lambda x: x.repeat(3, 1, 1))
])
eval_transform = v2.Compose([
    v2.Resize(size=(224, 224), antialias=True),
    v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
    v2.Normalize(mean=(0.5,), std=(0.5,)),
    v2.Lambda(lambda x: x.repeat(3, 1, 1))
])


train = MedicalDatasetCreator('train' ,train_transform)
valid = MedicalDatasetCreator('valid' ,eval_transform)
test = MedicalDatasetCreator('test' ,eval_transform)


image, mask = valid[250]
segmented_image_show(image[0, :, :], mask)


import multiprocessing
print(multiprocessing.cpu_count())  # Gives the number of CPU cores


set_seed(seed)
train_loader = DataLoader(train, 64, True, num_workers=4)
valid_loader = DataLoader(valid, 64, False, num_workers=4)
test_loader = DataLoader(test, 64, False, num_workers=4)


import segmentation_models_pytorch as smp
import torchmetrics as tm
from scipy.spatial.distance import directed_hausdorff

# Define loss and metrics
loss_fn = smp.losses.DiceLoss(mode='multilabel')
metric = tm.classification.Dice(num_classes=3, average='macro').to(device)  # assuming 3 classes
hausdorff = directed_hausdorff


UNet = smp.Unet(
    encoder_name=backbone,         # e.g., 'resnet34'
    encoder_weights='imagenet',
    in_channels=3,
    classes=3
).to(device)

UNetPlusPlus = smp.UnetPlusPlus(
    encoder_name=backbone,
    encoder_weights='imagenet',
    in_channels=3,
    classes=3
).to(device)

DeeplabV3 = smp.DeepLabV3(
    encoder_name=backbone,
    encoder_weights='imagenet',
    in_channels=3,
    classes=3
).to(device)


model3 = smp.DeepLabV3(encoder_name=backbone, encoder_weights='imagenet',
               in_channels=3, classes=3).to(device)

inputs, targets = next(iter(train_loader))
inputs = inputs.to(device)
targets = targets.to(device)

with torch.no_grad():
  outputs = model3(inputs)
  loss = loss_fn(outputs, targets)

print(loss)


mini_set, _ = random_split(valid, (500, len(valid)-500))
mini_loader = DataLoader(mini_set, 20, shuffle=True)


model3 = smp.DeepLabV3(encoder_name=backbone, encoder_weights='imagenet',
               in_channels=3, classes=3).to(device)
optimizer = torch.optim.SGD(model3.parameters(), lr=0.1, momentum=0.9)


num_epochs = 10
for epoch in range(num_epochs):
    model3, _, _ = train_one_epoch(model3, mini_loader, loss_fn, optimizer, metric, epoch+1)



device = 'cuda:1'
torch.cuda.empty_cache()


valid_loader = DataLoader(valid, batch_size=16, shuffle=True)


num_epochs = 3
metric = tm.Dice().to(device)

for lr in [0.5, 0.3, 0.1]:
    print(f'LR={lr}')

    model3 = smp.DeepLabV3(encoder_name=backbone, encoder_weights='imagenet',
               in_channels=3, classes=3).to(device)
    optimizer = optim.SGD(model3.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)

    for epoch in range(num_epochs):
        model3, _, _ = train_one_epoch(model3, valid_loader, loss_fn, optimizer, metric, epoch+1)

    print()


device = 'cuda:0'


set_seed(seed)
train_loader = DataLoader(train, 8, True, num_workers=4)
valid_loader = DataLoader(valid, 16, False, num_workers=4)


torch.cuda.empty_cache()


model3 = smp.DeepLabV3(encoder_name=backbone, encoder_weights='imagenet',
               in_channels=3, classes=3).to(device)


set_seed(seed)
lr = 0.3
wd = 1e-4

optimizer = optim.SGD(model3.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 7, 10], gamma=0.1)
metric = tm.Dice().to(device)


loss_train_hist = []
loss_valid_hist = []
metric_train_hist = []
metric_valid_hist = []

best_loss_valid = torch.inf
epoch_counter = 0


num_epochs = 10

for epoch in range(num_epochs):
  # Train
    model3, loss_train, metric_train = train_one_epoch(model3,
                                     train_loader,
                                     loss_fn,
                                     optimizer,
                                     metric,
                                     epoch+1)
    # Validation
    loss_valid, metric_valid = evaluate(model3,
                         valid_loader,
                         loss_fn,
                         metric)

    loss_train_hist.append(loss_train)
    loss_valid_hist.append(loss_valid)

    metric_train_hist.append(metric_train)
    metric_valid_hist.append(metric_valid)
    
    if loss_valid < best_loss_valid:
        torch.save(model3, f'/kaggle/working/deeplabModel.h5')
        best_loss_valid = loss_valid
        print('Model Saved!')

    print(f'Valid: Loss = {loss_valid:.4}, Metric = {metric_valid:.4}, LR = {lr_scheduler.get_last_lr()[0]}')
    print()
    
    lr_scheduler.step()

    epoch_counter += 1


plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(range(epoch_counter), loss_train_hist, 'r-', label='Train')
plt.plot(range(epoch_counter), loss_valid_hist, 'b-', label='Validation')
plt.title('Loss plot')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(epoch_counter), metric_train_hist, 'r-', label='Train')
plt.plot(range(epoch_counter), metric_valid_hist, 'b-', label='Validation')
plt.title('metric plot')
plt.xlabel('Epoch')
plt.ylabel('metric')
plt.grid(True)
plt.legend()


import os
from IPython.display import FileLink 

FileLink(r'deeplabModel.h5')


model3 = torch.load('/kaggle/working/deeplabModel.h5').to(device)
model3.eval()

loss_test, metric_test = evaluate(model3, test_loader, loss_fn, metric)
print(f'loss test = {loss_test} , metric test = {metric_test}')


model1 = torch.load('/kaggle/input/unet/pytorch/default/1/unetModel.h5').to(device);
model2 = torch.load('/kaggle/input/unetplus/pytorch/default/1/unetModel.h5').to(device);
model3 = torch.load('/kaggle/input/deeplabv3/pytorch/default/1/deeplabModel.h5').to(device);

test_loader = DataLoader(test, 1, True)


def segment(image, model):
    with torch.inference_mode():
        prediction = model(image)
    return torch.sigmoid(prediction).cpu()


img, mask = next(iter(test_loader))
output1 = segment(img.to(device), model1)
output2 = segment(img.to(device), model2)
output3 = segment(img.to(device), model3)

print('                             â™¦â™¦model : UNetâ™¦â™¦')
model_output_show(img[0, 0], mask[0], output1[0])
print('                             â™¦â™¦model : UNet++â™¦â™¦')
model_output_show(img[0, 0], mask[0], output2[0])
print('                             â™¦â™¦model : DeeplabV3â™¦â™¦')
model_output_show(img[0, 0], mask[0], output3[0])


img, mask = next(iter(test_loader))
output1 = segment(img.to(device), model1)
output2 = segment(img.to(device), model2)
output3 = segment(img.to(device), model3)

print('                             â™¦â™¦model : UNetâ™¦â™¦')
model_output_show(img[0, 0], mask[0], output1[0])
print('                             â™¦â™¦model : UNet++â™¦â™¦')
model_output_show(img[0, 0], mask[0], output2[0])
print('                             â™¦â™¦model : DeeplabV3â™¦â™¦')
model_output_show(img[0, 0], mask[0], output3[0])


path = []
sorted_test_ids = []
for dirname, _, filenames in os.walk('/kaggle/input/uw-madison-gi-tract-image-segmentation/train'):
    for filename in filenames:
        id = f'{dirname.split("/")[-2]}_slice_{filename.split("_")[1]}'
        if id in test_ids:
            path.append(os.path.join(dirname, filename))
            sorted_test_ids.append(id)


classes = ['large_bowel', 'small_bowel', 'stomach']

data = []
for id, pat in zip(sorted_test_ids, path):
    for class_name in classes:
        data.append([id, pat, class_name])
test_df = pd.DataFrame(data, columns=['id', 'path', 'class'])

test_df


image = Image.open(path[6])
image = TF.to_tensor(image)
image = eval_transform(image)
image = image.unsqueeze(0)
# plt.imshow(image, cmap='gray')


def segment(image, model):
    with torch.inference_mode():
        prediction = model(image)
    return torch.sigmoid(prediction).cpu()


output = segment(image.to(device), model1)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(image[0, 0], cmap='gray');
plt.title('Input')

plt.subplot(1, 3, 2)
plt.imshow(image[0, 0], cmap='gray');
plt.imshow(output[0].permute(1, 2, 0), alpha=0.4);
plt.title('Unet Model')







