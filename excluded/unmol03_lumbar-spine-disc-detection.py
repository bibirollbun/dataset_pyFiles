!pip install -q git+https://github.com/qubvel/segmentation_models.pytorch


import pandas as pd
from PIL import Image, ImageFilter
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import os
import copy
import pydicom 
import matplotlib.pyplot as plt
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torchvision
from fastai.vision.all import *
import math
import glob
import gc


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

H = W = 512
LEVELS = ["L1L2", "L2L3", "L3L4", "L4L5", "L5S1"]
K = 5
SIGMA = 5 # std deviation - spread of the heatmap keypoints
ANGLE = 30
LR = 5e-4
SIGMA = torch.as_tensor(SIGMA)
A = -1/(2*(SIGMA**2)).to(DEVICE)

EPOCHS = 2
WEIGHT_DECAY = 1e-4
TH = 0.5
S2 = 64
S2 = torch.as_tensor(S2)
BATCH_SIZE = 16

RSNA_TRAIN_IMGS = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images'
RSNA_TEST_IMGS = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images'

torch_resize = torchvision.transforms.Resize((H,W),antialias=True)

x_map = torch.stack([torch.arange(W)]*H).float()
y_map = torch.stack([torch.arange(H)]*W).float()
IDX_MAP = torch.stack([x_map,y_map.T]).view(1,2,H,W).to(DEVICE) # will be used to create the ground truth heatmaps


### RSNA DATASET ####

rsna = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv").dropna()
fn = rsna[rsna['condition'].isin(['Left Neural Foraminal Narrowing', 'Right Neural Foraminal Narrowing'])].drop(columns=['condition']).sort_values(['study_id','series_id','level']).reset_index(drop=True)
print(f"Number of studies with unlabeled foraminal discs keypoints : {(fn.series_id.value_counts() < 10).sum()}\n")


coordinates = {}
for i in range(len(fn)):
    row = fn.iloc[i]
    coordinates[row['study_id']] = {}
for i in range(len(fn)):
    row = fn.iloc[i]
    coordinates[row['study_id']][row['series_id']] = {}

for i in range(len(fn)):
    row = fn.iloc[i]
    coordinates[row['study_id']][row['series_id']][row['instance_number']] = {
        'L1/L2':{
            'x':torch.nan,
            'y':torch.nan
        },
        'L2/L3':{
            'x':torch.nan,
            'y':torch.nan
        },
        'L3/L4':{
            'x':torch.nan,
            'y':torch.nan
        },
        'L4/L5':{
            'x':torch.nan,
            'y':torch.nan
        },
        'L5/S1':{
            'x':torch.nan,
            'y':torch.nan
        }
    }

for i in range(len(fn)):
    row = fn.iloc[i]
    coordinates[row['study_id']][row['series_id']][row['instance_number']][row['level']]['x'] = row['x']
    coordinates[row['study_id']][row['series_id']][row['instance_number']][row['level']]['y'] = row['y']


## Create a dataframe where each row has a unique combination of study_id, series_id & instance_number
fn = fn[['study_id', 'series_id', 'instance_number']].groupby(['study_id', 'series_id', 'instance_number']).count().reset_index()

## We will only have a subset of keypoints for any instance numbers, rest will be torch.nan --> Have to impute
v = np.zeros((len(fn),10))
from tqdm import tqdm
for i in tqdm(range(len(fn))):
    row = fn.iloc[i]
    k = 0
    for level in coordinates[row['study_id']][row['series_id']][row['instance_number']]:
        v[i,k:k+2] = list(coordinates[row['study_id']][row['series_id']][row['instance_number']][level].values())
        k += 2

coor = ['x_L1L2','y_L1L2','x_L2L3','y_L2L3','x_L3L4','y_L3L4','x_L4L5','y_L4L5','x_L5S1','y_L5S1']
fn[coor] = v

print(f"% of rows with NaN : {(fn.isna().any(axis = 1).sum() / len(fn))*100}")


## For every combination of study_id + series_id, augment the data for instance_numbers not present in the current dataframe
### eg : instance_numbers for some (study_id, series_id) -> [3, 5, 8, 12]; missing ones are 4, 6, 7, 9, 10, 11

rows_to_add = []

for (study_id, series_id), df in tqdm(fn.groupby(['study_id', 'series_id'])):
    existing_instances = sorted(df['instance_number'].unique())
    min_inst, max_inst = min(existing_instances), max(existing_instances)
    full_range = list(range(min_inst, max_inst + 1))
    
    # Find missing instance numbers
    missing = sorted(set(full_range) - set(existing_instances))
    
    # Append missing rows (NaN coordinates)
    for inst in missing:
        rows_to_add.append({
            'study_id': int(study_id),
            'series_id': int(series_id),
            'instance_number': inst,
            **{c: torch.nan for c in coor}
        })

# Append to fn
if rows_to_add:
    fn = pd.concat([fn, pd.DataFrame(rows_to_add)], ignore_index=True)

fn[['study_id', 'series_id', 'instance_number']] = fn[['study_id', 'series_id', 'instance_number']].astype(np.int64)
fn = fn.sort_values(['study_id', 'series_id', 'instance_number']).reset_index(drop=True)


for col in coor:
    fn[col] = fn.groupby(['study_id', 'series_id'])[col].transform(
        lambda x: x.fillna(x.mean())
    )

data_T1 = fn
# fn_mean = fn.groupby(['study_id', 'series_id']).mean()
# fn_mean = fn_mean.loc[[(study_id,series_id) for study_id,series_id in fn[['study_id','series_id']].values]][coor].values # Create a fn_mean table with same shape as fn_values
# fn_values = fn[coor].values # To be imputed
# mask = fn[coor].isna() # True if value is NaN
# fn_values[mask] = fn_mean[mask] # imputation step
# fn[coor] = fn_values

# fn['filename'] = (
#     RSNA_TRAIN + '/' 
#     + fn['study_id'].astype(str) + '/' 
#     + fn['series_id'].astype(str) + '/' 
#     + fn['instance_number'].astype(str) + '.dcm'
# )
# fn = fn.drop(columns = ['study_id', 'series_id', 'instance_number'])

# # FInal Dataset
# data_T1 = fn.reset_index(drop=True)


data_T1


def augment_image_and_centers(image, centers, center=(H/2, W/2)):
    angle = torch.as_tensor(random.uniform(-ANGLE, ANGLE)) # -30 to 30 degress -> 0 degree meaning no rotation can also be applied
    image = torchvision.transforms.functional.rotate(
        image, angle.item(),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        center=center
    )

    angle = -angle * math.pi / 180
    s, c = torch.sin(angle), torch.cos(angle)
    rot = torch.stack([
        torch.stack([c, s]),
        torch.stack([-s, c])
    ]).to(centers.device)

    center = torch.as_tensor(center, device=centers.device).float()
    centers = ((centers - center) @ rot) + center

    return image, centers


class Sagittal_T1(Dataset):
    def __init__(self, df, VALID=False, alpha=0):
        self.data = df
        self.VALID = VALID
        self.alpha = alpha

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        centers = torch.as_tensor([x for x in row[coor]]).view(5,2).float()

        sample = f"{RSNA_TRAIN_IMGS}/{int(row['study_id'])}/{int(row['series_id'])}/{int(row['instance_number'])}.dcm"
        image = pydicom.dcmread(sample).pixel_array
        height, width = image.shape

        if height > width:
            d = width
            h = int((height - d)*(.5 + self.alpha*(.5 - np.random.rand()))) if not self.VALID else (height - d)//2
            image = image[h:h+d]
            centers[:,1] -= h
            height = width
        elif height < width:
            d = height
            w = int((width - d)*(.5 + self.alpha*(.5 - np.random.rand()))) if not self.VALID else (width - d)//2
            image = image[:,w:w+d]
            centers[:,0] -= w
            width = height

        image = torch_resize(torch.as_tensor((image / np.max(image)).astype(np.float32)).unsqueeze(0))
        image = image.float().to(DEVICE)

        centers[:,0] = centers[:,0]*W/width
        centers[:,1] = centers[:,1]*H/height

        if not self.VALID:
            image, centers = augment_image_and_centers(image, centers)

        return image, centers


tds = Sagittal_T1(fn)

for k in range(5):
    image,centers = tds.__getitem__(np.random.randint(len(tds)))
    centers = centers[centers.isnan().sum(1) == 0]
    mask = IDX_MAP - centers.view(len(centers),2,1,1).to(DEVICE)
    mask = (mask*mask).sum(1)
    mask = torch.exp(A*mask)
    mask = mask.sum(0)
    plt.imshow(image.cpu()[0] + .5*(mask.cpu() > TH))
    plt.show()
del tds


vds = Sagittal_T1(fn,VALID=True)

for k in range(5):
    image,centers = vds.__getitem__(np.random.randint(len(vds)))
    centers = centers[centers.isnan().sum(1) == 0]
    mask = IDX_MAP - centers.view(len(centers),2,1,1).to(DEVICE)
    mask = (mask*mask).sum(1)
    mask = torch.exp(A*mask)
    mask = mask.sum(0)
    plt.imshow(image.cpu()[0] + .5*(mask.cpu() > TH))
    plt.show()


class Model(nn.Module):
    def __init__(self, classes = K):
        super().__init__()
        self.classes = classes
        self.UNet = smp.Unet(
            encoder_name="resnet50",
            classes=classes,
            in_channels=1
        ).to(DEVICE)

    def forward(self,X):
        height, width = X.shape[-2:]
        x = self.UNet(X.view(-1,1,height,width)).view(-1,height*width)
        # Min-Max scaling of the predicted pixels
        min_values = x.min(-1)[0].view(-1,1)
        max_values = x.max(-1)[0].view(-1,1)
        d = (max_values - min_values)
        d[d == 0] = 1
        x = (x - min_values)/d
        
        return x.view(-1, self.classes, height, width)


class LossFunction(nn.Module):
    def __init__(self,alpha=.5, smooth = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth

    def clone(self):
        return LossFunction(self.alpha)

    def forward(self, heatmaps, centers):
        h, w = heatmaps.shape[-2:]
        heatmaps = heatmaps.view(-1,h*w)
        centers = centers.view(-1,2)
        m = centers.isnan().sum(1) == 0
        heatmaps = heatmaps[m]
        centers = centers[m]

        # Create ground truth heatmaps using the centers
        mask = IDX_MAP - centers.view(len(centers),2,1,1).to(DEVICE)
        mask = (mask*mask).sum(1)
        mask = torch.exp(A*mask)
        mask = mask.view(-1,h*w)

        # Loss formula
        D = 1 - ((mask*heatmaps).sum(-1))**2/((mask*mask).sum(-1)*(heatmaps*heatmaps).sum(-1)+self.smooth)
        return D.mean()


def nt(nmin,nmax,tcur,tmax):
    return (nmax - .5*(nmax-nmin)*(1+np.cos(tcur*np.pi/tmax))).astype(np.float32)

def cb(self):
    alpha = torch.as_tensor(nt(.25,1,learn.train_iter,EPOCHS*n_iter))
    learn.dls.train_ds.alpha = alpha
alpha_cb = Callback(before_batch=cb)


from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=2003)

data_T1['fold'] = -1
for fold, (_, val_idx) in enumerate(kf.split(data_T1)):
    data_T1.loc[val_idx, 'fold'] = fold


FOLDS = [0, 1, 2, 3, 4]

for f in FOLDS:
    model = Model()
    tdf = data_T1[data_T1['fold'] != f]
    vdf = data_T1[data_T1['fold'] == f]

    tds = Sagittal_T1(tdf)
    vds = Sagittal_T1(vdf, VALID = True)

    tdl = DataLoader(tds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    vdl = DataLoader(vds, batch_size=BATCH_SIZE, shuffle=False)

    dataloaders = DataLoaders(tdl, vdl)

    n_iter = len(tds) // BATCH_SIZE
    learn = Learner(
        dataloaders,
        model, 
        lr = LR, 
        loss_func = LossFunction(alpha = 0.5),
        cbs=[
            ShowGraphCallback(),
            alpha_cb
        ]
    )
    learn.fit_one_cycle(2) # 2 epochs
    torch.save(model, 'keypoint_detector_sagittal_T1_'+str(f))
    del tdl, vdl, dataloaders, model, learn
    gc.collect()


!pip install -q git+https://github.com/qubvel/segmentation_models.pytorch


import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import os
import glob
from tqdm import tqdm
import gc

import torchvision
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from fastai.vision.all import *
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SEED = 1337
FOLDS = [1,2,3,4,5]
TRAIN_PATH = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'
ENCODER_NAME = "resnet18"
PATCH_H = 512
PATCH_W = 512
ANGLE = 30
S2 = 64
BS = 16
LR = 1e-4
EPOCHS = 1
TH = .5

S2 = torch.as_tensor(S2)
A = -1/(2*S2).to(device)


df_coor = pd.read_csv(TRAIN_PATH + 'train_label_coordinates.csv')
df_coor.sample(10)


S = df_coor[
    df_coor['condition'] == 'Spinal Canal Stenosis'
].sort_values([
    'study_id',
    'series_id',
    'level'
]).reset_index(drop=True)
S.tail()


S['x_mean_fraction'] = S['x']/S.groupby(['study_id','series_id'])['x'].mean().loc[[(study_id,series_id) for study_id,series_id in S[['study_id','series_id']].values]].values
S.tail()


plt.boxplot(S['x_mean_fraction'])


S[S['x_mean_fraction'] < .8]


S = S[S['x_mean_fraction'] > .8]


coordinates = {}
for i in range(len(S)):
    row = S.iloc[i]
    coordinates[row['study_id']] = {}
for i in range(len(S)):
    row = S.iloc[i]
    coordinates[row['study_id']][row['series_id']] = {}
for i in range(len(S)):
    row = S.iloc[i]
    coordinates[row['study_id']][row['series_id']][row['instance_number']] = {
        'L1/L2':{
            'x':torch.nan,
            'y':torch.nan
        },
        'L2/L3':{
            'x':torch.nan,
            'y':torch.nan
        },
        'L3/L4':{
            'x':torch.nan,
            'y':torch.nan
        },
        'L4/L5':{
            'x':torch.nan,
            'y':torch.nan
        },
        'L5/S1':{
            'x':torch.nan,
            'y':torch.nan
        }
    }
for i in range(len(S)):
    row = S.iloc[i]
    coordinates[row['study_id']][row['series_id']][row['instance_number']][row['level']]['x'] = row['x']
    coordinates[row['study_id']][row['series_id']][row['instance_number']][row['level']]['y'] = row['y']


S =  S[[
    'study_id',
    'series_id',
    'instance_number'
]].groupby([
    'study_id',
    'series_id',
    'instance_number'
]).count().reset_index()
S.tail()


v = np.zeros((len(S),10))
for i in tqdm(range(len(S))):
    row = S.iloc[i]
    k = 0
    for level in coordinates[row['study_id']][row['series_id']][row['instance_number']]:
        v[i,k:k+2] = list(coordinates[row['study_id']][row['series_id']][row['instance_number']][level].values())
        k += 2


coor = [
    'x_L1L2',
    'y_L1L2',
    'x_L2L3',
    'y_L2L3',
    'x_L3L4',
    'y_L3L4',
    'x_L4L5',
    'y_L4L5',
    'x_L5S1',
    'y_L5S1'    
]


S[coor] = v
S.tail()


for (study_id,series_id),df in tqdm(S.groupby(['study_id','series_id'])):
    sample = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/" + str(study_id) + '/' + str(series_id)
    instance_numbers = [int(x.replace('\\','/').split('/')[-1].replace('.dcm','')) for x in glob.glob(sample+'/*.dcm')]
    instance_numbers.sort()
    instance_numbers = np.array(instance_numbers)
    D = len(instance_numbers)
    L = D//3
    FIRST = int(np.arange(D)[instance_numbers == df['instance_number'].min()])
    LAST = int(np.arange(D)[instance_numbers == df['instance_number'].max()])
    M = (FIRST + LAST)//2
    START = max([0,M - L//2])
    END = min([D,M+L-L//2+1])
    new = instance_numbers[START:END].tolist()
    if FIRST > 0: new.append(instance_numbers[FIRST - 1])
    if FIRST > 1: new.append(instance_numbers[FIRST - 2])
    if LAST < D - 1: new.append(instance_numbers[LAST + 1])
    if LAST < D - 2: new.append(instance_numbers[LAST + 2])
    L = len(new)
    S = pd.concat([
            S,
            pd.DataFrame({
                'study_id':[int(study_id)]*L,
                'series_id':[int(series_id)]*L,
                'instance_number':new,
                'x_L1L2':[torch.nan]*L,
                'y_L1L2':[torch.nan]*L,
                'x_L2L3':[torch.nan]*L,
                'y_L2L3':[torch.nan]*L,
                'x_L3L4':[torch.nan]*L,
                'y_L3L4':[torch.nan]*L,
                'x_L4L5':[torch.nan]*L,
                'y_L4L5':[torch.nan]*L,
                'x_L5S1':[torch.nan]*L,
                'y_L5S1':[torch.nan]*L
            })
        ])
    


S = S.reset_index(drop=True)
S[['study_id','series_id','instance_number']] = S[['study_id','series_id','instance_number']].astype(np.int64)
S.tail()


S_mean = S.groupby(['study_id','series_id']).mean()
S_mean.tail()


S_mean = S_mean.loc[[(study_id,series_id) for study_id,series_id in S[['study_id','series_id']].values]][coor].values
S_mean.shape


S_values = S[coor].values
S_values.shape


mask = S[coor].isna()
mask.shape


## This only imputes the NaN values determined by the mask, with the mean

S_values[mask] = S_mean[mask]
S[coor] = S_values
S.tail(10)


S[S.isna().sum(1) > 0].reset_index(drop=True).tail()


df_meta_f = pd.read_csv(TRAIN_PATH + 'train_series_descriptions.csv')
df_meta_f.tail()


S = S.merge(df_meta_f[['series_id','series_description']], left_on='series_id', right_on='series_id')
S.tail()


S.groupby('series_description').count()


S = S[S.series_description == 'Sagittal T2/STIR'].reset_index(drop=True)
S.groupby('series_description').count()


## For the below code, we need a precomputed fold data - train.csv which contains the final classifications
# S = S.merge(train[['study_id','fold']],left_on='study_id',right_on='study_id')
# S.tail()

S['fold'] = np.random.randint(1, 6, size=len(S))


S.groupby('fold').count()


def augment_image_and_centers(image,centers,center=(PATCH_H/2,PATCH_W/2)):
    angle = torch.as_tensor(random.uniform(-ANGLE, ANGLE))
    image = torchvision.transforms.functional.rotate(
        image,angle.item(),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        center=center
    )
    angle = -angle*math.pi/180
    s = torch.sin(angle)
    c = torch.cos(angle)
    rot = torch.stack([
        torch.stack([c, s]),
        torch.stack([-s, c])
    ])
    center = torch.as_tensor(center).float()
    centers = ((centers.cpu() - center) @ rot) + center

    return image,centers
torch_resize = torchvision.transforms.Resize((PATCH_H,PATCH_W),antialias=True)

x_map = torch.stack([torch.arange(PATCH_W)]*PATCH_H).float()
y_map = torch.stack([torch.arange(PATCH_H)]*PATCH_W).float()
idx_map = torch.stack([x_map,y_map.T]).view(1,2,PATCH_H,PATCH_W).to(device)


class Sagittal_T2_Dataset(Dataset):
    def __init__(self, df, VALID=False, alpha=0):
        self.data = df
        self.VALID = VALID
        self.alpha = alpha

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        centers = torch.as_tensor([x for x in row[coor]]).view(5,2).float()
        
        sample = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/" + str(row['study_id']) + '/' + str(row['series_id']) + '/'+str(row['instance_number']) + '.dcm'
        
        image = pydicom.dcmread(sample).pixel_array
        H,W = image.shape
        # Perform image crops
        if H > W:
            d = W
            if not self.VALID:
                h = int((H - d)*(.5 + self.alpha*(.5 - np.random.rand())))
            else:
                h = (H - d)//2
            image = image[h:h+d]
            centers[:,1] -= h
            H = W
        elif H < W:
            d = H
            if not self.VALID:
                w = int((W - d)*(.5 + self.alpha*(.5 - np.random.rand())))
            else:
                w = (W - d)//2
            image = image[:,w:w+d]
            centers[:,0] -= w
            W = H
        image = torch_resize(torch.as_tensor((image/np.max(image)).astype(np.float32)).unsqueeze(0))
        image = image.float().to(device)
        
        centers[:,0] = centers[:,0]*PATCH_W/W
        centers[:,1] = centers[:,1]*PATCH_H/H

        if not self.VALID: image,centers = augment_image_and_centers(image,centers)

        return image,centers


tds = Sagittal_T2_Dataset(S)
for k in range(5):
    image,centers = tds.__getitem__(np.random.randint(len(tds)))
    centers = centers[centers.isnan().sum(1) == 0]
    mask = idx_map - centers.view(len(centers),2,1,1).to(device)
    mask = (mask*mask).sum(1)
    mask = torch.exp(A*mask)
    mask = mask.sum(0)
    plt.imshow(image.cpu()[0] + .5*(mask.cpu() > TH))
    plt.show()


del tds
gc.collect()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class myUNet(nn.Module):
    def __init__(self,classes):
        super(myUNet, self).__init__()
        self.classes = classes
        self.UNet = smp.Unet(
            encoder_name=ENCODER_NAME,
            classes=classes,
            in_channels=1
        ).to(device)

    def forward(self,X):
        H,W = X.shape[-2:]
        x = self.UNet(X.view(-1,1,H,W)).view(-1,H*W)
        # Min-Max normalization
        min_values = x.min(-1)[0].view(-1,1)
        max_values = x.max(-1)[0].view(-1,1)
        d = (max_values - min_values)
        d[d == 0] = 1
        x = (x - min_values)/d
        
        return x.view(-1,self.classes,H,W)


class myLoss(nn.Module):
    def __init__(
            self,
            alpha=.5,
            smooth = 1e-6
        ):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth

    def clone(self):
        return myLoss(self.alpha)

    def forward(
            self,
            heatmaps,# Predictions
            centers # Targets
        ):
        H,W = heatmaps.shape[-2:]
        heatmaps = heatmaps.view(-1,H*W)
        centers = centers.view(-1,2)
        m = centers.isnan().sum(1) == 0
        heatmaps = heatmaps[m]
        centers = centers[m]
#       Ideal heatmaps
        mask = idx_map - centers.view(len(centers),2,1,1).to(device)
        mask = (mask*mask).sum(1)
        mask = torch.exp(A*mask)
        mask = mask.view(-1,H*W)
#       Distance
        D = 1 - ((mask*heatmaps).sum(-1))**2/((mask*mask).sum(-1)*(heatmaps*heatmaps).sum(-1)+self.smooth)
        return D.mean()


# CosineAnnealingAlpha
def nt(nmin,nmax,tcur,tmax):
    return (nmax - .5*(nmax-nmin)*(1+np.cos(tcur*np.pi/tmax))).astype(np.float32)

def cb(self):
    alpha = torch.as_tensor(nt(.25,1,learn.train_iter,EPOCHS*n_iter))
    learn.dls.train_ds.alpha = alpha
alpha_cb = Callback(before_batch=cb)


for f in FOLDS:
    seed_everything(SEED)
    model = myUNet(5)
    # model = torch.load(PATH + 'Sagittal_T1/level_segmentation/Sagittal_T1_sagittal_level_segmentation_'+str(f))
    
    tdf = S[S['fold'] != f]
    vdf = S[S['fold'] == f]

    tds = Sagittal_T2_Dataset(tdf)
    vds = Sagittal_T2_Dataset(vdf,VALID=True)
    
    tdl = torch.utils.data.DataLoader(tds, batch_size=BS, shuffle=True, drop_last=True)
    vdl = torch.utils.data.DataLoader(vds, batch_size=BS, shuffle=False)

    dls = DataLoaders(tdl,vdl)

    n_iter = len(tds)//BS

    learn = Learner(
        dls,
        model,
        lr=LR,
        loss_func=myLoss(alpha=0.5),
        cbs=[
            ShowGraphCallback(),
            alpha_cb
        ]
    )
    learn.fit_one_cycle(3)
    torch.save(model,'Sagittal_T2_sagittal_level_segmentation_'+str(f))
    del tdl,vdl,dls,model,learn
    gc.collect()


model = torch.load("/kaggle/working/Sagittal_T2_sagittal_level_segmentation_2", map_location='cuda', weights_only = False)


model.eval()
import random

idx = random.randint(0, len(vds) - 1)
image, gt_centers = vds[idx]

with torch.no_grad():
    pred_heatmaps = model(image.unsqueeze(0))


def heatmaps_to_keypoints(heatmaps):
    """
    heatmaps: Tensor [1, C, H, W]
    returns: list of (x, y)
    """
    heatmaps = heatmaps[0].cpu().numpy()
    coords = []

    for c in range(heatmaps.shape[0]):
        y, x = np.unravel_index(
            np.argmax(heatmaps[c]),
            heatmaps[c].shape
        )
        coords.append((x, y))

    return coords

pred_centers = heatmaps_to_keypoints(pred_heatmaps)


import matplotlib.pyplot as plt

img = image.cpu()[0]  # [H, W]

plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')
plt.axis('off')

# Ground truth (green)
gt = gt_centers[gt_centers.isnan().sum(1) == 0]
for (x, y) in gt:
    plt.scatter(x, y, c='lime', s=40, label='GT')

# Prediction (red)
for (x, y) in pred_centers:
    plt.scatter(x, y, c='red', s=40, label='Pred')

plt.title("Sagittal T2 – GT (green) vs Prediction (red)")
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import os
import glob
from tqdm import tqdm
import gc

import torchvision
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from fastai.vision.all import *
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SEED = 1337
FOLDS = [1,2,3,4,5]
PATH = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'
TRAIN_PATH = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/'
ENCODER_NAME = "resnet18"
ANGLE = 180
S2 = 64
PATCH_H = 512
PATCH_W = 512
BS = 24
LR = 2.5e-4
EPOCHS = 1
TH = .5

S2 = torch.as_tensor(S2)
A = -1/(2*S2).to(device)


df_coor = pd.read_csv(PATH + 'train_label_coordinates.csv')
df_coor.tail()


S = df_coor[
    df_coor['condition'].isin([
        'Left Subarticular Stenosis',
        'Right Subarticular Stenosis'
    ])
].sort_values([
    'study_id',
    'series_id',
    'level',
    'condition'
]).reset_index(drop=True)
S.tail()


centers = {}
for i in range(len(S)):
    row = S.iloc[i]
    centers[row['study_id']]={}
for i in range(len(S)):
    row = S.iloc[i]
    centers[row['study_id']][row['series_id']]={}
for i in range(len(S)):
    row = S.iloc[i]
    centers[row['study_id']][row['series_id']][row['instance_number']]={'L':[], 'R':[], 'level':[]}
for i in range(len(S)):
    row = S.iloc[i]
    centers[row['study_id']][row['series_id']][row['instance_number']][row['condition'][0]].append([row['x'],row['y']])


S = S[[
    'study_id',
    'series_id',
    'instance_number'
]].groupby([
    'study_id',
    'series_id',
    'instance_number'
]).count().reset_index()


coordinates = np.zeros((len(S),4))
coordinates[:] = np.nan
for i in range(len(S)):
    row = S.iloc[i]
    for side in ['L','R']:
            if len(centers[row['study_id']][row['series_id']][row['instance_number']][side]) > 0:
                c = np.array(centers[row['study_id']][row['series_id']][row['instance_number']][side]).mean(0)
                coordinates[
                    i,
                    {'L':0, 'R':2}[side]:{'L':0, 'R':2}[side]+2
                ] = c


coor = [
    'x_L',
    'y_L',
    'x_R',
    'y_R'    
]


S[coor] = coordinates
S.tail()


S[S[coor].isna().sum(1) > 0].reset_index(drop=True).tail()


# I suppose those are centered inside levels and I'll can safely assign neighbor slices to the same level
centroids = S[S[coor].isna().sum(1) == 0].reset_index(drop=True)
centroids.tail()


for (study_id,series_id),df in tqdm(centroids.groupby(['study_id','series_id'])):
    sample = TRAIN_PATH + str(study_id) + '/' + str(series_id)
    instance_numbers = [int(x.replace('\\','/').split('/')[-1].replace('.dcm','')) for x in glob.glob(sample+'/*.dcm')]
    instance_numbers.sort()
    instance_numbers = np.array(instance_numbers)
    D = len(instance_numbers)
    for i in range(len(df)):
        row = df[i:i+1]
        instance_number = row['instance_number'].values
        instance_number_index = int(np.arange(D)[instance_numbers == instance_number])
        if instance_number_index > 0:
            new = row.copy()
            new['instance_number'] = instance_numbers[instance_number_index - 1]
            centroids = pd.concat([
                centroids,
                new
            ])
        if instance_number_index < D - 1:
            new = row.copy()
            new['instance_number'] = instance_numbers[instance_number_index + 1]
            centroids = pd.concat([
                centroids,
                new
            ])


S = pd.concat([S[S[coor].isna().sum(1) > 0],centroids]).groupby(['study_id','series_id','instance_number']).mean().reset_index()
S.tail()


S[S[coor].isna().sum(1) > 0].reset_index(drop=True).tail()


S['flip'] = False
fS = S.copy()
fS['flip'] = True
S = pd.concat([S,fS]).reset_index(drop=True)


S['fold'] = np.random.randint(1, 6, size=len(S))


df_meta_f = pd.read_csv(PATH + 'train_series_descriptions.csv')
df_meta_f.tail()


S = S.merge(df_meta_f[['series_id','series_description']], left_on='series_id', right_on='series_id')
S.tail()


S.groupby('series_description').count()


S.groupby('fold').count()


def augment_image_and_centers(image,centers,center=(PATCH_H/2,PATCH_W/2)):
    # Randomly rotate the image.
    angle = torch.as_tensor(random.uniform(-ANGLE, ANGLE))
    image = torchvision.transforms.functional.rotate(
        image,angle.item(),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        center=center
    )
    angle = -angle*math.pi/180
    s = torch.sin(angle)
    c = torch.cos(angle)
    rot = torch.stack([
        torch.stack([c, s]),
        torch.stack([-s, c])
    ])
    center = torch.as_tensor(center).float()
    centers = ((centers.cpu() - center) @ rot) + center

    return image,centers

torch_resize = torchvision.transforms.Resize((PATCH_H,PATCH_W),antialias=True)

x_map = torch.stack([torch.arange(PATCH_W)]*PATCH_H).float()
y_map = torch.stack([torch.arange(PATCH_H)]*PATCH_W).float()
idx_map = torch.stack([x_map,y_map.T]).view(1,2,PATCH_H,PATCH_W).to(device)


class Axial_T2_axial_side_Dataset(Dataset):
    def __init__(self, df, VALID=False, alpha=0):
        self.data = df
        self.VALID = VALID
        self.alpha = alpha

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        centers = torch.as_tensor([x for x in row[coor]]).view(2,2).float().to(device)
        
        sample = TRAIN_PATH + str(row['study_id']) + '/'+ str(row['series_id']) + '/'+ str(row['instance_number']) + '.dcm'
        
        image = pydicom.dcmread(sample).pixel_array
        H,W = image.shape
        if H > W:
            d = W
            if not self.VALID:
                h = int((H - d)*(.5 + self.alpha*(.5 - np.random.rand())))
            else:
                h = (H - d)//2
            image = image[h:h+d]
            centers[:,1] -= h
            H = W
        elif H < W:
            d = H
            if not self.VALID:
                w = int((W - d)*(.5 + self.alpha*(.5 - np.random.rand())))
            else:
                w = (W - d)//2
            image = image[:,w:w+d]
            centers[:,0] -= w
            W = H
        image = torch_resize(torch.as_tensor((image/np.max(image)).astype(np.float32)).unsqueeze(0))
        image = image.float().to(device)
        
        centers[:,0] = centers[:,0]*PATCH_W/W
        centers[:,1] = centers[:,1]*PATCH_H/H

        if not self.VALID: image,centers = augment_image_and_centers(image,centers,centers.nanmean(0).tolist())

        if row['flip']:
            image = image.flip(2)
            centers = centers.flip(0)
            centers[:,0] = PATCH_W - centers[:,0] - 1

        return image,centers


tds = Axial_T2_axial_side_Dataset(S)
for k in range(10):
    image,centers = tds.__getitem__(np.random.randint(len(tds)))
    centers = centers[centers.isnan().sum(1) == 0]
    mask = idx_map - centers.view(len(centers),2,1,1).to(device)
    mask = (mask*mask).sum(1)
    mask = torch.exp(A*mask)
    mask = mask.sum(0)
    plt.imshow(image.cpu()[0] + .5*(mask.cpu() > TH))
    plt.show()


del tds
gc.collect()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class myUNet(nn.Module):
    def __init__(
        self,
        classes
        ):
        super(myUNet, self).__init__()

        self.classes = classes
        self.UNet = smp.Unet(
            encoder_name=ENCODER_NAME,
            classes=classes,
            in_channels=1
        ).to(device)

    def forward(self,X):
        H,W = X.shape[-2:]
        x = self.UNet(X.view(-1,1,H,W)).view(-1,H*W)
        min_values = x.min(-1)[0].view(-1,1)
        max_values = x.max(-1)[0].view(-1,1)
        d = (max_values - min_values)
        d[d == 0] = 1
        x = (x - min_values)/d
        
        return x.view(-1,self.classes,H,W)


class myLoss(nn.Module):
    def __init__(
            self,
            alpha=.5,
            smooth = 1e-6
        ):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth

    def clone(self):
        return myLoss(self.alpha)

    def forward(
            self,
            heatmaps,# Predictions
            centers # Targets
        ):
        H,W = heatmaps.shape[-2:]
        heatmaps = heatmaps.view(-1,H*W)
        centers = centers.view(-1,2)
        m = centers.isnan().sum(1) == 0
        heatmaps = heatmaps[m]
        centers = centers[m]
        # Ideal heatmaps
        mask = idx_map - centers.view(len(centers),2,1,1).to(device)
        mask = (mask*mask).sum(1)
        mask = torch.exp(A*mask)
        mask = mask.view(-1,H*W)
        # loss
        D = 1 - ((mask*heatmaps).sum(-1))**2/((mask*mask).sum(-1)*(heatmaps*heatmaps).sum(-1)+self.smooth)
        
        return D.mean()


# CosineAnnealingAlpha
def nt(nmin,nmax,tcur,tmax):
    return (nmax - .5*(nmax-nmin)*(1+np.cos(tcur*np.pi/tmax))).astype(np.float32)

# callback to update alpha during training
def cb(self):
    alpha = torch.as_tensor(nt(.25,1,learn.train_iter,EPOCHS*n_iter))
    learn.dls.train_ds.alpha = alpha
alpha_cb = Callback(before_batch=cb)


for f in FOLDS:
    seed_everything(SEED)
    model = myUNet(2)
    
    tdf = S[S['fold'] != f]
    vdf = S[S['fold'] == f]

    tds = Axial_T2_axial_side_Dataset(tdf)
    vds = Axial_T2_axial_side_Dataset(vdf,VALID=True)
    
    tdl = torch.utils.data.DataLoader(tds, batch_size=BS, shuffle=True, drop_last=True)
    vdl = torch.utils.data.DataLoader(vds, batch_size=BS, shuffle=False)

    dls = DataLoaders(tdl,vdl)

    n_iter = len(tds)//BS

    learn = Learner(
        dls,
        model,
        lr=LR,
        loss_func=myLoss(alpha=0.5),
        cbs=[
            ShowGraphCallback(),
            alpha_cb
        ]
    )
    learn.fit_one_cycle(2)
    torch.save(model,'Axial_T2_axial_side_segmentation_'+str(f))
    del tdl,vdl,dls,model,learn
    gc.collect()




