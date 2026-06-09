""" 2D & 2.5D Segmentation Open Source Library"""

# Pytorch Framework Model

### https://github.com/qubvel-org/segmentation_models.pytorch
### https://github.com/huggingface/pytorch-image-models[Timm]

# Tensorflow Framework Model 

### https://github.com/qubvel/segmentation_models

# Other FrameWork

### FaceBook Detectron2 
### Ultratics Yolo


## How to Use Plotly(Scatter, bar ...)

## Remove Noisy with Threshold(Quick Method) => 3. Removal Noisy

## Binary Classification with PytorchLigthning => 4. Binar Cls Model with PytorchLightning

## Using Triplet Attention Module in Classification => 4. Binar Cls Model with PytorchLightning

## How to use SWA(Stochastic Weight Averaging) in PytorchLightning => 4. Binar Cls Model with PytorchLightning

## Custom Augmentation in Collate_fn(ex: CutMix) => 6. Build DataLoaer & Display

## Multi Class Segmentation with Pytorch => 7. Build Multi Class Segmentation Model

## Using CBMA's Channel Attention Module in Segmentation => 7. Build Multi Class Segmentation Model

## How to use amp, swa in Pytorch => 9. Transfer Learning

## Custom Segmenation Loss => 8. Loss & Scheduler

#### 2class(empty, non-empty) filter out  => 11. Post Processing
###### Binary-Gate, Soft Weightning, Soft Weightning with confidence scaling 

#### 2stage model -> 3stage model => 11. Post Processing

#### Discard so small pixel values => 11. Post Processing

#### Find Optimal Threshold per each channel


# [PyPI Version]

## !pip install -q segmentation_models_pytorch

# [Lastest Version from GitHub]

!pip install -q git+https://github.com/qubvel/segmentation_models.pytorch


import os, ctypes
import random
from glob import glob
import shutil
from tqdm import tqdm
import time
import copy
from collections import defaultdict
import gc

import pandas as pd
import numpy as np
import cv2
from PIL import Image
from IPython.display import display, IFrame
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import iplot
from plotly.subplots import make_subplots

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.cuda import amp

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

import tensorflow as tf 

import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from colorama import Fore, Style
c_ = Fore.GREEN
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings('ignore')


import wandb 

try: 
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("WANDB")
    wandb.login(key=api_key)

    anonymous = None
except:
    anonymous = "must"
    print('To use your W&B account, \n Go to Add-ons -> Secrets and provide your W&B access token')


class CFG:
    segment_dir = 'runs/segment'
    cls_dir = 'runs/classify'
    seed = 2025
    debug = False # Full Training
    exp_name = '2.5D' # Stride = 2
    comment = 'unet-timm_efficientnet_b0-160x192-ep=5'
    model_name = 'unet_timm_efficientnet_b0'
    backbone = 'timm-efficientnet-b0'
    cls_model_name = 'tf_efficientnet_b0_ns'
    train_bs = 64
    valid_bs = 128
    img_size = [160, 192] # (Height, Width)
    cls_epochs = 5
    epochs = 20
    lr = 2e-3
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_mult = 1
    warmup_epochs = 0
    wd = 1e-6 
    n_accumulate = 1
    n_fold = 5
    fold = 0
    num_classes = 3 
    class_names = ['large_bowel','small_bowel','stomache']
    load_segment = False
    load_path = None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_count = torch.cuda.device_count()

os.makedirs(CFG.segment_dir, exist_ok=True)
os.makedirs(CFG.cls_dir, exist_ok=True)
print(f"{c_} => Device is {CFG.device}")
print(f"{c_} => Num GPU of machine is ", CFG.device_count)


def clean_memory():
    ctypes.CDLL('libc.so.6').malloc_trim(0)
    gc.collect()


def seed_everything(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    """When running CuDNN backend, two further options must be set"""
    """Pytorch, TensorFlow Framework are both using CuDNN backend"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(SEED)

seed_everything(CFG.seed)


path_df = pd.DataFrame(glob('/kaggle/input/uwmgi-25d-stride2-dataset/images/images/*'), columns=['image_path'])
path_df['mask_path'] = path_df.image_path.str.replace('image','mask')
path_df['id'] = path_df.image_path.apply(lambda x: x.split('/')[-1].replace('.npy',''))

print('Shape of DataFrame: ', path_df.shape)
path_df.head()


df = pd.read_csv('../input/uwmgi-mask-dataset/train.csv')
df['segmentation'] = df.segmentation.fillna('')
df['rle_len'] = df.segmentation.map(len)

df2 = df.groupby(['id'])['segmentation'].agg(list).reset_index()
df2 = df2.merge(df.groupby(['id'])['rle_len'].agg(sum).reset_index(), on='id', how='left')

df = df.drop(columns=['segmentation', 'class', 'rle_len'])
df = df.groupby(['id']).agg('first').reset_index()
df = df.merge(df2, on='id', how='left')
df['empty'] = (df.rle_len==0)

df = df.drop(columns=['image_path', 'mask_path'])
df = df.merge(path_df, on=['id'])


segment_list = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    segment_id = ' '.join(CFG.class_names[i] if row['segmentation'][i] else '' for i in range(CFG.num_classes))
    segment_list.append(segment_id)

df['segment'] = pd.Series(segment_list)
df['segment'] = df['segment'].apply(lambda x: ' '.join(x.strip().split()))

unique_labels = df['segment'].unique()
label_to_id = { label: i for i, label in enumerate(unique_labels)}
id_to_label = { i: label for i, label in enumerate(unique_labels)}

df['segment_id'] = df['segment'].map(label_to_id)

print(f'{c_}#'*25)
for key, value in label_to_id.items():
    print(f'{key} : {value}')

print('#'*25)


legend_elements = [
    Patch(facecolor = 'red', edgecolor = 'r', label='Large Bowel'),
    Patch(facecolor = 'green', edgecolor = 'g', label='Small Bowel'),
    Patch(facecolor = 'blue', edgecolor = 'b', label='Stomache'),
]


def load_img(path):
    img = np.load(path)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    img = (img*255).astype('uint8')    
    return img

def load_msk(path):
    msk = np.load(path)
    msk = msk.astype(np.float32)
    msk /= 255.0
    return msk

def msk2contour(mask, width=3):

    h = mask.shape[0]; w = mask.shape[1]

    mask2 = np.concatenate([mask[:,width:], np.zeros((h,width), dtype=np.uint8)], axis=1)
    mask2 = np.logical_xor(mask,mask2)

    mask3 = np.concatenate([mask[width:,:], np.zeros((width,w), dtype=np.uint8)], axis=0)
    mask3 = np.logical_xor(mask,mask3)

    mask = np.logical_or(mask2, mask3) * 255

    return mask




def rle_decode(mask_rle, shape):

    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zips(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape)

def rle_encode(img):

    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['segment_id'], groups=df['case'])):
    df.loc[val_idx, 'fold'] = fold

fold_df = df.groupby(['fold'])['segment_id',].value_counts().unstack().reset_index()
empty_df = df.groupby(['fold'])['empty'].value_counts().reset_index()
empty_df = empty_df[empty_df['empty'] == True].reset_index(drop=True)
empty_df['empty_ratio'] = empty_df['count'] / np.sum(empty_df['count']) * 100


from plotly import tools

data = []
for i in range(df['segment_id'].nunique()):
    r, g, b = np.random.randint(0, 255, 3)
    
    trace = go.Bar(
        y = fold_df.fold,
        x = fold_df[i],
        name = id_to_label[i],
        marker = dict(color = f'rgba({r},{g},{b},0.2)',
                  line=dict(color=f'rgb({r},{g},{b})', width=2)),
        orientation='h',
        
    )
    data.append(trace)

trace2 = go.Scatter(
    y = empty_df.fold,
    x = empty_df.empty_ratio,
    mode = 'lines',
    line=dict(color='rgb(63,72,204)'),
    name = 'empty_ratio',
    showlegend=True,
)


layout = go.Layout(barmode='group',
                   title=' StratifiedGroupKFold',
                   yaxis=dict(title='Fold Number'),
                   xaxis=dict( title='Count'),
                   legend=dict(title='Segment Labels', bordercolor='Black', borderwidth=1),
                   margin=dict(t=50, b=40, l=70, r=40),
                   height=500,
                   width=1000,
)


fig = tools.make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "scatter"}]], shared_xaxes=False,
                    shared_yaxes=True, vertical_spacing = 0.01)

for trace in data:
    fig.append_trace(trace, row=1, col=1)

fig.append_trace(trace2, row=1, col=2)

fig['layout'].update(layout)

fig.show(renderer="iframe")


def find_contour(img):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)[1]
    #thresh = cv2.erode(thresh, None, iterations=2)
    #thresh = cv2.dilate(thresh, None, iterations=2)

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)

    extreme_point = []
    extLeft = tuple(c[c[:,:,0].argmin()][0]); extreme_point.append(extLeft)
    extRight = tuple(c[c[:,:,0].argmax()][0]); extreme_point.append(extRight)
    extTop = tuple(c[c[:,:,1].argmin()][0]); extreme_point.append(extTop) 
    extBot = tuple(c[c[:,:,1].argmax()][0]); extreme_point.append(extBot)
    
    img_cnt = cv2.drawContours(img.copy(), c, -1, (0,255,255), 3)

    img_pnt = cv2.circle(img_cnt.copy(), extLeft, 5, (0, 0, 255), -1)
    img_pnt = cv2.circle(img_pnt, extRight, 5, (0, 255, 0), -1)
    img_pnt = cv2.circle(img_pnt, extTop, 5, (255, 0, 0), -1)
    img_pnt = cv2.circle(img_pnt, extBot, 5, (255, 255, 0), -1)
    
    return img_pnt, extreme_point


def crop_img(img, extreme_point):
    
    new_img = img[extreme_point[2][1]:extreme_point[3][1], extreme_point[0][0]:extreme_point[1][0]]
    
    return new_img

def remove_noise(img, extreme_point):

    new_img = np.zeros((img.shape[0],img.shape[1], img.shape[2]), dtype=np.uint8)
    
    new_img[extreme_point[2][1]:extreme_point[3][1], extreme_point[0][0]:extreme_point[1][0]] = img[extreme_point[2][1]:extreme_point[3][1], extreme_point[0][0]:extreme_point[1][0]]
    
    return new_img


def calculate_background(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    
    object_ = area / (gray.shape[0]*gray.shape[1]) * 100; 
    background = 100 - object_

    return background, object_


background_list = []; cropped_background_list = []
object_list = []; cropped_object_list = []

tmp = df.sample(500)

for i, path in enumerate(tqdm(tmp['image_path'],total=len(tmp['image_path']), desc='Calculating Background')):
    img = load_img(path)
    background, object_  = calculate_background(img)
    background_list.append(background)
    object_list.append(object_)

    _, extreme_point = find_contour(img)
    cropped_img = crop_img(img, extreme_point)
    background, object_  = calculate_background(cropped_img)
    cropped_background_list.append(background)
    cropped_object_list.append(object_)

object_df = pd.DataFrame(object_list, columns=['Object'])
background_df = pd.DataFrame(background_list, columns=['Background'])

area_df = pd.concat([object_df, background_df], axis=1)

cropped_object_df = pd.DataFrame(cropped_object_list, columns=['Object'])
cropped_background_df = pd.DataFrame(cropped_background_list, columns=['Background'])

cropped_area_df = pd.concat([cropped_object_df, cropped_background_df], axis=1)


trace1 = go.Histogram(
    x = area_df['Background'],
    opacity = 0.75,
    name = 'Background',
    marker = dict(color='rgba(0,0,255,0.2)'),
)
trace2 = go.Histogram(
    x = area_df['Object'],
    opacity = 0.75,
    name = 'Object',
    marker = dict(color='rgba(255,0,0,0.2)'),
)

trace3 = go.Histogram(
    x = cropped_area_df['Background'],
    opacity = 0.75,
    name = 'Background',
    marker = dict(color='rgba(0,0,255,0.2)'),
    showlegend=False,
)
trace4 = go.Histogram(
    x = cropped_area_df['Object'],
    opacity = 0.75,
    name = 'Object',
    marker = dict(color='rgba(255,0,0,0.2)'),
    showlegend=False,
)

fig = make_subplots(
    rows=1, cols=2, subplot_titles=("Before Cropping", "After Cropping")
)

fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=1, col=1)

fig.add_trace(trace3, row=1, col=2)
fig.add_trace(trace4, row=1, col=2)

fig.update_layout(
    title = "Space Ratio Distribution",
    barmode='overlay',
)

fig.show(renderer="iframe")


ROWS = 3; COLS = 6

fig, axes = plt.subplots(ROWS, COLS, figsize=(2*COLS, 2*ROWS))
axes = axes.flatten()

tmp = df.sample(ROWS*COLS)

for i in range(ROWS*COLS):
    path = tmp.iloc[i]['image_path']
    img = load_img(path)

    img, _ = find_contour(img)
    axes[i].imshow(img)
    axes[i].axis('off')

plt.tight_layout()
plt.show()


ROWS = 3; COLS = 6

fig, axes = plt.subplots(ROWS, COLS, figsize=(2*COLS, 2*ROWS))
axes = axes.flatten()

tmp = df.sample(ROWS*COLS)

for i in range(ROWS*COLS):
    path = tmp.iloc[i]['image_path']
    img = load_img(path)

    _, extreme_point = find_contour(img)
    
    axes[i].imshow(crop_img(img, extreme_point))
    axes[i].axis('off')

plt.tight_layout()
plt.show()


### Wandb.Login
### Wandb.init
### Wandb.Watch
### Wandb.log
### run finish


df['empty'] = df['empty'].astype(int)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for i, (_, val_idx) in enumerate(skf.split(df, df['empty'], groups=df['case'])):
    df.loc[val_idx, 'cls_fold'] = i


def get_train_transform():
    return A.Compose([
        A.Resize(*CFG.img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),    
        ], p=0.25),
    ])

def get_valid_transform():
    return A.Compose([
        A.Resize(*CFG.img_size),
    ])


class Cls_Dataset(Dataset):
    def __init__(self, data, transforms=None, remove=True):
        super(Cls_Dataset, self).__init__()
        self.data = data
        self.transforms = transforms
        self.remove = remove
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        img_path = row['image_path']
        img = load_img(img_path)

        m = np.nanmean(img)
        img = np.nan_to_num(img, nan=m)
        
        img = A.Compose(A.CLAHE(clip_limit=2, tile_grid_size=(4.0, 4.0)))(image=img)['image']

        if self.remove:
            _, extreme_point = find_contour(img)
            img = remove_noise(img, extreme_point)

        if self.transforms:
            img = self.transforms(image=img)['image']
            img = img / 255 # 0~1
            mean = np.array([0.5, 0.5, 0.5])
            std = np.array([0.5, 0.5, 0.5])
            img = (img - mean)/std

        X = torch.tensor(img, dtype=torch.float32).permute(2,0,1)
        y = torch.tensor(row['empty'], dtype=torch.float32)
        
        return X, y 


def accuracy(y_true, y_pred, thr=0.5):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)

    correct = (y_true == y_pred).sum()
    total = y_true.size(0)
    
    return correct/total

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes,kernel_size):
        super().__init__()
        self.out_channels = out_planes
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_planes, 
            out_planes,
            kernel_size=kernel_size,
            padding = self.padding,
            stride=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
        
class ZPool(nn.Module):
    def forward(self, x):

        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B,1,H,W)
        avg_out = torch.mean(x, dim=1, keepdim=True) # (B,1,H,W)

        cat = torch.cat([max_out, avg_out], dim=1) # (B,2,H,W)
        
        return cat

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 3
        self.compress = ZPool()
        self.conv = BasicConv(2,1,kernel_size)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.cw = AttentionGate() # Channel, Width
        self.hc = AttentionGate() # Height, Channel
        self.hw = AttentionGate() # Height, Width

    def forward(self, x):
        ## Channel, Width
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out1 = x_out1.permute(0,2,1,3).contiguous()
        ## Height, Channel
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out2 = x_out2.permute(0,3,2,1).contiguous()
        ## Height, Width
        x_out = self.hw(x)
        x_out = (x_out + x_out1 + x_out2) / 3

        return x_out


class Cls_EffNet(pl.LightningModule):
    def __init__(self):
        super(Cls_EffNet, self).__init__()
        self.feature_extractor = timm.create_model(CFG.cls_model_name, pretrained=True,
                                                   features_only=True)
        self.TAM = TripletAttention()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.channels = self.feature_extractor.feature_info[-1]['num_chs']
        self.classifier = nn.Linear(self.channels, 1)
        self.criterion = nn.BCELoss()
        self.history = defaultdict(list)
        self.true = []
        self.pred = []

    def forward(self, x):
        out = self.feature_extractor(x)[-1]
        out = self.TAM(out)
        out = self.gap(out)
        out = self.classifier(out.view(out.size(0),-1))
        return torch.sigmoid(out)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs.squeeze(1), targets)
        accuracy_score = accuracy(targets, outputs.squeeze(1))
        
        self.history['train_loss'].append(loss.item())
        self.log('train_loss', loss, prog_bar=True)
        self.log('accuracy_score', accuracy_score, prog_bar=True)

        lr = self.optimizers().param_groups[0]['lr']
        self.history['lr'].append(lr)
        self.log('lr',lr, prog_bar=True)

        wandb.log({"Train Loss": loss})
        wandb.log({"Learning Rate": lr})
        wandb.log({"Train Accuracy": accuracy_score})
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        self.log('Mem', mem, prog_bar=True)
        torch.cuda.empty_cache()
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        images, targets= batch
        outputs = self.forward(images)
        loss = self.criterion(outputs.squeeze(1), targets)
        accuracy_score = accuracy(targets, outputs.squeeze(1))
        self.log('val_accuracy_score', accuracy_score, prog_bar=True)

        self.log('valid_loss', loss, prog_bar=True)
        self.history['valid_loss'].append(loss.item())

        wandb.log({"Valid Loss": loss})
        wandb.log({"Valid Accuracy": accuracy_score})
        
        val_mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        self.log('Val_Mem', val_mem, prog_bar=True)
        
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images); 
        loss = self.criterion(outputs.squeeze(1), targets)
        accuracy_score = accuracy(targets, outputs.squeeze(1))
        
        self.pred.append(outputs.squeeze(1).cpu().numpy())
        self.true.append(targets.cpu().numpy())

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_accuracy_score', accuracy_score, prog_bar=True)

        torch.cuda.empty_cache()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=CFG.lr, weight_decay=CFG.wd)

        scheduler_dict = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.cls_epochs, eta_min=CFG.min_lr,
            last_epoch=-1, verbose=False
            ),
            'interval': 'epoch'
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}


train_df = df[df['cls_fold'] != 0].reset_index(drop=True)
valid_df = df[df['cls_fold'] == 0].reset_index(drop=True)

if CFG.debug:
    train_df = train_df.sample(500, random_state=CFG.seed).reset_index(drop=True)
    valid_df = valid_df.sample(500, random_state=CFG.seed).reset_index(drop=True)

train_ds = Cls_Dataset(train_df, transforms=get_train_transform())
valid_ds = Cls_Dataset(valid_df, transforms=get_valid_transform())

train_loader = DataLoader(train_ds, shuffle=True, batch_size=CFG.train_bs,
                          drop_last=False, num_workers = 4)
valid_loader = DataLoader(valid_ds, shuffle=False, batch_size=CFG.valid_bs,
                          drop_last=False, num_workers = 4)

## Pytorch LightningModule Provides also SWA(Stochastic Weighted Averaging)
from pytorch_lightning.callbacks import StochasticWeightAveraging
swa_callback = StochasticWeightAveraging(swa_epoch_start=int(0.8*CFG.cls_epochs), swa_lrs=CFG.min_lr * 2)

trainer_cls = pl.Trainer(
    max_epochs=CFG.cls_epochs,
    accelerator='auto',
    devices=CFG.device_count,
    callbacks = [LearningRateMonitor(logging_interval='step'),
                 swa_callback,]
    )

## Wandb
run = wandb.init(project='UWMGI Classification',
                 config={'cv': 'stratifiedkfold',
                         'img width': CFG.img_size[1],
                         'img height': CFG.img_size[0],
                         'epoch': CFG.cls_epochs,
                         'model_name': CFG.cls_model_name,
                         'learning rate': CFG.lr,
                         'scheduler': CFG.scheduler,
                          },
                name=f"{CFG.model_name}_ex0_debug_{CFG.debug}",
                )

cls_model = Cls_EffNet()
cls_model.to(CFG.device); cls_model.train()


trainer_cls.fit(model = cls_model, train_dataloaders = train_loader,
           val_dataloaders = valid_loader)
trainer_cls.save_checkpoint(f'{CFG.cls_dir}/best_effnet.ckpt')

run.finish()
display(IFrame(run.url, width=1000, height=700))


!ls /kaggle/working/wandb/


trainer_cls.test(dataloaders=valid_loader)
del train_ds, valid_ds, train_loader, valid_loader; clean_memory()


cls_pred = np.concatenate(cls_model.pred, axis=0)
cls_true = np.concatenate(cls_model.true, axis=0)

cls_pred_df = pd.DataFrame(cls_pred, columns=['pred'])
cls_true_df = pd.DataFrame(cls_true, columns=['true'])

from sklearn.metrics import accuracy_score
acc = accuracy_score(cls_true_df['true'], (cls_pred_df['pred'] >= 0.5).astype(int))
acc = acc * 100

x_min = 0
x_max = 1.001
bin_size = 0.05 

trace1 = go.Histogram(
    x=cls_pred_df['pred'],
    opacity=0.5,
    name='Predicted',
    marker=dict(color='blue'),
    xbins=dict(
        start=x_min,
        end=x_max,
        size=bin_size
    )
)

trace2 = go.Histogram(
    x=cls_true_df['true'],
    opacity=0.5,
    name='True',
    marker=dict(color='pink'),
    xbins=dict(
        start=x_min,
        end=x_max,
        size=bin_size
    )
)

data = [trace1, trace2]

layout = go.Layout(
    title=f"Empty(1) vs Non-Empty(0): Accuracy is {acc:.3f}%",
    barmode='overlay',
    legend=dict(
        bordercolor='black',
        borderwidth=1
    ),
    xaxis_title="Value",
    yaxis_title="Count"
)

fig = go.Figure(data=data, layout=layout)

fig.show(renderer="iframe")


## When Training Segmentation Model
## We're going to use only non-empty model 

df_origin = df.copy()
df = df[df['empty'] == 0].reset_index(drop=True)

skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups=df['case'])):
    df.loc[val_idx, 'fold'] = fold


from segmentation_models_pytorch.encoders import get_preprocessing_fn

preprocess_input = get_preprocessing_fn(CFG.backbone, pretrained='imagenet')


sns.set(style='whitegrid', context='notebook')
fig, axes = plt.subplots(1, 2, figsize=(12,6))


path = df.loc[0]['image_path']
img = np.load(path)
img = (img - img.min())/ (img.max() - img.min() + 1e-9)
img = (img*255).astype('uint8')

axes[0].hist(img.flatten(), alpha=0.5, color='blue')
axes[0].set_title("Before Preprocessing")

preprocessed_img = preprocess_input(img)

axes[1].hist(preprocessed_img.flatten(), alpha=0.5, color='red')
axes[1].set_title("After Preprocessing")

plt.tight_layout()
plt.show()


class UMWGI_Dataset(Dataset):
    def __init__(self, df, label=True, transforms=None, remove=True):
        self.df = df
        self.label = label
        self.img_paths = df['image_path'].to_list()
        self.msk_paths = df['mask_path'].to_list()
        self.transforms = transforms
        self.remove = remove

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = load_img(img_path)
        img = A.Compose(A.CLAHE(clip_limit=2, tile_grid_size=(4.0, 4.0)))(image=img)['image']

        if self.remove:
            _, extreme_point = find_contour(img)
            img = remove_noise(img, extreme_point)
            
        img = preprocess_input(img)
        
        if self.label: 
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.remove:
                msk = remove_noise(msk, extreme_point)
            
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data['image']
                msk = data['mask']

            # (Height, Width, Channel) => (Channel, Height, Width)
            # img = preprocess_input(img)
            img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)
            msk = torch.tensor(msk, dtype=torch.float32).permute(2,0,1)

            return img, msk
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img = data['image']

                # (Height, Width, Channel) => (Channel, Height, Width)
                # img = preprocess_input(img)
                img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)
            return img


## Batch: [dataset[0], datset[1], dataset[2], dataset[3]]
## Batch: [ (img0, msk0), (img1, msk1), (img2, msk2)]

def mixup_collate_fn(batch, mixup_prob=0.5, alpha=2.0):
    imgs, msks = zip(*batch)
    imgs = torch.stack(imgs)
    msks = torch.stack(msks)

    batch_size  = imgs.size(0)
    indices = torch.randperm(batch_size)

    for i in range(batch_size):
        if np.random.rand() <= mixup_prob:
            lam = np.random.beta(alpha, alpha)

            imgs[i] = lam * imgs[i] + (1 - lam) * imgs[indices[i]]
            msks[i] = lam * msks[i] + (1 - lam) * msks[indices[i]]

    return imgs, msks

def cutmix_collate_fn(batch, cutmix_prob=0.5, num_patches = 3, alpha=2.0):
    imgs, msks = zip(*batch)
    imgs = torch.stack(imgs)
    msks = torch.stack(msks)

    batch_size  = imgs.size(0)
    h, w = imgs.size(2), imgs.size(3)
    indices = torch.randperm(batch_size)

    for i in range(batch_size):
        if np.random.rand() <= cutmix_prob:
            for _ in range(num_patches):
                lam = np.random.beta(alpha, alpha)
    
                cw = min(int(w * (1-lam)), w // 8)
                ch = min(int(h * (1-lam)), h // 8)
                cx = np.random.randint(0, w - cw + 1)
                cy = np.random.randint(0, h - ch + 1)

                imgs[i, :, cy:cy + ch, cx:cx+cw] = imgs[indices[i], :, cy:cy + ch, cx:cx+cw]
                msks[i, :, cy:cy + ch, cx:cx+cw] = msks[indices[i], :, cy:cy + ch, cx:cx+cw]
    
    
    return imgs, msks


def prepare_loader(fold, debug=False):
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)

    if debug: 
        train_df = train_df.sample(500)
        valid_df = train_df.sample(250)

    train_dataset = UMWGI_Dataset(train_df, transforms = get_train_transform())
    valid_dataset = UMWGI_Dataset(valid_df, transforms = get_valid_transform())

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=CFG.train_bs,
                              num_workers=4, pin_memory=True, drop_last = False,
                              collate_fn = cutmix_collate_fn)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=CFG.valid_bs,
                              num_workers=4, pin_memory=True, drop_last = False,
                              collate_fn = torch.utils.data._utils.collate.default_collate)

    return train_loader, valid_loader


train_loader, valid_loader = prepare_loader(fold=CFG.fold, debug=True)

imgs, msks = next(iter(valid_loader))

imgs.size(), msks.size()


ROWS = 3
COLS = 6

fig, axes = plt.subplots(ROWS, COLS, figsize=(2*COLS, 2*ROWS))
axes = axes.flatten()

for i in range(ROWS):
    for j in range(COLS):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = imgs[i*COLS+j].permute(1,2,0).cpu().numpy()
        img = (img * std) + mean
        img = np.clip(img * 255, 0, 255).astype('uint8')
        
        msk = msks[i*COLS+j].permute(1,2,0).cpu().numpy()
        msk = (msk*255).astype('uint8')

        ## Applying msk2contour
        img[msk2contour(msk[:,:,0], width=1) == 255, 0] = 255
        img[msk2contour(msk[:,:,1], width=1) == 255, 1] = 255
        img[msk2contour(msk[:,:,2], width=1) == 255, 2] = 255

        
        axes[i*COLS+j].imshow(img)
        axes[i*COLS+j].axis('off')
        axes[i*COLS+j].legend(handles=legend_elements, loc='upper right', fontsize=8, 
                              labelspacing=0.1,     
                              borderpad=0.1,         
                              handlelength=0.5,    
                              handletextpad=0.2,   
                              borderaxespad=0.2,    
                              markerscale=0.5 )

        del img, msk

plt.tight_layout()
plt.show()


clean_memory()


"""Model Architecture"""

## Unet

# It consists of two main parts
# An Encoder(downsampling path) that extracts increasingly abstrct features
# A Decoder(upsampling path) that gradually recovers spatial details

# The key is the use of skip connections between corresponding encoder and decoder layers.
# These connections allow the decoder to access fine-grained details from earlier encoder layers
# which helps produce more precise segmentation masks.

from segmentation_models_pytorch.encoders import get_encoder_names
print(get_encoder_names())


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

# CAM: Channel Attention Module
class CAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels//reduction, in_channels, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        gap = self.gap(x); gap_out = self.mlp(gap)
        gmp = self.gmp(x); gmp_out = self.mlp(gmp)

        cam_out = gap_out + gmp_out
        cam_out = cam_out * x

        return cam_out

# SAM: Spatial Attention Module
class SAM(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1)//2
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=padding, bias=False)

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B,1,H,W)
        avg_out = torch.mean(x, dim=1, keepdim=True) # (B,1,H,W)

        cat = torch.cat([max_out, avg_out], dim=1) # (B,2,H,W)

        sam_out = self.conv(cat) # (B,1,H,W)
        sam_out = torch.sigmoid(sam_out) * x

        return sam_out

class CBAMBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cam = CAM(in_channels=in_channels)
        self.sam = SAM()

    def forward(self, x):
        out = self.cam(x)
        out = self.sam(out)
        
        return out


import segmentation_models_pytorch as smp

def build_model():
    model = smp.Unet(
        encoder_name = CFG.backbone,
        encoder_depth = 5, # range [3,5]
        encoder_weights = "imagenet", # Transfer Learning
        decoder_channels = (256, 128, 64, 32, 16),
        decoder_use_norm = True, # batchnorm, layernorm
        decoder_attention_type = None, # attention: scse
        decoder_interpolation = "nearest", # bilinear, bicubic, area, nearest-exact
        in_channels = 3, # RGB images
        classes = CFG.num_classes, # Large Bowel, Small Bowel, Stomache
        activation = "sigmoid", # softmax, logsoftmax, tanh 
        aux_params = None, # Multi Task Learning
    )
    
    for i, block in enumerate(model.decoder.blocks):
        skip_channels = block.conv1[0].in_channels
        out_channels = block.conv2[0].in_channels

        if i > 0 and i < 3:
            block.attention1.attention = CAM(skip_channels)
            block.attention2.attention = CAM(out_channels)
        elif i >= 3:
            block.attention1.attention = TripletAttention()
            block.attention2.attention = CAM(out_channels)
    
    model.to(CFG.device)
    return model

def load_model(path):
    
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    
    return model


model = build_model()

print(f"Encoder Depths are ", model.encoder._out_channels)

for i in range(len(model.encoder._out_channels) - 1):
    print(f'{Fore.BLACK}#'*25)
    if i == 0:
        print(f"{c_}decoder block {i} in_channels: ", model.encoder._out_channels[5-i] + model.encoder._out_channels[4-i])
    elif i != len(model.encoder._out_channels) - 2:
        print(f"{c_}decoder block {i} in_channels: ", model.encoder._out_channels[4-i] + model.decoder.blocks[i-1].conv1[0].out_channels)
    else:
        print(f"{c_}decoder block {i} in_channels: ", model.decoder.blocks[i-1].conv1[0].out_channels)
    print(f"{Fore.BLUE}decoder block {i} out_channels: {model.decoder.blocks[i].conv1[0].out_channels}")
    print(f'{Fore.BLACK}#'*25)


decoder_in_list = [(f'block_{i}', model.decoder.blocks[i].conv1[0].in_channels) for i in range(5)]
decoder_in_df = pd.DataFrame(decoder_in_list, columns=['block','decoder_in_channel'])

decoder_out_list = [(f'block_{i}', model.decoder.blocks[i].conv1[0].out_channels) for i in range(5)]
decoder_out_df = pd.DataFrame(decoder_out_list, columns=['block','decoder_out_channel'])

trace1 = go.Bar(
    x = decoder_in_df['block'],
    y = decoder_in_df['decoder_in_channel'],
    name = 'decoder_in_channel',
    marker = dict(color = f'rgba(0,0,255,0.2)',
                 line=dict(color=f'rgb(0,0,255)', width=2)),
)

trace2 = go.Bar(
    x = decoder_out_df['block'],
    y = decoder_out_df['decoder_out_channel'],
    name = 'decoder_out_channel',
    marker = dict(color = f'rgba(0,255,0,0.2)',
                 line=dict(color=f'rgb(0,255,0)', width=2)),
)

fig = make_subplots(rows=1, cols=2, subplot_titles=("In Channels in Decoder", "Out Channels in Encoder"))

fig.add_trace(trace1, 1,1)
fig.add_trace(trace2, 1,2)

fig.update_layout(
    title = "Channel Configuration",
    barmode='group',
    yaxis = dict(title='Channels', ticklen = 5, zeroline=False),
    legend = dict(title="I/O Channels", bordercolor='black', borderwidth=1),

)


fig.show(renderer="iframe")


!pip install -q torchinfo
from torchinfo import summary

summary(
        model, 
        input_size=(1, 3, *CFG.img_size),  # 배치 크기 포함
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=3,
    )


TORCH_VIZ = False

if TORCH_VIZ:
    !pip install -q torchviz
    from torchviz import make_dot

    x = torch.randn(1, 3, *CFG.img_size)
    y = model(x)

    make_dot(y, params = dict(model.named_parameters())).render(f"{CFG.model_name}", format="png")

    display(Image.open(f"{CFG.model_name}.png"))


COLS = 6; ROWS = 3

rand = np.random.choice(CFG.train_bs, 12, replace=False)
images = imgs[rand].to(CFG.device)

with torch.inference_mode():
    masks = model(images)

fig, axes = plt.subplots(ROWS, COLS, figsize=(2*COLS, 2*ROWS))
axes = axes.flatten()

for i in range(COLS*ROWS//2):

    msk = msks[rand[i]].permute(1,2,0).cpu().numpy()
    
    axes[2*i].imshow(msk)
    axes[2*i].axis('off')
    axes[2*i].set_title("Ground Truth")

    mask = masks[i].permute(1,2,0).cpu().numpy()
    
    axes[2*i+1].imshow(mask)
    axes[2*i+1].axis('off')
    axes[2*i+1].set_title("Predict")

plt.tight_layout()
plt.show()


### Region Loss
DiceLoss = smp.losses.DiceLoss(mode='multilabel', from_logits=False)
JaccardLoss = smp.losses.JaccardLoss(mode='multilabel', from_logits=False)
LovaszLoss = smp.losses.LovaszLoss(mode='multilabel', from_logits=False)
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', from_logits=False)

### Boundary Loss
BCELoss = smp.losses.SoftBCEWithLogitsLoss()

### Distribution Loss
CELoss = smp.losses.SoftCrossEntropyLoss()
FocalLoss = smp.losses.FocalLoss(mode='multilabel')


# Custom Loss = Region Loss + Boundary Loss
def criterion(y_pred, y_true):
    return 0.5 * DiceLoss(y_pred, y_true) + 0.5 * BCELoss(y_pred, y_true)


def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=1e-9):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    
    return dice
    
def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=1e-9):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    
    return iou


def shift_right(masks, shift=3):
    empty_masks = torch.zeros_like(masks)

    shifted_masks = torch.cat((masks[:,:,:,shift:], empty_masks[:,:,:,:shift]), axis=3)
    return shifted_masks


fig, axes = plt.subplots(1,6, figsize=(12, 2))
axes = axes.flatten()

for i in range(6):
    msk = msks[20].unsqueeze(0)
    shifted_msk = shift_right(msks[20].unsqueeze(0), shift=i*2)
    dice = dice_coef(msk,shifted_msk)
    iou = iou_coef(msk, shifted_msk)
    
    msk = msk.squeeze(0).permute(1,2,0).cpu().numpy()
    
    shifted_msk = shifted_msk.squeeze(0).permute(1,2,0).cpu().numpy()
    shifted_msk[...,0] = msk2contour(shifted_msk[...,0], width=1)
    shifted_msk[...,1] = msk2contour(shifted_msk[...,1], width=1)
    shifted_msk[...,2] = msk2contour(shifted_msk[...,2], width=1)

    axes[i].imshow(msk)
    axes[i].imshow(shifted_msk, alpha=0.5)
    axes[i].axis('off')
    axes[i].set_title(f'Shift {i*2}\n Dice Metric: {dice:.2f}\n Iou Metric: {iou:.2f}', size=8, fontweight='bold')

plt.tight_layout()
plt.show()


def fetch_scheduler(scheduler_type, optimizer, loader):
    if scheduler_type == 'ReduceLROnPlateau':  
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', verbose=False, factor=0.1, patience=20, threshold=1e-3,
            threshold_mode='abs', min_lr = CFG.min_lr
        )
        return scheduler
        
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr,
            last_epoch=-1, verbose=False
        )
        return scheduler
        
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0 = CFG.epochs, T_mult=CFG.T_mult, eta_min=CFG.min_lr,
            last_epoch=-1, verbose=False
        )
        return scheduler
        
    elif scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=CFG.epochs, gamma=0.1, last_epoch=-1
        ) 
        return scheduler
        
    elif scheduler_type == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr= 0.01,
            epochs=CFG.epochs,
            steps_per_epoch= len(loader), 
        )
        return scheduler
    
    elif scheduler_type == None:
        return None


model = nn.Linear(2,1)

optim_onecycle = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
optim_step     = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
optim_cosine   = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
optim_coswarm  = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)

one_cycle_lr = []; step_lr = []; cosineanneal_lr = []; cosineannealwarm_lr = []
one_cycle_sc = fetch_scheduler('OneCycleLR', optim_onecycle, train_loader)
step_sc = fetch_scheduler('StepLR', optim_step, train_loader)
cosineanneal_sc = fetch_scheduler('CosineAnnealingLR', optim_cosine, train_loader)
cosineannealwarm_sc = fetch_scheduler('CosineAnnealingWarmRestarts', optim_coswarm, train_loader)

for i in range(CFG.epochs*len(train_loader)):
    one_cycle_sc.step(); one_cycle_lr.append(optim_onecycle.param_groups[0]['lr'])
for i in range(CFG.epochs*len(train_loader)):
    step_sc.step(); step_lr.append(optim_step.param_groups[0]['lr'])
for i in range(CFG.epochs*len(train_loader)):
    cosineanneal_sc.step(); cosineanneal_lr.append(optim_cosine.param_groups[0]['lr'])
for i in range(CFG.epochs*len(train_loader)):
    cosineannealwarm_sc.step(); cosineannealwarm_lr.append(optim_coswarm.param_groups[0]['lr'])

one_cycle_df = pd.DataFrame(one_cycle_lr, columns=['lr']); step_df = pd.DataFrame(step_lr, columns=['lr'])
cosineanneal_df = pd.DataFrame(cosineanneal_lr, columns=['lr']); cosineannealwarm_df = pd.DataFrame(cosineannealwarm_lr, columns=['lr'])

trace1 = go.Scatter(
    x = one_cycle_df.index,
    y = one_cycle_df['lr'],
    name = 'One Cycle',
    marker = dict(color = 'rgba(0,0,125,0.3)'),
)
trace2 = go.Scatter(
    x = step_df.index,
    y = step_df['lr'],
    name = 'Step',
    marker = dict(color = 'rgba(0,125,125,0.3)'),
)
trace3 = go.Scatter(
    x = cosineanneal_df.index,
    y = cosineanneal_df['lr'],
    name = 'CosineAnnealing',
    marker = dict(color = 'rgba(125,125,255,0.3)'),
)
trace4 = go.Scatter(
    x = cosineannealwarm_df.index,
    y = cosineannealwarm_df['lr'],
    name = 'CosineAnnealingWarmRestarts',
    marker = dict(color = 'rgba(125,125,0,0.3)'),
)


fig = make_subplots(rows=2, cols=2, subplot_titles=("One Cycle", "StepLR", "CosineAnnealingLR", "CosineAnnealingWarmRestarts"))
fig.add_trace(trace1, 1, 1); fig.add_trace(trace2, 1, 2)
fig.add_trace(trace3, 2, 1); fig.add_trace(trace4, 2, 2)

fig.update_layout(
    title = "Scheduler Type",
    xaxis1 = dict(title='iteration', ticklen = 5, zeroline=False),
    xaxis2 = dict(title='iteration', ticklen = 5, zeroline=False),
    xaxis3 = dict(title='iteration', ticklen = 5, zeroline=False),
    xaxis4 = dict(title='iteration', ticklen = 5, zeroline=False),
    yaxis1 = dict(title='learning rate', ticklen = 5, zeroline=False),
    yaxis2 = dict(title='learning rate', ticklen = 5, zeroline=False),
    yaxis3 = dict(title='learning rate', ticklen = 5, zeroline=False),
    yaxis4 = dict(title='learning rate', ticklen = 5, zeroline=False),
    legend = dict(bordercolor='black', borderwidth=1),

)

fig.show(renderer="iframe")


# swa_model = torch.optim.swa_utils.AverageModel(model)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
# swa_start = 160
# swa_scheduler = SWALR(optimizer, swa_lr=0.05)

# if epoch > swa_start:
# swa_model.update_parameters(model)
# swa_scheduler.step()

# torch.optim.swa_utils.update_bn(loader, swa_model)
# preds = swa_model(test_input)


class Trainer:
    def __init__(self, model, optimizer, scheduler, fold):
        self.model = model
        ## SWA averaged model 
        self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_scheduler = torch.optim.swa_utils.SWALR(self.optimizer, anneal_strategy="cos", 
                                                         anneal_epochs=7, swa_lr=CFG.min_lr*2)
        
        self.swa_start = int(1.2 * CFG.epochs)
        
        self.history = defaultdict(list)
        self.fold = fold
        self.best_loss = 10**5
        self.best_epoch = -1
        self.best_dice = 0
        self.best_iou = 0

        print(f"{c_} ### Trainer Prepared, Device is {CFG.device}")
    
    def fit(self, train_loader, valid_loader):

        for epoch in range(1, CFG.epochs + 1):
            print(f'{Fore.BLUE}#'*25)
            print(f'{Fore.BLUE}### Epoch {epoch}/{CFG.epochs}')
            print(f'{Fore.BLUE}#'*25)

            train_loss, train_dice, train_iou = self.train_one_epoch(train_loader)
            valid_loss, valid_dice, valid_iou = self.valid_one_epoch(valid_loader)

            self.history['Learning Rate'].append(self.optimizer.param_groups[0]['lr'])
            
            if epoch >= self.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            else:
                if self.scheduler == 'ReduceLROnPlateau':
                    self.scheduler.step(metrics=train_loss)
                else:
                    self.scheduler.step()
            
            print(f'{Fore.BLUE}Train Loss: {train_loss:.3f} | Valid Loss: {valid_loss:.3f}')
            print(f'{Fore.BLUE}Train Dice: {train_dice:.3f} | Valid Dice: {valid_dice:.3f}')
            print(f'{Fore.BLUE}Train IoU: {train_iou:.3f} | Valid IoU: {valid_iou:.3f}')
            
            self.model.eval()
            last_model_wts = copy.deepcopy(self.model.state_dict())
            torch.save(last_model_wts, f'{CFG.segment_dir}/fold{self.fold:2d}_last-checkpoint.bin')

            if valid_loss <= self.best_loss:
                self.best_loss = valid_loss
                self.best_epoch = epoch
                best_model_wts = copy.deepcopy(self.model.state_dict())

                torch.save(best_model_wts, f'{CFG.segment_dir}/fold{self.fold:2d}_best-checkpoint.bin')
            
            if valid_dice >= self.best_dice:
                self.best_dice = valid_dice
            if valid_iou >= self.best_iou:
                self.best_iou = valid_iou

        torch.optim.swa_utils.update_bn(train_loader, self.swa_model, device=CFG.device)
        swa_model_wts = copy.deepcopy(self.swa_model.module.state_dict())
        torch.save(swa_model_wts, f'{CFG.segment_dir}/fold{self.fold:2d}_swa-checkpoint.bin')
        
        print(f'\n{Fore.BLUE}Best Dice: {self.best_dice:.3f} | Best IoU: {self.best_iou:.3f}')
        self.model.load_state_dict(torch.load(f'{CFG.segment_dir}/fold{self.fold:2d}_best-checkpoint.bin'))

        return self.model, self.history
    
        
    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter(); summary_dice = AverageMeter(); summary_iou = AverageMeter()
        scaler = amp.GradScaler()

        pbar = tqdm(enumerate(train_loader),total=len(train_loader), desc='Training')
        for step, (images, masks) in pbar:
            images = images.to(CFG.device, dtype=torch.float)
            masks = masks.to(CFG.device, dtype=torch.float)

            batch_size = images.size(0)

            with amp.autocast(enabled=True):
                y_pred = self.model(images)
                loss = criterion(y_pred, masks)
                loss = loss / CFG.n_accumulate

            scaler.scale(loss).backward()

            if (step + 1) % CFG.n_accumulate == 0:
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

                summary_loss.update(loss.detach().item(), batch_size)
                summary_dice.update(dice_coef(masks, y_pred).cpu().numpy(), batch_size)
                summary_iou.update(iou_coef(masks, y_pred).cpu().numpy(), batch_size)
                
                self.history['Train Loss'].append(loss.detach().item())
                self.history['Train Dice'].append(dice_coef(masks, y_pred).cpu().numpy())
                self.history['Train IoU'].append(iou_coef(masks, y_pred).cpu().numpy())

            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0 
            pbar.set_postfix(
                train_loss = f'{summary_loss.avg:.4f}',
                train_dice = f'{summary_dice.avg:.3f}',
                train_iou = f'{summary_iou.avg:.3f}',
                lr = f'{self.optimizer.param_groups[0]["lr"]:.5f}',
                gpu_mem = f'{mem} GB'
            )
        torch.cuda.empty_cache()
        clean_memory()
        
        return summary_loss.avg, summary_dice.avg, summary_iou.avg

    @torch.no_grad()
    def valid_one_epoch(self, valid_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        summary_val_dice = AverageMeter()
        summary_val_iou = AverageMeter()

        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Validation')
        for step, (images, masks) in pbar:
            with torch.no_grad():
                images = images.to(CFG.device, dtype=torch.float)
                masks = masks.to(CFG.device, dtype=torch.float)
                batch_size = images.size(0)

                y_pred = self.model(images)
                loss = criterion(y_pred, masks)

                summary_loss.update(loss.item(), batch_size)
                summary_val_dice.update(dice_coef(masks, y_pred).cpu().numpy(), batch_size)
                summary_val_iou.update(iou_coef(masks, y_pred).cpu().numpy(), batch_size)

                self.history['Valid Loss'].append(loss.item())
                self.history['Valid Dice'].append(dice_coef(masks, y_pred).cpu().numpy())
                self.history['Valid IoU'].append(iou_coef(masks, y_pred).cpu().numpy())

            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0 
            pbar.set_postfix(
                    valid_loss = f'{summary_loss.avg:.4f}',
                    valid_dice = f'{summary_val_dice.avg:.3f}',
                    valid_iou = f'{summary_val_iou.avg:.3f}',
                    lr = f'{self.optimizer.param_groups[0]["lr"]:.5f}',
                    gpu_mem = f'{mem} GB'
                )
        torch.cuda.empty_cache()
        clean_memory()

        return summary_loss.avg, summary_val_dice.avg, summary_val_iou.avg


def run_training(fold):
    
    train_loader, valid_loader = prepare_loader(fold, debug=CFG.debug)
    model = build_model();
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    scheduler = fetch_scheduler(CFG.scheduler, optimizer, train_loader)

    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, fold=fold)

    best_model, history = trainer.fit(train_loader, valid_loader)
    del train_loader, valid_loader
    
    clean_memory()
    return best_model, history


if not CFG.load_segment:
    best_model, history = run_training(fold=CFG.fold)


if not CFG.load_segment:
    df_train_lr = pd.DataFrame(history['Learning Rate'], columns=['Learning Rate'])

    trace1 = go.Scatter(x = df_train_lr.index[:int(CFG.epochs*0.8)],
        y = df_train_lr[:int(CFG.epochs*0.8)]['Learning Rate'],
        name = 'Before SWA',
        marker = dict(color = 'rgba(255,0,0,0.3)'),
    )
    trace2 = go.Scatter(x = df_train_lr.index[int(CFG.epochs*0.8):],
        y = df_train_lr[int(CFG.epochs*0.8):]['Learning Rate'],
        name = 'After SWA',
        marker = dict(color = 'rgba(0,0,255,0.3)'),
    )
    
    layout = go.Layout(
    title=f"Learning Rate with SWA",
    legend=dict(
        bordercolor='black',
        borderwidth=1
    ),
    xaxis_title="Iterations",
    yaxis_title="Learing Rate"
)
    
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.show(renderer="iframe")
    


if not CFG.load_segment:    
    df_train_ls = pd.DataFrame(history['Train Loss'], columns=['Train Loss']); df_valid_ls = pd.DataFrame(history['Valid Loss'], columns=['Valid Loss'])
    df_train_dice = pd.DataFrame(history['Train Dice'], columns=['Train Dice']); df_valid_dice = pd.DataFrame(history['Valid Dice'], columns=['Valid Dice'])
    df_train_iou = pd.DataFrame(history['Train IoU'], columns=['Train IoU']); df_valid_iou = pd.DataFrame(history['Valid IoU'], columns=['Valid IoU'])
    
    trace1 = go.Scatter(
        x = df_train_ls.index,
        y = df_train_ls['Train Loss'],
        name = 'Train Loss',
        marker = dict(color = 'rgba(255,0,0,0.3)'),
    )
    
    trace2 = go.Scatter(
        x = df_train_dice.index,
        y = df_train_dice['Train Dice'],
        name = 'Train Dice',
        marker = dict(color = 'rgba(0,255,0,0.3)'),
    )
    
    trace3 = go.Scatter(
        x = df_train_iou.index,
        y = df_train_iou['Train IoU'],
        name = 'Train IoU',
        marker = dict(color = 'rgba(0,0,255,0.3)'),
    )
    
    trace4 = go.Scatter(
        x = df_valid_ls.index,
        y = df_valid_ls['Valid Loss'],
        name = 'Valid Loss',
        marker = dict(color = 'rgba(255,0,0,0.3)'),
    )
    
    trace5 = go.Scatter(
        x = df_valid_dice.index,
        y = df_valid_dice['Valid Dice'],
        name = 'Valid Dice',
        marker = dict(color = 'rgba(0,255,0,0.3)'),
    )
    
    trace6 = go.Scatter(
        x = df_valid_iou.index,
        y = df_valid_iou['Valid IoU'],
        name = 'Valid IoU',
        marker = dict(color = 'rgba(0,0,255,0.3)'),
    )
    
    
    
    fig = make_subplots(rows=2, cols=3, subplot_titles=['Loss','Dice', 'IoU','Val Loss','Val Dice', 'Val IoU'])
    
    fig.add_trace(trace1, 1, 1); fig.add_trace(trace4, 2, 1)
    fig.add_trace(trace2, 1, 2); fig.add_trace(trace5, 2, 2)
    fig.add_trace(trace3, 1, 3); fig.add_trace(trace6, 2, 3)
    
    fig.update_layout(
        title = "Tracking",
        xaxis1 = dict(title='iteration', ticklen = 5, zeroline=False),
        xaxis2 = dict(title='iteration', ticklen = 5, zeroline=False),
        xaxis3 = dict(title='iteration', ticklen = 5, zeroline=False),
        xaxis4 = dict(title='iteration', ticklen = 5, zeroline=False),
        xaxis5 = dict(title='iteration', ticklen = 5, zeroline=False),
        xaxis6= dict(title='iteration', ticklen = 5, zeroline=False),
        legend = dict(bordercolor='black', borderwidth=1),
    
    )
    
    fig.show(renderer="iframe")


features = {}

if not CFG.load_segment:
    best_model.eval() # Inference Mode
else:
    best_model = load_model(CFG.load_path)


def hook_fn(module, input, output):
    features[module] = output

hooks = []
for i, block in enumerate(best_model.encoder.blocks):
    h = block.register_forward_hook(hook_fn)
    hooks.append(h)

img = imgs[0].unsqueeze(0).to(CFG.device)

with torch.inference_mode():
     best_model(img)

fig, axes = plt.subplots(1,7, figsize=(14,4))
axes = axes.flatten()

for i, (module, feat) in enumerate(features.items()):
    feat = feat.squeeze(0).permute(1,2,0).cpu().numpy()
    feat = np.mean(feat, axis=-1)
    axes[i].imshow(feat, cmap='bone')
    axes[i].set_title(f"Feature map {i}\n {feat.shape}", fontsize=8)
    axes[i].axis('off')

for j in range(i+1, 7):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


segments = {}

def hook_fn(module, input, output):
    segments[module] = output

hooks = []
for i, block in enumerate(best_model.decoder.blocks):
    h = block.register_forward_hook(hook_fn)
    hooks.append(h)

img = imgs[0].unsqueeze(0).to(CFG.device)

with torch.inference_mode():
     best_model(img)

fig, axes = plt.subplots(1,5, figsize=(10,4))
axes = axes.flatten()

for i, (module, seg) in enumerate(segments.items()):
    seg = seg.squeeze(0).permute(1,2,0).cpu().numpy()
    seg = np.mean(seg, axis=-1)
    axes[i].imshow(seg, cmap='bone')
    axes[i].set_title(f"Decoder Block {i}\n {seg.shape}", fontsize=8)
    axes[i].axis('off')

plt.tight_layout()
plt.show()


COLS = 6; ROWS = 3

rand = np.random.choice(CFG.train_bs, 12, replace=False)
images = imgs[rand].to(CFG.device)

with torch.inference_mode():
    masks = best_model(images)

fig, axes = plt.subplots(ROWS, COLS, figsize=(2*COLS, 2*ROWS))
axes = axes.flatten()

for i in range(COLS*ROWS//2):

    msk = msks[rand[i]].permute(1,2,0).cpu().numpy()
    
    axes[2*i].imshow(msk)
    axes[2*i].axis('off')
    axes[2*i].set_title("Ground Truth")

    mask = masks[i].permute(1,2,0).cpu().numpy()
    mask = (mask > 0.5).astype('float')
    
    axes[2*i+1].imshow(mask)
    axes[2*i+1].axis('off')
    axes[2*i+1].set_title("Predict")

plt.tight_layout()
plt.show()


del imgs, msks,best_model
clean_memory()


if not CFG.load_segment:
    model = load_model(f'/kaggle/working/{CFG.segment_dir}/fold{CFG.fold:2d}_best-checkpoint.bin',)
else:
    model = load_model(CFG.load_path)

if CFG.debug: 
    df_origin = df_origin.sample(500).reset_index(drop=True)
else:
    df_origin = df_origin.sample(10_000, random_state=CFG.seed).reset_index(drop=True)

ds = UMWGI_Dataset(df_origin, transforms = get_valid_transform())
data_loader = DataLoader(ds,shuffle=False, batch_size=CFG.valid_bs//2,
                              num_workers=4, pin_memory=True, drop_last = False,
                              collate_fn = torch.utils.data._utils.collate.default_collate)

model.eval()
all_pred = []; all_true = []
with torch.no_grad():
    for i, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Fold {CFG.fold} Predict"):
        images = images.to(CFG.device, dtype=torch.float)
        masks = masks.to(CFG.device, dtype=torch.float)
        pred = model(images)  # (B, C, H, W)
        pred = pred.cpu().numpy(); true = masks.cpu().numpy()
        all_pred.append(pred); all_true.append(true)

        del images, masks, pred, true

    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    torch.cuda.empty_cache()
    clean_memory(); del data_loader, model


all_pred = torch.from_numpy(all_pred).float(); all_true = torch.from_numpy(all_true).float()
print("Dice Score with only Full Empty Data is: ", dice_coef(all_true, all_pred).item())


ds = Cls_Dataset(df_origin, transforms = get_train_transform())
data_loader = DataLoader(ds, shuffle=False, batch_size=CFG.valid_bs,
                            num_workers=4, pin_memory=True, drop_last = False,
                            collate_fn = torch.utils.data._utils.collate.default_collate)

cls_model = Cls_EffNet.load_from_checkpoint(
    checkpoint_path=f'/kaggle/working/{CFG.cls_dir}/best_effnet.ckpt'
)
cls_model.to(CFG.device)
cls_model.eval()

trainer_cls.test(model=cls_model, dataloaders=data_loader)
cls_pred = np.concatenate(cls_model.pred, axis=0)
cls_pred_df = pd.DataFrame(cls_pred, columns=['prediction'])


tmp = all_pred.clone()
empty_idx = cls_pred_df.loc[cls_pred_df['prediction'] >= 0.5].index # 1: Empty, 0: Non-Empty
tmp[empty_idx] = 0


print(f"{c_}Dice Score with Binary Gate Approach: {Fore.BLUE}{dice_coef(all_true, tmp).item():.3f}")


empty_idx = cls_pred_df.loc[cls_pred_df['prediction'] >= 0.90].index
uncertain_idx = cls_pred_df.loc[
    (cls_pred_df['prediction'] < 0.90) & (cls_pred_df['prediction'] >= 0.10)
].index

uncertain_probs = cls_pred_df.loc[uncertain_idx, 'prediction'].values 
uncertain_probs = torch.tensor(uncertain_probs, dtype=all_pred.dtype, device=all_pred.device)
uncertain_probs = uncertain_probs.view(-1, 1, 1, 1)

tmp2 = all_pred.clone()
tmp2[uncertain_idx] *= uncertain_probs
tmp2[empty_idx] = 0


print(f"{c_}Dice Score with Soft Weighting Approach in Threhold: 0.5: {Fore.BLUE}{dice_coef(all_true, tmp2, thr=0.5).item():.3f}")
print(f"{c_}Dice Score with Soft Weighting Approach in Threhold: 0.4: {Fore.BLUE}{dice_coef(all_true, tmp2, thr=0.4).item():.3f}")
print(f"{c_}Dice Score with Soft Weighting Approach in Threhold: 0.3: {Fore.BLUE}{dice_coef(all_true, tmp2, thr=0.3).item():.3f}")
print(f"{c_}Dice Score with Soft Weighting Approach in Threhold: 0.2: {Fore.BLUE}{dice_coef(all_true, tmp2, thr=0.2).item():.3f}")
print(f"{c_}Dice Score with Soft Weighting Approach in Threhold: 0.1: {Fore.BLUE}{dice_coef(all_true, tmp2, thr=0.1).item():.3f}")


empty_idx = cls_pred_df.loc[cls_pred_df['prediction'] >= 0.9].index
uncertain_idx = cls_pred_df.loc[
    (cls_pred_df['prediction'] < 0.9) & (cls_pred_df['prediction'] >= 0.1)
].index

uncertain_probs = cls_pred_df.loc[uncertain_idx, 'prediction'].values 
uncertain_probs = torch.tensor(uncertain_probs, dtype=all_pred.dtype, device=all_pred.device)
uncertain_probs = uncertain_probs.view(-1, 1, 1, 1)

for thr in [0.4, 0.5]:
    for i in np.linspace(0.2,0.8,7):
        tmp3 = all_pred.clone()
        alpha = round(i,1)
        beta = round(1 - alpha, 1)
        tmp3[uncertain_idx] = tmp3[uncertain_idx] ** alpha * uncertain_probs ** beta
        tmp3[empty_idx] = 0
        
        print(f"{c_}Dice Score with Soft Weighting Approach in alpha: {alpha}, beta: {beta} in Thr = {thr}: {Fore.BLUE}{dice_coef(all_true, tmp3, thr=thr).item():.3f}")


# Channel 0: Large Bowel, Channel 1: Small Bowel, Channel 2: Stomache

discard_value = 10

tmp = all_pred.clone()
empty_idx = cls_pred_df.loc[cls_pred_df['prediction'] >= 0.5].index # 1: Empty, 0: Non-Empty
tmp[empty_idx] = 0

tmp2 = (tmp > 0.5).clone()

for idx in range(tmp2.size(0)):
    for i in range(CFG.num_classes):
        if torch.sum(tmp2[idx,i,:,:]) <= discard_value:
            tmp2[idx,i,:,:] = 0

print(f"{c_}Dice Score Before Discarding So small pixel values: {Fore.BLUE}{dice_coef(all_true, tmp).item():.4f}")
print(f"{c_}Dice Score After Discarding So small pixel values: {Fore.BLUE}{dice_coef(all_true, tmp2).item():.4f}")


### Debuging Mode

all_true = all_true[:1000]; all_pred = all_pred[:1000]


def dice_coef_channel(y_true, y_pred, thr1=0.5, thr2=0.5, thr3=0.5, dim=(2,3), epsilon=1e-9):
    y_true = y_true.to(torch.float32)
    y_pred[:,0] = (y_pred[:,0]>thr1).to(torch.float32) # channel 0: Large Bowel
    y_pred[:,1] = (y_pred[:,1]>thr2).to(torch.float32) # channel 1: Small Bowel
    y_pred[:,2] = (y_pred[:,2]>thr3).to(torch.float32) # channel 2: Stomache
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(0))
    
    return dice

def find_optimal_threshold(y_true, y_pred, steps=50):

    trial_x = [[],[],[]]; trial_y = [[],[],[]]
    threshold = [0.5,0.5,0.5]

    origin = dice_coef_channel(y_true, y_pred, thr1=threshold[0], thr2=threshold[1], thr3=threshold[2])
    best = origin.clone()
    
    for k, label in enumerate(CFG.class_names):
        print('#' * 25)
        print(f'### Caculating {label}')
        print('#' * 25)
        
        v = threshold[k]
        threshold2 = threshold.copy()

        for sign in [1,-1]:

            stop = 0
            while stop < steps:

                v += sign*0.001
                threshold2[k] = v
                metric = dice_coef_channel(y_true, y_pred, thr1=threshold2[0], thr2=threshold2[1], thr3=threshold2[2])

                trial_x[k].append(v); trial_y[k].append(metric[k])
                
                if metric[k] > best[k]:
                    best[k] = metric[k]
                    threshold = threshold2.copy()
                else:
                    stop += 1
                    
    return origin, best, trial_x, trial_y, threshold


origin, best, trial_x, trial_y, threshold = find_optimal_threshold(all_true, all_pred, steps=100)

print(f'\n{c_}Optimal Threshold in Large Bowel is {threshold[0]}')
print(f'=> Best Dice Score is in Large Bowel: {origin[0]:.4f} -> {best[0]:.4f}\n')

print(f'{c_}=> Optimal Threshold in Small Bowel is {threshold[1]}')
print(f'=> Best Dice Score is in Small Bowel: {origin[1]:.4f} -> {best[1]:.4f}\n')

print(f'{c_}=> Optimal Threshold in Stomache is {threshold[2]}')
print(f'=> Best Dice Score is in Stomache: {origin[2]:.4f} -> {best[2]:.4f}\n')



fig, axes = plt.subplots(1,3, figsize=(12,4))

for i, label in enumerate(CFG.class_names):
    axes[i].scatter(trial_x[i], trial_y[i])
    axes[i].set_title(f'Threshold in {label}', fontsize=10, fontweight='bold')
    axes[i].set_xlabel("Threshold"); axes[i].set_ylabel("Dice Coefficient")
    mn = np.min(trial_y[i])
    mx = np.max(trial_y[i])
    axes[i].set_ylim([mn,mx])

    
    axes[i].plot([threshold[i], threshold[i]], [mn,mx], '--', label='Optimal Threshold')
    axes[i].legend()

plt.tight_layout()
plt.show()

