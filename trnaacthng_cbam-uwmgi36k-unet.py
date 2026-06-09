!pip install -q segmentation_models_pytorch
!pip install rasterio
# !pip install -q scikit-learn==1.0


%load_ext autoreload
%autoreload 2


import numpy as np
import pandas as pd
pd.options.plotting.backend = "plotly"
import random
from glob import glob
import os, shutil
from tqdm import tqdm
tqdm.pandas()
import time
import copy
import joblib
from collections import defaultdict
import gc
from IPython import display as ipd

# visualization
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Sklearn
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

# PyTorch 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

import timm

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

import rasterio
from joblib import Parallel, delayed

# For colored terminal text
from colorama import Fore, Back, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class CFG:
    seed          = 101
    debug         = False # set debug=False for Full Training
    exp_name      = '2.5D'
    comment       = 'unet-efficientnet_b0-160x192-ep=5'
    model_name    = 'Unet'
    backbone      = 'efficientnet-b2'
    train_bs      = 32
    valid_bs      = 32
    img_size      = [512, 512]
    epochs        = 50
    lr            = 2e-3
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
    T_max         = int(30000/train_bs*epochs)+50
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-6
    n_accumulate  = max(1, 32//train_bs)
    n_fold        = 5
    folds         = [0]
    num_classes   = 3
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


path_df = pd.DataFrame(glob('/kaggle/input/uwmgi-25d-stride2-dataset/images/images/*'), columns=['image_path'])
path_df['mask_path'] = path_df.image_path.str.replace('image','mask')
path_df['id'] = path_df.image_path.map(lambda x: x.split('/')[-1].replace('.npy',''))
path_df.head()


df = pd.read_csv('/kaggle/input/uwmgi-dataset-folds5/train_folds.csv')
df.head()


df["image_path"][0]


df['empty'].value_counts().plot.bar()


def load_img(path):
    img = np.load(path)
    img = img.astype('float32') # original is uint16
    mx = np.max(img)
    if mx:
        img/=mx # scale image to [0, 1]
    return img

def load_msk(path):
    msk = np.load(path)
    msk = msk.astype('float32')
    msk/=255.0
    return msk
    

def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     img = clahe.apply(img)
#     plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='bone')
    
    if mask is not None:
        # plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles,labels)
    plt.axis('off')


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df         = df
        self.label      = label
        self.img_paths  = df['image_path'].tolist()
        self.msk_paths  = df['mask_path'].tolist()
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path  = self.img_paths[index]
        img = []
        img = load_img(img_path)
        
        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img  = data['image']
                msk  = data['mask']
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)


# data_transforms = {
#     "train": A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
#         A.OneOf([
#             A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
#             A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
#         ], p=0.25),
#         A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20,
#                          min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
#         ], p=1.0),
    
#     "valid": A.Compose([], p=1.0)
# }


data_transforms = {
    "train": A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=(0, 0.1), rotate_limit=20, border_mode=1, p=0.85),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.1, border_mode=1, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, border_mode=1, p=0.5)
        ], p=0.2),
        A.OneOf([
            A.GaussNoise(var_limit=(0.0001, 0.004), p=0.7),
            A.Blur(blur_limit=3, p=0.3)
        ], p=0.5),
        
        A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20,
                         min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0),
        
    "valid": A.Compose([], p=1.0)
}


fold = 0
train_df = df.query("fold!=@fold").reset_index(drop=True)
train_df.shape
valid_df = df.query("fold==@fold").reset_index(drop=True)
valid_df.shape


def prepare_loaders(fold, debug=False):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    if debug:
        train_df = train_df.head(32*5).query("empty==0")
        valid_df = valid_df.head(32*3).query("empty==0")
    train_dataset = BuildDataset(train_df, transforms=data_transforms['train'])
    valid_dataset = BuildDataset(valid_df, transforms=data_transforms['valid'])

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs if not debug else 20, 
                              num_workers=4, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs if not debug else 20, 
                              num_workers=4, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader



train_df


train_dataset = BuildDataset(train_df, transforms=data_transforms['train'])
train_dataset


def plot_batch(imgs, msks, size=3):
    plt.figure(figsize=(5*5, 5))
    for idx in range(size):
        plt.subplot(1, 5, idx+1)
        img = imgs[idx,].permute((1, 2, 0)).numpy()*255.0
        img = img.astype('uint8')
        msk = msks[idx,].permute((1, 2, 0)).numpy()*255.0
        show_img(img, msk)
    plt.tight_layout()
    plt.show()


# plot_batch(imgs, msks, size=5)


import gc
gc.collect()


import torch
from torch import nn


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            indexing='xy')
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride),
            indexing='xy')
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset


import os
import cv2
import torch
import numpy as np
import albumentations as A
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

channel_means = [0.598, 0.584, 0.565]
channel_stds  = [0.104, 0.103, 0.103]

def Bextraction(x):
    # 1) Convert to numpy array
    if isinstance(x, torch.Tensor):
        arr = x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        raise TypeError(f"Bextraction only supports Tensor or ndarray, not {type(x)}")

    # 2) Normalize to shape (C, H, W)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, 0)        # â†’ (1, H, W)
    elif arr.ndim == 3 and arr.shape[0] not in (1, 3) and arr.shape[2] in (1,3):
        # e.g. HÃ—WÃ—C case
        arr = arr.transpose(2, 0, 1)        # â†’ (C, H, W)
    # now arr is (C, H, W) for Câ‰¥1

    # 3) Prepare output
    C, H, W = arr.shape
    boundary = np.zeros_like(arr, dtype=arr.dtype)

    # diamond kernel
    DIAMOND_KERNEL_5 = np.array([
        [0,0,1,0,0],
        [0,1,1,1,0],
        [1,1,1,1,1],
        [0,1,1,1,0],
        [0,0,1,0,0],
    ], dtype=np.uint8)

    # 4) Extract per-channel edge = dilate(channel) - channel
    for c in range(C):
        channel = arr[c].astype(np.uint8)
        dilated = cv2.dilate(channel, DIAMOND_KERNEL_5)
        edge    = dilated.astype(arr.dtype) - channel
        boundary[c] = edge

    # 5) Return as float Tensor
    return torch.from_numpy(boundary).float()

class ImgToTensor(object):
    def __call__(self, img):
        tf = transforms.Compose([transforms.ToTensor(),                                                                                    
                                 transforms.Normalize(channel_means, channel_stds)])
        return tf(img)


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img).long()


class Crack_loader(Dataset):
    """ dataset class for Crack datasets
    """
    
    def __init__(self, img_dir, img_fnames, mask_dir, mask_fnames, isTrain=False, resize=False):
        self.img_dir = img_dir
        self.img_fnames = img_fnames

        self.mask_dir = mask_dir
        self.mask_fnames = mask_fnames

        self.resize  = resize
        self.isTrain = isTrain

        self.aug = A.Compose([
                            A.augmentations.crops.transforms.RandomResizedCrop(256,256,p=0.5),
                            A.augmentations.MotionBlur(p=0.1),
                            A.augmentations.transforms.ColorJitter(),
                            A.augmentations.geometric.rotate.SafeRotate(),
                            A.HorizontalFlip(),
                            A.VerticalFlip(),
                            A.augmentations.geometric.rotate.RandomRotate90(p=0.5)
                            ])

        self.img_totensor  = ImgToTensor()

        self.mask_totensor = MaskToTensor()
                
    def __getitem__(self, i):
        # read a image given a random integer index
        fname = self.img_fnames[i]
        fpath = os.path.join(self.img_dir, fname)
        img = cv2.imread(fpath) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                    # H,W,3 np.uint8

        mname = self.mask_fnames[i]
        mpath = os.path.join(self.mask_dir, mname)
        mask = cv2.imread(mpath, cv2.COLOR_BGR2GRAY)                                  # H,W, np.uint8

        if self.isTrain:
            img  = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)         # (256,256,3) np.uint8
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_CUBIC)        # (256,256) np.uint8

            # image augmentation     
            transformed = self.aug(image=img, mask=mask)
            img  = transformed['image']                                               # (256,256,3) np.uint8
            mask = transformed['mask']                                                # (256,256) np.uint8            
            
            # binarize segmentation
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        
            # totensor
            img  = self.img_totensor(Image.fromarray(img.copy()))
            mask = self.mask_totensor(mask.copy()).unsqueeze(0)
        
            # extract boundary
            boundary = Bextraction(mask)                                              # (1,256,256) torch.float32

            return {'image': img,
                    'mask': mask,
                    'boundary': boundary}

        else:
            if self.resize:
                img  = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)     # (256,256,3) np.uint8
                mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_CUBIC)    # (256,256) np.uint8
                img  = self.img_totensor(Image.fromarray(img.copy()))
                
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
            mask = self.mask_totensor(mask.copy()).unsqueeze(0)

            return {'image': img,
                    'mask': mask,
                    'img_path': fpath}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_fnames)


class BuildBoundaryDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None , isTrain=False):
        self.df         = df
        self.label      = label
        self.img_paths  = df['image_path'].tolist()
        self.msk_paths  = df['mask_path'].tolist()
        self.transforms = transforms
        self.isTrain = isTrain
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path  = self.img_paths[index]
        img = []
        img = load_img(img_path)

        if self.isTrain:
            if self.label:
                msk_path = self.msk_paths[index]
                msk = load_msk(msk_path)
                if self.transforms:
                    data = self.transforms(image=img, mask=msk)
                    img  = data['image']
                    msk  = data['mask']
                img = np.transpose(img, (2, 0, 1))
                msk = np.transpose(msk, (2, 0, 1))
                boundary   = Bextraction(msk)             # â†’ torch.Size([3, H, W])        
                
                return torch.tensor(img), torch.tensor(msk) , torch.tensor(boundary)
            else:
                if self.transforms:
                    data = self.transforms(image=img)
                    img  = data['image']
                img = np.transpose(img, (2, 0, 1))
                return torch.tensor(img)


# def prepare_loaders_boundary(fold, debug=False , isTrain=True):
#     train_df = df.query("fold!=@fold").reset_index(drop=True)
#     valid_df = df.query("fold==@fold").reset_index(drop=True)
#     if debug:
#         train_df = train_df.head(32*5).query("empty==0")
#         valid_df = valid_df.head(32*3).query("empty==0")
#     train_dataset = BuildBoundaryDataset(train_df, transforms=data_transforms['train'] , isTrain=True)
#     valid_dataset = BuildBoundaryDataset(valid_df, transforms=data_transforms['valid'] , isTrain=True)

#     train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs if not debug else 20, 
#                               num_workers=4, shuffle=True, pin_memory=True, drop_last=False)
#     valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs if not debug else 20, 
#                               num_workers=4, shuffle=False, pin_memory=True)
    
#     return train_loader, valid_loader



def prepare_loaders(fold, debug=False):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    if debug:
        train_df = train_df.head(32*5).query("empty==0")
        valid_df = valid_df.head(32*3).query("empty==0")
    train_dataset = BuildDataset(train_df, transforms=data_transforms['train'])
    valid_dataset = BuildDataset(valid_df, transforms=data_transforms['valid'])

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs if not debug else 20, 
                              num_workers=4, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs if not debug else 20, 
                              num_workers=4, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader



train_df


valid_df


import torch
import torch.nn as nn

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None

class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)


class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)


class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from typing import Optional, Sequence, List
# from segmentation_models_pytorch.base import modules as md


# class UnetDecoderBlock(nn.Module):
#     """A decoder block in the U-Net architecture that performs upsampling and feature fusion."""

#     def __init__(
#         self,
#         in_channels: int,
#         skip_channels: int,
#         out_channels: int,
#         use_batchnorm: bool = True,
#         attention_type: Optional[str] = None,
#         interpolation_mode: str = "nearest",
#     ):
#         super().__init__()
#         self.interpolation_mode = interpolation_mode
#         self.conv1 = Conv2dReLU(
#             in_channels + skip_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.attention1 = md.Attention(
#             attention_type, in_channels=in_channels + skip_channels
#         )
#         self.conv2 = Conv2dReLU(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.attention2 = md.Attention(attention_type, in_channels=out_channels)

#     def forward(
#         self,
#         feature_map: torch.Tensor,
#         target_height: int,
#         target_width: int,
#         skip_connection: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         feature_map = F.interpolate(
#             feature_map,
#             size=(target_height, target_width),
#             mode=self.interpolation_mode,
#         )
#         if skip_connection is not None:
#             feature_map = torch.cat([feature_map, skip_connection], dim=1)
#             feature_map = self.attention1(feature_map)
#         feature_map = self.conv1(feature_map)
#         feature_map = self.conv2(feature_map)
#         feature_map = self.attention2(feature_map)
#         return feature_map


# class UnetCenterBlock(nn.Sequential):
#     """Center block of the Unet decoder. Applied to the last feature map of the encoder."""

#     def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
#         conv1 = Conv2dReLU(
#             in_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         conv2 = Conv2dReLU(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         super().__init__(conv1, conv2)


# class UnetDecoder(nn.Module):
#     """The decoder part of the U-Net architecture.

#     Takes encoded features from different stages of the encoder and progressively upsamples them while
#     combining with skip connections. This helps preserve fine-grained details in the final segmentation.
#     """

#     def __init__(
#         self,
#         encoder_channels: Sequence[int],
#         decoder_channels: Sequence[int],
#         n_blocks: int = 5,
#         use_batchnorm: bool = True,
#         attention_type: Optional[str] = None,
#         add_center_block: bool = False,
#         interpolation_mode: str = "nearest",
#     ):
#         super().__init__()

#         if n_blocks != len(decoder_channels):
#             raise ValueError(
#                 "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
#                     n_blocks, len(decoder_channels)
#                 )
#             )

#         # remove first skip with same spatial resolution
#         encoder_channels = encoder_channels[1:]
#         # reverse channels to start from head of encoder
#         encoder_channels = encoder_channels[::-1]

#         # computing blocks input and output channels
#         head_channels = encoder_channels[0]
#         in_channels = [head_channels] + list(decoder_channels[:-1])
#         skip_channels = list(encoder_channels[1:]) + [0]
#         out_channels = decoder_channels

#         if add_center_block:
#             self.center = UnetCenterBlock(
#                 head_channels, head_channels, use_batchnorm=use_batchnorm
#             )
#         else:
#             self.center = nn.Identity()

#         # combine decoder keyword arguments
#         self.blocks = nn.ModuleList()
#         for block_in_channels, block_skip_channels, block_out_channels in zip(
#             in_channels, skip_channels, out_channels
#         ):
#             block = UnetDecoderBlock(
#                 block_in_channels,
#                 block_skip_channels,
#                 block_out_channels,
#                 use_batchnorm=use_batchnorm,
#                 attention_type=attention_type,
#                 interpolation_mode=interpolation_mode,
#             )
#             self.blocks.append(block)

#     def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
#         # spatial shapes of features: [hw, hw/2, hw/4, hw/8, ...]
#         spatial_shapes = [feature.shape[2:] for feature in features]
#         spatial_shapes = spatial_shapes[::-1]

#         features = features[1:]  # remove first skip with same spatial resolution
#         features = features[::-1]  # reverse channels to start from head of encoder

#         head = features[0]
#         skip_connections = features[1:]

#         x = self.center(head)

#         for i, decoder_block in enumerate(self.blocks):
#             # upsample to the next spatial shape
#             height, width = spatial_shapes[i + 1]
#             skip_connection = skip_connections[i] if i < len(skip_connections) else None
#             x = decoder_block(x, height, width, skip_connection=skip_connection)

#         return x



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Sequence, List
from segmentation_models_pytorch.base import modules as md

#########################################
# CBAM Implementation in PyTorch
#########################################

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        return self.sigmoid(out) * x

class CBAM(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 8, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class UnetDecoderBlock(nn.Module):
    """A decoder block in the U-Net architecture that performs upsampling and feature fusion."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        attention_type: Optional[str] = None,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.cbam = CBAM(in_channels + skip_channels, ratio=8, kernel_size=7)
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(
        self,
        feature_map: torch.Tensor,
        target_height: int,
        target_width: int,
        skip_connection: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feature_map = F.interpolate(
            feature_map,
            size=(target_height, target_width),
            mode=self.interpolation_mode,
        )
        if skip_connection is not None:
            feature_map = torch.cat([feature_map, skip_connection], dim=1)
            feature_map = self.attention1(feature_map)
            feature_map = self.cbam(feature_map)
        feature_map = self.conv1(feature_map)
        feature_map = self.conv2(feature_map)
        feature_map = self.attention2(feature_map)
        return feature_map


class UnetCenterBlock(nn.Sequential):
    """Center block of the Unet decoder. Applied to the last feature map of the encoder."""

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    """The decoder part of the U-Net architecture.

    Takes encoded features from different stages of the encoder and progressively upsamples them while
    combining with skip connections. This helps preserve fine-grained details in the final segmentation.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        n_blocks: int = 5,
        use_batchnorm: bool = True,
        attention_type: Optional[str] = None,
        add_center_block: bool = False,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if add_center_block:
            self.center = UnetCenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        self.blocks = nn.ModuleList()
        for block_in_channels, block_skip_channels, block_out_channels in zip(
            in_channels, skip_channels, out_channels
        ):
            block = UnetDecoderBlock(
                block_in_channels,
                block_skip_channels,
                block_out_channels,
                use_batchnorm=use_batchnorm,
                attention_type=attention_type,
                interpolation_mode=interpolation_mode,
            )
            self.blocks.append(block)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # spatial shapes of features: [hw, hw/2, hw/4, hw/8, ...]
        spatial_shapes = [feature.shape[2:] for feature in features]
        spatial_shapes = spatial_shapes[::-1]

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skip_connections = features[1:]

        x = self.center(head)

        for i, decoder_block in enumerate(self.blocks):
            # upsample to the next spatial shape
            height, width = spatial_shapes[i + 1]
            skip_connection = skip_connections[i] if i < len(skip_connections) else None
            x = decoder_block(x, height, width, skip_connection=skip_connection)

        return x



from typing import Any, Optional, Union, Callable, Sequence

from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading

class Unet(SegmentationModel):
    requires_divisible_input_shape = False

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        decoder_interpolation_mode: str = "nearest",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        add_center_block = encoder_name.startswith("vgg")
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            add_center_block=add_center_block,
            attention_type=decoder_attention_type,
            interpolation_mode=decoder_interpolation_mode,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()



# from typing import Any, Optional, Union, Callable, Sequence

# from segmentation_models_pytorch.base import (
#     ClassificationHead,
#     SegmentationHead,
#     SegmentationModel,
# )
# from segmentation_models_pytorch.encoders import get_encoder
# from segmentation_models_pytorch.base.hub_mixin import supports_config_loading

# class UnetBEM(SegmentationModel):

#     requires_divisible_input_shape = False

#     @supports_config_loading
#     def __init__(
#         self,
#         encoder_name: str = "resnet34",
#         encoder_depth: int = 5,
#         encoder_weights: Optional[str] = "imagenet",
#         decoder_use_batchnorm: bool = True,
#         decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
#         decoder_attention_type: Optional[str] = None,
#         decoder_interpolation_mode: str = "nearest",
#         in_channels: int = 3,
#         classes: int = 1,
#         activation: Optional[Union[str, Callable]] = None,
#         aux_params: Optional[dict] = None,
#         **kwargs: dict[str, Any],
#     ):
#         super().__init__()

#         self.encoder = get_encoder(
#             encoder_name,
#             in_channels=in_channels,
#             depth=encoder_depth,
#             weights=encoder_weights,
#             **kwargs,
#         )

#         self.boundary = nn.Sequential(
#             DeformConv2d(32, 32, modulation=True),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)
#         )
        
#         add_center_block = encoder_name.startswith("vgg")
#         self.decoder = UnetDecoder(
#             encoder_channels=self.encoder.out_channels,
#             decoder_channels=decoder_channels,
#             n_blocks=encoder_depth,
#             use_batchnorm=decoder_use_batchnorm,
#             add_center_block=add_center_block,
#             attention_type=decoder_attention_type,
#             interpolation_mode=decoder_interpolation_mode,
#         )

#         self.segmentation_head = SegmentationHead(
#             in_channels=decoder_channels[-1],
#             out_channels=classes,
#             activation=activation,
#             kernel_size=3,
#         )

#         if aux_params is not None:
#             self.classification_head = ClassificationHead(
#                 in_channels=self.encoder.out_channels[-1], **aux_params
#             )
#         else:
#             self.classification_head = None

#         self.name = "u-{}".format(encoder_name)
#         self.initialize()
        
#     def forward(self, x: torch.Tensor , istrain=False) -> torch.Tensor:
#         """
#         Forward pass:
#           1) Encoder
#           2) ASPP on last encoder feature
#           3) Decoder
#           4) (Optionally) ASPP on decoder's output
#           5) SEAM
#           6) Seg Head
#           7) (Optionally) Classification Head
#         """
#         # 1) Encode
#         features = self.encoder(x)  # List of multi-scale feature maps

#         # 2) BEM on deepest encoder feature

#         stage1 = features[-5]
#         print(stage1.shape)
#         B_out = self.boundary(stage1)
#         print(B_out.shape)
#         features[-5] = stage1 + B_out.repeat_interleave(int(stage1.shape[1]), dim=1)
#         B_out = F.interpolate(B_out, size=(320, 384), mode='bilinear', align_corners=True)

#         # 3) Decode
#         decoder_output = self.decoder(features)

#         # 6) Segmentation head
#         masks = self.segmentation_head(decoder_output)

#         # 7) Classification head (optional)
#         if self.classification_head is not None:
#             labels = self.classification_head(features[-1])
#             print("===============================================")
#             print("======================zzzzz====================")
#             print("===============================================")
#             return masks, labels
            
#         # if istrain == True:
#         #     print("===============================================")
#         #     print("======================xxxxx====================")
#         #     print("===============================================")
#         return masks, B_out
#         # else:
#         #     print("===============================================")
#         #     print("======================yyyyy====================")
#         #     print("===============================================")
#         #     return masks



import segmentation_models_pytorch as smp

def build_model():
    model = Unet(
        encoder_name=CFG.backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=CFG.num_classes,        # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to(CFG.device)
    return model

def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss    = smp.losses.DiceLoss(mode='multilabel')
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)

def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

def criterion(y_pred, y_true):
    return 0.5*BCELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    dataset_size = 0
    train_scores = []  # collect [dice, jaccard] per batch

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Train {epoch}')
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)

        optimizer.zero_grad()
        y_pred = model(images)
        loss   = criterion(y_pred, masks)
        loss.backward()
        optimizer.step()

        # accumulate loss
        batch_size    = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss    = running_loss / dataset_size

        # compute metrics
        y_prob        = torch.sigmoid(y_pred)
        train_dice    = dice_coef(masks, y_prob).cpu().detach().numpy()
        train_jaccard = iou_coef(masks, y_prob).cpu().detach().numpy()
        train_scores.append([train_dice, train_jaccard])

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:.4f}', lr=f'{current_lr:.5f}')

    # mean metrics over all batches
    train_scores_epoch = np.mean(train_scores, axis=0)  # [dice, jaccard]
    return epoch_loss, train_scores_epoch


# -------------------------------------------------------------------
# 2) VALIDATION EPOCH -----------------------------------------------
# -------------------------------------------------------------------
@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    val_scores   = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Valid {epoch}')
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)

        y_pred = model(images)
        loss   = criterion(y_pred, masks)

        batch_size    = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss    = running_loss / dataset_size

        y_prob        = torch.sigmoid(y_pred)
        val_dice      = dice_coef(masks, y_prob).cpu().detach().numpy()
        val_jaccard   = iou_coef(masks, y_prob).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])

        current_lr = optimizer.param_groups[0]['lr']
        mem        = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        pbar.set_postfix(valid_loss=f'{epoch_loss:.4f}',
                         lr=f'{current_lr:.5f}',
                         gpu_memory=f'{mem:.2f} GB')

    val_scores_epoch = np.mean(val_scores, axis=0)  # [dice, jaccard]
    torch.cuda.empty_cache()
    gc.collect()
    return epoch_loss, val_scores_epoch


import pickle

def run_training(model,
                 optimizer,
                 scheduler,
                 device,
                 num_epochs,
                 patience=5):
    """
    Runs training + validation with early stopping.
    :param patience: how many epochs to wait for improvement before stopping
    """
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name())

    start = time.time()
    best_model_wts   = copy.deepcopy(model.state_dict())
    best_dice        = -np.inf
    no_improve_epochs = 0
    history          = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')

        # ---- train ----
        train_loss, (train_dice, train_jaccard) = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )

        # ---- validate ----
        val_loss, (val_dice, val_jaccard) = valid_one_epoch(
            model, valid_loader, device, epoch
        )

        # ---- log to history ----
        history['Train Loss'].append(train_loss)
        history['Train Dice'].append(train_dice)
        history['Train Jaccard'].append(train_jaccard)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)

        print(f"  Train Dice: {train_dice:.4f} | Train Jaccard: {train_jaccard:.4f}")
        print(f"  Valid Dice: {val_dice:.4f} | Valid Jaccard: {val_jaccard:.4f}")

        # ---- check for improvement ----
        if val_dice > best_dice:
            print(f"  ğŸ�† Dice improved ({best_dice:.4f} â†’ {val_dice:.4f}), saving model.")
            best_dice        = val_dice
            best_model_wts   = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            torch.save(model.state_dict(), f"best_epoch-{fold:02d}.bin")
        else:
            no_improve_epochs += 1
            print(f"  âš ï¸�  No improvement for {no_improve_epochs}/{patience} epochs.")
            if no_improve_epochs >= patience:
                print(f"ğŸš¨ Early stopping triggered after {epoch} epochs.")
                break

        # ---- always save last ----
        torch.save(model.state_dict(), f"last_epoch-{fold:02d}.bin")

        # ---- scheduler step if used ----
        if scheduler is not None:
            scheduler.step()

    # ---- wrap up ----
    model.load_state_dict(best_model_wts)
    elapsed = time.time() - start
    h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
    print(f"\nTraining complete in {h}h {m}m")
    print(f"Best Validation Dice: {best_dice:.4f}")

    # record total training time
    history['Total Training Time (s)'].append(elapsed)

    # Save history
    with open("history.pkl", "wb") as f:
        import pickle
        pickle.dump(history, f)

    return model, history



def fetch_scheduler(optimizer):
    if CFG.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG.T_max, 
                                                   eta_min=CFG.min_lr)
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CFG.T_0, 
                                                             eta_min=CFG.min_lr)
    elif CFG.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=CFG.min_lr,)
    elif CFG.scheduer == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif CFG.scheduler == None:
        return None
        
    return scheduler


model = build_model()
optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
scheduler = fetch_scheduler(optimizer)


for fold in CFG.folds: ## 0,1
    print(f'#'*15)
    print(f'### Fold: {fold}')
    print(f'#'*15)

    train_loader, valid_loader = prepare_loaders(fold=fold, debug=CFG.debug)
    model     = build_model()
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    scheduler = fetch_scheduler(optimizer)
    model, history = run_training(model, optimizer, scheduler,
                                  device=CFG.device,
                                  num_epochs=CFG.epochs)



test_dataset = BuildDataset(df.query("fold==0 & empty==0").sample(frac=1.0), label=False, 
                            transforms=data_transforms['valid'])
test_loader  = DataLoader(test_dataset, batch_size=5, 
                          num_workers=4, shuffle=False, pin_memory=True)
imgs = next(iter(test_loader))
imgs = imgs.to(CFG.device, dtype=torch.float)

preds = []
for fold in CFG.folds:
    model = load_model(f"best_epoch-{fold:02d}.bin")
    with torch.no_grad():
        pred = model(imgs)
        pred = (nn.Sigmoid()(pred)>0.5).double()
    preds.append(pred)
    
imgs  = imgs.cpu().detach()
preds = torch.mean(torch.stack(preds, dim=0), dim=0).cpu().detach()


plot_batch(imgs, preds, size=5)







