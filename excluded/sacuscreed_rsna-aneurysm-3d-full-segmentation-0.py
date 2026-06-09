!pip install segmentation_models_pytorch==0.3.3

import os
import gc
import random
import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


FOLDS = [0]
SEED = 777
SIZE = 128
WSIZE = 256
RADIUS = 16
ANGLE_AUG = 30
EPOCHS = 12
BS = 4
LR = 1e-5
WORKERS = 4
ROTATION_PROB_DECAY = .5699 # (1 - p)**2 = p**3 | Equals single and triple rotations during augmentations


label_columns = [
    'Other Posterior Circulation',
    'Basilar Tip',
    'Right Posterior Communicating Artery',
    'Left Posterior Communicating Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Infraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Middle Cerebral Artery',
    'Left Middle Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Anterior Cerebral Artery',
    'Anterior Communicating Artery'
]


source_path = '/kaggle/input/rsna-raw-roi/NPZ/'
cases = [c[:-4] for c in os.listdir(source_path )]
print(len(cases))
cases[:5]


train = pd.read_csv('/kaggle/input/rsna-2d-binary-segmentation-preprocessing/rsna_train_folds.csv')
roi = pd.read_csv('/kaggle/input/rsna-raw-roi/rsna_roi.csv')
train = roi[roi.case.apply(lambda v: v in cases)].merge(train,left_on='case',right_on='SeriesInstanceUID').reset_index(drop=True)
train.tail()


train['flip'] = False
ftrain = train.copy()
ftrain['flip'] = True
train = pd.concat([train,ftrain]).reset_index(drop=True)
train.tail()


class RSNA_Dataset_3D(Dataset):
    def __init__(
        self,
        df,
        VALID=False
    ):
        """
        A PyTorch Dataset for 3D semantic segmentation of intracranial arteries from angiographic scans.

        This dataset handles preprocessed 3D volumes from the RSNA challenge. The task involves multi-class
        segmentation to identify and label 13 individual intracranial artery segments (classes 1-13) against
        the background (class 0).

        The data has been preprocessed from original NIfTI files into 3D numpy arrays:
        - Images: Angiographic scans (e.g., CTA or MRA) representing vascular intensity
        - Labels: Multi-class masks where each voxel is assigned a class (0-13)

        The dataset performs extensive on-the-fly augmentation during training, including:
        - Random asymmetric zoom and cropping around each class centroid (for the specified target)
        - 3D rotation and flipping (with symmetric label remapping for paired vessels)
        - Intensity adjustments (inversion, contrast, brightness, noise)

        For validation, a deterministic center-crop around the target centroid is used for consistent evaluation.

        Args:
            cases: List of case identifiers (SeriesInstanceUID)
            targets: The specific target class label around which to center the crop
            pmin: Minimum intensity value for normalization per case
            pmax: Maximum intensity value for normalization per case
            flip: Boolean flag indicating whether to apply left-right flipping
            d0, h0, w0: Original segmentation mask offset coordinates in the full volume
            d, h, w: Original segmentation mask dimensions in the full volume
            VALID: If True, disables augmentation for validation/inference mode
        """
        self.cases = df['SeriesInstanceUID']
        self.pmin = df['pmin']
        self.pmax = df['pmax']
        self.flip = df['flip']
        self.axis = df['axis']
        self.s = df['s']
        self.VALID = VALID

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):        
        case = self.cases[idx]
        pmin = self.pmin[idx]
        pmax = self.pmax[idx]
        axis = self.axis[idx]
        s = self.s[idx]/4
        f = self.flip[idx]

        npz =  np.load(source_path + case + '.npz')
        img = npz['volume']
        D,H,W = img.shape
        t = npz['t']
        l = npz['loc']
        AP = npz['AP']
        positive = sum(AP) > 0
        if positive:
            t = t[AP > 0]
            l = l[AP > 0]
        else:
            if len(l) > 1:
                t = t[l > 0]
                l = l[l > 0]

        if not self.VALID:
            ul = np.unique(l)
            k = ul[np.random.randint(len(ul))]
            kk = np.arange(len(l))[l == k]
            kk = kk[np.random.randint(len(kk))]
            d,h,w = np.rint(t[kk]).astype(int)
#           Random asymmetric zoom
            sd = s*(.9 + 0.2*np.random.rand())
            sh = s*(.9 + 0.2*np.random.rand())
            sw = s*(.9 + 0.2*np.random.rand())

            rd = np.rint(RADIUS*sd/SIZE).astype(int)
            rh = np.rint(RADIUS*sh/SIZE).astype(int)
            rw = np.rint(RADIUS*sw/SIZE).astype(int)

            dd = np.rint(sd).astype(int)
            hh = np.rint(sh).astype(int)
            ww = np.rint(sw).astype(int)

            ddd = np.rint(sd/2).astype(int)
            hhh = np.rint(sh/2).astype(int)
            www = np.rint(sw/2).astype(int)
            
            d0 = d - ddd//2 - rd - np.random.randint(dd - 2*rd)
            h0 = h - hhh//2 - rh - np.random.randint(hh - 2*rh)
            w0 = w - www//2 - rw - np.random.randint(ww - 2*rw)

            d = d0 + dd + ddd
            h = h0 + hh + hhh
            w = w0 + ww + www

            pad_d0 = max(0, -d0)
            pad_h0 = max(0, -h0)
            pad_w0 = max(0, -w0)
        
            pad_d = max(0, d - D)
            pad_h = max(0, h - H)
            pad_w = max(0, w - W)
        
            d0 = max(0, d0)
            h0 = max(0, h0)
            w0 = max(0, w0)
        
            d = min(D, d)
            h = min(H, h)
            w = min(W, w)
                
            img = torch.nn.functional.pad(
                (torch.from_numpy(img[d0:d,h0:h,w0:w]) - pmin)/(pmax - pmin),
                (pad_w0,pad_w,pad_h0,pad_h,pad_d0,pad_d)
            )
#           Free rotation
            p = 1
            t[:,0] += pad_d0 - d0
            t[:,1] += pad_h0 - h0
            t[:,2] += pad_w0 - w0
            axis_perm = [
                [2,0,1],
                [1,2,0]
            ][np.random.randint(2)]
            for _ in range(3):
                if np.random.rand() < p:
                    c = list(t[kk,1:][::-1])
                    angle = 2*ANGLE_AUG*(np.random.rand() - .5)
                    img = transforms.functional.rotate(
                        img,
                        angle,
                        transforms.InterpolationMode.BILINEAR,
                        center=c
                    )
#                   Convert angle to radians
                    angle_rad = math.radians(angle)
                    cos_angle = math.cos(angle_rad)
                    sin_angle = math.sin(angle_rad)
                    py = t[:,2]
                    px = t[:,1]
                    cy, cx = c
#                   Translate point to origin
                    translated_x = px - cx
                    translated_y = py - cy    
#                   Apply rotation
                    rotated_x = translated_x * cos_angle - translated_y * sin_angle
                    rotated_y = translated_x * sin_angle + translated_y * cos_angle    
#                   Translate back
                    t[:,1] = rotated_x + cx
                    t[:,2] = rotated_y + cy
                    p *= ROTATION_PROB_DECAY
#               Rotate axis
                img = img.permute(axis_perm)
                t[:] = t[:,axis_perm]

            img = img[
                ddd//2:dd+ddd//2,
                hhh//2:hh+hhh//2,
                www//2:ww+www//2
            ]
            t[:,0] -= ddd//2
            t[:,1] -= hhh//2
            t[:,2] -= www//2

        else:
            d,h,w = np.rint(t[idx%len(t)]).astype(int)
            dd = hh = ww = np.rint(s).astype(int)

            d0 = d - dd//2
            h0 = h - hh//2
            w0 = w - ww//2

            d = d0 + dd
            h = h0 + hh
            w = w0 + ww

            pad_d0 = max(0, -d0)
            pad_h0 = max(0, -h0)
            pad_w0 = max(0, -w0)
        
            pad_d = max(0, d - D)
            pad_h = max(0, h - H)
            pad_w = max(0, w - W)
        
            d0 = max(0, d0)
            h0 = max(0, h0)
            w0 = max(0, w0)
        
            d = min(D, d)
            h = min(H, h)
            w = min(W, w)
                
            img = torch.nn.functional.pad(
                (torch.from_numpy(img[d0:d,h0:h,w0:w]) - pmin)/(pmax - pmin),
                (pad_w0,pad_w,pad_h0,pad_h,pad_d0,pad_d)
            )
            t[:,0] += pad_d0 - d0
            t[:,1] += pad_h0 - h0
            t[:,2] += pad_w0 - w0
        img = F.interpolate(
            img.unsqueeze(0).unsqueeze(0),
            size=(SIZE,SIZE,SIZE),
            mode='trilinear',
            align_corners=False
        )[0]
        t[:,0] *= SIZE/dd
        t[:,1] *= SIZE/hh
        t[:,2] *= SIZE/ww

        if not self.VALID:
#           Intensity Inversion
            if np.random.rand() < .1:
                img = 1 - img
#           Contrast
            if np.random.rand() < .5:
                x = .9 + .2*torch.rand(1)
                img *= x - (x - 1)/2
#           Brightness
            if np.random.rand() < .5:
                img += .2*torch.rand(1) - .1
#           Gaussian Noise
            if np.random.rand() < .5:
                img += torch.normal(torch.tensor(0.),torch.tensor(.05),(1,SIZE,SIZE,SIZE))
#       Flip            
        if f:
            for k in range(3,12,2):
                mr = l == k
                ml = l == k + 1
                l[mr] = k + 1
                l[ml] = k
#       Mask generation
        msk = torch.zeros(SIZE,SIZE,SIZE)
        if positive:
            r = np.indices((SIZE,SIZE,SIZE)).reshape(1,3,SIZE,SIZE,SIZE) - t.reshape(*t.shape,1,1,1)
            r = np.sqrt((r*r).sum(1))
            m = r < RADIUS
            mm = r.argmin(0)
            for k in range(len(l)): msk[m[k]*(mm == k)] = l[k]
#       Reorientation to axial perspective            
        if axis == 0:
            img = torch.rot90(torch.rot90(img,1,(-3,-2)),-1,(-2,-1))
            msk = torch.rot90(torch.rot90(msk,1,(-3,-2)),-1,(-2,-1))
        if axis == 1:
            img = torch.rot90(img,1,(-3,-2))
            msk = torch.rot90(msk,1,(-3,-2))
#       Flip            
        if f:
            img = img.flip(-1)
            msk = msk.flip(-1)
        
        return img, msk.long()


ds = RSNA_Dataset_3D(train)


img,msk = ds.__getitem__(np.random.randint(len(ds)))
YX = msk.max(0)[0]
ZX = msk.max(1)[0]
ZY = msk.max(2)[0]
YX[0,:14] = ZX[0,:14] = ZY[0,:14] = torch.arange(14)
_, axs = plt.subplots(2, 3)
axs[0,0].imshow(img[0].max(0)[0])
axs[0,1].imshow(img[0].max(1)[0])
axs[0,2].imshow(img[0].max(2)[0])
axs[1,0].imshow(YX,cmap='turbo')
axs[1,1].imshow(ZX,cmap='turbo')
axs[1,2].imshow(ZY,cmap='turbo')
plt.show()
plt.imshow(np.arange(14).reshape(1,14),cmap='turbo')
plt.gca().get_yaxis().set_visible(False)
plt.xticks(ticks=np.arange(14), labels=np.arange(14))
plt.show()


del ds
gc.collect()


weights = torch.from_numpy(train[label_columns].values.sum(0)).float()
weights[[2,3,4,5,6,7,8,9,10,11]] = weights[[2,3,4,5,6,7,8,9,10,11]].view(-1,2).sum(-1).view(5,1).tile(1,2).view(-1)/2
weights = torch.cat([torch.tensor([len(train)]),weights])
weights = len(train)/weights
wmin = weights.min()
wmax = weights.max()
weights = 9 * (weights - wmin) / (wmax - wmin) +  1
print(f'Class weights: {[np.round(w.item(),3) for w in weights]}')
ds = RSNA_Dataset_3D(train, VALID=True)


img,msk = ds.__getitem__(np.random.randint(len(ds)))
YX = msk.max(0)[0]
ZX = msk.max(1)[0]
ZY = msk.max(2)[0]
YX[0,:14] = ZX[0,:14] = ZY[0,:14] = torch.arange(14)
_, axs = plt.subplots(2, 3)
axs[0,0].imshow(img[0].max(0)[0])
axs[0,1].imshow(img[0].max(1)[0])
axs[0,2].imshow(img[0].max(2)[0])
axs[1,0].imshow(YX,cmap='turbo')
axs[1,1].imshow(ZX,cmap='turbo')
axs[1,2].imshow(ZY,cmap='turbo')
plt.show()
plt.imshow(np.arange(14).reshape(1,14),cmap='turbo')
plt.gca().get_yaxis().set_visible(False)
plt.xticks(ticks=np.arange(14), labels=np.arange(14))
plt.show()


del ds
gc.collect()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# https://github.com/shuaizzZ/Dice-Loss-PyTorch/blob/master/dice_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    """Dice Loss PyTorch
        Created by: Zhang Shuai
        Email: shuaizzz666@gmail.com
        dice_loss = 1 - 2*p*t / (p^2 + t^2). p and t represent predict and target.
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight) # Normalized weight
        self.smooth = 1e-5

    def forward(self, predict, target):
        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1) # (N, C, *)
        target = target.view(N, 1, -1) # (N, 1, *)

        predict = F.softmax(predict, dim=1) # (N, C, *) ==> (N, C, *)
        ## convert target(N, 1, *) into one hot vector (N, C, *)
        target_onehot = torch.zeros(predict.size()).cuda()  # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1)  # (N, C, *)

        intersection = torch.sum(predict * target_onehot, dim=2)  # (N, C)
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)  # (N, C)
        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        if hasattr(self, 'weight'):
            if self.weight.type() != predict.type():
                self.weight = self.weight.type_as(predict)
            dice_coef = dice_coef * self.weight * C  # (N, C)
        dice_loss = 1 - torch.mean(dice_coef)  # 1

        return dice_loss


# https://github.com/AdeelH/pytorch-multi-class-focal-loss
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cpu',
               dtype=torch.float32) -> FocalLoss:
    """Factory function for FocalLoss.

    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.

    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl


class DiceFocalLoss(nn.Module):
    def __init__(self, weights=None, gamma=1.0, alpha=0.5):
        super().__init__()
        self.DL = DiceLoss(weight=weights)
        self.FL = FocalLoss(alpha=weights,gamma=gamma)
        self.alpha = alpha

    def forward(self, inputs, targets):
        DL = self.DL(inputs,targets)
        FL = self.FL(inputs,targets)

        return self.alpha * DL + (1 - self.alpha) * FL


def convert_2d_to_3d(model, is_top_level=True):
    for name, module in model.named_children():
        # Recursively convert child modules first
        convert_2d_to_3d(module, False)

        # Replace Conv2d with Conv3d
        if isinstance(module, nn.Conv2d):
            # Handle cases where kernel_size/stride/padding are ints (not tuples)
            kernel_size = module.kernel_size[0]
            stride = module.stride[0]
            padding = module.padding[0]

            # New Conv3d layer with expanded kernel
            new_conv = nn.Conv3d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            )
            
            # Initialize weights: tile 2D weights along depth and average
            weight_2d = module.weight.data
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, kernel_size, 1, 1) / kernel_size
            new_conv.weight.data = weight_3d
            
            setattr(model, name, new_conv)

        # Replace BatchNorm2d with BatchNorm3d
        elif isinstance(module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm3d(
                num_features=module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats
            ).to(device)
            # Copy existing parameters
            new_bn.load_state_dict(module.state_dict())
            setattr(model, name, new_bn)

        # Replace MaxPool2d with MaxPool3d (anisotropic)
        elif isinstance(module, nn.MaxPool2d):
            # Handle int vs. tuple for kernel_size, stride, padding
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding

            new_pool = nn.MaxPool3d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                ceil_mode=False
            )
            setattr(model, name, new_pool)

    if is_top_level and hasattr(model, 'segmentation_head'):
        old_weight = model.segmentation_head[0].weight.data
        new_weight = torch.cat([
            old_weight[0:1],  # Class 0 (foreground, unchanged)
            old_weight[1:2].repeat(13, 1, *[1] * (old_weight.dim() - 2))  # Repeat Class 1 for 13 new positives
        ], dim=0)
        model.segmentation_head[0] = nn.Conv3d(
            16,
            14,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        model.segmentation_head[0].weight.data = new_weight

    return model


for fold in FOLDS:
    seed_everything(SEED)
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=1,
        classes=2
    ).to(device)
    model = convert_2d_to_3d(model)
    model.load_state_dict(torch.load(f'/kaggle/input/rsna-from-2d-binary-to-3d-full-segmentation-{fold}/best_3d_model_{fold}.pth'))
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=0, factor=.5, min_lr=1e-6)

    train_df = train[train['fold'] != fold].reset_index(drop=True)
    valid_df = train[train['fold'] == fold].reset_index(drop=True)
    valid_df = valid_df[~valid_df.flip].reset_index(drop=True)

    weights = torch.from_numpy(train_df[label_columns].values.sum(0)).float()
    weights[[2,3,4,5,6,7,8,9,10,11]] = weights[[2,3,4,5,6,7,8,9,10,11]].view(-1,2).sum(-1).view(5,1).tile(1,2).view(-1)/2
    weights = torch.cat([torch.tensor([len(train_df)]),weights])
    weights = len(train_df)/weights
    wmin = weights.min()
    wmax = weights.max()
    weights = 9 * (weights - wmin) / (wmax - wmin) +  1
    print(f'Class weights: {[np.round(w.item(),3) for w in weights]}')
    criterion = DiceFocalLoss(weights=weights.to(device))

    train_dataset = RSNA_Dataset_3D(train_df)
    val_dataset = RSNA_Dataset_3D(valid_df, VALID=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BS,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BS,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True
    )
#   Metrics tracking
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')

#   Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0.0
    
#       Training phase
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
        
            outputs = model(images)
            loss = criterion(outputs,  masks)
        
            loss.backward()
            optimizer.step()
        
            epoch_train_loss += loss.item() * images.size(0)
    
#       Validation phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
            
                outputs = model(images)
                loss = criterion(outputs, masks)
            
                epoch_val_loss += loss.item() * images.size(0)
    
#       Calculate epoch metrics
        epoch_train_loss /= len(train_loader.dataset)
        epoch_val_loss /= len(val_loader.dataset)
    
        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)
    
#       Update learning rate
        scheduler.step(epoch_val_loss)
    
#       Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), f'best_aneurysm_3d_model_{fold}.pth')
    
        print(f'Epoch {epoch+1}/{EPOCHS} - '
              f'Train Loss: {epoch_train_loss:.4f} - '
              f'Val Loss: {epoch_val_loss:.4f} - '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

#   Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.savefig(f'training_history_{fold}.png')
    plt.show()
#   Validation Inference
    model.load_state_dict(torch.load(f'best_aneurysm_3d_model_{fold}.pth'))
    y_true = []
    y_pred = []
    with torch.no_grad():
        for v in tqdm(valid_df[['case','pmin','pmax','s','axis']+label_columns].values):
            case,pmin,pmax,s,axis = v[:5]
            y_true.append(v[5:].tolist())

            npz =  np.load(source_path + case + '.npz')
            img = npz['volume']
            D,H,W = img.shape
            t = npz['t']
            l = npz['loc']
            d,h,w = np.rint(t[l==0][0]).astype(int)
            dd = hh = ww = np.rint(s/2).astype(int)

            d0 = d - dd//2
            h0 = h - hh//2
            w0 = w - ww//2

            d = d0 + dd
            h = h0 + hh
            w = w0 + ww

            pad_d0 = max(0, -d0)
            pad_h0 = max(0, -h0)
            pad_w0 = max(0, -w0)
        
            pad_d = max(0, d - D)
            pad_h = max(0, h - H)
            pad_w = max(0, w - W)
        
            d0 = max(0, d0)
            h0 = max(0, h0)
            w0 = max(0, w0)
        
            d = min(D, d)
            h = min(H, h)
            w = min(W, w)
                
            img = torch.nn.functional.pad(
                (torch.from_numpy(img[d0:d,h0:h,w0:w]) - pmin)/(pmax - pmin),
                (pad_w0,pad_w,pad_h0,pad_h,pad_d0,pad_d)
            ).to(device)
            img = F.interpolate(
                img.unsqueeze(0).unsqueeze(0),
                size=(WSIZE,WSIZE,WSIZE),
                mode='trilinear',
                align_corners=False
            )
#           Reorientation to axial perspective
            if axis == 0: img = torch.rot90(torch.rot90(img,1,(-3,-2)),-1,(-2,-1))
            if axis == 1: img = torch.rot90(img,1,(-3,-2))
#           Prediction
            out = model(img).softmax(1)
            out += model(img.flip(-1)).flip(-1).softmax(1)[:,[0,1,2,4,3,6,5,8,7,10,9,12,11,13]]
            out = out[0].view(14,-1)/2
            y_pred.append(out[1:].max(-1)[0].tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print(f'Area under the ROC curve mean: {np.mean([roc_auc_score(y_true[:,k],y_pred[:,k]) for k in range(13)])}')

    del model,optimizer,scheduler,criterion,train_dataset,val_dataset,train_loader,val_loader
    gc.collect()

