!pip install segmentation_models_pytorch==0.3.3

import os
import gc
import random
import math
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import confusion_matrix

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
ANGLE_AUG = 30
EPOCHS = 20
BS = 2
LR = 5e-5
WORKERS = 4
ROTATION_PROB_DECAY = .5699 # (1 - p)**2 = p**3 | Equals single and triple rotations during augmentations


source_path = '/kaggle/input/rsna-3d-full-segmentation-preprocessing/'
cases = [c[:-4] for c in os.listdir(source_path + 'images')]
cases[:5]


train = pd.read_csv(source_path + 'rsna_3d_seg_folds.csv')
with open(source_path + 'centroids.pkl', 'rb') as f:
    centroids = pickle.load(f)

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
        if VALID:
            self.df = df[df['target'].apply(lambda v:v in [
                1, # Other Posterior Circulation
                2, # Basilar Tip
                5, # Right Infraclinoid Internal Carotid Artery
                6, # Left Infraclinoid Internal Carotid Artery
                9, # Right Middle Cerebral Artery
                10,# Left Middle Cerebral Artery
                13 # Anterior Communicating Artery
            ])].reset_index(drop=True)
        else:
            self.df = df
        self.VALID = VALID

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):        
        case = self.df['SeriesInstanceUID'][idx]
        target = self.df['target'][idx]
        pmin = self.df['pmin'][idx]
        pmax = self.df['pmax'][idx]
        f = self.df['flip'][idx]

        img = torch.from_numpy(np.load(source_path + 'images/' + case + '.npy')).float()
        msk = torch.zeros_like(img)
        d = self.df['d'][idx]
        h = self.df['h'][idx]
        w = self.df['w'][idx]
        r = (d*h*w)**(1./3)
        d0 = self.df['d0'][idx]
        h0 = self.df['h0'][idx]
        w0 = self.df['w0'][idx]
        msk[d0:d0+d,h0:h0+h,w0:w0+w] = torch.from_numpy(np.load(source_path + 'labels/' + case + '.npy'))

        D,H,W = img.shape

        if not self.VALID:
            z,y,x = centroids[case][target]
            d,h,w = np.rint(centroids[case][target]).astype(int)
#           Random asymmetric zoom
            rd = r*(.9 + 0.2*np.random.rand())
            rh = r*(.9 + 0.2*np.random.rand())
            rw = r*(.9 + 0.2*np.random.rand())

            dd = np.rint(rd).astype(int)
            hh = np.rint(rh).astype(int)
            ww = np.rint(rw).astype(int)

            ddd = np.rint(rd/2).astype(int)
            hhh = np.rint(rh/2).astype(int)
            www = np.rint(rw/2).astype(int)

            dd += ddd
            hh += hhh
            ww += www

            d += np.random.randint(ddd//2) - ddd//4
            h += np.random.randint(hhh//2) - hhh//4
            w += np.random.randint(www//2) - www//4

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
                (img[d0:d,h0:h,w0:w] - pmin)/(pmax - pmin),
                (pad_w0,pad_w,pad_h0,pad_h,pad_d0,pad_d)
            )
            msk = torch.nn.functional.pad(
                msk[d0:d,h0:h,w0:w],
                (pad_w0,pad_w,pad_h0,pad_h,pad_d0,pad_d)
            )
#           Free rotation
            p = 1
            center = np.array([
                z - d0 + pad_d0,
                y - h0 + pad_h0,
                x - w0 + pad_w0
            ])
            axis_perm = [
                [2,0,1],
                [1,2,0]
            ][np.random.randint(2)]
            for _ in range(3):
                if np.random.rand() < p:
                    c = list(center[1:][::-1])
                    angle = 2*ANGLE_AUG*(np.random.rand() - .5)
                    img = transforms.functional.rotate(
                        img,
                        angle,
                        transforms.InterpolationMode.BILINEAR,
                        center=c
                    )
                    msk = transforms.functional.rotate(
                        msk,
                        angle,
                        transforms.InterpolationMode.NEAREST,
                        center=c
                    )
                    p *= ROTATION_PROB_DECAY
#               Rotate axis
                img = img.permute(axis_perm)
                msk = msk.permute(axis_perm)
                center = center[axis_perm]
                
            img = img[ddd//2:dd-ddd//2,hhh//2:hh-hhh//2,www//2:ww-www//2]
            msk = msk[ddd//2:dd-ddd//2,hhh//2:hh-hhh//2,www//2:ww-www//2]

        else:
            d,h,w = np.rint(centroids[case][target]).astype(int)
            dd = hh = ww = np.rint(r).astype(int)

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
                (img[d0:d,h0:h,w0:w] - pmin)/(pmax - pmin),
                (pad_w0,pad_w,pad_h0,pad_h,pad_d0,pad_d)
            )
            msk = torch.nn.functional.pad(
                msk[d0:d,h0:h,w0:w],
                (pad_w0,pad_w,pad_h0,pad_h,pad_d0,pad_d)
            )
        img = F.interpolate(
            img.unsqueeze(0).unsqueeze(0),
            size=(SIZE,SIZE,SIZE),
            mode='trilinear',
            align_corners=False
        )[0]
        msk = F.interpolate(
            msk.unsqueeze(0).unsqueeze(0),
            size=(SIZE,SIZE,SIZE),
            mode='nearest'
        )[0]

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
            img = img.flip(-1)
            msk = msk.flip(-1)
            for t in range(3,12,2):
                mr = msk == t
                ml = msk == t + 1
                msk[mr] = t + 1
                msk[ml] = t
        
        return img, msk[0].long()


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
    def __init__(self, gamma=1.0, alpha=0.5):
        super().__init__()
        self.DL = DiceLoss()
        self.FL = FocalLoss(gamma=gamma)
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


criterion = DiceFocalLoss()
for fold in FOLDS:
    seed_everything(SEED)
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=1,
        classes=2
    ).to(device)
    model.load_state_dict(torch.load(f'/kaggle/input/rsna-2d-binary-segmentation-training-{fold}/best_model_{fold}.pth'))
    model = convert_2d_to_3d(model)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=.5, min_lr=1e-6)

    train_df = train[train['fold'] != fold].reset_index(drop=True)
    valid_df = train[train['fold'] == fold].reset_index(drop=True)

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
            torch.save(model.state_dict(), f'best_3d_model_{fold}.pth')
    
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
#   Confusion Matrix of the best model
    model.load_state_dict(torch.load(f'best_3d_model_{fold}.pth'))
    cm = np.zeros((14, 14))
    labels = np.arange(14)
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images = images.to(device, non_blocking=True) 
            outputs = model(images).argmax(1)
            cm += confusion_matrix(
                y_true=masks.flatten().tolist(),
                y_pred=outputs.flatten().tolist(),
                labels=labels
            )
    D = np.sqrt(cm[np.arange(14),np.arange(14)])
    cm = cm/(D.reshape(-1,1))
    cm = cm/(D.reshape(1,-1))

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        cm, 
        annot=True, 
        fmt=".2f",
        cmap='Blues', 
        vmin=0, 
        vmax=1,
        xticklabels=range(14), 
        yticklabels=range(14),
        cbar_kws={'label': 'Normalized Value'}
    )
    plt.title('Class-Normalized Confusion Matrix', pad=20, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(f' confusion_matrix_{fold}.png')
    plt.show()

    del model,optimizer,scheduler,train_dataset,val_dataset,train_loader,val_loader
    gc.collect()

