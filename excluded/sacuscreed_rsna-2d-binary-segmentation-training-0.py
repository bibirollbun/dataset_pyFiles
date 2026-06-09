!pip install segmentation_models_pytorch==0.3.3

import os
import gc
import random

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import segmentation_models_pytorch as smp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


FOLDS = [0]
SEED = 777
SIZE = 512
EPOCHS = 50
BS = 32
LR = 1e-4


source_path = '/kaggle/input/rsna-2d-binary-segmentation-preprocessing/'
output_images_dir = source_path + 'images/'
output_labels_dir = source_path + 'labels/'
train = pd.read_csv(source_path + 'rsna_2d_seg_folds.csv')
train.tail()


def elastic_deform(
    img: torch.Tensor,  # [1, H, W]
    msk: torch.Tensor,  # [1, H, W]
    alpha: float = 400.0,
    sigma: float = 10.0,
    grid_step: int = 16,
):
    _, H, W = img.shape
    
    # 1. Create coarse displacement grid (shape: [H//grid_step, W//grid_step])
    grid_h, grid_w = H // grid_step, W // grid_step
    dx = torch.randn(grid_h, grid_w) * alpha
    dy = torch.randn(grid_h, grid_w) * alpha
    
    # 2. Smooth coarse displacements
    dx = torch.from_numpy(gaussian_filter(dx.numpy(), sigma=sigma))
    dy = torch.from_numpy(gaussian_filter(dy.numpy(), sigma=sigma))
    
    # 3. Upsample to full resolution using bilinear interpolation
    dx_full = F.interpolate(
        dx.unsqueeze(0).unsqueeze(0),  # [1, 1, grid_h, grid_w]
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze()  # [H, W]
    
    dy_full = F.interpolate(
        dy.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    # 4. Create normalized grid (same as before)
    x, y = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack([y + dy_full, x + dx_full], dim=-1).float()  # [H, W, 2]
    
    # Normalize and apply deformation (unchanged)
    grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
    grid = grid.unsqueeze(0)  # [1, H, W, 2]
    
    img_deformed = F.grid_sample(
        img.unsqueeze(0),
        grid,
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    
    msk_deformed = F.grid_sample(
        msk.unsqueeze(0).float(),
        grid,
        mode='nearest',
        align_corners=False
    ).squeeze(0).long()
    
    return img_deformed, msk_deformed


class SliceDatasetNPY(Dataset):
    def __init__(
        self,
        fname,
        pmin,
        pmax,
        h,
        w,
        h0,
        w0,
        VALID=False
    ):
        """
        PyTorch Dataset for loading 2D slices from 3D medical volumes (numpy format) with 
        on-the-fly augmentation for intracranial aneurysm segmentation.
    
        Features:
        - Loads preprocessed .npy slices and corresponding masks
        - Normalizes intensities using case-specific percentile values (pmin/pmax)
        - Training mode includes:
          * Random asymmetric zoom/crop
          * Full rotation (0-360°)
          * Elastic deformations
          * Intensity variations
        - Validation mode uses center-cropping around mask centroid
        - Automatic resizing to 512x512 resolution
        - Supports all orthogonal views (YX, YZ, XZ)
    
        Args:
            fname: List of numpy filenames
            pmin: List of 1st percentile values for normalization
            pmax: List of 99th percentile values for normalization
            VALID: If True, disables augmentations for validation
        """
        self.fname = fname
        self.pmin = pmin
        self.pmax = pmax
        self.h = h
        self.w = w
        self.h0 = h0
        self.w0 = w0
        self.VALID = VALID
        self.img_resize = transforms.Resize((SIZE, SIZE), interpolation=transforms.InterpolationMode.BILINEAR)
        self.msk_resize = transforms.Resize((SIZE, SIZE), interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.fname)

    def __getitem__(self, idx):
        fname = self.fname[idx]
        img = torch.from_numpy(np.load(output_images_dir+fname)).unsqueeze(0).float()
        H,W = img.shape[1:]
        msk = torch.zeros((1,H,W), dtype=torch.long)
        h = self.h[idx]
        w = self.w[idx]
        h0 = self.h0[idx]
        w0 = self.w0[idx]
        msk[0,h0:h0+h,w0:w0+w] = torch.from_numpy(np.load(output_labels_dir+fname)).long()

        imin = self.pmin[idx]
        imax = self.pmax[idx]
        img = (img - imin) / (imax - imin + 1e-6)

        if H > W:
            h = W
            w = W
        else:
            h = H
            w = H

        if not self.VALID:
#           Random asymmetric zoom
            h = np.rint(h*(.8 + 0.4*np.random.rand())).astype(int)
            w = np.rint(w*(.8 + 0.4*np.random.rand())).astype(int)
            
            if H > h:
                h0 = np.random.randint(H - h)
                pad_h = 0
            else:
                h0 = 0
                pad_h = h - H
            if W > w:
                w0 = np.random.randint(W - w)
                pad_w = 0
            else:
                w0 = 0
                pad_w = w - H

            img = torch.nn.functional.pad(
                img,
                (pad_w//2,pad_w - pad_w//2,pad_h//2,pad_h - pad_h//2)
            )
            msk = torch.nn.functional.pad(
                msk,
                (pad_w//2,pad_w - pad_w//2,pad_h//2,pad_h - pad_h//2)
            )
#           Free rotation
            angle = 360*np.random.rand() - 180
            center = [w0 + w//2,h0 + h//2]
            img = transforms.functional.rotate(
                img,
                angle,
                transforms.InterpolationMode.BILINEAR,
                center=center
            )
            msk = transforms.functional.rotate(
                msk,
                angle,
                transforms.InterpolationMode.NEAREST,
                center=center
            )

        else:
            hh,ww = torch.where(msk[0])
            h0 = np.rint(hh.float().mean()).long() - h//2
            w0 = np.rint(ww.float().mean()).long() - w//2
            if h0 < 0: h0 = 0
            if w0 < 0: w0 = 0
            if h0 > H - h: h0 = H - h
            if w0 > W - w: w0 = W - w

        img = self.img_resize(img[:,h0:h0 + h,w0:w0 + w])
        msk = self.msk_resize(msk[:,h0:h0 + h,w0:w0 + w])

        if not self.VALID:
#           Intensity Inversion
            if np.random.rand() < .1:
                img = 1 - img
#           Elastic deformation
            if np.random.rand() < .5:
                img,msk = elastic_deform(img,msk)
#           Random flip            
            if np.random.rand() < .5:
                img = img.flip(-1)
                msk = msk.flip(-1)
#           Contrast
            if np.random.rand() < .5:
                f = .9 + .2*torch.rand(1)
                img *= f - (f - 1)/2
#           Brightness
            if np.random.rand() < .5:
                img += .2*torch.rand(1) - .1
#           Gaussian Noise
            if np.random.rand() < .5:
                img += torch.normal(torch.tensor(0.),torch.tensor(.05),(1,SIZE,SIZE))
        
        return img, msk[0].long()


ds = SliceDatasetNPY(
    fname=train['fname'],
    pmin=train['pmin'],
    pmax=train['pmax'],
    h=train['h'],
    w=train['w'],
    h0=train['h0'],
    w0=train['w0']
)


img,msk = ds.__getitem__(np.random.randint(len(ds)))
plt.imshow(img[0] + msk)


del ds
gc.collect()


ds = SliceDatasetNPY(
    fname=train['fname'],
    pmin=train['pmin'],
    pmax=train['pmax'],
    h=train['h'],
    w=train['w'],
    h0=train['h0'],
    w0=train['w0'],
    VALID=True
)


img,msk = ds.__getitem__(np.random.randint(len(ds)))
plt.imshow(img[0] + msk)


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


criterion = DiceFocalLoss()

for fold in FOLDS:
    seed_everything(SEED)
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,
        classes=2
    )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, min_lr=1e-6)

    train_df = train[train['fold'] != fold].reset_index(drop=True)
    valid_df = train[train['fold'] == fold].reset_index(drop=True)

    train_dataset = SliceDatasetNPY(
        train_df['fname'],
        train_df['pmin'],
        train_df['pmax'],
        h=train_df['h'],
        w=train_df['w'],
        h0=train_df['h0'],
        w0=train_df['w0']
    )
    val_dataset = SliceDatasetNPY(
        valid_df['fname'],
        valid_df['pmin'],
        valid_df['pmax'],
        h=valid_df['h'],
        w=valid_df['w'],
        h0=valid_df['h0'],
        w0=valid_df['w0'],
        VALID=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BS,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BS,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

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
            loss = criterion(outputs, masks)
        
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
            torch.save(model.state_dict(), f'best_model_{fold}.pth')
    
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

    del model,optimizer,scheduler,train_dataset,val_dataset,train_loader,val_loader
    gc.collect()

