!pip install segmentation_models_pytorch==0.3.3
!pip install connected-components-3d
!pip install zarr

import json
import matplotlib.pyplot as plt
import zarr
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import gc
import cc3d

import sys
sys.path.insert(0,'/kaggle/input/czii-metric')
from Metric import compute_metrics,score

import torchvision
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from fastai.vision.all import *
import segmentation_models_pytorch as smp

device = 'cuda' if torch.cuda.is_available() else 'cpu'


VOLUMES_PATH = '/kaggle/input/czii-cryo-et-object-identification/train/static/ExperimentRuns/'
LABELS_PATH = '/kaggle/input/czii-cryo-et-object-identification/train/overlay/ExperimentRuns/'
SAMPLES = [
    'TS_73_6',
    'TS_86_3',
    'TS_69_2',
    'TS_6_4',
    'TS_99_9',
    'TS_5_4',
    'TS_6_6'
]
SYNTH_SAMPLES = [
    'TS_0',
    'TS_1',
    'TS_10',
    'TS_11',
    'TS_12',
    'TS_13',
    'TS_14',
    'TS_15',
    'TS_16',
    'TS_17',
    'TS_18',
    'TS_19',
    'TS_2',
    'TS_20',
    'TS_21',
    'TS_22',
    'TS_23',
    'TS_24',
    'TS_25',
    'TS_26',
    'TS_3',
    'TS_4',
    'TS_5',
    'TS_6',
    'TS_7',
    'TS_8',
    'TS_9'
]

TARGETS = [
    'apo-ferritin',# easy
    'beta-galactosidase',# hard
    'ribosome',# easy
    'thyroglobulin',# hard
    'virus-like-particle'# easy
]
SEED = 1337
#FOLDS = [1]#[1,2,3,4,5,6,7]
ENCODER_NAME = "resnet18"#"efficientnet-b0","resnet18","timm-regnetx_002","densenet121",timm-resnest14d
ENCODER_DEPTH = 3
#DROPOUT = .2
PATCH_D = 64
PATCH_H = 128
PATCH_W = 128
radius = {
    'apo-ferritin':60,# easy
    'beta-galactosidase':90,# hard
    'ribosome':150,# easy
    'thyroglobulin':130,# hard
    'virus-like-particle':135# easy
}
core = {
    k:radius[k]*.05 for k in radius
}
shell = {
    k:radius[k]*.11 for k in radius
}
BS = 8
LR = 5e-4#5e-4
EPOCHS = 1#10
TH = .5
CONTRAST_AUG = .25#.125
BRIGTHNESS_AUG = .25#.125


V = {}
L = {}
for sample in SYNTH_SAMPLES:
        L[sample] = {}
        file = zarr.open('/kaggle/input/czii-synth-data/'+sample+'.zarr', mode='r')
        vol = np.array(file[0])
        pmin,pmax = np.percentile(vol,(1,99))
        V[sample] = (vol - pmin)/(pmax - pmin)
        V[sample] = torch.as_tensor(V[sample])
        D,H,W =V[sample].shape   
        h = 128 - H%128
        w = 128 - W%128
        V[sample] = torch.nn.functional.pad(
            V[sample].unsqueeze(0),
            (
                w//2,w - w//2,
                h//2,h - h//2
            ),
            mode='reflect'
        )[0,4:-4]
        for target in TARGETS:
            L[sample][target] = []
            for p in pd.read_json('/kaggle/input/czii-synth-data/synth_labels/'+sample+'/'+target+'.ndjson', lines=True)['location']:
                L[sample][target].append([
                    p['z']-4,
                    p['y']+h//2,
                    p['x']+w//2
                ])
            L[sample][target] = torch.tensor(L[sample][target]).float().to(device)


for sample in SAMPLES:
        L[sample] = {}
        file = zarr.open(VOLUMES_PATH + sample + '/VoxelSpacing10.000/denoised.zarr', mode='r')
        scale = file.attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']
        vol = np.array(file[0])
        pmin,pmax = np.percentile(vol,(1,99))
        V[sample] = (vol - pmin)/(pmax - pmin)
        V[sample] = torch.as_tensor(V[sample])
        D,H,W =V[sample].shape   
        h = 128 - H%128
        w = 128 - W%128
        d = 64 - D%64
        V[sample] = torch.nn.functional.pad(
            V[sample].unsqueeze(0),
            (
                w//2,w - w//2,
                h//2,h - h//2,
                d//2,d - d//2
            ),
            mode='reflect'
        )[0]
        for target in TARGETS:
            L[sample][target] = []
            f = open(LABELS_PATH + sample + '/Picks/' + target + '.json')
            for p in json.loads(f.read())['points']:
                L[sample][target].append([
                    p['location']['z']/scale[0]+d//2,
                    p['location']['y']/scale[1]+h//2,
                    p['location']['x']/scale[2]+w//2
                ])
            L[sample][target] = torch.tensor(L[sample][target]).float().to(device)


z = torch.stack([
    torch.stack([
        torch.arange(PATCH_D)
    ]*PATCH_H)
]*PATCH_W).permute(2,1,0)
x = torch.stack([
    torch.stack([
        torch.arange(PATCH_W)
    ]*PATCH_H)
]*PATCH_D)
y = torch.stack([
    torch.stack([
        torch.arange(PATCH_H)
    ]*PATCH_W)
]*PATCH_D).permute(0,2,1)
zyx = torch.stack([z,y,x]).float().to(device)


class CryoET_Dataset(Dataset):
    def __init__(self, V, L, samples, VALID=False):
        self.data = V
        self.L = L
        self.samples = samples
        self.sample = []
        self.d = []
        self.h = []
        self.w = []
        for sample in samples:
            d,h,w = np.where(np.ones((5,9,9)))
            d *= 32
            h *= 64
            w *= 64
            self.sample = self.sample + [sample]*len(d)
            self.d = self.d + list(d)
            self.h = self.h + list(h)
            self.w = self.w + list(w)
        self.VALID = VALID

    def __len__(self):
        return len(self.d)

    def __rawgetitem__(self, idx):
        
        sample = self.sample[idx]
        d = self.d[idx]
        h = self.h[idx]
        w = self.w[idx]

        if not self.VALID:
            d += np.random.randint(32) - 16
            h += np.random.randint(64) - 32
            w += np.random.randint(64) - 32

            if d < 0: d = 0
            if h < 0: h = 0
            if w < 0: w = 0

            if d > 128: d = 128
            if h > 512: h = 512
            if w > 512: w = 512

        image = torch.as_tensor(self.data[sample][
            d:d+PATCH_D,
            h:h+PATCH_H,
            w:w+PATCH_W
        ]).to(device)

        o = torch.tensor([d,h,w]).to(device)
        hm = torch.zeros(
            PATCH_D,
            PATCH_H,
            PATCH_W
        ).long().to(device)
        searching = True
        for k in range(len(TARGETS)):
#           Find and label particles within this voxel
            t = TARGETS[k]
            inside = (self.L[sample][t][:,0] > d)* \
                     (self.L[sample][t][:,1] > h + core[t])* \
                     (self.L[sample][t][:,2] > w + core[t])* \
                     (self.L[sample][t][:,0] < d + PATCH_D)* \
                     (self.L[sample][t][:,1] < h + PATCH_H - core[t])* \
                     (self.L[sample][t][:,2] < w + PATCH_W - core[t])
            if inside.sum() > 0:
                searching = False
                r = zyx.view(1,3,PATCH_D,PATCH_H,PATCH_W) - (self.L[sample][t][inside] - o).view(-1,3,1,1,1)
                cm = ((r*r).sum(1).sqrt() < core[t]).sum(0) > 0
                hm[cm] = k + 1
#           Find and mask partially present particles
            pinside = ~inside* \
                      (self.L[sample][t][:,0] > d - shell[t])* \
                      (self.L[sample][t][:,1] > h - shell[t])* \
                      (self.L[sample][t][:,2] > w - shell[t])* \
                      (self.L[sample][t][:,0] < d + PATCH_D + shell[t])* \
                      (self.L[sample][t][:,1] < h + PATCH_H + shell[t])* \
                      (self.L[sample][t][:,2] < w + PATCH_W + shell[t])
            if pinside.sum() > 0:
                r = zyx.view(1,3,PATCH_D,PATCH_H,PATCH_W) - (self.L[sample][t][pinside] - o).view(-1,3,1,1,1)
                sm = ((r*r).sum(1).sqrt() < shell[t]).sum(0) > 0
                image[sm] = image[sm][torch.randperm(sm.sum())]

        if not self.VALID:
#           Rot90
            angle = np.random.randint(4)
            image = torch.rot90(image, k=angle, dims=(-2, -1))
            hm = torch.rot90(hm, k=angle, dims=(-2, -1))
#           CONTRAST
            x = np.random.normal(1,CONTRAST_AUG)
            image *= x
#           BRIGTHNESS
            x = np.random.normal(0,BRIGTHNESS_AUG)
            image += x

        return image,hm,searching
    
    def __getitem__(self, idx):
        
        image,hm,searching = self.__rawgetitem__(idx)
        if not self.VALID:
            while searching:
                image,hm,searching = self.__rawgetitem__(np.random.randint(self.__len__()))

        return image,hm


tds = CryoET_Dataset(V,L,SYNTH_SAMPLES)


len(tds)


image,heatmap = tds.__getitem__(np.random.randint(len(tds)))
plt.imshow(image.sum(0).cpu() + (heatmap.sum(0).cpu()))
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


class repack_3D(nn.Module):
    def __init__(
        self
        ):
        super(repack_3D, self).__init__()

    def forward(self,X):
        C,H,W = X.shape[-3:]
        return X.view(-1,PATCH_D,C,H,W).permute(0,2,1,3,4)


class repack_2D(nn.Module):
    def __init__(
        self
        ):
        super(repack_2D, self).__init__()

    def forward(self,X):
        C,D,H,W = X.shape[-4:]
        return X.permute(0,2,1,3,4).reshape(-1,C,H,W)


class myUNet2Dto3D(nn.Module):
    def __init__(
        self,
        classes
        ):
        super(myUNet2Dto3D, self).__init__()

        self.classes = classes
        
        decoder_channels = (256, 128, 64, 32, 16)[-ENCODER_DEPTH:]
        decoder_in_channels = (768, 384, 192, 128 , 32)[-ENCODER_DEPTH:]
    
        
        self.UNet = smp.Unet(
            encoder_name=ENCODER_NAME,
            encoder_depth=ENCODER_DEPTH,
            decoder_channels=decoder_channels,
            classes=classes,
            in_channels=1
        ).to(device)

        self.UNet.encoder.layer3 = nn.Identity()
        self.UNet.encoder.layer4 = nn.Identity()        

        for k in range(ENCODER_DEPTH):
            self.UNet.decoder.blocks[k].conv1[0] = nn.Sequential(
#               nn.Dropout(DROPOUT),
                repack_3D(),
                nn.Conv3d(
                    decoder_in_channels[k],
                    decoder_channels[k],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                )
            )
            self.UNet.decoder.blocks[k].conv1[1] = nn.BatchNorm3d(
                decoder_channels[k],
                eps=1e-05, momentum=0.1,
                affine=True,
                track_running_stats=True
            )
            self.UNet.decoder.blocks[k].conv2[0] = nn.Conv3d(
                decoder_channels[k],
                decoder_channels[k],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
                )
            self.UNet.decoder.blocks[k].conv2[1] = nn.Sequential(
                nn.BatchNorm3d(
                    decoder_channels[k],
                    eps=1e-05, momentum=0.1,
                    affine=True,
                    track_running_stats=True
                ),
                repack_2D()
            )

        self.UNet.segmentation_head[0] = nn.Sequential(
            repack_3D(),
            nn.Conv3d(
                16,
                6,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self,X):
        H,W = X.shape[-2:]
        x = self.UNet(X.view(-1,1,H,W))
        
        return x


# https://arxiv.org/pdf/1708.02002
# https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/README.md
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
        if reduction not in ('mean', 'sum', 'none', 'avg_mean'):
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
#       Small addition, average mean
        elif self.reduction == 'avg_mean':
            loss = (loss.sum())/((self.alpha[y]*focal_term).sum())

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


class FL_DL(nn.Module):
    """https://academic.oup.com/bioinformaticsadvances/article/4/1/vbae169/7907198
       The combined focal loss and dice loss function improves
       the segmentation of beta-sheets in medium-resolution
       cryo-electron-microscopy density maps
    """
    def __init__(
            self,
            alpha: Optional[Sequence] = None,
            gamma: float = 0.,
            reduction: str = 'mean',
            ignore_index: int = -100,
            device='cpu',
            dtype=torch.float32,
            weight=None
        ):
        super(FL_DL, self).__init__()
        if alpha is not None:
            if not isinstance(alpha, Tensor):
                alpha = torch.tensor(alpha)
            alpha = alpha.to(device=device, dtype=dtype)
        self.fl = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction=reduction,
            ignore_index=ignore_index
        )

        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight) # Normalized weight
        self.smooth = 1e-5
        self.dl = DiceLoss(weight)

    def forward(self, predict, target, alpha=.5):
        fl = self.fl(predict, target)
        dl = self.dl(predict, target)        

        return alpha*fl + (1 - alpha)*dl


seed_everything(SEED)
model = myUNet2Dto3D(
    len(TARGETS)+1
)

tds = CryoET_Dataset(V,L,SYNTH_SAMPLES)
vds = CryoET_Dataset(V,L,SAMPLES,VALID=True)
    
tdl = torch.utils.data.DataLoader(tds, batch_size=BS, shuffle=True, drop_last=True)
vdl = torch.utils.data.DataLoader(vds, batch_size=BS, shuffle=False)

dls = DataLoaders(tdl,vdl)

learn = Learner(
        dls,
        model,
        lr=LR,
#       loss_func=focal_loss(alpha=weight,gamma=2,reduction='avg_mean',device=device),
        loss_func=DiceLoss(),
#       loss_func=FL_DL(alpha=weight,reduction='avg_mean',device=device),
        cbs=[
            ShowGraphCallback(),
#           GradientAccumulation(n_acc=4)
        ]
)
learn.fit_one_cycle(EPOCHS)
torch.save(model,ENCODER_NAME+'_'+str(ENCODER_DEPTH)+'_segmentation_2Dto3D_synthetic_pretraining')
del tdl,vdl,dls,model,learn
gc.collect()


def add_contour(mask,N):
    for _ in range(N):
        i,j = np.where(mask)
        i,j=np.clip(i,1,638),np.clip(j,1,638)
        mask[i+1,j] = True
        mask[i-1,j] = True
        mask[i,j+1] = True
        mask[i,j-1] = True
        mask[i+1,j+1] = True
        mask[i+1,j-1] = True
        mask[i-1,j+1] = True
        mask[i-1,j-1] = True

    return mask


D,H,W = 192,640,640
z = torch.stack([
    torch.stack([
        torch.arange(D)
    ]*H)
]*W).permute(2,1,0)
x = torch.stack([
    torch.stack([
        torch.arange(W)
    ]*H)
]*D)
y = torch.stack([
    torch.stack([
        torch.arange(H)
    ]*W)
]*D).permute(0,2,1)
zyx = torch.stack([z,y,x]).float().to(device)
HM = torch.zeros(192,640,640).long()


PATCH_D = 32
BS = 32
experiment = []
particle_type = []
x = []
y = []
z = []
GT_experiment = []
GT_particle_type = []
GT_x = []
GT_y = []
GT_z = []
model = torch.load(ENCODER_NAME+'_'+str(ENCODER_DEPTH)+'_segmentation_2Dto3D_synthetic_pretraining').eval()
print('TP: GREEN\nFN: RED\nFP: YELLOW')
for sample in SAMPLES:
    print(sample)
    volume = torch.as_tensor(V[sample]).to(device)
    MASK = torch.zeros(192,640,640,len(TARGETS)+1)
    for k in range(len(TARGETS)):
        for p in L[sample][TARGETS[k]]:
            r = zyx - p.view(3,1,1,1)
            cm = (r*r).sum(0).sqrt() < core[TARGETS[k]]
            HM[cm] = k + 1
    STEPS = 11
    with torch.no_grad():
            for k in tqdm(range(STEPS)):
                mask = model(
                        volume[
                            k*PATCH_D//2:k*PATCH_D//2+PATCH_D
                        ]
                )[0].argmax(0).cpu()
                
                patch = MASK[
                    k*PATCH_D//2:k*PATCH_D//2+PATCH_D
                ].reshape(-1,6)
                    
                patch[torch.arange(mask.numel()),mask.view(-1)] += 1

                MASK[
                    k*PATCH_D//2:k*PATCH_D//2+PATCH_D
                ] = patch.reshape(32,640,640,len(TARGETS)+1)

    MASK = MASK.argmax(-1).numpy()
    efig, e_axes = plt.subplots(1, 3, figsize=(10,10))
    hfig, h_axes = plt.subplots(1, 2, figsize=(10,10))
    for k in range(len(TARGETS)):
        TP = ((MASK == k+1)*(HM == k+1).numpy()).sum(0)
        FN = ((MASK != k+1)*(HM == k+1).numpy()).sum(0)
        FP = ((MASK == k+1)*(HM != k+1).numpy()).sum(0)
        TP = TP/TP.max()
        FN = FN/FN.max()
        FP = FP/FP.max()
        img = np.ones((640,640,3))
        img[add_contour(FP>0,1)] = 0,0,0
        img[FP>0] = 1,1,0
        img[add_contour(FN>0,1)] = 0,0,0
        img[FN>0] = 1,0,0
        img[add_contour(TP>0,1)] = 0,0,0
        img[TP>0] = 0,1,0
        if k in [0,2,4]:
            e_axes[k//2].imshow(img)
            e_axes[k//2].set_title(TARGETS[k])
        else:
            h_axes[k//2].imshow(img)
            h_axes[k//2].set_title(TARGETS[k])
        labels_out = cc3d.connected_components(MASK == k+1)
        stats = cc3d.statistics(labels_out)
        preds = stats['centroids'][1:]
        experiment = experiment + [sample]*len(preds)
        particle_type = particle_type + [TARGETS[k]]*len(preds)
        x = x + list(preds[:,2])
        y = y + list(preds[:,1])
        z = z + list(preds[:,0])

        GT_experiment = GT_experiment + [sample]*len(L[sample][TARGETS[k]])
        GT_particle_type = GT_particle_type + [TARGETS[k]]*len(L[sample][TARGETS[k]])
        GT_x = GT_x + list(L[sample][TARGETS[k]][:,2].tolist())
        GT_y = GT_y + list(L[sample][TARGETS[k]][:,1].tolist())
        GT_z = GT_z + list(L[sample][TARGETS[k]][:,0].tolist())

        '''preds = torch.tensor(preds).float().to(device)
        d = (preds.unsqueeze(1) - L[sample][TARGETS[k]].unsqueeze(0))
        d = (d*d).sum(-1).sqrt()
        hits = (d < .05*radius[TARGETS[k]]).max(0)[0].sum().item()
        miss = (d > .05*radius[TARGETS[k]]).min(1)[0].sum().item()

        print(TARGETS[k])
        print('hits: ',hits/len(L[sample][TARGETS[k]]))
        print('miss: ',miss/len(preds))
        print()'''

    plt.show()
    HM[:] = 0
    del volume,MASK
    gc.collect()


submission = pd.DataFrame({
    'id':np.arange(len(experiment)),
    'experiment':experiment,
    'particle_type':particle_type,
    'x':x,
    'y':y,
    'z':z
})
submission[['x','y','z']] = submission[['x','y','z']]*10
submission.tail()


solution = pd.DataFrame({
    'id':np.arange(len(GT_experiment)),
    'experiment':GT_experiment,
    'particle_type':GT_particle_type,
    'x':GT_x,
    'y':GT_y,
    'z':GT_z
})
solution[['x','y','z']] = solution[['x','y','z']]*10
solution.tail()


score(
    solution = solution,
    submission = submission,
    row_id_column_name = 'id',
    distance_multiplier = .5,
    beta = 4
)

