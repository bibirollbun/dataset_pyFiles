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

TARGETS = [
    'apo-ferritin',# easy
    'beta-galactosidase',# hard
    'ribosome',# easy
    'thyroglobulin',# hard
    'virus-like-particle'# easy
]
SEED = 1337
FOLDS = [1,2,3,4,5,6,7]
ENCODER_NAME = "resnet18"#"efficientnet-b0","resnet18","timm-regnetx_002","densenet121",timm-resnest14d
ENCODER_DEPTH = 3
PATCH_D = 64
PATCH_H = 256
PATCH_W = 256
radius = {
    'apo-ferritin':60,# easy
    'beta-galactosidase':90,# hard
    'ribosome':150,# easy
    'thyroglobulin':130,# hard
    'virus-like-particle':135# hard
}
core = {
    k:radius[k]*.05 for k in radius
}
shell = {
    k:radius[k]*.12 for k in radius
}
LR = 1e-4#5e-4
EPOCHS = 10#30#50#10
TH = .5
CONTRAST_AUG = .25#.125
BRIGTHNESS_AUG = .25#.125


V = {}
L = {}
for sample in SAMPLES:
        L[sample] = {}
        file = zarr.open(VOLUMES_PATH + sample + '/VoxelSpacing10.000/denoised.zarr', mode='r')
        scale = file.attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']
        vol = np.array(file[0])
        pmin,pmax = np.percentile(vol,(1,99))
        V[sample] = (vol - pmin)/(pmax - pmin)
        D,H,W =V[sample].shape   
        h = 128 - H%128
        w = 128 - W%128
        d = 64 - D%64
        V[sample] = torch.as_tensor(V[sample])
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


class CryoET_Mosaic_Dataset(Dataset):
#   Simple homemade data Dataset
#   We will fold 16 64x64x64 particle voxels in one single 64x256x256 "synthetic sample" and see what happens
#   The idea is to learn as much as possible from hard to remember synthetic data for train with real data after
    def __init__(self, V, L, samples):
        z = torch.stack([
            torch.stack([
                torch.arange(64)
            ]*64)
        ]*64).permute(2,1,0)
        x = torch.stack([
            torch.stack([
                torch.arange(64)
            ]*64)
        ]*64)
        y = torch.stack([
            torch.stack([
                torch.arange(64)
            ]*64)
        ]*64).permute(0,2,1)
        self.zyx = torch.stack([z,y,x]).float().to(device)
        self.data = V
        self.L = L
        self.samples = samples
        self.s = []
        self.t = []
        self.p = []
        for s in samples:
            for t in TARGETS:
                n = len(L[s][t])
                self.s = self.s + [s]*n
                self.t = self.t + [t]*n
                self.p = self.p + torch.arange(n).tolist()

        r = len(self.s)%16
        if r > 0:
            self.s = self.s + list(np.array(samples)[np.random.randint(0,len(samples),16-r)])
            self.t = self.t + ['background']*(16-r)
            self.p = self.p + [None]*(16-r)

        zipped = list(zip(self.s,self.t,self.p))
        random.shuffle(zipped)
        self.s,self.t,self.p = zip(*zipped)

    def __len__(self):
        return len(self.s)//16
    
    def __shuffle__(self):
        zipped = list(zip(self.s,self.t,self.p))
        random.shuffle(zipped)
        self.s,self.t,self.p = zip(*zipped)

    def __getitem__(self, idx):
        
        samples = self.s[idx*16:16*(idx+1)]
        targets = self.t[idx*16:16*(idx+1)]
        ipoints = self.p[idx*16:16*(idx+1)]

        image = []
        mask = []
        for k in range(16):
            if targets[k] != 'background':
                img = torch.zeros(64,64*3,64*3).to(device) + .5
                m = torch.zeros(64,64*3,64*3).long().to(device) - 100

                p = z,y,x = np.rint(self.L[samples[k]][targets[k]][ipoints[k]].cpu()).long().to(device)
                d = z - 32
                h = y - 96
                w = x - 96
                
                freedom = 32 - int(shell[targets[k]])
                d += np.random.randint(2*freedom) - freedom
                h += np.random.randint(2*freedom) - freedom
                w += np.random.randint(2*freedom) - freedom
                o = torch.tensor([d,h,w]).to(device)

                end_d = d + 64
                end_h = h + 192
                end_w = w + 192
                if d < 0:
                    start_d = - d
                    d = 0
                else:
                    start_d = 0
                if h < 0:
                    start_h = - h
                    h = 0
                else:
                    start_h = 0
                if w < 0:
                    start_w = - w
                    w = 0
                else:
                    start_w = 0

                crop = torch.as_tensor(self.data[samples[k]][
                    d:end_d,
                    h:end_h,
                    w:end_w
                ]).to(device)
                D,H,W = crop.shape
                
                img[
                    start_d:start_d+D,
                    start_h:start_h+H,
                    start_w:start_w+W
                ] = crop
                m[
                    start_d:start_d+D,
                    start_h:start_h+H,
                    start_w:start_w+W
                ] = 0
                center = (p - o)[[2,1]].tolist()
#               Free rotation around chosen particle
                angle = torch.as_tensor(random.uniform(-180, 180))
                img = torchvision.transforms.functional.rotate(
                    img,angle.item(),
#                   interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    center=center
                )
                m = torchvision.transforms.functional.rotate(
                    m,angle.item(),
#                   interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    center=center
                )
                angle = -angle*math.pi/180
                s = torch.sin(angle)
                c = torch.cos(angle)
                rot = torch.stack([
                    torch.stack([c, s]),
                    torch.stack([-s, c])
                ]).to(device)
                img = img[:,64:128,64:128]
                m = m[:,64:128,64:128]
                missing = m == - 100
                img[missing] = img[~missing].mean()
                m[:] = 0
#               What particles are completely within this voxel?
                for kk in range(5):
                    t = TARGETS[kk]
                    particles = (self.L[samples[k]][t] - o)
                    center = torch.as_tensor(center).float().to(device)
                    particles[:,[[2,1]]] = ((particles[:,[[2,1]]] - center) @ rot) + center - 64
                    inside = (particles[:,0] > 0)* \
                             (particles[:,0] < 64)* \
                             (particles[:,1] > shell[t])* \
                             (particles[:,2] > shell[t])* \
                             (particles[:,1] < 64 - shell[t])* \
                             (particles[:,2] < 64 - shell[t])
                    if inside.sum() > 0:
                        r = self.zyx.view(1,3,64,64,64) - particles[inside].view(-1,3,1,1,1)
                        r = (r*r).sum(1).sqrt()
                        cm = (r < core[t]).sum(0) > 0
                        m[cm] = kk + 1
#                   Find and mask partially present particles
                    pinside = ~inside* \
                              (particles[:,0] > - shell[t])* \
                              (particles[:,1] > - shell[t])* \
                              (particles[:,2] > - shell[t])* \
                              (particles[:,0] < 64 + shell[t])* \
                              (particles[:,1] < 64 + shell[t])* \
                              (particles[:,2] < 64 + shell[t])
                    if pinside.sum() > 0:
                        r = self.zyx.view(1,3,64,64,64) - particles[pinside].view(-1,3,1,1,1)
                        r = (r*r).sum(1).sqrt()
                        sm = (r < shell[t]).sum(0) > 0
#                       Confetti
                        img[sm] = img[sm][torch.randperm(sm.sum())]

#               Rot90
                angle = np.random.randint(4)
                img = torch.rot90(img, k=angle, dims=(-2, -1))
                m = torch.rot90(m, k=angle, dims=(-2, -1))
#               Flip party
                for axis in [0,1,2]:
                    if np.random.rand() < .5:
                        img = img.flip(axis)
                        m = m.flip(axis)

            else:
                searching = True
                while searching:
                    s = self.samples[np.random.randint(len(self.samples))]
                    D,H,W = self.data[s].shape
                    d = np.random.randint(D - 64)
                    h = np.random.randint(H - 64)
                    w = np.random.randint(W - 64)
#                   What particles are partially within this voxel?
                    searching =  False
                    for kk in range(5):
                        t = TARGETS[kk]
                        inside = (self.L[samples[k]][t][:,0] > d - shell[t])* \
                                 (self.L[samples[k]][t][:,1] > h - shell[t])* \
                                 (self.L[samples[k]][t][:,2] > w - shell[t])* \
                                 (self.L[samples[k]][t][:,0] < d + 64 + shell[t])* \
                                 (self.L[samples[k]][t][:,1] < h + 64 + shell[t])* \
                                 (self.L[samples[k]][t][:,2] < w + 64 + shell[t])
                        if inside.sum() > 0: searching = True

                img = torch.as_tensor(self.data[s][
                    d:d+64,
                    h:h+64,
                    w:w+64
                ]).to(device)
                m = torch.zeros(64,64,64).long().to(device)
                
            image.append(img)
            mask.append(m)

        image = torch.cat([
            torch.cat(image[:4],-1),
            torch.cat(image[4:8],-1),
            torch.cat(image[8:12],-1),
            torch.cat(image[12:16],-1)
        ],-2)
        mask = torch.cat([
            torch.cat(mask[:4],-1),
            torch.cat(mask[4:8],-1),
            torch.cat(mask[8:12],-1),
            torch.cat(mask[12:16],-1)
        ],-2)
#       Wrap
        for k in range(4):
                drift = np.random.randint(64)
                image[:,k*64:(k+1)*64] = torch.nn.functional.pad(
                    image[:,k*64:(k+1)*64],
                    (64,64),
                    mode='circular'
                )[:,:,drift:drift+256]
                mask[:,k*64:(k+1)*64] = torch.nn.functional.pad(
                    mask[:,k*64:(k+1)*64],
                    (64,64),
                    mode='circular'
                )[:,:,drift:drift+256]
#       Rot90
        angle = np.random.randint(4)
        image = torch.rot90(image, k=angle, dims=(-2, -1))
        mask = torch.rot90(mask, k=angle, dims=(-2, -1))
#       CONTRAST
        image *= np.random.normal(1,CONTRAST_AUG)
#       BRIGTHNESS
        image += np.random.normal(0,BRIGTHNESS_AUG)
#       Flip party
        for axis in [0,1,2]:
            if np.random.rand() < .5:
                image = image.flip(axis)
                mask = mask.flip(axis)

        return image,mask


ds = CryoET_Mosaic_Dataset(V,L,SAMPLES)
len(ds)


image,mask = ds.__getitem__(np.random.randint(len(ds)))
plt.imshow(image.sum(0).cpu() + mask.sum(0).cpu())
plt.show()


del ds
gc.collect()


# Callback to shuffle tds
def cb(self):
    learn.dls.train_ds.__shuffle__()
shuffle_cb = Callback(before_epoch=cb)


class CryoET_Dataset(Dataset):
    def __init__(self, V, L, samples, VALID=False, D=64, H=128, W=128):
        self.DHW = (D,H,W)
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
        self.zyx = torch.stack([z,y,x]).float().to(device)
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
        PATCH_D,PATCH_H,PATCH_W = self.DHW
        
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
                r = self.zyx.view(1,3,PATCH_D,PATCH_H,PATCH_W) - (self.L[sample][t][inside] - o).view(-1,3,1,1,1)
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
                r = self.zyx.view(1,3,PATCH_D,PATCH_H,PATCH_W) - (self.L[sample][t][pinside] - o).view(-1,3,1,1,1)
                sm = ((r*r).sum(1).sqrt() < shell[t]).sum(0) > 0
#               Confetti mask
                image[sm] = image[sm][torch.randperm(sm.sum())]

        if not self.VALID:
#           Rot90
            angle = np.random.randint(4)
            image = torch.rot90(image, k=angle, dims=(-2, -1))
            hm = torch.rot90(hm, k=angle, dims=(-2, -1))
#           Rot180
            if np.random.rand() < .5:
                axis = np.random.randint(2) - 2
                image = torch.rot90(image, k=2, dims=(-3, axis))
                hm = torch.rot90(hm, k=2, dims=(-3, axis))
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


ds = CryoET_Dataset(V,L,SAMPLES)
len(ds)


image,mask = ds.__getitem__(np.random.randint(len(ds)))
plt.imshow(image.sum(0).cpu() + mask.sum(0).cpu())


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
#       decoder_channels = (256, 128, 64, 32, 16)[-ENCODER_DEPTH:]
#       decoder_in_channels = (432, 296 , 64, 96, 32)[-ENCODER_DEPTH:]
        
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
                self.classes,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self,X):
        H,W = X.shape[-2:]
        x = self.UNet(X.view(-1,1,H,W))
        
        return x


# https://math-projects.elte.hu/media/works/187/report/tversky_loss_and_variants.pdf
# https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.fbeta_score.html
class myLoss(nn.Module):
    def __init__(
            self,
            beta=4
    ):
        super(myLoss, self).__init__()
        self.beta = beta
        
    def forward(
            self,
            pred,
            label,
            weight=torch.tensor([
                1.,
                2.,
                1.,
                2.,
                1.
            ]).to(device),
            smooth=1e-5
        ):
        pred = pred.softmax(1)
        FB = W = torch.tensor([smooth]).to(device)
        for k in range(len(TARGETS)):
            m = label == k + 1
            if m.sum() > 0:
                bm = m.view(len(m),-1).sum(-1) > 0
                y_true = m[bm].view(bm.sum(),-1).float()
                y_pred = pred[bm,k+1,:,:,:].view(bm.sum(),-1)
#               We'll check the max value other than ith, doesn't needs to evercome all of them to be positive
                y_pred_max = torch.cat([
                    pred[bm,:k+1],
                    pred[bm,k+2:]
                ],1).max(1)[0].detach().view(bm.sum(),-1)
                y_pred = y_pred/(y_pred+y_pred_max)

                TP = (y_pred*y_true).sum(-1)
                FN = ((1-y_pred)*y_true).sum(-1)
                FP = (y_pred*(1-y_true)).sum(-1)
                
                FB = FB + (weight[k]*((1 + self.beta*self.beta)*TP + smooth) / ((1 + self.beta*self.beta)*TP + FP + self.beta*self.beta*FN + smooth)).mean()
                W = W + weight[k]

        return 1 - FB/W


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


for f in FOLDS:
    seed_everything(SEED)
#   model = myUNet2Dto3D(len(TARGETS)+1)
    model = torch.load('/kaggle/input/czii-synthetic-data-pretraining/resnet18_3_segmentation_2Dto3D_synthetic_pretraining')
        
    train_samples = []
    valid_samples = []
    weight = torch.zeros(5)
    for sample in SAMPLES:
        if sample == SAMPLES[f-1]:
            valid_samples = valid_samples + [sample]
        else:
            train_samples = train_samples + [sample]

    tds = CryoET_Mosaic_Dataset(V,L,train_samples)
    vds = CryoET_Dataset(V,L,valid_samples,VALID=True)
    
    tdl = torch.utils.data.DataLoader(tds, batch_size=2, shuffle=True, drop_last=True)
    vdl = torch.utils.data.DataLoader(vds, batch_size=4, shuffle=False)

    dls = DataLoaders(tdl,vdl)

    learn = Learner(
        dls,
        model,
        lr=LR,
        loss_func=DiceLoss(),
        cbs=[
            ShowGraphCallback(),
            shuffle_cb,
#           GradientAccumulation(n_acc=4)
        ]
    )
    learn.fit_one_cycle(EPOCHS)
    torch.save(model,ENCODER_NAME+'_'+str(ENCODER_DEPTH)+'_segmentation_2Dto3D_pre_'+str(f))
    del tdl,vdl,dls,model,learn
    gc.collect()


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
for f in FOLDS:
    model = torch.load(ENCODER_NAME+'_'+str(ENCODER_DEPTH)+'_segmentation_2Dto3D_pre_'+str(f))
    sample = SAMPLES[f-1]
    print(sample)
    volume = torch.as_tensor(V[sample]).to(device)
    MASK = torch.zeros(192,640,640,len(TARGETS)+1)
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
    for k in range(len(TARGETS)):
                    
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

        try:
            preds = torch.tensor(preds).float().to(device)
            dis = (preds.unsqueeze(1) - L[sample][TARGETS[k]].unsqueeze(0))
            dis = (dis*dis).sum(-1).sqrt()
            hits = (dis < .05*radius[TARGETS[k]]).max(0)[0].sum().item()
            miss = (dis > .05*radius[TARGETS[k]]).min(1)[0].sum().item()

            print(TARGETS[k])
            print('hits: ',hits/len(L[sample][TARGETS[k]]))
            print('miss: ',miss/len(preds))
            print()
        except:
            None

    del volume,MASK,model


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

