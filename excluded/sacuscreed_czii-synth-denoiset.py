!pip install segmentation_models_pytorch==0.3.3
!pip install connected-components-3d
!pip install zarr
!pip3 install topaz-em

import json
import matplotlib.pyplot as plt
import zarr
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import gc
import cc3d
import topaz.denoise

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
    'TS_19', 'TS_17', 'TS_10',
    'TS_13', 'TS_25', 'TS_26', 'TS_15',
    'TS_5', 'TS_8', 'TS_20', 'TS_4',
    'TS_0', 'TS_12', 'TS_2', 'TS_24',
    'TS_22', 'TS_18', 'TS_16', 'TS_3',
    'TS_6', 'TS_11', 'TS_9', 'TS_23',
    'TS_7', 'TS_14', 'TS_21', 'TS_1'
]
TARGETS = [
    'apo-ferritin',# easy
    'beta-galactosidase',# hard
    'ribosome',# easy
    'thyroglobulin',# hard
    'virus-like-particle'# easy
]
SEED = 1337
ENCODER_NAME = "resnet18"
ENCODER_DEPTH = 3
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
BS = 4
LR = 2.5e-5
EPOCHS = 1
CONTRAST_AUG = .25
BRIGTHNESS_AUG = .25


target_to_path = {
    'apo-ferritin':'101/ferritin_complex-1.0_orientedpoint.ndjson',
    'beta-galactosidase':'103/beta_galactosidase-1.0_orientedpoint.ndjson',
    'ribosome':'104/cytosolic_ribosome-1.0_orientedpoint.ndjson',
    'thyroglobulin':'105/thyroglobulin-1.0_orientedpoint.ndjson',
    'virus-like-particle':'106/pp7_vlp-1.0_orientedpoint.ndjson'    
}


V = {}
L = {}
for sample in SYNTH_SAMPLES:
        L[sample] = {}
        file = zarr.open('/kaggle/input/czii-synth-data/'+sample+'.zarr', mode='r')
        scale = file.attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']
        vol = np.array(file[0])
        pmin,pmax = np.percentile(vol,(1,99))
        V[sample] = (vol - pmin)/(pmax - pmin)
        V[sample] = torch.as_tensor(V[sample])
        V[sample] = torch.nn.functional.pad(
            V[sample].unsqueeze(0),
            (
                5,5,
                5,5
            ),
            mode='reflect'
        )[0]
        for target in TARGETS:
            L[sample][target] = []
            for p in pd.read_json('/kaggle/input/czii-synth-data/synth_labels/'+sample+'/'+target+'.ndjson', lines=True)['location']:
                L[sample][target].append([
                    p['z'],
                    p['y'] + 5,
                    p['x'] + 5
                ])
            L[sample][target] = torch.tensor(L[sample][target]).float().to(device)


for sample in SAMPLES:
    L[sample] = {}
    file = zarr.open(VOLUMES_PATH + sample + '/VoxelSpacing10.000/denoised.zarr', mode='r')
    for target in TARGETS:
        L[sample][target] = []
        f = open(LABELS_PATH + sample + '/Picks/' + target + '.json')
        for p in json.loads(f.read())['points']:
            L[sample][target].append([
                p['location']['z']/scale[0],
                p['location']['y']/scale[1],
                p['location']['x']/scale[2]
            ])
        L[sample][target] = torch.tensor(L[sample][target]).float().to(device)


for sample in SAMPLES:
    L[sample]['distribution'] = []
    for target in TARGETS:
        L[sample]['distribution'].append(len(L[sample][target]))
    L[sample]['distribution'] = torch.tensor(L[sample]['distribution']).float()
    L[sample]['distribution'] /= L[sample]['distribution'].sum()
    plt.plot(L[sample]['distribution'],'lightgreen')
for sample in SYNTH_SAMPLES:
    L[sample]['distribution'] = []
    for target in TARGETS:
        L[sample]['distribution'].append(len(L[sample][target]))
    L[sample]['distribution'] = torch.tensor(L[sample]['distribution']).float()
    L[sample]['distribution'] /= L[sample]['distribution'].sum()
    plt.plot(L[sample]['distribution'],'r')


t = []
for sample in SYNTH_SAMPLES:
    total = 0
    for target in TARGETS:
        total += len(L[sample][target])
    t.append(total)
    L[sample]['t'] = total
plt.bar(np.arange(len(t)),t)


# 1 FOLD x3 and 6 FOLD x4
SYNTH_SAMPLES.sort(key=lambda k:L[k]['t'],reverse=True)
t = []
for sample in SYNTH_SAMPLES:
    total = 0
    for target in TARGETS:
        total += len(L[sample][target])
    t.append(total)
    L[sample]['t'] = total
plt.bar(np.arange(len(t)),t)


FOLDS = [SYNTH_SAMPLES[:3]]
FOLDS[0]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


REMAINING = SYNTH_SAMPLES[3:]
seed_everything(SEED)
random.shuffle(REMAINING)
for k in range(6):
    FOLDS.append(REMAINING[k*4:k*4+4])
    print(FOLDS[-1])


#2nd sample has the higher amount of particles
t = []
for sample in SAMPLES:
    total = 0
    for target in TARGETS:
        total += len(L[sample][target])
    t.append(total)
    L[sample]['t'] = total
plt.bar(np.arange(len(t)),t)


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
            d = d*32 + 4
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

        if d > 200 - 64: d = 200 - 64
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
#           image *= np.random.normal(1,CONTRAST_AUG)
#           BRIGTHNESS
#           image += np.random.normal(0,BRIGTHNESS_AUG)
#           To flip or not to flip, that is the question
            if np.random.rand() < .5:
                axis = np.random.randint(2) - 2
                image = image.flip(axis)
                hm = hm.flip(axis)

        return image,hm,searching
    
    def __getitem__(self, idx):
        
        image,hm,searching = self.__rawgetitem__(idx)
        if not self.VALID:
            while searching:
                image,hm,searching = self.__rawgetitem__(np.random.randint(self.__len__()))

        return image,hm


ds = CryoET_Dataset(V,L,SYNTH_SAMPLES)
len(ds)


image,heatmap = ds.__getitem__(np.random.randint(len(ds)))
plt.imshow(image.sum(0).cpu() + heatmap.sum(0).cpu())
plt.show()


del ds
gc.collect()


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
                classes,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self,X):
        H,W = X.shape[-2:]
        x = self.UNet(X.view(-1,1,H,W))
        
        return x


class myDenoisET(nn.Module):
    def __init__(
        self,
        ):
        super(myDenoisET, self).__init__()    
        
        self.model = topaz.denoise.Denoise3D('unet').model.to(device)
        
    def forward(self,X):
        H,W = X.shape[-2:]
        x = self.model(X.view(-1,1,H,W))
        
        return x


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


class myLoss(nn.Module):
    def __init__(self, model, weight=None):
        super(myLoss, self).__init__()
        self.Loss = DiceLoss(weight)
        self.model = model.eval()

    def forward(self, predict, target):
        predict = self.model(predict.view(-1,PATCH_D,PATCH_H,PATCH_W))

        return self.Loss(predict, target)


for f in range(len(FOLDS)):
    seed_everything(SEED)
    model = torch.nn.DataParallel(myDenoisET(), device_ids=[0,1])

    train_samples = concat(*FOLDS[:f],*FOLDS[f+1:])
    valid_samples = FOLDS[f]

    tds = CryoET_Dataset(V,L,train_samples)
    vds = CryoET_Dataset(V,L,valid_samples,VALID=True)

    tdl = torch.utils.data.DataLoader(tds, batch_size=BS, shuffle=True, drop_last=True)
    vdl = torch.utils.data.DataLoader(vds, batch_size=BS, shuffle=False)

    dls = DataLoaders(tdl,vdl)

    learn = Learner(
        dls,
        model,
        lr=LR,
        loss_func=myLoss(torch.nn.DataParallel(torch.load(
            '/kaggle/input/resnet18-3-segmentation-2dto3d/resnet18_3_segmentation_2Dto3D_'+str([2,1,3,4,5,6,7][f])#2nd sample has the higher amount of particles
        ), device_ids=[0,1])),
        cbs=[ShowGraphCallback()]
    )
    learn.fit_one_cycle(EPOCHS)
    torch.save(model,'myDenoisET_'+str([2,1,3,4,5,6,7][f]))#2nd sample has the higher amount of particles
    del tdl,vdl,dls,model,learn
    gc.collect()

