import json
import matplotlib.pyplot as plt
import seaborn as sns
import zarr
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import gc
import cc3d

import torchvision
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from fastai.vision.all import *
import segmentation_models_pytorch as smp

device = 'cuda' if torch.cuda.is_available() else 'cpu'


PATH = '/kaggle/input/czii-cryo-et-object-identification/test/static/ExperimentRuns/'
SAMPLES = [
    x for x in os.listdir(PATH)
]
TARGETS = [
    'apo-ferritin',# easy
    'beta-galactosidase',# hard
    'ribosome',# easy
    'thyroglobulin',# hard
    'virus-like-particle'# easy
]
MODELS = [
    '/kaggle/input/resnet18-3-segmentation-2dto3d/resnet18_3_segmentation_2Dto3D_1',
    '/kaggle/input/resnet18-3-segmentation-2dto3d/resnet18_3_segmentation_2Dto3D_2',
    '/kaggle/input/resnet18-3-segmentation-2dto3d/resnet18_3_segmentation_2Dto3D_3',
    '/kaggle/input/resnet18-3-segmentation-2dto3d/resnet18_3_segmentation_2Dto3D_4',
    '/kaggle/input/resnet18-3-segmentation-2dto3d/resnet18_3_segmentation_2Dto3D_5',
    '/kaggle/input/resnet18-3-segmentation-2dto3d/resnet18_3_segmentation_2Dto3D_6',
    '/kaggle/input/resnet18-3-segmentation-2dto3d/resnet18_3_segmentation_2Dto3D_7'
]


class repack_3D(nn.Module):
    def __init__(
        self
        ):
        super(repack_3D, self).__init__()

    def forward(self,X):
        D,C,H,W = X.shape[-4:]
        return X.view(-1,D,C,H,W).permute(0,2,1,3,4)


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
                nn.Dropout(DROPOUT),
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


if 1:#len(SAMPLES) > 3:
    models = []
    for path in MODELS:
#       models.append(torch.load(path,map_location=device).eval())
        models.append(torch.nn.DataParallel(torch.load(path).eval(), device_ids=[0, 1]))

    experiment = []
    particle_type = []
    x = []
    y = []
    z = []
    p = (1,99)
    rot = [0,1,2,3,0,1,2,3,0]
    y_pred = torch.zeros(2,6,36,640,640).float().to(device)
    with torch.no_grad():
        for sample in tqdm(SAMPLES):
            MASK = []
            file = zarr.open(PATH + sample + '/VoxelSpacing10.000/denoised.zarr', mode='r')
            scale = np.array([file.attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']])
            volume =np.array(file[0])
            pmin,pmax = np.percentile(volume,p)
            volume = (volume - pmin)/(pmax - pmin)
            volume = torch.as_tensor(volume)
            volume = torch.nn.functional.pad(
                volume.unsqueeze(0),
                (
                    5,5,
                    5,5
                ),
                mode='reflect'
            )[0]

            v = torch.stack(
                volume[:36],
                torch.rot90(volume[:36],1,(-2,-1)
            ).to(device)

            y_pred.zero_()
            for model in models:
                y_pred += model(v).softmax(1)

            y_pred[0] += torch.rot90(y_pred[1],-1,(-2,-1))
            
            MASK.append(y_pred[0,:,:20].argmax(1).cpu())
            mask = y_pred[0,:,20:].clone()
            
            for k in range(1,7):
#               Drill Scan
                v = torch.stack(
                    torch.rot90(volume[k*16+4:k*16+36],k=rot[k],dims=(-2,-1)),
                    torch.rot90(volume[k*16+4:k*16+36],k=rot[k]+1,dims=(-2,-1))
                ).to(device)

                y_pred.zero_()
                for model in models:
                    y_pred[:,:,:32] += model(v).softmax(1)
                
                y_pred[0,:,:32] += torch.rot90(y_pred[1,:,:32],-1,(-2,-1))

                y_pred[0,:,:32] = torch.rot90(y_pred[0,:,:32],k=-rot[k],dims=(-2,-1))
                y_pred[0,:,:16] += mask
                MASK.append((y_pred[0,:,:16]).argmax(1).cpu())
                mask = y_pred[0,:,16:32].clone()

            v = torch.stack(
                torch.rot90(volume[-36:],k=rot[7],dims=(-2,-1)),
                torch.rot90(volume[-36:],k=rot[7]+1,dims=(-2,-1))
            ).to(device)

            y_pred.zero_()
            for model in models:
                y_pred += model(v).softmax(1)

            y_pred[0] += torch.rot90(y_pred[1],-1,(-2,-1))

            y_pred[0] = torch.rot90(y_pred[0],k=-rot[7],dims=(-2,-1))
            y_pred[0,:,:16] += mask
            MASK.append(y_pred[0].argmax(1).cpu())
        
            mask = torch.concat(MASK)[:,5:-5,5:-5].numpy()
            for k in range(5):
                stats = cc3d.statistics(cc3d.connected_components(mask == k+1))
                preds = stats['centroids'][1:]*scale
                experiment = experiment + [sample]*len(preds)
                particle_type = particle_type + [TARGETS[k]]*len(preds)
                x = x + list(preds[:,2])
                y = y + list(preds[:,1])
                z = z + list(preds[:,0])

else:
    experiment = SAMPLES
    particle_type = [TARGETS[0]]*len(SAMPLES)
    x = [0]*len(SAMPLES)
    y = [0]*len(SAMPLES)
    z = [0]*len(SAMPLES)
    

pd.DataFrame({
    'id':np.arange(len(experiment)),
    'experiment':experiment,
    'particle_type':particle_type,
    'x':x,
    'y':y,
    'z':z
}).to_csv('submission.csv',index=False)

