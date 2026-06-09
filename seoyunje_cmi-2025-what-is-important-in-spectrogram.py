import os 
import random

import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import cv2
from PIL import Image


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import pytorch_lightning as pl
import timm ## Pytorch Image Model

from colorama import Fore, Style
c_ = Fore.BLUE
sr_ = Style.BRIGHT

import warnings
warnings.filterwarnings('ignore')


class CFG:
    sequence_len = 2048
    sampling_rate= 256
    img_size = [256,384]
    num_classes = 18
    model_name = 'tf_efficientnet_b0.ns_jft_in1k'


from scipy import interpolate

def inter_signal(x):
    original_len = len(x)
    interp_func = interpolate.interp1d(np.linspace(0, 1, original_len), x, kind='linear')
    x_interp = interp_func(np.linspace(0, 1, CFG.sequence_len))

    return x_interp


df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
print(f'{sr_}{c_} Shape of DataFrame: {df.shape}')
print(display(df))


acc_list = ['acc_x', 'acc_y', 'acc_z']
rot_list = ['rot_x', 'rot_y', 'rot_z', 'rot_w']

total_list = acc_list + rot_list


label_to_num = {
    'Above ear - pull hair': 0,  # < ------- TARGETS START
    'Cheek - pinch skin': 1,
    'Eyebrow - pull hair': 2,
    'Eyelash - pull hair': 3,
    'Forehead - pull hairline': 4,
    'Forehead - scratch': 5,
    'Neck - pinch skin': 6,
    'Neck - scratch': 7,  # < ------- TARGETS END
    'Drink from bottle/cup': 8,  # < ------- NON-TARGETS START
    'Feel around in tray and pull out an object': 9,
    'Glasses on/off': 10,
    'Pinch knee/leg skin': 11,
    'Pull air toward your face': 12,
    'Scratch knee/leg skin': 13,
    'Text on phone': 14,
    'Wave hello': 15,
    'Write name in air': 16,
    'Write name on leg': 17  # < ------- NON-TARGETS END
}

num_to_label = {k: v for k, v in enumerate(label_to_num)}


# one hot encoding
label_list = []

for label, _ in label_to_num.items():
    df[label] = np.where(df['gesture'] == label, 1.0, 0.0).astype('float32')
    label_list.append(label)

print(label_list)


import librosa

def spectrum_from_imu(sequence, sr=CFG.sampling_rate):

    img_acc = np.zeros((64, 64, len(acc_list)),dtype='float32')    
    img_rot = np.zeros((64, 48, len(rot_list)), dtype='float32')
    
    for i, col in enumerate(acc_list):
        x = sequence[col].values

        x = inter_signal(x)
        
        # RAW SPECTROGRAM
        mel_spec = librosa.feature.melspectrogram(y=x, sr=sr, hop_length=len(x)//64,
                                                  n_fft=256, n_mels=64, fmin=0, fmax=128)
        # LOG TRANSFORM
        width = (mel_spec.shape[1]//8)*8 

        # STANDARDIZE TO -1 TO 1
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]
        mel_spec_db = (mel_spec_db+40)/40
        img_acc[:,:,i] = mel_spec_db
        

    for i, col in enumerate(rot_list):
        x = sequence[col].values
        x = inter_signal(x)

        # RAW SPECTROGRAM
        mel_spec = librosa.feature.melspectrogram(y=x, sr=sr, hop_length=len(x)//48, 
                                                  n_fft=192, n_mels=64, fmin=0, fmax=128)

        width = (mel_spec.shape[1]//8)*8 
        # LOG TRANSFORM
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]
        
        # STANDARDIZE TO -1 TO 1
        mel_spec_db = (mel_spec_db+40)/40
        img_rot[:,:,i] = mel_spec_db
        
    return img_acc, img_rot


for ges in df['gesture'].unique():

    seq_list = np.random.choice(df[df['gesture'] == ges]['sequence_id'].unique(), 1)
    print(f'{c_}{sr_}#'*25)
    print(f"### Gesture Class: {ges}")
    print("#"*25)
    
    for i, seq_id in enumerate(seq_list):
        print(f"\n => Sequence_id: {seq_id}")
    
        tmp = df[df['sequence_id'] == seq_id].reset_index()
        img_acc, img_rot = spectrum_from_imu(tmp, sr=CFG.sampling_rate)
        
        
        plt.figure(figsize=(14,4))

        for i, col in enumerate(total_list):

            if i < len(acc_list): 
                plt.subplot(1,len(total_list),i+1)
                plt.imshow(img_acc[...,i], aspect='auto', origin='lower', cmap='jet')
                plt.xlabel('Time'); plt.ylabel('Frequency')
                plt.title(f'{col}_spectrum', size=10, fontweight='bold')

            else: 
                plt.subplot(1,len(total_list),i+1)
                plt.imshow(img_rot[...,i-len(acc_list)], aspect='auto', origin='lower', cmap='jet')
                plt.xlabel('Time'); plt.ylabel('Frequency')
                plt.title(f'{col}_spectrum', size=10, fontweight='bold')

        plt.tight_layout()
        plt.show()


class CMI_Dataset(Dataset):
    def __init__(self, data, col):
        super(CMI_Dataset, self).__init__()
        self.data = data
        self.seq_list = data['sequence_id'].unique()
        self.col = col

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):

        X, y = self.__data_generation(index)

        X = torch.tensor(X, dtype=torch.float32).permute(2,0,1)
        y = torch.tensor(y, dtype=torch.float32)

        return X, y

    def __data_generation(self, index):
        X = np.zeros((128, 192, 3),dtype='float32')
        y = np.zeros(CFG.num_classes, dtype='float32')
        
        seq_id = self.seq_list[index]

        row = self.data[self.data['sequence_id'] == seq_id]

        img_acc, img_rot = spectrum_from_imu(row, sr=CFG.sampling_rate)

        img_acc = np.concatenate([img_acc[:,:,0], img_acc[:,:,1], img_acc[:,:,2]], axis=1)
        img_rot = np.concatenate([img_rot[:,:,0], img_rot[:,:,1], img_rot[:,:,2], img_rot[:,:,3]], axis=1)

        img = np.concatenate([img_acc, img_rot], axis=0)

        X = np.tile(img[..., np.newaxis], (1, 1, 3))
        X = cv2.resize(X, (CFG.img_size[1], CFG.img_size[0]))

        y = row[label_list].values[0]
        
        return X, y


def mask2contour(mask, width=3):
    w = mask.shape[1]
    h = mask.shape[0]

    mask2 = np.concatenate([mask[:, width:], np.zeros((h,width))], axis=1)
    mask2 = np.logical_xor(mask,mask2)
    
    mask3 = np.concatenate([mask[width:, :], np.zeros((width,w))], axis=0)
    mask3 = np.logical_xor(mask, mask3)

    return np.logical_or(mask2, mask3)


class CMI_EffNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.feature_extractor = timm.create_model(
            CFG.model_name, pretrained=True, features_only=True
        )
        self.channels = self.feature_extractor.feature_info[-1]['num_chs']
        self.dropout = nn.Dropout(0.5)

        self.head = nn.Sequential(
            nn.Linear(self.channels, self.channels//2, bias=False),
            nn.BatchNorm1d(self.channels//2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(self.channels//2, CFG.num_classes)
        )


    def forward(self, x):
        feats = self.feature_extractor(x)  # (B, C, H, W)
        feats = self.dropout(feats[-1])
        feats = F.adaptive_avg_pool2d(feats, 1).reshape(x.size(0),-1) # (B, C)
        out = self.head(feats)
        
        return out


model = CMI_EffNet.load_from_checkpoint("/kaggle/input/cmi-2025-efficientnet/last_fold2.ckpt")

activations = {}
grads = {}

def forward_hook(module, input, output):
    activations["value"] = output.detach()

def backward_hook(module, grad_input, grad_output):
    grads["value"] = grad_output[0]  # grad_output은 tuple


target_layer_name = model.feature_extractor.feature_info[-1]['module']
target_layer = dict(model.feature_extractor.named_modules())[target_layer_name]

handle_f = target_layer.register_forward_hook(forward_hook)
handle_b = target_layer.register_full_backward_hook(backward_hook)


model.eval()
gclahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(8,8))

for i, ges in enumerate(df['gesture'].unique()):

    print(f"{c_}{sr_}#"*25)
    print(f'### Gesture: {ges}')
    print("#"*25)
    print("\n")
    
    tmp = df[df['gesture'] == ges].reset_index(drop=True)
    ds = CMI_Dataset(tmp, col=total_list)
    data_loader = DataLoader(ds, shuffle=False, batch_size=8)
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    axes = axes.flatten()

    imgs, tars = next(iter(data_loader))

    preds = model(imgs)  # forward pass

    target = (preds * tars).sum() 
    model.zero_grad()
    target.backward() 
    

    for k in range(imgs.size(0)):

        img = imgs[k].detach().cpu().permute(1,2,0).numpy()[...,0]
        img -= img.min()
        img /= img.max()
        img = (img * 255).astype('uint8')
        img = gclahe.apply(img)
    
        feature_map = activations["value"][k].unsqueeze(0)  # (1,C,H,W)
        grad_map = grads["value"][k].unsqueeze(0)           # (1,C,H,W)

        weights = grad_map.mean(dim=(2,3), keepdim=True)    # (1,C,1,1)
        cam = F.relu((weights * feature_map).sum(dim=1, keepdim=True))  # (1,1,H,W)
        cam = F.interpolate(cam, size=CFG.img_size, mode='bilinear', align_corners=False)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize 0~1

        cam[cam >= 0.7] = 100
        cam[cam < 0.7] = 0
        cam = mask2contour(cam)

        mx = np.max(img)

        img[cam > 0] = mx

        axes[k].imshow(img, aspect='auto', origin='lower')
        axes[k].axis('off')
    
    plt.tight_layout()
    plt.show()

# Hook 제거
handle_f.remove()
handle_b.remove()





