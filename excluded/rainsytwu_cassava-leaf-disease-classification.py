import os

OUTPUT_DIR = "./"
TEST_PATH = '../input/cassava-leaf-disease-classification/test_images'


import sys
sys.path.append('../input/cassava_models/')
sys.path.append('../input/pytorchimagemodels')

import time
import random
from functools import partial
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
from torch.utils.data import DataLoader, Dataset
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose, HueSaturationValue, 
    )
from albumentations.pytorch import ToTensorV2

import timm
from timm.models.vision_transformer import VisionTransformer

import warnings 
warnings.filterwarnings('ignore')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def seed_torch(seed=1006):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch()


test = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')
test.head()


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{TEST_PATH}/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image


def get_transforms(*, data, vit=False):
    
    if vit:
        MEAN = [0.5, 0.5, 0.5]
        STD = [0.5, 0.5, 0.5]
    else:
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
    
    if data == 'train':
        return Compose([
            RandomResizedCrop(IMG_SIZE, IMG_SIZE),
            #RandomCrop(IMG_SIZE, IMG_SIZE),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            #HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            #RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            #Resize(IMG_SIZE, IMG_SIZE),
            Normalize(
                mean=MEAN,
                std=STD,
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(IMG_SIZE, IMG_SIZE),
            Normalize(
                mean=MEAN,
                std=STD,
            ),
            ToTensorV2(),
        ])


class NetVit(nn.Module):
    def __init__(self, model_name, pretrained=False, n_class=5, att_activate=False, no_att=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Identity()
        
        if att_activate:
            self.att_layer = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
            )
        else:
            if no_att:
                pass
            else:
                self.att_layer = nn.Linear(n_features, 1)
            
        self.head = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        output = self.head(x)
        return output


class NetVit4(nn.Module):
    def __init__(self, model_name, pretrained=False, n_class=5, att_activate=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Identity()
        if att_activate:
            self.att_layer = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
            )
        else:
            self.att_layer = nn.Linear(n_features, 1)
            
        self.head = nn.Linear(n_features, n_class)

    def forward(self, x):
        l = x.shape[2] // 2
        h1 = self.model(x[:, :, :l, :l])
        h2 = self.model(x[:, :, :l, l:])
        h3 = self.model(x[:, :, l:, :l])
        h4 = self.model(x[:, :, l:, l:])

        a1 = self.att_layer(h1)
        a2 = self.att_layer(h2)
        a3 = self.att_layer(h3)
        a4 = self.att_layer(h4)

        w = F.softmax(torch.cat([a1, a2, a3, a4], dim=1), dim=1)

        h = h1 * w[:, 0].unsqueeze(-1) + h2 * w[:, 1].unsqueeze(-1) + \
            h3 * w[:, 2].unsqueeze(-1) + h4 * w[:, 3].unsqueeze(-1)
        output = self.head(h)
        return output


from collections import OrderedDict

def inference(model, states, test_loader, device, temp=1):
    model.to(device)
    preds = []
    for state in states:
        pred = []
        model.load_state_dict(state)
        model.eval()
        for i, image in enumerate(test_loader):
            with torch.no_grad():
                pred.append((model(image.to(device))*temp).softmax(1).to('cpu'))
        pred = torch.cat(pred, dim=0)
        preds.append(pred.numpy())
    return np.mean(preds, axis=0)


def multi2single(path, se=False):
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('se_module', 'dummy')
            k = k.replace('module.', '')
            k = k.replace('dummy', 'se_module')
        if 'attention_linear' in k:
            k = k.replace('attention_linear', 'att_layer')
        new_state_dict[k] = v
    return new_state_dict


temp = 1.0


    MODEL_NAME = "vit_base_patch16_384"
    MODEL_NUM = "single_vit"
    MODEL_DIR = "../input/cassava_models/pytorch/default/1/"
    IMG_SIZE = 384
    TTA = 3
    BATCH = 32

    model = NetVit(MODEL_NAME, pretrained=False, no_att=True)
    states = [multi2single(MODEL_DIR+f'{MODEL_NUM}_{fold+3}.pth') for fold in range(1)]
    if TTA == 1:
        test_dataset = TestDataset(test, transform=get_transforms(data='valid', vit=True))
    else:
        test_dataset = TestDataset(test, transform=get_transforms(data='train', vit=True))

    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    vit_predictions = np.zeros((len(test), 5))
    for _ in range(TTA):
        vit_predictions += inference(model, states, test_loader, device, temp) / TTA


if True:
    MODEL_NAME = "vit_base_patch16_224"
    MODEL_NUM = "single_vit4_type_A"
    MODEL_DIR = "../input/cassava_models/pytorch/default/1/"
    IMG_SIZE = 448
    TTA = 3
    BATCH = 32
    att_activate = True  # Changed to True to match pretrained weights

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NetVit4(MODEL_NAME, pretrained=False, att_activate=att_activate)    
    states = [multi2single(MODEL_DIR+f'{MODEL_NUM}_{fold+4}.pth') for fold in range(1)]
    if TTA == 1:
        test_dataset = TestDataset(test, transform=get_transforms(data='valid', vit=True))
    else:
        test_dataset = TestDataset(test, transform=get_transforms(data='train', vit=True))

    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    vit4_predictions_a = np.zeros((len(test), 5))
    for _ in range(TTA):
        vit4_predictions_a += inference(model, states, test_loader, device, temp=1.0) / TTA


if True:
    MODEL_NAME = "vit_base_patch16_224"
    MODEL_NUM = "single_vit4_type_B"
    MODEL_DIR = "../input/cassava_models/pytorch/default/1/"
    IMG_SIZE = 448
    TTA = 3
    BATCH = 32
    att_activate = False

    model = NetVit4(MODEL_NAME, pretrained=False, att_activate=att_activate)    
    states = [multi2single(MODEL_DIR+f'{MODEL_NUM}_{fold+4}.pth') for fold in range(1)]
    if TTA == 1:
        test_dataset = TestDataset(test, transform=get_transforms(data='valid', vit=True))
    else:
        test_dataset = TestDataset(test, transform=get_transforms(data='train', vit=True))

    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    vit4_predictions_b = np.zeros((len(test), 5))
    for _ in range(TTA):
        vit4_predictions_b += inference(model, states, test_loader, device, temp) / TTA


predictions = (vit_predictions * 0.45 + vit4_predictions_a * 0.55) / 9 * 10 + vit4_predictions_b * 0.08


# submission
test['label'] = predictions.argmax(1)
test[['image_id', 'label']].to_csv(OUTPUT_DIR+'submission.csv', index=False)
test.head()

