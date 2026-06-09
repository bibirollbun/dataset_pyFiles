%%bash

if [ ! -d "/kaggle/working/train" ]; then
    unzip -q /kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip -d /kaggle/working
fi

if [ ! -d "/kaggle/working/test" ]; then
    unzip -q /kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip -d /kaggle/working
fi



import os
import glob
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import cv2
import timm
from torch.optim.lr_scheduler import _LRScheduler
import math
import pandas as pd
import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
import albumentations as A
from albumentations.pytorch import ToTensorV2   
from sklearn.model_selection import KFold
import pprint


class Config:
    dog = 1
    cat = 0
    train_dir = '/kaggle/working/train'
    test_dir = '/kaggle/working/test'
    n_fold = 5
    num_workers = 16
    pin_memory = True
    batch_size = 64
    seed = 2025
    drop_last = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 2
    early_stopping = 3
    lr = 1e-4
    optimizer = torch.optim.AdamW
    warmup_epochs = 0
    criterion = nn.BCEWithLogitsLoss()
    size = (224, 224)

cfg = Config()


def seed_everything(seed=cfg.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            warmup_lr = [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
            return warmup_lr
        else:
            # Cosine Annealing phase
            cos_anneal_lr = [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / 
                              (self.total_epochs - self.warmup_epochs))) / 2
                for base_lr in self.base_lrs
            ]
            return cos_anneal_lr

def square_pad_and_resize(image, size):
    h, w, _ = image.shape
    
    max_dim = max(h, w)
    
    top = (max_dim - h) // 2
    bottom = max_dim - h - top
    left = (max_dim - w) // 2
    right = max_dim - w - left
    
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    resized_image = cv2.resize(padded_image, (size))
    return resized_image

class DC_Dataset(Dataset):
    def __init__(self, paths, valid=False):
        super().__init__()
        self.paths = paths
        self.valid = valid

        if not self.valid:
            self.transform = transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=180, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RGBShift(p=0.5),
                A.RandomSizedCrop(min_max_height=(cfg.size[0], cfg.size[0]//2), height=cfg.size[0], width=cfg.size[1]),
                A.Normalize(),
                ToTensorV2(),
            ])
        else:
            self.transform = transform = A.Compose([
                A.Normalize(),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = cv2.resize(cv2.imread(self.paths[index]), cfg.size)
        img = self.transform(image=img)['image']
        return img

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return torch.mean(x.clamp(min=self.eps).pow(self.p), dim=(-1,-2)).pow(1.0 / self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"

class Distill_Model(pl.LightningModule):
    def __init__(self, model_name='convnext_small', teacher_path=None, pretrained=True, num_batch=0, fold=0):
        super().__init__()
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1,  # ヘッドを後で定義
        )
        if not ('vit' in model_name or 'mobile' in model_name):
            num_features = self.model.head.in_features
            self.model.head = nn.Sequential(
                GeM(),
                nn.Linear(self.model.head.in_features, 1)
            )
        self.teacher = timm.create_model(
            'convnext_small',
            pretrained=pretrained,
            num_classes=1,  # ヘッドを後で定義
        )
        num_features = self.teacher.head.in_features
        self.teacher.head = nn.Sequential(
            GeM(),
            nn.Linear(self.teacher.head.in_features, 1)
        )
        #teacher_weight = torch.load(teacher_path,map_location='cpu')['state_dict']
        #for key in list(teacher_weight):
        #    teacher_weight[key.replace("model.", "")] = teacher_weight.pop(key)

        #self.teacher.load_state_dict(teacher_weight)
        #self.teacher.eval()

        self.fold = fold
        self.num_batch = num_batch
        self.criterion = cfg.criterion
        self.model_name = model_name
        self.pretrained = pretrained

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x).squeeze()

    def training_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        label = cfg.distill_ratio*torch.sigmoid(self.teacher(img).squeeze()) + label*(1-cfg.distill_ratio)
        loss = self.criterion(output, label)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.criterion(output, label)
        pred = torch.sigmoid(output) > 0.5
        acc = (pred == label).float().mean()
        if torch.isnan(loss):
            return None

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.criterion(output, label)
        pred = torch.sigmoid(output) > 0.5
        acc = (pred == label).float().mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = cfg.optimizer(self.model.parameters(), lr=cfg.lr)
        scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=cfg.warmup_epochs*self.num_batch, total_epochs=cfg.epochs*self.num_batch+1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

def collate(x):
    return


test_paths = glob.glob(cfg.test_dir + '/*')
image_ids = []

for path in test_paths:
    image_ids.append(os.path.basename(path).split('.')[0])


model_paths = glob.glob('/kaggle/input/dogs-vs-cats-distillation/lightning_logs/version_*/checkpoints/*.ckpt')
pprint.pprint(model_paths)


outputs = []


test_dataset = DC_Dataset(test_paths, valid=True)
test_loader = DataLoader(
    test_dataset, batch_size=cfg.batch_size, shuffle=False,
    num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
)


for model_path in model_paths:
    model = Distill_Model.load_from_checkpoint(model_path)
    model.eval()
    outputs.append([])
    with torch.no_grad():
        for img in tqdm.tqdm(test_loader):
            output = model(img.to(cfg.device))
            outputs[-1] += output.tolist()



outputs = torch.tensor(outputs)
outputs = outputs.mean(dim=0)
outputs = torch.sigmoid(outputs)


for clip in [0.01, 0.005, 0.015, 0.0125, 0.0025, 0, 0.0075, 0.004]:
    submission = pd.DataFrame({'id':image_ids, 'label':torch.clamp(outputs, min=clip, max=1-clip).tolist()})
    submission['id'] = pd.to_numeric(submission['id'])
    submission = submission.sort_values(by='id')
    submission.to_csv(f'/kaggle/working/submission-clip={clip}.csv', index=False)


submission


!rm -r /kaggle/working/train
!rm -r /kaggle/working/test




