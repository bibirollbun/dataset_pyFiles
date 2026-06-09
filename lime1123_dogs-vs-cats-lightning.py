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
import gc
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2   
from sklearn.model_selection import KFold
import tqdm


class Config:
    dog = 0.999
    cat = 0.001
    train_dir = '/kaggle/working/train'
    test_dir = '/kaggle/working/test'
    pretrain_dir = "/kaggle/input/cat-dog-images-for-classification/cat_dog"
    n_fold = 5
    num_workers = 4
    pin_memory = True
    batch_size = 64
    seed = 2025
    drop_last = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 2
    early_stopping = 20
    lr = 1e-5
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
        self.labels = []
        self.paths = paths
        self.valid = valid
        for path in self.paths:
            if 'dog' in os.path.basename(path):
                self.labels.append(1 if self.valid else cfg.dog)
            elif 'cat' in os.path.basename(path):
                self.labels.append(0 if self.valid else cfg.cat)

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
        label = torch.tensor(self.labels[index],dtype=torch.float32)
        return img, label

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return torch.mean(x.clamp(min=self.eps).pow(self.p), dim=(-1,-2)).pow(1.0 / self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"

class DC_Model(pl.LightningModule):
    def __init__(self, model_name='convnext_small', pretrained=True, num_batch=0, fold=0):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1,  # ヘッドを後で定義
        )
        if not 'vit' in model_name:
            num_features = self.model.head.in_features
            self.model.head = nn.Sequential(
                GeM(),
                nn.Linear(self.model.head.in_features, 1)
            )

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
        optimizer = cfg.optimizer(self.parameters(), lr=cfg.lr)
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


image_paths = glob.glob(cfg.train_dir + '/*')
kf = KFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)


def train():
    for fold, (train_index, valid_index) in enumerate(kf.split(image_paths)):
        train_paths = [image_paths[i] for i in train_index]
        valid_paths = [image_paths[i] for i in valid_index]

        train_dataset = DC_Dataset(
            paths=train_paths,
            valid=False
        )
        valid_dataset = DC_Dataset(
            paths=valid_paths,
            valid=True
        )

        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=cfg.drop_last,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        )

        trainer = pl.Trainer(
            max_epochs=cfg.epochs,
            accelerator='gpu',
            devices=2,
            strategy='ddp_notebook',
            precision='16-mixed',
            log_every_n_steps=1,
            enable_checkpointing=False,
            gradient_clip_val=1.0,
        )

        model = DC_Model(num_batch=len(train_loader), fold=fold)

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=cfg.early_stopping,
            mode='min',
            verbose=True
        )
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=f'/kaggle/working/lightning_logs/version_{trainer.logger.version}/checkpoints',
            filename=f'{model.model_name}-fold{fold}-{cfg.seed}-{{epoch:02d}}-{{val_loss:.4f}}',
            save_top_k=1,
            mode='min',
            verbose=True,
            save_last=True
        )

        trainer.callbacks += [checkpoint_callback, early_stopping]
        trainer.fit(model, train_loader, valid_loader)

train()


!rm -r /kaggle/working/train
!rm -r /kaggle/working/test


!zip -r /kaggle/working/lightning_logs.zip /kaggle/working/lightning_logs




