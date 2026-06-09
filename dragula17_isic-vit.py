!pip install torcheval


import os
import gc
import time
import math
import copy
import glob
import random
import warnings
import joblib

# System settings
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings("ignore")

# Numerical & Data Processing
import numpy as np
import pandas as pd

# Visualization
import cv2
import matplotlib.pyplot as plt

# PyTorch & Related
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

# PyTorch Metrics
from torcheval.metrics.functional import binary_auroc

# Scikit-learn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

# Image Models
import timm

# Data Augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Utility Libraries
from tqdm import tqdm
from collections import defaultdict

# Colored Terminal Text
from colorama import Fore, Style
b_ = Fore.BLUE
sr_ = Style.RESET_ALL


# Config
class Config:
    seed = 42
    T_max = None
    n_fold = 5
    epochs = 50
    train_batch_size = 32
    img_size = 384
    model_name = "vit_tiny_r_s16_p8_384"
    checkpoint_path = None
    valid_batch_size = 64
    learning_rate = 1e-4
    scheduler = "CosineAnnealingLR"
    min_lr = 1e-6
    weight_decay = 1e-6
    fold = 0
    n_accumulate = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_t_max(self, df: pd.DataFrame = None):
        if not df:
            self.T_max = 500
            return
        self.T_max = df.shape[0] * (self.n_fold-1) * self.epochs // self.train_batch_size // self.n_fold
        

config = Config()


# Non config constants
ROOT_DIR = "/kaggle/input/isic-2024-challenge"
TRAIN_DIR = f'{ROOT_DIR}/train-image/image'


# utils
def get_train_file_path(image_id):
    return f"{TRAIN_DIR}/{image_id}.jpg"

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


# logic

def get_train_images(train_dir):
    return sorted(glob.glob(f"{train_dir}/*.jpg"))

def get_data_df(root_dir: str) -> pd.DataFrame:
    df = pd.read_csv(f"{root_dir}/train-metadata.csv")
    
    print("        df.shape, # of positive cases, # of patients")
    print("original>", df.shape, df.target.sum(), df["patient_id"].unique().shape)
    
    df_positive = df[df["target"] == 1].reset_index(drop=True)
    df_negative = df[df["target"] == 0].reset_index(drop=True)
    
    df = pd.concat([df_positive, df_negative.iloc[:df_positive.shape[0]*20, :]])  # positive:negative = 1:20
    print("filtered>", df.shape, df.target.sum(), df["patient_id"].unique().shape)
    
    df['file_path'] = df['isic_id'].apply(get_train_file_path)
    df = df[ df["file_path"].isin(train_images) ].reset_index(drop=True)
    return df


def create_folds(df, n_fold):
    sgkf = StratifiedGroupKFold(n_splits=n_fold)
    
    for fold, ( _, val_) in enumerate(sgkf.split(df, df.target,df.patient_id)):
          df.loc[val_ , "kfold"] = int(fold)
    return df

def criterion(outputs, targets):
    return nn.BCELoss()(outputs, targets)


# classes

class ISICDataset_for_Train(Dataset):
    def __init__(self, df, transforms=None):
        self.df_positive = df[df["target"] == 1].reset_index()
        self.df_negative = df[df["target"] == 0].reset_index()
        self.file_names_positive = self.df_positive['file_path'].values
        self.file_names_negative = self.df_negative['file_path'].values
        self.targets_positive = self.df_positive['target'].values
        self.targets_negative = self.df_negative['target'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df_positive) * 2
    
    def __getitem__(self, index):
        if random.random() >= 0.5:
            df = self.df_positive
            file_names = self.file_names_positive
            targets = self.targets_positive
        else:
            df = self.df_negative
            file_names = self.file_names_negative
            targets = self.targets_negative
        index = index % df.shape[0]
        
        img_path = file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = targets[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'target': target
        }
    
class ISICDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.targets = df['target'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = self.targets[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'target': target
        }


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'


# class ISICModel(nn.Module):
#     def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None):
#         super(ISICModel, self).__init__()
#         self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)

#         # hardcoding here
        
#         # in_features = self.model.classifier.in_features
#         in_features=1024
#         # self.model.head = nn.Identity()
#         # self.model.global_pool = nn.Identity()
#         self.pooling = GeM()
#         self.linear = nn.Linear(in_features, num_classes)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, images):
#         tokens = self.model(images)  # shape [B, N, 192] typically

#         # 2) We want to convert [B, N, C] -> [B, C, H, W]
#         B, N, C = tokens.shape
#         H = W = int(N**0.5)
#         tokens = tokens.transpose(1, 2).view(B, C, H, W)
#         # features = self.model(images)
#         # print(features.shape)
#         pooled_features = self.pooling(tokens).flatten(1)
#         output = self.sigmoid(self.linear(pooled_features))
#         return output

    
# model = ISICModel(config.model_name, checkpoint_path=config.checkpoint_path)
# model.to(config.device);


# import torch
# import torch.nn as nn
# import timm

class ISICModel(nn.Module):
    def __init__(self, model_name="vit_tiny_r_s16_p8_384", num_classes=1, 
                 pretrained=True, checkpoint_path=None):
        super(ISICModel, self).__init__()
        # 1) Create the ViT model
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            checkpoint_path=checkpoint_path
        )
        
        # 2) Remove the original classification head so it just returns the final embedding
        #    This will give us a single vector of shape [batch_size, embed_dim].
        #    For vit_tiny_r_s16_p8_384, embed_dim is typically 192
        self.model.head = nn.Identity()
        
        # Check the actual embed_dim from the model
        # (Should be 192 for vit_tiny_r_s16_p8_384)
        embed_dim = getattr(self.model, 'embed_dim', 192)
        
        # 3) If you want a 1024-dim final feature vector:
        #    We'll add an extra Linear to map from 192 -> 1024
        self.feat_proj = nn.Linear(embed_dim, 1024)
        
        # 4) Classifier (192 -> 1) becomes (1024 -> 1) now
        self.linear = nn.Linear(1024, num_classes)
        
        # 5) Sigmoid for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        # Extract the final embedding (shape: [batch_size, 192] or [batch_size, embed_dim])
        features = self.model(images)  # [B, 192]
        
        # Optional: project from 192 -> 1024
        features_1024 = self.feat_proj(features)  # [B, 1024]
        
        # Classify
        output = self.sigmoid(self.linear(features_1024))  # [B, 1]
        return output

model = ISICModel(config.model_name, checkpoint_path=config.checkpoint_path)
model.to(config.device);


# data processing
set_seed(config.seed)

train_images = get_train_images(TRAIN_DIR)
df = get_data_df(ROOT_DIR)
config.set_t_max()
df = create_folds(df, config.n_fold)


# augmentations
data_transforms = {
    "train": A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Downscale(p=0.25),
        A.ShiftScaleRotate(shift_limit=0.1, 
                           scale_limit=0.15, 
                           rotate_limit=60, 
                           p=0.5),
        A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.),
    
    "valid": A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)
}


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    running_auroc  = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)
        loss = loss / config.n_accumulate
            
        loss.backward()
    
        if (step + 1) % config.n_accumulate == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        auroc = binary_auroc(input=outputs.squeeze(), target=targets).item()
        
        running_loss += (loss.item() * batch_size)
        running_auroc  += (auroc * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_Auroc=epoch_auroc,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    
    return epoch_loss, epoch_auroc


@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    running_auroc = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)

        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)

        auroc = binary_auroc(input=outputs.squeeze(), target=targets).item()
        running_loss += (loss.item() * batch_size)
        running_auroc  += (auroc * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Auroc=epoch_auroc,
                        LR=optimizer.param_groups[0]['lr'])   
    
    gc.collect()
    
    return epoch_loss, epoch_auroc


def run_training(model, optimizer, scheduler, device, num_epochs):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_auroc = -np.inf
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss, train_epoch_auroc = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=config.device, epoch=epoch)
        
        val_epoch_loss, val_epoch_auroc = valid_one_epoch(model, valid_loader, device=config.device, 
                                         epoch=epoch)
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Train AUROC'].append(train_epoch_auroc)
        history['Valid AUROC'].append(val_epoch_auroc)
        history['lr'].append( scheduler.get_lr()[0] )
        
        # deep copy the model
        if best_epoch_auroc <= val_epoch_auroc:
            print(f"{b_}Validation AUROC Improved ({best_epoch_auroc} ---> {val_epoch_auroc})")
            best_epoch_auroc = val_epoch_auroc
            best_model_wts = copy.deepcopy(model.state_dict())
            model_path = "AUROC{:.4f}_Loss{:.4f}_epoch{:.0f}.bin".format(val_epoch_auroc, val_epoch_loss, epoch)
            torch.save(model.state_dict(), model_path)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best AUROC: {:.4f}".format(best_epoch_auroc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history


def fetch_scheduler(optimizer):
    if config.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.T_max, 
                                                   eta_min=config.min_lr)
    elif config.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=config.T_0, 
                                                             eta_min=config.min_lr)
    elif config.scheduler == None:
        return None
        
    return scheduler


def prepare_loaders(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = ISICDataset_for_Train(df_train, transforms=data_transforms["train"])
    valid_dataset = ISICDataset(df_valid, transforms=data_transforms["valid"])

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, 
                              num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, 
                              num_workers=2, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader


train_loader, valid_loader = prepare_loaders(df, fold=config.fold)


optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                       weight_decay=config.weight_decay)
scheduler = fetch_scheduler(optimizer)


# model.model


model, history = run_training(model, optimizer, scheduler,
                              device=config.device,
                              num_epochs=config.epochs)


history = pd.DataFrame.from_dict(history)
history.to_csv("history.csv", index=False)


plt.plot( range(history.shape[0]), history["Train Loss"].values, label="Train Loss")
plt.plot( range(history.shape[0]), history["Valid Loss"].values, label="Valid Loss")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.show()


plt.plot( range(history.shape[0]), history["Train AUROC"].values, label="Train AUROC")
plt.plot( range(history.shape[0]), history["Valid AUROC"].values, label="Valid AUROC")
plt.xlabel("epochs")
plt.ylabel("AUROC")
plt.grid()
plt.legend()
plt.show()


plt.plot( range(history.shape[0]), history["lr"].values, label="lr")
plt.xlabel("epochs")
plt.ylabel("lr")
plt.grid()
plt.legend()
plt.show()


df[df["kfold"] == config.fold].target.sum()


for i in timm.list_models():
    print(i)

























