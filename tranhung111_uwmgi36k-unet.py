!pip install -q segmentation_models_pytorch
!pip install rasterio
# !pip install -q scikit-learn==1.0


%load_ext autoreload
%autoreload 2


import numpy as np
import pandas as pd
pd.options.plotting.backend = "plotly"
import random
from glob import glob
import os, shutil
from tqdm import tqdm
tqdm.pandas()
import time
import copy
import joblib
from collections import defaultdict
import gc
from IPython import display as ipd

# visualization
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Sklearn
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

# PyTorch 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

import timm

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

import rasterio
from joblib import Parallel, delayed

# For colored terminal text
from colorama import Fore, Back, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn



class CFG:
    seed          = 101
    debug         = False # set debug=False for Full Training
    exp_name      = '2.5D'
    comment       = 'unet-efficientnet_b0-160x192-ep=5'
    model_name    = 'Unet'
    backbone      = 'efficientnet-b1'
    train_bs      = 32
    valid_bs      = 32
    img_size      = [512, 512]
    epochs        = 50
    lr            = 2e-3
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
    T_max         = int(30000/train_bs*epochs)+50
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-6
    n_accumulate  = max(1, 32//train_bs)
    n_fold        = 5
    folds         = [0]
    num_classes   = 3
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_swa            = True
    swa_start_pct      = 0.8 
    swa_lr             = 1e-4
    swa_anneal_epochs  = 5 


path_df = pd.DataFrame(glob('/kaggle/input/uwmgi-25d-stride2-dataset/images/images/*'), columns=['image_path'])
path_df['mask_path'] = path_df.image_path.str.replace('image','mask')
path_df['id'] = path_df.image_path.map(lambda x: x.split('/')[-1].replace('.npy',''))
path_df.head()


df = pd.read_csv('/kaggle/input/uwmgi-dataset-folds5/train_folds.csv')
df.head()


df['empty'].value_counts().plot.bar()


def load_img(path):
    img = np.load(path)
    img = img.astype('float32') # original is uint16
    mx = np.max(img)
    if mx:
        img/=mx # scale image to [0, 1]
    return img

def load_msk(path):
    msk = np.load(path)
    msk = msk.astype('float32')
    msk/=255.0
    return msk
    

def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     img = clahe.apply(img)
#     plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='bone')
    
    if mask is not None:
        # plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles,labels)
    plt.axis('off')


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df         = df
        self.label      = label
        self.img_paths  = df['image_path'].tolist()
        self.msk_paths  = df['mask_path'].tolist()
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path  = self.img_paths[index]
        img = []
        img = load_img(img_path)
        
        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img  = data['image']
                msk  = data['mask']
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)


# data_transforms = {
#     "train": A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
#         A.OneOf([
#             A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
#             A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
#         ], p=0.25),
#         A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20,
#                          min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
#         ], p=1.0),
    
#     "valid": A.Compose([], p=1.0)
# }


data_transforms = {
    "train": A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=(0, 0.1), rotate_limit=20, border_mode=1, p=0.85),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.1, border_mode=1, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, border_mode=1, p=0.5)
        ], p=0.2),
        A.OneOf([
            A.GaussNoise(var_limit=(0.0001, 0.004), p=0.7),
            A.Blur(blur_limit=3, p=0.3)
        ], p=0.5),
        
        A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20,
                         min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0),
        
    "valid": A.Compose([], p=1.0)
}


fold = 0
train_df = df.query("fold!=@fold").reset_index(drop=True)
valid_df = df.query("fold==@fold").reset_index(drop=True)
valid_df


fold = 0
train_df = df.query("fold!=@fold").reset_index(drop=True)
train_df = train_df.head(32*5).query("empty==0")
train_df.shape


def prepare_loaders(fold, debug=False):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    if debug:
        train_df = train_df.head(32*5).query("empty==0")
        valid_df = valid_df.head(32*3).query("empty==0")
    train_dataset = BuildDataset(train_df, transforms=data_transforms['train'])
    valid_dataset = BuildDataset(valid_df, transforms=data_transforms['valid'])

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs if not debug else 20, 
                              num_workers=4, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs if not debug else 20, 
                              num_workers=4, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader



train_loader, valid_loader = prepare_loaders(fold=0, debug=True)


imgs, msks = next(iter(train_loader))
imgs.size(), msks.size()


def plot_batch(imgs, msks, size=3):
    plt.figure(figsize=(5*5, 5))
    for idx in range(size):
        plt.subplot(1, 5, idx+1)
        img = imgs[idx,].permute((1, 2, 0)).numpy()*255.0
        img = img.astype('uint8')
        msk = msks[idx,].permute((1, 2, 0)).numpy()*255.0
        show_img(img, msk)
    plt.tight_layout()
    plt.show()


plot_batch(imgs, msks, size=5)


import gc
gc.collect()


# import segmentation_models_pytorch as smp

# def build_model():
#     model = smp.UnetPlusPlus(
#         encoder_name='timm-efficientnet-b1',      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#         encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#         in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#         classes=CFG.num_classes,        # model output channels (number of classes in your dataset)
#         activation=None,
#     )
#     model.to(CFG.device)
#     return model

# def load_model(path):
#     model = build_model()
#     model.load_state_dict(torch.load(path))
#     model.eval()
#     return model


# import segmentation_models_pytorch as smp

# def build_model():
#     model = smp.DPT(
#         "tu-mobilevitv2_175.cvnets_in1k",
#         classes=CFG.num_classes,        # model output channels (number of classes in your dataset)
#         activation=None,
#     )
#     model.to(CFG.device)
#     return model

# def load_model(path):
#     model = build_model()
#     model.load_state_dict(torch.load(path))
#     model.eval()
#     return model


import segmentation_models_pytorch as smp

def build_model():
    model = smp.Segformer(
        encoder_name='efficientnet-b2',      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=CFG.num_classes,        # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to(CFG.device)
    return model

def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


import torch.nn.functional as F

JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss    = smp.losses.DiceLoss(mode='multilabel')
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)

def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

# def criterion(y_pred, y_true):
#     return 0.5*BCELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)

def criterion(y_pred, y_true):
    return 0.5*BCELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice


    #PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, weight=0.5):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs_sigmoid = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        inputs_sigmoid = inputs_sigmoid.view(-1)
        targets = targets.view(-1)

        intersection = (inputs_sigmoid * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_sigmoid.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')

        Dice_BCE = weight*BCE + (1-weight)*dice_loss

        return Dice_BCE

class LogDiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LogDiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, weight=0.5):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs_sigmoid = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        inputs_sigmoid = inputs_sigmoid.view(-1)
        targets = targets.view(-1)

        intersection = (inputs_sigmoid * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_sigmoid.sum() + targets.sum() + smooth)
        log_cosh_dice_loss = torch.log((torch.exp(dice_loss) + torch.exp(-dice_loss)) / 2)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')

        Dice_BCE = weight*BCE + (1-weight)*log_cosh_dice_loss

        return Dice_BCE

loss_fn = LogDiceBCELoss()


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (images, masks) in pbar:         
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss   = criterion(y_pred, masks)
            loss   = loss / CFG.n_accumulate
            
        scaler.scale(loss).backward()
    
        if (step + 1) % CFG.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_mem=f'{mem:0.2f} GB')
        torch.cuda.empty_cache()
        gc.collect()
    
    return epoch_loss


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    dataset_size = 0
    train_scores = []  # collect [dice, jaccard] per batch

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Train {epoch}')
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)

        optimizer.zero_grad()
        y_pred = model(images)
        loss   = criterion(y_pred, masks)
        loss.backward()
        optimizer.step()

        # accumulate loss
        batch_size    = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss    = running_loss / dataset_size

        # compute metrics
        y_prob        = torch.sigmoid(y_pred)
        train_dice    = dice_coef(masks, y_prob).cpu().detach().numpy()
        train_jaccard = iou_coef(masks, y_prob).cpu().detach().numpy()
        train_scores.append([train_dice, train_jaccard])

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:.4f}', lr=f'{current_lr:.5f}')

    # mean metrics over all batches
    train_scores_epoch = np.mean(train_scores, axis=0)  # [dice, jaccard]
    return epoch_loss, train_scores_epoch


# -------------------------------------------------------------------
# 2) VALIDATION EPOCH -----------------------------------------------
# -------------------------------------------------------------------
@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    val_scores   = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Valid {epoch}')
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)

        y_pred = model(images)
        loss   = criterion(y_pred, masks)

        batch_size    = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss    = running_loss / dataset_size

        y_prob        = torch.sigmoid(y_pred)
        val_dice      = dice_coef(masks, y_prob).cpu().detach().numpy()
        val_jaccard   = iou_coef(masks, y_prob).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])

        current_lr = optimizer.param_groups[0]['lr']
        mem        = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        pbar.set_postfix(valid_loss=f'{epoch_loss:.4f}',
                         lr=f'{current_lr:.5f}',
                         gpu_memory=f'{mem:.2f} GB')

    val_scores_epoch = np.mean(val_scores, axis=0)  # [dice, jaccard]
    torch.cuda.empty_cache()
    gc.collect()
    return epoch_loss, val_scores_epoch


import pickle
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

def run_training(model,
                 optimizer,
                 scheduler,
                 device,
                 num_epochs,
                 patience=5):
    """
    Runs training + validation with early stopping.
    :param patience: how many epochs to wait for improvement before stopping
    """
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name())

    start = time.time()
    best_model_wts   = copy.deepcopy(model.state_dict())
    best_dice        = -np.inf
    no_improve_epochs = 0
    history          = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')

        # ---- train ----
        train_loss, (train_dice, train_jaccard) = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )

        # ---- validate ----
        val_loss, (val_dice, val_jaccard) = valid_one_epoch(
            model, valid_loader, device, epoch
        )

        # ---- scheduler handling ----
        if CFG.use_swa and epoch > swa_start_epoch:
            # *SWA* phase: keep LR cycling via SWALR
            swa_model.update_parameters(model)   # <-- average weights
            swa_scheduler.step()
        else:
            # *Normal* phase: whatever scheduler you chose
            if scheduler is not None:
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)     # needs a metric
                else:
                    scheduler.step()

        if CFG.use_swa and epoch > swa_start_epoch:
            swa_model.to(CFG.device)

            print("\nğŸ”„  Updating BatchNorm statistics for SWA model â€¦")
            update_bn(train_loader, swa_model, device=CFG.device)   # <â”€â”€ key line
            swa_scheduler.step()                 # NEW â€” cosine LR inside SWA
            model = swa_model
            

        # ---- log to history ----
        history['Train Loss'].append(train_loss)
        history['Train Dice'].append(train_dice)
        history['Train Jaccard'].append(train_jaccard)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)

        print(f"  Train Dice: {train_dice:.4f} | Train Jaccard: {train_jaccard:.4f}")
        print(f"  Valid Dice: {val_dice:.4f} | Valid Jaccard: {val_jaccard:.4f}")

        # ---- check for improvement ----
        if val_dice > best_dice:
            print(f"  ğŸ�† Dice improved ({best_dice:.4f} â†’ {val_dice:.4f}), saving model.")
            best_dice        = val_dice
            best_model_wts   = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            torch.save(model.state_dict(), f"best_epoch-{fold:02d}.bin")
        else:
            no_improve_epochs += 1
            print(f"  âš ï¸�  No improvement for {no_improve_epochs}/{patience} epochs.")
            if no_improve_epochs >= patience:
                print(f"ğŸš¨ Early stopping triggered after {epoch} epochs.")
                break

        # ---- always save last ----
        torch.save(model.state_dict(), f"last_epoch-{fold:02d}.bin")

        # ---- scheduler step if used ----
        if scheduler is not None:
            scheduler.step()

    # ---- wrap up ----
    model.load_state_dict(best_model_wts)
    elapsed = time.time() - start
    h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
    print(f"\nTraining complete in {h}h {m}m")
    print(f"Best Validation Dice: {best_dice:.4f}")

    # record total training time
    history['Total Training Time (s)'].append(elapsed)

    # Save history
    with open("history.pkl", "wb") as f:
        import pickle
        pickle.dump(history, f)

    return model, history



def fetch_scheduler(optimizer):
    if CFG.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG.T_max, 
                                                   eta_min=CFG.min_lr)
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CFG.T_0, 
                                                             eta_min=CFG.min_lr)
    elif CFG.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=CFG.min_lr,)
    elif CFG.scheduer == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif CFG.scheduler == None:
        return None
        
    return scheduler


model = build_model()
optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
scheduler = fetch_scheduler(optimizer)

if CFG.use_swa:
    swa_start_epoch = int(CFG.epochs * CFG.swa_start_pct)
    swa_model      = AveragedModel(model)                     # running weight avg
    swa_scheduler  = SWALR(optimizer,
                           swa_lr=CFG.swa_lr,
                           anneal_epochs=CFG.swa_anneal_epochs,
                           anneal_strategy='cos')


for fold in CFG.folds: ## 0,1
    print(f'#'*15)
    print(f'### Fold: {fold}')
    print(f'#'*15)

    train_loader, valid_loader = prepare_loaders(fold=fold, debug=CFG.debug)
    model     = build_model()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    scheduler = fetch_scheduler(optimizer)
    model, history = run_training(model, optimizer, scheduler,
                                  device=CFG.device,
                                  num_epochs=CFG.epochs)



test_dataset = BuildDataset(df.query("fold==0 & empty==0").sample(frac=1.0), label=False, 
                            transforms=data_transforms['valid'])
test_loader  = DataLoader(test_dataset, batch_size=5, 
                          num_workers=4, shuffle=False, pin_memory=True)
imgs = next(iter(test_loader))
imgs = imgs.to(CFG.device, dtype=torch.float)

preds = []
for fold in CFG.folds:
    model = load_model(f"best_epoch-{fold:02d}.bin")
    with torch.no_grad():
        pred = model(imgs)
        pred = (nn.Sigmoid()(pred)>0.5).double()
    preds.append(pred)
    
imgs  = imgs.cpu().detach()
preds = torch.mean(torch.stack(preds, dim=0), dim=0).cpu().detach()


plot_batch(imgs, preds, size=5)







