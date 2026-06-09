!pip install segmentation_models_pytorch

import os
from tqdm.notebook import tqdm
import gc
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import math
import timm
import pandas as pl
import torch
import numpy as np
from torch.amp import GradScaler
import cv2
import random
from tqdm.notebook import tqdm
from torch.autograd import Variable
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import segmentation_models_pytorch as smp





def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(666)


train_msk = np.load('/kaggle/input/fdgdgfh/msk_array.npy')
train_images = sorted(os.listdir('/kaggle/input/fdgdgfh/data/train'))
test_images = sorted(os.listdir('/kaggle/input/fdgdgfh/data/test'))
test_msk = np.zeros((len(test_images), train_msk.shape[1], train_msk.shape[2]))

train_images = [f'/kaggle/input/fdgdgfh/data/train/{path}' for path in train_images]
test_images = [f'/kaggle/input/fdgdgfh/data/test/{path}' for path in test_images]
len(train_images)


import cv2
import numpy as np
import torch
import albumentations as A

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        image_paths, 
        masks=None, 
        augment=False, 
        image_size=(640, 640), 
        shift=(-25, -25), 
        final_size=(640, 640), 
        test=False
    ):
        self.image_paths = image_paths
        self.masks = masks
        self.augment = augment
        self.image_size = image_size
        self.shift = shift
        self.final_size = final_size
        self.test = test
        if self.augment and not self.test:
            self.combined_augmentation = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            
                A.Perspective(
                    scale=(0.05, 0.1), 
                    p=0.2
                ),
                A.Flip(p=0.5),  

                A.RandomBrightnessContrast(
                    brightness_limit=0.13, 
                    contrast_limit=0.13, 
                    p=0.45
                ),
                A.OneOf([
                    A.OpticalDistortion(
                        distort_limit=1, 
                        shift_limit=0.5, 
                        p=0.8
                    ),
                    A.Affine(
                        scale=(0.8, 1.2),              
                        translate_percent=(0.1, 0.1),    
                        rotate=(-45, 45),              
                        shear=(-15, 15),                 
                        p=0.2
                    ),
                    A.GridDistortion(p=0.5),
                ], p=0.8),
                
                A.CLAHE(p=0.8),

                A.RandomGamma(p=0.45),
                A.HueSaturationValue(
                    hue_shift_limit=15, 
                    sat_shift_limit=20, 
                    val_shift_limit=10, 
                    p=0.5
                ),
                A.RandomFog(
                    fog_coef_lower=0.1, 
                    fog_coef_upper=0.1, 
                    alpha_coef=0.1, 
                    p=0.5
                ),
                
                
                A.RandomSunFlare(
                    p=0.6
                ),
                
                A.OneOf([
                    A.Blur(blur_limit=3, p=0.3),
                    A.MedianBlur(blur_limit=3, p=0.2),
                ], p=0.5),
            
            
                A.GaussNoise(
                    var_limit=(10.0, 50.0), 
                    p=0.4
                ),
            ], additional_targets={'mask': 'mask'})
            
        else:
            self.combined_augmentation = None

        self.normalize = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def shift_mask(self, mask, dx, dy):
        h, w = mask.shape[:2]
        M = np.float32([
            [1, 0, dx],
            [0, 1, dy]
        ])
        shifted_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        return shifted_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)

        if self.test:
            img = cv2.resize(img, self.final_size, interpolation=cv2.INTER_LINEAR)
            img = self.normalize(image=img)['image']
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            return img_tensor

        mask = self.masks[idx]
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        dx, dy = self.shift
        if dx != 0 or dy != 0:
            mask = self.shift_mask(mask, dx, dy)
            h, w = self.image_size
            x_min = max(0, dx)
            x_max = w - max(0, -dx)
            y_min = max(0, dy)
            y_max = h - max(0, -dy)

            crop_transform = A.Compose([
                A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
            ], additional_targets={'mask': 'mask'})

            cropped = crop_transform(image=img, mask=mask)
            img, mask = cropped['image'], cropped['mask']

        if self.combined_augmentation is not None:
            augmented = self.combined_augmentation(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        img = cv2.resize(img, self.final_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.final_size, interpolation=cv2.INTER_NEAREST)

        img = self.normalize(image=img)['image']

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).long()

        return img_tensor, mask_tensor



class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.unet = smp.Unet('resnet18',
                             encoder_weights='imagenet',
                             classes=1,
                             decoder_channels=[256, 128, 64, 32, 16],
        )
    def forward(self, x):
        y = self.unet(x)
        return y


import gc
import torch
import segmentation_models_pytorch as smp
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


gc.collect()
torch.cuda.empty_cache()


batch_size = 4
valid_batch_size = 4
epochs = 12
lr = 3.55e-4
clip_grad_norm = 15.28
DEVICE = 'cuda'

params_train = {
    'batch_size': batch_size,
    'shuffle': True,
    'drop_last': True,
    'num_workers': 2
}
params_val = {
    'batch_size': valid_batch_size,
    'shuffle': False,
    'drop_last': False,
    'num_workers': 2
}


train_loader = torch.utils.data.DataLoader(
    Dataset(train_images[:-50], train_msk[:-50]), 
    **params_train
)
val_loader = torch.utils.data.DataLoader(
    Dataset(train_images[-50:], train_msk[-50:]), 
    **params_val
)


model = Model().to(DEVICE)
loss_func = smp.losses.DiceLoss(mode="binary", smooth=1.)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=len(train_loader) * epochs, eta_min=1e-6
)


scaler = GradScaler()

for epoch in range(epochs):

    model.train()
    running_loss = 0.0
    tk0 = tqdm(enumerate(train_loader), total=len(train_loader))
    
    for batch_number, (img, target) in tk0:
        img = img.to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad()
        with autocast():
            outputs = model(img)
            loss = loss_func(outputs, target)

        # backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.detach().cpu().item()
        tk0.set_postfix(
            loss=running_loss / (batch_number + 1),
            lr=scheduler.get_last_lr()[0],
            stage="train",
            epoch=epoch
        )

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        tk1 = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
        for val_batch_number, (val_img, val_target) in tk1:
            val_img = val_img.to(DEVICE)
            val_target = val_target.to(DEVICE)

            outputs = model(val_img)
            loss = loss_func(outputs, val_target)
            val_loss += loss.detach().cpu().item()

            tk1.set_postfix(
                val_loss=val_loss / (val_batch_number + 1),
                stage="val",
                epoch=epoch
            )

    val_loss /= len(val_loader)


    print(f"Epoch [{epoch}/{epochs}] | "
          f"Train Loss: {running_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss:.4f}")



params_val = {'batch_size': batch_size, 'shuffle': False, 'drop_last': False, 'num_workers': 2}
test_loader = torch.utils.data.DataLoader(Dataset(test_images, test_msk,test=True), **params_val)


preds = []
imgs_list = []
target_list = []
model.eval()
average_loss = 0
with torch.no_grad():
    for batch_number,  (img)  in enumerate(test_loader):
        img = img.to(DEVICE)
        target = target.to(DEVICE)

        with torch.amp.autocast('cuda'):
            outputs = model(img)

        preds += [outputs.sigmoid().to('cpu').numpy()]

preds = np.concatenate(preds)[:, 0, ...]


preds = (preds > 0.5).astype(np.uint8)


def rle_encode(x, fg_val=1):
    """
    Args:
        x:  numpy array of shape (height, width), 1 - mask, 0 - background
    Returns: run length encoding as list
    """

    dots = np.where(
        x.T.flatten() == fg_val)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def list_to_string(x):
    """
    Converts list to a string representation
    Empty list returns '-'
    """
    if x: # non-empty list
        s = str(x).replace("[", "").replace("]", "").replace(",", "")
    else:
        s = '-'
    return s


true_list = [list_to_string(rle_encode(ans)) for ans in preds]

predict_df = pd.DataFrame()
predict_df['Id'] = [f'{x:03d}.jpg' for x in range(150)]
predict_df['Target'] = true_list
predict_df.to_csv('oh_no3.csv', index = None)


from PIL import Image


img = Image.open('/kaggle/input/fdgdgfh/2025-03-07 145713.png')


img




