%%capture
pip install segmentation-models-pytorch


import torch
import numpy
import random
import os
import torch.backends.cudnn as cudnn
from PIL import Image
from IPython.display import display
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import numpy as np
from torchvision import transforms
from random import random, uniform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import disk
from scipy.ndimage import binary_closing, binary_fill_holes
import warnings
from glob import glob


warnings.filterwarnings("ignore")
def set_all_seeds(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None, images=[], augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = sorted(images)
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
       
        mask_name = self.images[idx].replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
       
        if self.augment:
            angle = uniform(-15, 15)
            image = image.rotate(angle, resample=Image.BICUBIC, expand=False)
            mask = mask.rotate(angle, resample=Image.NEAREST, expand=False)
           
            if random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
           
            if random() < 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
       
        if self.target_transform:
            mask = self.target_transform(mask)
       
        mask = np.array(mask) / 5
        mask[mask != 0] -= 4
       
        mask = torch.from_numpy(mask).long()
       
        if self.transform:
            image = self.transform(image)
       
        return image, mask


class UNet(nn.Module):
    def __init__(self, num_classes=40, encoder_name='efficientnet-b5', encoder_weights='imagenet'):
        super(UNet, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name, 
            encoder_weights=encoder_weights,  
            in_channels=3, 
            classes=num_classes,  
            activation=None  
        )
    
    def forward(self, x):
        return self.model(x)

def compute_iou(outputs, targets, num_classes=2, ignore_index=0):
    preds = torch.argmax(outputs, dim=1)
    Intersection = ((targets==preds )& (targets!=0)).sum()
    Union = ((targets!=0 )|  (preds!=0)).sum()
    IoU = Intersection/Union
    return float(IoU)


import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import KFold


img_size = (456,456)
train_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
])
train_target_transform = transforms.Compose([
    transforms.Resize(img_size, interpolation=Image.NEAREST),
])
val_target_transform = transforms.Compose([
    transforms.Resize(img_size, interpolation=Image.NEAREST),
])
image_dir = '/kaggle/input/nuu-data-i-cho/data/input'
mask_dir = '/kaggle/input/nuu-data-i-cho/data/target'
all_images = sorted(os.listdir(image_dir))

n_splits = 5  
fold_num = 0  
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

train_indices = None
val_indices = None
for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
    if fold == fold_num:
        train_indices = train_idx
        val_indices = val_idx
        break

if train_indices is None:
    raise ValueError(f"Invalid fold_num: {fold_num}. Must be between 0 and {n_splits-1}.")

train_images = [all_images[i] for i in train_indices]
val_images = [all_images[i] for i in val_indices]

train_dataset = SegmentationDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    transform=train_transform,
    target_transform=train_target_transform,
    images=train_images,
    augment=True
)
val_dataset = SegmentationDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    transform=val_transform,
    target_transform=val_target_transform,
    images=val_images,
    augment=False
)
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=4)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import os
os.mkdir(f'effnet_b5_{fold_num}')

num_epochs = 15
lr = 3e-4
device = 'cuda'
model = UNet(num_classes=40)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
num_classes = 40
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

model.train()
top_list = []
for epoch in range(num_epochs):
    running_loss = []
    pbar_train = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Train)')
  
    for images, masks in pbar_train:
        images = images.to(device)
        masks = masks.to(device).squeeze(1)
      
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
      
        running_loss.append(loss.item())
        pbar_train.set_postfix({'loss': np.mean(running_loss)})
   
    scheduler.step()
  
    model.eval()
    val_loss = []
    all_iou = 0.0
    num_batches = 0
  
    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Val)')
        for images, masks in pbar_val:
            images = images.to(device)
            masks = masks.to(device).squeeze(1)
          
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss.append(loss.item())
          
            batch_iou = compute_iou(outputs, masks, num_classes=num_classes)
            all_iou += batch_iou
            num_batches += 1
          
            pbar_val.set_postfix({'val_loss': np.mean(val_loss), 'iou': batch_iou})
  
    avg_val_loss = np.mean(val_loss)
    avg_iou = all_iou / num_batches
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch+1} val average loss: {avg_val_loss:.4f}, mean IoU: {avg_iou:.4f}, current LR: {current_lr:.2e}')
  
    current_path = f'effnet_b5_{fold_num}/checkpoint_epoch_{epoch+1}.pth'
    torch.save(model.state_dict(), current_path)
    top_list.append((avg_iou, current_path))
    top_list.sort(key=lambda x: x[0], reverse=True)
    top_list = top_list[:3]
    kept_paths = [path for _, path in top_list]
    for e in range(1, epoch + 2):
        old_path = f'effnet_b5_{fold_num}/checkpoint_epoch_{e}.pth'
        if old_path not in kept_paths and os.path.exists(old_path):
            os.remove(old_path)
  
    model.train()


import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import KFold


img_size = (512,512)
train_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
])
train_target_transform = transforms.Compose([
    transforms.Resize(img_size, interpolation=Image.NEAREST),
])
val_target_transform = transforms.Compose([
    transforms.Resize(img_size, interpolation=Image.NEAREST),
])
image_dir = '/kaggle/input/nuu-data-i-cho/data/input'
mask_dir = '/kaggle/input/nuu-data-i-cho/data/target'
all_images = sorted(os.listdir(image_dir))

n_splits = 5  
fold_num = 0  
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

train_indices = None
val_indices = None
for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
    if fold == fold_num:
        train_indices = train_idx
        val_indices = val_idx
        break

if train_indices is None:
    raise ValueError(f"Invalid fold_num: {fold_num}. Must be between 0 and {n_splits-1}.")

train_images = [all_images[i] for i in train_indices]
val_images = [all_images[i] for i in val_indices]

train_dataset = SegmentationDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    transform=train_transform,
    target_transform=train_target_transform,
    images=train_images,
    augment=True
)
val_dataset = SegmentationDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    transform=val_transform,
    target_transform=val_target_transform,
    images=val_images,
    augment=False
)
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=4)


class UNet(nn.Module):
    def __init__(self, num_classes=40, encoder_name='tu-tf_efficientnetv2_m.in21k_ft_in1k', encoder_weights='imagenet'):
        super(UNet, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name, 
            encoder_weights=encoder_weights,  
            in_channels=3, 
            classes=num_classes,  
            activation=None  
        )
    
    def forward(self, x):
        return self.model(x)

def compute_iou(outputs, targets, num_classes=2, ignore_index=0):
    preds = torch.argmax(outputs, dim=1)
    Intersection = ((targets==preds )& (targets!=0)).sum()
    Union = ((targets!=0 )|  (preds!=0)).sum()
    IoU = Intersection/Union
    return float(IoU)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import os
os.mkdir(f'effnet_v2m_{fold_num}')

num_epochs = 15
lr = 3e-4
device = 'cuda'
model = UNet(num_classes=40)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
num_classes = 40
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

model.train()
top_list = []
for epoch in range(num_epochs):
    running_loss = []
    pbar_train = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Train)')
  
    for images, masks in pbar_train:
        images = images.to(device)
        masks = masks.to(device).squeeze(1)
      
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
      
        running_loss.append(loss.item())
        pbar_train.set_postfix({'loss': np.mean(running_loss)})
   
    scheduler.step()
  
    model.eval()
    val_loss = []
    all_iou = 0.0
    num_batches = 0
  
    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Val)')
        for images, masks in pbar_val:
            images = images.to(device)
            masks = masks.to(device).squeeze(1)
          
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss.append(loss.item())
          
            batch_iou = compute_iou(outputs, masks, num_classes=num_classes)
            all_iou += batch_iou
            num_batches += 1
          
            pbar_val.set_postfix({'val_loss': np.mean(val_loss), 'iou': batch_iou})
  
    avg_val_loss = np.mean(val_loss)
    avg_iou = all_iou / num_batches
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch+1} val average loss: {avg_val_loss:.4f}, mean IoU: {avg_iou:.4f}, current LR: {current_lr:.2e}')
  
    current_path = f'effnet_v2m_{fold_num}/checkpoint_epoch_{epoch+1}.pth'
    torch.save(model.state_dict(), current_path)
    top_list.append((avg_iou, current_path))
    top_list.sort(key=lambda x: x[0], reverse=True)
    top_list = top_list[:3]
    kept_paths = [path for _, path in top_list]
    for e in range(1, epoch + 2):
        old_path = f'effnet_v2m_{fold_num}/checkpoint_epoch_{e}.pth'
        if old_path not in kept_paths and os.path.exists(old_path):
            os.remove(old_path)
  
    model.train()

