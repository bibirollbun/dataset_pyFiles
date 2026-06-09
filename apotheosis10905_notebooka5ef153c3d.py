import os
import cv2
import glob
import torch
import random
import imagehash
import numpy as np
import pandas as pd
import torch.nn as nn
import albumentations as A
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.amp import GradScaler, autocast
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

#Set seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

#Data import & augmentation
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0) 
        mask[mask > 0] = 1.0 

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1) 
            mask = mask.astype(np.float32)
            mask = torch.from_numpy(mask)
        return image, mask
        
train_transform = A.Compose(
    [
        A.Affine(scale=(0.8, 1.2), translate_percent=(-0.0625, 0.0625), rotate=(-15, 15)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, p=0.5),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.03), p=0.5),
            A.RandomBrightnessContrast(p=0.5),
             A.CLAHE(clip_limit=4.0),
        ], p=0.5),
        ToTensorV2(),
    ],
)

class TiffTestDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.file_paths = glob.glob(os.path.join(root_dir, "*.tif"))
        self.transforms = transforms

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image_id = int(os.path.splitext(os.path.basename(file_path))[0])
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        else:
            image = image.transpose(2, 0, 1).astype('float32') / 255.0
            image = torch.from_numpy(image)
        return image,image_id

test_transform = A.Compose([
    A.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])


#Data cleaning
def analyze_and_clean_dataset(root_dir, threshold=5):

    mask_paths = glob.glob(os.path.join(root_dir, "*_mask.tif"))
    image_paths = [p.replace("_mask.tif", ".tif") for p in mask_paths]
    hashes = []

    for img_path, mask_path in zip(image_paths, mask_paths):
        if not os.path.exists(img_path): continue
        
        try:
            img = Image.open(img_path)
            h = imagehash.phash(img)
            hashes.append((h, img_path, mask_path))
        except Exception as e:
            print(f"{img_path}: {e}")

    used = set()
    final_imgs, final_masks = [], []

    for i in range(len(hashes)):
        if i in used:
            continue

        h1, img1, msk1 = hashes[i]
        group = [(img1, msk1)]
        used.add(i)

        for j in range(i+1, len(hashes)):
            if j in used:
                continue
            h2, img2, msk2 = hashes[j]
            if h1 - h2 <= threshold:
                group.append((img2, msk2))
                used.add(j)

        if len(group) == 1:
            final_imgs.append(img1)
            final_masks.append(msk1)
        else:
            masks = [cv2.imread(m, 0) for _, m in group]
            avg = np.mean(masks, axis=0)
            merged = (avg > 0.5).astype(np.uint8)

            save_mask = group[0][1]
            cv2.imwrite(save_mask, merged * 255)

            final_imgs.append(group[0][0])
            final_masks.append(save_mask)

    print("-" * 30)
    print(f"Số lượng ảnh gốc: {len(hashes)}")
    print(f"Số lượng ảnh sạch: {len(final_imgs)}")
    print(f"Đã loại bỏ: {len(hashes) - len(final_imgs)}")
    print("-" * 30)
    
    return final_imgs, final_masks



#U-net and Dice loss define
class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channel, affine=True),nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=False),
        )

    def forward(self, x):
        return self.layers(x)

class CBAM(nn.Module):
    """Convolution Block Attention Module"""
    def __init__(self, in_channel,ratio=16):
        super(CBAM, self).__init__()
        self.in_channel = in_channel
        self.MLP = nn.Sequential(
            nn.Linear(in_channel, in_channel // ratio, bias=False),
            nn.LeakyReLU(0.1, inplace=True), # Thay ReLU
            nn.Linear(in_channel // ratio, in_channel, bias=False)
        )
        self.Conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
    def forward(self, x):
        #channel attention
        pre_mlp1 = F.adaptive_avg_pool2d(x, (1,1))
        pre_mlp2 = F.adaptive_max_pool2d(x, (1,1))

        pre_mlp1 = pre_mlp1.view(pre_mlp1.size(0), -1)
        pre_mlp2 = pre_mlp2.view(pre_mlp2.size(0), -1)

        out_channel_attention = F.sigmoid(self.MLP(pre_mlp1) + self.MLP(pre_mlp2))

        x = x * out_channel_attention.view(x.size(0), x.size(1), 1, 1)

        #spatial attention
        pre_conv1,_ = torch.max(x, dim=1, keepdim=True)
        pre_conv2 = torch.mean(x, dim=1, keepdim=True)
        pre_conv = torch.cat([pre_conv1,pre_conv2],1)
        out_spatial_attention = F.sigmoid(self.Conv(pre_conv))

        x = x * out_spatial_attention
        return x

class GABM(nn.Module):
    """
        Global Attention Block Module
        Replace normal skip connection
    """
    def __init__(self, in_channel1,in_channel2, out_channel):
        super(GABM, self).__init__()
        self.conv1 = nn.Conv2d(in_channel1, out_channel, kernel_size=1)
        self.norm1 = nn.InstanceNorm2d(out_channel, affine=True)
        
        self.conv2 = nn.Conv2d(in_channel2, out_channel, kernel_size=1)
        self.norm2 = nn.InstanceNorm2d(out_channel, affine=True)
        
        self.conv3 = nn.Conv2d(out_channel, 1, kernel_size=1)
        self.norm3 = nn.InstanceNorm2d(1, affine=True) 

        self.relu = nn.LeakyReLU(0.1, inplace=False)
    def forward(self, xl, xg):
        x1 = self.norm1(self.conv1(xl))
        if xg.size()[2:] != xl.size()[2:]:
            xg = F.interpolate(xg, size=xl.size()[2:], mode='bilinear', align_corners=True)
        x2 = self.norm2(self.conv2(xg))
        prex3 = self.relu(x1 + x2)
        x3 = torch.sigmoid(self.norm3(self.conv3(prex3)))
        out = xl * x3
        return out

class RGCM(nn.Module):
    """
        Residual Group Convolution Module
    """
    def __init__(self, in_channel, out_channel,groups=8, reduction=2):
        super(RGCM, self).__init__()
        self.diff_channel = False
        mid_channel = in_channel // reduction
        if mid_channel % groups != 0:
            mid_channel = ((mid_channel // groups) + 1) * groups

        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(mid_channel, affine=True)

        self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3,
                               stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.InstanceNorm2d(mid_channel, affine=True)

        self.conv3 = nn.Conv2d(mid_channel, out_channel, kernel_size=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(out_channel, affine=True)
        
        self.shortcut = nn.Sequential()
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_channel, affine=True)
            )
        self.relu = nn.LeakyReLU(0.1, inplace=False)
    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        # 1. Conv 1x1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=False),
        )
        # 2. Dilated Convs (rate 6, 12, 18)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=False),
        )
        # 3. Global Avg Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=False),
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False), # Concatenate 4 nhánh
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=False),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        x4 = self.avg_pool(x)
        x4 = self.conv4(x4)
        x4 = F.interpolate(x4, size=x.size()[2:], mode='bilinear', align_corners=True)

        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        return self.final_conv(x_cat)

class ARGA_U_net(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(ARGA_U_net, self).__init__()
        self.conv_in = Conv_Block(in_channel, 64)
        self.RCGM1 = RGCM(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RCGM2 = RGCM(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RCGM3 = RGCM(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RCGM4 = RGCM(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RCGM5 = RGCM(512,1024)
        self.ASPP = ASPP(1024,1024)
        self.cls_pool_avg = nn.AdaptiveAvgPool2d(1)
        self.cls_pool_max = nn.AdaptiveMaxPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.GABM6 = GABM(512,1024,1024)
        self.dec_conv6 = nn.Sequential(
            nn.Conv2d(1024 + 512, 512, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, inplace=False),
            RGCM(512, 512),
            CBAM(512)
        )
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.GABM7 = GABM(256,512,512)
        self.dec_conv7 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, inplace=False),
            RGCM(256, 256),
            CBAM(256)
        )

        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.GABM8 = GABM(128,256,256)
        self.dec_conv8 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, inplace=False),
            RGCM(128, 128),
            CBAM(128)
        )

        self.up8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.GABM9 = GABM(64,128,128)
        self.dec_conv9 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.1, inplace=False),
            RGCM(64, 64),
            CBAM(64)
        )
        self.aux_layer = nn.Conv2d(256, num_classes, kernel_size=1)
        self.last_layer = nn.Conv2d(64, num_classes, kernel_size=1)
    def forward(self, x):
        #encoder
        x = self.conv_in(x)
        x1 = self.RCGM1(x)
        x2 = self.RCGM2(self.pool1(x1))
        x3 = self.RCGM3(self.pool2(x2))
        x4 = self.RCGM4(self.pool3(x3))
        #bottleneck
        x5 = self.ASPP(self.RCGM5(self.pool4(x4)))
        #classifier
        x5_avg = self.cls_pool_avg(x5)
        x5_max = self.cls_pool_max(x5)
        x5_cat = torch.cat([x5_avg, x5_max], dim=1)
        
        cls_out = self.cls_head(x5_cat)
        #decoder
        x5_up = self.up5(x5)
        x4g = self.GABM6(xl=x4, xg=x5)
        if x5_up.size() != x4g.size():
             x5_up = F.interpolate(x5_up, size=x4g.shape[2:], mode='bilinear', align_corners=True)

        x6 = self.dec_conv6(torch.cat([x5_up, x4g], dim=1))

        x6_up = self.up6(x6)
        x3g = self.GABM7(xl=x3, xg=x6)
        if x6_up.size() != x3g.size():
            x6_up = F.interpolate(x6_up, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x7 = self.dec_conv7(torch.cat([x6_up, x3g], dim=1))
        #deep supervision
        aux_out = self.aux_layer(x7)
        if aux_out.size()[2:] != x.size()[2:]:
            aux_out = F.interpolate(aux_out, size=x.size()[2:], mode='bilinear', align_corners=True)

        x7_up = self.up7(x7)
        x2g = self.GABM8(xl=x2, xg=x7)
        if x7.size() != x2g.size():
            x7_up = F.interpolate(x7_up, size=x2g.shape[2:], mode='bilinear', align_corners=True)
        x8 = self.dec_conv8(torch.cat([x7_up, x2g], dim=1))

        x8_up = self.up8(x8)
        x1g = self.GABM9(xl=x1, xg=x8)
        if x8.size() != x1g.size():
            x8_up = F.interpolate(x8_up, size=x1g.shape[2:], mode='bilinear', align_corners=True)
        x9 = self.dec_conv9(torch.cat([x8_up, x1g], dim=1))

        out = self.last_layer(x9)
        return out, cls_out, aux_out

def calculate_dice_score(preds, targets, smooth=1e-5, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (preds_flat * targets_flat).sum()
    dice = (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
    return dice.item()

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
    def forward(self, pred, target, alpha=0.825, gamma=2):
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        F_loss = alpha_t * (1-pt)**gamma * bce
        focal_loss = F_loss.mean()
        return focal_loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.focal = FocalLoss()
    def forward(self, pred, target, smooth=1.):
        target = target.float()
        focal_loss = self.focal(pred, target, alpha=0.8, gamma=3.0)
        pred_soft = torch.sigmoid(pred)
        
        pred_flat = pred_soft.view(-1)
        target_flat = target.view(-1)
            
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 0.5 * focal_loss + 0.5 * (1 - dice)

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, outputs, masks, labels):
        seg_pred, cls_pred, seg_aux = outputs

        l_seg = self.dice_loss(seg_pred, masks)
        l_cls = self.focal(cls_pred, labels)
        l_aux = self.dice_loss(seg_aux, masks)

        return l_seg + 3 * l_cls + 0.5 * l_aux
    
def train(model, loader, optimizer, loss_fn, device, scaler):
    model.train()
    epoch_loss = 0
    
    for i, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1).float()
        labels = (masks.view(masks.size(0), -1).sum(dim=1) > 0).float().unsqueeze(1)
        with autocast(device_type=device, dtype=torch.float16):
            output = model(images)
            loss = loss_fn(output, masks, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(loader)
    
def val(model, loader, device, loss_fn, threshold=0.5):
    model.eval()
    dice_score = 0.0
    val_loss = 0.0
    cls_loss = 0.0
    cls_acc = 0.0
    cls_loss_fn = FocalLoss()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1).float()
            labels = (masks.view(masks.size(0), -1).sum(dim=1) > 0).float().unsqueeze(1)
            pred, probs, aux = model(images)
            cls_probs = torch.sigmoid(probs)
            cls_preds = (cls_probs > 0.5).float()
            val_loss += loss_fn((pred,probs,aux), masks, labels).item()
            dice_score += calculate_dice_score(pred, masks, threshold=threshold)
            cls_acc += (cls_preds == labels).float().mean().item()
            cls_loss += cls_loss_fn(probs, labels)

            all_preds.extend(cls_preds.cpu().numpy().flatten())
            all_targets.extend(labels.cpu().numpy().flatten())
            
    batch = len(loader)
    val_loss /= batch
    dice_score /= batch
    cls_loss /= batch
    cls_acc /=batch

    cls_recall = recall_score(all_targets, all_preds, zero_division=0)
    cls_precision = precision_score(all_targets, all_preds, zero_division=0)
    return val_loss, dice_score, cls_loss, cls_acc, cls_recall, cls_precision


#Data load and split
clean_imgs, clean_masks = analyze_and_clean_dataset('/kaggle/input/ultrasound-nerve-segmentation/train')

subject_ids = []
for img_path in clean_imgs:
    filename = os.path.basename(img_path)
    subject_id = filename.split('_')[0] # Lấy số '1' từ '1_1.tif'
    subject_ids.append(subject_id)

subject_ids = np.array(subject_ids)
clean_imgs_np = np.array(clean_imgs)
clean_masks_np = np.array(clean_masks)

# 2. Sử dụng GroupKFold để chia (đảm bảo cùng 1 subject không nằm ở 2 tập)
gkf = GroupKFold(n_splits=5) 
# Lấy fold đầu tiên làm validation (20%)
train_idx, val_idx = next(gkf.split(clean_imgs_np, clean_masks_np, groups=subject_ids))

img_train, val_train = clean_imgs_np[train_idx], clean_imgs_np[val_idx]
img_mask, val_mask = clean_masks_np[train_idx], clean_masks_np[val_idx]

train_labels = []
for mask_path in img_mask:
    mask = cv2.imread(mask_path, 0)
    if np.sum(mask) > 0:
        train_labels.append(1)
    else:
        train_labels.append(0)

train_labels = np.array(train_labels)

class_counts = np.bincount(train_labels)
num_samples = len(train_labels)
class_weights = 1. / class_counts
sample_weights = class_weights[train_labels]
sample_weights = torch.from_numpy(sample_weights).double()
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)

train_dataset = SegmentationDataset(img_train, img_mask, transform=train_transform)
val_dataset = SegmentationDataset(val_train, val_mask, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, shuffle=False, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)


#We need another model to classifier which model contain mask or not
#Or maybe not, this model is fuckingly strong, the only model need this might be vanilla U-net
#Yes we need a fucking classifier ni**a


#Setup Env
if torch.cuda.device_count() > 1:
    torch.cuda.empty_cache()
    device = "cuda:0"
else:
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.DataParallel(ARGA_U_net(3,1)).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_fn = MultiTaskLoss()
scaler = GradScaler('cuda')
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)


#Training
best_val_dice = 0.0
best_model = model
for epoch in range(50):
    train_loss = train(model, train_loader, optimizer, loss_fn, device, scaler)
    val_loss, dice_score, cls_loss, cls_acc, cls_recall, cls_precision = val(model, val_loader, device, loss_fn)
    scheduler.step()
    if dice_score > best_val_dice:
        best_val_dice = dice_score
        best_model = model
        torch.save(model.module.state_dict(), 'best_ARGA_U_net_model.pth')
    print(f"Epoch {epoch+1}:\n Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dice: {dice_score:.4f}")
    print(f" Val Classifier loss: {cls_loss:.4f} | Val Classifier Acc: {cls_acc:.4f} |\n Recall: {cls_recall:.4f} | Precision: {cls_precision:.4f}")
print(f"Training finished")


#Threshold checking
def find_best_thres(model,loader,device): 
    best_thres = 0
    best_dice = 0
    thres_test = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for threshold in thres_test:
        _, dice, cls_loss, cls_acc, cls_recall, cls_precision  = val(model, val_loader, device, loss_fn, threshold)
        if (dice > best_dice):
            best_thres = threshold
            best_dice = dice
    print(f'Best threshold: {best_thres}')
    print(f'Current Dice with best threshold: {best_dice}')
    return best_thres

best_thres = find_best_thres(best_model,val_loader,device)

def find_best_cls_threshold(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            labels = (masks.view(masks.size(0), -1).sum(dim=1) > 0).float().cpu().numpy()
            _, cls_logits,_ = model(images)
            probs = torch.sigmoid(cls_logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)
            
    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()
    best_f1 = 0
    best_cls_thres = 0.5
    thresholds = np.arange(0.1, 1.0, 0.05)
    
    print(f"{'Threshold':<10} | {'F1-Score':<10} | {'Acc':<10} | {'Recall':<10} | {'Precision':<10}")
    for thres in thresholds:
        preds = (all_probs > thres).astype(int)
        f1 = f1_score(all_labels, preds)
        acc = accuracy_score(all_labels, preds)
        recall = recall_score(all_labels, preds, zero_division=0)
        prec = precision_score(all_labels, preds, zero_division=0)
        print(f"{thres:.2f}       | {f1:.4f}     | {acc:.4f}     | {recall:.4f}     | {prec:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_cls_thres = thres
    print(f"Best Classifier Threshold (F1): {best_cls_thres}")
    print(f"Best F1 Score: {best_f1}")
    return best_cls_thres
    
best_cls_thres = find_best_cls_threshold(best_model, val_loader, device)


#Submission and Post Processing
test_set = TiffTestDataset('/kaggle/input/ultrasound-nerve-segmentation/test',test_transform)
data_test = DataLoader(test_set, batch_size=16)

def rle_encode(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
    
def remove_small_objects(mask, min_size=100):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            cleaned_mask[labels == i] = 1
    return cleaned_mask
    
def keep_largest_component(mask):
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels < 2:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned_mask = np.zeros_like(mask)
    cleaned_mask[labels == largest_label] = 1
    return cleaned_mask

def fill_holes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, -1, 1, thickness=cv2.FILLED)
    
    return filled_mask

data = []
model.eval()
with torch.no_grad():
    for images, ids in data_test:
        images = images.to(device)
        out_seg_orig, out_cls_orig = model(images)
        pred_seg_orig = torch.sigmoid(out_seg_orig)
        pred_cls_orig = torch.sigmoid(out_cls_orig)
        
        images_flip = torch.flip(images, dims=[3])
        out_seg_flip, out_cls_flip = model(images_flip)
        
        pred_seg_flip = torch.sigmoid(out_seg_flip)
        pred_seg_flip = torch.flip(pred_seg_flip, dims=[3])
        
        pred_cls_flip = torch.sigmoid(out_cls_flip)

        pred_seg = (pred_seg_orig + pred_seg_flip) / 2.0
        pred_cls = (pred_cls_orig + pred_cls_flip) / 2.0
        
        preds_seg_np = pred_seg.cpu().numpy()
        preds_cls_np = pred_cls.cpu().numpy()

        pred_seg = (pred_seg_orig + pred_seg_flip) / 2.0
        pred_cls = (pred_cls_orig + pred_cls_flip) / 2.0
        
        preds_seg_np = pred_seg.cpu().numpy()
        preds_cls_np = pred_cls.cpu().numpy()
        
        for i in range(images.size(0)):
            img_id = int(ids[i].item())
            mask_prob = preds_seg_np[i, 0] 
            nerve_prob = preds_cls_np[i, 0]
            
            mask_prob_resized = cv2.resize(mask_prob, (580, 420), interpolation=cv2.INTER_LINEAR)
            mask_binary = (mask_prob_resized > best_thres).astype(np.uint8)
            mask_clean = remove_small_objects(mask_binary)
            mask_clean = keep_largest_component(mask_clean)
            mask_clean = fill_holes(mask_clean)
            mask_area = np.sum(mask_clean)
            
            if nerve_prob < best_cls_thres: 
                mask_clean = np.zeros_like(mask_clean)
            else:
                if mask_area < 2000 and nerve_prob < 0.5:
                    mask_clean = np.zeros_like(mask_clean)
            
            if mask_clean.sum() == 0:
                encoded = ""
            else:
                encoded = rle_encode(mask_clean)
            data.append({
                "img": img_id,
                "pixels": encoded
            })
data = sorted(data, key=lambda x: x['img'])
df = pd.DataFrame(data)
df.to_csv('submission.csv', index=False)

