# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import os
import cv2
import zipfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(),
    ToTensorV2()
])

# Validation augmentations (only normalization)
val_transform = A.Compose([
    A.Normalize(),
    ToTensorV2()
])


class ResNetEncoder(nn.Module):
    def __init__(self, backbone_name='resnet34', pretrained=True):
        super().__init__()
        if backbone_name == 'resnet34':
            backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise ValueError("Unsupported backbone!")

        self.initial = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        return x0, x1, x2, x3, x4

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if x.shape != skip.shape:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class UNet(nn.Module):
    def __init__(self, backbone_name='resnet34', pretrained=True):
        super().__init__()
        self.encoder = ResNetEncoder(backbone_name, pretrained)
        if backbone_name == 'resnet34':
            channels = [64, 64, 128, 256, 512]
        else:
            channels = [64, 256, 512, 1024, 2048]

        self.decoder4 = DecoderBlock(channels[4] + channels[3], 512)
        self.decoder3 = DecoderBlock(512 + channels[2], 256)
        self.decoder2 = DecoderBlock(256 + channels[1], 128)
        self.decoder1 = DecoderBlock(128 + channels[0], 64)
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)
        d4 = self.decoder4(x4, x3)
        d3 = self.decoder3(d4, x2)
        d2 = self.decoder2(d3, x1)
        d1 = self.decoder1(d2, x0)
        out = self.final_conv(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = StableBCELoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)  # no sigmoid here!
        probs = torch.sigmoid(logits)         # apply sigmoid separately for dice
        intersection = (probs * targets).sum()
        dice_loss = 1 - (2. * intersection + 1e-6) / (probs.sum() + targets.sum() + 1e-6)
        return bce_loss + dice_loss


def mean(l, ignore_nan=False, empty=0):
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard
def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss
def lovasz_hinge_flat(logits, labels, ignore=None):
    if ignore is None:
        logits, labels = logits.view(-1), labels.view(-1)
    else:
        mask = (labels != ignore)
        logits, labels = logits[mask], labels[mask]
    signs = 2. * labels.float() - 1.
    errors = 1. - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss
def flatten_binary_scores(scores, labels, ignore=None):
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    return scores[valid], labels[valid]

class LovaszLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        logits = logits.squeeze(1)  # (B,1,H,W) -> (B,H,W)
        targets = targets.squeeze(1)
        return lovasz_hinge(logits, targets)


class ComboLoss(nn.Module):
    def __init__(self, weight_bce_dice=0.5, weight_lovasz=0.5):
        super().__init__()
        self.bce_dice = BCEDiceLoss()
        self.lovasz = LovaszLoss()
        self.w1 = weight_bce_dice
        self.w2 = weight_lovasz

    def forward(self, logits, targets):
        loss1 = self.bce_dice(logits, targets)
        loss2 = self.lovasz(logits, targets)
        return self.w1 * loss1 + self.w2 * loss2



zip_path = '/kaggle/input/tgs-salt-identification-challenge/train.zip'
output_dir = '/kaggle/working/train'

if not os.path.exists(output_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print("Train.zip extracted!")

# Set paths
image_dir = '/kaggle/working/train/images'
mask_dir = '/kaggle/working/train/masks'

# Load filenames
image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Full paths
image_paths = [os.path.join(image_dir, f) for f in image_filenames]
mask_paths = [os.path.join(mask_dir, f) for f in image_filenames]

# Helper: keep only images with some salt
def has_salt(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mask.sum() > 0  # non-empty mask

# Filter training images to avoid empty masks
filtered = [(img, msk) for img, msk in zip(image_paths, mask_paths) if has_salt(msk)]

# Unpack
filtered_images, filtered_masks = zip(*filtered)

# Train/Validation Split (80/20)
train_images, val_images, train_masks, val_masks = train_test_split(
    filtered_images,
    filtered_masks,
    test_size=0.2,
    random_state=42
)

print(f"Training images (non-empty masks): {len(train_images)}")
print(f"Validation images: {len(val_images)}")


class SaltDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        mask = mask.float() / 255.0  # <<<< ðŸŒŸ IMPORTANT: normalize mask to [0,1]

        return image, mask

# Create Datasets
train_dataset = SaltDataset(train_images, train_masks, transform=train_transform)
val_dataset = SaltDataset(val_images, val_masks, transform=val_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

print("âœ… DataLoaders are ready!")


def compute_iou(preds, targets, threshold=0.5, eps=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean()

def train_validate(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
    model.to(device)
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_iou += compute_iou(outputs, masks).item()
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val IoU: {val_iou:.4f}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")




Model_Final_Losss = UNet(backbone_name='resnet50', pretrained=True).to(device)
optimizer_exp3 = optim.Adam(Model_Final_Losss.parameters(), lr=1e-4)
criterion_Final_Loss = ComboLoss()
print("\nStarting Experiment 3: UNet(ResNet50) + BCE + Dice Loss")
train_validate(
    model=Model_Final_Losss,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer_exp3,
    criterion=criterion_Final_Loss,
    num_epochs=20,
    device=device
)

