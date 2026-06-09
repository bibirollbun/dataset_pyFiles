!pip install -q segmentation-models-pytorch --no-cache-dir



import os, numpy as np, matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import warnings

warnings.filterwarnings('ignore')


class SaltDataset(Dataset):
    def __init__(self, image_ids, image_dir, mask_dir):
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img = Image.open(os.path.join(self.image_dir, img_id)).convert("L")
        mask = Image.open(os.path.join(self.mask_dir, img_id)).convert("L")

        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0

        img = torch.tensor(img).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        return img, mask


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


def iou_score(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def f1_score(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds) > threshold
    preds = preds.view(-1).float()
    targets = targets.view(-1).float()

    tp = (preds * targets).sum()
    fp = ((1 - targets) * preds).sum()
    fn = (targets * (1 - preds)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return 2 * (precision * recall) / (precision + recall + 1e-8)


def train_model(loss_fn, train_loader, val_loader, model, epochs=3):
    model = model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_iou = 0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to("cuda"), y.to("cuda")
            preds = model(x)
            loss = loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        ious = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to("cuda"), y.to("cuda")
                preds = model(x)
                ious.append(iou_score(preds, y).item())
        mean_iou = np.mean(ious)
        print(f"Epoch {epoch+1}, Val IoU: {mean_iou:.4f}")
        best_iou = max(best_iou, mean_iou)
    
    return best_iou


import zipfile
import os

with zipfile.ZipFile("../input/tgs-salt-identification-challenge/train.zip", 'r') as zip_ref:
    zip_ref.extractall("./data/train")


# Setup
image_dir = "/kaggle/working/data/train/images"
mask_dir = "/kaggle/working/data/train/masks"
ids = os.listdir(image_dir)
train_ids, val_ids = train_test_split(ids, test_size=0.2, random_state=42)

train_dataset = SaltDataset(train_ids, image_dir, mask_dir)
val_dataset = SaltDataset(val_ids, image_dir, mask_dir)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model (U-Net)
model_dice = smp.Unet("resnet18", in_channels=1, classes=1)
model_focal = smp.Unet("resnet18", in_channels=1, classes=1)

# Compare
print("ğŸ”µ Training with Dice Loss:")
dice_iou = train_model(DiceLoss(), train_loader, val_loader, model_dice)

print("\nğŸ”´ Training with Focal Loss:")
focal_iou = train_model(FocalLoss(), train_loader, val_loader, model_focal)

print(f"\nâœ… Dice Loss Best IoU: {dice_iou:.4f}")
print(f"âœ… Focal Loss Best IoU: {focal_iou:.4f}")

