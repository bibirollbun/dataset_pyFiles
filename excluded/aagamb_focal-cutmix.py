import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
import random

# CutMix Bounding Box
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Random center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# Paths
DATA_DIR = "/kaggle/input/human-protein-atlas-image-classification"
# DATA_DIR = "/kaggle/input/humanatlas/d"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
CSV_PATH = os.path.join(DATA_DIR, "train.csv")

# Load CSV
df = pd.read_csv(CSV_PATH)
df['Target'] = df['Target'].apply(lambda x: list(map(int, str(x).split())) if pd.notna(x) else [])

# Train-validation split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Define Custom Dataset
class ProteinDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.num_classes = 28

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['Id']
        img_paths = [os.path.join(self.img_dir, f"{img_id}_{color}.png") for color in ["red", "green", "blue", "yellow"]]

        # Load 4-channel image
        channels = [np.array(Image.open(img_path), dtype=np.float32) if os.path.exists(img_path) else np.zeros((512, 512), dtype=np.float32) for img_path in img_paths]
        image = np.stack(channels, axis=-1)  # Shape: (H, W, 4)

        # Normalize to 0-1
        image = image / 255.0

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Multi-label target
        target = np.zeros(self.num_classes, dtype=np.float32)
        for label in self.df.iloc[idx]['Target']:
            target[label] = 1.0
        return image, torch.tensor(target, dtype=torch.float32)

# Define Data Augmentation
train_transform = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30),
    A.Normalize(mean=[0.5]*4, std=[0.5]*4, max_pixel_value=1.0),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.5]*4, std=[0.5]*4, max_pixel_value=1.0),
    ToTensorV2()
])

# Dataloaders
train_dataset = ProteinDataset(train_df, TRAIN_DIR, transform=train_transform)
val_dataset = ProteinDataset(val_df, TRAIN_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

# Modify ResNet to accept 4 channels
class CustomResNet(nn.Module):
    def __init__(self, num_classes=28):
        super().__init__()
        from torchvision.models import ResNet50_Weights
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Modify first conv layer for 4-channel input
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify output layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        targets = targets.type(inputs.type())

        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (self.alpha * (1 - pt) ** self.gamma)

        loss = focal_weight * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CustomResNet().to(device)
criterion = FocalLoss(alpha=1.0, gamma=2.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 50

# Evaluation Metrics
def mean_average_precision(y_true, y_pred):
    ap_per_class = []
    for i in range(y_true.shape[1]):
        ap = average_precision_score(y_true[:, i], y_pred[:, i])
        ap_per_class.append(ap)
    return np.mean(ap_per_class)

def f1_metric(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).astype(int)
    return f1_score(y_true, y_pred, average="macro")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        r = np.random.rand(1)
        cutmix_prob = 0.5  # Probability to apply CutMix
        optimizer.zero_grad()

        if r < cutmix_prob:
            # Apply CutMix
            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(images.size()[0]).to(device)
            target_a = targets
            target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]

            # Adjust lambda
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

            outputs = model(images)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation Loop
    model.eval()
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            y_true_list.append(targets.cpu().numpy())
            y_pred_list.append(torch.sigmoid(outputs).cpu().numpy())

    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)

    map_score = mean_average_precision(y_true, y_pred)
    f1_score_macro = f1_metric(y_true, y_pred)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, mAP: {map_score:.4f}, F1 Score: {f1_score_macro:.4f}")

print("Training Completed.")

