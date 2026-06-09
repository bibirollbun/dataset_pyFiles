import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from time import time
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

!apt-get install -y p7zip-full

!7z x /kaggle/input/cifar-10/train.7z -o/kaggle/working/train

class KaggleCIFAR10(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        classes = sorted(self.df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        label = self.class_to_idx[row["label"]]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Helper: compute accuracy
def accuracy(outputs, targets):
    """
    Compute classification accuracy.
    outputs: raw logits from the model (batch_size, num_classes)
    targets: ground-truth labels (batch_size,)
    """
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    Returns average loss and accuracy over the epoch.
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 1. Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 2. Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 3. Track metrics
        running_loss += loss.item() * images.size(0)
        running_acc += accuracy(outputs, labels) * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_acc / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on validation data.
    """
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            running_acc += accuracy(outputs, labels) * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_acc / len(dataloader.dataset)
    return epoch_loss, epoch_acc


# Simple transforms
baseline_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),  # mean of CIFAR-10
                         (0.2470, 0.2435, 0.2616))  # std of CIFAR-10
])

# Kaggle dataset paths
TRAIN_CSV = "/kaggle/input/cifar-10/trainLabels.csv"
TRAIN_DIR = "/kaggle/working/train"

# Load training dataset
train_dataset_baseline = KaggleCIFAR10(
    csv_file=TRAIN_CSV,
    img_dir=TRAIN_DIR,
    transform=baseline_transform
)

# Create validation split
val_ratio = 0.1
num_train = int(len(train_dataset_baseline) * (1 - val_ratio))
num_val   = len(train_dataset_baseline) - num_train

train_dataset_baseline, val_dataset_baseline = torch.utils.data.random_split(
    train_dataset_baseline,
    [num_train, num_val],
    generator=torch.Generator().manual_seed(42)
)


# Kaggle dataset paths
TRAIN_CSV = "/kaggle/input/cifar-10/trainLabels.csv"
TRAIN_DIR = "/kaggle/working/train/train"

# Load training dataset
train_dataset_baseline = KaggleCIFAR10(
    csv_file=TRAIN_CSV,
    img_dir=TRAIN_DIR,
    transform=baseline_transform
)

# Create validation split
val_ratio = 0.1
num_train = int(len(train_dataset_baseline) * (1 - val_ratio))
num_val   = len(train_dataset_baseline) - num_train

train_dataset_baseline, val_dataset_baseline = torch.utils.data.random_split(
    train_dataset_baseline,
    [num_train, num_val],
    generator=torch.Generator().manual_seed(42)
)


batch_size = 128

train_loader_baseline = DataLoader(train_dataset_baseline, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader_baseline = DataLoader(val_dataset_baseline, batch_size=batch_size, shuffle=False, num_workers=2)


class SimpleCNN(nn.Module):
    """
    Very simple CNN for CIFAR-10 (baseline).
    - Conv -> ReLU -> MaxPool
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Input: (3, 32, 32)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def run_baseline(num_epochs=10, lr=0.01):
    model = SimpleCNN(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        start = time()
        train_loss, train_acc = train_one_epoch(model, train_loader_baseline,
                                                criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader_baseline,
                                     criterion, device)
        elapsed = time() - start

        print(f"[Baseline][Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} "
              f"| Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "baseline_best.pth")

    print(f"[Baseline] Best Val Acc: {best_val_acc:.4f}")
    return model


# To run baseline:
baseline_model = run_baseline(num_epochs=10, lr=0.01)


# PART 2: IMPROVED MODEL 

# Data augmentation + better normalization
improved_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   
    transforms.RandomHorizontalFlip(),       
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])


improved_val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

val_ratio = 0.1


full_dataset_for_split = KaggleCIFAR10(
    csv_file=TRAIN_CSV,
    img_dir=TRAIN_DIR,
    transform=None
)

num_samples = len(full_dataset_for_split)
num_train = int(num_samples * (1 - val_ratio))
num_val = num_samples - num_train

# Reproducible random split
g = torch.Generator().manual_seed(42)
indices = torch.randperm(num_samples, generator=g).tolist()
train_indices = indices[:num_train]
val_indices = indices[num_train:]

# Two datasets with different transforms
train_dataset_improved_base = KaggleCIFAR10(
    csv_file=TRAIN_CSV,
    img_dir=TRAIN_DIR,
    transform=improved_train_transform
)

val_dataset_improved_base = KaggleCIFAR10(
    csv_file=TRAIN_CSV,
    img_dir=TRAIN_DIR,
    transform=improved_val_transform
)

from torch.utils.data import Subset

train_dataset_improved = Subset(train_dataset_improved_base, train_indices)
val_dataset_improved = Subset(val_dataset_improved_base, val_indices)

batch_size_improved = 128

train_loader_improved = DataLoader(
    train_dataset_improved,
    batch_size=batch_size_improved,
    shuffle=True,
    num_workers=2
)

val_loader_improved = DataLoader(
    val_dataset_improved,
    batch_size=batch_size_improved,
    shuffle=False,
    num_workers=2
)


class BetterCNN(nn.Module):
    """
    Improved CNN:
    - More conv layers
    - BatchNorm after each conv
    - Dropout in the classifier
    This increases model capacity and adds regularization.
    """
    def __init__(self, num_classes=10):
        super(BetterCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # (64, 16, 16)

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # (128, 8, 8)

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # (256, 4, 4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),          # dropout regularization
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def run_improved(num_epochs=20, lr=0.1):
    """
    Train the improved CNN:
    - Uses data augmentation
    - Uses weight decay (L2)
    - Uses MultiStepLR scheduler
    """
    model = BetterCNN(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4    # L2 regularization
    )

    # Reduce LR at epochs 10 and 15
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10, 15],
        gamma=0.1
    )

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        start = time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader_improved, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader_improved, criterion, device
        )

        scheduler.step()
        elapsed = time() - start

        current_lr = scheduler.get_last_lr()[0]

        print(f"[Improved][Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} "
              f"| Time: {elapsed:.1f}s | LR: {current_lr:.5f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "improved_best.pth")

    print(f"[Improved] Best Val Acc: {best_val_acc:.4f}")
    return model


# Run improved model:
improved_model = run_improved(num_epochs=20, lr=0.1)


# PART 3: TRANSFER LEARNING using ResNet18

from torchvision.models import resnet18, ResNet18_Weights

# ImageNet normalization
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std  = (0.229, 0.224, 0.225)

transfer_train_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomCrop(112),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

transfer_val_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# Create train/val split
val_ratio = 0.1

full_dataset_for_split_tf = KaggleCIFAR10(
    csv_file=TRAIN_CSV,
    img_dir=TRAIN_DIR,
    transform=None
)

num_samples_tf = len(full_dataset_for_split_tf)
num_train_tf = int(num_samples_tf * (1 - val_ratio))
num_val_tf = num_samples_tf - num_train_tf

g_tf = torch.Generator().manual_seed(42)
indices_tf = torch.randperm(num_samples_tf, generator=g_tf).tolist()
train_indices_tf = indices_tf[:num_train_tf]
val_indices_tf = indices_tf[num_train_tf:]

# Two datasets with different transforms
train_dataset_transfer_base = KaggleCIFAR10(
    csv_file=TRAIN_CSV,
    img_dir=TRAIN_DIR,
    transform=transfer_train_transform
)

val_dataset_transfer_base = KaggleCIFAR10(
    csv_file=TRAIN_CSV,
    img_dir=TRAIN_DIR,
    transform=transfer_val_transform
)

train_dataset_transfer = torch.utils.data.Subset(
    train_dataset_transfer_base, train_indices_tf
)
val_dataset_transfer = torch.utils.data.Subset(
    val_dataset_transfer_base, val_indices_tf
)

batch_size_transfer = 64

train_loader_transfer = DataLoader(
    train_dataset_transfer,
    batch_size=batch_size_transfer,
    shuffle=True,
    num_workers=2
)

val_loader_transfer = DataLoader(
    val_dataset_transfer,
    batch_size=batch_size_transfer,
    shuffle=False,
    num_workers=2
)


def create_resnet18_transfer(num_classes=10,
                             freeze_features=True,
                             unfreeze_last_block=True):
    """
    Load ResNet18 with pretrained weights from a local .pth file.
    """

    weight_path = "/kaggle/input/resnet18-weights/resnet18-f37072fd.pth"

    try:
        print("Loading local pretrained weights...")
        
        model = resnet18(weights=None)
        
        state_dict = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("Successfully loaded pretrained weights.")
    except Exception as e:
        print("ERROR loading pretrained weights:", e)
        print("Using randomly initialized ResNet18 instead.")
        model = resnet18(weights=None)


    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Freeze earlier layers
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

        if unfreeze_last_block:
            for param in model.layer4.parameters():
                param.requires_grad = True

    return model


def run_transfer(num_epochs=10, lr=1e-3):
    """
    Train ResNet18 with transfer learning.
    - Uses pretrained weights
    - Freezes most layers, fine-tunes last block + classifier
    """
    model = create_resnet18_transfer(
        num_classes=10,
        freeze_features=True,
        unfreeze_last_block=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    # Scheduler: reduce LR if val loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=2,
        verbose=True
    )

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        start = time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader_transfer, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader_transfer, criterion, device
        )

        scheduler.step(val_loss)
        elapsed = time() - start

        print(f"[Transfer][Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} "
              f"| Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "transfer_best.pth")

    print(f"[Transfer] Best Val Acc: {best_val_acc:.4f}")
    return model


# Run transfer learning:
transfer_model = run_transfer(num_epochs=3, lr=1e-3)

