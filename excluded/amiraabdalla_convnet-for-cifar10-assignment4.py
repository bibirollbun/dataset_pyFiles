# Cell 1 — Setup and installs (run first)
!apt-get update -qq
# libarchive for extracting .7z (competition test.7z)
!apt-get install -y libarchive-dev -qq
!pip install -q libarchive


# Cell 2 — Imports and basic config
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms, datasets


# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



# Cell 3 — Data preparation (training/validation/test)
# We'll use torchvision CIFAR10 for training & a validation split.
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])

data_root = "/kaggle/working/data_cifar"
os.makedirs(data_root, exist_ok=True)

train_full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
test_official = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

# Create train/val split from official train set for validation
val_size = 5000
train_size = len(train_full) - val_size
train_set, val_set = random_split(train_full, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(seed))

print("train size:", len(train_set), "val size:", len(val_set), "official test size:", len(test_official))



# Cell 4 — Dataloaders
batch_size = 128
num_workers = 2

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_official, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


# Cell 5 — Define ConvNet (randomly initialized; no pre-trained weights)
# A ResNet-style small model from scratch (implementation from scratch; weights randomly init)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class SmallResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32,  num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64,  num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64*block.expansion, num_classes)

        # Initialize weights (default nn init is fine but we make it explicit)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = SmallResNet(BasicBlock, [2,2,2], num_classes=10).to(device)
print(model)



# Cell 6 — Training utilities (train/validate functions)
criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss/total, correct/total

def evaluate(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss/total, correct/total



# Cell 7 — Train the model
# Hyperparameters — you may increase epochs for better accuracy (longer training yields better results)
lr = 0.1
epochs = 60   # Increase to 100-200 if you can afford time
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45], gamma=0.1)

best_val_acc = 0.0
history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
start_time = time.time()

for epoch in range(1, epochs+1):
    t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, device)
    v_loss, v_acc = evaluate(model, val_loader, device)
    scheduler.step()

    history["train_loss"].append(t_loss)
    history["train_acc"].append(t_acc)
    history["val_loss"].append(v_loss)
    history["val_acc"].append(v_acc)

    if v_acc > best_val_acc:
        best_val_acc = v_acc
        # Save best model
        torch.save(model.state_dict(), "/kaggle/working/best_model.pth")

    if epoch % 5 == 0 or epoch==1:
        print(f"Epoch {epoch}/{epochs} | train_loss={t_loss:.4f} train_acc={t_acc:.4f} | val_loss={v_loss:.4f} val_acc={v_acc:.4f} | best_val={best_val_acc:.4f}")

total_time = time.time() - start_time
print(f"Training finished in {total_time/60:.2f} minutes. Best val acc: {best_val_acc:.4f}")



# Cell 8 — Evaluate on official CIFAR10 test set (estimate Kaggle score)
model.load_state_dict(torch.load("/kaggle/working/best_model.pth"))
test_loss, test_acc = evaluate(model, test_loader, device)
print(f"Official CIFAR10 test accuracy (estimated Kaggle score): {test_acc:.4f}")


# Cell 9 — Extract competition test.7z and prepare file order
# The competition provides test.7z. On Kaggle notebooks, competition datasets usually are at /kaggle/input/<comp-name>
# Replace the input path below if competition data is located elsewhere.

# Many times Kaggle places the 'test.7z' in /kaggle/input/cifar-10/test.7z
input_test_archive = '/kaggle/input/cifar-10/test.7z'
output_test_folder = '/kaggle/working/test'
os.makedirs(output_test_folder, exist_ok=True)

!7z x -y {input_test_archive} -o{output_test_folder}
print("Extraction complete. Files:", len(list(Path(output_test_folder).glob("*"))))


!zip -r cifar10_extracted.zip /kaggle/working/test


!ls -lh /kaggle/working


from IPython.display import FileLink
FileLink('/kaggle/working/cifar10_extracted.zip')


!zip -r /kaggle/working/cifar10_extracted.zip /kaggle/working/test


# --------------------------------------------
# Final prediction + submission for Kaggle
# --------------------------------------------
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
import pandas as pd

# Dataset for Kaggle test images
class KaggleTestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.paths = sorted([p for p in Path(root).glob("*.png")])  # all .png files
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, path.stem  # return filename stem (like "1", "2", ...)

# ⚠️ Adjust root to where your uploaded dataset is located
test_root = "/kaggle/input/convnettest/kaggle/working/test/test"
kaggle_test_ds = KaggleTestDataset(test_root, transform=transform_test)
kaggle_test_loader = DataLoader(kaggle_test_ds, batch_size=256, shuffle=False, num_workers=2)

print("Number of test images found:", len(kaggle_test_ds))
print("First 5 sample IDs:", [kaggle_test_ds[i][1] for i in range(5)])

# Predict
model.eval()
preds = []
ids = []

with torch.no_grad():
    for imgs, names in tqdm(kaggle_test_loader):
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, predicted = outputs.max(1)
        preds.extend(predicted.cpu().numpy().tolist())
        ids.extend([int(n) for n in names])  # convert string to int

# ✅ Map numeric predictions to CIFAR-10 class names
label_map = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}
preds_str = [label_map[p] for p in preds]

# Build submission DataFrame
sub_df = pd.DataFrame({"id": ids, "label": preds_str})
sub_df = sub_df.sort_values("id")  # must be sorted by id
sub_df.to_csv("/kaggle/working/submission.csv", index=False)

print("✅ Submission saved to /kaggle/working/submission.csv")
print("Rows:", len(sub_df))
print(sub_df.head())



# Cell 11 — Download link instructions (Kaggle UI)
# In Kaggle you can click "Files" → download submission.csv or use the UI to submit.
!ls -lh /kaggle/working/submission.csv

