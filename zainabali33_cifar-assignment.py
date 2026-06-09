# =====================================================
# CIFAR-10 Competition - SimpleCNN (Random Init)
# Train on official CIFAR-10 (torchvision)
# Predict on uploaded Kaggle test folder (already extracted)
# =====================================================

import os, glob
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

# ===================
# 1. Settings
# ===================
batch_size = 128
num_epochs = 50   # increase to 100+ for higher accuracy
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===================
# 2. Compute CIFAR-10 Mean and Std
# ===================
raw_trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True,
    transform=transforms.ToTensor()
)

raw_loader = DataLoader(raw_trainset, batch_size=500, shuffle=False, num_workers=2)

mean = 0.0
std = 0.0
nb_samples = 0
for images, _ in raw_loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples
print("Calculated mean:", mean)
print("Calculated std:", std)

# ===================
# 3. Define Transforms with Computed Mean/Std
# ===================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# ===================
# 4. Official CIFAR-10 Train + Val datasets
# ===================
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=2)

valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform_test)
valloader = DataLoader(valset, batch_size=batch_size,
                       shuffle=False, num_workers=2)

class_names = trainset.classes
print("Classes:", class_names)

# ===================
# 5. Define Simple CNN
# ===================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*4*4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)

# ===================
# 6. Loss + Optimizer
# ===================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# ===================
# 7. Training & Validation
# ===================
def train_one_epoch():
    model.train()
    running_loss, correct, total = 0, 0, 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss/len(trainloader), 100.*correct/total

def validate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in valloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.*correct/total

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch()
    val_acc = validate()
    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | "
          f"Val Acc (Expected Kaggle Score): {val_acc:.2f}%")

torch.save(model.state_dict(), "cifar10_simplecnn.pth")

# ===================
# 8. Kaggle Test Dataset (already extracted and uploaded)
# ===================
test_dir = "/kaggle/input/cifar-test-dataset/test"
print("Found test images:", len(glob.glob(os.path.join(test_dir, '*.png'))))

class TestDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, os.path.splitext(os.path.basename(img_path))[0]

test_files = sorted(glob.glob(os.path.join(test_dir, '*.png')))
test_dataset = TestDataset(test_files, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ===================
# 9. Run predictions on Kaggle test set
# ===================
model.eval()
all_preds, all_ids = [], []
with torch.no_grad():
    for inputs, ids in tqdm(test_loader, desc="Predicting"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_ids.extend(ids)

submission = pd.DataFrame({
    "id": all_ids,
    "label": [class_names[p] for p in all_preds]
})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("✅ Submission file saved: /kaggle/working/submission.csv")

