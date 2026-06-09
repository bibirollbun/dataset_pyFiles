# ===================
# 1. Imports & Settings
# ===================
import os, glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
import torchvision.transforms as transforms

batch_size = 128
num_epochs = 200
base_lr = 0.001   # reduced for stability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===================
# 2. Compute CIFAR-10 Mean & Std
# ===================
raw_trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transforms.ToTensor()
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
print("CIFAR10 mean:", mean, "std:", std)

# ===================
# 3. Data Augmentation
# ===================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Split train into train+val
full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_size = int(0.9 * len(full_trainset))
val_size = len(full_trainset) - train_size
trainset, valset = random_split(full_trainset, [train_size, val_size])

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

class_names = full_trainset.classes

# ===================
# 4. Custom ConvNet with He Init + Dropout
# ===================
class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

net = CustomCNN().to(device)

# ===================
# 5. Optimizer & Scheduler
# ===================
optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=1e-4)

def lr_lambda(current_step):
    warmup_steps = 500
    total_steps = num_epochs * len(trainloader)
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + np.cos(np.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# SWA setup (last 20 epochs)
swa_model = AveragedModel(net)
swa_start = 180

# ===================
# 6. CutMix (Less Frequent)
# ===================
def rand_bbox(size, lam):
    W = size[2]; H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(x.device)
    target_a = y
    target_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, target_a, target_b, lam

# ===================
# 7. Training Loop (CutMix 30%, SWA after 180)
# ===================
for epoch in range(num_epochs):
    net.train()
    running_loss, correct, total = 0.0, 0, 0

    for i, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Apply CutMix with lower probability
        if np.random.rand() < 0.3:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets)
            outputs = net(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} Loss: {running_loss/len(trainloader):.4f} Train Acc: {acc:.2f}%")

    # Validation
    net.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x, y in valloader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            _, pred = out.max(1)
            val_total += y.size(0)
            val_correct += pred.eq(y).sum().item()
    val_acc = 100. * val_correct / val_total
    print(f"Validation Acc: {val_acc:.2f}%")

    if epoch >= swa_start:
        swa_model.update_parameters(net)

# ===================
# 8. Kaggle Test Dataset
# ===================
test_dir = "/kaggle/input/cifar-dataset/test"
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
# 9. Predictions & Submission
# ===================
swa_model.eval()
all_preds, all_ids = [], []
with torch.no_grad():
    for inputs, ids in tqdm(test_loader, desc="Predicting"):
        inputs = inputs.to(device)
        outputs = swa_model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_ids.extend(ids)

submission = pd.DataFrame({
    "id": all_ids,
    "label": [class_names[p] for p in all_preds]
})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("✅ Submission file saved: /kaggle/working/submission.csv")

