import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from tqdm import tqdm



def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================
# CUSTOM DATASET
# ========================
class GenData(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '_mask.gif'))

        # Check if mask file exists
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for image: {img_name}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)

        mask = transforms.Resize((224, 224))(mask)
        mask = transforms.ToTensor()(mask)
        mask = (mask > 0.5).long().squeeze(0)  # Binary mask for 2 classes

        return image, mask



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



dataset = GenData(
    image_dir="/kaggle/working/carvana/train/train",
    mask_dir="/kaggle/working/carvana/train_masks/train_masks",
    transform=transform
)

print(f"Total samples in dataset: {len(dataset)}")
print(f"Sample filenames: {dataset.image_files[:3]}")

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        conv = self.block(x)
        pooled = self.pool(conv)
        return conv, pooled

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, in_channels=3, out_classes=2):
        super().__init__()
        self.d1 = DownSample(in_channels, 64)
        self.d2 = DownSample(64, 128)
        self.d3 = DownSample(128, 256)
        self.d4 = DownSample(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.u1 = UpSample(1024, 512)
        self.u2 = UpSample(512, 256)
        self.u3 = UpSample(256, 128)
        self.u4 = UpSample(128, 64)

        self.final = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        c1, p1 = self.d1(x)
        c2, p2 = self.d2(p1)
        c3, p3 = self.d3(p2)
        c4, p4 = self.d4(p3)

        b = self.bottleneck(p4)

        u1 = self.u1(b, c4)
        u2 = self.u2(u1, c3)
        u3 = self.u3(u2, c2)
        u4 = self.u4(u3, c1)

        return self.final(u4)



model = UNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

epochs = 20
train_loss, val_loss = [], []
train_acc, val_acc = [], []


for epoch in tqdm(range(epochs), desc='Training'):
    model.train()
    running_loss = 0
    correct, total = 0, 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)

        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(output, dim=1)
        correct += (preds == y_batch).sum().item()
        total += torch.numel(y_batch)

    train_loss.append(running_loss / len(train_loader))
    train_acc.append(correct / total)

    # Validation
    model.eval()
    val_running_loss = 0
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            val_output = model(x_val)
            val_loss_i = criterion(val_output, y_val)
            val_running_loss += val_loss_i.item()

            val_preds = torch.argmax(val_output, dim=1)
            val_correct += (val_preds == y_val).sum().item()
            val_total += torch.numel(y_val)

    val_loss.append(val_running_loss / len(val_loader))
    val_acc.append(val_correct / val_total)

    scheduler.step()

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"| Train Loss: {train_loss[-1]:.4f}, Acc: {train_acc[-1]*100:.2f}% "
          f"| Val Loss: {val_loss[-1]:.4f}, Acc: {val_acc[-1]*100:.2f}%")


plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(train_acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")1
plt.legend()
plt.show()

