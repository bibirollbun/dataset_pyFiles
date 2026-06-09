# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
cnt = 0
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        # print(os.path.join(dirname, filename))
        cnt += 1
print(cnt)
        

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import zipfile
import os

def unzip_file(zip_path, extract_to="."):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Unzip training images and masks
unzip_file("/kaggle/input/carvana-image-masking-challenge/train.zip", "/kaggle/working/train/")
unzip_file("/kaggle/input/carvana-image-masking-challenge/train_masks.zip", "/kaggle/working/train_masks/")


from torchvision import transforms

# Paths after unzip
img_dir = '/kaggle/working/train/train/'
mask_dir = '/kaggle/working/train_masks/train_masks/'

# Resize values
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# Transformations
transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor()
])

target_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor()
])



from torch.utils.data import Dataset
from PIL import Image
import os
from torch.utils.data import DataLoader


class CarvanaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform

        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        mask_name = img_name.replace(".jpg", "_mask.gif")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask



transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

target_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = CarvanaDataset(img_dir, mask_dir, transform=transform, target_transform=target_transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Optional sanity check
img, msk = next(iter(train_loader))
print("Image batch shape:", img.shape)
print("Mask batch shape:", msk.shape)



import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = CBR(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.upconv4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.upconv3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out(d1))



import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        masks = masks.float()  # for BCELoss

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")



import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    sample_img, sample_mask = next(iter(train_loader))
    sample_img = sample_img.to(device)
    pred_mask = model(sample_img)
    pred_mask = pred_mask.cpu()

for i in range(3):  # Showing & saving 3 examples
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[0].imshow(sample_img[i].permute(1, 2, 0).cpu())
    ax[0].set_title("Input")
    ax[1].imshow(sample_mask[i][0], cmap='gray')
    ax[1].set_title("True Mask")
    ax[2].imshow(pred_mask[i][0], cmap='gray')
    ax[2].set_title("Predicted Mask")
    for a in ax:
        a.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"carvana_result_{i}.png")  # Save image
    plt.show()



torch.save(model.state_dict(), "carvana_segmentation_model.pth")




