# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

def load_images_from_directory(directory):
    image_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            img = Image.open(image_path)
            img_array = np.array(img)
            image_list.append(img_array)
    return np.array(image_list)

filtered_images_folder = "/kaggle/input/up-solving-tst-day-1/train/train/filtered_images"
real_images_folder = "/kaggle/input/up-solving-tst-day-1/train/train/real_images"

test_images_folder = "/kaggle/input/up-solving-tst-day-1/test/test/filtered_images"

filtered_images = load_images_from_directory(filtered_images_folder)
test_images = load_images_from_directory(test_images_folder)
real_images = load_images_from_directory(real_images_folder)

print(f"Filtered Images Shape: {filtered_images.shape}")
print(f"Real Images Shape: {real_images.shape}")


from scipy.ndimage import convolve
from scipy import ndimage
import cv2 as cv
import matplotlib.pyplot as plt

def restore_image_from_filtered(filtered_image):
    """
    Restore the original image from the filtered image using the provided kernel.
    
    Args:
    filtered_image: The filtered image (H, W, 3)
    kernel: The kernel (2x2) that describes the downsampling pattern.
    
    Returns:
    restored_image: The restored original image.
    """
    
    # Prepare an empty array for the restored image
    restored_image = np.zeros_like(filtered_image)

    # For each channel (R, G, B), we apply the restoration process
    for channel in range(3):
        img = filtered_image[:, :, channel]
        
        for i in range(0, 128, 2):
            for j in range(0, 128, 2):
                median = np.max(img[i:i+2, j:j+2].flatten())
                # if median < 10:    
                #     median = np.mean(img[i:i+2, j:j+2].flatten())
                for l in range(2):
                    for m in range(2):
                        if img[i + l, j + m] == 0:
                            restored_image[i + l, j + m, channel] = median
                            # print(median)
                        else:
                            restored_image[i + l, j + m, channel] = img[i + l, j + m]
               
        # restored_image[:, :, channel] = cv2.equalizeHist(restored_image[:, :, channel])

        # restored_image[:, :, channel] = cv.blur(restored_image[:, :, channel], (2, 2))
        # restored_image[:, :, channel] = ndimage.median_filter(restored_image[:, :, channel], size=2)
        restored_image[:, :, channel] = ndimage.gaussian_filter(restored_image[:, :, channel], sigma=0.7)


                
    return restored_image

def separate_and_plot_channels(img):
    # Separate the RGB channels
    r_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 2]
    
    # Plot the original image and its channels
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    
    # Display original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Display Red channel
    axes[1].imshow(r_channel, cmap='Reds')
    axes[1].set_title("Red Channel")
    axes[1].axis('off')
    
    # Display Green channel
    axes[2].imshow(g_channel, cmap='Greens')
    axes[2].set_title("Green Channel")
    axes[2].axis('off')
    
    # Display Blue channel
    axes[3].imshow(b_channel, cmap='Blues')
    axes[3].set_title("Blue Channel")
    axes[3].axis('off')
    
    # Show the plots
    plt.show()

idx = 100

sample_img = filtered_images[idx]
original_img = real_images[idx]
restored_img = restore_image_from_filtered(sample_img)

separate_and_plot_channels(sample_img)
separate_and_plot_channels(restored_img)
separate_and_plot_channels(original_img)

from skimage.metrics import peak_signal_noise_ratio as psnr

psnr_value = psnr(original_img, restored_img)
psnr_value


X_train = []
y_train = []

for fimg in filtered_images:
    X_train.append(restore_image_from_filtered(fimg))

for rimg in real_images:
    y_train.append(rimg)


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class Img(Dataset):
    def __init__(self, X, y):
        self.input_images = X
        self.target_images = y

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = self.input_images[idx]
        input_image = torch.tensor(input_image).permute(2, 0, 1).float()
        target_image = self.target_images[idx]
        target_image = torch.tensor(target_image).permute(2, 0, 1).float()
        return input_image, target_image


train_dataset = Img(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Downsampling path
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Upsampling path
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Down
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))

        # Bottleneck
        bn = self.bottleneck(self.pool3(d3))

        # Up
        u3 = self.up3(bn)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.up_conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up_conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up_conv1(u1)

        return self.final_conv(u1)


model = UNet()


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device


import torch.nn.functional as F

class PSNRLoss(nn.Module):
    def __init__(self, max_pixel_value=255.0, eps=1e-8):
        """
        Args:
            max_pixel_value: Maximum possible pixel value of the images.
                             For normalized images it's 1.0, for 8-bit images it's 255.
            eps: Small value to avoid division by zero or log(0).
        """
        super(PSNRLoss, self).__init__()
        self.max_pixel_value = max_pixel_value
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image tensor (B, C, H, W)
            target: Ground truth image tensor (B, C, H, W)
        Returns:
            PSNR loss value (scalar)
        """
        mse = F.mse_loss(pred, target, reduction='mean')
        psnr = 20 * torch.log10(self.max_pixel_value / torch.sqrt(mse + self.eps))
        return -psnr  # Negative because we want to maximize PSNR, but losses are minimized



model = model.to(device)

loss_fn = PSNRLoss()
optimizer = torch.optim.Adam(model.parameters())

size = len(train_dataloader.dataset)
# Set the model to training mode - important for batch normalization and dropout layers
# Unnecessary in this situation but added for best practices
model.train()

epochs = 30

for epoch in range(epochs):
    print(f'Epochs {epoch}')
    for batch, (X, y) in enumerate(train_dataloader):
        X = X.to(device)
        y = y.to(device)
        # plt.imshow(X.to('cpu')[0].permute(1, 2, 0))
        # plt.imshow(y.to('cpu')[0].permute(1, 2, 0))

        pred = model(X)
        loss = loss_fn(pred, y)
    
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
        if batch % 100 == 0:
            loss, current = loss.item(), batch * 64 + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


model.eval()

model = model.to('cpu')

with torch.no_grad():
    sample_img = torch.tensor(restore_image_from_filtered(filtered_images[-1])).permute(2, 0, 1).float().unsqueeze(0)

    output_img = model(sample_img)
    output_img = output_img[0].permute(1, 2, 0).int()
    print(output_img.shape)

plt.imshow(output_img)


plt.imshow(restore_image_from_filtered(real_images[-1]))


psnr_value = psnr(real_images[-1], np.array(output_img))
psnr_value


plt.imshow(restore_image_from_filtered(real_images[-2]))


train_dir = Path("/kaggle/input/up-solving-tst-day-1/train/train")
test_dir = Path("/kaggle/input/up-solving-tst-day-1/test/test")
test_filtered_dir = test_dir / "filtered_images"

real_dir = train_dir / "real_images"
filtered_dir = train_dir / "filtered_images"

test_filtered_images = test_filtered_dir.glob("*.png")

result = []
for image_path in (test_filtered_images):
    image = cv2.imread(str(image_path))
    filename = os.path.basename(image_path)
    id_ = int(os.path.splitext(filename)[0])

    image = restore_image_from_filtered(image)

    image = torch.tensor(image).permute(2, 0, 1).float()

    with torch.no_grad():
        image = model(image)
        image = image.permute(1, 2, 0).int()
        

    flat = image.flatten()
    row = [id_] + flat.tolist() 

    result.append(row)

df = pd.DataFrame(result)
df.columns = ["id"] + [f"pixel_{i}" for i in range(df.shape[1] - 1)]

df.to_csv("submission.csv", index=False)




