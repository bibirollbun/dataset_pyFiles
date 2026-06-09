import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


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


filters = []
for x in ['R','G','B']:
    for y in ['R','G','B']:
        for z in ['R','G','B']:
            if x != y and y != z and x != z:
                filters.append([x,y,z,x])
                filters.append([y,x,x,z])


def apply_fast_filter(img, pattern):
    new_pattern = []
    for i in pattern:
        new_pattern.append(['R','G','B'].index(i))
        
    filtered_img = np.zeros_like(img)
    # Верхний левый
    filtered_img[0::2, 0::2, new_pattern[0]] = img[0::2, 0::2, new_pattern[0]]
    # Верхний правый
    filtered_img[0::2, 1::2, new_pattern[1]] = img[0::2, 1::2, new_pattern[1]]
    # Нижний левый
    filtered_img[1::2, 0::2, new_pattern[2]] = img[1::2, 0::2, new_pattern[2]]
    # Нижний правый
    filtered_img[1::2, 1::2, new_pattern[3]] = img[1::2, 1::2, new_pattern[3]]

    return filtered_img


# from multiprocessing import Pool, cpu_count
# cpu_count()


# def apply_12_filters(rimg):
#     # X_train = []
#     # for rimg in tqdm(real_images):
#     X_train = []
#     for filter_ in filters:
#         fimg = apply_fast_filter(rimg, filter_)
#         X_train.append(restore_image_from_filtered(fimg))
#     return X_train

# with Pool(processes=cpu_count()) as p:
#     res = p.map(apply_12_filters, real_images)
# X_train = np.stack(res).reshape(-1, 128, 128, 3)
# np.save('X_train.npy', X_train)


from tqdm.auto import tqdm
X_train = []
y_train = []

X_train = np.load('/kaggle/input/ioai-prep-image-restoring-augs/X_train.npy')

for rimg in real_images:
    for _ in filters:
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """
    U-Net neural network for image segmentation.

    Args:
        in_channels (int): Number of input channels (e.g., 16 for your patches)

    The U-Net consists of an encoder (downsampling), a bottleneck (middle), 
    and a decoder (upsampling). It is widely used for segmenting images, 
    where every pixel needs to be classified (e.g., rain/no rain).
    """
    def __init__(self):
        super().__init__()
        # Encoder path ("contracting" path): extracts features, reduces size
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)  # Downsamples by factor of 2

        # Bottleneck (middle part)
        self.mid = self.conv_block(512, 1024)

        # Decoder path ("expanding" path): upscales, combines with encoder outputs
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)  # Upsample
        self.dec4 = self.conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Final layer: reduces to 1 output channel per pixel (for binary segmentation)
        self.final = nn.Conv2d(64, 3, kernel_size=1)  # Output: logits per pixel

    def conv_block(self, in_ch, out_ch):
        """
        Helper function to build a block of two convolutional layers, 
        each followed by BatchNorm and ReLU activation.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # Keeps spatial size the same
            nn.BatchNorm2d(out_ch),                   # Helps training
            nn.GELU(),                    # Non-linearity
            nn.Conv2d(out_ch, out_ch, 3, padding=1),  # Another conv layer
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, height, width]

        Returns:
            torch.Tensor: Output logits, shape [batch, 1, height, width]
        """
        # Encoder: save outputs for skip connections
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        m = self.mid(self.pool(e4))

        # Decoder: upsample, concatenate with encoder outputs (skip connections), then convolve
        d4 = self.dec4(torch.cat([self.up4(m), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)
        
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
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

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
        # break
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

# model = model.to('cpu')

with torch.no_grad():
    sample_img = torch.tensor(restore_image_from_filtered(filtered_images[-1])).permute(2, 0, 1).float()
    output_img = model(sample_img.unsqueeze(0).cuda()).squeeze(0).cpu()
    output_img = output_img.permute(1, 2, 0).int()
    print(output_img.shape)

plt.imshow(output_img)


# plt.imshow(restore_image_from_filtered(real_images[-1]))


# psnr_value = psnr(real_images[-2], np.array(output_img))
# psnr_value


# plt.imshow(restore_image_from_filtered(real_images[-2]))


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
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = model(image.cuda()).cpu()
        if len(image.shape) == 4:
            image = image.squeeze(0)
        image = image.permute(1, 2, 0).int()
        

    flat = image.flatten()
    row = [id_] + flat.tolist() 

    result.append(row)

df = pd.DataFrame(result)
df.columns = ["id"] + [f"pixel_{i}" for i in range(df.shape[1] - 1)]

df.to_csv("submission.csv", index=False)




