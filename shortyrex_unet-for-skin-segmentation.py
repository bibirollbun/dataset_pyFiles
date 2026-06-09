import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
!pip install torchviz
from torchviz import make_dot
import cv2
import torch.nn as nn
import torch.nn.functional as F
!pip install -U albumentations


import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ISICDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Match mask name
        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.masks_dir, mask_name)

        # Load image and mask using cv2 for Albumentations
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize raw image to match the transform size
        raw_image = cv2.resize(image, (128, 128))
        raw_image = torch.tensor(raw_image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Apply Albumentations transform
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        # Convert mask to binary (0 and 1) and add channel dimension
        mask = (mask > 0).float().unsqueeze(0)
        
        return image, mask, raw_image



import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(128,128),
    A.HorizontalFlip(p=0.5),
    A.ElasticTransform(alpha=120, sigma=6.0, p=1.0),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
    A.OpticalDistortion(distort_limit=2.0, p=1.0),
    A.Normalize(
        mean=(0.72183126, 0.61805737, 0.5664776),
        std=(0.1031459,  0.12656619, 0.14789347)
    ),
    ToTensorV2()
])




data_dir = "/kaggle/input/a0-2025-medical-image-segmentation/Dataset/Train"

images_dir = f"{data_dir}/Image"
masks_dir = f"{data_dir}/Mask"

# Initialize dataset
dataset = ISICDataset(images_dir=images_dir, masks_dir=masks_dir, transform=train_transform)

# Initialize dataloader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
# Get images and masks batch shapes
images, masks, raw= next(iter(dataloader))
print(f"Image batch shape: {images.shape}")
print(f"Mask batch shape: {masks.shape}")
print(f"Raw batch shape: {raw.shape}")
# # Calculate mean and std
# mean = 0.0
# std = 0.0
# nb_samples = 0.0

# for data, _, _ in dataloader:
#     batch_samples = data.size(0)
#     data = data.view(batch_samples, data.size(1), -1)
#     mean += data.mean(2).sum(0)
#     std += data.std(2).sum(0)
#     nb_samples += batch_samples

# mean /= nb_samples
# std /= nb_samples

# print("Mean:", mean.numpy())
# print("Std:", std.numpy())



# Transform for images
image_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# Initialize dataset
dataset = ISICDataset(images_dir=images_dir, masks_dir=masks_dir, transform = image_transform)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)





import matplotlib.pyplot as plt

# function that displays batch of images and masks
def show_batch(images, masks, raw):
    fig, ax = plt.subplots(3, 8, figsize=(20, 7))
    for i in range(8):
        ax[0][i].imshow(images[i].permute(1, 2, 0).clamp(0, 1))
        ax[0][i].set_title("Normalized Image")
        ax[0][i].axis("off")
        ax[1][i].imshow(raw[i].permute(1, 2, 0))
        ax[1][i].set_title("Image")
        ax[1][i].axis("off")
        ax[2][i].imshow(masks[i][0], cmap="gray")
        ax[2][i].set_title("Mask")
        ax[2][i].axis("off")
    plt.show()

show_batch(images, masks, raw)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder: Downsampling path
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder: Upsampling path
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        # Output layer
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Output layer
        output = self.output_conv(dec1)
        return output



# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # -------------------- DoubleConv --------------------
# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# # -------------------- Down (Encoder block) --------------------
# class Down(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Down, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.MaxPool2d(2, 2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.encoder(x)


# # -------------------- Up (Decoder block) --------------------
# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super(Up, self).__init__()

#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

#         self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x1, x2):
#         # Upsample
#         x1 = self.up(x1)

#         # Handle size mismatch (for odd input dimensions)
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2))

#         # Concatenate encoder feature maps with upsampled output
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# # -------------------- Out --------------------
# class Out(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Out, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)


# # -------------------- Final U-Net (UNETR style) --------------------
# class UNetModular(nn.Module):
#     def __init__(self, in_channels=3, out_channels=1, n_channels=64, bilinear=True):
#         super(UNetModular, self).__init__()

#         # Encoder
#         self.inc = DoubleConv(in_channels, n_channels)
#         self.down1 = Down(n_channels, 2 * n_channels)
#         self.down2 = Down(2 * n_channels, 4 * n_channels)
#         self.down3 = Down(4 * n_channels, 8 * n_channels)
#         self.down4 = Down(8 * n_channels, 16 * n_channels)

#         # Decoder
#         self.up1 = Up(16 * n_channels + 8 * n_channels, 8 * n_channels, bilinear)
#         self.up2 = Up(8 * n_channels + 4 * n_channels, 4 * n_channels, bilinear)
#         self.up3 = Up(4 * n_channels + 2 * n_channels, 2 * n_channels, bilinear)
#         self.up4 = Up(2 * n_channels + n_channels, n_channels, bilinear)

#         # Output
#         self.outc = Out(n_channels, out_channels)

#     def forward(self, x):
#         # Encoder path
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)

#         # Decoder path
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)

#         # Final segmentation output
#         return self.outc(x)



model = UNet(in_channels=3, out_channels=1)

# Create a dummy input tensor and pass it through the model
input_tensor = torch.randn(64, 3, 128, 128)
output = model(input_tensor)

print(f"Output shape: {output.shape}")


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Instantiate and run one forward pass
model = UNet(in_channels=3, out_channels=1)
x = torch.randn(1, 3, 64, 64)  # dummy input
y = model(x)

# Visualize computation graph
make_dot(y, params=dict(model.named_parameters())).render("unet_architecture", format="png")

img = mpimg.imread("unet_architecture.png")
plt.figure(figsize=(12, 25))
plt.imshow(img)
plt.axis('off')  # hide axes
plt.show()


from torch.utils.data import random_split, DataLoader

train_images_dir = f"{data_dir}/Image"
train_masks_dir = f"{data_dir}/Mask"

full_dataset = ISICDataset(images_dir=train_images_dir, masks_dir=train_masks_dir,
                           transform=train_transform)

val_ratio = 0.2
val_size = int(val_ratio * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.3, gamma=0.6, eps=1e-8):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        
        # Verify weights sum to 1
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, 
    
    def forward(self, pred, target):
        # Ensure target is in the same format as pred
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # Apply sigmoid to predictions if not already applied
        pred = torch.sigmoid(pred)
        
        # Calculate individual losses
        bce_loss = self.bce_loss(pred, target)
        boundary_loss = self.active_boundary_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        
        # Combine losses with weights
        total_loss = (self.alpha * bce_loss + 
                     self.beta * boundary_loss + 
                     self.gamma * dice_loss)
        
        return total_loss
    
    def bce_loss(self, pred, target):
        # Use BCE with logits for numerical stability, but since we already applied sigmoid, use regular BCE
        return F.binary_cross_entropy(pred, target)
    
    def dice_loss(self, pred, target):
        # Flatten predictions and targets
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)
        
        # Dice coefficient
        dice = (2. * intersection + self.eps) / (union + self.eps)
        
        # Return 1 - Dice (loss)
        return 1 - dice.mean()
    
    def active_boundary_loss(self, pred, target):
        # Calculate gradients to identify boundaries
        pred_grad_x = self._compute_gradient_x(pred)
        pred_grad_y = self._compute_gradient_y(pred)
        target_grad_x = self._compute_gradient_x(target)
        target_grad_y = self._compute_gradient_y(target)
        
        # Magnitude of gradients
        pred_grad_magnitude = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + self.eps)
        target_grad_magnitude = torch.sqrt(target_grad_x**2 + target_grad_y**2 + self.eps)
        
        # Boundary loss - encourage alignment of boundaries
        boundary_loss = F.l1_loss(pred_grad_magnitude, target_grad_magnitude)
        
        return boundary_loss
    
    def _compute_gradient_x(self, img):
        sobel_x = torch.tensor([[-1, 0, 1], 
                               [-2, 0, 2], 
                               [-1, 0, 1]], dtype=torch.float32, device=img.device)
        sobel_x = sobel_x.view(1, 1, 3, 3)
        return F.conv2d(img, sobel_x, padding=1)
    
    def _compute_gradient_y(self, img):
        sobel_y = torch.tensor([[-1, -2, -1], 
                               [0, 0, 0], 
                               [1, 2, 1]], dtype=torch.float32, device=img.device)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        return F.conv2d(img, sobel_y, padding=1)

# Alternative version with better numerical stability
class CombinedLossStable(nn.Module):
    def __init__(self, alpha=0.1, beta=0.3, gamma=0.6, eps=1e-8):
        super(CombinedLossStable, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        
        # Verify weights sum to 1
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights should sum to 1"
    
    def forward(self, pred, target):
        # Ensure target is in the same format as pred
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # Calculate individual losses (pred assumed to be logits)
        bce_loss = self.bce_loss(pred, target)
        boundary_loss = self.active_boundary_loss(torch.sigmoid(pred), target)
        dice_loss = self.dice_loss(torch.sigmoid(pred), target)
        
        # Combine losses with weights
        total_loss = (self.alpha * bce_loss + 
                     self.beta * boundary_loss + 
                     self.gamma * dice_loss)
        
        return total_loss
    
    def bce_loss(self, pred, target):
        # Use BCE with logits for better numerical stability
        return F.binary_cross_entropy_with_logits(pred, target)
    
    def dice_loss(self, pred, target):
        # Flatten predictions and targets
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)
        
        # Dice coefficient
        dice = (2. * intersection + self.eps) / (union + self.eps)
        
        # Return 1 - Dice (loss)
        return 1 - dice.mean()
    
    def active_boundary_loss(self, pred, target):
        # Calculate gradients to identify boundaries
        pred_grad_x = self._compute_gradient_x(pred)
        pred_grad_y = self._compute_gradient_y(pred)
        target_grad_x = self._compute_gradient_x(target)
        target_grad_y = self._compute_gradient_y(target)
        
        # Magnitude of gradients
        pred_grad_magnitude = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + self.eps)
        target_grad_magnitude = torch.sqrt(target_grad_x**2 + target_grad_y**2 + self.eps)
        
        # Boundary loss - encourage alignment of boundaries
        boundary_loss = F.l1_loss(pred_grad_magnitude, target_grad_magnitude)
        
        return boundary_loss
    
    def _compute_gradient_x(self, img):
        sobel_x = torch.tensor([[-1, 0, 1], 
                               [-2, 0, 2], 
                               [-1, 0, 1]], dtype=torch.float32, device=img.device)
        sobel_x = sobel_x.view(1, 1, 3, 3)
        return F.conv2d(img, sobel_x, padding=1)
    
    def _compute_gradient_y(self, img):
        sobel_y = torch.tensor([[-1, -2, -1], 
                               [0, 0, 0], 
                               [1, 2, 1]], dtype=torch.float32, device=img.device)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        return F.conv2d(img, sobel_y, padding=1)



import torch.optim as optim

# Model, loss function, and optimizer
model = UNet(in_channels=3, out_channels=1)
criterion = CombinedLossStable(alpha=0.1, beta=0.3, gamma=0.6) 
optimizer = optim.AdamW(model.parameters(), lr=0.001)


import os

print("Files in /kaggle/working/:")
print(os.listdir("/kaggle/working/"))



checkpoint = torch.load("/kaggle/input/unet/pytorch/default/1", map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
training_losses = checkpoint['training_losses']
validation_losses = checkpoint['validation_losses']


model.to(device)

num_epochs = 100
training_losses = []
validation_losses = []
best_val_loss = float("inf")  # Initialize before the loop

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for images, masks, _ in train_loader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks, _ in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save if this is the best model so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_losses': training_losses,
            'validation_losses': validation_losses
        }, "/kaggle/working/best_model_unet_300.pth")
        print(f"Saved new best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")

    # Append losses for visualization later
    training_losses.append(avg_train_loss)
    validation_losses.append(avg_val_loss)



plt.figsize=(10, 5)
plt.plot(training_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


def dice_coefficient(outputs, masks, threshold=0.5, epsilon=1e-6):
    # Convert logits/probabilities to binary mask
    outputs = (outputs > threshold).float()
    
    outputs = outputs.view(-1)
    masks = masks.view(-1)

    intersection = (outputs * masks).sum()
    dice = (2. * intersection + epsilon) / (outputs.sum() + masks.sum() + epsilon)
    return dice


def intersection_over_union(outputs, masks, threshold=0.5, epsilon=1e-6):
    outputs = (outputs > threshold).float()
    
    outputs = outputs.view(-1)
    masks = masks.view(-1)

    intersection = (outputs * masks).sum()
    union = outputs.sum() + masks.sum() - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    return iou


test_dice, test_iou = 0.0, 0.0
checkpoint = torch.load("best_model_unet_300.pth", map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    for images, masks, _ in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        outputs = torch.sigmoid(outputs)   # convert to probabilities

        test_dice += dice_coefficient(outputs, masks).item()
        test_iou += intersection_over_union(outputs, masks).item()

avg_dice = test_dice / len(val_loader)
avg_iou = test_iou / len(val_loader)
print(f"Average Dice Coefficient: {avg_dice:.4f}")
print(f"Average Intersection over Union: {avg_iou:.4f}")


torch.save(model.state_dict(), "/kaggle/working/unet_model.pth")




