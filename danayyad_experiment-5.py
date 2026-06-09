

import cv2
import torch
from typing import Dict
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import torch.nn as nn3
from sklearn.model_selection import train_test_split
from torchvision.models import resnet50
import pandas as pd
import os
from torch.cuda.amp import GradScaler, autocast
import glob
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
import torch.optim as optim



class MyDataset(Dataset):
    def __init__(self, meta, image_size=(512, 512), augment=False):
        self.meta = meta
        self.image_size = image_size
        self.augment = augment
        self.transforms = T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(90),
        #.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor()
    ]) if augment else None


    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        f = self.meta[index]
        if f is None:
            raise ValueError("File path not found in metadata")

        # Load image
        image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.image_size)
        image = torch.from_numpy(image).float().unsqueeze(0)  # Convert to tensor and add channel dimension
        image = norm_by_percentile(image)

        # Load mask
        maskfile = f.replace('images', 'labels')  # Replace 'images' with 'labels' in the file path
        if os.path.exists(maskfile):
            mask = cv2.imread(maskfile, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.image_size)
            mask = torch.from_numpy(mask).float().unsqueeze(0)  # Convert to tensor and add channel dimension
            mask = norm_by_percentile(mask)
        else:
            mask = torch.zeros_like(image)  # Create a zero mask with same shape as image

        if self.augment:
            # Apply the same transformation to both image and mask
            seed = torch.random.seed()  # Set the seed for reproducibility
            torch.random.manual_seed(seed)
            image = self.transforms(image)
            torch.random.manual_seed(seed)
            mask = self.transforms(mask)

        return image, mask




class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, attr):
        return self.get(attr)

    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__


def show_images(images, masks, n=5):
    """
    Display a batch of images and their corresponding masks.

    Args:
    - images (torch.Tensor): Batch of images.
    - masks (torch.Tensor): Batch of masks.
    - n (int): Number of images to display. Default is 5.
    """
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()

    batch_size = images.shape[0]
    n = min(n, batch_size)  # Ensure n does not exceed batch size

    plt.figure(figsize=(15, 10))
    for i in range(n):
        plt.subplot(n, 2, 2 * i + 1)
        plt.imshow(images[i, 0], cmap='gray')
        plt.title('Image')
        plt.axis('off')

        plt.subplot(n, 2, 2 * i + 2)
        plt.imshow(masks[i, 0], cmap='gray')
        plt.title('Mask')
        plt.axis('off')
    plt.show()


def save_checkpoint(epoch, model, optimizer, loss, file_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, file_path)

def load_checkpoint(file_path, model, optimizer):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, model, optimizer, loss


def norm_by_percentile(x, low=10, high=99.8, alpha=0.01):
    xmin = np.percentile(x, low)
    xmax = np.percentile(x, high)
    x = (x - xmin) / (xmax - xmin + 1e-5)  # Avoid division by zero
    x = np.clip(x, 0, 1)  # Ensure values are in the range [0, 1]
    if 1:
        x[x > 1] = (x[x > 1] - 1) * alpha + 1
        x[x < 0] = x[x < 0] * alpha
    return x
    #return x / 65536.0


class Net(nn.Module):
    def __init__(self, weights='IMAGENET1K_V1'):
        super(Net, self).__init__()

        self.backbone = models.resnet50(weights=weights)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.encoder = nn.Sequential(*list(self.backbone.children())[:-2])

        self.upconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(2048 + 1024, 1024, kernel_size=3, padding=1)

        self.upconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(1024 + 512, 512, kernel_size=3, padding=1)

        self.upconv3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1)

        self.upconv4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = nn.Conv2d(256 + 64, 64, kernel_size=3, padding=1)

        self.upconv5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.vessel_out = nn.Conv2d(32, 1, kernel_size=1)

        # Apply dropout in the model definition
        self.dropout = nn.Dropout(p=0.3)  # Add dropout layer

    def forward(self, x):
        # Encoder
        enc1 = self.backbone.conv1(x)
        enc1 = self.backbone.bn1(enc1)
        enc1 = self.backbone.relu(enc1)
        enc1 = self.backbone.maxpool(enc1)

        enc2 = self.backbone.layer1(enc1)
        enc3 = self.backbone.layer2(enc2)
        enc4 = self.backbone.layer3(enc3)
        enc5 = self.backbone.layer4(enc4)

        # Decoder with debugging prints
        dec1 = self.upconv1(enc5)
        if dec1.size()[2:] != enc4.size()[2:]:
            dec1 = F.interpolate(dec1, size=enc4.shape[2:], mode='bilinear', align_corners=True)
        dec1 = torch.cat((dec1, enc4), dim=1)
        dec1 = self.conv1(dec1)
        dec1 = self.dropout(dec1)  # Apply dropout before passing through a layer

        dec2 = self.upconv2(dec1)
        if dec2.size()[2:] != enc3.size()[2:]:
            dec2 = F.interpolate(dec2, size=enc3.shape[2:], mode='bilinear', align_corners=True)
        dec2 = torch.cat((dec2, enc3), dim=1)
        dec2 = self.conv2(dec2)
        dec2 = self.dropout(dec2)  # Apply dropout before passing through a layer

        dec3 = self.upconv3(dec2)
        if dec3.size()[2:] != enc2.size()[2:]:
            dec3 = F.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=True)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.conv3(dec3)
        dec3 = self.dropout(dec3)  # Apply dropout before passing through a layer

        dec4 = self.upconv4(dec3)
        if dec4.size()[2:] != enc1.size()[2:]:
            dec4 = F.interpolate(dec4, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        dec4 = torch.cat((dec4, enc1), dim=1)
        dec4 = self.conv4(dec4)
        dec4 = self.dropout(dec4)  # Apply dropout before passing through a layer

        dec5 = self.upconv5(dec4)
        if dec5.size()[2:] != x.size()[2:]:
            dec5 = F.interpolate(dec5, size=x.shape[2:], mode='bilinear', align_corners=True)
        dec5 = self.conv5(dec5)
        dec5 = self.dropout(dec5)  # Apply dropout before passing through a layer

        vessel = self.vessel_out(dec5)

        return vessel




def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def train(net, train_loader, val_loader, criterion, optimizer, start_epoch=0, total_epochs=10,
          accumulation_steps=4, gradient_clip=1.0, checkpoint_dir='/kaggle/working/',
          scheduler=None, min_delta=0.001, patience=4):
   
    net.train().cuda()
    scaler = GradScaler()
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, total_epochs):
        net.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.cuda(), masks.cuda()

            with autocast():
                vessel_pred = net(images)
                loss = criterion(vessel_pred, masks)
           
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        val_loss = validate(net, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch + 1}/{total_epochs}], Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')

        # Save best model
        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
        else:
            early_stop_counter += 1
            print(f"EarlyStopping counter: {early_stop_counter} / {patience}")

        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        if scheduler:
            scheduler.step(val_loss)

        if early_stop_counter >= patience:
            print(f"⏹️ Early stopping triggered at epoch {epoch + 1}")
            break

    plot_loss(train_losses, val_losses)
    print("✅ Training Complete")

def validate(net, val_loader, criterion):
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.cuda(), masks.cuda()
            vessel_pred = net(images)
            loss = criterion(vessel_pred, masks)
            val_loss += loss.item()
    return val_loss / len(val_loader)


pip install torchmetrics




# Load the model and optimizer
import time
net = Net(weights='IMAGENET1K_V1').cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=1e-5)

total_epochs = 75
checkpoint_path = "/kaggle/input/exp5-59checkpoints/exp5_checkpoint_epoch_50.pth"

# Load checkpoint if available
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']  # Resume from saved epoch
    print(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 0  # Start from scratch if no checkpoint is found



# Collect all image files from the training directory
train_files = []
patterns = [
   
    '/kaggle/input/kidney-3-upgraded/kidney_3_labels_combined/images/*.tif',
    '/kaggle/input/blood-vessel-segmentation/train/kidney_1_dense/images/*.tif',
    '/kaggle/input/blood-vessel-segmentation/train/kidney_1_voi/images/*.tif'
]

for pattern in patterns:
    train_files.extend(glob.glob(pattern, recursive=True))

train_meta = [
    DotDict(
        file=train_files,
        shape=(len(train_files), 512, 512),
        id=[os.path.splitext(os.path.basename(f))[0] for f in train_files]
    )
]

# Split the dataset into training and validation sets
train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)

# Prepare training DataLoader with augmentation
train_dataset = MyDataset(train_files, augment=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Prepare validation DataLoader without augmentation
val_dataset = MyDataset(val_files)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)




#Learning Rate Scheduling: Utilize advanced learning rate schedulers like OneCycleLR or CosineAnnealingLR.
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=total_epochs)


# Define ComboLoss using dice_score from torchmetrics
class ComboLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5):
        super(ComboLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.bce_loss = nn.BCEWithLogitsLoss()

    def dice_loss(self, inputs, targets):
        inputs = inputs.sigmoid().view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + 1e-5) / (inputs.sum() + targets.sum() + 1e-5)
        return 1 - dice

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        return self.weight_dice * dice + self.weight_bce * bce



criterion = ComboLoss()

# Continue training for additional epochs
#train(net, train_loader, val_loader, criterion, optimizer, total_epochs=total_epochs, scheduler=scheduler)
# train(net, train_loader, val_loader, criterion, optimizer, start_epoch=start_epoch, total_epochs=total_epochs, scheduler=scheduler)
train(
    net,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    start_epoch=start_epoch,
    total_epochs=total_epochs,
    scheduler=scheduler,
    min_delta=0.001,
    patience=4
)


def evaluate_metrics(y_true, y_pred):
    y_true = np.nan_to_num(y_true)
    y_true = (y_true > 0.5).astype(int)
    y_pred = np.nan_to_num(y_pred)
    y_pred = (y_pred > 0.5).astype(int)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    iou = jaccard_score(y_true_flat, y_pred_flat, average='binary')
    dice = f1_score(y_true_flat, y_pred_flat, average='binary')
    precision = precision_score(y_true_flat, y_pred_flat, average='binary')
    recall = recall_score(y_true_flat, y_pred_flat, average='binary')

    # Calculate accuracy
    accuracy = np.mean(y_true_flat == y_pred_flat)

    return iou, dice, precision, recall, accuracy





def evaluate_model(net, test_loader, checkpoint_dir="/kaggle/working/"):
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval().cuda()

    all_vessel_preds = []
    all_vessel_masks = []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.cuda(), masks.cuda()
            vessel_pred = net(images)  # Only get vessel predictions
            vessel_pred = torch.sigmoid(vessel_pred).cpu().numpy()
            vessel_masks = masks.cpu().numpy()

            all_vessel_preds.append(vessel_pred)
            all_vessel_masks.append(vessel_masks)

    all_vessel_preds = np.concatenate(all_vessel_preds)
    all_vessel_masks = np.concatenate(all_vessel_masks)

    iou_vessel, dice_vessel, precision_vessel, recall_vessel, accuracy_vessel = evaluate_metrics(all_vessel_masks, all_vessel_preds)

    return {
        'iou_vessel': iou_vessel,
        'dice_vessel': dice_vessel,
        'precision_vessel': precision_vessel,
        'recall_vessel': recall_vessel,
        'accuracy_vessel': accuracy_vessel  # Include accuracy in the returned metrics
    }




# Collect all image files from the test directory
test_files = glob.glob('/kaggle/input/blood-vessel-segmentation/train/kidney_2/images/*.tif', recursive=True)
# Prepare test DataLoader without augmentation
test_files = test_files[:500]
test_dataset = MyDataset(test_files)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


metrics = evaluate_model(net, test_loader)
print(metrics)


# import torch
# from torch.utils.data import DataLoader
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# def get_sample_from_dataset(dataset, idx):
#     image_tensor, mask = dataset[idx]
#     return image_tensor.unsqueeze(0), mask.numpy()

# def integrated_gradients(inputs, model, target_class=0, baseline=None, steps=50, device='cuda'):
#     if baseline is None:
#         baseline = torch.zeros_like(inputs).to(device)
#     inputs = inputs.to(device)
#     baseline = baseline.to(device)

#     scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(steps + 1)]

#     grads = []
#     for scaled_input in scaled_inputs:
#         scaled_input.requires_grad = True
#         output = model(scaled_input)
#         score = output[0, target_class].mean()
#         model.zero_grad()
#         score.backward(retain_graph=True)
#         grads.append(scaled_input.grad.detach().cpu().numpy())

#     grads = np.array(grads)
#     avg_grads = (grads[:-1] + grads[1:]) / 2.0
#     avg_grads = avg_grads.mean(axis=0).squeeze()

#     integrated_grads = (inputs.cpu().numpy().squeeze() - baseline.cpu().numpy().squeeze()) * avg_grads

#     if integrated_grads.ndim == 3:
#         integrated_grads = integrated_grads[0]

#     integrated_grads = np.abs(integrated_grads)
#     integrated_grads -= integrated_grads.min()
#     integrated_grads /= integrated_grads.max() + 1e-8

#     # ✅ Apply gamma correction
#     gamma = 0.5
#     integrated_grads = integrated_grads ** gamma

#     return integrated_grads

# def save_visualizations(image_tensor, mask, attribution_map, idx, output_dir='output'):
#     image_np = image_tensor.squeeze().cpu().numpy()
#     mask_2d = mask.squeeze()

#     folder = os.path.join(output_dir, f"image_{idx}")
#     os.makedirs(folder, exist_ok=True)

#     # Save input image
#     plt.imshow(image_np, cmap='gray')
#     plt.axis('off')
#     plt.savefig(os.path.join(folder, f"image_{idx}_input.png"), bbox_inches='tight', pad_inches=0)
#     plt.close()

#     # Save mask image
#     plt.imshow(mask_2d, cmap='gray')
#     plt.axis('off')
#     plt.savefig(os.path.join(folder, f"image_{idx}_mask.png"), bbox_inches='tight', pad_inches=0)
#     plt.close()

#     # Save attribution overlay with plasma colormap
#     fig, ax = plt.subplots()
#     heatmap = ax.imshow(image_np, cmap='gray')
#     overlay = ax.imshow(attribution_map, cmap='plasma', alpha=0.5)
#     plt.axis('off')
#     cbar = plt.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)
#     cbar.set_label('Attribution Intensity', rotation=270, labelpad=15)
#     plt.savefig(os.path.join(folder, f"image_{idx}_integrated_gradients.png"), bbox_inches='tight', pad_inches=0)
#     plt.close()


# # Main loop
# net.eval()
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# net.to(device)
# num_images = 100

# import random
# indices = random.sample(range(len(test_dataset)), num_images)
# for idx in indices:
#     image_tensor, mask = get_sample_from_dataset(test_dataset, idx)
#     image_tensor = image_tensor.to(device)

#     attribution_map = integrated_gradients(image_tensor, net, target_class=0, steps=100, device=device)

#     print(f"Image {idx} - Attribution min:", attribution_map.min())
#     print(f"Image {idx} - Attribution max:", attribution_map.max())

#     plt.hist(attribution_map.ravel(), bins=50)
#     plt.title(f"Attribution Histogram for Image {idx}")
#     plt.xlabel("Attribution Value")  # ✅ Axis label
#     plt.ylabel("Pixel Count")        # ✅ Axis label
#     plt.show()

#     save_visualizations(image_tensor.cpu(), mask, attribution_map, idx)






# print(f"Saved visualizations for {num_images} images under './output/' folder.")



