# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside  of the current session


import os
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

# dataset paths for training and testing
train_images_path = "/kaggle/input/slicee-my-face/images/train"
train_masks_path  = "/kaggle/input/slicee-my-face/annotations/train"

test_images_path = "/kaggle/input/slicee-my-face/images/test"
test_masks_path  = "/kaggle/input/slicee-my-face/annotations/test"

val_images_path = "/kaggle/input/slicee-my-face/images/val"
val_masks_path = "/kaggle/input/slicee-my-face/annotations/val"

train_image_files = sorted([
    f for f in os.listdir(train_images_path) 
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

train_mask_files = sorted([
    f for f in os.listdir(train_masks_path) 
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

print(f"Total train images used: {len(train_image_files)}")
print(f"Total train masks used: {len(train_mask_files)}")


test_image_files = sorted([
    f for f in os.listdir(test_images_path) 
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

test_mask_files = sorted([
    f for f in os.listdir(test_masks_path) 
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

print(f"Total test images used: {len(test_image_files)}")
print(f"Total test masks used: {len(test_mask_files)}")

val_image_files = sorted([
    f for f in os.listdir(val_images_path) 
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

val_mask_files = sorted([
    f for f in os.listdir(val_masks_path) 
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

print(f"Total val images used: {len(val_image_files)}")
print(f"Total val masks used: {len(val_mask_files)}")



"""
I wrote this script to preprocess my dataset for the segmentation task. 
It loads images and their corresponding masks from the specified directories, resizes them to 256x256, and normalizes the image pixel values to the range [0,1]. 
Since the masks are originally grayscale with decimal values (multiples of 1/255), I convert them to integer labels by multiplying by 255, rounding the result, and remapping the unique values to a contiguous range starting from 0. 
This script processes the training, test, and validation data separately, converts them into NumPy arrays, and prints the shapes of the processed arrays.
"""

IMG_SIZE = (256, 256)

def load_and_preprocess(image_path, mask_path):
    """
    Loads an image and its mask, resizes to IMG_SIZE,
    and normalizes pixel values to [0,1].
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)
    image = image.astype(np.float32) / 255.0
    
    # read mask (grayscale)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, IMG_SIZE)
    mask = mask.astype(np.float32) / 255.0
    
    return image, mask


#!!!!! converting labels of mask from decimal to int
def convert_mask_to_int(mask):
    """
    Convert a normalized mask (values in [0,1] multiples of 1/255) into an integer mask.
    The function here multiplies by 255, rounds the result, and then remaps the unique values
    to a contiguous range starting from 0.
    """
  
    mask_int = np.round(mask * 255).astype(np.uint8)
    unique_vals = np.unique(mask_int)
    mapping = {old_val: new_val for new_val, old_val in enumerate(np.sort(unique_vals))}
    
    remapped_mask = np.copy(mask_int)
    for old_val, new_val in mapping.items():
        remapped_mask[mask_int == old_val] = new_val
        
    return remapped_mask

# process training data
processed_train_images = []
processed_train_masks = []

for img_file, msk_file in zip(train_image_files, train_mask_files):
    img_path = os.path.join(train_images_path, img_file)
    mask_path = os.path.join(train_masks_path, msk_file)
    
    image, mask = load_and_preprocess(img_path, mask_path)
    mask = convert_mask_to_int(mask)  # converts mask to integer labels
    
    processed_train_images.append(image)
    processed_train_masks.append(mask)

print(f"Preprocessed {len(processed_train_images)} training images and {len(processed_train_masks)} training masks.")

processed_train_images_np = np.array(processed_train_images)  
processed_train_masks_np  = np.array(processed_train_masks)    

print("processed_train_images_np shape:", processed_train_images_np.shape)
print("processed_train_masks_np shape:", processed_train_masks_np.shape)

# process Test Data
processed_test_images = []
processed_test_masks = []

for img_file, msk_file in zip(test_image_files, test_mask_files):
    img_path = os.path.join(test_images_path, img_file)
    mask_path = os.path.join(test_masks_path, msk_file)
    
    image, mask = load_and_preprocess(img_path, mask_path)
    mask = convert_mask_to_int(mask)
    
    processed_test_images.append(image)
    processed_test_masks.append(mask)

print(f"Preprocessed {len(processed_test_images)} test images and {len(processed_test_masks)} test masks.")

processed_test_images_np = np.array(processed_test_images)  
processed_test_masks_np  = np.array(processed_test_masks)    

print("processed_test_images_np shape:", processed_test_images_np.shape)
print("processed_test_masks_np shape:", processed_test_masks_np.shape)

# process val data
processed_val_images = []
processed_val_masks = []

for img_file, msk_file in zip(val_image_files, val_mask_files):
    img_path = os.path.join(val_images_path, img_file)
    mask_path = os.path.join(val_masks_path, msk_file)
    
    image, mask = load_and_preprocess(img_path, mask_path)
    mask = convert_mask_to_int(mask)
    
    processed_val_images.append(image)
    processed_val_masks.append(mask)

print(f"Preprocessed {len(processed_val_images)} val images and {len(processed_val_masks)} val masks.")

processed_val_images_np = np.array(processed_val_images)  
processed_val_masks_np  = np.array(processed_val_masks)    

print("processed_val_images_np shape:", processed_val_images_np.shape)
print("processed_val_masks_np shape:", processed_val_masks_np.shape)




def show_random_samples(images, masks, num_samples=3):
    for _ in range(num_samples):
        idx = random.randint(0, len(images) - 1)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(images[idx])
        ax[0].set_title("Processed Image")
        ax[0].axis("off")
        ax[1].imshow(masks[idx], cmap="gray")
        ax[1].set_title("Processed Mask")
        ax[1].axis("off")
        plt.show()

# display 3 random samples
show_random_samples(processed_train_images, processed_train_masks, num_samples=3)


import numpy as np

for i in range(5):
    unique_vals = np.unique(processed_train_masks_np[i])
    print(f"Mask {i} unique values:", unique_vals)



"""
I have written this script to build a multi-class segmentation pipeline using PyTorch. 
In this code, I created a custom dataset class that handles image-mask pairs, where the images are normalized 
and the masks are in integer format representing different classes. I then built an attention-based U-Net model, 
named SegFace, which consists of an encoder, a bottleneck, and a decoder with attention gates to improve feature fusion. 

For training, I use the entire training dataset provided (processed_train_images_np and processed_train_masks_np) 
and a separate validation dataset (processed_val_images_np and processed_val_masks_np). The model is trained using 
cross-entropy loss and the Adam optimizer, and I evaluate its performance using the Dice coefficient. The best model 
(based on validation Dice) is saved for future use. 

I have also included a function to visualize the predictions on the validation set so that I can easily check how well 
the model is performing on unseen data. This script is written in a clear and understandable way, reflecting my own 
approach to solving the segmentation task.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class MulticlassSegDataset(Dataset):
    def __init__(self, images_np, masks_np, transform=None):
        self.images_np = images_np
        self.masks_np = masks_np
        self.transform = transform
    def __len__(self):
        return len(self.images_np)
    def __getitem__(self, idx):
        image = self.images_np[idx].astype(np.float32)
        mask = self.masks_np[idx].astype(np.int64)
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)
        return image_tensor, mask_tensor

X_train = processed_train_images_np
y_train = processed_train_masks_np
X_val = processed_val_images_np
y_val = processed_val_masks_np

print("Train set images shape:", X_train.shape, "masks shape:", y_train.shape)
print("Val set images shape:", X_val.shape, "masks shape:", y_val.shape)

train_dataset = MulticlassSegDataset(X_train, y_train)
val_dataset = MulticlassSegDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random

class MulticlassSegDataset(Dataset):
    def __init__(self, images_np, masks_np, transform=None):
        self.images_np = images_np
        self.masks_np = masks_np
        self.transform = transform
    def __len__(self):
        return len(self.images_np)
    def __getitem__(self, idx):
        image = self.images_np[idx].astype(np.float32)
        mask = self.masks_np[idx].astype(np.int64)
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)
        return image_tensor, mask_tensor

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class SegFace(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, features=[64, 128, 256, 512]):
        super(SegFace, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature
        self.bottleneck = self._block(features[-1], features[-1]*2)
        self.decoder = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.attention_gates.append(AttentionBlock(F_g=feature, F_l=feature, F_int=feature//2))
            self.decoder.append(self._block(feature*2, feature))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x):
        skip_connections = []
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = skip_connections[idx//2]
            attn_skip = self.attention_gates[idx//2](g=x, x=skip)
            x = torch.cat((attn_skip, x), dim=1)
            x = self.decoder[idx+1](x)
        return F.softmax(self.final_conv(x), dim=1)
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

def multiclass_dice_score(pred, target, smooth=1e-5):
    num_classes = pred.shape[1]
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    pred_argmax = pred.argmax(dim=1, keepdim=True)
    pred_onehot = F.one_hot(pred_argmax.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()
    intersection = (pred_onehot * target_onehot).sum(dim=(2,3))
    union = pred_onehot.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))
    dice_per_class = (2. * intersection + smooth) / (union + smooth)
    return dice_per_class.mean()

print("Unique labels in training masks:", np.unique(processed_train_masks_np))
num_classes = int(np.unique(processed_train_masks_np).max()) + 1
print("Number of classes:", num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegFace(in_channels=3, out_channels=num_classes).to(device)
print("SegFace model initialized with", num_classes, "classes.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 40
best_val_dice = 0.0
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * images.size(0)
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    model.eval()
    running_val_loss = 0.0
    running_val_dice = 0.0
    total_val_samples = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, masks)
            running_val_loss += val_loss.item() * images.size(0)
            batch_dice = multiclass_dice_score(outputs, masks)
            running_val_dice += batch_dice.item() * images.size(0)
            total_val_samples += images.size(0)
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_val_dice = running_val_dice / total_val_samples
    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Dice: {epoch_val_dice:.4f}")
    if epoch_val_dice > best_val_dice:
        best_val_dice = epoch_val_dice
        torch.save(model.state_dict(), "segface_multiclass_best.pth")
        print("** Model Saved! **")
print(f"Training complete. Best Validation Dice: {best_val_dice:.4f}")

def visualize_predictions(dataset, model, num_samples=10):
    indices = random.sample(range(len(dataset)), num_samples)
    for idx in indices:
        image_tensor, true_mask_tensor = dataset[idx]
        image_np = np.transpose(image_tensor.numpy(), (1, 2, 0))
        true_mask_np = true_mask_tensor.numpy()
        image_input = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_input)
        pred_mask = outputs.argmax(dim=1).squeeze(0).cpu().numpy()
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        ax[0].imshow(image_np)
        ax[0].set_title("Original Image")
        ax[0].axis("off")
        ax[1].imshow(true_mask_np, cmap="tab20")
        ax[1].set_title("True Mask")
        ax[1].axis("off")
        ax[2].imshow(pred_mask, cmap="tab20")
        ax[2].set_title("Predicted Mask")
        ax[2].axis("off")
        plt.tight_layout()
        plt.show()

visualize_predictions(val_dataset, model, num_samples=10)



"""
This code is for visualizing the model's predictions on random test images. 
The model takes preprocessed test images as input and generates predicted segmentation masks. 
Since the output mask contains class labels, I convert it to a grayscale image by stretching the values across the 0-255 range. 
For better comparison, I do the same for the true mask. 
Each visualization consists of the original test image, its corresponding ground truth mask, and the predicted mask, all displayed side by side.
""" 

import torch
import numpy as np
import matplotlib.pyplot as plt
import random

def stretch_mask(mask):
    max_val = mask.max()
    if max_val == 0:
        return mask.astype(np.uint8)
    stretched = (mask.astype(float) / max_val) * 255.0
    return stretched.astype(np.uint8)

# vistualize masks of test images
def visualize_test_predictions(model, processed_test_images_np, processed_test_masks_np, device, num_samples=10):
    model.eval()
    indices = random.sample(range(processed_test_images_np.shape[0]), num_samples)
    
    for idx in indices:
        image_np = processed_test_images_np[idx]
        true_mask_np = processed_test_masks_np[idx]
        
        image_tensor = torch.from_numpy(np.transpose(image_np, (2, 0, 1))).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
        pred_mask = outputs.argmax(dim=1).squeeze(0).cpu().numpy()
        
        pred_mask_gray = stretch_mask(pred_mask)
        true_mask_gray = stretch_mask(true_mask_np)
        
        fig, ax = plt.subplots(1, 3, figsize=(15,5))
        ax[0].imshow(image_np)
        ax[0].set_title(f"Test Image {idx}")
        ax[0].axis("off")
        
        ax[1].imshow(true_mask_gray, cmap='gray', vmin=0, vmax=255)
        ax[1].set_title("True Mask (Grayscale)")
        ax[1].axis("off")
        
        ax[2].imshow(pred_mask_gray, cmap='gray', vmin=0, vmax=255)
        ax[2].set_title("Predicted Mask (Grayscale)")
        ax[2].axis("off")
        
        plt.tight_layout()
        plt.show()

visualize_test_predictions(model, processed_test_images_np, processed_test_masks_np, device, num_samples=10)



"""
This code is for generating the submission file using the trained SegFace model. 
The test images are already processed, so I just pass them through the model to get predictions. 
Since it's a multi-class segmentation model, the output is converted to a binary mask by keeping all non-zero predictions as foreground. 
The original test image size needs to be restored before encoding, so I read the image dimensions and resize the predicted mask accordingly. 
Run-length encoding (RLE) is used to store the segmentation mask in a format suitable for submission. 
Finally, the results are written to a CSV file with image IDs and their corresponding RLE-encoded masks, the sorting of masks is done alphanumerically here.
"""

import os
import numpy as np
import torch
import csv
import cv2

def rle_encode(mask):
    pixels = mask.flatten()  
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if len(runs) % 2 == 1:
        runs = runs[:-1]
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# generate binary masks (resize masks to original dimensions as well)
def generate_submission(model, processed_test_images_np, test_image_files, test_images_path, device, num_classes, submission_filename="submission.csv"):
    test_image_files = sorted(test_image_files, key=lambda x: x.lower())
    model.eval()
    submission_data = []
    N = processed_test_images_np.shape[0]
    for i in range(N):
        image_np = processed_test_images_np[i]
        image_tensor = torch.from_numpy(np.transpose(image_np, (2, 0, 1))).float().unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
        pred_mask = outputs.argmax(dim=1).squeeze(0).cpu().numpy()
        binary_mask = (pred_mask > 0).astype(np.uint8)
        image_filename = test_image_files[i]
        original_image_path = os.path.join(test_images_path, image_filename)
        orig_img = cv2.imread(original_image_path)
        if orig_img is None:
            orig_h, orig_w = binary_mask.shape
        else:
            orig_h, orig_w = orig_img.shape[:2]
        upscaled_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        rle = rle_encode(upscaled_mask)
        image_id = os.path.splitext(image_filename)[0]
        submission_data.append([image_id, rle])
    with open(submission_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "predicted"])
        writer.writerows(submission_data)
    print(f"Submission file '{submission_filename}' created with {len(submission_data)} entries.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("segface_multiclass_best.pth", map_location=device))
model.eval()
generate_submission(model, processed_test_images_np, test_image_files, test_images_path, device, num_classes, submission_filename="submission.csv")





