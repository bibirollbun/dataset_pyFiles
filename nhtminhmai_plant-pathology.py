%%capture
!pip install imutils


import os
import math
import random
from typing import Dict, List,Tuple
import requests
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path, PurePath
import pathlib
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import json
import torchvision
from torchvision import datasets
from torchvision import transforms

import seaborn as sns
from sklearn.metrics import classification_report, multilabel_confusion_matrix, confusion_matrix, f1_score, precision_score

from PIL import Image

from sklearn.model_selection import train_test_split

from imutils import paths

import textwrap
from tqdm import tqdm





class config:
    # specify the paths to datasets
    DATA_DIR = Path('../input/plant-pathology-2021-fgvc8/train_images')
    ROOT_DIR = Path('./data')
    TRAIN_DIR = ROOT_DIR.joinpath('train')
    TEST_DIR = ROOT_DIR.joinpath('test')
    VAL_DIR = ROOT_DIR.joinpath('val')

    # set the input height and width
    INPUT_HEIGHT = 224
    INPUT_WIDTH = 224

    # set the input heig/ht and width
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    IMAGE_TYPE = '.jpg'
    BATCH_SIZE = 32
    # will use the vision transformer
    MODEL_NAME = 'vit_base'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TRAINING_PARAMS = 'training_hyperparams/default_train_params'
    LABELS = ['complex', 'frog_eye_leaf_spot', 'healthy', 'powdery_mildew', 'rust', 'scab']
    NUM_CLASSES = len(LABELS)
    CHECKPOINT_DIR = 'checkpoints'






!mkdir data


# # Set the desired output dimensions
# output_size = (224, 224)

# # Get a list of all image file paths in the input directory
# image_paths = list(paths.list_images(config.DATA_DIR))

# # Create a progress bar object
# progress_bar = tqdm(total=len(image_paths), desc='Resizing images')

# # Loop over all image file paths
# for image_path in image_paths:
#     # Load the image with PIL
#     image_path=Path(image_path)
#     image = Image.open(image_path)

#     # Resize the image
#     resized_image = image.resize(output_size)

#     # Get the output file path
#     output_path = config.ROOT_DIR / image_path.name

#     # Save the resized image to disk
#     resized_image.save(output_path)
#     # Update the progress bar
#     progress_bar.update(1)
    
# # Close the progress bar
# progress_bar.close()


# import shutil
# import os

# # Name of the output file
# output_filename = "my_kaggle_data"

# # Directory to zip (usually '/kaggle/working')
# directory_to_zip = "/kaggle/working"

# # Create the zip file
# shutil.make_archive(output_filename, 'zip', directory_to_zip)

# print(f"Created {output_filename}.zip")


# from IPython.display import FileLink

# # This will create a clickable link in the output area
# FileLink(r'my_kaggle_data.zip')


def split_df(csv_dir):
    '''
    This function take csv file and split it into train, valid, and test
    '''
    df = pd.read_csv(csv_dir)
    # df['image'] =  df['image'].apply(lambda x: './data/' + x) # Use when run resized by yourself
    df['image'] =  df['image'].apply(lambda x: '../input/multi-label-classification-plant-pathology/data/' + x)

    # train dataframe
    train_df, dummy_df = train_test_split(df,  train_size= 0.7, shuffle= True, random_state= 42)

    # valid and test dataframe
    valid_df, test_df = train_test_split(dummy_df,  train_size= 0.5, shuffle= True, random_state= 42)

    return train_df, valid_df, test_df


train_df, valid_df, test_df = split_df('../input/plant-pathology-2021-fgvc8/train.csv')


(train_df['labels'].value_counts()).plot(kind='bar')


all_labels = train_df['labels'].str.split(expand=True).stack().reset_index(drop=True)


all_labels.value_counts().plot(kind='bar')


def examine_images(df, num_images=20):
    image_paths = df['image'].sample(n=num_images, random_state=42)
    labels = df['labels'].loc[image_paths.index]

    num_rows = int(math.ceil(num_images/5))
    num_cols = 5
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(30, 30),tight_layout=True)
    axs = axs.ravel()

    for i, image_path in enumerate(image_paths):
        image = Image.open(Path(image_path))
        label = labels.iloc[i]
        axs[i].imshow(image)
        axs[i].set_title(f"Label: {label}", fontsize=25)
        axs[i].axis('off')
    plt.show()


examine_images(train_df, num_images=20)


# initialize our data augmentation functions
resize = transforms.Resize(size=(config.INPUT_HEIGHT,config.INPUT_WIDTH))
make_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
center_cropper = transforms.CenterCrop((config.INPUT_HEIGHT,config.INPUT_WIDTH))
random_resized_crop = transforms.RandomResizedCrop(size=(config.INPUT_HEIGHT, config.INPUT_WIDTH), scale=(0.6, 1.0))
random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.75)
random_vertical_flip = transforms.RandomVerticalFlip(p=0.75)
random_rotation = transforms.RandomRotation(degrees=90)
random_crop = transforms.RandomCrop(size=(200,200))
augmix = transforms.AugMix(severity = 3, mixture_width=3, alpha=0.2)
auto_augment = transforms.AutoAugment()
random_augment = transforms.RandAugment()
color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

# Stronger augmentations for training from scratch
train_transforms = transforms.Compose([
    # Randomly crop and resize (simulates different distances/zoom)
    random_resized_crop,
    random_horizontal_flip,
    random_vertical_flip,
    random_rotation,
    # Jitter brightness/contrast/saturation (simulates different lighting)
    color_jitter,
    make_tensor,
    normalize
])

val_transforms = transforms.Compose([resize, make_tensor, normalize])


def apply_transform(img: Image, transform) -> np.ndarray:
    """
    Applies a transform to a PIL Image and returns a numpy array of the transformed image.

    Args:
        img (PIL.Image): The input image to transform.
        transform (torchvision.transforms.Compose): The transform to apply to the image.

    Returns:
        np.ndarray: A numpy array representing the transformed image.
    """
    # Apply the transform to the image
    if isinstance(transform, torchvision.transforms.Compose):
        # Apply PyTorch transform to image array
        transformed_image = train_transforms(img)

    elif isinstance(transform, A.Compose):
        # Apply Albumentations transform to image array
        img_array = np.array(img)
        transformed_image = transform(image=img_array)["image"]

    # Convert the image tensor to a numpy array and transpose the axes to (height, width, channels)
    img_array = transformed_image.numpy().transpose((1, 2, 0))

    # Clip the pixel values to the range [0, 1]
    img_array = np.clip(img_array, 0, 1)

    return img_array


def visualize_transform(image: np.ndarray, original_image: np.ndarray = None) -> None:
    """
    Visualize the transformed image.

    Args:
        image (np.ndarray): A NumPy array representing the transformed image.
        original_image (np.ndarray, optional): A NumPy array representing the original image. Defaults to None.
    """
    fontsize = 18
    
    if original_image is None:
        # Create a plot with 1 row and 2 columns.
        f, ax = plt.subplots(1, 2, figsize=(12, 12))

        # Show the transformed image in the first column.
        ax[0].imshow(image)
    else:
        # Create a plot with 1 row and 2 columns.
        f, ax = plt.subplots(1, 2, figsize=(12, 12))

        # Show the original image in the first column.
        ax[0].imshow(original_image)
        ax[0].set_title('Original image', fontsize=fontsize)
        
        # Show the transformed image in the second column.
        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)
        
img = Image.open(train_df['image'].sample(n=1).iloc[0])
img_array = apply_transform(img, train_transforms)
visualize_transform(img_array, original_image=img)


def encode_label(labels, class_list):
    """Encode a list of labels using one-hot encoding.

    Args:
        label: A list of labels to encode.
        class_list: A list of all possible labels. Defaults to DEFAULT_LABELS.

    Returns:
        A tensor representing the one-hot encoding of the input labels.
    """
    # Create a tensor of zeros with the same length as the class list
    target = torch.zeros(len(class_list))
    for label in labels:
        # Find the index of the current label in the class list
        idx = class_list.index(label)
        # Set the corresponding index in the target tensor to 1
        target[idx] = 1
    return target



def decode_label(encoded_label, class_list):
    """Decode a one-hot encoded label into its original label(s).

    Args:
        encoded_label: A tensor representing the one-hot encoding of a label.
        class_list: A list of all possible labels. Defaults to DEFAULT_LABELS.

    Returns:
        A list of the decoded label(s).
    """
    # Use a list comprehension to create the decoded list
    decoded = [class_list[i] for i, val in enumerate(encoded_label) if val == 1]

    # Return the list of decoded label(s)
    return decoded


class PlantDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image_path = self.dataframe['image'].iloc[idx]
        image = Image.open(image_path)
        labels = self.dataframe.iloc[idx]['labels'].split(' ')
        encoded_labels = encode_label(labels, config.LABELS)
        if self.transform:
            image = self.transform(image)
        return image, encoded_labels


train_dataset = PlantDataset(train_df, transform = train_transforms)
val_dataset = PlantDataset(valid_df, transform = val_transforms)
test_dataset = PlantDataset(test_df, transform = val_transforms)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)


#get the class counts
class_counts = all_labels.value_counts()[config.LABELS]
class_counts


# Compute inverse class frequency
class_weights = torch.reciprocal(torch.tensor(class_counts.values).float()) # invert the counts and convert them to floats
class_weights /= torch.max(class_weights) # normalize the weights by the maximum weight

# Define loss function using class weights
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)


class_weights


import torchvision.transforms as transforms

# Stronger augmentations for training from scratch
train_transforms = transforms.Compose([
    # Randomly crop and resize (simulates different distances/zoom)
    transforms.RandomResizedCrop(size=(config.INPUT_HEIGHT, config.INPUT_WIDTH), scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    # Jitter brightness/contrast/saturation (simulates different lighting)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
])

val_transforms = transforms.Compose([
    transforms.Resize(size=(config.INPUT_HEIGHT, config.INPUT_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
])

# Re-initialize datasets with these new transforms
train_dataset = PlantDataset(train_df, transform=train_transforms)
val_dataset = PlantDataset(valid_df, transform=val_transforms)
test_dataset = PlantDataset(test_df, transform=val_transforms)

# Dataloaders (Keep your existing code, but ensure shuffle=True for train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)


import torch.nn as nn
import torch.nn.functional as F

class SimpleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CustomPlantCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResBlocks (Feature Extraction)
        self.layer1 = SimpleResBlock(64, 64, stride=1)
        self.layer2 = SimpleResBlock(64, 128, stride=2)
        self.layer3 = SimpleResBlock(128, 256, stride=2)
        self.layer4 = SimpleResBlock(256, 512, stride=2)
        
        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Dropout helps reduce overfitting
        self.dropout = nn.Dropout(p=0.5) 
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Initialize Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomPlantCNN(num_classes=config.NUM_CLASSES).to(device)
print(model)


# Use the loss function you already defined in your notebook (handling class imbalance)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Scheduler for "Fine Tuning" aspect (see Phase 4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    loop = tqdm(loader, leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        # Calculate simplistic accuracy for progress bar (threshold 0.5)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct_preds += (preds == labels).float().sum()
        total_preds += labels.numel() # Total number of individual label predictions
        
        loop.set_description(f"Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(loader)
    avg_acc = correct_preds / total_preds
    return avg_loss, avg_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_preds += (preds == labels).float().sum()
            total_preds += labels.numel()
            
    return running_loss / len(loader), correct_preds / total_preds


# num_epochs = 20
# history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
# best_val_loss = float('inf')

# for epoch in range(num_epochs):
#     print(f"Epoch {epoch+1}/{num_epochs}")
    
#     train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
#     val_loss, val_acc = validate(model, val_loader, criterion, device)
    
#     # Store history for monitoring
#     history['train_loss'].append(train_loss)
#     history['val_loss'].append(val_loss)
#     history['train_acc'].append(train_acc.item())
#     history['val_acc'].append(val_acc.item())
    
#     # "Fine-tune" aspect: Learning Rate Scheduling
#     scheduler.step(val_loss)
    
#     print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
#     # Save best model
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), 'best_model_scratch.pth')
#         print("Saved Best Model!")


# plt.figure(figsize=(12, 5))

# # Plot Loss
# plt.subplot(1, 2, 1)
# plt.plot(history['train_loss'], label='Train Loss')
# plt.plot(history['val_loss'], label='Validation Loss')
# plt.title('Training Process: Loss')
# plt.legend()

# # Plot Accuracy
# plt.subplot(1, 2, 2)
# plt.plot(history['train_acc'], label='Train Acc')
# plt.plot(history['val_acc'], label='Validation Acc')
# plt.title('Training Process: Accuracy')
# plt.legend()

# plt.show()


# --- Configuration ---
EPOCHS = 36 # Adjust as needed (30 recommended)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))

# ==========================================
# 1. TRAIN BASELINE MODEL
# ==========================================
print("--- Starting Baseline Training ---")
baseline_model = CustomPlantCNN(num_classes=config.NUM_CLASSES).to(device)
optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

best_baseline_f1 = 0.0

# --- METRIC LISTS ---
base_history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [], 'val_f1': []
}

for epoch in range(EPOCHS):
    baseline_model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    
    # Train Loop
    for images, labels in tqdm(train_loader, desc=f"Baseline Epoch {epoch+1}", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = baseline_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Calculate Stats
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        preds = (torch.sigmoid(outputs) > 0.5).float()
        acc = (preds == labels).float().mean().item()
        running_acc += acc * batch_size
        total_samples += batch_size

    # Compute Epoch Averages
    epoch_train_loss = running_loss / total_samples
    epoch_train_acc = running_acc / total_samples
    
    # Store in history dict
    base_history['train_loss'].append(epoch_train_loss)
    base_history['train_acc'].append(epoch_train_acc)
        
    # Validation
    val_loss, val_acc, val_f1 = evaluate_model(baseline_model, val_loader, criterion, device, config.LABELS, verbose=False)
    
    base_history['val_loss'].append(val_loss)
    base_history['val_acc'].append(val_acc)
    base_history['val_f1'].append(val_f1)
    
    scheduler.step(val_loss)
    
    print(f"  Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    # --- SAVE CHECKPOINT & HISTORY ---
    if val_f1 > best_baseline_f1:
        best_baseline_f1 = val_f1
        
        # 1. Save Model
        torch.save(baseline_model.state_dict(), 'baseline_model.pth')
        
        # 2. Save History to JSON (Easier to read/download)
        with open('baseline_history.json', 'w') as f:
            json.dump(base_history, f)
            
        print(f"  >> Saved New Best Baseline (F1: {val_f1:.4f}) and history.")

# ==========================================
# 2. TRAIN ADVANCED MODEL
# (MixUp + Cosine Annealing + COOLDOWN)
# ==========================================
print("\n--- Starting Advanced Training (MixUp + Cooldown) ---")
advanced_model = CustomPlantCNN(num_classes=config.NUM_CLASSES).to(device)
optimizer = torch.optim.AdamW(advanced_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_advanced_f1 = 0.0

# --- METRIC LISTS ---
adv_history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [], 'val_f1': []
}

cooldown_start = EPOCHS - 5

for epoch in range(EPOCHS):
    advanced_model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    
    use_mixup = True
    if epoch >= cooldown_start:
        use_mixup = False
        
    loop_desc = f"Adv Epoch {epoch+1} {'(MixUp)' if use_mixup else '(Cooldown)'}"
    
    for images, labels in tqdm(train_loader, desc=loop_desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        batch_size = images.size(0)
        
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4)
            outputs = advanced_model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            acc = lam * (preds == labels_a).float().mean().item() + (1 - lam) * (preds == labels_b).float().mean().item()
        else:
            outputs = advanced_model(images)
            loss = criterion(outputs, labels)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            acc = (preds == labels).float().mean().item()
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_size
        running_acc += acc * batch_size
        total_samples += batch_size
        
    scheduler.step()
    
    epoch_train_loss = running_loss / total_samples
    epoch_train_acc = running_acc / total_samples
    
    adv_history['train_loss'].append(epoch_train_loss)
    adv_history['train_acc'].append(epoch_train_acc)
    
    # Validation
    val_loss, val_acc, val_f1 = evaluate_model(advanced_model, val_loader, criterion, device, config.LABELS, verbose=False)
    
    adv_history['val_loss'].append(val_loss)
    adv_history['val_acc'].append(val_acc)
    adv_history['val_f1'].append(val_f1)
    
    print(f"  Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    # --- SAVE CHECKPOINT & HISTORY ---
    if val_f1 > best_advanced_f1:
        best_advanced_f1 = val_f1
        
        # 1. Save Model
        torch.save(advanced_model.state_dict(), 'advanced_model.pth')
        
        # 2. Save History
        with open('advanced_history.json', 'w') as f:
            json.dump(adv_history, f)
            
        print(f"  >> Saved New Best Advanced (F1: {val_f1:.4f}) and history.")


# --- 1. MixUp Helpers ---
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- 2. Threshold Calibration ---
def find_optimal_thresholds(model, val_loader, device):
    model.eval()
    val_probs = []
    val_targets = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            val_probs.append(torch.sigmoid(outputs).cpu())
            val_targets.append(labels.cpu())
            
    val_probs = torch.cat(val_probs)
    val_targets = torch.cat(val_targets)
    best_thresholds = []
    threshold_range = torch.arange(0.1, 0.95, 0.05)
    
    for i in range(val_probs.shape[1]):
        best_f1 = 0
        best_thresh = 0.5
        for thresh in threshold_range:
            preds = (val_probs[:, i] > thresh).float()
            score = f1_score(val_targets[:, i], preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = thresh.item()
        best_thresholds.append(best_thresh)
        
    return torch.tensor(best_thresholds).to(device)

# --- 3. Robust Evaluation Function ---
def evaluate_model(model, loader, criterion, device, class_names, thresholds=None, use_tta=False, verbose=True):
    model.eval()
    if thresholds is None:
        thresholds = torch.tensor([0.5] * len(class_names)).to(device)
    else:
        thresholds = thresholds.to(device)
        
    all_targets = []
    all_preds = []
    running_loss = 0.0
    
    if verbose:
        print(f"Evaluating... TTA={'ON' if use_tta else 'OFF'}")
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            logits_orig = model(images)
            
            if use_tta:
                # 3-Way TTA
                logits_h = model(torch.flip(images, dims=[3]))
                logits_v = model(torch.flip(images, dims=[2]))
                probs = (torch.sigmoid(logits_orig) + torch.sigmoid(logits_h) + torch.sigmoid(logits_v)) / 3.0
            else:
                probs = torch.sigmoid(logits_orig)
                
            loss = criterion(logits_orig, labels)
            running_loss += loss.item()
            
            preds = (probs > thresholds).float()
            all_targets.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    
    avg_loss = running_loss / len(loader)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    exact_acc = (all_targets == all_preds).all(axis=1).mean()
    
    return avg_loss, exact_acc, macro_f1


# Load the saved JSON files
with open('baseline_history.json', 'r') as f:
    base_history = json.load(f)

with open('advanced_history.json', 'r') as f:
    adv_history = json.load(f)

def plot_saved_metrics(history, title):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()

plot_saved_metrics(base_history, "Baseline Model")
plot_saved_metrics(adv_history, "Advanced Model")


base_model_path = "baseline_model.pth"
advanced_model_path = "advanced_model.pth"


# --- REPORT GENERATION ---
results = []

# 1. Load Best Baseline
model = CustomPlantCNN(num_classes=config.NUM_CLASSES).to(device)
model.load_state_dict(torch.load(base_model_path))

# Baseline: Standard Eval
loss, acc, f1 = evaluate_model(model, test_loader, criterion, device, config.LABELS, use_tta=False)
results.append({'Model': 'Baseline', 'Method': 'Standard', 'Test Acc': acc, 'F1 Score': f1})

# Baseline: + Thresholds
thresholds = find_optimal_thresholds(model, val_loader, device)
loss, acc, f1 = evaluate_model(model, test_loader, criterion, device, config.LABELS, thresholds=thresholds, use_tta=False)
results.append({'Model': 'Baseline', 'Method': '+ Threshold Calibration', 'Test Acc': acc, 'F1 Score': f1})

# Baseline: + Thresholds + TTA
loss, acc, f1 = evaluate_model(model, test_loader, criterion, device, config.LABELS, thresholds=thresholds, use_tta=True)
results.append({'Model': 'Baseline', 'Method': '+ TTA (Horizontal+Vertical)', 'Test Acc': acc, 'F1 Score': f1})


# 2. Load Best Advanced
model = CustomPlantCNN(num_classes=config.NUM_CLASSES).to(device)
model.load_state_dict(torch.load(advanced_model_path))

# Advanced: Standard Eval
loss, acc, f1 = evaluate_model(model, test_loader, criterion, device, config.LABELS, use_tta=False)
results.append({'Model': 'Advanced (MixUp+Cos)', 'Method': 'Standard', 'Test Acc': acc, 'F1 Score': f1})

# Advanced: + Thresholds
thresholds = find_optimal_thresholds(model, val_loader, device)
loss, acc, f1 = evaluate_model(model, test_loader, criterion, device, config.LABELS, thresholds=thresholds, use_tta=False)
results.append({'Model': 'Advanced (MixUp+Cos)', 'Method': '+ Threshold Calibration', 'Test Acc': acc, 'F1 Score': f1})

# Advanced: + Thresholds + TTA
loss, acc, f1 = evaluate_model(model, test_loader, criterion, device, config.LABELS, thresholds=thresholds, use_tta=True)
results.append({'Model': 'Advanced (MixUp+Cos)', 'Method': '+ TTA (Full Pipeline)', 'Test Acc': acc, 'F1 Score': f1})

# --- Display Final DataFrame ---
df_results = pd.DataFrame(results)
print("\n" + "="*40)
print("FINAL MODEL COMPARISON REPORT")
print("="*40)
display(df_results)

# Optional: Plot Comparison
plt.figure(figsize=(10,6))
plt.barh(df_results['Method'] + ' (' + df_results['Model'] + ')', df_results['Test Acc'], color='skyblue')
plt.xlabel('Test Accuracy')
plt.title('Impact of Improvements on Model Performance')
plt.axvline(x=0.84, color='r', linestyle='--', label='Target (0.84)')
plt.legend()
plt.show()


# --- 1. UTILITY FUNCTIONS (Decoding & Plotting) ---

def decode_label(encoded_label, class_list):
    """Converts binary tensor/list to list of class strings"""
    return [class_list[i] for i, val in enumerate(encoded_label) if val == 1]

def pred_and_plot_image(model, image_path, subplot, ground_truth=None, 
                        class_names=config.LABELS, thresholds=None, device=config.DEVICE):
    """Predicts and plots a single image with threshold support"""
    
    # Load Image
    if isinstance(image_path, pathlib.PosixPath) or isinstance(image_path, str):
        # Handle local path or string path
        if str(image_path).startswith('http'):
             img = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
        else:
             img = Image.open(image_path).convert('RGB')
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((config.INPUT_HEIGHT, config.INPUT_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        
        if thresholds is not None:
            # Broadcast thresholds to match batch size
            preds = (probs > thresholds).int()
        else:
            preds = (probs > 0.5).int()
            
    preds = preds.cpu().squeeze().tolist()
    predicted_labels = decode_label(preds, class_names)
    
    if not predicted_labels:
        predicted_labels = ["No Disease"]

    # Plot
    plt.subplot(*subplot)
    plt.imshow(img)
    
    if ground_truth:
        title = f"True: {ground_truth}\nPred: {', '.join(predicted_labels)}"
    else:
        title = f"Pred: {', '.join(predicted_labels)}"
        
    plt.title(title, fontsize=10)
    plt.axis('off')

def plot_random_test_images(model, test_df, thresholds=None):
    """Plots a grid of random test images with predictions"""
    num_images = 15
    sample = test_df.sample(n=num_images, random_state=42)
    image_paths = sample['image'].tolist()
    labels = sample['labels'].tolist()
    
    rows = int(np.ceil(num_images / 5))
    plt.figure(figsize=(20, rows * 4))
    
    for i, img_path in enumerate(image_paths):
        pred_and_plot_image(
            model=model,
            image_path=img_path,
            subplot=(rows, 5, i+1),
            ground_truth=labels[i],
            class_names=config.LABELS,
            thresholds=thresholds
        )
    plt.tight_layout()
    plt.show()

def analyze_model_performance(model, model_name, train_loader, val_loader, test_loader, thresholds=None):
    """
    Runs a full analysis: Train/Val/Test Loss, Detailed Metrics, Confusion Matrix, and Visuals
    """
    print(f"\n{'='*20} ANALYZING: {model_name} {'='*20}")
    model.to(device)
    
    # 1. Calculate Final Losses (Snapshot)
    print("Calculating final Train/Val/Test stats...")
    train_loss, train_acc, _ = evaluate_model(model, train_loader, criterion, device, config.LABELS, thresholds=thresholds, verbose=False)
    val_loss, val_acc, _ = evaluate_model(model, val_loader, criterion, device, config.LABELS, thresholds=thresholds, verbose=False)
    test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, criterion, device, config.LABELS, thresholds=thresholds, verbose=False)
    
    print(f"\n--- Overall Performance ---")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")
    print(f"Test F1:    {test_f1:.4f}")

    # 2. Detailed Classification Report (Test Set)
    print(f"\n--- Detailed Classification Report (Test Set) ---")
    
    # Get all predictions for reporting
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            if thresholds is not None:
                preds = (probs > thresholds).float()
            else:
                preds = (probs > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    print(classification_report(all_targets, all_preds, target_names=config.LABELS, zero_division=0))
    
    # 3. Dominant Class Confusion Matrix
    # (Converts multi-label to single-label based on max probability for cleaner visualization)
    print(f"\n--- Confusion Matrix (Dominant Class) ---")
    pred_indices = np.argmax(all_preds, axis=1)
    target_indices = np.argmax(all_targets, axis=1)
    
    cm = confusion_matrix(target_indices, pred_indices)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=config.LABELS, yticklabels=config.LABELS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()
    
    # 4. Visual Predictions
    print(f"\n--- Visual Predictions on Test Set ---")
    plot_random_test_images(model, test_df, thresholds=thresholds)


# --- 2. MAIN EXECUTION BLOCK ---

results = []

# ==================================
# A. ANALYZE BASELINE MODEL
# ==================================
baseline_model = CustomPlantCNN(num_classes=config.NUM_CLASSES).to(device)
baseline_model.load_state_dict(torch.load(base_model_path))

# 1. Get Baseline Stats for Summary Table
thresholds_base = find_optimal_thresholds(baseline_model, val_loader, device)
loss, acc, f1 = evaluate_model(baseline_model, test_loader, criterion, device, config.LABELS, thresholds=thresholds_base, use_tta=True)
results.append({'Model': 'Baseline', 'Test Acc': acc, 'F1 Score': f1, 'Technique': 'Thresholds + TTA'})

# 2. Run Deep Dive Analysis
analyze_model_performance(baseline_model, "Baseline Model", train_loader, val_loader, test_loader, thresholds=thresholds_base)


# ==================================
# B. ANALYZE ADVANCED MODEL
# ==================================
advanced_model = CustomPlantCNN(num_classes=config.NUM_CLASSES).to(device)
advanced_model.load_state_dict(torch.load(advanced_model_path))

# 1. Get Advanced Stats for Summary Table
thresholds_adv = find_optimal_thresholds(advanced_model, val_loader, device)
loss, acc, f1 = evaluate_model(advanced_model, test_loader, criterion, device, config.LABELS, thresholds=thresholds_adv, use_tta=True)
results.append({'Model': 'Advanced (MixUp+Cos)', 'Test Acc': acc, 'F1 Score': f1, 'Technique': 'Thresholds + TTA'})

# 2. Run Deep Dive Analysis
analyze_model_performance(advanced_model, "Advanced Model", train_loader, val_loader, test_loader, thresholds=thresholds_adv)


# --- 3. FINAL SUMMARY TABLE ---
df_results = pd.DataFrame(results)
print("\n" + "="*40)
print("FINAL REPORT SUMMARY")
print("="*40)
display(df_results)


import requests
from PIL import Image
from io import BytesIO
import textwrap

def predict_from_url(url, model, transform, class_names, device):
    # 1. Download the image
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 2. Preprocess
    # We use the same 'val_transforms' (Resize + Normalize)
    img_tensor = transform(img).unsqueeze(0) # Add batch dimension -> [1, 3, 224, 224]
    img_tensor = img_tensor.to(device)

    # 3. Predict
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.sigmoid(output) # Convert to 0-1 range
        
        # Apply threshold to get binary labels
        preds = (probs > 0.5).int().cpu().numpy()[0]
        
    # 4. Decode Labels
    predicted_labels = [class_names[i] for i, val in enumerate(preds) if val == 1]
    
    if not predicted_labels:
        predicted_labels = ["Uncertain/Healthy"] # Fallback if no class > 0.5
        
    prediction_text = " | ".join(predicted_labels)

    # 5. Visualize
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    
    # Add title with wrapping to avoid cutting off text
    title = f"Prediction: {prediction_text}"
    plt.title("\n".join(textwrap.wrap(title, width=30)), fontsize=14, color='darkblue')
    plt.show()

# --- Run Predictions on your URLs ---

urls = [
    'https://www.planetnatural.com/wp-content/uploads/2012/12/common-rust-disease.jpg',
    'https://www.greenlife.co.ke/wp-content/uploads/2022/04/powdery_mildew.jpg',
    'https://c8.alamy.com/comp/PB2H05/frog-eye-leaf-spot-or-cercospora-diseases-on-leaves-of-suicide-tree-PB2H05.jpg'
]

print("Running Predictions on Real Images...")
for url in urls:
    predict_from_url(url, model, val_transforms, config.LABELS, device)


# training_params =  training_hyperparams.get(config.TRAINING_PARAMS)


# # To reduce clutter in the notebook I've turned the verbosity off, you can turn it on to see the full output
# training_params["train_metrics_list"] = ['my_accuracy']
# training_params["valid_metrics_list"] = ['my_accuracy']
# training_params["metric_to_watch"] = "my_accuracy"

# # Set the silent mode to True to reduce clutter in the notebook, you can turn it on to see the full output
# training_params["silent_mode"] = True
# training_params["optimizer"] = 'AdamW'
# training_params['average_best_models'] = True
# training_params['ema'] = True
# training_params["criterion_params"] = {'smooth_eps': 0.20}
# training_params["max_epochs"] = 30
# training_params["initial_lr"] = 0.00001
# training_params["loss"] = criterion


# model = models.get(config.MODEL_NAME, num_classes=config.NUM_CLASSES, pretrained_weights='imagenet')


# best_full_model = models.get(config.MODEL_NAME,
#                         num_classes=config.NUM_CLASSES,
#                         checkpoint_path=os.path.join(full_model_trainer.checkpoints_dir_path, "average_model.pth"))


# full_model_trainer.test(model=best_full_model,
#             test_loader=test_loader,
#             test_metrics_list=['my_accuracy'])


# plot_random_test_images(best_full_model, test_df)


# pred_and_plot_image(image_path='https://www.planetnatural.com/wp-content/uploads/2012/12/common-rust-disease.jpg', subplot=(1, 1, 1))


# pred_and_plot_image(image_path='https://www.greenlife.co.ke/wp-content/uploads/2022/04/powdery_mildew.jpg', subplot=(1, 1, 1))


# pred_and_plot_image(image_path='https://soybeanresearchinfo.com/wp-content/uploads/2020/05/Frogeye-leaf-spot-Daren-Mueller-17-1300x867.jpg', subplot=(1, 1, 1))



# pred_and_plot_image(image_path='https://c8.alamy.com/comp/PB2H05/frog-eye-leaf-spot-or-cercospora-diseases-on-leaves-of-suicide-tree-PB2H05.jpg', subplot=(1, 1, 1))

