# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ===========================================
# Step 0: Import Libraries & Set Seeds
# ===========================================
import numpy as np
import pandas as pd
import os
import zipfile
import cv2
import random
from PIL import Image
from tqdm.notebook import tqdm 
import matplotlib.pyplot as plt
import time 

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

# --- For Reproducibility ---
SEED = 42
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multi-GPU
    # These are needed for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
print("Libraries imported and seed set.")


# ===========================================
# Step 1: Define Data Augmentations
# ===========================================
# Define image size
IMG_SIZE = 128 # U-Net works well with powers of 2

# Training augmentations
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Validation/Test augmentations
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
print("Augmentations defined.")


# ===========================================
# Step 2: Define UNet Model (Encoder + Decoder)
# ===========================================
class ResNetEncoder(nn.Module):
    def __init__(self, backbone_name='resnet34', pretrained=True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        if backbone_name == 'resnet34':
            backbone = models.resnet34(weights=weights)
            channels = [64, 64, 128, 256, 512]
        elif backbone_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError("Unsupported backbone")

        self.channels = channels
        self.initial = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        return x0, x1, x2, x3, x4

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
             x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class UNet(nn.Module):
    def __init__(self, backbone_name='resnet34', pretrained=True, n_classes=1):
        super().__init__()
        self.encoder = ResNetEncoder(backbone_name, pretrained)
        channels = self.encoder.channels

        self.decoder4 = DecoderBlock(channels[4], channels[3], 256)
        self.decoder3 = DecoderBlock(256, channels[2], 128)
        self.decoder2 = DecoderBlock(128, channels[1], 64)
        self.decoder1 = DecoderBlock(64, channels[0], 64)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.encoder(x)
        d4 = self.decoder4(x4, x3)
        d3 = self.decoder3(d4, x2)
        d2 = self.decoder2(d3, x1)
        d1 = self.decoder1(d2, x0)
        out = self.final_conv(d1)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        return out

print("U-Net Model defined.")


# ===========================================
# Step 3: Loss Functions
# ===========================================
class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=weight, size_average=size_average)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (probs * targets).sum(dim=(2, 3))
        dice_coeff = (2. * intersection + smooth) / (probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)
        dice_loss = 1 - dice_coeff.mean()
        return bce_loss + dice_loss

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge_flat(logits, labels):
    logits, labels = logits.view(-1), labels.view(-1)
    signs = 2. * labels.float() - 1.
    errors = 1. - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

class LovaszLoss(nn.Module):
    def __init__(self, per_image=True):
        super().__init__()
        self.per_image = per_image

    def forward(self, logits, targets):
        targets = targets.float()
        if self.per_image:
            loss = torch.stack([lovasz_hinge_flat(logits[i], targets[i]) for i in range(logits.shape[0])]).mean()
        else:
            loss = lovasz_hinge_flat(logits.view(-1), targets.view(-1))
        return loss

print("Loss functions defined.")


# ===========================================
# Step 4: Data Loading, Filtering, and Split
# ===========================================
# --- Unzip Data ---
import os
import zipfile
import shutil

# Define paths
train_zip_path = '/kaggle/input/tgs-salt-identification-challenge/train.zip'
test_zip_path = '/kaggle/input/tgs-salt-identification-challenge/test.zip'
working_dir = '/kaggle/working/'

# Check current working directory contents
print("Before extraction, working directory contains:", os.listdir(working_dir))

# Temporary extraction directory
temp_dir = os.path.join(working_dir, 'temp_extract')
os.makedirs(temp_dir, exist_ok=True)

# Extract and organize train data
if not os.path.exists(os.path.join(working_dir, 'train')):
    print("Extracting train.zip...")
    with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Create train directory
    os.makedirs(os.path.join(working_dir, 'train'), exist_ok=True)
    
    # Move extracted content to train directory
    for item in os.listdir(temp_dir):
        src = os.path.join(temp_dir, item)
        dst = os.path.join(working_dir, 'train', item)
        shutil.move(src, dst)
    
    print("Train.zip extracted and organized!")
else:
    print("Train directory already exists.")

# Clean up temp directory
if os.path.exists(temp_dir):
    os.rmdir(temp_dir)

# Extract and organize test data
temp_dir = os.path.join(working_dir, 'temp_extract')
os.makedirs(temp_dir, exist_ok=True)

if not os.path.exists(os.path.join(working_dir, 'test')):
    print("Extracting test.zip...")
    with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Create test directory
    os.makedirs(os.path.join(working_dir, 'test'), exist_ok=True)
    
    # Move extracted content to test directory
    for item in os.listdir(temp_dir):
        src = os.path.join(temp_dir, item)
        dst = os.path.join(working_dir, 'test', item)
        shutil.move(src, dst)
    
    print("Test.zip extracted and organized!")
else:
    print("Test directory already exists.")

# Clean up temp directory
if os.path.exists(temp_dir):
    os.rmdir(temp_dir)

# Check the final directory structure
print("\nAfter extraction, working directory contains:", os.listdir(working_dir))
if os.path.exists(os.path.join(working_dir, 'train')):
    print("Train directory contains:", os.listdir(os.path.join(working_dir, 'train')))
if os.path.exists(os.path.join(working_dir, 'test')):
    print("Test directory contains:", os.listdir(os.path.join(working_dir, 'test')))


# --- Set Paths ---
TRAIN_IMAGE_DIR = '/kaggle/working/train/images'
TRAIN_MASK_DIR = '/kaggle/working/train/masks'
TEST_IMAGE_DIR = '/kaggle/working/test/images'

# --- Load Train Filenames ---
train_ids = [f.split('.')[0] for f in os.listdir(TRAIN_IMAGE_DIR) if f.endswith('.png')]
train_image_paths = [os.path.join(TRAIN_IMAGE_DIR, f"{img_id}.png") for img_id in train_ids]
train_mask_paths = [os.path.join(TRAIN_MASK_DIR, f"{img_id}.png") for img_id in train_ids]

# --- Train/Validation Split (using all data for now) ---
train_images, val_images, train_masks, val_masks = train_test_split(
    train_image_paths, # Use all images
    train_mask_paths,
    test_size=0.2,
    random_state=SEED
)

print(f"Total training images: {len(train_ids)}")
print(f"Training images after split: {len(train_images)}")
print(f"Validation images: {len(val_images)}")

# --- Load Test Filenames ---
test_ids = sorted([f.split('.')[0] for f in os.listdir(TEST_IMAGE_DIR) if f.endswith('.png')]) # Sort for consistent submission order
test_image_paths = [os.path.join(TEST_IMAGE_DIR, f"{img_id}.png") for img_id in test_ids]
print(f"Test images: {len(test_ids)}")


# ===========================================
# Step 5: Define SaltDataset and Create DataLoaders
# ===========================================
class SaltDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None, is_test=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.is_test:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            image_id = os.path.basename(image_path).split('.')[0]
            return image, image_id
        else:
            mask_path = self.mask_paths[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: 
                mask = np.zeros((101, 101), dtype=np.uint8) 

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            if mask.ndim == 2:
                 mask = mask.unsqueeze(0)
            mask = mask.float() / 255.0
            return image, mask

# --- Create Datasets ---
train_dataset = SaltDataset(train_images, train_masks, transform=train_transform)
val_dataset = SaltDataset(val_images, val_masks, transform=val_transform)
test_dataset = SaltDataset(test_image_paths, mask_paths=None, transform=val_transform, is_test=True)

# --- Create DataLoaders ---
BATCH_SIZE = 32
NUM_WORKERS = 2

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True) # drop_last helps with batch norm stability
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print("Datasets and DataLoaders are ready!")


# ===========================================
# Step 6: IoU Metric and Train/Validate Loop
# ===========================================
def compute_iou(preds, targets, threshold=0.5, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds_binary = (preds > threshold).float()
    targets_binary = (targets > 0.5).float()
    intersection = (preds_binary * targets_binary).sum(dim=(1, 2, 3))
    union = (preds_binary + targets_binary).sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def train_validate_epoch(model, train_loader, val_loader, optimizer, criterion, device, scheduler=None):
    # --- Training Phase ---
    model.train()
    train_loss = 0.0
    train_iou = 0.0
    train_pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, masks in train_pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        batch_iou = compute_iou(outputs.detach(), masks).item()
        train_loss += batch_loss
        train_iou += batch_iou
        train_pbar.set_postfix(loss=f"{batch_loss:.4f}", iou=f"{batch_iou:.4f}")
    avg_train_loss = train_loss / len(train_loader)
    avg_train_iou = train_iou / len(train_loader)

    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_pbar = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for images, masks in val_pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            batch_loss = loss.item()
            batch_iou = compute_iou(outputs, masks).item()
            val_loss += batch_loss
            val_iou += batch_iou
            val_pbar.set_postfix(loss=f"{batch_loss:.4f}", iou=f"{batch_iou:.4f}")
    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)

    # --- Scheduler Step ---
    if scheduler:
         scheduler.step(avg_val_loss)

    return avg_train_loss, avg_train_iou, avg_val_loss, avg_val_iou

print("Metrics and Train/Validate loop defined.")


# ===========================================
# Step 7: Training Configuration and Execution - Loop through Experiments
# ===========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# --- Define Experiment Configurations ---
configurations = [
    {'backbone': 'resnet34', 'loss_fn_name': 'BCEDice'},
    {'backbone': 'resnet34', 'loss_fn_name': 'Lovasz'},
    {'backbone': 'resnet50', 'loss_fn_name': 'BCEDice'},
    {'backbone': 'resnet50', 'loss_fn_name': 'Lovasz'},
]

all_histories = {}
best_model_metrics = {}

# --- Create Directories ---
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- Loop Through Configurations ---
for config in configurations:
    backbone_name = config['backbone']
    loss_fn_name = config['loss_fn_name']
    config_name = f"{backbone_name}_{loss_fn_name}" 

    print(f"\n===== Starting Experiment: {config_name} =====")
    start_time = time.time()

    # --- Setup Model, Optimizer, Criterion, Scheduler ---
    set_seed(SEED) 
    model = UNet(backbone_name=backbone_name, pretrained=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if loss_fn_name == 'BCEDice':
        criterion = BCEDiceLoss().to(DEVICE)
    elif loss_fn_name == 'Lovasz':
        criterion = LovaszLoss().to(DEVICE)
    else:
        raise ValueError("Unsupported loss function")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True) # Schedule based on Val IoU

    # --- Training Loop for this Configuration ---
    current_best_val_iou = 0.0
    current_best_epoch = 0
    history = {'train_loss': [], 'train_iou': [], 'val_loss': [], 'val_iou': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"--- Epoch {epoch}/{NUM_EPOCHS} ({config_name}) ---")
        train_loss, train_iou, val_loss, val_iou = train_validate_epoch(
            model, train_loader, val_loader, optimizer, criterion, DEVICE, scheduler=None # Step scheduler manually based on IoU
        )

        # Store history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)

        print(f"Epoch {epoch} Summary - Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f} | Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")


        scheduler.step(val_iou)

        # Save the best model for this configuration
        if val_iou > current_best_val_iou:
            current_best_val_iou = val_iou
            current_best_epoch = epoch
            model_save_path = f"models/best_model_{config_name}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"✨ Best model for {config_name} saved to {model_save_path} (Val IoU: {current_best_val_iou:.4f})")

    end_time = time.time()
    print(f"--- Experiment {config_name} Finished ---")
    print(f"Best Validation IoU: {current_best_val_iou:.4f} at Epoch {current_best_epoch}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    # Store results for this configuration
    all_histories[config_name] = history
    best_model_metrics[config_name] = {'best_val_iou': current_best_val_iou, 'best_epoch': current_best_epoch}


# ===========================================
# Step 7b: Comparison Plotting
# ===========================================
print("\n--- Generating Comparison Plot ---")
plt.figure(figsize=(12, 7))

for config_name, history in all_histories.items():
    epochs_range = range(1, len(history['val_iou']) + 1)
    plt.plot(epochs_range, history['val_iou'], label=f"{config_name} (Best: {best_model_metrics[config_name]['best_val_iou']:.4f})")

plt.title('Validation IoU Comparison Across Experiments')
plt.xlabel('Epoch')
plt.ylabel('Validation IoU')
plt.legend(loc='lower right')
plt.grid(True)
plt.ylim(0, 1) # Set y-axis limits for IoU
plt.savefig("results/validation_iou_comparison.png")
plt.show()
print("Comparison plot saved to results/validation_iou_comparison.png")


# ===========================================
# Step 8: Prediction and Submission File Generation (Using Overall Best Model)
# ===========================================

# --- Find the Overall Best Model ---
overall_best_iou = -1.0
best_config_name = None
for config_name, metrics in best_model_metrics.items():
    if metrics['best_val_iou'] > overall_best_iou:
        overall_best_iou = metrics['best_val_iou']
        best_config_name = config_name

print(f"\n--- Overall Best Model based on Validation IoU: {best_config_name} (IoU: {overall_best_iou:.4f}) ---")

# --- Run-Length Encoding Function ---
def rle_encode(img):
    pixels = img.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# --- Prediction Function ---
def predict_test(model, test_loader, device, threshold=0.5):
    model.eval()
    predictions = {}
    test_pbar = tqdm(test_loader, desc="Predicting", leave=False)
    with torch.no_grad():
        for images, image_ids in test_pbar:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            probs_resized = F.interpolate(probs, size=(101, 101), mode='bilinear', align_corners=False)
            preds_binary = (probs_resized > threshold).cpu().numpy().astype(np.uint8)
            for i, img_id in enumerate(image_ids):
                 pred_mask = preds_binary[i].squeeze()
                 predictions[img_id] = pred_mask
    return predictions

# --- Load the Overall Best Model ---
# Re-instantiate the model architecture for the best config
best_backbone = best_config_name.split('_')[0]
best_model = UNet(backbone_name=best_backbone, pretrained=False).to(DEVICE) # No need for pretrained weights now

best_model_path = f"models/best_model_{best_config_name}.pth"
if os.path.exists(best_model_path):
    print(f"Loading overall best model from: {best_model_path}")
    best_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
else:
    print(f"Error: Best model path not found ({best_model_path}). Cannot generate submission.")
    # Handle error appropriately, maybe exit or use a default model if available

# --- Perform Prediction with Best Model ---
if os.path.exists(best_model_path):
    test_predictions = predict_test(best_model, test_loader, DEVICE, threshold=0.5) # Adjust threshold if needed

    # --- Generate Submission File ---
    submission_data = []
    for img_id in tqdm(test_ids, desc="Encoding"): # Use sorted test_ids
        if img_id in test_predictions:
            rle = rle_encode(test_predictions[img_id])
        else:
            rle = ''
            print(f"Warning: Prediction missing for image ID: {img_id}")
        submission_data.append({'id': img_id, 'rle_mask': rle})

    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv('submission.csv', index=False)

    print("\n--- Submission File Generated: submission.csv ---")
    print(submission_df.head())
else:
    print("Submission file not generated due to missing best model.")



print("\nZipping output files...")
if os.path.exists("models"):
    !zip -rq models.zip /kaggle/working/models # Use -q for quiet, -r for recursive
    print("models.zip created.")
if os.path.exists("results"):
    !zip -rq results.zip /kaggle/working/results
    print("results.zip created.")

print("\n--- Script Finished ---")

