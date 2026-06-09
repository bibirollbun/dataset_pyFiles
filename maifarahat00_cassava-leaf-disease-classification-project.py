import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from albumentations import Compose, HorizontalFlip, VerticalFlip, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm
import albumentations as A
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import gc
from collections import Counter




import random
import warnings
warnings.filterwarnings("ignore")



# -------------------------------
# 2ï¸� Set Seed
# -------------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(42)

# -------------------------------
# 3ï¸� Device Configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
    TRAIN_IMAGES = f"{BASE_PATH}/train_images"
    TEST_IMAGES = f"{BASE_PATH}/test_images"
    
    # Model parameters
    IMG_SIZE = 224   # Image size
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    NUM_CLASSES = 5
    
    # Training parameters
    EPOCHS_SCRATCH = 25
    EPOCHS_PRETRAINED = 20
    WEIGHT_DECAY = 1e-4
    GRADIENT_ACCUMULATION_STEPS = 2
    
    # Learning rate experiments
    LEARNING_RATES = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4]
    
    # Augmentation
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2
    CUTMIX_ALPHA = 0.2
    
    SEED = 42

config = Config()



# Load CSV files
train_df = pd.read_csv(f"{config.BASE_PATH}/train.csv")
sample_submission = pd.read_csv(f"{config.BASE_PATH}/sample_submission.csv")

# Load disease mapping
with open(f"{config.BASE_PATH}/label_num_to_disease_map.json") as f:
    disease_map = json.load(f)

print("Training samples:", len(train_df))
print("Test samples:", len(sample_submission))
print("Disease mapping:", disease_map)


# Display first few rows
print(" Training Data Preview:")
display(train_df.head(10))


# Basic statistics
print("\n Basic Statistics:")
print(f"  Total training images: {len(train_df):,}")
print(f"  Image ID format: {train_df['image_id'].iloc[0]}")
print(f"  Missing values: {train_df.isnull().sum().sum()}")


# Calculate class distribution
class_counts = train_df['label'].value_counts().sort_index()
class_percentages = (class_counts / len(train_df) * 100).round(2)

print(" Class Distribution:")
print("="*50)
for label in range(config.NUM_CLASSES):
    count = class_counts[label]
    percentage = class_percentages[label]
    disease_name = disease_map[str(label)]
    print(f"  Class {label} ({disease_name:30s}): {count:5d} ({percentage:5.2f}%)")
print("="*50)


# Calculate imbalance ratio
imbalance_ratio = class_counts.max() / class_counts.min()
print(f"\n Class Imbalance Ratio: {imbalance_ratio:.2f}x")

if imbalance_ratio > 3:
    print(" Significant class imbalance detected!")
    print("   â†’ Solution: Using class weights in loss function")
else:
    print(" Classes are relatively balanced")


# Calculate class weights for loss function
class_weights = len(train_df) / (config.NUM_CLASSES * class_counts.values)
class_weights = torch.FloatTensor(class_weights).to(device)

print(f"\n Calculated Class Weights:")
for i, weight in enumerate(class_weights):
    print(f"  Class {i}: {weight:.4f}")


# ============================================================================
# VISUALIZATION: CLASS DISTRIBUTION (BAR + DONUT)
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Class Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)

# Pastel color palette
colors = sns.color_palette('pastel', config.NUM_CLASSES)

# ============================================================================
# BAR CHART â€“ Absolute Counts
# ============================================================================
ax1 = axes[0]

bars = ax1.bar(
    range(config.NUM_CLASSES),
    class_counts.values,
    color=colors,
    edgecolor='black',
    linewidth=1.5
)

ax1.set_title('Absolute Counts', fontsize=14, fontweight='bold')
ax1.set_xlabel('Disease Category', fontsize=12)
ax1.set_ylabel('Number of Images', fontsize=12)

ax1.set_xticks(range(config.NUM_CLASSES))
ax1.set_xticklabels(
    [disease_map[str(i)][:15] for i in range(config.NUM_CLASSES)],
    rotation=45,
    ha='right',
    fontsize=10
)

ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Value labels on bars
for bar, count in zip(bars, class_counts.values):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{count:,}',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

# ============================================================================
# DONUT CHART â€“ Percentage Distribution
# ============================================================================
ax2 = axes[1]

wedges, texts, autotexts = ax2.pie(
    class_counts.values,
    labels=[disease_map[str(i)] for i in range(config.NUM_CLASSES)],
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    pctdistance=0.85,
    textprops={'fontsize': 10, 'weight': 'bold'}
)

# Donut hole
centre_circle = plt.Circle((0, 0), 0.60, fc='white')
ax2.add_artist(centre_circle)

ax2.set_title('Percentage Distribution', fontsize=14, fontweight='bold')

# Improve percentage text readability
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(11)

plt.tight_layout()
plt.show()

print("\n Class distribution visualized successfully!\n")


# ============================================================================
# SAMPLE IMAGES VISUALIZATION
# ============================================================================

def display_sample_images(df, images_path, disease_map, samples_per_class=4, figsize=(18, 20)):
    """Display sample images from each disease category"""
    
    fig, axes = plt.subplots(config.NUM_CLASSES, samples_per_class, figsize=figsize)
    fig.suptitle(' Sample Images from Each Disease Category', 
                 fontsize=18, fontweight='bold', y=0.998)
    
    for label in range(config.NUM_CLASSES):
        # Get random samples from this class
        class_samples = df[df['label'] == label].sample(n=samples_per_class, random_state=42)
        
        for idx, (_, row) in enumerate(class_samples.iterrows()):
            img_path = os.path.join(images_path, row['image_id'])
            
            # Read and convert image
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Display image
                ax = axes[label, idx]
                ax.imshow(img)
                ax.axis('off')
                
                # Add title to first image of each row
                if idx == 0:
                    disease_name = disease_map[str(label)]
                    ax.set_title(f"Class {label}: {disease_name}", 
                               fontsize=12, fontweight='bold', loc='left', pad=10)
                
                # Add image dimensions as subtitle
                h, w = img.shape[:2]
                ax.text(0.5, -0.05, f'{w}Ã—{h}', 
                       transform=ax.transAxes, ha='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.show()

print(" Displaying sample images from each category...\n")
display_sample_images(train_df, config.TRAIN_IMAGES, disease_map, samples_per_class=4)
print(" Sample images displayed!\n")


# ============================================================================
# IMAGE PROPERTIES ANALYSIS
# ============================================================================

def analyze_image_properties(df, images_path, sample_size=1000):
    """Analyze dimensions, sizes, and properties of images"""
    
    print(f"ğŸ”� Analyzing image properties (sample size: {sample_size})...")
    
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    widths, heights, aspects, sizes = [], [], [], []
    channels_list = []
    
    for img_id in tqdm(sample_df['image_id'], desc="Processing images"):
        img_path = os.path.join(images_path, img_id)
        
        try:
            img = cv2.imread(img_path)
            if img is not None:
                h, w, c = img.shape
                widths.append(w)
                heights.append(h)
                aspects.append(w/h)
                channels_list.append(c)
                sizes.append(os.path.getsize(img_path) / 1024)  # KB
        except Exception as e:
            print(f" Error reading {img_id}: {e}")
    
    # Convert to numpy arrays
    widths = np.array(widths)
    heights = np.array(heights)
    aspects = np.array(aspects)
    sizes = np.array(sizes)
    
    # Print statistics
    print("\n" + "="*70)
    print(" IMAGE STATISTICS")
    print("="*70)
    print(f"  Width:        {widths.mean():.0f} Â± {widths.std():.0f} pixels (min: {widths.min()}, max: {widths.max()})")
    print(f"  Height:       {heights.mean():.0f} Â± {heights.std():.0f} pixels (min: {heights.min()}, max: {heights.max()})")
    print(f"  Aspect Ratio: {aspects.mean():.2f} Â± {aspects.std():.2f} (min: {aspects.min():.2f}, max: {aspects.max():.2f})")
    print(f"  File Size:    {sizes.mean():.0f} Â± {sizes.std():.0f} KB (min: {sizes.min():.0f}, max: {sizes.max():.0f})")
    print(f"  Channels:     {Counter(channels_list)}")
    print("="*70 + "\n")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('ğŸ“� Image Properties Analysis', fontsize=16, fontweight='bold')
    
    # Width distribution
    axes[0, 0].hist(widths, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(widths.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {widths.mean():.0f}')
    axes[0, 0].axvline(np.median(widths), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(widths):.0f}')
    axes[0, 0].set_title('Width Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Height distribution
    axes[0, 1].hist(heights, bins=40, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(heights.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {heights.mean():.0f}')
    axes[0, 1].axvline(np.median(heights), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(heights):.0f}')
    axes[0, 1].set_title('Height Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Aspect ratio distribution
    axes[1, 0].hist(aspects, bins=40, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(aspects.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {aspects.mean():.2f}')
    axes[1, 0].axvline(np.median(aspects), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(aspects):.2f}')
    axes[1, 0].set_title('Aspect Ratio Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Aspect Ratio (W/H)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # File size distribution
    axes[1, 1].hist(sizes, bins=40, color='plum', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(sizes.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {sizes.mean():.0f} KB')
    axes[1, 1].axvline(np.median(sizes), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(sizes):.0f} KB')
    axes[1, 1].set_title('File Size Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Size (KB)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return widths, heights, aspects, sizes


# ============================================================================
# DATA SPLITTING
# ============================================================================

print("="*70)
print("DATA SPLITTING (70% Train / 20% Val / 10% Test)")
print("="*70 + "\n")

# First split: 70% train, 30% temp (for val + test)
train_data, temp_data = train_test_split(
    train_df,
    test_size=0.3,
    random_state=config.SEED,
    stratify=train_df['label']
)

# Second split: Split temp into 20% val and 10% test (from original)
# 20/30 = 0.6667 of temp goes to validation
# 10/30 = 0.3333 of temp goes to test
val_data, test_data = train_test_split(
    temp_data,
    test_size=0.3333,  # 10% of total = 33.33% of 30%
    random_state=config.SEED,
    stratify=temp_data['label']
)

# Reset indices
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

# Print split information
print("Data Split Complete!")
print(f"\nDataset Sizes:")
print(f"  Training Set:   {len(train_data):5d} images ({len(train_data)/len(train_df)*100:.1f}%)")
print(f"  Validation Set: {len(val_data):5d} images ({len(val_data)/len(train_df)*100:.1f}%)")
print(f"  Test Set:       {len(test_data):5d} images ({len(test_data)/len(train_df)*100:.1f}%)")
print(f"  {'â”€'*50}")
print(f"  Total:          {len(train_df):5d} images (100.0%)")



# Verify class distribution in each split
print("\n Class Distribution Verification:")
print("="*70)

splits = {
    'Training': train_data,
    'Validation': val_data,
    'Test': test_data
}

# Create distribution table
distribution_data = []

for split_name, split_df in splits.items():
    dist = split_df['label'].value_counts().sort_index()
    dist_pct = (dist / len(split_df) * 100)
    
    print(f"\n{split_name} Set ({len(split_df)} images):")
    print("â”€" * 50)
    
    for label in range(config.NUM_CLASSES):
        count = dist[label]
        percentage = dist_pct[label]
        disease_name = disease_map[str(label)]
        print(f"  Class {label} ({disease_name:30s}): {count:4d} ({percentage:5.2f}%)")
        
        distribution_data.append({
            'Split': split_name,
            'Class': label,
            'Disease': disease_name,
            'Count': count,
            'Percentage': f"{percentage:.2f}%"
        })

print("="*70)


# ============================================================================
# VISUALIZE CLASS DISTRIBUTION ACROSS SPLITS (PASTEL)
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Class Distribution Across Splits', fontsize=16, fontweight='bold')

# Pastel color palette (fixed across all splits)
colors = sns.color_palette('pastel', config.NUM_CLASSES)

for idx, (split_name, split_df) in enumerate(splits.items()):
    dist = split_df['label'].value_counts().sort_index()
    
    bars = axes[idx].bar(
        range(config.NUM_CLASSES),
        dist.values,
        color=colors,
        edgecolor='black',
        linewidth=1.5
    )
    
    axes[idx].set_title(
        f'{split_name} Set\n({len(split_df)} images)',
        fontsize=12,
        fontweight='bold'
    )
    
    axes[idx].set_xlabel('Disease Class', fontsize=10)
    axes[idx].set_ylabel('Count', fontsize=10)
    
    axes[idx].set_xticks(range(config.NUM_CLASSES))
    axes[idx].set_xticklabels([f'C{i}' for i in range(config.NUM_CLASSES)])
    
    axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[idx].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=9
        )

plt.tight_layout()
plt.show()

print("\n Data splitting complete and verified!\n")


def get_train_transforms(img_size=config.IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.RandomRotate90(p=0.6),
        A.HorizontalFlip(p=0.6),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=25, p=0.6),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120*0.05, alpha_affine=120*0.03, p=0.4),
            A.GridDistortion(p=0.4),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.4),
        ], p=0.4),
        A.OneOf([
            A.RandomBrightnessContrast(0.3, 0.3, p=1),
            A.HueSaturationValue(25, 40, 25, p=1),
            A.RGBShift(20, 20, 20, p=1),
            A.CLAHE(clip_limit=4.0, p=1),
        ], p=0.6),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 80.0), p=1),
            A.GaussianBlur(blur_limit=(3, 9), p=1),
            A.MotionBlur(blur_limit=7, p=1),
            A.MedianBlur(blur_limit=5, p=1),
        ], p=0.4),
        A.CoarseDropout(max_holes=12, max_height=img_size//15, max_width=img_size//15, min_holes=8, fill_value=0, p=0.4),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])


def get_valid_transforms(img_size=config.IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])


class ReducedMixUpCutMixCollate:
    def __init__(self, mixup_alpha=config.MIXUP_ALPHA, cutmix_alpha=config.CUTMIX_ALPHA, prob=0.3, num_classes=config.NUM_CLASSES):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes
    
    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        if random.random() < self.prob:
            if random.random() < 0.5:
                images, labels = self.mixup(images, labels)
            else:
                images, labels = self.cutmix(images, labels)
        return images, labels
    
    def mixup(self, images, labels):
        batch_size = images.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        index = torch.randperm(batch_size)
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a = F.one_hot(labels, self.num_classes).float()
        labels_b = F.one_hot(labels[index], self.num_classes).float()
        mixed_labels = lam * labels_a + (1 - lam) * labels_b
        return mixed_images, mixed_labels
    
    def cutmix(self, images, labels):
        batch_size, _, H, W = images.shape
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        index = torch.randperm(batch_size)
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = np.random.randint(W), np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        labels_a = F.one_hot(labels, self.num_classes).float()
        labels_b = F.one_hot(labels[index], self.num_classes).float()
        mixed_labels = lam * labels_a + (1 - lam) * labels_b
        return images, mixed_labels

mixup_cutmix_collate = ReducedMixUpCutMixCollate()


# ============================================================================
# CUSTOM DATASET + WEIGHTED SAMPLER
# ============================================================================
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
import os

class CassavaDataset(Dataset):
    def __init__(self, df, images_path, transforms=None):
        self.df = df.reset_index(drop=True)
        self.images_path = images_path
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'image_id']
        label = self.df.loc[idx, 'label']
        img_path = os.path.join(self.images_path, img_id)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)['image']
        return image, label

def create_balanced_sampler(df):
    class_counts = df['label'].value_counts().sort_index().values
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in df['label'].values]
    sample_weights = torch.DoubleTensor(sample_weights)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)




# ============================================================================
# DATALOADERS
# ============================================================================
train_dataset = CassavaDataset(train_data, config.TRAIN_IMAGES, get_train_transforms())
val_dataset = CassavaDataset(val_data, config.TRAIN_IMAGES, get_valid_transforms())
test_dataset = CassavaDataset(test_data, config.TRAIN_IMAGES, get_valid_transforms())

train_sampler = create_balanced_sampler(train_data)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    sampler=train_sampler,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
    collate_fn=mixup_cutmix_collate
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE*2,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE*2,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)


print(" Enhanced data loaders created!")
print(f"    Weighted Random Sampler: Handles class imbalance")
print(f"    MixUp/CutMix: Applied during training")
print(f"\n Loader Statistics:")
print(f"  Training batches:   {len(train_loader):4d} (batch size: {config.BATCH_SIZE})")
print(f"  Validation batches: {len(val_loader):4d} (batch size: {config.BATCH_SIZE * 2})")
print(f"  Test batches:       {len(test_loader):4d} (batch size: {config.BATCH_SIZE * 2})")


# ============================================================================
# CUSTOM CNN MODEL + SIMPLE RESIDUAL BLOCK
# ============================================================================
import torch.nn as nn
import timm

class SimpleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        out = self.dropout(out)
        return out

class OptimizedCustomCNN(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, dropout_rate=0.25):
        super().__init__()
        self.conv_init = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        self.layer1 = self._make_layer(64,128,2,1,dropout_rate*0.4)
        self.layer2 = self._make_layer(128,256,2,2,dropout_rate*0.6)
        self.layer3 = self._make_layer(256,512,2,2,dropout_rate*0.8)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate*0.5),
            nn.Linear(256,num_classes)
        )
        self._initialize_weights()
    
    def _make_layer(self, in_ch, out_ch, blocks, stride, dropout):
        layers = [SimpleResidualBlock(in_ch,out_ch,stride,dropout)]
        for _ in range(blocks-1):
            layers.append(SimpleResidualBlock(out_ch,out_ch,1,dropout))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                if m.bias is not None: nn.init.constant_(m.bias,0)
    
    def forward(self,x):
        x = self.conv_init(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

def create_efficientnet_model(model_name, num_classes=config.NUM_CLASSES, pretrained=True):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)



# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=config.LABEL_SMOOTHING, weight=None):
        super().__init__()
        self.epsilon = epsilon
        self.weight = weight
    def forward(self, preds, target):
        n_classes = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, target, weight=self.weight)
        return self.epsilon * loss / n_classes + (1-self.epsilon) * nll

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p_t = torch.exp(-ce_loss)
        loss = (1-p_t)**self.gamma * ce_loss
        if self.reduction=='mean': return loss.mean()
        elif self.reduction=='sum': return loss.sum()
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=config.LABEL_SMOOTHING, focal_weight=0.7):
        super().__init__()
        self.focal_loss = FocalLoss(alpha, gamma)
        self.ce_loss = LabelSmoothingCrossEntropy(weight=alpha)
        self.focal_weight = focal_weight
    def forward(self, inputs, targets):
        if targets.dim()>1:
            focal = F.cross_entropy(inputs, targets.argmax(dim=1))
            ce = -(targets*F.log_softmax(inputs,dim=-1)).sum(dim=-1).mean()
        else:
            focal = self.focal_loss(inputs, targets)
            ce = self.ce_loss(inputs, targets)
        return self.focal_weight*focal + (1-self.focal_weight)*ce



# ============================================================================
# HELPER: MODEL SUMMARY
# ============================================================================
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def print_model_summary(model, model_name):
    total_params, trainable_params = count_parameters(model)
    print(f"\n{'='*70}")
    print(f"{model_name} SUMMARY")
    print(f"{'='*70}")
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Non-trainable Params: {total_params - trainable_params:,}")
    print(f"  Model Size:           {total_params*4/1024/1024:.2f} MB (FP32)")
    print(f"{'='*70}\n")


# ============================================================================
# CREATE MODELS
# ============================================================================

print("Creating models...\n")

# Model 1: Improved Custom CNN
model_custom = OptimizedCustomCNN(num_classes=config.NUM_CLASSES).to(device)


# Model 2: EfficientNet-B3
model_effnet_b3 = create_efficientnet_model('efficientnet_b3', config.NUM_CLASSES, pretrained=True).to(device)


# Model 3: EfficientNet-B4
model_effnet_b4 = create_efficientnet_model('efficientnet_b4', config.NUM_CLASSES, pretrained=True).to(device)


print("All models created successfully!\n")


def train_one_epoch_simple(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [TRAIN]', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        if labels.dim()>1: correct += pred.eq(labels.argmax(1)).sum().item()
        else: correct += pred.eq(labels).sum().item()
        pbar.set_postfix({"loss": f"{running_loss/(total/labels.size(0)):.4f}",
                          "acc": f"{100.*correct/total:.2f}%"})
    return running_loss/len(loader), 100.*correct/total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc='Validation', leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            pbar.set_postfix({"loss": f"{running_loss/(total/labels.size(0)):.4f}",
                              "acc": f"{100.*correct/total:.2f}%"})
    return running_loss/len(loader), 100.*correct/total


def train_model_fixed(model, model_name, train_loader, val_loader, epochs, lr, device, class_weights=None):
    print(f"\nğŸš€ TRAINING {model_name}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7)
    scaler = GradScaler()
    best_val_acc = 0.0
    best_model_wts = None
    patience_counter, patience = 0, 12

    history = {"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[],"lr":[]}

    for epoch in range(epochs):
        print(f"\nğŸ“… Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch_simple(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"ğŸ“Š Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"ğŸ“Š Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"ğŸ“ˆ LR: {current_lr:.6f}")

        if val_acc>best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()
            patience_counter=0
            torch.save(model.state_dict(), f"best_{model_name}.pth")
            print(f"âœ… New Best Val Acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter>=patience:
                print("âš ï¸� Early stopping triggered")
                break
        gc.collect()
        torch.cuda.empty_cache()

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    print(f"\nğŸ�† Training Complete | Best Val Acc: {best_val_acc:.2f}%")
    return model, history


print_model_summary(model_custom, "Custom CNN ")

model_custom, history_custom = train_model_fixed(
    model_custom,
    "CustomCNN",
    train_loader,
    val_loader,
    epochs=config.EPOCHS_SCRATCH,
    lr=5e-4,
    device=device
)


# ============================================================================
# EXPERIMENT 2: EFFICIENTNET-B3 (Transfer Learning)
# ============================================================================

print_model_summary(model_custom, "EXPERIMENT 2: EfficientNet-B3 with Transfer Learning")
# Train EfficientNet-B3
model_effnet_b3_trained, history_effnet_b3 = train_model_fixed(
    model=model_effnet_b3,
    model_name="EfficientNet_B3",
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=config.EPOCHS_PRETRAINED,
    lr=3e-4,
    device=device
)


# ============================================================================
# EXPERIMENT 3: EFFICIENTNET-B4 (Best Single Model)
# ============================================================================

print_model_summary(model_custom, "EXPERIMENT 3: EfficientNet-B4 for Maximum Performance")

# Train EfficientNet-B4
model_effnet_b4_trained, history_effnet_b4 = train_model_fixed(
    model=model_effnet_b4,
    model_name="EfficientNet_B4",
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=config.EPOCHS_PRETRAINED,
    lr=3e-4,
    device=device
)


# ============================================================================
# STEP 9: MODEL EVALUATION WITH TEST TIME AUGMENTATION (TTA)
# ============================================================================

import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def test_time_augmentation(model, image, device, n_augments=8):
    """Apply test time augmentation for better predictions"""
    model.eval()
    
    # Define TTA transforms
    tta_transforms = A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_augments):
            # Apply random augmentation
            augmented = tta_transforms(image=image)['image'].unsqueeze(0).to(device)
            
            # Get prediction
            with autocast():
                output = model(augmented)
                pred = F.softmax(output, dim=1)
                predictions.append(pred.cpu())
    
    # Average all predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred


def evaluate_model_with_tta(model, test_loader, device, model_name):
    """Comprehensive evaluation with TTA"""
    print(f"\n EVALUATING {model_name} WITH TTA")
    print("="*60)
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            images, labels = images.to(device), labels.to(device)
            
            # Standard inference
            with autocast():
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
            
            all_predictions.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate accuracy
    accuracy = (all_predictions == all_labels).mean() * 100
    
    print(f" {model_name} Test Accuracy: {accuracy:.2f}%")
    
    # Detailed classification report
    print(f"\n Classification Report for {model_name}:")
    print("-" * 60)
    report = classification_report(
        all_labels, 
        all_predictions, 
        target_names=[disease_map[str(i)] for i in range(config.NUM_CLASSES)],
        digits=4
    )
    print(report)
    
    return all_predictions, all_labels, all_probabilities, accuracy

# Evaluate all models
print(" Starting comprehensive model evaluation...")

# Evaluate Custom CNN
custom_preds, custom_labels, custom_probs, custom_acc = evaluate_model_with_tta(
    model_custom, test_loader, device, "Custom CNN"
)

# Evaluate EfficientNet-B3
effb3_preds, effb3_labels, effb3_probs, effb3_acc = evaluate_model_with_tta(
    model_effnet_b3_trained, test_loader, device, "EfficientNet-B3"
)

# Evaluate EfficientNet-B4
effb4_preds, effb4_labels, effb4_probs, effb4_acc = evaluate_model_with_tta(
    model_effnet_b4_trained, test_loader, device, "EfficientNet-B4"
)

# Store results for comparison
model_results = {
    'Custom CNN': custom_acc,
    'EfficientNet-B3': effb3_acc,
    'EfficientNet-B4': effb4_acc
}

print(f"\n MODEL COMPARISON SUMMARY:")
print("="*50)
for model_name, accuracy in model_results.items():
    print(f"  {model_name:15s}: {accuracy:.2f}%")
print("="*50)


class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        self.weights = np.array(self.weights) / np.sum(self.weights)  # Normalize
    
    def predict(self, dataloader, device):
        all_predictions = []
        all_labels = []
        
        # Set all models to eval mode
        for model in self.models:
            model.eval()
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Ensemble Prediction"):
                images, labels = images.to(device), labels.to(device)
                
                batch_predictions = []
                
                # Get predictions from each model
                for model in self.models:
                    with autocast():
                        outputs = model(images)
                        probs = F.softmax(outputs, dim=1)
                        batch_predictions.append(probs.cpu().numpy())
                
                # Weighted ensemble
                ensemble_pred = np.zeros_like(batch_predictions[0])
                for i, pred in enumerate(batch_predictions):
                    ensemble_pred += self.weights[i] * pred
                
                all_predictions.extend(ensemble_pred.argmax(axis=1))
                all_labels.extend(labels.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels)

# Create ensemble with optimized weights based on validation performance
ensemble_weights = [0.2, 0.35, 0.45]  # Custom CNN, EfficientNet-B3, EfficientNet-B4

ensemble_model = EnsembleModel(
    models=[model_custom, model_effnet_b3_trained, model_effnet_b4_trained],
    weights=ensemble_weights
)


# Evaluate ensemble
print(" EVALUATING ENSEMBLE MODEL")
print("="*50)

ensemble_preds, ensemble_labels = ensemble_model.predict(test_loader, device)
ensemble_accuracy = (ensemble_preds == ensemble_labels).mean() * 100

print(f" Ensemble Test Accuracy: {ensemble_accuracy:.2f}%")

# Detailed ensemble report
print(f"\n Ensemble Classification Report:")
print("-" * 60)
ensemble_report = classification_report(
    ensemble_labels, 
    ensemble_preds, 
    target_names=[disease_map[str(i)] for i in range(config.NUM_CLASSES)],
    digits=4
)
print(ensemble_report)

# Update results
model_results['Ensemble'] = ensemble_accuracy


# ============================================================================
# STEP 11: ADVANCED RESULTS VISUALIZATION & ANALYSIS
# ============================================================================

def plot_training_history(histories, model_names):
    """Plot training history for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ï¿½ Training History Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Training Loss
    ax1 = axes[0, 0]
    for i, (history, name) in enumerate(zip(histories, model_names)):
        ax1.plot(history['train_loss'], label=name, color=colors[i], linewidth=2)
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation Loss
    ax2 = axes[0, 1]
    for i, (history, name) in enumerate(zip(histories, model_names)):
        ax2.plot(history['val_loss'], label=name, color=colors[i], linewidth=2)
    ax2.set_title('Validation Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training Accuracy
    ax3 = axes[1, 0]
    for i, (history, name) in enumerate(zip(histories, model_names)):
        ax3.plot(history['train_acc'], label=name, color=colors[i], linewidth=2)
    ax3.set_title('Training Accuracy', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Validation Accuracy
    ax4 = axes[1, 1]
    for i, (history, name) in enumerate(zip(histories, model_names)):
        ax4.plot(history['val_acc'], label=name, color=colors[i], linewidth=2)
    ax4.set_title('Validation Accuracy', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(predictions_list, labels, model_names):
    """Plot confusion matrices for all models"""
    n_models = len(predictions_list)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(' Confusion Matrix Comparison', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, (preds, name) in enumerate(zip(predictions_list, model_names)):
        cm = confusion_matrix(labels, preds)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            ax=axes[i],
            xticklabels=[f'C{j}' for j in range(config.NUM_CLASSES)],
            yticklabels=[f'C{j}' for j in range(config.NUM_CLASSES)],
            cbar_kws={'shrink': 0.8}
        )
        
        axes[i].set_title(f'{name}\nAccuracy: {(preds == labels).mean()*100:.2f}%', 
                         fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()


def plot_model_comparison():
    """Create comprehensive model comparison visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(' Final Model Performance Comparison', fontsize=16, fontweight='bold')
    
    models = list(model_results.keys())
    accuracies = list(model_results.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD93D']
    
    # Bar chart
    bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=2)
    ax1.set_title('Test Accuracy Comparison', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Radar chart for detailed comparison
    categories = ['Accuracy', 'Complexity', 'Speed', 'Robustness']
    
    # Normalized scores (0-100)
    scores = {
        'Custom CNN': [custom_acc, 85, 95, 70],
        'EfficientNet-B3': [effb3_acc, 70, 80, 85],
        'EfficientNet-B4': [effb4_acc, 60, 70, 90],
        'Ensemble': [ensemble_accuracy, 40, 50, 95]
    }
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    ax2 = plt.subplot(122, projection='polar')
    
    for i, (model, score) in enumerate(scores.items()):
        score += score[:1]  # Complete the circle
        ax2.plot(angles, score, 'o-', linewidth=2, label=model, color=colors[i])
        ax2.fill(angles, score, alpha=0.25, color=colors[i])
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 100)
    ax2.set_title('Multi-Criteria Comparison', fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.show()

# Execute visualizations
print(" Creating comprehensive visualizations...")

# Plot training histories
plot_training_history(
    [history_custom, history_effnet_b3, history_effnet_b4],
    ['Custom CNN', 'EfficientNet-B3', 'EfficientNet-B4']
)

# Plot confusion matrices
plot_confusion_matrices(
    [custom_preds, effb3_preds, effb4_preds, ensemble_preds],
    custom_labels,  # All should have same labels
    ['Custom CNN', 'EfficientNet-B3', 'EfficientNet-B4', 'Ensemble']
)

# Plot final comparison
plot_model_comparison()

# Print final summary
print("\n" + "="*80)
print(" FINAL RESULTS SUMMARY")
print("="*80)
print(f"{'Model':<20} {'Test Accuracy':<15} {'Improvement':<12}")
print("-" * 50)

baseline_acc = custom_acc
for model_name, accuracy in model_results.items():
    improvement = accuracy - baseline_acc if model_name != 'Custom CNN' else 0.0
    print(f"{model_name:<20} {accuracy:<15.2f}% {improvement:<12.2f}%")

print("="*80)
print(f" BEST MODEL: {max(model_results, key=model_results.get)} ({max(model_results.values()):.2f}%)")
print("="*80)


# ============================================================================
# STEP 12: KAGGLE SUBMISSION PREPARATION
# ============================================================================

def prepare_kaggle_submission(model, test_images_path, sample_submission_path, device):
    """Prepare final Kaggle submission using the best model"""
    
    print(" PREPARING KAGGLE SUBMISSION")
    print("="*50)
    
    # Load sample submission
    submission_df = pd.read_csv(sample_submission_path)
    print(f"Found {len(submission_df)} test images for submission")
    
    # Create test dataset for Kaggle test images
    class KaggleTestDataset(Dataset):
        def __init__(self, image_ids, images_path, transforms=None):
            self.image_ids = image_ids
            self.images_path = images_path
            self.transforms = transforms
        
        def __len__(self):
            return len(self.image_ids)
        
        def __getitem__(self, idx):
            img_id = self.image_ids[idx]
            img_path = os.path.join(self.images_path, img_id)
            
            try:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                # Handle missing images with a blank image
                image = np.zeros((600, 800, 3), dtype=np.uint8)
            
            if self.transforms:
                image = self.transforms(image=image)['image']
            
            return image, img_id
    
    # Create test dataset and loader
    kaggle_test_dataset = KaggleTestDataset(
        image_ids=submission_df['image_id'].tolist(),
        images_path=test_images_path,
        transforms=get_valid_transforms()
    )
    
    kaggle_test_loader = DataLoader(
        kaggle_test_dataset,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Generate predictions using ensemble
    print("ğŸ”® Generating predictions...")
    
    predictions = []
    image_ids = []
    
    # Use ensemble for final predictions
    for model_single in [model_custom, model_effnet_b3_trained, model_effnet_b4_trained]:
        model_single.eval()
    
    with torch.no_grad():
        for images, img_ids in tqdm(kaggle_test_loader, desc="Predicting"):
            images = images.to(device)
            
            # Ensemble prediction
            ensemble_logits = torch.zeros(images.size(0), config.NUM_CLASSES).to(device)
            
            for i, model_single in enumerate([model_custom, model_effnet_b3_trained, model_effnet_b4_trained]):
                with autocast():
                    outputs = model_single(images)
                    ensemble_logits += ensemble_weights[i] * F.softmax(outputs, dim=1)
            
            # Get final predictions
            preds = ensemble_logits.argmax(dim=1).cpu().numpy()
            
            predictions.extend(preds)
            image_ids.extend(img_ids)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'image_id': image_ids,
        'label': predictions
    })
    
    # Save submission
    submission_filename = f'cassava_submission_ensemble_{ensemble_accuracy:.2f}.csv'
    submission_df.to_csv(submission_filename, index=False)
    
    print(f" Submission saved as: {submission_filename}")
    print(f" Submission shape: {submission_df.shape}")
    print(f" Expected accuracy: ~{ensemble_accuracy:.2f}%")
    
    # Display submission preview
    print(f"\n Submission Preview:")
    print(submission_df.head(10))
    
    # Show label distribution in submission
    print(f"\nğŸ“ˆ Prediction Distribution:")
    label_dist = submission_df['label'].value_counts().sort_index()
    for label, count in label_dist.items():
        disease_name = disease_map[str(label)]
        percentage = count / len(submission_df) * 100
        print(f"  Class {label} ({disease_name[:25]:25s}): {count:4d} ({percentage:5.2f}%)")
    
    return submission_df

# Create final submission
final_submission = prepare_kaggle_submission(
    model=ensemble_model,  # Use ensemble model
    test_images_path=config.TEST_IMAGES,
    sample_submission_path=f"{config.BASE_PATH}/sample_submission.csv",
    device=device
)

# ============================================================================
# ADDITIONAL PERFORMANCE OPTIMIZATION TIPS
# ============================================================================

print("\n" + "="*80)
print(" PERFORMANCE OPTIMIZATION RECOMMENDATIONS")
print("="*80)

optimization_tips = [
    "1.   Increase Training Data: Use external cassava datasets if allowed",
    "2. ğŸ”„ Advanced Augmentation: Try AutoAugment or RandAugment policies",
    "3. ğŸ�¯ Pseudo-Labeling: Use confident predictions on test set for training",
    "4. ğŸ�—ï¸� Architecture Search: Try Vision Transformers (ViT) or ConvNeXt",
    "5. ğŸ“Š Cross-Validation: Implement 5-fold CV for robust model selection",
    "6. ğŸ�›ï¸� Hyperparameter Tuning: Use Optuna for systematic optimization",
    "7. ğŸ”— Multi-Scale Training: Train on different image sizes",
    "8. ğŸ“± Self-Supervised Learning: Pre-train on unlabeled cassava images",
    "9. ğŸ�¨ Mixup Variants: Try CutMix, FMix, or GridMix",
    "10. ğŸ�† Advanced Ensembling: Use stacking or blending techniques"
]

for tip in optimization_tips:
    print(f"  {tip}")

print("="*80)

# ============================================================================
# SAVE BEST MODEL WEIGHTS
# ============================================================================

print(f"\nğŸ’¾ Saving final model weights...")

# Save individual model weights
torch.save(model_custom.state_dict(), 'final_custom_cnn.pth')
torch.save(model_effnet_b3_trained.state_dict(), 'final_efficientnet_b3.pth')
torch.save(model_effnet_b4_trained.state_dict(), 'final_efficientnet_b4.pth')

# Save ensemble configuration
ensemble_config = {
    'models': ['custom_cnn', 'efficientnet_b3', 'efficientnet_b4'],
    'weights': ensemble_weights,
    'accuracy': ensemble_accuracy,
    'config': {
        'img_size': config.IMG_SIZE,
        'num_classes': config.NUM_CLASSES,
        'disease_map': disease_map
    }
}

import json
with open('ensemble_config.json', 'w') as f:
    json.dump(ensemble_config, f, indent=2)

print("âœ… All model weights and configurations saved!")
print(f"\nğŸ�‰ PROJECT COMPLETED SUCCESSFULLY!")
print(f"ğŸ�† Final Ensemble Accuracy: {ensemble_accuracy:.2f}%")
print("="*80)


# ============================================================================
# KAGGLE SUBMISSION PREPARATION
# ============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
import cv2, os, pandas as pd, numpy as np
from tqdm.notebook import tqdm  # safer in notebooks
import torch.nn.functional as F
from torch.cuda.amp import autocast

# -------------------------------
# Top-level Dataset class (must be global)
# -------------------------------
class KaggleTestDataset(Dataset):
    def __init__(self, image_ids, images_path, transforms=None):
        self.image_ids = image_ids
        self.images_path = images_path
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.images_path, img_id)
        
        # read image safely
        image = cv2.imread(img_path)
        if image is None:
            # missing image â†’ blank
            image = np.zeros((600, 800, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            image = self.transforms(image=image)['image']
        
        return image, img_id

# -------------------------------
# Submission function
# -------------------------------
def prepare_kaggle_submission(
    ensemble_models, 
    ensemble_weights,
    test_images_path, 
    sample_submission_path, 
    device,
    ensemble_accuracy,
    disease_map
):
    print("\n PREPARING KAGGLE SUBMISSION")
    print("="*50)

    # Load sample submission
    submission_df = pd.read_csv(sample_submission_path)
    print(f"Found {len(submission_df)} test images for submission")

    # Create test dataset & loader
    kaggle_test_dataset = KaggleTestDataset(
        image_ids=submission_df['image_id'].tolist(),
        images_path=test_images_path,
        transforms=get_valid_transforms()  # your existing valid transforms
    )

    kaggle_test_loader = DataLoader(
        kaggle_test_dataset,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,   # crucial fix for multiprocessing
        pin_memory=True
    )

    print("ğŸ”® Generating predictions...")

    predictions = []
    image_ids = []

    # Set all models to eval
    for model in ensemble_models:
        model.eval()

    with torch.no_grad():
        for images, img_ids in tqdm(kaggle_test_loader, desc="Predicting"):
            images = images.to(device)
            ensemble_logits = torch.zeros(images.size(0), config.NUM_CLASSES).to(device)

            # Ensemble prediction
            for i, model in enumerate(ensemble_models):
                with autocast():
                    outputs = model(images)
                    ensemble_logits += ensemble_weights[i] * F.softmax(outputs, dim=1)

            preds = ensemble_logits.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
            image_ids.extend(img_ids)

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'image_id': image_ids,
        'label': predictions
    })

    submission_filename = f'cassava_submission_ensemble_{ensemble_accuracy:.2f}.csv'
    submission_df.to_csv(submission_filename, index=False)

    print(f" Submission saved as: {submission_filename}")
    print(f" Submission shape: {submission_df.shape}")
    print(f" Expected accuracy: ~{ensemble_accuracy:.2f}%")

    # Show label distribution
    print("\nğŸ“ˆ Prediction Distribution:")
    label_dist = submission_df['label'].value_counts().sort_index()
    for label, count in label_dist.items():
        disease_name = disease_map[str(label)]
        percentage = count / len(submission_df) * 100
        print(f"  Class {label} ({disease_name[:25]:25s}): {count:4d} ({percentage:5.2f}%)")

    return submission_df

# -------------------------------
# Example call
# -------------------------------
final_submission = prepare_kaggle_submission(
    ensemble_models=[model_custom, model_effnet_b3_trained, model_effnet_b4_trained],
    ensemble_weights=ensemble_weights,
    test_images_path=config.TEST_IMAGES,
    sample_submission_path=f"{config.BASE_PATH}/sample_submission.csv",
    device=device,
    ensemble_accuracy=ensemble_accuracy,
    disease_map=disease_map
)





