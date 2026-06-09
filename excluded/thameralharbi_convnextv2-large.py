# === Standard Python Libraries ===
import os                                # Operating system file/directory handling
import re                                # Regular expression operations
import json                              # Read/write JSON files
import time                              # Time-related utilities
import random                            # Random number generation
import warnings                          # Warning filtering and control
import sys                               # System-specific functions and parameters
from collections import Counter          # Frequency counting
import collections.abc as container_abcs # Abstract container base classes

# === Data Handling ===
import numpy as np                       # Numerical computations and array operations
import pandas as pd                      # Data manipulation and analysis
from pathlib import Path                 # Filesystem path management

# === Visualization ===
import matplotlib.pyplot as plt          # Plotting and visualization
import seaborn as sns                    # Advanced visualizations and statistical plots
import matplotlib.patches as patches     # Shape drawing (e.g., rectangles for bounding boxes)

# === Image Processing ===
from PIL import Image                    # Image loading and manipulation
import cv2                               # OpenCV for image processing tasks
from torchvision import transforms       # PyTorch-compatible image transformations

# === PyTorch Core ===
import torch                             # Main PyTorch library
import torch.nn as nn                    # Neural network layers and modules
import torch.nn.functional as F          # Functional layer operations (e.g., activations, losses)
import torch.optim as optim              # Optimizers like SGD, Adam, etc.
from torch.optim import AdamW            # AdamW optimizer with weight decay
from torch.utils.data import Dataset, DataLoader  # Dataset and DataLoader utilities

# === Hugging Face Transformers ===
from transformers import (
    ViTFeatureExtractor,                 # Feature extractor for Vision Transformer
    ViTForImageClassification,           # Vision Transformer model for classification
    ConvNextForImageClassification,      # ConvNeXt model for image classification
    AutoFeatureExtractor,                # Automatically choose the correct feature extractor
    TrainingArguments,                   # Training configuration
    Trainer                              # High-level training/evaluation loop
)

# === Hugging Face Datasets ===
from datasets import load_dataset, Dataset as HFDataset  # Load and work with datasets

# === Data Augmentation ===
import albumentations as A                     # Flexible image augmentation
from albumentations.pytorch import ToTensorV2  # Convert images to PyTorch tensors

# === Scikit-learn Utilities ===
from sklearn.model_selection import train_test_split, StratifiedKFold  # Data splitting strategies
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix # Evaluation metrics
from sklearn.utils.class_weight import compute_class_weight            # Handle class imbalance

# === Additional Libraries ===
from timm.models.layers import trunc_normal_, DropPath  # Helper layers from the timm library

# === Transformers Output Types ===
from transformers.modeling_outputs import SequenceClassifierOutput  # Output structure for classifier models

# === Progress Bar ===
from tqdm.auto import tqdm                          # Smart progress bars for loops



# Data paths
TRAIN_LABELS_PATH = Path('/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv')
TRAIN_DIR = Path('/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train')
# TRAIN_DIR = Path("/kaggle/working/train")
TEST_DIR = Path('/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test')
DUMMY_SUB_PATH = Path('/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/dummy_sub.csv')
SEED = 2025


# Set seed for reproducibility
def set_seed(seed=2025):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)


# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Number of available GPUs: {torch.cuda.device_count()}")

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')

set_seed(SEED)
print(f"Seed: {SEED}")


# Load training labels
train_labels = pd.read_csv(TRAIN_LABELS_PATH)


# import pandas as pd
# from pathlib import Path
# import shutil

# # 1. Define directories
# working_train_dir = Path("train")

# # 2. Create the working train directory (with class subfolders)
# working_train_dir.mkdir(parents=True, exist_ok=True)

# # 3. Load the original training labels
# # train_df = pd.read_csv("/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv")


# # 4. Copy original training images into /kaggle/working/train/{label}/
# for _, row in train_labels.iterrows():
#     # label = row["label"]
#     src = TRAIN_DIR / Path(row["filename"])
#     dst_dir = working_train_dir
#     dst_dir.mkdir(exist_ok=True)
#     shutil.copy(src, dst_dir / Path(row["filename"]))

# # 5. Load the two pseudo label prediction files
# df1 = pd.read_csv("/kaggle/input/pseudo-labelling-sheep-test-data/submission (73).csv") # google/vit-base-patch16-224-in21k + Pseudo labelling
# df2 = pd.read_csv("/kaggle/input/pseudo-labelling-sheep-test-data/submission (84).csv") # ConvNextV2 CV 5-Folds

# df1 = df1.rename(columns={"label": "label_1"})
# df2 = df2.rename(columns={"label": "label_2"})
# merged = pd.merge(df1, df2, on="filename")

# # 6. Keep only the images where both predictions agree
# agreed = merged[merged["label_1"] == merged["label_2"]]
# pseudo_labeled = agreed[["filename", "label_1"]].rename(columns={"label_1": "label"})

# # 7. Copy agreed pseudo-labeled test images into /kaggle/working/train/{label}/
# for _, row in pseudo_labeled.iterrows():
#     # label = row["label"]
#     src = TEST_DIR / Path(row["filename"])
#     dst_dir = working_train_dir
#     dst_dir.mkdir(exist_ok=True)
#     shutil.copy(src, dst_dir / row["filename"])

# print(f"âœ… All training and pseudo-labeled images are now in: {working_train_dir}")


# # Merge with original training data
# train_labels = pd.concat([train_labels, pseudo_labeled], ignore_index=True)

# print(f"âœ… {len(pseudo_labeled)} images were copied from the test set to the train set and added to the training data.")

# TRAIN_DIR = working_train_dir


# Create label-to-index and index-to-label mappings
label_to_idx = {label: idx for idx, label in enumerate(sorted(train_labels['label'].unique()))}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
NUM_CLASSES = len(label_to_idx)


class_counts = train_labels['label'].value_counts()
print("\nClass distribution statistics:")
print(class_counts)


train_labels["label"].value_counts().plot(kind="bar")


def load_and_analyze_image(img_path):
    """Load and return image dimensions and basic stats"""
    img = Image.open(img_path)
    return img.size, np.array(img).mean()

# Analyze a sample of images
sample_size = len(train_labels)
sample_images = train_labels['filename'].sample(sample_size)

# Collect image statistics
image_sizes = []
image_means = []

print("Analyzing sample images...")
for img_file in sample_images:
    img_path = TRAIN_DIR / img_file
    if img_path.exists():
        size, mean_val = load_and_analyze_image(img_path)
        image_sizes.append(size)
        image_means.append(mean_val)

# Convert to DataFrame for analysis
img_stats = pd.DataFrame({
    'width': [s[0] for s in image_sizes],
    'height': [s[1] for s in image_sizes],
    'mean_pixel_value': image_means
})

# Display statistics
print("\nImage Statistics:")
display(img_stats.describe())

# Plot image dimensions distribution using matplotlib
plt.figure(figsize=(10, 8))
scatter = plt.scatter(img_stats['width'], img_stats['height'], 
                     c=img_stats['mean_pixel_value'], 
                     cmap='viridis')
plt.colorbar(scatter, label='Mean Pixel Value')
plt.title('Image Dimensions Distribution')
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')
plt.tight_layout()
plt.show()


# Function to display sample images
def display_sample_images(df, samples_per_breed=5):
    # Get unique breeds
    breeds = df['label'].unique()
    n_breeds = len(breeds)
    
    # Create a figure with subplots for each breed
    plt.figure(figsize=(20, 4*n_breeds))
    
    # For each breed
    for breed_idx, breed in enumerate(breeds):
        # Get 5 random samples for this breed
        breed_samples = df[df['label'] == breed].sample(min(samples_per_breed, len(df[df['label'] == breed])))
        
        # Display each sample
        for sample_idx, (_, row) in enumerate(breed_samples.iterrows(), 1):
            plt.subplot(n_breeds, samples_per_breed, breed_idx * samples_per_breed + sample_idx)
            img = Image.open(Path(TRAIN_DIR) / Path(row['filename']))
            plt.imshow(img)
            plt.title(f"{breed}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()
display_sample_images(train_labels, samples_per_breed=10)


class EnhancedSheepDataset(Dataset):
    def __init__(self, image_dir, labels_df=None, is_train=True):
        self.image_dir = image_dir
        self.is_train = is_train
        self.augment = get_train_augmentations() if is_train else get_val_augmentations()
        
        if is_train:
            self.labels_df = labels_df
            self.image_names = labels_df['filename'].tolist()
        else:
            self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented = self.augment(image=image)
        pixel_values = augmented['image']
        
        if self.is_train:
            label = self.labels_df.loc[self.labels_df['filename'] == img_name, 'label'].values[0]
            label_idx = label_to_idx[label]
            return {'pixel_values': pixel_values, 'labels': torch.tensor(label_idx, dtype=torch.long)}
        else:
            return {'pixel_values': pixel_values, 'filename': img_name}


def get_train_augmentations():
    return A.Compose([
        A.Resize(height=256, width=256),
        A.RandomResizedCrop(
            size=(224, 224),
            scale=(0.8, 1.0), 
            ratio=(0.75, 1.33),
            interpolation=cv2.INTER_LINEAR,
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Affine(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=7, p=0.2),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_augmentations():
    return A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])



def show_with_border(ax, image, title, border_color='black'):
    ax.imshow(image)
    ax.set_title(title, fontsize=14, fontweight='bold', color=border_color)
    ax.axis("off")
    
    # Add a colored border
    rect = patches.Rectangle((0, 0), image.shape[1], image.shape[0],
                             linewidth=6, edgecolor=border_color, facecolor='none')
    ax.add_patch(rect)

def tensor_to_image(tensor):
    """Convert PyTorch tensor to a NumPy image for visualization."""
    image = tensor.numpy().transpose(1, 2, 0)  # CHW -> HWC
    image = np.clip(image, 0, 1)
    return image

def visualize_transform(df, image_dir, transform, idx=0):
    row = df.iloc[idx]
    img_path = Path(image_dir) / row["filename"]

    img_cv = cv2.imread(str(img_path))
    if img_cv is None:
        print(f"404: {img_path}")
        return
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    img_pil = Image.open(str(img_path)).convert("RGB")
    transformed = transform(image=np.array(img_pil))["image"]  # Convert PIL â†’ NumPy â†’ Transform

    transformed_img = tensor_to_image(transformed)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    show_with_border(axs[0], img_rgb, "Original", border_color="darkgreen")
    show_with_border(axs[1], transformed_img, "Transformed", border_color="darkred")

    plt.tight_layout()
    plt.show()



df = train_labels.groupby("label").sample(1)
for i in range(len(df)):
    visualize_transform(df, TRAIN_DIR, get_train_augmentations(), idx=i)


# https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Fake torch._six.container_abcs to avoid import error

sys.modules['torch._six'] = type(sys)('torch._six')
sys.modules['torch._six'].container_abcs = container_abcs

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class ConvNeXtV2ForHF(ConvNeXtV2):
    def forward(self, pixel_values=None, labels=None):
        # call your existing forward_features + head
        features = super().forward_features(pixel_values)
        logits = self.head(features)    # (batch, num_classes)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_base_HF(**kwargs):
    model = ConvNeXtV2ForHF(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model



def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_large_HF(**kwargs):
    model = ConvNeXtV2ForHF(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model


# !wget https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_512_ema.pt
# !wget https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_512_ema.pt
# !wget https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_224_ema.pt
!wget https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_224_ema.pt
!wget https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_224_ema.pt


def get_kfold_datasets(k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_labels['filename'], train_labels['label'])):
        train_df = train_labels.iloc[train_idx]
        val_df = train_labels.iloc[val_idx]
        train_dataset = EnhancedSheepDataset(TRAIN_DIR, train_df)
        val_dataset = EnhancedSheepDataset(TRAIN_DIR, val_df)
        folds.append({
            'fold': fold_idx,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset
        })
    return folds


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(device) if class_weights is not None else None
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        pixel_values = inputs.pop("pixel_values")
        outputs = model(pixel_values)
        logits = outputs.logits
        
        loss_fct = FocalLoss(alpha=self.class_weights, gamma=self.gamma)
        loss = loss_fct(logits.view(-1, NUM_CLASSES), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(p):
    predictions = p.predictions
    labels = p.label_ids
    predictions = np.argmax(predictions, axis=1)
    # print(predictions, labels)
    
    accuracy = (predictions == labels).mean()
    macro_f1 = f1_score(labels, predictions, average='macro')
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    
    cm = confusion_matrix(labels, predictions)
    print("\nConfusion Matrix:")
    print(cm)
    
    per_class_f1 = f1_score(labels, predictions, average=None)
    print("\nF1 Score Per Class:")
    for idx, f1 in enumerate(per_class_f1):
        print(f"{idx_to_label[idx]}: {f1:.4f}")
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "avg_f1": (macro_f1 + weighted_f1) / 2
    }


def train_and_evaluate():
    labels = train_labels['label'].map(label_to_idx).values
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print("\nClass Weights:")
    for idx, weight in enumerate(class_weights):
        print(f"{idx_to_label[idx]}: {weight:.4f}")
    
    # model = ViTForImageClassification.from_pretrained(
    #     "google/vit-base-patch16-224-in21k",
    #     num_labels=NUM_CLASSES,
    #     id2label=idx_to_label,
    #     label2id=label_to_idx,
    #     ignore_mismatched_sizes=True
    # )
    # model = ConvNextForImageClassification.from_pretrained(
    #     "facebook/convnextv2-base-22k-384", 
    #     # num_labels=NUM_CLASSES,
    #     # id2label=idx_to_label,
    #     # label2id=label_to_idx,
    #     # ignore_mismatched_sizes=True  # so you can resize the head
    # )
    model = convnextv2_base_HF()
    ckpt = torch.load('convnextv2_base_22k_224_ema.pt', map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.head = nn.Linear(in_features=model.head.in_features, out_features=NUM_CLASSES)
        
    folds = get_kfold_datasets(k=5)
    test_predictions = []
    
    for fold_data in folds:
        fold_idx = fold_data['fold']
        print(f"\n========== Training Fold {fold_idx + 1}/5 ==========")
        
        # model = convnextv2_base_HF()
        # ckpt = torch.load('convnextv2_base_22k_224_ema.pt', map_location=device)
        
        # # model = convnextv2_large_HF()
        # # ckpt = torch.load('convnextv2_large_22k_224_ema.pt', map_location=device)
        # model.load_state_dict(ckpt["model"], strict=True)
        # model.head = nn.Linear(in_features=model.head.in_features, out_features=NUM_CLASSES)
        
        training_args = TrainingArguments(
            output_dir=f'./results_fold_{fold_idx}_Convnextv2',
            num_train_epochs=30,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=300,
            weight_decay=0.01,
            logging_dir=f'./logs_fold_{fold_idx}_Convnextv2',
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=1,
            learning_rate=5e-5,
            metric_for_best_model='macro_f1',
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            disable_tqdm=False,
            report_to=["tensorboard"],
            remove_unused_columns=False,
            label_names=["labels"],
        )
        
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=fold_data['train_dataset'],
            eval_dataset=fold_data['val_dataset'],
            compute_metrics=compute_metrics,
            class_weights=class_weights,
            gamma=1.5
        )
        
        trainer.train()
        eval_results = trainer.evaluate()
        print(f"\nValidation Results for Fold {fold_idx+1}:")
        print(eval_results)
        
        trainer.save_model(f"best_Convnextv2_model_fold_{fold_idx+1}")
        
        test_set = EnhancedSheepDataset(TEST_DIR, is_train=False)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
        
        fold_test_predictions = []
        filenames = []
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Predicting Fold {fold_idx+1}"):
                inputs = batch['pixel_values'].to(device)
                output = model(inputs)
                probs = torch.softmax(output.logits, dim=-1)
                
                fold_test_predictions.append(probs.detach().cpu().numpy())
                filenames.extend(batch['filename'])
        
        test_predictions.append(np.concatenate(fold_test_predictions))
    
    return np.array(test_predictions), filenames


def ensemble_predictions(predictions, filenames):
    avg_predictions = np.mean(predictions, axis=0)
    final_predictions = np.argmax(avg_predictions, axis=1)
    predicted_labels = [idx_to_label[idx] for idx in final_predictions]
    
    submission = pd.DataFrame({
        'filename': filenames,
        'label': predicted_labels
    })
    
    dummy_sub = pd.read_csv(DUMMY_SUB_PATH)
    submission = submission.sort_values('filename').reset_index(drop=True)
    return submission



if __name__ == "__main__":
    test_predictions, test_filenames = train_and_evaluate()
    submission = ensemble_predictions(test_predictions, test_filenames)
    submission.to_csv('submission.csv', index=False)
    print("\nPredictions saved to submission.csv")
    print(f"Number of test samples predicted: {len(submission)}")



# Function to display sample images
def display_sample_images(df, samples_per_breed=5):
    # Get unique breeds
    breeds = df['label'].unique()
    n_breeds = len(breeds)
    
    # Create a figure with subplots for each breed
    plt.figure(figsize=(20, 4*n_breeds))
    
    # For each breed
    for breed_idx, breed in enumerate(breeds):
        # Get 5 random samples for this breed
        breed_samples = df[df['label'] == breed].sample(min(samples_per_breed, len(df[df['label'] == breed])))
        
        # Display each sample
        for sample_idx, (_, row) in enumerate(breed_samples.iterrows(), 1):
            plt.subplot(n_breeds, samples_per_breed, breed_idx * samples_per_breed + sample_idx)
            img = Image.open(Path(TEST_DIR) / Path(row['filename']))
            plt.imshow(img)
            plt.title(f"{breed}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()



display_sample_images(submission, samples_per_breed=20)

