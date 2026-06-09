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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Data
# NOTE: On Kaggle, your path is likely "../input/[competition-name]/train.csv"
# You may need to update this string depending on where the file is located.
file_path = '/kaggle/input/noaa-right-whale-recognition/train.csv' 

try:
    df = pd.read_csv(file_path)
    print("âœ… Data loaded successfully!")
except FileNotFoundError:
    print(f"â�Œ File not found at {file_path}. Please check your path.")
    # Creating dummy data so the code runs for demonstration purposes if file is missing
    data = {'Image': [f'img_{i}.jpg' for i in range(100)], 
            'whaleID': ['whale_001']*20 + ['whale_002']*15 + ['whale_003']*10 + [f'whale_{i}' for i in range(4, 59)]}
    df = pd.DataFrame(data)

# 2. Basic Statistics
num_images = len(df)
num_whales = df['whaleID'].nunique()
images_per_whale = df['whaleID'].value_counts()

print("-" * 30)
print(f"Total Images: {num_images}")
print(f"Total Unique Whales: {num_whales}")
print("-" * 30)
print("Top 5 most frequent whales:")
print(images_per_whale.head())
print("-" * 30)
print("Stats on images per whale:")
print(images_per_whale.describe())
print("-" * 30)

# 3. Visualize the Top 20 Whales
plt.figure(figsize=(12, 8))
top_20_whales = images_per_whale.head(20)

sns.barplot(x=top_20_whales.index, y=top_20_whales.values, palette="viridis")

plt.title('Top 20 Most Frequent Whales', fontsize=16)
plt.xlabel('Whale ID', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.xticks(rotation=45, ha='right') # Rotate labels for readability
plt.tight_layout()

plt.show()

# 4. Visualize the "Long Tail" (Optional but recommended)
# This shows how many whales have very few images (imbalance check)
plt.figure(figsize=(10, 6))
plt.hist(images_per_whale.values, bins=50, color='teal', edgecolor='black')
plt.yscale('log') # Log scale because the disparity is usually huge
plt.title('Distribution of Images per Whale (Log Scale)', fontsize=16)
plt.xlabel('Number of Images', fontsize=12)
plt.ylabel('Count of Whales (Log Scale)', fontsize=12)
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()

plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
CSV_PATH = '/kaggle/input/noaa-right-whale-recognition/train.csv'

def run_simple_analysis():
    print("ğŸ“Š Loading dataset...")
    try:
        df = pd.read_csv(CSV_PATH)
        # Remove known bad image if present
        df = df[df['Image'] != 'w_7489.jpg']
    except FileNotFoundError:
        print("âš ï¸� CSV file not found. Please check the path.")
        return

    # --- INSIGHTS WITHOUT GRAPHS (Text Analysis) ---
    total_images = len(df)
    total_whales = df['whaleID'].nunique()
    counts = df['whaleID'].value_counts()
    
    single_image_whales = counts[counts == 1].count()
    rare_whales = counts[counts < 5].count()
    
    print("\n" + "="*30)
    print("ğŸ§� DATASET INSIGHTS SUMMARY")
    print("="*30)
    print(f"â€¢ Total Images:      {total_images}")
    print(f"â€¢ Unique Whales:     {total_whales}")
    print(f"â€¢ Average Images/Whale: {total_images / total_whales:.2f}")
    print("-" * 30)
    print(f"â€¢ Most Popular Whale:   {counts.index[0]} ({counts.iloc[0]} images)")
    print(f"â€¢ 'One-Shot' Whales:    {single_image_whales} whales have exactly 1 image.")
    print(f"â€¢ Rare Whales (<5 imgs): {rare_whales} whales ({rare_whales/total_whales*100:.1f}% of total).")
    print("="*30 + "\n")

    # --- GRAPH 1: THE "CELEBRITY" WHALES (Bar Chart) ---
    # Shows who the most common whales are.
    plt.figure(figsize=(12, 6))
    top_20 = counts.head(20)
    sns.barplot(x=top_20.index, y=top_20.values, palette="viridis")
    plt.title('Top 20 Most Photographed Whales', fontsize=15)
    plt.xlabel('Whale ID')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph_1_top_whales.png')
    plt.show()
    
    # --- GRAPH 2: THE "LONG TAIL" (Histogram) ---
    # Shows the extreme class imbalance.
    plt.figure(figsize=(10, 6))
    plt.hist(counts.values, bins=range(1, 50), color='#34495e', edgecolor='white')
    plt.title('Distribution of Images per Whale (Class Imbalance)', fontsize=15)
    plt.xlabel('Number of Images Available')
    plt.ylabel('Count of Whales')
    plt.axvline(x=5, color='red', linestyle='--', label='Rare Threshold (5 images)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph_2_imbalance_hist.png')
    plt.show()

    # --- GRAPH 3: DIFFICULTY BREAKDOWN (Pie Chart) ---
    # Visualizes the challenge level.
    labels = ['One-Shot (1 Img)', 'Few-Shot (2-10 Imgs)', 'Frequent (>10 Imgs)']
    
    c1 = counts[counts == 1].count()
    c2 = counts[(counts >= 2) & (counts <= 10)].count()
    c3 = counts[counts > 10].count()
    
    sizes = [c1, c2, c3]
    colors = ['#e74c3c', '#f39c12', '#2ecc71'] # Red (Hard), Orange (Medium), Green (Easy)
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, explode=(0.05, 0, 0))
    plt.title('Whale Rarity Breakdown (Difficulty Level)', fontsize=15)
    plt.savefig('graph_3_rarity_pie.png')
    plt.show()

    print("âœ… Analysis complete. Graphs saved as png files.")

if __name__ == "__main__":
    run_simple_analysis()


import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import zipfile
import glob

# --- CONFIGURATION ---
# Note: On Kaggle, input is read-only. We will extract to /kaggle/working/
CONFIG = {
    # Point this to the ZIP file if that's what you have
    'IMG_DIR': '/kaggle/input/noaa-right-whale-recognition/imgs.zip', 
    'CSV_PATH': '/kaggle/input/noaa-right-whale-recognition/train.csv',
    'IMG_SIZE': 256,
    'BATCH_SIZE': 32,
    'NUM_WORKERS': 2,
    'SEED': 42
}

def handle_zip_extraction(path_from_config):
    """
    Checks if IMG_DIR is a zip file. If so, extracts it to a writable directory
    and returns the new path to the folder containing images.
    """
    if not path_from_config.endswith('.zip'):
        return path_from_config

    # Define extraction target (Kaggle working dir)
    extract_root = "./extracted_data"
    
    # Check if already extracted to avoid re-doing it on re-runs
    if os.path.exists(extract_root) and len(os.listdir(extract_root)) > 0:
        print(f"âœ… Zip already extracted at: {extract_root}")
    else:
        print(f"ğŸ“¦ Extracting {path_from_config} to {extract_root}...")
        os.makedirs(extract_root, exist_ok=True)
        with zipfile.ZipFile(path_from_config, 'r') as zip_ref:
            zip_ref.extractall(extract_root)
        print("âœ… Extraction complete.")

    # CRITICAL STEP: Find where the images actually are.
    # Sometimes zip files have a folder inside them (e.g., imgs.zip -> imgs/ -> image.jpg)
    # or they have images at the root (imgs.zip -> image.jpg).
    
    # Check for a subdirectory that matches the zip name (common Kaggle pattern)
    subfolder_name = os.path.splitext(os.path.basename(path_from_config))[0] # 'imgs'
    potential_subfolder = os.path.join(extract_root, subfolder_name)
    
    if os.path.exists(potential_subfolder):
        return potential_subfolder
    
    # Otherwise, images are likely in the extract_root directly
    return extract_root

def load_and_preprocess_df(csv_path):
    """
    Loads CSV, encodes labels, and handles the train/val split logic
    specifically for the rare whale problem.
    """
    df = pd.read_csv(csv_path)
    
    # 1. Label Encoding
    encoder = LabelEncoder()
    df['label_idx'] = encoder.fit_transform(df['whaleID'])
    classes = encoder.classes_
    print(f"âœ… Label Encoding Complete. Found {len(classes)} unique whales.")
    
    # 2. Stratified Split Logic
    counts = df.whaleID.value_counts()
    single_shot_whales = counts[counts == 1].index
    
    df_single = df[df.whaleID.isin(single_shot_whales)]
    df_multi = df[~df.whaleID.isin(single_shot_whales)]
    
    print(f"ğŸ“Š Split Stats: {len(df_single)} whales have only 1 image (forced to Train).")
    
    train_multi, val_multi = train_test_split(
        df_multi, 
        test_size=0.1, 
        random_state=CONFIG['SEED'], 
        stratify=df_multi['whaleID']
    )
    
    df_train = pd.concat([train_multi, df_single]).sample(frac=1).reset_index(drop=True)
    df_val = val_multi.sample(frac=1).reset_index(drop=True)
    
    print(f"âœ… Final Split: Train: {len(df_train)} images, Val: {len(df_val)} images")
    
    return df_train, df_val, classes

# --- AUGMENTATION ---
def get_transforms(data='train'):
    if data == 'train':
        return A.Compose([
            A.Resize(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.7),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),
            A.GaussianBlur(p=0.1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    elif data == 'val':
        return A.Compose([
            A.Resize(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

# --- DATASET CLASS ---
class WhaleDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image']
        label = row['label_idx']
        
        # Construct full path
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load Image
        image = cv2.imread(img_path)
        
        if image is None:
            # Fallback/Debug info if image isn't found
            # print(f"Warning: Could not load {img_path}") 
            image = np.zeros((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

# --- VISUALIZATION HELPER ---
def visualize_batch(dataloader, classes):
    images, labels = next(iter(dataloader))
    plt.figure(figsize=(16, 8))
    for i in range(min(8, len(images))):
        ax = plt.subplot(2, 4, i + 1)
        img = images[i].permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(classes[labels[i]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    # 0. Handle Zip Extraction
    # This automatically updates the IMG_DIR to the folder where we extracted the files
    real_img_dir = handle_zip_extraction(CONFIG['IMG_DIR'])
    print(f"ğŸ“‚ Images will be loaded from: {real_img_dir}")

    # 1. Prepare DataFrames
    if os.path.exists(CONFIG['CSV_PATH']):
        train_df, val_df, class_names = load_and_preprocess_df(CONFIG['CSV_PATH'])
        
        # 2. Create Datasets using the REAL image directory
        train_dataset = WhaleDataset(train_df, real_img_dir, transform=get_transforms('train'))
        val_dataset = WhaleDataset(val_df, real_img_dir, transform=get_transforms('val'))
        
        # 3. Create DataLoaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=CONFIG['BATCH_SIZE'], 
            shuffle=True, 
            num_workers=CONFIG['NUM_WORKERS']
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=CONFIG['BATCH_SIZE'], 
            shuffle=False, 
            num_workers=CONFIG['NUM_WORKERS']
        )
        
        print("\nğŸ”� Visualizing a batch of augmented training data...")
        visualize_batch(train_loader, class_names)
        
        print("\nâœ… Ready for model training!")
    else:
        print(f"âš ï¸� CSV file not found at {CONFIG['CSV_PATH']}")


import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import zipfile
import glob

# --- CONFIGURATION ---
CONFIG = {
    'IMG_DIR': '/kaggle/input/noaa-right-whale-recognition/imgs.zip', 
    'CSV_PATH': '/kaggle/input/noaa-right-whale-recognition/train.csv',
    
    # T4 x2 OPTIMIZATIONS
    'IMG_SIZE': 384,         
    'BATCH_SIZE': 32,
    'NUM_WORKERS': 2,
    'SEED': 42
}

def handle_zip_extraction(path_from_config):
    extract_root = "./extracted_data"
    
    if path_from_config.endswith('.zip'):
        if os.path.exists(extract_root) and len(os.listdir(extract_root)) > 0:
            print(f"âœ… Found existing data in: {extract_root}. Skipping extraction.")
        else:
            print(f"ğŸ“¦ Extracting {path_from_config} to {extract_root}...")
            os.makedirs(extract_root, exist_ok=True)
            with zipfile.ZipFile(path_from_config, 'r') as zip_ref:
                zip_ref.extractall(extract_root)
            print("âœ… Extraction complete.")
    else:
        extract_root = path_from_config

    potential_subfolder = os.path.join(extract_root, 'imgs')
    if os.path.exists(potential_subfolder):
        return potential_subfolder
    
    return extract_root

def load_and_preprocess_df(csv_path):
    df = pd.read_csv(csv_path)
    
    # 1. REMOVE KNOWN MISSING IMAGES
    # w_7489.jpg is known to be missing in this dataset
    df = df[df['Image'] != 'w_7489.jpg'].copy()
    
    encoder = LabelEncoder()
    df['label_idx'] = encoder.fit_transform(df['whaleID'])
    classes = encoder.classes_
    print(f"âœ… Label Encoding Complete. Found {len(classes)} unique whales.")
    
    counts = df.whaleID.value_counts()
    single_shot_whales = counts[counts == 1].index
    
    df_single = df[df.whaleID.isin(single_shot_whales)]
    df_multi = df[~df.whaleID.isin(single_shot_whales)]
    
    print(f"ğŸ“Š Split Stats: {len(df_single)} whales have only 1 image (forced to Train).")
    
    train_multi, val_multi = train_test_split(
        df_multi, 
        test_size=0.1, 
        random_state=CONFIG['SEED'], 
        stratify=df_multi['whaleID']
    )
    
    df_train = pd.concat([train_multi, df_single]).sample(frac=1).reset_index(drop=True)
    df_val = val_multi.sample(frac=1).reset_index(drop=True)
    
    print(f"âœ… Final Split: Train: {len(df_train)} images, Val: {len(df_val)} images")
    
    return df_train, df_val, classes

# --- AUGMENTATION ---
def get_transforms(data='train'):
    if data == 'train':
        return A.Compose([
            # FIX: Ensure image is at least 1200x1200 before cropping
            # If image is smaller, it pads with zeros (black)
            A.PadIfNeeded(min_height=1200, min_width=1200, border_mode=cv2.BORDER_CONSTANT, value=0),
            
            # Smart "Zoom" - Crop the center 
            A.CenterCrop(height=1200, width=1200, p=1.0), 
            
            A.Resize(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.7),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),
            A.GaussianBlur(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif data == 'val':
        return A.Compose([
            A.PadIfNeeded(min_height=1200, min_width=1200, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.CenterCrop(height=1200, width=1200, p=1.0),
            A.Resize(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

# --- DATASET CLASS ---
class WhaleDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image']
        label = row['label_idx']
        img_path = os.path.join(self.img_dir, img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            # FIX: Fallback image must be large enough for the crop!
            # Using 1500x1500x3 ensures CenterCrop(1200, 1200) works fine.
            image = np.zeros((1500, 1500, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

# --- VISUALIZATION HELPER ---
def visualize_batch(dataloader, classes):
    try:
        images, labels = next(iter(dataloader))
    except StopIteration:
        print("âš ï¸� DataLoader is empty. Check your paths.")
        return

    plt.figure(figsize=(16, 8))
    for i in range(min(8, len(images))):
        ax = plt.subplot(2, 4, i + 1)
        img = images[i].permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(classes[labels[i]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    real_img_dir = handle_zip_extraction(CONFIG['IMG_DIR'])
    print(f"ğŸ“‚ Images will be loaded from: {real_img_dir}")

    if os.path.exists(CONFIG['CSV_PATH']):
        train_df, val_df, class_names = load_and_preprocess_df(CONFIG['CSV_PATH'])
        
        train_dataset = WhaleDataset(train_df, real_img_dir, transform=get_transforms('train'))
        val_dataset = WhaleDataset(val_df, real_img_dir, transform=get_transforms('val'))
        
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=CONFIG['NUM_WORKERS'])
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=CONFIG['NUM_WORKERS'])
        
        print("\nğŸ”� Visualizing a batch of augmented training data...")
        visualize_batch(train_loader, class_names)
        
        print("\nâœ… Ready for model training!")
    else:
        print(f"âš ï¸� CSV file not found at {CONFIG['CSV_PATH']}")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import timm
from tqdm import tqdm
import numpy as np
import time
import copy
import sys

try:
    from preprocessing_pipeline import CONFIG, train_loader, val_loader, class_names
except ImportError:
    if 'train_loader' not in globals():
        print("âš ï¸� Variables from preprocessing not found. Please run the preprocessing step first.")
        sys.exit(1)

# --- CONFIGURATION ---
TRAIN_CONFIG = {
    'MODEL_NAME': 'resnet26d', 
    'NUM_EPOCHS': 10,          
    'LEARNING_RATE': 3e-4,    
    'WEIGHT_DECAY': 1e-4,
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu") # Changed to generic 'cuda'
}

# --- MODEL DEFINITION ---
class WhaleClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(WhaleClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.model.num_features
        
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        features = self.model(x)
        output = self.head(features)
        return output

# --- TRAINING HELPER FUNCTIONS ---
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['NUM_EPOCHS']} [Train]")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['NUM_EPOCHS']} [Val]")
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# --- MAIN TRAINING LOOP ---
if __name__ == "__main__":
    print(f"ğŸš€ Training on device: {TRAIN_CONFIG['DEVICE']}")
    print(f"ğŸ�³ Number of classes: {len(class_names)}")
    
    # 1. Initialize Model
    model = WhaleClassifier(
        model_name=TRAIN_CONFIG['MODEL_NAME'],
        num_classes=len(class_names)
    )
    
    # --- MULTI-GPU LOGIC ---
    if torch.cuda.device_count() > 1:
        print(f"ğŸ”¥ Found {torch.cuda.device_count()} GPUs! Using DataParallel.")
        model = nn.DataParallel(model)
    
    model = model.to(TRAIN_CONFIG['DEVICE'])
    
    # 2. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['LEARNING_RATE'], weight_decay=TRAIN_CONFIG['WEIGHT_DECAY'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG['NUM_EPOCHS'], eta_min=1e-6)
    
    # 3. Training Loop
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(TRAIN_CONFIG['NUM_EPOCHS']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, TRAIN_CONFIG['DEVICE'], epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, TRAIN_CONFIG['DEVICE'], epoch)
        scheduler.step()
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_loss < best_loss:
            print(f"â­�ï¸� Validation Loss Improved ({best_loss:.4f} -> {val_loss:.4f}). Saving model...")
            best_loss = val_loss
            
            # Handle saving for DataParallel (unwrap 'module.')
            if isinstance(model, nn.DataParallel):
                best_model_wts = copy.deepcopy(model.module.state_dict())
            else:
                best_model_wts = copy.deepcopy(model.state_dict())
                
            torch.save(best_model_wts, "best_whale_model.pth")
            
    time_elapsed = time.time() - start_time
    print(f"\nâœ… Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"ğŸ�† Best Validation Loss: {best_loss:.4f}")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import timm
from tqdm import tqdm
import numpy as np
import time
import copy
import sys

try:
    from preprocessing_pipeline import CONFIG, train_loader, val_loader, class_names
except ImportError:
    if 'train_loader' not in globals():
        print("âš ï¸� Variables from preprocessing not found. Please run the preprocessing step first.")
        sys.exit(1)

# --- CONFIGURATION ---
TRAIN_CONFIG = {
    'MODEL_NAME': 'resnet26d', 
    'NUM_EPOCHS': 31,          # INCREASED to 30 to allow convergence
    'LEARNING_RATE': 3e-4,    
    'WEIGHT_DECAY': 1e-4,
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# --- METRIC HELPER ---
def calculate_accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

# --- MODEL DEFINITION ---
class WhaleClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(WhaleClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.model.num_features
        
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        features = self.model(x)
        output = self.head(features)
        return output

# --- TRAINING HELPER FUNCTIONS ---
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['NUM_EPOCHS']} [Train]")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        
        # Calculate Top-1 and Top-5
        acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))
        correct_1 += acc1.item() * labels.size(0)
        correct_5 += acc5.item() * labels.size(0)
        
        pbar.set_postfix({'loss': loss.item(), 'acc1': correct_1/total, 'acc5': correct_5/total})
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc1 = correct_1 / total
    epoch_acc5 = correct_5 / total
    return epoch_loss, epoch_acc1, epoch_acc5

def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['NUM_EPOCHS']} [Val]")
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            
            acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))
            correct_1 += acc1.item() * labels.size(0)
            correct_5 += acc5.item() * labels.size(0)
            
            pbar.set_postfix({'loss': loss.item(), 'acc1': correct_1/total, 'acc5': correct_5/total})
            
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc1 = correct_1 / total
    epoch_acc5 = correct_5 / total
    return epoch_loss, epoch_acc1, epoch_acc5

# --- MAIN TRAINING LOOP ---
if __name__ == "__main__":
    print(f"ğŸš€ Training on device: {TRAIN_CONFIG['DEVICE']}")
    print(f"ğŸ�³ Number of classes: {len(class_names)}")
    
    # 1. Initialize Model
    model = WhaleClassifier(
        model_name=TRAIN_CONFIG['MODEL_NAME'],
        num_classes=len(class_names)
    )
    
    # --- MULTI-GPU LOGIC ---
    if torch.cuda.device_count() > 1:
        print(f"ğŸ”¥ Found {torch.cuda.device_count()} GPUs! Using DataParallel.")
        model = nn.DataParallel(model)
    
    model = model.to(TRAIN_CONFIG['DEVICE'])
    
    # 2. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['LEARNING_RATE'], weight_decay=TRAIN_CONFIG['WEIGHT_DECAY'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG['NUM_EPOCHS'], eta_min=1e-6)
    
    # 3. Training Loop
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(TRAIN_CONFIG['NUM_EPOCHS']):
        # Train
        train_loss, train_acc1, train_acc5 = train_one_epoch(model, train_loader, criterion, optimizer, TRAIN_CONFIG['DEVICE'], epoch)
        
        # Validate
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, TRAIN_CONFIG['DEVICE'], epoch)
        
        # Scheduler Step
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} Acc1: {train_acc1:.4f} Acc5: {train_acc5:.4f}")
        print(f"          Val Loss:   {val_loss:.4f} Acc1: {val_acc1:.4f} Acc5: {val_acc5:.4f}")
        
        # Save Best Model
        if val_loss < best_loss:
            print(f"â­�ï¸� Val Loss Improved ({best_loss:.4f} -> {val_loss:.4f}). Saving...")
            best_loss = val_loss
            
            if isinstance(model, nn.DataParallel):
                best_model_wts = copy.deepcopy(model.module.state_dict())
            else:
                best_model_wts = copy.deepcopy(model.state_dict())
                
            torch.save(best_model_wts, "best_whale_model.pth")
            
    time_elapsed = time.time() - start_time
    print(f"\nâœ… Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"ğŸ�† Best Validation Loss: {best_loss:.4f}")


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import timm
import math

# Import necessary variables from previous steps
# If running in a notebook, these should already be in memory.
try:
    from preprocessing_pipeline import val_loader, class_names, CONFIG
except ImportError:
    print("âš ï¸� properties not found. Make sure you ran the preprocessing step!")

# --- REDEFINE MODEL (Must match training exactly) ---
class WhaleClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False):
        super(WhaleClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.model.num_features
        
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        features = self.model(x)
        output = self.head(features)
        return output

def visualize_model_predictions(model_path, num_images=12):
    """
    Loads the best model and visualizes predictions on the validation set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ğŸ”� Running inference on: {device}")
    
    # 1. Re-initialize the model structure
    model = WhaleClassifier(model_name='resnet26d', num_classes=len(class_names), pretrained=False)
    
    # 2. Load the trained weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("âœ… Model weights loaded successfully.")
    except Exception as e:
        print(f"â�Œ Failed to load model weights: {e}")
        return

    model = model.to(device)
    model.eval() # Set to evaluation mode
    
    # 3. Get a batch of validation data
    images, labels = next(iter(val_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    # 4. Predict
    with torch.no_grad():
        outputs = model(images)
        # Get probabilities using Softmax
        probs = torch.nn.functional.softmax(outputs, dim=1)
        # Get top 1 prediction
        confidences, preds = torch.max(probs, 1)
        
    # 5. Plotting
    # Calculate grid size (e.g., 3x4 for 12 images)
    cols = 4
    rows = math.ceil(min(num_images, len(images)) / cols)
    
    plt.figure(figsize=(20, 5 * rows))
    
    for i in range(min(num_images, len(images))):
        ax = plt.subplot(rows, cols, i + 1)
        
        # Un-normalize image for display
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406] # ImageNet stats
        img = np.clip(img, 0, 1)
        
        true_label = class_names[labels[i].cpu().item()]
        pred_label = class_names[preds[i].cpu().item()]
        confidence = confidences[i].cpu().item() * 100
        
        plt.imshow(img)
        plt.axis("off")
        
        # Color code the title
        if true_label == pred_label:
            color = 'green'
            title_text = f"âœ… {true_label}\nConf: {confidence:.1f}%"
        else:
            color = 'red'
            title_text = f"â�Œ True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)"
            
        plt.title(title_text, color=color, fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Check if the weight file exists
    import os
    if os.path.exists("best_whale_model.pth"):
        visualize_model_predictions("best_whale_model.pth", num_images=16)
    else:
        print("âš ï¸� 'best_whale_model.pth' not found. Did you finish training?")


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import timm
import math
from tqdm import tqdm

# Import necessary variables from previous steps
try:
    from preprocessing_pipeline import val_loader, class_names, CONFIG
except ImportError:
    print("âš ï¸� properties not found. Make sure you ran the preprocessing step!")

# --- CONFIGURATION ---
# Ensure this matches what you used in training!
MODEL_NAME = 'resnet26d' 

# --- REDEFINE MODEL (Must match training exactly) ---
class WhaleClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False):
        super(WhaleClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.model.num_features
        
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        features = self.model(x)
        output = self.head(features)
        return output

def calculate_accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
        return res

def evaluate_and_visualize(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ğŸ”� Loading model from: {model_path}")
    print(f"âš™ï¸�  Using Device: {device}")
    
    # 1. Initialize Model
    model = WhaleClassifier(model_name=MODEL_NAME, num_classes=len(class_names), pretrained=False)
    
    # 2. Load Weights
    try:
        if torch.cuda.device_count() > 1:
            print("Note: Training used multiple GPUs. Adjusting keys...")
        
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("âœ… Weights loaded successfully.")
    except Exception as e:
        print(f"â�Œ Error loading weights: {e}")
        return

    model = model.to(device)
    model.eval()
    
    # --- PART 1: CALCULATE OVERALL ACCURACY ---
    print("\nğŸ“Š Calculating Accuracy on ENTIRE Validation Set...")
    total_correct_1 = 0
    total_correct_5 = 0
    total_samples = 0
    
    # We iterate through the whole validation loader to get the real score
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Calculate batch accuracy
            acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))
            
            total_correct_1 += acc1
            total_correct_5 += acc5
            total_samples += labels.size(0)
            
    avg_acc1 = (total_correct_1 / total_samples) * 100
    avg_acc5 = (total_correct_5 / total_samples) * 100
    
    print("-" * 40)
    print(f"ğŸ�† Final Results for {MODEL_NAME}:")
    print(f"   Top-1 Accuracy: {avg_acc1:.2f}% (Exact Match)")
    print(f"   Top-5 Accuracy: {avg_acc5:.2f}% (Correct whale is in top 5 guesses)")
    print("-" * 40)

    # --- PART 2: VISUALIZE A BATCH ---
    print("\nğŸ–¼ï¸�  Visualizing Random Batch...")
    
    # Get a batch
    images, labels = next(iter(val_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidences, preds = torch.max(probs, 1)
        
    # Plot 16 images
    num_images = min(16, len(images))
    cols = 4
    rows = math.ceil(num_images / cols)
    
    plt.figure(figsize=(20, 5 * rows))
    
    for i in range(num_images):
        ax = plt.subplot(rows, cols, i + 1)
        
        # Un-normalize
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img = np.clip(img, 0, 1)
        
        true_label = class_names[labels[i].cpu().item()]
        pred_label = class_names[preds[i].cpu().item()]
        confidence = confidences[i].cpu().item() * 100
        
        plt.imshow(img)
        plt.axis("off")
        
        if true_label == pred_label:
            color = 'green'
            title_text = f"âœ… {true_label}\nConf: {confidence:.1f}%"
        else:
            color = 'red'
            title_text = f"â�Œ True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)"
            
        plt.title(title_text, color=color, fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import os
    if os.path.exists("best_whale_model.pth"):
        evaluate_and_visualize("best_whale_model.pth")
    else:
        print("âš ï¸� 'best_whale_model.pth' not found. Did you finish training?")


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import os
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- COMPACT CONFIG & CLASSES ---
CONFIG = {'CSV': '/kaggle/input/noaa-right-whale-recognition/train.csv', 'IMG': './extracted_data/imgs', 'SIZE': 384, 'BATCH': 32}

class WhaleDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df, self.img_dir, self.transform = df, img_dir, transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(os.path.join(self.img_dir, row['Image']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((1500, 1500, 3), np.uint8)
        return self.transform(image=img)['image'] if self.transform else img, row['label_idx']

class WhaleClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.head = nn.Sequential(nn.BatchNorm1d(self.model.num_features), nn.Dropout(0.3), nn.Linear(self.model.num_features, num_classes))
    def forward(self, x): return self.head(self.model(x))

# --- MAIN EVALUATION FUNCTION ---
def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"âš™ï¸�  Running Fast Evaluation on {device}...")

    # 1. Quick Data Setup
    df = pd.read_csv(CONFIG['CSV'])
    df = df[df['Image'] != 'w_7489.jpg'].copy() # Remove bad image
    
    encoder = LabelEncoder()
    df['label_idx'] = encoder.fit_transform(df['whaleID'])
    classes = encoder.classes_
    
    # Handle Single Image Whales
    counts = df.whaleID.value_counts()
    single_shot_whales = counts[counts == 1].index
    df_multi = df[~df.whaleID.isin(single_shot_whales)]
    
    # Stratified Split
    _, val_df = train_test_split(df_multi, test_size=0.1, random_state=42, stratify=df_multi['whaleID'])
    
    transforms = A.Compose([
        A.PadIfNeeded(min_height=1200, min_width=1200, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.CenterCrop(1200, 1200), A.Resize(CONFIG['SIZE'], CONFIG['SIZE']),
        A.Normalize(), ToTensorV2()
    ])
    loader = DataLoader(WhaleDataset(val_df, CONFIG['IMG'], transforms), batch_size=CONFIG['BATCH'], shuffle=False, num_workers=2)

    # 2. Load Model
    model = WhaleClassifier('resnet26d', len(classes))
    state_dict = torch.load("best_whale_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # 3. Get Predictions
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for img, label in tqdm(loader, desc="Predicting"):
            out = model(img.to(device))
            prob = torch.softmax(out, dim=1)
            all_preds.extend(prob.argmax(1).cpu().numpy())
            all_targets.extend(label.numpy())
            all_probs.extend(prob.cpu().numpy())

    # 4. Calculate Metrics
    all_targets, all_probs = np.array(all_targets), np.array(all_probs)
    all_preds = np.array(all_preds)
    
    acc1 = np.mean(all_preds == all_targets)
    top5 = sum([all_targets[i] in np.argsort(all_probs[i])[::-1][:5] for i in range(len(all_targets))]) / len(all_targets)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
    try: loss = log_loss(all_targets, all_probs, labels=list(range(len(classes))))
    except: loss = 99.9

    # 5. Print Report
    print(f"\n{'='*30}\nğŸ�† FINAL EVALUATION RESULTS\n{'='*30}")
    print(f"âœ… Top-1 Accuracy:  {acc1:.2%}")
    print(f"âœ… Top-5 Accuracy:  {top5:.2%}")
    print(f"ğŸ“‰ Log Loss:        {loss:.4f}")
    print(f"{'-'*30}")
    print(f"ğŸ�¯ Precision:       {p:.4f}")
    print(f"ğŸ“¡ Recall:          {r:.4f}")
    print(f"âš–ï¸�  F1-Score:        {f1:.4f}")
    print(f"{'='*30}")

    # --- 6. CONFUSION MATRIX (TOP 20) ---
    print("\nğŸ�¨ Generating Confusion Matrix for Top 20 Whales...")
    from collections import Counter
    
    # Get the 20 most frequent whales in the VALIDATION set
    counts = Counter(all_targets)
    top_20_indices = [k for k, v in counts.most_common(20)]
    top_20_names = [classes[i] for i in top_20_indices]
    
    # Filter targets and preds to only include these 20 whales
    mask = [i for i, t in enumerate(all_targets) if t in top_20_indices]
    filtered_targets = all_targets[mask]
    filtered_preds = all_preds[mask]
    
    # Generate Matrix
    cm = confusion_matrix(filtered_targets, filtered_preds, labels=top_20_indices)
    
    # Plot
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=top_20_names, yticklabels=top_20_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Top 20 Most Frequent Whales)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # --- SAVE THE IMAGE ---
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
    print("âœ… Saved plot to 'confusion_matrix.png'")
    
    plt.show()

if __name__ == "__main__":
    if os.path.exists("best_whale_model.pth"): run_evaluation()
    else: print("â�Œ Model not found.")

