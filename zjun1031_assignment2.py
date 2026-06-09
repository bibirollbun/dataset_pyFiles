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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, efficientnet_b3
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


# (1) Explore Dataset
# Dataset path
data_path = "/kaggle/input/siim-isic-melanoma-classification"

# List files
print(os.listdir(data_path))


# Load Metadata (CSV File)
train_df = pd.read_csv(f"{data_path}/train.csv")
test_df = pd.read_csv(f"{data_path}/test.csv")

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print("\nTraining data columns:", train_df.columns.tolist())
print("\nFirst 5 rows:")
print(train_df.head())


# (2) Metadata Cleaning and Preprocessing
def clean_metadata(df, is_train=True):
    """Clean and preprocess metadata"""
    df_clean = df.copy()
    
    # Handle missing values in age
    if 'age_approx' in df_clean.columns:
        # Fill missing age with median
        median_age = df_clean['age_approx'].median()
        df_clean['age_approx'].fillna(median_age, inplace=True)
        
        # Normalize age (0-1 scale)
        df_clean['age_normalized'] = df_clean['age_approx'] / 100.0
        print(f"Age missing values filled with median: {median_age}")
    
    # Handle missing values in sex
    if 'sex' in df_clean.columns:
        # Fill missing sex with mode
        mode_sex = df_clean['sex'].mode()[0] if not df_clean['sex'].mode().empty else 'male'
        df_clean['sex'].fillna(mode_sex, inplace=True)
        
        # One-hot encode sex
        sex_dummies = pd.get_dummies(df_clean['sex'], prefix='sex')
        df_clean = pd.concat([df_clean, sex_dummies], axis=1)
        print(f"Sex missing values filled with mode  : {mode_sex}")
    
    # Handle missing values in anatomical site
    if 'anatom_site_general_challenge' in df_clean.columns:
        # Fill missing site with 'unknown'
        df_clean['anatom_site_general_challenge'].fillna('unknown', inplace=True)
        
        # One-hot encode anatomical site
        site_dummies = pd.get_dummies(df_clean['anatom_site_general_challenge'], prefix='site')
        df_clean = pd.concat([df_clean, site_dummies], axis=1)
        print("Anatomical site missing values filled with 'unknown'")
    
    # Create additional features
    if is_train and 'target' in df_clean.columns:
        # Calculate class weights for imbalanced dataset
        target_counts = df_clean['target'].value_counts()
        print(f"\nClass distribution:")
        print(f"Benign (0)   : {target_counts[0]} ({target_counts[0]/len(df_clean)*100:.2f}%)")
        print(f"Malignant (1): {target_counts[1]} ({target_counts[1]/len(df_clean)*100:.2f}%)")
    
    return df_clean


# Clean training and test metadata
train_clean = clean_metadata(train_df, is_train=True)
test_clean = clean_metadata(test_df, is_train=False)

print(f"\nCleaned training data shape: {train_clean.shape}")
print(f"Cleaned test data shape    : {test_clean.shape}")


# (3) Patient-based Train/Validation Split
def create_patient_split(df, test_size=0.2, random_state=42):
    """Create train/validation split by patient ID to avoid data leakage"""
    
    # Get unique patients
    unique_patients = df['patient_id'].unique()
    print(f"Total unique patients: {len(unique_patients)}")
    
    # Split patients (not individual images)
    train_patients, val_patients = train_test_split(
        unique_patients, 
        test_size=test_size, 
        random_state=random_state,
        stratify=None  # Can't stratify by patient easily, would need more complex logic
    )
    
    # Create train/validation dataframes
    train_split = df[df['patient_id'].isin(train_patients)].copy()
    val_split = df[df['patient_id'].isin(val_patients)].copy()
    
    print(f"Training patients    : {len(train_patients)}")
    print(f"Validation patients  : {len(val_patients)}")
    print(f"Training images      : {len(train_split)}")
    print(f"Validation images    : {len(val_split)}")
    
    # Check target distribution in splits
    if 'target' in df.columns:
        print(f"\nTarget distribution in training split:")
        print(train_split['target'].value_counts(normalize=True))
        print(f"\nTarget distribution in validation split:")
        print(val_split['target'].value_counts(normalize=True))
    
    return train_split, val_split


# Create patient-based split
train_split, val_split = create_patient_split(train_clean)


# (4) Image Preprocessing Functions
class ImagePreprocessor:
    def __init__(self, target_size=(224, 224), normalize=True):
        self.target_size = target_size
        self.normalize = normalize
        
    def load_and_preprocess_image(self, image_path, augment=False):
        """Load and preprocess a single image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                return None
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, self.target_size)
            
            # Normalize pixel values to [0, 1]
            if self.normalize:
                image = image.astype(np.float32) / 255.0
            
            # Apply augmentation if specified
            if augment:
                image = self.apply_augmentation(image)
                
            return image
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def apply_augmentation(self, image):
        """Apply basic data augmentation"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random rotation (small angle)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            rows, cols = image.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            image = cv2.warpAffine(image, M, (cols, rows))
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 1)
            
        return image
    
    def preprocess_batch(self, image_paths, augment=False, batch_size=32):
        """Preprocess a batch of images"""
        images = []
        valid_paths = []
        
        for path in image_paths:
            img = self.load_and_preprocess_image(path, augment=augment)
            if img is not None:
                images.append(img)
                valid_paths.append(path)
                
        return np.array(images), valid_paths


# Initialize preprocessor
preprocessor = ImagePreprocessor(target_size=(224, 224), normalize=True)


# (5) Create Data Loading Functions
def create_image_paths(df, image_dir):
    """Create full image paths from dataframe"""
    return [os.path.join(image_dir, f"{img_id}.jpg") for img_id in df['image_name']]

def save_processed_data(train_df, val_df, test_df, output_dir='processed_data'):
    """Save processed dataframes"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save cleaned metadata
    train_df.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_processed.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)
    
    print(f"Processed data saved to {output_dir}/")
    
    # Save preprocessing summary
    with open(os.path.join(output_dir, 'preprocessing_summary.txt'), 'w') as f:
        f.write("SIIM-ISIC Data Preprocessing Summary\n")
        f.write("="*40 + "\n\n")
        f.write(f"Training samples: {len(train_df)}\n")
        f.write(f"Validation samples: {len(val_df)}\n")
        f.write(f"Test samples: {len(test_df)}\n")
        f.write(f"Image target size: {preprocessor.target_size}\n")
        f.write(f"Normalization applied: {preprocessor.normalize}\n")
        
        # Feature columns
        feature_cols = [col for col in train_df.columns if col not in ['image_name', 'patient_id', 'target']]
        f.write(f"\nFeature columns ({len(feature_cols)}):\n")
        for col in feature_cols:
            f.write(f"  - {col}\n")


# (6) Execute Preprocessing Pipeline
# Create image paths
train_paths = create_image_paths(train_split, f"{data_path}/jpeg/train/")
val_paths = create_image_paths(val_split, f"{data_path}/jpeg/train/")
test_paths = create_image_paths(test_clean, f"{data_path}/jpeg/test/")


# Validate that images exist
def validate_image_paths(paths, df, split_name):
    """Validate that image files exist"""
    existing_paths = []
    valid_indices = []
    
    for i, path in enumerate(paths):
        if os.path.exists(path):
            existing_paths.append(path)
            valid_indices.append(i)
    
    print(f"{split_name}: {len(existing_paths)}/{len(paths)} images found")
    return df.iloc[valid_indices].reset_index(drop=True), existing_paths


# Validate all splits
train_final, train_paths_final = validate_image_paths(train_paths, train_split, "Training")
val_final, val_paths_final = validate_image_paths(val_paths, val_split, "Validation")
test_final, test_paths_final = validate_image_paths(test_paths, test_clean, "Test")


# Save processed data
save_processed_data(train_final, val_final, test_final)


print(f"Training samples   : {len(train_final)}")
print(f"Validation samples : {len(val_final)}")
print(f"Test samples       : {len(test_final)}")


# Load and display a sample image
if len(train_paths_final) > 0:
    # Load a sample image
    sample_image = preprocessor.load_and_preprocess_image(train_paths_final[0])
    if sample_image is not None:
        print(f"Sample image shape: {sample_image.shape}")
        print(f"Sample image data type: {sample_image.dtype}")
        print(f"Sample image value range: [{sample_image.min():.3f}, {sample_image.max():.3f}]")
        
        plt.figure(figsize=(8, 6))
        plt.imshow(sample_image)
        plt.title("Sample Preprocessed Image")
        plt.axis('off')
        plt.show()


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ================================
# STEP 2: Custom Dataset Class
# ================================

class MelanomaDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, is_test=False):
        """
        Custom dataset for melanoma classification
        
        Args:
            dataframe: pandas DataFrame with image_name and target columns
            image_dir: directory containing images
            transform: torchvision transforms
            is_test: whether this is test data (no targets)
        """
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image name and construct path
        image_name = self.df.loc[idx, 'image_name']
        image_path = os.path.join(self.image_dir, f"{image_name}.jpg")
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, image_name
        else:
            target = torch.tensor(self.df.loc[idx, 'target'], dtype=torch.float32)
            return image, target


# ================================
# STEP 3: Data Transforms
# ================================

def get_transforms(image_size=224, augment=True):
    """Get training and validation transforms"""
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


# ================================
# STEP 4: EfficientNet Model
# ================================

class MelanomaEfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=1, pretrained=True, dropout_rate=0.3):
        """
        EfficientNet model for melanoma classification
        
        Args:
            model_name: which EfficientNet variant to use
            num_classes: number of output classes (1 for binary classification)
            pretrained: whether to use pretrained weights
            dropout_rate: dropout rate for regularization
        """
        super(MelanomaEfficientNet, self).__init__()
        
        # Load pretrained EfficientNet
        if model_name == 'efficientnet_b0':
            self.backbone = efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
        elif model_name == 'efficientnet_b3':
            self.backbone = efficientnet_b3(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


# ================================
# STEP 5: Training Functions
# ================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} - Training')
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        predictions = torch.sigmoid(outputs) > 0.5
        running_acc += (predictions == targets).float().mean().item()
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device, epoch):
    """Validate the model for one epoch"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} - Validation')
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(images).squeeze()
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item()
            predictions = torch.sigmoid(outputs)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    auc_score = roc_auc_score(all_targets, all_predictions)
    acc_score = accuracy_score(all_targets, all_predictions > 0.5)
    
    return epoch_loss, auc_score, acc_score

def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-4, weight_decay=1e-5, early_stopping_patience=5, early_stopping=True):
    """Complete training loop"""
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_auc': [], 'val_acc': []
    }
    
    best_auc = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    
    print("Starting training...")
    if early_stopping:
        print(f"Early stopping enabled with patience: {early_stopping_patience}")
    print("="*50)
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_auc, val_acc = validate_one_epoch(model, val_loader, criterion, device, epoch)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(f"New best AUC: {best_auc:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping check
        if early_stopping and epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            print(f"No improvement for {early_stopping_patience} consecutive epochs.")
            break
        
        print("-" * 50)
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nTraining completed! Best validation AUC: {best_auc:.4f}")
    
    return model, history




# ================================
# STEP 6: Visualization Functions
# ================================

def plot_training_history(history):
    """Plot training history"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # AUC
    ax3.plot(epochs, history['val_auc'], 'g-', label='Validation AUC')
    ax3.set_title('Validation AUC')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUC')
    ax3.legend()
    ax3.grid(True)
    
    # Learning rate would go here if tracked
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()


# ================================
# Create subset (for testing)
# ================================
def create_balanced_subset(df, subset_size=5000, min_positive_ratio=0.05):
    """Create a subset that maintains better class balance"""
    
    # Separate positive and negative cases
    positive_cases = df[df['target'] == 1]
    negative_cases = df[df['target'] == 0]
    
    # Calculate how many positives we want (at least 5% of subset)
    min_positives = int(subset_size * min_positive_ratio)
    available_positives = len(positive_cases)
    
    # Use all available positives if we don't have enough
    n_positives = min(min_positives, available_positives)
    n_negatives = subset_size - n_positives
    
    print(f"Creating balanced subset:")
    print(f"  Available positives: {available_positives}")
    print(f"  Using positives: {n_positives}")
    print(f"  Using negatives: {n_negatives}")
    
    # Sample
    if n_positives > 0:
        positive_subset = positive_cases.sample(n=n_positives, random_state=42)
    else:
        positive_subset = pd.DataFrame()
    
    negative_subset = negative_cases.sample(n=n_negatives, random_state=42)
    
    # Combine
    balanced_subset = pd.concat([positive_subset, negative_subset])
    
    print(f"Subset created:")
    print(f"  Total: {len(balanced_subset)}")
    print(f"  Positive: {len(positive_subset)} ({len(positive_subset)/len(balanced_subset)*100:.1f}%)")
    print(f"  Negative: {len(negative_subset)} ({len(negative_subset)/len(balanced_subset)*100:.1f}%)")
    
    return balanced_subset.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle


# ================================
# STEP 7: Main Training Script
# ================================

def main(use_subset=False, subset_size=5000):
    """Main function to run the training"""
    
    # Configuration
    CONFIG = {
        'model_name': 'efficientnet_b0',  # or 'efficientnet_b3'
        'image_size': 224,                # 224 is good for b0, 300 for b3
        'batch_size': 96,                 # Adjust based on GPU memory
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_workers': 4,                 # Adjust based on your system
        'dropout_rate': 0.3,
        'early_stopping': True,           # Enable/disable early stopping
        'early_stopping_patience': 5,     # Stop if no improvement for 5 epochs
        'use_subset': use_subset,
        'subset_size': subset_size
    }
    
    # Data paths (adjust these to match your setup)
    DATA_PATH = "/kaggle/input/siim-isic-melanoma-classification"
    PROCESSED_DATA_PATH = "processed_data"  # From member 1's preprocessing
    
    # Load processed data
    train_df = pd.read_csv(f"{PROCESSED_DATA_PATH}/train_processed.csv")
    val_df = pd.read_csv(f"{PROCESSED_DATA_PATH}/val_processed.csv")
    test_df = pd.read_csv(f"{PROCESSED_DATA_PATH}/test_processed.csv")

    print("="*50)
    print("ORIGINAL DATASET SIZES:")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # CREATE SUBSET IF REQUESTED
    if use_subset:
        print("\n" + "="*50)
        print("CREATING SUBSET FOR FAST EXPERIMENTATION:")
        
        # Create balanced subsets
        train_df = create_balanced_subset(train_df, subset_size=subset_size, min_positive_ratio=0.05)
        val_df = create_balanced_subset(val_df, subset_size=subset_size//5, min_positive_ratio=0.05)  # 20% of train size
        
        print("\nFINAL SUBSET SIZES:")
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)} (unchanged)")
    
    print("="*50)
    
    # Get transforms
    train_transform, val_transform = get_transforms(
        image_size=CONFIG['image_size'], 
        augment=True
    )
    
    # Create datasets
    train_dataset = MelanomaDataset(
        train_df, 
        f"{DATA_PATH}/jpeg/train",
        transform=train_transform
    )
    
    val_dataset = MelanomaDataset(
        val_df, 
        f"{DATA_PATH}/jpeg/train",
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'],
        shuffle=True, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if CONFIG['num_workers'] > 0 else False,
        prefetch_factor=2 if CONFIG['num_workers'] > 0 else 2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'],
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if CONFIG['num_workers'] > 0 else False,
        prefetch_factor=2 if CONFIG['num_workers'] > 0 else 2
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    model = MelanomaEfficientNet(
        model_name=CONFIG['model_name'],
        num_classes=1,
        pretrained=True,
        dropout_rate=CONFIG['dropout_rate']
    ).to(device)
    
    print(f"Model created: {CONFIG['model_name']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train the model
    trained_model, history = train_model(
        model, 
        train_loader, 
        val_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save the trained model with subset info
    model_filename = f"melanoma_efficientnet_{'subset_' if use_subset else 'full_'}model.pth"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': CONFIG,
        'history': history
    }, model_filename)
    
    print(f"Model saved as '{model_filename}'")
    
    return trained_model, history


# ================================
# STEP 8: Inference Functions
# ================================

def load_trained_model(model_path, model_name='efficientnet_b0', dropout_rate=0.3):
    """Load a trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = MelanomaEfficientNet(
        model_name=model_name,
        num_classes=1,
        pretrained=False,  # We're loading our own weights
        dropout_rate=dropout_rate
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint.get('config', {}), checkpoint.get('history', {})

def predict_test_set(model, test_df, data_path, transform, batch_size=32):
    """Generate predictions for test set"""
    test_dataset = MelanomaDataset(
        test_df, 
        f"{data_path}/jpeg/test",
        transform=transform,
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=2
    )
    
    model.eval()
    predictions = []
    image_names = []
    
    with torch.no_grad():
        for images, names in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)
            
            predictions.extend(probs.cpu().numpy())
            image_names.extend(names)
    
    return predictions, image_names

# ================================
# HOW TO USE IT
# ================================

# Option 1: Quick experiment with subset (30-60 minutes)
# trained_model, history = main(use_subset=True, subset_size=3000)

# Option 2: Full training (3-5 hours)  
# trained_model, history = main(use_subset=False)

# Option 3: Different subset sizes
# trained_model, history = main(use_subset=True, subset_size=8000)  # Larger subset

# Run the training when this script is executed
if __name__ == "__main__":
    # Uncomment the line below to start training
    trained_model, training_history = main(use_subset=True, subset_size=3000)
    print("Setup complete! Uncomment the last line to start training.")


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ======================
# DATA LOADING & PREPARATION
# ======================
print("Loading and preparing data...")

# Load processed data from Member 1
train_df = pd.read_csv('/kaggle/input/processed-data/train_processed.csv')
val_df = pd.read_csv('/kaggle/input/processed-data/val_processed.csv')

# Add image paths
DATA_PATH = "/kaggle/input/siim-isic-melanoma-classification"
train_df['image_path'] = f"{DATA_PATH}/jpeg/train/" + train_df['image_name'] + '.jpg'
val_df['image_path'] = f"{DATA_PATH}/jpeg/train/" + val_df['image_name'] + '.jpg'

# Define metadata features
feature_columns = [
    'age_normalized', 'sex_female', 'sex_male', 
    'site_head/neck', 'site_lower extremity', 'site_oral/genital',
    'site_palms/soles', 'site_torso', 'site_unknown', 'site_upper extremity'
]

# Extract metadata features and labels
X_meta_train = train_df[feature_columns].values.astype(np.float32)
y_train = train_df['target'].values
X_meta_val = val_df[feature_columns].values.astype(np.float32)
y_val = val_df['target'].values

# Convert to tensors
X_meta_train_tensor = torch.FloatTensor(X_meta_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
X_meta_val_tensor = torch.FloatTensor(X_meta_val).to(device)
y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)

print(f"Data loaded: Train {len(train_df)}, Val {len(val_df)}")



# ======================
# METADATA MLP 
# ======================
class MetadataMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 64], dropout_rate=0.3):
        super(MetadataMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def extract_features(self, x):
        features = x
        for layer in self.network[:-1]:
            features = layer(features)
        return features

print("Creating and training metadata MLP...")
metadata_mlp = MetadataMLP(len(feature_columns)).to(device)

# Loss, optimizer, scheduler
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(metadata_mlp.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)

# Early stopping setup
patience = 10           
best_val_auc = 0.0
best_epoch = 0
patience_counter = 0
max_epochs = 1000        

print("Epoch\tTrain Loss\tTrain Acc\tTrain AUC\tVal Loss\tVal Acc\tVal AUC")
print("-" * 80)

for epoch in range(max_epochs):
    # Training phase
    metadata_mlp.train()
    optimizer.zero_grad()
    outputs = metadata_mlp(X_meta_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        train_probs = torch.sigmoid(outputs)
        train_preds = (train_probs > 0.5).float()
        train_acc = (train_preds == y_train_tensor).float().mean()
        try:
            train_auc = roc_auc_score(y_train_tensor.cpu().numpy(), train_probs.cpu().numpy())
        except:
            train_auc = 0.5
    
    # Validation phase
    metadata_mlp.eval()
    with torch.no_grad():
        val_outputs = metadata_mlp(X_meta_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_probs = torch.sigmoid(val_outputs)
        val_preds = (val_probs > 0.5).float()
        val_acc = (val_preds == y_val_tensor).float().mean()
        try:
            val_auc = roc_auc_score(y_val_tensor.cpu().numpy(), val_probs.cpu().numpy())
        except:
            val_auc = 0.5
    
    scheduler.step(val_auc)
    
    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"{epoch+1:3d}\t{loss.item():.4f}\t\t{train_acc.item():.4f}\t\t{train_auc:.4f}\t\t"
              f"{val_loss.item():.4f}\t\t{val_acc.item():.4f}\t\t{val_auc:.4f}")
    
    # Check for improvement
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch = epoch
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1} (best epoch {best_epoch+1}, best val AUC {best_val_auc:.4f})")
        break

print("\nMetadata MLP training complete")



# ======================
# LOAD MEMBER 2's MODEL
# ======================
print("Loading Member 2's model...")

# Define the same architecture as Member 2
class MelanomaEfficientNet(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.3):
        super(MelanomaEfficientNet, self).__init__()
        
        # Load pretrained EfficientNet (same as Member 2)
        self.backbone = models.efficientnet_b0(pretrained=False)
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier (same as Member 2)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def extract_features(self, x):
        """Extract features before final classifier"""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

# Load Member 2's model
try:
    member2_model = MelanomaEfficientNet()
    
    # Load weights
    checkpoint = torch.load(
        '/kaggle/input/melanoma-efficientnet-model/melanoma_efficientnet_subset_model.pth',
        map_location=device,
        weights_only=False  # Allow loading potentially unsafe objects
    )
    
    # Handle both state dict and full checkpoint formats
    if 'model_state_dict' in checkpoint:
        member2_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        member2_model.load_state_dict(checkpoint)
    
    member2_model = member2_model.to(device)
    member2_model.eval()
    
    print("Member 2's model loaded successfully")
    
except Exception as e:
    print(f"Failed to load Member 2's model: {e}")
    print("Using ImageNet pretrained model as fallback...")
    member2_model = models.efficientnet_b0(pretrained=True)
    member2_model.classifier[1] = nn.Linear(member2_model.classifier[1].in_features, 1)
    member2_model = member2_model.to(device)
    member2_model.eval()


# ======================
# FUSION MODEL ARCHITECTURE
# ======================
class FusionModel(nn.Module):
    def __init__(self, image_model, meta_input_dim, meta_hidden=[128, 64],
                 fusion_hidden=[512, 256], dropout=0.3):
        super(FusionModel, self).__init__()
        self.image_model = image_model

        # ðŸ”¹ Get actual EfficientNet feature size
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(next(image_model.parameters()).device)
            img_out_dim = image_model.extract_features(dummy).shape[1]

        # Metadata MLP
        layers = []
        prev_dim = meta_input_dim
        for h in meta_hidden:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h
        self.meta_mlp = nn.Sequential(*layers)
        meta_out_dim = prev_dim

        # Fusion MLP
        fusion_layers = []
        prev_dim = img_out_dim + meta_out_dim
        for h in fusion_hidden:
            fusion_layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h
        fusion_layers.append(nn.Linear(prev_dim, 1))
        self.fusion = nn.Sequential(*fusion_layers)

    def forward(self, images, metadata):
        # Extract features properly
        image_features = self.image_model.extract_features(images)
        meta_features = self.meta_mlp(metadata)
        combined = torch.cat([image_features, meta_features], dim=1)
        return self.fusion(combined)

# Create fusion model
fusion_model = FusionModel(
    image_model=member2_model, 
    meta_input_dim=len(feature_columns)
).to(device)
print(f"Fusion model created. Trainable parameters: {sum(p.numel() for p in fusion_model.parameters() if p.requires_grad):,}")

# ======================
# FUSION DATASET
# ======================
from torch.utils.data import Dataset
from PIL import Image

class FusionDataset(Dataset):
    def __init__(self, dataframe, feature_columns, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
    
        # Load image
        image = Image.open(row['image_path']).convert("RGB")
        if self.transform:
            image = self.transform(image)
    
        # Metadata as tensor (force numeric)
        metadata = row[self.feature_columns].astype(float).values
        metadata = torch.tensor(metadata, dtype=torch.float32)
    
        # Target label
        target = torch.tensor(float(row['target']), dtype=torch.float32)
    
        return (image, metadata), target


# ======================
# SUBSET TRAINING 
# ======================
def create_balanced_subset(df, subset_size=5000, min_positive_ratio=0.05):
    """Create a subset that maintains better class balance"""
    
    # Separate positive and negative cases
    positive_cases = df[df['target'] == 1]
    negative_cases = df[df['target'] == 0]
    
    # Calculate how many positives we want (at least 5% of subset)
    min_positives = int(subset_size * min_positive_ratio)
    available_positives = len(positive_cases)
    
    # Use all available positives if we don't have enough
    n_positives = min(min_positives, available_positives)
    n_negatives = subset_size - n_positives
    
    print(f"Creating balanced subset:")
    print(f"  Available positives: {available_positives}")
    print(f"  Using positives: {n_positives}")
    print(f"  Using negatives: {n_negatives}")
    
    # Sample
    if n_positives > 0:
        positive_subset = positive_cases.sample(n=n_positives, random_state=42)
    else:
        positive_subset = pd.DataFrame()
    
    negative_subset = negative_cases.sample(n=n_negatives, random_state=42)
    
    # Combine
    balanced_subset = pd.concat([positive_subset, negative_subset])
    
    print(f"Subset created:")
    print(f"  Total: {len(balanced_subset)}")
    print(f"  Positive: {len(positive_subset)} ({len(positive_subset)/len(balanced_subset)*100:.1f}%)")
    print(f"  Negative: {len(negative_subset)} ({len(negative_subset)/len(balanced_subset)*100:.1f}%)")
    
    return balanced_subset.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

# Create balanced subsets for faster training
print("Creating balanced subsets for faster training...")
train_subset = create_balanced_subset(train_df, subset_size=3000, min_positive_ratio=0.05)
val_subset = create_balanced_subset(val_df, subset_size=600, min_positive_ratio=0.05)

print(f"Using subsets: Train {len(train_subset)}, Val {len(val_subset)}")


transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create subset datasets and loaders
train_subset_dataset = FusionDataset(train_subset, feature_columns, transform)
val_subset_dataset = FusionDataset(val_subset, feature_columns, transform)

train_subset_loader = DataLoader(train_subset_dataset, batch_size=94, shuffle=True, num_workers=2)
val_subset_loader = DataLoader(val_subset_dataset, batch_size=86, shuffle=False, num_workers=2)

print(f"Subset loaders created: {len(train_subset_loader)} train batches, {len(val_subset_loader)} val batches")

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []
    
    for (images, metadata), targets in tqdm(dataloader, desc='Training'):
        images, metadata, targets = images.to(device), metadata.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, metadata).squeeze()
        loss = criterion(outputs, targets.squeeze())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        probs = torch.sigmoid(outputs)
        
        all_preds.extend((probs > 0.5).float().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_auc = roc_auc_score(all_targets, all_probs)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_auc, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []
    
    with torch.no_grad():
        for (images, metadata), targets in tqdm(dataloader, desc='Validation'):
            images, metadata, targets = images.to(device), metadata.to(device), targets.to(device)
            
            outputs = model(images, metadata).squeeze()
            loss = criterion(outputs, targets.squeeze())
            
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            
            all_preds.extend((probs > 0.5).float().cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_auc = roc_auc_score(all_targets, all_probs)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_auc, epoch_acc

# ======================
# TRAINING LOOP WITH SUBSETS
# ======================
# Compute class weights
pos_weight = torch.tensor([len(train_subset[train_subset['target']==0]) / 
                           len(train_subset[train_subset['target']==1])]).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(fusion_model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)

# Create models directory if it doesn't exist
os.makedirs('/kaggle/working/models', exist_ok=True)

# Generate unique model filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
best_model_path = f'/kaggle/working/models/best_fusion_model_{timestamp}.pth'

best_auc = 0.0
epochs_without_improvement = 0
early_stopping_patience = 5  # Stop if no improvement for 5 epochs
history = {'train_loss': [], 'train_auc': [], 'train_acc': [], 'val_loss': [], 'val_auc': [],
           'val_acc':[]}

print("Starting fusion model training with subsets...")
print(f"Early stopping enabled with patience: {early_stopping_patience}")
print(f"Best model will be saved to: {best_model_path}")
print("=" * 50)

for epoch in range(20):
    print(f"\nEpoch {epoch+1}/20")
    print("-" * 40)
    
    # Train with subset
    train_loss, train_auc, train_acc = train_epoch(fusion_model, train_subset_loader, criterion, optimizer, device)
    
    # Validate with subset
    val_loss, val_auc, val_acc = validate_epoch(fusion_model, val_subset_loader, criterion, device)
    
    # Update history
    history['train_loss'].append(train_loss)
    history['train_auc'].append(train_auc)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_auc'].append(val_auc)
    history['val_acc'].append(val_acc)
    
    # Save best model
    if val_auc > best_auc:
        best_auc = val_auc
        epochs_without_improvement = 0
        torch.save(fusion_model.state_dict(), 'best_fusion_model.pth')
        print(f"New best model saved with AUC: {best_auc:.4f}")
        print(f"Model saved to: {best_model_path}")

        # Immediate verification
        if os.path.exists(best_model_path):
            file_size = os.path.getsize(best_model_path) / (1024 * 1024)
            print(f"File verified: {file_size:.2f} MB")
        else:
            print("Warning: File not found immediately after saving!")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s)")
    
    # Learning rate scheduling
    scheduler.step(val_auc)
    
    print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

    # Early stopping check
    if epochs_without_improvement >= early_stopping_patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs!")
        print(f"No improvement for {early_stopping_patience} consecutive epochs.")
        break
    
print(f"\nTraining completed! Best Validation AUC: {best_auc:.4f}")

# ======================
# MODEL VERIFICATION & DOWNLOAD LINKS
# ======================
print("\n" + "="*60)
print("MODEL SAVING VERIFICATION")
print("="*60)

# Check if model file exists
model_file_exists = os.path.exists(best_model_path)

print(f"Best model path: {best_model_path}")
print(f"Model file exists: {'True' if model_file_exists else 'False'}")

if model_file_exists:
    # Get file size
    file_size = os.path.getsize(best_model_path) / (1024 * 1024)
    print(f"Model file size: {file_size:.2f} MB")
    
    # Load the best model
    try:
        fusion_model.load_state_dict(torch.load(best_model_path))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    # Create download link
    from IPython.display import FileLink, display
    print("\nDOWNLOAD LINK:")
    display(FileLink(best_model_path, result_html_prefix="Click to download: "))
else:
    print("Model file not found! Checking directory...")
    print("\nFILES IN /kaggle/working/:")
    !ls -la /kaggle/working/
    print("\nFILES IN /kaggle/working/models/:")
    !ls -la /kaggle/working/models/ 2>/dev/null || echo "Models directory doesn't exist"

# Force sync to ensure files are visible
!sync
print("\nFile system synced. Check Kaggle file browser on the right.")

print(f"\nBest model should be at: {best_model_path}")




