# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

# Image processing and computer vision
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Deep learning frameworks
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# Alternative: TensorFlow/Keras (uncomment if using TF instead of PyTorch)
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, models, optimizers, losses, metrics
# from tensorflow.keras.applications import EfficientNetB3, ResNet50
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Metrics and evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Kaggle specific (if running on Kaggle)
# import kaggle

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Display settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)


# Dataset paths (adjust paths according to your Kaggle dataset)
# If running on Kaggle, the dataset should be in /kaggle/input/
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images"

# For local development, create sample data structure
if not os.path.exists(DATA_DIR):
    print("Creating sample dataset structure for local development...")
    os.makedirs("sample_data/train_images", exist_ok=True)
    DATA_DIR = "sample_data"
    TRAIN_DIR = "sample_data/train_images"

# Load the dataset labels
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")

# Create synthetic severity labels for demonstration (in real scenario, this would be provided)
np.random.seed(42)
severity_mapping = {
    0: (0, 2),    # Cassava Bacterial Blight (CBB) - low to moderate severity
    1: (3, 6),    # Cassava Brown Streak Disease (CBSD) - moderate to high severity  
    2: (1, 4),    # Cassava Green Mottle (CGM) - low to moderate severity
    3: (0, 1),    # Cassava Mosaic Disease (CMD) - very low severity
    4: (0, 0.5)   # Healthy - minimal severity
}

def generate_severity_score(label):
    """Generate realistic severity scores based on disease type"""
    min_sev, max_sev = severity_mapping[label]
    return np.random.uniform(min_sev, max_sev)

train_df['severity'] = train_df['label'].apply(generate_severity_score)

# Disease class names
class_names = {
    0: 'Cassava Bacterial Blight (CBB)',
    1: 'Cassava Brown Streak Disease (CBSD)', 
    2: 'Cassava Green Mottle (CGM)',
    3: 'Cassava Mosaic Disease (CMD)',
    4: 'Healthy'
}

train_df['disease_name'] = train_df['label'].map(class_names)

print(f"Dataset shape: {train_df.shape}")
print(f"Number of classes: {train_df['label'].nunique()}")
print(f"Images directory: {TRAIN_DIR}")
print("\nFirst few rows:")
print(train_df.head())

print("\nClass distribution:")
print(train_df['disease_name'].value_counts())

print(f"\nSeverity statistics:")
print(train_df['severity'].describe())


# Visualize class distribution and severity distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Class distribution
train_df['disease_name'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Disease Class Distribution')
axes[0].set_xlabel('Disease Type')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

# Severity distribution
axes[1].hist(train_df['severity'], bins=30, color='lightcoral', alpha=0.7)
axes[1].set_title('Severity Score Distribution')
axes[1].set_xlabel('Severity Score')
axes[1].set_ylabel('Frequency')

# Severity by class
sns.boxplot(data=train_df, x='label', y='severity', ax=axes[2])
axes[2].set_title('Severity by Disease Class')
axes[2].set_xlabel('Disease Class')
axes[2].set_ylabel('Severity Score')

plt.tight_layout()
plt.show()

# Split the dataset
train_data, val_data = train_test_split(
    train_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_df['label']
)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

# Reset indices
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)


# Image parameters
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5

# Data augmentation transforms
train_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=45, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.OneOf([
        A.MotionBlur(blur_limit=3),
        A.MedianBlur(blur_limit=3),
        A.GaussianBlur(blur_limit=3),
    ], p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class CassavaDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, row['image_id'])
        
        # Handle case where image might not exist (for demo purposes)
        if not os.path.exists(image_path):
            # Create a dummy image for demonstration
            image = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Get targets
        classification_target = torch.tensor(row['label'], dtype=torch.long)
        regression_target = torch.tensor(row['severity'], dtype=torch.float32)
        
        return image, classification_target, regression_target

# Create datasets
train_dataset = CassavaDataset(train_data, TRAIN_DIR, train_transforms)
val_dataset = CassavaDataset(val_data, TRAIN_DIR, val_transforms)

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=2,
    pin_memory=True
)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# Test the data loader
sample_batch = next(iter(train_loader))
images, class_labels, severity_labels = sample_batch
print(f"Batch shapes:")
print(f"Images: {images.shape}")
print(f"Classification labels: {class_labels.shape}")
print(f"Regression labels: {severity_labels.shape}")


class MultiOutputCassavaModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(MultiOutputCassavaModel, self).__init__()
        
        # Shared feature extractor (backbone)
        self.backbone = models.efficientnet_b3(pretrained=pretrained)
        
        # Get the number of features from the backbone
        num_features = self.backbone.classifier[-1].in_features
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Shared feature projection
        self.shared_features = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Regression head  
        self.regression_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        # Store intermediate activations for visualization
        self.features_for_visualization = {}
        self.register_hooks()
    
    def register_hooks(self):
        """Register hooks to capture intermediate activations"""
        def hook_fn(name):
            def hook(module, input, output):
                self.features_for_visualization[name] = output.detach()
            return hook
        
        # Register hooks for different layers
        self.backbone.features[4].register_forward_hook(hook_fn('conv_block_1'))
        self.backbone.features[6].register_forward_hook(hook_fn('conv_block_2'))
        self.shared_features[0].register_forward_hook(hook_fn('shared_features'))
    
    def forward(self, x):
        # Extract features using backbone
        backbone_features = self.backbone(x)
        
        # Apply shared feature projection
        shared_features = self.shared_features(backbone_features)
        
        # Get outputs from both heads
        classification_output = self.classification_head(shared_features)
        regression_output = self.regression_head(shared_features)
        
        return classification_output, regression_output.squeeze()
    
    def get_feature_maps(self):
        """Return stored feature maps for visualization"""
        return self.features_for_visualization

# Create the model
model = MultiOutputCassavaModel(num_classes=NUM_CLASSES, pretrained=True)
model = model.to(device)

# Model summary
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Model created successfully!")
print(f"Total trainable parameters: {total_params:,}")

# Test the model with a sample batch
model.eval()
with torch.no_grad():
    
    sample_images, _, _ = next(iter(val_loader))
    sample_images = sample_images.to(device)
    
    class_output, regression_output = model(sample_images)
    print(f"\nSample outputs:")
    print(f"Classification output shape: {class_output.shape}")
    print(f"Regression output shape: {regression_output.shape}")
    print(f"Classification probabilities (first sample): {F.softmax(class_output[0], dim=0)}")
    print(f"Regression prediction (first sample): {regression_output[0].item():.3f}")


class MultiTaskLoss(nn.Module):
    def __init__(self, classification_weight=1.0, regression_weight=1.0, adaptive_weights=False):
        super(MultiTaskLoss, self).__init__()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.adaptive_weights = adaptive_weights
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        
        # For adaptive weighting (learnable parameters)
        if adaptive_weights:
            self.log_classification_sigma = nn.Parameter(torch.zeros(1))
            self.log_regression_sigma = nn.Parameter(torch.zeros(1))
    
    def forward(self, class_pred, class_target, reg_pred, reg_target):
        # Calculate individual losses
        cls_loss = self.classification_loss(class_pred, class_target)
        reg_loss = self.regression_loss(reg_pred, reg_target)
        
        if self.adaptive_weights:
            # Adaptive weighting based on uncertainty (as in "Multi-Task Learning Using Uncertainty")
            precision_cls = torch.exp(-self.log_classification_sigma)
            precision_reg = torch.exp(-self.log_regression_sigma)
            
            # Weighted losses with uncertainty
            weighted_cls_loss = precision_cls * cls_loss + self.log_classification_sigma
            weighted_reg_loss = precision_reg * reg_loss + self.log_regression_sigma
            
            total_loss = weighted_cls_loss + weighted_reg_loss
            
            return total_loss, cls_loss, reg_loss, precision_cls.item(), precision_reg.item()
        else:
            # Fixed weighting
            total_loss = (self.classification_weight * cls_loss + 
                         self.regression_weight * reg_loss)
            
            return total_loss, cls_loss, reg_loss

# Create loss function
# Option 1: Fixed weights
criterion = MultiTaskLoss(classification_weight=1.0, regression_weight=0.5, adaptive_weights=False)

# Option 2: Adaptive weights (uncomment to use)
# criterion = MultiTaskLoss(adaptive_weights=True)

print("Custom multi-task loss function created!")
print(f"Classification weight: {criterion.classification_weight}")
print(f"Regression weight: {criterion.regression_weight}")
print(f"Adaptive weights: {criterion.adaptive_weights}")

# Test the loss function
model.train()
with torch.no_grad():
    sample_images, sample_class_labels, sample_reg_labels = next(iter(train_loader))
    sample_images = sample_images.to(device)
    sample_class_labels = sample_class_labels.to(device)
    sample_reg_labels = sample_reg_labels.to(device)
    
    class_output, reg_output = model(sample_images)
    
    if criterion.adaptive_weights:
        total_loss, cls_loss, reg_loss, cls_precision, reg_precision = criterion(
            class_output, sample_class_labels, reg_output, sample_reg_labels
        )
        print(f"\nSample loss calculation (adaptive):")
        print(f"Total loss: {total_loss.item():.4f}")
        print(f"Classification loss: {cls_loss.item():.4f}")
        print(f"Regression loss: {reg_loss.item():.4f}")
        print(f"Classification precision: {cls_precision:.4f}")
        print(f"Regression precision: {reg_precision:.4f}")
    else:
        total_loss, cls_loss, reg_loss = criterion(
            class_output, sample_class_labels, reg_output, sample_reg_labels
        )
        print(f"\nSample loss calculation (fixed):")
        print(f"Total loss: {total_loss.item():.4f}")
        print(f"Classification loss: {cls_loss.item():.4f}")
        print(f"Regression loss: {reg_loss.item():.4f}")


# Training parameters
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Optimizer and schedulerf
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

# Training history
history = {
    'train_loss': [], 'train_cls_loss': [], 'train_reg_loss': [],
    'train_cls_acc': [], 'train_reg_mae': [],
    'val_loss': [], 'val_cls_loss': [], 'val_reg_loss': [],
    'val_cls_acc': [], 'val_reg_mae': []
}

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    cls_loss_sum = 0
    reg_loss_sum = 0
    cls_correct = 0
    reg_mae_sum = 0
    total_samples = 0
    
    for batch_idx, (images, class_labels, reg_labels) in enumerate(train_loader):
        images = images.to(device)
        class_labels = class_labels.to(device)
        reg_labels = reg_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        class_output, reg_output = model(images)
        
        # Calculate loss
        if criterion.adaptive_weights:
            loss, cls_loss, reg_loss, _, _ = criterion(class_output, class_labels, reg_output, reg_labels)
        else:
            loss, cls_loss, reg_loss = criterion(class_output, class_labels, reg_output, reg_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        cls_loss_sum += cls_loss.item()
        reg_loss_sum += reg_loss.item()
        
        # Classification accuracy
        _, predicted = torch.max(class_output.data, 1)
        cls_correct += (predicted == class_labels).sum().item()
        
        # Regression MAE
        reg_mae_sum += torch.abs(reg_output - reg_labels).sum().item()
        total_samples += images.size(0)
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = cls_loss_sum / len(train_loader)
    avg_reg_loss = reg_loss_sum / len(train_loader)
    cls_accuracy = cls_correct / total_samples
    reg_mae = reg_mae_sum / total_samples
    
    return avg_loss, avg_cls_loss, avg_reg_loss, cls_accuracy, reg_mae

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    cls_loss_sum = 0
    reg_loss_sum = 0
    cls_correct = 0
    reg_mae_sum = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, class_labels, reg_labels in val_loader:
            images = images.to(device)
            class_labels = class_labels.to(device)
            reg_labels = reg_labels.to(device)
            
            # Forward pass
            class_output, reg_output = model(images)
            
            # Calculate loss
            if criterion.adaptive_weights:
                loss, cls_loss, reg_loss, _, _ = criterion(class_output, class_labels, reg_output, reg_labels)
            else:
                loss, cls_loss, reg_loss = criterion(class_output, class_labels, reg_output, reg_labels)
            
            # Accumulate metrics
            total_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            reg_loss_sum += reg_loss.item()
            
            # Classification accuracy
            _, predicted = torch.max(class_output.data, 1)
            cls_correct += (predicted == class_labels).sum().item()
            
            # Regression MAE
            reg_mae_sum += torch.abs(reg_output - reg_labels).sum().item()
            total_samples += images.size(0)
    
    avg_loss = total_loss / len(val_loader)
    avg_cls_loss = cls_loss_sum / len(val_loader)
    avg_reg_loss = reg_loss_sum / len(val_loader)
    cls_accuracy = cls_correct / total_samples
    reg_mae = reg_mae_sum / total_samples
    
    return avg_loss, avg_cls_loss, avg_reg_loss, cls_accuracy, reg_mae

print("Starting training...")
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 30)
    
    # Training
    train_loss, train_cls_loss, train_reg_loss, train_cls_acc, train_reg_mae = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    
    # Validation
    val_loss, val_cls_loss, val_reg_loss, val_cls_acc, val_reg_mae = validate_epoch(
        model, val_loader, criterion, device
    )
    
    # Update scheduler
    scheduler.step()
    
    # Save metrics
    history['train_loss'].append(train_loss)
    history['train_cls_loss'].append(train_cls_loss)
    history['train_reg_loss'].append(train_reg_loss)
    history['train_cls_acc'].append(train_cls_acc)
    history['train_reg_mae'].append(train_reg_mae)
    
    history['val_loss'].append(val_loss)
    history['val_cls_loss'].append(val_cls_loss)
    history['val_reg_loss'].append(val_reg_loss)
    history['val_cls_acc'].append(val_cls_acc)
    history['val_reg_mae'].append(val_reg_mae)
    
    # Print epoch results
    print(f"Train - Loss: {train_loss:.4f}, Cls Acc: {train_cls_acc:.4f}, Reg MAE: {train_reg_mae:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, Cls Acc: {val_cls_acc:.4f}, Reg MAE: {val_reg_mae:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_multitask_model.pth')
        print("âœ“ New best model saved!")

print("\nTraining completed!")


# Plot training history
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Total loss
axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
axes[0, 0].set_title('Total Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Classification loss
axes[0, 1].plot(history['train_cls_loss'], label='Train Cls Loss', color='blue')
axes[0, 1].plot(history['val_cls_loss'], label='Val Cls Loss', color='red')
axes[0, 1].set_title('Classification Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Regression loss
axes[0, 2].plot(history['train_reg_loss'], label='Train Reg Loss', color='blue')
axes[0, 2].plot(history['val_reg_loss'], label='Val Reg Loss', color='red')
axes[0, 2].set_title('Regression Loss')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Loss')
axes[0, 2].legend()
axes[0, 2].grid(True)

# Classification accuracy
axes[1, 0].plot(history['train_cls_acc'], label='Train Acc', color='blue')
axes[1, 0].plot(history['val_cls_acc'], label='Val Acc', color='red')
axes[1, 0].set_title('Classification Accuracy')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Regression MAE
axes[1, 1].plot(history['train_reg_mae'], label='Train MAE', color='blue')
axes[1, 1].plot(history['val_reg_mae'], label='Val MAE', color='red')
axes[1, 1].set_title('Regression MAE')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('MAE')
axes[1, 1].legend()
axes[1, 1].grid(True)

# Learning rate
lr_history = [group['lr'] for group in optimizer.param_groups]
axes[1, 2].plot(lr_history, color='green')
axes[1, 2].set_title('Learning Rate')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('LR')
axes[1, 2].grid(True)

plt.tight_layout()
plt.show()

# Load best model for final evaluation
model.load_state_dict(torch.load('best_multitask_model.pth'))
model.eval()

# Comprehensive evaluation
all_class_preds = []
all_class_true = []
all_reg_preds = []
all_reg_true = []

with torch.no_grad():
    for images, class_labels, reg_labels in val_loader:
        images = images.to(device)
        class_labels = class_labels.to(device)
        reg_labels = reg_labels.to(device)
        
        class_output, reg_output = model(images)
        
        # Get predictions
        _, class_preds = torch.max(class_output, 1)
        
        # Store results
        all_class_preds.extend(class_preds.cpu().numpy())
        all_class_true.extend(class_labels.cpu().numpy())
        all_reg_preds.extend(reg_output.cpu().numpy())
        all_reg_true.extend(reg_labels.cpu().numpy())

# Convert to numpy arrays
all_class_preds = np.array(all_class_preds)
all_class_true = np.array(all_class_true)
all_reg_preds = np.array(all_reg_preds)
all_reg_true = np.array(all_reg_true)

# Classification metrics
cls_accuracy = accuracy_score(all_class_true, all_class_preds)
print("=== CLASSIFICATION RESULTS ===")
print(f"Accuracy: {cls_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(all_class_true, all_class_preds, 
                          target_names=[class_names[i] for i in range(NUM_CLASSES)]))

# Regression metrics
reg_mse = mean_squared_error(all_reg_true, all_reg_preds)
reg_mae = mean_absolute_error(all_reg_true, all_reg_preds)
reg_r2 = r2_score(all_reg_true, all_reg_preds)

print("\n=== REGRESSION RESULTS ===")
print(f"MSE: {reg_mse:.4f}")
print(f"MAE: {reg_mae:.4f}")
print(f"RÂ²: {reg_r2:.4f}")

# Confusion matrix and regression scatter plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Confusion matrix
cm = confusion_matrix(all_class_true, all_class_preds)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], cmap='Blues',
            xticklabels=[class_names[i] for i in range(NUM_CLASSES)],
            yticklabels=[class_names[i] for i in range(NUM_CLASSES)])
axes[0].set_title('Classification Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Regression scatter plot
axes[1].scatter(all_reg_true, all_reg_preds, alpha=0.6, color='blue')
axes[1].plot([min(all_reg_true), max(all_reg_true)], 
             [min(all_reg_true), max(all_reg_true)], 'r--', lw=2)
axes[1].set_xlabel('True Severity')
axes[1].set_ylabel('Predicted Severity')
axes[1].set_title(f'Regression Results (RÂ² = {reg_r2:.3f})')
axes[1].grid(True)

plt.tight_layout()
plt.show()


def visualize_feature_maps(model, sample_image, layer_names, max_features=16):
    """
    Visualize feature maps from intermediate layers
    """
    model.eval()
    
    # Forward pass to get feature maps
    with torch.no_grad():
        if len(sample_image.shape) == 3:
            sample_image = sample_image.unsqueeze(0)  # Add batch dimension
        
        sample_image = sample_image.to(device)
        class_output, reg_output = model(sample_image)
        feature_maps = model.get_feature_maps()
    
    # Get predictions
    class_pred = torch.softmax(class_output, dim=1)
    predicted_class = torch.argmax(class_pred, dim=1).item()
    class_confidence = class_pred[0, predicted_class].item()
    severity_pred = reg_output.item()
    
    print(f"Predictions:")
    print(f"  Class: {class_names[predicted_class]} (confidence: {class_confidence:.3f})")
    print(f"  Severity: {severity_pred:.3f}")
    
    # Create visualization
    n_layers = len(layer_names)
    fig, axes = plt.subplots(n_layers + 1, max_features + 1, figsize=(20, 4 * (n_layers + 1)))
    
    # Original image
    original_img = sample_image[0].cpu().numpy()
    # Denormalize for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original_img = original_img.transpose(1, 2, 0)
    original_img = std * original_img + mean
    original_img = np.clip(original_img, 0, 1)
    
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title(f'Original\\nClass: {class_names[predicted_class]}\\nSeverity: {severity_pred:.2f}')
    axes[0, 0].axis('off')
    
    # Hide unused subplots in the first row
    for i in range(1, max_features + 1):
        axes[0, i].axis('off')
    
    # Visualize feature maps for each layer
    for layer_idx, layer_name in enumerate(layer_names):
        if layer_name in feature_maps:
            features = feature_maps[layer_name][0]  # First sample in batch
            n_features = min(features.shape[0], max_features)
            
            for i in range(n_features):
                try:
                    feature_map = features[i].cpu().numpy()
                    if feature_map.ndim != 2:
                        raise ValueError(f"Invalid feature map shape: {feature_map.shape}")
                    
                    # Normalize
                    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
    
                    ax = axes[layer_idx + 1, i]
                    ax.imshow(feature_map, cmap='viridis')
                    ax.set_title(f'Feature {i}')
                    ax.axis('off')
                except Exception as e:
                    ax = axes[layer_idx + 1, i]
                    ax.text(0.5, 0.5, 'Skipped\\nInvalid shape', ha='center', va='center')
                    ax.set_title(f'Feature {i} (invalid)')
                    ax.axis('off')
    
            # Hide unused slots
            for i in range(n_features, max_features + 1):
                axes[layer_idx + 1, i].axis('off')
            
            # Layer name
            axes[layer_idx + 1, 0].text(-0.1, 0.5, f'{layer_name}\\n({features.shape[0]} features)', 
                                        transform=axes[layer_idx + 1, 0].transAxes,
                                        rotation=90, va='center', ha='right', fontsize=10)

    
    plt.tight_layout()
    plt.show()

def visualize_class_activation_maps(model, sample_image, target_layer_name='conv_block_2'):
    """
    Generate and visualize Class Activation Maps (CAM)
    """
    model.eval()
    
    # Get feature maps and gradients
    feature_maps = {}
    gradients = {}
    
    def forward_hook(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach()
        return hook
    
    def backward_hook(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()
        return hook
    
    # Register hooks
    if target_layer_name == 'conv_block_1':
        layer = model.backbone.features[4]
    elif target_layer_name == 'conv_block_2':
        layer = model.backbone.features[6]
    else:
        layer = model.backbone.features[-1]
    
    forward_handle = layer.register_forward_hook(forward_hook(target_layer_name))
    backward_handle = layer.register_backward_hook(backward_hook(target_layer_name))
    
    try:
        if len(sample_image.shape) == 3:
            sample_image = sample_image.unsqueeze(0)
        
        sample_image = sample_image.to(device)
        sample_image.requires_grad_(True)
        
        # Forward pass
        class_output, reg_output = model(sample_image)
        
        # Get predicted class
        predicted_class = torch.argmax(class_output, dim=1)
        
        # Backward pass for the predicted class
        model.zero_grad()
        class_score = class_output[0, predicted_class]
        class_score.backward()
        
        # Generate CAM
        if target_layer_name in gradients and target_layer_name in feature_maps:
            grads = gradients[target_layer_name][0]
            features = feature_maps[target_layer_name][0]
            
            # Global average pooling of gradients
            weights = torch.mean(grads, dim=(1, 2))
            
            # Weighted combination of feature maps
            cam = torch.zeros(features.shape[1:], dtype=torch.float32,device=features.device)
            for i, w in enumerate(weights):
                cam += w * features[i]
            
            # ReLU and normalize
            cam = F.relu(cam)
            cam = cam.cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            # Resize to input image size
            cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
            
            # Prepare original image for display
            original_img = sample_image[0].cpu().detach().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            original_img = original_img.transpose(1, 2, 0)
            original_img = std * original_img + mean
            original_img = np.clip(original_img, 0, 1)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(original_img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # CAM heatmap
            axes[1].imshow(cam_resized, cmap='jet')
            axes[1].set_title(f'Class Activation Map\\n({target_layer_name})')
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(original_img)
            axes[2].imshow(cam_resized, cmap='jet', alpha=0.5)
            axes[2].set_title('CAM Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            return cam_resized
        
    finally:
        forward_handle.remove()
        backward_handle.remove()

# Get a sample image for visualization
sample_batch = next(iter(val_loader))
sample_images, sample_class_labels, sample_reg_labels = sample_batch

# Select first image from batch
sample_image = sample_images[0]
true_class = sample_class_labels[0].item()
true_severity = sample_reg_labels[0].item()

print(f"True labels - Class: {class_names[true_class]}, Severity: {true_severity:.3f}")
print("\\n" + "="*60)

# Visualize feature maps from different layers
layer_names = ['conv_block_1', 'conv_block_2', 'shared_features']
visualize_feature_maps(model, sample_image, layer_names, max_features=12)

print("\\n" + "="*60)
print("Generating Class Activation Maps...")

# Generate CAM for different layers
for layer_name in ['conv_block_1', 'conv_block_2']:
    print(f"\\n--- {layer_name} ---")
    cam = visualize_class_activation_maps(model, sample_image, layer_name)

# Analyze multiple samples
print("\\n" + "="*60)
print("Analyzing feature activation patterns across different disease types...")

# Get samples from each class
samples_per_class = {}
for images, class_labels, reg_labels in val_loader:
    for i, label in enumerate(class_labels):
        label_item = label.item()
        if label_item not in samples_per_class:
            samples_per_class[label_item] = []
        if len(samples_per_class[label_item]) < 2:  # Collect 2 samples per class
            samples_per_class[label_item].append((images[i], reg_labels[i].item()))
    
    # Break if we have samples for all classes
    if all(len(samples) >= 2 for samples in samples_per_class.values()):
        break

# Visualize CAM for one sample from each class
fig, axes = plt.subplots(NUM_CLASSES, 3, figsize=(12, 4 * NUM_CLASSES))

for class_idx in range(NUM_CLASSES):
    if class_idx in samples_per_class:
        sample_img, sample_severity = samples_per_class[class_idx][0]
        
        # Generate CAM
        cam = visualize_class_activation_maps(model, sample_img, 'conv_block_2')
        
        # Prepare original image
        original_img = sample_img.numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_img = std * original_img + mean
        original_img = np.clip(original_img, 0, 1)
        
        # Plot
        axes[class_idx, 0].imshow(original_img)
        axes[class_idx, 0].set_title(f'{class_names[class_idx]}\\nSeverity: {sample_severity:.2f}')
        axes[class_idx, 0].axis('off')
        
        axes[class_idx, 1].imshow(cam, cmap='jet')
        axes[class_idx, 1].set_title('Activation Map')
        axes[class_idx, 1].axis('off')
        
        axes[class_idx, 2].imshow(original_img)
        axes[class_idx, 2].imshow(cam, cmap='jet', alpha=0.5)
        axes[class_idx, 2].set_title('Overlay')
        axes[class_idx, 2].axis('off')

plt.tight_layout()
plt.show()

print("\\nFeature visualization complete!")

