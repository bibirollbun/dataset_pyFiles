# Cell 1: Import libraries and configuration
import os
import json
import tarfile
import shutil
from pathlib import Path
import gc
import warnings
import io
from collections import defaultdict
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import timm

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

print("All libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

class Config:
    # Paths
    DATA_DIR = "/kaggle/input/inaturalist-2019-fgvc6"
    WORK_DIR = "/kaggle/working"
    
    # Model parameters
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = None  
    IMG_SIZE = 224
    BATCH_SIZE = 8  
    NUM_WORKERS = 2
    
    EPOCHS = 150
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 50
    
    MAX_SAMPLES_PER_CLASS = 50   
    MIN_SAMPLES_PER_CLASS = 5    
    MAX_TOTAL_CLASSES = 100      
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mixed precision training
    USE_AMP = True

config = Config()
print("Configuration loaded:")
print(f"Device: {config.DEVICE}")
print(f"Image size: {config.IMG_SIZE}")
print(f"Batch size: {config.BATCH_SIZE}")
print(f"Max samples per class: {config.MAX_SAMPLES_PER_CLASS}")
print(f"Max total classes: {config.MAX_TOTAL_CLASSES}")
print(f"Epochs: {config.EPOCHS}")


# Cell 2: Data loading and processing (NO TAR EXTRACTION)
def check_available_files():
    """Check what files are available"""
    print("Checking available files...")
    
    for file in os.listdir(config.DATA_DIR):
        file_path = os.path.join(config.DATA_DIR, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  {file}: {size_mb:.2f} MB")
    
    return True

def extract_label_from_path(file_path):
    """Extract label from file path: train_val2019/Plants/400/filename.jpg -> Plants_400"""
    try:
        # Split path: train_val2019/Plants/400/filename.jpg
        parts = file_path.split('/')
        if len(parts) >= 3:
            category = parts[1]  # Plants
            subcategory = parts[2]  # 400
            label = f"{category}_{subcategory}"
            return label
        else:
            return "Unknown"
    except:
        return "Unknown"

def load_train_data():
    """Load training data and extract labels from file paths"""
    print("Loading training data...")
    
    train_json_path = os.path.join(config.DATA_DIR, "train2019.json")
    
    # Early return if file not found
    if not os.path.exists(train_json_path):
        print(f"â�Œ Error: {train_json_path} not found!")
        return None
    
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    
    print(f"âœ… Loaded: {len(train_data['images'])} images")
    
    # Extract labels from file paths (no annotations needed)
    image_data = []
    for img in train_data['images']:
        file_name = img['file_name']
        label_name = extract_label_from_path(file_name)
        
        image_data.append({
            'image_id': img['id'],
            'file_name': file_name,
            'label_name': label_name
        })
    
    print(f"âœ… Extracted labels from {len(image_data)} images")
    
    # Convert to DataFrame
    df = pd.DataFrame(image_data)
    
    # Show label distribution
    label_counts = df['label_name'].value_counts()
    print(f"Found {len(label_counts)} unique labels")
    print(f"Label distribution - Min: {label_counts.min()}, Max: {label_counts.max()}")
    
    return df

def aggressive_data_sampling(train_df):
    """Sample training data to fit memory constraints"""
    
    # Early return if no data
    if train_df is None or len(train_df) == 0:
        print("â�Œ No training data provided!")
        return None, None, None
    
    print("Performing data sampling...")
    
    # Count samples per label
    label_counts = train_df['label_name'].value_counts()
    print(f"Total labels available: {len(label_counts)}")
    
    # Filter labels with enough samples
    valid_labels = label_counts[label_counts >= config.MIN_SAMPLES_PER_CLASS].head(config.MAX_TOTAL_CLASSES).index
    print(f"Selected {len(valid_labels)} labels for training")
    
    # Early return if no valid labels
    if len(valid_labels) == 0:
        print("â�Œ No labels with sufficient samples!")
        return None, None, None
    
    # Sample from each selected label
    sampled_dfs = []
    total_samples = 0
    
    for label_name in tqdm(valid_labels, desc="Sampling labels"):
        label_data = train_df[train_df['label_name'] == label_name]
        
        # Sample up to MAX_SAMPLES_PER_CLASS
        sample_size = min(len(label_data), config.MAX_SAMPLES_PER_CLASS)
        if len(label_data) > sample_size:
            label_data = label_data.sample(n=sample_size, random_state=42)
        
        sampled_dfs.append(label_data)
        total_samples += len(label_data)
        
        # Stop if we have enough samples total
        if total_samples > 5000:  # Hard limit for memory
            break
    
    # Early return if no samples
    if not sampled_dfs:
        print("â�Œ No valid samples found!")
        return None, None, None
    
    # Combine sampled data
    train_df_sampled = pd.concat(sampled_dfs, ignore_index=True)
    
    # Create label mapping (string labels -> integer indices)
    unique_labels = sorted(train_df_sampled['label_name'].unique())
    label_mapping = {label_name: i for i, label_name in enumerate(unique_labels)}
    reverse_mapping = {i: label_name for label_name, i in label_mapping.items()}
    
    # Add integer labels
    train_df_sampled['label'] = train_df_sampled['label_name'].map(label_mapping)
    
    # Split into train/val (80/20)
    train_df_sampled = train_df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.8 * len(train_df_sampled))
    
    train_split = train_df_sampled[:split_idx].reset_index(drop=True)
    val_split = train_df_sampled[split_idx:].reset_index(drop=True)
    
    config.NUM_CLASSES = len(unique_labels)
    
    print(f"âœ… Final dataset: {len(train_split)} train, {len(val_split)} val")
    print(f"âœ… Number of classes: {config.NUM_CLASSES}")
    print(f"âœ… Total samples: {len(train_df_sampled)}")
    
    return train_split, val_split, label_mapping

# Main execution
print("ğŸš€ Loading data without extraction...")
check_available_files()

# Load training data (extract labels from file paths)
train_df_raw = load_train_data()

if train_df_raw is not None:
    print("âœ… Raw data loading successful!")
    
    # Show sample of extracted labels
    print("\nSample of extracted labels:")
    print(train_df_raw[['file_name', 'label_name']].head(10))
    
    # Sample data for training
    train_df, val_df, label_mapping = aggressive_data_sampling(train_df_raw)
    
    if train_df is not None:
        print("âœ… Data sampling successful!")
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Classes: {config.NUM_CLASSES}")
        
        # Show class distribution
        print("\nClass distribution:")
        print(train_df['label'].value_counts().head())
        
        # Show sample data
        print("\nSample training data:")
        print(train_df[['file_name', 'label_name', 'label']].head())
        
        # Show some unique labels
        print(f"\nSample unique labels:")
        unique_labels = sorted(train_df['label_name'].unique())
        for i, label in enumerate(unique_labels[:10]):
            print(f"  {i}: {label}")
        if len(unique_labels) > 10:
            print(f"  ... and {len(unique_labels)-10} more")
    else:
        print("â�Œ Data sampling failed!")
else:
    print("â�Œ Data loading failed!")


# Cell 3: Dataset classes (Direct TAR reading - No extraction)
def get_transforms():
    """Get training and validation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE + 32, config.IMG_SIZE + 32)),
        transforms.RandomCrop(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class DirectTarDataset(Dataset):
    """Dataset that reads images directly from tar archive WITHOUT extraction"""
    
    def __init__(self, df, tar_path, transform=None):
        self.df = df.reset_index(drop=True)
        self.tar_path = tar_path
        self.transform = transform
        
        print(f"ğŸ“¦ Creating dataset from tar: {os.path.basename(tar_path)}")
        print(f"ğŸ“Š Dataset size: {len(self.df)} samples")
        
        # Quick validation of tar file
        if not os.path.exists(tar_path):
            print(f"â�Œ TAR file not found: {tar_path}")
            return
        
        # Cache tar members for faster lookup (sample first 1000 to check)
        print("ğŸ”� Validating tar file structure...")
        self.tar_members = {}
        found_count = 0
        
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                members = tar.getmembers()
                print(f"ğŸ“‹ TAR contains {len(members)} total files")
                
                # Index all members
                for member in members:
                    if member.isfile() and member.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.tar_members[member.name] = member
                
                print(f"ğŸ“¸ Found {len(self.tar_members)} image files in TAR")
                
                # Check if our data files exist in TAR
                sample_files = self.df['file_name'].head(10).tolist()
                for file_name in sample_files:
                    if file_name in self.tar_members:
                        found_count += 1
                
                print(f"âœ… Validation: {found_count}/10 sample files found in TAR")
                
        except Exception as e:
            print(f"â�Œ Error reading TAR file: {e}")
            self.tar_members = {}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['file_name']
        label = row['label']
        
        try:
            # Check if file exists in tar
            if file_name not in self.tar_members:
                # Try without train_val2019/ prefix
                alt_name = '/'.join(file_name.split('/')[1:]) if '/' in file_name else file_name
                if alt_name in self.tar_members:
                    file_name = alt_name
                else:
                    raise Exception(f"File not found in TAR: {file_name}")
            
            # Read image directly from TAR
            with tarfile.open(self.tar_path, 'r:gz') as tar:
                member = self.tar_members[file_name]
                file_obj = tar.extractfile(member)
                
                if file_obj is None:
                    raise Exception("Could not extract file from TAR")
                
                # Load image
                image_data = file_obj.read()
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
        except Exception as e:
            # Create fallback image if loading fails
            if idx < 5:  # Only log first few errors
                print(f"âš ï¸� Error loading {file_name}: {e}")
            
            # Create a solid color fallback image
            image = Image.new('RGB', (config.IMG_SIZE, config.IMG_SIZE), (128, 128, 128))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_data_loaders():
    """Create data loaders using direct TAR reading"""
    
    # Early returns for missing data
    if 'train_df' not in globals() or train_df is None:
        print("â�Œ No train_df available!")
        return None, None
    
    if 'val_df' not in globals() or val_df is None:
        print("â�Œ No val_df available!")
        return None, None
    
    print(f"ğŸ“Š Creating data loaders...")
    print(f"   Train samples: {len(train_df)}")
    print(f"   Val samples: {len(val_df)}")
    
    # Find TAR file
    tar_file = "train_val2019.tar.gz"
    tar_path = os.path.join(config.DATA_DIR, tar_file)
    
    # Early return if TAR not found
    if not os.path.exists(tar_path):
        print(f"â�Œ TAR file not found: {tar_path}")
        print("Available files:")
        for f in os.listdir(config.DATA_DIR):
            if f.endswith('.tar.gz'):
                print(f"   ğŸ“¦ {f}")
        return None, None
    
    print(f"âœ… Using TAR file: {tar_file}")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets (NO EXTRACTION!)
    print("ğŸ“‹ Creating datasets (direct TAR reading)...")
    train_dataset = DirectTarDataset(train_df, tar_path, train_transform)
    val_dataset = DirectTarDataset(val_df, tar_path, val_transform)
    
    # Create data loaders with optimized settings for TAR reading
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use 0 workers for TAR files to avoid multiprocessing issues
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Use 0 workers for TAR files
        pin_memory=True
    )
    
    print(f"âœ… Data loaders created!")
    print(f"   ğŸ“¦ Train loader: {len(train_loader)} batches")
    print(f"   ğŸ“¦ Val loader: {len(val_loader)} batches")
    
    return train_loader, val_loader

# Create data loaders if training data is available
if 'train_df' in globals() and train_df is not None:
    print("ğŸš€ Creating data loaders (NO TAR EXTRACTION)...")
    train_loader, val_loader = create_data_loaders()
    
    if train_loader is not None:
        print("âœ… Data loaders created successfully!")
        
        # Test loading a batch
        print("ğŸ§ª Testing batch loading from TAR...")
        try:
            start_time = time.time()
            batch = next(iter(train_loader))
            load_time = time.time() - start_time
            
            images, labels = batch
            print(f"âœ… Batch loaded successfully in {load_time:.2f}s!")
            print(f"   ğŸ“Š Images shape: {images.shape}")
            print(f"   ğŸ“Š Labels shape: {labels.shape}")
            print(f"   ğŸ“Š Label range: {labels.min().item()} - {labels.max().item()}")
            print(f"   ğŸ“Š Memory usage: {images.element_size() * images.nelement() / 1024**2:.1f} MB")
            
            print("\nğŸ�¯ Ready for training with direct TAR reading!")
            
        except Exception as e:
            print(f"â�Œ Error loading batch: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        print("â�Œ Failed to create data loaders!")
else:
    print("â�Œ Cannot create data loaders - no training data available!")(f"ğŸ“Š Train loader: {len(train_loader)} batches")
    print(f"ğŸ“Š Val loader: {len(val_loader)} batches")

# Create data loaders
if 'train_df' in globals() and train_df is not None:
    print("ğŸš€ Creating data loaders...")
    train_loader, val_loader = create_data_loaders()
    
    if train_loader is not None:
        print("âœ… Data loaders created successfully!")
        
        # Test loading a batch
        print("ğŸ§ª Testing batch loading...")
        try:
            batch = next(iter(train_loader))
            images, labels = batch
            print(f"âœ… Batch loaded successfully!")
            print(f"   Images shape: {images.shape}")
            print(f"   Labels shape: {labels.shape}")
            print(f"   Label range: {labels.min().item()} - {labels.max().item()}")
            print(f"   Image dtype: {images.dtype}")
            print(f"   Memory usage: {images.element_size() * images.nelement() / 1024**2:.1f} MB")
            
            print("\nğŸ�¯ Ready for training!")
            
        except Exception as e:
            print(f"â�Œ Error loading batch: {e}")
            
    else:
        print("â�Œ Failed to create data loaders!")
else:
    print("â�Œ Cannot create data loaders - no training data available!")


# Cell 4: Model definition and setup
class iNaturalistModel(nn.Module):
    def __init__(self, num_classes, model_name="efficientnet_b0"):
        super(iNaturalistModel, self).__init__()
        
        if model_name == "efficientnet_b0":
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
            in_features = self.backbone.num_features
        
        # Simplified classifier for smaller dataset
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def calculate_class_weights(train_df):
    """Calculate class weights for imbalanced dataset"""
    labels = train_df['label'].values
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.FloatTensor(class_weights).to(config.DEVICE)

def setup_training():
    """Setup model, optimizer, criterion"""
    
    # Early return if prerequisites not met
    if config.NUM_CLASSES is None:
        print("â�Œ NUM_CLASSES not set!")
        return None, None, None, None, None
        
    if 'train_df' not in globals() or train_df is None:
        print("â�Œ No training data available!")
        return None, None, None, None, None
    
    print("Setting up training components...")
    
    # Create model
    model = iNaturalistModel(config.NUM_CLASSES, config.MODEL_NAME)
    model = model.to(config.DEVICE)
    
    # Use single GPU (avoid DataParallel issues)
    print(f"Using device: {config.DEVICE}")
    
    # Create criterion without class weights initially (to avoid dimension mismatch)
    criterion = nn.CrossEntropyLoss().to(config.DEVICE)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.USE_AMP else None
    
    return model, criterion, optimizer, scheduler, scaler

# Test model creation
if config.NUM_CLASSES is not None:
    print("Testing model...")
    
    try:
        test_model = iNaturalistModel(config.NUM_CLASSES, config.MODEL_NAME)
        test_model = test_model.to(config.DEVICE)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE).to(config.DEVICE)
        output = test_model(dummy_input)
        
        print(f"âœ… Model test successful!")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in test_model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        del test_model, dummy_input, output
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"â�Œ Model test failed: {e}")
else:
    print("â�Œ Cannot test model - NUM_CLASSES not set")

# Setup training components
print("\nğŸ”§ Setting up training...")
model, criterion, optimizer, scheduler, scaler = setup_training()

if model is not None:
    print("âœ… Training setup successful!")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Criterion device: {config.DEVICE}")
    print(f"Mixed precision: {config.USE_AMP}")
else:
    print("â�Œ Training setup failed!")


# Cell 5: Training functions
import time
from torch.cuda.amp import autocast, GradScaler

def train_epoch(model, train_loader, criterion, optimizer, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if config.USE_AMP and scaler is not None:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(val_loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            
            # Forward pass
            if config.USE_AMP:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def save_checkpoint(model, optimizer, epoch, best_acc, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'config': config.__dict__
    }
    torch.save(checkpoint, path)
    print(f"   ğŸ“� Checkpoint saved: {path}")

print("âœ… Training functions defined!")


# Cell 6: Main training loop
def run_training():
    """Main training function"""
    
    # Early returns for missing components
    if model is None:
        print("â�Œ Model not available!")
        return None
        
    if train_loader is None or val_loader is None:
        print("â�Œ Data loaders not available!")
        return None
    
    print("ğŸš€ STARTING TRAINING...")
    print("=" * 60)
    
    # Training tracking variables
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []
    
    print("\nğŸ“‹ TRAINING SETUP:")
    print(f"   ğŸ”§ Device: {config.DEVICE}")
    print(f"   ğŸ¤– Model: {type(model).__name__}")
    print(f"   ğŸ“Š Classes: {config.NUM_CLASSES}")
    print(f"   ğŸ“¦ Train batches: {len(train_loader)}")
    print(f"   ğŸ“¦ Val batches: {len(val_loader)}")
    print(f"   ğŸ�¯ Epochs: {config.EPOCHS}")
    print(f"   ğŸ“š Learning rate: {config.LEARNING_RATE}")
    print(f"   âš¡ Mixed precision: {config.USE_AMP}")
    
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()
        
        print(f"\nğŸ“… Epoch {epoch+1}/{config.EPOCHS}")
        print("-" * 40)
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        
        # Validation phase
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch results
        print(f"\nğŸ“Š Epoch {epoch+1} Results:")
        print(f"   ğŸ�‹ï¸� Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   âœ… Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"   â�±ï¸� Time: {epoch_time:.2f}s")
        
        # GPU Memory usage
        if torch.cuda.is_available():
            current_mem = torch.cuda.memory_allocated() / 1e9
            max_mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"   ğŸ’¾ GPU Memory: {current_mem:.2f}GB / Peak: {max_mem:.2f}GB")
        
        # Check for best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            best_model_path = f"{config.WORK_DIR}/best_model_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch+1, best_val_acc, best_model_path)
            print(f"   ğŸ�‰ New best validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"   â�³ Patience: {patience_counter}/{config.PATIENCE}")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nğŸ›‘ Early stopping triggered after {epoch+1} epochs")
            print(f"   Best validation accuracy: {best_val_acc:.2f}%")
            break
        
        # Memory cleanup
        torch.cuda.empty_cache()
    
    # Training completed
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("ğŸ�‰ TRAINING COMPLETED!")
    print("=" * 60)
    print(f"â�±ï¸� Total training time: {total_time/60:.2f} minutes")
    print(f"ğŸ�† Best validation accuracy: {best_val_acc:.2f}%")
    print(f"ğŸ“Š Epochs completed: {len(train_losses)}")
    
    # Final model save
    final_model_path = f"{config.WORK_DIR}/final_model.pth"
    save_checkpoint(model, optimizer, len(train_losses), best_val_acc, final_model_path)
    
    # Create training history
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'learning_rates': learning_rates,
        'best_val_acc': best_val_acc,
        'total_epochs': len(train_losses),
        'total_time': total_time
    }
    
    return training_history

# Check if we can start training
if 'model' in globals() and model is not None:
    if 'train_loader' in globals() and train_loader is not None:
        print("âœ… All components ready - starting training...")
        
        # Run training
        training_history = run_training()
        
        if training_history is not None:
            print("\nâœ… Training history saved!")
            print("ğŸ�¯ Training completed successfully!")
        else:
            print("\nâ�Œ Training failed!")
    else:
        print("â�Œ Data loaders not ready!")
else:
    print("â�Œ Model not ready!")


# Cell 7: Results visualization and analysis
def plot_training_curves(training_history):
    """Plot training curves"""
    
    # Early return if no training history
    if training_history is None:
        print("â�Œ No training history to plot!")
        return
    
    train_losses = training_history['train_losses']
    val_losses = training_history['val_losses']
    train_accuracies = training_history['train_accuracies']
    val_accuracies = training_history['val_accuracies']
    learning_rates = training_history.get('learning_rates', [])
    
    if len(train_losses) == 0:
        print("â�Œ Empty training history!")
        return
    
    # Create subplots
    n_plots = 3 if learning_rates else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 2:
        axes = [axes[0], axes[1]]
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    axes[0].plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, train_accuracies, 'b-o', label='Train Acc', linewidth=2, markersize=4)
    axes[1].plot(epochs, val_accuracies, 'r-s', label='Val Acc', linewidth=2, markersize=4)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate plot (if available)
    if learning_rates and len(learning_rates) > 0:
        axes[2].plot(epochs, learning_rates, 'g-^', label='Learning Rate', linewidth=2, markersize=4)
        axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(config.WORK_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ğŸ“Š Training curves saved: {save_path}")
    plt.show()

def print_training_summary(training_history):
    """Print comprehensive training summary"""
    
    # Early return if no training history
    if training_history is None:
        print("â�Œ No training history available!")
        return
    
    print("\n" + "="*70)
    print("ğŸ�¯ COMPREHENSIVE TRAINING RESULTS")
    print("="*70)
    
    # Extract data
    train_losses = training_history['train_losses']
    val_losses = training_history['val_losses']
    train_accuracies = training_history['train_accuracies']
    val_accuracies = training_history['val_accuracies']
    best_val_acc = training_history['best_val_acc']
    total_epochs = training_history['total_epochs']
    total_time = training_history['total_time']
    
    # Basic statistics
    best_epoch = val_accuracies.index(max(val_accuracies)) + 1
    final_train_acc = train_accuracies[-1]
    final_val_acc = val_accuracies[-1]
    
    print(f"ğŸ“Š Dataset Information:")
    print(f"   â€¢ Training samples: {len(train_loader.dataset) if 'train_loader' in globals() else 'N/A'}")
    print(f"   â€¢ Validation samples: {len(val_loader.dataset) if 'val_loader' in globals() else 'N/A'}")
    print(f"   â€¢ Number of classes: {config.NUM_CLASSES}")
    print(f"   â€¢ Batch size: {config.BATCH_SIZE}")
    print(f"   â€¢ Image size: {config.IMG_SIZE}x{config.IMG_SIZE}")
    
    print(f"\nğŸ�† Best Performance:")
    print(f"   â€¢ Best epoch: {best_epoch}")
    print(f"   â€¢ Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   â€¢ Best train accuracy at that epoch: {train_accuracies[best_epoch-1]:.2f}%")
    
    print(f"\nğŸ“ˆ Final Performance:")
    print(f"   â€¢ Final train accuracy: {final_train_acc:.2f}%")
    print(f"   â€¢ Final validation accuracy: {final_val_acc:.2f}%")
    print(f"   â€¢ Total epochs completed: {total_epochs}")
    print(f"   â€¢ Improvement from epoch 1: {final_val_acc - val_accuracies[0]:.2f}%")
    
    print(f"\nğŸ“‰ Loss Analysis:")
    print(f"   â€¢ Initial train loss: {train_losses[0]:.4f}")
    print(f"   â€¢ Final train loss: {train_losses[-1]:.4f}")
    print(f"   â€¢ Initial val loss: {val_losses[0]:.4f}")
    print(f"   â€¢ Final val loss: {val_losses[-1]:.4f}")
    print(f"   â€¢ Best val loss: {min(val_losses):.4f}")
    
    print(f"\nâ�±ï¸� Training Time:")
    print(f"   â€¢ Total time: {total_time/60:.1f} minutes")
    print(f"   â€¢ Average time per epoch: {total_time/total_epochs:.1f} seconds")
    
    print(f"\nğŸ’¾ Model Information:")
    print(f"   â€¢ Architecture: {config.MODEL_NAME}")
    print(f"   â€¢ Device: {config.DEVICE}")
    print(f"   â€¢ Mixed precision: {config.USE_AMP}")
    
    # Performance assessment
    print(f"\nğŸ�¯ Performance Assessment:")
    if best_val_acc >= 70:
        print(f"   ğŸ�† EXCELLENT! Your model achieved great accuracy!")
    elif best_val_acc >= 50:
        print(f"   ğŸ‘� GOOD! Solid performance for this dataset.")
    elif best_val_acc >= 30:
        print(f"   ğŸ‘Œ FAIR! Room for improvement with more data/training.")
    else:
        print(f"   ğŸ“ˆ NEEDS IMPROVEMENT! Consider more training or data augmentation.")
    
    print("="*70)

def analyze_results():
    """Main analysis function"""
    
    print("ğŸ”� Analyzing training results...")
    
    # Check if we have training history
    if 'training_history' not in globals() or training_history is None:
        print("â�Œ No training history found!")
        print("Make sure training completed successfully.")
        return
    
    print("âœ… Training history found - creating analysis...")
    
    # Create plots
    plot_training_curves(training_history)
    
    # Print summary
    print_training_summary(training_history)
    
    # Check for saved models
    print("\nğŸ”� Checking saved models...")
    try:
        model_files = [f for f in os.listdir(config.WORK_DIR) if f.endswith('.pth')]
        if model_files:
            print(f"âœ… Found {len(model_files)} model file(s):")
            for f in model_files:
                print(f"   ğŸ“„ {f}")
        else:
            print("â�Œ No model files found")
    except Exception as e:
        print(f"â�Œ Error checking model files: {e}")
    
    print("\nğŸ�‰ Analysis completed!")

# Run analysis if training history exists
if 'training_history' in globals() and training_history is not None:
    analyze_results()
else:
    print("âš ï¸� No training history available yet.")
    print("Run the training cell first to generate results.")


# Cell 8: Model testing and inference
def load_best_model():
    """Load the best trained model"""
    print("ğŸ“¥ Loading best trained model...")
    
    # Find model files
    model_files = []
    for file in os.listdir(config.WORK_DIR):
        if file.endswith('.pth') and ('best' in file.lower() or 'final' in file.lower()):
            model_files.append(file)
    
    # Early return if no models found
    if not model_files:
        print("â�Œ No trained model found!")
        return None, None
    
    # Use the best model
    model_file = None
    for file in model_files:
        if 'best' in file.lower():
            model_file = file
            break
    
    if not model_file:
        model_file = model_files[0]
    
    model_path = os.path.join(config.WORK_DIR, model_file)
    print(f"ğŸ“� Loading: {model_file}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        
        # Create model
        model = iNaturalistModel(config.NUM_CLASSES, config.MODEL_NAME)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(config.DEVICE)
        model.eval()
        
        # Get metadata
        best_acc = checkpoint.get('best_acc', 'Unknown')
        epoch = checkpoint.get('epoch', 'Unknown')
        
        print(f"âœ… Model loaded successfully!")
        print(f"   ğŸ�¯ Best accuracy: {best_acc}")
        print(f"   ğŸ“Š Saved at epoch: {epoch}")
        
        return model, checkpoint
        
    except Exception as e:
        print(f"â�Œ Error loading model: {e}")
        return None, None

def test_model_performance(model):
    """Test model on validation set"""
    
    # Early return if no model or data
    if model is None:
        print("â�Œ No model provided!")
        return
        
    if 'val_loader' not in globals() or val_loader is None:
        print("â�Œ No validation loader available!")
        return
    
    print("ğŸ“Š Testing model performance on validation set...")
    
    try:
        # Get a batch
        batch = next(iter(val_loader))
        images, labels = batch
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Calculate accuracy
            accuracy = (preds == labels).float().mean()
            
            print(f"âœ… Batch test completed!")
            print(f"   ğŸ“Š Batch size: {len(labels)}")
            print(f"   ğŸ�¯ Accuracy: {accuracy*100:.2f}%")
            print(f"   ğŸ“ˆ Correct predictions: {(preds == labels).sum().item()}/{len(labels)}")
            
            # Show some predictions vs ground truth
            print(f"\nğŸ“‹ Sample predictions vs ground truth:")
            for i in range(min(5, len(labels))):
                pred_class = preds[i].item()
                true_class = labels[i].item()
                match = "âœ…" if pred_class == true_class else "â�Œ"
                print(f"   {i+1}. Predicted: {pred_class}, True: {true_class} {match}")
    
    except Exception as e:
        print(f"â�Œ Error testing model: {e}")

def get_class_mapping():
    """Get class index to label name mapping"""
    print("ğŸ�·ï¸� Getting class mapping...")
    
    # Try to get from training data
    if 'train_df' in globals() and train_df is not None:
        if 'label_name' in train_df.columns:
            # Create mapping from training data
            class_mapping = {}
            for _, row in train_df.iterrows():
                label = row['label']
                label_name = row['label_name']
                if label not in class_mapping:
                    class_mapping[label] = label_name
            
            print(f"âœ… Created class mapping from training data: {len(class_mapping)} classes")
            return class_mapping
    
    # Try to get from label_mapping if available
    if 'label_mapping' in globals() and label_mapping is not None:
        # Reverse the mapping: label_name -> index becomes index -> label_name
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        print(f"âœ… Created class mapping from label_mapping: {len(reverse_mapping)} classes")
        return reverse_mapping
    
    # Fallback: Create generic mapping
    class_mapping = {i: f"Class_{i:03d}" for i in range(config.NUM_CLASSES)}
    print(f"âš ï¸� Using generic class mapping: {len(class_mapping)} classes")
    
    return class_mapping

def predict_sample_images(model, num_samples=5):
    """Predict on sample validation images"""
    
    # Early returns
    if model is None:
        print("â�Œ No model provided!")
        return
        
    if 'val_loader' not in globals() or val_loader is None:
        print("â�Œ No validation loader available!")
        return
    
    print(f"ğŸ”� Making predictions on {num_samples} sample images...")
    
    try:
        # Get class mapping
        class_mapping = get_class_mapping()
        
        # Get sample batch
        batch = next(iter(val_loader))
        images, labels = batch
        
        # Use only first few samples
        sample_images = images[:num_samples].to(config.DEVICE)
        sample_labels = labels[:num_samples].to(config.DEVICE)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(sample_images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted_classes = torch.max(outputs, 1)
        
        print("\nğŸ“‹ PREDICTION RESULTS:")
        print("-" * 50)
        
        for i in range(num_samples):
            true_class = sample_labels[i].item()
            pred_class = predicted_classes[i].item()
            confidence = probabilities[i][pred_class].item()
            
            true_label = class_mapping.get(true_class, f"Unknown_{true_class}")
            pred_label = class_mapping.get(pred_class, f"Unknown_{pred_class}")
            
            match = "âœ… CORRECT" if pred_class == true_class else "â�Œ WRONG"
            
            print(f"\nğŸ–¼ï¸� Sample {i+1}:")
            print(f"   True: {true_label} (class {true_class})")
            print(f"   Predicted: {pred_label} (class {pred_class})")
            print(f"   Confidence: {confidence*100:.2f}%")
            print(f"   Result: {match}")
    
    except Exception as e:
        print(f"â�Œ Error making predictions: {e}")

def show_sample_labels():
    """Show sample of the label names we're working with"""
    
    # Early return if no data
    if 'train_df' not in globals() or train_df is None:
        print("â�Œ No training data available!")
        return
    
    print("ğŸ�·ï¸� Sample of label names in dataset:")
    print("-" * 40)
    
    # Get unique labels
    if 'label_name' in train_df.columns:
        unique_labels = sorted(train_df['label_name'].unique())
        
        print(f"Total unique labels: {len(unique_labels)}")
        print("\nSample labels:")
        
        for i, label in enumerate(unique_labels[:15]):  # Show first 15
            count = len(train_df[train_df['label_name'] == label])
            print(f"   {i:2d}: {label} ({count} samples)")
        
        if len(unique_labels) > 15:
            print(f"   ... and {len(unique_labels) - 15} more labels")
            
        # Show some examples from different categories
        print(f"\nLabel format examples:")
        categories = {}
        for label in unique_labels[:20]:
            if '_' in label:
                category = label.split('_')[0]
                if category not in categories:
                    categories[category] = []
                categories[category].append(label)
        
        for category, labels in categories.items():
            print(f"   {category}: {labels[:3]}{'...' if len(labels) > 3 else ''}")
    else:
        print("â�Œ No label_name column found in training data")

def run_model_testing():
    """Main testing function"""
    print("ğŸ§ª STARTING MODEL TESTING...")
    print("=" * 50)
    
    # Show what labels we're working with
    show_sample_labels()
    
    # Load best model
    test_model, checkpoint = load_best_model()
    
    if test_model is not None:
        # Test model performance
        test_model_performance(test_model)
        
        # Make sample predictions
        predict_sample_images(test_model, num_samples=8)
        
        print("\nğŸ�‰ MODEL TESTING COMPLETED!")
        print("=" * 50)
        print(f"âœ… Model loaded and tested successfully")
        print(f"ğŸ�¯ Ready for inference on new images")
        print(f"ğŸ“Š Model can predict {config.NUM_CLASSES} different label types")
        
        return test_model
    else:
        print("â�Œ Could not load model for testing!")
        return None

# Run testing if training is complete
if 'training_history' in globals() and training_history is not None:
    print("âœ… Training completed - running model testing...")
    test_model = run_model_testing()
else:
    print("âš ï¸� Training not completed yet.")
    print("Complete training first, then run this cell for testing.")
    
    # But still show sample labels if available
    if 'train_df' in globals() and train_df is not None:
        print("\n" + "="*50)
        print("ğŸ“‹ PREVIEW OF DATASET LABELS")
        print("="*50)
        show_sample_labels()

# Memory cleanup
print("\nğŸ§¹ Cleaning up memory...")
torch.cuda.empty_cache()
gc.collect()
print("âœ… Memory cleanup completed!")

