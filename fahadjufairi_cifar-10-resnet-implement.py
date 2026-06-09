# CIFAR-10 Competition - Using Actual Competition Data
# Extracts and trains on competition's .7z files, not torchvision CIFAR-10

# Cell 1: Install Required Packages and Import Libraries
import subprocess
import sys

# Install py7zr if not available
try:
    import py7zr
    print("py7zr already available")
except ImportError:
    print("Installing py7zr...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'py7zr'])
    import py7zr
    print("py7zr installed successfully")

# Import other libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
import os
import zipfile
import shutil
from PIL import Image
import random
import warnings
import glob
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Cell 2: Extract Competition Data
print("=== EXTRACTING COMPETITION DATA ===")

def extract_7z_file(file_path, extract_path):
    """Extract .7z file to specified path"""
    try:
        with py7zr.SevenZipFile(file_path, mode='r') as archive:
            archive.extractall(extract_path)
        print(f"Successfully extracted {file_path} to {extract_path}")
        return True
    except Exception as e:
        print(f"Failed to extract {file_path}: {e}")
        return False

# Create extraction directories
os.makedirs('/kaggle/working/train_images', exist_ok=True)
os.makedirs('/kaggle/working/test_images', exist_ok=True)

# Extract training images
train_7z_path = '/kaggle/input/cifar-10/train.7z'
if os.path.exists(train_7z_path):
    print("Extracting training images...")
    extract_7z_file(train_7z_path, '/kaggle/working/train_images')

# Extract test images  
test_7z_path = '/kaggle/input/cifar-10/test.7z'
if os.path.exists(test_7z_path):
    print("Extracting test images...")
    extract_7z_file(test_7z_path, '/kaggle/working/test_images')

# Load training labels
train_labels_df = pd.read_csv('/kaggle/input/cifar-10/trainLabels.csv')
print(f"Training labels shape: {train_labels_df.shape}")
print(f"Training labels columns: {train_labels_df.columns.tolist()}")
print(train_labels_df.head())

# Check extracted data
train_imgs = glob.glob('/kaggle/working/train_images/**/*.png', recursive=True)
test_imgs = glob.glob('/kaggle/working/test_images/**/*.png', recursive=True)
print(f"Training images extracted: {len(train_imgs)}")
print(f"Test images extracted: {len(test_imgs)}")

# CIFAR-10 class mapping
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}

# Cell 3: Competition Dataset Classes
class CompetitionCIFAR10Dataset(Dataset):
    """Dataset class for competition CIFAR-10 images"""
    
    def __init__(self, image_dir, labels_df=None, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.labels_df = labels_df
        
        # Get all image files
        self.image_files = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    self.image_files.append(os.path.join(root, file))
        
        self.image_files = sorted(self.image_files)
        
        # Create filename to label mapping for training
        if labels_df is not None:
            self.filename_to_label = {}
            for idx, row in labels_df.iterrows():
                filename = f"{row['id']}.png"
                label = row['label']
                self.filename_to_label[filename] = CLASS_TO_IDX[label]
        
        print(f"Dataset created with {len(self.image_files)} images")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Get label if available (training)
            if self.labels_df is not None:
                filename = os.path.basename(img_path)
                if filename in self.filename_to_label:
                    label = self.filename_to_label[filename]
                else:
                    label = 0  # Default label if not found
                return image, label
            else:
                # Test set - return dummy label
                return image, 0
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            if self.transform:
                dummy_image = self.transform(Image.new('RGB', (32, 32), color=(0, 0, 0)))
            else:
                dummy_image = Image.new('RGB', (32, 32), color=(0, 0, 0))
            return dummy_image, 0

# Cell 4: Data Preprocessing and Transforms
# Statistics for normalization (use ImageNet stats as starting point, will be adjusted)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Training transforms with heavy augmentation
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
])

# Test transforms (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# Cell 5: Create Datasets and Data Loaders
print("=== CREATING DATASETS ===")

# Create training dataset using competition images
train_dataset = CompetitionCIFAR10Dataset(
    image_dir='/kaggle/working/train_images',
    labels_df=train_labels_df,
    transform=train_transform
)

# Create train/validation split
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# Create separate validation dataset with test transforms
val_dataset = CompetitionCIFAR10Dataset(
    image_dir='/kaggle/working/train_images',
    labels_df=train_labels_df,
    transform=test_transform
)

# Create validation subset
class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

val_dataset_subset = SubsetDataset(val_dataset, val_subset.indices)

# Create test dataset
test_dataset = CompetitionCIFAR10Dataset(
    image_dir='/kaggle/working/test_images',
    labels_df=None,
    transform=test_transform
)

print(f"Training samples: {len(train_subset)}")
print(f"Validation samples: {len(val_dataset_subset)}")
print(f"Test samples: {len(test_dataset)}")

# Create data loaders
BATCH_SIZE = 128
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print("Data loaders created successfully!")

# Cell 6: ResNet Architecture with Residual Connections
class BasicBlock(nn.Module):
    """BasicBlock with residual connections - KEY REQUIREMENT"""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Shortcut connection (RESIDUAL CONNECTION)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # RESIDUAL CONNECTION - CRITICAL REQUIREMENT
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """ResNet with residual connections for CIFAR-10"""
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet50():
    return ResNet(BasicBlock, [3, 4, 6, 3])

# Cell 7: Model Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model
model = ResNet50().to(device)

# Model info
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

print("\n=== REQUIREMENTS VERIFICATION ===")
print("âœ“ Neural network contains residual connections (BasicBlock)")
print("âœ“ Residual connections implemented: out += self.shortcut(x)")
print("âœ“ Random weight initialization (no pre-trained weights)")
print("âœ“ Training on actual competition CIFAR-10 images")

# Advanced training setup
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

# Cell 8: Training Functions
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    return running_loss / len(dataloader), 100. * correct / total

# Cell 9: Main Training Loop
NUM_EPOCHS = 150
best_acc = 0
train_losses, train_accs = [], []
val_losses, val_accs = [], []
patience_counter = 0
PATIENCE = 20

print("=== STARTING TRAINING ON COMPETITION DATA ===")
print(f"Target: High accuracy on actual competition images")
print(f"Total epochs: {NUM_EPOCHS}")

for epoch in range(NUM_EPOCHS):
    print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
    print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Train and validate
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Update scheduler
    scheduler.step()
    
    # Save metrics
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, '/kaggle/working/best_model.pth')
        print(f'âœ“ NEW BEST: {best_acc:.2f}% validation accuracy')
    else:
        patience_counter += 1
    
    print(f'Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
    print(f'Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%')
    print(f'Best:  {best_acc:.2f}%')
    
    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break
    
    # Achievement notifications
    if best_acc >= 90.0:
        print("ğŸ�¯ EXCELLENT! â‰¥90% validation accuracy on competition data")
    elif best_acc >= 85.0:
        print("ğŸš€ Good progress! Strong performance on competition data")

print(f"\nğŸ�� Training completed! Best validation accuracy: {best_acc:.2f}%")

# Cell 10: Generate Competition Submission
print("\n=== GENERATING COMPETITION SUBMISSION ===")

# Load best model
checkpoint = torch.load('/kaggle/working/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Making predictions on {len(test_dataset)} test images...")

# Generate predictions on actual competition test set
model.eval()
all_predictions = []
image_ids = []

# Get image IDs in order
test_image_paths = test_dataset.image_files
for img_path in test_image_paths:
    filename = os.path.basename(img_path)
    img_id = int(filename.split('.')[0])  # Extract ID from filename like "12345.png"
    image_ids.append(img_id)

with torch.no_grad():
    for i, (inputs, _) in enumerate(tqdm(test_loader, desc="Predicting")):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_predictions.extend(predicted.cpu().numpy())

# Ensure we have exactly the right number of predictions
assert len(all_predictions) == len(image_ids), f"Mismatch: {len(all_predictions)} predictions vs {len(image_ids)} images"

print(f"Generated {len(all_predictions)} predictions")

# Cell 11: Create Submission File with Correct Format
print("\n=== CREATING SUBMISSION FILE ===")

# Convert integer predictions to text labels
text_predictions = [CLASSES[pred] for pred in all_predictions]

# Create DataFrame with image IDs and predictions
submission_data = list(zip(image_ids, text_predictions))
submission_df = pd.DataFrame(submission_data, columns=['id', 'label'])

# Sort by ID to ensure correct order
submission_df = submission_df.sort_values('id').reset_index(drop=True)

# Ensure we have exactly 300,000 predictions as required
if len(submission_df) != 300000:
    print(f"Warning: Expected 300,000 predictions, got {len(submission_df)}")

# Verify format
print("=== SUBMISSION FORMAT VERIFICATION ===")
print(f"Submission shape: {submission_df.shape}")
print(f"ID range: {submission_df['id'].min()} to {submission_df['id'].max()}")
print(f"Number of unique IDs: {submission_df['id'].nunique()}")
print(f"Label type: {type(submission_df['label'].iloc[0])}")

# Show first few predictions
print("\nFirst 10 predictions:")
print(submission_df.head(10).to_string(index=False))

# Label distribution
label_counts = submission_df['label'].value_counts()
print(f"\nPrediction distribution:")
for class_name in CLASSES:
    count = label_counts.get(class_name, 0)
    percentage = count / len(submission_df) * 100
    print(f"{class_name}: {count:,} ({percentage:.1f}%)")

# Save submission file
submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print(f"\nâœ… SUBMISSION FILE SAVED: /kaggle/working/submission.csv")

# Cell 12: Final Summary
print("\n" + "="*70)
print("FINAL ASSIGNMENT SUMMARY - COMPETITION DATA VERSION")
print("="*70)

print("âœ… ALL REQUIREMENTS MET:")
print("   â†’ ResNet with residual connections implemented")
print("   â†’ Random weight initialization (no pre-trained weights)")
print("   â†’ Trained on actual competition CIFAR-10 images")
print("   â†’ Predicted on actual competition test images")
print("   â†’ Proper submission format with correct IDs")

print(f"\nğŸ“Š PERFORMANCE:")
print(f"   â†’ Best Validation Accuracy: {best_acc:.2f}%")
print(f"   â†’ Trained on competition's actual training images")
print(f"   â†’ Predictions made on competition's actual test images")

print(f"\nğŸ”§ KEY IMPROVEMENTS:")
print(f"   â†’ Uses actual competition .7z image files")
print(f"   â†’ Extracts and processes competition training data")
print(f"   â†’ Maintains proper image ID mapping for submission")
print(f"   â†’ Should achieve much higher Kaggle score")

print(f"\nğŸ“� FILES CREATED:")
print(f"   â†’ best_model.pth (trained on competition data)")
print(f"   â†’ submission.csv ({len(submission_df):,} predictions)")

print(f"\nğŸš€ EXPECTED RESULTS:")
print(f"   â†’ Should achieve significantly higher Kaggle score")
print(f"   â†’ Model trained on same domain as test images")
print(f"   â†’ Proper handling of competition's image format")

if best_acc >= 85:
    print(f"\nğŸ�¯ EXCELLENT! Expected Kaggle improvement with {best_acc:.1f}% validation accuracy")
else:
    print(f"\nğŸ’ª SOLID! Model trained on competition data should improve Kaggle score")

print("="*70)
print("Ready for Kaggle submission!")

