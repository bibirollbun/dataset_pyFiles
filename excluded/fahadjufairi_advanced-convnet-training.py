# CIFAR-10 Competition - Custom ResNet with Advanced Optimizer & Augmentation
# Requirements: AdamW optimizer, Cosine Annealing with Warmup, CutMix augmentation

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
import math
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

# Cell 3: CutMix Augmentation Implementation
def cutmix_data(inputs, targets, alpha=1.0):
    """Apply CutMix augmentation"""
    batch_size = inputs.size(0)
    
    # Generate random indices for mixing
    indices = torch.randperm(batch_size).to(inputs.device)
    shuffled_inputs = inputs[indices]
    shuffled_targets = targets[indices]
    
    # Generate lambda from beta distribution
    lam = np.random.beta(alpha, alpha)
    
    # Generate random box coordinates
    W = inputs.size(2)
    H = inputs.size(3)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform sampling of box center
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply CutMix
    inputs[:, :, bbx1:bbx2, bby1:bby2] = shuffled_inputs[:, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda based on actual box size
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return inputs, targets, shuffled_targets, lam

# Cell 4: Competition Dataset Classes with CutOut
class Cutout:
    """Cutout data augmentation"""
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

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

# Cell 5: Enhanced Data Preprocessing with Advanced Augmentation
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Training transforms with ADVANCED augmentation including CutOut
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    Cutout(n_holes=1, length=16),  # CutOut augmentation
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
])

# Test transforms (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# Cell 6: Create Datasets and Data Loaders
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

# Cell 7: Custom ResNet Architecture
class CustomResidualBlock(nn.Module):
    """Custom residual block with residual connections"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(CustomResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # Residual connection
        out = F.relu(out)
        
        return out

class CustomResNet(nn.Module):
    """Custom ResNet architecture built from basic layers"""
    def __init__(self, block_class, layers, num_classes=10):
        super(CustomResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block_class, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block_class, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block_class, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block_class, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
        
    def _make_layer(self, block_class, out_channels, num_blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(block_class(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(block_class(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Random weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def create_custom_resnet():
    """Create custom ResNet-50 architecture"""
    return CustomResNet(CustomResidualBlock, [3, 4, 6, 3])

# Cell 8: Model Initialization with AdamW Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model
model = create_custom_resnet().to(device)

def count_residual_blocks(model):
    count = 0
    for module in model.modules():
        if isinstance(module, CustomResidualBlock):
            count += 1
    return count

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
residual_blocks = count_residual_blocks(model)

print(f"Total trainable parameters: {total_params:,}")

print("\n=== NEW REQUIREMENTS VERIFICATION ===")
print("âœ“ Random weight initialization (no pre-trained weights)")
print("âœ“ Advanced optimizer: AdamW with weight decay")
print("âœ“ Cosine annealing learning rate with warmup")
print("âœ“ Advanced augmentation: CutOut + CutMix")
print(f"âœ“ Custom ResNet with {residual_blocks} residual blocks")

# Cell 9: Cosine Annealing with Warmup Scheduler
class CosineAnnealingWarmupRestarts:
    """Cosine annealing with linear warmup"""
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# Setup optimizer and scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ADVANCED OPTIMIZER: AdamW with weight decay
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05, betas=(0.9, 0.999))

# COSINE ANNEALING WITH WARMUP
NUM_EPOCHS = 150
WARMUP_EPOCHS = 10
scheduler = CosineAnnealingWarmupRestarts(optimizer, warmup_epochs=WARMUP_EPOCHS, max_epochs=NUM_EPOCHS, eta_min=1e-6)

print("Training setup completed!")
print(f"Optimizer: AdamW (lr=0.001, weight_decay=0.05)")
print(f"Scheduler: Cosine Annealing with {WARMUP_EPOCHS}-epoch warmup")
print(f"Augmentation: CutOut + CutMix")

# Cell 10: Training Functions with CutMix
def train_epoch_with_cutmix(model, dataloader, criterion, optimizer, scheduler, epoch, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply CutMix using the cutmix_data function
        r = np.random.rand()
        if r < 0.5:  # 50% chance to apply CutMix
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, alpha=1.0)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets_a) * lam + criterion(outputs, targets_b) * (1 - lam)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        loss.backward()
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
            inputs, targets = inputs.to(device), targets.to(device)
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

# Cell 11: Main Training Loop
best_acc = 0
train_losses, train_accs = [], []
val_losses, val_accs = [], []
patience_counter = 0
PATIENCE = 25

print("\n" + "="*70)
print("STARTING TRAINING WITH ADVANCED TECHNIQUES")
print("="*70)
print(f"Optimizer: AdamW")
print(f"LR Schedule: Cosine Annealing with Warmup")
print(f"Augmentation: CutOut + CutMix")
print(f"Total epochs: {NUM_EPOCHS}")

for epoch in range(NUM_EPOCHS):
    # Update learning rate with warmup
    current_lr = scheduler.step(epoch)
    
    print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
    print(f'Learning Rate: {current_lr:.6f}')
    
    # Train and validate
    train_loss, train_acc = train_epoch_with_cutmix(model, train_loader, criterion, optimizer, scheduler, epoch, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
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
            'best_acc': best_acc,
        }, '/kaggle/working/best_model_advanced.pth')
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
    
    # Target notification
    if best_acc >= 94.2:
        print("TARGET ACHIEVED! >=94.2% validation accuracy")

print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")

# Cell 12: Generate Competition Submission
print("\n=== GENERATING COMPETITION SUBMISSION ===")

# Load best model
checkpoint = torch.load('/kaggle/working/best_model_advanced.pth')
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Making predictions on {len(test_dataset)} test images...")

# Generate predictions
model.eval()
all_predictions = []
image_ids = []

test_image_paths = test_dataset.image_files
for img_path in test_image_paths:
    filename = os.path.basename(img_path)
    img_id = int(filename.split('.')[0])
    image_ids.append(img_id)

with torch.no_grad():
    for i, (inputs, _) in enumerate(tqdm(test_loader, desc="Predicting")):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_predictions.extend(predicted.cpu().numpy())

assert len(all_predictions) == len(image_ids), f"Mismatch: {len(all_predictions)} predictions vs {len(image_ids)} images"

print(f"Generated {len(all_predictions)} predictions")

# Create submission file
text_predictions = [CLASSES[pred] for pred in all_predictions]
submission_data = list(zip(image_ids, text_predictions))
submission_df = pd.DataFrame(submission_data, columns=['id', 'label'])
submission_df = submission_df.sort_values('id').reset_index(drop=True)

print(f"\nSubmission shape: {submission_df.shape}")
print(f"ID range: {submission_df['id'].min()} to {submission_df['id'].max()}")
print("\nFirst 10 predictions:")
print(submission_df.head(10))

# Save submission
submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print(f"\nâœ… SUBMISSION FILE SAVED: /kaggle/working/submission.csv")

# Cell 13: Final Summary
print("\n" + "="*70)
print("FINAL REQUIREMENTS SUMMARY")
print("="*70)

print("\nâœ… ALL NEW REQUIREMENTS MET:")
print("   â†’ Random weight initialization: âœ“")
print("   â†’ Advanced optimizer (AdamW): âœ“")
print("   â†’ Cosine annealing with warmup: âœ“")
print("   â†’ Advanced augmentation (CutOut + CutMix): âœ“")
print("   â†’ Custom neural network design: âœ“")

print(f"\nğŸ“Š PERFORMANCE:")
print(f"   â†’ Best Validation Accuracy: {best_acc:.2f}%")
print(f"   â†’ Target Score (>=94.2%): {'âœ“ ACHIEVED' if best_acc >= 94.2 else 'âœ— NOT YET'}")

print(f"\nğŸ”§ TECHNICAL DETAILS:")
print(f"   â†’ Optimizer: AdamW (lr=0.001, weight_decay=0.05)")
print(f"   â†’ LR Schedule: Cosine annealing with {WARMUP_EPOCHS}-epoch warmup")
print(f"   â†’ Data Augmentation: CutOut + CutMix")
print(f"   â†’ Architecture: Custom ResNet with {residual_blocks} residual blocks")
print(f"   â†’ Total parameters: {total_params:,}")

print(f"\nğŸ“� OUTPUT FILES:")
print(f"   â†’ best_model_advanced.pth")
print(f"   â†’ submission.csv (300,000 predictions)")

print("="*70)
print("Ready for Kaggle submission with advanced techniques!")
print("="*70)

