import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm 
import os
import sys
from PIL import Image
import json
import warnings
import time
from sklearn.metrics import confusion_matrix, classification_report
warnings.filterwarnings('ignore')


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for SimCLR
    Fixed version to prevent NaN losses
    """
    batch_size = z_i.size(0)
    device = z_i.device
    
    # Normalize representations with epsilon for numerical stability
    z_i = F.normalize(z_i, dim=1, eps=1e-8)
    z_j = F.normalize(z_j, dim=1, eps=1e-8)
    
    # Concatenate representations
    representations = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, embedding_dim]
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(representations, representations.T)  # [2*batch_size, 2*batch_size]
    
    # Clamp temperature to prevent division by very small numbers
    temperature = max(temperature, 1e-8)
    
    # Apply temperature scaling
    similarity_matrix = similarity_matrix / temperature
    
    # Clamp similarities to prevent overflow in exp
    similarity_matrix = torch.clamp(similarity_matrix, min=-50, max=50)
    
    # Create mask to remove self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    
    # Create positive pairs mask
    pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=device)
    
    # Set positive pairs
    for i in range(batch_size):
        pos_mask[i, i + batch_size] = True  # z_i[i] -> z_j[i]
        pos_mask[i + batch_size, i] = True  # z_j[i] -> z_i[i]
    
    # Compute InfoNCE loss more stably
    loss = 0
    for i in range(2 * batch_size):
        # Get positive similarity (there's only one positive pair per anchor)
        pos_sim = similarity_matrix[i][pos_mask[i]]
        
        # Get negative similarities (exclude self-similarity)
        neg_mask = ~mask[i] & ~pos_mask[i]
        neg_sim = similarity_matrix[i][neg_mask]
        
        # Combine positive and negative similarities
        all_sim = torch.cat([pos_sim, neg_sim])
        
        # Use more stable logsumexp computation
        max_sim = torch.max(all_sim)
        exp_sim = torch.exp(all_sim - max_sim)
        log_sum_exp = max_sim + torch.log(torch.sum(exp_sim) + 1e-8)
        
        # InfoNCE loss: -log(exp(pos)/sum(exp(all)))
        loss += -pos_sim + log_sum_exp
    
    return loss / (2 * batch_size)


class ResNet34Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet34Encoder, self).__init__()
        try:
            # Try to load pretrained model
            if pretrained:
                self.resnet = models.resnet34(pretrained=True)
            else:
                self.resnet = models.resnet34(pretrained=False)
        except Exception as e:
            print(f"Warning: Could not load pretrained weights due to network error: {e}")
            print("Loading ResNet34 without pretrained weights...")
            self.resnet = models.resnet34(pretrained=False)
        
        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
    def forward(self, x):
        features = self.resnet(x)
        return features.squeeze()
# Projection Head
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        return self.net(x)


# SimCLR Transform
class SimCLRTransform:
    def __init__(self, size=224):
        # Calculate kernel size properly (must be odd and positive)
        kernel_size = int(0.1 * size)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Make it odd
        if kernel_size < 3:
            kernel_size = 3  # Minimum kernel size
            
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

# Standard Transform for Fine-tuning
class StandardTransform:
    def __init__(self, size=224, is_train=True):
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(size * 1.125)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, x):
        return self.transform(x)


# AffectNet Dataset for Self-Supervised Learning
class AffectNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self._load_dataset()
        print(f"Loaded {len(self.image_paths)} images from AffectNet dataset")
    
    def _load_dataset(self):
        """Load dataset assuming ImageFolder structure"""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
        
        # Get class names (subdirectories)
        class_names = sorted([d for d in os.listdir(self.root_dir) 
                             if os.path.isdir(os.path.join(self.root_dir, d))])
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(self.root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.lower().endswith(valid_extensions):
                    self.image_paths.append(os.path.join(class_dir, file))
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class RAFDBDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split  # 'train' or 'test'
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # RAF-DB emotion mapping
        self.emotion_map = {
            'surprise': 0,
            'fear': 1,
            'disgust': 2,
            'happiness': 3,
            'sadness': 4,
            'anger': 5,
            'neutral': 6
        }
        
        self._load_dataset()
        print(f"Loaded {len(self.image_paths)} images from RAF-DB dataset ({self.split} split)")
        if len(self.image_paths) > 0:
            self._print_class_distribution()
    
    def _load_dataset(self):
        """Load RAF-DB dataset with train/test split structure"""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        # Construct the path to the split directory
        dataset_dir = os.path.join(self.root_dir, 'DATASET')
        split_dir = os.path.join(dataset_dir, self.split)
        
        # Check multiple possible directory structures
        possible_paths = [
            split_dir,  # /path/to/raf-db-dataset/DATASET/train
            os.path.join(self.root_dir, self.split),  # /path/to/raf-db-dataset/train
            self.root_dir  # /path/to/raf-db-dataset (if it contains numbered folders directly)
        ]
        
        split_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                split_dir = path
                break
        
        if split_dir is None:
            raise FileNotFoundError(f"Could not find {self.split} directory in any of: {possible_paths}")
        
        print(f"Loading from: {split_dir}")
        
        # Check if it's numbered folder structure (1, 2, 3, etc.)
        subdirs = [d for d in os.listdir(split_dir) 
                  if os.path.isdir(os.path.join(split_dir, d))]
        
        # Check if we have numbered folders (1, 2, 3, 4, 5, 6, 7)
        numbered_folders = [str(i) for i in range(1, 8)]
        if all(folder in subdirs for folder in numbered_folders):
            self._load_numbered_structure(split_dir)
        elif any(emotion in subdirs for emotion in self.emotion_map.keys()):
            # ImageFolder structure within the split
            self._load_imagefolder_structure(split_dir)
        else:
            # Flat structure - try to load from annotation files or infer from filenames
            self._load_flat_structure(split_dir)
    
    def _load_numbered_structure(self, split_dir):
        """Load dataset from numbered folder structure (1, 2, 3, etc.)"""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        # Map numbered folders to emotion indices (1-indexed to 0-indexed)
        for folder_num in range(1, 8):  # 1 to 7
            folder_path = os.path.join(split_dir, str(folder_num))
            if os.path.exists(folder_path):
                emotion_idx = folder_num - 1  # Convert to 0-indexed
                for file in os.listdir(folder_path):
                    if file.lower().endswith(valid_extensions):
                        self.image_paths.append(os.path.join(folder_path, file))
                        self.labels.append(emotion_idx)
    
    def _load_imagefolder_structure(self, split_dir):
        """Load dataset from ImageFolder structure within train/test split"""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        for emotion_name, emotion_idx in self.emotion_map.items():
            emotion_dir = os.path.join(split_dir, emotion_name)
            if os.path.exists(emotion_dir):
                for file in os.listdir(emotion_dir):
                    if file.lower().endswith(valid_extensions):
                        self.image_paths.append(os.path.join(emotion_dir, file))
                        self.labels.append(emotion_idx)
    
    def _load_flat_structure(self, split_dir):
        """Load dataset from flat structure with annotation files"""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        # Look for annotation files
        annotation_files = ['labels.txt', 'annotations.txt', 'list_patition_label.txt']
        annotation_path = None
        
        for ann_file in annotation_files:
            ann_path = os.path.join(split_dir, ann_file)
            if os.path.exists(ann_path):
                annotation_path = ann_path
                break
        
        if annotation_path:
            # Load from annotation file
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_name = parts[0]
                        label = int(parts[1]) - 1  # RAF-DB labels are 1-indexed
                        img_path = os.path.join(split_dir, img_name)
                        if os.path.exists(img_path):
                            self.image_paths.append(img_path)
                            self.labels.append(label)
        else:
            # Load all images and assign dummy labels (for unlabeled data)
            print("Warning: No annotation file found. Assigning dummy labels.")
            for file in os.listdir(split_dir):
                if file.lower().endswith(valid_extensions):
                    self.image_paths.append(os.path.join(split_dir, file))
                    self.labels.append(0)  # Dummy label
    
    def _print_class_distribution(self):
        """Print the distribution of classes in the dataset"""
        if not self.labels:
            return
        
        class_counts = {}
        emotion_names = list(self.emotion_map.keys())
        
        for label in self.labels:
            if label < len(emotion_names):
                emotion_name = emotion_names[label]
                class_counts[emotion_name] = class_counts.get(emotion_name, 0) + 1
        
        print("Class distribution:")
        for emotion, count in class_counts.items():
            print(f"  {emotion}: {count} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# SimCLR Model
class SimCLR(nn.Module):
    def __init__(self, encoder_dim=512, projection_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = ResNet34Encoder(pretrained=True)
        self.projection_head = ProjectionHead(in_dim=encoder_dim, out_dim=projection_dim)
        
    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, projections

# Classifier for Fine-tuning
class FinetuneClassifier(nn.Module):
    def __init__(self, encoder, num_classes=7):
        super(FinetuneClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)


# Enhanced plotting function for pre-training
def plot_pretraining_metrics(train_losses, save_path=None):
    """Plot pre-training metrics with enhanced visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curve
    ax1.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_title('SimCLR Pre-training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('NT-Xent Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Loss smoothed (moving average)
    if len(train_losses) >= 5:
        window_size = min(5, len(train_losses))
        smoothed_loss = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size-1, len(train_losses)), smoothed_loss, 'r-', linewidth=2, label='Smoothed Loss')
        ax2.set_title('Smoothed Training Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('NT-Xent Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # Loss distribution
    ax3.hist(train_losses, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_title('Loss Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Loss Value')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # Loss change rate
    if len(train_losses) > 1:
        loss_changes = np.diff(train_losses)
        ax4.plot(loss_changes, 'g-', linewidth=2, label='Loss Change Rate')
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax4.set_title('Loss Change Rate', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Change')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Enhanced plotting function for fine-tuning
def plot_finetuning_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_path=None):
    """Plot fine-tuning metrics with enhanced visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    ax1.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.plot(val_losses, 'r-', linewidth=2, label='Validation Loss')
    ax1.set_title('Fine-tuning Loss Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('CrossEntropy Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(train_accuracies, 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(val_accuracies, 'r-', linewidth=2, label='Validation Accuracy')
    ax2.set_title('Fine-tuning Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss gap (overfitting indicator)
    loss_gap = np.array(val_losses) - np.array(train_losses)
    ax3.plot(loss_gap, 'purple', linewidth=2, label='Val Loss - Train Loss')
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.set_title('Overfitting Indicator (Loss Gap)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Difference')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Accuracy gap
    acc_gap = np.array(train_accuracies) - np.array(val_accuracies)
    ax4.plot(acc_gap, 'orange', linewidth=2, label='Train Acc - Val Acc')
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax4.set_title('Overfitting Indicator (Accuracy Gap)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Difference (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix for test results"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate and display per-class metrics
    print("\nPer-class Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def pretrain_simclr(data_dir, epochs=50, batch_size=128, learning_rate=0.0003, save_dir='/kaggle/working/'):
    """Self-supervised pre-training with SimCLR with enhanced metrics"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Dataset and DataLoader
    transform = SimCLRTransform()
    dataset = AffectNetDataset(data_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    
    try:
        model = SimCLR()
        print("Successfully loaded SimCLR model with pretrained ResNet34")
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        print("Falling back to non-pretrained ResNet34...")
        model = SimCLR()
        model.encoder = ResNet34Encoder(pretrained=False)
    
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model = model.to(device)
    
    # Loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    train_losses = []
    epoch_times = []
    learning_rates = []
    
    print("Starting SimCLR Pre-training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, ((x_i, x_j), _) in enumerate(progress_bar):
            x_i, x_j = x_i.to(device, non_blocking=True), x_j.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                _, z_i = model(x_i)
                _, z_j = model(x_j)
                loss = nt_xent_loss(z_i, z_j, temperature=0.5)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        learning_rates.append(scheduler.get_last_lr()[0])
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}, Time: {epoch_time:.2f}s')
    
    total_time = time.time() - start_time
    print(f"\nPre-training completed in {total_time:.2f} seconds")
    
    # Save training metrics
    metrics = {
        'train_losses': train_losses,
        'epoch_times': epoch_times,
        'learning_rates': learning_rates,
        'total_time': total_time
    }
    
    with open(os.path.join(save_dir, 'pretrain_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot enhanced metrics
    plot_pretraining_metrics(train_losses, save_path=os.path.join(save_dir, 'pretrain_metrics.png'))
    
    # Save the encoder
    encoder_to_save = model.module.encoder if isinstance(model, DataParallel) else model.encoder
    torch.save(encoder_to_save.state_dict(), os.path.join(save_dir, 'pretrained_encoder.pth'))
    print(f"Encoder saved to: {os.path.join(save_dir, 'pretrained_encoder.pth')}")
    
    return encoder_to_save


def finetune_classifier(encoder, train_dir, test_dir, epochs=50, batch_size=128, learning_rate=0.0001, save_dir='/kaggle/working/'):
    """Fine-tune the pre-trained encoder for classification on RAF-DB with enhanced metrics"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create train and validation splits
    train_transform = StandardTransform(is_train=True)
    val_transform = StandardTransform(is_train=False)
    test_transform = StandardTransform(is_train=False)
    
    # Full training dataset (RAF-DB)
    full_train_dataset = RAFDBDataset(train_dir, split='train', transform=train_transform)
    
    # Split into train (85%) and validation (15%)
    train_size = int(0.85 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Create separate validation dataset with proper transform
    val_dataset_proper = RAFDBDataset(train_dir, split='train', transform=val_transform)
    val_indices = val_dataset.indices
    val_dataset = torch.utils.data.Subset(val_dataset_proper, val_indices)
    
    # Test dataset (RAF-DB)
    test_dataset = RAFDBDataset(test_dir, split='test', transform=test_transform)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    
    # Model (RAF-DB has 7 classes: Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral)
    model = FinetuneClassifier(encoder, num_classes=7)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    epoch_times = []
    learning_rates = []
    
    print("Starting Fine-tuning on RAF-DB...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(data)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({
                'Loss': loss.item(),
                'Acc': 100. * train_correct / train_total
            })
        
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(100. * train_correct / train_total)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                with autocast():
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100. * val_correct / val_total)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        learning_rates.append(scheduler.get_last_lr()[0])
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%')
        print(f'  Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.2f}%')
        print(f'  Time: {epoch_time:.2f}s, LR: {scheduler.get_last_lr()[0]:.6f}')
    
    total_time = time.time() - start_time
    print(f"\nFine-tuning completed in {total_time:.2f} seconds")
    
    # Test evaluation with detailed metrics
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(data)
            
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    test_accuracy = 100. * test_correct / test_total
    print(f'\nFinal Test Accuracy on RAF-DB: {test_accuracy:.2f}%')
    
    # Save training metrics
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'epoch_times': epoch_times,
        'learning_rates': learning_rates,
        'test_accuracy': test_accuracy,
        'total_time': total_time
    }
    
    with open(os.path.join(save_dir, 'finetune_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot enhanced metrics
    plot_finetuning_metrics(train_losses, val_losses, train_accuracies, val_accuracies, 
                           save_path=os.path.join(save_dir, 'finetune_metrics.png'))
    
    # Plot confusion matrix
    class_names = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
    plot_confusion_matrix(all_targets, all_predictions, class_names, 
                         save_path=os.path.join(save_dir, 'confusion_matrix.png'))
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    print(f"Final model saved to: {os.path.join(save_dir, 'final_model.pth')}")
    
    return model


def create_resnet34_encoder_offline():
    """Alternative function to create ResNet34 encoder without downloading pretrained weights"""
    print("Creating ResNet34 encoder without pretrained weights (offline mode)")
    return ResNet34Encoder(pretrained=False)


def main():
    """Main function to run fine-tuning only (skipping pre-training)"""
    print("="*60)
    print("SIMCLR PIPELINE: FINE-TUNING ONLY (USING SIMCLR STRUCTURE WITHOUT PRE-TRAINING)")
    print("="*60)
    
    # Configuration
    config = {
        'finetune_epochs': 50,
        'batch_size': 128,
        'finetune_lr': 0.0001,
        'save_dir': '/kaggle/working/',
        'rafdb_dir': '/kaggle/input/raf-db-dataset/DATASET'
    }
    
    # Print configuration
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Check if RAF-DB path exists
    if not os.path.exists(config['rafdb_dir']):
        print(f"\nError: RAF-DB dataset not found at {config['rafdb_dir']}")
        print("Please update the path to match your RAF-DB dataset location")
        return
    
    # Create SimCLR model without pre-training (random initialization)
    print("\n" + "="*50)
    print("CREATING SIMCLR MODEL (WITHOUT PRE-TRAINING)")
    print("="*50)
    
    print("Creating SimCLR model with randomly initialized ResNet34...")
    simclr_model = SimCLR(encoder_dim=512, projection_dim=128)
    # Extract the encoder from SimCLR model
    encoder = simclr_model.encoder
    print("Successfully created SimCLR ResNet34 encoder (pretrained=False)")
    
    # Phase 2: Supervised fine-tuning
    print("\n" + "="*50)
    print("PHASE: SUPERVISED FINE-TUNING ON RAF-DB (USING SIMCLR ENCODER)")
    print("="*50)
    
    final_model = finetune_classifier(
        encoder=encoder,
        train_dir=config['rafdb_dir'],
        test_dir=config['rafdb_dir'],
        epochs=config['finetune_epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['finetune_lr'],
        save_dir=config['save_dir']
    )
    
    print("\n" + "="*50)
    print("FINE-TUNING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"All outputs saved to: {config['save_dir']}")
    print("Files generated:")
    print("  - final_model.pth: Fine-tuned model weights")
    print("  - finetune_metrics.json: Fine-tuning metrics")
    print("  - finetune_metrics.png: Fine-tuning visualization")
    print("  - confusion_matrix.png: Test set confusion matrix")
    
    return final_model

if __name__ == "__main__":
    final_model = main()

