import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torch.cuda.amp import GradScaler, autocast

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import datasets

import timm

import os
import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image


def nt_xent_loss(z_i, z_j, temperature=0.1):
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


# Projection Head
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=192, hidden_dim=2048, out_dim=128):  # ViT-tiny has 192 dim
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        return self.net(x)


class SimCLRTransform:
    def __init__(self, size=224, s=0.8):
        self.size = size
        self.s = s
        
        # Calculate odd kernel size for GaussianBlur
        kernel_size = int(0.1 * size)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Make it odd
        
        # Data augmentation pipeline for SimCLR
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.8 * s,
                    contrast=0.8 * s,
                    saturation=0.8 * s,
                    hue=0.2 * s
                )
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        # Return two augmented versions of the same image
        return self.transform(x), self.transform(x)


class VGGFace2Dataset(Dataset):
    def __init__(self, root_dir, transform=None, max_images=40000):
        self.root_dir = root_dir
        self.transform = transform
        self.max_images = max_images
        self.image_paths = []
        
        # Find all image files in the identity folders
        self._find_images()
        
        print(f"Loaded {len(self.image_paths)} images from VGGFace2 dataset")
    
    def _find_images(self):
        """Find all image files in the identity folder structure, limited to max_images"""
        import random
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        if os.path.isdir(self.root_dir):
            # First, collect all identity folders
            identity_folders = [f for f in os.listdir(self.root_dir) 
                              if os.path.isdir(os.path.join(self.root_dir, f))]
            
            print(f"Found {len(identity_folders)} identity folders")
            
            # Calculate images per identity (roughly equal distribution)
            images_per_identity = self.max_images // len(identity_folders)
            remaining_images = self.max_images % len(identity_folders)
            
            print(f"Target: ~{images_per_identity} images per identity")
            
            # Shuffle identity folders for random sampling
            random.shuffle(identity_folders)
            
            for i, identity_folder in enumerate(identity_folders):
                identity_path = os.path.join(self.root_dir, identity_folder)
                
                # Some identities get one extra image to reach exactly max_images
                target_for_this_identity = images_per_identity + (1 if i < remaining_images else 0)
                
                # Get all images from this identity
                identity_images = []
                for image_file in os.listdir(identity_path):
                    if image_file.lower().endswith(valid_extensions):
                        identity_images.append(os.path.join(identity_path, image_file))
                
                # Randomly sample from this identity's images
                if len(identity_images) > target_for_this_identity:
                    sampled_images = random.sample(identity_images, target_for_this_identity)
                else:
                    sampled_images = identity_images  # Use all if less than target
                
                self.image_paths.extend(sampled_images)
                
                if len(self.image_paths) >= self.max_images:
                    break
        else:
            print(f"Warning: {self.root_dir} is not a valid directory")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        # For self-supervised learning, we don't need actual labels
        # Just return a dummy label (not used in SimCLR training)
        label = 0
        
        if self.transform:
            return self.transform(image), label
        
        return image, label

class AffectNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, total_images=12000, num_classes=8, excluded_paths=None):
        self.root_dir = root_dir
        self.transform = transform
        self.total_images = total_images
        self.num_classes = num_classes
        self.excluded_paths = excluded_paths or set()  # ADD THIS LINE
        self.image_paths = []
        self.labels = []
        
        # Find all image files in the class folders and balance them
        self._find_and_balance_images()
        
        print(f"Loaded {len(self.image_paths)} images from AffectNet dataset (balanced)")
    
    # 2. Modify _find_and_balance_images to exclude paths
    def _find_and_balance_images(self):
        """Find all image files in class folders and create balanced dataset"""
        import random
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        if not os.path.isdir(self.root_dir):
            print(f"Warning: {self.root_dir} is not a valid directory")
            return
        
        # Get all class folders (assuming they are named 0, 1, 2, ..., 7 or similar)
        class_folders = []
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                try:
                    # Try to convert folder name to int (class label)
                    class_label = int(item)
                    if 0 <= class_label < self.num_classes:
                        class_folders.append((item, class_label))
                except ValueError:
                    # If folder name is not a number, skip it
                    continue
        
        # Sort by class label
        class_folders.sort(key=lambda x: x[1])
        
        print(f"Found {len(class_folders)} class folders")
        
        if len(class_folders) == 0:
            print("No valid class folders found!")
            return
        
        # Calculate images per class (equal distribution)
        images_per_class = self.total_images // len(class_folders)
        remaining_images = self.total_images % len(class_folders)
        
        print(f"Target: {images_per_class} images per class")
        
        # Collect images from each class
        for i, (class_folder, class_label) in enumerate(class_folders):
            class_path = os.path.join(self.root_dir, class_folder)
            
            # Some classes get one extra image to reach exactly total_images
            target_for_this_class = images_per_class + (1 if i < remaining_images else 0)
            
            # Get all images from this class, excluding those in excluded_paths
            class_images = []
            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(valid_extensions):
                    image_path = os.path.join(class_path, image_file)
                    # ADD THIS CHECK TO EXCLUDE VALIDATION IMAGES
                    if image_path not in self.excluded_paths:
                        class_images.append(image_path)
            
            print(f"Class {class_label}: found {len(class_images)} images (after exclusions), sampling {target_for_this_class}")
            
            # Randomly sample from this class's images
            if len(class_images) > target_for_this_class:
                sampled_images = random.sample(class_images, target_for_this_class)
            else:
                sampled_images = class_images  # Use all if less than target
                print(f"Warning: Class {class_label} has only {len(class_images)} images, less than target {target_for_this_class}")
            
            # Add to dataset
            self.image_paths.extend(sampled_images)
            self.labels.extend([class_label] * len(sampled_images))
            
            if len(self.image_paths) >= self.total_images:
                break
        
        # Shuffle the final dataset
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)
        self.image_paths = list(self.image_paths)
        self.labels = list(self.labels)
        
        print(f"Final dataset: {len(self.image_paths)} images, {len(set(self.labels))} classes")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def create_dataset_from_paths(self, image_paths, transform=None):
        """Create dataset from specific image paths"""
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        for path in image_paths:
            # Extract class label from folder name
            class_label = int(os.path.basename(os.path.dirname(path)))
            self.image_paths.append(path)
            self.labels.append(class_label)

class TransformDataset:
    """Wrapper to apply transforms to a subset dataset"""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.subset)


def create_custom_validation_split(train_data_dir, val_images_per_class=275, num_classes=8, random_seed=42):
    """
    Create a custom validation set by sampling equally from each class in train folders
    
    Args:
        train_data_dir: Path to training data directory
        val_images_per_class: Number of validation images per class (275 * 8 = 2200 total)
        num_classes: Number of classes
        random_seed: Random seed for reproducible splits
    
    Returns:
        set of validation image paths
    """
    import random
    random.seed(random_seed)
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    validation_paths = set()
    
    print(f"Creating custom validation split: {val_images_per_class} images per class = {val_images_per_class * num_classes} total")
    
    # Process each class folder
    for class_label in range(num_classes):
        class_folder = str(class_label)
        class_path = os.path.join(train_data_dir, class_folder)
        
        if not os.path.exists(class_path):
            print(f"Warning: Class folder {class_path} does not exist")
            continue
        
        # Get all images from this class
        class_images = []
        for image_file in os.listdir(class_path):
            if image_file.lower().endswith(valid_extensions):
                image_path = os.path.join(class_path, image_file)
                class_images.append(image_path)
        
        print(f"Class {class_label}: found {len(class_images)} images, sampling {val_images_per_class} for validation")
        
        # Randomly sample validation images from this class
        if len(class_images) >= val_images_per_class:
            val_images_for_class = random.sample(class_images, val_images_per_class)
            validation_paths.update(val_images_for_class)
        else:
            print(f"Warning: Class {class_label} has only {len(class_images)} images, less than required {val_images_per_class}")
            validation_paths.update(class_images)
    
    print(f"Selected {len(validation_paths)} images for validation set")
    return validation_paths


def load_pretrained_and_finetune(checkpoint_path, start_epoch=0, total_epochs=50):
    """Load pretrained ViT model from checkpoint and finetune directly"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    
    # Configuration
    SAVE_DIR = '/kaggle/working/FACIAL_EXPRESSION/checkpoints/'
    METRICS_DIR = '/kaggle/working/FACIAL_EXPRESSION/metrics/'
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading pretrained model from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get feature dimension from checkpoint
    feature_dim = checkpoint.get('feature_dim', 192)  # Default for ViT-tiny
    print(f"Feature dimension from checkpoint: {feature_dim}")
    
    # Create ViT model without pretrained weights
    encoder = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0)
    
    # Load encoder weights from checkpoint
    encoder_state_dict = checkpoint['encoder_state_dict']
    
    # Handle DataParallel state dict if needed
    if any(key.startswith('module.') for key in encoder_state_dict.keys()):
        # Remove 'module.' prefix from keys
        encoder_state_dict = {key.replace('module.', ''): value for key, value in encoder_state_dict.items()}
    
    encoder.load_state_dict(encoder_state_dict)
    print("âœ… Encoder weights loaded from checkpoint")
    
    # Create projector
    projector = ProjectionHead(in_dim=feature_dim, hidden_dim=2048, out_dim=128)
    
    # Load projector weights if available
    if 'projector_state_dict' in checkpoint:
        projector_state_dict = checkpoint['projector_state_dict']
        # Handle DataParallel state dict if needed
        if any(key.startswith('module.') for key in projector_state_dict.keys()):
            projector_state_dict = {key.replace('module.', ''): value for key, value in projector_state_dict.items()}
        projector.load_state_dict(projector_state_dict)
        print("âœ… Projector weights loaded from checkpoint")
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        encoder = DataParallel(encoder)
        projector = DataParallel(projector)
    
    encoder = encoder.to(device)
    projector = projector.to(device)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(list(encoder.parameters()) + list(projector.parameters()), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    scaler = GradScaler()
    
    # Load optimizer and scaler states if available and continuing training
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("âœ… Optimizer state loaded from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
    
    if 'scaler_state_dict' in checkpoint:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("âœ… Scaler state loaded from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load scaler state: {e}")
    
    print("âœ… Pretrained model loaded successfully from checkpoint!")
    
    return encoder, projector, optimizer, scheduler, scaler, feature_dim



def finetune_simclr(encoder, projector, optimizer, scheduler, scaler, feature_dim, start_epoch=0, total_epochs=50):
    """Single-stage fine-tuning with custom validation split"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration
    SAVE_DIR = '/kaggle/working/FACIAL_EXPRESSION/checkpoints/'
    METRICS_DIR = '/kaggle/working/FACIAL_EXPRESSION/metrics/'
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Data directories
    train_data_dir = '/kaggle/input/affectnet/AffectNetCustom/train'
    test_data_dir = '/kaggle/input/affectnet/AffectNetCustom/test'
    
    print(f"Loading AffectNet dataset...")
    print(f"Train data dir: {train_data_dir}")
    print(f"Test data dir: {test_data_dir}")

    # CHANGE: Create custom validation split from train folders
    validation_paths = create_custom_validation_split(
        train_data_dir=train_data_dir,
        val_images_per_class=275,  # 275 * 8 = 2200 total validation images
        num_classes=8,
        random_seed=42
    )
    
    # CHANGE: Create datasets with exclusion for train set
    train_dataset = AffectNetDataset(
        root_dir=train_data_dir, 
        transform=train_transform, 
        total_images=12000, 
        num_classes=8,
        excluded_paths=validation_paths  # Exclude validation images from training
    )
    
    # CHANGE: Create validation dataset using AffectNetDataset
    val_dataset = AffectNetDataset(
        root_dir=train_data_dir,  # Dummy parameter, will be overridden
        transform=val_transform,
        total_images=0,  # Not used
        num_classes=8
    )
    # Override with specific validation paths
    val_dataset.image_paths = list(validation_paths)
    val_dataset.labels = [int(os.path.basename(os.path.dirname(path))) for path in validation_paths]
    
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=val_transform)
    
    print(f"Train set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images") 
    print(f"Test set: {len(test_dataset)} images")
    
    # DataLoaders
    FT_BATCH_SIZE = 128
    train_dataloader = DataLoader(train_dataset, batch_size=FT_BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=FT_BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=FT_BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=False)
    
    # Get number of classes
    num_classes = len(test_dataset.classes)
    print(f"Fine-tuning for {num_classes} classes")
    
    # Create encoder with classification head
    print("Creating encoder with classification head...")
    finetuned_encoder = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
    
    # Load pretrained weights
    if isinstance(encoder, DataParallel):
        pretrained_state = encoder.module.state_dict()
    else:
        pretrained_state = encoder.state_dict()
    
    # Transfer weights (excluding classification head)
    finetuned_state = finetuned_encoder.state_dict()
    pretrained_backbone = {k: v for k, v in pretrained_state.items() 
                          if not k.startswith('head.') and k in finetuned_state}
    
    print(f"Loading {len(pretrained_backbone)} pretrained parameters")
    finetuned_state.update(pretrained_backbone)
    finetuned_encoder.load_state_dict(finetuned_state)
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for fine-tuning")
        finetuned_encoder = DataParallel(finetuned_encoder)
    
    finetuned_encoder = finetuned_encoder.to(device)
    
    # Enable gradients for all parameters (full fine-tuning)
    print("=== Single-Stage Full Fine-tuning Approach ===")
    print("Training entire model for 50 epochs")
    for param in finetuned_encoder.parameters():
        param.requires_grad = True
    
    # Single optimizer with different learning rates for different parts
    if isinstance(finetuned_encoder, DataParallel):
        backbone_params = [p for n, p in finetuned_encoder.module.named_parameters() if not n.startswith('head.')]
        head_params = [p for n, p in finetuned_encoder.module.named_parameters() if n.startswith('head.')]
    else:
        backbone_params = [p for n, p in finetuned_encoder.named_parameters() if not n.startswith('head.')]
        head_params = [p for n, p in finetuned_encoder.named_parameters() if n.startswith('head.')]
    
    ft_optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 0.0002},  # Lower LR for pretrained backbone
        {'params': head_params, 'lr': 0.001}        # Higher LR for classifier
    ], weight_decay=1e-4)
    
    # Cosine annealing scheduler for full 50 epochs
    ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        ft_optimizer, T_max=total_epochs, eta_min=1e-6
    )
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Training metrics
    finetune_losses = []
    finetune_val_losses = []
    finetune_train_accuracies = []
    finetune_val_accuracies = []
    best_val_acc = 0.0
    
    print("=== Starting Single-Stage Fine-tuning Phase on AffectNet ===")
    
    for epoch in range(start_epoch, start_epoch + total_epochs):
        # Training phase
        finetuned_encoder.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        epoch_progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{start_epoch + total_epochs}", unit="batch")
        
        for images, labels in epoch_progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            ft_optimizer.zero_grad()
            
            with autocast():
                outputs = finetuned_encoder(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(ft_optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            epoch_progress.set_postfix(loss=loss.item(), acc=100.*train_correct/train_total)
        
        # Update learning rate
        ft_scheduler.step()
        
        # Validation phase
        finetuned_encoder.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_dataloader, desc="Validation", leave=False):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with autocast():
                    outputs = finetuned_encoder(images)
                    # Use regular CrossEntropyLoss for validation (no label smoothing)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        train_avg_loss = train_loss / len(train_dataloader)
        train_accuracy = 100. * train_correct / train_total
        val_avg_loss = val_loss / len(val_dataloader)
        val_accuracy = 100. * val_correct / val_total
        
        finetune_losses.append(train_avg_loss)
        finetune_val_losses.append(val_avg_loss)
        finetune_train_accuracies.append(train_accuracy)
        finetune_val_accuracies.append(val_accuracy)
        
        # Get current learning rate
        current_lr = ft_optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{start_epoch + total_epochs}]")
        print(f"Train - Loss: {train_avg_loss:.4f} Acc: {train_accuracy:.2f}% LR: {current_lr:.6f}")
        print(f"Val   - Loss: {val_avg_loss:.4f} Acc: {val_accuracy:.2f}%")
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            encoder_state = finetuned_encoder.module.state_dict() if isinstance(finetuned_encoder, DataParallel) else finetuned_encoder.state_dict()
            
            best_checkpoint = os.path.join(SAVE_DIR, "best_finetuned_vit_tiny_affectnet_single_stage.pth")
            torch.save({
                'encoder_state_dict': encoder_state,
                'optimizer_state_dict': ft_optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'best_val_accuracy': best_val_acc,
                'num_classes': num_classes
            }, best_checkpoint)
            print(f"New best model saved! Val Accuracy: {best_val_acc:.2f}%")
        
        # Save metrics
        finetune_metrics = {
            'train_losses': finetune_losses,
            'val_losses': finetune_val_losses,
            'train_accuracies': finetune_train_accuracies,
            'val_accuracies': finetune_val_accuracies,
            'best_val_accuracy': best_val_acc
        }
        
        with open(os.path.join(METRICS_DIR, 'finetune_vit_affectnet_metrics_single_stage.json'), 'w') as f:
            json.dump(finetune_metrics, f)
    
    print(f"Fine-tuning completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Final test evaluation
    print("\n=== Final Test Evaluation ===")
    finetuned_encoder.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Testing"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast():
                outputs = finetuned_encoder(images)
            
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_accuracy = 100. * test_correct / test_total
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    
    return finetuned_encoder, test_accuracy


def main():
    print("=== Pretrained ViT Fine-tuning Pipeline ===")
    print("Phase 1: Load pretrained ViT model from checkpoint")
    print("Phase 2: Fine-tuning on AffectNet with custom validation split")
    print("=" * 50)
    
    # SPECIFY YOUR CHECKPOINT PATH HERE
    CHECKPOINT_PATH = '/kaggle/input/pretrained-simclr-vitt/simclr_vit_tiny_pretrain_epoch_50 (2).pth'
    # Or use any other checkpoint path you have
        
    # Phase 1: Load pretrained model from checkpoint
    print(f"\nğŸš€ Loading pretrained ViT model from: {CHECKPOINT_PATH}")
    encoder, projector, optimizer, scheduler, scaler, feature_dim = load_pretrained_and_finetune(
        checkpoint_path=CHECKPOINT_PATH,
        start_epoch=0,
        total_epochs=50
    )
        
    # Phase 2: Fine-tuning on AffectNet
    print("\nğŸš€ Starting Fine-tuning on AffectNet...")
    finetuned_encoder, test_accuracy = finetune_simclr(
        encoder=encoder,
        projector=projector,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        feature_dim=feature_dim,
        start_epoch=0,
        total_epochs=50
    )
    
    print("âœ… Fine-tuning completed!")
    print(f"ğŸ�¯ Final Test Accuracy: {test_accuracy:.2f}%")
        
    return finetuned_encoder, test_accuracy


if __name__ == "__main__":
    try:
        final_model, final_accuracy = main()
        
        print("\n" + "="*50)
        print("ğŸ�‰ Training Pipeline Completed Successfully!")
        print(f"ğŸ�¯ Final Test Accuracy: {final_accuracy:.2f}%")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\nâš ï¸�  Training interrupted by user")
    except Exception as e:
        print(f"\nâ�Œ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

