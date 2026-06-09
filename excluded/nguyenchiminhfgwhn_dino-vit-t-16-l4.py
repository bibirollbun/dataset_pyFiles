import os
import argparse
import math
import numpy as np
import time
import sys
import cv2
import random
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF
from PIL import Image, ImageFile
from torch.cuda.amp import autocast, GradScaler
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm


# Define paths
VGGFACE2_PATH = "/kaggle/input/vggface2/train"
RAFDB_PATH = "/kaggle/input/raf-db-dataset/DATASET"
OUTPUT_DIR = "/kaggle/working/"
AFFECTNET_PATH = '/kaggle/input/affectnet/AffectNetCustom/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
NUM_EPOCHS_PRETRAINING = 50
SAVE_FREQ = 5
BATCH_SIZE = 256
NUM_WORKERS = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DINO Parameters
WARMUP_TEACHER_TEMP = 0.04
TEACHER_TEMP = 0.04
WARMUP_TEACHER_TEMP_EPOCHS = 1
USE_BN_IN_HEAD = False
NORM_LAST_LAYER = True
MOMENTUM_TEACHER = 0.996
OUT_DIM = 256
STUDENT_TEMP = 0.1
GLOBAL_CROPS_SCALE = (0.4, 1.0)
LOCAL_CROPS_SCALE = (0.05, 0.4)
LOCAL_CROPS_NUMBER = 8
WEIGHT_DECAY = 1e-6
WEIGHT_DECAY_END = 0.4
LR = 0.001
MIN_LR = 1e-6

# ViT-T/16 Parameters
PATCH_SIZE = 16
IMG_SIZE = 224
EMBED_DIM = 192
DEPTH = 12
NUM_HEADS = 3
MLP_RATIO = 4.0
DROP_RATE = 0.0
ATTN_DROP_RATE = 0.0


class DinoViTTiny(nn.Module):
    def __init__(self, output_dim=256, use_bn_in_head=False, norm_last_layer=True):
        super(DinoViTTiny, self).__init__()
        
        # Use timm's ViT-Tiny
        self.backbone = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            num_classes=0,
            global_pool='token'
        )
        
        # Get feature dimension from the backbone
        feat_dim = self.backbone.embed_dim  # Should be 192 for ViT-Tiny
        hidden_dim = 2048
        bottleneck_dim = 256
        
        # MLP head (projector) - follows DINO paper architecture
        mlp = [
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        ]
        
        self.mlp = nn.Sequential(*mlp)
        
        # Last layer (projection head)
        self.last_layer = nn.Linear(bottleneck_dim, output_dim, bias=False)
        if norm_last_layer:
            self.last_layer = nn.utils.weight_norm(self.last_layer)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m != self.last_layer:
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # CRITICAL FIX: Always resize to 224x224 for the backbone
        # This handles both global (224) and local (96) crops
        if x.size(-1) != 224 or x.size(-2) != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        x = self.backbone(x)  # CLS token features
        x = self.mlp(x)
        x = self.last_layer(x)
        return x


class MultiCropTransform:
    def __init__(self, 
                 global_crops_scale, 
                 local_crops_scale, 
                 local_crops_number, 
                 mean=(0.485, 0.456, 0.406), 
                 std=(0.229, 0.224, 0.225)):
        
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        # Global crops
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            normalize,
        ])
        
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
            transforms.RandomSolarize(170, p=0.2),
            normalize,
        ])
        
        # Local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class VGGFace2Dataset(Dataset):
    def __init__(self, root_dir, transform=None, max_total_images=40000):
        self.root_dir = root_dir
        self.transform = transform
        self.max_total_images = max_total_images
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        # Get all identity folders
        identity_folders = [f for f in os.listdir(self.root_dir) 
                          if os.path.isdir(os.path.join(self.root_dir, f))]
        
        if not identity_folders:
            print(f"Warning: No identity folders found in {self.root_dir}")
            return
        
        print(f"Found {len(identity_folders)} identity folders")
        
        # Calculate images per identity to reach target total
        images_per_identity = self.max_total_images // len(identity_folders)
        print(f"Target: {images_per_identity} images per identity (total ~{images_per_identity * len(identity_folders)})")
        
        total_loaded = 0
        
        for identity_idx, identity_folder in enumerate(identity_folders):
            identity_path = os.path.join(self.root_dir, identity_folder)
            
            # Get all image files in this identity folder
            image_files = [f for f in os.listdir(identity_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Randomly sample images_per_identity images (or all if less available)
            if len(image_files) > images_per_identity:
                selected_images = random.sample(image_files, images_per_identity)
            else:
                selected_images = image_files
            
            # Add selected images to samples
            for img_file in selected_images:
                img_path = os.path.join(identity_path, img_file)
                # Use identity_idx as pseudo-label (not used in DINO but needed for dataset structure)
                self.samples.append((img_path, identity_idx))
            
            total_loaded += len(selected_images)
            
            if (identity_idx + 1) % 100 == 0:
                print(f"  Processed {identity_idx + 1}/{len(identity_folders)} identities, "
                      f"loaded {total_loaded} images so far...")
        
        print(f"Total VGGFace2 samples loaded: {len(self.samples)} from {len(identity_folders)} identities")
        
        # Shuffle the samples to mix identities
        random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

class AffectNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, total_images=None, balance_dataset=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.total_images = total_images
        self.balance_dataset = balance_dataset
        
        # AffectNet emotion labels (0-7)
        self.emotion_names = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        import random
        
        # Load from train/val/test folders with 0-7 subfolders
        image_dir = os.path.join(self.root_dir, self.split)
        
        # Check if the directory exists
        if not os.path.exists(image_dir):
            print(f"Warning: Image directory {image_dir} not found!")
            return
        
        if not self.balance_dataset or self.total_images is None:
            # Original loading method - load all images
            for emotion_label in range(8):  # 0 to 7
                emotion_dir = os.path.join(image_dir, str(emotion_label))
                
                if not os.path.exists(emotion_dir):
                    print(f"Warning: Emotion directory {emotion_dir} not found, skipping...")
                    continue
                
                print(f"Loading emotion {emotion_label} from {emotion_dir}...")
                emotion_samples = []
                
                for img_name in os.listdir(emotion_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        img_path = os.path.join(emotion_dir, img_name)
                        emotion_samples.append((img_path, emotion_label))
                
                print(f"  Found {len(emotion_samples)} images for emotion {emotion_label}")
                self.samples.extend(emotion_samples)
        else:
            # Balanced loading method
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            
            # Get all class folders (0-7)
            class_folders = []
            for emotion_label in range(8):
                emotion_dir = os.path.join(image_dir, str(emotion_label))
                if os.path.exists(emotion_dir):
                    class_folders.append((str(emotion_label), emotion_label))
            
            print(f"Found {len(class_folders)} class folders")
            
            if len(class_folders) == 0:
                print("No valid class folders found!")
                return
            
            # Calculate images per class (equal distribution)
            images_per_class = self.total_images // len(class_folders)
            remaining_images = self.total_images % len(class_folders)
            
            print(f"Target: {images_per_class} images per class (total: {self.total_images})")
            
            # Collect images from each class
            for i, (class_folder, class_label) in enumerate(class_folders):
                class_path = os.path.join(image_dir, class_folder)
                
                # Some classes get one extra image to reach exactly total_images
                target_for_this_class = images_per_class + (1 if i < remaining_images else 0)
                
                # Get all images from this class
                class_images = []
                for image_file in os.listdir(class_path):
                    if image_file.lower().endswith(valid_extensions):
                        image_path = os.path.join(class_path, image_file)
                        class_images.append((image_path, class_label))
                
                print(f"Class {class_label}: found {len(class_images)} images, sampling {target_for_this_class}")
                
                # Randomly sample from this class's images
                if len(class_images) > target_for_this_class:
                    sampled_images = random.sample(class_images, target_for_this_class)
                else:
                    sampled_images = class_images  # Use all if less than target
                    print(f"Warning: Class {class_label} has only {len(class_images)} images, less than target {target_for_this_class}")
                
                # Add to dataset
                self.samples.extend(sampled_images)
                
                if len(self.samples) >= self.total_images:
                    break
            
            # Shuffle the final dataset
            random.shuffle(self.samples)
            
            print(f"Final dataset: {len(self.samples)} images, {len(set([label for _, label in self.samples]))} classes")
        
        print(f"AffectNet {self.split} samples loaded: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        # Temperature schedule
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)  # Only 2 global crops for teacher

        total_loss = 0
        n_loss_terms = 0
        
        # Student learns from teacher on all crops except same view
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue  # Skip same view
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / teacher_output.size(0)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def cosine_scheduler(epoch, max_epochs, initial_value, final_value):
    return final_value + 0.5 * (initial_value - final_value) * (
        1 + math.cos(math.pi * epoch / max_epochs)
    )


def train_dino(student_model, teacher_model, train_loader, optimizer, loss_fn, epoch, wd_schedule, scaler, device):
    student_model.train()
    teacher_model.eval()
    running_loss = 0.0
    total_samples = 0
    
    # Weight decay and learning rate scheduling
    for param_group in optimizer.param_groups:
        param_group["weight_decay"] = wd_schedule[epoch]
    
    # Cosine learning rate schedule
    lr = cosine_scheduler(epoch, len(wd_schedule), LR, MIN_LR)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    print(f"Epoch {epoch+1}, LR: {lr:.6f}, WD: {wd_schedule[epoch]:.4f}")
    
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}") as pbar:
        for idx, (all_crops, labels) in enumerate(train_loader):
            # Move crops to device
            all_crops = [crop.to(device, non_blocking=True) for crop in all_crops]
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                # Forward pass for student (all crops)
                student_outputs = []
                for crop in all_crops:
                    student_outputs.append(student_model(crop))
                student_output = torch.cat(student_outputs)
                
                # Forward pass for teacher (only first 2 global crops)
                with torch.no_grad():
                    teacher_outputs = []
                    for i in range(2):  # Only global crops
                        teacher_outputs.append(teacher_model(all_crops[i]))
                    teacher_output = torch.cat(teacher_outputs)
                
                # Loss calculation
                loss = loss_fn(student_output, teacher_output, epoch)
            
            # Check for invalid loss
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss at batch {idx}, skipping...")
                continue
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            
            # Optimizer step with scaling
            scaler.step(optimizer)
            scaler.update()
            
            # EMA update for teacher
            with torch.no_grad():
                momentum = 0.996  # Teacher momentum
                for param_q, param_k in zip(student_model.parameters(), teacher_model.parameters()):
                    param_k.data.mul_(momentum).add_(param_q.data, alpha=1. - momentum)
            
            # Statistics
            running_loss += loss.item() * len(labels)
            total_samples += len(labels)
            
            pbar.set_postfix({
                "Loss": f"{running_loss / total_samples:.4f}",
                "Grad": f"{grad_norm:.3f}",
                "Scale": f"{scaler.get_scale():.0f}"
            })
            pbar.update(1)
    
    return running_loss / total_samples


class FineTuneViT(nn.Module):
    def __init__(self, pretrained_dino_path, num_classes=7, freeze_backbone=False):
        super(FineTuneViT, self).__init__()
        
        # Load pretrained DINO model
        checkpoint = torch.load(pretrained_dino_path, map_location='cpu', weights_only=False)
        
        # Create backbone using timm
        self.backbone = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            num_classes=0,
            global_pool='token'
        )
        
        # Load pretrained weights (backbone only)
        pretrained_dict = checkpoint['student_state_dict']
        backbone_dict = {}
        for k, v in pretrained_dict.items():
            if k.startswith('backbone.'):
                backbone_dict[k[9:]] = v  # Remove 'backbone.' prefix
        
        self.backbone.load_state_dict(backbone_dict, strict=False)
        print("Loaded pretrained DINO weights into backbone")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen for fine-tuning")
        
        # Classification head
        feat_dim = self.backbone.embed_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.1),
            nn.Linear(feat_dim, num_classes)
        )
        
        # Initialize classifier
        self._init_classifier()
    
    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def evaluate_model(model, dataloader, device, criterion=None):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader) if criterion else 0
    
    return accuracy, avg_loss, all_preds, all_labels


def train_finetune_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}") as pbar:
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f"{running_loss / (pbar.n + 1):.4f}",
                'Acc': f"{100 * correct / total:.2f}%"
            })
            pbar.update(1)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def finetune_on_affectnet(pretrained_dino_path, affectnet_path, output_dir, num_epochs=50):
    print("Starting AffectNet Fine-tuning...")
    
    # Create fine-tuning model (8 classes for AffectNet)
    model = FineTuneViT(pretrained_dino_path, num_classes=8, freeze_backbone=False)
    model = model.to(DEVICE)
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Data transforms for fine-tuning
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load AffectNet dataset
    affectnet_train = AffectNetDataset(affectnet_path, split='train', transform=train_transform, 
                                     total_images=12000, balance_dataset=True)
    affectnet_val = AffectNetDataset(affectnet_path, split='val', transform=val_transform)
    affectnet_test = AffectNetDataset(affectnet_path, split='test', transform=val_transform)
        
    # Create dataloaders
    train_loader = DataLoader(affectnet_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(affectnet_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(affectnet_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"Train samples: {len(affectnet_train)}")
    print(f"Validation samples: {len(affectnet_val)}")
    print(f"Test samples: {len(affectnet_test)}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_finetune_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch)
        
        # Validate
        val_acc, val_loss, _, _ = evaluate_model(model, val_loader, DEVICE, criterion)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc * 100)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, os.path.join(output_dir, 'best_finetune_model.pth'))
            print(f"New best model saved! Val Acc: {val_acc*100:.2f}%")
        
        scheduler.step()
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_acc, test_loss, test_preds, test_labels = evaluate_model(model, test_loader, DEVICE, criterion)
    
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    
    # Generate classification report
    emotion_names = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
    report = classification_report(test_labels, test_preds, target_names=emotion_names)
    print("\nClassification Report:")
    print(report)
    
    # Save results
    plot_finetune_results(history, cm, output_dir)
    
    # Save final results
    results = {
        'best_val_acc': best_val_acc * 100,
        'test_acc': test_acc * 100,
        'test_loss': test_loss,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'history': history
    }
    
    with open(os.path.join(output_dir, 'finetune_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFine-tuning completed!")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Test accuracy: {test_acc*100:.2f}%")
    print(f"Results saved to: {output_dir}")


def plot_finetune_results(history, confusion_matrix, save_dir):
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Training Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Training Accuracy', color='blue')
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', color='red')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[0, 2].plot(history['learning_rates'], label='Learning Rate', color='green')
    axes[0, 2].set_title('Learning Rate Schedule')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale('log')
    
    # Confusion Matrix
    emotion_names = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_names, yticklabels=emotion_names, ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Training summary
    best_val_acc = max(history['val_acc'])
    best_val_loss = min(history['val_loss'])
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    
    axes[1, 1].text(0.1, 0.9, f'Best Val Acc: {best_val_acc:.2f}%', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.8, f'Best Val Loss: {best_val_loss:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, f'Final Train Acc: {final_train_acc:.2f}%', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f'Final Val Acc: {final_val_acc:.2f}%', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, f'Total Epochs: {len(history["train_loss"])}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.4, f'Architecture: ViT-T/16', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Training Summary')
    axes[1, 1].axis('off')
    
    # Per-class accuracy
    per_class_acc = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
    axes[1, 2].bar(emotion_names, per_class_acc * 100)
    axes[1, 2].set_title('Per-Class Accuracy')
    axes[1, 2].set_xlabel('Emotion')
    axes[1, 2].set_ylabel('Accuracy (%)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'finetune_results.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_history(history, save_path):
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.title('DINO Pre-training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate plot
    plt.subplot(2, 3, 2)
    plt.plot(history['learning_rates'], label='Learning Rate', color='green')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Teacher temperature plot
    plt.subplot(2, 3, 3)
    plt.plot(history['teacher_temps'], label='Teacher Temperature', color='red')
    plt.title('Teacher Temperature Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Weight decay plot
    plt.subplot(2, 3, 4)
    plt.plot(history['weight_decays'], label='Weight Decay', color='purple')
    plt.title('Weight Decay Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Decay')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss moving average
    plt.subplot(2, 3, 5)
    if len(history['train_loss']) > 5:
        moving_avg = np.convolve(history['train_loss'], np.ones(5)/5, mode='valid')
        plt.plot(range(4, len(history['train_loss'])), moving_avg, label='5-Epoch Moving Average', color='orange')
    plt.plot(history['train_loss'], alpha=0.5, label='Raw Loss', color='blue')
    plt.title('Loss with Moving Average')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Training metrics summary
    plt.subplot(2, 3, 6)
    final_loss = history['train_loss'][-1]
    min_loss = min(history['train_loss'])
    final_lr = history['learning_rates'][-1]
    plt.text(0.1, 0.8, f'Final Loss: {final_loss:.4f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f'Min Loss: {min_loss:.4f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f'Final LR: {final_lr:.2e}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f'Total Epochs: {len(history["train_loss"])}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f'Architecture: ViT-T/16', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.3, f'Embed Dim: {EMBED_DIM}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.2, f'Depth: {DEPTH}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.1, f'Heads: {NUM_HEADS}', fontsize=12, transform=plt.gca().transAxes)
    plt.title('Training Summary')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {save_path}")


def main():
    print(f"Using device: {DEVICE}")
    print("="*60)
    print("DINO ViT-T/16 Configuration:")
    print(f"  - Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  - Patch Size: {PATCH_SIZE}")
    print(f"  - Embedding Dimension: {EMBED_DIM}")
    print(f"  - Depth: {DEPTH}")
    print(f"  - Number of Heads: {NUM_HEADS}")
    print(f"  - MLP Ratio: {MLP_RATIO}")
    print(f"  - Output Dimension: {OUT_DIM}")
    print(f"  - Mixed Precision: Enabled")
    print("="*60)
    
    print("Creating ViT-T/16 models...")
    
    # Create models
    student = DinoViTTiny(output_dim=OUT_DIM, use_bn_in_head=USE_BN_IN_HEAD, 
                          norm_last_layer=NORM_LAST_LAYER).to(DEVICE)
    teacher = DinoViTTiny(output_dim=OUT_DIM, use_bn_in_head=USE_BN_IN_HEAD).to(DEVICE)
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        student = nn.DataParallel(student)
        teacher = nn.DataParallel(teacher)
    
    # Print model parameter count
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Student model parameters: {student_params:,}")
    
    # Initialize teacher with student weights
    for param_q, param_k in zip(student.parameters(), teacher.parameters()):
        param_k.data.copy_(param_q.data)
        param_k.requires_grad = False
    
    # Initialize mixed precision scaler
    scaler = GradScaler()
    print("Mixed precision training enabled with GradScaler")
    
    print("Setting up data transforms...")
    
    # Multi-crop transform for DINO
    dino_transform = MultiCropTransform(
        global_crops_scale=GLOBAL_CROPS_SCALE,
        local_crops_scale=LOCAL_CROPS_SCALE,
        local_crops_number=LOCAL_CROPS_NUMBER
    )
        
    # Custom collate function for multi-crop
    def multi_crop_collate(batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        # Apply multi-crop to each image
        multi_crops = []
        for img in images:
            crops = dino_transform(img)
            multi_crops.append(crops)
        
        # Stack all crops
        processed_crops = []
        for i in range(2 + LOCAL_CROPS_NUMBER):
            crop_i = torch.stack([multi_crops[j][i] for j in range(len(multi_crops))])
            processed_crops.append(crop_i)
        
        return processed_crops, torch.tensor(labels)
    
    # Create datasets and dataloaders
    print("Loading VGGFace2 dataset for pretraining...")

    vggface2_dataset = VGGFace2Dataset(
        VGGFACE2_PATH, 
        transform=transforms.Compose([
            transforms.Lambda(lambda x: x)
        ]),
        max_total_images=40000
    )
    
    train_loader = DataLoader(
        vggface2_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=multi_crop_collate,
        drop_last=True,
        pin_memory=True,        
        persistent_workers=True, 
        prefetch_factor=2       
    )
    
    print("Setting up loss function and optimizer...")
    
    # DINO loss
    dino_loss = DINOLoss(
        OUT_DIM,
        ncrops=LOCAL_CROPS_NUMBER + 2,
        warmup_teacher_temp=WARMUP_TEACHER_TEMP,
        teacher_temp=TEACHER_TEMP,
        warmup_teacher_temp_epochs=WARMUP_TEACHER_TEMP_EPOCHS,
        nepochs=NUM_EPOCHS_PRETRAINING,
        student_temp=STUDENT_TEMP
    ).to(DEVICE)
    
    # Prepare optimizer
    params_groups = [
        {'params': [p for n, p in student.named_parameters() if "last_layer" not in n]},
        {'params': [p for n, p in student.named_parameters() if "last_layer" in n], 'lr': LR}
    ]
    optimizer = optim.AdamW(params_groups, lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Prepare weight decay schedule
    weight_decay_schedule = np.linspace(WEIGHT_DECAY, WEIGHT_DECAY_END, NUM_EPOCHS_PRETRAINING)
    
    # Track history
    history = {
        'train_loss': [],
        'learning_rates': [],
        'teacher_temps': [],
        'weight_decays': [],
        'grad_scales': []  # Track gradient scaling
    }
    
    print(f"Starting DINO pretraining with ViT-T/16 for {NUM_EPOCHS_PRETRAINING} epochs...")
    print(f"Learning rate will decay from {LR} to {MIN_LR}")
    print(f"Teacher temperature will go from {WARMUP_TEACHER_TEMP} to {TEACHER_TEMP}")
    print(f"Mixed precision training: Enabled")
    
    # Training loop
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS_PRETRAINING):
        print(f"\n{'='*60}")
        print(f"Starting Epoch {epoch+1}/{NUM_EPOCHS_PRETRAINING}")
        print(f"{'='*60}")
        
        # Train for one epoch with scaler
        epoch_loss = train_dino(
            student, teacher, train_loader, optimizer, dino_loss, 
            epoch, weight_decay_schedule, scaler, DEVICE
        )
        
        # Record training metrics
        current_lr = optimizer.param_groups[0]['lr']
        current_wd = weight_decay_schedule[epoch]
        current_temp = dino_loss.teacher_temp_schedule[min(epoch, len(dino_loss.teacher_temp_schedule) - 1)]
        current_scale = scaler.get_scale()
        
        history['train_loss'].append(epoch_loss)
        history['learning_rates'].append(current_lr)
        history['weight_decays'].append(current_wd)
        history['teacher_temps'].append(current_temp)
        history['grad_scales'].append(current_scale)
        
        # Print epoch summary
        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        eta = avg_time_per_epoch * (NUM_EPOCHS_PRETRAINING - epoch - 1)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  - Loss: {epoch_loss:.6f}")
        print(f"  - Learning Rate: {current_lr:.6f}")
        print(f"  - Weight Decay: {current_wd:.6f}")
        print(f"  - Teacher Temp: {current_temp:.6f}")
        print(f"  - Grad Scale: {current_scale:.0f}")
        print(f"  - Time Elapsed: {elapsed_time/3600:.2f}h")
        print(f"  - ETA: {eta/3600:.2f}h")
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # Handle DataParallel models for saving
            student_state = student.module.state_dict() if hasattr(student, 'module') else student.state_dict()
            teacher_state = teacher.module.state_dict() if hasattr(teacher, 'module') else teacher.state_dict()
            
            best_checkpoint = {
                'epoch': epoch + 1,
                'student_state_dict': student_state,
                'teacher_state_dict': teacher_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # Save scaler state
                'loss': epoch_loss,
                'history': history,
                'config': {
                    'img_size': IMG_SIZE,
                    'patch_size': PATCH_SIZE,
                    'embed_dim': EMBED_DIM,
                    'depth': DEPTH,
                    'num_heads': NUM_HEADS,
                    'mlp_ratio': MLP_RATIO,
                    'out_dim': OUT_DIM
                }
            }
            torch.save(best_checkpoint, os.path.join(OUTPUT_DIR, 'dino_vit_tiny_best.pth'))
            print(f"  - New best model saved! (Loss: {epoch_loss:.6f})")
        
        # Save checkpoint at regular intervals
        if (epoch + 1) % SAVE_FREQ == 0 or epoch == NUM_EPOCHS_PRETRAINING - 1:
            student_state = student.module.state_dict() if hasattr(student, 'module') else student.state_dict()
            teacher_state = teacher.module.state_dict() if hasattr(teacher, 'module') else teacher.state_dict()
            
            checkpoint = {
                'epoch': epoch + 1,
                'student_state_dict': student_state,
                'teacher_state_dict': teacher_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': epoch_loss,
                'history': history,
                'config': {
                    'img_size': IMG_SIZE,
                    'patch_size': PATCH_SIZE,
                    'embed_dim': EMBED_DIM,
                    'depth': DEPTH,
                    'num_heads': NUM_HEADS,
                    'mlp_ratio': MLP_RATIO,
                    'out_dim': OUT_DIM
                }
            }
            checkpoint_path = os.path.join(OUTPUT_DIR, f'dino_vit_tiny_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"  - Checkpoint saved: {checkpoint_path}")
        
        # Save training history plot
        if (epoch + 1) % 10 == 0 or epoch == NUM_EPOCHS_PRETRAINING - 1:
            plot_path = os.path.join(OUTPUT_DIR, f'training_history_epoch_{epoch+1}.png')
            plot_training_history(history, plot_path)
    
    # Final training summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("DINO PRETRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Average time per epoch: {total_time/NUM_EPOCHS_PRETRAINING/60:.2f} minutes")
    print(f"Best loss achieved: {best_loss:.6f}")
    print(f"Final loss: {history['train_loss'][-1]:.6f}")
    print(f"Final gradient scale: {history['grad_scales'][-1]:.0f}")
    print(f"Model saved at: {os.path.join(OUTPUT_DIR, 'dino_vit_tiny_best.pth')}")
    
    # Save final training history
    final_plot_path = os.path.join(OUTPUT_DIR, 'final_training_history.png')
    plot_training_history(history, final_plot_path)
    
    # Save training history as JSON
    history_path = os.path.join(OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_history = {
            'train_loss': [float(x) for x in history['train_loss']],
            'learning_rates': [float(x) for x in history['learning_rates']],
            'teacher_temps': [float(x) for x in history['teacher_temps']],
            'weight_decays': [float(x) for x in history['weight_decays']],
            'grad_scales': [float(x) for x in history['grad_scales']]
        }
        json.dump(json_history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Mixed precision pretraining completed successfully!")
    
    print("\n" + "="*60)
    print("STARTING AffectNet FINE-TUNING")
    print("="*60)
    
    # Fine-tune on AFFECTNET
    pretrained_dino_path = os.path.join(OUTPUT_DIR, 'dino_vit_tiny_best.pth')
    finetune_on_affectnet(pretrained_dino_path, AFFECTNET_PATH, OUTPUT_DIR, num_epochs=50)

if __name__ == "__main__":
    main()

