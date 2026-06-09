#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete Video Classification Pipeline
Multi-label video classification using various 3D CNN architectures
"""

# ==================== INSTALL REQUIRED PACKAGES ====================
!pip install decord pytorchvideo scikit-learn matplotlib tqdm --quiet

# ==================== IMPORTS ====================
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import gc
from time import time
from IPython.display import clear_output
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from decord import VideoReader, cpu
from pytorchvideo.models.hub import slowfast_r50, i3d_r50, x3d_xs, x3d_s, x3d_m
from torchvision.models.video import mc3_18, r3d_18, r2plus1d_18

# ==================== SETUP ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== DATA ANALYSIS ====================
def analyze_video_dataset(csv_path, video_dir, sample_size=100):
    """Analyze video dataset: lengths, labels distribution, etc."""
    df = pd.read_csv(csv_path)
    
    # Fix label issues (like "cloud." -> "cloud")
    df['labels'] = df['labels'].str.replace('cloud.', 'cloud')
    
    print(f"Total videos: {len(df)}")
    print("\nSample data:")
    print(df.head())
    
    # Compute video lengths for a sample
    print(f"\nComputing video lengths for {sample_size} samples...")
    lengths = []
    errors = 0
    
    for idx, path in enumerate(tqdm(df['path'][:sample_size], desc="Sampling videos")):
        try:
            vr = VideoReader(os.path.join(video_dir, path), ctx=cpu(0))
            lengths.append(len(vr))
        except Exception as e:
            errors += 1
    
    if errors > 0:
        print(f"âš ï¸� Failed to read {errors} videos")
    
    if lengths:
        plt.figure(figsize=(10, 5))
        plt.hist(lengths, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel("Number of frames in video")
        plt.ylabel("Number of videos")
        plt.title("Video Length Distribution (sample)")
        plt.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lengths):.0f}')
        plt.axvline(np.median(lengths), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(lengths):.0f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"\nVideo length statistics:")
        print(f"Mean: {np.mean(lengths):.1f} frames")
        print(f"Median: {np.median(lengths):.1f} frames")
        print(f"Min: {np.min(lengths)} frames")
        print(f"Max: {np.max(lengths)} frames")
    
    # Analyze labels
    all_labels = []
    for label_str in df['labels']:
        all_labels.extend(label_str.split(", "))
    
    label_counts = pd.Series(all_labels).value_counts()
    print("\nLabel distribution:")
    print(label_counts)
    
    plt.figure(figsize=(10, 5))
    label_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Label Distribution")
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return df

# ==================== DATASET CLASS ====================
class VideoDataset(Dataset):
    """Enhanced video dataset with multiple sampling strategies."""
    
    def __init__(self, df, video_dir, transform=None, num_frames=16, 
                 cache_dir=None, sampling_strategy='uniform'):
        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        self.transform = transform
        self.num_frames = num_frames
        self.sampling_strategy = sampling_strategy
        self.cache_dir = cache_dir or os.path.join(video_dir, "_cached")
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Extract unique labels
        all_labels = set()
        for label_str in self.df['labels']:
            all_labels.update(label_str.split(", "))
        self.classes = sorted(list(all_labels))
        self.class2idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx2class = {idx: cls for idx, cls in enumerate(self.classes)}
        
        print(f"Found {len(self.classes)} classes: {self.classes}")
        
        # Calculate class weights for imbalanced data
        label_counts = np.zeros(len(self.classes))
        for label_str in self.df['labels']:
            for label in label_str.split(", "):
                if label in self.class2idx:
                    label_counts[self.class2idx[label]] += 1
        
        self.class_weights = len(self.df) / (len(self.classes) * label_counts + 1e-6)
        self.class_weights = torch.FloatTensor(self.class_weights)
    
    def __len__(self):
        return len(self.df)
    
    def _get_cache_path(self, filename):
        base = os.path.splitext(os.path.basename(filename))[0]
        return os.path.join(self.cache_dir, f"{base}_{self.num_frames}_{self.sampling_strategy}.pt")
    
    def _sample_frames_uniform(self, total_frames):
        """Uniform sampling across the video."""
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(np.int32)
        else:
            indices = np.concatenate([
                np.arange(total_frames),
                np.full(self.num_frames - total_frames, total_frames - 1)
            ]).astype(np.int32)
        return indices
    
    def _sample_frames_random(self, total_frames):
        """Random sampling with temporal order preserved."""
        if total_frames >= self.num_frames:
            indices = np.sort(np.random.choice(total_frames, self.num_frames, replace=False))
        else:
            indices = self._sample_frames_uniform(total_frames)
        return indices
    
    def _sample_frames_segment(self, total_frames):
        """Segment-based sampling for better temporal coverage."""
        if total_frames >= self.num_frames:
            segments = np.array_split(np.arange(total_frames), self.num_frames)
            indices = [np.random.choice(seg) for seg in segments if len(seg) > 0]
            indices = np.array(indices)
        else:
            indices = self._sample_frames_uniform(total_frames)
        return indices
    
    def _extract_frames(self, video_file):
        """Extract frames with specified sampling strategy."""
        vr = VideoReader(video_file, ctx=cpu(0))
        total_frames = len(vr)
        
        # Sample indices based on strategy
        if self.sampling_strategy == 'uniform':
            indices = self._sample_frames_uniform(total_frames)
        elif self.sampling_strategy == 'random':
            indices = self._sample_frames_random(total_frames)
        elif self.sampling_strategy == 'segment':
            indices = self._sample_frames_segment(total_frames)
        else:
            indices = self._sample_frames_uniform(total_frames)
        
        frames = vr.get_batch(indices).asnumpy()
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]
        
        if self.transform:
            frames_list = []
            for t in range(frames.shape[1]):
                frame = frames[:, t, :, :]  # [C, H, W]
                frame = self.transform(frame)
                frames_list.append(frame)
            frames = torch.stack(frames_list, dim=1)  # [C, T, H, W]
        
        return frames
    
    def __getitem__(self, idx):
        video_filename = self.df.iloc[idx]['path']
        label_str = self.df.iloc[idx]['labels']
        video_file = os.path.join(self.video_dir, video_filename)
        cache_file = self._get_cache_path(video_filename)
        
        # Try to load from cache
        if os.path.exists(cache_file) and self.sampling_strategy == 'uniform':
            try:
                frames = torch.load(cache_file)
            except:
                frames = self._extract_frames(video_file)
                torch.save(frames, cache_file)
        else:
            frames = self._extract_frames(video_file)
            if self.sampling_strategy == 'uniform':  # Only cache uniform sampling
                try:
                    torch.save(frames, cache_file)
                except:
                    pass  # Ignore cache errors
        
        # Create binary target
        target = torch.zeros(len(self.class2idx), dtype=torch.float32)
        for label in label_str.split(", "):
            if label in self.class2idx:
                target[self.class2idx[label]] = 1.0
        
        return frames, target

# ==================== LOSS FUNCTIONS ====================
class FocalBCELoss(nn.Module):
    """Focal loss for imbalanced multi-label classification."""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class WeightedBCELoss(nn.Module):
    """Weighted BCE loss using class weights."""
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        if self.class_weights is not None:
            weight = self.class_weights.to(inputs.device)
            loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            loss = loss * weight
            return loss.mean()
        else:
            return F.binary_cross_entropy_with_logits(inputs, targets)

# ==================== MODEL DEFINITIONS ====================
def pack_slowfast_pathway(frames, alpha=4):
    """Pack frames for SlowFast model."""
    fast = frames  # Original temporal resolution
    slow = frames[:, :, ::alpha, :, :]  # Subsample temporally
    return [slow, fast]

def get_model(name: str, num_classes=9, device=device, dropout_rate=0.5, 
              freeze_backbone=False, use_pretrained=True):
    """Enhanced model factory with more options."""
    
    if name == 'slowfast':
        model = slowfast_r50(pretrained=use_pretrained)
        in_features = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    elif name == 'i3d':
        model = i3d_r50(pretrained=use_pretrained)
        in_features = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    elif name == 'x3d_xs':
        model = x3d_xs(pretrained=use_pretrained)
        in_features = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    elif name == 'x3d_s':
        model = x3d_s(pretrained=use_pretrained)
        in_features = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    elif name == 'r3d':
        model = r3d_18(pretrained=use_pretrained)
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )
    
    elif name == 'mc3':
        model = mc3_18(pretrained=use_pretrained)
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )
    
    elif name == 'r2plus1d':
        model = r2plus1d_18(pretrained=use_pretrained)
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )
    
    else:
        raise ValueError(f"Unknown model: {name}")
    
    # Optionally freeze backbone
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name and 'proj' not in name:
                param.requires_grad = False
    
    return model.to(device)

# ==================== TRAINING UTILITIES ====================
class Trainer:
    """Enhanced trainer with all improvements."""
    
    def __init__(self, model, optimizer, criterion, device, scheduler=None, 
                 save_path="best_model.pth", model_name='not_slowfast', 
                 use_amp=True, gradient_clip=1.0):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.save_path = save_path
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.model_name = model_name
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_clip = gradient_clip
        
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.val_f1s = []
        self.val_precisions = []
        self.val_recalls = []
        self.learning_rates = []
    
    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc="ğŸš‚ Training", leave=False)
        for videos, labels in loop:
            try:
                videos, labels = videos.to(self.device), labels.to(self.device)
                
                if self.model_name == "slowfast":
                    videos = pack_slowfast_pathway(videos)
                
                self.optimizer.zero_grad()
                
                # Mixed precision training
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(videos)
                        loss = self.criterion(outputs, labels)
                    
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.gradient_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    
                    # Gradient clipping
                    if self.gradient_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    
                    self.optimizer.step()
                
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("âš ï¸� Out of memory! Clearing cache and skipping batch.")
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise e
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            loop = tqdm(val_loader, desc="ğŸ”� Validating", leave=False)
            for videos, labels in loop:
                videos, labels = videos.to(self.device), labels.to(self.device)
                
                if self.model_name == "slowfast":
                    videos = pack_slowfast_pathway(videos)
                
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
                targets = labels.cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets)
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Per-class and average metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='macro', zero_division=0
        )
        
        return avg_loss, f1, precision, recall
    
    def fit(self, train_loader, val_loader, num_epochs, patience=5):
        start = time()
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time()
            
            # Training
            train_loss = self.train_one_epoch(train_loader)
            
            # Validation
            val_loss, val_f1, val_precision, val_recall = self.validate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_f1s.append(val_f1)
            self.val_precisions.append(val_precision)
            self.val_recalls.append(val_recall)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Clear output and plot
            clear_output(wait=True)
            self._plot_live(epoch, num_epochs)
            
            epoch_finish = time()
            print(f"\nEpoch {epoch}/{num_epochs} (finished in {epoch_finish - epoch_start:.2f}s)")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val F1: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"Time passed: {epoch_finish - start:.0f}s (Avg: {(epoch_finish - start) / epoch:.2f}s/epoch)")
            
            # Save best model (based on F1 score)
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_f1': self.best_val_f1,
                    'best_val_loss': self.best_val_loss,
                }, self.save_path)
                print("ğŸ’¾ Saved new best model!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nâ�¹ï¸� Early stopping triggered! Best epoch was {best_epoch}")
                    break
    
    def _plot_live(self, current_epoch, total_epochs):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.train_losses, label='Train Loss', marker='o', markersize=4)
        ax1.plot(epochs, self.val_losses, label='Val Loss', marker='o', markersize=4)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training vs Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # F1 Score plot
        ax2.plot(epochs, self.val_f1s, label='Val F1', marker='o', color='green', markersize=4)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("F1 Score")
        ax2.set_title("Validation F1 Score")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Precision/Recall plot
        ax3.plot(epochs, self.val_precisions, label='Precision', marker='s', color='blue', markersize=4)
        ax3.plot(epochs, self.val_recalls, label='Recall', marker='^', color='orange', markersize=4)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Score")
        ax3.set_title("Precision and Recall")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Learning rate plot
        ax4.plot(epochs, self.learning_rates, label='Learning Rate', marker='d', color='red', markersize=4)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Learning Rate")
        ax4.set_title("Learning Rate Schedule")
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ==================== PREDICTION & EVALUATION ====================
def predict_with_tta(model, video_path, idx2class, num_frames=32, num_clips=5, 
                    transform=None, device='cuda', model_name='not_slowfast'):
    """Test-time augmentation for better predictions."""
    model.eval()
    predictions = []
    
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    for _ in range(num_clips):
        # Random temporal crop
        if total_frames > num_frames:
            start_idx = np.random.randint(0, total_frames - num_frames)
            indices = np.arange(start_idx, start_idx + num_frames)
        else:
            indices = np.arange(total_frames)
            if len(indices) < num_frames:
                indices = np.concatenate([indices, [total_frames-1] * (num_frames - len(indices))])
        
        frames = vr.get_batch(indices).asnumpy()
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        
        if transform:
            frames_list = []
            for t in range(frames.shape[1]):
                frame = frames[:, t, :, :]
                frame = transform(frame)
                frames_list.append(frame)
            frames = torch.stack(frames_list, dim=1)
        
        frames = frames.unsqueeze(0).to(device)
        
        if model_name == "slowfast":
            frames = pack_slowfast_pathway(frames)
        
        with torch.no_grad():
            logits = model(frames)
            probs = torch.sigmoid(logits)
            predictions.append(probs)
    
    # Average predictions
    avg_probs = torch.stack(predictions).mean(0).cpu().numpy()[0]
    return avg_probs

def predict_and_save(model, test_dir, output_csv, idx2class, model_weights_path=None,
                    num_frames=32, transform=None, threshold=0.5, device='cuda',
                    model_name='not_slowfast', use_tta=False):
    """Generate predictions with optional TTA."""
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)) if model_name == 'slowfast' else transforms.Resize((112, 112)),
            transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
        ])
    
    if model_weights_path is not None:
        checkpoint = torch.load(model_weights_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    
    results = []
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.mp4')])
    
    for idx, filename in enumerate(tqdm(test_files, desc="ğŸ”� Predicting")):
        video_path = os.path.join(test_dir, filename)
        
        try:
            if use_tta:
                probs = predict_with_tta(model, video_path, idx2class, num_frames, 
                                        transform=transform, device=device, model_name=model_name)
            else:
                # Standard prediction
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)
                
                if total_frames >= num_frames:
                    indices = np.linspace(0, total_frames - 1, num_frames).astype(np.int32)
                else:
                    indices = np.concatenate([
                        np.arange(total_frames),
                        np.full(num_frames - total_frames, total_frames - 1)
                    ]).astype(np.int32)
                
                frames = vr.get_batch(indices).asnumpy()
                frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
                
                if transform:
                    frames_list = []
                    for t in range(frames.shape[1]):
                        frame = frames[:, t, :, :]
                        frame = transform(frame)
                        frames_list.append(frame)
                    frames = torch.stack(frames_list, dim=1)
                
                frames = frames.unsqueeze(0).to(device)
                
                if model_name == "slowfast":
                    frames = pack_slowfast_pathway(frames)
                
                with torch.no_grad():
                    logits = model(frames)
                    probs = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Get predictions
            predicted_indices = [i for i, p in enumerate(probs) if p >= threshold]
            
            if not predicted_indices:
                predicted_indices = [int(probs.argmax())]
            
            predicted_labels = [idx2class[i] for i in predicted_indices]
            label_str = ", ".join(predicted_labels)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            label_str = idx2class[0]  # Default to first class
        
        results.append((idx, filename, label_str))
    
    # Save predictions
    df = pd.DataFrame(results, columns=["index", "file_name", "label"])
    df.to_csv(output_csv, index=False)
    print(f"ğŸ“� Saved predictions to: {output_csv}")

# ==================== MAIN TRAINING PIPELINE ====================
def train_video_model(model_name, train_csv, train_video_dir, test_video_dir, 
                     cache_dir, num_frames=32, batch_size=8, num_epochs=20, 
                     learning_rate=1e-4, split_ratio=0.8, dropout_rate=0.5,
                     use_scheduler=True, use_amp=True, use_tta=False,
                     loss_type='weighted_bce', sampling_strategy='uniform'):
    """Complete training pipeline."""
    
    print(f"ğŸ�¬ Training {model_name} model")
    print(f"Configuration: frames={num_frames}, batch_size={batch_size}, lr={learning_rate}")
    
    # Load and prepare data
    df = pd.read_csv(train_csv)
    df['labels'] = df['labels'].str.replace('cloud.', 'cloud')  # Fix label issues
    
    # Model-specific settings
    if model_name == 'slowfast':
        resize_size = (224, 224)
    else:
        resize_size = (112, 112)
    
    # Training transforms
    train_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2)),
        transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
    ])
    
    # Create datasets
    train_dataset = VideoDataset(
        df=df,
        video_dir=train_video_dir,
        transform=train_transform,
        num_frames=num_frames,
        cache_dir=cache_dir,
        sampling_strategy=sampling_strategy
    )
    
    val_dataset = VideoDataset(
        df=df,
        video_dir=train_video_dir,
        transform=val_transform,
        num_frames=num_frames,
        cache_dir=cache_dir,
        sampling_strategy='uniform'  # Always use uniform for validation
    )
    
    # Split dataset
    train_size = int(split_ratio * len(train_dataset))
    val_size = len(train_dataset) - train_size
    indices = torch.randperm(len(train_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, 
        num_workers=2, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=True
    )
    
    print(f"Dataset: {len(train_subset)} training, {len(val_subset)} validation")
    print(f"Classes: {train_dataset.classes}")
    
    # Create model
    num_classes = len(train_dataset.classes)
    model = get_model(model_name, num_classes=num_classes, dropout_rate=dropout_rate)
    
    # Loss function
    if loss_type == 'weighted_bce':
        criterion = WeightedBCELoss(train_dataset.class_weights)
    elif loss_type == 'focal':
        criterion = FocalBCELoss(gamma=2.0, alpha=0.25)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-4,
        amsgrad=True
    )
    
    # Learning rate scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, 
            verbose=True, min_lr=1e-6
        )
    
    # Create trainer
    save_path = f"/kaggle/working/{model_name}_f{num_frames}_bs{batch_size}_lr{learning_rate}_best.pth"
    trainer = Trainer(
        model, optimizer, criterion, device, scheduler,
        save_path=save_path, model_name=model_name, use_amp=use_amp
    )
    
    # Train model
    trainer.fit(train_loader, val_loader, num_epochs, patience=7)
    
    # Generate predictions on test set
    print("\nğŸ�¯ Generating predictions on test set...")
    predict_and_save(
        model,
        test_video_dir,
        f"/kaggle/working/predictions_{model_name}_f{num_frames}_{'tta' if use_tta else 'standard'}.csv",
        train_dataset.idx2class,
        model_weights_path=save_path,
        num_frames=num_frames,
        model_name=model_name,
        use_tta=use_tta
    )
    
    return model, train_dataset.idx2class

# ==================== ENSEMBLE PREDICTIONS ====================
def ensemble_predictions(model_configs, test_video_dir, output_csv):
    """Ensemble multiple models for better predictions."""
    all_predictions = []
    idx2class = None
    
    for config in model_configs:
        print(f"\nLoading {config['name']}...")
        model = get_model(config['name'], num_classes=config['num_classes'])
        checkpoint = torch.load(config['checkpoint'], map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        model.to(device)
        
        if idx2class is None:
            idx2class = config['idx2class']
        
        # Get predictions for all test videos
        test_files = sorted([f for f in os.listdir(test_video_dir) if f.endswith('.mp4')])
        model_preds = []
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)) if config['name'] == 'slowfast' else transforms.Resize((112, 112)),
            transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
        ])
        
        for filename in tqdm(test_files, desc=f"Predicting with {config['name']}"):
            video_path = os.path.join(test_video_dir, filename)
            
            try:
                # Standard prediction
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)
                num_frames = config['num_frames']
                
                if total_frames >= num_frames:
                    indices = np.linspace(0, total_frames - 1, num_frames).astype(np.int32)
                else:
                    indices = np.concatenate([
                        np.arange(total_frames),
                        np.full(num_frames - total_frames, total_frames - 1)
                    ]).astype(np.int32)
                
                frames = vr.get_batch(indices).asnumpy()
                frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
                
                frames_list = []
                for t in range(frames.shape[1]):
                    frame = frames[:, t, :, :]
                    frame = transform(frame)
                    frames_list.append(frame)
                frames = torch.stack(frames_list, dim=1).unsqueeze(0).to(device)
                
                if config['name'] == "slowfast":
                    frames = pack_slowfast_pathway(frames)
                
                with torch.no_grad():
                    logits = model(frames)
                    probs = torch.sigmoid(logits).cpu().numpy()[0]
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                probs = np.zeros(config['num_classes'])
                probs[0] = 1.0  # Default to first class
            
            model_preds.append(probs)
        
        all_predictions.append(np.array(model_preds))
    
    # Average predictions
    avg_predictions = np.mean(all_predictions, axis=0)
    
    # Generate final predictions
    results = []
    for idx, (filename, probs) in enumerate(zip(test_files, avg_predictions)):
        predicted_indices = [i for i, p in enumerate(probs) if p >= 0.5]
        if not predicted_indices:
            predicted_indices = [int(probs.argmax())]
        
        predicted_labels = [idx2class[i] for i in predicted_indices]
        label_str = ", ".join(predicted_labels)
        results.append((idx, filename, label_str))
    
    df = pd.DataFrame(results, columns=["index", "file_name", "label"])
    df.to_csv(output_csv, index=False)
    print(f"ğŸ“� Saved ensemble predictions to: {output_csv}")

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Paths
    TRAIN_CSV = '/kaggle/input/what-on-the-video/train.csv'
    TRAIN_VIDEO_DIR = '/kaggle/input/what-on-the-video/train'
    TEST_VIDEO_DIR = '/kaggle/input/what-on-the-video/test'
    CACHE_DIR = '/kaggle/working/cache/'
    
    # Analyze dataset
    print("ğŸ“Š Analyzing dataset...")
    df = analyze_video_dataset(TRAIN_CSV, TRAIN_VIDEO_DIR, sample_size=50)
    
    # Train multiple models
    models_to_train = [
        {'name': 'r3d', 'frames': 32, 'bs': 8, 'lr': 1e-4},
        {'name': 'r2plus1d', 'frames': 16, 'bs': 8, 'lr': 1e-4},
        # Uncomment to train more models:
        # {'name': 'mc3', 'frames': 16, 'bs': 16, 'lr': 2e-4},
        # {'name': 'x3d_xs', 'frames': 16, 'bs': 8, 'lr': 5e-5},
        # {'name': 'i3d', 'frames': 32, 'bs': 4, 'lr': 5e-5},
        # {'name': 'slowfast', 'frames': 32, 'bs': 4, 'lr': 5e-5},
    ]
    
    trained_models = []
    
    for config in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {config['name']} model")
        print(f"{'='*60}\n")
        
        try:
            model, idx2class = train_video_model(
                model_name=config['name'],
                train_csv=TRAIN_CSV,
                train_video_dir=TRAIN_VIDEO_DIR,
                test_video_dir=TEST_VIDEO_DIR,
                cache_dir=CACHE_DIR,
                num_frames=config['frames'],
                batch_size=config['bs'],
                num_epochs=25,
                learning_rate=config['lr'],
                split_ratio=0.8,
                dropout_rate=0.5,
                use_scheduler=True,
                use_amp=True,
                use_tta=True,  # Enable test-time augmentation
                loss_type='weighted_bce',
                sampling_strategy='segment'  # Better temporal coverage
            )
            
            trained_models.append({
                'name': config['name'],
                'checkpoint': f"/kaggle/working/{config['name']}_f{config['frames']}_bs{config['bs']}_lr{config['lr']}_best.pth",
                'num_frames': config['frames'],
                'num_classes': len(idx2class),
                'idx2class': idx2class
            })
            
        except Exception as e:
            print(f"â�Œ Error training {config['name']}: {str(e)}")
            continue
    
    # Ensemble predictions (if multiple models trained)
    if len(trained_models) > 1:
        print("\nğŸ�¯ Creating ensemble predictions...")
        ensemble_predictions(trained_models, TEST_VIDEO_DIR, "/kaggle/working/predictions_ensemble.csv")
    
    print("\nâœ… Complete! Check the output CSV files for predictions.")
    
    # Optional: Clean up cache
    # import shutil
    # if os.path.exists(CACHE_DIR):
    #     shutil.rmtree(CACHE_DIR)
    #     print("ğŸ§¹ Cleaned up cache directory")

