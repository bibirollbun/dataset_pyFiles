"""
LEVERAGING MACHINE LEARNING TECHNIQUES FOR THE CLASSIFICATION 
AND PREDICTION OF DETRIMENTAL NEURAL PATTERNS

Kaggle Persistent Training with Auto-Download Links

HOW IT WORKS:
1. Trains 2 epochs
2. Automatically creates download links for checkpoints
3. Download files to your computer
4. Next session: Upload files back
5. Automatically resumes from checkpoint!

INSTRUCTIONS:
- First run: Just run this cell
- Later runs: Upload checkpoint files first, then run
"""

print("=" * 80)
print("LEVERAGING MACHINE LEARNING TECHNIQUES")
print("FOR THE CLASSIFICATION AND PREDICTION OF")
print("DETRIMENTAL NEURAL PATTERNS")
print("=" * 80)
print("Kaggle Training with Checkpoint Persistence")
print("=" * 80)

# ============================================================================
# Setup
# ============================================================================
print("\n[1/11] Importing libraries...")

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import gc
from tqdm.auto import tqdm
import cv2
from datetime import datetime
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Install timm if needed
try:
    import timm
except:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm==0.9.12"])
    import timm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"âœ“ Device: {device}")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
print("âœ“ Libraries imported")

# ============================================================================
# Config
# ============================================================================
print("\n[2/11] Setting up configuration...")

class Config:
    BASE_PATH = Path('/kaggle/input/hms-harmful-brain-activity-classification')
    OUTPUT_DIR = Path('/kaggle/working')
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    MODEL_NAME = 'tf_efficientnetv2_m'
    IMG_SIZE = 512
    NUM_CLASSES = 6
    
    TARGET_COLS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 
                   'lrda_vote', 'grda_vote', 'other_vote']
    CLASS_NAMES = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
    
    BATCH_SIZE = 16
    NUM_WORKERS = 0
    EPOCHS_PER_RUN = 2
    TOTAL_EPOCHS = 20
    LEARNING_RATE = 1e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-5
    PATIENCE = 5
    
    DEVICE = device
    USE_AMP = True

config = Config()
print(f"âœ“ Configuration set")

# ============================================================================
# Check for Uploaded Checkpoints
# ============================================================================
print("\n[3/11] Checking for uploaded checkpoint files...")

# Look for checkpoints in /kaggle/input/ (uploaded files)
input_dirs = list(Path('/kaggle/input').glob('*checkpoint*'))
uploaded_checkpoint = None

for input_dir in input_dirs:
    checkpoint_files = list(input_dir.glob('checkpoint_epoch_*.pth'))
    if checkpoint_files:
        uploaded_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        print(f"âœ“ Found uploaded checkpoint: {uploaded_checkpoint.name}")
        # Copy to working directory
        shutil.copy(uploaded_checkpoint, config.OUTPUT_DIR / uploaded_checkpoint.name)
        
        # Also copy history if exists
        history_file = input_dir / 'training_history.pkl'
        if history_file.exists():
            shutil.copy(history_file, config.OUTPUT_DIR / 'training_history.pkl')
        break

if not uploaded_checkpoint:
    print("  No uploaded checkpoint found - starting fresh training")
    print("  (This is normal for the first run!)")

# ============================================================================
# Load Data
# ============================================================================
print("\n[4/11] Loading dataset...")

train_df = pd.read_csv(config.BASE_PATH / 'train.csv')

vote_sums = train_df[config.TARGET_COLS].sum(axis=1)
probabilities = train_df[config.TARGET_COLS].div(vote_sums, axis=0)
probabilities = probabilities.fillna(1.0 / config.NUM_CLASSES)
train_df['target_probs'] = list(probabilities.values)
train_df['target_label'] = train_df[config.TARGET_COLS].values.argmax(axis=1)

unique_patients = train_df['patient_id'].unique()
np.random.seed(42)
np.random.shuffle(unique_patients)
n_val_patients = int(len(unique_patients) * 0.2)
val_patients = unique_patients[:n_val_patients]

train_df['fold'] = 0
train_df.loc[train_df['patient_id'].isin(val_patients), 'fold'] = 1

print(f"âœ“ Loaded {len(train_df):,} samples")

# ============================================================================
# Dataset
# ============================================================================
print("\n[5/11] Creating datasets...")

class NeuralPatternsDataset(Dataset):
    """Dataset for Detrimental Neural Patterns Classification"""
    def __init__(self, df, mode='train'):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.spec_path = config.BASE_PATH / 'train_spectrograms'
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        spec_id = row['spectrogram_id']
        filepath = self.spec_path / f"{spec_id}.parquet"
        spec_df = pd.read_parquet(filepath)
        
        montages = ['LL', 'RL', 'LP', 'RP']
        channels = []
        for montage in montages:
            cols = [c for c in spec_df.columns if c.startswith(montage)]
            data = spec_df[cols].values
            channels.append(data)
        
        spec = np.stack(channels, axis=-1).astype(np.float32)
        spec = np.nan_to_num(spec, 0.0)
        
        spec_norm = np.zeros_like(spec)
        for i in range(4):
            ch = spec[:, :, i]
            min_val, max_val = ch.min(), ch.max()
            if max_val > min_val:
                spec_norm[:, :, i] = (ch - min_val) / (max_val - min_val)
        
        if self.mode == 'train' and np.random.rand() < 0.5:
            spec_norm = np.fliplr(spec_norm)
        
        spec_resized = cv2.resize(spec_norm, (config.IMG_SIZE, config.IMG_SIZE))
        spec_tensor = torch.from_numpy(spec_resized).permute(2, 0, 1)
        spec_tensor = (spec_tensor - 0.5) / 0.5
        
        target = torch.tensor(row['target_probs'], dtype=torch.float32)
        
        return spec_tensor, target

train_dataset = NeuralPatternsDataset(train_df[train_df['fold']==0], mode='train')
valid_dataset = NeuralPatternsDataset(train_df[train_df['fold']==1], mode='valid')

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                          shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE,
                          shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

print(f"âœ“ Datasets created")

# ============================================================================
# Model
# ============================================================================
print("\n[6/11] Building model...")

class EfficientNetModel(nn.Module):
    def __init__(self, model_name='tf_efficientnetv2_m', pretrained=True):
        super().__init__()
        
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       num_classes=config.NUM_CLASSES, in_chans=3)
        
        original_conv = self.model.conv_stem
        self.model.conv_stem = nn.Conv2d(4, original_conv.out_channels,
                                         kernel_size=original_conv.kernel_size,
                                         stride=original_conv.stride,
                                         padding=original_conv.padding, bias=False)
        
        with torch.no_grad():
            weight = original_conv.weight
            new_weight = weight.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
            self.model.conv_stem.weight = nn.Parameter(new_weight)
        
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(num_features, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, config.NUM_CLASSES)
        )
    
    def forward(self, x):
        x = self.model(x)
        return F.softmax(x, dim=1)

# ============================================================================
# Load Checkpoint if Available
# ============================================================================
print("\n[7/11] Loading model and checkpoint...")

checkpoint_files = list(config.OUTPUT_DIR.glob('checkpoint_epoch_*.pth'))

if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
    start_epoch = int(latest_checkpoint.stem.split('_')[-1])
    
    print(f"âœ“ Resuming from: {latest_checkpoint.name}")
    
    checkpoint = torch.load(latest_checkpoint, map_location=config.DEVICE)
    
    model = EfficientNetModel(config.MODEL_NAME, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    
    class KLDivLoss(nn.Module):
        def forward(self, pred, target):
            eps = 1e-7
            pred = torch.clamp(pred, eps, 1.0)
            target = torch.clamp(target, eps, 1.0)
            return torch.mean(torch.sum(target * torch.log(target / pred), dim=1))
    
    criterion = KLDivLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                   weight_decay=config.WEIGHT_DECAY)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                             T_max=config.TOTAL_EPOCHS, 
                                                             eta_min=config.MIN_LR)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    best_val_loss = checkpoint['best_val_loss']
    history = checkpoint['history']
    
    print(f"  Epoch: {start_epoch}/{config.TOTAL_EPOCHS}")
    print(f"  Best accuracy: {max(history['val_acc'])*100:.2f}%")
    
else:
    print("âœ“ Starting fresh training")
    
    start_epoch = 0
    
    model = EfficientNetModel(config.MODEL_NAME, pretrained=True)
    model = model.to(config.DEVICE)
    
    class KLDivLoss(nn.Module):
        def forward(self, pred, target):
            eps = 1e-7
            pred = torch.clamp(pred, eps, 1.0)
            target = torch.clamp(target, eps, 1.0)
            return torch.mean(torch.sum(target * torch.log(target / pred), dim=1))
    
    criterion = KLDivLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                   weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                             T_max=config.TOTAL_EPOCHS,
                                                             eta_min=config.MIN_LR)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}

scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)

total_params = sum(p.numel() for p in model.parameters())
print(f"âœ“ Model ready ({total_params/1e6:.1f}M parameters)")

# ============================================================================
# Training Functions
# ============================================================================
print("\n[8/11] Preparing training...")

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc='Training')
    
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Validation', leave=False):
            images, targets = images.to(device), targets.to(device)
            
            with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    loss = running_loss / len(loader)
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    pred_labels = preds.argmax(axis=1)
    true_labels = targets.argmax(axis=1)
    accuracy = (pred_labels == true_labels).mean()
    
    return loss, accuracy

print("âœ“ Training ready")

# ============================================================================
# Training Loop
# ============================================================================
print("\n[9/11] Starting training...")
print("=" * 80)
print(f"Training epochs {start_epoch + 1} to {min(start_epoch + config.EPOCHS_PER_RUN, config.TOTAL_EPOCHS)}")
print("=" * 80)

patience_counter = 0
start_time = datetime.now()

for epoch in range(start_epoch, min(start_epoch + config.EPOCHS_PER_RUN, config.TOTAL_EPOCHS)):
    print(f"\n{'='*80}")
    print(f"Epoch {epoch+1}/{config.TOTAL_EPOCHS}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    print('='*80)
    
    train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, config.DEVICE)
    val_loss, val_acc = validate(model, valid_loader, criterion, config.DEVICE)
    scheduler.step()
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(optimizer.param_groups[0]['lr'])
    
    print(f"\n{'='*80}")
    print(f"Results - Epoch {epoch+1}/{config.TOTAL_EPOCHS}")
    print(f"{'='*80}")
    print(f"  Train Loss:      {train_loss:.4f}")
    print(f"  Val Loss:        {val_loss:.4f}")
    print(f"  Val Accuracy:    {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Best Val Loss:   {best_val_loss:.4f}")
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'history': history,
    }
    
    checkpoint_path = config.OUTPUT_DIR / f'checkpoint_epoch_{epoch+1}.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"  ðŸ’¾ Checkpoint saved: {checkpoint_path.name}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_path = config.OUTPUT_DIR / 'best_model.pth'
        torch.save(model.state_dict(), best_model_path)
        print(f"  ðŸŽ‰ Best model saved! (Accuracy: {val_acc*100:.2f}%)")
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{config.PATIENCE})")
    
    # Save history
    history_path = config.OUTPUT_DIR / 'training_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

# ============================================================================
# Create Download Links
# ============================================================================
print("\n[10/11] Creating download links...")

from IPython.display import FileLink, display, HTML

completed_epochs = len(history['train_loss'])
latest_checkpoint = config.OUTPUT_DIR / f'checkpoint_epoch_{completed_epochs}.pth'

print("\n" + "=" * 80)
print("ðŸ“¥ DOWNLOAD THESE FILES (Click the links below):")
print("=" * 80)

if latest_checkpoint.exists():
    print(f"\n1. Latest Checkpoint (Epoch {completed_epochs}):")
    display(FileLink(str(latest_checkpoint)))

best_model_path = config.OUTPUT_DIR / 'best_model.pth'
if best_model_path.exists():
    print(f"\n2. Best Model:")
    display(FileLink(str(best_model_path)))

history_path = config.OUTPUT_DIR / 'training_history.pkl'
if history_path.exists():
    print(f"\n3. Training History:")
    display(FileLink(str(history_path)))

print("\n" + "=" * 80)
print("ðŸ’¡ IMPORTANT: Click and download ALL files above!")
print("=" * 80)

# ============================================================================
# Summary & Instructions
# ============================================================================
print("\n[11/11] Session summary...")
print("=" * 80)
print("SESSION COMPLETE!")
print("=" * 80)

remaining_epochs = config.TOTAL_EPOCHS - completed_epochs
elapsed_time = (datetime.now() - start_time).total_seconds() / 3600

print(f"\nðŸ“Š Progress:")
print(f"  Completed: {completed_epochs}/{config.TOTAL_EPOCHS} epochs")
print(f"  Best accuracy: {max(history['val_acc'])*100:.2f}%")
print(f"  Latest accuracy: {history['val_acc'][-1]*100:.2f}%")
print(f"  Session time: {elapsed_time:.1f}h")

if remaining_epochs > 0:
    sessions_needed = (remaining_epochs + config.EPOCHS_PER_RUN - 1) // config.EPOCHS_PER_RUN
    
    print(f"\n" + "=" * 80)
    print("ðŸ“‹ TO CONTINUE IN NEXT SESSION:")
    print("=" * 80)
    print(f"\n1. Download the checkpoint file (link above)")
    print(f"2. Create a NEW Kaggle Dataset:")
    print(f"   - Go to: https://www.kaggle.com/datasets")
    print(f"   - Click 'New Dataset'")
    print(f"   - Upload the checkpoint file")
    print(f"   - Name it: 'neural-patterns-checkpoint-{completed_epochs}'")
    print(f"   - Make it public")
    print(f"\n3. In your NEXT notebook:")
    print(f"   - Add that dataset as input")
    print(f"   - Run this cell again")
    print(f"   - It will auto-resume from epoch {completed_epochs}!")
    print(f"\n4. Repeat {sessions_needed} more times to reach 20 epochs")
else:
    print(f"\nðŸŽ‰ TRAINING COMPLETE!")
    print(f"  Final accuracy: {history['val_acc'][-1]*100:.2f}%")
    print(f"  Best accuracy: {max(history['val_acc'])*100:.2f}%")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['train_loss'], label='Train', marker='o')
axes[0].plot(history['val_loss'], label='Val', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Detrimental Neural Patterns - Training Progress')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot([x*100 for x in history['val_acc']], marker='o', color='green')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title(f'Validation Accuracy (Best: {max(history["val_acc"])*100:.1f}%)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = config.OUTPUT_DIR / 'training_progress.png'
plt.savefig(plot_path, dpi=100)
plt.show()

print(f"\n4. Training plot:")
display(FileLink(str(plot_path)))

gc.collect()
torch.cuda.empty_cache()

print("\n" + "=" * 80)
print("âœ… Session Complete!")
print("   Detrimental Neural Patterns Classification")
print("=" * 80)


# Quick download before session ends
import os
from pathlib import Path

# List files
print("Files in /kaggle/working:")
files = list(Path('/kaggle/working').glob('*.pth')) + list(Path('/kaggle/working').glob('*.pkl'))
for f in files:
    print(f"  âœ“ {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

# For Kaggle, files auto-save as "output files"
# They'll be in the version's output when you commit
print("\nðŸ’¡ TIP: Files are saved in this session's output")
print("   They'll be available after you 'Save Version'")

