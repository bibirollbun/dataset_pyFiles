import os
import gc
import time
import math
import random
import logging
import warnings
import numpy as np
import pandas as pd
import cv2
import librosa
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

# Suppress warnings and configure logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# Configuration Class
class CFG:
    seed = 42
    
    # Set to True for quick debugging (fewer epochs, small subset)
    debug = False  
    
    # Input Paths (Kaggle Input is Read-Only)
    train_datadir = '/kaggle/input/birdclef-2025/train_audio'
    train_csv = '/kaggle/input/birdclef-2025/train.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    
    # Path to pre-computed spectrograms (otherwise use CPU to compute would be very slow)
    spectrogram_npy = '/kaggle/input/birdclef25-mel-spectrograms/birdclef2025_melspec_5sec_256_256.npy'
    
    # Output Path (Must be in /kaggle/working/ to save files)
    output_dir = '/kaggle/working/model' 
    
    # Audio & Spectrogram Parameters
    FS = 32000              # Sampling Rate
    TARGET_DURATION = 5.0   # Duration in seconds
    TARGET_SHAPE = (256, 256) # Image Shape (H, W)
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    
    # Model Hyperparameters
    model_name = 'efficientnet_b0'
    pretrained = True
    in_channels = 1         # 1 for grayscale (spectrogram), 3 for RGB
    num_classes = 0         # Updated dynamically based on taxonomy
    
    # Training Hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10
    batch_size = 32
    num_workers = 2
    
    # Optimizer & Scheduler
    lr = 5e-4
    weight_decay = 1e-5
    min_lr = 1e-6
    
    # Validation Strategy (Single Split)
    val_pct = 0.2  # 20% for Validation, 80% for Training
    
    # Data Augmentation
    aug_prob = 0.5
    mixup_alpha = 0.5       # Mixup strength
    
    # Data Loading Strategy
    LOAD_DATA = True        # Attempt to load .npy file first
    
    def update_debug_settings(self):
        """Adjusts settings for quick debugging"""
        if self.debug:
            print("DEBUG MODE ENABLED: Running fewer epochs on a subset.")
            self.epochs = 2

cfg = CFG()


# Utilities
def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)


# Pre-processing Functions
def audio2melspec(audio_data, cfg):
    """
    Converts raw audio waveform to a Log-Mel Spectrogram.
    Returns a normalized numpy array.
    """
    # Handle NaNs
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    # Compute Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0
    )

    # Convert to dB (Log scale)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Min-Max Normalization to [0, 1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm

def process_audio_file(audio_path, cfg):
    """
    Loads an audio file, crops/pads it to 5 seconds, and returns the spectrogram.
    """
    try:
        # Load audio
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
        target_samples = int(cfg.TARGET_DURATION * cfg.FS)

        # Padding if too short
        if len(audio_data) < target_samples:
            n_copy = math.ceil(target_samples / len(audio_data))
            if n_copy > 1:
                audio_data = np.concatenate([audio_data] * n_copy)

        # Crop the center 5 seconds
        start_idx = max(0, int(len(audio_data) / 2 - target_samples / 2))
        end_idx = min(len(audio_data), start_idx + target_samples)
        center_audio = audio_data[start_idx:end_idx]

        # Final check for padding
        if len(center_audio) < target_samples:
            center_audio = np.pad(center_audio, (0, target_samples - len(center_audio)), mode='constant')

        # Convert to Spectrogram
        mel_spec = audio2melspec(center_audio, cfg)
        
        # Resize if necessary (ensure consistent input size)
        if mel_spec.shape != cfg.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

        return mel_spec.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


# Dataset Class
class BirdCLEFDataset(Dataset):
    def __init__(self, df, cfg, spectrograms=None, mode="train"):
        self.df = df
        self.cfg = cfg
        self.mode = mode
        self.spectrograms = spectrograms # Pre-loaded .npy dictionary
        
        # Load Taxonomy to map labels to integers
        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        # Prepare file paths
        if 'filepath' not in self.df.columns:
            self.df['filepath'] = self.cfg.train_datadir + '/' + self.df.filename
        
        # Create 'samplename' key for dictionary lookup
        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
        
        # Debugging: subset data
        if cfg.debug and len(self.df) > 1000:
            self.df = self.df.sample(1000, random_state=cfg.seed).reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        spec = None

        # Strategy 1: Load from RAM (Fastest)
        if self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
        # Strategy 2: Load from Disk (Slower, processes audio on the fly)
        elif not self.cfg.LOAD_DATA:
            spec = process_audio_file(row['filepath'], self.cfg)

        # Fallback for missing data
        if spec is None:
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            
        # Add Channel Dimension: (H, W) -> (1, H, W)
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

        # Apply Augmentations (Only during training)
        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec = self.apply_spec_augmentations(spec)
        
        # Encode Labels (One-Hot Encoding for Multi-Label)
        target = self.encode_label(row['primary_label'])
        
        # Handle Secondary Labels (Background birds)
        if 'secondary_labels' in row and isinstance(row['secondary_labels'], str):
            try:
                secondary_labels = eval(row['secondary_labels'])
                for label in secondary_labels:
                    if label in self.label_to_idx:
                        target[self.label_to_idx[label]] = 1.0
            except:
                pass
        
        return {
            'melspec': spec, 
            'target': torch.tensor(target, dtype=torch.float32)
        }
    
    def apply_spec_augmentations(self, spec):
        """Applies SpecAugment: Time masking and Frequency masking"""
        # Time masking
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                width = random.randint(5, 20)
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start:start+width] = 0
        
        # Frequency masking
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                height = random.randint(5, 20)
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start:start+height, :] = 0
            
        return spec
    
    def encode_label(self, label):
        """Creates a One-Hot vector for the target label"""
        target = np.zeros(self.num_classes)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target

def collate_fn(batch):
    """Custom collate function to handle batches."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return {}
    
    melspecs = torch.stack([item['melspec'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    
    return {'melspec': melspecs, 'target': targets}


# Model Architecture
class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Load Backbone using timm (Transfer Learning)
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            drop_rate=0.2,
            drop_path_rate=0.2
        )
        
        # Replace the original classification head
        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            backbone_out = self.backbone.num_features
            self.backbone.reset_classifier(0, '')
        
        # Global Average Pooling
        self.pooling = nn.AdaptiveAvgPool2d(1)
        
        # New Classification Head
        self.classifier = nn.Linear(backbone_out, cfg.num_classes)
        
        # Mixup Settings
        self.mixup_enabled = hasattr(cfg, 'mixup_alpha') and cfg.mixup_alpha > 0
            
    def forward(self, x, targets=None):
        # Apply Mixup during training
        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None
        
        # Feature Extraction
        features = self.backbone(x)
        
        # Handle different output formats from timm
        if isinstance(features, dict):
            features = features['features']
        
        # Pooling (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        # Classification
        logits = self.classifier(features)
        
        # Return loss directly if using Mixup
        if self.training and self.mixup_enabled and targets is not None:
            loss = self.mixup_criterion(nn.BCEWithLogitsLoss(), logits, targets_a, targets_b, lam)
            return logits, loss
            
        return logits
    
    def mixup_data(self, x, targets):
        """Mixes two images and their labels."""
        batch_size = x.size(0)
        lam = np.random.beta(self.cfg.mixup_alpha, self.cfg.mixup_alpha)
        indices = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[indices]
        return mixed_x, targets, targets[indices], lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Weighted loss for mixed samples."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Training Engine
def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    losses = []
    all_targets = []
    all_outputs = []
    
    pbar = tqdm(loader, desc="Training")
    
    for batch in pbar:
        inputs = batch['melspec'].to(device)
        targets = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (handles mixup internally)
        outputs = model(inputs, targets)
        
        # Calculate loss
        if isinstance(outputs, tuple):
            outputs, loss = outputs # Mixup returns (logits, loss)
        else:
            loss = criterion(outputs, targets)
            
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
            
        losses.append(loss.item())
        
        # Store for AUC calculation
        all_outputs.append(outputs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        
        pbar.set_postfix({'loss': np.mean(losses[-10:])})
    
    # Calculate Metrics
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    
    return np.mean(losses), auc

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []
    
    for batch in tqdm(loader, desc="Validation"):
        inputs = batch['melspec'].to(device)
        targets = batch['target'].to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        losses.append(loss.item())
        all_outputs.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    
    return np.mean(losses), auc

def calculate_auc(targets, outputs):
    """Calculates Mean ROC-AUC score across all classes."""
    num_classes = targets.shape[1]
    aucs = []
    probs = 1 / (1 + np.exp(-outputs)) # Sigmoid
    
    for i in range(num_classes):
        # Only calculate AUC if the class is present in the batch
        if np.sum(targets[:, i]) > 0:
            try:
                class_auc = roc_auc_score(targets[:, i], probs[:, i])
                aucs.append(class_auc)
            except:
                pass
    return np.mean(aucs) if aucs else 0.0


# Main Execution Loop (Single Split)
def run_training(df, cfg):
    # 0. Ensure Output Directory Exists
    if not os.path.exists(cfg.output_dir):
        print(f"Creating output directory: {cfg.output_dir}")
        os.makedirs(cfg.output_dir, exist_ok=True)

    # Setup Data Metadata
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    cfg.num_classes = len(taxonomy_df)
    
    if cfg.debug:
        cfg.update_debug_settings()

    # Try to load pre-computed spectrograms (Speed optimization)
    spectrograms = None
    if cfg.LOAD_DATA:
        print("Attempting to load pre-computed mel spectrograms...")
        try:
            spectrograms = np.load(cfg.spectrogram_npy, allow_pickle=True).item()
            print(f"Successfully loaded {len(spectrograms)} spectrograms.")
        except Exception as e:
            print(f"Failed to load NPY file: {e}")
            print("Switching to on-the-fly generation (Slower).")
            cfg.LOAD_DATA = False
    
    print(f'\n{"="*30} Training Start (Single Split) {"="*30}')
    
    # Create Train/Val Split
    # We use stratified split to ensure all bird classes are represented in validation
    train_df, val_df = train_test_split(
        df, 
        test_size=cfg.val_pct, 
        random_state=cfg.seed, 
        stratify=df['primary_label'] 
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    print(f"Train Size: {len(train_df)} | Val Size: {len(val_df)}")
    
    # 4. Create Datasets & Loaders
    train_dataset = BirdCLEFDataset(train_df, cfg, spectrograms, mode='train')
    val_dataset = BirdCLEFDataset(val_df, cfg, spectrograms, mode='valid')
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, 
                              num_workers=cfg.num_workers, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, 
                            num_workers=cfg.num_workers, collate_fn=collate_fn)
    
    # Initialize Model & Optimizer
    model = BirdCLEFModel(cfg).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    # Scheduler (OneCycleLR for faster convergence)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, steps_per_epoch=len(train_loader), 
        epochs=cfg.epochs, pct_start=0.1
    )
    
    # Training Loop
    best_auc = 0
    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch+1}/{cfg.epochs}")
        
        train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device, scheduler)
        val_loss, val_auc = validate(model, val_loader, criterion, cfg.device)
        
        print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val AUC:   {val_auc:.4f}")
        
        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join(cfg.output_dir, "best_model.pth")
            torch.save({'model_state_dict': model.state_dict(), 'cfg': cfg}, save_path)
            print(f"--> Saved Best Model to {save_path} (AUC: {best_auc:.4f})")
    
    # Cleanup
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    print(f"Starting Training on {cfg.device}...")
    
    # Load Metadata
    train_df = pd.read_csv(cfg.train_csv)
    
    # Run Training
    run_training(train_df, cfg)

