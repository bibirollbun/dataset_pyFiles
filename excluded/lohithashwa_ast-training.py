# BirdCLEF 2025: AST Training Notebook - Modified for Full Training
# =============================================================

# Install necessary packages
!pip install -q librosa==0.10.1 torchaudio timm soundfile audiomentations

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from timm.models.layers import to_2tuple, trunc_normal_
import matplotlib.pyplot as plt
import random
import warnings
import time
from tqdm.auto import tqdm
import librosa
import cv2
import json

# Silence warnings
warnings.filterwarnings("ignore")
print("Torch version:", torch.__version__)
print("Timm version:", timm.__version__)

# Set random seed for reproducibility
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
set_seed()

# Configuration
class CFG:
    # Paths
    train_audio_dir = '/kaggle/input/birdclef-2025/train_audio'
    train_csv = '/kaggle/input/birdclef-2025/train.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    output_dir = '/kaggle/working/ast_model'
    
    # Audio parameters
    sample_rate = 32000
    duration = 5  # seconds per clip
    
    # Mel spectrogram parameters
    n_mels = 128
    n_fft = 1024
    hop_length = 512
    fmin = 50
    fmax = 14000
    
    # AST model parameters
    fstride = 10
    tstride = 10
    patch_size = 16
    model_size = 'base224'  # Changed from base384 to base224 for better compatibility
    
    # Vision Transformer expected input size
    target_height = 224
    target_width = 224
    
    # Training parameters
    batch_size = 16
    epochs = 10
    lr = 1e-4
    weight_decay = 1e-6
    
    # Augmentation parameters
    freqm = 24  # Frequency mask max length
    timem = 96  # Time mask max length
    mixup_alpha = 0.5
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Debug mode - set to True to train on a small subset of data
    debug = False

if CFG.debug:
    CFG.epochs = 2
    CFG.batch_size = 8

os.makedirs(CFG.output_dir, exist_ok=True)
print(f"Using device: {CFG.device}")

# Load data
print("Loading data...")
df = pd.read_csv(CFG.train_csv)
taxonomy_df = pd.read_csv(CFG.taxonomy_csv)

# Add filepath column
df['filepath'] = df['filename'].apply(lambda x: os.path.join(CFG.train_audio_dir, x))

# Create a dictionary mapping from primary_label to index
# Important: We'll use it consistently for both training and inference
unique_labels = taxonomy_df['primary_label'].unique()
label_map = {label: idx for idx, label in enumerate(unique_labels)}
num_classes = len(label_map)
print(f"Number of classes: {num_classes}")

# Add label_idx column - make sure to use our consistent label mapping
df['label_idx'] = df['primary_label'].map(label_map)

# Check for any NaN values in label_idx column
if df['label_idx'].isna().any():
    print(f"WARNING: Found {df['label_idx'].isna().sum()} rows with NaN label_idx")
    print("Example rows with missing labels:")
    print(df[df['label_idx'].isna()].head())
    
    # Remove rows with NaN label_idx
    df = df.dropna(subset=['label_idx']).reset_index(drop=True)
    print(f"Removed rows with missing labels. New shape: {df.shape}")

# Convert label_idx to integer
df['label_idx'] = df['label_idx'].astype(int)

# For debug mode, use smaller dataset
if CFG.debug:
    df = df.sample(min(500, len(df)), random_state=SEED).reset_index(drop=True)

print(f"Training data shape: {df.shape}")
print(df.head())

# Define AST model
class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ASTModel(nn.Module):
    """Audio Spectrogram Transformer model"""
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
                 imagenet_pretrain=True, model_size='base224'):
        super(ASTModel, self).__init__()
        
        # Override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        
        # Print model configuration
        print(f'AST Model: size={model_size}, input_fdim={input_fdim}, input_tdim={input_tdim}')
        print(f'frequency stride={fstride}, time stride={tstride}')
        
        # Load model - first check what models are available
        available_models = timm.list_models('*vit*')
        print(f"Available models containing 'vit': {available_models[:5]} and {len(available_models)} more...")
        
        # Try to use a model that's available
        if model_size == 'tiny224':
            try:
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            except RuntimeError:
                # Fallback to a model that should be available
                print("Falling back to vit_tiny_patch16_224")
                self.v = timm.create_model('vit_tiny_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'small224':
            try:
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            except RuntimeError:
                print("Falling back to vit_small_patch16_224")
                self.v = timm.create_model('vit_small_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'base224':
            try:
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            except RuntimeError:
                print("Falling back to vit_base_patch16_224")
                self.v = timm.create_model('vit_base_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'base384':
            try:
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            except RuntimeError:
                print("Falling back to vit_base_patch16_384")
                try:
                    self.v = timm.create_model('vit_base_patch16_384', pretrained=imagenet_pretrain)
                except RuntimeError:
                    print("Falling back to vit_base_patch16_224")
                    self.v = timm.create_model('vit_base_patch16_224', pretrained=imagenet_pretrain)
        else:
            raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            
        # Check if model has distillation token
        self.has_dist_token = hasattr(self.v, 'dist_token')
        print(f"Model has distillation token: {self.has_dist_token}")
        
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), 
                                     nn.Linear(self.original_embedding_dim, label_dim))
        
        # Get shape automatically
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        
        print(f'number of patches={num_patches}')
            
        # Linear projection
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        if imagenet_pretrain:
            new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj
        
        # Positional embedding
        if imagenet_pretrain:
            # Get the positional embedding from model
            if self.has_dist_token:
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
            else:
                new_pos_embed = self.v.pos_embed[:, 1:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
            
            # Cut or interpolate position embedding
            if t_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                
            # Cut or interpolate position embedding
            if f_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                
            # Flatten the position embedding
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
            
            # Concatenate with cls token and distillation token
            if self.has_dist_token:
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :1, :].detach(), new_pos_embed], dim=1))
        else:
            # Random initialization
            if self.has_dist_token:
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
            else:
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 1, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)
        
    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
    
    def forward(self, x):
        """
        :param x: Input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins)
        :return: prediction
        """
        # Input shape: (batch_size, time_frame_num, frequency_bins)
        x = x.unsqueeze(1)        # Add channel dimension: (B, 1, T, F)
        x = x.transpose(2, 3)     # -> (B, 1, F, T)
        
        B = x.shape[0]
        x = self.v.patch_embed(x)
        
        # Handle both model types (with and without distillation token)
        if self.has_dist_token:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        for blk in self.v.blocks:
            x = blk(x)
            
        x = self.v.norm(x)
        
        # Handle both model types for output
        if self.has_dist_token:
            x = (x[:, 0] + x[:, 1]) / 2  # Average of cls and dist token
        else:
            x = x[:, 0]  # Just use cls token
        
        x = self.mlp_head(x)
        return x

# Audio processing functions
def audio_to_melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    # Handle NaN values
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)
    
    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm

def process_audio_file(audio_path, cfg):
    """Process a single audio file to get the mel spectrogram"""
    try:
        # Load audio
        audio_data, _ = librosa.load(audio_path, sr=cfg.sample_rate)
        
        # Calculate target length in samples
        target_length = int(cfg.duration * cfg.sample_rate)
        
        # Handle audio shorter than target duration
        if len(audio_data) < target_length:
            # Pad with zeros
            audio_data = np.pad(audio_data, 
                             (0, target_length - len(audio_data)),
                             mode='constant')
        
        # Take center segment if longer than target duration
        if len(audio_data) > target_length:
            start_idx = (len(audio_data) - target_length) // 2
            audio_data = audio_data[start_idx:start_idx + target_length]
        
        # Generate mel spectrogram
        mel_spec = audio_to_melspec(audio_data, cfg)
        
        return mel_spec.astype(np.float32)
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Resize spectrogram to match model's expected input dimensions
def resize_spectrogram(spec, target_height=224, target_width=224):
    """Resize a spectrogram to match the expected input dimensions of the model"""
    # Convert to numpy if it's a tensor
    if isinstance(spec, torch.Tensor):
        spec_np = spec.numpy()
    else:
        spec_np = spec
    
    # Ensure the input is 2D
    if spec_np.ndim != 2:
        print(f"Warning: Expected 2D spectrogram, got shape {spec_np.shape}")
        if spec_np.ndim == 3 and spec_np.shape[0] == 1:
            spec_np = spec_np.squeeze(0)  # Remove singleton dimension
    
    # Resize using OpenCV
    resized_spec = cv2.resize(spec_np, (target_width, target_height))
    
    # Convert back to tensor if input was a tensor
    if isinstance(spec, torch.Tensor):
        return torch.tensor(resized_spec, dtype=torch.float32)
    else:
        return resized_spec.astype(np.float32)

# Dataset class
class BirdCLEFDataset(Dataset):
    def __init__(self, df, cfg, transform=None):
        self.df = df
        self.cfg = cfg
        self.transform = transform
        self.num_classes = num_classes  # Use the global num_classes value
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get audio file path and label
        audio_path = row['filepath']
        label_idx = row['label_idx']
        
        # Process audio file
        spec = process_audio_file(audio_path, self.cfg)
        
        if spec is None:
            # Return zeros if processing failed
            spec = np.zeros((self.cfg.n_mels, int(self.cfg.duration * self.cfg.sample_rate / self.cfg.hop_length) + 1), dtype=np.float32)
        
        # Resize spectrogram to match model's expected input dimensions
        spec = resize_spectrogram(spec, target_height=self.cfg.target_height, target_width=self.cfg.target_width)
        
        # Convert to tensor
        spec_tensor = torch.tensor(spec, dtype=torch.float32)
        
        # Apply transformations if any
        if self.transform:
            spec_tensor = self.transform(spec_tensor)
            
        # Create one-hot encoded label tensor
        label_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
        label_tensor[label_idx] = 1.0
        
        # Add secondary labels if they exist
        if 'secondary_labels' in row and row['secondary_labels'] not in ['[]', '', None]:
            if isinstance(row['secondary_labels'], str):
                try:
                    secondary_labels = eval(row['secondary_labels'])
                    for label in secondary_labels:
                        if label in label_map:
                            sec_idx = label_map[label]
                            if 0 <= sec_idx < self.num_classes:  # Ensure the index is valid
                                label_tensor[sec_idx] = 1.0
                except:
                    pass  # Skip if there's an error parsing secondary labels
        
        return {
            'spectrogram': spec_tensor,
            'label': label_tensor
        }

# Augmentation transforms
class SpecAugment:
    def __init__(self, freq_mask=24, time_mask=96):
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        
    def __call__(self, spec):
        # Apply frequency masking
        if self.freq_mask > 0:
            num_masks = random.randint(1, 2)
            for _ in range(num_masks):
                freq_mask_size = random.randint(1, self.freq_mask)
                freq_start = random.randint(0, spec.shape[0] - freq_mask_size)
                spec[freq_start:freq_start + freq_mask_size, :] = 0
        
        # Apply time masking
        if self.time_mask > 0:
            num_masks = random.randint(1, 2)
            for _ in range(num_masks):
                time_mask_size = random.randint(1, self.time_mask)
                time_start = random.randint(0, spec.shape[1] - time_mask_size)
                spec[:, time_start:time_start + time_mask_size] = 0
                
        # Apply random brightness
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = torch.clamp(gain * spec + bias, 0, 1)
            
        return spec

# Mixup function
def mixup_data(x, y, alpha=0.4):
    """Applies mixup augmentation to the batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y

# No validation split - use all data for training
print(f"Training on all data: {len(df)} samples")

# Create dataset and dataloader
train_transform = SpecAugment(freq_mask=CFG.freqm, time_mask=CFG.timem)
train_dataset = BirdCLEFDataset(df, CFG, transform=train_transform)

# Check the shape of a sample item to ensure it's properly resized
sample_item = train_dataset[0]
print(f"Sample spectrogram shape: {sample_item['spectrogram'].shape}")
print(f"Sample label shape: {sample_item['label'].shape}")

train_loader = DataLoader(
    train_dataset, 
    batch_size=CFG.batch_size, 
    shuffle=True, 
    num_workers=2, 
    pin_memory=True
)

print(f"Train dataloader: {len(train_loader)} batches")

# Initialize model
model = ASTModel(
    label_dim=num_classes,
    fstride=CFG.fstride,
    tstride=CFG.tstride,
    input_fdim=CFG.target_height,  # Use target dimensions that match what we're resizing to
    input_tdim=CFG.target_width,
    imagenet_pretrain=True,
    model_size=CFG.model_size
)

model = model.to(CFG.device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=CFG.lr/100)

# Training function
def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha=0.5):
    model.train()
    losses = []
    pbar = tqdm(loader, desc='Training')
    
    for batch in pbar:
        # Get data
        specs = batch['spectrogram'].to(device)
        labels = batch['label'].to(device)
        
        # Apply mixup with probability 0.5
        if mixup_alpha > 0 and random.random() < 0.5:
            specs, labels = mixup_data(specs, labels, alpha=mixup_alpha)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record loss
        losses.append(loss.item())
        
        # Update progress bar
        pbar.set_postfix({'loss': np.mean(losses[-10:])})
    
    return np.mean(losses)

# Training loop
history = {'train_loss': []}

# Save initial model configuration
model_config = {}
for k, v in CFG.__dict__.items():
    if not k.startswith('__') and not callable(v):
        # Convert non-serializable objects to strings
        if k == 'device':
            model_config[k] = str(v)
        else:
            model_config[k] = v

with open(os.path.join(CFG.output_dir, 'model_config.json'), 'w') as f:
    json.dump(model_config, f)

for epoch in range(1, CFG.epochs + 1):
    print(f"\nEpoch {epoch}/{CFG.epochs}")
    
    # Train
    train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=CFG.device,
        mixup_alpha=CFG.mixup_alpha
    )
    
    # Update learning rate
    scheduler.step()
    
    # Update history
    history['train_loss'].append(train_loss)
    
    # Print metrics
    print(f"Train Loss: {train_loss:.4f}")
    print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Save checkpoint every epoch
    checkpoint_path = os.path.join(CFG.output_dir, f'model_epoch_{epoch}.pth')
    
    # Create a serializable config dictionary
    config_dict = {}
    for k, v in CFG.__dict__.items():
        if not k.startswith('__') and not callable(v):
            # Convert non-serializable objects to strings
            if k == 'device':
                config_dict[k] = str(v)
            else:
                config_dict[k] = v
                
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'num_classes': num_classes,
        'label_map': label_map,
        'config': config_dict
    }, checkpoint_path)
    
    print(f"Saved checkpoint to {checkpoint_path}")

# Save final model
config_dict = {}
for k, v in CFG.__dict__.items():
    if not k.startswith('__') and not callable(v):
        # Convert non-serializable objects to strings
        if k == 'device':
            config_dict[k] = str(v)
        else:
            config_dict[k] = v
            
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': CFG.epochs,
    'num_classes': num_classes,
    'label_map': label_map,
    'config': config_dict
}, os.path.join(CFG.output_dir, 'final_model.pth'))

# Save training history
with open(os.path.join(CFG.output_dir, 'history.json'), 'w') as f:
    json.dump(history, f)
    
print(f"Training complete!")

# Plot training history
plt.figure(figsize=(8, 4))
plt.plot(history['train_loss'], label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(CFG.output_dir, 'training_history.png'))
plt.show()

print(f"Model saved to {CFG.output_dir}")




