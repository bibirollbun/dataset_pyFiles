import os
import logging
import random
import gc
import time
import cv2
import math
import warnings
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import timm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
class CFG:
    
    seed = 42
    debug = True  
    apex = False
    print_freq = 100
    num_workers = 4  # Increased from 2
    
    # Detect environment
    # Check if we're in Kaggle
    if os.path.exists('/kaggle/input'):
        print("Running in Kaggle environment")
        is_kaggle = True
        BASE_PATH = '/kaggle/input/birdclef-2025'
    else:
        print("Running in local environment")
        is_kaggle = False
        # Look for the data in the current directory or parent directory
        if os.path.exists('./train.csv'):
            BASE_PATH = '.'
        elif os.path.exists('../train.csv'):
            BASE_PATH = '..'
        else:
            BASE_PATH = './data'  # Default fallback
    
    OUTPUT_DIR = '/kaggle/working/' if is_kaggle else './outputs'
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_datadir = f'{BASE_PATH}/train_audio'
    train_csv = f'{BASE_PATH}/train.csv'
    test_soundscapes = f'{BASE_PATH}/test_soundscapes'
    submission_csv = f'{BASE_PATH}/sample_submission.csv'
    taxonomy_csv = f'{BASE_PATH}/taxonomy.csv'
    
    spectrogram_npy = '/kaggle/input/birdclef25-mel-spectrograms/birdclef2025_melspec_5sec_256_256.npy' if is_kaggle else None
    
    model_name = 'efficientnetv2_s'  # Changed from mobilenetv3_small_100
    pretrained = False  # Changed to False for Kaggle (offline usage)
    pretrained_weights = None  # Path to local weights file, set this if you have downloaded weights
    in_channels = 1
    
    LOAD_DATA = True  
    USE_AMP = True  # Enable mixed precision
    PIN_MEMORY = True  # Pin memory for faster data loading
    
    FS = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)
    
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 30  # Increased from 10
    batch_size = 32  # Adjusted for EfficientNetV2-S
    gradient_accumulation_steps = 2  # Adjusted for EfficientNetV2-S
    criterion = 'BCEWithLogitsLoss'

    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]   

    optimizer = 'AdamW'
    lr = 1e-3  # Adjusted for EfficientNetV2-S
    weight_decay = 1e-2  # Adjusted for EfficientNetV2-S
  
    scheduler = 'CosineAnnealingWarmRestarts'  # Changed from CosineAnnealingLR
    min_lr = 1e-6
    T_0 = 5  # For CosineAnnealingWarmRestarts
    T_mult = 1  # For CosineAnnealingWarmRestarts

    aug_prob = 0.7  # Increased from 0.5
    mixup_alpha = 0.4
    cutmix_alpha = 0.4  # Added cutmix
    
    def update_debug_settings(self):
        if self.debug:
            self.epochs = 30
            self.selected_folds = [0]

cfg = CFG()
def set_seed(seed=42):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)

# Memory-efficient spectrograms loading function
def load_spectrograms_mmap(file_path):
    """Load spectrograms with optional memory mapping and chunking for large datasets"""
    if not os.path.exists(file_path):
        print(f"Warning: Spectrogram file not found at {file_path}")
        return {}
        
    try:
        print(f"Loading spectrograms from {file_path}")
        # First try to load with memory mapping
        return np.load(file_path, allow_pickle=True, mmap_mode='r').item()
    except ValueError as e:
        print(f"Memory mapping failed: {e}. Loading entire file into memory instead.")
        try:
            # Try loading everything at once
            data = np.load(file_path, allow_pickle=True).item()
            print(f"Successfully loaded {len(data)} spectrograms into memory")
            return data
        except MemoryError:
            # If we hit memory error, try to load in chunks
            print("Memory error when loading spectrograms. Attempting to load in chunks...")
            try:
                # Load just the metadata first
                data = {}
                with np.load(file_path, allow_pickle=True) as loaded:
                    # Process only a subset of data if in debug mode
                    if cfg.debug:
                        keys = list(loaded.item().keys())[:1000]
                        print(f"Debug mode: Loading only {len(keys)} spectrograms")
                        for k in keys:
                            data[k] = loaded.item()[k]
                    else:
                        # Try to process in smaller chunks
                        all_data = loaded.item()
                        keys = list(all_data.keys())
                        chunk_size = 1000
                        for i in range(0, len(keys), chunk_size):
                            chunk_keys = keys[i:i+chunk_size]
                            print(f"Loading chunk {i//chunk_size + 1}/{(len(keys)-1)//chunk_size + 1}...")
                            for k in chunk_keys:
                                data[k] = all_data[k]
                            # Force garbage collection
                            gc.collect()
                print(f"Successfully loaded {len(data)} spectrograms in chunks")
                return data
            except Exception as e2:
                print(f"Failed to load spectrograms even in chunks: {e2}")
                print("Continuing without pre-computed spectrograms")
                return {}

class BirdCLEFDatasetFromNPY(Dataset):
    def __init__(self, df, cfg, spectrograms=None, mode="train"):
        self.df = df
        self.cfg = cfg
        self.mode = mode
        self.spectrograms = spectrograms
        
        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        if 'filepath' not in self.df.columns:
            self.df['filepath'] = self.cfg.train_datadir + '/' + self.df.filename
        
        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

        self.sample_names = self.df['samplename'].values
        
        if cfg.debug:
            self.df = self.df.sample(min(1000, len(self.df)), random_state=cfg.seed).reset_index(drop=True)
            self.sample_names = self.df['samplename'].values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        
        # Faster lookup
        if self.spectrograms is not None:
            spec = self.spectrograms.get(samplename, None)
        else:
            spec = None

        if spec is None:
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)

        if isinstance(spec, np.memmap):
            spec = np.array(spec, dtype=np.float32)
            
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec = self.apply_spec_augmentations(spec)
        
        target = self.encode_label(row['primary_label'])
        
        if 'secondary_labels' in row and row['secondary_labels'] not in [[''], None, np.nan]:
            if isinstance(row['secondary_labels'], str):
                secondary_labels = eval(row['secondary_labels'])
            else:
                secondary_labels = row['secondary_labels']
            
            for label in secondary_labels:
                if label in self.label_to_idx:
                    target[self.label_to_idx[label]] = 1.0
        
        return {
            'melspec': spec, 
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': row['filename']
        }
    
    def apply_spec_augmentations(self, spec):
        """Apply enhanced augmentations to spectrogram"""
        # Time masking (horizontal stripes)
        if random.random() < 0.7:
            num_masks = random.randint(1, 5)  # Increased from 3
            for _ in range(num_masks):
                width = random.randint(5, 30)  # Increased max width from 20
                start = random.randint(0, spec.shape[2] - width)
                spec[:, :, start:start+width] = 0
        
        # Frequency masking (vertical stripes)
        if random.random() < 0.7:
            num_masks = random.randint(1, 5)  # Increased from 3
            for _ in range(num_masks):
                height = random.randint(5, 30)  # Increased max height from 20
                start = random.randint(0, spec.shape[1] - height)
                spec[:, start:start+height, :] = 0
        
        # Random brightness/contrast
        if random.random() < 0.7:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1)
            
        # Gaussian noise
        if random.random() < 0.5:
            noise = torch.randn_like(spec) * random.uniform(0.001, 0.02)
            spec = spec + noise
            spec = torch.clamp(spec, 0, 1)
            
        # Time shifting (roll horizontally)
        if random.random() < 0.5:
            shift = random.randint(-spec.shape[2]//8, spec.shape[2]//8)
            if shift != 0:
                spec = torch.roll(spec, shifts=shift, dims=2)
                
        # Frequency shifting (roll vertically)
        if random.random() < 0.3:
            shift = random.randint(-spec.shape[1]//20, spec.shape[1]//20)
            if shift != 0:
                spec = torch.roll(spec, shifts=shift, dims=1)
                
        return spec
    
    def encode_label(self, label):
        """Encode label to one-hot vector"""
        target = np.zeros(self.num_classes)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target

def collate_fn(batch):
    """Custom collate function to handle different sized spectrograms"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}
        
    result = {key: [] for key in batch[0].keys()}
    
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    
    for key in result:
        if key == 'target' and isinstance(result[key][0], torch.Tensor):
            result[key] = torch.stack(result[key])
        elif key == 'melspec' and isinstance(result[key][0], torch.Tensor):
            shapes = [t.shape for t in result[key]]
            if len(set(str(s) for s in shapes)) == 1:
                result[key] = torch.stack(result[key])
    
    return result

class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
        cfg.num_classes = len(taxonomy_df)
        
        # For Kaggle: create model with or without pretrained weights
        print(f"Creating model: {cfg.model_name}")
        try:
            self.backbone = timm.create_model(
                cfg.model_name,
                pretrained=cfg.pretrained,
                in_chans=cfg.in_channels,
                drop_rate=0.3,  # Adjusted for EfficientNetV2
                drop_path_rate=0.2  # Adjusted for EfficientNetV2
            )
            print(f"Successfully created {cfg.model_name}")
            # Print available methods and attributes for debugging
            print(f"Model structure: {type(self.backbone)}")
            if hasattr(self.backbone, 'classifier'):
                print(f"Classifier: {self.backbone.classifier}")
        except Exception as e:
            print(f"Error creating model: {e}")
            # Try alternative model name formats
            alternative_names = [
                'tf_efficientnetv2_s_in21k',  # Alternative name in newer timm
                'tf_efficientnetv2_s',    # TF variant
                'efficientnetv2_s_in21k'            # Simplified name
            ]
            for alt_name in alternative_names:
                try:
                    print(f"Trying alternative model name: {alt_name}")
                    self.backbone = timm.create_model(
                        alt_name,
                        pretrained=False,
                        in_chans=cfg.in_channels
                    )
                    # Update config to match successful model
                    cfg.model_name = alt_name
                    print(f"Successfully created {alt_name}")
                    break
                except Exception as e2:
                    print(f"Error with {alt_name}: {e2}")
        
        # Load pretrained weights from local file if specified
        if not cfg.pretrained and cfg.pretrained_weights:
            print(f"Loading pretrained weights from: {cfg.pretrained_weights}")
            try:
                state_dict = torch.load(cfg.pretrained_weights, map_location='cpu')
                # Handle case where state_dict might contain 'model' or 'state_dict' key
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # Remove prefix if it exists (like 'backbone.')
                if all(k.startswith('backbone.') for k in state_dict if k not in ['cls_token', 'pos_embed']):
                    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
                
                # Remove classifier weights
                for k in list(state_dict.keys()):
                    if 'classifier' in k or 'fc' in k or 'head' in k:
                        del state_dict[k]
                
                self.backbone.load_state_dict(state_dict, strict=False)
                print("Successfully loaded pretrained weights")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
        
        # Debug available classifier structures
        print(f"Available attributes: {dir(self.backbone)}")
        
        try:
            if 'efficientnet' in cfg.model_name:
                backbone_out = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Identity()
                print(f"Using EfficientNet classifier with {backbone_out} features")
            elif 'resnet' in cfg.model_name:
                backbone_out = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
                print(f"Using ResNet classifier with {backbone_out} features")
            elif 'mobilenetv3' in cfg.model_name:
                # MobileNetV3 classifier structure can vary between timm versions
                if hasattr(self.backbone, 'classifier') and hasattr(self.backbone.classifier, 'in_features'):
                    backbone_out = self.backbone.classifier.in_features
                    self.backbone.classifier = nn.Identity()
                    print(f"Using MobileNetV3 standard classifier with {backbone_out} features")
                elif hasattr(self.backbone, 'classifier') and isinstance(self.backbone.classifier, nn.Sequential):
                    # For MobileNetV3 with sequential classifier
                    backbone_out = 0  # Initialize before loop
                    for module in self.backbone.classifier:
                        if isinstance(module, nn.Linear):
                            backbone_out = module.in_features
                            break
                    if backbone_out == 0:
                        backbone_out = 1280  # Default for MobileNetV3 small
                    self.backbone.classifier = nn.Identity()
                    print(f"Using MobileNetV3 sequential classifier with {backbone_out} features")
                elif hasattr(self.backbone, 'head') and hasattr(self.backbone.head, 'fc'):
                    backbone_out = self.backbone.head.fc.in_features
                    self.backbone.head.fc = nn.Identity()
                    print(f"Using MobileNetV3 head.fc with {backbone_out} features")
                else:
                    # Fallback to typical mobilenetv3 small dimension
                    backbone_out = 1280  # Standard size for MobileNetV3 Small
                    if hasattr(self.backbone, 'classifier'):
                        self.backbone.classifier = nn.Identity()
                    print(f"Using fallback MobileNetV3 feature dimension: {backbone_out}")
            else:
                # Try to get classifier info for other models
                print("Using generic classifier detection")
                if hasattr(self.backbone, 'get_classifier') and callable(getattr(self.backbone, 'get_classifier')):
                    backbone_out = self.backbone.get_classifier().in_features
                    self.backbone.reset_classifier(0, '')
                else:
                    # Last resort - find any linear layer as a hint
                    backbone_out = 0
                    for name, module in self.backbone.named_modules():
                        if isinstance(module, nn.Linear):
                            backbone_out = module.in_features
                            print(f"Found linear layer with {backbone_out} features: {name}")
                            # Don't break, we want the last one
                    
                    if backbone_out == 0:
                        backbone_out = 1280  # Default fallback
                    print(f"Using fallback feature dimension: {backbone_out}")
        except Exception as e:
            print(f"Error setting up classifier: {e}")
            # Fallback to a reasonable size for MobileNetV3
            backbone_out = 1280
            print(f"Using emergency fallback feature dimension: {backbone_out}")
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        
        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(backbone_out, backbone_out // 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(backbone_out // 16, backbone_out, kernel_size=1),
            nn.Sigmoid()
        )
            
        self.feat_dim = backbone_out
        
        # Add multi-sample dropout for better generalization
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.3) for _ in range(5)
        ])
        
        self.classifier = nn.Linear(backbone_out, cfg.num_classes)
        
        self.mixup_enabled = hasattr(cfg, 'mixup_alpha') and cfg.mixup_alpha > 0
        self.cutmix_enabled = hasattr(cfg, 'cutmix_alpha') and cfg.cutmix_alpha > 0
        
        if self.mixup_enabled:
            self.mixup_alpha = cfg.mixup_alpha
        if self.cutmix_enabled:
            self.cutmix_alpha = cfg.cutmix_alpha
            
    def forward(self, x, targets=None):
        b = x.size(0)
        
        # Apply mixup or cutmix during training
        if self.training and targets is not None:
            if self.mixup_enabled and self.cutmix_enabled:
                # Randomly choose between mixup and cutmix
                if random.random() < 0.5:
                    x, targets_a, targets_b, lam = self.mixup_data(x, targets)
                else:
                    x, targets_a, targets_b, lam = self.cutmix_data(x, targets)
            elif self.mixup_enabled:
                x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            elif self.cutmix_enabled:
                x, targets_a, targets_b, lam = self.cutmix_data(x, targets)
            else:
                targets_a, targets_b, lam = targets, targets, 1.0
        else:
            targets_a, targets_b, lam = None, None, None
        
        features = self.backbone(x)
        
        # Handle different output formats from different backbones
        if isinstance(features, dict):
            features = features['features']
        
        # For MobileNetV3 and other models, ensure we have 4D tensor for attention
        # If features is already flattened (2D), reshape it to 4D for attention
        if len(features.shape) == 2:
            # Create pseudo spatial dimensions
            features = features.unsqueeze(-1).unsqueeze(-1)
            
        # Now features should be 4D, apply attention mechanism
        att = self.attention(features)
        features = features * att
        
        # Pool and flatten
        features = self.pooling(features)
        features = features.view(b, -1)
        
        # Multi-sample dropout for robust training
        if self.training:
            logits = torch.zeros(b, self.cfg.num_classes).to(features.device)
            for dropout in self.dropouts:
                logits += self.classifier(dropout(features))
            logits /= len(self.dropouts)
        else:
            logits = self.classifier(features)
        
        if self.training and (self.mixup_enabled or self.cutmix_enabled) and targets is not None:
            loss = self.mixup_criterion(F.binary_cross_entropy_with_logits, 
                                       logits, targets_a, targets_b, lam)
            return logits, loss
            
        return logits
    
    def mixup_data(self, x, targets):
        """Applies mixup to the data batch"""
        batch_size = x.size(0)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        indices = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[indices]
        
        return mixed_x, targets, targets[indices], lam
    
    def cutmix_data(self, x, targets):
        """Applies cutmix to the data batch"""
        batch_size = x.size(0)
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        # Generate random box
        W, H = x.size(2), x.size(3)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        
        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Limit box to image
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        
        # Random sample
        rand_index = torch.randperm(batch_size).to(x.device)
        
        # Apply cutmix - first verify the indices are valid
        x_cut = x.clone()
        
        # Only apply if the box has valid dimensions
        if bbx2 > bbx1 and bby2 > bby1:
            x_cut[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            
            # Adjust lambda
            cut_area = (bbx2 - bbx1) * (bby2 - bby1)
            lam = 1.0 - (cut_area / (W * H))
        else:
            print(f"Warning: Invalid cutmix box dimensions ({bbx1},{bby1})-({bbx2},{bby2})")
        
        return x_cut, targets, targets[rand_index], lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Applies mixup to the loss function"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_optimizer(model, cfg):
  
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented")
        
    return optimizer

def get_scheduler(optimizer, cfg):
   
    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.T_max,
            eta_min=cfg.min_lr
        )
    elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.T_0,
            T_mult=cfg.T_mult,
            eta_min=cfg.min_lr
        )
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=cfg.min_lr,
            verbose=True
        )
    elif cfg.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.epochs // 3,
            gamma=0.5
        )
    elif cfg.scheduler == 'OneCycleLR':
        scheduler = None  
    else:
        scheduler = None
        
    return scheduler

def get_criterion(cfg):
 
    if cfg.criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Criterion {cfg.criterion} not implemented")
        
    return criterion

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None, scaler=None):
    
    model.train()
    losses = []
    all_targets = []
    all_outputs = []
    optimizer.zero_grad()
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    
    for step, batch in pbar:
        # Skip empty batches
        if not batch:
            continue
            
        if isinstance(batch['melspec'], list):
            batch_outputs = []
            batch_losses = []
            
            for i in range(len(batch['melspec'])):
                inputs = batch['melspec'][i].unsqueeze(0).to(device)
                target = batch['target'][i].unsqueeze(0).to(device)
                
                if cfg.USE_AMP:
                    with autocast():
                        output = model(inputs)
                        loss = criterion(output, target)
                    
                    scaler.scale(loss).backward()
                    batch_outputs.append(output.detach().cpu())
                    batch_losses.append(loss.item())
                else:
                    output = model(inputs)
                    loss = criterion(output, target)
                    loss.backward()
                    batch_outputs.append(output.detach().cpu())
                    batch_losses.append(loss.item())
            
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if cfg.USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
            outputs = torch.cat(batch_outputs, dim=0).numpy()
            loss = np.mean(batch_losses)
            targets = batch['target'].numpy()
            
        else:
            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)
            
            if cfg.USE_AMP:
                with autocast():
                    outputs = model(inputs)
                    
                    if isinstance(outputs, tuple):
                        outputs, loss = outputs  
                    else:
                        loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                
                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(inputs)
                
                if isinstance(outputs, tuple):
                    outputs, loss = outputs  
                else:
                    loss = criterion(outputs, targets)
                    
                loss.backward()
                
                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
        
        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()
            
        all_outputs.append(outputs)
        all_targets.append(targets)
        losses.append(loss if isinstance(loss, float) else loss.item())
        
        pbar.set_postfix({
            'train_loss': np.mean(losses[-10:]) if losses else 0,
            'lr': optimizer.param_groups[0]['lr']
        })
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    
    return avg_loss, auc

@torch.no_grad()
def validate(model, loader, criterion, device):
   
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []
    
    for batch in tqdm(loader, desc="Validation"):
        # Skip empty batches
        if not batch:
            continue
            
        if isinstance(batch['melspec'], list):
            batch_outputs = []
            batch_losses = []
            
            for i in range(len(batch['melspec'])):
                inputs = batch['melspec'][i].unsqueeze(0).to(device)
                target = batch['target'][i].unsqueeze(0).to(device)
                
                if cfg.USE_AMP:
                    with autocast():
                        output = model(inputs)
                        loss = criterion(output, target)
                else:
                    output = model(inputs)
                    loss = criterion(output, target)
                
                batch_outputs.append(output.detach().cpu())
                batch_losses.append(loss.item())
            
            outputs = torch.cat(batch_outputs, dim=0).numpy()
            loss = np.mean(batch_losses)
            targets = batch['target'].numpy()
                
        else:
            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)
            
            if cfg.USE_AMP:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
        
        all_outputs.append(outputs)
        all_targets.append(targets)
        losses.append(loss if isinstance(loss, float) else loss.item())
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    
    return avg_loss, auc

def calculate_auc(targets, outputs):
    """Optimized AUC calculation"""
    num_classes = targets.shape[1]
    probs = 1 / (1 + np.exp(-outputs))
    
    # Vectorized approach for classes with positive samples
    aucs = []
    active_classes = np.where(np.sum(targets, axis=0) > 0)[0]
    
    for i in active_classes:
        class_auc = roc_auc_score(targets[:, i], probs[:, i])
        aucs.append(class_auc)
    
    return np.mean(aucs) if aucs else 0.0

def run_training(df, cfg):
    """Training function that can either use pre-computed spectrograms or generate them on-the-fly"""

    # Ensure torch is imported in the local scope
    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast, GradScaler
    from torch.optim import lr_scheduler

    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    species_ids = taxonomy_df['primary_label'].tolist()
    cfg.num_classes = len(species_ids)
    
    if cfg.debug:
        cfg.update_debug_settings()

    # Handle spectrograms loading
    spectrograms = None
    if cfg.spectrogram_npy and os.path.exists(cfg.spectrogram_npy):
        print(f"Loading pre-computed mel spectrograms from NPY file: {cfg.spectrogram_npy}")
        try:
            # Use improved loading function
            spectrograms = load_spectrograms_mmap(cfg.spectrogram_npy)
            print(f"Successfully loaded {len(spectrograms)} pre-computed spectrograms")
        except Exception as e:
            print(f"Error loading pre-computed spectrograms: {e}")
            spectrograms = None
    else:
        print("No pre-computed spectrograms available. Will generate on-the-fly.")
        # You could add code here to generate spectrograms if needed
    
    if 'filepath' not in df.columns:
        df['filepath'] = cfg.train_datadir + '/' + df.filename
    if 'samplename' not in df.columns:
        df['samplename'] = df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
        
    skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    
    best_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['primary_label'])):
        if fold not in cfg.selected_folds:
            continue
            
        print(f'\n{"="*30} Fold {fold} {"="*30}')
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f'Training set: {len(train_df)} samples')
        print(f'Validation set: {len(val_df)} samples')
        
        train_dataset = BirdCLEFDatasetFromNPY(train_df, cfg, spectrograms=spectrograms, mode='train')
        val_dataset = BirdCLEFDatasetFromNPY(val_df, cfg, spectrograms=spectrograms, mode='valid')
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            num_workers=cfg.num_workers,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=cfg.num_workers > 0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=False, 
            num_workers=cfg.num_workers,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=collate_fn,
            persistent_workers=cfg.num_workers > 0
        )
        
        model = BirdCLEFModel(cfg).to(cfg.device)
        
        # Try to enable torch.compile for PyTorch 2.0+
        if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
            try:
                # Set dynamo config to suppress errors and fall back to eager mode
                import torch._dynamo
                torch._dynamo.config.suppress_errors = True
                
                # You can set USE_COMPILE = False to disable compilation completely
                USE_COMPILE = True
                
                if USE_COMPILE:
                    model = torch.compile(model, backend='eager')  # Use 'eager' backend instead of default 'inductor'
                    print("Using torch.compile() with eager backend for JIT acceleration")
            except Exception as e:
                print(f"torch.compile() failed: {e}, using standard model")
                
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg)
        
        if cfg.scheduler == 'OneCycleLR':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.lr,
                steps_per_epoch=len(train_loader) // cfg.gradient_accumulation_steps,
                epochs=cfg.epochs,
                pct_start=0.1
            )
        else:
            scheduler = get_scheduler(optimizer, cfg)
        
        # Initialize AMP scaler
        scaler = GradScaler() if cfg.USE_AMP else None
        
        best_auc = 0
        best_epoch = 0
        
        for epoch in range(cfg.epochs):
            print(f"\nEpoch {epoch+1}/{cfg.epochs}")
            
            train_loss, train_auc = train_one_epoch(
                model, 
                train_loader, 
                optimizer, 
                criterion, 
                cfg.device,
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None,
                scaler
            )
            
            val_loss, val_auc = validate(model, val_loader, criterion, cfg.device)

            if scheduler is not None and not isinstance(scheduler, lr_scheduler.OneCycleLR):
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch + 1
                print(f"New best AUC: {best_auc:.4f} at epoch {best_epoch}")

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'epoch': epoch,
                    'val_auc': val_auc,
                    'train_auc': train_auc,
                    'cfg': cfg
                }, f"model_fold{fold}.pth")
        
        best_scores.append(best_auc)
        print(f"\nBest AUC for fold {fold}: {best_auc:.4f} at epoch {best_epoch}")
        
        # Clear memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n" + "="*60)
    print("Cross-Validation Results:")
    for fold, score in enumerate(best_scores):
        print(f"Fold {cfg.selected_folds[fold]}: {score:.4f}")
    print(f"Mean AUC: {np.mean(best_scores):.4f}")
    print("="*60)

def save_model(model, optimizer=None, scheduler=None, epoch=0, val_auc=0, train_auc=0, cfg=None, path="./bird_model.pth"):
    """
    Save model, optimizer, scheduler states and configuration to the specified path
    """
    # Get model state dict and handle compiled model case
    state_dict = model.state_dict()
    
    # Handle compiled model state dict (keys with "_orig_mod." prefix)
    fixed_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            fixed_state_dict[k[10:]] = v  # Remove '_orig_mod.' prefix (10 characters)
        else:
            fixed_state_dict[k] = v
    
    save_dict = {
        'model_state_dict': fixed_state_dict,
        'epoch': epoch,
        'val_auc': val_auc,
        'train_auc': train_auc
    }
    
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
        
    if cfg is not None:
        save_dict['cfg'] = cfg
        
    torch.save(save_dict, path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    print("Starting BirdCLEF training with EfficientNetV2-S...")
    
    # Check if timm is properly installed and can access models
    try:
        import timm
        print(f"TIMM version: {timm.__version__}")
        
        # List available EfficientNetV2 models
        available_models = [m for m in timm.list_models() if 'efficientnetv2_s' in m.lower()]
        print(f"Available EfficientNetV2_S models in timm: {available_models}")
        
        if not available_models:
            print("No EfficientNetV2_S models found, installing latest timm version...")
            import subprocess
            subprocess.run(["pip", "install", "-U", "timm"], check=True)
            print("Timm updated. Please restart the notebook/script.")
    except Exception as e:
        print(f"Error with timm: {e}")
        print("Installing timm...")
        import subprocess
        subprocess.run(["pip", "install", "timm"], check=True)
        print("Please restart the notebook/script after timm installation.")
    
    # Try to load and process the data
    try:
        if os.path.exists(cfg.train_csv):
            print(f"Loading train data from {cfg.train_csv}")
            df = pd.read_csv(cfg.train_csv)
            print(f"Loaded {len(df)} training samples")
            
            # Run training
            run_training(df, cfg)
        else:
            print(f"Training CSV not found at {cfg.train_csv}")
            # Use relative paths for local testing if Kaggle paths not available
            if not os.path.exists('./train.csv'):
                print("No training data found. Please make sure the data is available.")
            else:
                print("Using local data paths...")
                cfg.train_datadir = './train_audio'  
                cfg.train_csv = './train.csv'
                cfg.taxonomy_csv = './taxonomy.csv'
                df = pd.read_csv(cfg.train_csv)
                run_training(df, cfg)
    except Exception as e:
        print(f"Error running training: {e}")
        import traceback
        traceback.print_exc()




