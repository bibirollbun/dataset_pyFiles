import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torchaudio
import torchaudio.transforms as AT
import timm
import random
from tqdm import tqdm
import concurrent.futures
import gc
from typing import Union
import time


def apply_power_to_low_ranked_cols(
    p: np.ndarray,
    top_k: int = 30,
    exponent: Union[int, float] = 2,
    inplace: bool = True
) -> np.ndarray:
    """
    Rank columns by their column‑wise maximum and raise every column whose
    rank falls below `top_k` to a given power.
    """
    if not inplace:
        p = p.copy()

    # Identify columns whose max value ranks below `top_k`
    tail_cols = np.argsort(-p.max(axis=0))[top_k:]

    # Apply the power transformation to those columns
    p[:, tail_cols] = p[:, tail_cols] ** exponent
    return p


def normalize_std(spec, eps=1e-6):
    """Normalize spectrogram by standard deviation"""
    mean = torch.mean(spec)
    std = torch.std(spec)
    return torch.where(std == 0, spec-mean, (spec - mean) / (std+eps))


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
                drop_rate=0.2,  # Lower dropout for MobileNetV3
                drop_path_rate=0.2  # Lower stochastic depth for MobileNetV3
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
                'mobilenetv3_small.100_in1k',  # Alternative name in newer timm
                'tf_mobilenetv3_small_100',    # TF variant
                'mobilenetv3_small'            # Simplified name
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
                state_dict = torch.load(cfg.pretrained_weights, map_location='cpu',weights_only=True)
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
class CFG:
    
    seed = 42
    debug = False  
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
    
    model_name = 'mobilenetv3_small_050'  # Changed from mobilenetv3_small_100 to match pretrained weights
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
    epochs = 15  # Increased from 10
    batch_size = 64  # Increased for MobileNetV3 which is smaller than EfficientNet
    gradient_accumulation_steps = 1  # Reduced since MobileNetV3 is more memory efficient
    criterion = 'BCEWithLogitsLoss'

    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]   

    optimizer = 'AdamW'
    lr = 2e-4  # Slightly higher learning rate for MobileNetV3 which converges faster
    weight_decay = 5e-5  # Reduced for MobileNetV3 to prevent overfitting
  
    scheduler = 'CosineAnnealingWarmRestarts'  # Changed from CosineAnnealingLR
    min_lr = 1e-6
    T_0 = 5  # For CosineAnnealingWarmRestarts
    T_mult = 1  # For CosineAnnealingWarmRestarts

    aug_prob = 0.7  # Increased from 0.5
    mixup_alpha = 0.4
    cutmix_alpha = 0.4  # Added cutmix
    
    def update_debug_settings(self):
        if self.debug:
            self.epochs = 2
            self.selected_folds = [0]


# --- Config ve model yükleme ---
cfg = CFG()
cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create mel spectrogram transformer
mel_spectrogram = AT.MelSpectrogram(
    sample_rate=cfg.FS,
    n_fft=cfg.N_FFT,
    win_length=cfg.N_FFT,
    hop_length=cfg.HOP_LENGTH,
    center=True,
    f_min=cfg.FMIN,
    f_max=cfg.FMAX,
    pad_mode="reflect",
    power=2.0,
    norm='slaney',
    n_mels=cfg.N_MELS,
    mel_scale="htk",
)

# Improved audio to mel spectrogram conversion
def audio_to_mel(filepath):
    """Convert audio file to mel spectrogram tensors for all segments at once"""
    waveform, _ = torchaudio.load(filepath, backend="soundfile")
    len_wav = waveform.shape[1]
    waveform = waveform[0,:].reshape(1, len_wav)  # stereo->mono or mono->mono
    
    # Process all 12 segments at once
    segments = []
    for i in range(12):
        start_idx = i * cfg.FS * 5
        end_idx = start_idx + cfg.FS * 5
        
        # Handle case where audio might be shorter than expected
        if end_idx > len_wav:
            if start_idx < len_wav:
                # Pad the last segment
                segment = waveform[:, start_idx:len_wav]
                padding = end_idx - len_wav
                segment = F.pad(segment, (0, padding))
            else:
                # Create zeros if we're completely past the end
                segment = torch.zeros((1, cfg.FS * 5))
        else:
            segment = waveform[:, start_idx:end_idx]
            
        # Generate mel spectrogram for this segment
        melspec = mel_spectrogram(segment)
        melspec = torch.log(melspec + 1e-6)
        melspec = normalize_std(melspec)
        melspec = torch.unsqueeze(melspec, dim=0)  # Add batch dimension
        
        segments.append(melspec)
    
    # Stack all segments into a single batch tensor
    return torch.vstack(segments)

# Model checkpoint dosyasını seç (ör: model_fold0.pth)
MODEL_PATH = '/kaggle/input/distill0.64/pytorch/default/1/distilled_model_fold0.pth'  # Gerekirse değiştir

# Taxonomy ve class listesi
species_ids = pd.read_csv(cfg.taxonomy_csv)['primary_label'].tolist()
cfg.num_classes = len(species_ids)

# Modeli yükle
model = BirdCLEFModel(cfg)
print(f"Loading checkpoint from: {MODEL_PATH}")
try:
    checkpoint = torch.load(MODEL_PATH, map_location=cfg.device)
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove any prefix in the state_dict keys
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove _orig_mod. prefix if present
        if k.startswith('_orig_mod.'):
            new_key = k[10:]
        # Remove module. prefix if present (common in DataParallel models)
        elif k.startswith('module.'):
            new_key = k[7:]
        else:
            new_key = k
        new_state_dict[new_key] = v
    
    # Load the state dict with strict=False to ignore missing keys
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    print("Model loaded successfully with non-strict loading")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    print("Continuing with randomly initialized model")

model.to(cfg.device)
model.eval()

# --- Test dosyalarını bul ---
test_audio_dir = '../input/birdclef-2025/test_soundscapes/'
file_list = [f for f in sorted(os.listdir(test_audio_dir))] if os.path.exists(test_audio_dir) else []
file_list = [file.split('.')[0] for file in file_list if file.endswith('.ogg')]

debug = False
if len(file_list) == 0:
    debug = True
    debug_st_num = 5
    debug_num = 100
    test_audio_dir = '../input/birdclef-2025/train_soundscapes/'
    file_list = [f for f in sorted(os.listdir(test_audio_dir))] if os.path.exists(test_audio_dir) else []
    file_list = [file.split('.')[0] for file in file_list if file.endswith('.ogg')]
    file_list = file_list

print('Debug mode:', debug)
print('Number of test soundscapes:', len(file_list))

# --- Submission formatı ---
sample_sub = pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')
class_labels = list(sample_sub.columns)[1:]

# --- Improved prediction function ---
def predict_one_file(afile):
    path = os.path.join(test_audio_dir, afile + '.ogg')
    
    try:
        # Process all 12 segments at once
        with torch.inference_mode():  # Faster than no_grad
            # Convert audio to mel spectrograms (returns tensor of shape [12, 1, n_mels, time])
            mel_batch = audio_to_mel(path)
            mel_batch = mel_batch.to(cfg.device)
            
            # Get model predictions
            logits = model(mel_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # Apply post-processing to suppress low confidence predictions
            probs = apply_power_to_low_ranked_cols(probs, top_k=30, exponent=2)
            
        # Clean up GPU memory
        torch.cuda.empty_cache()
        return probs
    
    except Exception as e:
        print(f"Error processing file {afile}: {e}")
        return np.zeros((12, cfg.num_classes))

# --- Parallel prediction for all files ---
def process_all_files():
    all_rows = []
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(tqdm(executor.map(predict_one_file, file_list), total=len(file_list), desc="Processing files"))
    
    # Create submission rows from results
    for i, afile in enumerate(file_list):
        preds = results[i]  # Shape: (12, num_classes)
        for j in range(preds.shape[0]):
            row_id = f"{afile}_{(j+1)*5}"
            row = [row_id]
            
            # Add predictions for each class in the correct order
            for col in class_labels:
                if col in species_ids:
                    idx = species_ids.index(col)
                    row.append(preds[j, idx])
                else:
                    row.append(0.0)
            
            all_rows.append(row)
    
    return all_rows

# --- Create and save submission ---
start_time = time.time()
rows = process_all_files()
end_time = time.time()

print(f"Total processing time: {end_time - start_time:.2f} seconds")

# --- DataFrame and save ---
sub_df = pd.DataFrame(rows, columns=['row_id'] + class_labels)
sub_name = 'submission.csv'
sub_df.to_csv(sub_name, index=False)
print(f'Submission file created: {sub_name}') 




