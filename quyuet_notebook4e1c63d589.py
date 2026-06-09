# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import logging
import random
import gc
import time
import cv2
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import librosa 

import torch
import torchvision
import torchvision.transforms as transforms 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import timm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


class CFG:
    seed = 42
    debug = False
    apex = False
    print_freq = 100
    num_workers = 2 

    OUTPUT_DIR = '/kaggle/working/'

    # --- INPUT DATA ---
    train_datadir = '/kaggle/input/birdclef-2025/train_audio' 
    train_csv = '/kaggle/input/birdclef-2025/train.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
   
    spectrogram_npy = '/kaggle/input/melspec-train-audio-update/birdclef25_melspec_5s_randcrop_32k_2048fft_512hop_128mel_rs256.npy'

    # --- MODEL ---
    model_name = 'efficientnet_b0'
    pretrained = True
    in_channels = 3 
    dropout_prob = 0.3 

    # --- DATA HANDLING ---
    LOAD_DATA = True  
    # --- Đồng bộ tham số Audio/Mel với transforming.py ---
    FS = 32000
    TARGET_DURATION = 5.0
    N_FFT = 2048
    HOP_LENGTH = 512
    WIN_LENGTH = 2048 
    N_MELS = 128
    FMIN = 20
    FMAX = 16000
    TARGET_SHAPE = (256, 256) 
    # ------------------------------------------------------

    # --- TRAINING ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10 
    batch_size = 32
    criterion = 'FocalLossBCE'
    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]
    optimizer = 'AdamW'
    lr = 3e-4
    weight_decay = 1e-5
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = epochs

    # --- AUGMENTATION ---
    aug_prob = 0.5
    mixup_alpha = 0.5

    def update_debug_settings(self):
        if self.debug:
            self.epochs = 1
            self.selected_folds = [0]
            self.debug_limit = 1000 

cfg = CFG()


taxonomy_df_global = pd.read_csv(cfg.taxonomy_csv)
cfg.num_classes = len(taxonomy_df_global)
cfg.species_ids = taxonomy_df_global['primary_label'].tolist()
print(f"Number of classes: {cfg.num_classes}")
print(f"Device: {cfg.device}")


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)


def audio2melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        win_length=cfg.WIN_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0,
        center=True,
        pad_mode="reflect",
        norm='slaney',
        htk=True,
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm

def process_audio_file(audio_path, cfg):
    """Process a single audio file to get the mel spectrogram"""
    try:
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)

        target_samples = int(cfg.TARGET_DURATION * cfg.FS)

        if len(audio_data) < target_samples:
            n_copy = math.ceil(target_samples / len(audio_data))
            if n_copy > 1:
                audio_data = np.concatenate([audio_data] * n_copy)

        # Extract center 5 seconds
        start_idx = max(0, int(len(audio_data) / 2 - target_samples / 2))
        end_idx = min(len(audio_data), start_idx + target_samples)
        center_audio = audio_data[start_idx:end_idx]

        if len(center_audio) < target_samples:
            center_audio = np.pad(center_audio, 
                                 (0, target_samples - len(center_audio)), 
                                 mode='constant')

        mel_spec = audio2melspec(center_audio, cfg)
        
        if mel_spec.shape != cfg.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

        return mel_spec.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def generate_spectrograms(df, cfg):
    """Generate spectrograms from audio files"""
    print("Generating mel spectrograms from audio files...")
    start_time = time.time()

    all_bird_data = {}
    errors = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        if cfg.debug and i >= 1000:
            break
        
        try:
            samplename = row['samplename']
            filepath = row['filepath']
            
            mel_spec = process_audio_file(filepath, cfg)
            
            if mel_spec is not None:
                all_bird_data[samplename] = mel_spec
            
        except Exception as e:
            print(f"Error processing {row.filepath}: {e}")
            errors.append((row.filepath, str(e)))

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Successfully processed {len(all_bird_data)} files out of {len(df)}")
    print(f"Failed to process {len(errors)} files")
    
    return all_bird_data


class BirdCLEFDatasetFromNPY(Dataset):
    def __init__(self, df, cfg, spectrograms=None, mode="train"):
        self.df = df
        self.cfg = cfg
        self.mode = mode
        self.spectrograms = spectrograms

        self.species_ids = cfg.species_ids
        self.num_classes = cfg.num_classes
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        if 'samplename' not in self.df.columns:
             self.df['samplename'] = self.df.filename.map(lambda x: x.replace('.ogg',''))

        # Định nghĩa transform cho normalization (áp dụng cho cả train/valid)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Kiểm tra số lượng spectrograms khớp
        if self.spectrograms:
            sample_names_df = set(self.df['samplename'])
            found_samples = sum(1 for name in sample_names_df if name in self.spectrograms)
            missing_samples = len(sample_names_df) - found_samples
            print(f"Dataset '{mode}': Found {found_samples} matching spectrograms out of {len(sample_names_df)} unique samples.")
            if missing_samples > 0:
                print(f"Warning: {missing_samples} samples in the dataframe partition do not have matching spectrograms.")
        else:
             print(f"Warning: No spectrogram dictionary provided for dataset '{mode}'.")


        if cfg.debug and hasattr(cfg, 'debug_limit'):
            self.df = self.df.sample(min(cfg.debug_limit, len(self.df)), random_state=cfg.seed).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        spec = None

        if self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
        else:
            # Xử lý trường hợp thiếu spec (ví dụ: trả về spec toàn 0)
            # print(f"Warning: Spectrogram for {samplename} not found. Returning zeros.")
            # Cần shape gốc trước khi repeat kênh
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)

        # Đảm bảo spec có đúng shape trước khi repeat
        if spec.shape != self.cfg.TARGET_SHAPE:
             # print(f"Warning: Spectrogram for {samplename} has shape {spec.shape}, expected {self.cfg.TARGET_SHAPE}. Resizing/Padding might be needed or NPY mismatch.")
             if spec.shape[1] > 0 and spec.shape[0] > 0:
                 spec = cv2.resize(spec, (self.cfg.TARGET_SHAPE[1], self.cfg.TARGET_SHAPE[0]), interpolation=cv2.INTER_LINEAR)
             else: # Nếu shape không hợp lệ, trả về zero
                 spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)


        # *** Chuyển sang Tensor 3 kênh ***
        if len(spec.shape) == 3: 
             spec = spec.squeeze()
        if len(spec.shape) != 2:
             print(f"Error: Unexpected shape for spec {samplename}: {spec.shape}. Returning None.")
             return None 

        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1) # Tạo (3, H, W)

        # *** Áp dụng ImageNet Normalization ***
        spec_tensor = self.normalize(spec_tensor)

        # Áp dụng augmentations trên spectrogram (sau normalization)
        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec_tensor = self.apply_spec_augmentations(spec_tensor)

        target = self.encode_label(row['primary_label'])

        # Xử lý secondary_labels
        if 'secondary_labels' in row and isinstance(row['secondary_labels'], str) and row['secondary_labels'] != '[]':
             try:
                 secondary_labels = eval(row['secondary_labels'])
                 for label in secondary_labels:
                     if label in self.label_to_idx:
                         target[self.label_to_idx[label]] = 1.0
             except Exception:
                 pass 
        elif 'secondary_labels' in row and isinstance(row['secondary_labels'], list) and row['secondary_labels']:
             for label in row['secondary_labels']:
                 if label in self.label_to_idx:
                     target[self.label_to_idx[label]] = 1.0

        return {
            'melspec': spec_tensor,
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': row['filename']
        }

    def apply_spec_augmentations(self, spec):
        # Time masking
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                width = random.randint(5, 30) # Tăng nhẹ max width
                start = random.randint(0, max(0, spec.shape[2] - width)) 
                if width > 0 : spec[:, :, start:start+width] = 0

        # Frequency masking
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                height = random.randint(5, 30) # Tăng nhẹ max height
                start = random.randint(0, max(0, spec.shape[1] - height)) 
                if height > 0: spec[:, start:start+height, :] = 0

        # Thêm RandomErasing
        # spec = transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))(spec)

        return spec

    def encode_label(self, label):
        target = np.zeros(self.num_classes, dtype=np.float32)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target


def collate_fn(batch):
    # Lọc ra các item không hợp lệ (trả về None từ __getitem__)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None 

    elem = batch[0]
    melspecs = torch.stack([item['melspec'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    filenames = [item['filename'] for item in batch]

    return {'melspec': melspecs, 'target': targets, 'filename': filenames}


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels, # *** Sử dụng 3 kênh ***
            drop_rate=0.2,
            drop_path_rate=0.2
        )

        # Lấy số features output từ backbone
        if hasattr(self.backbone, 'get_classifier'):
             backbone_out = self.backbone.get_classifier().in_features
        elif hasattr(self.backbone, 'head') and hasattr(self.backbone.head, 'in_features'): # Xử lý các kiến trúc khác như Swin
             backbone_out = self.backbone.head.in_features
        elif hasattr(self.backbone, 'fc') and hasattr(self.backbone.fc, 'in_features'):
            backbone_out = self.backbone.fc.in_features
        elif hasattr(self.backbone, 'classifier') and hasattr(self.backbone.classifier, 'in_features'):
            backbone_out = self.backbone.classifier.in_features
        else: 
            try:
                backbone_out = self.backbone.num_features
            except AttributeError:
                raise ValueError(f"Không thể tự động xác định output features cho model {cfg.model_name}")

        # Reset classifier gốc của timm model
        if hasattr(self.backbone, 'reset_classifier'):
            self.backbone.reset_classifier(0, '')
        elif hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
             self.backbone.classifier = nn.Identity()

        self.pooling = nn.AdaptiveAvgPool2d(1)
        # *** Thêm lớp Dropout ***
        self.dropout = nn.Dropout(p=cfg.dropout_prob)
        self.classifier = nn.Linear(backbone_out, num_classes)

        # Mixup config
        self.mixup_enabled = hasattr(cfg, 'mixup_alpha') and cfg.mixup_alpha > 0
        if self.mixup_enabled:
            self.mixup_alpha = cfg.mixup_alpha

    def forward(self, x, targets=None):
        targets_a, targets_b, lam = None, None, None 

        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x

        features = self.backbone(x)

        if isinstance(features, dict):
             features = features.get('features', features.get('head_output', next(iter(features.values()))))

        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1) # Flatten

        # *** Áp dụng Dropout ***
        features = self.dropout(features)

        logits = self.classifier(features)

        if self.training:
            if self.mixup_enabled and targets is not None:
                 return logits, targets_a, targets_b, lam
            else:
                 return logits, targets, None, None 
        else: # Khi eval
             return logits

    def mixup_data(self, x, targets):
        batch_size = x.size(0)
        alpha = self.mixup_alpha if self.mixup_alpha > 0 else 1e-6
        lam = np.random.beta(alpha, alpha)
        indices = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[indices]
        return mixed_x, targets, targets[indices], lam


def get_optimizer(model, cfg):
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented")
    return optimizer

# --- Scheduler ---
def get_scheduler(optimizer, cfg):
    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr)
    elif cfg.scheduler == 'ReduceLROnPlateau':
        # Chú ý mode='max' vì chúng ta theo dõi AUC
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=cfg.min_lr, verbose=True)
    elif cfg.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.epochs // 3, gamma=0.5)
    elif cfg.scheduler == 'OneCycleLR':
        scheduler = None 
    else:
        scheduler = None
    return scheduler

# --- Loss Function ---
class FocalLossBCE(torch.nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "mean", bce_weight: float = 0.5, focal_weight: float = 0.5): # Điều chỉnh weight nếu muốn
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none') # Tính loss cho từng sample/class
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        # Sigmoid focal loss tính toán nội bộ sigmoid
        focal_loss = torchvision.ops.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction='none', # Tính loss cho từng sample/class
        )

        # Kết hợp loss
        combined_loss = self.bce_weight * bce_loss + self.focal_weight * focal_loss

        # Áp dụng reduction cuối cùng
        if self.reduction == "mean":
            return combined_loss.mean()
        elif self.reduction == "sum":
            return combined_loss.sum()
        else: # 'none'
            return combined_loss

def get_criterion(cfg):
    if cfg.criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif cfg.criterion == 'FocalLossBCE':
        criterion = FocalLossBCE(bce_weight=0.5, focal_weight=0.5)
    else:
        raise NotImplementedError(f"Criterion {cfg.criterion} not implemented")
    return criterion


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    losses = []
    all_targets_np = []
    all_outputs_np = []

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training", leave=False)

    for step, batch in pbar:
        if batch is None: continue

        inputs = batch['melspec'].to(device)
        targets_orig = batch['target'].to(device)

        optimizer.zero_grad()
        outputs, targets_a, targets_b, lam = model(inputs, targets_orig)

        if targets_a is not None: # Mixup
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            targets_for_auc = targets_orig 
        else: # No mixup
            loss = criterion(outputs, targets_orig)
            targets_for_auc = targets_orig

        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
        else:
            print(f"Warning: NaN/Inf loss detected at step {step}. Skipping step.")
            losses.append(np.nan) # Ghi nhận NaN

        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()

        all_outputs_np.append(outputs.detach().cpu().numpy())
        all_targets_np.append(targets_for_auc.detach().cpu().numpy())

        pbar.set_postfix({
            'train_loss': f"{np.nanmean(losses[-50:]):.4f}" if losses else 'N/A',
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })

    if not all_outputs_np:
        print("Warning: No valid batches processed in this epoch.")
        return 0.0, 0.0

    all_outputs_np = np.concatenate(all_outputs_np)
    all_targets_np = np.concatenate(all_targets_np)

    # Lọc NaN trước khi tính AUC
    valid_idx = ~np.isnan(all_outputs_np).any(axis=1)
    if not np.any(valid_idx):
         print("Warning: All outputs are NaN. Cannot calculate AUC.")
         return np.nanmean(losses), 0.0

    auc = calculate_auc(all_targets_np[valid_idx], all_outputs_np[valid_idx])
    avg_loss = np.nanmean(losses) # Tính trung bình bỏ qua NaN

    return avg_loss, auc


def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_targets_np = []
    all_outputs_np = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            if batch is None: continue

            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)

            outputs = model(inputs) # model.eval() chỉ trả về logits
            loss = criterion(outputs, targets)

            if not torch.isnan(loss) and not torch.isinf(loss):
                 losses.append(loss.item())
            else:
                 losses.append(np.nan)

            all_outputs_np.append(outputs.cpu().numpy())
            all_targets_np.append(targets.cpu().numpy())

    if not all_outputs_np:
        print("Warning: No valid batches processed in validation.")
        return 0.0, 0.0

    all_outputs_np = np.concatenate(all_outputs_np)
    all_targets_np = np.concatenate(all_targets_np)

    valid_idx = ~np.isnan(all_outputs_np).any(axis=1)
    if not np.any(valid_idx):
         print("Warning: All validation outputs are NaN. Cannot calculate AUC.")
         return np.nanmean(losses), 0.0

    auc = calculate_auc(all_targets_np[valid_idx], all_outputs_np[valid_idx])
    avg_loss = np.nanmean(losses)

    return avg_loss, auc

def calculate_auc(targets, outputs):
    num_classes = targets.shape[1]
    aucs = []
    # Áp dụng sigmoid cho logits để có xác suất
    probs = 1 / (1 + np.exp(-outputs))

    for i in range(num_classes):
        target_class = targets[:, i]
        if np.sum(target_class) > 0 and np.sum(target_class) < len(target_class):
            try:
                class_auc = roc_auc_score(target_class, probs[:, i])
                aucs.append(class_auc)
            except ValueError as e:
                # print(f"Skipping AUC for class {i} due to error: {e}")
                aucs.append(0.5) # Hoặc np.nan
        # elif np.sum(target_class) == 0: # Nếu lớp không có trong validation fold
        #      pass # 
        # else: # Nếu lớp chỉ có toàn 1 
        #      aucs.append(0.5) # 

    return np.mean(aucs) if aucs else 0.0 # Trả về 0 nếu không có lớp nào hợp lệ


def run_training(df, cfg):
    if cfg.debug:
        cfg.update_debug_settings()

    spectrograms = None
    if cfg.LOAD_DATA:
        print(f"Loading pre-computed mel spectrograms from: {cfg.spectrogram_npy}")
        npy_path = Path(cfg.spectrogram_npy)
        if not npy_path.is_file():
             print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print(f"ERROR: Spectrogram file NOT FOUND at {cfg.spectrogram_npy}")
             print(f"Please verify the path and ensure the transforming script ran successfully.")
             print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             return # Thoát nếu không tìm thấy file
        try:
            # Ghi nhớ thời gian load
            load_start = time.time()
            spectrograms = np.load(cfg.spectrogram_npy, allow_pickle=True).item()
            load_end = time.time()
            print(f"Loaded {len(spectrograms)} pre-computed mel spectrograms in {load_end - load_start:.2f} seconds.")
            if not spectrograms:
                 print("ERROR: Loaded spectrogram dictionary is empty!")
                 return

            # Kiểm tra shape của một sample
            first_key = next(iter(spectrograms))
            first_spec_shape = spectrograms[first_key].shape
            print(f"Shape of first spectrogram ('{first_key}'): {first_spec_shape}")
            # *** Kiểm tra shape có khớp với TARGET_SHAPE không ***
            if cfg.TARGET_SHAPE != first_spec_shape:
                 print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                 print(f"WARNING: Spectrogram shape in NPY {first_spec_shape} does NOT match CFG.TARGET_SHAPE {cfg.TARGET_SHAPE}!")
                 print(f"Ensure TARGET_SHAPE in this notebook matches the output shape of transforming.py.")
                 print(f"         (Remember that resize happens in transforming.py if DO_RESIZE=True)")
                 print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                 # return

        except Exception as e:
            print(f"Error loading pre-computed spectrograms: {e}")
            return 
    else:
        print("LOAD_DATA is False. Spectrograms will be generated on-the-fly.")

    skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    oof_scores = [] 

    # --- Vòng lặp qua các Fold ---
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['primary_label'])):
        if fold not in cfg.selected_folds:
            continue

        print(f'\n{"="*30} Fold {fold} {"="*30}')
        train_df_fold = df.iloc[train_idx].reset_index(drop=True)
        val_df_fold = df.iloc[val_idx].reset_index(drop=True)

        print(f'Training set samples: {len(train_df_fold)}')
        print(f'Validation set samples: {len(val_df_fold)}')

        # Tạo Datasets và DataLoaders cho fold hiện tại
        train_dataset = BirdCLEFDatasetFromNPY(train_df_fold, cfg, spectrograms=spectrograms, mode='train')
        val_dataset = BirdCLEFDatasetFromNPY(val_df_fold, cfg, spectrograms=spectrograms, mode='valid')

        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size * 2, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn)

        # Khởi tạo model, optimizer, criterion, scheduler cho fold
        # *** Truyền cfg.num_classes vào model ***
        model = BirdCLEFModel(cfg, cfg.num_classes).to(cfg.device)
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg)

        if cfg.scheduler == 'OneCycleLR':
            steps_per_epoch = len(train_loader)
            if steps_per_epoch == 0:
                 print("Warning: train_loader is empty. Cannot initialize OneCycleLR.")
                 scheduler = None
            else:
                 scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, steps_per_epoch=steps_per_epoch, epochs=cfg.epochs, pct_start=0.1)
        else:
            scheduler = get_scheduler(optimizer, cfg)

        best_fold_auc = 0
        best_epoch = 0
        fold_start_time = time.time()

        # --- Vòng lặp qua các Epoch ---
        for epoch in range(cfg.epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{cfg.epochs}")

            train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device, scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None)
            val_loss, val_auc = validate(model, val_loader, criterion, cfg.device)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            if scheduler is not None and not isinstance(scheduler, lr_scheduler.OneCycleLR):
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_auc) # Step dựa trên validation AUC
                else:
                    scheduler.step()

            print(f"Epoch {epoch+1} Summary:")
            print(f"  Time: {epoch_duration:.2f}s")
            print(f"  Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"  Val Loss  : {val_loss:.4f}, Val AUC  : {val_auc:.4f}")

            # Lưu model tốt nhất của fold này
            if val_auc > best_fold_auc:
                best_fold_auc = val_auc
                best_epoch = epoch + 1
                print(f"  >>> New best AUC for Fold {fold}: {best_fold_auc:.4f} at epoch {best_epoch} <<<")

                checkpoint_payload = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_auc': val_auc,
                    'train_auc': train_auc,
                    'cfg': {
                        'model_name': cfg.model_name,
                        'in_channels': cfg.in_channels,
                        'num_classes': cfg.num_classes,
                        'TARGET_SHAPE': cfg.TARGET_SHAPE,
                        'N_FFT': cfg.N_FFT,
                        'HOP_LENGTH': cfg.HOP_LENGTH,
                        'N_MELS': cfg.N_MELS,
                    }
                }
                if scheduler:
                     checkpoint_payload['scheduler_state_dict'] = scheduler.state_dict()

                torch.save(checkpoint_payload, f"{cfg.OUTPUT_DIR}model_fold{fold}_best.pth")
                print(f"  Saved best model checkpoint for Fold {fold}")

        fold_end_time = time.time()
        print(f"\nFinished Fold {fold} in {(fold_end_time - fold_start_time)/60:.2f} minutes. Best Val AUC: {best_fold_auc:.4f} at epoch {best_epoch}")
        oof_scores.append(best_fold_auc)

        del model, optimizer, scheduler, train_loader, val_loader, train_dataset, val_dataset
        if cfg.device == 'cuda':
             torch.cuda.empty_cache()
        gc.collect()

    # --- Conclusion CV ---
    print("\n" + "="*60)
    print("Cross-Validation Results:")
    for i, fold_idx in enumerate(cfg.selected_folds):
        if i < len(oof_scores):
             print(f"Fold {fold_idx}: {oof_scores[i]:.4f}")
        else:
             print(f"Fold {fold_idx}: Not run or score not available")
    if oof_scores:
        print(f"Mean OOF AUC: {np.mean(oof_scores):.4f}")
    print("="*60)


if __name__ == "__main__":
    print("\nLoading training data metadata...")
    train_df_main = pd.read_csv(cfg.train_csv)

    if cfg.LOAD_DATA and not Path(cfg.spectrogram_npy).is_file():
         print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print(f"ERROR: Spectrogram file NOT FOUND at {cfg.spectrogram_npy}")
         print(f"Please ensure the path is correct and the file exists before running training.")
         print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
         print("\nStarting training...")
         run_training(train_df_main, cfg)
         print("\nTraining complete!")

