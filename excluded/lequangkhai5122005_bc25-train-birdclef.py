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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.ops
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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

    train_datadir = '/kaggle/input/birdclef-2025/train_audio'
    train_csv = '/kaggle/input/birdclef-2025/train.csv'
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'

    spectrogram_npy = '/kaggle/input/my-new-dataset-melspec/new_melspec.npy'

    model_name = 'resnet34' 
    pretrained = True
    in_channels = 1

    LOAD_DATA = True  
    FS = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)
    
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 512
    FMIN = 20
    FMAX = 16000
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 15  
    batch_size = 32  

    # Tham số cho FocalLossBCE
    focal_alpha = 0.25
    focal_gamma = 2.0
    focal_bce_weight = 0.5 
    focal_focal_weight = 0.5
    loss_reduction = "mean"

    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]   

    optimizer = 'AdamW'
    lr = 5e-4 
    weight_decay = 1e-5
  
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = epochs

    aug_prob = 0.5  
    mixup_alpha = 0.5  
    
    def update_debug_settings(self):
        if self.debug:
            self.epochs = 2
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
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0,
        pad_mode="reflect",
        norm='slaney', # Sử dụng chuẩn hóa Slaney
        htk=True,      # Sử dụng bộ lọc HTK
        center=True,   # Căn giữa cửa sổ
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
        
        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        if 'filepath' not in self.df.columns:
            self.df['filepath'] = self.cfg.train_datadir + '/' + self.df.filename
        
        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

        sample_names = set(self.df['samplename'])
        if self.spectrograms:
            found_samples = sum(1 for name in sample_names if name in self.spectrograms)
            print(f"Found {found_samples} matching spectrograms for {mode} dataset out of {len(self.df)} samples")
        
        if cfg.debug:
            self.df = self.df.sample(min(1000, len(self.df)), random_state=cfg.seed).reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        spec = None

        if self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
        elif not self.cfg.LOAD_DATA:
            spec = process_audio_file(row['filepath'], self.cfg)

        if spec is None:
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            if self.mode == "train":  # Only print warning during training
                print(f"Warning: Spectrogram for {samplename} not found and could not be generated")

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
        """Apply augmentations to spectrogram"""
    
        # Time masking (horizontal stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                width = random.randint(5, 20)
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start:start+width] = 0
        
        # Frequency masking (vertical stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                height = random.randint(5, 20)
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start:start+height, :] = 0
        
        # Random brightness/contrast
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1) 
            
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


class FocalLossBCE(torch.nn.Module):
    def __init__(
            self,
            alpha: float = 0.25,    # Tham số alpha cho Focal Loss (cân bằng giữa các lớp positive/negative)
            gamma: float = 2,       # Tham số gamma cho Focal Loss (điều chỉnh sự tập trung vào các mẫu khó)
            reduction: str = "mean",# Cách tổng hợp loss ("mean", "sum", hoặc "none")
            bce_weight: float = 0.6,  # Trọng số cho thành phần BCEWithLogitsLoss
            focal_weight: float = 1.4, # Trọng số cho thành phần Focal Loss
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # Khởi tạo BCEWithLogitsLoss, nó sẽ nhận logits làm đầu vào
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        # Tính toán Sigmoid Focal Loss từ torchvision.ops
        # Hàm này cũng nhận logits làm đầu vào và tự động áp dụng sigmoid bên trong (tương tự BCEWithLogitsLoss)
        focall_loss_val = torchvision.ops.focal_loss.sigmoid_focal_loss( # Đổi tên biến để tránh nhầm lẫn
            inputs=logits,
            targets=targets, # targets nên ở dạng multi-hot encoding cho bài toán multi-label
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        # Tính toán BCEWithLogitsLoss
        bce_loss_val = self.bce(logits, targets) # Đổi tên biến

        # Kết hợp hai loss lại với các trọng số tương ứng
        combined_loss = self.bce_weight * bce_loss_val + self.focal_weight * focall_loss_val
        return combined_loss

# Hàm tiện ích để tạo instance của FocalLossBCE, có thể dùng với file config (CFG)
def get_criterion(cfg_object=None): # cfg_object là tùy chọn
    if cfg_object:
        # Giả sử các tham số này được định nghĩa trong đối tượng CFG của bạn
        return FocalLossBCE(
            alpha=getattr(cfg_object, 'focal_alpha', 0.25),
            gamma=getattr(cfg_object, 'focal_gamma', 2),
            reduction=getattr(cfg_object, 'loss_reduction', "mean"),
            bce_weight=getattr(cfg_object, 'focal_bce_weight', 0.6),
            focal_weight=getattr(cfg_object, 'focal_focal_weight', 1.4),
        )
    else:
        # Sử dụng giá trị mặc định nếu không có CFG
        return FocalLossBCE()


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
        cfg.num_classes = len(taxonomy_df)
        
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            drop_rate=0.2,
            drop_path_rate=0.2
        )
        
        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
            
        self.feat_dim = backbone_out
        
        self.classifier = nn.Linear(backbone_out, cfg.num_classes)
        
        self.mixup_enabled = hasattr(cfg, 'mixup_alpha') and cfg.mixup_alpha > 0
        if self.mixup_enabled:
            self.mixup_alpha = cfg.mixup_alpha
            
    def forward(self, x, targets=None, criterion=None):
    
        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None
        
        features = self.backbone(x)
        
        if isinstance(features, dict):
            features = features['features']
            
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        
        if self.training and targets is not None:
            if criterion is None:
                raise ValueError("Criterion must be provided during training when targets are present.")

            if self.mixup_enabled and targets_a is not None:
                # Truyền instance `criterion` (FocalLossBCE) vào mixup_criterion
                loss = self.mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                # Nếu không dùng MixUp, hoặc MixUp không được áp dụng
                # Sử dụng `criterion` (FocalLossBCE) trực tiếp
                loss = criterion(logits, targets)
            return logits, loss
        else:
            # Khi không training hoặc không có targets (ví dụ: inference)
            return logits
    
    def mixup_data(self, x, targets):
        """Applies mixup to the data batch"""
        batch_size = x.size(0)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        indices = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[indices]
        
        return mixed_x, targets, targets[indices], lam
    
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
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
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


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    
    model.train()
    losses = []
    all_targets = []
    all_outputs = []
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    
    for step, batch in pbar:
    
        if isinstance(batch['melspec'], list):
            batch_outputs = []
            batch_losses = []
            
            for i in range(len(batch['melspec'])):
                inputs = batch['melspec'][i].unsqueeze(0).to(device)
                target = batch['target'][i].unsqueeze(0).to(device)
                
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, target)
                loss.backward()
                
                batch_outputs.append(output.detach().cpu())
                batch_losses.append(loss.item())
            
            optimizer.step()
            outputs = torch.cat(batch_outputs, dim=0).numpy()
            loss = np.mean(batch_losses)
            targets = batch['target'].numpy()
            
        else:
            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if isinstance(outputs, tuple):
                outputs, loss = outputs  
            else:
                loss = criterion(outputs, targets)
                
            loss.backward()
            optimizer.step()
            
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

def validate(model, loader, criterion, device):
   
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            if isinstance(batch['melspec'], list):
                batch_outputs = []
                batch_losses = []
                
                for i in range(len(batch['melspec'])):
                    inputs = batch['melspec'][i].unsqueeze(0).to(device)
                    target = batch['target'][i].unsqueeze(0).to(device)
                    
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
  
    num_classes = targets.shape[1]
    aucs = []
    
    probs = 1 / (1 + np.exp(-outputs))
    
    for i in range(num_classes):
        
        if np.sum(targets[:, i]) > 0:
            class_auc = roc_auc_score(targets[:, i], probs[:, i])
            aucs.append(class_auc)
    
    return np.mean(aucs) if aucs else 0.0


def run_training(df, cfg):
    """Training function that can either use pre-computed spectrograms or generate them on-the-fly"""

    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    species_ids = taxonomy_df['primary_label'].tolist()
    cfg.num_classes = len(species_ids)
    
    if cfg.debug:
        cfg.update_debug_settings()

    spectrograms = None
    if cfg.LOAD_DATA:
        print("Loading pre-computed mel spectrograms from NPY file...")
        try:
            spectrograms = np.load(cfg.spectrogram_npy, allow_pickle=True).item()
            print(f"Loaded {len(spectrograms)} pre-computed mel spectrograms")
        except Exception as e:
            print(f"Error loading pre-computed spectrograms: {e}")
            print("Will generate spectrograms on-the-fly instead.")
            cfg.LOAD_DATA = False
    
    if not cfg.LOAD_DATA:
        print("Will generate spectrograms on-the-fly during training.")
        if 'filepath' not in df.columns:
            df['filepath'] = cfg.train_datadir + '/' + df.filename
        if 'samplename' not in df.columns:
            df['samplename'] = df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
        
    skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    
    best_scores = []
    # Thêm dictionary để lưu lịch sử loss
    loss_history = {f'fold_{fold}': {'train_loss': [], 'val_loss': []} for fold in cfg.selected_folds}
    
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
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=False, 
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        model = BirdCLEFModel(cfg).to(cfg.device)
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg)
        
        if cfg.scheduler == 'OneCycleLR':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.lr,
                steps_per_epoch=len(train_loader),
                epochs=cfg.epochs,
                pct_start=0.1
            )
        else:
            scheduler = get_scheduler(optimizer, cfg)
        
        best_auc = 0
        best_epoch = 0
        current_best_model_path = None
        
        for epoch in range(cfg.epochs):
            print(f"\nEpoch {epoch+1}/{cfg.epochs}")
            
            train_loss, train_auc = train_one_epoch(
                model, 
                train_loader, 
                optimizer, 
                criterion, 
                cfg.device,
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None
            )
            
            val_loss, val_auc = validate(model, val_loader, criterion, cfg.device)

            if scheduler is not None and not isinstance(scheduler, lr_scheduler.OneCycleLR):
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            # Lưu loss vào lịch sử
            loss_history[f'fold_{fold}']['train_loss'].append(train_loss)
            loss_history[f'fold_{fold}']['val_loss'].append(val_loss)
            
            if val_auc > best_auc:
                print(f"New best AUC: {val_auc:.4f} at epoch {epoch+1}")

                if current_best_model_path is not None and os.path.exists(current_best_model_path):
                     print(f"Deleting previous best model: {current_best_model_path}")
                     try:
                         os.remove(current_best_model_path)
                     except OSError as e:
                         print(f"Error deleting file {current_best_model_path}: {e}")

                new_best_model_filename = f"model_fold{fold}_{val_auc:.4f}.pth"
                new_best_model_path = os.path.join(cfg.OUTPUT_DIR, new_best_model_filename)

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'epoch': epoch + 1,
                    'val_auc': val_auc,
                    'train_auc': train_auc,
                    'cfg': cfg
                }, new_best_model_path)

                current_best_model_path = new_best_model_path
                best_auc = val_auc
                best_epoch = epoch + 1
        
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
    
    return loss_history  # Trả về lịch sử loss để vẽ đồ thị


def plot_loss_history(loss_history, cfg):
    """Vẽ đồ thị train và validation loss qua các epoch cho từng fold"""
    plt.figure(figsize=(15, 8))
    
    # Tạo danh sách màu sắc và kiểu đường để phân biệt các fold
    colors = ['b', 'g', 'r', 'c', 'm']
    linestyles = ['-', '--']
    
    for i, fold in enumerate(cfg.selected_folds):
        fold_key = f'fold_{fold}'
        epochs = range(1, len(loss_history[fold_key]['train_loss']) + 1)
        
        # Vẽ train loss
        plt.plot(
            epochs, 
            loss_history[fold_key]['train_loss'], 
            color=colors[i % len(colors)], 
            linestyle=linestyles[0], 
            label=f'Fold {fold} Train Loss'
        )
        
        # Vẽ validation loss
        plt.plot(
            epochs, 
            loss_history[fold_key]['val_loss'], 
            color=colors[i % len(colors)], 
            linestyle=linestyles[1], 
            label=f'Fold {fold} Val Loss'
        )
    
    plt.title('Train and Validation Loss per Epoch for Each Fold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'loss_plot.png'))  # Lưu đồ thị
    plt.show()


if __name__ == "__main__":
    import time
    
    print("\nLoading training data...")
    train_df = pd.read_csv(cfg.train_csv)
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)

    print("\nStarting training...")
    print(f"LOAD_DATA is set to {cfg.LOAD_DATA}")
    if cfg.LOAD_DATA:
        print("Using pre-computed mel spectrograms from NPY file")
    else:
        print("Will generate spectrograms on-the-fly during training")
    
    # Chạy training và lấy lịch sử loss
    loss_history = run_training(train_df, cfg)
    
    # Vẽ đồ thị loss
    plot_loss_history(loss_history, cfg)
    
    print("\nTraining complete!")

