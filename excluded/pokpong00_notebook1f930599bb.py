# **BirdCLEF 2025 Training Notebook**

# This is a baseline training pipeline for BirdCLEF 2025 using EfficientNetB0 with PyTorch and Timm.
# Modifications include precision-recall curves, micro/macro metrics, and enhanced visualizations for final results.

import os
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
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import timm

warnings.filterwarnings("ignore")

## Configuration
class CFG:
    seed = 42
    debug = False
    apex = False
    print_freq = 100
    num_workers = 4
    
    OUTPUT_DIR = '/kaggle/working/'
    train_datadir = '/kaggle/input/birdclef-2025/train_audio'
    train_csv = '/kaggle/input/birdclef-2025/train.csv'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    spectrogram_npy = '/kaggle/input/birdclef25-mel-spectrograms/birdclef2025_melspec_5sec_256_256.npy'
 
    model_name = 'efficientnet_b0'
    pretrained = True
    in_channels = 1

    LOAD_DATA = True
    FS = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 12
    batch_size = 32
    criterion = 'FocalLoss'

    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]

    optimizer = 'AdamW'
    lr = 3e-4
    weight_decay = 1e-5
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = 12

    aug_prob = 0.9
    mixup_alpha = 1.5
    
    def update_debug_settings(self):
        if self.debug:
            self.epochs = 2
            self.selected_folds = [0]

cfg = CFG()

## Utilities
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

## Pre-processing
def audio2melspec(audio_data, cfg):
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
        power=2.0
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return mel_spec_norm

def process_audio_file(audio_path, cfg, start_time=0):
    try:
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS, offset=start_time, duration=cfg.TARGET_DURATION)
        target_samples = int(cfg.TARGET_DURATION * cfg.FS)
        if len(audio_data) < target_samples:
            audio_data = np.pad(audio_data, (0, target_samples - len(audio_data)), mode='constant')
        mel_spec = audio2melspec(audio_data, cfg)
        if mel_spec.shape != cfg.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
        return mel_spec.astype(np.float32)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def generate_spectrograms(df, cfg):
    print("Generating mel spectrograms...")
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
            else:
                errors.append((filepath, "Failed to generate spectrogram"))
        except Exception as e:
            print(f"Error processing {row.filepath}: {e}")
            errors.append((row.filepath, str(e)))
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Successfully processed {len(all_bird_data)} files out of {len(df)}")
    print(f"Failed to process {len(errors)} files")
    return all_bird_data

## Dataset Preparation
class BirdCLEFDatasetFromNPY(Dataset):
    def __init__(self, df, cfg, spectrograms=None, mode="train"):
        self.df = df.reset_index(drop=True)
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
        print(f"Dataset size: {len(self.df)} samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if idx >= len(self.df):
            print(f"Index {idx} out of bounds for dataset of size {len(self.df)}")
            return None
        row = self.df.iloc[idx]
        samplename = row['samplename']
        spec = None
        if self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
        elif not self.cfg.LOAD_DATA:
            spec = process_audio_file(row['filepath'], self.cfg)
        if spec is None:
            print(f"Spectrogram for {samplename} not found, returning default")
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
        if spec.ndim != 2:
            print(f"Invalid spectrogram shape for {samplename}: {spec.shape}")
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
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
        if not isinstance(spec, torch.Tensor) or spec.ndim != 3:
            print(f"Invalid spec shape for augmentation: {spec.shape}")
            return spec
        try:
            if random.random() < 0.6:
                num_masks = random.randint(1, 5)
                for _ in range(num_masks):
                    width = random.randint(5, 25)
                    start = random.randint(0, spec.shape[2] - width)
                    spec[:, :, start:start+width] = 0
            if random.random() < 0.6:
                num_masks = random.randint(1, 5)
                for _ in range(num_masks):
                    height = random.randint(5, 25)
                    start = random.randint(0, spec.shape[1] - height)
                    spec[:, start:start+height, :] = 0
            if random.random() < 0.6:
                gain = random.uniform(0.7, 1.3)
                bias = random.uniform(-0.15, 0.15)
                spec = spec * gain + bias
                spec = torch.clamp(spec, 0, 1)
            if random.random() < 0.4:
                noise = torch.randn_like(spec) * 0.15
                spec = spec + noise
                spec = torch.clamp(spec, 0, 1)
        except Exception as e:
            print(f"Error in spec augmentation: {e}")
        return spec
    
    def encode_label(self, label):
        target = np.zeros(self.num_classes)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target

def collate_fn(batch):
    batch = [item for item in batch if item is not None and item['melspec'].shape == (1, 256, 256)]
    if len(batch) == 0:
        print("Empty batch after filtering")
        return {}
    result = {key: [] for key in batch[0].keys()}
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    for key in result:
        try:
            if key == 'target':
                result[key] = torch.stack(result[key])
            elif key == 'melspec':
                result[key] = torch.stack(result[key])
        except Exception as e:
            print(f"Error stacking {key}: {e}")
            return {}
    return result

## Model Definition
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
            drop_rate=0.6,
            drop_path_rate=0.6
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
    
    def forward(self, x, targets=None):
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
        if self.training and self.mixup_enabled and targets is not None:
            loss = self.mixup_criterion(F.binary_cross_entropy_with_logits, 
                                       logits, targets_a, targets_b, lam)
            return logits, loss
        return logits
    
    def mixup_data(self, x, targets):
        batch_size = x.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        indices = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[indices]
        return mixed_x, targets, targets[indices], lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

## Training Utilities
def get_optimizer(model, cfg):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    return optimizer

def get_scheduler(optimizer, cfg):
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.T_max,
        eta_min=cfg.min_lr
    )
    return scheduler

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=3.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def get_criterion(cfg):
    criterion = FocalLoss(alpha=1, gamma=3.5)
    return criterion

def find_optimal_threshold_per_class(targets, probs):
    thresholds = np.arange(0.05, 0.95, 0.05)
    best_thresholds = []
    for i in range(targets.shape[1]):
        best_f1 = 0
        best_thresh = 0.3
        for thresh in thresholds:
            preds = (probs[:, i] > thresh).astype(int)
            f1 = f1_score(targets[:, i], preds, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        best_thresholds.append(best_thresh)
    return best_thresholds

## Training and Validation Loops
def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    losses = []
    all_targets = []
    all_outputs = []
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    
    for step, batch in pbar:
        if not batch:
            continue
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
        losses.append(loss.item())
        
        pbar.set_postfix({
            'train_loss': np.mean(losses[-10:]) if losses else 0,
            'lr': optimizer.param_groups[0]['lr']
        })
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    probs = 1 / (1 + np.exp(-all_outputs))
    thresholds = find_optimal_threshold_per_class(all_targets, probs)
    y_pred = np.zeros_like(probs)
    for i, thresh in enumerate(thresholds):
        y_pred[:, i] = (probs[:, i] > thresh).astype(int)
    f1 = f1_score(all_targets, y_pred, average='macro')
    avg_loss = np.mean(losses)
    
    return avg_loss, auc, f1, thresholds

def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            if not batch:
                continue
            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            
            all_outputs.append(outputs)
            all_targets.append(targets)
            losses.append(loss.item())
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    auc = calculate_auc(all_targets, all_outputs)
    probs = 1 / (1 + np.exp(-all_outputs))
    thresholds = find_optimal_threshold_per_class(all_targets, probs)
    y_pred = np.zeros_like(probs)
    for i, thresh in enumerate(thresholds):
        y_pred[:, i] = (probs[:, i] > thresh).astype(int)
    f1 = f1_score(all_targets, y_pred, average='macro')
    avg_loss = np.mean(losses)
    
    return avg_loss, auc, f1, thresholds, all_outputs, all_targets

def calculate_auc(targets, outputs):
    num_classes = targets.shape[1]
    aucs = []
    probs = 1 / (1 + np.exp(-outputs))
    for i in range(num_classes):
        if np.sum(targets[:, i]) > 0:
            class_auc = roc_auc_score(targets[:, i], probs[:, i])
            aucs.append(class_auc)
    return np.mean(aucs) if aucs else 0.0

## Visualization Functions
def plot_roc_curves(fold_results, thresholds_per_fold):
    plt.figure(figsize=(8, 6))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for i, r in enumerate(fold_results):
        fpr, tpr, _ = roc_curve(r['y_true'].ravel(), r['y_score'].ravel())
        plt.plot(fpr, tpr, lw=1, alpha=0.6, label=f'Fold {i} (AUC={auc(fpr, tpr):.3f})')
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # Mark threshold point (average threshold across classes for visualization)
        avg_threshold = np.mean(thresholds_per_fold[i])
        thresh_idx = np.argmin(np.abs(fpr - avg_threshold))
        plt.scatter(fpr[thresh_idx], tpr[thresh_idx], marker='o', s=100, label=f'Fold {i} Threshold')
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, lw=2, color='navy', label=f'Mean ROC (AUC={mean_auc:.3f})')
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='grey', alpha=0.2)
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves — 5-Fold CV', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'roc_curves.png'))
    plt.show()

def plot_precision_recall_curves(fold_results, species_ids):
    num_classes = len(species_ids)
    cols = 3
    rows = math.ceil(num_classes / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), constrained_layout=True)
    axes = axes.ravel() if num_classes > 1 else [axes]

    for idx, species in enumerate(species_ids):
        mean_precision = np.linspace(0, 1, 100)
        recalls = []
        for r in fold_results:
            precision, recall, _ = precision_recall_curve(r['y_true'][:, idx], r['y_score'][:, idx])
            recalls.append(np.interp(mean_precision, precision[::-1], recall[::-1]))
        mean_recall = np.mean(recalls, axis=0)
        std_recall = np.std(recalls, axis=0)
        axes[idx].plot(mean_precision, mean_recall, lw=2, label=f'{species} (Mean)')
        axes[idx].fill_between(mean_precision, mean_recall - std_recall, mean_recall + std_recall, alpha=0.2)
        axes[idx].set_title(f'Precision-Recall: {species}', fontsize=10)
        axes[idx].set_xlabel('Precision', fontsize=8)
        axes[idx].set_ylabel('Recall', fontsize=8)
        axes[idx].legend(loc='lower left', fontsize=8)
        axes[idx].grid(True, linestyle='--', alpha=0.7)

    for idx in range(len(species_ids), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Precision-Recall Curves per Class', fontsize=14)
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'precision_recall_curves.png'))
    plt.show()

def plot_confusion_and_classic_metrics(fold_results):
    sens_list, spec_list, prec_list, f1_list, acc_list = [], [], [], [], []
    for i, r in enumerate(fold_results):
        cm = confusion_matrix(r['y_true'].ravel(), r['y_pred'].ravel())
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if tp + fn > 0 else 0
        spec = tn / (tn + fp) if tn + fp > 0 else 0
        prec, recall, f1, _ = precision_recall_fscore_support(
            r['y_true'].ravel(), r['y_pred'].ravel(), average='binary', zero_division=0)
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        print(f"\n––– Fold {i} Metrics –––")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Accuracy: {acc:.3f}, Sensitivity: {sens:.3f}, Specificity: {spec:.3f}")
        print(f"Precision: {prec:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        sens_list.append(sens)
        spec_list.append(spec)
        prec_list.append(prec)
        f1_list.append(f1)
        acc_list.append(acc)
    
    print("\n––– Average Over Folds –––")
    print(f"Accuracy: {np.mean(acc_list):.3f} ± {np.std(acc_list):.3f}")
    print(f"Sensitivity: {np.mean(sens_list):.3f} ± {np.std(sens_list):.3f}")
    print(f"Specificity: {np.mean(spec_list):.3f} ± {np.std(spec_list):.3f}")
    print(f"Precision: {np.mean(prec_list):.3f} ± {np.std(prec_list):.3f}")
    print(f"F1 Score: {np.mean(f1_list):.3f} ± {np.std(f1_list):.3f}")

def compute_per_class_metrics(fold_results, species_ids):
    per_class_metrics = []
    for idx, species in enumerate(species_ids):
        precisions, recalls = [], []
        for r in fold_results:
            prec, rec, _, _ = precision_recall_fscore_support(
                r['y_true'][:, idx], r['y_pred'][:, idx], average='binary', zero_division=0)
            precisions.append(prec)
            recalls.append(rec)
        per_class_metrics.append({
            'species': species,
            'precision': np.mean(precisions),
            'recall': np.mean(recalls)
        })
        print(f"Species: {species}, Precision: {np.mean(precisions):.4f}, Recall: {np.mean(recalls):.4f}")
    return pd.DataFrame(per_class_metrics)

def compute_micro_macro_metrics(fold_results, species_ids):
    micro_precisions, micro_recalls = [], []
    macro_precisions, macro_recalls = [], []
    
    for r in fold_results:
        prec, rec, _, _ = precision_recall_fscore_support(
            r['y_true'].ravel(), r['y_pred'].ravel(), average='micro', zero_division=0)
        micro_precisions.append(prec)
        micro_recalls.append(rec)
        prec, rec, _, _ = precision_recall_fscore_support(
            r['y_true'].ravel(), r['y_pred'].ravel(), average='macro', zero_division=0)
        macro_precisions.append(prec)
        macro_recalls.append(rec)
    
    print("\n––– Micro and Macro Averages –––")
    print(f"Micro Precision: {np.mean(micro_precisions):.4f} ± {np.std(micro_precisions):.4f}")
    print(f"Micro Recall: {np.mean(micro_recalls):.4f} ± {np.std(micro_recalls):.4f}")
    print(f"Macro Precision: {np.mean(macro_precisions):.4f} ± {np.std(macro_precisions):.4f}")
    print(f"Macro Recall: {np.mean(macro_recalls):.4f} ± {np.std(macro_recalls):.4f}")

## Training Loop
def run_training(df, cfg):
    print(f"→ 5-Fold Stratified CV: n_splits={cfg.n_fold}, shuffle=True, seed={cfg.seed}")
    
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    species_ids = taxonomy_df['primary_label'].tolist()
    cfg.num_classes = len(species_ids)
    
    if cfg.debug:
        cfg.update_debug_settings()
    
    spectrograms = None
    if cfg.LOAD_DATA:
        print("Loading pre-computed spectrograms...")
        try:
            spectrograms = np.load(cfg.spectrogram_npy, allow_pickle=True).item()
            print(f"Loaded {len(spectrograms)} spectrograms")
        except Exception as e:
            print(f"Error loading spectrograms: {e}")
            cfg.LOAD_DATA = False
    
    if not cfg.LOAD_DATA:
        if 'filepath' not in df.columns:
            df['filepath'] = cfg.train_datadir + '/' + df.filename
        if 'samplename' not in df.columns:
            df['samplename'] = df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
        spectrograms = generate_spectrograms(df, cfg)
    
    best_scores = []
    all_fold_results = []
    thresholds_per_fold = []
    
    for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed).split(df, df['primary_label'])):
        if fold not in cfg.selected_folds:
            continue
        
        print(f'\n{"="*30} Fold {fold} {"="*30}')
        
        fold_train_df = df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Training set: {len(fold_train_df)} samples")
        print(f"Validation set: {len(fold_val_df)} samples")
        
        train_dataset = BirdCLEFDatasetFromNPY(fold_train_df, cfg, spectrograms, mode='train')
        val_dataset = BirdCLEFDatasetFromNPY(fold_val_df, cfg, spectrograms, mode='valid')
        
        class_weights = fold_train_df['primary_label'].value_counts().map(lambda x: 1/(x+1e-6)**0.8).to_dict()
        sampler_weights = [class_weights[row['primary_label']] for _, row in fold_train_df.iterrows()]
        if len(sampler_weights) != len(train_dataset):
            print(f"Sampler weights size {len(sampler_weights)} does not match dataset size {len(train_dataset)}")
            raise ValueError("Sampler weights mismatch")
        sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights), replacement=True)
        
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, sampler=sampler, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn)
        
        model = BirdCLEFModel(cfg).to(cfg.device)
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg)
        scheduler = get_scheduler(optimizer, cfg)
        
        best_f1 = 0
        best_thresholds = [0.3] * cfg.num_classes
        best_epoch = 0
        patience = 10
        counter = 0
        
        for epoch in range(cfg.epochs):
            print(f"\nEpoch {epoch+1}/{cfg.epochs}")
            
            train_loss, train_auc, train_f1, train_thresholds = train_one_epoch(
                model, train_loader, optimizer, criterion, cfg.device)
            
            val_loss, val_auc, val_f1, val_thresholds, all_outputs, all_targets = validate(
                model, val_loader, criterion, cfg.device)
            
            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}")
            
            all_fold_results.append({
                'y_true': all_targets.astype(int),
                'y_score': 1 / (1 + np.exp(-all_outputs)),
                'y_pred': np.zeros_like(all_outputs),
                'fold': fold
            })
            for i, thresh in enumerate(val_thresholds):
                all_fold_results[-1]['y_pred'][:, i] = (all_fold_results[-1]['y_score'][:, i] > thresh).astype(int)
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_thresholds = val_thresholds
                best_epoch = epoch + 1
                print(f"New best F1: {best_f1:.4f} at epoch {best_epoch}")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_f1': val_f1,
                    'val_thresholds': val_thresholds
                }, f"model_fold{fold}.pth")
                counter = 0
            else:
                counter += 1
            
            if scheduler is not None:
                scheduler.step()
            
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        best_scores.append(best_f1)
        thresholds_per_fold.append(best_thresholds)
        print(f"\nBest F1 for fold {fold}: {best_f1:.4f} at epoch {best_epoch}")
        
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n" + "="*60)
    print("Cross-Validation Results:")
    for fold, score in enumerate(best_scores):
        print(f"Fold {fold}: {score:.4f}")
    print(f"Mean F1: {np.mean(best_scores):.4f}")
    print("="*60)
    
    # Plot ROC curves with thresholds
    plot_roc_curves(all_fold_results, thresholds_per_fold)
    
    # Plot precision-recall curves
    plot_precision_recall_curves(all_fold_results, species_ids)
    
    # Confusion matrix and classic metrics
    plot_confusion_and_classic_metrics(all_fold_results)
    
    # Per-class metrics
    print("\nPer-Class Metrics on Validation Set:")
    per_class_df = compute_per_class_metrics(all_fold_results, species_ids)
    per_class_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'per_class_metrics.csv'), index=False)
    
    # Micro and macro metrics
    compute_micro_macro_metrics(all_fold_results, species_ids)

if __name__ == "__main__":
    print("\nLoading training data...")
    train_df = pd.read_csv(cfg.train_csv)
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)

    print("\nStarting training...")
    print(f"LOAD_DATA is set to {cfg.LOAD_DATA}")
    if cfg.LOAD_DATA:
        print("Using pre-computed mel spectrograms from NPY file")
    else:
        print("Will generate spectrograms on-the-fly")
    
    run_training(train_df, cfg)
    
    print("\nTraining complete!")

