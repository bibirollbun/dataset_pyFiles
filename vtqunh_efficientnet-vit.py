import os
import pandas as pd
import librosa
import numpy as np
import sys
import torch
import cv2
import math
import time
import logging
import random
import gc
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as AT
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torchvision

import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import seaborn as sns
from sklearn.metrics import roc_auc_score

import timm
from sklearn.model_selection import StratifiedKFold



import warnings
warnings.filterwarnings("ignore")


#!pip install timm


DEBUG_MODE = False

OUTPUT_DIR = '/kaggle/working/'
DATA_ROOT = '/kaggle/input/birdclef-2025'
TRAIN_DIR = '/kaggle/input/birdclef-2025/train_audio'

TAXONOMY_CSV = '/kaggle/input/birdclef-2025/taxonomy.csv'
TRAIN_CSV = '/kaggle/input/birdclef-2025/train.csv'

FS = 32000     # tần số để cắt file ogg
    
# Mel spectrogram parameters
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
FMIN = 50
FMAX = 14000
    
TARGET_DURATION = 5.0
TARGET_SHAPE = (256, 256)  

AUG_PROB = 0.5               # xác suất sẽ augment data cho tập train
    
N_MAX = 50 if DEBUG_MODE else None  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Loading taxonomy data...")
taxonomy_df = pd.read_csv(f'{DATA_ROOT}/taxonomy.csv')
species_class_map = dict(zip(taxonomy_df['primary_label'], taxonomy_df['class_name']))    # dict map tên loài sang class name

# load dữ liệu training
print("Loading training metadata...")
train_df = pd.read_csv(f'{DATA_ROOT}/train.csv')


class BirdCLEFDataset(Dataset):
    # def __init__(self, spectrograms=None, df, taxonomy_csv, target_shape=(128, 313), mode="train"):
    def __init__(self, df, spectrograms=None, mode="train"):
        self.df = df
        self.mode = mode
        self.spectrograms = spectrograms

        # Load taxonomy để tạo label mapping từ id loài sang one hot code
        taxonomy_df = pd.read_csv(TAXONOMY_CSV)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        # thêm cột đường dẫn đến ogg nếu df chưa có
        if 'filepath' not in self.df.columns:
            self.df['filepath'] = TRAIN_DIR + '/' + self.df['filename']

        # thêm cột sample name để lấy kết quả từ npz
        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
                    
        sample_names = set(self.df['samplename'])
        # in ra số lượng spectrogrames lấy được trong npy với các sample name của từng tập train, val
        if self.spectrograms:
            found_samples = sum(1 for name in sample_names if name in self.spectrograms)
            print(f"Found {found_samples} matching spectrograms for {mode} dataset out of {len(self.df)} samples")

    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        spec = self.spectrograms[samplename]

        # nếu ko có spectrogram thì tạo ảnh trắng
        if spec is None:
            spec = np.zeros(TARGET_SHAPE, dtype=np.float32)
            if self.mode == "train":  
                print(f"Warning: Spectrogram for {samplename} not found and could not be generated")

        
        # Add channel dimension: [1, H, W]
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

        # có thể thêm data augmentation cho tập train, aug_prob = 0.5
        if self.mode == "train" and random.random() < AUG_PROB:
            spec = self.apply_spec_augmentations(spec)

        # One-hot encode label
        target = np.zeros(self.num_classes, dtype=np.float32)
        label = row['primary_label']
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0

        # nếu có label thứ 2
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


spectrograms = np.load("/kaggle/input/birdclef-melspec-data/bird_mel_spectrograms.npy", allow_pickle=True).item()

train_meta = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
train_df, val_df = train_test_split(train_meta, test_size=0.2, random_state=42)

train_dataset = BirdCLEFDataset(train_df, spectrograms, mode='train')
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=2,drop_last=True)

val_dataset = BirdCLEFDataset(val_df, spectrograms, mode='val')
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=1,drop_last=True)


class CFG:
    epochs = 8
    batch_size = 32  
    criterion = 'AsymmetricLossMultiLabel'

    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]   

    optimizer = 'AdamW'
    lr = 5e-4 
    weight_decay = 1e-5
  
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = epochs

    LOAD_DATA = True  
    
    def update_debug_settings(self):
        if self.debug:
            self.epochs = 2
            self.selected_folds = [0]



class BirdCLEFModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        #num_classes = len(taxonomy_df)

        self.patch_size = 16
        self.embed_dim = 768
        self.depth = 12
        self.num_heads = 12
        self.spec_height = 256
        self.spec_width = 256
        
        
        # Create Audio Spectrogram Transformer backbone
        # For this we use ViT but adapt it for audio spectrogram input
        self.ast = timm.create_model(
            'vit_base_patch16_224',  # Base ViT model
            pretrained=pretrained,
            img_size=(256, 256),  # Spectrogram dimensions
            in_chans=1,  # Usually 1 for mel spectrograms
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            drop_path_rate=0.2,
            drop_rate=0.3
        )
        
        # Get the output feature dimension
        backbone_out = self.ast.head.in_features
        self.ast.head = nn.Identity()  # Remove the classification head
        
        # Secondary feature extractor - can be a CNN for local features
        self.backbone2 = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            in_chans=1,
            drop_rate=0.3,
            drop_path_rate=0.2
        )
        
        # Get output features for backbone 2
        if 'efficientnet' in 'regnety_008':
            backbone2_out = self.backbone2.classifier.in_features
            self.backbone2.classifier = nn.Identity()
        elif 'resnet' in 'regnety_008':
            backbone2_out = self.backbone2.fc.in_features
            self.backbone2.fc = nn.Identity()
        elif 'convnext' in 'regnety_008':
            backbone2_out = self.backbone2.head.fc.in_features
            self.backbone2.head.fc = nn.Identity()
        else:
            backbone2_out = self.backbone2.get_classifier().in_features
            self.backbone2.reset_classifier(0, '')
        
        # Global pooling for CNN backbone
        self.pooling2 = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimensions
        self.feat_dim1 = backbone_out
        self.feat_dim2 = backbone2_out
        
        # Feature fusion layers (to combine transformer and CNN outputs)
        self.fusion = nn.Sequential(
            nn.Linear(backbone_out + backbone2_out, backbone_out),
            nn.BatchNorm1d(backbone_out),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Classifier head
        self.classifier = nn.Linear(backbone_out, num_classes)
        
        # Mixup and other augmentations
        self.mixup_alpha = 0.5
            
    def forward(self, x, targets=None):
        # Apply mixup if enabled and in training mode
        if self.training and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None
        
        # Extract features from AST
        features1 = self.ast(x)
        
        # Extract features from CNN backbone
        features2 = self.backbone2(x)
        
        # Handle feature maps if necessary for backbone 2
        if len(features2.shape) == 4:
            features2 = self.pooling2(features2)
            features2 = features2.view(features2.size(0), -1)
        
        # Concatenate features from both backbones
        combined_features = torch.cat([features1, features2], dim=1)
        
        # Fuse the features
        fused_features = self.fusion(combined_features)
        
        # Get logits from classifier
        logits = self.classifier(fused_features)
            
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


class AsymmetricLossMultiLabel(nn.Module):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=False,
        reduction="mean",
    ):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        if self.reduction == "mean":
            return -loss.mean()
        if self.reduction == "sum":
            return -loss.sum()

        return -loss


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

def get_criterion(cfg):
    if cfg.criterion == 'AsymmetricLossMultiLabel':
        criterion = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=1,
            clip=0.05,
            eps=1e-8,
            disable_torch_grad_focal_loss=False,
            reduction="mean"
        )
    elif cfg.criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Criterion {cfg.criterion} not implemented")
        
    return criterion


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


num_classes = len(taxonomy_df['primary_label'])


def run_training(df, cfg):
    """Training function that can either use pre-computed spectrograms or generate them on-the-fly"""

    species_ids = taxonomy_df['primary_label'].tolist()
    num_classes = len(species_ids)
    
    if DEBUG_MODE:
        cfg.update_debug_settings()
        
        model = BirdCLEFModel(num_classes=num_classes, pretrained=True).to(device)
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
        
        for epoch in range(cfg.epochs):
            print(f"\nEpoch {epoch+1}/{cfg.epochs}")
            
            train_loss, train_auc = train_one_epoch(
                model, 
                train_loader, 
                optimizer, 
                criterion, 
                device,
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None
            )
            
            val_loss, val_auc = validate(model, val_loader, criterion, device)

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
        
        print(f"{best_auc:.4f} at epoch {best_epoch}")
        
        # Clear memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n" + "="*60)
    print("Cross-Validation Results:")
    #for fold, score in enumerate(best_scores):
        #print(f"Fold {cfg.selected_folds[fold]}: {score:.4f}")
    print("="*60)


import time
cfg = CFG()
run_training(train_df, cfg)
    
print("\nTraining complete!")

