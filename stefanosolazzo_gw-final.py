# G2Net Gravitational Wave Detection
# Final Project Presentation
# Architecture: EfficientNet-B2 + CQT (Bandpass Filtered)

import os
import random
import time
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from scipy import signal
from tqdm.notebook import tqdm

# Suppress warnings for clean output
warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Install necessary libraries silently
print("[INFO] Installing libraries...")
os.system('pip install -q nnAudio timm > /dev/null 2>&1')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp as amp
import timm
from nnAudio.Spectrogram import CQT1992v2

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
class CFG:
    seed = 42
    model_name = 'tf_efficientnet_b2_ns'
    img_size = 256
    batch_size = 64
    epochs = 8 
    subset_size = 60000
    lr = 1e-3
    weight_decay = 1e-4
    n_fold = 5
    # CQT Parameters
    sr = 2048
    fmin = 20
    fmax = 1024
    hop_length = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 2

# -------------------------------------------------------------------------
# UTILS
# -------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_file_path(image_id):
    return "../input/g2net-gravitational-wave-detection/train/{}/{}/{}/{}.npy".format(
        image_id[0], image_id[1], image_id[2], image_id
    )

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

# -------------------------------------------------------------------------
# PREPROCESSING
# -------------------------------------------------------------------------
def apply_bandpass(x, lf=20, hf=500, order=4, sr=2048):
    """Whitening filter to remove low-frequency noise floor."""
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt(1/2048)
    return signal.sosfiltfilt(sos, x) * normalization

# -------------------------------------------------------------------------
# DATASET
# -------------------------------------------------------------------------
class G2NetDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.file_names = df['id'].values
        self.labels = df['target'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = get_file_path(self.file_names[idx])
        waves = np.load(file_path).astype(np.float32)
        
        # Bandpass Filter
        for i in range(3):
            waves[i] = apply_bandpass(waves[i])
            
        # Normalization
        waves = waves / np.max(np.abs(waves), axis=1, keepdims=True)
        
        return torch.tensor(waves, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# -------------------------------------------------------------------------
# MODEL
# -------------------------------------------------------------------------
class G2NetModel(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super(G2NetModel, self).__init__()
        self.cqt = CQT1992v2(
            sr=cfg.sr, fmin=cfg.fmin, fmax=cfg.fmax,
            hop_length=cfg.hop_length,
            output_format="Magnitude", verbose=False
        )
        self.backbone = timm.create_model(
            cfg.model_name, pretrained=pretrained, in_chans=3,
            num_classes=1, drop_rate=0.3, drop_path_rate=0.2
        )

    def forward(self, x):
        bs, ch, time_dim = x.shape
        x = x.view(bs * ch, time_dim)
        x = self.cqt(x)
        x = torch.log1p(x)
        x = x.view(bs, ch, x.size(1), x.size(2))
        
        # Standardize
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        x = (x - mean) / (std + 1e-7)
        
        if x.shape[2] != CFG.img_size:
            x = torch.nn.functional.interpolate(x, size=(CFG.img_size, CFG.img_size), 
                                              mode='bilinear', align_corners=False)
        return self.backbone(x)


# -------------------------------------------------------------------------
# TRAINING LOOPS
# -------------------------------------------------------------------------
def train_epoch(train_loader, model, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0
    preds_list = []
    labels_list = []
    
    pbar = tqdm(train_loader, desc="Training", leave=False, bar_format='{l_bar}{bar:10}{r_bar}')
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with amp.autocast():
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        preds_list.append(torch.sigmoid(outputs).detach().cpu().numpy())
        labels_list.append(labels.detach().cpu().numpy())
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    all_preds = np.concatenate(preds_list)
    all_labels = np.concatenate(labels_list)
    
    return running_loss/len(train_loader), roc_auc_score(all_labels, all_preds)

def validate_epoch(valid_loader, model, criterion, device):
    model.eval()
    running_loss = 0
    preds_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds_list.append(torch.sigmoid(outputs).cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            
    all_preds = np.concatenate(preds_list)
    all_labels = np.concatenate(labels_list)
    
    return running_loss/len(valid_loader), roc_auc_score(all_labels, all_preds), all_preds, all_labels

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------
if __name__ == '__main__':
    set_seed(CFG.seed)
    print(f"[INFO] V6 FINAL: EfficientNet-B2 | Epochs: {CFG.epochs} | Data: {CFG.subset_size}")
    
    df = pd.read_csv("../input/g2net-gravitational-wave-detection/training_labels.csv")
    df_subset = df.sample(n=CFG.subset_size, random_state=CFG.seed).reset_index(drop=True)
    
    skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    
    oof_df = df_subset.copy()
    oof_df['pred_b2'] = 0.0
    
    # List to store results for all folds (to be used in the next cell)
    history_data = [] 

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_subset, df_subset['target'])):
        print(f"\n=== FOLD {fold+1}/{CFG.n_fold} ===")
        
        train_ds = G2NetDataset(df_subset.iloc[train_idx].reset_index(drop=True))
        valid_ds = G2NetDataset(df_subset.iloc[val_idx].reset_index(drop=True))
        
        train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, 
                                num_workers=CFG.num_workers, pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=CFG.batch_size, shuffle=False, 
                                num_workers=CFG.num_workers, pin_memory=True)
        
        model = G2NetModel(CFG).to(CFG.device)
        optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        scaler = amp.GradScaler()
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.lr, 
                                                steps_per_epoch=len(train_loader), epochs=CFG.epochs)
        
        # Local dictionary for this fold
        fold_history = {'t_loss': [], 'v_loss': [], 't_auc': [], 'v_auc': []}
        
        best_auc = 0
        best_preds = None
        
        for epoch in range(CFG.epochs):
            t_loss, t_auc = train_epoch(train_loader, model, nn.BCEWithLogitsLoss(), 
                                      optimizer, scaler, CFG.device)
            v_loss, v_auc, v_preds, _ = validate_epoch(valid_loader, model, 
                                                     nn.BCEWithLogitsLoss(), CFG.device)
            
            # Record metrics
            fold_history['t_loss'].append(t_loss)
            fold_history['v_loss'].append(v_loss)
            fold_history['t_auc'].append(t_auc)
            fold_history['v_auc'].append(v_auc)
            
            # Print exact log format
            print(f"Ep {epoch+1}/{CFG.epochs} | Loss: {t_loss:.4f}/{v_loss:.4f} | AUC: {t_auc:.4f}/{v_auc:.4f}")
            
            if v_auc > best_auc:
                best_auc = v_auc
                best_preds = v_preds
            
            scheduler.step()
            
        # Store predictions
        oof_df.loc[val_idx, 'pred_b2'] = best_preds
        
        # Store history for the Final Visualization Cell
        history_data.append(fold_history)

    print(f"\n=== FINAL RESULTS ===")
    overall_auc = roc_auc_score(df_subset['target'], oof_df['pred_b2'])
    print(f"Global AUC (EfficientNet-B2): {overall_auc:.5f}")
    
    # Save OOF for analysis
    oof_df.to_csv('oof_predictions_b2.csv', index=False)


# -------------------------------------------------------------------------
# FINAL VISUALIZATION & REPORTING 
# -------------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# Configuration for plots
sns.set_style("white")
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.grid'] = False

# -------------------------------------------------------------------------
# 1. RESULTS PER FOLD (GRID LAYOUT) 
# -------------------------------------------------------------------------
def plot_grid_results(histories):
    n_folds = len(histories)
    epochs = range(1, len(histories[0]['t_loss']) + 1)
    colors = sns.color_palette("husl", n_folds)
    
    fig, axes = plt.subplots(2, n_folds, figsize=(20, 8), sharex=True)
    fig.text(0.125, 1.02, 'Results per fold', fontsize=28, fontweight='bold', ha='left')
    
    for i in range(n_folds):
        h = histories[i]
        c = colors[i]
        
        # ROW 1: LOSS
        ax_l = axes[0, i]
        ax_l.plot(epochs, h['t_loss'], label='Train', color=c, marker='o', lw=2)
        ax_l.plot(epochs, h['v_loss'], label='Valid', color='gray', marker='x', ls='--', lw=1.5)
        ax_l.set_title(f'Loss: Fold {i+1}', fontsize=14, fontweight='bold')
        if i == 0: ax_l.set_ylabel('BCE Loss', fontsize=12)
        ax_l.legend(loc='upper right')
        
        # ROW 2: AUC
        ax_a = axes[1, i]
        ax_a.plot(epochs, h['t_auc'], label='Train AUC', color=c, marker='o', lw=2)
        ax_a.plot(epochs, h['v_auc'], label='Valid AUC', color='dimgray', marker='s', lw=2)
        if i == 0: ax_a.set_ylabel('AUC Score', fontsize=12)
        ax_a.set_xlabel('Epoch', fontsize=12)
        ax_a.legend(loc='lower right')
        
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
# 2. HD SPECTROGRAMS 
# -------------------------------------------------------------------------
def plot_candidates_fixed_layout_inferno(dataset, device):
    # Specific Candidates
    target_ids = ['9c13c328bf', '3329ef4849']
    print(f"\n[INFO] Generating Plot: Logica 'Hard Threshold' + Layout 'Clean'...")
    
    # 1. Find indices
    found_indices = []
    for target in target_ids:
        for i in range(len(dataset)):
            if dataset.file_names[i] == target:
                found_indices.append(i)
                break
                
    if len(found_indices) < 2:
        print("! Error: Targets not found in current subset.")
        return

    # 2. Configure CQT for Visualization
    # fmin=20, fmax=500, bins=12 for "blocky" definition
    cqt_layer = CQT1992v2(sr=2048, fmin=20, fmax=500, hop_length=32, 
                          bins_per_octave=12, output_format="Magnitude", 
                          verbose=False).to(device)

    # 3. Plot Setup
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    plt.subplots_adjust(right=0.9, top=0.85)

    for i, idx in enumerate(found_indices):
        wave, label = dataset[idx]
        file_id = dataset.file_names[idx]
        wave_tensor = wave.unsqueeze(0).to(device) # Shape: [1, 3, 4096]

        with torch.no_grad():
            # Flatten channels to process CQT
            spec = cqt_layer(wave_tensor.view(1*3, -1))
            spec_np = spec.cpu().numpy()
            
            # Combine H1 + L1 detectors
            img = spec_np[0] + spec_np[1]

            # Processing
            img = np.log1p(img)
            
            # Median Subtraction
            median_per_row = np.median(img, axis=1, keepdims=True)
            img = img - median_per_row
            
            # Normalization 0-1
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)
            
            # HARD THRESHOLD 0.60
            img[img < 0.60] = 0

            # Plotting
            ax = axes[i]
            im = ax.imshow(img, aspect='auto', origin='lower', cmap='inferno', interpolation='bicubic')
            
            # Titles & Axes
            ax.set_title(f"ID: {file_id}\nSource: LIGO Hanford + Livingston (Combined)", 
                         fontsize=16, fontweight='bold', pad=10)
            ax.set_ylabel("Frequency (Low to High)", fontsize=12)
            ax.set_xlabel("Time steps", fontsize=12)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Normalized Amplitude (Threshold > 0.60)', fontsize=12)

    plt.suptitle("Gravitational Wave Signal Reconstruction (Deep Learning)", fontsize=22, y=1.05)
    plt.show()

# -------------------------------------------------------------------------
# 3. GLOBAL ROC CURVE (SHADED) 
# -------------------------------------------------------------------------
def plot_final_roc_shaded(df):
    y_true = df['target']
    y_pred = df['pred_b2']
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    score = roc_auc_score(y_true, y_pred)
    
    plt.figure(figsize=(9, 7))
    plt.plot(fpr, tpr, color='#8B0000', lw=3, label=f'ROC Curve (AUC = {score:.4f})')
    plt.fill_between(fpr, tpr, color='#8B0000', alpha=0.1, label='AUC Area')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (Global)', fontsize=20, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=14)
    sns.despine()
    plt.show()

# -------------------------------------------------------------------------
# 4. GLOBAL CONFUSION MATRIX (FIXED LAYOUT)
# -------------------------------------------------------------------------
def plot_global_confusion_matrix_fixed(df):
    y_true = df['target']
    y_pred_prob = df['pred_b2']
    
    # Threshold 0.5
    y_pred_binary = (y_pred_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / len(y_true)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    fig, ax = plt.subplots(figsize=(8, 7.5))
    sns.set_style("white")
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, 
                annot_kws={"size": 14, "weight": "bold"}, linewidths=2, linecolor='white')
    
    plt.title('Global Confusion Matrix (Threshold 0.5)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    
    # Bottom spacing
    plt.subplots_adjust(bottom=0.25)
    
    stats_text = (f"Accuracy: {accuracy:.4f}\n"
                  f"Sensitivity (Recall): {sensitivity:.4f}\n"
                  f"Specificity: {specificity:.4f}")
    
    plt.figtext(0.5, 0.08, stats_text, ha="center", fontsize=14, 
                bbox={"facecolor":"orange", "alpha":0.1, "pad":10, "edgecolor":"orange"})
    plt.show()

# -------------------------------------------------------------------------
# EXECUTE VISUALIZATIONS
# -------------------------------------------------------------------------
if 'history_data' in globals() and 'valid_ds' in globals() and 'oof_df' in globals():
    print("Generating Grid Results per Fold...")
    plot_grid_results(history_data)
    
    plot_candidates_fixed_layout_inferno(valid_ds, CFG.device)
    
    print("\nGenerating ROC Curve...")
    plot_final_roc_shaded(oof_df)
    
    print("\nGenerating Global Confusion Matrix...")
    plot_global_confusion_matrix_fixed(oof_df)
else:
    print("! Variables not found. Please run the training cell first.")

