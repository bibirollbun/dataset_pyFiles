import os
import gc
import numpy as np
import pandas as pd
import copy
from tqdm.auto import tqdm
import math
import random

# EEG Processing
import mne
from scipy.signal import butter, lfilter

# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchinfo import summary
import torchvision.transforms as T # Added for resizing
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter

warnings.filterwarnings('ignore')


!cp /kaggle/input/2nd-place-solution/x3d.py .
from x3d import create_x3d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Configuration
CFG = {
    'data_dir': '/kaggle/input/hms-harmful-brain-activity-classification',
    'train_csv': '/kaggle/input/hms-harmful-brain-activity-classification/train.csv',
    'train_eeg_dir': '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs',
    'processed_eeg_dir_50': '/kaggle/input/eeg-spec-61-75-50/processed_eeg_spectrograms_61_75_middle_50',
    'processed_eeg_dir_10': '/kaggle/input/eeg-spec-61-75-10/processed_eeg_spectrograms_61_75_middle_10',
    'model_path': '/kaggle/input/eeg_3d_cnn_torch/pytorch/default/1/x3d_model_fold0_best.pth',
    'use_saved_model': False,
    # Add other paths if needed (like for spectrograms if you extend the model)

    'seed': 42,
    'num_folds': 5,
    'selected_fold': 0, # Train only one fold in this example
    'epochs': 10, # Adjust number of epochs
    'batch_size': 16, # Adjust based on GPU memory
    'num_workers': 2,
    'lr': 5e-5,
    'weight_decay': 8e-4,
    'patience': 3, # For early stopping (optional)

    # Model Params
    'model_name': 'x3d_m',
    'x3d_input_clip_length': 16, # Matches the 16 differential EEG channels after montage
    'x3d_input_crop_size': 224, # 312 for x3d-l. Target H, W for the X3D model input
    'x3d_depth_factor': 5.0, #5.0 for x3d-l
    'num_classes': 6,
    'target_cols': ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote'],
    'class_names': ['Other', 'Seizure', 'LPD', 'GPD', 'GRDA', 'LRDA'],
    'use_amp': True
}

# Set Seed for Reproducibility
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['seed'])


df = pd.read_csv(CFG['train_csv'])
TARGETS = df.columns[-6:]

train = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_id':'first','spectrogram_label_offset_seconds':'min'})
train.columns = ['spec_id','min']

tmp = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_label_offset_seconds':'max'})
train['max'] = tmp

tmp = df.groupby('eeg_id')[['patient_id']].agg('first')
train['patient_id'] = tmp

tmp = df.groupby('eeg_id')[TARGETS].agg('sum')
for t in TARGETS:
    train[t] = tmp[t].values
    
y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1,keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby('eeg_id')[TARGETS].agg('sum')
sum_targets = tmp.sum(axis=1)
max_vote_percentage = tmp.max(axis=1) / sum_targets
train['max_vote_percentage'] = max_vote_percentage


tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
train['target'] = tmp

train = train.reset_index()
train = train[train['max_vote_percentage']>=.75]
train.head(3)


frequency = train['target'].value_counts()
print(frequency)


class AlaskaDataIter(Dataset):
    def __init__(self, df, cfg, training_flag=True, flip=False):
        self.df = df
        self.cfg = cfg
        self.training_flag = training_flag
        self.transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(10)
        ])
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.cfg['class_names'])}

    def __len__(self):
        return len(self.df)   

    def __getitem__(self, item):
        dp = self.df.iloc[item]
        
        eeg_id = dp['eeg_id']
        
        # Load both spectrograms
        spec_50_path = os.path.join(self.cfg['processed_eeg_dir_50'], f"{eeg_id}.pt")
        # spec_10_path = os.path.join(self.cfg['processed_eeg_dir_10'], f"{eeg_id}.pt")
        
        spec_50 = torch.load(spec_50_path).float() #[16, 224, 224]
        # spec_10 = torch.load(spec_10_path).float() #[16, 224, 224]
        
        if self.training_flag and random.random() > 0.5:
            spec_50 = self.transform(spec_50)
            # spec_10 = self.transform(spec_10)           

        targets = dp[TARGETS].values.astype(np.float32)
        # targets /= targets.sum() # Normalize
        targets_tensor = torch.from_numpy(targets)

        true_class_name = dp['target'] # This is 'expert_consensus' string
        true_class_idx = self.class_to_idx[true_class_name]
        true_class_idx_tensor = torch.tensor(true_class_idx, dtype=torch.long)

        return spec_50, targets_tensor, true_class_idx_tensor

# === Model Definition ===
class Modelx3d(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        try:
             # Create X3D, potentially loading pretrained weights if available/desired
             # NOTE: Check `create_x3d` documentation for pretrained weight options
            self.net = create_x3d(
                input_clip_length=cfg['x3d_input_clip_length'], # Corresponds to EEG channels (T)
                input_crop_size=cfg['x3d_input_crop_size'],   # Corresponds to Freq (H) and Time (W) after resize
                depth_factor=cfg['x3d_depth_factor']
            )
            # Modify final layers to use as feature extractor
            # Check the actual structure of your X3D model using print(self.net)
            # The layer names ('blocks', 'proj', etc.) might differ slightly
            self.net.blocks[5].dropout = nn.Identity()
            self.net.blocks[5].proj = nn.Identity()
            self.net.blocks[5].activation = nn.Identity()
            self.net.blocks[5].output_pool = nn.Identity() # Adjust if pooling name is different
            self.feature_dim = 2048 # Output dim of X3D_L feature extractor (verify this)

        except Exception as e:
            print(f"Error creating X3D model: {e}")
            print("Ensure 'create_x3d' parameters match the implementation in x3d.py")
            raise e

    def forward(self, x):
        x = self.net(x)
        return x

class Netx3dTrain(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg        

        # X3D Feature Extractor Backbone
        self.model_backbone = Modelx3d(cfg)
        
        # self.fc = nn.Linear(self.model_backbone.feature_dim, cfg['num_classes'])
        self.fc = nn.Sequential(nn.Dropout(0.3),
                                nn.Linear(self.model_backbone.feature_dim, cfg['num_classes'], bias=True)
                               )

    def forward(self, spec_50):
        bs = spec_50.size(0)
        
        # x = torch.cat([spec_10, spec_50], dim=1)
        
        x = torch.unsqueeze(spec_50, dim=1)       
        
        x = torch.cat([x, x, x], dim=1)        

        features = self.model_backbone(x)
        
        features = features.view(bs, -1)

        output = self.fc(features)

        return output

# === Training Utilities ===

def train_epoch(model, loader, optimizer, criterion, device, scaler=None, use_amp=False):
    model.train()
    train_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (spec_50, targets, _) in enumerate(pbar):
        # spec_10 = spec_10.to(device)
        spec_50 = spec_50.to(device)
        targets = targets.to(device) # Target shape [B, 6]

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            logits = model(spec_50)
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = criterion(log_probs, targets) # Assumes KLDivLoss

        # <<< Scale loss and backpropagate using scaler >>>
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()        

        train_loss += loss.item()
        all_preds.append(torch.softmax(logits, dim=-1).detach().cpu().numpy()) # Store probabilities
        all_targets.append(targets.cpu().numpy())

        pbar.set_postfix(loss=loss.item())
        if batch_idx % 10 == 0:
            print(f"Batch [{batch_idx}/{len(loader)}], Loss: {loss.item():.4f}")

    avg_loss = train_loss / len(loader)
    predictions = np.concatenate(all_preds)
    true_labels = np.concatenate(all_targets)

    # Calculate Accuracy (using argmax for simplicity, KLDiv is the competition metric)
    acc = accuracy_score(np.argmax(true_labels, axis=1), np.argmax(predictions, axis=1))

    return avg_loss, acc

def validate_epoch(model, loader, criterion, device, use_amp=False):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for spec_50, targets, _ in pbar:
            # spec_10 = spec_10.to(device)
            spec_50 = spec_50.to(device)
            targets = targets.to(device)

            with autocast(enabled=use_amp):
                logits = model(spec_50)
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = criterion(log_probs, targets)

            val_loss += loss.item()
            all_preds.append(torch.softmax(logits, dim=-1).cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            pbar.set_postfix(loss=loss.item())

    avg_loss = val_loss / len(loader)
    predictions = np.concatenate(all_preds)
    true_labels = np.concatenate(all_targets)

    # Calculate Accuracy
    acc = accuracy_score(np.argmax(true_labels, axis=1), np.argmax(predictions, axis=1))

    return avg_loss, acc

def run_training(cfg, train_df, val_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = AlaskaDataIter(train_df, cfg, training_flag=True)
    val_dataset = AlaskaDataIter(val_df, cfg, training_flag=False) # No augmentation for validation

    print("\nActual class distribution in validation dataset:")
    
    all_true_class_indices = []
    for i in range(len(val_dataset)):
        _, _, true_class_idx = val_dataset[i]
        all_true_class_indices.append(true_class_idx.item())
    
    target_counts = Counter(all_true_class_indices)
    for class_idx_enum, class_name_enum in enumerate(cfg['class_names']):
        print(f"{class_name_enum}: {target_counts.get(class_idx_enum, 0)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, 
                              num_workers=cfg['num_workers'], pin_memory=True, drop_last=True,
                              persistent_workers=True, prefetch_factor=2 )
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'] * 2, shuffle=False, 
                            num_workers=cfg['num_workers'], pin_memory=True, 
                            persistent_workers=True, prefetch_factor=2 )

    # Initialize model
    model = Netx3dTrain(cfg).to(device)
    
    # if cfg['use_saved_model']:
    #     model.load_state_dict(torch.load(cfg['model_path'], map_location=device))
    # else:
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    
    criterion = nn.KLDivLoss(reduction='batchmean')

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-6)
    
    scaler = GradScaler(enabled=cfg['use_amp'])
    
    best_val_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(cfg['epochs']):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler, use_amp=cfg['use_amp'])
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, use_amp=cfg['use_amp']) # Add other metrics if calculated

        if scheduler:
            scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}") # Print other metrics

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            model_path = f"x3d_model_fold{cfg['selected_fold']}_best.pth"
            torch.save(model.state_dict(), model_path)
            cfg['model_path'] = model_path
            print(f"Validation loss improved. Saved model to {model_path}")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

        # Early stopping (optional)
        if cfg['patience'] > 0 and epochs_no_improve >= cfg['patience']:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nTraining finished. Best validation loss {best_val_loss:.4f} at epoch {best_epoch+1}")
    # Load best model for potential further use/inference
    # model.load_state_dict(torch.load(f"x3d_model_fold{cfg['selected_fold']}_best.pth"))
    
    
    # model.load_state_dict(torch.load(cfg['model_path'], map_location=device))
    model.eval()
    
    # Prepare the validation dataset and dataloader
    val_dataset = AlaskaDataIter(val_df, cfg, training_flag=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'] * 2, shuffle=False, 
                            num_workers=cfg['num_workers'], pin_memory=True)

    # Collect predictions and true labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for spec_50, _, true_class_indices in val_loader:
            # spec_10 = spec_10.to(device)
            spec_50 = spec_50.to(device)
            true_class_indices = true_class_indices.to(device)

            logits = model(spec_50)
            preds = torch.argmax(logits, dim=1)  # Get predicted class indices

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(true_class_indices.cpu().numpy())  # Convert one-hot to class indices

    # classification report
    print(classification_report(all_labels, all_preds, target_names=cfg['class_names'], zero_division=0))

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=cfg['class_names'], 
                yticklabels=cfg['class_names'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    return model


if __name__ == '__main__':
    print("Starting Training Script")
    print("Loading train.csv...")
    
    # --- Data Splitting ---
    print(f"Setting up Fold {CFG['selected_fold']}...")
    gkf = GroupKFold(n_splits=CFG['num_folds'])
    splits = list(gkf.split(X=train, y=train[TARGETS], groups=train['patient_id'])) # Split based on first target, grouped by patient

    train_idx, val_idx = splits[CFG['selected_fold']]
    train_fold_df = train.iloc[train_idx].reset_index(drop=True)
    val_fold_df = train.iloc[val_idx].reset_index(drop=True)

    print(f"Train fold {CFG['selected_fold']} size: {len(train_fold_df)}")
    print(f"Validation fold {CFG['selected_fold']} size: {len(val_fold_df)}")

    # --- Run Training ---
    print("Starting training run...")
    trained_model = run_training(CFG, train_fold_df, val_fold_df)
    print("Training run complete.")


# logits = model(eeg) # Logits shape [B, 6]

        # CrossEntropyLoss expects logits [B, C] and target indices [B]
        # Our targets are probabilities [B, C]. Use KLDivLoss or convert targets.
        # Option 1: KLDivLoss (requires log_softmax output from model)
        # loss = criterion(torch.log_softmax(logits, dim=-1), targets)

        # Option 2: Convert targets to class indices for CrossEntropyLoss
        # This only works if one class is clearly dominant (not ideal for vote distributions)
        # target_indices = torch.argmax(targets, dim=1)
        # loss = criterion(logits, target_indices)

        # Option 3: Use CrossEntropyLoss with soft labels (probabilities)
        # This is equivalent to KLDivLoss between softmax(logits) and targets
        # log_probs = torch.log_softmax(logits, dim=-1)
        # loss = criterion(log_probs, targets) # criterion should be nn.KLDivLoss(reduction='batchmean')

