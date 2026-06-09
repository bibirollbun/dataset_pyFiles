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


!pip install pyts -q
!pip install torchinfo hiddenlayer -q
!pip install -q torchsummary


import os
import gc
import pickle
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, precision_recall_fscore_support, roc_auc_score, hamming_loss

# Signal & Image Processing
!pip install pyts -q
from pyts.image import GramianAngularField
from scipy.signal import spectrogram
from skimage.transform import resize

# PyTorch & Torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torchvision import models

# Model Summary & Visualization Libraries
from torchsummary import summary
import hiddenlayer as hl

import warnings
warnings.filterwarnings("ignore")



class CFG:
    # General
    DEBUG = False # Debug mode. If True, only 1 epoch is run on a subset of the data.
    
    # --- Paths ---
    INPUT_DIR = '/kaggle/input/cmi-detect-behavior-with-sensor-data/'
    TRAIN_CSV_PATH = os.path.join(INPUT_DIR, 'train.csv')
    WORK_DIR = '/kaggle/working/'
    PREPROCESSED_DIR = os.path.join(WORK_DIR, 'processed_data/') # Directory to save preprocessed data
    MODEL_OUTPUT_DIR = os.path.join(WORK_DIR, 'models/') # Directory to save trained models

    # --- Data & Preprocessing Parameters ---
    SEQUENCE_LENGTH = 64
    IMG_SIZE_2D = 128
    GAF_IMG_SIZE = 32
    SPECTROGRAM_IMG_SIZE = 32
    
    # --- Column Lists & Class Counts ---
    TARGET_COLS = ['orientation', 'behavior', 'gesture', 'phase']
    N_CLASSES = [] # This will be set dynamically after creating label_encoders
    IMU_COLS = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    THM_COLS = [f'thm_{i}' for i in range(1, 6)]
    TOF_COLS = [f'tof_{s}_v{v}' for s in range(1, 6) for v in range(64)]

    # --- Model Hyperparameters ---
    MODEL_WIDTH_MULT = 1.0
    MODEL_DEPTH_MULT = 1.0
    DROPOUT_RATE = 0.3
    
    # --- Training Parameters ---
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6
    
    # --- Hardware & Reproducibility ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FIX: Set NUM_WORKERS to 0 to prevent the AssertionError in notebooks
    NUM_WORKERS = 0 
    SEED = 42

if CFG.DEBUG:
    CFG.EPOCHS = 1

os.makedirs(CFG.PREPROCESSED_DIR, exist_ok=True)
os.makedirs(CFG.MODEL_OUTPUT_DIR, exist_ok=True)



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG.SEED)



df = pd.read_csv(CFG.TRAIN_CSV_PATH)


print(df.shape)
print(df.index)
print(df.dtypes)


df.sample(10)


df.info


print(len(df['sequence_id'].unique()))


print(df['gesture'].unique())
print(df['orientation'].unique())
print(df['phase'].unique())
print(df['behavior'].unique())


def process_and_save_data(df, sequence_ids, scalers, label_encoders, cfg):
    """
    Preprocesses each sequence and saves it locally as a tensor.
    """
    gaf = GramianAngularField(image_size=cfg.GAF_IMG_SIZE, method='summation')
    
    for seq_id in tqdm(sequence_ids, desc="Preprocessing Sequences"):
        save_path = Path(cfg.PREPROCESSED_DIR) / f"{seq_id}.pt"
        if save_path.exists():
            continue
        
        seq_df = df[df['sequence_id'] == seq_id].copy()
        
        # Fill missing values with linear interpolation
        for col_group in [cfg.IMU_COLS, cfg.THM_COLS, cfg.TOF_COLS]:
            seq_df[col_group] = seq_df[col_group].interpolate(method='linear', limit_direction='both').fillna(0)
        
        # Standardization
        seq_df[cfg.IMU_COLS] = scalers['imu'].transform(seq_df[cfg.IMU_COLS])
        seq_df[cfg.THM_COLS] = scalers['thm'].transform(seq_df[cfg.THM_COLS])
        seq_df[cfg.TOF_COLS] = scalers['tof'].transform(seq_df[cfg.TOF_COLS])
        
        # Fix sequence length (padding or truncation)
        current_len = len(seq_df)
        if current_len > cfg.SEQUENCE_LENGTH:
            seq_df = seq_df.iloc[:cfg.SEQUENCE_LENGTH]
        elif current_len < cfg.SEQUENCE_LENGTH:
            padding_len = cfg.SEQUENCE_LENGTH - current_len
            padding_df = pd.DataFrame(np.zeros((padding_len, len(df.columns))), columns=df.columns)
            # Copy non-numeric data like sequence_id to maintain it
            for col in ['sequence_id', 'subject'] + cfg.TARGET_COLS:
                 if not seq_df[col].dropna().empty:
                    padding_df[col] = seq_df[col].iloc[0]
            seq_df = pd.concat([seq_df, padding_df], ignore_index=True)
            
        # Create 3D tensor from ToF data
        tof_data = seq_df[cfg.TOF_COLS].values.reshape(cfg.SEQUENCE_LENGTH, 5, 8, 8)
        tensor_3d = torch.tensor(tof_data, dtype=torch.float32).permute(1, 0, 2, 3) # [C, D, H, W]
        
        # Generate GAF images from IMU data
        imu_data = seq_df[cfg.IMU_COLS].values.T
        imu_gaf = torch.tensor(gaf.fit_transform(imu_data), dtype=torch.float32)
        
        # Generate spectrograms from temperature data
        thm_data = seq_df[cfg.THM_COLS].values.T
        thm_specs = []
        for i in range(thm_data.shape[0]):
            _, _, Sxx = spectrogram(thm_data[i, :], fs=20, nperseg=16, noverlap=8)
            Sxx_resized = resize(np.log1p(Sxx), (cfg.SPECTROGRAM_IMG_SIZE, cfg.SPECTROGRAM_IMG_SIZE))
            thm_specs.append(Sxx_resized)
        thm_spec = torch.tensor(np.array(thm_specs), dtype=torch.float32)
        
        # Combine GAF images and spectrograms to create 2D tensor
        tensor_2d_combined = torch.cat([imu_gaf, thm_spec], dim=0)
        tensor_2d = F.interpolate(tensor_2d_combined.unsqueeze(0), size=(cfg.IMG_SIZE_2D, cfg.IMG_SIZE_2D), mode='bilinear', align_corners=False).squeeze(0)
        
        # Encode labels
        labels = {}
        for col in cfg.TARGET_COLS:
            # Dropna and get the first valid label for the sequence
            valid_labels = seq_df[col].dropna()
            if not valid_labels.empty:
                label_val = str(valid_labels.iloc[0])
                encoded_label = label_encoders[col].transform([label_val])[0]
                labels[col] = torch.tensor(encoded_label, dtype=torch.long)
            else: # Handle cases with no valid labels if necessary
                labels[col] = torch.tensor(-1, dtype=torch.long) # Or some placeholder
        
        torch.save({'tensor_3d': tensor_3d, 'tensor_2d': tensor_2d, 'labels': labels}, save_path)



label_encoders = {col: LabelEncoder().fit(df[col].astype(str).unique()) for col in CFG.TARGET_COLS}
CFG.N_CLASSES = [len(label_encoders[col].classes_) for col in CFG.TARGET_COLS]
scalers = {
    'imu': StandardScaler().fit(df[CFG.IMU_COLS].fillna(0)),
    'thm': StandardScaler().fit(df[CFG.THM_COLS].fillna(0)),
    'tof': StandardScaler().fit(df[CFG.TOF_COLS].fillna(0)),
}


with open(os.path.join(CFG.MODEL_OUTPUT_DIR, 'scalers.pkl'), 'wb') as f:
    pickle.dump(scalers, f)
with open(os.path.join(CFG.MODEL_OUTPUT_DIR, 'label_encoders.pkl'), 'wb') as f:
    pickle.dump(label_encoders, f)
print("Scalers and label encoders saved.")


all_sequence_ids = df['sequence_id'].unique()
process_and_save_data(df, all_sequence_ids, scalers, label_encoders, CFG)
del df
gc.collect()


class CustomDataset(Dataset):
    def __init__(self, sequence_ids, cfg):
        self.sequence_ids = sequence_ids
        self.cfg = cfg

    def __len__(self):
        return len(self.sequence_ids)

    def __getitem__(self, idx):
        seq_id = self.sequence_ids[idx]
        file_path = os.path.join(self.cfg.PREPROCESSED_DIR, f"{seq_id}.pt")
        data = torch.load(file_path)
        
        tensor_3d = data['tensor_3d']
        tensor_2d = data['tensor_2d']
        labels = data['labels']
        
        return tensor_3d, tensor_2d, labels


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("/kaggle/input/multilablehybrideffecinetnetmodel/MultiLableHybridEffecinetNetModel.png")
plt.imshow(img)
plt.axis("off")  
plt.show()


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, 1, bias=False), Swish(),
            nn.Conv3d(channels // reduction, channels, 1, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.excitation(self.squeeze(x)).expand_as(x)



class MBConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=1, se_ratio=0.25, dropout_rate=0.0):
        super().__init__()

        #Residual Connection
        #Preventing gradient vanishing and stabilizing learning
        self.use_residual = (stride == 1 and in_channels == out_channels)
        expanded_channels = in_channels * expand_ratio

        layers = []


        #Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv3d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm3d(expanded_channels),
                Swish()
            ])


        #Depthwise Convolution
        #groups=expanded_channels -> Depthwise Conv
        #->Extracting local features of space-time for each channel
        layers.extend([
            nn.Conv3d(expanded_channels, expanded_channels, kernel_size, stride=stride,
                      padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm3d(expanded_channels),
            Swish()
        ])

        #Squeeze-and-Excitation (SE Block)
        if se_ratio > 0:
            layers.append(SEBlock3D(expanded_channels, reduction=int(1/se_ratio)))


        #Projection
        layers.extend([
            nn.Conv3d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        residual = x
        x = self.conv(x)
        if self.dropout:
            x = self.dropout(x)
        if self.use_residual:
            x += residual
        return x


class EfficientNet3D_Branch(nn.Module):
    def __init__(self, in_channels=5, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2):
        super().__init__()
        init_channels = int(32 * width_mult)
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, init_channels, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(init_channels), Swish()
        )

        in_channels = init_channels
        mbconv_configs = [
            [1, 16, 1, (1,1,1), 3], [6, 24, 2, (2,2,2), 3], [6, 40, 2, (2,2,2), 5],
            [6, 80, 3, (2,2,2), 3], [6, 112, 3, (1,1,1), 5], [6, 192, 4, (2,2,2), 5],
            [6, 320, 1, (1,1,1), 3]
        ]

        self.mbconv_layers = nn.ModuleList()
        for expand_ratio, channels, num_layers, stride, kernel_size in mbconv_configs:
            out_channels = int(channels * width_mult)
            layers_repeat = int(np.ceil(num_layers * depth_mult))
            for i in range(layers_repeat):
                layer_stride = stride if i == 0 else (1,1,1)
                self.mbconv_layers.append(MBConv3D(in_channels, out_channels, kernel_size, layer_stride, expand_ratio, dropout_rate=dropout_rate))
                in_channels = out_channels

        self.final_channels = int(1280 * width_mult)
        self.final_conv = nn.Sequential(
            nn.Conv3d(in_channels, self.final_channels, 1, bias=False),
            nn.BatchNorm3d(self.final_channels), Swish()
        )
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.initial_conv(x)
        for layer in self.mbconv_layers:
            x = layer(x)
        x = self.final_conv(x)
        x = self.global_avg_pool(x)
        return x.view(x.size(0), -1)



class EfficientNet2D_Branch(nn.Module):
    def __init__(self, in_channels=12):
        super().__init__()
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        original_conv = self.base_model.features[0][0]
        self.base_model.features[0][0] = nn.Conv2d(
            in_channels, original_conv.out_channels,
            kernel_size=original_conv.kernel_size, stride=original_conv.stride,
            padding=original_conv.padding, bias=False
        )
        self.base_model.classifier = nn.Identity()

    def forward(self, x):
        return self.base_model(x)


class MultiTaskHybridNet(nn.Module):
    def __init__(self, target_cols, n_classes_list,CFG):
        super().__init__()
        self.branch_3d = EfficientNet3D_Branch(
            in_channels=5,
            width_mult=CFG.MODEL_WIDTH_MULT,
            depth_mult=CFG.MODEL_DEPTH_MULT,
            dropout_rate=CFG.DROPOUT_RATE
        )
        self.branch_2d = EfficientNet2D_Branch(
            in_channels=len(CFG.IMU_COLS) + len(CFG.THM_COLS)
        )
        combined_features_dim = self.branch_3d.final_channels + 1280
        self.dropout = nn.Dropout(p=CFG.DROPOUT_RATE)
        self.heads = nn.ModuleDict({
            col: nn.Linear(combined_features_dim, num_classes)
            for col, num_classes in zip(target_cols, n_classes_list)
        })

    def forward(self, x_3d, x_2d):
        with autocast(enabled=True):
            features_3d = self.branch_3d(x_3d)
            features_2d = self.branch_2d(x_2d)
            combined_features = torch.cat([features_3d, features_2d], dim=1)
            combined_features = self.dropout(combined_features)
            outputs = {col: head(combined_features) for col, head in self.heads.items()}
        return outputs


model_for_summary = MultiTaskHybridNet(
    target_cols=CFG.TARGET_COLS,
    n_classes_list=CFG.N_CLASSES,
    CFG=CFG
).to('cpu')

input_shape_3d = (5, CFG.SEQUENCE_LENGTH, 8, 8)
input_shape_2d = (len(CFG.IMU_COLS) + len(CFG.THM_COLS), CFG.IMG_SIZE_2D, CFG.IMG_SIZE_2D)

print("--- Summary for 3D Branch (EfficientNet3D_Branch) ---")
summary(model_for_summary.branch_3d, input_size=input_shape_3d, device='cpu')

print("\n" + "="*80 + "\n")

print("--- Summary for 2D Branch (EfficientNet2D_Branch) ---")
summary(model_for_summary.branch_2d, input_size=input_shape_2d, device='cpu')

del model_for_summary
gc.collect()


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(loader, model, criterions, optimizer, scheduler, device):
    model.train()
    scaler = GradScaler()
    loss_meter = AverageMeter()
    
    for x_3d, x_2d, labels in tqdm(loader, desc="Training"):
        x_3d, x_2d = x_3d.to(device), x_2d.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(x_3d, x_2d)          
            loss = 0
            for col in CFG.TARGET_COLS:
                loss += criterions[col](outputs[col], labels[col].to(device))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loss_meter.update(loss.item(), x_3d.size(0))
        
    scheduler.step()
    return loss_meter.avg


def valid_fn(loader, model, criterions, device):
    model.eval()
    loss_meter = AverageMeter()
    all_preds = {col: [] for col in CFG.TARGET_COLS}
    all_trues = {col: [] for col in CFG.TARGET_COLS}
    
    with torch.no_grad():
        for x_3d, x_2d, labels in tqdm(loader, desc="Validating"):
            x_3d, x_2d = x_3d.to(device), x_2d.to(device)
            outputs = model(x_3d, x_2d) 
            loss = 0
            for col in CFG.TARGET_COLS:
                loss += criterions[col](outputs[col], labels[col].to(device))
                all_preds[col].append(F.softmax(outputs[col], dim=1).cpu().numpy())
                all_trues[col].append(labels[col].numpy())

            loss_meter.update(loss.item(), x_3d.size(0))

    for col in CFG.TARGET_COLS:
        all_preds[col] = np.concatenate(all_preds[col])
        all_trues[col] = np.concatenate(all_trues[col])
        
    return loss_meter.avg, all_preds, all_trues


def comprehensive_evaluation_function(trues, preds, label_encoders):
    """
    Calculates and displays a comprehensive set of metrics for each target
    in a clean, tabular format.
    """
    print("=" * 60)
    print("Comprehensive Evaluation Metrics")
    print("=" * 60)

    # A list to store the results for each target as a dictionary
    results_list = []

    # Calculate metrics for each target
    for col in CFG.TARGET_COLS:
        y_true = trues[col]
        y_pred_proba = preds[col]
        y_pred_label = np.argmax(y_pred_proba, axis=1)
        labels = np.arange(len(label_encoders[col].classes_))

        # Accuracy
        accuracy = accuracy_score(y_true, y_pred_label)

        # Precision, Recall, F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_label, average='macro', zero_division=0, labels=labels
        )

        # Append the results for the current target to the list
        results_list.append({
            'Target': col,
            'Accuracy': f"{accuracy:.4f}",
            'Precision': f"{precision:.4f}",
            'Recall': f"{recall:.4f}",
            'F1-score': f"{f1:.4f}"
        })

    # Convert the list of results into a pandas DataFrame for beautiful printing
    results_df = pd.DataFrame(results_list)

    # Print the DataFrame without the index for a cleaner look
    print(results_df.to_string(index=False))
    print("=" * 60)




def run_training(train_ids, val_ids, cfg, label_encoders):
    # Set up Datasets and DataLoaders
    train_dataset = CustomDataset(train_ids, cfg)
    val_dataset = CustomDataset(val_ids, cfg)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    
    # Initialize the model
    model = MultiTaskHybridNet(
        target_cols=cfg.TARGET_COLS,
        n_classes_list=cfg.N_CLASSES,
        CFG=cfg
    ).to(cfg.DEVICE)
    
    # Set up loss functions, optimizer, and scheduler
    criterions = {col: nn.CrossEntropyLoss() for col in cfg.TARGET_COLS}
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
    
    best_loss = float('inf')
    
    # Dictionary to store training history for plotting
    history = {'train_loss': [], 'valid_loss': []}
    
    # Start the training loop
    for epoch in range(cfg.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{cfg.EPOCHS}")
        print("-" * 15)
        
        # Training step
        train_loss = train_fn(train_loader, model, criterions, optimizer, scheduler, cfg.DEVICE)
        
        # Validation step
        valid_loss, preds, trues = valid_fn(val_loader, model, criterions, cfg.DEVICE)
        
        # Record losses for this epoch
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        
        # Display evaluation metrics for the epoch
        comprehensive_evaluation_function(trues, preds, label_encoders)
        
        print(f"Summary - Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")

        # Save the best model based on validation loss
        if valid_loss < best_loss:
            best_loss = valid_loss
            print(f"  -> Validation loss improved. Saving model to best_model.pth")
            save_path = os.path.join(CFG.MODEL_OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            
    # Return the history dictionary for later analysis
    return history



train_ids, val_ids = train_test_split(all_sequence_ids, test_size=0.2, random_state=CFG.SEED)
run_training(train_ids, val_ids, CFG, label_encoders)

