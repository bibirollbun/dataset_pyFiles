!pip install -q iterative-stratification 


import os
import timm
import torch
import random
import shutil
import pydicom
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
import torch.nn as nn
from typing import List
from torch import Tensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


seed = 210
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print('Finish seeding with seed {}'.format(seed))

seed_everything(seed)
print('Training on device {}'.format(device))


train = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv")
train_coor = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv")
train_series = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv")
train_dummy = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv")
train_meta = pd.read_csv("/kaggle/input/meta-csv/meta.csv")


train_dummy = train_dummy.fillna("Normal/Mild")


train_coor = train_coor.merge(train_series[['study_id', 'series_id', 'series_description']], on=['study_id', 'series_id'])



class SpineCoorDataset(Dataset):
    def __init__(self, coor, meta, condition, mode):
        if condition == 'scs':
            self.coor = coor.loc[coor.condition == "Spinal Canal Stenosis"]
        elif condition == 'nfn':
            self.coor = coor.loc[coor.condition.isin([
                'Left Neural Foraminal Narrowing',
                'Right Neural Foraminal Narrowing'
            ])]
        elif condition == 'ss':
            self.coor = coor.loc[coor.condition.isin([
                'Left Subarticular Stenosis',
                'Right Subarticular Stenosis'
            ])]

        g_coor = self.coor.groupby(['study_id']).count()
        if condition == 'scs':
            self.id = g_coor[g_coor.series_id == 5].reset_index().study_id.unique()
        else:
            self.id = g_coor[g_coor.series_id == 10].reset_index().study_id.unique()

        if condition == 'ss':
            self.resize = v2.Resize((256, 256))
        else:
            self.resize = v2.Resize((384, 384)) # it can be adjusted accordingly to system resources

        self.id = list(set(self.id) - set([3637444890]))

        self.condition = condition
        self.meta = meta
        self.mode = mode

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        study_id = self.id[idx]
        if self.condition == 'scs':
            volume, label = self.volume_scs(study_id)
        elif self.condition == 'nfn':
            volume, label = self.volume_nfn(study_id)
        elif self.condition == 'ss':
            volume, label = self.volume_ss(study_id)

        return volume, label

    def volume_scs(self, study_id):
        all_levels = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
        meta = self.meta.loc[(self.meta.study_id == study_id) & (self.meta.series_description == 'Sagittal T2/STIR')]
        meta = meta.sort_values('ipp_x', ascending=True).reset_index(drop=True)
        coor = self.coor.loc[(self.coor.study_id == study_id) & (self.coor.series_description == 'Sagittal T2/STIR')]
        coor_dict = {}
        meta_list = []
        for _, row in coor.iterrows():
            series_id, instance_number = row.series_id, row.instance_number
            meta_list.append(meta.loc[(meta.series_id == series_id) & (meta.instance_number == instance_number)])
        sub_meta = pd.concat(meta_list)
        idx = meta.loc[meta.ipp_x == sub_meta.ipp_x.median()].index[0]
        # idx = meta.loc[meta.instance_number == sub_meta.instance_number.median()].index[0]
        img_row = meta.loc[idx]
        before_img_row = meta.loc[idx -1]
        after_img_row = meta.loc[idx + 1]
        img = self.normalize(self.load_dicom(f"/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{study_id}/{img_row.series_id}/{img_row.instance_number}.dcm"))
        bimg = self.normalize(self.load_dicom(f"/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{study_id}/{before_img_row.series_id}/{before_img_row.instance_number}.dcm"))
        aimg = self.normalize(self.load_dicom(f"/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{study_id}/{after_img_row.series_id}/{after_img_row.instance_number}.dcm"))
        height, width = img.shape
        for _, row in coor.iterrows():
            series_id, instance_number = row.series_id, row.instance_number
            # print(row.x)
            x = row.x/width
            # print(x)
            y = row.y/height
            coor_dict[row.level] = torch.tensor([x, y]).to(torch.float32)

        updated_dict = {}
        for level in all_levels:
            if level in coor_dict:
                updated_dict[level] = coor_dict[level]
            else:
                print(f"Missing level '{level}' in study ID: {study_id}")
                raise ValueError(f"Missing coordinate for level '{level}' in study ID: {study_id}")
        coor_dict = updated_dict

        img = self.resize(torch.tensor(img[None, ...]))
        bimg = self.resize(torch.tensor(bimg[None, ...]))
        aimg = self.resize(torch.tensor(aimg[None, ...]))
        img = torch.cat([bimg, img, aimg]).to(torch.float32)

        return img, coor_dict


    def volume_nfn(self, study_id):
        all_levels = ['left_L1/L2', 'left_L2/L3', 'left_L3/L4', 'left_L4/L5', 'left_L5/S1', 'right_L1/L2', 'right_L2/L3', 'right_L3/L4', 'right_L4/L5', 'right_L5/S1']
        meta = self.meta.loc[(self.meta.study_id == study_id) & (self.meta.series_description == 'Sagittal T1')]
        meta = meta.sort_values('ipp_x', ascending=True).reset_index(drop=True)
        coor = self.coor.loc[(self.coor.study_id == study_id) & (self.coor.series_description == 'Sagittal T1')]
        coor_dict = {}
        right_meta_list = []
        left_meta_list = []
        for _, row in coor.iterrows():
            series_id, instance_number = row.series_id, row.instance_number
            if row.condition == "Right Neural Foraminal Narrowing":
                right_meta_list.append(meta.loc[(meta.series_id == series_id) & (meta.instance_number == instance_number)])
            else:
                left_meta_list.append(meta.loc[(meta.series_id == series_id) & (meta.instance_number == instance_number)])
        right_sub_meta = pd.concat(right_meta_list)
        left_sub_meta = pd.concat(left_meta_list)
        right_idx = meta.loc[meta.ipp_x == right_sub_meta.ipp_x.median()].index[0]
        left_idx = meta.loc[meta.ipp_x == left_sub_meta.ipp_x.median()].index[0]
        # idx = meta.loc[meta.instance_number == sub_meta.instance_number.median()].index[0]
        right_img_row = meta.loc[right_idx]
        right_before_img_row = meta.loc[right_idx -1]
        right_after_img_row = meta.loc[right_idx + 1]

        left_img_row = meta.loc[left_idx]
        left_before_img_row = meta.loc[left_idx -1]
        left_after_img_row = meta.loc[left_idx + 1]

        rimg = self.normalize(self.load_dicom(f"/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{study_id}/{right_img_row.series_id}/{right_img_row.instance_number}.dcm"))
        rbimg = self.normalize(self.load_dicom(f"/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{study_id}/{right_before_img_row.series_id}/{right_before_img_row.instance_number}.dcm"))
        ramig = self.normalize(self.load_dicom(f"/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{study_id}/{right_after_img_row.series_id}/{right_after_img_row.instance_number}.dcm"))

        limg = self.normalize(self.load_dicom(f"/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{study_id}/{left_img_row.series_id}/{left_img_row.instance_number}.dcm"))
        lbimg = self.normalize(self.load_dicom(f"/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{study_id}/{left_before_img_row.series_id}/{left_before_img_row.instance_number}.dcm"))
        laimg = self.normalize(self.load_dicom(f"/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{study_id}/{left_after_img_row.series_id}/{left_after_img_row.instance_number}.dcm"))

        rheight, rwidth = rimg.shape
        lheight, lwidth = limg.shape

        for _, row in coor.iterrows():
            series_id, instance_number = row.series_id, row.instance_number
            if row.condition == 'Right Neural Foraminal Narrowing':
                x = row.x/rwidth
                y = row.y/rheight
                coor_dict['right_' + row.level] = torch.tensor([x, y]).to(torch.float32)
            else:
                x = row.x/lwidth
                y = row.y/lheight
                coor_dict['left_' + row.level] = torch.tensor([x, y]).to(torch.float32)

        updated_dict = {}
        for level in all_levels:
            if level in coor_dict:
                updated_dict[level] = coor_dict[level]
            else:
                print(f"Missing level '{level}' in study ID: {study_id}")
                raise ValueError(f"Missing coordinate for level '{level}' in study ID: {study_id}")
        coor_dict = updated_dict

        rimg = self.resize(torch.tensor(rimg[None, ...]))
        rbimg = self.resize(torch.tensor(rbimg[None, ...]))
        ramig = self.resize(torch.tensor(ramig[None, ...]))

        rimg = torch.cat([rbimg, rimg, ramig]).to(torch.float32)

        limg = self.resize(torch.tensor(limg[None, ...]))
        lbimg = self.resize(torch.tensor(lbimg[None, ...]))
        laimg = self.resize(torch.tensor(laimg[None, ...]))

        limg = torch.cat([lbimg, limg, laimg]).to(torch.float32)

        img = torch.stack([limg, rimg]).to(torch.float32).contiguous()

        return img, coor_dict

    def volume_ss(self, study_id):
        all_levels = ['left_L1/L2', 'left_L2/L3', 'left_L3/L4', 'left_L4/L5', 'left_L5/S1', 'right_L1/L2', 'right_L2/L3', 'right_L3/L4', 'right_L4/L5', 'right_L5/S1']
        meta = self.meta.loc[(self.meta.study_id == study_id) & (self.meta.series_description == 'Axial T2')]
        meta = meta.sort_values('ipp_z', ascending=True).reset_index(drop=True)
        coor = self.coor.loc[(self.coor.study_id == study_id) & (self.coor.series_description == 'Axial T2')]
        coor_dict = {}
        img_dict = {}
        for _, row in coor.iterrows():
            series_id, instance_number = row.series_id, row.instance_number
            img = self.load_dicom(f"/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{study_id}/{series_id}/{instance_number}.dcm")
            height, width = img.shape
            img = self.resize(torch.tensor(img[None, ...]))
            img = self.normalize(img.to(torch.float32))
            x = row.x/width
            y = row.y/height
            if row.condition == 'Left Subarticular Stenosis':
                coor_dict['left_' + row.level] = torch.tensor([x, y]).to(torch.float32)
                img_dict['left_' + row.level] = img
            else:
                coor_dict['right_' + row.level] = torch.tensor([x, y]).to(torch.float32)
                img_dict['right_' + row.level] = img
            

        updated_dict = {}
        img_list = []
        for level in all_levels:
            if level in coor_dict:
                updated_dict[level] = coor_dict[level]
                img_list.append(img_dict[level])
            else:
                print(f"Missing level '{level}' in study ID: {study_id}")
                raise ValueError(f"Missing coordinate for level '{level}' in study ID: {study_id}")
        coor_dict = updated_dict

        volume = torch.stack(img_list).contiguous()

        return volume, coor_dict

    def normalize(self, x):
        if self.condition == 'ss':
            lower, upper = torch.quantile(x, torch.tensor(0.01)), torch.quantile(x, torch.tensor(0.99))
            x = torch.clamp(x, lower, upper)
            x = x - torch.min(x)
            x = x / torch.max(x)
        else:
            lower, upper = np.percentile(x, (1, 99))
            x = np.clip(x, lower, upper)
            x = x - np.min(x)
            x = x / np.max(x)
        return x

    def load_dicom(self, path):
        return pydicom.dcmread(path).pixel_array


import torch
import torch.nn as nn
import timm

# ----------------- Gated Attention Block (with Residual) -----------------
class GatedAttention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Sigmoid()
        )
    def forward(self, x):
        gate = self.gate(x)
        return x * gate + x  # Residual connection

# ----------------- MobileNetV3Small Spine Detection Model -----------------
class MobileNetV3GatedSpine(nn.Module):
    def __init__(self):
        super().__init__()

        # Load MobileNetV3 Small backbone
        self.encoder = timm.create_model(
            'convnext_base.fb_in22k_ft_in1k_384',
            in_chans=3,
            pretrained=True,
            features_only=False,
            num_classes=0
        )


        # Feature dimension
        self.in_features = self.encoder.num_features
        print(self.in_features)

        # Adaptive Pooling + Flatten
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1)
        )

        # Gated Attention after encoder output
        self.attention = GatedAttention(self.in_features)

        # Project feature
        self.projector = nn.Sequential(
            nn.Linear(self.in_features, 1024),
            nn.SiLU(inplace=True)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.2)

        # Heads for 5 vertebral levels
        self.heads = nn.ModuleList([
            nn.Linear(1024, 2) for _ in range(5)
        ])

    def forward(self, x):
        # Feature extraction
        x = self.encoder.forward_features(x)
        x = self.flatten(x)

        # Gated Attention
        x = self.attention(x)

        # Project to feature dimension
        x = self.projector(x)

        # Small dropout
        x = self.dropout(x)

        # Predict each level
        output = {}
        levels = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']

        for i, level in enumerate(levels):
            output[level] = self.heads[i](x).sigmoid()
        
        return output



from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

label = train.columns[1:]
train_dummy['fold'] = -1  # Initialize before assigning
kfold = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_idx, valid_idx) in enumerate(kfold.split(train_dummy, train_dummy[label])):
    train_dummy.loc[valid_idx, 'fold'] = i
train_series = train_series.merge(train_dummy[['study_id', 'fold']], on='study_id')
train_coor = train_coor.merge(train_dummy[['study_id', 'fold']], on='study_id')
train = train.merge(train_dummy[['study_id', 'fold']], on='study_id')
train_meta = train_meta.merge(train_dummy[['study_id', 'fold']], on='study_id')



class SCSLoss(nn.Module):
    def __init__(self, condition="scs"):
        super(SCSLoss, self).__init__()
        self.condition = condition

    def forward(self, outputs, targets):
        loss = 0
        count = 0
        if self.condition == 'scs':
            expected_level = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
        else:
            expected_level = ['left_L1/L2', 'left_L2/L3', 'left_L3/L4', 'left_L4/L5', 'left_L5/S1', 'right_L1/L2', 'right_L2/L3', 'right_L3/L4', 'right_L4/L5', 'right_L5/S1']
        for level in expected_level:
            if level in targets and level in outputs:
                _loss = nn.functional.l1_loss(outputs[level], targets[level])
                loss += _loss
                count += 1

        # Return average loss, avoiding division by zero
        return loss / max(count, 1)


def calculate_mse_score(outputs, targets):
    """Computes Mean Squared Error (MSE) for model predictions."""
    total_mse = 0
    total_samples = 0
    
    # Only calculate for levels present in both outputs and targets
    for level in set(outputs.keys()).intersection(targets.keys()):
        # Ensure shapes match
        if outputs[level].shape == targets[level].shape:
            mse = nn.functional.mse_loss(outputs[level], targets[level], reduction='sum')
            total_mse += mse.item()
            total_samples += targets[level].numel()
    
    # Return average, avoiding division by zero
    return total_mse / max(total_samples, 1)


def calculate_regression_tolerance(preds, targets, tolerances=[0, 1, 2]):
    """
    Calculate regression tolerance matrix showing what percentage of predictions
    fall within various tolerance thresholds.
    
    Args:
        preds: Dictionary of model predictions
        targets: Dictionary of ground truth values
        tolerances: List of tolerance thresholds in mm
        
    Returns:
        Dictionary with tolerance percentages for each threshold
    """
    tolerance_counts = {f"Â±{tol}": 0 for tol in tolerances}
    tolerance_counts[">Â±2"] = 0
    total = 0

    for level in preds:
        if level not in targets:
            continue

        pred_vals = preds[level].detach().cpu().numpy().round().astype(int)
        true_vals = targets[level].detach().cpu().numpy().astype(int)

        for i in range(len(pred_vals)):
            for j in range(len(pred_vals[i])):
                diff = abs(pred_vals[i][j] - true_vals[i][j])
                matched = False
                for tol in tolerances:
                    if diff <= tol:
                        tolerance_counts[f"Â±{tol}"] += 1
                        matched = True
                        break
                if not matched:
                    tolerance_counts[">Â±2"] += 1
                total += 1

    return tolerance_counts, total


def train_spine_model(model, train_coor, train_meta, n_folds=5, epochs=18, batch_size=4,
                      learning_rate=0.001, weight_decay=0.0001, patience=5,
                      mixed_precision=True, experiment_name="spine_model",
                      tolerances=[0, 1, 2], condition='scs'):
    """
    Complete training function for spine model with improved practices

    Args:
        model: Your defined model
        train_coor: DataFrame containing coordinate annotations
        train_meta: DataFrame containing metadata
        n_folds: Number of folds for cross-validation
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        patience: Patience for early stopping
        mixed_precision: Whether to use mixed precision training
        experiment_name: Name for experiment logs
        tolerances: List of tolerance thresholds for regression tolerance calculation
    """

    # Create output directory for models
    os.makedirs("models", exist_ok=True)

    # Initialize tensorboard writer
    writer = SummaryWriter(f'runs/{experiment_name}')

    # Print training configuration
    print(f"\n===== TRAINING CONFIGURATION =====")
    print(f"Number of folds: {n_folds}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Mixed precision: {mixed_precision}")
    print(f"Tolerance thresholds: {tolerances}")
    print(f"==============================\n")

    # Initialize scaler for mixed precision
    scaler = torch.amp.GradScaler() if mixed_precision else None

    # Cross-validation loop
    all_val_mses = []

    for fold in range(n_folds):
        print(f"\n{'='*20} FOLD {fold+1}/{n_folds} {'='*20}")

        # Initialize fold-specific writer
        fold_writer = SummaryWriter(f'runs/{experiment_name}/fold_{fold}')

        # Initialize datasets and dataloaders
        train_dataset = SpineCoorDataset(train_coor.loc[train_coor.fold!=fold],
                                    train_meta.loc[train_meta.fold!=fold],
                                    condition, 'train')

        valid_dataset = SpineCoorDataset(train_coor.loc[train_coor.fold==fold],
                                    train_meta.loc[train_meta.fold==fold],
                                    condition, 'valid')

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=2, pin_memory=True)

        valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=2, pin_memory=True)

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Learning rate scheduler with warmup
        total_steps = epochs * len(train_loader)
        warmup_steps = int(0.1 * total_steps)  # 10% warmup

        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5
        )

        criterion = SCSLoss(condition=condition)

        # Initialize tracking variables
        train_losses, val_losses = [], []
        train_mses, val_mses = [], []
        best_val_mse = float('inf')
        counter = 0  # For early stopping

        # Ensure all expected levels are present
        if train_dataset.condition == 'scs':
            expected_levels = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        else:
            expected_levels = ['left_L1/L2', 'left_L2/L3', 'left_L3/L4', 'left_L4/L5', 'left_L5/S1',
                              'right_L1/L2', 'right_L2/L3', 'right_L3/L4', 'right_L4/L5', 'right_L5/S1']

        # Training loop
        for epoch in range(epochs):
            print(f"\nğŸš€ Epoch {epoch+1}/{epochs} - Fold {fold+1}/{n_folds}")

            # === Training phase ===
            model.train()
            running_loss = 0.0
            total_mse = 0.0
            epoch_train_tolerances = {f"Â±{tol}": 0 for tol in tolerances}
            epoch_train_tolerances[">Â±2"] = 0
            train_total_predictions = 0

            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                               desc="Training Progress", leave=False)

            for batch_idx, (volume, batch) in progress_bar:
                volume = volume.to(device)
                # Ensure all expected levels are present
                batch = {key: value.to(device) for key, value in batch['coor'].items() if key in expected_levels}

                # Reset gradients
                optimizer.zero_grad()

                # Mixed precision training
                if mixed_precision:
                    with torch.amp.autocast('cuda'):
                        outputs_raw = model(volume)
                        outputs = {}
                        for level in outputs_raw:
                            if isinstance(outputs_raw[level], dict):
                                outputs[level] = outputs_raw[level]['coor']  # Take only coordinate tensor
                            else:
                                outputs[level] = outputs_raw[level]  # In case it's not dict (safety)
                        loss = criterion(outputs, batch)

                    # Scale loss and backward pass
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard precision training
                    outputs = model(volume)
                    loss = criterion(outputs, batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # Step the scheduler
                scheduler.step()

                # Track metrics
                running_loss += loss.item()
                batch_mse = calculate_mse_score(outputs, batch)
                total_mse += batch_mse

                # Calculate tolerance metrics for this batch
                batch_tolerance_counts, batch_total = calculate_regression_tolerance(outputs, batch, tolerances)
                for k in batch_tolerance_counts:
                    epoch_train_tolerances[k] += batch_tolerance_counts[k]
                train_total_predictions += batch_total

                # Update progress bar
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mse=f"{batch_mse:.4f}",
                    lr=f"{current_lr:.6f}",
                    t_mse=f"{total_mse:.4f}"
                )

                # Free memory
                del volume, batch, outputs, loss
                torch.cuda.empty_cache()

            # Calculate epoch metrics
            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_mse = total_mse / len(train_loader)
            train_losses.append(epoch_train_loss)
            train_mses.append(epoch_train_mse)

            # Convert tolerance counts to percentages
            for k in epoch_train_tolerances:
                if train_total_predictions > 0:
                    epoch_train_tolerances[k] = 100 * epoch_train_tolerances[k] / train_total_predictions
                else:
                    epoch_train_tolerances[k] = 0.0

            # Log metrics
            fold_writer.add_scalar('Loss/train', epoch_train_loss, epoch)
            fold_writer.add_scalar('MSE/train', epoch_train_mse, epoch)
            writer.add_scalar(f'Loss/train/fold_{fold}', epoch_train_loss, epoch)
            writer.add_scalar(f'MSE/train/fold_{fold}', epoch_train_mse, epoch)

            # Log tolerance metrics
            for tolerance_key, tolerance_value in epoch_train_tolerances.items():
                fold_writer.add_scalar(f'Tolerance/train/{tolerance_key}', tolerance_value, epoch)
                writer.add_scalar(f'Tolerance/train/{tolerance_key}/fold_{fold}', tolerance_value, epoch)

            print(f"ğŸ”¥ Training Loss: {epoch_train_loss:.4f} | MSE Score: {epoch_train_mse:.4f}")
            print(f"ğŸ“� Training Tolerances: " + " | ".join([f"{k}: {v:.1f}%" for k, v in epoch_train_tolerances.items()]))

            # === Validation phase ===
            model.eval()
            val_running_loss = 0.0
            val_total_mse = 0.0
            epoch_val_tolerances = {f"Â±{tol}": 0 for tol in tolerances}
            epoch_val_tolerances[">Â±2"] = 0
            val_total_predictions = 0

            with torch.no_grad():
                for volume, batch in tqdm(valid_loader, desc="Validation Progress", leave=False):

                    volume = volume.to(device)
                    # Ensure all expected levels are present
                    batch = {key: value.to(device) for key, value in batch['coor'].items() if key in expected_levels}

                    outputs_raw = model(volume)
                    outputs = {}
                    for level in outputs_raw:
                        outputs[level] = outputs_raw[level]['coor']

                    loss = criterion(outputs, batch)
                    
                    val_running_loss += loss.item()
                    batch_mse = calculate_mse_score(outputs, batch)
                    val_total_mse += batch_mse

                    # Calculate tolerance metrics for validation batch
                    batch_tolerance_counts, batch_total = calculate_regression_tolerance(outputs, batch, tolerances)
                    for k in batch_tolerance_counts:
                        epoch_val_tolerances[k] += batch_tolerance_counts[k]
                    val_total_predictions += batch_total

                    del volume, batch, outputs, loss
                    torch.cuda.empty_cache()

            # Calculate validation metrics
            epoch_val_loss = val_running_loss / len(valid_loader)
            epoch_val_mse = val_total_mse / len(valid_loader)
            val_losses.append(epoch_val_loss)
            val_mses.append(epoch_val_mse)

            # Convert tolerance counts to percentages
            for k in epoch_val_tolerances:
                if val_total_predictions > 0:
                    epoch_val_tolerances[k] = 100 * epoch_val_tolerances[k] / val_total_predictions
                else:
                    epoch_val_tolerances[k] = 0.0

            # Log validation metrics
            fold_writer.add_scalar('Loss/val', epoch_val_loss, epoch)
            fold_writer.add_scalar('MSE/val', epoch_val_mse, epoch)
            writer.add_scalar(f'Loss/val/fold_{fold}', epoch_val_loss, epoch)
            writer.add_scalar(f'MSE/val/fold_{fold}', epoch_val_mse, epoch)

            # Log validation tolerance metrics
            for tolerance_key, tolerance_value in epoch_val_tolerances.items():
                fold_writer.add_scalar(f'Tolerance/val/{tolerance_key}', tolerance_value, epoch)
                writer.add_scalar(f'Tolerance/val/{tolerance_key}/fold_{fold}', tolerance_value, epoch)

            print(f"âœ… Validation Loss: {epoch_val_loss:.4f} | MSE Score: {epoch_val_mse:.4f}")
            print(f"ğŸ“� Validation Tolerances: " + " | ".join([f"{k}: {v:.1f}%" for k, v in epoch_val_tolerances.items()]))

            # Model checkpointing - save best model based on validation MSE
            if epoch_val_mse < best_val_mse:
                best_val_mse = epoch_val_mse
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': epoch_val_loss,
                    'val_mse': epoch_val_mse,
                    'val_tolerances': epoch_val_tolerances,
                }, f'models/{experiment_name}_fold_{fold}_best.pt')

                print(f"ğŸ“Œ New best model saved with MSE: {best_val_mse:.4f}")
                counter = 0  # Reset early stopping counter
            else:
                counter += 1
                print(f"âš ï¸� No improvement for {counter}/{patience} epochs")

            # Early stopping
            if counter >= patience:
                print(f"â›” Early stopping triggered after {epoch+1} epochs")
                break

        # End of fold - record best validation MSE
        all_val_mses.append(best_val_mse)
        print(f"Fold {fold+1} completed. Best validation MSE: {best_val_mse:.4f}")

        # Save final model for this fold
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_losses[-1],
            'val_mse': val_mses[-1],
            'val_tolerances': epoch_val_tolerances,
        }, f'models/{experiment_name}_fold_{fold}_final.pt')

        # Close fold writer
        fold_writer.close()

    # End of cross-validation
    avg_val_mse = sum(all_val_mses) / len(all_val_mses)
    print(f"\n===== TRAINING COMPLETE =====")
    print(f"Cross-validation results:")
    for fold, mse in enumerate(all_val_mses):
        print(f"Fold {fold+1}: MSE = {mse:.4f}")
    print(f"Average validation MSE: {avg_val_mse:.4f}")

    # Save experiment summary
    with open(f'models/{experiment_name}_summary.txt', 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Folds: {n_folds}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Weight decay: {weight_decay}\n")
        f.write(f"Mixed precision: {mixed_precision}\n")
        f.write(f"Tolerance thresholds: {tolerances}\n")
        f.write("\nResults:\n")
        for fold, mse in enumerate(all_val_mses):
            f.write(f"Fold {fold+1}: MSE = {mse:.4f}\n")
        f.write(f"Average validation MSE: {avg_val_mse:.4f}\n")

    # Close main writer
    writer.close()

    plt.figure(figsize=(10, 6))
    plt.plot(all_val_mses, label='Train MSE', marker='o')
    plt.plot(avg_val_mse, label='Validation MSE', marker='x')
    plt.title('Train vs Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'models/{experiment_name}_mse_epoch_plot.png')
    plt.show()


    return all_val_mses, avg_val_mse


model = MobileNetV3GatedSpine().to(device)
mses, avg_mse = train_spine_model(model, train_coor, train_meta, n_folds=5, condition='scs')


class ConvNextSCSDetect(nn.Module):
    def __init__(self):
        super().__init__()
        #self.size = 384
        self.encoder = timm.create_model(
            'maxvit_tiny_rw_224',   # <-- model name from timm
            pretrained=True,
            in_chans=3,
            features_only=False,
            num_classes=0
        )
        self.in_features = self.encoder.num_features
        print(self.in_features)
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(1),
                                    #nn.LayerNorm(self.in_features)
                                    )

        self.l1 = nn.Linear(self.in_features, 2)
        self.l2 = nn.Linear(self.in_features, 2)
        self.l3 = nn.Linear(self.in_features, 2)
        self.l4 = nn.Linear(self.in_features, 2)
        self.l5 = nn.Linear(self.in_features, 2)

    def forward(self, x, label=None):
        #for loc, img in x.items():
            #print(img.shape)
        #    img = self.encoder.forward_features(img)
        #    img = self.flatten(img)
        #    x[loc] = img
        x = self.encoder.forward_features(x)
        x = self.flatten(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        l3 = self.l3(x)
        l4 = self.l4(x)
        l5 = self.l5(x)
        return {'L1/L2': l1.sigmoid(), 'L2/L3': l2.sigmoid(), 'L3/L4': l3.sigmoid(), 'L4/L5': l4.sigmoid(), 'L5/S1': l5.sigmoid()}


model = ConvNextSCSDetect().to(device)
mses, avg_mse = train_spine_model(model, train_coor, train_meta, n_folds=5, condition='scs')


class ConvNextSCSDetect(nn.Module):
    def __init__(self):
        super().__init__()
        #self.size = 384
        self.encoder = timm.create_model(
            'efficientnetv2_s',   # <-- model name from timm
            pretrained=False,
            in_chans=3,
            features_only=False,
            num_classes=0
        )
        self.in_features = self.encoder.num_features
        print(self.in_features)
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(1),
                                    #nn.LayerNorm(self.in_features)
                                    )

        self.l1 = nn.Linear(self.in_features, 2)
        self.l2 = nn.Linear(self.in_features, 2)
        self.l3 = nn.Linear(self.in_features, 2)
        self.l4 = nn.Linear(self.in_features, 2)
        self.l5 = nn.Linear(self.in_features, 2)

    def forward(self, x, label=None):
        #for loc, img in x.items():
            #print(img.shape)
        #    img = self.encoder.forward_features(img)
        #    img = self.flatten(img)
        #    x[loc] = img
        x = self.encoder.forward_features(x)
        x = self.flatten(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        l3 = self.l3(x)
        l4 = self.l4(x)
        l5 = self.l5(x)
        return {'L1/L2': l1.sigmoid(), 'L2/L3': l2.sigmoid(), 'L3/L4': l3.sigmoid(), 'L4/L5': l4.sigmoid(), 'L5/S1': l5.sigmoid()}


model = ConvNextSCSDetect().to(device)
mses, avg_mse = train_spine_model(model, train_coor, train_meta, n_folds=5, condition='scs')


import gc

# âœ… After each epoch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
gc.collect()  # Optional, forces Python garbage collection

torch.cuda.empty_cache()



import torch
import torch.nn as nn
import timm

# ----------------- Gated Attention Block (with Residual) -----------------
class GatedAttention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Sigmoid()
        )
    def forward(self, x):
        gate = self.gate(x)
        return x * gate + x  # Residual connection

# ----------------- EfficientNetV2-S Spine Detection Model -----------------
class EfficientNetV2GatedSpine(nn.Module):
    def __init__(self):
        super().__init__()

        # Load EfficientNetV2-S backbone
        self.encoder = timm.create_model(
            'efficientnetv2_s',
            pretrained=False,
            in_chans=3,
            features_only=False,
            num_classes=0
        )

        # Feature dimension
        self.in_features = self.encoder.num_features  # 1280 for EfficientNetV2-S

        # Adaptive Pooling + Flatten
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1)
        )

        # Gated Attention after encoder output
        self.attention = GatedAttention(self.in_features)

        # Project feature
        self.projector = nn.Identity()

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.2)

        # Heads for 5 vertebral levels
        self.heads = nn.ModuleList([
            nn.Linear(self.in_features, 2) for _ in range(5)
        ])

    def forward(self, x):
        # Feature extraction
        x = self.encoder.forward_features(x)
        x = self.flatten(x)

        # Gated Attention
        x = self.attention(x)

        # Project to feature dimension
        x = self.projector(x)

        # Small dropout
        x = self.dropout(x)

        # Predict each level
        output = {}
        levels = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']

        for i, level in enumerate(levels):
            output[level] = self.heads[i](x).sigmoid()

        return output



# EfficientNetV2-S
model = EfficientNetV2GatedSpine().to(device)
mses, avg_mse = train_spine_model(model, train_coor, train_meta, n_folds=5, condition='scs')


import torch
import torch.nn as nn
import timm

# ----------------- Gated Attention Block (with Residual) -----------------
class GatedAttention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        gate = self.gate(x)
        return x * gate + x  # Residual connection

# ----------------- EfficientNetV2-S NFN Detection Model -----------------
class EfficientNetV2SNFN(nn.Module):
    def __init__(self):
        super().__init__()

        # Load EfficientNetV2-S backbone
        self.encoder = timm.create_model(
            'efficientnetv2_s',
            in_chans=3,
            pretrained=False,
            features_only=False,
            num_classes=0
        )

        # Feature dimension
        self.in_features = self.encoder.num_features

        # Adaptive Pooling + Flatten
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1)
        )

        # Gated Attention after encoder output
        self.attention = GatedAttention(self.in_features)

        # Project feature (increased from 1024 to 1280 for improved representational capacity)
        self.projector = nn.Sequential(
            nn.Linear(self.in_features, 1280),
            nn.SiLU(inplace=True)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.2)

        # Heads for 5 vertebral levels (left and right)
        self.ll1 = nn.Linear(1280, 2)
        self.ll2 = nn.Linear(1280, 2)
        self.ll3 = nn.Linear(1280, 2)
        self.ll4 = nn.Linear(1280, 2)
        self.ll5 = nn.Linear(1280, 2)
        
        self.rl1 = nn.Linear(1280, 2)
        self.rl2 = nn.Linear(1280, 2)
        self.rl3 = nn.Linear(1280, 2)
        self.rl4 = nn.Linear(1280, 2)
        self.rl5 = nn.Linear(1280, 2)

    def forward(self, x, label=None):
        # Reshape input to handle both left and right side images
        shape = x.shape
        x = x.reshape(shape[0] * shape[1], 3, shape[-2], shape[-1])  # Flatten the batch and side

        # Feature extraction through EfficientNetV2-S
        x = self.encoder.forward_features(x)
        x = self.flatten(x)
        x = x.reshape(shape[0], shape[1], -1)  # Split back into left and right

        # Split features for left and right
        x_left = x[:, 0, :]
        x_right = x[:, 1, :]

        # Apply Gated Attention
        x_left = self.attention(x_left)
        x_right = self.attention(x_right)

        # Project features into desired size
        x_left = self.projector(x_left)
        x_right = self.projector(x_right)

        # Apply Dropout
        x_left = self.dropout(x_left)
        x_right = self.dropout(x_right)

        # Predict each level for left and right sides
        output = {
            'left_L1/L2': self.ll1(x_left).sigmoid(),
            'left_L2/L3': self.ll2(x_left).sigmoid(),
            'left_L3/L4': self.ll3(x_left).sigmoid(),
            'left_L4/L5': self.ll4(x_left).sigmoid(),
            'left_L5/S1': self.ll5(x_left).sigmoid(),
            'right_L1/L2': self.rl1(x_right).sigmoid(),
            'right_L2/L3': self.rl2(x_right).sigmoid(),
            'right_L3/L4': self.rl3(x_right).sigmoid(),
            'right_L4/L5': self.rl4(x_right).sigmoid(),
            'right_L5/S1': self.rl5(x_right).sigmoid()
        }

        return output



# -------------------------------
# Dataset
# -------------------------------
class SpineCoorDataset(Dataset):
    def __init__(self, coor, meta, condition, mode):
        if condition == 'scs':
            self.coor = coor[coor.condition == "Spinal Canal Stenosis"]
        elif condition == 'nfn':
            self.coor = coor[coor.condition.isin([
                'Left Neural Foraminal Narrowing',
                'Right Neural Foraminal Narrowing'
            ])]
        elif condition == 'ss':
            self.coor = coor[coor.condition.isin([
                'Left Subarticular Stenosis',
                'Right Subarticular Stenosis'
            ])]

        g_coor = self.coor.groupby('study_id').count()
        if condition == 'scs':
            self.id = g_coor[g_coor.series_id == 5].reset_index().study_id.unique()
        else:
            self.id = g_coor[g_coor.series_id == 10].reset_index().study_id.unique()

        if condition == 'ss':
            self.resize = v2.Resize((256, 256))
        else:
            self.resize = v2.Resize((384, 384))

        # remove problematic ID
        self.id = list(set(self.id) - {3637444890})

        self.condition = condition
        self.meta = meta
        self.mode = mode

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        study_id = self.id[idx]
        volume, label = self.volume_scs(study_id)
        return volume, label

    def volume_scs(self, study_id):
        all_levels = ['L1/L2','L2/L3','L3/L4','L4/L5','L5/S1']
        meta = self.meta[(self.meta.study_id==study_id)&
                         (self.meta.series_description=='Sagittal T2/STIR')]
        meta = meta.sort_values('ipp_x').reset_index(drop=True)
        coor = self.coor[(self.coor.study_id==study_id)&
                         (self.coor.series_description=='Sagittal T2/STIR')]
        coor_dict, conf_dict = {}, {}

        # find central slice
        x_positions = torch.tensor([row.ipp_x for _,row in coor.iterrows()])
        median_x = x_positions.median().item()
        idx_row = meta[meta.ipp_x==median_x].index[0]
        slices = [idx_row-1, idx_row, idx_row+1]
        imgs = []
        for i in slices:
            r = meta.loc[i]
            arr = self.load_dicom(
                f"/kaggle/input/.../train_images/{study_id}/{r.series_id}/{r.instance_number}.dcm"
            )
            imgs.append(self.normalize(arr))
        h,w = imgs[1].shape

        for lvl in all_levels:
            lvl_coor = coor[coor.level==lvl]
            if not lvl_coor.empty:
                x = lvl_coor.iloc[0].x / w
                y = lvl_coor.iloc[0].y / h
                coor_dict[lvl] = torch.tensor([x,y])
                conf_dict[lvl] = torch.tensor(1.0)
            else:
                coor_dict[lvl] = torch.tensor([0.0,0.0])
                conf_dict[lvl] = torch.tensor(0.0)

        imgs = [self.resize(torch.tensor(im[None,...])) for im in imgs]
        volume = torch.cat(imgs,0).float()
        return volume, {'coor':coor_dict,'conf':conf_dict}

    def normalize(self, x):
        if self.condition=='ss':
            low,high = torch.quantile(torch.tensor(x),0.01).item(), torch.quantile(torch.tensor(x),0.99).item()
            x = np.clip(x,low,high)
        else:
            low,high = np.percentile(x,(1,99))
            x = np.clip(x,low,high)
        x = (x - x.min())/(x.max()-x.min())
        return x

    def load_dicom(self,path): return pydicom.dcmread(path).pixel_array


class MobileNetV3SmallSCSDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model('mobilenetv3_small_100',pretrained=True,in_chans=3,num_classes=0)
        self.in_features = self.encoder.num_features
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(1))
        self.gated_attention = GatedAttention(self.in_features)
        self.heads = nn.ModuleDict({lvl:nn.Linear(self.in_features,3)for lvl in ['L1/L2','L2/L3','L3/L4','L4/L5','L5/S1']})

    def forward(self,x):
        f = self.encoder.forward_features(x)
        f = self.flatten(f)
        f = self.gated_attention(f)
        out={}
        for lvl,head in self.heads.items():
            p = head(f)
            out[lvl] = {
                'coor': torch.sigmoid(p[:,:2]),
                'conf': p[:,2].unsqueeze(-1)  # raw logits
            }
        return out


# -------------------------------
# Loss
# -------------------------------
class SCSLoss(nn.Module):
    def __init__(self,lambda_conf=1.0):
        super().__init__()
        self.lambda_conf=lambda_conf

    def forward(self,outputs,targets):
        coord_loss=0.0; conf_loss=0.0; num_present=0.0
        B = next(iter(targets.values()))['conf'].shape[0]
        L = len(outputs)
        for lvl, pred in outputs.items():
            p_xy = pred['coor']; p_logit = pred['conf'].squeeze(-1)
            t_xy = targets[lvl]['coor']; t_conf = targets[lvl]['conf']
            mask = t_conf.unsqueeze(-1).expand_as(t_xy)
            coord_loss += F.smooth_l1_loss(p_xy*mask, t_xy*mask, reduction='sum')
            conf_loss  += F.binary_cross_entropy_with_logits(p_logit, t_conf, reduction='sum')
            num_present += t_conf.sum()
        coord_loss /= (num_present+1e-6)
        conf_loss  /= (B*L)
        return coord_loss + self.lambda_conf*conf_loss


# -------------------------------
# Metrics
# -------------------------------
def calculate_mse_score(outputs, targets):
    total_mse = 0.0
    count = 0
    for lvl in outputs:
        p_xy = outputs[lvl]['coor']
        t_xy = targets[lvl]['coor']
        mse = F.mse_loss(p_xy, t_xy, reduction='sum').item()
        total_mse += mse
        count += p_xy.numel()
    return total_mse / (count + 1e-6)


def calculate_regression_tolerance(outputs, targets, tolerances=[0,1,2]):
    counts = {f"Â±{tol}": 0 for tol in tolerances}
    counts[">Â±2"] = 0
    total = 0
    for lvl in outputs:
        p = outputs[lvl]['coor'].detach().cpu().numpy()
        t = targets[lvl]['coor'].detach().cpu().numpy()
        diffs = np.abs(p - t).astype(int)
        for drow in diffs:
            for d in drow:
                matched = False
                for tol in tolerances:
                    if d <= tol:
                        counts[f"Â±{tol}"] += 1
                        matched = True
                        break
                if not matched:
                    counts[">Â±2"] += 1
                total += 1
    return counts, total


# -------------------------------
# Training
# -------------------------------
def train_spine_model(model, train_coor, train_meta,
                      n_folds=5, epochs=18, batch_size=4,
                      learning_rate=1e-3, weight_decay=1e-4,
                      patience=5, mixed_precision=True,
                      experiment_name="spine_model"):

    os.makedirs("models", exist_ok=True)
    writer = SummaryWriter(f"runs/{experiment_name}")
    scaler = torch.amp.GradScaler() if mixed_precision else None

    all_val_mses = []
    for fold in range(n_folds):
        print(f"-- Fold {fold+1}/{n_folds} --")
        # Datasets & Loaders
        train_ds = SpineCoorDataset(
            train_coor[train_coor.fold != fold],
            train_meta[train_meta.fold != fold],
            'scs','train'
        )
        val_ds = SpineCoorDataset(
            train_coor[train_coor.fold == fold],
            train_meta[train_meta.fold == fold],
            'scs','valid'
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=2, pin_memory=True)

        optimizer = AdamW(model.parameters(), lr=learning_rate,
                          weight_decay=weight_decay)
        total_steps = epochs * len(train_loader)
        warmup = int(0.1 * total_steps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps)

        criterion = SCSLoss(lambda_conf=1.0)
        expected_levels = ['L1/L2','L2/L3','L3/L4','L4/L5','L5/S1']

        best_mse = float('inf'); counter=0
        for epoch in range(epochs):
            print(f" Epoch {epoch+1}/{epochs}")
            # Training
            model.train()
            train_loss = train_mse = 0.0
            tol_counts = {f"Â±{t}":0 for t in [0,1,2]}; tol_counts[">Â±2"]=0
            total_preds=0
            for vol, batch in tqdm(train_loader):
                vol = vol.to(device)
                # prepare targets per level
                targets = {lvl: {'coor': batch['coor'][lvl].to(device),
                                 'conf': batch['conf'][lvl].to(device)}
                           for lvl in expected_levels}

                optimizer.zero_grad()
                if mixed_precision:
                    with torch.amp.autocast(device_type='cuda'):
                        out = model(vol)
                        loss = criterion(out, targets)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(vol)
                    loss = criterion(out, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    optimizer.step()

                scheduler.step()
                train_loss += loss.item()
                mse = calculate_mse_score(out, targets)
                train_mse += mse
                bc, bt = calculate_regression_tolerance(out, targets)
                for k in tol_counts: tol_counts[k] += bc[k]
                total_preds += bt
                del vol, batch, out, loss

            # compute averages
            train_loss /= len(train_loader)
            train_mse /= len(train_loader)
            for k in tol_counts: tol_counts[k] = 100 * tol_counts[k]/(total_preds+1e-6)
            print(f" Train Loss: {train_loss:.4f}, MSE: {train_mse:.4f}")
            print(" Tolerances: ", tol_counts)

            # Validation
            model.eval()
            val_loss= val_mse=0.0
            val_tcounts = {f"Â±{t}":0 for t in [0,1,2]}; val_tcounts[">Â±2"]=0
            vpreds=0
            with torch.no_grad():
                for vol, batch in tqdm(val_loader):
                    vol=vol.to(device)
                    targets = {lvl:{'coor': batch['coor'][lvl].to(device),
                                    'conf': batch['conf'][lvl].to(device)}
                               for lvl in expected_levels}
                    out = model(vol)
                    loss = criterion(out, targets)
                    val_loss += loss.item()
                    mse = calculate_mse_score(out, targets)
                    val_mse += mse
                    bc,bt = calculate_regression_tolerance(out, targets)
                    for k in val_tcounts: val_tcounts[k]+=bc[k]
                    vpreds += bt
            val_loss /= len(val_loader)
            val_mse  /= len(val_loader)
            for k in val_tcounts: val_tcounts[k]=100*val_tcounts[k]/(vpreds+1e-6)
            print(f" Val Loss: {val_loss:.4f}, MSE: {val_mse:.4f}")
            print(" Val Tolerances: ", val_tcounts)

            # Early stopping & checkpoints
            if val_mse < best_mse:
                best_mse=val_mse; counter=0
                torch.save(model.state_dict(), f"models/best_fold{fold}.pth")
                print(" Saved best model.")
            else:
                counter+=1
                if counter>=patience:
                    print("Early stopping.")
                    break

        all_val_mses.append(best_mse)

    avg_mse = sum(all_val_mses)/len(all_val_mses)
    print("CV MSEs:", all_val_mses)
    print("Avg CV MSE:", avg_mse)
    return all_val_mses, avg_mse




model = MobileNetV3SmallSCSDetect().to(device)
mses, avg_mse = train_spine_model(model, train_coor, train_meta, n_folds=5, )


# Load model weights
def load_model_weights(model, weight_path, device):
    pretrained_dict = torch.load(weight_path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

# Image preprocessing
def preprocess_image(img_path):
    dicom_data = pydicom.dcmread(img_path)
    img = dicom_data.pixel_array

    # Normalize image (scaling intensity to [0, 1])
    lower, upper = np.percentile(img, (1, 99))
    img = np.clip(img, lower, upper)
    img = img - np.min(img)
    img = img / np.max(img)

    img = torch.tensor(img).unsqueeze(0).float()  # (1, H, W)
    img = v2.Resize((384, 384))(img)
    img = img.repeat(3, 1, 1)                     # (3, H, W)
    img = img.unsqueeze(0).to(device)              # (B=1, C=3, H, W)

    return img

# Predict coordinates and confidences
@torch.no_grad()
def predict_coordinates(img_path, model, device):
    img = preprocess_image(img_path)
    outputs = model(img)

    coordinates = {}
    confidences = {}

    for level, out in outputs.items():
        # out is a Tensor directly, (x, y, confidence)
        coor = out['coor'].squeeze(0).cpu().numpy()  # (x, y)
        conf = out['conf'].squeeze(0).cpu().numpy()  # confidence
        
        coordinates[level] = (float(coor[0]), float(coor[1]))
        confidences[level] = float(conf[0])

    return coordinates, confidences

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MobileNetV3SmallSCSDetect()
load_model_weights(model, '/kaggle/working/models/spine_model_fold_0_final.pt', device)
model = model.to(device)
model.eval()

# Example image path (change to actual path in your dataset)
img_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/4003253/702807833/1.dcm'

# Get coordinates and confidences
coordinates, confidences = predict_coordinates(img_path, model, device)

threshold = 0.5
for level in coordinates:
    x, y = coordinates[level]
    conf = confidences[level]
    if conf > threshold:
        print(f"{level}: x = {x:.2f}, y = {y:.2f}, confidence = {conf:.2f}")
    else:
        print(f"{level}: Prediction is below threshold")



# ----------------- Inference -----------------
def load_model_weights(model, weight_path, device='cuda'):
    """Load model weights from checkpoint"""
    checkpoint = torch.load(weight_path, map_location=device)
    
    # Check if checkpoint has 'model_state_dict' key
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint  # Assume it's just the state dict
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    return model


def preprocess_image(img_path, device='cuda'):
    """Preprocess image for inference"""
    dicom_data = pydicom.dcmread(img_path)
    img = dicom_data.pixel_array
    
    # Handle different pixel value ranges
    if img.max() > 0:  # Avoid division by zero
        lower, upper = np.percentile(img, (1, 99))
        img = np.clip(img, lower, upper)
        img = (img - lower) / max(upper - lower, 1e-8)  # Normalize to [0,1]
    else:
        # Handle empty or invalid images
        img = np.zeros_like(img)
    
    # Convert to tensor
    img = torch.tensor(img).unsqueeze(0).float()  # (1, H, W)
    img = v2.Resize((384, 384))(img)
    img = img.repeat(3, 1, 1)                     # (3, H, W)
    img = img.unsqueeze(0).to(device)             # (B=1, C=3, H, W)
    
    return img


@torch.no_grad()
def predict_coordinates(model, img_path, confidence_threshold=0.5, device='cuda'):
    """
    Predict coordinates for spinal levels with confidence scores
    Only returns levels where confidence is above threshold
    """
    # Preprocess image
    img = preprocess_image(img_path, device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get predictions
    outputs = model(img)
    
    # Process predictions
    coordinates = {}
    confidence_scores = {}
    
    for level, out in outputs.items():
        x, y, conf = out.squeeze(0).cpu().numpy()
        confidence_scores[level] = float(conf)
        
        # Only include coordinates with confidence above threshold
        if conf >= confidence_threshold:

