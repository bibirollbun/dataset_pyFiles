# ==================== COMPLETE IMPROVED 1D CNN SOLUTION WITH RESIDUAL CONNECTIONS ====================

validate_or_submit = 'submit'
verbose = True

import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import warnings
import json
import os
import gc
from collections import defaultdict
import polars as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== SCORING FUNCTIONS ====================

class HostVisibleError(Exception):
    pass

def single_lab_f1(lab_solution: pl.DataFrame, lab_submission: pl.DataFrame, beta: float = 1) -> float:
    label_frames: defaultdict[str, set[int]] = defaultdict(set)
    prediction_frames: defaultdict[str, set[int]] = defaultdict(set)

    for row in lab_solution.to_dicts():
        label_frames[row['label_key']].update(range(row['start_frame'], row['stop_frame']))

    for video in lab_solution['video_id'].unique():
        active_labels: str = lab_solution.filter(pl.col('video_id') == video)['behaviors_labeled'].first()
        active_labels: set[str] = set(json.loads(active_labels))
        predicted_mouse_pairs: defaultdict[str, set[int]] = defaultdict(set)

        for row in lab_submission.filter(pl.col('video_id') == video).to_dicts():
            if ','.join([str(row['agent_id']), str(row['target_id']), row['action']]) not in active_labels:
                continue
           
            new_frames = set(range(row['start_frame'], row['stop_frame']))
            new_frames = new_frames.difference(prediction_frames[row['prediction_key']])
            prediction_pair = ','.join([str(row['agent_id']), str(row['target_id'])])
            if predicted_mouse_pairs[prediction_pair].intersection(new_frames):
                raise HostVisibleError('Multiple predictions for the same frame from one agent/target pair')
            prediction_frames[row['prediction_key']].update(new_frames)
            predicted_mouse_pairs[prediction_pair].update(new_frames)

    tps = defaultdict(int)
    fns = defaultdict(int)
    fps = defaultdict(int)
    for key, pred_frames in prediction_frames.items():
        action = key.split('_')[-1]
        matched_label_frames = label_frames[key]
        tps[action] += len(pred_frames.intersection(matched_label_frames))
        fns[action] += len(matched_label_frames.difference(pred_frames))
        fps[action] += len(pred_frames.difference(matched_label_frames))

    distinct_actions = set()
    for key, frames in label_frames.items():
        action = key.split('_')[-1]
        distinct_actions.add(action)
        if key not in prediction_frames:
            fns[action] += len(frames)

    action_f1s = []
    for action in distinct_actions:
        if tps[action] + fns[action] + fps[action] == 0:
            action_f1s.append(0)
        else:
            action_f1s.append((1 + beta**2) * tps[action] / ((1 + beta**2) * tps[action] + beta**2 * fns[action] + fps[action]))
    return sum(action_f1s) / len(action_f1s)

# ==================== DATA LOADING ====================

train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
test = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')

train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)
body_parts_tracked_list = list(np.unique(train.body_parts_tracked))

drop_body_parts = ['headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 
                   'headpiece_bottomfrontright', 'headpiece_topbackleft', 'headpiece_topbackright', 
                   'headpiece_topfrontleft', 'headpiece_topfrontright', 'spine_1', 'spine_2', 
                   'tail_middle_1', 'tail_middle_2', 'tail_midpoint']

def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    assert traintest in ['train', 'test']
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    
    for _, row in dataset.iterrows():
        lab_id = row.lab_id
        if lab_id.startswith('MABe22'): 
            continue
        video_id = row.video_id

        if type(row.behaviors_labeled) != str:
            if verbose: 
                print('No labeled behaviors:', lab_id, video_id)
            continue

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
        
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        pvid /= row.pix_per_cm_approx

        vid_behaviors = json.loads(row.behaviors_labeled)
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
        
        if traintest == 'train':
            try:
                annot = pd.read_parquet(path.replace('train_tracking', 'train_annotation'))
            except FileNotFoundError:
                continue

        if generate_single:
            vid_behaviors_subset = vid_behaviors.query("target == 'self'")
            for mouse_id_str in np.unique(vid_behaviors_subset.agent):
                try:
                    mouse_id = int(mouse_id_str[-1])
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("agent == @mouse_id_str").action)
                    single_mouse = pvid.loc[:, mouse_id]
                    single_mouse_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': mouse_id_str,
                        'target_id': 'self',
                        'video_frame': single_mouse.index
                    })
                    if traintest == 'train':
                        single_mouse_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=single_mouse.index)
                        annot_subset = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            single_mouse_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'single', single_mouse, single_mouse_meta, single_mouse_label
                    else:
                        yield 'single', single_mouse, single_mouse_meta, vid_agent_actions
                except KeyError:
                    pass

        if generate_pair:
            vid_behaviors_subset = vid_behaviors.query("target != 'self'")
            if len(vid_behaviors_subset) > 0:
                for agent, target in itertools.permutations(np.unique(pvid.columns.get_level_values('mouse_id')), 2):
                    agent_str = f"mouse{agent}"
                    target_str = f"mouse{target}"
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("(agent == @agent_str) & (target == @target_str)").action)
                    if len(vid_agent_actions) == 0:
                        continue
                    mouse_pair = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                    mouse_pair_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': agent_str,
                        'target_id': target_str,
                        'video_frame': mouse_pair.index
                    })
                    if traintest == 'train':
                        mouse_pair_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=mouse_pair.index)
                        annot_subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            mouse_pair_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'pair', mouse_pair, mouse_pair_meta, mouse_pair_label
                    else:
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions

# ==================== FEATURE ENGINEERING ====================

def transform_single(single_mouse, body_parts_tracked):
    """Feature extraction for single mouse"""
    available_body_parts = single_mouse.columns.get_level_values(0)
    
    X = pd.DataFrame({
        f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.combinations(body_parts_tracked, 2) 
        if p1 in available_body_parts and p2 in available_body_parts
    })
    X = X.reindex(columns=[f"{p1}+{p2}" for p1, p2 in itertools.combinations(body_parts_tracked, 2)], copy=False)

    if all(p in single_mouse.columns for p in ['ear_left', 'ear_right', 'tail_base']):
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(10)
        X['sp_lf'] = np.square(single_mouse['ear_left'] - shifted['ear_left']).sum(axis=1, skipna=False)
        X['sp_rt'] = np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False)
        X['sp_tb'] = np.square(single_mouse['tail_base'] - shifted['tail_base']).sum(axis=1, skipna=False)
    
    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']
        cy = single_mouse['body_center']['y']
        
        for w in [5, 15, 30, 60]:
            X[f'cx_m{w}'] = cx.rolling(w, min_periods=1, center=True).mean()
            X[f'cy_m{w}'] = cy.rolling(w, min_periods=1, center=True).mean()
            X[f'cx_s{w}'] = cx.rolling(w, min_periods=1, center=True).std()
            X[f'cy_s{w}'] = cy.rolling(w, min_periods=1, center=True).std()
    
    return X

def transform_pair(mouse_pair, body_parts_tracked):
    """Feature extraction for mouse pairs"""
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)
    
    X = pd.DataFrame({
        f"12+{p1}+{p2}": np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.product(body_parts_tracked, repeat=2) 
        if p1 in avail_A and p2 in avail_B
    })
    X = X.reindex(columns=[f"12+{p1}+{p2}" for p1, p2 in itertools.product(body_parts_tracked, repeat=2)], copy=False)

    if ('A', 'body_center') in mouse_pair.columns and ('B', 'body_center') in mouse_pair.columns:
        cd = np.square(mouse_pair['A']['body_center'] - mouse_pair['B']['body_center']).sum(axis=1, skipna=False)
        for w in [5, 15, 30, 60]:
            X[f'd_m{w}'] = cd.rolling(w, min_periods=1, center=True).mean()
            X[f'd_s{w}'] = cd.rolling(w, min_periods=1, center=True).std()
    
    return X

# ==================== FOCAL LOSS FOR EXTREME CLASS IMBALANCE ====================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==================== RESIDUAL BLOCK ====================

class ResidualBlock1D(nn.Module):
    """1D Residual block with batch normalization and dropout"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        out = self.dropout(out)
        
        return out

# ==================== IMPROVED CNN WITH RESIDUAL CONNECTIONS ====================

class ImprovedConv1DClassifier(nn.Module):
    """Enhanced 1D CNN with residual connections and multi-scale features"""
    def __init__(self, input_dim, num_classes, window_size=60, dropout=0.3):
        super().__init__()
        self.window_size = window_size
        
        self.conv_init = nn.Conv1d(input_dim, 64, kernel_size=7, padding=3)
        self.bn_init = nn.BatchNorm1d(64)
        
        self.res_block1 = ResidualBlock1D(64, 128, kernel_size=5, dropout=dropout)
        self.res_block2 = ResidualBlock1D(128, 128, kernel_size=5, dropout=dropout)
        self.pool1 = nn.MaxPool1d(2)
        
        self.res_block3 = ResidualBlock1D(128, 256, kernel_size=3, dropout=dropout)
        self.res_block4 = ResidualBlock1D(256, 256, kernel_size=3, dropout=dropout)
        self.pool2 = nn.MaxPool1d(2)
        
        self.res_block5 = ResidualBlock1D(256, 256, kernel_size=3, dropout=dropout)
        self.res_block6 = ResidualBlock1D(256, 128, kernel_size=3, dropout=dropout)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        pooled_size = window_size // (2 ** 2)
        
        self.fc1 = nn.Linear(128 * pooled_size + 128 * 2, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn_fc3 = nn.BatchNorm1d(128)
        
        self.fc_out = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = F.relu(self.bn_init(self.conv_init(x)))
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.pool1(x)
        
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.pool2(x)
        
        x = self.res_block5(x)
        x = self.res_block6(x)
        
        x_flat = x.flatten(1)
        x_avg = self.global_avg_pool(x).squeeze(-1)
        x_max = self.global_max_pool(x).squeeze(-1)
        
        x = torch.cat([x_flat, x_avg, x_max], dim=1)
        
        x = self.dropout_fc(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout_fc(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.dropout_fc(F.relu(self.bn_fc3(self.fc3(x))))
        x = self.fc_out(x)
        
        return x

# ==================== IMPROVED DATASET WITH AUGMENTATION ====================

class ImprovedMouseBehaviorDataset(Dataset):
    """Dataset with support for weighted sampling and augmentation"""
    def __init__(self, features, labels, window_size=60, augment=False):
        self.features = features
        self.labels = labels
        self.window_size = window_size
        self.half_window = window_size // 2
        self.augment = augment
        
        self.valid_indices = list(range(self.half_window, len(features) - self.half_window))
        self.sample_weights = self._calculate_weights()
        
    def _calculate_weights(self):
        """Calculate sample weights based on label frequency"""
        weights = np.ones(len(self.valid_indices))
        
        for i, idx in enumerate(self.valid_indices):
            if self.labels[idx, 0] == 1:
                weights[i] = 10.0
        
        return weights
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        center_idx = self.valid_indices[idx]
        
        window = self.features[center_idx - self.half_window:center_idx + self.half_window]
        label = self.labels[center_idx]
        
        if self.augment and label[0] == 1:
            window = self._augment(window)
        
        return torch.FloatTensor(window), torch.FloatTensor(label)
    
    def _augment(self, window):
        """Apply data augmentation"""
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.01, window.shape)
            window = window + noise
        
        if np.random.rand() < 0.3:
            factor = np.random.uniform(0.9, 1.1)
            new_len = int(len(window) * factor)
            indices = np.linspace(0, len(window)-1, new_len)
            window = np.array([window[int(i)] for i in indices])
            if len(window) < self.window_size:
                pad = self.window_size - len(window)
                window = np.pad(window, ((0, pad), (0, 0)), mode='edge')
            else:
                window = window[:self.window_size]
        
        return window

# ==================== TRAINING FUNCTIONS ====================

def calculate_f1(preds, labels):
    """Helper function to calculate F1 score"""
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return f1

def train_model_improved(model, train_loader, val_loader, num_epochs=25, lr=1e-3, 
                        patience=5, use_focal_loss=False, pos_weight=None):
    """Enhanced training with multiple strategies for class imbalance"""
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    scheduler = OneCycleLR(optimizer, max_lr=lr*10, 
                          steps_per_epoch=len(train_loader), 
                          epochs=num_epochs, pct_start=0.3)
    
    if use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        if verbose:
            print("    Using Focal Loss")
    elif pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        if verbose:
            print(f"    Using Weighted BCE Loss (weight={pos_weight:.2f})")
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    best_val_f1 = 0
    patience_counter = 0
    best_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds_all = []
        train_labels_all = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_preds_all.append(preds.cpu())
            train_labels_all.append(batch_y.cpu())
        
        train_loss /= len(train_loader)
        
        train_preds_all = torch.cat(train_preds_all)
        train_labels_all = torch.cat(train_labels_all)
        train_f1 = calculate_f1(train_preds_all, train_labels_all)
        
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                all_preds.append(preds.cpu())
                all_labels.append(batch_y.cpu())
        
        val_loss /= len(val_loader)
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        val_f1 = calculate_f1(all_preds, all_labels)
        
        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}/{val_loss:.4f}, '
                  f'F1: {train_f1:.4f}/{val_f1:.4f}')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model

def find_best_threshold_advanced(model, val_loader, device):
    """Advanced threshold finding with multiple metrics"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = torch.sigmoid(model(batch_x))
            all_probs.append(outputs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    best_f1 = 0
    best_thresh = 0.5
    best_metrics = {}
    
    for thresh in np.arange(0.05, 0.95, 0.025):
        preds = (all_probs > thresh).astype(int)
        
        tp = ((preds == 1) & (all_labels == 1)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
    
    return best_thresh, best_f1, best_metrics

def predict_with_model(model, features, window_size=60):
    """Generate predictions for full sequence"""
    model.eval()
    half_window = window_size // 2
    
    all_probs = []
    
    with torch.no_grad():
        for i in range(half_window, len(features) - half_window):
            window = features[i - half_window:i + half_window]
            window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
            
            output = model(window_tensor)
            probs = torch.sigmoid(output).cpu().numpy()[0]
            all_probs.append(probs)
    
    all_probs = np.array(all_probs)
    
    full_probs = np.zeros((len(features), all_probs.shape[1]))
    full_probs[half_window:-half_window] = all_probs
    full_probs[:half_window] = all_probs[0]
    full_probs[-half_window:] = all_probs[-1]
    
    return full_probs

# ==================== PREDICTION TO SUBMISSION ====================

def predict_multiclass_adaptive(pred, meta, action_thresholds, min_duration=3):
    """Convert frame predictions to submission format"""
    pred_smoothed = pd.DataFrame(pred, columns=pred.columns if isinstance(pred, pd.DataFrame) else range(pred.shape[1]))
    pred_smoothed = pred_smoothed.rolling(window=5, min_periods=1, center=True).mean()
    
    ama = np.argmax(pred_smoothed.values, axis=1)
    max_probs = pred_smoothed.values.max(axis=1)
    
    threshold_mask = np.zeros(len(pred_smoothed), dtype=bool)
    for i, action in enumerate(pred_smoothed.columns):
        action_mask = (ama == i)
        threshold = action_thresholds.get(action, 0.27)
        threshold_mask |= (action_mask & (max_probs >= threshold))
    
    ama = np.where(threshold_mask, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame.values if hasattr(meta, 'video_frame') else range(len(ama)))
    
    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]
    mask = ama_changes.values >= 0
    mask[-1] = False
    
    if mask.sum() == 0:
        return pd.DataFrame(columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
    
    submission_part = pd.DataFrame({
        'video_id': meta_changes['video_id'].values[mask],
        'agent_id': meta_changes['agent_id'].values[mask],
        'target_id': meta_changes['target_id'].values[mask],
        'action': pred_smoothed.columns[ama_changes.values[mask]],
        'start_frame': ama_changes.index[mask],
        'stop_frame': ama_changes.index[1:][mask[:-1]]
    })
    
    for i in range(len(submission_part)):
        video_id = submission_part.video_id.iloc[i]
        if i < len(submission_part) - 1:
            if submission_part.video_id.iloc[i+1] != video_id:
                new_stop = meta.query("video_id == @video_id").video_frame.max() + 1
                submission_part.at[submission_part.index[i], 'stop_frame'] = new_stop
        else:
            new_stop = meta.query("video_id == @video_id").video_frame.max() + 1
            submission_part.at[submission_part.index[i], 'stop_frame'] = new_stop
    
    duration = submission_part.stop_frame - submission_part.start_frame
    submission_part = submission_part[duration >= min_duration].reset_index(drop=True)
    
    return submission_part

# ==================== MAIN PROCESSING FUNCTION ====================

def process_with_improved_cnn(body_parts_tracked_str, switch_tr, X_tr, label, meta):
    """Train improved CNN with residual connections and better class imbalance handling"""
    
    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    
    X_tr_filled = X_tr.fillna(0).values
    
    X_tr_clipped = np.clip(X_tr_filled, 
                           np.percentile(X_tr_filled, 1), 
                           np.percentile(X_tr_filled, 99))
    
    mean = X_tr_clipped.mean(axis=0, keepdims=True)
    std = X_tr_clipped.std(axis=0, keepdims=True) + 1e-8
    X_tr_normalized = (X_tr_clipped - mean) / std
    
    action_thresholds = {}
    models_dict = {}
    
    for action in label.columns:
        action_mask = ~label[action].isna().values
        y_action = label[action][action_mask].values
        
        if not (y_action == 0).all() and y_action.sum() >= 5:
            pos_count = y_action.sum()
            neg_count = (y_action == 0).sum()
            pos_weight = neg_count / pos_count
            imbalance_ratio = neg_count / pos_count
            
            if verbose:
                print(f"  Training Improved CNN for action: {action}")
                print(f"    Positive: {int(pos_count)}, Negative: {int(neg_count)}, "
                      f"Ratio: 1:{imbalance_ratio:.1f}")
            
            X_action = X_tr_normalized[action_mask]
            
            use_focal = imbalance_ratio > 100
            use_augmentation = pos_count < 500
            
            dataset = ImprovedMouseBehaviorDataset(
                X_action, y_action.reshape(-1, 1), 
                window_size=60,
                augment=use_augmentation
            )
            
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            if imbalance_ratio > 50:
                train_indices = train_dataset.indices
                train_weights = [dataset.sample_weights[i] for i in train_indices]
                sampler = WeightedRandomSampler(
                    weights=train_weights,
                    num_samples=len(train_weights),
                    replacement=True
                )
                train_loader = DataLoader(train_dataset, batch_size=32, 
                                        sampler=sampler, num_workers=0)
            else:
                train_loader = DataLoader(train_dataset, batch_size=32, 
                                        shuffle=True, num_workers=0)
            
            val_loader = DataLoader(val_dataset, batch_size=64, 
                                  shuffle=False, num_workers=0)
            
            model = ImprovedConv1DClassifier(
                input_dim=X_tr_normalized.shape[1],
                num_classes=1,
                window_size=60,
                dropout=0.3
            ).to(device)
            
            model = train_model_improved(
                model, train_loader, val_loader,
                num_epochs=25 if pos_count < 200 else 20,
                lr=5e-4 if pos_count < 100 else 1e-3,
                patience=5,
                use_focal_loss=use_focal,
                pos_weight=pos_weight if not use_focal else None
            )
            
            best_thresh, best_f1, metrics = find_best_threshold_advanced(
                model, val_loader, device
            )
            action_thresholds[action] = best_thresh
            
            if verbose:
                print(f"    Best threshold: {best_thresh:.3f}, Val F1: {best_f1:.4f}")
                print(f"    Precision: {metrics['precision']:.3f}, "
                      f"Recall: {metrics['recall']:.3f}")
            
            models_dict[action] = (model, mean, std)
            
            gc.collect()
            torch.cuda.empty_cache()
        
        elif verbose:
            print(f"  Skipping {action}: insufficient positive samples ({int(y_action.sum())})")
    
    test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
    generator = generate_mouse_data(test_subset, 'test',
                                    generate_single=(switch_tr == 'single'), 
                                    generate_pair=(switch_tr == 'pair'))
    
    submission_list = []
    
    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr
        
        try:
            if switch_te == 'single':
                X_te = transform_single(data_te, body_parts_tracked)
            else:
                X_te = transform_pair(data_te, body_parts_tracked)
            
            X_te_filled = X_te.fillna(0).values
            X_te_clipped = np.clip(X_te_filled,
                                  np.percentile(X_tr_filled, 1),
                                  np.percentile(X_tr_filled, 99))
            
            pred = pd.DataFrame(index=meta_te.video_frame)
            
            for action in models_dict.keys():
                if action in actions_te:
                    model, mean, std = models_dict[action]
                    X_te_normalized = (X_te_clipped - mean) / std
                    probs = predict_with_model(model, X_te_normalized)
                    pred[action] = probs[:, 0]
            
            if pred.shape[1] != 0:
                sub_part = predict_multiclass_adaptive(pred, meta_te, action_thresholds)
                submission_list.append(sub_part)
            
            del X_te, X_te_filled
            gc.collect()
            
        except Exception as e:
            if verbose:
                print(f'  ERROR: {str(e)[:100]}')
            gc.collect()
    
    return submission_list

# ==================== MAIN EXECUTION ====================

submission_list = []

print(f"\nProcessing {len(body_parts_tracked_list)} body part configurations...\n")

for section in range(1, min(3, len(body_parts_tracked_list))):
    body_parts_tracked_str = body_parts_tracked_list[section]
    
    try:
        body_parts_tracked = json.loads(body_parts_tracked_str)
        print(f"\n{section}. Processing: {len(body_parts_tracked)} body parts")
        
        if len(body_parts_tracked) > 5:
            body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
        
        train_subset = train[train.body_parts_tracked == body_parts_tracked_str]
        
        single_list, single_label_list, single_meta_list = [], [], []
        
        for switch, data, meta, label in generate_mouse_data(train_subset, 'train'):
            if switch == 'single':
                single_list.append(data)
                single_meta_list.append(meta)
                single_label_list.append(label)
            if len(single_list) >= 5:
                break
        
        if len(single_list) > 0:
            print(f"  Single mouse sequences: {len(single_list)}")
            single_mouse = pd.concat(single_list)
            single_label = pd.concat(single_label_list)
            single_meta = pd.concat(single_meta_list)
            
            X_tr = transform_single(single_mouse, body_parts_tracked)
            print(f"  Single features shape: {X_tr.shape}")
            
            sub_parts = process_with_improved_cnn(body_parts_tracked_str, 'single', X_tr, single_label, single_meta)
            submission_list.extend(sub_parts)
            
            del single_mouse, single_label, single_meta, X_tr
            gc.collect()
        
        pair_list, pair_label_list, pair_meta_list = [], [], []
        
        for switch, data, meta, label in generate_mouse_data(train_subset, 'train'):
            if switch == 'pair':
                pair_list.append(data)
                pair_meta_list.append(meta)
                pair_label_list.append(label)
            if len(pair_list) >= 5:
                break
        
        if len(pair_list) > 0:
            print(f"  Pair sequences: {len(pair_list)}")
            mouse_pair = pd.concat(pair_list)
            pair_label = pd.concat(pair_label_list)
            pair_meta = pd.concat(pair_meta_list)
            
            X_tr = transform_pair(mouse_pair, body_parts_tracked)
            print(f"  Pair features shape: {X_tr.shape}")
            
            sub_parts = process_with_improved_cnn(body_parts_tracked_str, 'pair', X_tr, pair_label, pair_meta)
            submission_list.extend(sub_parts)
            
            del mouse_pair, pair_label, pair_meta, X_tr
            gc.collect()
        
    except Exception as e:
        print(f'***Exception*** {str(e)[:100]}')
    
    gc.collect()
    torch.cuda.empty_cache()

# ==================== CREATE FINAL SUBMISSION ====================

if len(submission_list) > 0:
    submission = pd.concat(submission_list, ignore_index=True)
else:
    submission = pd.DataFrame({
        'video_id': [438887472],
        'agent_id': ['mouse1'],
        'target_id': ['self'],
        'action': ['rear'],
        'start_frame': [278],
        'stop_frame': [500]
    })

submission = submission[submission.start_frame < submission.stop_frame].reset_index(drop=True)
submission.index.name = 'row_id'
submission.to_csv('submission.csv')

print(f"\n✓ Submission created: {len(submission)} predictions")
print(f"  Actions found: {submission.action.nunique()}")
print(f"  Videos: {submission.video_id.nunique()}")

