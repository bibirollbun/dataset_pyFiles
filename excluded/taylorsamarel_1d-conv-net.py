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


# ==================== COMPLETE MABe SOLUTION WITH DUAL-SCALE CNN ====================

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
from torch.optim.lr_scheduler import CosineAnnealingLR

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

# ==================== FOCAL LOSS FOR CLASS IMBALANCE ====================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# ==================== DUAL-SCALE 1D CNN MODEL ====================

class DualScaleConv1D(nn.Module):
    def __init__(self, input_dim, num_classes, local_window=60, 
                 long_window=200, sparse_rate=5, dropout=0.3):
        super().__init__()
        self.local_window = local_window
        self.long_window = long_window
        self.sparse_rate = sparse_rate
        self.n_sparse = long_window // sparse_rate
        
        # Local branch (fine-grained)
        self.local_conv1 = nn.Conv1d(input_dim, 128, 5, padding=2)
        self.local_bn1 = nn.BatchNorm1d(128)
        self.local_conv2 = nn.Conv1d(128, 256, 5, padding=2)
        self.local_bn2 = nn.BatchNorm1d(256)
        self.local_conv3 = nn.Conv1d(256, 256, 3, padding=1)
        self.local_bn3 = nn.BatchNorm1d(256)
        
        # Long branch (sparse)
        self.long_conv1 = nn.Conv1d(input_dim, 64, 3, padding=1)
        self.long_bn1 = nn.BatchNorm1d(64)
        self.long_conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.long_bn2 = nn.BatchNorm1d(128)
        self.long_conv3 = nn.Conv1d(128, 128, 3, padding=1)
        self.long_bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        
        local_sz = local_window // 4
        long_sz = self.n_sparse // 4
        fusion_dim = 256 * local_sz + 128 * long_sz
        
        self.fc1 = nn.Linear(fusion_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x_local, x_long):
        # Local branch
        x_l = x_local.transpose(1, 2)
        x_l = self.dropout(self.pool(F.relu(self.local_bn1(self.local_conv1(x_l)))))
        x_l = self.dropout(self.pool(F.relu(self.local_bn2(self.local_conv2(x_l)))))
        x_l = self.dropout(F.relu(self.local_bn3(self.local_conv3(x_l))))
        x_l = x_l.flatten(1)
        
        # Long branch
        x_g = x_long.transpose(1, 2)
        x_g = self.dropout(self.pool(F.relu(self.long_bn1(self.long_conv1(x_g)))))
        x_g = self.dropout(self.pool(F.relu(self.long_bn2(self.long_conv2(x_g)))))
        x_g = self.dropout(F.relu(self.long_bn3(self.long_conv3(x_g))))
        x_g = x_g.flatten(1)
        
        # Fusion
        x = torch.cat([x_l, x_g], 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)

# ==================== DUAL-SCALE DATASET ====================

class DualScaleDataset(Dataset):
    def __init__(self, features, labels, local_window=60, 
                 long_window=200, sparse_rate=5):
        self.features = features
        self.labels = labels
        self.local_window = local_window
        self.long_window = long_window
        self.sparse_rate = sparse_rate
        self.half_local = local_window // 2
        self.half_long = long_window // 2
        self.valid_indices = list(range(self.half_long, len(features) - self.half_long))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        center = self.valid_indices[idx]
        local = self.features[center - self.half_local:center + self.half_local]
        sparse_idx = list(range(center - self.half_long, center + self.half_long, self.sparse_rate))
        long_range = self.features[sparse_idx]
        return (torch.FloatTensor(local), 
                torch.FloatTensor(long_range), 
                torch.FloatTensor(self.labels[center]))

# ==================== CLASS-BALANCED SAMPLING ====================

def create_balanced_sampler(labels, oversample_ratio=2.0):
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return None
    weights = np.zeros(len(labels))
    weights[labels == 1] = (n_neg / n_pos) * oversample_ratio
    weights[labels == 0] = 1.0
    return WeightedRandomSampler(weights, len(weights), replacement=True)

# ==================== TRAINING FUNCTION ====================

def train_dual_model(model, train_loader, val_loader, epochs=15, lr=1e-3, patience=3):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    best_loss = float('inf')
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for local, long_range, labels in train_loader:
            local, long_range, labels = local.to(device), long_range.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(local, long_range), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for local, long_range, labels in val_loader:
                local, long_range, labels = local.to(device), long_range.to(device), labels.to(device)
                val_loss += criterion(model(local, long_range), labels).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if verbose:
            print(f'  Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}')
        
        scheduler.step()
        
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            best_state = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                if verbose:
                    print(f'  Early stop at epoch {epoch+1}')
                break
    
    model.load_state_dict(best_state)
    return model

# ==================== PREDICTION FUNCTION ====================

def predict_dual(model, features, local_w=60, long_w=200, sparse_r=5):
    model.eval()
    half_local = local_w // 2
    half_long = long_w // 2
    probs = []
    
    with torch.no_grad():
        for i in range(half_long, len(features) - half_long):
            local = features[i - half_local:i + half_local]
            sparse_idx = list(range(i - half_long, i + half_long, sparse_r))
            long_range = features[sparse_idx]
            
            local_t = torch.FloatTensor(local).unsqueeze(0).to(device)
            long_t = torch.FloatTensor(long_range).unsqueeze(0).to(device)
            
            prob = torch.sigmoid(model(local_t, long_t)).cpu().numpy()[0]
            probs.append(prob)
    
    probs = np.array(probs)
    full = np.zeros((len(features), probs.shape[1]))
    full[half_long:-half_long] = probs
    full[:half_long] = probs[0]
    full[-half_long:] = probs[-1]
    return full

# ==================== OPTIMAL THRESHOLD CALCULATION ====================

def find_optimal_threshold(probs, labels):
    if labels.sum() < 10:
        return 0.27
    best_th, best_f1 = 0.27, 0
    for th in np.arange(0.1, 0.6, 0.05):
        pred = (probs > th).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        if tp + fp + fn > 0:
            f1 = 2 * tp / (2 * tp + fp + fn)
            if f1 > best_f1:
                best_f1, best_th = f1, th
    return best_th

# ==================== PREDICTION TO SUBMISSION ====================

def predict_multiclass_adaptive(pred, meta, action_thresholds, min_duration=3):
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

# ==================== MAIN TRAINING AND PREDICTION ====================

def process_with_dual_scale_cnn(body_parts_tracked_str, switch_tr, X_tr, label, meta):
    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    
    X_tr_filled = X_tr.fillna(0).values
    mean = X_tr_filled.mean(axis=0, keepdims=True)
    std = X_tr_filled.std(axis=0, keepdims=True) + 1e-8
    X_tr_normalized = (X_tr_filled - mean) / std
    
    action_thresholds = {}
    models_dict = {}
    
    for action in label.columns:
        action_mask = ~label[action].isna().values
        y_action = label[action][action_mask].values
        
        if not (y_action == 0).all() and y_action.sum() >= 10:
            pos_ratio = y_action.sum() / len(y_action) * 100
            if verbose:
                print(f"  Training {action}: {int(y_action.sum())} pos ({pos_ratio:.2f}%)")
            
            X_action = X_tr_normalized[action_mask]
            
            dataset = DualScaleDataset(X_action, y_action.reshape(-1, 1))
            
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_labels = y_action[dataset.valid_indices[:train_size]]
            sampler = create_balanced_sampler(train_labels, oversample_ratio=2.0)
            
            if sampler is not None:
                train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
            else:
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
            
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
            
            model = DualScaleConv1D(input_dim=X_tr_normalized.shape[1], num_classes=1).to(device)
            model = train_dual_model(model, train_loader, val_loader, epochs=15, lr=1e-3, patience=3)
            
            # Calculate optimal threshold
            val_probs, val_labels = [], []
            model.eval()
            with torch.no_grad():
                for local, long_range, labels in val_loader:
                    local, long_range = local.to(device), long_range.to(device)
                    outputs = model(local, long_range)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    val_probs.extend(probs[:, 0])
                    val_labels.extend(labels.numpy()[:, 0])
            
            threshold = find_optimal_threshold(np.array(val_probs), np.array(val_labels))
            if verbose:
                print(f"    Optimal threshold: {threshold:.3f}")
            
            action_thresholds[action] = threshold
            models_dict[action] = (model, mean, std)
            
            del train_loader, val_loader, dataset
            gc.collect()
            torch.cuda.empty_cache()
    
    # Generate test predictions
    test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
    generator = generate_mouse_data(test_subset, 'test',
                                    generate_single=(switch_tr == 'single'), 
                                    generate_pair=(switch_tr == 'pair'))
    
    submission_list = []
    
    for switch_te, data_te, meta_te, actions_te in generator:
        try:
            if switch_te == 'single':
                X_te = transform_single(data_te, body_parts_tracked)
            else:
                X_te = transform_pair(data_te, body_parts_tracked)
            
            X_te_filled = X_te.fillna(0).values
            pred = pd.DataFrame(index=meta_te.video_frame)
            
            for action in models_dict.keys():
                if action in actions_te:
                    model, mean, std = models_dict[action]
                    X_te_normalized = (X_te_filled - mean) / std
                    probs = predict_dual(model, X_te_normalized)
                    pred[action] = probs[:, 0]
            
            if pred.shape[1] > 0:
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

# Process first 3 configurations for speed (increase to process all)
for section in range(min(3, len(body_parts_tracked_list))):
    body_parts_tracked_str = body_parts_tracked_list[section]
    
    try:
        body_parts_tracked = json.loads(body_parts_tracked_str)
        print(f"\n{'='*70}")
        print(f"Section {section+1}: {len(body_parts_tracked)} body parts")
        print('='*70)
        
        if len(body_parts_tracked) > 5:
            body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
        
        train_subset = train[train.body_parts_tracked == body_parts_tracked_str]
        
        # Process single mouse behaviors
        single_list, single_label_list, single_meta_list = [], [], []
        
        for switch, data, meta, label in generate_mouse_data(train_subset, 'train'):
            if switch == 'single':
                single_list.append(data)
                single_label_list.append(label)
                single_meta_list.append(meta)
            if len(single_list) >= 10:  # Increase for better performance
                break
        
        if len(single_list) > 0:
            print(f"\nProcessing SINGLE mouse behaviors ({len(single_list)} sequences)")
            single_mouse = pd.concat(single_list)
            single_label = pd.concat(single_label_list)
            single_meta = pd.concat(single_meta_list)
            
            X_tr = transform_single(single_mouse, body_parts_tracked)
            print(f"Feature shape: {X_tr.shape}")
            
            sub_parts = process_with_dual_scale_cnn(body_parts_tracked_str, 'single', X_tr, single_label, single_meta)
            submission_list.extend(sub_parts)
            
            del single_mouse, single_label, single_meta, X_tr
            gc.collect()
        
        # Process pair behaviors
        pair_list, pair_label_list, pair_meta_list = [], [], []
        
        for switch, data, meta, label in generate_mouse_data(train_subset, 'train'):
            if switch == 'pair':
                pair_list.append(data)
                pair_label_list.append(label)
                pair_meta_list.append(meta)
            if len(pair_list) >= 10:  # Increase for better performance
                break
        
        if len(pair_list) > 0:
            print(f"\nProcessing PAIR behaviors ({len(pair_list)} sequences)")
            mouse_pair = pd.concat(pair_list)
            pair_label = pd.concat(pair_label_list)
            pair_meta = pd.concat(pair_meta_list)
            
            X_tr = transform_pair(mouse_pair, body_parts_tracked)
            print(f"Feature shape: {X_tr.shape}")
            
            sub_parts = process_with_dual_scale_cnn(body_parts_tracked_str, 'pair', X_tr, pair_label, pair_meta)
            submission_list.extend(sub_parts)
            
            del mouse_pair, pair_label, pair_meta, X_tr
            gc.collect()
        
    except Exception as e:
        print(f'***Exception*** {str(e)[:200]}')
    
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

print(f"\n{'='*70}")
print("SUBMISSION CREATED")
print('='*70)
print(f"Total predictions: {len(submission)}")
print(f"Unique actions: {submission.action.nunique()}")
print(f"Actions: {sorted(submission.action.unique())}")
print(f"Videos: {submission.video_id.nunique()}")
print(f"\nSubmission saved to: submission.csv")

