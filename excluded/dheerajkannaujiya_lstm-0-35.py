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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import polars as pl
from scipy import signal
import math
import joblib
import glob

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
TREE_MODELS_DIR = "/kaggle/input/trained-action-2/saved_models" 

# ======================== LSTM MODEL ========================

class LSTMDataset(Dataset):
    def __init__(self, X, y, seq_len=30):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.X) - self.seq_len + 1)

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.seq_len]
        y_val = self.y[idx + self.seq_len - 1]
        return x_seq, y_val  # y_val is (1,)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, 
                              dropout=dropout if num_layers > 1 else 0,
                              bidirectional=True)
    
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        
        out = self.fc(last_time_step_out)
        return out

def train_lstm(X_tr, y_tr, input_size, epochs=1, batch_size=512, lr=0.001, save_path= None):
    if len(X_tr) < 10:  # Too small
        def predict_dummy(X_te):
            return np.full(len(X_te), 0.5)
        return predict_dummy, 1

    seq_len_train = min(30, max(5, len(X_tr) // 4))  # Min 5 for small
    
    # FIX: Removed .values from y_tr because it's already a numpy array
    dataset = LSTMDataset(X_tr, y_tr.reshape(-1, 1), seq_len_train)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = LSTMClassifier(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # criterion = nn.BCELoss() not_perfect_define_values
    
    positives = y_tr.sum()
    negatives = len(y_tr) - positives
    pos_weight = torch.tensor([negatives / positives], device=device) if positives > 0 else torch.tensor([1.0], device=device)

    # BCEWithLogitsLoss zyada stable hota hai aur pos_weight handle karta hai
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for seq_x, seq_y in dataloader:
            seq_x = seq_x.to(device)
            seq_y = seq_y.to(device)  # (batch, 1)
            optimizer.zero_grad()
            out = model(seq_x)  # (batch, 1) - NO SQUEEZE!
            loss = criterion(out, seq_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Stable
            optimizer.step()
            total_loss += loss.item()
        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch}, loss: {total_loss / len(dataloader):.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        if verbose:
            print(f"âœ… Model and Scaler saved successfully at: {save_path}")

    # The rest of the function remains the same...
    def predict(X_te):
        if len(X_te) == 0:
            return np.array([])
        model.eval()
        with torch.no_grad():
            seq_len_test = min(seq_len_train, len(X_te))
            if seq_len_test < 1:
                seq_len_test = 1
            step = max(1, seq_len_test // 4)  # Overlap for better coverage
            indices = np.arange(0, len(X_te) - seq_len_test + 1, step)
            if len(indices) == 0:
                # Very small, use full as seq
                seq = X_te[:seq_len_test] if len(X_te) > 0 else X_te
                if len(seq) < seq_len_train:
                    pad_len = seq_len_train - len(seq)
                    pad = np.tile(seq[-1:] if len(seq)>0 else np.zeros((1, input_size)), (pad_len, 1))
                    seq = np.vstack([seq, pad])
                seq_t = torch.FloatTensor(seq).unsqueeze(0).to(device)
                prob = model(seq_t).cpu().numpy()[0, 0]
                return np.full(len(X_te), prob)

            batch_size_pred = 128
            all_probs = []
            for b_start in range(0, len(indices), batch_size_pred):
                b_end = min(b_start + batch_size_pred, len(indices))
                b_inds = indices[b_start:b_end]
                batch_seqs = []
                for j in b_inds:
                    seq_slice = X_te[j:j + seq_len_test]
                    if len(seq_slice) < seq_len_test:
                        pad_len = seq_len_test - len(seq_slice)
                        pad = np.tile(seq_slice[-1:], (pad_len, 1)) if len(seq_slice) > 0 else np.zeros((pad_len, input_size))
                        seq_slice = np.vstack([seq_slice, pad])
                    batch_seqs.append(seq_slice)
                if batch_seqs:
                    batch_t = torch.FloatTensor(np.stack(batch_seqs)).to(device)
                    b_out = model(batch_t).cpu().numpy()[:, 0]
                    all_probs.extend(b_out)

            # Assign probs to frame positions (last of each window)
            frame_probs = np.full(len(X_te), 0.5)
            for k, j in enumerate(indices):
                assign_frame = min(j + seq_len_test - 1, len(X_te) - 1)
                frame_probs[assign_frame] = all_probs[k]
            # Simple forward fill for gaps
            frame_probs = pd.Series(frame_probs).fillna(method='ffill').fillna(0.5).values
            return frame_probs

    return predict, seq_len_train

# ======================== SCORING FUNCTIONS (UNCHANGED) ========================

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

def mouse_fbeta(solution: pd.DataFrame, submission: pd.DataFrame, beta: float = 1) -> float:
    if len(solution) == 0 or len(submission) == 0:
        raise ValueError('Missing solution or submission data')

    expected_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']

    for col in expected_cols:
        if col not in solution.columns:
            raise ValueError(f'Solution is missing column {col}')
        if col not in submission.columns:
            raise ValueError(f'Submission is missing column {col}')

    solution: pl.DataFrame = pl.DataFrame(solution)
    submission: pl.DataFrame = pl.DataFrame(submission)
    assert (solution['start_frame'] <= solution['stop_frame']).all()
    assert (submission['start_frame'] <= submission['stop_frame']).all()
    solution_videos = set(solution['video_id'].unique())
    submission = submission.filter(pl.col('video_id').is_in(solution_videos))

    solution = solution.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('label_key'),
    )
    submission = submission.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('prediction_key'),
    )

    lab_scores = []
    for lab in solution['lab_id'].unique():
        lab_solution = solution.filter(pl.col('lab_id') == lab).clone()
        lab_videos = set(lab_solution['video_id'].unique())
        lab_submission = submission.filter(pl.col('video_id').is_in(lab_videos)).clone()
        lab_scores.append(single_lab_f1(lab_solution, lab_submission, beta=beta))

    return sum(lab_scores) / len(lab_scores)

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, beta: float = 1) -> float:
    solution = solution.drop(row_id_column_name, axis='columns', errors='ignore')
    submission = submission.drop(row_id_column_name, axis='columns', errors='ignore')
    return mouse_fbeta(solution, submission, beta=beta)

# ======================== DATA LOADING (MINOR FIXES) ========================

train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)
train_without_mabe22 = train.query("~ lab_id.str.startswith('MABe22_')")

test = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')
body_parts_tracked_list = list(np.unique(train.body_parts_tracked))

drop_body_parts = ['headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright',
                   'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright',
                   'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint']

def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    assert traintest in ['train', 'test']
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    for _, row in dataset.iterrows():
        lab_id = row.lab_id
        if lab_id.startswith('MABe22'): continue
        video_id = row.video_id

        if type(row.behaviors_labeled) != str:
            if verbose: print('No labeled behaviors:', lab_id, video_id)
            continue

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        if pvid.isna().any().any():
            if verbose and traintest == 'test': print('video with missing values', video_id, traintest, len(vid), 'frames')
            pvid = pvid.fillna(method='ffill').fillna(pvid.mean())  # Impute
        else:
            if verbose and traintest == 'test': print('video with all values', video_id, traintest, len(vid), 'frames')
        del vid
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
                    assert len(single_mouse) == len(pvid)
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
                        if verbose: print('- test single', video_id, mouse_id)
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
                    mouse_pair = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                    assert len(mouse_pair) == len(pvid)
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
                        if verbose: print('- test pair', video_id, agent, target)
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions


# ======================== ADVANCED FEATURE ENGINEERING FOR (TREES)========================
def safe_rolling(series, window, func, min_periods=None):
    """Safe rolling operation with NaN handling"""
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series.rolling(window, min_periods=min_periods, center=True).apply(func, raw=True)

def add_curvature_features(X, center_x, center_y):
    """Trajectory curvature"""
    vel_x = center_x.diff()
    vel_y = center_y.diff()
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()

    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)

    for window in [30, 60]:
        X[f'curv_mean_{window}'] = curvature.rolling(window, min_periods=5).mean()

    angle = np.arctan2(vel_y, vel_x)
    angle_change = np.abs(angle.diff())
    X['turn_rate_30'] = angle_change.rolling(30, min_periods=5).sum()

    return X

def add_multiscale_features(X, center_x, center_y):
    """Multi-scale temporal features"""
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2)

    scales = [10, 40, 160]
    for scale in scales:
        if len(speed) >= scale:
            X[f'sp_m{scale}'] = speed.rolling(scale, min_periods=max(1, scale//4)).mean()
            X[f'sp_s{scale}'] = speed.rolling(scale, min_periods=max(1, scale//4)).std()

    if len(scales) >= 2 and f'sp_m{scales[0]}' in X.columns and f'sp_m{scales[-1]}' in X.columns:
        X['sp_ratio'] = X[f'sp_m{scales[0]}'] / (X[f'sp_m{scales[-1]}'] + 1e-6)

    return X

def add_state_features(X, center_x, center_y):
    """Behavioral state transitions"""
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2)
    speed_ma = speed.rolling(15, min_periods=5).mean()

    try:
        speed_states = pd.cut(speed_ma, bins=[-np.inf, 0.5, 2.0, 5.0, np.inf], labels=[0, 1, 2, 3]).astype(float)
        
        for window in [60, 120]:
            if len(speed_states) >= window:
                for state in [0, 1, 2, 3]:
                    X[f's{state}_{window}'] = (speed_states == state).astype(float).rolling(window, min_periods=10).mean()
                
                state_changes = (speed_states != speed_states.shift(1)).astype(float)
                X[f'trans_{window}'] = state_changes.rolling(window, min_periods=10).sum()
    except:
        pass

    return X

def add_longrange_features(X, center_x, center_y):
    """Long-range temporal features"""
    for window in [120, 240]:
        if len(center_x) >= window:
            X[f'x_ml{window}'] = center_x.rolling(window, min_periods=20).mean()
            X[f'y_ml{window}'] = center_y.rolling(window, min_periods=20).mean()

    for span in [60, 120]:
        X[f'x_e{span}'] = center_x.ewm(span=span, min_periods=1).mean()
        X[f'y_e{span}'] = center_y.ewm(span=span, min_periods=1).mean()

    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2)
    for window in [60, 120]:
        if len(speed) >= window:
            X[f'sp_pct{window}'] = speed.rolling(window, min_periods=20).rank(pct=True)

    return X

def add_interaction_features(X, mouse_pair, avail_A, avail_B):
    """Social interaction features"""
    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return X

    rel_x = mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x']
    rel_y = mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y']
    rel_dist = np.sqrt(rel_x**2 + rel_y**2)

    A_vx = mouse_pair['A']['body_center']['x'].diff()
    A_vy = mouse_pair['A']['body_center']['y'].diff()
    B_vx = mouse_pair['B']['body_center']['x'].diff()
    B_vy = mouse_pair['B']['body_center']['y'].diff()

    A_lead = (A_vx * rel_x + A_vy * rel_y) / (np.sqrt(A_vx**2 + A_vy**2) * rel_dist + 1e-6)
    B_lead = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (np.sqrt(B_vx**2 + B_vy**2) * rel_dist + 1e-6)

    for window in [30, 60]:
        X[f'A_ld{window}'] = A_lead.rolling(window, min_periods=5).mean()
        X[f'B_ld{window}'] = B_lead.rolling(window, min_periods=5).mean()

    approach = -rel_dist.diff()
    chase = approach * B_lead
    X['chase_30'] = chase.rolling(30, min_periods=5).mean()

    for window in [60, 120]:
        A_sp = np.sqrt(A_vx**2 + A_vy**2)
        B_sp = np.sqrt(B_vx**2 + B_vy**2)
        X[f'sp_cor{window}'] = A_sp.rolling(window, min_periods=10).corr(B_sp)

    return X

def transform_single(single_mouse, body_parts_tracked):
    """Enhanced single mouse transform"""
    available_body_parts = single_mouse.columns.get_level_values(0)

    # Base distance features
    X = pd.DataFrame({
        f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.combinations(body_parts_tracked, 2) 
        if p1 in available_body_parts and p2 in available_body_parts
    })
    X = X.reindex(columns=[f"{p1}+{p2}" for p1, p2 in itertools.combinations(body_parts_tracked, 2)], copy=False)

    # Speed features
    if all(p in single_mouse.columns for p in ['ear_left', 'ear_right', 'tail_base']):
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(10)
        speeds = pd.DataFrame({
            'sp_lf': np.square(single_mouse['ear_left'] - shifted['ear_left']).sum(axis=1, skipna=False),
            'sp_rt': np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False),
            'sp_lf2': np.square(single_mouse['ear_left'] - shifted['tail_base']).sum(axis=1, skipna=False),
            'sp_rt2': np.square(single_mouse['ear_right'] - shifted['tail_base']).sum(axis=1, skipna=False),
        })
        X = pd.concat([X, speeds], axis=1)

    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)

    # Body angle
    if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
        v1 = single_mouse['nose'] - single_mouse['body_center']
        v2 = single_mouse['tail_base'] - single_mouse['body_center']
        X['body_ang'] = (v1['x'] * v2['x'] + v1['y'] * v2['y']) / (
            np.sqrt(v1['x']**2 + v1['y']**2) * np.sqrt(v2['x']**2 + v2['y']**2) + 1e-6)

    # Core temporal features
    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']
        cy = single_mouse['body_center']['y']
        
        for w in [5, 15, 30, 60]:
            X[f'cx_m{w}'] = cx.rolling(w, min_periods=1, center=True).mean()
            X[f'cy_m{w}'] = cy.rolling(w, min_periods=1, center=True).mean()
            X[f'cx_s{w}'] = cx.rolling(w, min_periods=1, center=True).std()
            X[f'cy_s{w}'] = cy.rolling(w, min_periods=1, center=True).std()
            X[f'x_rng{w}'] = cx.rolling(w, min_periods=1, center=True).max() - cx.rolling(w, min_periods=1, center=True).min()
            X[f'y_rng{w}'] = cy.rolling(w, min_periods=1, center=True).max() - cy.rolling(w, min_periods=1, center=True).min()
            X[f'disp{w}'] = np.sqrt(cx.diff().rolling(w, min_periods=1).sum()**2 + cy.diff().rolling(w, min_periods=1).sum()**2)
            X[f'act{w}'] = np.sqrt(cx.diff().rolling(w, min_periods=1).var() + cy.diff().rolling(w, min_periods=1).var())
        
        # Advanced features
        X = add_curvature_features(X, cx, cy)
        X = add_multiscale_features(X, cx, cy)
        X = add_state_features(X, cx, cy)
        X = add_longrange_features(X, cx, cy)

    # Nose-tail features
    if all(p in available_body_parts for p in ['nose', 'tail_base']):
        nt_dist = np.sqrt((single_mouse['nose']['x'] - single_mouse['tail_base']['x'])**2 + 
                         (single_mouse['nose']['y'] - single_mouse['tail_base']['y'])**2)
        for lag in [10, 20, 40]:
            X[f'nt_lg{lag}'] = nt_dist.shift(lag)
            X[f'nt_df{lag}'] = nt_dist - nt_dist.shift(lag)

    # Ear features
    if all(p in available_body_parts for p in ['ear_left', 'ear_right']):
        ear_d = np.sqrt((single_mouse['ear_left']['x'] - single_mouse['ear_right']['x'])**2 + 
                       (single_mouse['ear_left']['y'] - single_mouse['ear_right']['y'])**2)
        for off in [-20, -10, 10, 20]:
            X[f'ear_o{off}'] = ear_d.shift(-off)
        X['ear_con'] = ear_d.rolling(30, min_periods=1, center=True).std() / (ear_d.rolling(30, min_periods=1, center=True).mean() + 1e-6)

    return X

def transform_pair(mouse_pair, body_parts_tracked):
    """Enhanced pair transform"""
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)

    # Inter-mouse distances
    X = pd.DataFrame({
        f"12+{p1}+{p2}": np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.product(body_parts_tracked, repeat=2) 
        if p1 in avail_A and p2 in avail_B
    })
    X = X.reindex(columns=[f"12+{p1}+{p2}" for p1, p2 in itertools.product(body_parts_tracked, repeat=2)], copy=False)

    # Speed features
    if ('A', 'ear_left') in mouse_pair.columns and ('B', 'ear_left') in mouse_pair.columns:
        shA = mouse_pair['A']['ear_left'].shift(10)
        shB = mouse_pair['B']['ear_left'].shift(10)
        speeds = pd.DataFrame({
            'sp_A': np.square(mouse_pair['A']['ear_left'] - shA).sum(axis=1, skipna=False),
            'sp_AB': np.square(mouse_pair['A']['ear_left'] - shB).sum(axis=1, skipna=False),
            'sp_B': np.square(mouse_pair['B']['ear_left'] - shB).sum(axis=1, skipna=False),
        })
        X = pd.concat([X, speeds], axis=1)

    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)

    # Relative orientation
    if all(p in avail_A for p in ['nose', 'tail_base']) and all(p in avail_B for p in ['nose', 'tail_base']):
        dir_A = mouse_pair['A']['nose'] - mouse_pair['A']['tail_base']
        dir_B = mouse_pair['B']['nose'] - mouse_pair['B']['tail_base']
        X['rel_ori'] = (dir_A['x'] * dir_B['x'] + dir_A['y'] * dir_B['y']) / (
            np.sqrt(dir_A['x']**2 + dir_A['y']**2) * np.sqrt(dir_B['x']**2 + dir_B['y']**2) + 1e-6)

    # Approach rate
    if all(p in avail_A for p in ['nose']) and all(p in avail_B for p in ['nose']):
        cur = np.square(mouse_pair['A']['nose'] - mouse_pair['B']['nose']).sum(axis=1, skipna=False)
        shA_n = mouse_pair['A']['nose'].shift(10)
        shB_n = mouse_pair['B']['nose'].shift(10)
        past = np.square(shA_n - shB_n).sum(axis=1, skipna=False)
        X['appr'] = cur - past

    # Distance bins
    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd = np.sqrt((mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x'])**2 +
                    (mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y'])**2)
        X['v_cls'] = (cd < 5.0).astype(float)
        X['cls'] = ((cd >= 5.0) & (cd < 15.0)).astype(float)
        X['med'] = ((cd >= 15.0) & (cd < 30.0)).astype(float)
        X['far'] = (cd >= 30.0).astype(float)

    # Temporal interaction
    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd_full = np.square(mouse_pair['A']['body_center'] - mouse_pair['B']['body_center']).sum(axis=1, skipna=False)
        
        for w in [5, 15, 30, 60]:
            X[f'd_m{w}'] = cd_full.rolling(w, min_periods=1, center=True).mean()
            X[f'd_s{w}'] = cd_full.rolling(w, min_periods=1, center=True).std()
            X[f'd_mn{w}'] = cd_full.rolling(w, min_periods=1, center=True).min()
            X[f'd_mx{w}'] = cd_full.rolling(w, min_periods=1, center=True).max()
            
            d_var = cd_full.rolling(w, min_periods=1, center=True).var()
            X[f'int{w}'] = 1 / (1 + d_var)
            
            Axd = mouse_pair['A']['body_center']['x'].diff()
            Ayd = mouse_pair['A']['body_center']['y'].diff()
            Bxd = mouse_pair['B']['body_center']['x'].diff()
            Byd = mouse_pair['B']['body_center']['y'].diff()
            coord = Axd * Bxd + Ayd * Byd
            X[f'co_m{w}'] = coord.rolling(w, min_periods=1, center=True).mean()
            X[f'co_s{w}'] = coord.rolling(w, min_periods=1, center=True).std()

    # Nose-nose
    if 'nose' in avail_A and 'nose' in avail_B:
        nn = np.sqrt((mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x'])**2 +
                    (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y'])**2)
        for lag in [10, 20, 40]:
            X[f'nn_lg{lag}'] = nn.shift(lag)
            X[f'nn_ch{lag}'] = nn - nn.shift(lag)
            is_cl = (nn < 10.0).astype(float)
            X[f'cl_ps{lag}'] = is_cl.rolling(lag, min_periods=1).mean()

    # Velocity alignment
    if 'body_center' in avail_A and 'body_center' in avail_B:
        Avx = mouse_pair['A']['body_center']['x'].diff()
        Avy = mouse_pair['A']['body_center']['y'].diff()
        Bvx = mouse_pair['B']['body_center']['x'].diff()
        Bvy = mouse_pair['B']['body_center']['y'].diff()
        val = (Avx * Bvx + Avy * Bvy) / (np.sqrt(Avx**2 + Avy**2) * np.sqrt(Bvx**2 + Bvy**2) + 1e-6)
        
        for off in [-20, -10, 0, 10, 20]:
            X[f'va_{off}'] = val.shift(-off)
        
        X['int_con'] = cd_full.rolling(30, min_periods=1, center=True).std() / (cd_full.rolling(30, min_periods=1, center=True).mean() + 1e-6)
        
        # Advanced interaction
        X = add_interaction_features(X, mouse_pair, avail_A, avail_B)

    return X


# ======================== FEATURE ENGINEERING (IMPROVED) FOR (LSTM)========================
def get_or_create_body_center(mouse_data):
    """
    Yeh function check karta hai ki 'body_center' hai ya nahi.
    Agar hai, to use return karta hai.
    Agar nahi hai, to sabhi available body parts ka average lekar ek naya 'body_center' banata hai.
    """
    available_body_parts = mouse_data.columns.get_level_values(0).unique()
    
    if 'body_center' in available_body_parts:
        # Agar body_center pehle se hai, to use hi use karo
        return mouse_data['body_center']
    else:
        # Agar nahi hai, to ek naya banao
        # Saare x aur y coordinates ko alag karo
        x_coords = mouse_data.loc[:, (slice(None), 'x')]
        y_coords = mouse_data.loc[:, (slice(None), 'y')]
        
        # Unka average le lo
        center_x = x_coords.mean(axis=1)
        center_y = y_coords.mean(axis=1)
        
        # Ek naya DataFrame banakar return karo
        return pd.DataFrame({'x': center_x, 'y': center_y})

def get_orientation(mouse_data):
    if 'nose' in mouse_data and 'tail_base' in mouse_data:
        dx = mouse_data['nose']['x'] - mouse_data['tail_base']['x']
        dy = mouse_data['nose']['y'] - mouse_data['tail_base']['y']
        return np.arctan2(dy, dx)
    return pd.Series(0, index=mouse_data.index) # Agar data nahi hai
                        
# ======================== FEATURE ENGINEERING (SIMPLIFIED FOR LSTM) ========================
def transform_single_v2(single_mouse, row):
    X = pd.DataFrame(index=single_mouse.index)
    body_center = get_or_create_body_center(single_mouse)
    available_parts = single_mouse.columns.get_level_values(0).unique()

    # === Basic Kinematics (Aapke paas pehle se hain) ===
    X['speed'] = np.sqrt(body_center['x'].diff()**2 + body_center['y'].diff()**2).fillna(0)
    X['accel'] = X['speed'].diff().fillna(0)
    X['jerk'] = X['accel'].diff().fillna(0) # Jerk for sudden movements

    # === Posture & Position Features ===
    # Feature #1: Rearing (à¤–à¤¡à¤¼à¤¾ à¤¹à¥‹à¤¨à¤¾)
    X['y_pos'] = body_center['y'] # Y-coordinate
    X['y_vel'] = body_center['y'].diff().fillna(0) # Vertical velocity

    # Feature #6: Body Orientation
    if 'nose' in available_parts and 'tail_base' in available_parts:
        dx = single_mouse['nose']['x'] - single_mouse['tail_base']['x']
        dy = single_mouse['nose']['y'] - single_mouse['tail_base']['y']
        X['orientation'] = np.arctan2(dy, dx)
        X['orientation_change'] = X['orientation'].diff().fillna(0)
        X['body_length'] = np.sqrt(dx**2 + dy**2) # For elongation

    # Feature #4: Grooming / Scratching
    if 'nose' in available_parts and 'forepaw_left' in available_parts and 'forepaw_right' in available_parts:
        # Snout-to-forepaws distance
        dist_left = np.sqrt((single_mouse['nose']['x'] - single_mouse['forepaw_left']['x'])**2 + (single_mouse['nose']['y'] - single_mouse['forepaw_left']['y'])**2)
        dist_right = np.sqrt((single_mouse['nose']['x'] - single_mouse['forepaw_right']['x'])**2 + (single_mouse['nose']['y'] - single_mouse['forepaw_right']['y'])**2)
        X['snout_to_paw_dist'] = np.minimum(dist_left, dist_right)
        
        # Speed of forepaws for rhythmic movement
        X['paw_speed_left'] = np.sqrt(single_mouse['forepaw_left']['x'].diff()**2 + single_mouse['forepaw_left']['y'].diff()**2).fillna(0)

    arena_width = 100  
    arena_height = 100
    
    if 'arena_width_cm' in row:
       arena_width = row['arena_width_cm']
    if 'arena_height_cm' in row:
       arena_height = row['arena_height_cm']
    
    # Ab in safe variables ka istemaal karo
    dist_x = np.minimum(body_center['x'], arena_width - body_center['x'])
    dist_y = np.minimum(body_center['y'], arena_height - body_center['y'])
    X['dist_from_wall'] = np.minimum(dist_x, dist_y)
    X['dist_from_wall'] = np.minimum(dist_x, dist_y)

    return X

from scipy.spatial.distance import cdist
def transform_pair_v2(mouse_pair):
    X = pd.DataFrame(index=mouse_pair.index)
    body_center_A = get_or_create_body_center(mouse_pair['A'])
    body_center_B = get_or_create_body_center(mouse_pair['B'])
    available_parts_A = mouse_pair['A'].columns.get_level_values(0).unique()
    available_parts_B = mouse_pair['B'].columns.get_level_values(0).unique()

    # === Basic Relative Kinematics (Aapke paas pehle se hain) ===
    rel_x = body_center_A['x'] - body_center_B['x']
    rel_y = body_center_A['y'] - body_center_B['y']
    X['center_dist'] = np.sqrt(rel_x**2 + rel_y**2)

    # NEW: Velocity (Speed)
    X['rel_vel_x'] = rel_x.diff().fillna(0)
    X['rel_vel_y'] = rel_y.diff().fillna(0)
    X['dist_vel'] = X['center_dist'].diff().fillna(0) # Are they getting closer or farther?

    # NEW: Angle (Magic feature for chasing/sniffing)
    X['rel_angle'] = np.arctan2(rel_y, rel_x)
    X['rel_angle_vel'] = X['rel_angle'].diff().fillna(0)

    # NEW: Individual Speeds
    X['A_speed'] = np.sqrt((body_center_A.diff()**2).sum(axis=1)).fillna(0)
    X['B_speed'] = np.sqrt((body_center_B.diff()**2).sum(axis=1)).fillna(0)

    # transform_pair_v2 mein:
    X['A_orientation'] = get_orientation(mouse_pair['A'])
    X['B_orientation'] = get_orientation(mouse_pair['B'])
    
    # Feature for Approach / Disengaged
    X['dist_change_rate'] = X['center_dist'].diff().fillna(0)

    speed_A = np.sqrt(body_center_A['x'].diff()**2 + body_center_A['y'].diff()**2).fillna(0)
    speed_B = np.sqrt(body_center_B['x'].diff()**2 + body_center_B['y'].diff()**2).fillna(0)
    X['speed_diff'] = speed_A - speed_B
    X['speed_ratio'] = speed_A / (speed_B + 1e-6)

    # === Behavior-Specific Features ===
    # Feature for Attack / Push
    X['rel_accel'] = X['speed_diff'].diff().fillna(0) # Relative acceleration

    # Feature for Sniffing (Face & Anogenital)
    if 'nose' in available_parts_A and 'nose' in available_parts_B:
        X['nose_dist'] = np.sqrt((mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x'])**2 + (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y'])**2)
    
    if 'nose' in available_parts_A and 'tail_base' in available_parts_B:
        X['chase_dist'] = np.sqrt((mouse_pair['A']['nose']['x'] - mouse_pair['B']['tail_base']['x'])**2 + (mouse_pair['A']['nose']['y'] - mouse_pair['B']['tail_base']['y'])**2)

    # Feature for Mounting
    X['y_pos_diff'] = body_center_A['y'] - body_center_B['y'] # Vertical difference

    if 'nose' in mouse_pair['A'].columns.get_level_values(0) and 'nose' in mouse_pair['B'].columns.get_level_values(0):
       nose_dist_x = mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x']
       nose_dist_y = mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y']
       X['nose_to_nose_dist'] = np.sqrt(nose_dist_x**2 + nose_dist_y**2)

    # Feature for Allogroom
    if 'nose' in available_parts_A:
      X['allogroom_dist'] = np.sqrt(
        (mouse_pair['A']['nose']['x'] - body_center_B['x'])**2 + 
        (mouse_pair['A']['nose']['y'] - body_center_B['y'])**2
    )
     
    return X
    
action_thresholds = defaultdict(lambda: 0.30)

# ======================== NEW, IMPROVED PREDICTION-TO-EVENT FUNCTION ========================
def events_from_probabilities(probabilities, meta, action_thresholds, min_duration=5, merge_gap=3):
    """
    Ek behtar tareeka probabilities se action events nikalne ka.
    Yeh noise ko handle karta hai aur har action ko alag se process karta hai.
    """
    submission_parts = []
    
    # Har action column ke liye alag se process karo
    for action in probabilities.columns:
        # 1. Threshold lagao
        threshold = action_thresholds.get(action, 0.5)
        preds = (probabilities[action].values > threshold).astype(int)
        
        if preds.sum() == 0: # Agar koi bhi frame threshold se upar nahi hai, to skip
            continue
            
        # 2. Chote gaps (jaise 0,0,1,1,0,1,1,0,0) ko fill karo
        # Isse toote hue events jud jaate hain
        change_points = np.diff(preds, prepend=0, append=0)
        start_indices = np.where(change_points == 1)[0]
        stop_indices = np.where(change_points == -1)[0]
        
        for i in range(len(start_indices) - 1):
            gap = start_indices[i+1] - stop_indices[i]
            if gap <= merge_gap:
                preds[stop_indices[i] : start_indices[i+1]] = 1 # Gap ko 1 se bhar do
        
        # 3. Final events (start aur stop frames) nikalo
        change_points = np.diff(preds, prepend=0, append=0)
        start_indices = np.where(change_points == 1)[0]
        stop_indices = np.where(change_points == -1)[0]
        
        # 4. Chote events (jo min_duration se kam hain) ko filter karo
        for start_idx, stop_idx in zip(start_indices, stop_indices):
            duration = stop_idx - start_idx
            if duration >= min_duration:
                # Meta-data se video_id, agent_id, etc. lo
                # Humein actual frame numbers chahiye, index nahi
                start_frame = meta['video_frame'].iloc[start_idx]
                # stop_idx exclusive hota hai, isliye -1
                stop_frame = meta['video_frame'].iloc[stop_idx - 1] + 1 
                
                submission_parts.append({
                    'video_id': meta['video_id'].iloc[0],
                    'agent_id': meta['agent_id'].iloc[0],
                    'target_id': meta['target_id'].iloc[0],
                    'action': action,
                    'start_frame': start_frame,
                    'stop_frame': stop_frame
                })

    if not submission_parts:
        return pd.DataFrame()
    else:
        return pd.DataFrame(submission_parts)


# --- STEP 2: ADD THIS FUNCTION BEFORE 'submit_resnet_pipeline' ---
def predict_with_saved_trees(X_test, config_id, switch, action):
    """
    Saved Tree models (0-4) ko load karta hai aur Average Prediction deta hai.
    Features ko auto-align karta hai taaki error na aaye.
    """
    all_preds = []
    models_found = 0
    
    # Hum 5 folds (model0 se model4) check karenge
    for i in range(5):
        # Filename pattern tumhare bataye anusar:
        filename = f"model_config{config_id}_{switch}_{action}_model{i}.joblib"
        path = os.path.join(TREE_MODELS_DIR, filename)
        
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                
                # --- Feature Alignment Logic (Crash rokne ke liye) ---
                # Tree models ko wahi columns chahiye hote hain jinpar wo train huye.
                # Hum check karte hain agar model me feature names saved hain.
                try:
                    if hasattr(model, 'feature_name_'):
                        required_feats = model.feature_name_
                    elif hasattr(model, 'get_booster'): # XGBoost
                        required_feats = model.get_booster().feature_names
                    else:
                        required_feats = None

                    X_aligned = X_test.copy()
                    if required_feats:
                        # Missing columns ko 0 se bhar do
                        for col in required_feats:
                            if col not in X_aligned.columns:
                                X_aligned[col] = 0.0
                        # Extra columns hata do aur order same karo
                        X_aligned = X_aligned[required_feats]
                except:
                    X_aligned = X_test # Agar pata na chale to direct try karo
                
                # Prediction lo
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_aligned)[:, 1]
                else:
                    probs = model.predict(X_aligned)
                    
                all_preds.append(probs)
                models_found += 1
            except Exception as e:
                pass # Agar koi model corrupt hai to skip karo

    if models_found == 0:
        return None
    
    # Saare tree models ka average
    return np.mean(all_preds, axis=0)

# ======================== LSTM TRAINING & PREDICTION ========================

MODELS_DIR = "saved_models"

def submit_lstm(config_id,body_parts_tracked_str, switch_tr, X_tr, label, meta):
    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

    if X_tr.shape[1] == 0:
        print("No features, skip")
        return

    assert len(X_tr) == len(label), f"FATAL ERROR: Mismatch! X_tr length {len(X_tr)} != label length {len(label)}"
        
    input_size = X_tr.shape[1]
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)

    model_list = []
    for action in label.columns:
        action_mask = ~label[action].isna()
        if action_mask.sum() >= 5 and label[action][action_mask].sum() > 0:  # Has positives
            y_action = label[action][action_mask].values.astype(np.float32)
            X_action = X_tr_scaled[action_mask.values]
            model_filename = f"model_config_{config_id}_{switch_tr}_{action}.pth"
            model_path = os.path.join(MODELS_DIR, model_filename)
            predict_fn, _ = train_lstm(X_action, y_action, input_size, save_path=model_path)#,
            model_list.append((action, predict_fn, scaler))
            
            scaler_path = f'scaler_config_{config_id}_{switch_tr}_{action}.joblib'
            joblib.dump(scaler, scaler_path)

    del X_tr, label, X_tr_scaled
    gc.collect()

    test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
    generator = generate_mouse_data(test_subset, 'test', generate_single=(switch_tr == 'single'), generate_pair=(switch_tr == 'pair'))

    if verbose: print(f"n_videos: {len(test_subset)}, models: {len(model_list)}")

    for switch_te, data_te, meta_te, actions_te in generator:
        if switch_te != switch_tr: continue
        try:
            
            # 1. Generate Features for LSTM (using v2)
            if switch_te == 'single':
                X_te_lstm = transform_single_v2(data_te, body_parts_tracked)
                X_te_tree = transform_single(data_te, body_parts_tracked)
                
            else:
                X_te_lstm = transform_pair_v2(data_te)
                X_te_tree = transform_pair(data_te, body_parts_tracked)
            
            del data_te
            gc.collect()

            pred_df = pd.DataFrame(index=meta_te['video_frame'].values)  # Use values to avoid index issues
            for action, predict_fn, scaler_te in model_list:
                if action in actions_te:
                    X_te_scaled = scaler_te.transform(X_te_lstm)
                    probs_lstm = predict_fn(X_te_scaled)
                    probs_tree = predict_with_saved_trees(X_te_tree, config_id, switch_te, action)
                    if probs_tree is not None:
                        min_len = min(len(probs_lstm), len(probs_tree))
                        final_probs = (0.3 * probs_lstm[:min_len]) + (0.7 * probs_tree[:min_len])
                        if len(final_probs) < len(probs_lstm):
                            pad = np.zeros(len(probs_lstm) - len(final_probs))
                            final_probs = np.concatenate([final_probs, pad])
                    else:
                        final_probs = probs_lstm 
                        
                    pred_df[action] = final_probs
            
            if not pred_df.empty and pred_df.shape[1] > 0:
                 pred_df_smooth = pred_df.rolling(window=5, min_periods=1, center=True).mean()
                 sub_part = events_from_probabilities(pred_df_smooth, meta_te, action_thresholds, min_duration=5, merge_gap=3)
            
                 submission_list.append(sub_part)
            else:
               if verbose: 
                   print(f"  No models/actions for {switch_te}")
                    
        except Exception as e:
            if verbose: print(f'  ERROR in {switch_te}: {str(e)[:100]}')
            gc.collect()
            
# ======================== ROBUSTIFY (UNCHANGED) ========================

def robustify(submission, dataset, traintest, traintest_directory=None):
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    submission = submission[submission['start_frame'] < submission['stop_frame']]

    # No overlaps
    group_list = []
    for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values('start_frame')
        mask = np.ones(len(group), dtype=bool)
        last_stop = 0
        for i, (_, row) in enumerate(group.iterrows()):
            if row['start_frame'] < last_stop:
                mask[i] = False
            else:
                last_stop = row['stop_frame']
        group_list.append(group[mask])
    submission = pd.concat(group_list) if group_list else pd.DataFrame()

    # Fill empty
    s_list = []
    for idx, row in dataset.iterrows():
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'):
            continue
        video_id = row['video_id']
        if submission['video_id'].eq(video_id).any():
            continue

        if verbose: print(f"Video {video_id} has no predictions")
        
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)

        vid_behaviors = json.loads(row['behaviors_labeled'])
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])

        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1

        for (agent, target), actions in vid_behaviors.groupby(['agent', 'target']):
            if len(actions) == 0: continue
            batch_len = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, (_, action_row) in enumerate(actions.iterrows()):
                batch_start = start_frame + i * batch_len
                batch_stop = min(batch_start + batch_len, stop_frame)
                s_list.append((video_id, agent, target, action_row['action'], batch_start, batch_stop))

    if s_list:
        dummy_df = pd.DataFrame(s_list, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
        submission = pd.concat([submission, dummy_df], ignore_index=True)

    submission = submission.reset_index(drop=True)
    return submission

# ======================== MAIN LOOP ========================

submission_list = []

for section in range(1, len(body_parts_tracked_list)):
    body_parts_tracked_str = body_parts_tracked_list[section]
    try:
        print(f"{section}. Processing: {body_parts_tracked_str}")
        body_parts_tracked = json.loads(body_parts_tracked_str)
        if len(body_parts_tracked) > 5:
            body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

        train_subset = train[train.body_parts_tracked == body_parts_tracked_str]
        all_data_for_config = list(generate_mouse_data(train_subset, 'train'))

        # <<<<<<<<< STEP 2: Data ko type ke hisaab se alag-alag karo >>>>>>>>>
        single_data_list = [(d, m, l) for sw, d, m, l in all_data_for_config if sw == 'single' and not d.empty and not l.empty]
        pair_data_list = [(d, m, l) for sw, d, m, l in all_data_for_config if sw == 'pair' and not d.empty and not l.empty]

        if single_data_list:
            single_list = [item[0] for item in single_data_list]
            single_meta_list = [item[1] for item in single_data_list]
            single_label_list = [item[2] for item in single_data_list]

            single_mouse = pd.concat(single_list)
            single_meta = pd.concat(single_meta_list)
            single_label = pd.concat(single_label_list)

            # Ab reset_index karo
            single_mouse.reset_index(drop=True, inplace=True)
            single_meta.reset_index(drop=True, inplace=True)
            single_label.reset_index(drop=True, inplace=True)
            
            del single_list, single_label_list, single_meta_list
            gc.collect()
            
            X_tr_single = transform_single_v2(single_mouse, body_parts_tracked)
            
            del single_mouse
            print(f"  Single: {X_tr_single.shape}")
            submit_lstm(section,body_parts_tracked_str, 'single', X_tr_single, single_label, single_meta)
                
        if pair_data_list:
           pair_list = [item[0] for item in pair_data_list]
           pair_meta_list = [item[1] for item in pair_data_list]
           pair_label_list = [item[2] for item in pair_data_list]

           # Concat aab bilkul aaram se kaam karega
           mouse_pair = pd.concat(pair_list)
           pair_meta = pd.concat(pair_meta_list)
           pair_label = pd.concat(pair_label_list)

           # Reset index for alignment
           mouse_pair.reset_index(drop=True, inplace=True)
           pair_meta.reset_index(drop=True, inplace=True)
           pair_label.reset_index(drop=True, inplace=True)

           print(f"  Final check after concat & reset: Mouse Pair length = {len(mouse_pair)}, Pair Label length = {len(pair_label)}")
           X_tr_pair = transform_pair_v2(mouse_pair)
           del mouse_pair
           if X_tr_pair.shape[1] == 0:
               print(f"  Skipping submission for pair data as no features were generated.")
           else:
              print(f"  Pair: {X_tr_pair.shape}")
              submit_lstm(section, body_parts_tracked_str, 'pair', X_tr_pair, pair_label, pair_meta)
               
        else:
              print("  Skipping pair processing: All generated pair dataframes were empty.")
                
    except Exception as e:
        
        print(f'***Exception*** {str(e)[:100]}')

    gc.collect()
    print()

# ======================== FINAL SUBMISSION ========================
if submission_list:
    
    submission_list_non_empty = [df for df in submission_list if not df.empty]
    
    if submission_list_non_empty:
        submission = pd.concat(submission_list_non_empty, ignore_index=True)
    else:
        submission = pd.DataFrame()
else:
    submission = pd.DataFrame()

submission_robust = robustify(submission, test, 'test')

if submission_robust.empty:
    print("WARNING: Submission is still empty after robustify. Creating a minimal default submission.")
    submission_robust = pd.DataFrame({
        'video_id': [438887472], 'agent_id': ['mouse1'], 'target_id': ['self'],
        'action': ['rear'], 'start_frame': [278], 'stop_frame': [500]
    })

submission_robust.index.name = 'row_id'
submission_robust.to_csv('submission.csv')
print(f"\nSubmission created: {len(submission_robust)} predictions")

