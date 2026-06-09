import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import ast
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(invalid='ignore')
try:
    import torch
    from torch import nn, optim
    from torch.utils.data import Dataset, DataLoader
    from torch.cuda.amp import autocast, GradScaler
except ImportError as e:
    print(f"Error importing PyTorch: {e}")
    raise
import os
import psutil
import gc
from joblib import Parallel, delayed
# =============================================================================
# SHARED UTILITIES: Always load metadata and define functions
# =============================================================================
base_path = '/kaggle/input/MABe-mouse-behavior-detection/'
# Load metadata
train_meta = pd.read_csv(f'{base_path}train.csv')
train_meta['video_id'] = train_meta['video_id'].astype(str)
train_meta['body_parts_tracked'] = train_meta['body_parts_tracked'].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) else []
)
train_meta['behaviors_labeled'] = train_meta['behaviors_labeled'].apply(
    lambda x: [b.strip().strip("'\"") for b in ast.literal_eval(x)] if pd.notna(x) else []
)
test_meta = pd.read_csv(f'{base_path}test.csv')
test_meta['video_id'] = test_meta['video_id'].astype(str)
test_meta['body_parts_tracked'] = test_meta['body_parts_tracked'].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) else []
)
test_meta['behaviors_labeled'] = test_meta['behaviors_labeled'].apply(
    lambda x: [b.strip().strip("'\"") for b in ast.literal_eval(x)] if pd.notna(x) else []
)
def parse_behavior_label(b_label):
    """Parse 'mouse1,mouse2,approach' to (1,2,'approach')."""
    parts = [p.strip().strip("'\"") for p in b_label.split(',')]
    if len(parts) >= 3:
        agent_str = parts[0].replace('mouse', '').replace('self', '1')
        agent = int(agent_str)
        target_str = parts[1].replace('mouse', '').replace('self', str(agent))
        target = int(target_str)
        action = parts[2].strip()
        return agent, target, action
    return None
def load_video_features(lab, vid, is_train=True, N=10):
    """
    Load tracking; add features. For train: incl. labels; for test: no labels.
    """
    meta = train_meta if is_train else test_meta
    row = meta[(meta['lab_id'] == lab) & (meta['video_id'] == vid)].iloc[0]
    fps = row['frames_per_second']
    pix_per_cm = row['pix_per_cm_approx']
    total_frames = int(fps * row['video_duration_sec'])
    # Load tracking
    track_dir = 'train_tracking' if is_train else 'test_tracking'
    track_path = f'{base_path}{track_dir}/{lab}/{vid}.parquet'
    track = pd.read_parquet(track_path)
    track['mouse_id'] = track['mouse_id'].astype(int)
    track = track[track['video_frame'] % N == 0].copy()
    track['x_cm'] = track['x'] / pix_per_cm
    track['y_cm'] = track['y'] / pix_per_cm
    anns = None
    if is_train:
        ann_path = f'{base_path}train_annotation/{lab}/{vid}.parquet'
        anns = pd.read_parquet(ann_path)
        def expand_to_frames(anns, total_frames):
            frame_labels = np.full(total_frames, 'background')
            for _, r in anns.iterrows():
                action = f"{int(r['agent_id'])}_{int(r['target_id'])}_{r['action']}"
                for f in range(int(r['start_frame']), min(int(r['stop_frame']) + 1, total_frames)):
                    frame_labels[f] = action
            return frame_labels
        frame_labels = expand_to_frames(anns, total_frames)
        track['label'] = track['video_frame'].apply(
            lambda f: frame_labels[int(f)] if int(f) < len(frame_labels) else 'background'
        )
    # Per-mouse features - FIXED VERSION
    bc = track[track['bodypart'] == 'body_center'].sort_values(['mouse_id', 'video_frame'])
    
    # Calculate diffs with forward fill for first frame
    bc['x_cm_prev'] = bc.groupby('mouse_id')['x_cm'].shift(1)
    bc['y_cm_prev'] = bc.groupby('mouse_id')['y_cm'].shift(1)
    
    # Fill NaN positions with forward fill, then backward fill, then 0
    bc['x_cm'] = bc['x_cm'].fillna(method='ffill').fillna(method='bfill').fillna(0)
    bc['y_cm'] = bc['y_cm'].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Safe diff calculation
    bc['dx'] = bc['x_cm'] - bc['x_cm_prev']
    bc['dy'] = bc['y_cm'] - bc['y_cm_prev']
    bc['dx'] = bc['dx'].fillna(0)  # First frame has no previous, set velocity to 0
    bc['dy'] = bc['dy'].fillna(0)
    
    # Safe speed calculation
    bc['speed_sq'] = bc['dx']**2 + bc['dy']**2
    bc['speed'] = np.sqrt(np.maximum(bc['speed_sq'], 0))  # Ensure non-negative
    bc['speed'] = bc['speed'].fillna(0)
    
    # Safe heading calculation
    valid_vel = (bc['dx']**2 + bc['dy']**2) > 1e-8
    bc['heading'] = np.zeros_like(bc['dx'])
    bc.loc[valid_vel, 'heading'] = np.arctan2(bc.loc[valid_vel, 'dy'], bc.loc[valid_vel, 'dx'])
    bc['heading'] = bc['heading'].fillna(0)
    
    # Safe distance calculation
    bc['dist_center_sq'] = bc['x_cm']**2 + bc['y_cm']**2
    bc['dist_center'] = np.sqrt(np.maximum(bc['dist_center_sq'], 0))
    bc['dist_center'] = bc['dist_center'].fillna(0)
    
    # Drop temporary columns
    bc = bc.drop(['x_cm_prev', 'y_cm_prev', 'speed_sq', 'dist_center_sq'], axis=1)
    
    # Pairwise nose distances - FIXED VERSION
    nose = track[track['bodypart'] == 'nose'].copy()
    nose['x_cm'] = nose['x'] / pix_per_cm
    nose['y_cm'] = nose['y'] / pix_per_cm
    
    # Fill missing nose positions
    nose['x_cm'] = nose.groupby(['mouse_id', 'video_frame'])['x_cm'].fillna(0)
    nose['y_cm'] = nose.groupby(['mouse_id', 'video_frame'])['y_cm'].fillna(0)
    
    pairs = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    pair_dists = defaultdict(list)
    pair_dists['video_frame'] = []
    
    for frame in sorted(nose['video_frame'].unique()):
        frame_nose = nose[nose['video_frame'] == frame]
        pair_dists['video_frame'].append(frame)
        
        for a, b in pairs:
            # Get positions for each mouse, use mean if multiple detections
            pos_a = frame_nose[frame_nose['mouse_id'] == a][['x_cm', 'y_cm']]
            pos_b = frame_nose[frame_nose['mouse_id'] == b][['x_cm', 'y_cm']]
            
            x_a = pos_a['x_cm'].mean() if len(pos_a) > 0 else 0
            y_a = pos_a['y_cm'].mean() if len(pos_a) > 0 else 0
            x_b = pos_b['x_cm'].mean() if len(pos_b) > 0 else 0
            y_b = pos_b['y_cm'].mean() if len(pos_b) > 0 else 0
            
            # Safe distance calculation
            dx_pair = x_a - x_b
            dy_pair = y_a - y_b
            dist_sq = dx_pair**2 + dy_pair**2
            dist = np.sqrt(np.maximum(dist_sq, 0))
            pair_dists[f'dist_{a}_{b}'].append(dist)
    
    pair_df = pd.DataFrame(pair_dists)
    
    # Merge and final cleanup
    features_df = bc.merge(pair_df, on='video_frame', how='left').fillna(0)
    
    # Ensure all feature columns are non-negative before any sqrt operations
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if features_df[col].dtype in ['float64', 'float32']:
            features_df[col] = np.maximum(features_df[col], 0)
    
    if is_train:
        features_df['label'] = features_df['video_frame'].apply(
            lambda f: frame_labels[int(f)] if int(f) < len(frame_labels) else 'background'
        )
    
    return features_df, anns
# Training
non_mabe_labs = ~train_meta['lab_id'].str.contains('MABe22')
train_non_mabe = train_meta[non_mabe_labs]
all_behaviors = []
for behaviors in train_non_mabe['behaviors_labeled']:
    all_behaviors.extend(behaviors)
action_types = [parse_behavior_label(b)[2] for b in all_behaviors if parse_behavior_label(b)]
global_actions = sorted(set(action_types))
print(f"Global actions from non-MABe22: {global_actions}")
print(f"Number of unique classes: {len(global_actions)}")
print("Filtering non-MABe22 train data...")
annotated_non_mabe = train_non_mabe[train_non_mabe['behaviors_labeled'].apply(len) > 0]
print(f"Non-MABe22 annotated videos: {len(annotated_non_mabe)}")
def log_memory(phase):
    ram_percent = psutil.virtual_memory().percent
    print(f"{phase} - RAM usage: {ram_percent:.2f}%")
log_memory("After metadata load")
unique_labels = set(['background'])
def process_video(i):
    row = annotated_non_mabe.iloc[i]
    lab, vid = row['lab_id'], row['video_id']
    if lab == 'PleasantMeerkat' and vid == '1375833299':
        return None, set()
    df, _ = load_video_features(lab, vid)
    print(f"Loaded features for {lab}/{vid}")
    label_set = set(df['label'].unique()) if 'label' in df.columns else set()
    return df, label_set
print("Loading features in parallel...")
results = Parallel(n_jobs=-1)(delayed(process_video)(i) for i in range(len(annotated_non_mabe)))
all_dfs = [r[0] for r in results if r[0] is not None]
for r in results:
    unique_labels.update(r[1])
log_memory("After feature loading")
global_le = LabelEncoder()
global_le.fit(sorted(unique_labels))
print(f"Global LE fitted on {len(global_le.classes_)} unique labels: {global_le.classes_}")
all_features = []
for i, df in enumerate(all_dfs):
    if 'label' in df.columns and len(df[df['label'] != 'background']) > 0:
        df['video_id'] = annotated_non_mabe.iloc[i]['video_id']
        all_features.append(df)
print(f"Loaded {len(all_features)} videos with non-background labels.")
del all_dfs
gc.collect()
log_memory("After label encoding")
if all_features:
    feature_cols = ['x_cm', 'y_cm', 'speed', 'heading', 'dist_center', 'dist_1_2', 'dist_1_3', 'dist_1_4', 'dist_2_3', 'dist_2_4', 'dist_3_4', 'mouse_id']
    mouse_categories = [1, 2, 3, 4]
    seq_len = 3
    seq_X_list = []
    seq_y_list = []
    feat_cols_base = ['x_cm', 'y_cm', 'speed', 'heading', 'dist_center', 'dist_1_2', 'dist_1_3', 'dist_1_4', 'dist_2_3', 'dist_2_4', 'dist_3_4']
    dummy_cols = [f'mouse_{i}' for i in mouse_categories]
    feat_lstm = feat_cols_base + dummy_cols
    
    for i, df in enumerate(all_features):
        # Separate features and labels first
        feature_df = df[feature_cols].copy()
        feature_df = feature_df[feat_cols_base].fillna(0).astype(np.float32)  # Only numeric features
        
        # Mouse encoding separate
        mouse_ids = df['mouse_id'].copy()
        mouse_ids = pd.Categorical(mouse_ids, categories=mouse_categories)
        dummies = pd.get_dummies(mouse_ids, prefix='mouse')
        
        # Add missing dummies
        for col in set(dummy_cols) - set(dummies.columns):
            dummies[col] = 0
        
        # Combine
        feature_df = pd.concat([feature_df, dummies], axis=1)
        feature_df = feature_df[feat_lstm].values  # Now safe to convert to numpy
        
        # Encode labels separately if they exist
        if len(label_series) > 0 and 'label' in df.columns:
            y_encoded = global_le.transform(label_series)
        else:
            y_encoded = np.array([])
        
        # Create sequences per mouse
        for mouse_id in mouse_categories:
            mouse_mask = df['mouse_id'] == mouse_id
            if mouse_mask.sum() < seq_len:
                continue
                
            mouse_features = feature_df[mouse_mask].values
            mouse_labels = y_encoded[mouse_mask] if len(y_encoded) > 0 else np.array([])
            
            # Ensure we have enough frames
            if len(mouse_features) < seq_len:
                continue
                
            for j in range(len(mouse_features) - seq_len + 1):
                window = mouse_features[j:j+seq_len]
                seq_X_list.append(window)
                
                # Use the label from the last frame in the window
                if len(mouse_labels) > 0:
                    label_idx = j + seq_len - 1
                    if label_idx < len(mouse_labels):
                        seq_y_list.append(mouse_labels[label_idx])
        
        del feature_df, dummies
        gc.collect()
    
    if seq_X_list:
        seq_X = np.array(seq_X_list, dtype=np.float32)
        seq_y = np.array(seq_y_list, dtype=np.int32)
        print(f"LSTM training samples: {len(seq_y)} sequences")
    else:
        raise ValueError("No valid sequences found for training")
    
    del seq_X_list, seq_y_list
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_memory("After evaluation and cleanup")
# =============================================================================
# INFERENCE ON TEST SET & GENERATE SUBMISSION
# =============================================================================
submission_path = '/kaggle/working/submission.csv'
submission_rows = []
row_id = 0
processed_videos = set()
if os.path.exists(submission_path):
    existing_sub = pd.read_csv(submission_path)
    submission_rows = existing_sub.to_dict('records')
    row_id = existing_sub['row_id'].max() + 1 if not existing_sub.empty else 0
    processed_videos = set(existing_sub['video_id'].unique())
    print(f"Loaded existing submission with {len(submission_rows)} rows. Resuming from row_id {row_id}. Processed videos: {len(processed_videos)}")
print("Starting inference on test set...")
mouse_categories = [1, 2, 3, 4]
feat_cols_base = ['x_cm', 'y_cm', 'speed', 'heading', 'dist_center', 'dist_1_2', 'dist_1_3', 'dist_1_4', 'dist_2_3', 'dist_2_4', 'dist_3_4']
dummy_cols = [f'mouse_{i}' for i in mouse_categories]
feat_lstm = feat_cols_base + dummy_cols
seq_len = 3

for i, (idx, test_row) in enumerate(test_meta.iterrows()):
    lab, vid = test_row['lab_id'], test_row['video_id']
    if vid in processed_videos:
        print(f"Skipping already processed test video {i+1}/{len(test_meta)}: {lab}/{vid}")
        continue
    
    print(f"Processing test video: {lab}/{vid}")
    df_test, _ = load_video_features(lab, vid, is_train=False)
    print(f"Loaded test features for {lab}/{vid}")
    
    # Separate processing for test data (no labels)
    df_test['video_id'] = vid
    
    # Process features separately from categorical columns
    feature_cols_test = feat_cols_base + ['video_frame']
    df_features = df_test[feature_cols_test].copy().fillna(0).astype(np.float32)
    
    # Handle mouse_id encoding separately
    mouse_ids = df_test['mouse_id'].copy()
    mouse_ids = pd.Categorical(mouse_ids, categories=mouse_categories)
    dummies = pd.get_dummies(mouse_ids, prefix='mouse')
    
    # Add missing dummy columns
    missing_dummies = set(dummy_cols) - set(dummies.columns)
    for col in missing_dummies:
        dummies[col] = 0
    
    # Combine features with dummies
    df_test_full = pd.concat([df_features, dummies], axis=1)
    df_test_full['mouse_id'] = mouse_ids  # Keep original mouse_id for grouping
    
    # Now we can safely work with numeric features only for LSTM
    lstm_frame_probs = np.zeros((len(df_test), len(global_le.classes_)), dtype=np.float32)
    has_lstm_pred = np.zeros(len(df_test), dtype=bool)
    
    groups_test = df_test_full.groupby('mouse_id')
    lstm_model.eval()
    
    with torch.no_grad():
        for mouse, group in groups_test:
            if len(group) < seq_len:
                continue
                
            # Get sorted frames and corresponding original indices
            group_sorted = group.sort_values('video_frame').reset_index(drop=True)
            original_indices = df_test[df_test['video_frame'] == group_sorted['video_frame'].iloc[0]].index
            
            # Extract feature windows
            n_frames = len(group_sorted) - seq_len + 1
            if n_frames > 0:
                windows = []
                for j in range(n_frames):
                    window_features = group_sorted.iloc[j:j+seq_len][feat_lstm].values
                    windows.append(window_features)
                
                windows = np.array(windows, dtype=np.float32)
                
                # Scale features
                windows_reshaped = windows.reshape(-1, len(feat_lstm))
                windows_scaled = scaler.transform(windows_reshaped).reshape(windows.shape)
                
                # Model inference
                windows_t = torch.FloatTensor(windows_scaled).to(device, non_blocking=(device.type == 'cuda'))
                with autocast(enabled=(device.type == 'cuda')):
                    outs = lstm_model(windows_t)
                probs = torch.softmax(outs, dim=1).cpu().numpy()
                
                # Assign predictions to frames
                for j in range(n_frames):
                    # Find the corresponding frame in original df_test
                    target_frame = group_sorted['video_frame'].iloc[j + seq_len - 1]
                    frame_matches = df_test['video_frame'] == target_frame
                    if frame_matches.any():
                        frame_idx = df_test[frame_matches].index[0]
                        lstm_frame_probs[frame_idx] = probs[j]
                        has_lstm_pred[frame_idx] = True
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    
    # Use predictions for frames that have LSTM predictions, fallback to uniform for others
    valid_mask = has_lstm_pred
    frame_probs = lstm_frame_probs.copy()
    if not np.all(valid_mask):
        # For frames without LSTM predictions, use uniform distribution or background bias
        fallback_probs = np.ones(len(global_le.classes_)) / len(global_le.classes_)
        background_idx = global_le.transform(['background'])[0]
        fallback_probs[background_idx] = 0.8  # Bias toward background
        fallback_probs = fallback_probs / fallback_probs.sum()
        frame_probs[~valid_mask] = fallback_probs
    
    frame_preds = np.argmax(frame_probs, axis=1)
    frame_max_probs = np.max(frame_probs, axis=1)
    def test_preds_to_intervals(preds, probs, frames, video_behaviors, thresh=0.3):
        intervals = []
        global row_id
        current_action_idx = None
        start = None
        frames = frames.reset_index(drop=True)  # Ensure index starts from 0
        for i, (pred, prob, frame) in enumerate(zip(preds, probs, frames)):
            if prob > thresh and pred != current_action_idx:
                if current_action_idx is not None:
                    action_str = global_le.inverse_transform([current_action_idx])[0]
                    parts = action_str.split('_')
                    if len(parts) == 3:
                        agent, target, action = int(parts[0]), int(parts[1]), parts[2]
                        start_frame = int(start * 10)
                        stop_frame = int(frame * 10 - 1)
                        if stop_frame >= start_frame and any(parse_behavior_label(b) == (agent, target, action) for b in video_behaviors):
                            intervals.append({
                                'row_id': row_id,
                                'video_id': int(vid),  # Ensure integer type
                                'agent_id': f'mouse{agent}',
                                'target_id': f'mouse{target}',
                                'action': action,
                                'start_frame': start_frame,
                                'stop_frame': stop_frame
                            })
                            row_id += 1
                current_action_idx = pred
                start = frame
            elif prob <= thresh and current_action_idx is not None:
                action_str = global_le.inverse_transform([current_action_idx])[0]
                parts = action_str.split('_')
                if len(parts) == 3:
                    agent, target, action = int(parts[0]), int(parts[1]), parts[2]
                    start_frame = int(start * 10)
                    stop_frame = int(frame * 10 - 1)
                    if stop_frame >= start_frame and any(parse_behavior_label(b) == (agent, target, action) for b in video_behaviors):
                        intervals.append({
                            'row_id': row_id,
                            'video_id': int(vid),  # Ensure integer type
                            'agent_id': f'mouse{agent}',
                            'target_id': f'mouse{target}',
                            'action': action,
                            'start_frame': start_frame,
                            'stop_frame': stop_frame
                        })
                        row_id += 1
                current_action_idx = None
        if current_action_idx is not None:
            action_str = global_le.inverse_transform([current_action_idx])[0]
            parts = action_str.split('_')
            if len(parts) == 3:
                agent, target, action = int(parts[0]), int(parts[1]), parts[2]
                start_frame = int(start * 10)
                stop_frame = int(frames.iloc[-1] * 10 - 1)
                if stop_frame >= start_frame and any(parse_behavior_label(b) == (agent, target, action) for b in video_behaviors):
                    intervals.append({
                        'row_id': row_id,
                        'video_id': int(vid),  # Ensure integer type
                        'agent_id': f'mouse{agent}',
                        'target_id': f'mouse{target}',
                        'action': action,
                        'start_frame': start_frame,
                        'stop_frame': stop_frame
                    })
                    row_id += 1
        return intervals
    video_behaviors = test_row['behaviors_labeled']
    pred_ints = test_preds_to_intervals(frame_preds, frame_max_probs, df_test['video_frame'], video_behaviors)
    submission_rows.extend(pred_ints)
    print(f"Detected {len(pred_ints)} intervals for {vid}")
    
    del df_test, df_features, df_test_full, group_sorted
    gc.collect()
# Final submission formatting to match competition requirements
submission_df = pd.DataFrame(submission_rows)
if len(submission_df) > 0:
    submission_df = submission_df[['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']]
    submission_df['video_id'] = submission_df['video_id'].astype(int)      # Ensure integer type
    submission_df['row_id'] = submission_df['row_id'].astype(int)          # Ensure integer type
    submission_df['start_frame'] = submission_df['start_frame'].astype(int) # Ensure integer type
    submission_df['stop_frame'] = submission_df['stop_frame'].astype(int)   # Ensure integer type
    submission_df.sort_values(['video_id', 'start_frame'], inplace=True)
    submission_df['row_id'] = range(len(submission_df))  # Reset row_id to be sequential
else:
    # Create empty submission with correct format if no predictions
    submission_df = pd.DataFrame(columns=['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(f"Final submission shape: {submission_df.shape}")
print("Submission columns and dtypes:")
print(submission_df.dtypes)
print(submission_df.head())

