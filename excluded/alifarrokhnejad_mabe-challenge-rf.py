import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from collections import defaultdict, Counter
import ast
import warnings
warnings.filterwarnings('ignore')
import gc
from joblib import Parallel, delayed
import psutil
import os

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
    # Per-mouse features
    bc = track[track['bodypart'] == 'body_center'].sort_values(['mouse_id', 'video_frame'])
    bc['dx'] = bc.groupby('mouse_id')['x_cm'].diff()
    bc['dy'] = bc.groupby('mouse_id')['y_cm'].diff()
    bc['speed'] = np.sqrt(bc['dx']**2 + bc['dy']**2)
    bc['heading'] = np.arctan2(bc['dy'], bc['dx'])
    bc['dist_center'] = np.sqrt(bc['x_cm']**2 + bc['y_cm']**2)
    # Pairwise nose distances
    nose = track[track['bodypart'] == 'nose']
    pairs = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    pair_dists = defaultdict(list)
    pair_dists['video_frame'] = []
    for frame in nose['video_frame'].unique():
        frame_nose = nose[nose['video_frame'] == frame]
        pair_dists['video_frame'].append(frame)
        for a, b in pairs:
            pos_a = frame_nose[frame_nose['mouse_id'] == a][['x_cm', 'y_cm']].mean()
            pos_b = frame_nose[frame_nose['mouse_id'] == b][['x_cm', 'y_cm']].mean()
            dist = np.sqrt((pos_a['x_cm'] - pos_b['x_cm'])**2 + (pos_a['y_cm'] - pos_b['y_cm'])**2) if not pos_a.empty and not pos_b.empty else 0
            pair_dists[f'dist_{a}_{b}'].append(dist)
    pair_df = pd.DataFrame(pair_dists)
    # Merge
    features_df = bc.merge(pair_df, on='video_frame', how='left').fillna(0)
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
    X_list = [df[feature_cols].astype(np.float32) for df in all_features]
    X_all = pd.concat(X_list)
    X_all['mouse_id'] = pd.Categorical(X_all['mouse_id'], categories=mouse_categories)
    X_all = pd.get_dummies(X_all, columns=['mouse_id'], prefix='mouse').fillna(0).astype(np.float32)
    full_labels = [df['label'].tolist() for df in all_features]
    full_labels = [lbl for sublist in full_labels for lbl in sublist]
    y_all = np.array([global_le.transform([lbl])[0] for lbl in full_labels], dtype=np.int32)
    print(f"Total training samples: {len(y_all)}")
    print("Training RF on full data...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1, max_depth=10, max_features=0.8)
    rf.fit(X_all, y_all)
    rf_train_acc = accuracy_score(y_all, rf.predict(X_all))
    print(f"RF Train accuracy: {rf_train_acc:.4f}")
    feat_cols = X_all.columns.tolist()
    del X_all, y_all
    gc.collect()
    print("RF ready.")
    log_memory("After RF training")
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
feat_cols = feat_cols if 'feat_cols' in locals() else []
print("Starting inference on test set...")
mouse_categories = [1, 2, 3, 4]
for i, (idx, test_row) in enumerate(test_meta.iterrows()):
    lab, vid = test_row['lab_id'], test_row['video_id']
    if vid in processed_videos:
        print(f"Skipping already processed test video {i+1}/{len(test_meta)}: {lab}/{vid}")
        continue
    print(f"Processing test video: {lab}/{vid}")
    df_test, _ = load_video_features(lab, vid, is_train=False)
    print(f"Loaded test features for {lab}/{vid}")
    df_test['video_id'] = vid
    X_test = df_test[['x_cm', 'y_cm', 'speed', 'heading', 'dist_center', 'dist_1_2', 'dist_1_3', 'dist_1_4', 'dist_2_3', 'dist_2_4', 'dist_3_4', 'mouse_id']].fillna(0).astype(np.float32)
    X_test['mouse_id'] = pd.Categorical(X_test['mouse_id'], categories=mouse_categories)
    X_test = pd.get_dummies(X_test, columns=['mouse_id'], prefix='mouse').fillna(0).astype(np.float32)
    if len(feat_cols) > 0:
        missing_cols = set(feat_cols) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = 0
        X_test = X_test[feat_cols].astype(np.float32)
    rf_probs = rf.predict_proba(X_test)
    frame_probs = rf_probs
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
    del df_test, X_test
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

