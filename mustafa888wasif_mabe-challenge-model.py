import numpy as np
import pandas as pd
import os
from glob import glob

# Initialize lists to store data
all_tracking_dfs = []
all_annotation_dfs = []
total_frames = 0
total_sequences = 0
tracking_cols = set()
annotation_cols = set()

# Load CSV files
train_csv = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv') if os.path.exists('/kaggle/input/MABe-mouse-behavior-detection/train.csv') else None
test_csv = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv') if os.path.exists('/kaggle/input/MABe-mouse-behavior-detection/test.csv') else None
sample_submission = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/sample_submission.csv') if os.path.exists('/kaggle/input/MABe-mouse-behavior-detection/sample_submission.csv') else None

# Load Parquet files
for dirpath, _, filenames in os.walk('/kaggle/input/MABe-mouse-behavior-detection'):
    for filename in filenames:
        if filename.endswith('.parquet'):
            file_path = os.path.join(dirpath, filename)
            try:
                df = pd.read_parquet(file_path)
                if 'train_tracking' in dirpath or 'test_tracking' in dirpath:
                    all_tracking_dfs.append(df)
                    tracking_cols.update(df.columns)
                    total_frames += len(df)
                    total_sequences += 1
                elif 'train_annotation' in dirpath:
                    all_annotation_dfs.append(df)
                    annotation_cols.update(df.columns)
            except Exception:
                pass

print("Dataset is loaded.")

# Print insights
print("\n=== Dataset Insights ===")
print(f"Total sequences: {total_sequences}")
print(f"Total frames: {total_frames}")
print(f"Tracking columns: {len(tracking_cols)} (e.g., {list(tracking_cols)[:5]})")
print(f"Annotation columns: {len(annotation_cols)} (e.g., {list(annotation_cols)[:5]})")
if train_csv is not None:
    print(f"Train CSV shape: {train_csv.shape}")
if test_csv is not None:
    print(f"Test CSV shape: {test_csv.shape}")
if sample_submission is not None:
    print(f"Sample submission shape: {sample_submission.shape}")

# Behavior distribution (if annotations available)
behavior_cols = [col for col in annotation_cols if col not in tracking_cols and col not in ['frame', 'timestamp', 'sequence_id']]
if all_annotation_dfs and behavior_cols:
    combined_annotations = pd.concat(all_annotation_dfs, ignore_index=True)
    for col in behavior_cols[:2]:  # Limit to avoid clutter
        if col in combined_annotations.columns:
            print(f"\nDistribution of {col}:\n{combined_annotations[col].value_counts(normalize=True)}")



import numpy as np
import pandas as pd
import os
from glob import glob
from collections import defaultdict

# Set data directory
data_dir = '/kaggle/input/MABe-mouse-behavior-detection'

# Initialize minimal storage
csv_info = {'train.csv': {}, 'test.csv': {}, 'sample_submission.csv': {}}
tracking_info = defaultdict(list)
annotation_info = defaultdict(list)
unique_bodyparts = set()
unique_mouse_ids = set()
seq_lengths = []
num_mice_per_seq = []
bodyparts_per_mouse_list = []
nan_total = 0
unique_actions = set()
action_counts = defaultdict(int)
event_lengths = []
unique_agents = set()
unique_targets = set()
labeled_frames = 0
sequence_id_map = {}  # Map video_id to file paths

# Load CSVs minimally
for csv_file in csv_info:
    path = os.path.join(data_dir, csv_file)
    if os.path.exists(path):
        df = pd.read_csv(path, nrows=5)
        csv_info[csv_file]['shape'] = (pd.read_csv(path, usecols=[0]).shape[0], df.shape[1])
        csv_info[csv_file]['columns'] = ', '.join(df.columns)
    else:
        csv_info[csv_file] = {'shape': 'N/A', 'columns': 'N/A'}

# Build sequence ID map from train.csv
if csv_info['train.csv']['shape'] != 'N/A':
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'), usecols=['video_id'])
    for vid in train_df['video_id'].unique():
        sequence_id_map[vid] = {'tracking': None, 'annotation': None}

# Process tracking files
tracking_dirs = ['train_tracking', 'test_tracking']
for tdir in tracking_dirs:
    parquet_files = glob(os.path.join(data_dir, f'{tdir}/**/*.parquet'), recursive=True)
    for file in parquet_files:
        df = pd.read_parquet(file, columns=['video_frame', 'mouse_id', 'bodypart', 'x', 'y'])
        unique_bodyparts.update(df['bodypart'].unique())
        unique_mouse_ids.update(df['mouse_id'].unique())
        seq_lengths.append(df['video_frame'].max() - df['video_frame'].min() + 1)
        num_mice = df['mouse_id'].nunique()
        num_mice_per_seq.append(num_mice)
        bodyparts_per_mouse_list.append(df.groupby('mouse_id')['bodypart'].nunique().mean())
        nan_total += df[['x', 'y']].isna().sum().sum()
        sample = df[['x', 'y']].dropna().sample(n=min(100, len(df)), random_state=42)
        tracking_info['x'].extend(sample['x'])
        tracking_info['y'].extend(sample['y'])
        # Update sequence_id_map
        vid = os.path.basename(file).split('.')[0]
        if vid in sequence_id_map:
            sequence_id_map[vid]['tracking'] = file
        del df

# Process annotation files
annotation_files = glob(os.path.join(data_dir, 'train_annotation/**/*.parquet'), recursive=True)
for file in annotation_files:
    df = pd.read_parquet(file, columns=['action', 'start_frame', 'stop_frame', 'agent_id', 'target_id'])
    unique_actions.update(df['action'].unique())
    for act, count in df['action'].value_counts().items():
        action_counts[act] += count
    lengths = df['stop_frame'] - df['start_frame'] + 1
    event_lengths.extend(lengths)
    labeled_frames += lengths.sum()
    unique_agents.update(df['agent_id'].unique())
    unique_targets.update(df['target_id'].unique())
    vid = os.path.basename(file).split('.')[0]
    if vid in sequence_id_map:
        sequence_id_map[vid]['annotation'] = file
    del df

# Create summary tables
csv_summary = {
    'File': list(csv_info.keys()),
    'Shape': [csv_info[f]['shape'] for f in csv_info],
    'Columns': [csv_info[f]['columns'] for f in csv_info]
}
csv_df = pd.DataFrame(csv_summary)

x_stats = pd.Series(tracking_info['x']) if tracking_info['x'] else pd.Series()
y_stats = pd.Series(tracking_info['y']) if tracking_info['y'] else pd.Series()
tracking_summary = {
    'Metric': ['Unique Bodyparts', 'Sequences', 'Seq Lengths (mean/std)', 'Mice per Seq (mode)', 'Bodyparts per Mouse (mean)', 
               'X Coords (min/max/mean)', 'Y Coords (min/max/mean)', 'NaNs', 'Total Frames'],
    'Value': [f"{len(unique_bodyparts)}/{', '.join(sorted(unique_bodyparts)[:5])}...",
              len(seq_lengths),
              f"{np.mean(seq_lengths):.2f}/{np.std(seq_lengths):.2f}" if seq_lengths else 'N/A',
              pd.Series(num_mice_per_seq).mode()[0] if num_mice_per_seq else 'N/A',
              f"{np.mean(bodyparts_per_mouse_list):.2f}" if bodyparts_per_mouse_list else 'N/A',
              f"{x_stats.min():.2f}/{x_stats.max():.2f}/{x_stats.mean():.2f}" if not x_stats.empty else 'N/A',
              f"{y_stats.min():.2f}/{y_stats.max():.2f}/{y_stats.mean():.2f}" if not y_stats.empty else 'N/A',
              nan_total,
              sum(seq_lengths) if seq_lengths else 0]
}
tracking_df = pd.DataFrame(tracking_summary)

action_counts_sorted = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
annotation_summary = {
    'Metric': ['Unique Actions', 'Top 3 Actions', 'Event Lengths (mean/std)', 'Agent/Target IDs', 'Labeled Frames', 'Labeled Proportion'],
    'Value': [f"{len(unique_actions)}/{', '.join(sorted(unique_actions)[:5])}...",
              '; '.join([f"{act}: {cnt}" for act, cnt in action_counts_sorted[:3]]),
              f"{np.mean(event_lengths):.2f}/{np.std(event_lengths):.2f}" if event_lengths else 'N/A',
              f"{len(unique_agents)}/{len(unique_targets)}",
              labeled_frames,
              f"{labeled_frames / sum(seq_lengths):.4f}" if seq_lengths and sum(seq_lengths) > 0 else '0.0000']
}
annotation_df = pd.DataFrame(annotation_summary)

# Sample merge with robust matching
merge_summary = {}
if sequence_id_map:
    for vid, paths in list(sequence_id_map.items())[:1]:  # Try first valid pair
        if paths['tracking'] and paths['annotation']:
            df_ann = pd.read_parquet(paths['annotation'], columns=['action', 'start_frame', 'stop_frame'])
            df_track = pd.read_parquet(paths['tracking'], columns=['video_frame', 'mouse_id', 'bodypart', 'x', 'y'])
            df_frames = pd.DataFrame({'video_frame': range(df_track['video_frame'].min(), df_track['video_frame'].max() + 1)})
            df_frames['action'] = 'other'
            for _, row in df_ann.iterrows():
                mask = (df_frames['video_frame'] >= row['start_frame']) & (df_frames['video_frame'] <= row['stop_frame'])
                df_frames.loc[mask, 'action'] = row['action']
            df_pivot = df_track.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values=['x', 'y'])
            df_pivot.columns = ['_'.join(map(str, col)).strip() for col in df_pivot.columns.values]
            df_merged = df_frames.merge(df_pivot.reset_index(), on='video_frame', how='left')
            merge_summary = {
                'Metric': ['Merged Shape', 'Unique Actions', 'Input Features'],
                'Value': [f"{df_merged.shape[0]}/{df_merged.shape[1]}",
                          ', '.join(df_merged['action'].unique()),
                          len([col for col in df_merged.columns if col not in ['action', 'video_frame']])]
            }
            del df_ann, df_track, df_frames, df_pivot, df_merged
            break
merge_df = pd.DataFrame(merge_summary) if merge_summary else pd.DataFrame({'Metric': ['No Merge'], 'Value': ['N/A']})

# Display tables
print("\n=== CSV Summary ===")
print(csv_df.to_markdown(index=False))
print("\n=== Tracking Summary ===")
print(tracking_df.to_markdown(index=False))
print("\n=== Annotation Summary ===")
print(annotation_df.to_markdown(index=False))
print("\n=== Sample Merge Summary ===")
print(merge_df.to_markdown(index=False))
print("\nModel Tips: Normalize coords to [0,1], use GNN/LSTM/Transformer for temporal data, handle class imbalance (e.g., oversample sniff), semi-supervised for 15% labeled data.")


import numpy as np
import pandas as pd
import os
from glob import glob
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Function to prepare data for a sequence
def prepare_data(track_file, ann_file=None, mice=4, bodyparts=29, sample_rate=1):
    # Read tracking data with only necessary columns to save memory
    df_track = pd.read_parquet(track_file, columns=['video_frame', 'mouse_id', 'bodypart', 'x', 'y'])
    # Sort by video_frame to ensure order
    df_track = df_track.sort_values('video_frame')
    # Subsample if sample_rate > 1 (but default to 1 for accuracy on frames)
    if sample_rate > 1:
        df_track = df_track.iloc[::sample_rate]
    # Use pivot_table to handle any potential duplicates safely
    df_pivot = df_track.pivot_table(index='video_frame', columns=['mouse_id', 'bodypart'], values=['x', 'y'], aggfunc='first')
    df_pivot.columns = ['_'.join(map(str, col)).strip() for col in df_pivot.columns]
    df_pivot = df_pivot.fillna(0)  # Handle missing keypoints
    
    # Feature engineering: distances and velocities for key bodyparts
    feature_cols = []
    key_parts = ['nose', 'tail_base', 'body_center']
    for i in range(1, mice + 1):
        for j in range(i + 1, mice + 1):
            for part in key_parts:
                col_ix = f'x_{i}_{part}'
                col_iy = f'y_{i}_{part}'
                col_jx = f'x_{j}_{part}'
                col_jy = f'y_{j}_{part}'
                dist_col = f'dist_{i}_{j}_{part}'
                if all(col in df_pivot.columns for col in [col_ix, col_iy, col_jx, col_jy]):
                    dx = df_pivot[col_ix] - df_pivot[col_jx]
                    dy = df_pivot[col_iy] - df_pivot[col_jy]
                    df_pivot[dist_col] = np.nan_to_num(np.sqrt(np.maximum(0.0, dx**2 + dy**2)), nan=0.0)
                else:
                    df_pivot[dist_col] = 0.0
                feature_cols.append(dist_col)
        # Velocity for each mouse (using nose)
        col_nx = f'x_{i}_nose'
        col_ny = f'y_{i}_nose'
        vel_col = f'vel_{i}_nose'
        if col_nx in df_pivot.columns and col_ny in df_pivot.columns:
            dx = df_pivot[col_nx].diff()
            dy = df_pivot[col_ny].diff()
            df_pivot[vel_col] = np.nan_to_num(np.sqrt(np.maximum(0.0, dx**2 + dy**2)), nan=0.0)
        else:
            df_pivot[vel_col] = 0.0
        feature_cols.append(vel_col)
    
    X = df_pivot[feature_cols].copy()
    
    if ann_file:
        df_ann = pd.read_parquet(ann_file)
        frames = df_pivot.index  # Use subsampled frames if applicable
        df_labels = pd.DataFrame({'video_frame': frames}).set_index('video_frame')
        df_labels['action'] = 'other'
        for _, row in df_ann.iterrows():
            mask = (df_labels.index >= row['start_frame']) & (df_labels.index <= row['stop_frame'])
            df_labels.loc[mask, 'action'] = row['action']
        y = df_labels['action']
        return X, y
    return X, None

# Load metadata
train_csv = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
test_csv = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')

# Label encoder for actions
le = LabelEncoder()

# Train on a small subset to manage memory
num_train_seq = 5
X_train_list = []
y_train_list = []
for idx in range(num_train_seq):
    print(f"Processing training sequence {idx + 1}/{num_train_seq}")
    row = train_csv.iloc[idx]
    vid = row['video_id']
    track_files = glob(f'/kaggle/input/MABe-mouse-behavior-detection/train_tracking/**/{vid}.parquet', recursive=True)
    ann_files = glob(f'/kaggle/input/MABe-mouse-behavior-detection/train_annotation/**/{vid}.parquet', recursive=True)
    track_path = track_files[0] if track_files else None
    ann_path = ann_files[0] if ann_files else None
    if track_path and ann_path:
        X, y = prepare_data(track_path, ann_path)
        X_train_list.append(X)
        y_train_list.append(y)
        del X, y  # Free memory

if X_train_list:
    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)
    y_train_enc = le.fit_transform(y_train)
    
    # Compute class weights to handle imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_enc), y=y_train_enc)
    weights = np.array([class_weights[label] for label in y_train_enc])
    
    # LightGBM params for multi-class classification
    params = {
        'objective': 'multiclass',
        'num_class': len(le.classes_),
        'metric': 'multi_logloss',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbose': -1
    }
    train_set = lgb.Dataset(X_train, label=y_train_enc, weight=weights)
    model = lgb.train(params, train_set, num_boost_round=200)
    del X_train, y_train, train_set  # Free memory
else:
    print("No training data found. Using dummy predictions.")
    model = None

# Predict on test
submission = []
row_id = 0
for _, row in test_csv.iterrows():
    print(f"Processing test sequence for video_id: {row['video_id']}")
    vid = row['video_id']
    track_files = glob(f'/kaggle/input/MABe-mouse-behavior-detection/test_tracking/**/{vid}.parquet', recursive=True)
    track_path = track_files[0] if track_files else None
    if track_path:
        X_test, _ = prepare_data(track_path)
        if model is not None:
            preds = model.predict(X_test)
            pred_labels = le.inverse_transform(np.argmax(preds, axis=1))
        else:
            pred_labels = np.full(len(X_test), 'other')
        df_pred = pd.DataFrame({'video_frame': X_test.index, 'action': pred_labels})
        df_pred['shifted'] = df_pred['action'].shift(1)
        df_pred['group'] = (df_pred['action'] != df_pred['shifted']).cumsum()
        events = df_pred[df_pred['action'] != 'other'].groupby(['action', 'group']).agg(
            start_frame=('video_frame', 'min'),
            stop_frame=('video_frame', 'max')
        ).reset_index()  # Keep 'action' in columns
        events['video_id'] = vid
        events['agent_id'] = 1  # Dummy; improve with clustering
        events['target_id'] = 2
        events['row_id'] = range(row_id, row_id + len(events))
        row_id += len(events)
        submission.append(events)
        del X_test, df_pred, events  # Free memory

if submission:
    submission_df = pd.concat(submission, ignore_index=True)
    submission_df = submission_df[['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']]
else:
    submission_df = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/sample_submission.csv')

submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission created.")

