# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
x = []
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        x.append(str(os.path.join(dirname)))

print(set(x))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# --- Cell 1: Imports and Setup ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set a style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Define the base input path
BASE_PATH = Path('/kaggle/input/MABe-mouse-behavior-detection/')


# --- Cell 2: Load and Inspect Metadata ---
train_meta_df = pd.read_csv(BASE_PATH / 'train.csv')

print("Shape of the training metadata:")
print(train_meta_df.shape)

print("\nFirst 5 rows:")
display(train_meta_df.head())

print("\nData types and missing values:")
train_meta_df.info()


# --- Cell 3: Analyze Lab and Behavior Diversity ---
print("--- Lab Distribution ---")
lab_counts = train_meta_df['lab_id'].value_counts()
print(lab_counts)

plt.figure(figsize=(12, 7))
sns.barplot(x=lab_counts.index, y=lab_counts.values, palette='viridis')
plt.title('Number of Videos per Lab')
plt.xlabel('Lab ID')
plt.ylabel('Video Count')
plt.xticks(rotation=45, ha='right')
plt.show()

print("\n--- Unique sets of Body Parts Tracked ---")
print(train_meta_df['body_parts_tracked'].value_counts())

print("\n--- Tracking Method Distribution ---")
print(train_meta_df['tracking_method'].value_counts())


# --- Cell 4 (Corrected): Load Annotations and Add Metadata ---
import re

annotation_files = list(BASE_PATH.glob('train_annotation/*/*.parquet'))

all_annotations_list = []
for f in annotation_files:
    # Extract lab_id and video_id from the file path
    # The path is like: .../train_annotation/{lab_id}/{video_id}.parquet
    lab_id = f.parts[-2]
    video_id = int(f.stem) # f.stem gets the filename without extension

    # Load the data
    ann_df = pd.read_parquet(f)

    # Add the new columns
    ann_df['lab_id'] = lab_id
    ann_df['video_id'] = video_id

    all_annotations_list.append(ann_df)

all_annotations_df = pd.concat(all_annotations_list)

print("--- Annotations DataFrame with video_id and lab_id ---")
print(f"Total number of annotated events: {len(all_annotations_df)}")
display(all_annotations_df.head())

# The rest of the cell (plotting action counts) can remain the same.
action_counts = all_annotations_df['action'].value_counts()
print("\n--- Top 15 Most Frequent Behaviors ---")
print(action_counts.head(15))
# ... (plotting code) ...

plt.figure(figsize=(14, 8))
# Let's plot the top 20 for a better view
top_n = 20
sns.barplot(y=action_counts.index[:top_n], x=action_counts.values[:top_n], orient='h', palette='rocket')
plt.title(f'Top {top_n} Behavior Frequencies Across All Labs')
plt.xlabel('Number of Events')
plt.ylabel('Behavior')
plt.gca().invert_yaxis() # To show the most frequent at the top
plt.show()


# --- Cell 5 (No changes needed now): Inspect a Single Tracking File ---
# This cell should now work without errors.
# Let's find a video that has annotations
annotated_video_ids = all_annotations_df['video_id'].unique()
sample_video_id = annotated_video_ids[0]

# We already have the lab_id in our merged annotations dataframe
sample_lab_id = all_annotations_df[all_annotations_df['video_id'] == sample_video_id].iloc[0]['lab_id']

tracking_file_path = BASE_PATH / f'train_tracking/{sample_lab_id}/{sample_video_id}.parquet'

sample_tracking_df = pd.read_parquet(tracking_file_path)

print(f"--- Exploring Video ID: {sample_video_id} from Lab: {sample_lab_id} ---")
print("Shape of tracking data:", sample_tracking_df.shape)
print("\nUnique mice:", sample_tracking_df['mouse_id'].unique())
print("Unique body parts:", sample_tracking_df['bodypart'].unique())
print("\nFirst 5 rows:")
display(sample_tracking_df.head())

# --- Cell 6 (No changes needed): Pivot Tracking Data ---
# This cell should also work now.
# ... (pivot function and call) ...


# --- Cell 6: Pivot Tracking Data to Wide Format ---

def pivot_tracking_data(df):
    """Pivots the long-format tracking data to a wide format for a single video."""
    # First, get body parts on columns for each mouse
    pivoted_df = df.pivot_table(
        index=['video_frame', 'mouse_id'],
        columns='bodypart',
        values=['x', 'y']
    )
    # Flatten the multi-level column index (e.g., from ('x', 'nose') to 'x_nose')
    pivoted_df.columns = ['_'.join(col).strip() for col in pivoted_df.columns.values]
    pivoted_df = pivoted_df.reset_index()

    # Now, get each mouse's data onto the same row for each frame
    # We need to handle cases with more than 2 mice, but for now, let's focus on mouse1/mouse2
    # This assumes mouse_ids are like 'mouse1', 'mouse2'
    if all(pivoted_df['mouse_id'].isin(['mouse1', 'mouse2'])):
        final_df = pivoted_df.pivot(
            index='video_frame',
            columns='mouse_id',
        )
        # Flatten the multi-level columns again (e.g., from ('x_nose', 'mouse1') to 'x_nose_mouse1')
        final_df.columns = ['_'.join(col).strip() for col in final_df.columns.values]
        final_df = final_df.reset_index()
    else:
        # A more general approach if mouse_ids are not standard
        # This is more complex and we can defer it if not needed for the baseline
        print("Non-standard mouse IDs found. Returning intermediate pivot.")
        return pivoted_df

    return final_df

wide_df = pivot_tracking_data(sample_tracking_df)

print("--- Pivoted Wide Format ---")
print("Shape of wide data:", wide_df.shape)
display(wide_df.head())


import os
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from tqdm import tqdm
import gc

# --- 1. Configuration ---
class CONFIG:
    BASE_PATH = Path('/kaggle/input/MABe-mouse-behavior-detection/')
    BEHAVIORS_TO_TRAIN = ['sniff', 'attack', 'rear', 'approach', 'selfgroom']
    LGB_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 250,
        'learning_rate': 0.05,
        'num_leaves': 20,
        'max_depth': 5,
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    PROB_THRESHOLD = 0.5
    MIN_FRAMES_FOR_EVENT = 3
    NEGATIVE_SAMPLING_RATIO = 4

# --- 2. Helper Function: Load Annotations ---
def load_all_annotations(base_path):
    print("Loading all annotations...")
    annotation_files = list(base_path.glob('train_annotation/*/*.parquet'))
    all_annotations_list = []
    for f in tqdm(annotation_files):
        lab_id = f.parts[-2]
        video_id = int(f.stem)
        ann_df = pd.read_parquet(f)
        ann_df['lab_id'] = lab_id
        ann_df['video_id'] = video_id
        all_annotations_list.append(ann_df)
    return pd.concat(all_annotations_list).reset_index(drop=True)

# --- 3. Helper Function: Lightweight Feature Engineering ---
CANONICAL_SKELETON = ['nose', 'ear_left', 'ear_right', 'neck', 'body_center', 'tail_base']

def process_video_data_lightweight(video_id, lab_id, tracking_path):
    tracking_df = pd.read_parquet(tracking_path / f"{lab_id}/{video_id}.parquet")
    tracking_df['mouse_id'] = 'mouse' + tracking_df['mouse_id'].astype(str)
    cols_to_interpolate = ['x', 'y']
    tracking_df[cols_to_interpolate] = tracking_df.groupby(['mouse_id', 'bodypart'])[cols_to_interpolate].transform(
        lambda s: s.interpolate(method='linear', limit_direction='both')
    )
    wide_df = tracking_df.pivot_table(index='video_frame', columns=['mouse_id', 'bodypart'], values=['x', 'y']).fillna(0)
    wide_df.columns = ['_'.join(col) for col in wide_df.columns.values]
    features_list = []
    for mouse in ['mouse1', 'mouse2']:
        part_map = {'nose': 'nose', 'ear_left': 'ear_left', 'ear_right': 'ear_right', 'neck': 'neck', 'body_center': 'body_center', 'tail_base': 'tail_base'}
        raw_coords = {}
        for canonical_part in CANONICAL_SKELETON:
            for potential_name, assigned_name in part_map.items():
                if assigned_name == canonical_part and f'x_{mouse}_{potential_name}' in wide_df.columns:
                    raw_coords[f'{mouse}_{canonical_part}_x'] = wide_df[f'x_{mouse}_{potential_name}']
                    raw_coords[f'{mouse}_{canonical_part}_y'] = wide_df[f'y_{mouse}_{potential_name}']
                    break
            if f'{mouse}_{canonical_part}_x' not in raw_coords:
                 raw_coords[f'{mouse}_{canonical_part}_x'] = 0
                 raw_coords[f'{mouse}_{canonical_part}_y'] = 0
        center_x, center_y = raw_coords.get(f'{mouse}_body_center_x', 0), raw_coords.get(f'{mouse}_body_center_y', 0)
        for part in CANONICAL_SKELETON:
            x_norm = pd.Series(raw_coords.get(f'{mouse}_{part}_x', 0) - center_x, name=f'{mouse}_{part}_x_norm')
            y_norm = pd.Series(raw_coords.get(f'{mouse}_{part}_y', 0) - center_y, name=f'{mouse}_{part}_y_norm')
            features_list.extend([x_norm, y_norm])
    social_features_defs = [('nose', 'nose'), ('nose', 'body_center'), ('nose', 'tail_base'), ('body_center', 'body_center')]
    for part1, part2 in social_features_defs:
        try:
            m1_x, m1_y = wide_df[f'x_mouse1_{part1}'], wide_df[f'y_mouse1_{part1}']
            m2_x, m2_y = wide_df[f'x_mouse2_{part2}'], wide_df[f'y_mouse2_{part2}']
            dist = np.sqrt((m1_x - m2_x)**2 + (m1_y - m2_y)**2)
            dist_series = dist.fillna(0); dist_series.name = f'dist_m1{part1}_m2{part2}'
            features_list.append(dist_series)
        except KeyError: pass
    return pd.concat(features_list, axis=1).reset_index(drop=True)

# --- 4. Helper Function: Post-Processing [THIS FUNCTION WAS MISSING] ---
def probabilities_to_events(probs, threshold, min_frames):
    if probs is None or len(probs) == 0: return []
    binary_preds = (probs > threshold).astype(int)
    diffs = np.diff(binary_preds, prepend=0, append=0)
    starts = np.where(diffs == 1)[0]
    stops = np.where(diffs == -1)[0]
    events = []
    for start, stop in zip(starts, stops):
        if stop - start >= min_frames:
            events.append((start, stop - 1))
    return events

# --- 5. Main Training and Inference Logic ---
if __name__ == "__main__":
    train_meta_df = pd.read_csv(CONFIG.BASE_PATH / 'train.csv')
    test_meta_df = pd.read_csv(CONFIG.BASE_PATH / 'test.csv')
    
    all_annotations_df = load_all_annotations(CONFIG.BASE_PATH)
    annotated_video_ids = all_annotations_df['video_id'].unique()
    train_meta_df_filtered = train_meta_df[train_meta_df['video_id'].isin(annotated_video_ids)].copy()
    
    models = {}

    for behavior in CONFIG.BEHAVIORS_TO_TRAIN:
        print(f"\n--- Processing & Sampling data for behavior: {behavior} ---")
        X_train_list, y_train_list = [], []
        for row in tqdm(train_meta_df_filtered.itertuples(), total=len(train_meta_df_filtered), desc=f"Videos for {behavior}"):
            try:
                features_df = process_video_data_lightweight(row.video_id, row.lab_id, CONFIG.BASE_PATH / 'train_tracking')
                if features_df.empty: continue
            except (FileNotFoundError, KeyError) as e: continue
            labels = pd.Series(np.zeros(len(features_df)), name=behavior)
            video_ann = all_annotations_df[(all_annotations_df['video_id'] == row.video_id) & (all_annotations_df['action'] == behavior)]
            for _, ann_row in video_ann.iterrows():
                start, stop = ann_row['start_frame'], ann_row['stop_frame']
                if start < len(labels): labels.iloc[start : stop + 1] = 1
            positive_indices, negative_indices = labels[labels == 1].index, labels[labels == 0].index
            if len(positive_indices) == 0: continue
            num_neg_to_sample = min(int(len(positive_indices) * CONFIG.NEGATIVE_SAMPLING_RATIO), len(negative_indices))
            sampled_negative_indices = np.random.choice(negative_indices, size=num_neg_to_sample, replace=False)
            final_indices = np.concatenate([positive_indices, sampled_negative_indices])
            X_train_list.append(features_df.iloc[final_indices]); y_train_list.append(labels.iloc[final_indices])
        if not X_train_list: print(f"No data for {behavior} after sampling. Skipping."); continue
        X_train, y_train = pd.concat(X_train_list).reset_index(drop=True), pd.concat(y_train_list).reset_index(drop=True)
        print(f"--- Training model for: {behavior} ---"); print(f"Sampled training data shape: X={X_train.shape}, y={y_train.shape}")
        model = lgb.LGBMClassifier(**CONFIG.LGB_PARAMS); model.fit(X_train, y_train); models[behavior] = model
        del X_train, y_train, X_train_list, y_train_list; gc.collect()

    print("\n--- Starting Inference on Test Set ---")
    all_predictions = []
    for row in tqdm(test_meta_df.itertuples(), total=len(test_meta_df), desc="Inferring on test videos"):
        try:
            features_df = process_video_data_lightweight(row.video_id, row.lab_id, CONFIG.BASE_PATH / 'test_tracking')
            if features_df.empty: continue
        except (FileNotFoundError, KeyError): continue
        for behavior, model in models.items():
            probs = model.predict_proba(features_df)[:, 1]
            events = probabilities_to_events(probs, CONFIG.PROB_THRESHOLD, CONFIG.MIN_FRAMES_FOR_EVENT)
            for start, stop in events:
                agent_id = 'mouse1'
                target_id = 'mouse2' if behavior not in ['selfgroom', 'rear'] else 'mouse1'
                all_predictions.append({'video_id': row.video_id, 'agent_id': agent_id, 'target_id': target_id, 'action': behavior, 'start_frame': start, 'stop_frame': stop})
    
    submission_df = pd.DataFrame(all_predictions)
    if not submission_df.empty:
        submission_df['row_id'] = submission_df.index
        submission_df = submission_df[['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']]
    else:
        submission_df = pd.DataFrame(columns=['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
    submission_df.to_csv('submission.csv', index=False)
    print("\nSubmission file 'submission.csv' created successfully.")
    print(submission_df.head())




