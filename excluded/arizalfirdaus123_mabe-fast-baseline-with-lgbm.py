import os
import gc
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

verbose = True # Set to False to reduce output messages

class CONFIG:
    BASE_PATH = Path('/kaggle/input/MABe-mouse-behavior-detection/')
    OUTPUT_PATH = Path('/kaggle/working/')
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
        'verbose': -1,
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }
    
    PROB_THRESHOLD = 0.15
    MIN_FRAMES_FOR_EVENT = 1
    NEGATIVE_SAMPLING_RATIO = 4
    RANDOM_STATE = 42

CONFIG.OUTPUT_PATH.mkdir(exist_ok=True)
print("Configuration loaded.")


def load_all_annotations(base_path):
    """
    Loads all training annotation files, adds lab_id and video_id.
    """
    print("Loading all annotations...")
    annotation_files = list(base_path.glob('train_annotation/*/*.parquet'))
    all_annotations_list = []
    # Use tqdm for progress bar
    for f in tqdm(annotation_files, desc="Loading annotation files"):
        try:
            lab_id = f.parts[-2] # Get lab_id from folder name
            video_id = int(f.stem) # Get video_id from file name (without extension)
            ann_df = pd.read_parquet(f)
            ann_df['lab_id'] = lab_id
            ann_df['video_id'] = video_id
            all_annotations_list.append(ann_df)
        except Exception as e:
            print(f"Error loading {f}: {e}") # Handle potential errors
            continue # Skip this file if error occurs
            
    if not all_annotations_list:
        raise FileNotFoundError("No annotation files loaded. Check paths.")
        
    # Concatenate all loaded dataframes
    full_annotations_df = pd.concat(all_annotations_list).reset_index(drop=True)
    print(f"Loaded {len(full_annotations_df)} annotation events from {len(annotation_files)} files.")
    return full_annotations_df

# Load metadata
train_meta_df = pd.read_csv(CONFIG.BASE_PATH / 'train.csv')
test_meta_df = pd.read_csv(CONFIG.BASE_PATH / 'test.csv')

# Load annotations using the function
all_annotations_df = load_all_annotations(CONFIG.BASE_PATH)

# Display some info
print("\nTrain metadata shape:", train_meta_df.shape)
print("Test metadata shape:", test_meta_df.shape)
print("Annotations shape:", all_annotations_df.shape)
print("\nSample annotations:")
display(all_annotations_df.head())


# Define the body parts we consistently want features for
# This helps handle variations in tracked parts across labs
CANONICAL_SKELETON = ['nose', 'ear_left', 'ear_right', 'neck', 'body_center', 'tail_base']

# Define some useful distances between parts for social interaction features
SOCIAL_FEATURE_DEFS = [
    ('mouse1', 'nose', 'mouse2', 'nose'),
    ('mouse1', 'nose', 'mouse2', 'body_center'),
    ('mouse1', 'nose', 'mouse2', 'tail_base'),
    ('mouse1', 'body_center', 'mouse2', 'body_center'),
]

def process_video_data(video_id, lab_id, tracking_path, is_train=True):
    """
    Loads tracking data for a single video, performs lightweight feature engineering.
    Returns a dataframe with features for each frame.
    """
    # Construct the file path
    folder = 'train_tracking' if is_train else 'test_tracking'
    file_path = tracking_path / f"{lab_id}/{video_id}.parquet"

    try:
        # Load the raw tracking data
        tracking_df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"Warning: File not found {file_path}. Skipping video.")
        return pd.DataFrame() # Return empty dataframe if file not found

    # Standardize mouse_id format (e.g., to 'mouse1', 'mouse2')
    tracking_df['mouse_id'] = 'mouse' + tracking_df['mouse_id'].astype(str)

    # --- Simple Imputation: Linear Interpolation ---
    # Interpolate missing 'x' and 'y' values linearly within each mouse/bodypart group
    cols_to_interpolate = ['x', 'y']
    tracking_df[cols_to_interpolate] = tracking_df.groupby(['mouse_id', 'bodypart'])[cols_to_interpolate].transform(
        lambda s: s.interpolate(method='linear', limit_direction='both', limit_area='inside')
    )
    # Fill any remaining NaNs (e.g., at the very start/end if interpolation didn't cover) with 0
    tracking_df[cols_to_interpolate] = tracking_df[cols_to_interpolate].fillna(0)


    # --- Pivot to Wide Format ---
    # Get x, y for each bodypart/mouse onto a single row per frame
    try:
        wide_df = tracking_df.pivot_table(
            index='video_frame',
            columns=['mouse_id', 'bodypart'],
            values=['x', 'y']
        )
        # Flatten the multi-level column index (e.g., ('x', 'mouse1', 'nose') -> 'x_mouse1_nose')
        wide_df.columns = ['_'.join(col) for col in wide_df.columns.values]
        wide_df = wide_df.reset_index() # Make video_frame a regular column
    except Exception as e:
         print(f"Error pivoting video {video_id} ({lab_id}): {e}. Skipping video.")
         return pd.DataFrame() # Return empty if pivot fails

    # --- Feature Creation ---
    features_list = []
    
    # Check if wide_df is empty after pivot before proceeding
    if wide_df.empty:
        print(f"Warning: wide_df is empty after pivot for video {video_id} ({lab_id}). Skipping.")
        return pd.DataFrame()
        
    df_index = wide_df.index # Get index for creating placeholder series

    # 1. Normalized Body Part Coordinates (relative to body_center)
    for mouse in ['mouse1', 'mouse2']:
        center_x_col = f'x_{mouse}_body_center'
        center_y_col = f'y_{mouse}_body_center'

        # Use body_center if available, otherwise default to 0 (or mean coord)
        center_x = wide_df[center_x_col] if center_x_col in wide_df.columns else 0
        center_y = wide_df[center_y_col] if center_y_col in wide_df.columns else 0

    for part in CANONICAL_SKELETON:
            part_x_col = f'x_{mouse}_{part}'
            part_y_col = f'y_{mouse}_{part}'

            # Get part coordinate Series, or create a Series of zeros if missing
            part_x = wide_df[part_x_col] if part_x_col in wide_df.columns else pd.Series(0, index=df_index, dtype=float) # Ensure float type
            part_y = wide_df[part_y_col] if part_y_col in wide_df.columns else pd.Series(0, index=df_index, dtype=float) # Ensure float type

            # Calculate the difference
            diff_x = part_x - center_x
            diff_y = part_y - center_y

            # Explicitly create a named Series FROM the difference result
            # This handles cases where diff_x/diff_y might be scalar if wide_df has 1 row
            x_norm = pd.Series(diff_x, index=df_index, name=f'{mouse}_{part}_x_norm', dtype=float)
            y_norm = pd.Series(diff_y, index=df_index, name=f'{mouse}_{part}_y_norm', dtype=float)

            features_list.extend([x_norm, y_norm])

    for m1, p1, m2, p2 in SOCIAL_FEATURE_DEFS:
        m1_x_col, m1_y_col = f'x_{m1}_{p1}', f'y_{m1}_{p1}'
        m2_x_col, m2_y_col = f'x_{m2}_{p2}', f'y_{m2}_{p2}'

        if m1_x_col in wide_df.columns and m2_x_col in wide_df.columns:
            m1_x, m1_y = wide_df[m1_x_col], wide_df[m1_y_col]
            m2_x, m2_y = wide_df[m2_x_col], wide_df[m2_y_col]
            # Ensure calculations handle potential NaNs from pivot before sqrt
            dx = (m1_x - m2_x).fillna(0)
            dy = (m1_y - m2_y).fillna(0)
            dist = np.sqrt(dx**2 + dy**2)
            dist.name = f'dist_{m1}{p1}_{m2}{p2}'
            features_list.append(dist.astype(float)) # Ensure float type
        else:
             placeholder = pd.Series(0, index=df_index, name=f'dist_{m1}{p1}_{m2}{p2}', dtype=float) # Ensure float type
             features_list.append(placeholder)

    # Combine all feature series into a single DataFrame
    if not features_list:
        return pd.DataFrame() # Return empty if no features could be created
        
    features_df = pd.concat(features_list, axis=1)
    
    # Add video_frame back for potential merging later if needed (optional)
    # features_df['video_frame'] = wide_df['video_frame']

    return features_df.reset_index(drop=True) # Ensure clean index


# --- Test the function on one sample video ---
try:
    sample_video_id_test = train_meta_df.iloc[0]['video_id']
    sample_lab_id_test = train_meta_df.iloc[0]['lab_id']
    print(f"Testing feature engineering on video: {sample_video_id_test} (Lab: {sample_lab_id_test})")
    sample_features = process_video_data(sample_video_id_test, sample_lab_id_test, CONFIG.BASE_PATH / 'train_tracking', is_train=True)
    print("Feature engineering successful. Sample features shape:", sample_features.shape)
    display(sample_features.head())
    del sample_features # Clean up memory
    gc.collect()
except Exception as e:
    print(f"Error during feature engineering test: {e}")


# Filter train_meta_df to only include videos that actually have annotations
# This avoids processing videos for which we have no labels during training
annotated_video_ids = all_annotations_df['video_id'].unique()
train_meta_df_filtered = train_meta_df[train_meta_df['video_id'].isin(annotated_video_ids)].copy()

print(f"Original train videos: {len(train_meta_df)}")
print(f"Using {len(train_meta_df_filtered)} videos with annotations for training.")

# Dictionary to store the trained models, one for each behavior
models = {}


# Set seed for numpy's random operations (like sampling)
np.random.seed(CONFIG.RANDOM_STATE)

# Loop through each behavior we want to train a model for
for behavior in CONFIG.BEHAVIORS_TO_TRAIN:
    print(f"\n--- Processing & Sampling data for behavior: {behavior} ---")
    
    # Lists to collect sampled features and labels from all videos
    X_train_sampled_list = []
    y_train_sampled_list = []
    
    # Iterate through each training video (that has annotations)
    # Use tqdm for progress bar
    for row in tqdm(train_meta_df_filtered.itertuples(), total=len(train_meta_df_filtered), desc=f"Videos for {behavior}"):
        
        # Process the video to get features
        features_df = process_video_data(row.video_id, row.lab_id, CONFIG.BASE_PATH / 'train_tracking', is_train=True)
        
        # Skip if feature extraction failed or returned empty
        if features_df.empty:
            # print(f"Skipping video {row.video_id} (Lab: {row.lab_id}) due to empty features.")
            continue
            
        # --- Create Labels for the current behavior ---
        # Initialize labels series with all zeros (non-event)
        labels = pd.Series(np.zeros(len(features_df)), name=behavior, dtype=np.int8)
        
        # Get annotations specific to this video and behavior
        video_annotations = all_annotations_df[
            (all_annotations_df['video_id'] == row.video_id) &
            (all_annotations_df['action'] == behavior)
        ]
        
        # Mark frames within annotated event ranges as 1 (event)
        for _, ann_row in video_annotations.iterrows():
            start_f = ann_row['start_frame']
            stop_f = ann_row['stop_frame']
            # Ensure indices are within the bounds of the features_df
            if start_f < len(labels):
                # +1 because iloc is exclusive of the stop index
                labels.iloc[start_f : min(stop_f + 1, len(labels))] = 1

        # --- Negative Sampling ---
        positive_indices = labels[labels == 1].index
        negative_indices = labels[labels == 0].index
        
        # Only proceed if there are positive examples for this behavior in this video
        if len(positive_indices) == 0:
            # print(f"No positive examples for {behavior} in video {row.video_id}. Skipping sampling.")
            continue
            
        # Calculate number of negative samples needed based on ratio
        num_neg_to_sample = min(int(len(positive_indices) * CONFIG.NEGATIVE_SAMPLING_RATIO), len(negative_indices))
        
        # Randomly choose negative indices if there are any negatives to sample from
        if num_neg_to_sample > 0 and len(negative_indices) > 0:
            sampled_negative_indices = np.random.choice(negative_indices, size=num_neg_to_sample, replace=False)
            # Combine positive and sampled negative indices
            final_indices = np.concatenate([positive_indices, sampled_negative_indices])
        else:
            # If no negatives to sample or ratio is 0, just use positives
            final_indices = positive_indices
            
        # Append the sampled data to our lists
        if len(final_indices) > 0:
            X_train_sampled_list.append(features_df.iloc[final_indices])
            y_train_sampled_list.append(labels.iloc[final_indices])
            
        # --- Memory Management ---
        del features_df, labels, video_annotations, positive_indices, negative_indices, final_indices
        # gc.collect() # Optional: Force garbage collection more frequently if memory is tight

    # --- Train Model for the Behavior ---
    # Check if we collected any data at all for this behavior
    if not X_train_sampled_list:
        print(f"No data collected for behavior '{behavior}' after sampling across all videos. Skipping training.")
        continue
        
    # Concatenate all sampled data into final training sets
    X_train = pd.concat(X_train_sampled_list).reset_index(drop=True)
    y_train = pd.concat(y_train_sampled_list).reset_index(drop=True)
    
    print(f"--- Training model for: {behavior} ---")
    print(f"Sampled training data shape: X={X_train.shape}, y={y_train.shape}")
    
    # Initialize and train the LightGBM model
    model = lgb.LGBMClassifier(**CONFIG.LGB_PARAMS)
    model.fit(X_train, y_train)
    
    # Store the trained model
    models[behavior] = model
    
    # --- Memory Management ---
    print(f"Finished training for {behavior}. Cleaning up memory.")
    del X_train, y_train, X_train_sampled_list, y_train_sampled_list
    gc.collect() # Force garbage collection after training each model

print("\n--- Model training complete for selected behaviors ---")
print(f"Trained models for: {list(models.keys())}")


def probabilities_to_events(probs, threshold, min_frames):
    """
    Converts frame-wise probabilities into a list of event tuples (start_frame, stop_frame).
    (Code is omitted here for brevity, assume the correct function is loaded)
    """
    if probs is None or len(probs) == 0:
        return []

    # Apply threshold to get binary predictions (0 or 1)
    binary_preds = (probs > threshold).astype(np.int8)

    # Find where sequences of 1s start and end
    diffs = np.diff(binary_preds, prepend=0, append=0)

    start_frames = np.where(diffs == 1)[0]
    stop_frames = np.where(diffs == -1)[0] # stop_frames index is exclusive

    events = []
    # Pair up start and stop frames
    for start, stop in zip(start_frames, stop_frames):
        # Check if the duration meets the minimum requirement
        if (stop - start) >= min_frames:
            # Add event (inclusive frame indices)
            events.append((start, stop - 1)) # -1 because stop_frames was exclusive

    return events


print("\n--- Starting Inference on Test Set ---")
all_predictions = [] # List to store prediction dictionaries

# Define individual behaviors for target assignment (MUST be kept updated)
INDIVIDUAL_BEHAVIORS = ['selfgroom', 'rear'] # Add 'dig', 'investigation' etc., if you train them

# Iterate through each test video
for row in tqdm(test_meta_df.itertuples(), total=len(test_meta_df), desc="Inferring on test videos"):

    # Process the test video to get features
    features_df = process_video_data(row.video_id, row.lab_id, CONFIG.BASE_PATH / 'test_tracking', is_train=False)

    # Skip if feature extraction failed
    if features_df.empty:
        continue

    # Predict for each behavior using the corresponding trained model
    for behavior, model in models.items():
        # Only predict if the model was successfully trained
        if behavior in models:
            
            # Predict probabilities (get probability of the positive class, index 1)
            probs = model.predict_proba(features_df)[:, 1]

            # Convert probabilities to events using post-processing
            events = probabilities_to_events(probs, CONFIG.PROB_THRESHOLD, CONFIG.MIN_FRAMES_FOR_EVENT)

            # Format events for submission
            for start, stop in events:
                
                # --- CORRECTED AGENT/TARGET LOGIC ---
                agent_id = 'mouse1' 
                
                if behavior in INDIVIDUAL_BEHAVIORS:
                    target_id = agent_id # Target is self for individual behaviors
                else:
                    target_id = 'mouse2' # Target is the other mouse for social behaviors
                # --- END CORRECTION ---

                all_predictions.append({
                    'video_id': row.video_id,
                    'agent_id': agent_id,
                    'target_id': target_id,
                    'action': behavior,
                    'start_frame': start,
                    'stop_frame': stop
                })
        # else:
            # print(f"Warning: Model for behavior '{behavior}' not found. Skipping prediction.")


    # --- Memory Management ---
    del features_df
    gc.collect()

print("\n--- Inference complete ---")


import pandas as pd
import numpy as np

print("Creating submission dataframe...")

# Convert list of prediction dictionaries to a DataFrame
submission_df = pd.DataFrame(all_predictions)

# Ensure required columns exist even if no predictions were made initially
required_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
for col in required_cols:
    if col not in submission_df.columns:
        submission_df[col] = []

# --- CLEANING FUNCTION ---
def clean_overlaps(df):
    """Remove overlapping predictions for same agent/target pair"""
    if df.empty:
        return df
    
    cleaned = []
    for (vid, agent, target), group in df.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values('start_frame').reset_index(drop=True)
        
        if len(group) == 0:
            continue
            
        valid_mask = [True]
        last_stop = group.iloc[0]['stop_frame']
        
        for i in range(1, len(group)):
            current_start = group.iloc[i]['start_frame']
            if current_start < last_stop:
                # Overlap detected - skip this prediction
                valid_mask.append(False)
                if verbose:
                    print(f"  Removing overlap: video={vid}, agent={agent}, target={target}, frame={current_start}")
            else:
                valid_mask.append(True)
                last_stop = group.iloc[i]['stop_frame']
        
        cleaned.append(group[valid_mask])
    
    return pd.concat(cleaned, ignore_index=True) if cleaned else pd.DataFrame(columns=df.columns)

# --- MERGE OVERLAPPING INTERVALS FUNCTION ---
def merge_overlapping_intervals(df_group):
    """Merges overlapping or adjacent intervals within a group."""
    if df_group.empty:
        return df_group

    # Sort by start_frame is crucial
    df_group = df_group.sort_values('start_frame')

    merged = []
    current_start = -1
    current_stop = -1

    for _, row in df_group.iterrows():
        if current_start == -1:
            # Initialize with the first interval
            current_start = row['start_frame']
            current_stop = row['stop_frame']
        else:
            # Check for overlap or adjacency (stop >= start - 1)
            if row['start_frame'] <= current_stop + 1:
                # Merge by extending the current stop frame if necessary
                current_stop = max(current_stop, row['stop_frame'])
            else:
                # No overlap, finalize the previous interval and start a new one
                merged.append({'start_frame': current_start, 'stop_frame': current_stop})
                current_start = row['start_frame']
                current_stop = row['stop_frame']

    # Add the last processed interval
    if current_start != -1:
        merged.append({'start_frame': current_start, 'stop_frame': current_stop})

    # Return a new dataframe with merged intervals
    if merged:
        merged_df = pd.DataFrame(merged)
        # Get the keys from the first row of the original group
        keys = df_group.iloc[0][['video_id', 'agent_id', 'target_id', 'action']]
        for col, value in keys.items():
            merged_df[col] = value
        return merged_df
    else:
        return pd.DataFrame(columns=df_group.columns)

# --- APPLY CLEANING AND MERGING ---
if not submission_df.empty:
    print(f"Shape before cleaning overlaps: {submission_df.shape}")
    
    # Step 1: Clean hard overlaps (same start frame or overlapping ranges)
    submission_df = clean_overlaps(submission_df)
    print(f"Shape after cleaning overlaps: {submission_df.shape}")
    
    # Step 2: Merge adjacent/overlapping intervals for same action
    print(f"Shape before merging adjacent intervals: {submission_df.shape}")
    submission_merged = submission_df.groupby(
        ['video_id', 'agent_id', 'target_id', 'action'], 
        group_keys=False
    ).apply(merge_overlapping_intervals).reset_index(drop=True)
    print(f"Shape after merging adjacent intervals: {submission_merged.shape}")
    submission_df = submission_merged
    
    # Step 3: Filter out very short events (likely noise)
    duration = submission_df['stop_frame'] - submission_df['start_frame']
    before_filter = len(submission_df)
    submission_df = submission_df[duration >= 3].reset_index(drop=True)
    after_filter = len(submission_df)
    if before_filter > after_filter:
        print(f"Filtered out {before_filter - after_filter} short events (duration < 3 frames)")
else:
    print("Submission DataFrame is empty, skipping cleaning and merging.")

# --- ADD row_id COLUMN ---
if not submission_df.empty:
    # Sort for final consistency before adding row_id
    submission_df = submission_df.sort_values(
        by=['video_id', 'start_frame', 'action']
    ).reset_index(drop=True)
    submission_df['row_id'] = submission_df.index
else:
    submission_df['row_id'] = []

# Reorder columns
final_cols = ['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
if 'row_id' not in submission_df.columns:
    submission_df['row_id'] = []
submission_df = submission_df.reindex(columns=final_cols, fill_value=np.nan)

# --- VALIDATION BLOCK ---
print("\n--- Submission DataFrame Validation ---")
print("Shape:", submission_df.shape)
print("Columns:", submission_df.columns.tolist())
print("Data Types:\n", submission_df.dtypes)
print("Any NaNs:", submission_df.isnull().values.any())
print("NaNs per column:\n", submission_df.isnull().sum())

# Cast types safely ONLY IF not empty AND no NaNs in integer columns
if not submission_df.empty and not submission_df[['row_id', 'start_frame', 'stop_frame']].isnull().values.any():
    try:
        submission_df['row_id'] = submission_df['row_id'].astype(int)
        submission_df['video_id'] = submission_df['video_id'].astype(int)
        submission_df['start_frame'] = submission_df['start_frame'].astype(int)
        submission_df['stop_frame'] = submission_df['stop_frame'].astype(int)
        print("\nâœ“ Type casting successful")
        
        # Additional validations
        assert (submission_df['start_frame'] < submission_df['stop_frame']).all(), "ERROR: Some start_frame >= stop_frame"
        print("âœ“ All start_frame < stop_frame")
        
        # Check for duplicate rows
        dup_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame']
        duplicates = submission_df.duplicated(subset=dup_cols, keep=False)
        if duplicates.any():
            print(f"WARNING: Found {duplicates.sum()} duplicate rows")
            print(submission_df[duplicates][dup_cols])
        else:
            print("âœ“ No duplicate rows found")
        
        # Final overlap check
        print("\nChecking for remaining overlaps...")
        overlap_found = False
        for (vid, agent, target), group in submission_df.groupby(['video_id', 'agent_id', 'target_id']):
            group = group.sort_values('start_frame')
            frames_used = set()
            for _, row in group.iterrows():
                new_frames = set(range(row['start_frame'], row['stop_frame']))
                if frames_used.intersection(new_frames):
                    print(f"ERROR: Overlap still exists for video={vid}, agent={agent}, target={target}")
                    overlap_found = True
                    break
                frames_used.update(new_frames)
            if overlap_found:
                break
        
        if not overlap_found:
            print("âœ“ No overlaps detected")
        
    except Exception as e:
        print(f"\nERROR during validation: {e}")
else:
    print("\nWARNING: Submission DataFrame is empty or contains NaNs, skipping casting/validation")

print("------------------------------------\n")

# --- HANDLE EMPTY SUBMISSION ---
if submission_df.empty or len(submission_df) == 0:
    print("WARNING: Empty submission detected. Creating dummy submission.")
    submission_df = pd.DataFrame({
        'row_id': [0],
        'video_id': [test['video_id'].iloc[0] if len(test) > 0 else 438887472],
        'agent_id': ['mouse1'],
        'target_id': ['self'],
        'action': ['rear'],
        'start_frame': [0],
        'stop_frame': [100]
    })

# Save to csv file
submission_path = CONFIG.OUTPUT_PATH / 'submission.csv'
submission_df.to_csv(submission_path, index=False)

print(f"Submission file saved to: {submission_path}")
print(f"Total predictions: {len(submission_df)}")

if not submission_df.empty:
    print("\nSample of final submission file:")
    display(submission_df.head(10))
    
    # Show statistics
    print("\nSubmission Statistics:")
    print(f"  Unique videos: {submission_df['video_id'].nunique()}")
    print(f"  Unique actions: {submission_df['action'].nunique()}")
    print("  Actions distribution:")
    print(submission_df['action'].value_counts())
    print(f"\n  Average event duration: {(submission_df['stop_frame'] - submission_df['start_frame']).mean():.1f} frames")
else:
    print("\nFinal submission file is empty.")




