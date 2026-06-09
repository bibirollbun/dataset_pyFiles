import pandas as pd
import numpy as np
import polars as pl
import json
import os
import gc
import itertools
from collections import defaultdict
import warnings
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', context='notebook', palette='viridis')

import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import make_pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not found. The ensemble will proceed using only LightGBM.")

from scipy import signal

warnings.filterwarnings('ignore')
pd.options.display.max_columns = 100
tqdm.pandas()

BASE_PATH = '/kaggle/input/MABe-mouse-behavior-detection/'
TRAIN_TRACKING_DIR = os.path.join(BASE_PATH, 'train_tracking')
TEST_TRACKING_DIR = os.path.join(BASE_PATH, 'test_tracking')
TRAIN_ANNOTATION_DIR = os.path.join(BASE_PATH, 'train_annotation')

DROP_BODY_PARTS = [
    'headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
    'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
    'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint'
]

print("Environment setup complete.")
print(f"XGBoost available: {XGBOOST_AVAILABLE}")


# Load the metadata files
train_df = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
test_df = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))

# --- Data Cleaning: Exclude MABe22 Labs ---
# As per competition guidelines, these labs are from a different source and should not be used for training.
print(f"Original number of training videos: {len(train_df)}")
train_df = train_df[~train_df['lab_id'].str.startswith('MABe22_')].reset_index(drop=True)
print(f"Number of training videos after filtering MABe22: {len(train_df)}")

print("\n--- Train Metadata Sample ---")
display(train_df.head(3))

print("\n--- Test Metadata Sample ---")
display(test_df.head(3))


fig, axes = plt.subplots(2, 1, figsize=(16, 14))

lab_counts = train_df['lab_id'].value_counts()
sns.barplot(x=lab_counts.index, y=lab_counts.values, ax=axes[0], palette='plasma')
axes[0].set_title('Distribution of Videos per Lab in Training Set', fontsize=16)
axes[0].set_ylabel('Number of Videos')
axes[0].set_xlabel('Lab ID')
axes[0].tick_params(axis='x', rotation=45)

body_part_counts = train_df['body_parts_tracked'].apply(lambda x: f"{len(json.loads(x))} parts").value_counts()
sns.barplot(x=body_part_counts.index, y=body_part_counts.values, ax=axes[1], palette='magma')
axes[1].set_title('Distribution of Videos per Number of Tracked Body Parts', fontsize=16)
axes[1].set_ylabel('Number of Videos')
axes[1].set_xlabel('Number of Tracked Body Parts')

plt.tight_layout()
plt.show()

BODY_PART_CONFIGS = train_df['body_parts_tracked'].unique()
print(f"Found {len(BODY_PART_CONFIGS)} unique body part tracking configurations.")


behavior_durations = defaultdict(list)
TARGET_BEHAVIORS = {'attack', 'mount', 'chase'}

print("Analyzing behavior durations from a sample of videos...")
for _, row in tqdm(train_df.head(50).iterrows(), total=50): # Analyze first 50 videos
    annotation_path = os.path.join(TRAIN_ANNOTATION_DIR, row['lab_id'], f"{row['video_id']}.parquet")
    
    try:
        annotation_df = pd.read_parquet(annotation_path)
        # Ensure we only look at our target behaviors
        annotation_df = annotation_df[annotation_df['action'].isin(TARGET_BEHAVIORS)]
        
        if not annotation_df.empty:
            durations = annotation_df['stop_frame'] - annotation_df['start_frame']
            for action, duration in zip(annotation_df['action'], durations):
                behavior_durations[action].append(duration)
                
    except FileNotFoundError:
        # This is a graceful way to handle videos that have tracking data but no annotations.
        # Our pipeline must not fail if an annotation file is missing.
        # print(f"HANDLED ERROR: Annotation file not found for video {row['video_id']}. Skipping.")
        pass

# Plotting the distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle('Distribution of Behavior Durations (in Frames)', fontsize=18)

for i, action in enumerate(TARGET_BEHAVIORS):
    if behavior_durations[action]:
        sns.histplot(behavior_durations[action], ax=axes[i], bins=50, kde=True)
        mean_duration = np.mean(behavior_durations[action])
        axes[i].axvline(mean_duration, color='r', linestyle='--', label=f'Mean: {mean_duration:.1f} frames')
        axes[i].set_title(f'{action.capitalize()}')
        axes[i].set_xlabel('Duration (frames)')
        axes[i].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


def generate_mouse_data(metadata_df, mode='train'):
    """
    A generator that yields processed data for each mouse-pair interaction in each video.
    This approach is highly memory-efficient.
    
    Args:
        metadata_df (pd.DataFrame): A dataframe with video metadata.
        mode (str): 'train' or 'test'. In 'train' mode, it also yields labels.
    
    Yields:
        tuple: (interaction_type, tracking_data, metadata, labels/actions)
               - interaction_type: 'pair' (for this problem, we only focus on pairs)
               - tracking_data: DataFrame with processed coordinates for the agent-target pair.
               - metadata: DataFrame with video_id, frame_id, agent, target.
               - labels (train): DataFrame with binary labels for each frame.
               - actions (test): List of possible actions to predict for this pair.
    """
    assert mode in ['train', 'test'], "Mode must be 'train' or 'test'."
    
    for _, row in metadata_df.iterrows():
        tracking_path = os.path.join(
            TRAIN_TRACKING_DIR if mode == 'train' else TEST_TRACKING_DIR,
            row['lab_id'],
            f"{row['video_id']}.parquet"
        )
        
        if not os.path.exists(tracking_path):
            continue

        # Load and pivot tracking data
        tracking_df = pd.read_parquet(tracking_path)
        
        # Standardize body parts for high-dim sets
        if len(tracking_df['bodypart'].unique()) > 10:
             tracking_df = tracking_df[~tracking_df['bodypart'].isin(DROP_BODY_PARTS)]

        pivoted_df = tracking_df.pivot(
            index='video_frame', 
            columns=['mouse_id', 'bodypart'], 
            values=['x', 'y']
        )
        
        # Reorder levels for easier access: (mouse_id, bodypart, coordinate)
        pivoted_df = pivoted_df.reorder_levels([1, 2, 0], axis=1).sort_index(axis=1)
        
        # Normalize by pixel-to-cm ratio
        if 'pix_per_cm_approx' in row and row['pix_per_cm_approx'] > 0:
            pivoted_df /= row['pix_per_cm_approx']

        # Parse the labeled behaviors to find agent-target pairs
        try:
            labeled_behaviors = json.loads(row['behaviors_labeled'])
            behavior_df = pd.DataFrame([b.replace("'", "").split(',') for b in labeled_behaviors], 
                                       columns=['agent', 'target', 'action'])
        except (TypeError, json.JSONDecodeError):
            continue
            
        # Get all unique mice present in the video
        available_mice = pivoted_df.columns.get_level_values('mouse_id').unique()

        # Iterate through all directed pairs of mice (mouse1 -> mouse2, mouse2 -> mouse1, etc.)
        for agent_id, target_id in itertools.permutations(available_mice, 2):
            agent_str = f"mouse{agent_id}"
            target_str = f"mouse{target_id}"

            # Check which behaviors are relevant for this specific agent-target pair
            pair_actions = behavior_df[
                (behavior_df['agent'] == agent_str) & 
                (behavior_df['target'] == target_str)
            ]['action'].unique()
            
            # We only care about pairs involved in one of our target behaviors
            relevant_actions = list(set(pair_actions) & TARGET_BEHAVIORS)
            if not relevant_actions:
                continue

            # Create the dataframes for this specific pair
            agent_data = pivoted_df[agent_id]
            target_data = pivoted_df[target_id]
            pair_tracking_data = pd.concat([agent_data, target_data], axis=1, keys=['agent', 'target'])

            pair_meta_data = pd.DataFrame({
                'video_id': row['video_id'],
                'video_frame': pair_tracking_data.index,
                'agent_id': agent_str,
                'target_id': target_str,
            })
            
            if mode == 'train':
                annotation_path = os.path.join(TRAIN_ANNOTATION_DIR, row['lab_id'], f"{row['video_id']}.parquet")
                
                # Initialize labels as all-zero
                pair_labels = pd.DataFrame(0, index=pair_tracking_data.index, columns=relevant_actions)
                
                if os.path.exists(annotation_path):
                    ann_df = pd.read_parquet(annotation_path)
                    
                    # Filter annotations for the current agent-target pair and relevant actions
                    pair_ann = ann_df[
                        (ann_df['agent_id'] == agent_id) &
                        (ann_df['target_id'] == target_id) &
                        (ann_df['action'].isin(relevant_actions))
                    ]
                    
                    # Vectorize the labels: set frames within start/stop to 1
                    for _, ann_row in pair_ann.iterrows():
                        pair_labels.loc[ann_row['start_frame']:ann_row['stop_frame'], ann_row['action']] = 1
                
                yield 'pair', pair_tracking_data, pair_meta_data, pair_labels
            
            else: # mode == 'test'
                yield 'pair', pair_tracking_data, pair_meta_data, relevant_actions



def create_pair_features(pair_data, body_parts):
    """
    Engineers a comprehensive feature set from the tracking data of an agent-target pair.
    """
    X = pd.DataFrame(index=pair_data.index)
    
    # For safe access, check which body parts are available
    agent_parts = pair_data['agent'].columns.get_level_values(0).unique()
    target_parts = pair_data['target'].columns.get_level_values(0).unique()

    # --- Level 1: Geometric & Distance Features ---
    # Inter-mouse distances between all pairs of body parts
    for p1 in agent_parts:
        for p2 in target_parts:
            if p1 in body_parts and p2 in body_parts:
                X[f'dist_{p1}_{p2}'] = np.linalg.norm(
                    pair_data['agent'][p1].values - pair_data['target'][p2].values, axis=1
                )

    # Agent's body elongation (if parts available)
    if 'nose' in agent_parts and 'tail_base' in agent_parts and 'ear_left' in agent_parts and 'ear_right' in agent_parts:
        nose_tail_dist = np.linalg.norm(pair_data['agent']['nose'].values - pair_data['agent']['tail_base'].values, axis=1)
        ear_ear_dist = np.linalg.norm(pair_data['agent']['ear_left'].values - pair_data['agent']['ear_right'].values, axis=1)
        X['agent_elongation'] = nose_tail_dist / (ear_ear_dist + 1e-6)

    # --- Level 2: Kinematic Features (Agent-centric) ---
    if 'body_center' in agent_parts:
        center_x = pair_data['agent']['body_center']['x']
        center_y = pair_data['agent']['body_center']['y']
        
        vel_x = center_x.diff()
        vel_y = center_y.diff()
        speed = np.sqrt(vel_x**2 + vel_y**2)
        
        accel_x = vel_x.diff()
        accel_y = vel_y.diff()
        acceleration = np.sqrt(accel_x**2 + accel_y**2)

        for w in [5, 15, 45]: # Short, medium, long windows
            # Speed features
            X[f'agent_speed_mean_{w}'] = speed.rolling(w, min_periods=1, center=True).mean()
            X[f'agent_speed_std_{w}'] = speed.rolling(w, min_periods=1, center=True).std()
            
            # Acceleration features
            X[f'agent_accel_mean_{w}'] = acceleration.rolling(w, min_periods=1, center=True).mean()
            X[f'agent_accel_max_{w}'] = acceleration.rolling(w, min_periods=1, center=True).max()
    
    # --- Level 3: Interaction & Relational Features ---
    if 'body_center' in agent_parts and 'body_center' in target_parts:
        # Relative kinematics
        agent_center = pair_data['agent']['body_center']
        target_center = pair_data['target']['body_center']
        
        rel_pos_vec = agent_center - target_center
        rel_dist = np.linalg.norm(rel_pos_vec.values, axis=1)
        
        agent_vel_vec = agent_center.diff()
        target_vel_vec = target_center.diff()
        
        # Rate of approach/retreat
        X['dist_change'] = pd.Series(rel_dist).diff()
        
        # Are they moving in similar directions? (Velocity Correlation)
        agent_speed = np.linalg.norm(agent_vel_vec.values, axis=1)
        target_speed = np.linalg.norm(target_vel_vec.values, axis=1)
        
        # Use np.einsum for efficient row-wise dot product
        dot_product = np.einsum('ij,ij->i', agent_vel_vec.fillna(0), target_vel_vec.fillna(0))
        X['velocity_corr'] = dot_product / (agent_speed * target_speed + 1e-6)

    # --- Level 4: Advanced Trajectory & Signal Features (Agent-centric) ---
    if 'body_center' in agent_parts and len(speed.dropna()) > 128:
        # Curvature of agent's path
        angle = np.arctan2(vel_y, vel_x)
        turn_rate = angle.diff().abs()
        X['agent_turn_rate_mean_30'] = turn_rate.rolling(30, min_periods=1, center=True).mean()

        # Frequency domain feature: Dominant movement frequency
        # Useful for detecting rhythmic behaviors like mounting
        fs = 30 # Assuming ~30 FPS
        f, psd = signal.welch(speed.fillna(0), fs=fs, nperseg=min(128, len(speed.dropna())))
        X['agent_dominant_freq'] = f[np.argmax(psd)] if len(f) > 0 else 0
        
    # Reduce feature dimensionality by replacing NaNs from rolling ops
    # and dropping columns that are all-NaN (can happen with short videos)
    X = X.fillna(method='bfill').fillna(method='ffill')
    X = X.dropna(axis=1, how='all')
    
    return X


class StratifiedDownsamplingClassifier(BaseEstimator, ClassifierMixin):
    """
    A wrapper for any classifier that performs stratified downsampling before fitting.
    This is highly effective for severely imbalanced datasets.
    """
    def __init__(self, estimator, n_samples=100000, random_state=42):
        self.estimator = estimator
        self.n_samples = n_samples
        self.random_state = random_state

    def fit(self, X, y):
        n_total = len(y)
        if n_total <= self.n_samples:
            # If the dataset is small enough, use all of it
            self.estimator.fit(X, y)
        else:
            # Perform stratified sampling to create a balanced subset
            sss = StratifiedShuffleSplit(n_splits=1, train_size=self.n_samples, random_state=self.random_state)
            try:
                train_idx, _ = next(sss.split(X, y))
                self.estimator.fit(X[train_idx], y[train_idx])
            except Exception as e:
                # Fallback to simple random sampling if stratification fails (e.g., too few positive samples)
                print(f"Stratified sampling failed: {e}. Falling back to random sampling.")
                downsample_indices = np.random.choice(n_total, self.n_samples, replace=False)
                self.estimator.fit(X[downsample_indices], y[downsample_indices])
        
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
        
    def predict(self, X):
        return self.estimator.predict(X)


models_dict = {}

print("--- Starting Model Training ---")

for config_str in BODY_PART_CONFIGS:
    body_parts = json.loads(config_str)
    if len(body_parts) > 10:
        body_parts = [bp for bp in body_parts if bp not in DROP_BODY_PARTS]
    
    print(f"\nTraining models for configuration with {len(body_parts)} body parts...")
    
    # --- 1. Data Collection for this config ---
    config_train_df = train_df[train_df['body_parts_tracked'] == config_str]
    
    all_X = []
    all_y = []
    
    # Assuming a realistic number of pairs per video for tqdm
    data_generator = generate_mouse_data(config_train_df, mode='train')
    for _, tracking_data, _, labels in tqdm(data_generator, total=len(config_train_df) * 6):
        if labels.empty or tracking_data.empty:
            continue
            
        features = create_pair_features(tracking_data, body_parts)
        
        common_index = features.index.intersection(labels.index)
        if not common_index.empty:
            all_X.append(features.loc[common_index])
            all_y.append(labels.loc[common_index])
            
    if not all_X:
        print(f"  No valid training data found for this configuration. Skipping.")
        continue
        
    X_train = pd.concat(all_X)
    y_train = pd.concat(all_y)
    del all_X, all_y
    gc.collect()

    X_train_np = X_train.values
    
    # --- 2. Model Training for each behavior ---
    config_models = {}
    for behavior in y_train.columns:
        print(f"  Training for behavior: '{behavior}'...")
        
        # ==================================================================
        # CORE FIX: Handle NaNs by filtering data before training
        # ==================================================================
        
        # 1. Get the target series for the current behavior
        y_series = y_train[behavior]
        
        # 2. Create a boolean mask to identify rows with valid (non-NaN) labels
        valid_mask = y_series.notna()
        
        # 3. Apply the mask to both the features and the labels
        X_train_behavior = X_train_np[valid_mask]
        y_behavior_clean = y_series[valid_mask].values.astype(int)
        
        # Skip if there are not enough positive samples to train a meaningful model
        if np.sum(y_behavior_clean) < 20:
            print(f"    Not enough positive samples ({np.sum(y_behavior_clean)}). Skipping behavior.")
            continue
            
        # Define the ensemble
        ensemble = []
        
        # Model 1: LightGBM
        lgbm = lgb.LGBMClassifier(objective='binary', metric='logloss', n_estimators=300,
                                  learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, colsample_bytree=0.8)
        
        pipeline_lgbm = make_pipeline(
            SimpleImputer(strategy='mean'),
            StratifiedDownsamplingClassifier(estimator=lgbm, n_samples=150000)
        )
        # Fit on the cleaned, filtered data
        pipeline_lgbm.fit(X_train_behavior, y_behavior_clean)
        ensemble.append(pipeline_lgbm)

        # Model 2: XGBoost (if available)
        if XGBOOST_AVAILABLE:
            xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=250,
                                learning_rate=0.05, max_depth=5, use_label_encoder=False, 
                                random_state=42, n_jobs=-1, tree_method='hist')
            
            pipeline_xgb = make_pipeline(
                SimpleImputer(strategy='mean'),
                StratifiedDownsamplingClassifier(estimator=xgb, n_samples=150000)
            )
            # Fit on the cleaned, filtered data
            pipeline_xgb.fit(X_train_behavior, y_behavior_clean)
            ensemble.append(pipeline_xgb)

        config_models[behavior] = ensemble
        
    models_dict[config_str] = {
        'models': config_models,
        'feature_columns': X_train.columns
    }
    print(f"  Finished training for this configuration.")
    del X_train, y_train, X_train_np, X_train_behavior, y_behavior_clean
    gc.collect()

print("\n--- Model Training Complete ---")


def post_process_predictions(probs_df, metadata_df, threshold=0.5, min_duration=4):
    """
    Converts frame-wise probabilities into a submission-ready dataframe of events.
    """
    submission_events = []
    
    if probs_df.empty:
        return pd.DataFrame()
        
 
    smoothed_probs = probs_df.rolling(window=5, min_periods=1, center=True).mean()
    
    for behavior in smoothed_probs.columns:
        
        predictions = (smoothed_probs[behavior] > threshold).astype(int)
        
        
        diffs = predictions.diff()
        start_frames = metadata_df.loc[diffs == 1, 'video_frame']
        stop_frames = metadata_df.loc[diffs == -1, 'video_frame']
        
        
        if len(start_frames) > len(stop_frames):
            stop_frames = stop_frames.tolist() + [metadata_df['video_frame'].max() + 1]
            stop_frames = pd.Series(stop_frames, index=start_frames.index[:len(stop_frames)])
        
        if len(stop_frames) > len(start_frames):
            start_frames = [metadata_df['video_frame'].min()] + start_frames.tolist()
            start_frames = pd.Series(start_frames, index=stop_frames.index[:len(start_frames)])

        
        for start, stop in zip(start_frames, stop_frames):
            if stop - start >= min_duration:
                event = {
                    'video_id': metadata_df['video_id'].iloc[0],
                    'agent_id': metadata_df['agent_id'].iloc[0],
                    'target_id': metadata_df['target_id'].iloc[0],
                    'action': behavior,
                    'start_frame': start,
                    'stop_frame': stop
                }
                submission_events.append(event)
                
    return pd.DataFrame(submission_events)


all_submissions = []

print("--- Starting Prediction on Test Set ---")

for config_str in models_dict.keys():
    
    body_parts = json.loads(config_str)
    if len(body_parts) > 10:
        body_parts = [bp for bp in body_parts if bp not in DROP_BODY_PARTS]
    
    print(f"\nPredicting for configuration with {len(body_parts)} body parts...")
    
    config_test_df = test_df[test_df['body_parts_tracked'] == config_str]
    
    if config_test_df.empty:
        continue
        
    trained_models = models_dict[config_str]['models']
    feature_cols = models_dict[config_str]['feature_columns']
    
    test_generator = generate_mouse_data(config_test_df, mode='test')
    
    for _, tracking_data, metadata, actions in tqdm(test_generator, total=len(config_test_df)*6):
        
     
        features = create_pair_features(tracking_data, body_parts)
 
        features = features.reindex(columns=feature_cols).fillna(0)
        
        X_test_np = features.values
        
        probs_df = pd.DataFrame(index=features.index)
        
        for behavior, ensemble in trained_models.items():
            if behavior in actions: 
                behavior_probs = [model.predict_proba(X_test_np)[:, 1] for model in ensemble]
                probs_df[behavior] = np.mean(behavior_probs, axis=0)

        sub_part = post_process_predictions(probs_df, metadata)
        if not sub_part.empty:
            all_submissions.append(sub_part)

print("\n--- Prediction Complete ---")


if all_submissions:
    submission_df = pd.concat(all_submissions, ignore_index=True)
    submission_df = submission_df.sort_values(by=['video_id', 'agent_id', 'target_id', 'start_frame'])
    submission_df = submission_df.drop_duplicates(subset=['video_id', 'agent_id', 'target_id', 'action', 'start_frame'])
else:
    submission_df = pd.DataFrame(columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])

submission_df.to_csv('submission.csv', index_label='row_id')
print(f"\nSubmission file created with {len(submission_df)} events.")
display(submission_df.head())

