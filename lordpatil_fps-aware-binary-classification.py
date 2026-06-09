# In[1]:
# ===================================================================
# Section 1: Setup, Configuration, and Helper Classes
# ===================================================================
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import itertools
import warnings
import json
import os
import random
import gc
from collections import defaultdict

# Import Models
import lightgbm as lgb
from xgboost import XGBClassifier

# Sklearn utilities
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration Class ---
class CFG:
    """
    Configuration class for all hyperparameters and settings.
    Includes a DEBUG toggle for fast pipeline testing.
    """
    # Set DEBUG to True to run on a 5% sample of the data for quick testing.
    # Set to False for a full training run.
    DEBUG = False
    
    # Paths
    BASE_PATH = "/kaggle/input/MABe-mouse-behavior-detection"
    
    # Training parameters
    N_SAMPLES_SINGLE = 2_000_000  # Number of samples for training single-mouse models
    N_SAMPLES_PAIR = 900_000   # Number of samples for training mouse-pair models
    
    # Prediction parameters
    PREDICTION_THRESHOLD = 0.27
    # Action-specific thresholds can improve F1 score.
    # These are heuristics derived from public notebooks.
    ACTION_THRESHOLDS = {
        "default": 0.27,
        "single_default": 0.27,
        "pair_default": 0.27,
        "single": {"rear": 0.30},
    }

    # Seed for reproducibility
    SEED = 42

# --- Seed for Reproducibility ---
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

seed_everything(CFG.SEED)

# --- Helper Classifier for Memory Management ---
class StratifiedSubsetClassifier(ClassifierMixin, BaseEstimator):
    """
    A wrapper for scikit-learn estimators that fits the model on a stratified
    random subset of the training data. This is crucial for handling datasets
    that don't fit into memory.
    """
    def __init__(self, estimator, n_samples=None):
        self.estimator = estimator
        self.n_samples = n_samples

    def _to_numpy(self, X):
        if hasattr(X, 'to_numpy'):
            return X.to_numpy(np.float32, copy=False)
        return np.asarray(X, dtype=np.float32)

    def fit(self, X, y):
        Xn = self._to_numpy(X)
        y = np.asarray(y).ravel()

        if self.n_samples is None or len(Xn) <= int(self.n_samples):
            self.estimator.fit(Xn, y)
        else:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=int(self.n_samples), random_state=CFG.SEED)
            try:
                # Use stratified sampling to maintain class distribution
                idx, _ = next(sss.split(np.zeros_like(y), y))
                self.estimator.fit(Xn[idx], y[idx])
            except Exception:
                # Fallback to simple downsampling if stratification fails (e.g., too few samples of a class)
                step = max(len(Xn) // int(self.n_samples), 1)
                self.estimator.fit(Xn[::step], y[::step])
        
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, X):
        Xn = self._to_numpy(X)
        if len(self.classes_) == 1:
            # Handle cases where only one class was present in the training subset
            probs = np.zeros((len(Xn), 2), dtype=np.float32)
            if int(self.classes_[0]) < 2:
                 probs[:, int(self.classes_[0])] = 1.0
            return probs
        
        P = self.estimator.predict_proba(Xn)
        
        # Ensure output is always 2 columns for binary classification
        if P.shape[1] == 1:
            P1 = P[:, 0].astype(np.float32)
            return np.column_stack([1.0 - P1, P1])
        return P

# --- Body Parts to Exclude ---
# These are often redundant or noisy, derived from community insights.
DROP_BODY_PARTS = [
    'headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright',
    'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright',
    'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint'
]


# In[2]:
# ===================================================================
# Section 2: Data Loading and Preparation (with DEBUG mode fix)
# ===================================================================
print("Loading data...")
train_df = pd.read_csv(os.path.join(CFG.BASE_PATH, 'train.csv'))
test_df = pd.read_csv(os.path.join(CFG.BASE_PATH, 'test.csv'))

print("Preparing training data...")
# Filter out the MABe22 datasets as they are mostly unannotated for behavior.
train_df_filtered = train_df[~train_df['lab_id'].str.startswith('MABe22_')].copy()

# In DEBUG mode, use a small, representative sample of the data.
if CFG.DEBUG:
    print("DEBUG mode enabled: Using a small sample of the training data.")
    
    # FIX: Ensure the debug sample includes data for the public test set configuration.
    test_config = test_df['body_parts_tracked'].iloc[0]
    
    matching_train_sample = train_df_filtered[train_df_filtered['body_parts_tracked'] == test_config]
    if not matching_train_sample.empty:
        matching_train_sample = matching_train_sample.sample(n=1, random_state=CFG.SEED)
    
    other_train_df = train_df_filtered[train_df_filtered['body_parts_tracked'] != test_config]
    sample_size = max(1, int(len(train_df_filtered) * 0.05) - len(matching_train_sample))
    
    other_sample = other_train_df.sample(n=min(sample_size, len(other_train_df)), random_state=CFG.SEED)
    
    train_df_filtered = pd.concat([matching_train_sample, other_sample]).reset_index(drop=True)

# Get a unique list of all body part configurations in the dataset.
body_parts_tracked_list = list(np.unique(train_df['body_parts_tracked']))

print(f"Training on {len(train_df_filtered)} videos.")
print(f"Found {len(body_parts_tracked_list)} unique body part configurations.")


# In[3]:
# ===================================================================
# Section 3: FPS-Aware Feature Engineering
# ===================================================================

# --- FPS Scaling Helper Functions ---
def _scale(n_frames_at_30fps, fps, ref=30.0):
    """Scales a frame count from a 30 FPS reference to the actual video FPS."""
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))

def _fps_from_meta(meta_df):
    """Safely extracts FPS from a metadata dataframe."""
    return float(meta_df['frames_per_second'].iloc[0])

# --- Feature Engineering Functions ---

def transform_single(single_mouse, body_parts_tracked, fps):
    """
    Generates features for a single mouse. All temporal features are scaled by FPS.
    """
    available_body_parts = single_mouse.columns.get_level_values(0)
    X = pd.DataFrame(index=single_mouse.index)

    # Base distance features (invariant to rotation/translation)
    for p1, p2 in itertools.combinations(body_parts_tracked, 2):
        if p1 in available_body_parts and p2 in available_body_parts:
            X[f"{p1}+{p2}"] = np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)

    # Kinematic features (speed, acceleration)
    if 'body_center' in available_body_parts:
        center_x = single_mouse['body_center']['x']
        center_y = single_mouse['body_center']['y']
        
        speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
        acceleration = speed.diff() * float(fps)
        
        X['speed'] = speed
        X['acceleration'] = acceleration
        
        for w in [15, 30, 60]: # ~0.5s, 1s, 2s windows at 30fps
            ws = _scale(w, fps)
            X[f'speed_mean_{w}'] = speed.rolling(ws, min_periods=1, center=True).mean()
            X[f'speed_std_{w}'] = speed.rolling(ws, min_periods=1, center=True).std()
            
        for span in [30, 90, 180]:
            s = _scale(span, fps)
            X[f'speed_ema_{span}'] = speed.ewm(span=s, adjust=False).mean()

    # Geometric features
    if all(p in available_body_parts for p in ['nose', 'tail_base', 'ear_left', 'ear_right']):
        X['elongation'] = X.get(f'nose+tail_base', np.nan) / (X.get(f'ear_left+ear_right', np.nan) + 1e-6)

    if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
        v1 = single_mouse['nose'] - single_mouse['body_center']
        v2 = single_mouse['tail_base'] - single_mouse['body_center']
        dot = (v1['x'] * v2['x'] + v1['y'] * v2['y'])
        norm = np.sqrt(v1['x']**2 + v1['y']**2) * np.sqrt(v2['x']**2 + v2['y']**2)
        X['body_angle_cos'] = dot / (norm + 1e-6)

    return X.astype(np.float32, copy=False)

def transform_pair(mouse_pair, body_parts_tracked, fps):
    """
    Generates features for a pair of mice (A=agent, B=target).
    """
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)
    X = pd.DataFrame(index=mouse_pair.index)

    # Inter-mouse distances
    for p1 in body_parts_tracked:
        for p2 in body_parts_tracked:
            if p1 in avail_A and p2 in avail_B:
                X[f"12+{p1}+{p2}"] = np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1, skipna=False)

    # Social interaction features
    if 'body_center' in avail_A and 'body_center' in avail_B:
        center_dist = np.sqrt(X.get(f'12+body_center+body_center', np.nan))
        X['center_dist'] = center_dist
        
        for w in [15, 30, 60]:
            ws = _scale(w, fps)
            X[f'center_dist_mean_{w}'] = center_dist.rolling(ws, min_periods=1, center=True).mean()
            X[f'center_dist_std_{w}'] = center_dist.rolling(ws, min_periods=1, center=True).std()
            
        X['approach_speed'] = -center_dist.diff() * float(fps)

    # Relative orientation
    if all(p in avail_A for p in ['nose', 'tail_base']) and all(p in avail_B for p in ['nose', 'tail_base']):
        dir_A = mouse_pair['A']['nose'] - mouse_pair['A']['tail_base']
        dir_B = mouse_pair['B']['nose'] - mouse_pair['B']['tail_base']
        dot = (dir_A['x'] * dir_B['x'] + dir_A['y'] * dir_B['y'])
        norm = (np.sqrt(dir_A['x']**2 + dir_A['y']**2) * np.sqrt(dir_B['x']**2 + dir_B['y']**2))
        X['relative_orientation_cos'] = dot / (norm + 1e-6)

    return X.astype(np.float32, copy=False)


# In[4]:
# ===================================================================
# Section 4: Modeling and Prediction Pipeline (GPU-Enabled)
# ===================================================================

# --- Check for GPU availability ---
GPU_AVAILABLE = False
try:
    import torch
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        print("GPU is available. Models will be trained on GPU.")
    else:
        print("GPU not available. Models will be trained on CPU.")
except ImportError:
    print("PyTorch not found. Models will be trained on CPU.")


def generate_mouse_data(dataset, traintest, generate_single=True, generate_pair=True):
    """
    Generator function to process video files one by one, yielding feature-ready dataframes.
    This is the core of the memory-efficient pipeline.
    """
    traintest_directory = os.path.join(CFG.BASE_PATH, f"{traintest}_tracking")
    
    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Generating {traintest} data"):
        lab_id, video_id = row.lab_id, row.video_id
        
        if not isinstance(row.behaviors_labeled, str): continue

        path = os.path.join(traintest_directory, lab_id, f"{video_id}.parquet")
        if not os.path.exists(path): continue
            
        vid = pd.read_parquet(path)
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@DROP_BODY_PARTS)")
        
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        del vid; gc.collect()
        
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T / row.pix_per_cm_approx

        vid_behaviors = json.loads(row.behaviors_labeled)
        vid_behaviors = [b.replace("'", "").split(',') for b in sorted(list(set(vid_behaviors)))]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])

        if traintest == 'train':
            annot_path = path.replace('train_tracking', 'train_annotation')
            if not os.path.exists(annot_path): continue
            annot = pd.read_parquet(annot_path)

        # Yield single-mouse data
        if generate_single:
            subset = vid_behaviors.query("target == 'self'")
            for mouse_id_str in np.unique(subset.agent):
                try:
                    mouse_id = int(mouse_id_str[-1])
                    actions = np.unique(subset.query("agent == @mouse_id_str").action)
                    data = pvid.loc[:, mouse_id]
                    meta = pd.DataFrame({
                        'video_id': video_id, 'agent_id': mouse_id_str, 'target_id': 'self',
                        'video_frame': data.index, 'frames_per_second': row.frames_per_second
                    })
                    if traintest == 'train':
                        label = pd.DataFrame(0.0, columns=actions, index=data.index)
                        annot_sub = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for _, r in annot_sub.iterrows():
                            label.loc[r['start_frame']:r['stop_frame'], r.action] = 1.0
                        yield 'single', data, meta, label
                    else:
                        yield 'single', data, meta, actions
                except (KeyError, IndexError): continue

        # Yield mouse-pair data
        if generate_pair:
            subset = vid_behaviors.query("target != 'self'")
            if not subset.empty:
                for agent, target in itertools.permutations(np.unique(pvid.columns.get_level_values('mouse_id')), 2):
                    agent_str, target_str = f"mouse{agent}", f"mouse{target}"
                    actions = np.unique(subset.query("(agent == @agent_str) & (target == @target_str)").action)
                    if not actions.any(): continue
                    try:
                        data = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                        meta = pd.DataFrame({
                            'video_id': video_id, 'agent_id': agent_str, 'target_id': target_str,
                            'video_frame': data.index, 'frames_per_second': row.frames_per_second
                        })
                        if traintest == 'train':
                            label = pd.DataFrame(0.0, columns=actions, index=data.index)
                            annot_sub = annot.query("(agent_id == @agent) & (target_id == @target)")
                            for _, r in annot_sub.iterrows():
                                label.loc[r['start_frame']:r['stop_frame'], r.action] = 1.0
                            yield 'pair', data, meta, label
                        else:
                            yield 'pair', data, meta, actions
                    except (KeyError, IndexError): continue

def predict_multiclass_adaptive(pred, meta, thresholds):
    """FIXED: Correctly converts frame-wise probabilities into start/stop segments."""
    pred_smoothed = pred.rolling(window=5, min_periods=1, center=True).mean()
    
    ama = np.argmax(pred_smoothed.values, axis=1)
    max_probs = pred_smoothed.max(axis=1).values
    
    threshold_array = np.array([thresholds.get(col, CFG.PREDICTION_THRESHOLD) for col in pred.columns])
    action_thresholds = threshold_array[ama]

    ama = np.where(max_probs >= action_thresholds, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame)

    changes = (ama != ama.shift(1))
    change_indices = meta.loc[changes, 'video_frame'].tolist()
    actions_at_changes = ama[changes].tolist()
    
    if not change_indices or change_indices[-1] != meta['video_frame'].max():
         change_indices.append(meta['video_frame'].max() + 1)
         actions_at_changes.append(-1)

    starts, stops, action_indices = [], [], []
    for i in range(len(actions_at_changes) - 1):
        if actions_at_changes[i] >= 0:
            starts.append(change_indices[i])
            stops.append(change_indices[i+1])
            action_indices.append(actions_at_changes[i])
            
    if not starts:
        return pd.DataFrame()

    submission_part = pd.DataFrame({'action': pred.columns[action_indices], 'start_frame': starts, 'stop_frame': stops})
    
    meta_deduped = meta.drop_duplicates(subset=['video_id', 'agent_id', 'target_id'])
    submission_part['video_id'] = meta_deduped['video_id'].iloc[0]
    submission_part['agent_id'] = meta_deduped['agent_id'].iloc[0]
    submission_part['target_id'] = meta_deduped['target_id'].iloc[0]

    submission_part = submission_part[submission_part['stop_frame'] - submission_part['start_frame'] >= 3]
    return submission_part[['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']]

def submit_ensemble(body_parts_tracked_str, switch_tr, X_tr, label, meta, n_samples, train_feature_cols):
    global submission_list
    
    # --- GPU-Aware Model Ensemble Definition ---
    models = []
    lgbm_params = {'random_state': CFG.SEED, 'n_jobs': -1, 'force_col_wise': True}
    xgb_params = {'random_state': CFG.SEED, 'n_jobs': -1}

    if GPU_AVAILABLE:
        print("  Configuring models for GPU...")
        lgbm_params['device_type'] = 'gpu'
        xgb_params['tree_method'] = 'gpu_hist'
    else:
        print("  Configuring models for CPU...")
        xgb_params['tree_method'] = 'hist'

    models.append(make_pipeline(SimpleImputer(), StratifiedSubsetClassifier(
        lgb.LGBMClassifier(n_estimators=200, learning_rate=0.07, **lgbm_params), n_samples)))
    
    models.append(make_pipeline(SimpleImputer(), StratifiedSubsetClassifier(
        XGBClassifier(n_estimators=180, learning_rate=0.08, max_depth=6, **xgb_params), n_samples)))

    # --- Train one ensemble per action ---
    model_list = []
    for action in label.columns:
        action_mask = ~label[action].isna()
        y_action = label.loc[action_mask, action].values.astype(int)
        
        if np.sum(y_action) >= 5 and len(y_action) - np.sum(y_action) >= 5:
            trained_models = [clone(m).fit(X_tr[action_mask], y_action) for m in models]
            model_list.append((action, trained_models))

    del X_tr; gc.collect()

    # --- Inference on Test Set ---
    test_subset = test_df[test_df.body_parts_tracked == body_parts_tracked_str]
    if test_subset.empty: return
    
    generator = generate_mouse_data(test_subset, 'test', 
                                    generate_single=(switch_tr == 'single'), 
                                    generate_pair=(switch_tr == 'pair'))

    for switch_te, data_te, meta_te, actions_te in generator:
        try:
            fps = _fps_from_meta(meta_te)
            body_parts_tracked = json.loads(body_parts_tracked_str)
            X_te = transform_single(data_te, body_parts_tracked, fps) if switch_te == 'single' else transform_pair(data_te, body_parts_tracked, fps)
            
            # FIX: Align test features with training features
            X_te = X_te.reindex(columns=train_feature_cols, fill_value=np.nan)
            
            del data_te; gc.collect()

            pred = pd.DataFrame(index=meta_te.video_frame)
            for action, trained_models in model_list:
                if action in actions_te:
                    probs = [m.predict_proba(X_te)[:, 1] for m in trained_models]
                    pred[action] = np.mean(probs, axis=0)
            del X_te; gc.collect()
            
            if not pred.empty:
                submission_part = predict_multiclass_adaptive(pred, meta_te, CFG.ACTION_THRESHOLDS)
                submission_list.append(submission_part)
        except Exception as e:
            print(f"  ERROR during inference: {e}")
            continue


# In[5]:
# ===================================================================
# Section 5: Main Execution Loop and Submission Generation
# ===================================================================

def robustify(submission, dataset):
    """
    Cleans the submission file by removing overlaps and ensuring every video
    has at least one prediction to prevent scoring errors.
    """
    submission = submission[submission.start_frame < submission.stop_frame].copy()

    group_list = []
    for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values('start_frame').reset_index(drop=True)
        if not group.empty:
            last_stop_frame = 0
            for i, row in group.iterrows():
                if row['start_frame'] < last_stop_frame:
                    group.at[i, 'start_frame'] = last_stop_frame
                last_stop_frame = row['stop_frame']
        group_list.append(group)
    
    submission = pd.concat(group_list).reset_index(drop=True)
    submission = submission[submission.start_frame < submission.stop_frame]

    predicted_videos = set(submission['video_id'].unique())
    all_test_videos = set(dataset['video_id'].unique())
    missing_videos = all_test_videos - predicted_videos
    
    if missing_videos:
        print(f"Found {len(missing_videos)} videos with no predictions. Adding placeholders.")
        placeholders = []
        for video_id in missing_videos:
            behaviors_str = dataset[dataset.video_id == video_id]['behaviors_labeled'].iloc[0]
            if isinstance(behaviors_str, str) and behaviors_str != '[]':
                agent, target, action = json.loads(behaviors_str)[0].replace("'", "").split(',')
                placeholders.append({'video_id': video_id, 'agent_id': agent, 'target_id': target,
                                     'action': action, 'start_frame': 0, 'stop_frame': 100})
        if placeholders:
            submission = pd.concat([submission, pd.DataFrame(placeholders)], ignore_index=True)
            
    return submission.reset_index(drop=True)

# --- Main Loop ---
submission_list = []
loop_iterator = range(len(body_parts_tracked_list))

for section in loop_iterator:
    if section == 0: continue
    
    body_parts_tracked_str = body_parts_tracked_list[section]
    body_parts_tracked = json.loads(body_parts_tracked_str)
    print(f"\n--- Processing Section {section}/{len(body_parts_tracked_list)-1}: {len(body_parts_tracked)} body parts ---")

    train_subset = train_df_filtered[train_df_filtered.body_parts_tracked == body_parts_tracked_str]
    if train_subset.empty:
        print("No training data for this configuration.")
        continue

    single_mouse_list, single_label_list, single_meta_list = [], [], []
    pair_list, pair_label_list, pair_meta_list = [], [], []

    data_generator = generate_mouse_data(train_subset, 'train')
    for switch, data, meta, label in data_generator:
        (single_mouse_list if switch == 'single' else pair_list).append(data)
        (single_meta_list if switch == 'single' else pair_meta_list).append(meta)
        (single_label_list if switch == 'single' else pair_label_list).append(label)
    
    if single_mouse_list:
        print(f"Processing {len(single_mouse_list)} single-mouse batches...")
        all_data = pd.concat(single_mouse_list); all_meta = pd.concat(single_meta_list); all_labels = pd.concat(single_label_list)
        del single_mouse_list, single_meta_list, single_label_list; gc.collect()
        
        fps = _fps_from_meta(all_meta)
        X_tr = transform_single(all_data, body_parts_tracked, fps)
        train_feature_cols = X_tr.columns.tolist() # Capture feature names
        
        print(f"  Training on single-mouse data with shape {X_tr.shape}")
        submit_ensemble(body_parts_tracked_str, 'single', X_tr, all_labels, all_meta, CFG.N_SAMPLES_SINGLE, train_feature_cols)
        del X_tr, all_data, all_meta, all_labels; gc.collect()

    if pair_list:
        print(f"Processing {len(pair_list)} mouse-pair batches...")
        all_data = pd.concat(pair_list); all_meta = pd.concat(pair_meta_list); all_labels = pd.concat(pair_label_list)
        del pair_list, pair_meta_list, pair_label_list; gc.collect()
        
        fps = _fps_from_meta(all_meta)
        X_tr = transform_pair(all_data, body_parts_tracked, fps)
        train_feature_cols = X_tr.columns.tolist() # Capture feature names
        
        print(f"  Training on mouse-pair data with shape {X_tr.shape}")
        submit_ensemble(body_parts_tracked_str, 'pair', X_tr, all_labels, all_meta, CFG.N_SAMPLES_PAIR, train_feature_cols)
        del X_tr, all_data, all_meta, all_labels; gc.collect()

# --- Finalize and Save Submission ---
print("\n--- Finalizing Submission ---")
if submission_list:
    submission_df = pd.concat(submission_list, ignore_index=True)
else:
    submission_df = pd.DataFrame([{'video_id': test_df['video_id'].iloc[0], 'agent_id': 'mouse1', 
                                   'target_id': 'self', 'action': 'rear', 'start_frame': 0, 'stop_frame': 100}])

print(f"Generated {len(submission_df)} raw predictions.")

submission_final = robustify(submission_df, test_df)
submission_final.index.name = 'row_id'
submission_final.to_csv('submission.csv')

print(f"Final submission saved to 'submission.csv' with {len(submission_final)} rows.")
print("Done!")




