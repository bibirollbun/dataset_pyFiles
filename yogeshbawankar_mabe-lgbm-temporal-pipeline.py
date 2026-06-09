import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import itertools
import warnings
import json
import os
import lightgbm
from collections import defaultdict
import polars as pl
import gc
import joblib
from pathlib import Path

# Scikit-learn imports
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.model_selection import cross_val_predict, GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.exceptions import UndefinedMetricWarning

# --- CONFIGURATION ---
validate_or_submit = 'submit' 
verbose = True
SEED = 42

# --- MODEL AND FEATURE CONSTANTS ---
N_SAMPLES_SUBSET = 100000
N_ESTIMATORS_LGBM = 500
MIN_BEHAVIOR_SAMPLES = 100
MIN_GROUPS_FOR_CV = 5
MIN_SEGMENT_LENGTH = 4
MERGE_MAX_GAP = 2

# --- INITIALIZATION ---
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')

Path('./models').mkdir(exist_ok=True)
Path('./results').mkdir(exist_ok=True)


class TrainOnSubsetClassifier(ClassifierMixin, BaseEstimator):
    """
    Fit estimator to a stratified subset of the training data to speed up training
    while preserving class distribution.
    """
    def __init__(self, estimator, n_samples):
        self.estimator = estimator
        self.n_samples = n_samples

    def fit(self, X, y):
        if len(X) <= self.n_samples:
            self.estimator.fit(X, y)
        else:
            try:
                X_sample, _, y_sample, _ = train_test_split(
                    X, y,
                    train_size=self.n_samples,
                    stratify=y,
                    random_state=SEED,
                    shuffle=True
                )
                self.estimator.fit(X_sample, y_sample)
            except (ValueError, KeyError): # Fallback for non-stratifiable cases
                indices = np.random.choice(len(X), self.n_samples, replace=False)
                self.estimator.fit(X.iloc[indices], y.iloc[indices])

        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, X):
        if len(self.classes_) == 1:
            prob_array = np.zeros((len(X), 2))
            class_idx = self.classes_[0]
            prob_array[:, class_idx] = 1.0
            return prob_array
        return self.estimator.predict_proba(X)

def find_best_threshold(y_true, y_pred_proba):
    """Find the best F1 threshold from precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    thresholds = np.append(thresholds, 1.0)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-6)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

def predict_and_segment(pred_df, meta, thresholds):
    """Convert model probabilities to action segments using per-action thresholds."""
    submission_parts = []
    for action, threshold in thresholds.items():
        if action not in pred_df.columns:
            continue
        
        preds = (pred_df[action] >= threshold).astype(int)
        changes_mask = (preds != preds.shift(1))
        segment_boundaries = meta[changes_mask]
        action_values = preds[changes_mask]

        starts = segment_boundaries[action_values == 1]
        ends = segment_boundaries[action_values == 0]
        
        segs = []
        for _, start_row in starts.iterrows():
            end_frame_candidates = ends[ends.index > start_row.name]
            stop_frame = end_frame_candidates.index[0] if not end_frame_candidates.empty else meta.video_frame.max() + 1
            
            segs.append({
                'video_id': start_row['video_id'], 'agent_id': start_row['agent_id'],
                'target_id': start_row['target_id'], 'action': action,
                'start_frame': start_row.name, 'stop_frame': stop_frame,
            })
        
        if segs:
            submission_parts.append(pd.DataFrame(segs))

    if not submission_parts:
        return pd.DataFrame()

    full_submission = pd.concat(submission_parts).sort_values('start_frame').reset_index(drop=True)
    return refine_segments(full_submission)

def refine_segments(df):
    """Apply post-processing heuristics to clean up segments."""
    if df.empty:
        return df

    df['duration'] = df['stop_frame'] - df['start_frame']
    df = df[df['duration'] >= MIN_SEGMENT_LENGTH]
    
    df = df.sort_values(['video_id', 'agent_id', 'target_id', 'action', 'start_frame'])
    
    merged_rows = []
    for _, group in df.groupby(['video_id', 'agent_id', 'target_id', 'action']):
        if len(group) == 1:
            merged_rows.append(group)
            continue
        
        current_row = group.iloc[0].copy()
        for i in range(1, len(group)):
            next_row = group.iloc[i]
            if (next_row['start_frame'] - current_row['stop_frame']) <= MERGE_MAX_GAP:
                current_row['stop_frame'] = next_row['stop_frame']
            else:
                merged_rows.append(pd.DataFrame([current_row]))
                current_row = next_row.copy()
        merged_rows.append(pd.DataFrame([current_row]))
        
    if not merged_rows:
        return pd.DataFrame(columns=df.columns)

    return pd.concat(merged_rows).drop(columns=['duration']).reset_index(drop=True)


train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
test = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')

# --- Standardize Body Part Schemas ---
CANONICAL_SCHEMA_1 = ['body_center', 'ear_left', 'ear_right', 'lateral_left', 'lateral_right', 'neck', 'nose', 'tail_base', 'tail_midpoint', 'tail_tip']
CANONICAL_SCHEMA_2 = ['body_center', 'ear_left', 'ear_right', 'nose', 'tail_base']
BODY_PARTS_WHITELIST = [
    'body_center', 'ear_left', 'ear_right', 'hip_left', 'hip_right',
    'lateral_left', 'lateral_right', 'neck', 'nose', 'tail_base',
    'tail_midpoint', 'tail_tip'
]

def map_to_canonical_schema(parts_str):
    """Maps a list of body parts to the best-matching canonical schema."""
    parts = json.loads(parts_str)
    if all(p in parts for p in CANONICAL_SCHEMA_1):
        return json.dumps(CANONICAL_SCHEMA_1)
    if all(p in parts for p in CANONICAL_SCHEMA_2):
        return json.dumps(CANONICAL_SCHEMA_2)
    return parts_str

train['canonical_schema'] = train['body_parts_tracked'].apply(map_to_canonical_schema)
test['canonical_schema'] = test['body_parts_tracked'].apply(map_to_canonical_schema)

body_parts_schemas_to_process = train['canonical_schema'].unique()


def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    assert traintest in ['train', 'test']
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    
    for _, row in dataset.iterrows():
        lab_id, video_id = row.lab_id, row.video_id
        if lab_id.startswith('MABe22'): continue
        if not isinstance(row.behaviors_labeled, str):
            if verbose: print(f'No labeled behaviors for {lab_id}/{video_id}, skipping.')
            continue

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        try:
            vid = pd.read_parquet(path)
        except FileNotFoundError:
            if verbose: print(f"Tracking file not found: {path}")
            continue

        # Filter to whitelisted body parts for consistency
        vid = vid[vid.bodypart.isin(BODY_PARTS_WHITELIST)]
        
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        del vid; gc.collect()

        # Proactively interpolate missing coordinates
        pvid = pvid.interpolate(method='linear', limit_direction='both', axis=0)
        
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        pvid /= row.pix_per_cm_approx

        vid_behaviors = pd.DataFrame([b.split(',') for b in sorted(list(set(json.loads(row.behaviors_labeled))))], columns=['agent', 'target', 'action'])
        
        if traintest == 'train':
            try:
                annot = pd.read_parquet(path.replace(f'{traintest}_tracking', f'{traintest}_annotation'))
            except FileNotFoundError:
                continue

        # Generate single mouse data
        if generate_single:
            vid_behaviors_subset = vid_behaviors.query("target == 'self'")
            for mouse_id_str in np.unique(vid_behaviors_subset.agent):
                try:
                    mouse_id = int(mouse_id_str[-1])
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("agent == @mouse_id_str").action)
                    single_mouse = pvid.loc[:, mouse_id]
                    single_mouse_meta = pd.DataFrame({'video_id': video_id, 'agent_id': mouse_id_str, 'target_id': 'self', 'video_frame': single_mouse.index})
                    
                    if traintest == 'train':
                        single_mouse_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=single_mouse.index)
                        annot_subset = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for _, annot_row in annot_subset.iterrows():
                            single_mouse_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        # This 'yield' is crucial
                        yield 'single', single_mouse, single_mouse_meta, single_mouse_label
                    else:
                        if verbose: print(f'- test single {video_id} {mouse_id}')
                        # This 'yield' is crucial
                        yield 'single', single_mouse, single_mouse_meta, vid_agent_actions
                except KeyError:
                    pass
        
        # Generate mouse pair data
        if generate_pair:
            vid_behaviors_subset = vid_behaviors.query("target != 'self'")
            if len(vid_behaviors_subset) > 0:
                for agent, target in itertools.permutations(np.unique(pvid.columns.get_level_values('mouse_id')), 2):
                    agent_str, target_str = f"mouse{agent}", f"mouse{target}"
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("(agent == @agent_str) & (target == @target_str)").action)
                    if len(vid_agent_actions) == 0: continue
                    
                    mouse_pair = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                    mouse_pair_meta = pd.DataFrame({'video_id': video_id, 'agent_id': agent_str, 'target_id': target_str, 'video_frame': mouse_pair.index})
                    
                    if traintest == 'train':
                        mouse_pair_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=mouse_pair.index)
                        annot_subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                        for _, annot_row in annot_subset.iterrows():
                             mouse_pair_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        # This 'yield' is crucial
                        yield 'pair', mouse_pair, mouse_pair_meta, mouse_pair_label
                    else:
                        if verbose: print(f'- test pair {video_id} {agent} {target}')
                        # This 'yield' is crucial
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions

# --- Configurable Temporal Feature Parameters ---
TIME_WINDOWS = [5, 15, 30, 60]
LAG_OFFSETS = [10, 20, 40]
CONTEXT_OFFSETS = [-20, -10, 10, 20]

def transform_single(single_mouse, body_parts_tracked):
    """Generates features for a single mouse's tracking data."""
    available_parts = set(single_mouse.columns.get_level_values(0))
    X = pd.DataFrame(index=single_mouse.index)
    
    # Pairwise distances
    for p1, p2 in itertools.combinations(body_parts_tracked, 2):
        if p1 in available_parts and p2 in available_parts:
            X[f"{p1}+{p2}"] = np.sum(np.square(single_mouse[p1].values - single_mouse[p2].values), axis=1)

    # Long-range temporal features (guarded)
    if 'body_center' in available_parts:
        center_x, center_y = single_mouse['body_center']['x'], single_mouse['body_center']['y']
        for window in TIME_WINDOWS:
            # Calculate activity level (movement variance) over the window
            activity_sq = center_x.diff().rolling(window).std()**2 + center_y.diff().rolling(window).std()**2
            X[f'activity_level_{window}'] = np.sqrt(activity_sq)
    
    # Drop columns that are mostly empty to save memory
    X = X.dropna(axis=1, thresh=len(X) * 0.5)
    return X

def transform_pair(mouse_pair, body_parts_tracked):
    """Generates features for a pair of mice's tracking data."""
    available_A = set(mouse_pair['A'].columns.get_level_values(0))
    available_B = set(mouse_pair['B'].columns.get_level_values(0))
    X = pd.DataFrame(index=mouse_pair.index)

    # Inter-mouse distances
    for p1 in body_parts_tracked:
        for p2 in body_parts_tracked:
            if p1 in available_A and p2 in available_B:
                X[f"12+{p1}+{p2}"] = np.sum(np.square(mouse_pair['A'][p1].values - mouse_pair['B'][p2].values), axis=1)
    
    # Social zone indicators and temporal stats (guarded)
    if 'body_center' in available_A and 'body_center' in available_B:
        center_dist = np.linalg.norm(mouse_pair['A']['body_center'].values - mouse_pair['B']['body_center'].values, axis=1)
        
        for window in TIME_WINDOWS:
            X[f'dist_mean_{window}'] = pd.Series(center_dist, index=X.index).rolling(window, min_periods=1, center=True).mean()
            X[f'dist_std_{window}'] = pd.Series(center_dist, index=X.index).rolling(window, min_periods=1, center=True).std()

    # Drop columns that are mostly empty
    X = X.dropna(axis=1, thresh=len(X) * 0.5)
    return X


def cross_validate_and_train(schema_str, switch, X_tr, label, meta):
    """
    Performs GroupKFold CV, tunes thresholds, evaluates, and trains final models.
    """
    tuned_thresholds = {}
    print(f"  Starting cross-validation for {len(label.columns)} actions...")
    
    for action in tqdm(label.columns, desc="  Actions"):
        action_mask = label[action].notna()
        y_action = label.loc[action_mask, action].astype(int)
        
        if y_action.sum() < MIN_BEHAVIOR_SAMPLES:
            continue
            
        X_action = X_tr[action_mask]
        groups_action = meta.loc[action_mask, 'video_id']
        
        if len(groups_action.unique()) < MIN_GROUPS_FOR_CV:
            continue

        binary_classifier = make_pipeline(
            SimpleImputer(strategy='mean'),
            StandardScaler(),
            lightgbm.LGBMClassifier(
                n_estimators=N_ESTIMATORS_LGBM,
                class_weight='balanced',
                random_state=SEED,
                n_jobs=-1,
                verbose=-1
            )
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oof_action_preds = cross_val_predict(
                binary_classifier, X_action, y_action, 
                groups=groups_action, cv=GroupKFold(n_splits=MIN_GROUPS_FOR_CV), 
                method='predict_proba'
            )[:, 1]
        
        best_thresh = find_best_threshold(y_action, oof_action_preds)
        tuned_thresholds[action] = best_thresh
        f1 = f1_score(y_action, oof_action_preds >= best_thresh)
        print(f"  OOF F1 for '{action}': {f1:.4f} | Thresh: {best_thresh:.3f}")
        
        print(f"  Training final model for '{action}'...")
        final_model = clone(binary_classifier).fit(X_action, y_action)
        model_path = f'./models/{schema_str.replace(" ", "")}_{switch}_{action}.joblib'
        joblib.dump(final_model, model_path)
    
    joblib.dump(tuned_thresholds, f'./models/{schema_str.replace(" ", "")}_{switch}_thresholds.joblib')

def submit(schema_str, switch, test_dataset):
    """
    Loads trained models and thresholds to generate predictions on the test set.
    """
    submission_list = []
    
    try:
        thresholds = joblib.load(f'./models/{schema_str.replace(" ", "")}_{switch}_thresholds.joblib')
        models = {
            action: joblib.load(f'./models/{schema_str.replace(" ", "")}_{switch}_{action}.joblib')
            for action in thresholds.keys()
        }
    except FileNotFoundError:
        return pd.DataFrame()

    body_parts_tracked = json.loads(schema_str)
    
    generator = generate_mouse_data(
        test_dataset.query("canonical_schema == @schema_str"), 'test',
        generate_single=(switch == 'single'), 
        generate_pair=(switch == 'pair')
    )
    
    for switch_te, data_te, meta_te, actions_te in generator:
        if verbose: print(f"  Predicting for video {meta_te.video_id.iloc[0]}...")
        
        X_te = transform_single(data_te, body_parts_tracked) if switch_te == 'single' else transform_pair(data_te, body_parts_tracked)
        
        pred_df = pd.DataFrame(index=meta_te.index)
        for action, model in models.items():
            if action in actions_te:
                # --- FIX APPLIED HERE ---
                # Use the pipeline's `feature_names_in_` attribute to get the correct
                # feature names that the first step (imputer) was trained on.
                model_features = model.feature_names_in_
                X_te_aligned = X_te.reindex(columns=model_features, fill_value=0)
                pred_df[action] = model.predict_proba(X_te_aligned)[:, 1]

        if not pred_df.empty:
            submission_part = predict_and_segment(pred_df, meta_te, thresholds)
            submission_list.append(submission_part)
            
    return pd.concat(submission_list) if submission_list else pd.DataFrame()


submission_list = []

for i, schema_str in enumerate(body_parts_schemas_to_process):
    body_parts_tracked = json.loads(schema_str)
    print(f"\n--- Processing Schema {i+1}/{len(body_parts_schemas_to_process)}: {len(body_parts_tracked)} parts ---")
    
    train_subset = train[train.canonical_schema == schema_str]
    if len(train_subset) == 0:
        continue

    # Generate data for the current schema
    data_lists = {'single': [], 'pair': []}
    meta_lists = {'single': [], 'pair': []}
    label_lists = {'single': [], 'pair': []}
    for switch, data, meta, label in generate_mouse_data(train_subset, 'train'):
        data_lists[switch].append(data)
        meta_lists[switch].append(meta)
        label_lists[switch].append(label)

    # Process single-mouse and pair-mouse data separately
    for switch in ['single', 'pair']:
        if not data_lists[switch]:
            continue
            
        print(f"\nProcessing '{switch}' mouse data...")
        
        full_data = pd.concat(data_lists[switch])
        full_meta = pd.concat(meta_lists[switch])
        full_label = pd.concat(label_lists[switch])
        del data_lists[switch], meta_lists[switch], label_lists[switch]; gc.collect()

        print("  Transforming features...")
        X_tr = transform_single(full_data, body_parts_tracked) if switch == 'single' else transform_pair(full_data, body_parts_tracked)
        del full_data; gc.collect()
        
        print(f"  Feature matrix shape: {X_tr.shape}")

        # Align indices
        X_tr.reset_index(drop=True, inplace=True)
        full_label.reset_index(drop=True, inplace=True)
        full_meta.reset_index(drop=True, inplace=True)
      
        print("  Training and saving models for submission...")
        cross_validate_and_train(schema_str, switch, X_tr, full_label, full_meta)
        
        print("  Running inference on test set...")
        submission_part = submit(schema_str, switch, test)
        submission_list.append(submission_part)

        del X_tr, full_label, full_meta; gc.collect()

# Create final submission file
final_submission = pd.concat(submission_list).reset_index(drop=True)
final_submission.to_csv('submission.csv', index_label='row_id')
print("\nSubmission file created successfully!")




