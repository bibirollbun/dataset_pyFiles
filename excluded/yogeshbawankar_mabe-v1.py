import os

class Config:
    """A central class to hold all configuration parameters."""
    
    # --- Data Paths ---
    BASE_DIR = '/kaggle/input/MABe-mouse-behavior-detection'
    TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')
    TEST_CSV = os.path.join(BASE_DIR, 'test.csv')
    TRAIN_TRACKING_DIR = os.path.join(BASE_DIR, 'train_tracking')
    TEST_TRACKING_DIR = os.path.join(BASE_DIR, 'test_tracking')
    TRAIN_ANNOTATION_DIR = os.path.join(BASE_DIR, 'train_annotation')

    # --- Output ---
    SUBMISSION_FILE = 'submission.csv'

    # --- Model & Training Settings ---
    RANDOM_SEED = 42
    N_SAMPLES_MAX_SINGLE = 100_000
    N_SAMPLES_MAX_PAIR = 80_000
    
    # --- Feature Engineering ---
    # Body parts to exclude from high-dimensional datasets
    DROP_BODY_PARTS = [
        'headpiece_bottombackleft', 'headpiece_bottombackright', 
        'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
        'headpiece_topbackleft', 'headpiece_topbackright', 
        'headpiece_topfrontleft', 'headpiece_topfrontright', 
        'spine_1', 'spine_2', 'tail_middle_1', 
        'tail_middle_2', 'tail_midpoint'
    ]
    
    # --- Post-Processing ---
    DEFAULT_THRESHOLD = 0.27
    MIN_DURATION_FRAMES = 3 # Minimum length of a predicted behavior


import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import itertools
import warnings
import json
import gc
import lightgbm
import polars as pl
from collections import defaultdict
from typing import List, Union, Dict, Tuple

# --- Import models if available ---
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# --- Scikit-learn ---
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

# --- Setup ---
warnings.filterwarnings('ignore')
np.random.seed(Config.RANDOM_SEED)

print(f"XGBoost available: {XGBOOST_AVAILABLE}")
print(f"CatBoost available: {CATBOOST_AVAILABLE}")


def frames_since_event(
    series: pd.Series,
    threshold: float,
    above: bool = True,
    cap: int = 999
) -> np.ndarray:
    """
    Calculates the number of frames since the last event occurred.
    An event is defined as the series value crossing a given threshold.
    
    Args:
        series: The input time series data.
        threshold: The value threshold to define an event.
        above: If True, event is series > threshold. If False, event is series < threshold.
        cap: The maximum value for the counter.

    Returns:
        A NumPy array with the count of frames since the last event.
    """
    s_numpy = series.to_numpy()
    event_mask = s_numpy > threshold if above else s_numpy < threshold
    
    event_indices = np.where(event_mask)[0]
    event_indices = np.insert(event_indices, 0, -1)
    
    last_event_map = event_indices[np.searchsorted(event_indices, np.arange(len(s_numpy)), side='right') - 1]
    
    result = np.arange(len(s_numpy)) - last_event_map
    return np.minimum(result, cap)

def rolling_chunk_stats(
    series: pd.Series, 
    statistic: str = 'var', 
    chunk_size: int = 10, 
    window_chunks: int = 36
) -> pd.Series:
    """
    Calculates statistics over rolling windows of chunk means.
    
    Args:
        series: Input time series.
        statistic: The statistic to compute ('var' or 'trend').
        chunk_size: The number of frames in each small chunk.
        window_chunks: The number of chunks in the main rolling window.
        
    Returns:
        A pandas Series with the calculated statistic.
    """
    window_size = chunk_size * window_chunks
    
    # Calculate means of non-overlapping chunks
    chunk_means = series.rolling(chunk_size).mean().iloc[chunk_size-1::chunk_size]
    
    if statistic == 'var':
        # Variance of the chunk means in a larger rolling window
        rolling_stat = chunk_means.rolling(window_chunks, min_periods=2).var()
    elif statistic == 'trend':
        # Linear trend of the chunk means
        def trend(x):
            if len(x) < 2: return 0
            return np.polyfit(np.arange(len(x)), x, 1)[0]
        rolling_stat = chunk_means.rolling(window_chunks, min_periods=3).apply(trend, raw=True)
    else:
        raise ValueError("`statistic` must be 'var' or 'trend'")
        
    # Upsample to original series frequency
    return rolling_stat.reindex(series.index, method='ffill').fillna(0)

def add_sparse_temporal_features(X: pd.DataFrame, center_x: pd.Series, center_y: pd.Series) -> pd.DataFrame:
    """Adds lightweight sparse temporal features to the feature set."""
    try:
        speed = pd.Series(
            np.sqrt(center_x.diff().pow(2) + center_y.diff().pow(2)),
            index=center_x.index
        ).fillna(0)
        
        # 1. Time-since-event features
        X['fs_still'] = frames_since_event(speed, 0.5, above=False)
        X['fs_fast'] = frames_since_event(speed, 5.0, above=True)
        
        # 2. Chunked statistics (360 frames = 36 chunks of 10)
        X['chk_var'] = rolling_chunk_stats(speed, 'var', chunk_size=10, window_chunks=36)
        X['chk_trend'] = rolling_chunk_stats(speed, 'trend', chunk_size=10, window_chunks=36)
        
    except Exception as e:
        print(f"  Warning: Sparse temporal features failed. Error: {str(e)[:100]}")
    
    return X

def add_sparse_interaction_features(X: pd.DataFrame, mouse_pair: pd.DataFrame) -> pd.DataFrame:
    """Adds sparse social interaction features."""
    try:
        if ('A', 'body_center', 'x') not in mouse_pair.columns or ('B', 'body_center', 'x') not in mouse_pair.columns:
            return X
            
        rel_dist = pd.Series(
            np.sqrt(
                (mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x'])**2 +
                (mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y'])**2
            ), index=mouse_pair.index
        ).fillna(method='bfill').fillna(method='ffill')

        # Time since close proximity
        X['fs_close'] = frames_since_event(rel_dist, 10.0, above=False)
        
        # Chunked distance variability and trend
        X['d_chk_var'] = rolling_chunk_stats(rel_dist, 'var', chunk_size=10, window_chunks=24)
        X['d_trend'] = rolling_chunk_stats(rel_dist, 'trend', chunk_size=10, window_chunks=24)
        
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"  Warning: Sparse interaction features failed with numerical error: {str(e)[:100]}")
        # On failure, impute neutral values for these features
        for col in ['fs_close', 'd_chk_var', 'd_trend']:
            X[col] = 0
    except Exception as e:
        print(f"  Warning: Sparse interaction features failed. Error: {str(e)[:100]}")

    return X


def transform_single(single_mouse: pd.DataFrame, body_parts_tracked: List[str]) -> pd.DataFrame:
    """Transforms tracking data for a single mouse into a feature matrix."""
    available_body_parts = single_mouse.columns.get_level_values(0)
    
    # Base distance features
    feature_dict = {
        f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1)
        for p1, p2 in itertools.combinations(body_parts_tracked, 2) 
        if p1 in available_body_parts and p2 in available_body_parts
    }
    X = pd.DataFrame(feature_dict, index=single_mouse.index)
    
    # Core temporal features
    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']
        cy = single_mouse['body_center']['y']
        
        for w in [5, 15, 30, 60]:
            X[f'cx_s{w}'] = cx.rolling(w, min_periods=1, center=True).std()
            X[f'cy_s{w}'] = cy.rolling(w, min_periods=1, center=True).std()
            X[f'disp{w}'] = np.sqrt(cx.diff().rolling(w, min_periods=1).sum()**2 + cy.diff().rolling(w, min_periods=1).sum()**2)
            X[f'act{w}'] = np.sqrt(cx.diff().rolling(w, min_periods=1).var() + cy.diff().rolling(w, min_periods=1).var())
        
        # Add lightweight sparse temporal features
        X = add_sparse_temporal_features(X, cx, cy)
        
    return X

def transform_pair(mouse_pair: pd.DataFrame, body_parts_tracked: List[str]) -> pd.DataFrame:
    """Transforms tracking data for a pair of mice into a feature matrix."""
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)
    
    # Inter-mouse distances
    feature_dict = {
        f"12+{p1}+{p2}": np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1)
        for p1, p2 in itertools.product(body_parts_tracked, repeat=2) 
        if p1 in avail_A and p2 in avail_B
    }
    X = pd.DataFrame(feature_dict, index=mouse_pair.index)

    # Temporal interaction features
    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd_full = np.square(mouse_pair['A']['body_center'] - mouse_pair['B']['body_center']).sum(axis=1)
        
        for w in [5, 15, 30, 60]:
            X[f'd_m{w}'] = cd_full.rolling(w, min_periods=1, center=True).mean()
            X[f'd_s{w}'] = cd_full.rolling(w, min_periods=1, center=True).std()
            X[f'd_mx{w}'] = cd_full.rolling(w, min_periods=1, center=True).max()
        
        Axd = mouse_pair['A']['body_center']['x'].diff()
        Ayd = mouse_pair['A']['body_center']['y'].diff()
        Bxd = mouse_pair['B']['body_center']['x'].diff()
        Byd = mouse_pair['B']['body_center']['y'].diff()
        coord = Axd * Bxd + Ayd * Byd
        X[f'co_m30'] = coord.rolling(30, min_periods=1, center=True).mean()
        
        # Add sparse interaction features
        X = add_sparse_interaction_features(X, mouse_pair)
        
    return X


def generate_mouse_data(dataset: pd.DataFrame, traintest: str):
    """
    A generator that yields processed data for single mice and pairs.
    Reads tracking and annotation data video by video.
    """
    assert traintest in ['train', 'test']
    traintest_directory = Config.TRAIN_TRACKING_DIR if traintest == 'train' else Config.TEST_TRACKING_DIR
    
    for _, row in dataset.iterrows():
        lab_id, video_id = row.lab_id, row.video_id
        if lab_id.startswith('MABe22'): continue
        if not isinstance(row.behaviors_labeled, str): continue

        path = os.path.join(traintest_directory, lab_id, f"{video_id}.parquet")
        try:
            vid_df = pd.read_parquet(path)
        except FileNotFoundError:
            continue
            
        # Pre-process tracking data
        if len(vid_df['bodypart'].unique()) > 5:            
            vid_df = vid_df[~vid_df['bodypart'].isin(Config.DROP_BODY_PARTS)]
        pvid = vid_df.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).sort_index(axis=1)
        pvid /= row.pix_per_cm_approx
        del vid_df
        
        # Parse labeled behaviors
        vid_behaviors = pd.DataFrame(
            [b.split(',') for b in sorted(list(set(json.loads(row.behaviors_labeled))))],
            columns=['agent', 'target', 'action']
        )
        
        # Load annotations if training
        annot = None
        if traintest == 'train':
            try:
                annot_path = os.path.join(Config.TRAIN_ANNOTATION_DIR, lab_id, f"{video_id}.parquet")
                annot = pd.read_parquet(annot_path)
            except FileNotFoundError:
                continue

        # --- Yield Single Mouse Data ---
        single_behaviors = vid_behaviors.query("target == 'self'")
        for mouse_id_str in single_behaviors['agent'].unique():
            try:
                mouse_id = int(mouse_id_str[-1])
                actions = single_behaviors.query("agent == @mouse_id_str")['action'].unique()
                data = pvid.loc[:, mouse_id]
                meta = pd.DataFrame({'video_id': video_id, 'agent_id': mouse_id_str, 'target_id': 'self', 'video_frame': data.index})
                
                if traintest == 'train':
                    labels = pd.DataFrame(0.0, columns=actions, index=data.index)
                    annot_sub = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                    for _, r in annot_sub.iterrows():
                        labels.loc[r['start_frame']:r['stop_frame'], r.action] = 1.0
                    yield 'single', data, meta, labels
                else:
                    yield 'single', data, meta, actions
            except (KeyError, IndexError):
                pass

        # --- Yield Paired Mouse Data ---
        pair_behaviors = vid_behaviors.query("target != 'self'")
        mice_ids = pvid.columns.get_level_values('mouse_id').unique()
        for agent, target in itertools.permutations(mice_ids, 2):
            agent_str, target_str = f"mouse{agent}", f"mouse{target}"
            actions = pair_behaviors.query("(agent == @agent_str) & (target == @target_str)")['action'].unique()
            if len(actions) == 0: continue

            try:
                data = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                meta = pd.DataFrame({'video_id': video_id, 'agent_id': agent_str, 'target_id': target_str, 'video_frame': data.index})
                
                if traintest == 'train':
                    labels = pd.DataFrame(0.0, columns=actions, index=data.index)
                    annot_sub = annot.query("(agent_id == @agent) & (target_id == @target)")
                    for _, r in annot_sub.iterrows():
                        labels.loc[r['start_frame']:r['stop_frame'], r.action] = 1.0
                    yield 'pair', data, meta, labels
                else:
                    yield 'pair', data, meta, actions
            except (KeyError, IndexError):
                pass
        
        del pvid, annot
        gc.collect()


class StratifiedSubsetClassifier(ClassifierMixin, BaseEstimator):
    """A wrapper that fits an estimator on a stratified random subset of the data."""
    def __init__(self, estimator, n_samples: int):
        self.estimator = estimator
        self.n_samples = n_samples

    def fit(self, X, y):
        if len(X) <= self.n_samples:
            self.estimator.fit(X, y)
        else:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=self.n_samples, random_state=Config.RANDOM_SEED)
            try:
                train_idx, _ = next(sss.split(X, y))
                self.estimator.fit(X[train_idx], y[train_idx])
            except ValueError: # Handle cases with too few samples for stratification
                indices = np.random.choice(len(X), self.n_samples, replace=False)
                self.estimator.fit(X[indices], y[indices])
        
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
        
    def predict(self, X):
        return self.estimator.predict(X)

def get_model_ensemble() -> List[BaseEstimator]:
    """Defines and returns a list of models for the ensemble."""
    models = []
    # Model 1: LightGBM (Balanced)
    models.append(make_pipeline(SimpleImputer(), StratifiedSubsetClassifier(
        lightgbm.LGBMClassifier(n_estimators=225, learning_rate=0.07, random_state=Config.RANDOM_SEED, verbose=-1),
        n_samples=Config.N_SAMPLES_MAX_SINGLE
    )))
    # Model 2: LightGBM (Deeper)
    models.append(make_pipeline(SimpleImputer(), StratifiedSubsetClassifier(
        lightgbm.LGBMClassifier(n_estimators=150, num_leaves=63, random_state=Config.RANDOM_SEED, verbose=-1),
        n_samples=Config.N_SAMPLES_MAX_PAIR
    )))
    # Model 3: XGBoost
    if XGBOOST_AVAILABLE:
        models.append(make_pipeline(SimpleImputer(), StratifiedSubsetClassifier(
            XGBClassifier(n_estimators=180, learning_rate=0.08, tree_method='hist', random_state=Config.RANDOM_SEED, verbosity=0),
            n_samples=Config.N_SAMPLES_MAX_PAIR
        )))
    return models

def predict_to_submission(pred_probas: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Converts prediction probabilities to a submission-formatted DataFrame."""
    if pred_probas.empty:
        return pd.DataFrame()

    # Smooth probabilities and find the most likely action per frame
    pred_smoothed = pred_probas.rolling(window=5, min_periods=1, center=True).mean()
    ama = np.argmax(pred_smoothed.values, axis=1)
    max_probs = pred_smoothed.max(axis=1).values
    
    # Apply a default threshold
    threshold_mask = max_probs >= Config.DEFAULT_THRESHOLD

    ama_series = pd.Series(np.where(threshold_mask, ama, -1), index=meta.index)
    
    changes = ama_series[ama_series != ama_series.shift(1)]

    starts = changes[changes >= 0]
    if starts.empty:
        return pd.DataFrame()
        
    start_frames = starts.index
    stop_frames = []
    
    change_indices = changes.index
    for frame in start_frames:
        
        current_pos = change_indices.get_loc(frame)
        
        # If it's not the last change, the stop frame is the next change
        if current_pos + 1 < len(change_indices):
            stop_frames.append(change_indices[current_pos + 1])
        # If it is the last change, the stop frame is the end of the video
        else:
            stop_frames.append(meta.index.max() + 1)
            
    sub_part = pd.DataFrame({
        'video_id': meta.loc[start_frames, 'video_id'].values,
        'agent_id': meta.loc[start_frames, 'agent_id'].values,
        'target_id': meta.loc[start_frames, 'target_id'].values,
        'action': pred_probas.columns[starts.values],
        'start_frame': start_frames,
        'stop_frame': stop_frames
    })
    
    # Filter out short-duration events
    sub_part = sub_part[sub_part.stop_frame - sub_part.start_frame >= Config.MIN_DURATION_FRAMES]
    return sub_part.reset_index(drop=True)


# --- Load datasets ---
train_df = pd.read_csv(Config.TRAIN_CSV)
test_df = pd.read_csv(Config.TEST_CSV)
body_parts_tracked_list = list(train_df.body_parts_tracked.unique())

submission_list = []

for section, body_parts_tracked_str in enumerate(body_parts_tracked_list):
    if body_parts_tracked_str is np.nan: continue
    
    body_parts_tracked = json.loads(body_parts_tracked_str)
    print(f"\n--- Processing group {section+1}/{len(body_parts_tracked_list)}: {len(body_parts_tracked)} body parts ---")
    
    # Prepare training data for this body part configuration
    train_subset = train_df[train_df.body_parts_tracked == body_parts_tracked_str]
    
    all_data = {'single': [], 'pair': []}
    all_labels = {'single': [], 'pair': []}
    all_meta = {'single': [], 'pair': []}
    
    for switch, data, meta, label in generate_mouse_data(train_subset, 'train'):
        all_data[switch].append(data)
        all_labels[switch].append(label)
        all_meta[switch].append(meta)

    # --- Train and predict for each data type (single/pair) ---
    for switch in ['single', 'pair']:
        if not all_data[switch]:
            continue
            
        print(f"  Processing '{switch}' mouse data...")
        # After
        data_tr = pd.concat(all_data[switch]).reset_index(drop=True)
        label_tr = pd.concat(all_labels[switch]).reset_index(drop=True)
        
        # Feature Engineering
        transform_func = transform_single if switch == 'single' else transform_pair
        X_tr = transform_func(data_tr, body_parts_tracked)
        del data_tr
        gc.collect()

        print(f"    Train features shape: {X_tr.shape}")
        
        # Train one model per action
        action_models = {}
        for action in tqdm(label_tr.columns, desc=f"    Training on {switch} actions"):
            y_action = label_tr[action].dropna()
            if y_action.sum() < 10: continue # Skip actions with too few examples
            
            ensemble = get_model_ensemble()
            trained_models = []
            for model in ensemble:
                try:
                    m_clone = clone(model)
                    m_clone.fit(X_tr.loc[y_action.index].values, y_action.values)
                    trained_models.append(m_clone)
                except Exception as e:
                    print(f"      Warning: Model training failed for action '{action}'. Error: {e}")
            
            if trained_models:
                action_models[action] = trained_models
        
        del X_tr, label_tr
        gc.collect()
        
        # --- Inference on Test Set ---
        test_subset = test_df[test_df.body_parts_tracked == body_parts_tracked_str]
        test_generator = generate_mouse_data(test_subset, 'test')
        
        for switch_te, data_te, meta_te, actions_te in test_generator:
            if switch_te != switch: continue
            
            X_te = transform_func(data_te, body_parts_tracked)
            meta_te.set_index('video_frame', inplace=True)
            
            pred_probas = pd.DataFrame(index=X_te.index)
            for action, models in action_models.items():
                if action in actions_te:
                    try:
                        probs = [m.predict_proba(X_te.values)[:, 1] for m in models]
                        pred_probas[action] = np.mean(probs, axis=0)
                    except Exception as e:
                        print(f"      Warning: Prediction failed for action '{action}'. Error: {e}")
                        
            sub_part = predict_to_submission(pred_probas, meta_te)
            if not sub_part.empty:
                submission_list.append(sub_part)
                print(f"    Found {len(sub_part)} predictions for video {meta_te['video_id'].iloc[0]}")

            del X_te, data_te, meta_te, pred_probas
            gc.collect()


def robustify_submission(submission: pd.DataFrame) -> pd.DataFrame:
    """Post-processes the submission to remove overlaps."""
    if submission.empty:
        return submission
        
    submission = submission[submission.start_frame < submission.stop_frame].copy()
    
    clean_groups = []
    # Group by the agent/target pair within each video
    for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values('start_frame').reset_index(drop=True)
        
        # Greedily remove any prediction that overlaps with the previously accepted one
        if not group.empty:
            accepted = [group.iloc[0]]
            for i in range(1, len(group)):
                if group.iloc[i]['start_frame'] >= accepted[-1]['stop_frame']:
                    accepted.append(group.iloc[i])
            clean_groups.append(pd.DataFrame(accepted))
            
    return pd.concat(clean_groups).reset_index(drop=True) if clean_groups else pd.DataFrame()

# --- Create final submission file ---
if not submission_list:
    # Create a dummy submission if no predictions were made
    final_submission = pd.DataFrame({
        'video_id': [438887472], 'agent_id': ['mouse1'], 'target_id': ['self'],
        'action': ['rear'], 'start_frame': [100], 'stop_frame': [200]
    })
else:
    submission = pd.concat(submission_list).reset_index(drop=True)
    final_submission = robustify_submission(submission)

final_submission.index.name = 'row_id'
final_submission.to_csv(Config.SUBMISSION_FILE)

print(f"\n✅ Submission created: {Config.SUBMISSION_FILE}")
print(f"Total predictions: {len(final_submission)}")




