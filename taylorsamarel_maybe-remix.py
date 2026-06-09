# MABe Challenge - Advanced Ensemble with Expert + Generalized Models
# MEMORY OPTIMIZED VERSION - Complete working code

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
import lightgbm
from collections import defaultdict
import polars as pl
from scipy import signal, stats

from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.model_selection import cross_val_predict, GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

warnings.filterwarnings('ignore')

# Try importing additional models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

# ==================== IMPROVED CLASSIFIERS ====================

class StratifiedSubsetClassifier(ClassifierMixin, BaseEstimator):
    """Fit estimator with stratified sampling to maintain class balance"""
    def __init__(self, estimator, n_samples):
        self.estimator = estimator
        self.n_samples = n_samples

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)  # Memory: use float32
        y = np.asarray(y, dtype=np.int8)      # Memory: use int8
        
        if len(X) <= self.n_samples:
            self.estimator.fit(X, y)
        else:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=min(self.n_samples, len(X)), random_state=42)
            try:
                for train_idx, _ in sss.split(X, y):
                    self.estimator.fit(X[train_idx], y[train_idx])
            except:
                downsample = len(X) // self.n_samples
                downsample = max(downsample, 1)
                self.estimator.fit(X[::downsample], y[::downsample])
        
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if len(self.classes_) == 1:
            return np.full((len(X), 1), 1.0, dtype=np.float32)
        probs = self.estimator.predict_proba(X)
        return probs.astype(np.float32)
        
    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.estimator.predict(X)

# ==================== SCORING FUNCTIONS ====================

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

# ==================== DATA LOADING ====================

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
        else:
            if verbose and traintest == 'test': print('video with all values', video_id, traintest, len(vid), 'frames')
        del vid
        gc.collect()
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        pvid = pvid.astype(np.float32)  # Memory: use float32
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
                        single_mouse_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=single_mouse.index, dtype=np.float32)
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
                        mouse_pair_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=mouse_pair.index, dtype=np.float32)
                        annot_subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            mouse_pair_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'pair', mouse_pair, mouse_pair_meta, mouse_pair_label
                    else:
                        if verbose: print('- test pair', video_id, agent, target)
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions

# ==================== ADAPTIVE THRESHOLDING ====================

action_thresholds = defaultdict(lambda: 0.27)

def predict_multiclass_adaptive(pred, meta, action_thresholds):
    """Adaptive thresholding per action + temporal smoothing"""
    pred_smoothed = pred.rolling(window=5, min_periods=1, center=True).mean()
    
    ama = np.argmax(pred_smoothed.values, axis=1)
    
    max_probs = pred_smoothed.max(axis=1).values
    threshold_mask = np.zeros(len(pred_smoothed), dtype=bool)
    for i, action in enumerate(pred_smoothed.columns):
        action_mask = (ama == i)
        threshold = action_thresholds.get(action, 0.27)
        threshold_mask |= (action_mask & (max_probs >= threshold))
    
    ama = np.where(threshold_mask, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame.values)
    
    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]
    mask = ama_changes.values >= 0
    mask[-1] = False
    
    submission_part = pd.DataFrame({
        'video_id': meta_changes['video_id'].iloc[mask].values,
        'agent_id': meta_changes['agent_id'].iloc[mask].values,
        'target_id': meta_changes['target_id'].iloc[mask].values,
        'action': pred.columns[ama_changes.iloc[mask].values],
        'start_frame': ama_changes.index[mask],
        'stop_frame': ama_changes.index[1:][mask[:-1]]
    })
    
    stop_video_id = meta_changes['video_id'].iloc[1:][mask[:-1]].values
    stop_agent_id = meta_changes['agent_id'].iloc[1:][mask[:-1]].values
    stop_target_id = meta_changes['target_id'].iloc[1:][mask[:-1]].values
    
    for i in range(len(submission_part)):
        video_id = submission_part.video_id.iloc[i]
        agent_id = submission_part.agent_id.iloc[i]
        target_id = submission_part.target_id.iloc[i]
        if i < len(stop_video_id):
            if stop_video_id[i] != video_id or stop_agent_id[i] != agent_id or stop_target_id[i] != target_id:
                new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
                submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame
        else:
            new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
            submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame
    
    duration = submission_part.stop_frame - submission_part.start_frame
    submission_part = submission_part[duration >= 3].reset_index(drop=True)
    
    if len(submission_part) > 0:
        assert (submission_part.stop_frame > submission_part.start_frame).all(), 'stop <= start'
    
    if verbose: print(f'  actions found: {len(submission_part)}')
    return submission_part

# ==================== CORE FEATURE ENGINEERING ====================

def add_curvature_features(X, center_x, center_y):
    """Trajectory curvature"""
    vel_x = center_x.diff()
    vel_y = center_y.diff()
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()
    
    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = pd.Series(np.sqrt(vel_x**2 + vel_y**2), index=vel_x.index)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)
    
    for window in [30, 60]:
        X[f'curv_mean_{window}'] = curvature.rolling(window, min_periods=5).mean().astype(np.float32)
    
    angle = pd.Series(np.arctan2(vel_y, vel_x), index=vel_y.index)
    angle_change = np.abs(angle.diff())
    X['turn_rate_30'] = angle_change.rolling(30, min_periods=5).sum().astype(np.float32)
    
    return X

def add_multiscale_features(X, center_x, center_y):
    """Multi-scale temporal features"""
    speed = pd.Series(np.sqrt(center_x.diff()**2 + center_y.diff()**2), index=center_x.index)
    
    scales = [10, 40, 160]
    for scale in scales:
        if len(speed) >= scale:
            X[f'sp_m{scale}'] = speed.rolling(scale, min_periods=max(1, scale//4)).mean().astype(np.float32)
            X[f'sp_s{scale}'] = speed.rolling(scale, min_periods=max(1, scale//4)).std().astype(np.float32)
    
    if len(scales) >= 2 and f'sp_m{scales[0]}' in X.columns and f'sp_m{scales[-1]}' in X.columns:
        X['sp_ratio'] = (X[f'sp_m{scales[0]}'] / (X[f'sp_m{scales[-1]}'] + 1e-6)).astype(np.float32)
    
    return X

def add_state_features(X, center_x, center_y):
    """Behavioral state transitions"""
    speed = pd.Series(np.sqrt(center_x.diff()**2 + center_y.diff()**2), index=center_x.index)
    speed_ma = speed.rolling(15, min_periods=5).mean()
    
    try:
        speed_states = pd.cut(speed_ma, bins=[-np.inf, 0.5, 2.0, 5.0, np.inf], labels=[0, 1, 2, 3]).astype(float)
        
        for window in [60, 120]:
            if len(speed_states) >= window:
                for state in [0, 1, 2, 3]:
                    X[f's{state}_{window}'] = (speed_states == state).astype(float).rolling(window, min_periods=10).mean().astype(np.float32)
                
                state_changes = (speed_states != speed_states.shift(1)).astype(float)
                X[f'trans_{window}'] = state_changes.rolling(window, min_periods=10).sum().astype(np.float32)
    except:
        pass
    
    return X

def add_longrange_features(X, center_x, center_y):
    """Long-range temporal features"""
    for window in [120, 240]:
        if len(center_x) >= window:
            X[f'x_ml{window}'] = center_x.rolling(window, min_periods=20).mean().astype(np.float32)
            X[f'y_ml{window}'] = center_y.rolling(window, min_periods=20).mean().astype(np.float32)
    
    for span in [60, 120]:
        X[f'x_e{span}'] = center_x.ewm(span=span, min_periods=1).mean().astype(np.float32)
        X[f'y_e{span}'] = center_y.ewm(span=span, min_periods=1).mean().astype(np.float32)
    
    speed = pd.Series(np.sqrt(center_x.diff()**2 + center_y.diff()**2), index=center_x.index)
    for window in [60, 120]:
        if len(speed) >= window:
            X[f'sp_pct{window}'] = speed.rolling(window, min_periods=20).rank(pct=True).astype(np.float32)
    
    return X

def add_interaction_features(X, mouse_pair, avail_A, avail_B):
    """Social interaction features"""
    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return X
    
    rel_x = mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x']
    rel_y = mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y']
    rel_dist = pd.Series(np.sqrt(rel_x**2 + rel_y**2), index=rel_x.index)
    
    A_vx = mouse_pair['A']['body_center']['x'].diff()
    A_vy = mouse_pair['A']['body_center']['y'].diff()
    B_vx = mouse_pair['B']['body_center']['x'].diff()
    B_vy = mouse_pair['B']['body_center']['y'].diff()
    
    A_lead = (A_vx * rel_x + A_vy * rel_y) / (pd.Series(np.sqrt(A_vx**2 + A_vy**2), index=A_vx.index) * rel_dist + 1e-6)
    B_lead = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (pd.Series(np.sqrt(B_vx**2 + B_vy**2), index=B_vx.index) * rel_dist + 1e-6)
    
    for window in [30, 60]:
        X[f'A_ld{window}'] = A_lead.rolling(window, min_periods=5).mean().astype(np.float32)
        X[f'B_ld{window}'] = B_lead.rolling(window, min_periods=5).mean().astype(np.float32)
    
    approach = -rel_dist.diff()
    chase = approach * B_lead
    X['chase_30'] = chase.rolling(30, min_periods=5).mean().astype(np.float32)
    
    for window in [60, 120]:
        A_sp = pd.Series(np.sqrt(A_vx**2 + A_vy**2), index=A_vx.index)
        B_sp = pd.Series(np.sqrt(B_vx**2 + B_vy**2), index=B_vx.index)
        X[f'sp_cor{window}'] = A_sp.rolling(window, min_periods=10).corr(B_sp).astype(np.float32)
    
    return X

def add_acceleration_features(X, cx, cy):
    """Acceleration and jerk features"""
    vx = cx.diff()
    vy = cy.diff()
    ax = vx.diff()
    ay = vy.diff()
    
    acc_mag = pd.Series(np.sqrt(ax**2 + ay**2), index=ax.index)
    X['acc_mean_30'] = acc_mag.rolling(30, min_periods=5).mean().astype(np.float32)
    X['acc_std_30'] = acc_mag.rolling(30, min_periods=5).std().astype(np.float32)
    
    jx = ax.diff()
    jy = ay.diff()
    jerk_mag = pd.Series(np.sqrt(jx**2 + jy**2), index=jx.index)
    X['jerk_mean_30'] = jerk_mag.rolling(30, min_periods=5).mean().astype(np.float32)
    
    return X

def add_path_features(X, cx, cy):
    """Path tortuosity and efficiency"""
    for window in [60, 120]:
        if len(cx) >= window:
            path_len = pd.Series(np.sqrt(cx.diff()**2 + cy.diff()**2), index=cx.index).rolling(window, min_periods=10).sum()
            start_x = cx.shift(window-1)
            start_y = cy.shift(window-1)
            direct_dist = pd.Series(np.sqrt((cx - start_x)**2 + (cy - start_y)**2), index=cx.index)
            X[f'tortuous_{window}'] = (path_len / (direct_dist + 1e-6)).astype(np.float32)
    
    return X

def add_statistical_features(X, cx, cy):
    """Higher-order statistics"""
    speed = pd.Series(np.sqrt(cx.diff()**2 + cy.diff()**2), index=cx.index)
    
    for window in [60, 120]:
        if len(speed) >= window:
            X[f'sp_skew_{window}'] = speed.rolling(window, min_periods=20).skew().astype(np.float32)
            X[f'sp_kurt_{window}'] = speed.rolling(window, min_periods=20).kurt().astype(np.float32)
            X[f'sp_q25_{window}'] = speed.rolling(window, min_periods=20).quantile(0.25).astype(np.float32)
            X[f'sp_q75_{window}'] = speed.rolling(window, min_periods=20).quantile(0.75).astype(np.float32)
    
    return X

def add_territorial_features(X, cx, cy):
    """Model 1: Territorial/spatial memory"""
    for lookback in [600, 1800]:
        if len(cx) >= lookback:
            past_cx = cx.shift(lookback)
            past_cy = cy.shift(lookback)
            revisit_dist = pd.Series(np.sqrt((cx - past_cx)**2 + (cy - past_cy)**2), index=cx.index)
            X[f'revisit_{lookback}'] = revisit_dist.astype(np.float32)
            X[f'territorial_{lookback}'] = (revisit_dist < 3.0).astype(float).rolling(60, min_periods=10).mean().astype(np.float32)
    
    dist_from_center = pd.Series(np.sqrt(cx**2 + cy**2), index=cx.index)
    arena_radius = cx.rolling(3000, min_periods=100).max() - cx.rolling(3000, min_periods=100).min()
    X['edge_prox'] = (dist_from_center / (arena_radius/2 + 1e-6)).astype(np.float32)
    
    return X

def add_temporal_features(X, cx, cy):
    """Model 2: Circadian/temporal rhythm"""
    if len(cx) >= 1800:
        speed = pd.Series(np.sqrt(cx.diff()**2 + cy.diff()**2), index=cx.index)
        for ultra_window in [1800, 3600]:
            if len(speed) >= ultra_window:
                X[f'activity_rhythm_{ultra_window}'] = speed.rolling(ultra_window, min_periods=180).mean().astype(np.float32)
                X[f'activity_var_{ultra_window}'] = speed.rolling(ultra_window, min_periods=180).std().astype(np.float32)
    
    return X

def add_postural_features(X, single_mouse, available_body_parts):
    """Model 3: Body configuration/postural"""
    if all(p in available_body_parts for p in ['nose', 'tail_base', 'body_center']):
        body_length = pd.Series(np.sqrt((single_mouse['nose']['x'] - single_mouse['tail_base']['x'])**2 + 
                             (single_mouse['nose']['y'] - single_mouse['tail_base']['y'])**2), index=single_mouse.index)
        X['body_len_now'] = body_length.astype(np.float32)
        for lag in [30, 90]:
            X[f'body_len_lag_{lag}'] = body_length.shift(lag).astype(np.float32)
            X[f'body_stretch_{lag}'] = (body_length - body_length.shift(lag)).astype(np.float32)
        
        body_angle = pd.Series(np.arctan2(single_mouse['nose']['y'] - single_mouse['tail_base']['y'],
                                single_mouse['nose']['x'] - single_mouse['tail_base']['x']), index=single_mouse.index)
        X['orient_stability_60'] = body_angle.rolling(60, min_periods=10).std().astype(np.float32)
        
        orient_change = np.abs(body_angle.diff())
        orient_change = pd.Series(np.where(orient_change > np.pi, 2*np.pi - orient_change, orient_change), index=orient_change.index)
        X['turn_freq_120'] = (orient_change > 0.5).astype(float).rolling(120, min_periods=20).sum().astype(np.float32)
    
    return X

def add_social_memory_features(X, A_cx, A_cy, B_cx, B_cy):
    """Model 4: Social interaction/pheromone"""
    dist_AB = pd.Series(np.sqrt((A_cx - B_cx)**2 + (A_cy - B_cy)**2), index=A_cx.index)
    
    for history in [600, 1800]:
        if len(dist_AB) >= history:
            X[f'interact_hist_{history}'] = (dist_AB < 10.0).astype(float).rolling(history, min_periods=60).mean().astype(np.float32)
            X[f'close_time_{history}'] = (dist_AB < 5.0).astype(float).rolling(history, min_periods=60).sum().astype(np.float32)
    
    close_contact = (dist_AB < 5.0).astype(float)
    frames_since = []
    last_contact = -999999
    for i, val in enumerate(close_contact):
        if val > 0:
            last_contact = i
        frames_since.append(i - last_contact if last_contact >= 0 else 999999)
    X['frames_since_contact'] = pd.Series(frames_since, index=close_contact.index, dtype=np.float32)
    
    return X

def add_micromovement_features(X, single_mouse, available_body_parts):
    """Model 5: Fine-scale/micromovement"""
    if 'nose' in available_body_parts:
        nose_speed = pd.Series(np.sqrt(single_mouse['nose']['x'].diff()**2 + single_mouse['nose']['y'].diff()**2), index=single_mouse.index)
        X['nose_jitter_30'] = (nose_speed.rolling(30, min_periods=5).std() / (nose_speed.rolling(30, min_periods=5).mean() + 1e-6)).astype(np.float32)
        
        if len(nose_speed) >= 60:
            for period in [10, 15]:
                X[f'nose_rhythm_{period}'] = nose_speed.rolling(period, min_periods=3).std().astype(np.float32)
    
    if all(p in available_body_parts for p in ['ear_left', 'ear_right']):
        ear_d = pd.Series(np.sqrt((single_mouse['ear_left']['x'] - single_mouse['ear_right']['x'])**2 + 
                       (single_mouse['ear_left']['y'] - single_mouse['ear_right']['y'])**2), index=single_mouse.index)
        X['ear_micro'] = ear_d.rolling(15, min_periods=3).std().astype(np.float32)
    
    return X

def add_pair_social_features(X, mouse_pair, avail_A, avail_B):
    """Model 4: Pair social features"""
    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return X
    
    A_cx = mouse_pair['A']['body_center']['x']
    A_cy = mouse_pair['A']['body_center']['y']
    B_cx = mouse_pair['B']['body_center']['x']
    B_cy = mouse_pair['B']['body_center']['y']
    
    A_vx = A_cx.diff()
    A_vy = A_cy.diff()
    B_vx = B_cx.diff()
    B_vy = B_cy.diff()
    
    dist_AB = pd.Series(np.sqrt((A_cx - B_cx)**2 + (A_cy - B_cy)**2), index=A_cx.index)
    rel_x = B_cx - A_cx
    rel_y = B_cy - A_cy
    A_approach = (A_vx * rel_x + A_vy * rel_y) / (dist_AB + 1e-6)
    B_approach = (-B_vx * rel_x - B_vy * rel_y) / (dist_AB + 1e-6)
    
    for window in [60, 180]:
        X[f'A_approach_{window}'] = A_approach.rolling(window, min_periods=10).mean().astype(np.float32)
        X[f'approach_asym_{window}'] = (A_approach - B_approach).rolling(window, min_periods=10).mean().astype(np.float32)
    
    for delay in [10, 30]:
        if len(A_vx) >= delay:
            A_delayed_x = A_vx.shift(-delay)
            A_delayed_y = A_vy.shift(-delay)
            movement_corr = (B_vx * A_delayed_x + B_vy * A_delayed_y) / (
                pd.Series(np.sqrt(B_vx**2 + B_vy**2), index=B_vx.index) * pd.Series(np.sqrt(A_delayed_x**2 + A_delayed_y**2), index=A_delayed_x.index) + 1e-6)
            X[f'follow_delay_{delay}'] = movement_corr.rolling(60, min_periods=10).mean().astype(np.float32)
    
    return X

def add_pair_micromovement_features(X, mouse_pair, avail_A, avail_B):
    """Model 5: Pair micromovement"""
    if 'nose' in avail_A and 'nose' in avail_B:
        nose_dist = pd.Series(np.sqrt((mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x'])**2 + 
                           (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y'])**2), index=mouse_pair.index)
        X['nose_contact'] = (nose_dist < 2.0).astype(float).astype(np.float32)
        X['nose_contact_300'] = X['nose_contact'].rolling(300, min_periods=30).sum().astype(np.float32)
        
        if 'tail_base' in avail_B:
            nose_to_ano = pd.Series(np.sqrt((mouse_pair['A']['nose']['x'] - mouse_pair['B']['tail_base']['x'])**2 + 
                                 (mouse_pair['A']['nose']['y'] - mouse_pair['B']['tail_base']['y'])**2), index=mouse_pair.index)
            X['anogen_prox'] = (nose_to_ano < 3.0).astype(float).astype(np.float32)
            X['anogen_invest_180'] = X['anogen_prox'].rolling(180, min_periods=20).sum().astype(np.float32)
    
    return X

# ==================== TRANSFORM FUNCTIONS ====================

def transform_single_base(single_mouse, body_parts_tracked):
    """Base features for all models"""
    available_body_parts = single_mouse.columns.get_level_values(0)
    
    X = pd.DataFrame({
        f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.combinations(body_parts_tracked, 2) 
        if p1 in available_body_parts and p2 in available_body_parts
    }, dtype=np.float32)
    X = X.reindex(columns=[f"{p1}+{p2}" for p1, p2 in itertools.combinations(body_parts_tracked, 2)], copy=False)

    if all(p in single_mouse.columns for p in ['ear_left', 'ear_right', 'tail_base']):
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(10)
        speeds = pd.DataFrame({
            'sp_lf': np.square(single_mouse['ear_left'] - shifted['ear_left']).sum(axis=1, skipna=False),
            'sp_rt': np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False),
            'sp_lf2': np.square(single_mouse['ear_left'] - shifted['tail_base']).sum(axis=1, skipna=False),
            'sp_rt2': np.square(single_mouse['ear_right'] - shifted['tail_base']).sum(axis=1, skipna=False),
        }, dtype=np.float32)
        X = pd.concat([X, speeds], axis=1)
    
    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X['elong'] = (X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)).astype(np.float32)
    
    if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
        v1 = single_mouse['nose'] - single_mouse['body_center']
        v2 = single_mouse['tail_base'] - single_mouse['body_center']
        X['body_ang'] = ((v1['x'] * v2['x'] + v1['y'] * v2['y']) / (
            pd.Series(np.sqrt(v1['x']**2 + v1['y']**2), index=v1.index) * pd.Series(np.sqrt(v2['x']**2 + v2['y']**2), index=v2.index) + 1e-6)).astype(np.float32)
    
    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']
        cy = single_mouse['body_center']['y']
        
        for w in [5, 15, 30, 60]:
            X[f'cx_m{w}'] = cx.rolling(w, min_periods=1, center=True).mean().astype(np.float32)
            X[f'cy_m{w}'] = cy.rolling(w, min_periods=1, center=True).mean().astype(np.float32)
            X[f'cx_s{w}'] = cx.rolling(w, min_periods=1, center=True).std().astype(np.float32)
            X[f'cy_s{w}'] = cy.rolling(w, min_periods=1, center=True).std().astype(np.float32)
            X[f'x_rng{w}'] = (cx.rolling(w, min_periods=1, center=True).max() - cx.rolling(w, min_periods=1, center=True).min()).astype(np.float32)
            X[f'y_rng{w}'] = (cy.rolling(w, min_periods=1, center=True).max() - cy.rolling(w, min_periods=1, center=True).min()).astype(np.float32)
            X[f'disp{w}'] = pd.Series(np.sqrt(cx.diff().rolling(w, min_periods=1).sum()**2 + cy.diff().rolling(w, min_periods=1).sum()**2), index=cx.index).astype(np.float32)
            X[f'act{w}'] = pd.Series(np.sqrt(cx.diff().rolling(w, min_periods=1).var() + cy.diff().rolling(w, min_periods=1).var()), index=cx.index).astype(np.float32)
        
        X = add_curvature_features(X, cx, cy)
        X = add_multiscale_features(X, cx, cy)
        X = add_state_features(X, cx, cy)
        X = add_longrange_features(X, cx, cy)
    
    if all(p in available_body_parts for p in ['nose', 'tail_base']):
        nt_dist = pd.Series(np.sqrt((single_mouse['nose']['x'] - single_mouse['tail_base']['x'])**2 + 
                         (single_mouse['nose']['y'] - single_mouse['tail_base']['y'])**2), index=single_mouse.index)
        for lag in [10, 20, 40]:
            X[f'nt_lg{lag}'] = nt_dist.shift(lag).astype(np.float32)
            X[f'nt_df{lag}'] = (nt_dist - nt_dist.shift(lag)).astype(np.float32)
    
    if all(p in available_body_parts for p in ['ear_left', 'ear_right']):
        ear_d = pd.Series(np.sqrt((single_mouse['ear_left']['x'] - single_mouse['ear_right']['x'])**2 + 
                       (single_mouse['ear_left']['y'] - single_mouse['ear_right']['y'])**2), index=single_mouse.index)
        for off in [-20, -10, 10, 20]:
            X[f'ear_o{off}'] = ear_d.shift(-off).astype(np.float32)
        X['ear_con'] = (ear_d.rolling(30, min_periods=1, center=True).std() / (ear_d.rolling(30, min_periods=1, center=True).mean() + 1e-6)).astype(np.float32)
    
    return X, available_body_parts

def transform_single_model1(single_mouse, body_parts_tracked):
    X, available_body_parts = transform_single_base(single_mouse, body_parts_tracked)
    if 'body_center' in available_body_parts:
        X = add_territorial_features(X, single_mouse['body_center']['x'], single_mouse['body_center']['y'])
    return X

def transform_single_model2(single_mouse, body_parts_tracked):
    X, available_body_parts = transform_single_base(single_mouse, body_parts_tracked)
    if 'body_center' in available_body_parts:
        X = add_temporal_features(X, single_mouse['body_center']['x'], single_mouse['body_center']['y'])
    return X

def transform_single_model3(single_mouse, body_parts_tracked):
    X, available_body_parts = transform_single_base(single_mouse, body_parts_tracked)
    X = add_postural_features(X, single_mouse, available_body_parts)
    return X

def transform_single_model4(single_mouse, body_parts_tracked):
    X, _ = transform_single_base(single_mouse, body_parts_tracked)
    return X

def transform_single_model5(single_mouse, body_parts_tracked):
    X, available_body_parts = transform_single_base(single_mouse, body_parts_tracked)
    X = add_micromovement_features(X, single_mouse, available_body_parts)
    return X

def transform_single_generalized(single_mouse, body_parts_tracked):
    """Comprehensive features for generalized models"""
    X, available_body_parts = transform_single_base(single_mouse, body_parts_tracked)
    
    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']
        cy = single_mouse['body_center']['y']
        
        X = add_acceleration_features(X, cx, cy)
        X = add_path_features(X, cx, cy)
        X = add_statistical_features(X, cx, cy)
        
        if len(cx) >= 600:
            X = add_territorial_features(X, cx, cy)
        if len(cx) >= 1800:
            X = add_temporal_features(X, cx, cy)
    
    X = add_postural_features(X, single_mouse, available_body_parts)
    X = add_micromovement_features(X, single_mouse, available_body_parts)
    
    return X

def transform_pair_base(mouse_pair, body_parts_tracked):
    """Base pair features"""
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)
    
    X = pd.DataFrame({
        f"12+{p1}+{p2}": np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.product(body_parts_tracked, repeat=2) 
        if p1 in avail_A and p2 in avail_B
    }, dtype=np.float32)
    X = X.reindex(columns=[f"12+{p1}+{p2}" for p1, p2 in itertools.product(body_parts_tracked, repeat=2)], copy=False)

    if ('A', 'ear_left') in mouse_pair.columns and ('B', 'ear_left') in mouse_pair.columns:
        shA = mouse_pair['A']['ear_left'].shift(10)
        shB = mouse_pair['B']['ear_left'].shift(10)
        speeds = pd.DataFrame({
            'sp_A': np.square(mouse_pair['A']['ear_left'] - shA).sum(axis=1, skipna=False),
            'sp_AB': np.square(mouse_pair['A']['ear_left'] - shB).sum(axis=1, skipna=False),
            'sp_B': np.square(mouse_pair['B']['ear_left'] - shB).sum(axis=1, skipna=False),
        }, dtype=np.float32)
        X = pd.concat([X, speeds], axis=1)
    
    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X['elong'] = (X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)).astype(np.float32)
    
    if all(p in avail_A for p in ['nose', 'tail_base']) and all(p in avail_B for p in ['nose', 'tail_base']):
        dir_A = mouse_pair['A']['nose'] - mouse_pair['A']['tail_base']
        dir_B = mouse_pair['B']['nose'] - mouse_pair['B']['tail_base']
        X['rel_ori'] = ((dir_A['x'] * dir_B['x'] + dir_A['y'] * dir_B['y']) / (
            pd.Series(np.sqrt(dir_A['x']**2 + dir_A['y']**2), index=dir_A.index) * pd.Series(np.sqrt(dir_B['x']**2 + dir_B['y']**2), index=dir_B.index) + 1e-6)).astype(np.float32)
    
    if all(p in avail_A for p in ['nose']) and all(p in avail_B for p in ['nose']):
        cur = np.square(mouse_pair['A']['nose'] - mouse_pair['B']['nose']).sum(axis=1, skipna=False)
        shA_n = mouse_pair['A']['nose'].shift(10)
        shB_n = mouse_pair['B']['nose'].shift(10)
        past = np.square(shA_n - shB_n).sum(axis=1, skipna=False)
        X['appr'] = (cur - past).astype(np.float32)
    
    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd = pd.Series(np.sqrt((mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x'])**2 +
                    (mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y'])**2), index=mouse_pair.index)
        X['v_cls'] = (cd < 5.0).astype(float).astype(np.float32)
        X['cls'] = ((cd >= 5.0) & (cd < 15.0)).astype(float).astype(np.float32)
        X['med'] = ((cd >= 15.0) & (cd < 30.0)).astype(float).astype(np.float32)
        X['far'] = (cd >= 30.0).astype(float).astype(np.float32)
    
    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd_full = np.square(mouse_pair['A']['body_center'] - mouse_pair['B']['body_center']).sum(axis=1, skipna=False)
        
        for w in [5, 15, 30, 60]:
            X[f'd_m{w}'] = cd_full.rolling(w, min_periods=1, center=True).mean().astype(np.float32)
            X[f'd_s{w}'] = cd_full.rolling(w, min_periods=1, center=True).std().astype(np.float32)
            X[f'd_mn{w}'] = cd_full.rolling(w, min_periods=1, center=True).min().astype(np.float32)
            X[f'd_mx{w}'] = cd_full.rolling(w, min_periods=1, center=True).max().astype(np.float32)
            
            d_var = cd_full.rolling(w, min_periods=1, center=True).var()
            X[f'int{w}'] = (1 / (1 + d_var)).astype(np.float32)
            
            Axd = mouse_pair['A']['body_center']['x'].diff()
            Ayd = mouse_pair['A']['body_center']['y'].diff()
            Bxd = mouse_pair['B']['body_center']['x'].diff()
            Byd = mouse_pair['B']['body_center']['y'].diff()
            coord = Axd * Bxd + Ayd * Byd
            X[f'co_m{w}'] = coord.rolling(w, min_periods=1, center=True).mean().astype(np.float32)
            X[f'co_s{w}'] = coord.rolling(w, min_periods=1, center=True).std().astype(np.float32)
    
    if 'nose' in avail_A and 'nose' in avail_B:
        nn = pd.Series(np.sqrt((mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x'])**2 +
                    (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y'])**2), index=mouse_pair.index)
        for lag in [10, 20, 40]:
            X[f'nn_lg{lag}'] = nn.shift(lag).astype(np.float32)
            X[f'nn_ch{lag}'] = (nn - nn.shift(lag)).astype(np.float32)
            is_cl = (nn < 10.0).astype(float)
            X[f'cl_ps{lag}'] = is_cl.rolling(lag, min_periods=1).mean().astype(np.float32)
    
    if 'body_center' in avail_A and 'body_center' in avail_B:
        Avx = mouse_pair['A']['body_center']['x'].diff()
        Avy = mouse_pair['A']['body_center']['y'].diff()
        Bvx = mouse_pair['B']['body_center']['x'].diff()
        Bvy = mouse_pair['B']['body_center']['y'].diff()
        val = (Avx * Bvx + Avy * Bvy) / (pd.Series(np.sqrt(Avx**2 + Avy**2), index=Avx.index) * pd.Series(np.sqrt(Bvx**2 + Bvy**2), index=Bvx.index) + 1e-6)
        
        for off in [-20, -10, 0, 10, 20]:
            X[f'va_{off}'] = val.shift(-off).astype(np.float32)
        
        X['int_con'] = (cd_full.rolling(30, min_periods=1, center=True).std() / (cd_full.rolling(30, min_periods=1, center=True).mean() + 1e-6)).astype(np.float32)
        X = add_interaction_features(X, mouse_pair, avail_A, avail_B)
    
    return X, avail_A, avail_B

def transform_pair_model1(mouse_pair, body_parts_tracked):
    X, avail_A, avail_B = transform_pair_base(mouse_pair, body_parts_tracked)
    if 'body_center' in avail_A and 'body_center' in avail_B:
        X = add_social_memory_features(X, 
                                       mouse_pair['A']['body_center']['x'], mouse_pair['A']['body_center']['y'],
                                       mouse_pair['B']['body_center']['x'], mouse_pair['B']['body_center']['y'])
    return X

def transform_pair_model2(mouse_pair, body_parts_tracked):
    X, _, _ = transform_pair_base(mouse_pair, body_parts_tracked)
    return X

def transform_pair_model3(mouse_pair, body_parts_tracked):
    X, _, _ = transform_pair_base(mouse_pair, body_parts_tracked)
    return X

def transform_pair_model4(mouse_pair, body_parts_tracked):
    X, avail_A, avail_B = transform_pair_base(mouse_pair, body_parts_tracked)
    X = add_pair_social_features(X, mouse_pair, avail_A, avail_B)
    return X

def transform_pair_model5(mouse_pair, body_parts_tracked):
    X, avail_A, avail_B = transform_pair_base(mouse_pair, body_parts_tracked)
    X = add_pair_micromovement_features(X, mouse_pair, avail_A, avail_B)
    return X

def transform_pair_generalized(mouse_pair, body_parts_tracked):
    """Comprehensive features for generalized models"""
    X, avail_A, avail_B = transform_pair_base(mouse_pair, body_parts_tracked)
    
    if 'body_center' in avail_A and 'body_center' in avail_B:
        X = add_social_memory_features(X, 
                                       mouse_pair['A']['body_center']['x'], mouse_pair['A']['body_center']['y'],
                                       mouse_pair['B']['body_center']['x'], mouse_pair['B']['body_center']['y'])
        X = add_pair_social_features(X, mouse_pair, avail_A, avail_B)
    
    X = add_pair_micromovement_features(X, mouse_pair, avail_A, avail_B)
    
    return X

# ==================== ENSEMBLE TRAINING - MEMORY OPTIMIZED ====================

def submit_ensemble(body_parts_tracked_str, switch_tr, data_list, label, meta):
    """7-model ensemble - MEMORY OPTIMIZED"""
    
    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    
    # Transform functions
    if switch_tr == 'single':
        transform_funcs = [
            transform_single_model1,
            transform_single_model2,
            transform_single_model3,
            transform_single_model4,
            transform_single_model5,
            transform_single_generalized,
            transform_single_generalized,
        ]
    else:
        transform_funcs = [
            transform_pair_model1,
            transform_pair_model2,
            transform_pair_model3,
            transform_pair_model4,
            transform_pair_model5,
            transform_pair_generalized,
            transform_pair_generalized,
        ]
    
    # Build 7 models
    models = []
    models.append(make_pipeline(SimpleImputer(), StratifiedSubsetClassifier(
        lightgbm.LGBMClassifier(n_estimators=225, learning_rate=0.07, min_child_samples=40,
                                num_leaves=31, subsample=0.8, colsample_bytree=0.8, verbose=-1), 100000)))
    models.append(make_pipeline(SimpleImputer(), StratifiedSubsetClassifier(
        lightgbm.LGBMClassifier(n_estimators=150, learning_rate=0.1, min_child_samples=20,
                                num_leaves=63, max_depth=8, subsample=0.7, colsample_bytree=0.9,
                                reg_alpha=0.1, reg_lambda=0.1, verbose=-1), 80000)))
    models.append(make_pipeline(SimpleImputer(), StratifiedSubsetClassifier(
        lightgbm.LGBMClassifier(n_estimators=100, learning_rate=0.05, min_child_samples=30,
                                num_leaves=127, max_depth=10, subsample=0.75, verbose=-1), 60000)))
    if XGBOOST_AVAILABLE:
        models.append(make_pipeline(SimpleImputer(), StratifiedSubsetClassifier(
            XGBClassifier(n_estimators=180, learning_rate=0.08, max_depth=6,
                         min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                         tree_method='hist', verbosity=0), 85000)))
    if CATBOOST_AVAILABLE:
        models.append(make_pipeline(SimpleImputer(), StratifiedSubsetClassifier(
            CatBoostClassifier(iterations=120, learning_rate=0.1, depth=6,
                             verbose=False, allow_writing_files=False), 70000)))
    models.append(make_pipeline(SimpleImputer(), StratifiedSubsetClassifier(
        RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=20,
                              min_samples_leaf=10, max_features='sqrt', n_jobs=-1, random_state=42), 80000)))
    models.append(make_pipeline(SimpleImputer(), StratifiedSubsetClassifier(
        ExtraTreesClassifier(n_estimators=150, max_depth=15, min_samples_split=20,
                            min_samples_leaf=10, max_features='sqrt', n_jobs=-1, random_state=43), 80000)))
    
    # Train all models
    model_list = []
    for action in label.columns:
        action_mask = ~label[action].isna().values
        y_action = label[action][action_mask].values.astype(np.int8)

        if not (y_action == 0).all() and y_action.sum() >= 5:
            trained = []
            for i, (m, transform_func) in enumerate(zip(models, transform_funcs)):
                X_tr = transform_func(data_list, body_parts_tracked)
                m_clone = clone(m)
                m_clone.fit(X_tr[action_mask], y_action)
                trained.append((m_clone, transform_func))
                del X_tr
                gc.collect()
            model_list.append((action, trained))
    
    del data_list
    gc.collect()

    test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
    generator = generate_mouse_data(test_subset, 'test',
                                    generate_single=(switch_tr == 'single'), 
                                    generate_pair=(switch_tr == 'pair'))
    
    if verbose: print(f"n_videos: {len(test_subset)}, n_models: {len(models)}")
    
    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr
        try:
            pred = pd.DataFrame(index=meta_te.video_frame.values, dtype=np.float32)
            for action, trained_models in model_list:
                if action in actions_te:
                    probs = []
                    for m, transform_func in trained_models:
                        X_te = transform_func(data_te, body_parts_tracked)
                        probs.append(m.predict_proba(X_te)[:, 1])
                        del X_te
                        gc.collect()
                    pred[action] = np.mean(probs, axis=0).astype(np.float32)
            
            del data_te
            gc.collect()
            
            if pred.shape[1] != 0:
                sub_part = predict_multiclass_adaptive(pred, meta_te, action_thresholds)
                submission_list.append(sub_part)
            else:
                if verbose: print(f"  ERROR: no training data")
        except Exception as e:
            if verbose: print(f'  ERROR: {str(e)[:50]}')
            try:
                del data_te
            except:
                pass
            gc.collect()

def robustify(submission, dataset, traintest, traintest_directory=None):
    """Robustness post-processing"""
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    submission = submission[submission.start_frame < submission.stop_frame]
    
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
    submission = pd.concat(group_list) if group_list else submission

    s_list = []
    for idx, row in dataset.iterrows():
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'):
            continue
        video_id = row['video_id']
        if (submission.video_id == video_id).any():
            continue

        if verbose: print(f"Video {video_id} has no predictions")
        
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
    
        vid_behaviors = eval(row['behaviors_labeled'])
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
    
        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1
    
        for (agent, target), actions in vid_behaviors.groupby(['agent', 'target']):
            batch_len = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, (_, action_row) in enumerate(actions.iterrows()):
                batch_start = start_frame + i * batch_len
                batch_stop = min(batch_start + batch_len, stop_frame)
                s_list.append((video_id, agent, target, action_row['action'], batch_start, batch_stop))

    if len(s_list) > 0:
        submission = pd.concat([
            submission,
            pd.DataFrame(s_list, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
        ])

    submission = submission.reset_index(drop=True)
    return submission

# ==================== MAIN LOOP ====================

submission_list = []

print(f"XGBoost: {XGBOOST_AVAILABLE}, CatBoost: {CATBOOST_AVAILABLE}\n")

for section in range(1, len(body_parts_tracked_list)):
    body_parts_tracked_str = body_parts_tracked_list[section]
    try:
        body_parts_tracked = json.loads(body_parts_tracked_str)
        print(f"{section}. Processing: {len(body_parts_tracked)} body parts")
        if len(body_parts_tracked) > 5:
            body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    
        train_subset = train[train.body_parts_tracked == body_parts_tracked_str]
        single_list, single_label_list, single_meta_list = [], [], []
        pair_list, pair_label_list, pair_meta_list = [], [], []
    
        for switch, data, meta, label in generate_mouse_data(train_subset, 'train'):
            if switch == 'single':
                single_list.append(data)
                single_meta_list.append(meta)
                single_label_list.append(label)
            else:
                pair_list.append(data)
                pair_meta_list.append(meta)
                pair_label_list.append(label)
    
        if len(single_list) > 0:
            single_mouse = pd.concat(single_list)
            single_label = pd.concat(single_label_list)
            single_meta = pd.concat(single_meta_list)
            del single_list, single_label_list, single_meta_list
            gc.collect()
            
            print(f"  Single: training 7-model ensemble")
            submit_ensemble(body_parts_tracked_str, 'single', single_mouse, single_label, single_meta)
            del single_mouse, single_label, single_meta
            gc.collect()
                
        if len(pair_list) > 0:
            mouse_pair = pd.concat(pair_list)
            pair_label = pd.concat(pair_label_list)
            pair_meta = pd.concat(pair_meta_list)
            del pair_list, pair_label_list, pair_meta_list
            gc.collect()
        
            print(f"  Pair: training 7-model ensemble")
            submit_ensemble(body_parts_tracked_str, 'pair', mouse_pair, pair_label, pair_meta)
            del mouse_pair, pair_label, pair_meta
            gc.collect()
                
    except Exception as e:
        print(f'***Exception*** {str(e)[:100]}')
    
    gc.collect()
    print()

# Final submission
if len(submission_list) > 0:
    submission = pd.concat(submission_list)
else:
    submission = pd.DataFrame({
        'video_id': [438887472],
        'agent_id': ['mouse1'],
        'target_id': ['self'],
        'action': ['rear'],
        'start_frame': [278],
        'stop_frame': [500]
    })

submission_robust = robustify(submission, test, 'test')
submission_robust.index.name = 'row_id'
submission_robust.to_csv('submission.csv')
print(f"\nSubmission created: {len(submission_robust)} predictions")

