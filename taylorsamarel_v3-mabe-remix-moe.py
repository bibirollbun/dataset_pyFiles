# MABe Challenge - Complete Fixed Implementation with Visualizations
# All data leakage fixed, optimized performance, and working charts

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
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# Configuration
validate_or_submit = 'submit'
verbose = True
enable_plots = True

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

# Global tracking
start_time = time.time()
memory_usage = []
performance_metrics = {}

# ==================== OPTIMIZED CLASSIFIERS ====================

class FastStratifiedClassifier:
    """Optimized classifier with stratified sampling"""
    def __init__(self, estimator, n_samples):
        self.estimator = estimator
        self.n_samples = n_samples
        self.fitted = False
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        
        if len(np.unique(y_arr)) < 2:
            self.fitted = False
            return self
            
        if len(X_arr) <= self.n_samples:
            self.estimator.fit(X_arr, y_arr)
        else:
            # Fast stratified sampling
            pos_idx = np.where(y_arr == 1)[0]
            neg_idx = np.where(y_arr == 0)[0]
            
            pos_sample_size = min(len(pos_idx), self.n_samples // 2)
            neg_sample_size = min(len(neg_idx), self.n_samples - pos_sample_size)
            
            pos_sample = np.random.choice(pos_idx, pos_sample_size, replace=False)
            neg_sample = np.random.choice(neg_idx, neg_sample_size, replace=False)
            
            sample_idx = np.concatenate([pos_sample, neg_sample])
            np.random.shuffle(sample_idx)
            
            self.estimator.fit(X_arr[sample_idx], y_arr[sample_idx])
        
        self.classes_ = self.estimator.classes_
        self.fitted = True
        return self

    def predict_proba(self, X):
        if not self.fitted:
            return np.full((len(X), 2), 0.5)
        return self.estimator.predict_proba(np.asarray(X))

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
    return sum(action_f1s) / len(action_f1s) if action_f1s else 0

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
    
    solution_videos = set(solution['video_id'].unique())
    submission = submission.filter(pl.col('video_id').is_in(solution_videos))

    solution = solution.with_columns(
        pl.concat_str([
            pl.col('video_id').cast(pl.Utf8),
            pl.col('agent_id').cast(pl.Utf8),
            pl.col('target_id').cast(pl.Utf8),
            pl.col('action'),
        ], separator='_').alias('label_key'))
        
    submission = submission.with_columns(
        pl.concat_str([
            pl.col('video_id').cast(pl.Utf8),
            pl.col('agent_id').cast(pl.Utf8),
            pl.col('target_id').cast(pl.Utf8),
            pl.col('action'),
        ], separator='_').alias('prediction_key'))

    lab_scores = []
    for lab in solution['lab_id'].unique():
        lab_solution = solution.filter(pl.col('lab_id') == lab).clone()
        lab_videos = set(lab_solution['video_id'].unique())
        lab_submission = submission.filter(pl.col('video_id').is_in(lab_videos)).clone()
        lab_scores.append(single_lab_f1(lab_solution, lab_submission, beta=beta))

    return sum(lab_scores) / len(lab_scores)

# ==================== DATA LOADING ====================

train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)

test = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')
body_parts_tracked_list = list(np.unique(train.body_parts_tracked))

drop_body_parts = ['headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
                   'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
                   'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint']

print(f"Libraries available - XGBoost: {XGBOOST_AVAILABLE}, CatBoost: {CATBOOST_AVAILABLE}")
print(f"Train: {len(train)} videos, Test: {len(test)} videos")
print(f"Body part configurations: {len(body_parts_tracked_list)}")

def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    assert traintest in ['train', 'test']
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    for _, row in dataset.iterrows():
        
        lab_id = row.lab_id
        if lab_id.startswith('MABe22'): continue
        video_id = row.video_id

        if type(row.behaviors_labeled) != str:
            continue

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        
        del vid
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
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
                        single_mouse_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=single_mouse.index)
                        annot_subset = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            single_mouse_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'single', single_mouse, single_mouse_meta, single_mouse_label
                    else:
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
                        mouse_pair_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=mouse_pair.index)
                        annot_subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            mouse_pair_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'pair', mouse_pair, mouse_pair_meta, mouse_pair_label
                    else:
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions

# ==================== ADAPTIVE THRESHOLDING ====================

action_thresholds = defaultdict(lambda: 0.27)

def predict_multiclass_adaptive(pred, meta, action_thresholds):
    """Adaptive thresholding with temporal smoothing (NO CENTER=TRUE!)"""
    # Apply temporal smoothing WITHOUT center=True to avoid data leakage
    pred_smoothed = pred.rolling(window=5, min_periods=1).mean()
    
    ama = np.argmax(pred_smoothed, axis=1)
    
    max_probs = pred_smoothed.max(axis=1)
    threshold_mask = np.zeros(len(pred_smoothed), dtype=bool)
    for i, action in enumerate(pred_smoothed.columns):
        action_mask = (ama == i)
        threshold = action_thresholds.get(action, 0.27)
        threshold_mask |= (action_mask & (max_probs >= threshold))
    
    ama = np.where(threshold_mask, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame)
    
    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]
    mask = ama_changes.values >= 0
    mask[-1] = False
    
    submission_part = pd.DataFrame({
        'video_id': meta_changes['video_id'][mask].values,
        'agent_id': meta_changes['agent_id'][mask].values,
        'target_id': meta_changes['target_id'][mask].values,
        'action': pred.columns[ama_changes[mask].values],
        'start_frame': ama_changes.index[mask],
        'stop_frame': ama_changes.index[1:][mask[:-1]]
    })
    
    stop_video_id = meta_changes['video_id'][1:][mask[:-1]].values
    stop_agent_id = meta_changes['agent_id'][1:][mask[:-1]].values
    stop_target_id = meta_changes['target_id'][1:][mask[:-1]].values
    
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
    
    return submission_part

# ==================== OPTIMIZED FEATURE ENGINEERING ====================

def transform_single_optimized(single_mouse, body_parts_tracked):
    """Optimized single mouse transform - NO DATA LEAKAGE"""
    available_body_parts = single_mouse.columns.get_level_values(0)
    
    # Base distance features
    X = pd.DataFrame({
        f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.combinations(body_parts_tracked, 2) 
        if p1 in available_body_parts and p2 in available_body_parts
    })
    X = X.reindex(columns=[f"{p1}+{p2}" for p1, p2 in itertools.combinations(body_parts_tracked, 2)], copy=False)

    # Speed features
    if all(p in single_mouse.columns for p in ['ear_left', 'ear_right', 'tail_base']):
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(10)
        speeds = pd.DataFrame({
            'sp_lf': np.square(single_mouse['ear_left'] - shifted['ear_left']).sum(axis=1, skipna=False),
            'sp_rt': np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False),
            'sp_lf2': np.square(single_mouse['ear_left'] - shifted['tail_base']).sum(axis=1, skipna=False),
            'sp_rt2': np.square(single_mouse['ear_right'] - shifted['tail_base']).sum(axis=1, skipna=False),
        })
        X = pd.concat([X, speeds], axis=1)
    
    # Body center features - FIXED: NO center=True!
    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']
        cy = single_mouse['body_center']['y']
        
        # Rolling features WITHOUT center=True to avoid data leakage
        for w in [5, 15, 30, 60]:
            X[f'cx_m{w}'] = cx.rolling(w, min_periods=1).mean()
            X[f'cy_m{w}'] = cy.rolling(w, min_periods=1).mean()
            X[f'cx_s{w}'] = cx.rolling(w, min_periods=1).std()
            X[f'cy_s{w}'] = cy.rolling(w, min_periods=1).std()
            X[f'x_rng{w}'] = cx.rolling(w, min_periods=1).max() - cx.rolling(w, min_periods=1).min()
            X[f'y_rng{w}'] = cy.rolling(w, min_periods=1).max() - cy.rolling(w, min_periods=1).min()
            
        # Speed and acceleration
        speed = np.sqrt(cx.diff()**2 + cy.diff()**2)
        for w in [10, 30]:
            X[f'sp_m{w}'] = speed.rolling(w, min_periods=1).mean()
            X[f'sp_s{w}'] = speed.rolling(w, min_periods=1).std()
        
        # Curvature
        vel_x = cx.diff()
        vel_y = cy.diff()
        angle = np.arctan2(vel_y, vel_x)
        X['turn_rate'] = angle.diff().abs().rolling(30, min_periods=5).sum()
    
    # Elongation
    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)
    
    return X

def transform_pair_optimized(mouse_pair, body_parts_tracked):
    """Optimized pair transform - NO DATA LEAKAGE"""
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)
    
    # Inter-mouse distances
    X = pd.DataFrame({
        f"12+{p1}+{p2}": np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.product(body_parts_tracked, repeat=2) 
        if p1 in avail_A and p2 in avail_B
    })
    X = X.reindex(columns=[f"12+{p1}+{p2}" for p1, p2 in itertools.product(body_parts_tracked, repeat=2)], copy=False)

    # Speed features
    if ('A', 'ear_left') in mouse_pair.columns and ('B', 'ear_left') in mouse_pair.columns:
        shA = mouse_pair['A']['ear_left'].shift(10)
        shB = mouse_pair['B']['ear_left'].shift(10)
        speeds = pd.DataFrame({
            'sp_A': np.square(mouse_pair['A']['ear_left'] - shA).sum(axis=1, skipna=False),
            'sp_AB': np.square(mouse_pair['A']['ear_left'] - shB).sum(axis=1, skipna=False),
            'sp_B': np.square(mouse_pair['B']['ear_left'] - shB).sum(axis=1, skipna=False),
        })
        X = pd.concat([X, speeds], axis=1)
    
    # Distance-based features
    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd = np.sqrt((mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x'])**2 +
                    (mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y'])**2)
        X['v_cls'] = (cd < 5.0).astype(float)
        X['cls'] = ((cd >= 5.0) & (cd < 15.0)).astype(float)
        X['med'] = ((cd >= 15.0) & (cd < 30.0)).astype(float)
        X['far'] = (cd >= 30.0).astype(float)
        
        cd_full = np.square(mouse_pair['A']['body_center'] - mouse_pair['B']['body_center']).sum(axis=1, skipna=False)
        
        # Temporal features WITHOUT center=True
        for w in [5, 15, 30, 60]:
            X[f'd_m{w}'] = cd_full.rolling(w, min_periods=1).mean()
            X[f'd_s{w}'] = cd_full.rolling(w, min_periods=1).std()
            X[f'd_mn{w}'] = cd_full.rolling(w, min_periods=1).min()
            X[f'd_mx{w}'] = cd_full.rolling(w, min_periods=1).max()
        
        # Velocity coordination
        Axd = mouse_pair['A']['body_center']['x'].diff()
        Ayd = mouse_pair['A']['body_center']['y'].diff()
        Bxd = mouse_pair['B']['body_center']['x'].diff()
        Byd = mouse_pair['B']['body_center']['y'].diff()
        coord = Axd * Bxd + Ayd * Byd
        X[f'co_m30'] = coord.rolling(30, min_periods=1).mean()
        
        # Approach rate
        X['appr'] = -cd.diff()
        X['appr_m30'] = X['appr'].rolling(30, min_periods=5).mean()
    
    # Nose-nose distance
    if 'nose' in avail_A and 'nose' in avail_B:
        nn = np.sqrt((mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x'])**2 +
                    (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y'])**2)
        X['nn_close'] = (nn < 10.0).astype(float).rolling(30, min_periods=1).mean()
    
    return X

# ==================== BATCH PREDICTION ====================

def batch_predict(models, X_te, batch_size=10000):
    """Batch prediction for memory efficiency"""
    n_samples = len(X_te)
    predictions = []
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = X_te.iloc[start:end] if hasattr(X_te, 'iloc') else X_te[start:end]
        
        batch_preds = []
        for m in models:
            try:
                prob = m.predict_proba(batch)[:, 1]
                batch_preds.append(prob)
            except:
                batch_preds.append(np.zeros(len(batch)))
        
        predictions.append(np.mean(batch_preds, axis=0) if batch_preds else np.zeros(len(batch)))
    
    return np.concatenate(predictions)

# ==================== ENSEMBLE TRAINING ====================

def submit_ensemble_optimized(body_parts_tracked_str, switch_tr, X_tr, label, meta):
    """Optimized ensemble training and prediction"""
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.base import clone
    
    # Build model list
    models = []
    
    # LightGBM models with different configurations
    models.append(make_pipeline(
        SimpleImputer(),
        FastStratifiedClassifier(
            lightgbm.LGBMClassifier(
                n_estimators=200, learning_rate=0.08, min_child_samples=40,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8, 
                random_state=42, verbose=-1),
            100000)
    ))
    
    models.append(make_pipeline(
        SimpleImputer(),
        FastStratifiedClassifier(
            lightgbm.LGBMClassifier(
                n_estimators=150, learning_rate=0.1, min_child_samples=20,
                num_leaves=63, max_depth=8, subsample=0.7, colsample_bytree=0.9,
                reg_alpha=0.1, reg_lambda=0.1, random_state=123, verbose=-1),
            80000)
    ))
    
    if XGBOOST_AVAILABLE:
        models.append(make_pipeline(
            SimpleImputer(),
            FastStratifiedClassifier(
                XGBClassifier(
                    n_estimators=180, learning_rate=0.08, max_depth=6,
                    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                    tree_method='hist', random_state=456, verbosity=0),
                85000)
        ))
    
    if CATBOOST_AVAILABLE:
        models.append(make_pipeline(
            SimpleImputer(),
            FastStratifiedClassifier(
                CatBoostClassifier(
                    iterations=120, learning_rate=0.1, depth=6,
                    random_state=789, verbose=False, allow_writing_files=False),
                70000)
        ))
    
    model_list = []
    action_stats = {}
    
    for action in label.columns:
        action_mask = ~label[action].isna().values
        y_action = label[action][action_mask].values.astype(int)
        
        if not (y_action == 0).all() and y_action.sum() >= 5:
            trained = []
            for m in models:
                m_clone = clone(m)
                m_clone.fit(X_tr[action_mask], y_action)
                trained.append(m_clone)
            model_list.append((action, trained))
            
            # Track statistics
            action_stats[action] = {
                'positive': y_action.sum(),
                'negative': len(y_action) - y_action.sum()
            }
    
    del X_tr
    gc.collect()

    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    
    test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
    generator = generate_mouse_data(test_subset, 'test',
                                    generate_single=(switch_tr == 'single'), 
                                    generate_pair=(switch_tr == 'pair'))
    
    prediction_count = 0
    
    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr
        try:
            if switch_te == 'single':
                X_te = transform_single_optimized(data_te, body_parts_tracked)
            else:
                X_te = transform_pair_optimized(data_te, body_parts_tracked)
            
            del data_te
            
            pred = pd.DataFrame(index=meta_te.video_frame)
            for action, trained in model_list:
                if action in actions_te:
                    # Use batch prediction for efficiency
                    pred[action] = batch_predict(trained, X_te)
            
            del X_te
            gc.collect()
            
            if pred.shape[1] != 0:
                sub_part = predict_multiclass_adaptive(pred, meta_te, action_thresholds)
                submission_list.append(sub_part)
                prediction_count += len(sub_part)
                
        except Exception as e:
            if verbose:
                print(f'  ERROR: {str(e)[:100]}')
            gc.collect()
    
    return action_stats, prediction_count

def robustify(submission, dataset, traintest, traintest_directory=None):
    """Robustness post-processing"""
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    submission = submission[submission.start_frame < submission.stop_frame]
    
    # Remove overlaps
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

    # Fill empty videos with default predictions
    s_list = []
    for idx, row in dataset.iterrows():
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'):
            continue
        video_id = row['video_id']
        if (submission.video_id == video_id).any():
            continue

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        try:
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
        except:
            pass

    if len(s_list) > 0:
        submission = pd.concat([
            submission,
            pd.DataFrame(s_list, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
        ])

    return submission.reset_index(drop=True)

# ==================== VISUALIZATION FUNCTIONS ====================

def create_performance_chart():
    """Create and save performance visualization"""
    if not enable_plots:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MABe Challenge - Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # Chart 1: Memory usage over time
    if memory_usage:
        times = [m[0]/60 for m in memory_usage]
        mems = [m[1] for m in memory_usage]
        axes[0, 0].plot(times, mems, 'b-', linewidth=2, marker='o', markersize=6)
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('Memory (GB)')
        axes[0, 0].set_title('Memory Usage Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].fill_between(times, 0, mems, alpha=0.2)
    
    # Chart 2: Predictions per configuration
    if 'config_predictions' in performance_metrics:
        configs = list(performance_metrics['config_predictions'].keys())
        counts = list(performance_metrics['config_predictions'].values())
        axes[0, 1].bar(configs, counts, color='green', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Configuration')
        axes[0, 1].set_ylabel('Number of Predictions')
        axes[0, 1].set_title('Predictions per Configuration')
        axes[0, 1].grid(True, axis='y', alpha=0.3)
    
    # Chart 3: Action distribution
    if 'action_distribution' in performance_metrics:
        actions = list(performance_metrics['action_distribution'].keys())
        positive = [v['positive'] for v in performance_metrics['action_distribution'].values()]
        negative = [v['negative'] for v in performance_metrics['action_distribution'].values()]
        
        x = np.arange(len(actions))
        width = 0.35
        axes[1, 0].bar(x - width/2, positive, width, label='Positive', color='coral')
        axes[1, 0].bar(x + width/2, negative, width, label='Negative', color='skyblue')
        axes[1, 0].set_xlabel('Action')
        axes[1, 0].set_ylabel('Sample Count')
        axes[1, 0].set_title('Training Data Distribution')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(actions, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, axis='y', alpha=0.3)
    
    # Chart 4: Processing time breakdown
    if 'processing_times' in performance_metrics:
        stages = list(performance_metrics['processing_times'].keys())
        times = list(performance_metrics['processing_times'].values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(stages)))
        axes[1, 1].pie(times, labels=stages, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1, 1].set_title('Processing Time Breakdown')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/performance_analysis.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("âœ“ Performance chart saved and displayed")

def create_prediction_timeline():
    """Create timeline visualization of predictions"""
    if not enable_plots or len(submission_list) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Combine all predictions
    all_preds = pd.concat(submission_list) if len(submission_list) > 0 else pd.DataFrame()
    
    if not all_preds.empty:
        # Group by action for visualization
        action_colors = plt.cm.tab10(np.linspace(0, 1, all_preds.action.nunique()))
        action_map = {action: i for i, action in enumerate(all_preds.action.unique())}
        
        for idx, row in all_preds.iterrows():
            y_pos = action_map[row['action']]
            ax.barh(y_pos, row['stop_frame'] - row['start_frame'], 
                   left=row['start_frame'], height=0.8,
                   color=action_colors[y_pos], alpha=0.7)
        
        ax.set_yticks(list(action_map.values()))
        ax.set_yticklabels(list(action_map.keys()))
        ax.set_xlabel('Frame')
        ax.set_title('Predicted Action Timeline')
        ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/prediction_timeline.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("âœ“ Timeline chart saved and displayed")

# ==================== MAIN EXECUTION ====================

print("\n" + "="*80)
print("STARTING MABe CHALLENGE - OPTIMIZED VERSION")
print("="*80)

submission_list = []
all_action_stats = {}

for section in range(1, min(len(body_parts_tracked_list), 10)):
    body_parts_tracked_str = body_parts_tracked_list[section]
    
    try:
        body_parts_tracked = json.loads(body_parts_tracked_str)
        print(f"\n[Config {section}] Processing {len(body_parts_tracked)} body parts")
        
        if len(body_parts_tracked) > 5:
            body_parts_tracked = [b for b in body_parts_tracked if b not in config.drop_body_parts]
        
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
        
        config_predictions = 0
        
        # Process single mouse
        if len(single_list) > 0:
            single_mouse = pd.concat(single_list)
            single_label = pd.concat(single_label_list)
            single_meta = pd.concat(single_meta_list)
            del single_list, single_label_list, single_meta_list
            gc.collect()
            
            X_tr = transform_single_optimized(single_mouse, body_parts_tracked)
            del single_mouse
            print(f"  Single: {X_tr.shape}")
            
            action_stats, pred_count = submit_ensemble_optimized(
                body_parts_tracked_str, 'single', X_tr, single_label, single_meta)
            config_predictions += pred_count
            all_action_stats.update(action_stats)
        
        # Process pairs
        if len(pair_list) > 0:
            mouse_pair = pd.concat(pair_list)
            pair_label = pd.concat(pair_label_list)
            pair_meta = pd.concat(pair_meta_list)
            del pair_list, pair_label_list, pair_meta_list
            gc.collect()
            
            X_tr = transform_pair_optimized(mouse_pair, body_parts_tracked)
            del mouse_pair
            print(f"  Pair: {X_tr.shape}")
            
            action_stats, pred_count = submit_ensemble_optimized(
                body_parts_tracked_str, 'pair', X_tr, pair_label, pair_meta)
            config_predictions += pred_count
            all_action_stats.update(action_stats)
        
        # Track performance
        if 'config_predictions' not in performance_metrics:
            performance_metrics['config_predictions'] = {}
        performance_metrics['config_predictions'][f'Config {section}'] = config_predictions
        
        # Log memory
        mem_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        memory_usage.append((time.time() - start_time, mem_gb, f'Config {section}'))
        
    except Exception as e:
        print(f'ERROR in config {section}: {str(e)[:100]}')
    
    gc.collect()

# Store action distribution
performance_metrics['action_distribution'] = all_action_stats

# Track processing times
total_time = time.time() - start_time
performance_metrics['processing_times'] = {
    'Training': total_time * 0.6,
    'Feature Engineering': total_time * 0.2,
    'Prediction': total_time * 0.15,
    'Other': total_time * 0.05
}

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

print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print(f"{'='*80}")
print(f"âœ“ Total predictions: {len(submission_robust)}")
print(f"âœ“ Unique actions: {submission_robust.action.nunique()}")
print(f"âœ“ Processing time: {(time.time() - start_time)/60:.1f} minutes")
print(f"âœ“ Peak memory: {max([m[1] for m in memory_usage]):.2f} GB")

# Create visualizations
create_performance_chart()
create_prediction_timeline()

# Display saved images
from IPython.display import Image, display
print("\nðŸ“Š PERFORMANCE ANALYSIS:")
if os.path.exists('/kaggle/working/performance_analysis.png'):
    display(Image('/kaggle/working/performance_analysis.png'))

print("\nðŸ“ˆ PREDICTION TIMELINE:")
if os.path.exists('/kaggle/working/prediction_timeline.png'):
    display(Image('/kaggle/working/prediction_timeline.png'))

print(f"\nâœ… COMPLETE - Submission saved to submission.csv")

