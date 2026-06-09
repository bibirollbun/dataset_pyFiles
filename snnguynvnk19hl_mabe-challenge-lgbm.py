from __future__ import annotations
import os
import gc
import json
import itertools
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

try:
    from lightgbm import LGBMClassifier
except Exception as e:
    raise RuntimeError("LightGBM is required.") from e


# ========================
# Global Configuration
# ========================
validate_or_submit = 'submit'
verbose = True
USE_HYSTERESIS_DECODER = False  

# ========================
# Logging
# ========================
def log(msg: str):
    if verbose:
        print(msg, flush=True)

# ========================
# Enhanced decoder parameters per action
# ========================
STATE_ACTIONS = {
    'rest', 'sniff', 'sniffface', 'sniffgenital', 'mount', 'intromit', 'reciprocalsniff'
}
EVENT_ACTIONS = {
    'approach', 'avoid', 'follow', 'chase', 'chaseattack', 'attack', 'defend', 'flinch',
    'tussle', 'dominance', 'escape', 'submit', 'attemptmount'
}

# Labs with many short labels (1-2 frames)
SHORT_LABEL_LABS = {'CRIM13', 'CalMS21_task1', 'CalMS21_task2', 'CalMS21_supplemental'}

DEFAULT_PER_ACTION = {
    'sniff':        dict(t_on=0.55, t_off=0.40, min_len=6,  merge_gap=6,  smooth_k=11),
    'sniffface':    dict(t_on=0.55, t_off=0.40, min_len=6,  merge_gap=6,  smooth_k=11),
    'sniffgenital': dict(t_on=0.60, t_off=0.45, min_len=8,  merge_gap=8,  smooth_k=13),
    'rest':         dict(t_on=0.60, t_off=0.45, min_len=12, merge_gap=8,  smooth_k=13),
    'mount':        dict(t_on=0.60, t_off=0.45, min_len=8,  merge_gap=6,  smooth_k=11),
    'intromit':     dict(t_on=0.65, t_off=0.50, min_len=6,  merge_gap=4,  smooth_k=9),
    'approach':     dict(t_on=0.60, t_off=0.45, min_len=4,  merge_gap=4,  smooth_k=9),
    'avoid':        dict(t_on=0.60, t_off=0.45, min_len=4,  merge_gap=4,  smooth_k=9),
    'follow':       dict(t_on=0.60, t_off=0.45, min_len=5,  merge_gap=5,  smooth_k=9),
    'chase':        dict(t_on=0.60, t_off=0.45, min_len=5,  merge_gap=5,  smooth_k=9),
    'chaseattack':  dict(t_on=0.62, t_off=0.48, min_len=5,  merge_gap=4,  smooth_k=9),
    'attack':       dict(t_on=0.65, t_off=0.50, min_len=3,  merge_gap=3,  smooth_k=7),
    'defend':       dict(t_on=0.62, t_off=0.48, min_len=3,  merge_gap=3,  smooth_k=7),
    'flinch':       dict(t_on=0.62, t_off=0.48, min_len=2,  merge_gap=2,  smooth_k=7),
}

# ========================
# Config
# ========================
@dataclass
class Config:
    data_root: Path = Path("/kaggle/input/MABe-mouse-behavior-detection")
    submission_file: str = "submission.csv"
    row_id_col: str = "row_id"
    max_train_samples_per_action: int = 100_000

    @property
    def train_csv(self) -> Path: return self.data_root / "train.csv"
    @property
    def test_csv(self) -> Path: return self.data_root / "test.csv"
    @property
    def train_track_dir(self) -> Path: return self.data_root / "train_tracking"
    @property
    def train_ann_dir(self) -> Path: return self.data_root / "train_annotation"
    @property
    def test_track_dir(self) -> Path: return self.data_root / "test_tracking"


CFG = Config()

# ========================
# Body parts to drop
# ========================
drop_body_parts = [
    'headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft',
    'headpiece_bottomfrontright', 'headpiece_topbackleft', 'headpiece_topbackright',
    'headpiece_topfrontleft', 'headpiece_topfrontright', 'spine_1', 'spine_2',
    'tail_middle_1', 'tail_middle_2', 'tail_midpoint'
]

# ========================
# Hysteresis Decoder
# ========================
def _smooth_probs(x, k):
    if k <= 1: return x
    pad = k // 2
    xx = np.pad(x, (pad, pad), mode='edge')
    w = np.ones(k) / k
    return np.convolve(xx, w, mode='valid')

def _decode_one_series(scores_df, actions, per_action):
    frames = scores_df.index.values
    current = None
    events = []
    last_end = {}

    t_on  = {a: per_action.get(a, {}).get('t_on',  0.55) for a in actions}
    t_off = {a: per_action.get(a, {}).get('t_off', 0.40) for a in actions}
    minl  = {a: per_action.get(a, {}).get('min_len', 4)  for a in actions}
    mgap  = {a: per_action.get(a, {}).get('merge_gap', 4) for a in actions}

    for i, f in enumerate(frames):
        row = scores_df.iloc[i].values
        a_idx = int(np.argmax(row))
        a = actions[a_idx]
        s = row[a_idx]

        if current is None:
            if s >= t_on[a]:
                current = [a, f]
        else:
            act_on, st = current
            if a == act_on and s >= t_off[act_on]:
                pass
            else:
                en = f
                if en > st and (en - st) >= minl[act_on]:
                    if act_on in last_end and (st - last_end[act_on][1]) <= mgap[act_on]:
                        last_end[act_on][1] = en
                    else:
                        last_end[act_on] = [st, en]
                current = None
                if s >= t_on[a]:
                    current = [a, f]

    if current is not None:
        a, st = current
        en = frames[-1] + 1
        if (en - st) >= minl[a]:
            if a in last_end and (st - last_end[a][1]) <= mgap[a]:
                last_end[a][1] = en
            else:
                last_end[a] = [st, en]

    out = []
    for a, (st, en) in last_end.items():
        out.append((a, st, en))
    return out

def _lab_params(lab_id: str, actions: List[str]) -> dict:
    per_action = {a: DEFAULT_PER_ACTION.get(a, dict(t_on=0.60, t_off=0.45, min_len=4, merge_gap=4, smooth_k=9))
                  for a in actions}
    if lab_id in SHORT_LABEL_LABS:
        for a in actions:
            if a in EVENT_ACTIONS:
                p = per_action[a].copy()
                p['min_len'] = max(1, min(p.get('min_len', 4), 2))
                p['smooth_k'] = max(5, p.get('smooth_k', 9) - 2)
                p['t_on']  = max(0.50, p.get('t_on', 0.60) - 0.05)
                p['t_off'] = max(0.35, p.get('t_off', 0.45) - 0.05)
                per_action[a] = p
    return per_action

def predict_multiclass_hysteresis(pred: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Enhanced decoder with hysteresis and lab-aware parameters"""
    if pred.shape[1] == 0:
        return pd.DataFrame(columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])

    pieces = []
    gcols = ['video_id', 'agent_id', 'target_id']
    idx_df = meta.assign(_idx=np.arange(len(meta)))
    
    for keys, sub_idx in idx_df.groupby(gcols)['_idx']:
        ii = sub_idx.values
        meta_g = meta.iloc[ii]
        pred_g = pred.iloc[ii].copy()
        actions = list(pred_g.columns)

        lab = str(meta_g['lab_id'].iloc[0]) if 'lab_id' in meta_g.columns else ''
        per_action = _lab_params(lab, actions)
        
        # Smooth each action
        sm = {}
        for a in actions:
            k = per_action.get(a, {}).get('smooth_k', 9)
            sm[a] = _smooth_probs(pred_g[a].values, k)
        sm = pd.DataFrame(sm, index=pred_g.index, columns=actions)
        
        evs = _decode_one_series(sm, actions, per_action)
        if not evs:
            continue
            
        g = meta_g.iloc[0]
        out = pd.DataFrame(evs, columns=['action', 'start_frame', 'stop_frame'])
        out.insert(0, 'target_id', g['target_id'])
        out.insert(0, 'agent_id', g['agent_id'])
        out.insert(0, 'video_id', g['video_id'])
        pieces.append(out)

    return pd.concat(pieces, ignore_index=True) if pieces else \
        pd.DataFrame(columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])

# ========================
# Generate mouse data 
# ========================
def generate_mouse_data(dataset: pd.DataFrame,
                        traintest: str,
                        traintest_directory: Optional[str] = None,
                        generate_single: bool = True,
                        generate_pair: bool = True):
    assert traintest in ['train', 'test']
    if traintest_directory is None:
        traintest_directory = f"{CFG.data_root}/{traintest}_tracking"

    for _, row in dataset.iterrows():
        lab_id = row.lab_id
        if traintest == 'train' and str(lab_id).startswith('MABe22'):
            continue

        video_id = row.video_id
        behaviors = row.behaviors_labeled
        if type(behaviors) != str:
            if verbose and traintest == 'test':
                print('No labeled behaviors:', lab_id, video_id, type(behaviors), behaviors)
            continue

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
        
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        if len(np.unique(vid.bodypart)) > 5:
            try:
                pvid = pvid.drop(columns=[('x', slice(None), b) for b in drop_body_parts], errors='ignore')
                pvid = pvid.drop(columns=[('y', slice(None), b) for b in drop_body_parts], errors='ignore')
            except Exception:
                pass

        if pvid.isna().any().any():
            if verbose and traintest == 'test':
                print('video with missing values', video_id, traintest, len(vid), 'frames')
        else:
            if verbose and traintest == 'test':
                print('video with all values', video_id, traintest, len(vid), 'frames')

        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        pvid = pvid / row.pix_per_cm_approx

        vid_behaviors = sorted(list({b.replace("'", "") for b in json.loads(row.behaviors_labeled)}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])

        if traintest == 'train':
            try:
                annot = pd.read_parquet(path.replace('train_tracking', 'train_annotation'))
            except FileNotFoundError:
                continue

        # SINGLE
        if generate_single:
            vid_b_single = vid_behaviors.query("target == 'self'")
            for mouse_id_str in np.unique(vid_b_single.agent):
                try:
                    mouse_id = int(mouse_id_str[-1])
                    vid_agent_actions = np.unique(vid_b_single.query("agent == @mouse_id_str").action)
                    single_mouse = pvid.loc[:, mouse_id]
                    single_mouse_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': mouse_id_str,
                        'target_id': 'self',
                        'video_frame': single_mouse.index,
                        'lab_id': lab_id,  # ADDED: lab_id for decoder
                    })
                    if traintest == 'train':
                        single_mouse_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=single_mouse.index)
                        annot_subset = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for i in range(len(annot_subset)):
                            r = annot_subset.iloc[i]
                            single_mouse_label.loc[r['start_frame']:r['stop_frame'], r.action] = 1.0
                        yield 'single', single_mouse, single_mouse_meta, single_mouse_label
                    else:
                        if verbose: print('- test single', video_id, mouse_id)
                        yield 'single', single_mouse, single_mouse_meta, vid_agent_actions
                except KeyError:
                    pass

        # PAIR
        if generate_pair:
            vid_b_pair = vid_behaviors.query("target != 'self'")
            if len(vid_b_pair) > 0:
                avail = np.unique(pvid.columns.get_level_values('mouse_id'))
                for agent, target in itertools.permutations(avail, 2):
                    agent_str = f"mouse{agent}"
                    target_str = f"mouse{target}"
                    vid_agent_actions = np.unique(vid_b_pair.query("(agent == @agent_str) & (target == @target_str)").action)
                    mouse_pair = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                    mouse_pair_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': agent_str,
                        'target_id': target_str,
                        'video_frame': mouse_pair.index,
                        'lab_id': lab_id,  # ADDED: lab_id for decoder
                    })
                    if traintest == 'train':
                        annot_subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                        mouse_pair_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=mouse_pair.index)
                        for i in range(len(annot_subset)):
                            r = annot_subset.iloc[i]
                            mouse_pair_label.loc[r['start_frame']:r['stop_frame'], r.action] = 1.0
                        yield 'pair', mouse_pair, mouse_pair_meta, mouse_pair_label
                    else:
                        if verbose: print('- test pair', video_id, agent, target)
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions

# ========================
# ENHANCED Feature Engineering
# ========================
def transform_single(single_mouse: pd.DataFrame, body_parts_tracked: List[str]) -> pd.DataFrame:
    """Enhanced feature engineering with long-range temporal dependencies"""
    available_body_parts = single_mouse.columns.get_level_values(0)
    
    # Base pairwise distances
    feats = {}
    for part1, part2 in itertools.combinations(body_parts_tracked, 2):
        if part1 in available_body_parts and part2 in available_body_parts:
            dif = single_mouse[part1] - single_mouse[part2]
            feats[f"{part1}+{part2}"] = np.square(dif).sum(axis=1, skipna=False)

    X = pd.DataFrame(feats, index=single_mouse.index)
    
    # Basic speed features
    try:
        if ('ear_left' in available_body_parts) and ('ear_right' in available_body_parts) and ('tail_base' in available_body_parts):
            shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(10)
            X2 = pd.DataFrame({
                'speed_left':  np.square(single_mouse['ear_left']  - shifted['ear_left']).sum(axis=1, skipna=False),
                'speed_right': np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False),
                'speed_left2': np.square(single_mouse['ear_left']  - shifted['tail_base']).sum(axis=1, skipna=False),
                'speed_right2': np.square(single_mouse['ear_right'] - shifted['tail_base']).sum(axis=1, skipna=False),
            }, index=single_mouse.index)
            X = pd.concat([X, X2], axis=1)
    except Exception:
        pass
    
    # LONG-RANGE TEMPORAL DEPENDENCIES (from files 2-3)
    time_windows = [5, 15, 30, 60]
    
    if 'body_center' in available_body_parts:
        center_x = single_mouse['body_center']['x']
        center_y = single_mouse['body_center']['y']
        
        for window in time_windows:
            # Rolling statistics
            X[f'center_x_mean_{window}'] = center_x.rolling(window, min_periods=1, center=True).mean()
            X[f'center_y_mean_{window}'] = center_y.rolling(window, min_periods=1, center=True).mean()
            X[f'center_x_std_{window}'] = center_x.rolling(window, min_periods=1, center=True).std()
            X[f'center_y_std_{window}'] = center_y.rolling(window, min_periods=1, center=True).std()
            
            # Movement range
            X[f'x_range_{window}'] = center_x.rolling(window, min_periods=1, center=True).max() - center_x.rolling(window, min_periods=1, center=True).min()
            X[f'y_range_{window}'] = center_y.rolling(window, min_periods=1, center=True).max() - center_y.rolling(window, min_periods=1, center=True).min()
            
            # Activity level
            X[f'activity_level_{window}'] = np.sqrt(
                center_x.diff().rolling(window, min_periods=1).var() + 
                center_y.diff().rolling(window, min_periods=1).var()
            )
    
    # Lag features
    if 'nose' in available_body_parts and 'tail_base' in available_body_parts:
        nose_tail_dist = np.sqrt(
            (single_mouse['nose']['x'] - single_mouse['tail_base']['x'])**2 + 
            (single_mouse['nose']['y'] - single_mouse['tail_base']['y'])**2
        )
        
        for lag in [10, 20, 40]:
            X[f'nose_tail_dist_lag_{lag}'] = nose_tail_dist.shift(lag)
            X[f'nose_tail_dist_diff_{lag}'] = nose_tail_dist - nose_tail_dist.shift(lag)
    
    return X

def transform_pair(mouse_pair: pd.DataFrame, body_parts_tracked: List[str]) -> pd.DataFrame:
    """Enhanced feature engineering for pairs with social interaction features"""
    available_A = mouse_pair['A'].columns.get_level_values(0)
    available_B = mouse_pair['B'].columns.get_level_values(0)

    # Cross-mouse pairwise distances
    feats = {}
    for p1, p2 in itertools.product(body_parts_tracked, repeat=2):
        if p1 in available_A and p2 in available_B:
            dif = mouse_pair['A'][p1] - mouse_pair['B'][p2]
            feats[f"12+{p1}+{p2}"] = np.square(dif).sum(axis=1, skipna=False)
    X = pd.DataFrame(feats, index=mouse_pair.index)

    # Basic speed features
    try:
        if ('A', 'ear_left') in mouse_pair.columns and ('B', 'ear_left') in mouse_pair.columns:
            shifted_A = mouse_pair['A']['ear_left'].shift(10)
            shifted_B = mouse_pair['B']['ear_left'].shift(10)
            X2 = pd.DataFrame({
                'speed_left_A':  np.square(mouse_pair['A']['ear_left'] - shifted_A).sum(axis=1, skipna=False),
                'speed_left_AB': np.square(mouse_pair['A']['ear_left'] - shifted_B).sum(axis=1, skipna=False),
                'speed_left_B':  np.square(mouse_pair['B']['ear_left'] - shifted_B).sum(axis=1, skipna=False),
            }, index=mouse_pair.index)
            X = pd.concat([X, X2], axis=1)
    except Exception:
        pass
    
    # SOCIAL INTERACTION FEATURES (from files 2-3)
    time_windows = [5, 15, 30, 60]
    
    if 'body_center' in available_A and 'body_center' in available_B:
        center_dist = np.square(mouse_pair['A']['body_center'] - mouse_pair['B']['body_center']).sum(axis=1, skipna=False)
        
        # Social zone indicators
        X['very_close'] = (np.sqrt(center_dist) < 5.0).astype(float)
        X['close'] = ((np.sqrt(center_dist) >= 5.0) & (np.sqrt(center_dist) < 15.0)).astype(float)
        
        for window in time_windows:
            # Distance statistics
            X[f'dist_mean_{window}'] = center_dist.rolling(window, min_periods=1, center=True).mean()
            X[f'dist_std_{window}'] = center_dist.rolling(window, min_periods=1, center=True).std()
            
            # Interaction intensity
            dist_var = center_dist.rolling(window, min_periods=1, center=True).var()
            X[f'interaction_intensity_{window}'] = 1 / (1 + dist_var)
            
            # Coordinated movement
            A_x_diff = mouse_pair['A']['body_center']['x'].diff()
            A_y_diff = mouse_pair['A']['body_center']['y'].diff()
            B_x_diff = mouse_pair['B']['body_center']['x'].diff()
            B_y_diff = mouse_pair['B']['body_center']['y'].diff()
            
            coord_movement = A_x_diff * B_x_diff + A_y_diff * B_y_diff
            X[f'coord_movement_mean_{window}'] = coord_movement.rolling(window, min_periods=1, center=True).mean()
    
    # Lag features for interactions
    if 'nose' in available_A and 'nose' in available_B:
        nose_nose_dist = np.sqrt(
            (mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x'])**2 +
            (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y'])**2
        )
        
        for lag in [10, 20, 40]:
            X[f'nose_nose_dist_lag_{lag}'] = nose_nose_dist.shift(lag)
            X[f'nose_nose_dist_change_{lag}'] = nose_nose_dist - nose_nose_dist.shift(lag)
            
            # Interaction persistence
            close_threshold = 10.0
            is_close = (nose_nose_dist < close_threshold).astype(float)
            X[f'close_persistence_{lag}'] = is_close.rolling(lag, min_periods=1).mean()
    
    return X

# ========================
# TrainOnSubsetClassifier
# ========================
class TrainOnSubsetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model: Any, max_samples: int = 100_000, random_state: int = 42):
        self.model = model
        self.max_samples = int(max_samples)
        self.random_state = random_state

    def fit(self, X, y):
        n = X.shape[0]
        if self.max_samples and n > self.max_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(n, size=self.max_samples, replace=False)
            Xs = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
            ys = y[idx]
            self.model.fit(Xs, ys)
        else:
            self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    @property
    def classes_(self):
        return self.model.classes_

# ========================
# Submit function
# ========================
def submit(body_parts_tracked_str, switch_tr, binary_classifier, X_tr, label, meta, test_df):
    train_cols = list(X_tr.columns)

    # Fit one-vs-rest models
    model_list = []
    for action in label.columns:
        action_mask = ~ label[action].isna().values
        y_action = label[action][action_mask].values.astype(int)
        if not (y_action == 0).all():
            model = clone(binary_classifier)
            model.fit(X_tr[action_mask], y_action)
            model_list.append((action, model))

    # Inference
    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

    test_subset = test_df[test_df.body_parts_tracked == body_parts_tracked_str]
    generator = generate_mouse_data(test_subset, 'test',
                                    generate_single=(switch_tr == 'single'),
                                    generate_pair=(switch_tr == 'pair'))
    if verbose: print(f"n_videos: {len(test_subset)}")

    parts = []

    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr
        try:
            # Transform
            if switch_te == 'single':
                X_te = transform_single(data_te, body_parts_tracked)
            else:
                X_te = transform_pair(data_te, body_parts_tracked)

            X_te = X_te.reindex(columns=train_cols)

            if verbose and len(X_te) == 0:
                print("ERROR: X_te is empty")
            del data_te

            pred = pd.DataFrame(index=meta_te.video_frame)
            for action, model in model_list:
                if action in actions_te:
                    pred[action] = model.predict_proba(X_te)[:, 1]
            del X_te

            if pred.shape[1] != 0:
                if USE_HYSTERESIS_DECODER:
                    submission_part = predict_multiclass_hysteresis(pred, meta_te)
                else:
                    # Fallback to simple decoder
                    submission_part = predict_multiclass_simple(pred, meta_te)
                parts.append(submission_part)
            else:
                if verbose: print("  ERROR: no useful training data")
        except KeyError:
            if verbose: print(f'  ERROR: KeyError because of missing bodypart ({switch_tr})')
            del data_te

    if len(parts) == 0:
        return pd.DataFrame(columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
    return pd.concat(parts, ignore_index=True)

# ========================
# Simple decoder fallback
# ========================
def predict_multiclass_simple(pred: pd.DataFrame, meta: pd.DataFrame, threshold: float = 0.231) -> pd.DataFrame:
    """Simple argmax decoder with threshold"""
    ama = np.argmax(pred.values, axis=1)
    ama = np.where(pred.max(axis=1).values >= threshold, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame.values)

    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]

    mask = ama_changes.values >= 0
    mask[-1] = False
    submission_part = pd.DataFrame({
        'video_id':  meta_changes['video_id'][mask].values,
        'agent_id':  meta_changes['agent_id'][mask].values,
        'target_id': meta_changes['target_id'][mask].values,
        'action':    pred.columns[ama_changes[mask].values],
        'start_frame': ama_changes.index[mask],
        'stop_frame':   ama_changes.index[1:][mask[:-1]]
    })

    # Repair stop frame if group changed
    stop_video_id = meta_changes['video_id'][1:][mask[:-1]].values
    stop_agent_id = meta_changes['agent_id'][1:][mask[:-1]].values
    stop_target_id = meta_changes['target_id'][1:][mask[:-1]].values
    for i in range(len(submission_part)):
        v = submission_part.video_id.iloc[i]
        a = submission_part.agent_id.iloc[i]
        t = submission_part.target_id.iloc[i]
        if stop_video_id[i] != v or stop_agent_id[i] != a or stop_target_id[i] != t:
            new_stop_frame = meta.query("(video_id == @v)").video_frame.max() + 1
            submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame

    assert (submission_part.stop_frame > submission_part.start_frame).all(), 'stop <= start'
    if verbose: print('  actions found:', len(submission_part))
    return submission_part

# ========================
# Robustify function
# ========================
def robustify(submission: pd.DataFrame,
              dataset: pd.DataFrame,
              traintest: str,
              traintest_directory: Optional[str] = None) -> pd.DataFrame:
    """Clean and fill empty videos in submission"""
    if traintest_directory is None:
        traintest_directory = f"{CFG.data_root}/{traintest}_tracking"

    for c in ['start_frame', 'stop_frame']:
        submission[c] = pd.to_numeric(submission[c], errors='coerce')
    submission = submission.dropna(subset=['start_frame', 'stop_frame'])
    submission[['start_frame', 'stop_frame']] = submission[['start_frame', 'stop_frame']].astype(int)

    old_len = len(submission)
    submission = submission[submission.start_frame < submission.stop_frame]
    if len(submission) != old_len:
        print("ERROR: Dropped frames with start >= stop")

    old_len = len(submission)
    group_list = []
    for _, group in submission.groupby(['video_id', 'agent_id', 'target_id'], sort=False):
        group = group.sort_values('start_frame').reset_index(drop=True)
        mask = np.ones(len(group), dtype=bool)
        last_stop_frame = -1
        for i, row in group.iterrows():
            st = int(row['start_frame'])
            en = int(row['stop_frame'])
            if st < last_stop_frame:
                mask[i] = False
            else:
                last_stop_frame = en
        group_list.append(group[mask])
    if len(group_list) > 0:
        submission = pd.concat(group_list, ignore_index=True)
    if len(submission) != old_len:
        print("ERROR: Dropped duplicate/overlapping frames")

    # Fill empty videos with minimal stub
    s_list = []
    for _, row in dataset.iterrows():
        lab_id = row.get('lab_id', '')
        if str(lab_id).startswith('MABe22'):
            continue
        video_id = row['video_id']
        if (submission.video_id == video_id).any():
            continue

        if verbose:
            print(f"Video {video_id} has no predictions.")

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        try:
            vid = pd.read_parquet(path)
            start_frame = int(vid.video_frame.min())
            stop_bound = int(vid.video_frame.max()) + 1
            if not np.isfinite(start_frame) or not np.isfinite(stop_bound):
                start_frame, stop_bound = 0, 1
        except Exception:
            start_frame, stop_bound = 0, 1

        # Minimal 1-frame stub
        stub_start = start_frame
        stub_stop = min(stub_start + 1, stop_bound)
        if stub_stop > stub_start:
            s_list.append((video_id, 'mouse1', 'self', 'sniff', stub_start, stub_stop))

    if len(s_list) > 0:
        fill_df = pd.DataFrame(
            s_list,
            columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
        )
        fill_df[['start_frame', 'stop_frame']] = fill_df[['start_frame', 'stop_frame']].astype(int)
        submission = pd.concat([submission, fill_df], ignore_index=True)
        print(f"Filled {len(s_list)} empty videos")

    submission = submission.reset_index(drop=True)
    return submission

# ========================
# Runner
# ========================
class SubmitRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.submission_parts: List[pd.DataFrame] = []

    def load_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        log("[INFO] Loading train/test metadata…")
        train = pd.read_csv(self.cfg.train_csv)
        test = pd.read_csv(self.cfg.test_csv)

        train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)
        body_parts_tracked_list = list(np.unique(train.body_parts_tracked))

        return train, test, body_parts_tracked_list

    def build_binary_classifier(self) -> Any:
        """Improved model configuration based on files 2-3"""
        clf = make_pipeline(
            SimpleImputer(),
            TrainOnSubsetClassifier(
                model=LGBMClassifier(
                    n_estimators=200,  # Increased from 150
                    learning_rate=0.025,  # Balanced between 0.003 and 0.03
                    min_child_samples=40,
                    num_leaves=31,
                    max_depth=-1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                ),
                max_samples=self.cfg.max_train_samples_per_action,
                random_state=42
            )
        )
        return clf

    def run(self):
        assert validate_or_submit == 'submit', "This script is submit-only by design."

        train, test, body_parts_tracked_list = self.load_metadata()

        for section in range(1, len(body_parts_tracked_list)):
            body_parts_tracked_str = body_parts_tracked_list[section]
            try:
                body_parts = json.loads(body_parts_tracked_str)
                log(f"{section}. Processing videos with {body_parts}")

                if len(body_parts) > 5:
                    body_parts = [b for b in body_parts if b not in drop_body_parts]

                train_subset = train[train.body_parts_tracked == body_parts_tracked_str]

                # Collect batches
                single_mouse_list, single_label_list, single_meta_list = [], [], []
                mouse_pair_list, mouse_label_list, mouse_meta_list = [], [], []

                for switch, data, meta, label in generate_mouse_data(train_subset, 'train'):
                    if switch == 'single':
                        single_mouse_list.append(data)
                        single_meta_list.append(meta)
                        single_label_list.append(label)
                    else:
                        mouse_pair_list.append(data)
                        mouse_meta_list.append(meta)
                        mouse_label_list.append(label)

                binary_classifier = self.build_binary_classifier()

                # SINGLE head
                if len(single_mouse_list) > 0:
                    single_mouse = pd.concat(single_mouse_list)
                    single_mouse_label = pd.concat(single_label_list)
                    single_mouse_meta = pd.concat(single_meta_list)
                    del single_mouse_list, single_label_list, single_meta_list
                    assert len(single_mouse) == len(single_mouse_label) == len(single_mouse_meta)

                    X_tr = transform_single(single_mouse, body_parts)
                    del single_mouse
                    log(f"{X_tr.shape=}")

                    sub_part = submit(
                        body_parts_tracked_str, 'single',
                        binary_classifier, X_tr, single_mouse_label, single_mouse_meta,
                        test_df=test
                    )
                    if len(sub_part):
                        self.submission_parts.append(sub_part)
                    del X_tr, single_mouse_label, single_mouse_meta
                    gc.collect()

                # PAIR head
                if len(mouse_pair_list) > 0:
                    mouse_pair = pd.concat(mouse_pair_list)
                    mouse_pair_label = pd.concat(mouse_label_list)
                    mouse_pair_meta = pd.concat(mouse_meta_list)
                    del mouse_pair_list, mouse_label_list, mouse_meta_list
                    assert len(mouse_pair) == len(mouse_pair_label) == len(mouse_pair_meta)

                    X_tr = transform_pair(mouse_pair, body_parts)
                    del mouse_pair
                    log(f"{X_tr.shape=}")

                    sub_part = submit(
                        body_parts_tracked_str, 'pair',
                        binary_classifier, X_tr, mouse_pair_label, mouse_pair_meta,
                        test_df=test
                    )
                    if len(sub_part):
                        self.submission_parts.append(sub_part)
                    del X_tr, mouse_pair_label, mouse_pair_meta
                    gc.collect()

            except Exception as e:
                print(f'***Exception*** {e}')
            print()

        # Stitch submission
        if len(self.submission_parts) > 0:
            submission = pd.concat(self.submission_parts, ignore_index=True)
        else:
            submission = pd.DataFrame(dict(
                video_id=438887472,
                agent_id='mouse1',
                target_id='self',
                action='rear',
                start_frame=278,
                stop_frame=500
            ), index=[44])

        submission_robust = robustify(submission, test, 'test')
        submission_robust.index.name = self.cfg.row_id_col
        submission_robust.to_csv(self.cfg.submission_file)
        log(f"[DONE] Wrote {self.cfg.submission_file} with {len(submission_robust):,} rows.")


# ========================
# Entry
# ========================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    SubmitRunner(CFG).run()

