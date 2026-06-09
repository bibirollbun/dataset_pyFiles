# MABe Mouse Behavior Detection - CV Training and Inference Notebook
# Built from extracted snippets with best practices

validate_or_submit = 'submit'  # 'validate' or 'submit'
verbose = True

import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import warnings
import json
import os
import random
import gc
import lightgbm
from collections import defaultdict
import polars as pl
from scipy import signal, stats

from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.model_selection import cross_val_predict, GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, average_precision_score

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

# ==================== GPU DETECTION ====================
USE_GPU = True  # Enable GPU by default - will be verified by detection
GPU_DEVICE = 'cuda'  # Default to cuda for RTX 5090

try:
    import torch
    if torch.cuda.is_available():
        USE_GPU = True
        GPU_DEVICE = 'cuda'
        if verbose:
            print(f"[INFO] GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] CUDA version: {torch.version.cuda}")
except:
    pass

if not USE_GPU:
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            USE_GPU = True
            GPU_DEVICE = 'gpu'
            if verbose:
                print("[INFO] GPU detected via nvidia-smi")
    except:
        pass

if not USE_GPU and verbose:
    print("[INFO] No GPU detected - training will use CPU")

# ==================== LIBRARY-SPECIFIC GPU DETECTION ====================
# Conservative approach: Default to False, only enable if GPU actually works
def check_lightgbm_gpu():
    """Check if LightGBM actually supports GPU"""
    if not USE_GPU:
        return False
    try:
        import lightgbm as lgb
        import numpy as np
        # Try to actually train with GPU
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        train_data = lgb.Dataset(X, label=y)
        params = {'device': 'gpu', 'verbose': -1, 'num_threads': 1}
        # This will fail if GPU not available
        model = lgb.train(params, train_data, num_boost_round=1)
        return True
    except Exception as e:
        if verbose:
            print(f"  LightGBM GPU check failed: {str(e)[:100]}")
        return False

def check_xgboost_gpu():
    """Check if XGBoost actually supports GPU"""
    if not USE_GPU:
        return False
    try:
        from xgboost import XGBClassifier
        import numpy as np
        # Try to actually fit with gpu_hist
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        model = XGBClassifier(tree_method='gpu_hist', n_estimators=1, verbosity=0)
        model.fit(X, y)
        return True
    except Exception as e:
        if verbose:
            print(f"  XGBoost GPU check failed: {str(e)[:100]}")
        return False

def check_catboost_gpu():
    """Check if CatBoost supports GPU (it auto-falls back, so we can be optimistic)"""
    if not USE_GPU:
        return False
    # CatBoost handles GPU fallback automatically, so we can enable it
    # It will use CPU if GPU not available
    return True

# Set GPU availability flags (default to False for safety)
LIGHTGBM_GPU_AVAILABLE = check_lightgbm_gpu()
XGBOOST_GPU_AVAILABLE = check_xgboost_gpu()
CATBOOST_GPU_AVAILABLE = check_catboost_gpu()

if verbose:
    print(f"[INFO] GPU Support: LightGBM={LIGHTGBM_GPU_AVAILABLE}, XGBoost={XGBOOST_GPU_AVAILABLE}, CatBoost={CATBOOST_GPU_AVAILABLE}")

# ==================== ENVIRONMENT DETECTION ====================
IS_KAGGLE = os.path.exists('/kaggle/input')
if IS_KAGGLE:
    BASE_PATH = '/kaggle/input/MABe-mouse-behavior-detection'
    TRACKING_BASE_PATH = BASE_PATH
    if verbose:
        print("[INFO] Running on Kaggle - using Kaggle data paths")
else:
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'notebook':
        BASE_PATH = os.path.join('..', 'data', '01_raw')
        project_root = os.path.abspath('..')
    else:
        BASE_PATH = os.path.join('data', '01_raw')
        project_root = os.path.abspath('.')
    
    BASE_PATH = os.path.abspath(BASE_PATH)
    TRACKING_BASE_PATH = BASE_PATH  # Use same path as CSV files (all data in data/01_raw)
    if verbose:
        print(f"[INFO] Running locally - CSV data path: {BASE_PATH}")
        print(f"[INFO] Running locally - Tracking data path: {TRACKING_BASE_PATH}")

# ==================== CV CONFIGURATION ====================
USE_CV = True  # Toggle CV on/off
CV_N_SPLITS = 5  # Number of folds
CV_GROUP_COLUMN = 'video_id'  # Grouping column for GroupKFold

# ==================== LOCAL TESTING CONFIGURATION ====================
# Set to False to use full dataset parameters (same as Kaggle)
LOCAL_TESTING = False  # Disabled to use full dataset parameters
MAX_VIDEOS_LOCAL = 5
LOCAL_N_SAMPLES = 5000

if LOCAL_TESTING:
    validate_or_submit = 'submit'
    if verbose:
        print(f"[INFO] Local testing mode enabled: max_videos={MAX_VIDEOS_LOCAL}, n_samples={LOCAL_N_SAMPLES}")

else:
    if verbose and not IS_KAGGLE:
        print("[INFO] Running locally with full dataset parameters (same as Kaggle)")

# ==================== SEED EVERYTHING ====================
SEED = 1234
os.environ["PYTHONHASHSEED"] = str(SEED)
rnd = np.random.RandomState(SEED)
random.seed(SEED)
np.random.seed(SEED)

print(f"\n[INFO] Configuration:")
print(f"  - USE_CV: {USE_CV}")
print(f"  - CV_N_SPLITS: {CV_N_SPLITS}")
print(f"  - CV_GROUP_COLUMN: {CV_GROUP_COLUMN}")
print(f"  - XGBoost: {XGBOOST_AVAILABLE}, CatBoost: {CATBOOST_AVAILABLE}")
print(f"  - GPU: {USE_GPU} ({GPU_DEVICE})")



# Body parts to drop (optional, high-dimensional parts)
drop_body_parts = ['headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
                   'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
                   'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint']

def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    """
    Generator function to load and process video data one by one.
    Yields data for single mice (agent=target) and mouse pairs (agent!=target).
    """
    assert traintest in ['train', 'test']
    if traintest_directory is None:
        if IS_KAGGLE:
            traintest_directory = os.path.join(BASE_PATH, f"{traintest}_tracking")
        else:
            traintest_directory = os.path.join(TRACKING_BASE_PATH, f"{traintest}_tracking")
    
    for _, row in dataset.iterrows():
        lab_id = row.lab_id
        video_id = row.video_id

        if type(row.behaviors_labeled) != str:
            if verbose: 
                print('No labeled behaviors:', lab_id, video_id)
            continue

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        try:
            vid = pd.read_parquet(path)
        except FileNotFoundError:
            if verbose:
                print(f"File not found: {path}")
            continue
        
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
        
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        if pvid.isna().any().any():
            if verbose and traintest == 'test': 
                print('video with missing values', video_id, traintest, len(vid), 'frames')
        else:
            if verbose and traintest == 'test': 
                print('video with all values', video_id, traintest, len(vid), 'frames')
        del vid
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        pvid /= row.pix_per_cm_approx

        vid_behaviors = json.loads(row.behaviors_labeled)
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
        
        annot = None
        if traintest == 'train':
            try:
                if IS_KAGGLE:
                    annot_path = path.replace('train_tracking', 'train_annotation')
                else:
                    # Simple replacement - works for both Windows and Unix paths
                    annot_path = path.replace('train_tracking', 'train_annotation')
                annot = pd.read_parquet(annot_path)
            except FileNotFoundError:
                if verbose:
                    print(f"  WARNING: Annotation file not found: {annot_path}")
                    print(f"    Tracking path was: {path}")
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
                        'video_frame': single_mouse.index,
                        'frames_per_second': row.frames_per_second
                    })
                    if traintest == 'train' and annot is not None:
                        single_mouse_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=single_mouse.index)
                        if len(annot) > 0 and 'agent_id' in annot.columns and 'target_id' in annot.columns:
                            annot_subset = annot[(annot['agent_id'] == mouse_id) & (annot['target_id'] == mouse_id)]
                        else:
                            annot_subset = pd.DataFrame()
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            if annot_row['action'] in vid_agent_actions:
                                single_mouse_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row['action']] = 1.0
                        yield 'single', single_mouse, single_mouse_meta, single_mouse_label
                    else:
                        if verbose: 
                            print('- test single', video_id, mouse_id)
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
                        'video_frame': mouse_pair.index,
                        'frames_per_second': row.frames_per_second
                    })
                    if traintest == 'train' and annot is not None:
                        mouse_pair_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=mouse_pair.index)
                        if len(annot) > 0 and 'agent_id' in annot.columns and 'target_id' in annot.columns:
                            annot_subset = annot[
                                (annot['agent_id'] == agent) &
                                (annot['target_id'] == target)
                            ]
                        else:
                            annot_subset = pd.DataFrame()
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            if annot_row['action'] in vid_agent_actions:
                                mouse_pair_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'pair', mouse_pair, mouse_pair_meta, mouse_pair_label
                    else:
                        if verbose: 
                            print('- test pair', video_id, agent, target)
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions



def safe_rolling(series, window, func, min_periods=None):
    """Safe rolling operation with NaN handling"""
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series.rolling(window, min_periods=min_periods, center=True).apply(func, raw=True)

def _scale(n_frames_at_30fps, fps, ref=30.0):
    """Scale a frame count defined at 30 fps to the current video's fps."""
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))

def _scale_signed(n_frames_at_30fps, fps, ref=30.0):
    """Signed version of _scale for forward/backward shifts (keeps at least 1 frame when |n|>=1)."""
    if n_frames_at_30fps == 0:
        return 0
    s = 1 if n_frames_at_30fps > 0 else -1
    mag = max(1, int(round(abs(n_frames_at_30fps) * float(fps) / ref)))
    return s * mag

def _fps_from_meta(meta_df, fallback_lookup, default_fps=30.0):
    """Extract FPS from metadata DataFrame with fallback lookup."""
    if 'frames_per_second' in meta_df.columns and pd.notnull(meta_df['frames_per_second']).any():
        return float(meta_df['frames_per_second'].iloc[0])
    vid = meta_df['video_id'].iloc[0]
    return float(fallback_lookup.get(vid, default_fps))

def add_curvature_features(X, center_x, center_y, fps):
    """Trajectory curvature (window lengths scaled by fps)."""
    vel_x = center_x.diff()
    vel_y = center_y.diff()
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()

    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)

    for w in [30, 60]:
        ws = _scale(w, fps)
        X[f'curv_mean_{w}'] = curvature.rolling(ws, min_periods=max(1, ws // 6)).mean()

    angle = np.arctan2(vel_y, vel_x)
    angle_change = np.abs(angle.diff())
    w = 30
    ws = _scale(w, fps)
    X[f'turn_rate_{w}'] = angle_change.rolling(ws, min_periods=max(1, ws // 6)).sum()

    return X

def add_multiscale_features(X, center_x, center_y, fps):
    """Multi-scale temporal features (speed in cm/s; windows scaled by fps)."""
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)

    scales = [10, 40, 160]
    for scale in scales:
        ws = _scale(scale, fps)
        if len(speed) >= ws:
            X[f'sp_m{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).mean()
            X[f'sp_s{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).std()

    if len(scales) >= 2 and f'sp_m{scales[0]}' in X.columns and f'sp_m{scales[-1]}' in X.columns:
        X['sp_ratio'] = X[f'sp_m{scales[0]}'] / (X[f'sp_m{scales[-1]}'] + 1e-6)

    return X

def add_state_features(X, center_x, center_y, fps):
    """Behavioral state transitions; bins adjusted so semantics are fps-invariant."""
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
    w_ma = _scale(15, fps)
    speed_ma = speed.rolling(w_ma, min_periods=max(1, w_ma // 3)).mean()

    try:
        bins = [-np.inf, 0.5 * fps, 2.0 * fps, 5.0 * fps, np.inf]
        speed_states = pd.cut(speed_ma, bins=bins, labels=[0, 1, 2, 3]).astype(float)

        for window in [60, 120]:
            ws = _scale(window, fps)
            if len(speed_states) >= ws:
                for state in [0, 1, 2, 3]:
                    X[f's{state}_{window}'] = (
                        (speed_states == state).astype(float)
                        .rolling(ws, min_periods=max(1, ws // 6)).mean()
                    )
                state_changes = (speed_states != speed_states.shift(1)).astype(float)
                X[f'trans_{window}'] = state_changes.rolling(ws, min_periods=max(1, ws // 6)).sum()
    except Exception:
        pass

    return X

def add_longrange_features(X, center_x, center_y, fps):
    """Long-range temporal features (windows & spans scaled by fps)."""
    for window in [120, 240]:
        ws = _scale(window, fps)
        if len(center_x) >= ws:
            X[f'x_ml{window}'] = center_x.rolling(ws, min_periods=max(5, ws // 6)).mean()
            X[f'y_ml{window}'] = center_y.rolling(ws, min_periods=max(5, ws // 6)).mean()

    for span in [60, 120]:
        s = _scale(span, fps)
        X[f'x_e{span}'] = center_x.ewm(span=s, min_periods=1).mean()
        X[f'y_e{span}'] = center_y.ewm(span=s, min_periods=1).mean()

    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
    for window in [60, 120]:
        ws = _scale(window, fps)
        if len(speed) >= ws:
            X[f'sp_pct{window}'] = speed.rolling(ws, min_periods=max(5, ws // 6)).rank(pct=True)

    return X

def add_interaction_features(X, mouse_pair, avail_A, avail_B, fps):
    """Social interaction features (windows scaled by fps)."""
    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return X

    rel_x = mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x']
    rel_y = mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y']
    rel_dist = np.sqrt(rel_x**2 + rel_y**2)

    A_vx = mouse_pair['A']['body_center']['x'].diff()
    A_vy = mouse_pair['A']['body_center']['y'].diff()
    B_vx = mouse_pair['B']['body_center']['x'].diff()
    B_vy = mouse_pair['B']['body_center']['y'].diff()

    A_lead = (A_vx * rel_x + A_vy * rel_y) / (np.sqrt(A_vx**2 + A_vy**2) * rel_dist + 1e-6)
    B_lead = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (np.sqrt(B_vx**2 + B_vy**2) * rel_dist + 1e-6)

    for window in [30, 60]:
        ws = _scale(window, fps)
        X[f'A_ld{window}'] = A_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()
        X[f'B_ld{window}'] = B_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()

    approach = -rel_dist.diff()
    chase = approach * B_lead
    w = 30
    ws = _scale(w, fps)
    X[f'chase_{w}'] = chase.rolling(ws, min_periods=max(1, ws // 6)).mean()

    for window in [60, 120]:
        ws = _scale(window, fps)
        A_sp = np.sqrt(A_vx**2 + A_vy**2)
        B_sp = np.sqrt(B_vx**2 + B_vy**2)
        X[f'sp_cor{window}'] = A_sp.rolling(ws, min_periods=max(1, ws // 6)).corr(B_sp)

    return X



def transform_single(single_mouse, body_parts_tracked, fps):
    """Enhanced single mouse transform (FPS-aware windows/lags; distances in cm)."""
    available_body_parts = single_mouse.columns.get_level_values(0)

    # Base distance features (squared distances across body parts)
    X = pd.DataFrame({
        f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.combinations(body_parts_tracked, 2)
        if p1 in available_body_parts and p2 in available_body_parts
    })
    X = X.reindex(columns=[f"{p1}+{p2}" for p1, p2 in itertools.combinations(body_parts_tracked, 2)], copy=False)

    # Speed-like features via lagged displacements (duration-aware lag)
    if all(p in single_mouse.columns for p in ['ear_left', 'ear_right', 'tail_base']):
        lag = _scale(10, fps)
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(lag)
        speeds = pd.DataFrame({
            'sp_lf': np.square(single_mouse['ear_left'] - shifted['ear_left']).sum(axis=1, skipna=False),
            'sp_rt': np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False),
            'sp_lf2': np.square(single_mouse['ear_left'] - shifted['tail_base']).sum(axis=1, skipna=False),
            'sp_rt2': np.square(single_mouse['ear_right'] - shifted['tail_base']).sum(axis=1, skipna=False),
        })
        X = pd.concat([X, speeds], axis=1)

    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)

    # Body angle (orientation)
    if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
        v1 = single_mouse['nose'] - single_mouse['body_center']
        v2 = single_mouse['tail_base'] - single_mouse['body_center']
        X['body_ang'] = (v1['x'] * v2['x'] + v1['y'] * v2['y']) / (
            np.sqrt(v1['x']**2 + v1['y']**2) * np.sqrt(v2['x']**2 + v2['y']**2) + 1e-6)

    # Core temporal features (windows scaled by fps)
    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']
        cy = single_mouse['body_center']['y']

        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)
            X[f'cx_m{w}'] = cx.rolling(ws, **roll).mean()
            X[f'cy_m{w}'] = cy.rolling(ws, **roll).mean()
            X[f'cx_s{w}'] = cx.rolling(ws, **roll).std()
            X[f'cy_s{w}'] = cy.rolling(ws, **roll).std()
            X[f'x_rng{w}'] = cx.rolling(ws, **roll).max() - cx.rolling(ws, **roll).min()
            X[f'y_rng{w}'] = cy.rolling(ws, **roll).max() - cy.rolling(ws, **roll).min()
            X[f'disp{w}'] = np.sqrt(cx.diff().rolling(ws, min_periods=1).sum()**2 +
                                     cy.diff().rolling(ws, min_periods=1).sum()**2)
            X[f'act{w}'] = np.sqrt(cx.diff().rolling(ws, min_periods=1).var() +
                                   cy.diff().rolling(ws, min_periods=1).var())

        # Advanced features (fps-scaled)
        X = add_curvature_features(X, cx, cy, fps)
        X = add_multiscale_features(X, cx, cy, fps)
        X = add_state_features(X, cx, cy, fps)
        X = add_longrange_features(X, cx, cy, fps)

    # Nose-tail features with duration-aware lags
    if all(p in available_body_parts for p in ['nose', 'tail_base']):
        nt_dist = np.sqrt((single_mouse['nose']['x'] - single_mouse['tail_base']['x'])**2 +
                          (single_mouse['nose']['y'] - single_mouse['tail_base']['y'])**2)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            X[f'nt_lg{lag}'] = nt_dist.shift(l)
            X[f'nt_df{lag}'] = nt_dist - nt_dist.shift(l)

    # Ear features with duration-aware offsets
    if all(p in available_body_parts for p in ['ear_left', 'ear_right']):
        ear_d = np.sqrt((single_mouse['ear_left']['x'] - single_mouse['ear_right']['x'])**2 +
                        (single_mouse['ear_left']['y'] - single_mouse['ear_right']['y'])**2)
        for off in [-20, -10, 10, 20]:
            o = _scale_signed(off, fps)
            X[f'ear_o{off}'] = ear_d.shift(-o)  
        w = _scale(30, fps)
        X['ear_con'] = ear_d.rolling(w, min_periods=1, center=True).std() / \
                       (ear_d.rolling(w, min_periods=1, center=True).mean() + 1e-6)

    return X.astype(np.float32, copy=False)

def transform_pair(mouse_pair, body_parts_tracked, fps):
    """Enhanced pair transform (FPS-aware windows/lags; distances in cm)."""
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)

    # Inter-mouse distances (squared distances across all part pairs)
    X = pd.DataFrame({
        f"12+{p1}+{p2}": np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.product(body_parts_tracked, repeat=2)
        if p1 in avail_A and p2 in avail_B
    })
    X = X.reindex(columns=[f"12+{p1}+{p2}" for p1, p2 in itertools.product(body_parts_tracked, repeat=2)], copy=False)

    # Speed-like features via lagged displacements (duration-aware lag)
    if ('A', 'ear_left') in mouse_pair.columns and ('B', 'ear_left') in mouse_pair.columns:
        lag = _scale(10, fps)
        shA = mouse_pair['A']['ear_left'].shift(lag)
        shB = mouse_pair['B']['ear_left'].shift(lag)
        speeds = pd.DataFrame({
            'sp_A': np.square(mouse_pair['A']['ear_left'] - shA).sum(axis=1, skipna=False),
            'sp_AB': np.square(mouse_pair['A']['ear_left'] - shB).sum(axis=1, skipna=False),
            'sp_B': np.square(mouse_pair['B']['ear_left'] - shB).sum(axis=1, skipna=False),
        })
        X = pd.concat([X, speeds], axis=1)

    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)

    # Relative orientation
    if all(p in avail_A for p in ['nose', 'tail_base']) and all(p in avail_B for p in ['nose', 'tail_base']):
        dir_A = mouse_pair['A']['nose'] - mouse_pair['A']['tail_base']
        dir_B = mouse_pair['B']['nose'] - mouse_pair['B']['tail_base']
        X['rel_ori'] = (dir_A['x'] * dir_B['x'] + dir_A['y'] * dir_B['y']) / (
            np.sqrt(dir_A['x']**2 + dir_A['y']**2) * np.sqrt(dir_B['x']**2 + dir_B['y']**2) + 1e-6)

    # Approach rate (duration-aware lag)
    if all(p in avail_A for p in ['nose']) and all(p in avail_B for p in ['nose']):
        cur = np.square(mouse_pair['A']['nose'] - mouse_pair['B']['nose']).sum(axis=1, skipna=False)
        lag = _scale(10, fps)
        shA_n = mouse_pair['A']['nose'].shift(lag)
        shB_n = mouse_pair['B']['nose'].shift(lag)
        past = np.square(shA_n - shB_n).sum(axis=1, skipna=False)
        X['appr'] = cur - past

    # Distance bins (cm; unchanged by fps)
    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd = np.sqrt((mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x'])**2 +
                     (mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y'])**2)
        X['v_cls'] = (cd < 5.0).astype(float)
        X['cls']   = ((cd >= 5.0) & (cd < 15.0)).astype(float)
        X['med']   = ((cd >= 15.0) & (cd < 30.0)).astype(float)
        X['far']   = (cd >= 30.0).astype(float)

    # Temporal interaction features (fps-adjusted windows)
    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd_full = np.square(mouse_pair['A']['body_center'] - mouse_pair['B']['body_center']).sum(axis=1, skipna=False)

        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)
            X[f'd_m{w}']  = cd_full.rolling(ws, **roll).mean()
            X[f'd_s{w}']  = cd_full.rolling(ws, **roll).std()
            X[f'd_mn{w}'] = cd_full.rolling(ws, **roll).min()
            X[f'd_mx{w}'] = cd_full.rolling(ws, **roll).max()

            d_var = cd_full.rolling(ws, **roll).var()
            X[f'int{w}'] = 1 / (1 + d_var)

            Axd = mouse_pair['A']['body_center']['x'].diff()
            Ayd = mouse_pair['A']['body_center']['y'].diff()
            Bxd = mouse_pair['B']['body_center']['x'].diff()
            Byd = mouse_pair['B']['body_center']['y'].diff()
            coord = Axd * Bxd + Ayd * Byd
            X[f'co_m{w}'] = coord.rolling(ws, **roll).mean()
            X[f'co_s{w}'] = coord.rolling(ws, **roll).std()

    # Nose-nose dynamics (duration-aware lags)
    if 'nose' in avail_A and 'nose' in avail_B:
        nn = np.sqrt((mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x'])**2 +
                     (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y'])**2)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            X[f'nn_lg{lag}']  = nn.shift(l)
            X[f'nn_ch{lag}']  = nn - nn.shift(l)
            is_cl = (nn < 10.0).astype(float)
            X[f'cl_ps{lag}']  = is_cl.rolling(l, min_periods=1).mean()

    # Velocity alignment (duration-aware offsets)
    if 'body_center' in avail_A and 'body_center' in avail_B:
        Avx = mouse_pair['A']['body_center']['x'].diff()
        Avy = mouse_pair['A']['body_center']['y'].diff()
        Bvx = mouse_pair['B']['body_center']['x'].diff()
        Bvy = mouse_pair['B']['body_center']['y'].diff()
        val = (Avx * Bvx + Avy * Bvy) / (np.sqrt(Avx**2 + Avy**2) * np.sqrt(Bvx**2 + Bvy**2) + 1e-6)

        for off in [-20, -10, 0, 10, 20]:
            o = _scale_signed(off, fps)
            X[f'va_{off}'] = val.shift(-o)

        w = _scale(30, fps)
        X['int_con'] = cd_full.rolling(w, min_periods=1, center=True).std() / \
                       (cd_full.rolling(w, min_periods=1, center=True).mean() + 1e-6)

        # Advanced interaction (fps-adjusted internals)
        X = add_interaction_features(X, mouse_pair, avail_A, avail_B, fps)

    return X.astype(np.float32, copy=False)



class StratifiedSubsetClassifier(ClassifierMixin, BaseEstimator):
    """A wrapper to train an estimator on a stratified subsample of data."""
    def __init__(self, estimator, n_samples=None):
        self.estimator = estimator
        self.n_samples = n_samples  # if None → no subsampling

    def _to_numpy(self, X):
        try:
            return X.to_numpy(np.float32, copy=False)
        except AttributeError:
            return np.asarray(X, dtype=np.float32)

    def fit(self, X, y):
        Xn = self._to_numpy(X)
        y = np.asarray(y).ravel()

        # Handle rare cases where labels might be [0, 2]
        uniq = np.unique(y[~pd.isna(y)])
        if set(uniq.tolist()) == {0, 2}:
            y = (y > 0).astype(np.int8)

        # Calculate class weights for imbalance handling
        n_pos = int((y == 1).sum())
        n_neg = len(y) - n_pos
        class_weight = n_neg / n_pos if n_pos > 0 and n_neg > 0 else 1.0
        
        # Set class imbalance handling for different model types
        if hasattr(self.estimator, 'set_params'):
            # LightGBM
            if hasattr(self.estimator, 'is_unbalance'):
                self.estimator.set_params(is_unbalance=True)
            # XGBoost
            if hasattr(self.estimator, 'scale_pos_weight'):
                self.estimator.set_params(scale_pos_weight=class_weight)
            # CatBoost
            if hasattr(self.estimator, 'auto_class_weights'):
                try:
                    self.estimator.set_params(auto_class_weights='Balanced')
                except Exception:
                    pass

        # Calculate minimum positive samples required
        min_pos = max(100, int(self.n_samples * 0.01)) if self.n_samples else 100
        n_pos_actual = int((y == 1).sum())

        # If n_samples is None or data is small, fit on full data
        if self.n_samples is None or len(Xn) <= int(self.n_samples):
            self.estimator.fit(Xn, y)
        else:
            # If too few positives, use all data instead of subsampling
            if n_pos_actual < min_pos:
                self.estimator.fit(Xn, y)
            else:
                # Fit on a stratified subset that preserves class balance
                sss = StratifiedShuffleSplit(n_splits=1, train_size=int(self.n_samples), random_state=42)
                try:
                    idx, _ = next(sss.split(np.zeros_like(y), y))
                    # Verify we have positive samples in the subset
                    if (y[idx] == 1).sum() > 0:
                        self.estimator.fit(Xn[idx], y[idx])
                    else:
                        # Fallback: use all data if stratified split loses all positives
                        self.estimator.fit(Xn, y)
                except Exception:
                    # Fallback for cases where stratification fails
                    step = max(len(Xn) // int(self.n_samples), 1)
                    self.estimator.fit(Xn[::step], y[::step])

        try:
            self.classes_ = np.asarray(self.estimator.classes_)
        except Exception:
            self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        Xn = self._to_numpy(X)
        try:
            P = self.estimator.predict_proba(Xn)
        except Exception:
            if len(self.classes_) == 1:
                n = len(Xn)
                c = int(self.classes_[0])
                if c == 1:
                    return np.column_stack([np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)])
                else:
                    return np.column_stack([np.ones(n, dtype=np.float32), np.zeros(n, dtype=np.float32)])
            return np.full((len(Xn), 2), 0.5, dtype=np.float32)

        P = np.asarray(P)
        if P.ndim == 1:
            P1 = P.astype(np.float32)
            return np.column_stack([1.0 - P1, P1])
        if P.shape[1] == 1 and len(self.classes_) == 2:
            P1 = P[:, 0].astype(np.float32)
            return np.column_stack([1.0 - P1, P1])
        return P

    def predict(self, X):
        Xn = self._to_numpy(X)
        try:
            return self.estimator.predict(Xn)
        except Exception:
            return np.argmax(self.predict_proba(Xn), axis=1)



# ==================== EXTRACT AND APPLY OPTIMIZED THRESHOLDS ====================
# Extract optimized thresholds from CV metrics and update action_thresholds for inference
# This cell should be run AFTER training is complete (Cell 10)

# Initialize action_thresholds if it doesn't exist
if 'action_thresholds' not in globals():
    action_thresholds = {
        "default": 0.27,
        "single_default": 0.27,
        "pair_default": 0.27,
        "single": {},
        "pair": {},
    }

# Check if all_cv_metrics_global exists (only available after training)
if 'all_cv_metrics_global' in globals() and all_cv_metrics_global:
    # Create a dictionary to store optimized thresholds per action
    optimized_thresholds = {}
    threshold_counts = {}
    
    for key, metrics in all_cv_metrics_global.items():
        body_parts_str, mode = key
        if metrics:
            for action, cv_metrics in metrics.items():
                # Extract threshold if available, even if it's 0.5 (might be valid)
                if 'opt_threshold' in cv_metrics:
                    opt_thresh = cv_metrics['opt_threshold']
                    # Only use if it's not None and is a valid number
                    if opt_thresh is not None and not np.isnan(opt_thresh):
                        opt_thresh = float(opt_thresh)
                        # Store with mode prefix to handle single vs pair behaviors
                        action_key = f"{mode}_{action}"
                        if action_key not in optimized_thresholds:
                            optimized_thresholds[action_key] = []
                            threshold_counts[action_key] = 0
                        optimized_thresholds[action_key].append(opt_thresh)
                        threshold_counts[action_key] += 1
    
    # Average thresholds across body part configurations for each action
    if optimized_thresholds:
        # Ensure mode dictionaries exist
        if 'single' not in action_thresholds:
            action_thresholds['single'] = {}
        if 'pair' not in action_thresholds:
            action_thresholds['pair'] = {}
        
        for action_key, thresholds_list in optimized_thresholds.items():
            if len(thresholds_list) > 0:
                # Use median instead of mean for robustness to outliers
                avg_threshold = np.median(thresholds_list)
                # Clamp to reasonable range
                avg_threshold = np.clip(avg_threshold, 0.15, 0.50)
                
                if action_key.startswith('single_'):
                    action = action_key.replace('single_', '')
                    action_thresholds['single'][action] = float(avg_threshold)
                    if verbose:
                        print(f"  Extracted single threshold for {action}: {avg_threshold:.4f} (from {threshold_counts[action_key]} configs)")
                elif action_key.startswith('pair_'):
                    action = action_key.replace('pair_', '')
                    action_thresholds['pair'][action] = float(avg_threshold)
                    if verbose:
                        print(f"  Extracted pair threshold for {action}: {avg_threshold:.4f} (from {threshold_counts[action_key]} configs)")
        
        if verbose:
            print("\n" + "="*60)
            print("Optimized Thresholds Applied from CV")
            print("="*60)
            if action_thresholds.get('single'):
                print("\nSingle behavior thresholds:")
                for action, thresh in sorted(action_thresholds['single'].items()):
                    print(f"  {action}: {thresh:.4f}")
            if action_thresholds.get('pair'):
                print("\nPair behavior thresholds:")
                for action, thresh in sorted(action_thresholds['pair'].items()):
                    print(f"  {action}: {thresh:.4f}")
            print("="*60 + "\n")
    else:
        if verbose:
            print("\n" + "="*60)
            print("WARNING: No optimized thresholds found in CV metrics")
            print("="*60)
            print("This may indicate threshold optimization failed for all behaviors.")
            print("Using default thresholds instead.")
            print("="*60 + "\n")
else:
    if verbose:
        print("\n" + "="*60)
        print("NOTE: Optimized thresholds not available yet")
        print("="*60)
        print("This cell extracts optimized thresholds from CV metrics.")
        print("Please run the Training Execution cell (Cell 10) first.")
        print("="*60 + "\n")



# ==================== THRESHOLD CONFIGURATION ====================
# Initialize action_thresholds with defaults, but preserve optimized values if they exist
# This cell should run AFTER Cell 6 (optimized threshold extraction) to preserve optimized values

# Check if action_thresholds already exists (from Cell 6 optimized thresholds)
if 'action_thresholds' not in globals() or not isinstance(action_thresholds, dict):
    # Initialize with defaults only if it doesn't exist
    action_thresholds = {
        "default": 0.27,
        "single_default": 0.27,
        "pair_default": 0.27,
        "single": {},
        "pair": {},
    }
    # Set default for 'rear' if initializing fresh
    action_thresholds["single"]["rear"] = 0.30
else:
    # Preserve existing optimized thresholds, only set defaults if missing
    if "default" not in action_thresholds:
        action_thresholds["default"] = 0.27
    if "single_default" not in action_thresholds:
        action_thresholds["single_default"] = 0.27
    if "pair_default" not in action_thresholds:
        action_thresholds["pair_default"] = 0.27
    if "single" not in action_thresholds:
        action_thresholds["single"] = {}
    if "pair" not in action_thresholds:
        action_thresholds["pair"] = {}
    
    # Only set default for 'rear' if it wasn't already optimized
    if "rear" not in action_thresholds.get("single", {}):
        action_thresholds["single"]["rear"] = 0.30

def _select_threshold_map(thresholds, mode: str):
    """Select threshold map based on mode (single or pair)."""
    if isinstance(thresholds, dict):
        # mode-aware?
        if ("single" in thresholds) or ("pair" in thresholds) or \
           ("single_default" in thresholds) or ("pair_default" in thresholds):
            base_default = float(thresholds.get("default", 0.27))
            mode_default = float(thresholds.get(f"{mode}_default", base_default))
            mode_overrides = thresholds.get(mode, {}) or {}
            out = defaultdict(lambda: mode_default)
            out.update({str(k): float(v) for k, v in mode_overrides.items()})
            return out
        # plain per-action dict
        out = defaultdict(lambda: float(thresholds.get("default", 0.27)))
        out.update({str(k): float(v) for k, v in thresholds.items() if k != "default"})
        return out
    return defaultdict(lambda: 0.27)



def train_ensemble_single_split(X_tr, label, models, n_samples):
    """Original single-split training (fallback when CV is disabled)."""
    X_tr_np = X_tr.to_numpy(np.float32, copy=False)
    model_list = []
    
    for action in label.columns:
        y_raw = label[action].to_numpy()
        mask = ~pd.isna(y_raw)
        y_action = y_raw[mask].astype(int)
        if not (y_action == 0).all() and np.sum(y_action) >= 5:
            trained = []
            idx = np.flatnonzero(mask)
            for m in models:
                m_clone = clone(m)
                m_clone.fit(X_tr_np[idx], y_action)
                trained.append(m_clone)
            model_list.append((action, [trained]))  # Wrap in list for compatibility
    
    return model_list

def optimize_threshold(y_true, y_pred_proba):
    """
    Optimize threshold for F1 score on validation set.
    Enhanced with better error handling and more efficient search.
    """
    try:
        # Check for valid inputs
        if len(y_true) == 0 or len(y_pred_proba) == 0:
            if verbose:
                print("    Warning: Empty arrays for threshold optimization, using default 0.5")
            return 0.5, 0.0
        
        # Check if we have both classes
        unique_labels = np.unique(y_true)
        if len(unique_labels) < 2:
            if verbose:
                print(f"    Warning: Only one class present for threshold optimization, using default 0.5")
            return 0.5, 0.0
        
        # Use coarser search for speed (0.05 steps) but still accurate
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_f1 = 0
        
        for thresh in thresholds:
            try:
                y_pred = (y_pred_proba >= thresh).astype(int)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = thresh
            except Exception as e:
                if verbose:
                    print(f"    Warning: Error computing F1 at threshold {thresh}: {str(e)[:50]}")
                continue
        
        # Refine around best threshold with finer search
        if best_threshold > 0.1 and best_threshold < 0.9:
            refine_range = np.arange(max(0.1, best_threshold - 0.05), 
                                    min(0.9, best_threshold + 0.05), 0.01)
            for thresh in refine_range:
                try:
                    y_pred = (y_pred_proba >= thresh).astype(int)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = thresh
                except Exception:
                    continue
        
        # Clamp threshold to reasonable range
        best_threshold = np.clip(best_threshold, 0.15, 0.50)
        
        if verbose and best_f1 > 0:
            print(f"    Optimized threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")
        
        return best_threshold, best_f1
    except Exception as e:
        if verbose:
            print(f"    Warning: Threshold optimization failed: {str(e)[:100]}")
        return 0.5, 0.0

def train_ensemble_with_cv(X_tr, label, meta, models, n_samples, switch_tr, body_parts_tracked_str):
    """
    Train ensemble models using GroupKFold cross-validation.
    
    Returns:
        model_list: List of (action, list_of_fold_models) tuples
        oof_predictions: Dict mapping action -> DataFrame with OOF predictions
        cv_metrics: Dict mapping action -> CV metrics (mean ± std)
    """
    if not USE_CV:
        # Fall back to single-split training
        return train_ensemble_single_split(X_tr, label, models, n_samples), {}, {}
    
    # Get grouping column
    if CV_GROUP_COLUMN not in meta.columns:
        if verbose:
            print(f"Warning: {CV_GROUP_COLUMN} not in meta, falling back to single-split training")
        return train_ensemble_single_split(X_tr, label, models, n_samples), {}, {}
    
    groups = meta[CV_GROUP_COLUMN].values
    X_tr_np = X_tr.to_numpy(np.float32, copy=False)
    
    all_models = {}
    all_oof_preds = {}
    all_cv_metrics = {}
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training with {CV_N_SPLITS}-fold GroupKFold CV")
        print(f"Grouping by: {CV_GROUP_COLUMN}")
        print(f"{'='*60}\n")
    
    for action in label.columns:
        y_raw = label[action].to_numpy()
        mask = ~pd.isna(y_raw)
        y_action = y_raw[mask].astype(int)
        X_action = X_tr_np[mask]
        groups_action = groups[mask]
        
        # Skip if no positive examples or insufficient data
        if (y_action == 0).all() or np.sum(y_action) < 5:
            continue
        
        if len(np.unique(y_action)) < 2:
            continue
        
        # Calculate positive rate for adaptive CV folds and model parameters
        pos_rate = float(np.mean(y_action == 1))
        n_pos_total = int(np.sum(y_action == 1))
        
        # Adaptive CV folds based on class distribution
        if pos_rate < 0.01:  # Very rare (< 1%)
            n_splits = 3
        elif pos_rate < 0.05:  # Rare (< 5%)
            n_splits = 5
        else:
            n_splits = CV_N_SPLITS
        
        # Adaptive model parameters for rare behaviors
        # For rare behaviors, we'll adjust model parameters dynamically during training
        adaptive_params = {}
        if pos_rate < 0.01:  # Very rare behaviors
            adaptive_params = {
                'min_child_samples': 3,
                'min_data_in_leaf': 3,
                'num_leaves_multiplier': min(4.0, 1.0 / max(pos_rate, 0.001))
            }
        elif pos_rate < 0.05:  # Rare behaviors
            adaptive_params = {
                'min_child_samples': 5,
                'min_data_in_leaf': 5,
                'num_leaves_multiplier': min(2.0, 1.0 / max(pos_rate, 0.01))
            }
        else:
            adaptive_params = {
                'min_child_samples': 5,
                'min_data_in_leaf': 5,
                'num_leaves_multiplier': 1.0
            }
        
        # Ensure we have enough groups for the number of splits
        n_unique_groups = len(np.unique(groups_action))
        n_splits = min(n_splits, n_unique_groups)
        
        if verbose:
            print(f"Training CV for action: {action} (pos_rate={pos_rate:.4f}, n_splits={n_splits})")
        
        # Create GroupKFold with adaptive splits
        gkf = GroupKFold(n_splits=n_splits)
        
        # Generate CV splits
        try:
            splits = list(gkf.split(X_action, y_action, groups_action))
        except ValueError as e:
            if verbose:
                print(f"  Warning: GroupKFold failed for {action}: {e}. Using single-split training.")
            # Fall back to single-split for this action
            idx = np.flatnonzero(mask)
            trained = []
            for m in models:
                m_clone = clone(m)
                m_clone.fit(X_tr_np[idx], y_action)
                trained.append(m_clone)
            all_models[action] = [trained]  # Single "fold"
            continue
        
        action_models = []  # List of model lists, one per fold
        action_oof_preds = []  # List of OOF prediction arrays
        action_oof_indices = []  # List of validation indices
        action_fold_metrics = []  # List of metrics per fold
        
        # Train on each fold
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            # Edge case check 1: Empty validation fold
            if len(val_idx) == 0:
                if verbose:
                    print(f"  Fold {fold_idx + 1}: Skipping (no validation samples)")
                continue
            
            X_train_fold = X_action[train_idx]
            y_train_fold = y_action[train_idx]
            X_val_fold = X_action[val_idx]
            y_val_fold = y_action[val_idx]
            
            # Edge case check 2: No positive examples in validation
            val_pos = np.sum(y_val_fold == 1)
            train_pos = np.sum(y_train_fold == 1)
            
            # Minimum positive sample checks for rare classes
            min_train_pos = 50  # Minimum positive samples in training
            min_val_pos = 10   # Minimum positive samples in validation
            
            if val_pos < min_val_pos:
                if verbose:
                    print(f"  Fold {fold_idx + 1}: Skipping (insufficient positive examples in validation: {val_pos} < {min_val_pos})")
                continue
            
            if train_pos < min_train_pos:
                if verbose:
                    print(f"  Fold {fold_idx + 1}: Skipping (insufficient positive examples in training: {train_pos} < {min_train_pos})")
                continue
            
            # Edge case check 3: Single class in training
            if len(np.unique(y_train_fold)) < 2:
                if verbose:
                    print(f"  Fold {fold_idx + 1}: Skipping (only one class in training)")
                continue
            
            if verbose:
                print(f"  Fold {fold_idx + 1}/{len(splits)}: train={len(train_idx)}, val={len(val_idx)} (pos={val_pos})")
            
            # Train models on this fold with enhanced error handling and retry logic
            fold_models = []
            fold_oof_preds = []
            model_success_count = 0
            
            for m in models:
                m_clone = clone(m)
                model_trained = False
                error_msg = None
                
                try:
                    # Fit model (early stopping is handled by model parameters and StratifiedSubsetClassifier)
                    # The improved model parameters (is_unbalance, min_child_samples, etc.) already help with training stability
                    m_clone.fit(X_train_fold, y_train_fold)
                    
                    # Get OOF predictions
                    val_proba = m_clone.predict_proba(X_val_fold)[:, 1]
                    fold_oof_preds.append(val_proba)
                    fold_models.append(m_clone)
                    model_trained = True
                    model_success_count += 1
                except Exception as e:
                    error_msg = str(e)
                    # Check if it's a LightGBM split error
                    if 'left_count' in error_msg or 'split' in error_msg.lower() or 'best_split_info' in error_msg:
                        # Retry with more lenient parameters for LightGBM
                        try:
                            # Check if it's a LightGBM model in a pipeline
                            if hasattr(m_clone, 'steps') and len(m_clone.steps) > 0:
                                # It's a pipeline, try to access the estimator
                                for step_name, step_estimator in m_clone.steps:
                                    if hasattr(step_estimator, 'estimator') and hasattr(step_estimator.estimator, 'min_child_samples'):
                                        step_estimator.estimator.set_params(min_child_samples=1, min_data_in_leaf=1, min_split_gain=0.0)
                                        m_clone.fit(X_train_fold, y_train_fold)
                                        val_proba = m_clone.predict_proba(X_val_fold)[:, 1]
                                        fold_oof_preds.append(val_proba)
                                        fold_models.append(m_clone)
                                        model_trained = True
                                        model_success_count += 1
                                        if verbose:
                                            print(f"    Retry successful with lenient parameters")
                                        break
                        except Exception as retry_e:
                            if verbose:
                                print(f"    Warning: Model training failed even with retry: {str(retry_e)[:100]}")
                    else:
                        if verbose:
                            print(f"    Warning: Model training failed: {error_msg[:100]}")
                
                if not model_trained and error_msg and verbose:
                    if 'left_count' not in error_msg:
                        print(f"    Warning: Model training failed: {error_msg[:100]}")
            
            # Ensure at least one model succeeds - if all failed, try with simplest possible model
            if not fold_models and len(X_train_fold) > 0:
                if verbose:
                    print(f"    All models failed, trying fallback with minimal LightGBM model")
                try:
                    # Fallback: simplest possible LightGBM model
                    fallback_model = lightgbm.LGBMClassifier(
                        n_estimators=50, learning_rate=0.1, max_depth=3,
                        num_leaves=7, min_child_samples=1, min_data_in_leaf=1,
                        min_split_gain=0.0, force_row_wise=True,
                        is_unbalance=True,
                        random_state=SEED, verbose=-1,
                        device='gpu' if LIGHTGBM_GPU_AVAILABLE else 'cpu'
                    )
                    fallback_model.fit(X_train_fold, y_train_fold)
                    val_proba = fallback_model.predict_proba(X_val_fold)[:, 1]
                    fold_oof_preds.append(val_proba)
                    fold_models.append(fallback_model)
                    model_success_count = 1
                    if verbose:
                        print(f"    Fallback model trained successfully")
                except Exception as fallback_e:
                    if verbose:
                        print(f"    Fallback model also failed: {str(fallback_e)[:100]}")
            
            if not fold_models:
                if verbose:
                    print(f"  Fold {fold_idx + 1}: No models trained successfully")
                continue
            
            # Average OOF predictions across models for this fold
            fold_oof_mean = np.mean(fold_oof_preds, axis=0)
            action_oof_preds.append(fold_oof_mean)
            action_oof_indices.append(val_idx)
            action_models.append(fold_models)
            
            # Compute fold metrics
            fold_metrics = {
                'prauc': average_precision_score(y_val_fold, fold_oof_mean),
                'f1': f1_score(y_val_fold, (fold_oof_mean >= 0.5).astype(int), zero_division=0),
            }
            action_fold_metrics.append(fold_metrics)
            
            if verbose:
                print(f"    PRAUC={fold_metrics['prauc']:.4f}, F1={fold_metrics['f1']:.4f}")
        
        if not action_models:
            if verbose:
                print(f"  Warning: No valid folds for {action}, skipping")
            continue
        
        # Store models (list of lists, one inner list per fold)
        all_models[action] = action_models
        
        # Aggregate OOF predictions
        oof_df = pd.DataFrame(index=np.where(mask)[0])
        for fold_idx, (val_idx, oof_pred) in enumerate(zip(action_oof_indices, action_oof_preds)):
            original_val_idx = np.where(mask)[0][val_idx]
            oof_df.loc[original_val_idx, f'fold_{fold_idx}'] = oof_pred
        
        fold_cols = [c for c in oof_df.columns if c.startswith('fold_')]
        if fold_cols:
            oof_df['oof_mean'] = oof_df[fold_cols].mean(axis=1)
        all_oof_preds[action] = oof_df
        
        # Aggregate CV metrics and optimize threshold
        if action_fold_metrics:
            # Optimize threshold using aggregated OOF predictions
            if 'oof_mean' in oof_df.columns and len(oof_df) > 0:
                try:
                    # Get true labels for OOF predictions
                    # oof_df.index contains original indices (from np.where(mask)[0])
                    # y_action is already filtered by mask, so we need to map indices correctly
                    mask_indices = np.where(mask)[0]  # Original indices where mask is True
                    
                    # Create mapping: original_index -> position in masked array
                    # mask_indices[i] gives the original index, and i is the position in y_action
                    index_to_masked = {orig_idx: i for i, orig_idx in enumerate(mask_indices)}
                    
                    # Filter out NaN predictions (rows that weren't in any validation fold)
                    valid_mask = ~oof_df['oof_mean'].isna()
                    if valid_mask.sum() > 0:
                        oof_df_valid = oof_df[valid_mask]
                        original_indices = oof_df_valid.index.values
                        oof_pred_mean = oof_df_valid['oof_mean'].values
                        
                        # Map oof_indices from original space to masked space
                        oof_indices_masked = np.array([index_to_masked[idx] for idx in original_indices if idx in index_to_masked])
                        
                        # Only use indices that were successfully mapped
                        if len(oof_indices_masked) > 0 and len(oof_indices_masked) == len(oof_pred_mean):
                            oof_true = y_action[oof_indices_masked]
                            # Only optimize if we have both classes
                            if len(np.unique(oof_true)) >= 2:
                                opt_threshold, opt_f1 = optimize_threshold(oof_true, oof_pred_mean)
                            else:
                                opt_threshold, opt_f1 = 0.5, 0.0
                        else:
                            opt_threshold, opt_f1 = 0.5, 0.0
                    else:
                        opt_threshold, opt_f1 = 0.5, 0.0
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Threshold optimization failed: {e}")
                    opt_threshold, opt_f1 = 0.5, 0.0
            else:
                opt_threshold, opt_f1 = 0.5, 0.0
            
            cv_metrics = {
                'prauc_mean': np.mean([m['prauc'] for m in action_fold_metrics]),
                'prauc_std': np.std([m['prauc'] for m in action_fold_metrics]),
                'f1_mean': np.mean([m['f1'] for m in action_fold_metrics]),
                'f1_std': np.std([m['f1'] for m in action_fold_metrics]),
                'n_folds': len(action_fold_metrics),
                'opt_threshold': opt_threshold,
                'opt_f1': opt_f1,
            }
            all_cv_metrics[action] = cv_metrics
            
            if verbose:
                print(f"  CV Summary: PRAUC={cv_metrics['prauc_mean']:.4f}±{cv_metrics['prauc_std']:.4f}, "
                      f"F1={cv_metrics['f1_mean']:.4f}±{cv_metrics['f1_std']:.4f} ({cv_metrics['n_folds']} folds)\n")
    
    # Convert to list format compatible with submit_ensemble
    model_list = [(action, fold_models_list) for action, fold_models_list in all_models.items()]
    
    return model_list, all_oof_preds, all_cv_metrics



def predict_multiclass_adaptive(pred, meta, action_thresholds):
    """Adaptive thresholding per action + temporal smoothing"""
    # Apply temporal smoothing
    pred_smoothed = pred.rolling(window=5, min_periods=1, center=True).mean()

    mode = 'pair'
    try:
        if 'target_id' in meta.columns and meta['target_id'].eq('self').all():
            mode = 'single'
    except Exception:
        pass

    ama = np.argmax(pred_smoothed, axis=1)
    th_map = _select_threshold_map(action_thresholds, mode)

    max_probs = pred_smoothed.max(axis=1)
    threshold_mask = np.zeros(len(pred_smoothed), dtype=bool)
    for i, action in enumerate(pred_smoothed.columns):
        action_mask = (ama == i)
        threshold = th_map[action]
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
    
    # Filter out very short events (likely noise)
    duration = submission_part.stop_frame - submission_part.start_frame
    submission_part = submission_part[duration >= 3].reset_index(drop=True)
    
    if len(submission_part) > 0:
        assert (submission_part.stop_frame > submission_part.start_frame).all(), 'stop <= start'
    
    if verbose: 
        print(f'  actions found: {len(submission_part)}')
    return submission_part



def robustify(submission, dataset, traintest, traintest_directory=None):
    """Ensures all videos have at least one prediction and cleans overlaps."""
    if traintest_directory is None:
        if IS_KAGGLE:
            traintest_directory = os.path.join(BASE_PATH, f"{traintest}_tracking")
        else:
            traintest_directory = os.path.join(TRACKING_BASE_PATH, f"{traintest}_tracking")

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
    for _, row in dataset.iterrows():
        lab_id = row['lab_id']
        video_id = row['video_id']
        if (submission.video_id == video_id).any():
            continue

        if verbose:
            print(f"Video {video_id} has no predictions")

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        try:
            vid = pd.read_parquet(path)
        except FileNotFoundError:
            continue

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



# Load training data
train = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)

body_parts_tracked_list = list(np.unique(train.body_parts_tracked))

# Store trained models, CV metrics, and OOF predictions
all_trained_models = {}
all_cv_metrics_global = {}
all_oof_predictions_global = {}

print(f"Found {len(body_parts_tracked_list)} body part configurations\n")

# Process each body part configuration
for section in range(1, len(body_parts_tracked_list)):
    body_parts_tracked_str = body_parts_tracked_list[section]
    try:
        body_parts_tracked = json.loads(body_parts_tracked_str)
        print(f"{section}. Processing: {len(body_parts_tracked)} body parts")
        if len(body_parts_tracked) > 5:
            body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

        train_subset = train[train.body_parts_tracked == body_parts_tracked_str]
        
        if LOCAL_TESTING and len(train_subset) > MAX_VIDEOS_LOCAL:
            train_subset = train_subset.head(MAX_VIDEOS_LOCAL)
            if verbose:
                print(f"  [Local testing] Using first {MAX_VIDEOS_LOCAL} videos")

        _fps_lookup = (
            train_subset[['video_id', 'frames_per_second']]
            .drop_duplicates('video_id')
            .set_index('video_id')['frames_per_second']
            .to_dict()
        )

        # Process single mouse behaviors
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

        # Train single mouse models
        if len(single_list) > 0:
            single_feats_parts = []
            for data_i, meta_i in zip(single_list, single_meta_list):
                fps_i = _fps_from_meta(meta_i, _fps_lookup, default_fps=30.0)
                Xi = transform_single(data_i, body_parts_tracked, fps_i).astype(np.float32)
                single_feats_parts.append(Xi)

            X_tr = pd.concat(single_feats_parts, axis=0, ignore_index=True)
            single_label = pd.concat(single_label_list, axis=0, ignore_index=True)
            single_meta = pd.concat(single_meta_list, axis=0, ignore_index=True)

            del single_list, single_label_list, single_meta_list, single_feats_parts
            gc.collect()

            print(f"  Single: {X_tr.shape}")
            
            # Create models - 3 diverse LightGBM models for better ensemble
            models = []
            # Model 1: Conservative, balanced
            models.append(make_pipeline(
                StratifiedSubsetClassifier(
                    lightgbm.LGBMClassifier(
                        n_estimators=300, learning_rate=0.05, max_depth=6,
                        num_leaves=31, min_child_samples=5, min_data_in_leaf=5,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=0.1,
                        min_split_gain=0.0, force_row_wise=True,
                        is_unbalance=True,
                        random_state=SEED, verbose=-1, 
                        device='gpu' if LIGHTGBM_GPU_AVAILABLE else 'cpu'
                    ), n_samples=LOCAL_N_SAMPLES if LOCAL_TESTING else 2_000_000
                )
            ))
            # Model 2: Deeper, more complex
            models.append(make_pipeline(
                StratifiedSubsetClassifier(
                    lightgbm.LGBMClassifier(
                        n_estimators=200, learning_rate=0.08, max_depth=8,
                        num_leaves=63, min_child_samples=5, min_data_in_leaf=5,
                        subsample=0.75, colsample_bytree=0.85,
                        reg_alpha=0.2, reg_lambda=0.2,
                        min_split_gain=0.0, force_row_wise=True,
                        is_unbalance=True,
                        random_state=SEED, verbose=-1, 
                        device='gpu' if LIGHTGBM_GPU_AVAILABLE else 'cpu'
                    ), n_samples=LOCAL_N_SAMPLES if LOCAL_TESTING else int(2_000_000 * 0.9)
                )
            ))
            # Model 3: Larger leaves, more regularization
            models.append(make_pipeline(
                StratifiedSubsetClassifier(
                    lightgbm.LGBMClassifier(
                        n_estimators=150, learning_rate=0.1, max_depth=10,
                        num_leaves=127, min_child_samples=5, min_data_in_leaf=5,
                        subsample=0.7, colsample_bytree=0.9,
                        reg_alpha=0.3, reg_lambda=0.3,
                        min_split_gain=0.0, force_row_wise=True,
                        is_unbalance=True,
                        random_state=SEED, verbose=-1, 
                        device='gpu' if LIGHTGBM_GPU_AVAILABLE else 'cpu'
                    ), n_samples=LOCAL_N_SAMPLES if LOCAL_TESTING else int(2_000_000 * 0.8)
                )
            ))
            
            if XGBOOST_AVAILABLE:
                models.append(make_pipeline(
                    StratifiedSubsetClassifier(
                        XGBClassifier(
                            n_estimators=200, learning_rate=0.1, max_depth=7,
                            random_state=SEED, 
                            tree_method='gpu_hist' if XGBOOST_GPU_AVAILABLE else 'hist', 
                            verbosity=0
                        ), n_samples=LOCAL_N_SAMPLES if LOCAL_TESTING else int(2_000_000/1.5)
                    )
                ))
            
            if CATBOOST_AVAILABLE:
                models.append(make_pipeline(
                    StratifiedSubsetClassifier(
                        CatBoostClassifier(
                            iterations=120, learning_rate=0.1, depth=6,
                            verbose=False, allow_writing_files=False, 
                            task_type='GPU' if CATBOOST_GPU_AVAILABLE else 'CPU',
                            random_seed=SEED
                        ), n_samples=LOCAL_N_SAMPLES if LOCAL_TESTING else 2_000_000
                    )
                ))
            
            # Train with CV
            model_list, oof_preds, cv_metrics = train_ensemble_with_cv(
                X_tr, single_label, single_meta, models, 
                LOCAL_N_SAMPLES if LOCAL_TESTING else 2_000_000,
                'single', body_parts_tracked_str
            )
            
            all_trained_models[(body_parts_tracked_str, 'single')] = model_list
            all_cv_metrics_global[(body_parts_tracked_str, 'single')] = cv_metrics
            all_oof_predictions_global[(body_parts_tracked_str, 'single')] = oof_preds

            del X_tr, single_label, single_meta
            gc.collect()

        # Train pair mouse models
        if len(pair_list) > 0:
            pair_feats_parts = []
            for data_i, meta_i in zip(pair_list, pair_meta_list):
                fps_i = _fps_from_meta(meta_i, _fps_lookup, default_fps=30.0)
                Xi = transform_pair(data_i, body_parts_tracked, fps_i).astype(np.float32)
                pair_feats_parts.append(Xi)

            X_tr = pd.concat(pair_feats_parts, axis=0, ignore_index=True)
            pair_label = pd.concat(pair_label_list, axis=0, ignore_index=True)
            pair_meta = pd.concat(pair_meta_list, axis=0, ignore_index=True)

            del pair_list, pair_label_list, pair_meta_list, pair_feats_parts
            gc.collect()

            print(f"  Pair: {X_tr.shape}")
            
            # Create models - 3 diverse LightGBM models for better ensemble
            models = []
            # Model 1: Conservative, balanced
            models.append(make_pipeline(
                StratifiedSubsetClassifier(
                    lightgbm.LGBMClassifier(
                        n_estimators=300, learning_rate=0.05, max_depth=6,
                        num_leaves=31, min_child_samples=5, min_data_in_leaf=5,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=0.1,
                        min_split_gain=0.0, force_row_wise=True,
                        is_unbalance=True,
                        random_state=SEED, verbose=-1, 
                        device='gpu' if LIGHTGBM_GPU_AVAILABLE else 'cpu'
                    ), n_samples=LOCAL_N_SAMPLES if LOCAL_TESTING else 900_000
                )
            ))
            # Model 2: Deeper, more complex
            models.append(make_pipeline(
                StratifiedSubsetClassifier(
                    lightgbm.LGBMClassifier(
                        n_estimators=200, learning_rate=0.08, max_depth=8,
                        num_leaves=63, min_child_samples=5, min_data_in_leaf=5,
                        subsample=0.75, colsample_bytree=0.85,
                        reg_alpha=0.2, reg_lambda=0.2,
                        min_split_gain=0.0, force_row_wise=True,
                        is_unbalance=True,
                        random_state=SEED, verbose=-1, 
                        device='gpu' if LIGHTGBM_GPU_AVAILABLE else 'cpu'
                    ), n_samples=LOCAL_N_SAMPLES if LOCAL_TESTING else int(900_000 * 0.9)
                )
            ))
            # Model 3: Larger leaves, more regularization
            models.append(make_pipeline(
                StratifiedSubsetClassifier(
                    lightgbm.LGBMClassifier(
                        n_estimators=150, learning_rate=0.1, max_depth=10,
                        num_leaves=127, min_child_samples=5, min_data_in_leaf=5,
                        subsample=0.7, colsample_bytree=0.9,
                        reg_alpha=0.3, reg_lambda=0.3,
                        min_split_gain=0.0, force_row_wise=True,
                        is_unbalance=True,
                        random_state=SEED, verbose=-1, 
                        device='gpu' if LIGHTGBM_GPU_AVAILABLE else 'cpu'
                    ), n_samples=LOCAL_N_SAMPLES if LOCAL_TESTING else int(900_000 * 0.8)
                )
            ))
            
            if XGBOOST_AVAILABLE:
                models.append(make_pipeline(
                    StratifiedSubsetClassifier(
                        XGBClassifier(
                            n_estimators=200, learning_rate=0.1, max_depth=7,
                            random_state=SEED, 
                            tree_method='gpu_hist' if XGBOOST_GPU_AVAILABLE else 'hist', 
                            verbosity=0
                        ), n_samples=LOCAL_N_SAMPLES if LOCAL_TESTING else int(900_000/1.5)
                    )
                ))
            
            if CATBOOST_AVAILABLE:
                models.append(make_pipeline(
                    StratifiedSubsetClassifier(
                        CatBoostClassifier(
                            iterations=120, learning_rate=0.1, depth=6,
                            verbose=False, allow_writing_files=False, 
                            task_type='GPU' if CATBOOST_GPU_AVAILABLE else 'CPU',
                            random_seed=SEED
                        ), n_samples=LOCAL_N_SAMPLES if LOCAL_TESTING else 900_000
                    )
                ))
            
            # Train with CV
            model_list, oof_preds, cv_metrics = train_ensemble_with_cv(
                X_tr, pair_label, pair_meta, models,
                LOCAL_N_SAMPLES if LOCAL_TESTING else 900_000,
                'pair', body_parts_tracked_str
            )
            
            all_trained_models[(body_parts_tracked_str, 'pair')] = model_list
            all_cv_metrics_global[(body_parts_tracked_str, 'pair')] = cv_metrics
            all_oof_predictions_global[(body_parts_tracked_str, 'pair')] = oof_preds

            del X_tr, pair_label, pair_meta
            gc.collect()

    except Exception as e:
        print(f'***Exception*** {str(e)[:100]}')
        import traceback
        if verbose:
            traceback.print_exc()

    gc.collect()
    print()

# Display CV metrics summary
if verbose and all_cv_metrics_global:
    print("\n" + "="*60)
    print("CV Metrics Summary")
    print("="*60)
    for key, metrics in all_cv_metrics_global.items():
        if metrics:
            print(f"\n{key[0][:30]}... ({key[1]}):")
            for action, cv_metrics in metrics.items():
                print(f"  {action}: PRAUC={cv_metrics['prauc_mean']:.4f}±{cv_metrics['prauc_std']:.4f}, "
                      f"F1={cv_metrics['f1_mean']:.4f}±{cv_metrics['f1_std']:.4f} ({cv_metrics['n_folds']} folds)")

# Save CV metrics to CSV file
if all_cv_metrics_global:
    cv_results_list = []
    for key, metrics in all_cv_metrics_global.items():
        body_parts_str, mode = key
        if metrics:
            for action, cv_metrics in metrics.items():
                cv_results_list.append({
                    'body_parts_tracked': body_parts_str,
                    'mode': mode,
                    'action': action,
                    'prauc_mean': cv_metrics['prauc_mean'],
                    'prauc_std': cv_metrics['prauc_std'],
                    'f1_mean': cv_metrics['f1_mean'],
                    'f1_std': cv_metrics['f1_std'],
                    'n_folds': cv_metrics['n_folds'],
                    'opt_threshold': cv_metrics.get('opt_threshold', 0.5),
                    'opt_f1': cv_metrics.get('opt_f1', 0.0)
                })
    
    if cv_results_list:
        cv_results_df = pd.DataFrame(cv_results_list)
        cv_results_df = cv_results_df.sort_values(['body_parts_tracked', 'mode', 'action'])
        
        # Save to CSV
        cv_output_path = 'cv_results.csv' if IS_KAGGLE else os.path.join('..', 'cv_results.csv')
        cv_results_df.to_csv(cv_output_path, index=False)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"CV Results saved to: {cv_output_path}")
            print(f"Total actions evaluated: {len(cv_results_df)}")
            print(f"Average PRAUC: {cv_results_df['prauc_mean'].mean():.4f}")
            print(f"Average F1: {cv_results_df['f1_mean'].mean():.4f}")
            print(f"{'='*60}\n")

# Save OOF (Out-of-Fold) predictions
if all_oof_predictions_global:
    oof_output_path = 'oof_predictions.parquet' if IS_KAGGLE else os.path.join('..', 'oof_predictions.parquet')
    oof_csv_path = 'oof_predictions.csv' if IS_KAGGLE else os.path.join('..', 'oof_predictions.csv')
    
    # Collect all OOF predictions with metadata
    oof_data_list = []
    for key, oof_preds_dict in all_oof_predictions_global.items():
        body_parts_str, mode = key
        if oof_preds_dict:
            for action, oof_df in oof_preds_dict.items():
                # Add metadata columns
                oof_df_copy = oof_df.copy()
                oof_df_copy['body_parts_tracked'] = body_parts_str
                oof_df_copy['mode'] = mode
                oof_df_copy['action'] = action
                oof_df_copy = oof_df_copy.reset_index()
                oof_df_copy.rename(columns={'index': 'sample_index'}, inplace=True)
                oof_data_list.append(oof_df_copy)
    
    if oof_data_list:
        # Combine all OOF predictions
        oof_combined = pd.concat(oof_data_list, ignore_index=True)
        
        # Save as Parquet (more efficient for large data)
        try:
            oof_combined.to_parquet(oof_output_path, index=False, engine='pyarrow')
            parquet_saved = True
        except Exception as e:
            if verbose:
                print(f"Warning: Could not save Parquet file ({str(e)[:50]}), saving as CSV only")
            parquet_saved = False
        
        # Also save as CSV for easy viewing (may be large)
        # Only save CSV if not too large (limit to avoid memory issues)
        if len(oof_combined) < 10_000_000:  # 10M rows limit
            oof_combined.to_csv(oof_csv_path, index=False)
            if verbose:
                if parquet_saved:
                    print(f"OOF Predictions saved to:")
                    print(f"  - Parquet: {oof_output_path}")
                    print(f"  - CSV: {oof_csv_path}")
                else:
                    print(f"OOF Predictions saved to: {oof_csv_path}")
        else:
            if verbose:
                if parquet_saved:
                    print(f"OOF Predictions saved to: {oof_output_path} (CSV skipped - too large: {len(oof_combined):,} rows)")
                else:
                    print(f"OOF Predictions saved to: {oof_csv_path} (Parquet failed, CSV only)")
        
        if verbose:
            print(f"Total OOF samples: {len(oof_combined):,}")
            print(f"Actions with OOF predictions: {oof_combined['action'].nunique()}")
            print(f"Body part configurations: {oof_combined['body_parts_tracked'].nunique()}")
            print(f"{'='*60}\n")

# ==================== AUTO-APPLY OPTIMIZED THRESHOLDS ====================
# Automatically extract and apply optimized thresholds after training completes
# This ensures thresholds are ready for inference in non-interactive environments (e.g., Kaggle)

if all_cv_metrics_global:
    optimized_thresholds = {}
    
    for key, metrics in all_cv_metrics_global.items():
        body_parts_str, mode = key
        if metrics:
            for action, cv_metrics in metrics.items():
                if 'opt_threshold' in cv_metrics and cv_metrics['opt_threshold'] is not None:
                    opt_thresh = float(cv_metrics['opt_threshold'])
                    action_key = f"{mode}_{action}"
                    if action_key not in optimized_thresholds:
                        optimized_thresholds[action_key] = []
                    optimized_thresholds[action_key].append(opt_thresh)
    
    if optimized_thresholds:
        if 'single' not in action_thresholds:
            action_thresholds['single'] = {}
        if 'pair' not in action_thresholds:
            action_thresholds['pair'] = {}
        
        for action_key, thresholds_list in optimized_thresholds.items():
            if len(thresholds_list) > 0:
                avg_threshold = np.mean(thresholds_list)
                avg_threshold = np.clip(avg_threshold, 0.15, 0.50)
                
                if action_key.startswith('single_'):
                    action = action_key.replace('single_', '')
                    action_thresholds['single'][action] = float(avg_threshold)
                elif action_key.startswith('pair_'):
                    action = action_key.replace('pair_', '')
                    action_thresholds['pair'][action] = float(avg_threshold)
        
        if verbose:
            print("\n" + "="*60)
            print("✅ Optimized Thresholds Auto-Applied from CV")
            print("="*60)
            if action_thresholds.get('single'):
                print("\nSingle behavior thresholds:")
                for action, thresh in sorted(action_thresholds['single'].items()):
                    print(f"  {action}: {thresh:.4f}")
            if action_thresholds.get('pair'):
                print("\nPair behavior thresholds:")
                for action, thresh in sorted(action_thresholds['pair'].items()):
                    print(f"  {action}: {thresh:.4f}")
            print("="*60 + "\n")



# ==================== THRESHOLD VERIFICATION BEFORE INFERENCE ====================
# Verify that optimized thresholds are present and will be used during inference

print("\n" + "="*60)
print("FINAL THRESHOLDS FOR INFERENCE")
print("="*60)

if 'action_thresholds' in globals() and isinstance(action_thresholds, dict):
    print(f"\nDefault thresholds:")
    print(f"  Global default: {action_thresholds.get('default', 'N/A')}")
    print(f"  Single default: {action_thresholds.get('single_default', 'N/A')}")
    print(f"  Pair default: {action_thresholds.get('pair_default', 'N/A')}")
    
    single_thresholds = action_thresholds.get('single', {})
    pair_thresholds = action_thresholds.get('pair', {})
    
    if single_thresholds:
        print(f"\nSingle behavior thresholds ({len(single_thresholds)} actions):")
        for action, thresh in sorted(single_thresholds.items()):
            print(f"  {action:20s}: {thresh:.4f}")
    else:
        print("\nSingle behavior thresholds: None (will use default)")
    
    if pair_thresholds:
        print(f"\nPair behavior thresholds ({len(pair_thresholds)} actions):")
        for action, thresh in sorted(pair_thresholds.items()):
            print(f"  {action:20s}: {thresh:.4f}")
    else:
        print("\nPair behavior thresholds: None (will use default)")
    
    # Check if we have optimized thresholds from CV
    if 'all_cv_metrics_global' in globals() and all_cv_metrics_global:
        total_actions = sum(len(metrics) for metrics in all_cv_metrics_global.values() if metrics)
        print(f"\nCV Metrics Summary:")
        print(f"  Total actions with CV metrics: {total_actions}")
        print(f"  Actions with optimized thresholds: {len(single_thresholds) + len(pair_thresholds)}")
        
        if len(single_thresholds) + len(pair_thresholds) < total_actions:
            print(f"  WARNING: {total_actions - len(single_thresholds) - len(pair_thresholds)} actions missing optimized thresholds!")
            print(f"  These will use default thresholds during inference.")
    else:
        print("\nWARNING: CV metrics not available - thresholds may not be optimized!")
else:
    print("ERROR: action_thresholds not defined or invalid!")
    print("Inference will fail or use hardcoded defaults.")

print("="*60 + "\n")



if validate_or_submit == 'submit':
    # Load test data
    test = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))
    submission_list = []
    
    print(f"Generating predictions for {len(test)} test videos\n")
    
    # Process each body part configuration
    for section in range(1, len(body_parts_tracked_list)):
        body_parts_tracked_str = body_parts_tracked_list[section]
        try:
            body_parts_tracked = json.loads(body_parts_tracked_str)
            print(f"{section}. Processing: {len(body_parts_tracked)} body parts")
            if len(body_parts_tracked) > 5:
                body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

            test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
            
            if LOCAL_TESTING and len(test_subset) > MAX_VIDEOS_LOCAL:
                test_subset = test_subset.head(MAX_VIDEOS_LOCAL)
                if verbose:
                    print(f"  [Local testing] Using first {MAX_VIDEOS_LOCAL} videos")

            fps_lookup = (
                test_subset[['video_id', 'frames_per_second']]
                .drop_duplicates('video_id')
                .set_index('video_id')['frames_per_second']
                .to_dict()
            )

            # Process single mouse behaviors
            for switch_te, data_te, meta_te, actions_te in generate_mouse_data(
                test_subset, 'test',
                generate_single=True,
                generate_pair=False
            ):
                if switch_te != 'single':
                    continue
                
                key = (body_parts_tracked_str, 'single')
                if key not in all_trained_models:
                    continue
                
                model_list = all_trained_models[key]
                
                try:
                    fps_i = _fps_from_meta(meta_te, fps_lookup, default_fps=30.0)
                    X_te = transform_single(data_te, body_parts_tracked, fps_i).astype(np.float32)
                    X_te_np = X_te.to_numpy(np.float32, copy=False)
                    del X_te, data_te
                    gc.collect()

                    pred = pd.DataFrame(index=meta_te.video_frame)
                    for action, trained in model_list:
                        if action in actions_te:
                            # Handle CV models: trained is a list of fold model lists
                            if len(trained) > 1:
                                # CV mode: average across folds, then across models within each fold
                                fold_probs = []
                                for fold_models in trained:
                                    fold_model_probs = [m.predict_proba(X_te_np)[:, 1] for m in fold_models]
                                    fold_probs.append(np.mean(fold_model_probs, axis=0))
                                # Average across folds
                                pred[action] = np.mean(fold_probs, axis=0)
                            else:
                                # Single-split mode: unwrap the single fold and average across models
                                fold_models = trained[0]
                                probs = [m.predict_proba(X_te_np)[:, 1] for m in fold_models]
                                pred[action] = np.mean(probs, axis=0)

                    del X_te_np
                    gc.collect()

                    if pred.shape[1] != 0:
                        sub_part = predict_multiclass_adaptive(pred, meta_te, action_thresholds)
                        submission_list.append(sub_part)
                    else:
                        if verbose:
                            print("  ERROR: no training data")

                except Exception as e:
                    if verbose:
                        print(f"  ERROR: {str(e)[:50]}")
                    try:
                        del data_te
                    except:
                        pass
                    gc.collect()

            # Process pair mouse behaviors
            for switch_te, data_te, meta_te, actions_te in generate_mouse_data(
                test_subset, 'test',
                generate_single=False,
                generate_pair=True
            ):
                if switch_te != 'pair':
                    continue
                
                key = (body_parts_tracked_str, 'pair')
                if key not in all_trained_models:
                    continue
                
                model_list = all_trained_models[key]
                
                try:
                    fps_i = _fps_from_meta(meta_te, fps_lookup, default_fps=30.0)
                    X_te = transform_pair(data_te, body_parts_tracked, fps_i).astype(np.float32)
                    X_te_np = X_te.to_numpy(np.float32, copy=False)
                    del X_te, data_te
                    gc.collect()

                    pred = pd.DataFrame(index=meta_te.video_frame)
                    for action, trained in model_list:
                        if action in actions_te:
                            # Handle CV models: trained is a list of fold model lists
                            if len(trained) > 1:
                                # CV mode: average across folds, then across models within each fold
                                fold_probs = []
                                for fold_models in trained:
                                    fold_model_probs = [m.predict_proba(X_te_np)[:, 1] for m in fold_models]
                                    fold_probs.append(np.mean(fold_model_probs, axis=0))
                                # Average across folds
                                pred[action] = np.mean(fold_probs, axis=0)
                            else:
                                # Single-split mode: unwrap the single fold and average across models
                                fold_models = trained[0]
                                probs = [m.predict_proba(X_te_np)[:, 1] for m in fold_models]
                                pred[action] = np.mean(probs, axis=0)

                    del X_te_np
                    gc.collect()

                    if pred.shape[1] != 0:
                        sub_part = predict_multiclass_adaptive(pred, meta_te, action_thresholds)
                        submission_list.append(sub_part)
                    else:
                        if verbose:
                            print("  ERROR: no training data")

                except Exception as e:
                    if verbose:
                        print(f"  ERROR: {str(e)[:50]}")
                    try:
                        del data_te
                    except:
                        pass
                    gc.collect()

        except Exception as e:
            print(f'***Exception*** {str(e)[:100]}')
            import traceback
            if verbose:
                traceback.print_exc()

        gc.collect()
        print()

    # Combine all predictions
    if len(submission_list) > 0:
        submission = pd.concat(submission_list, ignore_index=True)
    else:
        # Fallback: create minimal submission
        submission = pd.DataFrame({
            'video_id': [test.video_id.iloc[0] if len(test) > 0 else 438887472],
            'agent_id': ['mouse1'],
            'target_id': ['self'],
            'action': ['rear'],
            'start_frame': [278],
            'stop_frame': [500]
        })

    # Apply robustify post-processing
    submission = robustify(submission, test, 'test')
    
    print(f"\nGenerated {len(submission)} predictions")
    print(f"Unique videos: {submission.video_id.nunique()}")
    print(f"Unique actions: {submission.action.nunique()}")
    
else:
    print("Skipping inference (validate_or_submit='validate')")
    submission = pd.DataFrame()



if validate_or_submit == 'submit' and len(submission) > 0:
    # Format submission for Kaggle (add row_id, ensure proper types and order)
    submission['row_id'] = range(len(submission))
    
    # Ensure proper column order (Kaggle format)
    required_columns = ['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
    submission = submission[required_columns]
    
    # Cast numeric columns to int (required for Kaggle)
    submission['row_id'] = submission['row_id'].astype(int)
    submission['video_id'] = submission['video_id'].astype(int)
    submission['start_frame'] = submission['start_frame'].astype(int)
    submission['stop_frame'] = submission['stop_frame'].astype(int)
    
    # Sort by video_id and start_frame for consistency
    submission = submission.sort_values(['video_id', 'start_frame']).reset_index(drop=True)
    
    # Reset row_id to be sequential after sorting
    submission['row_id'] = range(len(submission))
    
    # Save submission
    output_path = 'submission.csv'  # Always save to current directory for Kaggle compatibility
    submission.to_csv(output_path, index=False)
    
    print(f"\nSubmission saved to: {output_path}")
    print(f"\nSubmission preview:")
    print(submission.head(10))
    print(f"\nSubmission statistics:")
    print(f"  Total predictions: {len(submission)}")
    print(f"  Unique videos: {submission.video_id.nunique()}")
    print(f"  Unique actions: {submission.action.nunique()}")
    print(f"  Actions distribution:")
    print(submission.action.value_counts().head(10))
    print(f"\nSubmission format validation:")
    print(f"  Columns: {list(submission.columns)}")
    print(f"  Data types:")
    print(submission.dtypes)
else:
    print("No submission generated (validate mode or empty submission)")


