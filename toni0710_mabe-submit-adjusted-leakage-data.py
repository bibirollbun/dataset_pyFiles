#============= SETTINGS =============#
verbose = True
STATIC_THRESHOLD = 0.27
RANDOM_STATE = 42

#============= IMPORTS =============#
import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict
import itertools
import gc
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d

from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score

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

#============= SEEDS =============#
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
rnd = np.random.RandomState(SEED)
import random 
random.seed(SEED)
np.random.seed(SEED)

import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['OMP_NUM_THREADS'] = '1'
pd.set_option('compute.use_numexpr', False)

#============= KAGGLE PATHS & DATA =============#
BASE_PATH = '/kaggle/input/MABe-mouse-behavior-detection'
TRAIN_TRACKING_DIR = os.path.join(BASE_PATH, 'train_tracking')
TEST_TRACKING_DIR = os.path.join(BASE_PATH, 'test_tracking')
TRAIN_ANNOTATION_DIR = os.path.join(BASE_PATH, 'train_annotation')

DROP_BODY_PARTS = [
    'headpiece_bottombackleft','headpiece_bottombackright','headpiece_bottomfrontleft','headpiece_bottomfrontright',
    'headpiece_topbackleft','headpiece_topbackright','headpiece_topfrontleft','headpiece_topfrontright',
    'spine_1','spine_2','tail_middle_1','tail_middle_2','tail_midpoint'
]

#============= LOADING DATA =============#
train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)
train_without_mabe22 = train.query("~ lab_id.str.startswith('MABe22_')")

test = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')
body_parts_tracked_list = list(np.unique(train.body_parts_tracked))
#============= DATA GENERATOR =============#
def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    assert traintest in ['train', 'test']

    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    for _, row in dataset.iterrows():
        lab_id = row.lab_id
        video_id = row.video_id

        if type(row.behaviors_labeled) != str:
            if verbose:
                print('No labeled behaviors:', lab_id, video_id)
            continue

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query('~ bodypart.isin(@DROP_BODY_PARTS)')

        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        if pvid.isna().any().any():
            if verbose and traintest == 'test':
                print('    Video with missing values', video_id, traintest, len(vid), 'frames')
        else:
            if verbose and traintest == 'test':
                print('    Video with all values', video_id, traintest, len(vid), 'frames')

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
                    vid_agent_actions = np.unique(vid_behaviors_subset.query('agent == @mouse_id_str').action)
                    single_mouse = pvid.loc[:, mouse_id]
                    assert len(single_mouse) == len(pvid)
                    single_mouse_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': mouse_id_str,
                        'target_id': 'self',
                        'video_frame': single_mouse.index,
                        'frames_per_second': row.frames_per_second
                    })
                    if traintest == 'train':
                        single_mouse_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=single_mouse.index)
                        annot_subset = annot.query('(agent_id == @mouse_id) & (target_id == @mouse_id)')
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            single_mouse_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'single', single_mouse, single_mouse_meta, single_mouse_label
                    else:
                        if verbose: print('- Test single', video_id, mouse_id)
                        yield 'single', single_mouse, single_mouse_meta, vid_agent_actions
                except KeyError:
                    pass

        if generate_pair:
            vid_behaviors_subset = vid_behaviors.query("target != 'self'")
            if len(vid_behaviors_subset) > 0:
                for agent, target in itertools.permutations(np.unique(pvid.columns.get_level_values('mouse_id')), 2):
                    agent_str = f"mouse{agent}"
                    target_str = f"mouse{target}"
                    vid_agent_actions = np.unique(vid_behaviors_subset.query('(agent == @agent_str) & (target == @target_str)').action)
                    mouse_pair = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                    assert len(mouse_pair) == len(pvid)
                    mouse_pair_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': agent_str,
                        'target_id': target_str,
                        'video_frame': mouse_pair.index,
                        'frames_per_second': row.frames_per_second
                    })
                    if traintest == 'train':
                        mouse_pair_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=mouse_pair.index)
                        annot_subset = annot.query('(agent_id == @agent) & (target_id == @target)')
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            mouse_pair_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'pair', mouse_pair, mouse_pair_meta, mouse_pair_label
                    else:
                        if verbose: print('- Test pair', video_id, agent, target)
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions
#============= FEATURE ENGINEERING =============#
def safe_rolling(series, window, func, min_periods=None):
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series.rolling(window, min_periods=min_periods, center=True).apply(func, raw=True)

def _scale(n_frames_at_30fps, fps, ref=30.0):
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))

def _scale_signed(n_frames_at_30fps, fps, ref=30.0):
    if n_frames_at_30fps == 0:
        return 0
    s = 1 if n_frames_at_30fps > 0 else -1
    mag = max(1, int(round(abs(n_frames_at_30fps) * float(fps) / ref)))
    return s * mag

def _fps_from_meta(meta_df, fallback_lookup, default_fps=30.0):
    if 'frames_per_second' in meta_df.columns and pd.notnull(meta_df['frames_per_second']).any():
        return float(meta_df['frames_per_second'].iloc[0])
    vid = meta_df['video_id'].iloc[0]
    return float(fallback_lookup.get(vid, default_fps))

def add_curvature_features(X, center_x, center_y, fps):
    vel_x = center_x.diff()
    vel_y = center_y.diff()
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()

    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)

    for w in [30, 60, 75]:
        ws = _scale(w, fps)
        X[f'curv_mean_{w}'] = curvature.rolling(ws, min_periods=max(1, ws // 6)).mean()

    angle = np.arctan2(vel_y, vel_x)
    angle_change = np.abs(angle.diff())
    w = 30
    ws = _scale(w, fps)
    X[f'turn_rate_{w}'] = angle_change.rolling(ws, min_periods=max(1, ws // 6)).sum()

    return X

def add_multiscale_features(X, center_x, center_y, fps):
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)

    scales = [20, 40, 80, 160]
    for scale in scales:
        ws = _scale(scale, fps)
        if len(speed) >= ws:
            X[f'sp_m{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).mean()
            X[f'sp_s{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).std()

    if len(scales) >= 2 and f'sp_m{scales[0]}' in X.columns and f'sp_m{scales[-1]}' in X.columns:
        X['sp_ratio'] = X[f'sp_m{scales[0]}'] / (X[f'sp_m{scales[-1]}'] + 1e-6)

    return X

def add_state_features(X, center_x, center_y, fps):
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
    w_ma = _scale(15, fps)
    speed_ma = speed.rolling(w_ma, min_periods=max(1, w_ma // 3)).mean()

    try:
        bins = [-np.inf, 0.5 * fps, 2.0 * fps, 5.0 * fps, np.inf]
        speed_states = pd.cut(speed_ma, bins=bins, labels=[0, 1, 2, 3]).astype(float)

        for window in [20, 60, 80, 120]:
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
    for window in [120, 240]:
        ws = _scale(window, fps)
        if len(center_x) >= ws:
            X[f'x_ml{window}'] = center_x.rolling(ws, min_periods=max(5, ws // 6)).mean()
            X[f'y_ml{window}'] = center_y.rolling(ws, min_periods=max(5, ws // 6)).mean()

    for span in [30, 60, 120]:
        s = _scale(span, fps)
        X[f'x_e{span}'] = center_x.ewm(span=s, min_periods=1).mean()
        X[f'y_e{span}'] = center_y.ewm(span=s, min_periods=1).mean()

    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)

    for window in [30, 60, 120]:
        ws = _scale(window, fps)
        if len(speed) >= ws:
            X[f'sp_pct{window}'] = speed.rolling(ws, min_periods=max(5, ws // 6)).rank(pct=True)

    return X

def add_interaction_features(X, mouse_pair, avail_A, avail_B, fps):
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

def add_relative_heading_features(X, mouse_pair, fps):
    try:
        has_A = all(p in mouse_pair['A'].columns.get_level_values(0) for p in ['nose', 'tail_base'])
        has_B = all(p in mouse_pair['B'].columns.get_level_values(0) for p in ['nose', 'tail_base'])
        
        if has_A and has_B:
            Ax = mouse_pair['A']['nose']['x'] - mouse_pair['A']['tail_base']['x']
            Ay = mouse_pair['A']['nose']['y'] - mouse_pair['A']['tail_base']['y']
            Bx = mouse_pair['B']['nose']['x'] - mouse_pair['B']['tail_base']['x']
            By = mouse_pair['B']['nose']['y'] - mouse_pair['B']['tail_base']['y']
            
            ABx = mouse_pair['B']['body_center']['x'] - mouse_pair['A']['body_center']['x']
            ABy = mouse_pair['B']['body_center']['y'] - mouse_pair['A']['body_center']['y']
            
            norm_A = np.sqrt(Ax**2 + Ay**2) + 1e-6
            norm_B = np.sqrt(Bx**2 + By**2) + 1e-6
            norm_AB = np.sqrt(ABx**2 + ABy**2) + 1e-6
            
            X['heading_alignment'] = (Ax*Bx + Ay*By) / (norm_A * norm_B)
            
            X['A_facing_B'] = (Ax*ABx + Ay*ABy) / (norm_A * norm_AB)
            
            X['B_facing_A'] = (Bx*(-ABx) + By*(-ABy)) / (norm_B * norm_AB)
            
            w = _scale(15, fps)
            X['heading_alignment_sm'] = X['heading_alignment'].rolling(w, min_periods=1).mean()
            
    except Exception:
        pass
    return X

def add_acceleration_features(X, center_x, center_y, fps):
    try:
        vx = center_x.diff() * float(fps)
        vy = center_y.diff() * float(fps)
        speed = np.sqrt(vx**2 + vy**2)
        
        ax = vx.diff() * float(fps)
        ay = vy.diff() * float(fps)
        accel = np.sqrt(ax**2 + ay**2)
        
        jx = ax.diff() * float(fps)
        jy = ay.diff() * float(fps)
        jerk = np.sqrt(jx**2 + jy**2)
        
        for w in [15, 30, 60]:
            ws = _scale(w, fps)
            X[f'acc_m{w}'] = accel.rolling(ws, min_periods=max(1, ws//4)).mean()
            X[f'acc_s{w}'] = accel.rolling(ws, min_periods=max(1, ws//4)).std()
            X[f'jrk_m{w}'] = jerk.rolling(ws, min_periods=max(1, ws//4)).mean()
        
        w = _scale(10, fps)
        X['sudden_acc'] = (accel > accel.rolling(w, min_periods=1).mean() + 2 * accel.rolling(w, min_periods=1).std()).astype(float)
        
    except Exception:
        pass
    
    return X

def add_spatial_features(X, center_x, center_y, fps):
    try:
        dist_origin = np.sqrt(center_x**2 + center_y**2)
        
        for w in [60, 120]:
            ws = _scale(w, fps)
            X[f'orig_m{w}'] = dist_origin.rolling(ws, min_periods=max(1, ws//4)).mean()
            X[f'orig_s{w}'] = dist_origin.rolling(ws, min_periods=max(1, ws//4)).std()
        
        for w in [30, 60, 120]:
            ws = _scale(w, fps)
            x_range = center_x.rolling(ws, min_periods=max(1, ws//4)).max() - center_x.rolling(ws, min_periods=max(1, ws//4)).min()
            y_range = center_y.rolling(ws, min_periods=max(1, ws//4)).max() - center_y.rolling(ws, min_periods=max(1, ws//4)).min()
            X[f'area{w}'] = x_range * y_range
        
        for w in [30, 60]:
            ws = _scale(w, fps)
            straight_dist = np.sqrt((center_x - center_x.shift(ws))**2 + (center_y - center_y.shift(ws))**2)
            path_length = center_x.diff().abs().rolling(ws, min_periods=1).sum() + center_y.diff().abs().rolling(ws, min_periods=1).sum()
            X[f'eff{w}'] = straight_dist / (path_length + 1e-6)
            
    except Exception:
        pass
    
    return X

def add_frequency_features(X, center_x, center_y, fps):
    try:
        speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
        speed_centered = speed - speed.rolling(_scale(30, fps), min_periods=1).mean()
        
        for w in [30, 60]:
            ws = _scale(w, fps)
            zero_crossings = ((speed_centered.shift(1) * speed_centered) < 0).astype(float)
            X[f'zcr{w}'] = zero_crossings.rolling(ws, min_periods=1).sum()
        
        for w in [20, 40]:
            ws = _scale(w, fps)
            X[f'osc_x{w}'] = (center_x.diff().shift(1) * center_x.diff() < 0).astype(float).rolling(ws, min_periods=1).mean()
            X[f'osc_y{w}'] = (center_y.diff().shift(1) * center_y.diff() < 0).astype(float).rolling(ws, min_periods=1).mean()
            
    except Exception:
        pass
    
    return X

def add_posture_features(X, single_mouse, available_body_parts, fps):
    try:
        if all(p in available_body_parts for p in ['nose', 'tail_base', 'ear_left', 'ear_right']):
            nose_tail = np.sqrt((single_mouse['nose']['x'] - single_mouse['tail_base']['x'])**2 + 
                               (single_mouse['nose']['y'] - single_mouse['tail_base']['y'])**2)
            ear_dist = np.sqrt((single_mouse['ear_left']['x'] - single_mouse['ear_right']['x'])**2 + 
                              (single_mouse['ear_left']['y'] - single_mouse['ear_right']['y'])**2)
            X['compact'] = ear_dist / (nose_tail + 1e-6)
            
            w = _scale(30, fps)
            X['compact_var'] = X['compact'].rolling(w, min_periods=1).std()
        
        if all(p in available_body_parts for p in ['nose', 'body_center']):
            head_angle = np.arctan2(
                single_mouse['nose']['y'] - single_mouse['body_center']['y'],
                single_mouse['nose']['x'] - single_mouse['body_center']['x']
            )
            head_angle_change = np.abs(head_angle.diff())
            
            for w in [15, 30]:
                ws = _scale(w, fps)
                X[f'head_stb{w}'] = head_angle_change.rolling(ws, min_periods=1).std()
                
    except Exception:
        pass
    
    return X

def add_interaction_distance_bins(X, mouse_pair, avail_A, avail_B, fps):
    try:
        if 'body_center' not in avail_A or 'body_center' not in avail_B:
            return X
        
        dist = np.sqrt((mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x'])**2 +
                      (mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y'])**2)
        
        X['v_cls2'] = (dist < 3.0).astype(float)
        X['cls2'] = ((dist >= 3.0) & (dist < 8.0)).astype(float)
        X['med2'] = ((dist >= 8.0) & (dist < 20.0)).astype(float)
        
        for w in [30, 60, 120]:
            ws = _scale(w, fps)
            X[f't_vcls{w}'] = X['v_cls2'].rolling(ws, min_periods=1).mean()
            X[f't_cls{w}'] = X['cls2'].rolling(ws, min_periods=1).mean()
            
        dist_change = dist.diff() * float(fps)
        for w in [15, 30]:
            ws = _scale(w, fps)
            X[f'd_chg{w}'] = dist_change.rolling(ws, min_periods=1).mean()
            
    except Exception:
        pass
    
    return X

def add_relative_speed_features(X, mouse_pair, avail_A, avail_B, fps):
    try:
        if 'body_center' not in avail_A or 'body_center' not in avail_B:
            return X
        
        A_vx = mouse_pair['A']['body_center']['x'].diff() * float(fps)
        A_vy = mouse_pair['A']['body_center']['y'].diff() * float(fps)
        B_vx = mouse_pair['B']['body_center']['x'].diff() * float(fps)
        B_vy = mouse_pair['B']['body_center']['y'].diff() * float(fps)
        
        A_speed = np.sqrt(A_vx**2 + A_vy**2)
        B_speed = np.sqrt(B_vx**2 + B_vy**2)
        
        X['sp_ratio'] = A_speed / (B_speed + 1e-6)
        
        speed_diff = np.abs(A_speed - B_speed)
        for w in [30, 60]:
            ws = _scale(w, fps)
            X[f'sp_df{w}'] = speed_diff.rolling(ws, min_periods=1).mean()
        
        A_angle = np.arctan2(A_vy, A_vx)
        B_angle = np.arctan2(B_vy, B_vx)
        angle_diff = np.abs(np.unwrap(A_angle - B_angle))
        
        for w in [20, 40]:
            ws = _scale(w, fps)
            X[f'ang_df{w}'] = angle_diff.rolling(ws, min_periods=1).mean()
            X[f'same_dir{w}'] = (angle_diff < np.pi/4).astype(float).rolling(ws, min_periods=1).mean()
            X[f'opp_dir{w}'] = (angle_diff > 3*np.pi/4).astype(float).rolling(ws, min_periods=1).mean()
            
    except Exception:
        pass
    
    return X

def add_momentum_features(X, center_x, center_y, fps):
    try:
        vx = center_x.diff() * float(fps)
        vy = center_y.diff() * float(fps)
        speed = np.sqrt(vx**2 + vy**2)
        
        for w in [20, 40, 60]:
            ws = _scale(w, fps)
            speed_var = speed.rolling(ws, min_periods=max(1, ws//4)).var()
            X[f'momentum{w}'] = speed * speed_var
        
        direction = np.arctan2(vy, vx)
        direction_change = np.abs(direction.diff())
        
        for w in [30, 60, 90]:
            ws = _scale(w, fps)
            X[f'persist{w}'] = (direction_change < np.pi/6).astype(float).rolling(ws, min_periods=1).mean()
            X[f'erratic{w}'] = (direction_change > np.pi/3).astype(float).rolling(ws, min_periods=1).mean()
        
        for w in [30, 60]:
            ws = _scale(w, fps)
            speed_mean = speed.rolling(ws, min_periods=1).mean()
            speed_std = speed.rolling(ws, min_periods=1).std()
            X[f'sp_stab{w}'] = speed_std / (speed_mean + 1e-6)
            
    except Exception:
        pass
    
    return X

def add_rhythmic_features(X, center_x, center_y, fps):
    try:
        speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
        
        for lag in [5, 10, 15, 20]:
            lag_frames = _scale(lag, fps)
            speed_lagged = speed.shift(lag_frames)
            
            for w in [30, 60]:
                ws = _scale(w, fps)
                corr = speed.rolling(ws, min_periods=max(5, ws//4)).corr(speed_lagged)
                X[f'rhythm_l{lag}_w{w}'] = corr
        
        speed_ma = speed.rolling(_scale(5, fps), min_periods=1).mean()
        is_peak = ((speed_ma > speed_ma.shift(1)) & (speed_ma > speed_ma.shift(-1))).astype(float)
        
        for w in [30, 60, 90]:
            ws = _scale(w, fps)
            X[f'peaks{w}'] = is_peak.rolling(ws, min_periods=1).sum()
            
    except Exception:
        pass
    
    return X

def add_posture_dynamics(X, single_mouse, available_body_parts, fps):
    try:
        if all(p in available_body_parts for p in ['nose', 'tail_base']):
            body_length = np.sqrt((single_mouse['nose']['x'] - single_mouse['tail_base']['x'])**2 +
                                 (single_mouse['nose']['y'] - single_mouse['tail_base']['y'])**2)
            
            for w in [15, 30, 60]:
                ws = _scale(w, fps)
                X[f'len_m{w}'] = body_length.rolling(ws, min_periods=1).mean()
                X[f'len_s{w}'] = body_length.rolling(ws, min_periods=1).std()
                
            len_change = body_length.diff() * float(fps)
            for w in [15, 30]:
                ws = _scale(w, fps)
                X[f'len_chg{w}'] = len_change.rolling(ws, min_periods=1).mean()
        
        if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
            if 'body_ang' in X.columns:
                for w in [20, 40]:
                    ws = _scale(w, fps)
                    X[f'ang_m{w}'] = X['body_ang'].rolling(ws, min_periods=1).mean()
                    X[f'ang_s{w}'] = X['body_ang'].rolling(ws, min_periods=1).std()

        if all(p in available_body_parts for p in ['ear_left', 'ear_right', 'nose']):
            left_nose = np.sqrt((single_mouse['ear_left']['x'] - single_mouse['nose']['x'])**2 +
                               (single_mouse['ear_left']['y'] - single_mouse['nose']['y'])**2)
            right_nose = np.sqrt((single_mouse['ear_right']['x'] - single_mouse['nose']['x'])**2 +
                                (single_mouse['ear_right']['y'] - single_mouse['nose']['y'])**2)
            
            ear_asym = np.abs(left_nose - right_nose)
            for w in [15, 30]:
                ws = _scale(w, fps)
                X[f'ear_asy{w}'] = ear_asym.rolling(ws, min_periods=1).mean()
                
    except Exception:
        pass
    
    return X

def add_territorial_features(X, center_x, center_y, fps):
    try:
        x_median = center_x.rolling(_scale(300, fps), min_periods=30).median()
        y_median = center_y.rolling(_scale(300, fps), min_periods=30).median()
        
        for w in [60, 120, 240]:
            ws = _scale(w, fps)
            in_right = (center_x > x_median).astype(float)
            in_top = (center_y > y_median).astype(float)
            
            X[f'quad_x{w}'] = in_right.rolling(ws, min_periods=1).mean()
            X[f'quad_y{w}'] = in_top.rolling(ws, min_periods=1).mean()
            
            quad_change_x = (in_right != in_right.shift(1)).astype(float)
            quad_change_y = (in_top != in_top.shift(1)).astype(float)
            X[f'explr_x{w}'] = quad_change_x.rolling(ws, min_periods=1).sum()
            X[f'explr_y{w}'] = quad_change_y.rolling(ws, min_periods=1).sum()
        
        for w in [120, 240]:
            ws = _scale(w, fps)
            cx_min = center_x.rolling(ws, min_periods=1).min()
            cx_max = center_x.rolling(ws, min_periods=1).max()
            cy_min = center_y.rolling(ws, min_periods=1).min()
            cy_max = center_y.rolling(ws, min_periods=1).max()

            X[f'coverage{w}'] = (cx_max - cx_min) * (cy_max - cy_min)
            
    except Exception:
        pass
    
    return X

def add_interaction_patterns(X, mouse_pair, avail_A, avail_B, fps):
    try:
        if 'body_center' not in avail_A or 'body_center' not in avail_B:
            return X
        
        dist = np.sqrt((mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x'])**2 +
                      (mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y'])**2)

        dist_change = dist.diff()
        for w in [20, 40]:
            ws = _scale(w, fps)
            osc = ((dist_change.shift(1) * dist_change) < 0).astype(float)
            X[f'dist_osc{w}'] = osc.rolling(ws, min_periods=1).mean()
        
        A_vx = mouse_pair['A']['body_center']['x'].diff() * float(fps)
        A_vy = mouse_pair['A']['body_center']['y'].diff() * float(fps)
        B_vx = mouse_pair['B']['body_center']['x'].diff() * float(fps)
        B_vy = mouse_pair['B']['body_center']['y'].diff() * float(fps)
        
        A_speed = np.sqrt(A_vx**2 + A_vy**2)
        B_speed = np.sqrt(B_vx**2 + B_vy**2)
        
        to_B_x = mouse_pair['B']['body_center']['x'] - mouse_pair['A']['body_center']['x']
        to_B_y = mouse_pair['B']['body_center']['y'] - mouse_pair['A']['body_center']['y']
        to_B_dist = np.sqrt(to_B_x**2 + to_B_y**2) + 1e-6
        
        A_toward_B = (A_vx * to_B_x + A_vy * to_B_y) / (A_speed * to_B_dist + 1e-6)
        B_toward_A = (-B_vx * to_B_x - B_vy * to_B_y) / (B_speed * to_B_dist + 1e-6)
        
        for w in [20, 40, 60]:
            ws = _scale(w, fps)
            X[f'A_to_B{w}'] = A_toward_B.rolling(ws, min_periods=1).mean()
            X[f'B_to_A{w}'] = B_toward_A.rolling(ws, min_periods=1).mean()

            both_approach = ((A_toward_B > 0.5) & (B_toward_A > 0.5)).astype(float)
            X[f'mutual{w}'] = both_approach.rolling(ws, min_periods=1).mean()
            
            A_chase = ((A_toward_B > 0.5) & (B_toward_A < -0.5) & (A_speed > B_speed)).astype(float)
            X[f'A_chase{w}'] = A_chase.rolling(ws, min_periods=1).mean()
        
        if all(p in avail_A for p in ['nose', 'tail_base']) and all(p in avail_B for p in ['nose', 'tail_base']):
            A_orient = np.arctan2(
                mouse_pair['A']['nose']['y'] - mouse_pair['A']['tail_base']['y'],
                mouse_pair['A']['nose']['x'] - mouse_pair['A']['tail_base']['x']
            )
            B_orient = np.arctan2(
                mouse_pair['B']['nose']['y'] - mouse_pair['B']['tail_base']['y'],
                mouse_pair['B']['nose']['x'] - mouse_pair['B']['tail_base']['x']
            )
            
            orient_diff = np.abs(np.unwrap(A_orient - B_orient))
            for w in [30, 60]:
                ws = _scale(w, fps)
                X[f'parallel{w}'] = (orient_diff < np.pi/4).astype(float).rolling(ws, min_periods=1).mean()
                X[f'opposite{w}'] = (np.abs(orient_diff - np.pi) < np.pi/4).astype(float).rolling(ws, min_periods=1).mean()
        
        is_contact = (dist < 5.0).astype(float)
        contact_duration = pd.Series(0, index=is_contact.index, dtype=float)
        current_duration = 0
        for i in range(len(is_contact)):
            if is_contact.iloc[i] > 0:
                current_duration += 1
                contact_duration.iloc[i] = current_duration
            else:
                current_duration = 0
        X['contact_dur'] = contact_duration
        
    except Exception:
        pass
    
    return X

def add_body_part_velocities(X, single_mouse, available_body_parts, fps):
    try:
        parts_of_interest = ['nose', 'ear_left', 'ear_right', 'body_center', 'tail_base']
        
        for part in parts_of_interest:
            if part in available_body_parts:
                vx = single_mouse[part]['x'].diff() * float(fps)
                vy = single_mouse[part]['y'].diff() * float(fps)
                v = np.sqrt(vx**2 + vy**2)
                
                for w in [10, 20, 40]:
                    ws = _scale(w, fps)
                    X[f'{part}_v{w}'] = v.rolling(ws, min_periods=1).mean()
        
        if all(p in available_body_parts for p in ['nose', 'tail_base']):
            nose_v = np.sqrt(single_mouse['nose']['x'].diff()**2 + single_mouse['nose']['y'].diff()**2) * float(fps)
            tail_v = np.sqrt(single_mouse['tail_base']['x'].diff()**2 + single_mouse['tail_base']['y'].diff()**2) * float(fps)
            
            for w in [15, 30]:
                ws = _scale(w, fps)
                X[f'nose_tail_vr{w}'] = (nose_v - tail_v).rolling(ws, min_periods=1).mean()
                
    except Exception:
        pass
    
    return X

def add_statistical_moments(X, center_x, center_y, fps):
    try:
        speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
        
        for w in [60, 120]:
            ws = _scale(w, fps)
            if len(speed) >= ws:
                speed_mean = speed.rolling(ws, min_periods=max(5, ws//4)).mean()
                speed_std = speed.rolling(ws, min_periods=max(5, ws//4)).std()
                speed_centered = speed - speed_mean
                X[f'skew{w}'] = (speed_centered**3).rolling(ws, min_periods=max(5, ws//4)).mean() / (speed_std**3 + 1e-6)
                X[f'kurt{w}'] = (speed_centered**4).rolling(ws, min_periods=max(5, ws//4)).mean() / (speed_std**4 + 1e-6)
        
        for w in [60, 120]:
            ws = _scale(w, fps)
            speed_binned = pd.cut(speed, bins=5, labels=False)
            X[f'complex{w}'] = speed_binned.rolling(ws, min_periods=1).apply(lambda x: len(np.unique(x[~np.isnan(x)])), raw=False)
            
    except Exception:
        pass
    
    return X

def add_temporal_context_features(X, center_x, center_y, fps):
    try:
        speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
        
        for w in [15, 30, 60]:
            ws = _scale(w, fps)
            speed_past = speed.shift(ws).rolling(ws, min_periods=1).mean()
            speed_future = speed.shift(-ws).rolling(ws, min_periods=1).mean()
            X[f'sp_fut_past_{w}'] = speed_future / (speed_past + 1e-6)
            X[f'sp_trend_{w}'] = (speed_future - speed_past) / (speed_past + 1e-6)
        
        for w in [45, 90]:
            ws = _scale(w, fps)
            speed_ma = speed.rolling(ws, min_periods=1, center=True).mean()
            X[f'phase_accel_{w}'] = (speed > speed_ma).astype(float)
            X[f'phase_stable_{w}'] = (np.abs(speed - speed_ma) < 0.5 * speed_ma).astype(float)
            
    except Exception:
        pass
        
    return X

def add_body_part_triangulation(X, single_mouse, available_body_parts, fps):
    try:
        if all(p in available_body_parts for p in ['nose', 'ear_left', 'ear_right']):
            x1, y1 = single_mouse['nose']['x'], single_mouse['nose']['y']
            x2, y2 = single_mouse['ear_left']['x'], single_mouse['ear_left']['y']
            x3, y3 = single_mouse['ear_right']['x'], single_mouse['ear_right']['y']
            
            head_area = 0.5 * np.abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
            
            for w in [20, 40]:
                ws = _scale(w, fps)
                X[f'head_area_m{w}'] = head_area.rolling(ws, min_periods=1).mean()
                X[f'head_area_s{w}'] = head_area.rolling(ws, min_periods=1).std()
        
        if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
            front_half = np.sqrt((single_mouse['nose']['x'] - single_mouse['body_center']['x'])**2 +
                                (single_mouse['nose']['y'] - single_mouse['body_center']['y'])**2)
            back_half = np.sqrt((single_mouse['body_center']['x'] - single_mouse['tail_base']['x'])**2 +
                               (single_mouse['body_center']['y'] - single_mouse['tail_base']['y'])**2)
            
            X['body_balance'] = front_half / (back_half + 1e-6)
            
            for w in [20, 40]:
                ws = _scale(w, fps)
                X[f'balance_var_{w}'] = X['body_balance'].rolling(ws, min_periods=1).std()
                
    except Exception:
        pass
        
    return X

def add_spatial_dominance_features(X, mouse_pair, avail_A, avail_B, fps):
    try:
        if 'body_center' not in avail_A or 'body_center' not in avail_B:
            return X
        
        mid_x = (mouse_pair['A']['body_center']['x'] + mouse_pair['B']['body_center']['x']) / 2
        mid_y = (mouse_pair['A']['body_center']['y'] + mouse_pair['B']['body_center']['y']) / 2
        
        A_to_mid = np.sqrt((mouse_pair['A']['body_center']['x'] - mid_x)**2 +
                          (mouse_pair['A']['body_center']['y'] - mid_y)**2)
        B_to_mid = np.sqrt((mouse_pair['B']['body_center']['x'] - mid_x)**2 +
                          (mouse_pair['B']['body_center']['y'] - mid_y)**2)
        
        for w in [30, 60]:
            ws = _scale(w, fps)
            X[f'space_dom_{w}'] = (A_to_mid < B_to_mid).astype(float).rolling(ws, min_periods=1).mean()

        A_vx = mouse_pair['A']['body_center']['x'].diff()
        A_vy = mouse_pair['A']['body_center']['y'].diff()
        B_vx = mouse_pair['B']['body_center']['x'].diff()
        B_vy = mouse_pair['B']['body_center']['y'].diff()
        
        vel_cross = A_vx * B_vy - A_vy * B_vx
        vel_dot = A_vx * B_vx + A_vy * B_vy
        
        for w in [20, 40]:
            ws = _scale(w, fps)
            X[f'perpendicular_{w}'] = np.abs(vel_cross).rolling(ws, min_periods=1).mean()
            X[f'parallel_{w}'] = np.abs(vel_dot).rolling(ws, min_periods=1).mean()
            X[f'block_ratio_{w}'] = (np.abs(vel_cross) / (np.abs(vel_dot) + 1e-6)).rolling(ws, min_periods=1).mean()
            
    except Exception:
        pass
        
    return X

def add_repetitive_pattern_features(X, center_x, center_y, fps):
    try:
        speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
        
        for period in [10, 20, 30, 40]:
            period_frames = _scale(period, fps)
            speed_lagged = speed.shift(period_frames)
            
            for w in [60, 120]:
                ws = _scale(w, fps)
                corr = speed.rolling(ws, min_periods=max(5, ws//4)).corr(speed_lagged)
                X[f'cycle_{period}_{w}'] = corr
        
        for w in [60, 120, 180]:
            ws = _scale(w, fps)
            start_x = center_x.shift(ws)
            start_y = center_y.shift(ws)
            
            return_dist = np.sqrt((center_x - start_x)**2 + (center_y - start_y)**2)
            path_length = np.sqrt(center_x.diff()**2 + center_y.diff()**2).rolling(ws, min_periods=1).sum()
            
            X[f'circularity_{w}'] = return_dist / (path_length + 1e-6)
            
    except Exception:
        pass
        
    return X

def add_energy_features(X, center_x, center_y, fps):
    try:
        vx = center_x.diff() * float(fps)
        vy = center_y.diff() * float(fps)
        kinetic = vx**2 + vy**2
        
        for w in [30, 60, 90]:
            ws = _scale(w, fps)
            energy_change = kinetic.diff()
            X[f'energy_loss_{w}'] = (energy_change < 0).astype(float).rolling(ws, min_periods=1).mean()
            X[f'energy_gain_{w}'] = (energy_change > 0).astype(float).rolling(ws, min_periods=1).mean()

            X[f'energy_smooth_{w}'] = kinetic.rolling(ws, min_periods=1).std() / (kinetic.rolling(ws, min_periods=1).mean() + 1e-6)
        
    except Exception:
        pass
        
    return X

def add_attention_features(X, mouse_pair, avail_A, avail_B, fps):
    try:
        if not all(p in avail_A for p in ['nose', 'tail_base']) or \
           not all(p in avail_B for p in ['nose', 'tail_base']):
            return X
        
        A_dir_x = mouse_pair['A']['nose']['x'] - mouse_pair['A']['tail_base']['x']
        A_dir_y = mouse_pair['A']['nose']['y'] - mouse_pair['A']['tail_base']['y']
        
        to_B_x = mouse_pair['B']['body_center']['x'] - mouse_pair['A']['body_center']['x']
        to_B_y = mouse_pair['B']['body_center']['y'] - mouse_pair['A']['body_center']['y']
        
        facing_score = (A_dir_x * to_B_x + A_dir_y * to_B_y) / \
                      (np.sqrt(A_dir_x**2 + A_dir_y**2) * np.sqrt(to_B_x**2 + to_B_y**2) + 1e-6)
        
        is_facing = (facing_score > 0.7).astype(float)
        
        facing_duration = pd.Series(0, index=is_facing.index, dtype=float)
        current_dur = 0
        for i in range(len(is_facing)):
            if is_facing.iloc[i] > 0:
                current_dur += 1
                facing_duration.iloc[i] = current_dur
            else:
                current_dur = 0
        
        X['attention_dur'] = facing_duration
        
        for w in [30, 60]:
            ws = _scale(w, fps)
            X[f'attention_pct_{w}'] = is_facing.rolling(ws, min_periods=1).mean()
            X[f'attention_max_{w}'] = facing_duration.rolling(ws, min_periods=1).max()
        
    except Exception:
        pass
        
    return X

def add_behavioral_grammar_features(X, center_x, center_y, fps):
    try:
        speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
        speed_tokens = pd.cut(speed, bins=[-np.inf, 1, 3, 7, np.inf], labels=[0, 1, 2, 3])

        for window in [45, 90]:
            ws = _scale(window, fps)
            if len(speed_tokens) >= ws:
                transitions = (speed_tokens.diff() != 0).astype(float)
                X[f'transition_rate_{window}'] = transitions.rolling(ws, min_periods=ws//4).mean()
                
                def pattern_score(tokens):
                    if len(tokens) < 10:
                        return 0
                    unique_ratio = len(set(tokens)) / len(tokens)
                    return 1 - unique_ratio
                
                X[f'pattern_reg_{window}'] = speed_tokens.rolling(ws, min_periods=ws//3).apply(
                    lambda x: pattern_score(x.values), raw=False
                )
                
                if window == 45:
                    rest_to_fast = 0
                    fast_to_rest = 0
                    for i in range(len(speed_tokens)-1):
                        if speed_tokens.iloc[i] == 0 and speed_tokens.iloc[i+1] == 3:
                            rest_to_fast += 1
                        elif speed_tokens.iloc[i] == 3 and speed_tokens.iloc[i+1] == 0:
                            fast_to_rest += 1
                    X['sudden_start'] = rest_to_fast / len(speed_tokens)
                    X['sudden_stop'] = fast_to_rest / len(speed_tokens)
                
    except Exception:
        pass
        
    return X

def add_social_contagion_features(X, mouse_pair, avail_A, avail_B, fps):
    try:
        if 'body_center' not in avail_A or 'body_center' not in avail_B:
            return X
        
        A_vx = mouse_pair['A']['body_center']['x'].diff() * float(fps)
        A_vy = mouse_pair['A']['body_center']['y'].diff() * float(fps)
        B_vx = mouse_pair['B']['body_center']['x'].diff() * float(fps)
        B_vy = mouse_pair['B']['body_center']['y'].diff() * float(fps)
        A_speed = np.sqrt(A_vx**2 + A_vy**2)
        B_speed = np.sqrt(B_vx**2 + B_vy**2)
        
        for lag in [5, 10, 20]:
            lag_frames = _scale(lag, fps)
            ws = _scale(45, fps)
            
            if len(A_speed) >= ws + lag_frames:
                A_lead_corr = A_speed.rolling(ws, min_periods=ws//3).corr(
                    B_speed.shift(-lag_frames)
                )
                X[f'A_leads_B_{lag}'] = A_lead_corr
                
                B_lead_corr = B_speed.rolling(ws, min_periods=ws//3).corr(
                    A_speed.shift(-lag_frames)
                )
                X[f'B_leads_A_{lag}'] = B_lead_corr
                
                X[f'mutual_inf_{lag}'] = A_lead_corr * B_lead_corr
        
        A_active = (A_speed > A_speed.quantile(0.7)).astype(float)
        B_active = (B_speed > B_speed.quantile(0.7)).astype(float)
        ws = _scale(40, fps)
        
        A_then_B = pd.Series(0.0, index=A_active.index)
        for i in range(5, len(A_active)):
            if A_active.iloc[i-5:i].sum() > 0 and B_active.iloc[i] > 0:
                A_then_B.iloc[i] = 1.0
        X['contagion_A_to_B'] = A_then_B.rolling(ws, min_periods=ws//3).mean()
        
        B_then_A = pd.Series(0.0, index=B_active.index)
        for i in range(5, len(B_active)):
            if B_active.iloc[i-5:i].sum() > 0 and A_active.iloc[i] > 0:
                B_then_A.iloc[i] = 1.0
        X['contagion_B_to_A'] = B_then_A.rolling(ws, min_periods=ws//3).mean()
            
    except Exception:
        pass
        
    return X

def add_decision_point_features(X, center_x, center_y, fps):
    try:
        speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
        direction = np.arctan2(center_y.diff(), center_x.diff())
        
        speed_change = np.abs(speed.diff())
        direction_change = np.abs(direction.diff())
        
        speed_change_norm = (speed_change - speed_change.rolling(60, min_periods=1).mean()) / (speed_change.rolling(60, min_periods=1).std() + 1e-6)
        direction_change_norm = (direction_change - direction_change.rolling(60, min_periods=1).mean()) / (direction_change.rolling(60, min_periods=1).std() + 1e-6)
        
        decision_score = speed_change_norm * direction_change_norm
        is_decision = (decision_score > 1.5).astype(float)
        
        for window in [15, 30, 60, 120]:
            ws = _scale(window, fps)
            X[f'decision_freq_{window}'] = is_decision.rolling(ws, min_periods=1).sum()
            time_since_decision = pd.Series(0, index=is_decision.index)
            counter = 0
            
            for i in range(len(is_decision)):
                if is_decision.iloc[i] > 0:
                    counter = 0
                else:
                    counter += 1
                time_since_decision.iloc[i] = counter
                
            X[f'time_since_decision_{window}'] = time_since_decision
            X[f'predictability_{window}'] = 1.0 / (X[f'decision_freq_{window}'] + 1)
        
        for window in [20, 40]:
            ws = _scale(window, fps)
            decision_density = is_decision.rolling(ws, min_periods=1).sum()
            X[f'decision_burst_{window}'] = (decision_density > ws * 0.3).astype(float)
            
    except Exception:
        pass
        
    return X

def add_attention_field_features(X, mouse_pair, avail_A, avail_B, fps):
    try:
        if not all(p in avail_A for p in ['nose', 'ear_left', 'ear_right', 'body_center']):
            return X
        if not all(p in avail_B for p in ['nose', 'body_center']):
            return X

        A_ear_center_x = (mouse_pair['A']['ear_left']['x'] + mouse_pair['A']['ear_right']['x']) / 2
        A_ear_center_y = (mouse_pair['A']['ear_left']['y'] + mouse_pair['A']['ear_right']['y']) / 2
        
        gaze_dir_x = mouse_pair['A']['nose']['x'] - A_ear_center_x
        gaze_dir_y = mouse_pair['A']['nose']['y'] - A_ear_center_y
        gaze_length = np.sqrt(gaze_dir_x**2 + gaze_dir_y**2)
        
        attention_field_length = 15.0
        attention_x = mouse_pair['A']['nose']['x'] + (gaze_dir_x / gaze_length) * attention_field_length
        attention_y = mouse_pair['A']['nose']['y'] + (gaze_dir_y / gaze_length) * attention_field_length

        dist_to_attention = np.sqrt(
            (mouse_pair['B']['body_center']['x'] - attention_x)**2 +
            (mouse_pair['B']['body_center']['y'] - attention_y)**2
        )
        
        attention_radius = 10.0
        B_in_attention = (dist_to_attention < attention_radius).astype(float)
        
        for window in [15, 30, 60]:
            ws = _scale(window, fps)
            X[f'B_in_attention_field_{window}'] = B_in_attention.rolling(ws, min_periods=1).mean()
            
            attention_runs = []
            current_run = 0
            for val in B_in_attention:
                if val > 0:
                    current_run += 1
                else:
                    attention_runs.append(current_run)
                    current_run = 0
            
            X[f'max_attention_duration_{window}'] = pd.Series(
                B_in_attention.rolling(ws, min_periods=1).apply(
                    lambda x: np.max(np.diff(np.concatenate(([0], np.where(x == 0)[0], [len(x)]))))
                    if np.any(x > 0) else 0,
                    raw=True
                ),
                index=X.index
            )
        
        if all(p in avail_B for p in ['nose', 'ear_left', 'ear_right']):
            B_ear_center_x = (mouse_pair['B']['ear_left']['x'] + mouse_pair['B']['ear_right']['x']) / 2
            B_ear_center_y = (mouse_pair['B']['ear_left']['y'] + mouse_pair['B']['ear_right']['y']) / 2
            
            B_gaze_dir_x = mouse_pair['B']['nose']['x'] - B_ear_center_x
            B_gaze_dir_y = mouse_pair['B']['nose']['y'] - B_ear_center_y
            
            A_to_B_x = mouse_pair['B']['body_center']['x'] - mouse_pair['A']['nose']['x']
            A_to_B_y = mouse_pair['B']['body_center']['y'] - mouse_pair['A']['nose']['y']
            
            A_gaze_align = (gaze_dir_x * A_to_B_x + gaze_dir_y * A_to_B_y) / (gaze_length * np.sqrt(A_to_B_x**2 + A_to_B_y**2) + 1e-6)
            
            B_to_A_x = -A_to_B_x
            B_to_A_y = -A_to_B_y
            B_gaze_length = np.sqrt(B_gaze_dir_x**2 + B_gaze_dir_y**2)
            B_gaze_align = (B_gaze_dir_x * B_to_A_x + B_gaze_dir_y * B_to_A_y) / (B_gaze_length * np.sqrt(B_to_A_x**2 + B_to_A_y**2) + 1e-6)
            
            mutual_gaze = ((A_gaze_align > 0.7) & (B_gaze_align > 0.7)).astype(float)
            
            for window in [20, 40]:
                ws = _scale(window, fps)
                X[f'mutual_gaze_{window}'] = mutual_gaze.rolling(ws, min_periods=1).mean()
                
    except Exception:
        pass
        
    return X

def add_bout_structure_features(X, center_x, center_y, fps):
    try:
        speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
        activity_threshold = speed.quantile(0.6)
        is_active = (speed > activity_threshold).astype(float)

        bout_starts = []
        bout_ends = []
        in_bout = False
        
        for i in range(len(is_active)):
            if is_active.iloc[i] > 0 and not in_bout:
                bout_starts.append(i)
                in_bout = True
            elif is_active.iloc[i] == 0 and in_bout:
                bout_ends.append(i)
                in_bout = False
        
        if in_bout:
            bout_ends.append(len(is_active))
        
        bout_durations = pd.Series(0, index=speed.index)
        for start, end in zip(bout_starts, bout_ends):
            duration = end - start
            for k in range(start, min(end, len(bout_durations))):
                bout_durations.iloc[k] = duration
        
        for window in [90, 180]:
            ws = _scale(window, fps)
            X[f'bout_duration_{window}'] = bout_durations.rolling(ws, min_periods=ws//4).mean()
            
            bout_transitions = (is_active.diff().abs() > 0).astype(float)
            X[f'bout_freq_{window}'] = bout_transitions.rolling(ws, min_periods=ws//4).sum()
            X[f'bout_density_{window}'] = is_active.rolling(ws, min_periods=ws//4).mean()
            
    except Exception:
        pass
        
    return X

def add_graph_topology_features(X, mouse_pair, avail_A, avail_B, fps):
    try:
        if 'body_center' not in avail_A or 'body_center' not in avail_B:
            return X
        
        dist = np.sqrt(
            (mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x'])**2 +
            (mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y'])**2
        )
        
        edge_strength = 1.0 / (dist + 1)

        for window in [40, 90]:
            ws = _scale(window, fps)
            connection_threshold = 15.0
            is_connected = (dist < connection_threshold).astype(float)
            
            X[f'graph_density_{window}'] = is_connected.rolling(ws, min_periods=ws//4).mean()
            X[f'edge_weight_{window}'] = edge_strength.rolling(ws, min_periods=ws//4).mean()
            
            edge_changes = (is_connected.diff() != 0).astype(float)
            X[f'edge_stability_{window}'] = 1.0 - edge_changes.rolling(ws, min_periods=ws//4).mean()
        
    except Exception:
        pass
        
    return X

def add_speed_asymmetry_future_past_single(
    X: pd.DataFrame, cx: pd.Series, cy: pd.Series, fps: float,
    horizon_base: int = 30, agg: str = "mean"
) -> pd.DataFrame:
    w = max(3, _scale(horizon_base, fps))
    v = np.sqrt(cx.diff()**2 + cy.diff()**2) * float(fps)
    
    if agg == "median":
        v_past = v.rolling(w, min_periods=max(3, w//4), center=False).median()
        v_fut = v.iloc[::-1].rolling(w, min_periods=max(3, w//4)).median().iloc[::-1]
    else:
        v_past = v.rolling(w, min_periods=max(3, w//4), center=False).mean()
        v_fut = v.iloc[::-1].rolling(w, min_periods=max(3, w//4)).mean().iloc[::-1]
    
    X["spd_asym_1s"] = (v_fut - v_past).fillna(0.0)
    
    return X

def add_gauss_shift_speed_future_past_single(
    X: pd.DataFrame, cx: pd.Series, cy: pd.Series, fps: float,
    window_base: int = 30, eps: float = 1e-6
) -> pd.DataFrame:
    w = max(5, _scale(window_base, fps))
    v = np.sqrt(cx.diff()**2 + cy.diff()**2) * float(fps)

    mu_p = v.rolling(w, min_periods=max(3, w//4)).mean()
    va_p = v.rolling(w, min_periods=max(3, w//4)).var().clip(lower=eps)

    mu_f = v.iloc[::-1].rolling(w, min_periods=max(3, w//4)).mean().iloc[::-1]
    va_f = v.iloc[::-1].rolling(w, min_periods=max(3, w//4)).var().iloc[::-1].clip(lower=eps)

    kl_pf = 0.5 * ((va_p/va_f) + ((mu_f - mu_p)**2)/va_f - 1.0 + np.log(va_f/va_p))
    kl_fp = 0.5 * ((va_f/va_p) + ((mu_p - mu_f)**2)/va_p - 1.0 + np.log(va_p/va_f))
    X["spd_symkl_1s"] = (kl_pf + kl_fp).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    return X

def add_cumulative_distance_single(X, cx, cy, fps, horizon_frames_base: int = 180, colname: str = "path_cum180"):
    L = max(1, _scale(horizon_frames_base, fps))
    step = np.hypot(cx.diff(), cy.diff())
    path = step.rolling(2*L + 1, min_periods=max(5, L//6), center=True).sum()
    X[colname] = path.fillna(0.0).astype(np.float32)
    
    return X

def smooth_coordinates(df, fps, sigma_sec=0.05):
    sigma = sigma_sec * fps
    df_smooth = df.copy()
    for col in df.columns:
        if col in ['x', 'y'] or (isinstance(col, tuple) and col[1] in ['x', 'y']):
            df_smooth[col] = gaussian_filter1d(df[col].fillna(method='ffill').fillna(0), sigma=sigma)
    return df_smooth

#============= TRANSFORM SINGLE =============#
def transform_single(single_mouse, body_parts_tracked, fps):
    single_mouse = smooth_coordinates(single_mouse, fps)
    available_body_parts = single_mouse.columns.get_level_values(0)

    X = pd.DataFrame({
        f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.combinations(body_parts_tracked, 2)
        if p1 in available_body_parts and p2 in available_body_parts
    })
    X = X.reindex(columns=[f"{p1}+{p2}" for p1, p2 in itertools.combinations(body_parts_tracked, 2)], copy=False)

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

    if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
        v1 = single_mouse['nose'] - single_mouse['body_center']
        v2 = single_mouse['tail_base'] - single_mouse['body_center']
        X['body_ang'] = (v1['x'] * v2['x'] + v1['y'] * v2['y']) / (
            np.sqrt(v1['x']**2 + v1['y']**2) * np.sqrt(v2['x']**2 + v2['y']**2) + 1e-6)

        angle = np.arctan2(v1['y'], v1['x'])
        body_ang = np.arctan2(v2['y'], v2['x'])
        X['body_ang_diff'] = np.unwrap(angle - body_ang)

    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']
        cy = single_mouse['body_center']['y']

        for w in [10, 30, 90]:
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

        X = add_curvature_features(X, cx, cy, fps)
        X = add_multiscale_features(X, cx, cy, fps)
        X = add_state_features(X, cx, cy, fps)
        X = add_longrange_features(X, cx, cy, fps)

        lag_180 = _scale(180, fps)
        if len(cx) >= lag_180:
            long_disp = np.sqrt((cx - cx.shift(lag_180))**2 + (cy - cy.shift(lag_180))**2)
            X['longdist_bin1'] = (long_disp > 20.0).astype(float)
            
            speed_180 = np.sqrt(cx.diff()**2 + cy.diff()**2) * float(fps)
            X['longdist_bin2'] = (speed_180.rolling(lag_180, min_periods=max(5, lag_180 // 6)).mean() > 5.0).astype(float)

        X = add_acceleration_features(X, cx, cy, fps)
        X = add_spatial_features(X, cx, cy, fps)
        X = add_frequency_features(X, cx, cy, fps)
        X = add_momentum_features(X, cx, cy, fps)
        X = add_rhythmic_features(X, cx, cy, fps)
        X = add_territorial_features(X, cx, cy, fps)
        X = add_statistical_moments(X, cx, cy, fps)
        X = add_temporal_context_features(X, cx, cy, fps)
        X = add_body_part_triangulation(X, single_mouse, available_body_parts, fps)
        X = add_repetitive_pattern_features(X, cx, cy, fps)
        X = add_energy_features(X, cx, cy, fps)
        X = add_behavioral_grammar_features(X, cx, cy, fps)
        X = add_decision_point_features(X, cx, cy, fps)
        X = add_bout_structure_features(X, cx, cy, fps)
        X = add_speed_asymmetry_future_past_single(X, cx, cy, fps, horizon_base=30)
        X = add_gauss_shift_speed_future_past_single(X, cx, cy, fps, window_base=30)
        X = add_cumulative_distance_single(X, cx, cy, fps, horizon_frames_base=180)

    X = add_posture_features(X, single_mouse, available_body_parts, fps)
    X = add_posture_dynamics(X, single_mouse, available_body_parts, fps)
    X = add_body_part_velocities(X, single_mouse, available_body_parts, fps)

    if all(p in available_body_parts for p in ['nose', 'tail_base']):
        nt_dist = np.sqrt((single_mouse['nose']['x'] - single_mouse['tail_base']['x'])**2 +
                          (single_mouse['nose']['y'] - single_mouse['tail_base']['y'])**2)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            X[f'nt_lg{lag}'] = nt_dist.shift(l)
            X[f'nt_df{lag}'] = nt_dist - nt_dist.shift(l)

    if all(p in available_body_parts for p in ['ear_left', 'ear_right']):
        ear_d = np.sqrt((single_mouse['ear_left']['x'] - single_mouse['ear_right']['x'])**2 +
                        (single_mouse['ear_left']['y'] - single_mouse['ear_right']['y'])**2)
        for off in [-30, -20, -10, 10, 20, 30]:
            o = _scale_signed(off, fps)
            X[f'ear_o{off}'] = ear_d.shift(-o)
        w = _scale(30, fps)
        X['ear_con'] = ear_d.rolling(w, min_periods=1, center=True).std() / \
                       (ear_d.rolling(w, min_periods=1, center=True).mean() + 1e-6)

    return X.astype(np.float32, copy=False)

#============= TRANSFORM PAIR =============#
def transform_pair(mouse_pair, body_parts_tracked, fps):
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)

    X = pd.DataFrame({
        f"12+{p1}+{p2}": np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.product(body_parts_tracked, repeat=2)
        if p1 in avail_A and p2 in avail_B
    })
    X = X.reindex(columns=[f"12+{p1}+{p2}" for p1, p2 in itertools.product(body_parts_tracked, repeat=2)], copy=False)

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

    if all(p in avail_A for p in ['nose', 'tail_base']) and all(p in avail_B for p in ['nose', 'tail_base']):
        dir_A = mouse_pair['A']['nose'] - mouse_pair['A']['tail_base']
        dir_B = mouse_pair['B']['nose'] - mouse_pair['B']['tail_base']
        X['rel_ori'] = (dir_A['x'] * dir_B['x'] + dir_A['y'] * dir_B['y']) / (
            np.sqrt(dir_A['x']**2 + dir_A['y']**2) * np.sqrt(dir_B['x']**2 + dir_B['y']**2) + 1e-6)

    if all(p in avail_A for p in ['nose']) and all(p in avail_B for p in ['nose']):
        cur = np.square(mouse_pair['A']['nose'] - mouse_pair['B']['nose']).sum(axis=1, skipna=False)
        lag = _scale(10, fps)
        shA_n = mouse_pair['A']['nose'].shift(lag)
        shB_n = mouse_pair['B']['nose'].shift(lag)
        past = np.square(shA_n - shB_n).sum(axis=1, skipna=False)
        X['appr'] = cur - past

    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd = np.sqrt((mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x'])**2 +
                     (mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y'])**2)
        X['v_cls'] = (cd < 5.0).astype(float)
        X['cls']   = ((cd >= 5.0) & (cd < 15.0)).astype(float)
        X['med']   = ((cd >= 15.0) & (cd < 30.0)).astype(float)
        X['far']   = (cd >= 30.0).astype(float)

    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd_full = np.square(mouse_pair['A']['body_center'] - mouse_pair['B']['body_center']).sum(axis=1, skipna=False)

        for w in [10, 30, 90]:
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

        lag_180 = _scale(180, fps)
        if len(cd_full) >= lag_180:
            cd_dist = np.sqrt(cd_full)
            X['longdist_pair_bin1'] = (cd_dist.rolling(lag_180, min_periods=max(5, lag_180 // 6)).mean() > 30.0).astype(float)
            X['longdist_pair_bin2'] = (cd_dist.rolling(lag_180, min_periods=max(5, lag_180 // 6)).mean() < 10.0).astype(float)
    
    if 'nose' in avail_A and 'nose' in avail_B:
        nn = np.sqrt((mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x'])**2 +
                     (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y'])**2)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            X[f'nn_lg{lag}']  = nn.shift(l)
            X[f'nn_ch{lag}']  = nn - nn.shift(l)
            is_cl = (nn < 10.0).astype(float)
            X[f'cl_ps{lag}']  = is_cl.rolling(l, min_periods=1).mean()

    if 'body_center' in avail_A and 'body_center' in avail_B:
        Avx = mouse_pair['A']['body_center']['x'].diff()
        Avy = mouse_pair['A']['body_center']['y'].diff()
        Bvx = mouse_pair['B']['body_center']['x'].diff()
        Bvy = mouse_pair['B']['body_center']['y'].diff()
        val = (Avx * Bvx + Avy * Bvy) / (np.sqrt(Avx**2 + Avy**2) * np.sqrt(Bvx**2 + Bvy**2) + 1e-6)

        for off in [-30, -20, -10, 0, 10, 20, 30]:
            o = _scale_signed(off, fps)
            X[f'va_{off}'] = val.shift(-o)

        w = _scale(30, fps)
        X['int_con'] = cd_full.rolling(w, min_periods=1, center=True).std() / \
                       (cd_full.rolling(w, min_periods=1, center=True).mean() + 1e-6)

        X = add_interaction_features(X, mouse_pair, avail_A, avail_B, fps)
        X = add_relative_heading_features(X, mouse_pair, fps)
        X = add_interaction_distance_bins(X, mouse_pair, avail_A, avail_B, fps)
        X = add_relative_speed_features(X, mouse_pair, avail_A, avail_B, fps)   
        X = add_interaction_patterns(X, mouse_pair, avail_A, avail_B, fps)
        X = add_spatial_dominance_features(X, mouse_pair, avail_A, avail_B, fps)
        X = add_attention_features(X, mouse_pair, avail_A, avail_B, fps)
        X = add_social_contagion_features(X, mouse_pair, avail_A, avail_B, fps)
        X = add_attention_field_features(X, mouse_pair, avail_A, avail_B, fps)
        X = add_graph_topology_features(X, mouse_pair, avail_A, avail_B, fps)

    return X.astype(np.float32, copy=False)

#============= CLASSIFIER =============#
class StratifiedSubsetClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, n_samples=None):
        self.estimator = estimator
        self.n_samples = n_samples

    def _to_numpy(self, X):
        try:
            return X.to_numpy(np.float32, copy=False)
        except AttributeError:
            return np.asarray(X, dtype=np.float32)

    def fit(self, X, y):
        Xn = self._to_numpy(X)
        y = np.asarray(y).ravel()

        uniq = np.unique(y[~pd.isna(y)])
        if set(uniq.tolist()) == {0, 2}:
            y = (y > 0).astype(np.int8)

        if self.n_samples is None or len(Xn) <= int(self.n_samples):
            self.estimator.fit(Xn, y)
        else:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=int(self.n_samples), random_state=42)
            try:
                idx, _ = next(sss.split(np.zeros_like(y), y))
                self.estimator.fit(Xn[idx], y[idx])
            except Exception as e:
                if 'best_split_info.left_count' in str(e):
                    self.estimator.set_params(device_type='cpu')
                    self.estimator.fit(Xn[idx], y[idx])
                else:
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
#============= PREPROCESS TO SUBMIT =============#
ADAPTIVE_THRESHOLDS_v1 = {
    'single': {
        '1': {
            'rear': 0.21525800228118896
        },
        '2': {
            'huddle': 0.18310001492500305,
            'rear': 0.5,
            'selfgroom': 0.5
        },
        '3': {
            'rear': 0.1394311934709549
        },
        '6': {
            'biteobject': 0.0005995196762709933,
            'climb': 0.1665017008781433,
            'dig': 0.20503206551074982,
            'exploreobject': 0.002617048792027599,
            'rear': 0.06062667946700533,
            'selfgroom': 0.16229913172159882
        },
        '7': {
            'rear': 0.296084463596344,
            'rest': 0.19009769796937753,
            'selfgroom': 0.1503967693952951,
            'climb': 0.04347278145137728,
            'dig': 0.19580557942390442,
            'run': 0.08074358850717545
        },
        '8': {
            'rear': 0.12643475830554962,
            'selfgroom': 0.09562112390995026,
            'genitalgroom': 0.1659549388765812,
            'dig': 0.08209896832704544
        },
        '9': {
            'freeze': 0.19502536751134242,
            'rear': 0.10221560573017868
        }
    },
    'pair': {
        '1': {
            'approach': 0.11553158611059189,
            'attack': 0.14749008417129517,
            'avoid': 0.12643153965473175,
            'chase': 0.06645490974187851,
            'chaseattack': 0.18669064342975616,
            'submit': 0.01984693482518196
        },
        '2': {
            'reciprocalsniff': 0.21752649545669556,
            'sniff': 0.5,
            'sniffgenital': 0.27659541368484497,
            'intromit': 0.5,
            'mount': 0.5
        },
        '3': {
            'approach': 0.02425196021795273,
            'attack': 0.06918201595544815,
            'avoid': 0.05090678855776787,
            'chase': 0.010798096656799316,
            'chaseattack': 0.022090625017881393,
            'submit': 0.016084443777799606
        },
        '4': {
            'attack': 0.10026851296424866,
            'dominance': 0.11296072602272034,
            'sniff': 0.07157517224550247,
            'chase': 0.01268178690224886,
            'escape': 0.041602469980716705,
            'follow': 0.24068273603916168
        },
        '5': {
            'attack': 0.3286178410053253,
            'sniff': 0.3949454724788666,
            'defend': 0.26223069429397583,
            'escape': 0.22683562338352203,
            'mount': 0.13026556372642517,
            'sniffgenital': 0.5
        },
        '6': {
            'shepherd': 0.16575618088245392,
            'approach': 0.19152656197547913,
            'attack': 0.23048235476016998,
            'chase': 0.09831040352582932,
            'defend': 0.15771698951721191,
            'escape': 0.304587721824646,
            'flinch': 0.008557775989174843,
            'follow': 0.06308241188526154,
            'sniff': 0.21881681300167954,
            'sniffface': 0.21378189325332642,
            'sniffgenital': 0.06065356358885765,
            'tussle': 0.006215625908225775
        },
        '7': {
            'intromit': 0.5,
            'mount': 0.5,
            'sniff': 0.33832971822497443,
            'sniffgenital': 0.25398045778274536,
            'approach': 0.2360716611146927,
            'defend': 0.033372338861227036,
            'escape': 0.11022237351367613,
            'attemptmount': 0.030093489214777946
        },
        '8': {
            'approach': 0.25436079502105713,
            'attack': 0.304213285446167,
            'disengage': 0.23995341360569,
            'mount': 0.2929087281227112,
            'sniff': 0.3190191388130188,
            'sniffgenital': 0.27274999022483826,
            'dominancemount': 0.236189067363739,
            'sniffbody': 0.2834545855777041,
            'sniffface': 0.31102895736694336,
            'attemptmount': 0.060965705662965775,
            'intromit': 0.4145866930484772,
            'chase': 0.06262975931167603,
            'escape': 0.25139206647872925,
            'reciprocalsniff': 0.35057151317596436,
            'allogroom': 0.019169863313436508,
            'ejaculate': 0.04548828676342964,
            'dominancegroom': 0.041647788137197495
        },
        '9': {
            'approach': 0.06814590096473694,
            'attack': 0.45281786720697165,
            'defend': 0.2370478814378824,
            'escape': 0.43424616074872296,
            'sniff': 0.10713639110326767
        }
    }
}

ADAPTIVE_THRESHOLDS_v2 = {
    'single': {
        '1': {'rear': 0.21007365021308194},
        '2': {'huddle': 0.19735296523909912},
        '3': {'rear': 0.15701597511237975},
        '6': {
            'biteobject': 0.019280318405461752,
            'climb': 0.17041432069803572,
            'dig': 0.18322978307096568,
            'exploreobject': 0.03616068681012833,
            'rear': 0.14167958812060324,
            'selfgroom': 0.16369590503761555
        },
        '7': {
            'rear': 0.2833280211562102,
            'rest': 0.2335984627108944,
            'selfgroom': 0.16591909358979792,
            'climb': 0.12406076880998487,
            'dig': 0.18660886803797655,
            'run': 0.045829534706167005
        },
        '8': {
            'rear': 0.12779833881797478,
            'selfgroom': 0.10192123986447743,
            'genitalgroom': 0.2622886999648151,
            'dig': 0.08035279006049402
        },
        '9': {
            'freeze': 0.1534336287860117,
            'rear': 0.11200474950637435
        }
    },
    'pair': {
        '1': {
            'approach': 0.12387827146667493,
            'attack': 0.061858616667710684,
            'avoid': 0.19742925700695999,
            'chase': 0.10244937475279906,
            'chaseattack': 0.12397792072234151,
            'submit': 0.026304502835523096
        },
        '2': {
            'reciprocalsniff': 0.2507984189461601,
            'sniffgenital': 0.27431326505807785
        },
        '3': {
            'approach': 0.06031101706132876,
            'attack': 0.052739387804301124,
            'avoid': 0.05702548759221109,
            'chase': 0.020763902784874902,
            'chaseattack': 0.01764858005019092,
            'submit': 0.03319505167948223
        },
        '4': {
            'attack': 0.11977287991962667,
            'dominance': 0.12347981291961246,
            'sniff': 0.15427837209513517,
            'chase': 0.020285560347685753,
            'escape': 0.04395574874861563,
            'follow': 0.3634684520362762
        },
        '5': {
            'attack': 0.3795698675025961,
            'sniff': 0.4030891118910214,
            'defend': 0.3212448068921057,
            'escape': 0.32324332344141365,
            'mount': 0.2662508768385557
        },
        '6': {
            'shepherd': 0.2094737727212095,
            'approach': 0.2117071055838438,
            'attack': 0.25760099456407204,
            'chase': 0.1840792820129987,
            'defend': 0.20886460457788664,
            'escape': 0.2994354072207405,
            'flinch': 0.040431458312782054,
            'follow': 0.09344493854577517,
            'sniff': 0.2732194822024317,
            'sniffface': 0.24886296332576852,
            'sniffgenital': 0.05561604743836648,
            'tussle': 0.011138234224494668
        },
        '7': {
            'sniff': 0.3415880870848557,
            'sniffgenital': 0.2862918040171262,
            'approach': 0.21883488170298246,
            'defend': 0.06010772077562987,
            'escape': 0.1263435325085349,
            'attemptmount': 0.07515830439133601
        },
        '8': {
            'approach': 0.23995662261566278,
            'attack': 0.33376404322207925,
            'disengage': 0.24522130141859103,
            'mount': 0.3211466468884248,
            'sniff': 0.3360021167613508,
            'sniffgenital': 0.2615004578401469,
            'dominancemount': 0.21957091537858664,
            'sniffbody': 0.28360908033353077,
            'sniffface': 0.30462080424710697,
            'attemptmount': 0.07619999899436361,
            'intromit': 0.4302671785513684,
            'chase': 0.04856646280375858,
            'escape': 0.18002569235275614,
            'reciprocalsniff': 0.3961874780776454,
            'allogroom': 0.10826494978890203,
            'ejaculate': 0.6021891238089415,
            'dominancegroom': 0.077824002762503
        },
        '9': {
            'approach': 0.08412952885539599,
            'attack': 0.4264444115237341,
            'defend': 0.2390009497061875,
            'escape': 0.43846319371420767,
            'sniff': 0.18137406640085454
        }
    }
}

def _select_threshold_map(thresholds, mode: str, section_id: str = None):
    base_default = float(thresholds.get('default', 0.27))
    mode_default = float(thresholds.get(f"{mode}_default", base_default))
    
    if section_id and mode in thresholds:
        mode_dict = thresholds[mode]
        if isinstance(mode_dict, dict) and section_id in mode_dict:
            section_overrides = mode_dict[section_id]
            out = defaultdict(lambda: mode_default)
            out.update({str(k): float(v) for k, v in section_overrides.items()})
            return out
    
    return defaultdict(lambda: mode_default)

def predict_multiclass_adaptive(pred, meta, thresholds_v1, thresholds_v2, section_id=None, weight_v1=0.5, weight_v2=0.5):
    fps = 30.0
    if 'frames_per_second' in meta.columns:
        fps_val = meta['frames_per_second'].iloc[0]
        if pd.notnull(fps_val):
            fps = float(fps_val)
    
    mode = 'pair'
    try:
        if 'target_id' in meta.columns and meta['target_id'].eq('self').all():
            mode = 'single'
    except:
        pass

    pred_smoothed = pred.copy()
    base_window = max(5, int(round(9 * (fps / 30.0))))
    sigma = fps / 18.0
    
    for col in pred_smoothed.columns:
        vals = pred_smoothed[col].fillna(0).values
        if len(vals) > 5:
            vals = medfilt(vals, kernel_size=5)
        rolled = pd.Series(vals).rolling(
            window=base_window, min_periods=1, center=True
        ).mean().values
        gauss = gaussian_filter1d(rolled, sigma=sigma, mode='nearest')
        pred_smoothed[col] = 0.75 * gauss + 0.25 * vals
    
    th_map_v1 = _select_threshold_map(thresholds_v1, mode, section_id)
    th_map_v2 = _select_threshold_map(thresholds_v2, mode, section_id)
    
    ama = np.argmax(pred_smoothed, axis=1)
    max_probs = pred_smoothed.max(axis=1)
    
    threshold_mask = np.zeros(len(pred_smoothed), dtype=bool)
    
    for i, action in enumerate(pred_smoothed.columns):
        action_mask = (ama == i)
        
        base_th_v1 = th_map_v1[action]
        base_th_v2 = th_map_v2[action]
        
        has_v1 = False
        has_v2 = False
        
        if mode in thresholds_v1 and isinstance(thresholds_v1[mode], dict):
            if section_id in thresholds_v1[mode]:
                has_v1 = action in thresholds_v1[mode][section_id]
        
        if mode in thresholds_v2 and isinstance(thresholds_v2[mode], dict):
            if section_id in thresholds_v2[mode]:
                has_v2 = action in thresholds_v2[mode][section_id]
        
        if has_v1 and has_v2:
            base_th = weight_v1 * base_th_v1 + weight_v2 * base_th_v2
        elif has_v1 and not has_v2:
            base_th = base_th_v1
        elif has_v2 and not has_v1:
            base_th = base_th_v2
        
        action_lower = action.lower()
        
        if action_lower in ['attack', 'mount', 'chase', 'chaseattack', 'jump']:
            adjusted_th = base_th * 0.88
        elif action_lower in ['rear', 'climb', 'dig', 'selfgroom']:
            adjusted_th = base_th * 0.90
        elif action_lower in ['sniff', 'sniffgenital', 'sniffface', 'approach']:
            adjusted_th = base_th * 0.93
        elif action_lower in ['rest', 'huddle', 'freeze']:
            adjusted_th = base_th * 1.05
        else:
            adjusted_th = base_th * 0.95
        
        threshold_mask |= (action_mask & (max_probs >= adjusted_th))
    
    ama = np.where(threshold_mask, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame)
    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]
    mask = ama_changes.values >= 0
    mask[-1] = False
    
    if np.sum(mask) == 0:
        return pd.DataFrame(columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
    
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
            if (stop_video_id[i] != video_id or 
                stop_agent_id[i] != agent_id or 
                stop_target_id[i] != target_id):
                new_stop = meta.query('video_id == @video_id').video_frame.max() + 1
                submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop
        else:
            new_stop = meta.query('video_id == @video_id').video_frame.max() + 1
            submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop
    
    duration = submission_part.stop_frame - submission_part.start_frame
    min_duration = max(2, int(fps / 12))
    submission_part = submission_part[duration >= min_duration].reset_index(drop=True)
    
    if len(submission_part) > 1:
        merged_list = []
        current = submission_part.iloc[0].to_dict()
        
        for i in range(1, len(submission_part)):
            next_seg = submission_part.iloc[i]
            gap = next_seg['start_frame'] - current['stop_frame']
            if (gap <= 8 and 
                next_seg['action'] == current['action'] and
                next_seg['agent_id'] == current['agent_id'] and
                next_seg['target_id'] == current['target_id']):
                current['stop_frame'] = next_seg['stop_frame']
            else:
                merged_list.append(current)
                current = next_seg.to_dict()
        
        merged_list.append(current)
        submission_part = pd.DataFrame(merged_list)
    
    if len(submission_part) > 0:
        assert (submission_part.stop_frame > submission_part.start_frame).all()
    
    if verbose:
        print(f"    Actions found: {len(submission_part)}")
    
    return submission_part
#============= ROBUSTIFY & FALLBACK =============#
def robustify(submission, dataset, traintest, traintest_directory=None):
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    submission = submission[submission.start_frame < submission.stop_frame].copy()
    submission = submission.dropna(subset=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
    submission['start_frame'] = submission['start_frame'].astype(int)
    submission['stop_frame'] = submission['stop_frame'].astype(int)
    submission['video_id'] = submission['video_id'].astype(int)

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

    if len(group_list) > 0:
        submission = pd.concat(group_list, ignore_index=True)
    else:
        submission = pd.DataFrame(columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])

    batch_len = 200
    s_list = []

    for video_id in dataset.video_id.unique():
        video_id_int = int(video_id)
        vid_sub = submission[submission.video_id == video_id_int]
        lab_id = dataset[dataset.video_id == video_id_int].lab_id.iloc[0]
        path = f"{traintest_directory}/{lab_id}/{video_id_int}.parquet"
        
        try:
            vid = pd.read_parquet(path)
        except Exception:
            continue
            
        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1

        all_agents_targets = set()
        for agent in vid.mouse_id.unique():
            all_agents_targets.add((f"mouse{agent}", 'self'))
            for target in vid.mouse_id.unique():
                if agent != target:
                    all_agents_targets.add((f"mouse{agent}", f"mouse{target}"))

        for agent, target in all_agents_targets:
            actions = vid_sub[(vid_sub.agent_id == agent) & (vid_sub.target_id == target)]
            if len(actions) == 0:
                for i in range((stop_frame - start_frame + batch_len - 1) // batch_len):
                    batch_start = start_frame + i * batch_len
                    batch_stop = min(batch_start + batch_len, stop_frame)
                    s_list.append((video_id_int, agent, target, 'other', batch_start, batch_stop))
            else:
                covered = set()
                for _, action_row in actions.iterrows():
                    covered.update(range(int(action_row['start_frame']), int(action_row['stop_frame'])))

                uncovered = set(range(start_frame, stop_frame)) - covered
                if len(uncovered) > 0:
                    uncovered_sorted = sorted(uncovered)
                    gap_start = uncovered_sorted[0]
                    for j in range(1, len(uncovered_sorted)):
                        if uncovered_sorted[j] != uncovered_sorted[j-1] + 1:
                            gap_stop = uncovered_sorted[j-1] + 1
                            for k in range((gap_stop - gap_start + batch_len - 1) // batch_len):
                                batch_start = gap_start + k * batch_len
                                batch_stop = min(batch_start + batch_len, gap_stop)
                                s_list.append((video_id_int, agent, target, 'other', batch_start, batch_stop))
                            gap_start = uncovered_sorted[j]
                    gap_stop = uncovered_sorted[-1] + 1
                    for k in range((gap_stop - gap_start + batch_len - 1) // batch_len):
                        batch_start = gap_start + k * batch_len
                        batch_stop = min(batch_start + batch_len, gap_stop)
                        s_list.append((video_id_int, agent, target, 'other', batch_start, batch_stop))
                else:
                    for i, (_, action_row) in enumerate(actions.iterrows()):
                        batch_start = start_frame + i * batch_len
                        batch_stop = min(batch_start + batch_len, stop_frame)
                        s_list.append((video_id_int, agent, target, action_row['action'], batch_start, batch_stop))

    if len(s_list) > 0:
        submission = pd.concat([
            submission,
            pd.DataFrame(s_list, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
        ])

    submission = submission.reset_index(drop=True)
    submission['video_id'] = submission['video_id'].astype(int)
    submission['start_frame'] = submission['start_frame'].astype(int)
    submission['stop_frame'] = submission['stop_frame'].astype(int)
    
    return submission
#============= SUBMIT ENSEMBLE =============#
def submit_ensemble(body_parts_tracked_str, switch_tr, X_tr, label, meta, n_samples, section_id):
    models = []
    gpu_device = 'gpu' if GPU_AVAILABLE else 'cpu'
    
    if XGBOOST_AVAILABLE:
        xgb_device = 'gpu_hist' if GPU_AVAILABLE else 'hist'
        
        models.append(('XGB_v1', make_pipeline(
            StratifiedSubsetClassifier(
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.08,
                    max_depth=8,
                    min_child_weight=3,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    gamma=0.1,
                    reg_alpha=0.05,
                    reg_lambda=1.0,
                    tree_method=xgb_device, 
                    verbosity=0,
                    random_state=SEED
                ), int(n_samples/1.5),
            )
        )))
        
        models.append(('XGB_v2', make_pipeline(
            StratifiedSubsetClassifier(
                XGBClassifier(
                    n_estimators=320,
                    learning_rate=0.07,
                    max_depth=9,
                    min_child_weight=2,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0.15,
                    reg_alpha=0.1,
                    reg_lambda=1.2,
                    tree_method=xgb_device, 
                    verbosity=0,
                    random_state=SEED + 1
                ), int(n_samples/1.5),
            )
        )))
        
        models.append(('XGB_v3', make_pipeline(
            StratifiedSubsetClassifier(
                XGBClassifier(
                    n_estimators=280,
                    learning_rate=0.065,
                    max_depth=7,
                    min_child_weight=4,
                    subsample=0.82,
                    colsample_bytree=0.82,
                    gamma=0.12,
                    reg_alpha=0.08,
                    reg_lambda=1.1,
                    tree_method=xgb_device, 
                    verbosity=0,
                    random_state=SEED + 2
                ), int(n_samples/1.5),
            )
        )))
        
    if CATBOOST_AVAILABLE:
        cat_device = 'GPU' if GPU_AVAILABLE else 'CPU'

        models.append(('CatBoost_v1', make_pipeline(
            StratifiedSubsetClassifier(
                CatBoostClassifier(
                    iterations=280,
                    learning_rate=0.09,
                    depth=7,
                    l2_leaf_reg=3.0,
                    bootstrap_type='Bayesian',
                    bagging_temperature=0.8,
                    random_strength=1.0,
                    task_type=cat_device,
                    verbose=False, 
                    allow_writing_files=False,
                    random_seed=SEED
                ), n_samples,
            )
        )))
        
        models.append(('CatBoost_v2', make_pipeline(
            StratifiedSubsetClassifier(
                CatBoostClassifier(
                    iterations=300,
                    learning_rate=0.08,
                    depth=6,  
                    l2_leaf_reg=4.0,
                    bootstrap_type='MVS',
                    random_strength=0.8,
                    subsample=0.8,
                    task_type=cat_device,
                    verbose=False, 
                    allow_writing_files=False,
                    random_seed=SEED
                ), n_samples,
            )
        )))

    X_tr_np = X_tr.to_numpy(np.float32, copy=False)
    del X_tr; gc.collect()

    model_list = []
    for action in label.columns:
        y_raw = label[action].to_numpy()
        mask = ~pd.isna(y_raw)
        y_action = y_raw[mask].astype(int)
        
        if not (y_action == 0).all() and np.sum(y_action) >= 3:
            idx = np.flatnonzero(mask)
            X_action = X_tr_np[idx]
            
            groups_action = meta['video_id'].values[idx]
            
            max_cv_samples = 500000
            if len(X_action) > max_cv_samples:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=max_cv_samples, random_state=42)
                cv_idx, _ = next(sss.split(X_action, y_action))
                X_cv = X_action[cv_idx]
                y_cv = y_action[cv_idx]
                groups_cv = groups_action[cv_idx]
            else:
                X_cv = X_action
                y_cv = y_action
                groups_cv = groups_action
            
            model_scores = []
            if verbose:
                print(f"    Evaluating models for action: {action} (samples: {len(y_cv)})")
            
            for model_name, model in models:
                try:
                    split_successful = False
                    try:
                        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                        train_idx, val_idx = next(gss.split(X_cv, y_cv, groups=groups_cv))
                        
                        if np.sum(y_cv[train_idx]) > 0 and np.sum(y_cv[val_idx]) > 0:
                            split_successful = True
                    except Exception:
                        split_successful = False

                    if not split_successful:
                        sss_fallback = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                        train_idx, val_idx = next(sss_fallback.split(X_cv, y_cv))
                    
                    X_train, X_val = X_cv[train_idx], X_cv[val_idx]
                    y_train, y_val = y_cv[train_idx], y_cv[val_idx]
                    
                    m_temp = clone(model)
                    m_temp.fit(X_train, y_train)
                    
                    y_pred = m_temp.predict_proba(X_val)[:, 1]
                    
                    if len(np.unique(y_val)) == 2:
                        score = roc_auc_score(y_val, y_pred)
                    else:
                        y_pred_class = (y_pred > 0.5).astype(int)
                        score = f1_score(y_val, y_pred_class, average='weighted')
                    
                    model_scores.append((model_name, model, score))
                    
                    if verbose:
                        print(f"      {model_name}: {score:.4f}")
                    
                    del m_temp, X_train, X_val, y_train, y_val, y_pred
                    gc.collect()
                        
                except Exception as e:
                    if verbose:
                        print(f"      {model_name}: Failed - {str(e)[:50]}")
                    gc.collect()
                    continue
            
            del X_cv, y_cv, groups_cv
            gc.collect()
            
            model_scores.sort(key=lambda x: x[2], reverse=True)
            top_models = model_scores[:3]
            
            if len(top_models) > 0:
                if verbose:
                    print(f"    Selected top {len(top_models)} models: {[m[0] for m in top_models]}")
                
                trained = []
                for model_name, model, score in top_models:
                    m_clone = clone(model)
                    m_clone.fit(X_action, y_action)
                    trained.append(m_clone)
                    gc.collect()
                
                model_list.append((action, trained, [m[2] for m in top_models]))
            
            del X_action
            gc.collect()

    del X_tr_np; gc.collect()

    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in DROP_BODY_PARTS]

    test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
    generator = generate_mouse_data(
        test_subset, 'test',
        generate_single=(switch_tr == 'single'),
        generate_pair=(switch_tr == 'pair')
    )

    fps_lookup = (
        test_subset[['video_id', 'frames_per_second']]
        .drop_duplicates('video_id')
        .set_index('video_id')['frames_per_second']
        .to_dict()
    )

    if verbose:
        print(f"    n_videos: {len(test_subset)}, n_candidate_models: {len(models)}")

    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr
        try:
            fps_i = _fps_from_meta(meta_te, fps_lookup, default_fps=30.0)

            if switch_te == 'single':
                X_te = transform_single(data_te, body_parts_tracked, fps_i).astype(np.float32)
            else:
                X_te = transform_pair(data_te, body_parts_tracked, fps_i).astype(np.float32)

            X_te_np = X_te.to_numpy(np.float32, copy=False)
            del X_te, data_te; gc.collect()

            pred = pd.DataFrame(index=meta_te.video_frame)
            for action, trained, scores in model_list:
                if action in actions_te:
                    probs = [m.predict_proba(X_te_np)[:, 1] for m in trained]
                    
                    weights = np.array(scores)
                    weights = weights / weights.sum()

                    ensemble_prob = np.average(probs, axis=0, weights=weights)
                    ensemble_prob = np.clip(ensemble_prob, 0.01, 0.99)
                    ensemble_prob = np.power(ensemble_prob, 0.95)
                    pred[action] = ensemble_prob

            del X_te_np; gc.collect()

            if pred.shape[1] != 0:
                sub_part = predict_multiclass_adaptive(
                    pred, meta_te, 
                    ADAPTIVE_THRESHOLDS_v1, 
                    ADAPTIVE_THRESHOLDS_v2, 
                    section_id,
                    weight_v1=0.5,
                    weight_v2=0.5
                )
                submission_list.append(sub_part)
            else:   
                if verbose:
                    print(f'    Warning: No trained models for video {meta_te.video_id.iloc[0]}. Using safe defaults.')
                
                if len(actions_te) > 0:
                    for act in actions_te:
                        pred[act] = 0.001 
                    
                    sub_part = predict_multiclass_adaptive(
                        pred, meta_te, 
                        ADAPTIVE_THRESHOLDS_v1, 
                        ADAPTIVE_THRESHOLDS_v2, 
                        section_id
                    )
                    submission_list.append(sub_part)

        except Exception as e:
            if verbose:
                print(f"    Error: {str(e)[:50]}")
            try:
                del data_te
            except Exception:
                pass
            gc.collect()
#============= MAIN LOOP =============#
try:
    import torch
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
except:
    GPU_AVAILABLE = False

submission_list = []
print(f"XGBoost: {XGBOOST_AVAILABLE}, CatBoost: {CATBOOST_AVAILABLE}\n")

for section in range(1, len(body_parts_tracked_list)):
    body_parts_tracked_str = body_parts_tracked_list[section]
    try:
        body_parts_tracked = json.loads(body_parts_tracked_str)
        print(f"#============= {section}. Processing: {len(body_parts_tracked)} body parts =============#")
        print(f"Body parts list: {body_parts_tracked}")
        
        if len(body_parts_tracked) > 5:
            body_parts_tracked = [b for b in body_parts_tracked if b not in DROP_BODY_PARTS]

        train_subset = train[train.body_parts_tracked == body_parts_tracked_str]

        _fps_lookup = (
            train_subset[['video_id', 'frames_per_second']]
            .drop_duplicates('video_id')
            .set_index('video_id')['frames_per_second']
            .to_dict()
        )

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
            single_feats_parts = []
            for data_i, meta_i in zip(single_list, single_meta_list):
                fps_i = _fps_from_meta(meta_i, _fps_lookup, default_fps=30.0)
                Xi = transform_single(data_i, body_parts_tracked, fps_i).astype(np.float32)
                single_feats_parts.append(Xi)

            X_tr = pd.concat(single_feats_parts, axis=0, ignore_index=True)
 
            single_label = pd.concat(single_label_list, axis=0, ignore_index=True)
            single_meta  = pd.concat(single_meta_list,  axis=0, ignore_index=True)

            del single_list, single_label_list, single_meta_list, single_feats_parts
            gc.collect()

            print(f"● Single: {X_tr.shape}")
            submit_ensemble(body_parts_tracked_str, 'single', X_tr, single_label, single_meta, 2_000_000, str(section)) 

            del X_tr, single_label, single_meta
            gc.collect()

        if len(pair_list) > 0:
            pair_feats_parts = []
            for data_i, meta_i in zip(pair_list, pair_meta_list):
                fps_i = _fps_from_meta(meta_i, _fps_lookup, default_fps=30.0)
                Xi = transform_pair(data_i, body_parts_tracked, fps_i).astype(np.float32)
                pair_feats_parts.append(Xi)

            X_tr = pd.concat(pair_feats_parts, axis=0, ignore_index=True)
            
            pair_label = pd.concat(pair_label_list, axis=0, ignore_index=True)
            pair_meta  = pd.concat(pair_meta_list,  axis=0, ignore_index=True)

            del pair_list, pair_label_list, pair_meta_list, pair_feats_parts
            gc.collect()

            print(f"● Pair: {X_tr.shape}")
            submit_ensemble(body_parts_tracked_str, 'pair', X_tr, pair_label, pair_meta, 900_000, str(section))

            del X_tr, pair_label, pair_meta
            gc.collect()

    except Exception as e:
        print(f"    Exception {str(e)[:100]}")

    gc.collect()
    print()

if len(submission_list) > 0:
    submission = pd.concat(submission_list, ignore_index=True)
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
print(f"Submission created: {len(submission_robust)} predictions")

