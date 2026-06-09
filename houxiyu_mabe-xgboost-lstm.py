# Cell 1: 依赖 & 配置
import os
import json
import gc
import glob
import joblib
import warnings
import itertools
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import f1_score

from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ====== Config ======
class CFG:
    # Paths - Kaggle competition default layout
    TRAIN_CSV = "/kaggle/input/MABe-mouse-behavior-detection/train.csv"
    TEST_CSV = "/kaggle/input/MABe-mouse-behavior-detection/test.csv"
    TRAIN_TRACKING = "/kaggle/input/MABe-mouse-behavior-detection/train_tracking"
    TRAIN_ANNOT = "/kaggle/input/MABe-mouse-behavior-detection/train_annotation"
    TEST_TRACKING = "/kaggle/input/MABe-mouse-behavior-detection/test_tracking"

    MODEL_DIR = "/kaggle/working/models_xgb_lstm"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # XGB params (frame-level classifier)
    XGB_PARAMS = dict(
        verbosity=0,
        random_state=SEED,
        n_estimators=200,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False
    )

    N_SPLITS = 3
    SEQ_LEN = 16
    LSTM_HIDDEN = 64
    LSTM_LAYERS = 1
    LSTM_DROPOUT = 0.1
    LSTM_EPOCHS = 6
    BATCH_SIZE = 128
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", CFG.DEVICE)



# Cell 2: 公共工具 & 特征变换函数（transform_single / transform_pair 等）
# 包含：_scale, _scale_signed, _fps_from_meta, add_* 特征, transform_single, transform_pair

DROP_BODY_PARTS = [
    'headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright',
    'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright',
    'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint'
]

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

def safe_rolling(series, window, func, min_periods=None):
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series.rolling(window, min_periods=min_periods, center=True).apply(func, raw=True)

# --- complex feature helpers ---
def add_curvature_features(X, center_x, center_y, fps):
    vel_x = center_x.diff()
    vel_y = center_y.diff()
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()

    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)

    for w in [25, 50, 75]:
        ws = _scale(w, fps)
        X[f'curv_mean_{w}'] = curvature.rolling(ws, min_periods=max(1, ws // 5)).mean()

    angle = np.arctan2(vel_y, vel_x)
    angle_change = np.abs(angle.diff())
    w = 30
    ws = _scale(w, fps)
    X[f'turn_rate_{w}'] = angle_change.rolling(ws, min_periods=max(1, ws // 5)).sum()

    return X

def add_multiscale_features(X, center_x, center_y, fps):
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
    scales = [20, 40, 60, 80]
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
        for window in [20, 40, 60, 80]:
            ws = _scale(window, fps)
            if len(speed_states) >= ws:
                for state in [0, 1, 2, 3]:
                    X[f's{state}_{window}'] = (
                        (speed_states == state).astype(float)
                        .rolling(ws, min_periods=max(1, ws // 5)).mean()
                    )
                state_changes = (speed_states != speed_states.shift(1)).astype(float)
                X[f'trans_{window}'] = state_changes.rolling(ws, min_periods=max(1, ws // 5)).sum()
    except Exception:
        pass
    return X

def add_longrange_features(X, center_x, center_y, fps):
    for window in [30, 60, 120]:
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

# --- transforms (based on your template) ---
def transform_single(single_mouse, body_parts_tracked, fps):
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
        X = add_curvature_features(X, cx, cy, fps)
        X = add_multiscale_features(X, cx, cy, fps)
        X = add_state_features(X, cx, cy, fps)
        X = add_longrange_features(X, cx, cy, fps)
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
    return X.astype(np.float32, copy=False)



# Cell 3: generate_mouse_data, robustify, predict_multiclass (segmentation)
def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    """
    Yields tuples:
      ('single'/'pair', data_df, meta_df, label_df or actions_list)
    data_df: single -> MultiIndex columns (bodypart, x/y)
             pair -> columns keys ['A','B'] each has bodyparts
    meta_df: has video_id, agent_id, target_id, video_frame
    label_df: if traintest == 'train' returns DataFrame of per-frame binary labels for actions (index aligned)
    """
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    for _, row in dataset.iterrows():
        lab_id = row.lab_id
        # skip MABe22 if desired (they are supplemental)
        if lab_id.startswith('MABe22') or type(row.behaviors_labeled) != str:
            continue
        video_id = row.video_id
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        if not os.path.exists(path):
            continue
        try:
            vid = pd.read_parquet(path)
        except Exception:
            continue
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@DROP_BODY_PARTS)")
        try:
            pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        except Exception:
            continue
        del vid
        gc.collect()
        # ensure order of levels consistent
        try:
            pvid = pvid.reorder_levels([1,2,0], axis=1).T.sort_index().T
        except Exception:
            # fallback: ensure second level is bodypart
            pass
        pvid /= row.pix_per_cm_approx

        # behaviors available for this video
        try:
            vid_behaviors = json.loads(row.behaviors_labeled)
            vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
            vid_behaviors = [b.split(',') for b in vid_behaviors]
            vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
        except Exception:
            vid_behaviors = pd.DataFrame([["mouse1","self","groom"]], columns=['agent','target','action'])

        annot = None
        if traintest == 'train':
            annot_path = path.replace('train_tracking', 'train_annotation')
            if os.path.exists(annot_path):
                try:
                    annot = pd.read_parquet(annot_path)
                except Exception:
                    annot = None

        # SINGLE
        if generate_single:
            vid_behaviors_subset = vid_behaviors.query("target == 'self'") if not vid_behaviors.empty else pd.DataFrame([["mouse1","self","groom"]], columns=['agent','target','action'])
            for mouse_id_str in np.unique(vid_behaviors_subset.agent):
                try:
                    mouse_id = int(mouse_id_str[-1])
                    actions = np.unique(vid_behaviors_subset.query("agent == @mouse_id_str").action)
                    single_mouse = pvid.loc[:, mouse_id]
                    single_mouse_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': mouse_id_str,
                        'target_id': 'self',
                        'video_frame': single_mouse.index,
                        'frames_per_second': row.frames_per_second
                    })
                    if traintest == 'train' and annot is not None:
                        labels = pd.DataFrame(0.0, index=single_mouse.index, columns=actions)
                        annot_subset = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for _, arow in annot_subset.iterrows():
                            act = arow['action'] if arow['action'] in actions else actions[0]
                            start = max(0, int(arow['start_frame']))
                            stop = min(len(labels)-1, int(arow['stop_frame']))
                            labels.loc[start:stop, act] = 1.0
                        yield 'single', single_mouse.reset_index(drop=True), single_mouse_meta.reset_index(drop=True), labels.reset_index(drop=True)
                    else:
                        yield 'single', single_mouse.reset_index(drop=True), single_mouse_meta.reset_index(drop=True), actions
                except Exception:
                    continue

        # PAIR
        if generate_pair:
            vid_behaviors_subset = vid_behaviors.query("target != 'self'") if not vid_behaviors.empty else pd.DataFrame([["mouse1","mouse2","sniff"]], columns=['agent','target','action'])
            if len(vid_behaviors_subset) > 0:
                mice = pvid.columns.get_level_values(0).unique()
                for agent, target in itertools.permutations(mice, 2):
                    agent_str = f"mouse{agent}"
                    target_str = f"mouse{target}"
                    actions = np.unique(vid_behaviors_subset.query("(agent == @agent_str) & (target == @target_str)").action)
                    try:
                        agent_data = pvid.loc[:, agent]
                        target_data = pvid.loc[:, target]
                        pair_data = pd.concat([agent_data, target_data], axis=1, keys=['A','B'])
                        pair_meta = pd.DataFrame({
                            'video_id': video_id,
                            'agent_id': agent_str,
                            'target_id': target_str,
                            'video_frame': pair_data.index,
                            'frames_per_second': row.frames_per_second
                        })
                        if traintest == 'train' and annot is not None:
                            labels = pd.DataFrame(0.0, index=pair_data.index, columns=actions)
                            annot_subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                            for _, arow in annot_subset.iterrows():
                                act = arow['action'] if arow['action'] in actions else actions[0]
                                start = max(0, int(arow['start_frame']))
                                stop = min(len(labels)-1, int(arow['stop_frame']))
                                labels.loc[start:stop, act] = 1.0
                            yield 'pair', pair_data.reset_index(drop=True), pair_meta.reset_index(drop=True), labels.reset_index(drop=True)
                        else:
                            yield 'pair', pair_data.reset_index(drop=True), pair_meta.reset_index(drop=True), actions
                    except Exception:
                        continue

# robustify function (simpler variant)
def robustify(submission, dataset, traintest, traintest_directory=None):
    # basic cleaning + fill videos with default segments if nothing predicted
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    submission = submission[submission.start_frame < submission.stop_frame].copy()
    # remove overlapping segments inside same group
    group_list = []
    for _, group in submission.groupby(['video_id','agent_id','target_id']):
        group = group.sort_values('start_frame').reset_index(drop=True)
        keep = []
        last_stop = -1
        for i, row in group.iterrows():
            if row.start_frame >= last_stop:
                keep.append(i)
                last_stop = row.stop_frame
        group_list_



# Cell 4: Stage 1 - 生成帧级特征和标签（仅训练集）
train_meta = pd.read_csv(CFG.TRAIN_CSV)
train_meta['n_mice'] = 4 - train_meta[['mouse1_strain','mouse2_strain','mouse3_strain','mouse4_strain']].isna().sum(axis=1)

# Optionally limit how many videos to process for quick debug
MAX_VIDEOS = 80        # 调试时设小，发布前设为 None 或更大
MAX_SEGMENTS = 600     # 限制生成的片段数（防 OOM）

print("Collecting frame-level features and labels...")
frame_feats = []
frame_labels = []
frame_meta = []
processed = 0
fps_lookup = train_meta[['video_id','frames_per_second']].drop_duplicates('video_id').set_index('video_id')['frames_per_second'].to_dict()

for i, row in train_meta.iterrows():
    if MAX_VIDEOS is not None and i >= MAX_VIDEOS:
        break
    lab_id = row.lab_id
    if lab_id.startswith('MABe22'):
        continue
    # build small dataset for this single row
    sub = train_meta.loc[[i]]
    gen = generate_mouse_data(sub, 'train')
    for switch, data, meta, label in gen:
        try:
            fps = _fps_from_meta(meta, fps_lookup)
            if switch == 'single':
                body_parts = list(data.columns.get_level_values(0).unique())
                X = transform_single(data, body_parts, fps)
            else:
                body_parts = list(data['A'].columns.get_level_values(0).unique())
                X = transform_pair(data, body_parts, fps)
            if X.shape[0] == 0:
                continue
            # align label length
            if isinstance(label, pd.DataFrame):
                minlen = min(len(X), len(label))
                X = X.iloc[:minlen].reset_index(drop=True)
                label = label.iloc[:minlen].reset_index(drop=True)
                meta = meta.iloc[:minlen].reset_index(drop=True)
            else:
                # no labels (rare), skip
                continue
            frame_feats.append(X)
            frame_labels.append(label)
            frame_meta.append(meta)
            processed += 1
            if MAX_SEGMENTS is not None and processed >= MAX_SEGMENTS:
                break
        except Exception as e:
            print("Warning gen:", e)
            continue
    if MAX_SEGMENTS is not None and processed >= MAX_SEGMENTS:
        break

if len(frame_feats) == 0:
    raise RuntimeError("No frames collected. Check paths and data availability.")

X_all = pd.concat(frame_feats, ignore_index=True)
y_all = pd.concat(frame_labels, ignore_index=True).fillna(0.0)
meta_all = pd.concat(frame_meta, ignore_index=True)

print("Collected frames:", len(X_all), "features:", X_all.shape[1], "actions:", y_all.shape[1])

# standardize
scaler = StandardScaler()
X_all_vals = scaler.fit_transform(X_all.values.astype(np.float32))
joblib.dump(scaler, os.path.join(CFG.MODEL_DIR, "scaler.pkl"))
print("Saved scaler.")



# Cell 5: Stage 2 - 每行为 XGBoost 训练（Group CV）并生成 OOF 概率 + 阈值
action_list = list(y_all.columns)
print("Actions:", action_list)

cv = StratifiedGroupKFold(n_splits=min(CFG.N_SPLITS, len(meta_all.video_id.unique())))
oof = pd.DataFrame(0.0, index=range(len(X_all_vals)), columns=action_list)
xgb_models = {}

for action in action_list:
    y = y_all[action].values.astype(int)
    if y.sum() == 0:
        print(f"Action {action}: no positives, skip.")
        continue
    fold_preds = np.zeros(len(y))
    models = []
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_all_vals, y, groups=meta_all.video_id)):
        X_tr, X_val = X_all_vals[tr_idx], X_all_vals[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        clf = XGBClassifier(**CFG.XGB_PARAMS)
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False)
        p = clf.predict_proba(X_val)[:,1]
        fold_preds[val_idx] = p
        models.append(clf)
        print(f"Action {action} fold {fold} done.")
    oof[action] = fold_preds
    xgb_models[action] = models
    joblib.dump(models, os.path.join(CFG.MODEL_DIR, f"xgb_models_{action}.pkl"))
    print(f"Saved XGB models for action {action}.")

joblib.dump(oof, os.path.join(CFG.MODEL_DIR, "xgb_oof_probs.pkl"))
print("Saved OOF probs.")
# tune thresholds per action by simple grid search
thresholds = {}
for action in action_list:
    if y_all[action].sum() == 0:
        thresholds[action] = 0.5
        continue
    best_f1 = 0.0
    best_t = 0.5
    for t in np.linspace(0.05, 0.95, 19):
        pred = (oof[action].values >= t).astype(int)
        f1 = f1_score(y_all[action].values.astype(int), pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    thresholds[action] = float(best_t)
    print(f"Action {action}: best_th={best_t:.2f} f1={best_f1:.4f}")

joblib.dump(thresholds, os.path.join(CFG.MODEL_DIR, "xgb_thresholds.pkl"))
print("Saved thresholds.")



# Cell 6: Stage 3 - 用 XGB 的 OOF 概率 + 特征作为 LSTM 输入来学习时序校正
# Build seq inputs: [standardized features || xgb_oof_probs]
xgb_oof = joblib.load(os.path.join(CFG.MODEL_DIR, "xgb_oof_probs.pkl"))
X_seq_all = np.hstack([X_all_vals.astype(np.float32), xgb_oof.values.astype(np.float32)])
y_multi = y_all.values.astype(np.float32)

# make sequence windows grouped by video (non-overlapping windows of CFG.SEQ_LEN)
seq_feats = []
seq_labels = []
seq_groups = []
for vid in meta_all.video_id.unique():
    idxs = meta_all.index[meta_all.video_id==vid].tolist()
    if len(idxs) < CFG.SEQ_LEN:
        continue
    for start in range(0, len(idxs) - CFG.SEQ_LEN + 1, CFG.SEQ_LEN):
        i0 = idxs[start]
        i1 = idxs[start+CFG.SEQ_LEN-1] + 1
        seq_feats.append(X_seq_all[i0:i1])
        seq_labels.append(y_multi[i0:i1])
        seq_groups.append(int(vid))

seq_feats = np.array(seq_feats, dtype=np.float32)
seq_labels = np.array(seq_labels, dtype=np.float32)
print("Sequence windows:", seq_feats.shape)

# Dataset & dataloader
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X; self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=SEED)
train_idx, val_idx = next(gss.split(seq_feats, groups=seq_groups))
train_ds = SeqDataset(seq_feats[train_idx], seq_labels[train_idx])
val_ds = SeqDataset(seq_feats[val_idx], seq_labels[val_idx])
train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False)

# LSTM model
class LSTMRefiner(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=CFG.LSTM_HIDDEN, layers=CFG.LSTM_LAYERS, dropout=CFG.LSTM_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout if layers>1 else 0)
        self.fc = nn.Linear(hidden*2, output_dim)
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(h)

input_dim = seq_feats.shape[2]
output_dim = seq_labels.shape[2]
model = LSTMRefiner(input_dim, output_dim).to(CFG.DEVICE)

# pos_weight from train labels
train_flat = seq_labels[train_idx].reshape(-1, output_dim)
pos = train_flat.sum(axis=0)
neg = train_flat.shape[0] - pos
pos_weight = torch.tensor([(neg[i]/(pos[i]+1e-6)) if pos[i] > 0 else 1.0 for i in range(output_dim)], dtype=torch.float32).to(CFG.DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LR)

# train loop
best_val_loss = 1e9
for epoch in range(CFG.LSTM_EPOCHS):
    model.train()
    train_loss = 0.0
    for Xb, yb in train_loader:
        Xb = Xb.to(CFG.DEVICE)
        yb = yb.to(CFG.DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * Xb.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(CFG.DEVICE); yb = yb.to(CFG.DEVICE)
            logits = model(Xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * Xb.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{CFG.LSTM_EPOCHS} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(CFG.MODEL_DIR, "best_lstm.pth"))

# save model & xgb models container
joblib.dump(xgb_models, os.path.join(CFG.MODEL_DIR, "xgb_models_all.pkl"))
torch.save(model.state_dict(), os.path.join(CFG.MODEL_DIR, "lstm_final.pth"))
print("Saved models.")



# Cell 7: Stage 4 - 在 test 子集上做推理并生成 submission.csv
test_meta = pd.read_csv(CFG.TEST_CSV)
# Optionally use subset to speed up debugging
TEST_MAX_VIDEOS = 50
test_subset = test_meta.head(TEST_MAX_VIDEOS).reset_index(drop=True)

# load resources
xgb_models_all = joblib.load(os.path.join(CFG.MODEL_DIR, "xgb_models_all.pkl"))
scaler = joblib.load(os.path.join(CFG.MODEL_DIR, "scaler.pkl"))
thresholds = joblib.load(os.path.join(CFG.MODEL_DIR, "xgb_thresholds.pkl"))
# load lstm
lstm_model = LSTMRefiner(input_dim, output_dim)
lstm_model.load_state_dict(torch.load(os.path.join(CFG.MODEL_DIR, "best_lstm.pth"), map_location=CFG.DEVICE))
lstm_model.to(CFG.DEVICE)
lstm_model.eval()

def predict_xgb_probs_for_clip(data_te, switch, body_parts_tracked, fps):
    if switch == 'single':
        X_te = transform_single(data_te, body_parts_tracked, fps)
    else:
        X_te = transform_pair(data_te, body_parts_tracked, fps)
    if X_te.shape[0] == 0:
        return None, None
    X_te_vals = scaler.transform(X_te.values.astype(np.float32))
    pred_df = pd.DataFrame(index=X_te.index, columns=action_list, dtype=float)
    for action in action_list:
        models = xgb_models_all.get(action, None)
        if models is None:
            pred_df[action] = 0.0
            continue
        probs = np.zeros((len(X_te_vals),))
        for m in models:
            probs += m.predict_proba(X_te_vals)[:,1]
        probs /= len(models)
        pred_df[action] = probs
    return X_te, pred_df

def refine_with_lstm(X_te, xgb_prob_df):
    X_vals = scaler.transform(X_te.values.astype(np.float32)).astype(np.float32)
    arr = np.hstack([X_vals, xgb_prob_df.values.astype(np.float32)])
    L = arr.shape[0]
    pad = (CFG.SEQ_LEN - (L % CFG.SEQ_LEN)) % CFG.SEQ_LEN
    if pad>0:
        arr = np.pad(arr, ((0,pad),(0,0)), mode='constant')
    arr = arr.reshape(-1, CFG.SEQ_LEN, arr.shape[1]).astype(np.float32)
    probs = []
    with torch.no_grad():
        for block in arr:
            xb = torch.tensor(block[None]).to(CFG.DEVICE)
            logits = lstm_model(xb)
            p = torch.sigmoid(logits)[0].cpu().numpy()
            probs.append(p)
    probs = np.vstack(probs)[:L]
    return pd.DataFrame(probs, index=X_te.index, columns=action_list)

submission_parts = []
fps_lookup_test = test_subset[['video_id','frames_per_second']].drop_duplicates('video_id').set_index('video_id')['frames_per_second'].to_dict()
for switch_te, data_te, meta_te, actions_te in generate_mouse_data(test_subset, 'test', generate_single=True, generate_pair=True):
    try:
        if switch_te == 'single':
            body_parts = list(data_te.columns.get_level_values(0).unique())
        else:
            body_parts = list(data_te['A'].columns.get_level_values(0).unique())
    except Exception:
        body_parts = []
    fps_i = _fps_from_meta(meta_te, fps_lookup_test, default_fps=30.0)
    X_te, xgb_probs = predict_xgb_probs_for_clip(data_te, switch_te, body_parts, fps_i)
    if X_te is None:
        continue
    refined_probs = refine_with_lstm(X_te, xgb_probs)
    sub_part = predict_multiclass(refined_probs, meta_te, thresholds)
    if not sub_part.empty:
        submission_parts.append(sub_part)

if submission_parts:
    submission = pd.concat(submission_parts, ignore_index=True)
else:
    submission = pd.DataFrame({
        'video_id':[0],
        'agent_id':['mouse1'],
        'target_id':['self'],
        'action':['groom'],
        'start_frame':[0],
        'stop_frame':[10]
    })

submission_robust = robustify(submission, test_meta, 'test', traintest_directory=CFG.TEST_TRACKING)
submission_robust.index.name = 'row_id'
submission_robust.to_csv("submission_xgb_lstm.csv", index=True)
print("Saved submission_xgb_lstm.csv")
submission_robust.head()


