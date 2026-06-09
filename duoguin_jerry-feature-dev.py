# =========================================================================================
# JERRY FEATURE DEV: PHYSICS-BASED & FPS-AWARE FEATURE ENGINEERING
# (Adapted from mabe-extra-trees-gpu.ipynb)
# =========================================================================================

import numpy as np
import pandas as pd
import itertools
import warnings
import json
import re
import gc
from typing import Dict, Optional, Tuple

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------
# 1. FPS SCALING HELPERS (Vũ khí bí mật để xử lý video khác FPS)
# -----------------------------------------------------------------------

def _scale(n_frames_at_30fps, fps, ref=30.0):
    """Scale frame count to current video's FPS."""
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))

def _scale_signed(n_frames_at_30fps, fps, ref=30.0):
    """Signed version for lags/shifts."""
    if n_frames_at_30fps == 0: return 0
    s = 1 if n_frames_at_30fps > 0 else -1
    mag = max(1, int(round(abs(n_frames_at_30fps) * float(fps) / ref)))
    return s * mag

def _fps_from_meta(meta_df, fallback_lookup, default_fps=30.0):
    if 'frames_per_second' in meta_df.columns and pd.notnull(meta_df['frames_per_second']).any():
        return float(meta_df['frames_per_second'].iloc[0])
    vid = meta_df['video_id'].iloc[0]
    return float(fallback_lookup.get(vid, default_fps))

# -----------------------------------------------------------------------
# 2. FEATURE GENERATORS (Physics & Motion)
# -----------------------------------------------------------------------

def add_curvature_features(X, center_x, center_y, fps):
    vel_x = center_x.diff()
    vel_y = center_y.diff()
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()
    
    # Curvature: độ cong của quỹ đạo
    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)

    for w in [30, 60]:
        ws = _scale(w, fps)
        X[f'curv_mean_{w}'] = curvature.rolling(ws, min_periods=max(1, ws // 6)).mean()
    return X

def add_multiscale_features(X, center_x, center_y, fps):
    # Speed in cm/s (do pixel đã được normalized)
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)

    scales = [10, 40, 160]
    for scale in scales:
        ws = _scale(scale, fps)
        if len(speed) >= ws:
            X[f'sp_m{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).mean()
            X[f'sp_s{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).std()
    return X

def add_groom_microfeatures(X, df, fps):
    """Detect các chuyển động nhỏ đặc trưng của grooming"""
    parts = df.columns.get_level_values(0)
    if 'body_center' not in parts or 'nose' not in parts: return X

    cx, cy = df['body_center']['x'], df['body_center']['y']
    nx, ny = df['nose']['x'], df['nose']['y']

    cs = (np.sqrt(cx.diff()**2 + cy.diff()**2) * float(fps)).fillna(0)
    ns = (np.sqrt(nx.diff()**2 + ny.diff()**2) * float(fps)).fillna(0)

    w30 = _scale(30, fps)
    # Tỷ lệ chuyển động mũi / thân (Grooming: Mũi động, thân tĩnh)
    X['head_body_decouple'] = (ns / (cs + 1e-3)).clip(0, 10).rolling(w30, min_periods=1).median()
    
    # Bán kính mũi so với thân (biến động khi rúc đầu)
    r = np.sqrt((nx - cx)**2 + (ny - cy)**2)
    X['nose_rad_std'] = r.rolling(w30, min_periods=1).std().fillna(0)
    return X

def add_interaction_features(X, mouse_pair, avail_A, avail_B, fps):
    """Tính các feature xã hội quan trọng: Chase, Approach"""
    if 'body_center' not in avail_A or 'body_center' not in avail_B: return X

    # Vị trí tương đối
    rel_x = mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x']
    rel_y = mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y']
    rel_dist = np.sqrt(rel_x**2 + rel_y**2)

    # Vận tốc
    A_vx = mouse_pair['A']['body_center']['x'].diff()
    A_vy = mouse_pair['A']['body_center']['y'].diff()
    B_vx = mouse_pair['B']['body_center']['x'].diff()
    B_vy = mouse_pair['B']['body_center']['y'].diff()

    # Ai đang dẫn đầu? (Leader-Follower dynamics)
    # Chiếu vận tốc lên vector nối tâm
    A_lead = (A_vx * rel_x + A_vy * rel_y) / (np.sqrt(A_vx**2 + A_vy**2) * rel_dist + 1e-6)
    B_lead = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (np.sqrt(B_vx**2 + B_vy**2) * rel_dist + 1e-6)

    # Chase metric: Khoảng cách giảm VÀ B đang dẫn đầu (A đuổi B)
    approach = -rel_dist.diff()
    chase = approach * B_lead
    
    ws = _scale(30, fps)
    X[f'chase_{30}'] = chase.rolling(ws, min_periods=1).mean()
    
    # Correlation vận tốc (Di chuyển cùng nhau?)
    ws_cor = _scale(60, fps)
    A_sp = np.sqrt(A_vx**2 + A_vy**2)
    B_sp = np.sqrt(B_vx**2 + B_vy**2)
    X[f'sp_cor{60}'] = A_sp.rolling(ws_cor, min_periods=1).corr(B_sp)

    return X

# -----------------------------------------------------------------------
# 3. CORE TRANSFORM FUNCTIONS (GỌI TỪ FILE TRAIN/INFERENCE)
# -----------------------------------------------------------------------

def transform_single(single_mouse, body_parts_tracked, fps):
    """Main Feature Engineering cho hành vi SINGLE"""
    available_body_parts = single_mouse.columns.get_level_values(0)

    # 1. Distances between all body parts
    X = pd.DataFrame({
        f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.combinations(body_parts_tracked, 2)
        if p1 in available_body_parts and p2 in available_body_parts
    })
    
    # 2. Body Angle
    if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
        v1 = single_mouse['nose'] - single_mouse['body_center']
        v2 = single_mouse['tail_base'] - single_mouse['body_center']
        X['body_ang'] = (v1['x'] * v2['x'] + v1['y'] * v2['y']) / (
            np.sqrt(v1['x']**2 + v1['y']**2) * np.sqrt(v2['x']**2 + v2['y']**2) + 1e-6)

    # 3. Motion Stats (FPS-scaled)
    if 'body_center' in available_body_parts:
        cx, cy = single_mouse['body_center']['x'], single_mouse['body_center']['y']
        
        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)
            X[f'cx_m{w}'] = cx.rolling(ws, **roll).mean()
            X[f'act{w}'] = np.sqrt(cx.diff().rolling(ws, min_periods=1).var() +
                                   cy.diff().rolling(ws, min_periods=1).var()) # Activity level

        # Advanced Physics
        X = add_curvature_features(X, cx, cy, fps)
        X = add_multiscale_features(X, cx, cy, fps)
        X = add_groom_microfeatures(X, single_mouse, fps)

    # 4. Lagged features (Duration-aware)
    if all(p in available_body_parts for p in ['nose', 'tail_base']):
        nt_dist = np.sqrt((single_mouse['nose']['x'] - single_mouse['tail_base']['x'])**2 +
                          (single_mouse['nose']['y'] - single_mouse['tail_base']['y'])**2)
        for lag in [10, 20]:
            l = _scale(lag, fps)
            X[f'nt_df{lag}'] = nt_dist - nt_dist.shift(l) # Elongation change (Stretch/Compress)

    return X.astype(np.float32, copy=False)


def transform_pair(mouse_pair, body_parts_tracked, fps):
    """Main Feature Engineering cho hành vi PAIR"""
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)

    # 1. Cross-Mouse Distances
    X = pd.DataFrame({
        f"12+{p1}+{p2}": np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.product(body_parts_tracked, repeat=2)
        if p1 in avail_A and p2 in avail_B
    })

    # 2. Nose-Nose Dynamics (Quan trọng nhất cho Sniff/Attack)
    if 'nose' in avail_A and 'nose' in avail_B:
        nn = np.sqrt((mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x'])**2 +
                     (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y'])**2)
        for lag in [10, 20]:
            l = _scale(lag, fps)
            X[f'nn_ch{lag}']  = nn - nn.shift(l) # Đang đến gần nhau hay ra xa?
            is_cl = (nn < 10.0).astype(float)
            X[f'cl_ps{lag}']  = is_cl.rolling(l, min_periods=1).mean() # Thời gian tiếp xúc gần

    # 3. Advanced Interaction
    X = add_interaction_features(X, mouse_pair, avail_A, avail_B, fps)

    # 4. Relative Orientation
    if all(p in avail_A for p in ['nose', 'tail_base']) and all(p in avail_B for p in ['nose', 'tail_base']):
        dir_A = mouse_pair['A']['nose'] - mouse_pair['A']['tail_base']
        dir_B = mouse_pair['B']['nose'] - mouse_pair['B']['tail_base']
        # Cosine similarity giữa 2 vector thân
        X['rel_ori'] = (dir_A['x'] * dir_B['x'] + dir_A['y'] * dir_B['y']) / (
            np.sqrt(dir_A['x']**2 + dir_A['y']**2) * np.sqrt(dir_B['x']**2 + dir_B['y']**2) + 1e-6)

    return X.astype(np.float32, copy=False)

# -----------------------------------------------------------------------
# 4. DATA GENERATOR (Parser thông minh)
# -----------------------------------------------------------------------

def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    """
    Iterator thông minh: Đọc file tracking -> Tách luồng Single/Pair -> Yield data
    """
    assert traintest in ['train', 'test']
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    # Regex helper để resolve mouse_id (vấn đề đau đầu nhất của dataset này)
    def _to_num(x):
        if isinstance(x, (int, np.integer)): return int(x)
        m = re.search(r'(\d+)$', str(x))
        return int(m.group(1)) if m else None

    drop_body_parts = ['spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint'] # Giảm nhiễu

    for _, row in dataset.iterrows():
        lab_id, video_id = row.lab_id, row.video_id
        fps = float(row.frames_per_second)

        # Skip nếu video không có label string (lỗi dataset)
        if not isinstance(row.behaviors_labeled, str) and traintest == 'train': continue

        # --- LOAD TRACKING ---
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        try:
            vid = pd.read_parquet(path)
        except: continue
            
        # Pivot table để chuyển thành dạng [frame] x [mouse_id] x [body_part]
        pvid = vid.pivot(columns=['mouse_id','bodypart'], index='video_frame', values=['x','y'])
        del vid
        pvid = pvid.reorder_levels([1,2,0], axis=1).T.sort_index().T
        pvid = (pvid / float(row.pix_per_cm_approx)).astype('float32', copy=False) # Normalize sang cm

        avail = list(pvid.columns.get_level_values('mouse_id').unique())
        avail_set = set(avail) | set(map(str, avail)) | {f"mouse{_to_num(a)}" for a in avail if _to_num(a) is not None}

        # Hàm map ID trong label (vd: "mouse1") sang ID trong tracking (vd: 0, 1)
        def _resolve(agent_str):
            m = re.search(r'(\d+)$', str(agent_str))
            cand = [agent_str]
            if m:
                n = int(m.group(1))
                cand = [n, n-1, str(n), f"mouse{n}", agent_str]
            for c in cand:
                if c in avail_set:
                    if c in set(avail): return c
                    for a in avail:
                        if str(a) == str(c) or f"mouse{_to_num(a)}" == str(c): return a
            return None

        # --- PARSE LABELS ---
        if traintest == 'train':
            annot = pd.read_parquet(path.replace('train_tracking', 'train_annotation'))
        
        # Parse behaviors_labeled json string
        try:
            vb = json.loads(row.behaviors_labeled if traintest == 'train' else row.behaviors_labeled)
        except:
            vb = []
        vb = sorted(list({b.replace("'", "") for b in vb}))
        vb = pd.DataFrame([b.split(',') for b in vb], columns=['agent','target','action'])
        if vb.empty: continue
        
        # Meta info
        def _mk_meta(index, agent_id, target_id):
            return pd.DataFrame({
                'video_id': video_id, 'agent_id': agent_id, 'target_id': target_id,
                'video_frame': index.astype('int32'), 'frames_per_second': np.float32(fps)
            })

        # --- YIELD SINGLE DATA ---
        if generate_single:
            vb_single = vb.query("target == 'self'")
            for agent_str in pd.unique(vb_single['agent']):
                col_lab = _resolve(agent_str)
                if col_lab is None: continue
                
                actions = sorted(vb_single.loc[vb_single['agent'].eq(agent_str), 'action'].unique().tolist())
                if not actions: continue

                single = pvid.loc[:, col_lab]
                meta_df = _mk_meta(single.index, agent_str, 'self')

                if traintest == 'train':
                    # Tạo Binary Labels cho từng frame
                    y = pd.DataFrame(False, index=single.index.astype('int32'), columns=actions)
                    a_num = _to_num(col_lab)
                    a_sub = annot.query("(agent_id == @a_num) & (target_id == @a_num)")
                    for i in range(len(a_sub)):
                        ar = a_sub.iloc[i]
                        a = str(ar.action).lower()
                        if a in y.columns:
                            y.loc[int(ar['start_frame']):int(ar['stop_frame']), a] = True
                    yield 'single', single, meta_df, y
                else:
                    yield 'single', single, meta_df, actions

        # --- YIELD PAIR DATA ---
        if generate_pair:
            vb_pair = vb.query("target != 'self'")
            if len(vb_pair) > 0:
                allowed_pairs = set(map(tuple, vb_pair[['agent','target']].itertuples(index=False, name=None)))
                
                # Permutation tất cả các cặp có thể
                for agent_num, target_num in itertools.permutations(np.unique(pvid.columns.get_level_values('mouse_id')), 2):
                    agent_str = f"mouse{_to_num(agent_num)}"
                    target_str = f"mouse{_to_num(target_num)}"
                    
                    if (agent_str, target_str) not in allowed_pairs: continue

                    a_col = _resolve(agent_str)
                    b_col = _resolve(target_str)
                    if a_col is None or b_col is None: continue

                    actions = sorted(vb_pair.query("(agent == @agent_str) & (target == @target_str)")['action'].unique().tolist())
                    if not actions: continue

                    pair_xy = pd.concat([pvid[a_col], pvid[b_col]], axis=1, keys=['A','B'])
                    meta_df = _mk_meta(pair_xy.index, agent_str, target_str)

                    if traintest == 'train':
                        a_num = _to_num(a_col); b_num = _to_num(b_col)
                        y = pd.DataFrame(False, index=pair_xy.index.astype('int32'), columns=actions)
                        a_sub = annot.query("(agent_id == @a_num) & (target_id == @b_num)")
                        for i in range(len(a_sub)):
                            ar = a_sub.iloc[i]
                            a = str(ar.action).lower()
                            if a in y.columns:
                                y.loc[int(ar['start_frame']):int(ar['stop_frame']), a] = True
                        yield 'pair', pair_xy, meta_df, y
                    else:
                        yield 'pair', pair_xy, meta_df, actions

print("✅ Đã load xong các hàm Feature Engineering (Phiên bản Extra-Trees GPU).")

