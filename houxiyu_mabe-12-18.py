# ================= Step 1: ç�¯å¢ƒé…�ç½® =================
import pandas as pd
import numpy as np
import json
import gc
import os
import glob
import joblib
import itertools
import warnings
from tqdm.auto import tqdm
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

# è®¾ç½®æ˜¾ç¤ºé€‰é¡¹
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

class CFG:
    # è·¯å¾„é…�ç½®
    train_path = '/kaggle/input/MABe-mouse-behavior-detection/train.csv'
    test_path = '/kaggle/input/MABe-mouse-behavior-detection/test.csv'
    train_tracking_path = '/kaggle/input/MABe-mouse-behavior-detection/train_tracking'
    test_tracking_path = '/kaggle/input/MABe-mouse-behavior-detection/test_tracking'
    train_annotation_path = '/kaggle/input/MABe-mouse-behavior-detection/train_annotation'
    
    model_name = 'model'  # æ¨¡å�‹ä¿�å­˜è·¯å¾„
    seed = 42

print("Step 1: ç�¯å¢ƒé…�ç½®å®Œæˆ�ã€‚")


# ================= Step 2: è¯»å�–å…ƒæ•°æ�® =================
print("Step 2: æ­£åœ¨è¯»å�– CSV æ–‡ä»¶...")
train = pd.read_csv(CFG.train_path)
test = pd.read_csv(CFG.test_path)

# è¿‡æ»¤ 'MABe22' æ•°æ�®
train = train.query("~lab_id.str.startswith('MABe22_')").reset_index(drop=True)

# æ��å�–æ‰€æœ‰å‡ºç�°çš„â€œèº«ä½“éƒ¨ä½�è¿½è¸ªé…�ç½®â€�
body_parts_tracked_list = list(np.unique(train.body_parts_tracked))

# å®šä¹‰å†—ä½™éƒ¨ä½�
drop_body_parts = [
    'headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
    'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
    'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint'
]

print(f"è®­ç»ƒé›†: {train.shape}, æµ‹è¯•é›†: {test.shape}")


# ================= å��å¤„ç�†å·¥å…·å‡½æ•° =================
import pandas as pd
import numpy as np
import json
import os

def smooth_predictions(df, min_gap=30, min_duration=5):
    """
    å¼ºåŠ›å¹³æ»‘å‡½æ•°ï¼š
    1. Gap Filling: å¦‚æ�œä¸¤ä¸ªç›¸å�ŒåŠ¨ä½œé—´éš”å¾ˆçŸ­ (<30å¸§/1ç§’)ï¼ŒæŠŠå®ƒä»¬è¿�èµ·æ�¥ã€‚
    2. Remove Short: å¦‚æ�œåŠ¨ä½œæŒ�ç»­æ—¶é—´å¤ªçŸ­ (<5å¸§)ï¼Œè§†ä¸ºå™ªå£°åˆ æ�‰ã€‚
    """
    if df.empty: return df
    
    # æŒ‰è§†é¢‘ã€�è§’è‰²æ�’åº�
    df = df.sort_values(['video_id', 'agent_id', 'target_id', 'action', 'start_frame'])
    refined_rows = []
    
    # åˆ†ç»„å¤„ç�†
    for _, group in df.groupby(['video_id', 'agent_id', 'target_id', 'action']):
        group = group.to_dict('records')
        if not group: continue
        
        current = group[0]
        
        for i in range(1, len(group)):
            next_event = group[i]
            # è®¡ç®—é—´éš™
            gap = next_event['start_frame'] - current['stop_frame']
            
            # å¦‚æ�œé—´éš™å¾ˆå°�ï¼Œå�ˆå¹¶
            if gap <= min_gap:
                current['stop_frame'] = max(current['stop_frame'], next_event['stop_frame'])
            else:
                # é—´éš™å¤ªå¤§ï¼Œå…ˆä¿�å­˜å½“å‰�çš„(å¦‚æ�œå¤Ÿé•¿)ï¼Œç„¶å��å¼€å�¯æ–°çš„
                if (current['stop_frame'] - current['start_frame']) >= min_duration:
                    refined_rows.append(current)
                current = next_event
        
        # ä¿�å­˜æœ€å��ä¸€ä¸ª
        if (current['stop_frame'] - current['start_frame']) >= min_duration:
            refined_rows.append(current)
            
    return pd.DataFrame(refined_rows)

def robustify(submission, dataset, traintest, traintest_directory=None):
    """
    æ��äº¤æ–‡ä»¶æ¸…æ´—ä¸�å…œåº•å‡½æ•° (é˜²æ­¢æ��äº¤å¤±è´¥çš„å…³é”®)
    """
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    
    # 1. åŸºç¡€æ¸…æ´—
    submission['start_frame'] = pd.to_numeric(submission['start_frame'], errors='coerce').fillna(0).astype(int)
    submission['stop_frame'] = pd.to_numeric(submission['stop_frame'], errors='coerce').fillna(0).astype(int)
    submission = submission[submission.start_frame < submission.stop_frame]

    # 2. å�»é‡� (Resolve Overlaps)
    # ç®€å�•ç­–ç•¥ï¼šå��æ�¥çš„åŠ¨ä½œè¦†ç›–å‰�é�¢çš„åŠ¨ä½œï¼Œæˆ–è€…ä¿�ç•™è¾ƒé•¿çš„ã€‚
    # è¿™é‡Œä½¿ç”¨ï¼šæŒ‰æ—¶é—´æ�’åº�ï¼Œæˆªæ–­é‡�å� éƒ¨åˆ†
    group_list = []
    if not submission.empty:
        for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
            group = group.sort_values('start_frame')
            keep_mask = np.ones(len(group), dtype=bool)
            last_stop = -1
            
            for i, (_, row) in enumerate(group.iterrows()):
                if row['start_frame'] < last_stop:
                    # å�‘ç”Ÿé‡�å� ï¼Œç®€å�•ä¸¢å¼ƒå��ä¸€ä¸ª (æˆ–è€…ä½ å�¯ä»¥å†™æ›´å¤�æ�‚çš„é€»è¾‘)
                    keep_mask[i] = False
                else:
                    last_stop = row['stop_frame']
            group_list.append(group[keep_mask])
        submission = pd.concat(group_list) if group_list else submission

    # 3. å…œåº•å¡«å…… (Dummy Filling) - æœ€é‡�è¦�çš„ä¸€æ­¥ï¼�
    # æ£€æŸ¥æ˜¯å�¦æœ‰è§†é¢‘è¢«é�—æ¼�äº†ã€‚å¦‚æ�œæœ‰ï¼Œå¿…é¡»å¡«å�‡æ•°æ�®ï¼Œå�¦åˆ™ Kaggle ä¼šæŠ¥é”™ã€‚
    predicted_videos = set(submission.video_id.unique()) if not submission.empty else set()
    dummy_rows = []
    
    for _, row in dataset.iterrows():
        vid = row.video_id
        # å¦‚æ�œè¿™ä¸ªè§†é¢‘å·²ç»�æœ‰é¢„æµ‹äº†ï¼Œæˆ–è€…æ˜¯ä¸�éœ€è¦�é¢„æµ‹çš„ MABe22 æ•°æ�®ï¼Œè·³è¿‡
        if vid in predicted_videos or str(row.lab_id).startswith('MABe22'):
            continue
            
        # å¦‚æ�œæ˜¯æ¼�æ�‰çš„è§†é¢‘ï¼Œç”Ÿæˆ�ä¸€ä¸ª "other" åŠ¨ä½œå� ä½�
        # éœ€è¦�è§£æ��è¯¥è§†é¢‘æœ‰å“ªäº›è€�é¼ 
        try:
            b_labeled = json.loads(row.behaviors_labeled)
            # ç®€å�•è§£æ��å‡ºç¬¬ä¸€å¯¹ agent-target
            first_pair = b_labeled[0].replace("'", "").split(',')[:2] # [agent, target]
            agent, target = first_pair[0], first_pair[1]
            
            dummy_rows.append({
                'video_id': vid,
                'agent_id': agent,
                'target_id': target,
                'action': 'other', # æˆ–è€… 'investigation' ç­‰å¸¸è§�åŠ¨ä½œ
                'start_frame': 0,
                'stop_frame': 100
            })
        except:
            # å®�åœ¨è§£æ��ä¸�äº†ï¼Œç¡¬å¡«
            dummy_rows.append({
                'video_id': vid,
                'agent_id': 'mouse1',
                'target_id': 'mouse2',
                'action': 'other',
                'start_frame': 0,
                'stop_frame': 100
            })
            
    if dummy_rows:
        print(f"Robustify: Filled {len(dummy_rows)} missing videos with dummy predictions.")
        submission = pd.concat([submission, pd.DataFrame(dummy_rows)], ignore_index=True)
        
    return submission.reset_index(drop=True)

print("å��å¤„ç�†å·¥å…· (Robustify & Smoothing) å®šä¹‰å®Œæˆ�ã€‚")


# ================= Step 3: å·¥å…·å‡½æ•° (å�«é«˜çº§ç‰¹å¾�è®¡ç®—) =================

def _fps_from_meta(meta, fps_lookup=None, default_fps=30.0):
    if fps_lookup is not None:
        vid = meta.video_id.iloc[0] if isinstance(meta, pd.DataFrame) else meta['video_id']
        return fps_lookup.get(vid, default_fps)
    if 'frames_per_second' in meta.columns:
        return meta.frames_per_second.iloc[0]
    return default_fps

def _scale(window, fps):
    return int(max(1, window * fps / 30.0))

def _triangle_area(p1, p2, p3):
    return 0.5 * np.abs(p1['x']*(p2['y']-p3['y']) + p2['x']*(p3['y']-p1['y']) + p3['x']*(p1['y']-p2['y']))

def _angle_between(v1, v2):
    dot = v1['x']*v2['x'] + v1['y']*v2['y']
    norm = np.sqrt(v1['x']**2 + v1['y']**2) * np.sqrt(v2['x']**2 + v2['y']**2)
    return np.arccos(np.clip(dot/(norm+1e-6), -1.0, 1.0))

# --- é«˜çº§ç‰¹å¾�è®¡ç®— ---
def add_frequency_features(X, cx, cy, fps):
    """éœ‡åŠ¨ç‰¹å¾� (Jerk)"""
    vx = cx.diff().fillna(0); vy = cy.diff().fillna(0)
    ax = vx.diff().fillna(0); ay = vy.diff().fillna(0)
    jx = ax.diff().fillna(0); jy = ay.diff().fillna(0)
    jerk_mag = np.sqrt(jx**2 + jy**2) * (fps**3) 
    for w in [10, 30]:
        ws = _scale(w, fps)
        X[f'jerk_mean_{w}'] = jerk_mag.rolling(ws, min_periods=1).mean()
        X[f'jerk_std_{w}']  = jerk_mag.rolling(ws, min_periods=1).std()
    return X

def add_tortuosity_features(X, cx, cy, fps):
    """æ›²æŠ˜åº¦ç‰¹å¾�"""
    step_len = np.sqrt(cx.diff()**2 + cy.diff()**2).fillna(0)
    for w in [30, 90]:
        ws = _scale(w, fps)
        path_len = step_len.rolling(ws, min_periods=1).sum()
        dx = cx - cx.shift(ws).fillna(method='bfill')
        dy = cy - cy.shift(ws).fillna(method='bfill')
        disp = np.sqrt(dx**2 + dy**2)
        X[f'tortuosity_{w}'] = path_len / (disp + 0.1)
    return X

def add_wall_features(X, cx, cy):
    """è´´å¢™ç‰¹å¾�"""
    min_x, max_x = cx.min(), cx.max()
    min_y, max_y = cy.min(), cy.max()
    dist_l = cx - min_x; dist_r = max_x - cx
    dist_t = cy - min_y; dist_b = max_y - cy
    X['dist_to_wall'] = np.minimum.reduce([dist_l, dist_r, dist_t, dist_b])
    X['is_corner'] = (((dist_l < 5) | (dist_r < 5)) & ((dist_t < 5) | (dist_b < 5))).astype(float)
    return X

def add_relative_motion_state(X, mouse_pair, fps):
    """ç›¸å¯¹è¿�åŠ¨çŠ¶æ€�"""
    vA = np.sqrt(mouse_pair['A']['body_center'].diff()['x']**2 + mouse_pair['A']['body_center'].diff()['y']**2) * fps
    vB = np.sqrt(mouse_pair['B']['body_center'].diff()['x']**2 + mouse_pair['B']['body_center'].diff()['y']**2) * fps
    move_th = 2.0
    state = np.zeros(len(X))
    state += 1 * ((vA > move_th) & (vB <= move_th))
    state += 2 * ((vA <= move_th) & (vB > move_th))
    state += 3 * ((vA > move_th) & (vB > move_th))
    X['motion_state_code'] = state
    return X

def add_arena_features(X, cx, cy, arena_dims=None):
    if arena_dims is None or np.isnan(arena_dims).any(): return X
    w, h = arena_dims
    X['dist_center'] = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
    X['dist_wall'] = np.minimum(np.minimum(cx, w - cx), np.minimum(cy, h - cy))
    return X

# å…¼å®¹æ—§å‡½æ•°
def add_curvature_features(X, cx, cy, fps):
    vx = cx.diff().fillna(0); vy = cy.diff().fillna(0)
    ax = vx.diff().fillna(0); ay = vy.diff().fillna(0)
    curv = np.abs(vx*ay - vy*ax) / ((vx**2 + vy**2)**1.5 + 1e-6)
    for w in [25, 50]:
        ws = _scale(w, fps)
        X[f'curv_mean_{w}'] = curv.rolling(ws, min_periods=1).mean()
    return X

def add_multiscale_features(X, cx, cy, fps):
    speed = np.sqrt(cx.diff()**2 + cy.diff()**2) * fps
    for scale in [20, 60]:
        ws = _scale(scale, fps)
        X[f'sp_m{scale}'] = speed.rolling(ws, min_periods=1).mean()
        X[f'sp_s{scale}'] = speed.rolling(ws, min_periods=1).std()
    return X

# Robustify
def robustify(submission, dataset, traintest, traintest_directory=None):
    # (æ­¤å¤„çœ�ç•¥é•¿ä»£ç �ï¼Œä½¿ç”¨ä½ ä¹‹å‰�æ��ä¾›çš„ Robustify å�³å�¯ï¼Œæˆ–è€…ç›´æ�¥ç”¨æˆ‘ä¹‹å‰�ç»™ä½ çš„ç‰ˆæœ¬)
    # ä¸ºäº†å®Œæ•´æ€§å»ºè®®ç›´æ�¥å¤�åˆ¶ä¹‹å‰�çš„ robustify å‡½æ•°å®šä¹‰
    pass 
    
print("Step 3: å·¥å…·å‡½æ•°å®šä¹‰å®Œæˆ�ã€‚")


# ================= Step 4: æ•°æ�®ç”Ÿæˆ�å™¨ (å¸¦ç›‘æ�§ç‰ˆ) =================
# åœ¨å�Ÿæœ‰åŸºç¡€ä¸Šå¢�åŠ äº† verbose å�‚æ•°ï¼Œç”¨äº�æ‰“å�°è°ƒè¯•ä¿¡æ�¯

def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True, verbose=False):
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
        
    for idx, row in dataset.iterrows():
        if not isinstance(row.behaviors_labeled, str): continue
            
        lab_id = row.lab_id
        video_id = row.video_id
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        if not os.path.exists(path): continue
            
        try:
            vid = pd.read_parquet(path)
            # [ç›‘æ�§] æ‰“å�°å�Ÿå§‹æ•°æ�®å½¢çŠ¶
            if verbose and idx == 0: 
                print(f"    [Debug] å�Ÿå§‹ Parquet è¯»å�–æˆ�åŠŸ. Shape: {vid.shape}")
        except: continue

        if 'drop_body_parts' in globals() and len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
            
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        del vid; gc.collect()
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        if hasattr(row, 'pix_per_cm_approx') and row.pix_per_cm_approx > 0:
            pvid /= row.pix_per_cm_approx
            
        # [ç›‘æ�§] æ‰“å�°é¢„å¤„ç�†å��å½¢çŠ¶
        if verbose and idx == 0:
            print(f"    [Debug] é¢„å¤„ç�†/å½’ä¸€åŒ–å®Œæˆ�. Shape: {pvid.shape}")

        try:
            vid_behaviors = json.loads(row.behaviors_labeled)
            vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
            vid_behaviors = [b.split(',') for b in vid_behaviors]
            vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
        except: continue
        
        annot = None
        if traintest == 'train':
            annot_path = path.replace(f'{traintest}_tracking', 'train_annotation')
            if os.path.exists(annot_path):
                try: annot = pd.read_parquet(annot_path)
                except: continue
            else: continue

        if generate_single:
            subset = vid_behaviors.query("target == 'self'")
            unique_agents = np.unique(subset.agent)
            for agent_str in unique_agents:
                try:
                    mouse_id = int(agent_str.replace('mouse', '')) if 'mouse' in agent_str else int(agent_str)
                    if mouse_id in pvid.columns.get_level_values(0):
                        single_mouse = pvid.loc[:, mouse_id]
                        meta = pd.DataFrame({'video_id': video_id, 'agent_id': agent_str, 'target_id': 'self', 'video_frame': single_mouse.index, 'frames_per_second': row.frames_per_second})
                        possible_actions = np.unique(subset.query("agent == @agent_str").action)
                        if traintest == 'train':
                            y = pd.DataFrame(0, index=single_mouse.index, columns=possible_actions)
                            if annot is not None:
                                agent_annot = annot.query("agent_id == @mouse_id and target_id == @mouse_id")
                                for _, r in agent_annot.iterrows():
                                    if r['action'] in y.columns: y.loc[r['start_frame']:r['stop_frame'], r['action']] = 1
                            yield 'single', single_mouse, meta, y
                        else:
                            yield 'single', single_mouse, meta, possible_actions
                except: pass

        if generate_pair:
            subset = vid_behaviors.query("target != 'self'")
            if len(subset) > 0:
                all_mice = np.unique(pvid.columns.get_level_values(0))
                for m1, m2 in itertools.permutations(all_mice, 2):
                    agent_str = f"mouse{m1}"; target_str = f"mouse{m2}"
                    pair_actions = subset.query("agent == @agent_str and target == @target_str")
                    if len(pair_actions) == 0: continue
                    mouse_pair = pd.concat([pvid[m1], pvid[m2]], axis=1, keys=['A', 'B'])
                    meta = pd.DataFrame({'video_id': video_id, 'agent_id': agent_str, 'target_id': target_str, 'video_frame': mouse_pair.index, 'frames_per_second': row.frames_per_second})
                    possible_actions = np.unique(pair_actions.action)
                    if traintest == 'train':
                        y = pd.DataFrame(0, index=mouse_pair.index, columns=possible_actions)
                        if annot is not None:
                            pair_annot = annot.query("agent_id == @m1 and target_id == @m2")
                            for _, r in pair_annot.iterrows():
                                if r['action'] in y.columns: y.loc[r['start_frame']:r['stop_frame'], r['action']] = 1
                        yield 'pair', mouse_pair, meta, y
                    else:
                        yield 'pair', mouse_pair, meta, possible_actions

print("Step 4: æ•°æ�®ç”Ÿæˆ�å™¨å®šä¹‰å®Œæˆ�ã€‚")


# ================= Step 5: ç‰¹å¾�å·¥ç¨‹ (ç»´åº¦å¯¹é½�) =================
def transform_single_base(single_mouse, body_parts_tracked, fps, arena_dims=None):
    available_body_parts = single_mouse.columns.get_level_values(0)
    X = pd.DataFrame({
        f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1)
        for p1, p2 in itertools.combinations(body_parts_tracked, 2)
        if p1 in available_body_parts and p2 in available_body_parts
    })
    
    if all(p in available_body_parts for p in ['nose', 'ear_left', 'ear_right']):
        X['head_area'] = _triangle_area(single_mouse['nose'], single_mouse['ear_left'], single_mouse['ear_right'])
    if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
        v1 = single_mouse['nose'] - single_mouse['body_center']
        v2 = single_mouse['tail_base'] - single_mouse['body_center']
        X['spine_angle'] = _angle_between(v1, v2)

    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']; cy = single_mouse['body_center']['y']
        X = add_arena_features(X, cx, cy, arena_dims)
        for w in [5, 15, 30]:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)
            X[f'cx_m{w}'] = cx.rolling(ws, **roll).mean()
            X[f'cy_m{w}'] = cy.rolling(ws, **roll).mean()
            vx = cx.diff().fillna(0); vy = cy.diff().fillna(0)
            speed = np.sqrt(vx**2 + vy**2) * fps
            X[f'speed_mean_{w}'] = speed.rolling(ws, **roll).mean()
            acc_mag = np.sqrt(vx.diff()**2 + vy.diff()**2).fillna(0)
            X[f'jitter_{w}'] = acc_mag.rolling(ws, **roll).mean()
            X[f'rest_ratio_{w}'] = (speed < 1.0).astype(float).rolling(ws, **roll).mean()
        X = add_curvature_features(X, cx, cy, fps)
        X = add_multiscale_features(X, cx, cy, fps)
    return X.fillna(0).replace([np.inf, -np.inf], 0)

def transform_single(single_mouse, body_parts_tracked, fps, arena_dims=None):
    X = transform_single_base(single_mouse, body_parts_tracked, fps, arena_dims)
    if 'body_center' in single_mouse.columns.get_level_values(0):
        cx = single_mouse['body_center']['x']; cy = single_mouse['body_center']['y']
        X = add_frequency_features(X, cx, cy, fps) # +4
        X = add_tortuosity_features(X, cx, cy, fps)# +2
        X = add_wall_features(X, cx, cy)           # +2
    return X.fillna(0).replace([np.inf, -np.inf], 0)

def transform_pair(mouse_pair, body_parts_tracked, fps):
    X_A = transform_single_base(mouse_pair['A'], body_parts_tracked, fps)
    X_B = transform_single_base(mouse_pair['B'], body_parts_tracked, fps)
    X_A.columns = [f"A_{c}" for c in X_A.columns]
    X_B.columns = [f"B_{c}" for c in X_B.columns]
    X = pd.concat([X_A, X_B], axis=1)
    
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)

    for p1, p2 in itertools.product(body_parts_tracked, repeat=2):
        if p1 in avail_A and p2 in avail_B:
            X[f"12+{p1}+{p2}"] = np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1)

    if all(p in avail_A for p in ['nose', 'body_center']) and all(p in avail_B for p in ['nose', 'body_center']):
        cA = mouse_pair['A']['body_center']; cB = mouse_pair['B']['body_center']
        vec_AB = cB - cA
        dist_AB = np.sqrt(vec_AB['x']**2 + vec_AB['y']**2) + 1e-6
        vec_head_A = mouse_pair['A']['nose'] - cA; vec_head_B = mouse_pair['B']['nose'] - cB
        dot_A_AB = vec_head_A['x'] * vec_AB['x'] + vec_head_A['y'] * vec_AB['y']
        norm_head_A = np.sqrt(vec_head_A['x']**2 + vec_head_A['y']**2) + 1e-6
        X['A_facing_B'] = dot_A_AB / (norm_head_A * dist_AB)
        dot_heads = vec_head_A['x'] * vec_head_B['x'] + vec_head_A['y'] * vec_head_B['y']
        norm_head_B = np.sqrt(vec_head_B['x']**2 + vec_head_B['y']**2) + 1e-6
        X['relative_orientation'] = dot_heads / (norm_head_A * norm_head_B)

    if 'body_center' in avail_A and 'body_center' in avail_B:
        dist_full = np.sqrt(np.square(mouse_pair['A']['body_center'] - mouse_pair['B']['body_center']).sum(axis=1))
        for w in [10, 30]:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)
            X[f'approach_speed_{w}'] = (-dist_full.diff().fillna(0) * fps).rolling(ws, **roll).mean()
            vA = np.sqrt(mouse_pair['A']['body_center'].diff()['x']**2 + mouse_pair['A']['body_center'].diff()['y']**2)
            vB = np.sqrt(mouse_pair['B']['body_center'].diff()['x']**2 + mouse_pair['B']['body_center'].diff()['y']**2)
            X[f'speed_corr_{w}'] = vA.rolling(ws, **roll).corr(vB)

        X = add_relative_motion_state(X, mouse_pair, fps) # +1
        cx = (mouse_pair['A']['body_center']['x'] + mouse_pair['B']['body_center']['x']) / 2
        cy = (mouse_pair['A']['body_center']['y'] + mouse_pair['B']['body_center']['y']) / 2
        X = add_frequency_features(X, cx, cy, fps) # +4
        X = add_tortuosity_features(X, cx, cy, fps)# +2
        X = add_wall_features(X, cx, cy)           # +2

    return X.fillna(0).replace([np.inf, -np.inf], 0)

print("Step 5: ç‰¹å¾�å·¥ç¨‹å®šä¹‰å®Œæˆ�ã€‚")


# ================= Step 6: å®šä¹‰è®­ç»ƒå‡½æ•° (ç»ˆæ��åº•å±‚é�™éŸ³ç‰ˆ) =================
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import joblib
import os
import sys
import gc
import numpy as np
import warnings

# 1. å±�è”½ Python è­¦å‘Š
warnings.filterwarnings('ignore')

# 2. å®šä¹‰ã€�åº•å±‚ã€‘å¼ºåŠ›æ¶ˆéŸ³å™¨ (ä½¿ç”¨ os.dup2 æ‹¦æˆª C++ è¾“å‡º)
class SuppressAllOutput:
    def __enter__(self):
        # åˆ·æ–°ç¼“å†²åŒºï¼Œé˜²æ­¢ä¹‹å‰�çš„æ—¥å¿—æ²¡æ‰“å�°å‡ºæ�¥å°±è¢«æ��æ–­
        sys.stdout.flush()
        sys.stderr.flush()
        
        # 1. æ‰“å¼€é»‘æ´�
        self.null_fd = os.open(os.devnull, os.O_WRONLY)
        
        # 2. å¤‡ä»½å½“å‰�çš„æ ‡å‡†è¾“å‡º/é”™è¯¯çš„æ–‡ä»¶æ��è¿°ç¬¦ (File Descriptors)
        self.save_stdout_fd = os.dup(1)
        self.save_stderr_fd = os.dup(2)
        
        # 3. å°†æ ‡å‡†è¾“å‡º/é”™è¯¯ é‡�å®šå�‘åˆ°é»‘æ´� (è¿™æ˜¯ OS çº§åˆ«çš„æ“�ä½œ)
        os.dup2(self.null_fd, 1)
        os.dup2(self.null_fd, 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 4. æ�¢å¤�å�Ÿæ�¥çš„æ–‡ä»¶æ��è¿°ç¬¦
        os.dup2(self.save_stdout_fd, 1)
        os.dup2(self.save_stderr_fd, 2)
        
        # 5. å…³é—­ä¸´æ—¶å�¥æŸ„
        os.close(self.null_fd)
        os.close(self.save_stdout_fd)
        os.close(self.save_stderr_fd)

def tune_threshold(y_prob, y_true):
    best_f1 = 0; best_th = 0.3
    for th in np.arange(0.1, 0.65, 0.05):
        pred = (y_prob >= th).astype(int)
        if pred.sum() == 0 and y_true.sum() == 0: f1 = 1.0
        else: f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1: best_f1 = f1; best_th = th
    return best_th

def train_one_stage_classifiers(X, label, meta, section_name):
    print(f"  [Training] Action Classifiers...")
    
    save_dir_base = f"{CFG.model_name}/{section_name}"
    os.makedirs(save_dir_base, exist_ok=True)
    
    thresholds = {}
    pbar = tqdm(label.columns, desc="Progress", leave=False)
    
    for action in pbar:
        pbar.set_description(f"Train: {action}")
        
        # 1. å‡†å¤‡æ•°æ�®
        action_mask = ~label[action].isna().values
        if action_mask.sum() == 0: continue
        y_action = label[action][action_mask].values.astype(int)
        
        if y_action.sum() < 20: 
            thresholds[action] = 1.0; continue
            
        X_action = X[action_mask]
        groups_action = meta.video_id[action_mask]
        
        use_gpu = len(X_action) > 5000 
        model_params = {
            'objective': 'binary', 'metric': 'auc', 'n_estimators': 1000, 
            'learning_rate': 0.04, 'num_leaves': 31, 'max_depth': 8,
            'class_weight': 'balanced', 'reg_alpha': 0.5, 'reg_lambda': 0.5,
            'n_jobs': -1, 'random_state': 42,
            'verbosity': -1,
            'device': 'gpu' if use_gpu else 'cpu'
        }
        
        cv = GroupKFold(n_splits=3)
        oof_action = np.zeros(len(y_action))
        models = []
        
        # æ‰“å�°è¡Œå¤´
        msg = f"    ğŸ‘‰ {action:<12} ["
        print(msg, end="", flush=True)
        
        for fold, (tr_idx, va_idx) in enumerate(cv.split(X_action, y_action, groups_action)):
            print(f"{fold+1}..", end="", flush=True)
            
            try:
                # ã€�å…³é”®ã€‘ä½¿ç”¨åº•å±‚æ¶ˆéŸ³å™¨
                with SuppressAllOutput():
                    model = LGBMClassifier(**model_params)
                    model.fit(
                        X_action.iloc[tr_idx], y_action[tr_idx],
                        eval_set=[(X_action.iloc[va_idx], y_action[va_idx])],
                        eval_metric='auc',
                        callbacks=[early_stopping(30, verbose=False), log_evaluation(0)]
                    )
            except:
                model_params['device'] = 'cpu'
                with SuppressAllOutput():
                    model = LGBMClassifier(**model_params)
                    model.fit(
                        X_action.iloc[tr_idx], y_action[tr_idx],
                        eval_set=[(X_action.iloc[va_idx], y_action[va_idx])],
                        eval_metric='auc',
                        callbacks=[early_stopping(30, verbose=False), log_evaluation(0)]
                    )
            
            oof_action[va_idx] = model.predict_proba(X_action.iloc[va_idx])[:, 1]
            models.append(model)
            
        best_th = tune_threshold(oof_action, y_action)
        thresholds[action] = best_th
        f1 = f1_score(y_action, (oof_action >= best_th).astype(int), zero_division=0)
        
        print(f"] âœ… F1: {f1:.4f} (Pos: {y_action.sum()})")
        
        action_dir = f"{save_dir_base}/{action}"
        os.makedirs(action_dir, exist_ok=True)
        joblib.dump(models, f"{action_dir}/models.pkl")
        
    joblib.dump(thresholds, f"{save_dir_base}/thresholds.pkl")
    return thresholds

print("Step 6: å®šä¹‰å®Œæˆ�")


# # ================= Step 7: è®­ç»ƒä¸»å¾ªç�¯ (ä¿®å¤�å¯¹é½�Bugç‰ˆ) =================
# import gc
# import numpy as np
# import pandas as pd

# print("Step 7: å¼€å§‹ One-Stage è®­ç»ƒ ...")
# total_sections = len(body_parts_tracked_list)

# # èƒŒæ™¯å¸§ä¿�ç•™æ¯”ä¾‹ (10%)
# DOWNSAMPLE_RATIO = 0.1 
# # ç¡¬ä¸Šé™� (é˜²æ­¢çˆ†å†…å­˜)
# MAX_TRAIN_SAMPLES = 1500000 

# for section in range(total_sections):
#     bpt_str = body_parts_tracked_list[section]
#     print(f"\n{'='*10} Section {section+1}/{total_sections} {'='*10}")
    
#     train_subset = train[train.body_parts_tracked == bpt_str]
#     if len(train_subset) == 0: continue
    
#     try:
#         bpt = json.loads(bpt_str)
#         if len(bpt) > 5: bpt = [b for b in bpt if b not in drop_body_parts]
#     except: continue
    
#     _fps_lookup = (train_subset[['video_id', 'frames_per_second']]
#                    .drop_duplicates('video_id').set_index('video_id')['frames_per_second'].to_dict())

#     for mode in ['single', 'pair']:
#         print(f"  >>> Mode: {mode}")
        
#         X_parts = []
#         label_parts = []
#         group_parts = []
        
#         gen = generate_mouse_data(
#             train_subset, 'train', 
#             traintest_directory=CFG.train_tracking_path, 
#             generate_single=(mode=='single'), generate_pair=(mode=='pair')
#         )
        
#         count = 0
#         for switch, data, meta, label in gen:
#             if switch != mode: continue
            
#             try:
#                 # 1. æ��å�–ç‰¹å¾�
#                 fps = _fps_from_meta(meta, _fps_lookup)
#                 if mode == 'single': Xi = transform_single(data, bpt, fps)
#                 else: Xi = transform_pair(data, bpt, fps)
                
#                 # 2. è´Ÿæ ·æœ¬é™�é‡‡æ ·
#                 has_action = label.sum(axis=1) > 0
#                 keep_mask = has_action | (np.random.rand(len(has_action)) < DOWNSAMPLE_RATIO)
#                 if keep_mask.sum() == 0: keep_mask.iloc[0] = True
                
#                 # 3. ã€�å…³é”®ä¿®å¤�ã€‘å…ˆåˆ‡ç‰‡ï¼Œå†� Append (äº‹åŠ¡æ€§æ“�ä½œ)
#                 # è¿™æ ·å¦‚æ�œä»»ä½•ä¸€æ­¥æŠ¥é”™ï¼Œéƒ½ä¸�ä¼šæ±¡æŸ“ X_parts åˆ—è¡¨
#                 X_cut = Xi.loc[keep_mask]
#                 label_cut = label.loc[keep_mask]
                
#                 # å®‰å…¨è�·å�– video_id
#                 # æŸ�äº› Pandas ç‰ˆæœ¬ä¸‹ meta.video_id å�¯èƒ½æ˜¯å±�æ€§ï¼Œç”¨ ['video_id'] æ›´å®‰å…¨
#                 group_cut = meta['video_id'].loc[keep_mask]
                
#                 # å�Œé‡�æ£€æŸ¥é•¿åº¦æ˜¯å�¦ä¸€è‡´
#                 if len(X_cut) != len(group_cut):
#                     raise ValueError("Length mismatch inside loop")

#                 # ä¸€æ¬¡æ€§åŠ å…¥
#                 X_parts.append(X_cut)
#                 label_parts.append(label_cut)
#                 group_parts.append(group_cut)
                
#                 count += 1
#             except Exception as e:
#                 # print(f"Skipped video due to error: {e}")
#                 continue
        
#         if count == 0: continue
        
#         # æ‹¼æ�¥æ•°æ�®
#         X_all = pd.concat(X_parts, axis=0, ignore_index=True)
#         label_all = pd.concat(label_parts, axis=0, ignore_index=True)
#         groups_all = pd.concat(group_parts, axis=0, ignore_index=True)
        
#         # æ�„é€  Meta
#         meta_all = pd.DataFrame({'video_id': groups_all.values})
        
#         print(f"  Raw Shape: {X_all.shape}")
        
#         # 4. ç¡¬ä¸Šé™�æˆªæ–­ (é˜²æ­¢ Section 3 è¿™ç§�è¶…å¤§æ•°æ�®é›†)
#         if len(X_all) > MAX_TRAIN_SAMPLES:
#             print(f"  âš ï¸� æ•°æ�®é‡�è¿‡å¤§ ({len(X_all)}), å¼ºåˆ¶é‡‡æ ·è‡³ {MAX_TRAIN_SAMPLES}...")
#             indices = np.random.choice(len(X_all), MAX_TRAIN_SAMPLES, replace=False)
#             X_all = X_all.iloc[indices].reset_index(drop=True)
#             label_all = label_all.iloc[indices].reset_index(drop=True)
#             meta_all = meta_all.iloc[indices].reset_index(drop=True)
#             print(f"  Capped Shape: {X_all.shape}")

#         # å†…å­˜æ¸…ç�†
#         del X_parts, label_parts, group_parts, groups_all
#         gc.collect()
        
#         # å¼€å§‹è®­ç»ƒ
#         section_name = f"{section}_{mode}"
#         train_one_stage_classifiers(X_all, label_all, meta_all, section_name)
        
#         del X_all, label_all, meta_all
#         gc.collect()
        
# print("\nğŸ�‰ è®­ç»ƒå®Œæˆ�ï¼�")


# # ================= Step 8: æ�¨ç�†ä¸�æ��äº¤ (One-Stage æœ€ç»ˆä¿®æ­£ç‰ˆ) =================
# import joblib
# import gc
# import os
# import numpy as np
# import pandas as pd
# import json
# from tqdm.auto import tqdm

# # é‡�æ–°å®šä¹‰ï¼Œé˜²æ­¢å�˜é‡�ä¸¢å¤±
# drop_body_parts = [
#     'headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
#     'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
#     'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint'
# ]

# # --- 8.1 å��å¤„ç�†å·¥å…·å‡½æ•° ---
# def smooth_predictions(df, min_gap=30, min_duration=5):
#     if df.empty: return df
#     df = df.sort_values(['video_id', 'agent_id', 'target_id', 'action', 'start_frame'])
#     refined_rows = []
    
#     for _, group in df.groupby(['video_id', 'agent_id', 'target_id', 'action']):
#         group = group.to_dict('records')
#         if not group: continue
#         current = group[0]
        
#         for i in range(1, len(group)):
#             next_event = group[i]
#             gap = next_event['start_frame'] - current['stop_frame']
            
#             if gap <= min_gap:
#                 current['stop_frame'] = max(current['stop_frame'], next_event['stop_frame'])
#             else:
#                 if (current['stop_frame'] - current['start_frame']) >= min_duration:
#                     refined_rows.append(current)
#                 current = next_event
#         if (current['stop_frame'] - current['start_frame']) >= min_duration:
#             refined_rows.append(current)
            
#     return pd.DataFrame(refined_rows)

# def robustify(submission, dataset, traintest):
#     submission['start_frame'] = pd.to_numeric(submission['start_frame'], errors='coerce').fillna(0).astype(int)
#     submission['stop_frame'] = pd.to_numeric(submission['stop_frame'], errors='coerce').fillna(0).astype(int)
#     submission = submission[submission.start_frame < submission.stop_frame]
    
#     predicted_videos = set(submission.video_id.unique()) if not submission.empty else set()
#     dummy_rows = []
    
#     for _, row in dataset.iterrows():
#         if row.video_id in predicted_videos or str(row.lab_id).startswith('MABe22'): continue
#         dummy_rows.append({
#             'video_id': row.video_id, 'agent_id': 'mouse1', 'target_id': 'mouse2',
#             'action': 'other', 'start_frame': 0, 'stop_frame': 100
#         })
#     if dummy_rows:
#         submission = pd.concat([submission, pd.DataFrame(dummy_rows)], ignore_index=True)
#     return submission.reset_index(drop=True)

# # --- 8.2 ä¸»æ�¨ç�†å¾ªç�¯ ---
# print("Step 8: å¼€å§‹å…¨é‡�æ�¨ç�† (One-Stage Inference)...")
# MODEL_ROOT = CFG.model_name
# submission_list = []

# print(f"Loading models from: {MODEL_ROOT}")
# models_cache = {}

# for section in range(len(body_parts_tracked_list)):
#     models_cache[section] = {}
#     for mode in ['single', 'pair']:
#         section_name = f"{section}_{mode}"
#         model_dir = f"{MODEL_ROOT}/{section_name}" # <--- ä½ çš„è·¯å¾„æ˜¯å¯¹çš„ï¼Œç›´æ�¥æŒ‡å�‘ section æ–‡ä»¶å¤¹
        
#         try:
#             # æ£€æŸ¥è¿™é‡Œï¼šç›´æ�¥æ‰¾ thresholds.pkl
#             if os.path.exists(f"{model_dir}/thresholds.pkl"):
#                 thresholds = joblib.load(f"{model_dir}/thresholds.pkl")
#                 models_cache[section][mode] = {'models': {}, 'ths': thresholds}
                
#                 for action in thresholds.keys():
#                     # æ£€æŸ¥è¿™é‡Œï¼šç›´æ�¥æ‰¾ action æ–‡ä»¶å¤¹
#                     m_path = f"{model_dir}/{action}/models.pkl"
#                     if os.path.exists(m_path):
#                         models_cache[section][mode]['models'][action] = joblib.load(m_path)
#         except Exception as e:
#             pass

# print(f"Loaded Sections: {[k for k in models_cache.keys() if models_cache[k]]}")

# # é��å�†æµ‹è¯•é›†
# for section in range(len(body_parts_tracked_list)):
#     bpt_str = body_parts_tracked_list[section]
#     test_subset = test[test.body_parts_tracked == bpt_str]
#     if len(test_subset) == 0: continue
#     if section not in models_cache: continue
    
#     print(f"Processing Section {section} ({len(test_subset)} videos)...")
#     try:
#         bpt = json.loads(bpt_str)
#         if len(bpt) > 5: bpt = [b for b in bpt if b not in drop_body_parts]
#     except: continue

#     _fps_lookup = (test_subset[['video_id', 'frames_per_second']]
#                    .drop_duplicates('video_id').set_index('video_id')['frames_per_second'].to_dict())

#     for mode in ['single', 'pair']:
#         if mode not in models_cache[section] or not models_cache[section][mode]['models']:
#             continue
            
#         cache = models_cache[section][mode]
#         models_dict = cache['models']
        
#         gen = generate_mouse_data(test_subset, 'test', traintest_directory=CFG.test_tracking_path, 
#                                   generate_single=(mode=='single'), generate_pair=(mode=='pair'))
        
#         for switch, data, meta, _ in gen:
#             if switch != mode: continue
#             try:
#                 fps = _fps_from_meta(meta, _fps_lookup)
#                 if mode == 'single': X = transform_single(data, bpt, fps)
#                 else: X = transform_pair(data, bpt, fps)
                
#                 # å…¨é‡�æ‰“åˆ†
#                 pred_df = pd.DataFrame(index=meta.video_frame)
#                 for action, models in models_dict.items():
#                     probs = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
#                     pred_df[action] = probs
                
#                 if pred_df.empty:
#                     del X, data, meta; continue

#                 # å†³ç­–
#                 best_actions = pred_df.idxmax(axis=1)
#                 max_probs = pred_df.max(axis=1)
#                 mask = max_probs > 0.15 
#                 chosen_actions = best_actions[mask]
                
#                 if chosen_actions.empty:
#                     del X, data, meta, pred_df; continue

#                 # è½¬æ�¢æ ¼å¼�
#                 temp_res = pd.DataFrame({'action': chosen_actions.values, 'frame': chosen_actions.index.values})
#                 temp_res['grp'] = ((temp_res['frame'].diff() > 1) | (temp_res['action'] != temp_res['action'].shift())).cumsum()
                
#                 intervals = temp_res.groupby('grp').agg(
#                     action=('action', 'first'),
#                     start_frame=('frame', 'min'),
#                     stop_frame=('frame', 'max')
#                 )
#                 intervals['stop_frame'] += 1
                
#                 vid = meta.video_id.iloc[0]
#                 aid = meta.agent_id.iloc[0]
#                 tid = meta.target_id.iloc[0]
                
#                 for _, row in intervals.iterrows():
#                     submission_list.append({
#                         'video_id': vid, 'agent_id': aid, 'target_id': tid,
#                         'action': row['action'], 'start_frame': row['start_frame'], 'stop_frame': row['stop_frame']
#                     })
#                 del X, data, meta, pred_df, temp_res, intervals
                
#             except Exception as e:
#                 print(f"Error: {e}"); continue
#     gc.collect()

# # --- 8.3 ç”Ÿæˆ� CSV ---
# print("Raw inference done. Generating CSV...")
# if len(submission_list) > 0:
#     submission_df = pd.DataFrame(submission_list)
#     submission_df = smooth_predictions(submission_df, min_gap=30, min_duration=5)
# else:
#     submission_df = pd.DataFrame(columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])

# final_submission = robustify(submission_df, test, 'test')

# # ã€�ä¿®æ”¹å»ºè®®ã€‘: index=False æ˜¯æœ€ç¨³å¦¥çš„
# final_submission.to_csv('submission.csv', index=False)
# print(f"SUCCESS: submission.csv generated with {len(final_submission)} rows.")
# print(final_submission.head())


# ================= Step 7: è®­ç»ƒä¸»å¾ªç�¯ (å…¨é‡� & å†…å­˜å®‰å…¨) =================
import gc
import numpy as np
import pandas as pd
import json

# è®¾ä¸º None è¡¨ç¤ºè·‘æ‰€æœ‰ Section (å…¨é‡�æ¨¡å¼�)
TEST_SECTION_IDX = 1 

print("Step 7: å¼€å§‹å…¨é‡� One-Stage è®­ç»ƒ...")

# ã€�æ ¸å¿ƒé…�ç½®ã€‘
# 1. ä¸¢å¼ƒ 90% çš„èƒŒæ™¯å¸§ï¼Œä¿�ç•™åŠ¨ä½œå¸§
DOWNSAMPLE_RATIO = 0.1 
# 2. ç¡¬ä¸Šé™�ï¼šé˜²æ­¢å†…å­˜æº¢å‡ºï¼Œæ¯�ä¸ª Section æœ€å¤šç»ƒ 100 ä¸‡æ�¡
MAX_TRAIN_SAMPLES = 1000000 

total_sections = len(body_parts_tracked_list)

for section in range(total_sections):
    # å¦‚æ�œè®¾ç½®äº†æµ‹è¯•ç´¢å¼•ï¼Œè·³è¿‡å…¶ä»–
    if TEST_SECTION_IDX is not None and section != TEST_SECTION_IDX: continue
    
    bpt_str = body_parts_tracked_list[section]
    print(f"\n{'='*10} Section {section+1}/{total_sections} {'='*10}")
    
    train_subset = train[train.body_parts_tracked == bpt_str]
    if len(train_subset) == 0: continue
    
    try:
        bpt = json.loads(bpt_str)
        if len(bpt) > 5: bpt = [b for b in bpt if b not in drop_body_parts]
    except: continue
    
    _fps_lookup = (train_subset[['video_id', 'frames_per_second']]
                   .drop_duplicates('video_id').set_index('video_id')['frames_per_second'].to_dict())

    for mode in ['single', 'pair']:
        # å®¹å™¨æ¸…ç©º
        X_parts, label_parts, group_parts = [], [], []
        
        gen = generate_mouse_data(
            train_subset, 'train', 
            traintest_directory=CFG.train_tracking_path, 
            generate_single=(mode=='single'), generate_pair=(mode=='pair')
        )
        
        count = 0
        for switch, data, meta, label in gen:
            if switch != mode: continue
            try:
                fps = _fps_from_meta(meta, _fps_lookup)
                
                # ç‰¹å¾�æ��å�–
                if mode == 'single': Xi = transform_single(data, bpt, fps)
                else: Xi = transform_pair(data, bpt, fps)
                
                # é™�é‡‡æ · (ä¿�ç•™åŠ¨ä½œ + 10%èƒŒæ™¯)
                has_action = label.sum(axis=1) > 0
                keep_mask = has_action | (np.random.rand(len(has_action)) < DOWNSAMPLE_RATIO)
                if keep_mask.sum() == 0: keep_mask.iloc[0] = True
                
                # æˆªå�–ä¸�å¯¹é½�æ£€æŸ¥
                X_cut = Xi.loc[keep_mask]
                label_cut = label.loc[keep_mask]
                group_cut = meta['video_id'].loc[keep_mask]
                
                if len(X_cut) != len(group_cut): raise ValueError("Mismatch")

                X_parts.append(X_cut)
                label_parts.append(label_cut)
                group_parts.append(group_cut)
                count += 1
            except: continue
            
        if count == 0: continue
        
        # æ•°æ�®æ‹¼æ�¥
        X_all = pd.concat(X_parts, axis=0, ignore_index=True)
        label_all = pd.concat(label_parts, axis=0, ignore_index=True)
        groups_all = pd.concat(group_parts, axis=0, ignore_index=True)
        meta_all = pd.DataFrame({'video_id': groups_all.values})
        
        print(f"  Mode {mode}: Raw Shape {X_all.shape}")
        
        # ã€�å†…å­˜ä¿�æŠ¤ã€‘å¼ºåˆ¶æˆªæ–­
        if len(X_all) > MAX_TRAIN_SAMPLES:
            print(f"  âš ï¸� æ•°æ�®é‡�è¿‡å¤§ï¼Œéš�æœºé‡‡æ ·è‡³ {MAX_TRAIN_SAMPLES}...")
            indices = np.random.choice(len(X_all), MAX_TRAIN_SAMPLES, replace=False)
            X_all = X_all.iloc[indices].reset_index(drop=True)
            label_all = label_all.iloc[indices].reset_index(drop=True)
            meta_all = meta_all.iloc[indices].reset_index(drop=True)
            
        del X_parts, label_parts, group_parts, groups_all; gc.collect()
        
        # è®­ç»ƒæ¨¡å�‹
        section_name = f"{section}_{mode}"
        train_one_stage_classifiers(X_all, label_all, meta_all, section_name)
        
        del X_all, label_all, meta_all; gc.collect()
        
print("\nğŸ�‰ Step 7 å…¨é‡�è®­ç»ƒå®Œæˆ�ï¼�")


# ================= Step 8: æ�¨ç�†ä¸�æ��äº¤ (One-Stage æœ€ç»ˆå…¨é‡�ç‰ˆ) =================
import joblib
import gc
import os
import numpy as np
import pandas as pd
import json
from tqdm.auto import tqdm

# 1. é‡�æ–°å®šä¹‰é…�ç½® (é˜²æ­¢é‡�å�¯å��å�˜é‡�ä¸¢å¤±)
drop_body_parts = [
    'headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
    'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
    'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint'
]

# --- 8.1 å��å¤„ç�†å·¥å…·å‡½æ•° ---
def smooth_predictions(df, min_gap=30, min_duration=5):
    if df.empty: return df
    df = df.sort_values(['video_id', 'agent_id', 'target_id', 'action', 'start_frame'])
    refined_rows = []
    
    for _, group in df.groupby(['video_id', 'agent_id', 'target_id', 'action']):
        group = group.to_dict('records')
        if not group: continue
        current = group[0]
        
        for i in range(1, len(group)):
            next_event = group[i]
            gap = next_event['start_frame'] - current['stop_frame']
            
            if gap <= min_gap:
                current['stop_frame'] = max(current['stop_frame'], next_event['stop_frame'])
            else:
                if (current['stop_frame'] - current['start_frame']) >= min_duration:
                    refined_rows.append(current)
                current = next_event
        if (current['stop_frame'] - current['start_frame']) >= min_duration:
            refined_rows.append(current)
    return pd.DataFrame(refined_rows)

def robustify(submission, dataset, traintest):
    submission['start_frame'] = pd.to_numeric(submission['start_frame'], errors='coerce').fillna(0).astype(int)
    submission['stop_frame'] = pd.to_numeric(submission['stop_frame'], errors='coerce').fillna(0).astype(int)
    submission = submission[submission.start_frame < submission.stop_frame]
    
    predicted_videos = set(submission.video_id.unique()) if not submission.empty else set()
    dummy_rows = []
    
    for _, row in dataset.iterrows():
        if row.video_id in predicted_videos or str(row.lab_id).startswith('MABe22'): continue
        dummy_rows.append({
            'video_id': row.video_id, 'agent_id': 'mouse1', 'target_id': 'mouse2',
            'action': 'other', 'start_frame': 0, 'stop_frame': 100
        })
    if dummy_rows:
        print(f"Robustify: Filled {len(dummy_rows)} missing videos.")
        submission = pd.concat([submission, pd.DataFrame(dummy_rows)], ignore_index=True)
    return submission.reset_index(drop=True)

# --- 8.2 ä¸»æ�¨ç�†ç¨‹åº� ---
print("Step 8: å¼€å§‹å…¨é‡�æ�¨ç�† (æ­£å¼�æ��äº¤æ¨¡å¼�)...")

# è‡ªåŠ¨å¯»æ‰¾æ¨¡å�‹è·¯å¾„
MODEL_ROOT = CFG.model_name
if os.path.exists('/kaggle/working/model'): MODEL_ROOT = '/kaggle/working/model'
elif not os.path.exists(MODEL_ROOT): MODEL_ROOT = '.'
print(f"Model Root: {os.path.abspath(MODEL_ROOT)}")

submission_list = []
models_cache = {}

# 2. åŠ è½½æ‰€æœ‰å�¯ç”¨æ¨¡å�‹ (æ™ºèƒ½è·¯å¾„å…¼å®¹)
print("Loading all models...")
for section in range(len(body_parts_tracked_list)):
    models_cache[section] = {}
    for mode in ['single', 'pair']:
        section_name = f"{section}_{mode}"
        model_dir = f"{MODEL_ROOT}/{section_name}"
        try:
            # ç­–ç•¥ï¼šè‡ªåŠ¨å¯»æ‰¾ thresholds.pklï¼Œå…¼å®¹ stage2 æ–‡ä»¶å¤¹æˆ–æ ¹ç›®å½•
            th_path = f"{model_dir}/thresholds.pkl"
            
            if os.path.exists(th_path):
                thresholds = joblib.load(th_path)
                models_cache[section][mode] = {'models': {}, 'ths': thresholds}
                base_dir = os.path.dirname(th_path) # è�·å�– thresholds æ‰€åœ¨çš„ç›®å½•
                
                for action in thresholds.keys():
                    m_path = f"{base_dir}/{action}/models.pkl"
                    if os.path.exists(m_path):
                        models_cache[section][mode]['models'][action] = joblib.load(m_path)
        except Exception as e: pass

loaded = [k for k in models_cache.keys() if models_cache[k]]
print(f"âœ… Loaded Sections: {loaded}")

# 3. é��å�†æµ‹è¯•é›† (å…¨é‡�)
for section in range(len(body_parts_tracked_list)):
    bpt_str = body_parts_tracked_list[section]
    test_subset = test[test.body_parts_tracked == bpt_str]
    if len(test_subset) == 0: continue
    
    # å¦‚æ�œè¿™ä¸ª Section æˆ‘ä»¬æ²¡è®­ç»ƒè¿‡ï¼Œå°±è·³è¿‡ï¼Œè®© Robustify å¡«å…œåº•
    if section not in models_cache: continue 
    
    print(f"Processing Section {section} ({len(test_subset)} videos)...")
    try:
        bpt = json.loads(bpt_str)
        if len(bpt) > 5: bpt = [b for b in bpt if b not in drop_body_parts]
    except: continue

    _fps_lookup = (test_subset[['video_id', 'frames_per_second']]
                   .drop_duplicates('video_id').set_index('video_id')['frames_per_second'].to_dict())

    for mode in ['single', 'pair']:
        if mode not in models_cache[section] or not models_cache[section][mode]['models']: continue
        
        cache = models_cache[section][mode]
        models_dict = cache['models']
        
        gen = generate_mouse_data(test_subset, 'test', traintest_directory=CFG.test_tracking_path, 
                                  generate_single=(mode=='single'), generate_pair=(mode=='pair'))
        
        for switch, data, meta, _ in gen:
            if switch != mode: continue
            try:
                fps = _fps_from_meta(meta, _fps_lookup)
                if mode == 'single': X = transform_single(data, bpt, fps)
                else: X = transform_pair(data, bpt, fps)
                
                # æ‰“åˆ†
                pred_df = pd.DataFrame(index=meta.video_frame)
                for action, models in models_dict.items():
                    probs = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
                    pred_df[action] = probs
                if pred_df.empty: del X, data, meta; continue

                # å†³ç­–
                best_actions = pred_df.idxmax(axis=1)
                max_probs = pred_df.max(axis=1)
                mask = max_probs > 0.15 
                chosen_actions = best_actions[mask]
                if chosen_actions.empty: del X, data, meta, pred_df; continue

                # æ ¼å¼�åŒ–
                temp_res = pd.DataFrame({'action': chosen_actions.values, 'frame': chosen_actions.index.values})
                temp_res['grp'] = ((temp_res['frame'].diff() > 1) | (temp_res['action'] != temp_res['action'].shift())).cumsum()
                intervals = temp_res.groupby('grp').agg(
                    action=('action', 'first'), start_frame=('frame', 'min'), stop_frame=('frame', 'max')
                )
                intervals['stop_frame'] += 1
                
                vid = meta.video_id.iloc[0]; aid = meta.agent_id.iloc[0]; tid = meta.target_id.iloc[0]
                for _, row in intervals.iterrows():
                    submission_list.append({
                        'video_id': vid, 'agent_id': aid, 'target_id': tid,
                        'action': row['action'], 'start_frame': row['start_frame'], 'stop_frame': row['stop_frame']
                    })
                del X, data, meta, pred_df, temp_res, intervals
            except: continue
    gc.collect()

# --- 8.3 ç”Ÿæˆ� CSV ---
print("Generating final CSV...")
if len(submission_list) > 0:
    submission_df = pd.DataFrame(submission_list)
    submission_df = smooth_predictions(submission_df, min_gap=30, min_duration=5)
else:
    submission_df = pd.DataFrame(columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])

final_submission = robustify(submission_df, test, 'test')
final_submission.to_csv('submission.csv', index=False)
print(f"ğŸ�‰ DONE! submission.csv created with {len(final_submission)} rows.")
print(final_submission.head())

