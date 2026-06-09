# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import os, sys, gc, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


try:
    import polars as pl
    USE_POLARS = True
    print("Using polars")
except Exception as e:
    USE_POLARS = False
    print("polars not available â€” falling back to pandas")

INPUT_DIR = "/kaggle/input/MABe-mouse-behavior-detection"
print("Input exists?", os.path.exists(INPUT_DIR))
print("Top-level files/folders:")
for p in sorted(os.listdir(INPUT_DIR))[:200]:
    print(" ", p)



from typing import Tuple

def read_tracking_parquet(path: str):
    """Return pandas.DataFrame with columns:
       ['video_frame', 'mouse_id', 'bodypart', 'x', 'y', 'likelihood' (if exists)]"""
    if USE_POLARS:
        df = pl.read_parquet(path)
        df = df.to_pandas()
    else:
        df = pd.read_parquet(path)

    expected_cols = set(df.columns)
    
    print("Loaded", path, "shape:", df.shape)
    print(df.columns.tolist()[:40])
    return df


found = False
for folder in ["train_tracking", "test_tracking"]:
    folder_path = os.path.join(INPUT_DIR, folder)
    if os.path.exists(folder_path):
        
        for lab in sorted(os.listdir(folder_path))[:5]:
            lab_dir = os.path.join(folder_path, lab)
            if os.path.isdir(lab_dir):
                files = [f for f in os.listdir(lab_dir) if f.endswith(".parquet")]
                if len(files) > 0:
                    test_path = os.path.join(lab_dir, files[0])
                    print("Sample parquet:", test_path)
                    df_sample = read_tracking_parquet(test_path)
                    found = True
                    break
    if found: break

if not found:
    print("No parquet found in expected folders â€” check INPUT_DIR contents.")



def pivot_tracking(df):
    
    cols = df.columns.tolist()
    
    frame_col = None
    for c in ['video_frame','frame','frame_id','frameId']:
        if c in df.columns:
            frame_col = c
            break
    assert frame_col is not None, "Cannot find frame column in tracking file"

    
    df = df.rename(columns={frame_col: 'video_frame'})
    
    assert 'mouse_id' in df.columns and 'bodypart' in df.columns and 'x' in df.columns and 'y' in df.columns

    
    p = df.pivot_table(index='video_frame', columns=['mouse_id','bodypart'], values=['x','y'], aggfunc='first')
    
    p = p.sort_index(axis=1, level=[0,1])
    return p

def compute_body_centers(pivot_df):
    """Given pivot df with columns ('x'/'y', mouse_id, bodypart) -> return DataFrame with per-mouse center coords"""
    
    cols = pivot_df.columns
    mice = sorted(set([c[1] for c in cols]))
    centers = pd.DataFrame(index=pivot_df.index)
    for m in mice:
        
        x_cols = [c for c in cols if c[1]==m and c[0]=='x']
        y_cols = [c for c in cols if c[1]==m and c[0]=='y']
        if len(x_cols)==0:
            continue
        centers[f'mouse{m}_cx'] = pivot_df[x_cols].mean(axis=1)
        centers[f'mouse{m}_cy'] = pivot_df[y_cols].mean(axis=1)
    return centers

if 'df_sample' in globals():
    p = pivot_tracking(df_sample)
    centers = compute_body_centers(p)
    print("Pivot shape:", p.shape, "Centers shape:", centers.shape)
    display(centers.head())


import numpy as np
import pandas as pd

def add_motion_features(centers: pd.DataFrame, prefix: str):
    """
    Extract motion-related features for one mouse.
    - Ø§Ù„Ø³Ø±Ø¹Ø©ØŒ Ø§Ù„ØªØ³Ø§Ø±Ø¹ØŒ Ø§Ù„Ø§ØªØ¬Ø§Ù‡ØŒ ÙˆØ§Ù„ØªØºÙŠØ± Ø§Ù„Ø²Ø§ÙˆÙŠ
    """
    cx = centers[f'{prefix}_cx']
    cy = centers[f'{prefix}_cy']

    
    vx = cx.diff().fillna(0.0)
    vy = cy.diff().fillna(0.0)
    speed = np.sqrt(vx**2 + vy**2)

    
    ax = vx.diff().fillna(0.0)
    ay = vy.diff().fillna(0.0)
    accel = np.sqrt(ax**2 + ay**2)


    heading = np.arctan2(vy, vx).fillna(0.0)

    
    heading_unwrapped = np.unwrap(heading.to_numpy())
    ang_diff = np.abs(np.diff(heading_unwrapped, prepend=heading_unwrapped[0]))
    ang_diff = pd.Series(ang_diff, index=centers.index)

    features = pd.DataFrame({
        f'{prefix}_vx': vx,
        f'{prefix}_vy': vy,
        f'{prefix}_speed': speed,
        f'{prefix}_ax': ax,
        f'{prefix}_ay': ay,
        f'{prefix}_accel': accel,
        f'{prefix}_heading': heading,
        f'{prefix}_angdiff': ang_diff
    }, index=centers.index)

    return features


def add_pair_features(centers: pd.DataFrame, a: str, b: str):
    """
    Ø­Ø³Ø§Ø¨ Ø§Ù„Ù…Ø³Ø§Ù�Ø© ÙˆØ§Ù„Ø§ØªØ¬Ø§Ù‡ Ø¨ÙŠÙ† Ù�Ø£Ø±ÙŠÙ† (a Ùˆ b)
    """
    ax = centers[f'{a}_cx']; ay = centers[f'{a}_cy']
    bx = centers[f'{b}_cx']; by = centers[f'{b}_cy']

    rel_x = bx - ax
    rel_y = by - ay
    dist = np.sqrt(rel_x**2 + rel_y**2)
    angle = np.arctan2(rel_y, rel_x)

    pair_df = pd.DataFrame({
        f'{a}_{b}_dx': rel_x,
        f'{a}_{b}_dy': rel_y,
        f'{a}_{b}_dist': dist,
        f'{a}_{b}_angle': angle
    }, index=centers.index)
    return pair_df


# ğŸ”¹ ØªØ¬Ø±Ø¨Ø© Ø¹Ù…Ù„ÙŠØ© Ù„Ùˆ centers Ù…ÙˆØ¬ÙˆØ¯
if 'centers' in globals() and centers.shape[1] >= 4:
    f1 = add_motion_features(centers, 'mouse1')
    f2 = add_motion_features(centers, 'mouse2')
    fp = add_pair_features(centers, 'mouse1', 'mouse2')
    display(pd.concat([f1.iloc[:10], f2.iloc[:10], fp.iloc[:10]], axis=1).head())


def add_rolling_stats(df: pd.DataFrame, windows=[5, 15, 45, 120]):
    """Ø¥Ø¶Ø§Ù�Ø© mean/std/median/percentile Ù„ÙƒÙ„ Ø¹Ù…ÙˆØ¯ Ø¹Ø¯Ø¯ÙŠØ§Ù‹ Ø¹Ø¨Ø± Ù†ÙˆØ§Ù�Ø° Ù…Ø®ØªÙ„Ù�Ø©"""
    out = df.copy()
    for w in windows:
        suffix = f'_{w}'
        out = out.assign(**{
            f'{col}_rmean{suffix}': df[col].rolling(w, min_periods=1, center=True).mean()
            for col in df.columns
        })
        out = out.assign(**{
            f'{col}_rstd{suffix}': df[col].rolling(w, min_periods=1, center=True).std().fillna(0)
            for col in df.columns
        })
    return out

# ØªØ¬Ø±Ø¨Ø© Ø³Ø±ÙŠØ¹Ø©
if 'f1' in globals():
    rolled = add_rolling_stats(f1[['mouse1_speed','mouse1_accel']], windows=[5,15,45])
    display(rolled.head())


import os, gc
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import skew, kurtosis
from scipy.fft import rfft

# âœ… Ø§Ø³ØªØ®Ø¯Ø§Ù… polars Ù„Ùˆ Ù…ÙˆØ¬ÙˆØ¯ Ù„Ù„ØªØ³Ø±ÙŠØ¹
try:
    import polars as pl
    USE_POLARS = True
    print("Using polars")
except:
    USE_POLARS = False
    print("polars not available â€” falling back to pandas")

INPUT_DIR = "/kaggle/input/MABe-mouse-behavior-detection"
print("Input exists?", os.path.exists(INPUT_DIR))
print("Top-level files/folders:")
for p in sorted(os.listdir(INPUT_DIR))[:200]:
    print(" ", p)

# ğŸ”¹ Ù‚Ø±Ø§Ø¡Ø© Ù…Ù„Ù� parquet
def read_tracking_parquet(path: str):
    if USE_POLARS:
        df = pl.read_parquet(path).to_pandas()
    else:
        df = pd.read_parquet(path)
    print("Loaded", path, "shape:", df.shape)
    print(df.columns.tolist()[:40])
    return df

# ğŸ”¹ Ø¥ÙŠØ¬Ø§Ø¯ Ø£ÙˆÙ„ Ù…Ù„Ù� parquet Ù„Ù„ØªØ¬Ø±Ø¨Ø©
found = False
for folder in ["train_tracking", "test_tracking"]:
    folder_path = os.path.join(INPUT_DIR, folder)
    if os.path.exists(folder_path):
        for lab in sorted(os.listdir(folder_path))[:5]:
            lab_dir = os.path.join(folder_path, lab)
            if os.path.isdir(lab_dir):
                files = [f for f in os.listdir(lab_dir) if f.endswith(".parquet")]
                if files:
                    test_path = os.path.join(lab_dir, files[0])
                    print("Sample parquet:", test_path)
                    df_sample = read_tracking_parquet(test_path)
                    found = True
                    break
    if found: break

if not found:
    print("No parquet found in expected folders â€” check INPUT_DIR contents.")

# ğŸ”¹ Pivot Ù„Ù„Ù€ body_center Ù„ÙƒÙ„ Ù�Ø£Ø±
def pivot_tracking(df):
    frame_col = next((c for c in ['video_frame','frame','frame_id','frameId'] if c in df.columns), None)
    assert frame_col is not None, "Cannot find frame column in tracking file"
    df = df.rename(columns={frame_col: 'video_frame'})
    assert 'mouse_id' in df.columns and 'bodypart' in df.columns and 'x' in df.columns and 'y' in df.columns
    p = df.pivot_table(index='video_frame', columns=['mouse_id','bodypart'], values=['x','y'], aggfunc='first')
    p = p.sort_index(axis=1, level=[0,1])
    return p

# ğŸ”¹ Ø­Ø³Ø§Ø¨ Ù…Ø±Ø§ÙƒØ² Ø§Ù„Ù�Ø£Ø±
def compute_body_centers(pivot_df):
    cols = pivot_df.columns
    mice = sorted(set([c[1] for c in cols]))
    centers = pd.DataFrame(index=pivot_df.index)
    for m in mice:
        x_cols = [c for c in cols if c[1]==m and c[0]=='x']
        y_cols = [c for c in cols if c[1]==m and c[0]=='y']
        if not x_cols or not y_cols:
            continue
        centers[f'mouse{m}_cx'] = pivot_df[x_cols].mean(axis=1)
        centers[f'mouse{m}_cy'] = pivot_df[y_cols].mean(axis=1)
    return centers

# ğŸ”¹ Motion features Ù„ÙƒÙ„ Ù�Ø£Ø±
def add_motion_features(centers: pd.DataFrame, prefix: str):
    if f'{prefix}_cx' not in centers.columns or f'{prefix}_cy' not in centers.columns:
        return pd.DataFrame(index=centers.index)  # Ø¥Ø°Ø§ Ù…Ø§ Ù…ÙˆØ¬ÙˆØ¯Ø´ Ø§Ù„Ø¹Ù…ÙˆØ¯
    cx = centers[f'{prefix}_cx']
    cy = centers[f'{prefix}_cy']
    vx = cx.diff().fillna(0.0)
    vy = cy.diff().fillna(0.0)
    speed = np.sqrt(vx**2 + vy**2)
    ax = vx.diff().fillna(0.0)
    ay = vy.diff().fillna(0.0)
    accel = np.sqrt(ax**2 + ay**2)
    heading = np.arctan2(vy, vx).fillna(0.0)
    heading_unwrapped = np.unwrap(heading.to_numpy())
    ang_diff = np.abs(np.diff(heading_unwrapped, prepend=heading_unwrapped[0]))
    ang_diff = pd.Series(ang_diff, index=centers.index)
    features = pd.DataFrame({
        f'{prefix}_vx': vx,
        f'{prefix}_vy': vy,
        f'{prefix}_speed': speed,
        f'{prefix}_ax': ax,
        f'{prefix}_ay': ay,
        f'{prefix}_accel': accel,
        f'{prefix}_heading': heading,
        f'{prefix}_angdiff': ang_diff
    }, index=centers.index)
    return features

# ğŸ”¹ Relative / Pair features
def add_relative_features(centers: pd.DataFrame):
    mice = sorted(set([c.split('_')[0] for c in centers.columns]))
    rel_feats = pd.DataFrame(index=centers.index)
    for i,a in enumerate(mice):
        for j,b in enumerate(mice):
            if i >= j:
                continue
            if f'{a}_cx' not in centers.columns or f'{b}_cx' not in centers.columns:
                continue
            dx = centers[f'{b}_cx'] - centers[f'{a}_cx']
            dy = centers[f'{b}_cy'] - centers[f'{a}_cy']
            dist = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)
            rel_feats[f'{a}_{b}_dist'] = dist
            rel_feats[f'{a}_{b}_angle'] = angle
            rel_feats[f'{a}_{b}_approach_rate'] = dist.diff().fillna(0)
    return rel_feats

# ğŸ”¹ Rolling stats Ù…ØªÙ‚Ø¯Ù…Ø©
def add_advanced_rolling_stats(df: pd.DataFrame, windows=[5,15,45,90]):
    out = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    for w in windows:
        for col in numeric_cols:
            roll = df[col].rolling(w, min_periods=1, center=True)
            out[f'{col}_rmean_{w}'] = roll.mean()
            out[f'{col}_rstd_{w}'] = roll.std().fillna(0)
            out[f'{col}_rmedian_{w}'] = roll.median()
            out[f'{col}_rmin_{w}'] = roll.min()
            out[f'{col}_rmax_{w}'] = roll.max()
            out[f'{col}_rskew_{w}'] = roll.apply(lambda x: skew(x), raw=True)
            out[f'{col}_rkurt_{w}'] = roll.apply(lambda x: kurtosis(x), raw=True)
    return out

# ğŸ”¹ Fourier features
def add_frequency_features(centers: pd.DataFrame, n_freq=5):
    mice = sorted(set([c.split('_')[0] for c in centers.columns]))
    freq_feats = pd.DataFrame(index=centers.index)
    for m in mice:
        if f'{m}_cx' not in centers.columns:
            continue
        vx = centers[f'{m}_cx'].diff().fillna(0)
        vy = centers[f'{m}_cy'].diff().fillna(0)
        speed = np.sqrt(vx**2 + vy**2)
        fft_vals = np.abs(rfft(speed))
        top_n = min(n_freq, len(fft_vals))
        for k in range(top_n):
            col_name = f'{m}_fft_{k}'
            freq_feats[col_name] = fft_vals[k]
    return freq_feats

# ğŸ”¹ Ø§Ø³ØªØ®Ø±Ø§Ø¬ ÙƒÙ„ features Ø§Ù„Ù…ØªÙ‚Ø¯Ù…Ø© (Top 5)
def extract_top5_features(centers: pd.DataFrame):
    all_feats = []
    mice = sorted(set([c.split('_')[0] for c in centers.columns]))
    for m in mice:
        f = add_motion_features(centers, m)
        if not f.empty:
            all_feats.append(f)
    motion_df = pd.concat(all_feats, axis=1)
    pair_df = add_relative_features(centers)
    combined_df = pd.concat([motion_df, pair_df], axis=1)
    rolled_df = add_advanced_rolling_stats(combined_df, windows=[5,15,45,90])
    fft_df = add_frequency_features(centers, n_freq=5)
    final_df = pd.concat([rolled_df, fft_df], axis=1)
    return final_df

# ğŸ”¹ ØªØ¬Ø±Ø¨Ø© Ø¹Ù„Ù‰ Ø£ÙˆÙ„ 500 Ù�Ø±ÙŠÙ…
if 'df_sample' in globals():
    pivoted = pivot_tracking(df_sample)
    centers = compute_body_centers(pivoted)
    top5_features = extract_top5_features(centers.iloc[:500])
    display(top5_features.head())
    print("Shape of final features:", top5_features.shape)


# ğŸ”¹ Behavioral advanced features
def add_behavioral_features(centers: pd.DataFrame, distance_thresh=50, angle_thresh=np.pi/4):
    mice = sorted(set([c.split('_')[0] for c in centers.columns]))
    beh_feats = pd.DataFrame(index=centers.index)
    
    for m in mice:
        if f'{m}_cx' not in centers.columns: continue
        vx = centers[f'{m}_cx'].diff().fillna(0)
        vy = centers[f'{m}_cy'].diff().fillna(0)
        speed = np.sqrt(vx**2 + vy**2)
        accel = np.sqrt(vx.diff().fillna(0)**2 + vy.diff().fillna(0)**2)
        heading = np.arctan2(vy, vx).fillna(0)
        heading_unwrapped = np.unwrap(heading.to_numpy())
        ang_diff = np.abs(np.diff(heading_unwrapped, prepend=heading_unwrapped[0]))

        # 1ï¸�âƒ£ Acceleration peaks (> 95th percentile)
        acc_peak_thresh = np.percentile(accel, 95)
        beh_feats[f'{m}_acc_peak'] = (accel > acc_peak_thresh).astype(int)

        # 2ï¸�âƒ£ Direction changes (> angle_thresh)
        beh_feats[f'{m}_dir_change'] = (ang_diff > angle_thresh).astype(int)

        # 5ï¸�âƒ£ Cumulative distance
        beh_feats[f'{m}_cum_dist'] = speed.cumsum()

    # 3ï¸�âƒ£ Ùˆ 4ï¸�âƒ£ Relative features Ø¨ÙŠÙ† ÙƒÙ„ Ø²ÙˆØ¬
    for i,a in enumerate(mice):
        for j,b in enumerate(mice):
            if i >= j: continue
            if f'{a}_cx' not in centers.columns or f'{b}_cx' not in centers.columns: continue
            dx = centers[f'{b}_cx'] - centers[f'{a}_cx']
            dy = centers[f'{b}_cy'] - centers[f'{a}_cy']
            dist = np.sqrt(dx**2 + dy**2)
            speed_diff = np.abs(np.sqrt(dx.diff().fillna(0)**2 + dy.diff().fillna(0)**2))
            
            beh_feats[f'{a}_{b}_prox_alert'] = (dist < distance_thresh).astype(int)
            beh_feats[f'{a}_{b}_rel_speed'] = speed_diff
            
    return beh_feats

# ğŸ”¹ Ø§Ø³ØªØ®Ø±Ø§Ø¬ ÙƒÙ„ Ø§Ù„Ù€ features Ø§Ù„Ù…ØªÙ‚Ø¯Ù…Ø© + behavioral
def extract_top5_behavioral_features(centers: pd.DataFrame):
    all_feats = []
    mice = sorted(set([c.split('_')[0] for c in centers.columns]))
    for m in mice:
        f = add_motion_features(centers, m)
        if not f.empty: all_feats.append(f)
    motion_df = pd.concat(all_feats, axis=1)
    pair_df = add_relative_features(centers)
    combined_df = pd.concat([motion_df, pair_df], axis=1)
    rolled_df = add_advanced_rolling_stats(combined_df, windows=[5,15,45,90])
    fft_df = add_frequency_features(centers, n_freq=5)
    behavioral_df = add_behavioral_features(centers)
    
    final_df = pd.concat([rolled_df, fft_df, behavioral_df], axis=1)
    return final_df

# ğŸ”¹ ØªØ¬Ø±Ø¨Ø© Ø¹Ù„Ù‰ Ø£ÙˆÙ„ 500 Ù�Ø±ÙŠÙ…
if 'df_sample' in globals():
    pivoted = pivot_tracking(df_sample)
    centers = compute_body_centers(pivoted)
    top5_behavioral_features = extract_top5_behavioral_features(centers.iloc[:500])
    display(top5_behavioral_features.head())
    print("Shape of final behavioral features:", top5_behavioral_features.shape)




import os, gc
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
import lightgbm as lgb
import joblib

# ----------------- CONFIG -----------------
INPUT_DIR = "/kaggle/input/MABe-mouse-behavior-detection"
TRAIN_TRACKING = os.path.join(INPUT_DIR, "train_tracking")
TRAIN_ANNOT = os.path.join(INPUT_DIR, "train_annotation")
CACHE_DIR = "/kaggle/working/features_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

N_FOLDS = 5
RANDOM_STATE = 42
NUM_BOOST_ROUND = 3000
EARLY_STOP = 100
VERBOSE = 100


LIMIT_VIDEOS = None     
LIMIT_FRAMES_PER_VIDEO = None  

def load_annotations():
    """
    Try to read annotations in various expected locations.
    Returns DataFrame with at least: ['video_id','video_frame', <target columns>]
    """
    # try a top-level train.csv first
    train_csv = os.path.join(INPUT_DIR, "train.csv")
    if os.path.exists(train_csv):
        ann = pd.read_csv(train_csv)
        print("Loaded annotations from", train_csv, "shape:", ann.shape)
        return ann

    if os.path.exists(TRAIN_ANNOT):
        rows = []
        for p in sorted(glob.glob(os.path.join(TRAIN_ANNOT, "**", "*.csv"), recursive=True)):
            df = pd.read_csv(p)
            # try to infer video_id column or take parent folder name
            if 'video_id' not in df.columns and 'video' not in df.columns:
                # attempt to derive from filename
                vid = Path(p).stem
                df['video_id'] = vid
            rows.append(df)
        if rows:
            ann = pd.concat(rows, ignore_index=True)
            print("Loaded annotations from", TRAIN_ANNOT, "shape:", ann.shape)
            return ann

    raise FileNotFoundError("No annotation file found; please place train.csv or check train_annotation folder.")


def iter_tracking_files(limit=None):
    paths = []
    for lab in sorted(os.listdir(TRAIN_TRACKING)):
        lab_dir = os.path.join(TRAIN_TRACKING, lab)
        if not os.path.isdir(lab_dir): continue
        files = sorted([os.path.join(lab_dir, f) for f in os.listdir(lab_dir) if f.endswith(".parquet")])
        paths.extend(files)
        if limit and len(paths) >= limit:
            break
    if limit:
        paths = paths[:limit]
    return paths

def process_one_parquet(path_parquet, force_recompute=False, n_frames_limit=None):
    """
    Reads a tracking parquet, computes centers and advanced features,
    returns DataFrame features with columns: video_id, video_frame, <features...>
    Caches to CACHE_DIR by file stem.
    """
    stem = Path(path_parquet).stem
    cache_file = os.path.join(CACHE_DIR, f"{stem}_features.parquet")
    if os.path.exists(cache_file) and not force_recompute:
        df = pd.read_parquet(cache_file)
        return df


    try:
        raw = read_tracking_parquet(path_parquet)  
    except Exception:
        raw = pd.read_parquet(path_parquet)


    if n_frames_limit:
        raw = raw.iloc[:n_frames_limit]


    p = pivot_tracking(raw)  
    centers = compute_body_centers(p)

    centers.index.name = 'video_frame'


    try:
        feats = extract_top5_behavioral_features(centers) 
    except Exception:
        feats = extract_top5_features(centers)


    video_id = Path(path_parquet).parent.name
    feats = feats.reset_index().rename(columns={'index': 'video_frame'}) if 'video_frame' not in feats.columns else feats.reset_index()
    feats['video_id'] = video_id


    feats.to_parquet(cache_file, index=False)
    return feats

def build_feature_table(limit_videos=None, frames_limit=None, force_recompute=False):
    tracking_files = iter_tracking_files(limit=limit_videos)
    dfs = []
    for p in tqdm(tracking_files, desc="Processing tracking files"):
        df_feats = process_one_parquet(p, force_recompute=force_recompute, n_frames_limit=frames_limit)
        dfs.append(df_feats)
    features_all = pd.concat(dfs, ignore_index=True)
    print("Combined features shape:", features_all.shape)


    ann = load_annotations()


    if 'video_frame' not in ann.columns:
        for alt in ['frame', 'frame_id', 'frameId']:
            if alt in ann.columns:
                ann = ann.rename(columns={alt: 'video_frame'})
                break
    if 'video_id' not in ann.columns and 'video' in ann.columns:
        ann = ann.rename(columns={'video': 'video_id'})


    merged = features_all.merge(ann, how='left', on=['video_id', 'video_frame'])
    print("After merge shape:", merged.shape)
    return merged

def prepare_training_data(merged_df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ['video_id', 'video_frame']

    ann_cols = [c for c in merged_df.columns if c not in merged_df.select_dtypes(include=np.number).columns]
    
    ann = load_annotations()
    target_cols = [c for c in ann.columns if c not in ['video_id', 'video_frame']]
    if len(target_cols) == 0:
        
        for name in ['label', 'behavior', 'behavior_id', 'label_id', 'target']:
            if name in merged_df.columns:
                target_cols = [name]
                break

    if len(target_cols) == 0:
        raise ValueError("Cannot find target columns in annotations. Check annotation files and column names.")

    feature_cols = [c for c in merged_df.select_dtypes(include=np.number).columns if c not in (exclude_cols + target_cols)]
    print("Detected target columns:", target_cols)
    print("Feature count:", len(feature_cols))
    return merged_df, feature_cols, target_cols

def train_lgb_groupkfold(df, feature_cols, target_cols, n_folds=N_FOLDS, random_state=RANDOM_STATE):

    if 'video_id' not in df.columns:
        raise ValueError("video_id column is required for GroupKFold")

    groups = df['video_id'].values
    gkf = GroupKFold(n_splits=n_folds)

    oof_preds = {}
    models = []
    importance_df = pd.DataFrame(index=feature_cols)


    for target in target_cols:
        print("Training target:", target)
        y = df[target].values
        oof = np.zeros(len(df))
        fold_importances = pd.DataFrame()
        for fold, (train_idx, val_idx) in enumerate(gkf.split(df, y, groups)):
            print(f"Fold {fold+1}/{n_folds} â€” train {len(train_idx)} val {len(val_idx)}")
            X_train = df.loc[train_idx, feature_cols]
            X_val = df.loc[val_idx, feature_cols]
            y_train = y[train_idx]
            y_val = y[val_idx]

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        
            unique_vals = np.unique(y_train[~pd.isna(y_train)])
            if len(unique_vals) <= 2:
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'verbosity': -1,
                    'boosting_type': 'gbdt',
                    'seed': random_state,
                    'deterministic': True,
                }
            else:
                params = {
                    'objective': 'multiclass',
                    'num_class': int(len(unique_vals)),
                    'metric': 'multi_logloss',
                    'verbosity': -1,
                    'seed': random_state,
                    'deterministic': True,
                }

            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_val],
                verbose_eval=VERBOSE,
                num_boost_round=NUM_BOOST_ROUND,
                early_stopping_rounds=EARLY_STOP
            )

            models.append((target, fold, model))
        
            if params['objective'] == 'binary':
                preds = model.predict(X_val, num_iteration=model.best_iteration)
                oof[val_idx] = preds
            else:
                preds = model.predict(X_val, num_iteration=model.best_iteration)
                
                oof[val_idx] = np.argmax(preds, axis=1)

            # feature importance
            fi = pd.DataFrame()
            fi['feature'] = feature_cols
            fi['importance'] = model.feature_importance(importance_type='gain')
            fi['fold'] = fold
            fi['target'] = target
            fold_importances = pd.concat([fold_importances, fi], ignore_index=True)

            # cleanup
            del X_train, X_val, lgb_train, lgb_val
            gc.collect()

        oof_preds[target] = oof
        importance_df = importance_df.join(
            fold_importances.groupby('feature')['importance'].mean().rename(f'{target}_importance'),
            how='left'
        )

    return oof_preds, models, importance_df

merged = build_feature_table(limit_videos=LIMIT_VIDEOS, frames_limit=LIMIT_FRAMES_PER_VIDEO, force_recompute=False)


merged, feature_cols, target_cols = prepare_training_data(merged)


oof_preds, models, importance_df = train_lgb_groupkfold(merged, feature_cols, target_cols, n_folds=N_FOLDS)


importance_df_mean = importance_df.mean(axis=1).sort_values(ascending=False)
print("Top 50 features by average importance:")
display(importance_df_mean.head(50))


joblib.dump(models, "/kaggle/working/lgb_models.pkl")
for t, preds in oof_preds.items():
    pd.DataFrame({f"{t}_oof": preds}).to_csv(f"/kaggle/working/oof_{t}.csv", index=False)

print("Training complete. Models saved to /kaggle/working/lgb_models.pkl")


import pandas as pd
import numpy as np

submission_rows = []

threshold = 0.5

for target, preds in oof_preds.items():
    df_pred = merged[['video_id', 'video_frame']].copy()
    df_pred['prob'] = preds
    df_pred['target'] = target

    for vid in df_pred['video_id'].unique():
        df_vid = df_pred[df_pred['video_id'] == vid].sort_values('video_frame')
        frames = df_vid['video_frame'].values
        probs = df_vid['prob'].values

        active = probs > threshold
        start_frame = None

        for i, is_active in enumerate(active):
            if is_active and start_frame is None:
                start_frame = frames[i]
            elif not is_active and start_frame is not None:
                stop_frame = frames[i-1]
                submission_rows.append([vid, target, target, target, start_frame, stop_frame])
                start_frame = None

        if start_frame is not None:
            submission_rows.append([vid, target, target, target, start_frame, frames[-1]])

submission_df = pd.DataFrame(submission_rows, columns=['video_id','agent_id','target_id','action','start_frame','stop_frame'])
submission_df.insert(0, 'row_id', np.arange(len(submission_df)))
submission_df.to_csv('/kaggle/working/submission.csv', index=False)

print("Professional submission file created at /kaggle/working/submission.csv")


output.to_csv('submission.csv', index=False)

