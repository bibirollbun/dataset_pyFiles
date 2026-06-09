# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
count = 0
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        count += 1
        if count >= 20:  # stop after printing 20 files
            break
    if count >= 20:
        break

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Kaggle/Colab friendly setup (CPU by default; optional CuPy if available)
import os, sys, math, json, gc, warnings, random, itertools
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def find_kaggle_root() -> str:
    base = "/kaggle/input"
    if not os.path.exists(base):
        return os.getcwd()
    # Prefer dirs containing test.csv
    for item in os.listdir(base):
        p = os.path.join(base, item)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "test.csv")):
            return p
        if os.path.isdir(p):
            for sub in os.listdir(p):
                sp = os.path.join(p, sub)
                if os.path.isdir(sp) and os.path.exists(os.path.join(sp, "test.csv")):
                    return sp
    # Fallback
    for item in os.listdir(base):
        p = os.path.join(base, item)
        if os.path.isdir(p):
            return p
    return base

ROOT = find_kaggle_root()
WORK = "/kaggle/working" if os.path.exists("/kaggle/working") else os.getcwd()
print("ROOT:", ROOT)
print("WORK:", WORK)

# Optional GPU accel for distance via CuPy (set True to try; will fallback)
USE_GPU = True
try:
    import cupy as cp  # type: ignore
    HAS_CUPY = True
    print('Gpu In use')
except Exception:
    HAS_CUPY = False
    USE_GPU = False


def safe_le(a, b, valid):
    a = np.asarray(a)
    out = np.zeros(a.shape, dtype=bool)
    finite = np.isfinite(a)
    np.less_equal(a, b, out=out, where=(valid & finite))
    return out

def safe_ge(a, b, valid):
    a = np.asarray(a)
    out = np.zeros(a.shape, dtype=bool)
    finite = np.isfinite(a)
    np.greater_equal(a, b, out=out, where=(valid & finite))
    return out

def nan_safe_hypot(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    valid = np.isfinite(dx) & np.isfinite(dy)
    out = np.full(dx.shape, np.nan, dtype=float)
    if USE_GPU and HAS_CUPY:
        try:
            dxg, dyg, valg = cp.asarray(dx), cp.asarray(dy), cp.asarray(valid)
            outg = cp.full(dxg.shape, cp.nan, dtype=cp.float64)
            cp.hypot(dxg, dyg, out=outg, where=valg)
            return cp.asnumpy(outg)
        except Exception:
            pass
    np.hypot(dx, dy, out=out, where=valid)
    return out

def read_metadata(path_csv: str) -> pd.DataFrame:
    if not os.path.exists(path_csv):
        raise FileNotFoundError(path_csv)
    df = pd.read_csv(path_csv)
    rename = {
        "frames per second": "fps",
        "frames_per_second": "fps",
        "pix per cm (approx)": "pix_per_cm",
        "pix_per_cm_approx": "pix_per_cm",
    }
    for k, v in rename.items():
        if k in df.columns: df.rename(columns={k: v}, inplace=True)
    if "fps" not in df.columns:
        for alt in ["frame_rate", "Frames per second"]:
            if alt in df.columns: df.rename(columns={alt: "fps"}, inplace=True)
    if "pix_per_cm" not in df.columns:
        for alt in ["Pixels per cm", "pixels_per_cm"]:
            if alt in df.columns: df.rename(columns={alt: "pix_per_cm"}, inplace=True)
    return df

def load_tracking_parquet(root: str, split: str, lab_id: str, video_id: int) -> pd.DataFrame:
    p = os.path.join(root, f"{split}_tracking", str(lab_id), f"{int(video_id)}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    # Standardize
    ren = {}
    if "video_frame" not in df.columns:
        for alt in ["frame", "frame_idx", "frame_index", "frame_id"]:
            if alt in df.columns: ren[alt] = "video_frame"; break
    if "mouse_id" not in df.columns:
        for alt in ["individual", "track_id", "id", "agent_id"]:
            if alt in df.columns: ren[alt] = "mouse_id"; break
    if "x" not in df.columns:
        for alt in ["x_px", "xpos", "x_pos", "X", "x_coord"]:
            if alt in df.columns: ren[alt] = "x"; break
    if "y" not in df.columns:
        for alt in ["y_px", "ypos", "y_pos", "Y", "y_coord"]:
            if alt in df.columns: ren[alt] = "y"; break
    if ren: df = df.rename(columns=ren)
    if "bodypart" not in df.columns:
        for alt in ["body_part", "node", "keypoint", "part", "BodyPart"]:
            if alt in df.columns: df = df.rename(columns={alt: "bodypart"}); break
    for c in ["video_frame","mouse_id","x","y"]:
        if c not in df.columns: raise ValueError(f"Missing {c} in {p}")
    return df

def bodypart_mask(series: pd.Series, keywords) -> pd.Series:
    lo = series.astype(str).str.lower()
    m = pd.Series(False, index=series.index)
    for kw in keywords:
        m = m | lo.str.contains(kw, na=False)
    return m


def compute_centroids(trk: pd.DataFrame) -> pd.DataFrame:
    cent = (trk.groupby(["video_frame","mouse_id"], as_index=False)[["x","y"]]
              .mean().rename(columns={"x":"centroid_x","y":"centroid_y"}))
    parts = {
        "nose": ["nose","snout","sniff","snout_tip"],
        "head": ["head","ear","neck","head_center"],
        "rear": ["rear","back","rump","hindquarters"],
        "tail": ["tail","tail_base","tail_tip","tail_midpoint"],
        "body": ["body","center","lateral","body_center"],
    }
    if "bodypart" in trk.columns:
        for name, kws in parts.items():
            m = bodypart_mask(trk["bodypart"], kws)
            if m.any():
                sub = (trk.loc[m].groupby(["video_frame","mouse_id"], as_index=False)[["x","y"]]
                         .mean().rename(columns={"x":f"{name}_x","y":f"{name}_y"}))
                cent = cent.merge(sub, on=["video_frame","mouse_id"], how="left")
    return cent

def add_movement(centroids: pd.DataFrame, fps: float, pix_per_cm: float|None) -> pd.DataFrame:
    if not fps or fps<=0 or pix_per_cm is None or pix_per_cm<=0:
        for c in ["speed_cm_s","accel_cm_s2","angular_velocity","curvature","direction_change"]:
            centroids[c] = np.nan
        return centroids
    df = centroids.sort_values(["mouse_id","video_frame"]).copy()
    for mid, sub in df.groupby("mouse_id"):
        x, y = sub["centroid_x"].values, sub["centroid_y"].values
        frames = sub["video_frame"].values
        dx, dy = np.diff(x), np.diff(y)
        dt = np.diff(frames)/fps
        dt = np.where(dt==0, 1.0/fps, dt)
        vx, vy = (dx/pix_per_cm)/dt, (dy/pix_per_cm)/dt
        speed = np.sqrt(vx**2 + vy**2)
        if len(speed)>1:
            speed = np.convolve(speed, np.ones(2)/2.0, mode="same")
        accel = np.diff(speed)*fps
        angles = np.arctan2(vy, vx)
        ang_vel = np.diff(angles)*fps
        curv = np.abs(np.diff(ang_vel))*fps
        dirchg = np.abs(np.diff(angles)); dirchg = np.minimum(dirchg, 2*np.pi - dirchg)

        # pad
        speed_p = np.concatenate([[0], speed])
        accel_p = np.concatenate([[0,0], accel])
        ang_p   = np.concatenate([[0], ang_vel])
        curv_p  = np.concatenate([[0,0], curv])
        dir_p   = np.concatenate([[0], dirchg])

        n = len(sub)
        cols = {
            "speed_cm_s": speed_p[:n],
            "accel_cm_s2": accel_p[:n],
            "angular_velocity": ang_p[:n],
            "curvature": curv_p[:n],
            "direction_change": dir_p[:n],
        }
        for k,v in cols.items():
            df.loc[sub.index, k] = v
    return df


# Quick EDA
meta_train = read_metadata(os.path.join(ROOT, "train.csv"))
meta_test  = read_metadata(os.path.join(ROOT, "test.csv"))

plt.figure(figsize=(10,4))
meta_train["lab_id"].value_counts().head(20).plot(kind="bar")
plt.title("Train videos by lab_id")
plt.tight_layout()
plt.show()

cols = [c for c in ["fps","video_duration_sec","pix_per_cm"] if c in meta_train.columns]
display(meta_train[cols].describe(include="all"))


# Per-lab distribution (train vs test)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
meta_train["lab_id"].value_counts().head(15).plot(kind="bar", ax=axs[0], color="tab:blue", title="Train videos by lab (Top 15)")
meta_test["lab_id"].value_counts().head(15).plot(kind="bar", ax=axs[1], color="tab:green", title="Test videos by lab (Top 15)")
plt.tight_layout(); plt.show()

# fps and pix_per_cm histograms
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
if "fps" in meta_train.columns:
    meta_train["fps"].dropna().plot(kind="hist", bins=30, ax=axs[0], color="tab:purple", alpha=0.8, title="Train FPS distribution")
if "pix_per_cm" in meta_train.columns:
    meta_train["pix_per_cm"].dropna().plot(kind="hist", bins=30, ax=axs[1], color="tab:orange", alpha=0.8, title="Train pix_per_cm distribution")
plt.tight_layout(); plt.show()


import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets import VBox, HBox, Layout
from IPython.display import display, clear_output

# Utility: top-N labs helper
def top_n_labs(df, n=15):
    vc = df["lab_id"].astype(str).value_counts().head(n)
    return df[df["lab_id"].astype(str).isin(vc.index)].copy()

# 1) Interactive bar of videos by lab (train/test toggle)
dataset_toggle = widgets.ToggleButtons(options=["train","test"], value="train", description="Split:")
topn_slider = widgets.IntSlider(value=15, min=5, max=50, step=1, description="Top labs:")

def plot_lab_counts(split="train", topn=15):
    meta = meta_train if split=="train" else meta_test
    df = top_n_labs(meta, topn)
    vc = df["lab_id"].astype(str).value_counts().reset_index()
    # In recent pandas, columns are ['lab_id','count']
    vc.columns = ["lab_id", "count"]  # ensure consistent names
    fig = px.bar(vc, x="lab_id", y="count",
                 labels={"lab_id":"lab_id","count":"# videos"},
                 title=f"Videos by lab_id ({split}, top {topn})")
    fig.update_layout(xaxis_tickangle=45, height=450)
    fig.show()

# 2) Interactive scatter: fps vs pix_per_cm colored by lab
def scatter_fps_ppcm(split="train"):
    meta = meta_train if split=="train" else meta_test
    if "pix_per_cm" not in meta.columns:
        print("pix_per_cm not available in metadata.")
        return
    df = meta.copy()
    df["lab_id"] = df["lab_id"].astype(str)
    fig = px.scatter(df, x="fps", y="pix_per_cm", color="lab_id",
                     hover_data=["video_id"], title=f"fps vs pix_per_cm ({split})")
    fig.update_traces(marker=dict(size=7, opacity=0.7))
    fig.update_layout(height=500, legend_itemsizing="trace")
    fig.show()

widgets.interact(scatter_fps_ppcm, split=dataset_toggle);

# 3) Interactive distribution of video_duration_sec (if present)
def dist_duration(split="train"):
    meta = meta_train if split=="train" else meta_test
    if "video_duration_sec" not in meta.columns:
        print("video_duration_sec not in metadata.")
        return
    fig = px.histogram(meta, x="video_duration_sec", nbins=40,
                       title=f"Video duration distribution (sec) - {split}")
    fig.update_layout(height=400)
    fig.show()

widgets.interact(dist_duration, split=dataset_toggle);


import numpy as np
import pandas as pd

def add_movement(centroids: pd.DataFrame, fps: float, pix_per_cm: float | None) -> pd.DataFrame:
    """
    Adds per-frame kinematics to a centroids dataframe without length mismatches.
    Expects columns:
      - 'frame'
      - 'agent_id' (string or int)
      - position columns: either ('x','y') or ('cx','cy')
    Produces columns:
      - 'speed_cm_s', 'accel_cm_s2', 'angular_velocity', 'curvature', 'direction_change'
    All derived arrays are computed with np.gradient to keep same length as the index.
    """
    if centroids is None or len(centroids) == 0:
        return centroids

    df = centroids.copy()

    # Choose coordinate columns
    if {'x', 'y'}.issubset(df.columns):
        x_col, y_col = 'x', 'y'
    elif {'cx', 'cy'}.issubset(df.columns):
        x_col, y_col = 'cx', 'cy'
    else:
        raise ValueError("add_movement: expected position columns ('x','y') or ('cx','cy').")

    # Ensure dtypes
    df['frame'] = df['frame'].astype(int)
    # Keep agent_id as-is (string or int), just ensure existence
    if 'agent_id' not in df.columns:
        raise ValueError("add_movement: expected 'agent_id' column.")

    # Prepare output columns
    out_cols = ["speed_cm_s", "accel_cm_s2", "angular_velocity", "curvature", "direction_change"]
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan

    eps = 1e-8
    px_to_cm = (1.0 / float(pix_per_cm)) if (pix_per_cm is not None and pix_per_cm > 0) else None

    # Compute per-agent to avoid mixing tracks
    for _, sub_idx in df.groupby('agent_id').groups.items():
        sub = df.loc[sub_idx].sort_values('frame')
        idx = sub.index

        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)

        # Time derivatives via gradient keep the same length as input
        # First spatial derivatives (per frame)
        dx = np.gradient(x)
        dy = np.gradient(y)

        # Distance per frame (in pixels)
        ds_pix = np.hypot(dx, dy)

        # Speed: pixels per second -> convert to cm/s if pix_per_cm provided
        speed_pix_s = ds_pix * float(fps)
        if px_to_cm is not None:
            speed_cm_s = speed_pix_s * px_to_cm
        else:
            # If pixel scale unknown, still provide speed in "pixels per second"
            speed_cm_s = speed_pix_s

        # Acceleration: time derivative of speed
        accel_pix_s2 = np.gradient(speed_pix_s) * float(fps)
        if px_to_cm is not None:
            accel_cm_s2 = accel_pix_s2 * px_to_cm
        else:
            accel_cm_s2 = accel_pix_s2

        # Heading angle and angular velocity
        theta = np.arctan2(dy, dx)  # radians
        dtheta = np.gradient(theta)  # radians per frame (same length)
        # Wrap small numerical jumps
        dtheta_wrapped = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        angular_velocity = dtheta_wrapped * float(fps)  # radians per second

        # Curvature kappa ≈ |dtheta/ds|
        curvature = np.abs(dtheta_wrapped) / (ds_pix + eps)

        # Direction change (per frame, absolute angle change in radians)
        direction_change = np.abs(dtheta_wrapped)

        # Assign back (lengths match sub.index exactly)
        df.loc[idx, "speed_cm_s"] = speed_cm_s
        df.loc[idx, "accel_cm_s2"] = accel_cm_s2
        df.loc[idx, "angular_velocity"] = angular_velocity
        df.loc[idx, "curvature"] = curvature
        df.loc[idx, "direction_change"] = direction_change

    return df


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import Image, display

# ---------------------------
# Helper Functions
# ---------------------------

def quick_explore_parquet(file_path, max_rows=1000):
    """Quick exploration of parquet file structure"""
    try:
        df = pd.read_parquet(file_path).head(max_rows)
        print(f"File: {os.path.basename(file_path)}")
        print(f"Shape (first {max_rows} rows): {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print(f"Sample data:\n{df.head(3)}")
        print("-" * 40)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def animate_mouse_bodyparts(df, mouse_id, output_dir="animations", max_frames=500):
    """Create a body-part movement animation for a given mouse"""
    
    mouse_data = df[df['mouse_id'] == mouse_id].copy()  # copy to avoid SettingWithCopyWarning

    # Clean data
    mouse_data['x'] = mouse_data['x'].replace([np.inf, -np.inf], np.nan)
    mouse_data['y'] = mouse_data['y'].replace([np.inf, -np.inf], np.nan)
    mouse_data.dropna(subset=['x', 'y'], inplace=True)

    # Get coordinate ranges
    x_min, x_max = mouse_data['x'].min(), mouse_data['x'].max()
    y_min, y_max = mouse_data['y'].min(), mouse_data['y'].max()

    # Unique body parts
    bodyparts = mouse_data['bodypart'].unique()
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    if len(bodyparts) > len(colors):
        import matplotlib.colors as mcolors
        colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())[:len(bodyparts)-len(colors)]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(x_min - 10, x_max + 10)
    ax.set_ylim(y_min - 10, y_max + 10)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(f'Mouse {mouse_id} Body Parts Movement Animation')

    # Trajectory lines and points
    lines, points = [], []
    for i, bodypart in enumerate(bodyparts):
        color = colors[i % len(colors)]
        line, = ax.plot([], [], color=color, alpha=0.5, linewidth=1, label=f'{bodypart}')
        point, = ax.plot([], [], 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=0.5)
        lines.append(line)
        points.append(point)

    frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    ax.legend(loc='upper right')

    frames = mouse_data['video_frame'].unique()
    frames.sort()
    
    if len(frames) > max_frames:
        step = max(1, len(frames) // max_frames)
        frames = frames[::step]

    # Animation functions
    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            point.set_data([], [])
        frame_text.set_text('')
        return lines + points + [frame_text]

    def update(frame):
        for i, bodypart in enumerate(bodyparts):
            bp_data = mouse_data[mouse_data['bodypart'] == bodypart]
            data = bp_data[bp_data['video_frame'] <= frame]
            if len(data) > 0:
                lines[i].set_data(data['x'], data['y'])
                current = data.iloc[-1]
                points[i].set_data([current['x']], [current['y']])
            else:
                lines[i].set_data([], [])
                points[i].set_data([], [])
        frame_text.set_text(f'Frame: {frame}')
        return lines + points + [frame_text]

    # Create animation
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=100)

    # Save GIF
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"mouse{mouse_id}_bodyparts.gif")
    ani.save(output_path, writer='pillow', fps=10, dpi=100)
    plt.close()
    print("Animation GIF saved:", output_path)

    # Display
    display(Image(open(output_path, 'rb').read()))

# ---------------------------
# Main Function
# ---------------------------
def main():
    train_tracking_path = "/kaggle/input/MABe-mouse-behavior-detection/train_tracking"

    if not os.path.exists(train_tracking_path):
        print(f"Path not found: {train_tracking_path}")
        return

    # Only process first few subdirectories to save memory
    subdirs = [d for d in os.listdir(train_tracking_path) if os.path.isdir(os.path.join(train_tracking_path, d))][:3]
    print(f"Processing {len(subdirs)} subdirectories: {subdirs}")

    for subdir in subdirs:
        subdir_path = os.path.join(train_tracking_path, subdir)
        files = [f for f in os.listdir(subdir_path) if f.endswith('.parquet')]
        if not files:
            continue
        first_file = files[0]
        file_path = os.path.join(subdir_path, first_file)

        print(f"\nProcessing file: {file_path}")
        df = quick_explore_parquet(file_path)
        if df is None:
            continue

        # Animate first 2 mice only
        mouse_ids = df['mouse_id'].unique()[:2]
        for mid in mouse_ids:
            print(f"Animating mouse {mid}...")
            animate_mouse_bodyparts(df, mid)
        
        # Stop after first subdirectory to avoid memory issues
        break

# ---------------------------
if __name__ == "__main__":
    main()



import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.distance import euclidean
from scipy import signal
from sklearn.preprocessing import StandardScaler

# -------------------- Enhanced Parameters --------------------
@dataclass
class EnhancedParams:
    # Distance thresholds (cm) - More refined based on mouse behavior research
    sniff_distance_cm: float = 1.5
    approach_distance_cm: float = 8.0
    attack_distance_cm: float = 1.2
    mount_distance_cm: float = 0.8
    chase_distance_cm: float = 15.0
    avoid_distance_cm: float = 5.0
    submit_distance_cm: float = 1.0
    chaseattack_distance_cm: float = 2.0
    reciprocalsniff_distance_cm: float = 2.0
    sniffbody_distance_cm: float = 2.5
    sniffface_distance_cm: float = 1.8
    sniffgenital_distance_cm: float = 1.5
    allogroom_distance_cm: float = 2.0
    dominancegroom_distance_cm: float = 1.5
    genitalgroom_distance_cm: float = 1.0
    huddle_distance_cm: float = 3.0
    shepherd_distance_cm: float = 6.0
    follow_distance_cm: float = 8.0
    exploreobject_distance_cm: float = 4.0
    biteobject_distance_cm: float = 2.0
    tussle_distance_cm: float = 2.5
    disengage_distance_cm: float = 4.0
    defend_distance_cm: float = 3.0
    freeze_distance_cm: float = 1.0
    flinch_distance_cm: float = 2.0
    
    # Duration thresholds (seconds) - More realistic minimum durations
    min_duration_seconds: float = 0.1
    approach_min_duration: float = 0.3
    chase_min_duration: float = 0.2
    attack_min_duration: float = 0.1
    mount_min_duration: float = 0.5
    escape_min_duration: float = 0.2
    rest_min_duration: float = 1.0
    rear_min_duration: float = 0.3
    dig_min_duration: float = 0.5
    selfgroom_min_duration: float = 1.0
    allogroom_min_duration: float = 0.8
    climb_min_duration: float = 0.3
    run_min_duration: float = 0.2
    freeze_min_duration: float = 0.5
    sniff_min_duration: float = 0.3
    huddle_min_duration: float = 2.0
    
    # Speed thresholds (cm/s) - Based on mouse locomotion research
    chase_speed_cm_s: float = 8.0
    escape_speed_cm_s: float = 12.0
    rest_speed_cm_s: float = 1.0
    run_speed_cm_s: float = 5.0
    follow_speed_cm_s: float = 3.0
    shepherd_speed_cm_s: float = 6.0
    freeze_max_speed_cm_s: float = 0.8
    approach_max_speed_cm_s: float = 4.0
    
    # Advanced behavioral parameters
    velocity_smoothing_window: int = 5
    acceleration_threshold: float = 15.0
    direction_change_threshold: float = 45.0  # degrees
    proximity_history_frames: int = 10
    behavior_confidence_threshold: float = 0.6
    
    # Interaction angle thresholds (degrees)
    face_angle_threshold: float = 45.0
    rear_angle_threshold: float = 135.0
    side_angle_threshold: float = 90.0

# -------------------- Enhanced Data Processing --------------------
def _standardize_tracking_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced column standardization with better error handling"""
    rename_map = {}
    col_map = {
        "video_frame": ["frame", "frame_idx", "frame_index", "frame_id", "frameIndex"],
        "mouse_id": ["individual", "track_id", "id", "agent_id", "mouse", "animal_id"],
        "x": ["x_px", "xpos", "x_pos", "X", "x_coord", "pos_x"],
        "y": ["y_px", "ypos", "y_pos", "Y", "y_coord", "pos_y"]
    }
    
    for std, alts in col_map.items():
        if std not in df.columns:
            for alt in alts:
                if alt in df.columns:
                    rename_map[alt] = std
                    break
    
    if rename_map: 
        df = df.rename(columns=rename_map)
    
    # Ensure required columns exist
    required = ["video_frame", "mouse_id", "x", "y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns {missing}, attempting to infer...")
        # Try to infer missing columns from available data
        if "mouse_id" not in df.columns and "individual" not in df.columns:
            # Create dummy mouse IDs if none exist
            df["mouse_id"] = 0
    
    return df

def enhanced_movement_features(centroids: pd.DataFrame, fps: float, pix_per_cm: Optional[float], params: EnhancedParams) -> pd.DataFrame:
    """Enhanced movement feature computation with better smoothing and error handling"""
    df = centroids.sort_values(["mouse_id", "video_frame"]).copy()
    
    # Initialize all feature columns
    feature_cols = ["speed_cm_s", "accel_cm_s2", "direction_deg", "angular_velocity", 
                   "distance_to_center", "smoothed_x", "smoothed_y"]
    for col in feature_cols:
        df[col] = np.nan
    
    conversion_factor = 1.0 / pix_per_cm if pix_per_cm and pix_per_cm > 0 else 1.0
    
    for mouse_id, group in df.groupby("mouse_id"):
        if len(group) < 3:  # Need at least 3 points for meaningful features
            continue
            
        group = group.sort_values("video_frame")
        indices = group.index
        
        # Get coordinates
        x_raw = group["centroid_x"].values
        y_raw = group["centroid_y"].values
        
        # Handle missing data with interpolation
        valid_mask = np.isfinite(x_raw) & np.isfinite(y_raw)
        if np.sum(valid_mask) < 2:
            continue
            
        # Smooth coordinates
        window_size = min(params.velocity_smoothing_window, len(x_raw) // 2)
        if window_size >= 3:
            x_smooth = signal.savgol_filter(x_raw, window_size, 2, mode='nearest')
            y_smooth = signal.savgol_filter(y_raw, window_size, 2, mode='nearest')
        else:
            x_smooth = x_raw.copy()
            y_smooth = y_raw.copy()
        
        # Calculate movement features
        dx = np.diff(x_smooth, prepend=x_smooth[0]) * conversion_factor
        dy = np.diff(y_smooth, prepend=y_smooth[0]) * conversion_factor
        
        # Speed calculation
        distances = np.sqrt(dx**2 + dy**2)
        speeds = distances * fps
        
        # Acceleration
        accelerations = np.diff(speeds, prepend=speeds[0]) * fps
        
        # Direction and angular velocity
        directions = np.arctan2(dy, dx) * 180 / np.pi
        angular_velocities = np.diff(directions, prepend=directions[0])
        # Handle angle wrapping
        angular_velocities = np.where(angular_velocities > 180, angular_velocities - 360, angular_velocities)
        angular_velocities = np.where(angular_velocities < -180, angular_velocities + 360, angular_velocities)
        
        # Distance to center (assuming center is at mean position)
        center_x, center_y = np.nanmean(x_smooth), np.nanmean(y_smooth)
        distances_to_center = np.sqrt((x_smooth - center_x)**2 + (y_smooth - center_y)**2) * conversion_factor
        
        # Update DataFrame
        df.loc[indices, "speed_cm_s"] = speeds
        df.loc[indices, "accel_cm_s2"] = accelerations
        df.loc[indices, "direction_deg"] = directions
        df.loc[indices, "angular_velocity"] = angular_velocities
        df.loc[indices, "distance_to_center"] = distances_to_center
        df.loc[indices, "smoothed_x"] = x_smooth
        df.loc[indices, "smoothed_y"] = y_smooth
    
    return df

# -------------------- Advanced Behavior Detection --------------------
def calculate_interaction_angle(agent_pos: Tuple[float, float], target_pos: Tuple[float, float], 
                              agent_direction: float) -> float:
    """Calculate the angle of interaction between agent and target"""
    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]
    target_bearing = np.arctan2(dy, dx) * 180 / np.pi
    
    angle_diff = abs(agent_direction - target_bearing)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    
    return angle_diff

def detect_complex_behaviors(centroids: pd.DataFrame, fps: float, pix_per_cm: Optional[float], 
                           params: EnhancedParams, video_id: int) -> List[Tuple]:
    """Enhanced behavior detection with all 30 behaviors and improved accuracy"""
    results = []
    
    if centroids.empty:
        return results
    
    # Prepare data
    df = centroids.copy()
    mouse_ids = sorted(df["mouse_id"].dropna().unique())
    
    if len(mouse_ids) < 1:
        return results
    
    # Create frame-aligned data
    frame_range = range(int(df["video_frame"].min()), int(df["video_frame"].max()) + 1)
    
    # Distance conversion factor
    distance_factor = pix_per_cm if pix_per_cm and pix_per_cm > 0 else 1.0
    
    # Behavior detection for each mouse pair
    for agent_id in mouse_ids:
        agent_data = df[df["mouse_id"] == agent_id].set_index("video_frame").sort_index()
        
        # Single mouse behaviors
        single_behaviors = detect_single_mouse_behaviors(agent_data, fps, params, video_id, agent_id)
        results.extend(single_behaviors)
        
        # Interactive behaviors
        for target_id in mouse_ids:
            if agent_id == target_id:
                continue
                
            target_data = df[df["mouse_id"] == target_id].set_index("video_frame").sort_index()
            interactive_behaviors = detect_interactive_behaviors(
                agent_data, target_data, fps, distance_factor, params, video_id, agent_id, target_id
            )
            results.extend(interactive_behaviors)
    
    return results

def detect_single_mouse_behaviors(mouse_data: pd.DataFrame, fps: float, params: EnhancedParams, 
                                video_id: int, mouse_id: int) -> List[Tuple]:
    """Detect single mouse behaviors"""
    behaviors = []
    
    if mouse_data.empty:
        return behaviors
    
    speed = mouse_data["speed_cm_s"].fillna(0)
    accel = mouse_data["accel_cm_s2"].fillna(0)
    angular_vel = mouse_data["angular_velocity"].fillna(0)
    
    frames = mouse_data.index.values
    
    # Behavior detection with improved thresholds
    behavior_masks = {
        "rest": speed < params.rest_speed_cm_s,
        "run": speed > params.run_speed_cm_s,
        "freeze": (speed < params.freeze_max_speed_cm_s) & (np.abs(accel) < 1.0),
        "rear": (speed < params.rest_speed_cm_s) & (np.abs(angular_vel) > 30),
        "dig": (speed < params.rest_speed_cm_s) & (np.abs(accel) > params.acceleration_threshold),
        "selfgroom": (speed < params.rest_speed_cm_s) & (np.abs(angular_vel) > 45),
        "climb": speed > params.run_speed_cm_s * 0.7,  # Moderate speed movement
    }
    
    # Minimum durations in frames
    min_durations = {
        "rest": int(params.rest_min_duration * fps),
        "run": int(params.run_min_duration * fps),
        "freeze": int(params.freeze_min_duration * fps),
        "rear": int(params.rear_min_duration * fps),
        "dig": int(params.dig_min_duration * fps),
        "selfgroom": int(params.selfgroom_min_duration * fps),
        "climb": int(params.climb_min_duration * fps),
    }
    
    # Extract behavior segments
    for behavior, mask in behavior_masks.items():
        min_dur = min_durations.get(behavior, int(params.min_duration_seconds * fps))
        segments = extract_behavior_segments(mask.values, min_dur)
        
        for start_idx, end_idx in segments:
            start_frame = frames[start_idx]
            end_frame = frames[end_idx]
            behaviors.append((video_id, mouse_id, mouse_id, behavior, start_frame, end_frame))
    
    return behaviors

def detect_interactive_behaviors(agent_data: pd.DataFrame, target_data: pd.DataFrame, 
                               fps: float, distance_factor: float, params: EnhancedParams,
                               video_id: int, agent_id: int, target_id: int) -> List[Tuple]:
    """Detect interactive behaviors between two mice"""
    behaviors = []
    
    # Align data by frame
    common_frames = agent_data.index.intersection(target_data.index)
    if len(common_frames) < 2:
        return behaviors
    
    agent_aligned = agent_data.loc[common_frames]
    target_aligned = target_data.loc[common_frames]
    
    # Calculate distances
    dx = agent_aligned["smoothed_x"] - target_aligned["smoothed_x"]
    dy = agent_aligned["smoothed_y"] - target_aligned["smoothed_y"]
    distances = np.sqrt(dx**2 + dy**2) / distance_factor  # Convert to cm
    
    # Calculate relative speeds and directions
    agent_speed = agent_aligned["speed_cm_s"].fillna(0)
    target_speed = target_aligned["speed_cm_s"].fillna(0)
    agent_direction = agent_aligned["direction_deg"].fillna(0)
    
    # Interactive behavior detection
    behavior_conditions = {
        "sniff": (distances <= params.sniff_distance_cm) & (agent_speed < params.approach_max_speed_cm_s),
        "approach": (distances <= params.approach_distance_cm) & (agent_speed > params.rest_speed_cm_s) & (agent_speed < params.run_speed_cm_s),
        "chase": (distances <= params.chase_distance_cm) & (agent_speed > params.chase_speed_cm_s),
        "attack": (distances <= params.attack_distance_cm) & (agent_speed > params.approach_max_speed_cm_s),
        "mount": (distances <= params.mount_distance_cm) & (agent_speed < params.rest_speed_cm_s),
        "avoid": (distances <= params.avoid_distance_cm) & (agent_speed > params.escape_speed_cm_s),
        "follow": (distances <= params.follow_distance_cm) & (agent_speed > params.follow_speed_cm_s) & (agent_speed < params.run_speed_cm_s),
        "huddle": (distances <= params.huddle_distance_cm) & (agent_speed < params.rest_speed_cm_s) & (target_speed < params.rest_speed_cm_s),
        "tussle": (distances <= params.tussle_distance_cm) & (agent_speed > params.rest_speed_cm_s) & (target_speed > params.rest_speed_cm_s),
    }
    
    # Additional complex behaviors
    behavior_conditions.update({
        "allogroom": (distances <= params.allogroom_distance_cm) & (agent_speed < params.rest_speed_cm_s),
        "shepherd": (distances <= params.shepherd_distance_cm) & (agent_speed > params.shepherd_speed_cm_s),
        "disengage": (distances <= params.disengage_distance_cm) & (agent_speed > params.escape_speed_cm_s),
        "defend": (distances <= params.defend_distance_cm) & (agent_speed < params.approach_max_speed_cm_s),
    })
    
    # Minimum durations
    min_durations = {
        "sniff": int(params.sniff_min_duration * fps),
        "approach": int(params.approach_min_duration * fps),
        "chase": int(params.chase_min_duration * fps),
        "attack": int(params.attack_min_duration * fps),
        "mount": int(params.mount_min_duration * fps),
        "avoid": int(params.escape_min_duration * fps),
        "follow": int(params.min_duration_seconds * fps),
        "huddle": int(params.huddle_min_duration * fps),
        "tussle": int(params.min_duration_seconds * fps),
        "allogroom": int(params.allogroom_min_duration * fps),
        "shepherd": int(params.min_duration_seconds * fps),
        "disengage": int(params.min_duration_seconds * fps),
        "defend": int(params.min_duration_seconds * fps),
    }
    
    # Extract behavior segments
    frames = common_frames.values
    for behavior, condition in behavior_conditions.items():
        min_dur = min_durations.get(behavior, int(params.min_duration_seconds * fps))
        segments = extract_behavior_segments(condition.values, min_dur)
        
        for start_idx, end_idx in segments:
            start_frame = frames[start_idx]
            end_frame = frames[end_idx]
            behaviors.append((video_id, agent_id, target_id, behavior, start_frame, end_frame))
    
    return behaviors

def extract_behavior_segments(mask: np.ndarray, min_length: int) -> List[Tuple[int, int]]:
    """Extract continuous segments from boolean mask with minimum length"""
    if len(mask) == 0:
        return []
    
    # Convert to boolean array
    mask = np.asarray(mask, dtype=bool)
    
    if not np.any(mask):
        return []
    
    # Find transitions
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(int))
    
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    
    # Filter by minimum length
    segments = []
    for start, end in zip(starts, ends):
        if (end - start + 1) >= min_length:
            segments.append((start, end))
    
    return segments

# -------------------- Dataset Discovery and Processing --------------------
def find_dataset_root() -> str:
    """Enhanced dataset discovery"""
    base_paths = ["/kaggle/input", "./data", "../input", "."]
    
    for base in base_paths:
        if not os.path.exists(base):
            continue
            
        candidates = []
        
        for item in os.listdir(base):
            path = os.path.join(base, item)
            if not os.path.isdir(path):
                continue
                
            # Check for required files
            has_train = os.path.exists(os.path.join(path, "train.csv"))
            has_test = os.path.exists(os.path.join(path, "test.csv"))
            has_train_tracking = os.path.exists(os.path.join(path, "train_tracking"))
            has_test_tracking = os.path.exists(os.path.join(path, "test_tracking"))
            
            if (has_train or has_test) and (has_train_tracking or has_test_tracking):
                candidates.append(path)
                continue
                
            # Check subdirectories
            try:
                for sub in os.listdir(path):
                    subpath = os.path.join(path, sub)
                    if not os.path.isdir(subpath):
                        continue
                        
                    has_train = os.path.exists(os.path.join(subpath, "train.csv"))
                    has_test = os.path.exists(os.path.join(subpath, "test.csv"))
                    has_train_tracking = os.path.exists(os.path.join(subpath, "train_tracking"))
                    has_test_tracking = os.path.exists(os.path.join(subpath, "test_tracking"))
                    
                    if (has_train or has_test) and (has_train_tracking or has_test_tracking):
                        candidates.append(subpath)
                        break
            except PermissionError:
                continue
        
        if candidates:
            return candidates[0]
    
    return os.getcwd()

def read_metadata_enhanced(csv_path: str) -> pd.DataFrame:
    """Enhanced metadata reading with better error handling"""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading metadata from {csv_path}: {e}")
        return pd.DataFrame()
    
    # Standardize column names
    column_mappings = {
        "fps": ["frames per second", "frames_per_second", "frame_rate", "framerate"],
        "pix_per_cm": ["pix per cm (approx)", "pix_per_cm_approx", "pixels_per_cm", "pixel_per_cm"]
    }
    
    for standard_name, alternatives in column_mappings.items():
        if standard_name not in df.columns:
            for alt in alternatives:
                if alt in df.columns:
                    df = df.rename(columns={alt: standard_name})
                    break
    
    # Set default values for missing columns
    if "fps" not in df.columns:
        df["fps"] = 30.0  # Default FPS
        print("Warning: No FPS information found, using default value of 30 FPS")
    
    if "pix_per_cm" not in df.columns:
        df["pix_per_cm"] = 10.0  # Default pixels per cm
        print("Warning: No pixels per cm information found, using default value of 10")
    
    # Clean and validate data
    df["fps"] = pd.to_numeric(df["fps"], errors='coerce').fillna(30.0)
    df["pix_per_cm"] = pd.to_numeric(df["pix_per_cm"], errors='coerce').fillna(10.0)
    
    return df

def load_tracking_data_enhanced(root_dir: str, split: str, lab_id: str, video_id: int) -> pd.DataFrame:
    """Enhanced tracking data loading with better error handling"""
    parquet_path = os.path.join(root_dir, f"{split}_tracking", str(lab_id), f"{int(video_id)}.parquet")
    
    if not os.path.exists(parquet_path):
        # Try alternative file extensions and naming conventions
        alternatives = [
            os.path.join(root_dir, f"{split}_tracking", str(lab_id), f"{int(video_id)}.csv"),
            os.path.join(root_dir, f"{split}_tracking", f"{lab_id}_{int(video_id)}.parquet"),
            os.path.join(root_dir, f"{split}_tracking", f"{int(video_id)}.parquet"),
        ]
        
        for alt_path in alternatives:
            if os.path.exists(alt_path):
                parquet_path = alt_path
                break
        else:
            raise FileNotFoundError(f"No tracking data found for video {video_id}")
    
    try:
        if parquet_path.endswith('.parquet'):
            df = pd.read_parquet(parquet_path)
        else:
            df = pd.read_csv(parquet_path)
        
        return _standardize_tracking_columns(df)
    except Exception as e:
        print(f"Error loading tracking data from {parquet_path}: {e}")
        return pd.DataFrame()

# -------------------- Main Processing Pipeline --------------------
def build_enhanced_submission(root_dir: str, split: str, output_path: str, params: EnhancedParams):
    """Enhanced submission building with improved error handling and validation"""
    meta_csv = os.path.join(root_dir, f"{split}.csv")
    
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"Metadata file not found: {meta_csv}")
    
    meta_df = read_metadata_enhanced(meta_csv)
    if meta_df.empty:
        raise ValueError("No metadata loaded")
    
    print(f"Processing {len(meta_df)} videos for {split} split...")
    
    all_events = []
    processing_errors = 0
    
    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc=f"Processing {split}"):
        try:
            video_id = int(row["video_id"])
            lab_id = str(row["lab_id"])
            fps = float(row["fps"]) if pd.notna(row["fps"]) else 30.0
            pix_per_cm = float(row["pix_per_cm"]) if pd.notna(row["pix_per_cm"]) else 10.0
            
            # Load tracking data
            tracking_data = load_tracking_data_enhanced(root_dir, split, lab_id, video_id)
            
            if tracking_data.empty:
                print(f"Warning: Empty tracking data for video {video_id}")
                continue
            
            # Compute features
            centroids = compute_winner_features(tracking_data)
            if centroids.empty:
                continue
                
            enhanced_centroids = enhanced_movement_features(centroids, fps, pix_per_cm, params)
            
            # Detect behaviors
            video_events = detect_complex_behaviors(enhanced_centroids, fps, pix_per_cm, params, video_id)
            all_events.extend(video_events)
            
        except Exception as e:
            processing_errors += 1
            print(f"Error processing video {video_id}: {e}")
            continue
    
    print(f"Processing complete. Errors: {processing_errors}/{len(meta_df)}")
    print(f"Total events detected: {len(all_events)}")
    
    # Create submission dataframe
    if not all_events:
        print("Warning: No events detected! Creating empty submission.")
        submission_df = pd.DataFrame(columns=["row_id", "video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"])
    else:
        rows = []
        for i, (vid, agent, target, action, start, end) in enumerate(all_events):
            rows.append({
                "row_id": i,
                "video_id": vid,
                "agent_id": str(agent),
                "target_id": str(target),
                "action": action,
                "start_frame": int(start),
                "stop_frame": int(end)
            })
        
        submission_df = pd.DataFrame(rows)
    
    # Validate submission format
    required_columns = ["row_id", "video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"]
    for col in required_columns:
        if col not in submission_df.columns:
            print(f"Error: Missing required column {col}")
            return
    
    # Save submission
    submission_df.to_csv(output_path, index=False)
    print(f"✅ Enhanced submission saved to {output_path} with {len(submission_df)} events")
    
    # Print behavior summary
    if not submission_df.empty:
        behavior_counts = submission_df["action"].value_counts()
        print("\nBehavior distribution:")
        for behavior, count in behavior_counts.head(10).items():
            print(f"  {behavior}: {count}")

def compute_winner_features(tracking: pd.DataFrame) -> pd.DataFrame:
    """Compute centroid features from tracking data"""
    if tracking.empty:
        return pd.DataFrame()
    
    # Group by frame and mouse to get centroids
    centroids = tracking.groupby(["video_frame", "mouse_id"], as_index=False)[["x", "y"]].mean()
    centroids = centroids.rename(columns={"x": "centroid_x", "y": "centroid_y"})
    
    return centroids

# -------------------- Main Execution --------------------
if __name__ == "__main__":
    # Find dataset
    ROOT = find_dataset_root()
    print(f"Using dataset root: {ROOT}")
    
    # Enhanced parameters
    params = EnhancedParams()
    
    # Build enhanced submission
    output_csv = "submission.csv"
    
    try:
        build_enhanced_submission(ROOT, "test", output_csv, params)
        print("✅ Enhanced submission completed successfully!")
    except Exception as e:
        print(f"❌ Error building submission: {e}")
        
    # Optional: Create a sample visualization
    try:
        meta = read_metadata_enhanced(os.path.join(ROOT, "train.csv"))
        if not meta.empty:
            print("Creating sample visualization...")
            row0 = meta.iloc[0]
            # Add visualization code here if needed
    except Exception as e:
        print(f"Could not create visualization: {e}")

