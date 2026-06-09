# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Matplotlib defaults
plt.rcParams["figure.figsize"] = (8, 4)
plt.rcParams["axes.grid"] = True

DATA_DIR = Path("/kaggle/input/MABe-mouse-behavior-detection") 
print("Using DATA_DIR:", DATA_DIR.resolve())



train_meta_path = DATA_DIR / "train.csv"
test_meta_path  = DATA_DIR / "test.csv"
assert train_meta_path.exists(), f"Missing {train_meta_path}"
assert test_meta_path.exists(),  f"Missing {test_meta_path}"

train_meta = pd.read_csv(train_meta_path)
test_meta  = pd.read_csv(test_meta_path)
print("train_meta shape:", train_meta.shape)
display(train_meta.head())


n_labs = train_meta["lab_id"].nunique() if "lab_id" in train_meta.columns else np.nan
n_videos = train_meta["video_id"].nunique() if "video_id" in train_meta.columns else np.nan
print({"labs": n_labs, "videos": n_videos})
display(train_meta.describe(include="all").transpose())


train_meta.info()


train_meta.isnull().sum()


fps_col = "frames_per_second"
dur_col = "video_duration_sec"

if fps_col in train_meta.columns:
    train_meta[fps_col].dropna().astype(float).hist(bins=30)
    plt.title("Distribution of Frame Rates (FPS)")
    plt.xlabel("Frames per second")
    plt.ylabel("Count")
    plt.show()

if dur_col in train_meta.columns:
    train_meta[dur_col].dropna().astype(float).hist(bins=30)
    plt.title("Video Duration Distribution")
    plt.xlabel("Seconds")
    plt.ylabel("Count")
    plt.show()

print("FPS describe:\n", train_meta.get(fps_col, pd.Series(dtype=float)).describe())
print("Duration describe:\n", train_meta.get(dur_col, pd.Series(dtype=float)).describe())


arena_shape_col = "arena_shape"
arena_type_col = "arena_type"

if arena_shape_col in train_meta.columns:
    vc = train_meta[arena_shape_col].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(vc.index.astype(str), vc.values, color="skyblue", edgecolor="black")
    ax.bar_label(ax.containers[0])
    ax.set_title("Arena Shape Distribution")
    ax.set_xlabel("Arena Shape")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

if arena_type_col in train_meta.columns:
    vc = train_meta[arena_type_col].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(vc.index.astype(str), vc.values, color="skyblue", edgecolor="black")
    ax.bar_label(ax.containers[0])
    ax.set_title("Arena Type Distribution")
    ax.set_xlabel("Arena Type")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


train_meta.mouse1_condition.nunique(), train_meta.mouse1_condition.unique()


train_meta.mouse2_condition.nunique(), train_meta.mouse3_condition.nunique(),train_meta.mouse4_condition.nunique()


train_meta.mouse4_condition.unique()


train_meta.mouse1_condition.unique()[200:250]


# We will analize 10 top conditions

# Identify all mouse<n>_condition columns automatically
cond_cols = [c for c in train_meta.columns if c.startswith("mouse") and c.endswith("_condition")]

print(f"Found {len(cond_cols)} condition columns:", cond_cols)

summary_list = []

for col in cond_cols:
    vc = train_meta[col].value_counts(dropna=False)
    print(f"\n=== {col} ===")
    print(vc.head(10))  # top 10 values
    print(f"Unique: {vc.index.nunique()}, Missing: {train_meta[col].isna().sum()}")

    # save for combined summary
    summary_list.append(
        pd.DataFrame({
            "column": col,
            "condition": vc.index.astype(str),
            "count": vc.values
        })
    )

# Combine all results
df_conditions = pd.concat(summary_list, ignore_index=True)

# Clean up (remove weird spacing or quotes)
df_conditions["condition"] = df_conditions["condition"].str.strip().str.strip("'\"").replace({"nan": "Missing"})

# Sort by frequency
top_conditions = (
    df_conditions.groupby("condition")["count"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

print("\nTop overall conditions across all mice:")
print(top_conditions.head(10))

# Visualization
plt.figure(figsize=(8,5))
bars = plt.bar(top_conditions["condition"].head(10), top_conditions["count"].head(10), color="skyblue")
plt.title("Top 10 Mouse Conditions (All Columns Combined)")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Count")
plt.tight_layout()
plt.bar_label(bars, fmt="%d", label_type="edge", padding=3)
plt.show()


train_meta.head(7).behaviors_labeled


train_meta.behaviors_labeled[0]


train_meta.iloc[0,:]


import ast


col = "behaviors_labeled"
vid_col = "video_id"

records = []
skipped = 0
for i, row in train_meta.iterrows():
    s = row.get(col, None)
    vid = row.get(vid_col, None)
    if pd.isna(s):
        continue
    try:
        items = ast.literal_eval(s)
        if not isinstance(items, (list, tuple)):
            skipped += 1
            continue
        for it in items:
            parts = str(it).split(",")
            if len(parts) != 3:
                continue
            agent, target, behavior = [p.strip().strip("'\"") for p in parts]
            records.append({"video_id": vid, "agent": agent, "target": target, "behavior": behavior})
    except Exception:
        skipped += 1

df_beh = pd.DataFrame.from_records(records)
print("Parsed rows:", len(df_beh), "| Skipped rows:", skipped)
display(df_beh.head())


train_meta.video_id.nunique(), df_beh.video_id.nunique()


train_meta.video_id.nunique()-train_meta.video_id.isna().sum()-train_meta.behaviors_labeled.isna().sum()


df_beh.target.isna().sum(), df_beh.agent.isna().sum()


df_beh.agent.unique(), df_beh.target.unique()


df_beh.groupby('video_id')['target'].unique().apply(list).reset_index(name="behaviors_list")


df_beh.groupby('video_id')['agent'].unique().apply(list).reset_index(name="behaviors_list")


df_beh["is_self"] = (df_beh["target"].str.lower() == "self") | (df_beh["agent"].str.lower() == df_beh["target"].str.lower())
split_counts = df_beh["is_self"].map({True: "self", False: "social"}).value_counts()
display(split_counts.to_frame("count"))

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(split_counts.index.astype(str), split_counts.values)
ax.set_title("Self vs Social (All Videos)")
ax.set_xlabel("Type")
ax.set_ylabel("Count")
ax.bar_label(ax.containers[0], fmt='%d', label_type='edge', padding=3)
fig.tight_layout()
plt.show()


pair_counts = df_beh.groupby(["agent", "target"]).size()
display(pair_counts.to_frame("count"))


#11+17+276+290


labels = [f"{a}→{t}" for a,t in pair_counts.index]
fig, ax = plt.subplots(figsize=(8,4))
bars = ax.bar(labels, pair_counts.values)
ax.set_title("Top Agent→Target Pairs")
ax.set_xlabel("Pair")
ax.set_ylabel("Count")
plt.xticks(rotation=45, ha="right")
ax.bar_label(ax.containers[0], fmt='%d', label_type='edge', padding=3)
fig.tight_layout()
plt.show()



beh_counts = df_beh["behavior"].value_counts()
print(f"Number of unique behaviors is {df_beh.behavior.nunique()}")
display(beh_counts.to_frame("count"))

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(beh_counts.index.astype(str), beh_counts.values)
ax.set_title("Behavior Frequency (All Videos)")
ax.set_xlabel("Behavior")
ax.set_ylabel("Count")
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
ax.bar_label(ax.containers[0], fmt='%d', label_type='edge', padding=3)
plt.show()


train_meta.lab_id.value_counts()


train_meta.head()


lab_id_exp = "AdaptableSnail"
video_id_exp = 44566106


train_meta[(train_meta.lab_id == lab_id_exp) & (train_meta.video_id == video_id_exp)]


df_beh[df_beh.video_id == video_id_exp].groupby(["agent", "target"]).size()


df_beh[df_beh.video_id == video_id_exp]


track_dir = DATA_DIR /'train_tracking'/lab_id_exp
sample_track = None
if track_dir.exists():
    # try to find a parquet named after a train video_id
    cand = track_dir / f"{video_id_exp}.parquet"
    if cand.exists():
        sample_track = cand
    # or just take any parquet in the folder
    if sample_track is None:
        pq_files = list(track_dir.glob("*.parquet"))
        if pq_files:
            sample_track = pq_files[0]

if sample_track is not None:
    print("Sample tracking file:", sample_track)
    try:
        df_track = pd.read_parquet(sample_track)
    except Exception as e:
        print("Parquet read error:", e)
        df_track = None
else:
    print("No tracking parquet found under", track_dir)
    df_track = None

if df_track is not None and not df_track.empty:
    display(df_track.head())
    cols = df_track.columns.tolist()
    need = {"video_frame", "mouse_id", "bodypart", "x", "y"}
    if need.issubset(set(cols)):
        mouse0 = df_track["mouse_id"].iloc[0]
        body0 = df_track["bodypart"].iloc[0]
        path_df = df_track[(df_track["mouse_id"] == mouse0) & (df_track["bodypart"] == body0)]#.head(500)
        plt.plot(path_df["x"].values, path_df["y"].values, linewidth=1)
        plt.gca().invert_yaxis()
        plt.title(f"Trajectory of {body0} (Mouse {mouse0})")
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.show()
    else:
        print("Expected tracking columns not found. Got:", cols)


print(df_track.shape)
df_track


df_track[(df_track.mouse_id == 1)&(df_track.bodypart=='body_center')]


df_track.nunique()


df_track.groupby("mouse_id").size()


annotation_dir = DATA_DIR /'train_annotation'/lab_id_exp


annotation_data = pd.read_parquet(annotation_dir/ f"{video_id_exp}.parquet")
print(annotation_data.shape)
annotation_data.head()


annotation_data.head(20)


print(annotation_data.agent_id.unique())
annotation_data.agent_id.value_counts()


print(annotation_data.target_id.unique())
annotation_data.target_id.value_counts()


print(annotation_data.action.unique())
annotation_data.action.value_counts()


annotation_data.start_frame.min(), annotation_data.start_frame.max()


annotation_data.stop_frame.min(), annotation_data.stop_frame.max()


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_action_trajectory(track_parquet,
                           ann_parquet,
                           action_idx=0,
                           bodypart=None,
                           max_points=None):
    """
    Plot agent and target trajectories for one annotated action.
    - track_parquet: path to <video_id>.parquet with columns [video_frame, mouse_id, bodypart, x, y]
    - ann_parquet:   path to <video_id>.parquet with columns [agent_id, target_id, action, start_frame, stop_frame]
    - action_idx:    which row in the annotations to plot (after any external filtering)
    - bodypart:      e.g. 'nose' or 'center'. If None, uses centroid over all bodyparts per (frame, mouse).
    - max_points:    optionally limit number of plotted points for readability
    """
    track_parquet = Path(track_parquet)
    ann_parquet   = Path(ann_parquet)

    df_track = pd.read_parquet(track_parquet)
    df_ann   = pd.read_parquet(ann_parquet)

    assert len(df_ann) > action_idx, f"action_idx {action_idx} out of range (len={len(df_ann)})"
    ann = df_ann.iloc[action_idx]

    agent_id = int(ann["agent_id"])
    target_id = int(ann["target_id"])
    start_f = int(ann["start_frame"])
    stop_f  = int(ann["stop_frame"])
    action  = str(ann["action"])

    # Slice the time window
    mask_t = (df_track["video_frame"] >= start_f) & (df_track["video_frame"] <= stop_f)
    dfw = df_track.loc[mask_t].copy()

    # Harmonize dtypes
    dfw["mouse_id"] = pd.to_numeric(dfw["mouse_id"], errors="coerce").astype("Int64")

    # Either pick one bodypart, or compute centroid per (frame, mouse)
    if bodypart is not None and bodypart in dfw["bodypart"].unique():
        dfw = dfw[dfw["bodypart"] == bodypart]
        # Keep one row per (frame, mouse)
        dfw = dfw.sort_values(["mouse_id", "video_frame"])
    else:
        # centroid across available bodyparts at each (frame, mouse)
        dfw = (
            dfw.groupby(["video_frame", "mouse_id"], as_index=False)[["x","y"]]
               .mean()
               .sort_values(["mouse_id", "video_frame"])
        )

    # Extract trajectories
    traj_agent  = dfw[dfw["mouse_id"] == agent_id].copy()
    traj_target = dfw[dfw["mouse_id"] == target_id].copy()

    if max_points is not None:
        traj_agent  = traj_agent.iloc[::max(1, len(traj_agent)//max_points or 1)]
        traj_target = traj_target.iloc[::max(1, len(traj_target)//max_points or 1)]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    if agent_id == target_id:
        ax.plot(traj_agent["x"], traj_agent["y"], linewidth=2, label=f"agent=target={agent_id}")
        if not traj_agent.empty:
            ax.scatter(traj_agent["x"].iloc[0],  traj_agent["y"].iloc[0],  marker="o", s=60, label="start")
            ax.scatter(traj_agent["x"].iloc[-1], traj_agent["y"].iloc[-1], marker="X", s=80, label="end")
    else:
        ax.plot(traj_agent["x"],  traj_agent["y"],  linewidth=2, label=f"agent {agent_id}")
        ax.plot(traj_target["x"], traj_target["y"], linewidth=2, label=f"target {target_id}")
        if not traj_agent.empty:
            ax.scatter(traj_agent["x"].iloc[0],  traj_agent["y"].iloc[0],  marker="o", s=60, label="agent start")
            ax.scatter(traj_agent["x"].iloc[-1], traj_agent["y"].iloc[-1], marker="X", s=80, label="agent end")
        if not traj_target.empty:
            ax.scatter(traj_target["x"].iloc[0],  traj_target["y"].iloc[0],  marker="o", s=60, label="target start")
            ax.scatter(traj_target["x"].iloc[-1], traj_target["y"].iloc[-1], marker="X", s=80, label="target end")

    # Image coordinates: y grows downward → invert Y for natural overlay feel
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"{track_parquet.stem} — {action} | frames {start_f}–{stop_f}"
                 + (f" | bodypart='{bodypart}'" if bodypart else " | centroid"))
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.legend(
        loc="upper left",          # anchor point of the legend box itself
        bbox_to_anchor=(1.02, 1),  # position relative to the axes (1.02 → just outside right)
        borderaxespad=0,           # small padding
        frameon=True               # optional: show border
    )
    plt.tight_layout()
    plt.show()


video_id = "44566106"  # example
track_path = f"{track_dir}/{video_id_exp}.parquet"
ann_path   =  f"{annotation_dir}/{video_id_exp}.parquet"

# 1) Plot the first annotation using centroid across bodyparts
plot_action_trajectory(track_path, ann_path)



annotation_data[(annotation_data.agent_id == 1)& (annotation_data.target_id== 3)].sort_values("start_frame")


plot_action_trajectory(track_path, ann_path, action_idx = 222)


annotation_data[(annotation_data.agent_id == 3)& (annotation_data.target_id== 1)].sort_values("start_frame")


plot_action_trajectory(track_path, ann_path, action_idx = 221)


annotation_data[(annotation_data.agent_id == 1)& (annotation_data.target_id== 1)]


plot_action_trajectory(track_path, ann_path, action_idx = 287)


import pandas as pd

def find_reciprocal_behaviors(df_ann, min_overlap_frames=1):
    """
    Return all pairs of annotations where agent A->B overlaps in time with B->A.
    Works per-video; ensure df_ann is for one video_id.
    """
    df = df_ann.copy()

    # Normalize ints
    for c in ["agent_id", "target_id", "start_frame", "stop_frame"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Exclude self behaviors (A==B) for reciprocity
    df = df[df["agent_id"] != df["target_id"]].reset_index(drop=True)

    # Prepare a swapped copy for the self-merge (B->A)
    df_swapped = df.rename(columns={
        "agent_id": "target_id",
        "target_id": "agent_id",
        "action": "action_rev",
        "start_frame": "start_frame_rev",
        "stop_frame":  "stop_frame_rev"
    })

    # Self-merge on swapped agent/target (A->B matched with B->A)
    merged = df.merge(
        df_swapped,
        on=["agent_id", "target_id"],  # (A,B) in left matches (B,A) in right AFTER renaming
        how="inner",
        suffixes=("", "_drop")
    )

    # Time-overlap test: max(starts) <= min(stops) and overlap length >= min_overlap_frames
    start_max = merged[["start_frame", "start_frame_rev"]].max(axis=1)
    stop_min  = merged[["stop_frame",  "stop_frame_rev"]].min(axis=1)
    merged["overlap_len"] = (stop_min - start_max + 1).clip(lower=0)

    # Keep only overlapping intervals
    out = merged[merged["overlap_len"] >= min_overlap_frames].copy()

    # Optional: interval IoU-like measure (useful for filtering)
    union_len = (
        merged[["stop_frame", "stop_frame_rev"]].max(axis=1) -
        merged[["start_frame", "start_frame_rev"]].min(axis=1) + 1
    )
    out["overlap_iou"] = out["overlap_len"] / union_len

    # Keep just the essentials, tidy columns
    cols = [
        "agent_id", "target_id",
        "action", "start_frame", "stop_frame",
        "action_rev", "start_frame_rev", "stop_frame_rev",
        "overlap_len", "overlap_iou"
    ]
    return out[cols].sort_values(["agent_id", "target_id", "start_frame", "start_frame_rev"]).reset_index(drop=True)


recip = find_reciprocal_behaviors(annotation_data, min_overlap_frames=1)
recip


pair_counts = (
    recip.groupby(["action", "action_rev"])
         .size()
         .reset_index(name="count")
         .sort_values("count", ascending=False)
)
print(pair_counts.head())


same_action = recip[recip["action"] == recip["action_rev"]]
same_action


# --- Define the plotting function ---
def plot_mouse_trajectory(
    df: pd.DataFrame,
    frame_col: str = "frame",
    x_col: str = "x",
    y_col: str = "y",
    label_col: str = "label",
    title: str = "Mouse Trajectory by Frame & Annotation",
    colors: dict | None = None,
    show_unlabeled: bool = True,
):
    """Plot mouse trajectory segmented by annotation labels."""
    data = df.copy().sort_values(by=frame_col).reset_index(drop=True)

    def _norm_label(v):
        if pd.isna(v) or (isinstance(v, str) and v.strip() == ""):
            return None
        return v

    data["_label_norm"] = data[label_col].apply(_norm_label)

    unique_labels = [l for l in data["_label_norm"].dropna().unique().tolist()]
    if colors is None:
        tab10 = plt.get_cmap("tab10")
        auto_colors = {lab: tab10(i % 10) for i, lab in enumerate(unique_labels)}
    else:
        auto_colors = colors.copy()

    unlabeled_style = dict(color="0.5", linestyle="--", linewidth=1.5)

    segments = []
    if not data.empty:
        seg_start_idx = 0
        for i in range(1, len(data)):
            prev = data.iloc[i-1]
            cur = data.iloc[i]
            label_changed = (prev["_label_norm"] != cur["_label_norm"])
            frame_gap = (cur[frame_col] - prev[frame_col]) > 1
            if label_changed or frame_gap:
                segments.append((seg_start_idx, i-1))
                seg_start_idx = i
        segments.append((seg_start_idx, len(data)-1))

    fig, ax = plt.subplots(figsize=(8, 6))
    seen_for_legend = set()

    for (i0, i1) in segments:
        seg = data.iloc[i0:i1+1]
        lab = seg["_label_norm"].iloc[0]
        if (lab is None) and not show_unlabeled:
            continue

        if lab is None:
            style = unlabeled_style
            leg_text = "Unlabeled"
        else:
            style = dict(color=auto_colors.get(lab, None), linestyle="-", linewidth=2.5)
            leg_text = str(lab)

        label_kw = leg_text if leg_text not in seen_for_legend else None
        if label_kw is not None:
            seen_for_legend.add(leg_text)

        ax.plot(seg[x_col].values, seg[y_col].values, marker=None, label=label_kw, **style)

    # start_row = data.iloc[0]
    # end_row = data.iloc[-1]
    # ax.scatter([start_row[x_col]], [start_row[y_col]], s=100, facecolors="none", edgecolors="black", linewidths=2, marker="o", label="Start")
    # ax.scatter([end_row[x_col]], [end_row[y_col]], s=100, c="black", marker="x", linewidths=2, label="End")

    ax.set_title(title)
    ax.set_xlabel("X position (pixels)")
    ax.set_ylabel("Y position (pixels)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="upper left",          # anchor point of the legend box itself
        bbox_to_anchor=(1.02, 1),  # position relative to the axes (1.02 → just outside right)
        borderaxespad=0,           # small padding
        frameon=True               # optional: show border
    )
    plt.tight_layout()
    return fig, ax



video_id = "44566106"  # example
track_path = f"{track_dir}/{video_id_exp}.parquet"
ann_path   =  f"{annotation_dir}/{video_id_exp}.parquet"


# === Load & Plot from Parquet (MABe schema) ===
TRACK_PQ = track_path     # path to  tracking parquet
ANN_PQ   = ann_path         # path to  annotation parquet
MOUSE_ID = 3                               # which mouse to visualize
ROLE     = "agent"                       # "target" | "agent" | "either"
BODYPART = "body_center"                  # which bodypart to use for (x,y)

# Load parquet tables (expects columns as provided in your message)
track = pd.read_parquet(TRACK_PQ)
ann   = pd.read_parquet(ANN_PQ)

# Filter to the selected mouse & bodypart, and standardize columns to [frame, x, y]
pos = (
    track[(track["mouse_id"] == MOUSE_ID) & (track["bodypart"] == BODYPART)]
        .rename(columns={"video_frame": "frame"})
        .loc[:, ["frame", "x", "y"]]
        .sort_values("frame")
        .drop_duplicates("frame")
        .reset_index(drop=True)
)

# Build frame-level labels from annotation intervals
if ROLE == "target":
    ann_role = ann[ann["target_id"] == MOUSE_ID].copy()
elif ROLE == "agent":
    ann_role = ann[ann["agent_id"] == MOUSE_ID].copy()
else:
    # either: mark if mouse appears as either target or agent, prefixing the role
    ann_role = ann[(ann["target_id"] == MOUSE_ID) | (ann["agent_id"] == MOUSE_ID)].copy()
    ann_role["action"] = ann_role.apply(
        lambda r: ("agent:" if r["agent_id"] == MOUSE_ID else "target:") + str(r["action"]), axis=1
    )

# Initialize labels as None (unannotated)
labels = pd.Series(index=pos["frame"].values, data=[None] * len(pos), dtype=object)

# Paint actions over their [start_frame, stop_frame] intervals (inclusive)
for _, r in ann_role.iterrows():
    start, stop = int(r["start_frame"]), int(r["stop_frame"])  # inclusive bounds
    action = str(r["action"])
    mask = (pos["frame"] >= start) & (pos["frame"] <= stop)
    # Later rows overwrite earlier ones if intervals overlap
    labels.loc[pos.loc[mask, "frame"]] = action

# Attach labels to positions
pos["label"] = pos["frame"].map(labels)

# Plot
fig, ax = plot_mouse_trajectory(
    pos,
    frame_col="frame",
    x_col="x",
    y_col="y",
    label_col="label",
    title=f"Mouse {MOUSE_ID} ({ROLE}) — {BODYPART}", 
    show_unlabeled = False
)
plt.show()


test_parquet = pd.read_parquet('/kaggle/input/MABe-mouse-behavior-detection/test_tracking/AdaptableSnail/438887472.parquet')
test_parquet                             


test_sample = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')
test_sample


sample = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/sample_submission.csv')
sample




