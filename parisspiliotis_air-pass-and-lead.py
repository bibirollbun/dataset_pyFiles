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

This project introduces the Anticipation Index, a play-level metric that quantifies quarterback anticipation on downfield passes using the NFL’s tracking data. Rather than treating anticipation as a vague “throws before the break” idea, we decompose it into four interpretable components measured at the instant of ball release and over the ball’s flight.

First, Lead Time Advantage compares how long the ball will be in the air to how long it would take the targeted receiver, at their current speed and location, to reach the eventual landing point. Second, Route Movement Predictability captures how volatile the receiver’s direction and speed are in the first few frames after release, rewarding throws made while the route is still unstable. Third, Aim Accuracy to Future Position measures how close the ball’s landing point is to the receiver’s projected future position, assuming constant velocity. Finally, Defensive Contest Pressure summarizes how much closing coverage is present near the landing location at throw time.

We combine these components into a single 0–100 Anticipation Index, then derive an Anticipation Over Expected value by conditioning on receiver–to–landing distance at release. The metric is visualized with triangle plots, where each throw is a triangle connecting the quarterback, receiver at release, and landing point, colored by anticipation. We use these tools to compare quarterbacks, identify difficult receivers to anticipate, and highlight timing throws such as deep boundary and seam routes that demand early, accurate ball placement. The framework is simple enough for broadcast graphics yet rich enough to support deeper coaching and front office evaluation.

import pandas as pd
import numpy as np
import math

FRAME_RATE = 10.0  # frames per second, tracking is 10 Hz

def compute_aindex_for_play(play_df):
    # same ball landing / frames for all rows
    ball_x = play_df['ball_land_x'].iloc[0]
    ball_y = play_df['ball_land_y'].iloc[0]
    num_frames = int(play_df['num_frames_output'].iloc[0])
    frame_dt = 1.0 / FRAME_RATE
    T_ball = num_frames * frame_dt

    # --- Targeted receiver ---
    wr_df = play_df[play_df['player_role'] == 'Targeted Receiver'].sort_values('frame_id')
    if wr_df.empty:
        return None

    wr0 = wr_df.iloc[0]
    dist_to_land = math.hypot(wr0['x'] - ball_x, wr0['y'] - ball_y)
    s = max(wr0['s'], 0.1)              # speed in yards/s
    T_proj = dist_to_land / s           # time for WR to reach landing spot at current speed
    LTA = T_proj - T_ball               # >0 = earlier throw
    LTA_norm = 0.5 * (math.tanh(LTA / 0.7) + 1)   # map to [0,1]

    # --- Receiver movement volatility (difficulty) ---
    k = min(5, len(wr_df))
    sub = wr_df.iloc[:k]
    dirs = np.deg2rad(sub['dir'].to_numpy())
    mean_sin, mean_cos = np.sin(dirs).mean(), np.cos(dirs).mean()
    circ_var = 1 - math.hypot(mean_sin, mean_cos)
    speed_var = sub['s'].var() if k > 1 else 0.0
    U = max(0.0, min(1.0, 0.5 * circ_var + 0.02 * speed_var))  # 0–1 difficulty
    RMP_difficulty = U

    # --- Aim Accuracy to Future Position ---
    theta = math.radians(wr0['dir'])
    vx, vy = s * math.cos(theta), s * math.sin(theta)
    x_pred = wr0['x'] + vx * T_ball
    y_pred = wr0['y'] + vy * T_ball
    pred_err = math.hypot(x_pred - ball_x, y_pred - ball_y)
    AAFP = math.exp(-pred_err / 10.0)   # 0–1 (10-yd scale)

    # --- Defensive Contest Pressure ---
    first_frame = play_df['frame_id'].min()
    db_df = play_df[(play_df['player_side'] == 'defense') &
                    (play_df['frame_id'] == first_frame)]
    if db_df.empty:
        DCP = 0.0
    else:
        dists = np.hypot(db_df['x'] - ball_x, db_df['y'] - ball_y)
        DCP = float(np.exp(-dists / 7.0).mean())  # 0–1

    # --- Final Anticipation Index ---
    A_index = (0.25 * LTA_norm +
               0.30 * RMP_difficulty +
               0.30 * AAFP +
               0.15 * DCP)

    return {
        "LTA_norm": LTA_norm,
        "RMP_diff": RMP_difficulty,
        "AAFP": AAFP,
        "DCP": DCP,
        "A_index_raw": A_index,
        "A_index_100": 100 * A_index,
    }

***Anticipation Index for one play***

import math
import numpy as np
import pandas as pd

FRAME_RATE = 10.0          # tracking is 10 Hz
DT = 1.0 / FRAME_RATE

def compute_aindex_for_play(play_df):
    """
    play_df: tracking for a single (game_id, play_id), all players, frames 1..N
    returns dict with components + final A-Index, or None if cannot compute.
    """

    play_df = play_df.sort_values("frame_id")

    # shared info
    ball_x = play_df["ball_land_x"].iloc[0]
    ball_y = play_df["ball_land_y"].iloc[0]
    num_frames = int(play_df["num_frames_output"].iloc[0])
    T_ball = num_frames * DT

    # --- Targeted Receiver at/after throw ---
    wr_df = play_df[play_df["player_role"] == "Targeted Receiver"].sort_values("frame_id")
    if wr_df.empty:
        return None

    wr0 = wr_df.iloc[0]   # frame_id == 1 → time of throw

    # distance from WR at throw to ball landing
    dist_to_land = math.hypot(wr0["x"] - ball_x, wr0["y"] - ball_y)

    # speed at throw (avoid divide-by-zero)
    s = max(wr0["s"], 0.1)            # yards / second

    # ---------- (1) Lead Time Advantage ----------
    T_proj = dist_to_land / s         # time WR would need at current speed
    LTA = T_proj - T_ball             # >0 = early throw, <0 = late

    # squash into [0,1] smoothly
    LTA_norm = 0.5 * (math.tanh(LTA / 0.7) + 1.0)

    # ---------- (2) Receiver Movement Predictability / Difficulty ----------
    # take first K frames after throw
    K = min(5, len(wr_df))
    sub = wr_df.iloc[:K]

    dirs_rad = np.deg2rad(sub["dir"].to_numpy())
    mean_sin, mean_cos = np.sin(dirs_rad).mean(), np.cos(dirs_rad).mean()
    circ_var = 1 - math.hypot(mean_sin, mean_cos)    # 0 = very stable, 1 = very wiggly
    speed_var = sub["s"].var() if K > 1 else 0.0

    # simple normalized difficulty: high = hard to anticipate
    RMP_difficulty = max(0.0, min(1.0, 0.5 * circ_var + 0.02 * speed_var))

    # ---------- (3) Aim Accuracy to Future Position ----------
    theta = math.radians(wr0["dir"])
    vx, vy = s * math.cos(theta), s * math.sin(theta)
    x_pred = wr0["x"] + vx * T_ball
    y_pred = wr0["y"] + vy * T_ball

    pred_err = math.hypot(x_pred - ball_x, y_pred - ball_y)    # yards
    AAFP = math.exp(-pred_err / 10.0)                          # 0–1 (10-yd scale)

    # ---------- (4) Defensive Contest Pressure ----------
    first_frame = play_df["frame_id"].min()
    db_df = play_df[(play_df["player_side"] == "defense") &
                    (play_df["frame_id"] == first_frame)]

    if db_df.empty:
        DCP = 0.0
    else:
        dists = np.hypot(db_df["x"] - ball_x, db_df["y"] - ball_y)
        DCP = float(np.exp(-dists / 7.0).mean())    # 0–1

    # ---------- Final Anticipation Index (0–1, then scale to 0–100) ----------
    A_index = (0.25 * LTA_norm +
               0.30 * RMP_difficulty +
               0.30 * AAFP +
               0.15 * DCP)

    return {
        "LTA_norm": LTA_norm,
        "RMP_diff": RMP_difficulty,
        "AAFP": AAFP,
        "DCP": DCP,
        "A_index_raw": A_index,
        "A_index_100": 100 * A_index,
    }
***Loop over all weeks / games / plays***

import os
import zipfile
from pathlib import Path

# If you extracted the zip:
DATA_DIR = Path("114239_nfl_competition_files_published_analytics_final")

# If you kept it zipped, you can adapt this to read from zipfile instead.

weeks = range(1, 19)

all_rows = []

for w in weeks:
    file_path = DATA_DIR / "train" / f"input_2023_w{w:02d}.csv"
    print(f"Processing week {w} → {file_path}")

    week_df = pd.read_csv(file_path)

    # group by play
    for (game_id, play_id), g in week_df.groupby(["game_id", "play_id"]):
        res = compute_aindex_for_play(g)
        if res is None:
            continue

        # identify QB & Targeted WR at frame 1 for labeling
        at_throw = g[g["frame_id"] == g["frame_id"].min()]

        qb = at_throw[at_throw["player_role"] == "Passer"]
        wr = at_throw[at_throw["player_role"] == "Targeted Receiver"]

        qb_name = qb["player_name"].iloc[0] if not qb.empty else None
        qb_id   = qb["nfl_id"].iloc[0]      if not qb.empty else None

        wr_name = wr["player_name"].iloc[0] if not wr.empty else None
        wr_id   = wr["nfl_id"].iloc[0]      if not wr.empty else None

        row = {
            "week": w,
            "game_id": game_id,
            "play_id": play_id,
            "qb_name": qb_name,
            "qb_id": qb_id,
            "wr_name": wr_name,
            "wr_id": wr_id,
        }
        row.update(res)
        all_rows.append(row)

aindex_df = pd.DataFrame(all_rows)
print(aindex_df.head())

***Per-QB and per-WR leaderboards***

# Filter for a reasonable minimum number of attempts / targets
MIN_ATTEMPTS = 30

qb_leaders = (
    aindex_df
    .groupby(["qb_id", "qb_name"], dropna=True)["A_index_100"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "AIndex_mean", "count": "attempts"})
    .query("attempts >= @MIN_ATTEMPTS")
    .sort_values("AIndex_mean", ascending=False)
)

wr_leaders = (
    aindex_df
    .groupby(["wr_id", "wr_name"], dropna=True)["A_index_100"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "AIndex_mean", "count": "targets"})
    .query("targets >= @MIN_ATTEMPTS")
    .sort_values("AIndex_mean", ascending=False)
)

print("Top QBs by Anticipation Index:")
print(qb_leaders.head(10))

print("\nTop WRs by Anticipation Index on their targets:")
print(wr_leaders.head(10))

***Plot Triangles for one game or for all plays of a QB***

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

def plot_triangles_for_qb_week(week_df, aindex_week_df, qb_name_filter=None):
    """
    week_df: tracking for a single week (input_2023_wXX)
    aindex_week_df: subset of aindex_df for that week
    qb_name_filter: show only plays for this QB (optional)
    """

    poly_verts = []
    colors = []

    # merge per-play metric back into week_df groups
    metrics_by_key = aindex_week_df.set_index(["game_id", "play_id"])

    for (game_id, play_id), g in week_df.groupby(["game_id", "play_id"]):
        key = (game_id, play_id)
        if key not in metrics_by_key.index:
            continue

        met = metrics_by_key.loc[key]

        # optional QB filter
        if qb_name_filter is not None and met["qb_name"] != qb_name_filter:
            continue

        xs, ys = get_triangle_for_play(g)
        if xs is None:
            continue

        poly_verts.append(list(zip(xs, ys)))
        colors.append(met["A_index_100"])   # color by anticipation index

    if not poly_verts:
        print("No triangles to plot for this selection.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    coll = PolyCollection(poly_verts, array=np.array(colors),
                          cmap="viridis", alpha=0.7)
    ax.add_collection(coll)

    ax.set_xlim(0, 120)      # field length in yards (approx)
    ax.set_ylim(0, 53.3)     # field width
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Field X (yards)")
    ax.set_ylabel("Field Y (yards)")
    cbar = fig.colorbar(coll, ax=ax)
    cbar.set_label("Anticipation Index (0–100)")

    title = "Anticipation Triangles"
    if qb_name_filter:
        title += f" – {qb_name_filter}"
    ax.set_title(title)

    plt.show()

week = 1
week_input = pd.read_csv(DATA_DIR / "train" / f"input_2023_w{week:02d}.csv")
week_metrics = aindex_df[aindex_df["week"] == week]

# all plays in week 1
plot_triangles_for_qb_week(week_input, week_metrics)

# or just a specific QB
plot_triangles_for_qb_week(week_input, week_metrics, qb_name_filter="Patrick Mahomes")

**Graphs and others***

import math, numpy as np

FRAME_RATE=10.0
DT=1/FRAME_RATE

def compute_aindex_for_play(play_df):
    play_df=play_df.sort_values("frame_id")
    ball_x=play_df["ball_land_x"].iloc[0]
    ball_y=play_df["ball_land_y"].iloc[0]
    num_frames=int(play_df["num_frames_output"].iloc[0])
    T_ball=num_frames*DT
    wr_df=play_df[play_df["player_role"]=="Targeted Receiver"].sort_values("frame_id")
    if wr_df.empty: return None
    wr0=wr_df.iloc[0]
    dist_to_land=math.hypot(wr0["x"]-ball_x, wr0["y"]-ball_y)
    s=max(wr0["s"],0.1)
    T_proj=dist_to_land/s
    LTA_norm=0.5*(math.tanh((T_proj-T_ball)/0.7)+1)
    K=min(5,len(wr_df))
    sub=wr_df.iloc[:K]
    dirs_rad=np.deg2rad(sub["dir"])
    circ_var=1-math.hypot(np.sin(dirs_rad).mean(), np.cos(dirs_rad).mean())
    speed_var=sub["s"].var() if K>1 else 0
    RMP=max(0,min(1,0.5*circ_var+0.02*(speed_var if speed_var==speed_var else 0)))
    theta=math.radians(wr0["dir"])
    vx,vy=s*math.cos(theta),s*math.sin(theta)
    x_pred=wr0["x"]+vx*T_ball
    y_pred=wr0["y"]+vy*T_ball
    pred_err=math.hypot(x_pred-ball_x,y_pred-ball_y)
    AAFP=math.exp(-pred_err/10)
    first_frame=play_df["frame_id"].min()
    db_df=play_df[(play_df["player_side"]=="defense")&(play_df["frame_id"]==first_frame)]
    if db_df.empty:
        DCP=0
    else:
        dists=np.hypot(db_df["x"]-ball_x, db_df["y"]-ball_y)
        DCP=float(np.exp(-dists/7).mean())
    A=0.35*LTA_norm+0.25*RMP+0.25*AAFP+0.15*DCP
    return A

results=[]
for (gid,pid),g in week1.groupby(["game_id","play_id"]):
    A=compute_aindex_for_play(g)
    if A is not None:
        results.append({"game_id":gid,"play_id":pid,"A_index_100":A*100})

len(results)

import matplotlib.pyplot as plt
import pandas as pd

df=pd.DataFrame(results)
plt.figure(figsize=(8,5))
plt.hist(df["A_index_100"], bins=30)
plt.xlabel("A-index (0-100)")
plt.ylabel("Frequency")
plt.title("Distribution of Anticipation Index - Week 1")
plt.show()

import zipfile, pandas as pd, numpy as np, math
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# Load week1
zip_path='/mnt/data/nfl-big-data-bowl-2026-analytics (1).zip'
with zipfile.ZipFile(zip_path) as z:
    with z.open('114239_nfl_competition_files_published_analytics_final/train/input_2023_w01.csv') as f:
        week1=pd.read_csv(f)

FRAME_RATE=10.0
DT=1/FRAME_RATE

def compute_components(play_df):
    play_df=play_df.sort_values("frame_id")
    ball_x=play_df["ball_land_x"].iloc[0]
    ball_y=play_df["ball_land_y"].iloc[0]
    num_frames=int(play_df["num_frames_output"].iloc[0])
    T_ball=num_frames*DT
    wr_df=play_df[play_df["player_role"]=="Targeted Receiver"].sort_values("frame_id")
    if wr_df.empty: return None
    wr0=wr_df.iloc[0]
    dist_to_land=math.hypot(wr0["x"]-ball_x, wr0["y"]-ball_y)
    s=max(wr0["s"],0.1)
    T_proj=dist_to_land/s
    LTA_norm=0.5*(math.tanh((T_proj-T_ball)/0.7)+1)
    K=min(5,len(wr_df))
    sub=wr_df.iloc[:K]
    dirs_rad=np.deg2rad(sub["dir"])
    circ_var=1-math.hypot(np.sin(dirs_rad).mean(), np.cos(dirs_rad).mean())
    speed_var=sub["s"].var() if K>1 else 0
    RMP=max(0,min(1,0.5*circ_var+0.02*(speed_var if speed_var==speed_var else 0)))
    theta=math.radians(wr0["dir"])
    vx,vy=s*math.cos(theta),s*math.sin(theta)
    x_pred=wr0["x"]+vx*T_ball
    y_pred=wr0["y"]+vy*T_ball
    pred_err=math.hypot(x_pred-ball_x,y_pred-ball_y)
    AAFP=math.exp(-pred_err/10)
    first_frame=play_df["frame_id"].min()
    db_df=play_df[(play_df["player_side"]=="defense")&(play_df["frame_id"]==first_frame)]
    if db_df.empty:
        DCP=0
    else:
        dists=np.hypot(db_df["x"]-ball_x, db_df["y"]-ball_y)
        DCP=float(np.exp(-dists/7).mean())
    A=0.35*LTA_norm+0.25*RMP+0.25*AAFP+0.15*DCP
    return LTA_norm,RMP,AAFP,DCP,A*100

records=[]
for (gid,pid),g in week1.groupby(["game_id","play_id"]):
    comp=compute_components(g)
    if comp:
        L,R,Aa,D,Ai=comp
        # qb
        first= g[g["frame_id"]==g["frame_id"].min()]
        qb=first[first["player_role"]=="Passer"]
        qb_name= qb["player_name"].iloc[0] if not qb.empty else None
        records.append({"game_id":gid,"play_id":pid,"qb_name":qb_name,
                        "LTA":L,"RMP":R,"AAFP":Aa,"DCP":D,"A_index":Ai})

df=pd.DataFrame(records)

# 1 Histogram already done - now triangle plot
# function to get triangle
def get_triangle(g):
    g=g.sort_values("frame_id")
    first_frame=g["frame_id"].min()
    at_throw=g[g["frame_id"]==first_frame]
    qb=at_throw[at_throw["player_role"]=="Passer"]
    wr=at_throw[at_throw["player_role"]=="Targeted Receiver"]
    if qb.empty or wr.empty: return None
    qb=qb.iloc[0]; wr=wr.iloc[0]
    ball_x=g["ball_land_x"].iloc[0]; ball_y=g["ball_land_y"].iloc[0]
    xs=[qb["x"],wr["x"],ball_x]; ys=[qb["y"],wr["y"],ball_y]
    return xs,ys

# Sample triangles (limit to first 200 plays for performance)
week_gb = week1.groupby(["game_id","play_id"])
poly=[]; cols=[]
for idx,(gid,pid) in enumerate(week_gb.groups):
    if idx>200: break
    g=week_gb.get_group((gid,pid))
    tri=get_triangle(g)
    if tri:
        xs,ys=tri
        poly.append(list(zip(xs,ys)))
        Ai=df[(df.game_id==gid)&(df.play_id==pid)].A_index.values[0]
        cols.append(Ai)

fig,ax=plt.subplots(figsize=(10,6))
coll=PolyCollection(poly,array=np.array(cols),cmap="viridis",alpha=0.7)
ax.add_collection(coll)
ax.set_xlim(0,120); ax.set_ylim(0,53.3); ax.set_aspect("equal")
plt.colorbar(coll,ax=ax,label="A-index (0-100)")
ax.set_title("Triangle Anticipation Plot (Sample 200 plays)")
plt.show()

# 2 Top 10 QBs
top_qb = df.groupby("qb_name").A_index.mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(8,6))
top_qb.plot.bar(color='teal')
plt.ylabel("Average A-index")
plt.title("Top 10 QBs by Anticipation Index (Week1)")
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()

# 3 Scatterplot lead distance vs A-index (lead distance= LTA proxy)
plt.figure(figsize=(7,5))
plt.scatter(df["LTA"],df["A_index"],alpha=0.4)
plt.xlabel("LTA_norm")
plt.ylabel("A-index")
plt.title("Lead Time Advantage vs Anticipation Index")
plt.show()

# 4 Heatmap of WR predictability (RMP)
plt.figure(figsize=(7,5))
plt.hist(df["RMP"],bins=20,color='purple',alpha=0.7)
plt.xlabel("WR Predictability (RMP difficulty)")
plt.ylabel("Count")
plt.title("Distribution of WR Route Volatility")
plt.show()

import zipfile, pandas as pd, numpy as np, math
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# Reload and recompute to ensure state
zip_path='/mnt/data/nfl-big-data-bowl-2026-analytics (1).zip'
with zipfile.ZipFile(zip_path) as z:
    with z.open('114239_nfl_competition_files_published_analytics_final/train/input_2023_w01.csv') as f:
        week1=pd.read_csv(f)

FRAME_RATE=10.0
DT=1/FRAME_RATE

def compute_components(play_df):
    play_df=play_df.sort_values("frame_id")
    ball_x=play_df["ball_land_x"].iloc[0]
    ball_y=play_df["ball_land_y"].iloc[0]
    num_frames=int(play_df["num_frames_output"].iloc[0])
    T_ball=num_frames*DT
    wr_df=play_df[play_df["player_role"]=="Targeted Receiver"].sort_values("frame_id")
    if wr_df.empty: return None
    wr0=wr_df.iloc[0]
    dist_to_land=math.hypot(wr0["x"]-ball_x, wr0["y"]-ball_y)
    s=max(wr0["s"],0.1)
    T_proj=dist_to_land/s
    LTA_norm=0.5*(math.tanh((T_proj-T_ball)/0.7)+1)
    K=min(5,len(wr_df))
    sub=wr_df.iloc[:K]
    dirs_rad=np.deg2rad(sub["dir"])
    circ_var=1-math.hypot(np.sin(dirs_rad).mean(), np.cos(dirs_rad).mean())
    speed_var=sub["s"].var() if K>1 else 0
    RMP=max(0,min(1,0.5*circ_var+0.02*(speed_var if speed_var==speed_var else 0)))
    theta=math.radians(wr0["dir"])
    vx,vy=s*math.cos(theta),s*math.sin(theta)
    x_pred=wr0["x"]+vx*T_ball
    y_pred=wr0["y"]+vy*T_ball
    pred_err=math.hypot(x_pred-ball_x,y_pred-ball_y)
    AAFP=math.exp(-pred_err/10)
    first_frame=play_df["frame_id"].min()
    db_df=play_df[(play_df["player_side"]=="defense")&(play_df["frame_id"]==first_frame)]
    if db_df.empty:
        DCP=0
    else:
        dists=np.hypot(db_df["x"]-ball_x, db_df["y"]-ball_y)
        DCP=float(np.exp(-dists/7).mean())
    A=0.35*LTA_norm+0.25*RMP+0.25*AAFP+0.15*DCP
    return dict(LTA=LTA_norm,RMP=RMP,AAFP=AAFP,DCP=DCP,A_index=A*100,
                lead_dist=dist_to_land)

records=[]
week_gb = week1.groupby(["game_id","play_id"])
for (gid,pid),g in week_gb:
    comp=compute_components(g)
    if comp:
        first=g[g["frame_id"]==g["frame_id"].min()]
        qb=first[first["player_role"]=="Passer"]
        qb_name=qb["player_name"].iloc[0] if not qb.empty else None
        wr=first[first["player_role"]=="Targeted Receiver"]
        wr_name=wr["player_name"].iloc[0] if not wr.empty else None
        rec={"game_id":gid,"play_id":pid,"qb_name":qb_name,"wr_name":wr_name}
        rec.update(comp)
        records.append(rec)

df=pd.DataFrame(records)
len(df), df.head()

from matplotlib.collections import PolyCollection

def get_triangle(g):
    g=g.sort_values("frame_id")
    first_frame=g["frame_id"].min()
    at_throw=g[g["frame_id"]==first_frame]
    qb=at_throw[at_throw["player_role"]=="Passer"]
    wr=at_throw[at_throw["player_role"]=="Targeted Receiver"]
    if qb.empty or wr.empty: return None
    qb=qb.iloc[0]; wr=wr.iloc[0]
    ball_x=g["ball_land_x"].iloc[0]; ball_y=g["ball_land_y"].iloc[0]
    xs=[qb["x"],wr["x"],ball_x]; ys=[qb["y"],wr["y"],ball_y]
    return xs,ys

# 1) full-week triangle map
poly=[]; cols=[]
for (gid,pid),g in week_gb:
    tri=get_triangle(g)
    if not tri: 
        continue
    xs,ys=tri
    poly.append(list(zip(xs,ys)))
    cols.append(df[(df.game_id==gid)&(df.play_id==pid)].A_index.values[0])

fig,ax=plt.subplots(figsize=(10,6))
coll=PolyCollection(poly,array=np.array(cols),cmap="viridis",alpha=0.7)
ax.add_collection(coll)
ax.set_xlim(0,120); ax.set_ylim(0,53.3); ax.set_aspect("equal")
plt.colorbar(coll,ax=ax,label="A-index (0-100)")
ax.set_title("Full Week 1 Triangle Anticipation Map (All 819 plays)")
plt.show()

# 2) Player-specific maps for Mahomes + top 10 QBs by mean A_index
top10 = df.groupby("qb_name").A_index.mean().sort_values(ascending=False).head(10)
top10_qbs = list(top10.index)
if "Patrick Mahomes" not in top10_qbs and "Patrick Mahomes" in df.qb_name.unique():
    top10_qbs.insert(0,"Patrick Mahomes")

top10_qbs

# Player-specific triangle maps
for qb in top10_qbs:
    poly=[]; cols=[]
    for (gid,pid),g in week_gb:
        row = df[(df.game_id==gid)&(df.play_id==pid)]
        if row.empty or row.qb_name.iloc[0]!=qb:
            continue
        tri=get_triangle(g)
        if not tri: 
            continue
        xs,ys=tri
        poly.append(list(zip(xs,ys)))
        cols.append(row.A_index.iloc[0])
    if not poly:
        continue
    fig,ax=plt.subplots(figsize=(8,5))
    coll=PolyCollection(poly,array=np.array(cols),cmap="viridis",alpha=0.8)
    ax.add_collection(coll)
    ax.set_xlim(0,120); ax.set_ylim(0,53.3); ax.set_aspect("equal")
    plt.colorbar(coll,ax=ax,label="A-index (0-100)")
    ax.set_title(f"Anticipation Triangle Map – {qb} (Week 1)")
    plt.show()

# 3) WR anticipation difficulty rankings (by mean RMP)
wr_stats = df.groupby("wr_name").agg(
    mean_RMP=("RMP","mean"),
    targets=("A_index","count"),
    mean_Aindex=("A_index","mean")
).reset_index()
wr_stats = wr_stats[wr_stats["targets"]>=5].sort_values("mean_RMP",ascending=False)

top_wr_diff = wr_stats.head(15)

plt.figure(figsize=(8,6))
plt.barh(top_wr_diff["wr_name"], top_wr_diff["mean_RMP"])
plt.gca().invert_yaxis()
plt.xlabel("Average Route Volatility (RMP difficulty)")
plt.title("Most Difficult WRs to Anticipate – Week 1 (min 5 targets)")
plt.tight_layout()
plt.show()

top_wr_diff

# Recompute DBBI with a simpler approach and ensure column exists
def compute_db_metrics(play_df):
    play_df=play_df.sort_values("frame_id")
    ball_x=play_df["ball_land_x"].iloc[0]
    ball_y=play_df["ball_land_y"].iloc[0]
    first=play_df["frame_id"].min()
    last=min(first+3, play_df["frame_id"].max())
    dbs=play_df[(play_df["player_side"]=="defense") &
                (play_df["player_role"].str.contains("Coverage"))]
    bite_scores=[]
    for nid, g in dbs.groupby("nfl_id"):
        g=g[(g["frame_id"]>=first)&(g["frame_id"]<=last)].sort_values("frame_id")
        if len(g)<2: 
            continue
        dir_start=np.deg2rad(g["dir"].iloc[0])
        dir_end=np.deg2rad(g["dir"].iloc[-1])
        turn=abs(np.arctan2(np.sin(dir_end-dir_start), np.cos(dir_end-dir_start)))
        x0,y0=g["x"].iloc[0],g["y"].iloc[0]
        x1,y1=g["x"].iloc[-1],g["y"].iloc[-1]
        d0=np.hypot(x0-ball_x,y0-ball_y)
        d1=np.hypot(x1-ball_x,y1-ball_y)
        toward = max(0,d0-d1)
        bite_scores.append(float(turn*toward))
    if not bite_scores: 
        return None
    return float(np.mean(bite_scores))

db_records=[]
for (gid,pid),g in week_gb:
    val=compute_db_metrics(g)
    if val is not None:
        db_records.append({"game_id":gid,"play_id":pid,"DBBI_val":val})
dbdf=pd.DataFrame(db_records)

if not dbdf.empty:
    dbdf["DBBI_norm"]= (dbdf["DBBI_val"]-dbdf["DBBI_val"].min())/(dbdf["DBBI_val"].max()-dbdf["DBBI_val"].min())

    plt.figure(figsize=(7,5))
    plt.hist(dbdf["DBBI_norm"],bins=30)
    plt.xlabel("DB Bite Index (normalized)")
    plt.ylabel("Count")
    plt.title("Distribution of Defensive Bite Index – Week 1")
    plt.show()

dbdf.head()

week1["player_role"].unique()

# adjust filter to exact match 'Defensive Coverage'
def compute_db_metrics(play_df):
    play_df=play_df.sort_values("frame_id")
    ball_x=play_df["ball_land_x"].iloc[0]
    ball_y=play_df["ball_land_y"].iloc[0]
    first=play_df["frame_id"].min()
    last=min(first+3, play_df["frame_id"].max())
    dbs=play_df[play_df["player_role"]=="Defensive Coverage"]
    bite_scores=[]
    for nid, g in dbs.groupby("nfl_id"):
        g=g[(g["frame_id"]>=first)&(g["frame_id"]<=last)].sort_values("frame_id")
        if len(g)<2: 
            continue
        dir_start=np.deg2rad(g["dir"].iloc[0])
        dir_end=np.deg2rad(g["dir"].iloc[-1])
        turn=abs(np.arctan2(np.sin(dir_end-dir_start), np.cos(dir_end-dir_start)))
        x0,y0=g["x"].iloc[0],g["y"].iloc[0]
        x1,y1=g["x"].iloc[-1],g["y"].iloc[-1]
        d0=np.hypot(x0-ball_x,y0-ball_y)
        d1=np.hypot(x1-ball_x,y1-ball_y)
        toward = max(0,d0-d1)
        bite_scores.append(float(turn*toward))
    if not bite_scores: 
        return None
    return float(np.mean(bite_scores))

db_records=[]
for (gid,pid),g in week_gb:
    val=compute_db_metrics(g)
    if val is not None:
        db_records.append({"game_id":gid,"play_id":pid,"DBBI_val":val})
dbdf=pd.DataFrame(db_records)
dbdf.head(), len(dbdf)

dbdf["DBBI_norm"]=(dbdf["DBBI_val"]-dbdf["DBBI_val"].min())/(dbdf["DBBI_val"].max()-dbdf["DBBI_val"].min())

plt.figure(figsize=(7,5))
plt.hist(dbdf["DBBI_norm"],bins=30)
plt.xlabel("DB Bite Index (normalized)")
plt.ylabel("Count")
plt.title("Distribution of Defensive Bite Index – Week 1")
plt.show()

# 5) Simple A-index over Expected (A-SOE)
# We'll model expected A as function of lead_dist only (for simplicity, TV version)

# bin by lead distance
df["lead_bin"]=pd.cut(df["lead_dist"],bins=[0,5,10,15,20,30,50],include_lowest=True)
exp_by_bin=df.groupby("lead_bin").A_index.mean().rename("A_exp").reset_index()
df= df.merge(exp_by_bin,on="lead_bin",how="left")
df["A_SOE"]=df["A_index"]-df["A_exp"]  # positive = better than expected

# QB-level A_SOE
qb_asoe = df.groupby("qb_name").agg(
    A_index_mean=("A_index","mean"),
    A_SOE_mean=("A_SOE","mean"),
    attempts=("A_index","count")
).reset_index()
qb_asoe = qb_asoe[qb_asoe["attempts"]>=10].sort_values("A_SOE_mean",ascending=False)

top_asoe = qb_asoe.head(10)

plt.figure(figsize=(8,6))
plt.barh(top_asoe["qb_name"], top_asoe["A_SOE_mean"])
plt.gca().invert_yaxis()
plt.xlabel("A-index Over Expected (A-SOE)")
plt.title("Top QBs by Anticipation Over Expected – Week 1")
plt.tight_layout()
plt.show()

top_asoe

# 6) Simple animated play visualization for one Mahomes play
from matplotlib import animation

# pick one Mahomes play
mahomes_plays = df[df["qb_name"]=="Patrick Mahomes"][["game_id","play_id"]].iloc[0]
gid,pid = int(mahomes_plays.game_id), int(mahomes_plays.play_id)
play = week_gb.get_group((gid,pid)).sort_values("frame_id")

fig, ax = plt.subplots(figsize=(8,5))
ax.set_xlim(0,120); ax.set_ylim(0,53.3); ax.set_aspect("equal")
ax.set_title(f"Patrick Mahomes Anticipation – game {gid}, play {pid}")
qb_scatter, = ax.plot([], [], marker='o', linestyle='', label='QB')
wr_scatter, = ax.plot([], [], marker='o', linestyle='', label='Target WR')
db_scatter, = ax.plot([], [], marker='x', linestyle='', label='Defenders')
ball_scatter, = ax.plot([], [], marker='^', linestyle='', label='Ball')
ax.legend(loc='upper right')

frames_sorted = sorted(play["frame_id"].unique())

def init():
    qb_scatter.set_data([],[])
    wr_scatter.set_data([],[])
    db_scatter.set_data([],[])
    ball_scatter.set_data([],[])
    return qb_scatter, wr_scatter, db_scatter, ball_scatter

def animate(i):
    f=frames_sorted[i]
    fr=play[play["frame_id"]==f]
    qb=fr[fr["player_role"]=="Passer"]
    wr=fr[fr["player_role"]=="Targeted Receiver"]
    dbs=fr[fr["player_role"]=="Defensive Coverage"]
    qb_scatter.set_data(qb["x"],qb["y"])
    wr_scatter.set_data(wr["x"],wr["y"])
    db_scatter.set_data(dbs["x"],dbs["y"])
    # approximate ball location along line from qb at frame1 to landing
    ball_x = play["ball_land_x"].iloc[0]
    ball_y = play["ball_land_y"].iloc[0]
    qb0 = play[play["frame_id"]==frames_sorted[0]]
    qb0x,qb0y = qb0[qb0["player_role"]=="Passer"]["x"].iloc[0], qb0[qb0["player_role"]=="Passer"]["y"].iloc[0]
    t = i/len(frames_sorted)
    ball_scatter.set_data(qb0x + (ball_x-qb0x)*t, qb0y + (ball_y-qb0y)*t)
    return qb_scatter, wr_scatter, db_scatter, ball_scatter

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(frames_sorted), interval=200, blit=True)
plt.close(fig)
ani

import zipfile, pandas as pd, numpy as np, math
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from pathlib import Path

# Recreate data & metrics (to be sure everything is in scope)
zip_path = '/mnt/data/nfl-big-data-bowl-2026-analytics (1).zip'
with zipfile.ZipFile(zip_path) as z:
    with z.open('114239_nfl_competition_files_published_analytics_final/train/input_2023_w01.csv') as f:
        week1 = pd.read_csv(f)

FRAME_RATE = 10.0
DT = 1 / FRAME_RATE

def compute_components(play_df):
    play_df = play_df.sort_values("frame_id")
    ball_x = play_df["ball_land_x"].iloc[0]
    ball_y = play_df["ball_land_y"].iloc[0]
    num_frames = int(play_df["num_frames_output"].iloc[0])
    T_ball = num_frames * DT
    wr_df = play_df[play_df["player_role"]=="Targeted Receiver"].sort_values("frame_id")
    if wr_df.empty:
        return None
    wr0 = wr_df.iloc[0]
    dist_to_land = math.hypot(wr0["x"]-ball_x, wr0["y"]-ball_y)
    s = max(wr0["s"], 0.1)
    T_proj = dist_to_land / s
    LTA_norm = 0.5 * (math.tanh((T_proj - T_ball)/0.7) + 1)
    K = min(5, len(wr_df))
    sub = wr_df.iloc[:K]
    dirs_rad = np.deg2rad(sub["dir"])
    circ_var = 1 - math.hypot(np.sin(dirs_rad).mean(), np.cos(dirs_rad).mean())
    speed_var = sub["s"].var() if K > 1 else 0.0
    RMP = max(0, min(1, 0.5 * circ_var + 0.02 * (speed_var if speed_var == speed_var else 0)))
    theta = math.radians(wr0["dir"])
    vx, vy = s * math.cos(theta), s * math.sin(theta)
    x_pred = wr0["x"] + vx * T_ball
    y_pred = wr0["y"] + vy * T_ball
    pred_err = math.hypot(x_pred - ball_x, y_pred - ball_y)
    AAFP = math.exp(-pred_err/10)
    first_frame = play_df["frame_id"].min()
    db_df = play_df[(play_df["player_side"]=="defense") & (play_df["frame_id"]==first_frame)]
    if db_df.empty:
        DCP = 0.0
    else:
        dists = np.hypot(db_df["x"]-ball_x, db_df["y"]-ball_y)
        DCP = float(np.exp(-dists/7).mean())
    A = 0.25 * LTA_norm + 0.30 * RMP + 0.30 * AAFP + 0.15 * DCP
    return dict(LTA=LTA_norm, RMP=RMP, AAFP=AAFP, DCP=DCP, A_index=A*100, lead_dist=dist_to_land)

records = []
week_gb = week1.groupby(["game_id","play_id"])
for (gid,pid), g in week_gb:
    comp = compute_components(g)
    if comp:
        first = g[g["frame_id"]==g["frame_id"].min()]
        qb = first[first["player_role"]=="Passer"]
        wr = first[first["player_role"]=="Targeted Receiver"]
        rec = {
            "game_id": gid,
            "play_id": pid,
            "qb_name": qb["player_name"].iloc[0] if not qb.empty else None,
            "wr_name": wr["player_name"].iloc[0] if not wr.empty else None
        }
        rec.update(comp)
        records.append(rec)
df = pd.DataFrame(records)

# --- helper: triangle vertices ---
def get_triangle(g):
    g = g.sort_values("frame_id")
    first_frame = g["frame_id"].min()
    at_throw = g[g["frame_id"]==first_frame]
    qb = at_throw[at_throw["player_role"]=="Passer"]
    wr = at_throw[at_throw["player_role"]=="Targeted Receiver"]
    if qb.empty or wr.empty:
        return None
    qb = qb.iloc[0]; wr = wr.iloc[0]
    ball_x = g["ball_land_x"].iloc[0]; ball_y = g["ball_land_y"].iloc[0]
    xs = [qb["x"], wr["x"], ball_x]
    ys = [qb["y"], wr["y"], ball_y]
    return xs, ys

out_dir = Path("/mnt/data")

# 1) Mahomes triangle map
poly = []; cols = []
for (gid,pid), g in week_gb:
    row = df[(df.game_id==gid)&(df.play_id==pid)]
    if row.empty or row.qb_name.iloc[0] != "Patrick Mahomes":
        continue
    tri = get_triangle(g)
    if not tri:
        continue
    xs,ys = tri
    poly.append(list(zip(xs,ys)))
    cols.append(row.A_index.iloc[0])

fig, ax = plt.subplots(figsize=(10,6))
coll = PolyCollection(poly, array=np.array(cols), cmap="viridis", alpha=0.8)
ax.add_collection(coll)
ax.set_xlim(0,120); ax.set_ylim(0,53.3); ax.set_aspect("equal")
cbar = fig.colorbar(coll, ax=ax)
cbar.set_label("A-index (0–100)")
ax.set_title("Anticipation Triangle Map – Patrick Mahomes (Week 1)")
mahomes_path = out_dir / "mahomes_triangle_week1.png"
fig.savefig(mahomes_path, dpi=200, bbox_inches="tight")
plt.close(fig)

# 2) WR difficulty chart
wr_stats = df.groupby("wr_name").agg(
    mean_RMP=("RMP","mean"),
    targets=("A_index","count"),
    mean_Aindex=("A_index","mean")
).reset_index()
wr_stats = wr_stats[wr_stats["targets"]>=5].sort_values("mean_RMP",ascending=False)
top_wr_diff = wr_stats.head(15)

fig, ax = plt.subplots(figsize=(10,6))
ax.barh(top_wr_diff["wr_name"], top_wr_diff["mean_RMP"])
ax.invert_yaxis()
ax.set_xlabel("Average Route Volatility (RMP difficulty)")
ax.set_title("Most Difficult WRs to Anticipate – Week 1 (min 5 targets)")
wr_diff_path = out_dir / "wr_difficulty_week1.png"
fig.savefig(wr_diff_path, dpi=200, bbox_inches="tight")
plt.close(fig)

# 3) A-SOE (anticipation over expected) chart
df["lead_bin"]=pd.cut(df["lead_dist"],bins=[0,5,10,15,20,30,50],include_lowest=True)
exp_by_bin=df.groupby("lead_bin").A_index.mean().rename("A_exp").reset_index()
df= df.merge(exp_by_bin,on="lead_bin",how="left")
df["A_SOE"]=df["A_index"]-df["A_exp"]

qb_asoe = df.groupby("qb_name").agg(
    A_index_mean=("A_index","mean"),
    A_SOE_mean=("A_SOE","mean"),
    attempts=("A_index","count")
).reset_index()
qb_asoe = qb_asoe[qb_asoe["attempts"]>=10].sort_values("A_SOE_mean",ascending=False)
top_asoe = qb_asoe.head(10)

fig, ax = plt.subplots(figsize=(10,6))
ax.barh(top_asoe["qb_name"], top_asoe["A_SOE_mean"])
ax.invert_yaxis()
ax.set_xlabel("A-index Over Expected (A-SOE)")
ax.set_title("Top QBs by Anticipation Over Expected – Week 1")
asoe_path = out_dir / "qb_asoe_week1.png"
fig.savefig(asoe_path, dpi=200, bbox_inches="tight")
plt.close(fig)

# 4) Full-week triangle map
poly_all = []; cols_all = []
for (gid,pid), g in week_gb:
    tri = get_triangle(g)
    if not tri:
        continue
    xs,ys = tri
    poly_all.append(list(zip(xs,ys)))
    cols_all.append(df[(df.game_id==gid)&(df.play_id==pid)].A_index.values[0])

fig, ax = plt.subplots(figsize=(10,6))
coll = PolyCollection(poly_all, array=np.array(cols_all), cmap="viridis", alpha=0.7)
ax.add_collection(coll)
ax.set_xlim(0,120); ax.set_ylim(0,53.3); ax.set_aspect("equal")
cbar = fig.colorbar(coll, ax=ax)
cbar.set_label("A-index (0–100)")
ax.set_title("Full Week 1 Triangle Anticipation Map (All 819 plays)")
fullweek_path = out_dir / "fullweek_triangles_week1.png"
fig.savefig(fullweek_path, dpi=200, bbox_inches="tight")
plt.close(fig)

[str(mahomes_path), str(wr_diff_path), str(asoe_path), str(fullweek_path) ]

***Re-Run the full code***

import zipfile, pandas as pd, numpy as np, math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

zip_path = '/mnt/data/nfl-big-data-bowl-2026-analytics (1).zip'
base_prefix = '114239_nfl_competition_files_published_analytics_final/train/'

FRAME_RATE = 10.0
DT = 1 / FRAME_RATE

def compute_components_raw(play_df):
    """
    Return dict with raw LTA (un-normalized) and other components based on first frame after throw.
    """
    play_df = play_df.sort_values("frame_id")
    ball_x = play_df["ball_land_x"].iloc[0]
    ball_y = play_df["ball_land_y"].iloc[0]
    num_frames = int(play_df["num_frames_output"].iloc[0])
    T_ball = num_frames * DT

    wr_df = play_df[play_df["player_role"]=="Targeted Receiver"].sort_values("frame_id")
    if wr_df.empty:
        return None

    wr0 = wr_df.iloc[0]
    dist_to_land = math.hypot(wr0["x"]-ball_x, wr0["y"]-ball_y)
    s = max(wr0["s"], 0.1)

    T_proj = dist_to_land / s
    raw_LTA = T_proj - T_ball  # will be normalized globally later

    # route volatility
    K = min(5, len(wr_df))
    sub = wr_df.iloc[:K]
    dirs_rad = np.deg2rad(sub["dir"])
    circ_var = 1 - math.hypot(np.sin(dirs_rad).mean(), np.cos(dirs_rad).mean())
    speed_var = sub["s"].var() if K > 1 else 0.0
    RMP = max(0, min(1, 0.5 * circ_var + 0.02 * (speed_var if speed_var == speed_var else 0)))

    # future aim accuracy
    theta = math.radians(wr0["dir"])
    vx, vy = s * math.cos(theta), s * math.sin(theta)
    x_pred = wr0["x"] + vx * T_ball
    y_pred = wr0["y"] + vy * T_ball
    pred_err = math.hypot(x_pred - ball_x, y_pred - ball_y)
    AAFP = math.exp(-pred_err / 10.0)

    # defensive pressure
    first_frame = play_df["frame_id"].min()
    db_df = play_df[(play_df["player_side"]=="defense") & (play_df["frame_id"]==first_frame)]
    if db_df.empty:
        DCP = 0.0
    else:
        dists = np.hypot(db_df["x"]-ball_x, db_df["y"]-ball_y)
        DCP = float(np.exp(-dists / 7.0).mean())

    return dict(
        raw_LTA=raw_LTA,
        RMP=RMP,
        AAFP=AAFP,
        DCP=DCP,
        lead_dist=dist_to_land
    )

all_records = []
week_groups = {}  # to reuse later for triangles per QB if needed

with zipfile.ZipFile(zip_path) as z:
    for w in range(1, 19):
        fname = f'input_2023_w{w:02d}.csv'
        path_in_zip = base_prefix + fname
        try:
            with z.open(path_in_zip) as f:
                week_df = pd.read_csv(f)
        except KeyError:
            # no more weeks
            break

        gb = week_df.groupby(["game_id","play_id"])
        # keep reference to one week for triangles later if needed
        week_groups[w] = gb

        for (gid,pid), g in gb:
            comps = compute_components_raw(g)
            if not comps:
                continue
            first = g[g["frame_id"]==g["frame_id"].min()]
            qb = first[first["player_role"]=="Passer"]
            wr = first[first["player_role"]=="Targeted Receiver"]
            rec = {
                "week": w,
                "game_id": gid,
                "play_id": pid,
                "qb_name": qb["player_name"].iloc[0] if not qb.empty else None,
                "wr_name": wr["player_name"].iloc[0] if not wr.empty else None
            }
            rec.update(comps)
            all_records.append(rec)

len(all_records)

import zipfile, pandas as pd, numpy as np, math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

zip_path = '/mnt/data/nfl-big-data-bowl-2026-analytics (1).zip'
base_prefix = '114239_nfl_competition_files_published_analytics_final/train/'

FRAME_RATE = 10.0
DT = 1 / FRAME_RATE

def compute_components_raw(play_df):
    play_df = play_df.sort_values("frame_id")
    ball_x = play_df["ball_land_x"].iloc[0]
    ball_y = play_df["ball_land_y"].iloc[0]
    num_frames = int(play_df["num_frames_output"].iloc[0])
    T_ball = num_frames * DT

    wr_df = play_df[play_df["player_role"]=="Targeted Receiver"].sort_values("frame_id")
    if wr_df.empty:
        return None

    wr0 = wr_df.iloc[0]
    dist_to_land = math.hypot(wr0["x"]-ball_x, wr0["y"]-ball_y)
    s = max(wr0["s"], 0.1)

    T_proj = dist_to_land / s
    raw_LTA = T_proj - T_ball

    K = min(5, len(wr_df))
    sub = wr_df.iloc[:K]
    dirs_rad = np.deg2rad(sub["dir"])
    circ_var = 1 - math.hypot(np.sin(dirs_rad).mean(), np.cos(dirs_rad).mean())
    speed_var = sub["s"].var() if K > 1 else 0.0
    RMP = max(0, min(1, 0.5 * circ_var + 0.02 * (speed_var if speed_var == speed_var else 0)))

    theta = math.radians(wr0["dir"])
    vx, vy = s * math.cos(theta), s * math.sin(theta)
    x_pred = wr0["x"] + vx * T_ball
    y_pred = wr0["y"] + vy * T_ball
    pred_err = math.hypot(x_pred - ball_x, y_pred - ball_y)
    AAFP = math.exp(-pred_err / 10.0)

    first_frame = play_df["frame_id"].min()
    db_df = play_df[(play_df["player_side"]=="defense") & (play_df["frame_id"]==first_frame)]
    if db_df.empty:
        DCP = 0.0
    else:
        dists = np.hypot(db_df["x"]-ball_x, db_df["y"]-ball_y)
        DCP = float(np.exp(-dists / 7.0).mean())

    return dict(
        raw_LTA=raw_LTA,
        RMP=RMP,
        AAFP=AAFP,
        DCP=DCP,
        lead_dist=dist_to_land
    )

all_records = []
week_group_refs = {}

with zipfile.ZipFile(zip_path) as z:
    for w in range(1, 19):
        fname = f'input_2023_w{w:02d}.csv'
        path_in_zip = base_prefix + fname
        try:
            with z.open(path_in_zip) as f:
                week_df = pd.read_csv(f)
        except KeyError:
            break

        gb = week_df.groupby(["game_id","play_id"])
        week_group_refs[w] = week_df  # store df for potential later use

        for (gid,pid), g in gb:
            comps = compute_components_raw(g)
            if not comps:
                continue
            first = g[g["frame_id"]==g["frame_id"].min()]
            qb = first[first["player_role"]=="Passer"]
            wr = first[first["player_role"]=="Targeted Receiver"]
            rec = {
                "week": w,
                "game_id": gid,
                "play_id": pid,
                "qb_name": qb["player_name"].iloc[0] if not qb.empty else None,
                "wr_name": wr["player_name"].iloc[0] if not wr.empty else None
            }
            rec.update(comps)
            all_records.append(rec)

len(all_records)

*** NEW LTA***
raw_LTA = T_proj - T_ball

LTA_adj = 0.5 * (tanh( (raw_LTA - μ_LTA) / σ_LTA ) + 1)   # global normalization

A_index_v2 = 0.25 * LTA_adj
           + 0.30 * RMP
           + 0.30 * AAFP
           + 0.15 * DCP       # then ×100 for a 0–100 score

import zipfile, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# -------- CONFIG --------
ZIP_PATH = "nfl-big-data-bowl-2026-analytics (1).zip"
BASE_PREFIX = "114239_nfl_competition_files_published_analytics_final/train/"
FRAME_RATE = 10.0
DT = 1.0 / FRAME_RATE

# -------- 1. COMPONENTS PER PLAY (ALL WEEKS) --------

def compute_components_raw(play_df):
    """Return raw_LTA, RMP, AAFP, DCP, lead_dist for a single (game_id, play_id)."""
    play_df = play_df.sort_values("frame_id")

    ball_x = play_df["ball_land_x"].iloc[0]
    ball_y = play_df["ball_land_y"].iloc[0]
    num_frames = int(play_df["num_frames_output"].iloc[0])
    T_ball = num_frames * DT

    wr_df = play_df[play_df["player_role"] == "Targeted Receiver"].sort_values("frame_id")
    if wr_df.empty:
        return None

    wr0 = wr_df.iloc[0]
    dist_to_land = math.hypot(wr0["x"] - ball_x, wr0["y"] - ball_y)
    s = max(wr0["s"], 0.1)

    # raw LTA
    T_proj = dist_to_land / s
    raw_LTA = T_proj - T_ball

    # RMP (route volatility)
    K = min(5, len(wr_df))
    sub = wr_df.iloc[:K]
    dirs_rad = np.deg2rad(sub["dir"])
    # circular variance of direction
    circ_var = 1 - math.hypot(np.sin(dirs_rad).mean(), np.cos(dirs_rad).mean())
    speed_var = sub["s"].var() if K > 1 else 0.0
    RMP = max(0, min(1, 0.5 * circ_var + 0.02 * (speed_var if speed_var == speed_var else 0)))

    # AAFP (aim to future WR position)
    theta = math.radians(wr0["dir"])
    vx, vy = s * math.cos(theta), s * math.sin(theta)
    x_pred = wr0["x"] + vx * T_ball
    y_pred = wr0["y"] + vy * T_ball
    pred_err = math.hypot(x_pred - ball_x, y_pred - ball_y)
    AAFP = math.exp(-pred_err / 10.0)

    # DCP (defensive pressure)
    first_frame = play_df["frame_id"].min()
    db_df = play_df[
        (play_df["player_side"] == "defense")
        & (play_df["frame_id"] == first_frame)
    ]
    if db_df.empty:
        DCP = 0.0
    else:
        dists = np.hypot(db_df["x"] - ball_x, db_df["y"] - ball_y)
        DCP = float(np.exp(-dists / 7.0).mean())

    return dict(
        raw_LTA=raw_LTA,
        RMP=RMP,
        AAFP=AAFP,
        DCP=DCP,
        lead_dist=dist_to_land,
    )

all_records = []
week_frames = {}  # optional cache per week for triangle maps

with zipfile.ZipFile(ZIP_PATH) as z:
    for w in range(1, 19):
        fname = f"input_2023_w{w:02d}.csv"
        path_in_zip = BASE_PREFIX + fname
        try:
            with z.open(path_in_zip) as f:
                week_df = pd.read_csv(f)
        except KeyError:
            break   # no more weeks

        print(f"Week {w}: {len(week_df)} rows")

        # keep to reuse later if you want triangle maps per week
        week_frames[w] = week_df

        for (gid, pid), g in week_df.groupby(["game_id", "play_id"]):
            comps = compute_components_raw(g)
            if comps is None:
                continue
            first = g[g["frame_id"] == g["frame_id"].min()]
            qb = first[first["player_role"] == "Passer"]
            wr = first[first["player_role"] == "Targeted Receiver"]
            rec = {
                "week": w,
                "game_id": gid,
                "play_id": pid,
                "qb_name": qb["player_name"].iloc[0] if not qb.empty else None,
                "wr_name": wr["player_name"].iloc[0] if not wr.empty else None,
            }
            rec.update(comps)
            all_records.append(rec)

plays = pd.DataFrame(all_records)
print("Total plays:", len(plays))

# -------- 2. GLOBAL LTA NORMALIZATION + A-INDEX v2 --------

mu_LTA = plays["raw_LTA"].mean()
sigma_LTA = plays["raw_LTA"].std(ddof=0)

plays["LTA_z"] = (plays["raw_LTA"] - mu_LTA) / sigma_LTA
plays["LTA_adj"] = 0.5 * (np.tanh(plays["LTA_z"]) + 1.0)

plays["A_index"] = (
    0.25 * plays["LTA_adj"]
    + 0.30 * plays["RMP"]
    + 0.30 * plays["AAFP"]
    + 0.15 * plays["DCP"]
) * 100.0   # 0–100

# -------- 3. A-SOE (Over Expected by lead distance) --------

plays["lead_bin"] = pd.cut(
    plays["lead_dist"],
    bins=[0, 5, 10, 15, 20, 30, 50],
    include_lowest=True,
)

exp_by_bin = (
    plays.groupby("lead_bin")["A_index"]
    .mean()
    .rename("A_exp")
    .reset_index()
)

plays = plays.merge(exp_by_bin, on="lead_bin", how="left")
plays["A_SOE"] = plays["A_index"] - plays["A_exp"]

# Save the full per-play table for later analysis
plays.to_csv("plays_anticipation_all_weeks.csv", index=False)
print("Saved plays_anticipation_all_weeks.csv")

# -------- 4. QB & WR LEADERBOARDS --------

MIN_ATT = 50   # adjust as you like
qb_stats = (
    plays.groupby("qb_name")
    .agg(
        attempts=("A_index", "count"),
        A_index_mean=("A_index", "mean"),
        A_SOE_mean=("A_SOE", "mean"),
    )
    .reset_index()
)
qb_stats = qb_stats[qb_stats["attempts"] >= MIN_ATT]

# Top QBs by A-index
top_qb = qb_stats.sort_values("A_index_mean", ascending=False).head(10)
plt.figure(figsize=(8, 6))
plt.barh(top_qb["qb_name"], top_qb["A_index_mean"])
plt.gca().invert_yaxis()
plt.xlabel("Average Anticipation Index (0–100)")
plt.title("Top QBs by Anticipation Index – 2023 Season")
plt.tight_layout()
plt.savefig("qb_top10_Aindex_season.png", dpi=200)
plt.close()

# Top QBs by A-SOE
top_qb_asoe = qb_stats.sort_values("A_SOE_mean", ascending=False).head(10)
plt.figure(figsize=(8, 6))
plt.barh(top_qb_asoe["qb_name"], top_qb_asoe["A_SOE_mean"])
plt.gca().invert_yaxis()
plt.xlabel("A-index Over Expected (A-SOE)")
plt.title("Top QBs by Anticipation Over Expected – 2023 Season")
plt.tight_layout()
plt.savefig("qb_top10_ASOE_season.png", dpi=200)
plt.close()

# WR volatility rankings (min 30 targets)
wr_stats = (
    plays.groupby("wr_name")
    .agg(
        targets=("A_index", "count"),
        mean_RMP=("RMP", "mean"),
        mean_Aindex=("A_index", "mean"),
    )
    .reset_index()
)
wr_stats = wr_stats[wr_stats["targets"] >= 30]
top_wr_diff = wr_stats.sort_values("mean_RMP", ascending=False).head(15)

plt.figure(figsize=(8, 6))
plt.barh(top_wr_diff["wr_name"], top_wr_diff["mean_RMP"])
plt.gca().invert_yaxis()
plt.xlabel("Average Route Volatility (RMP)")
plt.title("Most Difficult WRs to Anticipate – 2023 Season")
plt.tight_layout()
plt.savefig("wr_top15_volatility_season.png", dpi=200)
plt.close()

print("Saved plots: qb_top10_Aindex_season.png, qb_top10_ASOE_season.png, wr_top15_volatility_season.png")

# -------- 5. OPTIONAL: MAHOMES TRIANGLE MAP ACROSS ALL WEEKS --------

def get_triangle_for_play(week_df, game_id, play_id):
    g = week_df[(week_df["game_id"] == game_id) & (week_df["play_id"] == play_id)]
    if g.empty:
        return None
    g = g.sort_values("frame_id")
    first_frame = g["frame_id"].min()
    at_throw = g[g["frame_id"] == first_frame]
    qb = at_throw[at_throw["player_role"] == "Passer"]
    wr = at_throw[at_throw["player_role"] == "Targeted Receiver"]
    if qb.empty or wr.empty:
        return None
    qb = qb.iloc[0]
    wr = wr.iloc[0]
    ball_x = g["ball_land_x"].iloc[0]
    ball_y = g["ball_land_y"].iloc[0]
    xs = [qb["x"], wr["x"], ball_x]
    ys = [qb["y"], wr["y"], ball_y]
    return xs, ys

mahomes_plays = plays[plays["qb_name"] == "Patrick Mahomes"]

poly = []
cols = []
for _, row in mahomes_plays.iterrows():
    w = int(row["week"])
    game_id = row["game_id"]
    play_id = row["play_id"]
    week_df = week_frames[w]
    tri = get_triangle_for_play(week_df, game_id, play_id)
    if tri is None:
        continue
    xs, ys = tri
    poly.append(list(zip(xs, ys)))
    cols.append(row["A_index"])

if poly:
    fig, ax = plt.subplots(figsize=(10, 6))
    coll = PolyCollection(poly, array=np.array(cols), cmap="viridis", alpha=0.8)
    ax.add_collection(coll)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_aspect("equal")
    cbar = fig.colorbar(coll, ax=ax)
    cbar.set_label("A-index (0–100)")
    ax.set_title("Patrick Mahomes – Anticipation Triangle Map (All 2023)")
    plt.tight_layout()
    plt.savefig("mahomes_triangles_season.png", dpi=200)
    plt.close()
    print("Saved mahomes_triangles_season.png")
else:
    print("No Mahomes plays found – check name spelling in data")

***Team Level Summaries***

team_stats = (
  plays
  .merge(team_roster[['nfl_id','team']], left_on='qb_id', right_on='nfl_id', how='left')
  .groupby('team')['A_index']
  .mean()
)
# plays: per-play table with columns: qb_name, A_index, A_SOE, week, game_id, play_id
# qb_team: DataFrame with columns ["qb_name", "team"] that you construct or merge

plays_team = plays.merge(qb_team, on="qb_name", how="left")

offense_team_stats = (
    plays_team.groupby("team")
    .agg(
        attempts=("A_index", "count"),
        A_index_mean=("A_index", "mean"),
        A_SOE_mean=("A_SOE", "mean"),
    )
    .reset_index()
    .sort_values("A_index_mean", ascending=False)
)

# Bar plot – offensive anticipation
import matplotlib.pyplot as plt

topN = 16
subset = offense_team_stats.head(topN)

plt.figure(figsize=(10, 6))
plt.barh(subset["team"], subset["A_index_mean"])
plt.gca().invert_yaxis()
plt.xlabel("Average Anticipation Index (0–100)")
plt.title("Top Offenses by Anticipation Index – 2023 Season")
plt.tight_layout()
plt.savefig("teams_offense_Aindex_top16.png", dpi=200)
plt.close()

# Table for slides (top 10)
offense_team_stats.head(10).to_csv("teams_offense_Aindex_top10.csv", index=False)

# plays_def: plays merged with defensive team label (e.g. def_team column from plays.csv)
# columns: def_team, A_index

defense_team_stats = (
    plays_def.groupby("def_team")
    .agg(
        targets=("A_index", "count"),
        opp_Aindex_mean=("A_index", "mean"),
    )
    .reset_index()
    .sort_values("opp_Aindex_mean")   # ascending: lowest allowed anticipation first
)

plt.figure(figsize=(10, 6))
topN = 16
subset = defense_team_stats.head(topN)
plt.barh(subset["def_team"], subset["opp_Aindex_mean"])
plt.gca().invert_yaxis()
plt.xlabel("Opponent Anticipation Index Allowed")
plt.title("Defenses Forcing Lowest Anticipation – 2023 Season")
plt.tight_layout()
plt.savefig("teams_defense_oppAindex_top16.png", dpi=200)
plt.close()

defense_team_stats.head(10).to_csv("teams_defense_oppAindex_top10.csv", index=False)

***Regression***

import zipfile, pandas as pd

zip_path='/mnt/data/nfl-big-data-bowl-2026-analytics (1).zip'
base='114239_nfl_competition_files_published_analytics_final/train/'

with zipfile.ZipFile(zip_path) as z:
    with z.open(base+'input_2023_w01.csv') as f:
        df=pd.read_csv(f)

df.columns

# Check what variables we currently have in the Python environment
vars_list = [k for k in globals().keys() if not k.startswith("_")]
vars_list

import zipfile, math
import pandas as pd
import numpy as np

zip_path = '/mnt/data/nfl-big-data-bowl-2026-analytics (1).zip'
base_prefix = '114239_nfl_competition_files_published_analytics_final/train/'

FRAME_RATE = 10.0
DT = 1.0 / FRAME_RATE

def compute_components_raw(play_df):
    play_df = play_df.sort_values("frame_id")
    ball_x = play_df["ball_land_x"].iloc[0]
    ball_y = play_df["ball_land_y"].iloc[0]
    num_frames = int(play_df["num_frames_output"].iloc[0])
    T_ball = num_frames * DT

    wr_df = play_df[play_df["player_role"]=="Targeted Receiver"].sort_values("frame_id")
    if wr_df.empty:
        return None

    wr0 = wr_df.iloc[0]
    dist_to_land = math.hypot(wr0["x"]-ball_x, wr0["y"]-ball_y)
    s = max(wr0["s"], 0.1)

    T_proj = dist_to_land / s
    raw_LTA = T_proj - T_ball

    K = min(5, len(wr_df))
    sub = wr_df.iloc[:K]
    dirs_rad = np.deg2rad(sub["dir"])
    mean_sin = np.sin(dirs_rad).mean()
    mean_cos = np.cos(dirs_rad).mean()
    circ_var = 1 - math.hypot(mean_sin, mean_cos)
    speed_var = sub["s"].var() if K > 1 else 0.0
    if math.isnan(speed_var):
        speed_var = 0.0
    RMP = 0.5 * circ_var + 0.02 * speed_var
    RMP = max(0.0, min(1.0, RMP))

    theta = math.radians(wr0["dir"])
    vx, vy = s * math.cos(theta), s * math.sin(theta)
    x_pred = wr0["x"] + vx * T_ball
    y_pred = wr0["y"] + vy * T_ball
    pred_err = math.hypot(x_pred - ball_x, y_pred - ball_y)
    AAFP = math.exp(-pred_err / 10.0)

    first_frame = play_df["frame_id"].min()
    db_df = play_df[(play_df["player_side"]=="defense") & (play_df["frame_id"]==first_frame)]
    if db_df.empty:
        DCP = 0.0
    else:
        dists = np.hypot(db_df["x"]-ball_x, db_df["y"]-ball_y)
        DCP = float(np.exp(-dists / 7.0).mean())

    return raw_LTA, RMP, AAFP, DCP

all_rows = []

with zipfile.ZipFile(zip_path) as z:
    for w in range(1, 19):
        fname = f'input_2023_w{w:02d}.csv'
        path_in_zip = base_prefix + fname
        try:
            with z.open(path_in_zip) as f:
                week_df = pd.read_csv(f)
        except KeyError:
            break

        for (gid, pid), g in week_df.groupby(["game_id","play_id"]):
            comps = compute_components_raw(g)
            if comps is None:
                continue
            raw_LTA, RMP, AAFP, DCP = comps
            all_rows.append({
                "week": w,
                "game_id": gid,
                "play_id": pid,
                "raw_LTA": raw_LTA,
                "RMP": RMP,
                "AAFP": AAFP,
                "DCP": DCP
            })

len(all_rows)

plays = pd.DataFrame(all_rows)
plays.describe()
plays[["raw_LTA","RMP","AAFP"]].head()
plays["DCP"].describe()

with zipfile.ZipFile(zip_path) as z:
    with z.open(base_prefix+'input_2023_w01.csv') as f:
        w1 = pd.read_csv(f)
w1["player_side"].unique()

def compute_components_raw(play_df):
    play_df = play_df.sort_values("frame_id")
    ball_x = play_df["ball_land_x"].iloc[0]
    ball_y = play_df["ball_land_y"].iloc[0]
    num_frames = int(play_df["num_frames_output"].iloc[0])
    T_ball = num_frames * DT

    wr_df = play_df[play_df["player_role"]=="Targeted Receiver"].sort_values("frame_id")
    if wr_df.empty:
        return None

    wr0 = wr_df.iloc[0]
    dist_to_land = math.hypot(wr0["x"]-ball_x, wr0["y"]-ball_y)
    s = max(wr0["s"], 0.1)

    T_proj = dist_to_land / s
    raw_LTA = T_proj - T_ball

    K = min(5, len(wr_df))
    sub = wr_df.iloc[:K]
    dirs_rad = np.deg2rad(sub["dir"])
    mean_sin = np.sin(dirs_rad).mean()
    mean_cos = np.cos(dirs_rad).mean()
    circ_var = 1 - math.hypot(mean_sin, mean_cos)
    speed_var = sub["s"].var() if K > 1 else 0.0
    if math.isnan(speed_var):
        speed_var = 0.0
    RMP = 0.5 * circ_var + 0.02 * speed_var
    RMP = max(0.0, min(1.0, RMP))

    theta = math.radians(wr0["dir"])
    vx, vy = s * math.cos(theta), s * math.sin(theta)
    x_pred = wr0["x"] + vx * T_ball
    y_pred = wr0["y"] + vy * T_ball
    pred_err = math.hypot(x_pred - ball_x, y_pred - ball_y)
    AAFP = math.exp(-pred_err / 10.0)

    first_frame = play_df["frame_id"].min()
    db_df = play_df[(play_df["player_side"]=="Defense") & (play_df["frame_id"]==first_frame)]
    if db_df.empty:
        DCP = 0.0
    else:
        dists = np.hypot(db_df["x"]-ball_x, db_df["y"]-ball_y)
        DCP = float(np.exp(-dists / 7.0).mean())

    return raw_LTA, RMP, AAFP, DCP

# recompute all_rows
all_rows = []
with zipfile.ZipFile(zip_path) as z:
    for w in range(1, 19):
        fname = f'input_2023_w{w:02d}.csv'
        path_in_zip = base_prefix + fname
        try:
            with z.open(path_in_zip) as f:
                week_df = pd.read_csv(f)
        except KeyError:
            break
        for (gid,pid), g in week_df.groupby(["game_id","play_id"]):
            comps = compute_components_raw(g)
            if comps is None:
                continue
            raw_LTA, RMP, AAFP, DCP = comps
            all_rows.append({
                "week": w,
                "game_id": gid,
                "play_id": pid,
                "raw_LTA": raw_LTA,
                "RMP": RMP,
                "AAFP": AAFP,
                "DCP": DCP
            })

plays = pd.DataFrame(all_rows)
plays.describe()

# Compute LTA_adj
mu = plays["raw_LTA"].mean()
sigma = plays["raw_LTA"].std(ddof=0)
plays["LTA_z"] = (plays["raw_LTA"] - mu) / sigma
plays["LTA_adj"] = 0.5 * (np.tanh(plays["LTA_z"]) + 1)

plays[["LTA_adj","RMP","AAFP","DCP"]].describe()

# Standardize components for PCA
X = plays[["LTA_adj","RMP","AAFP","DCP"]].to_numpy()
X_mean = X.mean(axis=0)
X_std = X.std(axis=0, ddof=0)
X_stdized = (X - X_mean) / X_std

# Correlation matrix and eigen-decomposition
corr = np.corrcoef(X_stdized, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(corr)  # returns ascending order
# take largest eigenvector (PC1)
idx = np.argsort(eigvals)[::-1]
eigvals_sorted = eigvals[idx]
eigvecs_sorted = eigvecs[:, idx]
pc1 = eigvecs_sorted[:,0]  # loadings for LTA_adj,RMP,AAFP,DCP

eigvals_sorted, pc1

weights_abs = np.abs(pc1)
weights_norm = weights_abs / weights_abs.sum()
weights_norm

REsult --> array([ 0.86392451, -0.02575203, -0.90559197, -0.77953205])

# Compare different weight schemes on all plays
L = plays["LTA_adj"].to_numpy()
R = plays["RMP"].to_numpy()
A = plays["AAFP"].to_numpy()
D = plays["DCP"].to_numpy()

# Hand weights (original)
w_hand = np.array([0.35,0.25,0.25,0.15])
A_hand = (w_hand[0]*L + w_hand[1]*R + w_hand[2]*A + w_hand[3]*D)*100

# Interpretability v2 (what we proposed later)
w_v2 = np.array([0.25,0.30,0.30,0.15])
A_v2 = (w_v2[0]*L + w_v2[1]*R + w_v2[2]*A + w_v2[3]*D)*100

# PCA weights (abs normalized)
w_pca = weights_norm
A_pca = (w_pca[0]*L + w_pca[1]*R + w_pca[2]*A + w_pca[3]*D)*100

import numpy as np
def corr(a,b): return np.corrcoef(a,b)[0,1]

corr_hand_v2 = corr(A_hand,A_v2)
corr_hand_pca = corr(A_hand,A_pca)
corr_v2_pca   = corr(A_v2,A_pca)
corr_hand_v2, corr_hand_pca, corr_v2_pca

LTA_z   = (raw_LTA - mean(raw_LTA)) / std(raw_LTA)
LTA_adj = 0.5 * (tanh(LTA_z) + 1)   # squeezed to ~[0,1]

Then, Built a big matrix X = [\text{LTA_adj}, \text{RMP}, \text{AAFP}, \text{DCP}] over 14,108 plays.


***All Season Application***

In my earlier code, RMP acted like volatility (higher=harder). In the work above for the whole season, I set it as predictability (higher=easier).

If you want to keep the original approach, just replace this block:

rmp = 1.0 - max(0.0, min(1.0, 0.5 * circ_var + 0.02 * speed_var))
RMP = float(np.clip(rmp, 0.0, 1.0))

with this one:

RMP = float(np.clip(0.5 * circ_var + 0.02 * speed_var, 0.0, 1.0)) in order to be aligned with the initial code and approach.

def compute_components_raw(play_df):
    """
    Returns:
      raw_LTA (seconds), RMP [0,1], AAFP [0,1], DCP [0,1], lead_dist (yards)
    For a single (game_id, play_id) group.
    """
    play_df = play_df.sort_values("frame_id")

    ball_x = float(play_df["ball_land_x"].iloc[0])
    ball_y = float(play_df["ball_land_y"].iloc[0])
    num_frames = int(play_df["num_frames_output"].iloc[0])
    T_ball = num_frames * DT

    wr_df = play_df[play_df["player_role"] == "Targeted Receiver"].sort_values("frame_id")
    if wr_df.empty:
        return None

    wr0 = wr_df.iloc[0]
    dist_to_land = math.hypot(float(wr0["x"]) - ball_x, float(wr0["y"]) - ball_y)
    s = max(float(wr0["s"]), 0.1)

    # ---- LTA: projected arrival time - ball flight time ----
    T_proj = dist_to_land / s
    raw_LTA = T_proj - T_ball

    # ---- RMP: route movement predictability (inverse volatility) ----
    K = min(5, len(wr_df))
    sub = wr_df.iloc[:K]
    dirs_rad = np.deg2rad(sub["dir"].astype(float))
    circ_var = 1.0 - math.hypot(np.sin(dirs_rad).mean(), np.cos(dirs_rad).mean())
    speed_var = float(sub["s"].var()) if K > 1 else 0.0
    if not np.isfinite(speed_var):
        speed_var = 0.0

    # volatility proxy → bounded to [0,1]
    # (higher value = more predictable / less chaotic early movement)
    rmp = 1.0 - max(0.0, min(1.0, 0.5 * circ_var + 0.02 * speed_var))
    RMP = float(np.clip(rmp, 0.0, 1.0))

    # ---- AAFP: accuracy to receiver future position ----
    theta = math.radians(float(wr0["dir"]))
    vx, vy = s * math.cos(theta), s * math.sin(theta)
    x_pred = float(wr0["x"]) + vx * T_ball
    y_pred = float(wr0["y"]) + vy * T_ball
    pred_err = math.hypot(x_pred - ball_x, y_pred - ball_y)
    AAFP = float(math.exp(-pred_err / 10.0))

    # ---- DCP: defensive contest pressure near landing at release ----
    first_frame = play_df["frame_id"].min()
    # dataset uses "Defense" (capital D) in many releases
    db_df = play_df[(play_df["player_side"] == "Defense") & (play_df["frame_id"] == first_frame)]
    if db_df.empty:
        DCP = 0.0
    else:
        dists = np.hypot(db_df["x"] - ball_x, db_df["y"] - ball_y)
        DCP = float(np.exp(-dists / 7.0).mean())

    return dict(raw_LTA=raw_LTA, RMP=RMP, AAFP=AAFP, DCP=DCP, lead_dist=dist_to_land)

records = []
week_cache = {}  # store week dfs for triangle plots if memory allows

for fp in week_files:
    # parse week number from filename
    base = os.path.basename(fp)
    w = int(base.split("_w")[1].split(".")[0])
    if WEEKS is not None and w not in WEEKS:
        continue

    week_df = pd.read_csv(fp)
    week_cache[w] = week_df
    print(f"Week {w}: rows={len(week_df):,}")

    for (gid, pid), g in week_df.groupby(["game_id", "play_id"]):
        comps = compute_components_raw(g)
        if comps is None:
            continue

        first = g[g["frame_id"] == g["frame_id"].min()]
        qb = first[first["player_role"] == "Passer"]
        wr = first[first["player_role"] == "Targeted Receiver"]

        rec = {
            "week": w,
            "game_id": gid,
            "play_id": pid,
            "qb_name": qb["player_name"].iloc[0] if not qb.empty else None,
            "wr_name": wr["player_name"].iloc[0] if not wr.empty else None,
        }
        rec.update(comps)
        records.append(rec)

plays = pd.DataFrame(records)
print("Total plays processed:", len(plays))
plays.head()

# ---- Global LTA normalization ----
mu = plays["raw_LTA"].mean()
sd = plays["raw_LTA"].std(ddof=0)
plays["LTA_z"] = (plays["raw_LTA"] - mu) / sd
plays["LTA_adj"] = 0.5 * (np.tanh(plays["LTA_z"]) + 1.0)

# ---- A-index (0–100) ----
plays["A_index"] = (
    0.25 * plays["LTA_adj"] +
    0.30 * plays["RMP"] +
    0.30 * plays["AAFP"] +
    0.15 * plays["DCP"]
) * 100.0

# ---- A-SOE (over expected by lead distance bins) ----
plays["lead_bin"] = pd.cut(
    plays["lead_dist"],
    bins=[0, 5, 10, 15, 20, 30, 50],
    include_lowest=True
)
exp_by_bin = plays.groupby("lead_bin")["A_index"].mean().rename("A_exp").reset_index()
plays = plays.merge(exp_by_bin, on="lead_bin", how="left")
plays["A_SOE"] = plays["A_index"] - plays["A_exp"]

# Save per-play table
plays_path = os.path.join(OUT_DIR, "plays_anticipation_full_season.csv")
plays.to_csv(plays_path, index=False)
print("Saved:", plays_path)

MIN_ATT = 100
qb_stats = (
    plays.groupby("qb_name")
    .agg(
        attempts=("A_index", "count"),
        A_index_mean=("A_index", "mean"),
        A_SOE_mean=("A_SOE", "mean"),
    )
    .reset_index()
)
qb_stats = qb_stats[qb_stats["attempts"] >= MIN_ATT]

# Top QBs by A-index
top_qb = qb_stats.sort_values("A_index_mean", ascending=False).head(10)
plt.figure(figsize=(8,6))
plt.barh(top_qb["qb_name"], top_qb["A_index_mean"])
plt.gca().invert_yaxis()
plt.xlabel("Average Anticipation Index (0–100)")
plt.title("Top QBs by Anticipation Index — Full Season")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "qb_top10_Aindex_season.png"), dpi=200)
plt.close()

# Top QBs by A-SOE
top_qb_asoe = qb_stats.sort_values("A_SOE_mean", ascending=False).head(10)
plt.figure(figsize=(8,6))
plt.barh(top_qb_asoe["qb_name"], top_qb_asoe["A_SOE_mean"])
plt.gca().invert_yaxis()
plt.xlabel("A-index Over Expected (A-SOE)")
plt.title("Top QBs by Anticipation Over Expected — Full Season")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "qb_top10_ASOE_season.png"), dpi=200)
plt.close()

# WR difficulty (predictability) — min targets
MIN_TGT = 50
wr_stats = (
    plays.groupby("wr_name")
    .agg(
        targets=("A_index", "count"),
        mean_RMP=("RMP", "mean"),
        mean_Aindex=("A_index", "mean"),
    )
    .reset_index()
)
wr_stats = wr_stats[wr_stats["targets"] >= MIN_TGT]
top_wr = wr_stats.sort_values("mean_RMP", ascending=False).head(15)

plt.figure(figsize=(8,6))
plt.barh(top_wr["wr_name"], top_wr["mean_RMP"])
plt.gca().invert_yaxis()
plt.xlabel("Average Route Predictability (RMP)")
plt.title("Most Predictable WR Routes (RMP) — Full Season")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "wr_top15_RMP_season.png"), dpi=200)
plt.close()

print("Saved plots in /kaggle/working:")
print("- qb_top10_Aindex_season.png")
print("- qb_top10_ASOE_season.png")
print("- wr_top15_RMP_season.png")

top_qb

def get_triangle_for_play(week_df, game_id, play_id):
    g = week_df[(week_df["game_id"] == game_id) & (week_df["play_id"] == play_id)]
    if g.empty:
        return None
    g = g.sort_values("frame_id")
    first_frame = g["frame_id"].min()
    at_throw = g[g["frame_id"] == first_frame]
    qb = at_throw[at_throw["player_role"] == "Passer"]
    wr = at_throw[at_throw["player_role"] == "Targeted Receiver"]
    if qb.empty or wr.empty:
        return None
    qb = qb.iloc[0]
    wr = wr.iloc[0]
    ball_x = float(g["ball_land_x"].iloc[0])
    ball_y = float(g["ball_land_y"].iloc[0])
    xs = [float(qb["x"]), float(wr["x"]), ball_x]
    ys = [float(qb["y"]), float(wr["y"]), ball_y]
    return xs, ys

# Sample 200 plays for a clean league triangle visualization
sample = plays.sample(n=min(200, len(plays)), random_state=42)

poly, cols = [], []
for _, row in sample.iterrows():
    w = int(row["week"])
    if w not in week_cache:
        continue
    tri = get_triangle_for_play(week_cache[w], row["game_id"], row["play_id"])
    if tri is None:
        continue
    xs, ys = tri
    poly.append(list(zip(xs, ys)))
    cols.append(row["A_index"])

fig, ax = plt.subplots(figsize=(10,6))
coll = PolyCollection(poly, array=np.array(cols), cmap="viridis", alpha=0.75)
ax.add_collection(coll)
ax.set_xlim(0,120); ax.set_ylim(0,53.3); ax.set_aspect("equal")
cbar = fig.colorbar(coll, ax=ax); cbar.set_label("A-index (0–100)")
ax.set_title("Triangle Anticipation Map — Sample 200 Plays (Full Season)")
ax.set_xlabel("Field X (yards, 0–120)")
ax.set_ylabel("Field Y (yards, 0–53.3)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "triangle_map_sample200.png"), dpi=200)
plt.close()

print("Saved: triangle_map_sample200.png")

mahomes = plays[plays["qb_name"] == "Patrick Mahomes"]
poly, cols = [], []

for _, row in mahomes.iterrows():
    w = int(row["week"])
    if w not in week_cache:
        continue
    tri = get_triangle_for_play(week_cache[w], row["game_id"], row["play_id"])
    if tri is None:
        continue
    xs, ys = tri
    poly.append(list(zip(xs, ys)))
    cols.append(row["A_index"])

if poly:
    fig, ax = plt.subplots(figsize=(10,6))
    coll = PolyCollection(poly, array=np.array(cols), cmap="viridis", alpha=0.80)
    ax.add_collection(coll)
    ax.set_xlim(0,120); ax.set_ylim(0,53.3); ax.set_aspect("equal")
    cbar = fig.colorbar(coll, ax=ax); cbar.set_label("A-index (0–100)")
    ax.set_title("Patrick Mahomes — Triangle Anticipation Map (Full Season)")
    ax.set_xlabel("Field X (yards, 0–120)")
    ax.set_ylabel("Field Y (yards, 0–53.3)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mahomes_triangles_season.png"), dpi=200)
    plt.close()
    print("Saved: mahomes_triangles_season.png")
else:
    print("No Mahomes plays found — check qb_name spelling in data.")

top5 = plays.sort_values("A_index", ascending=False).head(5).reset_index(drop=True)
top5_path = os.path.join(OUT_DIR, "top5_elite_plays.csv")
top5.to_csv(top5_path, index=False)
print("Saved:", top5_path)

elite_dir = os.path.join(OUT_DIR, "elite_triangles")
os.makedirs(elite_dir, exist_ok=True)

pngs = []
for i, row in top5.iterrows():
    w = int(row["week"])
    tri = get_triangle_for_play(week_cache[w], row["game_id"], row["play_id"])
    if tri is None:
        continue
    xs, ys = tri

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    poly = [list(zip(xs, ys))]
    coll = PolyCollection(poly, array=np.array([row["A_index"]]), cmap="viridis", alpha=0.92)
    ax.add_collection(coll)

    ax.scatter(xs, ys, zorder=3)
    ax.text(xs[0], ys[0], " QB", fontsize=10, va="center")
    ax.text(xs[1], ys[1], " WR", fontsize=10, va="center")
    ax.text(xs[2], ys[2], " Ball", fontsize=10, va="center")

    ax.set_xlim(0, 120); ax.set_ylim(0, 53.3); ax.set_aspect("equal")
    ax.set_xlabel("Field X (yards)")
    ax.set_ylabel("Field Y (yards)")
    ax.set_title(
        f"Elite Anticipation Play #{i+1}\n"
        f"Week {w} | {row['qb_name']} → {row['wr_name']} | A-index {row['A_index']:.1f}"
    )
    cbar = fig.colorbar(coll, ax=ax)
    cbar.set_label("A-index (0–100)")
    plt.tight_layout()

    out_path = os.path.join(elite_dir, f"elite_play_{i+1}.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    pngs.append(out_path)

print("Saved elite play triangles:", len(pngs))
top5[["week","game_id","play_id","qb_name","wr_name","A_index","A_SOE","raw_LTA","RMP","AAFP","DCP","lead_dist"]]


- All normalization and reported statistics are computed using the **full 2023 regular season**.
- Visual examples show a subset of plays for clarity.








