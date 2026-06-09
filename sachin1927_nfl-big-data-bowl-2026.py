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


# %% [code]
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use("default")

from pathlib import Path
from tqdm import tqdm

# Display options
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)



def load_and_concat(files, kind="input"):
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["week_file"] = f.name
        df["kind"] = kind
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)



from pathlib import Path 
DATA_DIR= Path("/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final")

input_files= sorted(DATA_DIR.glob("train/input_2023_w*.csv"))
output_files= sorted(DATA_DIR.glob("train/output_2023_w*.csv"))

print(f"Found {len(input_files)} input files and {len(output_files)} output files.")

input_df= load_and_concat(input_files, kind="input")
output_df= load_and_concat(output_files, kind="output")




# %% [code]
def load_and_concat(files, kind="input"):
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["week_file"] = f.name
        df["kind"] = kind
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

input_df = load_and_concat(input_files, kind="input")
output_df = load_and_concat(output_files, kind="output")

input_df.head(), output_df.head()



SUPP_FILE_PATH= Path("/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv")

if SUPP_FILE_PATH.exists():
    supp_df= pd.read_csv(SUPP_FILE_PATH)
    print("Supplementry data load sucessfully!")
    print(supp_df.head())
else:
    print(f"Error: File not Found at {SUPP_FILE_PATH}")


# %% [code]
# Keep only plays with targeted receiver in input
tr_input = input_df[input_df["player_role"] == "Targeted Receiver"].copy()

# Sanity checks
tr_input[["game_id", "play_id"]].drop_duplicates().shape, tr_input["nfl_id"].nunique()



# %% [code]
# Remove obvious garbage / missing landing point if any
tr_input = tr_input.dropna(subset=["ball_land_x", "ball_land_y", "num_frames_output"])

tr_input.shape



# %% [code]
throw_frame = (
    tr_input.groupby(["game_id", "play_id", "nfl_id"])["frame_id"]
    .max()
    .reset_index()
    .rename(columns={"frame_id": "throw_frame"})
)

tr_throw = tr_input.merge(throw_frame, on=["game_id", "play_id", "nfl_id"])
tr_throw = tr_throw[tr_throw["frame_id"] == tr_throw["throw_frame"]].copy()

tr_throw[["game_id", "play_id", "nfl_id", "frame_id", "throw_frame"]].head()



# %% [code]
tr_output = output_df.merge(
    tr_throw[["game_id", "play_id", "nfl_id", "num_frames_output", "ball_land_x", "ball_land_y"]],
    on=["game_id", "play_id", "nfl_id"],
    how="inner"
)

# Assume frame_id in output is already 1..num_frames_output
tr_output["t"] = tr_output["frame_id"].astype(int)

# Sanity
tr_output[["game_id", "play_id", "nfl_id", "t", "num_frames_output"]].head()



# %% [code]
def euclid(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Receiver distance to ball each frame
tr_output["dist_to_ball_rec"] = euclid(
    tr_output["x"],
    tr_output["y"],
    tr_output["ball_land_x"],
    tr_output["ball_land_y"],
)

# Sort and compute step distance
tr_output = tr_output.sort_values(["game_id", "play_id", "nfl_id", "t"])

tr_output[["x_prev", "y_prev"]] = tr_output.groupby(
    ["game_id", "play_id", "nfl_id"]
)[["x", "y"]].shift(1)

tr_output["step_len_rec"] = euclid(
    tr_output["x"],
    tr_output["y"],
    tr_output["x_prev"],
    tr_output["y_prev"],
)

tr_output.head()



# %% [code]
rec_path_feats = (
    tr_output.groupby(["game_id", "play_id", "nfl_id"])
    .agg(
        path_len_rec=("step_len_rec", "sum"),
        dist_to_ball_rec_start=("dist_to_ball_rec", "first"),
        dist_to_ball_rec_end=("dist_to_ball_rec", "last"),
        T=("t", "max"),
    )
    .reset_index()
)

# Straight-line distance from throw position to ball
throw_xy = tr_throw[
    ["game_id", "play_id", "nfl_id", "x", "y", "ball_land_x", "ball_land_y"]
].copy()

throw_xy["dist_straight_rec"] = euclid(
    throw_xy["x"], throw_xy["y"], throw_xy["ball_land_x"], throw_xy["ball_land_y"]
)

rec_feats = rec_path_feats.merge(
    throw_xy[["game_id", "play_id", "nfl_id", "dist_straight_rec"]],
    on=["game_id", "play_id", "nfl_id"],
    how="left",
)

# Frame rate (assume 10 Hz; change if docs say otherwise)
FPS = 10.0

rec_feats["route_efficiency_rec"] = (
    rec_feats["path_len_rec"] / rec_feats["dist_straight_rec"]
)

rec_feats["closing_speed_rec"] = (
    (rec_feats["dist_to_ball_rec_start"] - rec_feats["dist_to_ball_rec_end"])
    / (rec_feats["T"] / FPS)
)

rec_feats.head()



# %% [code]
# Defensive coverage players from input
cov_input = input_df[input_df["player_role"] == "Defensive Coverage"][
    ["game_id", "play_id", "nfl_id", "player_role", "player_side"]
].drop_duplicates()

cov_output = output_df.merge(
    cov_input, on=["game_id", "play_id", "nfl_id"], how="inner"
)

cov_output.head()



# %% [code]
ball_info = tr_throw[["game_id", "play_id", "ball_land_x", "ball_land_y"]].drop_duplicates()

cov_output = cov_output.merge(
    ball_info,
    on=["game_id", "play_id"],
    how="inner",
)

cov_output["t"] = cov_output["frame_id"].astype(int)

cov_output["dist_to_ball_def"] = euclid(
    cov_output["x"],
    cov_output["y"],
    cov_output["ball_land_x"],
    cov_output["ball_land_y"],
)

cov_output.head()



# %% [code]
cov_output = cov_output.sort_values(["game_id", "play_id", "t", "dist_to_ball_def"])

nearest_def_per_frame = cov_output.groupby(
    ["game_id", "play_id", "t"], as_index=False
).first()  # because sorted by dist_to_ball_def

nearest_def_per_frame.head()



# %% [code]
nearest_def_per_frame = nearest_def_per_frame.sort_values(
    ["game_id", "play_id", "t"]
)

nearest_def_per_frame[["x_prev", "y_prev"]] = nearest_def_per_frame.groupby(
    ["game_id", "play_id"]
)[["x", "y"]].shift(1)

nearest_def_per_frame["step_len_def"] = euclid(
    nearest_def_per_frame["x"],
    nearest_def_per_frame["y"],
    nearest_def_per_frame["x_prev"],
    nearest_def_per_frame["y_prev"],
)

def_feats = (
    nearest_def_per_frame.groupby(["game_id", "play_id"])
    .agg(
        path_len_def=("step_len_def", "sum"),
        dist_to_ball_def_start=("dist_to_ball_def", "first"),
        dist_to_ball_def_end=("dist_to_ball_def", "last"),
        T=("t", "max"),
        # capture which defender is nearest at end (optional)
        nfl_id_def_end=("nfl_id", "last"),
    )
    .reset_index()
)

def_feats["closing_speed_def"] = (
    (def_feats["dist_to_ball_def_start"] - def_feats["dist_to_ball_def_end"])
    / (def_feats["T"] / FPS)
)

def_feats.head()



# %% [code]
play_feats = rec_feats.merge(
    def_feats,
    on=["game_id", "play_id", "T"],  # join on T just to be safe
    how="inner",
)

# Add supplementary info (pass_result, route, coverage, etc.)
play_feats = play_feats.merge(
    supp_df[
        [
            "game_id",
            "play_id",
            "pass_result",
            "route_of_targeted_receiver",
            "team_coverage_man_zone",
            "team_coverage_type",
            "pass_length",
        ]
    ].drop_duplicates(),
    on=["game_id", "play_id"],
    how="left",
)

# Main metric: Ball-in-Flight Separation Advantage (BFSA)
play_feats["BFSA"] = (
    play_feats["dist_to_ball_rec_end"] - play_feats["dist_to_ball_def_end"]
)

play_feats.head()



# %% [code]
play_feats["is_complete"] = (play_feats["pass_result"] == "C").astype(int)

play_feats[["BFSA", "route_efficiency_rec", "closing_speed_rec", "closing_speed_def", "is_complete"]].describe()



# %% [code]
# BFSA by outcome
fig, ax = plt.subplots(figsize=(8, 5))
play_feats.boxplot(column="BFSA", by="pass_result", ax=ax)
ax.set_title("BFSA by Pass Result")
ax.set_ylabel("BFSA (Receiver closer to ball than nearest defender at arrival)")
plt.suptitle("")
plt.show()



# %% [code]
# Completion rate by BFSA bin
bins = np.linspace(play_feats["BFSA"].quantile(0.01),
                   play_feats["BFSA"].quantile(0.99), 15)
play_feats["BFSA_bin"] = pd.cut(play_feats["BFSA"], bins=bins)

comp_by_bin = play_feats.groupby("BFSA_bin")["is_complete"].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(len(comp_by_bin)), comp_by_bin["is_complete"], marker="o")
ax.set_xticks(range(len(comp_by_bin)))
ax.set_xticklabels(comp_by_bin["BFSA_bin"], rotation=90)
ax.set_ylabel("Completion Rate")
ax.set_title("Completion Rate vs BFSA")
plt.tight_layout()
plt.show()



# %% [code]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model_df = play_feats.dropna(
    subset=["BFSA", "route_efficiency_rec", "closing_speed_rec", "closing_speed_def", "is_complete"]
).copy()

X = model_df[["BFSA", "route_efficiency_rec", "closing_speed_rec", "closing_speed_def", "pass_length"]]
y = model_df["is_complete"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ]
)

pipe.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score, accuracy_score

y_pred_proba = pipe.predict_proba(X_valid)[:, 1]
y_pred = pipe.predict(X_valid)

print("Accuracy:", accuracy_score(y_valid, y_pred))
print("ROC AUC:", roc_auc_score(y_valid, y_pred_proba))



# %% [code]
coef = pipe.named_steps["logreg"].coef_[0]
feature_importance = pd.DataFrame(
    {"feature": X.columns, "coef": coef}
).sort_values("coef", ascending=False)
feature_importance



# %% [code]
def plot_play_frames(game_id, play_id, max_frames=20):
    rec_traj = tr_output[
        (tr_output["game_id"] == game_id)
        & (tr_output["play_id"] == play_id)
    ].copy()

    def_traj = nearest_def_per_frame[
        (nearest_def_per_frame["game_id"] == game_id)
        & (nearest_def_per_frame["play_id"] == play_id)
    ].copy()

    ball_x = rec_traj["ball_land_x"].iloc[0]
    ball_y = rec_traj["ball_land_y"].iloc[0]

    for t in range(1, min(max_frames, rec_traj["t"].max()) + 1):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 53.3)

        # receiver up to time t
        r = rec_traj[rec_traj["t"] <= t]
        ax.plot(r["x"], r["y"], marker="o", label="Targeted Receiver")

        # defender up to time t
        d = def_traj[def_traj["t"] <= t]
        ax.plot(d["x"], d["y"], marker="^", label="Nearest Defender")

        ax.scatter([ball_x], [ball_y], marker="x", s=100, label="Ball Landing")

        ax.set_title(f"Game {game_id}, Play {play_id}, Frame {t}")
        ax.legend(loc="upper right")
        plt.show()



play_feats[["game_id", "play_id"]].head(20)



plot_play_frames(game_id=2023090700, play_id=219, max_frames=20)



import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def animate_play_mp4(game_id, play_id, save_path="play.mp4", fps=5):
    """
    Create an MP4 animation of the ball-in-flight movement
    for the targeted receiver and nearest defender.
    """

    rec = tr_output[(tr_output.game_id == game_id) & (tr_output.play_id == play_id)]
    dev = nearest_def_per_frame[(nearest_def_per_frame.game_id == game_id) & (nearest_def_per_frame.play_id == play_id)]

    if rec.empty or dev.empty:
        print("No valid data for this play. Try another play_id.")
        return
    
    ball_x = rec["ball_land_x"].iloc[0]
    ball_y = rec["ball_land_y"].iloc[0]
    frames = int(rec["t"].max())

    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)

    rec_line, = ax.plot([], [], 'o-', label="Receiver", color="blue")
    def_line, = ax.plot([], [], '^-', label="Defender", color="red")
    ax.scatter(ball_x, ball_y, s=200, marker="x", color="black", label="Ball Landing")

    ax.legend()

    def update(frame):
        r = rec[rec.t <= frame]
        d = dev[dev.t <= frame]

        rec_line.set_data(r.x, r.y)
        def_line.set_data(d.x, d.y)

        ax.set_title(f"Game {game_id}, Play {play_id}, Frame {frame}")
        return rec_line, def_line

    writer = FFMpegWriter(fps=fps, metadata=dict(artist='BDB2026'))
    anim = FuncAnimation(fig, update, frames=frames, interval=200, blit=False)

    anim.save(save_path, writer=writer)
    plt.close()
    print(f"MP4 animation saved to: {save_path}")



animate_play_mp4(2023090700, 101, save_path="g1_p101.mp4")



def plot_bfsa_timeline(game_id, play_id):
    rec = tr_output[(tr_output.game_id == game_id) & (tr_output.play_id == play_id)].copy()
    dev = nearest_def_per_frame[(nearest_def_per_frame.game_id == game_id) & (nearest_def_per_frame.play_id == play_id)].copy()

    if rec.empty or dev.empty:
        print("Invalid play.")
        return

    df = rec[["game_id","play_id","t","dist_to_ball_rec"]].merge(
        dev[["game_id","play_id","t","dist_to_ball_def"]],
        on=["game_id","play_id","t"],
        how="inner"
    )

    df["BFSA_t"] = df["dist_to_ball_rec"] - df["dist_to_ball_def"]

    plt.figure(figsize=(8,4))
    plt.plot(df.t, df.dist_to_ball_rec, label="Receiver distance to ball")
    plt.plot(df.t, df.dist_to_ball_def, label="Defender distance to ball")
    plt.plot(df.t, df.BFSA_t, label="BFSA (Receiver - Defender)", linestyle="--")

    plt.axhline(0, color='black', linewidth=0.5)
    plt.xlabel("Frame (t)")
    plt.ylabel("Distance (yards)")
    plt.title("Ball-in-Flight Separation Timeline")
    plt.legend()
    plt.show()



plot_bfsa_timeline(2023090700, 101)





