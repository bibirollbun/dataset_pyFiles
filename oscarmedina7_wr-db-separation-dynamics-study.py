import pandas as pd
import numpy as np
import seaborn as sns
import os
import glob
import plotly.express as px
import category_encoders as ce
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from scipy.interpolate import griddata


#List files 
base_path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"

#Load data
input_files = glob.glob(f"{base_path}/input_2023_w*.csv")
output_files = glob.glob(f"{base_path}/output_2023_w*.csv")

print("Finded inputs:", len(input_files))
print("Finded outputs:", len(output_files))

#Create dataset for input and output
df_inputs= pd.concat([pd.read_csv(f) for f in input_files], ignore_index=True)
df_outputs= pd.concat([pd.read_csv(f) for f in output_files], ignore_index=True)
df_supplementary = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv")


df_inputs.head()


df_outputs.head()


df_supplementary.head()


def get_wr_db_closest(df):
    
    df = df.copy()

    # Columns we need
    base_cols = ["game_id","play_id","frame_id"]

    def_cols = base_cols + ["nfl_id","x","y","s","a","o","dir"]
    wr_cols  = base_cols + ["x","y","s","a","o","dir"]

    # Filter roles
    defenders = df[df["player_role"] == "Defensive Coverage"][def_cols].rename(
        columns={
            "x":"x_def", "y":"y_def", "s":"s_def", "a":"a_def",
            "o":"o_def", "dir":"dir_def"
        }
    )

    wr = df[df["player_role"] == "Targeted Receiver"][wr_cols].rename(
        columns={
            "x":"x_wr", "y":"y_wr", "s":"s_wr", "a":"a_wr",
            "o":"o_wr", "dir":"dir_wr"
        }
    )

    # Merge by play-frame
    merged = defenders.merge(wr, on=base_cols, how="left")

    # Separation distance
    merged["separation_dist"] = np.sqrt(
        (merged["x_def"] - merged["x_wr"])**2 +
        (merged["y_def"] - merged["y_wr"])**2
    )

    # Heading difference (clean, wrapped -180 to 180)
    merged["heading_diff"] = abs((merged["dir_wr"] - merged["dir_def"] + 180) % 360) - 180

    # Get closest DB per frame
    closest = merged.loc[
        merged.groupby(["game_id","play_id","frame_id"])["separation_dist"].idxmin()
    ].reset_index(drop=True)
    
    return closest



def compute_separation_features(df):
    df = df.copy()

    # Sort
    df = df.sort_values(["game_id", "play_id", "frame_id"])

    # sep_speed = d(separation)/dt
    df["sep_speed"] = df.groupby(["game_id","play_id"])["separation_dist"].diff() / 0.1
    df["sep_speed"] = df["sep_speed"].fillna(0)

    # sep_accel = d(sep_speed)/dt
    df["sep_accel"] = df.groupby(["game_id","play_id"])["sep_speed"].diff() / 0.1
    df["sep_accel"] = df["sep_accel"].fillna(0)

    # speed difference (WR - DB)
    df["speed_diff"] = df["s_wr"] - df["s_def"]

    # accel difference (WR - DB)
    df["accel_diff"] = df["a_wr"] - df["a_def"]

    return df



df_sep = get_wr_db_closest(df_inputs)
df_features = compute_separation_features(df_sep)


df_features


df_sep_dist = df_features[["separation_dist"]]


def plot_hist(df):
    plt.figure(figsize=(10,5))
    plt.hist(df_sep_dist["separation_dist"], bins=61, alpha=0.7, edgecolor="black")
    plt.xlim(0, 26)
    plt.xticks(range(0, 26, 1))
    plt.grid(axis="y", alpha=0.3)
    plt.xlabel("Separation Distance")
    plt.ylabel("Frecuency")
    plt.title("Frequency of separation distance")
    plt.show()


plot_hist(df_sep_dist)


#minimum_separation
def get_minimum_separation(df, top_k=5):
    df = df.copy()

    df["sep_rank"] = df.groupby(
     ["game_id","play_id"]
    )["separation_dist"].rank(method="first")

    top_frames = df[df["sep_rank"] <= top_k].copy()

    top_frames = top_frames.sort_values(
    ["game_id","play_id","sep_rank"]
    )

    df_top_frames = top_frames[["game_id","play_id","frame_id","nfl_id","separation_dist","sep_rank"]]
    return df_top_frames


top5_min_sep= get_minimum_separation(df_features, top_k=5)
top5_min_sep


def plot_min_sep_frame_hist(df_min_sep):
    plt.figure(figsize=(10,5))
    plt.hist(df_min_sep["frame_id"], bins=61, alpha=0.7, edgecolor="black")
    plt.xlim(0, 60)
    plt.xticks(range(0, 57, 2))
    plt.grid(axis="y", alpha=0.3)
    plt.xlabel("Frame_ID")
    plt.ylabel("Frecuency")
    plt.title("Frequency of frames with minimum separation")
    plt.show()


plot_min_sep_frame_hist(top5_min_sep)


#maximum_separation
def get_maximum_separation(df, top_k=5):
    df = df.copy()

    df["sep_rank"] = df.groupby(
     ["game_id","play_id"]
    )["separation_dist"].rank(method="first", ascending=False)

    top_frames = df[df["sep_rank"] <= top_k].copy()

    top_frames = top_frames.sort_values(
    ["game_id","play_id","sep_rank"]
    )

    df_top_frames = top_frames[["game_id","play_id","frame_id","nfl_id","separation_dist","sep_rank"]]
    return df_top_frames


top5_max_sep= get_maximum_separation(df_features, top_k=5)
top5_max_sep

top5_max_sep


def plot_max_sep_frame_hist(df_max_sep):
    plt.figure(figsize=(10,5))
    plt.hist(df_max_sep["frame_id"], bins=61, alpha=0.7, edgecolor="black")
    plt.xlim(0, 60)
    plt.xticks(range(0, 57, 2))
    plt.grid(axis="y", alpha=0.3)
    plt.xlabel("Frame_ID")
    plt.ylabel("Frecuency")
    plt.title("Frequency of frames with maximum separation")
    plt.show()


plot_max_sep_frame_hist(top5_max_sep)


#Create separation timeline 

def plot_separation_timeline (df, game_id, play_id):
    play = df[(df["game_id"] == game_id) & (df["play_id"] == play_id)]

    plt.figure(figsize=(10,5))
    plt.plot(play["frame_id"], play["separation_dist"], marker="o", linewidth=2)

    min_row = play.loc[play["separation_dist"].idxmin()]
    plt.scatter(min_row["frame_id"], min_row["separation_dist"], s=120)


    plt.title(f"Separation Timeline - Game {game_id}, Play {play_id}")
    plt.xlabel("Frame")
    plt.ylabel("Separation Distance (yards)")
    plt.grid(True)
    plt.show()

plot_separation_timeline(df_features, 2024010713, 4018)


play = df_features[
    (df_features["game_id"] == 2024010713) &
    (df_features["play_id"] == 4018)
].sort_values("frame_id")


#Frames in X
x = play["frame_id"].values
#Separation velocity in Y
y = play["sep_speed"].values

z = np.abs(play["sep_speed"].values)

xi = np.linspace(x.min(), x.max(), 300)
yi = np.linspace(y.min(), y.max(), 300)
xi, yi = np.meshgrid(xi, yi)

zi = griddata((x, y), z, (xi, yi), method="cubic")


plt.figure(figsize=(12, 6))

plt.imshow(
    zi,
    extent=(x.min(), x.max(), y.min(), y.max()),
    aspect="auto",
    origin="lower",
    cmap="coolwarm"
)

plt.colorbar(label="Separation Velocity Magnitude")
plt.xlabel("Frame ID (time)")
plt.ylabel("Separation Velocity (yd/s)")
plt.title("Separation Velocity Map- WR vs Closest DB")
plt.show()


game = 2024010713 #Example that we will take for game_id
play = 4018 #Example that we will take for play_id

df_play = df_features[
    (df_features["game_id"]== game) &
    (df_features["play_id"] == play)
][["game_id","play_id","frame_id","speed_diff","accel_diff","separation_dist"]].copy()

#Define function for normalize values in speed and acceleration
def min_max_norm(series):
    return (series - series.min())/ (series.max()-series.min() + 1e-9)


df_play["sep_norm"] = min_max_norm(df_play["separation_dist"])
df_play["speed_diff_norm"] = min_max_norm(df_play["speed_diff"])
df_play["accel_diff_norm"] = min_max_norm(df_play["accel_diff"])

#Create index for WR Advantage Curve
df_play["adv_index"] = (
    0.4 * df_play["sep_norm"] +
    0.3 * df_play["speed_diff_norm"] +
    0.3 * df_play["accel_diff_norm"]
)



plt.figure(figsize=(12,5))

plt.plot(df_play["frame_id"],df_play["adv_index"], linewidth=3)
plt.title(f"WR Advantage Curve over frames - Game: {game}, Play: {play}", fontsize= 14)
plt.xlabel("Frames", fontsize=12)
plt.ylabel("Advantage Index", fontsize=12)
plt.grid(True)
plt.show()


#Create data frame for polar route
df_polar_route = df_features[
    (df_features["game_id"]== game) &
    (df_features["play_id"] == play)
][["game_id","play_id","frame_id","separation_dist","s_wr"]].copy()

df_polar_route['theta'] = df_polar_route.groupby(['game_id', 'play_id'])['frame_id'].transform(
    lambda x: 2 * np.pi * (x - x.min()) / (x.max() - x.min())
)

# Create variables for make graphic
theta = df_polar_route['theta'].values
r = df_polar_route['separation_dist'].values
speed = df_polar_route['s_wr'].values

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection='polar')

# Scatter polar
sc = ax.scatter(theta, r, c=speed, cmap='viridis', s=40, alpha=0.9)

# Colorbar
cbar = plt.colorbar(sc, pad=0.1)
cbar.set_label("WR Speed")
ax.set_title(
    f"Polar Route–Separation Plot\nGame {game} | Play {play}",
    va='bottom',
    fontsize=14
)
ax.grid(True, linestyle='--', alpha=0.6)
plt.show()


#Create df 
df_angles = df_features[
    (df_features["game_id"]== game) &
    (df_features["play_id"] == play)
][["game_id","play_id","frame_id","x_def","y_def","x_wr","y_wr","dir_def"]].copy()

#Convert to radians
df_angles["real_angle"] = np.radians(df_angles["dir_def"])

#Calculate optimal angle frame-by-frame
dx = df_angles["x_wr"] - df_angles["x_def"]
dy = df_angles["y_wr"] - df_angles["y_def"]

df_angles["optimal_angle"] = np.arctan2(dy, dx)

#Angle error
raw_diff = df_angles["real_angle"] - df_angles["optimal_angle"]
df_angles["pursuit_error"] = (raw_diff + np.pi)% (2*np.pi) - np.pi

#Dynamic Ribbon Plot
plt.figure(figsize=(12,6))
plt.plot(df_angles["frame_id"], df_angles["optimal_angle"], label="Optimal Pursuit Angle", linewidth=2)
plt.plot(df_angles["frame_id"], df_angles["real_angle"], label="Real Angle (dir)", linewidth=2)

#Shaded area for error magnitude
plt.fill_between(
    df_angles["frame_id"],
    df_angles["optimal_angle"],
    df_angles["real_angle"],
    where=(df_angles["real_angle"] >= df_angles["optimal_angle"]),
    alpha=0.25,
    interpolate=True
)

plt.title("Pursuit Angle vs Real Angle")
plt.xlabel("Frame_Id")
plt.ylabel("Angle (radians)")
plt.legend()
plt.grid(True)
plt.show()


df_3d = df_features[
    (df_features["game_id"]== game) &
    (df_features["play_id"] == play)
][["game_id","play_id","frame_id","separation_dist","speed_diff","heading_diff"]].copy()


fig = px.scatter_3d(
    df_3d,
    x="separation_dist",
    y="speed_diff",
    z="heading_diff",
    color="frame_id",
    hover_data= df_3d.columns,
    title="3D Kinematic Space Plot"
)
fig.show()

