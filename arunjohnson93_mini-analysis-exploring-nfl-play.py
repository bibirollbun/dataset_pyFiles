# === Loading Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# 1) Load a sample week (change path if needed)
df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w01.csv')

# 2) Inspect shape and columns quickly
print("Rows,Cols:", df.shape)
print(df.columns.tolist())

# 3) Pick a single game/play/player to examine (replace with one that exists in your df)
sample_game = df['game_id'].iloc[0]
sample_play = df[df['game_id']==sample_game]['play_id'].iloc[0]
sample_nflid = df[(df['game_id']==sample_game) & (df['play_id']==sample_play)]['nfl_id'].iloc[0]

print("Example:", sample_game, sample_play, sample_nflid)

# 4) Filter to that player's rows for that play and sort by frame_id
player_df = df[(df['game_id']==sample_game) & 
               (df['play_id']==sample_play) & 
               (df['nfl_id']==sample_nflid)].sort_values('frame_id').reset_index(drop=True)

player_df[['frame_id','x','y','s','a']].head(8)

# 5) Compute Euclidean distance between consecutive frames (per-player)
player_df['x_prev'] = player_df['x'].shift(1)
player_df['y_prev'] = player_df['y'].shift(1)

# For the first row there is no previous point so distance is NaN -> fill 0
player_df['frame_dist'] = np.sqrt((player_df['x'] - player_df['x_prev'])**2 +
                                  (player_df['y'] - player_df['y_prev'])**2).fillna(0)

# 6) Total distance (yards)
total_distance = player_df['frame_dist'].sum()
print(f"Total distance (yards) for the player in this play: {total_distance:.3f}")

# 7) Average speed: method A -> use frame counts.
# NOTE: we don't know exact seconds per frame from dataset here.
# If frames are recorded at regular intervals (FRAME_DT seconds), average_speed = total_distance / total_time
# You can compute average distance per frame (yards/frame) as a baseline:
avg_dist_per_frame = player_df['frame_dist'].mean()
print(f"Average distance per frame: {avg_dist_per_frame:.4f} yards/frame")

# If you assume a frame rate, e.g., 10 frames per second (common in tracking), then:
frame_rate = 10.0   # frames per second (change if you prefer another assumption)
total_time_seconds = player_df['frame_id'].nunique() / frame_rate
avg_speed_from_distance = total_distance / total_time_seconds if total_time_seconds>0 else np.nan
print(f"Assuming {frame_rate} fps -> total_time = {total_time_seconds:.3f}s; avg speed = {avg_speed_from_distance:.3f} yards/s")

# 8) Average speed: method B -> mean of column 's' (if provided)
if 's' in player_df.columns:
    avg_s_column = player_df['s'].mean()
    print(f"Average of provided 's' column: {avg_s_column:.3f} yards/s")

# 9) Quick dataframe view with distances per frame (first 10 rows)
player_df[['frame_id','x','y','frame_dist','s']].head(10)



# === Per-Player Distance for One Play ===

# 1) Filter same game_id and play_id as before
play_df = df[(df['game_id'] == sample_game) & (df['play_id'] == sample_play)].copy()

# 2) Sort by player and frame
play_df = play_df.sort_values(['nfl_id', 'frame_id']).reset_index(drop=True)

# 3) Compute per-frame distances for each player
play_df['x_prev'] = play_df.groupby('nfl_id')['x'].shift(1)
play_df['y_prev'] = play_df.groupby('nfl_id')['y'].shift(1)

play_df['frame_dist'] = np.sqrt((play_df['x'] - play_df['x_prev'])**2 +
                                (play_df['y'] - play_df['y_prev'])**2).fillna(0)

# 4) Aggregate to per-player summary
player_summary = (
    play_df.groupby(['nfl_id', 'player_name'], as_index=False)
    .agg(total_distance=('frame_dist', 'sum'),
         avg_s=('s', 'mean'),
         num_frames=('frame_id', 'nunique'))
)

# 5) Assume 10 frames per second → total time = frames / fps
frame_rate = 10.0
player_summary['total_time_s'] = player_summary['num_frames'] / frame_rate
player_summary['avg_speed_from_distance'] = player_summary['total_distance'] / player_summary['total_time_s']

# 6) Sort by total_distance descending
player_summary = player_summary.sort_values('total_distance', ascending=False)

# 7) Display first 10
player_summary.head(10)



import matplotlib.pyplot as plt

# Filter one player's data from your DataFrame
# Replace with any game_id, play_id, nfl_id you have
one_player = df[(df['game_id'] == 2023090700) & 
                (df['play_id'] == 101) & 
                (df['nfl_id'] == 54527)]

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(one_player['x'], one_player['y'], marker='o', linestyle='-', color='blue')

# Label axes and title
plt.title("Player Movement Path (x vs y)")
plt.xlabel("Field Length (yards)")
plt.ylabel("Field Width (yards)")
plt.grid(True)
plt.show()



# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 5))

# Set up axis limits similar to your earlier plot
ax.set_xlim(one_player['x'].min() - 2, one_player['x'].max() + 2)
ax.set_ylim(one_player['y'].min() - 2, one_player['y'].max() + 2)
ax.set_title("Player Movement Animation")
ax.set_xlabel("Field Length (yards)")
ax.set_ylabel("Field Width (yards)")
ax.grid(True)

# Plot initial point
point, = ax.plot([], [], 'bo', markersize=8)   # blue dot
path, = ax.plot([], [], 'b-', alpha=0.3)       # trail line

# Initialization function
def init():
    point.set_data([], [])
    path.set_data([], [])
    return point, path

# Update function
def update(frame):
    # up to current frame index
    xdata = one_player['x'].iloc[:frame]
    ydata = one_player['y'].iloc[:frame]
    point.set_data(one_player['x'].iloc[frame-1], one_player['y'].iloc[frame-1])
    path.set_data(xdata, ydata)
    return point, path

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(one_player),
                              init_func=init, interval=200, blit=True)

# Display the animation
from IPython.display import HTML
HTML(ani.to_jshtml())



# Normalize speeds for color mapping (0–1 scale)
norm = plt.Normalize(one_player['s'].min(), one_player['s'].max())
colors = plt.cm.plasma(norm(one_player['s']))  # 'plasma' gives a nice red-yellow color scale

# Set up figure
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(one_player['x'].min() - 2, one_player['x'].max() + 2)
ax.set_ylim(one_player['y'].min() - 2, one_player['y'].max() + 2)
ax.set_xlabel("Field Length (yards)")
ax.set_ylabel("Field Width (yards)")
ax.set_title("Player Movement with Speed & Frame Labels")
ax.grid(True)

# Initialize objects: moving point + path line + text label
point, = ax.plot([], [], 'o', color='blue', markersize=8)
path, = ax.plot([], [], '-', color='gray', alpha=0.4)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10)

# Initialization function
def init():
    point.set_data([], [])
    path.set_data([], [])
    time_text.set_text('')
    return point, path, time_text

# Update function for each frame
def update(frame):
    xdata = one_player['x'].iloc[:frame]
    ydata = one_player['y'].iloc[:frame]
    path.set_data(xdata, ydata)

    # Current position and color by speed
    point.set_data(one_player['x'].iloc[frame-1], one_player['y'].iloc[frame-1])
    point.set_color(colors[frame-1])

    # Update time label
    frame_id = one_player['frame_id'].iloc[frame-1]
    speed = one_player['s'].iloc[frame-1]
    time_text.set_text(f"Frame: {frame_id} | Speed: {speed:.2f} y/s")

    return point, path, time_text

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(one_player),
                              init_func=init, interval=200, blit=True)

HTML(ani.to_jshtml())


