import warnings
warnings.filterwarnings('ignore')


%%time
import pandas as pd
import os

# âœ… Correct dataset path
base_path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"

# Load one sample week (Week 1)
week1_path = os.path.join(base_path, "input_2023_w01.csv")
week1 = pd.read_csv(week1_path)
print("âœ… Loaded:", week1_path)
print("Shape:", week1.shape)
print("Columns:", list(week1.columns))

# Preview first few rows
display(week1.head())

# --- Supplementary data ---
supp_path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv"
supp = pd.read_csv(supp_path)
print("\nâœ… Supplementary data loaded.")
print("Shape:", supp.shape)
print("Columns:", list(supp.columns))
display(supp.head())



%%time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Base paths
input_path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w01.csv"
supp_path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv"

# Load
input_df = pd.read_csv(input_path)
supp_df = pd.read_csv(supp_path)

# --- Basic Info ---
print("Input shape:", input_df.shape)
print("Supplementary shape:", supp_df.shape)
print("\nInput columns:\n", input_df.columns.tolist())
print("\nSupplementary columns:\n", supp_df.columns.tolist())

# --- Missing values ---
missing = input_df.isnull().mean().sort_values(ascending=False)
print("\nMissing Values (%):\n", missing[missing > 0])

# --- Player & Role Distribution ---
plt.figure(figsize=(8,4))
sns.countplot(y=input_df['player_role'], order=input_df['player_role'].value_counts().index, palette="cool")
plt.title("Player Role Distribution (Week 1)")
plt.xlabel("Count")
plt.ylabel("Player Role")
plt.show()

# --- Player Position Distribution ---
plt.figure(figsize=(10,4))
sns.countplot(y=input_df['player_position'], order=input_df['player_position'].value_counts().index, palette="viridis")
plt.title("Player Position Distribution (Week 1)")
plt.show()

# --- Speed & Acceleration Distribution ---
fig, ax = plt.subplots(1, 2, figsize=(12,4))
sns.histplot(input_df['s'], bins=30, kde=True, ax=ax[0])
ax[0].set_title("Speed Distribution (yards/sec)")
sns.histplot(input_df['a'], bins=30, kde=True, ax=ax[1])
ax[1].set_title("Acceleration Distribution (yards/secÂ²)")
plt.show()

# --- Field Position Plot ---
sample_play = input_df[(input_df['game_id'] == input_df['game_id'].iloc[0]) &
                       (input_df['play_id'] == input_df['play_id'].iloc[0])]
plt.figure(figsize=(8,4))
sns.scatterplot(data=sample_play, x='x', y='y', hue='player_side', style='player_role', s=70)
plt.xlim(0, 120)
plt.ylim(0, 53.3)
plt.title("Player Positions Example (Sample Play)")
plt.show()

# --- Join with Supplementary ---
merged = input_df.merge(supp_df, on=['game_id', 'play_id'], how='left')
print("\nMerged shape:", merged.shape)
print("Unique plays:", merged['play_id'].nunique())

# --- Pass Result Distribution ---
plt.figure(figsize=(6,4))
sns.countplot(x=merged['pass_result'], order=merged['pass_result'].value_counts().index)
plt.title("Pass Result Distribution (Week 1)")
plt.xlabel("Pass Result")
plt.ylabel("Count")
plt.show()


%%time
import matplotlib.pyplot as plt
import seaborn as sns

# pick one sample play (you can change the index)
sample = input_df[['game_id', 'play_id']].drop_duplicates().iloc[0]
game_id = sample['game_id']
play_id = sample['play_id']

# subset for that play
play_df = input_df[(input_df['game_id'] == game_id) & (input_df['play_id'] == play_id)]

# plot trajectories for each player
plt.figure(figsize=(10,5))
sns.set_style("whitegrid")

for pid, pdata in play_df.groupby('nfl_id'):
    plt.plot(pdata['x'], pdata['y'], alpha=0.7, label=pdata['player_name'].iloc[0])

plt.scatter(play_df['ball_land_x'].iloc[0], play_df['ball_land_y'].iloc[0],
            c='red', s=120, marker='*', label='Ball landing spot')

plt.title(f"Player Trajectories (Game {game_id}, Play {play_id})")
plt.xlim(0,120)
plt.ylim(0,53.3)
plt.xlabel("Field X (yards)")
plt.ylabel("Field Y (yards)")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.show()


%%time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =====================================
# CONFIG
# =====================================
base_path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
week_file = "input_2023_w01.csv"  # â†� change to another week to animate a different play
save_dir = "frames"
os.makedirs(save_dir, exist_ok=True)

# =====================================
# LOAD SAMPLE PLAY
# =====================================
df = pd.read_csv(f"{base_path}/{week_file}")
print("Data shape:", df.shape)
print("Unique plays:", df[['game_id','play_id']].drop_duplicates().shape[0])

# Choose a single play to animate (to keep it small)
sample_play = df.groupby(['game_id','play_id']).size().idxmax()
data = df[(df['game_id']==sample_play[0]) & (df['play_id']==sample_play[1])].copy()

frame_ids = sorted(data['frame_id'].unique())
print(f"Animating Game {sample_play[0]}, Play {sample_play[1]} with {len(frame_ids)} frames.")

# =====================================
# DRAW FOOTBALL FIELD
# =====================================
def draw_field(ax):
    ax.set_facecolor('green')
    ax.plot([0,120],[0,0],color='white')
    ax.plot([0,120],[53.3,53.3],color='white')
    ax.plot([10,10],[0,53.3],color='white')
    ax.plot([110,110],[0,53.3],color='white')
    for x in range(20,110,10):
        ax.plot([x,x],[0,53.3],color='white',linestyle='--',linewidth=1)
    ax.axvspan(0,10,facecolor='blue',alpha=0.2)
    ax.axvspan(110,120,facecolor='red',alpha=0.2)
    ax.set_xlim(0,120)
    ax.set_ylim(0,53.3)
    ax.axis('off')

# =====================================
# ROLE COLORS
# =====================================
role_colors = {
    'Targeted Receiver': 'lime',
    'Passer': 'darkred',
    'Offense': 'red',
    'Defense': 'blue',
    'Football': 'gold',
}

def role_color(role):
    if pd.isna(role): return 'gray'
    for k,v in role_colors.items():
        if k.lower() in role.lower():
            return v
    return 'white'

# =====================================
# GENERATE FRAMES
# =====================================
for i, frame_id in enumerate(frame_ids):
    frame = data[data['frame_id']==frame_id]
    fig, ax = plt.subplots(figsize=(16,8))
    draw_field(ax)

    for _, row in frame.iterrows():
        c = role_color(row['player_role'])
        ax.scatter(row['x'], row['y'], c=c, s=150, edgecolor='black', alpha=0.8)
        if "target" in str(row['player_role']).lower():
            ax.text(row['x'], row['y']+2, row['player_name'], color='white',
                    ha='center', fontsize=8, fontweight='bold')

    # Draw ball landing point (if available)
    if not pd.isna(frame.iloc[0]['ball_land_x']):
        ax.scatter(frame.iloc[0]['ball_land_x'], frame.iloc[0]['ball_land_y'],
                   color='yellow', s=400, marker='*', edgecolor='black', linewidth=2)

    title = f"Game {sample_play[0]} | Play {sample_play[1]} | Frame {frame_id}"
    ax.set_title(title, fontsize=14, color='white', pad=10)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/frame_{i:04d}.png", dpi=120, bbox_inches='tight', facecolor='green')
    plt.close()

    if (i+1) % 10 == 0:
        print(f"Processed {i+1}/{len(frame_ids)} frames")

# =====================================
# CREATE GIF
# =====================================
print("Creating animation...")
frames = [Image.open(f"{save_dir}/frame_{i:04d}.png") for i in range(len(frame_ids)) if os.path.exists(f"{save_dir}/frame_{i:04d}.png")]
frames[0].save('nfl_play.gif', save_all=True, append_images=frames[1:], duration=120, loop=0)
print("âœ… Animation created: nfl_play.gif")


%%time
from IPython.display import HTML
HTML('<img src="nfl_play.gif" width="600">')


%%time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =====================================
# CONFIG
# =====================================
base_path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
week_file = "input_2023_w01.csv"  # â†� you can change to any week
save_dir = "frames_enhanced"
os.makedirs(save_dir, exist_ok=True)

# =====================================
# LOAD A SAMPLE PLAY
# =====================================
df = pd.read_csv(f"{base_path}/{week_file}")
print("Data shape:", df.shape)
sample_play = df.groupby(['game_id','play_id']).size().idxmax()
data = df[(df['game_id']==sample_play[0]) & (df['play_id']==sample_play[1])].copy()

frame_ids = sorted(data['frame_id'].unique())
print(f"Animating Game {sample_play[0]}, Play {sample_play[1]} with {len(frame_ids)} frames.")

# =====================================
# DRAW FOOTBALL FIELD
# =====================================
def draw_field(ax):
    ax.set_facecolor('#1E5128')
    ax.plot([0,120],[0,0],color='white')
    ax.plot([0,120],[53.3,53.3],color='white')
    for x in range(10,120,10):
        ax.plot([x,x],[0,53.3],color='white',linestyle='--',linewidth=0.8)
    ax.axvspan(0,10,facecolor='blue',alpha=0.15)
    ax.axvspan(110,120,facecolor='red',alpha=0.15)
    ax.set_xlim(0,120)
    ax.set_ylim(0,53.3)
    ax.axis('off')

# =====================================
# ROLE COLORS + MARKERS
# =====================================
def role_color(role):
    if pd.isna(role): return 'gray'
    if 'target' in str(role).lower(): return 'lime'
    if 'passer' in str(role).lower(): return 'orange'
    if 'defense' in str(role).lower(): return '#2E86C1'
    if 'offense' in str(role).lower(): return '#E74C3C'
    if 'ball' in str(role).lower(): return 'gold'
    return 'white'

def team_marker(side):
    return 'o' if side == 'home' else 's'

# =====================================
# MOTION TRAILS STORAGE
# =====================================
trail_length = 5
player_trails = {}

# =====================================
# GENERATE FRAMES
# =====================================
for i, frame_id in enumerate(frame_ids):
    frame = data[data['frame_id']==frame_id]
    fig, ax = plt.subplots(figsize=(16,8))
    draw_field(ax)

    for _, row in frame.iterrows():
        pid = row['nfl_id']
        player_trails.setdefault(pid, []).append((row['x'], row['y']))
        if len(player_trails[pid]) > trail_length:
            player_trails[pid].pop(0)

        # Draw trails
        trail = np.array(player_trails[pid])
        ax.plot(trail[:,0], trail[:,1], color=role_color(row['player_role']), alpha=0.4, linewidth=2)

        # Draw player
        ax.scatter(row['x'], row['y'],
                   c=role_color(row['player_role']),
                   s=150,
                   edgecolor='black',
                   marker=team_marker(row['player_side']),
                   zorder=3)

        # Add direction arrow (velocity vector)
        dx = np.cos(np.deg2rad(row['dir'])) * (row['s'] / 3)
        dy = np.sin(np.deg2rad(row['dir'])) * (row['s'] / 3)
        ax.arrow(row['x'], row['y'], dx, dy, color='white', head_width=0.6, alpha=0.6, zorder=2)

        # Label key players
        if "target" in str(row['player_role']).lower() or "passer" in str(row['player_role']).lower():
            ax.text(row['x'], row['y']+1.5, row['player_name'].split()[0],
                    color='white', fontsize=7, ha='center', weight='bold')

    # Ball landing marker
    if not pd.isna(frame.iloc[0]['ball_land_x']):
        ax.scatter(frame.iloc[0]['ball_land_x'], frame.iloc[0]['ball_land_y'],
                   color='yellow', s=300, marker='*', edgecolor='black', linewidth=1.5, zorder=4)

    title = f"Game {sample_play[0]} | Play {sample_play[1]} | Frame {frame_id}"
    ax.set_title(title, fontsize=14, color='white', pad=10)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/frame_{i:04d}.png", dpi=120, bbox_inches='tight', facecolor='#1E5128')
    plt.close()

    if (i+1) % 10 == 0:
        print(f"Processed {i+1}/{len(frame_ids)} frames")

# =====================================
# CREATE GIF
# =====================================
print("Creating enhanced animation...")
frames = [Image.open(f"{save_dir}/frame_{i:04d}.png") for i in range(len(frame_ids)) if os.path.exists(f"{save_dir}/frame_{i:04d}.png")]
frames[0].save('nfl_play_enhanced.gif', save_all=True, append_images=frames[1:], duration=120, loop=0)
print("âœ… Enhanced animation created: nfl_play_enhanced.gif")


from IPython.display import HTML
HTML('<img src="nfl_play_enhanced.gif" width="600">')


%%time
# =========================================================
# ğŸ�ˆ NFL Big Data Bowl 2026 â€” Player Tracking Animation V1
# =========================================================
# Author: dedquoc
# Description:
#   Visualize player movement from tracking data
#   and generate a dynamic football play animation.
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
base_path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
week_file = os.path.join(base_path, "input_2023_w01.csv")
data = pd.read_csv(week_file)

print("Data loaded âœ…")
print("Shape:", data.shape)
print("Columns:", list(data.columns))

# ---------------------------------------------------------
# 2. Select Example Play
# ---------------------------------------------------------
# Pick a random play or specific one to visualize
example_play = data.groupby(['game_id', 'play_id']).size().reset_index(name='n_frames')
example_play = example_play.sort_values('n_frames', ascending=False).iloc[0]  # longest play
game_id, play_id = example_play['game_id'], example_play['play_id']

print(f"Selected example play: Game {game_id}, Play {play_id}")

play_data = data[(data['game_id'] == game_id) & (data['play_id'] == play_id)].copy()
play_data = play_data.sort_values('frame_id').reset_index(drop=True)

# ---------------------------------------------------------
# 3. Draw Football Field Helper
# ---------------------------------------------------------
def draw_football_field(ax):
    ax.plot([0, 120], [0, 0], color='white', linewidth=2)
    ax.plot([0, 120], [53.3, 53.3], color='white', linewidth=2)
    ax.plot([10, 10], [0, 53.3], color='white', linewidth=2)
    ax.plot([110, 110], [0, 53.3], color='white', linewidth=2)
    ax.plot([60, 60], [0, 53.3], color='white', linewidth=2)
    ax.axvspan(0, 10, facecolor='blue', alpha=0.2)
    ax.axvspan(110, 120, facecolor='red', alpha=0.2)
    for x in range(20, 110, 10):
        ax.plot([x, x], [0, 53.3], color='white', linestyle='--', linewidth=1)

# ---------------------------------------------------------
# 4. Define Role Colors
# ---------------------------------------------------------
role_colors = {
    'Offense': 'red', 'Defense': 'blue', 'Football': 'gold', 'Ball': 'gold',
    'QB': 'darkred', 'WR': 'orange', 'RB': 'yellow', 'TE': 'goldenrod',
    'OL': 'lightcoral', 'LB': 'lightblue', 'DB': 'cyan', 'DL': 'navy',
    'S': 'deepskyblue', 'CB': 'aqua', 'K': 'purple', 'P': 'violet',
    'Targeted Receiver': 'lime'
}

# ---------------------------------------------------------
# 5. Generate Frame Images
# ---------------------------------------------------------
os.makedirs('frames', exist_ok=True)
frame_ids = sorted(play_data['frame_id'].unique())

for i, frame_id in enumerate(frame_ids):
    frame = play_data[play_data['frame_id'] == frame_id]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor('green')
    draw_football_field(ax)

    # Plot by player role
    for role, color in role_colors.items():
        subset = frame[frame['player_role'].fillna('').str.contains(role.split()[0], case=False)]
        if not subset.empty:
            ax.scatter(subset['x'], subset['y'], s=120, color=color, alpha=0.8, edgecolor='black', label=role)

    # Add ball landing point
    if 'ball_land_x' in frame.columns and not pd.isna(frame.iloc[0]['ball_land_x']):
        ax.scatter(frame.iloc[0]['ball_land_x'], frame.iloc[0]['ball_land_y'],
                   s=400, color='yellow', marker='*', edgecolor='black', linewidth=2, label='Ball Landing')

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_title(f"Game {game_id} | Play {play_id} | Frame {frame_id}", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.7)
    ax.axis('off')

    plt.savefig(f"frames/frame_{i:04d}.png", dpi=120, bbox_inches='tight')
    plt.close(fig)

print(f"âœ… Generated {len(frame_ids)} frame images")

# ---------------------------------------------------------
# 6. Create GIF Animation
# ---------------------------------------------------------
frames = []
for i in range(len(frame_ids)):
    frame_file = f"frames/frame_{i:04d}.png"
    if os.path.exists(frame_file):
        frames.append(Image.open(frame_file))

if frames:
    frames[0].save('football_tracking.gif', save_all=True,
                   append_images=frames[1:], duration=120, loop=0)
    print("ğŸ�¥ Animation created: football_tracking.gif")
else:
    print("âš ï¸� No frames found for animation")

from IPython.display import Image as IPImage
IPImage(open('football_tracking.gif', 'rb').read())

# ---------------------------------------------------------
# 7. Simple Play Summary Metrics
# ---------------------------------------------------------
summary = {
    "game_id": game_id,
    "play_id": play_id,
    "total_players": play_data['nfl_id'].nunique(),
    "frames": play_data['frame_id'].nunique(),
    "avg_speed": round(play_data['s'].mean(), 2),
    "max_speed": round(play_data['s'].max(), 2),
    "avg_accel": round(play_data['a'].mean(), 2)
}

summary_df = pd.DataFrame([summary])
display(summary_df)

print("\nâœ… Version 1 complete â€” ready to share on Kaggle!")

