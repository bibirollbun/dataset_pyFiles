#Importing libraries
import pandas as pd
import numpy as np
import matplotlib as plt
import glob


# path = "/content/drive/MyDrive/2026_BDB_PK/" #Change your path depending upon the file location


# #Vertical stacking of all the 18 input files and naming it 'all_input'
# all_input_files = sorted(glob.glob(path + "input_*.csv"))

# all_input = pd.concat((pd.read_csv(f) for f in all_input_files), ignore_index=True)
# all_input.shape


# #Vertical stacking of all the 18 output files and naming it 'all_output'
# all_output_files = sorted(glob.glob(path + "output_*.csv"))

# all_output = pd.concat((pd.read_csv(r) for r in all_output_files), ignore_index=True)
# all_output.shape


# #Filtering out all false instances from the input file
# all_false = all_input[all_input['player_to_predict']==False].reset_index(drop = True)


# #Filtering out all true instances from the input file
# all_true = all_input[all_input['player_to_predict']==True].reset_index(drop=True)


# #Forming unique combinations to parse through dataset
# unique_set = None
# unique_set = all_true[['game_id', 'play_id', 'nfl_id']]
# unique_set = unique_set.drop_duplicates()

# unique_set_list = []
# for x in unique_set.values:
#   unique_set_list.append(x.tolist())


# #This code merges TRUE players across input files and merges the corresponding output file with continued frame id in order

# from tqdm import tqdm
# pieces = []
# for x in tqdm(unique_set.values, desc = "Processing"):
#   a = all_true[(all_true[['game_id', 'play_id', 'nfl_id']]==x).all(axis=1)]
#   b = all_output[(all_output[['game_id', 'play_id', 'nfl_id']]==x).all(axis=1)]
#   b.loc[:,'frame_id'] = b.loc[:,'frame_id'] + max(a['frame_id'])
#   pieces.append(a)
#   pieces.append(b)


# #Combining with the remaining FALSE player data
# combined_data = pd.concat(pieces+[all_false], ignore_index = True)


# #Identifying the columns to forward fill and performing the operation
# columns_to_ffill = ['player_to_predict', 'play_direction', 'absolute_yardline_number', 'player_name', 'player_height',
#                      'player_weight', 'player_birth_date', 'player_position', 'player_side', 'player_role',
#                      'num_frames_output', 'ball_land_x', 'ball_land_y']


# for cols in columns_to_ffill:
#   combined_data[cols] = combined_data[cols].ffill()


#Saving the data as 'super_data.csv'
##combined_data.to_csv('/content/drive/MyDrive/2026_BDB_PK/super_data.csv', index=False)


# #Reading back the data
# import pandas as pd
# data = pd.read_csv('/content/drive/MyDrive/2026_BDB_PK/super_data.csv')


# data.shape


# #Reading supplementary data
# supplementary = pd.read_csv('/content/drive/MyDrive/2026_BDB_PK/supplementary_data.csv')


# #Identifying columns to be used from supplementary data to join with the super_data
# cols_to_join = ['game_id', 'play_id', 'quarter', 'game_clock', 'down', 'yards_to_go', 'home_team_abbr', 'visitor_team_abbr', 'possession_team',
#        'defensive_team', 'yardline_side', 'yardline_number', 'play_nullified_by_penalty', 'pass_result', 'pass_length',
#        'offense_formation',  'receiver_alignment', 'route_of_targeted_receiver', 'play_action', 'dropback_type', 'dropback_distance', 'pass_location_type',
#         'defenders_in_the_box', 'team_coverage_man_zone', 'team_coverage_type', 'penalty_yards' , 'pre_penalty_yards_gained', 'yards_gained' ]

# supplementary = supplementary[cols_to_join]


# #Merging both the dataframes using left join

# merged_super_data = data.merge(supplementary, on = ['game_id', 'play_id'], how = 'left')


#Saving the final merged version as 'merged_super_data.csv'
##merged_super_data.to_csv('/content/drive/MyDrive/2026_BDB_PK/merged_super_data.csv', index=False)


#Importing libraries
import pandas as pd
import numpy as np
import matplotlib as plt
import warnings
warnings.filterwarnings("ignore")


# Install dependencies as needed:
#pip install kagglehub[pandas-datasets]
import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter

print("This block of code takes a while to execute (~ 1-2 minutes)")

# Set the path to the file you'd like to load
file_path = "merged_super_data.csv"

# Load the latest version
merged = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "yashkalpeshshah1196/merged-super-data",
  file_path,
  # Provide any additional arguments like
  # sql_query or pandas_kwargs. See the
  # documentation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

merged = pd.DataFrame(merged)
print("Data imported successfully!")


merged


#Creating list of unique combinations of game_id and play_id from the main data
merge_set = merged[['game_id', 'play_id']].drop_duplicates()
merge_set_list = []
for x in merge_set.values:
  merge_set_list.append(x.tolist())


#We are giving the user two options - run a random play / enter a valid combination of game_id, play_id

import random

#To use a random play, remove the comment from next 2 lines
game_id, play_id = random.choice(merge_set_list)
game_id,play_id

print(f"\n{'='*100}")
print(f"RANDOMLY SELECTED: Game {game_id}, Play {play_id}")
print(f"{'='*100}\n")

# Lock these values for all subsequent cells
SELECTED_GAME_ID = game_id
SELECTED_PLAY_ID = play_id

#To use a manual play, remove the comments from next 2 lines
# game_id, play_id = int(input("Enter Game Id: ")),int(input("Enter Play Id: "))
# game_id, play_id


play_data = merged[(merged[['game_id', 'play_id']]==[game_id,play_id]).all(axis=1)].reset_index(drop=True)
play_data


#Calculating frame details for the corresponding play
total_frames = max(play_data['frame_id'])
output_frames = int(max(play_data['num_frames_output']))
output_start = total_frames - output_frames + 1

print(f"The loaded play consists of {total_frames} frames in total. It has {output_frames} frames in the output which begins at frame {output_start}")


#Extracting the coordinates of the Landing Point of the Ball rounded off to 2 decimal places
LP_x, LP_y = play_data[['ball_land_x', 'ball_land_y']].round(3).iloc[0]
LP_x, LP_y


# ========= COLOR MAP SETUP =========
# This block defines the color assignment logic for all artifacts
# Run this FIRST before any other scripts that follow

position_base_colors = {
    'CB': (0.1, 0.4, 0.9),      # Blue
    'S': (1.0, 0.5, 0.0),       # Orange (Safety)
    'SS': (1.0, 0.6, 0.0),      # Light Orange (Strong Safety)
    'FS': (1.0, 0.7, 0.0),      # Lighter Orange (Free Safety)
    'WR': (0.9, 0.1, 0.1),      # Red
    'OLB': (0.2, 0.8, 0.2),     # Green (Outside Linebacker)
    'ILB': (0.3, 0.9, 0.3),     # Light Green (Inside Linebacker)
    'RB': (0.9, 0.2, 0.7),      # Pink
    'TE': (0.8, 0.2, 0.8),      # Magenta
    'FB': (0.7, 0.3, 0.9),      # Purple
    'MLB': (0.4, 0.9, 0.4),     # Light Green (Middle Linebacker)
    'DE': (1.0, 0.2, 0.2),      # Bright Red
    'DT': (0.9, 0.8, 0.0),      # Gold
    'NT': (0.95, 0.85, 0.1),    # Light Gold
    'LB': (0.5, 0.8, 0.5),      # Medium Green
    'QB': (0.2, 0.2, 0.8),      # Dark Blue
    'T': (0.6, 0.3, 0.9),       # Light Purple
    'P': (0.5, 0.5, 0.5),       # Gray
    'K': (0.6, 0.6, 0.6),       # Light Gray
}

def vary_color_hue(base_color, variation_idx, position=None, max_variations=5):
    """
    Vary a base color by adjusting brightness for players at same position.
    Uses HSV color space to lighten/darken the color instead of rotating hue.

    Args:
        base_color: (r, g, b) tuple
        variation_idx: Index of variation (0, 1, 2, etc.)
        position: Player position code (e.g., 'CB', 'S', 'OLB')
        max_variations: Maximum number of variations

    Returns:
        (r, g, b) tuple with adjusted brightness
    """
    import colorsys

    if variation_idx == 0:
        return base_color

    # Convert RGB to HSV
    h, s, v = colorsys.rgb_to_hsv(base_color[0], base_color[1], base_color[2])

    # Adjust VALUE (brightness) with clear distinctions
    # Create alternating bright and dark versions
    # variation_idx=1: much darker, variation_idx=2: much lighter, variation_idx=3: darker, etc.
    if variation_idx == 1:
        v = v * 0.4  # Much darker
    elif variation_idx == 2:
        v = min(1.0, v * 1.6)  # Much lighter
    elif variation_idx == 3:
        v = v * 0.25  # Even darker
    elif variation_idx == 4:
        v = min(1.0, v * 1.8)  # Even lighter

    # Keep saturation high for color distinctness
    s = max(0.7, s)

    # Convert back to RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b)

def build_color_map(all_players_df, df_play_all, targeted_receiver=None):
    """
    Build color_map based on ALL players in the play.

    Rules:
    - ALL players get color assignments (ensures consistent colors everywhere)
    - Targeted receiver: always bright RED (overrides position)
    - Offensive players: always bright RED
    - Defensive players: bright color based on their POSITION
      (players with same position get different hues for distinction)

    Args:
        all_players_df: DataFrame with player_name, player_side, player_position columns
                       (typically created from df_play_all[['player_name', 'player_side', 'player_position']].drop_duplicates())
        df_play_all: DataFrame with all play data (used for position info if needed)
        targeted_receiver: Name of targeted receiver (optional, will be highlighted RED)

    Returns:
        color_map: Dictionary mapping player_name -> (r, g, b) tuple
    """
    color_map = {}

    # Count players per position to assign different hues
    position_counts = {}
    position_indices = {}

    # First pass: count defensive players per position
    for idx, row in all_players_df.iterrows():
        player_side = row['player_side'] if 'player_side' in row else None
        position = row['player_position'] if 'player_position' in row else 'Unknown'

        # Only count defensive players (offense all get red)
        if player_side == 'Defense':
            position_counts[position] = position_counts.get(position, 0) + 1

    # Initialize position tracking
    position_indices = {pos: 0 for pos in position_counts}

    # Second pass: assign colors to ALL players
    for idx, row in all_players_df.iterrows():
        player_name = row['player_name']
        player_side = row['player_side'] if 'player_side' in row else None
        position = row['player_position'] if 'player_position' in row else 'Unknown'

        # Targeted receiver always bright red
        if player_name == targeted_receiver:
            color_map[player_name] = (1.0, 0.0, 0.0)
        # Offensive players always bright red
        elif player_side == 'Offense':
            color_map[player_name] = (1.0, 0.0, 0.0)
        # Defensive players get bright color by position with variation
        elif player_side == 'Defense':
            base_color = position_base_colors.get(position, (0.5, 0.5, 0.5))
            variation_idx = position_indices.get(position, 0)
            color_map[player_name] = vary_color_hue(base_color, variation_idx, position=position)
            position_indices[position] += 1
        else:
            color_map[player_name] = (0.5, 0.5, 0.5)

    return color_map

def get_player_color(player_name, color_map, player_side=None, has_pei=True):
    """
    Get color for a player based on color_map.

    Args:
        player_name: Name of player
        color_map: Color map dictionary
        player_side: 'Offense' or 'Defense' (used if player not in color_map)
        has_pei: Whether player has PEI values (if False, returns gray)

    Returns:
        (r, g, b) tuple color
    """
    if not has_pei:
        return (0.6, 0.6, 0.6)  # Gray for no PEI

    return color_map.get(player_name, (0.5, 0.5, 0.5))

print("Color map setup block loaded")


import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from matplotlib import rc
import matplotlib.patheffects as pe
import gc

# # ---- Render animation inline (no ffmpeg needed)
rc('animation', html='jshtml')
plt.rcParams['figure.max_open_warning'] = 0

# ========= CONFIG =========

FIELD_W = 53.3
FIELD_L = 120.0
FRAME_SKIP = 1
INTERVAL_MS = 100

# Use metadata to ensure correct play
if 'single_play_metadata' in dir():
    game_id = single_play_metadata['game_id']
    play_id = single_play_metadata['play_id']
    pass_result = single_play_metadata['pass_result']

print(f"\n{'='*100}")
print(f"ANIMATION: Field Only - Game {game_id}, Play {play_id}")
print(f"Speed: Real-time (10 frames per second)")
print(f"{'='*100}\n")

# ========= LOAD & PREPROCESS =========
print("Loading data...")
gc.collect()

# Use only necessary columns to save memory
cols_to_use = ['game_id','play_id','frame_id','nfl_id','player_name','player_role', 'pass_result',
        'player_side','x','y','num_frames_output','ball_land_x','ball_land_y',
        'player_to_predict','ball_x','ball_y','absolute_yardline_number','yards_gained','team_coverage_type','player_position']

exist_cols = [c for c in cols_to_use if c in play_data.columns]
# Use play_data (created by Document 31 at the end)
if 'play_data' not in dir():
    raise NameError("play_data not found. Run play selection cell first!")

print(f"âœ“ Using play_data: Game {game_id}, Play {play_id}")
print(f"  Rows: {len(play_data)}, Frames: {play_data['frame_id'].nunique()}\n")

# ========= USE METADATA FROM PLAY SELECTION =========
if 'single_play_metadata' in dir():
    game_id = single_play_metadata['game_id']
    play_id = single_play_metadata['play_id']
    pass_result = single_play_metadata['pass_result']
    targeted_route = single_play_metadata['targeted_route']
    targeted_receiver = single_play_metadata['targeted_receiver']
    print(f"âœ“ Using saved metadata: Game {game_id}, Play {play_id}, Result {pass_result}\n")
else:
    print(f"âš  Warning: Using global variables (may be incorrect)")

play = play_data[exist_cols].copy()

# CRITICAL: Verify we have all frames including TRUE window
print(f"Verifying frame coverage:")
print(f"  Total frames in play: {play['frame_id'].min()} to {play['frame_id'].max()}")
print(f"  TRUE window expected: {output_start} to {total_frames}")

if play['frame_id'].max() < total_frames:
    raise ValueError(f"play_data is missing frames! Has up to {play['frame_id'].max()}, need {total_frames}")

print(f"  âœ“ All frames present\n")

print(f"âœ“ Using df_play_all: Game {game_id}, Play {play_id}")
print(f"  Rows: {len(play)}, Frames: {play['frame_id'].nunique()}\n")


#play['frame_id'] = pd.to_numeric(play['frame_id'], errors='coerce')
play = play.sort_values(['frame_id','nfl_id']).reset_index(drop=True)

# Get basic info
N = output_frames
fmin = 1
fmax = total_frames

# Ball landing
bx = LP_x
by = LP_y

# ========= EXTRACT BALL TRAJECTORY =========
print("Processing ball trajectory...")
ball_trajectory = {}
frames_all = play['frame_id'].unique()
frame_map = {f: i for i, f in enumerate(frames_all)}

# Determine throw frame using num_frames_output
throw_frame = output_start
true_end = total_frames
print(f"  Throw frame: {throw_frame}")
print(f"  End frame: {true_end}")
print(f"  Ball landing: ({bx}, {by})")

# If no direct ball data, calculate trajectory from passer location to landing
print("  Calculating ball trajectory from passer to landing...")

passer_data = play[play['player_role'] == 'Passer'].sort_values('frame_id')

# Get passer's LAST position (not first)
passer_x = float(passer_data['x'].iloc[-1])
passer_y = float(passer_data['y'].iloc[-1])
passer_last_frame = int(passer_data['frame_id'].iloc[-1])

print(f"  Passer's last location: ({passer_x:.1f}, {passer_y:.1f}) at frame {passer_last_frame}")
for fid in frames_all:
  if fid >= throw_frame:
    progress = (fid - throw_frame) / max(1, (true_end - throw_frame))
    progress = np.clip(progress, 0.0, 1.0)
    ball_x_est = float(passer_x + (bx - passer_x) * progress)
    ball_y_est = float(passer_y + (by - passer_y) * progress)
    ball_trajectory[fid] = (np.round(ball_x_est,2),np.round(ball_y_est,2))

print(f"  Calculated trajectory for {len(ball_trajectory)} frames")


# Targeted receiver
tr_data = play[play['player_role'] == 'Targeted Receiver']
targeted_receiver = tr_data['player_name'].iloc[0]

print(f"  Frames: {fmin} to {fmax}")
print(f"  Targeted Receiver: {targeted_receiver}")

# ========= BUILD PLAYER DATA =========
print("Processing players...")

player_list = play[['nfl_id','player_name']].drop_duplicates().values
players = []

frame_count = len(frames_all)

# Build color_map from ALL players in this play (consistent approach)
print("  Building color_map from all players in play...")
all_players_data = play[['player_name', 'player_side', 'player_position']].drop_duplicates()
color_map = build_color_map(all_players_data, play, targeted_receiver)
print(f"  âœ“ Color map built with {len(color_map)} players")


# Build player data
for pid, pname in player_list:
    p_data = play[play['nfl_id'] == pid].sort_values('frame_id')
    x_al = np.full(frame_count, np.nan, dtype=np.float32)
    y_al = np.full(frame_count, np.nan, dtype=np.float32)

    for _, row in p_data.iterrows():
        idx = frame_map.get(int(row['frame_id']))
        if idx is not None:
            x_al[idx] = float(row['x'])
            y_al[idx] = float(row['y'])

    p_role = p_data['player_role'].iloc[0] if 'player_role' in p_data.columns else None
    p_side = p_data['player_side'].iloc[0] if 'player_side' in p_data.columns else None
    p_pred = bool(p_data['player_to_predict'].iloc[0]) if 'player_to_predict' in p_data.columns else False

    # Check if player has entry in color_map
    has_color = pname in color_map

    # If player_to_predict == False, use gray color
    if not p_pred:
        color = (0.6, 0.6, 0.6)  # Gray for non-tracked players
        marker_size = 4
        line_width = 0.8
        alpha = 0.9
        is_key_player = False
    else:
        # Get color using setup function
        color = get_player_color(pname, color_map, player_side=p_side, has_pei=has_color)

        # Determine styling based on whether player is tracked/targeted
        if pname == targeted_receiver:
            marker_size = 8
            line_width = 2.5
            alpha = 1.0
            is_key_player = True
        elif p_side == 'Defense' or p_role == 'Targeted Receiver':
            marker_size = 7
            line_width = 2.2
            alpha = 1.0
            is_key_player = True
        else:
            marker_size = 5
            line_width = 1.2
            alpha = 0.6
            is_key_player = False

    players.append({
        'name': pname,
        'x': x_al,
        'y': y_al,
        'color': color,
        'marker_size': marker_size,
        'line_width': line_width,
        'alpha': alpha,
        'is_key_player': is_key_player
    })

print(f"  {len(players)} players loaded")
print(f"  {frame_count} frames total")

# ========= FIELD ANIMATION =========
fig, ax_field = plt.subplots(figsize=(14, 7), dpi=100)

# Main field background
ax_field.add_patch(patches.Rectangle((0, 0), FIELD_L, FIELD_W,
                               facecolor='#8DBF87', edgecolor='#2E7D32', linewidth=2, zorder=0))

# Endzones (darker shade at each end)
ax_field.add_patch(patches.Rectangle((0, 0), 10, FIELD_W,
                               facecolor='#5a8f5a', edgecolor='#2E7D32', linewidth=1, zorder=0, alpha=0.8))
ax_field.add_patch(patches.Rectangle((FIELD_L - 10, 0), 10, FIELD_W,
                               facecolor='#5a8f5a', edgecolor='#2E7D32', linewidth=1, zorder=0, alpha=0.8))

# Yard lines every 10 yards
for xline in range(10, int(FIELD_L), 10):
    ax_field.plot([xline, xline], [0, FIELD_W], color='white', linewidth=0.8, alpha=0.4, zorder=0)

# Scrimmage line based on absolute_yardline_number
scrimmage_x = 50.0  # Default midfield approximation
if 'absolute_yardline_number' in play.columns:
    yardline_data = play['absolute_yardline_number'].dropna()
    if len(yardline_data) > 0:
        scrimmage_x = float(yardline_data.iloc[0])
        print(f"  Scrimmage line at: {scrimmage_x:.1f}")

# Draw scrimmage line as a thick yellow line
ax_field.plot([scrimmage_x, scrimmage_x], [0, FIELD_W], color='yellow', linestyle='-', linewidth=3, alpha=0.7, zorder=1, label='Scrimmage Line')

ax_field.set_xlim(-1, FIELD_L + 1)
ax_field.set_ylim(-1, FIELD_W + 1)
ax_field.set_aspect('equal', adjustable='box')
ax_field.set_xlabel('X (yards)', fontsize=10)
ax_field.set_ylabel('Y (yards)', fontsize=10)

# Get additional play information for title
yards_gained = 0.0
team_coverage_type = "Unknown"

if 'yards_gained' in play.columns:
    yg_data = play['yards_gained'].dropna()
    if len(yg_data) > 0:
        yards_gained = float(yg_data.iloc[0])

if 'team_coverage_type' in play.columns:
    cov_data = play['team_coverage_type'].dropna()
    if len(cov_data) > 0:
        team_coverage_type = str(cov_data.iloc[0])

ax_field.set_title(f'Game {game_id} | Play {play_id} | Result: {play["pass_result"].values[0]} | Yards: {yards_gained:.1f} | Coverage: {team_coverage_type}\nBright=Tracked | Gray=Not Tracked',fontsize=12, fontweight='bold')

# Ball landing marker
ax_field.plot(bx, by, marker='X', markersize=12, color='black', mew=1.5, zorder=3, label='Ball Landing')

# Create plot objects for all players
trail_lines = []
point_dots = []
text_labels = []

for p in players:
    (ln,) = ax_field.plot([], [], color=p['color'], lw=p['line_width'], alpha=p['alpha'], zorder=2)
    trail_lines.append(ln)

    pt = ax_field.plot([], [], marker='o', markersize=p['marker_size'], color=p['color'],
                 mec='k', mew=0.4, zorder=4, alpha=p['alpha'])[0]
    point_dots.append(pt)

    txt = ax_field.text(0, 0, p['name'], fontsize=7, color=p['color'], weight='bold',
                  zorder=5, alpha=p['alpha'], visible=True, path_effects=[pe.withStroke(linewidth=0.2, foreground='black')])
    text_labels.append(txt)

# Ball trajectory
(ball_trail_line,) = ax_field.plot([], [], color='goldenrod', linestyle=':', linewidth=2.5, alpha=0.8, zorder=3, label='Ball Trail')

# Football patches
ball_patches = []

# Frame label
frame_label = ax_field.text(2, FIELD_W - 1, "", fontsize=11, color="black", zorder=6, weight='bold')

# Legend
ax_field.legend(loc='lower right', fontsize=9, framealpha=0.95)

# Subsample frames for animation
anim_frames = frames_all[::FRAME_SKIP]

print(f"Animating {len(anim_frames)} frames (real-time speed: 10 fps)...\n")

def update(anim_idx):
    frame_idx = anim_idx * FRAME_SKIP
    if frame_idx >= frame_count:
        frame_idx = frame_count - 1

    fnum = int(frames_all[frame_idx])
    frame_label.set_text(f"Frame {anim_idx+1}/{len(anim_frames)}")

    # ===== UPDATE BALL TRAJECTORY =====
    current_ball_trail_x = []
    current_ball_trail_y = []

    for bf in sorted(ball_trajectory.keys()):
        if bf <= fnum:
            bx_pos, by_pos = ball_trajectory[bf]
            current_ball_trail_x.append(bx_pos)
            current_ball_trail_y.append(by_pos)

    if len(current_ball_trail_x) > 0:
        ball_trail_line.set_data(current_ball_trail_x, current_ball_trail_y)

        # Remove old football patches
        for patch in ball_patches:
            patch.remove()
        ball_patches.clear()

        # Draw new football at current position
        current_x = current_ball_trail_x[-1]
        current_y = current_ball_trail_y[-1]

        # Calculate direction angle for football orientation
        angle = 0  # default horizontal
        if len(current_ball_trail_x) > 1:
            # Get direction from last two positions
            dx = current_ball_trail_x[-1] - current_ball_trail_x[-2]
            dy = current_ball_trail_y[-1] - current_ball_trail_y[-2]

            # Calculate angle in degrees (pointing in direction of travel)
            if dx != 0 or dy != 0:
                angle = np.degrees(np.arctan2(dy, dx))

        # Draw football oriented in direction of travel
        football = patches.Ellipse((current_x, current_y), width=1.2, height=0.6,
                                  angle=angle, color='brown', zorder=10, ec='black', linewidth=1.2)
        ax_field.add_patch(football)
        ball_patches.append(football)

        # Draw laces perpendicular to direction of travel
        lace_count = 5
        lace_length = 0.5

        # Rotate lace positions based on football orientation
        angle_rad = np.radians(angle)
        for i in range(lace_count):
            # Lace position along the center (rotated)
            t = -0.4 + (i * 0.2)
            lace_x1 = current_x + t * np.cos(angle_rad)
            lace_y1 = current_y + t * np.sin(angle_rad)

            # Perpendicular direction for lace width
            perp_x = -np.sin(angle_rad) * (lace_length / 2)
            perp_y = np.cos(angle_rad) * (lace_length / 2)

            lace_x2 = lace_x1 + perp_x
            lace_y2 = lace_y1 + perp_y
            lace_x3 = lace_x1 - perp_x
            lace_y3 = lace_y1 - perp_y

            line = ax_field.plot([lace_x2, lace_x3], [lace_y2, lace_y3],color='white', linewidth=0.4, zorder=11)[0]
            ball_patches.append(line)
    else:
        ball_trail_line.set_data([], [])
        for patch in ball_patches:
            patch.remove()
        ball_patches.clear()

    # ===== UPDATE PLAYERS =====
    for p_idx, p in enumerate(players):
        x = p['x'][:frame_idx + 1]
        y = p['y'][:frame_idx + 1]

        valid = ~(np.isnan(x) | np.isnan(y))
        if np.any(valid):
            trail_lines[p_idx].set_data(x[valid], y[valid])
            point_dots[p_idx].set_data([x[valid][-1]], [y[valid][-1]])
            text_labels[p_idx].set_position((x[valid][-1] + 0.3, y[valid][-1] + 0.3))
            text_labels[p_idx].set_visible(True)
        else:
            trail_lines[p_idx].set_data([], [])
            point_dots[p_idx].set_data([], [])
            text_labels[p_idx].set_visible(False)

    # Return all objects including current ball patches
    return trail_lines + point_dots + text_labels + [frame_label, ball_trail_line] + ball_patches

anim = FuncAnimation(fig, update, frames=len(anim_frames), interval=INTERVAL_MS,
                    blit=True, repeat=True, repeat_delay=1000)

plt.tight_layout()
print("Animation complete!\n")

anim
# ========= SAVE VARIABLES FOR EXPORT =========
print("\n" + "="*100)
print("SAVING VARIABLES FOR EXPORT")
print("="*100)

# Save ball trajectory for export
ball_trajectory_export = ball_trajectory.copy()
print(f"âœ“ Saved ball_trajectory_export: {len(ball_trajectory_export)} frames")

# Save frames_all for export (THIS IS CRITICAL)
frames_all_export = frames_all.copy()
print(f"âœ“ Saved frames_all_export: {len(frames_all_export)} frames")

# Verify other required variables exist
print(f"âœ“ players: {len(players)} players")
print(f"âœ“ scrimmage_x: {scrimmage_x:.1f}")

print("="*100 + "\n")

# ========= DISPLAY FIELD ANIMATION AS GIF IN OUTPUT =========
print("\n" + "="*100)
print("RENDERING FIELD ANIMATION AS GIF FOR DISPLAY")
print("="*100 + "\n")

try:
    from PIL import Image
    from IPython.display import Image as IPImage, display
    import io
    import os
    
    # Create temporary GIF in memory
    print(f"Extracting {len(anim_frames)} frames from animation...")
    
    field_frames = []
    
    for anim_idx in range(len(anim_frames)):
        # Update animation to this frame
        update(anim_idx)
        
        # Draw the current state
        fig.canvas.draw()
        
        # Convert to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
        buf.seek(0)
        
        # Open and convert to RGB
        frame = Image.open(buf)
        frame_rgb = frame.convert('RGB')
        field_frames.append(frame_rgb)
        buf.close()
        
        # Progress update every 10%
        if (anim_idx + 1) % max(1, len(anim_frames) // 10) == 0:
            print(f"  âœ“ Extracted {anim_idx + 1}/{len(anim_frames)} frames")
    
    print(f"\nCreating GIF for display...")
    
    # Create GIF in memory buffer
    gif_buffer = io.BytesIO()
    field_frames[0].save(
        gif_buffer,
        format='GIF',
        save_all=True,
        append_images=field_frames[1:],
        duration=100,  # 100ms per frame = 10 fps
        loop=0,
        optimize=False
    )
    gif_buffer.seek(0)
    
    gif_size = len(gif_buffer.getvalue()) / (1024*1024)
    
    print(f"\nâœ“ Field animation GIF created!")
    print(f"  Frames: {len(field_frames)}")
    print(f"  Size: {gif_size:.1f} MB")
    print("\n" + "="*100 + "\n")
    
    # Display the GIF in the notebook output
    print("Displaying animation:\n")
    display(IPImage(data=gif_buffer.getvalue()))
    
except Exception as e:
    print(f"âš  Error creating field animation GIF: {str(e)}\n")


# ========= USE CORRECT PLAY METADATA =========
if 'single_play_metadata' in dir():
    game_id = single_play_metadata['game_id']
    play_id = single_play_metadata['play_id']
    pass_result = single_play_metadata['pass_result']
    print(f"Distance Graph: Game {game_id}, Play {play_id}\n")
    
# Calculate distance data for each player
dlp_results = []

for player in players:
    pname = player['name']
    x_al = player['x']
    y_al = player['y']

    # Get valid frames for this player
    valid_mask = ~(np.isnan(x_al) | np.isnan(y_al))
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 2:
        continue

    # Collect ALL frames starting from frame 1 for visualization
    all_frame_indices = valid_indices

    distances_all = []
    all_frame_nums = []

    for idx in all_frame_indices:
        x_pos = float(x_al[idx])
        y_pos = float(y_al[idx])
        dist = np.sqrt((x_pos - bx)**2 + (y_pos - by)**2)
        distances_all.append(dist)
        all_frame_nums.append(int(frames_all[idx]))

    if distances_all:
        dlp_results.append({
            'name': pname,
            'frames': all_frame_nums,
            'distances': distances_all,
            'color': player['color'],
            'alpha': player['alpha'],
            'line_width': player['line_width'],
            'is_key_player': player['is_key_player'],
        })

# Create figure
fig, ax = plt.subplots(figsize=(14, 8), dpi=80)

# Plot distance to ball for each player
for result in dlp_results:
    # Only add to legend if player is NOT gray (i.e., is tracked)
    if result['color'] != (0.6, 0.6, 0.6):
        ax.plot(result['frames'], result['distances'],
               color=result['color'], linewidth=result['line_width'],
               alpha=result['alpha'], label=result['name'])
    else:
        ax.plot(result['frames'], result['distances'],
               color=result['color'], linewidth=result['line_width'],
               alpha=result['alpha'])

# Add optimal path for players in color_map (starting from throw_frame)
# Calculate optimal distance progression - only plot from throw_frame onwards
min_frame = int(frames_all[0])
max_frame = int(frames_all[-1])

# For each player with color assignment, calculate optimal distance
for player in players:
    # if not player['has_color']:
    #     continue

    # Skip gray players from optimal path plotting
    if player['color'] == (0.6, 0.6, 0.6):
        continue

    pname = player['name']
    x_al = player['x']
    y_al = player['y']

    # Find first valid position in TRUE window (starting from throw_frame)
    valid_mask = ~(np.isnan(x_al) | np.isnan(y_al))
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        continue

    # Find first position at or after throw_frame
    throw_idx = None
    for idx in valid_indices:
        if frames_all[idx] >= output_start:
            throw_idx = idx
            break

    if throw_idx is None:
        continue

    # Distance at throw_frame
    xs = float(x_al[throw_idx])
    ys = float(y_al[throw_idx])
    d_at_throw = np.sqrt((xs - bx)**2 + (ys - by)**2)

    # Create optimal progression from throw_frame to landing
    true_frames_range = np.arange(output_start, total_frames + 1)
    optimal_distances = []

    for fid in true_frames_range:
        progress = (fid - output_start) / max(1, (total_frames - output_start))
        progress = np.clip(progress, 0.0, 1.0)
        optimal_dist = d_at_throw * (1.0 - progress)
        optimal_distances.append(optimal_dist)

    # Plot optimal for players with color assignment
    player_color = player['color']
    ax.plot(true_frames_range, optimal_distances,
           color=player_color, linewidth=1.5, linestyle='--',
           alpha=0.6, label=f"{pname} (Optimal)")

ax.axvline(x=throw_frame, color='black', linestyle='--', linewidth=1.5, label='Throw Frame')
ax.set_xlabel('Frame ID', fontsize=11, fontweight='bold')
ax.set_ylabel('Distance to Ball Landing (yards)', fontsize=11, fontweight='bold')
ax.set_title(f'Game {game_id} | Play {play_id} - Distance to Ball Landing Over Time',
            fontsize=12, fontweight='bold')
# Set x-axis to start from frame 1
ax.set_xlim(min_frame - 1, max_frame + 1)
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)

# Add animated red cursor line
cursor_line = ax.axvline(x=min_frame, color='red', linestyle='-', linewidth=1.5, alpha=0.6, zorder=10)

# Animation function for cursor
def animate_cursor(frame_num):
    cursor_line.set_xdata([frame_num, frame_num])
    return [cursor_line]

# Create animation
FRAME_SKIP = 1
INTERVAL_MS = 100  # 100ms interval = 10 fps
anim_frames = frames_all[::FRAME_SKIP]

anim = FuncAnimation(fig, animate_cursor, frames=anim_frames, interval=INTERVAL_MS,
                    blit=True, repeat=True, repeat_delay=1000)

plt.tight_layout()
print("Graph complete!\n")

# Store figure and animation for export
distance_animation_fig = fig
distance_animation_cursor = animate_cursor
distance_animation_frames = anim_frames

anim


FPS= 10.0
DT= 1.0 / FPS

# # ========= Determine TRUE window (ball-in-air) =========

fmin = 1
fmax = total_frames
true_start = output_start
true_frames = output_frames

# ========= Deviation computation (all TRUE players) =========
rows = []
dev_series = {}   # for plotting: {player_name: {frame,long,lat}}

for (pid, name), g in play.groupby(['nfl_id','player_name']):
    g = g[g['frame_id'].isin(frames_all[output_start-1:total_frames])]
    if g.empty:
        continue

    # Starting position at the first TRUE frame this player appears in
    f0 = output_start
    if (g['frame_id'] == f0).any():
        xs, ys = float(g.loc[g['frame_id']==f0, 'x'].iloc[0]), float(g.loc[g['frame_id']==f0, 'y'].iloc[0])
    else:
        # if player enters later, start from their first TRUE frame
        xs, ys = float(g['x'].iloc[0]), float(g['y'].iloc[0])
        f0 = int(g['frame_id'].iloc[0])

    # Path to ball
    dir_vec = np.array([bx - xs, by - ys], dtype=float)
    L = float(np.hypot(*dir_vec)) + 1e-9
    u = dir_vec / L  # unit direction

    frames_p = g['frame_id'].to_numpy().astype(int)
    K = len(frames_p)

    if K == 1:
        # Single TRUE frame: zero dev by definition
        long_dev = np.array([0.0])
        lat_dev  = np.array([0.0])
        dev_series[name] = {'frame': frames_p, 'long': long_dev, 'lat': lat_dev}
        final_long = float(long_dev[-1]); final_lat = float(lat_dev[-1])
        final_total = float(np.hypot(final_long, final_lat))
        rows.append({
            'player_name': name,
            'MAE_long(yd)': 0.0, 'RMSE_long(yd)': 0.0,
            'MAE_lat(yd)': 0.0,  'RMSE_lat(yd)': 0.0,
            'Max|lat|(yd)': 0.0, 'Area|lat|(yd*sec)': 0.0,
            'Final_long_dev(yd)': final_long,
            'Final_lat_dev(yd)': final_lat,
            'Final_total_dev(yd)': final_total
        })
        continue

    # Map each TRUE frame to a [0..1] progress so the optimal reaches the ball at true_end
    alphas = (frames_p - true_start) / max(1, (true_end - true_start))  # linear arrival
    alphas = np.clip(alphas, 0.0, 1.0)

    # Actual displacement from start
    xp = g['x'].to_numpy(); yp = g['y'].to_numpy()
    r  = np.stack([xp - xs, yp - ys], axis=1)      # (K,2)

    # Along-path and lateral deviations
    along    = r @ u                                # signed distance along the path
    alongopt = alphas * L                           # optimal along distance
    long_dev = along - alongopt                     # +ahead / -behind

    # Signed lateral = 2D cross(u, r)
    lat_signed = u[0]*r[:,1] - u[1]*r[:,0]
    lat_dev    = lat_signed

    dev_series[name] = {'frame': frames_p, 'long': long_dev, 'lat': lat_dev}

    # Summaries (including "final value" at the last TRUE frame for this player)
    mae_long  = float(np.mean(np.abs(long_dev)))
    rmse_long = float(np.sqrt(np.mean(long_dev**2)))
    mae_lat   = float(np.mean(np.abs(lat_dev)))
    rmse_lat  = float(np.sqrt(np.mean(lat_dev**2)))
    max_abs_lat = float(np.max(np.abs(lat_dev)))
    area_abs_lat= float(np.sum(np.abs(lat_dev)) * DT)

    final_long = float(long_dev[-1])
    final_lat  = float(lat_dev[-1])
    final_total= float(np.hypot(final_long, final_lat))

    rows.append({
        'player_name': name,
        'MAE_long(yd)': mae_long, 'RMSE_long(yd)': rmse_long,
        'MAE_lat(yd)': mae_lat,   'RMSE_lat(yd)': rmse_lat,
        'Max|lat|(yd)': max_abs_lat, 'Area|lat|(yd*sec)': area_abs_lat,
        'Final_long_dev(yd)': final_long,
        'Final_lat_dev(yd)': final_lat,
        'Final_total_dev(yd)': final_total
    })

# ========= Results table (by player_name) =========
metrics = pd.DataFrame(rows).sort_values('Final_total_dev(yd)').reset_index(drop=True)
print("\nPer-player deviation summary (TRUE window):")

# Optionally save
# metrics.to_csv(f'deviation_metrics_{game_id}_{play_id}.csv', index=False)

# ========= Plots (names in legend) =========
# Longitudinal
plt.figure(figsize=(12,5))
for name, ser in dev_series.items():
    t = (ser['frame'] - true_start) * DT
    plt.plot(t, ser['long'], lw=1.8, label=name, color = color_map[name])
plt.axhline(0, ls='--', c='gray', lw=1)
plt.title(f'Game {game_id} Play {play_id} â€” Longitudinal deviation (ahead/behind optimal)')
plt.xlabel('Time since throw (s)'); plt.ylabel('Along-path deviation (yd)')
plt.grid(True, ls='--', alpha=0.5)
plt.legend(title='player_name', ncol=2, fontsize=8)
plt.tight_layout(); plt.show()

# Lateral
plt.figure(figsize=(12,5))
for name, ser in dev_series.items():
    t = (ser['frame'] - true_start) * DT
    plt.plot(t, ser['lat'], lw=1.8, label=name, color = color_map[name])
plt.axhline(0, ls='--', c='gray', lw=1)
plt.title(f'Game {game_id} Play {play_id} â€” Lateral deviation (Â± perpendicular to optimal)')
plt.xlabel('Time since throw (s)'); plt.ylabel('Lateral deviation (yd)  (+left / -right)')
plt.grid(True, ls='--', alpha=0.5)
plt.legend(title='player_name', ncol=2, fontsize=8)
plt.tight_layout(); plt.show()


metrics


game_id, play_id


#Enter C- Completed, I- Incomplete, IN - Intercepted
# pass_result_filter = input("Enter pass result type (C/I/IN): ").strip().upper()

# Filter plays by pass result and check availability
# available_plays = merged[merged["pass_result"] == pass_result_filter]

# # Get unique game_id and play_id combinations
# unique_plays = available_plays.groupby(["game_id", "play_id"]).size().reset_index(name="count")

# # User input for game_id
# print(f"Available game_ids: {unique_plays['game_id'].astype(int).unique()[:10]}...\n")

# print("Hit \'Enter' if you want to continue with the previously loaded game_id")
# game_id_input = input("Enter game_id or 'random': ").strip()

# if game_id_input.lower() == "random":
#     game_id = int(unique_plays["game_id"].sample(n=1).iloc[0])
#     print(f"âœ“ Randomly selected game_id: {game_id}")

# elif game_id_input.strip() == "":
#     # If user enters nothing, keep the existing game_id
#     print("Continuing with previous game_id:", game_id)

# else:
#     # If the user typed a valid number, convert it
#     game_id = int(game_id_input)
#     print("âœ“ Using game_id:", game_id)


# #Filter to plays in the selected game
# plays_in_game = unique_plays[unique_plays["game_id"] == game_id].copy()

# print(f"  ({len(plays_in_game)} plays available in Game {game_id} with result '{pass_result_filter}')\n")
# print(f"  Available play_ids in Game {game_id}: {plays_in_game['play_id'].unique()[:20]}...")

# print("Hit \'Enter' if you want to continue with the previously loaded play_id")
# play_id_input = input("Enter play_id or 'random': ").strip()

# if play_id_input.lower() == "random":
#     play_id = int(plays_in_game["play_id"].sample(n=1).iloc[0])
#     print(f"âœ“ Randomly selected play_id: {play_id}")

# elif play_id_input.strip() == "":
#     # If user enters nothing, keep the existing game_id
#     print("Continuing with previous play_id:", play_id)
# else:
#     # If the user typed a valid number, convert it
#     play_id = int(play_id_input)
#     print("âœ“ Using play_id:", play_id)

# print(f"\n{'='*100}")
# print(f"SELECTED PLAY: Game {game_id}, Play {play_id}, Pass Result: {pass_result_filter}")
# print(f"{'='*100}\n")

# Subset that play - use the selected values
df_play = merged[(merged[['game_id', 'play_id']]==[SELECTED_GAME_ID, SELECTED_PLAY_ID]).all(axis=1)].reset_index(drop=True).copy()

# Verify we got the right play
if len(df_play) == 0:
    raise ValueError(f"No data found for Game {SELECTED_GAME_ID}, Play {SELECTED_PLAY_ID}")

# Set game_id and play_id from the actual data (in case of any mismatch)
game_id = int(df_play['game_id'].iloc[0])
play_id = int(df_play['play_id'].iloc[0])

print(f"âœ“ Loaded play data: Game {game_id}, Play {play_id}")
df_play = df_play.sort_values(["nfl_id", "frame_id"])

# Get pass result, targeted receiver, and route
pass_result = df_play["pass_result"].iloc[0]
targeted_route = df_play["route_of_targeted_receiver"].iloc[0] if "route_of_targeted_receiver" in df_play.columns else "Unknown"

# Identify targeted receiver by player_role column
targeted_receiver_data = df_play[df_play["player_role"] == "Targeted Receiver"]
if not targeted_receiver_data.empty:
    targeted_receiver = targeted_receiver_data["player_name"].iloc[0]
else:
    targeted_receiver = None

#Calculating frame details for the corresponding play
total_frames = max(df_play['frame_id'])
output_frames = int(df_play['num_frames_output'].iloc[0])
output_start = total_frames - output_frames + 1

print(f"The loaded play consists of {total_frames} frames in total. It has {output_frames} frames in the output which begins at frame {output_start}")

# Compute throw frame
num_frames_output = df_play["num_frames_output"].iloc[0]
max_frame = df_play["frame_id"].max()
min_frame = df_play["frame_id"].min()
throw_frame = max_frame - num_frames_output + 1

# TRUE window (ball-in-air)
true_end = max_frame
true_start = max_frame - num_frames_output + 1
true_frames = np.arange(true_start, true_end + 1)

# Ball landing (static target)
bx = float(df_play["ball_land_x"].dropna().iloc[0])
by = float(df_play["ball_land_y"].dropna().iloc[0])

# Set LP_x and LP_y for animations
LP_x = bx
LP_y = by

print(f"Ball landing point: ({LP_x:.1f}, {LP_y:.1f})")

# Keep ALL frames for distance plotting
df_play_all = df_play.copy()

# Keep TRUE frames only for PEI calculation
df_play_true = df_play[df_play["frame_id"].isin(true_frames)].copy()

print(f"Game {game_id}, Play {play_id}, Throw Frame: {throw_frame}")
print(f"Pass Result: {pass_result}")
print(f"Targeted Receiver: {targeted_receiver} - Route: {targeted_route}")

# Track adjustments for summary
adjustments_made = []

# For interceptions, find the defender closest to ball landing point at final frame
intercepting_player = None
if pass_result == "IN":
    final_frame_data = df_play_true[df_play_true["frame_id"] == true_end]
    defenders = final_frame_data[final_frame_data["player_side"] == "Defense"]
    if not defenders.empty:
        defenders_with_dist = defenders.copy()
        defenders_with_dist["dist_to_ball"] = np.hypot(
            defenders_with_dist["x"] - bx,
            defenders_with_dist["y"] - by
        )
        intercepting_player = defenders_with_dist.loc[defenders_with_dist["dist_to_ball"].idxmin(), "player_name"]
        print(f"Intercepting Player: {intercepting_player}")

# Calculate PEI for each player and store distance data for ALL frames
pei_results = []
distance_data = {}

# First, get distance data for ALL frames
for player_id, group in df_play_all.groupby("nfl_id"):
    player_name = group["player_name"].iloc[0]
    g = group.sort_values("frame_id").copy()
    frames_all = g["frame_id"].to_numpy().astype(int)

    # Store distance data for plotting (ALL frames)
    actual_distance_all = np.hypot(g["x"].to_numpy() - bx, g["y"].to_numpy() - by)

    # Calculate optimal distance starting from throw frame
    throw_frame_mask = frames_all >= throw_frame
    if np.any(throw_frame_mask):
        throw_idx = np.where(throw_frame_mask)[0][0]
        x_at_throw = g.iloc[throw_idx]["x"]
        y_at_throw = g.iloc[throw_idx]["y"]

        # Distance at throw frame
        d_at_throw = np.hypot(x_at_throw - bx, y_at_throw - by)

        # Calculate optimal linear progression from throw frame to ball landing
        frames_from_throw = frames_all[throw_frame_mask]
        alphas_from_throw = (frames_from_throw - throw_frame) / max(1, (true_end - throw_frame))
        alphas_from_throw = np.clip(alphas_from_throw, 0.0, 1.0)

        # Optimal distance: linear decrease from d_at_throw to 0
        optimal_distance_from_throw = d_at_throw * (1.0 - alphas_from_throw)

        # Full optimal array (NaN before throw, then optimal progression)
        optimal_distance_all = np.full_like(actual_distance_all, np.nan, dtype=float)
        optimal_distance_all[throw_frame_mask] = optimal_distance_from_throw
    else:
        optimal_distance_all = np.full_like(actual_distance_all, np.nan, dtype=float)

    distance_data[player_name] = {
        "frames": frames_all,
        "actual_distance": actual_distance_all,
        "optimal_distance": optimal_distance_all
    }

# Now calculate PEI for TRUE frames only
for player_id, group in df_play_true.groupby("nfl_id"):
    player_name = group["player_name"].iloc[0]
    player_role = group["player_role"].iloc[0] if "player_role" in group.columns else None
    player_side = group["player_side"].iloc[0] if "player_side" in group.columns else None
    g = group.sort_values("frame_id").copy()
    frames = g["frame_id"].to_numpy().astype(int)
    K = len(frames)

    if K < 2:
        pei_results.append({
            "player_name": player_name,
            "PEI_simple": 1.0,
            "dir_MAE_norm": 0.0,
            "path_MAE_norm": 0.0,
            "rad_excess_mean_norm": 0.0,
            "L_path_to_ball": 0.0
        })
        continue

    # Start at this player's first TRUE frame (throw frame)
    xs, ys = float(g.iloc[0]["x"]), float(g.iloc[0]["y"])

    # Vector to ball from throw position + length and unit
    v = np.array([bx - xs, by - ys], dtype=float)
    L = float(np.hypot(*v)) + 1e-9
    u = v / L

    # Linear progress schedule alpha(t) from 0â†’1 across TRUE frames
    alphas = (frames - true_start) / max(1, (true_end - true_start))
    alphas = np.clip(alphas, 0.0, 1.0)

    # Actual displacement from start
    xp = g["x"].to_numpy()
    yp = g["y"].to_numpy()
    r = np.stack([xp - xs, yp - ys], axis=1)

    # FIX: Path-based components - these were commented out in original code
    along = r @ u  # Projection along optimal path
    along_opt = alphas * L  # Expected progress along optimal path
    long_dev = along - along_opt  # Longitudinal deviation (timing)
    lat_dev = u[0] * r[:, 1] - u[1] * r[:, 0]  # Lateral deviation (direction)

    # Distance-to-ball component
    d_actual = np.hypot(xp - bx, yp - by)
    d_opt = (1.0 - alphas) * L

    # Use minimum distance achieved for hitch route handling
    min_distance_achieved = np.min(d_actual)
    rad_excess_min = np.maximum(0.0, d_actual - min_distance_achieved)

    # Normalize by L and average across frames (bounded 0..1)
    dir_MAE_norm = float(np.clip(np.mean(np.abs(lat_dev)) / L, 0.0, 1.0))
    path_MAE_norm = float(np.clip(np.mean(np.abs(long_dev)) / L, 0.0, 1.0))
    rad_mean_norm = float(np.clip(np.mean(rad_excess_min) / L, 0.0, 1.0))

    # Store original values for adjustments tracking
    original_dir = dir_MAE_norm
    original_path = path_MAE_norm
    original_rad = rad_mean_norm

    # Check if this defender should be awarded for tight coverage
    is_close_defender = False
    if player_side == "Defense" and pass_result == "I":
        last_10_frames = sorted(frames)[-10:] if len(frames) >= 10 else frames
        targeted_receiver_data_check = df_play_true[
            (df_play_true["player_role"] == "Targeted Receiver") &
            (df_play_true["frame_id"].isin(last_10_frames))
        ]

        if not targeted_receiver_data_check.empty:
            defender_last_frames = g[g["frame_id"].isin(last_10_frames)]

            for _, def_frame in defender_last_frames.iterrows():
                tgt_frame = targeted_receiver_data_check[
                    targeted_receiver_data_check["frame_id"] == def_frame["frame_id"]
                ]
                if not tgt_frame.empty:
                    dist = np.hypot(
                        def_frame["x"] - tgt_frame.iloc[0]["x"],
                        def_frame["y"] - tgt_frame.iloc[0]["y"]
                    )
                    if dist <= 2.0:
                        is_close_defender = True
                        break

    # Outcome-adjusted components BEFORE PEI calculation
    if pass_result == "C" and player_role == "Targeted Receiver":
        path_MAE_norm = 0.0
        dir_MAE_norm = dir_MAE_norm * 0.3
        rad_mean_norm = rad_mean_norm * 0.3
        adjustments_made.append({
            "player_name": player_name,
            "role": "Targeted Receiver (Caught)",
            "adjustment_type": "Completion Bonus",
            "path_change": f"{original_path:.3f} â†’ 0.000 (zeroed)",
            "dir_change": f"{original_dir:.3f} â†’ {dir_MAE_norm:.3f} (70% reduction)",
            "rad_change": f"{original_rad:.3f} â†’ {rad_mean_norm:.3f} (70% reduction)"
        })

    elif pass_result == "IN" and player_name == intercepting_player:
        path_MAE_norm = 0.0
        dir_MAE_norm = dir_MAE_norm * 0.3
        rad_mean_norm = rad_mean_norm * 0.3
        adjustments_made.append({
            "player_name": player_name,
            "role": "Intercepting Defender",
            "adjustment_type": "Interception Bonus",
            "path_change": f"{original_path:.3f} â†’ 0.000 (zeroed)",
            "dir_change": f"{original_dir:.3f} â†’ {dir_MAE_norm:.3f} (70% reduction)",
            "rad_change": f"{original_rad:.3f} â†’ {rad_mean_norm:.3f} (70% reduction)"
        })

    elif is_close_defender:
        dir_MAE_norm = dir_MAE_norm * 0.5
        path_MAE_norm = path_MAE_norm * 0.5
        rad_mean_norm = rad_mean_norm * 0.5
        adjustments_made.append({
            "player_name": player_name,
            "role": "Tight Coverage Defender",
            "adjustment_type": "Tight Coverage Bonus (within 2 yards in last 10 frames)",
            "path_change": f"{original_path:.3f} â†’ {path_MAE_norm:.3f} (50% reduction)",
            "dir_change": f"{original_dir:.3f} â†’ {dir_MAE_norm:.3f} (50% reduction)",
            "rad_change": f"{original_rad:.3f} â†’ {rad_mean_norm:.3f} (50% reduction)"
        })

    # Simple hybrid: equal parts of the three
    PEI_simple = 1.0 - (dir_MAE_norm + path_MAE_norm + rad_mean_norm) / 3.0
    PEI_simple = float(np.clip(PEI_simple, 0.0, 1.0))

    pei_results.append({
        "player_name": player_name,
        "PEI_simple": PEI_simple,
        "dir_MAE_norm": dir_MAE_norm,
        "path_MAE_norm": path_MAE_norm,
        "rad_excess_mean_norm": rad_mean_norm,
        "L_path_to_ball": float(L)
    })

# Convert to DataFrame
pei_df = pd.DataFrame(pei_results).sort_values("PEI_simple", ascending=False).reset_index(drop=True)

all_players_in_play = df_play_all[['player_name', 'player_side', 'player_position']].drop_duplicates()
color_map = build_color_map(all_players_in_play, df_play_all, targeted_receiver)

# Building PEI Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))

colors = [get_player_color(name, color_map, has_pei=True) for name in pei_df["player_name"]]

ax.barh(pei_df["player_name"], pei_df["PEI_simple"], color=colors)
ax.invert_yaxis()
ax.set_xlabel("PEI_simple (0..1)", fontsize=12, fontweight="bold")
ax.set_title(f"PEI_simple â€” Game {game_id} | Play {play_id} | Pass Result: {pass_result} | Route: {targeted_route}",
             fontsize=14, fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Print PEI results
print("\n" + "="*100)
print(f"Simple Pursuit Efficiency Index (PEI) â€” 1 = perfect | Route: {targeted_route}")
print("="*100)
print(pei_df[["player_name", "PEI_simple", "dir_MAE_norm", "path_MAE_norm",
              "rad_excess_mean_norm", "L_path_to_ball"]].to_string(index=False))
print("="*100)

# Print adjustments summary
if adjustments_made:
    print("\n" + "="*100)
    print("PEI ADJUSTMENTS APPLIED:")
    print("="*100)
    for adj in adjustments_made:
        print(f"\nğŸ�¯ {adj['player_name']} - {adj['role']}")
        print(f"   Logic: {adj['adjustment_type']}")
        print(f"   Component Changes:")
        print(f"      â€¢ path_MAE_norm:       {adj['path_change']}")
        print(f"      â€¢ dir_MAE_norm:        {adj['dir_change']}")
        print(f"      â€¢ rad_excess_mean_norm: {adj['rad_change']}")

    print("\n" + "="*100)
    print("ADJUSTMENT LOGIC SUMMARY:")
    print("="*100)
    print("1. Completion Bonus (Targeted Receiver catches ball):")
    print("   - path_MAE_norm set to 0 (timing was perfect)")
    print("   - dir_MAE_norm and rad_excess_mean_norm reduced by 70%")
    print("\n2. Interception Bonus (Defender closest to ball at final frame on IN plays):")
    print("   - path_MAE_norm set to 0 (timing was perfect)")
    print("   - dir_MAE_norm and rad_excess_mean_norm reduced by 70%")
    print("\n3. Tight Coverage Bonus (Defender within 2 yards of targeted receiver in last 10 frames on I plays):")
    print("   - All three components reduced by 50%")
    print("="*100)
else:
    print("\n" + "="*100)
    print("No PEI adjustments applied (standard calculation for all players)")
    print("="*100)

# ========= SAVE SINGLE-PLAY DATA FOR EXPORT =========
# This prevents data from being overwritten by later cells

# Save the single-play PEI DataFrame
pei_df_single_play = pei_df.copy()

# Save the raw results list
pei_results_single_play = pei_results.copy()

# Save play metadata
single_play_metadata = {
    'game_id': game_id,
    'play_id': play_id,
    'pass_result': pass_result,
    'targeted_route': targeted_route,
    'targeted_receiver': targeted_receiver
}

print("\n" + "="*100)
print("SAVED SINGLE-PLAY DATA FOR EXPORT")
print("="*100)
print(f"âœ“ pei_df_single_play: {len(pei_df_single_play)} players")
print(f"âœ“ pei_results_single_play: {len(pei_results_single_play)} entries")
print(f"âœ“ Play: Game {game_id}, Play {play_id}, Result: {pass_result}, Route: {targeted_route}")
print("="*100 + "\n")

# ========= CREATE play_data FOR ANIMATIONS =========
# This ensures field animation and distance graph use the correct play
play_data = df_play_all.copy()

print("\n" + "="*100)
print("CREATED play_data FOR ANIMATIONS")
print("="*100)
print(f"âœ“ play_data: {len(play_data)} rows for Game {game_id}, Play {play_id}")
print(f"âœ“ Unique frames: {play_data['frame_id'].nunique()}")
print(f"âœ“ Unique players: {play_data['player_name'].nunique()}")
print("="*100 + "\n")


# ğŸ�¯ Filter only players_to_predict == True
df_pred = merged[merged["player_to_predict"] == True].copy()

# Pre-compute TRUE frames using vectorized operations
df_pred['max_frame'] = df_pred.groupby(['game_id', 'play_id'])['frame_id'].transform('max')
df_pred['min_frame'] = df_pred.groupby(['game_id', 'play_id'])['frame_id'].transform('min')
df_pred['throw_frame'] = df_pred['max_frame'] - df_pred['num_frames_output'] + 1
df_pred['true_start'] = df_pred[['min_frame', 'throw_frame']].max(axis=1)
df_pred['true_end'] = df_pred['max_frame']
df_pred['is_true_frame'] = (df_pred['frame_id'] >= df_pred['true_start']) & (df_pred['frame_id'] <= df_pred['true_end'])

# Filter to TRUE frames only
df_true = df_pred[df_pred['is_true_frame']].copy()

print(f"Processing {len(df_true)} TRUE frame rows...")

# Pre-identify intercepting defenders and tight coverage defenders per play
interception_bonuses = []
tight_coverage_bonuses = []

for (game_id, play_id), play_group in df_true.groupby(['game_id', 'play_id']):
    pass_result = play_group['pass_result'].iloc[0]
    true_end = play_group['true_end'].iloc[0]
    bx = float(play_group['ball_land_x'].iloc[0])
    by = float(play_group['ball_land_y'].iloc[0])

    # Interception bonus
    if pass_result == 'IN':
        final_frame = play_group[play_group['frame_id'] == true_end]
        defenders = final_frame[final_frame['player_side'] == 'Defense']
        if not defenders.empty:
            defenders = defenders.copy()
            defenders['dist_to_ball'] = np.hypot(defenders['x'] - bx, defenders['y'] - by)
            intercepting_nfl_id = defenders.loc[defenders['dist_to_ball'].idxmin(), 'nfl_id']
            interception_bonuses.append((game_id, play_id, intercepting_nfl_id))

    # Tight coverage bonus
    if pass_result == 'I':
        max_frame_in_play = play_group['frame_id'].max()
        last_10_start = max_frame_in_play - 9
        last_10_frames = play_group[play_group['frame_id'] >= last_10_start]

        targeted_receiver = last_10_frames[last_10_frames['player_role'] == 'Targeted Receiver']
        defenders = last_10_frames[last_10_frames['player_side'] == 'Defense']

        if not targeted_receiver.empty and not defenders.empty:
            for frame_id in last_10_frames['frame_id'].unique():
                tr_frame = targeted_receiver[targeted_receiver['frame_id'] == frame_id]
                def_frame = defenders[defenders['frame_id'] == frame_id]

                if not tr_frame.empty and not def_frame.empty:
                    tr_x, tr_y = tr_frame.iloc[0][['x', 'y']]
                    def_frame = def_frame.copy()
                    def_frame['dist_to_tr'] = np.hypot(def_frame['x'] - tr_x, def_frame['y'] - tr_y)
                    close_defenders = def_frame[def_frame['dist_to_tr'] <= 2.0]['nfl_id'].unique()

                    for nfl_id in close_defenders:
                        if (game_id, play_id, nfl_id) not in tight_coverage_bonuses:
                            tight_coverage_bonuses.append((game_id, play_id, nfl_id))

interception_set = set(interception_bonuses)
tight_coverage_set = set(tight_coverage_bonuses)

print(f"Found {len(interception_set)} interceptions and {len(tight_coverage_set)} tight coverage instances")

# Calculate PEI for each player in each play
all_pei_results = []

for (game_id, play_id, nfl_id), group in df_true.groupby(["game_id", "play_id", "nfl_id"]):
    player_name = group["player_name"].iloc[0]
    player_position = group["player_position"].iloc[0]
    player_role = group["player_role"].iloc[0] if "player_role" in group.columns else None
    player_side = group["player_side"].iloc[0] if "player_side" in group.columns else None
    pass_result = group["pass_result"].iloc[0]

    g = group.sort_values("frame_id")
    frames = g["frame_id"].to_numpy().astype(int)
    K = len(frames)

    if K < 2:
        all_pei_results.append({
            "nfl_id": nfl_id,
            "player_name": player_name,
            "player_position": player_position,
            "game_id": game_id,
            "play_id": play_id,
            "PEI_simple": 1.0,
            "dir_MAE_norm": 0.0,
            "path_MAE_norm": 0.0,
            "rad_excess_mean_norm": 0.0,
            "L_path_to_ball": 0.0,
            "frames_after_throw": group["num_frames_output"].iloc[0]
        })
        continue

    # Ball landing
    bx = float(g['ball_land_x'].iloc[0])
    by = float(g['ball_land_y'].iloc[0])

    # Start position
    xs, ys = float(g.iloc[0]["x"]), float(g.iloc[0]["y"])

    # Vector to ball
    v = np.array([bx - xs, by - ys], dtype=float)
    L = float(np.hypot(*v)) + 1e-9
    u = v / L

    # Progress schedule
    true_start = g['true_start'].iloc[0]
    true_end = g['true_end'].iloc[0]
    alphas = (frames - true_start) / max(1, (true_end - true_start))
    alphas = np.clip(alphas, 0.0, 1.0)

    # Actual positions
    xp = g["x"].to_numpy()
    yp = g["y"].to_numpy()
    r = np.stack([xp - xs, yp - ys], axis=1)

    # Components
    along = r @ u
    along_opt = alphas * L
    long_dev = along - along_opt
    lat_dev = u[0] * r[:, 1] - u[1] * r[:, 0]

    d_actual = np.hypot(xp - bx, yp - by)
    min_distance_achieved = np.min(d_actual)
    rad_excess_min = np.maximum(0.0, d_actual - min_distance_achieved)

    # Normalize
    dir_MAE_norm = float(np.clip(np.mean(np.abs(lat_dev)) / L, 0.0, 1.0))
    path_MAE_norm = float(np.clip(np.mean(np.abs(long_dev)) / L, 0.0, 1.0))
    rad_mean_norm = float(np.clip(np.mean(rad_excess_min) / L, 0.0, 1.0))

    # Apply bonuses
    lookup_key = (game_id, play_id, nfl_id)

    if pass_result == "C" and player_role == "Targeted Receiver":
        path_MAE_norm = 0.0
        dir_MAE_norm *= 0.3
        rad_mean_norm *= 0.3
    elif lookup_key in interception_set:
        path_MAE_norm = 0.0
        dir_MAE_norm *= 0.3
        rad_mean_norm *= 0.3
    elif lookup_key in tight_coverage_set:
        dir_MAE_norm *= 0.5
        path_MAE_norm *= 0.5
        rad_mean_norm *= 0.5

    # Calculate PEI
    PEI_simple = 1.0 - (dir_MAE_norm + path_MAE_norm + rad_mean_norm) / 3.0
    PEI_simple = float(np.clip(PEI_simple, 0.0, 1.0))

    all_pei_results.append({
        "nfl_id": nfl_id,
        "player_name": player_name,
        "player_position": player_position,
        "game_id": game_id,
        "play_id": play_id,
        "PEI_simple": PEI_simple,
        "dir_MAE_norm": dir_MAE_norm,
        "path_MAE_norm": path_MAE_norm,
        "rad_excess_mean_norm": rad_mean_norm,
        "L_path_to_ball": float(L),
        "frames_after_throw": group["num_frames_output"].iloc[0]
    })

# Convert to DataFrame
pei_df = pd.DataFrame(all_pei_results)
print(f"Calculated PEI for {len(pei_df)} player-play combinations")

# Group by player and calculate averages
player_summary = pei_df.groupby(["nfl_id", "player_name", "player_position"]).agg(
    avg_PEI=("PEI_simple", "mean"),
    avg_dir_MAE=("dir_MAE_norm", "mean"),
    avg_path_MAE=("path_MAE_norm", "mean"),
    avg_rad_excess=("rad_excess_mean_norm", "mean"),
    avg_path_length=("L_path_to_ball", "mean"),
    avg_frames_after_throw=("frames_after_throw", "mean"),
    total_plays_predicted=("PEI_simple", "count")
).reset_index()

# Sort by average PEI (descending = better performance)
player_summary = player_summary.sort_values("avg_PEI", ascending=False).reset_index(drop=True)

# Filter for players with more than 9 plays
player_summary_filtered = player_summary[player_summary["total_plays_predicted"] > 9].reset_index(drop=True)

# Display tables separated by position
print("=" * 100)
print("PLAYER PEI SUMMARY BY POSITION (player_to_predict == True)")
print("=" * 100)

if len(player_summary_filtered) == 0:
    print("\nâš ï¸�  No players found with more than 9 plays.")
    print(f"Total players before filter: {len(player_summary)}")
    print(f"\nPlay count distribution:")
    print(player_summary['total_plays_predicted'].describe())
else:
    positions = sorted(player_summary_filtered["player_position"].unique())

    for position in positions:
        position_df = player_summary_filtered[player_summary_filtered["player_position"] == position].reset_index(drop=True)

        print(f"\n{'='*100}")
        print(f"POSITION: {position}")
        print(f"{'='*100}")
        print(position_df.to_string(index=False))
        print(f"\nPlayers in {position}: {len(position_df)}")
        print(f"Total plays for {position}: {position_df['total_plays_predicted'].sum()}")
        print(f"Avg PEI for {position}: {position_df['avg_PEI'].mean():.4f}")

    print("\n" + "=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)
    print(f"Total unique players: {len(player_summary_filtered)}")
    print(f"Total unique plays analyzed: {pei_df[['game_id', 'play_id']].drop_duplicates().shape[0]}")
    print(f"Total player-play combinations: {len(pei_df)}")
    print(f"Overall average PEI: {player_summary_filtered['avg_PEI'].mean():.4f}")
    print("=" * 100)


all_pei_results = []

for (game_id, play_id, nfl_id), group in df_true.groupby(["game_id", "play_id", "nfl_id"]):
    player_name = group["player_name"].iloc[0]
    player_side = group["player_side"].iloc[0]
    player_role = group["player_role"].iloc[0] if "player_role" in group.columns else None
    route = group["route_of_targeted_receiver"].iloc[0]
    pass_length = group["pass_length"].iloc[0]
    pass_result = group["pass_result"].iloc[0]

    g = group.sort_values("frame_id")
    frames = g["frame_id"].to_numpy().astype(int)
    K = len(frames)

    if K < 2:
        all_pei_results.append({
            "game_id": game_id,
            "play_id": play_id,
            "nfl_id": nfl_id,
            "player_name": player_name,
            "player_side": player_side,
            "route_of_targeted_receiver": route,
            "pass_length": pass_length,
            "pass_result": pass_result,
            "PEI_simple": 1.0
        })
        continue

    # Ball landing
    bx = float(g['ball_land_x'].iloc[0])
    by = float(g['ball_land_y'].iloc[0])

    # Start position
    xs, ys = float(g.iloc[0]["x"]), float(g.iloc[0]["y"])

    # Vector to ball
    v = np.array([bx - xs, by - ys], dtype=float)
    L = float(np.hypot(*v)) + 1e-9
    u = v / L

    # Progress schedule
    true_start = g['true_start'].iloc[0]
    true_end = g['true_end'].iloc[0]
    alphas = (frames - true_start) / max(1, (true_end - true_start))
    alphas = np.clip(alphas, 0.0, 1.0)

    # Actual positions
    xp = g["x"].to_numpy()
    yp = g["y"].to_numpy()
    r = np.stack([xp - xs, yp - ys], axis=1)

    # Components
    along = r @ u
    along_opt = alphas * L
    long_dev = along - along_opt
    lat_dev = u[0] * r[:, 1] - u[1] * r[:, 0]

    d_actual = np.hypot(xp - bx, yp - by)
    min_distance_achieved = np.min(d_actual)
    rad_excess_min = np.maximum(0.0, d_actual - min_distance_achieved)

    # Normalize
    dir_MAE_norm = float(np.clip(np.mean(np.abs(lat_dev)) / L, 0.0, 1.0))
    path_MAE_norm = float(np.clip(np.mean(np.abs(long_dev)) / L, 0.0, 1.0))
    rad_mean_norm = float(np.clip(np.mean(rad_excess_min) / L, 0.0, 1.0))

    # Apply bonuses
    lookup_key = (game_id, play_id, nfl_id)

    if pass_result == "C" and player_role == "Targeted Receiver":
        path_MAE_norm = 0.0
        dir_MAE_norm *= 0.3
        rad_mean_norm *= 0.3
    elif lookup_key in interception_set:
        path_MAE_norm = 0.0
        dir_MAE_norm *= 0.3
        rad_mean_norm *= 0.3
    elif lookup_key in tight_coverage_set:
        dir_MAE_norm *= 0.5
        path_MAE_norm *= 0.5
        rad_mean_norm *= 0.5

    # Calculate PEI
    PEI_simple = 1.0 - (dir_MAE_norm + path_MAE_norm + rad_mean_norm) / 3.0
    PEI_simple = float(np.clip(PEI_simple, 0.0, 1.0))

    all_pei_results.append({
        "game_id": game_id,
        "play_id": play_id,
        "nfl_id": nfl_id,
        "player_name": player_name,
        "player_side": player_side,
        "route_of_targeted_receiver": route,
        "pass_length": pass_length,
        "pass_result": pass_result,
        "PEI_simple": PEI_simple
    })

# Convert to DataFrame
pei_df = pd.DataFrame(all_pei_results)

# Count players tracked per play
players_per_play = pei_df.groupby(["game_id", "play_id"]).size().reset_index(name="players_tracked")
pei_df = pei_df.merge(players_per_play, on=["game_id", "play_id"])

# Separate offense and defense
offense_df = pei_df[pei_df["player_side"] == "Offense"].copy()
defense_df = pei_df[pei_df["player_side"] == "Defense"].copy()

# Aggregate by route
offense_summary = offense_df.groupby("route_of_targeted_receiver").agg(
    avg_PEI_offense=("PEI_simple", "mean"),
    num_plays_offense=("PEI_simple", "count")
).reset_index()

defense_summary = defense_df.groupby("route_of_targeted_receiver").agg(
    avg_PEI_defense=("PEI_simple", "mean"),
    num_plays_defense=("PEI_simple", "count")
).reset_index()

play_counts = pei_df.groupby(["route_of_targeted_receiver", "game_id", "play_id"])["players_tracked"].first().reset_index()
players_tracked_summary = play_counts.groupby("route_of_targeted_receiver").agg(
    avg_players_tracked_minus_1=("players_tracked", lambda x: x.mean() - 1)
).reset_index()

pass_length_summary = pei_df.groupby("route_of_targeted_receiver").agg(
    avg_pass_length=("pass_length", "mean")
).reset_index()

# Count interceptions per route
interception_summary = pei_df[pei_df["pass_result"] == "IN"].groupby("route_of_targeted_receiver").agg(
    num_interceptions=("pass_result", "count")
).reset_index()

# Merge summaries
route_summary = offense_summary.merge(defense_summary, on="route_of_targeted_receiver", how="outer")
route_summary = route_summary.merge(players_tracked_summary, on="route_of_targeted_receiver", how="outer")
route_summary = route_summary.merge(pass_length_summary, on="route_of_targeted_receiver", how="outer")
route_summary = route_summary.merge(interception_summary, on="route_of_targeted_receiver", how="outer")
route_summary = route_summary.fillna(0)
route_summary["num_plays_offense"] = route_summary["num_plays_offense"].astype(int)
route_summary["num_plays_defense"] = route_summary["num_plays_defense"].astype(int)
route_summary["num_interceptions"] = route_summary["num_interceptions"].astype(int)
route_summary = route_summary.sort_values("route_of_targeted_receiver").reset_index(drop=True)

# Display
print("=" * 100)
print("ROUTE PEI SUMMARY (Offense vs Defense)")
print("=" * 100)
print(route_summary.to_string(index=False))
print("=" * 100)
print(f"\nTotal unique routes: {len(route_summary)}")
print(f"Total unique plays analyzed: {pei_df[['game_id', 'play_id']].drop_duplicates().shape[0]}")
print(f"Total interceptions: {interception_summary['num_interceptions'].sum()}")
print("\nNote: Higher PEI = Better pursuit efficiency (0 to 1 scale)")
print("\nBonus Logic Applied:")
print("- Completion: Targeted receiver gets path_MAE_norm=0.0, dir/rad multiplied by 0.3")
print("- Interception: Defender gets path_MAE_norm=0.0, dir/rad multiplied by 0.3")
print("- Tight Coverage (Incomplete): Defenders within 2.0 yards get all components multiplied by 0.5")
print("=" * 100)


# Pre-identify completion bonuses and defensive bonuses
completion_bonuses = set()
interception_bonuses = set()
tight_coverage_bonuses = set()

for (game_id, play_id), play_group in df_true.groupby(['game_id', 'play_id']):
    pass_result = play_group['pass_result'].iloc[0]
    true_end = play_group['true_end'].iloc[0]
    bx = float(play_group['ball_land_x'].iloc[0])
    by = float(play_group['ball_land_y'].iloc[0])

    # Completion bonus - targeted receiver who caught
    if pass_result == 'C':
        targeted = play_group[play_group['player_role'] == 'Targeted Receiver']
        if not targeted.empty:
            nfl_id = targeted['nfl_id'].iloc[0]
            completion_bonuses.add((game_id, play_id, nfl_id))

    # Interception bonus - defender closest to ball at play end
    if pass_result == 'IN':
        final_frame = play_group[play_group['frame_id'] == true_end]
        defenders = final_frame[final_frame['player_side'] == 'Defense']
        if not defenders.empty:
            defenders = defenders.copy()
            defenders['dist_to_ball'] = np.hypot(defenders['x'] - bx, defenders['y'] - by)
            intercepting_nfl_id = defenders.loc[defenders['dist_to_ball'].idxmin(), 'nfl_id']
            interception_bonuses.add((game_id, play_id, intercepting_nfl_id))

    # Tight coverage bonus - defenders within 2.0 yards of targeted receiver in last 10 frames
    if pass_result == 'I':
        max_frame_in_play = play_group['frame_id'].max()
        last_10_start = max_frame_in_play - 9
        last_10_frames = play_group[play_group['frame_id'] >= last_10_start]

        targeted_receiver = last_10_frames[last_10_frames['player_role'] == 'Targeted Receiver']
        defenders = last_10_frames[last_10_frames['player_side'] == 'Defense']

        if not targeted_receiver.empty and not defenders.empty:
            for frame_id in last_10_frames['frame_id'].unique():
                tr_frame = targeted_receiver[targeted_receiver['frame_id'] == frame_id]
                def_frame = defenders[defenders['frame_id'] == frame_id]

                if not tr_frame.empty and not def_frame.empty:
                    tr_x, tr_y = tr_frame.iloc[0][['x', 'y']]
                    def_frame = def_frame.copy()
                    def_frame['dist_to_tr'] = np.hypot(def_frame['x'] - tr_x, def_frame['y'] - tr_y)
                    close_defenders = def_frame[def_frame['dist_to_tr'] <= 2.0]['nfl_id'].unique()

                    for nfl_id in close_defenders:
                        if (game_id, play_id, nfl_id) not in tight_coverage_bonuses:
                            tight_coverage_bonuses.add((game_id, play_id, nfl_id))

# Calculate PEI for each player in each play
all_pei_results = []

for (game_id, play_id, nfl_id), group in df_true.groupby(["game_id", "play_id", "nfl_id"]):
    player_side = group["player_side"].iloc[0]
    player_role = group["player_role"].iloc[0] if "player_role" in group.columns else None
    route = group["route_of_targeted_receiver"].iloc[0]
    pass_length = group["pass_length"].iloc[0]
    pass_result = group["pass_result"].iloc[0]

    g = group.sort_values("frame_id")
    frames = g["frame_id"].to_numpy().astype(int)
    K = len(frames)

    if K < 2:
        all_pei_results.append({
            "game_id": game_id,
            "play_id": play_id,
            "nfl_id": nfl_id,
            "player_side": player_side,
            "route_of_targeted_receiver": route,
            "pass_length": pass_length,
            "pass_result": pass_result,
            "PEI_simple": 1.0
        })
        continue

    # Ball landing
    bx = float(g['ball_land_x'].iloc[0])
    by = float(g['ball_land_y'].iloc[0])

    # Start position
    xs, ys = float(g.iloc[0]["x"]), float(g.iloc[0]["y"])

    # Vector to ball
    v = np.array([bx - xs, by - ys], dtype=float)
    L = float(np.hypot(*v)) + 1e-9
    u = v / L

    # Progress schedule
    true_start = g['true_start'].iloc[0]
    true_end = g['true_end'].iloc[0]
    alphas = (frames - true_start) / max(1, (true_end - true_start))
    alphas = np.clip(alphas, 0.0, 1.0)

    # Actual positions
    xp = g["x"].to_numpy()
    yp = g["y"].to_numpy()
    r = np.stack([xp - xs, yp - ys], axis=1)

    # Components
    along = r @ u
    along_opt = alphas * L
    long_dev = along - along_opt
    lat_dev = u[0] * r[:, 1] - u[1] * r[:, 0]

    d_actual = np.hypot(xp - bx, yp - by)
    min_distance_achieved = np.min(d_actual)
    rad_excess_min = np.maximum(0.0, d_actual - min_distance_achieved)

    # Normalize
    dir_MAE_norm = float(np.clip(np.mean(np.abs(lat_dev)) / L, 0.0, 1.0))
    path_MAE_norm = float(np.clip(np.mean(np.abs(long_dev)) / L, 0.0, 1.0))
    rad_mean_norm = float(np.clip(np.mean(rad_excess_min) / L, 0.0, 1.0))

    # Apply bonuses
    lookup_key = (game_id, play_id, nfl_id)

    # Completion bonus: Targeted receiver on completed pass
    if lookup_key in completion_bonuses:
        path_MAE_norm = 0.0
        dir_MAE_norm *= 0.3
        rad_mean_norm *= 0.3
    # Interception bonus: Defender who intercepted the ball
    elif lookup_key in interception_bonuses:
        path_MAE_norm = 0.0
        dir_MAE_norm *= 0.3
        rad_mean_norm *= 0.3
    # Tight coverage bonus: Defender in coverage on incomplete pass
    elif lookup_key in tight_coverage_bonuses:
        dir_MAE_norm *= 0.5
        path_MAE_norm *= 0.5
        rad_mean_norm *= 0.5

    # Calculate PEI
    PEI_simple = 1.0 - (dir_MAE_norm + path_MAE_norm + rad_mean_norm) / 3.0
    PEI_simple = float(np.clip(PEI_simple, 0.0, 1.0))

    all_pei_results.append({
        "game_id": game_id,
        "play_id": play_id,
        "nfl_id": nfl_id,
        "player_side": player_side,
        "route_of_targeted_receiver": route,
        "pass_length": pass_length,
        "pass_result": pass_result,
        "PEI_simple": PEI_simple
    })

# Convert to DataFrame
pei_df = pd.DataFrame(all_pei_results)

# Filter for offense only
offense_df = pei_df[pei_df["player_side"] == "Offense"].copy()

# Group by pass completion status
completion_summary = offense_df.groupby("pass_result").agg(
    avg_PEI_offense=("PEI_simple", "mean"),
    median_PEI_offense=("PEI_simple", "median"),
    num_plays=("PEI_simple", "count"),
    std_PEI=("PEI_simple", "std")
).reset_index()

# Also calculate overall stats
overall_stats = pd.DataFrame([{
    "pass_result": "ALL PASSES",
    "avg_PEI_offense": offense_df["PEI_simple"].mean(),
    "median_PEI_offense": offense_df["PEI_simple"].median(),
    "num_plays": len(offense_df),
    "std_PEI": offense_df["PEI_simple"].std()
}])

# Combine
completion_summary = pd.concat([completion_summary, overall_stats], ignore_index=True)

# Display the results
print("=" * 100)
print("OFFENSIVE PEI BY PASS COMPLETION STATUS")
print("=" * 100)
print(completion_summary.to_string(index=False))
print("=" * 100)
print("\nNote: Higher PEI = Better pursuit efficiency (0 to 1 scale)")
print("\nBonus Logic Applied:")
print("- Completion: Targeted receiver gets path_MAE_norm=0.0, dir/rad multiplied by 0.3")
print("- Interception: Defender gets path_MAE_norm=0.0, dir/rad multiplied by 0.3")
print("- Tight Coverage (Incomplete): Defenders within 2.0 yards get all components multiplied by 0.5")
print("=" * 100)

# Additional breakdown by route for completed vs incomplete
print("\n" + "=" * 100)
print("OFFENSIVE PEI BY ROUTE AND COMPLETION STATUS")
print("=" * 100)

route_completion_summary = offense_df.groupby(["route_of_targeted_receiver", "pass_result"]).agg(
    avg_PEI=("PEI_simple", "mean"),
    num_plays=("PEI_simple", "count")
).reset_index()

# Pivot to show completed vs incomplete side by side
route_pivot = route_completion_summary.pivot_table(
    index="route_of_targeted_receiver",
    columns="pass_result",
    values=["avg_PEI", "num_plays"],
    fill_value=0
).reset_index()

# Flatten column names
route_pivot.columns = ['_'.join(str(col).strip() for col in cols if col) if isinstance(cols, tuple) else cols
                       for cols in route_pivot.columns.values]
route_pivot.columns = [col.replace('route_of_targeted_receiver_', 'route') for col in route_pivot.columns]

print(route_pivot.to_string(index=False))
print("=" * 100)


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
from PIL import Image
import io
import os
import numpy as np
import pandas as pd

# ========= CONFIG =========
output_dir = "/kaggle/working"
os.makedirs(output_dir, exist_ok=True)

print("="*100)
print("EXPORTING ALL VISUALIZATIONS TO KAGGLE")
print("="*100 + "\n")

files_exported = []

# ========= USE SAVED METADATA FOR CORRECT FILENAMES =========
if 'single_play_metadata' in dir():
    game_id = single_play_metadata['game_id']
    play_id = single_play_metadata['play_id']
    pass_result = single_play_metadata['pass_result']
    targeted_route = single_play_metadata['targeted_route']
    print(f"âœ“ Export using: Game {game_id}, Play {play_id}, Result: {pass_result}")
else:
    raise NameError("single_play_metadata not found! Run play selection cell (Document 31) first.")

print()    
# ========= GET BALL LANDING POINT (use LP_x, LP_y directly) =========
ball_land_x = LP_x
ball_land_y = LP_y
print(f"Ball landing point: ({ball_land_x:.1f}, {ball_land_y:.1f})\n")

# ========= STEP 1: EXPORT FIELD ANIMATION TO GIF =========
print("STEP 1: Exporting Field Animation to GIF")
print("-"*100 + "\n")

# DIAGNOSTIC: Check ball_trajectory availability
print("Checking ball trajectory availability:")
if 'ball_trajectory_export' in dir():
    print(f"  âœ“ ball_trajectory_export exists: {len(ball_trajectory_export)} frames")
elif 'ball_trajectory' in dir():
    print(f"  âœ“ ball_trajectory exists: {len(ball_trajectory)} frames")
else:
    print(f"  âœ— NO ball trajectory found - ball will not appear in GIF!")
print()

try:
    field_gif_filename = f"{output_dir}/{game_id}_{play_id}_{pass_result}_01_FIELD_ANIMATION.gif"
    
    print(f"Building field animation frames...")
    
    # Use saved variables from field animation cell
    frames_to_export = frames_all_export if 'frames_all_export' in dir() else frames_all
    ball_traj = ball_trajectory_export if 'ball_trajectory_export' in dir() else (ball_trajectory if 'ball_trajectory' in dir() else {})
    
    field_frames = []
    total_frame_count = len(frames_to_export)
    
    print(f"  Total frames to export: {total_frame_count}")
    print(f"  Ball trajectory frames: {len(ball_traj)}\n")
    
    for anim_idx in range(total_frame_count):
        # Create fresh figure for each frame
        fig_export, ax_export = plt.subplots(figsize=(14, 7), dpi=80)
        
        # Draw field background
        ax_export.add_patch(patches.Rectangle((0, 0), 120.0, 53.3,
                                   facecolor='#8DBF87', edgecolor='#2E7D32', linewidth=2, zorder=0))
        ax_export.add_patch(patches.Rectangle((0, 0), 10, 53.3,
                                   facecolor='#5a8f5a', edgecolor='#2E7D32', linewidth=1, zorder=0, alpha=0.8))
        ax_export.add_patch(patches.Rectangle((110, 0), 10, 53.3,
                                   facecolor='#5a8f5a', edgecolor='#2E7D32', linewidth=1, zorder=0, alpha=0.8))
        
        # Yard lines
        for xline in range(10, 120, 10):
            ax_export.plot([xline, xline], [0, 53.3], color='white', linewidth=0.8, alpha=0.4, zorder=0)
        
        # Scrimmage line
        ax_export.plot([scrimmage_x, scrimmage_x], [0, 53.3], color='yellow', linestyle='-', linewidth=3, alpha=0.7, zorder=1)
        
        # Ball landing marker (FIXED: using ball_land_x, ball_land_y)
        ax_export.plot(ball_land_x, ball_land_y, marker='X', markersize=12, color='black', mew=1.5, zorder=3)
        
        ax_export.set_xlim(-1, 121)
        ax_export.set_ylim(-1, 54.3)
        ax_export.set_aspect('equal', adjustable='box')
        ax_export.set_xlabel('X (yards)', fontsize=10)
        ax_export.set_ylabel('Y (yards)', fontsize=10)
        
        # Get frame number
        frame_idx = anim_idx
        fnum = int(frames_to_export[frame_idx])
        
        # Draw players
        for p in players:
            x = p['x'][:frame_idx + 1]
            y = p['y'][:frame_idx + 1]
            
            valid = ~(np.isnan(x) | np.isnan(y))
            if np.any(valid):
                # Trail line
                ax_export.plot(x[valid], y[valid], color=p['color'], lw=p['line_width'],
                       alpha=p['alpha'], zorder=2)
                # Current position dot
                marker_sz = p.get('marker_size', 5)
                ax_export.plot([x[valid][-1]], [y[valid][-1]], marker='o', markersize=marker_sz,
                       color=p['color'], mec='k', mew=0.4, zorder=4, alpha=p['alpha'])
                # Name label
                ax_export.text(x[valid][-1] + 0.3, y[valid][-1] + 0.3, p['name'],
                       fontsize=7, color=p['color'], weight='bold', zorder=5, alpha=p['alpha'])

        # Draw ball trajectory
        current_ball_trail_x = []
        current_ball_trail_y = []
        
        # Use ball trajectory
        bt = ball_traj
        
        for bf in sorted(bt.keys()):
            if bf <= fnum:
                bx_pos, by_pos = bt[bf]
                current_ball_trail_x.append(float(bx_pos))
                current_ball_trail_y.append(float(by_pos))
        
        if len(current_ball_trail_x) > 0:
            ax_export.plot(current_ball_trail_x, current_ball_trail_y, color='goldenrod',
                   linestyle=':', linewidth=2.5, alpha=0.8, zorder=3)
            
            # Draw football
            current_x = current_ball_trail_x[-1]
            current_y = current_ball_trail_y[-1]
            
            angle = 0
            if len(current_ball_trail_x) > 1:
                dx = current_ball_trail_x[-1] - current_ball_trail_x[-2]
                dy = current_ball_trail_y[-1] - current_ball_trail_y[-2]
                if dx != 0 or dy != 0:
                    angle = np.degrees(np.arctan2(dy, dx))
            
            football = patches.Ellipse((current_x, current_y), width=1.2, height=0.6,
                                      angle=angle, color='brown', zorder=10, ec='black', linewidth=1.2)
            ax_export.add_patch(football)
            
            # Draw laces
            angle_rad = np.radians(angle)
            for i in range(5):
                t = -0.4 + (i * 0.2)
                lace_x1 = current_x + t * np.cos(angle_rad)
                lace_y1 = current_y + t * np.sin(angle_rad)
                perp_x = -np.sin(angle_rad) * 0.25
                perp_y = np.cos(angle_rad) * 0.25
                ax_export.plot([lace_x1 + perp_x, lace_x1 - perp_x],
                       [lace_y1 + perp_y, lace_y1 - perp_y], 'w-', linewidth=0.6, zorder=11)
        
        ax_export.set_title(f'Game {game_id} | Play {play_id} | Result: {pass_result}\nBright=Tracked | Gray=Not Tracked',
                    fontsize=12, fontweight='bold')
        ax_export.grid(True, alpha=0.2)
        
        # Frame label
        ax_export.text(2, 51.3, f"Frame {fnum}", fontsize=11, color="black", weight='bold', zorder=6)
        
        # Render to image
        buf = io.BytesIO()
        fig_export.savefig(buf, format='png', bbox_inches='tight', dpi=80)
        buf.seek(0)
        
        frame = Image.open(buf)
        frame_rgb = frame.convert('RGB')
        field_frames.append(frame_rgb)
        buf.close()
        plt.close(fig_export)
        
        # Progress update
        if (anim_idx + 1) % max(1, total_frame_count // 10) == 0:
            print(f"  âœ“ Extracted {anim_idx + 1}/{total_frame_count} frames")
    
    print(f"\nCreating field animation GIF...")
    
    # Save as GIF
    field_frames[0].save(
        field_gif_filename,
        save_all=True,
        append_images=field_frames[1:],
        duration=100,
        loop=0,
        optimize=False
    )
    
    field_size = os.path.getsize(field_gif_filename) / (1024*1024)
    print(f"âœ“ Field animation GIF saved!")
    print(f"  Filename: {os.path.basename(field_gif_filename)}")
    print(f"  Frames: {len(field_frames)}")
    print(f"  File size: {field_size:.1f} MB\n")
    
    files_exported.append(field_gif_filename)

except NameError as e:
    print(f"âš  Field animation not available: {str(e)}")
    print("  Make sure to run the field animation cell first!\n")
except Exception as e:
    print(f"âš  Error exporting field animation: {str(e)}\n")

# ========= STEP 2: EXPORT DISTANCE GRAPH ANIMATION TO GIF =========
print("STEP 2: Exporting Distance Graph Animation to GIF")
print("-"*100 + "\n")

try:
    distance_gif_filename = f"{output_dir}/{game_id}_{play_id}_{pass_result}_02_DISTANCE_GRAPH.gif"
    
    print(f"Extracting frames from distance graph animation...")
    
    distance_frames = []
    
    for anim_idx in range(len(distance_animation_frames)):
        # Update the animation
        distance_animation_cursor(distance_animation_frames[anim_idx])
        
        # Draw to canvas
        distance_animation_fig.canvas.draw()
        
        # Convert to image
        buf = io.BytesIO()
        distance_animation_fig.savefig(buf, format='png', bbox_inches='tight', dpi=70)
        buf.seek(0)
        
        frame = Image.open(buf)
        frame_rgb = frame.convert('RGB')
        distance_frames.append(frame_rgb)
        buf.close()
        
        # Progress update
        if (anim_idx + 1) % max(1, len(distance_animation_frames) // 10) == 0:
            print(f"  âœ“ Extracted {anim_idx + 1}/{len(distance_animation_frames)} frames")
    
    print(f"\nCreating distance graph GIF...")
    
    # Save as GIF
    distance_frames[0].save(
        distance_gif_filename,
        save_all=True,
        append_images=distance_frames[1:],
        duration=100,
        loop=0,
        optimize=False
    )
    
    distance_size = os.path.getsize(distance_gif_filename) / (1024*1024)
    print(f"âœ“ Distance graph GIF saved!")
    print(f"  Filename: {os.path.basename(distance_gif_filename)}")
    print(f"  Frames: {len(distance_frames)}")
    print(f"  File size: {distance_size:.1f} MB\n")
    
    files_exported.append(distance_gif_filename)

except NameError as e:
    print(f"âš  Distance graph not available: {str(e)}")
    print("  Make sure to run the distance graph cell first!\n")
except Exception as e:
    print(f"âš  Error exporting distance graph: {str(e)}\n")

# ========= STEP 3: EXPORT PEI BAR CHART AS PNG =========
print("STEP 3: Exporting PEI Bar Chart as PNG")
print("-"*100 + "\n")

try:
    pei_png_filename = f"{output_dir}/{game_id}_{play_id}_{pass_result}_03_PEI_BARCHART.png"
    
    print(f"Creating PEI bar chart figure...")
    
    # Use the single-play pei_df (from Document 7)
    # Need to check which DataFrame is available and has player_name
    
    # Try to find the correct pei dataframe with player names
    pei_chart_df = None
    
    # Check if pei_df has player_name column
    if 'pei_df' in dir() and 'player_name' in pei_df.columns:
        pei_chart_df = pei_df.copy()
        print("  Using pei_df with player_name column")
    # Otherwise try to rebuild from pei_results if available
    elif 'pei_results' in dir() and len(pei_results) > 0:
        pei_chart_df = pd.DataFrame(pei_results).sort_values("PEI_simple", ascending=False).reset_index(drop=True)
        print("  Using pei_results list")
    else:
        raise NameError("No valid PEI data with player_name found. Run the single-play PEI cell first.")
    
    fig_pei, ax_pei = plt.subplots(figsize=(12, 8), dpi=100)
    
    # Get colors
    try:
        colors = [get_player_color(name, color_map, has_pei=True) for name in pei_chart_df["player_name"]]
    except (NameError, KeyError):
        # Fallback: use colormap based on PEI values
        import matplotlib.cm as cm
        norm = plt.Normalize(pei_chart_df["PEI_simple"].min(), pei_chart_df["PEI_simple"].max())
        colors = cm.RdYlGn(norm(pei_chart_df["PEI_simple"]))
        print("  Using fallback colormap (RdYlGn)")
    
    # Create horizontal bar chart
    bars = ax_pei.barh(pei_chart_df["player_name"], pei_chart_df["PEI_simple"], color=colors)
    ax_pei.invert_yaxis()
    
    # Add PEI values as text labels
    for i, (player_name, pei_value) in enumerate(zip(pei_chart_df["player_name"], pei_chart_df["PEI_simple"])):
        text_color = 'white' if pei_value > 0.5 else 'black'
        ax_pei.text(pei_value / 2, i, f'{pei_value:.3f}',
                   va='center', ha='center', fontsize=10, fontweight='bold', color=text_color)
    
    ax_pei.set_xlabel("PEI_simple (0..1)", fontsize=12, fontweight="bold")
    ax_pei.set_title(f"PEI_simple â€” Game {game_id} | Play {play_id} | Pass Result: {pass_result} | Route: {targeted_route}",
                    fontsize=14, fontweight="bold")
    ax_pei.grid(axis="x", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(pei_png_filename, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig_pei)
    
    pei_size = os.path.getsize(pei_png_filename) / (1024*1024)
    print(f"âœ“ PEI bar chart PNG saved!")
    print(f"  Filename: {os.path.basename(pei_png_filename)}")
    print(f"  File size: {pei_size:.2f} MB\n")
    
    files_exported.append(pei_png_filename)

except NameError as e:
    print(f"âš  PEI bar chart not available: {str(e)}")
    print("  Make sure to run the single-play PEI calculation cell (Document 7) first!\n")
except Exception as e:
    print(f"âš  Error exporting PEI bar chart: {str(e)}\n")

# ========= STEP 4: EXPORT CSV FILES =========
print("STEP 4: Exporting CSV Files")
print("-"*100 + "\n")

# Export all_pei_results (FIXED: simplified filename)
try:
    pei_csv_filename = f"{output_dir}/all_pei_result.csv"
    
    # Convert to DataFrame if it's a list
    if isinstance(all_pei_results, list):
        pei_results_df = pd.DataFrame(all_pei_results)
    else:
        pei_results_df = all_pei_results
    
    pei_results_df.to_csv(pei_csv_filename, index=False)
    
    pei_csv_size = os.path.getsize(pei_csv_filename) / 1024
    print(f"âœ“ All PEI Results CSV saved!")
    print(f"  Filename: {os.path.basename(pei_csv_filename)}")
    print(f"  Rows: {len(pei_results_df)}")
    print(f"  File size: {pei_csv_size:.1f} KB\n")
    
    files_exported.append(pei_csv_filename)

except NameError as e:
    print(f"âš  all_pei_results not available: {str(e)}")
    print("  Make sure to run the PEI calculation cell first!\n")
except Exception as e:
    print(f"âš  Error exporting all_pei_results: {str(e)}\n")

# Export route_summary
try:
    route_csv_filename = f"{output_dir}/route_summary.csv"
    
    route_summary.to_csv(route_csv_filename, index=False)
    
    route_csv_size = os.path.getsize(route_csv_filename) / 1024
    print(f"âœ“ Route Summary CSV saved!")
    print(f"  Filename: {os.path.basename(route_csv_filename)}")
    print(f"  Rows: {len(route_summary)}")
    print(f"  File size: {route_csv_size:.1f} KB\n")
    
    files_exported.append(route_csv_filename)

except NameError as e:
    print(f"âš  route_summary not available: {str(e)}")
    print("  Make sure to run the route summary cell first!\n")
except Exception as e:
    print(f"âš  Error exporting route_summary: {str(e)}\n")

# ========= SUMMARY =========
print("="*100)
print("EXPORT COMPLETE")
print("="*100 + "\n")

if files_exported:
    print(f"Successfully exported {len(files_exported)} file(s) to /kaggle/working:\n")
    
    total_size = 0
    for file_path in files_exported:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            total_size += file_size
            
            if file_path.endswith('.gif'):
                file_type = "[GIF]"
                size_str = f"{file_size/(1024*1024):.1f} MB"
            elif file_path.endswith('.png'):
                file_type = "[PNG]"
                size_str = f"{file_size/(1024*1024):.2f} MB"
            else:
                file_type = "[CSV]"
                size_str = f"{file_size/1024:.1f} KB"
            
            print(f"  {file_type} {os.path.basename(file_path)} ({size_str})")
    
    print(f"\nTotal size: {total_size/(1024*1024):.2f} MB")
    print("\nğŸ“� Files are available in the 'Output' tab on the right panel")
    print("   Click on any file to preview or download")
else:
    print("No files were exported. Make sure to run all prerequisite cells first.")

print("\n" + "="*100)

