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


import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

# --- 1. SET UP PATHS ---
# The competition data is stored in a nested subfolder
base_path = Path('/kaggle/input/nfl-big-data-bowl-2026-analytics')
# Find the actual data folder inside (e.g., '114239_nfl_competition_files...')
data_dir = next(base_path.glob('*/')) 

# Define specific file paths
supp_path = data_dir / 'supplementary_data.csv'
train_dir = data_dir / 'train'

# --- 2. LOAD DATA ---
print(f"Loading Play Data from: {supp_path}")
# Load the supplementary data (contains play descriptions, downs, etc.)
# plays = pd.read_csv(supp_path)
plays = pd.read_csv(supp_path, low_memory=False)

# Load tracking data for Weeks 1-3
print("Loading Tracking Data (Weeks 1-3)...")
# Note: We use 'output_*.csv' because it contains frames AFTER the throw
file_patterns = [
    str(train_dir / 'output_2023_w01.csv'),
    str(train_dir / 'output_2023_w02.csv'),
    str(train_dir / 'output_2023_w03.csv')
]

df_list = []
for file in file_patterns:
    if os.path.exists(file):
        df_temp = pd.read_csv(file)
        df_list.append(df_temp)
    else:
        print(f"Warning: File not found: {file}")

df = pd.concat(df_list, ignore_index=True)
print(f"Successfully loaded {len(df):,} frames of tracking data.")


# --- CELL 2: PREPROCESSING (ERROR-PROOF VERSION) ---

print("Step 1: Extracting metadata from input files...")

# Define the columns we absolutely need from the input files
input_cols = ['game_id', 'play_id', 'nfl_id', 'player_role', 'dir', 'ball_land_x', 'ball_land_y']
input_file_patterns = [
    str(train_dir / 'input_2023_w01.csv'),
    str(train_dir / 'input_2023_w02.csv'),
    str(train_dir / 'input_2023_w03.csv')
]

meta_list = []
for file in input_file_patterns:
    if os.path.exists(file):
        t_df = pd.read_csv(file, usecols=input_cols)
        # Take the state at the end of the input (moment of throw)
        t_df = t_df.drop_duplicates(subset=['game_id', 'play_id', 'nfl_id'], keep='last')
        meta_list.append(t_df)

df_meta = pd.concat(meta_list, ignore_index=True)

# --- CLEAN UP EXISTING COLUMNS TO PREVENT MERGE ERRORS ---
# If df already has metadata columns from a previous failed run, drop them now
cols_to_remove = ['player_role', 'dir', 'ball_land_x', 'ball_land_y']
df = df.drop(columns=[c for c in cols_to_remove if c in df.columns])

# --- STANDARDIZE TYPES ---
for col in ['game_id', 'play_id', 'nfl_id']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.int64)
    df_meta[col] = pd.to_numeric(df_meta[col], errors='coerce').fillna(0).astype(np.int64)

print(f"Step 2: Merging... (Tracking rows: {len(df)}, Metadata rows: {len(df_meta)})")

# Merge with explicit suffixes to avoid duplicate naming conflicts
df = df.merge(df_meta, on=['game_id', 'play_id', 'nfl_id'], how='inner')

print(f"Post-Merge row count: {len(df)}")

if len(df) == 0:
    print("!!! ERROR: Merge resulted in 0 rows. Check ID matching.")
else:
    # Step 3: Filter for Defensive Coverage
    print("Step 3: Filtering for Defensive Coverage...")
    if 'player_role' in df.columns:
        df_def = df[df['player_role'] == 'Defensive Coverage'].copy()
        
        if len(df_def) == 0:
            print("!!! ERROR: No 'Defensive Coverage' found. Roles found:", df['player_role'].unique())
        else:
            # Step 4: Final Cleanup
            # Drop rows where critical tracking or target data is missing
            subset_to_check = ['x', 'y', 'ball_land_x', 'ball_land_y', 'dir']
            existing_subset = [c for c in subset_to_check if c in df_def.columns]
            
            df_def = df_def.dropna(subset=existing_subset)
            print(f"Success! Final dataset contains {len(df_def):,} frames.")
            print(df_def[['game_id', 'play_id', 'player_role', 'x', 'ball_land_x']].head())
    else:
        print("!!! ERROR: 'player_role' column still missing!")


# --- CELL 3: PHYSICS ENGINE ---

def calculate_ideal_heading(player_x, player_y, target_x, target_y):
    """
    Calculates the 'Perfect Angle' from player to target in NFL coordinate system.
    NFL System: 0=North (Y+), 90=East (X+), 180=South (Y-), 270=West (X-)
    """
    # Vector components
    dx = target_x - player_x
    dy = target_y - player_y
    
    # Standard math angle (0=East, Counter-Clockwise)
    rads = np.arctan2(dy, dx)
    deg_standard = np.degrees(rads)
    
    # Convert Standard to NFL (0=North, Clockwise)
    # Formula: NFL_Angle = (90 - Standard_Angle) % 360
    nfl_deg = (90 - deg_standard) % 360
    return nfl_deg

def get_angle_difference(angle1, angle2):
    """Calculates smallest difference between two angles (handling 0/360 wrapping)."""
    diff = np.abs(angle1 - angle2)
    return np.minimum(diff, 360 - diff)

# Apply calculations using Vectorization (Fast!)
print("Calculating Ideal Vectors...")
df_def['ideal_dir'] = calculate_ideal_heading(
    df_def['x'].values, 
    df_def['y'].values, 
    df_def['ball_land_x'].values, 
    df_def['ball_land_y'].values
)

print("Calculating Efficiency Metric...")
# The Metric: Directional Error (Lower is Better)
df_def['dir_error'] = get_angle_difference(df_def['dir'], df_def['ideal_dir'])

# Calculate Distance to Ball for context
df_def['dist_to_ball'] = np.sqrt(
    (df_def['x'] - df_def['ball_land_x'])**2 + 
    (df_def['y'] - df_def['ball_land_y'])**2
)


# --- CELL 4: SCORING & ANALYSIS ---

# 1. Group by Play and Player to get an average score per play
play_stats = df_def.groupby(['game_id', 'play_id', 'nfl_id']).agg(
    avg_error=('dir_error', 'mean'),        # The Metric
    max_error=('dir_error', 'max'),         # Did they get completely turned around?
    start_dist=('dist_to_ball', 'first'),   # How far were they at the throw?
    end_dist=('dist_to_ball', 'last'),      # How close were they at the catch?
    frames=('frame_id', 'count')            # Duration of play
).reset_index()

# 2. Apply Logical Filters
# Filter: Only look at defenders who started within 30 yards of the landing spot.
# Filter: Only look at plays that lasted at least 1 second (10 frames).
relevant_plays = play_stats[
    (play_stats['start_dist'] < 30) & 
    (play_stats['frames'] > 10)
]

# 3. Merge with Play Description for Context
analysis = relevant_plays.merge(
    plays[['game_id', 'play_id', 'play_description', 'pass_result', 'defensive_team']], 
    on=['game_id', 'play_id']
)

# 4. Identify the "Best" and "Worst" Tracking
top_performers = analysis.sort_values('avg_error').head(5)
bottom_performers = analysis.sort_values('avg_error', ascending=False).head(5)

print("\n--- TOP 5 'LASER GUIDED' DEFENDERS (Best Tracking) ---")
print(top_performers[['nfl_id', 'avg_error', 'play_description']].to_string(index=False))

print("\n--- TOP 5 'LOST' DEFENDERS (Worst Tracking) ---")
print(bottom_performers[['nfl_id', 'avg_error', 'play_description']].to_string(index=False))


# --- CELL 5: VISUALIZATION ---
import matplotlib.pyplot as plt
def plot_tracking_efficiency(game_id, play_id, nfl_id, tracking_df):
    """
    Plots a single play showing the defender's path colored by their efficiency.
    """
    # Extract data for this specific player and play
    play_data = tracking_df[
        (tracking_df['game_id'] == game_id) & 
        (tracking_df['play_id'] == play_id) & 
        (tracking_df['nfl_id'] == nfl_id)
    ].sort_values('frame_id')
    
    if play_data.empty:
        print("Error: No data found for this play.")
        return

    # Create the Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 1. Draw the Field (Simplified)
    ax.axhspan(0, 53.3, color='#f8f9fa') # Grass color
    for x in range(0, 121, 10):
        ax.axvline(x, color='lightgrey', linestyle='--', alpha=0.5)
    
    # 2. Plot the Ball Landing Spot
    land_x = play_data.iloc[0]['ball_land_x']
    land_y = play_data.iloc[0]['ball_land_y']
    ax.scatter(land_x, land_y, s=300, c='gold', marker='*', edgecolors='black', label='Ball Landing', zorder=10)
    
    # 3. Plot the Path (Colored by Error Metric)
    # We use a scatter plot to simulate a line with changing colors
    sc = ax.scatter(
        play_data['x'], 
        play_data['y'], 
        c=play_data['dir_error'], 
        cmap='RdYlGn_r',  # Red-Yellow-Green (Reversed so 0 error is Green)
        s=100, 
        edgecolors='black',
        linewidth=0.5,
        label='Defender Path'
    )
    
    # 4. Add Directional Arrows every few frames
    for i in range(0, len(play_data), 5): # Every 5th frame
        row = play_data.iloc[i]
        # Calculate arrow components based on 'dir'
        dx = np.sin(np.radians(row['dir'])) * 1.5
        dy = np.cos(np.radians(row['dir'])) * 1.5
        ax.arrow(row['x'], row['y'], dx, dy, head_width=0.5, head_length=0.5, fc='black', alpha=0.3)

    # 5. Annotation and Formatting
    cbar = plt.colorbar(sc)
    cbar.set_label('Directional Error (Degrees)\nGreen = Perfect Tracking | Red = Poor Tracking')
    
    avg_err = play_data['dir_error'].mean()
    ax.set_title(f"Defensive Reaction Efficiency (DRE)\nAvg Error: {avg_err:.1f}°", fontsize=16, fontweight='bold')
    ax.set_xlabel('Long Axis (Yards)')
    ax.set_ylabel('Short Axis (Yards)')
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    
    # Zoom in on the action (Auto-scale)
    buffer = 10
    ax.set_xlim(min(play_data['x'].min(), land_x) - buffer, max(play_data['x'].max(), land_x) + buffer)
    ax.set_ylim(min(play_data['y'].min(), land_y) - buffer, max(play_data['y'].max(), land_y) + buffer)
    
    plt.show()

# --- EXECUTE VISUALIZATION ---
# Automatically plot the #1 Best Play found in the previous cell
if not top_performers.empty:
    best_play = top_performers.iloc[0]
    print(f"Visualizing Best Play: {best_play['play_description']}")
    plot_tracking_efficiency(best_play['game_id'], best_play['play_id'], best_play['nfl_id'], df_def)
else:
    print("No relevant plays found to visualize.")
    # Technical Analysis: Time to Correct (TTC)
# Calculating how many frames it takes for the defender to get under 15 degrees of error
def calculate_ttc(df, threshold=15):
    # Find the first frame where error is below threshold
    corrected_frames = df[df['dir_error'] <= threshold]
    if not corrected_frames.empty:
        return corrected_frames['frame_id'].min() - df['frame_id'].min()
    return np.nan

print(f"Frames to reach efficient tracking: {calculate_ttc(df_def[df_def['play_id'] == best_play['play_id']])} frames")


# --- CELL 6: GENERATE OUTPUT DATA ---

# 1. Final Aggregation of your DRE Metric
# We group by player and team to find the most efficient trackers in the league
league_leaderboard = analysis.groupby(['nfl_id', 'defensive_team']).agg(
    avg_dre_error=('avg_error', 'mean'),
    total_plays_analyzed=('play_id', 'count'),
    min_error_achieved=('avg_error', 'min')
).reset_index()

# 2. Filter for sample size (e.g., players with at least 5 qualifying plays)
leaderboard_final = league_leaderboard[league_leaderboard['total_plays_analyzed'] >= 3]
leaderboard_final = leaderboard_final.sort_values('avg_dre_error', ascending=True)

# 3. Save to CSV (This will appear in your /kaggle/working/ directory)
output_filename = 'defensive_reaction_efficiency_results.csv'
leaderboard_final.to_csv(output_filename, index=False)

print(f"Successfully generated {output_filename}")
print("\n--- PREVIEW OF LEADERBOARD ---")
print(leaderboard_final.head(10))







