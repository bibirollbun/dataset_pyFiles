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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import glob
import os
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore')

# Configuration
# Adjust these paths if the directory structure changes in your specific notebook environment
BASE_DIR = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final'
TRAIN_DIR = f"{BASE_DIR}/train"

# Graphics Settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("Environment Configured.")


def load_and_merge_data(weeks=None):
    """
    Loads Input (Context) and Output (Trajectory) data and links them.
    """
    data_frames = []
    
    # If weeks is None, load all. For memory safety in this demo, we default to first 3 weeks.
    if weeks is None:
        week_files = sorted(glob.glob(f"{TRAIN_DIR}/input_2023_w*.csv"))[:3]
    else:
        week_files = [f"{TRAIN_DIR}/input_2023_w{w:02d}.csv" for w in weeks]
        
    print(f"Processing {len(week_files)} weeks of data...")
    
    for input_file in tqdm(week_files):
        # 1. Load Input (Context: Ball Landing, Role)
        df_in = pd.read_csv(input_file)
        
        # Filter for relevant plays immediately to save memory
        # We need 'ball_land_x' to exist
        df_in = df_in.dropna(subset=['ball_land_x', 'ball_land_y'])
        
        # 2. Load Output (Trajectory)
        # Construct matching output filename
        week_num = input_file.split('_w')[-1].split('.')[0]
        output_file = f"{TRAIN_DIR}/output_2023_w{week_num}.csv"
        
        if not os.path.exists(output_file):
            continue
            
        df_out = pd.read_csv(output_file)
        
        # 3. Merge Context into Trajectory
        # We need 'ball_land' and 'play_direction' inside the Output frames to do math
        context_cols = ['game_id', 'play_id', 'nfl_id', 'play_direction', 
                        'ball_land_x', 'ball_land_y', 'player_role', 'player_side']
        
        # We merge on game/play/nfl to get role, but game/play to get ball_land
        # Optimization: Create a play-level lookup for ball_land/direction
        play_meta = df_in[['game_id', 'play_id', 'play_direction', 'ball_land_x', 'ball_land_y']].drop_duplicates()
        player_meta = df_in[['game_id', 'play_id', 'nfl_id', 'player_role', 'player_side']].drop_duplicates()
        
        # Merge Play Metadata
        df_merged = df_out.merge(play_meta, on=['game_id', 'play_id'], how='inner')
        # Merge Player Metadata
        df_merged = df_merged.merge(player_meta, on=['game_id', 'play_id', 'nfl_id'], how='inner')
        
        data_frames.append(df_merged)
        
    full_df = pd.concat(data_frames, ignore_index=True)
    return full_df

# Load Data (Weeks 1-3 for demonstration speed)
tracking_data = load_and_merge_data(weeks=[1, 2, 3])
print(f"Loaded {len(tracking_data):,} frames of tracking data.")


def standardize_tracking(df):
    """
    Standardizes coordinates so all plays move Left-to-Right.
    Calculates Velocity Vectors if missing from Output.
    """
    df = df.copy()
    
    # 1. Flip Field (Left -> Right)
    # If play_direction is left, x becomes 120-x, y becomes 53.3-y
    mask = df['play_direction'] == 'left'
    df.loc[mask, 'x'] = 120.0 - df.loc[mask, 'x']
    df.loc[mask, 'y'] = 53.3 - df.loc[mask, 'y']
    df.loc[mask, 'ball_land_x'] = 120.0 - df.loc[mask, 'ball_land_x']
    df.loc[mask, 'ball_land_y'] = 53.3 - df.loc[mask, 'ball_land_y']
    
    # 2. Calculate Velocity Vectors (vx, vy)
    # Output data often has just x,y. We need dx, dy.
    # Sort by player and frame
    df = df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    
    # Groupby shift to get previous position
    df['prev_x'] = df.groupby(['game_id', 'play_id', 'nfl_id'])['x'].shift(1)
    df['prev_y'] = df.groupby(['game_id', 'play_id', 'nfl_id'])['y'].shift(1)
    
    # Calculate velocity (units: yards/frame)
    # Note: 1 frame = 0.1s. To get yards/s, multiply by 10.
    df['vx'] = (df['x'] - df['prev_x']).fillna(0)
    df['vy'] = (df['y'] - df['prev_y']).fillna(0)
    
    # Calculate Speed (Scalar)
    df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2) * 10.0 # yards/sec
    
    return df

processed_df = standardize_tracking(tracking_data)
print("Physics Standardization Complete.")


def calculate_metrics(df):
    # Filter: Only analyze Defensive players
    # We want to see how THEY react to the ball
    df = df[df['player_role'] == 'Defensive Coverage'].copy()
    
    # 1. Construct Ideal Vector (Player -> Ball)
    df['ideal_vx'] = df['ball_land_x'] - df['x']
    df['ideal_vy'] = df['ball_land_y'] - df['y']
    
    # 2. Compute Magnitudes
    df['mag_ideal'] = np.sqrt(df['ideal_vx']**2 + df['ideal_vy']**2)
    df['mag_actual'] = np.sqrt(df['vx']**2 + df['vy']**2)
    
    # 3. Compute Dot Product
    df['dot_prod'] = (df['vx'] * df['ideal_vx']) + (df['vy'] * df['ideal_vy'])
    
    # 4. Cosine Similarity
    # Add epsilon to avoid divide by zero for stationary players
    df['similarity'] = df['dot_prod'] / ((df['mag_ideal'] * df['mag_actual']) + 1e-6)
    
    # 5. Aggregate into "Reaction Time" per play
    results = []
    
    for (game, play, nfl), grp in tqdm(df.groupby(['game_id', 'play_id', 'nfl_id'])):
        grp = grp.sort_values('frame_id')
        
        # Reaction Threshold: When do they orient > 0.75 similarity?
        # We look for a sustained reaction (e.g., 3 consecutive frames)
        react_mask = grp['similarity'] > 0.75
        
        # Rolling window to ensure it's not just a twitch
        sustained_reaction = react_mask.rolling(3).min().fillna(0)
        
        reaction_frames = grp[sustained_reaction == 1]
        
        if not reaction_frames.empty:
            # First frame of sustained reaction
            first_react = reaction_frames.iloc[0]['frame_id']
            # Convert to seconds (frames start at 1)
            reaction_time = first_react / 10.0
        else:
            # Did not react efficiently (or play was too short)
            reaction_time = np.nan
            
        # Efficiency: Average similarity during the play
        avg_efficiency = grp['similarity'].mean()
        
        results.append({
            'nfl_id': nfl,
            'reaction_time': reaction_time,
            'path_efficiency': avg_efficiency,
            'game_id': game,
            'play_id': play
        })
        
    return pd.DataFrame(results)

metrics_df = calculate_metrics(processed_df)
print("Metrics Calculated.")
print(metrics_df.head())


# 1. Cleaning
valid_metrics = metrics_df.dropna()

# 2. Summary Stats
print(f"Average Defensive Reaction Time: {valid_metrics['reaction_time'].mean():.3f} seconds")
print(f"Average Path Efficiency (Cosine Sim): {valid_metrics['path_efficiency'].mean():.3f}")

# 3. Distribution Plot
plt.figure(figsize=(10, 5))
sns.histplot(valid_metrics['reaction_time'], bins=30, kde=True, color='navy')
plt.title('Distribution of Defensive Reaction Times (Post-Throw)')
plt.xlabel('Seconds to Lock-On')
plt.ylabel('Frequency')
plt.axvline(valid_metrics['reaction_time'].mean(), color='red', linestyle='--', label='Mean')
plt.legend()
plt.show()


def plot_reaction_play(play_df, nfl_id):
    """
    Plots a single player's path colored by their 'Ball-Hawk' similarity score.
    """
    player_track = play_df[play_df['nfl_id'] == nfl_id].sort_values('frame_id')
    
    if player_track.empty:
        return
    
    # Setup Field
    plt.figure(figsize=(12, 6))
    plt.xlim(0, 120)
    plt.ylim(0, 53.3)
    plt.axvline(10, color='white', linestyle='-') # Endzone
    plt.axvline(110, color='white', linestyle='-') # Endzone
    plt.gca().set_facecolor('darkgreen')
    
    # Plot Ball Landing
    ball_x = player_track.iloc[0]['ball_land_x']
    ball_y = player_track.iloc[0]['ball_land_y']
    plt.scatter(ball_x, ball_y, color='yellow', s=200, marker='*', label='Ball Land', zorder=10)
    
    # Plot Path Segments colored by Similarity
    # We iterate segments to color them individually
    x = player_track['x'].values
    y = player_track['y'].values
    sim = player_track['similarity'].values
    
    # Create a colormap (Red=Bad, Green=Good)
    cmap = plt.get_cmap('RdYlGn')
    
    for i in range(len(x) - 1):
        # Color based on similarity score (-1 to 1 mapped to 0 to 1)
        # Using max(0, sim) effectively maps negatives to Red
        color_val = max(0.0, min(1.0, (sim[i] + 0.2) / 1.2)) 
        plt.plot([x[i], x[i+1]], [y[i], y[i+1]], color=cmap(color_val), linewidth=4)
        
    plt.title(f"Ball-Hawk Visualization: Player {nfl_id}", fontsize=14, color='white')
    plt.xlabel("Field Length", color='white')
    plt.ylabel("Field Width", color='white')
    plt.tick_params(colors='white')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

# Select a random play to visualize
# We need to re-calculate similarity for plotting since we only stored aggregates
sample_play_key = processed_df[['game_id', 'play_id']].drop_duplicates().iloc[5] # Grab 5th play
sample_df = processed_df[(processed_df['game_id'] == sample_play_key.game_id) & 
                         (processed_df['play_id'] == sample_play_key.play_id)].copy()

# Recalculate sim just for this chunk
sample_df = sample_df[sample_df['player_role'] == 'Defensive Coverage']
if not sample_df.empty:
    sample_df['ideal_vx'] = sample_df['ball_land_x'] - sample_df['x']
    sample_df['ideal_vy'] = sample_df['ball_land_y'] - sample_df['y']
    sample_df['mag_ideal'] = np.sqrt(sample_df['ideal_vx']**2 + sample_df['ideal_vy']**2)
    sample_df['mag_actual'] = np.sqrt(sample_df['vx']**2 + sample_df['vy']**2)
    sample_df['dot_prod'] = (sample_df['vx'] * sample_df['ideal_vx']) + (sample_df['vy'] * sample_df['ideal_vy'])
    sample_df['similarity'] = sample_df['dot_prod'] / ((sample_df['mag_ideal'] * sample_df['mag_actual']) + 1e-6)

    # Plot for the first defender found
    defender_id = sample_df['nfl_id'].unique()[0]
    plot_reaction_play(sample_df, defender_id)

