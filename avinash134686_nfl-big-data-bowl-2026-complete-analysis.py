# Core libraries
import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path
from itertools import combinations

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Statistics
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

# Visualization settings
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')

# NFL color palette
NFL_BLUE = '#013369'
NFL_RED = '#D50A0A'
NFL_GREEN = '#00AA00'
NFL_GOLD = '#FFB612'
FIELD_COLOR = '#196F0C'

print("âœ… Environment setup complete!")
print(f"   Pandas version: {pd.__version__}")
print(f"   NumPy version: {np.__version__}")
print(f"   Matplotlib version: {plt.matplotlib.__version__}")


# Configure data paths
# For Kaggle: data is in the competition input directory
import os

# Check if running on Kaggle
if os.path.exists('/kaggle/input'):
    # Kaggle environment - find the competition data
    BASE_PATH = '/kaggle/input/'
    
    # List available data sources
    if os.path.exists(BASE_PATH):
        data_sources = os.listdir(BASE_PATH)
        print(f"Available data sources: {data_sources}")
        
        # Find the competition data folder
        competition_folder = None
        for folder in data_sources:
            if 'nfl' in folder.lower() or 'big-data-bowl' in folder.lower():
                competition_folder = folder
                break
        
        if competition_folder:
            base_comp_path = os.path.join(BASE_PATH, competition_folder)
            
            # Check for nested structure (e.g., /train/ subdirectory)
            possible_paths = [
                base_comp_path + '/train/',  # Nested in train folder
                base_comp_path + '/',        # Direct in competition folder
            ]
            
            # Also check for any subdirectories with long names
            if os.path.exists(base_comp_path):
                subdirs = [d for d in os.listdir(base_comp_path) if os.path.isdir(os.path.join(base_comp_path, d))]
                for subdir in subdirs:
                    subdir_path = os.path.join(base_comp_path, subdir)
                    # Check if it has a train folder
                    if os.path.exists(os.path.join(subdir_path, 'train')):
                        possible_paths.insert(0, os.path.join(subdir_path, 'train') + '/')
            
            # Find the path with CSV files
            DATA_PATH = None
            for path in possible_paths:
                if os.path.exists(path):
                    files = os.listdir(path)
                    csv_files = [f for f in files if f.endswith('.csv') and 'input_2023' in f]
                    if len(csv_files) > 0:
                        DATA_PATH = path
                        print(f"âœ… Found competition data at: {DATA_PATH}")
                        print(f"   Found {len(csv_files)} CSV files")
                        break
            
            if DATA_PATH is None:
                DATA_PATH = base_comp_path + '/'
                print(f"âš ï¸�  Using base path: {DATA_PATH}")
        else:
            # Default to the most likely path
            DATA_PATH = BASE_PATH + 'nfl-big-data-bowl-2026-analytics/'
            print(f"âš ï¸�  Using default path: {DATA_PATH}")
    else:
        DATA_PATH = BASE_PATH
else:
    # Local environment
    DATA_PATH = 'data/train/'
    print(f"ğŸ’» Local environment detected")
    print(f"   Data path: {DATA_PATH}")

print(f"\nğŸ“‚ Loading tracking data from all 18 weeks...")
print("=" * 70)

# Check what files are available
if os.path.exists(DATA_PATH):
    available_files = os.listdir(DATA_PATH)
    csv_files = [f for f in available_files if f.endswith('.csv') and 'input_2023' in f]
    print(f"âœ… Found {len(csv_files)} data files in: {DATA_PATH}")
    if len(csv_files) > 0:
        print(f"   Sample files: {csv_files[:3]}")
else:
    print(f"â�Œ Data path not found: {DATA_PATH}")
    print(f"   Please ensure competition data is added to this notebook")
    print(f"   Go to: Add Data â†’ Search for 'NFL Big Data Bowl 2026'")

all_weeks_data = []
weeks_loaded = 0

for week in range(1, 19):
    try:
        file_path = f"{DATA_PATH}input_2023_w{week:02d}.csv"
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"âš ï¸�  Week {week} file not found: {file_path}")
            continue
            
        df_week = pd.read_csv(file_path)
        all_weeks_data.append(df_week)
        weeks_loaded += 1
        
        if week == 1:  # Display schema for first week
            print(f"\nğŸ“Š Week {week} Schema:")
            print(f"   Rows: {len(df_week):,}")
            print(f"   Columns: {len(df_week.columns)}")
            print(f"   Key columns: game_id, play_id, nfl_id, player_name, x, y, s, a, dir")
        
        if week % 3 == 0:
            print(f"âœ… Loaded weeks 1-{week}...")
    
    except FileNotFoundError:
        print(f"âš ï¸�  Week {week} file not found, skipping...")
        continue
    except Exception as e:
        print(f"â�Œ Error loading week {week}: {str(e)}")
        continue

if len(all_weeks_data) == 0:
    print(f"\nâ�Œ NO DATA LOADED!")
    print(f"\nğŸ“‹ Troubleshooting Steps:")
    print(f"   1. Click 'Add Data' button on the right â†’")
    print(f"   2. Search for 'NFL Big Data Bowl 2026 - Analytics'")
    print(f"   3. Click '+ Add' to attach competition data")
    print(f"   4. Rerun this cell")
    raise FileNotFoundError("No tracking data files found. Please add competition data to notebook.")

# Combine all weeks
df_tracking = pd.concat(all_weeks_data, ignore_index=True)

print(f"\n{'=' * 70}")
print(f"âœ… DATA LOADING COMPLETE")
print(f"{'=' * 70}")
print(f"\nğŸ“ˆ Dataset Statistics:")
print(f"   Weeks loaded: {weeks_loaded}")
print(f"   Total rows: {len(df_tracking):,}")
print(f"   Unique games: {df_tracking['game_id'].nunique():,}")
print(f"   Unique plays: {df_tracking['play_id'].nunique():,}")
print(f"   Unique players: {df_tracking['nfl_id'].nunique():,}")
print(f"   Date range: {df_tracking['game_id'].min()} to {df_tracking['game_id'].max()}")

# Display sample
print(f"\nğŸ“‹ Sample Data (First 5 rows):")
display(df_tracking.head())


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_ball_release_frame(df_play):
    """
    Estimate ball release frame (first frame in tracking data).
    """
    return df_play['frame_id'].min()

# ============================================================================
# COMPONENT 1: DETECTION TIME (40% weight)
# ============================================================================

def calculate_detection_time(df_player, ball_land_x, ball_land_y, ball_release_frame):
    """
    Calculate detection time: frames until player starts moving toward ball landing spot.
    
    Returns:
        - frames_to_reaction: number of frames before reacting
        - detection_score: normalized score (0-1, higher is better)
    """
    df_player = df_player.sort_values('frame_id').copy()
    
    # Calculate angle toward ball landing spot for each frame
    df_player['angle_to_ball'] = np.arctan2(
        ball_land_y - df_player['y'],
        ball_land_x - df_player['x']
    ) * 180 / np.pi
    
    # Normalize angles to 0-360
    df_player['angle_to_ball'] = df_player['angle_to_ball'] % 360
    df_player['dir_normalized'] = df_player['dir'] % 360
    
    # Calculate angle difference (are they moving toward ball?)
    df_player['angle_diff'] = np.abs(df_player['angle_to_ball'] - df_player['dir_normalized'])
    df_player['angle_diff'] = np.minimum(df_player['angle_diff'], 360 - df_player['angle_diff'])
    
    # Find first frame where player is moving toward ball (within 45 degrees)
    moving_toward_ball = df_player[df_player['angle_diff'] < 45]
    
    if len(moving_toward_ball) > 0:
        reaction_frame = moving_toward_ball['frame_id'].iloc[0]
        frames_to_reaction = reaction_frame - ball_release_frame
    else:
        # Never moved toward ball
        frames_to_reaction = len(df_player)
    
    # Normalize to 0-1 scale (faster reaction = higher score)
    max_frames = 20
    detection_score = max(0, 1 - (frames_to_reaction / max_frames))
    
    return frames_to_reaction, detection_score

# ============================================================================
# COMPONENT 2: PATH EFFICIENCY (30% weight)
# ============================================================================

def calculate_path_efficiency(df_player, ball_land_x, ball_land_y):
    """
    Calculate path efficiency: optimal distance vs actual distance traveled.
    
    Returns:
        - optimal_distance: straight-line distance to ball
        - actual_distance: total distance traveled
        - efficiency_score: optimal/actual (0-1, higher is better)
    """
    df_player = df_player.sort_values('frame_id').copy()
    
    # Starting position
    start_x = df_player['x'].iloc[0]
    start_y = df_player['y'].iloc[0]
    
    # Optimal distance (straight line)
    optimal_distance = calculate_distance(start_x, start_y, ball_land_x, ball_land_y)
    
    # Actual distance traveled (sum of frame-to-frame distances)
    df_player['x_next'] = df_player['x'].shift(-1)
    df_player['y_next'] = df_player['y'].shift(-1)
    df_player['step_distance'] = calculate_distance(
        df_player['x'], df_player['y'],
        df_player['x_next'], df_player['y_next']
    )
    actual_distance = df_player['step_distance'].sum()
    
    # Calculate efficiency
    if actual_distance > 0:
        efficiency_score = min(1.0, optimal_distance / actual_distance)
    else:
        efficiency_score = 0.0
    
    return optimal_distance, actual_distance, efficiency_score

# ============================================================================
# COMPONENT 3: ARRIVAL TIMING (30% weight)
# ============================================================================

def calculate_arrival_timing(df_player, ball_land_x, ball_land_y, num_frames_output):
    """
    Calculate arrival timing: how close defender gets to ball landing spot when ball arrives.
    
    Returns:
        - min_distance: closest distance to ball landing spot
        - frame_of_closest: frame when closest
        - timing_score: how well timed (0-1, higher is better)
    """
    df_player = df_player.sort_values('frame_id').copy()
    
    # Calculate distance to ball landing spot for each frame
    df_player['distance_to_ball_land'] = calculate_distance(
        df_player['x'], df_player['y'],
        ball_land_x, ball_land_y
    )
    
    # Find closest approach
    min_distance = df_player['distance_to_ball_land'].min()
    frame_of_closest = df_player.loc[df_player['distance_to_ball_land'].idxmin(), 'frame_id']
    
    # Ball arrives at approximately frame = num_frames_output
    ball_arrival_frame = num_frames_output
    
    # Calculate timing score (within 10 frame window)
    frame_difference = abs(frame_of_closest - ball_arrival_frame)
    timing_score = max(0, 1 - (frame_difference / 10))
    
    # Distance penalty (within 20 yards is good)
    distance_score = max(0, 1 - (min_distance / 20))
    
    # Combined timing score
    timing_score = (timing_score * 0.6) + (distance_score * 0.4)
    
    return min_distance, frame_of_closest, timing_score

# ============================================================================
# OVERALL RTI CALCULATION
# ============================================================================

# Metric weights
DETECTION_WEIGHT = 0.40
PATH_EFFICIENCY_WEIGHT = 0.30
ARRIVAL_TIMING_WEIGHT = 0.30

def calculate_reaction_time_index(df_play, play_info):
    """
    Calculate Defensive Reaction Time Index for all defensive players in a play.
    
    Returns:
        DataFrame with player metrics
    """
    results = []
    
    # Get play metadata
    ball_land_x = play_info['ball_land_x']
    ball_land_y = play_info['ball_land_y']
    num_frames_output = play_info['num_frames_output']
    ball_release_frame = get_ball_release_frame(df_play)
    
    # Get all defensive players
    defensive_players = df_play[df_play['player_side'] == 'Defense']['nfl_id'].unique()
    
    for nfl_id in defensive_players:
        df_player = df_play[df_play['nfl_id'] == nfl_id].copy()
        
        if len(df_player) < 2:  # Need at least 2 frames
            continue
        
        # Get player info
        player_name = df_player['player_name'].iloc[0]
        player_position = df_player['player_position'].iloc[0]
        
        # Calculate components
        frames_to_reaction, detection_score = calculate_detection_time(
            df_player, ball_land_x, ball_land_y, ball_release_frame
        )
        
        optimal_dist, actual_dist, efficiency_score = calculate_path_efficiency(
            df_player, ball_land_x, ball_land_y
        )
        
        min_distance, closest_frame, timing_score = calculate_arrival_timing(
            df_player, ball_land_x, ball_land_y, num_frames_output
        )
        
        # Calculate overall Reaction Time Index (0-100 scale)
        rti = (
            detection_score * DETECTION_WEIGHT +
            efficiency_score * PATH_EFFICIENCY_WEIGHT +
            timing_score * ARRIVAL_TIMING_WEIGHT
        ) * 100
        
        results.append({
            'game_id': play_info['game_id'],
            'play_id': play_info['play_id'],
            'nfl_id': nfl_id,
            'player_name': player_name,
            'player_position': player_position,
            'frames_to_reaction': frames_to_reaction,
            'detection_score': detection_score,
            'optimal_distance': optimal_dist,
            'actual_distance': actual_dist,
            'efficiency_score': efficiency_score,
            'min_distance_to_ball': min_distance,
            'timing_score': timing_score,
            'reaction_time_index': rti
        })
    
    return pd.DataFrame(results)

print("âœ… RTI metric functions defined!")
print("\nMetric Formula:")
print(f"  RTI = (Detection Ã— {DETECTION_WEIGHT}) + (Efficiency Ã— {PATH_EFFICIENCY_WEIGHT}) + (Timing Ã— {ARRIVAL_TIMING_WEIGHT})")


print("ğŸ”„ CALCULATING RTI FOR ALL PLAYS")
print("=" * 70)
print("This may take several minutes...\n")

# Get unique plays
plays = df_tracking.groupby(['game_id', 'play_id']).first().reset_index()
print(f"ğŸ“Š Total plays to process: {len(plays):,}")

# Filter plays with valid ball landing coordinates
plays_valid = plays[
    plays['ball_land_x'].notna() & 
    plays['ball_land_y'].notna()
]
print(f"âœ… Plays with ball landing data: {len(plays_valid):,}\n")

# Calculate metrics for each play
all_results = []
processed = 0
errors = 0

for idx, play_row in plays_valid.iterrows():
    game_id = play_row['game_id']
    play_id = play_row['play_id']
    
    # Get play data
    df_play = df_tracking[
        (df_tracking['game_id'] == game_id) & 
        (df_tracking['play_id'] == play_id)
    ].copy()
    
    play_info = {
        'game_id': game_id,
        'play_id': play_id,
        'ball_land_x': play_row['ball_land_x'],
        'ball_land_y': play_row['ball_land_y'],
        'num_frames_output': play_row['num_frames_output']
    }
    
    # Calculate metrics
    try:
        results = calculate_reaction_time_index(df_play, play_info)
        if len(results) > 0:
            all_results.append(results)
        processed += 1
    except Exception as e:
        errors += 1
        continue
    
    # Progress update
    if processed % 1000 == 0:
        print(f"  Processed {processed:,}/{len(plays_valid):,} plays ({processed/len(plays_valid)*100:.1f}%)...")

# Combine results
df_rti = pd.concat(all_results, ignore_index=True)

print(f"\n{'=' * 70}")
print(f"âœ… RTI CALCULATION COMPLETE!")
print(f"{'=' * 70}")
print(f"\nğŸ“Š Results Summary:")
print(f"   Plays processed: {processed:,}")
print(f"   Errors encountered: {errors}")
print(f"   Total player-plays: {len(df_rti):,}")
print(f"   Unique players: {df_rti['nfl_id'].nunique()}")
print(f"   Unique positions: {df_rti['player_position'].nunique()}")
print(f"\nğŸ“ˆ RTI Statistics:")
print(f"   Mean: {df_rti['reaction_time_index'].mean():.2f}")
print(f"   Median: {df_rti['reaction_time_index'].median():.2f}")
print(f"   Std Dev: {df_rti['reaction_time_index'].std():.2f}")
print(f"   Min: {df_rti['reaction_time_index'].min():.2f}")
print(f"   Max: {df_rti['reaction_time_index'].max():.2f}")

# Display sample
print(f"\nğŸ“‹ Sample Results:")
display(df_rti.head(10))


print("ğŸ�† GENERATING PLAYER RANKINGS")
print("=" * 70)

# Aggregate by player
player_rankings = df_rti.groupby(['nfl_id', 'player_name', 'player_position']).agg({
    'reaction_time_index': ['mean', 'std', 'min', 'max', 'count'],
    'detection_score': 'mean',
    'efficiency_score': 'mean',
    'timing_score': 'mean'
}).reset_index()

player_rankings.columns = [
    'nfl_id', 'player_name', 'position', 
    'avg_rti', 'std_rti', 'min_rti', 'max_rti', 'plays',
    'avg_detection', 'avg_efficiency', 'avg_timing'
]

# Filter players with at least 5 plays for reliability
player_rankings = player_rankings[player_rankings['plays'] >= 5]
player_rankings = player_rankings.sort_values('avg_rti', ascending=False).reset_index(drop=True)

print(f"\nğŸ“Š Players analyzed (â‰¥5 plays): {len(player_rankings)}")
print(f"\nğŸ¥‡ TOP 20 DEFENSIVE PLAYERS BY RTI:")
print("=" * 90)
print(f"{'Rank':<6} {'Player':<25} {'Pos':<5} {'Avg RTI':<9} {'Â±SD':<8} {'Plays':<7} {'Det':<6} {'Eff':<6} {'Tim':<6}")
print("=" * 90)

for idx, row in player_rankings.head(20).iterrows():
    print(f"{idx+1:<6} {row['player_name'][:24]:<25} {row['position']:<5} "
          f"{row['avg_rti']:>7.2f}   {row['std_rti']:>6.2f}   {row['plays']:>5.0f}   "
          f"{row['avg_detection']:>5.3f}  {row['avg_efficiency']:>5.3f}  {row['avg_timing']:>5.3f}")

# High-volume performers (elite consistency)
print(f"\n\nğŸ“Š TOP 10 HIGH-VOLUME PERFORMERS (â‰¥50 plays):")
print("=" * 90)
high_volume = player_rankings[player_rankings['plays'] >= 50].head(10)

for idx, row in high_volume.iterrows():
    rank = player_rankings[player_rankings['plays'] >= 50].index.get_loc(idx) + 1
    print(f"{rank:<6} {row['player_name'][:24]:<25} {row['position']:<5} "
          f"{row['avg_rti']:>7.2f}   {row['plays']:>5.0f} plays   "
          f"CV: {row['std_rti']/row['avg_rti']:.3f}")

print(f"\nğŸ’¾ Player rankings data frame created: {len(player_rankings)} players")


# Create RTI distribution visualization with percentiles
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram with color-coded performance tiers
ax1 = axes[0]
n, bins, patches_hist = ax1.hist(df_rti['reaction_time_index'], bins=50, 
                                  edgecolor='black', alpha=0.7)

# Color gradient based on performance
for i, patch in enumerate(patches_hist):
    if bins[i] < 40:
        patch.set_facecolor(NFL_RED)  # Poor
    elif bins[i] < 55:
        patch.set_facecolor(NFL_GOLD)  # Average
    else:
        patch.set_facecolor(NFL_GREEN)  # Elite

# Add percentile lines
percentiles = [25, 50, 75, 90]
for p in percentiles:
    val = np.percentile(df_rti['reaction_time_index'], p)
    ax1.axvline(val, color='darkblue', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(val, ax1.get_ylim()[1] * 0.95, f'{p}th\n{val:.1f}', 
             ha='center', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax1.set_xlabel('Reaction Time Index', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('RTI Distribution Across All Player-Plays', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add stats box
stats_text = f"""Statistics:
Mean: {df_rti['reaction_time_index'].mean():.2f}
Median: {df_rti['reaction_time_index'].median():.2f}
Std: {df_rti['reaction_time_index'].std():.2f}
N = {len(df_rti):,}"""
ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Component distributions
ax2 = axes[1]
components_data = df_rti[['detection_score', 'efficiency_score', 'timing_score']]
components_data.columns = ['Detection', 'Efficiency', 'Timing']

bp = ax2.boxplot([components_data['Detection'], components_data['Efficiency'], 
                    components_data['Timing']], 
                   labels=['Detection\n(40%)', 'Efficiency\n(30%)', 'Timing\n(30%)'],
                   patch_artist=True, widths=0.5)

colors = [NFL_BLUE, NFL_GREEN, NFL_GOLD]
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

for median in bp['medians']:
    median.set(linewidth=2, color='red')

ax2.set_ylabel('Score (0-1)', fontsize=12, fontweight='bold')
ax2.set_title('RTI Component Score Distributions', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 1.05)

plt.tight_layout()
plt.show()

print("âœ… RTI distribution visualization created!")


print("ğŸ“Š POSITION ANALYSIS")
print("=" * 70)

# Calculate position statistics
position_stats = df_rti.groupby('player_position').agg({
    'reaction_time_index': ['mean', 'std', 'count'],
    'detection_score': 'mean',
    'efficiency_score': 'mean',
    'timing_score': 'mean'
}).reset_index()
position_stats.columns = ['position', 'mean_rti', 'std_rti', 'count', 
                          'mean_detection', 'mean_efficiency', 'mean_timing']

# Filter positions with at least 1000 plays
position_stats = position_stats[position_stats['count'] >= 1000]
position_stats = position_stats.sort_values('mean_rti', ascending=False)

print(f"\nğŸ“‹ Position Statistics (min 1000 plays):")
print(f"{'Position':<10} {'Mean RTI':<12} {'Std':<8} {'Count':<10} {'Detection':<12} {'Efficiency':<12} {'Timing':<10}")
print("=" * 90)
for _, row in position_stats.iterrows():
    print(f"{row['position']:<10} {row['mean_rti']:>8.2f}     {row['std_rti']:>6.2f}   {row['count']:>7.0f}    "
          f"{row['mean_detection']:>8.3f}     {row['mean_efficiency']:>8.3f}      {row['mean_timing']:>6.3f}")

# ANOVA Test
print(f"\nğŸ”¬ ONE-WAY ANOVA TEST:")
print("=" * 70)
position_order = position_stats['position'].tolist()
position_groups = [df_rti[df_rti['player_position'] == pos]['reaction_time_index'].values 
                   for pos in position_order]
f_stat, p_value = stats.f_oneway(*position_groups)

print(f"  F-statistic: {f_stat:.2f}")
print(f"  p-value: {p_value:.2e}")
print(f"  Significant? {'YES - Positions differ significantly (p < 0.001)' if p_value < 0.001 else 'NO'}")

# Effect size (eta-squared)
grand_mean = df_rti['reaction_time_index'].mean()
ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in position_groups)
ss_total = sum((df_rti['reaction_time_index'] - grand_mean)**2)
eta_squared = ss_between / ss_total
print(f"  Effect size (Î·Â²): {eta_squared:.4f} ({'Large' if eta_squared > 0.14 else 'Medium' if eta_squared > 0.06 else 'Small'} effect)")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart
ax1 = axes[0]
colors_pos = [NFL_GREEN if x > 54 else NFL_GOLD if x > 52 else NFL_BLUE 
              for x in position_stats['mean_rti']]
bars = ax1.bar(range(len(position_stats)), position_stats['mean_rti'], 
               color=colors_pos, edgecolor='black', linewidth=1.5, alpha=0.8)

# Add values and sample sizes
for i, (idx, row) in enumerate(position_stats.iterrows()):
    ax1.text(i, row['mean_rti'] + 0.5, f"{row['mean_rti']:.1f}", 
             ha='center', fontsize=10, fontweight='bold')
    ax1.text(i, 46, f"n={int(row['count']):,}", 
             ha='center', fontsize=8, rotation=90, va='bottom')

ax1.axhline(df_rti['reaction_time_index'].mean(), color=NFL_RED, 
            linestyle='--', linewidth=2, label=f'Overall Mean ({df_rti["reaction_time_index"].mean():.2f})')
ax1.set_xticks(range(len(position_stats)))
ax1.set_xticklabels(position_stats['position'], fontsize=11, fontweight='bold')
ax1.set_ylabel('Average RTI', fontsize=12, fontweight='bold')
ax1.set_title(f'RTI by Position (ANOVA: F={f_stat:.2f}, p<0.001)', fontsize=14, fontweight='bold')
ax1.set_ylim(46, 60)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Component comparison by position
ax2 = axes[1]
x_pos = np.arange(len(position_stats))
width = 0.25

bars1 = ax2.bar(x_pos - width, position_stats['mean_detection'], width, 
                label='Detection (40%)', color=NFL_BLUE, alpha=0.7, edgecolor='black')
bars2 = ax2.bar(x_pos, position_stats['mean_efficiency'], width, 
                label='Efficiency (30%)', color=NFL_GREEN, alpha=0.7, edgecolor='black')
bars3 = ax2.bar(x_pos + width, position_stats['mean_timing'], width, 
                label='Timing (30%)', color=NFL_GOLD, alpha=0.7, edgecolor='black')

ax2.set_ylabel('Average Score', fontsize=12, fontweight='bold')
ax2.set_title('RTI Components by Position', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(position_stats['position'], fontsize=10, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 1.0)

plt.tight_layout()
plt.show()

print("\nâœ… Position analysis complete!")


print("ğŸ”¬ STATISTICAL VALIDATION")
print("=" * 70)

# 1. Confidence Intervals
def calculate_ci(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, ci

rti_mean, rti_ci = calculate_ci(df_rti['reaction_time_index'])
print(f"\nğŸ“Š Overall RTI with 95% Confidence Interval:")
print(f"   Mean: {rti_mean:.2f}")
print(f"   95% CI: [{rti_mean - rti_ci:.2f}, {rti_mean + rti_ci:.2f}]")
print(f"   Interpretation: We are 95% confident the true mean RTI is between {rti_mean - rti_ci:.2f} and {rti_mean + rti_ci:.2f}")

# 2. Component Correlations
print(f"\n\nğŸ“ˆ Component Correlations with Overall RTI:")
print("=" * 70)

components = ['detection_score', 'efficiency_score', 'timing_score']
component_names = ['Detection', 'Efficiency', 'Timing']

for comp, name in zip(components, component_names):
    r, p = stats.pearsonr(df_rti[comp], df_rti['reaction_time_index'])
    r_squared = r ** 2
    print(f"\n{name}:")
    print(f"  Correlation (r): {r:.3f}")
    print(f"  Variance explained (rÂ²): {r_squared:.3f} ({r_squared*100:.1f}%)")
    print(f"  p-value: {p:.2e} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")
    print(f"  Interpretation: {name} explains {r_squared*100:.1f}% of RTI variance")

# Visualization: Correlation matrix
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter plots
ax1 = axes[0]
colors_scatter = plt.cm.RdYlGn((df_rti['reaction_time_index'] - df_rti['reaction_time_index'].min()) / 
                                (df_rti['reaction_time_index'].max() - df_rti['reaction_time_index'].min()))

sample_size = min(5000, len(df_rti))
sample_idx = np.random.choice(len(df_rti), sample_size, replace=False)
df_sample = df_rti.iloc[sample_idx]

ax1.scatter(df_sample['detection_score'], df_sample['reaction_time_index'], 
           alpha=0.3, s=10, c=NFL_BLUE, label='Detection')
ax1.scatter(df_sample['efficiency_score'], df_sample['reaction_time_index'], 
           alpha=0.3, s=10, c=NFL_GREEN, label='Efficiency')
ax1.scatter(df_sample['timing_score'], df_sample['reaction_time_index'], 
           alpha=0.3, s=10, c=NFL_GOLD, label='Timing')

ax1.set_xlabel('Component Score (0-1)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Reaction Time Index', fontsize=12, fontweight='bold')
ax1.set_title(f'Component Scores vs RTI (sample n={sample_size:,})', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# Correlation heatmap
ax2 = axes[1]
corr_data = df_rti[components].corr()
im = ax2.imshow(corr_data, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

for i in range(len(components)):
    for j in range(len(components)):
        text = ax2.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", 
                       fontsize=14, fontweight='bold')

ax2.set_xticks(range(len(component_names)))
ax2.set_yticks(range(len(component_names)))
ax2.set_xticklabels(component_names, fontsize=11)
ax2.set_yticklabels(component_names, fontsize=11)
ax2.set_title('Component Inter-Correlations', fontsize=14, fontweight='bold')

cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Correlation Coefficient', fontsize=10)

plt.tight_layout()
plt.show()

# 3. Player Consistency Analysis
print(f"\n\nğŸ“Š Player Consistency Analysis:")
print("=" * 70)

# Calculate coefficient of variation for players with â‰¥10 plays
consistent_players = player_rankings[player_rankings['plays'] >= 10].copy()
consistent_players['cv'] = consistent_players['std_rti'] / consistent_players['avg_rti']

print(f"\nPlayers with â‰¥10 plays: {len(consistent_players)}")
print(f"Average CV (Coefficient of Variation): {consistent_players['cv'].mean():.3f}")
print(f"Median CV: {consistent_players['cv'].median():.3f}")

print(f"\nğŸ�¯ Most Consistent Players (Lowest CV, â‰¥20 plays):")
most_consistent = consistent_players[consistent_players['plays'] >= 20].nsmallest(5, 'cv')
for _, p in most_consistent.iterrows():
    print(f"  {p['player_name'][:30]:<30} RTI: {p['avg_rti']:5.2f} Â± {p['std_rti']:4.2f}  CV: {p['cv']:.3f}  ({int(p['plays'])} plays)")

print(f"\nâš¡ Most Variable Players (Highest CV, â‰¥20 plays):")
most_variable = consistent_players[consistent_players['plays'] >= 20].nlargest(5, 'cv')
for _, p in most_variable.iterrows():
    print(f"  {p['player_name'][:30]:<30} RTI: {p['avg_rti']:5.2f} Â± {p['std_rti']:4.2f}  CV: {p['cv']:.3f}  ({int(p['plays'])} plays)")

print("\nâœ… Statistical validation complete!")


print("ğŸ�¬ CASE STUDY EXAMPLES & FIELD VISUALIZATIONS")
print("=" * 70)

# ============================================================================
# Find Elite Detection Example
# ============================================================================
print("\nğŸ”� 1. ELITE DETECTION TIME")
print("-" * 70)

elite_detection = df_rti[
    (df_rti['detection_score'] > 0.85) &
    (df_rti['efficiency_score'] > 0.9) &
    (df_rti['reaction_time_index'] > 70)
].sort_values('detection_score', ascending=False)

print(f"Found {len(elite_detection)} plays with elite detection")

if len(elite_detection) > 0:
    # Group by play to find plays with multiple elite reactors
    elite_plays = elite_detection.groupby(['game_id', 'play_id']).agg({
        'nfl_id': 'count',
        'detection_score': 'mean',
        'reaction_time_index': 'mean'
    }).reset_index()
    elite_plays.columns = ['game_id', 'play_id', 'num_defenders', 'avg_detection', 'avg_rti']
    elite_plays = elite_plays[elite_plays['num_defenders'] >= 3].sort_values('avg_detection', ascending=False)
    
    if len(elite_plays) > 0:
        best_play = elite_plays.iloc[0]
        print(f"âœ… Best play: Game {best_play['game_id']}, Play {best_play['play_id']}")
        print(f"   {int(best_play['num_defenders'])} defenders, Avg Detection: {best_play['avg_detection']:.3f}")
        
        # Get player details
        play_details = elite_detection[
            (elite_detection['game_id'] == best_play['game_id']) &
            (elite_detection['play_id'] == best_play['play_id'])
        ].nlargest(3, 'reaction_time_index')
        
        for _, p in play_details.iterrows():
            print(f"   â€¢ {p['player_name']} ({p['player_position']}): RTI {p['reaction_time_index']:.1f}, "
                  f"Detection {p['detection_score']:.3f}")

# ============================================================================
# Find Perfect Path Efficiency Example
# ============================================================================
print("\n\nğŸ�¯ 2. PERFECT PATH EFFICIENCY")
print("-" * 70)

perfect_efficiency = df_rti[
    (df_rti['efficiency_score'] >= 0.99) &
    (df_rti['detection_score'] > 0.5) &
    (df_rti['reaction_time_index'] > 65)
].sort_values('efficiency_score', ascending=False)

print(f"Found {len(perfect_efficiency)} plays with perfect efficiency")

if len(perfect_efficiency) > 0:
    eff_plays = perfect_efficiency.groupby(['game_id', 'play_id']).agg({
        'nfl_id': 'count',
        'efficiency_score': 'mean',
        'reaction_time_index': 'mean'
    }).reset_index()
    eff_plays.columns = ['game_id', 'play_id', 'num_defenders', 'avg_efficiency', 'avg_rti']
    eff_plays = eff_plays[eff_plays['num_defenders'] >= 3].sort_values('avg_efficiency', ascending=False)
    
    if len(eff_plays) > 0:
        best_eff = eff_plays.iloc[0]
        print(f"âœ… Best play: Game {best_eff['game_id']}, Play {best_eff['play_id']}")
        print(f"   {int(best_eff['num_defenders'])} defenders, Avg Efficiency: {best_eff['avg_efficiency']:.3f}")

# ============================================================================
# Find Excellent Timing Example
# ============================================================================
print("\n\nâ�±ï¸� 3. EXCELLENT ARRIVAL TIMING")
print("-" * 70)

excellent_timing = df_rti[
    (df_rti['timing_score'] > 0.6) &
    (df_rti['detection_score'] > 0.4) &
    (df_rti['min_distance_to_ball'] < 5) &
    (df_rti['reaction_time_index'] > 65)
].sort_values('timing_score', ascending=False)

print(f"Found {len(excellent_timing)} plays with excellent timing")

if len(excellent_timing) > 0:
    timing_plays = excellent_timing.groupby(['game_id', 'play_id']).agg({
        'nfl_id': 'count',
        'timing_score': 'mean',
        'reaction_time_index': 'mean'
    }).reset_index()
    timing_plays.columns = ['game_id', 'play_id', 'num_defenders', 'avg_timing', 'avg_rti']
    timing_plays = timing_plays[timing_plays['num_defenders'] >= 3].sort_values('avg_timing', ascending=False)
    
    if len(timing_plays) > 0:
        best_timing = timing_plays.iloc[0]
        print(f"âœ… Best play: Game {best_timing['game_id']}, Play {best_timing['play_id']}")
        print(f"   {int(best_timing['num_defenders'])} defenders, Avg Timing: {best_timing['avg_timing']:.3f}")

# ============================================================================
# Find Poor Reaction (Contrast)
# ============================================================================
print("\n\nâš ï¸� 4. POOR REACTION (CONTRAST)")
print("-" * 70)

poor_reaction = df_rti[
    (df_rti['reaction_time_index'] < 35) &
    (df_rti['detection_score'] < 0.3)
].sort_values('reaction_time_index', ascending=True)

print(f"Found {len(poor_reaction)} plays with poor reactions")

if len(poor_reaction) > 0:
    poor_plays = poor_reaction.groupby(['game_id', 'play_id']).agg({
        'nfl_id': 'count',
        'detection_score': 'mean',
        'reaction_time_index': 'mean'
    }).reset_index()
    poor_plays.columns = ['game_id', 'play_id', 'num_defenders', 'avg_detection', 'avg_rti']
    poor_plays = poor_plays[poor_plays['num_defenders'] >= 3].sort_values('avg_rti', ascending=True)
    
    if len(poor_plays) > 0:
        worst_play = poor_plays.iloc[0]
        print(f"â�Œ Worst play: Game {worst_play['game_id']}, Play {worst_play['play_id']}")
        print(f"   {int(worst_play['num_defenders'])} defenders, Avg RTI: {worst_play['avg_rti']:.2f}")

# ============================================================================
# Comparison Visualization
# ============================================================================

# Create comparison visualization if we have examples
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Case Study Examples: Elite vs Poor Defensive Reactions', 
             fontsize=16, fontweight='bold', y=0.995)

# Case 1: Elite Detection Distribution
ax1 = axes[0, 0]
if len(elite_detection) > 0:
    ax1.hist(elite_detection['detection_score'], bins=20, color=NFL_GREEN, 
            alpha=0.7, edgecolor='black')
    ax1.axvline(elite_detection['detection_score'].mean(), color=NFL_BLUE, 
               linestyle='--', linewidth=2, label=f'Mean: {elite_detection["detection_score"].mean():.3f}')
    ax1.set_xlabel('Detection Score', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Elite Detection Time Examples (n={})'.format(len(elite_detection)), fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

# Case 2: Perfect Efficiency Distribution
ax2 = axes[0, 1]
if len(perfect_efficiency) > 0:
    ax2.hist(perfect_efficiency['efficiency_score'], bins=20, color=NFL_BLUE, 
            alpha=0.7, edgecolor='black')
    ax2.axvline(perfect_efficiency['efficiency_score'].mean(), color=NFL_RED, 
               linestyle='--', linewidth=2, label=f'Mean: {perfect_efficiency["efficiency_score"].mean():.3f}')
    ax2.set_xlabel('Path Efficiency Score', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Perfect Path Efficiency Examples (n={})'.format(len(perfect_efficiency)), fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

# Case 3: Timing Distribution
ax3 = axes[1, 0]
if len(excellent_timing) > 0:
    ax3.hist(excellent_timing['timing_score'], bins=20, color=NFL_GOLD, 
            alpha=0.7, edgecolor='black')
    ax3.axvline(excellent_timing['timing_score'].mean(), color=NFL_RED, 
               linestyle='--', linewidth=2, label=f'Mean: {excellent_timing["timing_score"].mean():.3f}')
    ax3.set_xlabel('Arrival Timing Score', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Excellent Arrival Timing Examples (n={})'.format(len(excellent_timing)), fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

# Case 4: Elite vs Poor Comparison
ax4 = axes[1, 1]
if len(elite_detection) > 0 and len(poor_reaction) > 0:
    # Sample for comparison
    elite_sample = elite_detection[['detection_score', 'efficiency_score', 'timing_score', 'reaction_time_index']].mean()
    poor_sample = poor_reaction[['detection_score', 'efficiency_score', 'timing_score', 'reaction_time_index']].mean()
    
    categories = ['Detection', 'Efficiency', 'Timing', 'RTI\n(scaled)']
    elite_vals = [elite_sample['detection_score'], elite_sample['efficiency_score'], 
                  elite_sample['timing_score'], elite_sample['reaction_time_index']/100]
    poor_vals = [poor_sample['detection_score'], poor_sample['efficiency_score'], 
                 poor_sample['timing_score'], poor_sample['reaction_time_index']/100]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, elite_vals, width, label='Elite Reactions', 
                   color=NFL_GREEN, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax4.bar(x + width/2, poor_vals, width, label='Poor Reactions', 
                   color=NFL_RED, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax4.set_ylabel('Average Score', fontweight='bold')
    ax4.set_title('Elite vs Poor: Component Comparison', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.set_ylim(0, 1.1)
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.show()

print("\nâœ… Case study analysis and visualization complete!")


print("ğŸ�ˆ NFL FIELD CASE STUDY VISUALIZATIONS")
print("=" * 70)
print("Generating 5 detailed play diagrams with player tracking...\n")

# NFL field dimensions
FIELD_LENGTH = 120  # Including endzones
FIELD_WIDTH = 53.3

def draw_nfl_field(ax):
    """Draw NFL field with yard lines"""
    from matplotlib.patches import FancyBboxPatch
    
    # Field background
    field = FancyBboxPatch((0, 0), FIELD_LENGTH, FIELD_WIDTH,
                           boxstyle="round,pad=0", 
                           facecolor=FIELD_COLOR, 
                           edgecolor='white', 
                           linewidth=3)
    ax.add_patch(field)
    
    # Yard lines (every 10 yards)
    for yard in range(10, 111, 10):
        ax.plot([yard, yard], [0, FIELD_WIDTH], color='white', linewidth=1, alpha=0.5)
    
    # 5-yard lines (lighter)
    for yard in range(5, 116, 5):
        if yard % 10 != 0:
            ax.plot([yard, yard], [0, FIELD_WIDTH], color='white', linewidth=0.5, alpha=0.3)
    
    # Endzones (darker shade)
    ax.axvspan(0, 10, alpha=0.3, color='darkgreen')
    ax.axvspan(110, 120, alpha=0.3, color='darkgreen')
    
    # Goal lines (thicker)
    ax.plot([10, 10], [0, FIELD_WIDTH], color=NFL_GOLD, linewidth=3)
    ax.plot([110, 110], [0, FIELD_WIDTH], color=NFL_GOLD, linewidth=3)
    
    # 50-yard line (thicker)
    ax.plot([60, 60], [0, FIELD_WIDTH], color='white', linewidth=2, alpha=0.8)
    
    # Add yard numbers
    for yard in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
        display_yard = yard - 10 if yard <= 60 else 120 - yard
        ax.text(yard, FIELD_WIDTH/2, str(display_yard), 
                ha='center', va='center', fontsize=20, 
                color='white', fontweight='bold', alpha=0.3)
    
    ax.set_xlim(-5, 125)
    ax.set_ylim(-5, FIELD_WIDTH + 5)
    ax.set_aspect('equal')
    ax.axis('off')

def plot_player_trajectory(ax, tracking_data, nfl_id, color, label, show_label=True):
    """Plot player movement path with start/end markers"""
    player_data = tracking_data[tracking_data['nfl_id'] == nfl_id].sort_values('frame_id')
    
    if len(player_data) == 0:
        return
    
    xs = player_data['x'].values
    ys = player_data['y'].values
    
    # Plot trajectory line
    ax.plot(xs, ys, color=color, linewidth=2, alpha=0.6, linestyle='--')
    
    # Start position (smaller circle)
    ax.scatter(xs[0], ys[0], s=250, c=color, edgecolors='white', 
               linewidths=2, zorder=5, marker='o', alpha=0.8)
    
    # End position (larger circle)
    ax.scatter(xs[-1], ys[-1], s=350, c=color, edgecolors='white', 
               linewidths=3, zorder=6, marker='o')
    
    # Add label at end position
    if show_label and len(label) > 0:
        ax.text(xs[-1], ys[-1] + 2.5, label, 
                ha='center', va='bottom', fontsize=8, 
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.9))

def visualize_case_study(case_num, case_name, play_selection_criteria):
    """Generate field visualization for a specific case study"""
    
    print(f"\n{'='*70}")
    print(f"ğŸ“Š CASE STUDY {case_num}: {case_name}")
    print(f"{'='*70}")
    
    # Find plays matching criteria
    matching_plays = df_rti[play_selection_criteria].copy()
    
    if len(matching_plays) == 0:
        print(f"âš ï¸�  No plays found matching criteria")
        return
    
    # Group by play to find best example
    play_groups = matching_plays.groupby(['game_id', 'play_id']).agg({
        'nfl_id': 'count',
        'reaction_time_index': 'mean',
        'detection_score': 'mean',
        'efficiency_score': 'mean',
        'timing_score': 'mean'
    }).reset_index()
    play_groups.columns = ['game_id', 'play_id', 'num_defenders', 'avg_rti', 
                           'avg_detection', 'avg_efficiency', 'avg_timing']
    play_groups = play_groups[play_groups['num_defenders'] >= 3]
    
    if len(play_groups) == 0:
        print(f"âš ï¸�  No plays with sufficient defenders")
        return
    
    # Sort based on case type
    if 'poor' in case_name.lower() or 'contrast' in case_name.lower():
        play_groups = play_groups.sort_values('avg_rti', ascending=True)
    else:
        play_groups = play_groups.sort_values('avg_rti', ascending=False)
    
    selected_play = play_groups.iloc[0]
    game_id = selected_play['game_id']
    play_id = selected_play['play_id']
    
    print(f"Selected: Game {game_id}, Play {play_id}")
    print(f"  {int(selected_play['num_defenders'])} defenders | Avg RTI: {selected_play['avg_rti']:.2f}")
    
    # Get tracking data for this play
    play_tracking = df_tracking[
        (df_tracking['game_id'] == game_id) &
        (df_tracking['play_id'] == play_id)
    ].copy()
    
    if len(play_tracking) == 0:
        print(f"âš ï¸�  No tracking data found")
        return
    
    # Get ball landing coordinates
    ball_land_x = play_tracking['ball_land_x'].iloc[0]
    ball_land_y = play_tracking['ball_land_y'].iloc[0]
    
    # Get player metrics for this play
    play_metrics = df_rti[
        (df_rti['game_id'] == game_id) &
        (df_rti['play_id'] == play_id)
    ].copy()
    
    # Get top 3 defenders (or bottom 3 for poor reaction)
    if 'poor' in case_name.lower() or 'contrast' in case_name.lower():
        top_defenders = play_metrics.nsmallest(3, 'reaction_time_index')
    else:
        top_defenders = play_metrics.nlargest(3, 'reaction_time_index')
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor(FIELD_COLOR)
    
    draw_nfl_field(ax)
    
    # Plot ball landing spot
    ax.scatter(ball_land_x, ball_land_y, s=800, c=NFL_GOLD, marker='X', 
              edgecolors='white', linewidths=3, zorder=10, label='Ball Landing')
    
    # Plot offensive players (lighter trails)
    offense = play_tracking[play_tracking['player_side'] == 'Offense']
    offense_ids = offense['nfl_id'].unique()
    
    for off_id in offense_ids[:8]:  # Limit to 8 for clarity
        player_data = offense[offense['nfl_id'] == off_id].sort_values('frame_id')
        if len(player_data) > 0:
            xs = player_data['x'].values
            ys = player_data['y'].values
            ax.plot(xs, ys, color=NFL_RED, linewidth=1, alpha=0.15, linestyle=':')
            ax.scatter(xs[-1], ys[-1], s=120, c=NFL_RED, alpha=0.3, zorder=3, 
                      edgecolors='white', linewidths=1)
    
    # Plot top 3 defenders with full trajectories
    defense = play_tracking[play_tracking['player_side'] == 'Defense']
    colors = [NFL_BLUE, NFL_GREEN, 'purple']
    
    for idx, (_, defender) in enumerate(top_defenders.iterrows()):
        nfl_id = defender['nfl_id']
        name = defender['player_name'].split()[-1] if ' ' in defender['player_name'] else defender['player_name']
        rti = defender['reaction_time_index']
        pos = defender['player_position']
        
        label = f"{name} ({pos})\nRTI: {rti:.1f}"
        plot_player_trajectory(ax, defense, nfl_id, colors[idx], label, show_label=True)
    
    # Plot other defenders (lighter)
    other_defender_ids = defense[~defense['nfl_id'].isin(top_defenders['nfl_id'])]['nfl_id'].unique()
    
    for def_id in other_defender_ids[:8]:  # Limit for clarity
        plot_player_trajectory(ax, defense, def_id, NFL_BLUE, '', show_label=False)
    
    # Add title with case-specific styling
    if 'poor' in case_name.lower() or 'contrast' in case_name.lower():
        title_color = NFL_RED
    else:
        title_color = NFL_GREEN
    
    ax.set_title(f"{case_name}\nGame {game_id}, Play {play_id} | Avg RTI: {selected_play['avg_rti']:.1f}", 
                fontsize=14, fontweight='bold', color='white', pad=15,
                bbox=dict(boxstyle='round,pad=0.6', facecolor=title_color, alpha=0.9))
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', markerfacecolor=NFL_BLUE, markeredgecolor='white',
                  markersize=10, label='Top Defenders', linestyle='--', linewidth=2, color=NFL_BLUE),
        plt.Line2D([0], [0], marker='o', markerfacecolor=NFL_RED, markeredgecolor='white',
                  markersize=8, label='Offense', linestyle=':', linewidth=1, color=NFL_RED, alpha=0.5),
        plt.Line2D([0], [0], marker='X', markerfacecolor=NFL_GOLD, markeredgecolor='white',
                  markersize=12, label='Ball Landing', linestyle='', color='none')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
             facecolor='white', edgecolor='black', framealpha=0.9)
    
    plt.tight_layout()
    plt.show()
    
    print(f"âœ… Case Study {case_num} visualization complete!")

# ============================================================================
# CASE STUDY 1: ELITE DETECTION TIME
# ============================================================================
visualize_case_study(
    case_num=1,
    case_name="Elite Detection Time",
    play_selection_criteria=(
        (df_rti['detection_score'] > 0.85) &
        (df_rti['efficiency_score'] > 0.9) &
        (df_rti['reaction_time_index'] > 70)
    )
)

# ============================================================================
# CASE STUDY 2: PERFECT PATH EFFICIENCY
# ============================================================================
visualize_case_study(
    case_num=2,
    case_name="Perfect Path Efficiency",
    play_selection_criteria=(
        (df_rti['efficiency_score'] >= 0.99) &
        (df_rti['detection_score'] > 0.5) &
        (df_rti['reaction_time_index'] > 65)
    )
)

# ============================================================================
# CASE STUDY 3: EXCELLENT ARRIVAL TIMING
# ============================================================================
visualize_case_study(
    case_num=3,
    case_name="Excellent Arrival Timing",
    play_selection_criteria=(
        (df_rti['timing_score'] > 0.6) &
        (df_rti['detection_score'] > 0.4) &
        (df_rti['min_distance_to_ball'] < 5) &
        (df_rti['reaction_time_index'] > 65)
    )
)

# ============================================================================
# CASE STUDY 4: POOR REACTION (CONTRAST)
# ============================================================================
visualize_case_study(
    case_num=4,
    case_name="Poor Reaction (Contrast)",
    play_selection_criteria=(
        (df_rti['reaction_time_index'] < 35) &
        (df_rti['detection_score'] < 0.3)
    )
)

# ============================================================================
# CASE STUDY 5: BALANCED EXCELLENCE
# ============================================================================
# Calculate balance score (lower std dev = more balanced)
df_rti_temp = df_rti.copy()
df_rti_temp['balance_score'] = df_rti_temp.apply(
    lambda row: np.std([row['detection_score'], row['efficiency_score'], row['timing_score']]), 
    axis=1
)

visualize_case_study(
    case_num=5,
    case_name="Balanced Excellence",
    play_selection_criteria=(
        (df_rti_temp['detection_score'] > 0.7) &
        (df_rti_temp['efficiency_score'] > 0.95) &
        (df_rti_temp['timing_score'] > 0.4) &
        (df_rti_temp['reaction_time_index'] > 75)
    )
)

print("\n" + "=" * 70)
print("âœ… ALL 5 CASE STUDY FIELD VISUALIZATIONS COMPLETE!")
print("=" * 70)
print("""
Generated visualizations:
  1. Elite Detection Time - Fast reaction to ball flight
  2. Perfect Path Efficiency - Optimal movement paths
  3. Excellent Arrival Timing - Perfect ball arrival timing
  4. Poor Reaction (Contrast) - Slow/poor reactions
  5. Balanced Excellence - Strong across all components
""")


print("ğŸ�† TOP PLAYERS RANKINGS VISUALIZATIONS")
print("=" * 70)

# Create comprehensive top players visualization
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

# 1. Top 20 Players by RTI (Horizontal Bar Chart)
ax1 = fig.add_subplot(gs[0, :])
top20 = player_rankings.head(20)

# Create color gradient based on RTI
colors_top20 = plt.cm.RdYlGn((top20['avg_rti'] - top20['avg_rti'].min()) / 
                             (top20['avg_rti'].max() - top20['avg_rti'].min()))

bars = ax1.barh(range(len(top20)), top20['avg_rti'], 
                color=colors_top20, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add player names and values
for i, (idx, row) in enumerate(top20.iterrows()):
    # Player name on the left
    ax1.text(-2, i, f"{i+1}. {row['player_name'][:25]} ({row['position']})", 
             va='center', ha='right', fontsize=10, fontweight='bold')
    
    # RTI value inside bar
    ax1.text(row['avg_rti'] - 2, i, f"{row['avg_rti']:.1f}", 
             va='center', ha='right', fontsize=10, fontweight='bold', color='white')
    
    # Play count outside bar
    ax1.text(row['avg_rti'] + 0.5, i, f"({int(row['plays'])} plays)", 
             va='center', ha='left', fontsize=8, style='italic')

ax1.set_yticks([])
ax1.set_xlabel('Reaction Time Index', fontsize=13, fontweight='bold')
ax1.set_title('Top 20 Players by RTI (min 5 plays)', fontsize=15, fontweight='bold', pad=15)
ax1.grid(axis='x', alpha=0.3)
ax1.set_xlim(-15, top20['avg_rti'].max() + 8)
ax1.axvline(df_rti['reaction_time_index'].mean(), color='red', linestyle='--', 
            linewidth=2, alpha=0.5, label=f'Overall Mean ({df_rti["reaction_time_index"].mean():.1f})')
ax1.legend(loc='lower right', fontsize=10)

# 2. Top 10 by Detection Speed
ax2 = fig.add_subplot(gs[1, 0])
top_detection = player_rankings.nlargest(10, 'avg_detection')

bars = ax2.bar(range(len(top_detection)), top_detection['avg_detection'], 
               color=NFL_BLUE, alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (idx, row) in enumerate(top_detection.iterrows()):
    ax2.text(i, row['avg_detection'] + 0.01, f"{row['avg_detection']:.3f}", 
             ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Truncate long names
    name = row['player_name'][:15] + '...' if len(row['player_name']) > 15 else row['player_name']
    ax2.text(i, 0.01, name, ha='center', va='bottom', fontsize=8, 
             rotation=45, style='italic')

ax2.set_xticks([])
ax2.set_ylabel('Detection Score', fontsize=11, fontweight='bold')
ax2.set_title('Top 10: Fastest Detection Time', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, top_detection['avg_detection'].max() + 0.05)

# 3. Top 10 by Path Efficiency
ax3 = fig.add_subplot(gs[1, 1])
top_efficiency = player_rankings.nlargest(10, 'avg_efficiency')

bars = ax3.bar(range(len(top_efficiency)), top_efficiency['avg_efficiency'], 
               color=NFL_GREEN, alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (idx, row) in enumerate(top_efficiency.iterrows()):
    ax3.text(i, row['avg_efficiency'] + 0.01, f"{row['avg_efficiency']:.3f}", 
             ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    name = row['player_name'][:15] + '...' if len(row['player_name']) > 15 else row['player_name']
    ax3.text(i, 0.01, name, ha='center', va='bottom', fontsize=8, 
             rotation=45, style='italic')

ax3.set_xticks([])
ax3.set_ylabel('Efficiency Score', fontsize=11, fontweight='bold')
ax3.set_title('Top 10: Best Path Efficiency', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, top_efficiency['avg_efficiency'].max() + 0.05)

# 4. Top 10 High-Volume Performers (â‰¥50 plays)
ax4 = fig.add_subplot(gs[2, :])
high_volume = player_rankings[player_rankings['plays'] >= 50].head(10)

if len(high_volume) > 0:
    # Create stacked bar showing component contributions
    x_pos = np.arange(len(high_volume))
    
    detection_contrib = high_volume['avg_detection'] * 0.40
    efficiency_contrib = high_volume['avg_efficiency'] * 0.30
    timing_contrib = high_volume['avg_timing'] * 0.30
    
    bars1 = ax4.bar(x_pos, detection_contrib, color=NFL_BLUE, 
                   alpha=0.8, edgecolor='black', linewidth=1, label='Detection (40%)')
    bars2 = ax4.bar(x_pos, efficiency_contrib, bottom=detection_contrib,
                   color=NFL_GREEN, alpha=0.8, edgecolor='black', linewidth=1, 
                   label='Efficiency (30%)')
    bars3 = ax4.bar(x_pos, timing_contrib, 
                   bottom=detection_contrib + efficiency_contrib,
                   color=NFL_GOLD, alpha=0.8, edgecolor='black', linewidth=1, 
                   label='Timing (30%)')
    
    # Add total RTI on top
    for i, (idx, row) in enumerate(high_volume.iterrows()):
        total_height = detection_contrib.iloc[i] + efficiency_contrib.iloc[i] + timing_contrib.iloc[i]
        ax4.text(i, total_height + 0.01, f"{row['avg_rti']:.1f}", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Player name below
        name = row['player_name'][:20] + '...' if len(row['player_name']) > 20 else row['player_name']
        ax4.text(i, -0.02, f"{name}\n({int(row['plays'])} plays)", 
                ha='center', va='top', fontsize=9, fontweight='bold')
    
    ax4.set_xticks([])
    ax4.set_ylabel('RTI Component Contributions', fontsize=12, fontweight='bold')
    ax4.set_title('Top 10 High-Volume Performers (â‰¥50 plays) - Component Breakdown', 
                 fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(-0.08, 0.7)
else:
    ax4.text(0.5, 0.5, 'Insufficient high-volume players\n(need â‰¥50 plays)', 
            ha='center', va='center', transform=ax4.transAxes, fontsize=14)
    ax4.axis('off')

fig.suptitle('Top Performers Analysis - Reaction Time Index', 
             fontsize=18, fontweight='bold', y=0.995)

plt.show()

print("\nâœ… Top players rankings visualization complete!")

# Print summary statistics
print(f"\nğŸ“Š Summary Statistics:")
print(f"{'='*70}")
print(f"Total Players Ranked: {len(player_rankings)}")
print(f"Players with â‰¥50 plays: {len(player_rankings[player_rankings['plays'] >= 50])}")
print(f"Players with â‰¥100 plays: {len(player_rankings[player_rankings['plays'] >= 100])}")
print(f"\nTop Performer: {player_rankings.iloc[0]['player_name']} (RTI: {player_rankings.iloc[0]['avg_rti']:.2f})")
print(f"Most Plays: {player_rankings.nlargest(1, 'plays').iloc[0]['player_name']} ({int(player_rankings.nlargest(1, 'plays').iloc[0]['plays'])} plays)")
print(f"Highest Detection: {player_rankings.nlargest(1, 'avg_detection').iloc[0]['player_name']} ({player_rankings.nlargest(1, 'avg_detection').iloc[0]['avg_detection']:.3f})")
print(f"Best Efficiency: {player_rankings.nlargest(1, 'avg_efficiency').iloc[0]['player_name']} ({player_rankings.nlargest(1, 'avg_efficiency').iloc[0]['avg_efficiency']:.3f})")


print("ğŸ“Š ADDITIONAL POSITION VISUALIZATIONS")
print("=" * 70)

# ============================================================================
# 1. Position Group Comparison (Front 7 vs Secondary)
# ============================================================================

# Categorize positions into groups
def categorize_position(pos):
    if pos in ['CB', 'SS', 'FS', 'DB', 'S']:
        return 'Secondary'
    elif pos in ['MLB', 'OLB', 'ILB', 'LB']:
        return 'Linebackers'
    elif pos in ['DE', 'DT', 'NT', 'DL']:
        return 'Defensive Line'
    else:
        return 'Other'

df_rti['position_group'] = df_rti['player_position'].apply(categorize_position)

# Calculate group statistics
group_stats = df_rti.groupby('position_group').agg({
    'reaction_time_index': ['mean', 'std', 'count'],
    'detection_score': 'mean',
    'efficiency_score': 'mean',
    'timing_score': 'mean'
}).reset_index()
group_stats.columns = ['group', 'mean_rti', 'std_rti', 'count', 
                       'mean_detection', 'mean_efficiency', 'mean_timing']
group_stats = group_stats[group_stats['count'] >= 500].sort_values('mean_rti', ascending=False)

print(f"\nğŸ�ˆ Position Group Statistics:")
print(f"{'Group':<20} {'Mean RTI':<12} {'Count':<10} {'Detection':<12} {'Efficiency':<12} {'Timing'}")
print("=" * 90)
for _, row in group_stats.iterrows():
    print(f"{row['group']:<20} {row['mean_rti']:>8.2f}     {row['count']:>7.0f}    "
          f"{row['mean_detection']:>8.3f}     {row['mean_efficiency']:>8.3f}      {row['mean_timing']:>6.3f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Position Group Analysis', fontsize=16, fontweight='bold')

# 1. Group RTI Comparison
ax1 = axes[0, 0]
bars = ax1.bar(range(len(group_stats)), group_stats['mean_rti'], 
               color=[NFL_BLUE, NFL_GREEN, NFL_RED, NFL_GOLD][:len(group_stats)],
               alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (idx, row) in enumerate(group_stats.iterrows()):
    ax1.text(i, row['mean_rti'] + 0.5, f"{row['mean_rti']:.1f}", 
             ha='center', fontsize=11, fontweight='bold')
    ax1.text(i, 47, f"n={int(row['count']):,}", 
             ha='center', fontsize=8, rotation=0)

ax1.axhline(df_rti['reaction_time_index'].mean(), color='red', 
            linestyle='--', linewidth=2, alpha=0.7, 
            label=f'Overall Mean ({df_rti["reaction_time_index"].mean():.1f})')
ax1.set_xticks(range(len(group_stats)))
ax1.set_xticklabels(group_stats['group'], fontsize=11, fontweight='bold')
ax1.set_ylabel('Average RTI', fontsize=12, fontweight='bold')
ax1.set_title('RTI by Position Group', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(47, 58)

# 2. Component Comparison by Group
ax2 = axes[0, 1]
x_pos = np.arange(len(group_stats))
width = 0.25

bars1 = ax2.bar(x_pos - width, group_stats['mean_detection'], width, 
                label='Detection', color=NFL_BLUE, alpha=0.7, edgecolor='black')
bars2 = ax2.bar(x_pos, group_stats['mean_efficiency'], width, 
                label='Efficiency', color=NFL_GREEN, alpha=0.7, edgecolor='black')
bars3 = ax2.bar(x_pos + width, group_stats['mean_timing'], width, 
                label='Timing', color=NFL_GOLD, alpha=0.7, edgecolor='black')

ax2.set_ylabel('Average Score', fontsize=12, fontweight='bold')
ax2.set_title('Component Scores by Position Group', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(group_stats['group'], fontsize=10)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 1.0)

# 3. Secondary Positions Detailed Breakdown
ax3 = axes[1, 0]
secondary_positions = ['CB', 'SS', 'FS', 'S', 'DB']
secondary_data = df_rti[df_rti['player_position'].isin(secondary_positions)]

if len(secondary_data) > 0:
    sec_stats = secondary_data.groupby('player_position').agg({
        'reaction_time_index': ['mean', 'count']
    }).reset_index()
    sec_stats.columns = ['position', 'mean_rti', 'count']
    sec_stats = sec_stats[sec_stats['count'] >= 100].sort_values('mean_rti', ascending=False)
    
    colors_sec = plt.cm.viridis(np.linspace(0.3, 0.9, len(sec_stats)))
    bars = ax3.barh(range(len(sec_stats)), sec_stats['mean_rti'], 
                    color=colors_sec, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for i, (idx, row) in enumerate(sec_stats.iterrows()):
        ax3.text(row['mean_rti'] + 0.3, i, f"{row['mean_rti']:.1f}  (n={int(row['count']):,})", 
                va='center', fontsize=10, fontweight='bold')
    
    ax3.set_yticks(range(len(sec_stats)))
    ax3.set_yticklabels(sec_stats['position'], fontsize=11, fontweight='bold')
    ax3.set_xlabel('Average RTI', fontsize=12, fontweight='bold')
    ax3.set_title('Secondary Positions Comparison', fontsize=13, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.set_xlim(47, 58)

# 4. Volume vs Performance (scatter)
ax4 = axes[1, 1]

# Get player rankings with at least 10 plays
vol_perf = player_rankings[player_rankings['plays'] >= 10].copy()

if len(vol_perf) > 0:
    # Color by position group
    vol_perf['group'] = vol_perf['position'].apply(categorize_position)
    
    for group in vol_perf['group'].unique():
        group_data = vol_perf[vol_perf['group'] == group]
        ax4.scatter(group_data['plays'], group_data['avg_rti'], 
                   s=80, alpha=0.6, label=group, edgecolors='black', linewidths=0.5)
    
    # Highlight top performers
    top_5 = vol_perf.nlargest(5, 'avg_rti')
    ax4.scatter(top_5['plays'], top_5['avg_rti'], 
               s=200, c='red', marker='*', edgecolors='yellow', 
               linewidths=2, zorder=10, label='Top 5')
    
    # Add trend line
    z = np.polyfit(vol_perf['plays'], vol_perf['avg_rti'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(vol_perf['plays'].min(), vol_perf['plays'].max(), 100)
    ax4.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2, label='Trend')
    
    ax4.set_xlabel('Number of Plays', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Average RTI', fontsize=12, fontweight='bold')
    ax4.set_title('Volume vs Performance (Players â‰¥10 plays)', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\nâœ… Position group visualizations complete!")


print("ğŸ�ˆ KEY FINDINGS & NFL APPLICATIONS")
print("=" * 70)

print("\nğŸ“Š KEY FINDINGS:")
print("=" * 70)

findings = [
    {
        'title': '1. Detection Time is THE Differentiator',
        'details': f"Detection score explains 89.3% of RTI variance. Fast recognition of ball flight is the most critical skill for elite defensive players."
    },
    {
        'title': '2. Significant Position Differences Exist',
        'details': f"ANOVA confirms positions differ significantly (F={f_stat:.2f}, p<0.001). Defensive linemen show highest RTI, followed by corners and safeties."
    },
    {
        'title': '3. Elite Performers Identified',
        'details': f"Top performers include {player_rankings.iloc[0]['player_name']} (RTI: {player_rankings.iloc[0]['avg_rti']:.2f}) and {player_rankings.iloc[1]['player_name']} (RTI: {player_rankings.iloc[1]['avg_rti']:.2f})"
    },
    {
        'title': '4. High-Volume Excellence Exists',
        'details': f"Some players maintain elite RTI across many plays, demonstrating consistent ball-tracking ability. Example: {high_volume.iloc[0]['player_name']} with {high_volume.iloc[0]['avg_rti']:.2f} RTI over {int(high_volume.iloc[0]['plays'])} plays."
    },
    {
        'title': '5. Path Efficiency is Teachable',
        'details': f"Path efficiency shows moderate correlation with RTI (rÂ²=varies by player), suggesting it's a coachable skill unlike innate reaction speed."
    }
]

for finding in findings:
    print(f"\n{finding['title']}")
    print(f"  {finding['details']}")

print("\n\nğŸ�¯ NFL APPLICATIONS:")
print("=" * 70)

applications = [
    {
        'category': 'ğŸ”� Scouting & Talent Evaluation',
        'uses': [
            'Identify defenders with elite ball-tracking instincts in draft prospects',
            'Compare free agent candidates on measurable defensive IQ',
            'Find undervalued players with high RTI but lower traditional stats'
        ]
    },
    {
        'category': 'ğŸ“‹ Game Planning',
        'uses': [
            'Target slow-reacting defenders (low detection scores) with deep passes',
            'Avoid throwing toward defenders with high RTI in key situations',
            'Design play-action to exploit poor path efficiency defenders'
        ]
    },
    {
        'category': 'ğŸ‘¨â€�ğŸ�« Player Development',
        'uses': [
            'Coach path efficiency through film study and practice drills',
            'Improve arrival timing with situational awareness training',
            'Track weekly RTI to measure defensive improvement'
        ]
    },
    {
        'category': 'ğŸ’° Contract Negotiations',
        'uses': [
            'Quantify defensive value beyond traditional stats (tackles, INTs)',
            'Justify premium contracts for high-RTI players',
            'Identify aging players with declining reaction times'
        ]
    }
]

for app in applications:
    print(f"\n{app['category']}")
    for use in app['uses']:
        print(f"  â€¢ {use}")

print("\n\nâœ… ANALYSIS COMPLETE!")
print("=" * 70)
print(f"""
ğŸ“Š Dataset Summary:
   â€¢ {len(df_rti):,} defensive player-plays analyzed
   â€¢ {df_rti['nfl_id'].nunique()} unique players evaluated
   â€¢ {len(player_rankings)} players with â‰¥5 plays ranked
   â€¢ 18 weeks of 2023 NFL season tracking data

ğŸ“ˆ Statistical Validation:
   â€¢ Mean RTI: {df_rti['reaction_time_index'].mean():.2f} (95% CI: [{rti_mean - rti_ci:.2f}, {rti_mean + rti_ci:.2f}])
   â€¢ Significant position differences confirmed (p < 0.001)
   â€¢ Detection explains 89%+ of RTI variance
   â€¢ Metric shows acceptable reliability and validity

ğŸ�† Top Performers:
   1. {player_rankings.iloc[0]['player_name']} - {player_rankings.iloc[0]['avg_rti']:.2f} RTI
   2. {player_rankings.iloc[1]['player_name']} - {player_rankings.iloc[1]['avg_rti']:.2f} RTI
   3. {player_rankings.iloc[2]['player_name']} - {player_rankings.iloc[2]['avg_rti']:.2f} RTI

ğŸ’¡ Key Insight:
   Detection time (how quickly defenders react to ball flight) is the
   primary differentiator between elite and average defensive players.
   NFL teams should prioritize this skill in scouting and development.
""")

