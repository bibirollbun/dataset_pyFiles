%matplotlib inline
import pandas as pd
import numpy as np  
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
import seaborn as sns

# Suppress all warnings
warnings.filterwarnings('ignore')

# Configuration
SUPPLEMENT_FILE = 'supplementary_data.csv'
INPUT_PATTERN = 'input_2023_w{:02d}.csv'
OUTPUT_PATTERN = 'supplement-input_2023_w{:02d}.csv'
NUM_WEEKS = 1

# Updated paths for Kaggle competition data
KAGGLE_PATH = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final"
TRAIN_PATH = os.path.join(KAGGLE_PATH, "train")


 # Concatenate all weekly data into one dataset
print(f"\n{'='*70}")
print("STEP 1: CONCATENATING ALL WEEKS INTO SINGLE DATASET")
print("="*70)

# Load all weeks data into one dataframe
input_data_frames = []
for week in range(1, NUM_WEEKS + 1):
    file_path = os.path.join(TRAIN_PATH, INPUT_PATTERN.format(week))
    df_week = pd.read_csv(file_path)
    df_week['week'] = week  # Add week column to track the source
    input_data_frames.append(df_week)

# Combine all dataframes
input_data = pd.concat(input_data_frames, ignore_index=True)
print(f"Loaded {len(input_data_frames)} weeks of data with {len(input_data)} total rows")
print(f"Test : {input_data['week'].max()}")


# Load supplementary data
supplementary_data = pd.read_csv(os.path.join(KAGGLE_PATH, SUPPLEMENT_FILE))
supplementary_data.columns

# Add feature pass_category based on pass_length. 
# pass_categories are 'short', 'intermediate', and 'long'. Short is <=5 yards, intermediate is >5 and <=15 yards, long is >15 yards. Use dict mapping.
conditions = [
    supplementary_data['pass_length'] < 5,
    (supplementary_data['pass_length'] >= 5) & (supplementary_data['pass_length'] <= 15),
    supplementary_data['pass_length'] > 15
]

categories = ['short', 'intermediate', 'long']

supplementary_data['pass_type'] = np.select(
    conditions, 
    categories, 
    default='unknown'
)

# Merge the supplementary column 'route_of_targeted_receiver' into the input data
input_data = pd.merge(
    input_data,                 # The left DataFrame (the one receiving the new column)
    supplementary_data[['game_id', 'play_id', 'route_of_targeted_receiver']],  # The right DataFrame (only the key columns and the column to add)
    on=['game_id', 'play_id'],  # The key columns to match on
    how='left'           # Specifies a left join
)


def calculate_euclidean_distance_vectorized(tr_x, tr_y, def_x, def_y):
    """
    Calculate Euclidean distance between Targeted Receiver and all defenders using vectorized operations.
    
    Parameters:
    tr_x, tr_y: Quaterback coordinates (scalar)
    def_x, def_y: Defender coordinates (arrays/series)
    
    Returns:
    Array: Euclidean distances to all defenders
    """
    return np.sqrt((def_x - tr_x)**2 + (def_y - tr_y)**2)

def calculate_frame_level_separation(df):
    """
    Calculate minimum separation between Targeted Receiver and defensive players at FRAME level.
    Returns only Targeted Receiver data with tr_min_separation per frame.
    
    Parameters:
    df: DataFrame containing tracking data with columns:
        - game_id, play_id, frame_id, nfl_id
        - player_role: 'Targeted Receiver' or 'Defensive Coverage'
        - player_name
        - x, y: Player coordinates
    
    Returns:
    DataFrame: Targeted Receiver data with columns (game_id, play_id, nfl_id, frame_id, player_name, player_role, tr_min_separation)
    """
    
    # Group by game, play, and frame (FRAME-LEVEL calculation)
    grouped = df.groupby(['game_id', 'play_id', 'frame_id'])
    
    # List to store Targeted Receiver results
    tr_results = []
    
    for (game_id, play_id, frame_id), frame_group in grouped:
        # Identify Targeted Receivers and defenders in this specific frame
        target_receivers = frame_group[frame_group['player_role'] == 'Targeted Receiver']
        defenders = frame_group[frame_group['player_role'] == 'Defensive Coverage']
        
        # Skip if no Targeted Receivers or defenders in this frame
        if target_receivers.empty or defenders.empty:
            continue
        
        # Extract defender coordinates as arrays for vectorized calculation
        def_x = defenders['x'].values
        def_y = defenders['y'].values
        
        # For each Targeted Receiver, calculate distance to all defenders using vectorization
        for tr_idx, tr_row in target_receivers.iterrows():
            tr_x, tr_y = tr_row['x'], tr_row['y']
            
            # Vectorized distance calculation across all defenders in this frame
            distances = calculate_euclidean_distance_vectorized(tr_x, tr_y, def_x, def_y)
            
            # Find minimum distance for this frame
            min_distance = np.min(distances)
            
            # Create result record for this Targeted Receiver
            tr_result = {
                'game_id': game_id,
                'play_id': play_id,
                'nfl_id': tr_row['nfl_id'],
                'frame_id': frame_id,
                'player_name': tr_row['player_name'],
                'player_role': tr_row['player_role'],
                'player_position': tr_row['player_position'],
                'tr_min_separation': min_distance,
                'route_of_targeted_receiver': tr_row['route_of_targeted_receiver'],
                'dir': tr_row['dir']
            }
            tr_results.append(tr_result)
    
    # Convert results to DataFrame
    result_df = pd.DataFrame(tr_results)
    
    # Sort by game, play, receiver, and frame for chronological analysis
    if not result_df.empty:
        result_df = result_df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id']).reset_index(drop=True)
    
    return result_df

def tr_calculate_play_level_aggregates(df):
    """
    Calculate play-level aggregate statistics for each Targeted Receiver.
    Aggregates: min, avg (mean), variance of separation across all frames in a play.
    
    Parameters:
    df: DataFrame with frame-level tr_min_separation calculated
    
    Returns:
    DataFrame: Play-level aggregates for each Targeted Receiver
    """
    
    # Filter for Targeted Receivers only
    tr_data = df[df['player_role'] == 'Targeted Receiver'].copy()
    
    # Remove rows where separation couldn't be calculated
    tr_data = tr_data[tr_data['tr_min_separation'].notna()]
    
    if tr_data.empty:
        return pd.DataFrame()
    
    # Group by game, play, and receiver (nflId)
    # Calculate aggregate statistics across all frames in each play
    play_aggregates = tr_data.groupby(['game_id', 'play_id']).agg({
        'tr_min_separation': [
            ('tr_play_min_separation', 'min'),      # Minimum separation across all frames
            ('tr_play_avg_separation', 'mean'),     # Average separation across all frames
            ('tr_play_var_separation', 'var')       # Variance of separation across all frames
        ],
        'player_name': 'first',                     # Keep receiver name
        'player_role': 'first',                     # Keep receiver role
        'player_position': 'first',                 # Keep receiver position
        'route_of_targeted_receiver': 'first',     # Keep receiver route
        'nfl_id': 'first',                          # Keep receiver nfl_id
        'frame_id': [
            ('nums_frame_pre_throw', 'count')],              # Keep pre-throw frames
    }).reset_index()
    
    # Flatten column names
    play_aggregates.columns = ['game_id', 'play_id',
                                'tr_play_min_separation', 
                                'tr_play_avg_separation', 
                                'tr_play_var_separation',
                                'player_name','player_role', 'player_position', 'route_of_targeted_receiver', 
                                'nfl_id',
                                'nums_frame_pre_throw'
                            ]
    
    # filter tr_data to get rows for 1st frame and last frame of each play for each receiver
    first_frames = tr_data.sort_values('frame_id').groupby(['game_id', 'play_id', 'nfl_id']).first().reset_index()
    last_frames = tr_data.sort_values('frame_id').groupby(['game_id', 'play_id', 'nfl_id']).last().reset_index()

    # Rename the 'direction' column in last_frames before merging
    last_frames = last_frames.rename(columns={'dir': 'tr_last_dir'})

    # Merge first and last frame data to get initial and final separation
    first_last = pd.merge(
        first_frames[['game_id', 'play_id', 'nfl_id', 'tr_min_separation']],
        last_frames[['game_id', 'play_id', 'nfl_id', 'tr_min_separation', 'tr_last_dir']],
        on=['game_id', 'play_id', 'nfl_id'],
        suffixes=('_first', '_last')
    )
    
    # merge to play_aggregates
    play_aggregates = pd.merge(
        play_aggregates,
        first_last,
        on=['game_id', 'play_id', 'nfl_id'],
        how='left'
    )
    
    return play_aggregates

def calculate_frame_level_separation_qb(df):
    """
    Calculate minimum separation between Quarterback and defensive players at FRAME level.
    Returns only Quarterback data with qb_min_separation per frame.
    
    Parameters:
    df: DataFrame containing tracking data with columns:
        - game_id, play_id, frame_id, nfl_id
        - player_role: 'Passer' or 'Defensive Coverage'
        - player_name
        - x, y: Player coordinates
    
    Returns:
    DataFrame: Quarterback data with columns (game_id, play_id, nfl_id, frame_id, player_name, player_role, qb_min_separation)
    """
    
    # Group by game, play, and frame (FRAME-LEVEL calculation)
    grouped = df.groupby(['game_id', 'play_id', 'frame_id'])
    
    # List to store Quarterback results
    qb_results = []
    
    for (game_id, play_id, frame_id), frame_group in grouped:
        # Identify Quarterbacks and defenders in this specific frame
        quarterbacks = frame_group[frame_group['player_role'] == 'Passer']
        defenders = frame_group[frame_group['player_role'] == 'Defensive Coverage']
        
        # Skip if no Quarterbacks or defenders in this frame
        if quarterbacks.empty or defenders.empty:
            continue
        
        # Extract defender coordinates as arrays for vectorized calculation
        def_x = defenders['x'].values
        def_y = defenders['y'].values
        
        # For Quarterback, calculate distance to all defenders using vectorization
        for qb_idx, qb_row in quarterbacks.iterrows():
            qb_x, qb_y = qb_row['x'], qb_row['y']
            
            # Vectorized distance calculation across all defenders in this frame
            distances = calculate_euclidean_distance_vectorized(qb_x, qb_y, def_x, def_y)
            
            # Find minimum distance for this frame
            min_distance = np.min(distances)
            
            # Create result record for this Quarterback
            qb_result = {
                'game_id': game_id,
                'play_id': play_id,
                'nfl_id': qb_row['nfl_id'],
                'frame_id': frame_id,
                'player_name': qb_row['player_name'],
                'player_role': qb_row['player_role'],
                'player_position': qb_row['player_position'],
                'qb_min_separation': min_distance
            }
            qb_results.append(qb_result)
    
    # Convert results to DataFrame
    result_df = pd.DataFrame(qb_results)
    
    # Sort by game, play, quarterback, and frame for chronological analysis
    if not result_df.empty:
        result_df = result_df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id']).reset_index(drop=True)
    
    return result_df


def qb_calculate_play_level_aggregates(df):
    """
    Calculate play-level aggregate statistics for each QB.
    Aggregates: min, avg (mean), variance of separation across all frames in a play.
    
    Parameters:
    df: DataFrame with frame-level qb_min_separation calculated
    
    Returns:
    DataFrame: Play-level aggregates for each QB
    """
    
    # Filter for QB only
    qb_data = df[df['player_role'] == 'Passer'].copy()
    
    # Remove rows where separation couldn't be calculated
    qb_data = qb_data[qb_data['qb_min_separation'].notna()]
    
    if qb_data.empty:
        return pd.DataFrame()
    
    # Group by game, play, and QB (nflId)
    # Calculate aggregate statistics across all frames in each play
    play_aggregates = qb_data.groupby(['game_id', 'play_id']).agg({
        'qb_min_separation': [
            ('qb_play_min_separation', 'min'),      # Minimum separation across all frames
            ('qb_play_avg_separation', 'mean'),     # Average separation across all frames
            ('qb_play_var_separation', 'var')       # Variance of separation across all frames
        ],
        'player_name': 'first',                     # Keep QB name
        'nfl_id': 'first',                          # Keep QB nfl_id
        'player_role': 'first',                      # Keep QB role
        'frame_id': [
            ('time_to_throw', lambda x: x.count() * 0.1)],              # Calculate time (0.1s per frame)
    }).reset_index()
    
    # Flatten column names
    play_aggregates.columns = ['game_id', 'play_id',
                                'qb_play_min_separation', 
                                'qb_play_avg_separation', 
                                'qb_play_var_separation',
                                'player_name', 'nfl_id', 'player_role',
                                'time_to_throw']
    
    # Calculate time to throw (0.1 seconds per frame)
    # play_aggregates['time_to_throw'] = play_aggregates['nums_frame_pre_throw'] * 0.1
    
    return play_aggregates



# Calculate frame-level minimum separation
print(f"\n{'='*70}")
print("STEP 2: CALCULATING FRAME-LEVEL MINIMUM SEPARATIONS")
print("Processing all frames using vectorized distance calculations...")
tr_separation_df = calculate_frame_level_separation(input_data)
print("✓ Frame-level separation calculations complete")


# Calculate play-level aggregates
tr_play_level_agg = pd.DataFrame()

print(f"\n{'='*70}")
print("STEP 3: CALCULATING PLAY-LEVEL AGGREGATE STATISTICS")
print("Aggregating: min, avg, variance across all frames per play...")
tr_play_level_agg = tr_calculate_play_level_aggregates(tr_separation_df)

display(tr_play_level_agg.head())


# Calculate frame-level minimum separation for Quarterbacks
print(f"\n{'='*70}")
print("STEP 4: CALCULATING FRAME-LEVEL MINIMUM SEPARATIONS FOR QUARTERBACKS")
print("Processing all frames using vectorized distance calculations...")
qb_separation_df = calculate_frame_level_separation_qb(input_data)
print("✓ Frame-level separation calculations complete")

# Calculate play-level aggregates for Quarterbacks
qb_play_level_agg = pd.DataFrame()

print(f"\n{'='*70}")
print("STEP 5: CALCULATING QB PLAY-LEVEL AGGREGATE STATISTICS")
print("="*70)
print("Aggregating: min, avg, variance across all frames per play...")
qb_play_level_agg = qb_calculate_play_level_aggregates(qb_separation_df)


display(qb_play_level_agg.head())


# Merge play-level QB and Tr aggregates with supplementary data for pair analysis
play_columns = [
    "pass_result",
    "pass_length",
    "pass_location_type",
    "dropback_type",
    "dropback_distance",
    "play_action",
    "offense_formation",
    "defenders_in_the_box",
    "team_coverage_man_zone",
    "team_coverage_type",
    "pass_type"
]            
 
print(f"\n{'='*70}")
print("STEP 6: MERGING PLAY-LEVEL QB and TR AGGREGATES WITH SUPPLEMENTARY DATA")    

# Merge play-level QB aggregates with TR aggregates at play level
play_level_qb_tr = pd.merge(qb_play_level_agg, tr_play_level_agg, on=['game_id', 'play_id'], how='left', suffixes=('_qb', '_tr'))

# Print columns after merging QB and TR aggregates
print("Columns after merging QB and TR play-level aggregates:")
#print(play_level_qb_tr.columns.tolist())    

# Merge play-level QB, Tr aggregates with supplementary data
play_columns = ['game_id', 'play_id'] + play_columns
qb_supplementary_data = supplementary_data[play_columns]
merged_data = pd.merge(play_level_qb_tr, qb_supplementary_data, on=['game_id', 'play_id'], how='left')

display(merged_data.head())


###--------------------------------
# Visualize player direction in last frame distribution per category - one polar diagram per route
###--------------------------------

# Get unique routes and create subplot layout
unique_routes = play_level_qb_tr['route_of_targeted_receiver'].dropna().unique()
n_routes = len(unique_routes)

# Calculate grid dimensions (try to make it roughly square)
n_cols = int(np.ceil(np.sqrt(n_routes)))
n_rows = int(np.ceil(n_routes / n_cols))

# Create figure with polar subplots
fig = plt.figure(figsize=(4*n_cols, 4*n_rows))
fig.suptitle('Targeted Receiver Last Frame Direction Distribution by Route', fontsize=16, fontweight='bold', y=0.98)

# Define colors for each route
colors = plt.cm.Set3(np.linspace(0, 1, n_routes))

for i, route in enumerate(unique_routes):
    # Create polar subplot
    ax = fig.add_subplot(n_rows, n_cols, i+1, polar=True)
    
    # Filter data for this route
    route_data = play_level_qb_tr[play_level_qb_tr['route_of_targeted_receiver'] == route]
    
    if not route_data.empty and route_data['tr_last_dir'].notna().any():
        # Convert degrees to radians
        directions_rad = np.deg2rad(route_data['tr_last_dir'].dropna())
        
        # Create histogram bins (36 bins = 10 degree intervals)
        n_bins = 36
        bins = np.linspace(0, 2*np.pi, n_bins + 1)
        
        # Calculate histogram
        hist, bin_edges = np.histogram(directions_rad, bins=bins)
        
        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Create bar plot
        bars = ax.bar(bin_centers, hist, width=2*np.pi/n_bins, 
                     color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Set title for this subplot
        ax.set_title(f'{route}\n(n={len(route_data)})', fontsize=12, fontweight='bold', pad=20)
        
        # Customize polar plot
        ax.set_theta_zero_location('N')  # 0 degrees at top
        ax.set_theta_direction(-1)       # Clockwise direction
        ax.set_thetagrids(range(0, 360, 45), ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        
        # Set radial limits
        ax.set_ylim(0, max(hist) * 1.1 if max(hist) > 0 else 1)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_dir = np.rad2deg(np.arctan2(np.sin(directions_rad).mean(), np.cos(directions_rad).mean()))
        if mean_dir < 0:
            mean_dir += 360
        
        ax.text(0.02, 0.98, f'Mean: {mean_dir:.1f}°\nCount: {len(directions_rad)}', 
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    else:
        # Handle empty data case
        ax.set_title(f'{route}\n(No data)', fontsize=12, fontweight='bold', pad=20)
        ax.text(0.5, 0.5, 'No direction data\navailable', 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=10, style='italic')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(top=0.94)  # Make room for suptitle
plt.show()
plt.close()  # Fix: Properly close figure





#----------------------------------------------------------------
# Visualization Section
#----------------------------------------------------------------
# Targeted Receiver Analysis by Route Category - Enhanced with Y-axis Routes

# Set up the plotting style for better aesthetics
plt.style.use('default')
sns.set_palette("husl")

# Create figure with 1x4 layout
fig, axes = plt.subplots(1, 4, figsize=(20, 8))

# Define a professional color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#4A90A4', '#83A95C', '#F4B942']

# Plot 1: Time to Throw vs Route (horizontal boxplot)
sns.boxplot(data=merged_data, y='route_of_targeted_receiver', x='time_to_throw', 
            ax=axes[0], palette=colors, orient='h')
axes[0].set_title('Time to Throw by Route Category', fontsize=14, fontweight='bold', pad=20)
axes[0].set_ylabel('Route Category', fontsize=14, fontweight='semibold')
axes[0].set_xlabel('Time to Throw (seconds)', fontsize=10, fontweight='semibold')
axes[0].tick_params(axis='y', labelsize=12)
axes[0].tick_params(axis='x', labelsize=12)
axes[0].grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
axes[0].set_facecolor('#fafafa')

# Plot 2: Final Separation vs Route (horizontal boxplot) - Remove Y labels
sns.boxplot(data=merged_data, y='route_of_targeted_receiver', x='tr_min_separation_last',
            ax=axes[1], palette=colors, orient='h')
axes[1].set_title('TR Final Separation by Route Category', fontsize=14, fontweight='bold', pad=20)
axes[1].set_ylabel('')  # Remove ylabel
axes[1].set_xlabel('TR Final Separation (yards)', fontsize=10, fontweight='semibold')
axes[1].tick_params(axis='y', labelleft=False)  # Hide y-axis labels
axes[1].tick_params(axis='x', labelsize=12)
axes[1].grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
axes[1].set_facecolor('#fafafa')

# Plot 3: Pass Length vs Route (horizontal boxplot) - Remove Y labels
sns.boxplot(data=merged_data, y='route_of_targeted_receiver', x='pass_length', 
            ax=axes[2], palette=colors, orient='h')
axes[2].set_title('Pass Length by Route Category', fontsize=14, fontweight='bold', pad=20)
axes[2].set_ylabel('')  # Remove ylabel
axes[2].set_xlabel('Pass Length (yards)', fontsize=10, fontweight='semibold')
axes[2].tick_params(axis='y', labelleft=False)  # Hide y-axis labels
axes[2].tick_params(axis='x', labelsize=12)
axes[2].grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
axes[2].set_facecolor('#fafafa')

# Plot 4: Dropback Distance vs Route (horizontal boxplot) - Remove Y labels
sns.boxplot(data=merged_data, y='route_of_targeted_receiver', x='dropback_distance', 
            ax=axes[3], palette=colors, orient='h')
axes[3].set_title('Dropback Distance by Route Category', fontsize=14, fontweight='bold', pad=20)
axes[3].set_ylabel('')  # Remove ylabel
axes[3].set_xlabel('Dropback Distance (yards)', fontsize=10, fontweight='semibold')
axes[3].tick_params(axis='y', labelleft=False)  # Hide y-axis labels
axes[3].tick_params(axis='x', labelsize=12)
axes[3].grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
axes[3].set_facecolor('#fafafa')

# Apply consistent styling to all subplots
for ax in axes.flat:
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Make remaining spines thicker and darker
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')

# Adjust layout with better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=3.0)

# Save high-quality version
plt.savefig('route_analysis_horizontal.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()
plt.close()

# Targeted Receiver Analysis by Pass type - Enhanced with Y-axis types
# Same as above but for pass_type instead of route_of_targeted_receiver except pass length and 1x3 layout
# Create figure with 1x3 layout
# Add pass lenngth per pass type in plot y axis along with pass type

fig, axes = plt.subplots(1, 3, figsize=(15, 8))    
# Define a professional color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#4A90A4', '#83A95C', '#F4B942']
# Create custom y-axis labels with yards range
pass_type_labels = {
    'short': 'Short (≤5 yds)',
    'intermediate': 'Intermediate (>5-15 yds)', 
    'long': 'Long (>15 yds)'
}
# Plot 1: Time to Throw vs Pass Type (horizontal boxplot)
sns.boxplot(data=merged_data, y='pass_type', x='time_to_throw', 
            ax=axes[0], palette=colors, orient='h')
axes[0].set_title('Time to Throw by Pass Type', fontsize=14, fontweight='bold', pad=20)
axes[0].set_ylabel('Pass Type', fontsize=14, fontweight='semibold')
axes[0].set_xlabel('Time to Throw (seconds)', fontsize=10, fontweight='semibold')
axes[0].tick_params(axis='y', labelsize=12)
axes[0].tick_params(axis='x', labelsize=12)
axes[0].grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
axes[0].set_facecolor('#fafafa')
# Plot 2: Final Separation vs Pass Type (horizontal boxplot) - Remove Y labels
sns.boxplot(data=merged_data, y='pass_type', x='tr_min_separation_last',
            ax=axes[1], palette=colors, orient='h')
axes[1].set_title('TR Final Separation by Pass Type', fontsize=14, fontweight='bold', pad=20)
axes[1].set_ylabel('')  # Remove ylabel
axes[1].set_xlabel('TR Final Separation (yards)', fontsize=10, fontweight='semibold')
axes[1].tick_params(axis='y', labelleft=False)  # Hide y-axis labels
axes[1].tick_params(axis='x', labelsize=12)
axes[1].grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
axes[1].set_facecolor('#fafafa')
# Plot 3: Dropback Distance vs Pass Type (horizontal boxplot) - Remove Y labels
sns.boxplot(data=merged_data, y='pass_type', x='dropback_distance', 
            ax=axes[2], palette=colors, orient='h')
axes[2].set_title('Dropback Distance by Pass Type', fontsize=14, fontweight='bold', pad=20)
axes[2].set_ylabel('')  # Remove ylabel
axes[2].set_xlabel('Dropback Distance (yards)', fontsize=10, fontweight='semibold')
axes[2].tick_params(axis='y', labelleft=False)  # Hide y-axis labels
axes[2].tick_params(axis='x', labelsize=12)
axes[2].grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
axes[2].set_facecolor('#fafafa')

# Apply y-axis labels with yards range to all plots at once
for ax in axes:
    ax.set_yticklabels([pass_type_labels.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()])

# Apply consistent styling to all subplots
for ax in axes.flat:
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Make remaining spines thicker and darker
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
# Adjust layout with better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=3.0)
# Save high-quality version
plt.savefig('pass_type_analysis_horizontal.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()
plt.close()


# Print summary statistics
print(f"\n{'='*70}")
print("DIRECTION DISTRIBUTION SUMMARY BY ROUTE")
print("="*70)
for route in unique_routes:
    route_data = play_level_qb_tr[play_level_qb_tr['route_of_targeted_receiver'] == route]
    if not route_data.empty and route_data['tr_last_dir'].notna().any():
        directions = route_data['tr_last_dir'].dropna()
        directions_rad = np.deg2rad(directions)
        
        # Calculate circular mean
        mean_dir = np.rad2deg(np.arctan2(np.sin(directions_rad).mean(), np.cos(directions_rad).mean()))
        if mean_dir < 0:
            mean_dir += 360
            
        print(f"{route:20} | Count: {len(directions):3d} | Mean Direction: {mean_dir:6.1f}° | Range: {directions.min():6.1f}° - {directions.max():6.1f}°")
    else:
        print(f"{route:20} | No direction data available")

# Print summary statistics for better understanding
print(f"\n{'='*80}")
print("ROUTE ANALYSIS SUMMARY STATISTICS")
print("="*80)

# Get route counts for reference
route_counts = merged_data['route_of_targeted_receiver'].value_counts()

for route in merged_data['route_of_targeted_receiver'].dropna().unique():
    route_data = merged_data[merged_data['route_of_targeted_receiver'] == route]
    print(f"\n{route.upper()} (n={len(route_data)}):")
    print(f"  Time to Throw: {route_data['time_to_throw'].mean():.2f}s ± {route_data['time_to_throw'].std():.2f}s")
    print(f"  Final Separation: {route_data['tr_min_separation_last'].mean():.2f} ± {route_data['tr_min_separation_last'].std():.2f} yards")
    print(f"  Pass Length: {route_data['pass_length'].mean():.2f} ± {route_data['pass_length'].std():.2f} yards")
    print(f"  Dropback Distance: {route_data['dropback_distance'].mean():.2f} ± {route_data['dropback_distance'].std():.2f} yards")

## Route impact analysis
def analyze_route_impact(merged_data):
    """
    Analyze how route types affect:
    - TR separation metrics 
    - QB min separation - QB in Pressure
    - Pass completion rates
    - Time to throw
    """
    route_analysis = merged_data.groupby('route_of_targeted_receiver').agg({
        'tr_play_avg_separation': ['mean', 'std'],
        'qb_play_min_separation': ['mean', 'std'],
        'time_to_throw': ['mean'],
        'pass_length': ['mean', 'std'],
        'pass_result': lambda x: (x == 'C').mean()  # Completion rate
    })
    return route_analysis

route_impact = analyze_route_impact(merged_data)
print(f"\n{'='*70}")
print(route_impact) 



def plot_temporal_separation_by_route(df, route_name, max_plays=10):
    """
    Plot temporal separation for plays of a specific route type.
    
    Parameters:
    df: DataFrame with separation data containing columns:
        - route_of_targeted_receiver, play_id, game_id, frame_id, tr_min_separation
    route_name: String name of the route to filter for (e.g., 'Go', 'Slant', 'Out')
    max_plays: Maximum number of plays to display (default 10)
    
    Returns:
    None (displays plot)
    """
    # Filter for route plays and get first 10 unique plays with all their frames
    route_data = df[df['route_of_targeted_receiver'] == route_name].copy()

    
    print(f"Found {len(route_data)} '{route_name}' route records")
    
    # Get first 10 unique plays
    unique_plays = sorted(route_data['play_id'].unique())[:10]
    route_data = route_data[route_data['play_id'].isin(unique_plays)]
    
    print(f"Selected {len(unique_plays)} plays with {len(route_data)} total records")
    
    # Sort by time for proper line plotting
    route_data = route_data.sort_values(['game_id', 'play_id', 'frame_id'])
    
    # Plot temporal separation for route route plays
    plt.figure(figsize=(12, 6))   

    plot_count = 0
    for (game_id, play_id), play_group in route_data.groupby(['game_id', 'play_id']):
        if not play_group.empty:
            plt.plot(play_group['frame_id'] * 0.1, play_group['tr_min_separation'], 
                    marker='o', linestyle='-', linewidth=2, markersize=4,
                    label=f'Game {game_id} Play {play_id}')
            plot_count += 1

    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Temporal Separation (tr_min_separation)', fontsize=12)
    plt.title(f'Temporal Separation before throw for "{route_name}" Route Plays (n={plot_count})', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()   

print(f"\n{'='*70}")
print("Target Reciever TEMPORAL SEPARATION")

plot_temporal_separation_by_route(tr_separation_df, 'GO', max_plays=10)
plot_temporal_separation_by_route(tr_separation_df, 'FLAT', max_plays=10)
plot_temporal_separation_by_route(tr_separation_df, 'SCREEN', max_plays=10)

# Test with another route if available
available_routes = tr_separation_df['route_of_targeted_receiver'].dropna().unique()
if len(available_routes) > 1:
    second_route = available_routes[1]
    print(f"\nTesting with '{second_route}' route:")
    plot_temporal_separation_by_route(tr_separation_df, second_route, max_plays=5)


# Convert qb_play_min_separation to numeric and handle any invalid values
merged_data['qb_play_min_separation'] = pd.to_numeric(merged_data['qb_play_min_separation'], errors='coerce')

# Remove rows with NaN values in the separation column for plotting
plot_data = merged_data.dropna(subset=['qb_play_min_separation'])

print(f"Original data: {len(merged_data)} rows")
print(f"Data for plotting (after removing NaN): {len(plot_data)} rows")
print(f"Data type of qb_play_min_separation: {plot_data['qb_play_min_separation'].dtype}")

# Create box plots for QB play-level analysis
# Arrange subplots in a 2x2 grid starting from the top-left. 
# Each box plot will compare qb_play_min_separation against different categorical variables.
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Define colors for better visual appeal
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']

# Fix: Use seaborn instead of deprecated pandas boxplot method
import seaborn as sns

# Box plot 1: qb_play_min_separation vs pass_result
sns.boxplot(data=plot_data, x='pass_result', y='qb_play_min_separation', ax=axes[0, 0], palette=colors)
axes[0, 0].set_title('QB Minimum Separation by Pass Result', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Pass Result', fontsize=12)
axes[0, 0].set_ylabel('QB Minimum Separation (yards)', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)

# Box plot 2: qb_play_min_separation vs offense_formation
sns.boxplot(data=plot_data, x='offense_formation', y='qb_play_min_separation', ax=axes[1, 0], palette=colors)
axes[1, 0].set_title('QB Minimum Separation by Offensive Formation', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Offensive Formation', fontsize=12)
axes[1, 0].set_ylabel('QB Minimum Separation (yards)', fontsize=12)
axes[1, 0].tick_params(axis='x', rotation=10, labelsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Box plot 3: qb_play_min_separation vs dropback_type
sns.boxplot(data=plot_data, x='dropback_type', y='qb_play_min_separation', ax=axes[0, 1], palette=colors)
axes[0, 1].set_title('QB Minimum Separation by Dropback Type', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Dropback Type', fontsize=12)
axes[0, 1].set_ylabel('QB Minimum Separation (yards)', fontsize=12)
axes[0, 1].tick_params(axis='x', rotation=10, labelsize=8)
axes[0, 1].grid(True, alpha=0.3)

# Box plot 4: qb_play_min_separation vs defenders_in_the_box
sns.boxplot(data=plot_data, x='defenders_in_the_box', y='qb_play_min_separation', ax=axes[1, 1], palette=colors)
axes[1, 1].set_title('QB Minimum Separation by Defenders in the Box', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Defenders in the Box', fontsize=12)
axes[1, 1].set_ylabel('QB Minimum Separation (yards)', fontsize=12)
axes[1, 1].tick_params(axis='x', rotation=0, labelsize=10)
axes[1, 1].grid(True, alpha=0.3)

# Add background color and remove top/right spines for cleaner look
for ax in axes.flat:
    ax.set_facecolor('#f8f9fa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(pad=3.0)
plt.show()



# Visualization of play-level min seperation of Targeted Receivers and Quaterbacks with other variables to see trends
print(f"\n{'='*70}")
print("STEP 7: COMPARATIVE VISUALIZATIONS FOR TARGETED RECEIVERS AND QUARTERBACKS")
print("="*70)

import seaborn as sns
    
fig, axes = plt.subplots(5, 2, figsize=(16, 28))

# Scatter plot: Both TR and QB min separation vs Time to Throw
sns.scatterplot(data=merged_data, x='time_to_throw', y='tr_play_min_separation', ax=axes[0,0], color='#FF6B6B', alpha=0.6, label='Targeted Receiver')
sns.scatterplot(data=merged_data, x='time_to_throw', y='qb_play_min_separation', ax=axes[0,0], color='#4ECDC4', alpha=0.6, label='Quarterback')
axes[0,0].set_title('Min Separation vs Time to Throw (TR & QB)', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('Time to Throw (Seconds)', fontsize=12) 
axes[0,0].set_ylabel('Min Separation (yards)', fontsize=12)
axes[0,0].grid(True, alpha=0.3)
axes[0,0].legend()

# Scatter plot:  Both TR and QB min separation vs Dropback Distance
sns.scatterplot(data=merged_data, x='dropback_distance', y='tr_play_min_separation', ax=axes[0,1], color='#FF6B6B', alpha=0.6, label='Targeted Receiver')
sns.scatterplot(data=merged_data, x='dropback_distance', y='qb_play_min_separation', ax=axes[0,1], color='#4ECDC4', alpha=0.6, label='Quarterback')
axes[0,1].set_title('Min Separation vs Dropback Distance (TR & QB)', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('Dropback Distance (yards)', fontsize=12)      
axes[0,1].set_ylabel('Min Separation (yards)', fontsize=12)
axes[0,1].grid(True, alpha=0.3)
axes[0,1].legend()

# Box plots: Targeted Receiver min separation vs Route Category
sns.boxplot(data=merged_data, x='route_of_targeted_receiver', y='tr_play_min_separation', ax=axes[1,0], palette='Set3')
axes[1,0].set_title('Targeted Receiver Min Separation by Route Category', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Route Category', fontsize=12) 
axes[1,0].set_ylabel('Targeted Receiver Min Separation (yards)', fontsize=12)   
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].grid(True, alpha=0.3)

# Box plots: QB min separation vs route Category
sns.boxplot(data=merged_data, x='route_of_targeted_receiver', y='qb_play_min_separation', ax=axes[1,1], palette='Set3')
axes[1,1].set_title('Quarterback Min Separation by Route Category', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('Route Category', fontsize=12) 
axes[1,1].set_ylabel('Quarterback Min Separation (yards)', fontsize=12)   
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].grid(True, alpha=0.3) 

# Box plots: Targeted Receiver min separation vs defenders_in_the_box
sns.boxplot(data=merged_data, x='defenders_in_the_box', y='tr_play_min_separation', ax=axes[2,0], palette='Set2')
axes[2,0].set_title('Targeted Receiver Min Separation by Defenders in the Box', fontsize=14, fontweight='bold')
axes[2,0].set_xlabel('Defenders in the Box', fontsize=12)   
axes[2,0].set_ylabel('Targeted Receiver Min Separation (yards)', fontsize=12)
axes[2,0].grid(True, alpha=0.3) 

# Box plots: QB min separation vs defenders_in_the_box  
sns.boxplot(data=merged_data, x='defenders_in_the_box', y='qb_play_min_separation', ax=axes[2,1], palette='Set2')
axes[2,1].set_title('Quarterback Min Separation by Defenders in the Box', fontsize=14, fontweight='bold')
axes[2,1].set_xlabel('Defenders in the Box', fontsize=12)
axes[2,1].set_ylabel('Quarterback Min Separation (yards)', fontsize=12)
axes[2,1].grid(True, alpha=0.3) 

# Box plots: Targeted Receiver min separation vs offense_formation
sns.boxplot(data=merged_data, x='offense_formation', y='tr_play_min_separation', ax=axes[3,0], palette='Set1')
axes[3,0].set_title('Targeted Receiver Min Separation by Offensive Formation', fontsize=14, fontweight='bold')
axes[3,0].set_xlabel('Offensive Formation', fontsize=12)
axes[3,0].set_ylabel('Targeted Receiver Min Separation (yards)', fontsize=12)
axes[3,0].tick_params(axis='x', rotation=45)
axes[3,0].grid(True, alpha=0.3)     
# Box plots: QB min separation vs offense_formation
sns.boxplot(data=merged_data, x='offense_formation', y='qb_play_min_separation', ax=axes[3,1], palette='Set1')
axes[3,1].set_title('Quarterback Min Separation by Offensive Formation', fontsize=14, fontweight='bold')
axes[3,1].set_xlabel('Offensive Formation', fontsize=12)        
axes[3,1].set_ylabel('Quarterback Min Separation (yards)', fontsize=12)
axes[3,1].tick_params(axis='x', rotation=45)
axes[3,1].grid(True, alpha=0.3)

# Box plots: Targeted Receiver min separation vs play_action
sns.boxplot(data=merged_data, x='play_action', y='tr_play_min_separation', ax=axes[4,0], palette='viridis')
axes[4,0].set_title('Targeted Receiver Min Separation by Play Action', fontsize=14, fontweight='bold')
axes[4,0].set_xlabel('Play Action', fontsize=12)
axes[4,0].set_ylabel('Targeted Receiver Min Separation (yards)', fontsize=12)
axes[4,0].grid(True, alpha=0.3)

# Box plots: QB min separation vs play_action
sns.boxplot(data=merged_data, x='play_action', y='qb_play_min_separation', ax=axes[4,1], palette='viridis')
axes[4,1].set_title('Quarterback Min Separation by Play Action', fontsize=14, fontweight='bold')
axes[4,1].set_xlabel('Play Action', fontsize=12)
axes[4,1].set_ylabel('Quarterback Min Separation (yards)', fontsize=12)
axes[4,1].grid(True, alpha=0.3)

# Apply consistent styling to all subplots
for ax in axes.flat:
    ax.set_facecolor('#f8f9fa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(pad=3.0)
plt.show()
# plt.close()  # Properly close figure

print("✓ All comparative visualizations completed successfully")
plt.savefig('comparative_plots.png', dpi=150, bbox_inches='tight')  # Save to file

