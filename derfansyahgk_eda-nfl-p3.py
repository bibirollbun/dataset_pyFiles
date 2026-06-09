#part 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#part 1
import os
import json
import random
import torch
from datetime import datetime


#part 2
import glob
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import euclidean


#extra for non installed
from scipy.spatial.distance import cdist
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold


#other
from scipy import stats
import joblib
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss


#part 3
import warnings
warnings.filterwarnings('ignore')


#import gc and other
from IPython.display import HTML
from IPython.display import Image as IPImage
from PIL import Image #Image downloader
import gc


#seed setting
SEED = 36
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Initiating full mode
MODE = 'full'  # Change to 'full' for production
initial_path = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final'
main_o_path = '/kaggle/working' #main output


# Create output directory
os.makedirs( main_o_path, exist_ok=True)
os.makedirs(f'{main_o_path}/artifacts', exist_ok=True)
os.makedirs(f'{main_o_path}/media', exist_ok=True)


# Initiating Log run manifest
manifest = {
    'project': 'NFL Big Data Bowl 2026 - CIS Suite',
    'mode': MODE,
    'seed': SEED,
    'timestamp': datetime.now().isoformat(),
    'environment': {
        'platform': 'Kaggle',
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None
    }
}


#gc intermezzo
gc.collect()


# the json initiation
with open(f'{main_o_path}/run_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
#finishing setups
print(f"✅ Setup complete! Mode: {MODE}, Seed: {SEED}")
print(f"Manifest saved to {main_o_path}/run_manifest.json")
print(f"GPU: {manifest['environment']['gpu'] if 'gpu' in manifest['environment'] else 'CPU'}")


#path initiation
train_path = f"{initial_path}/train"
supplementary_path = f"{initial_path}/supplementary_data.csv"


# Load Supplementary data
print("Loading supplementary data...")
supp_df = pd.read_csv(supplementary_path) #dataframe for supplementary
print(f"Supplementary loaded: {supp_df.shape[0]} rows, {supp_df.shape[1]} cols")


#declaring input_path 
input_df_paths = f"{train_path}/input_2023_w*.csv"
# Load input tracking data (multiple weeks)
input_files = glob.glob(input_df_paths)
print(f"Found {len(input_files)} input files")
input_dfs = [pd.read_csv(f) for f in input_files]
input_df = pd.concat(input_dfs, ignore_index=True, copy=True, sort=True)
print(f"Input tracking loaded: {input_df.shape[0]} rows, {input_df.shape[1]} cols")


#declaring output path
output_df_paths = f"{train_path}/output_2023_w*.csv"
# Load output tracking data (multiple weeks)
output_files = glob.glob(output_df_paths)
print(f"Found {len(output_files)} output files")
output_dfs = [pd.read_csv(f) for f in output_files]
output_df = pd.concat(output_dfs, ignore_index=True, copy=True, sort=True)
print(f"Output tracking loaded: {output_df.shape[0]} rows, {output_df.shape[1]} cols")


#inspecting length of line
print(f"the length in input is {len(input_dfs)}") #input
print(f"the length in output is {len(output_dfs)}")


#extra gc
gc.collect()


#data type converter in output 
output_df['game_id'] = output_df['game_id'].astype(str)
output_df['play_id'] = output_df['play_id'].astype(str)


# Merge supplementary with input on game_id and play_id
print("Merging supplementary with input...")
merged_df = pd.merge(input_df, supp_df, on=['game_id', 'play_id'], how='left', copy = True)
print(f"Merged data: {merged_df.shape[0]} rows")


#emergency gc
gc.collect()


# Basic validation: Check for NaNs in critical columns
critical_cols = ['ball_land_x', 'ball_land_y', 'player_side', 'pass_result']
#inspection in any missing columns
nan_summary = merged_df[critical_cols].isnull().sum()
print("NaN counts in critical columns:")
print(nan_summary)


#declaring critical numerical
critical_nums = [critical_cols[0], critical_cols[1]]
#Impute missing numerical columns by interpolation in 'ball landing'
merged_df[critical_nums] = merged_df[critical_nums].interpolate(method='linear', limit_direction='forward')
merged_df[critical_nums] = merged_df[critical_nums].interpolate(method='linear', limit_direction='backward')


#filling string type
merged_df['pass_result'] = merged_df['pass_result'].fillna('I')  # Default to incomplete
merged_df['player_side'] = merged_df['player_side'].fillna('Unidentified')


#critical condition the second
crit_col_2 = ['x','y']
# column inspection in any missing columns for output
nan_summary_2 = output_df[crit_col_2].isnull().sum()
print("NaN counts in critical columns:")
print(nan_summary_2)


#initiating gc
gc.collect()


#intial data analysis for categories
def full_df_maker(df_input, support):
    #length declaraiton
    [short, middle, long] = [5, 10, 15]
    #filtering data in supplementary
    conditions = [
    support['pass_length'] < 5,
    (support['pass_length'] >= 5) & (support['pass_length'] <= 10),
    (support['pass_length'] >= 10) & (support['pass_length'] <= 15),
    support['pass_length'] > 15
]
    #declaring categories
    categories = ['close throw','short','intermediate', 'long']
    #creating pass type for
    support['pass_type'] = np.select(
        conditions, 
        categories, 
        default='unknown'
    )
    #listing candidates to be merged
    full_input_df =  pd.merge(
        df_input,                 # The left DataFrame (the one receiving the new column)
        support[['game_id', 'play_id', 'route_of_targeted_receiver']],  # The right DataFrame (only the key columns and the column to add)
        on=['game_id', 'play_id'],  # The key columns to match on
        how='left',           # Specifies a left join
        copy = True
    )
    return full_input_df


#execution to generate input (for all week)
initial_merge = full_df_maker(input_df, supp_df)


#inspecting merging result
initial_merge.info()


#extra gc
gc.collect()


#absolute value
def vectorized_distance(tr_x, tr_y, def_x, def_y):
    powered_r = ((def_x - tr_x)**2) + ((def_y - tr_y)**2)
    rval = np.sqrt(powered_r)
    del powered_r
    return rval #to the next funciton


#function to implement 'rval' in analysis
def calculate_frame_level_separation(df):
    # Group by game, play, and frame (FRAME-LEVEL calculation)
    grouped = df.groupby(['game_id', 'play_id','frame_id'])
    
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
            distances = vectorized_distance(tr_x, tr_y, def_x, def_y)
            
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


#agregate tracking device
def tr_calculate_play_level_aggregates(df):
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
        suffixes=('_first', '_last'), copy = True
    )
     # merge to play_aggregates
    play_aggregates = pd.merge(
        play_aggregates,
        first_last,
        on=['game_id', 'play_id', 'nfl_id'],
        how='left'
    )
    
    return play_aggregates


#calculation in quarterback
def calculate_frame_level_separation_qb(df):
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
            distances = vectorized_distance(qb_x, qb_y, def_x, def_y)
            
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
#to aggregate


#qb aggregate for analysis
def qb_aggregate_calculation(df):
    #filtering qb
    qbres = df[df['player_role'] == 'Passer'].copy()
    # Remove rows where separation couldn't be calculated
    qbres = qbres[qbres['qb_min_separation'].notna()]
    #checking dataframe
    if qbres.empty:
        return pd.DataFrame()
    #calculate aggregate before grouping
    play_aggregates = qbres.groupby(['game_id','play_id']).agg({
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
    }).copy().reset_index()
    #flatten column names
    play_aggregates.columns = ['game_id', 'play_id',
                                'qb_play_min_separation', 
                                'qb_play_avg_separation', 
                                'qb_play_var_separation',
                                'player_name', 'nfl_id', 'player_role',
                                'time_to_throw']
    
    # returning the result
    
    return play_aggregates


# emergency gc
gc.collect()


#executing calculation in minimum to calculate frame-level minimum separation
print(f"\n{'='*70}")
print("Next step: CALCULATING FRAME-LEVEL MINIMUM SEPARATIONS")
print("Processing all frames using vectorized distance calculations...")
tr_separation_df = calculate_frame_level_separation(initial_merge)


#initiating gc space
gc.collect()


#execution to generate tr_calculation
tr_play_level_agg = pd.DataFrame()
print(f"\n{'='*70}")
print("Comming up: CALCULATING PLAY-LEVEL AGGREGATE STATISTICS")
print("Aggregating: min, avg, variance across all frames per play...")
tr_play_level_agg = tr_calculate_play_level_aggregates(tr_separation_df)


#next step : gc before quarte back
gc.collect()


#quarterback inspection to calculate frame-level minimum separation for Quarterbacks
print(f"\n{'='*70}")
print("Following: CALCULATING FRAME-LEVEL MINIMUM SEPARATIONS FOR QUARTERBACKS")
print("Processing all frames using vectorized distance calculations...")
qb_separation_df = calculate_frame_level_separation_qb(initial_merge)


#next step : qb analysis to calculate play-level aggregates for Quarterbacks
qb_play_level_agg = pd.DataFrame()

print(f"\n{'='*70}")
print("The next step: CALCULATING QB PLAY-LEVEL AGGREGATE STATISTICS")
print("="*70)
print("Aggregating: min, avg, variance across all frames per play...")
qb_play_level_agg = qb_aggregate_calculation(qb_separation_df)


#extra gc
gc.collect()


#data merging for 'visual inspection' (column declaraiton)
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


#identification for similar codes
similar_ids = ['game_id', 'play_id']


#initiating upadated_play 
all_params = similar_ids + play_columns


#another gc
gc.collect()


# real merging in qb analytics
play_level_qb_tr = pd.merge(qb_play_level_agg, tr_play_level_agg, on = similar_ids, how = 'left', suffixes=('_qb', '_tr'))


#selecting the final result
qb_supplementary_data = supp_df[all_params]


#inititating merged data 
raw_data = pd.merge(play_level_qb_tr, qb_supplementary_data, on=['game_id', 'play_id'], how='left')


#displaying result
display(raw_data.head(6))


#extra gc
gc.collect()


# Get unique routes and create subplot layout
unique_routes = play_level_qb_tr['route_of_targeted_receiver'].dropna().unique().copy()


#declaring total routes
n_routes = len(unique_routes)
#inspection
print(f"there are {n_routes} for all week")


#grid calculation
n_cols = int(np.ceil(np.sqrt(n_routes))) #columns
n_rows = int(np.ceil(n_routes / n_cols)) #rows
#inspection
print(f"there are {n_cols} columns & {n_rows} rows for all week")


# preparation in qb metric
raw_data['qb_play_min_separation'] = pd.to_numeric(raw_data['qb_play_min_separation'], errors='coerce')


# Remove rows with NaN values in the separation column for plotting
plot_data = raw_data.dropna(subset=['qb_play_min_separation'])
#results inspection
print(f"Original data: {len(raw_data)} rows")
print(f"Data for plotting (after removing NaN): {len(plot_data)} rows")
print(f"Data type of qb_play_min_separation: {plot_data['qb_play_min_separation'].dtype}")


#subplots for frame chart
fig = plt.figure(figsize=(4*n_cols, 4*n_rows))
fig.suptitle('Targeted Receiver Last Frame Direction Distribution by Route', fontsize=16, fontweight='bold', y=0.98)
# Define colors for each route
colors = plt.cm.Set3(np.linspace(0, 1, n_routes))
#iteration to identify link
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
plt.close()

#visualization section
# Set up the plotting style for better aesthetics
plt.style.use('default')
sns.set_palette("husl")

# Create figure with 1x4 layout
fig, axes = plt.subplots(1, 4, figsize=(20, 8))

# Define a professional color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#4A90A4', '#83A95C', '#F4B942']

# Plot 1: Time to Throw vs Route (horizontal boxplot)
sns.boxplot(data=raw_data, y='route_of_targeted_receiver', x='time_to_throw', 
            ax=axes[0], palette=colors, orient='h')
axes[0].set_title('Time to Throw by Route Category', fontsize=14, fontweight='bold', pad=20)
axes[0].set_ylabel('Route Category', fontsize=14, fontweight='semibold')
axes[0].set_xlabel('Time to Throw (seconds)', fontsize=10, fontweight='semibold')
axes[0].tick_params(axis='y', labelsize=12)
axes[0].tick_params(axis='x', labelsize=12)
axes[0].grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
axes[0].set_facecolor('#fafafa')

# Plot 2: Final Separation vs Route (horizontal boxplot) - Remove Y labels
sns.boxplot(data=raw_data, y='route_of_targeted_receiver', x='tr_min_separation_last',
            ax=axes[1], palette=colors, orient='h')
axes[1].set_title('TR Final Separation by Route Category', fontsize=14, fontweight='bold', pad=20)
axes[1].set_ylabel('')  # Remove ylabel
axes[1].set_xlabel('TR Final Separation (yards)', fontsize=10, fontweight='semibold')
axes[1].tick_params(axis='y', labelleft=False)  # Hide y-axis labels
axes[1].tick_params(axis='x', labelsize=12)
axes[1].grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
axes[1].set_facecolor('#fafafa')

# Plot 3: Pass Length vs Route (horizontal boxplot) - Remove Y labels
sns.boxplot(data=raw_data, y='route_of_targeted_receiver', x='pass_length', 
            ax=axes[2], palette=colors, orient='h')
axes[2].set_title('Pass Length by Route Category', fontsize=14, fontweight='bold', pad=20)
axes[2].set_ylabel('')  # Remove ylabel
axes[2].set_xlabel('Pass Length (yards)', fontsize=10, fontweight='semibold')
axes[2].tick_params(axis='y', labelleft=False)  # Hide y-axis labels
axes[2].tick_params(axis='x', labelsize=12)
axes[2].grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
axes[2].set_facecolor('#fafafa')
# Plot 4: Dropback Distance vs Route (horizontal boxplot) - Remove Y labels
sns.boxplot(data=raw_data, y='route_of_targeted_receiver', x='dropback_distance', 
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
plt.savefig(f'{main_o_path}/media/route_analysis_horizontal.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()
plt.close()


#throw analysis: Create figure with 1x3 layout
# Add pass lenngth per pass type in plot y axis along with pass type

fig, axes = plt.subplots(1, 3, figsize=(15, 8))    
# Define a professional color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#4A90A4', '#83A95C', '#F4B942']
# Create custom y-axis labels with yards range
pass_type_labels = {
    'close throw' : 'Close throw (≤5 yds)',
    'short': 'Short (>5 yds and < 10m)',
    'intermediate': 'Intermediate (>5-15 yds)', 
    'long': 'Long (>15 yds)'
}
# Plot 1: Time to Throw vs Pass Type (horizontal boxplot)
sns.boxplot(data=raw_data, y='pass_type', x='time_to_throw', 
            ax=axes[0], palette=colors, orient='h')
axes[0].set_title('Time to Throw by Pass Type', fontsize=14, fontweight='bold', pad=20)
axes[0].set_ylabel('Pass Type', fontsize=14, fontweight='semibold')
axes[0].set_xlabel('Time to Throw (seconds)', fontsize=10, fontweight='semibold')
axes[0].tick_params(axis='y', labelsize=12)
axes[0].tick_params(axis='x', labelsize=12)
axes[0].grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
axes[0].set_facecolor('#fafafa')
# Plot 2: Final Separation vs Pass Type (horizontal boxplot) - Remove Y labels
sns.boxplot(data=raw_data, y='pass_type', x='tr_min_separation_last',
            ax=axes[1], palette=colors, orient='h')
axes[1].set_title('TR Final Separation by Pass Type', fontsize=14, fontweight='bold', pad=20)
axes[1].set_ylabel('')  # Remove ylabel
axes[1].set_xlabel('TR Final Separation (yards)', fontsize=10, fontweight='semibold')
axes[1].tick_params(axis='y', labelleft=False)  # Hide y-axis labels
axes[1].tick_params(axis='x', labelsize=12)
axes[1].grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
axes[1].set_facecolor('#fafafa')
# Plot 3: Dropback Distance vs Pass Type (horizontal boxplot) - Remove Y labels
sns.boxplot(data=raw_data, y='pass_type', x='dropback_distance', 
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
plt.savefig(f'{main_o_path}/media/pass_type_analysis_horizontal.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()
plt.close()


#visualisation in qb analysis so Each box plot will compare qb_play_min_separation against different categorical variables.
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
route_counts = raw_data['route_of_targeted_receiver'].value_counts()

for route in raw_data['route_of_targeted_receiver'].dropna().unique():
    route_data = raw_data[raw_data['route_of_targeted_receiver'] == route]
    print(f"\n{route.upper()} (n={len(route_data)}):")
    print(f"  Time to Throw: {route_data['time_to_throw'].mean():.2f}s ± {route_data['time_to_throw'].std():.2f}s")
    print(f"  Final Separation: {route_data['tr_min_separation_last'].mean():.2f} ± {route_data['tr_min_separation_last'].std():.2f} yards")
    print(f"  Pass Length: {route_data['pass_length'].mean():.2f} ± {route_data['pass_length'].std():.2f} yards")
    print(f"  Dropback Distance: {route_data['dropback_distance'].mean():.2f} ± {route_data['dropback_distance'].std():.2f} yards")



def analyze_route_impact(merged_data):
    route_analysis = np.round(merged_data.groupby('route_of_targeted_receiver').agg({
        'tr_play_avg_separation': ['mean', 'std'],
        'qb_play_min_separation': ['mean', 'std'],
        'time_to_throw': ['mean'],
        'pass_length': ['mean', 'std'],
        'pass_result': lambda x: (x == 'C').mean()  # Completion rate
    }),5)
    return route_analysis


#execution in route
route_impact = analyze_route_impact(raw_data)
print(f"\n{'='*70}")
print(route_impact) 


#selecting columns 
def column_selection (prefix, basic_path, selected_columns):
    #declaring all dfs
    all_dfs = []
    pattern = os.path.join(basic_path, f'{file_prefix}_2023_w*.csv')
    file_paths = glob.glob(pattern)
    #checking if it fails or not
    if not file_paths:
        raise ValueError(f"No files found matching pattern: {file_pattern}. Please verify the path and file structure.")
        
    print(f"Found {len(file_paths)} files for prefix '{file_prefix}'. Loading...")
    
    for file_path in file_paths:
        try:
            # We use low_memory=False to ensure correct dtype handling across files
            all_dfs.append(pd.read_csv(file_path, usecols=use_cols, low_memory=False))
        except ValueError as e:
            # Catch the missing column error specifically.
            print(f"Error loading {file_path}. Check if all 'usecols' are present in the file. Error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while loading {file_path}: {e}")
            raise
    finished_df = pd.concat(all_dfs, ignore_index=True, copy = True, sort = True)
    return finished_df


# Define needed columns
output_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y']
input_cols = ['game_id', 'play_id', 'nfl_id', 'player_side', 'player_role', 'ball_land_x', 'ball_land_y', 's', 'a', 'dir'] # output
supp_cols = ['game_id', 'play_id', 'pass_result', 'team_coverage_man_zone', 'expected_points_added']


#declaring selected columns (input)
selected_input = column_selection('input', train_path, input_cols)


#selection in output
selected_output = column_selection('output', train_path, output_cols)


#uploaidng in supplementary
selected_supp = pd.read_csv(supplementary_path, usecols=supp_cols)


#filtering result
def data_preparation(dfin, dfout, support):
    player_info_df = dfin.drop_duplicates(subset=['game_id', 'play_id', 'nfl_id']).copy()
    tracking_data = dfout.merge(player_info_df.drop(columns=['ball_land_x', 'ball_land_y', 's', 'a', 'dir']), 
                                on=['game_id', 'play_id', 'nfl_id'], 
                                how='left')
    #initiating merge
    ball_land_df = dfin[['game_id', 'play_id', 'ball_land_x', 'ball_land_y']].dropna().drop_duplicates()
    tracking_data = tracking_data.merge(ball_land_df, on=['game_id', 'play_id'], how='left')
    tracking_data = tracking_data.merge(support, on=['game_id', 'play_id'], how='left')
    # 3. Final filtering for analysis scope (Completed, Incomplete, or Intercepted passes that have ball landing data)
    tracking_data = tracking_data[tracking_data['pass_result'].isin(['C', 'I', 'IN'])].dropna(subset=['ball_land_x'])
    tracking_data.dropna(subset=['nfl_id'], inplace=True) 
    return tracking_data 


#execution in data preparation
next_step = data_preparation(selected_input, selected_output, selected_supp)


#inspection in finalized df
df_shape = next_step[['game_id', 'play_id']].copy().drop_duplicates().shape[0]
#data summary
print(f"\n--- Data Loading Summary ---")
print(f"Total plays for analysis: {df_shape}")
print(f"Total frames after filtering: {len(next_step)}")


# the function
def find_nearest_defender(frame_group):
    # Check for plays/frames where the receiver or defenders are missing
    if tr_df.empty or defender_df.empty:
        # Return the original group so processing continues without error
        return frame_group
        
    # TR Position (1x2 array)
    tr_pos = tr_df[['x', 'y']].values[0].reshape(1, 2)
    defender_pos = defender_df[['x', 'y']].values
    
    # Calculate all distances efficiently (Receiver-to-all-Defenders)
    distances = cdist(tr_pos, defender_pos, metric='euclidean')
    
    # Find the minimum distance and the index of the closest defender
    min_dist_index = np.argmin(distances)
    min_dist = distances[0, min_dist_index]
    nearest_defender_nflid = defender_df.iloc[min_dist_index]['nfl_id']
    
    # Assign the results to ALL rows in the frame group (for easy filtering later)
    frame_group['min_sep_distance'] = min_dist
    frame_group['nearest_defender_nflid'] = nearest_defender_nflid
    
    return frame_group


#execution in function
print("Starting Nearest Defender Calculation (This may take several minutes on the full dataset)...")
# Apply the function across all frames in all plays
tracking_data_processed = (tracking_data.groupby(['game_id', 'play_id', 'frame_id'])
                                        .apply(find_nearest_defender)
                                        .reset_index(drop=True))


#filtering 'essential info' for data 
filtered_result = tracking_data_processed[
    (tracking_data_processed['nfl_id'] == tracking_data_processed['nearest_defender_nflid']) | 
    (tracking_data_processed['player_role'] == 'Targeted Receiver')
].copy()


#first gc
gc.collect()


#input file separation
[input_01, input_02, input_03, input_04, input_05, input_06, input_07, input_08, input_09, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18] = input_dfs


#input check after separation
input_01.info()


#output separation
[output_01, output_02, output_03, output_04, output_05, output_06, output_07, output_08, output_09, output_10, output_11, output_12, output_13, output_14, output_15, output_16, output_17, output_18] = output_dfs


#result check after separation
output_01.info()


#declaring save files
save_dir = "frames"
os.makedirs(save_dir, exist_ok=True)


#early gc
gc.collect()


# function to pick longest play
def play_picker(data):
    # Selected pick
    example_play = data.groupby(['game_id', 'play_id']).size().reset_index(name='n_frames')
    example_play = example_play.sort_values('n_frames', ascending=False).iloc[0]  # longest play
    game_id, play_id = example_play['game_id'], example_play['play_id']
    
    print(f"Selected example play: Game {game_id}, Play {play_id}")
    
    play_data = data[(data['game_id'] == game_id) & (data['play_id'] == play_id)].copy()
    play_data = play_data.sort_values('frame_id').reset_index(drop=True)
    return play_data


#result picking in week one
result_w01 = play_picker(input_01)
result_w02 = play_picker(input_02)
result_w03 = play_picker(input_03)
result_w04 = play_picker(input_04)


#result picking in part 2
result_w05 = play_picker(input_05)
result_w06 = play_picker(input_06)
result_w07 = play_picker(input_07)
result_w08 = play_picker(input_08)


#part 3
result_w09 = play_picker(input_09)
result_w10 = play_picker(input_10)
result_w11 = play_picker(input_11)
result_w12 = play_picker(input_12)


#part 4
result_w13 = play_picker(input_13)
result_w14 = play_picker(input_14)
result_w15 = play_picker(input_15)
result_w16 = play_picker(input_16)


#final part
result_w17 = play_picker(input_17)
result_w18 = play_picker(input_18)


#result inspection
result_w01.info()


#gc intermezzo 
gc.collect()


# 3. Draw Football Field Helper
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


#declaration of sub files 'images'
main_img_path = f"{main_o_path}/frames/frames_analytics"
os.makedirs(main_img_path, exist_ok=True)


#function to generate weekly folders
def create_week_folder(week_number):
    week_folder = f"{main_img_path}/week_{week_number:02d}"
    img_folder = f"{week_folder}/images"
    
    #generating weekly folders and images
    os.makedirs(img_folder, exist_ok=True)

    return week_folder, img_folder


#role color declaration
role_colors = {
    'Offense': 'red', 'Defense': 'blue', 'Football': 'gold', 'Ball': 'gold',
    'QB': 'darkred', 'WR': 'orange', 'RB': 'yellow', 'TE': 'goldenrod',
    'OL': 'lightcoral', 'LB': 'lightblue', 'DB': 'cyan', 'DL': 'navy',
    'S': 'deepskyblue', 'CB': 'aqua', 'K': 'purple', 'P': 'violet',
    'Targeted Receiver': 'lime'
}


#image generator
def frame_img_generator(result_data, week_number):

    # Create week folder and image folder
    week_folder, img_folder = create_week_folder(week_number)
    
    game_id = sorted(result_data['game_id'].unique().copy())
    play_id = sorted(result_data['play_id'].unique().copy())
    frame_ids = sorted(result_data['frame_id'].unique().copy())

    for i, frame_id in enumerate(frame_ids):

        frame = result_data[result_data['frame_id'] == frame_id]

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_facecolor('green')

        draw_football_field(ax)

        # Plot players by role
        for role, color in role_colors.items():
            subset = frame[frame['player_role'].fillna('')
                           .str.contains(role.split()[0], case=False)]
            if not subset.empty:
                ax.scatter(subset['x'], subset['y'], s=120,
                           color=color, alpha=0.8, edgecolor='black',
                           label=role)

        # Ball landing point
        if 'ball_land_x' in frame.columns and not pd.isna(frame.iloc[0]['ball_land_x']):
            ax.scatter(frame.iloc[0]['ball_land_x'], frame.iloc[0]['ball_land_y'],
                       s=400, color='yellow', marker='*',
                       edgecolor='black', linewidth=2, label='Ball Landing')

        ax.set_xlim(0, 120)
        ax.set_ylim(0, 53.3)
        ax.set_title(
            f"Game {game_id} | Play {play_id} | Frame {frame_id} | Week {week_number}",
            fontsize=12,
            fontweight='bold'
        )
        ax.legend(loc='upper right', fontsize=8, framealpha=0.7)
        ax.axis('off')

        # Save frame
        plt.savefig(f"{img_folder}/frame_{i:04d}.png",
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    print(f"✅ Generated {len(frame_ids)} frame images for Week {week_number}")
    return week_folder, img_folder


#execution in week one
rep_w01, img_w01 = frame_img_generator(result_w01, 1)


#execution in week two
rep_w02,img_w02 = frame_img_generator(result_w02, 2)


#inspection in field three
rep_w03,img_w03 = frame_img_generator(result_w03, 3)


#inspection in field four
rep_w04,img_w04 = frame_img_generator(result_w04, 4)


#inspecting file in week five
rep_w05, img_w05 = frame_img_generator(result_w05, 5)


#inspecting file in week six
rep_w06, img_w06 = frame_img_generator(result_w06, 6)


#inspecting file in week seven
rep_w07, img_w07 = frame_img_generator(result_w07, 7)


#inspecting file in week eight
rep_w08, img_w08 = frame_img_generator(result_w08, 8)


#Execution in week 10
rep_w10, img_w10 = frame_img_generator(result_w10, 10)


#Execution in week 11
rep_w11, img_w11 = frame_img_generator(result_w11, 11)


#Execution in week 12
rep_w12, img_w12 = frame_img_generator(result_w12, 12)


#initiating gc
gc.collect()


#function to store frames
def gif_maker(result_data, week_number):
    #producing empty frames
    frames = []
    #redeclaration
    frame_ids = sorted(result_data['frame_id'].unique().copy())
    for a in range(len(frame_ids)):
        img_path = f"{main_img_path}/week_{week_number:02d}/images"
        img_file = f"{img_path}/frame_{a:04d}.png" #save page
        if os.path.exists(img_file):
            frames.append(Image.open(img_file))
    #generating gifs
    if frames :
        gif_path = f"{main_img_path}/week_{week_number:02d}"
        finished_animation = frames[0].save(f'{gif_path}/football_tracking_{week_number:02d}.gif', save_all=True,
                   append_images=frames[1:], duration=120, loop=0)
    else:
        print("⚠️ No frames found for animation")

   #returning the result
    return finished_animation


#gc intermezzo
gc.collect()


#execution in week one
gif_w01 = gif_maker(result_w01, 1)


#execution in week 02
gif_w02 = gif_maker(result_w02, 2)


#execution in week 03
gif_w03 = gif_maker(result_w03, 3)

