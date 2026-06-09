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


# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ Libraries loaded successfully")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")


# Define data paths (with subdirectory)
data_path = Path('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final')

# Load supplementary data
print("Loading supplementary data...")
supplementary = pd.read_csv(data_path / 'supplementary_data.csv')
print(f"✓ Supplementary data loaded: {supplementary.shape[0]:,} plays")

# Load all input files (before ball is thrown)
print("\nLoading input files (pre-throw)...")
input_files = sorted((data_path / 'train').glob('input_*.csv'))
input_dfs = []
for file in input_files:
    df = pd.read_csv(file)
    input_dfs.append(df)
    print(f"  ✓ {file.name}: {len(df):,} rows")

input_data = pd.concat(input_dfs, ignore_index=True)
print(f"\n✓ Total input data: {len(input_data):,} rows")

# Load all output files (ball in the air)
print("\nLoading output files (ball in air)...")
output_files = sorted((data_path / 'train').glob('output_*.csv'))
output_dfs = []
for file in output_files:
    df = pd.read_csv(file)
    output_dfs.append(df)
    print(f"  ✓ {file.name}: {len(df):,} rows")

output_data = pd.concat(output_dfs, ignore_index=True)
print(f"\n✓ Total output data: {len(output_data):,} rows")

# Quick summary
print(f"\n{'='*60}")
print(f"Unique games: {supplementary['game_id'].nunique():,}")
print(f"Unique plays: {len(supplementary):,}")
print(f"\nPass results breakdown:")
print(supplementary['pass_result'].value_counts())


# Check output data structure
print("Output data columns:")
print(output_data.columns.tolist())
print(f"\nOutput data shape: {output_data.shape}")
print(f"\nSample of output data:")
print(output_data.head())

# Check input data for ball landing location
print("\n" + "="*60)
print("Input data columns:")
print(input_data.columns.tolist())
print(f"\nChecking for ball_land_x and ball_land_y...")
if 'ball_land_x' in input_data.columns:
    print("✓ Ball landing coordinates found in input data")
    print(f"  Non-null ball_land_x: {input_data['ball_land_x'].notna().sum():,}")
    print(f"  Non-null ball_land_y: {input_data['ball_land_y'].notna().sum():,}")


# Get unique play-level info from input data (take last frame before throw)
print("Extracting play-level metadata from input data...")
play_metadata = input_data.groupby(['game_id', 'play_id', 'nfl_id']).last().reset_index()

# Keep only relevant columns
metadata_cols = ['game_id', 'play_id', 'nfl_id', 'player_name', 'player_role', 
                 'player_side', 'play_direction', 'ball_land_x', 'ball_land_y']
play_metadata = play_metadata[metadata_cols]

print(f"✓ Play metadata extracted: {len(play_metadata):,} player-play combinations")

# Merge output data with metadata
print("\nMerging output tracking with metadata...")
tracking = output_data.merge(play_metadata, on=['game_id', 'play_id', 'nfl_id'], how='left')
print(f"✓ Merged data: {len(tracking):,} rows")

# Filter to only relevant players (Targeted Receiver and Defensive Coverage)
print("\nFiltering to relevant players...")
print(f"Player roles in data:")
print(tracking['player_role'].value_counts())

tracking_filtered = tracking[tracking['player_role'].isin(['Targeted Receiver', 'Defensive Coverage'])].copy()
print(f"\n✓ Filtered to {len(tracking_filtered):,} rows")
print(f"  Targeted Receiver frames: {len(tracking_filtered[tracking_filtered['player_role']=='Targeted Receiver']):,}")
print(f"  Defensive Coverage frames: {len(tracking_filtered[tracking_filtered['player_role']=='Defensive Coverage']):,}")

# Check for missing values
print(f"\nMissing ball landing coordinates: {tracking_filtered[['ball_land_x', 'ball_land_y']].isna().sum().sum()}")


# Calculate distance from each player to ball landing point
print("Calculating distances to ball landing point...")
tracking_filtered['dist_to_ball'] = np.sqrt(
    (tracking_filtered['x'] - tracking_filtered['ball_land_x'])**2 + 
    (tracking_filtered['y'] - tracking_filtered['ball_land_y'])**2
)

print(f"✓ Distances calculated")
print(f"  Mean distance: {tracking_filtered['dist_to_ball'].mean():.2f} yards")
print(f"  Median distance: {tracking_filtered['dist_to_ball'].median():.2f} yards")

# For each play and frame, find the receiver distance and nearest defender distance
print("\nCalculating ACD for each play-frame...")

# Get receiver distances
receiver_dist = tracking_filtered[tracking_filtered['player_role'] == 'Targeted Receiver'].copy()
receiver_dist = receiver_dist[['game_id', 'play_id', 'frame_id', 'dist_to_ball']].rename(
    columns={'dist_to_ball': 'receiver_dist'}
)

# Get nearest defender distance per play-frame
defender_dist = tracking_filtered[tracking_filtered['player_role'] == 'Defensive Coverage'].copy()
nearest_defender = defender_dist.groupby(['game_id', 'play_id', 'frame_id'])['dist_to_ball'].min().reset_index()
nearest_defender = nearest_defender.rename(columns={'dist_to_ball': 'nearest_defender_dist'})

# Merge to calculate ACD
acd_data = receiver_dist.merge(nearest_defender, on=['game_id', 'play_id', 'frame_id'], how='inner')
acd_data['ACD'] = acd_data['nearest_defender_dist'] - acd_data['receiver_dist']

print(f"✓ ACD calculated for {len(acd_data):,} play-frames")
print(f"\nACD Statistics:")
print(f"  Mean ACD: {acd_data['ACD'].mean():.2f} yards")
print(f"  Median ACD: {acd_data['ACD'].median():.2f} yards")
print(f"  Std Dev: {acd_data['ACD'].std():.2f} yards")
print(f"\nACD Distribution:")
print(f"  Positive (receiver closer): {(acd_data['ACD'] > 0).sum():,} ({(acd_data['ACD'] > 0).mean()*100:.1f}%)")
print(f"  Negative (defender closer): {(acd_data['ACD'] < 0).sum():,} ({(acd_data['ACD'] < 0).mean()*100:.1f}%)")
print(f"  Near zero (±1 yard): {(acd_data['ACD'].abs() <= 1).sum():,} ({(acd_data['ACD'].abs() <= 1).mean()*100:.1f}%)")


# Merge ACD data with supplementary data to get pass results
print("Merging ACD data with pass results...")
acd_with_results = acd_data.merge(
    supplementary[['game_id', 'play_id', 'pass_result', 'play_description']], 
    on=['game_id', 'play_id'], 
    how='left'
)

print(f"✓ Merged: {len(acd_with_results):,} rows")
print(f"\nPass result distribution in ACD data:")
print(acd_with_results.groupby('pass_result').size())

# Calculate play-level ACD statistics
print("\nCalculating play-level ACD metrics...")
play_acd_stats = acd_with_results.groupby(['game_id', 'play_id', 'pass_result']).agg({
    'ACD': ['mean', 'min', 'max', 'std'],
    'frame_id': 'max'  # Number of frames
}).reset_index()

play_acd_stats.columns = ['game_id', 'play_id', 'pass_result', 'mean_acd', 'min_acd', 'max_acd', 'std_acd', 'num_frames']

print(f"✓ Play-level stats calculated for {len(play_acd_stats):,} plays")

# Compare ACD by outcome
print("\nMean ACD by Pass Result:")
outcome_summary = play_acd_stats.groupby('pass_result')['mean_acd'].describe()
print(outcome_summary)


# Create distribution plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Distribution of mean ACD by outcome
ax1 = axes[0]
for outcome in ['C', 'I', 'IN']:
    data = play_acd_stats[play_acd_stats['pass_result'] == outcome]['mean_acd']
    ax1.hist(data, bins=50, alpha=0.6, label=f'{outcome} (n={len(data)})', edgecolor='black')

ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero (Even Control)')
ax1.set_xlabel('Mean ACD (yards)', fontsize=12)
ax1.set_ylabel('Number of Plays', fontsize=12)
ax1.set_title('Distribution of Mean ACD by Pass Outcome', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Box plot
ax2 = axes[1]
outcome_data = [
    play_acd_stats[play_acd_stats['pass_result'] == 'C']['mean_acd'],
    play_acd_stats[play_acd_stats['pass_result'] == 'I']['mean_acd'],
    play_acd_stats[play_acd_stats['pass_result'] == 'IN']['mean_acd']
]
bp = ax2.boxplot(outcome_data, labels=['Complete', 'Incomplete', 'Interception'],
                  patch_artist=True, showmeans=True)

# Color the boxes
colors = ['#2ecc71', '#f39c12', '#e74c3c']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero (Even Control)')
ax2.set_ylabel('Mean ACD (yards)', fontsize=12)
ax2.set_title('Mean ACD by Pass Outcome', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3, axis='y')
ax2.legend()

plt.tight_layout()
plt.show()

print("\nKey Insight:")
print("Completions show consistent positive ACD (receiver controls airspace)")
print("Interceptions show negative ACD (defenders gain control early)")
print("Incompletions are mixed - some contested, some just missed connections")


# Function to find Point of No Return for each play
def find_point_of_no_return(group):
    """
    Find the frame where ACD control no longer changes hands.
    Returns None if no sign change occurs (dominated play).
    """
    acd_values = group['ACD'].values
    frames = group['frame_id'].values
    
    if len(acd_values) < 2:
        return None
    
    # Track sign changes
    signs = np.sign(acd_values)
    sign_changes = np.where(np.diff(signs) != 0)[0]
    
    if len(sign_changes) == 0:
        # No sign change - dominated throughout
        return None
    else:
        # Return the frame AFTER the last sign change
        last_change_idx = sign_changes[-1]
        if last_change_idx + 1 < len(frames):
            return frames[last_change_idx + 1]
        else:
            return frames[-1]

print("Calculating Point of No Return for each play...")
pnr_results = acd_with_results.groupby(['game_id', 'play_id']).apply(find_point_of_no_return).reset_index()
pnr_results.columns = ['game_id', 'play_id', 'pnr_frame']

# Merge back with play stats
play_acd_stats = play_acd_stats.merge(pnr_results, on=['game_id', 'play_id'], how='left')

print(f"✓ PNR calculated for {len(play_acd_stats):,} plays")
print(f"\nPNR Statistics:")
print(f"  Plays with PNR: {play_acd_stats['pnr_frame'].notna().sum():,} ({play_acd_stats['pnr_frame'].notna().mean()*100:.1f}%)")
print(f"  Plays dominated (no PNR): {play_acd_stats['pnr_frame'].isna().sum():,} ({play_acd_stats['pnr_frame'].isna().mean()*100:.1f}%)")

# For plays with PNR, calculate when it occurred (as % of total frames)
plays_with_pnr = play_acd_stats[play_acd_stats['pnr_frame'].notna()].copy()
plays_with_pnr['pnr_pct'] = (plays_with_pnr['pnr_frame'] / plays_with_pnr['num_frames']) * 100

print(f"\nPNR Timing (for plays with sign changes):")
print(f"  Mean PNR occurs at: {plays_with_pnr['pnr_pct'].mean():.1f}% of ball flight")
print(f"  Median PNR occurs at: {plays_with_pnr['pnr_pct'].median():.1f}% of ball flight")

# Compare PNR by outcome
print(f"\nPNR by Outcome:")
for outcome in ['C', 'I', 'IN']:
    outcome_data = plays_with_pnr[plays_with_pnr['pass_result'] == outcome]
    pct_with_pnr = len(outcome_data) / len(play_acd_stats[play_acd_stats['pass_result'] == outcome]) * 100
    mean_pnr_timing = outcome_data['pnr_pct'].mean() if len(outcome_data) > 0 else 0
    print(f"  {outcome}: {pct_with_pnr:.1f}% have PNR (avg at {mean_pnr_timing:.1f}% of flight)")


# Function to find interesting example plays
def select_example_plays(acd_with_results, play_acd_stats):
    """Select 3-5 representative plays for visualization"""
    examples = []
    
    # 1. Clean completion - receiver dominated (high positive ACD, no PNR)
    clean_complete = play_acd_stats[
        (play_acd_stats['pass_result'] == 'C') & 
        (play_acd_stats['mean_acd'] > 4) & 
        (play_acd_stats['pnr_frame'].isna())
    ].head(1)
    if len(clean_complete) > 0:
        examples.append(('Clean Completion', clean_complete.iloc[0]))
    
    # 2. Contested completion - sign changed but receiver won
    contested_complete = play_acd_stats[
        (play_acd_stats['pass_result'] == 'C') & 
        (play_acd_stats['pnr_frame'].notna()) &
        (play_acd_stats['mean_acd'] > 0)
    ].head(1)
    if len(contested_complete) > 0:
        examples.append(('Contested Completion', contested_complete.iloc[0]))
    
    # 3. Interception - defender took control
    interception = play_acd_stats[
        (play_acd_stats['pass_result'] == 'IN') & 
        (play_acd_stats['mean_acd'] < -1)
    ].head(1)
    if len(interception) > 0:
        examples.append(('Interception', interception.iloc[0]))
    
    # 4. Close incompletion - truly contested
    close_incomplete = play_acd_stats[
        (play_acd_stats['pass_result'] == 'I') & 
        (play_acd_stats['mean_acd'].abs() < 1) &
        (play_acd_stats['pnr_frame'].notna())
    ].head(1)
    if len(close_incomplete) > 0:
        examples.append(('True 50-50 Incompletion', close_incomplete.iloc[0]))
    
    return examples

print("Selecting representative example plays...")
example_plays = select_example_plays(acd_with_results, play_acd_stats)

print(f"\n✓ Selected {len(example_plays)} example plays:\n")
for i, (label, play) in enumerate(example_plays, 1):
    print(f"{i}. {label}")
    print(f"   Game ID: {play['game_id']}, Play ID: {play['play_id']}")
    print(f"   Mean ACD: {play['mean_acd']:.2f} yards")
    print(f"   PNR Frame: {play['pnr_frame']}")
    print(f"   Outcome: {play['pass_result']}")
    print()


def visualize_play(game_id, play_id, acd_with_results, tracking_filtered, play_info, title):
    """
    Create two-panel visualization:
    Left: Field view with trajectories
    Right: ACD over time
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get play data
    play_acd = acd_with_results[(acd_with_results['game_id'] == game_id) & 
                                 (acd_with_results['play_id'] == play_id)].copy()
    play_tracking = tracking_filtered[(tracking_filtered['game_id'] == game_id) & 
                                       (tracking_filtered['play_id'] == play_id)].copy()
    
    # Get ball landing point
    ball_x = play_tracking['ball_land_x'].iloc[0]
    ball_y = play_tracking['ball_land_y'].iloc[0]
    
    # Get receiver trajectory
    receiver_track = play_tracking[play_tracking['player_role'] == 'Targeted Receiver'].sort_values('frame_id')
    
    # Get nearest defender at each frame
    defenders_track = play_tracking[play_tracking['player_role'] == 'Defensive Coverage']
    nearest_defender_ids = []
    for frame in play_acd['frame_id'].unique():
        frame_defenders = defenders_track[defenders_track['frame_id'] == frame]
        if len(frame_defenders) > 0:
            nearest = frame_defenders.loc[frame_defenders['dist_to_ball'].idxmin()]
            nearest_defender_ids.append(nearest['nfl_id'])
    
    # Get most common nearest defender
    if nearest_defender_ids:
        from collections import Counter
        primary_defender_id = Counter(nearest_defender_ids).most_common(1)[0][0]
        defender_track = defenders_track[defenders_track['nfl_id'] == primary_defender_id].sort_values('frame_id')
    else:
        defender_track = pd.DataFrame()
    
    # LEFT PANEL - Field View
    ax1 = axes[0]
    
    # Draw field area
    ax1.add_patch(plt.Rectangle((0, 0), 120, 53.3, fill=False, edgecolor='black', linewidth=2))
    
    # Plot ball landing point
    ax1.scatter(ball_x, ball_y, s=500, c='gold', marker='*', edgecolors='black', 
                linewidths=2, label='Ball Landing Point', zorder=5)
    
    # Plot receiver trajectory
    ax1.plot(receiver_track['x'], receiver_track['y'], 'o-', color='#2ecc71', 
             linewidth=3, markersize=8, label=f'Receiver: {receiver_track["player_name"].iloc[0]}', alpha=0.8)
    ax1.scatter(receiver_track['x'].iloc[0], receiver_track['y'].iloc[0], 
                s=200, c='#2ecc71', marker='s', edgecolors='black', linewidths=2, zorder=4)
    
    # Plot defender trajectory
    if len(defender_track) > 0:
        ax1.plot(defender_track['x'], defender_track['y'], 'o-', color='#e74c3c', 
                 linewidth=3, markersize=8, label=f'Nearest Defender: {defender_track["player_name"].iloc[0]}', alpha=0.8)
        ax1.scatter(defender_track['x'].iloc[0], defender_track['y'].iloc[0], 
                    s=200, c='#e74c3c', marker='s', edgecolors='black', linewidths=2, zorder=4)
    
    ax1.set_xlim(min(receiver_track['x'].min(), ball_x) - 5, max(receiver_track['x'].max(), ball_x) + 5)
    ax1.set_ylim(min(receiver_track['y'].min(), ball_y) - 5, max(receiver_track['y'].max(), ball_y) + 5)
    ax1.set_xlabel('X Position (yards)', fontsize=12)
    ax1.set_ylabel('Y Position (yards)', fontsize=12)
    ax1.set_title('Player Trajectories to Ball Landing Point', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_aspect('equal')
    
    # RIGHT PANEL - ACD Over Time
    ax2 = axes[1]
    
    frames = play_acd['frame_id'].values
    acd_values = play_acd['ACD'].values
    
    # Plot ACD
    ax2.plot(frames, acd_values, 'o-', linewidth=3, markersize=8, color='#3498db')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero (Even Control)')
    ax2.fill_between(frames, 0, acd_values, where=(acd_values > 0), alpha=0.3, color='#2ecc71', label='Receiver Advantage')
    ax2.fill_between(frames, 0, acd_values, where=(acd_values < 0), alpha=0.3, color='#e74c3c', label='Defender Advantage')
    
    # Mark PNR if exists
    pnr_frame = play_info['pnr_frame']
    if pd.notna(pnr_frame):
        pnr_idx = play_acd[play_acd['frame_id'] == pnr_frame].index
        if len(pnr_idx) > 0:
            pnr_acd = play_acd.loc[pnr_idx[0], 'ACD']
            ax2.axvline(x=pnr_frame, color='purple', linestyle=':', linewidth=3, label=f'Point of No Return')
            ax2.scatter(pnr_frame, pnr_acd, s=200, c='purple', marker='D', edgecolors='black', 
                       linewidths=2, zorder=5)
    
    ax2.set_xlabel('Frame Number', fontsize=12)
    ax2.set_ylabel('ACD (yards)', fontsize=12)
    ax2.set_title('Airspace Control Differential Over Time', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(alpha=0.3)
    
    # Overall title
    outcome_names = {'C': 'Complete', 'I': 'Incomplete', 'IN': 'Interception'}
    fig.suptitle(f'{title}\nOutcome: {outcome_names[play_info["pass_result"]]} | Mean ACD: {play_info["mean_acd"]:.2f} yards', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.show()

print("✓ Visualization function defined")
print("\nReady to visualize example plays!")


# Visualize each example play
print("Creating visualizations for example plays...\n")

for i, (label, play_info) in enumerate(example_plays, 1):
    print(f"Visualizing Play {i}: {label}")
    visualize_play(
        game_id=play_info['game_id'],
        play_id=play_info['play_id'],
        acd_with_results=acd_with_results,
        tracking_filtered=tracking_filtered,
        play_info=play_info,
        title=f"Example {i}: {label}"
    )
    print()


# Create final summary visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: PNR timing distribution by outcome
ax1 = axes[0, 0]
for outcome in ['C', 'I', 'IN']:
    data = plays_with_pnr[plays_with_pnr['pass_result'] == outcome]['pnr_pct']
    if len(data) > 0:
        ax1.hist(data, bins=20, alpha=0.6, label=f'{outcome} (n={len(data)})', edgecolor='black')

ax1.set_xlabel('PNR Timing (% of Ball Flight)', fontsize=11)
ax1.set_ylabel('Number of Plays', fontsize=11)
ax1.set_title('When Does Control Become Inevitable?', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Mean ACD vs Number of Frames
ax2 = axes[0, 1]
for outcome, color in [('C', '#2ecc71'), ('I', '#f39c12'), ('IN', '#e74c3c')]:
    data = play_acd_stats[play_acd_stats['pass_result'] == outcome]
    ax2.scatter(data['num_frames'], data['mean_acd'], alpha=0.5, s=30, 
               label=outcome, color=color)

ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Number of Frames (Ball Flight Time)', fontsize=11)
ax2.set_ylabel('Mean ACD (yards)', fontsize=11)
ax2.set_title('ACD vs Ball Flight Duration', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Dominated vs Contested plays
ax3 = axes[1, 0]
contested_counts = []
outcome_labels = []
for outcome in ['C', 'I', 'IN']:
    outcome_data = play_acd_stats[play_acd_stats['pass_result'] == outcome]
    contested = (outcome_data['pnr_frame'].notna()).sum()
    dominated = (outcome_data['pnr_frame'].isna()).sum()
    contested_counts.append([dominated, contested])
    outcome_labels.append(outcome)

contested_counts = np.array(contested_counts)
x = np.arange(len(outcome_labels))
width = 0.6

ax3.bar(x, contested_counts[:, 0], width, label='Dominated (No PNR)', color='#95a5a6', edgecolor='black')
ax3.bar(x, contested_counts[:, 1], width, bottom=contested_counts[:, 0], 
       label='Contested (Has PNR)', color='#9b59b6', edgecolor='black')

ax3.set_ylabel('Number of Plays', fontsize=11)
ax3.set_title('Dominated vs Contested Plays by Outcome', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(['Complete', 'Incomplete', 'Interception'])
ax3.legend()
ax3.grid(alpha=0.3, axis='y')

# Plot 4: ACD variability by outcome
ax4 = axes[1, 1]
for outcome, color in [('C', '#2ecc71'), ('I', '#f39c12'), ('IN', '#e74c3c')]:
    data = play_acd_stats[play_acd_stats['pass_result'] == outcome]
    ax4.scatter(data['mean_acd'], data['std_acd'], alpha=0.5, s=30, 
               label=outcome, color=color)

ax4.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_xlabel('Mean ACD (yards)', fontsize=11)
ax4.set_ylabel('ACD Std Dev (yards)', fontsize=11)
ax4.set_title('ACD Stability: Mean vs Variability', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Final aggregate analysis complete")


# Save key datasets for reference
print("Saving analysis outputs...")

# Save play-level summary
play_summary = play_acd_stats[['game_id', 'play_id', 'pass_result', 'mean_acd', 
                                'min_acd', 'max_acd', 'num_frames', 'pnr_frame']].copy()
play_summary.to_csv('play_level_acd_summary.csv', index=False)
print(f"✓ Saved play_level_acd_summary.csv ({len(play_summary):,} plays)")

# Print final summary statistics
print("\n" + "="*60)
print("FINAL ANALYSIS SUMMARY")
print("="*60)
print(f"Total plays analyzed: {len(play_acd_stats):,}")
print(f"Total frames analyzed: {len(acd_data):,}")
print(f"\nOutcome breakdown:")
for outcome in ['C', 'I', 'IN']:
    count = len(play_acd_stats[play_acd_stats['pass_result'] == outcome])
    pct = count / len(play_acd_stats) * 100
    mean_acd = play_acd_stats[play_acd_stats['pass_result'] == outcome]['mean_acd'].mean()
    print(f"  {outcome}: {count:,} plays ({pct:.1f}%) | Mean ACD: {mean_acd:+.2f} yards")

print(f"\nContested vs Dominated:")
print(f"  Contested (has PNR): {play_acd_stats['pnr_frame'].notna().sum():,} ({play_acd_stats['pnr_frame'].notna().mean()*100:.1f}%)")
print(f"  Dominated (no PNR): {play_acd_stats['pnr_frame'].isna().sum():,} ({play_acd_stats['pnr_frame'].isna().mean()*100:.1f}%)")

print("\n✓ Analysis complete!")



# Install nfl-tracks library
!pip install nfl_tracks -q

print("✓ nfl-tracks installed successfully")


from nfl import visuals
import matplotlib.pyplot as plt

# Prepare tracking data for nfl-tracks (it expects input format, not output)
def create_enhanced_visualization(game_id, play_id, frame_id, title, save_name):
    """
    Create a relay dashboard view for a specific play frame
    """
    # Get the play data from input_data
    play_data = input_data[(input_data['game_id'] == game_id) & 
                           (input_data['play_id'] == play_id)].copy()
    
    if len(play_data) == 0:
        print(f"No data found for Game {game_id}, Play {play_id}")
        return None
    
    # Initialize Play object
    play = visuals.Play(play_data, game_id, play_id, supplementary)
    
    # Get the last frame (moment of throw)
    max_frame = play_data['frame_id'].max()
    
    # Create relay dashboard view
    fig, ax = play.plot_snap(frameId=max_frame, relay=True, size=150)
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Save for media gallery
    plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved {save_name}.png")
    
    plt.show()
    
    return fig

print("Creating enhanced visualizations for example plays...\n")

# Create enhanced viz for each example (at moment of throw)
example_titles = [
    "Clean Completion - Receiver Dominated Airspace",
    "Contested Completion - Late Defensive Pressure", 
    "Interception - Defender Controlled Throughout",
    "True 50-50 Ball - Tight Coverage"
]

for i, ((label, play_info), title) in enumerate(zip(example_plays, example_titles), 1):
    print(f"\nExample {i}: {label}")
    create_enhanced_visualization(
        game_id=play_info['game_id'],
        play_id=play_info['play_id'],
        frame_id=1,  # Moment of throw
        title=title,
        save_name=f'example_play_{i}_enhanced'
    )


# Create animations for key plays and save them for YouTube upload
print("Creating animations for key plays...")
print("Note: This may take 1-2 minutes per animation\n")

from matplotlib.animation import PillowWriter
import os

def create_play_animation(game_id, play_id, label, save_name):
    """Create animation for a specific play and save to file"""
    play_data = input_data[(input_data['game_id'] == game_id) & 
                           (input_data['play_id'] == play_id)].copy()
    
    if len(play_data) == 0:
        print(f"No data found for Game {game_id}, Play {play_id}")
        return None
    
    # Initialize Play object
    play = visuals.Play(play_data, game_id, play_id, supplementary)
    
    # Create relay animation
    print(f"Creating animation for: {label}")
    animation = play.animate(relay=True, kaggle=False, speed=100, size=150)
    
    # Save as GIF
    gif_filename = f"{save_name}.gif"
    writer = PillowWriter(fps=10)
    animation.save(gif_filename, writer=writer)
    print(f"✓ Saved {gif_filename}")
    
    # Display in notebook
    from IPython.display import Image, display
    display(Image(filename=gif_filename))
    
    print(f"✓ Animation created and saved for {label}\n")
    return animation

# Create animation for contested completion
print("="*60)
print("Animation 1: Contested Completion")
print("="*60)
contested_play = example_plays[1][1]  # Second example - contested completion
animation_1 = create_play_animation(
    game_id=contested_play['game_id'],
    play_id=contested_play['play_id'],
    label="Contested Completion - Receiver vs Defender Race",
    save_name="contested_completion_animation"
)

# Create animation for interception
print("\n" + "="*60)
print("Animation 2: Interception")
print("="*60)
interception_play = example_plays[2][1]
animation_2 = create_play_animation(
    game_id=interception_play['game_id'],
    play_id=interception_play['play_id'],
    label="Interception - Defender Dominates Airspace",
    save_name="interception_animation"
)

print("\n" + "="*60)
print("ANIMATIONS SAVED ")
print("="*60)
print("1. contested_completion_animation.gif")
print("2. interception_animation.gif")



# Create a compelling thumbnail image for your writeup card
print("Creating thumbnail image for writeup submission...\n")

# Use the most visually interesting play
thumbnail_play = example_plays[1][1]  # Contested completion

play_data = input_data[(input_data['game_id'] == thumbnail_play['game_id']) & 
                       (input_data['play_id'] == thumbnail_play['play_id'])].copy()

play = visuals.Play(play_data, thumbnail_play['game_id'], thumbnail_play['play_id'], supplementary)

# Get an interesting frame (middle of the play)
max_frame = play_data['frame_id'].max()
mid_frame = max_frame // 2

# Create relay dashboard for thumbnail
fig, ax = play.plot_snap(frameId=mid_frame, relay=True, size=200)

# Add title overlay
plt.suptitle("When Is a Catch Decided?\nMeasuring Airspace Control on Downfield Passes", 
             fontsize=18, fontweight='bold', y=0.98)

# Save at exact dimensions for Kaggle (560 x 280)
fig.set_size_inches(11.2, 5.6)
plt.savefig('writeup_thumbnail.png', dpi=50, bbox_inches='tight', facecolor='white')
print("✓ Saved writeup_thumbnail.png (for Kaggle submission)")

plt.show()

print("\n" + "="*60)
print("MEDIA GALLERY FILES READY:")
print("="*60)
print("1. writeup_thumbnail.png - Use this for your card image")
print("2. example_play_1_enhanced.png")
print("3. example_play_2_enhanced.png")
print("4. example_play_3_enhanced.png")
print("5. example_play_4_enhanced.png")
print("6. contested_completion_animation.gif (if animation created)")


