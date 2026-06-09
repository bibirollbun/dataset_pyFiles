import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from matplotlib import rc
from IPython.display import HTML
import seaborn as sns
from scipy.spatial.distance import euclidean
from scipy.spatial import Voronoi
from scipy.stats import pearsonr, ttest_ind
import warnings
import os

warnings.filterwarnings('ignore')
os.makedirs('visuals', exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11


base_path = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/'
train_path = base_path + 'train/'


# Load supplementary (play-level data with pass_result, routes, coverage, etc.)
supp = pd.read_csv(base_path + 'supplementary_data.csv')

# Load BOTH input AND output tracking data (output has player_role, ball_land_x/y)
input_list = []
output_list = []
for week in range(1, 19): 
    input_df = pd.read_csv(f'{train_path}input_2023_w{week:02d}.csv')
    output_df = pd.read_csv(f'{train_path}output_2023_w{week:02d}.csv')
    
    input_df['week'] = week
    output_df['week'] = week
    
    input_list.append(input_df)
    output_list.append(output_df)

# Concatenate all data
input_df = pd.concat(input_list, ignore_index=True)
output_df = pd.concat(output_list, ignore_index=True)

# Start with output data (only has x, y positions)
df = output_df.copy()

metadata_cols = ['game_id', 'play_id', 'nfl_id', 'player_role', 'player_side', 
                 'ball_land_x', 'ball_land_y', 'player_name', 'player_position',
                 's', 'a', 'dir', 'o', 'play_direction']

# Add optional columns if they exist
if 'event' in input_df.columns:
    metadata_cols.append('event')

input_metadata = input_df[metadata_cols].drop_duplicates()

input_metadata = input_metadata.groupby(['game_id', 'play_id', 'nfl_id']).first().reset_index()

df = df.merge(input_metadata, on=['game_id', 'play_id', 'nfl_id'], how='left')

df = df.merge(supp[['game_id', 'play_id', 'pass_result', 'route_of_targeted_receiver', 
                     'team_coverage_type', 'pass_length', 'offense_formation', 
                     'receiver_alignment', 'play_action', 'dropback_type']], 
              on=['game_id', 'play_id'], 
              how='left')

tracking_df = df.copy()


# Visualize Ball-in-Air Window Concept
# Select a sample play to demonstrate the temporal window

# Find a play with complete data
sample_plays = df.groupby(['game_id', 'play_id']).filter(lambda x: len(x) > 20).groupby(['game_id', 'play_id']).head(1)
if len(sample_plays) > 0:
    sample_game_id = sample_plays.iloc[0]['game_id']
    sample_play_id = sample_plays.iloc[0]['play_id']
    sample_play = df[(df['game_id'] == sample_game_id) & (df['play_id'] == sample_play_id)].copy()
    
    # Get receiver and ball data
    receiver = sample_play[sample_play['player_role'] == 'Targeted Receiver']
    
    if len(receiver) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Timeline visualization
        ax1 = axes[0]
        frames = sorted(sample_play['frame_id'].unique())
        
        # Simulate ball-in-air window (typically middle frames)
        total_frames = len(frames)
        ball_in_air_start = int(total_frames * 0.3)
        ball_in_air_end = int(total_frames * 0.8)
        
        # Draw timeline
        ax1.barh([0], [total_frames], height=0.3, color='lightgray', edgecolor='black', label='All Frames')
        ax1.barh([0], [ball_in_air_end - ball_in_air_start], 
                left=ball_in_air_start, height=0.3, color='#4CAF50', 
                edgecolor='black', linewidth=2, label='Ball-in-Air Window')
        
        # Add markers
        ax1.plot([ball_in_air_start], [0], 'o', markersize=15, color='blue', 
                markeredgecolor='black', markeredgewidth=2, label='Pass Release', zorder=10)
        ax1.plot([ball_in_air_end], [0], 'X', markersize=15, color='red', 
                markeredgecolor='black', markeredgewidth=2, label='Ball Arrival', zorder=10)
        
        # Annotations
        ax1.text(ball_in_air_start/2, 0.25, 'Pre-Snap\n& Setup', 
                ha='center', va='center', fontsize=10, color='gray')
        ax1.text((ball_in_air_start + ball_in_air_end)/2, 0.25, 
                f'GAI Calculation Window\n({ball_in_air_end - ball_in_air_start} frames)', 
                ha='center', va='center', fontsize=11, fontweight='bold', color='darkgreen')
        ax1.text((ball_in_air_end + total_frames)/2, 0.25, 'Post-\nCatch', 
                ha='center', va='center', fontsize=10, color='gray')
        
        ax1.set_xlim(-2, total_frames + 2)
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
        ax1.set_title('Ball-in-Air Window Timeline', fontsize=14, fontweight='bold')
        ax1.set_yticks([])
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(axis='x', alpha=0.3)
        
        # Right plot: Receiver position over time
        ax2 = axes[1]
        receiver_sorted = receiver.sort_values('frame_id')
        
        # Plot full trajectory
        ax2.plot(receiver_sorted['x'], receiver_sorted['y'], 
                'o-', color='lightgray', linewidth=2, markersize=4, 
                alpha=0.5, label='Full Route')
        
        # Highlight ball-in-air portion
        ball_frames = receiver_sorted.iloc[ball_in_air_start:ball_in_air_end]
        ax2.plot(ball_frames['x'], ball_frames['y'], 
                'o-', color='#4CAF50', linewidth=3, markersize=6, 
                label='Ball-in-Air (GAI Window)')
        
        # Mark release and catch points
        if len(receiver_sorted) >= ball_in_air_start:
            release_point = receiver_sorted.iloc[ball_in_air_start]
            ax2.plot(release_point['x'], release_point['y'], 
                    'o', markersize=15, color='blue', 
                    markeredgecolor='black', markeredgewidth=2, 
                    label='Pass Release', zorder=10)
        
        if len(receiver_sorted) >= ball_in_air_end:
            catch_point = receiver_sorted.iloc[ball_in_air_end-1]
            ax2.plot(catch_point['x'], catch_point['y'], 
                    'X', markersize=15, color='red', 
                    markeredgecolor='black', markeredgewidth=2, 
                    label='Ball Arrival', zorder=10)
        
        # Right plot: Distribution of ball-in-air window lengths across all plays
        ax2 = axes[1]
        
        # Calculate ball-in-air window lengths for multiple plays
        play_window_lengths = []
        for (gid, pid), play_group in df.groupby(['game_id', 'play_id']):
            n_frames = len(play_group['frame_id'].unique())
            if n_frames > 10:  # Only include plays with sufficient frames
                # Estimate ball-in-air window (typically 30-80% of frames)
                window_length = int(n_frames * 0.5)  # Approximate
                play_window_lengths.append(window_length)
            if len(play_window_lengths) >= 500:  # Limit for performance
                break
        
        if len(play_window_lengths) > 0:
            ax2.hist(play_window_lengths, bins=30, color='#4CAF50', 
                    edgecolor='black', alpha=0.7)
            ax2.axvline(np.median(play_window_lengths), color='red', 
                       linestyle='--', linewidth=2, 
                       label=f'Median: {np.median(play_window_lengths):.1f} frames')
            ax2.set_xlabel('Ball-in-Air Window Length (frames)', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Distribution of Ball-in-Air Windows', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('visuals/00_ball_in_air_window_concept.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("No targeted receiver found in sample play")
else:
    print("No suitable sample play found")


fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Speed distribution
ax1 = axes[0, 0]
speeds = df['s'].dropna()
ax1.hist(speeds, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Speed (yards/sec)')
ax1.set_ylabel('Frequency')
ax1.set_title('Player Speed Distribution')
ax1.axvline(speeds.median(), color='red', linestyle='--', linewidth=2)

# Acceleration distribution
ax2 = axes[0, 1]
accel = df['a'].dropna()
ax2.hist(accel, bins=50, color='coral', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Acceleration (yards/secÂ²)')
ax2.set_ylabel('Frequency')
ax2.set_title('Acceleration Distribution')

# Direction distribution
ax3 = axes[0, 2]
directions = df['dir'].dropna()
ax3.hist(directions, bins=36, color='lightgreen', edgecolor='black', alpha=0.7)
ax3.set_xlabel('Direction (degrees)')
ax3.set_ylabel('Frequency')
ax3.set_title('Direction Distribution')

# Pass results
ax4 = axes[1, 0]
if 'pass_result' in df.columns:
    pass_results = df.groupby('play_id')['pass_result'].first().value_counts()
    colors = {'C': 'green', 'I': 'red', 'IN': 'orange'}
    ax4.bar(range(len(pass_results)), pass_results.values, 
           color=[colors.get(x, 'gray') for x in pass_results.index],
           edgecolor='black', alpha=0.7)
    ax4.set_xticks(range(len(pass_results)))
    ax4.set_xticklabels(pass_results.index)
    ax4.set_ylabel('Count')
    ax4.set_title('Pass Results')

# Field heatmap
ax5 = axes[1, 1]
sample = df.sample(min(10000, len(df)))
heatmap, xedges, yedges = np.histogram2d(sample['x'], sample['y'], bins=30)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im = ax5.imshow(heatmap.T, extent=extent, origin='lower', cmap='YlOrRd', aspect='auto')
ax5.set_xlabel('X Position (yards)')
ax5.set_ylabel('Y Position (yards)')
ax5.set_title('Player Position Heatmap')



plt.tight_layout()
plt.savefig('visuals/01_dataset_overview.png', dpi=300, bbox_inches='tight')
plt.show()



def nfl_velocity(speed, direction_deg):
    """
    Convert NFL tracking direction to Cartesian velocity.
    NFL: 0Â° = north (+y), 90Â° = east (+x), clockwise.
    """
    theta = np.deg2rad(90 - direction_deg)
    return np.array([
        speed * np.cos(theta),
        speed * np.sin(theta)
    ])

def calculate_gai_for_frame(frame_df, receiver_id, catch_xy):
    """
    Compute frame-level GAI toward fixed catch point.
    """

    # --- Receiver ---
    rec = frame_df[frame_df["nfl_id"] == receiver_id]
    if rec.empty:
        return None
    rec = rec.iloc[0]

    # --- Defenders ---
    defenders = frame_df[frame_df["player_side"] == "Defense"]
    if defenders.empty:
        return None

    # --- Receiver position ---
    rec_pos = np.array([rec["x"], rec["y"]], dtype=float)
    if np.isnan(rec_pos).any():
        return None

    # --- Distances to defenders ---
    def_pos = defenders[["x", "y"]].to_numpy(dtype=float)
    if np.isnan(def_pos).any():
        return None
    dists = np.linalg.norm(def_pos - rec_pos, axis=1)
    separation = float(dists.min())

    # --- Direction to catch point ---
    to_ball = catch_xy - rec_pos
    dist_to_ball = np.linalg.norm(to_ball)
    if dist_to_ball < 0.5:
        return None
    ball_dir = to_ball / dist_to_ball

    # --- Receiver velocity toward catch point ---
    rec_vel = nfl_velocity(rec["s"], rec["dir"])
    rec_vel_toward = float(np.dot(rec_vel, ball_dir))

    # --- Defender velocity & angle toward catch point ---
    def_vels_toward = []
    def_angles = []

    for _, d in defenders.iterrows():
        d_vel = nfl_velocity(d["s"], d["dir"])
        proj = float(np.dot(d_vel, ball_dir))
        def_vels_toward.append(proj)

        norm = np.linalg.norm(d_vel)
        if norm > 0.1:
            angle = np.arccos(np.clip(proj / norm, -1.0, 1.0))
            def_angles.append(angle)

    best_def_vel = max(def_vels_toward) if def_vels_toward else 0.0
    vel_adv = rec_vel_toward - best_def_vel

    # --- Angular advantage ---
    rec_speed_norm = np.linalg.norm(rec_vel)
    rec_angle = (
        np.arccos(np.clip(rec_vel_toward / rec_speed_norm, -1.0, 1.0))
        if rec_speed_norm > 0.1 else np.pi / 2
    )
    best_def_angle = min(def_angles) if def_angles else np.pi
    ang_adv = best_def_angle - rec_angle

    # --- Local congestion ---
    nearby = int((dists < 5.0).sum())
    space = 1.0 / (nearby + 1.0)

    # --- Normalize components ---
    sep_norm = min(separation / 10.0, 1.0)
    vel_norm = np.clip((vel_adv + 5.0) / 10.0, 0.0, 1.0)
    ang_norm = np.clip((ang_adv + np.pi) / (2 * np.pi), 0.0, 1.0)

    gai = (
        0.30 * sep_norm +
        0.30 * vel_norm +
        0.20 * ang_norm +
        0.20 * space
    )

    return {
        "gai": float(gai),
        "separation": separation,
        "velocity_advantage": vel_adv,
        "angular_advantage": ang_adv,
        "space_score": space,
        "nearby_defenders": nearby
    }

def calculate_dynamic_gai_for_play(play_df, game_id, play_id, max_frames=10):
    """
    Compute play-level GAI by averaging valid in-flight frames.
    """

    # --- Catch point (ground truth) ---
    ball_x = play_df["ball_land_x"].iloc[0]
    ball_y = play_df["ball_land_y"].iloc[0]
    if pd.isna(ball_x) or pd.isna(ball_y):
        return None
    catch_xy = np.array([ball_x, ball_y], dtype=float)

    # --- Identify receiver ONCE ---
    rec_df = play_df[play_df["player_role"] == "Targeted Receiver"]
    if rec_df.empty:
        return None
    receiver_id = rec_df["nfl_id"].iloc[0]
    receiver_name = rec_df["player_name"].iloc[0]

    # --- Ball-in-air frames ---
    frames = play_df[
        play_df["player_role"] == "Targeted Receiver"
    ]["frame_id"].astype(int).unique()

    if len(frames) == 0:
        return None

    # Sample evenly if too many frames
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = frames[idx]

    frame_outputs = []

    for fid in frames:
        frame_df = play_df[play_df["frame_id"] == fid]
        out = calculate_gai_for_frame(frame_df, receiver_id, catch_xy)
        if out:
            frame_outputs.append(out)

    if not frame_outputs:
        return None

    gai_vals = [f["gai"] for f in frame_outputs]

    return {
        "game_id": game_id,
        "play_id": play_id,
        "receiver_id": receiver_id,
        "receiver_name": receiver_name,

        "gai": float(np.mean(gai_vals)),
        "gai_max": float(np.max(gai_vals)),
        "gai_min": float(np.min(gai_vals)),
        "gai_std": float(np.std(gai_vals)),
        "frames_tracked": len(gai_vals),

        "separation_avg": float(np.mean([f["separation"] for f in frame_outputs])),
        "velocity_advantage_avg": float(np.mean([f["velocity_advantage"] for f in frame_outputs])),
        "angular_advantage_avg": float(np.mean([f["angular_advantage"] for f in frame_outputs])),
        "space_score_avg": float(np.mean([f["space_score"] for f in frame_outputs])),
        "nearby_defenders_avg": float(np.mean([f["nearby_defenders"] for f in frame_outputs]))
    }

results = []

for (gid, pid), play_df in df.groupby(["game_id", "play_id"]):
    res = calculate_dynamic_gai_for_play(play_df, gid, pid)
    if res:
        results.append(res)

gai_df = pd.DataFrame(results)

print(f"Computed GAI for {len(gai_df)} plays")
gai_df.head()

if "supp" in globals():
    merge_cols = [
        "game_id", "play_id", "pass_result", "route_of_targeted_receiver",
        "team_coverage_type", "pass_length", "offense_formation",
        "receiver_alignment", "play_action", "dropback_type",
        "yards_gained", "expected_points_added"
    ]

    available = [c for c in merge_cols if c in supp.columns]
    gai_df = gai_df.merge(
        supp[available].drop_duplicates(),
        on=["game_id", "play_id"],
        how="left"
    )

    if "pass_result" in gai_df.columns:
        gai_df["completion"] = (gai_df["pass_result"] == "C").astype(int)




completed = gai_df[gai_df['completion'] == 1]['gai']
incomplete = gai_df[gai_df['completion'] == 0]['gai']

plt.hist(incomplete, bins=30, alpha=0.6, color='red', label='Incomplete')
plt.hist(completed, bins=30, alpha=0.6, color='green', label='Complete')

plt.xlabel('GAI')
plt.ylabel('Count')
plt.title('GAI Distribution by Play Outcome')
plt.legend()
plt.tight_layout()
plt.savefig('visuals/gai_distribution_by_outcome.png', dpi=300)
plt.show()



gai_df['gai_quartile'] = pd.qcut(
    gai_df['gai'],
    4,
    labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
)

quartile_stats = (
    gai_df
    .groupby('gai_quartile')['completion']
    .agg(['mean', 'count'])
)

plt.bar(
    quartile_stats.index,
    quartile_stats['mean'] * 100,
    color=['#d62728','#ff7f0e','#2ca02c','#1f77b4'],
    edgecolor='black'
)

plt.ylabel('Completion Rate (%)')
plt.title('Completion Rate by GAI Quartile')
plt.ylim(0, 100)

for i, (_, row) in enumerate(quartile_stats.iterrows()):
    plt.text(
        i,
        row['mean'] * 100 + 1,
        f"{row['mean']*100:.1f}%\n(n={int(row['count'])})",
        ha='center',
        fontsize=9
    )

plt.tight_layout()
plt.savefig('visuals/gai_quartile_completion.png', dpi=300)
plt.show()

gai_df.drop(columns='gai_quartile', inplace=True)



# --- Fix duplicated columns from merge ---
rename_map = {
    'route_of_targeted_receiver_x': 'route_of_targeted_receiver',
    'team_coverage_type_x': 'team_coverage_type',
    'pass_length_x': 'pass_length',
    'yards_gained_x': 'yards_gained',
    'expected_points_added_x': 'expected_points_added',
    'pass_result_x': 'pass_result'
}

for old, new in rename_map.items():
    if old in gai_df.columns:
        gai_df[new] = gai_df[old]

# Optional: drop _y columns to avoid confusion
drop_cols = [c for c in gai_df.columns if c.endswith('_y')]
gai_df.drop(columns=drop_cols, inplace=True)




# Filter to top routes
route_counts = gai_df['route_of_targeted_receiver'].value_counts()
top_routes = route_counts.head(10).index
route_df = gai_df[gai_df['route_of_targeted_receiver'].isin(top_routes)].copy()

fig, ax = plt.subplots(figsize=(12, 6))

route_df.boxplot(column='gai', by='route_of_targeted_receiver', ax=ax, 
                    patch_artist=True, showmeans=True)

ax.set_xlabel('Route Type', fontsize=12, fontweight='bold')
ax.set_ylabel('GAI Score', fontsize=12, fontweight='bold')
ax.set_title('GAI Distribution by Route Type', fontsize=14, fontweight='bold')
plt.suptitle('')  # Remove default title
ax.grid(alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('visuals/06_gai_by_route.png', dpi=300, bbox_inches='tight')
plt.show()


# Filter to main coverage types
coverage_counts = gai_df['team_coverage_type'].value_counts()
top_coverages = coverage_counts.head(6).index
cov_df = gai_df[gai_df['team_coverage_type'].isin(top_coverages)].copy()

fig, ax = plt.subplots(figsize=(12, 6))

parts = ax.violinplot([cov_df[cov_df['team_coverage_type'] == cov]['gai'].values 
                        for cov in top_coverages],
                        positions=range(len(top_coverages)),
                        showmeans=True, showmedians=True)

ax.set_xlabel('Coverage Type', fontsize=12, fontweight='bold')
ax.set_ylabel('GAI Score', fontsize=12, fontweight='bold')
ax.set_title('GAI Distribution by Coverage Type', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(top_coverages)))
ax.set_xticklabels(top_coverages, rotation=45, ha='right')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('visuals/07_gai_by_coverage.png', dpi=300, bbox_inches='tight')
plt.show()


route_df = (
    gai_df
    .groupby('route_of_targeted_receiver')
    .agg(
        avg_gai=('gai', 'mean'),
        std_gai=('gai', 'std'),
        catch_rate=('completion', 'mean'),
        n_plays=('completion', 'count')
    )
    .reset_index()
)

# Filter low-volume routes
route_df = route_df[route_df['n_plays'] >= 10].copy()

if len(route_df) > 2:
    corr_gai_catch, pval_gai_catch = pearsonr(
        route_df['avg_gai'],
        route_df['catch_rate']
    )
else:
    corr_gai_catch, pval_gai_catch = np.nan, np.nan

top_routes = (
    route_df
    .sort_values('n_plays', ascending=False)
    .head(10)
    .sort_values('avg_gai', ascending=True)
)

plt.figure(figsize=(8,5))
plt.barh(
    top_routes['route_of_targeted_receiver'],
    top_routes['avg_gai'],
    color='steelblue',
    edgecolor='black'
)

plt.xlabel('Average GAI')
plt.title('Average In-Flight GAI by Route Type')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/05_avg_gai_by_route.png', dpi=300)
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=route_df,
    x='avg_gai',
    y='catch_rate',
    size='n_plays',
    sizes=(50,500),
    alpha=0.8,
    edgecolor='black'
)

# Trend line
z = np.polyfit(route_df['avg_gai'], route_df['catch_rate'], 1)
p = np.poly1d(z)
x_vals = np.linspace(route_df['avg_gai'].min(), route_df['avg_gai'].max(), 100)
plt.plot(x_vals, p(x_vals), 'r--', linewidth=2)

plt.xlabel('Average GAI')
plt.ylabel('Catch Rate')
plt.title('Route-Level GAI vs Catch Rate')
plt.grid(alpha=0.3)

plt.text(
    0.05, 0.95,
    f'Pearson r = {corr_gai_catch:.2f}\np = {pval_gai_catch:.3g}',
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4)
)

plt.tight_layout()
plt.savefig('visuals/05_route_gai_vs_catchrate.png', dpi=300)
plt.show()



player_df = (
    gai_df
    .groupby('receiver_name')
    .agg(
        avg_gai=('gai', 'mean'),
        catch_rate=('completion', 'mean'),
        targets=('completion', 'count')
    )
    .reset_index()
)

# Volume filter
player_df = player_df[player_df['targets'] >= 10].copy()

# -------------------------
# Top players by GAI
# -------------------------
top_players = (
    player_df
    .sort_values('avg_gai', ascending=False)
    .head(15)
    .sort_values('avg_gai', ascending=True)
)

plt.figure(figsize=(8,6))
plt.barh(
    top_players['receiver_name'],
    top_players['avg_gai'],
    color='steelblue',
    edgecolor='black'
)

plt.xlabel('Average GAI')
plt.title('Top Players by In-Flight Geometric Advantage')
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('visuals/06_top_players_by_gai.png', dpi=300)
plt.show()

corr_player, pval_player = pearsonr(
    player_df['avg_gai'],
    player_df['catch_rate']
)

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=player_df,
    x='avg_gai',
    y='catch_rate',
    size='targets',
    sizes=(50,500),
    alpha=0.8,
    edgecolor='black'
)

# Trend line
z = np.polyfit(player_df['avg_gai'], player_df['catch_rate'], 1)
p = np.poly1d(z)
x_vals = np.linspace(
    player_df['avg_gai'].min(),
    player_df['avg_gai'].max(),
    100
)
plt.plot(x_vals, p(x_vals), 'r--', linewidth=2)

plt.xlabel('Average GAI')
plt.ylabel('Catch Rate')
plt.title('Player-Level GAI vs Catch Rate')
plt.grid(alpha=0.3)

plt.text(
    0.05, 0.95,
    f'Pearson r = {corr_player:.2f}\np = {pval_player:.3g}',
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4)
)

plt.tight_layout()
plt.savefig('visuals/06_player_gai_vs_catchrate.png', dpi=300)
plt.show()




median_gai = player_df['avg_gai'].median()
median_cr = player_df['catch_rate'].median()

player_df['archetype'] = np.select(
    [
        (player_df['avg_gai'] >= median_gai) & (player_df['catch_rate'] >= median_cr),
        (player_df['avg_gai'] >= median_gai) & (player_df['catch_rate'] < median_cr),
        (player_df['avg_gai'] < median_gai) & (player_df['catch_rate'] >= median_cr),
        (player_df['avg_gai'] < median_gai) & (player_df['catch_rate'] < median_cr)
    ],
    [
        'High GAI / High Catch',
        'High GAI / Low Catch',
        'Low GAI / High Catch',
        'Low GAI / Low Catch'
    ],
    default='Unclassified'
)

plt.figure(figsize=(9,7))
sns.scatterplot(
    data=player_df,
    x='avg_gai',
    y='catch_rate',
    hue='archetype',
    size='targets',
    sizes=(50,400),
    alpha=0.8
)

# Median reference lines
plt.axvline(median_gai, color='black', linestyle='--', linewidth=1)
plt.axhline(median_cr, color='black', linestyle='--', linewidth=1)

plt.xlabel('Average GAI')
plt.ylabel('Catch Rate')
plt.title('Player Archetypes: Process vs Outcome')

# Axis limits
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

# Common text box style
box = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.75, edgecolor='black')

# Quadrant labels (large + boxed)
plt.text(
    (median_gai + x_max) / 2, (median_cr + y_max) / 2,
    'ELITE SEPARATORS\nHigh GAI â€¢ High Catch',
    ha='center', va='center', fontsize=12, fontweight='bold', bbox=box
)

plt.text(
    (median_gai + x_max) / 2, (y_min + median_cr) / 2,
    'PROCESS WINNERS\nHigh GAI â€¢ Low Catch',
    ha='center', va='center', fontsize=12, fontweight='bold', bbox=box
)

plt.text(
    (x_min + median_gai) / 2, (median_cr + y_max) / 2,
    'CONTESTED SPECIALISTS\nLow GAI â€¢ High Catch',
    ha='center', va='center', fontsize=12, fontweight='bold', bbox=box
)

plt.text(
    (x_min + median_gai) / 2, (y_min + median_cr) / 2,
    'LOW IMPACT TARGETS\nLow GAI â€¢ Low Catch',
    ha='center', va='center', fontsize=12, fontweight='bold', bbox=box
)

plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visuals/06_player_archetypes.png', dpi=300)
plt.show()



rc('animation', html='jshtml')
plt.rcParams['figure.max_open_warning'] = 0

def animate_gai_play(
    play_df,
    metadata,
    case_label,
    marker_color,
    box_color,
    outfile
):
    FIELD_W = 53.3
    FIELD_L = 120.0

    frames = sorted(play_df['frame_id'].unique())
    if len(frames) == 0:
        return

    bx = play_df['ball_land_x'].iloc[0]
    by = play_df['ball_land_y'].iloc[0]

    fig, ax = plt.subplots(figsize=(16, 8), dpi=100)

    # ---- Field ----
    ax.add_patch(patches.Rectangle((0, 0), FIELD_L, FIELD_W,
                                   facecolor='#8DBF87', edgecolor='#2E7D32',
                                   linewidth=2, zorder=0))
    ax.add_patch(patches.Rectangle((0, 0), 10, FIELD_W,
                                   facecolor='#5a8f5a', alpha=0.8, zorder=0))
    ax.add_patch(patches.Rectangle((FIELD_L - 10, 0), 10, FIELD_W,
                                   facecolor='#5a8f5a', alpha=0.8, zorder=0))

    for xline in range(10, int(FIELD_L), 10):
        ax.plot([xline, xline], [0, FIELD_W], color='white',
                linewidth=0.8, alpha=0.4, zorder=0)

    ax.plot(bx, by, marker='X', markersize=14, color=marker_color,
            markeredgewidth=2, markeredgecolor='black', zorder=10)

    ax.set_xlim(-5, FIELD_L + 5)
    ax.set_ylim(-2, FIELD_W + 2)
    ax.set_aspect('equal')
    ax.set_title(
        f"{case_label}\nGAI={metadata['gai']:.3f} | Game {metadata['game_id']} Play {metadata['play_id']}",
        fontsize=14, fontweight='bold'
    )

    # ---- Players ----
    receiver = play_df[play_df['player_name'] == metadata['targeted_receiver']]
    defenders = play_df[play_df['player_side'] == 'Defense']
    offense = play_df[
        (play_df['player_side'] == 'Offense') &
        (play_df['player_name'] != metadata['targeted_receiver'])
    ]

    rec_pt, = ax.plot([], [], 'o', color='cyan', markersize=14,
                      markeredgecolor='blue', markeredgewidth=2, zorder=6)
    rec_tr, = ax.plot([], [], '-', color='cyan', linewidth=3, alpha=0.6, zorder=4)

    def_pts, def_trs = [], []
    for def_id in defenders['nfl_id'].unique()[:5]:
        pt, = ax.plot([], [], 'o', color='red', markersize=10,
                      markeredgecolor='darkred', zorder=5)
        tr, = ax.plot([], [], '-', color='red', linewidth=2, alpha=0.3, zorder=4)
        def_pts.append(pt)
        def_trs.append(tr)

    off_pts = []
    for off_id in offense['nfl_id'].unique()[:5]:
        pt, = ax.plot([], [], 'o', color='gray', markersize=6, alpha=0.5, zorder=3)
        off_pts.append(pt)

    frame_txt = ax.text(
        0.02, 0.98, '',
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
    )

    gai_txt = ax.text(
        0.98, 0.98, case_label,
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.9)
    )

    sep_txt = ax.text(
        0.02, 0.02, '',
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )

    def init():
        frame_txt.set_text('')
        sep_txt.set_text('')
        return []

    def animate(i):
        f = frames[i]

        r = receiver[receiver['frame_id'] == f]
        if not r.empty:
            rx, ry = r[['x', 'y']].iloc[0]
            rec_pt.set_data([rx], [ry])
            trail = receiver[receiver['frame_id'] <= f]
            rec_tr.set_data(trail['x'], trail['y'])

            d = defenders[defenders['frame_id'] == f]
            if not d.empty:
                dist = np.sqrt((d['x'] - rx)**2 + (d['y'] - ry)**2).min()
                sep_txt.set_text(f'Separation: {dist:.1f} yds')

        for j, def_id in enumerate(defenders['nfl_id'].unique()[:5]):
            d = defenders[(defenders['nfl_id'] == def_id) & (defenders['frame_id'] == f)]
            if not d.empty:
                dx, dy = d[['x', 'y']].iloc[0]
                def_pts[j].set_data([dx], [dy])
                trail = defenders[defenders['nfl_id'] == def_id]
                trail = trail[trail['frame_id'] <= f]
                def_trs[j].set_data(trail['x'], trail['y'])

        for j, off_id in enumerate(offense['nfl_id'].unique()[:5]):
            o = offense[(offense['nfl_id'] == off_id) & (offense['frame_id'] == f)]
            if not o.empty:
                ox, oy = o[['x', 'y']].iloc[0]
                off_pts[j].set_data([ox], [oy])

        frame_txt.set_text(f'Frame {i+1}/{len(frames)}')
        return []

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(frames), interval=100, blit=False)

    anim.save(outfile, writer='pillow', fps=10, dpi=80)
    plt.close(fig)

    return HTML(anim.to_jshtml())



# Thresholds (robust, data-driven)
gai_low = gai_df['gai'].quantile(0.25)
gai_high = gai_df['gai'].quantile(0.75)

# Lucky: LOW GAI but COMPLETED
lucky_candidates = gai_df[
    (gai_df['gai'] <= gai_low) &
    (gai_df['completion'] == 1)
]

# Unlucky: HIGH GAI but INCOMPLETE
unlucky_candidates = gai_df[
    (gai_df['gai'] >= gai_high) &
    (gai_df['completion'] == 0)
]

lucky_play = lucky_candidates.sort_values('gai').iloc[0]
unlucky_play = unlucky_candidates.sort_values('gai', ascending=False).iloc[0]



lucky_play_metadata = {
    'game_id': lucky_play['game_id'],
    'play_id': lucky_play['play_id'],
    'gai': lucky_play['gai'],
    'targeted_receiver': lucky_play['receiver_name'],
    'separation_avg': lucky_play['separation_avg'],
    'velocity_advantage_avg': lucky_play['velocity_advantage_avg'],
}

play_data_lucky = df[
    (df['game_id'] == lucky_play['game_id']) &
    (df['play_id'] == lucky_play['play_id'])
].copy()

unlucky_play_metadata = {
    'game_id': unlucky_play['game_id'],
    'play_id': unlucky_play['play_id'],
    'gai': unlucky_play['gai'],
    'targeted_receiver': unlucky_play['receiver_name'],
    'separation_avg': unlucky_play['separation_avg'],
    'velocity_advantage_avg': unlucky_play['velocity_advantage_avg'],
}

play_data_unlucky = df[
    (df['game_id'] == unlucky_play['game_id']) &
    (df['play_id'] == unlucky_play['play_id'])
].copy()



animate_gai_play(
    play_df=play_data_lucky,
    metadata=lucky_play_metadata,
    case_label='LOW GAI â€¢ COMPLETED',
    marker_color='lime',
    box_color='khaki',
    outfile=f"visuals/low_gai_completion_{lucky_play_metadata['game_id']}_{lucky_play_metadata['play_id']}.gif"
)



animate_gai_play(
    play_df=play_data_unlucky,
    metadata=unlucky_play_metadata,
    case_label='HIGH GAI â€¢ INCOMPLETE',
    marker_color='orange',
    box_color='lightgreen',
    outfile=f"visuals/high_gai_incompletion_{unlucky_play_metadata['game_id']}_{unlucky_play_metadata['play_id']}.gif"
)



route_col = 'route_of_targeted_receiver'

# Aggregate GAI by player and route
player_route = (
    gai_df
    .groupby(['receiver_name', route_col])
    .agg(
        avg_gai=('gai', 'mean'),
        plays=('play_id', 'count')
    )
    .reset_index()
)

# Route-level volume filter
player_route = player_route[player_route['plays'] >= 5]

# Keep only top-N players by total targets
top_players = (
    gai_df
    .groupby('receiver_name')['play_id']
    .count()
    .nlargest(12)          # ðŸ‘ˆ adjust N here (e.g., 10â€“15)
    .index
)

player_route = player_route[player_route['receiver_name'].isin(top_players)]

# Pivot for heatmap
pivot = player_route.pivot(
    index='receiver_name',
    columns=route_col,
    values='avg_gai'
)

# Plot heatmap
plt.figure(figsize=(13, 7))
sns.heatmap(
    pivot,
    cmap='RdYlGn',
    annot=True,
    fmt='.2f',
    linewidths=0.3,
    cbar_kws={'label': 'Average GAI'}
)

plt.title('Player GAI by Route Type (Top Volume Players)')
plt.xlabel('Route Type')
plt.ylabel('Receiver')
plt.tight_layout()
plt.savefig('visuals/player_gai_by_route_heatmap.png', dpi=300)
plt.show()



coverage_df = (
    gai_df
    .groupby(['receiver_name', 'team_coverage_type'])
    .agg(
        avg_gai=('gai', 'mean'),
        plays=('gai', 'count')
    )
    .reset_index()
)

coverage_df = coverage_df[coverage_df['plays'] >= 5]

top_players = (
    gai_df
    .groupby('receiver_name')['gai']
    .count()
    .nlargest(10)
    .index
)

coverage_df = coverage_df[coverage_df['receiver_name'].isin(top_players)]

pivot = coverage_df.pivot(
    index='receiver_name',
    columns='team_coverage_type',
    values='avg_gai'
)

pivot.plot(kind='barh', figsize=(10,7))
plt.xlabel('Average GAI')
plt.title('Player GAI by Coverage Type')
plt.tight_layout()
plt.savefig('visuals/player_gai_by_coverage.png', dpi=300)
plt.show()



# Aggregate at (player, route) level
player_route_df = (
    gai_df
    .groupby(['receiver_name', 'route_of_targeted_receiver'])
    .agg(
        avg_gai=('gai', 'mean'),
        catch_rate=('completion', 'mean'),
        plays=('completion', 'count')
    )
    .reset_index()
)

# Minimum volume filter
player_route_df = player_route_df[player_route_df['plays'] >= 5]

plt.figure(figsize=(9,7))
sns.scatterplot(
    data=player_route_df,
    x='avg_gai',
    y='catch_rate',
    size='plays',
    sizes=(40, 400),
    alpha=0.75,
    edgecolor='black'
)

plt.xlabel('Average GAI (Playerâ€“Route)')
plt.ylabel('Catch Rate')
plt.title('Playerâ€“Route GAI vs Catch Rate')

plt.tight_layout()
plt.savefig('visuals/07_player_route_gai_vs_catchrate.png', dpi=300)
plt.show()


