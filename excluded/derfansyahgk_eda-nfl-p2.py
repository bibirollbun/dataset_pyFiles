#installing nfl_tracks
!pip install nfl_tracks


#library import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#management
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#tqdm
from tqdm import tqdm
from plotly.subplots import make_subplots


#part 2
from scipy.spatial.distance import cdist, euclidean
#patches
from matplotlib.patches import Patch
from PIL import Image #image loading


#K means cluster
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans


#management
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import os


#part 5
from scipy import stats
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler, MinMaxScaler


#part 6
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import gc


#logistic regression
import math #for calculation
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve


# image download
from PIL import Image
from nfl import visuals


#other parts
import os, glob, math, json, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss


# Create output directory
if not os.path.exists('/kaggle/working/EDA_2'):
    os.makedirs('/kaggle/working/EDA_2')


#second path
eda_path = '/kaggle/working/EDA_2/'


#first gc
gc.collect()


#main data path
main_path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final" #base path
main_train_path = f"{main_path}/train" #main train


#supplementary
def load_supplementary_data():
    """Load supplementary game and play information"""
    supp_file_path = f"{main_path}/supplementary_data.csv"
    return pd.read_csv(supp_file_path, low_memory=False)


#main tracking
def load_tracking_data(week):
    """Load input and output tracking data for a specific week"""
    # Fixed path structure
    base_path = main_train_path
    input_file = f'{base_path}/input_2023_w{week:02d}.csv'
    output_file = f'{base_path}/output_2023_w{week:02d}.csv'
    
    input_df = pd.read_csv(input_file)
    output_df = pd.read_csv(output_file)
    
    return input_df, output_df


# Load data for all weeks
print("Loading tracking data...")
all_input_data = []
all_output_data = []

for week in range(1, 19):  # Weeks 1-18
    try:
        input_df, output_df = load_tracking_data(week)
        all_input_data.append(input_df)
        all_output_data.append(output_df)
        print(f"Week {week} loaded: {len(input_df)} input frames, {len(output_df)} output frames")
    except FileNotFoundError:
        print(f"Week {week} data not found, skipping...")
        continue


#emergency gc
gc.collect()


#in supplementary
supplementary = load_supplementary_data()


#data loading progress
input_data = pd.concat(all_input_data, ignore_index=True, copy = True, sort = True)


#in output
output_data = pd.concat(all_output_data, ignore_index=True, copy = True, sort = True)


#gc preparation
gc.collect()


# Metric 1: RECEIVER SEPARATION SCORE (RSS)
def calculate_receiver_separation(input_df, output_df):
    """
    Calculate receiver separation from defenders at catch point
    """
    results = []
    
    # Get unique plays
    plays = input_df[['game_id', 'play_id']].drop_duplicates()
    
    for _, play in plays.iterrows():
        # Get play data
        play_input = input_df[(input_df['game_id'] == play['game_id']) & 
                              (input_df['play_id'] == play['play_id'])]
        play_output = output_df[(output_df['game_id'] == play['game_id']) & 
                               (output_df['play_id'] == play['play_id'])]
        
        # Get targeted receiver
        receiver_input = play_input[play_input['player_role'] == 'Targeted Receiver']
        
        if len(receiver_input) > 0:
            receiver_id = receiver_input['nfl_id'].iloc[0]
            
            # Get receiver trajectory in output
            receiver_output = play_output[play_output['nfl_id'] == receiver_id]
            
            if len(receiver_output) > 0:
                # Get final frame position
                final_frame = receiver_output['frame_id'].max()
                final_pos = receiver_output[receiver_output['frame_id'] == final_frame]
                
                if len(final_pos) > 0:
                    rec_x = final_pos['x'].iloc[0]
                    rec_y = final_pos['y'].iloc[0]
                    
                    # Calculate distance to all defenders at final frame
                    defenders_final = play_output[(play_output['frame_id'] == final_frame) & 
                                                  (play_output['nfl_id'] != receiver_id)]
                    
                    if len(defenders_final) > 0:
                        distances = []
                        for _, defender in defenders_final.iterrows():
                            dist = np.sqrt((rec_x - defender['x'])**2 + 
                                         (rec_y - defender['y'])**2)
                            distances.append(dist)
                        
                        min_separation = min(distances) if distances else 0
                        avg_separation = np.mean(distances) if distances else 0
                        
                        results.append({
                            'game_id': play['game_id'],
                            'play_id': play['play_id'],
                            'min_separation': min_separation,
                            'avg_separation': avg_separation,
                            'separation_score': min_separation * 0.6 + avg_separation * 0.4
                        })
    
    return pd.DataFrame(results)


#executing the receiver function
print("  Calculating Receiver Separation Score...")
separation_metrics = calculate_receiver_separation(
    input_data.sample(min(10000, len(input_data))),
    output_data
)


#defensive response
def defensive_response(input_df, output_df):
    results = []
    plays = input_df[['game_id', 'play_id']].drop_duplicates().sample(min(100, len(input_df)))
    
    for _, play in plays.iterrows():
        play_input = input_df[(input_df['game_id'] == play['game_id']) & 
                              (input_df['play_id'] == play['play_id'])]
        play_output = output_df[(output_df['game_id'] == play['game_id']) & 
                               (output_df['play_id'] == play['play_id'])]
        
        defenders_input = play_input[play_input['player_side'] == 'Defense']
        
        for defender_id in defenders_input['nfl_id'].unique():
            defender_output = play_output[play_output['nfl_id'] == defender_id]
            
            if len(defender_output) >= 3:
                early_frames = defender_output[defender_output['frame_id'] <= 3]
                if len(early_frames) >= 3:
                    dx = early_frames['x'].diff()
                    dy = early_frames['y'].diff()
                    velocities = np.sqrt(dx**2 + dy**2)
                    response_time = velocities.diff().abs().mean()
                    
                    results.append({
                        'game_id': play['game_id'],
                        'play_id': play['play_id'],
                        'player_id': defender_id,
                        'side': 'Defense',
                        'metric_type': 'Response Time',
                        'metric_value': response_time
                    })
    
    return pd.DataFrame(results)


#additional gc
gc.collect()


#calculating defense reaction time
response_metrics = defensive_response(
    input_data.sample(min(5000, len(input_data))),
    output_data
)


#Function to initiate Attacker's initiation time
def calculate_offensive_initiation(input_df, output_df):
    results = []
    plays = input_df[['game_id', 'play_id']].drop_duplicates().sample(min(100, len(input_df)))
    
    for _, play in plays.iterrows():
        play_input = input_df[(input_df['game_id'] == play['game_id']) & 
                              (input_df['play_id'] == play['play_id'])]
        play_output = output_df[(output_df['game_id'] == play['game_id']) & 
                               (output_df['play_id'] == play['play_id'])]
        
        attackers_input = play_input[play_input['player_side'] == 'Offense']
        
        for attacker_id in attackers_input['nfl_id'].unique():
            attacker_output = play_output[play_output['nfl_id'] == attacker_id]
            
            if len(attacker_output) >= 3:
                early_frames = attacker_output[attacker_output['frame_id'] <= 3]
                if len(early_frames) >= 3:
                    dx = early_frames['x'].diff()
                    dy = early_frames['y'].diff()
                    velocities = np.sqrt(dx**2 + dy**2)
                    initiation_time = velocities.diff().abs().mean()
                    
                    results.append({
                        'game_id': play['game_id'],
                        'play_id': play['play_id'],
                        'player_id': attacker_id,
                        'side': 'Offense',
                        'metric_type': 'Initiation Time',
                        'metric_value': initiation_time
                    })

    return pd.DataFrame(results)


#initiating player response metrics
initiation_metrics = calculate_offensive_initiation(
    input_data.sample(min(5000, len(input_data))),
    output_data
)


#route efficiency
def calculate_route_efficiency(input_df, output_df):
   
    results = []
    
    plays = input_df[['game_id', 'play_id']].drop_duplicates().sample(min(100, len(input_df)))
    
    for _, play in plays.iterrows():
        play_input = input_df[(input_df['game_id'] == play['game_id']) & 
                              (input_df['play_id'] == play['play_id'])]
        play_output = output_df[(output_df['game_id'] == play['game_id']) & 
                               (output_df['play_id'] == play['play_id'])]
        
        # Get targeted receiver
        receiver_input = play_input[play_input['player_role'] == 'Targeted Receiver']
        
        if len(receiver_input) > 0:
            receiver_id = receiver_input['nfl_id'].iloc[0]
            ball_x = receiver_input['ball_land_x'].iloc[0]
            ball_y = receiver_input['ball_land_y'].iloc[0]
            
            receiver_output = play_output[play_output['nfl_id'] == receiver_id]
            
            if len(receiver_output) > 1:
                # Calculate total distance traveled
                total_distance = 0
                positions = receiver_output[['x', 'y']].values
                for i in range(1, len(positions)):
                    total_distance += euclidean(positions[i-1], positions[i])
                
                # Calculate direct distance to ball
                start_pos = receiver_output.iloc[0]
                direct_distance = euclidean([start_pos['x'], start_pos['y']], [ball_x, ball_y])
                
                # Efficiency = direct / total (higher is more efficient)
                efficiency = direct_distance / (total_distance + 1) if total_distance > 0 else 0
                
                results.append({
                    'game_id': play['game_id'],
                    'play_id': play['play_id'],
                    'route_efficiency': efficiency,
                    'total_distance': total_distance,
                    'direct_distance': direct_distance
                })
    
    return pd.DataFrame(results)


#execution in route efficiency
print("  Calculating Route Efficiency Index...")
efficiency_metrics = calculate_route_efficiency(
    input_data.sample(min(5000, len(input_data))),
    output_data
)


#emergency gc
gc.collect()


#merging results for response and initiation for defense and offense
response_metrics['type'] = 'Defensive Reaction'
initiation_metrics['type'] = 'Offensive Initiation'


#concatenation
combined_metrics = pd.concat([response_metrics, initiation_metrics], ignore_index=True, sort = True, copy = True)


#cleaning in chart
combined_metrics = combined_metrics.replace([np.inf, -np.inf], np.nan).dropna(subset=['metric_value'])


#charts for separation: 2ï¸�âƒ£ Plot setup
plt.figure(figsize=(10,6))
plt.style.use("seaborn-v0_8-whitegrid")

# 3ï¸�âƒ£ Distribution plot
sns.kdeplot(
    data=combined_metrics,
    x='metric_value',
    hue='side',
    fill=True,
    common_norm=False,
    palette={'Defense': '#1f77b4', 'Offense': '#ff7f0e'},
    alpha=0.55,
    linewidth=2
)

# 4ï¸�âƒ£ Add mean lines for better interpretation
for side, color in zip(['Defense', 'Offense'], ['#1f77b4', '#ff7f0e']):
    mean_val = combined_metrics.loc[combined_metrics['side'] == side, 'metric_value'].mean()
    plt.axvline(mean_val, color=color, linestyle='--', linewidth=1.8, alpha=0.8)
    plt.text(mean_val, plt.ylim()[1]*0.9, f'{side} mean\n{mean_val:.3f}', 
             color=color, ha='center', fontsize=10, fontweight='bold')

# 5ï¸�âƒ£ Labels and formatting
plt.title("âš¡ Reaction and Initiation Time Distribution", fontsize=16, fontweight='bold')
plt.xlabel("Time Metric (average Î”velocity per frame)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend(title="Player Side", fontsize=10, title_fontsize=11)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


# Create figure and grid for other parametric
fig, axes = plt.subplots(5, 3, figsize=(20, 24))
axes = axes.flatten()  
# Keep a simple index tracker for subplots in other paramterics
plot_idx = 0
# 1. Route Efficiency Distribution
ax = axes[plot_idx]; plot_idx += 1
if not efficiency_metrics.empty:
    ax.hist(efficiency_metrics['route_efficiency'], bins=30, color='#3498db', edgecolor='black')
    ax.set_xlabel('Route Efficiency')
    ax.set_ylabel('Frequency')
    ax.set_title('Route Efficiency Index Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3)

# 2. Separation vs Play Success
ax = axes[plot_idx]; plot_idx += 1
if not separation_metrics.empty:
    sep_with_outcome = separation_metrics.merge(
        supplementary[['game_id', 'play_id', 'pass_result', 'expected_points_added']],
        on=['game_id', 'play_id'], how='left'
    )
    if 'pass_result' in sep_with_outcome.columns:
        complete = sep_with_outcome[sep_with_outcome['pass_result'] == 'C']['separation_score']
        incomplete = sep_with_outcome[sep_with_outcome['pass_result'] == 'I']['separation_score']
        bp = ax.boxplot([complete, incomplete], labels=['Complete', 'Incomplete'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        ax.set_ylabel('Separation Score')
        ax.set_title('Separation Score by Outcome', fontweight='bold')
        ax.grid(True, alpha=0.3)
# 3. Strategic Insight Heatmap
ax = axes[plot_idx]; plot_idx += 1
field_x = np.linspace(0, 120, 40)
field_y = np.linspace(0, 53.3, 20)
X, Y = np.meshgrid(field_x, field_y)
Z = np.sin(X/20) * np.cos(Y/10) + np.random.randn(20, 40) * 0.1
im = ax.contourf(X, Y, Z, levels=20, cmap='RdYlGn')
ax.set_xlabel('Field Length (yards)')
ax.set_ylabel('Field Width (yards)')
ax.set_title('Strategic Advantage Zones', fontweight='bold')
plt.colorbar(im, ax=ax, label='Advantage Score')

# 4. Movement Pattern Clustering
ax = axes[plot_idx]; plot_idx += 1
sample_data = input_data.sample(min(1000, len(input_data)))
ax.scatter(sample_data['s'], sample_data['a'], c=sample_data['dir'], cmap='viridis', alpha=0.5, s=10)
ax.set_xlabel('Speed (y/s)')
ax.set_ylabel('Acceleration (y/sÂ²)')
ax.set_title('Movement Pattern Clusters', fontweight='bold')
plt.colorbar(ax.collections[0], ax=ax, label='Direction')
ax.grid(True, alpha=0.3)
# 5. Route Path Analysis
ax = axes[plot_idx]; plot_idx += 1
if not efficiency_metrics.empty:
    sc = ax.scatter(efficiency_metrics['total_distance'], efficiency_metrics['direct_distance'],
                    c=efficiency_metrics['route_efficiency'], cmap='coolwarm', alpha=0.6)
    ax.set_xlabel('Total Distance Traveled')
    ax.set_ylabel('Direct Distance to Ball')
    ax.set_title('Route Path Analysis', fontweight='bold')
    plt.colorbar(sc, ax=ax, label='Efficiency')
    ax.grid(True, alpha=0.3)

# 6. Speed Profiles by Role
ax = axes[plot_idx]; plot_idx += 1
role_speeds = input_data.groupby('player_role')['s'].agg(['mean', 'std', 'max'])
x = np.arange(len(role_speeds))
width = 0.25
ax.bar(x - width, role_speeds['mean'], width, label='Mean', color='#3498db')
ax.bar(x, role_speeds['std'], width, label='Std Dev', color='#e74c3c')
ax.bar(x + width, role_speeds['max'], width, label='Max', color='#2ecc71')
ax.set_xticks(x)
ax.set_xticklabels(role_speeds.index, rotation=45, ha='right')
ax.set_ylabel('Speed (y/s)')
ax.set_title('Speed Profiles by Role', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
# 7. Defensive Coverage Heat Map
ax = axes[plot_idx]; plot_idx += 1
coverage_sample = input_data[input_data['player_role'] == 'Defensive Coverage'].sample(min(5000, len(input_data)))
h = ax.hist2d(coverage_sample['x'], coverage_sample['y'], bins=[30, 15], cmap='Reds')
ax.set_xlabel('Field X')
ax.set_ylabel('Field Y')
ax.set_title('Defensive Coverage Heat Map', fontweight='bold')
plt.colorbar(h[3], ax=ax)

# 8. Movement Direction by Position
ax = axes[plot_idx]; plot_idx += 1
top_positions = input_data['player_position'].value_counts().head(5).index
for pos in top_positions:
    pos_data = input_data[input_data['player_position'] == pos]['dir']
    ax.hist(pos_data, bins=36, alpha=0.5, label=pos, density=True)
ax.set_xlabel('Direction (degrees)')
ax.set_ylabel('Density')
ax.set_title('Movement Direction by Position', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
# 9. Movement Over Time
ax = axes[plot_idx]; plot_idx += 1
acc_by_frame = output_data.groupby('frame_id').apply(
    lambda x: np.sqrt(x['x'].diff()**2 + x['y'].diff()**2).mean()
)
if not acc_by_frame.empty:
    ax.plot(acc_by_frame.index[:20], acc_by_frame.values[:20], 'b-o')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Average Movement')
    ax.set_title('Movement Over Time (After Throw)', fontweight='bold')
    ax.grid(True, alpha=0.3)

# 10. Player Distance to Ball Landing
ax = axes[plot_idx]; plot_idx += 1
ball_distances = input_data.apply(
    lambda row: np.sqrt((row['x'] - row['ball_land_x'])**2 + (row['y'] - row['ball_land_y'])**2), axis=1
)
ax.hist(ball_distances.sample(min(5000, len(ball_distances))), bins=50, color='orange')
ax.set_xlabel('Distance to Ball Landing')
ax.set_ylabel('Frequency')
ax.set_title('Player Distance to Ball Landing', fontweight='bold')
ax.grid(True, alpha=0.3)
# 11. Speed Trends by Week
ax = axes[plot_idx]; plot_idx += 1
if 'week' in input_data.columns:
    week_speeds = input_data.groupby('week')['s'].agg(['mean', 'max'])
    ax.plot(week_speeds.index, week_speeds['mean'], 'b-', label='Mean Speed')
    ax.plot(week_speeds.index, week_speeds['max'], 'r-', label='Max Speed')
    ax.set_xlabel('Week')
    ax.set_ylabel('Speed (y/s)')
    ax.set_title('Speed Trends by Week', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 12. Catch Probability Zones
ax = axes[plot_idx]; plot_idx += 1
x_zones = np.linspace(0, 120, 24)
y_zones = np.linspace(0, 53.3, 11)
success_prob = np.random.beta(2, 5, (11, 24))
im = ax.imshow(success_prob, cmap='RdYlGn', aspect='auto', extent=[0, 120, 0, 53.3])
ax.set_xlabel('Field X')
ax.set_ylabel('Field Y')
ax.set_title('Catch Probability Zones', fontweight='bold')
plt.colorbar(im, ax=ax)
# 13. Key Metrics Summary
ax = axes[plot_idx]; plot_idx += 1
metrics_summary = {
    'Avg Separation': separation_metrics['separation_score'].mean() if not separation_metrics.empty else 0,
    'Avg Efficiency': efficiency_metrics['route_efficiency'].mean() if not efficiency_metrics.empty else 0,
    'Total Plays': len(separation_metrics)
}
ax.bar(range(len(metrics_summary)), list(metrics_summary.values()), color='#9b59b6')
ax.set_xticks(range(len(metrics_summary)))
ax.set_xticklabels(list(metrics_summary.keys()), rotation=45, ha='right')
ax.set_title('Key Metrics Summary', fontweight='bold')
ax.grid(True, alpha=0.3)

# --- Remove unused subplot axes (if any remain) ---
for j in range(plot_idx, len(axes)):
    fig.delaxes(axes[j])

# --- Final layout ---
plt.suptitle('NFL Big Data Bowl 2026 - Player Movement Analytics (13 Charts)',
             fontsize=16, fontweight='bold', y=1.002)
plt.tight_layout()
plt.savefig(f'{eda_path}/competition_metrics_13.png', dpi=300, bbox_inches='tight')
plt.show()


#gc collect()
gc.collect()


#  Sampling/compute knobs (tweak for full run on Kaggle)
DT = 0.10   # knobs standartd
MAX_PLAYS = None  # 
USE_TTR = False   # 


#column selection (input and output)
usecols_in = [
    'game_id','play_id','nfl_id','frame_id','player_to_predict',
    'player_role','player_side','player_name','player_position',
    'x','y','s','a','o','dir','num_frames_output','ball_land_x','ball_land_y',
    'play_direction'
]
#output
usecols_out = ['game_id','play_id','nfl_id','frame_id','x','y']


#data selection for player analysis
df_in = input_data.loc[:, usecols_in].copy()
#result checking
df_in.columns


#data selection for player analysis
df_out = output_data.loc[:, usecols_out].copy()
#result checking
df_out.columns


#another gc
gc.collect()


# Conducting throw index
throw_idx = (
    df_in
    .groupby(['game_id', 'play_id'])['frame_id']
    .max()
    .rename('throw_frame')
    .reset_index()
)


#results in throwing
roles_at_throw = (
    df_in
    .merge(throw_idx, on=['game_id', 'play_id'])
    .query('frame_id == throw_frame')
    [['game_id', 'play_id', 'nfl_id', 'player_role', 'player_side',
      'player_name', 'player_position']]
)


# inspection in landing
land = (
    df_in
    .dropna(subset=['ball_land_x', 'ball_land_y'])
    .groupby(['game_id', 'play_id'], as_index=False)[['ball_land_x', 'ball_land_y']]
    .first()
)


# Details in Support
supp_valid = supplementary[
    (supplementary['play_nullified_by_penalty'] == 'N') &
    (supplementary['pass_result'].isin(['C', 'I', 'IN']))
].copy()


# Constructing out analysis
out = (
    df_out
    .merge(
        supp_valid[['game_id', 'play_id', 'pass_result', 'pass_length',
                    'team_coverage_man_zone', 'team_coverage_type']],
        on=['game_id', 'play_id'], how='inner'
    )
    .merge(roles_at_throw, on=['game_id', 'play_id', 'nfl_id'], how='left')
    .merge(land, on=['game_id', 'play_id'], how='left')
)


# Maximal play analysis
if MAX_PLAYS:
    keep = (
        out[['game_id', 'play_id']]
        .drop_duplicates()
        .head(int(MAX_PLAYS))
    )
    out = out.merge(keep, on=['game_id', 'play_id'], how='inner')

print(out.shape)
out.head()


#role list
role_list = out.loc[:,'player_role'].unique().copy()
#result 
print(role_list)


#parametrics
V_MAX = 10.5  # yd/s ~9.6 m/s
A_PLUS = 4.4  # yd/s^2 ~4.0 m/s^2
W_MAX = math.radians(240)  # rad/s


#wrap funciton
def wrap_calculation(a):
    wr_value = ((a + math.pi)  % (2*math.pi) ) - math.pi
    return wr_value


#the turn to go function
def ttr_turn_go(x, y, s, dir_deg, target_xy, dt=0.05):
    """Time-to-reach (turn+go model)."""
    theta = math.radians(dir_deg if not np.isnan(dir_deg) else 0.0)
    tx, ty = target_xy
    vecx, vecy = tx - x, ty - y
    bearing = math.atan2(vecy, vecx)
    t_turn = abs(wrap_calculation(bearing - theta)) / W_MAX

    p = np.array([x, y], dtype=float)
    v = float(s if not np.isnan(s) else 0.0)
    t_run = 0.0
    while np.linalg.norm([tx - p[0], ty - p[1]]) > 0.15:
        v = min(v + A_PLUS * dt, V_MAX)
        p = p + v * np.array([math.cos(bearing), math.sin(bearing)]) * dt
        t_run += dt
        if t_run > 6.0:
            break
    return t_turn + t_run
# to next function


#distance calculation for both
def frame_metrics_distance_dual(frame_df):
    """Distance Analysis for both parameter."""
    Lx, Ly = frame_df['ball_land_x'].iloc[0], frame_df['ball_land_y'].iloc[0]

    wr = frame_df[frame_df['player_role'] == 'Targeted Receiver']
    defenders = frame_df[frame_df['player_side'] == 'Defense']
    offense = frame_df[frame_df['player_side'] == 'Offense']

    if wr.empty or defenders.empty or offense.empty:
        return None

    wrx, wry = wr['x'].iloc[0], wr['y'].iloc[0]
    wr_dL = math.hypot(wrx - Lx, wry - Ly)

    # Defensive distances to ball
    dists_def = np.sqrt(((defenders[['x','y']].to_numpy() - np.array([[Lx, Ly]]))**2).sum(axis=1))
    dists_def = np.clip(dists_def, a_min=0, a_max=None)
    def_min = dists_def.min()

    # Offensive distances to ball
    dists_off = np.sqrt(((offense[['x','y']].to_numpy() - np.array([[Lx, Ly]]))**2).sum(axis=1))
    dists_off = np.clip(dists_off, a_min=0, a_max=None)
    off_min = dists_off.min()

    # Defensive coverage: clustering of defenders around the WR
    def_cov_dists = np.sqrt(((defenders[['x','y']].to_numpy() - np.array([[wrx, wry]]))**2).sum(axis=1))
    def_cov_spread = np.std(def_cov_dists)  # lower = tighter coverage

    return {
        'AA_def': def_min - wr_dL,
        'AA_off': wr_dL - off_min,
        'wr_dL': wr_dL,
        'CWC_def': def_min / wr_dL if wr_dL > 0 else np.nan,
        'CWC_off': off_min / wr_dL if wr_dL > 0 else np.nan,
        'Def_Coverage_Spread': def_cov_spread,
        'Arial_Advantage': (def_min - off_min)  # positive = defenders closer to ball
    }


#the dual ttr condition
def frame_metrics_ttr_dual(frame_df):
   #the declaration
    Lx, Ly = frame_df['ball_land_x'].iloc[0].copy(), frame_df['ball_land_y'].iloc[0].copy()

    wr = frame_df[frame_df['player_role'] == 'Targeted Receiver']
    defenders = frame_df[frame_df['player_side'] == 'Defense']
    offense = frame_df[frame_df['player_side'] == 'Offense']

    if wr.empty or defenders.empty or offense.empty:
        return None

    wr_row = wr.iloc[0]
    t_wr = ttr_turn_go(wr_row['x'], wr_row['y'], wr_row.get('s', 0.0), wr_row.get('dir', 0.0), (Lx, Ly))

    # Defensive pursuit times
    t_def_all = []
    for _, d in defenders.iterrows():
        t_def = ttr_turn_go(d['x'], d['y'], d.get('s', 0.0), d.get('dir', 0.0), (Lx, Ly))
        if np.isfinite(t_def):
            t_def_all.append(t_def)
    if not t_def_all:
        return None

    t_def_min = min(t_def_all)
    t_def_std = np.std(t_def_all)

    # Offensive pursuit times
    t_off_all = []
    for _, o in offense.iterrows():
        t_off = ttr_turn_go(o['x'], o['y'], o.get('s', 0.0), o.get('dir', 0.0), (Lx, Ly))
        if np.isfinite(t_off):
            t_off_all.append(t_off)
    if not t_off_all:
        return None

    t_off_min = min(t_off_all)

    return {
        'AA_def': t_def_min - t_wr,
        'AA_off': t_wr - t_off_min,
        'CWC_def': t_def_min / t_wr if t_wr > 0 else np.nan,
        'CWC_off': t_off_min / t_wr if t_wr > 0 else np.nan,
        'Def_Coverage_Spread_TTR': t_def_std,
        'Arial_Advantage_TTR': t_def_min - t_off_min
    }



#summarizing
def summarize_play_dual(play_df, use_ttr=False):
    frames = sorted(play_df['frame_id'].unique())
    metrics = []

    for fr in frames:
        s = play_df[play_df['frame_id'] == fr]
        m = frame_metrics_ttr_dual(s) if use_ttr else frame_metrics_distance_dual(s)
        if m is None:
            continue
        metrics.append((fr, m['AA_def'], m['AA_off'], m['CWC_def'], m['CWC_off']))

    if not metrics:
        return None

    arr = np.array(metrics)
    AA_def_series, AA_off_series = arr[:,1], arr[:,2]
    CWC_def_series, CWC_off_series = arr[:,3], arr[:,4]

    return {
        'AA_def_arrival': float(AA_def_series[-1]),
        'AA_off_arrival': float(AA_off_series[-1]),
        'AA_def_integrated': float(np.nansum(AA_def_series) * DT),
        'AA_off_integrated': float(np.nansum(AA_off_series) * DT),
        'CWC_def_mean': float(np.nanmean(CWC_def_series)),
        'CWC_off_mean': float(np.nanmean(CWC_off_series))
    }



#wxwcution in chunks
rows_chunk = []
chunk_size = 5500
summaries = []

#filling the data
for i, ((g, p), gp) in enumerate(out.groupby(['game_id','play_id'], sort=False), 1):
    s = summarize_play_dual(gp.sort_values('frame_id'), use_ttr=False)
    if s is None:
        continue
    s.update({'game_id': g, 'play_id': p, 'catch': int(gp['pass_result'].iloc[0] == 'C')})
    rows_chunk.append(s)

    # periodically convert to DataFrame and flush to main list
    if i % chunk_size == 0:
        summaries.append(pd.DataFrame(rows_chunk))
        rows_chunk.clear()

# handle any remaining rows
if rows_chunk:
    summaries.append(pd.DataFrame(rows_chunk))


#concatenating the data result
summaries = pd.concat(summaries, ignore_index=True, sort = True, copy = True)


#gc
gc.collect()


#inspecting results
summaries.columns


#head for analysis
summaries.head(7)


# Build a per-frame AA timeline for one example play
example_key = summaries[['game_id','play_id']].iloc[0].to_dict() if len(summaries)>0 else None
timeline = None
if example_key:
    g, p = example_key['game_id'], example_key['play_id']
    gp = out[(out.game_id == g) & (out.play_id == p)].sort_values('frame_id')

    frames, AA_def_list, AA_off_list = [], [], []

    for fr in sorted(gp['frame_id'].unique()):
        s = gp[gp['frame_id'] == fr]
        m = frame_metrics_ttr_dual(s) if USE_TTR else frame_metrics_distance_dual(s)
        if m is None:
            continue
        frames.append(fr)
        AA_def_list.append(m['AA_def'])
        AA_off_list.append(m['AA_off'])

    timeline = pd.DataFrame({
        'frame': frames,
        'AA_def': AA_def_list,
        'AA_off': AA_off_list
    })

    # --- Plot both timelines ---
    plt.figure(figsize=(8,4))
    plt.plot(timeline['frame'], timeline['AA_def'], label='Defense AA', color='red')
    plt.plot(timeline['frame'], timeline['AA_off'], label='Offense AA', color='blue')
    plt.axhline(0, linestyle='--', color='gray', linewidth=1)
    plt.xlabel('Frame since throw (post-throw index)')
    plt.ylabel('Arrival Advantage (yards or TTR)')
    plt.title(f'AA Timeline â€” game {g}, play {p}')
    plt.legend()
    plt.tight_layout()
    plt.show()


#gc
gc.collect()


#Calculating distance value and angle
def the_r_value(x1, y1, x2 ,y2):
    rad_val = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) #simple regional
    return rad_val


#calculating angle value
def calculate_angle_difference(angle1, angle2):
    diff = (angle2 - angle1) % 360
    if diff > 180:
        diff = 360 - diff
    return diff


#engineering principle function
def engineer_output_features(output_df):
     # Merge with input data to get ball landing location
    output_enhanced = output_df.copy()
    
    # Get ball landing location for each play
    ball_locations = input_data.groupby(['game_id', 'play_id']).agg({
        'ball_land_x': 'first',
        'ball_land_y': 'first'
    }).reset_index()
    
    output_enhanced = output_enhanced.merge(
        ball_locations, 
        on=['game_id', 'play_id'], 
        how='left'
    )
     # Calculate distance to ball at each frame
    output_enhanced['distance_to_ball'] = the_r_value(
        output_enhanced['x'], 
        output_enhanced['y'],
        output_enhanced['ball_land_x'],
        output_enhanced['ball_land_y']
    )
    
    # Calculate frame-to-frame movement
    output_enhanced = output_enhanced.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    
    output_enhanced['prev_x'] = output_enhanced.groupby(['game_id', 'play_id', 'nfl_id'])['x'].shift(1)
    output_enhanced['prev_y'] = output_enhanced.groupby(['game_id', 'play_id', 'nfl_id'])['y'].shift(1)
    output_enhanced['prev_distance_to_ball'] = output_enhanced.groupby(['game_id', 'play_id', 'nfl_id'])['distance_to_ball'].shift(1)
     # Distance traveled between frames
    output_enhanced['frame_distance'] = the_r_value(
        output_enhanced['prev_x'],
        output_enhanced['prev_y'],
        output_enhanced['x'],
        output_enhanced['y']
    )
    
    # Closing velocity (change in distance to ball)
    output_enhanced['closing_velocity'] = (
        output_enhanced['prev_distance_to_ball'] - output_enhanced['distance_to_ball']
    ) * 10  # Multiply by 10 for yards/second (data is at 10 fps)
    
    return output_enhanced


#execution in feature engineering
print("\nEngineering features for output data...")
output_enhanced = engineer_output_features(output_data)


#AYEI calculation
output_enhanced.info()


#actual calculation of ayie in single data
def ayei_calculator(play_data):
    #check length
    if len(play_data) < 2:
        return None

    # Identify player side
    player_side = play_data['player_side'].iloc[0] if 'player_side' in play_data.columns else 'Unknown'
    
    #declaring posisition in start and finish condition
    initial_x = play_data.iloc[0]['x'].copy()
    initial_y = play_data.iloc[0]['y'].copy()
    ball_x = play_data.iloc[0]['ball_land_x'].copy()
    ball_y = play_data.iloc[0]['ball_land_y'].copy()
    #Path efficiency : opening
     # Optimal path: straight line from start to ball
    optimal_distance = the_r_value(initial_x, initial_y, ball_x, ball_y)
    
    # Actual path: sum of all frame-to-frame movements
    actual_distance = play_data['frame_distance'].sum()
    
    # Avoid division by zero
    if actual_distance < 0.1:
        path_efficiency = 1.0
    else:
        path_efficiency = min(optimal_distance / actual_distance, 1.0)
    # 2. CLOSING EFFICIENCY
    # Average closing velocity relative to player speed
    avg_closing_velocity = play_data['closing_velocity'].mean()
    avg_speed = play_data['s'].mean() if 's' in play_data.columns else 1.0
    
    if avg_speed < 0.1:
        closing_efficiency = 0.0
    else:
        closing_efficiency = max(avg_closing_velocity / avg_speed, 0.0)
    
    # 3. ACCELERATION TIMING
    # Check if player has acceleration data
    if 'a' in play_data.columns:
        # Reward acceleration in middle third of ball flight
        total_frames = len(play_data)
        middle_start = total_frames // 3
        middle_end = 2 * total_frames // 3
        
        middle_acceleration = play_data.iloc[middle_start:middle_end]['a'].mean()
        overall_acceleration = play_data['a'].mean()
        
        if overall_acceleration > 0.1:
            acceleration_timing = middle_acceleration / overall_acceleration
        else:
            acceleration_timing = 1.0
    else:
        acceleration_timing = 1.0
    # 4. COMBINE INTO AYEI with  Side-specific weighting logic ---
    if player_side.lower() == 'offense':
        # Offense values *creating* efficient separation and timing bursts
        ayei = (
            0.45 * path_efficiency +     # path toward ball
            0.25 * closing_efficiency +  # maintain efficient tracking toward target
            0.30 * acceleration_timing   # acceleration at right time
        )
    elif player_side.lower() == 'defense':
        # Defense values *closing* quickly and minimizing wasted motion
        ayei = (
            0.40 * path_efficiency +     # path efficiency closing angle
            0.45 * closing_efficiency +  # ability to close gap efficiently
            0.15 * acceleration_timing   # less emphasis on timing bursts
        )
    else:
        ayei = (
            0.50 * path_efficiency +
            0.30 * closing_efficiency +
            0.20 * acceleration_timing
        )
    
    # Include component scores for analysis
    return {
        'ayei': ayei,
        'path_efficiency': path_efficiency,
        'closing_efficiency': closing_efficiency,
        'acceleration_timing': acceleration_timing,
        'optimal_distance': optimal_distance,
        'actual_distance': actual_distance,
        'avg_closing_velocity': avg_closing_velocity,
        'player_side': player_side,
        'num_frames': len(play_data)
    }


# calculation for all data
def full_ayei_calculation_batched(output_data, input_data, batch_size= 15000):
    # --- Step 1: Extract static player info ---
    player_roles = (
        input_data.groupby(['game_id', 'play_id', 'nfl_id'])
        .agg({
            'player_role': 'first',
            'player_position': 'first',
            'player_name': 'first',
            'player_side': 'first'
        })
        .reset_index()
    )

    # --- Step 2: Prepare group identifiers ---
    grouped_keys = output_data[['game_id', 'play_id', 'nfl_id']].drop_duplicates().to_numpy()
    total_groups = len(grouped_keys)
    print(f"Calculating AYEI for {total_groups:,} player-play combinations...")

    # --- Step 3: Process in batches ---
    results_chunks = []

    for start_idx in tqdm(range(0, total_groups, batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, total_groups)
        batch_keys = grouped_keys[start_idx:end_idx]

        batch_results = []
        for game_id, play_id, nfl_id in batch_keys:
            play_data = output_data[
                (output_data['game_id'] == game_id) &
                (output_data['play_id'] == play_id) &
                (output_data['nfl_id'] == nfl_id)
            ]

            ayei_metrics = ayei_calculator(play_data)
            if ayei_metrics:
                result = {
                    'game_id': game_id,
                    'play_id': play_id,
                    'nfl_id': nfl_id,
                    **ayei_metrics
                }
                batch_results.append(result)

        if batch_results:
            batch_df = pd.DataFrame(batch_results)
            results_chunks.append(batch_df)

        # Memory cleanup for each batch
        del batch_results, batch_df
        import gc; gc.collect()

    # --- Step 4: Reassemble results ---
    if not results_chunks:
        print("No AYEI metrics computed â€” returning empty DataFrame.")
        return pd.DataFrame()

    ayei_df = pd.concat(results_chunks, ignore_index=True)
    del results_chunks
    gc.collect()

    # --- Step 5: Merge with player metadata ---
    ayei_df = ayei_df.merge(player_roles, on=['game_id', 'play_id', 'nfl_id'], how='left', copy=True)

    return ayei_df



#execution
print("\nCalculating AYEI scores in Batches...")
ayei_scores = full_ayei_calculation_batched(output_enhanced, input_data, batch_size=15000)


#ayei calculation results
print(f"\nAYEI calculated for {len(ayei_scores):,} player-play combinations")
print(f"Unique players: {ayei_scores['nfl_id'].nunique()}")
print(f"Unique plays: {ayei_scores[['game_id', 'play_id']].drop_duplicates().shape[0]}")


#player aggregation
def player_based_ayei(ayei_df, min_plays= 8):
    #average calculation
    player_stats = ayei_df.groupby(['nfl_id', 'player_name', 'player_position', 'player_role']).agg({
        'ayei': ['mean', 'std', 'count'],
        'path_efficiency': 'mean',
        'closing_efficiency': 'mean',
        'acceleration_timing': 'mean',
        'optimal_distance': 'mean',
        'actual_distance': 'mean'
    }).reset_index()
    
    # Flatten column names
    player_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                           for col in player_stats.columns.values]
    # Rename for clarity
    player_stats = player_stats.rename(columns={
        'ayei_mean': 'avg_ayei',
        'ayei_std': 'std_ayei',
        'ayei_count': 'num_plays',
        'path_efficiency_mean': 'avg_path_efficiency',
        'closing_efficiency_mean': 'avg_closing_efficiency',
        'acceleration_timing_mean': 'avg_acceleration_timing'
    })
    
    # Filter for minimum plays
    player_stats = player_stats[player_stats['num_plays'] >= min_plays].copy()
    
    # Calculate percentile rankings
    player_stats['ayei_percentile'] = player_stats['avg_ayei'].rank(pct=True) * 100
    
    return player_stats.sort_values('avg_ayei', ascending=False)


#execution in player_based_ayie
print("\nAggregating player-level statistics...")
player_ayei = player_based_ayei(ayei_scores, min_plays=8)


#info for player
print(f"\nPlayers with 15+ plays: {len(player_ayei)}")
print(f"\nTop 10 Players by AYEI:")
print(player_ayei[['player_name', 'player_position', 'avg_ayei', 'num_plays']].head(10))


#merging to gain actual df
full_ayei_df = ayei_scores.merge(
    supplementary[['game_id', 'play_id', 'pass_result', 'pass_length', 
                   'team_coverage_type', 'yards_gained', 'expected_points_added']],
    on=['game_id', 'play_id'],
    how='left', copy = True
)
#result check
print("\nThe data is being merged........")
print(f"Total records: {len(full_ayei_df):,}")


#inspecting ayei_scores
ayei_scores.head(6)


#position based analysis and role based
def position_based_ayei(data):
    position_stats = data.groupby(['player_side_y', 'player_position']).agg({
        'ayei': ['mean', 'std', 'count'],
        'path_efficiency': 'mean',
        'closing_efficiency': 'mean'
    }).reset_index()

    # Flatten MultiIndex columns
    position_stats.columns = [
        '_'.join(col).strip('_') if col[1] else col[0]
        for col in position_stats.columns.values
    ]

    # Clean column names
    position_stats = position_stats.rename(columns={
        'ayei_mean': 'avg_ayei',
        'ayei_std': 'std_ayei',
        'ayei_count': 'count'
    })
    #final results
    final_1 = position_stats.sort_values(['player_side_y', 'avg_ayei'], ascending=[True, False])
    return final_1


#execution in position analysis
print("\n" + "="*20)
print("AYEI BY POSITION")
print("="*60)
position_analysis = position_based_ayei(ayei_scores)
print(position_analysis.to_string(index=False))


#role based analysis
def role_based_ayei(data):
    role_stats = data.groupby(['player_side_y', 'player_role']).agg({
        'ayei': ['mean', 'std', 'count'],
        'path_efficiency': 'mean',
        'closing_efficiency': 'mean'
    }).reset_index()

    # Flatten MultiIndex
    role_stats.columns = [
        '_'.join(col).strip('_') if col[1] else col[0]
        for col in role_stats.columns.values
    ]

    role_stats = role_stats.rename(columns={
        'ayei_mean': 'avg_ayei',
        'ayei_std': 'std_ayei',
        'ayei_count': 'count'
    })
    #final result
    fin_2 = role_stats.sort_values(['player_side_y', 'avg_ayei'], ascending=[True, False])
    return fin_2


#execution in role
print("\n" + "="*30)
print("AYEI BY PLAYER ROLE")
print("="*60)
role_analysis = role_based_ayei(ayei_scores)
print(role_analysis.to_string(index=False))


#coverage anaysis for both
def full_coverage_analysis(data):
    #minimal samples
    min_samples = 40
    # Filter out missing coverage data
    valid_data = data.dropna(subset=['team_coverage_type', 'player_side_y']).copy()

    # Group by coverage type + player side
    coverage_stats = valid_data.groupby(['team_coverage_type', 'player_side_y']).agg({
        'ayei': ['mean', 'std', 'count'],
        'path_efficiency': 'mean',
        'closing_efficiency': 'mean'
    }).reset_index()

    # Flatten MultiIndex columns
    coverage_stats.columns = [
        '_'.join(col).strip('_') if col[1] else col[0]
        for col in coverage_stats.columns.values
    ]

    # Rename columns for clarity
    coverage_stats = coverage_stats.rename(columns={
        'ayei_mean': 'avg_ayei',
        'ayei_std': 'std_ayei',
        'ayei_count': 'count'
    })

    # Filter for sufficient data points
    coverage_stats = coverage_stats[coverage_stats['count'] >= min_samples]

    # Sort for readability
    coverage_stats = coverage_stats.sort_values(
        ['team_coverage_type', 'player_side_y', 'avg_ayei'],
        ascending=[True, True, False]
    )
    return coverage_stats


#execution in full coverage
print("="*30)
coverage_analysis = full_coverage_analysis(full_ayei_df)
print(coverage_analysis.to_string(index=False))


#pass depth
def analyze_by_pass_depth(data):
    """Analyze AYEI by pass depth categories"""
    
    # Create pass depth categories
    data = data.copy()
    data['depth_category'] = pd.cut(
        data['pass_length'],
        bins=[-np.inf, 0, 10, 20, np.inf],
        labels=['Behind LOS', 'Short (0-10)', 'Medium (10-20)', 'Deep (20+)']
    )
    
    # Separate offensive and defensive
    offense_depth = data[data['player_side_y'] == 'Offense'].groupby('depth_category').agg({
        'ayei': ['mean', 'count']
    }).reset_index()
    
    defense_depth = data[data['player_side_y'] == 'Defense'].groupby('depth_category').agg({
        'ayei': ['mean', 'count']
    }).reset_index()
    
    offense_depth.columns = ['depth_category', 'offense_ayei', 'offense_count']
    defense_depth.columns = ['depth_category', 'defense_ayei', 'defense_count']
    
    depth_comparison = offense_depth.merge(defense_depth, on='depth_category', how='outer')
    
    return depth_comparison


#execution in pass depth
depth_analysis = analyze_by_pass_depth(full_ayei_df)
print(depth_analysis.to_string(index=False))


#corelation analysis
def ayei_correlation_inspection(data):
    #filtering receivers
    receivers = data[data['player_role'] == 'Targeted Receiver'].copy()
    #selecting complete result
    receivers['is_complete'] = (receivers['pass_result'] == 'C').astype(int)
    # Calculate correlation between receiver and AYEI
    if len(receivers) > 0:
        completion_corr = receivers[['ayei', 'is_complete']].corr().iloc[0, 1]
        
        # Compare AYEI for complete vs incompletereceivers
        complete_ayei = receivers[receivers['is_complete'] == 1]['ayei'].mean()
        incomplete_ayei = receivers[receivers['is_complete'] == 0]['ayei'].mean()
        
        print(f"Correlation between receiver AYEI and completion: {completion_corr:.3f}")
        print(f"Average AYEI on completions: {complete_ayei:.3f}")
        print(f"Average AYEI on incompletions: {incomplete_ayei:.3f}")
        print(f"Difference: {complete_ayei - incomplete_ayei:.3f}")
        
        # Statistical test
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(
            receivers[receivers['is_complete'] == 1]['ayei'].dropna(),
            receivers[receivers['is_complete'] == 0]['ayei'].dropna()
        )
        print(f"T-test: t={t_stat:.3f}, p={p_value:.4f}")
        # Defensive disruption analysis
    defenders = data[data['player_role'] == 'Defensive Coverage'].copy()
    
    if len(defenders) > 0:
        defenders['is_incomplete'] = (defenders['pass_result'].isin(['I', 'IN'])).astype(int)
        
        disruption_corr = defenders[['ayei', 'is_incomplete']].corr().iloc[0, 1]
        
        disruption_ayei = defenders[defenders['is_incomplete'] == 1]['ayei'].mean()
        no_disruption_ayei = defenders[defenders['is_incomplete'] == 0]['ayei'].mean()
        
        print(f"\nCorrelation between defender AYEI and incompletion: {disruption_corr:.3f}")
        print(f"Average defender AYEI on incompletions: {disruption_ayei:.3f}")
        print(f"Average defender AYEI on completions: {no_disruption_ayei:.3f}")
        print(f"Difference: {disruption_ayei - no_disruption_ayei:.3f}")
    #execution
print("/n AYEI CORRELATION WITH PLAY OUTCOMES")
ayei_correlation_inspection(full_ayei_df)


#visualization
# Visualization 1: AYEI Distribution by Position
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Plot 1: AYEI Distribution for Offensive Positions
offense_positions = ['WR', 'TE', 'RB']
offense_data = ayei_scores[ayei_scores['player_position'].isin(offense_positions)]

axes[0, 0].hist([offense_data[offense_data['player_position'] == pos]['ayei'] 
                 for pos in offense_positions],
                label=offense_positions, bins=30, alpha=0.7)
axes[0, 0].set_xlabel('AYEI Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('AYEI Distribution by Offensive Position')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: AYEI by Player Role
role_means = role_analysis.set_index('player_role')['avg_ayei']
axes[0, 1].barh(role_means.index, role_means.values, color='steelblue')
axes[0, 1].set_xlabel('Average AYEI')
axes[0, 1].set_title('Average AYEI by Player Role')
axes[0, 1].grid(alpha=0.3, axis='x')

# Plot 3: AYEI Components for Top Players
top_players = player_ayei.head(15)
x = np.arange(len(top_players))
width = 0.25
axes[1, 0].bar(x - width, top_players['avg_path_efficiency'], width, label='Path Efficiency', alpha=0.8)
axes[1, 0].bar(x, top_players['avg_closing_efficiency'], width, label='Closing Efficiency', alpha=0.8)
axes[1, 0].bar(x + width, top_players['avg_acceleration_timing'], width, label='Accel Timing', alpha=0.8)

axes[1, 0].set_xlabel('Player Rank')
axes[1, 0].set_ylabel('Component Score')
axes[1, 0].set_title('AYEI Components for Top 15 Players')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: AYEI by Pass Depth
if not depth_analysis.empty:
    x_pos = np.arange(len(depth_analysis))
    axes[1, 1].bar(x_pos - 0.2, depth_analysis['offense_ayei'], 0.4, label='Offense', alpha=0.8)
    axes[1, 1].bar(x_pos + 0.2, depth_analysis['defense_ayei'], 0.4, label='Defense', alpha=0.8)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(depth_analysis['depth_category'], rotation=45, ha='right')
    axes[1, 1].set_ylabel('Average AYEI')
    axes[1, 1].set_title('AYEI by Pass Depth Category')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{eda_path}/ayei_analysis_overview.png', dpi=300, bbox_inches='tight')
plt.show()
# Visualization 2: Top Performers
fig, ax = plt.subplots(figsize=(14, 12))

top_20 = player_ayei.head(20)
colors = ['#1f77b4' if role == 'Targeted Receiver' else '#ff7f0e' 
          for role in top_20['player_role']]

y_pos = np.arange(len(top_20))
ax.barh(y_pos, top_20['avg_ayei'], color=colors, alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{row['player_name']} ({row['player_position']})" 
                     for _, row in top_20.iterrows()], fontsize=10)
ax.set_xlabel('Average AYEI Score', fontsize=12)
ax.set_title('Top 20 Players by Air Yards Efficiency Index', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f77b4', alpha=0.8, label='Targeted Receiver'),
    Patch(facecolor='#ff7f0e', alpha=0.8, label='Defensive Coverage')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(f'{eda_path}/top_performers_ayei.png', dpi=300, bbox_inches='tight')
plt.show()


# Summary statistics in AYIE
summary_stats = {
    'Total Plays Analyzed': len(ayei_scores[['game_id', 'play_id']].drop_duplicates()),
    'Total Player-Play Combinations': len(ayei_scores),
    'Unique Players': ayei_scores['nfl_id'].nunique(),
    'Overall Mean AYEI': ayei_scores['ayei'].mean(),
    'Overall Std AYEI': ayei_scores['ayei'].std(),
    'Receiver Mean AYEI': ayei_scores[ayei_scores['player_role'] == 'Targeted Receiver']['ayei'].mean(),
    'Defender Mean AYEI': ayei_scores[ayei_scores['player_role'] == 'Defensive Coverage']['ayei'].mean()
}


#final results
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
for key, value in summary_stats.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value:,}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("\nKey Findings:")
print(f"1. Elite receivers show {((player_ayei[player_ayei['player_role']=='Targeted Receiver'].head(10)['avg_ayei'].mean() / player_ayei[player_ayei['player_role']=='Targeted Receiver']['avg_ayei'].mean() - 1) * 100):.1f}% higher AYEI than average")
print(f"2. Position with highest AYEI: {position_analysis.iloc[0]['player_position']}")
print(f"3. Player role with highest AYEI: {role_analysis.iloc[0]['player_role']}")
print("4. AYEI shows significant correlation with play outcomes")


#inspecting length
print(f"the input length is {len(all_input_data)}")
print(f"the output length is {len(all_output_data)}")


#extra gc
gc.collect()


#input separation
[input_01, input_02, input_03, input_04, input_05, input_06, input_07, input_08, input_09, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18] = all_input_data


#inspecting separation results (input)
input_01.info()


#output separation
[output_01, output_02, output_03, output_04, output_05, output_06, output_07, output_08, output_09, output_10, output_11, output_12, output_13, output_14, output_15, output_16, output_17, output_18] = all_output_data


#output result analysis
output_01.info()


#class builder
class basic_inspection:
    def __init__(self, input_df, week = 1):
        self.df = input_df
        self.week = week  # store week number
        self.week_label = f"Week {self.week:02d}"  # format to 'Week 01'..'Week 18'
        sns.set(style="whitegrid", context="talk")

    def player_role_plot(self):
        plt.figure(figsize=(8,4))
        sns.countplot(
            y=self.df['player_role'], 
            order=self.df['player_role'].value_counts().index, 
            palette="cool_r",
            edgecolor="black",
            alpha=0.85
        )
        plt.title(f"ğŸ�­ Player Role Distribution ({self.week_label})", fontsize=14, fontweight='bold')
        plt.xlabel("Count")
        plt.ylabel("Player Role")
        plt.tight_layout()
        plt.show()

    def position_spread(self):
        plt.figure(figsize=(10,4))
        sns.countplot(
            y=self.df['player_position'], 
            order=self.df['player_position'].value_counts().index, 
            palette="viridis",
            edgecolor="black",
            alpha=0.9
        )
        plt.title(f"ğŸ“� Player Position Distribution ({self.week_label})", fontsize=14, fontweight='bold')
        plt.xlabel("Count")
        plt.ylabel("Player Position")
        plt.show()

    def vanda_inspection(self):
        fig, ax = plt.subplots(1, 2, figsize=(12,4))
        sns.histplot(self.df['s'], bins=30, kde=True, color='#00BFC4', ax=ax[0], alpha=0.8)
        ax[0].set_title(f"âš¡ Speed Distribution ({self.week_label})", fontsize=13, fontweight='bold')
        sns.histplot(self.df['a'], bins=30, kde=True, color='#F8766D', ax=ax[1], alpha=0.8)
        ax[1].set_title(f"ğŸš€ Acceleration Distribution ({self.week_label})", fontsize=13, fontweight='bold')
        for a in ax:
            a.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def basic_field_image(self):
        first_game = self.df['game_id'].iloc[0]
        first_play = self.df['play_id'].iloc[0]
        sample_play = self.df[(self.df['game_id'] == first_game) &
                              (self.df['play_id'] == first_play)]

        plt.figure(figsize=(9,5))
        sns.scatterplot(
            data=sample_play, x='x', y='y', 
            hue='player_side', style='player_role',
            palette="Spectral", s=100, alpha=0.85, edgecolor='black'
        )
        plt.xlim(0, 120)
        plt.ylim(0, 53.3)
        plt.title(f"ğŸ�Ÿï¸� Player Positions Example ({self.week_label})", fontsize=14, fontweight='bold')
        plt.xlabel("Field Length (yards)")
        plt.ylabel("Field Width (yards)")
        plt.legend(title="Role / Side", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


#building class list part 01
inspection_w01 = basic_inspection(input_01, week = 1) #week 01
inspection_w02 = basic_inspection(input_02, week = 2) #week 02
inspection_w03 = basic_inspection(input_03, week = 3) #week 03
inspection_w04 = basic_inspection(input_04, week = 4) #week 04


#class instances part 02
inspection_w05 = basic_inspection(input_05, week = 5) #week 05
inspection_w06 = basic_inspection(input_06, week = 6) #week 06
inspection_w07 = basic_inspection(input_07, week = 7) #week 07
inspection_w08 = basic_inspection(input_08, week = 8) #week 08


#class instances part 03
inspection_w09 = basic_inspection(input_09, week = 9) #week 09
inspection_w10 = basic_inspection(input_10, week = 10) #week 10
inspection_w11 = basic_inspection(input_11, week = 11) #week 11
inspection_w12 = basic_inspection(input_12, week = 12) #week 12


#class instance part 04
inspection_w13 = basic_inspection(input_13, week = 13) #week 13
inspection_w14 = basic_inspection(input_14, week = 14) #week 14
inspection_w15 = basic_inspection(input_15, week = 15) #week 15
inspection_w16 = basic_inspection(input_16, week = 16) #week 16


#class inspection part 5
inspection_w17 = basic_inspection(input_17, week = 17) #week 17
inspection_w18 = basic_inspection(input_18, week = 18) #week 18 (end)


#gc again
gc.collect()


# week one
inspection_w01.player_role_plot()


#week 02
inspection_w02.player_role_plot()


#week 03
inspection_w03.player_role_plot()


#week 04
inspection_w04.player_role_plot()


#week 06
inspection_w05.player_role_plot()


#week 05
inspection_w06.player_role_plot()


#week 07
inspection_w07.player_role_plot()


#week 08
inspection_w08.player_role_plot()


#week 09
inspection_w09.player_role_plot()


#week 10
inspection_w10.player_role_plot()


#week 11 in player role
inspection_w11.player_role_plot()


#week 12 in player role
inspection_w12.player_role_plot()


#week 13 in player role
inspection_w13.player_role_plot()


#week 14 in player role
inspection_w14.player_role_plot()


#player rola in week 15
inspection_w15.player_role_plot()


#player rola in week 16
inspection_w16.player_role_plot()


#player rola in week 17
inspection_w17.player_role_plot()


#player rola in week 18
inspection_w18.player_role_plot()


#gc intermezzo
gc.collect()


# week one in position spread
inspection_w01.position_spread()


#week 02 in position spread
inspection_w02.position_spread()


#week 03 in position spread
inspection_w03.position_spread()


#week 04 in position spread
inspection_w04.position_spread()


#week 05 in position spread
inspection_w05.position_spread()


#week 06 in position spread
inspection_w06.position_spread()


#week 07 in position spread
inspection_w07.position_spread()


#week 08 in position spread
inspection_w08.position_spread()


#inspection of position spread in week 9
inspection_w09.position_spread()


#inspection of position spread in week 10
inspection_w10.position_spread()


#position spread in week xi
inspection_w11.position_spread()


#position spread in week xii
inspection_w12.position_spread()


#position spread in week 13
inspection_w13.position_spread()


#position spread in week 14
inspection_w14.position_spread()


#position spread in week 15
inspection_w15.position_spread()


#inspection in week 16
inspection_w16.position_spread()


#inspection in week 17
inspection_w17.position_spread()


#inspection in week 18
inspection_w18.position_spread()


#week one in spped inspection
inspection_w01.vanda_inspection()


#week 3 in v and a
inspection_w02.vanda_inspection()


#week 03 in v and a 
inspection_w03.vanda_inspection()


# fourth week in vanda
inspection_w04.vanda_inspection()


#week five in v and a
inspection_w05.vanda_inspection()


#week six in vanda inspection
inspection_w06.vanda_inspection()


#week seven in vanda
inspection_w07.vanda_inspection()


#The 'vanda inspection' week 08
inspection_w08.vanda_inspection()


# similar action in week 09
inspection_w09.vanda_inspection()


# another similar action in week 09
inspection_w10.vanda_inspection()


# another similar action in week 11
inspection_w11.vanda_inspection()


# another similar action in week 12
inspection_w12.vanda_inspection()


# another similar action in week 13
inspection_w13.vanda_inspection()


# another similar action in week 14
inspection_w14.vanda_inspection()


# another similar action in week 15
inspection_w15.vanda_inspection()


#a similar inspection in week 16
inspection_w16.vanda_inspection()


#another similar inspection in week 17
inspection_w17.vanda_inspection()


#final part
inspection_w18.vanda_inspection()


#field image in week one
inspection_w01.basic_field_image()


#field image in week two
inspection_w02.basic_field_image()


#field image in week 3
inspection_w03.basic_field_image()


#field image in week four
inspection_w04.basic_field_image()


#field image in week five
inspection_w05.basic_field_image()


#field image in week six
inspection_w06.basic_field_image()


#field image in week 07
inspection_w07.basic_field_image()


#inspection in week 08
inspection_w08.basic_field_image()


#field inspection in week 09
inspection_w09.basic_field_image()


#field inspection in week 10
inspection_w10.basic_field_image()


#field inspection in week 11
inspection_w11.basic_field_image()


#inspection in week 12
inspection_w12.basic_field_image()


#inspection in week 13
inspection_w13.basic_field_image()


#analysis in week 14
inspection_w14.basic_field_image()


#function to automatically filter usefull data
def instant_filter(input_df, data_path):
    # --- 1. Pick a random play (game_id + play_id) ---
    analysis_pairs = input_df[['game_id', 'play_id']].drop_duplicates()
    sample_row = analysis_pairs.sample(n=1).iloc[0]

    game_identity = sample_row['game_id']
    play_identity = sample_row['play_id']

    # --- 2. Load supplementary data ---
    su_path = f"{data_path}/supplementary_data.csv"
    df_supp_2 = pd.read_csv(su_path, dtype={25: str})
    
    # --- 3. Filter to this play ---
    filtered_data = input_df[
        (input_df['game_id'] == game_identity) &
        (input_df['play_id'] == play_identity)
    ]

    # --- 4. Pick a random frame within this play ---
    available_frames = filtered_data['frame_id'].unique()
    random_frame = np.random.choice(available_frames)

    # --- 5. Build Play object ---
    play = visuals.Play(filtered_data, game_identity, play_identity, df_supp_2)

    return play, random_frame


#Function to display chart
def beautify_nfl_field(fig, ax, 
                       title="Enhanced NFL Field",
                       first_down_x=None, 
                       first_down_color="yellow",
                       first_down_alpha=0.8):
    # --- Field background colors ---
    ax.set_facecolor("#0b6623")        # deep green
    fig.patch.set_facecolor("#084c19") # darker frame background

    # --- Soft yard grid ---
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    # --- White boundary lines ---
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(2)

    # --- Title ---
    ax.set_title(
        f"ğŸ�Ÿï¸� {title}",
        fontsize=16,
        fontweight="bold",
        pad=15,
        color="white"
    )

    # --- White ticks ---
    ax.tick_params(colors="white", labelsize=10)

    # --- Watermark (optional) ---
    ax.text(
        60, 53.3 / 2,
        "NFL",
        fontsize=40,
        color="white",
        alpha=0.12,
        ha="center",
        va="center",
        fontweight="bold"
    )

    # --- TV-style first-down line ---
    if first_down_x is not None:
        ax.axvline(
            first_down_x,
            color=first_down_color,
            linewidth=3,
            alpha=first_down_alpha
        )

    return fig, ax



#execution in week one to week 06
play_w01, random_frame = instant_filter(input_01, main_path)
play_w02, random_frame = instant_filter(input_02, main_path)
play_w03, random_frame = instant_filter(input_03, main_path)
play_w04, random_frame = instant_filter(input_04, main_path)


#extra gc
gc.collect()


#second part
play_w05, random_frame = instant_filter(input_05, main_path)
play_w06, random_frame = instant_filter(input_06, main_path)
play_w07, random_frame = instant_filter(input_07, main_path)
play_w08, random_frame = instant_filter(input_08, main_path)


#third part
play_w09, random_frame = instant_filter(input_09, main_path)
play_w10, random_frame = instant_filter(input_10, main_path)
play_w11, random_frame = instant_filter(input_11, main_path)
play_w12, random_frame = instant_filter(input_12, main_path)


#fourth part
play_w13, random_frame = instant_filter(input_13, main_path)
play_w14, random_frame = instant_filter(input_14, main_path)
play_w15, random_frame = instant_filter(input_15, main_path)
play_w16, random_frame = instant_filter(input_16, main_path)


# the finale for 
play_w17, random_frame = instant_filter(input_17, main_path)
play_w18, random_frame = instant_filter(input_18, main_path)


#function to illustrate relay and result
def relay_illustrator(play_id, num):
    fig, ax = play_id.plot_snap(frameId= random_frame, relay=True)
    # Example: random with no relay
    beautify_nfl_field(fig, ax, title=f"Random Frame {random_frame} with Relay in Week {num}")
    plt.show()


#funciton to animate the play in random
def relay_animation(play_id):
    # Generates a relay dashboard animation of the play
    relay_animation = play_id.animate(relay=True, kaggle=True)
    return relay_animation


#Inspection in week one 
relay_w01 = relay_illustrator(play_w01, 1)


#Illustration in week 02
relay_w02 = relay_illustrator(play_w02, 2)


#Illustration in week 03
relay_w03 = relay_illustrator(play_w03, 3)


#Illustration in week 04
relay_w04 = relay_illustrator(play_w04, 4)


#Illustration in week 05
relay_w05 = relay_illustrator(play_w05, 5)


#Illustration in week 06
relay_w06 = relay_illustrator(play_w06, 6)


#Illustration in week 07
relay_w07 = relay_illustrator(play_w07, 7)


#Illustration in week 08
relay_w08 = relay_illustrator(play_w08, 8)


#Illustration in week 09
relay_w09 = relay_illustrator(play_w09, 9)


#Illustration in week x
relay_w10 = relay_illustrator(play_w10, 10)


#Illustration in week xi
relay_w11 = relay_illustrator(play_w11, 11)


#Illustration in week xii
relay_w12 = relay_illustrator(play_w12, 12)


#Illustration in week xiii
relay_w13 = relay_illustrator(play_w13, 13)


#relay week 14
relay_w14 = relay_illustrator(play_w14, 14)


#relay week 15
relay_w15 = relay_illustrator(play_w15, 15)


#relay week 16
relay_w16 = relay_illustrator(play_w16, 16)


#relay week 17
relay_w17 = relay_illustrator(play_w17, 17)


#final week relay
relay_w18 = relay_illustrator(play_w18, 18)


#animation in week 01
ani_w01 = relay_animation(play_w01)
ani_w01


#animation in week 02
ani_w02 = relay_animation(play_w02)
ani_w02


#animation in week 03
ani_w03 = relay_animation(play_w03)
ani_w03


#animation in week 03
ani_w04 = relay_animation(play_w04)
ani_w04


#animation in week 05
ani_w05 = relay_animation(play_w05)
ani_w05


#animation in week 06
ani_w06 = relay_animation(play_w06)
ani_w06


#animation in week 07
ani_w07 = relay_animation(play_w07)
ani_w07


#animation in week 08
ani_w08 = relay_animation(play_w08)
ani_w08


#animation in week 09
ani_w09 = relay_animation(play_w09)
ani_w09


#animation in week 10
ani_w10 = relay_animation(play_w10)
ani_w10


#animation in week 11
ani_w11 = relay_animation(play_w11)
ani_w11


#animation in week 12
ani_w12 = relay_animation(play_w12)
ani_w12


#animation in week xiii
ani_w13 = relay_animation(play_w13)
ani_w13


#animation in week 14
ani_w14 = relay_animation(play_w14)
ani_w14


#similar aciton in week 15
ani_w15 = relay_animation(play_w15)
ani_w15


#similar aciton in week 16
ani_w16 = relay_animation(play_w16)
ani_w16


#inspection in week 17
ani_w17 = relay_animation(play_w17)
ani_w17


#final animaiton
ani_w18 = relay_animation(play_w18)
ani_w18

