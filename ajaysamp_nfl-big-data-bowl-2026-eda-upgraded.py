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
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML, display
from warnings import filterwarnings
filterwarnings('ignore')

# Snazzy styles
sns.set(style="whitegrid", palette="deep")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

# Base path (adjust if mount changes)
base_path = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/'

# Load supplementary (meta: players, games, plays?)
supplementary = pd.read_csv(base_path + 'supplementary_data.csv')

# Load input tracking (pre-throw: snap to pass_forward)
pre_throw_dfs = []
for week in range(1, 19):
    df = pd.read_csv(base_path + f'train/input_2023_w{week:02}.csv')
    pre_throw_dfs.append(df)
pre_throw = pd.concat(pre_throw_dfs, ignore_index=True)

# Load output tracking (in-air: pass_forward to arrival)
in_air_dfs = []
for week in range(1, 19):
    df = pd.read_csv(base_path + f'train/output_2023_w{week:02}.csv')
    in_air_dfs.append(df)
in_air = pd.concat(in_air_dfs, ignore_index=True)

# Merged full tracking for context (assume same keys like gameId, playId)
merged = pd.concat([pre_throw, in_air], ignore_index=True).sort_values(['game_id', 'play_id', 'frame_id'])

display(HTML(f"<div class='insight-box'>ğŸ“Š Data Shapes: Supplementary {supplementary.shape} | Pre-Throw {pre_throw.shape} | In-Air {in_air.shape} | Merged {merged.shape}</div>"))


# Print columns for each
display(HTML("<div class='insight-box'>ğŸ•µï¸� Supplementary Columns: " + ', '.join(supplementary.columns) + "</div>"))
display(HTML("<div class='insight-box'>ğŸ•µï¸� Pre-Throw Columns: " + ', '.join(pre_throw.columns) + "</div>"))
display(HTML("<div class='insight-box'>ğŸ•µï¸� In-Air Columns: " + ', '.join(in_air.columns) + "</div>"))

# Sample heads
display(supplementary.head())
display(pre_throw.head())
display(in_air.head())


# Number of unique plays
unique_plays = merged[['game_id', 'play_id']].drop_duplicates()
display(HTML(f"<div class='insight-box'>ğŸ”¥ Number of Pass Plays: {len(unique_plays)}</div>"))

# Merge with supp (play-level: no nfl_id)
merged = merged.merge(supplementary, on=['game_id', 'play_id'], how='left')

# Quick stats table (fixed: use go.Table)
stats_table = merged.describe().T.reset_index().rename(columns={'index': 'metric'})
fig = go.Figure(data=[go.Table(
    header=dict(values=['metric', 'mean', 'std', 'min', 'max'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[stats_table['metric'], stats_table['mean'], stats_table['std'], stats_table['min'], stats_table['max']],
               fill_color='lavender',
               align='left'))
])
fig.update_layout(title='Tracking Data Stats Snapshot')
fig.show(renderer='iframe')


# Original throw time analysis (maintained and enhanced)
throw_times = pre_throw.groupby(['game_id', 'play_id'])['num_frames_output'].max() / 10

# Enhanced throw time analysis with strategic context
throw_time_context = pre_throw.groupby(['game_id', 'play_id']).agg({
    'num_frames_output': 'max',
    'ball_land_x': 'first', 
    'ball_land_y': 'first'
}).reset_index()

throw_time_context['throw_time'] = throw_time_context['num_frames_output'] / 10
throw_time_context['pass_distance'] = np.sqrt(
    (throw_time_context['ball_land_x'] - 0)**2 + 
    (throw_time_context['ball_land_y'] - 26.65)**2  # Approximate field center
)

# Enhanced throw time visualization
fig = px.histogram(
    throw_time_context, 
    x='throw_time', 
    nbins=50,
    title='Throw Time Distribution with Strategic Context ğŸ“ˆâš¡',
    labels={'throw_time': 'Throw Time (s)', 'count': 'Number of Plays'},
    color_discrete_sequence=['#228B22'],
    marginal='box'  # Added box plot for better distribution view
)

fig.add_annotation(
    text=f"Avg: {throw_times.mean():.2f}s | Median: {throw_times.median():.2f}s",
    x=0.7, y=0.9, xref="paper", yref="paper",
    showarrow=False, font_size=12
)
fig.show()

# Enhanced speed analysis by position (maintained core, added insights)
positions = ['WR', 'CB', 'FS', 'SS', 'TE', 'LB', 'QB']

# Add physics validation
print("âš–ï¸� PHYSICS VALIDATION ANALYSIS")
print("-" * 30)

max_human_speed = 11.0  # ~25 mph
max_human_accel = 8.0   # realistic limits

speed_violations = (pre_throw['s'] > max_human_speed).sum()
accel_violations = (pre_throw['a'] > max_human_accel).sum()

print(f"ğŸš¨ Speed violations (>{max_human_speed} yds/s): {speed_violations:,}/{len(pre_throw):,} ({100*speed_violations/len(pre_throw):.3f}%)")
print(f"ğŸš¨ Acceleration violations (>{max_human_accel} yds/sÂ²): {accel_violations:,}/{len(pre_throw):,} ({100*accel_violations/len(pre_throw):.3f}%)")

# Enhanced position speed analysis
position_data = pre_throw[pre_throw['player_position'].isin(positions)].copy()
position_stats = position_data.groupby('player_position')['s'].agg(['mean', 'std', 'max']).round(2)

print(f"\nğŸ�ƒ Position Speed Statistics:")
for pos in position_stats.index:
    stats = position_stats.loc[pos]
    print(f"   â€¢ {pos}: Î¼={stats['mean']}, Ïƒ={stats['std']}, max={stats['max']} yds/s")

# Enhanced violin plot (maintained style, added annotations)  
fig = px.violin(
    position_data,
    x='player_position', 
    y='s',
    color='player_position',
    title='Enhanced Speed Distribution by Position (Physics-Validated) ğŸ�»âš¡',
    box=True,
    points='outliers'  # Show outliers for physics validation
)

# Add physics constraint line
fig.add_hline(y=max_human_speed, line_dash="dash", line_color="red", 
              annotation_text="Human Speed Limit (~25 mph)")
fig.show(renderer='iframe')


def nearest_defender_dist(row, df_frame):
    if row['player_position'] not in ['WR', 'TE', 'RB']:
        return np.nan
    off_pos = np.array([row['x'], row['y']])
    def_players = df_frame[(df_frame['player_side'] == 'Defense') & (df_frame['nfl_id'] != row['nfl_id'])]
    if def_players.empty:
        return np.nan
    def_pos = def_players[['x', 'y']].values
    dists = np.sqrt(np.sum((def_pos - off_pos)**2, axis=1))
    return np.min(dists)

# Sample play (from in_air; merge pre for position)
sample_game_id, sample_play_id = in_air[['game_id', 'play_id']].iloc[0]
sample_pre = pre_throw[(pre_throw['game_id'] == sample_game_id) & (pre_throw['play_id'] == sample_play_id)]
sample_play = in_air[(in_air['game_id'] == sample_game_id) & (in_air['play_id'] == sample_play_id)].merge(
    sample_pre[['nfl_id', 'frame_id', 'player_position', 'player_name', 'player_side', 'player_role']], on=['nfl_id', 'frame_id'], how='left')
sample_play['separation'] = sample_play.apply(lambda row: nearest_defender_dist(row, sample_play[sample_play['frame_id'] == row['frame_id']]), axis=1)

fig = px.line(sample_play[sample_play['player_role'] == 'Targeted Receiver'], x='frame_id', y='separation', color='player_name',
              title='Receiver Separation Timeline in Sample Play â�±ï¸�', markers=True)
fig.show(renderer='iframe')


def enhanced_separation_analysis(row, df_frame):
    """Enhanced separation analysis with multiple defensive considerations"""
    if row['player_position'] not in ['WR', 'TE', 'RB']:
        return {'nearest_defender_dist': np.nan, 'defenders_within_3yds': np.nan, 'avg_defender_dist': np.nan}
    
    off_pos = np.array([row['x'], row['y']])
    def_players = df_frame[
        (df_frame['player_side'] == 'Defense') & 
        (df_frame['nfl_id'] != row['nfl_id'])
    ]
    
    if def_players.empty:
        return {'nearest_defender_dist': np.nan, 'defenders_within_3yds': np.nan, 'avg_defender_dist': np.nan}
    
    def_positions = def_players[['x', 'y']].values
    distances = np.sqrt(np.sum((def_positions - off_pos)**2, axis=1))
    
    return {
        'nearest_defender_dist': np.min(distances),
        'defenders_within_3yds': np.sum(distances <= 3),
        'avg_defender_dist': np.mean(distances)
    }

# Enhanced sample play analysis (maintained core functionality)
sample_game_id, sample_play_id = in_air[['game_id', 'play_id']].iloc[0]
sample_pre = pre_throw[
    (pre_throw['game_id'] == sample_game_id) & 
    (pre_throw['play_id'] == sample_play_id)
]

sample_play = in_air[
    (in_air['game_id'] == sample_game_id) & 
    (in_air['play_id'] == sample_play_id)
].merge(
    sample_pre[['nfl_id', 'frame_id', 'player_position', 'player_name', 'player_side', 'player_role']],
    on=['nfl_id', 'frame_id'], 
    how='left'
)

# Apply enhanced separation analysis
print("ğŸ”„ Computing separation metrics...")
separation_metrics = []

for _, row in sample_play.iterrows():
    frame_data = sample_play[sample_play['frame_id'] == row['frame_id']]
    metrics = enhanced_separation_analysis(row, frame_data)
    metrics['frame_id'] = row['frame_id']
    metrics['player_name'] = row['player_name']
    metrics['player_role'] = row['player_role']
    separation_metrics.append(metrics)

separation_df = pd.DataFrame(separation_metrics)
sample_play = sample_play.merge(separation_df, on=['frame_id', 'player_name', 'player_role'], how='left')

# Enhanced separation visualization
receivers_only = sample_play[sample_play['player_role'] == 'Targeted Receiver']

if len(receivers_only) > 0:
    fig = go.Figure()
    
    # Nearest defender distance
    fig.add_trace(go.Scatter(
        x=receivers_only['frame_id'],
        y=receivers_only['nearest_defender_dist'],
        mode='lines+markers',
        name='Nearest Defender',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))
    
    # Average defender distance
    fig.add_trace(go.Scatter(
        x=receivers_only['frame_id'],
        y=receivers_only['avg_defender_dist'],
        mode='lines+markers', 
        name='Average Defender Distance',
        line=dict(color='blue', width=2, dash='dot'),
        marker=dict(size=6)
    ))
    
    # Defenders within 3 yards (secondary y-axis would be ideal)
    fig.add_trace(go.Scatter(
        x=receivers_only['frame_id'],
        y=receivers_only['defenders_within_3yds'],
        mode='lines+markers',
        name='Defenders Within 3 Yards',
        line=dict(color='orange', width=2),
        marker=dict(size=6),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Receiver Separation Analysis: Multi-Metric Timeline â�±ï¸�ğŸ›¡ï¸�',
        xaxis_title='Frame ID (Time)',
        yaxis_title='Distance (yards)',
        yaxis2=dict(title='Count of Nearby Defenders', overlaying='y', side='right'),
        legend=dict(x=0.02, y=0.98),
        height=500
    )
    
    fig.show(renderer='iframe')


def plot_football_field(ax):
    ax.set_facecolor('#228B22')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    for x in range(10, 110, 10):
        ax.axvline(x, color='white', linestyle='--', linewidth=1)
    ez1 = patches.Rectangle((0, 0), 10, 53.3, facecolor='#00008B', alpha=0.6)
    ez2 = patches.Rectangle((110, 0), 10, 53.3, facecolor='#00008B', alpha=0.6)
    ax.add_patch(ez1)
    ax.add_patch(ez2)
    return ax

def animate_play(play_df):
    fig, ax = plt.subplots()
    ax = plot_football_field(ax)
    scat_off = ax.scatter([], [], c='blue', s=50, label='Offense ğŸŸ¦', edgecolor='white')
    scat_def = ax.scatter([], [], c='red', s=50, label='Defense ğŸŸ¥', edgecolor='white')
    scat_ball = ax.scatter([], [], c='brown', s=30, label='Ball ğŸ�ˆ')
    
    def init():
        scat_off.set_offsets(np.empty((0,2)))
        scat_def.set_offsets(np.empty((0,2)))
        scat_ball.set_offsets(np.empty((0,2)))
        return scat_off, scat_def, scat_ball
    
    def update(frame):
        frame_df = play_df[play_df['frame_id'] == frame]
        off = frame_df[frame_df['player_side'] == 'Offense']
        def_ = frame_df[frame_df['player_side'] == 'Defense']
        ball = frame_df[frame_df['nfl_id'].isna()]  # Assume NaN nfl_id for ball; or use ball_land_x/y
        scat_off.set_offsets(off[['x', 'y']])
        scat_def.set_offsets(def_[['x', 'y']])
        scat_ball.set_offsets(ball[['x', 'y']])
        ax.set_title(f"Frame {frame}: Ball in Air Drama! ğŸ�¥", color='black')
        return scat_off, scat_def, scat_ball
    
    frames = sorted(play_df['frame_id'].unique())
    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=100)
    plt.legend(loc='upper right', labelcolor='black')
    plt.close()
    return anim

# Animate sample in-air play
anim = animate_play(sample_play)
HTML(anim.to_jshtml())


def enhanced_plot_football_field(ax):
    """Enhanced field plotting with strategic zones"""
    # Original field setup (maintained)
    ax.set_facecolor('#228B22')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    
    # Yard lines
    for x in range(10, 110, 10):
        ax.axvline(x, color='white', linestyle='--', linewidth=1, alpha=0.7)
    
    # End zones
    ez1 = patches.Rectangle((0, 0), 10, 53.3, facecolor='#00008B', alpha=0.6)
    ez2 = patches.Rectangle((110, 0), 10, 53.3, facecolor='#00008B', alpha=0.6)
    ax.add_patch(ez1)
    ax.add_patch(ez2)
    
    # Enhanced: Strategic zones
    # Red zone (20 yards from goal)
    rz1 = patches.Rectangle((0, 0), 20, 53.3, facecolor='red', alpha=0.1)
    rz2 = patches.Rectangle((100, 0), 20, 53.3, facecolor='red', alpha=0.1)
    ax.add_patch(rz1)
    ax.add_patch(rz2)
    
    # Hash marks for reference
    for x in range(10, 110, 5):
        ax.plot([x, x], [23.36, 24.36], 'white', linewidth=1, alpha=0.5)
        ax.plot([x, x], [29.64, 28.64], 'white', linewidth=1, alpha=0.5)
    
    return ax

def enhanced_animate_play(play_df):
    """Enhanced animation with separation tracking"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Main field view
    ax1 = enhanced_plot_football_field(ax1)
    
    # Initialize scatter plots
    scat_off = ax1.scatter([], [], c='blue', s=80, label='Offense ğŸŸ¦', edgecolor='white', linewidth=2)
    scat_def = ax1.scatter([], [], c='red', s=80, label='Defense ğŸŸ¥', edgecolor='white', linewidth=2)
    scat_target = ax1.scatter([], [], c='gold', s=120, label='Target ğŸ�¯', edgecolor='black', linewidth=2)
    
    # Separation tracking plot
    ax2.set_xlim(1, play_df['frame_id'].max())
    ax2.set_ylim(0, 15)
    ax2.set_xlabel('Frame ID')
    ax2.set_ylabel('Separation Distance (yards)')
    ax2.set_title('Real-time Separation Tracking ğŸ“Š')
    ax2.grid(True, alpha=0.3)
    
    separation_line, = ax2.plot([], [], 'gold', linewidth=3, label='Target Separation')
    ax2.legend()
    
    # Data storage for separation tracking
    frame_history = []
    separation_history = []
    
    def init():
        scat_off.set_offsets(np.empty((0, 2)))
        scat_def.set_offsets(np.empty((0, 2)))
        scat_target.set_offsets(np.empty((0, 2)))
        separation_line.set_data([], [])
        return scat_off, scat_def, scat_target, separation_line
    
    def update(frame):
        frame_df = play_df[play_df['frame_id'] == frame]
        
        if frame_df.empty:
            return scat_off, scat_def, scat_target, separation_line
        
        # Separate by team
        offense = frame_df[frame_df['player_side'] == 'Offense']
        defense = frame_df[frame_df['player_side'] == 'Defense']
        target = frame_df[frame_df['player_role'] == 'Targeted Receiver']
        
        # Update positions
        scat_off.set_offsets(offense[['x', 'y']].values if len(offense) > 0 else np.empty((0, 2)))
        scat_def.set_offsets(defense[['x', 'y']].values if len(defense) > 0 else np.empty((0, 2)))
        scat_target.set_offsets(target[['x', 'y']].values if len(target) > 0 else np.empty((0, 2)))
        
        # Update separation tracking
        if len(target) > 0 and 'nearest_defender_dist' in target.columns:
            current_separation = target['nearest_defender_dist'].iloc[0]
            if not np.isnan(current_separation):
                frame_history.append(frame)
                separation_history.append(current_separation)
                separation_line.set_data(frame_history, separation_history)
        
        # Dynamic title with insights
        avg_sep = np.mean(separation_history) if separation_history else 0
        ax1.set_title(f"Frame {frame}: Ball in Air Drama! ğŸ�¥ | Avg Separation: {avg_sep:.1f} yds", 
                     color='black', fontsize=12, fontweight='bold')
        
        return scat_off, scat_def, scat_target, separation_line
    
    frames = sorted(play_df['frame_id'].unique())
    anim = FuncAnimation(fig, update, frames=frames, init_func=init, 
                        blit=False, interval=200, repeat=True)
    
    ax1.legend(loc='upper right', labelcolor='black')
    plt.tight_layout()
    plt.close()
    
    return anim

# Create enhanced animation
if len(sample_play) > 0:
    print("ğŸ�¬ Creating enhanced animation with real-time separation tracking...")
    enhanced_anim = enhanced_animate_play(sample_play)
    HTML(enhanced_anim.to_jshtml())


# Ensure sample_play has 's' by merging again with pre_throw
sample_play_3d = in_air[(in_air['game_id'] == sample_game_id) & (in_air['play_id'] == sample_play_id)].merge(
    pre_throw[['nfl_id', 'frame_id', 'player_position', 'player_name', 'player_side', 'player_role', 's']], on=['nfl_id', 'frame_id'], how='left')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(sample_play_3d['x'], sample_play_3d['y'], sample_play_3d['frame_id'], c=sample_play_3d['s'], cmap='viridis')
ax.set_title('3D Play Scatter: X/Y/Frame with Speed Color ğŸŒŒ')
ax.set_xlabel('X (Yards)')
ax.set_ylabel('Y (Width)')
ax.set_zlabel('Frame ID (Time)')
plt.colorbar(scatter, label='Speed (s)')
plt.show()


# Enhanced 3D data preparation
sample_play_3d = in_air[
    (in_air['game_id'] == sample_game_id) & 
    (in_air['play_id'] == sample_play_id)
].merge(
    pre_throw[['nfl_id', 'frame_id', 'player_position', 'player_name', 'player_side', 'player_role', 's', 'a', 'dir']],
    on=['nfl_id', 'frame_id'], 
    how='left'
)

if len(sample_play_3d) > 0:
    # Enhanced 3D scatter with multiple dimensions
    fig = go.Figure()
    
    # Different traces for different roles
    roles = sample_play_3d['player_role'].dropna().unique()
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, role in enumerate(roles):
        role_data = sample_play_3d[sample_play_3d['player_role'] == role]
        
        fig.add_trace(go.Scatter3d(
            x=role_data['x'],
            y=role_data['y'], 
            z=role_data['frame_id'],
            mode='markers+lines',
            marker=dict(
                size=role_data['s'] * 2,  # Size based on speed
                color=colors[i % len(colors)],
                opacity=0.8,
                colorscale='Viridis',
                showscale=True if i == 0 else False
            ),
            line=dict(color=colors[i % len(colors)], width=3),
            name=f"{role} (Speed-sized)",
            text=[f"Player: {name}<br>Speed: {speed:.1f}<br>Frame: {frame}" 
                  for name, speed, frame in zip(role_data['player_name'], role_data['s'], role_data['frame_id'])],
            hovertemplate='<b>%{text}</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Time: %{z}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Enhanced 3D Play Analysis: Multi-Role Movement Patterns ğŸŒŒâš¡',
        scene=dict(
            xaxis_title='Field Length (yards)',
            yaxis_title='Field Width (yards)', 
            zaxis_title='Frame ID (Time)',
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
        ),
        height=600,
        showlegend=True
    )
    
    fig.show(renderer='iframe')


# Enhanced player density analysis by role
roles_of_interest = ['Targeted Receiver', 'Defensive Coverage', 'Passer']

fig = go.Figure()

# Create subplots for different roles
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=['All Players', 'Targeted Receivers', 'Defensive Coverage', 'Strategic Overlay'],
    specs=[[{"type": "histogram2d"}, {"type": "histogram2d"}],
           [{"type": "histogram2d"}, {"type": "scatter"}]]
)

# All players heatmap (enhanced from original)
fig.add_trace(
    go.Histogram2d(
        x=in_air['x'],
        y=in_air['y'],
        nbinsx=60, nbinsy=27,
        colorscale='Greens',
        showscale=True,
        name='All Players'
    ),
    row=1, col=1
)

# Role-specific heatmaps
if 'player_role' in merged.columns:
    # Targeted receivers
    receivers = merged[merged['player_role'] == 'Targeted Receiver']
    if len(receivers) > 0:
        fig.add_trace(
            go.Histogram2d(
                x=receivers['x'],
                y=receivers['y'],
                nbinsx=60, nbinsy=27,
                colorscale='Reds',
                showscale=False,
                name='Receivers'
            ),
            row=1, col=2
        )
    
    # Defensive coverage
    defenders = merged[merged['player_role'] == 'Defensive Coverage']
    if len(defenders) > 0:
        fig.add_trace(
            go.Histogram2d(
                x=defenders['x'],
                y=defenders['y'],
                nbinsx=60, nbinsy=27,
                colorscale='Blues',
                showscale=False,
                name='Defenders'
            ),
            row=2, col=1
        )

# Strategic overlay with field zones
strategic_zones = {
    'Red Zone': {'x': [0, 20, 20, 0, 0], 'y': [0, 0, 53.3, 53.3, 0]},
    'Deep Zone': {'x': [60, 80, 80, 60, 60], 'y': [0, 0, 53.3, 53.3, 0]},
    'Hash Marks': {'x': [0, 120], 'y': [23.36, 23.36]}
}

# Add strategic zones
fig.add_trace(
    go.Scatter(
        x=strategic_zones['Red Zone']['x'],
        y=strategic_zones['Red Zone']['y'],
        mode='lines',
        line=dict(color='red', width=3),
        name='Red Zone',
        fill='toself',
        fillcolor='rgba(255,0,0,0.1)'
    ),
    row=2, col=2
)

fig.add_trace(
    go.Scatter(
        x=strategic_zones['Deep Zone']['x'],
        y=strategic_zones['Deep Zone']['y'],
        mode='lines',
        line=dict(color='blue', width=3),
        name='Deep Zone',
        fill='toself',
        fillcolor='rgba(0,0,255,0.1)'
    ),
    row=2, col=2
)

fig.update_layout(
    title_text="Enhanced Strategic Heatmap Analysis ğŸŒ¡ï¸�ğŸ�ˆ",
    height=800,
    showlegend=True
)

# Update axis labels
for i in range(1, 3):
    for j in range(1, 3):
        fig.update_xaxes(title_text="Field Length (yards)", row=i, col=j)
        fig.update_yaxes(title_text="Field Width (yards)", row=i, col=j)

fig.show()


# Multi-play traj (sample 5 plays)
sample_plays = merged[merged['player_role'] == 'Targeted Receiver']['play_id'].unique()[:5]
traj_df = merged[merged['play_id'].isin(sample_plays) & (merged['player_role'] == 'Targeted Receiver')]
fig = px.line(traj_df, x='x', y='y', color='play_id', title='Trajectory Comparer: Targeted Routes Across Plays ğŸ“�',
              labels={'play_id': 'Play ID'}, markers=True)
fig.show(renderer='iframe')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# For advanced visualizations
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr

class NFLBowl2026EDA:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.train_input = {}
        self.train_output = {}
        self.test_input = None
        self.test_df = None
        self.sample_submission = None
        
    def load_data(self, sample_weeks=2):
        """Load training and test data"""
        print("Loading NFL Big Data Bowl 2026 data...")
        
        for week in range(1, 19):
            file_path = self.data_path / 'train' / f'input_2023_w{week:02d}.csv'
            if file_path.exists():
                self.train_input[f'week_{week}'] = pd.read_csv(file_path)
                print(f"Loaded input_2023_w{week:02d}.csv")
            if sample_weeks and week >= sample_weeks:
                break
        
        for week in range(1, 19):
            file_path = self.data_path / 'train' / f'output_2023_w{week:02d}.csv'
            if file_path.exists():
                self.train_output[f'week_{week}'] = pd.read_csv(file_path)
                print(f"Loaded output_2023_w{week:02d}.csv")
            if sample_weeks and week >= sample_weeks:
                break
        
        test_input_path = self.data_path / 'test_input.csv'
        if test_input_path.exists():
            self.test_input = pd.read_csv(test_input_path)
            print("Loaded test_input.csv")
            
        test_df_path = self.data_path / 'test.csv'
        if test_df_path.exists():
            self.test_df = pd.read_csv(test_df_path)
            print("Loaded test.csv")
            
        sample_sub_path = self.data_path / 'sample_submission.csv'
        if sample_sub_path.exists():
            self.sample_submission = pd.read_csv(sample_sub_path)
            print("Loaded sample_submission.csv")
            
        print("Data loading completed!")
        self.print_data_shapes()
    
    def print_data_shapes(self):
        """Print shapes of all loaded dataframes"""
        print("\n" + "="*50)
        print("DATA SHAPES")
        print("="*50)
        
        print(f"Training Input Weeks: {len(self.train_input)}")
        for week, data in self.train_input.items():
            print(f"  {week}: {data.shape}")
            
        print(f"Training Output Weeks: {len(self.train_output)}")
        for week, data in self.train_output.items():
            print(f"  {week}: {data.shape}")
            
        if self.test_input is not None:
            print(f"Test Input: {self.test_input.shape}")
        if self.test_df is not None:
            print(f"Test DataFrame: {self.test_df.shape}")
        if self.sample_submission is not None:
            print(f"Sample Submission: {self.sample_submission.shape}")
    
    def data_quality_checks(self):
        """Perform data quality checks: missing values, duplicates, etc."""
        if not self.train_input:
            return
        
        print("\n" + "="*50)
        print("DATA QUALITY CHECKS")
        print("="*50)
        
        first_week_input = list(self.train_input.values())[0]
        first_week_output = list(self.train_output.values())[0]
        
        print("Missing values in input data:")
        print(first_week_input.isnull().sum())
        
        print("\nMissing values in output data:")
        print(first_week_output.isnull().sum())
        
        print("\nDuplicate rows in input data:", first_week_input.duplicated().sum())
        print("Duplicate rows in output data:", first_week_output.duplicated().sum())
        
        print("\nInput data types:")
        print(first_week_input.dtypes)
        
        print("\nOutput data types:")
        print(first_week_output.dtypes)
    
    def analyze_train_input_structure(self):
        """Analyze the structure of training input data"""
        if not self.train_input:
            return
            
        print("\n" + "="*50)
        print("TRAINING INPUT DATA ANALYSIS")
        print("="*50)
        
        first_week = list(self.train_input.values())[0]
        
        print("Columns in input data:")
        for col in first_week.columns:
            print(f"  {col}")
        
        print(f"\nFirst week shape: {first_week.shape}")
        print(f"Unique games: {first_week['game_id'].nunique()}")
        print(f"Unique plays: {first_week['play_id'].nunique()}")
        print(f"Unique players: {first_week['nfl_id'].nunique()}")
        
        numeric_cols = ['x', 'y', 's', 'a', 'o', 'dir', 'absolute_yardline_number', 'num_frames_output', 'ball_land_x', 'ball_land_y']
        available_numeric = [col for col in numeric_cols if col in first_week.columns]
        
        print("\nBasic statistics:")
        print(first_week[available_numeric].describe())
        
        if 'player_to_predict' in first_week.columns:
            print(f"\nPlayer to predict distribution:")
            print(first_week['player_to_predict'].value_counts())
            print(f"Percentage to predict: {first_week['player_to_predict'].mean():.2%}")
    
    def analyze_train_output_structure(self):
        """Analyze the structure of training output data (targets)"""
        if not self.train_output:
            return
            
        print("\n" + "="*50)
        print("TRAINING OUTPUT DATA ANALYSIS (TARGETS)")
        print("="*50)
        
        first_output = list(self.train_output.values())[0]
        
        print(f"Output data shape: {first_output.shape}")
        print("Columns in output data:")
        for col in first_output.columns:
            print(f"  {col}")
        
        print("\nTARGET VARIABLES (x, y):")
        print(f"x - min: {first_output['x'].min():.1f}, max: {first_output['x'].max():.1f}, mean: {first_output['x'].mean():.1f}")
        print(f"y - min: {first_output['y'].min():.1f}, max: {first_output['y'].max():.1f}, mean: {first_output['y'].mean():.1f}")
        
        missing_x = first_output['x'].isna().sum()
        missing_y = first_output['y'].isna().sum()
        print(f"Missing x: {missing_x}, Missing y: {missing_y}")
        
        fig = px.histogram(first_output, x='x', nbins=50, title='Distribution of Target X Coordinates', color_discrete_sequence=['blue'])
        fig.show()
        
        fig = px.histogram(first_output, x='y', nbins=50, title='Distribution of Target Y Coordinates', color_discrete_sequence=['red'])
        fig.show()
        
        fig = px.scatter(first_output, x='x', y='y', opacity=0.3, title='Target Locations (x, y)')
        fig.show()
        
        fig = self._draw_football_field_plotly()
        fig.add_trace(go.Scatter(x=first_output['x'], y=first_output['y'], mode='markers', opacity=0.1, marker=dict(size=1)))
        fig.update_layout(title='Target Locations on Football Field')
        fig.show()
    
    def analyze_correlations(self):
        """Analyze correlations between features and targets"""
        if not self.train_input or not self.train_output:
            return
            
        print("\n" + "="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        
        input_df = list(self.train_input.values())[0]
        output_df = list(self.train_output.values())[0]
        
        merged = input_df.merge(output_df.groupby(['game_id', 'play_id', 'nfl_id'])[['x', 'y']].mean().reset_index(),
                                on=['game_id', 'play_id', 'nfl_id'], suffixes=('_input', '_target'))
        
        corr_features = ['s', 'a', 'o', 'dir', 'x_input', 'y_input', 'x_target', 'y_target']
        corr_matrix = merged[corr_features].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Key Features and Targets')
        plt.show()
        
        for col in ['s', 'a', 'o', 'dir']:
            corr_x, p_x = pearsonr(merged[col], merged['x_target'])
            corr_y, p_y = pearsonr(merged[col], merged['y_target'])
            print(f"{col} vs x_target: corr={corr_x:.2f}, p={p_x:.4f}")
            print(f"{col} vs y_target: corr={corr_y:.2f}, p={p_y:.4f}")
    
    def analyze_player_characteristics(self):
        """Analyze player demographics and characteristics"""
        if not self.train_input:
            return
            
        print("\n" + "="*50)
        print("PLAYER CHARACTERISTICS ANALYSIS")
        print("="*50)
        
        first_week = list(self.train_input.values())[0]
        
        if 'player_position' in first_week.columns:
            print("Player position distribution:")
            pos_counts = first_week['player_position'].value_counts()
            print(pos_counts)
            
            fig = px.bar(pos_counts, title='Player Position Distribution')
            fig.show()
        
        if 'player_side' in first_week.columns:
            print(f"\nPlayer side distribution:")
            print(first_week['player_side'].value_counts())
        
        if 'player_role' in first_week.columns:
            print(f"\nPlayer role distribution:")
            role_counts = first_week['player_role'].value_counts()
            print(role_counts)
            
            fig = px.bar(role_counts, title='Player Role Distribution')
            fig.show()
        
        if 'player_height' in first_week.columns and 'player_weight' in first_week.columns:
            first_week['height_inches'] = first_week['player_height'].apply(self._convert_height_to_inches)
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=first_week['height_inches'], name='Height (inches)'))
            fig.update_layout(title='Player Height Distribution')
            fig.show()
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=first_week['player_weight'], name='Weight (lbs)'))
            fig.update_layout(title='Player Weight Distribution')
            fig.show()
            
            top_positions = first_week['player_position'].value_counts().head(5).index
            fig = px.scatter(first_week, x='height_inches', y='player_weight', color='player_position',
                             title='Height vs Weight by Position', opacity=0.6)
            fig.show()
    
    def analyze_tracking_metrics(self):
        """Analyze player tracking metrics"""
        if not self.train_input:
            return
            
        print("\n" + "="*50)
        print("PLAYER TRACKING METRICS ANALYSIS")
        print("="*50)
        
        first_week = list(self.train_input.values())[0]
        
        tracking_metrics = ['x', 'y', 's', 'a', 'o', 'dir']
        available_metrics = [m for m in tracking_metrics if m in first_week.columns]
        
        print("Tracking metrics statistics:")
        print(first_week[available_metrics].describe())
        
        for metric in available_metrics:
            fig = px.histogram(first_week, x=metric, title=f'Distribution of {metric.upper()}')
            fig.show()
        
        fig = px.box(first_week, y='s', title='Boxplot of Speed (s)')
        fig.show()
        
        if 'player_position' in first_week.columns:
            top_positions = first_week['player_position'].value_counts().head(8).index
            speed_data = [first_week[first_week['player_position'] == pos]['s'].dropna() for pos in top_positions]
            
            fig = go.Figure()
            for pos, speeds in zip(top_positions, speed_data):
                fig.add_trace(go.Box(y=speeds, name=pos))
            fig.update_layout(title='Speed Distribution by Position')
            fig.show()
    
    def analyze_play_characteristics(self):
        """Analyze play-level characteristics"""
        if not self.train_input:
            return
            
        print("\n" + "="*50)
        print("PLAY CHARACTERISTICS ANALYSIS")
        print("="*50)
        
        first_week = list(self.train_input.values())[0]
        
        if 'play_direction' in first_week.columns:
            print("Play direction distribution:")
            print(first_week['play_direction'].value_counts())
        
        if 'absolute_yardline_number' in first_week.columns:
            fig = px.histogram(first_week, x='absolute_yardline_number', nbins=50,
                               title='Distribution of Field Position (Absolute Yardline)')
            fig.show()
        
        if 'ball_land_x' in first_week.columns and 'ball_land_y' in first_week.columns:
            fig = px.histogram(first_week, x='ball_land_x', nbins=50, title='Ball Landing X Position')
            fig.show()
            
            fig = px.histogram(first_week, x='ball_land_y', nbins=50, title='Ball Landing Y Position')
            fig.show()
            
            field_fig = self._draw_football_field_plotly()
            field_fig.add_trace(go.Scatter(x=first_week['ball_land_x'], y=first_week['ball_land_y'],
                                           mode='markers', opacity=0.5, marker=dict(size=1)))
            field_fig.update_layout(title='Ball Landing Positions on Field')
            field_fig.show()
    
    def analyze_frame_predictions(self):
        """Analyze frame prediction requirements"""
        if not self.train_input:
            return
            
        print("\n" + "="*50)
        print("FRAME PREDICTION ANALYSIS")
        print("="*50)
        
        first_week = list(self.train_input.values())[0]
        
        if 'num_frames_output' in first_week.columns:
            print("Number of frames to predict:")
            print(first_week['num_frames_output'].describe())
            
            fig = px.histogram(first_week, x='num_frames_output', nbins=30,
                               title='Distribution of Frames to Predict')
            fig.show()
            
            if 'player_role' in first_week.columns:
                role_frame_means = first_week.groupby('player_role')['num_frames_output'].mean().sort_values(ascending=False)
                fig = px.bar(role_frame_means, title='Average Frames to Predict by Player Role')
                fig.show()
    
    def analyze_test_structure(self):
        """Analyze test data structure"""
        if self.test_input is None or self.test_df is None:
            return
            
        print("\n" + "="*50)
        print("TEST DATA ANALYSIS")
        print("="*50)
        
        print(f"Test input shape: {self.test_input.shape}")
        print(f"Test prediction targets shape: {self.test_df.shape}")
        
        print("\nTest input columns:")
        for col in self.test_input.columns:
            print(f"  {col}")
            
        print("\nTest prediction targets columns:")
        for col in self.test_df.columns:
            print(f"  {col}")
        
        train_cols = set(list(self.train_input.values())[0].columns)
        test_cols = set(self.test_input.columns)
        
        print(f"\nColumns in train but not in test: {train_cols - test_cols}")
        print(f"Columns in test but not in train: {test_cols - train_cols}")
        
        if self.sample_submission is not None:
            print(f"\nSample submission shape: {self.sample_submission.shape}")
            print("Sample submission head:")
            print(self.sample_submission.head())
    
    def visualize_sample_plays(self, n_plays=2):
        """Visualize sample plays to understand player movements"""
        if not self.train_input:
            return
            
        first_week = list(self.train_input.values())[0]
        
        unique_plays = first_week[['game_id', 'play_id']].drop_duplicates().sample(n_plays, random_state=42)
        
        for _, (game_id, play_id) in unique_plays.iterrows():
            self._plot_single_play(first_week, game_id, play_id)
    
    def _plot_single_play(self, data, game_id, play_id):
        """Plot a single play with player positions using Plotly"""
        play_data = data[
            (data['game_id'] == game_id) & 
            (data['play_id'] == play_id)
        ].copy()
        
        if play_data.empty:
            return
        
        print(f"\nVisualizing Play: Game {game_id}, Play {play_id}")
        
        play_info = play_data.iloc[0]
        if 'play_direction' in play_info:
            print(f"Play direction: {play_info['play_direction']}")
        if 'absolute_yardline_number' in play_info:
            print(f"Field position: {play_info['absolute_yardline_number']}")
        
        fig = self._draw_football_field_plotly()
        
        if 'player_side' in play_data.columns:
            colors = {'Offense': 'blue', 'Defense': 'red'}
            for side, color in colors.items():
                side_data = play_data[play_data['player_side'] == side]
                fig.add_trace(go.Scatter(x=side_data['x'], y=side_data['y'], mode='markers',
                                         marker=dict(color=color, size=10), name=side, opacity=0.7))
                
                for _, player in side_data.iterrows():
                    fig.add_annotation(x=player['x'], y=player['y'], text=player.get('player_position', '?'),
                                       showarrow=False, font=dict(size=8))
        
        if 'ball_land_x' in play_info and 'ball_land_y' in play_info:
            if not pd.isna(play_info['ball_land_x']) and not pd.isna(play_info['ball_land_y']):
                fig.add_trace(go.Scatter(x=[play_info['ball_land_x']], y=[play_info['ball_land_y']],
                                         mode='markers', marker=dict(color='gold', size=15, symbol='star'),
                                         name='Ball Landing'))
        
        fig.update_layout(title=f'Play {play_id} - Player Positions')
        fig.show()
    
    def _draw_football_field(self, ax):
        """Draw a football field background (matplotlib)"""
        field_length = 120
        field_width = 53.3
        
        rect = plt.Rectangle((0, 0), field_length, field_width, 
                             linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.3)
        ax.add_patch(rect)
        
        for x in range(10, field_length, 10):
            ax.axvline(x=x, color='white', alpha=0.5, linestyle='-', linewidth=1)
        
        ax.axvline(x=60, color='white', alpha=1, linestyle='-', linewidth=2)
        
        ax.set_xlim(0, field_length)
        ax.set_ylim(0, field_width)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _draw_football_field_plotly(self):
        """Draw a football field background (plotly)"""
        field_length = 120
        field_width = 53.3
        
        fig = go.Figure()
        
        fig.add_shape(type="rect", x0=0, y0=0, x1=field_length, y1=field_width,
                      line=dict(color="green", width=2), fillcolor="lightgreen", opacity=0.3)
        
        for x in range(10, field_length, 10):
            fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=field_width,
                          line=dict(color="white", width=1, dash="solid"))
        
        fig.add_shape(type="line", x0=60, y0=0, x1=60, y1=field_width,
                      line=dict(color="white", width=2))
        
        fig.update_layout(xaxis_range=[0, field_length], yaxis_range=[0, field_width],
                          width=800, height=400, showlegend=True)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        return fig
    
    def _convert_height_to_inches(self, height_str):
        """Convert height string (ft-in) to inches"""
        if pd.isna(height_str):
            return np.nan
        try:
            feet, inches = height_str.split('-')
            return int(feet) * 12 + int(inches)
        except:
            return np.nan
    
    def analyze_trajectories(self):
        """Analyze player trajectories over frames"""
        if not self.train_input or not self.train_output:
            return
            
        print("\n" + "="*50)
        print("PLAYER TRAJECTORY ANALYSIS")
        print("="*50)
        
        input_df = list(self.train_input.values())[0]
        output_df = list(self.train_output.values())[0]
        
        sample_play = input_df[['game_id', 'play_id']].drop_duplicates().sample(1)
        game_id, play_id = sample_play.iloc[0]
        
        input_play = input_df[(input_df['game_id'] == game_id) & (input_df['play_id'] == play_id)]
        output_play = output_df[(output_df['game_id'] == game_id) & (output_df['play_id'] == play_id)]
        
        fig = self._draw_football_field_plotly()
        
        for nfl_id in input_play['nfl_id'].unique():
            input_traj = input_play[input_play['nfl_id'] == nfl_id].sort_values('frame_id')
            output_traj = output_play[output_play['nfl_id'] == nfl_id].sort_values('frame_id')
            
            fig.add_trace(go.Scatter(x=input_traj['x'], y=input_traj['y'], mode='lines+markers',
                                     name=f'Player {nfl_id} Input', line=dict(dash='solid')))
            fig.add_trace(go.Scatter(x=output_traj['x'], y=output_traj['y'], mode='lines+markers',
                                     name=f'Player {nfl_id} Output', line=dict(dash='dash')))
        
        fig.update_layout(title=f'Trajectories for Game {game_id}, Play {play_id}')
        fig.show()
    
    def compare_offense_defense(self):
        """Compare metrics between offense and defense"""
        if not self.train_input:
            return
            
        print("\n" + "="*50)
        print("OFFENSE vs DEFENSE COMPARISON")
        print("="*50)
        
        first_week = list(self.train_input.values())[0]
        
        if 'player_side' in first_week.columns:
            offense = first_week[first_week['player_side'] == 'Offense']
            defense = first_week[first_week['player_side'] == 'Defense']
            
            metrics = ['s', 'a']
            for metric in metrics:
                fig = go.Figure()
                fig.add_trace(go.Box(y=offense[metric], name='Offense'))
                fig.add_trace(go.Box(y=defense[metric], name='Defense'))
                fig.update_layout(title=f'{metric.upper()} Comparison: Offense vs Defense')
                fig.show()
    
    def analyze_distance_to_ball_landing(self):
        """Analyze player distance to ball landing position over frames"""
        if not self.train_input or not self.train_output:
            return
            
        print("\n" + "="*50)
        print("DISTANCE TO BALL LANDING ANALYSIS")
        print("="*50)
        
        input_df = list(self.train_input.values())[0]
        output_df = list(self.train_output.values())[0]
        
        # Sample a play
        sample_play = input_df[['game_id', 'play_id']].drop_duplicates().sample(1)
        game_id, play_id = sample_play.iloc[0]
        
        input_play = input_df[(input_df['game_id'] == game_id) & (input_df['play_id'] == play_id)]
        output_play = output_df[(output_df['game_id'] == game_id) & (output_df['play_id'] == play_id)]
        
        # Calculate distance to ball landing for each frame
        ball_land_x = input_play['ball_land_x'].iloc[0]
        ball_land_y = input_play['ball_land_y'].iloc[0]
        
        input_play['distance_to_ball'] = np.sqrt((input_play['x'] - ball_land_x)**2 + (input_play['y'] - ball_land_y)**2)
        output_play['distance_to_ball'] = np.sqrt((output_play['x'] - ball_land_x)**2 + (output_play['y'] - ball_land_y)**2)
        
        fig = go.Figure()
        for nfl_id in input_play['nfl_id'].unique():
            input_traj = input_play[input_play['nfl_id'] == nfl_id].sort_values('frame_id')
            output_traj = output_play[output_play['nfl_id'] == nfl_id].sort_values('frame_id')
            
            fig.add_trace(go.Scatter(x=input_traj['frame_id'], y=input_traj['distance_to_ball'],
                                     mode='lines+markers', name=f'Player {nfl_id} Input', line=dict(dash='solid')))
            fig.add_trace(go.Scatter(x=output_traj['frame_id'], y=output_traj['distance_to_ball'],
                                     mode='lines+markers', name=f'Player {nfl_id} Output', line=dict(dash='dash')))
        
        fig.update_layout(title=f'Distance to Ball Landing vs Frame for Game {game_id}, Play {play_id}',
                          xaxis_title='Frame ID', yaxis_title='Distance to Ball Landing (yards)')
        fig.show()
    
    def analyze_movement_patterns(self):
        """Analyze player movement patterns by role"""
        if not self.train_input or not self.train_output:
            return
            
        print("\n" + "="*50)
        print("MOVEMENT PATTERNS BY PLAYER ROLE")
        print("="*50)
        
        input_df = list(self.train_input.values())[0]
        output_df = list(self.train_output.values())[0]
        
        # Compute displacement per frame
        merged = input_df.merge(output_df, on=['game_id', 'play_id', 'nfl_id', 'frame_id'], how='outer')
        merged['displacement'] = np.sqrt((merged['x_y'] - merged['x_x'])**2 + (merged['y_y'] - merged['y_x'])**2)
        
        # Aggregate by player role
        if 'player_role' in merged.columns:
            role_displacement = merged.groupby('player_role')['displacement'].mean().sort_values(ascending=False)
            
            fig = px.bar(role_displacement, title='Average Displacement per Frame by Player Role',
                         labels={'value': 'Average Displacement (yards)', 'player_role': 'Player Role'})
            fig.show()
    
    def analyze_speed_acceleration_relationship(self):
        """Analyze relationship between speed and acceleration by player role"""
        if not self.train_input:
            return
            
        print("\n" + "="*50)
        print("SPEED vs ACCELERATION ANALYSIS")
        print("="*50)
        
        first_week = list(self.train_input.values())[0]
        
        if 'player_role' in first_week.columns:
            fig = px.scatter(first_week, x='s', y='a', color='player_role', opacity=0.5,
                             title='Speed vs Acceleration by Player Role',
                             labels={'s': 'Speed (yards/second)', 'a': 'Acceleration (yards/second^2)'})
            fig.show()
    
    def visualize_play_heatmap(self):
        """Visualize heatmap of player positions across all plays"""
        if not self.train_input:
            return
            
        print("\n" + "="*50)
        print("PLAYER POSITION HEATMAP")
        print("="*50)
        
        first_week = list(self.train_input.values())[0]
        
        fig = self._draw_football_field_plotly()
        fig.add_trace(go.Histogram2dContour(x=first_week['x'], y=first_week['y'],
                                            colorscale='Viridis', showscale=True))
        fig.update_layout(title='Heatmap of Player Positions Across All Plays')
        fig.show()
    
    def run_complete_analysis(self, sample_weeks=2):
        """Run complete EDA analysis"""
        print("Starting NFL Big Data Bowl 2026 EDA")
        print("="*60)
        
        self.load_data(sample_weeks=sample_weeks)
        self.data_quality_checks()
        self.analyze_train_input_structure()
        self.analyze_train_output_structure()
        self.analyze_correlations()
        self.analyze_player_characteristics()
        self.analyze_tracking_metrics()
        self.analyze_play_characteristics()
        self.analyze_frame_predictions()
        self.analyze_test_structure()
        self.visualize_sample_plays(n_plays=2)
        self.analyze_trajectories()
        self.compare_offense_defense()
        self.analyze_distance_to_ball_landing()
        self.analyze_movement_patterns()
        self.analyze_speed_acceleration_relationship()
        self.visualize_play_heatmap()
        
        print("\n" + "="*60)
        print("EDA COMPLETED!")
        print("="*60)

def create_data_summary(data_path):
    """Create a comprehensive data summary"""
    summary = {
        "competition": "NFL Big Data Bowl 2026",
        "task": "Predict player (x, y) coordinates after pass is thrown",
        "key_insights": {
            "input_data": "Tracking data before pass is thrown",
            "output_data": "Tracking data after pass is thrown (targets)",
            "prediction_scope": "Only players with player_to_predict=True",
            "time_series": "Multiple frames to predict for each player",
            "evaluation": "Mean Euclidean distance error"
        }
    }
    
    return summary

if __name__ == "__main__":
    data_path = "/kaggle/input/nfl-big-data-bowl-2026-prediction"
    
    eda = NFLBowl2026EDA(data_path)
    eda.run_complete_analysis(sample_weeks=5)
    
    summary = create_data_summary(data_path)
    print("\nCompetition Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")




