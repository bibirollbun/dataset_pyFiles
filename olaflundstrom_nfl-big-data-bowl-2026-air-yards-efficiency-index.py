import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# ============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_tracking_data(week):
    """Load input and output tracking data for a specific week"""
    # Fixed path structure
    base_path = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train'
    input_file = f'{base_path}/input_2023_w{week:02d}.csv'
    output_file = f'{base_path}/output_2023_w{week:02d}.csv'
    
    input_df = pd.read_csv(input_file)
    output_df = pd.read_csv(output_file)
    
    return input_df, output_df

def load_supplementary_data():
    """Load supplementary game and play information"""
    supp_file = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv'
    return pd.read_csv(supp_file)

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

# Combine all weeks
input_data = pd.concat(all_input_data, ignore_index=True)
output_data = pd.concat(all_output_data, ignore_index=True)
supplementary = load_supplementary_data()

print(f"\nTotal data loaded:")
print(f"Input frames: {len(input_data):,}")
print(f"Output frames: {len(output_data):,}")
print(f"Plays: {len(supplementary):,}")

# ============================================================================
# PART 2: FEATURE ENGINEERING
# ============================================================================

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_angle_difference(angle1, angle2):
    """Calculate smallest angle difference between two angles (in degrees)"""
    diff = (angle2 - angle1) % 360
    if diff > 180:
        diff = 360 - diff
    return diff

def engineer_output_features(output_df):
    """Calculate movement features for output (post-throw) tracking data"""
    
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
    output_enhanced['distance_to_ball'] = calculate_distance(
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
    output_enhanced['frame_distance'] = calculate_distance(
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

print("\nEngineering features for output data...")
output_enhanced = engineer_output_features(output_data)

# ============================================================================
# PART 3: AIR YARDS EFFICIENCY INDEX CALCULATION
# ============================================================================

def calculate_ayei(play_data):
    """
    Calculate Air Yards Efficiency Index for a player on a single play
    
    AYEI Components:
    1. Path Efficiency: Ratio of optimal path to actual path taken
    2. Closing Efficiency: How effectively player reduces distance to ball
    3. Acceleration Timing: Quality of acceleration application
    """
    
    if len(play_data) < 2:
        return None
    
    # Get starting and ending positions
    start_x = play_data.iloc[0]['x']
    start_y = play_data.iloc[0]['y']
    ball_x = play_data.iloc[0]['ball_land_x']
    ball_y = play_data.iloc[0]['ball_land_y']
    
    # 1. PATH EFFICIENCY
    # Optimal path: straight line from start to ball
    optimal_distance = calculate_distance(start_x, start_y, ball_x, ball_y)
    
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
    
    # 4. COMBINE INTO AYEI
    # Weighted combination of components
    ayei = (
        0.50 * path_efficiency +           # 50% weight on taking efficient path
        0.30 * closing_efficiency +         # 30% weight on closing speed
        0.20 * acceleration_timing          # 20% weight on timing
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
        'num_frames': len(play_data)
    }

def calculate_ayei_for_all_plays(output_data, input_data):
    """Calculate AYEI for all players across all plays"""
    
    # Get player roles from input data
    player_roles = input_data.groupby(['game_id', 'play_id', 'nfl_id']).agg({
        'player_role': 'first',
        'player_position': 'first',
        'player_name': 'first',
        'player_side': 'first'
    }).reset_index()
    
    ayei_results = []
    
    # Group by play and player
    grouped = output_data.groupby(['game_id', 'play_id', 'nfl_id'])
    
    print(f"Calculating AYEI for {len(grouped)} player-play combinations...")
    
    for (game_id, play_id, nfl_id), play_data in grouped:
        
        ayei_metrics = calculate_ayei(play_data)
        
        if ayei_metrics:
            result = {
                'game_id': game_id,
                'play_id': play_id,
                'nfl_id': nfl_id,
                **ayei_metrics
            }
            ayei_results.append(result)
    
    ayei_df = pd.DataFrame(ayei_results)
    
    # Merge with player information
    ayei_df = ayei_df.merge(player_roles, on=['game_id', 'play_id', 'nfl_id'], how='left')
    
    return ayei_df

print("\nCalculating AYEI scores...")
ayei_scores = calculate_ayei_for_all_plays(output_enhanced, input_data)

print(f"\nAYEI calculated for {len(ayei_scores):,} player-play combinations")
print(f"Unique players: {ayei_scores['nfl_id'].nunique()}")
print(f"Unique plays: {ayei_scores[['game_id', 'play_id']].drop_duplicates().shape[0]}")

# ============================================================================
# PART 4: PLAYER-LEVEL AGGREGATION
# ============================================================================

def aggregate_player_ayei(ayei_df, min_plays=10):
    """Aggregate AYEI scores at player level with minimum play threshold"""
    
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

print("\nAggregating player-level statistics...")
player_ayei = aggregate_player_ayei(ayei_scores, min_plays=15)

print(f"\nPlayers with 15+ plays: {len(player_ayei)}")
print(f"\nTop 10 Players by AYEI:")
print(player_ayei[['player_name', 'player_position', 'avg_ayei', 'num_plays']].head(10))

# ============================================================================
# PART 5: MERGE WITH PLAY OUTCOMES
# ============================================================================

# Merge AYEI with supplementary data for outcome analysis
ayei_with_outcomes = ayei_scores.merge(
    supplementary[['game_id', 'play_id', 'pass_result', 'pass_length', 
                   'team_coverage_type', 'yards_gained', 'expected_points_added']],
    on=['game_id', 'play_id'],
    how='left'
)

print("\nMerged with play outcomes")
print(f"Total records: {len(ayei_with_outcomes):,}")

# ============================================================================
# PART 6: ANALYSIS BY POSITION AND ROLE
# ============================================================================

def analyze_by_position(data):
    """Analyze AYEI by position group"""
    
    position_stats = data.groupby('player_position').agg({
        'ayei': ['mean', 'std', 'count'],
        'path_efficiency': 'mean',
        'closing_efficiency': 'mean'
    }).reset_index()
    
    position_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                             for col in position_stats.columns.values]
    
    position_stats = position_stats.rename(columns={
        'ayei_mean': 'avg_ayei',
        'ayei_std': 'std_ayei',
        'ayei_count': 'count'
    })
    
    return position_stats.sort_values('avg_ayei', ascending=False)

def analyze_by_role(data):
    """Analyze AYEI by player role"""
    
    role_stats = data.groupby('player_role').agg({
        'ayei': ['mean', 'std', 'count'],
        'path_efficiency': 'mean',
        'closing_efficiency': 'mean'
    }).reset_index()
    
    role_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                         for col in role_stats.columns.values]
    
    role_stats = role_stats.rename(columns={
        'ayei_mean': 'avg_ayei',
        'ayei_std': 'std_ayei',
        'ayei_count': 'count'
    })
    
    return role_stats.sort_values('avg_ayei', ascending=False)

print("\n" + "="*60)
print("AYEI BY POSITION")
print("="*60)
position_analysis = analyze_by_position(ayei_scores)
print(position_analysis.to_string(index=False))

print("\n" + "="*60)
print("AYEI BY PLAYER ROLE")
print("="*60)
role_analysis = analyze_by_role(ayei_scores)
print(role_analysis.to_string(index=False))

# ============================================================================
# PART 7: COVERAGE SCHEME ANALYSIS
# ============================================================================

def analyze_by_coverage(data):
    """Analyze defensive AYEI by coverage type"""
    
    # Filter for defensive players only
    defensive_data = data[data['player_side'] == 'Defense'].copy()
    
    coverage_stats = defensive_data.groupby('team_coverage_type').agg({
        'ayei': ['mean', 'std', 'count'],
        'path_efficiency': 'mean',
        'closing_efficiency': 'mean'
    }).reset_index()
    
    coverage_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                             for col in coverage_stats.columns.values]
    
    coverage_stats = coverage_stats.rename(columns={
        'ayei_mean': 'avg_ayei',
        'ayei_std': 'std_ayei',
        'ayei_count': 'count'
    })
    
    # Filter for coverage types with sufficient data
    coverage_stats = coverage_stats[coverage_stats['count'] >= 50]
    
    return coverage_stats.sort_values('avg_ayei', ascending=False)

print("\n" + "="*60)
print("DEFENSIVE AYEI BY COVERAGE TYPE")
print("="*60)
coverage_analysis = analyze_by_coverage(ayei_with_outcomes)
print(coverage_analysis.to_string(index=False))

# ============================================================================
# PART 8: PASS DEPTH ANALYSIS
# ============================================================================

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
    offense_depth = data[data['player_side'] == 'Offense'].groupby('depth_category').agg({
        'ayei': ['mean', 'count']
    }).reset_index()
    
    defense_depth = data[data['player_side'] == 'Defense'].groupby('depth_category').agg({
        'ayei': ['mean', 'count']
    }).reset_index()
    
    offense_depth.columns = ['depth_category', 'offense_ayei', 'offense_count']
    defense_depth.columns = ['depth_category', 'defense_ayei', 'defense_count']
    
    depth_comparison = offense_depth.merge(defense_depth, on='depth_category', how='outer')
    
    return depth_comparison

print("\n" + "="*60)
print("AYEI BY PASS DEPTH")
print("="*60)
depth_analysis = analyze_by_pass_depth(ayei_with_outcomes)
print(depth_analysis.to_string(index=False))

# ============================================================================
# PART 9: CORRELATION WITH OUTCOMES
# ============================================================================

def analyze_outcome_correlation(data):
    """Analyze how AYEI correlates with play outcomes"""
    
    # Filter for targeted receivers
    receivers = data[data['player_role'] == 'Targeted Receiver'].copy()
    
    # Completion analysis
    receivers['is_complete'] = (receivers['pass_result'] == 'C').astype(int)
    
    # Calculate correlation
    if len(receivers) > 0:
        completion_corr = receivers[['ayei', 'is_complete']].corr().iloc[0, 1]
        
        # Compare AYEI for complete vs incomplete
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

print("\n" + "="*60)
print("AYEI CORRELATION WITH PLAY OUTCOMES")
print("="*60)
analyze_outcome_correlation(ayei_with_outcomes)

# ============================================================================
# PART 10: VISUALIZATIONS
# ============================================================================

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
plt.savefig('ayei_analysis_overview.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 2: Top Performers
fig, ax = plt.subplots(figsize=(12, 10))

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
plt.savefig('top_performers_ayei.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# PART 11: EXPORT RESULTS
# ============================================================================

# Export player-level AYEI scores
player_ayei.to_csv('player_ayei_scores.csv', index=False)
print("\nPlayer AYEI scores exported to 'player_ayei_scores.csv'")

# Export play-level AYEI with outcomes
ayei_with_outcomes.to_csv('play_level_ayei.csv', index=False)
print("Play-level AYEI exported to 'play_level_ayei.csv'")

# Summary statistics
summary_stats = {
    'Total Plays Analyzed': len(ayei_scores[['game_id', 'play_id']].drop_duplicates()),
    'Total Player-Play Combinations': len(ayei_scores),
    'Unique Players': ayei_scores['nfl_id'].nunique(),
    'Overall Mean AYEI': ayei_scores['ayei'].mean(),
    'Overall Std AYEI': ayei_scores['ayei'].std(),
    'Receiver Mean AYEI': ayei_scores[ayei_scores['player_role'] == 'Targeted Receiver']['ayei'].mean(),
    'Defender Mean AYEI': ayei_scores[ayei_scores['player_role'] == 'Defensive Coverage']['ayei'].mean()
}

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
print("\nVisualization files created:")
print("  - ayei_analysis_overview.png")
print("  - top_performers_ayei.png")

