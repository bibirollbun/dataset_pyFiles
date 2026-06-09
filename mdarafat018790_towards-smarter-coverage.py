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


# =============================================
# Cell 1: Setup & Configuration
# Big Data Bowl 2026 - University Track
# =============================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
plt.rcParams['font.size'] = 12

# Machine Learning
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import roc_auc_score, roc_curve, auc

# Configuration
DATA_PATH = "/kaggle/input/nfl-big-data-bowl-2026-analytics"
MAX_PLAYS = None  # Set to 500 for testing, None for full analysis
OUTPUT_DIR = "./outputs"
PLOTS_DIR = "./plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Football constants
YARD_LENGTH = 120
YARD_WIDTH = 53.3
HASH_MARK_DISTANCE = 23.33

print("âœ… Setup complete! Ready for Big Data Bowl 2026 analysis.")


# =============================================
# Cell 2: Data Loading & Exploration
# =============================================

def load_all_tracking_data(data_path):
    """Load all weekly tracking data files"""
    tracking_files = []
    for i in range(1, 19):
        week_file = f"{data_path}/114239_nfl_competition_files_published_analytics_final/train/input_2023_w{i:02d}.csv"
        if os.path.exists(week_file):
            tracking_files.append(week_file)
    
    print(f"ğŸ“� Found {len(tracking_files)} weekly tracking files")
    
    # Load all tracking data
    all_tracking = []
    for file in tqdm(tracking_files, desc="Loading weekly data"):
        try:
            df_week = pd.read_csv(file)
            df_week['week'] = int(file.split('_w')[-1].split('.')[0])
            all_tracking.append(df_week)
        except Exception as e:
            print(f"âš ï¸� Error loading {file}: {e}")
    
    if all_tracking:
        tracking_df = pd.concat(all_tracking, ignore_index=True)
        print(f"âœ… Combined tracking data: {len(tracking_df):,} rows")
        return tracking_df
    else:
        raise ValueError("No tracking data loaded!")

def load_supplementary_data(data_path):
    """Load supplementary data"""
    supp_file = f"{data_path}/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv"
    if os.path.exists(supp_file):
        supp_df = pd.read_csv(supp_file)
        print(f"âœ… Supplementary data: {supp_df.shape}")
        return supp_df
    else:
        print("âš ï¸� No supplementary data found")
        return None

print("ğŸš€ Loading Big Data Bowl 2026 Data...")

# Load tracking data
df = load_all_tracking_data(DATA_PATH)

# Load supplementary data
supp_df = load_supplementary_data(DATA_PATH)

# Display actual data structure
print("\nğŸ”� ACTUAL COLUMN NAMES IN TRACKING DATA:")
print(df.columns.tolist())

print(f"\nğŸ“Š TRACKING DATA INFO:")
print(f"Shape: {df.shape}")
print(f"Sample data:")
print(df[['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y', 's', 'a', 'ball_land_x', 'ball_land_y']].head(2))

if supp_df is not None:
    print(f"\nğŸ“Š SUPPLEMENTARY DATA INFO:")
    print(f"Shape: {supp_df.shape}")
    print(f"Key columns: {[col for col in supp_df.columns if 'pass' in col.lower() or 'result' in col.lower()]}")


# =============================================
# Cell 3: Data Preprocessing & Cleaning
# =============================================

print("ğŸ”§ Preprocessing and cleaning data...")

# Convert to proper data types
df['game_id'] = df['game_id'].astype(int)
df['play_id'] = df['play_id'].astype(int)
df['frame_id'] = df['frame_id'].astype(int)
df['nfl_id'] = df['nfl_id'].fillna(-1).astype(int)

# Identify players vs ball
df['isBall'] = False
# Football typically has no nfl_id or special position
df.loc[df['nfl_id'].isna() | (df['nfl_id'] == -1), 'isBall'] = True

# Create team identification based on available columns
df['team'] = 'unknown'

# Method 1: Use player_position to identify offense/defense
if 'player_position' in df.columns:
    offensive_positions = ['QB', 'WR', 'RB', 'TE', 'FB', 'C', 'G', 'T', 'OT', 'OG']
    defensive_positions = ['CB', 'S', 'LB', 'DE', 'DT', 'NT', 'OLB', 'ILB', 'MLB', 'DB', 'DL']
    
    df.loc[df['player_position'].isin(offensive_positions), 'team'] = 'offense'
    df.loc[df['player_position'].isin(defensive_positions), 'team'] = 'defense'

# Method 2: Use player_role
if 'player_role' in df.columns:
    df.loc[df['player_role'].str.contains('offense', case=False, na=False), 'team'] = 'offense'
    df.loc[df['player_role'].str.contains('defense', case=False, na=False), 'team'] = 'defense'

# Method 3: Use player_side
if 'player_side' in df.columns:
    df.loc[df['player_side'].isin(['home', 'away']), 'team'] = 'offense'
    df.loc[df['player_side'] == 'defense', 'team'] = 'defense'

print(f"ğŸ�ˆ Identified {df['isBall'].sum():,} football records")
print(f"ğŸ‘¥ Team distribution: {df['team'].value_counts().to_dict()}")

# Create velocity components from speed and direction
if all(col in df.columns for col in ['s', 'dir']):
    print("ğŸ“� Computing velocity components from speed and direction...")
    # Convert direction from degrees to radians
    dir_rad = np.radians(df['dir'])
    df['vx'] = df['s'] * np.sin(dir_rad)  # x-component
    df['vy'] = df['s'] * np.cos(dir_rad)  # y-component
    print("âœ… Velocity components computed")

print("âœ… Data preprocessing complete!")
print(f"ğŸ“Š Final data shape: {df.shape}")
print(f"ğŸ�¯ Unique plays: {df[['game_id', 'play_id']].drop_duplicates().shape[0]}")


# =============================================
# Cell 4: Football Analysis Utilities
# =============================================

class FootballField:
    """Football field visualization utilities"""
    
    @staticmethod
    def create_field(ax=None, linecolor='white', linewidth=2, show_numbers=True):
        """Create professional football field background"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6.33))
        
        # Green field background
        ax.add_patch(patches.Rectangle((0, 0), YARD_LENGTH, YARD_WIDTH, 
                                     edgecolor=linecolor, facecolor='#2E8B57', linewidth=linewidth))
        
        # Yard lines every 5 yards
        for yard in range(0, YARD_LENGTH + 1, 5):
            if yard % 10 == 0:  # Every 10 yards
                ax.axvline(yard, color=linecolor, linewidth=linewidth, alpha=0.8)
                if show_numbers and 10 <= yard <= 110:
                    ax.text(yard, YARD_WIDTH/2 - 5, str(min(yard, 120-yard)), 
                           ha='center', va='center', fontsize=10, color=linecolor, fontweight='bold')
            else:  # Every 5 yards
                ax.axvline(yard, color=linecolor, linewidth=1, alpha=0.5)
        
        # Hash marks
        hash_yards = [HASH_MARK_DISTANCE/2, YARD_WIDTH - HASH_MARK_DISTANCE/2]
        for yard in range(11, YARD_LENGTH-9):
            for hash_y in hash_yards:
                ax.plot([yard, yard], [hash_y-0.5, hash_y+0.5], color=linecolor, linewidth=1)
        
        # End zones
        ax.add_patch(patches.Rectangle((0, 0), 10, YARD_WIDTH, 
                                     edgecolor=linecolor, facecolor='#006400', alpha=0.6))
        ax.add_patch(patches.Rectangle((YARD_LENGTH-10, 0), 10, YARD_WIDTH, 
                                     edgecolor=linecolor, facecolor='#006400', alpha=0.6))
        
        ax.set_xlim(0, YARD_LENGTH)
        ax.set_ylim(0, YARD_WIDTH)
        ax.set_aspect('equal')
        ax.axis('off')
        
        return ax

def calculate_movement_efficiency(trajectory_x, trajectory_y, target_x, target_y):
    """Calculate how efficiently a player moves toward target"""
    if len(trajectory_x) < 2:
        return 0.0
    
    efficiencies = []
    for i in range(1, len(trajectory_x)):
        # Vector to target
        dx_target = target_x - trajectory_x[i-1]
        dy_target = target_y - trajectory_y[i-1]
        
        # Player movement vector
        dx_move = trajectory_x[i] - trajectory_x[i-1]
        dy_move = trajectory_y[i] - trajectory_y[i-1]
        
        # Normalize vectors
        target_mag = np.sqrt(dx_target**2 + dy_target**2)
        move_mag = np.sqrt(dx_move**2 + dy_move**2)
        
        if target_mag > 0 and move_mag > 0:
            # Cosine similarity between movement and target direction
            cos_similarity = (dx_move * dx_target + dy_move * dy_target) / (move_mag * target_mag)
            efficiency = max(0, cos_similarity)  # Only positive movement toward target
            efficiencies.append(efficiency)
    
    return np.mean(efficiencies) if efficiencies else 0.0

def identify_pass_plays_from_supplementary(supp_df):
    """Identify pass plays from supplementary data"""
    if supp_df is None:
        return None
    
    # Filter for pass plays based on available columns
    pass_plays = supp_df.copy()
    
    if 'pass_result' in supp_df.columns:
        pass_plays = pass_plays[pass_plays['pass_result'].notna()]
        print(f"ğŸ�¯ Found {len(pass_plays)} pass plays from pass_result column")
    else:
        # Use play_description to identify pass plays
        pass_keywords = ['pass', 'throw', 'quarterback']
        pass_plays = pass_plays[
            pass_plays['play_description'].str.contains('|'.join(pass_keywords), case=False, na=False)
        ]
        print(f"ğŸ�¯ Found {len(pass_plays)} potential pass plays from play_description")
    
    return pass_plays[['game_id', 'play_id', 'pass_result', 'play_description']]

print("âœ… Football analysis utilities defined!")


# =============================================
# Cell 5: Separation Efficiency Index Engine
# =============================================

class SeparationEfficiencyAnalyzer:
    """Advanced SEI calculation engine"""
    
    def __init__(self):
        self.metrics_history = []
    
    def analyze_play_separation(self, play_data, ball_landing_x, ball_landing_y):
        """Comprehensive separation analysis for a single play"""
        
        # Identify offensive and defensive players
        offense = play_data[play_data['team'] == 'offense']
        defense = play_data[play_data['team'] == 'defense']
        
        if offense.empty or defense.empty:
            return None
        
        play_metrics = {
            'game_id': play_data['game_id'].iloc[0],
            'play_id': play_data['play_id'].iloc[0],
            'ball_landing_x': ball_landing_x,
            'ball_landing_y': ball_landing_y
        }
        
        # Analyze each offensive player
        offensive_metrics = []
        for player_id in offense['nfl_id'].unique():
            if player_id == -1:  # Skip unknown players
                continue
                
            player_metrics = self._analyze_player_separation(
                play_data, player_id, ball_landing_x, ball_landing_y, offense, defense
            )
            if player_metrics:
                offensive_metrics.append(player_metrics)
        
        if not offensive_metrics:
            return None
        
        # Aggregate play-level metrics
        off_metrics_df = pd.DataFrame(offensive_metrics)
        
        # Key SEI components
        play_metrics.update({
            'offensive_players': len(offensive_metrics),
            'avg_movement_efficiency': off_metrics_df['movement_efficiency'].mean(),
            'max_movement_efficiency': off_metrics_df['movement_efficiency'].max(),
            'avg_separation_gain': off_metrics_df['separation_gain'].mean(),
            'max_separation_gain': off_metrics_df['separation_gain'].max(),
            'avg_speed_efficiency': off_metrics_df['speed_efficiency'].mean(),
            'defensive_pressure': off_metrics_df['defensive_pressure'].mean(),
            'best_receiver_id': off_metrics_df.loc[off_metrics_df['movement_efficiency'].idxmax(), 'nfl_id'],
            'best_receiver_efficiency': off_metrics_df['movement_efficiency'].max()
        })
        
        # Calculate comprehensive SEI
        play_metrics['SEI'] = self._calculate_comprehensive_sei(play_metrics)
        
        return play_metrics
    
    def _analyze_player_separation(self, play_data, player_id, ball_x, ball_y, offense, defense):
        """Analyze separation metrics for individual player"""
        player_data = play_data[play_data['nfl_id'] == player_id].sort_values('frame_id')
        
        if len(player_data) < 5:  # Need sufficient frames
            return None
        
        # Extract trajectory and movement data
        player_x = player_data['x'].values
        player_y = player_data['y'].values
        player_speed = player_data['s'].values
        
        # Movement efficiency toward ball
        movement_efficiency = calculate_movement_efficiency(player_x, player_y, ball_x, ball_y)
        
        # Speed efficiency (normalized)
        max_speed = np.max(player_speed)
        speed_efficiency = min(max_speed / 12.0, 1.0)  # Normalize to max NFL speed
        
        # Separation analysis
        separation_gain = self._calculate_separation_gain(player_data, defense)
        
        # Defensive pressure
        defensive_pressure = self._calculate_defensive_pressure(player_data, defense)
        
        return {
            'nfl_id': player_id,
            'movement_efficiency': movement_efficiency,
            'speed_efficiency': speed_efficiency,
            'separation_gain': separation_gain,
            'defensive_pressure': defensive_pressure,
            'max_speed': max_speed,
            'frames_analyzed': len(player_data)
        }
    
    def _calculate_separation_gain(self, player_data, defense_data):
        """Calculate how much separation player gains from defenders"""
        separation_changes = []
        
        for frame_id in player_data['frame_id'].unique():
            player_frame = player_data[player_data['frame_id'] == frame_id]
            if player_frame.empty:
                continue
                
            player_x, player_y = player_frame['x'].iloc[0], player_frame['y'].iloc[0]
            
            # Find nearest defender
            defender_dists = []
            for _, defender in defense_data[defense_data['frame_id'] == frame_id].iterrows():
                dist = np.sqrt((player_x - defender['x'])**2 + (player_y - defender['y'])**2)
                defender_dists.append(dist)
            
            if defender_dists:
                separation_changes.append(min(defender_dists))
        
        if len(separation_changes) > 1:
            return separation_changes[-1] - separation_changes[0]  # Separation gain
        return 0.0
    
    def _calculate_defensive_pressure(self, player_data, defense_data):
        """Calculate defensive pressure on player throughout play"""
        pressure_scores = []
        
        for frame_id in player_data['frame_id'].unique():
            player_frame = player_data[player_data['frame_id'] == frame_id]
            if player_frame.empty:
                continue
                
            player_x, player_y = player_frame['x'].iloc[0], player_frame['y'].iloc[0]
            
            # Count defenders within pressure radius
            defenders_near = 0
            for _, defender in defense_data[defense_data['frame_id'] == frame_id].iterrows():
                dist = np.sqrt((player_x - defender['x'])**2 + (player_y - defender['y'])**2)
                if dist < 5.0:  # 5 yard pressure radius
                    defenders_near += 1
            
            pressure_scores.append(defenders_near)
        
        return np.mean(pressure_scores) if pressure_scores else 0.0
    
    def _calculate_comprehensive_sei(self, play_metrics):
        """Calculate final Separation Efficiency Index"""
        # Weighted combination of key factors
        movement_score = play_metrics['max_movement_efficiency'] * 0.35
        separation_score = min(play_metrics['max_separation_gain'] / 10.0, 1.0) * 0.30
        speed_score = play_metrics['avg_speed_efficiency'] * 0.20
        pressure_score = max(0, 1 - play_metrics['defensive_pressure'] / 5.0) * 0.15
        
        sei = movement_score + separation_score + speed_score + pressure_score
        return min(max(sei, 0), 1)  # Clamp to [0, 1]

print("âœ… Separation Efficiency Index engine defined!")


# =============================================
# Cell 6: Play Identification & SEI Calculation
# =============================================

print("ğŸ�¯ Identifying pass plays and calculating SEI...")

def identify_pass_plays_with_ball(tracking_df, supp_df, max_plays=None):
    """Identify pass plays using supplementary data and ball landing positions"""
    
    # First, get pass plays from supplementary data
    pass_plays_from_supp = identify_pass_plays_from_supplementary(supp_df)
    
    if pass_plays_from_supp is None or pass_plays_from_supp.empty:
        print("â�Œ No pass plays identified from supplementary data")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"ğŸ�¯ Found {len(pass_plays_from_supp)} pass plays in supplementary data")
    
    # Get unique game_id, play_id combinations from tracking data that match supplementary pass plays
    tracking_plays = tracking_df[['game_id', 'play_id']].drop_duplicates()
    valid_pass_plays = pass_plays_from_supp.merge(tracking_plays, on=['game_id', 'play_id'])
    
    print(f"ğŸ“Š {len(valid_pass_plays)} pass plays have tracking data")
    
    if max_plays:
        valid_pass_plays = valid_pass_plays.head(max_plays)
        print(f"ğŸ”� Analyzing first {max_plays} plays for efficiency")
    
    pass_play_details = []
    ball_landing_data = []
    
    play_count = 0
    for _, play_info in tqdm(valid_pass_plays.iterrows(), total=len(valid_pass_plays), desc="Processing pass plays"):
        game_id = play_info['game_id']
        play_id = play_info['play_id']
        
        # Get all frames for this play
        play_data = tracking_df[
            (tracking_df['game_id'] == game_id) & 
            (tracking_df['play_id'] == play_id)
        ]
        
        if play_data.empty:
            continue
        
        # Get ball landing position from tracking data
        ball_data = play_data[['ball_land_x', 'ball_land_y']].dropna()
        if ball_data.empty:
            continue
            
        # Use the first available ball landing position
        ball_landing_x = ball_data['ball_land_x'].iloc[0]
        ball_landing_y = ball_data['ball_land_y'].iloc[0]
        
        # Skip if ball landing position is invalid
        if pd.isna(ball_landing_x) or pd.isna(ball_landing_y):
            continue
        
        # Get pass result from supplementary data if available
        pass_result = play_info.get('pass_result', 'Unknown')
        
        pass_play_details.append({
            'game_id': game_id,
            'play_id': play_id,
            'pass_result': pass_result,
            'ball_landing_x': ball_landing_x,
            'ball_landing_y': ball_landing_y,
            'total_frames': len(play_data),
            'offensive_players': len(play_data[play_data['team'] == 'offense']['nfl_id'].unique()),
            'defensive_players': len(play_data[play_data['team'] == 'defense']['nfl_id'].unique())
        })
        
        play_count += 1
    
    pass_plays_df = pd.DataFrame(pass_play_details)
    
    print(f"âœ… Identified {len(pass_plays_df)} pass plays with valid ball tracking")
    return pass_plays_df

# Identify pass plays
pass_plays_df = identify_pass_plays_with_ball(df, supp_df, MAX_PLAYS)

# Calculate SEI for all plays
print("\nğŸ�›ï¸� Calculating Separation Efficiency Index...")
sei_analyzer = SeparationEfficiencyAnalyzer()
play_metrics_list = []

if not pass_plays_df.empty:
    for _, play_info in tqdm(pass_plays_df.iterrows(), total=len(pass_plays_df), desc="Calculating SEI"):
        game_id = play_info['game_id']
        play_id = play_info['play_id']
        ball_x = play_info['ball_landing_x']
        ball_y = play_info['ball_landing_y']
        
        try:
            # Get all play data
            play_data = df[
                (df['game_id'] == game_id) & 
                (df['play_id'] == play_id)
            ]
            
            if play_data.empty:
                continue
            
            # Analyze separation efficiency
            play_metrics = sei_analyzer.analyze_play_separation(play_data, ball_x, ball_y)
            
            if play_metrics:
                # Add supplementary information
                play_metrics['pass_result'] = play_info.get('pass_result', 'Unknown')
                play_metrics['week'] = play_info.get('week', play_data['week'].iloc[0] if 'week' in play_data.columns else 0)
                play_metrics_list.append(play_metrics)
                
        except Exception as e:
            continue

    # Create comprehensive features dataframe
    if play_metrics_list:
        features_df = pd.DataFrame(play_metrics_list)
        
        # Add additional derived features
        features_df['distance_to_sideline'] = np.minimum(
            features_df['ball_landing_y'], 
            YARD_WIDTH - features_df['ball_landing_y']
        )
        features_df['field_position'] = features_df['ball_landing_x'] / YARD_LENGTH
        features_df['red_zone'] = (features_df['ball_landing_x'] >= 100).astype(int)
        
        # Convert pass_result to binary completion
        if 'pass_result' in features_df.columns:
            completion_map = {
                'C': 1, 'COMPLETE': 1,
                'I': 0, 'INCOMPLETE': 0, 'IN': 0, 'INC': 0
            }
            features_df['completion'] = features_df['pass_result'].map(completion_map)
            # For any unmapped values, try to infer from the string
            unmapped = features_df['completion'].isna()
            if unmapped.any():
                features_df.loc[unmapped & features_df['pass_result'].str.contains('complete', case=False, na=False), 'completion'] = 1
                features_df.loc[unmapped & features_df['pass_result'].str.contains('incomplete', case=False, na=False), 'completion'] = 0
        
        print(f"âœ… Successfully analyzed {len(features_df)} plays")
        
        # Display SEI statistics
        print(f"\nğŸ“ˆ SEI STATISTICS:")
        print(f"Mean SEI: {features_df['SEI'].mean():.3f}")
        print(f"Std SEI: {features_df['SEI'].std():.3f}")
        print(f"Min SEI: {features_df['SEI'].min():.3f}")
        print(f"Max SEI: {features_df['SEI'].max():.3f}")
        
        if 'completion' in features_df.columns:
            completion_rate = features_df['completion'].mean()
            print(f"Completion rate: {completion_rate:.3f} ({completion_rate*100:.1f}%)")
        
        # Save features
        features_df.to_csv(os.path.join(OUTPUT_DIR, 'play_features_with_sei.csv'), index=False)
        print(f"ğŸ’¾ Saved features to {OUTPUT_DIR}/play_features_with_sei.csv")
        
    else:
        print("â�Œ No plays successfully analyzed!")
        features_df = pd.DataFrame() 
else:
    print("â�Œ No pass plays available for SEI calculation")
    features_df = pd.DataFrame()


# =============================================
# Cell 7: Machine Learning & Predictive Modeling
# =============================================

if not features_df.empty and 'completion' in features_df.columns:
    print("ğŸ¤– Building machine learning models...")
    
    # Feature selection for modeling
    feature_columns = [
        'avg_movement_efficiency', 'max_movement_efficiency',
        'avg_separation_gain', 'max_separation_gain', 
        'avg_speed_efficiency', 'defensive_pressure',
        'offensive_players', 'defensive_players',
        'distance_to_sideline', 'field_position', 'red_zone', 'SEI'
    ]
    
    # Ensure all features exist
    available_features = [f for f in feature_columns if f in features_df.columns]
    X = features_df[available_features].fillna(0)
    y = features_df['completion']
    groups = features_df['game_id']
    
    if len(X) > 10:  # Enough samples for modeling
        # Models
        models = {
            'Logistic Regression': Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('scale', StandardScaler()),
                ('lr', LogisticRegression(max_iter=1000, random_state=42))
            ]),
            'Random Forest': Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
            ])
        }
        
        # Cross-validation
        gkf = GroupKFold(n_splits=3)
        results = {}
        
        for name, model in models.items():
            try:
                cv_scores = cross_validate(
                    model, X, y, cv=gkf.split(X, y, groups),
                    scoring=['roc_auc'], 
                    return_train_score=False
                )
                results[name] = {
                    'auc_mean': np.mean(cv_scores['test_roc_auc']),
                    'auc_std': np.std(cv_scores['test_roc_auc'])
                }
                print(f"âœ… {name}: AUC = {results[name]['auc_mean']:.3f} Â± {results[name]['auc_std']:.3f}")
            except Exception as e:
                print(f"âš ï¸� {name} failed: {e}")
        
        # Train final model on all data
        if results:
            final_model_name = max(results.keys(), key=lambda x: results[x]['auc_mean'])
            final_model = models[final_model_name]
            final_model.fit(X, y)
            
            # Feature importance
            if hasattr(final_model.named_steps.get('rf', None), 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': available_features,
                    'importance': final_model.named_steps['rf'].feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
                print("\nğŸ’¡ TOP FEATURES BY IMPORTANCE:")
                for _, row in importance_df.head().iterrows():
                    print(f"   {row['feature']}: {row['importance']:.3f}")
            
            # Save model predictions
            features_df['predicted_completion'] = final_model.predict_proba(X)[:, 1]
            features_df.to_csv(os.path.join(OUTPUT_DIR, 'model_predictions.csv'), index=False)
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y, features_df['predicted_completion'])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic - SEI Model')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
    else:
        print("âš ï¸� Not enough samples for machine learning modeling")
else:
    print("â„¹ï¸� Machine learning requires completion labels")


# =============================================
# Cell 8: Advanced Analytics & Insights
# =============================================

if not features_df.empty:
    print("ğŸ“ˆ Generating advanced analytics and insights...")
    
    # 8.1 Comprehensive SEI Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # SEI Distribution
    axes[0,0].hist(features_df['SEI'], bins=30, alpha=0.7, color='#2E8B57', edgecolor='white')
    axes[0,0].axvline(features_df['SEI'].mean(), color='red', linestyle='--', label=f'Mean: {features_df["SEI"].mean():.3f}')
    axes[0,0].set_xlabel('Separation Efficiency Index (SEI)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('SEI Distribution', fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # SEI vs Movement Efficiency
    axes[0,1].scatter(features_df['avg_movement_efficiency'], features_df['SEI'], alpha=0.6)
    axes[0,1].set_xlabel('Average Movement Efficiency')
    axes[0,1].set_ylabel('SEI')
    axes[0,1].set_title('SEI vs Movement Efficiency', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    # SEI vs Defensive Pressure
    axes[0,2].scatter(features_df['defensive_pressure'], features_df['SEI'], alpha=0.6, color='red')
    axes[0,2].set_xlabel('Defensive Pressure')
    axes[0,2].set_ylabel('SEI')
    axes[0,2].set_title('SEI vs Defensive Pressure', fontweight='bold')
    axes[0,2].grid(True, alpha=0.3)
    
    # SEI by Red Zone
    if 'red_zone' in features_df.columns:
        red_zone_sei = features_df[features_df['red_zone'] == 1]['SEI']
        normal_sei = features_df[features_df['red_zone'] == 0]['SEI']
        axes[1,0].boxplot([normal_sei, red_zone_sei], labels=['Normal', 'Red Zone'])
        axes[1,0].set_ylabel('SEI')
        axes[1,0].set_title('SEI: Normal vs Red Zone', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
    
    # SEI vs Completion
    if 'completion' in features_df.columns:
        complete_sei = features_df[features_df['completion'] == 1]['SEI']
        incomplete_sei = features_df[features_df['completion'] == 0]['SEI']
        axes[1,1].boxplot([incomplete_sei, complete_sei], labels=['Incomplete', 'Complete'])
        axes[1,1].set_ylabel('SEI')
        axes[1,1].set_title('SEI by Pass Completion', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
    
    # Field Position Impact
    if 'field_position' in features_df.columns:
        pos_bins = pd.cut(features_df['field_position'], bins=5)
        sei_by_pos = features_df.groupby(pos_bins)['SEI'].mean()
        sei_by_pos.plot(kind='bar', ax=axes[1,2], color='orange', alpha=0.7)
        axes[1,2].set_xlabel('Field Position')
        axes[1,2].set_ylabel('Average SEI')
        axes[1,2].set_title('SEI by Field Position', fontweight='bold')
        axes[1,2].tick_params(axis='x', rotation=45)
        axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'comprehensive_sei_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8.2 Correlation Analysis
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    correlation_with_sei = features_df[numeric_cols].corr()['SEI'].sort_values(ascending=False)
    
    print("\nğŸ”— TOP CORRELATIONS WITH SEI:")
    for feature, corr in correlation_with_sei.head(8).items():
        if feature != 'SEI':
            print(f"   {feature}: {corr:.3f}")
    
    # 8.3 High vs Low SEI Comparison
    high_sei_threshold = features_df['SEI'].quantile(0.8)
    low_sei_threshold = features_df['SEI'].quantile(0.2)
    
    high_sei_plays = features_df[features_df['SEI'] >= high_sei_threshold]
    low_sei_plays = features_df[features_df['SEI'] <= low_sei_threshold]
    
    print(f"\nğŸ“Š HIGH vs LOW SEI COMPARISON:")
    print(f"High SEI threshold (>80%): {high_sei_threshold:.3f}")
    print(f"Low SEI threshold (<20%): {low_sei_threshold:.3f}")
    print(f"High SEI plays: {len(high_sei_plays)}")
    print(f"Low SEI plays: {len(low_sei_plays)}")
    
    if 'completion' in features_df.columns:
        high_sei_completion = high_sei_plays['completion'].mean()
        low_sei_completion = low_sei_plays['completion'].mean()
        print(f"High SEI completion rate: {high_sei_completion:.3f} ({high_sei_completion*100:.1f}%)")
        print(f"Low SEI completion rate: {low_sei_completion:.3f} ({low_sei_completion*100:.1f}%)")
        print(f"Completion difference: {high_sei_completion - low_sei_completion:.3f}")
    
    print(f"High SEI avg movement efficiency: {high_sei_plays['avg_movement_efficiency'].mean():.3f}")
    print(f"Low SEI avg movement efficiency: {low_sei_plays['avg_movement_efficiency'].mean():.3f}")
    print(f"High SEI avg defensive pressure: {high_sei_plays['defensive_pressure'].mean():.2f}")
    print(f"Low SEI avg defensive pressure: {low_sei_plays['defensive_pressure'].mean():.2f}")


# =============================================
# Cell 9: Player Rankings & NFL Applications
# =============================================

if not features_df.empty:
    print("ğŸ�† Generating player rankings and NFL applications...")
    
    # 9.1 Player Performance Rankings
    if 'best_receiver_id' in features_df.columns:
        player_performance = []
        
        for _, play in features_df.iterrows():
            if play['best_receiver_id'] != -1:
                player_performance.append({
                    'nfl_id': play['best_receiver_id'],
                    'game_id': play['game_id'],
                    'play_id': play['play_id'],
                    'SEI': play['SEI'],
                    'movement_efficiency': play['best_receiver_efficiency'],
                    'defensive_pressure': play['defensive_pressure']
                })
        
        if player_performance:
            player_df = pd.DataFrame(player_performance)
            
            # Aggregate by player
            player_rankings = player_df.groupby('nfl_id').agg({
                'SEI': ['mean', 'std', 'count'],
                'movement_efficiency': 'mean',
                'defensive_pressure': 'mean'
            }).round(4)
            
            # Flatten column names
            player_rankings.columns = ['_'.join(col).strip() for col in player_rankings.columns.values]
            player_rankings = player_rankings.rename(columns={
                'SEI_mean': 'avg_sei',
                'SEI_std': 'std_sei', 
                'SEI_count': 'plays_analyzed',
                'movement_efficiency_mean': 'avg_movement_eff',
                'defensive_pressure_mean': 'avg_def_pressure'
            })
            
            # Filter players with sufficient plays
            player_rankings = player_rankings[player_rankings['plays_analyzed'] >= 3]
            player_rankings = player_rankings.sort_values('avg_sei', ascending=False)
            
            player_rankings.to_csv(os.path.join(OUTPUT_DIR, 'player_sei_rankings.csv'))
            
            print(f"âœ… Ranked {len(player_rankings)} players with sufficient data")
            print("\nğŸ�… TOP 10 PLAYERS BY SEI:")
            for i, (player_id, row) in enumerate(player_rankings.head(10).iterrows()):
                print(f"   {i+1:2d}. Player {player_id}: SEI = {row['avg_sei']:.3f} ({row['plays_analyzed']} plays)")
    
    # 9.2 Coaching Applications
    print("\nğŸ�¯ COACHING APPLICATIONS OF SEI:")
    print("1. ğŸ�¯ Player Evaluation: Identify most efficient route runners")
    print("2. ğŸ�ˆ Game Planning: Target matchups with favorable SEI profiles") 
    print("3. ğŸ“Š Scheme Design: Optimize routes based on SEI components")
    print("4. ğŸ”� Scouting: Evaluate receiver separation ability in draft")
    print("5. ğŸ“ˆ Development: Target specific areas for player improvement")
    
    # 9.3 Team-level Insights
    print(f"\nğŸ“Š TEAM-LEVEL INSIGHTS:")
    print(f"â€¢ Average SEI across all plays: {features_df['SEI'].mean():.3f}")
    print(f"â€¢ Percentage of high-efficiency plays (SEI > 0.7): {(features_df['SEI'] > 0.7).mean()*100:.1f}%")
    
    if 'red_zone' in features_df.columns:
        red_zone_boost = features_df[features_df['red_zone']==1]['SEI'].mean() - features_df[features_df['red_zone']==0]['SEI'].mean()
        print(f"â€¢ Red zone efficiency impact: {red_zone_boost:+.3f}")
    
    if 'completion' in features_df.columns:
        completion_corr = features_df['SEI'].corr(features_df['completion'])
        print(f"â€¢ Correlation with completion: {completion_corr:.3f}")
    
    # 9.4 Save final report
    final_report = {
        'analysis_summary': {
            'total_plays_analyzed': len(features_df),
            'analysis_period': '2023 Season Weeks 1-18',
            'key_metric': 'Separation Efficiency Index (SEI)',
            'metric_range': '0-1 (higher = better separation)'
        },
        'key_findings': {
            'average_sei': float(features_df['SEI'].mean()),
            'consistency': float(features_df['SEI'].std()),
            'high_efficiency_threshold': 0.7,
            'high_efficiency_plays_pct': float((features_df['SEI'] > 0.7).mean() * 100)
        },
        'nfl_applications': [
            "Player evaluation and development",
            "Game planning and matchup optimization", 
            "Draft scouting and free agency",
            "Scheme design and play calling",
            "Performance tracking and analytics"
        ]
    }
    
    import json
    with open(os.path.join(OUTPUT_DIR, 'final_analysis_report.json'), 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\nğŸ’¾ Final report saved to {OUTPUT_DIR}/final_analysis_report.json")


# =============================================
# Cell 10: Final Summary & Submission Preparation
# =============================================

print("""
ğŸ�‰ BIG DATA BOWL 2026 ANALYSIS COMPLETE!
==========================================

ğŸ“Š ANALYSIS SUMMARY:
""")

if not features_df.empty:
    print(f"â€¢ Plays Analyzed: {len(features_df):,}")
    print(f"â€¢ Average SEI: {features_df['SEI'].mean():.3f}")
    print(f"â€¢ SEI Range: {features_df['SEI'].min():.3f} - {features_df['SEI'].max():.3f}")
    print(f"â€¢ High Efficiency Plays (SEI > 0.7): {(features_df['SEI'] > 0.7).sum():,} ({(features_df['SEI'] > 0.7).mean()*100:.1f}%)")
    
    if os.path.exists(os.path.join(OUTPUT_DIR, 'player_sei_rankings.csv')):
        player_rankings = pd.read_csv(os.path.join(OUTPUT_DIR, 'player_sei_rankings.csv'))
        print(f"â€¢ Players Ranked: {len(player_rankings):,}")

print(f"""
ğŸ“� OUTPUTS GENERATED:
{OUTPUT_DIR}/
   â”œâ”€â”€ play_features_with_sei.csv (Complete play-level features)
   â”œâ”€â”€ player_sei_rankings.csv (Player performance rankings) 
   â”œâ”€â”€ feature_importance.csv (ML model feature importance)
   â”œâ”€â”€ model_predictions.csv (Play outcome predictions)
   â””â”€â”€ final_analysis_report.json (Comprehensive summary)

{PLOTS_DIR}/
   â”œâ”€â”€ comprehensive_sei_dashboard.png (Main results dashboard)
   â”œâ”€â”€ roc_curve.png (Model performance)
   â””â”€â”€ ball_landing_distribution.png (Spatial analysis)

ğŸ�ˆ SEPARATION EFFICIENCY INDEX (SEI) INNOVATIONS:
1. Multi-dimensional player movement analysis
2. Context-aware efficiency calculations  
3. Defensive pressure quantification
4. Real-time separation tracking
5. Field position adjustments

ğŸ�¯ NFL APPLICATIONS:
â€¢ Player Evaluation: Identify elite separators
â€¢ Game Planning: Optimize matchups and schemes  
â€¢ Draft Analysis: Quantify receiver separation skills
â€¢ Development: Target specific improvement areas
â€¢ Strategy: Data-driven play calling decisions

ğŸ“� SUBMISSION NEXT STEPS:
1. Write 2000-word narrative explaining SEI methodology and applications
2. Create executive summary for coaches and scouts
3. Prepare presentation highlighting key insights
4. Showcase visualizations in media gallery
5. Demonstrate NFL operational value

Ø¨Ù�Ø³Ù’Ù…Ù� Ø§Ù„Ù„Ù‡Ù� Ø§Ù„Ø±Ù�Ù‘Ø­Ù’Ù…Ù°Ù†Ù� Ø§Ù„Ø±Ù�Ù‘Ø­Ù�ÙŠÙ’Ù…Ù�
May Allah grant you success in this competition! ğŸŒŸ
""")

# Cleanup
import gc
gc.collect()
print("ğŸ§¹ Memory cleanup completed!")

