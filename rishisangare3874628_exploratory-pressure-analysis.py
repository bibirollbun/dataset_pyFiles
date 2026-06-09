# =============================================================================
# CELL 6: LOAD LIBRARIES
# =============================================================================
# Purpose: Import all required packages for analysis
# Note: These are all available in the Kaggle environment
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
import warnings
import os
import gc

# For 3D visualizations (Plotly is available on Kaggle)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# For progress tracking
from tqdm.notebook import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Pandas display settings
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

# Matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# COLOR PALETTE (NFL-Themed)
# =============================================================================
# These colors will be used consistently throughout all visualizations

COLORS = {
    'primary': '#013369',      # NFL Blue - main color for charts
    'secondary': '#D50A0A',    # NFL Red - accent/comparison color
    'success': '#00AA00',      # Green - positive outcomes
    'warning': '#FFB612',      # Gold - neutral/caution
    'background': '#f8f9fa',   # Light gray background
    'text': '#333333'          # Dark text
}

# =============================================================================
# FIGURE SIZE DEFAULTS
# =============================================================================

FIG_SIZES = {
    'small': (8, 5),
    'medium': (12, 6),
    'large': (14, 8),
    'wide': (16, 6)
}

print("âœ… Libraries loaded successfully")
print(f"   Pandas version: {pd.__version__}")
print(f"   NumPy version: {np.__version__}")



# =============================================================================
# CELL 7: LOAD DATA
# =============================================================================
# Purpose: Load tracking data and supplementary information
# 
# Data Structure:
# - Input data: Player positions BEFORE the throw (frame-by-frame)
# - Supplementary data: Play context (route type, coverage, outcome)
# =============================================================================

# â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
# â”‚  ğŸ“� FILE PATHS - ADJUST IF NEEDED                                          â”‚
# â”‚  [YOUR INPUT]: If your data is in a different location, update these paths â”‚
# â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

# Kaggle paths (default)
BASE_PATH = '/kaggle/input/nfl-big-data-bowl-2026-analytics/'
DATA_PATH = BASE_PATH + '114239_nfl_competition_files_published_analytics_final/'
TRAIN_PATH = DATA_PATH + 'train/'

# =============================================================================
# STEP 1: Load Supplementary Data (Play Context)
# =============================================================================
# This contains route types, coverage, and outcomes - essential for our analysis

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

print("\nğŸ“Š Loading supplementary data (play context)...")
supp_df = pd.read_csv(DATA_PATH + 'supplementary_data.csv', low_memory=False)
print(f"   â†’ {len(supp_df):,} plays loaded")
print(f"   â†’ Columns: {len(supp_df.columns)}")

# =============================================================================
# STEP 2: Load Tracking Data (All Weeks)
# =============================================================================
# This may take a few minutes - we're loading ~14,000 plays Ã— ~50 frames Ã— ~22 players

print("\nğŸ“Š Loading tracking data (this may take 2-5 minutes)...")
all_weeks = []

for week in range(1, 19):  # Weeks 1-18
    try:
        week_file = f'input_2023_w{week:02d}.csv'
        week_path = TRAIN_PATH + week_file
        
        week_df = pd.read_csv(week_path)
        week_df['week'] = week  # Add week identifier
        all_weeks.append(week_df)
        
        print(f"   âœ“ Week {week:2d}: {len(week_df):>10,} frames")
        
    except FileNotFoundError:
        print(f"   âœ— Week {week:2d}: Not found (skipping)")

# Combine all weeks into single DataFrame
print("\nğŸ“Š Combining all weeks...")
input_df = pd.concat(all_weeks, ignore_index=True)

# Free memory
del all_weeks
gc.collect()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("DATA LOADING COMPLETE")
print("=" * 70)
print(f"\nâœ… Total tracking frames: {len(input_df):,}")
print(f"âœ… Unique plays: {input_df.groupby(['game_id', 'play_id']).ngroups:,}")
print(f"âœ… Supplementary plays: {len(supp_df):,}")
print(f"\nğŸ’¾ Memory usage: {input_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")



# =============================================================================
# CELL 8A: DEBUG - INSPECT ACTUAL DATA VALUES (NEW)
# =============================================================================
# PURPOSE: Before we start processing, let's see what values actually exist
#          in the critical columns. This helps us parse correctly.
# =============================================================================

print("=" * 70)
print("ğŸ”� DATA INSPECTION: WHAT VALUES ACTUALLY EXIST?")
print("=" * 70)

# Check play_action values
print("\n1ï¸�âƒ£ PLAY ACTION VALUES:")
print("-" * 60)
if 'play_action' in supp_df.columns:
    play_action_values = supp_df['play_action'].value_counts().head(10)
    print(play_action_values)
    print(f"\nUnique values: {supp_df['play_action'].unique()[:20]}")
else:
    print("   âš ï¸� 'play_action' column not found")

# Check coverage columns
print("\n2ï¸�âƒ£ COVERAGE TYPE VALUES:")
print("-" * 60)
if 'team_coverage_man_zone' in supp_df.columns:
    coverage_mz = supp_df['team_coverage_man_zone'].value_counts().head(10)
    print("team_coverage_man_zone:")
    print(coverage_mz)
    print(f"\nUnique values: {supp_df['team_coverage_man_zone'].unique()[:20]}")
else:
    print("   âš ï¸� 'team_coverage_man_zone' column not found")

if 'team_coverage_type' in supp_df.columns:
    coverage_type = supp_df['team_coverage_type'].value_counts().head(10)
    print("\nteam_coverage_type:")
    print(coverage_type)
    print(f"\nUnique values: {supp_df['team_coverage_type'].unique()[:20]}")
else:
    print("   âš ï¸� 'team_coverage_type' column not found")

# Check frame_id distribution
print("\n3ï¸�âƒ£ FRAME_ID DISTRIBUTION:")
print("-" * 60)
frame_stats = input_df.groupby(['game_id', 'play_id'])['frame_id'].agg(['min', 'max', 'count'])
print(f"Average frames per play: {frame_stats['count'].mean():.1f}")
print(f"Average min frame_id: {frame_stats['min'].mean():.1f}")
print(f"Average max frame_id: {frame_stats['max'].mean():.1f}")
print(f"\nSample frame ranges:")
print(frame_stats.head(5))

# Check if week exists
print("\n4ï¸�âƒ£ WEEK/GAME_DATE INFO:")
print("-" * 60)
if 'week' in supp_df.columns:
    print(f"Week column exists: {supp_df['week'].nunique()} unique weeks")
    print(supp_df['week'].value_counts().sort_index())
elif 'game_date' in supp_df.columns:
    print("Week column not found, but game_date exists")
    print(f"Date range: {supp_df['game_date'].min()} to {supp_df['game_date'].max()}")
else:
    print("   âš ï¸� Neither 'week' nor 'game_date' found")

print("\n" + "=" * 70)
print("âœ… INSPECTION COMPLETE - Using these values for parsing")
print("=" * 70)



# =============================================================================
# CELL 8: DATA EXPLORATION
# =============================================================================
# Purpose: Understand the structure of our data before analysis
# This helps us verify we have the columns we need
# =============================================================================

print("=" * 70)
print("INPUT DATA (Tracking) - Pre-Throw Player Positions")
print("=" * 70)

print(f"\nğŸ“Š Shape: {input_df.shape[0]:,} rows Ã— {input_df.shape[1]} columns")
print(f"\nğŸ“‹ Columns:")
for i, col in enumerate(input_df.columns, 1):
    print(f"   {i:2d}. {col}")

print("\nğŸ“‹ Sample Row:")
display(input_df.head(1).T)

# =============================================================================

print("\n" + "=" * 70)
print("SUPPLEMENTARY DATA (Play Context)")
print("=" * 70)

print(f"\nğŸ“Š Shape: {supp_df.shape[0]:,} rows Ã— {supp_df.shape[1]} columns")

# Check for KEY columns we need
print("\nğŸ”‘ KEY COLUMNS CHECK:")
key_cols = [
    'game_id', 'play_id',                          # Identifiers
    'pass_result',                                  # Outcome (C, I, IN)
    'route_of_targeted_receiver',                   # CRITICAL!
    'team_coverage_type', 'team_coverage_man_zone', # Coverage
    'pass_length',                                  # Route depth
    'play_action',                                  # Play action
    'down', 'yards_to_go',                         # Situation
    'yards_gained', 'expected_points_added'        # Outcomes
]

for col in key_cols:
    if col in supp_df.columns:
        print(f"   âœ“ {col}")
    else:
        print(f"   âœ— {col} (NOT FOUND - may need adjustment)")

# =============================================================================

print("\n" + "=" * 70)
print("PLAYER ROLES (Critical for Separation Calculation)")
print("=" * 70)

print("\nğŸ“‹ Player roles in tracking data:")
print(input_df['player_role'].value_counts())



# =============================================================================
# CELL 9: DATA VALIDATION
# =============================================================================
# Purpose: Ensure data is clean and usable for our analysis
# We check for:
# 1. Required columns exist
# 2. Player roles are as expected
# 3. Route types are available
# =============================================================================

print("DATA VALIDATION")
print("=" * 70)

validation_passed = True

# =============================================================================
# CHECK 1: Required columns in tracking data
# =============================================================================

print("\n1ï¸�âƒ£ TRACKING DATA COLUMNS")
required_tracking_cols = ['game_id', 'play_id', 'frame_id', 'player_role', 'x', 'y', 'player_name']

missing = [col for col in required_tracking_cols if col not in input_df.columns]
if missing:
    print(f"   â�Œ Missing columns: {missing}")
    validation_passed = False
else:
    print("   âœ… All required columns present")

# =============================================================================
# CHECK 2: Player roles
# =============================================================================

print("\n2ï¸�âƒ£ PLAYER ROLES")
roles = input_df['player_role'].unique()
print(f"   Found roles: {list(roles)}")

# Check for the roles we need
if 'Targeted Receiver' in roles:
    n_targeted = len(input_df[input_df['player_role'] == 'Targeted Receiver'])
    print(f"   âœ… Targeted Receiver frames: {n_targeted:,}")
else:
    print("   â�Œ 'Targeted Receiver' role not found!")
    validation_passed = False

if 'Defensive Coverage' in roles:
    n_defense = len(input_df[input_df['player_role'] == 'Defensive Coverage'])
    print(f"   âœ… Defensive Coverage frames: {n_defense:,}")
else:
    print("   â�Œ 'Defensive Coverage' role not found!")
    validation_passed = False

# =============================================================================
# CHECK 3: Route types in supplementary data
# =============================================================================

print("\n3ï¸�âƒ£ ROUTE TYPES")
if 'route_of_targeted_receiver' in supp_df.columns:
    routes = supp_df['route_of_targeted_receiver'].value_counts()
    print(f"   âœ… Route types found: {len(routes)}")
    print("\n   Top 10 routes:")
    for route, count in routes.head(10).items():
        print(f"      {route}: {count:,} plays")
else:
    print("   â�Œ Route column not found!")
    validation_passed = False

# =============================================================================
# CHECK 4: Coverage types
# =============================================================================

print("\n4ï¸�âƒ£ COVERAGE TYPES")
if 'team_coverage_type' in supp_df.columns:
    coverages = supp_df['team_coverage_type'].value_counts()
    print(f"   âœ… Coverage types found: {len(coverages)}")
    print("\n   Coverage distribution:")
    for cov, count in coverages.head(8).items():
        print(f"      {cov}: {count:,} plays")

# =============================================================================
# FINAL VERDICT
# =============================================================================

print("\n" + "=" * 70)
if validation_passed:
    print("âœ… ALL VALIDATION CHECKS PASSED - Ready to proceed!")
else:
    print("â�Œ VALIDATION FAILED - Please check the issues above")
print("=" * 70)



# =============================================================================
# CELL 10: FILTER & MERGE DATA
# =============================================================================
# Purpose: 
# 1. Keep only completed and incomplete passes (exclude interceptions for now)
# 2. Merge tracking data with play context (route type, coverage, etc.)
# 
# This creates our main analysis DataFrame!
# =============================================================================

print("FILTERING AND MERGING DATA")
print("=" * 70)

# =============================================================================
# STEP 1: Filter supplementary data to pass plays only
# =============================================================================
print("\n1ï¸�âƒ£ FILTERING TO PASS PLAYS")

# Check pass result values
print(f"   Pass result values: {supp_df['pass_result'].unique()}")

# Keep only Completions (C) and Incompletions (I)
# Exclude interceptions (IN) for cleaner analysis
pass_plays = supp_df[supp_df['pass_result'].isin(['C', 'I'])].copy()
print(f"   â†’ Pass plays (C or I): {len(pass_plays):,}")

# =============================================================================
# STEP 2: Define columns to keep from supplementary data
# =============================================================================
# We select ALL relevant columns for our 7 findings
supp_cols = [
    # Core identifiers
    'game_id', 'play_id',
    
    # FINDING 1 & 2: Route timing & completion
    'pass_result', 
    'route_of_targeted_receiver',
    'pass_length',
    
    # FINDING 3: Man vs Zone
    'team_coverage_type', 
    'team_coverage_man_zone',
    
    # FINDING 4: Play action
    'play_action',
    
    # FINDING 5: Situational timing
    'down', 
    'yards_to_go',
    
    # FINDING 7: EPA validation
    'yards_gained', 
    'expected_points_added',
    
    # Additional context
    'offense_formation',
    'receiver_alignment',
    'dropback_type',
    'dropback_distance',
    'defenders_in_the_box',
    'quarter',
    'game_clock'
]

# Keep only columns that exist in the data
available_cols = [c for c in supp_cols if c in pass_plays.columns]
print(f"\n2ï¸�âƒ£ COLUMNS SELECTED: {len(available_cols)} of {len(supp_cols)}")

# =============================================================================
# STEP 2A: PARSE COVERAGE AND PLAY ACTION (before merge)
# =============================================================================
print("\n2Aï¸�âƒ£ PARSING COVERAGE & PLAY ACTION")
print("-" * 60)

# Parse coverage type (Man vs Zone)
def parse_coverage(val):
    val_str = str(val).upper()
    if 'MAN' in val_str:  # Matches "MAN_COVERAGE" or "COVER_1_MAN"
        return 'Man'
    elif 'ZONE' in val_str:  # Matches "ZONE_COVERAGE" or "COVER_3_ZONE"
        return 'Zone'
    return 'Unknown'

pass_plays['man_zone_parsed'] = pass_plays['team_coverage_man_zone'].apply(parse_coverage)
cov_counts = pass_plays['man_zone_parsed'].value_counts()
print(f"   Coverage parsed:")
for c, cnt in cov_counts.items():
    print(f"      {c}: {cnt:,} plays")

# Parse play action (Y/N)
def parse_pa(val):
    if val in [True, 'Y', 'YES', 'TRUE', 1]:
        return 'Y'
    elif val in [False, 'N', 'NO', 'FALSE', 0, None]:
        return 'N'
    return 'Unknown'

pass_plays['play_action_parsed'] = pass_plays['play_action'].apply(parse_pa)
pa_counts = pass_plays['play_action_parsed'].value_counts()
print(f"   Play Action parsed:")
for pa, cnt in pa_counts.items():
    print(f"      {pa}: {cnt:,} plays")

# Add parsed columns to available_cols
available_cols.extend(['man_zone_parsed', 'play_action_parsed'])

# =============================================================================
# STEP 3: Merge tracking data with supplementary data
# =============================================================================
print("\n3ï¸�âƒ£ MERGING TRACKING DATA WITH PLAY CONTEXT")
print("   This may take a minute...")

merged_df = input_df.merge(
    pass_plays[available_cols],
    on=['game_id', 'play_id'],
    how='inner'
)

print(f"   â†’ Merged data: {len(merged_df):,} frames")

# =============================================================================
# STEP 4: Verify route and coverage info
# =============================================================================
print("\n4ï¸�âƒ£ ROUTE & COVERAGE DISTRIBUTION")

# Route types
route_counts = merged_df.groupby(['game_id', 'play_id'])['route_of_targeted_receiver'].first().value_counts()
print(f"\n   ğŸ“Š Top Route Types:")
for route, count in route_counts.head(10).items():
    print(f"      {route}: {count:,} plays")

# Coverage types
if 'team_coverage_type' in merged_df.columns:
    coverage_counts = merged_df.groupby(['game_id', 'play_id'])['team_coverage_type'].first().value_counts()
    print(f"\n   ğŸ›¡ï¸� Top Coverage Types:")
    for cov, count in coverage_counts.head(8).items():
        print(f"      {cov}: {count:,} plays")

# CLEANUP: Free memory from original tracking data
# =============================================================================
print("\n5ï¸�âƒ£ CLEANING UP MEMORY")
del input_df
gc.collect()
print(f"   â†’ Current memory usage: {merged_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("âœ… DATA PREPARATION COMPLETE")
print("=" * 70)
print(f"\nğŸ“Š Final dataset: {len(merged_df):,} frames")
print(f"ğŸ“Š Unique plays: {merged_df.groupby(['game_id', 'play_id']).ngroups:,}")
print(f"ğŸ“Š Route types: {merged_df['route_of_targeted_receiver'].nunique()}")
print(f"\nğŸ�¯ Ready for methodology section!")



# =============================================================================
# CELL 12: SEPARATION CALCULATION FUNCTION
# =============================================================================
# Purpose: Define the CORE function to calculate separation at each frame
# 
# This is the heart of our analysis!
# 
# Input: DataFrame for a single play
# Output: Timeline of separation values (one per frame)
# =============================================================================

def calculate_separation_timeline(play_df):
    """
    Calculate separation between targeted receiver and nearest defender
    at each frame of a play.
    
    Parameters:
    -----------
    play_df : DataFrame
        Tracking data for a single play (all frames, all players)
        Must contain: frame_id, player_role, x, y
        
    Returns:
    --------
    DataFrame with columns: 
        - frame_id: Frame number
        - separation: Distance to nearest defender (yards)
        - time_seconds: Time in seconds (frame_id / 10)
        - receiver_x, receiver_y: Receiver position
        
    Returns None if data is insufficient
    """
    
    # =========================================================================
    # STEP 1: Identify the targeted receiver and defenders
    # =========================================================================
    
    receiver = play_df[play_df['player_role'] == 'Targeted Receiver']
    defenders = play_df[play_df['player_role'] == 'Defensive Coverage']
    
    # Validation: Need both receiver and defenders
    if len(receiver) == 0 or len(defenders) == 0:
        return None
    
    # =========================================================================
    # STEP 2: Calculate separation at each frame
    # =========================================================================
    
    timeline = []
    
    # Get all unique frames
    frames = sorted(receiver['frame_id'].unique())
    
    for frame_id in frames:
        
        # Get receiver position at this frame
        rec_frame = receiver[receiver['frame_id'] == frame_id]
        if len(rec_frame) == 0:
            continue
        
        rec_x = rec_frame['x'].values[0]
        rec_y = rec_frame['y'].values[0]
        
        # Get all defender positions at this frame
        def_frame = defenders[defenders['frame_id'] == frame_id]
        if len(def_frame) == 0:
            continue
        
        # Calculate Euclidean distance to each defender
        # Formula: sqrt((x2-x1)Â² + (y2-y1)Â²)
        distances = np.sqrt(
            (def_frame['x'].values - rec_x)**2 + 
            (def_frame['y'].values - rec_y)**2
        )
        
        # Minimum distance = separation from nearest threat
        min_separation = distances.min()
        
        timeline.append({
            'frame_id': frame_id,
            'separation': min_separation,
            'time_seconds': frame_id / 10.0,  # 10 fps â†’ seconds
            'receiver_x': rec_x,
            'receiver_y': rec_y
        })
    
    # Return as DataFrame (or None if empty)
    return pd.DataFrame(timeline) if timeline else None


print("âœ… Separation calculation function defined")
print("\nğŸ“‹ Function signature:")
print("   calculate_separation_timeline(play_df) â†’ DataFrame")
print("\nğŸ“‹ Returns columns:")
print("   - frame_id: Frame number")
print("   - separation: Distance to nearest defender (yards)")
print("   - time_seconds: Time in seconds")
print("   - receiver_x, receiver_y: Receiver position")



# =============================================================================
# CELL 12A: PRESSURE CALCULATION FUNCTION
# =============================================================================
# Purpose: Calculate when pass rush pressure arrives at the QB
# Key insight: Optimal timing must occur BEFORE pressure collapses the pocket
# =============================================================================

def calculate_pressure_timeline(play_df, threshold=7.0):
    """
    Calculate pressure timeline for a play
    
    Parameters:
    -----------
    play_df : DataFrame
        Tracking data for a single play
    threshold : float
        Distance (yards) at which we consider pressure imminent
        Default: 7.0 yards (conservative, QB must throw NOW)
    
    Returns:
    --------
    dict with:
        - pressure_timeline: array of distances to nearest rusher per frame
        - pressure_frame: frame when pressure arrives (or None)
        - pressure_time: time in seconds when pressure arrives (or None)
        - clean_pocket: whether play had a clean pocket throughout
    """
    
    # =========================================================================
    # STEP 1: Identify the QB
    # =========================================================================
    
    qb = play_df[play_df['player_role'] == 'Passer']
    
    # Validation: Need QB
    if len(qb) == 0:
        return None
    
    # =========================================================================
    # STEP 2: Identify defenders (all defenders, not just coverage)
    # =========================================================================
    
    # Get ALL defenders (coverage + rush)
    defenders = play_df[play_df['nfl_id'].isin(
        play_df[play_df['player_role'].str.contains('Defensive', na=False)]['nfl_id'].unique()
    ) | play_df['player_role'].str.contains('Pass Rush', na=False)]
    
    # Fallback: just use all defenders
    if len(defenders) == 0:
        defenders = play_df[~play_df['player_role'].isin(['Passer', 'Targeted Receiver'])]
    
    if len(defenders) == 0:
        return None
    
    # =========================================================================
    # STEP 3: Calculate distance to QB at each frame
    # =========================================================================
    
    timeline = []
    
    # Get all unique frames
    frames = sorted(qb['frame_id'].unique())
    
    for frame_id in frames:
        
        # Get QB position at this frame
        qb_frame = qb[qb['frame_id'] == frame_id]
        if len(qb_frame) == 0:
            continue
        
        qb_x = qb_frame['x'].values[0]
        qb_y = qb_frame['y'].values[0]
        
        # Get all defender positions at this frame
        def_frame = defenders[defenders['frame_id'] == frame_id]
        if len(def_frame) == 0:
            continue
        
        # Calculate Euclidean distance to each defender
        distances = np.sqrt(
            (def_frame['x'].values - qb_x)**2 + 
            (def_frame['y'].values - qb_y)**2
        )
        
        # Minimum distance = nearest threat
        min_distance = distances.min()
        
        timeline.append({
            'frame_id': frame_id,
            'min_distance_to_qb': min_distance,
            'time_seconds': frame_id / 10.0,  # 10 fps â†’ seconds
        })
    
    if not timeline:
        return None
    
    # Convert to arrays for analysis
    timeline_df = pd.DataFrame(timeline)
    distances = timeline_df['min_distance_to_qb'].values
    frames = timeline_df['frame_id'].values
    
    # =========================================================================
    # STEP 4: Find when pressure arrives
    # =========================================================================
    
    # Find first frame where distance < threshold
    pressure_frames = np.where(distances < threshold)[0]
    
    if len(pressure_frames) > 0:
        pressure_frame_idx = pressure_frames[0]
        pressure_frame = frames[pressure_frame_idx]
        pressure_time = pressure_frame / 10.0
        clean_pocket = False
    else:
        pressure_frame = None
        pressure_time = None
        clean_pocket = True
    
    return {
        'pressure_timeline': distances,
        'pressure_frames': frames,
        'pressure_frame': pressure_frame,
        'pressure_time': pressure_time,
        'clean_pocket': clean_pocket,
        'avg_distance': distances.mean()
    }


print("âœ… Pressure calculation function defined")
print("\nğŸ“‹ Function signature:")
print("   calculate_pressure_timeline(play_df, threshold=7.0) â†’ dict")
print("\nğŸ“‹ Returns:")
print("   - pressure_timeline: array of min distances per frame")
print("   - pressure_frame: frame when pressure < threshold")
print("   - pressure_time: time in seconds when pressure arrives")
print("   - clean_pocket: True if no pressure detected")
print("\nğŸ“‹ Threshold values to consider:")
print("   - 1.5 yards: Very imminent (aggressive)")
print("   - 2.0 yards: Imminent (recommended default)")
print("   - 2.5 yards: Functional pocket (conservative)")




# =============================================================================
# CELL 13: SINGLE PLAY DEMONSTRATION
# =============================================================================
# Purpose: Show how separation calculation works on ONE play
# This helps validate our methodology before scaling up
# =============================================================================

print("SINGLE PLAY DEMONSTRATION")
print("=" * 70)

# =============================================================================
# STEP 1: Pick a sample play (SLANT route for clear example)
# =============================================================================

# Find plays with SLANT routes
slant_plays = merged_df[merged_df['route_of_targeted_receiver'] == 'SLANT']

if len(slant_plays) == 0:
    # Fallback: use most common route
    most_common_route = merged_df['route_of_targeted_receiver'].mode()[0]
    print(f"   Note: No SLANT routes found. Using {most_common_route} instead.")
    sample_plays = merged_df[merged_df['route_of_targeted_receiver'] == most_common_route]
else:
    sample_plays = slant_plays

# Get first play from this route type
sample_play_info = sample_plays.groupby(['game_id', 'play_id']).first().reset_index().iloc[0]
sample_game = sample_play_info['game_id']
sample_play_id = sample_play_info['play_id']

print(f"\nğŸ“‹ Sample Play Selected:")
print(f"   Game ID: {sample_game}")
print(f"   Play ID: {sample_play_id}")
print(f"   Route: {sample_play_info['route_of_targeted_receiver']}")
print(f"   Result: {sample_play_info['pass_result']}")

# =============================================================================
# STEP 2: Get tracking data for this play
# =============================================================================

play_data = merged_df[
    (merged_df['game_id'] == sample_game) & 
    (merged_df['play_id'] == sample_play_id)
]

print(f"\nğŸ“Š Play Data:")
print(f"   Total frames: {play_data['frame_id'].nunique()}")
print(f"   Players tracked: {play_data['player_role'].nunique()} roles")

# =============================================================================
# STEP 3: Calculate separation timeline
# =============================================================================

sep_timeline = calculate_separation_timeline(play_data)

print(f"\nğŸ“Š Separation Timeline:")
print(f"   Frames analyzed: {len(sep_timeline)}")
print(f"   Duration: {sep_timeline['time_seconds'].max():.1f} seconds")
print(f"   Min separation: {sep_timeline['separation'].min():.2f} yards")
print(f"   Max separation: {sep_timeline['separation'].max():.2f} yards")

# =============================================================================
# STEP 4: Find peak separation (optimal throw moment)
# =============================================================================

peak_idx = sep_timeline['separation'].idxmax()
peak_time = sep_timeline.loc[peak_idx, 'time_seconds']
peak_sep = sep_timeline.loc[peak_idx, 'separation']

print(f"\nâ­� PEAK SEPARATION (Optimal Throw Moment):")
print(f"   Time: {peak_time:.2f} seconds")
print(f"   Separation: {peak_sep:.2f} yards")

# =============================================================================
# STEP 5: Visualize
# =============================================================================

fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])

# Plot separation over time
ax.plot(sep_timeline['time_seconds'], sep_timeline['separation'], 
        linewidth=3, color=COLORS['primary'], marker='o', markersize=4,
        label='Separation from nearest defender')

# Mark peak separation
ax.scatter([peak_time], [peak_sep], s=200, c=COLORS['secondary'], 
           zorder=5, marker='*', label=f'Peak: {peak_sep:.1f} yds at {peak_time:.1f}s')

# Add vertical line at peak
ax.axvline(peak_time, color=COLORS['secondary'], linestyle='--', alpha=0.5)

# Labels and formatting
ax.set_xlabel('Time After Snap (seconds)', fontsize=12)
ax.set_ylabel('Separation from Nearest Defender (yards)', fontsize=12)
ax.set_title(f'Separation Timeline - {sample_play_info["route_of_targeted_receiver"]} Route\n(Result: {sample_play_info["pass_result"]})', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

# Add annotation
ax.annotate('OPTIMAL\nTHROW\nWINDOW', 
            xy=(peak_time, peak_sep),
            xytext=(peak_time + 0.3, peak_sep - 0.5),
            fontsize=10, fontweight='bold', color=COLORS['secondary'],
            arrowprops=dict(arrowstyle='->', color=COLORS['secondary']))

plt.tight_layout()
plt.savefig('01_single_play_separation.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nâœ… Visualization saved as '01_single_play_separation.png'")



# =============================================================================
# CELL 15: PEAK DETECTION & TIMING ANALYSIS (PRESSURE-CONSTRAINED VERSION)
# =============================================================================
# Purpose: Find optimal throw timing BEFORE pressure arrives
# Key change: Optimization is now constrained by pocket collapse
# =============================================================================

def find_optimal_throw_pressure_constrained(sep_timeline_df, pressure_data):
    """
    Find the optimal throw timing considering pressure constraints
    
    OLD LOGIC: Find peak separation (unconstrained)
    NEW LOGIC: Find peak separation BEFORE pressure arrives
    
    Parameters:
    -----------
    sep_timeline_df : DataFrame
        Separation timeline from calculate_separation_timeline()
    pressure_data : dict
        Pressure data from calculate_pressure_timeline()
    
    Returns:
    --------
    dict with optimal timing info + pressure context
    """
    
    if sep_timeline_df is None or len(sep_timeline_df) < 3:
        return None
    
    if pressure_data is None:
        # Fallback to old method if no pressure data
        peak_idx = sep_timeline_df['separation'].values.argmax()
        peak_frame = sep_timeline_df.iloc[peak_idx]
        
        return {
            'optimal_time': peak_frame['time_seconds'],
            'optimal_separation': peak_frame['separation'],
            'optimal_frame': peak_frame['frame_id'],
            'pressure_time': None,
            'pressure_constrained': False,
            'clean_pocket': True
        }
    
    # =========================================================================
    # STEP 1: Get pressure constraint
    # =========================================================================
    
    pressure_time = pressure_data['pressure_time']
    clean_pocket = pressure_data['clean_pocket']
    
    # =========================================================================
    # STEP 2: Filter separation timeline to viable frames
    # =========================================================================
    
    if pressure_time is not None:
        # Can only consider frames BEFORE pressure
        viable_timeline = sep_timeline_df[
            sep_timeline_df['time_seconds'] < pressure_time
        ]
        pressure_constrained = True
    else:
        # No pressure detected, use all frames
        viable_timeline = sep_timeline_df
        pressure_constrained = False
    
    # =========================================================================
    # STEP 3: Find peak separation within viable window
    # =========================================================================
    
    if len(viable_timeline) == 0:
        # Pressure arrived immediately - emergency situation
        # Use first available frame
        peak_frame = sep_timeline_df.iloc[0]
        emergency = True
    else:
        peak_idx = viable_timeline['separation'].values.argmax()
        peak_frame = viable_timeline.iloc[peak_idx]
        emergency = False
    
    return {
        'optimal_time': peak_frame['time_seconds'],
        'optimal_separation': peak_frame['separation'],
        'optimal_frame': peak_frame['frame_id'],
        'pressure_time': pressure_time,
        'pressure_constrained': pressure_constrained,
        'clean_pocket': clean_pocket,
        'emergency_throw': emergency,
        'viable_window_size': len(viable_timeline) * 0.1  # seconds
    }


# =============================================================================
# TIMING ANALYSIS WITH PRESSURE CATEGORIES
# =============================================================================

TIMING_THRESHOLD = 0.4  # seconds (Â±0.4s window around optimal)

def analyze_throw_timing_with_pressure(actual_time, optimal_time, pressure_time, threshold=TIMING_THRESHOLD):
    """
    Classify throw timing with pressure awareness
    
    CATEGORIES (NEW, more nuanced):
    1. on_time_clean: Hit window, no pressure
    2. on_time_pressured: Hit window despite pressure
    3. anticipation: Early throw, no pressure yet (can be good on deep routes)
    4. pressure_forced: Threw early due to pressure
    5. missed_window: Late, but before pressure
    6. under_pressure: Late, pressure arrived
    
    Parameters:
    -----------
    actual_time : float
        When QB actually threw (seconds)
    optimal_time : float
        Optimal throw time from find_optimal_throw()
    pressure_time : float or None
        When pressure arrived (seconds)
    threshold : float
        Window size (seconds)
    
    Returns:
    --------
    dict with timing classification and metrics
    """
    
    # Calculate timing error
    timing_error = actual_time - optimal_time
    
    # Check if on-time (within threshold)
    on_time = abs(timing_error) <= threshold
    
    # Set pressure time to infinity if None (no pressure)
    if pressure_time is None:
        pressure_time = float('inf')
    
    # Pressure status
    under_pressure = actual_time >= pressure_time
    pressure_coming = (pressure_time - actual_time) < 0.5  # Within 0.5s of pressure
    
    # =========================================================================
    # CLASSIFY INTO NUANCED CATEGORIES
    # =========================================================================
    
    if on_time:
        if under_pressure:
            category = 'on_time_pressured'
        else:
            category = 'on_time_clean'
    
    elif timing_error < 0:  # Threw early
        if pressure_coming or under_pressure:
            category = 'pressure_forced'
        else:
            category = 'anticipation'
    
    else:  # Threw late
        if under_pressure:
            category = 'under_pressure'
        else:
            category = 'missed_window'
    
    # =========================================================================
    # LEGACY CATEGORIES (for backward compatibility)
    # =========================================================================
    
    if on_time:
        timing_category_simple = 'on_time'
    elif timing_error < 0:
        timing_category_simple = 'too_early'
    else:
        timing_category_simple = 'too_late'
    
    return {
        'actual_throw_time': actual_time,
        'timing_error': timing_error,
        'timing_category': timing_category_simple,  # Legacy
        'timing_category_detailed': category,  # NEW
        'on_time': on_time,
        'under_pressure': under_pressure
    }


print("âœ… Pressure-constrained timing functions defined")
print("\nğŸ“‹ Key changes:")
print("   1. Optimal timing now considers pressure constraint")
print("   2. Six timing categories instead of three")
print("   3. Separates QB skill from OL quality")




# =============================================================================
# CELL 16: PROCESS ALL PLAYS (PRESSURE-CONSTRAINED VERSION)
# =============================================================================
# Purpose: Apply pressure-constrained analysis to ALL plays in the dataset
# Key change: Now includes pressure calculation and pressure-aware timing
# 
# âš ï¸� WARNING: This cell may take 10-20 minutes to run on the full dataset!
# =============================================================================

def process_single_play(play_df, supp_row):
    """
    Process a single play with PRESSURE-CONSTRAINED timing analysis
    
    NEW: Adds pressure calculation and pressure-aware timing categories
    """
    
    # =========================================================================
    # STEP 1: Calculate separation timeline (UNCHANGED)
    # =========================================================================
    
    sep_timeline = calculate_separation_timeline(play_df)
    
    if sep_timeline is None or len(sep_timeline) < 3:
        return None
    
    # =========================================================================
    # STEP 2: Calculate pressure timeline (NEW!)
    # =========================================================================
    
    pressure_data = calculate_pressure_timeline(play_df, threshold=7.0)
    
    # =========================================================================
    # STEP 3: Find optimal throw timing (NOW PRESSURE-CONSTRAINED)
    # =========================================================================
    
    optimal = find_optimal_throw_pressure_constrained(sep_timeline, pressure_data)
    
    if optimal is None:
        return None
    
    # =========================================================================
    # STEP 4: Analyze actual throw timing (NOW WITH PRESSURE CATEGORIES)
    # =========================================================================
    
    # Infer actual throw time from data
    # Assuming last frame is when ball was thrown
    actual_throw_time = sep_timeline['time_seconds'].max()
    
    timing = analyze_throw_timing_with_pressure(
        actual_throw_time, 
        optimal['optimal_time'],
        optimal['pressure_time']
    )
    
    # =========================================================================
    # STEP 5: Extract player names
    # =========================================================================
    
    qb_df = play_df[play_df['player_role'] == 'Passer']
    rec_df = play_df[play_df['player_role'] == 'Targeted Receiver']
    
    qb_name = qb_df['player_name'].iloc[0] if len(qb_df) > 0 else 'Unknown'
    rec_name = rec_df['player_name'].iloc[0] if len(rec_df) > 0 else 'Unknown'
    
    # =========================================================================
    # STEP 6: Compile results (NOW WITH PRESSURE DATA)
    # =========================================================================
    
    def safe_get(key, default='Unknown'):
        if key not in supp_row:
            return default
        val = supp_row[key]
        if pd.isna(val):
            return default
        return val
    
    return {
        # Identifiers
        'game_id': supp_row['game_id'],
        'play_id': supp_row['play_id'],
        
        # Player info
        'qb_name': qb_name,
        'receiver_name': rec_name,
        
        # Core analysis
        'route_type': safe_get('route_of_targeted_receiver'),
        'pass_result': supp_row['pass_result'],
        'pass_length': safe_get('pass_length', 0),
        
        # Coverage
        'coverage_type': safe_get('team_coverage_type', 'Unknown'),
        'man_zone': safe_get('man_zone_parsed', 'Unknown'),
        
        # Play action
        'play_action': safe_get('play_action_parsed', 'N'),
        
        # Situational
        'down': safe_get('down', 0),
        'yards_to_go': safe_get('yards_to_go', 0),
        
        # Outcomes
        'yards_gained': safe_get('yards_gained', 0),
        'epa': safe_get('expected_points_added', 0),
        
        # Additional context
        'formation': safe_get('offense_formation', 'Unknown'),
        'receiver_alignment': safe_get('receiver_alignment', 'Unknown'),
        'dropback_type': safe_get('dropback_type', 'Unknown'),
        'dropback_distance': safe_get('dropback_distance', 0),
        'defenders_in_box': safe_get('defenders_in_the_box', 0),
        
        # Optimal timing results (with pressure context)
        'optimal_time': optimal['optimal_time'],
        'optimal_separation': optimal['optimal_separation'],
        'pressure_time': optimal['pressure_time'],
        'pressure_constrained': optimal['pressure_constrained'],
        'clean_pocket': optimal['clean_pocket'],
        'viable_window_size': optimal.get('viable_window_size', None),
        
        # Actual timing analysis (with pressure categories)
        'actual_throw_time': timing['actual_throw_time'],
        'timing_error': timing['timing_error'],
        'timing_category': timing['timing_category'],  # Legacy (on_time/too_early/too_late)
        'timing_category_detailed': timing['timing_category_detailed'],  # NEW (6 categories)
        'on_time': timing['on_time'],
        'under_pressure': timing['under_pressure']
    }



# =============================================================================
# TEST CELL: VALIDATE PRESSURE-CONSTRAINED ANALYSIS
# =============================================================================
# Purpose: Test on 10 plays before running full dataset
# This helps catch bugs early
# =============================================================================

print("ğŸ§ª TESTING PRESSURE-CONSTRAINED ANALYSIS ON SAMPLE")
print("=" * 70)

# =============================================================================
# STEP 1: Select 10 random plays for testing
# =============================================================================

sample_plays = merged_df.groupby(['game_id', 'play_id']).ngroups
print(f"\nğŸ“Š Total plays available: {sample_plays:,}")
print(f"ğŸ“Š Testing on: 10 plays\n")

# Get 10 random game_id/play_id combinations
play_keys = merged_df[['game_id', 'play_id']].drop_duplicates().sample(n=10, random_state=42)

test_results = []
test_errors = []

print("ğŸ”¬ Processing test plays...")
print("-" * 70)

for idx, (_, row) in enumerate(play_keys.iterrows(), 1):
    game_id = row['game_id']
    play_id = row['play_id']
    
    print(f"\n[{idx}/10] Play: {game_id}-{play_id}")
    
    try:
        # Get play data
        play_df = merged_df[
            (merged_df['game_id'] == game_id) & 
            (merged_df['play_id'] == play_id)
        ]
        
        # Get supplementary data
        supp_row = pass_plays[
            (pass_plays['game_id'] == game_id) & 
            (pass_plays['play_id'] == play_id)
        ]
        
        if len(supp_row) == 0:
            print("   âš ï¸� No supplementary data - skipping")
            continue
        
        supp_row = supp_row.iloc[0]
        
        # Process the play
        result = process_single_play(play_df, supp_row)
        
        if result:
            test_results.append(result)
            
            # Display key info
            print(f"   âœ… Processed successfully")
            print(f"      Route: {result['route_type']}")
            print(f"      Optimal time: {result['optimal_time']:.2f}s")
            print(f"      Pressure time: {result['pressure_time']:.2f}s" if result['pressure_time'] else "      Pressure: None (clean pocket)")
            print(f"      Clean pocket: {'YES' if result['clean_pocket'] else 'NO'}")
            print(f"      Timing category: {result['timing_category_detailed']}")
        else:
            print("   âš ï¸� Processing returned None")
            test_errors.append(f"{game_id}-{play_id}: returned None")
            
    except Exception as e:
        print(f"   â�Œ Error: {str(e)[:100]}")
        test_errors.append(f"{game_id}-{play_id}: {str(e)[:100]}")

# =============================================================================
# STEP 2: Validate Results
# =============================================================================

print("\n" + "=" * 70)
print("ğŸ�¯ TEST RESULTS SUMMARY")
print("=" * 70)

if len(test_results) > 0:
    test_df = pd.DataFrame(test_results)
    
    print(f"\nâœ… Successfully processed: {len(test_results)}/10 plays")
    print(f"â�Œ Errors/skipped: {len(test_errors)}/10 plays")
    
    print(f"\nğŸ“Š NEW COLUMNS CREATED:")
    new_cols = ['pressure_time', 'pressure_constrained', 'clean_pocket', 
                'timing_category_detailed', 'under_pressure', 'viable_window_size']
    for col in new_cols:
        if col in test_df.columns:
            if col in ['pressure_constrained', 'clean_pocket', 'under_pressure']:
                count = test_df[col].sum()
                print(f"   âœ… {col}: {count}/{len(test_df)} plays")
            elif col == 'timing_category_detailed':
                cats = test_df[col].value_counts().to_dict()
                print(f"   âœ… {col}: {cats}")
            else:
                non_null = test_df[col].notna().sum()
                print(f"   âœ… {col}: {non_null}/{len(test_df)} plays with data")
        else:
            print(f"   â�Œ {col}: MISSING!")
    
    print(f"\nğŸ“Š PRESSURE DATA QUALITY:")
    if 'pressure_time' in test_df.columns:
        pressure_plays = test_df[test_df['pressure_time'].notna()]
        if len(pressure_plays) > 0:
            avg_pressure = pressure_plays['pressure_time'].mean()
            print(f"   Average pressure time: {avg_pressure:.2f}s")
            print(f"   Range: {pressure_plays['pressure_time'].min():.2f}s - {pressure_plays['pressure_time'].max():.2f}s")
            
            if 1.0 < avg_pressure < 4.0:
                print(f"   âœ… Pressure times look realistic (1-4 seconds)")
            else:
                print(f"   âš ï¸� WARNING: Pressure times seem unusual")
        
        clean_rate = test_df['clean_pocket'].sum() / len(test_df) * 100
        print(f"   Clean pocket rate: {clean_rate:.1f}%")
        
        if 10 < clean_rate < 60:
            print(f"   âœ… Clean pocket rate looks realistic (10-60%)")
        else:
            print(f"   âš ï¸� WARNING: Clean pocket rate seems unusual")
    
    print(f"\nğŸ“Š TIMING CATEGORIES DISTRIBUTION:")
    if 'timing_category_detailed' in test_df.columns:
        for cat, count in test_df['timing_category_detailed'].value_counts().items():
            print(f"   {cat}: {count}")
    
    print(f"\nğŸ�¯ VERDICT:")
    if len(test_results) >= 8 and 'pressure_time' in test_df.columns:
        print("   âœ… ALL SYSTEMS GO! Ready to process full dataset.")
        print("   âœ… Run Cell 16 when ready (will take 10-20 minutes)")
    elif len(test_results) >= 5:
        print("   âš ï¸� PARTIAL SUCCESS - some plays failed but core logic works")
        print("   âš ï¸� Can proceed but watch for errors in full run")
    else:
        print("   â�Œ TOO MANY ERRORS - need to debug before full run")
        print("   â�Œ Share error details for debugging")

else:
    print(f"\nâ�Œ NO PLAYS PROCESSED SUCCESSFULLY!")
    print(f"â�Œ Errors encountered: {len(test_errors)}")
    print("\nâ�Œ Error details:")
    for error in test_errors[:5]:
        print(f"   {error}")

if test_errors:
    print(f"\nâš ï¸� Errors/Skips ({len(test_errors)}):")
    for error in test_errors[:3]:
        print(f"   {error}")

print("\n" + "=" * 70)


# =============================================================================
# DEBUG: WHY IS NO PRESSURE BEING DETECTED?
# =============================================================================

print("ğŸ”� DEBUGGING PRESSURE DETECTION")
print("=" * 70)

# Pick one play from the test
test_game_id = 2023112608
test_play_id = 2407

play_df = merged_df[
    (merged_df['game_id'] == test_game_id) & 
    (merged_df['play_id'] == test_play_id)
]

print(f"\nğŸ“Š Play: {test_game_id}-{test_play_id}")
print(f"   Total rows: {len(play_df)}")
print(f"   Unique frames: {play_df['frame_id'].nunique()}")

# Check player roles
print(f"\nğŸ“Š Player roles in this play:")
role_counts = play_df['player_role'].value_counts()
for role, count in role_counts.items():
    print(f"   {role}: {count}")

# Check if we can find QB
qb = play_df[play_df['player_role'] == 'Passer']
print(f"\nğŸ“Š QB found: {len(qb) > 0}")
if len(qb) > 0:
    print(f"   QB frames: {len(qb)}")

# Check available columns
print(f"\nğŸ“Š Columns containing 'name':")
name_cols = [col for col in play_df.columns if 'name' in col.lower()]
for col in name_cols:
    print(f"   {col}")

# Try to find defenders MANUALLY
print(f"\nğŸ“Š Trying different defender identification methods:")

# Method 1: player_role contains "Defensive"
def_method1 = play_df[play_df['player_role'].str.contains('Defensive', na=False)]
print(f"   Method 1 (role contains 'Defensive'): {len(def_method1)} rows")

# Method 2: player_role contains "Pass Rush"
def_method2 = play_df[play_df['player_role'].str.contains('Pass Rush', na=False)]
print(f"   Method 2 (role contains 'Pass Rush'): {len(def_method2)} rows")

# Method 3: player_role == 'Defensive Coverage'
def_method3 = play_df[play_df['player_role'] == 'Defensive Coverage']
print(f"   Method 3 (role == 'Defensive Coverage'): {len(def_method3)} rows")

# Method 4: Exclude offense
offense_roles = ['Passer', 'Targeted Receiver', 'Offensive Line', 'Running Back', 'Tight End', 'Wide Receiver']
def_method4 = play_df[~play_df['player_role'].isin(offense_roles)]
print(f"   Method 4 (exclude offense): {len(def_method4)} rows")

# Calculate pressure with different methods
print(f"\nğŸ“Š Testing pressure calculation:")

if len(qb) > 0 and len(def_method4) > 0:
    # Test with one frame
    test_frame = qb['frame_id'].iloc[0]
    
    qb_frame = qb[qb['frame_id'] == test_frame]
    def_frame = def_method4[def_method4['frame_id'] == test_frame]
    
    if len(qb_frame) > 0 and len(def_frame) > 0:
        qb_x = qb_frame['x'].values[0]
        qb_y = qb_frame['y'].values[0]
        
        distances = np.sqrt(
            (def_frame['x'].values - qb_x)**2 + 
            (def_frame['y'].values - qb_y)**2
        )
        
        min_dist = distances.min()
        print(f"   Frame {test_frame}: Closest defender at {min_dist:.2f} yards")
        print(f"   All distances: {sorted(distances)[:5]}")  # Show 5 closest
        
        if min_dist < 2.0:
            print(f"   âœ… Would detect pressure (< 2.0 yards)")
        else:
            print(f"   â�Œ No pressure detected (threshold: 2.0 yards)")
            print(f"   ğŸ’¡ Try threshold: {min_dist + 1:.1f} yards to catch this")

# Show actual pressure calculation result
print(f"\nğŸ“Š Actual pressure_timeline result:")
pressure_data = calculate_pressure_timeline(play_df, threshold=7.0)

if pressure_data:
    print(f"   âœ… Function returned data")
    print(f"   Pressure time: {pressure_data['pressure_time']}")
    print(f"   Clean pocket: {pressure_data['clean_pocket']}")
    if len(pressure_data['pressure_timeline']) > 0:
        print(f"   Min distance: {pressure_data['pressure_timeline'].min():.2f} yards")
        print(f"   Max distance: {pressure_data['pressure_timeline'].max():.2f} yards")
        print(f"   Avg distance: {pressure_data['pressure_timeline'].mean():.2f} yards")
else:
    print(f"   â�Œ Function returned None")

print("\n" + "=" * 70)


# =============================================================================
# THRESHOLD OPTIMIZATION: FIND THE RIGHT CUTOFF
# =============================================================================

print("ğŸ”¬ ANALYZING DEFENDER PROXIMITY DISTRIBUTION")
print("=" * 70)

# Sample 50 plays for analysis
sample_plays_for_threshold = merged_df.groupby(['game_id', 'play_id']).ngroups
play_keys_sample = merged_df[['game_id', 'play_id']].drop_duplicates().sample(n=50, random_state=42)

all_min_distances = []

print("\nğŸ“Š Calculating minimum distances for 50 sample plays...")

for idx, (_, row) in enumerate(play_keys_sample.iterrows()):
    game_id = row['game_id']
    play_id = row['play_id']
    
    play_df = merged_df[
        (merged_df['game_id'] == game_id) & 
        (merged_df['play_id'] == play_id)
    ]
    
    # Calculate pressure with current method
    pressure_data = calculate_pressure_timeline(play_df, threshold=99.9)  # High threshold to get all data
    
    if pressure_data and len(pressure_data['pressure_timeline']) > 0:
        # Get the MINIMUM distance across all frames for this play
        min_dist_in_play = pressure_data['pressure_timeline'].min()
        all_min_distances.append(min_dist_in_play)

if len(all_min_distances) > 0:
    all_min_distances = np.array(all_min_distances)
    
    print(f"\nğŸ“Š DISTRIBUTION OF MINIMUM DEFENDER DISTANCES:")
    print("=" * 70)
    print(f"   Sample size: {len(all_min_distances)} plays")
    print(f"   Mean: {all_min_distances.mean():.2f} yards")
    print(f"   Median: {np.median(all_min_distances):.2f} yards")
    print(f"   Std Dev: {all_min_distances.std():.2f} yards")
    print(f"   Min: {all_min_distances.min():.2f} yards")
    print(f"   Max: {all_min_distances.max():.2f} yards")
    
    print(f"\nğŸ“Š PERCENTILES:")
    percentiles = [10, 25, 33, 50, 67, 75, 90]
    for p in percentiles:
        val = np.percentile(all_min_distances, p)
        print(f"   {p}th percentile: {val:.2f} yards")
    
    print(f"\nğŸ“Š THRESHOLD SCENARIOS:")
    print("=" * 70)
    
    thresholds_to_test = [4.0, 5.0, 6.0, 7.0, 8.0]
    
    for threshold in thresholds_to_test:
        pressure_rate = (all_min_distances < threshold).sum() / len(all_min_distances) * 100
        print(f"   Threshold {threshold:.1f} yards â†’ {pressure_rate:.1f}% of plays affected")
    
    # Recommend threshold
    print(f"\nğŸ�¯ RECOMMENDATION:")
    print("=" * 70)
    
    # Target: 25-40% of plays should be "constrained"
    target_percentile = 33  # 33rd percentile = ~33% of plays affected
    recommended = np.percentile(all_min_distances, target_percentile)
    
    print(f"   Target: ~33% of plays should face spatial constraint")
    print(f"   Recommended threshold: {recommended:.1f} yards")
    print(f"   This represents the 33rd percentile of minimum distances")
    
    print(f"\nğŸ’¡ INTERPRETATION:")
    print(f"   Plays where defenders get within {recommended:.1f} yards represent")
    print(f"   tighter-than-usual spatial pressure on the QB.")
    print(f"   This affects ~1 in 3 plays - frequent enough to matter,")
    print(f"   rare enough to be meaningful.")
    
    # Visualization
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(all_min_distances, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(recommended, color='red', linestyle='--', linewidth=2, label=f'Recommended: {recommended:.1f}y')
    ax.axvline(all_min_distances.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {all_min_distances.mean():.1f}y')
    ax.axvline(np.median(all_min_distances), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_min_distances):.1f}y')
    
    ax.set_xlabel('Minimum Defender Distance (yards)', fontsize=12)
    ax.set_ylabel('Number of Plays', fontsize=12)
    ax.set_title('Distribution of Closest Defender Proximity', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
else:
    print("â�Œ Could not calculate distances")

print("\n" + "=" * 70)




# =============================================================================
# PROCESS ALL PLAYS
# =============================================================================

print("PROCESSING ALL PLAYS (PRESSURE-CONSTRAINED VERSION)")
print("=" * 70)
print("\nâš ï¸� This may take 10-20 minutes on the full dataset...")
print("   Progress bar will update below.\n")

results = []
errors = 0

# Group by play
play_groups = merged_df.groupby(['game_id', 'play_id'])
total_plays = play_groups.ngroups

print(f"ğŸ“Š Total plays to process: {total_plays:,}\n")

# Process each play with progress bar
for (game_id, play_id), play_df in tqdm(play_groups, total=total_plays, desc="Processing plays"):
    
    try:
        # Get supplementary info for this play
        supp_row = pass_plays[
            (pass_plays['game_id'] == game_id) & 
            (pass_plays['play_id'] == play_id)
        ]
        
        if len(supp_row) == 0:
            continue
        
        supp_row = supp_row.iloc[0]
        
        # Process the play
        result = process_single_play(play_df, supp_row)
        
        if result:
            results.append(result)
            
    except Exception as e:
        errors += 1
        continue

# Convert to DataFrame
results_df = pd.DataFrame(results)

# =============================================================================
# ADD DERIVED COLUMNS
# =============================================================================

# Add on_time boolean for easy filtering
results_df['on_time'] = results_df['timing_category'] == 'on_time'

# Add completion boolean
results_df['completed'] = results_df['pass_result'] == 'C'

# =============================================================================
# SUMMARY (UPDATED WITH PRESSURE DATA)
# =============================================================================

print("\n" + "=" * 70)
print("âœ… PROCESSING COMPLETE (PRESSURE-CONSTRAINED)")
print("=" * 70)

print(f"\nğŸ“Š Results Summary:")
print(f"   Plays processed successfully: {len(results_df):,}")
print(f"   Plays with errors/skipped: {errors:,}")
print(f"   Success rate: {len(results_df)/total_plays*100:.1f}%")

print(f"\nğŸ“Š Data Available for Analysis:")
print(f"   Routes analyzed: {results_df['route_type'].nunique()}")
print(f"   QBs analyzed: {results_df['qb_name'].nunique()}")
print(f"   Coverage types: {results_df['coverage_type'].nunique()}")

print(f"\nğŸ“Š Timing Distribution (Legacy 3-way):")
timing_dist = results_df['timing_category'].value_counts()
for cat, count in timing_dist.items():
    pct = count / len(results_df) * 100
    print(f"   {cat}: {count:,} plays ({pct:.1f}%)")

print(f"\nğŸ“Š Timing Distribution (NEW 6-way with Pressure):")
timing_detailed = results_df['timing_category_detailed'].value_counts()
for cat, count in timing_detailed.items():
    pct = count / len(results_df) * 100
    print(f"   {cat}: {count:,} plays ({pct:.1f}%)")

print(f"\nğŸ“Š Pressure Analysis:")
print(f"   Clean pocket plays: {results_df['clean_pocket'].sum():,}")
print(f"   Pressure-constrained plays: {results_df['pressure_constrained'].sum():,}")
print(f"   Throws under pressure: {results_df['under_pressure'].sum():,}")
if results_df['pressure_time'].notna().sum() > 0:
    print(f"   Average pressure arrival: {results_df['pressure_time'].mean():.2f}s")

print(f"\nğŸ“Š Context:")
print(f"   Play Action Plays: {(results_df['play_action'] == 'Y').sum():,}")
print(f"   Man Coverage Plays: {(results_df['man_zone'] == 'Man').sum():,}")
print(f"   Zone Coverage Plays: {(results_df['man_zone'] == 'Zone').sum():,}")

print(f"\nğŸ�¯ Ready for Pressure-Aware Finding Analysis!")


# =============================================================================
# FINAL VALIDATION: FULL RESULTS CHECK
# =============================================================================

print("ğŸ“Š FINAL RESULTS VALIDATION")
print("=" * 70)

print(f"\nâœ… Total plays processed: {len(results_df):,}")

print(f"\nğŸ“Š PRESSURE DATA COVERAGE:")
print(f"   Plays with pressure data: {results_df['pressure_time'].notna().sum():,}")
print(f"   Clean pocket plays: {results_df['clean_pocket'].sum():,} ({results_df['clean_pocket'].sum()/len(results_df)*100:.1f}%)")
print(f"   Pressure-constrained plays: {results_df['pressure_constrained'].sum():,} ({results_df['pressure_constrained'].sum()/len(results_df)*100:.1f}%)")
print(f"   Throws under pressure: {results_df['under_pressure'].sum():,} ({results_df['under_pressure'].sum()/len(results_df)*100:.1f}%)")

print(f"\nğŸ“Š PRESSURE TIMING:")
pressure_plays = results_df[results_df['pressure_time'].notna()]
if len(pressure_plays) > 0:
    print(f"   Average pressure arrival: {pressure_plays['pressure_time'].mean():.2f}s")
    print(f"   Median pressure arrival: {pressure_plays['pressure_time'].median():.2f}s")
    print(f"   Range: {pressure_plays['pressure_time'].min():.2f}s - {pressure_plays['pressure_time'].max():.2f}s")

print(f"\nğŸ“Š NEW TIMING CATEGORIES (6-way):")
for cat, count in results_df['timing_category_detailed'].value_counts().items():
    pct = count / len(results_df) * 100
    comp_rate = results_df[results_df['timing_category_detailed'] == cat]['completed'].mean() * 100
    print(f"   {cat:20s}: {count:>6,} ({pct:5.1f}%) | Comp: {comp_rate:.1f}%")

print(f"\nğŸ“Š ROUTE-LEVEL PRESSURE IMPACT:")
route_pressure = results_df.groupby('route_type').agg({
    'optimal_time': 'mean',
    'pressure_time': 'mean',
    'pressure_constrained': lambda x: (x.sum() / len(x) * 100)
}).round(2)
route_pressure.columns = ['Avg_Optimal_Time', 'Avg_Pressure_Time', 'Pressure_Rate_%']
route_pressure['Time_Margin'] = route_pressure['Avg_Pressure_Time'] - route_pressure['Avg_Optimal_Time']
route_pressure = route_pressure.sort_values('Time_Margin')

print("\n   Top 5 routes MOST constrained by pressure:")
print(route_pressure.head(5).to_string())

print("\n   Top 5 routes LEAST constrained by pressure:")
print(route_pressure.tail(5).to_string())

print(f"\nğŸ�¯ QUALITY CHECKS:")

# Check 1: Pressure times reasonable
avg_p = results_df['pressure_time'].mean()
if pd.notna(avg_p):
    if 0.5 < avg_p < 5.0:
        print(f"   âœ… Average pressure time ({avg_p:.2f}s) is realistic")
    else:
        print(f"   âš ï¸� Average pressure time ({avg_p:.2f}s) seems unusual")

# Check 2: Clean pocket rate reasonable
clean_rate = results_df['clean_pocket'].sum() / len(results_df) * 100
if 50 < clean_rate < 80:
    print(f"   âœ… Clean pocket rate ({clean_rate:.1f}%) is realistic")
else:
    print(f"   âš ï¸� Clean pocket rate ({clean_rate:.1f}%) seems unusual")

# Check 3: Pressure-constrained rate
pc_rate = results_df['pressure_constrained'].sum() / len(results_df) * 100
if 20 < pc_rate < 50:
    print(f"   âœ… Pressure-constrained rate ({pc_rate:.1f}%) is realistic")
else:
    print(f"   âš ï¸� Pressure-constrained rate ({pc_rate:.1f}%) seems unusual")

print("\n" + "=" * 70)

