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


"""
NFL Big Data Bowl 2026 - Pitch Control Analysis
================================================================
Cell 1: Imports and Setup

This notebook visualizes the space on the field controlled by 
receivers and defensive backs during pass plays using empirical
movement probability distributions derived from NFL tracking data.

Key Features:
- Data-driven movement clouds showing WHERE players are likely to be
- p95 contours showing boundaries containing 95% of movement probability
- Open space detection highlighting high-value offensive targets
- Catch radius expansion (1.5 yards) for realistic reachability
================================================================
"""

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter

# Scientific computing
from scipy import stats
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

# Interactive widgets
from IPython.display import HTML, display, Image, clear_output
import ipywidgets as widgets

# Utilities
import os
import glob
from datetime import datetime
from io import BytesIO
from collections import defaultdict
import warnings
import pickle

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)

# Matplotlib settings
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Enable inline plotting
# %matplotlib inline  # Uncomment in Kaggle notebook

print("=" * 70)
print("NFL BIG DATA BOWL 2026 - PITCH CONTROL ANALYSIS")
print("=" * 70)
print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nâœ“ All libraries imported successfully")
print("\nApproach: Empirical Movement Probability Distributions")
print("  â€¢ Movement clouds derived from actual NFL tracking data")
print("  â€¢ p95 contours show realistic reachability boundaries")
print("  â€¢ 1.5-yard catch radius for physical player reach")
print("=" * 70)


"""
NFL Big Data Bowl 2026 - Pitch Control Analysis
================================================================
Cell 2: Configuration

All tunable parameters in one place.
Uses empirical movement model with catch radius expansion.
================================================================
"""

print("=" * 70)
print("CONFIGURATION")
print("=" * 70)

# ============================================================================
# FIELD PARAMETERS
# ============================================================================

FIELD_LENGTH = 120          # Total length including endzones (yards)
FIELD_WIDTH = 53.3          # Standard NFL width (yards)
ENDZONE_LENGTH = 10         # Each endzone (yards)
PLAYING_FIELD_START = 10    # Start of playing field (yards)
PLAYING_FIELD_END = 110     # End of playing field (yards)

# Hash mark locations (yards from left sideline)
HASH_LEFT = 14.85
HASH_RIGHT = 38.45

# Grid resolution for pitch control calculation
GRID_SPACING = 0.5          # Yards between grid points (finer = better contours)

print("\nField Parameters:")
print(f"  â€¢ Field: {FIELD_LENGTH} Ã— {FIELD_WIDTH} yards")
print(f"  â€¢ Grid spacing: {GRID_SPACING} yards")

# ============================================================================
# CATCH RADIUS - THE KEY PHYSICS PARAMETER
# ============================================================================
# Players occupy space, not dimensionless points.
# Tracking data gives CENTER position (GPS chip on back).
# Players can catch/control balls within their reach radius:
#   - Arm span: ~1 yard
#   - Diving/stretching: +0.5-1 yard
# Validation finding: 1.5 yard radius achieves 90%+ accuracy

CATCH_RADIUS = 1.5          # Yards - validated optimal value

print(f"\nCatch Radius: {CATCH_RADIUS} yards")
print("  â€¢ Players can catch balls within this distance of their body")
print("  â€¢ Expands probability contours for realistic reachability")

# ============================================================================
# EMPIRICAL MODEL PARAMETERS
# ============================================================================

# Velocity bins (yards per second) - how fast player is moving at pass release
VELOCITY_BINS = {
    'stationary': (0, 1.5),      # Standing or barely moving
    'jogging': (1.5, 3.5),       # Light movement
    'running': (3.5, 5.5),       # Moderate speed
    'fast': (5.5, 7.5),          # Fast running
    'sprinting': (7.5, 15.0)     # Full sprint
}

# Time bins (seconds after throw) - how long until ball arrives
TIME_BINS = [0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5]

# Angle to ball bins (degrees) - direction of ball relative to player's motion
# 0Â° = ball directly ahead of player
# 90Â° = ball to the side
# 180Â° = ball behind player
ANGLE_TO_BALL_BINS = {
    'ahead': (0, 45),           # Ball in front - easy to reach
    'oblique': (45, 90),        # Ball to the side-front
    'lateral': (90, 135),       # Ball to the side-back  
    'behind': (135, 180)        # Ball behind - need to turn
}

# Role categories for separate distributions
ROLE_CATEGORIES = {
    'receiver': ['Targeted Receiver', 'Other Route Runner'],
    'defender': ['Defensive Coverage', 'Pass Rush', 'Other Defender']
}

# Distribution grid parameters
DISTRIBUTION_GRID_EXTENT = 15   # Yards in each direction from origin
DISTRIBUTION_GRID_RESOLUTION = 0.5  # Yards per cell in distribution

# Minimum samples required per bin (for statistical reliability)
MIN_SAMPLES_PER_BIN = 30

print("\nEmpirical Model Parameters:")
print(f"  â€¢ Velocity bins: {len(VELOCITY_BINS)}")
print(f"  â€¢ Time bins: {len(TIME_BINS)}")
print(f"  â€¢ Angle bins: {len(ANGLE_TO_BALL_BINS)}")
print(f"  â€¢ Min samples per bin: {MIN_SAMPLES_PER_BIN}")

# ============================================================================
# PITCH CONTROL PARAMETERS
# ============================================================================

# Probability contour settings
P95_THRESHOLD = 0.95            # Contour containing 95% of probability mass

# Control calculation
CONTROL_EPSILON = 0.001         # Prevent division by zero

# Open space detection
OPEN_SPACE_SEPARATION = 3.0     # Yards from nearest defender for "open"
CONTROL_THRESHOLD_OPEN = 0.65   # Control ratio above this = offense favored

# Maximum pass distance for visualization
MAX_PASS_DISTANCE = 50          # Yards - don't calculate beyond this

print("\nPitch Control Parameters:")
print(f"  â€¢ Probability contour: p{int(P95_THRESHOLD*100)}")
print(f"  â€¢ Open space separation: {OPEN_SPACE_SEPARATION} yards")

# ============================================================================
# DATA PARAMETERS
# ============================================================================

FRAME_RATE = 10                 # Frames per second in tracking data

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Colors (NFL field aesthetic)
COLOR_FIELD = '#2d5016'         # Dark grass green
COLOR_ENDZONE = '#1a3d0a'       # Darker green for endzones
COLOR_LINES = 'white'           # Yard lines and markings

# Team colors
COLOR_OFFENSE = 'blue'
COLOR_DEFENSE = 'red'
COLOR_OPEN_SPACE = 'gold'

# Visualization settings
ALPHA_SCALE = 0.8               # Scale factor for density-based alpha
VELOCITY_VECTOR_SCALE = 0.4     # Scale for velocity arrows

print("\nVisualization Parameters:")
print(f"  â€¢ Velocity vector scale: {VELOCITY_VECTOR_SCALE}x")
print(f"  â€¢ Colors: Blue (offense), Red (defense), Gold (open space)")

# ============================================================================
# ANIMATION PARAMETERS
# ============================================================================

DEFAULT_FPS = 5                 # Default frames per second for animations
MAX_ANIMATION_FRAMES = 20       # Maximum frames to render (for performance)

print("\nAnimation Parameters:")
print(f"  â€¢ Default FPS: {DEFAULT_FPS}")
print(f"  â€¢ Max frames: {MAX_ANIMATION_FRAMES}")

print("\n" + "=" * 70)
print("âœ“ CONFIGURATION LOADED")
print("=" * 70)


"""
NFL Big Data Bowl 2026 - Pitch Control Analysis
================================================================
Cell 3: Data Loading

Loads tracking and supplementary data, normalizes coordinates
and directions so all plays run left-to-right with mathematical
direction convention (0Â°=East, 90Â°=North).
================================================================
"""

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR ENVIRONMENT
# ============================================================================

BASE_PATH = '../input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final'
SUPPLEMENTARY_PATH = f'{BASE_PATH}/supplementary_data.csv'
TRAIN_PATH = f'{BASE_PATH}/train'

# ============================================================================
# LOAD SUPPLEMENTARY DATA
# ============================================================================

print("\n[1/4] Loading supplementary data...")
supplementary = pd.read_csv(SUPPLEMENTARY_PATH, low_memory=False)
print(f"  âœ“ Loaded: {len(supplementary):,} plays")

# Show coverage types
if 'team_coverage_type' in supplementary.columns:
    print(f"\n  Coverage types:")
    for coverage, count in supplementary['team_coverage_type'].value_counts().head(6).items():
        print(f"    â€¢ {coverage}: {count:,}")

# Show pass results
if 'pass_result' in supplementary.columns:
    print(f"\n  Pass results:")
    result_map = {'C': 'Complete', 'I': 'Incomplete', 'IN': 'Interception'}
    for result, count in supplementary['pass_result'].value_counts().items():
        name = result_map.get(result, result)
        print(f"    â€¢ {name}: {count:,}")

# ============================================================================
# LOAD TRACKING DATA
# ============================================================================

print(f"\n[2/4] Loading INPUT tracking data...")
input_files = sorted(glob.glob(os.path.join(TRAIN_PATH, 'input_*.csv')))
print(f"  Found {len(input_files)} files")

input_dfs = []
for file in input_files:
    df_week = pd.read_csv(file)
    df_week['data_type'] = 'input'
    input_dfs.append(df_week)

df_input = pd.concat(input_dfs, ignore_index=True)
print(f"  âœ“ Total INPUT rows: {len(df_input):,}")

print(f"\n[3/4] Loading OUTPUT tracking data...")
output_files = sorted(glob.glob(os.path.join(TRAIN_PATH, 'output_*.csv')))
print(f"  Found {len(output_files)} files")

output_dfs = []
for file in output_files:
    df_week = pd.read_csv(file)
    df_week['data_type'] = 'output'
    output_dfs.append(df_week)

df_output = pd.concat(output_dfs, ignore_index=True)
print(f"  âœ“ Total OUTPUT rows: {len(df_output):,}")

# ============================================================================
# MERGE SUPPLEMENTARY DATA
# ============================================================================

print(f"\n[4/4] Merging supplementary data...")

supp_cols = ['game_id', 'play_id']
optional_cols = ['team_coverage_type', 'pass_result', 'offense_formation', 
                 'pass_length', 'down', 'yards_to_go']

for col in optional_cols:
    if col in supplementary.columns:
        supp_cols.append(col)

supp_unique = supplementary[supp_cols].drop_duplicates(subset=['game_id', 'play_id'])

df_input = df_input.merge(supp_unique, on=['game_id', 'play_id'], how='left')
df_output = df_output.merge(supp_unique, on=['game_id', 'play_id'], how='left')

print(f"  âœ“ Merge complete")

# ============================================================================
# NORMALIZE COORDINATES AND DIRECTIONS
# ============================================================================

print(f"\nNormalizing coordinates and directions...")
print(f"  Converting NFL directions to Math convention")
print(f"  Flipping left-facing plays so all plays run leftâ†’right")

def normalize_play_data(df, has_directions=True):
    """
    Normalize coordinates AND directions so offense always plays left to right.
    
    Direction Conventions:
    - NFL: 0Â°=North (upfield), 90Â°=East, 180Â°=South, 270Â°=West
    - Math: 0Â°=East (rightward), 90Â°=North, 180Â°=West, 270Â°=South
    - Conversion: Math = (90 - NFL) % 360
    
    For left-facing plays:
    - Flip x: x_norm = 120 - x
    - Flip y: y_norm = 53.3 - y  
    - Flip direction: dir_norm = (dir_math + 180) % 360
    """
    df = df.copy()
    
    # Determine play direction
    if 'play_direction' in df.columns:
        df['play_direction_left'] = df['play_direction'] == 'left'
    else:
        # For OUTPUT data, get play_direction from INPUT
        play_directions = df_input[['game_id', 'play_id', 'play_direction']].drop_duplicates()
        df = df.merge(play_directions, on=['game_id', 'play_id'], how='left')
        df['play_direction_left'] = df['play_direction'] == 'left'
    
    # Convert directions from NFL to Math convention
    if has_directions and 'dir' in df.columns:
        df['dir_math'] = (90 - df['dir']) % 360
        
        # Normalize directions (flip for left-facing plays)
        df['dir_norm'] = np.where(
            df['play_direction_left'],
            (df['dir_math'] + 180) % 360,
            df['dir_math']
        )
    
    if has_directions and 'o' in df.columns:
        df['o_math'] = (90 - df['o']) % 360
        df['o_norm'] = np.where(
            df['play_direction_left'],
            (df['o_math'] + 180) % 360,
            df['o_math']
        )
    
    # Normalize spatial coordinates
    df['x_norm'] = np.where(df['play_direction_left'], 120 - df['x'], df['x'])
    df['y_norm'] = np.where(df['play_direction_left'], 53.3 - df['y'], df['y'])
    
    # Normalize ball landing location
    if 'ball_land_x' in df.columns:
        df['ball_land_x_norm'] = np.where(
            df['play_direction_left'], 
            120 - df['ball_land_x'], 
            df['ball_land_x']
        )
    
    if 'ball_land_y' in df.columns:
        df['ball_land_y_norm'] = np.where(
            df['play_direction_left'], 
            53.3 - df['ball_land_y'], 
            df['ball_land_y']
        )
    
    return df

# Normalize INPUT (has directions)
df_input = normalize_play_data(df_input, has_directions=True)

# Normalize OUTPUT (no directions, just positions)
df_output = normalize_play_data(df_output, has_directions=False)

print("  âœ“ Normalization complete")
print("  âœ“ All plays: offense leftâ†’right, 0Â°=East, 90Â°=North")

# ============================================================================
# SUMMARY
# ============================================================================

n_plays = df_input.groupby(['game_id', 'play_id']).ngroups

print("\n" + "=" * 70)
print("DATA LOADING COMPLETE")
print("=" * 70)
print(f"\n  â€¢ INPUT rows: {len(df_input):,}")
print(f"  â€¢ OUTPUT rows: {len(df_output):,}")
print(f"  â€¢ Unique plays: {n_plays:,}")
print(f"\n  â€¢ Normalized columns: x_norm, y_norm, dir_norm, ball_land_x_norm, ball_land_y_norm")
print("=" * 70)


"""
NFL Big Data Bowl 2026 - Pitch Control Analysis
================================================================
Cell 4: Data Preparation

Creates:
- Player metadata lookup (side, role, name for each player-play)
- Field coordinate grid for pitch control calculations
- Field visualization function
================================================================
"""

print("=" * 70)
print("DATA PREPARATION")
print("=" * 70)

# ============================================================================
# PLAYER METADATA LOOKUP
# ============================================================================

def create_player_metadata_lookup(df_input):
    """
    Create lookup table for player metadata from INPUT data.
    
    Returns nested dict: {game_id: {play_id: {nfl_id: {side, role, name}}}}
    """
    print("\n[1/3] Creating player metadata lookup...")
    
    player_info = df_input[['game_id', 'play_id', 'nfl_id', 'player_side', 
                            'player_role', 'player_name']].drop_duplicates()
    
    lookup = {}
    for _, row in player_info.iterrows():
        game_id = row['game_id']
        play_id = row['play_id']
        nfl_id = row['nfl_id']
        
        if game_id not in lookup:
            lookup[game_id] = {}
        if play_id not in lookup[game_id]:
            lookup[game_id][play_id] = {}
        
        lookup[game_id][play_id][nfl_id] = {
            'side': row['player_side'],
            'role': row['player_role'],
            'name': row['player_name']
        }
    
    n_combinations = len(player_info)
    print(f"  âœ“ Metadata for {n_combinations:,} player-play combinations")
    
    return lookup

# Create the lookup table
player_metadata_lookup = create_player_metadata_lookup(df_input)

# ============================================================================
# FIELD COORDINATE GRID
# ============================================================================

def create_field_grid(grid_spacing=GRID_SPACING):
    """
    Create a grid of points covering the entire NFL field.
    
    Grid points are placed at the CENTER of each square.
    
    Returns:
        grid_points: Array of shape (N, 2) with [x, y] coordinates
        grid_x: 2D array of x-coordinates (for plotting)
        grid_y: 2D array of y-coordinates (for plotting)
    """
    # Create coordinates at center of each square
    x_coords = np.arange(0 + grid_spacing/2, FIELD_LENGTH, grid_spacing)
    y_coords = np.arange(0 + grid_spacing/2, FIELD_WIDTH, grid_spacing)
    
    # Create meshgrid for visualization
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    
    # Flatten to list of points for calculations
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    return grid_points, grid_x, grid_y

print("\n[2/3] Creating field grid...")
grid_points, grid_x, grid_y = create_field_grid()
print(f"  âœ“ Grid: {grid_x.shape[1]} Ã— {grid_x.shape[0]} = {len(grid_points):,} points")
print(f"  âœ“ Spacing: {GRID_SPACING} yards")

# ============================================================================
# FIELD VISUALIZATION FUNCTION
# ============================================================================

def draw_football_field(ax, show_hash_marks=True, show_numbers=True):
    """
    Draw an NFL football field with proper markings.
    
    Parameters:
        ax: matplotlib axes object
        show_hash_marks: Draw hash marks
        show_numbers: Draw yard numbers
        
    Returns:
        ax: Axes with field drawn
    """
    # Field background
    ax.set_facecolor(COLOR_FIELD)
    
    # Endzones (darker green)
    ax.add_patch(patches.Rectangle(
        (0, 0), ENDZONE_LENGTH, FIELD_WIDTH, 
        facecolor=COLOR_ENDZONE, edgecolor='none', zorder=0
    ))
    ax.add_patch(patches.Rectangle(
        (PLAYING_FIELD_END, 0), ENDZONE_LENGTH, FIELD_WIDTH,
        facecolor=COLOR_ENDZONE, edgecolor='none', zorder=0
    ))
    
    # Boundary lines
    for x in [0, FIELD_LENGTH]:
        ax.plot([x, x], [0, FIELD_WIDTH], color=COLOR_LINES, linewidth=3, zorder=1)
    for y in [0, FIELD_WIDTH]:
        ax.plot([0, FIELD_LENGTH], [y, y], color=COLOR_LINES, linewidth=3, zorder=1)
    
    # Goal lines
    ax.plot([ENDZONE_LENGTH, ENDZONE_LENGTH], [0, FIELD_WIDTH], 
            color=COLOR_LINES, linewidth=3, zorder=1)
    ax.plot([PLAYING_FIELD_END, PLAYING_FIELD_END], [0, FIELD_WIDTH], 
            color=COLOR_LINES, linewidth=3, zorder=1)
    
    # Yard lines every 5 yards
    for x in range(PLAYING_FIELD_START + 5, PLAYING_FIELD_END, 5):
        linewidth = 2 if x == 60 else 1  # Midfield thicker
        ax.plot([x, x], [0, FIELD_WIDTH], color=COLOR_LINES, 
                linewidth=linewidth, alpha=0.7, zorder=1)
    
    # Hash marks
    if show_hash_marks:
        for x in range(PLAYING_FIELD_START + 1, PLAYING_FIELD_END):
            ax.plot([x, x], [HASH_LEFT - 0.3, HASH_LEFT + 0.3], 
                   color=COLOR_LINES, linewidth=1, alpha=0.5, zorder=1)
            ax.plot([x, x], [HASH_RIGHT - 0.3, HASH_RIGHT + 0.3],
                   color=COLOR_LINES, linewidth=1, alpha=0.5, zorder=1)
    
    # Yard numbers
    if show_numbers:
        # Left side (10-40)
        for x in range(20, 60, 10):
            yard_num = x - 10
            ax.text(x, HASH_LEFT - 3, str(yard_num), color=COLOR_LINES, 
                   fontsize=12, ha='center', weight='bold', alpha=0.5, zorder=1)
            ax.text(x, HASH_RIGHT + 3, str(yard_num), color=COLOR_LINES,
                   fontsize=12, ha='center', weight='bold', alpha=0.5, zorder=1)
        
        # Midfield
        ax.text(60, HASH_LEFT - 3, '50', color=COLOR_LINES,
               fontsize=14, ha='center', weight='bold', alpha=0.5, zorder=1)
        ax.text(60, HASH_RIGHT + 3, '50', color=COLOR_LINES,
               fontsize=14, ha='center', weight='bold', alpha=0.5, zorder=1)
        
        # Right side (40-10)
        for x in range(70, 110, 10):
            yard_num = 110 - x
            ax.text(x, HASH_LEFT - 3, str(yard_num), color=COLOR_LINES,
                   fontsize=12, ha='center', weight='bold', alpha=0.5, zorder=1)
            ax.text(x, HASH_RIGHT + 3, str(yard_num), color=COLOR_LINES,
                   fontsize=12, ha='center', weight='bold', alpha=0.5, zorder=1)
    
    # Axis properties
    ax.set_xlim(0, FIELD_LENGTH)
    ax.set_ylim(0, FIELD_WIDTH)
    ax.set_aspect('equal')
    ax.set_xlabel('Field Position (yards)', fontsize=11, color='white')
    ax.set_ylabel('Field Width (yards)', fontsize=11, color='white')
    ax.tick_params(colors='white')
    
    return ax

print("\n[3/3] Field visualization function created")

# ============================================================================
# HELPER: GET PLAYER INFO FROM LOOKUP
# ============================================================================

def get_player_info(game_id, play_id, nfl_id, field='side'):
    """
    Get player info from metadata lookup.
    
    Parameters:
        game_id, play_id, nfl_id: Identifiers
        field: 'side', 'role', or 'name'
        
    Returns:
        str: Requested field value or 'Unknown'
    """
    try:
        return player_metadata_lookup[game_id][play_id][nfl_id][field]
    except KeyError:
        return 'Unknown'

# ============================================================================
# VALIDATION
# ============================================================================

print("\n" + "=" * 70)
print("DATA PREPARATION COMPLETE")
print("=" * 70)
print("\nAvailable objects:")
print("  â€¢ player_metadata_lookup: Player info by game/play/nfl_id")
print("  â€¢ grid_points, grid_x, grid_y: Field grid coordinates")
print("  â€¢ draw_football_field(ax): Field visualization function")
print("  â€¢ get_player_info(game_id, play_id, nfl_id, field): Helper function")
print("=" * 70)


# """
# This cell generates the original ball flight time model. It has been commented out. To run again, un-comment by selecting all and ctrl + /
# NFL Big Data Bowl 2026 - Pitch Control Analysis
# ================================================================
# Cell 5: Ball Flight Time Model

# The dataset doesn't directly provide pass flight time, so we
# calculate it by finding when the targeted receiver reaches the
# ball landing point in OUTPUT tracking data.

# Method:
# 1. Get QB position at release (last INPUT frame)
# 2. Get ball landing position
# 3. Find OUTPUT frame where receiver is closest to ball landing
# 4. Flight time = frames elapsed / frame rate
# ================================================================
# """

# print("=" * 70)
# print("BALL FLIGHT TIME MODEL")
# print("=" * 70)

# # ============================================================================
# # CALCULATE FLIGHT TIMES FROM RECEIVER ARRIVAL
# # ============================================================================

# def calculate_flight_times():
#     """
#     For each play, find when targeted receiver reaches ball landing point.
    
#     Returns DataFrame with actual flight times for all plays.
#     """
#     print("\nCalculating ball flight times from receiver arrival...")
    
#     # Column names
#     x_col = 'x_norm'
#     y_col = 'y_norm'
#     ball_x_col = 'ball_land_x_norm'
#     ball_y_col = 'ball_land_y_norm'
    
#     # Get unique plays with ball landing info
#     play_info = df_input.groupby(['game_id', 'play_id']).first().reset_index()
#     play_info = play_info[['game_id', 'play_id', ball_x_col, ball_y_col]].copy()
#     play_info = play_info.rename(columns={ball_x_col: 'ball_land_x', ball_y_col: 'ball_land_y'})
#     play_info = play_info.dropna(subset=['ball_land_x', 'ball_land_y'])
    
#     # Get QB release position (last INPUT frame)
#     qb_data = df_input[df_input['player_role'] == 'Passer'].copy()
#     qb_last = qb_data.sort_values('frame_id').groupby(['game_id', 'play_id']).last().reset_index()
#     qb_positions = qb_last[['game_id', 'play_id', x_col, y_col]].rename(
#         columns={x_col: 'qb_x', y_col: 'qb_y'}
#     )
    
#     # Get targeted receiver IDs
#     targeted = df_input[df_input['player_role'] == 'Targeted Receiver']
#     targeted = targeted[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
#     targeted = targeted.rename(columns={'nfl_id': 'receiver_id'})
    
#     # Merge all play info
#     plays = play_info.merge(qb_positions, on=['game_id', 'play_id'], how='inner')
#     plays = plays.merge(targeted, on=['game_id', 'play_id'], how='inner')
    
#     print(f"  â€¢ Plays with complete data: {len(plays):,}")
    
#     # Calculate pass distance
#     plays['pass_distance'] = np.sqrt(
#         (plays['ball_land_x'] - plays['qb_x'])**2 + 
#         (plays['ball_land_y'] - plays['qb_y'])**2
#     )
    
#     # For each play, find when receiver reaches ball
#     results = []
    
#     for idx, row in plays.iterrows():
#         game_id = row['game_id']
#         play_id = row['play_id']
#         receiver_id = row['receiver_id']
#         ball_x = row['ball_land_x']
#         ball_y = row['ball_land_y']
        
#         # Get receiver tracking in OUTPUT
#         output_play = df_output[
#             (df_output['game_id'] == game_id) & 
#             (df_output['play_id'] == play_id)
#         ]
        
#         receiver_data = output_play[output_play['nfl_id'] == receiver_id].copy()
        
#         if len(receiver_data) == 0:
#             continue
        
#         # Calculate distance to ball at each frame
#         receiver_data['dist_to_ball'] = np.sqrt(
#             (receiver_data[x_col] - ball_x)**2 + 
#             (receiver_data[y_col] - ball_y)**2
#         )
        
#         # Find frame with minimum distance (catch frame)
#         min_idx = receiver_data['dist_to_ball'].idxmin()
#         catch_frame = receiver_data.loc[min_idx, 'frame_id']
#         min_distance = receiver_data.loc[min_idx, 'dist_to_ball']
        
#         # Calculate frames from first OUTPUT frame to catch
#         first_frame = output_play['frame_id'].min()
#         flight_frames = catch_frame - first_frame
        
#         results.append({
#             'game_id': game_id,
#             'play_id': play_id,
#             'pass_distance': row['pass_distance'],
#             'flight_frames': flight_frames,
#             'flight_time': flight_frames / FRAME_RATE,
#             'receiver_dist_at_catch': min_distance
#         })
        
#         if len(results) % 500 == 0:
#             print(f"    Processed {len(results):,} plays...")
    
#     df_flights = pd.DataFrame(results)
#     print(f"  âœ“ Processed {len(df_flights):,} plays")
    
#     return df_flights

# # Calculate flight times
# df_flight_times = calculate_flight_times()

# # ============================================================================
# # BUILD FLIGHT TIME LOOKUP TABLE
# # ============================================================================

# def build_flight_time_lookup(df_flights):
#     """
#     Build lookup table: {distance: flight_time}
    
#     Returns:
#         flight_time_lookup: dict mapping distance (int) to mean flight time
#         flight_distributions: dict with full statistics per distance
#     """
#     print("\nBuilding flight time lookup...")
    
#     # Filter to good catches (receiver got close to ball)
#     df_valid = df_flights[
#         (df_flights['flight_frames'] > 0) &
#         (df_flights['receiver_dist_at_catch'] < 3.0) &
#         (df_flights['pass_distance'] >= 0) &
#         (df_flights['pass_distance'] <= 60)
#     ].copy()
    
#     print(f"  â€¢ Valid plays: {len(df_valid):,}")
    
#     # Create distance bins
#     df_valid['distance_bin'] = df_valid['pass_distance'].round().astype(int)
    
#     # Build lookup (mean time per distance)
#     flight_time_lookup = df_valid.groupby('distance_bin')['flight_time'].mean().to_dict()
    
#     # Build distributions (full statistics)
#     flight_distributions = {}
    
#     for dist, group in df_valid.groupby('distance_bin'):
#         times = group['flight_time'].values
#         if len(times) >= 3:
#             flight_distributions[dist] = {
#                 'mean': np.mean(times),
#                 'std': np.std(times),
#                 'p10': np.percentile(times, 10),
#                 'p50': np.percentile(times, 50),
#                 'p90': np.percentile(times, 90),
#                 'n': len(times)
#             }
    
#     # Interpolate gaps
#     if len(flight_time_lookup) > 0:
#         all_dists = sorted(flight_time_lookup.keys())
#         for dist in range(min(all_dists), max(all_dists) + 1):
#             if dist not in flight_time_lookup:
#                 # Linear interpolation
#                 lower = max([d for d in all_dists if d < dist], default=None)
#                 upper = min([d for d in all_dists if d > dist], default=None)
#                 if lower and upper:
#                     weight = (dist - lower) / (upper - lower)
#                     flight_time_lookup[dist] = (
#                         flight_time_lookup[lower] + 
#                         weight * (flight_time_lookup[upper] - flight_time_lookup[lower])
#                     )
    
#     print(f"  âœ“ Lookup covers {len(flight_time_lookup)} distance bins")
    
#     return flight_time_lookup, flight_distributions

# # Build lookup tables
# flight_time_lookup, flight_distributions = build_flight_time_lookup(df_flight_times)

# # ============================================================================
# # UNIFIED FLIGHT TIME FUNCTION
# # ============================================================================

# def get_ball_flight_time(game_id, play_id):
#     """
#     Get ball flight time for a specific play.
    
#     Uses lookup table based on pass distance, with physics-based fallback.
    
#     Returns:
#         flight_time: float (seconds)
#         metadata: dict with additional info
#     """
#     play_data = df_input[
#         (df_input['game_id'] == game_id) & 
#         (df_input['play_id'] == play_id)
#     ]
    
#     if len(play_data) == 0:
#         return 1.5, {'source': 'default', 'error': 'no_data'}
    
#     # Get QB position
#     qb_data = play_data[play_data['player_role'] == 'Passer']
#     if len(qb_data) == 0:
#         return 1.5, {'source': 'default', 'error': 'no_qb'}
    
#     qb_row = qb_data.iloc[-1]
#     qb_x = qb_row['x_norm']
#     qb_y = qb_row['y_norm']
    
#     # Get ball landing position
#     ball_x = play_data['ball_land_x_norm'].iloc[0]
#     ball_y = play_data['ball_land_y_norm'].iloc[0]
    
#     if pd.isna(ball_x) or pd.isna(ball_y):
#         return 1.5, {'source': 'default', 'error': 'no_ball_landing'}
    
#     # Calculate distance
#     distance = np.sqrt((ball_x - qb_x)**2 + (ball_y - qb_y)**2)
#     distance_bin = int(round(distance))
    
#     # Look up flight time
#     if distance_bin in flight_time_lookup:
#         flight_time = flight_time_lookup[distance_bin]
#         source = 'lookup'
#     else:
#         # Physics-based fallback
#         if distance < 15:
#             flight_time = distance / 18.0 + 0.25
#         elif distance < 30:
#             flight_time = distance / 20.0 + 0.25
#         else:
#             flight_time = distance / 22.0 + 0.25
#         source = 'physics_fallback'
    
#     metadata = {
#         'source': source,
#         'distance': distance,
#         'qb_position': (qb_x, qb_y),
#         'ball_landing': (ball_x, ball_y)
#     }
    
#     return flight_time, metadata

# # ============================================================================
# # DISPLAY SAMPLE LOOKUP VALUES
# # ============================================================================

# print("\n" + "=" * 70)
# print("FLIGHT TIME LOOKUP TABLE")
# print("=" * 70)
# print(f"\n{'Distance':<10} {'Time (s)':<10} {'Samples':<10}")
# print("-" * 30)

# for dist in [5, 10, 15, 20, 25, 30, 35, 40]:
#     if dist in flight_distributions:
#         d = flight_distributions[dist]
#         print(f"{dist:<10} {d['mean']:<10.2f} {d['n']:<10}")

# print("\n" + "=" * 70)
# print("BALL FLIGHT TIME MODEL COMPLETE")
# print("=" * 70)
# print("\nUsage:")
# print("  flight_time, metadata = get_ball_flight_time(game_id, play_id)")
# print("=" * 70)


# # Run this ONCE to generate the hardcoded lookup:

# def print_lookup_as_code():
#     """Print the flight time lookup as copy-pasteable Python code"""
    
#     print("# Pre-computed flight time lookup table")
#     print("# Generated from calculate_flight_times() on full dataset")
#     print("flight_time_lookup = {")
    
#     for dist in sorted(flight_time_lookup.keys()):
#         time = flight_time_lookup[dist]
#         print(f"    {dist}: {time:.4f},")
    
#     print("}")
    
#     print("\n# Flight time distributions (optional - for detailed analysis)")
#     print("flight_distributions = {")
    
#     for dist in sorted(flight_distributions.keys()):
#         d = flight_distributions[dist]
#         print(f"    {dist}: {{'mean': {d['mean']:.4f}, 'std': {d['std']:.4f}, "
#               f"'p10': {d['p10']:.4f}, 'p50': {d['p50']:.4f}, 'p90': {d['p90']:.4f}, 'n': {d['n']}}},")
    
#     print("}")

# print_lookup_as_code()


# Pre-computed flight time lookup table
# Generated from calculate_flight_times() on full dataset
print("=" * 70)
print("PITCH CONTROL MODEL VALIDATION (PRE-COMPUTED)")
print("=" * 70)

print("""
flight_time_lookup = {
    2: 0.6000,
    3: 0.5000,
    4: 0.2000,
    5: 0.2857,
    6: 0.3583,
    7: 0.3186,
    8: 0.3828,
    9: 0.4134,
    10: 0.4552,
    11: 0.4969,
    12: 0.5308,
    13: 0.5605,
    14: 0.6004,
    15: 0.6422,
    16: 0.6735,
    17: 0.7124,
    18: 0.7533,
    19: 0.7894,
    20: 0.8139,
    21: 0.8612,
    22: 0.8939,
    23: 0.9504,
    24: 0.9830,
    25: 1.0219,
    26: 1.0927,
    27: 1.1284,
    28: 1.1667,
    29: 1.2586,
    30: 1.2786,
    31: 1.4060,
    32: 1.4317,
    33: 1.4931,
    34: 1.5767,
    35: 1.6328,
    36: 1.7344,
    37: 1.7024,
    38: 1.8204,
    39: 1.8987,
    40: 1.9583,
    41: 2.0105,
    42: 2.0820,
    43: 2.1254,
    44: 2.1704,
    45: 2.2382,
    46: 2.2444,
    47: 2.3079,
    48: 2.3848,
    49: 2.4943,
    50: 2.4973,
    51: 2.4786,
    52: 2.5765,
    53: 2.6654,
    54: 2.6875,
    55: 2.7360,
    56: 2.7800,
    57: 2.9067,
    58: 2.9286,
    59: 2.7833,
    60: 3.0000,
}

# Flight time distributions (optional - for detailed analysis)
flight_distributions = {
    5: {'mean': 0.2857, 'std': 0.1245, 'p10': 0.1600, 'p50': 0.3000, 'p90': 0.4400, 'n': 7},
    6: {'mean': 0.3583, 'std': 0.1891, 'p10': 0.1000, 'p50': 0.4000, 'p90': 0.5000, 'n': 12},
    7: {'mean': 0.3186, 'std': 0.1351, 'p10': 0.1000, 'p50': 0.3000, 'p90': 0.5000, 'n': 43},
    8: {'mean': 0.3828, 'std': 0.1538, 'p10': 0.2000, 'p50': 0.4000, 'p90': 0.6000, 'n': 99},
    9: {'mean': 0.4134, 'std': 0.1472, 'p10': 0.2000, 'p50': 0.4000, 'p90': 0.6000, 'n': 201},
    10: {'mean': 0.4552, 'std': 0.1550, 'p10': 0.2000, 'p50': 0.5000, 'p90': 0.6000, 'n': 297},
    11: {'mean': 0.4969, 'std': 0.1576, 'p10': 0.3000, 'p50': 0.5000, 'p90': 0.7000, 'n': 457},
    12: {'mean': 0.5308, 'std': 0.1636, 'p10': 0.3000, 'p50': 0.6000, 'p90': 0.7000, 'n': 577},
    13: {'mean': 0.5605, 'std': 0.1509, 'p10': 0.4000, 'p50': 0.6000, 'p90': 0.7000, 'n': 656},
    14: {'mean': 0.6004, 'std': 0.1612, 'p10': 0.4000, 'p50': 0.6000, 'p90': 0.8000, 'n': 715},
    15: {'mean': 0.6422, 'std': 0.1895, 'p10': 0.4000, 'p50': 0.7000, 'p90': 0.8000, 'n': 780},
    16: {'mean': 0.6735, 'std': 0.1665, 'p10': 0.5000, 'p50': 0.7000, 'p90': 0.8000, 'n': 671},
    17: {'mean': 0.7124, 'std': 0.1669, 'p10': 0.5000, 'p50': 0.7000, 'p90': 0.9000, 'n': 630},
    18: {'mean': 0.7533, 'std': 0.1699, 'p10': 0.6000, 'p50': 0.8000, 'p90': 0.9000, 'n': 636},
    19: {'mean': 0.7894, 'std': 0.1733, 'p10': 0.6000, 'p50': 0.8000, 'p90': 1.0000, 'n': 630},
    20: {'mean': 0.8139, 'std': 0.1851, 'p10': 0.6000, 'p50': 0.8000, 'p90': 1.0000, 'n': 482},
    21: {'mean': 0.8612, 'std': 0.2022, 'p10': 0.6000, 'p50': 0.9000, 'p90': 1.0000, 'n': 498},
    22: {'mean': 0.8939, 'std': 0.2059, 'p10': 0.7000, 'p50': 0.9000, 'p90': 1.1000, 'n': 440},
    23: {'mean': 0.9504, 'std': 0.2134, 'p10': 0.8000, 'p50': 1.0000, 'p90': 1.1000, 'n': 399},
    24: {'mean': 0.9830, 'std': 0.2204, 'p10': 0.7700, 'p50': 1.0000, 'p90': 1.2000, 'n': 358},
    25: {'mean': 1.0219, 'std': 0.2390, 'p10': 0.8000, 'p50': 1.0000, 'p90': 1.3000, 'n': 347},
    26: {'mean': 1.0927, 'std': 0.2373, 'p10': 0.9000, 'p50': 1.1000, 'p90': 1.3000, 'n': 315},
    27: {'mean': 1.1284, 'std': 0.2155, 'p10': 0.9000, 'p50': 1.1500, 'p90': 1.3000, 'n': 268},
    28: {'mean': 1.1667, 'std': 0.2701, 'p10': 0.9000, 'p50': 1.2000, 'p90': 1.4000, 'n': 252},
    29: {'mean': 1.2586, 'std': 0.2042, 'p10': 1.0000, 'p50': 1.3000, 'p90': 1.5000, 'n': 249},
    30: {'mean': 1.2786, 'std': 0.1912, 'p10': 1.1000, 'p50': 1.3000, 'p90': 1.5000, 'n': 192},
    31: {'mean': 1.4060, 'std': 0.2217, 'p10': 1.2000, 'p50': 1.4000, 'p90': 1.7000, 'n': 182},
    32: {'mean': 1.4317, 'std': 0.2538, 'p10': 1.2000, 'p50': 1.4000, 'p90': 1.7000, 'n': 161},
    33: {'mean': 1.4931, 'std': 0.2308, 'p10': 1.3000, 'p50': 1.5000, 'p90': 1.8000, 'n': 160},
    34: {'mean': 1.5767, 'std': 0.2291, 'p10': 1.3000, 'p50': 1.6000, 'p90': 1.9000, 'n': 129},
    35: {'mean': 1.6328, 'std': 0.3413, 'p10': 1.4000, 'p50': 1.6000, 'p90': 1.9400, 'n': 137},
    36: {'mean': 1.7344, 'std': 0.5457, 'p10': 1.5000, 'p50': 1.7000, 'p90': 2.0000, 'n': 122},
    37: {'mean': 1.7024, 'std': 0.2583, 'p10': 1.4200, 'p50': 1.7000, 'p90': 2.0000, 'n': 83},
    38: {'mean': 1.8204, 'std': 0.2781, 'p10': 1.5000, 'p50': 1.8000, 'p90': 2.1000, 'n': 98},
    39: {'mean': 1.8987, 'std': 0.2351, 'p10': 1.6000, 'p50': 1.9000, 'p90': 2.2000, 'n': 78},
    40: {'mean': 1.9583, 'std': 0.2265, 'p10': 1.7000, 'p50': 2.0000, 'p90': 2.2000, 'n': 72},
    41: {'mean': 2.0105, 'std': 0.2261, 'p10': 1.7000, 'p50': 2.0000, 'p90': 2.3000, 'n': 57},
    42: {'mean': 2.0820, 'std': 0.2364, 'p10': 1.8000, 'p50': 2.1000, 'p90': 2.3000, 'n': 61},
    43: {'mean': 2.1254, 'std': 0.1988, 'p10': 1.9000, 'p50': 2.1000, 'p90': 2.4000, 'n': 59},
    44: {'mean': 2.1704, 'std': 0.2370, 'p10': 1.9000, 'p50': 2.2000, 'p90': 2.4700, 'n': 54},
    45: {'mean': 2.2382, 'std': 0.2252, 'p10': 1.9000, 'p50': 2.2000, 'p90': 2.5000, 'n': 55},
    46: {'mean': 2.2444, 'std': 0.2266, 'p10': 1.9000, 'p50': 2.3000, 'p90': 2.5000, 'n': 45},
    47: {'mean': 2.3079, 'std': 0.1869, 'p10': 2.1000, 'p50': 2.3000, 'p90': 2.6000, 'n': 38},
    48: {'mean': 2.3848, 'std': 0.2686, 'p10': 2.1000, 'p50': 2.4000, 'p90': 2.6500, 'n': 46},
    49: {'mean': 2.4943, 'std': 0.1706, 'p10': 2.3000, 'p50': 2.5000, 'p90': 2.7000, 'n': 35},
    50: {'mean': 2.4973, 'std': 0.1732, 'p10': 2.3000, 'p50': 2.5000, 'p90': 2.7000, 'n': 37},
    51: {'mean': 2.4786, 'std': 0.1544, 'p10': 2.2700, 'p50': 2.5000, 'p90': 2.6300, 'n': 28},
    52: {'mean': 2.5765, 'std': 0.2016, 'p10': 2.3300, 'p50': 2.6000, 'p90': 2.8000, 'n': 34},
    53: {'mean': 2.6654, 'std': 0.2056, 'p10': 2.3500, 'p50': 2.7000, 'p90': 2.9000, 'n': 26},
    54: {'mean': 2.6875, 'std': 0.1855, 'p10': 2.5000, 'p50': 2.7000, 'p90': 2.8700, 'n': 24},
    55: {'mean': 2.7360, 'std': 0.1808, 'p10': 2.5000, 'p50': 2.7000, 'p90': 3.0000, 'n': 25},
    56: {'mean': 2.7800, 'std': 0.1904, 'p10': 2.6000, 'p50': 2.8000, 'p90': 3.0000, 'n': 15},
    57: {'mean': 2.9067, 'std': 0.1340, 'p10': 2.7400, 'p50': 2.9000, 'p90': 3.0600, 'n': 15},
    58: {'mean': 2.9286, 'std': 0.1868, 'p10': 2.8000, 'p50': 2.8500, 'p90': 3.1700, 'n': 14},
    59: {'mean': 2.7833, 'std': 0.1572, 'p10': 2.6500, 'p50': 2.7500, 'p90': 2.9500, 'n': 6},
    60: {'mean': 3.0000, 'std': 0.1633, 'p10': 2.8400, 'p50': 3.0000, 'p90': 3.1600, 'n': 3},
}
""")

print("=" * 70)


# Cell 5: Ball Flight Time Model (PRE-COMPUTED)
# ============================================================================
# These lookup tables were pre-computed from the full dataset.
# To regenerate, run calculate_flight_times() and build_flight_time_lookup()
# ============================================================================

print("Loading pre-computed flight time lookup...")

# Pre-computed flight time lookup table
flight_time_lookup = {
    0: 0.2500,
    1: 0.3056,
    2: 0.3611,
    # ... paste all values here ...
    40: 2.1234,
}

flight_distributions = {
    # ... paste distributions here if needed ...
}

def get_ball_flight_time(game_id, play_id):
    """
    Get ball flight time for a specific play.
    Uses pre-computed lookup table based on pass distance.
    """
    play_data = df_input[
        (df_input['game_id'] == game_id) & 
        (df_input['play_id'] == play_id)
    ]
    
    if len(play_data) == 0:
        return 1.5, {'source': 'default', 'error': 'no_data'}
    
    # Get QB position
    qb_data = play_data[play_data['player_role'] == 'Passer']
    if len(qb_data) == 0:
        return 1.5, {'source': 'default', 'error': 'no_qb'}
    
    qb_row = qb_data.iloc[-1]
    qb_x = qb_row['x_norm']
    qb_y = qb_row['y_norm']
    
    # Get ball landing position
    ball_x = play_data['ball_land_x_norm'].iloc[0]
    ball_y = play_data['ball_land_y_norm'].iloc[0]
    
    if pd.isna(ball_x) or pd.isna(ball_y):
        return 1.5, {'source': 'default', 'error': 'no_ball_landing'}
    
    # Calculate distance
    distance = np.sqrt((ball_x - qb_x)**2 + (ball_y - qb_y)**2)
    distance_bin = int(round(distance))
    
    # Look up flight time
    if distance_bin in flight_time_lookup:
        flight_time = flight_time_lookup[distance_bin]
        source = 'lookup'
    else:
        # Physics-based fallback for out-of-range distances
        if distance < 15:
            flight_time = distance / 18.0 + 0.25
        elif distance < 30:
            flight_time = distance / 20.0 + 0.25
        else:
            flight_time = distance / 22.0 + 0.25
        source = 'physics_fallback'
    
    return flight_time, {'source': source, 'distance': distance}

print("âœ“ Flight time lookup ready (pre-computed)")


# """
# This cell has been commented out. To un-comment, select all and ctrl + /
# NFL Big Data Bowl 2026 - Pitch Control Analysis
# ================================================================
# Cell 6.1: Empirical Movement Distributions

# Builds probability distributions showing where players actually
# move during pass plays. This is the core of our data-driven
# approach - using observed behavior rather than physics assumptions.

# Method:
# 1. For each player in OUTPUT, get their state at pass release (INPUT)
# 2. Track where they actually moved frame-by-frame
# 3. Normalize to player-relative coordinates (facing = +X)
# 4. Bin by: initial velocity, angle to ball, time elapsed
# 5. Build 2D kernel density estimates for each bin

# The result is a lookup table of movement "clouds" that predict
# where a player is likely to be given their initial state.
# ================================================================
# """

# print("=" * 70)
# print("EMPIRICAL MOVEMENT DISTRIBUTIONS")
# print("=" * 70)

# # ============================================================================
# # HELPER FUNCTIONS
# # ============================================================================

# def get_velocity_bin(speed):
#     """Map speed to velocity bin name."""
#     for bin_name, (low, high) in VELOCITY_BINS.items():
#         if low <= speed < high:
#             return bin_name
#     return 'sprinting' if speed >= 7.5 else 'stationary'


# def get_time_bin(time_elapsed):
#     """Map elapsed time to nearest time bin."""
#     time_bins_array = np.array(TIME_BINS)
#     idx = np.abs(time_bins_array - time_elapsed).argmin()
#     return TIME_BINS[idx]


# def get_angle_bin(angle):
#     """Map angle to ball to bin name."""
#     for bin_name, (low, high) in ANGLE_TO_BALL_BINS.items():
#         if low <= angle < high:
#             return bin_name
#     return 'behind' if angle >= 135 else 'ahead'


# def get_role_category(player_role):
#     """Map detailed role to category (receiver/defender)."""
#     if player_role in ROLE_CATEGORIES['receiver']:
#         return 'receiver'
#     elif player_role in ROLE_CATEGORIES['defender']:
#         return 'defender'
#     return 'other'


# def calculate_angle_to_point(player_x, player_y, player_dir, target_x, target_y):
#     """
#     Calculate angle from player's facing direction to a target point.
#     Returns 0-180 degrees (0 = target ahead, 180 = target behind).
#     """
#     dx = target_x - player_x
#     dy = target_y - player_y
    
#     if dx == 0 and dy == 0:
#         return 0
    
#     # Direction to target (math convention)
#     target_dir = np.degrees(np.arctan2(dy, dx))
    
#     # Angle difference (0-180, unsigned)
#     angle_diff = abs(((target_dir - player_dir) + 180) % 360 - 180)
    
#     return angle_diff


# def rotate_to_player_frame(dx, dy, player_direction):
#     """
#     Rotate displacement so player's direction becomes +X axis.
    
#     This normalizes all movements to a common reference frame
#     where the player is facing East (rightward).
#     """
#     theta = np.radians(player_direction)
#     cos_t = np.cos(-theta)
#     sin_t = np.sin(-theta)
    
#     dx_norm = dx * cos_t - dy * sin_t
#     dy_norm = dx * sin_t + dy * cos_t
    
#     return dx_norm, dy_norm


# # ============================================================================
# # EXTRACT MOVEMENT SAMPLES
# # ============================================================================

# def extract_movement_samples(verbose=True):
#     """
#     Extract movement samples by linking INPUT states to OUTPUT trajectories.
    
#     For each player:
#     1. Get their state from last frame of INPUT (position, velocity, direction)
#     2. Track their positions through OUTPUT frames
#     3. Compute normalized displacement for each time point
#     4. Record angle to QB/ball at pass release
    
#     Returns DataFrame with all movement samples.
#     """
#     if verbose:
#         print("\nExtracting movement samples from tracking data...")
    
#     all_samples = []
    
#     # Get plays present in both datasets
#     input_plays = set(zip(df_input['game_id'], df_input['play_id']))
#     output_plays = set(zip(df_output['game_id'], df_output['play_id']))
#     common_plays = input_plays & output_plays
    
#     if verbose:
#         print(f"  â€¢ Plays with both INPUT and OUTPUT: {len(common_plays):,}")
    
#     processed = 0
    
#     for game_id, play_id in common_plays:
#         # Get INPUT data
#         input_play = df_input[
#             (df_input['game_id'] == game_id) & 
#             (df_input['play_id'] == play_id)
#         ]
        
#         # Get last INPUT frame for each player (state at pass release)
#         input_final = input_play.loc[
#             input_play.groupby('nfl_id')['frame_id'].idxmax()
#         ]
        
#         # Get OUTPUT data
#         output_play = df_output[
#             (df_output['game_id'] == game_id) & 
#             (df_output['play_id'] == play_id)
#         ]
        
#         if len(output_play) == 0:
#             continue
        
#         output_min_frame = output_play['frame_id'].min()
        
#         # Get QB position at release (for angle calculations)
#         qb_data = input_final[input_final['player_role'] == 'Passer']
#         if len(qb_data) > 0:
#             qb_x = qb_data['x_norm'].iloc[0]
#             qb_y = qb_data['y_norm'].iloc[0]
#         else:
#             qb_x = input_final['x_norm'].mean()
#             qb_y = input_final['y_norm'].mean()
        
#         # Process each player in OUTPUT
#         for nfl_id in output_play['nfl_id'].unique():
#             if pd.isna(nfl_id):
#                 continue
            
#             player_input = input_final[input_final['nfl_id'] == nfl_id]
            
#             if len(player_input) == 0:
#                 continue
            
#             # Initial state from INPUT
#             initial_x = player_input['x_norm'].iloc[0]
#             initial_y = player_input['y_norm'].iloc[0]
#             initial_velocity = player_input['s'].iloc[0]
#             initial_direction = player_input['dir_norm'].iloc[0]
#             player_role = player_input['player_role'].iloc[0]
            
#             # Get role category
#             role_category = get_role_category(player_role)
#             if role_category == 'other':
#                 continue
            
#             # Calculate angle to QB at pass release
#             angle_to_qb = calculate_angle_to_point(
#                 initial_x, initial_y, initial_direction, qb_x, qb_y
#             )
#             angle_bin = get_angle_bin(angle_to_qb)
            
#             # Get player trajectory through OUTPUT
#             player_output = output_play[
#                 output_play['nfl_id'] == nfl_id
#             ].sort_values('frame_id')
            
#             for _, row in player_output.iterrows():
#                 frame_num = row['frame_id']
                
#                 # Time elapsed since throw
#                 frames_elapsed = frame_num - output_min_frame + 1
#                 time_elapsed = frames_elapsed / FRAME_RATE
                
#                 if time_elapsed < 0.2 or time_elapsed > 3.0:
#                     continue
                
#                 # Displacement from initial position
#                 dx = row['x_norm'] - initial_x
#                 dy = row['y_norm'] - initial_y
#                 displacement = np.sqrt(dx**2 + dy**2)
                
#                 # Skip tiny displacements
#                 if displacement < 0.05:
#                     dx_norm, dy_norm = 0, 0
#                 else:
#                     # Rotate to player frame (facing = +X)
#                     dx_norm, dy_norm = rotate_to_player_frame(dx, dy, initial_direction)
                
#                 # Bin assignments
#                 velocity_bin = get_velocity_bin(initial_velocity)
#                 time_bin = get_time_bin(time_elapsed)
                
#                 all_samples.append({
#                     'game_id': game_id,
#                     'play_id': play_id,
#                     'nfl_id': nfl_id,
#                     'role_category': role_category,
#                     'initial_velocity': initial_velocity,
#                     'initial_direction': initial_direction,
#                     'angle_to_qb': angle_to_qb,
#                     'angle_bin': angle_bin,
#                     'time_elapsed': time_elapsed,
#                     'dx_norm': dx_norm,
#                     'dy_norm': dy_norm,
#                     'displacement': displacement,
#                     'velocity_bin': velocity_bin,
#                     'time_bin': time_bin
#                 })
        
#         processed += 1
#         if verbose and processed % 2000 == 0:
#             print(f"    Processed {processed:,}/{len(common_plays):,} plays...")
    
#     df_samples = pd.DataFrame(all_samples)
    
#     if verbose:
#         print(f"\n  âœ“ Total samples: {len(df_samples):,}")
#         print(f"\n  By role:")
#         for role in ['receiver', 'defender']:
#             count = (df_samples['role_category'] == role).sum()
#             print(f"    â€¢ {role}: {count:,}")
#         print(f"\n  By velocity bin:")
#         for vbin in VELOCITY_BINS.keys():
#             count = (df_samples['velocity_bin'] == vbin).sum()
#             print(f"    â€¢ {vbin}: {count:,}")
    
#     return df_samples


# # ============================================================================
# # BUILD 2D DISTRIBUTIONS
# # ============================================================================

# def build_distributions(df_samples, verbose=True):
#     """
#     Build 2D probability distributions for each (velocity, time) combination.
    
#     Uses kernel density estimation to create smooth probability surfaces
#     showing where players are likely to move given their initial state.
    
#     Returns dict: {(velocity_bin, time_bin): distribution_dict}
#     """
#     if verbose:
#         print("\nBuilding 2D probability distributions...")
    
#     distributions = {}
    
#     # Grid for distributions
#     grid_1d = np.arange(
#         -DISTRIBUTION_GRID_EXTENT, 
#         DISTRIBUTION_GRID_EXTENT + DISTRIBUTION_GRID_RESOLUTION, 
#         DISTRIBUTION_GRID_RESOLUTION
#     )
#     dist_grid_x, dist_grid_y = np.meshgrid(grid_1d, grid_1d)
#     positions = np.vstack([dist_grid_x.ravel(), dist_grid_y.ravel()])
    
#     if verbose:
#         print(f"  â€¢ Distribution grid: {len(grid_1d)}Ã—{len(grid_1d)} points")
    
#     # Build distribution for each velocity Ã— time bin
#     for velocity_bin in VELOCITY_BINS.keys():
#         for time_bin in TIME_BINS:
            
#             mask = (
#                 (df_samples['velocity_bin'] == velocity_bin) &
#                 (df_samples['time_bin'] == time_bin)
#             )
#             bin_samples = df_samples[mask]
#             n_samples = len(bin_samples)
            
#             if n_samples < MIN_SAMPLES_PER_BIN:
#                 continue
            
#             # Get displacement data
#             points = bin_samples[['dx_norm', 'dy_norm']].values.T
#             displacements = bin_samples['displacement'].values
            
#             # Statistics
#             mean_dx = points[0].mean()
#             mean_dy = points[1].mean()
#             std_dx = points[0].std()
#             std_dy = points[1].std()
#             p95_reach = np.percentile(displacements, 95)
#             mean_reach = np.mean(displacements)
            
#             # Build KDE
#             try:
#                 kernel = stats.gaussian_kde(points)
#                 density = kernel(positions).reshape(dist_grid_x.shape)
#                 density = density / (density.sum() + 1e-10)
#             except Exception:
#                 # Fallback to simple Gaussian if KDE fails
#                 sigma = max(std_dx, std_dy, 1.0)
#                 density = np.exp(-((dist_grid_x - mean_dx)**2 + (dist_grid_y - mean_dy)**2) / (2 * sigma**2))
#                 density = density / (density.sum() + 1e-10)
            
#             key = (velocity_bin, time_bin)
#             distributions[key] = {
#                 'grid_x': grid_1d.copy(),
#                 'grid_y': grid_1d.copy(),
#                 'density': density,
#                 'n_samples': n_samples,
#                 'mean_dx': mean_dx,
#                 'mean_dy': mean_dy,
#                 'std_dx': std_dx,
#                 'std_dy': std_dy,
#                 'p95_reach': p95_reach,
#                 'mean_reach': mean_reach
#             }
    
#     if verbose:
#         print(f"  âœ“ Built {len(distributions)} distributions")
    
#     return distributions


# # ============================================================================
# # UNIFIED DISTRIBUTION LOOKUP
# # ============================================================================

# def get_player_distribution(speed, flight_time, fallback_velocity='running'):
#     """
#     Get the movement distribution for a player given their state.
    
#     Parameters:
#         speed: Player velocity (yards/second)
#         flight_time: Time until ball arrives (seconds)
#         fallback_velocity: Velocity bin to try if exact match not found
        
#     Returns:
#         tuple: (distribution_dict or None, velocity_bin, time_bin)
#     """
#     vel_bin = get_velocity_bin(speed)
#     time_bin = get_time_bin(flight_time)
    
#     key = (vel_bin, time_bin)
    
#     if key in distributions:
#         return distributions[key], vel_bin, time_bin
    
#     # Try fallback velocity
#     fallback_key = (fallback_velocity, time_bin)
#     if fallback_key in distributions:
#         return distributions[fallback_key], fallback_velocity, time_bin
    
#     # Try nearby time bins
#     for t in TIME_BINS:
#         nearby_key = (vel_bin, t)
#         if nearby_key in distributions:
#             return distributions[nearby_key], vel_bin, t
    
#     return None, vel_bin, time_bin


# # ============================================================================
# # CATCH RADIUS EXPANSION
# # ============================================================================

# def create_disk_kernel(radius, grid_resolution):
#     """
#     Create a disk-shaped kernel for morphological dilation.
#     Represents the physical space a player can control from any point.
#     """
#     radius_cells = int(np.ceil(radius / grid_resolution))
#     size = 2 * radius_cells + 1
    
#     y, x = np.ogrid[-radius_cells:radius_cells+1, -radius_cells:radius_cells+1]
#     dist = np.sqrt((x * grid_resolution)**2 + (y * grid_resolution)**2)
    
#     return (dist <= radius).astype(float)


# def expand_density_with_catch_radius(density, grid_resolution, catch_radius=CATCH_RADIUS):
#     """
#     Expand a player's probability density to account for their physical reach.
    
#     Each probability point represents where the player's CENTER could be.
#     But players can catch balls within catch_radius of their center.
    
#     This dilation treats each probability "point" as a disk,
#     which is physically accurate - players occupy space, not dimensionless points.
#     """
#     if catch_radius <= 0:
#         return density
    
#     kernel = create_disk_kernel(catch_radius, grid_resolution)
#     expanded = ndimage.maximum_filter(density, footprint=kernel)
    
#     return expanded


# # ============================================================================
# # BUILD THE DISTRIBUTIONS
# # ============================================================================

# print("\nStep 1: Extracting movement samples...")
# movement_samples = extract_movement_samples(verbose=True)

# print("\nStep 2: Building distributions...")
# distributions = build_distributions(movement_samples, verbose=True)

# # ============================================================================
# # DISPLAY SUMMARY
# # ============================================================================

# print("\n" + "=" * 70)
# print("DISTRIBUTION SUMMARY")
# print("=" * 70)

# print(f"\n{'Velocity':<12} {'Time':<8} {'Samples':<10} {'Mean Dx':<10} {'Mean Dy':<10} {'P95 Reach':<10}")
# print("-" * 60)

# for vel_bin in ['stationary', 'jogging', 'running', 'fast', 'sprinting']:
#     for time_bin in [0.5, 1.0, 1.5]:
#         key = (vel_bin, time_bin)
#         if key in distributions:
#             d = distributions[key]
#             print(f"{vel_bin:<12} {time_bin:<8.1f} {d['n_samples']:<10,} "
#                   f"{d['mean_dx']:<10.2f} {d['mean_dy']:<10.2f} {d['p95_reach']:<10.2f}")

# print("\n" + "=" * 70)
# print("EMPIRICAL MOVEMENT DISTRIBUTIONS COMPLETE")
# print("=" * 70)
# print("\nUsage:")
# print("  dist, vel_bin, time_bin = get_player_distribution(speed, flight_time)")
# print("  expanded = expand_density_with_catch_radius(density, grid_resolution)")
# print("=" * 70)


# # Visualize Movement Distributions by Velocity (1-second flight time)
# # ====================================================================

# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import numpy as np

# def plot_movement_distributions_by_velocity(distributions, time_bin=1.0, figsize=(16, 4)):
#     """
#     Plot movement distribution clouds for all velocity bins at a fixed time.
#     Shows how faster players have larger, more forward-biased clouds.
#     """
    
#     velocity_bins = ['stationary', 'jogging', 'running', 'fast', 'sprinting']
#     velocity_labels = ['Stationary\n(0-2 yd/s)', 'Jogging\n(2-4 yd/s)', 'Running\n(4-6 yd/s)', 
#                        'Fast\n(6-7.5 yd/s)', 'Sprinting\n(7.5+ yd/s)']
    
#     fig, axes = plt.subplots(1, 5, figsize=figsize)
#     fig.suptitle(f'Empirical Movement Distributions (Ball Flight Time = {time_bin}s)', 
#                  fontsize=14, fontweight='bold', y=1.02)
    
#     # Find global max for consistent color scaling
#     max_density = 0
#     for vel_bin in velocity_bins:
#         key = (vel_bin, time_bin)
#         if key in distributions:
#             max_density = max(max_density, distributions[key]['density'].max())
    
#     for idx, (vel_bin, vel_label) in enumerate(zip(velocity_bins, velocity_labels)):
#         ax = axes[idx]
#         key = (vel_bin, time_bin)
        
#         if key not in distributions:
#             ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
#             ax.set_title(vel_label)
#             continue
        
#         dist = distributions[key]
#         grid_x = dist['grid_x']
#         grid_y = dist['grid_y']
#         density = dist['density']
        
#         # Create meshgrid for plotting
#         X, Y = np.meshgrid(grid_x, grid_y)
        
#         # Plot density as filled contours
#         levels = np.linspace(0, max_density, 20)
#         contour = ax.contourf(X, Y, density, levels=levels, cmap='Blues', alpha=0.8)
        
#         # Add contour lines for P50 and P95
#         # Find density thresholds
#         flat_density = density.flatten()
#         sorted_density = np.sort(flat_density)[::-1]
#         cumsum = np.cumsum(sorted_density)
#         cumsum = cumsum / cumsum[-1]
        
#         # P50 and P95 thresholds
#         p50_idx = np.searchsorted(cumsum, 0.50)
#         p95_idx = np.searchsorted(cumsum, 0.95)
#         p50_threshold = sorted_density[min(p50_idx, len(sorted_density)-1)]
#         p95_threshold = sorted_density[min(p95_idx, len(sorted_density)-1)]
        
#         # Draw contours
#         ax.contour(X, Y, density, levels=[p95_threshold], colors='navy', 
#                    linewidths=1.5, linestyles='--', alpha=0.8)
#         ax.contour(X, Y, density, levels=[p50_threshold], colors='darkblue', 
#                    linewidths=2, linestyles='-', alpha=0.9)
        
#         # Mark player starting position (origin)
#         ax.plot(0, 0, 'ko', markersize=10, markerfacecolor='red', markeredgecolor='black', 
#                 markeredgewidth=2, zorder=10, label='Player')
        
#         # Draw arrow showing facing direction (+X)
#         ax.annotate('', xy=(2, 0), xytext=(0, 0),
#                     arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
#         # Mark mean displacement
#         ax.plot(dist['mean_dx'], dist['mean_dy'], 'w*', markersize=12, 
#                 markeredgecolor='black', markeredgewidth=1, zorder=11)
        
#         # Labels
#         ax.set_title(vel_label, fontsize=11, fontweight='bold')
#         ax.set_xlabel('Forward/Back (yards)')
#         if idx == 0:
#             ax.set_ylabel('Left/Right (yards)')
        
#         # Set equal aspect and limits
#         ax.set_xlim(-12, 12)
#         ax.set_ylim(-8, 8)
#         ax.set_aspect('equal')
#         ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
#         ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
#         ax.grid(True, alpha=0.3)
        
#         # Add stats annotation
#         stats_text = f"n={dist['n_samples']:,}\nP95={dist['p95_reach']:.1f} yd"
#         ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
#                 verticalalignment='top', horizontalalignment='right',
#                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
#     # Add legend
#     from matplotlib.lines import Line2D
#     legend_elements = [
#         Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
#                markeredgecolor='black', markersize=10, label='Player Start'),
#         Line2D([0], [0], marker='*', color='w', markerfacecolor='white', 
#                markeredgecolor='black', markersize=12, label='Mean Position'),
#         Line2D([0], [0], color='darkblue', linewidth=2, linestyle='-', label='50% Contour'),
#         Line2D([0], [0], color='navy', linewidth=1.5, linestyle='--', label='95% Contour'),
#     ]
#     fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
#                bbox_to_anchor=(0.5, -0.08), fontsize=10)
    
#     plt.tight_layout()
#     return fig


# # Generate the visualization
# fig = plot_movement_distributions_by_velocity(distributions, time_bin=1.0)
# plt.savefig('/mnt/user-data/outputs/movement_clouds_1sec.png', dpi=150, bbox_inches='tight', 
#             facecolor='white', edgecolor='none')
# plt.show()

# print("\nâœ“ Saved: movement_clouds_1sec.png")


# # Run this ONCE after your distributions are built
# import pickle
# import numpy as np

# def save_distributions():
#     """Save the computed distributions to files"""
    
#     # Option A: Pickle (saves everything including grid arrays)
#     with open('/kaggle/working/movement_distributions.pkl', 'wb') as f:
#         pickle.dump(distributions, f)
    
#     # Also save movement_samples summary stats if needed elsewhere
#     # (Don't save the full df_samples - it's huge and only needed to BUILD distributions)
    
#     print(f"âœ“ Saved {len(distributions)} distributions to movement_distributions.pkl")
#     print(f"  File size: {os.path.getsize('/kaggle/working/movement_distributions.pkl') / 1024:.1f} KB")

# save_distributions()


import os
print(os.listdir('/kaggle/input/nfl-movement-distributions/'))


# Cell 6.1: Empirical Movement Distributions (PRE-COMPUTED)
# ============================================================================

import pickle
import os
from scipy import ndimage

print("=" * 70)
print("LOADING PRE-COMPUTED MOVEMENT DISTRIBUTIONS")
print("=" * 70)

# ============================================================================
# HELPER FUNCTIONS (still needed)
# ============================================================================

def get_velocity_bin(speed):
    """Map speed to velocity bin name."""
    for bin_name, (low, high) in VELOCITY_BINS.items():
        if low <= speed < high:
            return bin_name
    return 'sprinting' if speed >= 7.5 else 'stationary'


def get_time_bin(time_elapsed):
    """Map elapsed time to nearest time bin."""
    time_bins_array = np.array(TIME_BINS)
    idx = np.abs(time_bins_array - time_elapsed).argmin()
    return TIME_BINS[idx]


def get_player_distribution(speed, flight_time, fallback_velocity='running'):
    """Get the movement distribution for a player given their state."""
    vel_bin = get_velocity_bin(speed)
    time_bin = get_time_bin(flight_time)
    
    key = (vel_bin, time_bin)
    
    if key in distributions:
        return distributions[key], vel_bin, time_bin
    
    fallback_key = (fallback_velocity, time_bin)
    if fallback_key in distributions:
        return distributions[fallback_key], fallback_velocity, time_bin
    
    for t in TIME_BINS:
        nearby_key = (vel_bin, t)
        if nearby_key in distributions:
            return distributions[nearby_key], vel_bin, t
    
    return None, vel_bin, time_bin


def create_disk_kernel(radius, grid_resolution):
    """Create a disk-shaped kernel for morphological dilation."""
    radius_cells = int(np.ceil(radius / grid_resolution))
    size = 2 * radius_cells + 1
    
    y, x = np.ogrid[-radius_cells:radius_cells+1, -radius_cells:radius_cells+1]
    dist = np.sqrt((x * grid_resolution)**2 + (y * grid_resolution)**2)
    
    return (dist <= radius).astype(float)


def expand_density_with_catch_radius(density, grid_resolution, catch_radius=CATCH_RADIUS):
    """Expand probability density to account for player's physical reach."""
    if catch_radius <= 0:
        return density
    
    kernel = create_disk_kernel(catch_radius, grid_resolution)
    expanded = ndimage.maximum_filter(density, footprint=kernel)
    
    return expanded


# ============================================================================
# LOAD PRE-COMPUTED DISTRIBUTIONS
# ============================================================================

# Check multiple locations - INPUT persists, WORKING does not
POSSIBLE_PATHS = [
    '/kaggle/input/nfl-movement-distributions/movement_distributions.pkl',  # <-- Your dataset name here
    '/kaggle/working/movement_distributions.pkl',  # Fallback for first run
]

distributions = None

for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        print(f"\nâœ“ Found distributions at: {path}")
        print(f"  Loading...")
        with open(path, 'rb') as f:
            distributions = pickle.load(f)
        print(f"  âœ“ Loaded {len(distributions)} distributions")
        break

if distributions is None:
    print("\n" + "!" * 70)
    print("âš ï¸�  NO PRE-COMPUTED DISTRIBUTIONS FOUND")
    print("!" * 70)
    print("\nTo fix this, you need to either:")
    print("  1. Run Cell 6.1 (full computation) once, then save_distributions()")
    print("  2. Create a dataset from the output and add it as input")
    print("\nSee notebook comments for detailed instructions.")
    distributions = {}

# ============================================================================
# DISPLAY SUMMARY
# ============================================================================

if len(distributions) > 0:
    print(f"\n{'Velocity':<12} {'Time':<8} {'Samples':<10} {'P95 Reach':<10}")
    print("-" * 45)
    
    for vel_bin in ['stationary', 'jogging', 'running', 'fast', 'sprinting']:
        for time_bin in [0.5, 1.0, 1.5]:
            key = (vel_bin, time_bin)
            if key in distributions:
                d = distributions[key]
                print(f"{vel_bin:<12} {time_bin:<8.1f} {d['n_samples']:<10,} {d['p95_reach']:<10.2f}")

print("\n" + "=" * 70)
print("âœ“ MOVEMENT DISTRIBUTIONS READY")
print("=" * 70)


"""
NFL Big Data Bowl 2026 - Pitch Control Analysis
================================================================
Cell 7: Frame Velocity Calculation

The OUTPUT tracking data doesn't include velocity/direction columns,
so we calculate them from position changes between frames.

For the first OUTPUT frame, we use the INPUT velocities (which
are already normalized to math convention via dir_norm).
================================================================
"""

print("=" * 70)
print("FRAME VELOCITY CALCULATION")
print("=" * 70)

def calculate_frame_velocities(df_output_play, game_id=None, play_id=None):
    """
    Calculate actual velocity and direction for each player at each frame.
    
    Method:
    - For frames 2+: Calculate from position change between frames
    - For frame 1: Use INPUT velocities (already in Math convention)
    
    Parameters:
        df_output_play: OUTPUT data for a single play
        game_id, play_id: Optional identifiers (extracted from data if not provided)
        
    Returns:
        DataFrame with added columns: speed_actual, dir_actual, has_velocity
    """
    df = df_output_play.copy()
    
    # Sort by player and frame
    df = df.sort_values(['nfl_id', 'frame_id'])
    
    # Get game_id and play_id if not provided
    if game_id is None:
        game_id = df['game_id'].iloc[0]
    if play_id is None:
        play_id = df['play_id'].iloc[0]
    
    # Coordinate columns
    x_col = 'x_norm' if 'x_norm' in df.columns else 'x'
    y_col = 'y_norm' if 'y_norm' in df.columns else 'y'
    
    # Initialize new columns
    df['speed_actual'] = 0.0
    df['dir_actual'] = 0.0
    df['has_velocity'] = False
    
    # Calculate frame-to-frame velocities
    for nfl_id, player_frames in df.groupby('nfl_id'):
        if len(player_frames) < 2:
            continue
        
        positions = player_frames[[x_col, y_col, 'frame_id']].values
        indices = player_frames.index.values
        
        for i in range(1, len(positions)):
            current_idx = indices[i]
            
            x_curr, y_curr, frame_curr = positions[i]
            x_prev, y_prev, frame_prev = positions[i-1]
            
            # Displacement
            dx = x_curr - x_prev
            dy = y_curr - y_prev
            distance = np.sqrt(dx**2 + dy**2)
            
            # Time elapsed
            frame_diff = frame_curr - frame_prev
            time_elapsed = frame_diff / FRAME_RATE
            
            if time_elapsed > 0:
                # Speed
                speed = distance / time_elapsed
                
                # Direction (math convention: 0Â°=East, 90Â°=North)
                direction = np.degrees(np.arctan2(dy, dx))
                if direction < 0:
                    direction += 360
                
                df.loc[current_idx, 'speed_actual'] = speed
                df.loc[current_idx, 'dir_actual'] = direction
                df.loc[current_idx, 'has_velocity'] = True
    
    # For first frame of each player, use INPUT velocities
    first_frames = df.groupby('nfl_id')['frame_id'].idxmin()
    
    # Get INPUT velocities
    input_play = df_input[
        (df_input['game_id'] == game_id) & 
        (df_input['play_id'] == play_id)
    ]
    
    if len(input_play) > 0:
        # Get last INPUT frame for each player
        input_last = input_play.sort_values('frame_id').groupby('nfl_id').last()
        
        for idx in first_frames:
            if not df.loc[idx, 'has_velocity']:
                nfl_id = df.loc[idx, 'nfl_id']
                
                if nfl_id in input_last.index:
                    input_row = input_last.loc[nfl_id]
                    
                    speed_input = input_row['s'] if 's' in input_row.index else 0
                    dir_input = input_row['dir_norm'] if 'dir_norm' in input_row.index else 0
                    
                    df.loc[idx, 'speed_actual'] = speed_input
                    df.loc[idx, 'dir_actual'] = dir_input
                    df.loc[idx, 'has_velocity'] = True
    
    return df


def get_input_velocities(game_id, play_id):
    """
    Get velocity and direction from INPUT's last frame for each player.
    
    Returns DataFrame with: nfl_id, x_last, y_last, speed_last, dir_last
    """
    input_play = df_input[
        (df_input['game_id'] == game_id) & 
        (df_input['play_id'] == play_id)
    ]
    
    if len(input_play) == 0:
        return pd.DataFrame(columns=['nfl_id', 'x_last', 'y_last', 'speed_last', 'dir_last'])
    
    # Get last frame for each player
    last_frame = input_play.groupby('nfl_id').last().reset_index()
    
    result = pd.DataFrame({
        'nfl_id': last_frame['nfl_id'],
        'x_last': last_frame['x_norm'],
        'y_last': last_frame['y_norm'],
        'speed_last': last_frame['s'] if 's' in last_frame.columns else 0,
        'dir_last': last_frame['dir_norm'] if 'dir_norm' in last_frame.columns else 0
    })
    
    return result


print("\nâœ“ Frame velocity calculation functions ready")
print("\nFunctions:")
print("  â€¢ calculate_frame_velocities(df_output_play): Add velocity to OUTPUT data")
print("  â€¢ get_input_velocities(game_id, play_id): Get INPUT velocities")
print("\nAll directions in Math convention: 0Â°=East, 90Â°=North")
print("=" * 70)


"""
NFL Big Data Bowl 2026 - Pitch Control Analysis
================================================================
Cell 8: Pitch Control Calculation (FIXED VERSION)

FIXES APPLIED:
1. Catch radius is now applied to density BEFORE accumulating team totals
   - Previously: expanded_density computed but not used for control_ratio
   - Now: control_ratio uses expanded densities for both teams

2. Time bin interpolation for smoother, more accurate predictions
   - Previously: snapped to nearest time bin
   - Now: linear interpolation between neighboring bins

Core pitch control model using empirical movement distributions.

For each player:
1. Get their empirical 2D movement distribution based on velocity/time
2. INTERPOLATE between time bins for accurate reach prediction
3. Transform to field coordinates (rotate + translate)
4. EXPAND by catch radius (players control space, not just their body position)
5. Accumulate EXPANDED densities by team

Pitch control = offense_expanded / (offense_expanded + defense_expanded)

Open space = high offense probability + low defense probability + physical separation
================================================================
"""

print("=" * 70)
print("PITCH CONTROL CALCULATION (FIXED)")
print("=" * 70)
print("\nFixes applied:")
print("  âœ“ Catch radius applied to density BEFORE control ratio calculation")
print("  âœ“ Time interpolation between bins for smooth, accurate predictions")

# ============================================================================
# TIME INTERPOLATION FUNCTIONS
# ============================================================================

def interpolate_distributions(dist1, dist2, weight):
    """
    Linearly interpolate between two probability distributions.
    
    This provides smooth transitions between time bins and more accurate
    reach predictions for flight times that fall between bins.
    
    Parameters:
        dist1: Distribution dict for lower time bin
        dist2: Distribution dict for upper time bin
        weight: Interpolation weight (0.0 = dist1, 1.0 = dist2)
        
    Returns:
        Interpolated distribution dict
    """
    if dist1 is None and dist2 is None:
        return None
    if dist1 is None:
        return dist2
    if dist2 is None:
        return dist1
    
    # Interpolate the 2D density grids
    density_interp = (1 - weight) * dist1['density'] + weight * dist2['density']
    
    # Renormalize to ensure it's still a valid probability distribution
    total = density_interp.sum()
    if total > 0:
        density_interp = density_interp / total
    
    # Interpolate scalar statistics
    return {
        'grid_x': dist1['grid_x'],  # Grids are identical
        'grid_y': dist1['grid_y'],
        'density': density_interp,
        'n_samples': int((1 - weight) * dist1['n_samples'] + weight * dist2['n_samples']),
        'p95_reach': (1 - weight) * dist1['p95_reach'] + weight * dist2['p95_reach'],
        'mean_reach': (1 - weight) * dist1['mean_reach'] + weight * dist2['mean_reach'],
    }


def get_player_distribution_interpolated(speed, flight_time):
    """
    Get movement distribution with time interpolation.
    
    Instead of snapping to the nearest time bin, this interpolates between
    neighboring bins for more accurate reach predictions and smoother
    animation transitions.
    
    Parameters:
        speed: Player speed (yards/second)
        flight_time: Time until ball arrives (seconds)
        
    Returns:
        tuple: (distribution_dict, velocity_bin, effective_time, interpolation_type)
    """
    vel_bin = get_velocity_bin(speed)
    time_bins_array = np.array(TIME_BINS)
    
    # Handle edge cases - clamp to bin range
    if flight_time <= TIME_BINS[0]:
        key = (vel_bin, TIME_BINS[0])
        dist = distributions.get(key)
        if dist is None:
            dist = distributions.get(('running', TIME_BINS[0]))
        return dist, vel_bin, TIME_BINS[0], 'clamped_low'
    
    if flight_time >= TIME_BINS[-1]:
        key = (vel_bin, TIME_BINS[-1])
        dist = distributions.get(key)
        if dist is None:
            dist = distributions.get(('running', TIME_BINS[-1]))
        return dist, vel_bin, TIME_BINS[-1], 'clamped_high'
    
    # Find the two bins that bracket flight_time
    upper_idx = np.searchsorted(time_bins_array, flight_time, side='right')
    lower_idx = upper_idx - 1
    
    t_lower = TIME_BINS[lower_idx]
    t_upper = TIME_BINS[upper_idx]
    
    # Get distributions for both bins
    dist_lower = distributions.get((vel_bin, t_lower))
    dist_upper = distributions.get((vel_bin, t_upper))
    
    # Fallback to 'running' velocity bin if specific bin not available
    if dist_lower is None:
        dist_lower = distributions.get(('running', t_lower))
    if dist_upper is None:
        dist_upper = distributions.get(('running', t_upper))
    
    # Handle missing distributions
    if dist_lower is None and dist_upper is None:
        return None, vel_bin, flight_time, 'none'
    if dist_lower is None:
        return dist_upper, vel_bin, t_upper, 'single_upper'
    if dist_upper is None:
        return dist_lower, vel_bin, t_lower, 'single_lower'
    
    # Calculate interpolation weight
    weight = (flight_time - t_lower) / (t_upper - t_lower)
    
    # Interpolate between the two distributions
    dist_interp = interpolate_distributions(dist_lower, dist_upper, weight)
    
    return dist_interp, vel_bin, flight_time, 'interpolated'


# Keep the original function available for comparison
def get_player_distribution_snapped(speed, flight_time, fallback_velocity='running'):
    """
    Original distribution lookup (snaps to nearest bin).
    Kept for comparison purposes.
    """
    vel_bin = get_velocity_bin(speed)
    time_bin = get_time_bin(flight_time)
    
    key = (vel_bin, time_bin)
    
    if key in distributions:
        return distributions[key], vel_bin, time_bin
    
    fallback_key = (fallback_velocity, time_bin)
    if fallback_key in distributions:
        return distributions[fallback_key], fallback_velocity, time_bin
    
    for t in TIME_BINS:
        nearby_key = (vel_bin, t)
        if nearby_key in distributions:
            return distributions[nearby_key], vel_bin, t
    
    return None, vel_bin, time_bin


# ============================================================================
# TRANSFORM DISTRIBUTION TO FIELD COORDINATES
# ============================================================================

def transform_distribution_to_field(dist, player_x, player_y, player_direction,
                                    field_grid_x, field_grid_y):
    """
    Transform a player's normalized 2D distribution to field coordinates.
    
    The distribution is in player-relative coordinates:
    - Origin = player position
    - +X axis = player's facing direction
    
    We rotate by player_direction and translate to (player_x, player_y).
    
    Parameters:
        dist: Distribution dict with 'grid_x', 'grid_y', 'density'
        player_x, player_y: Player position on field
        player_direction: Player direction (math convention: 0Â°=East)
        field_grid_x, field_grid_y: 2D meshgrids of field coordinates
        
    Returns:
        ndarray: Probability density at each field grid point
    """
    if dist is None or 'density' not in dist:
        return np.zeros_like(field_grid_x)
    
    density = dist['density']
    x_1d = np.array(dist['grid_x'])
    y_1d = np.array(dist['grid_y'])
    
    # Build interpolator for the distribution
    try:
        interpolator = RegularGridInterpolator(
            (y_1d, x_1d), density,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
    except Exception:
        return np.zeros_like(field_grid_x)
    
    # Transform field coordinates to player-relative frame
    dx = field_grid_x - player_x
    dy = field_grid_y - player_y
    
    # Rotate to align with distribution frame (player facing +X in distribution)
    theta = np.radians(player_direction)
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    
    rel_x = dx * cos_t - dy * sin_t
    rel_y = dx * sin_t + dy * cos_t
    
    # Query distribution at transformed coordinates
    points = np.column_stack([rel_y.ravel(), rel_x.ravel()])
    field_density = interpolator(points).reshape(field_grid_x.shape)
    
    return np.maximum(field_density, 0)


def create_fallback_distribution(player_x, player_y, player_speed, player_direction,
                                 flight_time, field_grid_x, field_grid_y):
    """
    Create a simple asymmetric Gaussian when no empirical distribution available.
    Elongated in the direction of movement.
    """
    direction_rad = np.radians(player_direction)
    
    # Expected displacement based on current velocity
    move_dist = player_speed * flight_time * 0.6  # Players don't maintain full speed
    expected_x = player_x + move_dist * np.cos(direction_rad)
    expected_y = player_y + move_dist * np.sin(direction_rad)
    
    # Asymmetric sigmas - more spread forward than lateral
    sigma_forward = max(2.0, player_speed * flight_time * 0.5)
    sigma_lateral = max(1.5, player_speed * flight_time * 0.25)
    
    # Transform to player frame
    dx = field_grid_x - player_x
    dy = field_grid_y - player_y
    
    cos_t = np.cos(-direction_rad)
    sin_t = np.sin(-direction_rad)
    
    rel_x = dx * cos_t - dy * sin_t
    rel_y = dx * sin_t + dy * cos_t
    
    # Shift to expected position
    rel_x_shifted = rel_x - move_dist
    
    # Asymmetric Gaussian
    density = np.exp(
        -(rel_x_shifted**2 / (2 * sigma_forward**2)) 
        -(rel_y**2 / (2 * sigma_lateral**2))
    )
    
    # Normalize
    if density.sum() > 0:
        density = density / density.sum()
    
    return density


# ============================================================================
# P95 THRESHOLD CALCULATION
# ============================================================================

def compute_p95_threshold(density):
    """
    Compute the density threshold containing P95_THRESHOLD of probability mass.
    
    Finds the contour level where integrating all density values above
    that level gives 95% of total probability.
    """
    if density is None or density.sum() == 0:
        return 0
    
    flat = density.flatten()
    sorted_density = np.sort(flat)[::-1]
    cumsum = np.cumsum(sorted_density)
    total = cumsum[-1]
    
    target = P95_THRESHOLD * total
    threshold_idx = np.searchsorted(cumsum, target)
    
    if threshold_idx >= len(sorted_density):
        return 0
    
    return sorted_density[threshold_idx]


# ============================================================================
# OPEN SPACE DETECTION
# ============================================================================

def identify_open_space(grid_x, grid_y, offense_density, defense_density,
                        control_ratio, defense_positions, 
                        separation_threshold=OPEN_SPACE_SEPARATION,
                        control_threshold=CONTROL_THRESHOLD_OPEN):
    """
    Identify open offensive space on the field.
    
    Open space criteria:
    1. High control ratio (offense favored)
    2. Reasonable offense probability (someone going there)
    3. Low defense probability (no defenders likely)
    4. Physical separation from defender BODY positions
    
    Note: The density inputs should already include catch radius expansion,
    so control_ratio accounts for where players can REACH. The separation
    check uses actual body positions for physical spacing.
    """
    # Normalize for thresholds
    off_max = offense_density.max() if offense_density.max() > 0 else 1
    def_max = defense_density.max() if defense_density.max() > 0 else 1
    
    offense_norm = offense_density / off_max
    defense_norm = defense_density / def_max
    
    # Criteria
    high_control = control_ratio > control_threshold
    offense_likely = offense_norm > 0.15
    defense_unlikely = defense_norm < 0.10
    
    # Physical separation from defender BODY positions
    # Note: We add catch radius here because the defender can contest
    # from within their catch radius of their body position
    separation_ok = np.ones_like(grid_x, dtype=bool)
    effective_separation = separation_threshold + CATCH_RADIUS
    
    if len(defense_positions) > 0:
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                point = np.array([grid_x[i, j], grid_y[i, j]])
                distances = np.sqrt(np.sum((defense_positions - point)**2, axis=1))
                separation_ok[i, j] = distances.min() >= effective_separation
    
    return high_control & offense_likely & defense_unlikely & separation_ok


# ============================================================================
# MAIN PITCH CONTROL CALCULATION (FIXED)
# ============================================================================

def calculate_pitch_control(df_frame, game_id, play_id, flight_time,
                            field_grid_x=None, field_grid_y=None,
                            grid_spacing=GRID_SPACING,
                            use_interpolation=True):
    """
    Calculate pitch control for a single frame using empirical distributions.
    
    FIXED VERSION:
    - Uses EXPANDED densities (with catch radius) for control ratio calculation
    - Optionally interpolates between time bins for smoother predictions
    
    Parameters:
        df_frame: Player positions/velocities for this frame
        game_id, play_id: Play identifiers
        flight_time: Time until ball arrives (seconds)
        field_grid_x, field_grid_y: Optional custom grid
        grid_spacing: Grid resolution (used if no custom grid)
        use_interpolation: If True, interpolate between time bins
        
    Returns:
        dict: Comprehensive results including control values, densities, open space
    """
    
    # Coordinate columns
    x_col = 'x_norm' if 'x_norm' in df_frame.columns else 'x'
    y_col = 'y_norm' if 'y_norm' in df_frame.columns else 'y'
    
    # Speed/direction columns
    if 'speed_actual' in df_frame.columns:
        speed_col = 'speed_actual'
        dir_col = 'dir_actual'
    elif 's' in df_frame.columns:
        speed_col = 's'
        dir_col = 'dir_norm' if 'dir_norm' in df_frame.columns else 'dir'
    else:
        speed_col = None
        dir_col = None
    
    # Add metadata if not present
    if 'player_side' not in df_frame.columns:
        metadata = player_metadata_lookup.get(game_id, {}).get(play_id, {})
        df_frame = df_frame.copy()
        df_frame['player_side'] = df_frame['nfl_id'].apply(
            lambda x: metadata.get(x, {}).get('side', 'Unknown')
        )
        df_frame['player_role'] = df_frame['nfl_id'].apply(
            lambda x: metadata.get(x, {}).get('role', 'Unknown')
        )
    
    # Create grid if not provided
    if field_grid_x is None or field_grid_y is None:
        x_min = df_frame[x_col].min() - 15
        x_max = df_frame[x_col].max() + 15
        x_range = (max(0, x_min), min(120, x_max))
        y_range = (0, 53.3)
        
        x_coords = np.arange(x_range[0], x_range[1] + grid_spacing, grid_spacing)
        y_coords = np.arange(y_range[0], y_range[1] + grid_spacing, grid_spacing)
        field_grid_x, field_grid_y = np.meshgrid(x_coords, y_coords)
    
    # Initialize density arrays
    # These will accumulate EXPANDED densities (including catch radius)
    offense_density = np.zeros_like(field_grid_x, dtype=float)
    defense_density = np.zeros_like(field_grid_x, dtype=float)
    
    # Store per-player results
    player_results = []
    
    # Process each player
    for _, player in df_frame.iterrows():
        player_x = player.get(x_col)
        player_y = player.get(y_col)
        
        if pd.isna(player_x) or pd.isna(player_y):
            continue
        
        player_speed = player.get(speed_col, 3.0) if speed_col else 3.0
        player_dir = player.get(dir_col, 0) if dir_col else 0
        player_side = player.get('player_side', 'Unknown')
        player_role = player.get('player_role', 'Unknown')
        
        if pd.isna(player_speed):
            player_speed = 3.0
        if pd.isna(player_dir):
            player_dir = 0
        
        if player_side not in ['Offense', 'Defense']:
            continue
        
        # Get empirical distribution - WITH INTERPOLATION
        if use_interpolation:
            dist, vel_bin, effective_time, interp_type = get_player_distribution_interpolated(
                player_speed, flight_time
            )
            time_bin = effective_time  # For logging
        else:
            dist, vel_bin, time_bin = get_player_distribution_snapped(player_speed, flight_time)
            interp_type = 'snapped'
        
        # Transform to field coordinates
        if dist is not None:
            player_density = transform_distribution_to_field(
                dist, player_x, player_y, player_dir,
                field_grid_x, field_grid_y
            )
        else:
            player_density = create_fallback_distribution(
                player_x, player_y, player_speed, player_dir,
                flight_time, field_grid_x, field_grid_y
            )
        
        # CRITICAL FIX: Expand by catch radius BEFORE accumulation
        # This is where catch radius gets applied to the control calculation
        expanded_density = expand_density_with_catch_radius(
            player_density, grid_spacing, CATCH_RADIUS
        )
        
        # Normalize the EXPANDED density per-player
        # This ensures each player contributes proportionally regardless of spread
        max_density = expanded_density.max()
        if max_density > 0:
            density_normalized = expanded_density / max_density
        else:
            density_normalized = expanded_density
        
        # CRITICAL FIX: Accumulate EXPANDED (catch-radius-applied) density
        if player_side == 'Offense':
            offense_density += density_normalized
        else:
            defense_density += density_normalized
        
        # Store player data (both raw and expanded for visualization)
        p95_threshold = compute_p95_threshold(expanded_density)
        
        player_results.append({
            'nfl_id': player.get('nfl_id'),
            'x': player_x,
            'y': player_y,
            'speed': player_speed,
            'direction': player_dir,
            'side': player_side,
            'role': player_role,
            'vel_bin': vel_bin,
            'time_bin': time_bin,
            'interp_type': interp_type,
            'density': player_density,           # Raw (for debugging)
            'density_expanded': expanded_density, # With catch radius
            'p95_threshold': p95_threshold
        })
    
    # Calculate control ratio from EXPANDED densities
    total_density = offense_density + defense_density + CONTROL_EPSILON
    control_ratio = offense_density / total_density
    control_ratio = np.clip(control_ratio, 0, 1)
    
    # Team-level p95 thresholds (already expanded)
    offense_p95 = compute_p95_threshold(offense_density)
    defense_p95 = compute_p95_threshold(defense_density)
    
    # Get defense BODY positions for open space separation check
    defense_positions = np.array([
        [p['x'], p['y']] for p in player_results if p['side'] == 'Defense'
    ])
    
    # Identify open space
    open_space_mask = identify_open_space(
        field_grid_x, field_grid_y,
        offense_density, defense_density,
        control_ratio, defense_positions
    )
    
    # Statistics
    valid_mask = (offense_density > 1e-6) | (defense_density > 1e-6)
    valid_points = valid_mask.sum()
    offense_controlled = ((control_ratio > 0.5) & valid_mask).sum()
    defense_controlled = ((control_ratio <= 0.5) & valid_mask).sum()
    open_count = open_space_mask.sum()
    
    stats = {
        'flight_time': flight_time,
        'catch_radius': CATCH_RADIUS,
        'total_grid_points': field_grid_x.size,
        'valid_points': int(valid_points),
        'offense_controlled': int(offense_controlled),
        'defense_controlled': int(defense_controlled),
        'open_space_points': int(open_count),
        'offense_pct': offense_controlled / max(valid_points, 1) * 100,
        'defense_pct': defense_controlled / max(valid_points, 1) * 100,
        'open_pct': open_count / max(offense_controlled, 1) * 100,
        'use_interpolation': use_interpolation
    }
    
    return {
        'grid_x': field_grid_x,
        'grid_y': field_grid_y,
        'offense_density': offense_density,
        'defense_density': defense_density,
        'control_ratio': control_ratio,
        'offense_p95': offense_p95,
        'defense_p95': defense_p95,
        'open_space_mask': open_space_mask,
        'player_results': player_results,
        'stats': stats,
        'game_id': game_id,
        'play_id': play_id,
        'flight_time': flight_time
    }


# ============================================================================
# CONVENIENCE FUNCTION FOR SINGLE PLAY
# ============================================================================

def calculate_pitch_control_for_play(game_id, play_id, frame_idx=0,
                                     grid_spacing=GRID_SPACING,
                                     use_interpolation=True):
    """
    Calculate pitch control for a specific play and frame.
    
    This is a convenience wrapper that handles:
    - Getting OUTPUT data
    - Calculating frame velocities
    - Getting flight time
    - Calling the main pitch control function
    
    Parameters:
        game_id, play_id: Play identifiers
        frame_idx: Which OUTPUT frame (0 = first)
        grid_spacing: Grid resolution
        use_interpolation: Use time interpolation for smoother predictions
        
    Returns:
        dict: Pitch control results (or None if data unavailable)
    """
    # Get OUTPUT data
    output_play = df_output[
        (df_output['game_id'] == game_id) & 
        (df_output['play_id'] == play_id)
    ].copy()
    
    if len(output_play) == 0:
        print(f"  âš ï¸� No OUTPUT data for game {game_id}, play {play_id}")
        return None
    
    # Get frame
    frames = sorted(output_play['frame_id'].unique())
    frame_id = frames[min(frame_idx, len(frames) - 1)]
    
    # Calculate velocities
    output_play = calculate_frame_velocities(output_play, game_id, play_id)
    
    # Get frame data
    df_frame = output_play[output_play['frame_id'] == frame_id].copy()
    
    # Get flight time
    flight_time, _ = get_ball_flight_time(game_id, play_id)
    
    # Adjust for elapsed time
    min_frame = output_play['frame_id'].min()
    elapsed = (frame_id - min_frame) / FRAME_RATE
    remaining_time = max(0.2, flight_time - elapsed)
    
    # Calculate pitch control
    result = calculate_pitch_control(
        df_frame, game_id, play_id, remaining_time,
        grid_spacing=grid_spacing,
        use_interpolation=use_interpolation
    )
    
    return result


print("\n" + "=" * 70)
print("âœ“ FIXED PITCH CONTROL FUNCTIONS READY")
print("=" * 70)
print("\nKey changes from original:")
print("  1. Catch radius now applied to team density accumulation")
print("     â†’ control_ratio reflects where players can REACH, not just body position")
print("  2. Time interpolation enabled by default")
print("     â†’ Smoother animations, more accurate reach for intermediate flight times")
print("\nFunctions:")
print("  â€¢ calculate_pitch_control(df_frame, game_id, play_id, flight_time)")
print("  â€¢ calculate_pitch_control_for_play(game_id, play_id, frame_idx=0)")
print("  â€¢ get_player_distribution_interpolated(speed, flight_time)")
print(f"\nParameters:")
print(f"  â€¢ Catch radius: {CATCH_RADIUS} yards (applied to all players)")
print(f"  â€¢ Time interpolation: ON (set use_interpolation=False to disable)")
print("=" * 70)


"""
NFL Big Data Bowl 2026 - Pitch Control Analysis
================================================================
Cell 9: Visualization Functions

Static visualization of pitch control results.

Shows:
- Density shading (blue=offense, red=defense gradient)
- p95 contours showing reachability boundaries
- Open space overlay (gold)
- Player positions with velocity vectors
================================================================
"""

print("=" * 70)
print("VISUALIZATION FUNCTIONS")
print("=" * 70)

# ============================================================================
# MAIN PITCH CONTROL VISUALIZATION
# ============================================================================

def plot_pitch_control(result, ax=None, figsize=(16, 9),
                       show_players=True, show_field=True,
                       show_player_contours=True,
                       show_team_contours=True,
                       show_open_space=True,
                       title=None):
    """
    Visualize pitch control with probability distributions.
    
    Parameters:
        result: Output from calculate_pitch_control
        ax: matplotlib axes (creates new if None)
        figsize: Figure size
        show_players: Plot player positions
        show_field: Draw field markings
        show_player_contours: Show individual p95 contours
        show_team_contours: Show team-level p95 contours
        show_open_space: Overlay gold for open space
        title: Custom title
        
    Returns:
        fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    grid_x = result['grid_x']
    grid_y = result['grid_y']
    control_ratio = result['control_ratio']
    offense_density = result['offense_density']
    defense_density = result['defense_density']
    open_space_mask = result['open_space_mask']
    player_results = result['player_results']
    stats = result['stats']
    
    # Draw field
    if show_field:
        draw_football_field(ax, show_hash_marks=True, show_numbers=True)
    
    extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
    
    # ========================================================================
    # DENSITY SHADING
    # ========================================================================
    
    # Normalize densities
    off_max = offense_density.max() if offense_density.max() > 0 else 1
    def_max = defense_density.max() if defense_density.max() > 0 else 1
    
    offense_norm = offense_density / off_max
    defense_norm = defense_density / def_max
    
    # Create RGBA image based on control ratio
    rgba = np.zeros((*grid_x.shape, 4))
    
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            off_d = offense_norm[i, j]
            def_d = defense_norm[i, j]
            ctrl = control_ratio[i, j]
            total_d = off_d + def_d
            
            # Skip very low density areas
            if total_d < 0.01:
                rgba[i, j] = [0.1, 0.1, 0.1, 0.15]
                continue
            
            # Color based on control ratio
            # Blue (offense) to Red (defense) gradient
            if ctrl >= 0.5:
                intensity = (ctrl - 0.5) * 2
                r = 0.2 * (1 - intensity)
                g = 0.3 * (1 - intensity) + 0.2
                b = 0.5 + 0.5 * intensity
            else:
                intensity = (0.5 - ctrl) * 2
                r = 0.5 + 0.5 * intensity
                g = 0.3 * (1 - intensity) + 0.2
                b = 0.2 * (1 - intensity)
            
            # Alpha based on density
            alpha = min(0.85, 0.15 + 0.7 * np.sqrt(total_d))
            rgba[i, j] = [r, g, b, alpha]
    
    ax.imshow(rgba, extent=extent, origin='lower', aspect='auto',
              interpolation='bilinear', zorder=1)
    
    # ========================================================================
    # OPEN SPACE OVERLAY
    # ========================================================================
    
    if show_open_space and open_space_mask.any():
        gold_rgba = np.zeros((*grid_x.shape, 4))
        gold_rgba[open_space_mask, 0] = 1.0    # R
        gold_rgba[open_space_mask, 1] = 0.84   # G
        gold_rgba[open_space_mask, 2] = 0.0    # B
        gold_rgba[open_space_mask, 3] = 0.5    # A
        
        ax.imshow(gold_rgba, extent=extent, origin='lower', aspect='auto',
                  interpolation='nearest', zorder=2)
    
    # ========================================================================
    # INDIVIDUAL PLAYER p95 CONTOURS
    # ========================================================================
    
    if show_player_contours:
        for p in player_results:
            if p['p95_threshold'] > 0:
                contour_density = p.get('density_expanded', p['density'])
                color = 'royalblue' if p['side'] == 'Offense' else 'indianred'
                try:
                    ax.contour(grid_x, grid_y, contour_density,
                              levels=[p['p95_threshold']],
                              colors=[color], linewidths=1.0,
                              linestyles=':', alpha=0.6, zorder=3)
                except:
                    pass
    
    # ========================================================================
    # TEAM-LEVEL p95 CONTOURS
    # ========================================================================
    
    if show_team_contours:
        # Offense boundary
        if result['offense_p95'] > 0:
            try:
                ax.contour(grid_x, grid_y, result['offense_expanded'],
                          levels=[result['offense_p95']],
                          colors=['blue'], linewidths=2.5,
                          linestyles='solid', alpha=0.9, zorder=4)
            except:
                pass
        
        # Defense boundary
        if result['defense_p95'] > 0:
            try:
                ax.contour(grid_x, grid_y, result['defense_expanded'],
                          levels=[result['defense_p95']],
                          colors=['red'], linewidths=2.5,
                          linestyles='--', alpha=0.9, zorder=4)
            except:
                pass
    
    # ========================================================================
    # PLAYERS
    # ========================================================================
    
    if show_players:
        for p in player_results:
            if p['side'] == 'Offense':
                color = 'blue'
                marker = 'o'
            elif p['side'] == 'Defense':
                color = 'red'
                marker = 's'
            else:
                continue
            
            ax.scatter(p['x'], p['y'], c=color, s=180, marker=marker,
                      edgecolors='white', linewidths=2.5, zorder=10)
            
            # Velocity vector
            if p['speed'] > 0.5:
                dx = p['speed'] * VELOCITY_VECTOR_SCALE * np.cos(np.radians(p['direction']))
                dy = p['speed'] * VELOCITY_VECTOR_SCALE * np.sin(np.radians(p['direction']))
                ax.arrow(p['x'], p['y'], dx, dy,
                        head_width=0.8, head_length=0.4,
                        fc='yellow', ec='black', linewidth=1,
                        zorder=9, alpha=0.9)
    
    # ========================================================================
    # LABELS AND LEGEND
    # ========================================================================
    
    ax.set_xlabel('Field Position (yards)', fontsize=11)
    ax.set_ylabel('Field Width (yards)', fontsize=11)
    
    if title is None:
        title = f"Pitch Control - Game {result['game_id']}, Play {result['play_id']}"
        title += f"\nFlight: {stats['flight_time']:.2f}s | Catch radius: {stats['catch_radius']:.1f}yd"
        title += f" | Off: {stats['offense_pct']:.0f}% | Def: {stats['defense_pct']:.0f}%"
        title += f" | Open: {stats['open_pct']:.0f}%"
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.set_xlim(grid_x.min(), grid_x.max())
    ax.set_ylim(grid_y.min(), grid_y.max())
    ax.set_aspect('equal')
    
    # Legend
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Offense Control'),
        Patch(facecolor='red', alpha=0.7, label='Defense Control'),
        Patch(facecolor='gold', alpha=0.5, label='Open Space'),
        Line2D([0], [0], color='blue', linewidth=2.5, label='Offense p95'),
        Line2D([0], [0], color='red', linewidth=2.5, linestyle='--', label='Defense p95'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)
    
    return fig, ax


# ============================================================================
# CONVENIENCE FUNCTION FOR VISUALIZING A PLAY
# ============================================================================

def visualize_play(game_id, play_id, frame_idx=0, figsize=(16, 9), 
                   grid_spacing=GRID_SPACING, save_path=None):
    """
    One-step visualization of pitch control for a play.
    
    Parameters:
        game_id, play_id: Play identifiers
        frame_idx: Which OUTPUT frame (0 = first)
        figsize: Figure size
        grid_spacing: Grid resolution
        save_path: Optional path to save figure
        
    Returns:
        fig, result
    """
    print(f"\n{'='*60}")
    print(f"PITCH CONTROL: Game {game_id}, Play {play_id}")
    print(f"{'='*60}")
    
    # Calculate pitch control
    result = calculate_pitch_control_for_play(game_id, play_id, frame_idx, grid_spacing)
    
    if result is None:
        return None, None
    
    stats = result['stats']
    print(f"\n  â€¢ Flight time: {stats['flight_time']:.2f}s")
    print(f"  â€¢ Catch radius: {stats['catch_radius']:.1f} yards")
    print(f"  â€¢ Offense control: {stats['offense_pct']:.1f}%")
    print(f"  â€¢ Defense control: {stats['defense_pct']:.1f}%")
    print(f"  â€¢ Open space: {stats['open_pct']:.1f}%")
    
    # Create visualization
    fig, ax = plot_pitch_control(result, figsize=figsize)
    fig.patch.set_facecolor('#1a1a1a')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
        print(f"\n  âœ“ Saved to {save_path}")
    
    print(f"\n{'='*60}")
    
    return fig, result


# ============================================================================
# BALL TRAJECTORY OVERLAY
# ============================================================================

def add_ball_trajectory(ax, qb_position, ball_landing, current_position=None,
                        show_qb=True, show_target=True):
    """
    Add ball trajectory visualization to an existing plot.
    
    Parameters:
        ax: matplotlib axes
        qb_position: (x, y) of QB at release
        ball_landing: (x, y) of ball landing point
        current_position: (x, y) of ball at current time (optional)
        show_qb: Show QB marker
        show_target: Show target marker
    """
    qb_x, qb_y = qb_position
    ball_x, ball_y = ball_landing
    
    # Trajectory line
    if current_position:
        curr_x, curr_y = current_position
        ax.plot([qb_x, curr_x], [qb_y, curr_y],
               'yellow', linestyle='--', linewidth=2, alpha=0.7, zorder=11)
        ax.scatter([curr_x], [curr_y], c='yellow', s=200,
                  marker='o', edgecolors='black', linewidths=2, zorder=12)
    else:
        ax.plot([qb_x, ball_x], [qb_y, ball_y],
               'yellow', linestyle='--', linewidth=2, alpha=0.7, zorder=11)
    
    # Target marker
    if show_target:
        ax.scatter([ball_x], [ball_y], c='yellow', s=150, marker='*',
                  edgecolors='black', linewidths=1.5, alpha=0.8, zorder=11)
    
    # QB marker
    if show_qb:
        ax.scatter([qb_x], [qb_y], c='navy', s=200, marker='s',
                  edgecolors='gold', linewidths=2, zorder=11)


print("\nâœ“ Visualization functions ready")
print("\nFunctions:")
print("  â€¢ plot_pitch_control(result): Full visualization of pitch control")
print("  â€¢ visualize_play(game_id, play_id): One-step visualization")
print("  â€¢ add_ball_trajectory(ax, qb_pos, ball_landing): Add ball path overlay")
print("=" * 70)


visualize_play(2023110510, 3666)


"""
NFL Big Data Bowl 2026 - Pitch Control Analysis
================================================================
Cell 10: Animation Functions

Creates animated visualizations of pitch control through a play.
Shows how spatial control evolves as the ball is in the air.

Features:
- Frame-by-frame pitch control
- Ball trajectory visualization
- Play/pause/navigation controls
- Progress tracking
================================================================
"""

print("=" * 70)
print("ANIMATION FUNCTIONS")
print("=" * 70)

# ============================================================================
# CREATE ANIMATION FRAMES
# ============================================================================

def create_animation(game_id, play_id, fps=DEFAULT_FPS, max_frames=MAX_ANIMATION_FRAMES,
                     grid_spacing=GRID_SPACING, save_path=None):
    """
    Create pitch control animation for a play.
    
    Parameters:
        game_id, play_id: Play identifiers
        fps: Frames per second
        max_frames: Maximum frames to render
        grid_spacing: Grid resolution
        save_path: Optional path to save GIF
        
    Returns:
        frame_images: List of frame image bytes
        metadata: Animation metadata dict
    """
    print(f"\n{'='*60}")
    print(f"CREATING ANIMATION: Game {game_id}, Play {play_id}")
    print(f"{'='*60}")
    
    # Get data
    output_play = df_output[
        (df_output['game_id'] == game_id) & 
        (df_output['play_id'] == play_id)
    ].copy()
    
    input_play = df_input[
        (df_input['game_id'] == game_id) & 
        (df_input['play_id'] == play_id)
    ]
    
    if len(output_play) == 0:
        print("  âš ï¸� No OUTPUT data")
        return None, None
    
    # Get QB position
    qb_data = input_play[input_play['player_role'] == 'Passer']
    if len(qb_data) > 0:
        qb_data = qb_data.iloc[-1]
        qb_x = qb_data['x_norm']
        qb_y = qb_data['y_norm']
    else:
        qb_x, qb_y = 25, 26.65
    
    # Ball landing
    ball_x = input_play['ball_land_x_norm'].iloc[0]
    ball_y = input_play['ball_land_y_norm'].iloc[0]
    
    # Flight time
    flight_time, flight_meta = get_ball_flight_time(game_id, play_id)
    if flight_time is None:
        flight_time = 1.5
    
    print(f"  â€¢ QB: ({qb_x:.1f}, {qb_y:.1f})")
    print(f"  â€¢ Ball landing: ({ball_x:.1f}, {ball_y:.1f})")
    print(f"  â€¢ Flight time: {flight_time:.2f}s")
    
    # Calculate frame velocities
    output_play = calculate_frame_velocities(output_play, game_id, play_id)
    
    # Get frames
    frames = sorted(output_play['frame_id'].unique())
    if len(frames) > max_frames:
        step = len(frames) // max_frames
        frames = frames[::step]
    
    print(f"  â€¢ Frames: {len(frames)}")
    
    # Set up grid
    x_col = 'x_norm'
    y_col = 'y_norm'
    
    x_min = output_play[x_col].min() - 15
    x_max = output_play[x_col].max() + 15
    x_range = (max(0, x_min), min(120, x_max))
    y_range = (0, 53.3)
    
    x_coords = np.arange(x_range[0], x_range[1] + grid_spacing, grid_spacing)
    y_coords = np.arange(y_range[0], y_range[1] + grid_spacing, grid_spacing)
    field_grid_x, field_grid_y = np.meshgrid(x_coords, y_coords)
    
    # Process frames
    print(f"\n  Processing frames...")
    frame_images = []
    min_frame = output_play['frame_id'].min()
    
    for idx, frame_id in enumerate(frames):
        df_frame = output_play[output_play['frame_id'] == frame_id].copy()
        
        # Calculate time remaining
        elapsed = (frame_id - min_frame) / FRAME_RATE
        remaining_time = max(0.2, flight_time - elapsed)
        
        # Calculate pitch control
        result = calculate_pitch_control(
            df_frame, game_id, play_id, remaining_time,
            field_grid_x, field_grid_y, grid_spacing
        )
        
        # Create figure
        fig, ax = plot_pitch_control(result, figsize=(16, 9),
                                     show_player_contours=True,
                                     show_team_contours=True)
        
        # Add ball trajectory
        progress = min(1.0, elapsed / flight_time) if flight_time > 0 else 1.0
        ball_curr_x = qb_x + (ball_x - qb_x) * progress
        ball_curr_y = qb_y + (ball_y - qb_y) * progress
        
        add_ball_trajectory(
            ax, (qb_x, qb_y), (ball_x, ball_y), 
            current_position=(ball_curr_x, ball_curr_y)
        )
        
        # Frame info overlay
        stats = result['stats']
        info_text = (f"Frame {idx+1}/{len(frames)} | "
                    f"t = {remaining_time:.2f}s remaining\n"
                    f"Off: {stats['offense_pct']:.0f}% | "
                    f"Def: {stats['defense_pct']:.0f}% | "
                    f"Open: {stats['open_pct']:.0f}%")
        
        ax.text(0.02, 0.02, info_text,
               transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        fig.patch.set_facecolor('#1a1a1a')
        
        # Save frame
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                   facecolor='#1a1a1a')
        buf.seek(0)
        frame_images.append(buf.getvalue())
        plt.close(fig)
        
        if (idx + 1) % 5 == 0 or idx == len(frames) - 1:
            print(f"    Frame {idx+1}/{len(frames)}: t={remaining_time:.2f}s")
    
    # Save as GIF if requested
    if save_path and len(frame_images) > 0:
        print(f"\n  Saving to {save_path}...")
        try:
            from PIL import Image as PILImage
            pil_images = [PILImage.open(BytesIO(img)) for img in frame_images]
            pil_images[0].save(
                save_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=1000 // fps,
                loop=0
            )
            print(f"  âœ“ Saved!")
        except Exception as e:
            print(f"  âš ï¸� Could not save GIF: {e}")
    
    metadata = {
        'game_id': game_id,
        'play_id': play_id,
        'num_frames': len(frame_images),
        'flight_time': flight_time,
        'pass_distance': flight_meta.get('distance'),
        'catch_radius': CATCH_RADIUS,
        'fps': fps
    }
    
    print(f"\n{'='*60}")
    print(f"âœ“ ANIMATION COMPLETE ({len(frame_images)} frames)")
    print(f"{'='*60}")
    
    return frame_images, metadata


# ============================================================================
# INTERACTIVE ANIMATION PLAYER
# ============================================================================

def display_animation_player(frame_images, metadata):
    """
    Display animation with interactive play/pause controls.
    
    Parameters:
        frame_images: List of frame image bytes
        metadata: Animation metadata dict
    """
    if frame_images is None or len(frame_images) == 0:
        print("No frames to display")
        return
    
    import time
    
    # State
    state = {
        'current_frame': 0,
        'playing': False,
        'speed': 1.0
    }
    
    # Widgets
    image_widget = widgets.Image(
        value=frame_images[0],
        format='png',
        width=900,
        height=500
    )
    
    frame_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(frame_images) - 1,
        step=1,
        description='Frame:',
        continuous_update=False,
        layout=widgets.Layout(width='400px')
    )
    
    frame_label = widgets.HTML(
        value=f"<b>Frame 1 / {len(frame_images)}</b>"
    )
    
    play_button = widgets.ToggleButton(
        value=False,
        description='â–¶ Play',
        button_style='success',
        layout=widgets.Layout(width='100px')
    )
    
    speed_dropdown = widgets.Dropdown(
        options=[('0.5x', 0.5), ('1x', 1.0), ('1.5x', 1.5), ('2x', 2.0)],
        value=1.0,
        description='Speed:',
        layout=widgets.Layout(width='120px')
    )
    
    first_button = widgets.Button(description='â�®', layout=widgets.Layout(width='50px'))
    prev_button = widgets.Button(description='â—€', layout=widgets.Layout(width='50px'))
    next_button = widgets.Button(description='â–¶', layout=widgets.Layout(width='50px'))
    last_button = widgets.Button(description='â�­', layout=widgets.Layout(width='50px'))
    
    # Update functions
    def update_display():
        image_widget.value = frame_images[state['current_frame']]
        frame_label.value = f"<b>Frame {state['current_frame'] + 1} / {len(frame_images)}</b>"
        frame_slider.value = state['current_frame']
    
    def on_slider_change(change):
        if not state['playing']:
            state['current_frame'] = change['new']
            update_display()
    
    def on_play_toggle(change):
        state['playing'] = change['new']
        if state['playing']:
            play_button.description = 'â�¸ Pause'
            play_button.button_style = 'warning'
            play_animation()
        else:
            play_button.description = 'â–¶ Play'
            play_button.button_style = 'success'
    
    def play_animation():
        base_delay = 1.0 / metadata.get('fps', DEFAULT_FPS)
        
        while state['playing']:
            state['current_frame'] = (state['current_frame'] + 1) % len(frame_images)
            update_display()
            
            delay = base_delay / state['speed']
            time.sleep(delay)
            
            if state['current_frame'] == 0:
                state['playing'] = False
                play_button.value = False
                break
    
    def on_speed_change(change):
        state['speed'] = change['new']
    
    def on_first(b):
        state['current_frame'] = 0
        update_display()
    
    def on_prev(b):
        state['current_frame'] = max(0, state['current_frame'] - 1)
        update_display()
    
    def on_next(b):
        state['current_frame'] = min(len(frame_images) - 1, state['current_frame'] + 1)
        update_display()
    
    def on_last(b):
        state['current_frame'] = len(frame_images) - 1
        update_display()
    
    # Connect callbacks
    frame_slider.observe(on_slider_change, names='value')
    play_button.observe(on_play_toggle, names='value')
    speed_dropdown.observe(on_speed_change, names='value')
    first_button.on_click(on_first)
    prev_button.on_click(on_prev)
    next_button.on_click(on_next)
    last_button.on_click(on_last)
    
    # Layout
    nav_controls = widgets.HBox([
        first_button, prev_button, play_button, next_button, last_button, speed_dropdown
    ])
    
    progress_row = widgets.HBox([frame_slider, frame_label])
    
    # Title
    title_html = f"""
    <h3>ğŸ�ˆ Pitch Control Animation</h3>
    <p>Game {metadata['game_id']}, Play {metadata['play_id']} | 
    {metadata['num_frames']} frames | 
    Flight: {metadata.get('flight_time', 'N/A'):.2f}s | 
    Catch radius: {metadata.get('catch_radius', CATCH_RADIUS)} yd</p>
    """
    
    ui = widgets.VBox([
        widgets.HTML(title_html),
        image_widget,
        progress_row,
        nav_controls
    ])
    
    display(ui)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def animate_play(game_id, play_id, fps=DEFAULT_FPS, max_frames=MAX_ANIMATION_FRAMES,
                 grid_spacing=GRID_SPACING, save_path=None):
    """
    One-step function to create and display animation.
    
    Parameters:
        game_id, play_id: Play identifiers
        fps: Frames per second
        max_frames: Maximum frames
        grid_spacing: Grid resolution
        save_path: Optional save path for GIF
        
    Returns:
        frame_images, metadata
    """
    frame_images, metadata = create_animation(
        game_id, play_id, fps, max_frames, grid_spacing, save_path
    )
    
    if frame_images:
        display_animation_player(frame_images, metadata)
    
    return frame_images, metadata


print("\nâœ“ Animation functions ready")
print("\nFunctions:")
print("  â€¢ create_animation(game_id, play_id): Generate frame images")
print("  â€¢ display_animation_player(frames, metadata): Interactive player")
print("  â€¢ animate_play(game_id, play_id): One-step animation")
print(f"\nSettings:")
print(f"  â€¢ Default FPS: {DEFAULT_FPS}")
print(f"  â€¢ Max frames: {MAX_ANIMATION_FRAMES}")
print("=" * 70)


# Generate and Save Pitch Control GIF
# ============================================================================

from IPython.display import Image, display
from PIL import Image as PILImage
from io import BytesIO
import os

# ============================================================================
# CHOOSE YOUR PLAY HERE
# ============================================================================
GAME_ID = 2023110510
PLAY_ID = 3666
# ============================================================================

# Use Kaggle working directory (downloadable after notebook runs)
save_path = f"/kaggle/working/pitch_control_{GAME_ID}_{PLAY_ID}.gif"

# Generate the animation frames
frame_images, metadata = create_animation(
    game_id=GAME_ID,
    play_id=PLAY_ID,
    fps=5,
    max_frames=20,
    save_path=None  # Don't use built-in save, we'll do it manually
)

# Save GIF manually
if frame_images and len(frame_images) > 0:
    print(f"\nSaving GIF to {save_path}...")
    
    # Convert bytes to PIL images
    pil_images = [PILImage.open(BytesIO(img)) for img in frame_images]
    
    # Save as GIF
    pil_images[0].save(
        save_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=200,  # milliseconds per frame (200ms = 5fps)
        loop=0
    )
    
    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"âœ“ Saved! ({file_size:.2f} MB)")
    
    # Display in notebook
    print("\n" + "=" * 60)
    print("GIF PREVIEW:")
    print("=" * 60)
    display(Image(filename=save_path))
    
    print(f"\nğŸ“¥ To download: Look in the 'Output' tab on the right sidebar")
    print(f"   or find the file at: {save_path}")
else:
    print("âš ï¸� No frames generated")


# Generate and Save Pitch Control GIF
# ============================================================================

from IPython.display import Image, display
from PIL import Image as PILImage
from io import BytesIO
import os

# ============================================================================
# CHOOSE YOUR PLAY HERE
# ============================================================================
GAME_ID = 2023110506
PLAY_ID = 892
# ============================================================================

# Use Kaggle working directory (downloadable after notebook runs)
save_path = f"/kaggle/working/pitch_control_{GAME_ID}_{PLAY_ID}.gif"

# Generate the animation frames
frame_images, metadata = create_animation(
    game_id=GAME_ID,
    play_id=PLAY_ID,
    fps=5,
    max_frames=20,
    save_path=None  # Don't use built-in save, we'll do it manually
)

# Save GIF manually
if frame_images and len(frame_images) > 0:
    print(f"\nSaving GIF to {save_path}...")
    
    # Convert bytes to PIL images
    pil_images = [PILImage.open(BytesIO(img)) for img in frame_images]
    
    # Save as GIF
    pil_images[0].save(
        save_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=200,  # milliseconds per frame (200ms = 5fps)
        loop=0
    )
    
    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"âœ“ Saved! ({file_size:.2f} MB)")
    
    # Display in notebook
    print("\n" + "=" * 60)
    print("GIF PREVIEW:")
    print("=" * 60)
    display(Image(filename=save_path))
    
    print(f"\nğŸ“¥ To download: Look in the 'Output' tab on the right sidebar")
    print(f"   or find the file at: {save_path}")
else:
    print("âš ï¸� No frames generated")


"""
NFL Big Data Bowl 2026 - Pitch Control Analysis
================================================================
Cell 11: Interactive Play Selector

Enhanced play selection interface with filtering by:
- Coverage type (Cover 3, Man, etc.)
- Pass outcome (Complete, Incomplete, Interception)
- Pass distance (Short, Medium, Deep)

Provides one-click visualization and animation generation.
================================================================
"""

print("=" * 70)
print("INTERACTIVE PLAY SELECTOR")
print("=" * 70)

# ============================================================================
# BUILD PLAY METADATA FOR FILTERING
# ============================================================================

def build_play_metadata():
    """
    Build comprehensive metadata for all plays to enable filtering.
    """
    print("\nBuilding play metadata...")
    
    # Get unique plays
    play_info = df_input.groupby(['game_id', 'play_id']).first().reset_index()
    
    # Coverage type
    if 'team_coverage_type' in play_info.columns:
        play_info['coverage_type'] = play_info['team_coverage_type'].fillna('Unknown')
    else:
        play_info['coverage_type'] = 'Unknown'
    
    # Pass outcome
    if 'pass_result' in play_info.columns:
        outcome_map = {
            'C': 'Complete',
            'I': 'Incomplete',
            'IN': 'Interception'
        }
        play_info['outcome'] = play_info['pass_result'].map(outcome_map).fillna('Unknown')
    else:
        play_info['outcome'] = 'Unknown'
    
    # Calculate pass distance
    qb_data = df_input[df_input['player_role'] == 'Passer'].copy()
    qb_last = qb_data.groupby(['game_id', 'play_id']).last().reset_index()
    
    if 'x_norm' in qb_last.columns and 'ball_land_x_norm' in qb_last.columns:
        qb_last['pass_distance'] = np.sqrt(
            (qb_last['ball_land_x_norm'] - qb_last['x_norm'])**2 +
            (qb_last['ball_land_y_norm'] - qb_last['y_norm'])**2
        )
        
        play_info = play_info.merge(
            qb_last[['game_id', 'play_id', 'pass_distance']],
            on=['game_id', 'play_id'],
            how='left'
        )
    else:
        play_info['pass_distance'] = np.nan
    
    # Categorize distance
    def categorize_distance(d):
        if pd.isna(d):
            return 'Unknown'
        elif d < 10:
            return 'Short (0-10 yd)'
        elif d < 20:
            return 'Medium (10-20 yd)'
        else:
            return 'Deep (20+ yd)'
    
    play_info['distance_category'] = play_info['pass_distance'].apply(categorize_distance)
    
    print(f"  âœ“ {len(play_info):,} plays indexed")
    
    return play_info[['game_id', 'play_id', 'coverage_type', 'outcome', 
                      'pass_distance', 'distance_category']]

# Build metadata
play_metadata = build_play_metadata()

# ============================================================================
# INTERACTIVE SELECTOR
# ============================================================================

def create_play_selector(fps=DEFAULT_FPS, max_frames=MAX_ANIMATION_FRAMES,
                         grid_spacing=GRID_SPACING):
    """
    Create interactive play selection interface.
    
    Returns selected play info when user clicks buttons.
    """
    print("\n" + "=" * 60)
    print("PLAY SELECTOR")
    print("=" * 60)
    
    # Build filter options
    coverage_options = ['All'] + sorted(play_metadata['coverage_type'].unique().tolist())
    outcome_options = ['All'] + sorted(play_metadata['outcome'].unique().tolist())
    distance_options = ['All', 'Short (0-10 yd)', 'Medium (10-20 yd)', 'Deep (20+ yd)']
    
    # Output area for results
    output_area = widgets.Output()
    
    # Filter widgets
    coverage_dropdown = widgets.Dropdown(
        options=coverage_options,
        value='All',
        description='Coverage:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='200px')
    )
    
    outcome_dropdown = widgets.Dropdown(
        options=outcome_options,
        value='All',
        description='Outcome:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='200px')
    )
    
    distance_dropdown = widgets.Dropdown(
        options=distance_options,
        value='All',
        description='Distance:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='200px')
    )
    
    # Play selection
    play_dropdown = widgets.Dropdown(
        options=[],
        description='Play:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='450px')
    )
    
    info_label = widgets.HTML(value="<b>Filtered plays: 0</b>")
    
    # Action buttons
    random_button = widgets.Button(
        description='ğŸ�² Random',
        button_style='warning',
        layout=widgets.Layout(width='100px')
    )
    
    visualize_button = widgets.Button(
        description='ğŸ“· Static',
        button_style='info',
        layout=widgets.Layout(width='100px')
    )
    
    animate_button = widgets.Button(
        description='â–¶ Animate',
        button_style='success',
        layout=widgets.Layout(width='100px')
    )
    
    # Results storage
    results = {'game_id': None, 'play_id': None}
    
    # Filter logic
    def get_filtered_plays():
        filtered = play_metadata.copy()
        
        if coverage_dropdown.value != 'All':
            filtered = filtered[filtered['coverage_type'] == coverage_dropdown.value]
        if outcome_dropdown.value != 'All':
            filtered = filtered[filtered['outcome'] == outcome_dropdown.value]
        if distance_dropdown.value != 'All':
            filtered = filtered[filtered['distance_category'] == distance_dropdown.value]
        
        return filtered
    
    def update_play_list(*args):
        with output_area:
            clear_output(wait=True)
        
        filtered = get_filtered_plays()
        
        # Create play options
        play_options = []
        for _, row in filtered.iterrows():
            dist_str = f"{row['pass_distance']:.0f}yd" if pd.notna(row['pass_distance']) else "?"
            label = f"G{row['game_id']}/P{row['play_id']} | {row['outcome']} | {row['coverage_type']} | {dist_str}"
            value = (row['game_id'], row['play_id'])
            play_options.append((label, value))
        
        play_dropdown.options = play_options
        info_label.value = f"<b>Filtered plays: {len(play_options):,}</b>"
        
        with output_area:
            if len(play_options) > 0:
                print(f"âœ“ Found {len(play_options):,} plays")
            else:
                print("âš ï¸� No plays match filters")
    
    def on_random(b):
        filtered = get_filtered_plays()
        if len(filtered) == 0:
            return
        play = filtered.sample(1).iloc[0]
        
        # Find and select in dropdown
        for opt_label, opt_value in play_dropdown.options:
            if opt_value == (play['game_id'], play['play_id']):
                play_dropdown.value = opt_value
                break
        
        with output_area:
            clear_output(wait=True)
            print(f"ğŸ�² Selected: Game {play['game_id']}, Play {play['play_id']}")
    
    def on_visualize(b):
        with output_area:
            clear_output(wait=True)
            
            if play_dropdown.value is None:
                print("âš ï¸� Select a play first")
                return
            
            game_id, play_id = play_dropdown.value
            results['game_id'] = game_id
            results['play_id'] = play_id
            
            fig, result = visualize_play(game_id, play_id, grid_spacing=grid_spacing)
            
            if fig:
                plt.show()
    
    def on_animate(b):
        with output_area:
            clear_output(wait=True)
            
            if play_dropdown.value is None:
                print("âš ï¸� Select a play first")
                return
            
            game_id, play_id = play_dropdown.value
            results['game_id'] = game_id
            results['play_id'] = play_id
            
            save_path = f'/kaggle/working/pitch_control_g{game_id}_p{play_id}.gif'
            
            frame_images, metadata = create_animation(
                game_id, play_id, fps=fps, max_frames=max_frames,
                grid_spacing=grid_spacing, save_path=save_path
            )
            
            if frame_images:
                display_animation_player(frame_images, metadata)
    
    # Connect callbacks
    coverage_dropdown.observe(update_play_list, names='value')
    outcome_dropdown.observe(update_play_list, names='value')
    distance_dropdown.observe(update_play_list, names='value')
    
    random_button.on_click(on_random)
    visualize_button.on_click(on_visualize)
    animate_button.on_click(on_animate)
    
    # Initialize
    update_play_list()
    
    # Layout
    filter_row = widgets.HBox([coverage_dropdown, outcome_dropdown, distance_dropdown])
    play_row = widgets.HBox([play_dropdown, random_button])
    action_row = widgets.HBox([visualize_button, animate_button])
    
    ui = widgets.VBox([
        widgets.HTML("<h3>ğŸ�ˆ Pitch Control - Play Selector</h3>"),
        widgets.HTML("<b>Filters:</b>"),
        filter_row,
        widgets.HTML("<hr>"),
        info_label,
        play_row,
        widgets.HTML("<hr>"),
        widgets.HTML("<b>Actions:</b>"),
        action_row,
        output_area
    ])
    
    display(ui)
    
    return results


# ============================================================================
# QUICK FUNCTIONS
# ============================================================================

def random_play(coverage=None, outcome=None, distance=None):
    """
    Get a random play matching criteria.
    
    Parameters:
        coverage: Coverage type filter
        outcome: Outcome filter ('Complete', 'Incomplete', 'Interception')
        distance: Distance category filter
        
    Returns:
        game_id, play_id
    """
    filtered = play_metadata.copy()
    
    if coverage:
        filtered = filtered[filtered['coverage_type'] == coverage]
    if outcome:
        filtered = filtered[filtered['outcome'] == outcome]
    if distance:
        filtered = filtered[filtered['distance_category'] == distance]
    
    if len(filtered) == 0:
        print("No plays match criteria")
        return None, None
    
    play = filtered.sample(1).iloc[0]
    return play['game_id'], play['play_id']


def quick_animate(coverage=None, outcome=None, distance=None):
    """
    Quickly animate a random play matching criteria.
    """
    game_id, play_id = random_play(coverage, outcome, distance)
    
    if game_id is None:
        return None, None
    
    print(f"Selected: Game {game_id}, Play {play_id}")
    
    return animate_play(game_id, play_id)


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("PLAY SELECTOR READY")
print("=" * 70)

print(f"\nPlay metadata summary:")
print(f"  â€¢ Total plays: {len(play_metadata):,}")

if 'coverage_type' in play_metadata.columns:
    print(f"\n  By coverage:")
    for cov, count in play_metadata['coverage_type'].value_counts().head(5).items():
        print(f"    â€¢ {cov}: {count:,}")

if 'outcome' in play_metadata.columns:
    print(f"\n  By outcome:")
    for out, count in play_metadata['outcome'].value_counts().items():
        print(f"    â€¢ {out}: {count:,}")

print("\n" + "-" * 70)
print("Usage:")
print("-" * 70)
print("\n  # Interactive selector:")
print("  create_play_selector()")
print("")
print("  # Quick random animation:")
print("  quick_animate(outcome='Interception')")
print("  quick_animate(coverage='Cover 3', distance='Deep (20+ yd)')")
print("")
print("  # Get random play:")
print("  game_id, play_id = random_play(outcome='Complete')")
print("=" * 70)


# create_play_selector()


# # Cell 12: Model Validation (SIMPLIFIED & FOCUSED)
# #This is the full validation and has been commented out. To un-comment, select all and ctrl + /
# # ============================================================================
# # Validates the pitch control model with three clear metrics:
# # 1. Completions: Does ball land within targeted receiver's P95 reach?
# # 2. Interceptions: Does ball land within any defender's P95 reach?
# # 3. Control ratio: How does control differ between outcomes?
# # ============================================================================

# print("=" * 70)
# print("PITCH CONTROL MODEL VALIDATION")
# print("=" * 70)

# def validate_pitch_control_model(sample_size=500, verbose=True):
#     """
#     Validate pitch control model with three focused metrics:
    
#     1. RECEIVER REACHABILITY (Completions)
#        - Is ball landing point within targeted receiver's P95 contour + catch radius?
       
#     2. DEFENDER REACHABILITY (Interceptions)  
#        - Is ball landing point within ANY defender's P95 contour + catch radius?
       
#     3. CONTROL RATIO SEPARATION
#        - Do completions have higher offense control than interceptions?
#     """
    
#     print("\nValidating pitch control model...")
    
#     # Get plays by outcome
#     completions = play_metadata[play_metadata['outcome'] == 'Complete']
#     incompletions = play_metadata[play_metadata['outcome'] == 'Incomplete']
#     interceptions = play_metadata[play_metadata['outcome'] == 'Interception']
    
#     n_comp = min(sample_size, len(completions))
#     n_inc = min(sample_size, len(incompletions))
#     n_int = min(sample_size, len(interceptions))
    
#     completions = completions.sample(n_comp, random_state=42)
#     incompletions = incompletions.sample(n_inc, random_state=42)
#     interceptions = interceptions.sample(n_int, random_state=42)
    
#     if verbose:
#         print(f"  â€¢ Testing {n_comp} completions")
#         print(f"  â€¢ Testing {n_inc} incompletions")
#         print(f"  â€¢ Testing {n_int} interceptions")
    
#     results = {
#         'completions': [],
#         'incompletions': [],
#         'interceptions': []
#     }
    
#     def analyze_play(game_id, play_id, outcome):
#         """
#         For a single play, compute:
#         - Whether ball lands in targeted receiver's P95 + catch radius
#         - Whether ball lands in any defender's P95 + catch radius
#         - Control ratio at ball landing point
#         """
#         try:
#             # Get INPUT data
#             input_play = df_input[
#                 (df_input['game_id'] == game_id) & 
#                 (df_input['play_id'] == play_id)
#             ]
            
#             if len(input_play) == 0:
#                 return None
            
#             # Get ball landing position
#             ball_x = input_play['ball_land_x_norm'].iloc[0]
#             ball_y = input_play['ball_land_y_norm'].iloc[0]
            
#             if pd.isna(ball_x) or pd.isna(ball_y):
#                 return None
            
#             # Get OUTPUT data (first frame after release)
#             output_play = df_output[
#                 (df_output['game_id'] == game_id) & 
#                 (df_output['play_id'] == play_id)
#             ].copy()
            
#             if len(output_play) == 0:
#                 return None
            
#             # Get first OUTPUT frame
#             first_frame_id = output_play['frame_id'].min()
#             df_frame = output_play[output_play['frame_id'] == first_frame_id].copy()
            
#             # Calculate velocities for this frame
#             df_frame = calculate_frame_velocities(df_frame.copy(), game_id, play_id)
            
#             # Get flight time
#             flight_time, _ = get_ball_flight_time(game_id, play_id)
#             if flight_time is None:
#                 flight_time = 1.5
            
#             # Add metadata
#             metadata = player_metadata_lookup.get(game_id, {}).get(play_id, {})
#             df_frame['player_side'] = df_frame['nfl_id'].apply(
#                 lambda x: metadata.get(x, {}).get('side', 'Unknown')
#             )
#             df_frame['player_role'] = df_frame['nfl_id'].apply(
#                 lambda x: metadata.get(x, {}).get('role', 'Unknown')
#             )
            
#             # ================================================================
#             # CHECK 1: Is ball within targeted receiver's P95 + catch radius?
#             # ================================================================
            
#             targeted_receiver = df_frame[df_frame['player_role'] == 'Targeted Receiver']
#             ball_in_receiver_reach = False
#             receiver_distance = None
#             receiver_p95_reach = None
            
#             if len(targeted_receiver) > 0:
#                 rec = targeted_receiver.iloc[0]
#                 rec_x = rec['x_norm']
#                 rec_y = rec['y_norm']
#                 rec_speed = rec['speed_actual'] if 'speed_actual' in rec and pd.notna(rec['speed_actual']) else rec.get('s', 3.0)
#                 rec_dir = rec['dir_actual'] if 'dir_actual' in rec and pd.notna(rec['dir_actual']) else rec.get('dir_norm', 0)
                
#                 if pd.isna(rec_speed):
#                     rec_speed = 3.0
#                 if pd.isna(rec_dir):
#                     rec_dir = 0
                
#                 # Get receiver's movement distribution
#                 dist, vel_bin, _, _ = get_player_distribution_interpolated(rec_speed, flight_time)
                
#                 if dist is not None:
#                     # P95 reach from distribution + catch radius
#                     receiver_p95_reach = dist['p95_reach'] + CATCH_RADIUS
                    
#                     # Distance from receiver's current position to ball
#                     # But we need to account for movement direction...
#                     # Simpler approach: transform ball to player-relative coords
#                     dx = ball_x - rec_x
#                     dy = ball_y - rec_y
                    
#                     # Rotate to player frame
#                     theta = np.radians(rec_dir)
#                     dx_rel = dx * np.cos(-theta) - dy * np.sin(-theta)
#                     dy_rel = dx * np.sin(-theta) + dy * np.cos(-theta)
                    
#                     # Check if ball is within distribution's P95 contour
#                     # Create small grid around ball landing point
#                     grid_x = np.array(dist['grid_x'])
#                     grid_y = np.array(dist['grid_y'])
#                     density = dist['density']
                    
#                     # Find nearest grid point to ball (in player-relative coords)
#                     x_idx = np.argmin(np.abs(grid_x - dx_rel))
#                     y_idx = np.argmin(np.abs(grid_y - dy_rel))
                    
#                     # Check if within grid bounds
#                     if 0 <= x_idx < len(grid_x) and 0 <= y_idx < len(grid_y):
#                         ball_density = density[y_idx, x_idx]
                        
#                         # Compute P95 threshold
#                         p95_threshold = compute_p95_threshold(density)
                        
#                         # Ball is reachable if its density >= P95 threshold
#                         # OR if straight-line distance < P95 reach + catch radius
#                         receiver_distance = np.sqrt(dx**2 + dy**2)
#                         ball_in_receiver_reach = (ball_density >= p95_threshold) or (receiver_distance <= receiver_p95_reach)
#                     else:
#                         # Ball is outside grid - check simple distance
#                         receiver_distance = np.sqrt(dx**2 + dy**2)
#                         ball_in_receiver_reach = receiver_distance <= receiver_p95_reach
#                 else:
#                     # Fallback: simple distance check
#                     receiver_distance = np.sqrt((ball_x - rec_x)**2 + (ball_y - rec_y)**2)
#                     receiver_p95_reach = rec_speed * flight_time * 0.8 + CATCH_RADIUS
#                     ball_in_receiver_reach = receiver_distance <= receiver_p95_reach
            
#             # ================================================================
#             # CHECK 2: Is ball within ANY defender's P95 + catch radius?
#             # ================================================================
            
#             defenders = df_frame[df_frame['player_side'] == 'Defense']
#             ball_in_any_defender_reach = False
#             min_defender_distance = None
#             closest_defender_p95 = None
            
#             for _, defender in defenders.iterrows():
#                 def_x = defender['x_norm']
#                 def_y = defender['y_norm']
#                 def_speed = defender['speed_actual'] if 'speed_actual' in defender and pd.notna(defender['speed_actual']) else defender.get('s', 3.0)
#                 def_dir = defender['dir_actual'] if 'dir_actual' in defender and pd.notna(defender['dir_actual']) else defender.get('dir_norm', 0)
                
#                 if pd.isna(def_speed):
#                     def_speed = 3.0
#                 if pd.isna(def_dir):
#                     def_dir = 0
                
#                 # Get defender's movement distribution
#                 dist, vel_bin, _, _ = get_player_distribution_interpolated(def_speed, flight_time)
                
#                 if dist is not None:
#                     defender_p95_reach = dist['p95_reach'] + CATCH_RADIUS
                    
#                     # Transform ball to defender-relative coords
#                     dx = ball_x - def_x
#                     dy = ball_y - def_y
#                     theta = np.radians(def_dir)
#                     dx_rel = dx * np.cos(-theta) - dy * np.sin(-theta)
#                     dy_rel = dx * np.sin(-theta) + dy * np.cos(-theta)
                    
#                     grid_x = np.array(dist['grid_x'])
#                     grid_y = np.array(dist['grid_y'])
#                     density = dist['density']
                    
#                     x_idx = np.argmin(np.abs(grid_x - dx_rel))
#                     y_idx = np.argmin(np.abs(grid_y - dy_rel))
                    
#                     defender_distance = np.sqrt(dx**2 + dy**2)
                    
#                     if 0 <= x_idx < len(grid_x) and 0 <= y_idx < len(grid_y):
#                         ball_density = density[y_idx, x_idx]
#                         p95_threshold = compute_p95_threshold(density)
                        
#                         in_reach = (ball_density >= p95_threshold) or (defender_distance <= defender_p95_reach)
#                     else:
#                         in_reach = defender_distance <= defender_p95_reach
#                 else:
#                     defender_distance = np.sqrt((ball_x - def_x)**2 + (ball_y - def_y)**2)
#                     defender_p95_reach = def_speed * flight_time * 0.8 + CATCH_RADIUS
#                     in_reach = defender_distance <= defender_p95_reach
                
#                 if in_reach:
#                     ball_in_any_defender_reach = True
                
#                 # Track closest defender
#                 if min_defender_distance is None or defender_distance < min_defender_distance:
#                     min_defender_distance = defender_distance
#                     closest_defender_p95 = defender_p95_reach
            
#             # ================================================================
#             # CHECK 3: Control ratio at ball landing point
#             # ================================================================
            
#             # Use the full pitch control calculation
#             pc_result = calculate_pitch_control(
#                 df_frame, game_id, play_id, flight_time,
#                 grid_spacing=1.0, use_interpolation=True
#             )
            
#             grid_x = pc_result['grid_x']
#             grid_y = pc_result['grid_y']
#             control_ratio = pc_result['control_ratio']
            
#             # Find control at ball landing point
#             distances = np.sqrt((grid_x - ball_x)**2 + (grid_y - ball_y)**2)
#             nearest_idx = np.unravel_index(np.argmin(distances), distances.shape)
#             ball_control = control_ratio[nearest_idx]
            
#             return {
#                 'game_id': game_id,
#                 'play_id': play_id,
#                 'outcome': outcome,
#                 'ball_x': ball_x,
#                 'ball_y': ball_y,
#                 'ball_in_receiver_reach': ball_in_receiver_reach,
#                 'receiver_distance': receiver_distance,
#                 'receiver_p95_reach': receiver_p95_reach,
#                 'ball_in_defender_reach': ball_in_any_defender_reach,
#                 'min_defender_distance': min_defender_distance,
#                 'closest_defender_p95': closest_defender_p95,
#                 'control_ratio': ball_control,
#                 'offense_favored': ball_control > 0.5
#             }
            
#         except Exception as e:
#             if verbose:
#                 print(f"    Error on {game_id}-{play_id}: {e}")
#             return None
    
#     # Process completions
#     if verbose:
#         print("\n  Processing completions...")
    
#     for idx, (_, play) in enumerate(completions.iterrows()):
#         result = analyze_play(play['game_id'], play['play_id'], 'Complete')
#         if result:
#             results['completions'].append(result)
        
#         if verbose and (idx + 1) % 100 == 0:
#             print(f"    {idx + 1}/{n_comp} completions")
    
#     # Process incompletions
#     if verbose:
#         print("\n  Processing incompletions...")
    
#     for idx, (_, play) in enumerate(incompletions.iterrows()):
#         result = analyze_play(play['game_id'], play['play_id'], 'Incomplete')
#         if result:
#             results['incompletions'].append(result)
        
#         if verbose and (idx + 1) % 100 == 0:
#             print(f"    {idx + 1}/{n_inc} incompletions")
    
#     # Process interceptions
#     if verbose:
#         print("\n  Processing interceptions...")
    
#     for idx, (_, play) in enumerate(interceptions.iterrows()):
#         result = analyze_play(play['game_id'], play['play_id'], 'Interception')
#         if result:
#             results['interceptions'].append(result)
        
#         if verbose and (idx + 1) % 50 == 0:
#             print(f"    {idx + 1}/{n_int} interceptions")
    
#     return results


# def summarize_validation(results):
#     """
#     Summarize validation results with clear metrics.
#     """
    
#     df_comp = pd.DataFrame(results['completions'])
#     df_inc = pd.DataFrame(results['incompletions'])
#     df_int = pd.DataFrame(results['interceptions'])
    
#     print("\n" + "=" * 70)
#     print("VALIDATION RESULTS")
#     print("=" * 70)
    
#     # ========================================================================
#     # METRIC 1: Receiver Reachability for Completions
#     # ========================================================================
    
#     print("\n" + "-" * 70)
#     print("METRIC 1: RECEIVER REACHABILITY")
#     print("'Does the ball land within the targeted receiver's P95 reach + catch radius?'")
#     print("-" * 70)
    
#     comp_in_reach = df_comp['ball_in_receiver_reach'].sum()
#     comp_total = len(df_comp)
#     comp_reach_pct = comp_in_reach / comp_total * 100 if comp_total > 0 else 0
    
#     inc_in_reach = df_inc['ball_in_receiver_reach'].sum()
#     inc_total = len(df_inc)
#     inc_reach_pct = inc_in_reach / inc_total * 100 if inc_total > 0 else 0
    
#     print(f"\n  COMPLETIONS:")
#     print(f"    Ball in receiver's reach: {comp_in_reach} / {comp_total} ({comp_reach_pct:.1f}%)")
#     print(f"    Mean distance to receiver: {df_comp['receiver_distance'].mean():.2f} yards")
#     print(f"    Mean receiver P95 reach: {df_comp['receiver_p95_reach'].mean():.2f} yards")
    
#     print(f"\n  INCOMPLETIONS:")
#     print(f"    Ball in receiver's reach: {inc_in_reach} / {inc_total} ({inc_reach_pct:.1f}%)")
#     print(f"    Mean distance to receiver: {df_inc['receiver_distance'].mean():.2f} yards")
#     print(f"    Mean receiver P95 reach: {df_inc['receiver_p95_reach'].mean():.2f} yards")
    
#     print(f"\n  INTERPRETATION:")
#     if comp_reach_pct > 80:
#         print(f"    âœ… Excellent! {comp_reach_pct:.0f}% of completions land where receiver can reach")
#     elif comp_reach_pct > 60:
#         print(f"    âœ“ Good. {comp_reach_pct:.0f}% of completions land where receiver can reach")
#     else:
#         print(f"    âš ï¸� Only {comp_reach_pct:.0f}% of completions land where receiver can reach")
    
#     if comp_reach_pct > inc_reach_pct + 10:
#         print(f"    âœ… Clear separation: completions {comp_reach_pct:.0f}% vs incompletions {inc_reach_pct:.0f}%")
    
#     # ========================================================================
#     # METRIC 2: Defender Reachability for Interceptions
#     # ========================================================================
    
#     print("\n" + "-" * 70)
#     print("METRIC 2: DEFENDER REACHABILITY")
#     print("'Does the ball land within ANY defender's P95 reach + catch radius?'")
#     print("-" * 70)
    
#     int_in_def_reach = df_int['ball_in_defender_reach'].sum()
#     int_total = len(df_int)
#     int_def_reach_pct = int_in_def_reach / int_total * 100 if int_total > 0 else 0
    
#     comp_in_def_reach = df_comp['ball_in_defender_reach'].sum()
#     comp_def_reach_pct = comp_in_def_reach / comp_total * 100 if comp_total > 0 else 0
    
#     print(f"\n  INTERCEPTIONS:")
#     print(f"    Ball in defender's reach: {int_in_def_reach} / {int_total} ({int_def_reach_pct:.1f}%)")
#     print(f"    Mean distance to closest defender: {df_int['min_defender_distance'].mean():.2f} yards")
    
#     print(f"\n  COMPLETIONS (for comparison):")
#     print(f"    Ball in defender's reach: {comp_in_def_reach} / {comp_total} ({comp_def_reach_pct:.1f}%)")
#     print(f"    Mean distance to closest defender: {df_comp['min_defender_distance'].mean():.2f} yards")
    
#     print(f"\n  INTERPRETATION:")
#     if int_def_reach_pct > 80:
#         print(f"    âœ… Excellent! {int_def_reach_pct:.0f}% of interceptions land where a defender can reach")
#     elif int_def_reach_pct > 60:
#         print(f"    âœ“ Good. {int_def_reach_pct:.0f}% of interceptions land where a defender can reach")
#     else:
#         print(f"    âš ï¸� Only {int_def_reach_pct:.0f}% of interceptions land where a defender can reach")
    
#     if int_def_reach_pct > comp_def_reach_pct + 10:
#         print(f"    âœ… Clear separation: interceptions {int_def_reach_pct:.0f}% vs completions {comp_def_reach_pct:.0f}%")
    
#     # ========================================================================
#     # METRIC 3: Control Ratio by Outcome
#     # ========================================================================
    
#     print("\n" + "-" * 70)
#     print("METRIC 3: CONTROL RATIO BY OUTCOME")
#     print("'Do completions have higher offense control than interceptions?'")
#     print("-" * 70)
    
#     comp_control_mean = df_comp['control_ratio'].mean()
#     comp_control_median = df_comp['control_ratio'].median()
#     comp_offense_favored = df_comp['offense_favored'].mean() * 100
    
#     inc_control_mean = df_inc['control_ratio'].mean()
#     inc_control_median = df_inc['control_ratio'].median()
#     inc_offense_favored = df_inc['offense_favored'].mean() * 100
    
#     int_control_mean = df_int['control_ratio'].mean()
#     int_control_median = df_int['control_ratio'].median()
#     int_offense_favored = df_int['offense_favored'].mean() * 100
    
#     print(f"\n  {'Outcome':<15} {'Mean Ctrl':>12} {'Median':>10} {'Off Favored':>14}")
#     print("  " + "-" * 55)
#     print(f"  {'Completions':<15} {comp_control_mean:>12.3f} {comp_control_median:>10.3f} {comp_offense_favored:>13.1f}%")
#     print(f"  {'Incompletions':<15} {inc_control_mean:>12.3f} {inc_control_median:>10.3f} {inc_offense_favored:>13.1f}%")
#     print(f"  {'Interceptions':<15} {int_control_mean:>12.3f} {int_control_median:>10.3f} {int_offense_favored:>13.1f}%")
    
#     separation = comp_control_mean - int_control_mean
    
#     print(f"\n  SEPARATION (Completions - Interceptions): {separation:.3f}")
    
#     print(f"\n  INTERPRETATION:")
#     if separation > 0.15:
#         print(f"    âœ… Strong separation! Model clearly distinguishes outcomes by control")
#     elif separation > 0.05:
#         print(f"    âœ“ Moderate separation. Model has predictive power")
#     elif separation > 0:
#         print(f"    âš ï¸� Weak separation. Model has limited predictive power")
#     else:
#         print(f"    â�Œ No separation or inverted. Model may have issues")
    
#     if comp_offense_favored > 60:
#         print(f"    âœ… {comp_offense_favored:.0f}% of completions land in offense-controlled space")
    
#     if int_offense_favored < 40:
#         print(f"    âœ… Only {int_offense_favored:.0f}% of interceptions land in offense-controlled space")
    
#     # ========================================================================
#     # CONTROL RATIO BINS
#     # ========================================================================
    
#     print("\n" + "-" * 70)
#     print("CONTROL RATIO DISTRIBUTION BY OUTCOME")
#     print("-" * 70)
    
#     bins = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
#     bin_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.8', '0.8-1.0']
    
#     print(f"\n  {'Control Bin':<12} {'Completions':>12} {'Incompletions':>14} {'Interceptions':>14}")
#     print("  " + "-" * 55)
    
#     for i in range(len(bins) - 1):
#         low, high = bins[i], bins[i+1]
        
#         comp_count = ((df_comp['control_ratio'] >= low) & (df_comp['control_ratio'] < high)).sum()
#         comp_pct = comp_count / len(df_comp) * 100 if len(df_comp) > 0 else 0
        
#         inc_count = ((df_inc['control_ratio'] >= low) & (df_inc['control_ratio'] < high)).sum()
#         inc_pct = inc_count / len(df_inc) * 100 if len(df_inc) > 0 else 0
        
#         int_count = ((df_int['control_ratio'] >= low) & (df_int['control_ratio'] < high)).sum()
#         int_pct = int_count / len(df_int) * 100 if len(df_int) > 0 else 0
        
#         print(f"  {bin_labels[i]:<12} {comp_pct:>11.1f}% {inc_pct:>13.1f}% {int_pct:>13.1f}%")
    
#     # ========================================================================
#     # SUMMARY
#     # ========================================================================
    
#     print("\n" + "=" * 70)
#     print("VALIDATION SUMMARY")
#     print("=" * 70)
    
#     print(f"""
#   RECEIVER REACHABILITY:
#     â€¢ {comp_reach_pct:.1f}% of completions land within receiver's P95 reach
#     â€¢ {inc_reach_pct:.1f}% of incompletions land within receiver's P95 reach
    
#   DEFENDER REACHABILITY:
#     â€¢ {int_def_reach_pct:.1f}% of interceptions land within a defender's P95 reach
#     â€¢ {comp_def_reach_pct:.1f}% of completions land within a defender's P95 reach
    
#   CONTROL RATIO:
#     â€¢ Completions avg control: {comp_control_mean:.3f} ({comp_offense_favored:.0f}% offense-favored)
#     â€¢ Interceptions avg control: {int_control_mean:.3f} ({int_offense_favored:.0f}% offense-favored)
#     â€¢ Separation: {separation:.3f}
# """)
    
#     # Overall assessment
#     score = 0
#     if comp_reach_pct > 70:
#         score += 1
#     if int_def_reach_pct > 70:
#         score += 1
#     if separation > 0.10:
#         score += 1
#     if comp_offense_favored > 55:
#         score += 1
    
#     if score >= 4:
#         print("  OVERALL: âœ… Model validation PASSED - strong predictive power")
#     elif score >= 2:
#         print("  OVERALL: âœ“ Model validation ACCEPTABLE - moderate predictive power")
#     else:
#         print("  OVERALL: âš ï¸� Model validation MARGINAL - limited predictive power")
    
#     return {
#         'completions': {
#             'n': comp_total,
#             'in_receiver_reach_pct': comp_reach_pct,
#             'in_defender_reach_pct': comp_def_reach_pct,
#             'mean_control': comp_control_mean,
#             'offense_favored_pct': comp_offense_favored
#         },
#         'incompletions': {
#             'n': inc_total,
#             'in_receiver_reach_pct': inc_reach_pct,
#             'mean_control': inc_control_mean,
#             'offense_favored_pct': inc_offense_favored
#         },
#         'interceptions': {
#             'n': int_total,
#             'in_defender_reach_pct': int_def_reach_pct,
#             'mean_control': int_control_mean,
#             'offense_favored_pct': int_offense_favored
#         },
#         'separation': separation
#     }


# # ============================================================================
# # RUN VALIDATION
# # ============================================================================

# print("\n" + "=" * 70)
# print("RUNNING VALIDATION")
# print("=" * 70)

# validation_results = validate_pitch_control_model(sample_size=500, verbose=True)
# summary = summarize_validation(validation_results)

# print("\n" + "=" * 70)
# print("VALIDATION COMPLETE")
# print("=" * 70)


print("=" * 70)
print("PITCH CONTROL MODEL VALIDATION (PRE-COMPUTED)")
print("=" * 70)

print("""
======================================================================
VALIDATION RESULTS
======================================================================

----------------------------------------------------------------------
METRIC 1: RECEIVER REACHABILITY
'Does the ball land within the targeted receiver's P95 reach + catch radius?'
----------------------------------------------------------------------

  COMPLETIONS:
    Ball in receiver's reach: 497 / 500 (99.4%)
    Mean distance to receiver: 5.08 yards
    Mean receiver P95 reach: 11.12 yards

  INCOMPLETIONS:
    Ball in receiver's reach: 497 / 500 (99.4%)
    Mean distance to receiver: 9.00 yards
    Mean receiver P95 reach: 14.49 yards

  INTERPRETATION:
    âœ… Excellent! 99% of completions land where receiver can reach

----------------------------------------------------------------------
METRIC 2: DEFENDER REACHABILITY
'Does the ball land within ANY defender's P95 reach + catch radius?'
----------------------------------------------------------------------

  INTERCEPTIONS:
    Ball in defender's reach: 331 / 338 (97.9%)
    Mean distance to closest defender: 7.46 yards

  COMPLETIONS (for comparison):
    Ball in defender's reach: 393 / 500 (78.6%)
    Mean distance to closest defender: 7.23 yards

  INTERPRETATION:
    âœ… Excellent! 98% of interceptions land where a defender can reach
    âœ… Clear separation: interceptions 98% vs completions 79%

----------------------------------------------------------------------
METRIC 3: CONTROL RATIO BY OUTCOME
'Do completions have higher offense control than interceptions?'
----------------------------------------------------------------------

  Outcome            Mean Ctrl     Median    Off Favored
  -------------------------------------------------------
  Completions            0.655      0.667          64.2%
  Incompletions          0.424      0.435          37.4%
  Interceptions          0.267      0.243          14.8%

  SEPARATION (Completions - Interceptions): 0.389

  INTERPRETATION:
    âœ… Strong separation! Model clearly distinguishes outcomes by control
    âœ… 64% of completions land in offense-controlled space
    âœ… Only 15% of interceptions land in offense-controlled space

----------------------------------------------------------------------
CONTROL RATIO DISTRIBUTION BY OUTCOME
----------------------------------------------------------------------

  Control Bin   Completions  Incompletions  Interceptions
  -------------------------------------------------------
  0.0-0.2              8.2%          26.8%          44.7%
  0.2-0.4             13.8%          19.6%          27.2%
  0.4-0.5             13.8%          16.2%          13.3%
  0.5-0.6              9.0%          12.6%           6.2%
  0.6-0.8             13.6%           9.8%           4.7%
  0.8-1.0             41.6%          15.0%           3.8%

======================================================================
VALIDATION SUMMARY
======================================================================

  RECEIVER REACHABILITY:
    â€¢ 99.4% of completions land within receiver's P95 reach
    â€¢ 99.4% of incompletions land within receiver's P95 reach
    
  DEFENDER REACHABILITY:
    â€¢ 97.9% of interceptions land within a defender's P95 reach
    â€¢ 78.6% of completions land within a defender's P95 reach
    
  CONTROL RATIO:
    â€¢ Completions avg control: 0.655 (64% offense-favored)
    â€¢ Interceptions avg control: 0.267 (15% offense-favored)
    â€¢ Separation: 0.389

  OVERALL: âœ… Model validation PASSED - strong predictive power

======================================================================
VALIDATION COMPLETE
======================================================================
""")

print("=" * 70)


# """
# This original cell has been commented out. To uncomment, select and ctrl + /
# NFL Big Data Bowl 2026 - Pitch Control Analysis
# ================================================================
# Calibration Curve: Control Ratio â†’ Completion Probability

# This analysis establishes the empirical relationship between
# our pitch control model's control_ratio and actual completion
# probability.

# Steps:
# 1. For each play: calculate control_ratio at ball landing point
# 2. Bin by control_ratio, compute completion rate per bin
# 3. Fit smooth calibration function
# 4. Validate the calibration

# Output: A function that converts control_ratio â†’ P(completion)
# ================================================================
# """

# print("=" * 70)
# print("CALIBRATION CURVE: CONTROL RATIO â†’ COMPLETION PROBABILITY")
# print("=" * 70)

# # ============================================================================
# # STEP 1: EXTRACT CONTROL RATIO AT BALL LANDING FOR ALL PLAYS
# # ============================================================================

# def extract_control_at_landing(sample_size=None, verbose=True):
#     """
#     For each play, calculate pitch control at pass release and extract
#     the control_ratio at the ball landing point.
    
#     Parameters:
#         sample_size: If specified, randomly sample this many plays (for speed)
#         verbose: Print progress
        
#     Returns:
#         DataFrame with columns:
#         - game_id, play_id
#         - ball_x, ball_y (landing coordinates)
#         - control_ratio (at landing point)
#         - outcome ('Complete', 'Incomplete', 'Interception')
#         - flight_time
#         - pass_distance
#     """
#     print("\nExtracting control ratio at ball landing for all plays...")
    
#     # Get plays with known outcomes
#     plays_with_outcome = play_metadata[play_metadata['outcome'].isin(['Complete', 'Incomplete', 'Interception'])].copy()
    
#     if sample_size and sample_size < len(plays_with_outcome):
#         plays_with_outcome = plays_with_outcome.sample(sample_size, random_state=42)
    
#     n_plays = len(plays_with_outcome)
#     print(f"  â€¢ Processing {n_plays:,} plays")
    
#     results = []
#     errors = {'no_output': 0, 'no_ball_landing': 0, 'calc_error': 0}
    
#     for idx, (_, play) in enumerate(plays_with_outcome.iterrows()):
#         game_id = play['game_id']
#         play_id = play['play_id']
#         outcome = play['outcome']
        
#         try:
#             # Get INPUT data for ball landing location
#             input_play = df_input[
#                 (df_input['game_id'] == game_id) & 
#                 (df_input['play_id'] == play_id)
#             ]
            
#             if len(input_play) == 0:
#                 errors['no_output'] += 1
#                 continue
            
#             # Get ball landing point
#             ball_x = input_play['ball_land_x_norm'].iloc[0]
#             ball_y = input_play['ball_land_y_norm'].iloc[0]
            
#             if pd.isna(ball_x) or pd.isna(ball_y):
#                 errors['no_ball_landing'] += 1
#                 continue
            
#             # Get OUTPUT data
#             output_play = df_output[
#                 (df_output['game_id'] == game_id) & 
#                 (df_output['play_id'] == play_id)
#             ].copy()
            
#             if len(output_play) == 0:
#                 errors['no_output'] += 1
#                 continue
            
#             # Calculate frame velocities
#             output_play = calculate_frame_velocities(output_play, game_id, play_id)
            
#             # Get first frame (pass release)
#             frame_id = output_play['frame_id'].min()
#             df_frame = output_play[output_play['frame_id'] == frame_id].copy()
            
#             # Get flight time
#             flight_time, flight_meta = get_ball_flight_time(game_id, play_id)
#             if flight_time is None:
#                 flight_time = 1.5
            
#             pass_distance = flight_meta.get('distance', np.nan)
            
#             # Calculate pitch control (using fixed version with interpolation and catch radius)
#             result = calculate_pitch_control(
#                 df_frame, game_id, play_id, flight_time,
#                 grid_spacing=0.5,  # Fine grid for accurate lookup
#                 use_interpolation=True
#             )
            
#             # Find control ratio at ball landing point
#             grid_x = result['grid_x']
#             grid_y = result['grid_y']
#             control_ratio = result['control_ratio']
            
#             # Find nearest grid point to ball landing
#             distances = np.sqrt((grid_x - ball_x)**2 + (grid_y - ball_y)**2)
#             nearest_idx = np.unravel_index(np.argmin(distances), distances.shape)
            
#             landing_control = control_ratio[nearest_idx]
#             landing_offense_density = result['offense_density'][nearest_idx]
#             landing_defense_density = result['defense_density'][nearest_idx]
            
#             # Also check if in open space
#             in_open_space = result['open_space_mask'][nearest_idx]
            
#             results.append({
#                 'game_id': game_id,
#                 'play_id': play_id,
#                 'ball_x': ball_x,
#                 'ball_y': ball_y,
#                 'control_ratio': landing_control,
#                 'offense_density': landing_offense_density,
#                 'defense_density': landing_defense_density,
#                 'in_open_space': in_open_space,
#                 'outcome': outcome,
#                 'flight_time': flight_time,
#                 'pass_distance': pass_distance
#             })
            
#         except Exception as e:
#             errors['calc_error'] += 1
#             continue
        
#         if verbose and (idx + 1) % 100 == 0:
#             n_complete = sum(1 for r in results if r['outcome'] == 'Complete')
#             print(f"    {idx + 1:,}/{n_plays:,} plays... ({len(results)} valid, {n_complete} completions)")
    
#     df_results = pd.DataFrame(results)
    
#     print(f"\n  âœ“ Processed {len(df_results):,} plays successfully")
#     print(f"  â€¢ Completions: {(df_results['outcome'] == 'Complete').sum():,}")
#     print(f"  â€¢ Incompletions: {(df_results['outcome'] == 'Incomplete').sum():,}")
#     print(f"  â€¢ Interceptions: {(df_results['outcome'] == 'Interception').sum():,}")
#     print(f"\n  Errors: {errors}")
    
#     return df_results


# # Run extraction (use sample for speed, or None for full dataset)
# print("\n" + "-" * 70)
# df_landing_control = extract_control_at_landing(sample_size=None, verbose=True)


# # ============================================================================
# # STEP 2: BIN BY CONTROL RATIO AND COMPUTE COMPLETION RATES
# # ============================================================================

# def compute_completion_rates_by_bin(df, n_bins=10):
#     """
#     Bin plays by control_ratio and compute completion rate per bin.
    
#     Returns DataFrame with bin statistics.
#     """
#     print("\n" + "-" * 70)
#     print("COMPLETION RATES BY CONTROL RATIO BIN")
#     print("-" * 70)
    
#     # Create bins
#     df = df.copy()
#     bin_edges = np.linspace(0, 1, n_bins + 1)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
#     df['control_bin'] = pd.cut(df['control_ratio'], bins=bin_edges, labels=bin_centers, include_lowest=True)
#     df['control_bin'] = df['control_bin'].astype(float)
    
#     # Compute stats per bin
#     bin_stats = []
    
#     for center in bin_centers:
#         bin_data = df[df['control_bin'] == center]
#         n_total = len(bin_data)
        
#         if n_total == 0:
#             continue
        
#         n_complete = (bin_data['outcome'] == 'Complete').sum()
#         n_incomplete = (bin_data['outcome'] == 'Incomplete').sum()
#         n_int = (bin_data['outcome'] == 'Interception').sum()
        
#         completion_rate = n_complete / n_total
#         int_rate = n_int / n_total
        
#         # Standard error for completion rate (binomial)
#         se = np.sqrt(completion_rate * (1 - completion_rate) / n_total) if n_total > 0 else 0
        
#         bin_stats.append({
#             'bin_center': center,
#             'bin_low': center - 0.05,
#             'bin_high': center + 0.05,
#             'n_plays': n_total,
#             'n_complete': n_complete,
#             'n_incomplete': n_incomplete,
#             'n_interception': n_int,
#             'completion_rate': completion_rate,
#             'interception_rate': int_rate,
#             'standard_error': se
#         })
    
#     df_bins = pd.DataFrame(bin_stats)
    
#     # Display
#     print(f"\n{'Bin':<12} {'N Plays':<10} {'Complete':<10} {'Incomp':<10} {'INT':<8} {'Comp Rate':<12} {'INT Rate':<10}")
#     print("-" * 82)
    
#     for _, row in df_bins.iterrows():
#         print(f"{row['bin_center']:.2f}         {row['n_plays']:<10} {row['n_complete']:<10} "
#               f"{row['n_incomplete']:<10} {row['n_interception']:<8} "
#               f"{row['completion_rate']*100:>6.1f}%      {row['interception_rate']*100:>6.1f}%")
    
#     # Overall statistics
#     print("-" * 82)
#     total = len(df)
#     total_comp = (df['outcome'] == 'Complete').sum()
#     total_int = (df['outcome'] == 'Interception').sum()
#     print(f"{'TOTAL':<12} {total:<10} {total_comp:<10} {total - total_comp - total_int:<10} "
#           f"{total_int:<8} {total_comp/total*100:>6.1f}%      {total_int/total*100:>6.1f}%")
    
#     return df_bins


# df_bin_stats = compute_completion_rates_by_bin(df_landing_control, n_bins=10)


# # ============================================================================
# # STEP 3: FIT CALIBRATION CURVE
# # ============================================================================

# def fit_calibration_curve(df, df_bins):
#     """
#     Fit a smooth function mapping control_ratio â†’ P(completion).
    
#     Tries multiple approaches:
#     1. Logistic regression on raw data
#     2. Weighted polynomial fit on bin centers
    
#     Returns calibration function and fit statistics.
#     """
#     print("\n" + "-" * 70)
#     print("FITTING CALIBRATION CURVE")
#     print("-" * 70)
    
#     # Prepare data
#     X = df['control_ratio'].values
#     y = (df['outcome'] == 'Complete').astype(int).values
    
#     # Approach 1: Logistic Regression
#     print("\n  [1] Logistic Regression...")
    
#     from scipy.optimize import curve_fit
#     from scipy.special import expit  # Logistic function
    
#     def logistic(x, L, k, x0):
#         """Logistic function: L / (1 + exp(-k*(x-x0)))"""
#         return L * expit(k * (x - x0))
    
#     def logistic_simple(x, a, b):
#         """Simple logistic: 1 / (1 + exp(-(a + b*x)))"""
#         return expit(a + b * x)
    
#     try:
#         # Fit simple logistic
#         popt_simple, _ = curve_fit(logistic_simple, X, y, p0=[0, 2], maxfev=5000)
#         a_fit, b_fit = popt_simple
        
#         y_pred_logistic = logistic_simple(X, a_fit, b_fit)
        
#         # Compute metrics
#         from sklearn.metrics import brier_score_loss, log_loss
#         brier = brier_score_loss(y, y_pred_logistic)
#         logloss = log_loss(y, np.clip(y_pred_logistic, 1e-10, 1-1e-10))
        
#         print(f"      Fitted: P(complete) = 1 / (1 + exp(-({a_fit:.3f} + {b_fit:.3f} * control_ratio)))")
#         print(f"      Brier Score: {brier:.4f} (lower is better, 0.25 = random)")
#         print(f"      Log Loss: {logloss:.4f}")
        
#         logistic_fit = {'a': a_fit, 'b': b_fit, 'brier': brier, 'logloss': logloss}
        
#     except Exception as e:
#         print(f"      Failed: {e}")
#         logistic_fit = None
    
#     # Approach 2: Polynomial fit on bin centers (weighted by sample size)
#     print("\n  [2] Weighted Polynomial Fit on Bins...")
    
#     try:
#         bin_x = df_bins['bin_center'].values
#         bin_y = df_bins['completion_rate'].values
#         bin_weights = df_bins['n_plays'].values
        
#         # Fit polynomial (degree 2 or 3)
#         for degree in [2, 3]:
#             coeffs = np.polyfit(bin_x, bin_y, degree, w=np.sqrt(bin_weights))
#             poly = np.poly1d(coeffs)
            
#             y_pred_poly = poly(bin_x)
#             residuals = bin_y - y_pred_poly
#             weighted_mse = np.average(residuals**2, weights=bin_weights)
            
#             print(f"      Degree {degree}: Weighted MSE = {weighted_mse:.6f}")
        
#         # Use degree 3 for flexibility
#         coeffs_final = np.polyfit(bin_x, bin_y, 3, w=np.sqrt(bin_weights))
#         poly_fit = np.poly1d(coeffs_final)
        
#         print(f"      Using degree 3 polynomial")
        
#     except Exception as e:
#         print(f"      Failed: {e}")
#         poly_fit = None
    
#     # Approach 3: Isotonic regression (monotonic, non-parametric)
#     print("\n  [3] Isotonic Regression (monotonic)...")
    
#     try:
#         from sklearn.isotonic import IsotonicRegression
        
#         iso_reg = IsotonicRegression(y_min=0, y_max=1, increasing=True)
#         iso_reg.fit(X, y)
        
#         y_pred_iso = iso_reg.predict(X)
#         brier_iso = brier_score_loss(y, y_pred_iso)
        
#         print(f"      Brier Score: {brier_iso:.4f}")
        
#         isotonic_fit = iso_reg
        
#     except ImportError:
#         print("      sklearn not available, skipping")
#         isotonic_fit = None
#     except Exception as e:
#         print(f"      Failed: {e}")
#         isotonic_fit = None
    
#     return {
#         'logistic': logistic_fit,
#         'polynomial': poly_fit,
#         'isotonic': isotonic_fit,
#         'raw_data': (X, y),
#         'bin_data': (df_bins['bin_center'].values, df_bins['completion_rate'].values)
#     }


# calibration_fits = fit_calibration_curve(df_landing_control, df_bin_stats)


# # ============================================================================
# # STEP 4: CREATE CALIBRATION FUNCTION
# # ============================================================================

# def create_calibration_function(fits, method='logistic'):
#     """
#     Create a function that maps control_ratio â†’ P(completion).
    
#     Parameters:
#         fits: Output from fit_calibration_curve
#         method: 'logistic', 'polynomial', or 'isotonic'
        
#     Returns:
#         Callable: control_ratio â†’ P(completion)
#     """
#     if method == 'logistic' and fits['logistic']:
#         a, b = fits['logistic']['a'], fits['logistic']['b']
        
#         def calibration_func(control_ratio):
#             """Map control_ratio to completion probability (logistic)."""
#             from scipy.special import expit
#             return expit(a + b * np.asarray(control_ratio))
        
#         return calibration_func
    
#     elif method == 'polynomial' and fits['polynomial']:
#         poly = fits['polynomial']
        
#         def calibration_func(control_ratio):
#             """Map control_ratio to completion probability (polynomial)."""
#             p = poly(np.asarray(control_ratio))
#             return np.clip(p, 0, 1)  # Ensure valid probability
        
#         return calibration_func
    
#     elif method == 'isotonic' and fits['isotonic']:
#         iso = fits['isotonic']
        
#         def calibration_func(control_ratio):
#             """Map control_ratio to completion probability (isotonic)."""
#             return iso.predict(np.asarray(control_ratio).reshape(-1, 1) if np.asarray(control_ratio).ndim == 0 else np.asarray(control_ratio))
        
#         return calibration_func
    
#     else:
#         # Fallback: linear interpolation from bins
#         bin_x, bin_y = fits['bin_data']
        
#         def calibration_func(control_ratio):
#             """Map control_ratio to completion probability (linear interp)."""
#             return np.interp(control_ratio, bin_x, bin_y)
        
#         return calibration_func


# # Create the primary calibration function (logistic)
# get_completion_probability = create_calibration_function(calibration_fits, method='logistic')

# print("\n" + "-" * 70)
# print("CALIBRATION FUNCTION CREATED")
# print("-" * 70)
# print("\n  Usage: p_complete = get_completion_probability(control_ratio)")
# print("\n  Example values:")
# for cr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#     p = get_completion_probability(cr)
#     print(f"    control_ratio = {cr:.1f} â†’ P(complete) = {p:.1%}")


# # ============================================================================
# # STEP 5: VISUALIZE CALIBRATION CURVE
# # ============================================================================

# def plot_calibration_curve(df, df_bins, fits, save_path=None):
#     """
#     Visualize the calibration curve with empirical data.
#     """
#     print("\n" + "-" * 70)
#     print("PLOTTING CALIBRATION CURVE")
#     print("-" * 70)
    
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
#     # Left plot: Calibration curve with bins
#     ax1 = axes[0]
    
#     # Plot bin completion rates with error bars
#     bin_x = df_bins['bin_center'].values
#     bin_y = df_bins['completion_rate'].values
#     bin_se = df_bins['standard_error'].values
#     bin_n = df_bins['n_plays'].values
    
#     # Size points by sample size
#     sizes = 50 + 200 * (bin_n / bin_n.max())
    
#     ax1.errorbar(bin_x, bin_y, yerr=1.96*bin_se, fmt='o', markersize=8,
#                  capsize=5, capthick=2, color='blue', alpha=0.7,
#                  label='Empirical (95% CI)')
    
#     # Plot fitted curves
#     x_smooth = np.linspace(0, 1, 100)
    
#     if fits['logistic']:
#         a, b = fits['logistic']['a'], fits['logistic']['b']
#         from scipy.special import expit
#         y_logistic = expit(a + b * x_smooth)
#         ax1.plot(x_smooth, y_logistic, 'r-', linewidth=2.5, 
#                  label=f'Logistic fit (Brier={fits["logistic"]["brier"]:.3f})')
    
#     if fits['polynomial']:
#         y_poly = np.clip(fits['polynomial'](x_smooth), 0, 1)
#         ax1.plot(x_smooth, y_poly, 'g--', linewidth=2, alpha=0.7,
#                  label='Polynomial fit')
    
#     # Reference lines
#     ax1.plot([0, 1], [0, 1], 'k:', alpha=0.3, label='Perfect calibration')
#     ax1.axhline(df['outcome'].eq('Complete').mean(), color='gray', linestyle='--', 
#                 alpha=0.5, label=f'Base rate ({df["outcome"].eq("Complete").mean():.1%})')
    
#     ax1.set_xlabel('Control Ratio (Offense / Total)', fontsize=12)
#     ax1.set_ylabel('Completion Probability', fontsize=12)
#     ax1.set_title('Calibration Curve: Control Ratio â†’ P(Completion)', fontsize=13, fontweight='bold')
#     ax1.legend(loc='lower right', fontsize=9)
#     ax1.set_xlim(-0.02, 1.02)
#     ax1.set_ylim(-0.02, 1.02)
#     ax1.grid(True, alpha=0.3)
    
#     # Right plot: Distribution of control ratios by outcome
#     ax2 = axes[1]
    
#     complete = df[df['outcome'] == 'Complete']['control_ratio']
#     incomplete = df[df['outcome'] == 'Incomplete']['control_ratio']
#     interception = df[df['outcome'] == 'Interception']['control_ratio']
    
#     bins_hist = np.linspace(0, 1, 21)
    
#     ax2.hist(complete, bins=bins_hist, alpha=0.5, label=f'Complete (n={len(complete):,})', 
#              color='green', density=True)
#     ax2.hist(incomplete, bins=bins_hist, alpha=0.5, label=f'Incomplete (n={len(incomplete):,})', 
#              color='orange', density=True)
#     ax2.hist(interception, bins=bins_hist, alpha=0.5, label=f'Interception (n={len(interception):,})', 
#              color='red', density=True)
    
#     ax2.axvline(complete.mean(), color='green', linestyle='-', linewidth=2, alpha=0.8)
#     ax2.axvline(incomplete.mean(), color='orange', linestyle='-', linewidth=2, alpha=0.8)
#     ax2.axvline(interception.mean(), color='red', linestyle='-', linewidth=2, alpha=0.8)
    
#     ax2.set_xlabel('Control Ratio', fontsize=12)
#     ax2.set_ylabel('Density', fontsize=12)
#     ax2.set_title('Control Ratio Distribution by Outcome', fontsize=13, fontweight='bold')
#     ax2.legend(loc='upper left', fontsize=9)
#     ax2.set_xlim(-0.02, 1.02)
#     ax2.grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
#         print(f"\n  âœ“ Saved to {save_path}")
    
#     plt.show()
    
#     # Print summary statistics
#     print("\n  SUMMARY STATISTICS")
#     print(f"  {'Outcome':<15} {'Mean Control':<15} {'Median':<15} {'Std Dev':<15}")
#     print("  " + "-" * 60)
#     for outcome, data in [('Complete', complete), ('Incomplete', incomplete), ('Interception', interception)]:
#         print(f"  {outcome:<15} {data.mean():<15.3f} {data.median():<15.3f} {data.std():<15.3f}")
    
#     return fig


# fig = plot_calibration_curve(df_landing_control, df_bin_stats, calibration_fits,
#                              save_path='/mnt/user-data/outputs/calibration_curve.png')


# # ============================================================================
# # STEP 6: VALIDATE CALIBRATION
# # ============================================================================

# def validate_calibration(df, calibration_func):
#     """
#     Validate calibration by checking if predicted probabilities match observed frequencies.
#     """
#     print("\n" + "-" * 70)
#     print("CALIBRATION VALIDATION")
#     print("-" * 70)
    
#     df = df.copy()
    
#     # Get predicted probabilities
#     df['p_complete_predicted'] = calibration_func(df['control_ratio'].values)
#     df['is_complete'] = (df['outcome'] == 'Complete').astype(int)
    
#     # Bin by predicted probability and check actual completion rate
#     df['pred_bin'] = pd.cut(df['p_complete_predicted'], bins=10, labels=False)
    
#     print("\n  Predicted vs Actual Completion Rates:")
#     print(f"  {'Pred P(comp)':<15} {'Actual Rate':<15} {'N Plays':<10} {'Calibration':<15}")
#     print("  " + "-" * 55)
    
#     calibration_errors = []
    
#     for bin_idx in range(10):
#         bin_data = df[df['pred_bin'] == bin_idx]
#         if len(bin_data) < 5:
#             continue
        
#         pred_mean = bin_data['p_complete_predicted'].mean()
#         actual_rate = bin_data['is_complete'].mean()
#         n = len(bin_data)
        
#         error = actual_rate - pred_mean
#         calibration_errors.append(error)
        
#         status = "âœ“" if abs(error) < 0.05 else ("âš ï¸�" if abs(error) < 0.10 else "â�Œ")
        
#         print(f"  {pred_mean:<15.1%} {actual_rate:<15.1%} {n:<10} {error:+.1%} {status}")
    
#     # Overall calibration metrics
#     mean_calibration_error = np.mean(np.abs(calibration_errors))
    
#     print(f"\n  Mean Absolute Calibration Error: {mean_calibration_error:.1%}")
    
#     if mean_calibration_error < 0.03:
#         print("  âœ… Excellent calibration")
#     elif mean_calibration_error < 0.05:
#         print("  âœ“ Good calibration")
#     elif mean_calibration_error < 0.08:
#         print("  âš ï¸� Moderate calibration - consider adjustments")
#     else:
#         print("  â�Œ Poor calibration - model may need revision")
    
#     return df


# df_validated = validate_calibration(df_landing_control, get_completion_probability)


# # ============================================================================
# # SUMMARY
# # ============================================================================

# print("\n" + "=" * 70)
# print("CALIBRATION COMPLETE")
# print("=" * 70)

# print("""
# WHAT WE BUILT:
#   â€¢ Extracted control_ratio at ball landing point for all plays
#   â€¢ Binned by control_ratio and computed empirical completion rates
#   â€¢ Fitted logistic calibration curve
#   â€¢ Validated calibration accuracy

# KEY FUNCTION:
#   p_complete = get_completion_probability(control_ratio)
  
#   Maps control_ratio (0-1) to completion probability (0-1)

# INTERPRETATION:
#   â€¢ Control ratio captures meaningful signal about completion probability
#   â€¢ The relationship is monotonic (higher control â†’ higher completion)
#   â€¢ The calibration allows us to convert spatial control to expected outcomes

# NEXT STEPS:
#   â€¢ Use calibrated probabilities for Value-Weighted Openness
#   â€¢ Compare to your completion cloud approach
#   â€¢ Analyze QB decision quality using calibrated probabilities
# """)

# print("=" * 70)


# Cell: Calibration Curve (PRE-COMPUTED)
# ============================================================================
# Calibration parameters computed from fit_calibration_curve() on full dataset
# To regenerate, run the original calibration code
# ============================================================================

from scipy.special import expit
import numpy as np

print("=" * 70)
print("CALIBRATION CURVE: CONTROL RATIO â†’ COMPLETION PROBABILITY")
print("=" * 70)

# ============================================================================
# PRE-COMPUTED LOGISTIC FIT PARAMETERS
# ============================================================================

CALIBRATION_A = -0.448
CALIBRATION_B = 2.320
CALIBRATION_BRIER = 0.1893

print(f"""
LOGISTIC CALIBRATION FIT
----------------------------------------------------------------------
  Formula: P(complete) = 1 / (1 + exp(-({CALIBRATION_A:.3f} + {CALIBRATION_B:.3f} Ã— control_ratio)))
  
  Parameters:
    a = {CALIBRATION_A:.4f}
    b = {CALIBRATION_B:.4f}
    
  Fit Quality:
    Brier Score: {CALIBRATION_BRIER:.4f} (lower is better, 0.25 = random)
""")

# ============================================================================
# CALIBRATION FUNCTION
# ============================================================================

def get_completion_probability(control_ratio):
    """
    Map control_ratio to completion probability using pre-fitted logistic curve.
    
    Parameters:
        control_ratio: float or array, values between 0-1
        
    Returns:
        Completion probability (0-1)
    """
    return expit(CALIBRATION_A + CALIBRATION_B * np.asarray(control_ratio))


# ============================================================================
# PRE-COMPUTED RESULTS
# ============================================================================

print("""
----------------------------------------------------------------------
COMPLETION RATES BY CONTROL RATIO BIN
----------------------------------------------------------------------

Bin          N Plays    Complete   Incomp     INT      Comp Rate    INT Rate  
----------------------------------------------------------------------------------
0.05         1446       538        799        109        37.2%         7.5%
0.15         656        392        223        41         59.8%         6.2%
0.25         960        542        376        42         56.5%         4.4%
0.35         1356       846        469        41         62.4%         3.0%
0.45         1895       1110       730        55         58.6%         2.9%
0.55         1392       920        450        22         66.1%         1.6%
0.65         939        709        218        12         75.5%         1.3%
0.75         703        540        159        4          76.8%         0.6%
0.85         783        627        152        4          80.1%         0.5%
0.95         3978       3514       456        8          88.3%         0.2%
----------------------------------------------------------------------------------
TOTAL        14108      9738       4032       338        69.0%         2.4%

----------------------------------------------------------------------
EXAMPLE VALUES
----------------------------------------------------------------------

  Usage: p_complete = get_completion_probability(control_ratio)

    control_ratio = 0.1 â†’ P(complete) = 44.6%
    control_ratio = 0.2 â†’ P(complete) = 50.4%
    control_ratio = 0.3 â†’ P(complete) = 56.2%
    control_ratio = 0.4 â†’ P(complete) = 61.8%
    control_ratio = 0.5 â†’ P(complete) = 67.1%
    control_ratio = 0.6 â†’ P(complete) = 72.0%
    control_ratio = 0.7 â†’ P(complete) = 76.4%
    control_ratio = 0.8 â†’ P(complete) = 80.3%
    control_ratio = 0.9 â†’ P(complete) = 83.8%
""")

print("=" * 70)
print("âœ“ CALIBRATION FUNCTION READY")
print("=" * 70)


# """
# This was an interesting analysis of empiric completion probability clouds, but not included in the final report due to time constraints. 
# It has been commented out.
# NFL Big Data Bowl 2026 - Pitch Control Analysis
# ================================================================
# Completion Cloud: Empirical Outcome Distribution in Receiver Space

# This analysis builds a spatial map of where completions, incompletions,
# and interceptions occur relative to the receiver's position and direction
# at pass release.

# Approach:
# 1. For each play, get targeted receiver's position/direction at pass release
# 2. Transform ball landing point to receiver-relative coordinates
#    - Receiver at origin (0, 0)
#    - Receiver's facing direction = +X axis
# 3. Bin by receiver's initial velocity, flight time (same bins as movement cloud)
# 4. Plot all ball landing points, colored by outcome
# 5. Build density maps showing where each outcome type occurs

# This creates an empirical "where do completions happen?" map that can be
# compared to the movement cloud to understand the gap between physical
# capability and realized outcomes.
# ================================================================
# """

# print("=" * 70)
# print("COMPLETION CLOUD: EMPIRICAL OUTCOME DISTRIBUTION")
# print("=" * 70)

# # ============================================================================
# # STEP 1: EXTRACT BALL LANDING IN RECEIVER-RELATIVE COORDINATES
# # ============================================================================

# def extract_receiver_relative_landings(verbose=True):
#     """
#     For each play, transform the ball landing point to receiver-relative
#     coordinates (receiver at origin, facing +X).
    
#     Returns DataFrame with:
#     - game_id, play_id
#     - Receiver state at pass release (position, velocity, direction)
#     - Ball landing in receiver-relative coordinates (dx_norm, dy_norm)
#     - Outcome (Complete/Incomplete/Interception)
#     - Flight time, velocity bin, time bin
#     """
#     print("\nExtracting ball landing points in receiver-relative coordinates...")
    
#     # Get plays with outcomes
#     plays_with_outcome = play_metadata[
#         play_metadata['outcome'].isin(['Complete', 'Incomplete', 'Interception'])
#     ].copy()
    
#     n_plays = len(plays_with_outcome)
#     print(f"  â€¢ Processing {n_plays:,} plays")
    
#     results = []
#     errors = {'no_targeted': 0, 'no_output': 0, 'no_ball_landing': 0, 'other': 0}
    
#     for idx, (_, play) in enumerate(plays_with_outcome.iterrows()):
#         game_id = play['game_id']
#         play_id = play['play_id']
#         outcome = play['outcome']
        
#         try:
#             # Get INPUT data
#             input_play = df_input[
#                 (df_input['game_id'] == game_id) & 
#                 (df_input['play_id'] == play_id)
#             ]
            
#             if len(input_play) == 0:
#                 errors['no_output'] += 1
#                 continue
            
#             # Get ball landing point (normalized coordinates)
#             ball_x = input_play['ball_land_x_norm'].iloc[0]
#             ball_y = input_play['ball_land_y_norm'].iloc[0]
            
#             if pd.isna(ball_x) or pd.isna(ball_y):
#                 errors['no_ball_landing'] += 1
#                 continue
            
#             # Get targeted receiver from INPUT (last frame)
#             targeted = input_play[input_play['player_role'] == 'Targeted Receiver']
            
#             if len(targeted) == 0:
#                 errors['no_targeted'] += 1
#                 continue
            
#             # Get receiver's state at pass release (last INPUT frame)
#             receiver_final = targeted.sort_values('frame_id').iloc[-1]
            
#             receiver_x = receiver_final['x_norm']
#             receiver_y = receiver_final['y_norm']
#             receiver_speed = receiver_final['s']
#             receiver_dir = receiver_final['dir_norm']  # Already in math convention
            
#             if pd.isna(receiver_x) or pd.isna(receiver_y):
#                 errors['other'] += 1
#                 continue
            
#             if pd.isna(receiver_speed):
#                 receiver_speed = 0
#             if pd.isna(receiver_dir):
#                 receiver_dir = 0
            
#             # Calculate displacement from receiver to ball landing
#             dx = ball_x - receiver_x
#             dy = ball_y - receiver_y
            
#             # Distance from receiver to ball landing
#             landing_distance = np.sqrt(dx**2 + dy**2)
            
#             # Rotate to receiver-relative frame (receiver facing = +X)
#             # This uses the same transformation as the movement cloud
#             theta = np.radians(receiver_dir)
#             cos_t = np.cos(-theta)
#             sin_t = np.sin(-theta)
            
#             dx_norm = dx * cos_t - dy * sin_t
#             dy_norm = dx * sin_t + dy * cos_t
            
#             # Get QB position for angle calculation
#             qb_data = input_play[input_play['player_role'] == 'Passer']
#             if len(qb_data) > 0:
#                 qb_final = qb_data.sort_values('frame_id').iloc[-1]
#                 qb_x = qb_final['x_norm']
#                 qb_y = qb_final['y_norm']
                
#                 # Angle from receiver to QB (for binning)
#                 qb_dx = qb_x - receiver_x
#                 qb_dy = qb_y - receiver_y
#                 qb_direction = np.degrees(np.arctan2(qb_dy, qb_dx))
#                 angle_to_qb = abs(((qb_direction - receiver_dir) + 180) % 360 - 180)
#             else:
#                 angle_to_qb = 90  # Default
            
#             # Get flight time
#             flight_time, _ = get_ball_flight_time(game_id, play_id)
#             if flight_time is None:
#                 flight_time = 1.5
            
#             # Assign bins (same as movement cloud)
#             velocity_bin = get_velocity_bin(receiver_speed)
#             time_bin = get_time_bin(flight_time)
            
#             results.append({
#                 'game_id': game_id,
#                 'play_id': play_id,
#                 'outcome': outcome,
#                 # Receiver state
#                 'receiver_x': receiver_x,
#                 'receiver_y': receiver_y,
#                 'receiver_speed': receiver_speed,
#                 'receiver_dir': receiver_dir,
#                 'angle_to_qb': angle_to_qb,
#                 # Ball landing (field coordinates)
#                 'ball_x': ball_x,
#                 'ball_y': ball_y,
#                 # Ball landing (receiver-relative)
#                 'dx_norm': dx_norm,
#                 'dy_norm': dy_norm,
#                 'landing_distance': landing_distance,
#                 # Bins
#                 'velocity_bin': velocity_bin,
#                 'time_bin': time_bin,
#                 'flight_time': flight_time
#             })
            
#         except Exception as e:
#             errors['other'] += 1
#             continue
        
#         if verbose and (idx + 1) % 1000 == 0:
#             print(f"    {idx + 1:,}/{n_plays:,} plays processed...")
    
#     df_results = pd.DataFrame(results)
    
#     print(f"\n  âœ“ Extracted {len(df_results):,} plays")
#     print(f"  â€¢ Completions: {(df_results['outcome'] == 'Complete').sum():,}")
#     print(f"  â€¢ Incompletions: {(df_results['outcome'] == 'Incomplete').sum():,}")
#     print(f"  â€¢ Interceptions: {(df_results['outcome'] == 'Interception').sum():,}")
#     print(f"\n  Errors: {errors}")
    
#     return df_results


# # Extract data
# df_completion_cloud = extract_receiver_relative_landings(verbose=True)


# # ============================================================================
# # STEP 2: SUMMARY STATISTICS
# # ============================================================================

# def summarize_landing_distributions(df):
#     """
#     Summarize where ball landing points occur relative to receivers.
#     """
#     print("\n" + "-" * 70)
#     print("BALL LANDING DISTRIBUTION SUMMARY")
#     print("-" * 70)
    
#     print(f"\n  Overall Landing Statistics (receiver-relative coordinates):")
#     print(f"  {'Metric':<25} {'All':<12} {'Complete':<12} {'Incomplete':<12} {'INT':<12}")
#     print("  " + "-" * 63)
    
#     for metric, col in [('Mean X (forward)', 'dx_norm'), 
#                         ('Mean Y (lateral)', 'dy_norm'),
#                         ('Mean Distance', 'landing_distance')]:
#         all_val = df[col].mean()
#         comp_val = df[df['outcome'] == 'Complete'][col].mean()
#         inc_val = df[df['outcome'] == 'Incomplete'][col].mean()
#         int_val = df[df['outcome'] == 'Interception'][col].mean()
        
#         print(f"  {metric:<25} {all_val:<12.2f} {comp_val:<12.2f} {inc_val:<12.2f} {int_val:<12.2f}")
    
#     print(f"\n  Standard Deviations:")
#     for metric, col in [('Std X (forward)', 'dx_norm'), 
#                         ('Std Y (lateral)', 'dy_norm')]:
#         all_val = df[col].std()
#         comp_val = df[df['outcome'] == 'Complete'][col].std()
#         inc_val = df[df['outcome'] == 'Incomplete'][col].std()
#         int_val = df[df['outcome'] == 'Interception'][col].std()
        
#         print(f"  {metric:<25} {all_val:<12.2f} {comp_val:<12.2f} {inc_val:<12.2f} {int_val:<12.2f}")
    
#     # Percentiles for landing distance
#     print(f"\n  Landing Distance Percentiles:")
#     print(f"  {'Percentile':<15} {'Complete':<12} {'Incomplete':<12} {'INT':<12}")
#     print("  " + "-" * 51)
    
#     for pct in [25, 50, 75, 90, 95]:
#         comp_val = df[df['outcome'] == 'Complete']['landing_distance'].quantile(pct/100)
#         inc_val = df[df['outcome'] == 'Incomplete']['landing_distance'].quantile(pct/100)
#         int_val = df[df['outcome'] == 'Interception']['landing_distance'].quantile(pct/100)
        
#         print(f"  {pct}th percentile    {comp_val:<12.2f} {inc_val:<12.2f} {int_val:<12.2f}")


# summarize_landing_distributions(df_completion_cloud)


# # ============================================================================
# # STEP 3: BUILD COMPLETION CLOUD BY BINS
# # ============================================================================

# def build_completion_clouds(df, grid_extent=15, grid_resolution=0.5):
#     """
#     Build spatial outcome distributions for each velocity Ã— time bin.
    
#     For each bin, creates:
#     - 2D density of completions
#     - 2D density of incompletions
#     - 2D density of interceptions
#     - Completion rate at each grid point
    
#     Returns dict of completion clouds keyed by (velocity_bin, time_bin)
#     """
#     print("\n" + "-" * 70)
#     print("BUILDING COMPLETION CLOUDS BY BIN")
#     print("-" * 70)
    
#     # Create grid (same as movement cloud)
#     grid_1d = np.arange(-grid_extent, grid_extent + grid_resolution, grid_resolution)
#     grid_x, grid_y = np.meshgrid(grid_1d, grid_1d)
    
#     print(f"\n  Grid: {len(grid_1d)} Ã— {len(grid_1d)} = {len(grid_1d)**2} points")
#     print(f"  Extent: Â±{grid_extent} yards, Resolution: {grid_resolution} yards")
    
#     completion_clouds = {}
    
#     # Process each velocity Ã— time bin
#     for velocity_bin in VELOCITY_BINS.keys():
#         for time_bin in TIME_BINS:
            
#             # Filter to this bin
#             mask = (
#                 (df['velocity_bin'] == velocity_bin) &
#                 (df['time_bin'] == time_bin)
#             )
#             bin_data = df[mask]
#             n_total = len(bin_data)
            
#             if n_total < 20:  # Need minimum samples
#                 continue
            
#             n_complete = (bin_data['outcome'] == 'Complete').sum()
#             n_incomplete = (bin_data['outcome'] == 'Incomplete').sum()
#             n_int = (bin_data['outcome'] == 'Interception').sum()
            
#             # Get landing points by outcome
#             complete_points = bin_data[bin_data['outcome'] == 'Complete'][['dx_norm', 'dy_norm']].values
#             incomplete_points = bin_data[bin_data['outcome'] == 'Incomplete'][['dx_norm', 'dy_norm']].values
#             int_points = bin_data[bin_data['outcome'] == 'Interception'][['dx_norm', 'dy_norm']].values
            
#             # Build KDE for each outcome type (if enough samples)
#             positions = np.vstack([grid_x.ravel(), grid_y.ravel()])
            
#             # Completions
#             if len(complete_points) >= 10:
#                 try:
#                     kde_complete = stats.gaussian_kde(complete_points.T)
#                     density_complete = kde_complete(positions).reshape(grid_x.shape)
#                 except:
#                     density_complete = np.zeros_like(grid_x)
#             else:
#                 density_complete = np.zeros_like(grid_x)
            
#             # Incompletions
#             if len(incomplete_points) >= 10:
#                 try:
#                     kde_incomplete = stats.gaussian_kde(incomplete_points.T)
#                     density_incomplete = kde_incomplete(positions).reshape(grid_x.shape)
#                 except:
#                     density_incomplete = np.zeros_like(grid_x)
#             else:
#                 density_incomplete = np.zeros_like(grid_x)
            
#             # Interceptions (often sparse)
#             if len(int_points) >= 5:
#                 try:
#                     kde_int = stats.gaussian_kde(int_points.T)
#                     density_int = kde_int(positions).reshape(grid_x.shape)
#                 except:
#                     density_int = np.zeros_like(grid_x)
#             else:
#                 density_int = np.zeros_like(grid_x)
            
#             # Normalize each density to sum to 1 (probability distribution)
#             if density_complete.sum() > 0:
#                 density_complete = density_complete / density_complete.sum()
#             if density_incomplete.sum() > 0:
#                 density_incomplete = density_incomplete / density_incomplete.sum()
#             if density_int.sum() > 0:
#                 density_int = density_int / density_int.sum()
            
#             # Calculate completion rate at each point
#             # Weight densities by outcome counts
#             total_density = (n_complete * density_complete + 
#                            n_incomplete * density_incomplete + 
#                            n_int * density_int)
            
#             # Completion rate = P(complete | ball lands here)
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 completion_rate = np.where(
#                     total_density > 1e-10,
#                     (n_complete * density_complete) / total_density,
#                     0.5  # Default to 50% where no data
#                 )
            
#             # Store
#             key = (velocity_bin, time_bin)
#             completion_clouds[key] = {
#                 'grid_x': grid_1d.copy(),
#                 'grid_y': grid_1d.copy(),
#                 'density_complete': density_complete,
#                 'density_incomplete': density_incomplete,
#                 'density_interception': density_int,
#                 'completion_rate': completion_rate,
#                 'n_total': n_total,
#                 'n_complete': n_complete,
#                 'n_incomplete': n_incomplete,
#                 'n_interception': n_int,
#                 'overall_completion_rate': n_complete / n_total,
#                 # Raw points for plotting
#                 'points_complete': complete_points,
#                 'points_incomplete': incomplete_points,
#                 'points_interception': int_points,
#                 # Statistics
#                 'mean_landing_x': bin_data['dx_norm'].mean(),
#                 'mean_landing_y': bin_data['dy_norm'].mean(),
#                 'mean_landing_distance': bin_data['landing_distance'].mean()
#             }
    
#     print(f"\n  âœ“ Built {len(completion_clouds)} completion clouds")
    
#     # Summary table
#     print(f"\n  {'Velocity':<12} {'Time':<8} {'N Total':<10} {'Complete':<10} {'Comp Rate':<10}")
#     print("  " + "-" * 50)
    
#     for vel_bin in ['stationary', 'jogging', 'running', 'fast', 'sprinting']:
#         for time_bin in [0.5, 1.0, 1.5]:
#             key = (vel_bin, time_bin)
#             if key in completion_clouds:
#                 c = completion_clouds[key]
#                 print(f"  {vel_bin:<12} {time_bin:<8.1f} {c['n_total']:<10} "
#                       f"{c['n_complete']:<10} {c['overall_completion_rate']*100:>6.1f}%")
    
#     return completion_clouds


# completion_clouds = build_completion_clouds(df_completion_cloud)


# # ============================================================================
# # STEP 4: VISUALIZE COMPLETION CLOUD
# # ============================================================================

# def plot_completion_cloud(cloud, velocity_bin, time_bin, show_points=True, 
#                           figsize=(16, 5), save_path=None):
#     """
#     Visualize a single completion cloud.
    
#     Shows:
#     1. Raw scatter of landing points by outcome
#     2. Completion rate heatmap
#     3. Density comparison (completions vs incompletions)
#     """
#     if cloud is None:
#         print(f"  No cloud available for {velocity_bin}, {time_bin}s")
#         return None
    
#     fig, axes = plt.subplots(1, 3, figsize=figsize)
    
#     grid_1d = cloud['grid_x']
#     grid_x, grid_y = np.meshgrid(grid_1d, grid_1d)
    
#     # ========================================================================
#     # Plot 1: Raw scatter of landing points
#     # ========================================================================
#     ax1 = axes[0]
    
#     if show_points:
#         # Plot in order: INT (back), Incomplete (middle), Complete (front)
#         if len(cloud['points_interception']) > 0:
#             ax1.scatter(cloud['points_interception'][:, 0], 
#                        cloud['points_interception'][:, 1],
#                        c='red', alpha=0.6, s=30, label=f"INT (n={cloud['n_interception']})")
        
#         if len(cloud['points_incomplete']) > 0:
#             ax1.scatter(cloud['points_incomplete'][:, 0], 
#                        cloud['points_incomplete'][:, 1],
#                        c='orange', alpha=0.4, s=20, label=f"Incomplete (n={cloud['n_incomplete']})")
        
#         if len(cloud['points_complete']) > 0:
#             ax1.scatter(cloud['points_complete'][:, 0], 
#                        cloud['points_complete'][:, 1],
#                        c='green', alpha=0.4, s=20, label=f"Complete (n={cloud['n_complete']})")
    
#     # Mark receiver position (origin)
#     ax1.scatter([0], [0], c='blue', s=200, marker='o', edgecolors='white', 
#                 linewidths=2, zorder=10, label='Receiver')
#     ax1.arrow(0, 0, 3, 0, head_width=0.5, head_length=0.3, fc='blue', ec='blue',
#               zorder=9, alpha=0.7)
    
#     ax1.axhline(0, color='gray', linestyle='--', alpha=0.3)
#     ax1.axvline(0, color='gray', linestyle='--', alpha=0.3)
    
#     ax1.set_xlim(-15, 15)
#     ax1.set_ylim(-15, 15)
#     ax1.set_aspect('equal')
#     ax1.set_xlabel('Forward (yards) â†’', fontsize=10)
#     ax1.set_ylabel('â†� Left    Right â†’', fontsize=10)
#     ax1.set_title(f'Ball Landing Points\n{velocity_bin}, {time_bin}s flight', fontsize=11)
#     ax1.legend(loc='upper right', fontsize=8)
#     ax1.grid(True, alpha=0.3)
    
#     # ========================================================================
#     # Plot 2: Completion rate heatmap
#     # ========================================================================
#     ax2 = axes[1]
    
#     # Custom colormap: red (0%) -> yellow (50%) -> green (100%)
#     cmap = plt.cm.RdYlGn
    
#     im = ax2.imshow(cloud['completion_rate'], 
#                     extent=[-15, 15, -15, 15],
#                     origin='lower', 
#                     cmap=cmap, 
#                     vmin=0, vmax=1,
#                     aspect='equal',
#                     alpha=0.8)
    
#     # Receiver marker
#     ax2.scatter([0], [0], c='blue', s=200, marker='o', edgecolors='white', 
#                 linewidths=2, zorder=10)
#     ax2.arrow(0, 0, 3, 0, head_width=0.5, head_length=0.3, fc='blue', ec='blue',
#               zorder=9, alpha=0.7)
    
#     # Contour lines at key thresholds
#     contour_levels = [0.3, 0.5, 0.7]
#     cs = ax2.contour(grid_x, grid_y, cloud['completion_rate'], 
#                      levels=contour_levels, colors='black', linewidths=1, alpha=0.5)
#     ax2.clabel(cs, inline=True, fontsize=8, fmt='%.0f%%')
    
#     ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)
#     ax2.axvline(0, color='gray', linestyle='--', alpha=0.3)
    
#     ax2.set_xlim(-15, 15)
#     ax2.set_ylim(-15, 15)
#     ax2.set_xlabel('Forward (yards) â†’', fontsize=10)
#     ax2.set_ylabel('â†� Left    Right â†’', fontsize=10)
#     ax2.set_title(f'Completion Rate by Location\nOverall: {cloud["overall_completion_rate"]*100:.1f}%', fontsize=11)
    
#     cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
#     cbar.set_label('Completion Rate', fontsize=9)
    
#     # ========================================================================
#     # Plot 3: Density comparison
#     # ========================================================================
#     ax3 = axes[2]
    
#     # Show completion density minus incompletion density
#     # Green = more completions, Red = more incompletions
#     density_diff = cloud['density_complete'] - cloud['density_incomplete']
#     max_abs = max(abs(density_diff.min()), abs(density_diff.max()), 1e-6)
    
#     im3 = ax3.imshow(density_diff, 
#                      extent=[-15, 15, -15, 15],
#                      origin='lower', 
#                      cmap='RdYlGn',
#                      vmin=-max_abs, vmax=max_abs,
#                      aspect='equal',
#                      alpha=0.8)
    
#     # Receiver marker
#     ax3.scatter([0], [0], c='blue', s=200, marker='o', edgecolors='white', 
#                 linewidths=2, zorder=10)
#     ax3.arrow(0, 0, 3, 0, head_width=0.5, head_length=0.3, fc='blue', ec='blue',
#               zorder=9, alpha=0.7)
    
#     ax3.axhline(0, color='gray', linestyle='--', alpha=0.3)
#     ax3.axvline(0, color='gray', linestyle='--', alpha=0.3)
    
#     ax3.set_xlim(-15, 15)
#     ax3.set_ylim(-15, 15)
#     ax3.set_xlabel('Forward (yards) â†’', fontsize=10)
#     ax3.set_ylabel('â†� Left    Right â†’', fontsize=10)
#     ax3.set_title('Density Difference\n(Green = more completions)', fontsize=11)
    
#     cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
#     cbar3.set_label('Complete - Incomplete Density', fontsize=9)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
#         print(f"  âœ“ Saved to {save_path}")
    
#     plt.show()
    
#     return fig


# # Plot a few example clouds
# print("\n" + "-" * 70)
# print("EXAMPLE COMPLETION CLOUDS")
# print("-" * 70)

# # Running receiver, medium flight time
# if ('running', 1.0) in completion_clouds:
#     print("\n  Example 1: Running receiver, 1.0s flight time")
#     plot_completion_cloud(completion_clouds[('running', 1.0)], 'running', 1.0)

# # Fast receiver, shorter flight
# if ('fast', 0.8) in completion_clouds:
#     print("\n  Example 2: Fast receiver, 0.8s flight time")
#     plot_completion_cloud(completion_clouds[('fast', 0.8)], 'fast', 0.8)


# # ============================================================================
# # STEP 5: COMPARE COMPLETION CLOUD TO MOVEMENT CLOUD
# # ============================================================================

# def compare_clouds(completion_cloud, movement_dist, velocity_bin, time_bin,
#                    figsize=(14, 6), save_path=None):
#     """
#     Compare completion cloud to movement cloud for same bin.
    
#     Shows where receivers physically go vs where completions happen.
#     """
#     if completion_cloud is None or movement_dist is None:
#         print(f"  Missing data for {velocity_bin}, {time_bin}s")
#         return None
    
#     fig, axes = plt.subplots(1, 2, figsize=figsize)
    
#     grid_1d = completion_cloud['grid_x']
#     grid_x, grid_y = np.meshgrid(grid_1d, grid_1d)
    
#     # ========================================================================
#     # Plot 1: Movement cloud (where receivers GO)
#     # ========================================================================
#     ax1 = axes[0]
    
#     # Get movement density on same grid
#     movement_grid = movement_dist['grid_x']
#     movement_density = movement_dist['density']
    
#     # Interpolate to completion cloud grid if needed
#     if len(movement_grid) != len(grid_1d):
#         from scipy.interpolate import RegularGridInterpolator
#         interp = RegularGridInterpolator(
#             (movement_grid, movement_grid), movement_density,
#             method='linear', bounds_error=False, fill_value=0
#         )
#         points = np.column_stack([grid_y.ravel(), grid_x.ravel()])
#         movement_on_grid = interp(points).reshape(grid_x.shape)
#     else:
#         movement_on_grid = movement_density
    
#     im1 = ax1.imshow(movement_on_grid, 
#                      extent=[-15, 15, -15, 15],
#                      origin='lower', 
#                      cmap='Blues',
#                      aspect='equal',
#                      alpha=0.8)
    
#     # p95 contour
#     threshold = compute_p95_threshold(movement_on_grid)
#     if threshold > 0:
#         ax1.contour(grid_x, grid_y, movement_on_grid, levels=[threshold],
#                    colors='blue', linewidths=2, linestyles='solid')
    
#     ax1.scatter([0], [0], c='blue', s=200, marker='o', edgecolors='white', 
#                 linewidths=2, zorder=10)
#     ax1.arrow(0, 0, 3, 0, head_width=0.5, head_length=0.3, fc='blue', ec='blue', zorder=9)
    
#     ax1.set_xlim(-15, 15)
#     ax1.set_ylim(-15, 15)
#     ax1.set_xlabel('Forward (yards) â†’', fontsize=10)
#     ax1.set_ylabel('â†� Left    Right â†’', fontsize=10)
#     ax1.set_title(f'Movement Cloud (where receivers GO)\n{velocity_bin}, {time_bin}s', fontsize=11)
    
#     # ========================================================================
#     # Plot 2: Completion cloud (where completions HAPPEN)
#     # ========================================================================
#     ax2 = axes[1]
    
#     # Use completion density (not rate)
#     im2 = ax2.imshow(completion_cloud['density_complete'], 
#                      extent=[-15, 15, -15, 15],
#                      origin='lower', 
#                      cmap='Greens',
#                      aspect='equal',
#                      alpha=0.8)
    
#     # p95 contour of completions
#     comp_threshold = compute_p95_threshold(completion_cloud['density_complete'])
#     if comp_threshold > 0:
#         ax2.contour(grid_x, grid_y, completion_cloud['density_complete'], 
#                    levels=[comp_threshold], colors='green', linewidths=2, linestyles='solid')
    
#     ax2.scatter([0], [0], c='blue', s=200, marker='o', edgecolors='white', 
#                 linewidths=2, zorder=10)
#     ax2.arrow(0, 0, 3, 0, head_width=0.5, head_length=0.3, fc='blue', ec='blue', zorder=9)
    
#     ax2.set_xlim(-15, 15)
#     ax2.set_ylim(-15, 15)
#     ax2.set_xlabel('Forward (yards) â†’', fontsize=10)
#     ax2.set_ylabel('â†� Left    Right â†’', fontsize=10)
#     ax2.set_title(f'Completion Cloud (where completions HAPPEN)\nn={completion_cloud["n_complete"]}', fontsize=11)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
#         print(f"  âœ“ Saved to {save_path}")
    
#     plt.show()
    
#     return fig


# # Compare clouds for running receiver
# print("\n" + "-" * 70)
# print("COMPARING MOVEMENT CLOUD VS COMPLETION CLOUD")
# print("-" * 70)

# for vel_bin, time_bin in [('running', 1.0), ('fast', 0.8), ('jogging', 0.5)]:
#     key = (vel_bin, time_bin)
#     if key in completion_clouds and key in distributions:
#         print(f"\n  {vel_bin}, {time_bin}s:")
#         compare_clouds(completion_clouds[key], distributions[key], vel_bin, time_bin)


# # ============================================================================
# # STEP 6: AGGREGATE STATISTICS
# # ============================================================================

# def analyze_completion_cloud_patterns(df, clouds):
#     """
#     Identify key patterns in completion clouds.
#     """
#     print("\n" + "-" * 70)
#     print("COMPLETION CLOUD PATTERNS")
#     print("-" * 70)
    
#     # Pattern 1: Leading distance (how far ahead do completions land?)
#     print("\n  LEADING DISTANCE (mean forward displacement by outcome):")
#     print(f"\n  {'Outcome':<15} {'Mean X':<10} {'Std X':<10} {'Interpretation'}")
#     print("  " + "-" * 55)
    
#     for outcome in ['Complete', 'Incomplete', 'Interception']:
#         subset = df[df['outcome'] == outcome]
#         mean_x = subset['dx_norm'].mean()
#         std_x = subset['dx_norm'].std()
        
#         if outcome == 'Complete':
#             interp = "Ideal leading distance"
#         elif outcome == 'Incomplete':
#             interp = "Under/overthrows"
#         else:
#             interp = "Defender territory"
        
#         print(f"  {outcome:<15} {mean_x:<10.2f} {std_x:<10.2f} {interp}")
    
#     # Pattern 2: Lateral accuracy
#     print("\n  LATERAL ACCURACY (displacement from receiver's path):")
#     print(f"\n  {'Outcome':<15} {'Mean |Y|':<12} {'Interpretation'}")
#     print("  " + "-" * 45)
    
#     for outcome in ['Complete', 'Incomplete', 'Interception']:
#         subset = df[df['outcome'] == outcome]
#         mean_abs_y = subset['dy_norm'].abs().mean()
        
#         if outcome == 'Complete':
#             interp = "On-target laterally"
#         elif outcome == 'Incomplete':
#             interp = "Off-target laterally"
#         else:
#             interp = "Defender position"
        
#         print(f"  {outcome:<15} {mean_abs_y:<12.2f} {interp}")
    
#     # Pattern 3: Completion rate by zone
#     print("\n  COMPLETION RATE BY ZONE:")
    
#     # Define zones in receiver-relative space
#     zones = [
#         ('Ahead & close (0-5 yd)', (0, 5, -3, 3)),
#         ('Ahead & far (5-10 yd)', (5, 10, -5, 5)),
#         ('Behind receiver', (-10, 0, -5, 5)),
#         ('Far left', (-5, 10, -10, -3)),
#         ('Far right', (-5, 10, 3, 10)),
#     ]
    
#     print(f"\n  {'Zone':<30} {'N Throws':<12} {'Comp Rate':<12}")
#     print("  " + "-" * 54)
    
#     for zone_name, (x_min, x_max, y_min, y_max) in zones:
#         zone_mask = (
#             (df['dx_norm'] >= x_min) & (df['dx_norm'] < x_max) &
#             (df['dy_norm'] >= y_min) & (df['dy_norm'] < y_max)
#         )
#         zone_data = df[zone_mask]
        
#         if len(zone_data) > 0:
#             comp_rate = (zone_data['outcome'] == 'Complete').mean()
#             print(f"  {zone_name:<30} {len(zone_data):<12} {comp_rate*100:>6.1f}%")


# analyze_completion_cloud_patterns(df_completion_cloud, completion_clouds)


# # ============================================================================
# # SUMMARY
# # ============================================================================

# print("\n" + "=" * 70)
# print("COMPLETION CLOUD ANALYSIS COMPLETE")
# print("=" * 70)

# print("""
# WHAT WE BUILT:
#   â€¢ Extracted ball landing points in receiver-relative coordinates
#   â€¢ Built completion clouds for each velocity Ã— time bin
#   â€¢ Visualized where completions, incompletions, and INTs occur
#   â€¢ Compared to movement clouds to see capability vs. outcomes

# KEY OBJECTS:
#   â€¢ df_completion_cloud: Raw data with receiver-relative landing points
#   â€¢ completion_clouds: Dict of spatial outcome distributions by bin
  
# KEY INSIGHTS:
#   â€¢ Completions cluster in a specific zone ahead of the receiver
#   â€¢ Incompletions are more spread out (under/overthrows)
#   â€¢ Interceptions tend to be in zones behind or lateral to receiver
#   â€¢ Movement cloud (capability) is larger than completion cloud (usage)

# COMPARISON TO CALIBRATION CURVE:
#   â€¢ Calibration curve: control_ratio â†’ P(completion) (single number)
#   â€¢ Completion cloud: spatial distribution of where outcomes occur
#   â€¢ Together: understand BOTH the probability AND the spatial patterns

# NEXT STEPS:
#   â€¢ Overlay completion cloud with movement cloud at same scale
#   â€¢ Identify "sweet spots" where completions concentrate
#   â€¢ Use both approaches for Value-Weighted Openness
# """)

# print("=" * 70)


# Cell: Completion Cloud Analysis (PRE-COMPUTED)
# ============================================================================
# Results computed from build_completion_clouds() on full dataset
# Shows empirical spatial distribution of where completions occur
# relative to receiver position and direction
# ============================================================================

print("=" * 70)
print("COMPLETION CLOUD: EMPIRICAL OUTCOME DISTRIBUTION (PRE-COMPUTED)")
print("=" * 70)

print("""
----------------------------------------------------------------------
BALL LANDING DISTRIBUTION SUMMARY
----------------------------------------------------------------------

  Overall Landing Statistics (receiver-relative coordinates):
  Metric                    All          Complete     Incomplete   INT         
  ---------------------------------------------------------------
  Mean X (forward)          5.86         4.66         8.56         8.06        
  Mean Y (lateral)          -0.02        -0.05        0.05         -0.11       
  Mean Distance             6.77         5.49         9.64         9.37        

  Standard Deviations:
  Std X (forward)           5.71         4.39         7.23         7.15        
  Std Y (lateral)           3.01         2.56         3.81         3.98        

  Landing Distance Percentiles:
  Percentile      Complete     Incomplete   INT         
  ---------------------------------------------------
  25th            2.76         4.57         4.95        
  50th            4.74         7.36         7.18        
  75th            6.82         13.41        12.82       
  90th            10.03        20.07        20.03       
  95th            13.47        23.91        23.43       

----------------------------------------------------------------------
COMPLETION CLOUDS BY BIN
----------------------------------------------------------------------

  Velocity     Time     N Total    Complete   Comp Rate 
  --------------------------------------------------
  stationary   1.0      204        168          82.4%
  jogging      0.5      27         22           81.5%
  jogging      1.0      839        676          80.6%
  running      0.5      46         42           91.3%
  running      1.0      1772       1383         78.0%
  fast         1.0      1649       1302         79.0%
  sprinting    1.0      321        255          79.4%

----------------------------------------------------------------------
COMPLETION CLOUD PATTERNS
----------------------------------------------------------------------

  LEADING DISTANCE (mean forward displacement by outcome):

  Outcome         Mean X     Std X      Interpretation
  -------------------------------------------------------
  Complete        4.66       4.39       Ideal leading distance
  Incomplete      8.56       7.23       Under/overthrows
  Interception    8.06       7.15       Defender territory

  LATERAL ACCURACY (displacement from receiver's path):

  Outcome         Mean |Y|     Interpretation
  ---------------------------------------------
  Complete        1.94         On-target laterally
  Incomplete      2.94         Off-target laterally
  Interception    3.17         Defender position

  COMPLETION RATE BY ZONE:

  Zone                           N Throws     Comp Rate   
  ------------------------------------------------------
  Ahead & close (0-5 yd)         5439           81.6%
  Ahead & far (5-10 yd)          3748           71.5%
  Behind receiver                1093           78.0%
  Far left                       1277           62.5%
  Far right                      1233           61.5%

----------------------------------------------------------------------
KEY INSIGHTS
----------------------------------------------------------------------

  â€¢ Completions land 4.66 yards ahead of receiver on average
  â€¢ Incompletions land 8.56 yards ahead - significant overthrows
  â€¢ Interceptions also occur deeper (8.06 yards) - into defender territory
  
  â€¢ Lateral accuracy is critical:
    - Completions: 1.94 yards off-center
    - Incompletions: 2.94 yards off-center (+51% worse)
    - Interceptions: 3.17 yards off-center (+63% worse)
  
  â€¢ "Sweet spot" is 0-5 yards ahead with 81.6% completion rate
  â€¢ Far lateral throws (left/right) drop to ~62% completion rate
  
  â€¢ The 95th percentile landing distance for completions (13.47 yards)
    is much smaller than for incompletions (23.91 yards), showing
    that successful passes stay within a tighter window.

""")

print("=" * 70)
print("âœ“ COMPLETION CLOUD SUMMARY READY")
print("=" * 70)


# """
# Original VWO cell
# NFL Big Data Bowl 2026 - Pitch Control Analysis
# ================================================================
# Value-Weighted Openness (VWO)

# Calculates the expected value of each receiver based on:
# - Where they can reach (movement cloud)
# - Probability of completion (control ratio â†’ calibration curve)
# - Value if completed (EPA)

# Uses LAST FRAME of INPUT data to include ALL receivers, not just
# the targeted receiver.

# Components:
# 1. Standard EP lookup table (by field position, down, distance)
# 2. EPA surface calculation for each receiver's cloud
# 3. VWO integration: âˆ«âˆ« P(reach) Ã— P(complete) Ã— EPA
# 4. Optimal target identification
# 5. QB decision quality metrics
# ================================================================
# """

# print("=" * 70)
# print("VALUE-WEIGHTED OPENNESS (VWO) ANALYSIS")
# print("=" * 70)

# # ============================================================================
# # EXPECTED POINTS (EP) LOOKUP TABLE
# # ============================================================================

# def build_ep_table():
#     """
#     Build standard Expected Points lookup table.
    
#     EP values based on NFL averages by field position, down, and distance.
#     These are well-established values from public EP models.
    
#     Returns:
#         Function that takes (field_pos, down, yards_to_go) and returns EP
#     """
#     print("\nBuilding Expected Points lookup table...")
    
#     # Base EP by field position (yards from own goal line, 0-100)
#     # Source: Standard NFL EP curves (approximation)
#     # More positive = better field position for offense
    
#     def base_ep_by_field_position(field_pos):
#         """
#         Base EP assuming 1st and 10.
        
#         Approximates standard NFL EP curve:
#         - Own goal line (0-5): Very negative (safety risk)
#         - Own 20: Slightly negative to neutral
#         - Midfield (50): ~2 points
#         - Opponent 20: ~4 points
#         - Opponent goal line (95-100): ~6 points
#         """
#         field_pos = np.clip(field_pos, 0, 100)
        
#         # Piecewise linear approximation of EP curve
#         if isinstance(field_pos, np.ndarray):
#             ep = np.zeros_like(field_pos, dtype=float)
            
#             # Own 0-10: Safety danger zone
#             mask = field_pos <= 10
#             ep[mask] = -2.0 + (field_pos[mask] / 10) * 1.5  # -2.0 to -0.5
            
#             # Own 10-25: Poor field position
#             mask = (field_pos > 10) & (field_pos <= 25)
#             ep[mask] = -0.5 + ((field_pos[mask] - 10) / 15) * 1.0  # -0.5 to 0.5
            
#             # Own 25-50: Neutral to positive
#             mask = (field_pos > 25) & (field_pos <= 50)
#             ep[mask] = 0.5 + ((field_pos[mask] - 25) / 25) * 1.5  # 0.5 to 2.0
            
#             # Opponent 50-75: Good field position
#             mask = (field_pos > 50) & (field_pos <= 75)
#             ep[mask] = 2.0 + ((field_pos[mask] - 50) / 25) * 2.0  # 2.0 to 4.0
            
#             # Opponent 25-95: Red zone approach
#             mask = (field_pos > 75) & (field_pos <= 95)
#             ep[mask] = 4.0 + ((field_pos[mask] - 75) / 20) * 2.0  # 4.0 to 6.0
            
#             # Opponent 5-yard line: Goal line
#             mask = field_pos > 95
#             ep[mask] = 6.0 + ((field_pos[mask] - 95) / 5) * 0.5  # 6.0 to 6.5
            
#             return ep
#         else:
#             # Scalar version
#             if field_pos <= 10:
#                 return -2.0 + (field_pos / 10) * 1.5
#             elif field_pos <= 25:
#                 return -0.5 + ((field_pos - 10) / 15) * 1.0
#             elif field_pos <= 50:
#                 return 0.5 + ((field_pos - 25) / 25) * 1.5
#             elif field_pos <= 75:
#                 return 2.0 + ((field_pos - 50) / 25) * 2.0
#             elif field_pos <= 95:
#                 return 4.0 + ((field_pos - 75) / 20) * 2.0
#             else:
#                 return 6.0 + ((field_pos - 95) / 5) * 0.5
    
#     def down_distance_adjustment(down, yards_to_go, field_pos):
#         """
#         Adjust EP based on down and distance.
        
#         Key factors:
#         - Later downs with more yards = lower EP (harder to convert)
#         - 4th down far from opponent = very low EP (likely punt)
#         - Short yardage = higher EP (likely conversion)
#         """
#         base = base_ep_by_field_position(field_pos)
        
#         # Ensure inputs are arrays for vectorized ops
#         down = np.atleast_1d(down)
#         yards_to_go = np.atleast_1d(yards_to_go)
#         field_pos = np.atleast_1d(field_pos)
#         base = np.atleast_1d(base)
        
#         adjustment = np.zeros_like(base, dtype=float)
        
#         # Down penalty (later downs = harder)
#         down_penalty = np.where(down == 1, 0.0,
#                        np.where(down == 2, -0.3,
#                        np.where(down == 3, -0.8, -2.0)))  # 4th down big penalty
        
#         # Distance penalty (more yards = harder)
#         distance_penalty = np.clip((yards_to_go - 5) * 0.05, 0, 1.0)
        
#         # Short yardage bonus
#         short_yardage_bonus = np.where(yards_to_go <= 2, 0.3,
#                               np.where(yards_to_go <= 4, 0.1, 0.0))
        
#         # 4th down adjustment: if not in FG range or short, likely punt
#         fourth_down_punt = np.where(
#             (down == 4) & (field_pos < 60) & (yards_to_go > 3),
#             -1.5,  # Punt expected, big EP hit
#             0.0
#         )
        
#         # 4th down in FG range
#         fourth_down_fg = np.where(
#             (down == 4) & (field_pos >= 60) & (field_pos < 95),
#             0.5,  # FG attempt gives ~1.5 points expected
#             0.0
#         )
        
#         adjustment = down_penalty - distance_penalty + short_yardage_bonus + fourth_down_punt + fourth_down_fg
        
#         result = base + adjustment
        
#         # Return scalar if single input
#         if len(result) == 1:
#             return float(result[0])
#         return result
    
#     def get_ep(field_pos, down, yards_to_go):
#         """
#         Get Expected Points for a given situation.
        
#         Parameters:
#             field_pos: Yards from own goal line (0-100)
#             down: Current down (1-4)
#             yards_to_go: Yards needed for first down
            
#         Returns:
#             Expected points from this situation
#         """
#         # Clip field position to valid range
#         field_pos = np.clip(field_pos, 0, 100)
        
#         # Clip yards to go
#         yards_to_go = np.clip(yards_to_go, 1, 40)
        
#         return down_distance_adjustment(down, yards_to_go, field_pos)
    
#     print("  âœ“ EP table built")
#     print("\n  Sample EP values:")
#     print(f"    Own 20, 1st & 10:  {get_ep(20, 1, 10):+.2f}")
#     print(f"    Own 20, 3rd & 15:  {get_ep(20, 3, 15):+.2f}")
#     print(f"    Midfield, 1st & 10: {get_ep(50, 1, 10):+.2f}")
#     print(f"    Midfield, 3rd & 8:  {get_ep(50, 3, 8):+.2f}")
#     print(f"    Opp 20, 1st & 10:  {get_ep(80, 1, 10):+.2f}")
#     print(f"    Opp 5, 1st & Goal:  {get_ep(95, 1, 5):+.2f}")
    
#     return get_ep


# # Build the EP function
# get_ep = build_ep_table()


# # ============================================================================
# # EPA CALCULATION FOR A GRID POINT
# # ============================================================================

# def calculate_epa_at_point(grid_x, los_x, down, yards_to_go):
#     """
#     Calculate EPA if ball is caught at a given field x-coordinate.
    
#     Parameters:
#         grid_x: Field x-coordinate(s) where ball would be caught (normalized)
#         los_x: Line of scrimmage x-coordinate (absolute_yardline_number)
#         down: Current down (1-4)
#         yards_to_go: Yards needed for first down
        
#     Returns:
#         EPA value(s) at each grid point
#     """
#     # Current field position (yards from own goal, 0-100 scale)
#     current_field_pos = los_x - 10  # Account for endzone offset
    
#     # Air yards (yards gained/lost if caught at grid_x)
#     air_yards = grid_x - los_x
    
#     # New field position after catch
#     new_field_pos = current_field_pos + air_yards
    
#     # Determine new down and distance
#     gains_first_down = air_yards >= yards_to_go
    
#     # Handle as arrays
#     air_yards = np.atleast_1d(air_yards)
#     gains_first_down = np.atleast_1d(gains_first_down)
#     new_field_pos = np.atleast_1d(new_field_pos)
    
#     new_down = np.where(gains_first_down, 1, np.minimum(down + 1, 4))
#     new_distance = np.where(
#         gains_first_down,
#         np.minimum(10, 100 - new_field_pos),  # 1st & 10, or goal to go
#         np.maximum(1, yards_to_go - air_yards)
#     )
    
#     # Handle touchdown (field_pos >= 100)
#     is_touchdown = new_field_pos >= 100
    
#     # EP before play
#     ep_before = get_ep(current_field_pos, down, yards_to_go)
    
#     # EP after catch (7 points for TD)
#     ep_after = np.where(
#         is_touchdown,
#         7.0,
#         get_ep(np.clip(new_field_pos, 0, 99), new_down, new_distance)
#     )
    
#     # EPA = EP_after - EP_before
#     epa = ep_after - ep_before
    
#     return epa


# # ============================================================================
# # GET LAST INPUT FRAME FOR A PLAY
# # ============================================================================

# def get_last_input_frame(game_id, play_id):
#     """
#     Get the last frame of INPUT data for a play (all players).
    
#     This is the moment of pass release, with all receivers and defenders.
    
#     Returns:
#         DataFrame of player positions/velocities at pass release
#     """
#     play_data = df_input[
#         (df_input['game_id'] == game_id) & 
#         (df_input['play_id'] == play_id)
#     ]
    
#     if len(play_data) == 0:
#         return None
    
#     # Get last frame
#     last_frame_id = play_data['frame_id'].max()
#     last_frame = play_data[play_data['frame_id'] == last_frame_id].copy()
    
#     return last_frame


# # ============================================================================
# # CALCULATE VWO FOR A SINGLE RECEIVER
# # ============================================================================

# def calculate_receiver_vwo(receiver_row, all_players_frame, los_x, down, yards_to_go,
#                            flight_time, grid_spacing=1.0, grid_extent=15):
#     """
#     Calculate Value-Weighted Openness for a single receiver.
    
#     VWO = âˆ«âˆ« P(reach) Ã— P(complete | control) Ã— EPA dx dy
    
#     Parameters:
#         receiver_row: Series with receiver's position/velocity
#         all_players_frame: DataFrame with all players (for pitch control)
#         los_x: Line of scrimmage (absolute_yardline_number)
#         down: Current down
#         yards_to_go: Yards to first down
#         flight_time: Expected ball flight time
#         grid_spacing: Resolution of integration grid
#         grid_extent: How far to extend grid from receiver
        
#     Returns:
#         Dict with VWO and component breakdowns
#     """
#     # Receiver state
#     receiver_x = receiver_row['x_norm']
#     receiver_y = receiver_row['y_norm']
#     receiver_speed = receiver_row['s'] if pd.notna(receiver_row['s']) else 0
#     receiver_dir = receiver_row['dir_norm'] if pd.notna(receiver_row['dir_norm']) else 0
    
#     # Get velocity and time bins
#     velocity_bin = get_velocity_bin(receiver_speed)
#     time_bin = get_time_bin(flight_time)
    
#     # Get movement distribution
#     if hasattr(get_player_distribution_interpolated, '__call__'):
#         dist, _, _, _ = get_player_distribution_interpolated(receiver_speed, flight_time)
#     else:
#         dist, _, _ = get_player_distribution(receiver_speed, flight_time)
    
#     if dist is None:
#         # No distribution available - use fallback
#         return {
#             'vwo': 0,
#             'total_reach_prob': 0,
#             'avg_completion_prob': 0,
#             'avg_epa': 0,
#             'sweet_spot_x': receiver_x,
#             'sweet_spot_y': receiver_y,
#             'max_vwo_density': 0,
#             'status': 'no_distribution'
#         }
    
#     # Create field grid centered on receiver
#     x_coords = np.arange(receiver_x - grid_extent, receiver_x + grid_extent + grid_spacing, grid_spacing)
#     y_coords = np.arange(
#         max(0, receiver_y - grid_extent),
#         min(53.3, receiver_y + grid_extent) + grid_spacing,
#         grid_spacing
#     )
#     field_grid_x, field_grid_y = np.meshgrid(x_coords, y_coords)
    
#     # Transform movement distribution to field coordinates
#     reach_density = transform_distribution_to_field(
#         dist, receiver_x, receiver_y, receiver_dir,
#         field_grid_x, field_grid_y
#     )
    
#     # Expand by catch radius
#     reach_density_expanded = expand_density_with_catch_radius(
#         reach_density, grid_spacing, CATCH_RADIUS
#     )
    
#     # Normalize to probability distribution
#     if reach_density_expanded.sum() > 0:
#         reach_prob = reach_density_expanded / reach_density_expanded.sum()
#     else:
#         reach_prob = reach_density_expanded
    
#     # Calculate pitch control at each grid point
#     # (This is computationally expensive - we're calculating full pitch control)
#     # For efficiency, we calculate control just for this receiver's grid
    
#     # Get all player densities on this grid
#     offense_density = np.zeros_like(field_grid_x, dtype=float)
#     defense_density = np.zeros_like(field_grid_x, dtype=float)
    
#     for _, player in all_players_frame.iterrows():
#         player_x = player['x_norm']
#         player_y = player['y_norm']
#         player_speed = player['s'] if pd.notna(player['s']) else 0
#         player_dir = player['dir_norm'] if pd.notna(player['dir_norm']) else 0
#         player_side = player['player_side']
        
#         if pd.isna(player_x) or pd.isna(player_y):
#             continue
#         if player_side not in ['Offense', 'Defense']:
#             continue
        
#         # Get player's distribution
#         if hasattr(get_player_distribution_interpolated, '__call__'):
#             p_dist, _, _, _ = get_player_distribution_interpolated(player_speed, flight_time)
#         else:
#             p_dist, _, _ = get_player_distribution(player_speed, flight_time)
        
#         if p_dist is not None:
#             player_density = transform_distribution_to_field(
#                 p_dist, player_x, player_y, player_dir,
#                 field_grid_x, field_grid_y
#             )
#         else:
#             player_density = create_fallback_distribution(
#                 player_x, player_y, player_speed, player_dir,
#                 flight_time, field_grid_x, field_grid_y
#             )
        
#         # Expand by catch radius
#         player_expanded = expand_density_with_catch_radius(
#             player_density, grid_spacing, CATCH_RADIUS
#         )
        
#         # Normalize per player
#         if player_expanded.max() > 0:
#             player_norm = player_expanded / player_expanded.max()
#         else:
#             player_norm = player_expanded
        
#         # Accumulate
#         if player_side == 'Offense':
#             offense_density += player_norm
#         else:
#             defense_density += player_norm
    
#     # Control ratio
#     total_density = offense_density + defense_density + CONTROL_EPSILON
#     control_ratio = np.clip(offense_density / total_density, 0, 1)
    
#     # Completion probability at each point (from calibration curve)
#     completion_prob = get_completion_probability(control_ratio)
    
#     # EPA at each point
#     epa_surface = calculate_epa_at_point(field_grid_x, los_x, down, yards_to_go)
    
#     # VWO density at each point
#     vwo_density = reach_prob * completion_prob * epa_surface
    
#     # Total VWO (integral)
#     total_vwo = np.sum(vwo_density) * (grid_spacing ** 2)
    
#     # Find sweet spot (max VWO density point)
#     max_idx = np.unravel_index(np.argmax(vwo_density), vwo_density.shape)
#     sweet_spot_x = field_grid_x[max_idx]
#     sweet_spot_y = field_grid_y[max_idx]
#     max_vwo_density = vwo_density[max_idx]
    
#     # Component averages (weighted by reach probability)
#     avg_completion_prob = np.sum(reach_prob * completion_prob)
#     avg_epa = np.sum(reach_prob * epa_surface)
#     total_reach_prob = reach_prob.sum()
    
#     return {
#         'vwo': total_vwo,
#         'total_reach_prob': total_reach_prob,
#         'avg_completion_prob': avg_completion_prob,
#         'avg_epa': avg_epa,
#         'sweet_spot_x': sweet_spot_x,
#         'sweet_spot_y': sweet_spot_y,
#         'max_vwo_density': max_vwo_density,
#         'velocity_bin': velocity_bin,
#         'time_bin': time_bin,
#         # Store surfaces for visualization
#         'grid_x': field_grid_x,
#         'grid_y': field_grid_y,
#         'reach_prob': reach_prob,
#         'completion_prob': completion_prob,
#         'epa_surface': epa_surface,
#         'vwo_density': vwo_density,
#         'control_ratio': control_ratio,
#         'status': 'success'
#     }


# # ============================================================================
# # CALCULATE VWO FOR ALL RECEIVERS ON A PLAY
# # ============================================================================

# def calculate_play_vwo(game_id, play_id, verbose=False):
#     """
#     Calculate VWO for all eligible receivers on a play.
    
#     Returns:
#         Dict with:
#         - All receivers' VWO
#         - Optimal target
#         - Actual target
#         - Decision quality metrics
#     """
#     # Get play metadata
#     play_data = df_input[
#         (df_input['game_id'] == game_id) & 
#         (df_input['play_id'] == play_id)
#     ]
    
#     if len(play_data) == 0:
#         return None
    
#     # Get situation
#     los_x = play_data['absolute_yardline_number'].iloc[0]
#     down = play_data['down'].iloc[0]
#     yards_to_go = play_data['yards_to_go'].iloc[0]
#     pass_result = play_data['pass_result'].iloc[0]
    
#     # Get last INPUT frame (all players)
#     last_frame = get_last_input_frame(game_id, play_id)
#     if last_frame is None:
#         return None
    
#     # Get flight time (add one frame since we're using INPUT not OUTPUT)
#     base_flight_time, _ = get_ball_flight_time(game_id, play_id)
#     if base_flight_time is None:
#         base_flight_time = 1.5
#     flight_time = base_flight_time + (1 / FRAME_RATE)  # Add one frame
    
#     # Identify receivers
#     receivers = last_frame[
#         last_frame['player_role'].isin(['Targeted Receiver', 'Other Route Runner'])
#     ]
    
#     if len(receivers) == 0:
#         return None
    
#     # Calculate VWO for each receiver
#     receiver_vwos = []
    
#     for _, receiver in receivers.iterrows():
#         vwo_result = calculate_receiver_vwo(
#             receiver, last_frame, los_x, down, yards_to_go, flight_time,
#             grid_spacing=1.0, grid_extent=12
#         )
        
#         vwo_result['nfl_id'] = receiver['nfl_id']
#         vwo_result['player_name'] = receiver.get('player_name', 'Unknown')
#         vwo_result['player_role'] = receiver['player_role']
#         vwo_result['receiver_x'] = receiver['x_norm']
#         vwo_result['receiver_y'] = receiver['y_norm']
#         vwo_result['receiver_speed'] = receiver['s']
        
#         receiver_vwos.append(vwo_result)
    
#     # Find optimal and actual targets
#     vwo_df = pd.DataFrame([{
#         'nfl_id': r['nfl_id'],
#         'player_name': r['player_name'],
#         'player_role': r['player_role'],
#         'vwo': r['vwo'],
#         'avg_completion_prob': r['avg_completion_prob'],
#         'avg_epa': r['avg_epa'],
#         'sweet_spot_x': r['sweet_spot_x'],
#         'sweet_spot_y': r['sweet_spot_y']
#     } for r in receiver_vwos if r['status'] == 'success'])
    
#     if len(vwo_df) == 0:
#         return None
    
#     # Optimal target = highest VWO
#     optimal_idx = vwo_df['vwo'].idxmax()
#     optimal_target = vwo_df.loc[optimal_idx]
    
#     # Actual target
#     actual_target = vwo_df[vwo_df['player_role'] == 'Targeted Receiver']
    
#     if len(actual_target) == 0:
#         actual_vwo = 0
#         target_quality = 0
#     else:
#         actual_target = actual_target.iloc[0]
#         actual_vwo = actual_target['vwo']
        
#         # Target selection quality
#         if optimal_target['vwo'] > 0:
#             target_quality = actual_vwo / optimal_target['vwo']
#         else:
#             target_quality = 1.0 if actual_vwo == 0 else 0
    
#     if verbose:
#         print(f"\n  Play {game_id}-{play_id}: {down}&{yards_to_go} at {los_x}")
#         print(f"  Receivers analyzed: {len(vwo_df)}")
#         print(f"  Optimal: {optimal_target['player_name']} (VWO={optimal_target['vwo']:.3f})")
#         if len(actual_target) > 0:
#             print(f"  Actual:  {actual_target['player_name']} (VWO={actual_vwo:.3f})")
#         print(f"  Target Quality: {target_quality:.2%}")
    
#     return {
#         'game_id': game_id,
#         'play_id': play_id,
#         'down': down,
#         'yards_to_go': yards_to_go,
#         'los_x': los_x,
#         'pass_result': pass_result,
#         'flight_time': flight_time,
#         'n_receivers': len(vwo_df),
#         'receiver_vwos': receiver_vwos,
#         'vwo_summary': vwo_df,
#         'optimal_target_id': optimal_target['nfl_id'],
#         'optimal_target_name': optimal_target['player_name'],
#         'optimal_vwo': optimal_target['vwo'],
#         'actual_target_id': actual_target['nfl_id'] if len(actual_target) > 0 else None,
#         'actual_target_name': actual_target['player_name'] if len(actual_target) > 0 else None,
#         'actual_vwo': actual_vwo,
#         'target_quality': target_quality,
#         'chose_optimal': optimal_target['nfl_id'] == (actual_target['nfl_id'] if len(actual_target) > 0 else None)
#     }


# # ============================================================================
# # BATCH ANALYSIS
# # ============================================================================

# def analyze_vwo_batch(sample_size=100, verbose=True):
#     """
#     Analyze VWO for a batch of plays.
    
#     Returns DataFrame with decision quality metrics.
#     """
#     print("\n" + "-" * 70)
#     print(f"ANALYZING VWO FOR {sample_size} PLAYS")
#     print("-" * 70)
    
#     # Get unique plays
#     plays = df_input.groupby(['game_id', 'play_id']).first().reset_index()
    
#     # Sample
#     if sample_size and sample_size < len(plays):
#         plays = plays.sample(sample_size, random_state=42)
    
#     results = []
#     errors = 0
    
#     for idx, (_, play) in enumerate(plays.iterrows()):
#         game_id = play['game_id']
#         play_id = play['play_id']
        
#         try:
#             result = calculate_play_vwo(game_id, play_id, verbose=False)
            
#             if result is not None:
#                 results.append({
#                     'game_id': result['game_id'],
#                     'play_id': result['play_id'],
#                     'down': result['down'],
#                     'yards_to_go': result['yards_to_go'],
#                     'pass_result': result['pass_result'],
#                     'n_receivers': result['n_receivers'],
#                     'optimal_vwo': result['optimal_vwo'],
#                     'actual_vwo': result['actual_vwo'],
#                     'target_quality': result['target_quality'],
#                     'chose_optimal': result['chose_optimal'],
#                     'optimal_target_name': result['optimal_target_name'],
#                     'actual_target_name': result['actual_target_name']
#                 })
#         except Exception as e:
#             errors += 1
#             continue
        
#         if verbose and (idx + 1) % 20 == 0:
#             print(f"  Processed {idx + 1}/{len(plays)} plays...")
    
#     df_results = pd.DataFrame(results)
    
#     print(f"\n  âœ“ Analyzed {len(df_results)} plays ({errors} errors)")
    
#     return df_results


# # ============================================================================
# # SUMMARY STATISTICS
# # ============================================================================

# def summarize_vwo_analysis(df_results):
#     """
#     Summarize VWO analysis results.
#     """
#     print("\n" + "=" * 70)
#     print("VWO ANALYSIS SUMMARY")
#     print("=" * 70)
    
#     n_plays = len(df_results)
    
#     # Overall decision quality
#     print(f"\n  OVERALL METRICS ({n_plays} plays)")
#     print(f"  " + "-" * 40)
#     print(f"  Mean Target Quality:     {df_results['target_quality'].mean():.1%}")
#     print(f"  Median Target Quality:   {df_results['target_quality'].median():.1%}")
#     print(f"  Chose Optimal Target:    {df_results['chose_optimal'].mean():.1%}")
    
#     # VWO statistics
#     print(f"\n  VWO STATISTICS")
#     print(f"  " + "-" * 40)
#     print(f"  Mean Optimal VWO:        {df_results['optimal_vwo'].mean():.3f}")
#     print(f"  Mean Actual VWO:         {df_results['actual_vwo'].mean():.3f}")
#     print(f"  VWO Gap (left on table): {(df_results['optimal_vwo'] - df_results['actual_vwo']).mean():.3f}")
    
#     # By down
#     print(f"\n  TARGET QUALITY BY DOWN")
#     print(f"  " + "-" * 40)
#     for down in [1, 2, 3, 4]:
#         down_data = df_results[df_results['down'] == down]
#         if len(down_data) > 5:
#             print(f"  {down}st/nd/rd down: {down_data['target_quality'].mean():.1%} "
#                   f"(n={len(down_data)}, chose optimal: {down_data['chose_optimal'].mean():.1%})")
    
#     # By outcome
#     print(f"\n  TARGET QUALITY BY OUTCOME")
#     print(f"  " + "-" * 40)
#     for outcome in ['C', 'I', 'IN']:
#         outcome_data = df_results[df_results['pass_result'] == outcome]
#         if len(outcome_data) > 5:
#             outcome_name = {'C': 'Complete', 'I': 'Incomplete', 'IN': 'Interception'}[outcome]
#             print(f"  {outcome_name}: {outcome_data['target_quality'].mean():.1%} "
#                   f"(n={len(outcome_data)}, mean actual VWO: {outcome_data['actual_vwo'].mean():.3f})")
    
#     # Distribution of target quality
#     print(f"\n  TARGET QUALITY DISTRIBUTION")
#     print(f"  " + "-" * 40)
#     bins = [0, 0.5, 0.7, 0.85, 0.95, 1.0, float('inf')]
#     labels = ['<50%', '50-70%', '70-85%', '85-95%', '95-100%', '>100%']
    
#     df_results['quality_bin'] = pd.cut(df_results['target_quality'], bins=bins, labels=labels)
#     for label in labels:
#         count = (df_results['quality_bin'] == label).sum()
#         pct = count / len(df_results) * 100
#         print(f"  {label:>10}: {count:>5} plays ({pct:>5.1f}%)")
    
#     return df_results


# # ============================================================================
# # VISUALIZATION
# # ============================================================================

# def visualize_play_vwo(game_id, play_id, save_path=None):
#     """
#     Visualize VWO analysis for a single play.
    
#     Shows:
#     - Field with all receivers
#     - VWO surface for each receiver
#     - Optimal vs actual target
#     """
#     result = calculate_play_vwo(game_id, play_id, verbose=True)
    
#     if result is None:
#         print("  Could not analyze this play")
#         return None
    
#     # Get receivers with full VWO data
#     receivers = [r for r in result['receiver_vwos'] if r['status'] == 'success']
#     n_receivers = len(receivers)
    
#     if n_receivers == 0:
#         print("  No receivers with valid VWO")
#         return None
    
#     # Create figure
#     fig, axes = plt.subplots(1, min(n_receivers + 1, 4), figsize=(5 * min(n_receivers + 1, 4), 5))
#     if n_receivers == 0:
#         axes = [axes]
#     elif n_receivers < 3:
#         axes = list(axes)
    
#     # Sort receivers by VWO
#     receivers = sorted(receivers, key=lambda x: x['vwo'], reverse=True)
    
#     # Plot each receiver's VWO surface
#     for i, receiver in enumerate(receivers[:3]):  # Max 3 receivers
#         ax = axes[i]
        
#         vwo_density = receiver['vwo_density']
#         grid_x = receiver['grid_x']
#         grid_y = receiver['grid_y']
        
#         # Plot VWO surface
#         im = ax.imshow(
#             vwo_density,
#             extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
#             origin='lower',
#             cmap='YlOrRd',
#             aspect='equal'
#         )
        
#         # Mark receiver position
#         ax.scatter(receiver['receiver_x'], receiver['receiver_y'],
#                    c='blue', s=150, marker='o', edgecolors='white', linewidths=2, zorder=10)
        
#         # Mark sweet spot
#         ax.scatter(receiver['sweet_spot_x'], receiver['sweet_spot_y'],
#                    c='gold', s=100, marker='*', edgecolors='black', linewidths=1, zorder=11)
        
#         # Mark LOS
#         ax.axvline(result['los_x'], color='yellow', linestyle='--', alpha=0.7)
        
#         # Title
#         is_optimal = receiver['nfl_id'] == result['optimal_target_id']
#         is_actual = receiver['nfl_id'] == result['actual_target_id']
        
#         title = f"{receiver['player_name']}\nVWO: {receiver['vwo']:.3f}"
#         if is_optimal:
#             title += " â˜…OPTIMAL"
#         if is_actual:
#             title += " â†�ACTUAL"
        
#         ax.set_title(title, fontsize=10)
#         ax.set_xlabel('Field X')
#         ax.set_ylabel('Field Y')
        
#         plt.colorbar(im, ax=ax, shrink=0.6, label='VWO Density')
    
#     # Summary panel
#     if n_receivers < 3:
#         ax_summary = axes[-1]
#     else:
#         ax_summary = axes[3] if len(axes) > 3 else None
    
#     if ax_summary:
#         ax_summary.axis('off')
        
#         summary_text = f"""
# PLAY SUMMARY
# {result['down']}{'st' if result['down']==1 else 'nd' if result['down']==2 else 'rd' if result['down']==3 else 'th'} & {result['yards_to_go']}
# LOS: {result['los_x']}

# DECISION ANALYSIS
# Optimal: {result['optimal_target_name']}
#   VWO: {result['optimal_vwo']:.3f}
  
# Actual: {result['actual_target_name']}
#   VWO: {result['actual_vwo']:.3f}

# Target Quality: {result['target_quality']:.1%}
# Result: {result['pass_result']}
#         """
#         ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
#                        fontsize=10, verticalalignment='top', fontfamily='monospace')
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
#         print(f"  âœ“ Saved to {save_path}")
    
#     plt.show()
    
#     return result


# # ============================================================================
# # RUN ANALYSIS
# # ============================================================================

# print("\n" + "=" * 70)
# print("VWO SYSTEM READY")
# print("=" * 70)

# print("""
# AVAILABLE FUNCTIONS:

#   calculate_play_vwo(game_id, play_id)
#       Calculate VWO for all receivers on a play
#       Returns optimal target, actual target, decision quality
      
#   analyze_vwo_batch(sample_size=100)
#       Analyze multiple plays, return summary DataFrame
      
#   visualize_play_vwo(game_id, play_id)
#       Visual analysis of VWO for a single play
      
#   summarize_vwo_analysis(df_results)
#       Summary statistics from batch analysis

# QUICK START:
#   # Single play analysis
#   result = calculate_play_vwo(game_id, play_id, verbose=True)
  
#   # Batch analysis
#   df_vwo = analyze_vwo_batch(sample_size=100)
#   summarize_vwo_analysis(df_vwo)
  
#   # Visualization
#   visualize_play_vwo(game_id, play_id)
# """)

# print("=" * 70)


# """
# Version 2.0 VWO
# NFL Big Data Bowl 2026 - Pitch Control Analysis
# ================================================================
# VWO Version 2: Three-Outcome Expected Value with Spatial Metrics

# Improvements over V1:
# 1. Three-outcome expected value: Complete, Incomplete, Interception
# 2. Situation-dependent EPA for incomplete passes
# 3. Field-position dependent EPA for interceptions
# 4. Spatial openness metrics (Open Area, Danger Area, Window Ratio)
# 5. Risk-adjusted VWO with lambda parameter

# E[Value] = P(complete) Ã— EPA_complete 
#          + P(incomplete) Ã— EPA_incomplete 
#          + P(INT) Ã— EPA_INT

# Risk_Adjusted_VWO = VWO - Î» Ã— Danger_Area
# ================================================================
# """

# print("=" * 70)
# print("VWO V2: THREE-OUTCOME EXPECTED VALUE WITH SPATIAL METRICS")
# print("=" * 70)


# # ============================================================================
# # SECTION A: INTERCEPTION PROBABILITY CALIBRATION
# # ============================================================================

# print("\n" + "-" * 70)
# print("SECTION A: INTERCEPTION PROBABILITY CALIBRATION")
# print("-" * 70)

# def build_interception_calibration():
#     """
#     Build empirical calibration curve for P(INT | control_ratio).
#     Same approach as completion calibration.
#     """
#     print("\n  Building interception probability calibration...")
    
#     # Get plays with outcomes
#     plays_with_outcome = df_input.groupby(['game_id', 'play_id']).first().reset_index()
#     plays_with_outcome = plays_with_outcome[
#         plays_with_outcome['pass_result'].isin(['C', 'I', 'IN'])
#     ]
    
#     results = []
    
#     for idx, row in plays_with_outcome.iterrows():
#         game_id = row['game_id']
#         play_id = row['play_id']
#         outcome = row['pass_result']
        
#         try:
#             # Get ball landing point
#             ball_x = row['ball_land_x_norm']
#             ball_y = row['ball_land_y_norm']
            
#             if pd.isna(ball_x) or pd.isna(ball_y):
#                 continue
            
#             # Get last INPUT frame for pitch control calculation
#             last_frame = get_last_input_frame(game_id, play_id)
#             if last_frame is None:
#                 continue
            
#             # Get flight time
#             flight_time, _ = get_ball_flight_time(game_id, play_id)
#             if flight_time is None:
#                 flight_time = 1.5
#             flight_time += (1 / FRAME_RATE)  # Add one frame for INPUT
            
#             # Calculate pitch control at ball landing point
#             control_at_landing = calculate_control_at_point(
#                 ball_x, ball_y, last_frame, flight_time
#             )
            
#             results.append({
#                 'game_id': game_id,
#                 'play_id': play_id,
#                 'outcome': outcome,
#                 'control_ratio': control_at_landing,
#                 'is_complete': 1 if outcome == 'C' else 0,
#                 'is_int': 1 if outcome == 'IN' else 0
#             })
            
#         except Exception as e:
#             continue
    
#     df_calib = pd.DataFrame(results)
#     print(f"  Calibration data: {len(df_calib)} plays")
    
#     return df_calib


# def calculate_control_at_point(target_x, target_y, players_frame, flight_time):
#     """
#     Calculate pitch control at a specific point.
#     Simplified version for calibration.
#     """
#     offense_density = 0.0
#     defense_density = 0.0
    
#     for _, player in players_frame.iterrows():
#         player_x = player['x_norm']
#         player_y = player['y_norm']
#         player_speed = player['s'] if pd.notna(player['s']) else 0
#         player_dir = player['dir_norm'] if pd.notna(player['dir_norm']) else 0
#         player_side = player['player_side']
        
#         if pd.isna(player_x) or pd.isna(player_y):
#             continue
#         if player_side not in ['Offense', 'Defense']:
#             continue
        
#         # Simple distance-based density (for speed in calibration)
#         dist = np.sqrt((target_x - player_x)**2 + (target_y - player_y)**2)
        
#         # Estimate reachable distance based on speed and time
#         max_reach = player_speed * flight_time + CATCH_RADIUS + 2.0  # Buffer
        
#         if dist <= max_reach:
#             # Higher density for closer players
#             density = np.exp(-dist / (max_reach / 2))
#         else:
#             density = 0.0
        
#         if player_side == 'Offense':
#             offense_density += density
#         else:
#             defense_density += density
    
#     total = offense_density + defense_density + CONTROL_EPSILON
#     control_ratio = offense_density / total
    
#     return np.clip(control_ratio, 0, 1)


# # Check if we already have calibration data, if not build it
# if 'df_int_calibration' not in dir():
#     print("  Building new calibration data...")
#     df_int_calibration = build_interception_calibration()
# else:
#     print("  Using existing calibration data")


# # ============================================================================
# # SECTION B: ANALYZE INTERCEPTION RATES AND FIND DANGER THRESHOLD
# # ============================================================================

# print("\n" + "-" * 70)
# print("SECTION B: EMPIRICAL INTERCEPTION RATES BY CONTROL RATIO")
# print("-" * 70)

# def analyze_int_rates(df_calib, n_bins=20):
#     """
#     Analyze interception rates by control ratio bins.
#     Find empirical danger threshold.
#     """
#     # Use quantile bins for better distribution
#     df_calib['control_bin'] = pd.qcut(
#         df_calib['control_ratio'], 
#         q=n_bins, 
#         duplicates='drop'
#     )
    
#     # Calculate rates by bin
#     bin_stats = df_calib.groupby('control_bin').agg({
#         'is_complete': ['sum', 'count', 'mean'],
#         'is_int': ['sum', 'mean'],
#         'control_ratio': 'mean'
#     }).reset_index()
    
#     bin_stats.columns = ['bin', 'completions', 'n_plays', 'completion_rate', 
#                          'interceptions', 'int_rate', 'mean_control']
    
#     print(f"\n  {'Control Range':<20} {'N Plays':<10} {'Comp Rate':<12} {'INT Rate':<12}")
#     print("  " + "-" * 54)
    
#     for _, row in bin_stats.iterrows():
#         bin_str = f"{row['bin']}"[:18]
#         print(f"  {bin_str:<20} {int(row['n_plays']):<10} "
#               f"{row['completion_rate']*100:>6.1f}%     {row['int_rate']*100:>6.2f}%")
    
#     # Find danger threshold (where INT rate exceeds 2x baseline)
#     baseline_int_rate = df_calib['is_int'].mean()
#     print(f"\n  Baseline INT rate: {baseline_int_rate*100:.2f}%")
    
#     # Find threshold where INT rate > 2x baseline
#     danger_threshold = None
#     for _, row in bin_stats.sort_values('mean_control', ascending=False).iterrows():
#         if row['int_rate'] > 2 * baseline_int_rate:
#             danger_threshold = row['mean_control']
    
#     # Also find where INT rate > 5%
#     high_danger_threshold = None
#     for _, row in bin_stats.sort_values('mean_control', ascending=False).iterrows():
#         if row['int_rate'] > 0.05:
#             high_danger_threshold = row['mean_control']
    
#     print(f"\n  Danger threshold (INT > 2x baseline): {danger_threshold:.2f}" if danger_threshold else "")
#     print(f"  High danger threshold (INT > 5%): {high_danger_threshold:.2f}" if high_danger_threshold else "")
    
#     return bin_stats, baseline_int_rate, danger_threshold


# bin_stats, baseline_int_rate, empirical_danger_threshold = analyze_int_rates(df_int_calibration)

# # Set danger threshold (use empirical or default)
# DANGER_THRESHOLD = empirical_danger_threshold if empirical_danger_threshold else 0.35
# OPEN_THRESHOLD = 0.70
# print(f"\n  Using DANGER_THRESHOLD = {DANGER_THRESHOLD:.2f}")
# print(f"  Using OPEN_THRESHOLD = {OPEN_THRESHOLD:.2f}")


# # ============================================================================
# # SECTION C: FIT INTERCEPTION PROBABILITY FUNCTION
# # ============================================================================

# print("\n" + "-" * 70)
# print("SECTION C: FIT INTERCEPTION PROBABILITY FUNCTION")
# print("-" * 70)

# def fit_interception_probability(df_calib):
#     """
#     Fit a smooth function for P(INT | control_ratio).
    
#     INT probability is highest at low control ratios and decreases
#     as control increases. We fit a decreasing function.
#     """
#     from scipy.optimize import curve_fit
    
#     # Aggregate by control ratio bins
#     df_calib['control_bin_fine'] = pd.cut(df_calib['control_ratio'], bins=20)
#     bin_data = df_calib.groupby('control_bin_fine').agg({
#         'control_ratio': 'mean',
#         'is_int': ['mean', 'count']
#     }).reset_index()
#     bin_data.columns = ['bin', 'control', 'int_rate', 'n']
#     bin_data = bin_data.dropna()
    
#     # Filter to bins with enough data
#     bin_data = bin_data[bin_data['n'] >= 20]
    
#     # Fit exponential decay: P(INT) = a * exp(-b * control) + c
#     def int_prob_func(x, a, b, c):
#         return a * np.exp(-b * x) + c
    
#     try:
#         popt, _ = curve_fit(
#             int_prob_func,
#             bin_data['control'].values,
#             bin_data['int_rate'].values,
#             p0=[0.1, 3.0, 0.01],
#             bounds=([0, 0, 0], [0.5, 10, 0.05]),
#             maxfev=5000
#         )
#         a, b, c = popt
#         print(f"\n  Fitted INT probability: P(INT) = {a:.4f} * exp(-{b:.2f} * control) + {c:.4f}")
        
#         # Create function
#         def get_int_prob(control_ratio):
#             control_ratio = np.clip(control_ratio, 0, 1)
#             return np.clip(a * np.exp(-b * control_ratio) + c, 0, 0.25)
        
#     except Exception as e:
#         print(f"\n  Curve fit failed: {e}")
#         print("  Using piecewise linear approximation")
        
#         # Fallback: piecewise linear
#         def get_int_prob(control_ratio):
#             control_ratio = np.atleast_1d(control_ratio)
#             result = np.zeros_like(control_ratio, dtype=float)
            
#             # High INT risk at low control
#             mask1 = control_ratio < 0.2
#             result[mask1] = 0.08 - 0.2 * control_ratio[mask1]
            
#             # Medium INT risk
#             mask2 = (control_ratio >= 0.2) & (control_ratio < 0.5)
#             result[mask2] = 0.05 - 0.06 * (control_ratio[mask2] - 0.2)
            
#             # Low INT risk at high control
#             mask3 = control_ratio >= 0.5
#             result[mask3] = 0.03 - 0.05 * (control_ratio[mask3] - 0.5)
            
#             return np.clip(result, 0.003, 0.15)
    
#     # Print sample values
#     print(f"\n  Sample P(INT) values:")
#     for ctrl in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#         print(f"    Control {ctrl:.1f}: P(INT) = {get_int_prob(ctrl)*100:.2f}%")
    
#     return get_int_prob


# get_interception_probability = fit_interception_probability(df_int_calibration)


# # ============================================================================
# # SECTION D: SITUATION-DEPENDENT EPA FUNCTIONS
# # ============================================================================

# print("\n" + "-" * 70)
# print("SECTION D: SITUATION-DEPENDENT EPA FUNCTIONS")
# print("-" * 70)

# def calculate_epa_incomplete(current_field_pos, down, yards_to_go):
#     """
#     EPA for an incomplete pass.
    
#     Normal downs: EP(same position, down+1, same distance) - EP(current)
#     4th down: Turnover on downs - opponent gets ball
    
#     Parameters:
#         current_field_pos: Yards from own goal (0-100 scale)
#         down: Current down (1-4)
#         yards_to_go: Yards to first down
#     """
#     ep_before = get_ep(current_field_pos, down, yards_to_go)
    
#     if down < 4:
#         # Next down, same position and distance
#         ep_after = get_ep(current_field_pos, down + 1, yards_to_go)
#     else:
#         # 4th down incomplete = turnover on downs
#         # Opponent gets ball at current position
#         opponent_field_pos = 100 - current_field_pos
#         # Opponent's positive EP is our negative
#         ep_after = -get_ep(opponent_field_pos, 1, 10)
    
#     return ep_after - ep_before


# def calculate_epa_interception(int_field_pos, current_field_pos, down, yards_to_go):
#     """
#     EPA for an interception.
    
#     Opponent gets ball at interception location with 1st & 10.
#     Their positive EP becomes our negative value.
    
#     Parameters:
#         int_field_pos: Where INT occurs (yards from our goal, 0-100)
#         current_field_pos: Where play started (yards from our goal, 0-100)
#         down: Current down
#         yards_to_go: Yards to first down
#     """
#     ep_before = get_ep(current_field_pos, down, yards_to_go)
    
#     # Opponent gets ball - their field position is mirror of ours
#     opponent_field_pos = 100 - int_field_pos
#     opponent_field_pos = np.clip(opponent_field_pos, 1, 99)
    
#     # Opponent's EP (positive for them = negative for us)
#     ep_opponent = get_ep(opponent_field_pos, 1, 10)
    
#     return -ep_opponent - ep_before


# def calculate_epa_complete(catch_field_pos, current_field_pos, down, yards_to_go):
#     """
#     EPA for a completed pass.
    
#     Parameters:
#         catch_field_pos: Where catch is made (yards from own goal, 0-100)
#         current_field_pos: Where play started
#         down: Current down
#         yards_to_go: Yards to first down
#     """
#     ep_before = get_ep(current_field_pos, down, yards_to_go)
    
#     # Yards gained
#     yards_gained = catch_field_pos - current_field_pos
    
#     # Check for touchdown
#     if catch_field_pos >= 100:
#         return 7.0 - ep_before
    
#     # Check for first down
#     if yards_gained >= yards_to_go:
#         new_down = 1
#         new_distance = min(10, 100 - catch_field_pos)  # Goal to go if close
#     else:
#         new_down = min(down + 1, 4)
#         new_distance = max(1, yards_to_go - yards_gained)
    
#     ep_after = get_ep(catch_field_pos, new_down, new_distance)
    
#     return ep_after - ep_before


# # Test the EPA functions
# print("\n  Testing EPA functions:")
# print(f"\n  Scenario: 1st & 10 at own 25 (field_pos=25)")
# print(f"    EP before: {get_ep(25, 1, 10):.2f}")
# print(f"    Complete at own 35 (+10 yards, 1st down): EPA = {calculate_epa_complete(35, 25, 1, 10):.2f}")
# print(f"    Complete at own 30 (+5 yards, 2nd & 5): EPA = {calculate_epa_complete(30, 25, 1, 10):.2f}")
# print(f"    Incomplete (2nd & 10): EPA = {calculate_epa_incomplete(25, 1, 10):.2f}")
# print(f"    INT at own 25: EPA = {calculate_epa_interception(25, 25, 1, 10):.2f}")
# print(f"    INT at own 40: EPA = {calculate_epa_interception(40, 25, 1, 10):.2f}")

# print(f"\n  Scenario: 3rd & 8 at own 40 (field_pos=40)")
# print(f"    EP before: {get_ep(40, 3, 8):.2f}")
# print(f"    Complete at own 50 (+10 yards, 1st down): EPA = {calculate_epa_complete(50, 40, 3, 8):.2f}")
# print(f"    Complete at own 45 (+5 yards, 4th & 3): EPA = {calculate_epa_complete(45, 40, 3, 8):.2f}")
# print(f"    Incomplete (4th & 8): EPA = {calculate_epa_incomplete(40, 3, 8):.2f}")
# print(f"    INT at midfield: EPA = {calculate_epa_interception(50, 40, 3, 8):.2f}")

# print(f"\n  Scenario: 4th & 2 at opponent 30 (field_pos=70)")
# print(f"    EP before: {get_ep(70, 4, 2):.2f}")
# print(f"    Complete at opponent 25 (+5 yards, 1st down): EPA = {calculate_epa_complete(75, 70, 4, 2):.2f}")
# print(f"    Incomplete (turnover on downs): EPA = {calculate_epa_incomplete(70, 4, 2):.2f}")
# print(f"    INT at opponent 25: EPA = {calculate_epa_interception(75, 70, 4, 2):.2f}")


# # ============================================================================
# # SECTION E: THREE-OUTCOME EXPECTED VALUE AT GRID POINT
# # ============================================================================

# print("\n" + "-" * 70)
# print("SECTION E: THREE-OUTCOME EXPECTED VALUE CALCULATION")
# print("-" * 70)

# def calculate_expected_value_surface(grid_x, control_ratio, los_x, down, yards_to_go):
#     """
#     Calculate three-outcome expected value at each grid point.
    
#     E[Value] = P(complete) Ã— EPA_complete 
#              + P(incomplete) Ã— EPA_incomplete 
#              + P(INT) Ã— EPA_INT
    
#     Parameters:
#         grid_x: 2D array of field x-coordinates
#         control_ratio: 2D array of control ratios at each point
#         los_x: Line of scrimmage (absolute_yardline_number)
#         down: Current down
#         yards_to_go: Yards to first down
        
#     Returns:
#         expected_value: 2D array of E[Value] at each point
#         p_complete: 2D array of P(complete)
#         p_int: 2D array of P(INT)
#     """
#     # Current field position (0-100 scale)
#     current_field_pos = los_x - 10
    
#     # Catch field position at each grid point
#     catch_field_pos = grid_x - 10
    
#     # Probabilities at each point
#     p_complete = get_completion_probability(control_ratio)
#     p_int = get_interception_probability(control_ratio)
#     p_incomplete = 1 - p_complete - p_int
    
#     # Ensure probabilities are valid
#     p_incomplete = np.maximum(p_incomplete, 0)
    
#     # EPA surfaces
#     epa_complete = np.zeros_like(grid_x, dtype=float)
#     epa_int = np.zeros_like(grid_x, dtype=float)
    
#     # Calculate EPA at each grid point
#     for i in range(grid_x.shape[0]):
#         for j in range(grid_x.shape[1]):
#             catch_pos = catch_field_pos[i, j]
#             epa_complete[i, j] = calculate_epa_complete(catch_pos, current_field_pos, down, yards_to_go)
#             epa_int[i, j] = calculate_epa_interception(catch_pos, current_field_pos, down, yards_to_go)
    
#     # EPA incomplete is constant (doesn't depend on where throw goes)
#     epa_incomplete = calculate_epa_incomplete(current_field_pos, down, yards_to_go)
    
#     # Three-outcome expected value
#     expected_value = (p_complete * epa_complete + 
#                       p_incomplete * epa_incomplete + 
#                       p_int * epa_int)
    
#     return expected_value, p_complete, p_int, epa_complete, epa_incomplete, epa_int


# print("  Three-outcome expected value function created")


# # ============================================================================
# # SECTION F: SPATIAL OPENNESS METRICS
# # ============================================================================

# print("\n" + "-" * 70)
# print("SECTION F: SPATIAL OPENNESS METRICS")
# print("-" * 70)

# def calculate_spatial_metrics(reach_prob, control_ratio, 
#                               open_threshold=OPEN_THRESHOLD, 
#                               danger_threshold=DANGER_THRESHOLD):
#     """
#     Calculate spatial openness metrics for a receiver's cloud.
    
#     Partitions the reachable space by control level:
#     - Open Zone: control > open_threshold (safe throws)
#     - Contested Zone: between thresholds
#     - Danger Zone: control < danger_threshold (INT risk)
    
#     Parameters:
#         reach_prob: 2D array of P(receiver reaches this point)
#         control_ratio: 2D array of control at each point
#         open_threshold: Control level above which is "open"
#         danger_threshold: Control level below which is "danger"
        
#     Returns:
#         Dict with spatial metrics
#     """
#     # Create masks
#     open_mask = control_ratio > open_threshold
#     danger_mask = control_ratio < danger_threshold
#     contested_mask = ~open_mask & ~danger_mask
    
#     # Calculate probability-weighted areas
#     total_prob = reach_prob.sum()
    
#     if total_prob > 0:
#         open_area = (reach_prob * open_mask).sum() / total_prob
#         contested_area = (reach_prob * contested_mask).sum() / total_prob
#         danger_area = (reach_prob * danger_mask).sum() / total_prob
#     else:
#         open_area = 0
#         contested_area = 0
#         danger_area = 0
    
#     # Window ratio: proportion of reachable space that's safe vs dangerous
#     if (open_area + danger_area) > 0:
#         window_ratio = open_area / (open_area + danger_area)
#     else:
#         window_ratio = 0.5
    
#     # Average control in reachable space
#     if total_prob > 0:
#         avg_control = (reach_prob * control_ratio).sum() / total_prob
#     else:
#         avg_control = 0.5
    
#     return {
#         'open_area': open_area,
#         'contested_area': contested_area,
#         'danger_area': danger_area,
#         'window_ratio': window_ratio,
#         'avg_control': avg_control
#     }


# print(f"  Spatial metrics function created")
# print(f"  Open threshold: {OPEN_THRESHOLD}")
# print(f"  Danger threshold: {DANGER_THRESHOLD}")


# # ============================================================================
# # SECTION G: VWO V2 - FULL RECEIVER CALCULATION
# # ============================================================================

# print("\n" + "-" * 70)
# print("SECTION G: VWO V2 - RECEIVER CALCULATION")
# print("-" * 70)

# # Risk adjustment parameter (default 1.0, tune later)
# LAMBDA_RISK = 1.0

# def calculate_receiver_vwo_v2(receiver_row, all_players_frame, los_x, down, yards_to_go,
#                                flight_time, grid_spacing=1.0, grid_extent=12,
#                                lambda_risk=LAMBDA_RISK):
#     """
#     Calculate Value-Weighted Openness V2 for a single receiver.
    
#     Includes:
#     - Three-outcome expected value (complete, incomplete, INT)
#     - Spatial openness metrics
#     - Risk-adjusted VWO
    
#     Parameters:
#         receiver_row: Series with receiver's position/velocity
#         all_players_frame: DataFrame with all players
#         los_x: Line of scrimmage
#         down: Current down
#         yards_to_go: Yards to first down
#         flight_time: Ball flight time
#         grid_spacing: Resolution of integration grid
#         grid_extent: How far to extend grid from receiver
#         lambda_risk: Risk aversion parameter (higher = more risk-averse)
        
#     Returns:
#         Dict with VWO and all component metrics
#     """
#     # Receiver state
#     receiver_x = receiver_row['x_norm']
#     receiver_y = receiver_row['y_norm']
#     receiver_speed = receiver_row['s'] if pd.notna(receiver_row['s']) else 0
#     receiver_dir = receiver_row['dir_norm'] if pd.notna(receiver_row['dir_norm']) else 0
    
#     # Get velocity and time bins
#     velocity_bin = get_velocity_bin(receiver_speed)
#     time_bin = get_time_bin(flight_time)
    
#     # Get movement distribution
#     if hasattr(get_player_distribution_interpolated, '__call__'):
#         dist, _, _, _ = get_player_distribution_interpolated(receiver_speed, flight_time)
#     else:
#         dist, _, _ = get_player_distribution(receiver_speed, flight_time)
    
#     if dist is None:
#         return {
#             'vwo': 0,
#             'vwo_complete_only': 0,
#             'vwo_risk_adjusted': 0,
#             'open_area': 0,
#             'contested_area': 0,
#             'danger_area': 0,
#             'window_ratio': 0.5,
#             'avg_control': 0.5,
#             'avg_completion_prob': 0,
#             'avg_int_prob': 0,
#             'avg_epa_complete': 0,
#             'epa_incomplete': 0,
#             'avg_epa_int': 0,
#             'sweet_spot_x': receiver_x,
#             'sweet_spot_y': receiver_y,
#             'max_vwo_density': 0,
#             'velocity_bin': velocity_bin,
#             'time_bin': time_bin,
#             'status': 'no_distribution'
#         }
    
#     # Create field grid centered on receiver
#     x_coords = np.arange(receiver_x - grid_extent, receiver_x + grid_extent + grid_spacing, grid_spacing)
#     y_coords = np.arange(
#         max(0, receiver_y - grid_extent),
#         min(53.3, receiver_y + grid_extent) + grid_spacing,
#         grid_spacing
#     )
#     field_grid_x, field_grid_y = np.meshgrid(x_coords, y_coords)
    
#     # Transform movement distribution to field coordinates
#     reach_density = transform_distribution_to_field(
#         dist, receiver_x, receiver_y, receiver_dir,
#         field_grid_x, field_grid_y
#     )
    
#     # Expand by catch radius
#     reach_density_expanded = expand_density_with_catch_radius(
#         reach_density, grid_spacing, CATCH_RADIUS
#     )
    
#     # Normalize to probability distribution
#     if reach_density_expanded.sum() > 0:
#         reach_prob = reach_density_expanded / reach_density_expanded.sum()
#     else:
#         reach_prob = reach_density_expanded
    
#     # Calculate pitch control at each grid point
#     offense_density = np.zeros_like(field_grid_x, dtype=float)
#     defense_density = np.zeros_like(field_grid_x, dtype=float)
    
#     for _, player in all_players_frame.iterrows():
#         player_x = player['x_norm']
#         player_y = player['y_norm']
#         player_speed = player['s'] if pd.notna(player['s']) else 0
#         player_dir = player['dir_norm'] if pd.notna(player['dir_norm']) else 0
#         player_side = player['player_side']
        
#         if pd.isna(player_x) or pd.isna(player_y):
#             continue
#         if player_side not in ['Offense', 'Defense']:
#             continue
        
#         # Get player's distribution
#         if hasattr(get_player_distribution_interpolated, '__call__'):
#             p_dist, _, _, _ = get_player_distribution_interpolated(player_speed, flight_time)
#         else:
#             p_dist, _, _ = get_player_distribution(player_speed, flight_time)
        
#         if p_dist is not None:
#             player_density = transform_distribution_to_field(
#                 p_dist, player_x, player_y, player_dir,
#                 field_grid_x, field_grid_y
#             )
#         else:
#             player_density = create_fallback_distribution(
#                 player_x, player_y, player_speed, player_dir,
#                 flight_time, field_grid_x, field_grid_y
#             )
        
#         # Expand by catch radius
#         player_expanded = expand_density_with_catch_radius(
#             player_density, grid_spacing, CATCH_RADIUS
#         )
        
#         # Normalize per player
#         if player_expanded.max() > 0:
#             player_norm = player_expanded / player_expanded.max()
#         else:
#             player_norm = player_expanded
        
#         # Accumulate
#         if player_side == 'Offense':
#             offense_density += player_norm
#         else:
#             defense_density += player_norm
    
#     # Control ratio
#     total_density = offense_density + defense_density + CONTROL_EPSILON
#     control_ratio = np.clip(offense_density / total_density, 0, 1)
    
#     # Three-outcome expected value at each point
#     expected_value, p_complete, p_int, epa_complete, epa_incomplete, epa_int = \
#         calculate_expected_value_surface(field_grid_x, control_ratio, los_x, down, yards_to_go)
    
#     # Spatial metrics
#     spatial_metrics = calculate_spatial_metrics(reach_prob, control_ratio)
    
#     # VWO V2: Three-outcome (integrate expected value over reach probability)
#     vwo_density = reach_prob * expected_value
#     total_vwo = np.sum(vwo_density) * (grid_spacing ** 2)
    
#     # VWO V1: Complete-only for comparison
#     epa_complete_only = epa_complete  # This is already the complete EPA
#     vwo_complete_only_density = reach_prob * p_complete * epa_complete
#     vwo_complete_only = np.sum(vwo_complete_only_density) * (grid_spacing ** 2)
    
#     # Risk-adjusted VWO
#     vwo_risk_adjusted = total_vwo - lambda_risk * spatial_metrics['danger_area']
    
#     # Find sweet spot (max VWO density point)
#     max_idx = np.unravel_index(np.argmax(vwo_density), vwo_density.shape)
#     sweet_spot_x = field_grid_x[max_idx]
#     sweet_spot_y = field_grid_y[max_idx]
#     max_vwo_density = vwo_density[max_idx]
    
#     # Component averages (weighted by reach probability)
#     total_reach = reach_prob.sum()
#     if total_reach > 0:
#         avg_completion_prob = (reach_prob * p_complete).sum() / total_reach
#         avg_int_prob = (reach_prob * p_int).sum() / total_reach
#         avg_epa_complete = (reach_prob * epa_complete).sum() / total_reach
#         avg_epa_int = (reach_prob * epa_int).sum() / total_reach
#     else:
#         avg_completion_prob = 0
#         avg_int_prob = 0
#         avg_epa_complete = 0
#         avg_epa_int = 0
    
#     return {
#         # Value metrics
#         'vwo': total_vwo,
#         'vwo_complete_only': vwo_complete_only,
#         'vwo_risk_adjusted': vwo_risk_adjusted,
        
#         # Spatial metrics
#         'open_area': spatial_metrics['open_area'],
#         'contested_area': spatial_metrics['contested_area'],
#         'danger_area': spatial_metrics['danger_area'],
#         'window_ratio': spatial_metrics['window_ratio'],
#         'avg_control': spatial_metrics['avg_control'],
        
#         # Probability metrics
#         'avg_completion_prob': avg_completion_prob,
#         'avg_int_prob': avg_int_prob,
        
#         # EPA metrics
#         'avg_epa_complete': avg_epa_complete,
#         'epa_incomplete': epa_incomplete,
#         'avg_epa_int': avg_epa_int,
        
#         # Location metrics
#         'sweet_spot_x': sweet_spot_x,
#         'sweet_spot_y': sweet_spot_y,
#         'max_vwo_density': max_vwo_density,
#         'receiver_x': receiver_x,
#         'receiver_y': receiver_y,
        
#         # Bins
#         'velocity_bin': velocity_bin,
#         'time_bin': time_bin,
        
#         # Surfaces for visualization
#         'grid_x': field_grid_x,
#         'grid_y': field_grid_y,
#         'reach_prob': reach_prob,
#         'control_ratio': control_ratio,
#         'expected_value': expected_value,
#         'vwo_density': vwo_density,
        
#         'status': 'success'
#     }


# print(f"  VWO V2 receiver calculation function created")
# print(f"  Lambda risk parameter: {LAMBDA_RISK}")


# # ============================================================================
# # SECTION H: PLAY-LEVEL VWO V2 ANALYSIS
# # ============================================================================

# print("\n" + "-" * 70)
# print("SECTION H: PLAY-LEVEL VWO V2 ANALYSIS")
# print("-" * 70)

# def calculate_play_vwo_v2(game_id, play_id, verbose=False, lambda_risk=LAMBDA_RISK):
#     """
#     Calculate VWO V2 for all eligible receivers on a play.
    
#     Returns optimal target under both risk-neutral and risk-adjusted criteria.
#     """
#     # Get play metadata
#     play_data = df_input[
#         (df_input['game_id'] == game_id) & 
#         (df_input['play_id'] == play_id)
#     ]
    
#     if len(play_data) == 0:
#         return None
    
#     # Get situation
#     los_x = play_data['absolute_yardline_number'].iloc[0]
#     down = play_data['down'].iloc[0]
#     yards_to_go = play_data['yards_to_go'].iloc[0]
#     pass_result = play_data['pass_result'].iloc[0]
    
#     # Get last INPUT frame (all players)
#     last_frame = get_last_input_frame(game_id, play_id)
#     if last_frame is None:
#         return None
    
#     # Get flight time
#     base_flight_time, _ = get_ball_flight_time(game_id, play_id)
#     if base_flight_time is None:
#         base_flight_time = 1.5
#     flight_time = base_flight_time + (1 / FRAME_RATE)
    
#     # Identify receivers
#     receivers = last_frame[
#         last_frame['player_role'].isin(['Targeted Receiver', 'Other Route Runner'])
#     ]
    
#     if len(receivers) == 0:
#         return None
    
#     # Calculate VWO V2 for each receiver
#     receiver_vwos = []
    
#     for _, receiver in receivers.iterrows():
#         vwo_result = calculate_receiver_vwo_v2(
#             receiver, last_frame, los_x, down, yards_to_go, flight_time,
#             grid_spacing=1.0, grid_extent=12, lambda_risk=lambda_risk
#         )
        
#         vwo_result['nfl_id'] = receiver['nfl_id']
#         vwo_result['player_name'] = receiver.get('player_name', 'Unknown')
#         vwo_result['player_role'] = receiver['player_role']
        
#         receiver_vwos.append(vwo_result)
    
#     # Build summary dataframe
#     vwo_df = pd.DataFrame([{
#         'nfl_id': r['nfl_id'],
#         'player_name': r['player_name'],
#         'player_role': r['player_role'],
#         'vwo': r['vwo'],
#         'vwo_complete_only': r['vwo_complete_only'],
#         'vwo_risk_adjusted': r['vwo_risk_adjusted'],
#         'open_area': r['open_area'],
#         'danger_area': r['danger_area'],
#         'window_ratio': r['window_ratio'],
#         'avg_completion_prob': r['avg_completion_prob'],
#         'avg_int_prob': r['avg_int_prob'],
#         'avg_epa_complete': r['avg_epa_complete']
#     } for r in receiver_vwos if r['status'] == 'success'])
    
#     if len(vwo_df) == 0:
#         return None
    
#     # Find optimal targets under different criteria
#     optimal_vwo_idx = vwo_df['vwo'].idxmax()
#     optimal_risk_adj_idx = vwo_df['vwo_risk_adjusted'].idxmax()
    
#     optimal_vwo_target = vwo_df.loc[optimal_vwo_idx]
#     optimal_risk_adj_target = vwo_df.loc[optimal_risk_adj_idx]
    
#     # Actual target
#     actual_target = vwo_df[vwo_df['player_role'] == 'Targeted Receiver']
    
#     if len(actual_target) == 0:
#         actual_vwo = 0
#         actual_vwo_risk_adj = 0
#         target_quality = 0
#         target_quality_risk_adj = 0
#     else:
#         actual_target = actual_target.iloc[0]
#         actual_vwo = actual_target['vwo']
#         actual_vwo_risk_adj = actual_target['vwo_risk_adjusted']
        
#         # Target quality
#         if optimal_vwo_target['vwo'] != 0:
#             target_quality = actual_vwo / optimal_vwo_target['vwo'] if optimal_vwo_target['vwo'] > 0 else 0
#         else:
#             target_quality = 1.0
            
#         if optimal_risk_adj_target['vwo_risk_adjusted'] != 0:
#             target_quality_risk_adj = actual_vwo_risk_adj / optimal_risk_adj_target['vwo_risk_adjusted'] \
#                 if optimal_risk_adj_target['vwo_risk_adjusted'] > 0 else 0
#         else:
#             target_quality_risk_adj = 1.0
    
#     if verbose:
#         print(f"\n  Play {game_id}-{play_id}: {down}&{yards_to_go} at {los_x}")
#         print(f"  Receivers analyzed: {len(vwo_df)}")
#         print(f"\n  VWO Rankings:")
#         for _, row in vwo_df.sort_values('vwo', ascending=False).iterrows():
#             role_marker = "â†’ " if row['player_role'] == 'Targeted Receiver' else "  "
#             print(f"    {role_marker}{row['player_name']}: VWO={row['vwo']:.3f}, "
#                   f"Risk-Adj={row['vwo_risk_adjusted']:.3f}, "
#                   f"Window={row['window_ratio']:.2f}, Danger={row['danger_area']:.2f}")
#         print(f"\n  Optimal (VWO): {optimal_vwo_target['player_name']} ({optimal_vwo_target['vwo']:.3f})")
#         print(f"  Optimal (Risk-Adj): {optimal_risk_adj_target['player_name']} ({optimal_risk_adj_target['vwo_risk_adjusted']:.3f})")
#         print(f"  Target Quality: {target_quality:.1%} (Risk-Adj: {target_quality_risk_adj:.1%})")
    
#     return {
#         'game_id': game_id,
#         'play_id': play_id,
#         'down': down,
#         'yards_to_go': yards_to_go,
#         'los_x': los_x,
#         'pass_result': pass_result,
#         'flight_time': flight_time,
#         'n_receivers': len(vwo_df),
#         'receiver_vwos': receiver_vwos,
#         'vwo_summary': vwo_df,
        
#         # Optimal targets (risk-neutral)
#         'optimal_target_id': optimal_vwo_target['nfl_id'],
#         'optimal_target_name': optimal_vwo_target['player_name'],
#         'optimal_vwo': optimal_vwo_target['vwo'],
        
#         # Optimal targets (risk-adjusted)
#         'optimal_risk_adj_id': optimal_risk_adj_target['nfl_id'],
#         'optimal_risk_adj_name': optimal_risk_adj_target['player_name'],
#         'optimal_vwo_risk_adj': optimal_risk_adj_target['vwo_risk_adjusted'],
        
#         # Actual target
#         'actual_target_id': actual_target['nfl_id'] if len(actual_target) > 0 else None,
#         'actual_target_name': actual_target['player_name'] if len(actual_target) > 0 else None,
#         'actual_vwo': actual_vwo,
#         'actual_vwo_risk_adj': actual_vwo_risk_adj,
#         'actual_window_ratio': actual_target['window_ratio'] if len(actual_target) > 0 else 0,
#         'actual_danger_area': actual_target['danger_area'] if len(actual_target) > 0 else 0,
        
#         # Decision quality
#         'target_quality': target_quality,
#         'target_quality_risk_adj': target_quality_risk_adj,
#         'chose_optimal': optimal_vwo_target['nfl_id'] == (actual_target['nfl_id'] if len(actual_target) > 0 else None),
#         'chose_optimal_risk_adj': optimal_risk_adj_target['nfl_id'] == (actual_target['nfl_id'] if len(actual_target) > 0 else None)
#     }


# print("  Play-level VWO V2 function created")


# # ============================================================================
# # SECTION I: BATCH ANALYSIS
# # ============================================================================

# print("\n" + "-" * 70)
# print("SECTION I: BATCH ANALYSIS FUNCTION")
# print("-" * 70)

# def analyze_vwo_v2_batch(sample_size=100, verbose=True, lambda_risk=LAMBDA_RISK):
#     """
#     Analyze VWO V2 for a batch of plays.
#     """
#     print(f"\n  Analyzing {sample_size} plays with VWO V2...")
#     print(f"  Lambda risk parameter: {lambda_risk}")
    
#     plays = df_input.groupby(['game_id', 'play_id']).first().reset_index()
    
#     if sample_size and sample_size < len(plays):
#         plays = plays.sample(sample_size, random_state=42)
    
#     results = []
#     errors = 0
    
#     for idx, (_, play) in enumerate(plays.iterrows()):
#         try:
#             result = calculate_play_vwo_v2(
#                 play['game_id'], play['play_id'], 
#                 verbose=False, lambda_risk=lambda_risk
#             )
            
#             if result is not None:
#                 results.append({
#                     'game_id': result['game_id'],
#                     'play_id': result['play_id'],
#                     'down': result['down'],
#                     'yards_to_go': result['yards_to_go'],
#                     'pass_result': result['pass_result'],
#                     'n_receivers': result['n_receivers'],
                    
#                     # VWO metrics
#                     'optimal_vwo': result['optimal_vwo'],
#                     'optimal_vwo_risk_adj': result['optimal_vwo_risk_adj'],
#                     'actual_vwo': result['actual_vwo'],
#                     'actual_vwo_risk_adj': result['actual_vwo_risk_adj'],
                    
#                     # Spatial metrics for actual target
#                     'actual_window_ratio': result['actual_window_ratio'],
#                     'actual_danger_area': result['actual_danger_area'],
                    
#                     # Decision quality
#                     'target_quality': result['target_quality'],
#                     'target_quality_risk_adj': result['target_quality_risk_adj'],
#                     'chose_optimal': result['chose_optimal'],
#                     'chose_optimal_risk_adj': result['chose_optimal_risk_adj'],
                    
#                     # Target names
#                     'optimal_target': result['optimal_target_name'],
#                     'optimal_risk_adj_target': result['optimal_risk_adj_name'],
#                     'actual_target': result['actual_target_name']
#                 })
#         except Exception as e:
#             errors += 1
#             continue
        
#         if verbose and (idx + 1) % 20 == 0:
#             print(f"    Processed {idx + 1}/{len(plays)} plays...")
    
#     df_results = pd.DataFrame(results)
    
#     print(f"\n  âœ“ Analyzed {len(df_results)} plays ({errors} errors)")
    
#     return df_results


# def summarize_vwo_v2(df_results):
#     """
#     Summarize VWO V2 results comparing risk-neutral vs risk-adjusted.
#     """
#     print("\n" + "=" * 70)
#     print("VWO V2 ANALYSIS SUMMARY")
#     print("=" * 70)
    
#     n = len(df_results)
    
#     print(f"\n  DECISION QUALITY COMPARISON ({n} plays)")
#     print("  " + "-" * 50)
#     print(f"  {'Metric':<35} {'Risk-Neutral':<15} {'Risk-Adjusted':<15}")
#     print("  " + "-" * 65)
#     print(f"  {'Mean Target Quality':<35} {df_results['target_quality'].mean():>10.1%}     "
#           f"{df_results['target_quality_risk_adj'].mean():>10.1%}")
#     print(f"  {'Median Target Quality':<35} {df_results['target_quality'].median():>10.1%}     "
#           f"{df_results['target_quality_risk_adj'].median():>10.1%}")
#     print(f"  {'% Chose Optimal':<35} {df_results['chose_optimal'].mean():>10.1%}     "
#           f"{df_results['chose_optimal_risk_adj'].mean():>10.1%}")
    
#     print(f"\n  SPATIAL METRICS FOR ACTUAL TARGETS")
#     print("  " + "-" * 50)
#     print(f"  Mean Window Ratio: {df_results['actual_window_ratio'].mean():.2f}")
#     print(f"  Mean Danger Area: {df_results['actual_danger_area'].mean():.2f}")
    
#     print(f"\n  VWO COMPARISON")
#     print("  " + "-" * 50)
#     print(f"  Mean Optimal VWO: {df_results['optimal_vwo'].mean():.3f}")
#     print(f"  Mean Actual VWO: {df_results['actual_vwo'].mean():.3f}")
#     print(f"  Mean VWO Gap: {(df_results['optimal_vwo'] - df_results['actual_vwo']).mean():.3f}")
    
#     print(f"\n  BY OUTCOME")
#     print("  " + "-" * 50)
#     print(f"  {'Outcome':<12} {'N':<8} {'Quality':<12} {'Quality (Risk-Adj)':<18} {'Window Ratio':<15}")
#     print("  " + "-" * 65)
    
#     for outcome, label in [('C', 'Complete'), ('I', 'Incomplete'), ('IN', 'INT')]:
#         subset = df_results[df_results['pass_result'] == outcome]
#         if len(subset) > 3:
#             print(f"  {label:<12} {len(subset):<8} {subset['target_quality'].mean():>8.1%}     "
#                   f"{subset['target_quality_risk_adj'].mean():>12.1%}       "
#                   f"{subset['actual_window_ratio'].mean():>8.2f}")
    
#     return df_results


# print("  Batch analysis and summary functions created")


# # ============================================================================
# # SUMMARY
# # ============================================================================

# print("\n" + "=" * 70)
# print("VWO V2 SYSTEM READY")
# print("=" * 70)

# print(f"""
# IMPROVEMENTS IN VWO V2:
#   â€¢ Three-outcome expected value (complete, incomplete, INT)
#   â€¢ Situation-dependent EPA for incompletes (worse on 3rd/4th down)
#   â€¢ Field-position dependent EPA for interceptions
#   â€¢ Spatial openness metrics (Open Area, Danger Area, Window Ratio)
#   â€¢ Risk-adjusted VWO with lambda parameter

# KEY PARAMETERS:
#   â€¢ OPEN_THRESHOLD = {OPEN_THRESHOLD} (control above this = "open")
#   â€¢ DANGER_THRESHOLD = {DANGER_THRESHOLD:.2f} (control below this = "danger")  
#   â€¢ LAMBDA_RISK = {LAMBDA_RISK} (risk aversion, tune later)

# FUNCTIONS:
#   â€¢ calculate_play_vwo_v2(game_id, play_id, verbose=True)
#   â€¢ analyze_vwo_v2_batch(sample_size=100)
#   â€¢ summarize_vwo_v2(df_results)

# QUICK START:
#   # Single play analysis
#   result = calculate_play_vwo_v2(game_id, play_id, verbose=True)
  
#   # Batch analysis
#   df_vwo_v2 = analyze_vwo_v2_batch(sample_size=200)
#   summarize_vwo_v2(df_vwo_v2)
# """)

# print("=" * 70)


# Cell: Value-Weighted Openness (VWO) V2 - CONSOLIDATED
# ============================================================================
# Three-Outcome Expected Value with Spatial Metrics
#
# E[Value] = P(complete) Ã— EPA_complete 
#          + P(incomplete) Ã— EPA_incomplete 
#          + P(INT) Ã— EPA_INT
#
# Includes: Open Area, Danger Area, Window Ratio, Risk-Adjusted VWO
# ============================================================================

print("=" * 70)
print("VALUE-WEIGHTED OPENNESS (VWO) V2")
print("=" * 70)

# ============================================================================
# PARAMETERS
# ============================================================================

OPEN_THRESHOLD = 0.70      # Control above this = "open"
DANGER_THRESHOLD = 0.35    # Control below this = "danger" (update with your empirical value)
LAMBDA_RISK = 1.0          # Risk aversion parameter

print(f"\n  Parameters:")
print(f"    OPEN_THRESHOLD = {OPEN_THRESHOLD}")
print(f"    DANGER_THRESHOLD = {DANGER_THRESHOLD}")
print(f"    LAMBDA_RISK = {LAMBDA_RISK}")


# ============================================================================
# EXPECTED POINTS (EP) LOOKUP TABLE (from v1)
# ============================================================================

def build_ep_table():
    """
    Build standard Expected Points lookup table.
    Based on NFL averages by field position, down, and distance.
    """
    
    def base_ep_by_field_position(field_pos):
        """Base EP assuming 1st and 10."""
        field_pos = np.clip(field_pos, 0, 100)
        
        if isinstance(field_pos, np.ndarray):
            ep = np.zeros_like(field_pos, dtype=float)
            
            mask = field_pos <= 10
            ep[mask] = -2.0 + (field_pos[mask] / 10) * 1.5
            
            mask = (field_pos > 10) & (field_pos <= 25)
            ep[mask] = -0.5 + ((field_pos[mask] - 10) / 15) * 1.0
            
            mask = (field_pos > 25) & (field_pos <= 50)
            ep[mask] = 0.5 + ((field_pos[mask] - 25) / 25) * 1.5
            
            mask = (field_pos > 50) & (field_pos <= 75)
            ep[mask] = 2.0 + ((field_pos[mask] - 50) / 25) * 2.0
            
            mask = (field_pos > 75) & (field_pos <= 95)
            ep[mask] = 4.0 + ((field_pos[mask] - 75) / 20) * 2.0
            
            mask = field_pos > 95
            ep[mask] = 6.0 + ((field_pos[mask] - 95) / 5) * 0.5
            
            return ep
        else:
            if field_pos <= 10:
                return -2.0 + (field_pos / 10) * 1.5
            elif field_pos <= 25:
                return -0.5 + ((field_pos - 10) / 15) * 1.0
            elif field_pos <= 50:
                return 0.5 + ((field_pos - 25) / 25) * 1.5
            elif field_pos <= 75:
                return 2.0 + ((field_pos - 50) / 25) * 2.0
            elif field_pos <= 95:
                return 4.0 + ((field_pos - 75) / 20) * 2.0
            else:
                return 6.0 + ((field_pos - 95) / 5) * 0.5
    
    def down_distance_adjustment(down, yards_to_go, field_pos):
        """Adjust EP based on down and distance."""
        base = base_ep_by_field_position(field_pos)
        
        down = np.atleast_1d(down)
        yards_to_go = np.atleast_1d(yards_to_go)
        field_pos = np.atleast_1d(field_pos)
        base = np.atleast_1d(base)
        
        down_penalty = np.where(down == 1, 0.0,
                       np.where(down == 2, -0.3,
                       np.where(down == 3, -0.8, -2.0)))
        
        distance_penalty = np.clip((yards_to_go - 5) * 0.05, 0, 1.0)
        
        short_yardage_bonus = np.where(yards_to_go <= 2, 0.3,
                              np.where(yards_to_go <= 4, 0.1, 0.0))
        
        fourth_down_punt = np.where(
            (down == 4) & (field_pos < 60) & (yards_to_go > 3),
            -1.5, 0.0
        )
        
        fourth_down_fg = np.where(
            (down == 4) & (field_pos >= 60) & (field_pos < 95),
            0.5, 0.0
        )
        
        adjustment = down_penalty - distance_penalty + short_yardage_bonus + fourth_down_punt + fourth_down_fg
        result = base + adjustment
        
        if len(result) == 1:
            return float(result[0])
        return result
    
    def get_ep(field_pos, down, yards_to_go):
        """Get Expected Points for a given situation."""
        field_pos = np.clip(field_pos, 0, 100)
        yards_to_go = np.clip(yards_to_go, 1, 40)
        return down_distance_adjustment(down, yards_to_go, field_pos)
    
    return get_ep

# Build EP function
get_ep = build_ep_table()
print("\n  âœ“ EP lookup table ready")


# ============================================================================
# INTERCEPTION PROBABILITY (PRE-COMPUTED)
# ============================================================================
# Fitted from 14,108 plays: P(INT) = a * exp(-b * control) + c

INT_PROB_A = 0.0931
INT_PROB_B = 1.54
INT_PROB_C = 0.0000

def get_interception_probability(control_ratio):
    """
    Get P(INT | control_ratio) using pre-fitted exponential decay.
    Higher control = lower INT probability.
    """
    control_ratio = np.clip(control_ratio, 0, 1)
    p_int = INT_PROB_A * np.exp(-INT_PROB_B * control_ratio) + INT_PROB_C
    return np.clip(p_int, 0.003, 0.15)

print(f"\n  âœ“ Interception probability function ready")
print(f"    P(INT) = {INT_PROB_A:.4f} Ã— exp(-{INT_PROB_B:.2f} Ã— control) + {INT_PROB_C:.4f}")
print(f"    Sample: Control=0.5 â†’ P(INT)={get_interception_probability(0.5)*100:.2f}%")

# ============================================================================
# GET LAST INPUT FRAME (from v1)
# ============================================================================

def get_last_input_frame(game_id, play_id):
    """
    Get the last frame of INPUT data for a play (all players).
    This is the moment of pass release.
    """
    play_data = df_input[
        (df_input['game_id'] == game_id) & 
        (df_input['play_id'] == play_id)
    ]
    
    if len(play_data) == 0:
        return None
    
    last_frame_id = play_data['frame_id'].max()
    last_frame = play_data[play_data['frame_id'] == last_frame_id].copy()
    
    return last_frame


# ============================================================================
# THREE-OUTCOME EPA FUNCTIONS
# ============================================================================

def calculate_epa_incomplete(current_field_pos, down, yards_to_go):
    """EPA for an incomplete pass."""
    ep_before = get_ep(current_field_pos, down, yards_to_go)
    
    if down < 4:
        ep_after = get_ep(current_field_pos, down + 1, yards_to_go)
    else:
        opponent_field_pos = 100 - current_field_pos
        ep_after = -get_ep(opponent_field_pos, 1, 10)
    
    return ep_after - ep_before


def calculate_epa_interception(int_field_pos, current_field_pos, down, yards_to_go):
    """EPA for an interception."""
    ep_before = get_ep(current_field_pos, down, yards_to_go)
    
    opponent_field_pos = 100 - int_field_pos
    opponent_field_pos = np.clip(opponent_field_pos, 1, 99)
    ep_opponent = get_ep(opponent_field_pos, 1, 10)
    
    return -ep_opponent - ep_before


def calculate_epa_complete(catch_field_pos, current_field_pos, down, yards_to_go):
    """EPA for a completed pass."""
    ep_before = get_ep(current_field_pos, down, yards_to_go)
    
    yards_gained = catch_field_pos - current_field_pos
    
    if catch_field_pos >= 100:
        return 7.0 - ep_before
    
    if yards_gained >= yards_to_go:
        new_down = 1
        new_distance = min(10, 100 - catch_field_pos)
    else:
        new_down = min(down + 1, 4)
        new_distance = max(1, yards_to_go - yards_gained)
    
    ep_after = get_ep(catch_field_pos, new_down, new_distance)
    
    return ep_after - ep_before


# ============================================================================
# THREE-OUTCOME EXPECTED VALUE SURFACE
# ============================================================================

def calculate_expected_value_surface(grid_x, control_ratio, los_x, down, yards_to_go):
    """
    Calculate three-outcome expected value at each grid point.
    
    E[Value] = P(complete) Ã— EPA_complete 
             + P(incomplete) Ã— EPA_incomplete 
             + P(INT) Ã— EPA_INT
    """
    current_field_pos = los_x - 10
    catch_field_pos = grid_x - 10
    
    p_complete = get_completion_probability(control_ratio)
    p_int = get_interception_probability(control_ratio)
    p_incomplete = np.maximum(1 - p_complete - p_int, 0)
    
    epa_complete = np.zeros_like(grid_x, dtype=float)
    epa_int = np.zeros_like(grid_x, dtype=float)
    
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            catch_pos = catch_field_pos[i, j]
            epa_complete[i, j] = calculate_epa_complete(catch_pos, current_field_pos, down, yards_to_go)
            epa_int[i, j] = calculate_epa_interception(catch_pos, current_field_pos, down, yards_to_go)
    
    epa_incomplete = calculate_epa_incomplete(current_field_pos, down, yards_to_go)
    
    expected_value = (p_complete * epa_complete + 
                      p_incomplete * epa_incomplete + 
                      p_int * epa_int)
    
    return expected_value, p_complete, p_int, epa_complete, epa_incomplete, epa_int


# ============================================================================
# SPATIAL OPENNESS METRICS
# ============================================================================

def calculate_spatial_metrics(reach_prob, control_ratio, 
                              open_threshold=OPEN_THRESHOLD, 
                              danger_threshold=DANGER_THRESHOLD):
    """
    Calculate spatial openness metrics for a receiver's cloud.
    
    Partitions reachable space by control level:
    - Open Zone: control > open_threshold
    - Contested Zone: between thresholds
    - Danger Zone: control < danger_threshold
    """
    open_mask = control_ratio > open_threshold
    danger_mask = control_ratio < danger_threshold
    contested_mask = ~open_mask & ~danger_mask
    
    total_prob = reach_prob.sum()
    
    if total_prob > 0:
        open_area = (reach_prob * open_mask).sum() / total_prob
        contested_area = (reach_prob * contested_mask).sum() / total_prob
        danger_area = (reach_prob * danger_mask).sum() / total_prob
    else:
        open_area = contested_area = danger_area = 0
    
    if (open_area + danger_area) > 0:
        window_ratio = open_area / (open_area + danger_area)
    else:
        window_ratio = 0.5
    
    if total_prob > 0:
        avg_control = (reach_prob * control_ratio).sum() / total_prob
    else:
        avg_control = 0.5
    
    return {
        'open_area': open_area,
        'contested_area': contested_area,
        'danger_area': danger_area,
        'window_ratio': window_ratio,
        'avg_control': avg_control
    }


# ============================================================================
# VWO V2 - RECEIVER CALCULATION
# ============================================================================

def calculate_receiver_vwo_v2(receiver_row, all_players_frame, los_x, down, yards_to_go,
                               flight_time, grid_spacing=1.0, grid_extent=12,
                               lambda_risk=LAMBDA_RISK):
    """
    Calculate Value-Weighted Openness V2 for a single receiver.
    
    Includes three-outcome expected value, spatial metrics, and risk adjustment.
    """
    # Receiver state
    receiver_x = receiver_row['x_norm']
    receiver_y = receiver_row['y_norm']
    receiver_speed = receiver_row['s'] if pd.notna(receiver_row['s']) else 0
    receiver_dir = receiver_row['dir_norm'] if pd.notna(receiver_row['dir_norm']) else 0
    
    velocity_bin = get_velocity_bin(receiver_speed)
    time_bin = get_time_bin(flight_time)
    
    # Get movement distribution
    if hasattr(get_player_distribution_interpolated, '__call__'):
        dist, _, _, _ = get_player_distribution_interpolated(receiver_speed, flight_time)
    else:
        dist, _, _ = get_player_distribution(receiver_speed, flight_time)
    
    if dist is None:
        return {
            'vwo': 0, 'vwo_complete_only': 0, 'vwo_risk_adjusted': 0,
            'open_area': 0, 'contested_area': 0, 'danger_area': 0,
            'window_ratio': 0.5, 'avg_control': 0.5,
            'avg_completion_prob': 0, 'avg_int_prob': 0,
            'avg_epa_complete': 0, 'epa_incomplete': 0, 'avg_epa_int': 0,
            'sweet_spot_x': receiver_x, 'sweet_spot_y': receiver_y,
            'max_vwo_density': 0, 'velocity_bin': velocity_bin, 'time_bin': time_bin,
            'status': 'no_distribution'
        }
    
    # Create field grid
    x_coords = np.arange(receiver_x - grid_extent, receiver_x + grid_extent + grid_spacing, grid_spacing)
    y_coords = np.arange(
        max(0, receiver_y - grid_extent),
        min(53.3, receiver_y + grid_extent) + grid_spacing,
        grid_spacing
    )
    field_grid_x, field_grid_y = np.meshgrid(x_coords, y_coords)
    
    # Transform movement distribution to field
    reach_density = transform_distribution_to_field(
        dist, receiver_x, receiver_y, receiver_dir,
        field_grid_x, field_grid_y
    )
    
    reach_density_expanded = expand_density_with_catch_radius(
        reach_density, grid_spacing, CATCH_RADIUS
    )
    
    if reach_density_expanded.sum() > 0:
        reach_prob = reach_density_expanded / reach_density_expanded.sum()
    else:
        reach_prob = reach_density_expanded
    
    # Calculate pitch control
    offense_density = np.zeros_like(field_grid_x, dtype=float)
    defense_density = np.zeros_like(field_grid_x, dtype=float)
    
    for _, player in all_players_frame.iterrows():
        player_x = player['x_norm']
        player_y = player['y_norm']
        player_speed = player['s'] if pd.notna(player['s']) else 0
        player_dir = player['dir_norm'] if pd.notna(player['dir_norm']) else 0
        player_side = player['player_side']
        
        if pd.isna(player_x) or pd.isna(player_y):
            continue
        if player_side not in ['Offense', 'Defense']:
            continue
        
        if hasattr(get_player_distribution_interpolated, '__call__'):
            p_dist, _, _, _ = get_player_distribution_interpolated(player_speed, flight_time)
        else:
            p_dist, _, _ = get_player_distribution(player_speed, flight_time)
        
        if p_dist is not None:
            player_density = transform_distribution_to_field(
                p_dist, player_x, player_y, player_dir,
                field_grid_x, field_grid_y
            )
        else:
            player_density = create_fallback_distribution(
                player_x, player_y, player_speed, player_dir,
                flight_time, field_grid_x, field_grid_y
            )
        
        player_expanded = expand_density_with_catch_radius(
            player_density, grid_spacing, CATCH_RADIUS
        )
        
        if player_expanded.max() > 0:
            player_norm = player_expanded / player_expanded.max()
        else:
            player_norm = player_expanded
        
        if player_side == 'Offense':
            offense_density += player_norm
        else:
            defense_density += player_norm
    
    total_density = offense_density + defense_density + CONTROL_EPSILON
    control_ratio = np.clip(offense_density / total_density, 0, 1)
    
    # Three-outcome expected value
    expected_value, p_complete, p_int, epa_complete, epa_incomplete, epa_int = \
        calculate_expected_value_surface(field_grid_x, control_ratio, los_x, down, yards_to_go)
    
    # Spatial metrics
    spatial_metrics = calculate_spatial_metrics(reach_prob, control_ratio)
    
    # VWO calculations
    vwo_density = reach_prob * expected_value
    total_vwo = np.sum(vwo_density) * (grid_spacing ** 2)
    
    vwo_complete_only_density = reach_prob * p_complete * epa_complete
    vwo_complete_only = np.sum(vwo_complete_only_density) * (grid_spacing ** 2)
    
    vwo_risk_adjusted = total_vwo - lambda_risk * spatial_metrics['danger_area']
    
    # Sweet spot
    max_idx = np.unravel_index(np.argmax(vwo_density), vwo_density.shape)
    sweet_spot_x = field_grid_x[max_idx]
    sweet_spot_y = field_grid_y[max_idx]
    max_vwo_density = vwo_density[max_idx]
    
    # Component averages
    total_reach = reach_prob.sum()
    if total_reach > 0:
        avg_completion_prob = (reach_prob * p_complete).sum() / total_reach
        avg_int_prob = (reach_prob * p_int).sum() / total_reach
        avg_epa_complete = (reach_prob * epa_complete).sum() / total_reach
        avg_epa_int = (reach_prob * epa_int).sum() / total_reach
    else:
        avg_completion_prob = avg_int_prob = avg_epa_complete = avg_epa_int = 0
    
    return {
        'vwo': total_vwo,
        'vwo_complete_only': vwo_complete_only,
        'vwo_risk_adjusted': vwo_risk_adjusted,
        'open_area': spatial_metrics['open_area'],
        'contested_area': spatial_metrics['contested_area'],
        'danger_area': spatial_metrics['danger_area'],
        'window_ratio': spatial_metrics['window_ratio'],
        'avg_control': spatial_metrics['avg_control'],
        'avg_completion_prob': avg_completion_prob,
        'avg_int_prob': avg_int_prob,
        'avg_epa_complete': avg_epa_complete,
        'epa_incomplete': epa_incomplete,
        'avg_epa_int': avg_epa_int,
        'sweet_spot_x': sweet_spot_x,
        'sweet_spot_y': sweet_spot_y,
        'max_vwo_density': max_vwo_density,
        'receiver_x': receiver_x,
        'receiver_y': receiver_y,
        'velocity_bin': velocity_bin,
        'time_bin': time_bin,
        'grid_x': field_grid_x,
        'grid_y': field_grid_y,
        'reach_prob': reach_prob,
        'control_ratio': control_ratio,
        'expected_value': expected_value,
        'vwo_density': vwo_density,
        'status': 'success'
    }


# ============================================================================
# PLAY-LEVEL VWO V2
# ============================================================================

def calculate_play_vwo_v2(game_id, play_id, verbose=False, lambda_risk=LAMBDA_RISK):
    """
    Calculate VWO V2 for all eligible receivers on a play.
    Returns optimal target under both risk-neutral and risk-adjusted criteria.
    """
    play_data = df_input[
        (df_input['game_id'] == game_id) & 
        (df_input['play_id'] == play_id)
    ]
    
    if len(play_data) == 0:
        return None
    
    los_x = play_data['absolute_yardline_number'].iloc[0]
    down = play_data['down'].iloc[0]
    yards_to_go = play_data['yards_to_go'].iloc[0]
    pass_result = play_data['pass_result'].iloc[0]
    
    last_frame = get_last_input_frame(game_id, play_id)
    if last_frame is None:
        return None
    
    base_flight_time, _ = get_ball_flight_time(game_id, play_id)
    if base_flight_time is None:
        base_flight_time = 1.5
    flight_time = base_flight_time + (1 / FRAME_RATE)
    
    receivers = last_frame[
        last_frame['player_role'].isin(['Targeted Receiver', 'Other Route Runner'])
    ]
    
    if len(receivers) == 0:
        return None
    
    receiver_vwos = []
    
    for _, receiver in receivers.iterrows():
        vwo_result = calculate_receiver_vwo_v2(
            receiver, last_frame, los_x, down, yards_to_go, flight_time,
            grid_spacing=1.0, grid_extent=12, lambda_risk=lambda_risk
        )
        
        vwo_result['nfl_id'] = receiver['nfl_id']
        vwo_result['player_name'] = receiver.get('player_name', 'Unknown')
        vwo_result['player_role'] = receiver['player_role']
        
        receiver_vwos.append(vwo_result)
    
    vwo_df = pd.DataFrame([{
        'nfl_id': r['nfl_id'],
        'player_name': r['player_name'],
        'player_role': r['player_role'],
        'vwo': r['vwo'],
        'vwo_complete_only': r['vwo_complete_only'],
        'vwo_risk_adjusted': r['vwo_risk_adjusted'],
        'open_area': r['open_area'],
        'danger_area': r['danger_area'],
        'window_ratio': r['window_ratio'],
        'avg_completion_prob': r['avg_completion_prob'],
        'avg_int_prob': r['avg_int_prob'],
        'avg_epa_complete': r['avg_epa_complete']
    } for r in receiver_vwos if r['status'] == 'success'])
    
    if len(vwo_df) == 0:
        return None
    
    optimal_vwo_idx = vwo_df['vwo'].idxmax()
    optimal_risk_adj_idx = vwo_df['vwo_risk_adjusted'].idxmax()
    
    optimal_vwo_target = vwo_df.loc[optimal_vwo_idx]
    optimal_risk_adj_target = vwo_df.loc[optimal_risk_adj_idx]
    
    actual_target = vwo_df[vwo_df['player_role'] == 'Targeted Receiver']
    
    if len(actual_target) == 0:
        actual_vwo = actual_vwo_risk_adj = target_quality = target_quality_risk_adj = 0
    else:
        actual_target = actual_target.iloc[0]
        actual_vwo = actual_target['vwo']
        actual_vwo_risk_adj = actual_target['vwo_risk_adjusted']
        
        target_quality = actual_vwo / optimal_vwo_target['vwo'] if optimal_vwo_target['vwo'] > 0 else 1.0
        target_quality_risk_adj = actual_vwo_risk_adj / optimal_risk_adj_target['vwo_risk_adjusted'] \
            if optimal_risk_adj_target['vwo_risk_adjusted'] > 0 else 1.0
    
    if verbose:
        print(f"\n  Play {game_id}-{play_id}: {down}&{yards_to_go} at {los_x}")
        print(f"  Receivers analyzed: {len(vwo_df)}")
        print(f"\n  VWO Rankings:")
        for _, row in vwo_df.sort_values('vwo', ascending=False).iterrows():
            role_marker = "â†’ " if row['player_role'] == 'Targeted Receiver' else "  "
            print(f"    {role_marker}{row['player_name']}: VWO={row['vwo']:.3f}, "
                  f"Risk-Adj={row['vwo_risk_adjusted']:.3f}, "
                  f"Window={row['window_ratio']:.2f}")
        print(f"\n  Optimal (VWO): {optimal_vwo_target['player_name']} ({optimal_vwo_target['vwo']:.3f})")
        print(f"  Target Quality: {target_quality:.1%}")
    
    return {
        'game_id': game_id,
        'play_id': play_id,
        'down': down,
        'yards_to_go': yards_to_go,
        'los_x': los_x,
        'pass_result': pass_result,
        'flight_time': flight_time,
        'n_receivers': len(vwo_df),
        'receiver_vwos': receiver_vwos,
        'vwo_summary': vwo_df,
        'optimal_target_id': optimal_vwo_target['nfl_id'],
        'optimal_target_name': optimal_vwo_target['player_name'],
        'optimal_vwo': optimal_vwo_target['vwo'],
        'optimal_risk_adj_id': optimal_risk_adj_target['nfl_id'],
        'optimal_risk_adj_name': optimal_risk_adj_target['player_name'],
        'optimal_vwo_risk_adj': optimal_risk_adj_target['vwo_risk_adjusted'],
        'actual_target_id': actual_target['nfl_id'] if len(actual_target) > 0 else None,
        'actual_target_name': actual_target['player_name'] if len(actual_target) > 0 else None,
        'actual_vwo': actual_vwo,
        'actual_vwo_risk_adj': actual_vwo_risk_adj,
        'actual_window_ratio': actual_target['window_ratio'] if len(actual_target) > 0 else 0,
        'actual_danger_area': actual_target['danger_area'] if len(actual_target) > 0 else 0,
        'target_quality': target_quality,
        'target_quality_risk_adj': target_quality_risk_adj,
        'chose_optimal': optimal_vwo_target['nfl_id'] == (actual_target['nfl_id'] if len(actual_target) > 0 else None),
        'chose_optimal_risk_adj': optimal_risk_adj_target['nfl_id'] == (actual_target['nfl_id'] if len(actual_target) > 0 else None)
    }


# ============================================================================
# BATCH ANALYSIS
# ============================================================================

def analyze_vwo_v2_batch(sample_size=100, verbose=True, lambda_risk=LAMBDA_RISK):
    """Analyze VWO V2 for a batch of plays."""
    print(f"\n  Analyzing {sample_size} plays with VWO V2...")
    
    plays = df_input.groupby(['game_id', 'play_id']).first().reset_index()
    
    if sample_size and sample_size < len(plays):
        plays = plays.sample(sample_size, random_state=42)
    
    results = []
    errors = 0
    
    for idx, (_, play) in enumerate(plays.iterrows()):
        try:
            result = calculate_play_vwo_v2(
                play['game_id'], play['play_id'], 
                verbose=False, lambda_risk=lambda_risk
            )
            
            if result is not None:
                results.append({
                    'game_id': result['game_id'],
                    'play_id': result['play_id'],
                    'down': result['down'],
                    'yards_to_go': result['yards_to_go'],
                    'pass_result': result['pass_result'],
                    'n_receivers': result['n_receivers'],
                    'optimal_vwo': result['optimal_vwo'],
                    'optimal_vwo_risk_adj': result['optimal_vwo_risk_adj'],
                    'actual_vwo': result['actual_vwo'],
                    'actual_vwo_risk_adj': result['actual_vwo_risk_adj'],
                    'actual_window_ratio': result['actual_window_ratio'],
                    'actual_danger_area': result['actual_danger_area'],
                    'target_quality': result['target_quality'],
                    'target_quality_risk_adj': result['target_quality_risk_adj'],
                    'chose_optimal': result['chose_optimal'],
                    'chose_optimal_risk_adj': result['chose_optimal_risk_adj'],
                    'optimal_target': result['optimal_target_name'],
                    'actual_target': result['actual_target_name']
                })
        except:
            errors += 1
            continue
        
        if verbose and (idx + 1) % 20 == 0:
            print(f"    Processed {idx + 1}/{len(plays)} plays...")
    
    df_results = pd.DataFrame(results)
    print(f"\n  âœ“ Analyzed {len(df_results)} plays ({errors} errors)")
    
    return df_results


def summarize_vwo_v2(df_results):
    """Summarize VWO V2 results."""
    print("\n" + "=" * 70)
    print("VWO V2 ANALYSIS SUMMARY")
    print("=" * 70)
    
    n = len(df_results)
    
    print(f"\n  DECISION QUALITY ({n} plays)")
    print("  " + "-" * 50)
    print(f"  Mean Target Quality:    {df_results['target_quality'].mean():.1%}")
    print(f"  % Chose Optimal:        {df_results['chose_optimal'].mean():.1%}")
    print(f"  Mean VWO Gap:           {(df_results['optimal_vwo'] - df_results['actual_vwo']).mean():.3f}")
    
    print(f"\n  BY OUTCOME")
    print("  " + "-" * 50)
    for outcome, label in [('C', 'Complete'), ('I', 'Incomplete'), ('IN', 'INT')]:
        subset = df_results[df_results['pass_result'] == outcome]
        if len(subset) > 3:
            print(f"  {label:<12} n={len(subset):<5} Quality={subset['target_quality'].mean():.1%}")
    
    return df_results


# ============================================================================
# READY
# ============================================================================

print("\n" + "=" * 70)
print("âœ“ VWO V2 SYSTEM READY")
print("=" * 70)

print("""
FUNCTIONS:
  â€¢ calculate_play_vwo_v2(game_id, play_id, verbose=True)
  â€¢ analyze_vwo_v2_batch(sample_size=100)
  â€¢ summarize_vwo_v2(df_results)

QUICK START:
  result = calculate_play_vwo_v2(game_id, play_id, verbose=True)
""")


# # Cell: VWO V2 Batch Analysis and Target Rank Analysis
# # ============================================================================

# print("=" * 70)
# print("VWO V2 BATCH ANALYSIS")
# print("=" * 70)

# # ============================================================================
# # STEP 1: RUN BATCH ANALYSIS (creates df_vwo_v2)
# # ============================================================================

# # Adjust sample_size as needed (larger = slower but more robust)
# df_vwo_v2 = analyze_vwo_v2_batch(sample_size=5000, verbose=True)

# # Quick summary
# summarize_vwo_v2(df_vwo_v2)


# # ============================================================================
# # STEP 2: TARGET RANK ANALYSIS
# # ============================================================================

# def analyze_target_rank_v2(df_results):
#     """
#     Rank-based analysis that handles negative VWO.
#     """
#     print("\n" + "=" * 70)
#     print("TARGET RANK ANALYSIS (VWO V2)")
#     print("=" * 70)
    
#     # Already have chose_optimal
#     print(f"\n  % Chose Optimal (Risk-Neutral): {df_results['chose_optimal'].mean():.1%}")
#     print(f"  % Chose Optimal (Risk-Adjusted): {df_results['chose_optimal_risk_adj'].mean():.1%}")
    
#     # VWO Gap analysis
#     df_results['vwo_gap'] = df_results['optimal_vwo'] - df_results['actual_vwo']
#     df_results['vwo_gap_risk_adj'] = df_results['optimal_vwo_risk_adj'] - df_results['actual_vwo_risk_adj']
    
#     print(f"\n  VWO Gap Analysis:")
#     print(f"    Mean Gap (Risk-Neutral): {df_results['vwo_gap'].mean():.3f}")
#     print(f"    Mean Gap (Risk-Adjusted): {df_results['vwo_gap_risk_adj'].mean():.3f}")
    
#     # By outcome
#     print(f"\n  By Outcome:")
#     print(f"  {'Outcome':<12} {'% Optimal':<12} {'% Optimal (RA)':<16} {'Mean Gap':<12} {'Window Ratio':<14}")
#     print("  " + "-" * 66)
    
#     for outcome, label in [('C', 'Complete'), ('I', 'Incomplete'), ('IN', 'INT')]:
#         subset = df_results[df_results['pass_result'] == outcome]
#         if len(subset) > 3:
#             pct_opt = subset['chose_optimal'].mean() * 100
#             pct_opt_ra = subset['chose_optimal_risk_adj'].mean() * 100
#             gap = subset['vwo_gap'].mean()
#             window = subset['actual_window_ratio'].mean()
#             print(f"  {label:<12} {pct_opt:>6.1f}%      {pct_opt_ra:>6.1f}%          {gap:>8.3f}     {window:>8.2f}")
    
#     # Correlation: Window Ratio vs being targeted
#     print(f"\n  Correlation Analysis:")
#     print(f"    Window Ratio â†” Chose Optimal: {df_results['actual_window_ratio'].corr(df_results['chose_optimal']):.3f}")
#     print(f"    Danger Area â†” Chose Optimal: {df_results['actual_danger_area'].corr(df_results['chose_optimal']):.3f}")
    
#     # Plays where optimal VWO > 0
#     positive_optimal = df_results[df_results['optimal_vwo'] > 0]
#     print(f"\n  Plays with Positive Optimal VWO (n={len(positive_optimal)}):")
#     print(f"    % Chose Optimal: {positive_optimal['chose_optimal'].mean():.1%}")
#     print(f"    Mean Actual VWO: {positive_optimal['actual_vwo'].mean():.3f}")
    
#     # Plays where actual target had negative VWO
#     negative_actual = df_results[df_results['actual_vwo'] < 0]
#     print(f"\n  Plays with Negative Actual VWO (n={len(negative_actual)}):")
#     if len(negative_actual) > 0:
#         print(f"    Mean Actual VWO: {negative_actual['actual_vwo'].mean():.3f}")
#         print(f"    Mean Optimal VWO: {negative_actual['optimal_vwo'].mean():.3f}")
#         print(f"    % Complete: {(negative_actual['pass_result'] == 'C').mean():.1%}")
#         print(f"    % INT: {(negative_actual['pass_result'] == 'IN').mean():.1%}")
    
#     # Target quality distribution
#     print(f"\n  Target Quality Distribution:")
#     bins = [(0, 0.5, '<50%'), (0.5, 0.8, '50-80%'), (0.8, 1.0, '80-100%'), (1.0, float('inf'), 'â‰¥100%')]
#     for low, high, label in bins:
#         count = ((df_results['target_quality'] >= low) & (df_results['target_quality'] < high)).sum()
#         pct = count / len(df_results) * 100
#         print(f"    {label:<10}: {count:>5} plays ({pct:>5.1f}%)")
    
#     return df_results


# # Run the target rank analysis
# df_vwo_v2 = analyze_target_rank_v2(df_vwo_v2)

# print("\n" + "=" * 70)
# print("âœ“ ANALYSIS COMPLETE")
# print("=" * 70)


print("=" * 70)
print("VWO V2 Analysis Summary (PRE-COMPUTED)")
print("=" * 70)

print("""

======================================================================
VWO V2 ANALYSIS SUMMARY
======================================================================

  DECISION QUALITY (5000 plays)
  --------------------------------------------------
  Mean Target Quality:    5.5%
  % Chose Optimal:        18.0%
  Mean VWO Gap:           0.688

  BY OUTCOME
  --------------------------------------------------
  Complete     n=3397  Quality=-11.3%
  Incomplete   n=1473  Quality=38.7%
  INT          n=130   Quality=66.9%

======================================================================
TARGET RANK ANALYSIS (VWO V2)
======================================================================

  % Chose Optimal (Risk-Neutral): 18.0%
  % Chose Optimal (Risk-Adjusted): 18.1%

  VWO Gap Analysis:
    Mean Gap (Risk-Neutral): 0.688
    Mean Gap (Risk-Adjusted): 0.691

  By Outcome:
  Outcome      % Optimal    % Optimal (RA)   Mean Gap     Window Ratio  
  ------------------------------------------------------------------
  Complete       14.9%        15.4%             0.765         0.61
  Incomplete     24.2%        23.6%             0.536         0.35
  INT            28.5%        24.6%             0.370         0.29

  Correlation Analysis:
    Window Ratio â†” Chose Optimal: -0.233
    Danger Area â†” Chose Optimal: 0.200

  Plays with Positive Optimal VWO (n=3636):
    % Chose Optimal: 17.7%
    Mean Actual VWO: 0.793

  Plays with Negative Actual VWO (n=2416):
    Mean Actual VWO: -2.093
    Mean Optimal VWO: -1.094
    % Complete: 74.0%
    % INT: 2.0%

  Target Quality Distribution:
    <50%      :   640 plays ( 12.8%)
    50-80%    :   601 plays ( 12.0%)
    80-100%   :   698 plays ( 14.0%)
    â‰¥100%     :  2009 plays ( 40.2%)

======================================================================
âœ“ ANALYSIS COMPLETE
======================================================================
""")

print("=" * 70)


# # Diagnostic 1: Compare decision patterns by down

# def analyze_by_down(df_results):
#     """
#     Compare QB decision-making on early downs (1st/2nd) vs late downs (3rd/4th)
#     """
#     print("=" * 70)
#     print("DECISION QUALITY BY DOWN")
#     print("=" * 70)
    
#     # Split by down type
#     df_results['down_type'] = df_results['down'].apply(
#         lambda x: 'Early (1st/2nd)' if x <= 2 else 'Late (3rd/4th)'
#     )
    
#     print(f"\n  {'Down Type':<18} {'N':<8} {'% Optimal':<12} {'VWO Gap':<12} {'Window Ratio':<14} {'Danger Area':<12}")
#     print("  " + "-" * 76)
    
#     for down_type in ['Early (1st/2nd)', 'Late (3rd/4th)']:
#         subset = df_results[df_results['down_type'] == down_type]
#         if len(subset) > 5:
#             pct_opt = subset['chose_optimal'].mean() * 100
#             gap = subset['vwo_gap'].mean()
#             window = subset['actual_window_ratio'].mean()
#             danger = subset['actual_danger_area'].mean()
#             print(f"  {down_type:<18} {len(subset):<8} {pct_opt:>6.1f}%      {gap:>8.3f}     {window:>8.2f}       {danger:>8.3f}")
    
#     # Individual downs
#     print(f"\n  By Individual Down:")
#     print(f"  {'Down':<8} {'N':<8} {'% Optimal':<12} {'VWO Gap':<12} {'Window Ratio':<14} {'Mean Actual VWO':<16}")
#     print("  " + "-" * 70)
    
#     for down in [1, 2, 3, 4]:
#         subset = df_results[df_results['down'] == down]
#         if len(subset) > 3:
#             pct_opt = subset['chose_optimal'].mean() * 100
#             gap = subset['vwo_gap'].mean()
#             window = subset['actual_window_ratio'].mean()
#             actual_vwo = subset['actual_vwo'].mean()
#             print(f"  {down:<8} {len(subset):<8} {pct_opt:>6.1f}%      {gap:>8.3f}     {window:>8.2f}       {actual_vwo:>10.3f}")
    
#     # Correlation by down type
#     print(f"\n  Correlation (Window Ratio with Chose Optimal):")
#     for down_type in ['Early (1st/2nd)', 'Late (3rd/4th)']:
#         subset = df_results[df_results['down_type'] == down_type]
#         if len(subset) > 10:
#             corr = subset['actual_window_ratio'].corr(subset['chose_optimal'])
#             print(f"    {down_type}: {corr:.3f}")
    
#     return df_results

# analyze_by_down(df_vwo_v2)


print("=" * 70)
print("Decision Quality by Down (PRE-COMPUTED)")
print("=" * 70)

print("""

======================================================================
DECISION QUALITY BY DOWN
======================================================================

  Down Type          N        % Optimal    VWO Gap      Window Ratio   Danger Area 
  ----------------------------------------------------------------------------
  Early (1st/2nd)    3547       17.4%         0.622         0.54          0.168
  Late (3rd/4th)     1453       19.3%         0.847         0.49          0.180

  By Individual Down:
  Down     N        % Optimal    VWO Gap      Window Ratio   Mean Actual VWO 
  ----------------------------------------------------------------------
  1        1871       17.2%         0.614         0.55           -0.267
  2        1676       17.6%         0.632         0.54           -0.080
  3        1321       19.5%         0.856         0.49           -0.786
  4        132        17.4%         0.759         0.46            1.500

  Correlation (Window Ratio with Chose Optimal):
    Early (1st/2nd): -0.251
    Late (3rd/4th): -0.190
""")

print("=" * 70)


# # Diagnostic 2: 3rd/4th down with yards-to-go analysis

# def analyze_late_downs_detail(df_results):
#     """
#     Deep dive into 3rd and 4th down decisions.
#     Key question: Do QBs seek value more when they NEED yards?
#     """
#     print("=" * 70)
#     print("3RD AND 4TH DOWN DETAILED ANALYSIS")
#     print("=" * 70)
    
#     # Filter to 3rd and 4th down only
#     late_downs = df_results[df_results['down'] >= 3].copy()
    
#     print(f"\n  Total 3rd/4th down plays: {len(late_downs)}")
    
#     # Categorize by yards to go
#     def ytg_category(ytg):
#         if ytg <= 3:
#             return 'Short (1-3)'
#         elif ytg <= 7:
#             return 'Medium (4-7)'
#         else:
#             return 'Long (8+)'
    
#     late_downs['ytg_category'] = late_downs['yards_to_go'].apply(ytg_category)
    
#     print(f"\n  BY YARDS TO GO:")
#     print(f"  {'YTG Category':<15} {'N':<8} {'% Optimal':<12} {'VWO Gap':<12} {'Window Ratio':<14} {'Completion %':<12}")
#     print("  " + "-" * 73)
    
#     for cat in ['Short (1-3)', 'Medium (4-7)', 'Long (8+)']:
#         subset = late_downs[late_downs['ytg_category'] == cat]
#         if len(subset) > 3:
#             pct_opt = subset['chose_optimal'].mean() * 100
#             gap = subset['vwo_gap'].mean()
#             window = subset['actual_window_ratio'].mean()
#             comp_pct = (subset['pass_result'] == 'C').mean() * 100
#             print(f"  {cat:<15} {len(subset):<8} {pct_opt:>6.1f}%      {gap:>8.3f}     {window:>8.2f}       {comp_pct:>6.1f}%")
    
#     # By outcome on 3rd/4th down
#     print(f"\n  BY OUTCOME (3rd/4th down only):")
#     print(f"  {'Outcome':<12} {'N':<8} {'% Optimal':<12} {'VWO Gap':<12} {'Window Ratio':<14}")
#     print("  " + "-" * 58)
    
#     for outcome, label in [('C', 'Complete'), ('I', 'Incomplete')]:
#         subset = late_downs[late_downs['pass_result'] == outcome]
#         if len(subset) > 3:
#             pct_opt = subset['chose_optimal'].mean() * 100
#             gap = subset['vwo_gap'].mean()
#             window = subset['actual_window_ratio'].mean()
#             print(f"  {label:<12} {len(subset):<8} {pct_opt:>6.1f}%      {gap:>8.3f}     {window:>8.2f}")
    
#     # Compare to early downs
#     early_downs = df_results[df_results['down'] <= 2]
    
#     print(f"\n  COMPARISON SUMMARY:")
#     print(f"  " + "-" * 50)
#     print(f"  {'Metric':<25} {'1st/2nd Down':<15} {'3rd/4th Down':<15}")
#     print(f"  " + "-" * 55)
    
#     metrics = [
#         ('% Chose Optimal', 'chose_optimal', lambda x: f"{x.mean()*100:.1f}%"),
#         ('Mean VWO Gap', 'vwo_gap', lambda x: f"{x.mean():.3f}"),
#         ('Mean Window Ratio', 'actual_window_ratio', lambda x: f"{x.mean():.2f}"),
#         ('Mean Danger Area', 'actual_danger_area', lambda x: f"{x.mean():.3f}"),
#         ('Mean Actual VWO', 'actual_vwo', lambda x: f"{x.mean():.3f}"),
#     ]
    
#     for label, col, fmt in metrics:
#         early_val = fmt(early_downs[col]) if len(early_downs) > 0 else "N/A"
#         late_val = fmt(late_downs[col]) if len(late_downs) > 0 else "N/A"
#         print(f"  {label:<25} {early_val:<15} {late_val:<15}")
    
#     # Test hypothesis: On 3rd & long, do QBs go for value more?
#     print(f"\n  HYPOTHESIS TEST: 3rd & Long (8+ yards)")
#     third_long = df_results[(df_results['down'] == 3) & (df_results['yards_to_go'] >= 8)]
#     if len(third_long) > 3:
#         print(f"    N plays: {len(third_long)}")
#         print(f"    % Chose Optimal: {third_long['chose_optimal'].mean()*100:.1f}%")
#         print(f"    Mean Window Ratio: {third_long['actual_window_ratio'].mean():.2f}")
#         print(f"    Mean VWO Gap: {third_long['vwo_gap'].mean():.3f}")
#         print(f"    Completion Rate: {(third_long['pass_result']=='C').mean()*100:.1f}%")
#     else:
#         print(f"    Not enough plays (n={len(third_long)})")
    
#     return late_downs

# late_down_analysis = analyze_late_downs_detail(df_vwo_v2)


print("=" * 70)
print("3RD and 4TH down detailed analysis (PRE-COMPUTED)")
print("=" * 70)

print("""

======================================================================
3RD AND 4TH DOWN DETAILED ANALYSIS
======================================================================

  Total 3rd/4th down plays: 1453

  BY YARDS TO GO:
  YTG Category    N        % Optimal    VWO Gap      Window Ratio   Completion %
  -------------------------------------------------------------------------
  Short (1-3)     348        21.8%         0.634         0.55         65.8%
  Medium (4-7)    524        18.7%         0.737         0.45         63.0%
  Long (8+)       581        18.4%         1.075         0.49         60.2%

  BY OUTCOME (3rd/4th down only):
  Outcome      N        % Optimal    VWO Gap      Window Ratio  
  ----------------------------------------------------------
  Complete     909        17.7%         0.951         0.58
  Incomplete   497        21.1%         0.697         0.34

  COMPARISON SUMMARY:
  --------------------------------------------------
  Metric                    1st/2nd Down    3rd/4th Down   
  -------------------------------------------------------
  % Chose Optimal           17.4%           19.3%          
  Mean VWO Gap              0.622           0.847          
  Mean Window Ratio         0.54            0.49           
  Mean Danger Area          0.168           0.180          
  Mean Actual VWO           -0.178          -0.579         

  HYPOTHESIS TEST: 3rd & Long (8+ yards)
    N plays: 543
    % Chose Optimal: 18.8%
    Mean Window Ratio: 0.50
    Mean VWO Gap: 1.072
    Completion Rate: 62.2%
""")

print("=" * 70)


# # Diagnostic: Do QBs choose based on openness rather than value?

# def analyze_openness_vs_value(sample_size=5000):
#     """
#     Test whether QBs rank targets by openness metrics rather than VWO.
    
#     Compare % optimal under different ranking criteria:
#     - VWO (value-weighted)
#     - Window Ratio (pure openness/safety)
#     - Open Area (amount of open space)
#     - Avg Control (mean control in cloud)
#     - Avg Completion Probability
#     """
#     print("=" * 70)
#     print("WHAT DRIVES QB TARGET SELECTION?")
#     print("=" * 70)
    
#     # Get plays with full receiver data
#     plays = df_input.groupby(['game_id', 'play_id']).first().reset_index()
#     if sample_size and sample_size < len(plays):
#         plays = plays.sample(sample_size, random_state=42)
    
#     results = []
    
#     print(f"\n  Analyzing {len(plays)} plays...")
    
#     for idx, (_, play) in enumerate(plays.iterrows()):
#         try:
#             result = calculate_play_vwo_v2(
#                 play['game_id'], play['play_id'], 
#                 verbose=False, lambda_risk=LAMBDA_RISK
#             )
            
#             if result is None or len(result['vwo_summary']) < 2:
#                 continue
            
#             vwo_df = result['vwo_summary'].copy()
            
#             # Find actual target
#             actual = vwo_df[vwo_df['player_role'] == 'Targeted Receiver']
#             if len(actual) == 0:
#                 continue
#             actual_id = actual.iloc[0]['nfl_id']
            
#             # Rank by different metrics (higher = better for all)
#             rankings = {}
            
#             # VWO (current)
#             vwo_df['rank_vwo'] = vwo_df['vwo'].rank(ascending=False)
#             optimal_vwo = vwo_df.loc[vwo_df['vwo'].idxmax(), 'nfl_id']
#             rankings['vwo'] = optimal_vwo == actual_id
            
#             # Window Ratio (openness/safety)
#             vwo_df['rank_window'] = vwo_df['window_ratio'].rank(ascending=False)
#             optimal_window = vwo_df.loc[vwo_df['window_ratio'].idxmax(), 'nfl_id']
#             rankings['window_ratio'] = optimal_window == actual_id
            
#             # Open Area
#             vwo_df['rank_open_area'] = vwo_df['open_area'].rank(ascending=False)
#             optimal_open = vwo_df.loc[vwo_df['open_area'].idxmax(), 'nfl_id']
#             rankings['open_area'] = optimal_open == actual_id
            
#             # Avg Control
#             avg_control_col = 'avg_control' if 'avg_control' in vwo_df.columns else None
#             if avg_control_col:
#                 # Need to get avg_control from the full receiver results
#                 pass
            
#             # Avg Completion Probability
#             vwo_df['rank_comp_prob'] = vwo_df['avg_completion_prob'].rank(ascending=False)
#             optimal_comp = vwo_df.loc[vwo_df['avg_completion_prob'].idxmax(), 'nfl_id']
#             rankings['completion_prob'] = optimal_comp == actual_id
            
#             # Danger Area (lower = better, so rank ascending)
#             vwo_df['rank_danger'] = vwo_df['danger_area'].rank(ascending=True)  # Lower is better
#             optimal_safe = vwo_df.loc[vwo_df['danger_area'].idxmin(), 'nfl_id']
#             rankings['lowest_danger'] = optimal_safe == actual_id
            
#             # EPA if complete (pure value, ignoring probability)
#             vwo_df['rank_epa'] = vwo_df['avg_epa_complete'].rank(ascending=False)
#             optimal_epa = vwo_df.loc[vwo_df['avg_epa_complete'].idxmax(), 'nfl_id']
#             rankings['pure_epa'] = optimal_epa == actual_id
            
#             # Get actual target's ranks
#             actual_row = vwo_df[vwo_df['nfl_id'] == actual_id].iloc[0]
            
#             results.append({
#                 'game_id': play['game_id'],
#                 'play_id': play['play_id'],
#                 'down': result['down'],
#                 'n_receivers': len(vwo_df),
#                 'pass_result': result['pass_result'],
                
#                 # Did QB choose optimal under each criterion?
#                 'chose_optimal_vwo': rankings['vwo'],
#                 'chose_optimal_window': rankings['window_ratio'],
#                 'chose_optimal_open_area': rankings['open_area'],
#                 'chose_optimal_comp_prob': rankings['completion_prob'],
#                 'chose_optimal_lowest_danger': rankings['lowest_danger'],
#                 'chose_optimal_pure_epa': rankings['pure_epa'],
                
#                 # Actual target's rank under each criterion
#                 'rank_by_vwo': actual_row['rank_vwo'],
#                 'rank_by_window': actual_row['rank_window'],
#                 'rank_by_open_area': actual_row['rank_open_area'],
#                 'rank_by_comp_prob': actual_row['rank_comp_prob'],
#                 'rank_by_danger': actual_row['rank_danger'],
#                 'rank_by_epa': actual_row['rank_epa'],
                
#                 # Actual values
#                 'actual_vwo': actual_row['vwo'],
#                 'actual_window_ratio': actual_row['window_ratio'],
#                 'actual_comp_prob': actual_row['avg_completion_prob'],
#             })
            
#         except Exception as e:
#             continue
        
#         if (idx + 1) % 50 == 0:
#             print(f"    Processed {idx + 1}/{len(plays)} plays...")
    
#     df = pd.DataFrame(results)
    
#     # Summary
#     print(f"\n  Analyzed {len(df)} plays with 2+ receivers")
    
#     print(f"\n" + "-" * 70)
#     print("  WHICH METRIC BEST PREDICTS QB TARGET SELECTION?")
#     print("-" * 70)
    
#     metrics = [
#         ('Window Ratio (Openness)', 'chose_optimal_window', 'rank_by_window'),
#         ('Open Area', 'chose_optimal_open_area', 'rank_by_open_area'),
#         ('Completion Probability', 'chose_optimal_comp_prob', 'rank_by_comp_prob'),
#         ('Lowest Danger Area', 'chose_optimal_lowest_danger', 'rank_by_danger'),
#         ('VWO (Value-Weighted)', 'chose_optimal_vwo', 'rank_by_vwo'),
#         ('Pure EPA (Value Only)', 'chose_optimal_pure_epa', 'rank_by_epa'),
#     ]
    
#     print(f"\n  {'Ranking Criterion':<30} {'% Chose #1':<15} {'Mean Rank':<15}")
#     print("  " + "-" * 60)
    
#     for label, opt_col, rank_col in metrics:
#         pct_optimal = df[opt_col].mean() * 100
#         mean_rank = df[rank_col].mean()
#         print(f"  {label:<30} {pct_optimal:>8.1f}%       {mean_rank:>8.2f}")
    
#     # By down
#     print(f"\n" + "-" * 70)
#     print("  % CHOSE OPTIMAL BY DOWN")
#     print("-" * 70)
    
#     print(f"\n  {'Criterion':<25} {'1st':<10} {'2nd':<10} {'3rd':<10} {'4th':<10}")
#     print("  " + "-" * 65)
    
#     for label, opt_col, _ in metrics[:4]:  # Top 4 metrics
#         row = f"  {label:<25}"
#         for down in [1, 2, 3, 4]:
#             subset = df[df['down'] == down]
#             if len(subset) > 3:
#                 pct = subset[opt_col].mean() * 100
#                 row += f" {pct:>6.1f}%   "
#             else:
#                 row += f"    N/A    "
#         print(row)
    
#     # By outcome
#     print(f"\n" + "-" * 70)
#     print("  % CHOSE OPTIMAL BY OUTCOME")
#     print("-" * 70)
    
#     print(f"\n  {'Criterion':<30} {'Complete':<15} {'Incomplete':<15}")
#     print("  " + "-" * 60)
    
#     for label, opt_col, _ in metrics[:4]:
#         complete = df[df['pass_result'] == 'C']
#         incomplete = df[df['pass_result'] == 'I']
        
#         pct_c = complete[opt_col].mean() * 100 if len(complete) > 0 else 0
#         pct_i = incomplete[opt_col].mean() * 100 if len(incomplete) > 0 else 0
        
#         print(f"  {label:<30} {pct_c:>8.1f}%       {pct_i:>8.1f}%")
    
#     # Correlation between metrics and being targeted
#     print(f"\n" + "-" * 70)
#     print("  RANDOM BASELINE COMPARISON")
#     print("-" * 70)
    
#     # Calculate expected random rate based on number of receivers
#     random_rate = (1 / df['n_receivers']).mean() * 100
#     print(f"\n  Expected random selection rate: {random_rate:.1f}%")
#     print(f"\n  Metrics that beat random:")
    
#     for label, opt_col, _ in metrics:
#         pct = df[opt_col].mean() * 100
#         diff = pct - random_rate
#         if diff > 0:
#             print(f"    {label}: {pct:.1f}% (+{diff:.1f}% vs random)")
    
#     return df

# df_openness = analyze_openness_vs_value(sample_size=5000)


print("=" * 70)
print("Openness vs Value")
print("=" * 70)

print("""
  Analyzed 5000 plays with 2+ receivers

----------------------------------------------------------------------
  WHICH METRIC BEST PREDICTS QB TARGET SELECTION?
----------------------------------------------------------------------

  Ranking Criterion              % Chose #1      Mean Rank      
  ------------------------------------------------------------
  Window Ratio (Openness)            23.8%           2.73
  Open Area                          25.5%           2.67
  Completion Probability             25.1%           2.69
  Lowest Danger Area                 22.1%           2.81
  VWO (Value-Weighted)               18.0%           2.89
  Pure EPA (Value Only)              16.9%           2.92

----------------------------------------------------------------------
  % CHOSE OPTIMAL BY DOWN
----------------------------------------------------------------------

  Criterion                 1st        2nd        3rd        4th       
  -----------------------------------------------------------------
  Window Ratio (Openness)     24.4%      23.0%      23.9%      23.5%   
  Open Area                   25.9%      24.2%      26.9%      25.0%   
  Completion Probability      25.3%      23.5%      26.8%      26.5%   
  Lowest Danger Area          23.3%      21.5%      21.1%      23.5%   

----------------------------------------------------------------------
  % CHOSE OPTIMAL BY OUTCOME
----------------------------------------------------------------------

  Criterion                      Complete        Incomplete     
  ------------------------------------------------------------
  Window Ratio (Openness)            28.3%           14.9%
  Open Area                          30.2%           16.3%
  Completion Probability             29.5%           16.6%
  Lowest Danger Area                 24.8%           17.0%

----------------------------------------------------------------------
  RANDOM BASELINE COMPARISON
----------------------------------------------------------------------

  Expected random selection rate: 22.4%

  Metrics that beat random:
    Window Ratio (Openness): 23.8% (+1.4% vs random)
    Open Area: 25.5% (+3.2% vs random)
    Completion Probability: 25.1% (+2.8% vs random)
""")

print("=" * 70)


# # Part 1: QB-Weighted Openness (QWO) - Find weights that predict target selection

# def build_qb_weighted_openness(sample_size=5000):
#     """
#     Build a QB-Weighted Openness score that predicts which receiver QBs target.
    
#     Uses logistic regression to find optimal weights for:
#     - Completion Probability
#     - Open Area
#     - Window Ratio
#     - Danger Area (negative weight expected)
#     - VWO (to see if value contributes at all)
#     """
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.preprocessing import StandardScaler
    
#     print("=" * 70)
#     print("BUILDING QB-WEIGHTED OPENNESS (QWO) MODEL")
#     print("=" * 70)
    
#     # Collect receiver-level data
#     plays = df_input.groupby(['game_id', 'play_id']).first().reset_index()
#     if sample_size and sample_size < len(plays):
#         plays = plays.sample(sample_size, random_state=42)
    
#     receiver_data = []
#     errors = 0
    
#     print(f"\n  Collecting data from {len(plays)} plays...")
    
#     for idx, (_, play) in enumerate(plays.iterrows()):
#         try:
#             result = calculate_play_vwo_v2(
#                 play['game_id'], play['play_id'], 
#                 verbose=False, lambda_risk=LAMBDA_RISK
#             )
            
#             if result is None:
#                 errors += 1
#                 continue
            
#             vwo_df = result['vwo_summary']
#             if len(vwo_df) < 2:
#                 continue
            
#             # Get QB info - look for "Passer" in player_role
#             play_data = df_input[
#                 (df_input['game_id'] == play['game_id']) & 
#                 (df_input['play_id'] == play['play_id'])
#             ]
            
#             qb_name = 'Unknown'
#             qb_id = None
            
#             # Find the Passer
#             qb_row = play_data[play_data['player_role'] == 'Passer']
#             if len(qb_row) > 0:
#                 qb_name = qb_row['player_name'].iloc[0] if 'player_name' in qb_row.columns else 'Unknown'
#                 qb_id = qb_row['nfl_id'].iloc[0] if 'nfl_id' in qb_row.columns else None
            
#             for _, rec in vwo_df.iterrows():
#                 receiver_data.append({
#                     'game_id': play['game_id'],
#                     'play_id': play['play_id'],
#                     'qb_name': qb_name,
#                     'qb_id': qb_id,
#                     'receiver_name': rec['player_name'],
#                     'nfl_id': rec['nfl_id'],
#                     'is_target': 1 if rec['player_role'] == 'Targeted Receiver' else 0,
#                     'pass_result': result['pass_result'],
#                     'down': result['down'],
#                     'n_receivers': len(vwo_df),
                    
#                     # Features
#                     'completion_prob': rec['avg_completion_prob'],
#                     'open_area': rec['open_area'],
#                     'window_ratio': rec['window_ratio'],
#                     'danger_area': rec['danger_area'],
#                     'vwo': rec['vwo'],
#                     'avg_epa_complete': rec['avg_epa_complete'],
#                 })
            
#         except Exception as e:
#             errors += 1
#             if errors <= 5:
#                 print(f"    Error on play {play['game_id']}-{play['play_id']}: {e}")
#             continue
        
#         if (idx + 1) % 100 == 0:
#             print(f"    Processed {idx + 1}/{len(plays)} plays... ({len(receiver_data)} receivers collected)")
    
#     # Check if we got any data
#     if len(receiver_data) == 0:
#         print(f"\n  ERROR: No receiver data collected! ({errors} errors)")
#         print("  Check that calculate_play_vwo_v2 is working correctly.")
#         return None, None, None, None
    
#     df_rec = pd.DataFrame(receiver_data)
    
#     n_plays = df_rec['play_id'].nunique()
#     n_receivers = len(df_rec)
#     print(f"\n  Collected {n_receivers} receiver observations from {n_plays} plays ({errors} errors)")
    
#     # Prepare features for logistic regression
#     feature_cols = ['completion_prob', 'open_area', 'window_ratio', 'danger_area', 'vwo']
    
#     X = df_rec[feature_cols].values
#     y = df_rec['is_target'].values
    
#     # Handle any NaN
#     mask = ~np.isnan(X).any(axis=1)
#     X = X[mask]
#     y = y[mask]
#     df_rec_clean = df_rec[mask].copy()
    
#     print(f"  After removing NaN: {len(df_rec_clean)} observations")
    
#     # Standardize features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     # Fit logistic regression
#     model = LogisticRegression(random_state=42, max_iter=1000)
#     model.fit(X_scaled, y)
    
#     # Get coefficients
#     coefficients = dict(zip(feature_cols, model.coef_[0]))
    
#     print(f"\n" + "-" * 70)
#     print("  LOGISTIC REGRESSION COEFFICIENTS (Standardized)")
#     print("-" * 70)
#     print(f"\n  {'Feature':<25} {'Coefficient':<15} {'Direction':<20}")
#     print("  " + "-" * 60)
    
#     for feat, coef in sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
#         direction = "â†’ MORE likely target" if coef > 0 else "â†’ LESS likely target"
#         print(f"  {feat:<25} {coef:>+10.3f}      {direction}")
    
#     # Apply QWO scores to data
#     df_rec_clean['qwo'] = model.predict_proba(
#         scaler.transform(df_rec_clean[feature_cols].values)
#     )[:, 1]  # Probability of being targeted
    
#     # Evaluate: What % of time is highest QWO receiver the actual target?
#     qwo_optimal_rate = []
#     vwo_optimal_rate = []
#     comp_prob_optimal_rate = []
#     open_area_optimal_rate = []
    
#     for (game_id, play_id), group in df_rec_clean.groupby(['game_id', 'play_id']):
#         if len(group) < 2:
#             continue
        
#         actual_target = group[group['is_target'] == 1]
#         if len(actual_target) == 0:
#             continue
        
#         actual_id = actual_target.iloc[0]['nfl_id']
        
#         # QWO optimal
#         qwo_optimal_id = group.loc[group['qwo'].idxmax(), 'nfl_id']
#         qwo_optimal_rate.append(qwo_optimal_id == actual_id)
        
#         # VWO optimal
#         vwo_optimal_id = group.loc[group['vwo'].idxmax(), 'nfl_id']
#         vwo_optimal_rate.append(vwo_optimal_id == actual_id)
        
#         # Completion prob optimal
#         comp_optimal_id = group.loc[group['completion_prob'].idxmax(), 'nfl_id']
#         comp_prob_optimal_rate.append(comp_optimal_id == actual_id)
        
#         # Open area optimal
#         open_optimal_id = group.loc[group['open_area'].idxmax(), 'nfl_id']
#         open_area_optimal_rate.append(open_optimal_id == actual_id)
    
#     print(f"\n" + "-" * 70)
#     print("  PREDICTION ACCURACY COMPARISON")
#     print("-" * 70)
    
#     random_rate = (1 / df_rec_clean.groupby(['game_id', 'play_id']).size()).mean() * 100
    
#     print(f"\n  {'Metric':<30} {'% Chose #1':<15} {'vs Random':<15}")
#     print("  " + "-" * 60)
#     print(f"  {'QWO (Fitted Model)':<30} {np.mean(qwo_optimal_rate)*100:>8.1f}%      {np.mean(qwo_optimal_rate)*100 - random_rate:>+6.1f}%")
#     print(f"  {'Completion Probability':<30} {np.mean(comp_prob_optimal_rate)*100:>8.1f}%      {np.mean(comp_prob_optimal_rate)*100 - random_rate:>+6.1f}%")
#     print(f"  {'Open Area':<30} {np.mean(open_area_optimal_rate)*100:>8.1f}%      {np.mean(open_area_optimal_rate)*100 - random_rate:>+6.1f}%")
#     print(f"  {'VWO (Value-Weighted)':<30} {np.mean(vwo_optimal_rate)*100:>8.1f}%      {np.mean(vwo_optimal_rate)*100 - random_rate:>+6.1f}%")
#     print(f"  {'Random Baseline':<30} {random_rate:>8.1f}%")
    
#     return df_rec_clean, coefficients, model, scaler


# # Run the QWO model building
# df_receivers, qwo_coefficients, qwo_model, qwo_scaler = build_qb_weighted_openness(sample_size=5000)


# # Improved QWO with more features and non-linear model

# def build_qwo_v2(sample_size=5000):
#     """
#     Enhanced QWO with:
#     1. Additional features (situational, positional)
#     2. Within-play normalization (rank, percentile)
#     3. Non-linear model (Random Forest)
#     """
#     from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.preprocessing import StandardScaler
    
#     print("=" * 70)
#     print("BUILDING QWO V2 (ENHANCED)")
#     print("=" * 70)
    
#     # Collect receiver-level data with more features
#     plays = df_input.groupby(['game_id', 'play_id']).first().reset_index()
#     if sample_size and sample_size < len(plays):
#         plays = plays.sample(sample_size, random_state=42)
    
#     receiver_data = []
    
#     print(f"\n  Collecting data from {len(plays)} plays...")
    
#     for idx, (_, play) in enumerate(plays.iterrows()):
#         try:
#             result = calculate_play_vwo_v2(
#                 play['game_id'], play['play_id'], 
#                 verbose=False, lambda_risk=LAMBDA_RISK
#             )
            
#             if result is None:
#                 continue
            
#             vwo_df = result['vwo_summary']
#             if len(vwo_df) < 2:
#                 continue
            
#             # Get play context
#             down = result['down']
#             yards_to_go = result['yards_to_go']
#             n_receivers = len(vwo_df)
            
#             # Calculate within-play rankings and percentiles
#             vwo_df['rank_vwo'] = vwo_df['vwo'].rank(ascending=False)
#             vwo_df['rank_comp'] = vwo_df['avg_completion_prob'].rank(ascending=False)
#             vwo_df['rank_open'] = vwo_df['open_area'].rank(ascending=False)
            
#             vwo_df['pct_vwo'] = vwo_df['vwo'].rank(pct=True)
#             vwo_df['pct_comp'] = vwo_df['avg_completion_prob'].rank(pct=True)
            
#             # Max values for normalization
#             max_vwo = vwo_df['vwo'].max()
#             max_comp = vwo_df['avg_completion_prob'].max()
            
#             # Get QB info
#             play_data = df_input[
#                 (df_input['game_id'] == play['game_id']) & 
#                 (df_input['play_id'] == play['play_id'])
#             ]
#             qb_row = play_data[play_data['player_role'] == 'Passer']
#             qb_name = qb_row['player_name'].iloc[0] if len(qb_row) > 0 else 'Unknown'
            
#             for _, rec in vwo_df.iterrows():
#                 receiver_data.append({
#                     'game_id': play['game_id'],
#                     'play_id': play['play_id'],
#                     'qb_name': qb_name,
#                     'receiver_name': rec['player_name'],
#                     'nfl_id': rec['nfl_id'],
#                     'is_target': 1 if rec['player_role'] == 'Targeted Receiver' else 0,
#                     'pass_result': result['pass_result'],
                    
#                     # Situational features
#                     'down': down,
#                     'yards_to_go': yards_to_go,
#                     'n_receivers': n_receivers,
#                     'is_third_fourth': 1 if down >= 3 else 0,
                    
#                     # Core features
#                     'completion_prob': rec['avg_completion_prob'],
#                     'open_area': rec['open_area'],
#                     'window_ratio': rec['window_ratio'],
#                     'danger_area': rec['danger_area'],
#                     'vwo': rec['vwo'],
#                     'avg_epa_complete': rec['avg_epa_complete'],
                    
#                     # Within-play rankings (is this the BEST option?)
#                     'is_rank1_comp': 1 if rec['rank_comp'] == 1 else 0,
#                     'is_rank1_vwo': 1 if rec['rank_vwo'] == 1 else 0,
#                     'is_rank1_open': 1 if rec['rank_open'] == 1 else 0,
                    
#                     # Percentile within play
#                     'pct_comp': rec['pct_comp'],
#                     'pct_vwo': rec['pct_vwo'],
                    
#                     # Relative to best on play
#                     'comp_vs_max': rec['avg_completion_prob'] / max_comp if max_comp > 0 else 0,
#                     'vwo_vs_max': rec['vwo'] / max_vwo if max_vwo > 0 and max_vwo > 0 else 0,
#                 })
            
#         except Exception as e:
#             continue
        
#         if (idx + 1) % 100 == 0:
#             print(f"    Processed {idx + 1}/{len(plays)} plays...")
    
#     df_rec = pd.DataFrame(receiver_data)
#     print(f"\n  Collected {len(df_rec)} receiver observations from {df_rec['play_id'].nunique()} plays")
    
#     # Define feature sets to test
#     feature_sets = {
#         'basic': ['completion_prob', 'open_area', 'window_ratio', 'danger_area', 'vwo'],
#         'with_ranks': ['completion_prob', 'open_area', 'window_ratio', 'danger_area', 'vwo',
#                        'is_rank1_comp', 'is_rank1_open', 'pct_comp'],
#         'with_situation': ['completion_prob', 'open_area', 'window_ratio', 'danger_area', 'vwo',
#                           'is_rank1_comp', 'is_rank1_open', 'pct_comp',
#                           'is_third_fourth', 'n_receivers'],
#         'full': ['completion_prob', 'open_area', 'window_ratio', 'danger_area', 'vwo',
#                  'is_rank1_comp', 'is_rank1_open', 'is_rank1_vwo', 'pct_comp', 'pct_vwo',
#                  'comp_vs_max', 'is_third_fourth', 'n_receivers', 'avg_epa_complete']
#     }
    
#     # Prepare data
#     y = df_rec['is_target'].values
    
#     # Handle NaN
#     df_rec_clean = df_rec.dropna(subset=feature_sets['full'])
#     y_clean = df_rec_clean['is_target'].values
    
#     print(f"  After removing NaN: {len(df_rec_clean)} observations")
    
#     # Test different models and feature sets
#     print(f"\n" + "-" * 70)
#     print("  MODEL COMPARISON")
#     print("-" * 70)
    
#     results = []
    
#     for feat_name, feat_cols in feature_sets.items():
#         X = df_rec_clean[feat_cols].values
        
#         # Standardize
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
        
#         # Test models
#         models = {
#             'Logistic': LogisticRegression(random_state=42, max_iter=1000),
#             'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
#             'GradientBoost': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
#         }
        
#         for model_name, model in models.items():
#             model.fit(X_scaled, y_clean)
            
#             # Get predictions
#             df_rec_clean[f'pred_{feat_name}_{model_name}'] = model.predict_proba(X_scaled)[:, 1]
            
#             # Calculate accuracy
#             optimal_rate = []
#             for (game_id, play_id), group in df_rec_clean.groupby(['game_id', 'play_id']):
#                 if len(group) < 2:
#                     continue
#                 actual = group[group['is_target'] == 1]
#                 if len(actual) == 0:
#                     continue
#                 actual_id = actual.iloc[0]['nfl_id']
#                 pred_col = f'pred_{feat_name}_{model_name}'
#                 pred_best_id = group.loc[group[pred_col].idxmax(), 'nfl_id']
#                 optimal_rate.append(pred_best_id == actual_id)
            
#             accuracy = np.mean(optimal_rate) * 100
#             results.append({
#                 'features': feat_name,
#                 'model': model_name,
#                 'accuracy': accuracy,
#                 'n_features': len(feat_cols)
#             })
    
#     # Display results
#     results_df = pd.DataFrame(results).sort_values('accuracy', ascending=False)
    
#     random_rate = (1 / df_rec_clean.groupby(['game_id', 'play_id']).size()).mean() * 100
    
#     print(f"\n  {'Features':<20} {'Model':<15} {'Accuracy':<12} {'vs Random':<12}")
#     print("  " + "-" * 59)
    
#     for _, row in results_df.iterrows():
#         vs_random = row['accuracy'] - random_rate
#         print(f"  {row['features']:<20} {row['model']:<15} {row['accuracy']:>6.1f}%      {vs_random:>+6.1f}%")
    
#     print(f"\n  Random baseline: {random_rate:.1f}%")
    
#     # Get best model
#     best = results_df.iloc[0]
#     print(f"\n  Best: {best['features']} + {best['model']} = {best['accuracy']:.1f}%")
    
#     # Feature importance for best tree-based model
#     print(f"\n" + "-" * 70)
#     print("  FEATURE IMPORTANCE (Best Random Forest)")
#     print("-" * 70)
    
#     # Retrain best RF with full features
#     feat_cols = feature_sets['full']
#     X_full = df_rec_clean[feat_cols].values
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_full)
    
#     rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
#     rf_model.fit(X_scaled, y_clean)
    
#     importances = dict(zip(feat_cols, rf_model.feature_importances_))
    
#     print(f"\n  {'Feature':<25} {'Importance':<15}")
#     print("  " + "-" * 40)
#     for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
#         bar = "â–ˆ" * int(imp * 50)
#         print(f"  {feat:<25} {imp:>6.3f}  {bar}")
    
#     return df_rec_clean, results_df, rf_model, scaler, feat_cols


# # Run enhanced QWO
# df_qwo_v2, qwo_results, best_model, best_scaler, best_features = build_qwo_v2(sample_size=5000)


# VWO Visualization Functions

def visualize_receiver_vwo(game_id, play_id, figsize=(20, 8)):
    """
    Visualize VWO for all receivers on a play.
    
    Shows:
    - Left: Pitch control (existing)
    - Middle: EPA surface
    - Right: VWO density for each receiver
    """
    print(f"\n{'='*70}")
    print(f"VWO VISUALIZATION: Game {game_id}, Play {play_id}")
    print(f"{'='*70}")
    
    # Get VWO results
    result = calculate_play_vwo_v2(game_id, play_id, verbose=False)
    
    if result is None:
        print("  Could not calculate VWO")
        return None
    
    # Get play info
    los_x = result['los_x']
    down = result['down']
    yards_to_go = result['yards_to_go']
    
    # Get receiver data with surfaces
    receiver_vwos = result['receiver_vwos']
    vwo_summary = result['vwo_summary']
    
    # Find receivers with full surface data
    receivers_with_surfaces = [r for r in receiver_vwos if r['status'] == 'success' and 'grid_x' in r]
    
    if len(receivers_with_surfaces) == 0:
        print("  No receiver surfaces available")
        return None
    
    # Create figure
    n_receivers = min(len(receivers_with_surfaces), 4)  # Max 4 receivers
    fig, axes = plt.subplots(1, n_receivers + 1, figsize=(5 * (n_receivers + 1), 8))
    
    if n_receivers == 0:
        return None
    
    # Sort receivers by VWO
    receivers_with_surfaces = sorted(receivers_with_surfaces, key=lambda x: x['vwo'], reverse=True)
    
    # Get field bounds from first receiver
    first_rec = receivers_with_surfaces[0]
    grid_x = first_rec['grid_x']
    grid_y = first_rec['grid_y']
    extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
    
    # ========================================================================
    # PANEL 0: Combined pitch control / EPA overview
    # ========================================================================
    ax0 = axes[0]
    
    # Get last INPUT frame for pitch control
    last_frame = get_last_input_frame(game_id, play_id)
    flight_time, _ = get_ball_flight_time(game_id, play_id)
    if flight_time is None:
        flight_time = 1.5
    flight_time += (1 / FRAME_RATE)
    
    # Create EPA surface for the field
    field_x = np.arange(los_x - 10, los_x + 40, 1.0)
    field_y = np.arange(0, 53.3, 1.0)
    epa_grid_x, epa_grid_y = np.meshgrid(field_x, field_y)
    
    epa_surface = np.zeros_like(epa_grid_x)
    current_field_pos = los_x - 10
    
    for i in range(epa_grid_x.shape[0]):
        for j in range(epa_grid_x.shape[1]):
            catch_pos = epa_grid_x[i, j] - 10
            epa_surface[i, j] = calculate_epa_complete(catch_pos, current_field_pos, down, yards_to_go)
    
    # Plot EPA surface
    epa_extent = [field_x.min(), field_x.max(), field_y.min(), field_y.max()]
    im0 = ax0.imshow(epa_surface, extent=epa_extent, origin='lower', 
                     cmap='RdYlGn', alpha=0.7, aspect='auto',
                     vmin=-2, vmax=4)
    
    # Add field lines
    ax0.axvline(x=los_x, color='yellow', linewidth=2, linestyle='--', label='LOS')
    first_down_x = los_x + yards_to_go
    ax0.axvline(x=first_down_x, color='orange', linewidth=2, linestyle='-', label='1st Down')
    
    # Plot receiver positions
    for rec in receivers_with_surfaces[:n_receivers]:
        color = 'blue' if rec['player_role'] == 'Targeted Receiver' else 'cyan'
        marker = 'â˜…' if rec['player_role'] == 'Targeted Receiver' else 'o'
        ax0.scatter(rec['receiver_x'], rec['receiver_y'], c=color, s=200, 
                   marker='o', edgecolors='white', linewidths=2, zorder=10)
        ax0.annotate(f"{rec['vwo']:.2f}", (rec['receiver_x'], rec['receiver_y'] + 2),
                    ha='center', fontsize=9, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax0.set_title(f"EPA Surface\n{down}&{yards_to_go} at {los_x}", fontsize=11, fontweight='bold')
    ax0.set_xlabel('Field Position (yards)')
    ax0.set_ylabel('Field Width (yards)')
    ax0.set_xlim(epa_extent[0], epa_extent[1])
    ax0.set_ylim(0, 53.3)
    
    plt.colorbar(im0, ax=ax0, label='EPA if Complete', shrink=0.6)
    
    # ========================================================================
    # PANELS 1-N: Individual receiver VWO surfaces
    # ========================================================================
    
    for i, rec in enumerate(receivers_with_surfaces[:n_receivers]):
        ax = axes[i + 1]
        
        rec_grid_x = rec['grid_x']
        rec_grid_y = rec['grid_y']
        vwo_density = rec['vwo_density']
        reach_prob = rec['reach_prob']
        control_ratio = rec['control_ratio']
        expected_value = rec['expected_value']
        
        rec_extent = [rec_grid_x.min(), rec_grid_x.max(), rec_grid_y.min(), rec_grid_y.max()]
        
        # Option 1: Show VWO density (reach Ã— completion Ã— EPA)
        # Normalize for visualization
        vwo_norm = vwo_density / (np.abs(vwo_density).max() + 1e-6)
        
        # Create custom colormap: red (negative) -> white (zero) -> green (positive)
        im = ax.imshow(expected_value, extent=rec_extent, origin='lower',
                       cmap='RdYlGn', alpha=0.8, aspect='auto',
                       vmin=-2, vmax=3)
        
        # Overlay reach probability as contours
        if reach_prob.max() > 0:
            reach_levels = [0.01, 0.05, 0.1]
            ax.contour(rec_grid_x, rec_grid_y, reach_prob, levels=reach_levels,
                      colors='blue', linewidths=1.5, linestyles='-', alpha=0.7)
        
        # Overlay control ratio contours
        ax.contour(rec_grid_x, rec_grid_y, control_ratio, levels=[0.5, 0.7],
                  colors=['gray', 'white'], linewidths=[1, 2], linestyles=['--', '-'], alpha=0.8)
        
        # Mark receiver position
        ax.scatter(rec['receiver_x'], rec['receiver_y'], c='blue', s=250,
                  marker='o', edgecolors='white', linewidths=3, zorder=10)
        
        # Mark sweet spot
        ax.scatter(rec['sweet_spot_x'], rec['sweet_spot_y'], c='gold', s=150,
                  marker='â˜…', edgecolors='black', linewidths=1.5, zorder=11)
        
        # Title with metrics
        role = "â†’ TARGET" if rec['player_role'] == 'Targeted Receiver' else ""
        title = f"{rec.get('player_name', 'Receiver')[:15]} {role}\n"
        title += f"VWO: {rec['vwo']:.3f} | Comp: {rec['avg_completion_prob']*100:.0f}%\n"
        title += f"Window: {rec['window_ratio']:.2f} | Danger: {rec['danger_area']:.2f}"
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Field X')
        ax.set_ylabel('Field Y')
        
        plt.colorbar(im, ax=ax, label='E[Value]', shrink=0.6)
    
    plt.suptitle(f"VWO Analysis: Game {game_id}, Play {play_id}\n"
                 f"Optimal: {result['optimal_target_name']} (VWO={result['optimal_vwo']:.3f}) | "
                 f"Actual: {result['actual_target_name']} (VWO={result['actual_vwo']:.3f})",
                 fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    return fig, result


def visualize_vwo_comparison(game_id, play_id, figsize=(16, 10)):
    """
    Side-by-side comparison of all receivers' VWO components.
    
    Shows bar charts comparing:
    - VWO (total value)
    - Completion probability
    - Open area / Window ratio
    - Danger area
    - EPA if complete
    """
    result = calculate_play_vwo_v2(game_id, play_id, verbose=False)
    
    if result is None:
        print("Could not calculate VWO")
        return None
    
    vwo_df = result['vwo_summary'].copy()
    vwo_df = vwo_df.sort_values('vwo', ascending=True)  # For horizontal bar chart
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Colors: highlight actual target
    colors = ['green' if role == 'Targeted Receiver' else 'steelblue' 
              for role in vwo_df['player_role']]
    
    # Truncate names
    names = [name[:12] for name in vwo_df['player_name']]
    
    # Panel 1: VWO
    ax = axes[0, 0]
    bars = ax.barh(names, vwo_df['vwo'], color=colors)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('VWO (Expected Points)')
    ax.set_title('Value-Weighted Openness', fontweight='bold')
    
    # Panel 2: Completion Probability
    ax = axes[0, 1]
    ax.barh(names, vwo_df['avg_completion_prob'] * 100, color=colors)
    ax.set_xlabel('Completion Probability (%)')
    ax.set_title('Completion Probability', fontweight='bold')
    ax.set_xlim(0, 100)
    
    # Panel 3: Window Ratio
    ax = axes[0, 2]
    ax.barh(names, vwo_df['window_ratio'], color=colors)
    ax.set_xlabel('Window Ratio')
    ax.set_title('Window Ratio (Safety)', fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Panel 4: Open Area
    ax = axes[1, 0]
    ax.barh(names, vwo_df['open_area'], color=colors)
    ax.set_xlabel('Open Area (proportion)')
    ax.set_title('Open Area', fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Panel 5: Danger Area
    ax = axes[1, 1]
    ax.barh(names, vwo_df['danger_area'], color='indianred')
    ax.set_xlabel('Danger Area (proportion)')
    ax.set_title('Danger Area (INT Risk)', fontweight='bold')
    ax.set_xlim(0, 0.5)
    
    # Panel 6: EPA if Complete
    ax = axes[1, 2]
    ax.barh(names, vwo_df['avg_epa_complete'], color=colors)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('EPA if Complete')
    ax.set_title('Expected Value (if caught)', fontweight='bold')
    
    plt.suptitle(f"VWO Component Breakdown: Game {game_id}, Play {play_id}\n"
                 f"{result['down']}&{result['yards_to_go']} | "
                 f"Optimal: {result['optimal_target_name']} | "
                 f"Actual: {result['actual_target_name']} (green)",
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    return fig, result


def visualize_vwo_surface_single(game_id, play_id, receiver_idx=0, figsize=(14, 10)):
    """
    Detailed VWO surface visualization for a single receiver.
    
    Shows 4 panels:
    - Reach probability (movement cloud)
    - Control ratio (pitch control)
    - Expected value surface
    - VWO density (product)
    """
    result = calculate_play_vwo_v2(game_id, play_id, verbose=False)
    
    if result is None:
        return None
    
    # Get receiver
    receivers = [r for r in result['receiver_vwos'] if r['status'] == 'success' and 'grid_x' in r]
    receivers = sorted(receivers, key=lambda x: x['vwo'], reverse=True)
    
    if receiver_idx >= len(receivers):
        print(f"Only {len(receivers)} receivers available")
        return None
    
    rec = receivers[receiver_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    grid_x = rec['grid_x']
    grid_y = rec['grid_y']
    extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
    
    # Panel 1: Reach Probability
    ax = axes[0, 0]
    im = ax.imshow(rec['reach_prob'], extent=extent, origin='lower',
                   cmap='Blues', aspect='auto')
    ax.scatter(rec['receiver_x'], rec['receiver_y'], c='blue', s=200,
              marker='o', edgecolors='white', linewidths=2, zorder=10)
    ax.set_title('P(Reach) - Movement Cloud', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Probability', shrink=0.8)
    
    # Panel 2: Control Ratio
    ax = axes[0, 1]
    im = ax.imshow(rec['control_ratio'], extent=extent, origin='lower',
                   cmap='RdBu', vmin=0, vmax=1, aspect='auto')
    ax.contour(grid_x, grid_y, rec['control_ratio'], levels=[0.5],
              colors='black', linewidths=2)
    ax.contour(grid_x, grid_y, rec['control_ratio'], levels=[OPEN_THRESHOLD],
              colors='green', linewidths=2, linestyles='--')
    ax.contour(grid_x, grid_y, rec['control_ratio'], levels=[DANGER_THRESHOLD],
              colors='red', linewidths=2, linestyles='--')
    ax.scatter(rec['receiver_x'], rec['receiver_y'], c='yellow', s=200,
              marker='o', edgecolors='black', linewidths=2, zorder=10)
    ax.set_title(f'Control Ratio (Open>{OPEN_THRESHOLD}, Danger<{DANGER_THRESHOLD:.2f})', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Offense Control', shrink=0.8)
    
    # Panel 3: Expected Value Surface
    ax = axes[1, 0]
    im = ax.imshow(rec['expected_value'], extent=extent, origin='lower',
                   cmap='RdYlGn', vmin=-3, vmax=4, aspect='auto')
    ax.scatter(rec['receiver_x'], rec['receiver_y'], c='blue', s=200,
              marker='o', edgecolors='white', linewidths=2, zorder=10)
    ax.scatter(rec['sweet_spot_x'], rec['sweet_spot_y'], c='gold', s=150,
              marker='â˜…', edgecolors='black', linewidths=1.5, zorder=11)
    ax.set_title('E[Value] = P(C)Ã—EPA_C + P(I)Ã—EPA_I + P(INT)Ã—EPA_INT', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Expected Points', shrink=0.8)
    
    # Panel 4: VWO Density
    ax = axes[1, 1]
    vwo_density = rec['vwo_density']
    vmax = max(abs(vwo_density.min()), abs(vwo_density.max()))
    im = ax.imshow(vwo_density, extent=extent, origin='lower',
                   cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.scatter(rec['receiver_x'], rec['receiver_y'], c='blue', s=200,
              marker='o', edgecolors='white', linewidths=2, zorder=10)
    ax.scatter(rec['sweet_spot_x'], rec['sweet_spot_y'], c='gold', s=150,
              marker='â˜…', edgecolors='black', linewidths=1.5, zorder=11)
    ax.set_title('VWO Density = P(Reach) Ã— E[Value]', fontweight='bold')
    plt.colorbar(im, ax=ax, label='VWO Contribution', shrink=0.8)
    
    # Add labels to all
    for ax in axes.flat:
        ax.set_xlabel('Field X (yards)')
        ax.set_ylabel('Field Y (yards)')
    
    role = "TARGET" if rec['player_role'] == 'Targeted Receiver' else ""
    plt.suptitle(f"{rec.get('player_name', 'Receiver')} {role}\n"
                 f"VWO: {rec['vwo']:.3f} | Comp: {rec['avg_completion_prob']*100:.0f}% | "
                 f"Window: {rec['window_ratio']:.2f} | Open: {rec['open_area']:.2f} | "
                 f"Danger: {rec['danger_area']:.2f}",
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    return fig, rec


# Quick test function
def test_vwo_visualization(outcome='C'):
    """Quick test of VWO visualization with a random play"""
    game_id, play_id = random_play(outcome=outcome)
    if game_id:
        print(f"Visualizing: Game {game_id}, Play {play_id}")
        fig1, _ = visualize_receiver_vwo(game_id, play_id)
        plt.show()
        fig2, _ = visualize_vwo_comparison(game_id, play_id)
        plt.show()
        fig3, _ = visualize_vwo_surface_single(game_id, play_id, receiver_idx=0)
        plt.show()
        return game_id, play_id
    return None, None


print("VWO Visualization functions ready!")
print("\nFunctions:")
print("  â€¢ visualize_receiver_vwo(game_id, play_id) - Multi-receiver overview")
print("  â€¢ visualize_vwo_comparison(game_id, play_id) - Bar chart comparison")
print("  â€¢ visualize_vwo_surface_single(game_id, play_id, receiver_idx) - Detailed single receiver")
print("  â€¢ test_vwo_visualization(outcome='C') - Quick test with random play")


visualize_vwo_comparison(2023110510, 3666)


# VWO Field Visualization - Fixed Color Scaling

def visualize_vwo_field(game_id, play_id, figsize=(16, 10), 
                        show_defenders=True, show_vwo_labels=True,
                        show_sweet_spots=True, show_value_contours=False,  # Off by default now
                        vwo_alpha=0.7, velocity_scale=0.5):
    """
    Visualize VWO for all receivers on the field.
    
    Color scheme:
    - Offense: Blue circles
    - Defense: Red squares
    - QB: Yellow square
    - VWO shading: Blue (positive) â†’ White (zero) â†’ Red (negative)
    """
    print(f"\n{'='*70}")
    print(f"VWO FIELD VISUALIZATION: Game {game_id}, Play {play_id}")
    print(f"{'='*70}")
    
    # Get play data
    play_input = df_input[
        (df_input['game_id'] == game_id) & 
        (df_input['play_id'] == play_id)
    ]
    
    if len(play_input) == 0:
        print("  No data found")
        return None, None
    
    # Get play context
    los_x = play_input['absolute_yardline_number'].iloc[0]
    down = play_input['down'].iloc[0]
    yards_to_go = play_input['yards_to_go'].iloc[0]
    pass_result = play_input['pass_result'].iloc[0]
    
    # Get last INPUT frame
    last_frame = get_last_input_frame(game_id, play_id)
    if last_frame is None:
        print("  Could not get last input frame")
        return None, None
    
    # Get flight time + 1 frame adjustment
    base_flight_time, _ = get_ball_flight_time(game_id, play_id)
    if base_flight_time is None:
        base_flight_time = 1.5
    flight_time = base_flight_time + (1 / FRAME_RATE)
    
    print(f"  Situation: {down}&{yards_to_go} at {los_x}")
    print(f"  Flight time: {flight_time:.2f}s")
    print(f"  Result: {pass_result}")
    
    # Identify players
    offense_roles = ['Passer', 'Targeted Receiver', 'Other Route Runner']
    offense_players = last_frame[last_frame['player_role'].isin(offense_roles)]
    
    if 'coverage_player' in last_frame.columns:
        defense_players = last_frame[last_frame['coverage_player'] == True]
    else:
        defense_players = last_frame[last_frame['player_side'] == 'Defense']
    
    receivers = last_frame[
        last_frame['player_role'].isin(['Targeted Receiver', 'Other Route Runner'])
    ]
    
    qb_data = last_frame[last_frame['player_role'] == 'Passer']
    
    print(f"  Receivers: {len(receivers)}")
    print(f"  Defenders: {len(defense_players)}")
    
    # Calculate VWO for each receiver
    receiver_vwo_data = []
    
    for _, rec in receivers.iterrows():
        vwo_result = calculate_receiver_vwo_v2(
            rec, last_frame, los_x, down, yards_to_go, flight_time,
            grid_spacing=1.0, grid_extent=12, lambda_risk=LAMBDA_RISK
        )
        
        if vwo_result['status'] == 'success':
            vwo_result['nfl_id'] = rec['nfl_id']
            vwo_result['player_name'] = rec.get('player_name', 'Unknown')
            vwo_result['player_role'] = rec['player_role']
            vwo_result['speed'] = rec['s'] if pd.notna(rec['s']) else 0
            vwo_result['direction'] = rec['dir_norm'] if pd.notna(rec['dir_norm']) else 0
            receiver_vwo_data.append(vwo_result)
    
    if len(receiver_vwo_data) == 0:
        print("  No receiver VWO data")
        return None, None
    
    # Sort by VWO
    receiver_vwo_data = sorted(receiver_vwo_data, key=lambda x: x['vwo'], reverse=True)
    
    # DIAGNOSTIC: Print VWO ranges
    print(f"\n  VWO Diagnostics:")
    for rec_data in receiver_vwo_data:
        vwo_density = rec_data.get('vwo_density')
        if vwo_density is not None:
            print(f"    {rec_data['player_name'][:15]}: VWO={rec_data['vwo']:.3f}, "
                  f"density range=[{vwo_density.min():.3f}, {vwo_density.max():.3f}]")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    draw_football_field(ax, show_hash_marks=True, show_numbers=True)
    
    # Determine field bounds
    all_x = [r['receiver_x'] for r in receiver_vwo_data]
    all_y = [r['receiver_y'] for r in receiver_vwo_data]
    
    if len(defense_players) > 0:
        all_x.extend(defense_players['x_norm'].tolist())
        all_y.extend(defense_players['y_norm'].tolist())
    
    if len(qb_data) > 0:
        all_x.append(qb_data.iloc[0]['x_norm'])
        all_y.append(qb_data.iloc[0]['y_norm'])
    
    x_min = min(all_x) - 15
    x_max = max(all_x) + 15
    y_min = max(0, min(all_y) - 10)
    y_max = min(53.3, max(all_y) + 10)
    
    # ========================================================================
    # PLOT DEFENDER MOVEMENT CLOUDS
    # ========================================================================
    
    if show_defenders:
        for _, defender in defense_players.iterrows():
            def_x = defender['x_norm']
            def_y = defender['y_norm']
            def_speed = defender['s'] if pd.notna(defender['s']) else 3.0
            def_dir = defender['dir_norm'] if pd.notna(defender['dir_norm']) else 0
            
            dist, vel_bin, effective_time, _ = get_player_distribution_interpolated(
                def_speed, flight_time
            )
            
            if dist is not None:
                def_grid_x_coords = np.arange(def_x - 12, def_x + 12, 1.0)
                def_grid_y_coords = np.arange(max(0, def_y - 12), min(53.3, def_y + 12), 1.0)
                def_grid_x, def_grid_y = np.meshgrid(def_grid_x_coords, def_grid_y_coords)
                
                def_density = transform_distribution_to_field(
                    dist, def_x, def_y, def_dir, def_grid_x, def_grid_y
                )
                
                def_density_expanded = expand_density_with_catch_radius(
                    def_density, 1.0, CATCH_RADIUS
                )
                
                if def_density_expanded.max() > 0:
                    def_density_norm = def_density_expanded / def_density_expanded.max()
                else:
                    continue
                
                def_extent = [def_grid_x.min(), def_grid_x.max(), 
                             def_grid_y.min(), def_grid_y.max()]
                
                # Red shading for defenders
                #red_rgba = np.zeros((*def_density_norm.shape, 4))
                #red_rgba[:, :, 0] = 0.9
                #red_rgba[:, :, 1] = 0.2
                #red_rgba[:, :, 2] = 0.2
                #red_rgba[:, :, 3] = def_density_norm * 0.5
                
                #ax.imshow(red_rgba, extent=def_extent, origin='lower', 
                         #aspect='auto', zorder=2)
                
                p95 = compute_p95_threshold(def_density_expanded)
                if p95 > 0:
                    try:
                        ax.contour(def_grid_x, def_grid_y, def_density_expanded,
                                  levels=[p95], colors=['darkred'], linewidths=1.5,
                                  linestyles='--', alpha=0.8, zorder=3)
                    except:
                        pass
            
            # Defender marker (RED SQUARE)
            ax.scatter(def_x, def_y, c='red', s=200, marker='s',
                      edgecolors='white', linewidths=2, zorder=10)
            
            # Velocity arrow
            if def_speed > 0.5:
                dx = def_speed * velocity_scale * np.cos(np.radians(def_dir))
                dy = def_speed * velocity_scale * np.sin(np.radians(def_dir))
                ax.arrow(def_x, def_y, dx, dy,
                        head_width=0.6, head_length=0.3,
                        fc='red', ec='darkred', linewidth=1.5,
                        zorder=11, alpha=0.9)
    
    # ========================================================================
    # PLOT RECEIVER VWO CLOUDS - FIXED COLOR SCALING
    # ========================================================================
    
    # Use a FIXED scale for VWO coloring so colors are consistent
    # Typical VWO range is roughly -2 to +2
    VWO_SCALE = 1.0  # Values beyond +/- this will saturate to full color
    
    for rank, rec_data in enumerate(receiver_vwo_data):
        rec_x = rec_data['receiver_x']
        rec_y = rec_data['receiver_y']
        rec_speed = rec_data['speed']
        rec_dir = rec_data['direction']
        
        if 'grid_x' not in rec_data:
            continue
        
        grid_x = rec_data['grid_x']
        grid_y = rec_data['grid_y']
        vwo_density = rec_data['vwo_density']
        reach_prob = rec_data['reach_prob']
        
        extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
        
        # FIXED SCALING: Normalize VWO density to fixed range
        # This ensures blue appears for positive values
        vwo_normalized = vwo_density / VWO_SCALE
        vwo_normalized = np.clip(vwo_normalized, -1, 1)
        
        # Create RGBA array
        rgba = np.zeros((*vwo_density.shape, 4))
        
        for i in range(vwo_density.shape[0]):
            for j in range(vwo_density.shape[1]):
                vwo_val = vwo_normalized[i, j]
                reach_val = reach_prob[i, j]
                
                # Skip very low reach probability areas
                if reach_val < 0.0005:
                    continue
                
                # STRONGER color mapping
                if vwo_val > 0:
                    # BLUE for positive VWO - make it more saturated
                    intensity = min(1.0, vwo_val * 2)  # Amplify to make blue more visible
                    rgba[i, j, 0] = 0.3 * (1 - intensity)  # R low
                    rgba[i, j, 1] = 0.5 * (1 - intensity) + 0.3  # G medium-low
                    rgba[i, j, 2] = 0.7 + 0.3 * intensity  # B high
                else:
                    # RED for negative VWO
                    intensity = min(1.0, abs(vwo_val) * 2)
                    rgba[i, j, 0] = 0.7 + 0.3 * intensity  # R high
                    rgba[i, j, 1] = 0.3 * (1 - intensity)  # G low
                    rgba[i, j, 2] = 0.3 * (1 - intensity)  # B low
                
                # Alpha based on reach probability - make it more visible
                base_alpha = 0.3  # Minimum alpha for visible areas
                reach_alpha = min(0.8, reach_val * 10)  # Scale up reach contribution
                rgba[i, j, 3] = min(vwo_alpha, base_alpha + reach_alpha * 0.5)
        
        ax.imshow(rgba, extent=extent, origin='lower', aspect='auto', zorder=4)
        
        # P95 contour for reach
        reach_expanded = expand_density_with_catch_radius(reach_prob, 1.0, CATCH_RADIUS)
        p95_reach = compute_p95_threshold(reach_expanded)
        
        if p95_reach > 0:
            try:
                contour_color = 'darkblue' if rec_data['player_role'] == 'Targeted Receiver' else 'blue'
                contour_width = 3.0 if rec_data['player_role'] == 'Targeted Receiver' else 2.0
                ax.contour(grid_x, grid_y, reach_expanded,
                          levels=[p95_reach], colors=[contour_color], linewidths=contour_width,
                          linestyles='-', alpha=0.9, zorder=5)
            except:
                pass
        
        # Value contour (VWO = 0) - only if enabled
        if show_value_contours:
            try:
                ax.contour(grid_x, grid_y, vwo_density,
                          levels=[0], colors=['white'], linewidths=1.0,
                          linestyles='-', alpha=0.5, zorder=5)
            except:
                pass
        
        # Sweet spot marker
        if show_sweet_spots:
            ax.scatter(rec_data['sweet_spot_x'], rec_data['sweet_spot_y'],
                      c='gold', s=150, marker='*', edgecolors='black',
                      linewidths=1.5, zorder=11)
        
        # Receiver position (BLUE CIRCLE)
        marker_color = 'darkblue' if rec_data['player_role'] == 'Targeted Receiver' else 'blue'
        marker_size = 250 if rec_data['player_role'] == 'Targeted Receiver' else 200
        ax.scatter(rec_x, rec_y, c=marker_color, s=marker_size, marker='o',
                  edgecolors='white', linewidths=2.5, zorder=12)
        
        # Velocity arrow
        if rec_speed > 0.5:
            dx = rec_speed * velocity_scale * np.cos(np.radians(rec_dir))
            dy = rec_speed * velocity_scale * np.sin(np.radians(rec_dir))
            ax.arrow(rec_x, rec_y, dx, dy,
                    head_width=0.6, head_length=0.3,
                    fc='blue', ec='darkblue', linewidth=1.5,
                    zorder=13, alpha=0.9)
        
        # VWO label
        if show_vwo_labels:
            vwo_val = rec_data['vwo']
            comp_pct = rec_data['avg_completion_prob'] * 100
            
            if vwo_val > 0.3:
                label_color = 'darkblue'
            elif vwo_val > 0:
                label_color = 'blue'
            elif vwo_val > -0.3:
                label_color = 'orange'
            else:
                label_color = 'red'
            
            label_text = f"VWO: {vwo_val:.2f}\n{comp_pct:.0f}%"
            
            if rank == 0:
                label_text = f"#1 OPTIMAL\n" + label_text
            elif rec_data['player_role'] == 'Targeted Receiver':
                label_text = f"-> TARGET\n" + label_text
            
            ax.annotate(label_text, (rec_x, rec_y + 3),
                       ha='center', va='bottom', fontsize=9, fontweight='bold',
                       color=label_color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                alpha=0.9, edgecolor=label_color))
    
    # ========================================================================
    # PLOT QB (YELLOW)
    # ========================================================================
    
    if len(qb_data) > 0:
        qb = qb_data.iloc[0]
        qb_x = qb['x_norm']
        qb_y = qb['y_norm']
        qb_speed = qb['s'] if pd.notna(qb['s']) else 0
        qb_dir = qb['dir_norm'] if pd.notna(qb['dir_norm']) else 0
        
        ax.scatter(qb_x, qb_y, c='yellow', s=300, marker='s',
                  edgecolors='black', linewidths=3, zorder=14)
        ax.annotate('QB', (qb_x, qb_y - 2.5),
                   ha='center', fontsize=10, fontweight='bold', color='black')
        
        if qb_speed > 0.5:
            dx = qb_speed * velocity_scale * np.cos(np.radians(qb_dir))
            dy = qb_speed * velocity_scale * np.sin(np.radians(qb_dir))
            ax.arrow(qb_x, qb_y, dx, dy,
                    head_width=0.6, head_length=0.3,
                    fc='yellow', ec='black', linewidth=1.5,
                    zorder=15, alpha=0.9)
    
    # ========================================================================
    # FIELD MARKERS
    # ========================================================================
    
    ax.axvline(x=los_x, color='blue', linewidth=3, linestyle='-', alpha=0.8, zorder=6)
    ax.annotate('LOS', (los_x, y_max - 2), ha='center', fontsize=10, 
               fontweight='bold', color='blue')
    
    first_down_x = los_x + yards_to_go
    if first_down_x < 110:
        ax.axvline(x=first_down_x, color='yellow', linewidth=3, linestyle='-', 
                  alpha=0.8, zorder=6)
        ax.annotate('1st', (first_down_x, y_max - 2), ha='center', fontsize=10,
                   fontweight='bold', color='gold')
    
    # ========================================================================
    # LEGEND AND TITLE
    # ========================================================================
    
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor='royalblue', alpha=0.7, label='Positive VWO (good target)'),
        Patch(facecolor='indianred', alpha=0.7, label='Negative VWO (poor target)'),
        Patch(facecolor='red', alpha=0.5, label='Defender coverage zone'),
        Line2D([0], [0], color='darkblue', linewidth=2.5, label='Receiver reach (P95)'),
        Line2D([0], [0], color='darkred', linewidth=1.5, linestyle='--', label='Defender reach (P95)'),
        Line2D([0], [0], marker='*', color='gold', markersize=12, linestyle='None', 
               markeredgecolor='black', label='Sweet spot (max VWO)'),
        Line2D([0], [0], marker='o', color='blue', markersize=10, linestyle='None',
               markeredgecolor='white', label='Receiver'),
        Line2D([0], [0], marker='s', color='red', markersize=10, linestyle='None',
               markeredgecolor='white', label='Defender'),
        Line2D([0], [0], marker='s', color='yellow', markersize=10, linestyle='None',
               markeredgecolor='black', label='Quarterback'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.95)
    
    optimal = receiver_vwo_data[0]
    actual = next((r for r in receiver_vwo_data if r['player_role'] == 'Targeted Receiver'), None)
    
    title = f"Value-Weighted Openness (VWO) - Game {game_id}, Play {play_id}\n"
    title += f"{down}&{yards_to_go} | Flight: {flight_time:.2f}s | Result: {pass_result}\n"
    title += f"Optimal: {optimal['player_name'][:15]} (VWO={optimal['vwo']:.2f})"
    if actual:
        title += f" | Actual: {actual['player_name'][:15]} (VWO={actual['vwo']:.2f})"
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Field Position (yards)', fontsize=11)
    ax.set_ylabel('Field Width (yards)', fontsize=11)
    ax.set_aspect('equal')
    
    fig.patch.set_facecolor('#f5f5f5')
    plt.tight_layout()
    
    print(f"\n  Visualization complete!")
    print(f"  Optimal: {optimal['player_name']} (VWO={optimal['vwo']:.3f})")
    if actual:
        print(f"  Actual: {actual['player_name']} (VWO={actual['vwo']:.3f})")
    
    return fig, {'receivers': receiver_vwo_data, 'optimal': optimal, 'actual': actual}


print("Fixed VWO Visualization ready!")
print("""
Key changes:
1. Fixed VWO scale (VWO_SCALE=1.0) - ensures blue appears for positive values
2. Stronger color saturation - blue/red more visible
3. Higher base alpha - colors show up better
4. Removed black dashed lines by default (show_value_contours=False)
5. Added VWO density diagnostics in output

Test with:
  fig, data = visualize_vwo_field(2023110510, 3666)
  plt.show()
""")


fig, data = visualize_vwo_field(2023110510, 3666)
plt.show()


# # Diagnostic: What are the actual VWO density values?

# def diagnose_vwo_ranges(game_id, play_id):
#     """Check the actual value ranges in VWO calculation"""
    
#     play_input = df_input[
#         (df_input['game_id'] == game_id) & 
#         (df_input['play_id'] == play_id)
#     ]
    
#     los_x = play_input['absolute_yardline_number'].iloc[0]
#     down = play_input['down'].iloc[0]
#     yards_to_go = play_input['yards_to_go'].iloc[0]
    
#     last_frame = get_last_input_frame(game_id, play_id)
#     base_flight_time, _ = get_ball_flight_time(game_id, play_id)
#     flight_time = (base_flight_time or 1.5) + (1 / FRAME_RATE)
    
#     receivers = last_frame[
#         last_frame['player_role'].isin(['Targeted Receiver', 'Other Route Runner'])
#     ]
    
#     print("=" * 70)
#     print(f"VWO DIAGNOSTIC: Game {game_id}, Play {play_id}")
#     print(f"Situation: {down}&{yards_to_go}")
#     print("=" * 70)
    
#     for _, rec in receivers.iterrows():
#         result = calculate_receiver_vwo_v2(
#             rec, last_frame, los_x, down, yards_to_go, flight_time,
#             grid_spacing=1.0, grid_extent=12, lambda_risk=LAMBDA_RISK
#         )
        
#         if result['status'] != 'success' or 'grid_x' not in result:
#             continue
        
#         reach_prob = result['reach_prob']
#         expected_value = result['expected_value']
#         vwo_density = result['vwo_density']
#         control_ratio = result['control_ratio']
        
#         print(f"\n{rec.get('player_name', 'Unknown')[:20]}:")
#         print(f"  VWO (integrated): {result['vwo']:.4f}")
#         print(f"  ")
#         print(f"  reach_prob:      min={reach_prob.min():.6f}, max={reach_prob.max():.6f}, sum={reach_prob.sum():.4f}")
#         print(f"  expected_value:  min={expected_value.min():.3f}, max={expected_value.max():.3f}")
#         print(f"  vwo_density:     min={vwo_density.min():.6f}, max={vwo_density.max():.6f}")
#         print(f"  control_ratio:   min={control_ratio.min():.3f}, max={control_ratio.max():.3f}")
#         print(f"  ")
#         print(f"  Completion prob: {result['avg_completion_prob']*100:.1f}%")
#         print(f"  Avg EPA complete: {result['avg_epa_complete']:.3f}")

# # Run diagnostic
# diagnose_vwo_ranges(2023110510, 3666)


# VWO Field Visualization - Shade by Expected Value

def visualize_vwo_field(game_id, play_id, figsize=(16, 10), 
                        show_defenders=True, show_vwo_labels=True,
                        show_sweet_spots=True, 
                        vwo_alpha=0.85, velocity_scale=0.5,
                        epa_range=(-2.0, 2.0)):
    """
    Visualize VWO for all receivers on the field.
    
    NEW APPROACH:
    - Color = Expected Value at each point (Blue=positive EPA, Red=negative)
    - Opacity = Reach probability (darker where receiver more likely to be)
    
    This shows: "If we throw HERE, it's worth X points, and receiver has Y% chance of getting there"
    """
    print(f"\n{'='*70}")
    print(f"VWO FIELD VISUALIZATION: Game {game_id}, Play {play_id}")
    print(f"{'='*70}")
    
    # Get play data
    play_input = df_input[
        (df_input['game_id'] == game_id) & 
        (df_input['play_id'] == play_id)
    ]
    
    if len(play_input) == 0:
        print("  No data found")
        return None, None
    
    # Get play context
    los_x = play_input['absolute_yardline_number'].iloc[0]
    down = play_input['down'].iloc[0]
    yards_to_go = play_input['yards_to_go'].iloc[0]
    pass_result = play_input['pass_result'].iloc[0]
    
    # Get last INPUT frame
    last_frame = get_last_input_frame(game_id, play_id)
    if last_frame is None:
        print("  Could not get last input frame")
        return None, None
    
    # Get flight time + 1 frame adjustment
    base_flight_time, _ = get_ball_flight_time(game_id, play_id)
    if base_flight_time is None:
        base_flight_time = 1.5
    flight_time = base_flight_time + (1 / FRAME_RATE)
    
    print(f"  Situation: {down}&{yards_to_go} at {los_x}")
    print(f"  Flight time: {flight_time:.2f}s")
    print(f"  Result: {pass_result}")
    
    # Identify players
    if 'coverage_player' in last_frame.columns:
        defense_players = last_frame[last_frame['coverage_player'] == True]
    else:
        defense_players = last_frame[last_frame['player_side'] == 'Defense']
    
    receivers = last_frame[
        last_frame['player_role'].isin(['Targeted Receiver', 'Other Route Runner'])
    ]
    
    qb_data = last_frame[last_frame['player_role'] == 'Passer']
    
    print(f"  Receivers: {len(receivers)}")
    print(f"  Defenders: {len(defense_players)}")
    
    # Calculate VWO for each receiver
    receiver_vwo_data = []
    
    for _, rec in receivers.iterrows():
        vwo_result = calculate_receiver_vwo_v2(
            rec, last_frame, los_x, down, yards_to_go, flight_time,
            grid_spacing=1.0, grid_extent=12, lambda_risk=LAMBDA_RISK
        )
        
        if vwo_result['status'] == 'success':
            vwo_result['nfl_id'] = rec['nfl_id']
            vwo_result['player_name'] = rec.get('player_name', 'Unknown')
            vwo_result['player_role'] = rec['player_role']
            vwo_result['speed'] = rec['s'] if pd.notna(rec['s']) else 0
            vwo_result['direction'] = rec['dir_norm'] if pd.notna(rec['dir_norm']) else 0
            receiver_vwo_data.append(vwo_result)
    
    if len(receiver_vwo_data) == 0:
        print("  No receiver VWO data")
        return None, None
    
    # Sort by VWO
    receiver_vwo_data = sorted(receiver_vwo_data, key=lambda x: x['vwo'], reverse=True)
    
    # Print summary
    print(f"\n  Receiver VWO Summary:")
    for i, rec_data in enumerate(receiver_vwo_data):
        role_marker = " <- TARGET" if rec_data['player_role'] == 'Targeted Receiver' else ""
        opt_marker = " [OPTIMAL]" if i == 0 else ""
        print(f"    {rec_data['player_name'][:15]:<15}: VWO={rec_data['vwo']:>6.3f}, "
              f"Comp={rec_data['avg_completion_prob']*100:>4.0f}%, "
              f"EPA={rec_data['avg_epa_complete']:>6.3f}{opt_marker}{role_marker}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    draw_football_field(ax, show_hash_marks=True, show_numbers=True)
    
    # Determine field bounds
    all_x = [r['receiver_x'] for r in receiver_vwo_data]
    all_y = [r['receiver_y'] for r in receiver_vwo_data]
    
    if len(defense_players) > 0:
        all_x.extend(defense_players['x_norm'].tolist())
        all_y.extend(defense_players['y_norm'].tolist())
    
    if len(qb_data) > 0:
        all_x.append(qb_data.iloc[0]['x_norm'])
        all_y.append(qb_data.iloc[0]['y_norm'])
    
    x_min = min(all_x) - 15
    x_max = max(all_x) + 15
    y_min = max(0, min(all_y) - 10)
    y_max = min(53.3, max(all_y) + 10)
    
    # ========================================================================
    # PLOT DEFENDER MOVEMENT CLOUDS (Red shading)
    # ========================================================================
    
    if show_defenders:
        for _, defender in defense_players.iterrows():
            def_x = defender['x_norm']
            def_y = defender['y_norm']
            def_speed = defender['s'] if pd.notna(defender['s']) else 3.0
            def_dir = defender['dir_norm'] if pd.notna(defender['dir_norm']) else 0
            
            dist, vel_bin, effective_time, _ = get_player_distribution_interpolated(
                def_speed, flight_time
            )
            
            if dist is not None:
                def_grid_x_coords = np.arange(def_x - 12, def_x + 12, 1.0)
                def_grid_y_coords = np.arange(max(0, def_y - 12), min(53.3, def_y + 12), 1.0)
                def_grid_x, def_grid_y = np.meshgrid(def_grid_x_coords, def_grid_y_coords)
                
                def_density = transform_distribution_to_field(
                    dist, def_x, def_y, def_dir, def_grid_x, def_grid_y
                )
                
                def_density_expanded = expand_density_with_catch_radius(
                    def_density, 1.0, CATCH_RADIUS
                )
                
                if def_density_expanded.max() > 0:
                    def_density_norm = def_density_expanded / def_density_expanded.max()
                else:
                    continue
                
                def_extent = [def_grid_x.min(), def_grid_x.max(), 
                             def_grid_y.min(), def_grid_y.max()]
                
                # Red shading for defenders
                #red_rgba = np.zeros((*def_density_norm.shape, 4))
                #red_rgba[:, :, 0] = 0.85  # Red
                #red_rgba[:, :, 1] = 0.15  # Low green
                #red_rgba[:, :, 2] = 0.15  # Low blue
                #red_rgba[:, :, 3] = def_density_norm * 0.6  # Alpha
                
                #ax.imshow(red_rgba, extent=def_extent, origin='lower', 
                         #aspect='auto', zorder=2)
                
                # P95 contour
                p95 = compute_p95_threshold(def_density_expanded)
                if p95 > 0:
                    try:
                        ax.contour(def_grid_x, def_grid_y, def_density_expanded,
                                  levels=[p95], colors=['darkred'], linewidths=1.5,
                                  linestyles='--', alpha=0.8, zorder=3)
                    except:
                        pass
            
            # Defender marker (RED SQUARE)
            ax.scatter(def_x, def_y, c='red', s=200, marker='s',
                      edgecolors='white', linewidths=2, zorder=10)
            
            # Velocity arrow
            if def_speed > 0.5:
                dx = def_speed * velocity_scale * np.cos(np.radians(def_dir))
                dy = def_speed * velocity_scale * np.sin(np.radians(def_dir))
                ax.arrow(def_x, def_y, dx, dy,
                        head_width=0.6, head_length=0.3,
                        fc='red', ec='darkred', linewidth=1.5,
                        zorder=11, alpha=0.9)
    
    # ========================================================================
    # PLOT RECEIVER CLOUDS - Color by Expected Value, Alpha by Reach Prob
    # ========================================================================
    
    epa_min, epa_max = epa_range
    
    for rank, rec_data in enumerate(receiver_vwo_data):
        rec_x = rec_data['receiver_x']
        rec_y = rec_data['receiver_y']
        rec_speed = rec_data['speed']
        rec_dir = rec_data['direction']
        
        if 'grid_x' not in rec_data:
            continue
        
        grid_x = rec_data['grid_x']
        grid_y = rec_data['grid_y']
        reach_prob = rec_data['reach_prob']
        expected_value = rec_data['expected_value']
        
        extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
        
        # Normalize expected_value to [0, 1] for colormap
        # epa_min -> 0, epa_max -> 1
        epa_normalized = (expected_value - epa_min) / (epa_max - epa_min)
        epa_normalized = np.clip(epa_normalized, 0, 1)
        
        # Normalize reach_prob for alpha (scale to make visible)
        reach_max = reach_prob.max()
        if reach_max > 0:
            reach_normalized = reach_prob / reach_max
        else:
            continue
        
        # Create RGBA array
        # Color: Blue (high EPA) -> White (zero) -> Red (negative EPA)
        rgba = np.zeros((*expected_value.shape, 4))
        
        for i in range(expected_value.shape[0]):
            for j in range(expected_value.shape[1]):
                epa_norm = epa_normalized[i, j]
                reach_norm = reach_normalized[i, j]
                
                # Skip very low reach probability areas
                if reach_norm < 0.02:
                    continue
                
                # Map EPA to color
                # 0 (min EPA) = Red, 0.5 (zero EPA) = White, 1 (max EPA) = Blue
                if epa_norm >= 0.5:
                    # White to Blue (positive EPA)
                    t = (epa_norm - 0.5) * 2  # 0 to 1
                    rgba[i, j, 0] = 1.0 - t * 0.8       # R: 1.0 -> 0.2
                    rgba[i, j, 1] = 1.0 - t * 0.6       # G: 1.0 -> 0.4
                    rgba[i, j, 2] = 1.0                  # B: stays 1.0
                else:
                    # Red to White (negative EPA)
                    t = epa_norm * 2  # 0 to 1
                    rgba[i, j, 0] = 1.0                  # R: stays 1.0
                    rgba[i, j, 1] = t * 0.7             # G: 0 -> 0.7
                    rgba[i, j, 2] = t * 0.7             # B: 0 -> 0.7
                
                # Alpha based on reach probability
                # Square root to make lower probabilities more visible
                rgba[i, j, 3] = vwo_alpha * np.sqrt(reach_norm)
        
        ax.imshow(rgba, extent=extent, origin='lower', aspect='auto', zorder=4)
        
        # P95 contour for reach
        reach_expanded = expand_density_with_catch_radius(reach_prob, 1.0, CATCH_RADIUS)
        p95_reach = compute_p95_threshold(reach_expanded)
        
        if p95_reach > 0:
            try:
                contour_color = 'navy' if rec_data['player_role'] == 'Targeted Receiver' else 'blue'
                contour_width = 3.0 if rec_data['player_role'] == 'Targeted Receiver' else 2.0
                ax.contour(grid_x, grid_y, reach_expanded,
                          levels=[p95_reach], colors=[contour_color], linewidths=contour_width,
                          linestyles='-', alpha=0.9, zorder=5)
            except:
                pass
        
        # Sweet spot marker (max VWO point)
        if show_sweet_spots:
            ax.scatter(rec_data['sweet_spot_x'], rec_data['sweet_spot_y'],
                      c='gold', s=180, marker='*', edgecolors='black',
                      linewidths=1.5, zorder=11)
        
        # Receiver position (BLUE CIRCLE)
        marker_color = 'navy' if rec_data['player_role'] == 'Targeted Receiver' else 'blue'
        marker_size = 280 if rec_data['player_role'] == 'Targeted Receiver' else 220
        ax.scatter(rec_x, rec_y, c=marker_color, s=marker_size, marker='o',
                  edgecolors='white', linewidths=2.5, zorder=12)
        
        # Velocity arrow
        if rec_speed > 0.5:
            dx = rec_speed * velocity_scale * np.cos(np.radians(rec_dir))
            dy = rec_speed * velocity_scale * np.sin(np.radians(rec_dir))
            ax.arrow(rec_x, rec_y, dx, dy,
                    head_width=0.6, head_length=0.3,
                    fc='dodgerblue', ec='navy', linewidth=1.5,
                    zorder=13, alpha=0.9)
        
        # VWO label
        if show_vwo_labels:
            vwo_val = rec_data['vwo']
            comp_pct = rec_data['avg_completion_prob'] * 100
            epa_val = rec_data['avg_epa_complete']
            
            # Color based on VWO value
            if vwo_val > 0.3:
                label_color = 'darkblue'
                label_bg = '#e6f0ff'
            elif vwo_val > 0:
                label_color = 'blue'
                label_bg = '#f0f5ff'
            elif vwo_val > -0.3:
                label_color = 'darkorange'
                label_bg = '#fff5e6'
            else:
                label_color = 'darkred'
                label_bg = '#ffe6e6'
            
            # Build label text
            label_lines = []
            if rank == 0:
                label_lines.append("#1 OPTIMAL")
            elif rec_data['player_role'] == 'Targeted Receiver':
                label_lines.append("-> TARGET")
            
            label_lines.append(f"VWO: {vwo_val:+.2f}")
            label_lines.append(f"Comp: {comp_pct:.0f}%")
            label_lines.append(f"EPA: {epa_val:+.2f}")
            
            label_text = "\n".join(label_lines)
            
            ax.annotate(label_text, (rec_x, rec_y + 3),
                       ha='center', va='bottom', fontsize=8, fontweight='bold',
                       color=label_color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=label_bg, 
                                alpha=0.95, edgecolor=label_color))
    
    # ========================================================================
    # PLOT QB (YELLOW)
    # ========================================================================
    
    if len(qb_data) > 0:
        qb = qb_data.iloc[0]
        qb_x = qb['x_norm']
        qb_y = qb['y_norm']
        qb_speed = qb['s'] if pd.notna(qb['s']) else 0
        qb_dir = qb['dir_norm'] if pd.notna(qb['dir_norm']) else 0
        
        ax.scatter(qb_x, qb_y, c='gold', s=350, marker='s',
                  edgecolors='black', linewidths=3, zorder=14)
        ax.annotate('QB', (qb_x, qb_y - 2.5),
                   ha='center', fontsize=10, fontweight='bold', color='black')
        
        if qb_speed > 0.5:
            dx = qb_speed * velocity_scale * np.cos(np.radians(qb_dir))
            dy = qb_speed * velocity_scale * np.sin(np.radians(qb_dir))
            ax.arrow(qb_x, qb_y, dx, dy,
                    head_width=0.6, head_length=0.3,
                    fc='gold', ec='black', linewidth=1.5,
                    zorder=15, alpha=0.9)
    
    # ========================================================================
    # FIELD MARKERS
    # ========================================================================
    
    # Line of scrimmage
    ax.axvline(x=los_x, color='yellow', linewidth=3, linestyle='-', alpha=0.9, zorder=6)
    ax.annotate('LOS', (los_x, y_max - 1.5), ha='center', fontsize=10, 
               fontweight='bold', color='yellow',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # First down line
    first_down_x = los_x + yards_to_go
    if first_down_x < 110:
        ax.axvline(x=first_down_x, color='orange', linewidth=3, linestyle='-', 
                  alpha=0.9, zorder=6)
        ax.annotate('1st', (first_down_x, y_max - 1.5), ha='center', fontsize=10,
                   fontweight='bold', color='orange',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # ========================================================================
    # LEGEND AND TITLE
    # ========================================================================
    
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor='royalblue', alpha=0.8, label=f'Positive EPA (+{epa_max:.0f})'),
        Patch(facecolor='white', edgecolor='gray', alpha=0.8, label='Zero EPA'),
        Patch(facecolor='indianred', alpha=0.8, label=f'Negative EPA ({epa_min:.0f})'),
        Patch(facecolor='darkred', alpha=0.5, label='Defender zone'),
        Line2D([0], [0], color='navy', linewidth=2.5, label='Receiver reach (P95)'),
        Line2D([0], [0], color='darkred', linewidth=1.5, linestyle='--', label='Defender reach (P95)'),
        Line2D([0], [0], marker='*', color='gold', markersize=12, linestyle='None', 
               markeredgecolor='black', label='Sweet spot (max VWO)'),
        Line2D([0], [0], marker='o', color='blue', markersize=10, linestyle='None',
               markeredgecolor='white', label='Receiver'),
        Line2D([0], [0], marker='s', color='red', markersize=10, linestyle='None',
               markeredgecolor='white', label='Defender'),
        Line2D([0], [0], marker='s', color='gold', markersize=10, linestyle='None',
               markeredgecolor='black', label='Quarterback'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.95,
             title='Color = EPA, Opacity = Reach Prob', title_fontsize=8)
    
    # Title
    optimal = receiver_vwo_data[0]
    actual = next((r for r in receiver_vwo_data if r['player_role'] == 'Targeted Receiver'), None)
    
    title = f"Value-Weighted Openness (VWO) - Game {game_id}, Play {play_id}\n"
    title += f"{down}&{yards_to_go} | Flight: {flight_time:.2f}s | Result: {pass_result}\n"
    title += f"Optimal: {optimal['player_name'][:15]} (VWO={optimal['vwo']:+.2f})"
    if actual:
        vwo_diff = optimal['vwo'] - actual['vwo']
        title += f" | Actual: {actual['player_name'][:15]} (VWO={actual['vwo']:+.2f})"
        if vwo_diff > 0.1:
            title += f" | Gap: {vwo_diff:.2f} EPA"
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Field Position (yards)', fontsize=11)
    ax.set_ylabel('Field Width (yards)', fontsize=11)
    ax.set_aspect('equal')
    
    fig.patch.set_facecolor('#f0f0f0')
    plt.tight_layout()
    
    print(f"\n  Visualization complete!")
    
    return fig, {'receivers': receiver_vwo_data, 'optimal': optimal, 'actual': actual}


# Test it
print("Updated VWO Visualization ready!")
print("""
NEW APPROACH:
  - COLOR = Expected Value at each grid point
    â€¢ Blue = positive EPA (valuable target location)
    â€¢ White = zero EPA  
    â€¢ Red = negative EPA (backward/losing yards)
  
  - OPACITY = Reach probability
    â€¢ Darker = receiver more likely to reach this spot
    â€¢ Faint = receiver unlikely to get there
    
  - Labels now show VWO, Completion %, AND EPA

Test with:
  fig, data = visualize_vwo_field(2023110510, 3666)
  plt.show()
""")


fig, data = visualize_vwo_field(2023110510, 3666)
plt.show()


# # Analyze VWO Distribution

# def analyze_vwo_distribution(sample_size=500):
#     """
#     Analyze the distribution of VWO values to determine appropriate scaling.
#     """
#     print("=" * 70)
#     print("VWO DISTRIBUTION ANALYSIS")
#     print("=" * 70)
    
#     # Collect VWO values from sample of plays
#     plays = df_input.groupby(['game_id', 'play_id']).first().reset_index()
#     if sample_size and sample_size < len(plays):
#         plays = plays.sample(sample_size, random_state=42)
    
#     all_vwos = []
    
#     print(f"\n  Collecting VWO from {len(plays)} plays...")
    
#     for idx, (_, play) in enumerate(plays.iterrows()):
#         try:
#             result = calculate_play_vwo_v2(
#                 play['game_id'], play['play_id'], 
#                 verbose=False, lambda_risk=LAMBDA_RISK
#             )
            
#             if result is None:
#                 continue
            
#             vwo_df = result['vwo_summary']
#             for _, rec in vwo_df.iterrows():
#                 all_vwos.append({
#                     'vwo': rec['vwo'],
#                     'is_target': rec['player_role'] == 'Targeted Receiver',
#                     'comp_prob': rec['avg_completion_prob'],
#                     'epa': rec['avg_epa_complete']
#                 })
#         except:
#             continue
        
#         if (idx + 1) % 100 == 0:
#             print(f"    Processed {idx + 1}/{len(plays)} plays...")
    
#     df_vwo = pd.DataFrame(all_vwos)
    
#     print(f"\n  Collected {len(df_vwo)} VWO observations")
    
#     # Basic statistics
#     print(f"\n" + "-" * 70)
#     print("  DISTRIBUTION STATISTICS")
#     print("-" * 70)
    
#     print(f"\n  All Receivers:")
#     print(f"    Min:      {df_vwo['vwo'].min():>8.3f}")
#     print(f"    5th %:    {df_vwo['vwo'].quantile(0.05):>8.3f}")
#     print(f"    25th %:   {df_vwo['vwo'].quantile(0.25):>8.3f}")
#     print(f"    Median:   {df_vwo['vwo'].median():>8.3f}")
#     print(f"    Mean:     {df_vwo['vwo'].mean():>8.3f}")
#     print(f"    75th %:   {df_vwo['vwo'].quantile(0.75):>8.3f}")
#     print(f"    95th %:   {df_vwo['vwo'].quantile(0.95):>8.3f}")
#     print(f"    Max:      {df_vwo['vwo'].max():>8.3f}")
#     print(f"    Std Dev:  {df_vwo['vwo'].std():>8.3f}")
    
#     # Skewness
#     from scipy import stats
#     skewness = stats.skew(df_vwo['vwo'].dropna())
#     kurtosis = stats.kurtosis(df_vwo['vwo'].dropna())
    
#     print(f"\n    Skewness: {skewness:>8.3f}  (0 = symmetric, >0 = right-skewed)")
#     print(f"    Kurtosis: {kurtosis:>8.3f}  (0 = normal, >0 = heavy tails)")
    
#     # Check symmetry around 0
#     pct_negative = (df_vwo['vwo'] < 0).mean() * 100
#     pct_positive = (df_vwo['vwo'] > 0).mean() * 100
    
#     print(f"\n    % Negative: {pct_negative:.1f}%")
#     print(f"    % Positive: {pct_positive:.1f}%")
    
#     # Percentile breakdown
#     print(f"\n" + "-" * 70)
#     print("  PERCENTILE TABLE")
#     print("-" * 70)
    
#     percentiles = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    
#     print(f"\n  {'Percentile':<12} {'VWO Value':<12} {'Would map to (0-10 linear)':<25}")
#     print("  " + "-" * 50)
    
#     vwo_min_99 = df_vwo['vwo'].quantile(0.01)
#     vwo_max_99 = df_vwo['vwo'].quantile(0.99)
    
#     for p in percentiles:
#         vwo_val = df_vwo['vwo'].quantile(p/100)
#         # Linear mapping
#         linear_score = (vwo_val - vwo_min_99) / (vwo_max_99 - vwo_min_99) * 10
#         linear_score = np.clip(linear_score, 0, 10)
#         print(f"  {p:>5}th      {vwo_val:>+8.3f}      {linear_score:>8.1f}")
    
#     # Histogram data
#     print(f"\n" + "-" * 70)
#     print("  VWO HISTOGRAM (text)")
#     print("-" * 70)
    
#     bins = [-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    
#     print(f"\n  {'VWO Range':<18} {'Count':>8} {'Pct':>8} {'Bar':<30}")
#     print("  " + "-" * 65)
    
#     for i in range(len(bins) - 1):
#         count = ((df_vwo['vwo'] >= bins[i]) & (df_vwo['vwo'] < bins[i+1])).sum()
#         pct = count / len(df_vwo) * 100
#         bar = "â–ˆ" * int(pct * 2)
#         print(f"  [{bins[i]:>+5.2f}, {bins[i+1]:>+5.2f})   {count:>6}   {pct:>6.1f}%  {bar}")
    
#     # Compare targeted vs non-targeted
#     print(f"\n" + "-" * 70)
#     print("  TARGETED vs NON-TARGETED RECEIVERS")
#     print("-" * 70)
    
#     targeted = df_vwo[df_vwo['is_target'] == True]['vwo']
#     non_targeted = df_vwo[df_vwo['is_target'] == False]['vwo']
    
#     print(f"\n  {'Metric':<15} {'Targeted':<15} {'Non-Targeted':<15}")
#     print("  " + "-" * 45)
#     print(f"  {'Count':<15} {len(targeted):<15} {len(non_targeted):<15}")
#     print(f"  {'Mean':<15} {targeted.mean():>+8.3f}       {non_targeted.mean():>+8.3f}")
#     print(f"  {'Median':<15} {targeted.median():>+8.3f}       {non_targeted.median():>+8.3f}")
#     print(f"  {'Std Dev':<15} {targeted.std():>8.3f}       {non_targeted.std():>8.3f}")
    
#     # Recommendation
#     print(f"\n" + "=" * 70)
#     print("  TRANSFORMATION RECOMMENDATION")
#     print("=" * 70)
    
#     if abs(skewness) < 0.5:
#         print(f"\n  Distribution is approximately SYMMETRIC (skew={skewness:.2f})")
#         print(f"  â†’ Linear transformation is appropriate")
#     else:
#         print(f"\n  Distribution is SKEWED (skew={skewness:.2f})")
#         print(f"  â†’ Consider percentile-based transformation")
    
#     if abs(df_vwo['vwo'].mean()) > 0.1:
#         print(f"\n  Mean is not centered at 0 (mean={df_vwo['vwo'].mean():.3f})")
#         print(f"  â†’ Consider using median ({df_vwo['vwo'].median():.3f}) as neutral point")
#     else:
#         print(f"\n  Mean is close to 0 (mean={df_vwo['vwo'].mean():.3f})")
#         print(f"  â†’ 0 can be used as neutral point")
    
#     # Suggested ranges
#     print(f"\n  Suggested Mapping Ranges:")
#     print(f"    Conservative: ({df_vwo['vwo'].quantile(0.05):.2f}, {df_vwo['vwo'].quantile(0.95):.2f}) â†’ (0, 10)")
#     print(f"    Full range:   ({df_vwo['vwo'].quantile(0.01):.2f}, {df_vwo['vwo'].quantile(0.99):.2f}) â†’ (0, 10)")
    
#     return df_vwo


# # Run the analysis
# df_vwo_dist = analyze_vwo_distribution(sample_size=500)


print("=" * 70)
print("PITCH CONTROL MODEL VALIDATION (PRE-COMPUTED)")
print("=" * 70)

print("""
======================================================================
VWO DISTRIBUTION ANALYSIS
======================================================================

  Collecting VWO from 500 plays...
    Processed 100/500 plays...
    Processed 200/500 plays...
    Processed 300/500 plays...
    Processed 400/500 plays...
    Processed 500/500 plays...

  Collected 2306 VWO observations

----------------------------------------------------------------------
  DISTRIBUTION STATISTICS
----------------------------------------------------------------------

  All Receivers:
    Min:        -7.784
    5th %:      -4.360
    25th %:     -1.615
    Median:      0.068
    Mean:       -0.311
    75th %:      0.834
    95th %:      3.584
    Max:         6.221
    Std Dev:     2.301

    Skewness:   -0.183  (0 = symmetric, >0 = right-skewed)
    Kurtosis:    0.347  (0 = normal, >0 = heavy tails)

    % Negative: 48.0%
    % Positive: 52.0%

----------------------------------------------------------------------
  PERCENTILE TABLE
----------------------------------------------------------------------

  Percentile   VWO Value    Would map to (0-10 linear)
  --------------------------------------------------
      1th        -5.991           0.0
      5th        -4.360           1.4
     10th        -3.561           2.1
     20th        -2.477           3.1
     30th        -1.012           4.4
     40th        -0.281           5.0
     50th        +0.068           5.3
     60th        +0.347           5.6
     70th        +0.641           5.8
     80th        +1.082           6.2
     90th        +2.342           7.3
     95th        +3.584           8.4
     99th        +5.373          10.0

----------------------------------------------------------------------
  VWO HISTOGRAM (text)
----------------------------------------------------------------------

  VWO Range             Count      Pct Bar                           
  -----------------------------------------------------------------
  [-2.00, -1.50)       72      3.1%  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ
  [-1.50, -1.00)      101      4.4%  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ
  [-1.00, -0.75)       73      3.2%  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ
  [-0.75, -0.50)       59      2.6%  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆ
  [-0.50, -0.25)      110      4.8%  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ
  [-0.25, +0.00)      168      7.3%  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ
  [+0.00, +0.25)      197      8.5%  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ
  [+0.25, +0.50)      213      9.2%  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ
  [+0.50, +0.75)      174      7.5%  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ
  [+0.75, +1.00)      123      5.3%  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ
  [+1.00, +1.50)      148      6.4%  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ
  [+1.50, +2.00)       79      3.4%  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ

----------------------------------------------------------------------
  TARGETED vs NON-TARGETED RECEIVERS
----------------------------------------------------------------------

  Metric          Targeted        Non-Targeted   
  ---------------------------------------------
  Count           500             1806           
  Mean              -0.348         -0.300
  Median            +0.073         +0.064
  Std Dev            2.312          2.298

======================================================================
  TRANSFORMATION RECOMMENDATION
======================================================================

  Distribution is approximately SYMMETRIC (skew=-0.18)
  â†’ Linear transformation is appropriate

  Mean is not centered at 0 (mean=-0.311)
  â†’ Consider using median (0.068) as neutral point

  Suggested Mapping Ranges:
    Conservative: (-4.36, 3.58) â†’ (0, 10)
    Full range:   (-5.99, 5.37) â†’ (0, 10)
""")

print("=" * 70)



# # VWO Scoring System and Updated Visualization

# # =============================================================================
# # VWO SCORE TRANSFORMATION
# # =============================================================================

# def vwo_to_score(vwo, clip=True):
#     """
#     Transform raw VWO (EPA units) to 0-10 score.
    
#     Formula: Score = 5 + (VWO Ã— 1.25)
    
#     This maps:
#         VWO -4.0 â†’ Score 0
#         VWO -2.0 â†’ Score 2.5
#         VWO  0.0 â†’ Score 5.0 (neutral)
#         VWO +2.0 â†’ Score 7.5
#         VWO +4.0 â†’ Score 10
    
#     Parameters:
#         vwo: Raw VWO value in EPA units
#         clip: If True, clip to [0, 10] range
        
#     Returns:
#         Score on 0-10 scale
#     """
#     score = 5 + (vwo * 1.25)
#     if clip:
#         score = np.clip(score, 0, 10)
#     return score


# def score_to_vwo(score):
#     """Reverse transformation: Score to VWO"""
#     return (score - 5) / 1.25


# def vwo_score_to_grade(score):
#     """
#     Convert 0-10 score to letter grade.
    
#     Grade boundaries:
#         A+ : 9.0+ (elite target)
#         A  : 8.0-8.9 (excellent)
#         B  : 7.0-7.9 (good)
#         C  : 5.5-6.9 (average)
#         D  : 4.0-5.4 (below average)
#         F  : 2.0-3.9 (poor)
#         F- : <2.0 (terrible)
#     """
#     if score >= 9.0:
#         return 'A+'
#     elif score >= 8.0:
#         return 'A'
#     elif score >= 7.0:
#         return 'B'
#     elif score >= 5.5:
#         return 'C'
#     elif score >= 4.0:
#         return 'D'
#     elif score >= 2.0:
#         return 'F'
#     else:
#         return 'F-'


# def vwo_score_to_color(score):
#     """
#     Map score to color for visualization.
    
#     Returns RGB tuple.
#     """
#     if score >= 7.0:
#         # Blue gradient (good)
#         t = min(1.0, (score - 7.0) / 3.0)
#         return (0.1, 0.3 + 0.2*t, 0.7 + 0.3*t)  # Darker blue for higher
#     elif score >= 5.0:
#         # White to light blue (neutral to slightly good)
#         t = (score - 5.0) / 2.0
#         return (1.0 - 0.5*t, 1.0 - 0.3*t, 1.0)
#     elif score >= 3.0:
#         # Light red to white (slightly bad to neutral)
#         t = (score - 3.0) / 2.0
#         return (1.0, 0.7 + 0.3*t, 0.7 + 0.3*t)
#     else:
#         # Red gradient (bad)
#         t = min(1.0, (3.0 - score) / 3.0)
#         return (0.7 + 0.3*t, 0.1, 0.1)  # Darker red for lower


# # Print reference table
# print("=" * 70)
# print("VWO SCORE REFERENCE")
# print("=" * 70)
# print("""
#   Formula: Score = 5 + (VWO Ã— 1.25)
  
#   Score   VWO      Grade   Interpretation
#   -----   -----    -----   --------------
#    10     +4.0      A+     Elite - wide open, high value downfield
#     9     +3.2      A+     Excellent target
#     8     +2.4      A      Very good target
#     7     +1.6      B      Good target - clear advantage
#     6     +0.8      C      Above average
#     5      0.0      C      Neutral - average play
#     4     -0.8      D      Below average
#     3     -1.6      F      Poor target
#     2     -2.4      F      Bad target - losing value
#     1     -3.2      F-     Very bad target
#     0     -4.0      F-     Terrible - don't throw here
# """)


# # =============================================================================
# # UPDATED VISUALIZATION WITH SCORES (No Defender Shading)
# # =============================================================================

# def visualize_vwo_field(game_id, play_id, figsize=(16, 10), 
#                         show_defenders=True, 
#                         show_defender_shading=False,  # OFF by default now
#                         show_vwo_labels=True,
#                         show_sweet_spots=True, 
#                         vwo_alpha=0.85, velocity_scale=0.5,
#                         epa_range=(-3.2, 3.2)):  # Maps to score 1-9
#     """
#     Visualize VWO for all receivers on the field.
    
#     NEW FEATURES:
#     - VWO Score (0-10) displayed instead of raw EPA
#     - Defender shading removed by default (cleaner view)
#     - Color scale matched to score system
#     """
#     print(f"\n{'='*70}")
#     print(f"VWO FIELD VISUALIZATION: Game {game_id}, Play {play_id}")
#     print(f"{'='*70}")
    
#     # Get play data
#     play_input = df_input[
#         (df_input['game_id'] == game_id) & 
#         (df_input['play_id'] == play_id)
#     ]
    
#     if len(play_input) == 0:
#         print("  No data found")
#         return None, None
    
#     # Get play context
#     los_x = play_input['absolute_yardline_number'].iloc[0]
#     down = play_input['down'].iloc[0]
#     yards_to_go = play_input['yards_to_go'].iloc[0]
#     pass_result = play_input['pass_result'].iloc[0]
    
#     # Get last INPUT frame
#     last_frame = get_last_input_frame(game_id, play_id)
#     if last_frame is None:
#         print("  Could not get last input frame")
#         return None, None
    
#     # Get flight time + 1 frame adjustment
#     base_flight_time, _ = get_ball_flight_time(game_id, play_id)
#     if base_flight_time is None:
#         base_flight_time = 1.5
#     flight_time = base_flight_time + (1 / FRAME_RATE)
    
#     print(f"  Situation: {down}&{yards_to_go} at {los_x}")
#     print(f"  Flight time: {flight_time:.2f}s")
#     print(f"  Result: {pass_result}")
    
#     # Identify players
#     if 'coverage_player' in last_frame.columns:
#         defense_players = last_frame[last_frame['coverage_player'] == True]
#     else:
#         defense_players = last_frame[last_frame['player_side'] == 'Defense']
    
#     receivers = last_frame[
#         last_frame['player_role'].isin(['Targeted Receiver', 'Other Route Runner'])
#     ]
    
#     qb_data = last_frame[last_frame['player_role'] == 'Passer']
    
#     print(f"  Receivers: {len(receivers)}")
#     print(f"  Defenders: {len(defense_players)}")
    
#     # Calculate VWO for each receiver
#     receiver_vwo_data = []
    
#     for _, rec in receivers.iterrows():
#         vwo_result = calculate_receiver_vwo_v2(
#             rec, last_frame, los_x, down, yards_to_go, flight_time,
#             grid_spacing=1.0, grid_extent=12, lambda_risk=LAMBDA_RISK
#         )
        
#         if vwo_result['status'] == 'success':
#             vwo_result['nfl_id'] = rec['nfl_id']
#             vwo_result['player_name'] = rec.get('player_name', 'Unknown')
#             vwo_result['player_role'] = rec['player_role']
#             vwo_result['speed'] = rec['s'] if pd.notna(rec['s']) else 0
#             vwo_result['direction'] = rec['dir_norm'] if pd.notna(rec['dir_norm']) else 0
#             # Add score
#             vwo_result['vwo_score'] = vwo_to_score(vwo_result['vwo'])
#             vwo_result['grade'] = vwo_score_to_grade(vwo_result['vwo_score'])
#             receiver_vwo_data.append(vwo_result)
    
#     if len(receiver_vwo_data) == 0:
#         print("  No receiver VWO data")
#         return None, None
    
#     # Sort by VWO
#     receiver_vwo_data = sorted(receiver_vwo_data, key=lambda x: x['vwo'], reverse=True)
    
#     # Print summary with SCORES
#     print(f"\n  Receiver VWO Summary:")
#     print(f"  {'Name':<15} {'Score':>6} {'Grade':>6} {'VWO':>8} {'Comp%':>6}")
#     print(f"  " + "-" * 50)
#     for i, rec_data in enumerate(receiver_vwo_data):
#         role_marker = " <-" if rec_data['player_role'] == 'Targeted Receiver' else ""
#         opt_marker = " [#1]" if i == 0 else ""
#         print(f"  {rec_data['player_name'][:15]:<15} {rec_data['vwo_score']:>6.1f} "
#               f"{rec_data['grade']:>6} {rec_data['vwo']:>+7.2f} "
#               f"{rec_data['avg_completion_prob']*100:>5.0f}%{opt_marker}{role_marker}")
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=figsize)
#     draw_football_field(ax, show_hash_marks=True, show_numbers=True)
    
#     # Determine field bounds
#     all_x = [r['receiver_x'] for r in receiver_vwo_data]
#     all_y = [r['receiver_y'] for r in receiver_vwo_data]
    
#     if len(defense_players) > 0:
#         all_x.extend(defense_players['x_norm'].tolist())
#         all_y.extend(defense_players['y_norm'].tolist())
    
#     if len(qb_data) > 0:
#         all_x.append(qb_data.iloc[0]['x_norm'])
#         all_y.append(qb_data.iloc[0]['y_norm'])
    
#     x_min = min(all_x) - 15
#     x_max = max(all_x) + 15
#     y_min = max(0, min(all_y) - 10)
#     y_max = min(53.3, max(all_y) + 10)
    
#     # ========================================================================
#     # PLOT DEFENDERS (No shading, just markers and contours)
#     # ========================================================================
    
#     if show_defenders:
#         for _, defender in defense_players.iterrows():
#             def_x = defender['x_norm']
#             def_y = defender['y_norm']
#             def_speed = defender['s'] if pd.notna(defender['s']) else 3.0
#             def_dir = defender['dir_norm'] if pd.notna(defender['dir_norm']) else 0
            
#             dist, vel_bin, effective_time, _ = get_player_distribution_interpolated(
#                 def_speed, flight_time
#             )
            
#             if dist is not None:
#                 def_grid_x_coords = np.arange(def_x - 12, def_x + 12, 1.0)
#                 def_grid_y_coords = np.arange(max(0, def_y - 12), min(53.3, def_y + 12), 1.0)
#                 def_grid_x, def_grid_y = np.meshgrid(def_grid_x_coords, def_grid_y_coords)
                
#                 def_density = transform_distribution_to_field(
#                     dist, def_x, def_y, def_dir, def_grid_x, def_grid_y
#                 )
                
#                 def_density_expanded = expand_density_with_catch_radius(
#                     def_density, 1.0, CATCH_RADIUS
#                 )
                
#                 if def_density_expanded.max() > 0:
#                     def_density_norm = def_density_expanded / def_density_expanded.max()
                
#                     # OPTIONAL: Defender shading (only if enabled)
#                     if show_defender_shading:
#                         def_extent = [def_grid_x.min(), def_grid_x.max(), 
#                                      def_grid_y.min(), def_grid_y.max()]
                        
#                         red_rgba = np.zeros((*def_density_norm.shape, 4))
#                         red_rgba[:, :, 0] = 0.85
#                         red_rgba[:, :, 1] = 0.15
#                         red_rgba[:, :, 2] = 0.15
#                         red_rgba[:, :, 3] = def_density_norm * 0.5
                        
#                         ax.imshow(red_rgba, extent=def_extent, origin='lower', 
#                                  aspect='auto', zorder=2)
                    
#                     # P95 contour (always show)
#                     p95 = compute_p95_threshold(def_density_expanded)
#                     if p95 > 0:
#                         try:
#                             ax.contour(def_grid_x, def_grid_y, def_density_expanded,
#                                       levels=[p95], colors=['darkred'], linewidths=1.5,
#                                       linestyles='--', alpha=0.7, zorder=3)
#                         except:
#                             pass
            
#             # Defender marker (RED SQUARE)
#             ax.scatter(def_x, def_y, c='red', s=200, marker='s',
#                       edgecolors='white', linewidths=2, zorder=10)
            
#             # Velocity arrow
#             if def_speed > 0.5:
#                 dx = def_speed * velocity_scale * np.cos(np.radians(def_dir))
#                 dy = def_speed * velocity_scale * np.sin(np.radians(def_dir))
#                 ax.arrow(def_x, def_y, dx, dy,
#                         head_width=0.6, head_length=0.3,
#                         fc='red', ec='darkred', linewidth=1.5,
#                         zorder=11, alpha=0.9)
    
#     # ========================================================================
#     # PLOT RECEIVER CLOUDS - Color by Expected Value (Score), Alpha by Reach
#     # ========================================================================
    
#     epa_min, epa_max = epa_range  # Default (-3.2, 3.2) â†’ Score (1, 9)
    
#     for rank, rec_data in enumerate(receiver_vwo_data):
#         rec_x = rec_data['receiver_x']
#         rec_y = rec_data['receiver_y']
#         rec_speed = rec_data['speed']
#         rec_dir = rec_data['direction']
        
#         if 'grid_x' not in rec_data:
#             continue
        
#         grid_x = rec_data['grid_x']
#         grid_y = rec_data['grid_y']
#         reach_prob = rec_data['reach_prob']
#         expected_value = rec_data['expected_value']
        
#         extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
        
#         # Convert EPA to score for color mapping
#         # epa_min â†’ score 1, epa_max â†’ score 9, 0 â†’ score 5
#         epa_to_score_grid = 5 + (expected_value * 1.25)
#         epa_to_score_grid = np.clip(epa_to_score_grid, 0, 10)
        
#         # Normalize reach_prob for alpha
#         reach_max = reach_prob.max()
#         if reach_max > 0:
#             reach_normalized = reach_prob / reach_max
#         else:
#             continue
        
#         # Create RGBA array based on SCORE
#         rgba = np.zeros((*expected_value.shape, 4))
        
#         for i in range(expected_value.shape[0]):
#             for j in range(expected_value.shape[1]):
#                 score = epa_to_score_grid[i, j]
#                 reach_norm = reach_normalized[i, j]
                
#                 # Skip very low reach probability areas
#                 if reach_norm < 0.02:
#                     continue
                
#                 # Map score to color
#                 if score >= 6.0:
#                     # Blue gradient (good: score 6-10)
#                     t = min(1.0, (score - 6.0) / 4.0)
#                     rgba[i, j, 0] = 0.2 - 0.1*t       # R: low
#                     rgba[i, j, 1] = 0.4 + 0.2*t       # G: medium
#                     rgba[i, j, 2] = 0.8 + 0.2*t       # B: high
#                 elif score >= 5.0:
#                     # White to light blue (neutral: score 5-6)
#                     t = (score - 5.0)
#                     rgba[i, j, 0] = 1.0 - 0.8*t       # R
#                     rgba[i, j, 1] = 1.0 - 0.6*t       # G
#                     rgba[i, j, 2] = 1.0               # B
#                 elif score >= 4.0:
#                     # Light red to white (below average: score 4-5)
#                     t = (score - 4.0)
#                     rgba[i, j, 0] = 1.0               # R
#                     rgba[i, j, 1] = 0.6 + 0.4*t       # G
#                     rgba[i, j, 2] = 0.6 + 0.4*t       # B
#                 else:
#                     # Red gradient (poor: score 0-4)
#                     t = min(1.0, (4.0 - score) / 4.0)
#                     rgba[i, j, 0] = 0.8 + 0.2*t       # R: high
#                     rgba[i, j, 1] = 0.2 - 0.1*t       # G: low
#                     rgba[i, j, 2] = 0.2 - 0.1*t       # B: low
                
#                 # Alpha based on reach probability
#                 rgba[i, j, 3] = vwo_alpha * np.sqrt(reach_norm)
        
#         ax.imshow(rgba, extent=extent, origin='lower', aspect='auto', zorder=4)
        
#         # P95 contour for reach
#         reach_expanded = expand_density_with_catch_radius(reach_prob, 1.0, CATCH_RADIUS)
#         p95_reach = compute_p95_threshold(reach_expanded)
        
#         if p95_reach > 0:
#             try:
#                 contour_color = 'navy' if rec_data['player_role'] == 'Targeted Receiver' else 'blue'
#                 contour_width = 3.0 if rec_data['player_role'] == 'Targeted Receiver' else 2.0
#                 ax.contour(grid_x, grid_y, reach_expanded,
#                           levels=[p95_reach], colors=[contour_color], linewidths=contour_width,
#                           linestyles='-', alpha=0.9, zorder=5)
#             except:
#                 pass
        
#         # Sweet spot marker (max VWO point)
#         if show_sweet_spots:
#             ax.scatter(rec_data['sweet_spot_x'], rec_data['sweet_spot_y'],
#                       c='gold', s=180, marker='*', edgecolors='black',
#                       linewidths=1.5, zorder=11)
        
#         # Receiver position (BLUE CIRCLE)
#         marker_color = 'navy' if rec_data['player_role'] == 'Targeted Receiver' else 'blue'
#         marker_size = 280 if rec_data['player_role'] == 'Targeted Receiver' else 220
#         ax.scatter(rec_x, rec_y, c=marker_color, s=marker_size, marker='o',
#                   edgecolors='white', linewidths=2.5, zorder=12)
        
#         # Velocity arrow
#         if rec_speed > 0.5:
#             dx = rec_speed * velocity_scale * np.cos(np.radians(rec_dir))
#             dy = rec_speed * velocity_scale * np.sin(np.radians(rec_dir))
#             ax.arrow(rec_x, rec_y, dx, dy,
#                     head_width=0.6, head_length=0.3,
#                     fc='dodgerblue', ec='navy', linewidth=1.5,
#                     zorder=13, alpha=0.9)
        
#         # VWO label with SCORE
#         if show_vwo_labels:
#             score = rec_data['vwo_score']
#             grade = rec_data['grade']
#             comp_pct = rec_data['avg_completion_prob'] * 100
            
#             # Color based on score
#             if score >= 7.0:
#                 label_color = 'darkblue'
#                 label_bg = '#d0e0ff'
#             elif score >= 5.5:
#                 label_color = 'blue'
#                 label_bg = '#e8f0ff'
#             elif score >= 4.0:
#                 label_color = 'darkorange'
#                 label_bg = '#fff0e0'
#             else:
#                 label_color = 'darkred'
#                 label_bg = '#ffe0e0'
            
#             # Build label
#             label_lines = []
#             if rank == 0:
#                 label_lines.append("#1 OPTIMAL")
#             elif rec_data['player_role'] == 'Targeted Receiver':
#                 label_lines.append("-> TARGET")
            
#             label_lines.append(f"Score: {score:.1f} ({grade})")
#             label_lines.append(f"Comp: {comp_pct:.0f}%")
            
#             label_text = "\n".join(label_lines)
            
#             ax.annotate(label_text, (rec_x, rec_y + 3),
#                        ha='center', va='bottom', fontsize=9, fontweight='bold',
#                        color=label_color,
#                        bbox=dict(boxstyle='round,pad=0.3', facecolor=label_bg, 
#                                 alpha=0.95, edgecolor=label_color))
    
#     # ========================================================================
#     # PLOT QB (YELLOW)
#     # ========================================================================
    
#     if len(qb_data) > 0:
#         qb = qb_data.iloc[0]
#         qb_x = qb['x_norm']
#         qb_y = qb['y_norm']
#         qb_speed = qb['s'] if pd.notna(qb['s']) else 0
#         qb_dir = qb['dir_norm'] if pd.notna(qb['dir_norm']) else 0
        
#         ax.scatter(qb_x, qb_y, c='gold', s=350, marker='s',
#                   edgecolors='black', linewidths=3, zorder=14)
#         ax.annotate('QB', (qb_x, qb_y - 2.5),
#                    ha='center', fontsize=10, fontweight='bold', color='black')
        
#         if qb_speed > 0.5:
#             dx = qb_speed * velocity_scale * np.cos(np.radians(qb_dir))
#             dy = qb_speed * velocity_scale * np.sin(np.radians(qb_dir))
#             ax.arrow(qb_x, qb_y, dx, dy,
#                     head_width=0.6, head_length=0.3,
#                     fc='gold', ec='black', linewidth=1.5,
#                     zorder=15, alpha=0.9)
    
#     # ========================================================================
#     # FIELD MARKERS
#     # ========================================================================
    
#     ax.axvline(x=los_x, color='yellow', linewidth=3, linestyle='-', alpha=0.9, zorder=6)
#     ax.annotate('LOS', (los_x, y_max - 1.5), ha='center', fontsize=10, 
#                fontweight='bold', color='yellow',
#                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
#     first_down_x = los_x + yards_to_go
#     if first_down_x < 110:
#         ax.axvline(x=first_down_x, color='orange', linewidth=3, linestyle='-', 
#                   alpha=0.9, zorder=6)
#         ax.annotate('1st', (first_down_x, y_max - 1.5), ha='center', fontsize=10,
#                    fontweight='bold', color='orange',
#                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
#     # ========================================================================
#     # LEGEND AND TITLE
#     # ========================================================================
    
#     from matplotlib.patches import Patch
#     from matplotlib.lines import Line2D
    
#     legend_elements = [
#         Patch(facecolor='royalblue', alpha=0.8, label='Score 7+ (Good)'),
#         Patch(facecolor='lightblue', alpha=0.8, label='Score 5-7 (Neutral)'),
#         Patch(facecolor='lightsalmon', alpha=0.8, label='Score 4-5 (Below Avg)'),
#         Patch(facecolor='indianred', alpha=0.8, label='Score <4 (Poor)'),
#         Line2D([0], [0], color='navy', linewidth=2.5, label='Receiver reach'),
#         Line2D([0], [0], color='darkred', linewidth=1.5, linestyle='--', label='Defender reach'),
#         Line2D([0], [0], marker='*', color='gold', markersize=12, linestyle='None', 
#                markeredgecolor='black', label='Sweet spot'),
#         Line2D([0], [0], marker='o', color='blue', markersize=10, linestyle='None',
#                markeredgecolor='white', label='Receiver'),
#         Line2D([0], [0], marker='s', color='red', markersize=10, linestyle='None',
#                markeredgecolor='white', label='Defender'),
#         Line2D([0], [0], marker='s', color='gold', markersize=10, linestyle='None',
#                markeredgecolor='black', label='Quarterback'),
#     ]
    
#     ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.95,
#              title='VWO Score (0-10)', title_fontsize=9)
    
#     # Title with SCORES
#     optimal = receiver_vwo_data[0]
#     actual = next((r for r in receiver_vwo_data if r['player_role'] == 'Targeted Receiver'), None)
    
#     title = f"Value-Weighted Openness (VWO) - Game {game_id}, Play {play_id}\n"
#     title += f"{down}&{yards_to_go} | Flight: {flight_time:.2f}s | Result: {pass_result}\n"
#     title += f"Optimal: {optimal['player_name'][:15]} (Score: {optimal['vwo_score']:.1f})"
#     if actual:
#         score_gap = optimal['vwo_score'] - actual['vwo_score']
#         title += f" | Actual: {actual['player_name'][:15]} (Score: {actual['vwo_score']:.1f})"
#         if score_gap > 0.5:
#             title += f" | Gap: {score_gap:.1f} pts"
    
#     ax.set_title(title, fontsize=12, fontweight='bold')
    
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.set_xlabel('Field Position (yards)', fontsize=11)
#     ax.set_ylabel('Field Width (yards)', fontsize=11)
#     ax.set_aspect('equal')
    
#     fig.patch.set_facecolor('#f0f0f0')
#     plt.tight_layout()
    
#     print(f"\n  Visualization complete!")
    
#     return fig, {'receivers': receiver_vwo_data, 'optimal': optimal, 'actual': actual}


# # Quick test function
# def quick_vwo_viz(outcome='Complete', coverage=None):
#     """Quick test with random play"""
#     game_id, play_id = random_play(outcome=outcome, coverage=coverage)
#     if game_id:
#         fig, data = visualize_vwo_field(game_id, play_id)
#         plt.show()
#         return game_id, play_id, fig, data
#     return None, None, None, None


# print("\n" + "=" * 70)
# print("VWO SCORE SYSTEM READY")
# print("=" * 70)
# print("""
# Score Scale:
#   10  = Elite target (VWO +4.0)
#    7  = Good target (VWO +1.6)  
#    5  = Neutral (VWO 0.0)
#    3  = Poor target (VWO -1.6)
#    0  = Terrible (VWO -4.0)

# Test with:
#   fig, data = visualize_vwo_field(2023110510, 3666)
#   plt.show()
# """)


print("=" * 70)
print("VWO Score Transformation (PRE-COMPUTED)")
print("=" * 70)

print("""
======================================================================
VWO SCORE REFERENCE
======================================================================

  Formula: Score = 5 + (VWO Ã— 1.25)
  
  Score   VWO      Grade   Interpretation
  -----   -----    -----   --------------
   10     +4.0      A+     Elite - wide open, high value downfield
    9     +3.2      A+     Excellent target
    8     +2.4      A      Very good target
    7     +1.6      B      Good target - clear advantage
    6     +0.8      C      Above average
    5      0.0      C      Neutral - average play
    4     -0.8      D      Below average
    3     -1.6      F      Poor target
    2     -2.4      F      Bad target - losing value
    1     -3.2      F-     Very bad target
    0     -4.0      F-     Terrible - don't throw here


======================================================================
VWO SCORE SYSTEM READY
======================================================================

Score Scale:
  10  = Elite target (VWO +4.0)
   7  = Good target (VWO +1.6)  
   5  = Neutral (VWO 0.0)
   3  = Poor target (VWO -1.6)
   0  = Terrible (VWO -4.0)

Test with:
  fig, data = visualize_vwo_field(2023110510, 3666)
  plt.show()
""")

print("=" * 70)



fig, data = visualize_vwo_field(2023110510, 3666)
plt.show()


# # Diagnostic: Understand VWO Value Distribution

# print("=" * 70)
# print("DIAGNOSING VWO VALUE DISTRIBUTION")
# print("=" * 70)

# # Use the proximity data we already have
# print("\n1. VWO VALUE STATISTICS")
# print("-" * 50)

# print(f"\n  OPTIMAL VWO (highest per play):")
# print(f"    Mean:   {df_proximity['optimal_vwo'].mean():.4f}")
# print(f"    Median: {df_proximity['optimal_vwo'].median():.4f}")
# print(f"    Min:    {df_proximity['optimal_vwo'].min():.4f}")
# print(f"    Max:    {df_proximity['optimal_vwo'].max():.4f}")
# print(f"    % Negative: {(df_proximity['optimal_vwo'] < 0).mean():.1%}")
# print(f"    % Zero:     {(df_proximity['optimal_vwo'] == 0).mean():.1%}")
# print(f"    % Positive: {(df_proximity['optimal_vwo'] > 0).mean():.1%}")

# print(f"\n  ACTUAL VWO (targeted receiver):")
# print(f"    Mean:   {df_proximity['actual_vwo'].mean():.4f}")
# print(f"    Median: {df_proximity['actual_vwo'].median():.4f}")
# print(f"    Min:    {df_proximity['actual_vwo'].min():.4f}")
# print(f"    Max:    {df_proximity['actual_vwo'].max():.4f}")
# print(f"    % Negative: {(df_proximity['actual_vwo'] < 0).mean():.1%}")
# print(f"    % Zero:     {(df_proximity['actual_vwo'] == 0).mean():.1%}")
# print(f"    % Positive: {(df_proximity['actual_vwo'] > 0).mean():.1%}")

# print(f"\n  WORST VWO (lowest per play):")
# print(f"    Mean:   {df_proximity['worst_vwo'].mean():.4f}")
# print(f"    Median: {df_proximity['worst_vwo'].median():.4f}")
# print(f"    Min:    {df_proximity['worst_vwo'].min():.4f}")
# print(f"    Max:    {df_proximity['worst_vwo'].max():.4f}")

# print("\n2. VWO DISTRIBUTION BY OUTCOME")
# print("-" * 50)

# for outcome, label in [('C', 'Complete'), ('I', 'Incomplete'), ('IN', 'INT')]:
#     subset = df_proximity[df_proximity['pass_result'] == outcome]
#     if len(subset) > 3:
#         print(f"\n  {label} (n={len(subset)}):")
#         print(f"    Mean Optimal VWO: {subset['optimal_vwo'].mean():.4f}")
#         print(f"    Mean Actual VWO:  {subset['actual_vwo'].mean():.4f}")
#         print(f"    % Actual VWO < 0: {(subset['actual_vwo'] < 0).mean():.1%}")
#         print(f"    % Actual VWO = 0: {(subset['actual_vwo'] == 0).mean():.1%}")

# print("\n3. INVESTIGATING THE 'WORST CHOICE' PLAYS")
# print("-" * 50)

# worst_plays = df_proximity[df_proximity['chose_worst'] == True]
# print(f"\n  Plays where QB chose worst VWO receiver (n={len(worst_plays)}):")
# print(f"    Mean Actual VWO:    {worst_plays['actual_vwo'].mean():.4f}")
# print(f"    Mean Optimal VWO:   {worst_plays['optimal_vwo'].mean():.4f}")
# print(f"    % Actual VWO = 0:   {(worst_plays['actual_vwo'] == 0).mean():.1%}")
# print(f"    % Optimal VWO = 0:  {(worst_plays['optimal_vwo'] == 0).mean():.1%}")
# print(f"    Completion Rate:    {(worst_plays['pass_result'] == 'C').mean():.1%}")

# # Check if "worst" means VWO=0 (no distribution)
# print(f"\n  Breakdown of 'worst' plays by VWO value:")
# print(f"    Actual VWO = 0:   {(worst_plays['actual_vwo'] == 0).sum()} plays")
# print(f"    Actual VWO < 0:   {(worst_plays['actual_vwo'] < 0).sum()} plays")
# print(f"    Actual VWO > 0:   {(worst_plays['actual_vwo'] > 0).sum()} plays")

# print("\n4. INVESTIGATING THE 'OPTIMAL' PLAYS")
# print("-" * 50)

# optimal_plays = df_proximity[df_proximity['chose_optimal'] == True]
# print(f"\n  Plays where QB chose optimal VWO receiver (n={len(optimal_plays)}):")
# print(f"    Mean Optimal VWO:   {optimal_plays['optimal_vwo'].mean():.4f}")
# print(f"    % Optimal VWO = 0:  {(optimal_plays['optimal_vwo'] == 0).mean():.1%}")
# print(f"    % Optimal VWO < 0:  {(optimal_plays['optimal_vwo'] < 0).mean():.1%}")
# print(f"    Completion Rate:    {(optimal_plays['pass_result'] == 'C').mean():.1%}")

# print("\n5. SINGLE PLAY DEEP DIVE")
# print("-" * 50)

# # Look at one "optimal" play that was incomplete
# opt_incomplete = df_proximity[(df_proximity['chose_optimal']) & (df_proximity['pass_result'] == 'I')]
# if len(opt_incomplete) > 0:
#     sample = opt_incomplete.iloc[0]
#     print(f"\n  Example: Optimal choice but INCOMPLETE")
#     print(f"    Game: {sample['game_id']}, Play: {sample['play_id']}")
#     print(f"    Optimal VWO: {sample['optimal_vwo']:.4f}")
#     print(f"    N Receivers: {sample['n_receivers']}")
    
#     # Get full details
#     try:
#         result = calculate_play_vwo_v2(sample['game_id'], sample['play_id'], verbose=True)
#     except:
#         print("    Could not retrieve full details")

# # Look at one "worst" play that was complete
# worst_complete = df_proximity[(df_proximity['chose_worst']) & (df_proximity['pass_result'] == 'C')]
# if len(worst_complete) > 0:
#     sample = worst_complete.iloc[0]
#     print(f"\n  Example: Worst choice but COMPLETE")
#     print(f"    Game: {sample['game_id']}, Play: {sample['play_id']}")
#     print(f"    Actual VWO: {sample['actual_vwo']:.4f}")
#     print(f"    Optimal VWO: {sample['optimal_vwo']:.4f}")
#     print(f"    N Receivers: {sample['n_receivers']}")
    
#     try:
#         result = calculate_play_vwo_v2(sample['game_id'], sample['play_id'], verbose=True)
#     except:
#         print("    Could not retrieve full details")

# print("\n6. CHECK FOR VWO=0 DUE TO MISSING DISTRIBUTIONS")
# print("-" * 50)

# # How many plays have VWO=0 for all receivers?
# all_zero = df_proximity[(df_proximity['optimal_vwo'] == 0) & (df_proximity['worst_vwo'] == 0)]
# print(f"  Plays where ALL receivers have VWO=0: {len(all_zero)} ({len(all_zero)/len(df_proximity):.1%})")

# # How many plays have at least one VWO=0 receiver?
# has_zero = df_proximity[df_proximity['worst_vwo'] == 0]
# print(f"  Plays where at least one receiver has VWO=0: {len(has_zero)} ({len(has_zero)/len(df_proximity):.1%})")

# print("\n" + "=" * 70)


print("=" * 70)
print("VWO Value Distributions (PRE-COMPUTED)")
print("=" * 70)

print("""
======================================================================
DIAGNOSING VWO VALUE DISTRIBUTION
======================================================================

1. VWO VALUE STATISTICS
--------------------------------------------------

  OPTIMAL VWO (highest per play):
    Mean:   0.3929
    Median: 0.6523
    Min:    -7.3039
    Max:    8.1029
    % Negative: 27.3%
    % Zero:     0.0%
    % Positive: 72.7%

  ACTUAL VWO (targeted receiver):
    Mean:   -0.2947
    Median: 0.0396
    Min:    -8.5099
    Max:    7.9456
    % Negative: 48.3%
    % Zero:     0.0%
    % Positive: 51.7%

  WORST VWO (lowest per play):
    Mean:   -1.0686
    Median: -0.6995
    Min:    -10.2268
    Max:    6.6964

2. VWO DISTRIBUTION BY OUTCOME
--------------------------------------------------

  Complete (n=3397):
    Mean Optimal VWO: 0.3237
    Mean Actual VWO:  -0.4416
    % Actual VWO < 0: 52.6%
    % Actual VWO = 0: 0.0%

  Incomplete (n=1473):
    Mean Optimal VWO: 0.5757
    Mean Actual VWO:  0.0394
    % Actual VWO < 0: 39.3%
    % Actual VWO = 0: 0.0%

  INT (n=130):
    Mean Optimal VWO: 0.1282
    Mean Actual VWO:  -0.2423
    % Actual VWO < 0: 37.7%
    % Actual VWO = 0: 0.0%

3. INVESTIGATING THE 'WORST CHOICE' PLAYS
--------------------------------------------------

  Plays where QB chose worst VWO receiver (n=1080):
    Mean Actual VWO:    -1.1291
    Mean Optimal VWO:   0.2659
    % Actual VWO = 0:   0.0%
    % Optimal VWO = 0:  0.0%
    Completion Rate:    78.2%

  Breakdown of 'worst' plays by VWO value:
    Actual VWO = 0:   0 plays
    Actual VWO < 0:   808 plays
    Actual VWO > 0:   272 plays

4. INVESTIGATING THE 'OPTIMAL' PLAYS
--------------------------------------------------

  Plays where QB chose optimal VWO receiver (n=898):
    Mean Optimal VWO:   0.4597
    % Optimal VWO = 0:  0.0%
    % Optimal VWO < 0:  28.2%
    Completion Rate:    56.2%

5. SINGLE PLAY DEEP DIVE
--------------------------------------------------

  Example: Optimal choice but INCOMPLETE
    Game: 2023110510, Play: 4050
    Optimal VWO: 1.0434
    N Receivers: 4

  Play 2023110510-4050: 1&10 at 67
  Receivers analyzed: 4

  VWO Rankings:
    â†’ CeeDee Lamb: VWO=1.043, Risk-Adj=0.756, Window=0.00
      Brandin Cooks: VWO=1.015, Risk-Adj=0.796, Window=0.15
      KaVontae Turpin: VWO=0.671, Risk-Adj=0.137, Window=0.00
      Jake Ferguson: VWO=0.347, Risk-Adj=0.128, Window=0.45

  Optimal (VWO): CeeDee Lamb (1.043)
  Target Quality: 100.0%

  Example: Worst choice but COMPLETE
    Game: 2023091003, Play: 2902
    Actual VWO: -2.1399
    Optimal VWO: 0.5915
    N Receivers: 5

  Play 2023091003-2902: 3&5 at 57
  Receivers analyzed: 5

  VWO Rankings:
      Zay Jones: VWO=0.592, Risk-Adj=0.501, Window=0.82
      Evan Engram: VWO=0.406, Risk-Adj=0.256, Window=0.00
      Calvin Ridley: VWO=0.120, Risk-Adj=-0.076, Window=0.03
      Christian Kirk: VWO=-0.499, Risk-Adj=-0.546, Window=0.72
    â†’ Travis Etienne: VWO=-2.140, Risk-Adj=-2.174, Window=0.96

  Optimal (VWO): Zay Jones (0.592)
  Target Quality: -361.8%

6. CHECK FOR VWO=0 DUE TO MISSING DISTRIBUTIONS
--------------------------------------------------
  Plays where ALL receivers have VWO=0: 0 (0.0%)
  Plays where at least one receiver has VWO=0: 0 (0.0%)

======================================================================
""")

print("=" * 70)



# ============================================================================
# CREATE SUBMISSION OUTPUT FILE
# ============================================================================
# The Big Data Bowl requires at least one output file for submission.
# This file summarizes the key findings from this analysis.
# ============================================================================

import pandas as pd

# Create a summary of your analysis
submission_summary = pd.DataFrame({
    'metric': [
        'Total Plays Analyzed',
        'Optimal Target Rate',
        'High VWO Completion Rate',
        'Low VWO Completion Rate',
        'High VWO INT Rate',
        'Low VWO INT Rate'
    ],
    'value': [
        5000,       # Update with your actual numbers
        0.18,       # 18% chose optimal
        0.562,      # 56.2% completion
        0.782,      # 78.2% completion  
        0.041,      # 4.1% INT
        0.015       # 1.5% INT
    ]
})

# Save to output directory
submission_summary.to_csv('/kaggle/working/vwo_analysis_summary.csv', index=False)

print("âœ“ Submission file created: vwo_analysis_summary.csv")
print(submission_summary)

