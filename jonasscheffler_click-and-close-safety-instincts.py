# Core libraries
import numpy as np
import pandas as pd
import os
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


# =========================
# CONFIGURATION
# =========================

DATA_DIR = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final"
INPUT_FILE_PATTERN = os.path.join(
    DATA_DIR, "train", "input_2023_w{week:02d}.csv"
)
OUTPUT_FILE_PATTERN = os.path.join(
    DATA_DIR, "train", "output_2023_w{week:02d}.csv"
)

WEEKS = list(range(1, 19))

# --- TUNING KNOBS ---
# Players must be within this many yards of the landing spot (at throw OR at arrival) 
# to be considered "Relevant" to the play.
RELEVANCE_RADIUS_START = 30.0
RELEVANCE_RADIUS_END = 10.0
ALIGNMENT_THRESHOLD_DEG = 15.0 # Degrees of tolerance for "facing the ball"
# Minimum number of relevant plays required for inclusion
MIN_PLAYS = 100


def calculate_euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def calculate_path_length(x_series, y_series):
    x_diff = np.diff(x_series)
    y_diff = np.diff(y_series)
    return np.sum(np.sqrt(x_diff**2 + y_diff**2))

def calculate_derived_metrics(df):
    """
    Manually calculates Speed (s) and Direction (dir) from x/y coordinates.
    """
    dt = 0.1
    dx = df['x'].diff().fillna(0)
    dy = df['y'].diff().fillna(0)
    
    dist = np.sqrt(dx**2 + dy**2)
    df['s'] = dist / dt
    
    df['move_angle'] = np.degrees(np.arctan2(dy, dx)) % 360
    return df

def calculate_target_angle(player_x, player_y, target_x, target_y):
    """Calculates angle from player to target."""
    dy = target_y - player_y
    dx = target_x - player_x
    return np.degrees(np.arctan2(dy, dx)) % 360

def get_angle_diff(angle1, angle2):
    """Calculates smallest difference between two angles."""
    diff = abs(angle1 - angle2) % 360
    return min(diff, 360 - diff)

def analyze_safety_range(weeks):
    print(f"--- STARTING ANALYSIS: REACTION + CLOSING (Weeks: {weeks}) ---")
    
    player_stats = []

    for w in weeks:
        input_path = INPUT_FILE_PATTERN.format(week=w)
        output_path = OUTPUT_FILE_PATTERN.format(week=w)
        
        if not os.path.exists(input_path) or not os.path.exists(output_path):
            print(f"Missing files for week {w}, skipping.")
            continue
            
        print(f"Processing Week {w}...")
        
        # Load Data
        input_cols = ['game_id', 'play_id', 'nfl_id', 'player_name', 'player_position', 'player_role', 'frame_id', 'x', 'y', 'ball_land_x', 'ball_land_y']
        inp_df = pd.read_csv(input_path, usecols=input_cols)
        
        output_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y']
        out_df = pd.read_csv(output_path, usecols=output_cols)

        # Process Each Play
        for (game_id, play_id), play_in in inp_df.groupby(['game_id', 'play_id']):
            
            ball_land_x = play_in['ball_land_x'].iloc[0]
            ball_land_y = play_in['ball_land_y'].iloc[0]
            if pd.isna(ball_land_x): continue

            play_out_raw = out_df[(out_df['game_id'] == game_id) & (out_df['play_id'] == play_id)].copy()
            if play_out_raw.empty: continue

            safeties = play_in[play_in['player_position'].isin(['FS', 'SS', 'S'])]
            
            if safeties.empty: continue

            # Process Safeties
            for safety_id in safeties['nfl_id'].unique():
                
                d_track = play_out_raw[play_out_raw['nfl_id'] == safety_id].sort_values('frame_id')
                if len(d_track) < 5: continue
                
                start_x, start_y = d_track.iloc[0]['x'], d_track.iloc[0]['y']
                end_x, end_y = d_track.iloc[-1]['x'], d_track.iloc[-1]['y']
                
                # RELEVANCE FILTER
                dist_start = calculate_euclidean_distance(start_x, start_y, ball_land_x, ball_land_y)
                dist_end = calculate_euclidean_distance(end_x, end_y, ball_land_x, ball_land_y)
                
                if not ((dist_start <= RELEVANCE_RADIUS_START) or (dist_end <= RELEVANCE_RADIUS_END)):
                    continue

                # --- METRIC 1: REACTION TIME (Vector Alignment) ---
                d_track = calculate_derived_metrics(d_track)
                reaction_time = np.nan
                
                for _, row in d_track.iterrows():
                    
                    # Calculate angle from current position to ball landing spot
                    ideal_angle = calculate_target_angle(row['x'], row['y'], ball_land_x, ball_land_y)
                    actual_angle = row['move_angle']
                    
                    diff = get_angle_diff(ideal_angle, actual_angle)
                    
                    # Identify first frame where movement direction commits toward landing spot
                    if diff < ALIGNMENT_THRESHOLD_DEG and row['s'] > 2.0:
                        reaction_frames = row['frame_id'] - d_track['frame_id'].min()
                        reaction_time = reaction_frames * 0.1
                        break
                
                # --- METRIC 2: CLOSING PCT ---
                yards_closed = dist_start - dist_end
                closing_pct = max((yards_closed / dist_start), 0)
                
                def_info = play_in[play_in['nfl_id'] == safety_id].iloc[0]
                player_stats.append({
                    'name': def_info['player_name'],
                    'position': def_info['player_position'],
                    'reaction_time': reaction_time,
                    'closing_pct': closing_pct,
                    'plays': 1
                })

    # --- AGGREGATION & RANKING ---
    df = pd.DataFrame(player_stats)
    if df.empty: return

    # 1. Group by Player
    leaderboard = df.groupby(['name', 'position']).agg(
        total_plays=('plays', 'count'),
        avg_reaction=('reaction_time', 'mean'),
        avg_closing_pct=('closing_pct', 'mean')
    ).reset_index()

    # 2. Filter for Sample Size (e.g., at least 100 relevant plays)
    leaderboard = leaderboard[leaderboard['total_plays'] >= MIN_PLAYS]

    # 3. CALCULATE RANKS (The "Composite Score")
    # Rank 1 = Best. 
    # Reaction: Ascending (Lower is better)
    # Percentage of Yards Closed: Descending (Higher is better)
    
    leaderboard['rank_reaction'] = leaderboard['avg_reaction'].rank(ascending=True)
    leaderboard['rank_closing'] = leaderboard['avg_closing_pct'].rank(ascending=False)
    
    # 4. Sum the Ranks (Lower score = Better Overall)
    leaderboard['composite_score'] = (leaderboard['rank_reaction'] + 
                                      leaderboard['rank_closing'])

    # 5. Sort by Composite Score
    leaderboard = leaderboard.sort_values('composite_score', ascending=True).reset_index(drop=True)
    
    # Add an explicit "Overall Rank" column
    leaderboard.index += 1
    leaderboard.index.name = 'Rank'
    
    # --- PRINT THE OFFICIAL LEADERBOARD ---
    
    print("\n" + "="*100)
    print("OFFICIAL SAFETY INSTINCT LEADERBOARD (Weeks analyzed: {})".format(weeks))
    print("Scoring: Sum of ranks across Reaction and Closing Ability.")
    print("Goal: Lower Composite Score is better (Consistency across both).")
    print("="*100)
    
    # Select columns to display
    display_cols = ['name', 'position', 'total_plays', 
                    'avg_reaction', 'rank_reaction', 
                    'avg_closing_pct', 'rank_closing', 
                    'composite_score']
    
    print(leaderboard[display_cols].head(10).to_string())


analyze_safety_range(weeks=WEEKS)

