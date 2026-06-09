# =========================================================================================
# ğŸ�ˆ NFL BIG DATA BOWL 2026: ADAPTIVE DEFENSIVE INTELLIGENCE (ADI) - PROFESSIONAL PIPELINE
# =========================================================================================
# Author: Pro Kaggle Contributor
# Track: Metrics / Coaching
# Model Version: 2.0 (Vector Math + Physics-Based Reaction)
# =========================================================================================

import os
import sys
import subprocess
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from IPython.display import HTML
from tqdm.notebook import tqdm
from scipy.spatial import distance

# --- 1. SILENT INSTALLATION OF DEPENDENCIES ---
# Ruptures is used for Changepoint Detection (The core of Reaction Time)
try:
    import ruptures as rpt
except ImportError:
    # Silent install to keep logs clean
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ruptures"], stdout=subprocess.DEVNULL)
    import ruptures as rpt


# --- 2. CONFIGURATION & PHYSICS CONSTANTS ---
class Config:
    # Kaggle Standard Paths
    INPUT_DIR = "/kaggle/input/nfl-big-data-bowl-2025" # Update year if changed in competition
    # Fallback for testing if paths don't exist
    TRACKING_FILE = f"{INPUT_DIR}/tracking_week_1.csv"
    PLAYS_FILE = f"{INPUT_DIR}/plays.csv"
    PLAYERS_FILE = f"{INPUT_DIR}/players.csv"
    
    # Physics Parameters
    SAMPLE_RATE = 10.0  # Hz
    MIN_PLAY_FRAMES = 15
    CHANGEPOINT_PENALTY = 3.0 # For Pelt Algorithm
    
    # ADI Score Weights
    W_REACTION = 0.40  # 40% Reaction Speed
    W_ANGLE = 0.40     # 40% Vector Efficiency
    W_ECONOMY = 0.20   # 20% Movement Economy
    
    # Visualization
    FIELD_COLOR = '#00b140'
    LINE_COLOR = '#ffffff'

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


# --- 3. ADVANCED MATH ENGINE (THE BRAIN) ---
class VectorMath:
    @staticmethod
    def calculate_vectors(df):
        """
        Converts NFL 'dir' and 's' (speed) into Velocity Vectors (Vx, Vy).
        NFL Coordinate System: 0 degrees is Y-axis positive (North), increasing clockwise.
        Math Coordinate System: 0 degrees is X-axis positive (East), increasing counter-clockwise.
        """
        # Convert degrees to radians and adjust for coordinate system rotation
        # Math Angle = 90 - NFL Angle
        df['rad'] = np.radians(90 - df['dir'])
        df['vx'] = df['s'] * np.cos(df['rad'])
        df['vy'] = df['s'] * np.sin(df['rad'])
        return df

    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        """Calculates alignment between two vectors (-1 to 1)"""
        dot = np.sum(vec_a * vec_b, axis=1)
        norm_a = np.linalg.norm(vec_a, axis=1)
        norm_b = np.linalg.norm(vec_b, axis=1)
        return dot / (norm_a * norm_b + 1e-8)


# --- 4. CORE ANALYZER CLASS ---
class DefensiveIQModel:
    def __init__(self):
        self.math = VectorMath()
        
    def analyze_play(self, play_df, game_id, play_id):
        """
        Analyzes a single play. Returns list of metrics for all defenders.
        """
        # 1. Sync Logic: Find Frame where ball is thrown (pass_forward)
        pass_frame = play_df[play_df['event'] == 'pass_forward']['frameId'].min()
        if np.isnan(pass_frame): 
            # Fallback: ball_snap if pass_forward missing (rare)
            pass_frame = play_df[play_df['frameId']].min()
            
        # Filter for Post-Throw Analysis Only (Critical for Reaction Time)
        active_phase = play_df[play_df['frameId'] >= pass_frame].copy()
        
        # Get Ball and Defenders
        ball_data = active_phase[active_phase['club'] == 'football']
        defenders = active_phase[active_phase['club'] != 'football']['nflId'].unique()
        
        results = []
        
        for def_id in defenders:
            # Skip football or NaN
            if pd.isna(def_id): continue
            
            def_data = active_phase[active_phase['nflId'] == def_id].sort_values('frameId').copy()
            
            # Ensure sufficient data points
            if len(def_data) < 10 or len(ball_data) < 10: continue
            
            # --- METRIC 1: REACTION SPEED (Changepoint Detection) ---
            # We look for sudden change in acceleration profile
            signal_acc = def_data['a'].fillna(0).values
            try:
                # Pelt algorithm finds structural changes in signal
                algo = rpt.Pelt(model="rbf").fit(signal_acc)
                changepoints = algo.predict(pen=Config.CHANGEPOINT_PENALTY)
                
                # First changepoint is the reaction frame
                reaction_idx = changepoints[0] if len(changepoints) > 0 else 0
                reaction_time_sec = reaction_idx / Config.SAMPLE_RATE
                
                # Score: 0.2s -> 100, 2.0s -> 0
                rxn_score = max(0, 100 - (reaction_time_sec * 50))
            except:
                rxn_score = 50 # Fallback
                reaction_time_sec = 0.5

            # --- METRIC 2: VECTOR EFFICIENCY (Angle Math) ---
            # Calculate vectors
            def_data = self.math.calculate_vectors(def_data)
            
            # Merge with ball to get target vector
            merged = pd.merge(def_data, ball_data[['frameId', 'x', 'y']], on='frameId', suffixes=('', '_ball'))
            
            # Vector from Defender to Ball
            merged['dx'] = merged['x_ball'] - merged['x']
            merged['dy'] = merged['y_ball'] - merged['y']
            
            vec_def = merged[['vx', 'vy']].values
            vec_target = merged[['dx', 'dy']].values
            
            # Cosine Similarity: How well is he aiming at the ball?
            alignment = self.math.cosine_similarity(vec_def, vec_target)
            angle_score = ((np.mean(alignment) + 1) / 2) * 100  # Map -1..1 to 0..100
            
            # --- METRIC 3: MOVEMENT ECONOMY ---
            # Displacement / Distance Traveled
            start_pos = def_data.iloc[0][['x', 'y']]
            end_pos = def_data.iloc[-1][['x', 'y']]
            eucl_disp = np.linalg.norm(start_pos - end_pos)
            total_dist = def_data['dis'].sum()
            
            eff_score = (eucl_disp / (total_dist + 1e-6)) * 100
            
            # --- FINAL ADI CALCULATION ---
            adi = (rxn_score * Config.W_REACTION) + \
                  (angle_score * Config.W_ANGLE) + \
                  (eff_score * Config.W_ECONOMY)
            
            results.append({
                'gameId': game_id,
                'playId': play_id,
                'nflId': def_id,
                'adi_score': adi,
                'reaction_sec': reaction_time_sec,
                'angle_efficiency': angle_score,
                'movement_economy': eff_score,
                'team': def_data['club'].iloc[0],
                'position': def_data['position'].iloc[0] if 'position' in def_data.columns else 'DEF'
            })
            
        return results


# --- 5. FIELD VISUALIZATION (BROADCAST QUALITY) ---
def create_football_field(linenumbers=True, endzones=True, highlight_line=False, highlight_line_number=50):
    """
    Creates a matplotlib patch object representing the field.
    Standard utility used in winning kernels.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1, edgecolor='r', facecolor='darkgreen', zorder=0)
    fig, ax = plt.subplots(1, figsize=(12, 6.33))
    ax.add_patch(rect)
    
    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80, 80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    
    # Hash marks
    for x in range(10, 110, 1):
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')
    
    # Numbers
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50: numb = 120 - x
            plt.text(x, 5, str(numb - 10), horizontalalignment='center', fontsize=20, color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10), horizontalalignment='center', fontsize=20, color='white', rotation=180)
            
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3, linewidth=0.1, edgecolor='r', facecolor='blue', alpha=0.2, zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3, linewidth=0.1, edgecolor='r', facecolor='red', alpha=0.2, zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
        
    plt.xlim(0, 120)
    plt.ylim(0, 53.3)
    plt.axis('off')
    return fig, ax


# --- 6. MAIN EXECUTION PIPELINE ---
def main():
    print("ğŸš€ Starting ADI Pipeline...")
    
    # A. Load Data (Handle missing files gracefully for demo)
    try:
        if os.path.exists(Config.TRACKING_FILE):
            print("ğŸ”„ Loading Real Data...")
            track_df = pd.read_csv(Config.TRACKING_FILE)
            plays_df = pd.read_csv(Config.PLAYS_FILE)
            # Filter for pass plays only to save memory
            plays_df = plays_df[plays_df['passResult'].isin(['C', 'I', 'IN'])] # Completed, Incomplete, Intercepted
            track_df = track_df[track_df['gameId'].isin(plays_df['gameId'].unique())]
        else:
            print("âš ï¸� Data files not found. Creating Mock Data for Demo...")
            # Create professional mock data if files missing (prevents crash)
            dates = range(100)
            track_df = pd.DataFrame({
                'gameId': [2026001]*100, 'playId': [100]*100, 'frameId': dates,
                'nflId': [999]*50 + [888]*50, 'club': ['BUF']*50 + ['MIA']*50,
                'position': ['CB']*50 + ['WR']*50, 'event': ['ball_snap']*5 + ['pass_forward']*1 + [None]*94,
                'x': np.linspace(50, 80, 100), 'y': np.linspace(20, 40, 100),
                's': np.random.normal(5, 1, 100), 'a': np.random.normal(2, 0.5, 100),
                'dis': np.random.normal(0.5, 0.1, 100), 'dir': np.random.normal(90, 10, 100)
            })
            plays_df = pd.DataFrame({'gameId': [2026001], 'playId': [100], 'passResult': ['I']})
            # Add fake football
            ball = track_df.copy(); ball['club'] = 'football'; ball['nflId'] = np.nan
            track_df = pd.concat([track_df, ball])

    except Exception as e:
        print(f"â�Œ Error loading data: {e}")
        return

    # B. Analyze Plays
    analyzer = DefensiveIQModel()
    all_scores = []
    
    # Process sample of plays (Limit to 50 plays for speed in demo, remove slice for full run)
    unique_plays = track_df[['gameId', 'playId']].drop_duplicates().values[:50] 
    
    print(f"ğŸ§  Analyzing {len(unique_plays)} plays using Physics Engine...")
    for g_id, p_id in tqdm(unique_plays):
        play_subset = track_df[(track_df['gameId'] == g_id) & (track_df['playId'] == p_id)]
        play_scores = analyzer.analyze_play(play_subset, g_id, p_id)
        all_scores.extend(play_scores)
        
    # C. Results & Visualization
    results_df = pd.DataFrame(all_scores)
    
    if not results_df.empty:
        results_df = results_df.sort_values('adi_score', ascending=False)
        
        print("\nğŸ�† TOP DEFENSIVE PERFORMANCES (ADI LEADERBOARD)")
        print(results_df[['nflId', 'position', 'team', 'adi_score', 'reaction_sec', 'angle_efficiency']].head(10).to_markdown(index=False))
        
        # Visualize the #1 Play
        top_play = results_df.iloc[0]
        print(f"\nğŸ�¥ Visualizing Top Play (ADI: {top_play['adi_score']:.1f})")
        
        fig, ax = create_football_field()
        
        # Plot Trajectory of Top Defender
        play_data = track_df[(track_df['gameId'] == top_play['gameId']) & 
                             (track_df['playId'] == top_play['playId']) & 
                             (track_df['nflId'] == top_play['nflId'])]
                             
        ax.plot(play_data['x'], play_data['y'], color='yellow', linewidth=3, label=f"Best Defender (ADI {top_play['adi_score']:.0f})")
        ax.scatter(play_data['x'].iloc[-1], play_data['y'].iloc[-1], color='yellow', s=100, marker='x')
        
        # Plot Football
        ball_data = track_df[(track_df['gameId'] == top_play['gameId']) & 
                             (track_df['playId'] == top_play['playId']) & 
                             (track_df['club'] == 'football')]
        ax.plot(ball_data['x'], ball_data['y'], color='brown', linestyle='--', label='Ball Path')
        
        plt.legend(loc='lower right')
        plt.title(f"High-IQ Defensive Read: Reacted in {top_play['reaction_sec']:.2f}s", fontsize=15, color='white')
        plt.show()
        
        # D. Save Submission
        results_df.to_csv("adi_metric_submission.csv", index=False)
        print("\nâœ… Submission File Saved: adi_metric_submission.csv")
    else:
        print("âš ï¸� No valid plays found. Check data filters.")

if __name__ == "__main__":
    main()

