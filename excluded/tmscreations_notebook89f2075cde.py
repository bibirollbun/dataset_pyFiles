# standard stuff
import pandas as pd
import numpy as np
import glob
import math
import plotly.express as px
import plotly.graph_objects as go
import warnings

# mute the annoying pandas setting warnings
warnings.filterwarnings('ignore')

# set plot style
pd.options.display.max_columns = 50


# Point to the data
base_dir = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/'
train_dir = base_dir + 'train/'

# Config: Change this to ['w*'] if you want to run ALL data (takes longer)
# keeping it simple with week 1 for now to make sure it finishes fast
WEEKS_TO_LOAD = ['w01'] 

def load_data(weeks):
    dfs_in = []
    dfs_out = []
    
    for w in weeks:
        # find the specific week files
        i_files = glob.glob(f"{train_dir}input_2023_{w}*.csv")
        o_files = glob.glob(f"{train_dir}output_2023_{w}*.csv")
        
        if len(i_files) > 0:
            print(f"Reading {i_files[0]}...")
            dfs_in.append(pd.read_csv(i_files[0]))
            dfs_out.append(pd.read_csv(o_files[0]))
            
    return pd.concat(dfs_in), pd.concat(dfs_out)

# Grab the tracking data
input_df, output_df = load_data(WEEKS_TO_LOAD)

# Grab the context (play results, yards gained, etc)
print("Grabbing play metadata...")
plays_df = pd.read_csv(base_dir + 'supplementary_data.csv')

print("Done. Ready to crunch numbers.")


# --- 1. PREP WORK ---
# We need the "Snap" state (Input) and the "Reaction" state (Output)

# Just get the last frame before the throw (the 'input' file state)
# We only care about defenders (coverage guys)
snap_state = input_df.groupby(['game_id', 'play_id', 'nfl_id']).tail(1).copy()
defenders = snap_state[snap_state['player_role'] == 'Defensive Coverage'].copy()

# Filter down columns to what we actually need
# Note: Output files don't have 'o' or 'dir', so we only grab them from input for reference
defenders = defenders[['game_id', 'play_id', 'nfl_id', 'player_name', 'player_position', 
                       'ball_land_x', 'ball_land_y', 'x', 'y']]
defenders.rename(columns={'x':'snap_x', 'y':'snap_y'}, inplace=True)

# Merge in pass info (length, yards gained) so we can weigh the risk later
play_context = plays_df[['game_id', 'play_id', 'pass_length', 'yards_gained', 'pass_result']].dropna(subset=['pass_length'])
defenders = defenders.merge(play_context, on=['game_id', 'play_id'])

# --- 2. THE REACTION WINDOW ---
# Look at the first 1.0 second (10 frames) after the ball is thrown
# This is where the decision happens.
reaction_window = output_df[output_df['frame_id'] <= 11].copy()

# Join with our defender details
# This creates a row for every frame of every defender
analysis_df = reaction_window.merge(defenders, on=['game_id', 'play_id', 'nfl_id'])

# --- 3. MATH TIME ---

# A. REACTION TAX (Angle Error)
# Calculate vector from Snap -> Current Frame
analysis_df['dx_actual'] = analysis_df['x'] - analysis_df['snap_x']
analysis_df['dy_actual'] = analysis_df['y'] - analysis_df['snap_y']

# Calculate vector from Snap -> Ball Landing Spot (The Perfect Line)
analysis_df['dx_ideal'] = analysis_df['ball_land_x'] - analysis_df['snap_x']
analysis_df['dy_ideal'] = analysis_df['ball_land_y'] - analysis_df['snap_y']

# Get angles (atan2 handles the quadrants correctly)
# Convert to degrees because radians are hard to read
analysis_df['angle_actual'] = np.degrees(np.arctan2(analysis_df['dy_actual'], analysis_df['dx_actual']))
analysis_df['angle_ideal'] = np.degrees(np.arctan2(analysis_df['dy_ideal'], analysis_df['dx_ideal']))

# Diff
analysis_df['angle_error'] = abs(analysis_df['angle_actual'] - analysis_df['angle_ideal'])
# Fix the circle wrap (359 vs 1 is 2 degrees diff, not 358)
analysis_df['angle_error'] = analysis_df['angle_error'].apply(lambda x: 360 - x if x > 180 else x)

# B. GHOST MODEL (Counterfactual)
# If they ran 9.0 yds/s straight at the ball, where would they be?
MAX_SPEED = 9.0 
# Time since snap (frames are 0.1s)
analysis_df['time_sec'] = (analysis_df['frame_id'] - 1) * 0.1
analysis_df['ghost_dist'] = MAX_SPEED * analysis_df['time_sec']

# Total distance they needed to cover
analysis_df['total_dist_needed'] = np.sqrt(analysis_df['dx_ideal']**2 + analysis_df['dy_ideal']**2)
# Avoid division by zero
analysis_df['total_dist_needed'] = analysis_df['total_dist_needed'].replace(0, 0.001)

analysis_df['ghost_progress'] = analysis_df['ghost_dist'] / analysis_df['total_dist_needed']
# Cap it at 1.0 (can't go past the ball)
analysis_df.loc[analysis_df['ghost_progress'] > 1, 'ghost_progress'] = 1

# Ghost XY coordinates (Lerp)
analysis_df['ghost_x'] = analysis_df['snap_x'] + (analysis_df['dx_ideal'] * analysis_df['ghost_progress'])
analysis_df['ghost_y'] = analysis_df['snap_y'] + (analysis_df['dy_ideal'] * analysis_df['ghost_progress'])

# How far behind the ghost are they?
analysis_df['lost_yards'] = np.sqrt((analysis_df['x'] - analysis_df['ghost_x'])**2 + (analysis_df['y'] - analysis_df['ghost_y'])**2)

print("Calculations done. We have the metrics.")


# --- 1. PREP WORK ---
# We need the "Snap" state (Input) and the "Reaction" state (Output)

# Just get the last frame before the throw (the 'input' file state)
# We only care about defenders (coverage guys)
snap_state = input_df.groupby(['game_id', 'play_id', 'nfl_id']).tail(1).copy()
defenders = snap_state[snap_state['player_role'] == 'Defensive Coverage'].copy()

# Filter down columns to what we actually need
# Note: Output files don't have 'o' or 'dir', so we only grab them from input for reference
defenders = defenders[['game_id', 'play_id', 'nfl_id', 'player_name', 'player_position', 
                       'ball_land_x', 'ball_land_y', 'x', 'y']]
defenders.rename(columns={'x':'snap_x', 'y':'snap_y'}, inplace=True)

# Merge in pass info (length, yards gained) so we can weigh the risk later
play_context = plays_df[['game_id', 'play_id', 'pass_length', 'yards_gained', 'pass_result']].dropna(subset=['pass_length'])
defenders = defenders.merge(play_context, on=['game_id', 'play_id'])

# --- 2. THE REACTION WINDOW ---
# Look at the first 1.0 second (10 frames) after the ball is thrown
# This is where the decision happens.
reaction_window = output_df[output_df['frame_id'] <= 11].copy()

# Join with our defender details
# This creates a row for every frame of every defender
analysis_df = reaction_window.merge(defenders, on=['game_id', 'play_id', 'nfl_id'])

# --- 3. MATH TIME ---

# A. REACTION TAX (Angle Error)
# Calculate vector from Snap -> Current Frame
analysis_df['dx_actual'] = analysis_df['x'] - analysis_df['snap_x']
analysis_df['dy_actual'] = analysis_df['y'] - analysis_df['snap_y']

# Calculate vector from Snap -> Ball Landing Spot (The Perfect Line)
analysis_df['dx_ideal'] = analysis_df['ball_land_x'] - analysis_df['snap_x']
analysis_df['dy_ideal'] = analysis_df['ball_land_y'] - analysis_df['snap_y']

# Get angles (atan2 handles the quadrants correctly)
# Convert to degrees because radians are hard to read
analysis_df['angle_actual'] = np.degrees(np.arctan2(analysis_df['dy_actual'], analysis_df['dx_actual']))
analysis_df['angle_ideal'] = np.degrees(np.arctan2(analysis_df['dy_ideal'], analysis_df['dx_ideal']))

# Diff
analysis_df['angle_error'] = abs(analysis_df['angle_actual'] - analysis_df['angle_ideal'])
# Fix the circle wrap (359 vs 1 is 2 degrees diff, not 358)
analysis_df['angle_error'] = analysis_df['angle_error'].apply(lambda x: 360 - x if x > 180 else x)

# B. GHOST MODEL (Counterfactual)
# If they ran 9.0 yds/s straight at the ball, where would they be?
MAX_SPEED = 9.0 
# Time since snap (frames are 0.1s)
analysis_df['time_sec'] = (analysis_df['frame_id'] - 1) * 0.1
analysis_df['ghost_dist'] = MAX_SPEED * analysis_df['time_sec']

# Total distance they needed to cover
analysis_df['total_dist_needed'] = np.sqrt(analysis_df['dx_ideal']**2 + analysis_df['dy_ideal']**2)
# Avoid division by zero
analysis_df['total_dist_needed'] = analysis_df['total_dist_needed'].replace(0, 0.001)

analysis_df['ghost_progress'] = analysis_df['ghost_dist'] / analysis_df['total_dist_needed']
# Cap it at 1.0 (can't go past the ball)
analysis_df.loc[analysis_df['ghost_progress'] > 1, 'ghost_progress'] = 1

# Ghost XY coordinates (Lerp)
analysis_df['ghost_x'] = analysis_df['snap_x'] + (analysis_df['dx_ideal'] * analysis_df['ghost_progress'])
analysis_df['ghost_y'] = analysis_df['snap_y'] + (analysis_df['dy_ideal'] * analysis_df['ghost_progress'])

# How far behind the ghost are they?
analysis_df['lost_yards'] = np.sqrt((analysis_df['x'] - analysis_df['ghost_x'])**2 + (analysis_df['y'] - analysis_df['ghost_y'])**2)

print("Calculations done. We have the metrics.")


# Focus on the status at Frame 11 (1 full second after throw)
final_frames = analysis_df[analysis_df['frame_id'] == 11].copy()

# Add the "Severity Multiplier" (Risk Weighted Score)
# Mistake on a 50 yard bomb is 5x worse than on a 10 yard slant
final_frames['risk_score'] = final_frames['angle_error'] * (final_frames['pass_length'] / 10)

# Aggregate per player
# Filter: Must have moved at least 0.5 yards (ignore guys standing still)
active_plays = final_frames[final_frames['lost_yards'] > 0.5]

leaderboard = active_plays.groupby(['nfl_id', 'player_name', 'player_position']).agg(
    Avg_Reaction_Tax=('angle_error', 'mean'),
    Avg_Risk_Score=('risk_score', 'mean'),
    Avg_Lost_Yards=('lost_yards', 'mean'),
    Plays=('game_id', 'count')
).reset_index()

# Cutoff: Only rank players with enough sample size (5+ plays in this dataset)
ranked = leaderboard[leaderboard['Plays'] >= 5].sort_values('Avg_Risk_Score')

print("--- TOP 5 ELITE COVERAGE (Best Reaction) ---")
print(ranked.head(5)[['player_name', 'player_position', 'Avg_Reaction_Tax', 'Avg_Risk_Score']])

print("\n--- BOTTOM 5 LIABILITY (Worst Reaction) ---")
print(ranked.tail(5)[['player_name', 'player_position', 'Avg_Reaction_Tax', 'Avg_Risk_Score']])


#VISUALIZATION (THE GHOST ON THE FIELD) ---
import plotly.graph_objects as go

# Function to draw the NFL Field background
def create_football_field(fig):
    # Field dimensions (120 yards long, 53.3 yards wide)
    fig.update_layout(
        xaxis=dict(range=[0, 120], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 53.3], showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='forestgreen',
        width=800, height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Yard lines (every 10 yards)
    for x in range(10, 111, 10):
        fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=53.3,
                      line=dict(color="white", width=2, dash="solid"))
        
        # Yard numbers (10, 20, 30...)
        if x > 10 and x < 110:
            num = x - 10 if x <= 60 else 110 - x
            # Bottom numbers
            fig.add_annotation(x=x, y=5, text=str(num), showarrow=False, 
                               font=dict(color="white", size=20))
            # Top numbers (rotated 180 degrees) - THIS IS THE FIX
            fig.add_annotation(x=x, y=48.3, text=str(num), showarrow=False, 
                               textangle=180, font=dict(color="white", size=20))

    # Endzones (0-10 and 110-120)
    fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=53.3, fillcolor="darkgreen", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=110, y0=0, x1=120, y1=53.3, fillcolor="darkgreen", layer="below", line_width=0)
    
    return fig

# --- GENERATE THE PLOT ---
# Find the play where a player lost the most yards to the "Ghost"
bad_play = analysis_df.sort_values('lost_yards', ascending=False).iloc[0]

# Get the tracking data for that specific play and player
track = analysis_df[(analysis_df['game_id']==bad_play['game_id']) & 
                    (analysis_df['play_id']==bad_play['play_id']) & 
                    (analysis_df['nfl_id']==bad_play['nfl_id'])]

fig_ghost = go.Figure()

# 1. Apply Field
fig_ghost = create_football_field(fig_ghost)

# 2. Add Paths
# Real Path (Yellow to stand out on green)
fig_ghost.add_trace(go.Scatter(x=track['x'], y=track['y'], mode='lines+markers', 
                               name=f"{bad_play['player_name']} (Real)", 
                               line=dict(color='yellow', width=4))) 

# Ghost Path (White Dotted)
fig_ghost.add_trace(go.Scatter(x=track['ghost_x'], y=track['ghost_y'], mode='lines', 
                               name='Perfect Ghost', 
                               line=dict(color='white', dash='dot', width=3))) 

# 3. Add Target (Catch Point)
fig_ghost.add_trace(go.Scatter(x=[bad_play['ball_land_x']], y=[bad_play['ball_land_y']], 
                               mode='markers', name='Catch Point', 
                               marker=dict(size=15, color='red', symbol='x')))

fig_ghost.update_layout(title=f"The Cost of Hesitation: {bad_play['player_name']} vs. The Ghost")
fig_ghost.show()


# Validation: Does bad reaction = more yards allowed?
# Check specific plays where the pass was completed
completed = analysis_df[(analysis_df['pass_result']=='C') & (analysis_df['frame_id']==11)].copy()

# Remove garbage data (0 yards usually means error or touchback)
completed = completed[completed['yards_gained'] > 0]

# Bin the Reaction Tax into groups to make the chart readable
# We separate "Elite" reaction (<15 degrees error) from "Terrible" (>90 degrees)
completed['Reaction_Quality'] = pd.qcut(completed['angle_error'], q=4, labels=["Elite (<15 deg)", "Good", "Bad", "Terrible (>90 deg)"])

# Group and see avg yards gained for each bucket
proof = completed.groupby('Reaction_Quality')['yards_gained'].mean().reset_index()

# Plot
fig_proof = px.bar(proof, x='Reaction_Quality', y='yards_gained',
                   title="Validation: Bad Angles = More Yards Allowed",
                   labels={'yards_gained':'Avg Yards Allowed', 'Reaction_Quality':'Defender Reaction'},
                   color='yards_gained', color_continuous_scale='Reds')
fig_proof.show()

print("Check the chart: If the bars get taller to the right, hypothesis is proven.")

