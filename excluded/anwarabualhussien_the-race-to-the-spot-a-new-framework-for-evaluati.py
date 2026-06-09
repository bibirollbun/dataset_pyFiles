# --- Preliminaries: Import Libraries and Set Options ---
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization and pandas display options
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

print("Libraries imported and options set.")


# --- Step 1: Load the Required Data Files ---
print("\n" + "="*50)
print("STEP 1: LOADING DATA")
print("="*50)

# Define the TWO directories based on the confirmed paths
BASE_DIR = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/'
TRAIN_DIR = os.path.join(BASE_DIR, 'train/')

print(f"Base Directory: {BASE_DIR}")
print(f"Train Directory: {TRAIN_DIR}")

try:
    # Load supplementary data from the BASE directory
    print("\nLoading supplementary_data.csv (the 'Context' file)...")
    # A DtypeWarning is expected here, which is fine for our current analysis. We will ignore it.
    supp_df = pd.read_csv(os.path.join(BASE_DIR, 'supplementary_data.csv'))
    print(f" -> Loaded successfully. Shape: {supp_df.shape}")

    # Load weekly tracking data from the TRAIN directory
    print("\nLoading input_2023_w01.csv (the 'Movement' file)...")
    tracking_w1_df = pd.read_csv(os.path.join(TRAIN_DIR, 'input_2023_w01.csv'))
    print(f" -> Loaded successfully. Shape: {tracking_w1_df.shape}")
    
except FileNotFoundError as e:
    print(f"\nERROR: A file was not found. Please double-check your Kaggle input directory.")
    print(f"Specific error: {e}")
    # Create empty dataframes to prevent the script from crashing
    supp_df = pd.DataFrame()
    tracking_w1_df = pd.DataFrame()

    
# --- Step 2: Merge DataFrames for a Unified View ---
print("\n" + "="*50)
print("STEP 2: MERGING 'MOVEMENT' AND 'CONTEXT' DATAFRAMES")
print("="*50)

# Ensure the required dataframes were loaded before trying to merge
if not tracking_w1_df.empty and not supp_df.empty:
    
    print(f"Merging the {tracking_w1_df.shape[0]} tracking rows with context data...")

    # Merge the tracking data with the supplementary play context data
    # We use a left merge to ensure we keep every single frame of tracking data.
    merged_df = pd.merge(
        tracking_w1_df,
        supp_df,
        on=['game_id', 'play_id'],
        how='left'
    )
    
    print(f"\nMerge complete. Shape of new unified DataFrame: {merged_df.shape}")

    # --- Final Inspection of the Merged DataFrame ---
    print("\n" + "="*50)
    print("INSPECTING THE FINAL MERGED DATAFRAME")
    print("="*50)
    
    print("\nFirst 5 rows:")
    display(merged_df.head())
    
    print("\nVerifying the merge...")
    null_pass_results = merged_df['pass_result'].isnull().sum()
    print(f"Number of rows with null 'pass_result' after merge: {null_pass_results}")

    if null_pass_results == 0:
        print(" -> Verification successful! All tracking rows were matched with a play outcome.")
    else:
        print(" -> Warning: Some tracking rows could not be matched with a play outcome.")
        
    print("\nFinal info of the merged DataFrame:")
    merged_df.info()

else:
    print("\nERROR: One or more required DataFrames were not loaded correctly in Step 1. Cannot perform merge.")


# ===================================================================
# STEP 2: FEATURE ENGINEERING - DISTANCE TO BALL LANDING SPOT
# ===================================================================

print("Starting Step 2: Feature Engineering...")

if 'merged_df' in locals() and not merged_df.empty:
    
    # --- Calculate Euclidean Distance ---
    # distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
    print("Calculating distance from each player to the ball's landing spot for every frame...")
    
    merged_df['dist_to_land_spot'] = np.sqrt(
        (merged_df['x'] - merged_df['ball_land_x'])**2 +
        (merged_df['y'] - merged_df['ball_land_y'])**2
    )
    
    print(" -> 'dist_to_land_spot' column created successfully.")

    # --- Inspect the new feature ---
    print("\n--- Inspection of the New Feature ---")
    print("Showing relevant columns for the first 5 rows to verify the calculation:")
    display(merged_df[['frame_id', 'player_name', 'x', 'y', 'ball_land_x', 'ball_land_y', 'dist_to_land_spot']].head())

else:
    print("ERROR: The 'merged_df' DataFrame was not found. Please run Step 1 first.")


# ===================================================================
# STEP 3: VISUALIZING THE NEW FEATURE
# ===================================================================

print("Starting Step 3: Visualizing 'dist_to_land_spot' for a single play...")

if 'merged_df' in locals() and not merged_df.empty:

    # --- Select an interesting play to visualize ---
    # We will find a completed pass with a decent pass length to make the visualization clear.
    completed_passes = merged_df[merged_df['pass_result'] == 'C'].copy()
    
    # Sort by pass length and pick a good example (you can change the index [0] to see other plays)
    example_play_df = completed_passes[completed_passes['pass_length'] > 20].sort_values(by='pass_length', ascending=False)
    
    if not example_play_df.empty:
        example_play = example_play_df.iloc[0]
        example_game_id = example_play['game_id']
        example_play_id = example_play['play_id']

        print(f"\nVisualizing a completed pass: Game ID {example_game_id}, Play ID {example_play_id}")

        # Isolate all data for this single play
        play_df = merged_df[(merged_df['game_id'] == example_game_id) & (merged_df['play_id'] == example_play_id)].copy()

        # --- Identify Key Players ---
        # We'll use a proxy to find the targeted receiver: the offensive player who gets closest to the landing spot.
        offense_players = play_df[play_df['player_side'] == 'Offense']
        # .loc is used to get the entire row where the minimum distance occurs for each player
        min_dist_rows_off = offense_players.loc[offense_players.groupby('nfl_id')['dist_to_land_spot'].idxmin()]
        targeted_receiver_row = min_dist_rows_off.sort_values('dist_to_land_spot').iloc[0]
        targeted_receiver_id = targeted_receiver_row['nfl_id']
        targeted_receiver_name = targeted_receiver_row['player_name']
        print(f" -> Identified Targeted Receiver (proxy): {targeted_receiver_name}")

        # Now, find the closest defender in a similar way
        defense_players = play_df[play_df['player_side'] == 'Defense']
        min_dist_rows_def = defense_players.loc[defense_players.groupby('nfl_id')['dist_to_land_spot'].idxmin()]
        closest_defender_row = min_dist_rows_def.sort_values('dist_to_land_spot').iloc[0]
        closest_defender_id = closest_defender_row['nfl_id']
        closest_defender_name = closest_defender_row['player_name']
        print(f" -> Identified Closest Defender: {closest_defender_name}")
        
        # --- Create the Plot ---
        plt.figure(figsize=(16, 9))
        
        # Plot for the targeted receiver
        receiver_data = play_df[play_df['nfl_id'] == targeted_receiver_id]
        sns.lineplot(x='frame_id', y='dist_to_land_spot', data=receiver_data, label=f'Receiver: {targeted_receiver_name}', lw=3.5, color='dodgerblue')

        # Plot for the closest defender
        defender_data = play_df[play_df['nfl_id'] == closest_defender_id]
        sns.lineplot(x='frame_id', y='dist_to_land_spot', data=defender_data, label=f'Defender: {closest_defender_name}', lw=3.5, linestyle='--', color='crimson')

        plt.title(f'The Race to the Spot\nPlay ID: {example_play_id}', fontsize=20, fontweight='bold')
        plt.xlabel('Frame ID (Time)', fontsize=14)
        plt.ylabel('Distance to Ball Landing Spot (Yards)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()
    
    else:
        print("Could not find a suitable example play to visualize.")

else:
    print("ERROR: The 'merged_df' DataFrame was not found. Please run the previous steps.")


# ===================================================================
# STEP 4: CALCULATING METRICS FOR ALL PLAYS
# ===================================================================

from tqdm import tqdm
# tqdm is a great library for showing progress bars on long operations.
tqdm.pandas()

print("Starting Step 4: Calculating metrics for all Week 1 plays...")

def calculate_play_metrics(play_df):
    """
    This function takes the DataFrame for a single play and calculates
    our custom metrics: Separation at Catch and Target Convergence Rate.
    """
    # --- 1. Identify Targeted Receiver ---
    offense_players = play_df[play_df['player_side'] == 'Offense']
    if offense_players.empty:
        return None
    
    # Find the row where each offensive player was closest to the landing spot
    min_dist_rows_off = offense_players.loc[offense_players.groupby('nfl_id')['dist_to_land_spot'].idxmin()]
    # The targeted receiver is the one who got the closest overall
    targeted_receiver_row = min_dist_rows_off.sort_values('dist_to_land_spot').iloc[0]
    
    receiver_id = targeted_receiver_row['nfl_id']
    receiver_name = targeted_receiver_row['player_name']
    
    # This is our proxy for the moment of the catch
    catch_frame = targeted_receiver_row['frame_id']
    receiver_dist_at_catch = targeted_receiver_row['dist_to_land_spot']

    # --- 2. Calculate Separation at Catch ---
    defense_players = play_df[play_df['player_side'] == 'Defense']
    if defense_players.empty:
        return None # No defenders on the play? Skip.
        
    # Get the state of all defenders AT THE MOMENT OF THE CATCH
    defenders_at_catch_frame = defense_players[defense_players['frame_id'] == catch_frame]
    if defenders_at_catch_frame.empty:
        # This can happen if players' tracking data ends at different frames.
        # We'll take the closest defender from the last available frame.
        last_frame = defense_players['frame_id'].max()
        defenders_at_catch_frame = defense_players[defense_players['frame_id'] == last_frame]

    # Find the defender closest to the landing spot at that moment
    closest_defender_row = defenders_at_catch_frame.sort_values('dist_to_land_spot').iloc[0]
    defender_dist_at_catch = closest_defender_row['dist_to_land_spot']
    
    separation_at_catch = defender_dist_at_catch - receiver_dist_at_catch
    
    # --- 3. Calculate Target Convergence Rate (TCR) ---
    receiver_play_data = play_df[play_df['nfl_id'] == receiver_id]
    
    # Get the receiver's starting distance at the first frame of the play
    dist_start = receiver_play_data['dist_to_land_spot'].iloc[0]
    frame_start = receiver_play_data['frame_id'].iloc[0]
    
    dist_end = receiver_dist_at_catch
    frame_end = catch_frame
    
    # Avoid division by zero for plays with very few frames
    if (frame_end - frame_start) == 0:
        tcr = 0
    else:
        # Calculate distance closed and time elapsed (10 frames per second)
        distance_closed = dist_start - dist_end
        time_elapsed_sec = (frame_end - frame_start) / 10.0
        tcr = distance_closed / time_elapsed_sec # Yards per Second

    # --- 4. Return results as a dictionary ---
    results = {
        'targeted_receiver': receiver_name,
        'separation_at_catch': separation_at_catch,
        'target_convergence_rate_yps': tcr,
        'pass_result': play_df['pass_result'].iloc[0],
        'pass_length': play_df['pass_length'].iloc[0]
    }
    
    return pd.Series(results)


# --- Apply the function to every play ---
if 'merged_df' in locals() and not merged_df.empty:
    # We only want to analyze completed or incomplete passes
    pass_plays_df = merged_df[merged_df['pass_result'].isin(['C', 'I'])].copy()
    
    print(f"Processing {pass_plays_df.groupby(['game_id', 'play_id']).ngroups} unique pass plays...")
    
    # Group by play and apply our function
    play_metrics_df = pass_plays_df.groupby(['game_id', 'play_id']).progress_apply(calculate_play_metrics)
    
    print("\n--- Metrics Calculation Complete ---")
    print("First 10 rows of our new metrics DataFrame:")
    display(play_metrics_df.head(10))
    
    print("\nBasic statistics for our new metrics:")
    display(play_metrics_df.describe())
    
else:
    print("ERROR: The 'merged_df' DataFrame was not found. Please run the previous steps.")


# ===================================================================
# STEP 5: PLAYER-LEVEL AGGREGATION AND LEADERBOARDS
# ===================================================================

print("Starting Step 5: Aggregating metrics to the player level...")

if 'play_metrics_df' in locals() and not play_metrics_df.empty:

    # --- Group by player and calculate aggregate stats ---
    player_summary = play_metrics_df.groupby('targeted_receiver').agg(
        num_targets=('pass_result', 'count'),
        avg_separation=('separation_at_catch', 'mean'),
        avg_tcr=('target_convergence_rate_yps', 'mean'),
        avg_pass_length=('pass_length', 'mean')
    ).reset_index()

    # --- Create fair leaderboards by filtering for a minimum number of targets ---
    # Let's set a minimum of 5 targets for Week 1 to make the rankings more meaningful
    MIN_TARGETS = 5
    qualified_players = player_summary[player_summary['num_targets'] >= MIN_TARGETS].copy()
    
    print(f"\nAnalyzing {len(qualified_players)} receivers with at least {MIN_TARGETS} targets in Week 1.")
    
    # --- Leaderboard 1: Top Separation Artists ---
    # Who creates the most space?
    top_separation = qualified_players.sort_values(by='avg_separation', ascending=False).head(10)
    
    print("\n" + "="*50)
    print("LEADERBOARD: Top 10 Receivers by Average Separation (Yards)")
    print("="*50)
    display(top_separation[['targeted_receiver', 'avg_separation', 'num_targets']])

    # --- Leaderboard 2: Top "To the Spot" Speedsters ---
    # Who has the best Target Convergence Rate?
    top_tcr = qualified_players.sort_values(by='avg_tcr', ascending=False).head(10)
    
    print("\n" + "="*50)
    print("LEADERBOARD: Top 10 Receivers by Target Convergence Rate (Yds/Sec)")
    print("="*50)
    display(top_tcr[['targeted_receiver', 'avg_tcr', 'num_targets']])
    
    # --- Visualization: Combining the two metrics ---
    print("\n" + "="*50)
    print("VISUALIZATION: Separation vs. TCR for all Qualified Receivers")
    print("="*50)
    
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        data=qualified_players,
        x='avg_tcr',
        y='avg_separation',
        size='num_targets',  # Make the dot size proportional to the number of targets
        hue='avg_pass_length', # Color by average pass length
        palette='viridis',
        sizes=(50, 500)
    )

    # Add labels for a few standout players
    for i, row in qualified_players.iterrows():
        # Annotate players who are in the top right quadrant (good at both)
        if row['avg_tcr'] > 3.5 and row['avg_separation'] > 2.5:
             plt.text(row['avg_tcr'] + 0.05, row['avg_separation'], row['targeted_receiver'], fontsize=10, fontweight='bold')
    
    plt.title('Receiver Performance Matrix (Week 1)', fontsize=20, fontweight='bold')
    plt.xlabel('Target Convergence Rate (Yds/Sec) -> [Faster to the Spot]', fontsize=14)
    plt.ylabel('Average Separation at Catch (Yards) -> [More Open]', fontsize=14)
    plt.axhline(qualified_players['avg_separation'].mean(), ls='--', color='grey')
    plt.axvline(qualified_players['avg_tcr'].mean(), ls='--', color='grey')
    plt.legend(title='Avg Pass Length', loc='upper left')
    plt.grid(True)
    plt.show()


else:
    print("ERROR: The 'play_metrics_df' DataFrame was not found. Please run Step 4 first.")


# ===================================================================
# STEP 6: SCALING THE ANALYSIS TO THE FULL SEASON
# ===================================================================

print("Starting Step 6: Scaling analysis to the full season...")

# We already have the 'calculate_play_metrics' function from Step 4.
# We also have the supplementary data 'supp_df' loaded in memory.

def process_week_data(week_number, supp_df, train_dir):
    """
    Processes a single week of tracking data from loading to metric calculation.
    """
    print(f"  -> Processing Week {week_number}...")
    try:
        # Construct filename and load tracking data
        tracking_filename = f"input_2023_w{week_number:02d}.csv"
        tracking_df = pd.read_csv(os.path.join(train_dir, tracking_filename))

        # Merge with supplementary data
        week_merged_df = pd.merge(tracking_df, supp_df, on=['game_id', 'play_id'], how='left')

        # Engineer the 'dist_to_land_spot' feature
        week_merged_df['dist_to_land_spot'] = np.sqrt(
            (week_merged_df['x'] - week_merged_df['ball_land_x'])**2 +
            (week_merged_df['y'] - week_merged_df['ball_land_y'])**2
        )
        
        # Filter for only pass plays we want to analyze
        pass_plays_df = week_merged_df[week_merged_df['pass_result'].isin(['C', 'I'])].copy()
        
        if pass_plays_df.empty:
            print(f"  -> No pass plays found for Week {week_number}. Skipping.")
            return None
        
        # Group by play and calculate metrics
        weekly_metrics = pass_plays_df.groupby(['game_id', 'play_id']).apply(calculate_play_metrics)
        return weekly_metrics
    
    except FileNotFoundError:
        print(f"  -> File not found for Week {week_number}. Skipping.")
        return None
    except Exception as e:
        print(f"  -> An error occurred processing Week {week_number}: {e}")
        return None

# --- Main Loop to Process All Weeks ---
all_weeks_metrics = []
if 'supp_df' in locals() and not supp_df.empty:
    for week in tqdm(range(1, 19), desc="Processing All Weeks"): # NFL season has 18 weeks
        weekly_result = process_week_data(week, supp_df, TRAIN_DIR)
        if weekly_result is not None:
            all_weeks_metrics.append(weekly_result)

    # --- Combine all results into a single DataFrame ---
    full_season_metrics_df = pd.concat(all_weeks_metrics)
    print(f"\n\nFull season processing complete. Total plays analyzed: {len(full_season_metrics_df)}")

    # --- Re-run Player-Level Aggregation on Full Season Data ---
    print("\n" + "="*50)
    print("GENERATING FINAL, FULL-SEASON LEADERBOARDS")
    print("="*50)
    
    # Aggregate stats
    player_summary_season = full_season_metrics_df.groupby('targeted_receiver').agg(
        num_targets=('pass_result', 'count'),
        avg_separation=('separation_at_catch', 'mean'),
        avg_tcr=('target_convergence_rate_yps', 'mean')
    ).reset_index()

    # Set a higher minimum target count for a full season
    MIN_TARGETS_SEASON = 50 
    qualified_players_season = player_summary_season[player_summary_season['num_targets'] >= MIN_TARGETS_SEASON].copy()
    
    print(f"\nAnalyzing {len(qualified_players_season)} receivers with at least {MIN_TARGETS_SEASON} targets over the full season.")
    
    # --- Leaderboard 1: Top Separation Artists (Full Season) ---
    top_separation_season = qualified_players_season.sort_values(by='avg_separation', ascending=False).head(10)
    print("\n--- SEASON LEADERBOARD: Top 10 by Average Separation ---")
    display(top_separation_season[['targeted_receiver', 'avg_separation', 'num_targets']])

    # --- Leaderboard 2: Top "To the Spot" Speedsters (Full Season) ---
    top_tcr_season = qualified_players_season.sort_values(by='avg_tcr', ascending=False).head(10)
    print("\n--- SEASON LEADERBOARD: Top 10 by Target Convergence Rate ---")
    display(top_tcr_season[['targeted_receiver', 'avg_tcr', 'num_targets']])

else:
    print("ERROR: The 'supp_df' DataFrame was not found. Please re-run the initial setup cells.")


# ===================================================================
# FINAL STEP: VISUALIZING FULL-SEASON PERFORMANCE MATRIX
# ===================================================================

print("Generating the final, full-season performance matrix visualization...")

if 'qualified_players_season' in locals() and not qualified_players_season.empty:

    plt.figure(figsize=(18, 12))
    
    # Create the scatter plot
    plot = sns.scatterplot(
        data=qualified_players_season,
        x='avg_tcr',
        y='avg_separation',
        size='num_targets',
        hue='num_targets', # Using hue to reinforce the size aesthetic
        palette='viridis_r', # Reversed viridis palette
        sizes=(50, 800),
        alpha=0.8
    )

    # Add labels for a few standout players in each quadrant
    for i, row in qualified_players_season.iterrows():
        # Elite Quadrant (Top Right)
        if row['avg_tcr'] > 3.4 and row['avg_separation'] > 2.2:
             plt.text(row['avg_tcr'] + 0.02, row['avg_separation'], row['targeted_receiver'], fontsize=11, fontweight='bold')
        # Separation Specialist Quadrant (Top Left)
        if row['avg_tcr'] < 3.0 and row['avg_separation'] > 2.5:
             plt.text(row['avg_tcr'] + 0.02, row['avg_separation'], row['targeted_receiver'], fontsize=11, fontweight='bold')
        # Deep Threat Quadrant (Bottom Right)
        if row['avg_tcr'] > 3.7 and row['avg_separation'] < 2.0:
             plt.text(row['avg_tcr'] + 0.02, row['avg_separation'], row['targeted_receiver'], fontsize=11, fontweight='bold')
             
    # --- Aesthetics and Labels ---
    plt.title('Receiver Performance Matrix (Full 2023 Season)', fontsize=24, fontweight='bold', pad=20)
    plt.xlabel('Target Convergence Rate (Yds/Sec) -> [Faster to the Spot]', fontsize=16)
    plt.ylabel('Average Separation at Catch (Yards) -> [More Open]', fontsize=16)
    
    # Add average lines
    plt.axhline(qualified_players_season['avg_separation'].mean(), ls='--', color='grey', lw=2)
    plt.axvline(qualified_players_season['avg_tcr'].mean(), ls='--', color='grey', lw=2)
    
    # Customize legend
    h, l = plot.get_legend_handles_labels()
    plt.legend(h[1:7], l[1:7], title='Number of Targets', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

else:
    print("ERROR: The 'qualified_players_season' DataFrame was not found. Please re-run Step 6.")

