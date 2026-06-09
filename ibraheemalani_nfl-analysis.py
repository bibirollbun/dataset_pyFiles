import pandas as pd
import glob
import os

# 1. Find the correct directory automatically
search_path = '/kaggle/input/nfl-big-data-bowl-2026-analytics/**/train'
found_dirs = glob.glob(search_path, recursive=True)

if not found_dirs:
    raise FileNotFoundError("Could not find the 'train' directory. Please check the dataset.")

base_path = found_dirs[0]
print(f"Data directory found: {base_path}")

# 2. Load Week 1 Data
print("Loading Week 1 Data...")
# Output = Movement data (Ball in air)
df_output = pd.read_csv(f'{base_path}/output_2023_w01.csv') 
# Input = Context data (Player roles, names, sides)
df_input = pd.read_csv(f'{base_path}/input_2023_w01.csv')   

# 3. Create a "Play Metadata" DataFrame
# We need to know who is a Receiver and who is a Defender for every specific play.
# We extract just the unique roles per play from the input file.
play_metadata = df_input[['game_id', 'play_id', 'nfl_id', 'player_name', 'player_role', 'player_side']].drop_duplicates()

# 4. Merge
# Connect the roles to the movement tracking
df_tracking = df_output.merge(play_metadata, on=['game_id', 'play_id', 'nfl_id'], how='left')

print("Success! Data loaded and merged.")
print(f"Tracking Rows: {len(df_tracking)}")
print(df_tracking.head())


import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. Fixed Field Drawing Function ---
def create_football_field(figsize=(12, 6.33)):
    """
    Creates a simple green football field with white yard lines.
    """
    fig, ax = plt.subplots(1, figsize=figsize)
    
    # Green Background (The Field)
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)
    ax.add_patch(rect)

    # White Yard Lines (Draw a vertical line every 10 yards)
    for x in range(10, 110, 10):
        ax.plot([x, x], [0, 53.3], color='white', alpha=0.5, zorder=1)

    # Endzones (Blue rectangles at both ends)
    ez1 = patches.Rectangle((0, 0), 10, 53.3, alpha=0.2, facecolor='blue', zorder=1)
    ez2 = patches.Rectangle((110, 0), 10, 53.3, alpha=0.2, facecolor='blue', zorder=1)
    ax.add_patch(ez1)
    ax.add_patch(ez2)
    
    # Set limits and hide axis numbers
    plt.xlim(0, 120)
    plt.ylim(0, 53.3)
    plt.axis('off')
    
    return fig, ax

# --- 2. Select a Random Play to Plot ---
# We pick the first play available in our loaded data
sample_game = df_tracking['game_id'].unique()[0]
sample_play = df_tracking[df_tracking['game_id'] == sample_game]['play_id'].unique()[0]

play_data = df_tracking[(df_tracking['game_id'] == sample_game) & 
                        (df_tracking['play_id'] == sample_play)]

print(f"Visualizing Game: {sample_game}, Play: {sample_play}")

# --- 3. Plot the Players ---
fig, ax = create_football_field()

# Define colors for different roles
role_colors = {
    'Defensive Coverage': 'red',
    'Targeted Receiver': 'cyan',
    'Passer': 'white',
    'Pass Rush': 'orange',    # Some data uses this
    'Blocking': 'gray'        # Some data uses this
}

# Plot each player
for role in play_data['player_role'].unique():
    subset = play_data[play_data['player_role'] == role]
    
    # Choose color (default to yellow if role is unknown)
    color = role_colors.get(role, 'yellow')
    
    # Scatter plot
    ax.scatter(subset['x'], subset['y'], c=color, s=30, label=role, edgecolors='black', zorder=2)

# Add Legend and Title
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.title(f"Player Positions: Game {sample_game} Play {sample_play}")
plt.show()


# 1. Update the Metadata Selection
# We added 'ball_land_x' and 'ball_land_y' to the list of columns we want
play_metadata = df_input[['game_id', 'play_id', 'nfl_id', 'player_role', 'ball_land_x', 'ball_land_y']].drop_duplicates()

# 2. Merge again
# Now every row in our tracking data knows where the ball is going to land
df_tracking = df_output.merge(play_metadata, on=['game_id', 'play_id', 'nfl_id'], how='left')

# 3. Calculate Distance to Ball Landing Spot (Pythagorean Theorem)
import numpy as np

df_tracking['dist_to_ball'] = np.sqrt(
    (df_tracking['x'] - df_tracking['ball_land_x'])**2 + 
    (df_tracking['y'] - df_tracking['ball_land_y'])**2
)

print("Distance calculated! Ready to analyze.")


# 1. Filter for our specific sample play
sample_game = df_tracking['game_id'].unique()[0]
sample_play = df_tracking[df_tracking['game_id'] == sample_game]['play_id'].unique()[0]

play_data = df_tracking[(df_tracking['game_id'] == sample_game) & 
                        (df_tracking['play_id'] == sample_play)]

# 2. Filter for Defense Only
defense_data = play_data[play_data['player_role'] == 'Defensive Coverage']

# 3. Calculate the Metrics per Frame
# Mean Distance: How close is the ENTIRE defense?
# Min Distance: How close is the CLOSEST defender?
swarm_stats = defense_data.groupby('frame_id')['dist_to_ball'].agg(['mean', 'min'])

# 4. Plot the Result
plt.figure(figsize=(10, 6))

# Plot Average Distance (The Swarm)
plt.plot(swarm_stats.index, swarm_stats['mean'], label='Avg Distance (Team)', color='red', linewidth=3)

# Plot Closest Defender (The Threat)
plt.plot(swarm_stats.index, swarm_stats['min'], label='Closest Defender', color='orange', linestyle='--', linewidth=2)

plt.title(f"Defensive Swarm Intensity\nGame {sample_game} Play {sample_play}")
plt.xlabel("Frame (Time)")
plt.ylabel("Distance to Landing Spot (Yards)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().invert_yaxis() # Invert Y so that "Moving Closer" goes UP (optional, removes confusion)
plt.show()


import numpy as np

# 1. Prepare a list to store results
results = []

# 2. Group data by Game and Play
# This might take 30-60 seconds to run because we are looping through many plays
print("Calculating Swarm Rate for all plays...")

for (game_id, play_id), play_df in df_tracking.groupby(['game_id', 'play_id']):
    
    # Filter for Defense only
    defense_df = play_df[play_df['player_role'] == 'Defensive Coverage']
    
    # We need at least a few frames to calculate a slope
    if len(defense_df) < 10:
        continue
        
    # Calculate average distance to ball per frame
    avg_dist_per_frame = defense_df.groupby('frame_id')['dist_to_ball'].mean()
    
    # DATA SCIENCE MAGIC: Calculate the Slope (The "Swarm Rate")
    # We fit a straight line to the distance curve.
    # A negative slope means distance is decreasing (Good).
    # A more negative number = Faster swarm.
    try:
        slope, intercept = np.polyfit(avg_dist_per_frame.index, avg_dist_per_frame.values, 1)
        
        # Store the result
        results.append({
            'game_id': game_id,
            'play_id': play_id,
            'swarm_rate': slope, # Yards per frame (negative is better)
            'starting_dist': avg_dist_per_frame.iloc[0],
            'ending_dist': avg_dist_per_frame.iloc[-1]
        })
    except:
        continue

# 3. Convert to DataFrame and Sort
df_results = pd.DataFrame(results)

# Sort by 'swarm_rate' ascending (most negative first = fastest closing speed)
df_results = df_results.sort_values('swarm_rate', ascending=True)

print("Calculation Complete!")
print("Top 5 'Best Swarming' Plays (Fastest reaction):")
print(df_results.head(5))


# 1. Select the Top Play from your results
best_game = 2023091010
best_play = 4426

print(f"Visualizing the #1 Swarm Play: Game {best_game} Play {best_play}")

play_data = df_tracking[(df_tracking['game_id'] == best_game) & 
                        (df_tracking['play_id'] == best_play)]

# 2. Setup the Plot
fig, ax = create_football_field()

# 3. Plot the movement
# We will use alpha (transparency) to show the 'ghost' trails of where they started
for role in play_data['player_role'].unique():
    subset = play_data[play_data['player_role'] == role]
    
    color = 'red' if role == 'Defensive Coverage' else 'cyan' if role == 'Targeted Receiver' else 'gray'
    if role == 'Passer': color = 'white'
    
    # Plot the path (lighter dots)
    ax.scatter(subset['x'], subset['y'], c=color, s=10, alpha=0.3)
    
    # Plot the FINAL position (darker, larger dots)
    final_pos = subset.tail(1)
    ax.scatter(final_pos['x'], final_pos['y'], c=color, s=50, edgecolors='black', zorder=10)

plt.title(f"The 'Swarm': Fastest Defensive Reaction (Rate: -0.90)\nGame {best_game} Play {best_play}")
plt.legend(['Defensive Path', 'Receiver Path'], loc='upper right')
plt.show()




