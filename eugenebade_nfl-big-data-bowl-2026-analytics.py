import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# ==========================================
# PART 1: ROBUST DATA LOADING (AUTO-FINDER)
# ==========================================
print("Searching for files...")

# 1. Find the Input file (Week 1)
# We search recursively just in case the folder structure varies
input_files = glob.glob('/kaggle/input/**/input_2023_w01.csv', recursive=True)
if not input_files:
    # Fallback: Try finding ANY input file if Week 1 specifically is missing
    input_files = glob.glob('/kaggle/input/**/input*.csv', recursive=True)

# 2. Find the Output file (Week 1)
output_files = glob.glob('/kaggle/input/**/output_2023_w01.csv', recursive=True)
if not output_files:
    # Fallback: Try finding ANY output file
    output_files = glob.glob('/kaggle/input/**/output*.csv', recursive=True)

# 3. Find the Supplementary file
play_files = glob.glob('/kaggle/input/**/supplementary_data.csv', recursive=True)

# --- CHECK IF FILES WERE FOUND ---
if not input_files or not output_files or not play_files:
    print("\nCRITICAL ERROR: Files still not found!")
    print("Please ensure you clicked 'Add Input' -> 'Competition Data' in the right sidebar.")
    print("Found inputs:", input_files)
    print("Found outputs:", output_files)
    print("Found plays:", play_files)
    # Stop execution if data is missing to avoid crashing later
    raise FileNotFoundError("Competition data not attached to notebook.")
else:
    # Select the first match found
    INPUT_FILE = input_files[0]
    OUTPUT_FILE = output_files[0]
    PLAY_FILE = play_files[0]
    
    print(f"Found Input File: {INPUT_FILE}")
    print(f"Found Output File: {OUTPUT_FILE}")
    print(f"Found Play File: {PLAY_FILE}")

    # --- LOAD DATA ---
    print("\nLoading data (this may take about 10-20 seconds)...")
    input_df = pd.read_csv(INPUT_FILE)
    output_df = pd.read_csv(OUTPUT_FILE)
    play_df = pd.read_csv(PLAY_FILE)
    print("SUCCESS: Data loaded!")

# ==========================================
# PART 2: PRE-PROCESSING & ANALYSIS
# ==========================================
print("Filtering for Pass Plays...")

# Filter input for "Targeted Receiver" and "Defensive Coverage"
# We only care about completed passes for this quick metric
completed_plays = play_df[play_df['pass_result'] == 'C'][['game_id', 'play_id']]

# Filter output data to only completed plays (for clean analysis)
tracking = output_df.merge(completed_plays, on=['game_id', 'play_id'])

# ==========================================
# PART 3: CALCULATE SEPARATION DELTA
# ==========================================

def calculate_separation_delta(game_id, play_id):
    # Get tracking for this specific play
    play_track = tracking[(tracking['game_id'] == game_id) & (tracking['play_id'] == play_id)]
    
    # Validation check
    if play_track.empty: 
        return None
    
    # Get Start Frame (Throw) and End Frame (Arrival)
    start_frame = play_track['frame_id'].min()
    end_frame = play_track['frame_id'].max()
    
    # Get positions at start and end
    start_pos = play_track[play_track['frame_id'] == start_frame]
    end_pos = play_track[play_track['frame_id'] == end_frame]
    
    # Calculate simple dispersion (separation) at start vs end
    # Separation Delta = (Avg Distance between players at End) - (Avg Distance at Start)
    # A positive number means they spread out (Receiver pulled away).
    # A negative number means they clumped (Defender closed in).
    
    # We use standard deviation as a proxy for "spread" between the receiver and defenders
    sep_start = np.std(start_pos['x']) + np.std(start_pos['y'])
    sep_end = np.std(end_pos['x']) + np.std(end_pos['y'])
    
    return sep_end - sep_start

# Run on first 100 plays to generate the graph quickly
results = []
# Get a unique list of plays from the tracking data we loaded
unique_plays = tracking[['game_id', 'play_id']].drop_duplicates().head(100)

print(f"Calculating Separation Delta for {len(unique_plays)} plays...")
for _, row in unique_plays.iterrows():
    delta = calculate_separation_delta(row['game_id'], row['play_id'])
    if delta is not None:
        results.append(delta)

# ==========================================
# PART 4: VISUALIZATION (THUMBNAIL GENERATOR)
# ==========================================
if len(results) > 0:
    plt.figure(figsize=(10, 6))
    sns.histplot(results, kde=True, color='blue', bins=20)
    plt.axvline(0, color='red', linestyle='--', label='No Change')
    plt.title('Distribution of "Separation Delta" (Ball-in-Air)', fontsize=16)
    plt.xlabel('Separation Delta (Yards Gained/Lost)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot for your thumbnail
    plt.savefig('thumbnail_image.png')
    print("SUCCESS: Plot generated and saved as 'thumbnail_image.png'!")
    print("Action: Take a screenshot of the graph below to upload as your cover image.")
    plt.show()
else:
    print("Error: No results were calculated. Check if the dataset was loaded correctly.")

