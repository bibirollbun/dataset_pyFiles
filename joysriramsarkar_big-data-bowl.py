# -*- coding: utf-8 -*-
"""
NFL Big Data Bowl 2026 - Analytics Submission
---------------------------------------------
Project: Analyzing Receiver Separation to Understand Pass Success

This notebook analyzes receiver separation from defenders using NFL Next Gen Stats data.
The primary goal is to visualize how separation impacts the success or failure of a pass.
"""

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("Step 1: All necessary libraries have been imported successfully.")

# --- Constants & Functions ---

# Store the correct paths to data files in variables
BASE_DATA_DIR = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final'
SUPPLEMENTARY_FILE_PATH = os.path.join(BASE_DATA_DIR, 'supplementary_data.csv')
TRAIN_DIR_PATH = os.path.join(BASE_DATA_DIR, 'train')

def analyze_and_plot_play(game_id, play_id, supplementary_df, train_dir, analysis_title="Play Analysis"):
    """
    This function performs a complete analysis of a specific play (identified by game_id and play_id)
    and generates a separation profile plot.
    """
    print("\n" + "="*50)
    print(f"Starting Analysis: {analysis_title}")
    print(f"Game ID: {game_id}, Play ID: {play_id}")
    print("="*50)
    
    try:
        # Step 2: Find information for the specific play
        play_info = supplementary_df[(supplementary_df['game_id'] == game_id) & (supplementary_df['play_id'] == play_id)].iloc[0]
        week = play_info['week']
        season = play_info['season']
        
        # Step 3: Load the corresponding tracking files
        input_path = f'{train_dir}/input_{season}_w{week:02d}.csv'
        output_path = f'{train_dir}/output_{season}_w{week:02d}.csv'
        
        input_df = pd.read_csv(input_path)
        output_df = pd.read_csv(output_path)
        
        # Step 4: Concatenate the two parts of the play to create complete tracking data
        play_input_data = input_df[(input_df['game_id'] == game_id) & (input_df['play_id'] == play_id)]
        play_output_data = output_df[(output_df['game_id'] == game_id) & (output_df['play_id'] == play_id)]
        
        common_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y']
        full_tracking_df = pd.concat([
            play_input_data[common_cols], 
            play_output_data[common_cols]
        ]).sort_values('frame_id').reset_index(drop=True)
        print("Step 4: Play tracking data successfully reconstructed.")
        
        # Step 5: Add player information like name, position, etc.
        player_info_cols = ['nfl_id', 'player_name', 'player_position', 'player_side', 'player_role']
        player_info = play_input_data[player_info_cols].drop_duplicates()
        analysis_df = pd.merge(full_tracking_df, player_info, on='nfl_id')
        
        # Step 6: Isolate the targeted receiver and defenders
        targeted_receiver = analysis_df[analysis_df['player_role'] == 'Targeted Receiver']
        defenders = analysis_df[analysis_df['player_side'] == 'Defense']
        
        if targeted_receiver.empty:
            print("Error: No 'Targeted Receiver' found for this play.")
            return
            
        # Step 7: Calculate the distance (separation) between the receiver and the nearest defender for each frame
        separation_data = []
        for frame in targeted_receiver['frame_id'].unique():
            receiver_pos = targeted_receiver[targeted_receiver['frame_id'] == frame][['x', 'y']].iloc[0]
            defenders_pos = defenders[defenders['frame_id'] == frame]
            
            if not defenders_pos.empty:
                # Calculate Euclidean distance
                distances = np.sqrt(((defenders_pos[['x', 'y']] - receiver_pos)**2).sum(axis=1))
                separation_data.append({'frame_id': frame, 'separation': distances.min()})
        
        separation_df = pd.DataFrame(separation_data)
        print("Step 7: Separation calculated successfully.")
        
        # Step 8: Visualize the results using a plot
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        pass_res_text = "Successful" if play_info['pass_result'] == 'C' else "Incomplete"
        line_color = 'green' if play_info['pass_result'] == 'C' else 'red'
        
        ax.plot(separation_df['frame_id'], separation_df['separation'], marker='o', linestyle='-', color=line_color, label=f'Pass Result: {pass_res_text}')
        
        receiver_name = targeted_receiver['player_name'].iloc[0]
        route = play_info['route_of_targeted_receiver']
        coverage = play_info['team_coverage_type']

        title = (f'Receiver Separation Profile: {receiver_name} on a "{route}" Route\n'
                 f'Result: {pass_res_text} Pass (vs. {coverage} Coverage)')
        
        ax.set_title(title, fontsize=18, weight='bold')
        ax.set_xlabel('Frame ID (Time)', fontsize=14)
        ax.set_ylabel('Separation from Nearest Defender (Yards)', fontsize=14)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Annotate the most important moment in the plot
        min_sep_frame = separation_df.loc[separation_df['separation'].idxmin()]
        ax.annotate(f'Minimum Separation\n({min_sep_frame["separation"]:.2f} yds at Frame {min_sep_frame["frame_id"]})',
                    xy=(min_sep_frame['frame_id'], min_sep_frame['separation']),
                    xytext=(min_sep_frame['frame_id'] + 2, min_sep_frame['separation'] + 1.5),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    weight='bold')
        
        plt.show()
        print("Step 8: Plot generated successfully.")

    except Exception as e:
        print(f"An error occurred while analyzing play {game_id}-{play_id}: {e}")


# --- Main Analysis ---

print("\n\n--- NFL Big Data Bowl 2026 Analytics Report ---")
try:
    supplementary_df = pd.read_csv(SUPPLEMENTARY_FILE_PATH, low_memory=False)
    print("Main supplementary data loaded successfully.")

    # Case Study 1: Analysis of a Successful Pass
    # We are looking for a play where the route was 'IN' and the pass was successful.
    successful_play = supplementary_df[(supplementary_df['pass_result'] == 'C') & (supplementary_df['route_of_targeted_receiver'] == 'IN')].iloc[0]
    analyze_and_plot_play(successful_play['game_id'], successful_play['play_id'], supplementary_df, TRAIN_DIR_PATH, 
                          analysis_title="Case Study 1: A Successful 'IN' Route Pass")

    # Case Study 2: Analysis of an Incomplete Pass (for comparison)
    # Now we are looking for a play where the route was 'OUT' and the pass was incomplete.
    incomplete_play = supplementary_df[(supplementary_df['pass_result'] == 'I') & (supplementary_df['route_of_targeted_receiver'] == 'OUT')].iloc[0]
    analyze_and_plot_play(incomplete_play['game_id'], incomplete_play['play_id'], supplementary_df, TRAIN_DIR_PATH, 
                          analysis_title="Case Study 2: An Incomplete 'OUT' Route Pass")
                          
    # Case Study 3: A deep pass against Man-to-Man coverage
    # A play where the defense was aggressive (Cover 1 Man) and the pass was long.
    deep_pass_play = supplementary_df[(supplementary_df['pass_result'] == 'C') & (supplementary_df['team_coverage_type'] == 'COVER_1_MAN') & (supplementary_df['pass_length'] > 20)].iloc[0]
    analyze_and_plot_play(deep_pass_play['game_id'], deep_pass_play['play_id'], supplementary_df, TRAIN_DIR_PATH, 
                          analysis_title="Case Study 3: A Successful Deep Pass Against Man Coverage")

except FileNotFoundError:
    print("\nError: Data files not found. Please check the file path.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")

print("\n\n--- Analysis Complete ---")
print("In this notebook, we have observed the difference in separation between successful and incomplete passes and analyzed various play scenarios.")



