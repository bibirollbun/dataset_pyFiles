import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
pd.set_option('display.max_columns', None)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



df1 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv')



w1= pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w01.csv')

ow1= pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/output_2023_w01.csv')


filtered_df1 = df1[df1['game_id'] == 2023090700]
filtered_df1 = filtered_df1.sort_values(by=['quarter', 'game_clock'], ascending=[True, False])
display(filtered_df1)


df2 = df1.copy()
display(df2.head())


merged_df_2 = pd.merge(df2, w1, on=['game_id', 'play_id'])
display(merged_df_2.head())
merged_df_2


df2 = df2.sort_values(by=['game_id', 'quarter', 'game_clock'], ascending=[True, True, False])
#display(df2.head())
df2


play_data = w1[(w1['game_id'] == 2023090700) & (w1['play_id'] == 194)]
unique_frame_ids = play_data['frame_id'].unique().tolist()

print(f"Filtered play_data for game_id 2023090700 and play_id 101. Number of rows: {len(play_data)}")
print(f"Unique frame IDs: {unique_frame_ids}")


df2_2023_w1 = df2[(df2['season'] == 2023) & (df2['week'] == 1)]
display(df2_2023_w1.head())


All_2023_w1 = pd.merge(w1, df2_2023_w1, on=['game_id', 'play_id'], how='left')
display(All_2023_w1.head())
All_2023_w1


for week_num in range(2, 19):
    

    # Load the weekly input data
    input_df = pd.read_csv(f'/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w{week_num:02d}.csv')

    # Filter df1 for the current week and season
    df1_weekly = df1[(df1['season'] == 2023) & (df1['week'] == week_num)]

    # Perform a left merge
    merged_df = pd.merge(input_df, df1_weekly, on=['game_id', 'play_id'], how='left')

    # Dynamically assign the merged DataFrame to a new global variable
    globals()[f'All_2023_w{week_num}'] = merged_df

    print(f"Created DataFrame All_2023_w{week_num} with {len(merged_df)} rows.")



ow1 = ow1.rename(columns={'x': 'x (after pass)', 'y': 'y (after pass)'})
display(ow1.head())


All_2023_w1 = pd.merge(All_2023_w1, ow1, on=['game_id', 'play_id', 'nfl_id', 'frame_id'], how='left')
display(All_2023_w1.head())


for week_num in range(2, 19):
  

    # Load weekly output data
    ow_weekly = pd.read_csv(f"/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/output_2023_w{week_num:02d}.csv")

    # Rename 'x' and 'y' columns
    ow_weekly = ow_weekly.rename(columns={'x': 'x (after pass)', 'y': 'y (after pass)'})

    # Merge into the corresponding All_2023_wX DataFrame
    globals()[f"All_2023_w{week_num}"] = pd.merge(globals()[f"All_2023_w{week_num}"], ow_weekly, on=['game_id', 'play_id', 'nfl_id', 'frame_id'], how='left')

    print(f"Merged All_2023_w{week_num} with output data.")

# Display the head of the last generated DataFrame (All_2023_w18)
display(All_2023_w18.head())



weekly_dataframes = []

for week_num in range(1, 19):
    # Construct the name of the DataFrame for the current week
    df_name = f"All_2023_w{week_num}"

    # Access the DataFrame using globals() and append to the list
    if df_name in globals():
        weekly_dataframes.append(globals()[df_name])
    else:
        print(f"DataFrame {df_name} not found.")

# Concatenate all weekly DataFrames into a single DataFrame
All_2023 = pd.concat(weekly_dataframes, ignore_index=True)

print(f"Combined all weekly DataFrames into All_2023 with {len(All_2023)} rows.")
# Display the head of the new combined DataFrame
display(All_2023.head())


All_4_zone = All_2023[All_2023['team_coverage_type'] == 'COVER_4_ZONE']
display(All_4_zone.head())


animation_data = All_2023[(All_2023['game_id'] == 2023090700) & (All_2023['play_id'] == 101)]
unique_frame_ids = animation_data['frame_id'].unique().tolist()
max_initial_frame_id = max(unique_frame_ids)

print(f"Filtered animation_data for game_id 2023090700 and play_id 194. Number of rows: {len(animation_data)}")
print(f"Unique frame IDs: {unique_frame_ids}")
print(f"Maximum initial frame ID: {max_initial_frame_id}")


last_frame_data = animation_data[animation_data['frame_id'] == max_initial_frame_id]

last_initial_positions = {}
for index, row in last_frame_data.iterrows():
    last_initial_positions[row['nfl_id']] = (row['x'], row['y'])

print(f"Captured last initial positions for {len(last_initial_positions)} players at frame {max_initial_frame_id}.")
# Display a sample of the dictionary
print("Sample of last_initial_positions:")
for i, (nfl_id, pos) in enumerate(last_initial_positions.items()):
    if i >= 5: # Display only first 5 for brevity
        break
    print(f"  NFL ID: {nfl_id}, Position: {pos}")


import imageio
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt

plot_frames = []

# Define field dimensions for consistent axis limits
x_min, x_max = 0, 120
y_min, y_max = 0, 53.3

for frame_id in unique_frame_ids:
    frame_data = animation_data[animation_data['frame_id'] == frame_id]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Assign colors based on player_side
    # Plot offense and defense separately to create legend entries
    for side in frame_data['player_side'].unique():
        side_data = frame_data[frame_data['player_side'] == side]
        color = 'blue' if side == 'Offense' else 'orange'
        ax.scatter(side_data['x'], side_data['y'], s=50, alpha=0.7, c=color, label=side)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f"Game ID: {animation_data['game_id'].iloc[0]}, Play ID: {animation_data['play_id'].iloc[0]}, Frame ID: {frame_id}")
    ax.set_aspect('equal', adjustable='box') # Keep aspect ratio for field
    ax.legend() # Add legend

    # Save the current plot to a buffer as a PNG, then append to frames list
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plot_frames.append(np.array(img))

    plt.close(fig) # Close the figure to free up memory

print(f"Generated {len(plot_frames)} plot frames for initial player movement.")


for frame_id in unique_frame_ids:
    frame_data_after_pass = animation_data[animation_data['frame_id'] == frame_id].copy()

    # Use 'x (after pass)' and 'y (after pass)' if available, otherwise fallback
    frame_data_after_pass['current_x'] = frame_data_after_pass.apply(
        lambda row: row['x (after pass)'] if pd.notna(row['x (after pass)']) else last_initial_positions.get(row['nfl_id'], (row['x'], row['y']))[0],
        axis=1
    )
    frame_data_after_pass['current_y'] = frame_data_after_pass.apply(
        lambda row: row['y (after pass)'] if pd.notna(row['y (after pass)']) else last_initial_positions.get(row['nfl_id'], (row['x'], row['y']))[1],
        axis=1
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot offense and defense separately to create legend entries
    for side in frame_data_after_pass['player_side'].unique():
        side_data = frame_data_after_pass[frame_data_after_pass['player_side'] == side]
        color = 'blue' if side == 'Offense' else 'orange'
        ax.scatter(side_data['current_x'], side_data['current_y'], s=50, alpha=0.7, c=color, label=side)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f"Game ID: {animation_data['game_id'].iloc[0]}, Play ID: {animation_data['play_id'].iloc[0]}, Frame ID: {frame_id} (After Pass)")
    ax.set_aspect('equal', adjustable='box') # Keep aspect ratio for field
    ax.legend()

    # Save the current plot to a buffer as a PNG, then append to frames list
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plot_frames.append(np.array(img))

    plt.close(fig) # Close the figure to free up memory

print(f"Generated {len(plot_frames) - max_initial_frame_id} plot frames for after pass movement.")


gif_filename_combined = 'player_movement_combined.gif'
imageio.mimsave(gif_filename_combined, plot_frames, fps=10) # Adjust fps as needed

print(f"Combined GIF animation saved as {gif_filename_combined}")

# Display the GIF (this works in environments like Jupyter/Colab)
from IPython.display import Image, display
display(Image(filename=gif_filename_combined))


All_skill_players_4_zone = All_4_zone[All_4_zone['player_position'].isin(['WR', 'RB', 'TE','QB'])]
display(All_skill_players_4_zone.head())
display(All_skill_players_4_zone)
len(All_skill_players_4_zone)


original_max_rows = pd.options.display.max_rows
pd.options.display.max_rows = None
filtered_skill_players = All_skill_players_4_zone[(All_skill_players_4_zone['game_id'] == 2023090700) & (All_skill_players_4_zone['play_id'] == 219)]
display(filtered_skill_players)
pd.options.display.max_rows = original_max_rows # Reset to original setting


All_skill_frame1 = All_skill_players_4_zone[(All_skill_players_4_zone['frame_id'] == 1)]
display(All_skill_frame1)


import pandas as pd
import numpy as np

def calculate_receiver_number(play_df):
    # Ensure a copy to avoid SettingWithCopyWarning
    play_df_copy = play_df.copy()

    # 2. Identify the QB's y-coordinate
    qb_row = play_df_copy[play_df_copy['player_position'] == 'QB']
    if not qb_row.empty:
        qb_y = qb_row['y'].iloc[0]
    else:
        # If no QB is found, cannot calculate receiver numbers meaningfully based on QB position
        play_df_copy['receiver_number'] = 'N/A'
        return play_df_copy

    # 3. Create a new column 'receiver_number' and initialize it with NaN
    play_df_copy['receiver_number'] = np.nan

    # 4. For all rows where player_position is 'QB', set the 'receiver_number' to 'N/A'
    play_df_copy.loc[play_df_copy['player_position'] == 'QB', 'receiver_number'] = 'N/A'

    # 5. Filter the DataFrame to include only offensive skill players (WR, RB, TE)
    skill_players_mask = ((play_df_copy['player_side'] == 'Offense') &
                          (play_df_copy['player_position'].isin(['WR', 'RB', 'TE'])))
    skill_players = play_df_copy[skill_players_mask].copy()

    if not skill_players.empty:
        # 6. Calculate the absolute difference between their 'y' coordinate and qb_y
        skill_players['distance_from_qb_y'] = abs(skill_players['y'] - qb_y)

        # 7. Create two sub-DataFrames: 'above' and 'below' qb_y
        above_qb = skill_players[skill_players['y'] > qb_y].copy()
        below_qb = skill_players[skill_players['y'] < qb_y].copy()

        # 8. For 'above' sub-DataFrame, sort by distance and assign ranks
        if not above_qb.empty:
            above_qb = above_qb.sort_values(by='distance_from_qb_y', ascending=False)
            above_qb['receiver_number'] = (above_qb['distance_from_qb_y'].rank(method='dense', ascending=False)).astype(int)

        # 9. Repeat for 'below' sub-DataFrame
        if not below_qb.empty:
            below_qb = below_qb.sort_values(by='distance_from_qb_y', ascending=False)
            # If there are players both above and below, start ranking from 1 for each group
            below_qb['receiver_number'] = (below_qb['distance_from_qb_y'].rank(method='dense', ascending=False)).astype(int)

        # 10. Update the 'receiver_number' column in the original DataFrame
        play_df_copy.loc[above_qb.index, 'receiver_number'] = above_qb['receiver_number']
        play_df_copy.loc[below_qb.index, 'receiver_number'] = below_qb['receiver_number']

    # 11. Return the modified input DataFrame
    return play_df_copy

print("Function 'calculate_receiver_number' defined successfully.")


All_skill_frame1 = All_skill_frame1.groupby(['game_id', 'play_id']).apply(calculate_receiver_number).reset_index(drop=True)
display(All_skill_frame1.head())


import pandas as pd
import numpy as np

def calculate_receiver_number(play_df):
    # Ensure a copy to avoid SettingWithCopyWarning
    play_df_copy = play_df.copy()

    # 2. Identify the QB's y-coordinate
    qb_row = play_df_copy[play_df_copy['player_position'] == 'QB']
    if not qb_row.empty:
        qb_y = qb_row['y'].iloc[0]
    else:
        # If no QB is found, cannot calculate receiver numbers meaningfully based on QB position
        play_df_copy['receiver_number'] = 'N/A'
        return play_df_copy

    # 3. Create a new column 'receiver_number' and initialize it with NaN, cast to object to allow mixed types
    play_df_copy['receiver_number'] = np.nan
    play_df_copy['receiver_number'] = play_df_copy['receiver_number'].astype(object)

    # 4. For all rows where player_position is 'QB', set the 'receiver_number' to 'N/A'
    play_df_copy.loc[play_df_copy['player_position'] == 'QB', 'receiver_number'] = 'N/A'

    # 5. Filter the DataFrame to include only offensive skill players (WR, RB, TE)
    skill_players_mask = ((play_df_copy['player_side'] == 'Offense') &
                          (play_df_copy['player_position'].isin(['WR', 'RB', 'TE'])))
    skill_players = play_df_copy[skill_players_mask].copy()

    if not skill_players.empty:
        # 6. Calculate the absolute difference between their 'y' coordinate and qb_y
        skill_players['distance_from_qb_y'] = abs(skill_players['y'] - qb_y)

        # 7. Create two sub-DataFrames: 'above' and 'below' qb_y
        above_qb = skill_players[skill_players['y'] > qb_y].copy()
        below_qb = skill_players[skill_players['y'] < qb_y].copy()

        # 8. For 'above' sub-DataFrame, sort by distance and assign ranks
        if not above_qb.empty:
            above_qb = above_qb.sort_values(by='distance_from_qb_y', ascending=False)
            above_qb['receiver_number'] = (above_qb['distance_from_qb_y'].rank(method='dense', ascending=False)).astype(int)

        # 9. Repeat for 'below' sub-DataFrame
        if not below_qb.empty:
            below_qb = below_qb.sort_values(by='distance_from_qb_y', ascending=False)
            # If there are players both above and below, start ranking from 1 for each group
            below_qb['receiver_number'] = (below_qb['distance_from_qb_y'].rank(method='dense', ascending=False)).astype(int)

        # 10. Update the 'receiver_number' column in the original DataFrame
        play_df_copy.loc[above_qb.index, 'receiver_number'] = above_qb['receiver_number']
        play_df_copy.loc[below_qb.index, 'receiver_number'] = below_qb['receiver_number']

    # 11. Return the modified input DataFrame
    return play_df_copy

print("Function 'calculate_receiver_number' defined successfully.")


All_skill_frame1 = All_skill_frame1.groupby(['game_id', 'play_id']).apply(calculate_receiver_number).reset_index(drop=True)
display(All_skill_frame1.head())


display(All_skill_frame1.head(30))


All_skill_players_4_zone = pd.merge(All_skill_players_4_zone, All_skill_frame1[['game_id', 'play_id', 'nfl_id', 'receiver_number']],
                                   on=['game_id', 'play_id', 'nfl_id'],
                                   how='left')
display(All_skill_players_4_zone.head())
display(All_skill_players_4_zone)


All_4_zone = pd.merge(All_4_zone, All_skill_players_4_zone[['game_id', 'play_id', 'nfl_id', 'receiver_number']],
                                   on=['game_id', 'play_id', 'nfl_id'],
                                   how='left')
display(All_4_zone.head())


display(All_4_zone)


import pandas as pd

def get_player_x_at_max_frame(play_df, nfl_id):
    """
    Finds the 'x' coordinate of a player at the maximum frame_id for a given play.

    Args:
        play_df (pd.DataFrame): DataFrame containing play data.
        nfl_id (int): The NFL ID of the player.

    Returns:
        float: The 'x' coordinate of the player at the max frame_id, or None if not found.
    """
    max_frame_id = play_df['frame_id'].max()
    filtered_player_data = play_df[(play_df['nfl_id'] == nfl_id) & (play_df['frame_id'] == max_frame_id)]
    if not filtered_player_data.empty:
        return filtered_player_data['x'].iloc[0]
    return None

def get_qb_coordinates_at_frame1(play_df):
    """
    Finds the 'x' and 'y' coordinates of the QB at frame_id 1 for a given play.

    Args:
        play_df (pd.DataFrame): DataFrame containing play data.

    Returns:
        tuple: (x_coordinate, y_coordinate) of the QB at frame 1, or (None, None) if no QB is found.
    """
    qb_at_frame1 = play_df[(play_df['player_position'] == 'QB') & (play_df['frame_id'] == 1)]
    if not qb_at_frame1.empty:
        return (qb_at_frame1['x'].iloc[0], qb_at_frame1['y'].iloc[0])
    return (None, None)

def get_player_side_of_qb(player_y, qb_y):
    """
    Determines if a player is to the 'left' or 'right' of the QB based on their y-coordinates.
    Assuming a field orientation where higher y-values are 'left' relative to the QB's position.

    Args:
        player_y (float): The y-coordinate of the player.
        qb_y (float): The y-coordinate of the Quarterback.

    Returns:
        str: 'left' if player_y > qb_y, 'right' otherwise.
    """
    if player_y > qb_y:
        return 'left'
    return 'right'

print("Helper functions 'get_player_x_at_max_frame', 'get_qb_coordinates_at_frame1', and 'get_player_side_of_qb' defined successfully.")


import numpy as np

def check_condition1(play_df):
    """
    Evaluates specific scenarios for defensive players and offensive receivers with receiver_number = 1.

    Args:
        play_df (pd.DataFrame): DataFrame containing play data for a single game_id and play_id.

    Returns:
        bool: True if either scenario is met for any CB-WR1 pair, False otherwise.
    """
    # 2. Get QB coordinates at frame_id=1
    qb_x_frame1, qb_y_frame1 = get_qb_coordinates_at_frame1(play_df)
    if qb_x_frame1 is None or qb_y_frame1 is None:
        return False

    # Get max_frame_id for the play
    max_frame_id = play_df['frame_id'].max()

    # 3. Filter defensive players with player_position == 'CB' at frame_id=1
    cb_players_frame1 = play_df[
        (play_df['player_side'] == 'Defense') &
        (play_df['player_position'] == 'CB') &
        (play_df['frame_id'] == 1)
    ]
    if cb_players_frame1.empty:
        return False

    # 5. Filter offensive players with receiver_number == 1 at frame_id=1
    wr1_players_frame1 = play_df[
        (play_df['player_side'] == 'Offense') &
        (play_df['receiver_number'] == 1) &
        (play_df['frame_id'] == 1)
    ]
    if wr1_players_frame1.empty:
        return False

    # Iterate through each CB to find matching WR1
    for _, cb_row in cb_players_frame1.iterrows():
        cb_nfl_id = cb_row['nfl_id']
        cb_y_frame1 = cb_row['y']
        cb_side = get_player_side_of_qb(cb_y_frame1, qb_y_frame1)

        # Find WR1s on the same side of the QB
        for _, wr1_row in wr1_players_frame1.iterrows():
            wr1_nfl_id = wr1_row['nfl_id']
            wr1_y_frame1 = wr1_row['y']
            wr1_side = get_player_side_of_qb(wr1_y_frame1, qb_y_frame1)

            if cb_side == wr1_side:
                # 6. A matching offensive player (WR1) is found

                # 6a. Get WR1's x coordinate at frame_id=1 and max_frame_id
                wr1_x_frame1_data = play_df[
                    (play_df['nfl_id'] == wr1_nfl_id) &
                    (play_df['frame_id'] == 1)
                ]
                wr1_x_max_frame_data = play_df[
                    (play_df['nfl_id'] == wr1_nfl_id) &
                    (play_df['frame_id'] == max_frame_id)
                ]

                if wr1_x_frame1_data.empty or wr1_x_max_frame_data.empty:
                    continue # Skip if WR1 data is incomplete

                wr1_x_frame1_val = wr1_x_frame1_data['x'].iloc[0]
                wr1_x_max_frame_val = wr1_x_max_frame_data['x'].iloc[0]

                # 6b. Calculate the change in the WR1's x coordinate
                x_change_wr1 = wr1_x_max_frame_val - wr1_x_frame1_val

                # 6c & 6d. Get CB's and WR1's x and y coordinates at max_frame_id
                cb_max_frame_data = play_df[
                    (play_df['nfl_id'] == cb_nfl_id) &
                    (play_df['frame_id'] == max_frame_id)
                ]
                if cb_max_frame_data.empty:
                    continue # Skip if CB data is incomplete at max frame
                cb_x_max = cb_max_frame_data['x'].iloc[0]
                cb_y_max = cb_max_frame_data['y'].iloc[0]

                wr1_x_max = wr1_x_max_frame_data['x'].iloc[0] # Already retrieved
                wr1_y_max = wr1_x_max_frame_data['y'].iloc[0]

                # 6e. Calculate the Euclidean distance between CB and WR1 at max_frame_id
                distance = np.sqrt((cb_x_max - wr1_x_max)**2 + (cb_y_max - wr1_y_max)**2)

                # 6f. Evaluate two scenarios:
                # Scenario 1
                if x_change_wr1 >= 10 and distance <= 3:
                    return True
                # Scenario 2
                if x_change_wr1 < 10 and distance >= 5:
                    return True
    # 8. If the loop completes without satisfying any of the conditions, return False
    return False

print("Function 'check_condition1' defined successfully.")


import numpy as np

def check_condition2(play_df):
    """
    Evaluates specific scenarios for defensive players (FS or SS) and offensive receivers with receiver_number = 2.

    Args:
        play_df (pd.DataFrame): DataFrame containing play data for a single game_id and play_id.

    Returns:
        bool: True if either scenario is met for any FS/SS-WR2 pair, False otherwise.
    """
    # 2. Get QB coordinates at frame_id=1
    qb_x_frame1, qb_y_frame1 = get_qb_coordinates_at_frame1(play_df)
    if qb_x_frame1 is None or qb_y_frame1 is None:
        return False

    # Get max_frame_id for the play
    max_frame_id = play_df['frame_id'].max()

    # 3. Filter defensive players with player_position == 'FS' or 'SS' at frame_id=1
    defensive_fs_ss_frame1 = play_df[
        (play_df['player_side'] == 'Defense') &
        (play_df['player_position'].isin(['FS', 'SS'])) &
        (play_df['frame_id'] == 1)
    ]
    if defensive_fs_ss_frame1.empty:
        return False

    # 4. Filter offensive players with receiver_number == 2 at frame_id=1
    receiver2_players_frame1 = play_df[
        (play_df['player_side'] == 'Offense') &
        (play_df['receiver_number'] == 2) &
        (play_df['frame_id'] == 1)
    ]
    if receiver2_players_frame1.empty:
        return False

    # Iterate through each defensive FS/SS player
    for _, def_row_frame1 in defensive_fs_ss_frame1.iterrows():
        def_nfl_id = def_row_frame1['nfl_id']
        def_y_frame1 = def_row_frame1['y']
        def_side = get_player_side_of_qb(def_y_frame1, qb_y_frame1)

        # Iterate through each receiver_number=2 player
        for _, rec2_row_frame1 in receiver2_players_frame1.iterrows():
            rec2_nfl_id = rec2_row_frame1['nfl_id']
            rec2_y_frame1 = rec2_row_frame1['y']
            rec2_side = get_player_side_of_qb(rec2_y_frame1, qb_y_frame1)

            # 8. If a FS/SS player and a receiver_number=2 player are on the same side of the QB
            if def_side == rec2_side:
                # 8a. Get receiver's x coordinate at frame_id=1 and max_frame_id
                rec2_x_frame1_data = play_df[
                    (play_df['nfl_id'] == rec2_nfl_id) &
                    (play_df['frame_id'] == 1)
                ]
                rec2_x_max_frame_data = play_df[
                    (play_df['nfl_id'] == rec2_nfl_id) &
                    (play_df['frame_id'] == max_frame_id)
                ]

                if rec2_x_frame1_data.empty or rec2_x_max_frame_data.empty:
                    continue  # Skip if receiver data is incomplete

                rec2_x_frame1_val = rec2_x_frame1_data['x'].iloc[0]
                rec2_x_max_frame_val = rec2_x_max_frame_data['x'].iloc[0]

                # 8b. Calculate the change in the receiver's x coordinate
                x_change_receiver = rec2_x_max_frame_val - rec2_x_frame1_val

                # 8c. Get FS/SS player's and the receiver's x and y coordinates at max_frame_id
                def_max_frame_data = play_df[
                    (play_df['nfl_id'] == def_nfl_id) &
                    (play_df['frame_id'] == max_frame_id)
                ]
                if def_max_frame_data.empty:
                    continue  # Skip if defensive player data is incomplete at max frame

                def_x_max = def_max_frame_data['x'].iloc[0]
                def_y_max = def_max_frame_data['y'].iloc[0]

                rec2_x_max = rec2_x_max_frame_data['x'].iloc[0] # Already retrieved
                rec2_y_max = rec2_x_max_frame_data['y'].iloc[0]

                # 8d. Calculate the Euclidean distance between FS/SS player and receiver at max_frame_id
                distance = np.sqrt((def_x_max - rec2_x_max)**2 + (def_y_max - rec2_y_max)**2)

                # 8e. Evaluate two scenarios:
                # Scenario 1
                if x_change_receiver >= 10 and distance <= 3:
                    return True
                # Scenario 2
                if x_change_receiver < 10 and distance >= 5:
                    return True

    # 9. If the loop completes without satisfying any of the conditions, return False
    return False

print("Function 'check_condition2' defined successfully with new logic.")


unique_plays_4_zone = All_4_zone[['game_id', 'play_id']].drop_duplicates()

quarters_results = []

for index, row in unique_plays_4_zone.iterrows():
    game_id = row['game_id']
    play_id = row['play_id']

    # Filter All_4_zone for the current game_id and play_id
    current_play_df = All_4_zone[(All_4_zone['game_id'] == game_id) & (All_4_zone['play_id'] == play_id)]

    if current_play_df.empty:
        continue

    # Apply condition1 and condition2
    condition1_status = check_condition1(current_play_df)
    condition2_status = check_condition2(current_play_df)

    # The 'Quarters' column should be True if *either* condition is met
    quarters_status = condition1_status or condition2_status

    quarters_results.append({
        'game_id': game_id,
        'play_id': play_id,
        'Quarters': quarters_status
    })

# Convert results to a DataFrame
quarters_df = pd.DataFrame(quarters_results)

print(f"Processed {len(quarters_df)} unique plays with updated conditions.")
display(quarters_df.head())


if 'Quarters' in All_4_zone.columns:
    All_4_zone = All_4_zone.drop(columns=['Quarters'])

All_4_zone = pd.merge(All_4_zone, quarters_df, on=['game_id', 'play_id'], how='left')

print("Head of All_4_zone after adding 'Quarters' column:")
display(All_4_zone.head())

print("\nValue counts for the 'Quarters' column:")
display(All_4_zone['Quarters'].value_counts())


import pandas as pd

def get_qb_y_at_frame1(play_df):
    """
    Retrieves the y-coordinate of the Quarterback at frame_id=1 from a given play DataFrame.

    Args:
        play_df (pd.DataFrame): DataFrame containing play data.

    Returns:
        float: The y-coordinate of the QB at frame 1, or None if no QB is found at frame 1.
    """
    qb_at_frame1 = play_df[(play_df['player_position'] == 'QB') & (play_df['frame_id'] == 1)]
    if not qb_at_frame1.empty:
        return qb_at_frame1['y'].iloc[0]
    return None

print("Function 'get_qb_y_at_frame1' defined successfully.")


import numpy as np

def check_condition2_scenario1_only(play_df):
    """
    Checks if Scenario 1 of Condition 2 is met for FS/SS and receiver_number=2 offensive players.

    Args:
        play_df (pd.DataFrame): DataFrame containing play data for a single game_id and play_id.

    Returns:
        bool: True if Scenario 1 is met for any FS/SS-WR2 pair, False otherwise.
    """
    # 2. Get QB coordinates at frame_id=1
    qb_x_frame1, qb_y_frame1 = get_qb_coordinates_at_frame1(play_df)
    if qb_x_frame1 is None or qb_y_frame1 is None:
        return False

    # Get max_frame_id for the play
    max_frame_id = play_df['frame_id'].max()

    # 4. Filter defensive players with player_position == 'FS' or 'SS' at frame_id=1
    defensive_fs_ss_frame1 = play_df[
        (play_df['player_side'] == 'Defense') &
        (play_df['player_position'].isin(['FS', 'SS'])) &
        (play_df['frame_id'] == 1)
    ]
    if defensive_fs_ss_frame1.empty:
        return False

    # 5. Filter offensive players with receiver_number == 2 at frame_id=1
    receiver2_players_frame1 = play_df[
        (play_df['player_side'] == 'Offense') &
        (play_df['receiver_number'] == 2) &
        (play_df['frame_id'] == 1)
    ]
    if receiver2_players_frame1.empty:
        return False

    # 6. Iterate through each defensive FS/SS player
    for _, def_row_frame1 in defensive_fs_ss_frame1.iterrows():
        def_nfl_id = def_row_frame1['nfl_id']
        def_y_frame1 = def_row_frame1['y']
        def_side = get_player_side_of_qb(def_y_frame1, qb_y_frame1)

        # Iterate through each receiver_number=2 player
        for _, rec2_row_frame1 in receiver2_players_frame1.iterrows():
            rec2_nfl_id = rec2_row_frame1['nfl_id']
            rec2_y_frame1 = rec2_row_frame1['y']
            rec2_side = get_player_side_of_qb(rec2_y_frame1, qb_y_frame1)

            # 8. If an FS/SS player and a receiver_number=2 player are on the same side of the QB
            if def_side == rec2_side:
                # 8a. Get receiver's x coordinate at frame_id=1 and max_frame_id
                rec2_x_frame1_data = play_df[
                    (play_df['nfl_id'] == rec2_nfl_id) &
                    (play_df['frame_id'] == 1)
                ]
                rec2_x_max_frame_data = play_df[
                    (play_df['nfl_id'] == rec2_nfl_id) &
                    (play_df['frame_id'] == max_frame_id)
                ]

                if rec2_x_frame1_data.empty or rec2_x_max_frame_data.empty:
                    continue  # Skip if receiver data is incomplete

                rec2_x_frame1_val = rec2_x_frame1_data['x'].iloc[0]
                rec2_x_max_frame_val = rec2_x_max_frame_data['x'].iloc[0]

                # 8b. Calculate the change in the receiver's x coordinate
                x_change_receiver = rec2_x_max_frame_val - rec2_x_frame1_val

                # 8c. Get FS/SS player's and the receiver's x and y coordinates at max_frame_id
                def_max_frame_data = play_df[
                    (play_df['nfl_id'] == def_nfl_id) &
                    (play_df['frame_id'] == max_frame_id)
                ]
                if def_max_frame_data.empty:
                    continue  # Skip if defensive player data is incomplete at max frame

                def_x_max = def_max_frame_data['x'].iloc[0]
                def_y_max = def_max_frame_data['y'].iloc[0]

                rec2_x_max = rec2_x_max_frame_data['x'].iloc[0] # Already retrieved
                rec2_y_max = rec2_x_max_frame_data['y'].iloc[0]

                # 8d. Calculate the Euclidean distance between FS/SS player and receiver at max_frame_id
                distance = np.sqrt((def_x_max - rec2_x_max)**2 + (def_y_max - rec2_y_max)**2)

                # 8e. Check Scenario 1 conditions
                if x_change_receiver >= 10 and distance <= 5:
                    return True

    # 9. If the loop completes without satisfying any of the conditions, return False
    return False

print("Function 'check_condition2_scenario1_only' defined successfully.")


import numpy as np

def check_condition3(play_df):
    """
    Evaluates three scenarios for 'OLB' players and 'receiver_number = 2' offensive players.

    Args:
        play_df (pd.DataFrame): DataFrame containing play data for a single game_id and play_id.

    Returns:
        bool: True if any of the three scenarios are met, False otherwise.
    """
    # Get QB y-coordinate at frame_id=1 for boundary checks
    qb_y_frame1 = get_qb_y_at_frame1(play_df)
    if qb_y_frame1 is None:
        return False

    # Get max_frame_id for the play
    max_frame_id = play_df['frame_id'].max()

    # Filter offensive players with receiver_number == 2 at frame_id=1
    receiver2_players_frame1 = play_df[
        (play_df['player_side'] == 'Offense') &
        (play_df['receiver_number'] == 2) &
        (play_df['frame_id'] == 1)
    ]
    if receiver2_players_frame1.empty:
        # Check if Scenario 1 of Condition 2 is met even if no receiver_number=2 is found
        return check_condition2_scenario1_only(play_df)

    # Filter defensive players with player_position == 'OLB' at frame_id=1
    olb_players_frame1 = play_df[
        (play_df['player_side'] == 'Defense') &
        (play_df['player_position'] == 'OLB') &
        (play_df['frame_id'] == 1)
    ]

    # Filter defensive players with player_position == 'MLB' or 'ILB' at frame_id=1
    mlb_ilb_players_frame1 = play_df[
        (play_df['player_side'] == 'Defense') &
        (play_df['player_position'].isin(['MLB', 'ILB'])) &
        (play_df['frame_id'] == 1)
    ]

    for _, rec2_row_frame1 in receiver2_players_frame1.iterrows():
        rec2_nfl_id = rec2_row_frame1['nfl_id']
        rec2_y_frame1 = rec2_row_frame1['y']

        # Get receiver's x coordinate at frame_id=1 and max_frame_id
        rec2_x_frame1_data = play_df[
            (play_df['nfl_id'] == rec2_nfl_id) &
            (play_df['frame_id'] == 1)
        ]
        rec2_x_max_frame_data = play_df[
            (play_df['nfl_id'] == rec2_nfl_id) &
            (play_df['frame_id'] == max_frame_id)
        ]

        if rec2_x_frame1_data.empty or rec2_x_max_frame_data.empty:
            continue # Skip if receiver data is incomplete

        rec2_x_frame1_val = rec2_x_frame1_data['x'].iloc[0]
        rec2_x_max_frame_val = rec2_x_max_frame_data['x'].iloc[0]

        x_change_receiver = rec2_x_max_frame_val - rec2_x_frame1_val

        # Get receiver's y coordinate at max_frame_id
        rec2_y_max_frame_val = rec2_x_max_frame_data['y'].iloc[0]

        # Scenario 1: Offensive player's x-coordinate increases by < 10 yards, runs towards boundary, < 10 yards from OLB
        if x_change_receiver < 10:
            # Determine if running towards boundary relative to QB's initial y-position
            is_towards_boundary = False
            if abs(rec2_y_max_frame_val - qb_y_frame1) > abs(rec2_y_frame1 - qb_y_frame1):
                is_towards_boundary = True

            if is_towards_boundary:
                for _, olb_row_frame1 in olb_players_frame1.iterrows():
                    olb_nfl_id = olb_row_frame1['nfl_id']
                    olb_max_frame_data = play_df[
                        (play_df['nfl_id'] == olb_nfl_id) &
                        (play_df['frame_id'] == max_frame_id)
                    ]
                    if olb_max_frame_data.empty: continue

                    olb_x_max = olb_max_frame_data['x'].iloc[0]
                    olb_y_max = olb_max_frame_data['y'].iloc[0]

                    distance_to_olb = np.sqrt((rec2_x_max_frame_val - olb_x_max)**2 + (rec2_y_max_frame_val - olb_y_max)**2)
                    if distance_to_olb < 10:
                        return True

        # Scenario 2: Offensive player's x-coordinate increases by < 10 yards, runs away from boundary, < 10 yards from MLB or ILB
        if x_change_receiver < 10:
            is_away_from_boundary = False
            if abs(rec2_y_max_frame_val - qb_y_frame1) < abs(rec2_y_frame1 - qb_y_frame1):
                 is_away_from_boundary = True

            if is_away_from_boundary:
                for _, mlb_ilb_row_frame1 in mlb_ilb_players_frame1.iterrows():
                    mlb_ilb_nfl_id = mlb_ilb_row_frame1['nfl_id']
                    mlb_ilb_max_frame_data = play_df[
                        (play_df['nfl_id'] == mlb_ilb_nfl_id) &
                        (play_df['frame_id'] == max_frame_id)
                    ]
                    if mlb_ilb_max_frame_data.empty: continue

                    mlb_ilb_x_max = mlb_ilb_max_frame_data['x'].iloc[0]
                    mlb_ilb_y_max = mlb_ilb_max_frame_data['y'].iloc[0]

                    distance_to_mlb_ilb = np.sqrt((rec2_x_max_frame_val - mlb_ilb_x_max)**2 + (rec2_y_max_frame_val - mlb_ilb_y_max)**2)
                    if distance_to_mlb_ilb < 10:
                        return True

    # Scenario 3: Scenario 1 of the original Condition 2 is met
    # This is handled by a separate function and checked if receiver_number=2 players were not found
    # However, it can also be part of the overall logic if any receiver_number=2 exists.
    # To avoid double-checking if no rec2 players were found, we check it here too.
    if check_condition2_scenario1_only(play_df):
        return True

    return False

print("Function 'check_condition3' defined successfully.")


unique_plays_to_recheck = All_4_zone[All_4_zone['Quarters'] == True][['game_id', 'play_id']].drop_duplicates()

updated_quarters_results = []

for index, row in unique_plays_to_recheck.iterrows():
    game_id = row['game_id']
    play_id = row['play_id']

    current_play_df = All_4_zone[(All_4_zone['game_id'] == game_id) & (All_4_zone['play_id'] == play_id)]

    if current_play_df.empty:
        continue

    condition3_status = check_condition3(current_play_df)

    updated_quarters_results.append({
        'game_id': game_id,
        'play_id': play_id,
        'New_Quarters_Status': condition3_status
    })

updated_quarters_df = pd.DataFrame(updated_quarters_results)

# Merge the new statuses back into All_4_zone
# We'll first set 'Quarters' to False for plays that had it True and now fail Condition3
All_4_zone = pd.merge(All_4_zone, updated_quarters_df, on=['game_id', 'play_id'], how='left')

# Update 'Quarters' column where New_Quarters_Status is False
# and the original Quarters was True (implicitly handled by unique_plays_to_recheck)
All_4_zone.loc[All_4_zone['New_Quarters_Status'] == False, 'Quarters'] = False

# Drop the temporary 'New_Quarters_Status' column
All_4_zone = All_4_zone.drop(columns=['New_Quarters_Status'])

print("Head of All_4_zone after re-evaluating 'Quarters' column:")
display(All_4_zone.head())

print("\nValue counts for the updated 'Quarters' column:")
display(All_4_zone['Quarters'].value_counts())


All_4_zone_quarters = All_4_zone[All_4_zone['Quarters'] == True].copy()
display(All_4_zone_quarters.head())
print(f"Shape of All_4_zone_quarters: {All_4_zone_quarters.shape}")


unique_plays_in_quarters = All_4_zone_quarters[['game_id', 'play_id']].drop_duplicates()
num_unique_plays = len(unique_plays_in_quarters)

print(f"There are {num_unique_plays} different plays in the All_4_zone_quarters DataFrame.")


import numpy as np

All_4_zone_quarters['E Distance'] = np.sqrt(
    (All_4_zone_quarters['ball_land_x'] - All_4_zone_quarters['x (after pass)'])**2 +
    (All_4_zone_quarters['ball_land_y'] - All_4_zone_quarters['y (after pass)'])**2
)

display(All_4_zone_quarters.head())
display(All_4_zone_quarters)


original_max_rows = pd.options.display.max_rows
pd.options.display.max_rows = 200
display(All_4_zone_quarters.head(200))
pd.options.display.max_rows = original_max_rows


Quarters_last_movement = All_4_zone_quarters.dropna(subset=['E Distance']).groupby(['game_id', 'play_id', 'nfl_id']).apply(lambda x: x.loc[x['frame_id'].idxmax()]).reset_index(drop=True)

display(Quarters_last_movement.head())
print(f"Shape of Quarters_last_movement: {Quarters_last_movement.shape}")


Quarters_last_movement_filtered = Quarters_last_movement[Quarters_last_movement['pass_result'] != 'I']
display(Quarters_last_movement_filtered.head())
print(f"Shape of Quarters_last_movement_filtered: {Quarters_last_movement_filtered.shape}")


unique_plays_in_all_4_zone = All_4_zone[['game_id', 'play_id']].drop_duplicates()
num_unique_plays_all_4_zone = len(unique_plays_in_all_4_zone)

print(f"There are {num_unique_plays_all_4_zone} different plays in the All_4_zone DataFrame.")


Min_distance = Quarters_last_movement_filtered.groupby(['game_id', 'play_id', 'yards_gained'])['E Distance'].min().reset_index()

display(Min_distance.head())
print(f"Shape of Min_distance: {Min_distance.shape}")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Rename 'E Distance' in Min_distance to avoid confusion during merge and indicate it's the play's overall minimum
Min_distance_renamed = Min_distance.rename(columns={'E Distance': 'play_min_E_Distance'})

# Merge Min_distance (containing play's overall min E Distance) with Quarters_last_movement_filtered
# This brings the route information for all players in the filtered movement data.
merged_data_with_play_min_e_distance = pd.merge(
    Quarters_last_movement_filtered,
    Min_distance_renamed,
    on=['game_id', 'play_id'],
    how='inner'
)

# Now, filter to keep only the rows where the player's E Distance matches the overall play's minimum E Distance.
# This identifies the route(s) associated with the closest reception(s) in each play.
routes_at_play_min_e_distance = merged_data_with_play_min_e_distance[
    merged_data_with_play_min_e_distance['E Distance'] == merged_data_with_play_min_e_distance['play_min_E_Distance']
].copy()

# Group by route_of_targeted_receiver and calculate the average of these 'play_min_E_Distance' values
average_play_min_e_distance_by_route = routes_at_play_min_e_distance.groupby('route_of_targeted_receiver')['play_min_E_Distance'].mean().sort_values(ascending=False).reset_index()

display(average_play_min_e_distance_by_route)


merged_route_metrics = average_play_min_e_distance_by_route.copy()

# Calculate the amount of unique plays for each route
plays_per_route = Quarters_last_movement_filtered.groupby('route_of_targeted_receiver')['play_id'].nunique().reset_index(name='amount_of_plays')

# Merge this data into merged_route_metrics
merged_route_metrics = pd.merge(merged_route_metrics, plays_per_route, on='route_of_targeted_receiver', how='left')

# Calculate 'total_yards' column, assuming 'yards_gained' is a valid metric for each play within Quarters_last_movement_filtered
# To get a meaningful 'yards_gained' for this calculation, I'll take the mean of 'yards_gained' per route
# from Quarters_last_movement_filtered, and then multiply by amount_of_plays.
yards_gained_per_route = Quarters_last_movement_filtered.groupby('route_of_targeted_receiver')['yards_gained'].mean().reset_index(name='avg_yards_gained_per_play')
merged_route_metrics = pd.merge(merged_route_metrics, yards_gained_per_route, on='route_of_targeted_receiver', how='left')
merged_route_metrics['total_yards'] = merged_route_metrics['amount_of_plays'] * merged_route_metrics['avg_yards_gained_per_play']

print("Number of plays included for each route:")
display(merged_route_metrics[['route_of_targeted_receiver', 'amount_of_plays', 'avg_yards_gained_per_play', 'total_yards']])


plt.figure(figsize=(16, 9))
sns.barplot(x='route_of_targeted_receiver', y='play_min_E_Distance', data=average_play_min_e_distance_by_route, palette='viridis')
plt.title('Average Minimum Distance from Route of Targeted Receiver per Play')
plt.xlabel('Route of Targeted Receiver')
plt.ylabel('Minimum Distance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

