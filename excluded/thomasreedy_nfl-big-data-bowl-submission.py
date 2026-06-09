# PREPARE THE DATA

# import necessary libraries
import pandas as pd
import numpy as np 
from pathlib import Path
import os
import matplotlib.pyplot as plt

# store output from all games
all_file_results = []

# get the correct paths
data_dir = Path("/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final")
train_path = data_dir / "train" #next(data_dir.glob("*/train"))

import warnings
warnings.filterwarnings('ignore')

# go through all of the files in the data folder
for i in range(1,19):
    # zero-pad to 2 digits
    week = f"{i:02d}"  

    # load in the files needed
    input_df = pd.read_csv(os.path.join(train_path,  f"input_2023_w{week}.csv"))
    output_df = pd.read_csv(os.path.join(train_path,  f"output_2023_w{week}.csv"))

    # add unique names
    #names.append(input_df[['nfl_id', 'player_name']].drop_duplicates())

    # I only care about target players and their data for merging
    input_df = input_df[input_df['player_to_predict'] == True]

    # coerce differences in formatting
    for df in [input_df, output_df]:
        for k in ["game_id", "play_id"]:
            if k in df.columns:
                df.loc[:, k] = pd.to_numeric(df[k], errors="coerce").astype("Int64")

    # get the place the ball lands
    ball_land = (
        input_df[["game_id", "play_id", "ball_land_x", "ball_land_y"]] # select landing cols
                .dropna(subset=["ball_land_x", "ball_land_y"])         # require both coords
                .drop_duplicates(["game_id", "play_id"])               # one landing per play
    )

    # get the direction and player position
    extras = input_df[['game_id', 'play_id', 'nfl_id', 'player_name', 'player_position', 'play_direction']] \
        .groupby(['game_id', 'play_id', 'nfl_id']).max().reset_index()

    # make a frame_id for the total play
    input_df['frame_id_adj'] = input_df['frame_id']
    
    ### focuses on defense
    # selected data from input_df I want to merge, basically the predicted player for a single game
    defensive_merge = input_df[input_df['player_role'] == 'Defensive Coverage'] \
        [['game_id','play_id', 'nfl_id', 'frame_id', 'x', 'y', 'frame_id_adj']] \
            .groupby(['game_id', 'play_id', 'nfl_id', 'frame_id']).max().reset_index() \
            .groupby(['game_id', 'play_id', 'nfl_id']) \
            .tail(17)

    # gets the maximum frame number for input files
    input_max_frames = defensive_merge[['game_id','play_id', 'nfl_id', 'frame_id_adj']] \
        .groupby(['game_id', 'play_id', 'nfl_id']).max().reset_index()

    # adds the maximum frame number to the output dataframe
    output_mod = pd.merge(output_df, input_max_frames, \
                        on = ['game_id', 'play_id', 'nfl_id'], how = 'inner', suffixes=('', '_max'))

    # adjusts the frame_id to be continuous across the play
    output_mod['frame_id_adj'] = output_mod['frame_id_adj'] + output_mod['frame_id']

    # put it all together
    defensive_combined = pd.concat([defensive_merge, output_mod], ignore_index=True)

    ### focuses on offense
    # selected data from input_df I want to merge, basically the predicted player for a single game
    offensive_merge = input_df[input_df['player_role'] == 'Targeted Receiver'] \
        [['game_id','play_id', 'nfl_id', 'frame_id', 'x', 'y', 'frame_id_adj']] \
            .groupby(['game_id', 'play_id', 'nfl_id', 'frame_id']).max().reset_index() \
            .groupby(['game_id', 'play_id', 'nfl_id']) \
            .tail(17)

    # gets the maximum frame number for input files
    input_max_frames = offensive_merge[['game_id','play_id', 'nfl_id', 'frame_id_adj']] \
        .groupby(['game_id', 'play_id', 'nfl_id']).max().reset_index()

    # adds the maximum frame number to the output dataframe
    output_mod = pd.merge(output_df, input_max_frames, \
                        on = ['game_id', 'play_id', 'nfl_id'], how = 'inner', suffixes=('', '_max'))

    # adjusts the frame_id to be continuous across the play
    output_mod['frame_id_adj'] = output_mod['frame_id_adj'] + output_mod['frame_id']

    # put it all together
    offensive_combined = pd.concat([offensive_merge, output_mod], ignore_index=True)

    ## merge offenseive and defensive data together
    combined = pd.merge(defensive_combined, offensive_combined[['game_id', 'play_id', 'frame_id_adj', 'x', 'y']], 
                    on=['game_id', 'play_id', 'frame_id_adj'], suffixes=('_def', '_off'))
    combined = pd.merge(combined, ball_land, on=['game_id', 'play_id'], how='left')

    # add in extra data
    #combined = pd.merge(combined, extras, on=['game_id', 'play_id', 'nfl_id'], how='left')


    # make sure all the values are sorted
    combined = combined.sort_values(["game_id", "play_id", "nfl_id", "frame_id_adj"])

    # get the speed
    combined['dx'] = combined['x_def'].diff()
    combined['dy'] = combined['y_def'].diff()
    combined['speed'] = np.sqrt(combined['dx']**2 + combined['dy']**2)

    # get the acceleration
    combined['acceleration'] = combined['speed'].diff() #np.sqrt(combined['ddx']**2 + combined['ddy']**2)

    # get the direction as an angle that the player is traveling
    combined['dir_rad'] = np.arctan2(combined['dy'], combined['dx'])
    combined['dir_degree'] = (np.degrees(combined['dir_rad']) + 360) % 360

    # get the direction from player to ball landing spot
    combined['to_ball_dir'] = np.degrees(np.arctan2(combined['ball_land_y'] - combined['y_def'], 
                                                    combined['ball_land_x'] - combined['x_def']))
    combined['to_ball_dir'] = (combined['to_ball_dir'] + 360) % 360

    # get the distance between defender and receiver
    combined['dist_def_off'] = np.sqrt((combined['x_def'] - combined['x_off'])**2 + 
                                    (combined['y_def'] - combined['y_off'])**2)

    # make vector from defender to ball landing spot
    combined['to_rx'] = combined['ball_land_x'] - combined['x_def']
    combined['to_ry'] = combined['ball_land_y'] - combined['y_def']

    # unit vector towards ball landing spot
    combined['dist_to_ball'] = np.sqrt(combined['to_rx']**2 + combined['to_ry']**2)
    combined['ux'] = combined['to_rx'] / combined['dist_to_ball']
    combined['uy'] = combined['to_ry'] / combined['dist_to_ball']

    # accel vector components
    combined['ax'] = combined['acceleration'] * np.cos(combined['dir_rad'])
    combined['ay'] = combined['acceleration'] * np.sin(combined['dir_rad'])

    # finally, a dot product
    combined['accel_towards_ball'] = combined['ax'] * combined['ux'] + combined['ay'] * combined['uy']

    combined_final = (
        combined
            #.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id']) # should already be sorted
            .groupby(['game_id', 'play_id', 'nfl_id'])
            .apply(lambda g: g.iloc[2:])  # drop first 2 rows in each group
            .reset_index(drop=True)
    )

    input_max_frames.rename(columns={'frame_id_adj': 'rel_frame_id'}, inplace=True)

    # get the max number of frame for the input => needed when a throw is made in < 1.7 seconds
    combined_f = pd.merge(combined_final, input_max_frames, on=['game_id', 'play_id'], how='left')
    # type => 0 is before throw, 1 is after throw
    combined_f['rel_frame_id'] = combined_f['frame_id_adj'] - combined_f['rel_frame_id']

    # create the angle difference between player direction and ball direction
    combined_f['angle_dif'] = np.abs(combined_f['dir_degree'] - combined_f['to_ball_dir'])
    combined_f['angle_dif'] = combined_f['angle_dif'].apply(lambda x: min(x, 360 - x))
    # create tolerance based on distance from receiver
    combined_f['angle_tol'] = np.clip(2 * combined_f['dist_def_off'] + 5, 10, 45)

    combined_f['reaction'] = ((combined_f['accel_towards_ball'] > 0) & 
                            (combined_f['angle_dif'] < combined_f['angle_tol'])).astype(int)

    reaction_results = combined_f[combined_f['reaction'] == 1].groupby(['game_id', 'play_id', 'nfl_id_x']) \
        [['reaction', 'rel_frame_id']].min().reset_index()
    lacking = combined_f[combined_f['reaction'] == 0].groupby(['game_id', 'play_id', 'nfl_id_x']) \
        [['reaction', 'rel_frame_id']].max().reset_index()

    # combine
    final_results = pd.concat([reaction_results, lacking]).drop_duplicates(subset=["play_id", "nfl_id_x"])

        # for data insights later... find the distance, angle from receiver at time of throw
    zero_frame = combined_f[combined_f['rel_frame_id'] == 0][['game_id', 'play_id', 'nfl_id_x',
                                                            'dist_def_off', 'dir_degree']]
    #zero_frame = combined_f.groupby(['game_id', 'play_id', 'nfl_id_x']).first().reset_index() \
    #    [['game_id', 'play_id', 'nfl_id_x','dist_def_off', 'dir_degree']]
    alt_final_results = pd.merge(final_results, zero_frame, on=['game_id', 'play_id', 'nfl_id_x'], how='left')
    #alt_final_results.rename(columns={'nfl_id_x': 'nfl_id'}, inplace=True)

    # add in extra data
    #alt_final_results = pd.merge(alt_final_results, extras, on=['game_id', 'play_id', 'nfl_id'], how='left')
    alt_final_results2 = pd.merge(
        alt_final_results, extras, 
        left_on=['game_id', 'play_id', 'nfl_id_x'],
        right_on=['game_id', 'play_id', 'nfl_id'],
        how='left'
    )

    # cherry on top, combine from prior
    all_file_results.append(alt_final_results2)
    # print(f"Week {week} processed")
#print("All weeks processed")

# concatenate files
df = pd.concat(all_file_results, ignore_index=True)

# alter the dir_degree based on play direction
df['dir_degree'] = np.where(
    df['play_direction'] == 'left',
    (df['dir_degree'] + 180) % 360,
    df['dir_degree']
)


# CALCULATE SDRI

# function to compute first wave of stats for each defender
def compute_stats(group):
    rel = group['rel_frame_id']
    react = group['reaction']
    
    return pd.Series({
        # counts
        'count_entries': len(rel),
        'anticipation': ((react == 1) & (rel <= 0)).sum(),
        'reaction_count': ((react == 1) & (rel > 0)).sum(),
        'nr_count': (react == 0).sum(),

        # average reaction times
        'avg_reaction_time': rel[react == 1].mean(),
        'avg_reaction_time_valid': rel[(react == 1) & (rel > 0)].mean(),
        'avg_nr_time': rel[react == 0].mean()
    })

# create stats
final = df.groupby(['nfl_id', 'player_name', 'player_position']).apply(compute_stats).reset_index()

# only select qualified defenders with enough entries
qualified_final = final[final['reaction_count'] >= 8].copy()

# adding a penalty of 1.5x the throw time for no reactions
qualified_final['avg_nr_time'] = qualified_final['avg_nr_time'].fillna(0)
qualified_final['nr_penalty'] = qualified_final['avg_nr_time'] * 1.5

# get reaction and anticipation percentages
qualified_final['reaction_pct'] = qualified_final['reaction_count'] / (qualified_final['count_entries'] - qualified_final['nr_count']) 
qualified_final['anticipation_pct'] = qualified_final['anticipation'] / (qualified_final['count_entries'] - qualified_final['nr_count'])

# calculate a final score
# calculated as mean of valid reaction times and penalized no-reaction times combined
qualified_final['reaction_score'] = qualified_final['avg_reaction_time_valid'] + qualified_final['nr_penalty'] * (1 - (qualified_final['nr_count'] / qualified_final['count_entries']))


# CREATE HEATMAPS

# convert polar back to cartesian
df["theta_rad"] = np.deg2rad(df["dir_degree"])
df["x"] = df["dist_def_off"] * np.cos(df["theta_rad"])
df["y"] = df["dist_def_off"] * np.sin(df["theta_rad"])

# define grid resolution
xbins = 50
ybins = 50

max_range = max(df["x"].abs().max(), df["y"].abs().max())
x_edges = np.linspace(-max_range, max_range, xbins + 1)
y_edges = np.linspace(-max_range, max_range, ybins + 1)

# bin the data
df["x_bin"] = np.digitize(df["x"], x_edges) - 1
df["y_bin"] = np.digitize(df["y"], y_edges) - 1

extent = [-max_range, max_range, -max_range, max_range]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ------------------------------
# Top row: Frequency
# ------------------------------
# Reaction = 0
heatmap_freq0 = df[df['reaction'] == 0].groupby(['y_bin', 'x_bin']).size().unstack().fillna(0)
im00 = axes[0, 0].imshow(heatmap_freq0, origin='lower', aspect='equal', extent=extent)
axes[0, 0].set_title("No Reaction Frequency Distribution")
axes[0, 0].set_xlabel("X Coordinate Relative to Receiver")
axes[0, 0].set_ylabel("Y Coordinate Relative to Receiver")
fig.colorbar(im00, ax=axes[0, 0])

# Reaction = 1
heatmap_freq1 = df[df['reaction'] == 1].groupby(['y_bin', 'x_bin']).size().unstack().fillna(0)
im01 = axes[0, 1].imshow(heatmap_freq1, origin='lower', aspect='equal', extent=extent)
axes[0, 1].set_title("Reaction Frequency Distribution")
axes[0, 1].set_xlabel("X Coordinate Relative to Receiver")
axes[0, 1].set_ylabel("Y Coordinate Relative to Receiver")
fig.colorbar(im01, ax=axes[0, 1])

# ------------------------------
# Bottom row: Average score
# ------------------------------
# Reaction = 0
heatmap_score0 = df[(df['reaction'] == 0) & (df['rel_frame_id'] > -14)].groupby(['y_bin','x_bin'])['rel_frame_id'].mean().unstack()
im10 = axes[1, 0].imshow(heatmap_score0, origin='lower', aspect='equal', extent=extent)
axes[1, 0].set_title("Average Frame Length of Passes with No Reaction")
axes[1, 0].set_xlabel("X Coordinate Relative to Receiver")
axes[1, 0].set_ylabel("Y Coordinate Relative to Receiver")
fig.colorbar(im10, ax=axes[1, 0])

# Reaction = 1
heatmap_score1 = df[(df['reaction'] == 1) & (df['rel_frame_id'] > -14)].groupby(['y_bin','x_bin'])['rel_frame_id'].mean().unstack()
im11 = axes[1, 1].imshow(heatmap_score1, origin='lower', aspect='equal', extent=extent)
axes[1, 1].set_title("Average Reaction Time of Passes with Reaction")
axes[1, 1].set_xlabel("X Coordinate Relative to Receiver")
axes[1, 1].set_ylabel("Y Coordinate Relative to Receiver")
fig.colorbar(im11, ax=axes[1, 1])

# Optional crosshairs at 0,0 for all
for ax_row in axes:
    for ax in ax_row:
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)

#plt.title('Distribution of Defender Locations and Average Reaction Times', fontsize=14)
fig.suptitle("Distribution of Defender Locations and Average Reaction Times", fontsize=16)

plt.tight_layout()
plt.show()


# CREATE DISTRIBUTIONS

# create visualization
plt.figure(figsize=(8, 5))

# Prepare the data split by category
data = [
    df.loc[(df['reaction'] == 1), 'rel_frame_id'],
    df.loc[(df['reaction'] == 0), 'rel_frame_id']
]


# plot a stacked histogram
plt.hist(
    data,
    bins=50,
    stacked=True,
    color=['steelblue', 'red'],
    edgecolor='black',
    label=['Reacted', 'No Reaction']
)

# labels and styling
plt.title('Distribution of Defense Reaction Times', fontsize=14)
plt.xlabel('Frame Relative to Throw', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()

plt.grid(False)
plt.tight_layout()
plt.show()


# GET DEFENDER'S SCORES
# I uploaded an image of these rather than print the code, but this is how
# you could print the code
#qualified_defenders = qualified_final \
#    .sort_values(by='reaction_score', ascending=True).reset_index().copy()
#qualified_defenders[['player_name', 'player_position', 'reaction_score', 'reaction_count']]


# Display Image
from IPython.display import Image, display

display(Image("/kaggle/input/resultimage/sample_SDRI_results.jpg", width=600))

