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


motion_data = pd.read_csv('Motion_Data.csv')
tracking_data = pd.read_csv('tracking_test.csv')
player_play = pd.read_csv('player_play.csv')
games = pd.read_csv('games.csv')
plays = pd.read_csv('plays.csv')
players = pd.read_csv('players.csv')
tracking_df = pd.read_csv('tracking_week_1.csv')
filtered_tracking = pd.read_csv('filtered_tracking_data.csv')


def identify_jet_and_fly_motion(
    tracking_data,
    motion_speed_threshold_before_snap=1.0,  # Reduced to 1 mph before snap
    motion_speed_threshold_at_snap=1.5,      # Reduced to 1.5 mph at the time of snap
    radius_x_at_snap=5.0,                    # Radius in the x-direction (5 yards)
    radius_y_at_snap=5.0                     # Radius in the y-direction (5 yards)
):
    """
    Identifies jet and fly motion plays based on the player speed and position relative to QB at the time of the snap.
    The allowed motion area is modeled as an ellipse with specified radii for x and y directions.
    """
    # Ensure the data is sorted by gameId, playId, nflId, frameId
    tracking_data = tracking_data.sort_values(by=["gameId", "playId", "nflId", "frameId"]).copy()

    # Identify quarterback positions for each play (for all frames)
    qb_positions = tracking_data[tracking_data["position"] == "QB"][["gameId", "playId", "frameId", "x", "y"]]
    qb_positions = qb_positions.rename(columns={"x": "qb_x", "y": "qb_y"})

    # Merge quarterback positions into the tracking data
    tracking_data = pd.merge(tracking_data, qb_positions, on=["gameId", "playId", "frameId"], how="left")

    grouped = tracking_data.groupby(["gameId", "playId", "nflId"])
    motion_flags = []

    for (game_id, play_id, nfl_id), group in grouped:
        # Filter the frames before the snap (i.e., BEFORE_SNAP frameType)
        pre_snap = group[group["frameType"] == "BEFORE_SNAP"].copy()

        if pre_snap.empty:
            motion_flags.append({"gameId": game_id, "playId": play_id, "nflId": nfl_id, "motionType": np.nan})
            continue

        # Identify player in motion: moving above the 1 mph threshold before snap
        player_in_motion = pre_snap[pre_snap["s"] >= motion_speed_threshold_before_snap]

        if player_in_motion.empty:
            motion_flags.append({"gameId": game_id, "playId": play_id, "nflId": nfl_id, "motionType": np.nan})
            continue

        # Find the frame just before the snap (last frame in BEFORE_SNAP)
        snap_frame = pre_snap["frameId"].max()
        snap_frame_data = group[group["frameId"] == snap_frame]

        # Ensure the player is moving at least 1.5 mph at the time of snap
        player_at_snap = snap_frame_data[snap_frame_data["s"] >= motion_speed_threshold_at_snap]

        if player_at_snap.empty:
            motion_flags.append({"gameId": game_id, "playId": play_id, "nflId": nfl_id, "motionType": np.nan})
            continue

        # Get the player's position and the quarterback's position at the snap
        player_at_snap_x = player_at_snap["x"].values[0]
        player_at_snap_y = player_at_snap["y"].values[0]
        qb_at_snap_x = snap_frame_data["qb_x"].values[0]
        qb_at_snap_y = snap_frame_data["qb_y"].values[0]

        # Calculate the Euclidean distances in x and y directions
        delta_x = player_at_snap_x - qb_at_snap_x
        delta_y = player_at_snap_y - qb_at_snap_y

        # Print distances for debugging
        print(f"Player {nfl_id} distance to QB (x-direction): {delta_x} yards, (y-direction): {delta_y} yards")

        # Check if the player is within the allowed motion area (ellipse)
        if (delta_x / radius_x_at_snap)**2 + (delta_y / radius_y_at_snap)**2 <= 1:
            motion_flags.append({"gameId": game_id, "playId": play_id, "nflId": nfl_id, "motionType": "Jet and Fly Motion"})
        else:
            motion_flags.append({"gameId": game_id, "playId": play_id, "nflId": nfl_id, "motionType": np.nan})

    # Convert motion_flags to DataFrame and merge with the original data
    motion_flags_df = pd.DataFrame(motion_flags)
    tracking_data = pd.merge(tracking_data, motion_flags_df, on=["gameId", "playId", "nflId"], how="left")

    return tracking_data


# Apply the function to get the updated data with motionType column
updated_data = identify_jet_and_fly_motion(tracking_data)

# Filter for Jet and Fly Motion plays
jet_and_fly_motion_plays = updated_data[updated_data["motionType"] == "Jet and Fly Motion"]

# Get a list of all unique plays with Jet Motion
motion_play_ids = jet_and_fly_motion_plays[["gameId", "playId"]].drop_duplicates()
num_jet_and_fly_motion_plays = jet_and_fly_motion_plays["playId"].nunique()

# Print the result
print(f"Number of Jet and Fly Motion Plays: {num_jet_and_fly_motion_plays}")
print("Plays identified with Jet and Fly Motion:")
print(motion_play_ids)


def identify_return_motion(
    tracking_data,
    motion_speed_threshold_before_snap=1.0,  # Reduced to 1 mph before snap
    motion_speed_threshold_at_snap=1.5,      # Reduced to 1.5 mph at the time of snap
    radius_x_at_snap=5.0,                    # Radius in the x-direction (5 yards)
    radius_y_at_snap=5.0,                    # Radius in the y-direction (5 yards)
    start_tracking_frame=20                  # Start tracking from frame 20
):
    """
    Identifies return motion plays where the player crosses the QB's Y-coordinate twice and changes direction,
    returning to their original alignment at the time of the snap.
    Only tracks motion starting from frame 20 to avoid break of huddle confusion.
    """
    # Ensure the data is sorted by gameId, playId, nflId, frameId
    tracking_data = tracking_data.sort_values(by=["gameId", "playId", "nflId", "frameId"]).copy()

    # Identify quarterback positions for each play (for all frames)
    qb_positions = tracking_data[tracking_data["position"] == "QB"][["gameId", "playId", "frameId", "x", "y"]]
    qb_positions = qb_positions.rename(columns={"x": "qb_x", "y": "qb_y"})

    # Merge quarterback positions into the tracking data
    tracking_data = pd.merge(tracking_data, qb_positions, on=["gameId", "playId", "frameId"], how="left")

    grouped = tracking_data.groupby(["gameId", "playId", "nflId"])
    motion_flags = []

    for (game_id, play_id, nfl_id), group in grouped:
        # Filter the frames before frame 20 (to avoid the break of huddle)
        group = group[group["frameId"] >= start_tracking_frame]

        # Filter the frames before the snap (i.e., BEFORE_SNAP frameType)
        pre_snap = group[group["frameType"] == "BEFORE_SNAP"].copy()

        if pre_snap.empty:
            motion_flags.append({"gameId": game_id, "playId": play_id, "nflId": nfl_id, "motionType": np.nan})
            continue

        # Identify player in motion: moving above the motion speed threshold before snap
        player_in_motion = pre_snap[pre_snap["s"] >= motion_speed_threshold_before_snap]

        if player_in_motion.empty:
            motion_flags.append({"gameId": game_id, "playId": play_id, "nflId": nfl_id, "motionType": np.nan})
            continue

        # Track the player's Y-position relative to QB's Y at each frame
        player_y_positions = pre_snap["y"].values
        qb_y_position = pre_snap["qb_y"].values[0]

        # Check for crossing the QB's Y-coordinate
        crosses_qb_y = []
        for i in range(1, len(player_y_positions)):
            # Check if the player's Y-coordinate crosses the QB's Y-coordinate
            if (player_y_positions[i-1] < qb_y_position and player_y_positions[i] > qb_y_position) or \
               (player_y_positions[i-1] > qb_y_position and player_y_positions[i] < qb_y_position):
                crosses_qb_y.append(True)
            else:
                crosses_qb_y.append(False)

        # If the player crosses the QB's Y-coordinate twice, it's considered return motion
        if crosses_qb_y.count(True) >= 2:
            # Find the frame just before the snap (last frame in BEFORE_SNAP)
            snap_frame = pre_snap["frameId"].max()
            snap_frame_data = group[group["frameId"] == snap_frame]

            # Ensure the player is moving at least 1.5 mph at the time of snap
            player_at_snap = snap_frame_data[snap_frame_data["s"] >= motion_speed_threshold_at_snap]

            if player_at_snap.empty:
                motion_flags.append({"gameId": game_id, "playId": play_id, "nflId": nfl_id, "motionType": np.nan})
                continue

            # Get the player's position and the quarterback's position at the snap
            player_at_snap_x = player_at_snap["x"].values[0]
            player_at_snap_y = player_at_snap["y"].values[0]
            qb_at_snap_x = snap_frame_data["qb_x"].values[0]
            qb_at_snap_y = snap_frame_data["qb_y"].values[0]

            # Calculate the Euclidean distances in x and y directions
            delta_x = player_at_snap_x - qb_at_snap_x
            delta_y = player_at_snap_y - qb_at_snap_y

            # Debug output for troubleshooting
            print(f"Player {nfl_id} distance to QB (x-direction): {delta_x} yards, (y-direction): {delta_y} yards")

            # Check if the player is within the allowed motion area (ellipse)
            if (delta_x / radius_x_at_snap)**2 + (delta_y / radius_y_at_snap)**2 <= 1:
                motion_flags.append({"gameId": game_id, "playId": play_id, "nflId": nfl_id, "motionType": "Return Motion"})
            else:
                motion_flags.append({"gameId": game_id, "playId": play_id, "nflId": nfl_id, "motionType": np.nan})
        else:
            motion_flags.append({"gameId": game_id, "playId": play_id, "nflId": nfl_id, "motionType": np.nan})

    # Convert motion_flags to DataFrame and merge with the original data
    motion_flags_df = pd.DataFrame(motion_flags)
    tracking_data = pd.merge(tracking_data, motion_flags_df, on=["gameId", "playId", "nflId"], how="left")

    return tracking_data



# Apply the function to get the updated data with motionType column
updated_data = identify_return_motion(tracking_data)

# Filter for Return Motion plays
return_motion_plays = updated_data[updated_data["motionType"] == "Return Motion"]

# Get a list of all unique plays with Return Motion
motion_play_ids = return_motion_plays[["gameId", "playId"]].drop_duplicates()
num_return_motion_plays = return_motion_plays["playId"].nunique()

# Print the result
print(f"Number of Return Motion Plays: {num_return_motion_plays}")
print("Plays identified with Return Motion:")
print(motion_play_ids)


# Load the required CSV files
tracking_df = pd.read_csv('tracking_week_1.csv')
players = pd.read_csv('players.csv')

# Merge the players data to get 'position' into tracking data
tracking_df = pd.merge(tracking_df, players[['nflId', 'position']], on='nflId', how='left')

def identify_jet_and_fly_motion(
    tracking_data,
    motion_speed_threshold_before_snap=1.0,  # Reduced to 1 mph before snap
    motion_speed_threshold_at_snap=1.5,      # Reduced to 1.5 mph at the time of snap
    radius_x_at_snap=5.0,                    # Radius in the x-direction (5 yards)
    radius_y_at_snap=5.0                     # Radius in the y-direction (5 yards)
):
    """
    Identifies jet and fly motion plays based on the player speed and position relative to QB at the time of the snap.
    The allowed motion area is modeled as an ellipse with specified radii for x and y directions.
    """
    # Ensure the data is sorted by gameId, playId, nflId, frameId
    tracking_data = tracking_data.sort_values(by=["gameId", "playId", "nflId", "frameId"]).copy()

    # Identify quarterback positions for each play (for all frames)
    qb_positions = tracking_data[tracking_data["position"] == "QB"][["gameId", "playId", "frameId", "x", "y"]]
    qb_positions = qb_positions.rename(columns={"x": "qb_x", "y": "qb_y"})

    # Merge quarterback positions into the tracking data
    tracking_data = pd.merge(tracking_data, qb_positions, on=["gameId", "playId", "frameId"], how="left")

    grouped = tracking_data.groupby(["gameId", "playId", "nflId"])
    motion_flags = []

    for (game_id, play_id, nfl_id), group in grouped:
        # Filter the frames before the snap (i.e., BEFORE_SNAP frameType)
        pre_snap = group[group["frameType"] == "BEFORE_SNAP"].copy()

        if pre_snap.empty:
            continue

        # Identify player in motion: moving above the 1 mph threshold before snap
        player_in_motion = pre_snap[pre_snap["s"] >= motion_speed_threshold_before_snap]

        if player_in_motion.empty:
            continue

        # Find the frame just before the snap (last frame in BEFORE_SNAP)
        snap_frame = pre_snap["frameId"].max()
        snap_frame_data = group[group["frameId"] == snap_frame]

        # Ensure the player is moving at least 1.5 mph at the time of snap
        player_at_snap = snap_frame_data[snap_frame_data["s"] >= motion_speed_threshold_at_snap]

        if player_at_snap.empty:
            continue

        # Get the player's position and the quarterback's position at the snap
        player_at_snap_x = player_at_snap["x"].values[0]
        player_at_snap_y = player_at_snap["y"].values[0]
        qb_at_snap_x = snap_frame_data["qb_x"].values[0]
        qb_at_snap_y = snap_frame_data["qb_y"].values[0]

        # Calculate the Euclidean distances in x and y directions
        delta_x = player_at_snap_x - qb_at_snap_x
        delta_y = player_at_snap_y - qb_at_snap_y

        # Check if the player is within the allowed motion area (ellipse)
        if (delta_x / radius_x_at_snap)**2 + (delta_y / radius_y_at_snap)**2 <= 1:
            motion_flags.append({"gameId": game_id, "playId": play_id, "nflId": nfl_id, "motionType": "Jet and Fly Motion"})

    # Convert motion_flags to DataFrame
    motion_flags_df = pd.DataFrame(motion_flags)

    # Return only the unique gameId, playId, and nflId for jet and fly motion
    return motion_flags_df[["gameId", "playId", "nflId"]].drop_duplicates()

# Apply the function to identify Jet and Fly Motion plays
jet_and_fly_motion_plays = identify_jet_and_fly_motion(tracking_df)

# Save the result to a new CSV file
jet_and_fly_motion_plays.to_csv("jet_and_fly_motion_plays_week_1.csv", index=False)

# Print the result
print(f"Number of Jet and Fly Motion Plays: {jet_and_fly_motion_plays.shape[0]}")
print("Jet and Fly Motion Plays:")
print(jet_and_fly_motion_plays)



# Load the required CSV files
tracking_df = pd.read_csv('tracking_week_1.csv')
players = pd.read_csv('players.csv')

# Merge the players data to get 'position' into tracking data
tracking_df = pd.merge(tracking_df, players[['nflId', 'position']], on='nflId', how='left')

# Function to identify return motion
def identify_return_motion(
    tracking_data,
    motion_speed_threshold_before_snap=1.0,  # Reduced to 1 mph before snap
    motion_speed_threshold_at_snap=1.5,      # Reduced to 1.5 mph at the time of snap
    radius_x_at_snap=5.0,                    # Radius in the x-direction (5 yards)
    radius_y_at_snap=5.0,                    # Radius in the y-direction (5 yards)
    start_tracking_frame=20                  # Start tracking from frame 20
):
    """
    Identifies return motion plays where the player crosses the QB's Y-coordinate twice and changes direction,
    returning to their original alignment at the time of the snap.
    Only tracks motion starting from frame 20 to avoid break of huddle confusion.
    """
    # Ensure the data is sorted by gameId, playId, nflId, frameId
    tracking_data = tracking_data.sort_values(by=["gameId", "playId", "nflId", "frameId"]).copy()

    # Identify quarterback positions for each play (for all frames)
    qb_positions = tracking_data[tracking_data["position"] == "QB"][["gameId", "playId", "frameId", "x", "y"]]
    qb_positions = qb_positions.rename(columns={"x": "qb_x", "y": "qb_y"})

    # Merge quarterback positions into the tracking data
    tracking_data = pd.merge(tracking_data, qb_positions, on=["gameId", "playId", "frameId"], how="left")

    grouped = tracking_data.groupby(["gameId", "playId", "nflId"])
    motion_flags = []

    for (game_id, play_id, nfl_id), group in grouped:
        # Filter the frames before frame 20 (to avoid the break of huddle)
        group = group[group["frameId"] >= start_tracking_frame]

        # Filter the frames before the snap (i.e., BEFORE_SNAP frameType)
        pre_snap = group[group["frameType"] == "BEFORE_SNAP"].copy()

        if pre_snap.empty:
            motion_flags.append({"gameId": game_id, "playId": play_id, "nflId": nfl_id, "motionType": np.nan})
            continue

        # Identify player in motion: moving above the motion speed threshold before snap
        player_in_motion = pre_snap[pre_snap["s"] >= motion_speed_threshold_before_snap]

        if player_in_motion.empty:
            motion_flags.append({"gameId": game_id, "playId": play_id, "nflId": nfl_id, "motionType": np.nan})
            continue

        # Track the player's Y-position relative to QB's Y at each frame
        player_y_positions = pre_snap["y"].values
        qb_y_position = pre_snap["qb_y"].values[0]

        # Check for crossing the QB's Y-coordinate
        crosses_qb_y = []
        for i in range(1, len(player_y_positions)):
            if (player_y_positions[i-1] < qb_y_position and player_y_positions[i] > qb_y_position) or \
               (player_y_positions[i-1] > qb_y_position and player_y_positions[i] < qb_y_position):
                crosses_qb_y.append(True)
            else:
                crosses_qb_y.append(False)

        # If the player crosses the QB's Y-coordinate twice, it's considered return motion
        if crosses_qb_y.count(True) >= 2:
            snap_frame = pre_snap["frameId"].max()
            snap_frame_data = group[group["frameId"] == snap_frame]

            # Ensure the player is moving at least 1.5 mph at the time of snap
            player_at_snap = snap_frame_data[snap_frame_data["s"] >= motion_speed_threshold_at_snap]

            if player_at_snap.empty:
                motion_flags.append({"gameId": game_id, "playId": play_id, "nflId": nfl_id, "motionType": np.nan})
                continue

            player_at_snap_x = player_at_snap["x"].values[0]
            player_at_snap_y = player_at_snap["y"].values[0]
            qb_at_snap_x = snap_frame_data["qb_x"].values[0]
            qb_at_snap_y = snap_frame_data["qb_y"].values[0]

            delta_x = player_at_snap_x - qb_at_snap_x
            delta_y = player_at_snap_y - qb_at_snap_y

            if (delta_x / radius_x_at_snap)**2 + (delta_y / radius_y_at_snap)**2 <= 1:
                motion_flags.append({"gameId": game_id, "playId": play_id, "nflId": nfl_id, "motionType": "Return Motion"})
            else:
                motion_flags.append({"gameId": game_id, "playId": play_id, "nflId": nfl_id, "motionType": np.nan})
        else:
            motion_flags.append({"gameId": game_id, "playId": play_id, "nflId": nfl_id, "motionType": np.nan})

    # Return gameId, playId, and nflId for plays with return motion
    motion_flags_df = pd.DataFrame(motion_flags)
    return motion_flags_df[motion_flags_df["motionType"] == "Return Motion"][["gameId", "playId", "nflId"]].drop_duplicates()

# Apply the function to get the updated data with Return Motion plays
return_motion_plays = identify_return_motion(tracking_df)

# Save the result to a new CSV file
return_motion_plays.to_csv("return_motion_plays_with_nflId_week_1.csv", index=False)

# Print the result
print(f"Number of Return Motion Plays: {return_motion_plays.shape[0]}")
print("Plays identified with Return Motion:")
print(return_motion_plays)



import pandas as pd
import os

# Create an empty list to store all DataFrames
combined_data = []

# Loop through each week (from 1 to 9)
for week in range(1, 10):
    # Define the file name based on the week number
    file_name = f"jet_and_fly_motion_plays_week_{week}.csv"
    
    # Check if the file exists
    if os.path.exists(file_name):
        # Read the CSV into a DataFrame
        week_data = pd.read_csv(file_name)
        
        # Add a new column 'motionType' with the value 'Jet/Fly'
        week_data['motionType'] = 'Jet/Fly'
        
        # Append the data to the list
        combined_data.append(week_data)
    else:
        print(f"File {file_name} not found!")

# Combine all DataFrames into one
combined_df = pd.concat(combined_data, ignore_index=True)

# Remove the specific row with gameId, playId, and nflId
combined_df = combined_df[~((combined_df['gameId'] == 2022090800) & 
                             (combined_df['playId'] == 80) & 
                             (combined_df['nflId'] == 47857.0) & 
                             (combined_df['motionType'] == 'Jet/Fly'))]

# Save the combined DataFrame to a new CSV file
combined_df.to_csv("combined_jetflymotion.csv", index=False)

# Print a confirmation message
print(f"Combined DataFrame saved as combined_jetflymotion.csv without the removed row.")



# Create an empty list to store all DataFrames
combined_data = []

# Loop through each week (from 1 to 9)
for week in range(1, 10):
    # Define the file name based on the week number
    file_name = f"return_motion_plays_with_nflId_week_{week}.csv"
    
    # Check if the file exists
    if os.path.exists(file_name):
        # Read the CSV into a DataFrame
        week_data = pd.read_csv(file_name)
        
        # Add a new column 'motionType' with the value 'Return'
        week_data['motionType'] = 'Return'
        
        # Append the data to the list
        combined_data.append(week_data)
    else:
        print(f"File {file_name} not found!")

# Combine all DataFrames into one
combined_df = pd.concat(combined_data, ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv("combined_return.csv", index=False)

# Print a confirmation message
print(f"Combined DataFrame saved as combined_return.csv")



# Read the two CSV files
jetflymotion_df = pd.read_csv('combined_jetflymotion.csv')
return_df = pd.read_csv('combined_return.csv')

# Combine the two DataFrames by concatenating them vertically (along rows)
combined_df = pd.concat([jetflymotion_df, return_df], ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv("combined_motion.csv", index=False)

# Print a confirmation message
print("The two CSV files have been combined and saved as combined_motion.csv.")



# Load required library
library(dplyr)

setwd("/Users/willstokes/Desktop/Big Data Bowl/")

# Step 1: Load the datasets
games <- read.csv("games.csv")
plays <- read.csv("plays.csv")
players <- read.csv("players.csv")
player_play <- read.csv("player_play.csv")


#join plays with player_plays
player_play_with_plays <- player_play %>%
  inner_join(plays, by = c("gameId", "playId"))

#join players to combined plays and player_plays
player_play_with_plays_and_players <- player_play_with_plays %>%
  left_join(players %>% select(nflId, displayName, position), by = "nflId")

#add games
final_dataset <- player_play_with_plays_and_players %>%
  left_join(games, by = "gameId")

#filter to plays with motion
 filtered_dataset <- final_dataset %>%
  filter(position %in% c("WR", "TE", "RB", "FB")
  )

# Create a new dataset with only the specified variables, keeping the order

Motion_Data2 <- filtered_dataset %>%
  select(
    #game, play, situation, descriptive 
    gameId, playId, nflId, teamAbbr, week, playDescription,position, displayName, quarter, down,
    yardsToGo,possessionTeam, defensiveTeam, 
    #Motion and receiving/running
    inMotionAtBallSnap, motionSinceLineset, shiftSinceLineset, hadRushAttempt, 
    receiverAlignment, wasRunningRoute, routeRan, wasTargettedReceiver, hadPassReception, targetX, targetY,
    rushingYards, receivingYards, yardsGained, yardageGainedAfterTheCatch, 
    #defensive coverage & assignments 
    pff_defensiveCoverageAssignment,pff_primaryDefensiveCoverageMatchupNflId,
    pff_secondaryDefensiveCoverageMatchupNflId,pff_passCoverage, pff_manZone,
    #offensive play type
    offenseFormation,pff_runConceptPrimary, pff_runConceptSecondary, pff_runPassOption, isDropback, 
    #field position and clock 
     yardlineSide, yardlineNumber, absoluteYardlineNumber, gameClock, 
   #win probability metrics 
    preSnapHomeScore, preSnapVisitorScore, preSnapHomeTeamWinProbability, preSnapVisitorTeamWinProbability,
    homeTeamWinProbabilityAdded, visitorTeamWinProbilityAdded, expectedPoints, expectedPointsAdded,
  #plays to take out 
    playNullifiedByPenalty, qbKneel, qbSneak 
  )

Motion_Data2 <- Motion_Data2 %>%
  mutate(
    motionType = case_when(
      is.na(inMotionAtBallSnap) | is.na(motionSinceLineset) ~ "No Motion", # NA values result in "No Motion"
      !inMotionAtBallSnap & !motionSinceLineset ~ "No Motion", # Both are FALSE
      inMotionAtBallSnap | motionSinceLineset ~ "Motion"      # Either is TRUE
    )
  )


Motion_Final <- Motion_Data2 %>%
  left_join(combined_motion, by = c("gameId", "playId", "nflId"), relationship = "many-to-many") %>% # Merge on gameId, playId, and nflId
  mutate(
    motionType = ifelse(!is.na(motionType.y), motionType.y, motionType.x) # Replace motionType if a match is found
  ) %>%
  select(-motionType.x, -motionType.y) %>% # Drop intermediate columns
  select(motionType, everything()) %>% # Move motionType to the front
  filter(motionType %in% c("Jet/Fly", "Return")) # Keep only plays with motionType 'Fly/Jet' or 'Return'

# View the resulting dataset
head(Motion_Final)



 
write.csv(Motion_Final, "/Users/willstokes//Desktop/Motion_Final.csv", row.names = FALSE)


