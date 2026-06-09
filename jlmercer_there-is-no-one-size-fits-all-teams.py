


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.spatial.distance import cdist


# Load each dataset
games = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/games.csv')
player_play = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/player_play.csv')
players = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/players.csv')
plays = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/plays.csv')

# Load weekly data
week1 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_1.csv')
week2 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_2.csv')
week3 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_3.csv')
week4 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_4.csv')
week5 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_5.csv')
week6 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_6.csv')
week7 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_7.csv')
week8 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_8.csv')
week9 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_9.csv')


# Pulling together a singular table with relevant information
player_pass_attempt = player_play[player_play['hadDropback'] == 1][['nflId', 'gameId', 'playId', 'teamAbbr']].groupby(['nflId', 'gameId', 'playId', 'teamAbbr']).sum().reset_index()
player_pass_reception = player_play[player_play['hadPassReception'] == 1][['gameId', 'playId', 'receivingYards', 'teamAbbr']].groupby(['gameId', 'playId', 'teamAbbr']).sum().reset_index()
player_rec_yards = pd.merge(player_pass_attempt, player_pass_reception, how = 'left', on = ['gameId', 'playId', 'teamAbbr'])
## Add formation to play results and then figure out best average rush by player
player_rush_yards = player_play[player_play['hadRushAttempt'] == 1][['nflId', 'gameId', 'playId', 'rushingYards', 'teamAbbr']].groupby(['nflId', 'gameId', 'playId', 'teamAbbr']).sum().reset_index()
player_pass_attempt = player_play[player_play['hadDropback'] == 1][['nflId', 'gameId', 'playId', 'teamAbbr']].groupby(['nflId', 'gameId', 'playId', 'teamAbbr']).sum().reset_index()
player_pass_reception = player_play[player_play['hadPassReception'] == 1][['gameId', 'playId', 'receivingYards', 'teamAbbr']].groupby(['gameId', 'playId', 'teamAbbr']).sum().reset_index()
player_rec_yards = pd.merge(player_pass_attempt, player_pass_reception, how = 'left', on = ['gameId', 'playId', 'teamAbbr'])
rusher_stats = pd.merge(left = player_rush_yards, right = players[['nflId', 'displayName']], how = 'inner', on = 'nflId')
rec_stats = pd.merge(left = player_rec_yards, right = players[['nflId', 'displayName']], how = 'inner', on = 'nflId')
play_formation = plays[['gameId', 'playId', 'offenseFormation']].groupby(['gameId', 'playId', 'offenseFormation']).count().reset_index(level = ['gameId', 'playId', 'offenseFormation'])
rush_data = pd.merge(left = rusher_stats, right = play_formation, how = 'inner', on = ['gameId', 'playId'])
rec_data =  pd.merge(left = rec_stats, right = play_formation, how = 'inner', on = ['gameId', 'playId'])
yard_data = pd.concat([rush_data, rec_data], axis = 0)
yard_data_play_enh = pd.merge(left = yard_data, right = plays[['gameId', 'playId', 'passResult', 'receiverAlignment', 'quarter', 'down', 'yardsToGo', 'gameClock', 'preSnapHomeScore', 'preSnapVisitorScore', 'absoluteYardlineNumber']], how = 'left', on = ['gameId', 'playId'])
yard_data_home_enh = pd.merge(left = yard_data_play_enh, right = games[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']], how = 'inner', on = ['gameId'])
## Was there motion on the play??
play_motion = player_play[player_play['motionSinceLineset'] == 1][['gameId', 'playId', 'motionSinceLineset']].groupby(['gameId', 'playId']).count().reset_index()
for row in range(len(play_motion)):
    if play_motion.at[row, 'motionSinceLineset'] > 1:
        play_motion.at[row, 'motionSinceLineset'] = 1
    else: 
        play_motion.at[row, 'motionSinceLineset'] = 0
# Merging the motion information
yard_data_motion = pd.merge(left = yard_data_home_enh, right = play_motion, how = 'left', on = ['gameId', 'playId'])
# Finding the 11 offensive players
players_on_play = []
play_group = player_play.groupby(['gameId', 'playId'])
for (game_id, play_id), teams in play_group:
    teams_group = teams.groupby('teamAbbr')
    for team_abbr, team_group in teams_group:
        player_ids = team_group['nflId'].unique()

        if len(player_ids) == 11:
            players_on_play.append({
                'gameId': game_id,
                'playId': play_id,
                'teamAbbr': team_abbr,
                **{f'player_{i+1}': player_ids[i] for i in range(11)}
            })
players_on_play_frame = pd.DataFrame(players_on_play)
# Merging the player information
yard_data_enh = pd.merge(left = yard_data_motion, right = players_on_play_frame, how = 'left', on = ['gameId', 'playId', 'teamAbbr'])
yard_data_enh = yard_data_enh.reset_index()
yard_data_enh.drop('index', axis = 1, inplace = True)
yard_data_enh['motionSinceLineset'][yard_data_enh['motionSinceLineset'].isna()] = 0


# Creating our prediction variable - rush
for row in range(len(yard_data_enh)):
    if pd.notnull(yard_data_enh.at[row, 'rushingYards']):
        yard_data_enh.at[row, 'rush'] = 1
    else:
        yard_data_enh.at[row, 'rush'] = 0


# Expanding contextual information
for row in range(len(yard_data_enh)):
    if yard_data_enh.at[row, 'homeTeamAbbr'] == yard_data_enh.at[row, 'teamAbbr']:
        yard_data_enh.at[row, 'score_diff'] = yard_data_enh.at[row, 'preSnapHomeScore'] - yard_data_enh.at[row, 'preSnapVisitorScore']
    else:
        yard_data_enh.at[row, 'score_diff'] = yard_data_enh.at[row, 'preSnapVisitorScore'] - yard_data_enh.at[row, 'preSnapHomeScore']
# yard bucket
for row in range(len(yard_data_enh)):
    if yard_data_enh.at[row, 'yardsToGo'] > 7: 
        yard_data_enh.at[row, 'yard_bucket'] = 'long'
    elif yard_data_enh.at[row, 'yardsToGo'] <= 2:
        yard_data_enh.at[row, 'yard_bucket'] = 'short'
    else:
        yard_data_enh.at[row, 'yard_bucket'] = 'mid'
# distance bucket
for row in range(len(yard_data_enh)):
    if yard_data_enh.at[row, 'absoluteYardlineNumber'] > 50: 
        yard_data_enh.at[row, 'distance_bucket'] = 'long'
    elif yard_data_enh.at[row, 'absoluteYardlineNumber'] <= 20:
        yard_data_enh.at[row, 'distance_bucket'] = 'short'
    else:
        yard_data_enh.at[row,'distance_bucket'] = 'mid'
# Creating gametime remaining
for row in range(len(yard_data_enh)):
    if yard_data_enh.at[row, 'quarter'] == 1:
        yard_data_enh.at[row, 'gameTimeRemaining'] = str(int(yard_data_enh.at[row, 'gameClock'][:2]) + 45) + yard_data_enh.at[row, 'gameClock'][3:]
    elif yard_data_enh.at[row, 'quarter'] == 2:
        yard_data_enh.at[row, 'gameTimeRemaining'] = str(int(yard_data_enh.at[row, 'gameClock'][:2]) + 30) + yard_data_enh.at[row, 'gameClock'][3:]
    elif yard_data_enh.at[row, 'quarter'] == 3:
        yard_data_enh.at[row, 'gameTimeRemaining'] = str(int(yard_data_enh.at[row, 'gameClock'][:2]) + 15) + yard_data_enh.at[row, 'gameClock'][3:]
    else:
        yard_data_enh.at[row, 'gameTimeRemaining'] = yard_data_enh.at[row, 'gameClock'][:2] + yard_data_enh.at[row, 'gameClock'][3:]
# Half time and final two minutes
for row in range(len(yard_data_enh)):
    if int(yard_data_enh.at[row, 'gameTimeRemaining']) <= 200:
        yard_data_enh.at[row, 'time_bucket'] = 'FTMW'
    elif (int(yard_data_enh.at[row, 'gameTimeRemaining']) <= 3200) & (int(yard_data_enh.at[row, 'gameTimeRemaining']) >= 3001):
        yard_data_enh.at[row, 'time_bucket'] = 'HTMW'
    else:
        yard_data_enh.at[row, 'time_bucket'] = ''
# Adding the defensive team
for row in range(len(yard_data_enh)):
    if yard_data_enh.at[row, 'homeTeamAbbr'] == yard_data_enh.at[row, 'teamAbbr']:
        yard_data_enh.at[row, 'defTeam'] = yard_data_enh.at[row, 'visitorTeamAbbr']
    else:
        yard_data_enh.at[row, 'defTeam'] = yard_data_enh.at[row, 'homeTeamAbbr']
# Knowing how many pass catchers and defender variety
all_player_play = player_play[['gameId', 'playId', 'nflId']]
receiver_data = player_play[player_play['wasRunningRoute'] == 1][['gameId', 'playId', 'nflId', 'routeRan']]
player_position_data = players[['nflId', 'position']]
receiver_data_pos = pd.merge(receiver_data, player_position_data, on = ['nflId'], how = 'left')
player_pos = pd.merge(all_player_play, player_position_data, on = ['nflId'], how = 'left')
offensive_summary = player_pos.groupby(['gameId', 'playId']).agg(
    WR = ('position', lambda x: (x == 'WR').sum()),
    TE = ('position', lambda x: (x == 'TE').sum()),
    RB = ('position', lambda x: (x == 'RB').sum()),
).reset_index()
defender_summary = player_pos.groupby(['gameId', 'playId']).agg(
    CB = ('position', lambda x: (x == 'CB').sum()),
    ILB = ('position', lambda x: (x == 'ILB').sum()),
    OLB = ('position', lambda x: (x == 'OLB').sum()),
    DE = ('position', lambda x: (x == 'DE').sum()),
    DT = ('position', lambda x: (x == 'DT').sum()),
    NT = ('position', lambda x: (x == 'NT').sum()),
    FS = ('position', lambda x: (x == 'FS').sum()),
    SS = ('position', lambda x: (x == 'SS').sum()),
).reset_index()
receiver_data_summary = pd.merge(receiver_data_pos, offensive_summary, on = ['gameId', 'playId'], how = 'left')
receiver_defender_summary = pd.merge(receiver_data_summary, defender_summary[['gameId', 'playId', 'CB', 'ILB', 'OLB', 'DE', 'DT', 'FS', 'SS']], on = ['gameId', 'playId'])
receiver_def_data = pd.merge(receiver_defender_summary, yard_data_enh, how = 'inner', on = ['gameId', 'playId'])
receiver_def_data.rename(columns = {'nflId_x': 'nflId'}, inplace = True)
receiver_data_enh = pd.merge(receiver_def_data, plays[['gameId', 'playId', 'pff_passCoverage', 'pff_manZone']], on = ['gameId', 'playId'], how = 'left')
rush_off_data = pd.merge(yard_data_enh, offensive_summary, on = ['gameId', 'playId'], how = 'left')
rush_def_data = pd.merge(rush_off_data, defender_summary[['gameId', 'playId', 'CB', 'ILB', 'OLB', 'DE', 'DT', 'FS', 'SS']], on = ['gameId', 'playId'])
rush_def_data.rename(columns = {'nflId_x': 'nflId'}, inplace = True)   


rush_def_data['motionSinceLineset'][rush_def_data['motionSinceLineset'].isna()] = 0
plays_sorted = plays.sort_values(by = ['gameId', 'playId', 'possessionTeam', 'quarter', 'gameClock']).reset_index(drop = True)
plays_sorted['previousPlayResult'] = plays_sorted.groupby(['gameId', 'possessionTeam'])['yardsGained'].shift(1)
plays_sorted['previousPlayType'] = plays_sorted.groupby(['gameId', 'possessionTeam'])['isDropback'].shift(1)
plays_sorted['priorPreviousPlayResult'] = plays_sorted.groupby(['gameId', 'possessionTeam'])['yardsGained'].shift(2)
plays_sorted['priorPreviousPlayType'] = plays_sorted.groupby(['gameId', 'possessionTeam'])['isDropback'].shift(2)
plays_sorted['previousPlayType'] = plays_sorted['previousPlayType'].fillna('First')
plays_sorted[plays_sorted['previousPlayType'] == True]['previousPlayType'] = 'Pass'
plays_sorted['priorPreviousPlayType'] = plays_sorted['priorPreviousPlayType'].fillna('First')
plays_sorted['previousPlayResult'] = plays_sorted['previousPlayResult'].fillna(0.01)
plays_sorted['priorPreviousPlayResult'] = plays_sorted['priorPreviousPlayResult'].fillna(0.01)
for play in range(len(plays_sorted)):
    if plays_sorted.at[play, 'previousPlayType'] == True:
        plays_sorted.at[play, 'previousPlayType'] = 'Pass'
    else:
        plays_sorted.at[play, 'previousPlayType'] = 'Run'
    if plays_sorted.at[play, 'priorPreviousPlayType'] == True:
        plays_sorted.at[play, 'priorPreviousPlayType'] = 'Pass'
    else:
        plays_sorted.at[play, 'priorPreviousPlayType'] = 'Run'
plays_sorted['isFirstPlay'] = (plays_sorted['possessionTeam'] != plays_sorted.groupby(['gameId'])['possessionTeam'].shift(1)).astype(int)
plays_sorted['isFirst10Plays'] = (plays_sorted.groupby(['gameId', 'possessionTeam']).cumcount() < 10).astype(int)
plays_sorted['isFirstPossession'] = (plays_sorted[plays_sorted['isFirstPlay'] == 1].groupby(['gameId', 'possessionTeam', 'isFirstPlay']).cumcount() < 1).astype(int)
plays_sorted['isFirstPossession'].ffill(axis = 0, inplace = True)
rush_data_enh = pd.merge(rush_def_data, plays_sorted[['gameId', 'playId', 'isFirstPlay', 'isFirst10Plays', 'isFirstPossession',
                                                      'previousPlayResult', 'previousPlayType', 'priorPreviousPlayResult', 'priorPreviousPlayType']],
                                                      on = ['gameId', 'playId'], how = 'left')


# Incorporate spacing data
relevant_events = ["huddle_break_offense", "line_set", "man_in_motion", "ball_snap"]
tracking_filtered_week1 = week1[week1['event'].isin(relevant_events)]
tracking_filtered_week2 = week2[week2['event'].isin(relevant_events)]
tracking_filtered_week3 = week3[week3['event'].isin(relevant_events)]
tracking_filtered_week4 = week4[week4['event'].isin(relevant_events)]
tracking_filtered_week5 = week5[week5['event'].isin(relevant_events)]
tracking_filtered_week6 = week6[week6['event'].isin(relevant_events)]
tracking_filtered_week7 = week7[week7['event'].isin(relevant_events)]
tracking_filtered_week8 = week8[week8['event'].isin(relevant_events)]
tracking_filtered_week9 = week9[week9['event'].isin(relevant_events)]


# Function to calculate time differences between key events
def calculate_time_differences(tracking_data):
    results = []

    # Process each play
    for (gameId, playId), play_data in tracking_data.groupby(['gameId', 'playId']):
        # Sort by time for proper sequencing
        play_data = play_data.sort_values(by='time')

        # Extract timestamps for each relevant event
        timestamps = {}
        for event in relevant_events:
            event_time = play_data[play_data['event'] == event]['time']
            timestamps[event] = pd.to_datetime(event_time.iloc[0]) if not event_time.empty else None

        # Calculate time differences (in seconds)
        huddle_to_line = (
            (timestamps["line_set"] - timestamps["huddle_break_offense"]).total_seconds()
            if timestamps["line_set"] and timestamps["huddle_break_offense"]
            else None
        )
        line_to_motion = (
            (timestamps["man_in_motion"] - timestamps["line_set"]).total_seconds()
            if timestamps["man_in_motion"] and timestamps["line_set"]
            else None
        )
        line_to_snap = (
            (timestamps["ball_snap"] - timestamps["line_set"]).total_seconds()
            if timestamps["ball_snap"] and timestamps["line_set"]
            else None
        )

        # Append results for the play
        results.append({
            'gameId': gameId,
            'playId': playId,
            'timeHuddleToLineSet': huddle_to_line,
            'timeLineSetToMotion': line_to_motion,
            'timeLineSetToSnap': line_to_snap
        })

    return pd.DataFrame(results)

# Calculate time differences for each play
time_differences_week1 = calculate_time_differences(tracking_filtered_week1)
time_differences_week2 = calculate_time_differences(tracking_filtered_week2)
time_differences_week3 = calculate_time_differences(tracking_filtered_week3)
time_differences_week4 = calculate_time_differences(tracking_filtered_week4)
time_differences_week5 = calculate_time_differences(tracking_filtered_week5)
time_differences_week6 = calculate_time_differences(tracking_filtered_week6)
time_differences_week7 = calculate_time_differences(tracking_filtered_week7)
time_differences_week8 = calculate_time_differences(tracking_filtered_week8)
time_differences_week9 = calculate_time_differences(tracking_filtered_week9)


FIELD_WIDTH = 53.3
def calculate_player_distances(tracking_filtered):
    results = []
    # Drop rows with NAN in nflId column
    tracking_filtered = tracking_filtered.dropna(subset=['nflId'])
    # Ensure `y` column is numeric
    tracking_filtered['y'] = pd.to_numeric(tracking_filtered['y'], errors='coerce')
    # Drop rows with NaN in `y` column
    tracking_filtered = tracking_filtered.dropna(subset=['y'])

    relevant_events = ["line_set", "ball_snap"]
    tracking_data = tracking_filtered[tracking_filtered['event'].isin(relevant_events)]
    
    # Group by gameId and playId
    for (gameId, playId), play_data in tracking_data.groupby(['gameId', 'playId']):
        # Separate data for line_set and ball_snap
        line_set_data = play_data[play_data['event'] == 'line_set'].set_index('nflId')
        ball_snap_data = play_data[play_data['event'] == 'ball_snap'].set_index('nflId')

        # Ensure players exist in both line_set and ball_snap
        common_players = line_set_data.index.intersection(ball_snap_data.index)

        # Calculate distances for the three closest players to each sideline at line_set
        line_set_data['sidelineDistance'] = line_set_data['y'].apply(lambda y: min(y, FIELD_WIDTH - y))
        closest_to_sideline = line_set_data.sort_values('sidelineDistance').head(3)

        for nflId in closest_to_sideline.index:
            if nflId not in ball_snap_data.index:
                continue

            # Extract player positions at line_set and ball_snap
            x_line, y_line = line_set_data.loc[nflId, ['x', 'y']].values
            x_snap, y_snap = ball_snap_data.loc[nflId, ['x', 'y']].values

            # Skip rows where y_line or y_snap is not a valid float
            if not isinstance(y_line, (int, float)) or not isinstance(y_snap, (int, float)):
                print(f"Skipping nflId={nflId} in gameId={gameId}, playId={playId} due to invalid y values")
                continue

            # Calculate left and right neighbors at line_set
            left_neighbors_line = line_set_data[line_set_data['y'] < y_line].sort_values('y', ascending=False)
            right_neighbors_line = line_set_data[line_set_data['y'] > y_line].sort_values('y')

            left_distance_line = (
                abs(y_line - left_neighbors_line.iloc[0]['y'])
                if not left_neighbors_line.empty
                else min(y_line, FIELD_WIDTH - y_line)
            )
            right_distance_line = (
                abs(y_line - right_neighbors_line.iloc[0]['y'])
                if not right_neighbors_line.empty
                else min(y_line, FIELD_WIDTH - y_line)
            )

            # Calculate left and right neighbors at ball_snap
            left_neighbors_snap = ball_snap_data[ball_snap_data['y'] < y_snap].sort_values('y', ascending=False)
            right_neighbors_snap = ball_snap_data[ball_snap_data['y'] > y_snap].sort_values('y')

            left_distance_snap = (
                abs(y_snap - left_neighbors_snap.iloc[0]['y'])
                if not left_neighbors_snap.empty
                else min(y_snap, FIELD_WIDTH - y_snap)
            )
            right_distance_snap = (
                abs(y_snap - right_neighbors_snap.iloc[0]['y'])
                if not right_neighbors_snap.empty
                else min(y_snap, FIELD_WIDTH - y_snap)
            )

            # Calculate net changes
            net_left_distance_change = left_distance_snap - left_distance_line
            net_right_distance_change = right_distance_snap - right_distance_line

            # Append results
            results.append({
                'gameId': gameId,
                'playId': playId,
                'nflId': nflId,
                'sidelineDistance': line_set_data.loc[nflId, 'sidelineDistance'],
                'netLeftDistanceChange': net_left_distance_change,
                'netRightDistanceChange': net_right_distance_change
            })

    return pd.DataFrame(results)


# Calculate player distances and add them to the tracking dataset
player_distances_week1 = calculate_player_distances(tracking_filtered_week1)
player_distances_week2 = calculate_player_distances(tracking_filtered_week2)
player_distances_week3 = calculate_player_distances(tracking_filtered_week3)
player_distances_week4 = calculate_player_distances(tracking_filtered_week4)
player_distances_week5 = calculate_player_distances(tracking_filtered_week5)
player_distances_week6 = calculate_player_distances(tracking_filtered_week6)
player_distances_week7 = calculate_player_distances(tracking_filtered_week7)
player_distances_week8 = calculate_player_distances(tracking_filtered_week8)
player_distances_week9 = calculate_player_distances(tracking_filtered_week9)


def consolidate_to_play_level(player_distances):
    play_results = []

    for (gameId, playId), play_data in player_distances.groupby(['gameId', 'playId']):
        # Sort players by sideline distance for consistent ordering
        play_data = play_data.sort_values(by=['sidelineDistance']).reset_index(drop=True)

        # Extract the six closest players and their net distances
        players = play_data.head(6)  # Take the top 6 players (3 closest to each sideline)
        player_ids = players['nflId'].tolist()
        net_left_distances = players['netLeftDistanceChange'].tolist()
        net_right_distances = players['netRightDistanceChange'].tolist()

        # Append play-level data
        play_results.append({
            'gameId': gameId,
            'playId': playId,
            'player1_nflId': player_ids[0] if len(player_ids) > 0 else None,
            'player2_nflId': player_ids[1] if len(player_ids) > 1 else None,
            'player3_nflId': player_ids[2] if len(player_ids) > 2 else None,
            'player4_nflId': player_ids[3] if len(player_ids) > 3 else None,
            'player5_nflId': player_ids[4] if len(player_ids) > 4 else None,
            'player6_nflId': player_ids[5] if len(player_ids) > 5 else None,
            'player1_netLeftDistanceChange': net_left_distances[0] if len(net_left_distances) > 0 else None,
            'player2_netLeftDistanceChange': net_left_distances[1] if len(net_left_distances) > 1 else None,
            'player3_netLeftDistanceChange': net_left_distances[2] if len(net_left_distances) > 2 else None,
            'player4_netLeftDistanceChange': net_left_distances[3] if len(net_left_distances) > 3 else None,
            'player5_netLeftDistanceChange': net_left_distances[4] if len(net_left_distances) > 4 else None,
            'player6_netLeftDistanceChange': net_left_distances[5] if len(net_left_distances) > 5 else None,
            'player1_netRightDistanceChange': net_right_distances[0] if len(net_right_distances) > 0 else None,
            'player2_netRightDistanceChange': net_right_distances[1] if len(net_right_distances) > 1 else None,
            'player3_netRightDistanceChange': net_right_distances[2] if len(net_right_distances) > 2 else None,
            'player4_netRightDistanceChange': net_right_distances[3] if len(net_right_distances) > 3 else None,
            'player5_netRightDistanceChange': net_right_distances[4] if len(net_right_distances) > 4 else None,
            'player6_netRightDistanceChange': net_right_distances[5] if len(net_right_distances) > 5 else None,
        })

    return pd.DataFrame(play_results)


play_level_tracking_week1 = consolidate_to_play_level(player_distances_week1)
play_level_tracking_week2 = consolidate_to_play_level(player_distances_week2)
play_level_tracking_week3 = consolidate_to_play_level(player_distances_week3)
play_level_tracking_week4 = consolidate_to_play_level(player_distances_week4)
play_level_tracking_week5 = consolidate_to_play_level(player_distances_week5)
play_level_tracking_week6 = consolidate_to_play_level(player_distances_week6)
play_level_tracking_week7 = consolidate_to_play_level(player_distances_week7)
play_level_tracking_week8 = consolidate_to_play_level(player_distances_week8)
play_level_tracking_week9 = consolidate_to_play_level(player_distances_week9)


time_differences = [time_differences_week1, time_differences_week2, time_differences_week3,
                    time_differences_week4, time_differences_week5, time_differences_week6,
                    time_differences_week7, time_differences_week8, time_differences_week9]
player_distances = [play_level_tracking_week1, play_level_tracking_week2, play_level_tracking_week3,
                    play_level_tracking_week4, play_level_tracking_week5, play_level_tracking_week6,
                    play_level_tracking_week7, play_level_tracking_week8, play_level_tracking_week9]

time_data = pd.concat(time_differences, axis = 0, ignore_index = True)
player_dist_data = pd.concat(player_distances, axis = 0, ignore_index = True)
time_data.fillna(0, inplace = True)
player_dist_data.fillna(0, inplace = True)


time_data_merge = pd.merge(rush_data_enh, time_data, on = ['gameId', 'playId'], how = 'left')
dist_data_merge = pd.merge(time_data_merge, player_dist_data, on = ['gameId', 'playId'], how = 'left')
dist_data_merge.fillna(0, inplace = True)


categorical_variables = ['offenseFormation', 'receiverAlignment', 'quarter', 'down', 
                         'motionSinceLineset', 'time_bucket', 'WR', 'TE', 'RB',
                         'isFirstPossession', 'isFirstPlay', 'isFirst10Plays', 'previousPlayType',
                         'priorPreviousPlayType', 'player1_nflId', 'player2_nflId', 'player3_nflId']
numeric_variables = ['score_diff', 'yardsToGo', 'absoluteYardlineNumber', 'previousPlayResult', 'priorPreviousPlayResult',
                     'timeHuddleToLineSet', 'timeLineSetToMotion', 'timeLineSetToSnap', 'player1_netLeftDistanceChange', 
                     'player2_netLeftDistanceChange', 'player3_netLeftDistanceChange', 'player1_netRightDistanceChange', 
                     'player2_netRightDistanceChange', 'player3_netRightDistanceChange']
variables = ['offenseFormation', 'receiverAlignment', 'quarter', 'down', 'score_diff', 
             'yardsToGo', 'absoluteYardlineNumber', 'motionSinceLineset', 'time_bucket', 
             'WR', 'TE', 'RB', 'isFirstPossession', 'isFirstPlay', 'isFirst10Plays',
             'previousPlayType', 'previousPlayResult', 'priorPreviousPlayType', 'priorPreviousPlayResult',
             'player1_nflId', 'player2_nflId', 'player3_nflId', 'timeHuddleToLineSet', 'timeLineSetToMotion', 
             'timeLineSetToSnap', 'player1_netLeftDistanceChange', 
                     'player2_netLeftDistanceChange', 'player3_netLeftDistanceChange', 'player1_netRightDistanceChange', 
                     'player2_netRightDistanceChange', 'player3_netRightDistanceChange']
target = 'rush'

team_results = []
teams = dist_data_merge['teamAbbr'].unique()

for team in teams:
    print(f"Training model for {team}")

    team_data = dist_data_merge[dist_data_merge['teamAbbr'] == team]

    X = team_data[variables]
    y = team_data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    ## One Hot Encode the categorical values
    preprocessing = ColumnTransformer(
        transformers = [('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_variables),
                        ('num', StandardScaler(), numeric_variables)
                        ]
    )
    # Basic Logistic Regression Model
    log_model = Pipeline(
        steps = [
            ('preprocessor', preprocessing),
            ('classifier', LogisticRegression(solver = 'liblinear'))
        ]
    )
    # Fit the data
    log_model.fit(X_train, y_train)
    # Make a prediction
    y_predict = log_model.predict(X_test)
    # Accuracy
    log_model_accuracy = accuracy_score(y_test, y_predict)

    # Get the model and its steps
    logistic_regression_model = log_model.named_steps['classifier']
    preprocessor = log_model.named_steps['preprocessor']

    categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_variables)
    all_features = list(categorical_features) + numeric_variables

    coefficents = logistic_regression_model.coef_[0]
    intercepts = logistic_regression_model.intercept_[0]

    log_df = pd.DataFrame({
        'Feature': all_features,
        'Coefficient': coefficents
    }).sort_values(by = 'Coefficient', key = abs, ascending = False)

    team_results.append({'team': team,
                         'accuracy': log_model_accuracy,
                         'coefficients': log_df})

full_tracking_team_summary = pd.DataFrame({
    'Team': [results['team'] for results in team_results],
    'Accuracy': [results['accuracy'] for results in team_results]
})


sns.barplot(data = full_tracking_team_summary, x = 'Team', y = 'Accuracy')
plt.title('Team Predictibility Scores')
plt.xlabel('Teams')
plt.ylabel('Accuracy')
plt.ylim(0.55, 0.85)
plt.xticks(rotation = 45)
plt.show()


categorical_variables = ['offenseFormation', 'receiverAlignment', 'quarter', 'down', 
                         'motionSinceLineset', 'time_bucket', 'player1_nflId', 'player2_nflId', 'player3_nflId']
numeric_variables = ['score_diff', 'yardsToGo', 'absoluteYardlineNumber',
                     'timeHuddleToLineSet', 'timeLineSetToMotion', 'timeLineSetToSnap', 'player1_netLeftDistanceChange', 
                     'player2_netLeftDistanceChange', 'player3_netLeftDistanceChange', 'player1_netRightDistanceChange', 
                     'player2_netRightDistanceChange', 'player3_netRightDistanceChange']
variables = ['offenseFormation', 'receiverAlignment', 'quarter', 'down', 'score_diff', 
             'yardsToGo', 'absoluteYardlineNumber', 'motionSinceLineset', 'time_bucket', 
             'player1_nflId', 'player2_nflId', 'player3_nflId', 'timeHuddleToLineSet', 'timeLineSetToMotion', 
             'timeLineSetToSnap', 'player1_netLeftDistanceChange', 
             'player2_netLeftDistanceChange', 'player3_netLeftDistanceChange', 'player1_netRightDistanceChange', 
            'player2_netRightDistanceChange', 'player3_netRightDistanceChange']
target = 'rush'

team_results = []
teams = dist_data_merge['teamAbbr'].unique()

for team in teams:
    print(f"Training model for {team}")

    team_data = dist_data_merge[dist_data_merge['teamAbbr'] == team]

    X = team_data[variables]
    y = team_data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    ## One Hot Encode the categorical values
    preprocessing = ColumnTransformer(
        transformers = [('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_variables),
                        ('num', StandardScaler(), numeric_variables)
                        ]
    )
    # Basic Logistic Regression Model
    log_model = Pipeline(
        steps = [
            ('preprocessor', preprocessing),
            ('classifier', LogisticRegression(solver = 'liblinear'))
        ]
    )
    # Fit the data
    log_model.fit(X_train, y_train)
    # Make a prediction
    y_predict = log_model.predict(X_test)
    # Accuracy
    log_model_accuracy = accuracy_score(y_test, y_predict)

    # Get the model and its steps
    logistic_regression_model = log_model.named_steps['classifier']
    preprocessor = log_model.named_steps['preprocessor']

    categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_variables)
    all_features = list(categorical_features) + numeric_variables

    coefficents = logistic_regression_model.coef_[0]
    intercepts = logistic_regression_model.intercept_[0]

    log_df = pd.DataFrame({
        'Feature': all_features,
        'Coefficient': coefficents
    }).sort_values(by = 'Coefficient', key = abs, ascending = False)

    team_results.append({'team': team,
                         'accuracy': log_model_accuracy,
                         'coefficients': log_df})

less_pc_team_summary = pd.DataFrame({
    'Team': [results['team'] for results in team_results],
    'Accuracy': [results['accuracy'] for results in team_results]
})


sns.barplot(data = less_pc_team_summary, x = 'Team', y = 'Accuracy')
plt.title('Team Predictibility Scores')
plt.xlabel('Teams')
plt.ylabel('Accuracy')
plt.ylim(0.5, 0.85)
plt.xticks(rotation = 45)
plt.show()


categorical_variables = ['offenseFormation', 'receiverAlignment', 'quarter', 'down', 
                         'motionSinceLineset', 'time_bucket']
numeric_variables = ['score_diff', 'yardsToGo', 'absoluteYardlineNumber',
                     'timeHuddleToLineSet', 'timeLineSetToMotion', 'timeLineSetToSnap']
variables = ['offenseFormation', 'receiverAlignment', 'quarter', 'down', 'score_diff', 
             'yardsToGo', 'absoluteYardlineNumber', 'motionSinceLineset', 'time_bucket', 
             'timeHuddleToLineSet', 'timeLineSetToMotion', 
             'timeLineSetToSnap']
target = 'rush'

team_results = []
teams = dist_data_merge['teamAbbr'].unique()

for team in teams:
    print(f"Training model for {team}")

    team_data = dist_data_merge[dist_data_merge['teamAbbr'] == team]

    X = team_data[variables]
    y = team_data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    ## One Hot Encode the categorical values
    preprocessing = ColumnTransformer(
        transformers = [('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_variables),
                        ('num', StandardScaler(), numeric_variables)
                        ]
    )
    # Basic Logistic Regression Model
    log_model = Pipeline(
        steps = [
            ('preprocessor', preprocessing),
            ('classifier', LogisticRegression(solver = 'liblinear'))
        ]
    )
    # Fit the data
    log_model.fit(X_train, y_train)
    # Make a prediction
    y_predict = log_model.predict(X_test)
    # Accuracy
    log_model_accuracy = accuracy_score(y_test, y_predict)

    # Get the model and its steps
    logistic_regression_model = log_model.named_steps['classifier']
    preprocessor = log_model.named_steps['preprocessor']

    categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_variables)
    all_features = list(categorical_features) + numeric_variables

    coefficents = logistic_regression_model.coef_[0]
    intercepts = logistic_regression_model.intercept_[0]

    log_df = pd.DataFrame({
        'Feature': all_features,
        'Coefficient': coefficents
    }).sort_values(by = 'Coefficient', key = abs, ascending = False)

    team_results.append({'team': team,
                         'accuracy': log_model_accuracy,
                         'coefficients': log_df})

less_tracking_team_summary = pd.DataFrame({
    'Team': [results['team'] for results in team_results],
    'Accuracy': [results['accuracy'] for results in team_results]
})


sns.barplot(data = less_tracking_team_summary, x = 'Team', y = 'Accuracy')
plt.title('Team Predictibility Scores')
plt.xlabel('Teams')
plt.ylabel('Accuracy')
plt.ylim(0.5, 0.85)
plt.xticks(rotation = 45)
plt.show()


categorical_variables = ['offenseFormation', 'receiverAlignment', 'quarter', 'down', 
                         'motionSinceLineset', 'time_bucket']
numeric_variables = ['score_diff', 'yardsToGo', 'absoluteYardlineNumber']
variables = ['offenseFormation', 'receiverAlignment', 'quarter', 'down', 'score_diff', 
             'yardsToGo', 'absoluteYardlineNumber', 'motionSinceLineset', 'time_bucket']
target = 'rush'

team_results = []
teams = dist_data_merge['teamAbbr'].unique()

for team in teams:
    print(f"Training model for {team}")

    team_data = dist_data_merge[dist_data_merge['teamAbbr'] == team]

    X = team_data[variables]
    y = team_data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    ## One Hot Encode the categorical values
    preprocessing = ColumnTransformer(
        transformers = [('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_variables),
                        ('num', StandardScaler(), numeric_variables)
                        ]
    )
    # Basic Logistic Regression Model
    log_model = Pipeline(
        steps = [
            ('preprocessor', preprocessing),
            ('classifier', LogisticRegression(solver = 'liblinear'))
        ]
    )
    # Fit the data
    log_model.fit(X_train, y_train)
    # Make a prediction
    y_predict = log_model.predict(X_test)
    # Accuracy
    log_model_accuracy = accuracy_score(y_test, y_predict)

    # Get the model and its steps
    logistic_regression_model = log_model.named_steps['classifier']
    preprocessor = log_model.named_steps['preprocessor']

    categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_variables)
    all_features = list(categorical_features) + numeric_variables

    coefficents = logistic_regression_model.coef_[0]
    intercepts = logistic_regression_model.intercept_[0]

    log_df = pd.DataFrame({
        'Feature': all_features,
        'Coefficient': coefficents
    }).sort_values(by = 'Coefficient', key = abs, ascending = False)

    team_results.append({'team': team,
                         'accuracy': log_model_accuracy,
                         'coefficients': log_df})

basic_team_summary = pd.DataFrame({
    'Team': [results['team'] for results in team_results],
    'Accuracy': [results['accuracy'] for results in team_results]
})


sns.barplot(data = basic_team_summary, x = 'Team', y = 'Accuracy')
plt.title('Team Predictibility Scores')
plt.xlabel('Teams')
plt.ylabel('Accuracy')
plt.ylim(0.5, 0.85)
plt.xticks(rotation = 45)
plt.show()


# Combine all summaries into one DataFrame for direct comparison
comparison_df = pd.DataFrame({
    'Team': full_tracking_team_summary['Team'],
    'Full Tracking': full_tracking_team_summary['Accuracy'],
    'Less PC': less_pc_team_summary['Accuracy'],
    'Less Tracking': less_tracking_team_summary['Accuracy'],
    'Basic': basic_team_summary['Accuracy']
})

# Display the comparison DataFrame
print("Comparison of Model Accuracies Across Summaries:")
print(comparison_df)

# Visualization: Comparative Performance Across Summaries
plt.figure(figsize=(12, 8))
width = 0.2
x = np.arange(len(comparison_df['Team']))

plt.bar(x - 1.5*width, comparison_df['Basic'], width, label='Basic', color='blue')
plt.bar(x - 0.5*width, comparison_df['Less Tracking'], width, label='Less Tracking', color='orange')
plt.bar(x + 0.5*width, comparison_df['Less PC'], width, label='Less PC', color='green')
plt.bar(x + 1.5*width, comparison_df['Full Tracking'], width, label='Full Tracking', color='purple')

plt.title('Comparison of Model Accuracies Across Summaries')
plt.xlabel('Teams')
plt.ylabel('Accuracy')
plt.xticks(x, comparison_df['Team'], rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

