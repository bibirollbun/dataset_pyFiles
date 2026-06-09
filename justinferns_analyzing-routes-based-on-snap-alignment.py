#import pandas, numpy, etc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


#Import given csvs
players = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/players.csv')
plays = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/plays.csv')

#add new variable game_play_id
plays['game_play_id'] = plays['gameId'].astype(str) + (
    '-' + plays['playId'].astype(str))


#combining all tracking files
tracking = []

# Loop through each file and append the DataFrame to the list
for i in range(1, 10):
    player_tracking = pd.read_csv(f'/kaggle/input/nfl-big-data-bowl-2025/tracking_week_{i}.csv')
    player_tracking['Week'] = i
    player_tracking['nflId'] = pd.to_numeric(player_tracking['nflId'], errors='coerce').fillna(0).astype(int)
    player_tracking['game_play_nfl_id'] = player_tracking['gameId'].astype(str) + (
    '-' + player_tracking['playId'].astype(str) + '-' + player_tracking['nflId'].astype(str))
    tracking.append(player_tracking)

# Concatenate all DataFrames into one
tracking = pd.concat(tracking, ignore_index=True)

# Save the combined DataFrame to a single CSV file
tracking.to_csv('combined_tracking.csv')

print("All files have been combined and saved to 'combined_tracking.csv'.")


#get all targeted atlanta receivers 
atl_target_plays = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/player_play.csv')

#Limit to just atlanta players
atl_target_plays = atl_target_plays[atl_target_plays['teamAbbr'] == 'ATL']
atl_target_plays = atl_target_plays[atl_target_plays['wasTargettedReceiver'] == 1]

#limit to just top 3 in targets
atl_target_plays = atl_target_plays[atl_target_plays['nflId'].isin([48374, 53433, 54473])]
number_to_name = {48374: 'Olamide Zaccheaus',53433: 'Kyle Pitts',54473: 'Drake London'}

# Add a new column based on the mapping
atl_target_plays['displayName'] = atl_target_plays['nflId'].map(number_to_name)

atl_target_plays['game_play_nfl_id'] = atl_target_plays['gameId'].astype(str) + (
    '-' + atl_target_plays['playId'].astype(str) + '-' + atl_target_plays['nflId'].astype(str))


#Show what routes each player ran
route_counts = atl_target_plays.groupby(['displayName', 'routeRan']).size().unstack(fill_value=0)

# Bar plot
route_counts.plot(kind='bar', stacked=True)
plt.title("Routes Run by Each Player")
plt.xlabel("Player")
plt.xticks(rotation=30, ha='right')
plt.ylabel("Number of Routes")
plt.legend(title = 'Route', bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.show()


#get all atlanta target tracking
#tracking = pd.read_csv('combined_tracking.csv') #(uncomment this line if you just want to run the code below, should take less time)
atl_target_tracking = tracking[tracking['game_play_nfl_id'].isin(atl_target_plays['game_play_nfl_id'])]
atl_target_tracking.to_csv('atl_target_tracking.csv')


#Gathering data of where they are at snap
atl_target_atsnap = atl_target_tracking[atl_target_tracking['frameType'] == 'SNAP']
color = {'Olamide Zaccheaus': 'blue','Kyle Pitts': 'green', 'Drake London': 'red'}


# Add a new column based on the  color mapping
atl_target_plays['Color'] = atl_target_plays['displayName'].map(color)
atl_target_plays['Color'].fillna('gray')

plt.figure(figsize=(8,6))
plt.scatter(atl_target_atsnap['x'], atl_target_atsnap['y'], c=atl_target_plays['Color'], s=20)
plt.axvline(x=10, color='black', linestyle='--', linewidth=2, label='End Zone Line')
plt.axvspan(-plt.xlim()[1], 10, color='gray', alpha=0.3, label='End Zone')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(0, 120)
plt.ylim(0, 53.3)
plt.title('Scatter Plot of Alignment at Snap')
plt.suptitle('Drake London: Red, Olamide Zaccheaus: Blue, Kyle Pitts: Green')
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.grid(visible=True, which='major', axis = 'x', linestyle='--', linewidth=0.5, color='gray')



#plot all the routes run 
atl_target_atsnap = atl_target_tracking[atl_target_tracking['frameType'] == 'SNAP'] 
atl_target_atsnap = atl_target_atsnap.merge(atl_target_plays[['game_play_nfl_id', 'routeRan']], on='game_play_nfl_id', how='left')


fig, ax = plt.subplots()
for route, group in atl_target_atsnap.groupby('routeRan'):
    ax.scatter(group['x'], group['y'], label=route)
plt.title('Routes Run')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
plt.axvline(x=10, color='black', linestyle='--', linewidth=2, label='End Zone Line')
plt.axvline(x=110, color='black', linestyle='--', linewidth=2, label='End Zone Line')
plt.axvspan(0, 10, color='gray', alpha=0.3, label='Endzone')
plt.axvspan(110, 120, color='gray', alpha=0.3, label='Endzone')
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.show()


#analyzing Drake london Targets

#grab just drake london
atl_dl_targets = atl_target_tracking[atl_target_tracking['frameType'] == 'SNAP'] 
atl_dl_targets = atl_dl_targets[atl_dl_targets['displayName'] == 'Drake London']
dl_target_plays = atl_target_plays[atl_target_plays['nflId'] == 54473]
atl_dl_targets = atl_dl_targets.merge(dl_target_plays[['game_play_nfl_id', 'routeRan']], on='game_play_nfl_id', how='left')
atl_dl_targets['game_play_id'] = atl_dl_targets['gameId'].astype(str) + (
    '-' + atl_dl_targets['playId'].astype(str))
dl_plays = plays[plays['game_play_id'].isin(atl_dl_targets['game_play_id'])]
merged_df = pd.merge(dl_plays, atl_dl_targets, on='game_play_id')

#Apply the equation 120 - x if letters match and x > 60
merged_df['x'] = merged_df.apply(
   lambda row: 120 - row['x'] if row['possessionTeam'] == row['yardlineSide'] and row['x'] > 60 else row['x'],
   axis=1
)

fig, ax = plt.subplots()
for route, group in atl_dl_targets.groupby('routeRan'):
    ax.scatter(group['x'], group['y'], label=route)
plt.title('Drake London')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
plt.axvline(x=10, color='black', linestyle='--', linewidth=2, label='End Zone Line')
plt.axvline(x=110, color='black', linestyle='--', linewidth=2, label='End Zone Line')
plt.axvspan(0, 10, color='gray', alpha=0.3, label='Endzone')
plt.axvspan(110, 120, color='gray', alpha=0.3, label='Endzone')
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.show()


#analyzing Kyle pitts Targets

#grab just Kyle pitts
atl_kp_targets = atl_target_tracking[atl_target_tracking['frameType'] == 'SNAP'] 
atl_kp_targets = atl_kp_targets[atl_kp_targets['displayName'] == 'Kyle Pitts']
kp_target_plays = atl_target_plays[atl_target_plays['nflId'] == 53433]
atl_kp_targets = atl_kp_targets.merge(kp_target_plays[['game_play_nfl_id', 'routeRan']], on='game_play_nfl_id', how='left')
atl_kp_targets['game_play_id'] = atl_kp_targets['gameId'].astype(str) + (
    '-' + atl_kp_targets['playId'].astype(str))
kp_plays = plays[plays['game_play_id'].isin(atl_kp_targets['game_play_id'])]
merged_df = pd.merge(kp_plays, atl_kp_targets, on='game_play_id')

#Apply the equation 120 - x if letters match and x > 60
merged_df['x'] = merged_df.apply(
   lambda row: 120 - row['x'] if row['possessionTeam'] == row['yardlineSide'] and row['x'] > 60 else row['x'],
   axis=1
)


fig, ax = plt.subplots()
for route, group in atl_kp_targets.groupby('routeRan'):
    ax.scatter(group['x'], group['y'], label=route)
plt.title('Kyle Pitts')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
plt.axvline(x=10, color='black', linestyle='--', linewidth=2, label='End Zone Line')
plt.axvline(x=110, color='black', linestyle='--', linewidth=2, label='End Zone Line')
plt.axvspan(0, 10, color='gray', alpha=0.3, label='Endzone')
plt.axvspan(110, 120, color='gray', alpha=0.3, label='Endzone')
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.show()


#analyzing OZs Targets

#grab just OZ
atl_oz_targets = atl_target_tracking[atl_target_tracking['frameType'] == 'SNAP'] 
atl_oz_targets = atl_oz_targets[atl_oz_targets['displayName'] == 'Olamide Zaccheaus']
oz_target_plays = atl_target_plays[atl_target_plays['nflId'] == 48374]

atl_oz_targets = atl_oz_targets.merge(oz_target_plays[['game_play_nfl_id', 'routeRan']], on='game_play_nfl_id', how='left')

atl_oz_targets['game_play_id'] = atl_oz_targets['gameId'].astype(str) + (
    '-' + atl_oz_targets['playId'].astype(str))

oz_plays = plays[plays['game_play_id'].isin(atl_oz_targets['game_play_id'])]
merged_df = pd.merge(oz_plays, atl_oz_targets, on='game_play_id')

#apply the equation 120 - x if letters match and x > 60
merged_df['x'] = merged_df.apply(
   lambda row: 120 - row['x'] if row['possessionTeam'] == row['yardlineSide'] and row['x'] > 60 else row['x'],
   axis=1
)

fig, ax = plt.subplots()
for route, group in merged_df.groupby('routeRan'):
    ax.scatter(group['x'], group['y'], label=route)

plt.title('Olamide Zaccheaus')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
plt.axvline(x=10, color='black', linestyle='--', linewidth=2, label='End Zone Line')
plt.axvline(x=110, color='black', linestyle='--', linewidth=2, label='End Zone Line')
plt.axvspan(0, 10, color='gray', alpha=0.3, label='Endzone')
plt.axvspan(110, 120, color='gray', alpha=0.3, label='Endzone')
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.show()


#calculating distance from center line of the field

centerline_y = 53.3 / 2

# Calculate the absolute distance from the centerline
atl_target_atsnap['distance_from_centerline'] = abs(atl_target_atsnap['y'] - centerline_y)

# Calculate the average distance for each route
average_distances = atl_target_atsnap.groupby('routeRan')['distance_from_centerline'].mean()

# Create a scatter plot
fig, ax = plt.subplots()
routes = average_distances.index
avg_distances = average_distances.values
ax.scatter(routes, avg_distances, color='red', s=100)
plt.title('Average Distance from Centerline (y = 26.65) by Route')
ax.set_xlabel('Route')
ax.set_ylabel('Average Distance from Centerline')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

