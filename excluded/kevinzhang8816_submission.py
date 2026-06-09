'''
Kevin Zhang
'''


import pandas as pd 
import numpy as np
import os
import seaborn as sns 
from collections import Counter
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.animation as animation
from matplotlib import rc
rc('animation', html='html5')


pd.set_option('display.max_rows', None)
path = '/kaggle/input/nfl-big-data-bowl-2025/'


#What each player did on each play
player_play_df = pd.read_csv(path+'player_play.csv')
player_play_df.head()


#Plays where an offensive player went in motion
motion_playersId_df = player_play_df[player_play_df['motionSinceLineset'] == True]
motion_playersId_df.head()


#playId and gameId
motion_playId_df = motion_playersId_df[['gameId', 'playId']].drop_duplicates()
motion_playId_df.head()


#The plays and results
plays_df = pd.read_csv(path+'plays.csv')
plays_df.head()


#Split into two dfs, one where there is a player in motion, one without
motion_plays_df = pd.merge(motion_playId_df[['gameId', 'playId']], plays_df, on=['gameId', 'playId'], how='inner')
non_motion_plays_df = pd.merge(plays_df, motion_playId_df[['gameId', 'playId']], on=['gameId', 'playId'], how='outer', indicator=True)
non_motion_plays_df = non_motion_plays_df[non_motion_plays_df['_merge'] == 'left_only'].drop(columns=['_merge'])


print("Number of plays with motion: " + str(motion_plays_df.shape[0]))
print("Average EPA with motion: " + str(motion_plays_df['expectedPointsAdded'].mean()))
print("Number of plays without motion: " + str(non_motion_plays_df.shape[0]))
print("Average EPA without motion: " + str(non_motion_plays_df['expectedPointsAdded'].mean()))


#Tracking data for a week
#Further analysis will only concern Week 1 of 2022
tracking_df=pd.read_csv(path+'tracking_week_1.csv')
tracking_df.head()


#Get the frames of the player in motion between line_set and ball_snap are kept
tracking_motion_df = pd.merge(tracking_df, motion_playersId_df[['gameId', 'playId', 'nflId']], on=['gameId', 'playId', 'nflId'], how='inner')
tracking_motion_df = tracking_motion_df[tracking_motion_df["frameType"] != "AFTER_SNAP"]
line_set_df = tracking_motion_df[tracking_motion_df['event'] == 'line_set']
line_set_df = line_set_df[['gameId', 'playId', 'nflId', 'frameId']]
tracking_motion_df = tracking_motion_df.merge(line_set_df, on=['gameId', 'playId', 'nflId'], suffixes=['_play_frame', '_lineset_frame'])
tracking_motion_df = tracking_motion_df[tracking_motion_df['frameId_play_frame'] >= tracking_motion_df['frameId_lineset_frame']]
tracking_motion_df = tracking_motion_df.drop(columns=['frameId_lineset_frame'])
tracking_motion_df.head()


#This will keep tracki of where the ball is placed at the snap, so player coordinates will be adjusted relative to the football
football_pos = tracking_df[tracking_df["displayName"] == "football"]
football_pos = football_pos[football_pos['frameType'] == "SNAP"]
football_pos = football_pos[['gameId', 'playId', 'x', 'y']]
football_pos['x'] = football_pos['x'].round(2)
football_pos['y'] = football_pos['y'].round(2)
football_pos.head()


#Groups all x and y coordinates into a single column so operations will be performed easier
grouped_tracking_df = tracking_motion_df.groupby(['gameId', 'playId', 'nflId']).agg({
    'x': list,
    'y': list,
    's': list,
    'dir': list,
    'playDirection': 'first'
}).reset_index()

grouped_tracking_df.head()


#The following cell contains operations performed to prepare the data for k-means clustering.

#Account for position relative to football
mod_tracking_df = grouped_tracking_df.merge(football_pos, on=['gameId', 'playId'], suffixes=['_player_pos', '_football_pos'])
mod_tracking_df['x'] = mod_tracking_df.apply(lambda row: [round(val - row['x_football_pos'], 2) for val in row['x_player_pos']], axis=1)
mod_tracking_df['y'] = mod_tracking_df.apply(lambda row: [round(val - row['y_football_pos'], 2) for val in row['y_player_pos']], axis=1)
mod_tracking_df = mod_tracking_df.drop(columns=['x_player_pos', 'y_player_pos', 'x_football_pos', 'y_football_pos'])

#Whether player is at top of the screen
mod_tracking_df['top_of_screen'] = mod_tracking_df['y'].apply(lambda y: y[0] > 0)
#Adjust x and y based on player location and play direction
mod_tracking_df['x'] = mod_tracking_df.apply(lambda row: [i * -1 for i in row['x']] if row['playDirection']=="left" else row['x'], axis=1)
mod_tracking_df['y'] = mod_tracking_df.apply(lambda row: [i * -1 for i in row['y']] if not row['top_of_screen'] else row['y'], axis=1)

#Rotate the direction for consistency
#Upfield: 0+-k*360
#Sideline: 90+-k*360
#Backwards: 180+-k*360
#Middle of field: 270+-k*360
def rotate_dir(dir_list, play_direction, top_of_screen):
    if play_direction == 'right':
        if top_of_screen:
            return [90 - d for d in dir_list]
        else:
            return [d - 90 for d in dir_list]
    else:
        if top_of_screen:
            return [d + 90 for d in dir_list]
        else:
            return [(-d - 90) for d in dir_list]

mod_tracking_df['dir'] = mod_tracking_df.apply(
    lambda row: rotate_dir(row['dir'], row['playDirection'], row['top_of_screen']), axis=1
)

#Take sine and cosine of direction (This is done to make a 1 degree turn similar to 359 degrees)
mod_tracking_df['dir_x'] = mod_tracking_df['dir'].apply(lambda angles: [round(np.cos(np.radians(angle)), 3) for angle in angles])
mod_tracking_df['dir_y'] = mod_tracking_df['dir'].apply(lambda angles: [round(np.sin(np.radians(angle)), 3) for angle in angles])

#Only presnap plays of 15 seconds or less allowed
mod_tracking_df = mod_tracking_df[mod_tracking_df['x'].apply(len) <= 150]
mod_tracking_df = mod_tracking_df[mod_tracking_df['y'].apply(len) <= 150]

#Most frames in a play remaining
most_frames = mod_tracking_df['x'].apply(len).max()

#The following is done to make the amount of frames in each play consistent: 
#Frames are added before the tracking starts for shorter plays
#These added frames will have position as the same as the first frame, but the speed of the player is 0
def add_first_frames(lst, most_frames):
    padding = [lst[0]] * (most_frames - len(lst))
    return padding + lst

def add_zero_frames(lst, most_frames):
    padding = [0] * (most_frames - len(lst))
    return padding + lst

mod_tracking_df['x'] = mod_tracking_df['x'].apply(lambda x: add_first_frames(x, most_frames))
mod_tracking_df['y'] = mod_tracking_df['y'].apply(lambda y: add_first_frames(y, most_frames))
mod_tracking_df['s'] = mod_tracking_df['s'].apply(lambda s: add_zero_frames(s, most_frames))

#Merge speed and direction to get directional speed
mod_tracking_df['dir_x'] = mod_tracking_df['dir_x'].apply(lambda s: add_first_frames(s, most_frames))
mod_tracking_df['dir_y'] = mod_tracking_df['dir_y'].apply(lambda s: add_first_frames(s, most_frames))
mod_tracking_df['s_x'] = mod_tracking_df.apply(lambda row: [s * dx for s, dx in zip(row['s'], row['dir_x'])], axis=1)
mod_tracking_df['s_y'] = mod_tracking_df.apply(lambda row: [s * dy for s, dy in zip(row['s'], row['dir_y'])], axis=1)

#Other cleanup
mod_tracking_df = mod_tracking_df.drop(columns=['s', 'dir', 'dir_x', 'dir_y', 'playDirection', 'top_of_screen'])
mod_tracking_df = mod_tracking_df.reset_index(drop=True)

#Get rid of all players ahead of the LOS or are 3 plus yards in the backfield
mod_tracking_df = mod_tracking_df[mod_tracking_df['x'].apply(lambda lst: lst[0] < 0 and lst[0] > -3)]
#Get rid of all players who run onto the field or are next to the offensive line
mod_tracking_df = mod_tracking_df[mod_tracking_df['y'].apply(lambda lst: lst[0] < 27 and lst[0] > 5)]
#We will be trying to isolate motions by players lined up as a receiver or tight end

mod_tracking_df.head()


#The grouped data is now ungrouped in order to further prepare for the clustering algorithm
x_frame = pd.DataFrame(mod_tracking_df['x'].to_list())
x_frame.columns = [f'x_{i}' for i in range(x_frame.shape[1])]
final_tracking_df = pd.concat([mod_tracking_df.drop(columns=['x']), x_frame], axis=1)

y_frame = pd.DataFrame(mod_tracking_df['y'].to_list())
y_frame.columns = [f'y_{i}' for i in range(y_frame.shape[1])]
final_tracking_df = pd.concat([final_tracking_df.drop(columns=['y']), y_frame], axis=1)


sx_frame = pd.DataFrame(mod_tracking_df['s_x'].to_list())
sx_frame.columns = [f's_x_{i}' for i in range(sx_frame.shape[1])]
final_tracking_df = pd.concat([final_tracking_df.drop(columns=['s_x']), sx_frame], axis=1)

diry_frame = pd.DataFrame(mod_tracking_df['s_y'].to_list())
diry_frame.columns = [f's_y_{i}' for i in range(diry_frame.shape[1])]
final_tracking_df = pd.concat([final_tracking_df.drop(columns=['s_y']), diry_frame], axis=1)

#NaNs from getting rid of players ahead of the LOS
final_tracking_df = final_tracking_df.dropna()

final_tracking_df.head()


#Apply k-means clustering

X = final_tracking_df.iloc[:,3:].values #Turn tracking data into numpy array. All ids are ignored

#Normalize data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(X)

#Create weights for frames
#These weights grow exponentially further into the play so the ending behavior is more heavily weighted
#Through experimentation, we found temperature values of 17 and 20 to have the best results
positional_weights = [np.exp(i/17) for i in range(most_frames)] 
directional_speed_weights = [np.exp(i/20) for i in range(most_frames)]
weights_per_frame = positional_weights*2 + directional_speed_weights*2

#Apply weights
weighted_df = normalized_data * weights_per_frame

#Apply K-Means Clustering
#Through experimentation, 10 had the best results 
kmeans = KMeans(n_clusters=10, random_state=0, n_init = 10)
kmeans.fit(weighted_df)

#Add cluster labels
clustered_tracking_df = final_tracking_df.copy()
clustered_tracking_df['Cluster'] = kmeans.labels_
clustered_tracking_df.head()


#The center of the clusters will be used in football diagram
cluster_centers = kmeans.cluster_centers_
cluster_centers = cluster_centers/weights_per_frame
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(cluster_centers, columns=clustered_tracking_df.columns[3:-1])
cluster_centers.head()


#Put into a format where a gif can be created
melted_clustered_tracking_df = clustered_tracking_df.melt(
        id_vars = ['gameId', 'playId', 'nflId', 'Cluster'],
        value_vars = [col for col in clustered_tracking_df.columns if col.startswith('x')],
        var_name = 'temp',
        value_name = 'x'
)

melted_clustered_tracking_df['y'] = clustered_tracking_df.melt(
    id_vars = ['gameId', 'playId', 'nflId', 'Cluster'],
    value_vars = [col for col in clustered_tracking_df.columns if col.startswith('y')],
    value_name='y'
)['y']

melted_clustered_tracking_df = melted_clustered_tracking_df.drop(columns = 'temp')
melted_clustered_tracking_df = melted_clustered_tracking_df.sort_values(by = ['gameId', 'playId', 'nflId'], ascending = [True, True, True])
melted_clustered_tracking_df = melted_clustered_tracking_df.reset_index(drop = True)
melted_clustered_tracking_df['frameId'] = (melted_clustered_tracking_df.index % most_frames) + 1

melted_clustered_tracking_df.head()


#Play will be shown to start in the middle of the field
x_displacement = 50
y_displacement = 27


#Two functions that will help create GIFs
#These are slightly edited versions of the methods created by Nick Wan
def get_play_by_frame(fid, ax, los, one_play):
  
  ax.cla()
  gid = one_play['gameId'].unique()[0]
  pid = one_play['playId'].unique()[0]

  one_frame = one_play.loc[one_play['frameId']==fid]

  #Plots all players who had a specific motion
  fig1 = sns.scatterplot(x='x',y='y',data=one_frame, hue = 'Cluster',
                         ax=ax, s=50)
  
  # Add football
  fig1.scatter(x_displacement, y_displacement, color = 'brown')
  
  # For all clusters, red dot is average motion
  cluster_nums = one_play['Cluster'].unique()
  for cluster_num in cluster_nums:
    cluster_center_x = cluster_centers.loc[cluster_num, 'x_'+str(fid)]
    cluster_center_y = cluster_centers.loc[cluster_num, 'y_'+str(fid)]
    fig1.scatter(cluster_center_x + x_displacement, cluster_center_y + y_displacement, color = 'red')
    fig1.text(cluster_center_x + x_displacement, cluster_center_y + y_displacement, str(cluster_num))

  fig1.axvline(los, c='k', ls=':')
  fig1.axvline(0, c='k', ls='-')
  fig1.axvline(100, c='k', ls='-')
  fig1.set_title('Motion cluster '  + ', '.join(map(str, cluster_nums)))
  fig1.legend([]).set_visible(False)
  sns.despine(left=True)
  fig1.set_ylabel('')
  fig1.set_yticks([])
  fig1.set_xlim(-10,110)    
  fig1.set_ylim(0,54) 

def animate_play(one_play):    
  one_play = one_play.copy()
  one_play['x'] = one_play['x']+x_displacement # Player locations in one_play are relative to the football
  one_play['y'] = one_play['y']+y_displacement 
  
  # get game and play IDs
  gid = one_play['gameId'].unique()[0]
  pid = one_play['playId'].unique()[0]
  los = 50
  fig = plt.figure(figsize=(14.4, 6.4))
  ax = fig.gca()
  ani = animation.FuncAnimation(fig, get_play_by_frame, 
                                frames=one_play['frameId'].unique().shape[0],
                                interval=100, repeat=True, 
                                fargs=(ax,los,one_play,))
  
  plt.close()
  return ani   


#Get a cluster and animate it
cluster_number = 1
cluster0_df = melted_clustered_tracking_df[melted_clustered_tracking_df['Cluster'] == cluster_number]
animate_play(cluster0_df)


#Useful method for merging proportions less than 4% when graphing pie charts to avoid overlapping text
def adjustDict(myDict):
    total = sum(myDict.values())
    threshold = 0.04 * total
    small_categories = {key: value for key, value in myDict.items() if value < threshold}
    small_categories_sum = sum(small_categories.values())
    myDict = {key: value for key, value in myDict.items() if value >= threshold}  # Remove small counts
    if small_categories_sum != 0:
        myDict['Other'] = small_categories_sum
    return myDict


#Function for perfomring analysis on a function
def cluster_analysis(cluster_df):

    #Get game data for each play
    cluster_play_df = cluster_df[cluster0_df['frameId'] == 1]
    cluster_play_df = cluster_play_df[['gameId', 'playId', 'nflId', 'Cluster']]
    cluster_play_df = pd.merge(cluster_play_df, motion_plays_df, on=['gameId', 'playId'])

    #Create a dictionary called playProportions that counts what type of passes and runs the team called when there was a specific motion
    cluster_run_play_df = cluster_play_df[cluster_play_df['isDropback'] == False]
    cluster_run_play_df.loc[:, 'pff_runConceptPrimary'] = cluster_run_play_df['pff_runConceptPrimary'].apply(lambda x: x.title())
    playProportions = cluster_run_play_df['pff_runConceptPrimary'].value_counts().to_dict()
    playProportions = {key + " run": value for key, value in playProportions.items()}
    cluster_play_pass_df = cluster_play_df[cluster_play_df['isDropback'] == True]
    cluster_pa_df = cluster_play_pass_df[cluster_play_pass_df['playAction'] == True]
    cluster_nonpaPass_df = cluster_play_pass_df[cluster_play_pass_df['playAction'] == False]
    cluster_rpoPass_df = cluster_pa_df[cluster_pa_df['pff_runPassOption'] == 1]
    playProportions['Traditional pass'] = cluster_nonpaPass_df.shape[0]
    playProportions['RPO pass'] = cluster_rpoPass_df.shape[0]
    playProportions['PA pass'] = cluster_pa_df.shape[0]-playProportions['RPO pass'] #RPO passes are considered play action, so this seperates the two
    playProportions = adjustDict(playProportions)

    #Play call pie chart
    labels1 = playProportions.keys()
    sizes1 = playProportions.values()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].pie(sizes1, labels=labels1, autopct='%1.1f%%', startangle=90, labeldistance=1.15, pctdistance=.85)
    axes[0].set_title('Plays chosen with motion type '+str(cluster_number))

    #Get motion player action on each play
    cluster_players_df = cluster_df[cluster0_df['frameId'] == 1]
    cluster_players_df = cluster_players_df[['gameId', 'playId', 'nflId', 'Cluster']]
    cluster_players_df = pd.merge(cluster_players_df, motion_playersId_df, on=['gameId', 'playId', 'nflId'])

    #Dictionary of motion player routes
    list_of_routes_ran = cluster_players_df['routeRan'].dropna().tolist()
    routes_count = Counter(list_of_routes_ran)
    routes_count = adjustDict(routes_count)

    # Create the pie chart
    labels2 = routes_count.keys()
    sizes2 = routes_count.values() 
    axes[1].pie(sizes2, labels=labels2, autopct='%1.1f%%', startangle=90, labeldistance=1.15, pctdistance=.85)
    axes[1].set_title('Motion player route proportions')

    # Display the charts
    plt.tight_layout()
    plt.show()

    #Other simple analysis
    print('Motion player rush attempt rate: ' + str(cluster_players_df['hadRushAttempt'].mean()))
    print('Motion player target rate: ' + str(cluster_players_df['wasTargettedReceiver'].mean()))
    print('Motion player reception rate: ' + str(cluster_players_df['hadPassReception'].mean()))


#Perform analysis on chosen cluster
cluster_analysis(cluster0_df)

