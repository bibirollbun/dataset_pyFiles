import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import math
import statistics as stat
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


supplementary_data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv')


nested_groups_dict={}

all_input_dfs = []

for week in range(1, 19):
    filename = f"/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w{week:02d}.csv"   # zero-padded (01, 02, …, 18)
    df = pd.read_csv(filename)
    all_input_dfs.append(df)
combined_input_dfs = pd.concat(all_input_dfs, ignore_index=True)
# combined_df
grouped_input_data = combined_input_dfs.groupby(['game_id', 'play_id'])

# groups_dict = {play: group for play, group in grouped_data}
for (game_id, play_id), group_df in grouped_input_data: 
    if game_id not in nested_groups_dict:
        nested_groups_dict[game_id] = {}

    nested_groups_dict[game_id][play_id] = {
        "input": group_df
    }


all_outputs_dfs = []

for week in range(1, 19):
    filename = f"/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/output_2023_w{week:02d}.csv"   # zero-padded (01, 02, …, 18)
    df = pd.read_csv(filename)
    all_outputs_dfs.append(df)
combined_outputs_df = pd.concat(all_outputs_dfs, ignore_index=True)
# combined_df
grouped_output_data = combined_outputs_df.groupby(['game_id', 'play_id'])
# groups_output_dict = {play: group for play, group in grouped_data}
for (game_id, play_id), group_df in grouped_output_data: 
    nested_groups_dict[game_id][play_id]["output"] = group_df


supplementary_2023=supplementary_data[supplementary_data['season']==2023]
grouped_supplementary_data=supplementary_2023.groupby(['game_id', 'play_id'])
for (game_id, play_id), group_df in grouped_supplementary_data: 
    nested_groups_dict[game_id][play_id]["supplementary"] = group_df


for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        datasets_dict.setdefault("metrics", {})
        play_output=datasets_dict['output']
        unique_ids=[]
        for _,fram in play_output.iterrows():
            if fram['nfl_id'] not in unique_ids:
                unique_ids.append(fram['nfl_id'])
        if len(unique_ids)>1:
            supplementary_play=datasets_dict['supplementary']    
            route = supplementary_play['route_of_targeted_receiver'].iloc[0]
            play_input=datasets_dict['input']
            target = play_input.loc[play_input['player_role'] == 'Targeted Receiver', 'nfl_id']

            receiving_player_id = int(target.iloc[0])
            results=[]
            
            for frame_id in play_output['frame_id'].unique():
                frame_data = play_output[play_output['frame_id'] == frame_id]
                    
                defender_position_x = []
                defender_position_y = []
                defensive_player_list = []
        
                for _, frame in frame_data.iterrows():
                    if frame['nfl_id'] == receiving_player_id:  
                        receiver_position_x = frame['x']
                        receiver_position_y = frame['y']
                    else:
                        defender_position_x.append(frame['x'])
                        defender_position_y.append(frame['y'])
                        defensive_player_list.append(frame['nfl_id'])
                distance_list = []
                for i in range(len(defensive_player_list)):
                    sep = math.sqrt((receiver_position_x - defender_position_x[i])**2 + (receiver_position_y - defender_position_y[i])**2)
                    distance_list.append(sep)
                min_separation = min(distance_list)
                defensive_player_index = distance_list.index(min_separation)
                defending_player_id = defensive_player_list[defensive_player_index]
        
                results.append({
                        'frame_id': frame_id,
                        'receiving_player_id': receiving_player_id,
                        'closest_defender_id': defending_player_id,
                        'separation': min_separation
                    })
        
            frame_analysis_df = pd.DataFrame(results)
            sep_diff=float((frame_analysis_df['separation'][-1:]-frame_analysis_df['separation'][0]).iloc[0])
            num_frames=len(frame_analysis_df)
            sep_diff_per_sec=sep_diff/(num_frames/10)

            datasets_dict["metrics"]["separation"] = sep_diff_per_sec


separation_list_zone=[]
separation_list_man=[]

for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        if 'separation' in datasets_dict['metrics']:  
            # print(datasets_dict['supplementary']['team_coverage_man_zone'].iloc[0])
            if datasets_dict['supplementary']['team_coverage_man_zone'].iloc[0]=='ZONE_COVERAGE':
                separation_list_zone.append(datasets_dict['metrics']['separation'])
            else:
                separation_list_man.append(datasets_dict['metrics']['separation'])
    #     separation=datasets_dict['metrics']['separation']
    #     sum_separation+=separation
    #     play_count+=1
# average_separation=sum_separation/play_count
mean_sep_zone=stat.mean(separation_list_zone)
std_sep_zone=stat.stdev(separation_list_zone)
mean_sep_man=stat.mean(separation_list_man)
std_sep_man=stat.stdev(separation_list_man)


for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        if 'separation' in datasets_dict['metrics']:   
            sep_diff_per_sec=datasets_dict['metrics']['separation']
            if datasets_dict['supplementary']['team_coverage_man_zone'].iloc[0]=='ZONE_COVERAGE':
                sep_z_zone = (sep_diff_per_sec - mean_sep_zone) / std_sep_zone
                datasets_dict["metrics"]["separation_z_score"] = sep_z_zone
            else:
                sep_z_man=(sep_diff_per_sec - mean_sep_man) / std_sep_man
                datasets_dict["metrics"]["separation_z_score"] = sep_z_man


for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        datasets_dict.setdefault("player information", {})
        play_input = datasets_dict['input']
        target_name = play_input.loc[play_input['player_role'] == 'Targeted Receiver', 'player_name']
        receiving_player_name = target_name.iloc[0]
        datasets_dict["player information"]["receiving player name"] = receiving_player_name


player_sep_dict = {}

for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        if 'separation' in datasets_dict['metrics']: 
            player_name = datasets_dict['player information']['receiving player name']
            if player_name in player_sep_dict:
                sep = datasets_dict['metrics']['separation_z_score']
            if player_name not in player_sep_dict:
                player_sep_dict[player_name] = []

            player_sep_dict[player_name].append(sep)


player_sep_df = pd.DataFrame([
    {"player_name": p, "avg_sep_z": sum(vals)/len(vals), "num_plays": len(vals)}
    for p, vals in player_sep_dict.items()
]).sort_values("avg_sep_z", ascending=False)


player_sep_df


player_sep_dict_mz = {}

for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        if 'separation_z_score' in datasets_dict['metrics']: 
            player_name = datasets_dict['player information']['receiving player name']
            coverage = datasets_dict['supplementary']['team_coverage_man_zone'].iloc[0]
            sep_z = datasets_dict['metrics']['separation_z_score']
            if coverage=='MAN_COVERAGE':
                cov_key = 'Man'
            else:
                cov_key = 'Zone'
    
            if player_name not in player_sep_dict_mz:
                player_sep_dict_mz[player_name] = {'Man': [], 'Zone': []}
            player_sep_dict_mz[player_name][cov_key].append(sep_z)


player_sep_stats = []
for player, cvg in player_sep_dict_mz.items():
    player_sep_stats.append({
        "player_name": player,
        "mean_man": stat.mean(cvg['Man']) if cvg['Man'] else None,
        "mean_zone": stat.mean(cvg['Zone']) if cvg['Zone'] else None,
        "num_man": len(cvg['Man']),
        "num_zone": len(cvg['Zone'])
    })
player_sep_df = pd.DataFrame(player_sep_stats)


common_sep_players=player_sep_df[(player_sep_df['num_man']>20) & (player_sep_df['num_zone']>20)]
common_sep_players


X = common_sep_players[['mean_man', 'mean_zone']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


k = 6
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to DataFrame
common_sep_players['cluster'] = clusters+1


plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=common_sep_players,
    x='mean_man',
    y='mean_zone',
    hue='cluster',
    palette='Set2',
    s=100,
    alpha=0.9
)

plt.title('K-Means Clustering of Players by Separation (Man vs Zone)')
plt.xlabel('Mean Separation (Man Coverage)')
plt.ylabel('Mean Separation (Zone Coverage)')
plt.legend(title='Cluster')
plt.show()


sse = []
list_k = list(range(1, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(X_scaled)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')


common_sep_players[
    (common_sep_players['player_name'] == 'Drake London') | 
    (common_sep_players['player_name'] == 'Rashee Rice')
]


player_max_speed_dict={}
for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        play_input=datasets_dict['input']
        receiving_player_name=datasets_dict['player information']['receiving player name']
        # print(receiving_player_name)
        speed_list=[]
        for _,fram in play_input.iterrows():
            if fram['player_name']==receiving_player_name:
                # print(fram['s'])
                speed_list.append(fram['s'])
        # print(speed_list)
        max_speed=max(speed_list)
        if receiving_player_name in player_max_speed_dict:
            if max_speed>player_max_speed_dict[receiving_player_name]:
                player_max_speed_dict[receiving_player_name]= max_speed
        if receiving_player_name not in player_max_speed_dict:
            player_max_speed_dict[receiving_player_name]= max_speed


for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        distance_from_optimal_list = []
        speed_list = []
        uncatchable_throw=False
        
        play_output = datasets_dict['output']
        play_input=datasets_dict['input']
        
        target = play_input.loc[play_input['player_role'] == 'Targeted Receiver', 'nfl_id']
        receiving_player_id = int(target.iloc[0])

        receiver_input = play_input[play_input['nfl_id'] == receiving_player_id].reset_index(drop=True)
        receiving_player_output=play_output[play_output['nfl_id']==receiving_player_id].reset_index(drop=True)
        
        time_til_land = len(receiving_player_output)

        ball_land_x=play_input['ball_land_x'].iloc[0]
        ball_land_y=play_input['ball_land_y'].iloc[0]
        
        first_frame_direction = receiver_input[-1:]['dir'].iloc[0]
        player_name=receiver_input[-1:]['player_name'].iloc[0]

        theta_v = math.radians(first_frame_direction)

        max_turn_deg = 15  # max turn angle per frame in degrees
        max_velocity=player_max_speed_dict[player_name]
        prev_dire = theta_v   # to store previous frame's direction in radians

        for i, frame in receiving_player_output.iterrows():
            x_pos = frame['x']
            y_pos = frame['y']

            if i != 0:
                # print(i)
                per_frame_dist = math.sqrt((optimal_x_pos - x_pos) ** 2 + (optimal_y_pos - y_pos) ** 2)
                distance_from_optimal_list.append(per_frame_dist)
            distance = math.sqrt((ball_land_x - x_pos) ** 2 + (ball_land_y - y_pos) ** 2)
            time = (time_til_land - i) / 10
            optimal_speed = distance / time
            if i==0:
                if optimal_speed>max_velocity:
                    uncatchable_throw=True
                    break
            dx = ball_land_x - x_pos
            dy = ball_land_y - y_pos

            # Desired direction toward the ball
            desired_dire = math.atan2(dy, dx)  # radians

            if prev_dire is None:
            # First frame, no previous direction; just use desired direction
                new_dire = desired_dire
            else:
            # Calculate difference in direction in degrees (wrapped between -180 to 180)
                diff_deg = math.degrees(desired_dire - prev_dire)
                diff_deg = (diff_deg + 180) % 360 - 180  # wrap between -180 and 180

        # Limit the turn angle
                if diff_deg > max_turn_deg:
                    diff_deg = max_turn_deg
                elif diff_deg < -max_turn_deg:
                    diff_deg = -max_turn_deg

        # Update new direction by limited turn
                new_dire = prev_dire + math.radians(diff_deg)

    # Update previous direction for next iteration
            prev_dire = new_dire

    # Calculate new optimal position using smoothed direction
            optimal_x_pos = x_pos + optimal_speed * math.cos(new_dire) * 0.1
            optimal_y_pos = y_pos + optimal_speed * math.sin(new_dire) * 0.1

        if len(distance_from_optimal_list)>0:
            RMSE = np.sqrt(np.mean(np.square(distance_from_optimal_list)))
        
            datasets_dict["metrics"]['route_path_RMSE']=RMSE


RMSE_list=[]
for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        if 'route_path_RMSE' in datasets_dict['metrics']:
            RMSE_list.append(datasets_dict['metrics']['route_path_RMSE'])
mean_RMSE=stat.mean(RMSE_list)
min_RMSE=min(RMSE_list)
max_RMSE=max(RMSE_list)
std_RMSE=stat.stdev(RMSE_list)


for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        if 'route_path_RMSE' in datasets_dict['metrics']:   
            route_path_RMSE=datasets_dict['metrics']['route_path_RMSE']
            route_path = (route_path_RMSE - mean_RMSE) / std_RMSE
            datasets_dict["metrics"]["route_path"] = route_path


player_route_path_dict={}
for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
            if 'route_path_RMSE' in datasets_dict['metrics']:   
                player_name = datasets_dict['player information']['receiving player name']
                route_path=datasets_dict['metrics']['route_path']
                if player_name not in player_route_path_dict:
                    player_route_path_dict[player_name]=[]
                player_route_path_dict[player_name].append(route_path)


player_route_path_stats=[]
for player,route_path in player_route_path_dict.items():
    player_route_path_stats.append({
        "player_name":player,
        "mean_rp_score":stat.mean(route_path),
        "num_plays":len(route_path)    
    })
player_route_path_df=pd.DataFrame(player_route_path_stats)


top_receivers_routes_df=player_route_path_df[player_route_path_df['num_plays']>70]
sorted_trr_df=top_receivers_routes_df.sort_values(by='mean_rp_score',ascending=True)


player_route_path_coi = {}

for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        if 'route_path_RMSE' in datasets_dict['metrics']: 
            player_name = datasets_dict['player information']['receiving player name']
    
            coi = datasets_dict['supplementary']['pass_result'].iloc[0]
            route_path = datasets_dict['metrics']['route_path_RMSE']
            if coi=='C':
                outcome_key = 'Complete'
            else:
                outcome_key = 'Incomplete'
    
            if player_name not in player_route_path_coi:
                player_route_path_coi[player_name] = {'Complete': [], 'Incomplete': []}
            player_route_path_coi[player_name][outcome_key].append(route_path)


player_rp_stats = []
for player, outcome in player_route_path_coi.items():
    player_rp_stats.append({
        "player_name": player,
        "mean_complete": stat.mean(outcome['Complete']) if outcome['Complete'] else None,
        "mean_incomplete": stat.mean(outcome['Incomplete']) if outcome['Incomplete'] else None,
        "num_complete": len(outcome['Complete']),
        "num_incomplete": len(outcome['Incomplete'])
    })
player_rp_df = pd.DataFrame(player_rp_stats)


player_rp_df[player_rp_df['num_complete']>50]


records = []

for game_id, plays in nested_groups_dict.items():
    for play_id, data in plays.items():
        if 'route_path_RMSE' in data['metrics']: 
            records.append({
                'game_id': game_id,
                'play_id': play_id,
                'rp_RMSE': data['metrics']['route_path_RMSE']
            })



df = pd.DataFrame(records)

df['fan_friendly_rp_grade'] = (
    (1-df['rp_RMSE']
      .rank(ascending=True, pct=True))
      * 100
)


for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
            if 'route_path_RMSE' in datasets_dict['metrics']:  
                row = df[(df['game_id'] == game_id) &(df['play_id'] == play_id)]
                fan_grade = row.iloc[0]['fan_friendly_rp_grade']
                datasets_dict['metrics']['fan_friendly_rp_grade']=fan_grade


friendly_player_route_path_dict={}
for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
            if 'route_path_RMSE' in datasets_dict['metrics']:   
                player_name = datasets_dict['player information']['receiving player name']
                friendly_route_path=datasets_dict['metrics']['fan_friendly_rp_grade']
                if player_name not in friendly_player_route_path_dict:
                    friendly_player_route_path_dict[player_name]=[]
                friendly_player_route_path_dict[player_name].append(friendly_route_path)


player_route_path_stats=[]
for player,route_path in friendly_player_route_path_dict.items():
    player_route_path_stats.append({
        "player_name":player,
        "mean_rp_score":stat.mean(route_path),
        "num_plays":len(route_path)    
    })
player_route_path_df=pd.DataFrame(player_route_path_stats)


player_route_path_df[player_route_path_df['num_plays']>75].sort_values(by='mean_rp_score')


records = []

for game_id, plays in nested_groups_dict.items():
    for play_id, data in plays.items():
        if 'separation' in data['metrics']: 
            records.append({
                'game_id': game_id,
                'play_id': play_id,
                'separation_z_score': data['metrics']['separation_z_score']
            })


df = pd.DataFrame(records)

df['fan_friendly_separation_grade'] = (
    (df['separation_z_score']
      .rank(ascending=True, pct=True))
      * 100
)


for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
            if 'separation' in datasets_dict['metrics']:  
                row = df[(df['game_id'] == game_id) &(df['play_id'] == play_id)]
                fan_grade = row.iloc[0]['fan_friendly_separation_grade']
                datasets_dict['metrics']['fan_friendly_separation_grade']=fan_grade


for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        if 'separation' in datasets_dict['metrics'] and 'route_path' in datasets_dict['metrics']:
            rp_grade=datasets_dict['metrics']['fan_friendly_rp_grade']
            sep_grade=datasets_dict['metrics']['fan_friendly_separation_grade']
            ball_track_grade=(rp_grade+sep_grade)/2
            datasets_dict['metrics']['overall_ball_tracking_grade']=ball_track_grade


ball_track_dict={}
for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
            if 'overall_ball_tracking_grade' in datasets_dict['metrics']:   
                player_name = datasets_dict['player information']['receiving player name']
                ball_track=datasets_dict['metrics']['overall_ball_tracking_grade']
                if player_name not in ball_track_dict:
                    ball_track_dict[player_name]=[]
                ball_track_dict[player_name].append(ball_track)


ball_track_stats=[]
for player,track in ball_track_dict.items():
    ball_track_stats.append({
        "player_name":player,
        "mean_ball_track_score":stat.mean(track),
        "num_plays":len(track)    
    })
player_ball_track_df=pd.DataFrame(ball_track_stats)


player_ball_track_df[player_ball_track_df['num_plays']>80].sort_values(by='mean_ball_track_score',ascending=False)


ball_track_coi = {}

for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        if 'overall_ball_tracking_grade' in datasets_dict['metrics']: 
            player_name = datasets_dict['player information']['receiving player name']
    
            coi = datasets_dict['supplementary']['pass_result'].iloc[0]
            ball_track = datasets_dict['metrics']['overall_ball_tracking_grade']
            if coi=='C':
                outcome_key = 'Complete'
            else:
                outcome_key = 'Incomplete'
    
            if player_name not in ball_track_coi:
                ball_track_coi[player_name] = {'Complete': [], 'Incomplete': []}
            ball_track_coi[player_name][outcome_key].append(ball_track)


player_bt_stats = []
for player, outcome in ball_track_coi.items():
    player_bt_stats.append({
        "player_name": player,
        "mean_complete": stat.mean(outcome['Complete']) if outcome['Complete'] else None,
        "mean_incomplete": stat.mean(outcome['Incomplete']) if outcome['Incomplete'] else None,
        "num_complete": len(outcome['Complete']),
        "num_incomplete": len(outcome['Incomplete'])
    })
player_bt_df = pd.DataFrame(player_bt_stats)


player_bt_df[player_bt_df['num_complete']>50]


ball_track_complete=[]
ball_track_incomplete=[]
for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        if 'overall_ball_tracking_grade' in datasets_dict['metrics']: 
            if datasets_dict['supplementary']['pass_result'].iloc[0]=='C':
                ball_track_complete.append(datasets_dict['metrics']['overall_ball_tracking_grade'])
            else:
                ball_track_incomplete.append(datasets_dict['metrics']['overall_ball_tracking_grade'])
print("Mean ball tracking grade for complete passes:", stat.mean(ball_track_complete))
print("Mean ball tracking grade for incomplete and intercepted passes:",stat.mean(ball_track_incomplete))


both_stats_dict = {}

for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        if 'overall_ball_tracking_grade' in datasets_dict['metrics']: 
            player_name = datasets_dict['player information']['receiving player name']
    
            separation=datasets_dict['metrics']['fan_friendly_separation_grade']
            route_path=datasets_dict['metrics']['fan_friendly_rp_grade']
            if player_name not in both_stats_dict:
                both_stats_dict[player_name] = {'separation': [], 'route_path': []}
            both_stats_dict[player_name]['separation'].append(separation)
            both_stats_dict[player_name]['route_path'].append(route_path)


player_bs_stats = []
for player, outcome in both_stats_dict.items():
    player_bs_stats.append({
        "player_name": player,
        "mean_separation": stat.mean(outcome['separation']) if outcome['separation'] else None,
        "mean_route_path": stat.mean(outcome['route_path']) if outcome['route_path'] else None,
        "num_plays": len(outcome['separation'])
    })
player_bt_df = pd.DataFrame(player_bs_stats)


common_bt_players=player_bt_df[player_bt_df['num_plays']>70]


X = common_bt_players[['mean_separation', 'mean_route_path']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


k = 6
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to DataFrame
common_bt_players['cluster'] = clusters+1


plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=common_bt_players,
    x='mean_separation',
    y='mean_route_path',
    hue='cluster',
    palette='Set2',
    s=100,
    alpha=0.9
)

plt.title('K-Means Clustering of Players by Separation and Route Path Scores')
plt.xlabel('Mean Separation Score')
plt.ylabel('Mean Route Path Score')
plt.legend(title='Cluster')
plt.show()


sse = []
list_k = list(range(1, 15))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(X_scaled)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')


common_bt_players


import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, FancyArrow, Wedge
from IPython.display import HTML
from scipy.spatial.distance import cdist

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("=" * 80)
print("NFL BIG DATA BOWL 2026 - COMPLETE ANALYSIS & ANIMATION SUITE")
print("=" * 80)

# Load data
print("\nLOADING DATA...")

supp_data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv')
input_w01 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w01.csv')
output_w01 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/output_2023_w01.csv')

print(f"Supplementary: {supp_data.shape}")
print(f"Input Week 1: {input_w01.shape}")
print(f"Output Week 1: {output_w01.shape}")

# Get available plays
available_plays = input_w01[['game_id', 'play_id']].drop_duplicates()
print(f"\nFound {len(available_plays)} plays in Week 1")

# Helper functions
def create_football_field(ax, los=None, first_down=None):
    """Create football field visualization"""
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    rect = Rectangle((0, 0), 120, 53.3, linewidth=2, 
                     edgecolor='black', facecolor='green', alpha=0.3)
    ax.add_patch(rect)
    
    for yard in range(10, 111, 10):
        ax.plot([yard, yard], [0, 53.3], color='gray', linewidth=1, alpha=0.5)
    
    for yard in range(10, 111, 1):
        ax.plot([yard, yard], [23.36, 23.36], color='gray', linewidth=0.5, alpha=0.3)
        ax.plot([yard, yard], [29.94, 29.94], color='gray', linewidth=0.5, alpha=0.3)
    
    rect1 = Rectangle((0, 0), 10, 53.3, linewidth=2, 
                      edgecolor='black', facecolor='lightgray', alpha=0.3)
    rect2 = Rectangle((110, 0), 10, 53.3, linewidth=2, 
                      edgecolor='black', facecolor='lightgray', alpha=0.3)
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    
    if los is not None:
        ax.axvline(x=los, color='blue', linewidth=3, linestyle='--', label='LOS', alpha=0.8)
    if first_down is not None:
        ax.axvline(x=first_down, color='orange', linewidth=3, linestyle='--', label='1st Down', alpha=0.8)
    
    ax.set_xlabel('Yards', fontsize=12)
    ax.set_ylabel('Width (yards)', fontsize=12)
    return ax

# Visualization 1: Man vs Zone Coverage Analysis
print("\n" + "=" * 80)
print("VISUALIZATION 1: MAN VS ZONE COVERAGE ANALYSIS")
print("=" * 80)

def analyze_man_vs_zone(supp_data):
    """Compare man and zone coverage effectiveness"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Man vs Zone Coverage Analysis', fontsize=20, weight='bold', y=0.98)
    
    coverage_data = supp_data[supp_data['team_coverage_man_zone'].notna()]
    
    # 1. Completion Rate
    ax = axes[0, 0]
    comp_rate = coverage_data.groupby('team_coverage_man_zone')['pass_result'].apply(
        lambda x: (x == 'C').sum() / len(x) * 100
    )
    bars = ax.bar(comp_rate.index, comp_rate.values, 
                  color=['steelblue', 'coral'], alpha=0.7)
    ax.set_ylabel('Completion %', fontsize=12)
    ax.set_title('Completion Rate', fontsize=14, weight='bold')
    ax.grid(alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11)
    
    # 2. Average Yards Allowed
    ax = axes[0, 1]
    avg_yards = coverage_data.groupby('team_coverage_man_zone')['yards_gained'].mean()
    bars = ax.bar(avg_yards.index, avg_yards.values,
                  color=['steelblue', 'coral'], alpha=0.7)
    ax.set_ylabel('Avg Yards', fontsize=12)
    ax.set_title('Average Yards Allowed', fontsize=14, weight='bold')
    ax.grid(alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11)
    
    # 3. Interception Rate
    ax = axes[0, 2]
    int_rate = coverage_data.groupby('team_coverage_man_zone')['pass_result'].apply(
        lambda x: (x == 'IN').sum() / len(x) * 100
    )
    bars = ax.bar(int_rate.index, int_rate.values,
                  color=['steelblue', 'coral'], alpha=0.7)
    ax.set_ylabel('Interception %', fontsize=12)
    ax.set_title('Interception Rate', fontsize=14, weight='bold')
    ax.grid(alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=11)
    
    # 4. EPA Distribution
    ax = axes[1, 0]
    for cov_type in coverage_data['team_coverage_man_zone'].unique():
        data = coverage_data[coverage_data['team_coverage_man_zone'] == cov_type]['expected_points_added']
        ax.hist(data, bins=30, alpha=0.6, label=cov_type)
    ax.set_xlabel('EPA', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('EPA Distribution', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 5. Success Rate by Down
    ax = axes[1, 1]
    success_by_down = coverage_data.groupby(['down', 'team_coverage_man_zone'])['pass_result'].apply(
        lambda x: (x == 'C').sum() / len(x) * 100
    ).unstack()
    success_by_down.plot(kind='bar', ax=ax, color=['steelblue', 'coral'], alpha=0.7)
    ax.set_xlabel('Down', fontsize=12)
    ax.set_ylabel('Completion %', fontsize=12)
    ax.set_title('Success Rate by Down', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    
    # 6. Play Usage
    ax = axes[1, 2]
    play_count = coverage_data['team_coverage_man_zone'].value_counts()
    bars = ax.bar(play_count.index, play_count.values,
                  color=['steelblue', 'coral'], alpha=0.7)
    ax.set_ylabel('Number of Plays', fontsize=12)
    ax.set_title('Coverage Usage', fontsize=14, weight='bold')
    ax.grid(alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    return fig

fig1 = analyze_man_vs_zone(supp_data)
plt.show()

# Visualization 2: Press Coverage Analysis
print("\n" + "=" * 80)
print("VISUALIZATION 2: PRESS COVERAGE ANALYSIS")
print("=" * 80)

def analyze_press_coverage(input_data, supp_data, sample_size=500):
    """Analyze press coverage (defenders within 3 yards at snap)"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Press Coverage Analysis (Defenders <= 3 yards at snap)', fontsize=18,
                weight='bold', y=0.98)
    
    press_plays = []
    sample_plays = input_data[['game_id', 'play_id']].drop_duplicates().head(sample_size)
    
    print(f"Analyzing {len(sample_plays)} plays...")
    
    for idx, (_, row) in enumerate(sample_plays.iterrows()):
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(sample_plays)} plays...")
        
        game_id, play_id = row['game_id'], row['play_id']
        play_input = input_data[(input_data['game_id'] == game_id) & 
                                (input_data['play_id'] == play_id)]
        
        first_frame = play_input[play_input['frame_id'] == play_input['frame_id'].min()]
        
        receivers = first_frame[first_frame['player_role'].isin(['Targeted Receiver', 'Other Route Runner'])]
        defenders = first_frame[first_frame['player_side'] == 'Defense']
        
        press_count = 0
        for _, rec in receivers.iterrows():
            for _, def_p in defenders.iterrows():
                dist = np.sqrt((rec['x'] - def_p['x'])**2 + (rec['y'] - def_p['y'])**2)
                if dist <= 3:
                    press_count += 1
                    break
        
        play_supp = supp_data[(supp_data['game_id'] == game_id) & (supp_data['play_id'] == play_id)]
        if not play_supp.empty:
            press_plays.append({
                'press_count': press_count,
                'is_press': press_count >= 1,
                'pass_result': play_supp.iloc[0]['pass_result'],
                'yards_gained': play_supp.iloc[0]['yards_gained']
            })
    
    press_df = pd.DataFrame(press_plays)
    
    # 1. Completion Rate
    ax = axes[0, 0]
    comp_rate = press_df.groupby('is_press')['pass_result'].apply(
        lambda x: (x == 'C').sum() / len(x) * 100
    )
    bars = ax.bar(['No Press', 'Press'], comp_rate.values,
                  color=['green', 'red'], alpha=0.6)
    ax.set_ylabel('Completion %', fontsize=12)
    ax.set_title('Completion Rate', fontsize=14, weight='bold')
    ax.grid(alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11)
    
    # 2. Average Yards
    ax = axes[0, 1]
    avg_yards = press_df.groupby('is_press')['yards_gained'].mean()
    bars = ax.bar(['No Press', 'Press'], avg_yards.values,
                  color=['green', 'red'], alpha=0.6)
    ax.set_ylabel('Avg Yards', fontsize=12)
    ax.set_title('Average Yards Gained', fontsize=14, weight='bold')
    ax.grid(alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11)
    
    # 3. Press Count Distribution
    ax = axes[1, 0]
    press_df['press_count'].hist(bins=range(0, 6), ax=ax, color='gray', alpha=0.6)
    ax.set_xlabel('Number of Receivers in Press', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Press Coverage Distribution', fontsize=14, weight='bold')
    ax.grid(alpha=0.3)
    
    # 4. Yards by Press Count
    ax = axes[1, 1]
    yards_by_press = press_df.groupby('press_count')['yards_gained'].mean()
    bars = ax.bar(yards_by_press.index, yards_by_press.values,
                  color='orange', alpha=0.6)
    ax.set_xlabel('Number of Receivers in Press', fontsize=12)
    ax.set_ylabel('Avg Yards', fontsize=12)
    ax.set_title('Yards by Press Intensity', fontsize=14, weight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

fig2 = analyze_press_coverage(input_w01, supp_data, sample_size=500)
plt.show()

# Visualization 3: Prevent Defense Analysis
print("\n" + "=" * 80)
print("VISUALIZATION 3: PREVENT DEFENSE ANALYSIS")
print("=" * 80)

def analyze_prevent_defense(supp_data):
    """Analyze prevent defense situations (deep coverage, late game)"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Prevent Defense Analysis (Q4, Winning, >10 yard throws)', 
                fontsize=18, weight='bold', y=0.98)
    
    # Define prevent situations: Q4, winning, deep throws
    prevent_df = supp_data[
        (supp_data['quarter'] == 4) &
        (supp_data['pass_length'] > 10) &
        (supp_data['pre_snap_home_score'] != supp_data['pre_snap_visitor_score'])
    ].copy()
    
    # Determine if team is winning
    prevent_df['winning'] = prevent_df.apply(
        lambda x: (x['possession_team'] == x['home_team_abbr'] and 
                  x['pre_snap_home_score'] > x['pre_snap_visitor_score']) or
                 (x['possession_team'] == x['visitor_team_abbr'] and 
                  x['pre_snap_visitor_score'] > x['pre_snap_home_score']),
        axis=1
    )
    
    # 1. Completion Rate vs Score Differential
    ax = axes[0, 0]
    comp_rate = prevent_df.groupby('winning')['pass_result'].apply(
        lambda x: (x == 'C').sum() / len(x) * 100
    )
    bars = ax.bar(['Losing', 'Winning'], comp_rate.values,
                  color=['red', 'green'], alpha=0.6)
    ax.set_ylabel('Completion %', fontsize=12)
    ax.set_title('Prevent: Completion Rate', fontsize=14, weight='bold')
    ax.grid(alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11)
    
    # 2. Yards Per Attempt
    ax = axes[0, 1]
    ypa = prevent_df.groupby('winning')['yards_gained'].mean()
    bars = ax.bar(['Losing', 'Winning'], ypa.values,
                  color=['red', 'green'], alpha=0.6)
    ax.set_ylabel('Yards Per Attempt', fontsize=12)
    ax.set_title('Prevent: YPA', fontsize=14, weight='bold')
    ax.grid(alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11)
    
    # 3. Pass Length Distribution
    ax = axes[1, 0]
    for win in [False, True]:
        data = prevent_df[prevent_df['winning'] == win]['pass_length']
        ax.hist(data, bins=20, alpha=0.6, label='Winning' if win else 'Losing')
    ax.set_xlabel('Pass Length (yards)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Pass Length Distribution', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Big Play Rate (20+ yards)
    ax = axes[1, 1]
    big_plays = prevent_df.groupby('winning')['yards_gained'].apply(
        lambda x: (x >= 20).sum() / len(x) * 100
    )
    bars = ax.bar(['Losing', 'Winning'], big_plays.values,
                  color=['red', 'green'], alpha=0.6)
    ax.set_ylabel('Big Play %', fontsize=12)
    ax.set_title('Prevent: Big Plays (20+ yards)', fontsize=14, weight='bold')
    ax.grid(alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    return fig

fig3 = analyze_prevent_defense(supp_data)
plt.show()

# Visualization 4: Interception Analysis
print("\n" + "=" * 80)
print("VISUALIZATION 4: INTERCEPTION ANALYSIS")
print("=" * 80)

def analyze_interceptions(supp_data):
    """Analyze interception patterns"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Interception Analysis', fontsize=20, weight='bold', y=0.98)
    
    int_df = supp_data[supp_data['pass_result'] == 'IN']
    all_passes = supp_data[supp_data['pass_result'].notna()]
    
    # 1. INT Rate by Coverage Type
    ax = axes[0, 0]
    int_by_cov = all_passes.groupby('team_coverage_man_zone')['pass_result'].apply(
        lambda x: (x == 'IN').sum() / len(x) * 100
    )
    bars = ax.bar(int_by_cov.index, int_by_cov.values, color='red', alpha=0.6)
    ax.set_ylabel('INT %', fontsize=12)
    ax.set_title('INT Rate by Coverage', fontsize=14, weight='bold')
    ax.grid(alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 2. INT Rate by Down
    ax = axes[0, 1]
    int_by_down = all_passes.groupby('down')['pass_result'].apply(
        lambda x: (x == 'IN').sum() / len(x) * 100
    )
    bars = ax.bar(int_by_down.index.astype(str), int_by_down.values, color='red', alpha=0.6)
    ax.set_xlabel('Down', fontsize=12)
    ax.set_ylabel('INT %', fontsize=12)
    ax.set_title('INT Rate by Down', fontsize=14, weight='bold')
    ax.grid(alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 3. Pass Length on INTs
    ax = axes[0, 2]
    ax.hist(int_df['pass_length'].dropna(), bins=20, color='red', alpha=0.6)
    ax.axvline(int_df['pass_length'].mean(), color='blue', linestyle='--', 
               linewidth=2, label=f"Mean: {int_df['pass_length'].mean():.1f} yds")
    ax.set_xlabel('Pass Length (yards)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Pass Length on INTs', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. INTs by Quarter
    ax = axes[1, 0]
    int_by_qtr = int_df['quarter'].value_counts().sort_index()
    bars = ax.bar(int_by_qtr.index.astype(str), int_by_qtr.values, color='red', alpha=0.6)
    ax.set_xlabel('Quarter', fontsize=12)
    ax.set_ylabel('Number of INTs', fontsize=12)
    ax.set_title('INTs by Quarter', fontsize=14, weight='bold')
    ax.grid(alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    # 5. Yards to Go on INTs
    ax = axes[1, 1]
    ax.hist(int_df['yards_to_go'].dropna(), bins=15, color='red', alpha=0.6)
    ax.set_xlabel('Yards to Go', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Yards to Go on INTs', fontsize=14, weight='bold')
    ax.grid(alpha=0.3)
    
    # 6. Formation on INTs
    ax = axes[1, 2]
    form_ints = int_df['offense_formation'].value_counts().head(6)
    bars = ax.barh(range(len(form_ints)), form_ints.values, color='red', alpha=0.6)
    ax.set_yticks(range(len(form_ints)))
    ax.set_yticklabels(form_ints.index)
    ax.set_xlabel('Number of INTs', fontsize=12)
    ax.set_title('Top Formations on INTs', fontsize=14, weight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig

fig4 = analyze_interceptions(supp_data)
plt.show()

# Visualization 5: Defensive Flocking Analysis
print("\n" + "=" * 80)
print("VISUALIZATION 5: DEFENSIVE FLOCKING ANALYSIS")
print("=" * 80)

def analyze_flocking(input_data, output_data, supp_data, game_id, play_id):
    """Analyze defensive flocking behavior (defenders moving toward ball)"""
    play_input = input_data[(input_data['game_id'] == game_id) & 
                            (input_data['play_id'] == play_id)]
    play_output = output_data[(output_data['game_id'] == game_id) & 
                              (output_data['play_id'] == play_id)]
    play_supp = supp_data[(supp_data['game_id'] == game_id) & 
                          (supp_data['play_id'] == play_id)]
    
    if play_supp.empty or play_input.empty:
        print("No play data found")
        return None
    
    play_supp = play_supp.iloc[0]
    all_frames = pd.concat([play_input, play_output])
    los = play_input['absolute_yardline_number'].iloc[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Defensive Flocking Analysis\nGame {game_id}, Play {play_id}',
                fontsize=18, weight='bold', y=0.98)
    
    ball_x = play_input['ball_land_x'].iloc[0]
    ball_y = play_input['ball_land_y'].iloc[0]
    
    # Calculate flocking metrics per frame
    flocking_data = []
    for frame_id in sorted(all_frames['frame_id'].unique()):
        frame_data = all_frames[all_frames['frame_id'] == frame_id]
        defenders = frame_data[frame_data['player_side'] == 'Defense']
        
        if len(defenders) > 0:
            distances = np.sqrt((defenders['x'] - ball_x)**2 + 
                              (defenders['y'] - ball_y)**2)
            
            if len(defenders) > 1:
                positions = defenders[['x', 'y']].values
                dist_matrix = cdist(positions, positions)
                cohesion = np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])
            else:
                cohesion = 0
            
            flocking_data.append({
                'frame_id': frame_id,
                'avg_dist_to_ball': distances.mean(),
                'min_dist_to_ball': distances.min(),
                'cohesion': cohesion,
                'num_defenders': len(defenders)
            })
    
    flock_df = pd.DataFrame(flocking_data)
    
    # 1. Distance to Ball Over Time
    ax = axes[0, 0]
    ax.plot(flock_df['frame_id'], flock_df['avg_dist_to_ball'], 
            color='blue', linewidth=2, marker='o', markersize=4, label='Avg Distance')
    ax.plot(flock_df['frame_id'], flock_df['min_dist_to_ball'],
            color='red', linewidth=2, marker='s', markersize=4, label='Min Distance')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Distance (yards)', fontsize=12)
    ax.set_title('Defender Distance to Ball', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Defensive Cohesion
    ax = axes[0, 1]
    ax.plot(flock_df['frame_id'], flock_df['cohesion'],
            color='purple', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Cohesion (avg distance)', fontsize=12)
    ax.set_title('Defensive Cohesion Over Time', fontsize=14, weight='bold')
    ax.grid(alpha=0.3)
    
    # 3. Convergence Rate
    ax = axes[1, 0]
    flock_df['convergence_rate'] = -flock_df['avg_dist_to_ball'].diff()
    ax.plot(flock_df['frame_id'], flock_df['convergence_rate'],
            color='green', linewidth=2, marker='o', markersize=4)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Convergence Rate (yards/frame)', fontsize=12)
    ax.set_title('Flocking Speed', fontsize=14, weight='bold')
    ax.grid(alpha=0.3)
    

def animate_play(input_data, output_data, supp_data, game_id, play_id):
    """Create animated visualization of a complete play"""
    play_input = input_data[(input_data['game_id'] == game_id) & 
                            (input_data['play_id'] == play_id)]
    play_output = output_data[(output_data['game_id'] == game_id) & 
                              (output_data['play_id'] == play_id)]
    play_supp = supp_data[(supp_data['game_id'] == game_id) & 
                          (supp_data['play_id'] == play_id)]
    
    if play_supp.empty or play_input.empty:
        print("No play data found")
        return None
    
    play_supp = play_supp.iloc[0]
    all_frames = pd.concat([play_input, play_output])
    frames_list = sorted(all_frames['frame_id'].unique())
    
    print(f"Creating animation with {len(frames_list)} frames...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ball_x = play_input['ball_land_x'].iloc[0]
    ball_y = play_input['ball_land_y'].iloc[0]
    los = play_input['absolute_yardline_number'].iloc[0]
    
    def update(frame_idx):
        ax.clear()
        frame = frames_list[frame_idx]
        
        first_down = los + play_supp['yards_to_go']
        create_football_field(ax, los, first_down)
        
        frame_data = all_frames[all_frames['frame_id'] == frame]
        
        offense = frame_data[frame_data['player_side'] == 'Offense']
        defense = frame_data[frame_data['player_side'] == 'Defense']
        passer = frame_data[frame_data['player_role'] == 'Passer']
        target = frame_data[frame_data['player_role'] == 'Targeted Receiver']
        
        ax.scatter(offense['x'], offense['y'], c='red', s=200,
                  marker='o', edgecolors='black', linewidth=2,
                  label='Offense', alpha=0.8, zorder=3)
        
        ax.scatter(defense['x'], defense['y'], c='blue', s=200,
                  marker='^', edgecolors='black', linewidth=2,
                  label='Defense', alpha=0.8, zorder=3)
        
        if not passer.empty:
            ax.scatter(passer['x'], passer['y'], c='gold', s=400,
                      marker='*', edgecolors='black', linewidth=2,
                      label='QB', zorder=4)
        
        if not target.empty:
            ax.scatter(target['x'], target['y'], c='lime', s=300,
                      marker='D', edgecolors='black', linewidth=2,
                      label='Target', zorder=4)
        
        for _, player in frame_data.iterrows():
            if not pd.isna(player['s']) and player['s'] > 0:
                dx = player['s'] * np.cos(np.radians(player['dir'])) * 1.5
                dy = player['s'] * np.sin(np.radians(player['dir'])) * 1.5
                ax.arrow(player['x'], player['y'], dx, dy,
                        head_width=1, head_length=0.5,
                        fc='yellow', ec='yellow', alpha=0.6, linewidth=1.5)
        
        if frame_idx > len(play_input['frame_id'].unique()):
            ax.scatter(ball_x, ball_y, c='brown', s=200,
                      marker='o', edgecolors='black', linewidth=2,
                      label='Ball', zorder=5)
        
        title = f"Game {game_id} | Play {play_id}\n"
        title += f"Down: {play_supp['down']} | Yards to Go: {play_supp['yards_to_go']} | "
        title += f"Result: {play_supp['pass_result']} | Yards: {play_supp['yards_gained']}\n"
        title += f"Coverage: {play_supp.get('team_coverage_type', 'N/A')}"
        ax.set_title(title, fontsize=12, pad=20, weight='bold')
        
        ax.legend(loc='upper left', fontsize=9, ncol=2)
        
        ax.text(5, 50, f'Frame: {frame_idx+1}/{len(frames_list)}',
               fontsize=14, weight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    anim = FuncAnimation(fig, update, frames=range(len(frames_list)),
                        interval=100, repeat=True, blit=False)
    
    return anim

example_game = available_plays.iloc[0]['game_id']
example_play = available_plays.iloc[0]['play_id']

print(f"Animating Game {example_game}, Play {example_play}...")
anim1 = animate_play(input_w01, output_w01, supp_data, example_game, example_play)

if anim1:
    display(HTML(anim1.to_html5_video()))
    plt.close()

# Animation 2: Coverage Shell Animation
print("\n" + "=" * 80)
print("ANIMATION 2: COVERAGE SHELL VISUALIZATION")
print("=" * 80)

def animate_coverage_shell(input_data, output_data, supp_data, game_id, play_id):
    """Animate with coverage shell visualization"""
    play_input = input_data[(input_data['game_id'] == game_id) & 
                            (input_data['play_id'] == play_id)]
    play_output = output_data[(output_data['game_id'] == game_id) & 
                              (output_data['play_id'] == play_id)]
    play_supp = supp_data[(supp_data['game_id'] == game_id) & 
                          (supp_data['play_id'] == play_id)]
    
    if play_supp.empty or play_input.empty:
        print("No play data found")
        return None
    
    play_supp = play_supp.iloc[0]
    all_frames = pd.concat([play_input, play_output])
    frames_list = sorted(all_frames['frame_id'].unique())
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    los = play_input['absolute_yardline_number'].iloc[0]
    
    def update(frame_idx):
        ax.clear()
        frame = frames_list[frame_idx]
        
        create_football_field(ax, los, los + play_supp['yards_to_go'])
        
        frame_data = all_frames[all_frames['frame_id'] == frame]
        
        offense = frame_data[frame_data['player_side'] == 'Offense']
        defense = frame_data[frame_data['player_side'] == 'Defense']
        
        ax.scatter(offense['x'], offense['y'], c='red', s=200,
                  marker='o', edgecolors='black', linewidth=2,
                  label='Offense', alpha=0.8, zorder=3)
        
        for _, def_player in defense.iterrows():
            ax.scatter(def_player['x'], def_player['y'], c='blue', s=200,
                      marker='^', edgecolors='black', linewidth=2, alpha=0.8, zorder=3)
            
            circle = Circle((def_player['x'], def_player['y']), 5,
                          color='blue', alpha=0.2, zorder=1)
            ax.add_patch(circle)
        
        title = f"Coverage Shell: {play_supp.get('team_coverage_type', 'N/A')}\n"
        title += f"Man/Zone: {play_supp.get('team_coverage_man_zone', 'N/A')} | Frame: {frame_idx+1}/{len(frames_list)}"
        ax.set_title(title, fontsize=14, pad=20, weight='bold')
        
        ax.legend(loc='upper left', fontsize=10)
    
    anim = FuncAnimation(fig, update, frames=range(len(frames_list)),
                        interval=100, repeat=True, blit=False)
    
    return anim

anim2 = animate_coverage_shell(input_w01, output_w01, supp_data, example_game, example_play)

if anim2:
    display(HTML(anim2.to_html5_video()))
    plt.close()

# Create sample visualizations
print("\n" + "=" * 60)
print("CREATING SAMPLE VISUALIZATIONS")
print("=" * 60)

if all([input_w01 is not None, output_w01 is not None, supp_data is not None]):
    
    sample_game = available_plays.iloc[0]['game_id']
    sample_play = available_plays.iloc[0]['play_id']
    
    print(f"\nGenerating visualizations for Game {sample_game}, Play {sample_play}...")
    
    print("\nCreating animations for first 3 plays...")
    animations_created = []
    try:
        for i in range(min(3, len(available_plays))):
            game_id = available_plays.iloc[i]['game_id']
            play_id = available_plays.iloc[i]['play_id']
            
            print(f"\nAnimating Play {i+1}: Game {game_id}, Play {play_id}")
            anim = animate_play(input_w01, output_w01, supp_data, game_id, play_id)
            
            if anim:
                filename = f'play_animation_{game_id}_{play_id}.html'
                with open(filename, 'w') as f:
                    f.write(anim.to_jshtml())
                print(f"Saved: {filename}")
                animations_created.append(filename)
                
                try:
                    gif_file = f'play_animation_{game_id}_{play_id}.gif'
                    anim.save(gif_file, writer='pillow', fps=10, dpi=100)
                    print(f"Saved: {gif_file}")
                    animations_created.append(gif_file)
                except:
                    print("Could not save GIF for this play")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)


# ============================================================
# SAVE ALL STATIC FIGURES
# ============================================================

import os

OUTPUT_DIR = "./workflow_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

figures_to_save = {
    "viz1_man_vs_zone.png": fig1,
    "viz2_press_coverage.png": fig2,
    "viz3_prevent_defense.png": fig3,
    "viz4_interceptions.png": fig4,
}

for filename, fig in figures_to_save.items():
    if fig is not None:
        fig.savefig(
            os.path.join(OUTPUT_DIR, filename),
            dpi=300,
            bbox_inches="tight"
        )

print("All static figures saved to:", OUTPUT_DIR)



from matplotlib.animation import PillowWriter
import os

# Directory for workflow outputs
output_dir = './workflow_outputs/'
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# ANIMATION 1: FULL PLAY
# ----------------------------
play_gif  = os.path.join(output_dir, f'play_animation_{example_game}_{example_play}.gif')
play_html = os.path.join(output_dir, f'play_animation_{example_game}_{example_play}.html')

anim1.save(play_gif, writer=PillowWriter(fps=10))
with open(play_html, 'w') as f:
    f.write(anim1.to_jshtml())

print("ANIMATION 1: FULL PLAY VISUALIZATION SAVED")
print("   ", play_gif)
print("   ", play_html)


# ----------------------------
# ANIMATION 2: COVERAGE SHELL
# ----------------------------
coverage_gif  = os.path.join(output_dir, f'coverage_shell_{example_game}_{example_play}.gif')
coverage_html = os.path.join(output_dir, f'coverage_shell_{example_game}_{example_play}.html')

anim2.save(coverage_gif, writer=PillowWriter(fps=10))
with open(coverage_html, 'w') as f:
    f.write(anim2.to_jshtml())

print("✅ ANIMATION 2: COVERAGE SHELL VISUALIZATION SAVED")
print("   ", coverage_gif)
print("   ", coverage_html)


