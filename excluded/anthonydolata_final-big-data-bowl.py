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


# NFL Big Data Bowl 2026 - Complete Data Loading and Analysis
import numpy as np
import pandas as pd
import os

# Display all files in input directory
print("=" * 60)
print("ğŸ“� AVAILABLE FILES")
print("=" * 60)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

print("\n" + "=" * 60)
print("ğŸ“Š LOADING DATA")
print("=" * 60)

# ============================================================
# LOAD SUPPLEMENTARY DATA
# ============================================================
supp_data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv')

print(f"\nâœ… Supplementary data loaded: {supp_data.shape[0]} rows, {supp_data.shape[1]} columns")
print(f"\nColumns available: {supp_data.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(supp_data.head())

# ============================================================
# OPTIONAL: LOAD PLAYER DATA (if exists)
# ============================================================
try:
    players = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/players.csv')
    print(f"\nâœ… Players data loaded: {players.shape[0]} rows, {players.shape[1]} columns")
except:
    print("\nâš ï¸� Players data not found - continuing without player names")
    players = None

# ============================================================
# OPTIONAL: LOAD SAMPLE TRACKING DATA (Week 1 as example)
# ============================================================
try:
    input_w01 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w01.csv')
    output_w01 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/output_2023_w01.csv')
    print(f"\nâœ… Week 1 tracking data loaded")
    print(f"   Input shape: {input_w01.shape}")
    print(f"   Output shape: {output_w01.shape}")
except:
    print("\nâš ï¸� Tracking data not loaded")

# ============================================================
# CATCH %, YARDS, AND PLAY OUTCOME STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("ğŸ“Š BALL OUTCOME, CATCH %, YARDS, AND RECEIVER METRICS")
print("=" * 60)

# Ensure required columns exist
required_cols = ["pass_result", "yards_gained"]
missing_cols = [col for col in required_cols if col not in supp_data.columns]
if missing_cols:
    print(f"\nâ�Œ ERROR: Missing columns: {missing_cols}")
    print(f"Available columns: {supp_data.columns.tolist()}")
else:
    # ---------------------------------------
    # 1. Basic Play Outcome Percentages
    # ---------------------------------------
    total_passes = len(supp_data)
    
    catch_pct = (supp_data["pass_result"] == "C").mean() * 100
    inc_pct   = (supp_data["pass_result"] == "I").mean() * 100
    int_pct   = (supp_data["pass_result"] == "IN").mean() * 100 if "IN" in supp_data["pass_result"].unique() else 0
    touchdown_pct = (supp_data["pass_result"] == "TD").mean() * 100 if "TD" in supp_data["pass_result"].unique() else 0
    
    print("\n===== PASS RESULT SUMMARY =====")
    print(f"Total pass plays: {total_passes}")
    print(f"Catch %:            {catch_pct:.2f}%")
    print(f"Incompletion %:     {inc_pct:.2f}%")
    print(f"Interception %:     {int_pct:.2f}%")
    print(f"Touchdown %:        {touchdown_pct:.2f}%")
    
    # ---------------------------------------
    # 2. Yardage Summary
    # ---------------------------------------
    print("\n===== YARDAGE SUMMARY =====")
    
    avg_yards = supp_data["yards_gained"].mean()
    median_yards = supp_data["yards_gained"].median()
    max_yards = supp_data["yards_gained"].max()
    min_yards = supp_data["yards_gained"].min()
    
    print(f"Average yards gained:  {avg_yards:.2f}")
    print(f"Median yards gained:   {median_yards:.2f}")
    print(f"Max yards gained:      {max_yards}")
    print(f"Min yards gained:      {min_yards}")
    
    # ---------------------------------------
    # 3. Receiver-Specific Metrics (Targets, Catches, Catch %)
    # ---------------------------------------
    if "target_nfl_id" in supp_data.columns:
        print("\n===== RECEIVER TARGET METRICS =====")
        
        receiver_stats = (
            supp_data.groupby("target_nfl_id")
            .agg(
                targets=("target_nfl_id", "count"),
                catches=("pass_result", lambda x: (x == "C").sum()),
                yards=("yards_gained", "mean")
            )
        )
        
        receiver_stats["catch_pct"] = receiver_stats["catches"] / receiver_stats["targets"] * 100
        
        # Add player names if available
        if players is not None and "display_name" in players.columns and "nfl_id" in players.columns:
            receiver_stats = receiver_stats.merge(
                players[["nfl_id", "display_name"]],
                left_index=True,
                right_on="nfl_id",
                how="left"
            ).set_index("display_name")
        
        print("\nTop 10 Most Targeted Receivers:")
        print(receiver_stats.sort_values("targets", ascending=False).head(10))
    
    # ---------------------------------------
    # 4. Coverage-Type Summary
    # ---------------------------------------
    if "team_coverage_type" in supp_data.columns:
        print("\n===== COVERAGE TYPE PERFORMANCE =====")
        
        coverage_stats = (
            supp_data.groupby("team_coverage_type")
            .agg(
                plays=("team_coverage_type", "count"),
                catch_pct=("pass_result", lambda x: (x == "C").mean() * 100),
                avg_yards=("yards_gained", "mean")
            )
        )
        
        print(coverage_stats)
    
    # ---------------------------------------
    # 5. EPA-style Summary (optional if EPA column exists)
    # ---------------------------------------
    if "expected_points_added" in supp_data.columns:
        print("\n===== EPA SUMMARY =====")
        epa_summary = supp_data["expected_points_added"].describe()
        print(epa_summary)
    
    # ---------------------------------------
    # 6. Pass Result Distribution
    # ---------------------------------------
    print("\n===== PASS RESULT DISTRIBUTION =====")
    print(supp_data["pass_result"].value_counts())
    
    # ---------------------------------------
    # 7. Summary Printout
    # ---------------------------------------
    print("\n" + "=" * 60)
    print("âœ… Stats Calculated:")
    print("  - Catch %, incompletion %, interception %, TD %")
    print("  - Average, median, max yards gained")
    print("  - Receiver-level: targets, catches, catch %")
    print("  - Coverage-type performance (if available)")
    print("  - Optional EPA summaries (if available)")
    print("=" * 60)

print("\nğŸ�‰ Analysis complete!")


supplementary_data=df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv')


import pandas as pd

nested_groups_dict={}

all_input_dfs = []

for week in range(1, 19):
    filename = f"/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w{week:02d}.csv"   # zero-padded (01, 02, â€¦, 18)
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
    filename = f"/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/output_2023_w{week:02d}.csv"   # zero-padded (01, 02, â€¦, 18)
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


import math
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
            # print(route)
            # print(game_id,play_id)
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
                # print(distance_list)
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


import statistics as stat

separation_list_zone=[]
separation_list_man=[]

for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        if datasets_dict['metrics']['separation']:  
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
        if datasets_dict['metrics']['separation']:   
            sep_diff_per_sec=datasets_dict['metrics']['separation']
            if datasets_dict['supplementary']['team_coverage_man_zone'].iloc[0]=='ZONE_COVERAGE':
                sep_z_zone = (sep_diff_per_sec - mean_sep_zone) / std_sep_zone
                datasets_dict["metrics"]["separation z score"] = sep_z_zone
            else:
                sep_z_man=(sep_diff_per_sec - mean_sep_man) / std_sep_man
                datasets_dict["metrics"]["separation z score"] = sep_z_man


for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        datasets_dict.setdefault("player information", {})
        play_input = datasets_dict['input']
        target_name = play_input.loc[play_input['player_role'] == 'Targeted Receiver', 'player_name']
        receiving_player_name = target_name.iloc[0]
        datasets_dict["player information"]["receiving player name"] = receiving_player_name


player_sep_dict_mz = {}

for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        if datasets_dict['metrics']: 
            player_name = datasets_dict['player information']['receiving player name']
    
            coverage = datasets_dict['supplementary']['team_coverage_man_zone'].iloc[0]
            sep_z = datasets_dict['metrics']['separation z score']
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


player_sep_dict_mz = {}

for game_id, plays_dict in nested_groups_dict.items():
    for play_id, datasets_dict in plays_dict.items():
        if datasets_dict['metrics']: 
            player_name = datasets_dict['player information']['receiving player name']
    
            coverage = datasets_dict['supplementary']['team_coverage_man_zone'].iloc[0]
            sep_z = datasets_dict['metrics']['separation z score']
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


import pandas as pd
from sklearn.preprocessing import StandardScaler


from sklearn.cluster import KMeans

k = 6
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to DataFrame
common_sep_players['cluster'] = clusters+1


import matplotlib.pyplot as plt
import seaborn as sns

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
plt.ylabel('Sum of squared distance');




