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


#take in the teams
m_teams = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')
w_teams = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WTeams.csv')
m_teams['power'] = 0
w_teams['power'] = 0
w_teams
#w_teams
#the power ranking will be similar to an elo ranking in chess


m_games = pd.read_csv('/kaggle/input/mgamecitieseditted/MGameCitiesEditted.csv')
w_games = pd.read_csv('/kaggle/input/wgamecitieseditted/WGameCities.csv')


for _, game in w_games.iterrows():
    if game['Season'] == 2020:
        W_Team = w_teams[w_teams['TeamID'] == game['WTeamID']]
        L_Team = w_teams[w_teams['TeamID'] == game['LTeamID']]
        if not W_Team.empty and W_Team['power'].iloc[0] == 0:  # Ensure 'W_Team' isn't empty
            W_Team.loc[W_Team.index[0], 'power'] = 10
        elo = L_Team.loc[L_Team.index[0], 'power']
        LG = elo * 0.1
        if not W_Team.empty:
            w_teams.loc[w_teams['TeamID'] == game['WTeamID'], 'power'] = W_Team['power'].iloc[0] + LG
        if not L_Team.empty:
            w_teams.loc[w_teams['TeamID'] == game['LTeamID'], 'power'] = L_Team['power'].iloc[0] - LG

    if game['Season'] == 2021:
        W_Team = w_teams[w_teams['TeamID'] == game['WTeamID']]
        L_Team = w_teams[w_teams['TeamID'] == game['LTeamID']]
        if not W_Team.empty and W_Team['power'].iloc[0] == 0:  # Ensure 'W_Team' isn't empty
            W_Team.loc[W_Team.index[0], 'power'] = 10
        elo = L_Team.loc[L_Team.index[0], 'power']
        LG = elo * 0.15
        if not W_Team.empty:
            w_teams.loc[w_teams['TeamID'] == game['WTeamID'], 'power'] = W_Team['power'].iloc[0] + LG
        if not L_Team.empty:
            w_teams.loc[w_teams['TeamID'] == game['LTeamID'], 'power'] = L_Team['power'].iloc[0] - LG

    if game['Season'] == 2022:
        W_Team = w_teams[w_teams['TeamID'] == game['WTeamID']]
        L_Team = w_teams[w_teams['TeamID'] == game['LTeamID']]
        if not W_Team.empty and W_Team['power'].iloc[0] == 0:  # Ensure 'W_Team' isn't empty
            W_Team.loc[W_Team.index[0], 'power'] = 10
        elo = L_Team.loc[L_Team.index[0], 'power']
        LG = elo * 0.2
        if not W_Team.empty:
            w_teams.loc[w_teams['TeamID'] == game['WTeamID'], 'power'] = W_Team['power'].iloc[0] + LG
        if not L_Team.empty:
            w_teams.loc[w_teams['TeamID'] == game['LTeamID'], 'power'] = L_Team['power'].iloc[0] - LG

    if game['Season'] == 2023:
        W_Team = w_teams[w_teams['TeamID'] == game['WTeamID']]
        L_Team = w_teams[w_teams['TeamID'] == game['LTeamID']]
        if not W_Team.empty and W_Team['power'].iloc[0] == 0:  # Ensure 'W_Team' isn't empty
            W_Team.loc[W_Team.index[0], 'power'] = 10
        elo = L_Team.loc[L_Team.index[0], 'power']
        LG = elo * 0.25
        if not W_Team.empty:
            w_teams.loc[w_teams['TeamID'] == game['WTeamID'], 'power'] = W_Team['power'].iloc[0] + LG
        if not L_Team.empty:
            w_teams.loc[w_teams['TeamID'] == game['LTeamID'], 'power'] = L_Team['power'].iloc[0] - LG

    if game['Season'] == 2024:
        W_Team = w_teams[w_teams['TeamID'] == game['WTeamID']]
        L_Team = w_teams[w_teams['TeamID'] == game['LTeamID']]
        if not W_Team.empty and W_Team['power'].iloc[0] == 0:  # Ensure 'W_Team' isn't empty
            W_Team.loc[W_Team.index[0], 'power'] = 10
        elo = L_Team.loc[L_Team.index[0], 'power']
        LG = elo * 0.3
        if not W_Team.empty:
            w_teams.loc[w_teams['TeamID'] == game['WTeamID'], 'power'] = W_Team['power'].iloc[0] + LG
        if not L_Team.empty:
            w_teams.loc[w_teams['TeamID'] == game['LTeamID'], 'power'] = L_Team['power'].iloc[0] - LG

    if game['Season'] == 2025:
        # Elo calculation jumps by a full 10% here because this is the most recent data
        W_Team = w_teams[w_teams['TeamID'] == game['WTeamID']]
        L_Team = w_teams[w_teams['TeamID'] == game['LTeamID']]
        if not W_Team.empty and W_Team['power'].iloc[0] == 0:  # Ensure 'W_Team' isn't empty
            W_Team.loc[W_Team.index[0], 'power'] = 10
        elo = L_Team.loc[L_Team.index[0], 'power']
        LG = elo * 0.4
        if not W_Team.empty:
            w_teams.loc[w_teams['TeamID'] == game['WTeamID'], 'power'] = W_Team['power'].iloc[0] + LG
        if not L_Team.empty:
            w_teams.loc[w_teams['TeamID'] == game['LTeamID'], 'power'] = L_Team['power'].iloc[0] - LG



true_womens_teams = pd.read_csv('/kaggle/input/womenteams2/WTeams.csv')
w_teams = w_teams[w_teams['TeamID'].isin(true_womens_teams['TeamID'])]


for _, game in m_games.iterrows():
    if game['Season'] == 2020:
        W_Team = m_teams[m_teams['TeamID'] == game['WTeamID']]
        L_Team = m_teams[m_teams['TeamID'] == game['LTeamID']]
        if not W_Team.empty and W_Team['power'].iloc[0] == 0: # Ensure 'W_Team' isn't empty
            W_Team.loc[W_Team.index[0], 'power'] = 10
        elo = L_Team.loc[L_Team.index[0], 'power'] 
        LG = elo*0.1  
        if not W_Team.empty:
            m_teams.loc[m_teams['TeamID'] == game['WTeamID'], 'power'] = W_Team['power'].iloc[0] + LG
        if not L_Team.empty:
            m_teams.loc[m_teams['TeamID'] == game['LTeamID'], 'power'] = L_Team['power'].iloc[0] - LG
    
    if game['Season'] == 2021:
        W_Team = m_teams[m_teams['TeamID'] == game['WTeamID']]
        L_Team = m_teams[m_teams['TeamID'] == game['LTeamID']]
        if not W_Team.empty and W_Team['power'].iloc[0] == 0: # Ensure 'W_Team' isn't empty
            W_Team.loc[W_Team.index[0], 'power'] = 10
        elo = L_Team.loc[L_Team.index[0], 'power'] 
        LG = elo*0.15  
        if not W_Team.empty:
            m_teams.loc[m_teams['TeamID'] == game['WTeamID'], 'power'] = W_Team['power'].iloc[0] + LG
        if not L_Team.empty:
            m_teams.loc[m_teams['TeamID'] == game['LTeamID'], 'power'] = L_Team['power'].iloc[0] - LG
        
    if game['Season'] == 2022:
        W_Team = m_teams[m_teams['TeamID'] == game['WTeamID']]
        L_Team = m_teams[m_teams['TeamID'] == game['LTeamID']]
        if not W_Team.empty and W_Team['power'].iloc[0] == 0: # Ensure 'W_Team' isn't empty
            W_Team.loc[W_Team.index[0], 'power'] = 10
        elo = L_Team.loc[L_Team.index[0], 'power'] 
        LG = elo*0.2  
        if not W_Team.empty:
            m_teams.loc[m_teams['TeamID'] == game['WTeamID'], 'power'] = W_Team['power'].iloc[0] + LG
        if not L_Team.empty:
            m_teams.loc[m_teams['TeamID'] == game['LTeamID'], 'power'] = L_Team['power'].iloc[0] - LG
        
    if game['Season'] == 2023:
        W_Team = m_teams[m_teams['TeamID'] == game['WTeamID']]
        L_Team = m_teams[m_teams['TeamID'] == game['LTeamID']]
        if not W_Team.empty and W_Team['power'].iloc[0] == 0: # Ensure 'W_Team' isn't empty
            W_Team.loc[W_Team.index[0], 'power'] = 10
        elo = L_Team.loc[L_Team.index[0], 'power'] 
        LG = elo*0.25  
        if not W_Team.empty:
            m_teams.loc[m_teams['TeamID'] == game['WTeamID'], 'power'] = W_Team['power'].iloc[0] + LG
        if not L_Team.empty:
            m_teams.loc[m_teams['TeamID'] == game['LTeamID'], 'power'] = L_Team['power'].iloc[0] - LG

    if game['Season'] == 2024:
        W_Team = m_teams[m_teams['TeamID'] == game['WTeamID']]
        L_Team = m_teams[m_teams['TeamID'] == game['LTeamID']]
        if not W_Team.empty and W_Team['power'].iloc[0] == 0: # Ensure 'W_Team' isn't empty
            W_Team.loc[W_Team.index[0], 'power'] = 10
        elo = L_Team.loc[L_Team.index[0], 'power'] 
        LG = elo*0.3  
        if not W_Team.empty:
            m_teams.loc[m_teams['TeamID'] == game['WTeamID'], 'power'] = W_Team['power'].iloc[0] + LG
        if not L_Team.empty:
            m_teams.loc[m_teams['TeamID'] == game['LTeamID'], 'power'] = L_Team['power'].iloc[0] - LG
        
    if game['Season'] == 2025:
        #Elo calculation jumps by a full 10% percent here because this is the most recent data
        W_Team = m_teams[m_teams['TeamID'] == game['WTeamID']]
        L_Team = m_teams[m_teams['TeamID'] == game['LTeamID']]
        if not W_Team.empty and W_Team['power'].iloc[0] == 0: # Ensure 'W_Team' isn't empty
            W_Team.loc[W_Team.index[0], 'power'] = 10
        elo = L_Team.loc[L_Team.index[0], 'power'] 
        LG = elo*0.4  
        if not W_Team.empty:
            m_teams.loc[m_teams['TeamID'] == game['WTeamID'], 'power'] = W_Team['power'].iloc[0] + LG
        if not L_Team.empty:
            m_teams.loc[m_teams['TeamID'] == game['LTeamID'], 'power'] = L_Team['power'].iloc[0] - LG



m_teams = m_teams[m_teams['LastD1Season'] == 2025]


#shows the teams elos
for _, team1 in m_teams.iterrows():
    print(f"Team ({team1['TeamID']}): {team1['TeamName']} Power Ranking: {team1['power']}")
#print(f"Team: {m_teams['TeamName']} Power Ranking: {m_teams['power']}")



games = {'ID': [], 'Pred': []}

# Loop through each pair of teams to simulate the games
# Loop through each pair of teams to simulate the games
# Loop through each pair of teams to simulate the games
# Loop through each pair of teams to simulate the games
for i, team1 in m_teams.iterrows():
    for j, team2 in m_teams.iterrows():
        if i < j:  # Ensure each pair is processed only once (team1 vs team2 and not team2 vs team1)
            pred = 0
            if team1['power'] > team2['power']:
                pred = (team2['power'] / team1['power']) / 2
            elif team2['power'] > team1['power']:
                pred = (team1['power'] / team2['power']) / 2

            # Build the game ID (using team1 and team2 IDs in the correct order)
            game_id = f"2025_{team1['TeamID']}_{team2['TeamID']}"

            # Add the game ID to the games dictionary
            games['ID'].append(game_id)

            # Add the prediction to the games dictionary
            if team1['power'] > team2['power']:
                games['Pred'].append(1.00 - pred)  # If team1 wins, 1 - pred
            else:
                games['Pred'].append(pred)  # If team2 wins, pred value

            time.sleep(0.00001)  # Optional: to simulate a delay





# Loop through each pair of teams to simulate the games
# Loop through each pair of teams to simulate the games
# Loop through each pair of teams to simulate the games
# Loop through each pair of teams to simulate the games
for i, team1 in w_teams.iterrows():
    for j, team2 in w_teams.iterrows():
        if i < j:  # Ensure each pair is processed only once (team1 vs team2 and not team2 vs team1)
            pred = 0
            if team1['power'] > team2['power']:
                pred = (team2['power'] / team1['power']) / 2
            elif team2['power'] > team1['power']:
                pred = (team1['power'] / team2['power']) / 2

            # Build the game ID (using team1 and team2 IDs in the correct order)
            game_id = f"2025_{team1['TeamID']}_{team2['TeamID']}"

            # Add the game ID to the games dictionary
            games['ID'].append(game_id)

            # Add the prediction to the games dictionary
            if team1['power'] > team2['power']:
                games['Pred'].append(1.00 - pred)  # If team1 wins, 1 - pred
            else:
                games['Pred'].append(pred)  # If team2 wins, pred value

            time.sleep(0.00001)  # Optional: to simulate a delay




# Create a DataFrame from the 'games' dictionary
print(len(games['ID']))
df = pd.DataFrame(games)


df.to_csv('m_game_predictions.csv', index=False)
#(Year,IDs,Pred) SubmissionFormat



