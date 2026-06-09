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


import numpy as np
import pandas as pd 
from scipy.stats import linregress
from tqdm import tqdm
import matplotlib.pyplot as plt


def calculate_elo(
    teams, data, initial_rating=2000, k=140, alpha=None, weights=False, lowerlim=float("-inf")
    ):
    '''
    Calculate Elo ratings for each team based on match data.

    Parameters:
    - teams (array-like): Containing Team-IDs.
    - data (pd.DataFrame): DataFrame with all matches in chronological order.
    - initial_rating (float): Initial rating of an unranked team (default: 2000). 
    - k (float): K-factor, determining the impact of each match on team ratings (default: 140).
    - alpha (float or None): Tuning parameter for the multiplier for the margin of victory. No multiplier if None.

    Returns: 
    - list: Historical ratings of the winning team (WTeam).
    - list: Historical ratings of the losing team (LTeam).
    - list: Brier score for each match (due to symmetry only for 1 team)
    '''
    
    # Dictionary to keep track of current ratings for each team
    team_dict = {}
    for team in teams:
        team_dict[team] = initial_rating
        
    # Lists to store ratings for each team in each game
    r1, r2 = [], []
    loss = []
    margin_of_victory = 1
    weight = 1

    # Iterate through the game data
    for wteam, lteam, ws, ls, w  in tqdm(zip(data.WTeamID, data.LTeamID, data.WScore, data.LScore, data.weight), total=len(data)):

        # Calculate expected outcomes based on Elo ratings
        rateW = 1 / (1 + 10 ** ((team_dict[lteam] - team_dict[wteam]) / initial_rating))
        rateL = 1 / (1 + 10 ** ((team_dict[wteam] - team_dict[lteam]) / initial_rating))
        
        if alpha:
                margin_of_victory = (ws - ls)/alpha
        if isinstance(weights, (list, np.ndarray, pd.Series)):
            weight = w

        # Update ratings for winning and losing teams
        team_dict[wteam] += w * k * margin_of_victory * (1 - rateW)
        team_dict[lteam] += w * k * margin_of_victory * (0 - rateL)

        # Ensure that ratings do not go below lower limit
        if team_dict[lteam] < lowerlim:
            team_dict[lteam] = lowerlim
            
        # Append current ratings for teams to lists
        r1.append(team_dict[wteam])
        r2.append(team_dict[lteam])
        loss.append((1-rateW)**2)
        
    return r1, r2, loss

def create_elo_data(teams, data, initial_rating=2000, k=140, alpha=None, weights=None, lowerlim=float("-inf")):
    '''
    Create a DataFrame with summary statistics of Elo ratings for teams based on historical match data.

    Parameters:
    - teams (array-like): Containing Team-IDs.
    - data (pd.DataFrame): DataFrame with all matches in chronological order.
    - initial_rating (float): Initial rating of an unranked team (default: 2000).
    - k (float): K-factor, determining the impact of each match on team ratings (default: 140).
    - weights (array-like): Containing weights for each match.

    Returns: 
    - DataFrame: Summary statistics of Elo ratings for teams throughout a season.
    '''
    
    if isinstance(weights, (list, np.ndarray, pd.Series)):
        data['weight'] = weights
    else:
        data['weight'] = 1
    
    r1, r2, loss = calculate_elo(teams, data, initial_rating, k, alpha, weights, lowerlim)
    # Calculate loss only on tourney results
    loss = np.mean(np.array(loss)[data.tourney == 1])
    print(f"=== Brier Score: {loss:.5f} (Only  Tournaments) ===")
    
    # Concatenate arrays vertically
    seasons = np.concatenate([data.Season, data.Season])
    days = np.concatenate([data.DayNum, data.DayNum])
    teams = np.concatenate([data.WTeamID, data.LTeamID])
    tourney = np.concatenate([data.tourney, data.tourney])
    ratings = np.concatenate([r1, r2])
    # Create a DataFrame
    rating_df = pd.DataFrame({
        'Season': seasons,
        'DayNum': days,
        'TeamID': teams,
        'Rating': ratings,
        'Tourney': tourney
    })

    # Sort DataFrame and remove tournament data
    rating_df.sort_values(['TeamID', 'Season', 'DayNum'], inplace=True)
    rating_df = rating_df[rating_df['Tourney'] == 0]
    grouped = rating_df.groupby(['TeamID', 'Season'])
    results = grouped['Rating'].agg(['mean', 'median', 'std', 'min', 'max', 'last'])
    results.columns = ['Rating_Mean', 'Rating_Median', 'Rating_Std', 'Rating_Min', 'Rating_Max', 'Rating_Last']
    results['Rating_Trend'] = grouped.apply(lambda x: linregress(range(len(x)), x['Rating']).slope, include_groups=False)
    results.reset_index(inplace=True)
    
    return results, loss


# Load and Process Data Men's Tourney
regular_m = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')
tourney_m = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
teams_m = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')

regular_m['tourney'] = 0
tourney_m['tourney'] = 1
regular_m['weight'] = 1
tourney_m['weight'] = 0.7

data_m = pd.concat([regular_m, tourney_m])
data_m.sort_values(['Season', 'DayNum'], inplace=True)
data_m.reset_index(inplace=True, drop=True)

elo_df_men, _ = create_elo_data(teams_m.TeamID, data_m, initial_rating=1200, k=125, alpha=None, weights=data_m['weight'])
elo_df_men.tail(10)


# Load and Process Data Women's Tourney
regular_w = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv')
tourney_w = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv')
teams_w = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WTeams.csv')

regular_w['tourney'] = 0
tourney_w['tourney'] = 1
regular_w['weight'] = 0.95
tourney_w['weight'] = 1

data_w = pd.concat([regular_w, tourney_w])
data_w.sort_values(['Season', 'DayNum'], inplace=True)
data_w.reset_index(inplace=True, drop=True)

elo_df_women, _ = create_elo_data(teams_w.TeamID, data_w, initial_rating=1200, k=190, alpha=None, weights=data_w['weight'])
elo_df_women.tail(10)


elo_df_men.to_csv('mens_elo_rating.csv')
elo_df_women.to_csv('womens_elo_rating.csv')


tmp_df_men = pd.merge(elo_df_men, teams_m, on='TeamID', how='left')
tmp_df_men = tmp_df_men[tmp_df_men['Season'] == 2025]
top_men_teams = tmp_df_men.sort_values('Rating_Last', ascending=False)[:20][['TeamName', 'Rating_Last', 'Rating_Trend']]
top_men_teams = top_men_teams.reindex(index=top_men_teams.index[::-1])

# Women's Teams
tmp_df_women = pd.merge(elo_df_women, teams_w, on='TeamID', how='left')
tmp_df_women = tmp_df_women[tmp_df_women['Season'] == 2025]
top_women_teams = tmp_df_women.sort_values('Rating_Last', ascending=False)[:20][['TeamName', 'Rating_Last', 'Rating_Trend']]
top_women_teams = top_women_teams.reindex(index=top_women_teams.index[::-1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

# Men's Teams
ax1.barh(top_men_teams['TeamName'], top_men_teams['Rating_Last'], color='skyblue', label='Rating_Last')
ax1.set_title("Top Men's Teams - 2025")
ax1.set_xlabel('Last Rating')
ax1.set_ylabel('TeamName')
ax1.legend()

# Women's Teams
ax2.barh(top_women_teams['TeamName'], top_women_teams['Rating_Last'], color='#3F51B5', label='Rating_Last')
ax2.set_title("Top Women's Teams - 2025")
ax2.set_xlabel('Last Rating')
ax2.set_ylabel('TeamName')
ax2.legend()

plt.tight_layout()
plt.show()


submission = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv")

# Split the ID into Season, T1_TeamID, and T2_TeamID
sub = submission.ID.str.split('_', expand=True).astype(int)
sub.columns = ["Season", "T1_TeamID", "T2_TeamID"]

# Turn elo dfs into dict
elo_dict = pd.concat([
    elo_df_women[elo_df_women.Season == 2025],
    elo_df_men[elo_df_men.Season == 2025]
]).set_index("TeamID")["Rating_Last"].to_dict()

# Calculate probabilities
submission.Pred = 1 / (1 + 10**((sub.T2_TeamID.map(elo_dict) - sub.T1_TeamID.map(elo_dict))/1200))
submission.to_csv("submission.csv", index=False)
submission.head()




