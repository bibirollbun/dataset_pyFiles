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


#All Neede Libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from itertools import combinations


import pandas as pd
import os

def load_data():
    data_files = {
         'm_teams': '/kaggle/input/march-machine-learning-mania-2025/MTeams.csv',
        'w_teams': '/kaggle/input/march-machine-learning-mania-2025/WTeams.csv',
        'm_seasons': '/kaggle/input/march-machine-learning-mania-2025/MSeasons.csv',
        'w_seasons': '/kaggle/input/march-machine-learning-mania-2025/WSeasons.csv',
        'm_tourney_seeds': '/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv',
        'w_tourney_seeds': '/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv',
        'm_regular_results': '/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv',
        'w_regular_results': '/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv',
        'm_tourney_results': '/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv',
        'w_tourney_results': '/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv'}
    
    datasets = {key: pd.read_csv(path) for key, path in data_files.items()}
    for key, df in datasets.items():
        df['Gender'] = 'M' if key.startswith('m_') else 'W'

    teams = pd.concat([datasets['m_teams'], datasets['w_teams']], ignore_index=True)
    seasons = pd.concat([datasets['m_seasons'], datasets['w_seasons']], ignore_index=True)
    tourney_seeds = pd.concat([datasets['m_tourney_seeds'], datasets['w_tourney_seeds']], ignore_index=True)
    regular_results = pd.concat([datasets['m_regular_results'], datasets['w_regular_results']], ignore_index=True)
    tourney_results = pd.concat([datasets['m_tourney_results'], datasets['w_tourney_results']], ignore_index=True)

    return teams, seasons, tourney_seeds, regular_results, tourney_results


def team_features(regular_results, tourney_seeds):
    #wins dataframe
    wins_df = regular_results[['Season', 'WTeamID', 'Gender', 'WScore', 'LScore']].copy()
    wins_df['TeamID'] = wins_df['WTeamID']
    wins_df['Wins'] = 1
    wins_df['Losses'] = 0
    wins_df['PointsScored'] = wins_df['WScore']
    wins_df['PointsAllowed'] = wins_df['LScore']
    
    #losses datafram
    losses_df = regular_results[['Season', 'LTeamID', 'Gender', 'LScore', 'WScore']].copy()
    losses_df['TeamID'] = losses_df['LTeamID']
    losses_df['Wins'] = 0
    losses_df['Losses'] = 1
    losses_df['PointsScored'] = losses_df['LScore']
    losses_df['PointsAllowed'] = losses_df['WScore']
    
    # Combine wins and losses
    all_games = pd.concat([wins_df, losses_df], ignore_index=True)
    
    # Aggregate statistics by team, season, and gender
    team_stats = all_games.groupby(['Season', 'TeamID', 'Gender']).agg(
        Wins=('Wins', 'sum'),
        Losses=('Losses', 'sum'),
        AvgPointsScored=('PointsScored', 'mean'),
        AvgPointsAllowed=('PointsAllowed', 'mean')
    ).reset_index()
    
    team_stats['WinRate'] = team_stats['Wins'] / (team_stats['Wins'] + team_stats['Losses'])

    team_stats = team_stats.merge(tourney_seeds[['Season', 'TeamID', 'Seed', 'Gender']],
                                  on=['Season', 'TeamID', 'Gender'], how='left')
    team_stats['Seed'] = team_stats['Seed'].str.extract(r'(\d+)').astype(float).fillna(16)

    return team_stats


def prepare_training_data(tourney_results, team_stats):
    #matchups with team IDs
    matchups = tourney_results.copy()
    matchups['Team1'] = matchups[['WTeamID', 'LTeamID']].min(axis=1)
    matchups['Team2'] = matchups[['WTeamID', 'LTeamID']].max(axis=1)
    matchups['Target'] = (matchups['WTeamID'] == matchups['Team1']).astype(int)

    #team stats with suffix
    team_stats_team1 = team_stats.rename(columns=lambda x: x + '_team1' if x not in ['Season', 'TeamID', 'Gender'] else x)
    team_stats_team2 = team_stats.rename(columns=lambda x: x + '_team2' if x not in ['Season', 'TeamID', 'Gender'] else x)

    #merging stats for team1
    matchups = matchups.merge(team_stats_team1, left_on=['Season', 'Team1', 'Gender'],
                              right_on=['Season', 'TeamID', 'Gender'], how='left')

    #for team2
    matchups = matchups.merge(team_stats_team2, left_on=['Season', 'Team2', 'Gender'],
                              right_on=['Season', 'TeamID', 'Gender'], how='left')

    #feature set
    features = ['WinRate_team1', 'AvgPointsScored_team1', 'AvgPointsAllowed_team1', 'Seed_team1',
                'WinRate_team2', 'AvgPointsScored_team2', 'AvgPointsAllowed_team2', 'Seed_team2']
    X = matchups[features]
    y = matchups['Target']

    return X, y, matchups


def train_model(X, y):
    #spleting the data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    #train of the model
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    #evaluation
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    brier = brier_score_loss(y_val, y_pred_proba)
    print(f'Validation Brier Score: {brier:.4f}')

    #full dataset
    model.fit(X, y)
    return model


def generate_predictions(model, team_stats, season=2025):
    """Create predictions for all possible 2025 tournament matchups."""
    #latest season's data
    latest_stats = team_stats[team_stats['Season'] == team_stats['Season'].max()]

    #the unique teams ID by gender
    men_teams = latest_stats[latest_stats['Gender'] == 'M']['TeamID'].unique()
    women_teams = latest_stats[latest_stats['Gender'] == 'W']['TeamID'].unique()

    #all of the possible matchups
    men_matchups = pd.DataFrame(combinations(men_teams, 2), columns=['Team1', 'Team2'])
    women_matchups = pd.DataFrame(combinations(women_teams, 2), columns=['Team1', 'Team2'])
    men_matchups['Gender'] = 'M'
    women_matchups['Gender'] = 'W'
    matchups = pd.concat([men_matchups, women_matchups], ignore_index=True)
    matchups['Season'] = season

    #team stats with suffixes
    team_stats_team1 = team_stats.rename(columns=lambda x: x + '_team1' if x not in ['Season', 'TeamID', 'Gender'] else x)
    team_stats_team2 = team_stats.rename(columns=lambda x: x + '_team2' if x not in ['Season', 'TeamID', 'Gender'] else x)

    #merging stats for team1
    matchups = matchups.merge(team_stats_team1, left_on=['Season', 'Team1', 'Gender'],
                              right_on=['Season', 'TeamID', 'Gender'], how='left')

    #for team2
    matchups = matchups.merge(team_stats_team2, left_on=['Season', 'Team2', 'Gender'],
                              right_on=['Season', 'TeamID', 'Gender'], how='left')
    features = ['WinRate_team1', 'AvgPointsScored_team1', 'AvgPointsAllowed_team1', 'Seed_team1',
                'WinRate_team2', 'AvgPointsScored_team2', 'AvgPointsAllowed_team2', 'Seed_team2']
    X_pred = matchups[features]

    matchups['Pred'] = model.predict_proba(X_pred)[:, 1]

    #THE SUBMISSION FORMAT
    matchups['ID'] = str(season) + '_' + matchups['Team1'].astype(str) + '_' + matchups['Team2'].astype(str)
    submission = matchups[['ID', 'Pred']]

    return submission


def main():
    #complete prediciton pipline
    #loading data
    teams, seasons, tourney_seeds, regular_results, tourney_results = load_data()

    #then generating features
    team_stats = team_features(regular_results, tourney_seeds)

    #preparing the training data
    X, y, _ = prepare_training_data(tourney_results, team_stats)

    #model train
    model = train_model(X, y)

    #generate predictions and save
    submission = generate_predictions(model, team_stats)
    submission.to_csv('submission.csv', index=False)
    print("Submission file 'submission.csv' created successfully!")

if __name__ == "__main__":
    main()




