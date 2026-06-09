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


# Libraries
import pandas as pd
import numpy as np

from sklearn.metrics import brier_score_loss

from xgboost import XGBClassifier

import glob


# Getting all files
path = "/kaggle/input/march-machine-learning-mania-2025/**"
data = {p.split('/')[-1].split('.')[0] : pd.read_csv(p, encoding='latin-1') for p in glob.glob(path)}


# Loading data

MTeams = data["MTeams"]
WTeams = data["WTeams"]

MNCAATourneySeeds = data["MNCAATourneySeeds"]
WNCAATourneySeeds = data["WNCAATourneySeeds"]

MNCAATourneySlots = data["MNCAATourneySlots"]
WNCAATourneySlots = data["WNCAATourneySlots"]

MRegularSeasonDetailedResults = data["MRegularSeasonDetailedResults"]
WRegularSeasonDetailedResults = data["WRegularSeasonDetailedResults"]

MNCAATourneyDetailedResults = data["MNCAATourneyDetailedResults"]
WNCAATourneyDetailedResults = data["WNCAATourneyDetailedResults"]

MSecondaryTourneyTeams = data["MSecondaryTourneyTeams"]
WSecondaryTourneyTeams = data["WSecondaryTourneyTeams"]

MSecondaryTourneyCompactResults = data["MSecondaryTourneyCompactResults"]
WSecondaryTourneyCompactResults = data["WSecondaryTourneyCompactResults"]

Cities = data["Cities"]

MGameCities = data["MGameCities"]
WGameCities = data["WGameCities"]





# To be used for storing data, and later to take the IDs
df = data["SampleSubmissionStage1"]


# Creating year, left team, and right team columns

df['Year'] = [int(yr[0:4]) for yr in df['ID']]
df['LTeam'] = [int(L[5:9]) for L in df['ID']]
df['RTeam'] = [int(R[10:14]) for R in df['ID']]



# Calculate ELO differences

def calculate_elo_ratings(regular_season_data, tourney_data, k_factor=20, initial_elo=1500, home_advantage=100):
    team_elo = {}
    
    # Process both regular season and tournament games
    for games_df in [regular_season_data, tourney_data]:
        for index, game in games_df.iterrows():
            season = game['Season']
            w_team = game['WTeamID']
            l_team = game['LTeamID']
            w_loc = game['WLoc']
            
            # Initialize season if needed
            if season not in team_elo:
                team_elo[season] = {}
            
            # Initialize teams if needed
            for team in [w_team, l_team]:
                if team not in team_elo[season]:
                    team_elo[season][team] = initial_elo
            
            # Get base Elo ratings
            w_elo = team_elo[season][w_team]
            l_elo = team_elo[season][l_team]
            
            # Apply home court advantage
            if w_loc == "H":
                w_elo += home_advantage
            elif w_loc == "A":
                l_elo += home_advantage
            
            # Calculate expected probabilities
            expected_w = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
            
            # Update ratings
            elo_change = k_factor * (1 - expected_w)
            team_elo[season][w_team] = team_elo[season][w_team] + elo_change
            team_elo[season][l_team] = team_elo[season][l_team] - elo_change
            
    return team_elo

def calculate_elo_differences(df, mens_regular, womens_regular, mens_tourney, womens_tourney):
    # Calculate separate Elo ratings for men's and women's teams
    mens_elo = calculate_elo_ratings(mens_regular, mens_tourney)
    womens_elo = calculate_elo_ratings(womens_regular, womens_tourney)
    
    def get_elo_diff(row):
        season = row['Year']
        left_team = row['LTeam']
        right_team = row['RTeam']
        
        # Determine if this is a men's or women's matchup
        # Assuming the first half of df is men's teams and second half is women's
        is_mens = df.index.get_loc(row.name) < len(df) // 2
        
        # Get the appropriate Elo ratings
        elo_ratings = mens_elo if is_mens else womens_elo
        
        # Get Elo ratings for both teams
        left_elo = elo_ratings[season].get(left_team, 1500)
        right_elo = elo_ratings[season].get(right_team, 1500)
        
        return left_elo - right_elo
    
    # Calculate regular season Elo differences
    df['RegELODiff'] = df.apply(get_elo_diff, axis=1)
    
    # Recalculate including tournament games and store as TourneyELODiff
    mens_elo = calculate_elo_ratings(mens_regular, mens_tourney)
    womens_elo = calculate_elo_ratings(womens_regular, womens_tourney)
    df['TourneyELODiff'] = df.apply(get_elo_diff, axis=1)
    
    return df

df = calculate_elo_differences(
    df,
    MRegularSeasonDetailedResults,
    WRegularSeasonDetailedResults,
    MNCAATourneyDetailedResults,
    WNCAATourneyDetailedResults
)

print(df.head())
print(df.tail())



teams = pd.concat([data['MTeams'], data['WTeams']])
teams_spelling = pd.concat([data['MTeamSpellings'], data['WTeamSpellings']])
teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()
teams_spelling.columns = ['TeamID', 'TeamNameCount']
teams = pd.merge(teams, teams_spelling, how='left', on=['TeamID'])
del teams_spelling

season_cresults = pd.concat([data['MRegularSeasonCompactResults'], data['WRegularSeasonCompactResults']])
season_dresults = pd.concat([data['MRegularSeasonDetailedResults'], data['WRegularSeasonDetailedResults']])
tourney_cresults = pd.concat([data['MNCAATourneyCompactResults'], data['WNCAATourneyCompactResults']])
tourney_dresults = pd.concat([data['MNCAATourneyDetailedResults'], data['WNCAATourneyDetailedResults']])
slots = pd.concat([data['MNCAATourneySlots'], data['WNCAATourneySlots']])
seeds = pd.concat([data['MNCAATourneySeeds'], data['WNCAATourneySeeds']])
gcities = pd.concat([data['MGameCities'], data['WGameCities']])
seasons = pd.concat([data['MSeasons'], data['WSeasons']])

seeds = {'_'.join(map(str,[int(k1),k2])):int(v[1:3]) for k1, v, k2 in seeds[['Season', 'Seed', 'TeamID']].values}
cities = data['Cities']
sub = data['SampleSubmissionStage1']
del data

season_cresults['ST'] = 'S'
season_dresults['ST'] = 'S'
tourney_cresults['ST'] = 'T'
tourney_dresults['ST'] = 'T'
games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
games.reset_index(drop=True, inplace=True)
games['WLoc'] = games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})

games['ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']]+sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games['IDTeams'] = games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[0], axis=1)
games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[1], axis=1)
games['IDTeam1'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
games['IDTeam2'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)

games['Team1Seed'] = games['IDTeam1'].map(seeds).fillna(0)
games['Team2Seed'] = games['IDTeam2'].map(seeds).fillna(0)

games['ScoreDiff'] = games['WScore'] - games['LScore']
games['Pred'] = games.apply(lambda r: 1. if sorted([r['WTeamID'],r['LTeamID']])[0]==r['WTeamID'] else 0., axis=1)
games['ScoreDiffNorm'] = games.apply(lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0. else r['ScoreDiff'], axis=1)
games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed']
games = games.fillna(-1)

c_score_col = ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl',
 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl',
 'LBlk', 'LPF']
c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
gb = games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
gb.columns = [''.join(c) + '_c_score' for c in gb.columns]

games = games[games['ST']=='T']

sub['WLoc'] = 3
sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
sub['Season'] = sub['Season'].astype(int)
sub['Team1'] = sub['ID'].map(lambda x: x.split('_')[1])
sub['Team2'] = sub['ID'].map(lambda x: x.split('_')[2])
sub['IDTeams'] = sub.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)
sub['IDTeam1'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
sub['IDTeam2'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
sub['Team1Seed'] = sub['IDTeam1'].map(seeds).fillna(0)
sub['Team2Seed'] = sub['IDTeam2'].map(seeds).fillna(0)
sub['SeedDiff'] = sub['Team1Seed'] - sub['Team2Seed']
sub = sub.fillna(-1)

games = pd.merge(games, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
sub = pd.merge(sub, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')

col = [c for c in games.columns if c not in ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2',
                                             'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 'ScoreDiffNorm',
                                             'WLoc'] + c_score_col]


from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_absolute_error, brier_score_loss
from xgboost import XGBRegressor

X = games[col].fillna(-1)
sub_X = sub[col].fillna(-1)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('xgb', XGBRegressor(n_estimators=5000, learning_rate=0.03, max_depth=6, random_state=42))
])

pipeline.fit(X, games['Pred'])

pred = pipeline.predict(X).clip(0.001, 0.999)
sub_pred = pipeline.predict(sub_X).clip(0.001, 0.999)

cv_scores = cross_val_score(pipeline, X, games['Pred'], cv=5, scoring="neg_mean_squared_error")


# Results
print(f'Log Loss: {log_loss(games["Pred"], pred):.4f}')
print(f'Mean Absolute Error: {mean_absolute_error(games["Pred"], pred):.4f}')
print(f'Brier Score: {brier_score_loss(games["Pred"], pred):.4f}')
print(f'Cross-validated MSE: {-cv_scores.mean():.4f}')

