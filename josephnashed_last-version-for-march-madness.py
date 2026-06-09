# Libraries
import pandas as pd # I wanted to use fireducks instead, but it was giving me issues
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor

import glob


# Getting all files
path = "/kaggle/input/march-machine-learning-mania-2025/**"
data = {p.split('/')[-1].split('.')[0] : pd.read_csv(p, encoding='latin-1') for p in glob.glob(path)}


# Loading data
"""
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
"""

# To be used for storing data, and later to take the IDs
df = data["SampleSubmissionStage2"]


# Creating year, left team, and right team columns
"""
df['Year'] = [int(yr[0:4]) for yr in df['ID']]
df['LTeam'] = [int(L[5:9]) for L in df['ID']]
df['RTeam'] = [int(R[10:14]) for R in df['ID']]
"""


# Lots of feature selecting and engineering
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
sub = data['SampleSubmissionStage2']
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


# Selecting training data
X = games[col].fillna(-1)
sub_X = sub[col].fillna(-1)


# GridSearchCV

# Do some grid search here for the parameters
# After doing the grid search, save the best parameters and create the param_grid in this cell


# XGB parameters
param_grid = {
    'n_estimators': 5000,
    'learning_rate': 0.03,
    'max_depth': 6
}

# Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('xgb', XGBRegressor(**param_grid, device="gpu", random_state=42))
])

# Fitting pipeline
pipeline.fit(X, games['Pred'])

# Predicting games and submissions
pred = pipeline.predict(X).clip(0.001, 0.999)
sub_pred = pipeline.predict(sub_X).clip(0.001, 0.999)

# Cross validation (for the MSE)
cv_scores = cross_val_score(pipeline, X, games['Pred'], cv=5, scoring="neg_mean_squared_error")


# Results
print(f'Log Loss: {log_loss(games["Pred"], pred):.5f}')
print(f'Mean Absolute Error: {mean_absolute_error(games["Pred"], pred):.5f}')
print(f'Brier Score: {brier_score_loss(games["Pred"], pred):.5f}')
print(f'Cross-validated MSE: {-cv_scores.mean():.5f}')


# Creating submission dataframe
submission_df = pd.DataFrame({
    'ID': df['ID'],
    'Pred': sub_pred
})

# Shape and head/tail of submission
print(f"{submission_df.shape} \n")
print(f"{submission_df.head().to_string(index=False)} \n")
print(submission_df.tail().to_string(index=False))

# Saving to csv
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission file saved! Good luck!!! :)")

