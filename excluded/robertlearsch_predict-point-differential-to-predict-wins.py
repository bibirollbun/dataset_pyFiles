# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


## data from mens' seasons
df_detailed_mens = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv')
df_Madness_mens = pd.read_csv(
    '/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyDetailedResults.csv')
M_seeds = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')

## data from womens' seasons
df_detailed_womens = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonDetailedResults.csv')
df_Madness_womens = pd.read_csv(
    '/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyDetailedResults.csv')
W_seeds = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')



## Combine into a single dataframe
### Mark the gender of the teams
### No need to mark the seed data, the IDs are unique
df_detailed_mens['Male'] = 1
df_detailed_womens['Male'] = 0
df_Madness_womens['Male'] = 0
df_Madness_mens['Male']=1

df_Madness = pd.concat([df_Madness_mens, df_Madness_womens])
df_detailed = pd.concat([df_detailed_mens, df_detailed_womens])
df_seeds = pd.concat([M_seeds, W_seeds])

## Parse the seeds into region and seed number columns
df_seeds['Region'] = df_seeds['Seed'].str[0]
df_seeds= pd.get_dummies(df_seeds, columns=['Region'], prefix='Region')
df_seeds['Seed_Num'] = df_seeds['Seed'].str[1:3]
df_seeds['Seed_Num'] = df_seeds['Seed_Num'].astype(int)
df_seeds.head()


## These help parse the data, getting rid of the winning team's stats when the team
## I'm looking at is the losing team, and visca-versa
L_columns_drops= ['WTeamID', 'LTeamID', 'WLoc',
                   'NumOT','WFGM', 'WFGA', 'WFGM3', 'WFGA3', 
                 'WFTM', 'WFTA', 'WOR', 'WDR','WAst', 
                 'WTO', 'WStl', 'WBlk', 'WPF']
W_columns_drops = ['WTeamID', 'LTeamID', 'WLoc',
                   'NumOT','LFGM', 'LFGA', 'LFGM3', 'LFGA3', 
                  'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst',
                  'LTO', 'LStl', 'LBlk', 'LPF']

L_mapper = {'LScore':'Points_scored','WScore':"Points_allowed",
            'LFGM':'FGM', 'LFGA':"FGA", 'LFGM3':"FGM3", 'LFGA3':"FGA3", 
            'LFTM':"FTM", 'LFTA':"FTA", 'LOR':"OR", 'LDR':"DR", 'LAst':"Ast",
            'LTO':"TO", 'LStl':"Stl", 'LBlk':"Blk", 'LPF':"PF"}
W_mapper = {'WScore':'Points_scored','LScore':"Points_allowed",
            'WFGM':'FGM', 'WFGA':"FGA", 'WFGM3':"FGM3", 'WFGA3':"FGA3", 
            'WFTM':"FTM", 'WFTA':"FTA", 'WOR':"OR", 'WDR':"DR", 'WAst':"Ast",
            'WTO':"TO", 'WStl':"Stl", 'WBlk':"Blk", 'WPF':"PF"}

all_team_IDs = np.unique(df_detailed[['LTeamID','WTeamID']].values)


def box_score_seed_mean_std(season, season_window_width):
    season = season
    season_window_width = season_window_width
    df_season = df_detailed[(df_detailed.Season<=season)&(df_detailed.Season>(season-season_window_width))]
    df_madness_year = df_Madness[df_Madness.Season==season]
    detailed_stats = pd.DataFrame()
    num_TeamIDs=max(df_detailed.LTeamID.max(), df_detailed.WTeamID.max())
    all_team_IDs = np.unique(df_detailed[['LTeamID','WTeamID']].values)
    team_df_list = []
    y_list=[]
    for teamID in all_team_IDs[:]:
        team_season_list = []
        for season_count in df_season.Season.unique():
            df_single_season = df_season[df_season.Season==season_count]
            df_seed_single_season = df_seeds[df_seeds.Season==season_count]
            teamID_L = df_single_season[df_single_season.LTeamID==teamID]
            teamID_L = teamID_L.drop(columns=L_columns_drops)
            teamID_L.rename(columns=L_mapper, inplace=True)
            teamID_L_madness = df_madness_year[df_madness_year.LTeamID==teamID]
            teamID_L_madness = teamID_L_madness.drop(columns=L_columns_drops)
            teamID_L_madness.rename(columns=L_mapper, inplace=True)
            #repeat for their wins
            teamID_W = df_single_season[df_single_season.WTeamID==teamID]
            teamID_W = teamID_W.drop(columns=W_columns_drops)
            teamID_W.rename(columns=W_mapper, inplace=True)
            teamID_W_madness = df_madness_year[df_madness_year.WTeamID==teamID]
            teamID_W_madness = teamID_W_madness.drop(columns=W_columns_drops)
            teamID_W_madness.rename(columns=W_mapper, inplace=True)
            
            # Combine
            teamID_madness = pd.concat([teamID_W_madness, teamID_L_madness])
            ## Calculate score differential
            madness_score_diff_std = np.std(teamID_madness['Points_scored'] - teamID_madness['Points_allowed'])
            madness_score_diff = teamID_madness.mean()['Points_scored'] - teamID_madness.mean()['Points_allowed']
            teamID_df = pd.concat([teamID_W, teamID_L])
            teamID_df.Season = np.abs(teamID_df.Season - season)
            y1,y2 = madness_score_diff, madness_score_diff_std

            ## Add seed information 
            teamID_seed = df_seed_single_season[df_seed_single_season.TeamID==teamID][['Region_W','Region_X','Region_Y','Region_Z','Seed_Num']]
            teamID_seed.reset_index(drop=True, inplace=True)
            if teamID_seed.empty:
                teamID_seed.loc[0] = np.nan
            team_season_list.append(pd.concat([pd.DataFrame(teamID_df.mean()).transpose(),
                                    pd.DataFrame(teamID_df.std()).transpose(),teamID_seed],
                                              axis=1))
            if season_count == season:
                y_list.append([y1,y2])
        team_df_list.append(team_season_list)
    X_values = np.array(team_df_list)
    y_values = np.array(y_list)

    X = X_values.reshape(X_values.shape[0],-1)
    y = y_values
    return X, y


X, y = box_score_seed_mean_std(2023,1)
X


df_Madness_2023 = df_Madness[df_Madness.Season==2023]
df_Madness_2023 = df_Madness_2023[['WTeamID','LTeamID','WScore','LScore']]
df_Madness_2023.reset_index(drop=True, inplace=True)

## record winners, repeat 100 times, produce probability of winning 
df_Madness_2023_temp = df_Madness_2023.copy()
df_Madness_2023_temp['team1_wins'] = 0
df_Madness_2023_temp['team2_wins'] = 0
num_simulations=10
for n in range(num_simulations):
    for row in df_Madness_2023.index:
        WTeamID, LTeamID = df_Madness_2023.loc[row,'WTeamID'], df_Madness_2023.loc[row,'LTeamID']
        team1 = y[np.where(all_team_IDs == WTeamID)]
        team2 = y[np.where(all_team_IDs == LTeamID)]
        team1_points_diff, team2_points_diff = np.random.normal(team1[0][0],team1[0][1]), np.random.normal(team2[0][0],team2[0][1]) 
        if team1_points_diff - team2_points_diff >= 0:
            #team 1 winner
            df_Madness_2023_temp.loc[row,'team1_wins'] += 1
        if team1_points_diff - team2_points_diff < 0:
            #team 2 winner
            df_Madness_2023_temp.loc[row,'team2_wins'] += 1

df_Madness_2023_temp['probability'] = df_Madness_2023_temp['team1_wins']/(
    df_Madness_2023_temp['team1_wins']+df_Madness_2023_temp['team2_wins'])
sklearn.metrics.brier_score_loss(
    np.ones(len(df_Madness_2023_temp.probability)),np.array(df_Madness_2023_temp.probability))


def clean_data(X, y, X_nan=-1, y_nan_0=0, y_nan_1=0):
    # Filter X and y
    X_clean = np.nan_to_num(X.astype(float), nan=X_nan) #can't score negative points or give negative fouls
    #two teams that didn't play... coinflip... 
    y_clean = y.astype(float)
    y_clean[np.isnan(y_clean[:, 0]), 0] = y_nan_0
    y_clean[np.isnan(y_clean[:, 1]), 1] = y_nan_1
    return X_clean, y_clean


def train_rand_forest(X_clean, y_clean):
    model = RandomForestRegressor()
    trained_model = model.fit(X_clean,y_clean)
    score = model.score(X_clean,y_clean)
    return trained_model, score


season=2023
season_window_width = 1

# Most simple training loop:
X_train, y_train = box_score_seed_mean_std(season, season_window_width)
X_clean, y_clean = clean_data(X_train, y_train)
trained_model, model_score = train_rand_forest(X_clean, y_clean)
print('Predict '+str(season)+' season tournament data from regular season box scores. \nModel score:'+str(model_score))


X_test, _ = box_score_seed_mean_std(season+1, season_window_width)
X_test_clean, _ = clean_data(X_test, _)
y_pred = trained_model.predict(X_test_clean)


df_Madness_2024 = df_Madness[df_Madness.Season==2024]
df_Madness_2024 = df_Madness_2024[['WTeamID','LTeamID','WScore','LScore']]
df_Madness_2024.reset_index(drop=True, inplace=True)

#team_IDs = np.unique(df_Madness_2024[['LTeamID','WTeamID']].values)
## record winners, repeat 100 times, produce probability of winning 
df_Madness_2024_temp = df_Madness_2024.copy()
df_Madness_2024_temp['team1_wins'] = 0
df_Madness_2024_temp['team2_wins'] = 0
num_simulations=100
for n in range(num_simulations):
    for row in df_Madness_2024.index:
        WTeamID, LTeamID = df_Madness_2024.loc[row,'WTeamID'], df_Madness_2024.loc[row,'LTeamID']
        team1 = y_pred[np.where(all_team_IDs == WTeamID)]
        team2 = y_pred[np.where(all_team_IDs == LTeamID)]
        team1_points_diff, team2_points_diff = np.random.normal(team1[0][0],team1[0][1]), np.random.normal(team2[0][0],team2[0][1]) 
        if team1_points_diff - team2_points_diff >= 0:
            #team 1 winner
            df_Madness_2024_temp.loc[row,'team1_wins'] += 1
        if team1_points_diff - team2_points_diff < 0:
            #team 2 winner
            df_Madness_2024_temp.loc[row,'team2_wins'] += 1

df_Madness_2024_temp['probability'] = df_Madness_2024_temp['team1_wins']/(
    df_Madness_2024_temp['team1_wins']+df_Madness_2024_temp['team2_wins'])
bscore = sklearn.metrics.brier_score_loss(
    np.ones(len(df_Madness_2024_temp.probability)),np.array(df_Madness_2024_temp.probability))
print('Predicting 2024 Tournament with 2024 regular season box scores, Brier score: '+str(bscore))

## If I'm going to use this code a lot, it would be good to turn it into a function

