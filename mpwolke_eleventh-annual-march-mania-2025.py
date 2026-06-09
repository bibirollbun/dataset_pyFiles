# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_colwidth', None)

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

sns.set()
%matplotlib inline
warnings.filterwarnings(action='ignore')

import matplotlib as mpl
import matplotlib.cm as cmap
import matplotlib.colors as mpl_colors


#By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

palette = [
     '#FDA736', '#17449B', '#008000', '#023e7d', '#02c39a', '#045093', '#086788', '#0c0f0a', '#1d4a60', '#2E765E', '#2e294e', '#2ec4b6',
    '#50276E', '#540b0e', '#545454', '#568f8b', '#5DD9FB', '#5a189a', '#7fc8f8', '#B6E5D8', '#BE0C3D', '#DA1818',
    '#b4d2b1', '#c1121f', '#cd7e59', '#d15252', '#ddb247', '#ff9f1c']

palette_rgb = [hex_to_rgb(x) for x in palette]
cmap = mpl_colors.ListedColormap(palette_rgb)
colors = cmap.colors
bg_color = '#EFEAE0'

custom_params = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    'grid.alpha':0.3,
    'figure.figsize': (16, 6),
    'axes.titlesize': 'Large',
    'axes.labelsize': 'Large',
      'figure.facecolor': bg_color,
    'axes.facecolor': bg_color
}

sns.set_theme(
    style='whitegrid',
    palette=sns.color_palette(palette),
    rc=custom_params
)

warnings.simplefilter("ignore", UserWarning)

# disable pandas rows and columns limit
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

DIR = '/kaggle/input'


mens_files = []
all_files = []
for file in os.listdir(f'{DIR}/march-machine-learning-mania-2025/'):
    all_files.append(file)
    if file.startswith('M'):
        mens_files.append(file)
print(f"Total files: {len(all_files)}, Mens files: {len(mens_files)}")
print(sorted(mens_files))
print(sorted(all_files))


MTeams = pd.read_csv(f'{DIR}/march-machine-learning-mania-2025/MTeams.csv')
MSeasons = pd.read_csv(f'{DIR}/march-machine-learning-mania-2025/MSeasons.csv')

display(MTeams.head())
display(MSeasons.head())
display(MTeams.describe())
display(MSeasons.describe())
display(MTeams.info())
display(MSeasons.info())


print(MSeasons['Season'].unique())


MTeams['SeasonPlayed'] = MTeams['LastD1Season'] - MTeams['FirstD1Season']


##By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023

fig, ax = plt.subplots(2, 1, figsize=(24, 12))
                       
sns.barplot(
    x=MTeams.sort_values(by='SeasonPlayed', ascending=False).head(50)['TeamName'],
    y=MTeams.sort_values(by='SeasonPlayed', ascending=False).head(50)['SeasonPlayed'],
    ax=ax[0]
)
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
ax[0].set_xlabel('Team Name', fontsize=12)
ax[0].set_ylabel('Number of Seasons Played', fontsize=12)
for container in ax[0].containers:
    ax[0].bar_label(container, padding=3, fontsize=12, color='black', label_type='edge', weight='bold')

ax[0].set_title('Teams that have played the most number of seasons', fontsize=16)
    
sns.barplot(
    x=MTeams.sort_values(by='SeasonPlayed', ascending=True).head(50)['TeamName'],
    y=MTeams.sort_values(by='SeasonPlayed', ascending=True).head(50)['SeasonPlayed'],
    ax=ax[1]
)
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
ax[1].set_xlabel('Team Name', fontsize=12)
ax[1].set_ylabel('Number of Seasons Played', fontsize=12)
for container in ax[1].containers:
    ax[1].bar_label(container, padding=3, fontsize=12, color='black', label_type='edge', weight='bold')
ax[1].set_title('Teams that have played the least number of seasons', fontsize=16)

plt.tight_layout()
plt.show()


newest_teams = MTeams[MTeams['FirstD1Season'] >= 2010].sort_values(by='FirstD1Season', ascending=True)
print(newest_teams.shape)
display(newest_teams)


#By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023

fig = plt.figure(figsize=(12, 6))
                       
ax = sns.barplot(
    x=newest_teams.sort_values(by='SeasonPlayed', ascending=False).head(50)['TeamName'],
    y=newest_teams.sort_values(by='SeasonPlayed', ascending=False).head(50)['SeasonPlayed'],
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel('Team Name', fontsize=12)
ax.set_ylabel('Number of Seasons Played', fontsize=12)
for container in ax.containers:
    ax.bar_label(container, padding=3, fontsize=12, color='black', label_type='edge', weight='bold')

ax.set_title('Teams that started playing after 2010', fontsize=16)
plt.tight_layout()
plt.show()


for file in mens_files:
    if file.startswith('MNCAA'):
        print(file)


#By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023


MNCAATourneySeeds = pd.read_csv(f'{DIR}/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
MNCAATourneySlots = pd.read_csv(f'{DIR}/march-machine-learning-mania-2025/MNCAATourneySlots.csv')
MNCAATourneySeedRoundSlots = pd.read_csv(f'{DIR}/march-machine-learning-mania-2025/MNCAATourneySeedRoundSlots.csv')
MNCAATourneyCompactResults = pd.read_csv(f'{DIR}/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
MNCAATourneyDetailedResults = pd.read_csv(f'{DIR}/march-machine-learning-mania-2025/MNCAATourneyDetailedResults.csv')
display(MNCAATourneySeeds.head())
display(MNCAATourneySlots.head())
display(MNCAATourneySeedRoundSlots.head())
display(MNCAATourneyCompactResults.head())
display(MNCAATourneyDetailedResults.head())


#By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023


display(MNCAATourneySeeds.info())
display(MNCAATourneySlots.info())
display(MNCAATourneySeedRoundSlots.info())
display(MNCAATourneyCompactResults.info())
display(MNCAATourneyDetailedResults.info())


def basic_info(df):
    display(df.head())
    display(df.describe())
    display(df.info())


basic_info(MNCAATourneySeeds)


#By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023


fig = plt.figure(figsize=(20, 8))
ax = sns.countplot(x='Season', data=MNCAATourneySeeds)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', fontsize=12, color='black', padding=3, weight='bold')
plt.title("Number of matches per season", fontsize=16)
plt.show()


MNCAATourneySeeds[(MNCAATourneySeeds['Season'] >= 2017) & (MNCAATourneySeeds['TeamID'].isin(newest_teams['TeamID'].values))]


basic_info(MNCAATourneySlots)


MNCAATourneySlots[(MNCAATourneySlots['Season'] == 2017) & (MNCAATourneySlots['StrongSeed'] == 'W01')]


basic_info(MNCAATourneySeedRoundSlots)


basic_info(MNCAATourneyCompactResults)


#By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023


fig = plt.figure(figsize=(20, 8))
avg_score = MNCAATourneyCompactResults.groupby('Season')[['WScore', 'LScore']].mean()
ax = sns.lineplot(data=avg_score)
plt.title('Mean Winning and Lossing score over seasons', fontsize=16)
plt.show()


#By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023


max_scores = MNCAATourneyCompactResults.groupby('Season')[['WScore', 'LScore']].max().reset_index().rename(columns={'WScore': 'MaxWScore', 'LScore': 'MaxLScore'})
min_scores = MNCAATourneyCompactResults.groupby('Season')[['WScore', 'LScore']].min().reset_index().rename(columns={'WScore': 'MinWScore', 'LScore': 'MinLScore'})
max_min_scores = pd.merge(max_scores, min_scores, on='Season')
max_min_scores.set_index('Season', inplace=True)
display(max_min_scores.head())

fig = plt.figure(figsize=(20, 8))
ax = sns.lineplot(data=max_min_scores)
plt.title('Maximum and Minimum winning and lossing scores across seasons', fontsize=16)
plt.show()


matches_won_per_team_per_season = MNCAATourneyCompactResults.groupby(['Season', 'WTeamID'])['WScore'].count().reset_index().rename(columns={'WScore': 'WinCount'})
matches_lost_per_team_per_season = MNCAATourneyCompactResults.groupby(['Season', 'LTeamID'])['LScore'].count().reset_index().rename(columns={'LScore': 'LossCount'})
matches_per_team_per_season = pd.merge(matches_won_per_team_per_season, matches_lost_per_team_per_season, left_on=['Season', 'WTeamID'], right_on=['Season', 'LTeamID'])
matches_per_team_per_season['TotalMatches'] = matches_per_team_per_season['WinCount'] + matches_per_team_per_season['LossCount']
matches_per_team_per_season.rename(columns={'WTeamID': 'TeamID'}, inplace=True)
matches_per_team_per_season.drop(columns=['LTeamID'], inplace=True)
matches_per_team_per_season.set_index(['Season'], inplace=True)
matches_per_team_per_season['WinPercentage'] = ((matches_per_team_per_season['WinCount'] / matches_per_team_per_season['TotalMatches']) * 100).round(2)
display(matches_per_team_per_season.head())


display(matches_per_team_per_season[matches_per_team_per_season['TeamID'] == 1104])


#By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023


fig = plt.figure(figsize=(20, 8))
ax = sns.lineplot(data=matches_per_team_per_season['WinPercentage'])
plt.title('Win Percentage trend over all seasons', fontsize=16)
plt.show()


(
    matches_per_team_per_season['TotalMatches'].max(),
    matches_per_team_per_season['TotalMatches'].min(),
    matches_per_team_per_season['WinPercentage'].max(),
    matches_per_team_per_season['WinPercentage'].min(),
    matches_per_team_per_season['TeamID'].nunique(),
    matches_per_team_per_season[matches_per_team_per_season.index >= 2015]['TeamID'].nunique(),
)


matches_per_team_per_season[matches_per_team_per_season['TotalMatches'] == 6]


#By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023


MNCAATourneyDetailedResults = pd.read_csv(f'{DIR}/march-machine-learning-mania-2025/MNCAATourneyDetailedResults.csv')
MTeams = pd.read_csv(f'{DIR}/march-machine-learning-mania-2025/MTeams.csv')
display(MNCAATourneyDetailedResults.head())
display(MTeams.head())


#By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023


stats_acroynm_to_description = {
    'WScore': 'Winning Team Score',
    'WFGM': 'Winning Team Field Goals Made',
    'WFGA': 'Winning Team Field Goals Attempted',
    'WFGM3': 'Winning Team 3-Point Field Goals Made',
    'WFGA3': 'Winning Team 3-Point Field Goals Attempted',
    'WFTM': 'Winning Team Free Throws Made',
    'WFTA': 'Winning Team Free Throws Attempted',
    'WOR': 'Winning Team Offensive Rebounds',
    'WDR': 'Winning Team Defensive Rebounds',
    'WAst': 'Winning Team Assists',
    'WTO': 'Winning Team Turnovers',
    'WStl': 'Winning Team Steals',
    'WBlk': 'Winning Team Blocks',
    'WPF': 'Winning Team Personal Fouls',
    'LScore': 'Losing Team Score',
    'LFGM': 'Losing Team Field Goals Made',
    'LFGA': 'Losing Team Field Goals Attempted',
    'LFGM3': 'Losing Team 3-Point Field Goals Made',
    'LFGA3': 'Losing Team 3-Point Field Goals Attempted',
    'LFTM': 'Losing Team Free Throws Made',
    'LFTA': 'Losing Team Free Throws Attempted',
    'LOR': 'Losing Team Offensive Rebounds',
    'LDR': 'Losing Team Defensive Rebounds',
    'LAst': 'Losing Team Assists',
    'LTO': 'Losing Team Turnovers',
    'LStl': 'Losing Team Steals',
    'LBlk': 'Losing Team Blocks',
    'LPF': 'Losing Team Personal Fouls',
}
winning_team_columns = ['WScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']
lossing_team_columns = ['LScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']

def detailed_results_season_and_team_level_results(df):
    winning_team_columns = ['WScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']
    lossing_team_columns = ['LScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
    matches_won_per_team_per_season = df.groupby(['Season', 'WTeamID'])['WScore'].count().reset_index().rename(columns={'WScore': 'WinCount'})
    matches_lost_per_team_per_season = df.groupby(['Season', 'LTeamID'])['LScore'].count().reset_index().rename(columns={'LScore': 'LossCount'})
    matches_per_team_per_season = pd.merge(matches_won_per_team_per_season, matches_lost_per_team_per_season, left_on=['Season', 'WTeamID'], right_on=['Season', 'LTeamID'])
    matches_per_team_per_season['TotalMatches'] = matches_per_team_per_season['WinCount'] + matches_per_team_per_season['LossCount']
    matches_per_team_per_season.rename(columns={'WTeamID': 'TeamID'}, inplace=True)
    matches_per_team_per_season.drop(columns=['LTeamID'], inplace=True)
    matches_per_team_per_season.set_index(['Season'], inplace=True)
    matches_per_team_per_season['WinPercentage'] = ((matches_per_team_per_season['WinCount'] / matches_per_team_per_season['TotalMatches']) * 100).round(2)
    winning_team_stats_total_per_season = df.groupby(['Season', 'WTeamID'])[winning_team_columns].sum().reset_index().rename(
            columns={
                'WScore': 'WScoreTotal', 'WFGM': 'WFGMTotal', 'WFGA': 'WFGATotal', 'WFGM3': 'WFGM3Total', 'WFGA3': 'WFGA3Total', 'WFTM': 'WFTMTotal', 'WFTA': 'WFTATotal',
                'WOR': 'WORTotal', 'WDR': 'WDRTotal', 'WAst': 'WAstTotal', 'WTO': 'WTOTotal', 'WStl': 'WStlTotal', 'WBlk': 'WBlkTotal', 'WPF': 'WPFTotal', 'WTeamID': 'TeamID',
            })
    winning_team_stats_mean_per_season = df.groupby(['Season', 'WTeamID'])[winning_team_columns].mean().reset_index().rename(
            columns={
                'WScore': 'WScoreMean', 'WFGM': 'WFGMMean', 'WFGA': 'WFGAMean', 'WFGM3': 'WFGM3Mean', 'WFGA3': 'WFGA3Mean','WFTM': 'WFTMMean', 'WFTA': 'WFTAMean', 'WOR': 'WORMean',
                'WDR': 'WDRMean', 'WAst': 'WAstMean', 'WTO': 'WTOMean', 'WStl': 'WStlMean', 'WBlk': 'WBlkMean', 'WPF': 'WPFMean', 'WTeamID': 'TeamID',
            }
        )

    lossing_team_stats_total_per_season = df.groupby(['Season', 'LTeamID'])[lossing_team_columns].sum().reset_index().rename(
        columns={
            'LScore': 'LScoreTotal', 'LFGM': 'LFGMTotal', 'LFGA': 'LFGATotal', 'LFGM3': 'LFGM3Total', 'LFGA3': 'LFGA3Total', 'LFTM': 'LFTMTotal', 'LFTA': 'LFTATotal', 'LOR': 'LORTotal',
            'LDR': 'LDRTotal', 'LAst': 'LAstTotal', 'LTO': 'LTOTotal', 'LStl': 'LStlTotal', 'LBlk': 'LBlkTotal', 'LPF': 'LPFTotal', 'LTeamID': 'TeamID'
        }
    )
    lossing_team_stats_mean_per_season = df.groupby(['Season', 'LTeamID'])[lossing_team_columns].mean().reset_index().rename(
        columns={
            'LScore': 'LScoreMean', 'LFGM': 'LFGMMean', 'LFGA': 'LFGAMean', 'LFGM3': 'LFGM3Mean', 'LFGA3': 'LFGA3Mean', 'LFTM': 'LFTMMean', 'LFTA': 'LFTAMean', 'LOR': 'LORMean',
            'LDR': 'LDRMean', 'LAst': 'LAstMean', 'LTO': 'LTOMean', 'LStl': 'LStlMean', 'LBlk': 'LBlkMean', 'LPF': 'LPFMean', 'LTeamID': 'TeamID'        
        }
    )
    matches_per_team_per_season = pd.merge(matches_per_team_per_season, winning_team_stats_total_per_season, on=['Season', 'TeamID'])
    matches_per_team_per_season = pd.merge(matches_per_team_per_season, winning_team_stats_mean_per_season, on=['Season', 'TeamID'])
    matches_per_team_per_season = pd.merge(matches_per_team_per_season, lossing_team_stats_total_per_season, on=['Season', 'TeamID'])
    matches_per_team_per_season = pd.merge(matches_per_team_per_season, lossing_team_stats_mean_per_season, on=['Season', 'TeamID'])
    matches_per_team_per_season = matches_per_team_per_season.merge(MTeams, on='TeamID', how='left')
    display(matches_per_team_per_season.head())
    return matches_per_team_per_season
    
def get_team_level_stats_from_season_wise_stats(df):
    team_win_loss_count = df.groupby('TeamName').agg({'WinCount': 'sum', 'LossCount': 'sum', 'Season': 'count'}).reset_index()
    team_win_loss_count['TotalMatches'] = team_win_loss_count['WinCount'] + team_win_loss_count['LossCount']
    team_win_loss_count = team_win_loss_count.rename(columns={'Season': 'SeasonCount'})
    team_win_loss_count['WinPercentage'] = ((team_win_loss_count['WinCount'] / (team_win_loss_count['WinCount'] + team_win_loss_count['LossCount'])) * 100).round(2)
    team_win_loss_count = team_win_loss_count.sort_values(by='WinCount', ascending=False).reset_index(drop=True)
    display(team_win_loss_count.head())
    return team_win_loss_count

def get_head_to_head_stats_from_mens_tourney_detailed_results(df):
    def get_wins_per_teams(row):
        team1_wins = MNCAATourneyDetailedResults[(MNCAATourneyDetailedResults['Team1'] == row['Team1']) & (MNCAATourneyDetailedResults['Team2'] == row['Team2']) & (MNCAATourneyDetailedResults['WTeamID'] == row['Team1'])].shape[0]
        team2_wins = MNCAATourneyDetailedResults[(MNCAATourneyDetailedResults['Team1'] == row['Team1']) & (MNCAATourneyDetailedResults['Team2'] == row['Team2']) & (MNCAATourneyDetailedResults['WTeamID'] == row['Team2'])].shape[0]
        return team1_wins, team2_wins

    def get_team_name_from_team_id(row):
        team1_name = MTeams[MTeams['TeamID'] == row['Team1']]['TeamName'].values[0]
        team2_name = MTeams[MTeams['TeamID'] == row['Team2']]['TeamName'].values[0]
        return team1_name, team2_name
    df[['Team1', 'Team2']] = df.apply(
        lambda x: sorted([x['WTeamID'], x['LTeamID']]), axis=1, result_type='expand'
    )
    head_to_head_matches = df.groupby(['Team1', 'Team2']).size().reset_index(name='MatchCount').sort_values(by='MatchCount', ascending=False)
    head_to_head_matches[['Team1Wins', 'Team2Wins']] = head_to_head_matches.apply(get_wins_per_teams, axis=1, result_type='expand')
    head_to_head_matches['Team1WinPercentage'] = head_to_head_matches.apply(lambda x: round(x['Team1Wins'] / x['MatchCount'] * 100, 2), axis=1)
    head_to_head_matches['Team2WinPercentage'] = head_to_head_matches.apply(lambda x: round(x['Team2Wins'] / x['MatchCount'] * 100, 2), axis=1)
    head_to_head_matches[['Team1Name', 'Team2Name']] = head_to_head_matches.apply(get_team_name_from_team_id, axis=1, result_type='expand')
    head_to_head_matches['Team1 vs Team2'] = head_to_head_matches.apply(lambda x: f"{x['Team1Name']} vs {x['Team2Name']}", axis=1)
    head_to_head_matches['Team2 vs Team1'] = head_to_head_matches.apply(lambda x: f"{x['Team2Name']} vs {x['Team1Name']}", axis=1)
    display(head_to_head_matches.head())
    return head_to_head_matches

def detailed_results_season_and_team_level_results_for_womens(df):
    winning_team_columns = ['WScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']
    lossing_team_columns = ['LScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
    matches_won_per_team_per_season = df.groupby(['Season', 'WTeamID'])['WScore'].count().reset_index().rename(columns={'WScore': 'WinCount'})
    matches_lost_per_team_per_season = df.groupby(['Season', 'LTeamID'])['LScore'].count().reset_index().rename(columns={'LScore': 'LossCount'})
    matches_per_team_per_season = pd.merge(matches_won_per_team_per_season, matches_lost_per_team_per_season, left_on=['Season', 'WTeamID'], right_on=['Season', 'LTeamID'])
    matches_per_team_per_season['TotalMatches'] = matches_per_team_per_season['WinCount'] + matches_per_team_per_season['LossCount']
    matches_per_team_per_season.rename(columns={'WTeamID': 'TeamID'}, inplace=True)
    matches_per_team_per_season.drop(columns=['LTeamID'], inplace=True)
    matches_per_team_per_season.set_index(['Season'], inplace=True)
    matches_per_team_per_season['WinPercentage'] = ((matches_per_team_per_season['WinCount'] / matches_per_team_per_season['TotalMatches']) * 100).round(2)
    winning_team_stats_total_per_season = df.groupby(['Season', 'WTeamID'])[winning_team_columns].sum().reset_index().rename(
            columns={
                'WScore': 'WScoreTotal', 'WFGM': 'WFGMTotal', 'WFGA': 'WFGATotal', 'WFGM3': 'WFGM3Total', 'WFGA3': 'WFGA3Total', 'WFTM': 'WFTMTotal', 'WFTA': 'WFTATotal',
                'WOR': 'WORTotal', 'WDR': 'WDRTotal', 'WAst': 'WAstTotal', 'WTO': 'WTOTotal', 'WStl': 'WStlTotal', 'WBlk': 'WBlkTotal', 'WPF': 'WPFTotal', 'WTeamID': 'TeamID',
            })
    winning_team_stats_mean_per_season = df.groupby(['Season', 'WTeamID'])[winning_team_columns].mean().reset_index().rename(
            columns={
                'WScore': 'WScoreMean', 'WFGM': 'WFGMMean', 'WFGA': 'WFGAMean', 'WFGM3': 'WFGM3Mean', 'WFGA3': 'WFGA3Mean','WFTM': 'WFTMMean', 'WFTA': 'WFTAMean', 'WOR': 'WORMean',
                'WDR': 'WDRMean', 'WAst': 'WAstMean', 'WTO': 'WTOMean', 'WStl': 'WStlMean', 'WBlk': 'WBlkMean', 'WPF': 'WPFMean', 'WTeamID': 'TeamID',
            }
        )

    lossing_team_stats_total_per_season = df.groupby(['Season', 'LTeamID'])[lossing_team_columns].sum().reset_index().rename(
        columns={
            'LScore': 'LScoreTotal', 'LFGM': 'LFGMTotal', 'LFGA': 'LFGATotal', 'LFGM3': 'LFGM3Total', 'LFGA3': 'LFGA3Total', 'LFTM': 'LFTMTotal', 'LFTA': 'LFTATotal', 'LOR': 'LORTotal',
            'LDR': 'LDRTotal', 'LAst': 'LAstTotal', 'LTO': 'LTOTotal', 'LStl': 'LStlTotal', 'LBlk': 'LBlkTotal', 'LPF': 'LPFTotal', 'LTeamID': 'TeamID'
        }
    )
    lossing_team_stats_mean_per_season = df.groupby(['Season', 'LTeamID'])[lossing_team_columns].mean().reset_index().rename(
        columns={
            'LScore': 'LScoreMean', 'LFGM': 'LFGMMean', 'LFGA': 'LFGAMean', 'LFGM3': 'LFGM3Mean', 'LFGA3': 'LFGA3Mean', 'LFTM': 'LFTMMean', 'LFTA': 'LFTAMean', 'LOR': 'LORMean',
            'LDR': 'LDRMean', 'LAst': 'LAstMean', 'LTO': 'LTOMean', 'LStl': 'LStlMean', 'LBlk': 'LBlkMean', 'LPF': 'LPFMean', 'LTeamID': 'TeamID'        
        }
    )
    matches_per_team_per_season = pd.merge(matches_per_team_per_season, winning_team_stats_total_per_season, on=['Season', 'TeamID'])
    matches_per_team_per_season = pd.merge(matches_per_team_per_season, winning_team_stats_mean_per_season, on=['Season', 'TeamID'])
    matches_per_team_per_season = pd.merge(matches_per_team_per_season, lossing_team_stats_total_per_season, on=['Season', 'TeamID'])
    matches_per_team_per_season = pd.merge(matches_per_team_per_season, lossing_team_stats_mean_per_season, on=['Season', 'TeamID'])
    matches_per_team_per_season = matches_per_team_per_season.merge(WTeams, on='TeamID', how='left')
    display(matches_per_team_per_season.head())
    return matches_per_team_per_season
def get_head_to_head_stats_from_womens_tourney_detailed_results(df):
    def get_wins_per_teams(row):
        team1_wins = WNCAATourneyDetailedResults[(WNCAATourneyDetailedResults['Team1'] == row['Team1']) & (WNCAATourneyDetailedResults['Team2'] == row['Team2']) & (WNCAATourneyDetailedResults['WTeamID'] == row['Team1'])].shape[0]
        team2_wins = WNCAATourneyDetailedResults[(WNCAATourneyDetailedResults['Team1'] == row['Team1']) & (WNCAATourneyDetailedResults['Team2'] == row['Team2']) & (WNCAATourneyDetailedResults['WTeamID'] == row['Team2'])].shape[0]
        return team1_wins, team2_wins

    def get_team_name_from_team_id(row):
        team1_name = WTeams[WTeams['TeamID'] == row['Team1']]['TeamName'].values[0]
        team2_name = WTeams[WTeams['TeamID'] == row['Team2']]['TeamName'].values[0]
        return team1_name, team2_name
    df[['Team1', 'Team2']] = df.apply(
        lambda x: sorted([x['WTeamID'], x['LTeamID']]), axis=1, result_type='expand'
    )
    head_to_head_matches = df.groupby(['Team1', 'Team2']).size().reset_index(name='MatchCount').sort_values(by='MatchCount', ascending=False)
    head_to_head_matches[['Team1Wins', 'Team2Wins']] = head_to_head_matches.apply(get_wins_per_teams, axis=1, result_type='expand')
    head_to_head_matches['Team1WinPercentage'] = head_to_head_matches.apply(lambda x: round(x['Team1Wins'] / x['MatchCount'] * 100, 2), axis=1)
    head_to_head_matches['Team2WinPercentage'] = head_to_head_matches.apply(lambda x: round(x['Team2Wins'] / x['MatchCount'] * 100, 2), axis=1)
    head_to_head_matches[['Team1Name', 'Team2Name']] = head_to_head_matches.apply(get_team_name_from_team_id, axis=1, result_type='expand')
    head_to_head_matches['Team1 vs Team2'] = head_to_head_matches.apply(lambda x: f"{x['Team1Name']} vs {x['Team2Name']}", axis=1)
    head_to_head_matches['Team2 vs Team1'] = head_to_head_matches.apply(lambda x: f"{x['Team2Name']} vs {x['Team1Name']}", axis=1)
    display(head_to_head_matches.head())
    return head_to_head_matches

def get_head_to_head_stats_from_mens_regular_session_detailed_results(df):
    def get_wins_per_teams(row):
        team1_wins = MRegularSeasonDetailedResults[
            (MRegularSeasonDetailedResults['Team1'] == row['Team1']) &
            (MRegularSeasonDetailedResults['Team2'] == row['Team2']) &
            (MRegularSeasonDetailedResults['WTeamID'] == row['Team1'])
        ].shape[0]
        team2_wins = MRegularSeasonDetailedResults[
            (MRegularSeasonDetailedResults['Team1'] == row['Team1']) &
            (MRegularSeasonDetailedResults['Team2'] == row['Team2']) &
            (MRegularSeasonDetailedResults['WTeamID'] == row['Team2'])
        ].shape[0]
        return team1_wins, team2_wins

    def get_team_name_from_team_id(row):
        team1_name = MTeams[MTeams['TeamID'] == row['Team1']]['TeamName'].values[0]
        team2_name = MTeams[MTeams['TeamID'] == row['Team2']]['TeamName'].values[0]
        return team1_name, team2_name
    df[['Team1', 'Team2']] = df.apply(
        lambda x: sorted([x['WTeamID'], x['LTeamID']]), axis=1, result_type='expand'
    )
    head_to_head_matches = df.groupby(['Team1', 'Team2']).size().reset_index(name='MatchCount').sort_values(by='MatchCount', ascending=False)
    head_to_head_matches[['Team1Wins', 'Team2Wins']] = head_to_head_matches.apply(get_wins_per_teams, axis=1, result_type='expand')
    head_to_head_matches['Team1WinPercentage'] = head_to_head_matches.apply(lambda x: round(x['Team1Wins'] / x['MatchCount'] * 100, 2), axis=1)
    head_to_head_matches['Team2WinPercentage'] = head_to_head_matches.apply(lambda x: round(x['Team2Wins'] / x['MatchCount'] * 100, 2), axis=1)
    head_to_head_matches[['Team1Name', 'Team2Name']] = head_to_head_matches.apply(get_team_name_from_team_id, axis=1, result_type='expand')
    head_to_head_matches['Team1 vs Team2'] = head_to_head_matches.apply(lambda x: f"{x['Team1Name']} vs {x['Team2Name']}", axis=1)
    head_to_head_matches['Team2 vs Team1'] = head_to_head_matches.apply(lambda x: f"{x['Team2Name']} vs {x['Team1Name']}", axis=1)
    display(head_to_head_matches.head())
    return head_to_head_matches

def get_head_to_head_stats_from_womens_regular_session_detailed_results(df):
    def get_wins_per_teams(row):
        team1_wins = WRegularSeasonDetailedResults[
            (WRegularSeasonDetailedResults['Team1'] == row['Team1']) &
            (WRegularSeasonDetailedResults['Team2'] == row['Team2']) &
            (WRegularSeasonDetailedResults['WTeamID'] == row['Team1'])
        ].shape[0]
        team2_wins = WRegularSeasonDetailedResults[
            (WRegularSeasonDetailedResults['Team1'] == row['Team1']) &
            (WRegularSeasonDetailedResults['Team2'] == row['Team2']) &
            (WRegularSeasonDetailedResults['WTeamID'] == row['Team2'])
        ].shape[0]
        return team1_wins, team2_wins

    def get_team_name_from_team_id(row):
        team1_name = WTeams[WTeams['TeamID'] == row['Team1']]['TeamName'].values[0]
        team2_name = WTeams[WTeams['TeamID'] == row['Team2']]['TeamName'].values[0]
        return team1_name, team2_name
    df[['Team1', 'Team2']] = df.apply(
        lambda x: sorted([x['WTeamID'], x['LTeamID']]), axis=1, result_type='expand'
    )
    head_to_head_matches = df.groupby(['Team1', 'Team2']).size().reset_index(name='MatchCount').sort_values(by='MatchCount', ascending=False)
    head_to_head_matches[['Team1Wins', 'Team2Wins']] = head_to_head_matches.apply(get_wins_per_teams, axis=1, result_type='expand')
    head_to_head_matches['Team1WinPercentage'] = head_to_head_matches.apply(lambda x: round(x['Team1Wins'] / x['MatchCount'] * 100, 2), axis=1)
    head_to_head_matches['Team2WinPercentage'] = head_to_head_matches.apply(lambda x: round(x['Team2Wins'] / x['MatchCount'] * 100, 2), axis=1)
    head_to_head_matches[['Team1Name', 'Team2Name']] = head_to_head_matches.apply(get_team_name_from_team_id, axis=1, result_type='expand')
    head_to_head_matches['Team1 vs Team2'] = head_to_head_matches.apply(lambda x: f"{x['Team1Name']} vs {x['Team2Name']}", axis=1)
    head_to_head_matches['Team2 vs Team1'] = head_to_head_matches.apply(lambda x: f"{x['Team2Name']} vs {x['Team1Name']}", axis=1)
    display(head_to_head_matches.head())
    return head_to_head_matches


mens_matches_per_team_per_season = detailed_results_season_and_team_level_results(
    MNCAATourneyDetailedResults
)


#By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023


matches_per_season = MNCAATourneyDetailedResults.groupby('Season')[['DayNum']].count().reset_index().rename(columns={'DayNum': 'MatchCount'})

fig = plt.figure(figsize=(16, 6))
ax = sns.barplot(
    y=matches_per_season['Season'],
    x=matches_per_season['MatchCount'],
    orient='h'
)
ax.set_title('Matches per season', fontsize=14)
ax.set_xlabel('Match Count', fontsize=14)
ax.set_ylabel('Season', fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right', fontsize=14)
for container in ax.containers:
    ax.bar_label(container, padding=3, fontsize=12, color='black', label_type='edge', weight='bold')
plt.tight_layout()
plt.show()


#By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023


teams_per_season = MNCAATourneyDetailedResults.groupby('Season')[['WTeamID', 'LTeamID']].nunique().reset_index().rename(columns={'WTeamID': 'WinningTeamCount', 'LTeamID': 'LosingTeamCount'})
teams_per_season['TotalTeams'] = teams_per_season['WinningTeamCount'] + teams_per_season['LosingTeamCount']

fig = plt.figure(figsize=(16, 6))
ax = sns.barplot(y=teams_per_season['Season'], x=teams_per_season['TotalTeams'], orient='h')
ax.set_title('Teams count per season', fontsize=14)
ax.set_xlabel('Season', fontsize=14)
ax.set_ylabel('Team Count', fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right', fontsize=14)
for container in ax.containers:
    ax.bar_label(container, padding=3, fontsize=12, color='black', label_type='edge', weight='bold')
plt.tight_layout()
plt.show()


rows, columns = len(winning_team_columns), 3
fig, ax = plt.subplots(rows, columns, figsize=(32, 7 * rows), sharex=False, sharey=False)
current_row, current_column, i = 0, 0, 0
for winning_stat_column, lossing_stat_column in zip(winning_team_columns, lossing_team_columns):
    stat_description = stats_acroynm_to_description[winning_stat_column]
    stat_description_update = stat_description.replace('Winning Team', 'Winning vs Losing Team Total')
    winning_stat_column = f"{winning_stat_column}Total"
    lossing_stat_column = f"{lossing_stat_column}Total"

    sns.histplot(mens_matches_per_team_per_season[winning_stat_column], ax=ax[current_row, 1], label='Winning Team', color=palette[0], alpha=0.5, bins=20, kde=True)
    sns.histplot(mens_matches_per_team_per_season[lossing_stat_column], ax=ax[current_row, 1], label='Lossing Team', color=palette[1], alpha=0.5, bins=20, kde=True)
    
    ax[current_row, 1].set_title(stat_description_update, fontsize=16)
    ax[current_row, 1].set_xlabel(stat_description, fontsize=12)
    ax[current_row, 1].set_ylabel('Frequency', fontsize=12)
    ax[current_row, 1].legend()
    
    sns.lineplot(data=mens_matches_per_team_per_season,  x='Season', y=winning_stat_column, ax=ax[current_row, 2], marker='o', markersize=5, label='Winning Team')
    sns.lineplot(data=mens_matches_per_team_per_season, x='Season', y=lossing_stat_column, ax=ax[current_row, 2], marker='x', markersize=5, label='Lossing Team')
    
    temp_df = mens_matches_per_team_per_season[['Season', winning_stat_column]]
    temp_df['Type'] = 'Winning Team'
    temp_df.rename(columns={winning_stat_column: winning_stat_column[1:]}, inplace=True)
    temp_df1 = mens_matches_per_team_per_season[['Season', lossing_stat_column]]
    temp_df1['Type'] = 'Lossing Team'
    temp_df1.rename(columns={lossing_stat_column: lossing_stat_column[1:]}, inplace=True)
    temp_df = pd.concat([temp_df, temp_df1])
    sns.boxplot(x='Season', y=winning_stat_column[1:], hue='Type', data=temp_df, ax=ax[current_row, 0])

    current_row += 1
plt.tight_layout()
plt.show()


##By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023

rows, columns = len(winning_team_columns), 3
fig, ax = plt.subplots(rows, columns, figsize=(32, 7 * rows), sharex=False, sharey=False)
current_row, current_column, i = 0, 0, 0
for winning_stat_column, lossing_stat_column in zip(winning_team_columns, lossing_team_columns):
    stat_description = stats_acroynm_to_description[winning_stat_column]
    stat_description_update = stat_description.replace('Winning Team', 'Winning vs Losing Team Mean')
    winning_stat_column = f"{winning_stat_column}Mean"
    lossing_stat_column = f"{lossing_stat_column}Mean"

    sns.histplot(mens_matches_per_team_per_season[winning_stat_column], ax=ax[current_row, 1], label='Winning Team', color=palette[0], alpha=0.5, bins=20, kde=True)
    sns.histplot(mens_matches_per_team_per_season[lossing_stat_column], ax=ax[current_row, 1], label='Lossing Team', color=palette[1], alpha=0.5, bins=20, kde=True)
    ax[current_row, 1].set_title(stat_description_update, fontsize=16)
    ax[current_row, 1].set_xlabel(stat_description, fontsize=12)
    ax[current_row, 1].set_ylabel('Frequency', fontsize=12)
    ax[current_row, 1].legend()
    
    sns.lineplot(data=mens_matches_per_team_per_season,  x='Season', y=winning_stat_column, ax=ax[current_row, 2], marker='o', markersize=5, label='Winning Team')
    sns.lineplot(data=mens_matches_per_team_per_season, x='Season', y=lossing_stat_column, ax=ax[current_row, 2], marker='x', markersize=5, label='Lossing Team')

    temp_df = mens_matches_per_team_per_season[['Season', winning_stat_column]]
    temp_df['Type'] = 'Winning Team'
    temp_df.rename(columns={winning_stat_column: winning_stat_column[1:]}, inplace=True)
    temp_df1 = mens_matches_per_team_per_season[['Season', lossing_stat_column]]
    temp_df1['Type'] = 'Lossing Team'
    temp_df1.rename(columns={lossing_stat_column: lossing_stat_column[1:]}, inplace=True)
    temp_df = pd.concat([temp_df, temp_df1])
    sns.boxplot(x='Season', y=winning_stat_column[1:], hue='Type', data=temp_df, ax=ax[current_row, 0])

    current_row += 1
plt.tight_layout()
plt.show()


rows, columns = mens_matches_per_team_per_season['Season'].nunique(), 2

fig, ax = plt.subplots(rows, columns, figsize=(24, 6 * rows), sharex=False, sharey=False)
current_row = 0
for season in mens_matches_per_team_per_season['Season'].unique():
    required_df = mens_matches_per_team_per_season[mens_matches_per_team_per_season['Season'] == season].sort_values(by=['WinPercentage', 'WScoreTotal'], ascending=False).head(10)
    sns.barplot(x='TeamName', y='WinPercentage', data=required_df, ax=ax[current_row, 0])
    ax[current_row, 0].set_title(f'Top 10 Teams with Highest Win Percentage in {season}')
    ax[current_row, 0].set_xlabel(None)
    ax[current_row, 0].set_ylabel(None)
    ax[current_row, 0].set_xticklabels(ax[current_row, 0].get_xticklabels(), rotation=45, horizontalalignment='right')
    for container in ax[current_row, 0].containers:
        ax[current_row, 0].bar_label(container, padding=3, fontsize=12, color='black', label_type='edge', weight='bold')

    required_df = mens_matches_per_team_per_season[mens_matches_per_team_per_season['Season'] == season].sort_values(by=['WinPercentage', 'WScoreTotal'], ascending=True).head(10)
    sns.barplot(x='TeamName', y='WinPercentage', data=required_df, ax=ax[current_row, 1])
    ax[current_row, 1].set_title(f'Top 10 Teams with Lowest Win Percentage in {season}')
    ax[current_row, 1].set_xlabel(None)
    ax[current_row, 1].set_ylabel(None)
    ax[current_row, 1].set_xticklabels(ax[current_row, 1].get_xticklabels(), rotation=45, horizontalalignment='right')
    for container in ax[current_row, 1].containers:
        ax[current_row, 1].bar_label(container, padding=3, fontsize=12, color='black', label_type='edge', weight='bold')
    current_row += 1
plt.tight_layout()
plt.show()


from collections import Counter
top5_win_percentage_teams_across_all_seasons = []
bottom5_win_percentage_teams_across_all_seasons = []
for season in mens_matches_per_team_per_season['Season'].unique():
    required_df = mens_matches_per_team_per_season[mens_matches_per_team_per_season['Season'] == season].sort_values(by=['WinPercentage', 'WScoreTotal'], ascending=False).head(10)
    top5_win_percentage_teams_across_all_seasons.extend(required_df['TeamName'].values.tolist())
    required_df = mens_matches_per_team_per_season[mens_matches_per_team_per_season['Season'] == season].sort_values(by=['WinPercentage', 'WScoreTotal'], ascending=True).head(10)
    bottom5_win_percentage_teams_across_all_seasons.extend(required_df['TeamName'].values.tolist())
top5_repeat_count_df = pd.DataFrame.from_dict(Counter(top5_win_percentage_teams_across_all_seasons), orient='index').reset_index().rename(columns={0: 'Count', 'index': 'TeamName'}).sort_values(by='Count', ascending=False)
bottom5_repeat_count_df = pd.DataFrame.from_dict(Counter(bottom5_win_percentage_teams_across_all_seasons), orient='index').reset_index().rename(columns={0: 'Count', 'index': 'TeamName'}).sort_values(by='Count', ascending=False)
print(set(top5_repeat_count_df['TeamName'].head(10).values.tolist()).intersection(set(bottom5_repeat_count_df['TeamName'].head(10).values.tolist())))

fig, ax = plt.subplots(1, 2, figsize=(24, 6), sharex=False, sharey=False)

sns.barplot(x=top5_repeat_count_df.head(10)['TeamName'], y=top5_repeat_count_df.head(10)['Count'], ax=ax[0])
ax[0].set_title('Team with most appearances in Top 10 Teams with Highest Win Percentage across all Seasons', fontsize=14)
ax[0].set_xlabel(None)
ax[0].set_ylabel(None)
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=12)
for container in ax[0].containers:
    ax[0].bar_label(container, padding=3, fontsize=12, color='black', label_type='edge', weight='bold')
    
sns.barplot(x=bottom5_repeat_count_df.head(10)['TeamName'], y=bottom5_repeat_count_df.head(10)['Count'], ax=ax[1])
ax[1].set_title('Team with most appearances in bottom 10 Teams with Lowest Win Percentage across all Seasons', fontsize=14)
ax[1].set_xlabel(None)
ax[1].set_ylabel(None)
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=12)
for container in ax[1].containers:
    ax[1].bar_label(container, padding=3, fontsize=12, color='black', label_type='edge', weight='bold')

plt.tight_layout()
plt.show();


mens_team_win_loss_count = get_team_level_stats_from_season_wise_stats(mens_matches_per_team_per_season)


mens_head_to_head_matches = get_head_to_head_stats_from_mens_tourney_detailed_results(MNCAATourneyDetailedResults)


##By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023

fig, ax = plt.subplots(1, 2, figsize=(28, 6))

sns.barplot(
    x=mens_head_to_head_matches[
        (mens_head_to_head_matches['Team1WinPercentage'] > 50) &
        (mens_head_to_head_matches['MatchCount'] > 2)
    ].sort_values(
        by='Team1WinPercentage', ascending=False
    ).head(20)['Team1 vs Team2'],
    y=mens_head_to_head_matches[
        (mens_head_to_head_matches['Team1WinPercentage'] > 50) &
        (mens_head_to_head_matches['MatchCount'] > 2)
    ].sort_values(by='Team1WinPercentage', ascending=False).head(20)['Team1WinPercentage'],
    ax=ax[0]
)
ax[0].set_title('Team1 vs Team2 with Team1 winning more than 50% of the matches', fontsize=14)
ax[0].set_xlabel(None)
ax[0].set_ylabel(None)
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=12)
labels = mens_head_to_head_matches[
    (mens_head_to_head_matches['Team1WinPercentage'] > 50) &
    (mens_head_to_head_matches['MatchCount'] > 2)
].sort_values(by='Team1WinPercentage', ascending=False).head(20).apply(lambda x: f"{x['Team1WinPercentage']}% ({x['MatchCount']})", axis=1)
for container in ax[0].containers:
    ax[0].bar_label(container, padding=3, fontsize=12, color='black', label_type='edge', weight='bold', labels=labels)
    
# Plot for Team2 vs Team1 with Team2 winning more than 50% of the matches
sns.barplot(
    x=mens_head_to_head_matches[
        (mens_head_to_head_matches['Team2WinPercentage'] > 50) &
        (mens_head_to_head_matches['MatchCount'] > 2)
    ].sort_values(by='Team2WinPercentage', ascending=False).head(20)['Team2 vs Team1'],
    y=mens_head_to_head_matches[
        (mens_head_to_head_matches['Team2WinPercentage'] > 50) &
        (mens_head_to_head_matches['MatchCount'] > 2)
    ].sort_values(by='Team2WinPercentage', ascending=False).head(20)['Team2WinPercentage'],
    ax=ax[1]
)
ax[1].set_title('Team2 vs Team1 with Team2 winning more than 50% of the matches', fontsize=14)
ax[1].set_xlabel(None)
ax[1].set_ylabel(None)
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=12)
labels = mens_head_to_head_matches[
    (mens_head_to_head_matches['Team2WinPercentage'] > 50) &
    (mens_head_to_head_matches['MatchCount'] > 2)   
].sort_values(by='Team2WinPercentage', ascending=False).head(20).apply(lambda x: f"{x['Team2WinPercentage']}% ({x['MatchCount']})", axis=1)
for container in ax[1].containers:
    ax[1].bar_label(container, padding=3, fontsize=12, color='black', label_type='edge', weight='bold', labels=labels)

plt.tight_layout()
plt.show()


MRegularSeasonDetailedResults = pd.read_csv(f'{DIR}/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv')
print(f"Total number of regular season matches: {MRegularSeasonDetailedResults.shape[0]}")
print(f"Columns: {MRegularSeasonDetailedResults.columns}")
display(MRegularSeasonDetailedResults.head())


#By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023


matches_per_season = MRegularSeasonDetailedResults.groupby('Season')[['DayNum']].count().reset_index().rename(columns={'DayNum': 'MatchCount'})

fig = plt.figure(figsize=(16, 6))
ax = sns.barplot(
    y=matches_per_season['Season'],
    x=matches_per_season['MatchCount'],
    orient='h'
)
ax.set_title('Matches per season', fontsize=14)
ax.set_xlabel('Match Count', fontsize=14)
ax.set_ylabel('Season', fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right', fontsize=14)
for container in ax.containers:
    ax.bar_label(container, padding=3, fontsize=12, color='black', label_type='edge', weight='bold')
plt.tight_layout()
plt.show() 


#By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023

teams_per_season = MRegularSeasonDetailedResults.groupby('Season')[['WTeamID', 'LTeamID']].nunique().reset_index().rename(columns={'WTeamID': 'WinningTeamCount', 'LTeamID': 'LosingTeamCount'})
teams_per_season['TotalTeams'] = teams_per_season['WinningTeamCount'] + teams_per_season['LosingTeamCount']

# Plot teams count per season sns.barplot
fig = plt.figure(figsize=(16, 6))
ax = sns.barplot(y=teams_per_season['Season'], x=teams_per_season['TotalTeams'], orient='h')
ax.set_title('Teams count per season', fontsize=14)
ax.set_xlabel('Season', fontsize=14)
ax.set_ylabel('Team Count', fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right', fontsize=14)
for container in ax.containers:
    ax.bar_label(container, padding=3, fontsize=12, color='black', label_type='edge', weight='bold')
plt.tight_layout()
plt.show()


mens_regular_matches_per_team_per_season = detailed_results_season_and_team_level_results(
    MRegularSeasonDetailedResults
)
print(f"Total number of teams: {mens_regular_matches_per_team_per_season.shape}")


rows, columns = len(winning_team_columns), 3
fig, ax = plt.subplots(rows, columns, figsize=(32, 7 * rows), sharex=False, sharey=False)
current_row, current_column, i = 0, 0, 0
for winning_stat_column, lossing_stat_column in zip(winning_team_columns, lossing_team_columns):
    stat_description = stats_acroynm_to_description[winning_stat_column]
    stat_description_update = stat_description.replace('Winning Team', 'Winning vs Losing Team Total')
    winning_stat_column = f"{winning_stat_column}Total"
    lossing_stat_column = f"{lossing_stat_column}Total"

    sns.histplot(mens_regular_matches_per_team_per_season[winning_stat_column], ax=ax[current_row, 1], label='Winning Team', color=palette[0], alpha=0.5, bins=20, kde=True)
    sns.histplot(mens_regular_matches_per_team_per_season[lossing_stat_column], ax=ax[current_row, 1], label='Lossing Team', color=palette[1], alpha=0.5, bins=20, kde=True)
    ax[current_row, 1].set_title(stat_description_update, fontsize=16)
    ax[current_row, 1].set_xlabel(stat_description, fontsize=12)
    ax[current_row, 1].set_ylabel('Frequency', fontsize=12)
    ax[current_row, 1].legend()
    
    sns.lineplot(data=mens_regular_matches_per_team_per_season,  x='Season', y=winning_stat_column, ax=ax[current_row, 2], marker='o', markersize=5, label='Winning Team')
    sns.lineplot(data=mens_regular_matches_per_team_per_season, x='Season', y=lossing_stat_column, ax=ax[current_row, 2], marker='x', markersize=5, label='Lossing Team')

    temp_df = mens_regular_matches_per_team_per_season[['Season', winning_stat_column]]
    temp_df['Type'] = 'Winning Team'
    temp_df.rename(columns={winning_stat_column: winning_stat_column[1:]}, inplace=True)
    temp_df1 = mens_regular_matches_per_team_per_season[['Season', lossing_stat_column]]
    temp_df1['Type'] = 'Lossing Team'
    temp_df1.rename(columns={lossing_stat_column: lossing_stat_column[1:]}, inplace=True)
    temp_df = pd.concat([temp_df, temp_df1])
    sns.boxplot(x='Season', y=winning_stat_column[1:], hue='Type', data=temp_df,  ax=ax[current_row, 0])
    
    current_row += 1
plt.tight_layout()
plt.show()


mens_regular_matches_per_team_per_season.sort_values(
    by='WinPercentage', ascending=True
).head()


##By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023

rows, columns = mens_regular_matches_per_team_per_season['Season'].nunique(), 2

fig, ax = plt.subplots(rows, columns, figsize=(24, 6 * rows), sharex=False, sharey=False)
current_row = 0
for season in mens_regular_matches_per_team_per_season['Season'].unique():
    required_df = mens_regular_matches_per_team_per_season[
        mens_regular_matches_per_team_per_season['Season'] == season
    ].sort_values(by=['WinPercentage', 'WScoreTotal'], ascending=False).head(10)
    sns.barplot(
        x='TeamName', y='WinPercentage', data=required_df, ax=ax[current_row, 0]
    )
    ax[current_row, 0].set_title(f'Top 10 Teams with Highest Win Percentage in {season}')
    ax[current_row, 0].set_xlabel(None)
    ax[current_row, 0].set_ylabel(None)
    ax[current_row, 0].set_xticklabels(ax[current_row, 0].get_xticklabels(), rotation=45, horizontalalignment='right')
    for container in ax[current_row, 0].containers:
        ax[current_row, 0].bar_label(container, padding=3, fontsize=12, color='black', label_type='edge', weight='bold')

    required_df = mens_regular_matches_per_team_per_season[
        mens_regular_matches_per_team_per_season['Season'] == season
        ].sort_values(by=['WinPercentage', 'WScoreTotal'], ascending=True).head(10)
    sns.barplot(x='TeamName', y='WinPercentage', data=required_df, ax=ax[current_row, 1])
    ax[current_row, 1].set_title(f'Top 10 Teams with Lowest Win Percentage in {season}')
    ax[current_row, 1].set_xlabel(None)
    ax[current_row, 1].set_ylabel(None)
    ax[current_row, 1].set_xticklabels(ax[current_row, 1].get_xticklabels(), rotation=45, horizontalalignment='right')
    for container in ax[current_row, 1].containers:
        ax[current_row, 1].bar_label(container, padding=3, fontsize=12, color='black', label_type='edge', weight='bold')
    current_row += 1
plt.tight_layout()
plt.show()


mens_regular_team_win_loss_count = get_team_level_stats_from_season_wise_stats(
    mens_regular_matches_per_team_per_season
)


mens_regular_head_to_head_matches = get_head_to_head_stats_from_mens_regular_session_detailed_results(MRegularSeasonDetailedResults)


#By Pradeep Singh https://www.kaggle.com/code/pardeep19singh/extensive-eda-with-actionable-insights#March-Machine-Learning-Mania-2023


fig, ax = plt.subplots(2, 1, figsize=(28, 14))

# Plot for Team1 vs Team2 with Team1 winning more than 50% of the matches
sns.barplot(
    x=mens_regular_head_to_head_matches[
        (mens_regular_head_to_head_matches['Team1WinPercentage'] > 50) &
        (mens_regular_head_to_head_matches['MatchCount'] > 2)
    ].sort_values(
        by='Team1WinPercentage', ascending=False
    ).head(20)['Team1 vs Team2'],
    y=mens_regular_head_to_head_matches[
        (mens_regular_head_to_head_matches['Team1WinPercentage'] > 50) &
        (mens_regular_head_to_head_matches['MatchCount'] > 2)
    ].sort_values(by='Team1WinPercentage', ascending=False).head(20)['Team1WinPercentage'],
    ax=ax[0]
)
ax[0].set_title('Team1 vs Team2 with Team1 winning more than 50% of the matches', fontsize=14)
ax[0].set_xlabel(None)
ax[0].set_ylabel(None)
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=12)
labels = mens_regular_head_to_head_matches[
    (mens_regular_head_to_head_matches['Team1WinPercentage'] > 50) &
    (mens_regular_head_to_head_matches['MatchCount'] > 2)
].sort_values(by='Team1WinPercentage', ascending=False).head(20).apply(lambda x: f"{x['Team1WinPercentage']}% ({x['MatchCount']})", axis=1)
for container in ax[0].containers:
    ax[0].bar_label(container, padding=3, fontsize=12, color='black', label_type='edge', weight='bold', labels=labels)
    
# Plot for Team2 vs Team1 with Team2 winning more than 50% of the matches
sns.barplot(
    x=mens_regular_head_to_head_matches[
        (mens_regular_head_to_head_matches['Team2WinPercentage'] > 50) &
        (mens_regular_head_to_head_matches['MatchCount'] > 2)
    ].sort_values(by='Team2WinPercentage', ascending=False).head(20)['Team2 vs Team1'],
    y=mens_regular_head_to_head_matches[
        (mens_regular_head_to_head_matches['Team2WinPercentage'] > 50) &
        (mens_regular_head_to_head_matches['MatchCount'] > 2)
    ].sort_values(by='Team2WinPercentage', ascending=False).head(20)['Team2WinPercentage'],
    ax=ax[1]
)
ax[1].set_title('Team2 vs Team1 with Team2 winning more than 50% of the matches', fontsize=14)
ax[1].set_xlabel(None)
ax[1].set_ylabel(None)
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=12)
labels = mens_regular_head_to_head_matches[
    (mens_regular_head_to_head_matches['Team2WinPercentage'] > 50) &
    (mens_regular_head_to_head_matches['MatchCount'] > 2)   
].sort_values(by='Team2WinPercentage', ascending=False).head(20).apply(lambda x: f"{x['Team2WinPercentage']}% ({x['MatchCount']})", axis=1)
for container in ax[1].containers:
    ax[1].bar_label(container, padding=3, fontsize=12, color='black', label_type='edge', weight='bold', labels=labels)

plt.tight_layout()
plt.show()

