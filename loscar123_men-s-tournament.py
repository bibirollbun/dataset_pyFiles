import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
%matplotlib inline


!ls -GFlash ../input/march-machine-learning-mania-2025/


INPUT_DIR = '../input/march-machine-learning-mania-2025/'

# men´s read data
men_tourney = pd.read_csv(INPUT_DIR + 'MNCAATourneyCompactResults.csv')
men_seeds = pd.read_csv(INPUT_DIR + 'MNCAATourneySeeds.csv')
men_teams = pd.read_csv(INPUT_DIR + 'MTeams.csv')
men_seasons = pd.read_csv(INPUT_DIR + 'MRegularSeasonCompactResults.csv').query('Season == 2025')
men_slots = pd.read_csv(INPUT_DIR + 'MNCAATourneySlots.csv').query('Season == 2024')

# Other data
conferences = pd.read_csv(INPUT_DIR + 'MTeamConferences.csv')
regular_detail = pd.read_csv(INPUT_DIR + 'MRegularSeasonDetailedResults.csv')
tourney_compact = pd.read_csv(INPUT_DIR + 'MNCAATourneyCompactResults.csv')
matches = pd.read_csv(INPUT_DIR + 'SampleSubmissionStage2.csv')


team_map = men_teams.set_index('TeamID')['TeamName']


# men´s counts
men_wins_counts = men_seasons.groupby(['Season','WTeamID']).size().reset_index(name='WinCount')
men_lose_counts = men_seasons.groupby(['Season','LTeamID']).size().reset_index(name='LoseCount')

men_counts = pd.concat([men_wins_counts, men_lose_counts], ignore_index = True)
men_counts


matches.head()


men_seasons = men_seasons.drop(['NumOT' ,'WLoc'], axis = 1)
men_seasons['ScoreDiff'] = men_seasons['WScore'] - men_seasons['LScore']

men_seasons['WTeamName'] = men_seasons['WTeamID'].map(team_map)
men_seasons['LTeamName'] = men_seasons['LTeamID'].map(team_map)


# each team´s # of wins
num_win = men_seasons.groupby(['Season', 'WTeamID']).count()
num_win = num_win.reset_index()[['Season', 'WTeamID', 'DayNum']] \
    .rename(columns = {'DayNum': 'NumWins', 'WTeamID': 'TeamID'})

# each team´s # of loss
num_loss = men_seasons.groupby(['Season', 'LTeamID']).count()
num_loss = num_loss.reset_index()[['Season', 'LTeamID', 'DayNum']] \
    .rename(columns = {'DayNum': 'NumLosses', 'LTeamID': 'TeamID'})

men_season_win_loss = num_win.merge(num_loss, on = ['Season', 'TeamID'], how = 'outer')
men_season_win_loss['TeamName'] = men_season_win_loss['TeamID'].map(team_map)


# how much points they scored more in average
gap_win = men_seasons.groupby(['Season', 'WTeamID'])['ScoreDiff'].mean().reset_index()
gap_win = gap_win[['Season', 'WTeamID', 'ScoreDiff']] \
    .rename(columns = {'ScoreDiff': 'DiffWins', 'WTeamID': 'TeamID'})

# how much points they scored less in average
gap_loss = men_seasons.groupby(['Season', 'LTeamID'])['ScoreDiff'].mean().reset_index()
gap_loss = gap_loss[['Season', 'LTeamID', 'ScoreDiff']] \
    .rename(columns = {'ScoreDiff': 'DiffLosses', 'LTeamID': 'TeamID'})


gap = pd.concat([gap_win, gap_loss], ignore_index = True)


gap


# Features season wins
men_features_season_wins = men_seasons.groupby(['Season', 'WTeamID']) \
    .count().reset_index()[['Season', 'WTeamID']].rename(columns = {'WTeamID': 'TeamID'})

# Features season loss
men_features_season_loss = men_seasons.groupby(['Season', 'LTeamID']) \
    .count().reset_index()[['Season', 'LTeamID']].rename(columns = {'LTeamID': 'TeamID'})

# Features differences season´s wins and losses
men_features = pd.concat([
    men_features_season_wins,
    men_features_season_loss], axis = 0) \
    .drop_duplicates() \
    .sort_values(['Season', 'TeamID']).reset_index(drop = True)

men_features = men_features.merge(num_win, on = ['Season', 'TeamID'], how = 'outer')
men_features = men_features.merge(num_loss, on = ['Season', 'TeamID'], how = 'outer')
men_features = men_features.merge(gap_win, on = ['Season', 'TeamID'], how = 'outer')
men_features = men_features.merge(gap_loss, on = ['Season', 'TeamID'], how = 'outer')

men_features = men_features.fillna(0)

men_features[['NumWins', 'NumLosses']] = men_features[['NumWins', 'NumLosses']].astype('int')


men_features['WinRatio'] = men_features['NumWins'] / (men_features['NumWins'] * men_features['NumLosses']) 
men_features['ScoreDiffAvg'] = (
    (men_features['NumWins'] * men_features['DiffWins'] - 
     men_features['NumLosses'] * men_features['DiffLosses'])
    / (men_features['NumWins'] * men_features['NumLosses'])
)


win_teams = pd.DataFrame()
lose_teams = pd.DataFrame()

columns = ['Season', 'TeamID', 'Points', 'OppPoints',
       'Loc', 'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
       'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'OppFGM', 'OppFGA',
       'OppFGM3', 'OppFGA3', 'OppFTM', 'OppFTA', 'OppOR', 'OppDR', 'OppAst', 'OppTO',
       'OppStl', 'OppBlk', 'OppPF']

win_teams[columns] = regular_detail[['Season', 'WTeamID', 'WScore', 'LScore',
       'WLoc', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA',
       'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA',
       'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO',
       'LStl', 'LBlk', 'LPF']]

win_teams['Wins'] = 1
win_teams['Losses'] = 0

lose_teams[columns] = regular_detail[['Season', 'LTeamID', 'LScore', 'WScore',
       'WLoc', 'NumOT', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
       'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'WFGM', 'WFGA',
       'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO',
       'WStl', 'WBlk', 'WPF']]

lose_teams['Wins'] = 0
lose_teams['Losses'] = 1


def change_loc(loc):
    if loc == 'Home':
        return 'Away'
    elif loc == 'Away':
        return 'None'

lose_teams['Loc'] = lose_teams['Loc'].apply(change_loc)


win_lose_teams = pd.concat([win_teams, lose_teams], ignore_index = True)
combined_teams = win_lose_teams.groupby(['Season', 'TeamID']).sum()
combined_teams['Num_Games'] = combined_teams['Wins'] + combined_teams['Losses']

combined_teams.columns.values


regular_season_input = pd.DataFrame()

regular_season_input['Win_Ratio'] = combined_teams['Wins'] / combined_teams['Num_Games']

regular_season_input['Points_Per_Game'] = combined_teams['Points'] / combined_teams['Num_Games']
regular_season_input['Points_Allowed_Per_Game'] = combined_teams['OppPoints'] / combined_teams['Num_Games']
regular_season_input['Points_Ratio'] = combined_teams['Points'] / combined_teams['OppPoints']

regular_season_input['OTs_Per_Game'] = combined_teams['NumOT'] / combined_teams['Num_Games']

regular_season_input['FG_Per_Game'] = combined_teams['FGM'] / combined_teams['Num_Games']
regular_season_input['FG_Ratio'] = combined_teams['FGM'] / combined_teams['FGA']
regular_season_input['FG_Allowed_Per_Game'] = combined_teams['OppFGM'] / combined_teams['Num_Games']

regular_season_input['FG3_Per_Game'] = combined_teams['FGM3'] / combined_teams['Num_Games']
regular_season_input['FG3_Ratio'] = combined_teams['FGM3'] / combined_teams['FGA3']
regular_season_input['FG3_Allowed_Per_Game'] = combined_teams['OppFGM3'] / combined_teams['Num_Games']

regular_season_input['FT_Per_Game'] = combined_teams['FTM'] / combined_teams['Num_Games']
regular_season_input['FT_Ratio'] = combined_teams['FTM'] / combined_teams['FTA']
regular_season_input['FT_Allowed_Per_Game'] = combined_teams['OppFTM'] / combined_teams['Num_Games']

regular_season_input['OR_Ratio'] = combined_teams['OR'] / (combined_teams['OR'] + combined_teams['OppDR'])
regular_season_input['DR_Ratio'] = combined_teams['DR'] / (combined_teams['DR'] + combined_teams['OppOR'])
regular_season_input['Ast_Per_game'] = combined_teams['Ast'] / combined_teams['Num_Games']
regular_season_input['TO_Per_game'] = combined_teams['TO'] / combined_teams['Num_Games']
regular_season_input['Stl_Per_game'] = combined_teams['Stl'] / combined_teams['Num_Games']
regular_season_input['Blk_Per_game'] = combined_teams['Blk'] / combined_teams['Num_Games']
regular_season_input['PF_Per_game'] = combined_teams['PF'] / combined_teams['Num_Games']

regular_season_input


seed_dict = men_seeds.set_index(['Season', 'TeamID'])
seed_dict


tourney_input = pd.DataFrame()

win_IDs = tourney_compact['WTeamID']
Lose_IDs = tourney_compact['LTeamID']
season = tourney_compact['Season']

winners = pd.DataFrame()
winners[['Season', 'Team1', 'Team2']] = tourney_compact[['Season', 'WTeamID', 'LTeamID']]
winners['Result'] = 1

losers = pd.DataFrame()
losers[['Season', 'Team1', 'Team2']] = tourney_compact[['Season', 'LTeamID', 'WTeamID']]
losers['Result'] = 0


tourney_input = pd.concat([winners, losers])
tourney_input = tourney_input[tourney_input['Season'] == 2024].reset_index(drop = True)
tourney_input


team1_seeds = []
team2_seeds = []

for x in range(len(tourney_input)):
    idx = (tourney_input['Season'][x], tourney_input['Team1'][x])
    seed = seed_dict.loc[idx].values[0]
    if len(seed) == 4:
        seed = int(seed[1:-1])
    else:
        seed = int(seed[1:])
    team1_seeds.append(seed)
    
    idx = (tourney_input['Season'][x], tourney_input['Team2'][x])
    seed = seed_dict.loc[idx].values[0]
    if len(seed) == 4:
        seed = int(seed[1:-1])
    else:
        seed = int(seed[1:])
    team2_seeds.append(seed)

tourney_input['Team1_Seeds'] = team1_seeds
tourney_input['Team2_Seeds'] = team2_seeds
tourney_input


outscores = []

for x in range(len(tourney_input)):
    idx = (tourney_input['Season'][x], tourney_input['Team1'][x])
    team1_score = regular_season_input.loc[idx]
    team1_score['Seed'] = tourney_input['Team1_Seeds'][x]
    
    idx = (tourney_input['Season'][x], tourney_input['Team2'][x])
    team2_score = regular_season_input.loc[idx]
    team2_score['Seed'] = tourney_input['Team2_Seeds'][x]
    
    outscore = team1_score - team2_score
    outscore['Result'] = tourney_input['Result'][x]
    outscores.append(outscore)
    
outscores = pd.DataFrame(outscores)
outscores


correlation = round(outscores.corr(), 2) 
display(np.abs(correlation['Result']))


plt.figure(figsize = (15, 10))
sns.heatmap(correlation)
plt.show()


x = outscores[outscores.columns[:-1]].values
y = outscores['Result'].values

np.random.seed(1)
idx = np.random.permutation(len(x))
train_idx = idx[:int(-.2 * len(x))]
test_idx = idx[int(-.2 * len(x)):]

x_train = x[train_idx]
x_test = x[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

mins = x_train.min(axis = 0)
maxs = x_train.max(axis = 0)

x_train = (x_train - mins) / (maxs - mins)
x_test = (x_test - mins) / (maxs - mins)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


model = RandomForestClassifier(random_state = 1)
model = model.fit(x_train, y_train)
model.score(x_test, y_test)


outscores


outscores['Ratio'] = outscores['Win_Ratio'] + outscores['Points_Ratio']
outscores


matches['Pred'] = (matches['Pred'] + outscores['Ratio']) / model.score(x_test, y_test)
matches.head()


matches.to_csv('submission.csv', index = False)


outscores.to_csv('details.csv', index = False)


submission = pd.read_csv('submission.csv')
submission




