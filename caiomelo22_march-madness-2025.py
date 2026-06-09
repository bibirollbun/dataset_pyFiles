# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from datetime import datetime, timedelta
from IPython.display import clear_output
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


start_data_season = {
    'M': 2003,
    'W': 2009
}
classifiers = {
    'M': RandomForestClassifier(max_depth=80, max_features='sqrt', min_samples_leaf=4,
                       n_estimators=1600, random_state=0),
    'W': RandomForestClassifier(max_depth=100, max_features=None, min_samples_leaf=4,
                       min_samples_split=5, n_estimators=2000, random_state=0)
}
columns = [
            'Season', 'TeamA', 'TeamB', 'TeamNameA', 'TeamNameB', 
            'ASeedNum', 'ALastTournPct', 'AELO', 'ARegSznPct',
            'APts', 'APtsOpp', 'AFG', 'AFG3', 'AFT', 'AAst', 'ATO',
            'BSeedNum', 'BLastTournPct', 'BELO', 'BRegSznPct',
            'BPts', 'BPtsOpp', 'BFG', 'BFG3', 'BFT', 'BAst', 'BTO',
            'Winner'
        ]
season = 2025


def get_teams_info(gender):
    teams_file = f'/kaggle/input/march-machine-learning-mania-2025/{gender}Teams.csv'
    teams_df = pd.read_csv(teams_file)
    display(teams_df)
    return teams_df

m_teams_info_df = get_teams_info('M')
w_teams_info_df = get_teams_info('W')


def get_k(vic_margin, elo_diff_winner):
    return 20*((vic_margin+3)**0.8)/(7.5 + 0.006*elo_diff_winner)

def get_e_team(team_elo, opp_team_elo):
    return 1/(1+10**((opp_team_elo - team_elo)/400))
                
def update_elo(elo_dict, winning_team_id, losing_team_id, winning_team_pts, losing_team_pts):
    winning_team_elo = elo_dict[winning_team_id]
    losing_team_elo = elo_dict[losing_team_id]

    vic_margin = winning_team_pts - losing_team_pts
    elo_diff_winner = winning_team_elo - losing_team_elo
    elo_dict[winning_team_id] = round(get_k(vic_margin, elo_diff_winner)*(1 - get_e_team(winning_team_elo, losing_team_elo)) + winning_team_elo, 2)
    elo_dict[losing_team_id] = round(get_k(vic_margin, elo_diff_winner)*(0 - get_e_team(losing_team_elo, winning_team_elo)) + losing_team_elo, 2)

def initialize_teams_elo_dict(teams_df, full_games_df):
    full_games_df["WTeamELO"] = 1500
    full_games_df["LTeamELO"] = 1500
    
    elo_dict = dict()
    for team_id in teams_df["TeamID"]:
        elo_dict[team_id] = 1500

    return elo_dict


def get_seed_number(seed):
    if 'a' in seed or 'b' in seed:
        return 17
    return int(seed[1:])

def generate_numeric_cols(df, team):
    df[f'{team}FG'] = (df[f'{team}FGM'] * 100) / df[f'{team}FGA']
    df[f'{team}FG3'] = (df[f'{team}FGM3'] * 100) / df[f'{team}FGA3']
    df[f'{team}FT'] = (df[f'{team}FTM'] * 100) / df[f'{team}FTA']

def generate_game_id(game):
    return f"{game['Season']}_{game['DayNum']}_{game['WTeamID']}_{game['LTeamID']}"

def get_files_by_competition(gender):
    # Getting teams df
    teams_file = f'/kaggle/input/march-machine-learning-mania-2025/{gender}TeamConferences.csv'
    teams_df = pd.read_csv(teams_file)
    
    # Getting regular season games df
    reg_szn_file = f'/kaggle/input/march-machine-learning-mania-2025/{gender}RegularSeasonDetailedResults.csv'
    reg_szn_df = pd.read_csv(reg_szn_file)
    reg_szn_df["GameId"] = reg_szn_df.apply(generate_game_id, axis=1)
    reg_szn_df["IsTourn"] = 0
    reg_szn_df.set_index("GameId", inplace=True)
    generate_numeric_cols(reg_szn_df, 'W')
    generate_numeric_cols(reg_szn_df, 'L')
    
    # Getting tournament season games df
    tourn_file = f'/kaggle/input/march-machine-learning-mania-2025/{gender}NCAATourneyCompactResults.csv'
    tourn_df = pd.read_csv(tourn_file)
    tourn_df["GameId"] = tourn_df.apply(generate_game_id, axis=1)
    tourn_df["IsTourn"] = 1
    tourn_df.set_index("GameId", inplace=True)
    
    # Getting seeding df
    seed_file = f'/kaggle/input/march-machine-learning-mania-2025/{gender}NCAATourneySeeds.csv'
    seed_df = pd.read_csv(seed_file)
    seed_df['SeedNum'] = seed_df.Seed.apply(lambda x: get_seed_number(x))

    game_defining_cols = ["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc", "NumOT", "IsTourn"]
    full_games_df = pd.concat([df[game_defining_cols] for df in [reg_szn_df, tourn_df]]).sort_values(by=['Season', "DayNum"])

    teams_elo = initialize_teams_elo_dict(teams_df, full_games_df)

    for index, row in tqdm(full_games_df.iterrows(), total=len(full_games_df)):
        if row["IsTourn"]:
            tourn_df.at[index, "WTeamELO"] = teams_elo[row["WTeamID"]]
            tourn_df.at[index, "LTeamELO"] = teams_elo[row["LTeamID"]]
        else:
            reg_szn_df.at[index, "WTeamELO"] = teams_elo[row["WTeamID"]]
            reg_szn_df.at[index, "LTeamELO"] = teams_elo[row["LTeamID"]]

        update_elo(
            teams_elo,
            row["WTeamID"],
            row["LTeamID"],
            row["WScore"],
            row["LScore"],
        )

    display(full_games_df)
    
    return teams_df, reg_szn_df, tourn_df, seed_df, teams_elo


def get_reg_szn_stats(season, team, reg_szn_df):
    w_games = reg_szn_df[reg_szn_df['WTeamID'] == team].rename(columns = {
        'WFG': 'FG', 'WFG3': 'FG3', 'WFT': 'FT', 'WAst': 'Ast', 'WTO': 'TO', 'WOR': 'OR', 'WDR': 'DR', 'WStl': 'Stl', 'WBlk': 'Blk', 'WPF': 'PF',
        'WScore': 'Score', 'WTeamELO': 'ELO',
        
        'LFG': 'OppFG', 'LFG3': 'OppFG3', 'LFT': 'OppFT', 'LAst': 'OppAst', 'LTO': 'OppTO', 'LOR': 'OppOR', 'LDR': 'OppDR', 'LStl': 'OppStl', 'LBlk': 'OppBlk', 'LPF': 'OppPF',
        'LScore': 'OppScore', 'LTeamELO': 'OppELO',
        })
    w_games['Won'] = 1
    
    l_games = reg_szn_df[reg_szn_df['LTeamID'] == team].rename(columns = {
        'LFG': 'FG', 'LFG3': 'FG3', 'LFT': 'FT', 'LAst': 'Ast', 'LTO': 'TO', 'LOR': 'OR', 'LDR': 'DR', 'LStl': 'Stl', 'LBlk': 'Blk', 'LPF': 'PF',
        'LScore': 'Score', 'LTeamELO': 'ELO',
        
        'WFG': 'OppFG', 'WFG3': 'OppFG3', 'WFT': 'OppFT', 'WAst': 'OppAst', 'WTO': 'OppTO', 'WOR': 'OppOR', 'WDR': 'OppDR', 'WStl': 'OppStl', 'WBlk': 'OppBlk', 'WPF': 'OppPF',
        'WScore': 'OppScore', 'WTeamELO': 'OppELO',
        })
    l_games['Won'] = 0
    
    games = pd.concat([w_games, l_games], axis=0, ignore_index=True)
    
    reg_szn_pct = (len(w_games) * 100) / len(games)
    
    return [   
               reg_szn_pct, 
               games['Score'].mean(), games['OppScore'].mean(), games['FG'].mean(), games['FG3'].mean(), games['FT'].mean(), games['Ast'].mean(), games['TO'].mean(),
           ]


def get_last_tourn_pct(season, team, tourn_df):
    wins = len(tourn_df[(tourn_df['Season'] == season-1) & (tourn_df['WTeamID'] == team)])
    total_games = len(tourn_df[(tourn_df['Season'] == season-1) & ((tourn_df['WTeamID'] == team) | (tourn_df['LTeamID'] == team))])
    
    if not total_games: return 0
    return (wins * 100) / total_games


def set_team_name(team_id, teams_df):
    return teams_df[teams_df['TeamID'] == team_id].reset_index().loc[0, 'TeamName']

def get_game_stats(gender, season, team_a, team_b, team_a_elo, team_b_elo, reg_szn_df, seed_df, tourn_df):
    a_reg_szn_stats = get_reg_szn_stats(season, team_a, reg_szn_df)
    b_reg_szn_stats = get_reg_szn_stats(season, team_b, reg_szn_df)

    try:
        a_seed = seed_df[(seed_df['Season'] == season) & (seed_df['TeamID'] == team_a)].reset_index().loc[0, 'SeedNum']
    except:
        a_seed = 24
    try:
        b_seed = seed_df[(seed_df['Season'] == season) & (seed_df['TeamID'] == team_b)].reset_index().loc[0, 'SeedNum']
    except:
        b_seed = 24

    a_last_tourn_pct = get_last_tourn_pct(season, team_a, tourn_df)
    b_last_tourn_pct = get_last_tourn_pct(season, team_b, tourn_df)

    stats_a = [a_seed, a_last_tourn_pct, team_a_elo] + (a_reg_szn_stats)
    stats_b = [b_seed, b_last_tourn_pct, team_b_elo] + (b_reg_szn_stats)
    
    teams_info_df = m_teams_info_df if gender == 'M' else w_teams_info_df
    team_a_name = set_team_name(team_a, teams_info_df)
    team_b_name = set_team_name(team_b, teams_info_df)
    
    return [season, team_a, team_b, team_a_name, team_b_name] + stats_a + stats_b

def build_dataset(gender, start_data_season, reg_szn_df, tourn_df, seed_df, end_season):
    data = []

    for season in range(start_data_season + 1, end_season + 1):
        tourney_games = tourn_df[tourn_df['Season'] == season].reset_index(drop=True)
        for idx, g in tourney_games.iterrows():

            clear_output(wait=True)
            print(f"{season}: {idx}/{len(tourney_games)}")

            team_a = min([g['WTeamID'], g['LTeamID']])
            team_b = max([g['WTeamID'], g['LTeamID']])

            if team_a == g['WTeamID']:
                winner = 'A'
                
                team_a_score = g['WScore']
                team_b_score = g['LScore']

                team_a_elo = g["WTeamELO"]
                team_b_elo = g["LTeamELO"]
            else:
                winner = 'B'
                
                team_a_score = g['LScore']
                team_b_score = g['WScore']

                team_a_elo = g["LTeamELO"]
                team_b_elo = g["WTeamELO"]

            print(f"{team_a} x {team_b}")

            game_stats = get_game_stats(gender, season, team_a, team_b, team_a_elo, team_b_elo, reg_szn_df, seed_df, tourn_df)

            data.append(game_stats + [winner])

    data_df = pd.DataFrame(data, columns=columns)
    display(data_df)
    
    return data_df


def train_model(x_train, y_train, gender):
    classifier = classifiers[gender]
    classifier.fit(x_train, y_train)
    return classifier

def predict_seasons(gender, start_data_season, data_df, detailed_results):
    acc_sum = 0
    seasons_count = 0

    for season_itr in range(start_data_season + 2, season + 1):
        data_train = data_df[(data_df['Season'] < season_itr)].reset_index(drop=True)
        data_test = data_df[(data_df['Season'] == season_itr)].reset_index(drop=True)

        if not len(data_test):
            continue

        x_train = data_train.drop(['TeamNameA', 'TeamNameB', 'Winner'], axis=1)
        y_train = data_train.loc[:, 'Winner']
            
        classifier = train_model(x_train, y_train, gender)

        x_test = data_test.drop(['TeamNameA', 'TeamNameB', 'Winner'], axis=1)
        y_test = data_test.loc[:, 'Winner']

        predictions = classifier.predict(x_test)
        
        acc = accuracy_score(y_test, predictions)
        acc_sum += acc
        seasons_count += 1
        
        if detailed_results:
            print(f'\nResults for {gender} season {season_itr}:')
            print('Accuracy predictions:', acc)

            cm = confusion_matrix(y_test, predictions)
            print('Confusion matrix:')
            cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
            cm_disp.plot()
            plt.show()

    print(f'\n\n{gender} Mean Accuracy: {acc_sum/seasons_count}')
    
    return classifier


def get_current_predictions(gender, classifier, reg_szn_df, tourn_df, seed_df, team_elo_dict):
    data = []
    teams = seed_df[seed_df['Season'] == season].TeamID.unique()
    teams.sort()
    
    for idx, team_a in enumerate(teams):
        for idx_b in range(idx + 1, len(teams)):
            team_b = teams[idx_b]
            
            clear_output(wait=True)
            print(f"{idx}/{len(teams)}")
            print(f"{team_a} x {team_b}")

            team_a_elo = team_elo_dict[team_a]
            team_b_elo = team_elo_dict[team_b]
            
            game_stats = get_game_stats(gender, season, team_a, team_b, team_a_elo, team_b_elo, reg_szn_df, seed_df, tourn_df)
            data.append(game_stats)
            
    data_df = pd.DataFrame(data, columns=columns[:-1])
    data_to_predict = data_df.drop(['TeamNameA', 'TeamNameB'], axis=1)
    
    probs = classifier.predict_proba(data_to_predict)
    pred = classifier.predict(data_to_predict)

    data_df['AProb'] = probs[:, 0]
    data_df['BProb'] = probs[:, 1]
    data_df['Pred'] = pred
    
    return data_df


def pipeline(gender, detailed_results=False):
    teams_df, reg_szn_df, tourn_df, seed_df, team_elo_dict = get_files_by_competition(gender)
    data_df = build_dataset(gender, start_data_season[gender], reg_szn_df, tourn_df, seed_df, season)
    classifier = predict_seasons(gender, start_data_season[gender], data_df, detailed_results)
    current_szn_df = get_current_predictions(gender, classifier, reg_szn_df, tourn_df, seed_df, team_elo_dict)
    display(current_szn_df)
    return current_szn_df, teams_df, team_elo_dict


m_current_szn_df, m_teams_df, m_team_elo_dict = pipeline('M')


w_current_szn_df, w_teams_df, w_team_elo_dict = pipeline('W')


def set_game_id(row):
    return f"{row['Season']}_{row['TeamA']}_{row['TeamB']}"

def set_impossible_games_probs(gender, current_szn_df, teams_df):
    teams_a = current_szn_df['TeamA'].unique().tolist()
    teams_b = current_szn_df['TeamB'].unique().tolist()
    tourney_teams = set(teams_a + teams_b)

    teams = teams_df[teams_df['Season'] == season]['TeamID'].unique().tolist()

    data = []

    for idx, team_a in enumerate(teams):
        for idx_b in range(idx + 1, len(teams)):
            team_b = teams[idx_b]

            if team_a in tourney_teams and team_b in tourney_teams:
                continue
                
            teams_info_df = m_teams_info_df if gender == 'M' else w_teams_info_df
            # There is no reason for setting team names or probabilities if these matchups will never happen in the tournament
            team_a_name = None # set_team_name(team_a, teams_info_df)
            team_b_name = None # set_team_name(team_b, teams_info_df)

            game_id = f"{season}_{team_a}_{team_b}"
            data.append([game_id, team_a, team_b, team_a_name, team_b_name, 0.5])

    data_df = pd.DataFrame(data, columns=['ID', 'TeamA', 'TeamB', 'TeamNameA', 'TeamNameB', 'Pred'])
    return data_df

current_szn_df = pd.concat([m_current_szn_df, w_current_szn_df], axis=0).reset_index(drop=True)
current_szn_df['ID'] = current_szn_df.apply(lambda row: set_game_id(row), axis=1)

m_impossible_df = set_impossible_games_probs('M', m_current_szn_df, m_teams_df)
w_impossible_df = set_impossible_games_probs('W', w_current_szn_df, w_teams_df)

final_df = current_szn_df.loc[:, ['ID', 'TeamA', 'TeamB', 'TeamNameA', 'TeamNameB', 'AProb']].rename({'AProb': 'Pred'}, axis=1)
final_df = pd.concat([final_df, m_impossible_df, w_impossible_df], axis=0).sort_values(by='ID').reset_index(drop=True)

display(final_df)


team_a = 'Baylor'
team_b = 'Duke'

# seed_file = f'/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv'
# seed_df = pd.read_csv(seed_file)
# seed_df['SeedNum'] = seed_df.Seed.apply(lambda x: get_seed_number(x)) # 1297 1224
# display(seed_df[(seed_df['SeedNum'] == 17) & (seed_df['Season'] == season)])

game = final_df[((final_df['TeamNameA'].str.contains(team_a)) & (final_df['TeamNameB'].str.contains(team_b))) |
               ((final_df['TeamNameB'].str.contains(team_a)) & (final_df['TeamNameA'].str.contains(team_b)))]

display(game)


def create_submission_file(df):
    df_filtered = df.loc[:, ['ID', 'Pred']]
    df_filtered.to_csv('submission.csv', index=False)
    
create_submission_file(final_df)


def get_rf_best_parameters(gender):
    teams_df, reg_szn_df, tourn_df, seed_df = get_files_by_competition(gender)

    data_df = build_dataset(gender, start_data_season[gender], reg_szn_df, tourn_df, seed_df, season) 
    x = data_df.drop(['TeamNameA', 'TeamNameB', 'Winner'], axis=1)
    y = data_df.loc[:, 'Winner']

    from sklearn.model_selection import RandomizedSearchCV
    rs_optimizer = RandomForestClassifier()

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['sqrt', 'log2', None]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

    rf_random = RandomizedSearchCV(estimator = rs_optimizer, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)

    rf_random.fit(x, y)

    best_random = rf_random.best_estimator_
    best_parameters = rf_random.cv_results_
    #     print(best_parameters)
    print(best_random)
    
# get_rf_best_parameters('M')

