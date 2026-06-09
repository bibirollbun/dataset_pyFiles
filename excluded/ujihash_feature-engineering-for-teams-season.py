import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import pprint as pp

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

from tqdm.notebook import tqdm

pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')


# Run Local or Kaggle
is_kaggle = True


if is_kaggle:
    data_dir = '/kaggle/input/march-machine-learning-mania-2025'
else:
    data_dir = './march-machine-learning-mania-2025'

# Get data file list
data_file_list = os.listdir(data_dir)
print(f'Number of files: {len(data_file_list)}')

# If the file name starts with M or W, remove M and W
data_file_list = [file_name[1:] if file_name[0] in ['M', 'W'] else file_name for file_name in data_file_list]
data_file_list = list(set(data_file_list))
print(f'Number of files types: {len(data_file_list)}')

print('\nfilename list:')
pp.pprint(sorted(data_file_list))


regseason_file = 'RegularSeasonDetailedResults.csv'
seeds_file = 'NCAATourneySeeds.csv'
teams_file = 'Teams.csv'
tounary_compact_file = 'NCAATourneyCompactResults.csv'
tounary_detailed_file = 'NCAATourneyDetailedResults.csv'
submission_file = 'SampleSubmissionStage2.csv'


df_m_regseason = pd.read_csv(os.path.join(data_dir, 'M' + regseason_file))
df_w_regseason = pd.read_csv(os.path.join(data_dir, 'W' + regseason_file))
df_regseason = pd.concat([df_m_regseason, df_w_regseason])

df_m_seeds = pd.read_csv(os.path.join(data_dir, 'M' + seeds_file))
df_w_seeds = pd.read_csv(os.path.join(data_dir, 'W' + seeds_file))
df_seeds = pd.concat([df_m_seeds, df_w_seeds])

df_m_teams = pd.read_csv(os.path.join(data_dir, 'M' + teams_file))
df_w_teams = pd.read_csv(os.path.join(data_dir, 'W' + teams_file))
df_teams = pd.concat([df_m_teams, df_w_teams])

df_m_tounary = pd.read_csv(os.path.join(data_dir, 'M' + tounary_compact_file))
df_w_tounary = pd.read_csv(os.path.join(data_dir, 'W' + tounary_compact_file))
df_tounary = pd.concat([df_m_tounary, df_w_tounary])

# We can read the sex from the TeamId
# Maybe we don't need sex information???

df_subm = pd.read_csv(os.path.join(data_dir, submission_file))


def get_seed_value(
    team_id: int,
    season: int
) -> int:
    """
    Get the seed value of a team in a season.

    Args:
        team_id (int): team id.
        season (int): season.

    Returns:
        int: seed value.
    """
    # Get the seed value
    seed_info = df_seeds[(df_seeds['Season'] == season) &
                         (df_seeds['TeamID'] == team_id)]

    if len(seed_info) == 0:
        return 16

    seed_str = seed_info.iloc[0]['Seed']
    return int(seed_str[1:3])


def culculate_team_stats(
    team_id: int,
    df_season_results: pd.DataFrame
) -> dict:
    """
    Calculate the team stats.
    Because of the difference in the number of games,
    it should be a ratio rather than a value.

    Args:
        team_id (int): team id.
        df_season_results (pd.DataFrame): season results.

    Returns:
        dict: team stats.
    """

    # Get the games that the team won and lost
    df_win_games = df_season_results[df_season_results['WTeamID'] == team_id]
    df_lose_games = df_season_results[df_season_results['LTeamID'] == team_id]

    # Number of games
    num_games = len(df_win_games) + len(df_lose_games)
    if num_games == 0:
        return {}

    # Rate of winning
    win_rate = len(df_win_games) / num_games

    # Points Scored per game
    points_scored = (df_win_games['WScore'].sum(
    ) + df_lose_games['LScore'].sum()) / num_games
    # Points Allowed per game
    points_allowed = (df_win_games['LScore'].sum(
    ) + df_lose_games['WScore'].sum()) / num_games

    # Point Difference per game
    point_diff = points_scored - points_allowed

    # Field Goal Rate
    fg_made = df_win_games['WFGM'].sum() + df_lose_games['LFGM'].sum()
    fg_att = df_win_games['WFGA'].sum() + df_lose_games['LFGA'].sum()
    fg_rate = fg_made / fg_att if fg_att > 0 else 0

    # 3-Point Field Goal Rate
    fg3_made = df_win_games['WFGM3'].sum() + df_lose_games['LFGM3'].sum()
    fg3_att = df_win_games['WFGA3'].sum() + df_lose_games['LFGA3'].sum()
    fg3_rate = fg3_made / fg3_att if fg3_att > 0 else 0

    # Free Throw Rate
    ft_made = df_win_games['WFTM'].sum() + df_lose_games['LFTM'].sum()
    ft_att = df_win_games['WFTA'].sum() + df_lose_games['LFTA'].sum()
    ft_rate = ft_made / ft_att if ft_att > 0 else 0

    # Rebound Rate
    team_rebouns = df_win_games['WOR'].sum(
    ) + df_win_games['WDR'].sum() + df_lose_games['LOR'].sum() + df_lose_games['LDR'].sum()
    opp_rebounds = df_win_games['LOR'].sum(
    ) + df_win_games['LDR'].sum() + df_lose_games['WOR'].sum() + df_lose_games['WDR'].sum()
    total_rebounds = team_rebouns + opp_rebounds
    rebound_rate = team_rebouns / total_rebounds if total_rebounds > 0 else 0

    # ã‚¿ãƒ¼ãƒ³ã‚ªãƒ¼ãƒ�ãƒ¼æ•°ã‚’è¿½è¨˜

    # Assist Rate
    team_assists = df_win_games['WAst'].sum() + df_lose_games['LAst'].sum()
    assist_rate = team_assists / fg_made if fg_made > 0 else 0

    return {
        'num_games': num_games,
        'win_rate': win_rate,
        'points_scored': points_scored,
        'points_allowed': points_allowed,
        'point_diff': point_diff,
        'fg_rate': fg_rate,
        'fg3_rate': fg3_rate,
        'ft_rate': ft_rate,
        'rebound_rate': rebound_rate,
        # 'turnover_rate': turnover_rate,
        'assist_rate': assist_rate
    }


def get_season_team_stats(
    season: int,
) -> pd.DataFrame:
    """
    Get all team stats per season.

    Args:
        season (int): season.

    Returns:
        pd.DataFrame: team stats.
    """
    # Get the team ids for the season
    season_results = df_regseason[df_regseason['Season'] == season].copy()
    team_ids = pd.unique(pd.concat([
        season_results['WTeamID'],
        season_results['LTeamID']
    ]))

    # Get the team stats
    team_stats = pd.DataFrame([
        {'TeamID': team_id, **culculate_team_stats(team_id, season_results)}
        for team_id in team_ids
    ])

    return team_stats


def get_matchup_stats(
    team1_id: int,
    team2_id: int,
    season: int,
    team_stats: pd.DataFrame
) -> pd.DataFrame:
    """
    Get the matchup stats between two teams.

    Args:
        team1_id (int): team 1 id.
        team2_id (int): team 2 id.
        season (int): season.
        team_stats (pd.DataFrame): team stats.

    Returns:
        pd.DataFrame: matchup stats.
    """
    # Get the team stats
    team1_stats = team_stats[team_stats['TeamID'] == team1_id].iloc[0]
    team2_stats = team_stats[team_stats['TeamID'] == team2_id].iloc[0]

    # Get seed value
    team1_seed = get_seed_value(team1_id, season)
    team2_seed = get_seed_value(team2_id, season)

    # Calculate the matchup stats
    matchup_stats = {
        'num_games': team1_stats['num_games'] + team2_stats['num_games'],
        'seed_diff': team1_seed - team2_seed,
        'win_rate_diff': team1_stats['win_rate'] - team2_stats['win_rate'],
        'points_scored_diff': team1_stats['points_scored'] - team2_stats['points_scored'],
        'points_allowed_diff': team1_stats['points_allowed'] - team2_stats['points_allowed'],
        'point_diff_diff': team1_stats['point_diff'] - team2_stats['point_diff'],
        'fg_rate_diff': team1_stats['fg_rate'] - team2_stats['fg_rate'],
        'fg3_rate_diff': team1_stats['fg3_rate'] - team2_stats['fg3_rate'],
        'ft_rate_diff': team1_stats['ft_rate'] - team2_stats['ft_rate'],
        'rebound_rate_diff': team1_stats['rebound_rate'] - team2_stats['rebound_rate'],
        'assist_rate_diff': team1_stats['assist_rate'] - team2_stats['assist_rate']
    }

    return pd.DataFrame([matchup_stats])


def create_train(start_season: int, end_season: int) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Create training data.
    Old data may not be useful, so we use the latest data.


    Args:
        start_season (int): start season.
        end_season (int): end season.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: training data.
    """
    features = []
    labels = []

    for season in range(start_season, end_season):

        # Get the team stats
        team_stats = get_season_team_stats(season)

        # Get the tournament results for the season
        season_results = df_tounary[df_tounary['Season'] == season]

        # Create the features per game
        for _, game in season_results.iterrows():
            # Winning team features
            features_win = get_matchup_stats(
                team1_id=game['WTeamID'],
                team2_id=game['LTeamID'],
                season=season,
                team_stats=team_stats
            )
            features.append(features_win)
            labels.append(1)

            # Losing team features
            features_lose = -features_win
            features.append(features_lose)
            labels.append(0)

    return pd.concat(features), np.array(labels)


X_train, y_train = create_train(2020, 2025)


model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)


def extract_game_info(id_str):
    # Extract year and team_ids
    parts = id_str.split('_')
    season = int(parts[0])
    teamID1 = int(parts[1])
    teamID2 = int(parts[2])
    return season, teamID1, teamID2

df_subm[['Season', 'TeamA', 'TeamB']] = df_subm['ID'].apply(lambda x: extract_game_info(x)).to_list()


preds = []

# Get team stats for season
team_stats = get_season_team_stats(2025)

for _, game in tqdm(df_subm.iterrows(), total=len(df_subm)):

    # Create features
    features = get_matchup_stats(
        team1_id=game['TeamA'],
        team2_id=game['TeamB'],
        season=game['Season'],
        team_stats=team_stats
    )
    
    pred = model.predict_proba(features)[0][1]
    
    preds.append(pred)


df_subm['Pred'] = preds

submission = df_subm[['ID', 'Pred']]
if is_kaggle:
    submission.to_csv('submission.csv', index=False)
    print('Submission file created.')




