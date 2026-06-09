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


import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#ML
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import brier_score_loss, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

#Regulazation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

# DL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

#Tool
from tqdm import tqdm
import itertools
import logging
import psutil # To monitoring  CPU, Memory, Disk, Network 
import joblib # To save/load pkl 
import gc # to Memery Recovery


logging.basicConfig(level=logging.INFO)

# --- 1. Load & Tag Function ---
def load_and_tag(filepath, source):
    df = pd.read_csv(filepath)
    df['Source'] = source
    return df

# --- 2. Load Data Function  ---
def load_data(gender_prefix, data_path):
    detailed_season = pd.read_csv(f'{data_path}/{gender_prefix}RegularSeasonDetailedResults.csv')
    compact_season = pd.read_csv(f'{data_path}/{gender_prefix}RegularSeasonCompactResults.csv')
    detailed_tourney = pd.read_csv(f'{data_path}/{gender_prefix}NCAATourneyDetailedResults.csv')
    compact_tourney = pd.read_csv(f'{data_path}/{gender_prefix}NCAATourneyCompactResults.csv')
    compact_secondary = pd.read_csv(f'{data_path}/{gender_prefix}SecondaryTourneyCompactResults.csv')
    teams = pd.read_csv(f'{data_path}/{gender_prefix}Teams.csv')
    conferences = pd.read_csv(f'{data_path}/{gender_prefix}TeamConferences.csv')
    game_cities = pd.read_csv(f'{data_path}/{gender_prefix}GameCities.csv')
    cities = pd.read_csv(f'{data_path}/Cities.csv')
    seeds = pd.read_csv(f'{data_path}/{gender_prefix}NCAATourneySeeds.csv')


    return detailed_season, compact_season, detailed_tourney, compact_tourney, compact_secondary,teams, conferences, game_cities, cities, seeds



def process_team_stats(detailed_df, compact_df):
    # Winner Stats
    win_stats = detailed_df[[
        'Season', 'WTeamID', 'WScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA',
        'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LTeamID', 'LScore'
    ]].copy()
    win_stats.rename(columns={
        'WTeamID': 'TeamID', 'WScore': 'Score', 'WFGM': 'FGM', 'WFGA': 'FGA', 'WFGM3': 'FGM3',
        'WFGA3': 'FGA3', 'WFTM': 'FTM', 'WFTA': 'FTA', 'WOR': 'OR', 'WDR': 'DR',
        'WAst': 'Ast', 'WTO': 'TO', 'WStl': 'Stl', 'WBlk': 'Blk', 'WPF': 'PF',
        'LTeamID': 'OppTeamID', 'LScore': 'OppScore'
    }, inplace=True)

    # Loser Stats
    lose_stats = detailed_df[[
        'Season', 'LTeamID', 'LScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
        'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'WTeamID', 'WScore'
    ]].copy()
    lose_stats.rename(columns={
        'LTeamID': 'TeamID', 'LScore': 'Score', 'LFGM': 'FGM', 'LFGA': 'FGA', 'LFGM3': 'FGM3',
        'LFGA3': 'FGA3', 'LFTM': 'FTM', 'LFTA': 'FTA', 'LOR': 'OR', 'LDR': 'DR',
        'LAst': 'Ast', 'LTO': 'TO', 'LStl': 'Stl', 'LBlk': 'Blk', 'LPF': 'PF',
        'WTeamID': 'OppTeamID', 'WScore': 'OppScore'
    }, inplace=True)

    # Combine
    all_stats = pd.concat([win_stats, lose_stats], ignore_index=True)

    # íŒŒìƒ� ìŠ¤íƒ¯
    all_stats['FG%'] = all_stats['FGM'] / all_stats['FGA']
    all_stats['3P%'] = all_stats['FGM3'] / all_stats['FGA3']
    all_stats['FT%'] = all_stats['FTM'] / all_stats['FTA']
    all_stats['ORB%'] = all_stats['OR'] / (all_stats['OR'] + all_stats['DR'])
    all_stats['TOV%'] = all_stats['TO'] / (all_stats['FGA'] + 0.44 * all_stats['FTA'] + all_stats['TO'])
    all_stats['FT_Rate'] = all_stats['FTM'] / all_stats['FGA']

    # Possessions ê³„ì‚°
    all_stats['Possessions'] = all_stats['FGA'] + 0.44 * all_stats['FTA'] - all_stats['OR'] + all_stats['TO']

    # OffEff & DefEff
    all_stats['OffEff'] = (all_stats['Score'] / all_stats['Possessions']) * 100
    all_stats['DefEff'] = (all_stats['OppScore'] / all_stats['Possessions']) * 100

    # === 1ï¸�âƒ£ Averg. season by team 
    team_avg_stats = all_stats.groupby(['Season', 'TeamID']).mean().reset_index()

    # === 2ï¸�âƒ£ WinRate ì¶”ê°€ ---
    win_df = compact_df[['Season', 'WTeamID']].copy()
    win_df['Wins'] = 1
    lose_df = compact_df[['Season', 'LTeamID']].copy()
    lose_df['Wins'] = 0
    win_df.rename(columns={'WTeamID': 'TeamID'}, inplace=True)
    lose_df.rename(columns={'LTeamID': 'TeamID'}, inplace=True)
    winlose_df = pd.concat([win_df, lose_df], ignore_index=True)
    winrate = winlose_df.groupby(['Season', 'TeamID'])['Wins'].mean().reset_index().rename(columns={'Wins': 'WinRate'})

    team_avg_stats = team_avg_stats.merge(winrate, on=['Season', 'TeamID'], how='left')

    # === 3ï¸�âƒ£ NaN ì±„ìš°ê¸° ===
    numeric_cols = ['OffEff', 'DefEff', 'WinRate']
    for col in numeric_cols:
        team_avg_stats[col] = team_avg_stats.groupby('Season')[col].transform(lambda x: x.fillna(x.mean()))

    # === 4ï¸�âƒ£ í•„ìš”ì—†ëŠ” ì»¬ëŸ¼ ì œê±° ===
    drop_cols = ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'Possessions', 'Score', 'OppScore', 'FG%', '3P%', 'FT%', 'ORB%', 'TOV%', 'FT_Rate']
    team_avg_stats = team_avg_stats.drop(columns=[col for col in drop_cols if col in team_avg_stats.columns])

    return team_avg_stats



# --- 4. Prepare Compact ---
def prepare_compact(df, is_winner=True):
    df_prepared = df.copy()
    
    if is_winner:
        df_prepared['TeamID'] = df_prepared['WTeamID']
        df_prepared['Score'] = df_prepared['WScore']
        df_prepared['Result'] = 1  # Win
    else:
        df_prepared['TeamID'] = df_prepared['LTeamID']
        df_prepared['Score'] = df_prepared['LScore']
        df_prepared['Result'] = 0  # lose
    
    return df_prepared


def attach_state_info(compact_df, game_cities_df, cities_df):
    # 1. ê²Œì�„ ì •ë³´ì—� CityID ë³‘í•©
    games_with_city = compact_df.merge(
        game_cities_df[['Season', 'DayNum', 'CityID']].rename(columns={'CityID': 'GameCityID'}),
        on=['Season', 'DayNum'],
        how='left'
    )

    # 2. ë�„ì‹œëª…ê³¼ ì£¼(State) ì •ë³´ ë³‘í•©
    games_with_location = games_with_city.merge(
        cities_df[['CityID', 'City', 'State']].rename(columns={
            'CityID': 'GameCityID',
            'City': 'GameCity',
            'State': 'GameState'
        }),
        on='GameCityID',
        how='left'
    )
    
    # 3ï¸� ìµœì¢… ì»¬ëŸ¼ ì •ë¦¬
    final_cols = compact_df.columns.tolist() + ['GameCityID', 'GameCity', 'GameState']
    final_cols = list(dict.fromkeys(final_cols))  # ì¤‘ë³µ ì œê±°

    return games_with_location[final_cols]




# --- 9. Process H2H ---
def process_h2h(compact_df):
    h2h_records = []

    for season in compact_df['Season'].unique():
        season_df = compact_df[compact_df['Season'] == season]
        h2h_dict = {}

        for idx, row in season_df.iterrows():
            team1, team2 = row['WTeamID'], row['LTeamID']
            score_diff = row['WScore'] - row['LScore']

            # Team1 wins
            h2h_dict.setdefault((team1, team2), {'games': 0, 'wins': 0, 'score_sum': 0})
            h2h_dict[(team1, team2)]['games'] += 1
            h2h_dict[(team1, team2)]['wins'] += 1
            h2h_dict[(team1, team2)]['score_sum'] += score_diff

            # Team2 loses
            h2h_dict.setdefault((team2, team1), {'games': 0, 'wins': 0, 'score_sum': 0})
            h2h_dict[(team2, team1)]['games'] += 1
            h2h_dict[(team2, team1)]['score_sum'] -= score_diff  # Negative

        for (t1, t2), record in h2h_dict.items():
            win_rate = record['wins'] / record['games'] if record['games'] > 0 else 0.5
            avg_score_diff = record['score_sum'] / record['games'] if record['games'] > 0 else 0

            h2h_records.append({
                'Season': season,
                'Team1': t1,
                'Team2': t2,
                'TotalGames': record['games'],
                'Wins': record['wins'],
                'WinRate': win_rate,
                'AvgScoreDiff': avg_score_diff
            })

    return pd.DataFrame(h2h_records)




def integrate_seed_data(game_df, seeds_df):
    """
    Integrate seed data into the game-level dataset and calculate seed-based metrics.

    Parameters:
    - game_df: DataFrame containing game-level data with 'Season', 'Team1ID', and 'Team2ID'.
    - seeds_df: DataFrame containing seed information with 'Season', 'TeamID', and 'Seed'.

    Returns:
    - DataFrame with seed information and calculated metrics.
    """
    seeds_df = seeds_df.copy()

    # 1. Seed ìˆ«ì�� ë¶€ë¶„ë§Œ ì¶”ì¶œ (ì˜ˆ: W01 -> 1)
    seeds_df['SeedRank'] = seeds_df['Seed'].str.extract(r'(\d+)').astype(int)

    # 2. Team1 Seed
    game_df = game_df.merge(seeds_df[['Season', 'TeamID', 'SeedRank']].rename(columns={'TeamID': 'Team1ID', 'SeedRank': 'T1_SeedRank'}),
                            on=['Season', 'Team1ID'], how='left')

    # 3. Team2 Seed
    game_df = game_df.merge(seeds_df[['Season', 'TeamID', 'SeedRank']].rename(columns={'TeamID': 'Team2ID', 'SeedRank': 'T2_SeedRank'}),
                            on=['Season', 'Team2ID'], how='left')

    # 4. Seed ì°¨ì�´
    game_df['SeedDiff'] = game_df['T1_SeedRank'] - game_df['T2_SeedRank']

    # 5. Normalized Seed (1/Seed)
    game_df['T1_SeedNorm'] = 1 / game_df['T1_SeedRank']
    game_df['T2_SeedNorm'] = 1 / game_df['T2_SeedRank']

    return game_df



def process_conference_stats(df_compact_team, tourney_teams):
    """
    Calculate Conference-level stats: win rate & tourney participation rate only.
    """

    # --- 1. Win rate ---
    team_wins = (
        df_compact_team[df_compact_team['Result'] == 1]
        .groupby(['Season', 'ConfAbbrev'], as_index=False)
        .size()
        .rename(columns={'size': 'Wins'})
    )
    team_games = (
        df_compact_team
        .groupby(['Season', 'ConfAbbrev'], as_index=False)
        .size()
        .rename(columns={'size': 'TotalGames'})
    )

    conf_stats = pd.merge(team_games, team_wins, on=['Season', 'ConfAbbrev'], how='left')
    conf_stats['Wins'] = conf_stats['Wins'].fillna(0).astype(int)
    conf_stats['Conf_WinRate'] = conf_stats['Wins'] / conf_stats['TotalGames']

    # --- 2. Tourney ì§„ì¶œ ë¹„ìœ¨ ---
    tourney_teams_unique = tourney_teams[['Season', 'TeamID']].drop_duplicates()
    tourney_conf = pd.merge(
        tourney_teams_unique,
        df_compact_team[['Season', 'TeamID', 'ConfAbbrev']].drop_duplicates(),
        on=['Season', 'TeamID'],
        how='left'
    )

    tourney_counts = (
        tourney_conf
        .groupby(['Season', 'ConfAbbrev'], as_index=False)
        .size()
        .rename(columns={'size': 'TourneyTeams'})
    )

    conf_stats = pd.merge(conf_stats, tourney_counts, on=['Season', 'ConfAbbrev'], how='left')
    conf_stats['TourneyTeams'] = conf_stats['TourneyTeams'].fillna(0).astype(int)
    conf_stats['Conf_TourneyRate'] = conf_stats['TourneyTeams'] / conf_stats['TotalGames']

    # --- ë¶ˆí•„ìš”í•œ ì»¬ëŸ¼ ì œê±° ---
    conf_stats_final = conf_stats[['Season', 'ConfAbbrev', 'Conf_WinRate', 'Conf_TourneyRate']]

    return conf_stats_final



def shuffle_game_level(df):
    df = df.copy()
    swap_mask = np.random.rand(len(df)) < 0.5

    # íŒ€ ID ìŠ¤ì™‘
    df['Team1ID'], df['Team2ID'] = (
        np.where(swap_mask, df['Team2ID'], df['Team1ID']),
        np.where(swap_mask, df['Team1ID'], df['Team2ID'])
    )
    
    # Seed ìŠ¤ì™‘
    df['T1_SeedRank'], df['T2_SeedRank'] = (
        np.where(swap_mask, df['T2_SeedRank'], df['T1_SeedRank']),
        np.where(swap_mask, df['T1_SeedRank'], df['T2_SeedRank'])
    )

    # íŒ€ ìŠ¤íƒ¯ ìŠ¤ì™‘
    cols_to_swap = ['T1_OffEff', 'T1_DefEff', 'T1_WinRate', 'T1_Conf_WinRate', 'T1_Conf_TourneyRate']
    for col in cols_to_swap:
        opp_col = col.replace('T1_', 'T2_')
        df[col], df[opp_col] = (
            np.where(swap_mask, df[opp_col], df[col]),
            np.where(swap_mask, df[col], df[opp_col])
        )

    # Result ìŠ¤ì™‘
    df['Result'] = np.where(swap_mask, 1 - df['Result'], df['Result'])

    # â­�ï¸� Seed ê´€ë ¨ NaN â†’ 20ìœ¼ë¡œ ì²˜ë¦¬ (ì�´ ë¶€ë¶„ ìˆ˜ì •)
    df['T1_SeedRank'] = df['T1_SeedRank'].fillna(20)
    df['T2_SeedRank'] = df['T2_SeedRank'].fillna(20)
    df['SeedDiff'] = df['T1_SeedRank'] - df['T2_SeedRank']



    return df




def create_game_team_level(compact_df, team_stats_df, h2h_df, conf_df=None, conf_stats_df=None, seeds=None):
    df = compact_df.copy()

    # --- Team IDs ---
    if 'WTeamID' in df.columns and 'LTeamID' in df.columns:
        df['Team1ID'] = df['WTeamID']
        df['Team2ID'] = df['LTeamID']
    elif 'TeamID_1' in df.columns and 'TeamID_2' in df.columns:
        df['Team1ID'] = df['TeamID_1']
        df['Team2ID'] = df['TeamID_2']
    elif 'Team1ID' in df.columns and 'Team2ID' in df.columns:  # Stage ë�°ì�´í„°ìš©
        df['Team1ID'] = df['Team1ID']
        df['Team2ID'] = df['Team2ID']
    else:
        raise KeyError(f"Team ID columns not found in compact_df. Found columns: {df.columns.tolist()}")

    # --- Team stats ---
    if team_stats_df is not None:
        t1_stats = team_stats_df.rename(columns=lambda x: f'T1_{x}' if x not in ['Season', 'TeamID'] else x)
        df = df.merge(t1_stats, left_on=['Season', 'Team1ID'], right_on=['Season', 'TeamID'], how='left').drop(columns=['TeamID'])
        t2_stats = team_stats_df.rename(columns=lambda x: f'T2_{x}' if x not in ['Season', 'TeamID'] else x)
        df = df.merge(t2_stats, left_on=['Season', 'Team2ID'], right_on=['Season', 'TeamID'], how='left').drop(columns=['TeamID'])

        # --- â­�ï¸� ê²°ì¸¡ì¹˜ ì²˜ë¦¬ (ì‹œì¦Œ í�‰ê· ìœ¼ë¡œ OffEff, DefEff, WinRate ì±„ìš°ê¸°) ---
        fill_cols = ['T1_OffEff', 'T1_DefEff', 'T1_WinRate', 'T2_OffEff', 'T2_DefEff', 'T2_WinRate']
        for col in fill_cols:
            # ì‹œì¦Œ í�‰ê· 
            df[col] = df.groupby('Season')[col].transform(lambda x: x.fillna(x.mean()))
            # ì‹œì¦Œ í�‰ê· ìœ¼ë¡œë�„ ì•ˆ ì±„ì›Œì¡Œìœ¼ë©´ â†’ ì „ì²´ í�‰ê· ìœ¼ë¡œ
            df[col] = df[col].fillna(df[col].mean())

    # --- H2H Stats ---
    df = df.merge(
        h2h_df[['Season', 'Team1', 'Team2', 'TotalGames', 'WinRate', 'AvgScoreDiff']].rename(columns={
            'Team1': 'Team1ID', 'Team2': 'Team2ID', 'AvgScoreDiff': 'H2H_AvgScoreDiff'
        }),
        on=['Season', 'Team1ID', 'Team2ID'], how='left'
    ).fillna({'TotalGames': 0, 'WinRate': 0.5, 'H2H_AvgScoreDiff': 0})
    df.rename(columns={'TotalGames': 'H2H_Games', 'WinRate': 'H2H_WinRate'}, inplace=True)

    # --- Conference Info ---
    if conf_df is not None:
        conf_info = conf_df[['TeamID', 'ConfAbbrev']].drop_duplicates('TeamID')
        df = df.merge(conf_info.rename(columns={'TeamID': 'Team1ID', 'ConfAbbrev': 'T1_ConfAbbrev'}), on='Team1ID', how='left')
        df = df.merge(conf_info.rename(columns={'TeamID': 'Team2ID', 'ConfAbbrev': 'T2_ConfAbbrev'}), on='Team2ID', how='left')
    if conf_stats_df is not None:
        conf_stats_cols = ['Season', 'ConfAbbrev', 'Conf_WinRate', 'Conf_TourneyRate']
        t1_conf_stats = conf_stats_df[conf_stats_cols].rename(columns={
            'ConfAbbrev': 'T1_ConfAbbrev',
            'Conf_WinRate': 'T1_Conf_WinRate',
            'Conf_TourneyRate': 'T1_Conf_TourneyRate'
        })
        df = df.merge(t1_conf_stats, on=['Season', 'T1_ConfAbbrev'], how='left')
        t2_conf_stats = conf_stats_df[conf_stats_cols].rename(columns={
            'ConfAbbrev': 'T2_ConfAbbrev',
            'Conf_WinRate': 'T2_Conf_WinRate',
            'Conf_TourneyRate': 'T2_Conf_TourneyRate'
        })
        df = df.merge(t2_conf_stats, on=['Season', 'T2_ConfAbbrev'], how='left')

    # --- Seed Info ---
    if seeds is not None:
        seeds_df = seeds[['Season', 'TeamID', 'Seed']].copy()
        seeds_df['SeedRank'] = seeds_df['Seed'].str.extract(r'(\d+)').astype(float)
        t1_seeds = seeds_df.rename(columns={'TeamID': 'Team1ID', 'SeedRank': 'T1_SeedRank'})
        df = df.merge(t1_seeds[['Season', 'Team1ID', 'T1_SeedRank']], on=['Season', 'Team1ID'], how='left')
        t2_seeds = seeds_df.rename(columns={'TeamID': 'Team2ID', 'SeedRank': 'T2_SeedRank'})
        df = df.merge(t2_seeds[['Season', 'Team2ID', 'T2_SeedRank']], on=['Season', 'Team2ID'], how='left')
        
        # â­�ï¸� Seed ì—†ëŠ” íŒ€ì�€ 20ìœ¼ë¡œ ì²˜ë¦¬!
        df['T1_SeedRank'] = df['T1_SeedRank'].fillna(20)
        df['T2_SeedRank'] = df['T2_SeedRank'].fillna(20)
        df['SeedDiff'] = df['T1_SeedRank'] - df['T2_SeedRank']

    # --- Source Column ---
    df['Source'] = 'CompactResults'

    # --- Result Column ---
    df['Result'] = 1  # í•­ìƒ� Team1 (Winner) ê¸°ì¤€ 1

    # --- Final Columns ---
    final_cols = [
        'Season', 'Team1ID', 'Team2ID', 'GameCity', 'GameState',
        'H2H_Games', 'H2H_WinRate', 'H2H_AvgScoreDiff', 'Source',
        'T1_OffEff', 'T1_DefEff', 'T1_WinRate', 'T1_Conf_WinRate', 'T1_Conf_TourneyRate',
        'T2_OffEff', 'T2_DefEff', 'T2_WinRate', 'T2_Conf_WinRate', 'T2_Conf_TourneyRate',
        'T1_SeedRank', 'T2_SeedRank', 'SeedDiff', 'Result'
    ]

    existing_cols = [col for col in final_cols if col in df.columns]
    missing_cols = [col for col in final_cols if col not in df.columns]
    if missing_cols:
        print(f"âš ï¸� ëˆ„ë�½ë�œ ì»¬ëŸ¼ë“¤: {missing_cols}")

    return df[existing_cols].copy()



# --- Compact ë�°ì�´í„° ì •ë¦¬ ---
def clean_compact_df(df):
    drop_cols = ['NumOT']
    existing_cols = [col for col in drop_cols if col in df.columns]
    return df.drop(columns=existing_cols).copy()

# --- íŒ€ ì •ë³´ ë°� Conference ì •ë³´ ë³‘í•© ---
def add_team_info(df, df_teams, df_conferences):
    df_result = df.copy()

    # Conference ì •ë³´ ë³‘í•©
    if 'ConfAbbrev' in df_conferences.columns:
        df_result = df_result.merge(
            df_conferences[['Season', 'TeamID', 'ConfAbbrev']],
            on=['Season', 'TeamID'], how='left'
        )
    else:
        print("âš ï¸� Warning: 'ConfAbbrev' not found in conferences data.")

    # íŒ€ ì�´ë¦„ ë°� ì‹ ê·œíŒ€ ì—¬ë¶€ ë³‘í•©
    if 'FirstD1Season' in df_teams.columns:
        df_result = df_result.merge(
            df_teams[['TeamID', 'TeamName', 'FirstD1Season']],
            on='TeamID', how='left'
        )
        df_result['IsNewTeam'] = df_result['Season'] <= df_result['FirstD1Season']
        df_result['IsNewTeam'] = df_result['IsNewTeam'].astype(int)
    else:
        df_result = df_result.merge(
            df_teams[['TeamID', 'TeamName']],
            on='TeamID', how='left'
        )
        df_result['IsNewTeam'] = 0  # Default ê°’ ëª…ì‹œì �ìœ¼ë¡œ ì„¤ì • (ê¸°ì¡´íŒ€)

    return df_result


def fill_missing_stats(df, stat_cols=None):
    df_filled = df.copy()
    
    # ì²˜ë¦¬í•  ìˆ˜ì¹˜í˜• ì»¬ëŸ¼ ì��ë�™ ì„ íƒ�
    if stat_cols is None:
        exclude_cols = ['Season', 'TeamID', 'Score', 'OppScore', 'FirstD1Season', 'IsNewTeam']
        stat_cols = [col for col in df_filled.select_dtypes(include='number').columns if col not in exclude_cols]
    
    # ì‹œì¦Œ + ì»¨í�¼ëŸ°ìŠ¤ í�‰ê·  â†’ ì‹œì¦Œ í�‰ê· ìœ¼ë¡œ ì±„ìš°ê¸°
    for col in stat_cols:
        df_filled[col] = df_filled.groupby(['Season', 'ConfAbbrev'])[col].transform(lambda x: x.fillna(x.mean()))
        df_filled[col] = df_filled.groupby('Season')[col].transform(lambda x: x.fillna(x.mean()))

    return df_filled


def fill_missing_game_level(df):
    df_filled = df.copy()

    # --- 1ï¸� GameCity, GameState â†’ Neutral ---
    if 'GameCity' in df_filled.columns:
        df_filled['GameCity'] = df_filled['GameCity'].fillna('Neutral')
    if 'GameState' in df_filled.columns:
        df_filled['GameState'] = df_filled['GameState'].fillna('Neutral')

    # --- 2ï¸� OffEff, DefEff (ì‹œì¦Œ+Sourceë³„ í�‰ê· ) ---
    eff_cols = ['T1_OffEff', 'T1_DefEff', 'T2_OffEff', 'T2_DefEff']
    for col in eff_cols:
        if col in df_filled.columns:
            df_filled[col] = df_filled.groupby(['Season', 'Source'])[col].transform(lambda x: x.fillna(x.mean()))

    # --- 3ï¸� SeedRank â†’ ì—†ëŠ” íŒ€ 20 ---
    if 'T1_SeedRank' in df_filled.columns:
        df_filled['T1_SeedRank'] = df_filled['T1_SeedRank'].fillna(20)
    if 'T2_SeedRank' in df_filled.columns:
        df_filled['T2_SeedRank'] = df_filled['T2_SeedRank'].fillna(20)
    if 'SeedDiff' in df_filled.columns:
        df_filled['SeedDiff'] = df_filled['T1_SeedRank'] - df_filled['T2_SeedRank']

    return df_filled






def reduce_memory(df):
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

def check_memory(stage):
    mem = psutil.virtual_memory()
    print(f"[{stage}] ğŸ’¾ Memory Usage: {mem.percent}% ({mem.used / (1024 ** 3):.2f} GB used of {mem.total / (1024 ** 3):.2f} GB)")

def downcast_dtypes(df):
    float_cols = df.select_dtypes(include=['float64']).columns
    int_cols = df.select_dtypes(include=['int64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    df[int_cols] = df[int_cols].astype('int32')
    return df


def process_pipeline(gender, data_path):
    print(f'--- [{gender}] Pipeline ì‹œì�‘ ---')

    # 1. Load Data
    detailed_season, compact_season, detailed_tourney, compact_tourney, compact_secondary, teams, conferences, game_cities, cities, seeds = load_data(gender, data_path)
    check_memory("Data Loaded")

    # 2. Clean compact datasets early
    compact_season = clean_compact_df(compact_season)
    compact_tourney = clean_compact_df(compact_tourney)
    compact_secondary = clean_compact_df(compact_secondary)

    # 3. Combine compact results
    df_compact_all = pd.concat([compact_season, compact_tourney, compact_secondary], ignore_index=True)
    df_compact_all = attach_state_info(df_compact_all, game_cities, cities)
    check_memory("Compact Combined")

    del compact_season, compact_tourney, compact_secondary, game_cities, cities
    gc.collect()

    # 4. Combine detailed results & calculate Team Avg Stats
    detailed_season = clean_compact_df(detailed_season)
    detailed_tourney = clean_compact_df(detailed_tourney)
    df_detailed_all = pd.concat([detailed_season, detailed_tourney], ignore_index=True)
    team_avg_stats = process_team_stats(df_detailed_all, df_compact_all)
    check_memory("Team Avg Stats Calculated")

    del detailed_season, detailed_tourney, df_detailed_all
    gc.collect()

    # 5. Prepare Team-level Compact + Add Conference
    win_compact = prepare_compact(df_compact_all, is_winner=True)
    lose_compact = prepare_compact(df_compact_all, is_winner=False)
    df_compact_team = pd.concat([win_compact, lose_compact], ignore_index=True)

    # Add Conference Info
    conf_copy = conferences[['TeamID', 'ConfAbbrev']].drop_duplicates(subset=['TeamID'])
    df_compact_team = df_compact_team.merge(conf_copy, on='TeamID', how='left')

    # === ğŸ”¥ NaN Filing ===
    df_compact_team = fill_missing_stats(df_compact_team)

    # 6. Conference Stats
    conf_stats = process_conference_stats(df_compact_team, df_compact_team)
    check_memory("Conference Stats Done")

    del win_compact, lose_compact, df_compact_team
    gc.collect()

    # 7. H2H Stats
    h2h_df = process_h2h(df_compact_all)
    check_memory("H2H Stats Done")

    # 8. Drop unnecessary columns
    df_compact_all.drop(columns=['DayNum'], inplace=True, errors='ignore')

    # 9. Game-level dataset ìƒ�ì„±
    df_game_level = create_game_team_level(
        compact_df=df_compact_all,
        team_stats_df=team_avg_stats,
        h2h_df=h2h_df,
        conf_df=conferences,
        conf_stats_df=conf_stats,
        seeds=seeds
    )
    check_memory("Game-level Dataset Created")

    # === ğŸ”¥ 2ï¸� Game-level ê²°ì¸¡ì¹˜ ì²˜ë¦¬ ===
    df_game_level = fill_missing_game_level(df_game_level)

    del df_compact_all, team_avg_stats, h2h_df, conf_stats, conferences, teams
    gc.collect()

    # ğŸ‘‰ 10. Shuffle Game-level ë�°ì�´í„° & Result ì�¬ê³„ì‚°
    df_game_level = shuffle_game_level(df_game_level)

    # 11. ì»¬ëŸ¼ í™•ì�¸
    print(" Final Game-level Data Columns check:")
    print(df_game_level.columns.tolist())

    # 12. Save CSV
    df_game_level.to_csv(f'{gender}_game_level_final.csv', index=False)
    print(f" [{gender}] Final CSV save Completed: {gender}_game_level_final.csv")

    return df_game_level


# === ì‹¤í–‰ ===
data_path = '/kaggle/input/march-machine-learning-mania-2025'
df_game_level_m = process_pipeline('M', data_path)
df_game_level_w = process_pipeline('W', data_path)

# ì €ì�¥ í›„ ì�¬í™•ì�¸
print(df_game_level_m.shape)
print(df_game_level_m['Result'].value_counts())


# 1. ì „ì²˜ë¦¬ ë��ë‚œ í›„ ì €ì�¥
df_game_level_m.to_csv('/kaggle/working/M_game_level_final.csv', index=False)
df_game_level_w.to_csv('/kaggle/working/W_game_level_final.csv', index=False)

print(f"âœ… {gender}_game_level_final.csv ì €ì�¥ ì™„ë£Œ")

#  2. ì €ì�¥í•œ íŒŒì�¼ ë‹¤ì‹œ ë¶ˆëŸ¬ì™€ì„œ í™•ì�¸ (í˜¹ì‹œë�¼ë�„ ê¹¨ì§€ê±°ë‚˜ ë¬¸ì œ ì—†ëŠ”ì§€)
df_loaded = pd.read_csv(f'/kaggle/working/{gender}_game_level_final.csv')
print(f"ğŸ”� ë¶ˆëŸ¬ì˜¨ {gender} ë�°ì�´í„° shape:", df_loaded.shape)

#  3. Result ë¶„í�¬ í™•ì�¸
print(df_loaded['Result'].value_counts())

#  4. NaN ì»¬ëŸ¼ í™•ì�¸
nan_cols = df_loaded.columns[df_loaded.isna().any()].tolist()
print("ğŸ”� NaN í�¬í•¨ë�œ ì»¬ëŸ¼:", nan_cols)

#  5. NaN ê°œìˆ˜ ì¶œë ¥
if nan_cols:
    print("\nğŸ“Š ì»¬ëŸ¼ë³„ NaN ê°œìˆ˜:")
    print(df_loaded[nan_cols].isna().sum())
else:
    print("âœ… NaN ì—†ì�Œ!")



def preprocess_for_nn(df, target_col='Result', encoders=None, scaler=None, inference=False, test_size=0.2, random_state=42, apply_smote=False):
    df = df.copy()
    
    # --- 1. Feature / Target Split ---
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # --- 2. Encoding ---
    if not encoders:
        encoders = {}
        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
    else:
        for col, le in encoders.items():
            X[col] = le.transform(X[col])

    # --- 3. Train/Test Split ---
    if not inference:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        X_train, X_test, y_train, y_test = X, None, y, None

    # --- 4. SMOTE (Optional) ---
    if not inference and apply_smote:
        print("ğŸŸ¢ Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    else:
        print("â�Œ SMOTE Skipped.")
    
    # --- 5. Scaling ---
    if not scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_test is not None:
            X_test = scaler.transform(X_test)
    else:
        X_train = scaler.transform(X_train)
        if X_test is not None:
            X_test = scaler.transform(X_test)

    print(f"Train X NaNs: {np.isnan(X_train).sum()}")
    print(f"Train X Max: {np.max(X_train)}, Min: {np.min(X_train)}")

    
    return X_train, X_test, y_train, y_test, encoders, scaler



def check_for_nans(X, y, gender='M'):
    """
    Check for NaN values in input and labels before training.
    """
    X_nan_count = np.isnan(X).sum()
    y_nan_count = np.isnan(y).sum()

    if X_nan_count > 0 or y_nan_count > 0:
        print(f"ğŸš¨ [{gender}] NaN Detected - X NaNs: {X_nan_count}, y NaNs: {y_nan_count}")
        raise ValueError(f"[{gender}] NaN detected! Clean data before training.")
    else:
        print(f"âœ… [{gender}] No NaNs in data. Safe to proceed.")



# --- NCAA_NN ëª¨ë�¸ ---
class NCAA_NN(nn.Module):
    def __init__(self, input_dim):
        super(NCAA_NN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),              # â­�ï¸� BatchNorm ì¶”ê°€
            nn.LeakyReLU(),                   # (optional) ReLU ëŒ€ì‹  LeakyReLU
            nn.Dropout(0.3),                  # â­�ï¸� Dropout ì¶”ê°€ (30% í™•ë¥ ë¡œ ë…¸ë“œ ë�”)

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),               # BatchNorm
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 2)  # CrossEntropyLoss â†’ Output 2 classes
        )

    def forward(self, x):
        return self.model(x)



# Training loop
def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    running_loss = 0

    for X, y in tqdm(dataloader,desc="Traing Loop", leave= False):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def brier_score(y_true, y_probs):
    """
    y_true: Tensor of shape (batch_size,), contains 0 or 1
    y_probs: Tensor of shape (batch_size,), predicted probabilities for class 1
    """
    return torch.mean((y_probs - y_true.float()) ** 2).item()



def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    test_loss, correct = 0, 0
    num_batches = len(dataloader)
    
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Test Loop", leave=False):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            probs = torch.softmax(pred, dim=1)[:, 1]  # í™•ë¥ ê°’ (class 1 ê¸°ì¤€)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            all_probs.append(probs.cpu())
            all_targets.append(y.cpu())

    # Loss, Accuracy
    test_loss /= num_batches
    correct /= len(dataloader.dataset)

    # --- Brier Score ì¶”ê°€ ---
    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)
    brier = torch.mean((all_probs - all_targets.float()) ** 2).item()

    return test_loss, correct, brier



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_wts = None 

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_wts = model.state_dict() 
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True



import random

def reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # GPUì—�ì„œ ë�™ì�¼ ê²°ê³¼
    torch.backends.cudnn.benchmark = False


def train_pipeline(df, gender='M'):
    print(f"ğŸ“Œ [{gender}] ë�°ì�´í„° ì „ì²˜ë¦¬ ì‹œì�‘ ì¤‘")
    X_train, X_test, y_train, y_test, encoders, scaler = preprocess_for_nn(df)
    print(f"ğŸ“Œ [{gender}] ë�°ì�´í„° ì „ì²˜ë¦¬ ì™„ë£Œ")

     # === ğŸš¨ NaN ì²´í�¬ ===
    check_for_nans(X_train, y_train, gender)

    # --- Tensor ---
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)  # y_trainì�€ ì—¬ì „í�ˆ DataFrame/Series
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)


    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=6400, shuffle=True, num_workers=2, pin_memory= False)
    test_dataloader = DataLoader(test_dataset, batch_size=64000, shuffle=False, num_workers=2, pin_memory= False)

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    model = NCAA_NN(input_dim)
    if torch.cuda.device_count() > 1:
        print(f"ğŸš€ Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- Early Stopping ---
    early_stopper = EarlyStopping(patience=5, min_delta=0.001)

    train_loss_values = []
    test_loss_values = []
    epoch_count = []

    # --- Train Loop ---
    print(f"ğŸš€ [{gender}] Training Start")
    for epoch in tqdm(range(10), desc=f"[{gender}] Epochs"):
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device)
        val_loss, val_acc, val_brier = test_loop(test_dataloader, model, loss_fn, device)
        print(f"Epoch [{epoch+1}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
        train_loss_values.append(train_loss)
        test_loss_values.append(val_loss)
        epoch_count.append(epoch+1)
    
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"â�— Early stopping triggered at epoch {epoch+1}")
            break
    
    print(" Applying Best Model from EarlyStopping")
    model.load_state_dict(early_stopper.best_model_wts)
    torch.save(model.state_dict(), f'{gender}_model.pth')

    # --- Save ---
    torch.save(model.state_dict(), f'{gender}_model.pth')
    joblib.dump(encoders, f'{gender}_encoders.pkl')
    joblib.dump(scaler, f'{gender}_scaler.pkl')

    # --- Loss Plot ---
    plt.plot(epoch_count, train_loss_values, label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")
    plt.title(f"{gender} Training Loss Curve")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
    
    return model, encoders, scaler, device, test_dataloader, y_test



from tqdm.notebook import tqdm  # Notebook í™˜ê²½ìš© tqdm ì‚¬ìš©

def predict_and_evaluate(model, test_loader, device, threshold=0.5):  # device ì¶”ê°€
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, _ in tqdm(test_loader, desc="Predicting", unit="batch", leave=False):
            X_batch = X_batch.to(device)  # ğŸ”¥ ë°˜ë“œì‹œ GPUë¡œ ì˜¬ë¦¼
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()  # CPUë¡œ ë‹¤ì‹œ ì˜®ê¸´ í›„ numpy ë³€í™˜
            preds = (probs >= threshold).astype(int)                # 0/1 ì˜ˆì¸¡
            
            all_probs.extend(probs)
            all_preds.extend(preds)
    
    # ê²°ê³¼ DataFrame ì •ë¦¬
    result_df = pd.DataFrame({
        'Predicted_Label': all_preds,
        'Predicted_Prob': all_probs
    })
    
    return result_df



def show_misclassified(df, true_labels):
    df['True_Label'] = true_labels
    misclassified = df[df['Predicted_Label'] != df['True_Label']]
    print(f"Total Misclassified: {len(misclassified)}")
    return misclassified



def save_predictions(result_df, filename='predictions.csv'):
    result_df.to_csv(filename, index=False)
    print(f"Saved predictions to {filename}")



from tqdm.notebook import tqdm  

reset_seeds(42)

#--- Training ---
model_m, encoders_m, scaler_m, device_m, test_loader_m, y_test_m = train_pipeline(df_game_level_m, gender='M')

# --- Check ---
result_df = predict_and_evaluate(model_m, test_loader_m, device_m)
print(result_df.head())
misclassified = show_misclassified(result_df, y_test_m)
save_predictions(result_df, 'final_predictions_m.csv')


from tqdm.notebook import tqdm 

reset_seeds(42)


#--- Training ---
model_w, encoders_w, scaler_w, device_w, test_loader_w, y_test_w = train_pipeline(df_game_level_w, gender='w')

# --- Check ---
result_df = predict_and_evaluate(model_w, test_loader_w,device_w)
print(result_df.head())
misclassified = show_misclassified(result_df, y_test_w)
save_predictions(result_df, 'final_predictions_w.csv')


# --- ë�°ì�´í„° ê²½ë¡œ ---
data_path = '/kaggle/input/march-machine-learning-mania-2025'

def save_submission(sample_submission_path, result_df, output_filename):
    sample_submission = pd.read_csv(sample_submission_path)

    # === ID ì»¬ëŸ¼ì—�ì„œ Season, Team1ID, Team2ID ë¶„í•´ ===
    sample_submission[['Season', 'Team1ID', 'Team2ID']] = sample_submission['ID'].str.split('_', expand=True).astype(int)

    # === Predicted Prob ì±„ìš°ê¸° ===
    sample_submission['Pred'] = result_df['Predicted_Prob']

    # === ìµœì¢… ì œì¶œ í�¬ë§· ===
    submission = sample_submission[['ID', 'Pred']]
    submission.to_csv(output_filename, index=False)
    print(f"âœ… {output_filename} ì €ì�¥ ì™„ë£Œ!")

    return submission




# --- Stage1 ë‚¨ì„± ---
result_df_m = predict_and_evaluate(model_m, test_loader_m, device_m)
submission_m_stage1 = save_submission(f'{data_path}/SampleSubmissionStage1.csv', result_df_m, 'SampleSubmissionStage1_m.csv')

# --- Stage1 ì—¬ì„± ---
result_df_w = predict_and_evaluate(model_w, test_loader_w, device_w)
submission_w_stage1 = save_submission(f'{data_path}/SampleSubmissionStage1.csv', result_df_w, 'SampleSubmissionStage1_w.csv')

# Stage1 í†µí•©
final_stage1 = pd.concat([submission_m_stage1, submission_w_stage1], ignore_index=True)
final_stage1.to_csv('SampleSubmissionStage1.csv', index=False)
print("ğŸš€ Stage1 ìµœì¢… ì œì¶œíŒŒì�¼: SampleSubmissionStage1.csv")


# === Stage2 ì „ì²´ íŒŒì�´í”„ë�¼ì�¸ (ìµœì¢… ìˆ˜ì •ë³¸) ===

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib

# (0) í•™ìŠµ ë•Œ ì‹¤ì œ ì¡´ì�¬í•˜ë�˜ ì˜ˆì‹œê°’ (ë°˜ë“œì‹œ Training ë�°ì�´í„°ì—� ì�ˆë�˜ City/State)
TRAIN_DEFAULT_CITY = "Philadelphia"  # í•™ìŠµ ë�°ì�´í„°ì—� ì¡´ì�¬í•˜ë�˜ GameCity
TRAIN_DEFAULT_STATE = "PA"           # í•™ìŠµ ë�°ì�´í„°ì—� ì¡´ì�¬í•˜ë�˜ GameState

# ìŠ¤ì¼€ì�¼ëŸ¬ê°€ í•™ìŠµ ì‹œì �ì—� ë´¤ë�˜ í”¼ì²˜ ìˆœì„œ( scikit-learn>=1.0 )
train_cols_m = list(scaler_m.feature_names_in_)  # ë‚¨ì„±ìš©
train_cols_w = list(scaler_w.feature_names_in_)  # ì—¬ì„±ìš©

# --- 1. Stage2 ë�°ì�´í„° ë¶ˆëŸ¬ì˜¤ê¸° ---
stage2_m = pd.read_csv(f'{data_path}/SampleSubmissionStage2.csv')
stage2_w = pd.read_csv(f'{data_path}/SampleSubmissionStage2.csv')

# --- 2. IDì—�ì„œ í•„ìš”í•œ ì»¬ëŸ¼ ì¶”ì¶œ ---
stage2_m[['Season', 'Team1ID', 'Team2ID']] = stage2_m['ID'].str.split('_', expand=True).astype(int)
stage2_w[['Season', 'Team1ID', 'Team2ID']] = stage2_w['ID'].str.split('_', expand=True).astype(int)

# --- 3. Feature ìƒ�ì„± ---
stage2_features_m = prepare_inference_features(stage2_m, gender='M')
stage2_features_w = prepare_inference_features(stage2_w, gender='W')

# --- 4. GameCity / GameState ì±„ì›Œë„£ê¸° ---
for df_stg2 in [stage2_features_m, stage2_features_w]:
    # ë§Œì•½ ì»¬ëŸ¼ì�´ ì—†ìœ¼ë©´ ì¶”ê°€, NaNì�´ë©´ fill
    if 'GameCity' not in df_stg2.columns:
        df_stg2['GameCity'] = TRAIN_DEFAULT_CITY
    else:
        df_stg2['GameCity'].fillna(TRAIN_DEFAULT_CITY, inplace=True)

    if 'GameState' not in df_stg2.columns:
        df_stg2['GameState'] = TRAIN_DEFAULT_STATE
    else:
        df_stg2['GameState'].fillna(TRAIN_DEFAULT_STATE, inplace=True)

# --- 5. Encoding & Scaling ---
# 5-1) ID ì»¬ëŸ¼ ì œê±°
X_stage2_m = stage2_features_m.drop(columns=['ID'], errors='ignore')
X_stage2_w = stage2_features_w.drop(columns=['ID'], errors='ignore')

# 5-2) 'Result' ì»¬ëŸ¼ ì‚­ì œ
if 'Result' in X_stage2_m.columns:
    X_stage2_m.drop(columns=['Result'], inplace=True)
if 'Result' in X_stage2_w.columns:
    X_stage2_w.drop(columns=['Result'], inplace=True)

# 5-3) ë‚¨ì„±ìš© LabelEncoder
for col, encoder in encoders_m.items():
    if col in X_stage2_m.columns:
        X_stage2_m[col] = encoder.transform(X_stage2_m[col])

# 5-4) ì—¬ì„±ìš© LabelEncoder
for col, encoder in encoders_w.items():
    if col in X_stage2_w.columns:
        X_stage2_w[col] = encoder.transform(X_stage2_w[col])

# 5-5) REINDEX â†’ í•™ìŠµ ë•Œ ë³¸ í”¼ì²˜ ìˆœì„œ / ê°œìˆ˜ ë§�ì¶”ê¸°
# fill_value=0 ë“±ìœ¼ë¡œ ì—†ë�˜ ì»¬ëŸ¼ì�„ ë³´ì™„
X_stage2_m = X_stage2_m.reindex(columns=train_cols_m, fill_value=0)
X_stage2_w = X_stage2_w.reindex(columns=train_cols_w, fill_value=0)

# ì�´ì œ Scaler transform
X_stage2_m_scaled = scaler_m.transform(X_stage2_m)
X_stage2_w_scaled = scaler_w.transform(X_stage2_w)

# --- 6. Tensor ë³€í™˜ ---
X_stage2_m_tensor = torch.tensor(X_stage2_m_scaled, dtype=torch.float32)
X_stage2_w_tensor = torch.tensor(X_stage2_w_scaled, dtype=torch.float32)

dummy_y_m = torch.zeros(X_stage2_m_tensor.shape[0], dtype=torch.long)
dummy_y_w = torch.zeros(X_stage2_w_tensor.shape[0], dtype=torch.long)

stage2_dataset_m = TensorDataset(X_stage2_m_tensor, dummy_y_m)
stage2_dataset_w = TensorDataset(X_stage2_w_tensor, dummy_y_w)

test_loader_m_stage2 = DataLoader(stage2_dataset_m, batch_size=64000, shuffle=False)
test_loader_w_stage2 = DataLoader(stage2_dataset_w, batch_size=64000, shuffle=False)

# --- 7. Stage2 ì˜ˆì¸¡ ---
result_df_m_stage2 = predict_and_evaluate(model_m, test_loader_m_stage2, device_m)
result_df_w_stage2 = predict_and_evaluate(model_w, test_loader_w_stage2, device_w)

# --- 8. Submission íŒŒì�¼ ì €ì�¥ ---
stage2_submission_m = pd.read_csv(f'{data_path}/SampleSubmissionStage2.csv')
stage2_submission_w = pd.read_csv(f'{data_path}/SampleSubmissionStage2.csv')

stage2_submission_m['Pred'] = result_df_m_stage2['Predicted_Prob']
stage2_submission_w['Pred'] = result_df_w_stage2['Predicted_Prob']

stage2_submission_m.to_csv('/kaggle/working/SampleSubmissionStage2_m.csv', index=False)
stage2_submission_w.to_csv('/kaggle/working/SampleSubmissionStage2_w.csv', index=False)

# --- 9. ìµœì¢… í†µí•© ---
final_stage2 = pd.concat([stage2_submission_m, stage2_submission_w], ignore_index=True)
final_stage2.to_csv('/kaggle/working/SampleSubmissionStage2.csv', index=False)
print(" Stage2 ìµœì¢… ì œì¶œ íŒŒì�¼ ì™„ì„±!")


data_path = '/kaggle/input/march-machine-learning-mania-2025'

final_stage1.to_csv('/kaggle/working/submission.csv', index=False)


