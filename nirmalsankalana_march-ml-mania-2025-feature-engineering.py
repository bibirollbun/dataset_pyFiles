import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import plotly.io as pio
pio.renderers.default = 'iframe'


def d_types_report(df):
    columns=[]
    d_types=[]
    uniques=[]
    n_uniques=[]
    null_values=[]
    null_values_percentage=[]
    rows = df.shape[0]
    
    for i in df.columns:
        columns.append(i)
        d_types.append(df[i].dtypes)
        uniques.append(df[i].unique()[:5])
        n_uniques.append(df[i].nunique())
        null_values.append(df[i].isna().sum())
        null_values_percentage.append(null_values[-1] * 100 / rows)

    return pd.DataFrame({"Columns": columns, "Data_Types": d_types, "Unique_values": uniques, "N_Uniques": n_uniques,  "Null_Values": null_values, "Null_Values_percentage": null_values_percentage})


PTH_SUBMISSION = '/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv'
submission_df = pd.read_csv(PTH_SUBMISSION)
submission_df.head()


submission_df.tail()


M_TEAMS_PTH = '/kaggle/input/march-machine-learning-mania-2025/MTeams.csv'
df_mteams = pd.read_csv(M_TEAMS_PTH)
df_mteams.head()


df_mteams.tail()


d_types_report(df_mteams)


df_mteams = df_mteams[df_mteams['LastD1Season'] == 2025]
df_mteams


df_mteams['LastD1Season'] == 2025
active_teams = df_mteams['TeamID'].unique()


M_SEASONS_PATH = '/kaggle/input/march-machine-learning-mania-2025/MSeasons.csv'
df_mseasons = pd.read_csv(M_SEASONS_PATH)
df_mseasons.head()


df_mseasons.tail()


d_types_report(df_mseasons)


M_TOURNEY_SEEDS_PTH = '/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv'
df_m_seeds = pd.read_csv(M_TOURNEY_SEEDS_PTH)
df_m_seeds.head(10)


df_m_seeds['SeedValue'] = df_m_seeds['Seed'].str.extract('(\d+)').astype(int)
df_m_seeds['Region'] = df_m_seeds['Seed'].str[0]
df_m_seeds = df_m_seeds.drop('Seed', axis=1)
df_m_seeds.tail()


d_types_report(df_m_seeds)


M_REGULAR_RES = '/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv'
m_regular = pd.read_csv(M_REGULAR_RES)
m_regular.head()


m_regular.tail()


d_types_report(m_regular)


M_TOURNAMENT_RES = '/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv'
m_tournament = pd.read_csv(M_TOURNAMENT_RES)
m_tournament.head()


m_tournament.tail()


d_types_report(m_tournament)


PTH_M_TOURNEY = '/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv'
m_tourney = pd.read_csv(PTH_M_TOURNEY)
m_tourney.head()


m_tourney.tail()


d_types_report(m_tourney)


W_TEAMS_PTH = '/kaggle/input/march-machine-learning-mania-2025/WTeams.csv'
df_wteams = pd.read_csv(W_TEAMS_PTH)
df_wteams.head()


df_wteams.tail()


d_types_report(df_wteams)


active_teams = df_wteams['TeamID'].unique()


W_SEASONS_PATH = '/kaggle/input/march-machine-learning-mania-2025/WSeasons.csv'
df_wseasons = pd.read_csv(W_SEASONS_PATH)
df_wseasons.tail()


d_types_report(df_wseasons)


W_TOURNEY_SEEDS_PTH = '/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv'
df_w_seeds = pd.read_csv(W_TOURNEY_SEEDS_PTH)
df_w_seeds.head()


df_w_seeds['SeedValue'] = df_w_seeds['Seed'].str.extract('(\d+)').astype(int)
df_w_seeds['Region'] = df_w_seeds['Seed'].str[0]
df_w_seeds = df_w_seeds.drop('Seed', axis=1)
df_w_seeds.head()


df_w_seeds.tail()


d_types_report(df_w_seeds)


W_REGULAR_PTH = '/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv'
w_regular = pd.read_csv(W_REGULAR_PTH)
w_regular.head()


w_regular.tail()


d_types_report(w_regular)


W_TOURNAMENT_PTH = '/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv'
w_tournament = pd.read_csv(W_TOURNAMENT_PTH)
w_tournament.head()


w_tournament.tail()


d_types_report(w_tournament)


PTH_W_TOURNEY = '/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv'
w_tourney = pd.read_csv(PTH_W_TOURNEY)
w_tourney.head()


w_tourney.tail()


d_types_report(w_tourney)


m_regular["type"] = "regular"
m_tournament["type"] = "tournament"
# m_tourney["type"] = "tourney"
m_combined = pd.concat([m_regular, m_tournament], ignore_index=True)
m_combined.head()


# m_combined = m_combined.merge(df_m_seeds[['Season', 'TeamID', 'SeedValue', 'Region']],
#                               left_on=['Season', 'WTeamID'], 
#                               right_on=['Season', 'TeamID'], 
#                               how='left').rename(columns={'SeedValue': 'Team1_Seed', 'Region': 'Team1_Region'})

# m_combined.drop(columns=['TeamID'], inplace=True)  # Drop duplicate column after merge

# # Merge seeds for Team2
# m_combined = m_combined.merge(df_m_seeds[['Season', 'TeamID', 'SeedValue', 'Region']],
#                               left_on=['Season', 'LTeamID'], 
#                               right_on=['Season', 'TeamID'], 
#                               how='left').rename(columns={'SeedValue': 'Team2_Seed', 'Region': 'Team2_Region'})

# m_combined.drop(columns=['TeamID'], inplace=True) 
# m_combined.head(10)


m_combined.tail()


d_types_report(m_combined)


w_regular["type"] = "regular"
w_tournament["type"] = "tournament"
# w_tourney["type"] = "tourney"
w_combined = pd.concat([w_regular, w_tournament], ignore_index=True)
w_combined.head()


# w_combined = w_combined.merge(df_w_seeds[['Season', 'TeamID', 'SeedValue', 'Region']],
#                               left_on=['Season', 'WTeamID'], 
#                               right_on=['Season', 'TeamID'], 
#                               how='left').rename(columns={'SeedValue': 'Team1_Seed', 'Region': 'Team1_Region'})

# w_combined.drop(columns=['TeamID'], inplace=True)  # Drop duplicate column after merge

# # Merge seeds for Team2
# w_combined = w_combined.merge(df_w_seeds[['Season', 'TeamID', 'SeedValue', 'Region']],
#                               left_on=['Season', 'LTeamID'], 
#                               right_on=['Season', 'TeamID'], 
#                               how='left').rename(columns={'SeedValue': 'Team2_Seed', 'Region': 'Team2_Region'})

# w_combined.drop(columns=['TeamID'], inplace=True) 
# w_combined.head()


w_combined.tail()


d_types_report(w_combined)


w_combined["gender"] = "w"
m_combined["gender"] = "m"
all_combined = pd.concat([w_combined, m_combined], ignore_index=True)
all_combined['diff'] = all_combined['WScore'] - all_combined['LScore']
all_combined = all_combined.drop(['NumOT'], axis=1)
year_start = 1985
all_combined['day'] = (all_combined['Season'] - year_start) * 365 + all_combined['DayNum']
all_combined = all_combined.sort_values(by='day', ascending=True)
all_combined.head()


all_combined.tail()


d_types_report(all_combined)


submission_df[['Season', 'WTeamID', 'LTeamID']] = submission_df['ID'].str.split('_', expand=True)

# Convert Season and Team IDs to integers
submission_df['Season'] = submission_df['Season'].astype(int)
submission_df['WTeamID'] = submission_df['WTeamID'].astype(int)
submission_df['LTeamID'] = submission_df['LTeamID'].astype(int)
submission_df['type'] = 'tournament'
submission_df['gender'] = submission_df['Season'].apply(lambda x: 'm' if x < 3000 else 'w')
submission_df = submission_df.drop(['Pred', 'ID'], axis=1)
submission_df.head()


submission_df.tail()


columns_needed = ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'type', 'gender', 'diff', 'day']

# Add missing columns to `submission_df` and set them to NaN
for col in columns_needed:
    if col not in submission_df.columns:
        submission_df[col] = 0

# Concatenate both DataFrames
combined_df = pd.concat([all_combined, submission_df], ignore_index=True)

# Display the first few rows
combined_df.tail()


hehe = combined_df[(combined_df['Season'] == 2025) & (combined_df['type'] == 'tournament')].copy()
hehe.head()


d_types_report(combined_df)


data_list = []
previous_match_results = {}  # Store previous match results
team_wins_last_season = {}   # Store total wins per team for the previous season
team_scores_last_season = {} # Store total scores per team in the previous tournament
previous_match_scores = {}   # Store previous match scores between Team1 and Team2

for index, row in combined_df.iterrows():
    season = row['Season']
    teamW = row['WTeamID']
    teamL = row['LTeamID']
    gameType = row['type']
    gender = row['gender']
    team1 = min(teamW, teamL)
    team2 = max(teamW, teamL)
    isTeam1Winning = 1 if teamW == team1 else 0
    
    # if(season == 2025 and gameType == 'tournament'):
    #     print('hehe')
    # Extract team scores
    scoreW = row['WScore']
    scoreL = row['LScore']

    # Create unique keys for tracking matchups
    match_key = (season, team1, team2)

    # Find previous match results iteratively until found or season < 1985
    prev_season = season
    while prev_season >= 1985:
        prev_match_key = (prev_season, team1, team2)
    
        previous_match_team1_won = previous_match_results.get(prev_match_key, None)
        if previous_match_team1_won is not None:
            break  # Stop if found
    
        prev_season -= 1  # Move to the previous season
    
    # Find previous match scores iteratively
    prev_season = season
    while prev_season >= 1985:
        prev_match_key = (prev_season, team1, team2)
    
        team1_prev_match_score, team2_prev_match_score = previous_match_scores.get(prev_match_key, (None, None))
        if team1_prev_match_score is not None and team2_prev_match_score is not None:
            break  # Stop if found
    
        prev_season -= 1  # Move to the previous season
    
    # If still None, set default values
    if previous_match_team1_won is None:
        previous_match_team1_won = 0
    
    if team1_prev_match_score is None or team2_prev_match_score is None:
        team1_prev_match_score, team2_prev_match_score = (0, 0)


    prev_match_score_diff = team1_prev_match_score - team2_prev_match_score

    # Retrieve previous season's win counts
    prev_season = season - 1
    if gameType == 'regular':
        team1_prev_reg_wins = team_wins_last_season.get((prev_season, team1, 'regular'), 0)
        team2_prev_reg_wins = team_wins_last_season.get((prev_season, team2, 'regular'), 0)
        team1_prev_tour_wins = team_wins_last_season.get((prev_season, team1, 'tournament'), 0)
        team2_prev_tour_wins = team_wins_last_season.get((prev_season, team2, 'tournament'), 0)

    if gameType == 'tournament':
        team1_prev_reg_wins = team_wins_last_season.get((season, team1, 'regular'), 0)
        team2_prev_reg_wins = team_wins_last_season.get((season, team2, 'regular'), 0)
        team1_prev_tour_wins = team_wins_last_season.get((prev_season, team1, 'tournament'), 0)
        team2_prev_tour_wins = team_wins_last_season.get((prev_season, team2, 'tournament'), 0)

    # Retrieve previous tournament total scores
    if gameType == 'regular':
        team1_prev_reg_scores = team_scores_last_season.get((prev_season, team1, 'regular'), 0)
        team2_prev_reg_scores = team_scores_last_season.get((prev_season, team2, 'regular'), 0)
        team1_prev_tour_scores = team_scores_last_season.get((prev_season, team1, 'tournament'), 0)
        team2_prev_tour_scores = team_scores_last_season.get((prev_season, team2, 'tournament'), 0)

    if gameType == 'tournament':
        team1_prev_reg_scores = team_scores_last_season.get((season, team1, 'regular'), 0)
        team2_prev_reg_scores = team_scores_last_season.get((season, team2, 'regular'), 0)
        team1_prev_tour_scores = team_scores_last_season.get((prev_season, team1, 'tournament'), 0)
        team2_prev_tour_scores = team_scores_last_season.get((prev_season, team2, 'tournament'), 0)

    prev_reg_matches_diff = team1_prev_reg_wins - team2_prev_reg_wins
    prev_tour_matches_diff = team1_prev_tour_wins - team2_prev_tour_wins
    prev_reg_score_diff = team1_prev_reg_scores - team2_prev_reg_scores
    prev_tour_score_diff = team1_prev_tour_scores - team2_prev_tour_scores

    # Store match result for future reference
    previous_match_results[match_key] = isTeam1Winning

    # Update win records for the current season
    team_wins_last_season[(season, teamW, gameType)] = team_wins_last_season.get((season, teamW, gameType), 0) + 1

    # Update score records for the current season
    team_scores_last_season[(season, teamW, gameType)] = team_scores_last_season.get((season, teamW, gameType), 0) + scoreW
    team_scores_last_season[(season, teamL, gameType)] = team_scores_last_season.get((season, teamL, gameType), 0) + scoreL

    # Store scores for this matchup for future reference
    team1_score = scoreW if teamW == team1 else scoreL
    team2_score = scoreW if teamW == team2 else scoreL
    previous_match_scores[match_key] = (team1_score, team2_score)
    id = f"{season}_{team1}_{team2}"

    data_list.append({
        'ID': id,
        'Season': season,
        'Team1': team1,
        'Team2': team2,
        'Type': gameType,
        'Gender': gender,
        'isPreviousMatchWithTeam2Team1Won': previous_match_team1_won if previous_match_team1_won is not None else 0,
        'team1ScoreWithTeam2PrevMatch': team1_prev_match_score,
        'team2ScoreWithTeam1PrevMatch': team2_prev_match_score,
        'prevMatchScoreDiff': prev_match_score_diff,
        'Team1Winning': isTeam1Winning,
        'Team1WonRegMatchesPrev': team1_prev_reg_wins,
        'Team2WonRegMatchesPrev': team2_prev_reg_wins,
        'Team1WonTourMatchesPrev': team1_prev_tour_wins,
        'Team2WonTourMatchesPrev': team2_prev_tour_wins,
        'Team1TotalScoresPrevReg': team1_prev_reg_scores,
        'Team2TotalScoresPrevReg': team2_prev_reg_scores,
        'Team1TotalScoresPrevTour': team1_prev_tour_scores,
        'Team2TotalScoresPrevTour': team2_prev_tour_scores,
        'PrevRegDiff': prev_reg_matches_diff,
        'PrevTourDiff': prev_tour_matches_diff,
        'PrevRegScoreDiff': prev_reg_score_diff,
        'PrevTourScoreDiff': prev_tour_score_diff
    })

# Create the DataFrame from the list of dictionaries
dataset = pd.DataFrame(data_list)

dataset.head(10)


dataset.tail(20)


d_types_report(dataset)


# Separate 2025 regular season rows into test dataset
test = dataset[(dataset['Season'] == 2025) & (dataset['Type'] == 'tournament')].copy()

# Remove the 'Team1Winning' column
test = test.drop(columns=['Team1Winning'])

# Remove those rows from the original dataset
dataset = dataset[~((dataset['Season'] == 2025) & (dataset['Type'] == 'regular'))]


d_types_report(test)


d_types_report(dataset)


test.to_csv('test.csv', index=False)
dataset.to_csv('train.csv', index=False)

