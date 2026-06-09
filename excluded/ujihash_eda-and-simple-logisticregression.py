import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import pprint as pp

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)


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
pp.pprint(data_file_list)


regseason_file = 'RegularSeasonCompactResults.csv'
teams_file = 'Teams.csv'
seasons_file = 'Seasons.csv'
tourney_seeds_file = 'NCAATourneySeeds.csv'
submission_file = 'SampleSubmissionStage2.csv'


# Read regular season data
df_m_regseason = pd.read_csv(os.path.join(data_dir, 'M' + regseason_file))
df_w_regseason = pd.read_csv(os.path.join(data_dir, 'W' + regseason_file))

# Read teams data
df_m_teams = pd.read_csv(os.path.join(data_dir, 'M' + teams_file))
df_w_teams = pd.read_csv(os.path.join(data_dir, 'W' + teams_file))

# Read seasons data
df_m_seasons = pd.read_csv(os.path.join(data_dir, 'M' + seasons_file))
df_w_seasons = pd.read_csv(os.path.join(data_dir, 'W' + seasons_file))

# Read tourney seeds data
df_m_tourney_seeds = pd.read_csv(os.path.join(data_dir, 'M' + tourney_seeds_file))
df_w_tourney_seeds = pd.read_csv(os.path.join(data_dir, 'W' + tourney_seeds_file))


# Output data information for each data
def explore_data(df, name):
    print(f'========== {name} ==========')
    print(f'Shape: {df.shape}')
    print('\nData info:')
    print(df.info())
    print('\nStatistics:')
    display(df.describe())
    print('\n')
    
explore_data(df_m_regseason, 'Men\'s Regular Season')
explore_data(df_w_regseason, 'Women\'s Regular Season')
explore_data(df_m_teams, 'Men\'s Teams')
explore_data(df_w_teams, 'Women\'s Teams')
explore_data(df_m_seasons, 'Men\'s Seasons')
explore_data(df_w_seasons, 'Women\'s Seasons')
explore_data(df_m_tourney_seeds, 'Men\'s Tourney Seeds')
explore_data(df_w_tourney_seeds, 'Women\'s Tourney Seeds')


# Lower Team ID
df_m_regseason['lower_team'] = df_m_regseason[['WTeamID', 'LTeamID']].min(axis=1)
df_w_regseason['lower_team'] = df_w_regseason[['WTeamID', 'LTeamID']].min(axis=1)

# Target
df_m_regseason['target'] = df_m_regseason.apply(lambda x: 1 if x['WTeamID'] == x['lower_team'] else 0, axis=1)
df_w_regseason['target'] = df_w_regseason.apply(lambda x: 1 if x['WTeamID'] == x['lower_team'] else 0, axis=1)

print('Men\'s Target:')
print(df_m_regseason['target'].value_counts())
print('\nWomen\'s Target:')
print(df_w_regseason['target'].value_counts())


fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

fig.suptitle('Lower Team Win Count')

# Men's target
sns.countplot(ax=ax[0], data=df_m_regseason, x='target')
ax[0].set_title('Men\'s Lower Team Win Count')
ax[0].set_xlabel('Target (Win: 1, Lose: 0)')
ax[0].set_ylabel('Count')

# Women's target
sns.countplot(ax=ax[1], data=df_w_regseason, x='target')
ax[1].set_title('Women\'s Lower Team Win Count')
ax[1].set_xlabel('Target (Win: 1, Lose: 0)')
ax[1].set_ylabel('Count')

fig.tight_layout()
plt.show()


# Summarized lower team winning rate by season

df_m_yealy = df_m_regseason.groupby('Season')['target'].mean().reset_index()
df_m_yealy.rename(columns={'target': 'lower_team_win_rate'}, inplace=True)

df_w_yearly = df_w_regseason.groupby('Season')['target'].mean().reset_index()
df_w_yearly.rename(columns={'target': 'lower_team_win_rate'}, inplace=True)


fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

fig.suptitle('Lower Team Win Rate by Season')

# Men's lower team win rate by season
sns.lineplot(ax=ax[0], data=df_m_yealy, x='Season', y='lower_team_win_rate', marker='o')
ax[0].set_title('Men\'s Lower Team Win Rate by Season')
ax[0].set_xlabel('Season')
ax[0].set_ylabel('Win Rate')

# Women's lower team win rate by season
sns.lineplot(ax=ax[1], data=df_w_yearly, x='Season', y='lower_team_win_rate', marker='o')
ax[1].set_title('Women\'s Lower Team Win Rate by Season')
ax[1].set_xlabel('Season')
ax[1].set_ylabel('Win Rate')

fig.tight_layout()
plt.show()


# Summarized winning rate by location

df_m_loc = df_m_regseason.groupby('WLoc')['target'].mean().reset_index()
df_m_loc.rename(columns={'target': 'lower_team_win_rate'}, inplace=True)

df_w_loc = df_w_regseason.groupby('WLoc')['target'].mean().reset_index()
df_w_loc.rename(columns={'target': 'lower_team_win_rate'}, inplace=True)


fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

fig.suptitle('Win Rate by WLoc')

# Men's lower team win rate by location
sns.barplot(ax=ax[0], data=df_m_loc, x='WLoc', y='lower_team_win_rate')
ax[0].set_title('Men\'s Lower Team Win Rate by WLoc')
ax[0].set_xlabel('Location')
ax[0].set_ylabel('Win Rate')

# Women's lower team win rate by location
sns.barplot(ax=ax[1], data=df_w_loc, x='WLoc', y='lower_team_win_rate')
ax[1].set_title('Women\'s Lower Team Win Rate by WLoc')
ax[1].set_xlabel('Location')
ax[1].set_ylabel('Win Rate')

fig.tight_layout()
plt.show()


def assign_lower_team_lose_location(row):
    if row['lower_team'] == row['WTeamID']:
        return row['WLoc']
    else:
        if row['WLoc'] == 'H':
            return 'A'
        elif row['WLoc'] == 'A':
            return 'H'
        else:
            return 'N'
        
df_m_regseason['lower_team_WLoc'] = df_m_regseason.apply(assign_lower_team_lose_location, axis=1)
df_w_regseason['lower_team_WLoc'] = df_w_regseason.apply(assign_lower_team_lose_location, axis=1)


# Summarized lower team winning rate by location

df_m_loc = df_m_regseason.groupby('lower_team_WLoc')['target'].mean().reset_index()
df_m_loc.rename(columns={'target': 'lower_team_win_rate'}, inplace=True)

df_w_loc = df_w_regseason.groupby('lower_team_WLoc')['target'].mean().reset_index()
df_w_loc.rename(columns={'target': 'lower_team_win_rate'}, inplace=True)


fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

fig.suptitle('Lower Team Win Rate by Lower Team Location')

# Men's lower team win rate by lower team location
sns.barplot(ax=ax[0], data=df_m_loc, x='lower_team_WLoc', y='lower_team_win_rate')
ax[0].set_title('Men\'s Lower Team Win Rate by Lower Team Location')
ax[0].set_xlabel('Location')
ax[0].set_ylabel('Win Rate')

# Women's lower team win rate by lower team location
sns.barplot(ax=ax[1], data=df_w_loc, x='lower_team_WLoc', y='lower_team_win_rate')
ax[1].set_title('Women\'s Lower Team Win Rate by Lower Team Location')
ax[1].set_xlabel('Location')
ax[1].set_ylabel('Win Rate')

fig.tight_layout()
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

fig.suptitle('Winner vs Loser Score Distribution')

# Men's Regular Season Score Distribution
sns.histplot(ax=ax[0], data=df_m_regseason['WScore'], color='blue', label='Winning Score', kde=True, bins=30, alpha=0.3)
sns.histplot(ax=ax[0], data=df_m_regseason['LScore'], color='red', label='Losing Score', kde=True, bins=30, alpha=0.3)
ax[0].set_title('Men\'s Regular Season Score Distribution')
ax[0].set_xlabel('Score')
ax[0].set_ylabel('Frequency')
ax[0].legend()

# Women's Regular Season Score Distribution
sns.histplot(ax=ax[1], data=df_w_regseason['WScore'], color='blue', label='Winning Score', kde=True, bins=30, alpha=0.3)
sns.histplot(ax=ax[1], data=df_w_regseason['LScore'], color='red', label='Losing Score', kde=True, bins=30, alpha=0.3)
ax[1].set_title('Women\'s Regular Season Score Distribution')
ax[1].set_xlabel('Score')
ax[1].set_ylabel('Frequency')
ax[1].legend()

fig.tight_layout()
plt.show()


df_m_regseason['ScoreMargin'] = df_m_regseason['WScore'] - df_m_regseason['LScore']
df_w_regseason['ScoreMargin'] = df_w_regseason['WScore'] - df_w_regseason['LScore']

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

fig.suptitle('Score Margin Distribution')

# Men's Regular Season Score Margin Distribution
sns.histplot(ax=ax[0], data=df_m_regseason['ScoreMargin'], color='green', kde=True, bins=30, alpha=0.3)
ax[0].set_title('Men\'s Regular Season Score Margin Distribution')
ax[0].set_xlabel('Score Margin')
ax[0].set_ylabel('Frequency')

# Women's Regular Season Score Margin Distribution
sns.histplot(ax=ax[1], data=df_w_regseason['ScoreMargin'], color='green', kde=True, bins=30, alpha=0.3)
ax[1].set_title('Women\'s Regular Season Score Margin Distribution')
ax[1].set_xlabel('Score Margin')
ax[1].set_ylabel('Frequency')

fig.tight_layout()
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

fig.suptitle('Score Margin by DayNum')

# Men's Regular Season Score Margin by DayNum
sns.scatterplot(ax=ax[0], data=df_m_regseason, x='DayNum', y='ScoreMargin', alpha=0.3)
ax[0].set_title('Men\'s Regular Season Score Margin by DayNum')
ax[0].set_xlabel('DayNum')
ax[0].set_ylabel('Score Margin')

# Women's Regular Season Score Margin by DayNum
sns.scatterplot(ax=ax[1], data=df_w_regseason, x='DayNum', y='ScoreMargin', alpha=0.3)
ax[1].set_title('Women\'s Regular Season Score Margin by DayNum')
ax[1].set_xlabel('DayNum')
ax[1].set_ylabel('Score Margin')

plt.show()


df_m_regseason = pd.read_csv(os.path.join(data_dir, 'M' + regseason_file))
df_w_regseason = pd.read_csv(os.path.join(data_dir, 'W' + regseason_file))
df_subm = pd.read_csv(os.path.join(data_dir, submission_file))

df_regseson = pd.concat([df_m_regseason, df_w_regseason])


# Create cols for lower TeamID(TeamA) and upper TeamID(TeamB)
df_regseson['TeamA'] = df_regseson[['WTeamID', 'LTeamID']].min(axis=1)
df_regseson['TeamB'] = df_regseson[['WTeamID', 'LTeamID']].max(axis=1)

# Create target variable
df_regseson['target'] = df_regseson.apply(lambda x: 1 if x['WTeamID'] == x['TeamA'] else 0, axis=1)


# Winning rate per team (not from the lower team, just WTeamID)
team_wins = df_regseson.groupby('WTeamID').size()
team_losees = df_regseson.groupby('LTeamID').size()
team_counts = (team_wins + team_losees).fillna(0)
team_win_rate = (team_wins / team_counts).fillna(0)

df_regseson['TeamA_win_rate'] = df_regseson['TeamA'].map(team_win_rate)
df_regseson['TeamB_win_rate'] = df_regseson['TeamB'].map(team_win_rate)


X = df_regseson[['TeamA_win_rate', 'TeamB_win_rate']]
y = df_regseson['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print('Train Complete')


y_pred = model.predict_proba(X_test)[:, 1]
brier = brier_score_loss(y_test, y_pred)

print(f'Brier Score Loss: {brier}')


def extract_game_info(id_str):
    # Extract year and team_ids
    parts = id_str.split('_')
    year = int(parts[0])
    teamID1 = int(parts[1])
    teamID2 = int(parts[2])
    return year, teamID1, teamID2

df_subm[['Season', 'TeamA', 'TeamB']] = df_subm['ID'].apply(lambda x: extract_game_info(x)).to_list()


df_subm['TeamA_win_rate'] = df_subm['TeamA'].map(team_win_rate)
df_subm['TeamB_win_rate'] = df_subm['TeamB'].map(team_win_rate)


X_subm = df_subm[['TeamA_win_rate', 'TeamB_win_rate']]

df_subm['Pred'] = model.predict_proba(X_subm)[:, 1]


submission = df_subm[['ID', 'Pred']]

if is_kaggle:
    submission.to_csv('submission.csv', index=False)
    print('Submission file created')




