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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


# Replace with the actual file paths
m_teams = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')
w_teams = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WTeams.csv')
m_seasons = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MSeasons.csv')
m_tourney_seeds = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
m_regular_season = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')
m_tourney_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')


# Merge team names with tournament seeds
m_tourney_seeds = m_tourney_seeds.merge(m_teams[['TeamID', 'TeamName']], on='TeamID', how='left')


# Extract the numeric part of the seed (e.g., W01 -> 1)
m_tourney_seeds['SeedNumeric'] = m_tourney_seeds['Seed'].apply(lambda x: int(x[1:3]))


# Merge seed data with tournament results
# For each game, add the seeds of both teams
m_tourney_results = m_tourney_results.merge(
    m_tourney_seeds[['Season', 'TeamID', 'SeedNumeric']],
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={'SeedNumeric': 'WSeed'}).drop(columns=['TeamID'])

m_tourney_results = m_tourney_results.merge(
    m_tourney_seeds[['Season', 'TeamID', 'SeedNumeric']],
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={'SeedNumeric': 'LSeed'}).drop(columns=['TeamID'])


# Create a target variable: 1 if the first team wins, 0 otherwise
# For simplicity, we'll use the lower TeamID as the first team
m_tourney_results['Team1'] = np.where(
    m_tourney_results['WTeamID'] < m_tourney_results['LTeamID'],
    m_tourney_results['WTeamID'],
    m_tourney_results['LTeamID']
)
m_tourney_results['Team2'] = np.where(
    m_tourney_results['WTeamID'] < m_tourney_results['LTeamID'],
    m_tourney_results['LTeamID'],
    m_tourney_results['WTeamID']
)
m_tourney_results['Team1Seed'] = np.where(
    m_tourney_results['WTeamID'] < m_tourney_results['LTeamID'],
    m_tourney_results['WSeed'],
    m_tourney_results['LSeed']
)
m_tourney_results['Team2Seed'] = np.where(
    m_tourney_results['WTeamID'] < m_tourney_results['LTeamID'],
    m_tourney_results['LSeed'],
    m_tourney_results['WSeed']
)
m_tourney_results['Team1Win'] = np.where(
    m_tourney_results['WTeamID'] < m_tourney_results['LTeamID'],
    1,
    0
)


# Prepare features and target
features = ['Team1Seed', 'Team2Seed']
X = m_tourney_results[features]
y = m_tourney_results['Team1Win']


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Train a simple Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# Evaluate the model on the validation set
y_pred = model.predict_proba(X_val)[:, 1]
val_loss = log_loss(y_val, y_pred)
print(f'Validation Log Loss: {val_loss:.4f}')


# Make predictions for the 2025 tournament
# Create a DataFrame with all possible matchups
team_ids = m_teams['TeamID'].unique()
matchups = []
for i, team1 in enumerate(team_ids):
    for team2 in team_ids[i+1:]:  # Avoid duplicate matchups (e.g., Team1 vs Team2 and Team2 vs Team1)
        matchups.append((2025, team1, team2))

matchups_df = pd.DataFrame(matchups, columns=['Season', 'Team1', 'Team2'])


# Add seed data for 2025 (assuming seeds are available)
# For now, we'll use the average seed as a placeholder
matchups_df = matchups_df.merge(
    m_tourney_seeds[['Season', 'TeamID', 'SeedNumeric']],
    left_on=['Season', 'Team1'],
    right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={'SeedNumeric': 'Team1Seed'}).drop(columns=['TeamID'])

matchups_df = matchups_df.merge(
    m_tourney_seeds[['Season', 'TeamID', 'SeedNumeric']],
    left_on=['Season', 'Team2'],
    right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={'SeedNumeric': 'Team2Seed'}).drop(columns=['TeamID'])


# Fill missing seeds with the average seed
avg_seed = m_tourney_seeds['SeedNumeric'].mean()
matchups_df['Team1Seed'] = matchups_df['Team1Seed'].fillna(avg_seed)
matchups_df['Team2Seed'] = matchups_df['Team2Seed'].fillna(avg_seed)


# Predict probabilities
matchups_df['Pred'] = model.predict_proba(matchups_df[features])[:, 1]


# Prepare the submission file
submission = matchups_df[['Season', 'Team1', 'Team2', 'Pred']].copy()
submission['ID'] = submission['Season'].astype(str) + '_' + submission['Team1'].astype(str) + '_' + submission['Team2'].astype(str)
submission = submission[['ID', 'Pred']]


# Save the submission file
submission.to_csv('submission.csv', index=False)
print('Submission file saved as submission.csv')

