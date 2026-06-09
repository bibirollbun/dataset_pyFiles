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


# Load key files
teams = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')
seeds = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
regular_season = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')
tourney_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')


data_dict = {
    "teams": teams,
    "seeds": seeds,
    "regular_season": regular_season,
    "tourney_results": tourney_results
}


for name, data_set in data_dict.items():
    print(f"The shape of {name} dataset is {data_set.shape}")


for name, data_set in data_dict.items():
    print(f"The  {name} dataset has {data_set.duplicated().sum()} values")


for name, data_set in data_dict.items():
    print(f"The {name} dataset has {data_set.isnull().sum()} values")
    print("________________________")


for name, data_set in data_dict.items():
    print(data_set.describe().to_string())
    print("==================================================")


for name, data_set in data_dict.items():
    print(data_set.info())
    print("==================================================")


for name, data_set in data_dict.items():
    print(data_set.head(5))
    print("==================================================")


for name, data_set in data_dict.items():
    print(f"The {name} has {len(data_set.columns)} columns: {data_set.columns}")


# Compute statistics for each team
team_stats = regular_season.groupby('WTeamID').agg(
    total_wins=('WTeamID', 'count'),
    avg_win_score=('WScore', 'mean'),
    avg_loss_score=('LScore', 'mean'),
).reset_index()

# Compute total games played
games_played = regular_season.groupby('WTeamID')['WTeamID'].count() + regular_season.groupby('LTeamID')['LTeamID'].count()
games_played = games_played.reset_index().rename(columns={0: 'total_games'})

# Merge statistics
team_stats = team_stats.merge(games_played, left_on='WTeamID', right_on='WTeamID')
team_stats['win_rate'] = team_stats['total_wins'] / team_stats['total_games']
team_stats['avg_point_margin'] = team_stats['avg_win_score'] - team_stats['avg_loss_score']

# Rename WTeamID to TeamID for merging later
team_stats.rename(columns={'WTeamID': 'TeamID'}, inplace=True)
print(team_stats.head().to_string())


# Convert Seed (e.g., 'W01', 'X16') into a numeric feature (1-16)
seeds['SeedNum'] = seeds['Seed'].str.extract('(\d+)').astype(int)
seeds.drop(columns=['Seed'], inplace=True)

print(seeds.head())


# Create dataframe with matchups
matches = tourney_results[['Season', 'WTeamID', 'LTeamID']]
matches = matches.merge(seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])
matches = matches.rename(columns={'SeedNum': 'Seed_W'})
matches = matches.drop(columns=['TeamID'])

matches = matches.merge(seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'])
matches = matches.rename(columns={'SeedNum': 'Seed_L'})
matches = matches.drop(columns=['TeamID'])

# Merge team stats
matches = matches.merge(team_stats, left_on='WTeamID', right_on='TeamID').rename(
    columns={'win_rate': 'win_rate_W', 'avg_point_margin': 'margin_W'}
)
matches = matches.drop(columns=['TeamID'])

matches = matches.merge(team_stats, left_on='LTeamID', right_on='TeamID').rename(
    columns={'win_rate': 'win_rate_L', 'avg_point_margin': 'margin_L'}
)
matches = matches.drop(columns=['TeamID'])

# Create difference features
matches['seed_diff'] = matches['Seed_W'] - matches['Seed_L']
matches['win_rate_diff'] = matches['win_rate_W'] - matches['win_rate_L']
matches['margin_diff'] = matches['margin_W'] - matches['margin_L']

# Target variable (1 if WTeam won, 0 otherwise)
matches['target'] = 1
print(matches.head().to_string())


# Create reverse matches (LTeam wins)
reversed_matches = matches.copy()
reversed_matches['target'] = 0  # Losing team wins
reversed_matches = reversed_matches.rename(
    columns={
        'WTeamID': 'LTeamID', 'LTeamID': 'WTeamID',
        'Seed_W': 'Seed_L', 'Seed_L': 'Seed_W',
        'win_rate_W': 'win_rate_L', 'win_rate_L': 'win_rate_W',
        'margin_W': 'margin_L', 'margin_L': 'margin_W',
    }
)


# Flip the feature differences
reversed_matches['seed_diff'] = -reversed_matches['seed_diff']
reversed_matches['win_rate_diff'] = -reversed_matches['win_rate_diff']
reversed_matches['margin_diff'] = -reversed_matches['margin_diff']


# Combine both versions
final_data = pd.concat([matches, reversed_matches])


# Select final features
X = final_data[['seed_diff', 'win_rate_diff', 'margin_diff']]
y = final_data['target']

print(X.head(), y.head())


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05)
xgb_model.fit(X_train, y_train)

# Evaluate
y_pred_xgb = xgb_model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb)}")

