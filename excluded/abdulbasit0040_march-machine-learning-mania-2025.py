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


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import brier_score_loss, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tqdm import tqdm
import itertools

# Set up visualizations
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


# Load all datasets
# Data Section 1 - The Basics
m_teams = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')
w_teams = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WTeams.csv')
m_seasons = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MSeasons.csv')
w_seasons = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WSeasons.csv')
m_tourney_seeds = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
w_tourney_seeds = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')
m_reg = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')
w_reg = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv')
m_tourney = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
w_tourney = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv')

# Data Section 2 - Team Box Scores
m_reg_detailed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv')
w_reg_detailed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonDetailedResults.csv')
m_tourney_detailed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyDetailedResults.csv')
w_tourney_detailed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyDetailedResults.csv')

# Data Section 3 - Geography
cities = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/Cities.csv')
m_game_cities = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MGameCities.csv')
w_game_cities = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WGameCities.csv')

# Data Section 4 - Public Rankings
m_massey_ordinals = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MMasseyOrdinals.csv')

# Data Section 5 - Supplements
m_team_coaches = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeamCoaches.csv')
conferences = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/Conferences.csv')
m_team_conferences = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeamConferences.csv')
w_team_conferences = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WTeamConferences.csv')
m_conf_tourney_games = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MConferenceTourneyGames.csv')
w_conf_tourney_games = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WConferenceTourneyGames.csv')
m_secondary_tourney_teams = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MSecondaryTourneyTeams.csv')
w_secondary_tourney_teams = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WSecondaryTourneyTeams.csv')
m_secondary_tourney_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MSecondaryTourneyCompactResults.csv')
w_secondary_tourney_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WSecondaryTourneyCompactResults.csv')
m_team_spellings = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeamSpellings.csv')
w_team_spellings = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WTeamSpellings.csv')
m_tourney_slots = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySlots.csv')
w_tourney_slots = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySlots.csv')
m_tourney_seed_round_slots = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeedRoundSlots.csv')



# Display first few rows of datasets
print("MTeams Sample:")
print(m_teams.head())
print("\nWTeams Sample:")
print(w_teams.head())
print("\nMRegularSeasonCompactResults Sample:")
print(m_reg.head())
print("\nWNCAATourneyCompactResults Sample:")
print(w_tourney.head())


# Combine men's and women's data where applicable
def combine_data(df1, df2, gender_col):
    df1['Gender'] = gender_col[0]
    df2['Gender'] = gender_col[1]
    return pd.concat([df1, df2])


# Combine regular season and tournament data
reg_data = combine_data(m_reg, w_reg, ['M', 'W'])
tourney_data = combine_data(m_tourney, w_tourney, ['M', 'W'])


# Combine detailed results
reg_detailed = combine_data(m_reg_detailed, w_reg_detailed, ['M', 'W'])
tourney_detailed = combine_data(m_tourney_detailed, w_tourney_detailed, ['M', 'W'])



# Combine team data
teams = combine_data(m_teams, w_teams, ['M', 'W'])


# Combine seeds data
seeds = combine_data(m_tourney_seeds, w_tourney_seeds, ['M', 'W'])


# Feature Engineering
def compute_team_stats(data):
    team_stats = {}
    for _, row in data.iterrows():
        season = row['Season']
        w_team = row['WTeamID']
        l_team = row['LTeamID']
        w_score = row['WScore']
        l_score = row['LScore']
        
        # Update stats for winning team
        if (season, w_team) not in team_stats:
            team_stats[(season, w_team)] = {'wins': 0, 'losses': 0, 'points_for': 0, 'points_against': 0}
        team_stats[(season, w_team)]['wins'] += 1
        team_stats[(season, w_team)]['points_for'] += w_score
        team_stats[(season, w_team)]['points_against'] += l_score
        
        # Update stats for losing team
        if (season, l_team) not in team_stats:
            team_stats[(season, l_team)] = {'wins': 0, 'losses': 0, 'points_for': 0, 'points_against': 0}
        team_stats[(season, l_team)]['losses'] += 1
        team_stats[(season, l_team)]['points_for'] += l_score
        team_stats[(season, l_team)]['points_against'] += w_score
    
    # Convert to DataFrame
    stats_df = pd.DataFrame.from_dict(team_stats, orient='index')
    stats_df.reset_index(inplace=True)
    stats_df.rename(columns={'level_0': 'Season', 'level_1': 'TeamID'}, inplace=True)
    return stats_df


team_stats = compute_team_stats(reg_data)


# Merge team stats with seeds
data = pd.merge(team_stats, seeds, on=['Season', 'TeamID'], how='left')


# Create features
data['win_pct'] = data['wins'] / (data['wins'] + data['losses'])
data['points_diff'] = data['points_for'] - data['points_against']
data['seed'] = data['Seed'].apply(lambda x: int(x[1:3]) if pd.notnull(x) else 16)  # Handle missing seeds


# Visualize feature distributions
plt.figure(figsize=(12, 6))
sns.histplot(data['win_pct'], bins=30, kde=True, color='blue')
plt.title('Win Percentage Distribution')
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(data['points_diff'], bins=30, kde=True, color='green')
plt.title('Points Difference Distribution')
plt.show()


# Prepare training data
X = data[['win_pct', 'points_diff', 'seed']]
y = data['seed']  # Use seed as a proxy for team strength


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Map class labels to start from 0
y_train_mapped = y_train - 1
y_test_mapped = y_test - 1

# Hyperparameter tuning for XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}


xgb_model = XGBClassifier(objective='multi:softprob', num_class=16, random_state=42)
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_log_loss')
grid_search.fit(X_train, y_train_mapped)  # Use mapped labels for training


# Best model
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")


from sklearn.metrics import log_loss


# Evaluate model
y_pred = best_model.predict_proba(X_test)

# Log Loss (Cross-Entropy Loss)
log_loss_score = log_loss(y_test_mapped, y_pred)
print(f'Log Loss: {log_loss_score}')

# ROC AUC Score (Multi-Class)
roc_auc = roc_auc_score(y_test_mapped, y_pred, multi_class='ovr')
print(f'ROC AUC Score: {roc_auc}')


# Confusion Matrix
y_pred_labels = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_test_mapped, y_pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Generate predictions for 2025
def generate_predictions(teams, model):
    predictions = []
    for team1, team2 in tqdm(itertools.combinations(teams['TeamID'], 2)):
        if team1 < team2:
            team1_stats = data[(data['TeamID'] == team1) & (data['Season'] == 2025)]
            team2_stats = data[(data['TeamID'] == team2) & (data['Season'] == 2025)]
            if not team1_stats.empty and not team2_stats.empty:
                features = np.array([team1_stats['win_pct'].values[0] - team2_stats['win_pct'].values[0],
                                     team1_stats['points_diff'].values[0] - team2_stats['points_diff'].values[0],
                                     team1_stats['seed'].values[0] - team2_stats['seed'].values[0]])
                pred = model.predict_proba(features.reshape(1, -1))[0][1]
                predictions.append({'ID': f'2025_{team1}_{team2}', 'Pred': pred})
    return pd.DataFrame(predictions)


# Get 2025 teams
teams_2025 = teams[teams['LastD1Season'] >= 2025]


# Generate predictions
predictions = generate_predictions(teams_2025, best_model)


# Save submission file
predictions.to_csv('submission.csv', index=False)

