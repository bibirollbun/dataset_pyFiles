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
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Load Data
teams = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')
seasons = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MSeasons.csv')
seeds = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')

# Feature Engineering
def preprocess_data(results, seeds):
    results['ScoreDiff'] = results['WScore'] - results['LScore']
    team_stats = results.groupby('WTeamID')['ScoreDiff'].mean().reset_index()
    team_stats.columns = ['TeamID', 'AvgScoreDiff']
    
    seeds['SeedValue'] = seeds['Seed'].str.extract(r'(\d+)').astype(int)
    seeds = seeds[['Season', 'TeamID', 'SeedValue']]
    
    return team_stats, seeds

team_stats, seeds = preprocess_data(results, seeds)

# Creating Training Data
def create_training_data(results, team_stats, seeds):
    matchups = results[['Season', 'WTeamID', 'LTeamID']]
    team_stats_W = team_stats.rename(columns={'AvgScoreDiff': 'AvgScoreDiff_W', 'TeamID': 'WTeamID'})
    team_stats_L = team_stats.rename(columns={'AvgScoreDiff': 'AvgScoreDiff_L', 'TeamID': 'LTeamID'})
    matchups = matchups.merge(team_stats_W, on='WTeamID', how='left')
    matchups = matchups.merge(team_stats_L, on='LTeamID', how='left')
    
    seeds_W = seeds.rename(columns={'SeedValue': 'SeedValue_W', 'TeamID': 'WTeamID'})
    seeds_L = seeds.rename(columns={'SeedValue': 'SeedValue_L', 'TeamID': 'LTeamID'})
    matchups = matchups.merge(seeds_W, on=['Season', 'WTeamID'], how='left')
    matchups = matchups.merge(seeds_L, on=['Season', 'LTeamID'], how='left')
    
    matchups['SeedDiff'] = matchups['SeedValue_W'] - matchups['SeedValue_L']
    matchups['ScoreDiff'] = matchups['AvgScoreDiff_W'] - matchups['AvgScoreDiff_L']
    matchups['Result'] = 1
    
    matchups_flipped = matchups.copy()
    matchups_flipped = matchups_flipped.rename(columns={
        'WTeamID': 'LTeamID', 'LTeamID': 'WTeamID',
        'SeedValue_W': 'SeedValue_L', 'SeedValue_L': 'SeedValue_W',
        'AvgScoreDiff_W': 'AvgScoreDiff_L', 'AvgScoreDiff_L': 'AvgScoreDiff_W'
    })
    matchups_flipped['SeedDiff'] = -matchups_flipped['SeedDiff']
    matchups_flipped['ScoreDiff'] = -matchups_flipped['ScoreDiff']
    matchups_flipped['Result'] = 0
    
    final_dataset = pd.concat([matchups, matchups_flipped], axis=0).reset_index(drop=True)
    features = final_dataset[['SeedDiff', 'ScoreDiff']]
    labels = final_dataset['Result']
    return features, labels

X, y = create_training_data(results, team_stats, seeds)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning
grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
}

xgb_model = xgb.XGBClassifier(random_state=42)
clf = GridSearchCV(xgb_model, grid, scoring='accuracy', cv=5, n_jobs=-1)
clf.fit(X_train, y_train)

# Best Model Evaluation
y_pred = clf.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Optimized Accuracy: {accuracy:.4f}')

# Submission
df_submission = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv')
df_submission[['Season', 'Team1', 'Team2']] = df_submission['ID'].str.split('_', expand=True)
df_submission['Season'] = df_submission['Season'].astype(int)
df_submission['Team1'] = df_submission['Team1'].astype(int)
df_submission['Team2'] = df_submission['Team2'].astype(int)

df_submission = df_submission.merge(team_stats.rename(columns={'TeamID': 'Team1'}), on='Team1', how='left')
df_submission = df_submission.merge(team_stats.rename(columns={'TeamID': 'Team2'}), on='Team2', how='left', suffixes=('_T1', '_T2'))

df_submission = df_submission.merge(seeds.rename(columns={'TeamID': 'Team1'}), on=['Season', 'Team1'], how='left')
df_submission = df_submission.merge(seeds.rename(columns={'TeamID': 'Team2'}), on=['Season', 'Team2'], how='left', suffixes=('_T1', '_T2'))

df_submission['SeedDiff'] = df_submission['SeedValue_T1'] - df_submission['SeedValue_T2']
df_submission['ScoreDiff'] = df_submission['AvgScoreDiff_T1'] - df_submission['AvgScoreDiff_T2']
X_submission = df_submission[['SeedDiff', 'ScoreDiff']].fillna(0)

df_submission['Pred'] = clf.best_estimator_.predict_proba(X_submission)[:, 1]
df_submission[['ID', 'Pred']].to_csv('submission.csv', index=False)

print("✅ Optimized submission file saved as submission.csv")

