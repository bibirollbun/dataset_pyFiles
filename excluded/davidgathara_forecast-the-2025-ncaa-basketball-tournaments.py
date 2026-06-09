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


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# Load all data files
def load_data():
    # Base data
    seeds = pd.concat([
        pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv'),
        pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')
    ])
    
    games = pd.concat([
        pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv'),
        pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonDetailedResults.csv')
    ])
    
    rankings = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MMasseyOrdinals.csv')
    conferences = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/Conferences.csv')
    team_conferences = pd.concat([
        pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeamConferences.csv'),
        pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WTeamConferences.csv')
    ])
    slots = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeedRoundSlots.csv')
    
    return seeds, games, rankings, conferences, team_conferences, slots

seeds, games, rankings, conferences, team_conferences, slots = load_data()


# Clean seed data
seeds['SeedNum'] = seeds['Seed'].str.extract('(\d+)').astype(int)


# Get latest rankings for each team
latest_rankings = rankings.sort_values('RankingDayNum').groupby(['Season', 'TeamID']).last().reset_index()


# Enhanced feature engineering
def calculate_enhanced_stats(games_df):
    # Calculate advanced metrics
    games_df['WScoreMargin'] = games_df['WScore'] - games_df['LScore']
    games_df['WFG_Pct'] = games_df['WFGM'] / games_df['WFGA']
    games_df['LFG_Pct'] = games_df['LFGM'] / games_df['LFGA']
    
    # Winning team stats
    win_stats = games_df.groupby(['Season', 'WTeamID']).agg({
        'WScore': ['mean', 'std', 'max'],
        'LScore': ['mean', 'min'],
        'WScoreMargin': ['mean', 'std'],
        'WFG_Pct': 'mean',
        'WTO': 'mean',
        'WOR': 'mean'
    }).reset_index()
    win_stats.columns = ['Season', 'TeamID'] + [f'Win_{col[0]}_{col[1]}' if col[1] else f'Win_{col[0]}' 
                                               for col in win_stats.columns[2:]]
    
    # Losing team stats
    loss_stats = games_df.groupby(['Season', 'LTeamID']).agg({
        'LScore': ['mean', 'std', 'max'],
        'WScore': ['mean', 'min'],
        'LFG_Pct': 'mean',
        'LTO': 'mean',
        'LOR': 'mean'
    }).reset_index()
    loss_stats.columns = ['Season', 'TeamID'] + [f'Loss_{col[0]}_{col[1]}' if col[1] else f'Loss_{col[0]}' 
                                                for col in loss_stats.columns[2:]]
    
    # Merge stats
    team_stats = pd.merge(win_stats, loss_stats, on=['Season', 'TeamID'])
    
    # Add last 10 games performance
    last10 = games_df.sort_values(['Season', 'DayNum']).groupby(['Season', 'WTeamID']).tail(10)
    last10_wins = last10.groupby(['Season', 'WTeamID']).size().reset_index(name='Last10_Wins')
    last10_games = last10.groupby(['Season', 'WTeamID']).size().add(
        last10.groupby(['Season', 'LTeamID']).size(), fill_value=0).reset_index(name='Last10_Games')
    
    team_stats = team_stats.merge(
        last10_wins, 
        left_on=['Season', 'TeamID'], 
        right_on=['Season', 'WTeamID'],
        how='left'
    ).drop('WTeamID', axis=1)
    
    team_stats = team_stats.merge(
        last10_games,
        left_on=['Season', 'TeamID'],
        right_on=['Season', 'WTeamID'],
        how='left'
    ).drop('WTeamID', axis=1)
    
    team_stats['Last10_WinPct'] = team_stats['Last10_Wins'] / team_stats['Last10_Games']
    
    # Add conference strength
    conf_strength = team_conferences.merge(conferences, on='ConfAbbrev')
    conf_strength = conf_strength.merge(
        team_stats[['Season', 'TeamID', 'Win_WScore_mean']],
        on=['Season', 'TeamID']
    ).groupby(['Season', 'ConfAbbrev'])['Win_WScore_mean'].mean().reset_index()
    conf_strength.rename(columns={'Win_WScore_mean': 'ConfStrength'}, inplace=True)
    
    team_stats = team_stats.merge(
        team_conferences,
        left_on=['Season', 'TeamID'],
        right_on=['Season', 'TeamID']
    ).merge(
        conf_strength,
        on=['Season', 'ConfAbbrev']
    )
    
    return team_stats
team_stats = calculate_enhanced_stats(games)



# Prepare training data
def prepare_training_data(games_df, team_stats_df, seeds_df, rankings_df):
    data = []
    
    for _, row in games_df.iterrows():
        season = row['Season']
        t1, t2 = row['WTeamID'], row['LTeamID']
        
        try:
            t1_stats = team_stats_df[(team_stats_df['Season'] == season) & 
                                   (team_stats_df['TeamID'] == t1)].iloc[0]
            t2_stats = team_stats_df[(team_stats_df['Season'] == season) & 
                                   (team_stats_df['TeamID'] == t2)].iloc[0]
            
            t1_seed = seeds_df[(seeds_df['Season'] == season) & 
                              (seeds_df['TeamID'] == t1)]['SeedNum'].values[0]
            t2_seed = seeds_df[(seeds_df['Season'] == season) & 
                              (seeds_df['TeamID'] == t2)]['SeedNum'].values[0]
            
            t1_rank = rankings_df[(rankings_df['Season'] == season) & 
                                (rankings_df['TeamID'] == t1)]['OrdinalRank'].values[0]
            t2_rank = rankings_df[(rankings_df['Season'] == season) & 
                                (rankings_df['TeamID'] == t2)]['OrdinalRank'].values[0]
            
            features = {
                'SeedDiff': t1_seed - t2_seed,
                'RankDiff': t1_rank - t2_rank,
                'ConfStrengthDiff': t1_stats['ConfStrength'] - t2_stats['ConfStrength'],
                'Last10_WinPctDiff': t1_stats['Last10_WinPct'] - t2_stats['Last10_WinPct'],
                'FG_Pct_Diff': t1_stats['Win_WFG_Pct_mean'] - t2_stats['Loss_LFG_Pct_mean'],
                'TO_Diff': t1_stats['Win_WTO_mean'] - t2_stats['Loss_LTO_mean'],
                'OR_Diff': t1_stats['Win_WOR_mean'] - t2_stats['Loss_LOR_mean']
            }
            data.append({**features, 'Outcome': 1})
            
            # Reverse features
            reverse_features = {k: -v for k, v in features.items()}
            data.append({**reverse_features, 'Outcome': 0})
            
        except (IndexError, KeyError):
            continue
            
    return pd.DataFrame(data)


# Prepare training data from tournament games
tourney_games = pd.concat([
    pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyDetailedResults.csv'),
    pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyDetailedResults.csv')
])


train_data = prepare_training_data(tourney_games, team_stats, seeds, latest_rankings)


# Split data
X = train_data.drop('Outcome', axis=1)
y = train_data['Outcome']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Model optimization
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}


xgb = XGBClassifier(
    tree_method='auto',  
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)


grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)


best_xgb = grid_search.best_estimator_


# Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=7,
    n_jobs=-1,
    random_state=42
)


rf.fit(X_train, y_train)


# Ensemble model
ensemble = VotingClassifier(
    estimators=[('xgb', best_xgb), ('rf', rf)],
    voting='soft'
)


ensemble.fit(X_train, y_train)


# Evaluate
print(f"XGBoost Validation Accuracy: {best_xgb.score(X_val, y_val):.4f}")
print(f"Random Forest Validation Accuracy: {rf.score(X_val, y_val):.4f}")
print(f"Ensemble Validation Accuracy: {ensemble.score(X_val, y_val):.4f}")


# Prediction function for 2025
def predict_2025_matchups(model):
    teams_2025 = seeds[seeds['Season'] == 2025]['TeamID'].unique()
    matchups = [(t1, t2) for i, t1 in enumerate(teams_2025) 
              for t2 in teams_2025[i+1:] if t1 < t2]
    
    pred_data = []
    valid_matchups = []
    
    for t1, t2 in matchups:
        try:
            t1_stats = team_stats[(team_stats['Season'] == 2025) & 
                                (team_stats['TeamID'] == t1)].iloc[0]
            t2_stats = team_stats[(team_stats['Season'] == 2025) & 
                                (team_stats['TeamID'] == t2)].iloc[0]
            
            t1_seed = seeds[(seeds['Season'] == 2025) & 
                          (seeds['TeamID'] == t1)]['SeedNum'].values[0]
            t2_seed = seeds[(seeds['Season'] == 2025) & 
                          (seeds['TeamID'] == t2)]['SeedNum'].values[0]
            
            t1_rank = latest_rankings[(latest_rankings['Season'] == 2025) & 
                                    (latest_rankings['TeamID'] == t1)]['OrdinalRank'].values[0]
            t2_rank = latest_rankings[(latest_rankings['Season'] == 2025) & 
                                    (latest_rankings['TeamID'] == t2)]['OrdinalRank'].values[0]
            
            features = {
                'SeedDiff': t1_seed - t2_seed,
                'RankDiff': t1_rank - t2_rank,
                'ConfStrengthDiff': t1_stats['ConfStrength'] - t2_stats['ConfStrength'],
                'Last10_WinPctDiff': t1_stats['Last10_WinPct'] - t2_stats['Last10_WinPct'],
                'FG_Pct_Diff': t1_stats['Win_WFG_Pct_mean'] - t2_stats['Loss_LFG_Pct_mean'],
                'TO_Diff': t1_stats['Win_WTO_mean'] - t2_stats['Loss_LTO_mean'],
                'OR_Diff': t1_stats['Win_WOR_mean'] - t2_stats['Loss_LOR_mean']
            }
            pred_data.append(features)
            valid_matchups.append((t1, t2))
        except (IndexError, KeyError):
            continue
    
    X_pred = pd.DataFrame(pred_data)
    pred_probs = model.predict_proba(X_pred)[:, 1]
    
    submission = pd.DataFrame({
        'ID': [f"2025_{t1}_{t2}" for t1, t2 in valid_matchups],
        'Pred': pred_probs
    })
    
    return submission


# Generate final predictions
submission = predict_2025_matchups(ensemble)


submission.to_csv('submission.csv', index=False)
print(f"Submission created with {len(submission)} predictions")

