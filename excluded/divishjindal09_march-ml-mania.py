def project_intro():
    print("""
    ==================================================
    Project: Building a Predictive Model for NCAA March Madness Outcomes
    ==================================================

    Project Objective:
    - Build a machine learning model to predict the outcomes of NCAA March Madness games.
    - Use historical data to train and evaluate the model.

    Data Used:
    - MTeams.csv: Teams data.
    - MSeasons.csv: Seasons data.
    - MNCAATourneySeeds.csv: Tournament seeding data.
    - MRegularSeasonCompactResults.csv: Regular season game results.
    - MNCAATourneyCompactResults.csv: Tournament game results.
    - MRegularSeasonDetailedResults.csv: Detailed game statistics.
    - MMasseyOrdinals.csv: Team rankings.

    Workflow:
    1. Load the data.
    2. Perform exploratory data analysis (EDA).
    3. Conduct feature engineering.
    4. Build the model using LightGBM.
    5. Generate a submission file for the competition.

    Tools Used:
    - Python, Pandas, Matplotlib, Seaborn, LightGBM, Scikit-learn.

    Expected Outcomes:
    - Accurate predictions for game outcomes.
    - A Log Loss of less than 0.45.
    - A ready-to-submit file for the competition.
    """)

# Run the introduction
project_intro()




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
import seaborn as sns
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

teams = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')
seasons = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MSeasons.csv') 
seeds = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv') 
regular_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')
tourney_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv') 
detailed_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv')  
massey_ordinals = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MMasseyOrdinals.csv')  


display(teams.head())
display(seasons.head())
display(seeds.head())
display(regular_results.head())
display(tourney_results.head())
display(detailed_results.head())
display(massey_ordinals.head())


#Getting Info
display(teams.info())
display(seasons.info())
display(seeds.info())
display(regular_results.info())
display(tourney_results.info())
display(detailed_results.info())
display(massey_ordinals.info())


# Checking NULL values
display(teams.isnull().sum())
display(seasons.isnull().sum())
display(seeds.isnull().sum())
display(regular_results.isnull().sum())
display(tourney_results.isnull().sum())
display(detailed_results.isnull().sum())
display(massey_ordinals.isnull().sum())


# Checking Missing value
display(teams.isnull().sum())
display(seasons.isnull().sum())
display(seeds.isnull().sum())
display(regular_results.isnull().sum())
display(tourney_results.isnull().sum())
display(detailed_results.isnull().sum())
display(massey_ordinals.isnull().sum())


#plotting  curve between Winning Points & Losing Points
plt.figure(figsize=(12, 6))
sns.histplot(regular_results['WScore'], bins=30, kde=True, label='Winning Team Score')
sns.histplot(regular_results['LScore'], bins=30, kde=True, color='red', label='Losing Team Score')
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Distribution of Winning & Losing Scores")
plt.legend()
plt.show()


win_counts = regular_results['WTeamID'].value_counts()
loss_counts = regular_results['LTeamID'].value_counts()
total_games = win_counts.add(loss_counts, fill_value=0)
win_ratio = win_counts / total_games
win_ratio = win_ratio.sort_values(ascending=False)
    
plt.figure(figsize=(12, 8))
win_ratio.head(20).plot(kind='bar', color='red')
plt.title('Top 20 teams in terms of participation rate')
plt.show()


away_neutral_games = regular_results[regular_results['WLoc'].isin(['A', 'N'])]
    
away_wins = away_neutral_games['WTeamID'].value_counts()
total_away_games = away_neutral_games['WTeamID'].value_counts() + away_neutral_games['LTeamID'].value_counts()
away_win_ratio = (away_wins / total_away_games).sort_values(ascending=False)
    
plt.figure(figsize=(10, 6))
away_win_ratio.head(10).plot(kind='bar')
plt.title("Top 10 Teams in Away/Neutral Games (Win Ratio)")
plt.xlabel("Team ID")
plt.ylabel("Win Ratio")
plt.show()


# Feature Engineering
def feature_engineering(regular_results, detailed_results, massey_ordinals):
    team_stats = regular_results.groupby('WTeamID').agg({'WScore': ['mean', 'count']})
    team_stats.columns = ['AvgPointsScored', 'GamesWon']
    team_stats['AvgPointsAllowed'] = regular_results.groupby('LTeamID')['LScore'].mean()
    team_stats['GamesLost'] = regular_results.groupby('LTeamID')['LScore'].count()
    team_stats['TotalGames'] = team_stats['GamesWon'] + team_stats['GamesLost']
    team_stats['WinRatio'] = team_stats['GamesWon'] / team_stats['TotalGames']
    
    detailed_results['OffensiveEfficiency'] = (detailed_results['WFGM'] + 1.5 * detailed_results['WFGM3']) / detailed_results['WFGA']
    detailed_results['DefensiveEfficiency'] = (detailed_results['LFGM'] + 1.5 * detailed_results['LFGM3']) / detailed_results['LFGA']
    
    latest_rankings = massey_ordinals[massey_ordinals['RankingDayNum'] == 133]
    team_stats = team_stats.merge(latest_rankings[['TeamID', 'OrdinalRank']], left_index=True, right_on='TeamID', how='left')
    
    return team_stats


#BUILDING MODEL
def build_model(train_data):
    # Split Data
    X = train_data.drop(['WinRatio'], axis=1)
    y = train_data['WinRatio']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # LightGBM Model
    params = {
        'objective': 'binary',
        'metric': 'logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=1000, early_stopping_rounds=50)
    
    return model


def generate_submission(model, teams, seasons, seeds, output_path="submission.csv"):

    tourney_teams = seeds[seeds['Season'] == 2025]['TeamID'].unique()
    
    from itertools import combinations
    matchups = list(combinations(tourney_teams, 2))
    
    submission_data = []
    for team1, team2 in matchups:
        if team1 < team2:
            matchup_id = f"2025_{team1}_{team2}"
        else:
            matchup_id = f"2025_{team2}_{team1}"
        
        team1_stats = teams[teams['TeamID'] == team1].iloc[0]
        team2_stats = teams[teams['TeamID'] == team2].iloc[0]
        
        features = {
            'PointDiff': team1_stats['AvgPointsScored'] - team2_stats['AvgPointsAllowed'],
            'WinRatioDiff': team1_stats['WinRatio'] - team2_stats['WinRatio'],
            'RankDiff': team1_stats['OrdinalRank'] - team2_stats['OrdinalRank']
        }
        
        submission_data.append([matchup_id, features])
    
    submission_df = pd.DataFrame(submission_data, columns=['ID', 'Features'])
    
    X_submission = pd.DataFrame(submission_df['Features'].tolist())
    submission_df['Pred'] = model.predict(X_submission)
    
    submission_df[['ID', 'Pred']].to_csv(output_path, index=False)
    print(f"تم حفظ ملف التسليم في: {output_path}")



def main():
    # load data
    teams, seasons, seeds, regular_results, tourney_results, detailed_results, massey_ordinals = load_data()
    
    # EDA
    perform_eda(teams, seasons, seeds, regular_results, tourney_results, detailed_results, massey_ordinals)
    
    # feature_engineering
    team_stats = feature_engineering(regular_results, detailed_results, massey_ordinals)
    
    # build_model
    model = build_model(team_stats)
    
    # predict
    y_pred = model.predict(team_stats.drop(['WinRatio'], axis=1))
    print(f"Log Loss: {log_loss(team_stats['WinRatio'], y_pred)}")
    
    # generate_submission
    generate_submission(model, teams, seasons, seeds)

