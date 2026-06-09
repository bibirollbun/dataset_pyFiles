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


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import lightgbm as lgb

DATA_PATH = "/kaggle/input/march-machine-learning-mania-2025"

# Load Data
def load_data():
    teams = pd.read_csv(f"{DATA_PATH}/MTeams.csv")
    seasons = pd.read_csv(f"{DATA_PATH}/MSeasons.csv") 
    seeds = pd.read_csv(f"{DATA_PATH}/MNCAATourneySeeds.csv") 
    regular_results = pd.read_csv(f"{DATA_PATH}/MRegularSeasonCompactResults.csv")
    tourney_results = pd.read_csv(f"{DATA_PATH}/MNCAATourneyCompactResults.csv") 
    detailed_results = pd.read_csv(f"{DATA_PATH}/MRegularSeasonDetailedResults.csv")  
    massey_ordinals = pd.read_csv(f"{DATA_PATH}/MMasseyOrdinals.csv")  
    
    return teams, seasons, seeds, regular_results, tourney_results, detailed_results, massey_ordinals

teams, seasons, seeds, regular_results, tourney_results, detailed_results, massey_ordinals = load_data()

# Show 5 first columns
display(teams.head())
display(seasons.head())
display(seeds.head())
display(regular_results.head())
display(tourney_results.head())
display(detailed_results.head())
display(massey_ordinals.head())


display(teams.info())
display(seasons.info())
display(seeds.info())
display(regular_results.info())
display(tourney_results.info())
display(detailed_results.info())
display(massey_ordinals.info())


display(teams.describe())
display(seasons.describe())
display(seeds.describe())
display(regular_results.describe())
display(tourney_results.describe())
display(detailed_results.describe())
display(massey_ordinals.describe())


display(teams.isnull().sum())
display(seasons.isnull().sum())
display(seeds.isnull().sum())
display(regular_results.isnull().sum())
display(tourney_results.isnull().sum())
display(detailed_results.isnull().sum())
display(massey_ordinals.isnull().sum())


display(teams.duplicated().sum())
display(seasons.duplicated().sum())
display(seeds.duplicated().sum())
display(regular_results.duplicated().sum())
display(tourney_results.duplicated().sum())
display(detailed_results.duplicated().sum())
display(massey_ordinals.duplicated().sum())


plt.figure(figsize=(10, 6))
sns.histplot(regular_results['WScore'], kde=True, label='Winning points')
sns.histplot(regular_results['LScore'], kde=True, label='Losing points')
plt.title('Distribution of points in matches')
plt.legend()
plt.show()


regular_results['PointDiff'] = regular_results['WScore'] - regular_results['LScore']
plt.figure(figsize=(10, 6))
sns.histplot(regular_results['PointDiff'], kde=True)
plt.title('Distribution of differences between points')
plt.show()


win_counts = regular_results['WTeamID'].value_counts()
loss_counts = regular_results['LTeamID'].value_counts()
total_games = win_counts.add(loss_counts, fill_value=0)
win_ratio = win_counts / total_games
win_ratio = win_ratio.sort_values(ascending=False)
    
plt.figure(figsize=(12, 8))
win_ratio.head(20).plot(kind='bar')
plt.title('Top 20 teams in terms of participation rate')
plt.show


merged_data = tourney_results.merge(seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])
merged_data = merged_data.merge(seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], suffixes=('_W', '_L'))
    
seed_win_ratios = merged_data.groupby(['Seed_W', 'Seed_L']).size().unstack().fillna(0)
seed_win_ratios = seed_win_ratios.div(seed_win_ratios.sum(axis=1), axis=0)
    
plt.figure(figsize=(12, 8))
sns.heatmap(seed_win_ratios, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Win Ratios by Seed Matchups")
plt.xlabel("Losing Seed")
plt.ylabel("Winning Seed")
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




