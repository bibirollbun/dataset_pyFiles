import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve


import warnings
warnings.filterwarnings("ignore")


# Load the datasets
data_dir = "/kaggle/input/march-machine-learning-mania-2025/"


# Load Data
def load_data():
    datasets = {
        "m_teams": "MTeams.csv",
        "m_tourney_results": "MNCAATourneyCompactResults.csv",
        "m_regular_results": "MRegularSeasonCompactResults.csv",
        "m_seeds": "MNCAATourneySeeds.csv",
        "w_teams": "WTeams.csv",
        "w_tourney_results": "WNCAATourneyCompactResults.csv",
        "w_regular_results": "WRegularSeasonCompactResults.csv",
        "w_seeds": "WNCAATourneySeeds.csv"
    }
    return {name: pd.read_csv(data_dir + file) for name, file in datasets.items()}

data = load_data()
# Preprocessing Seeds
def load_data():
    datasets = {
        "m_teams": "MTeams.csv",
        "m_tourney_results": "MNCAATourneyCompactResults.csv",
        "m_regular_results": "MRegularSeasonCompactResults.csv",
        "m_seeds": "MNCAATourneySeeds.csv",
    }
    return {name: pd.read_csv(data_dir + file) for name, file in datasets.items()}

data = load_data()

def prepare_seed(seeds):
    seeds['seed_int'] = seeds['Seed'].str.extract(r'(\d+)').astype(float)
    return seeds[['Season', 'TeamID', 'seed_int']]

data['m_seeds'] = prepare_seed(data['m_seeds'])

def prepare_data(regular_results, tourney_results):
    games = pd.concat([regular_results, tourney_results], ignore_index=True)
    games["WTeamWon"] = (games["WTeamID"] < games["LTeamID"]).astype(int)
    games["Team1ID"] = games[["WTeamID", "LTeamID"]].min(axis=1)
    games["Team2ID"] = games[["WTeamID", "LTeamID"]].max(axis=1)
    return games

data['m_games'] = prepare_data(data['m_regular_results'], data['m_tourney_results'])

def merge_seed_features(games, seeds):
    games = games.merge(seeds, left_on=['Season', 'Team1ID'], right_on=['Season', 'TeamID'], how='left').rename(columns={'seed_int': 'Team1Seed'})
    games = games.merge(seeds, left_on=['Season', 'Team2ID'], right_on=['Season', 'TeamID'], how='left').rename(columns={'seed_int': 'Team2Seed'})
    games.drop(columns=['TeamID_x', 'TeamID_y'], inplace=True)
    return games

data['m_games'] = merge_seed_features(data['m_games'], data['m_seeds'])
data['m_games']['SeedDiff'] = data['m_games']['Team1Seed'].fillna(18) - data['m_games']['Team2Seed'].fillna(18)


def add_features(games):
    games['ScoreDiff'] = games['WScore'] - games['LScore']
    games['WinPct'] = games.groupby('Team1ID')['WTeamWon'].transform(lambda x: x.expanding().mean())
    return games

data['m_games'] = add_features(data['m_games'])


def train_logistic_regression(games):
    features = ['SeedDiff', 'ScoreDiff', 'WinPct']
    games = games.dropna(subset=features + ['WTeamWon'])
    X = games[features]
    y = games['WTeamWon']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = GridSearchCV(LogisticRegression(), {'C': [0.01, 0.1, 1, 10]}, cv=5)
    model.fit(X_train, y_train)
    preds = model.best_estimator_.predict_proba(X_test)[:, 1]
    
    return model.best_estimator_, scaler, X_test, y_test, preds

data['model_m'], data['scaler_m'], X_test_m, y_test_m, preds_m = train_logistic_regression(data['m_games'])
print(f"Men's Brier Score: {brier_score_loss(y_test_m, preds_m):.4f}")



def plot_feature_distributions(games):
    plt.figure(figsize=(12, 5))
    sns.histplot(games['SeedDiff'], kde=True, bins=20)
    plt.title("Seed Difference Distribution")
    plt.show()
    
    plt.figure(figsize=(12, 5))
    sns.histplot(games['ScoreDiff'], kde=True, bins=20)
    plt.title("Score Difference Distribution")
    plt.show()
    
    plt.figure(figsize=(12, 5))
    sns.histplot(games['WinPct'].dropna(), kde=True, bins=20)
    plt.title("Win Percentage Distribution")
    plt.show()

plot_feature_distributions(data['m_games'])

def plot_calibration_curve(y_true, y_prob, title):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o')
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability")
    plt.title(title)
    plt.show()

plot_calibration_curve(y_test_m, preds_m, "Men's Calibration Curve")



import os
print(os.listdir(data_dir))
def predict_submission(submission, games, seeds, model, scaler):
    submission[['Season', 'Team1ID', 'Team2ID']] = submission['ID'].str.split('_', expand=True).astype(int)

    # Merge Seeds
    submission = submission.merge(seeds, left_on=['Season', 'Team1ID'], right_on=['Season', 'TeamID'], how='left').rename(columns={'seed_int': 'Team1Seed'}).drop(columns=['TeamID'])
    submission = submission.merge(seeds, left_on=['Season', 'Team2ID'], right_on=['Season', 'TeamID'], how='left').rename(columns={'seed_int': 'Team2Seed'}).drop(columns=['TeamID'])

    # Compute SeedDiff
    submission['SeedDiff'] = submission['Team1Seed'].fillna(18) - submission['Team2Seed'].fillna(18)

    # Compute ScoreDiff and WinPct from historical games
    avg_scores = games.groupby('WTeamID')['ScoreDiff'].mean().rename('AvgScoreDiff').reset_index()
    submission = submission.merge(avg_scores, left_on='Team1ID', right_on='WTeamID', how='left').drop(columns=['WTeamID'])
    submission = submission.merge(avg_scores, left_on='Team2ID', right_on='WTeamID', how='left', suffixes=('_T1', '_T2')).drop(columns=['WTeamID'])

    submission['ScoreDiff'] = submission['AvgScoreDiff_T1'].fillna(0) - submission['AvgScoreDiff_T2'].fillna(0)

    win_pct = games.groupby('WTeamID')['WTeamWon'].mean().rename('WinPct').reset_index()
    submission = submission.merge(win_pct, left_on='Team1ID', right_on='WTeamID', how='left').drop(columns=['WTeamID'])
    submission = submission.merge(win_pct, left_on='Team2ID', right_on='WTeamID', how='left', suffixes=('_T1', '_T2')).drop(columns=['WTeamID'])

    submission['WinPct'] = submission['WinPct_T1'].fillna(0) - submission['WinPct_T2'].fillna(0)

    # Prepare features
    X_submission = scaler.transform(submission[['SeedDiff', 'ScoreDiff', 'WinPct']])
    submission['Pred'] = model.predict_proba(X_submission)[:, 1]

    return submission[['ID', 'Pred']]
submission = pd.read_csv(data_dir + "SampleSubmissionStage1.csv")
m_submission = predict_submission(submission.copy(), data['m_games'], data['m_seeds'], data['model_m'], data['scaler_m'])
m_submission.to_csv('submission.csv', index=False)
print(m_submission.head())


