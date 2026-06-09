import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# datasets
regular_results = pd.read_csv("/kaggle/input/ml-model/MRegularSeasonCompactResults.csv")
tourney_results = pd.read_csv("/kaggle/input/ml-model/MNCAATourneyCompactResults.csv")
seeds = pd.read_csv("/kaggle/input/ml-model/MNCAATourneySeeds.csv")
teams = pd.read_csv("/kaggle/input/ml-model/MTeams.csv")
sample_submission = pd.read_csv("/kaggle/input/ml-model/SampleSubmissionStage1.csv")


# Team Stats
regular_results['ScoreDiff'] = regular_results['WScore'] - regular_results['LScore']

wins = regular_results.groupby('WTeamID').agg(
    Wins=('WTeamID', 'count'), 
    AvgWinMargin=('ScoreDiff', 'mean')
).reset_index()

losses = regular_results.groupby('LTeamID').agg(
    Losses=('LTeamID', 'count')
).reset_index()

# Merge win/loss stats
team_stats = pd.merge(wins, losses, left_on='WTeamID', right_on='LTeamID', how='outer').fillna(0)
team_stats['TeamID'] = team_stats['WTeamID'].fillna(team_stats['LTeamID']).astype(int)
team_stats = team_stats[['TeamID', 'Wins', 'Losses', 'AvgWinMargin']]
team_stats['Win%'] = team_stats['Wins'] / (team_stats['Wins'] + team_stats['Losses'])


team_stats_named = pd.merge(team_stats, teams, on='TeamID', how='left')


sns.histplot(team_stats['Wins'], bins=30, kde=True)
plt.title("Distribution of Wins")
plt.xlabel("Number of Wins")
plt.show()


#Distribution of Wins
sns.histplot(team_stats['Wins'], bins=30, kde=True)
plt.title("Distribution of Wins")
plt.xlabel("Number of Wins")
plt.show()


#Distribution of Win Percentages
sns.histplot(team_stats['Win%'], bins=30, kde=True, color='orange')
plt.title("Win Percentage Distribution")
plt.xlabel("Win %")
plt.show()


# Distribution of Average Win Margin
sns.histplot(team_stats['AvgWinMargin'], bins=30, kde=True, color='green')
plt.title("Average Win Margin")
plt.xlabel("Score Margin")
plt.show()


#Wins vs Avg Win Margin
sns.scatterplot(data=team_stats, x='Wins', y='AvgWinMargin')
plt.title("Wins vs Average Win Margin")
plt.xlabel("Wins")
plt.ylabel("Avg Win Margin")
plt.show()


# Count total wins by year
wins_by_year = regular_results.groupby('Season')['WTeamID'].count()

plt.figure(figsize=(12, 6))
sns.lineplot(x=wins_by_year.index, y=wins_by_year.values)
plt.title("Number of Wins per Season")
plt.xlabel("Season")
plt.ylabel("Number of Wins")
plt.grid(True)
plt.show()


regular_results['ScoreDiff'] = regular_results['WScore'] - regular_results['LScore']

plt.figure(figsize=(10, 5))
sns.histplot(regular_results['ScoreDiff'], bins=30, kde=True)
plt.title("Distribution of Winning Margins")
plt.xlabel("Score Difference")
plt.ylabel("Frequency")
plt.show()


seeds['SeedNum'] = seeds['Seed'].str.extract('(\d+)').astype(int)


def create_training_data(tourney_results, seeds, team_stats):
    df = tourney_results.copy()

    # Merge seed info
    df = df.merge(seeds.rename(columns={'TeamID': 'WTeamID', 'SeedNum': 'WSeed'}), on=['Season', 'WTeamID'])
    df = df.merge(seeds.rename(columns={'TeamID': 'LTeamID', 'SeedNum': 'LSeed'}), on=['Season', 'LTeamID'])

    # Merge team stats
    df = df.merge(team_stats.rename(columns={'TeamID': 'WTeamID'}), on='WTeamID')
    df = df.merge(team_stats.rename(columns={'TeamID': 'LTeamID'}), on='LTeamID', suffixes=('_W', '_L'))

    # Feature differences
    df['SeedDiff'] = df['WSeed'] - df['LSeed']
    df['WinPctDiff'] = df['Win%_W'] - df['Win%_L']
    df['MarginDiff'] = df['AvgWinMargin_W'] - df['AvgWinMargin_L']

    # Features and label
    features = df[['SeedDiff', 'WinPctDiff', 'MarginDiff']]
    labels = [1] * len(features)  # 1 = WTeam wins

    # Flip and add the other side (to balance dataset)
    df_flipped = df.copy()
    df_flipped['SeedDiff'] *= -1
    df_flipped['WinPctDiff'] *= -1
    df_flipped['MarginDiff'] *= -1
    features = pd.concat([features, df_flipped[['SeedDiff', 'WinPctDiff', 'MarginDiff']]])
    labels += [0] * len(df_flipped)  # 0 = LTeam wins

    return features, labels


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Create features and labels
X, y = create_training_data(tourney_results, seeds, team_stats)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))


sample_submission[['Season', 'Team1', 'Team2']] = sample_submission['ID'].str.split('_', expand=True).astype(int)


# Merge Seeds
s = seeds[['Season', 'TeamID', 'SeedNum']]
sample_submission = sample_submission.merge(s, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left').rename(columns={'SeedNum': 'Seed1'}).drop(columns='TeamID')
sample_submission = sample_submission.merge(s, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left').rename(columns={'SeedNum': 'Seed2'}).drop(columns='TeamID')

# Merge Stats
ts = team_stats[['TeamID', 'Win%', 'AvgWinMargin']]
sample_submission = sample_submission.merge(ts, left_on='Team1', right_on='TeamID', how='left').rename(columns={'Win%': 'WinPct1', 'AvgWinMargin': 'Margin1'}).drop(columns='TeamID')
sample_submission = sample_submission.merge(ts, left_on='Team2', right_on='TeamID', how='left').rename(columns={'Win%': 'WinPct2', 'AvgWinMargin': 'Margin2'}).drop(columns='TeamID')



#Create Feature Differences
sample_submission['SeedDiff'] = sample_submission['Seed1'] - sample_submission['Seed2']
sample_submission['WinPctDiff'] = sample_submission['WinPct1'] - sample_submission['WinPct2']
sample_submission['MarginDiff'] = sample_submission['Margin1'] - sample_submission['Margin2']

# Fill missing values (if any)
features_2025 = sample_submission[['SeedDiff', 'WinPctDiff', 'MarginDiff']].fillna(0)


# Predict Probabilities
sample_submission['Pred'] = model.predict_proba(features_2025)[:, 1]


#save submission file
submission = sample_submission[['ID', 'Pred']].rename(columns={'Pred': 'Pred'})
submission.to_csv("prediction_submission.csv", index=False)




