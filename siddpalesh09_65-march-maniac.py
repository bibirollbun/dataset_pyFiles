import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
reg_season = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')
tourney = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
teams = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')
seeds = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')


# Checking missing values 
reg_season.head()
tourney.head()
teams.head()
seeds.head()

print(reg_season.isnull().sum())



# Number of games per season
games_per_season = reg_season['Season'].value_counts().sort_index()
games_per_season.plot(kind='bar', figsize=(12, 5), title="Number of Games per Season")
plt.ylabel("Game Count")
plt.show()

# Point differences
reg_season['PointDiff'] = reg_season['WScore'] - reg_season['LScore']
sns.histplot(reg_season['PointDiff'], bins=30, kde=True)
plt.title("Distribution of Winning Margins")
plt.xlabel("Point Difference")
plt.show()



# Winning team stats
wins = reg_season.groupby('WTeamID').agg({'WScore': ['mean', 'count']}).reset_index()
wins.columns = ['TeamID', 'AvgWinScore', 'NumWins']

# Losing team stats
losses = reg_season.groupby('LTeamID').agg({'LScore': ['mean', 'count']}).reset_index()
losses.columns = ['TeamID', 'AvgLossScore', 'NumLosses']

# Merge win/loss stats
team_stats = pd.merge(wins, losses, on='TeamID', how='outer').fillna(0)

# Calculate overall games and average score
team_stats['TotalGames'] = team_stats['NumWins'] + team_stats['NumLosses']
team_stats['WinRate'] = team_stats['NumWins'] / team_stats['TotalGames']
team_stats['AvgScore'] = (team_stats['AvgWinScore'] * team_stats['NumWins'] + 
                          team_stats['AvgLossScore'] * team_stats['NumLosses']) / team_stats['TotalGames']

team_stats = pd.merge(team_stats, teams, on='TeamID')
team_stats.sort_values('WinRate', ascending=False).head(10)



# Distribution of seed positions
seeds['SeedNum'] = seeds['Seed'].str.extract('(\d+)').astype(int)
sns.histplot(seeds['SeedNum'], bins=16)
plt.title("Seed Number Distribution")
plt.xlabel("Seed")
plt.show()

# Win percentage by seed
tourney = tourney.merge(seeds.rename(columns={'TeamID': 'WTeamID', 'SeedNum': 'WSeedNum'}), on=['Season', 'WTeamID'], how='left')
tourney = tourney.merge(seeds.rename(columns={'TeamID': 'LTeamID', 'SeedNum': 'LSeedNum'}), on=['Season', 'LTeamID'], how='left')

tourney['SeedDiff'] = tourney['WSeedNum'] - tourney['LSeedNum']
sns.histplot(tourney['SeedDiff'], kde=True)
plt.title("Winning Seed Advantage (WSeed - LSeed)")
plt.xlabel("Seed Difference")
plt.show()



# Extract seed number (e.g., "W01" -> 1)
seeds['SeedNum'] = seeds['Seed'].str.extract('(\d+)').astype(int)

# Merge in winner and loser seed
df = tourney.copy()
df = df.merge(seeds[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
df = df.rename(columns={'SeedNum': 'WSeed'})
df = df.drop('TeamID', axis=1)

df = df.merge(seeds[['Season', 'TeamID', 'SeedNum']], left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
df = df.rename(columns={'SeedNum': 'LSeed'})
df = df.drop('TeamID', axis=1)

# Randomize which team is Team1 and Team2
np.random.seed(42)
df['Team1'] = np.where(np.random.rand(len(df)) < 0.5, df['WTeamID'], df['LTeamID'])
df['Team2'] = np.where(df['Team1'] == df['WTeamID'], df['LTeamID'], df['WTeamID'])

# Assign seed accordingly
df['Team1Seed'] = np.where(df['Team1'] == df['WTeamID'], df['WSeed'], df['LSeed'])
df['Team2Seed'] = np.where(df['Team2'] == df['WTeamID'], df['WSeed'], df['LSeed'])

# Target: 1 if Team1 wins, else 0
df['Team1Win'] = np.where(df['Team1'] == df['WTeamID'], 1, 0)

# Feature: Seed difference
df['SeedDiff'] = df['Team1Seed'] - df['Team2Seed']


# Feature and target
X = df[['SeedDiff']]
y = df['Team1Win']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print(classification_report(y_test, y_pred))


