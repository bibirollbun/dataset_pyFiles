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


# Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Suppress Warnings
import warnings
warnings.filterwarnings("ignore")



# Load Datasets
df_seeds_m = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv")
df_seeds_w = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv")

df_regular_m = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv")
df_regular_w = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv")

df_tourney_m = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv")
df_tourney_w = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv")

df_rankings = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MMasseyOrdinals.csv")

df_teams_m = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MTeams.csv")
df_teams_w = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WTeams.csv")

df_submission = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv")



# View Seeds Data
print(df_seeds_m.head())
print(df_seeds_w.head())

# View Regular Season Results
print(df_regular_m.head())
print(df_regular_w.head())

# View Tournament Results
print(df_tourney_m.head())
print(df_tourney_w.head())

# View Rankings Data
print(df_rankings.head())

# View Teams Metadata
print(df_teams_m.head())
print(df_teams_w.head())

# View Submission Format
print(df_submission.head())



# Extract numeric seed values
df_seeds_m["SeedNumber"] = df_seeds_m["Seed"].str.extract("(\d+)").astype(int)
df_seeds_w["SeedNumber"] = df_seeds_w["Seed"].str.extract("(\d+)").astype(int)

# Drop original Seed column
df_seeds_m.drop(columns=["Seed"], inplace=True)
df_seeds_w.drop(columns=["Seed"], inplace=True)

print(df_seeds_m.head())
print(df_seeds_w.head())



# Compute win ratios for men
df_team_wins = df_regular_m.groupby("WTeamID").size().reset_index(name="Wins")
df_team_losses = df_regular_m.groupby("LTeamID").size().reset_index(name="Losses")

df_win_ratio = pd.merge(df_team_wins, df_team_losses, left_on="WTeamID", right_on="LTeamID", how="outer")
df_win_ratio.fillna(0, inplace=True)

df_win_ratio["TotalGames"] = df_win_ratio["Wins"] + df_win_ratio["Losses"]
df_win_ratio["WinRatio"] = df_win_ratio["Wins"] / df_win_ratio["TotalGames"]

df_win_ratio = df_win_ratio[["WTeamID", "WinRatio"]]
df_win_ratio.rename(columns={"WTeamID": "TeamID"}, inplace=True)

print(df_win_ratio.head())



# Get most recent ranking for each team
df_rankings_latest = df_rankings[df_rankings["Season"] == df_rankings["Season"].max()]
df_rankings_latest = df_rankings_latest.groupby("TeamID")["OrdinalRank"].mean().reset_index()

print(df_rankings_latest.head())



df_matchups = df_submission.copy()

# Split matchup IDs
df_matchups["Season"] = df_matchups["ID"].apply(lambda x: int(x.split("_")[0]))
df_matchups["Team1"] = df_matchups["ID"].apply(lambda x: int(x.split("_")[1]))
df_matchups["Team2"] = df_matchups["ID"].apply(lambda x: int(x.split("_")[2]))

print(df_matchups.head())



# Merge Seed Numbers
df_matchups = df_matchups.merge(df_seeds_m, left_on=["Season", "Team1"], right_on=["Season", "TeamID"], how="left").rename(columns={"SeedNumber": "Seed1"}).drop(columns=["TeamID"])
df_matchups = df_matchups.merge(df_seeds_m, left_on=["Season", "Team2"], right_on=["Season", "TeamID"], how="left").rename(columns={"SeedNumber": "Seed2"}).drop(columns=["TeamID"])

# Merge Win Ratios
df_matchups = df_matchups.merge(df_win_ratio, left_on="Team1", right_on="TeamID", how="left").rename(columns={"WinRatio": "WinRatio1"}).drop(columns=["TeamID"])
df_matchups = df_matchups.merge(df_win_ratio, left_on="Team2", right_on="TeamID", how="left").rename(columns={"WinRatio": "WinRatio2"}).drop(columns=["TeamID"])

# Compute Seed & Win Ratio Differences
df_matchups["SeedDiff"] = df_matchups["Seed1"] - df_matchups["Seed2"]
df_matchups["WinRatioDiff"] = df_matchups["WinRatio1"] - df_matchups["WinRatio2"]

print(df_matchups.head())



# Prepare Training Data: Create target variable
df_tourney_m["Winner"] = (df_tourney_m["WTeamID"] < df_tourney_m["LTeamID"]).astype(int)

# Merge Tournament Data with Features
df_train = df_tourney_m.merge(df_seeds_m, left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"], how="left").rename(columns={"SeedNumber": "Seed1"}).drop(columns=["TeamID"])
df_train = df_train.merge(df_seeds_m, left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"], how="left").rename(columns={"SeedNumber": "Seed2"}).drop(columns=["TeamID"])

df_train = df_train.merge(df_win_ratio, left_on="WTeamID", right_on="TeamID", how="left").rename(columns={"WinRatio": "WinRatio1"}).drop(columns=["TeamID"])
df_train = df_train.merge(df_win_ratio, left_on="LTeamID", right_on="TeamID", how="left").rename(columns={"WinRatio": "WinRatio2"}).drop(columns=["TeamID"])

# Compute Seed & Win Ratio Differences
df_train["SeedDiff"] = df_train["Seed1"] - df_train["Seed2"]
df_train["WinRatioDiff"] = df_train["WinRatio1"] - df_train["WinRatio2"]

# X (features) and y (target)
X_train = df_train[["SeedDiff", "WinRatioDiff"]]
y_train = df_train["Winner"]

print(X_train.head(), y_train.head())



import xgboost as xgb

# Define XGBoost Model
model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5)

# Train the model
model.fit(X_train, y_train)

# Predict on Kaggle Matchups
X_test = df_matchups[["SeedDiff", "WinRatioDiff"]]
df_matchups["Pred"] = model.predict_proba(X_test)[:, 1]

# Save Submission File
df_matchups[["ID", "Pred"]].to_csv("submission.csv", index=False)

print("✅ Submission file saved successfully!")





