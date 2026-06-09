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





best_model = grid_search.best_estimator_



import joblib

# After training, save your model to disk
joblib.dump(best_model, 'best_model.pkl')



import joblib

best_model = joblib.load('best_model.pkl')
print("Loaded best_model successfully.")



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

# Load datasets
regular_season_results = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv")
tourney_seeds = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv")
sample_submission = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv")

# Check a sample of each
print("Regular Season Results (sample):")
print(regular_season_results.head())

print("\nTournament Seeds (sample):")
print(tourney_seeds.head())

print("\nSample Submission Stage2 (sample):")
print(sample_submission.head())



# Aggregate winning data
wins = regular_season_results.groupby("WTeamID").agg(
    Wins=('WTeamID', 'count'),
    PointsScored_W=('WScore', 'sum'),
    PointsAllowed_W=('LScore', 'sum')
).reset_index().rename(columns={"WTeamID": "TeamID"})

# Aggregate losing data
losses = regular_season_results.groupby("LTeamID").agg(
    Losses=('LTeamID', 'count'),
    PointsScored_L=('LScore', 'sum'),
    PointsAllowed_L=('WScore', 'sum')
).reset_index().rename(columns={"LTeamID": "TeamID"})

# Merge wins and losses
team_stats = pd.merge(wins, losses, on="TeamID", how="outer").fillna(0)

# Total games played
team_stats["TotalGames"] = team_stats["Wins"] + team_stats["Losses"]

# Win rate and point differential
team_stats["WinRate"] = team_stats["Wins"] / team_stats["TotalGames"]
team_stats["PointDifferential"] = (
    (team_stats["PointsScored_W"] - team_stats["PointsAllowed_W"]) +
    (team_stats["PointsScored_L"] - team_stats["PointsAllowed_L"])
) / team_stats["TotalGames"]

print("Team Stats (sample):")
print(team_stats.head())



# Compute home win rate per team from regular season results (WLoc = 'H' for win at home)
home_adv = regular_season_results.groupby("WTeamID").apply(
    lambda x: (x['WLoc'] == 'H').sum() / len(x)
).reset_index(name="HomeWinRate").rename(columns={"WTeamID": "TeamID"})

print("Home Court Advantage (sample):")
print(home_adv.head())



# Define a function to arrange games so that the lower TeamID is first
def process_game(row):
    # Determine lower and higher team IDs
    team_low = min(row['WTeamID'], row['LTeamID'])
    team_high = max(row['WTeamID'], row['LTeamID'])
    # If the winning team is the lower ID, label 1; otherwise 0.
    outcome = 1 if row['WTeamID'] == team_low else 0
    return pd.Series([row['Season'], team_low, team_high, outcome], 
                     index=['Season', 'Team1_Game', 'Team2_Game', 'Outcome'])

# Apply processing to all regular season games
training_games = regular_season_results.apply(process_game, axis=1)

# Merge team-level features for the matchup (Team1)
training = pd.merge(training_games, team_stats[['TeamID', 'WinRate', 'PointDifferential']], 
                    left_on='Team1_Game', right_on='TeamID', how='left') \
           .rename(columns={'WinRate': 'WinRate_Team1', 'PointDifferential': 'PointDifferential_Team1'}) \
           .drop(columns=["TeamID"])

# Merge team-level features for Team2
training = pd.merge(training, team_stats[['TeamID', 'WinRate', 'PointDifferential']], 
                    left_on='Team2_Game', right_on='TeamID', how='left') \
           .rename(columns={'WinRate': 'WinRate_Team2', 'PointDifferential': 'PointDifferential_Team2'}) \
           .drop(columns=["TeamID"])

# Merge home court advantage metrics for Team1 and Team2
training = pd.merge(training, home_adv, left_on='Team1_Game', right_on='TeamID', how='left') \
           .rename(columns={'HomeWinRate': 'HomeWinRate_Team1'}).drop(columns=["TeamID"])
training = pd.merge(training, home_adv, left_on='Team2_Game', right_on='TeamID', how='left') \
           .rename(columns={'HomeWinRate': 'HomeWinRate_Team2'}).drop(columns=["TeamID"])

# Compute feature differences
training['WinRateDifference'] = training['WinRate_Team1'] - training['WinRate_Team2']
training['PointDifferentialDifference'] = training['PointDifferential_Team1'] - training['PointDifferential_Team2']
training['HomeWinRateDifference'] = training['HomeWinRate_Team1'] - training['HomeWinRate_Team2']

# Final training features and label
features = ['WinRateDifference', 'PointDifferentialDifference', 'HomeWinRateDifference']
X = training[features]
y = training['Outcome']

print("Training features (sample):")
print(X.head())
print("Training labels (sample):")
print(y.head())



lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='binary_logloss',
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
)



from sklearn.metrics import brier_score_loss

# Get predicted probabilities for the positive class on the validation set
val_preds = lgb_model.predict_proba(X_val)[:, 1]
val_brier = brier_score_loss(y_val, val_preds)
print("Validation Brier Score:", val_brier)



print("X_test columns:", X_test.columns)
print("X_test shape:", X_test.shape)



# --- Assuming test_df was created from SampleSubmissionStage2.csv ---
# Extract Season, Team1ID, and Team2ID from 'ID'
test_df['Season'] = test_df['ID'].str[:4].astype(int)
test_df['Team1ID'] = test_df['ID'].str.split('_').str[1].astype(int)
test_df['Team2ID'] = test_df['ID'].str.split('_').str[2].astype(int)

# Merge team statistics for WinRate and PointDifferential
# For Team1:
test_df = pd.merge(
    test_df,
    team_stats[['TeamID', 'WinRate', 'PointDifferential']],
    left_on='Team1ID',
    right_on='TeamID',
    how='left'
).rename(columns={'WinRate': 'WinRate_Team1', 'PointDifferential': 'PointDifferential_Team1'}).drop(columns=['TeamID'])

# For Team2:
test_df = pd.merge(
    test_df,
    team_stats[['TeamID', 'WinRate', 'PointDifferential']],
    left_on='Team2ID',
    right_on='TeamID',
    how='left'
).rename(columns={'WinRate': 'WinRate_Team2', 'PointDifferential': 'PointDifferential_Team2'}).drop(columns=['TeamID'])

# Merge home court advantage for Team1 and Team2
# For Team1:
test_df = pd.merge(
    test_df,
    home_adv[['TeamID', 'HomeWinRate']],
    left_on='Team1ID',
    right_on='TeamID',
    how='left'
).rename(columns={'HomeWinRate': 'HomeWinRate_Team1'}).drop(columns=['TeamID'])

# For Team2:
test_df = pd.merge(
    test_df,
    home_adv[['TeamID', 'HomeWinRate']],
    left_on='Team2ID',
    right_on='TeamID',
    how='left'
).rename(columns={'HomeWinRate': 'HomeWinRate_Team2'}).drop(columns=['TeamID'])

# Compute feature differences
test_df['WinRateDifference'] = test_df['WinRate_Team1'] - test_df['WinRate_Team2']
test_df['PointDifferentialDifference'] = test_df['PointDifferential_Team1'] - test_df['PointDifferential_Team2']
test_df['HomeWinRateDifference'] = test_df['HomeWinRate_Team1'] - test_df['HomeWinRate_Team2']

# Now, build X_test with all three features:
X_test = test_df[['WinRateDifference', 'PointDifferentialDifference', 'HomeWinRateDifference']]
print("Updated X_test columns:", X_test.columns)
print("Updated X_test shape:", X_test.shape)



---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-66-767124e631a3> in <cell line: 3>()
      1 # --- Assuming test_df was created from SampleSubmissionStage2.csv ---
      2 # Extract Season, Team1ID, and Team2ID from 'ID'
----> 3 test_df['Season'] = test_df['ID'].str[:4].astype(int)
      4 test_df['Team1ID'] = test_df['ID'].str.split('_').str[1].astype(int)
      5 test_df['Team2ID'] = test_df['ID'].str.split('_').str[2].astype(int)

NameError: name 'test_df' is not defined


import pandas as pd

# Load test_df from SampleSubmissionStage2.csv
test_df = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv")
print("Loaded test_df:")
print(test_df.head())

# Extract Season, Team1ID, and Team2ID from 'ID'
test_df['Season'] = test_df['ID'].str[:4].astype(int)
test_df['Team1ID'] = test_df['ID'].str.split('_').str[1].astype(int)
test_df['Team2ID'] = test_df['ID'].str.split('_').str[2].astype(int)

# Merge team statistics for Team1
test_df = pd.merge(
    test_df,
    team_stats[['TeamID', 'WinRate', 'PointDifferential']],
    left_on='Team1ID',
    right_on='TeamID',
    how='left'
).rename(columns={'WinRate': 'WinRate_Team1', 'PointDifferential': 'PointDifferential_Team1'}).drop(columns=['TeamID'])

# Merge team statistics for Team2
test_df = pd.merge(
    test_df,
    team_stats[['TeamID', 'WinRate', 'PointDifferential']],
    left_on='Team2ID',
    right_on='TeamID',
    how='left'
).rename(columns={'WinRate': 'WinRate_Team2', 'PointDifferential': 'PointDifferential_Team2'}).drop(columns=['TeamID'])

# Merge home court advantage for Team1
test_df = pd.merge(
    test_df,
    home_adv[['TeamID', 'HomeWinRate']],
    left_on='Team1ID',
    right_on='TeamID',
    how='left'
).rename(columns={'HomeWinRate': 'HomeWinRate_Team1'}).drop(columns=['TeamID'])

# Merge home court advantage for Team2
test_df = pd.merge(
    test_df,
    home_adv[['TeamID', 'HomeWinRate']],
    left_on='Team2ID',
    right_on='TeamID',
    how='left'
).rename(columns={'HomeWinRate': 'HomeWinRate_Team2'}).drop(columns=['TeamID'])

# Compute feature differences
test_df['WinRateDifference'] = test_df['WinRate_Team1'] - test_df['WinRate_Team2']
test_df['PointDifferentialDifference'] = test_df['PointDifferential_Team1'] - test_df['PointDifferential_Team2']
test_df['HomeWinRateDifference'] = test_df['HomeWinRate_Team1'] - test_df['HomeWinRate_Team2']

# Create X_test with all three features
X_test = test_df[['WinRateDifference', 'PointDifferentialDifference', 'HomeWinRateDifference']]
print("Updated X_test columns:", X_test.columns)
print("Updated X_test shape:", X_test.shape)




test_predictions = lgb_model.predict_proba(X_test)[:, 1]
test_df['Pred'] = test_predictions



print("Loaded test_df:")
print(test_df.head())
print("Updated X_test columns:", X_test.columns)
print("Updated X_test shape:", X_test.shape)



# Generate predictions for the test set.
# This will produce probabilities for the event that the lower TeamID wins.
test_predictions = lgb_model.predict_proba(X_test)[:, 1]

# Add the predictions to the test_df DataFrame.
test_df['Pred'] = test_predictions

# Prepare the final submission DataFrame with only the required columns (ID and Pred)
final_submission = test_df[['ID', 'Pred']]

# Check the first few rows and the shape of the submission
print(final_submission.head())
print("Final submission shape:", final_submission.shape)

# Save the final submission file as CSV.
final_submission.to_csv("submission.csv", index=False)
print("Final submission file saved as 'submission.csv'")





