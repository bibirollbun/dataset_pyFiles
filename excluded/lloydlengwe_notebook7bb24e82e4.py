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
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss


# Load historical NCAA games data
train_df = pd.read_csv("../input/march-machine-learning-mania-2025/MNCAATourneyDetailedResults.csv")
test_df = pd.read_csv("../input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv")
teams_df = pd.read_csv("../input/march-machine-learning-mania-2025/MTeams.csv")
seeds_df = pd.read_csv("../input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv")



def preprocess_data(df, seeds_df):
    df["Season"] = df["Season"].astype(int)
    
    # Merge seed information
    seeds_df["Seed"] = seeds_df["Seed"].apply(lambda x: int(x[1:3]))  # Extract only the numeric seed value
    df = df.merge(seeds_df, left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"], how="left")
    df.rename(columns={"Seed": "WSeed"}, inplace=True)
    df.drop(columns=["TeamID"], inplace=True)

    df = df.merge(seeds_df, left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"], how="left")
    df.rename(columns={"Seed": "LSeed"}, inplace=True)
    df.drop(columns=["TeamID"], inplace=True)

    # Calculate seed difference
    df["SeedDiff"] = df["WSeed"] - df["LSeed"]

    # Assign label (1 if the lower TeamId wins, else 0)
    df["Win"] = df["WTeamID"] < df["LTeamID"]
    df["Win"] = df["Win"].astype(int)

    return df[["Season", "WTeamID", "LTeamID", "SeedDiff", "Win"]]

train_df = preprocess_data(train_df, seeds_df)



X = train_df[["SeedDiff"]]
y = train_df["Win"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



y_pred = model.predict_proba(X_val)[:, 1]  # Get probability predictions
brier = brier_score_loss(y_val, y_pred)

print(f"Brier Score: {brier}")



# Create all possible team matchups
teams = teams_df["TeamID"].unique()
matchups = []

for team1 in teams:
    for team2 in teams:
        if team1 < team2:
            matchups.append([2025, team1, team2])

matchups_df = pd.DataFrame(matchups, columns=["Season", "WTeamID", "LTeamID"])
matchups_df["SeedDiff"] = matchups_df["WTeamID"] - matchups_df["LTeamID"]

# Predict probabilities
X_test = matchups_df[["SeedDiff"]]
matchups_df["Pred"] = model.predict_proba(X_test)[:, 1]

# Format submission file
matchups_df["ID"] = matchups_df.apply(lambda row: f"{row.Season}_{row.WTeamID}_{row.LTeamID}", axis=1)
submission = matchups_df[["ID", "Pred"]]

# Save submission file
submission.to_csv("submission.csv", index=False)

print("Submission file saved successfully!")


