import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import brier_score_loss, mean_squared_error


w_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')
m_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
seed_df = pd.concat([m_seed, w_seed], axis=0).fillna(0.05)
submission_df = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')


seed_df.head()


submission_df.head()


data_path = "/kaggle/input/march-machine-learning-mania-2025"
m_seeds = pd.read_csv(f"{data_path}/MNCAATourneySeeds.csv")
w_seeds = pd.read_csv(f"{data_path}/WNCAATourneySeeds.csv")

# Extract region from Seed column (first letter)
m_seeds["Region"] = m_seeds["Seed"].str[0]  # W, X, Y, Z
w_seeds["Region"] = w_seeds["Seed"].str[0]

# Count unique teams per region
m_team_region_counts = m_seeds.groupby("Region")["TeamID"].nunique().sort_index()
w_team_region_counts = w_seeds.groupby("Region")["TeamID"].nunique().sort_index()

# Plot team count per region (Line chart)
plt.figure(figsize=(8, 5))
sns.lineplot(x=m_team_region_counts.index, y=m_team_region_counts.values, marker="o", label="Men", color="royalblue")
sns.lineplot(x=w_team_region_counts.index, y=w_team_region_counts.values, marker="o", label="Women", color="darkorange")

# Add data labels
for i, txt in enumerate(m_team_region_counts.values):
    plt.text(m_team_region_counts.index[i], txt, str(txt), ha="center", va="bottom", fontsize=10, color="royalblue")
for i, txt in enumerate(w_team_region_counts.values):
    plt.text(w_team_region_counts.index[i], txt, str(txt), ha="center", va="top", fontsize=10, color="darkorange")

plt.xlabel("Region")
plt.ylabel("Number of Teams")
plt.title("Number of Teams in Each Region (with Values)")
plt.legend()
plt.show()


# Load NCAA tournament results
m_tourney_results = pd.read_csv(f"{data_path}/MNCAATourneyCompactResults.csv")
w_tourney_results = pd.read_csv(f"{data_path}/WNCAATourneyCompactResults.csv")

# Combine winning and losing team scores
m_tourney_scores = pd.concat([m_tourney_results["WScore"], m_tourney_results["LScore"]])
w_tourney_scores = pd.concat([w_tourney_results["WScore"], w_tourney_results["LScore"]])

# Count score frequencies
m_score_dist = m_tourney_scores.value_counts().sort_index()
w_score_dist = w_tourney_scores.value_counts().sort_index()

# Plot score distribution (Line chart)
plt.figure(figsize=(8, 5))
sns.lineplot(x=m_score_dist.index, y=m_score_dist.values, marker="o", label="Men", color="royalblue")
sns.lineplot(x=w_score_dist.index, y=w_score_dist.values, marker="o", label="Women", color="darkorange")

plt.xlabel("Points Scored")
plt.ylabel("Frequency")
plt.title("NCAA Tournament Score Distribution (Line Chart)")
plt.legend()
plt.show()


# Group by Season and calculate the average winning score
m_season_wins = m_tourney_results.groupby("Season")["WScore"].mean()
w_season_wins = w_tourney_results.groupby("Season")["WScore"].mean()

# Plot winning score trends over seasons
plt.figure(figsize=(8, 5))
sns.lineplot(x=m_season_wins.index, y=m_season_wins.values, marker="o", label="Men", color="royalblue")
sns.lineplot(x=w_season_wins.index, y=w_season_wins.values, marker="o", label="Women", color="darkorange")

plt.xlabel("Season")
plt.ylabel("Average Winning Score")
plt.title("Average Winning Score per Season (Men & Women)")
plt.legend()
plt.show()


import re


def extract_game_info(id_str):
    # Extract year and team_ids
    parts = id_str.split('_')
    year = int(parts[0])
    teamID1 = int(parts[1])
    teamID2 = int(parts[2])
    return year, teamID1, teamID2

def extract_seed_value(seed_str):
    # Extract seed value
    numeric_part = re.sub(r'\D', '', seed_str)
    try:
        return int(numeric_part)
    # Set seed to 16 for unselected teams and errors
    except ValueError:
        return 16


# Reformat the data
submission_df[['Season', 'TeamID1', 'TeamID2']] = submission_df['ID'].apply(lambda x: pd.Series(extract_game_info(x)))
latest_season = seed_df['Season'].max()
latest_seed_df = seed_df.query("Season == @latest_season").copy()
latest_seed_df['SeedValue'] = latest_seed_df['Seed'].apply(extract_seed_value)


submission_df.head(), latest_seed_df.head()


submission_with_seed = submission_df.copy()


# Merge seed information for TeamID1
submission_with_seed = submission_with_seed.merge(
    latest_seed_df[['TeamID', 'SeedValue']], 
    left_on='TeamID1', right_on='TeamID', 
    how='left'
).rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])

# Merge seed information for TeamID2
submission_with_seed = submission_with_seed.merge(
    latest_seed_df[['TeamID', 'SeedValue']], 
    left_on='TeamID2', right_on='TeamID', 
    how='left'
).rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])

# Fill missing seed values with 16 (assuming unseeded teams are the lowest rank)
submission_with_seed[['SeedValue1', 'SeedValue2']] = submission_with_seed[['SeedValue1', 'SeedValue2']].fillna(16)


submission_with_seed.describe()


# Calculate seed difference
submission_with_seed['SeedDiff'] = submission_with_seed['SeedValue1'] - submission_with_seed['SeedValue2']
submission_with_seed['Pred'] = 0.5 + (0.03 * submission_with_seed['SeedDiff'])
submission_with_seed = submission_with_seed[['ID', 'Pred']].fillna(0.5)
submission_with_seed.head()


stats = submission_with_seed.iloc[:, 1].describe()
stats


# Create a dataframe of ground truth values
solution_df = submission_with_seed.copy()
solution_df['Pred'] = 1

# Now calculate the Brier score
y_true = solution_df['Pred']
y_pred = submission_df['Pred']
brier_score = brier_score_loss(y_true, y_pred)
print(f"Brier Score: {brier_score}")


submission_with_seed.to_csv('/kaggle/working/submission.csv', index=False)


submission_with_seed.head()




