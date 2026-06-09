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

# Correct way to load the file in Kaggle
file_path = "/kaggle/input/cities/Cities.csv"  # Ensure this matches the actual dataset path
df_cities = pd.read_csv(file_path)

# Display the first few rows
df_cities.head()



# Check for missing values
missing_values = df_cities.isnull().sum()

# Check for duplicate rows
duplicate_rows = df_cities.duplicated().sum()

# Get unique counts of cities and states
unique_cities = df_cities["City"].nunique()
unique_states = df_cities["State"].nunique()

# Find the top 10 most common city names
top_cities = df_cities["City"].value_counts().head(10)

# Count the number of cities per state
cities_per_state = df_cities["State"].value_counts()

# Print the results
print("Missing Values:\n", missing_values)
print("\nNumber of Duplicate Rows:", duplicate_rows)
print("\nNumber of Unique Cities:", unique_cities)
print("\nNumber of Unique States:", unique_states)
print("\nTop 10 Most Common City Names:\n", top_cities)
print("\nNumber of Cities per State:\n", cities_per_state)



import pandas as pd

# Load team information
mteams = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MTeams.csv")
wteams = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WTeams.csv")

# Load season details
mseasons = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MSeasons.csv")
wseasons = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WSeasons.csv")

# Load tournament seeds
mtourney_seeds = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv")
wtourney_seeds = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv")

# Load regular season game results
mresults = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv")
wresults = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv")

# Display sample data
mteams.head()



# Check dataset structures
print(mteams.info())  # Check columns and data types
print(mresults.head())  # View first few rows of game results
print(mtourney_seeds["Seed"].unique())  # See unique tournament seeds

# Summary statistics
print(mresults.describe())  # Stats for regular season results





# Define dataset path
dataset_path = "/kaggle/input/march-machine-learning-mania-2025"

# Load men's and women's teams data
mteams = pd.read_csv(f"{dataset_path}/MTeams.csv")
wteams = pd.read_csv(f"{dataset_path}/WTeams.csv")

# Display the first few rows
print("Men's Teams Dataset:")
print(mteams.head(), "\n")

print("Women's Teams Dataset:")
print(wteams.head())



print("\nMissing values in Men's Seasons:")
print(mseasons.isnull().sum())

print("\nMissing values in Women's Seasons:")
print(wseasons.isnull().sum())



# Count the number of seasons recorded
print(f"Total Seasons (Men's): {mseasons.shape[0]}")
print(f"Total Seasons (Women's): {wseasons.shape[0]}")

# Check the earliest and latest recorded seasons
print(f"\nMen's Seasons range: {mseasons['Season'].min()} - {mseasons['Season'].max()}")
print(f"Women's Seasons range: {wseasons['Season'].min()} - {wseasons['Season'].max()}")

# Check the most recent 'DayZero' dates
print("\nMost recent 'DayZero' dates:")
print(mseasons[['Season', 'DayZero']].tail())





# Define dataset path
dataset_path = "/kaggle/input/march-machine-learning-mania-2025"

# Load tournament seeds data
m_seeds = pd.read_csv(f"{dataset_path}/MNCAATourneySeeds.csv")
w_seeds = pd.read_csv(f"{dataset_path}/WNCAATourneySeeds.csv")

# Display the first few rows
print("Men's NCAA Tournament Seeds:")
print(m_seeds.head(), "\n")

print("Women's NCAA Tournament Seeds:")
print(w_seeds.head())



print("\nMissing values in Men's Tournament Seeds:")
print(m_seeds.isnull().sum())

print("\nMissing values in Women's Tournament Seeds:")
print(w_seeds.isnull().sum())





# Count the number of seasons recorded
print(f"Total Seasons (Men's): {m_seeds['Season'].nunique()}")
print(f"Total Seasons (Women's): {w_seeds['Season'].nunique()}")

# Check the range of seasons covered
print(f"\nMen's Tournament Seasons range: {m_seeds['Season'].min()} - {m_seeds['Season'].max()}")
print(f"Women's Tournament Seasons range: {w_seeds['Season'].min()} - {w_seeds['Season'].max()}")

# Check unique seed values
print("\nUnique seed values (Men's):")
print(m_seeds['Seed'].unique())

print("\nUnique seed values (Women's):")
print(w_seeds['Seed'].unique())



# Extract the region (first letter) and seed number (digits)
m_seeds['Region'] = m_seeds['Seed'].str[0]
m_seeds['SeedNum'] = m_seeds['Seed'].str[1:3].astype(int)  # Convert to integer
m_seeds['PlayIn'] = m_seeds['Seed'].str.len() == 4  # True if play-in game (4-character seed)

w_seeds['Region'] = w_seeds['Seed'].str[0]
w_seeds['SeedNum'] = w_seeds['Seed'].str[1:3].astype(int)
w_seeds['PlayIn'] = w_seeds['Seed'].str.len() == 4

# Display updated datasets
print("Processed Men's Seeds:")
print(m_seeds.head())

print("\nProcessed Women's Seeds:")
print(w_seeds.head())



# Count the number of play-in games per season for men's and women's tournaments
playin_mens = m_seeds[m_seeds['PlayIn']].groupby('Season').size()
playin_womens = w_seeds[w_seeds['PlayIn']].groupby('Season').size()

# Display the first few records
print("Men's Play-In Games per Season:")
print(playin_mens.head())

print("\nWomen's Play-In Games per Season:")
print(playin_womens.head())



import matplotlib.pyplot as plt

# Plot play-in game trends over the years
plt.figure(figsize=(10, 5))
plt.plot(playin_mens.index, playin_mens.values, marker='o', label="Men's Play-In Games", linestyle='dashed')
plt.plot(playin_womens.index, playin_womens.values, marker='s', label="Women's Play-In Games", linestyle='dashed')

plt.xlabel("Season")
plt.ylabel("Number of Play-In Games")
plt.title("NCAA Tournament Play-In Games Over Time")
plt.legend()
plt.grid(True)
plt.show()



# Get unique play-in seeds for men and women
playin_mens_seeds = m_seeds[m_seeds['PlayIn']]['Seed'].unique()
playin_womens_seeds = w_seeds[w_seeds['PlayIn']]['Seed'].unique()

print("Unique Men's Play-In Seeds:", playin_mens_seeds)
print("Unique Women's Play-In Seeds:", playin_womens_seeds)




# Define dataset path
dataset_path = "/kaggle/input/march-machine-learning-mania-2025"

# Load men's and women's regular season game results
m_regular_season = pd.read_csv(f"{dataset_path}/MRegularSeasonCompactResults.csv")
w_regular_season = pd.read_csv(f"{dataset_path}/WRegularSeasonCompactResults.csv")

# Display first few rows
print("Men's Regular Season Data:")
print(m_regular_season.head(), "\n")

print("Women's Regular Season Data:")
print(w_regular_season.head())



print("\nMissing values in Men's Regular Season Data:")
print(m_regular_season.isnull().sum())

print("\nMissing values in Women's Regular Season Data:")
print(w_regular_season.isnull().sum())



print(f"\nTotal Men's Regular Season Games: {m_regular_season.shape[0]}")
print(f"Total Women's Regular Season Games: {w_regular_season.shape[0]}")

# Number of unique seasons
print("\nUnique seasons in Men's Data:", m_regular_season['Season'].nunique())
print("Unique seasons in Women's Data:", w_regular_season['Season'].nunique())

# Check win location distribution
print("\nWin Location Distribution (Men's):")
print(m_regular_season['WLoc'].value_counts())

print("\nWin Location Distribution (Women's):")
print(w_regular_season['WLoc'].value_counts())

# Check overtime game frequency
print("\nOvertime Games (Men's):", (m_regular_season['NumOT'] > 0).sum())
print("Overtime Games (Women's):", (w_regular_season['NumOT'] > 0).sum())



import matplotlib.pyplot as plt

# Count the number of games per season
men_games_per_season = m_regular_season.groupby("Season").size()
women_games_per_season = w_regular_season.groupby("Season").size()

# Plot the data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
men_games_per_season.plot(kind="line", marker="o", color="blue", label="Men")
plt.title("Number of Games per Season (Men)")
plt.xlabel("Season")
plt.ylabel("Number of Games")
plt.legend()

plt.subplot(1, 2, 2)
women_games_per_season.plot(kind="line", marker="o", color="red", label="Women")
plt.title("Number of Games per Season (Women)")
plt.xlabel("Season")
plt.ylabel("Number of Games")
plt.legend()

plt.tight_layout()
plt.show()



import seaborn as sns

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.countplot(x=m_regular_season["WLoc"], palette="Blues")
plt.title("Win Location Distribution (Men)")

plt.subplot(1, 2, 2)
sns.countplot(x=w_regular_season["WLoc"], palette="Reds")
plt.title("Win Location Distribution (Women)")

plt.tight_layout()
plt.show()



# Overtime games count
men_overtime_games = m_regular_season[m_regular_season["NumOT"] > 0].groupby("Season").size()
women_overtime_games = w_regular_season[w_regular_season["NumOT"] > 0].groupby("Season").size()

# Plot OT games per season
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
men_overtime_games.plot(kind="line", marker="o", color="blue", label="Men")
plt.title("Overtime Games per Season (Men)")
plt.xlabel("Season")
plt.ylabel("Overtime Games")
plt.legend()

plt.subplot(1, 2, 2)
women_overtime_games.plot(kind="line", marker="o", color="red", label="Women")
plt.title("Overtime Games per Season (Women)")
plt.xlabel("Season")
plt.ylabel("Overtime Games")
plt.legend()

plt.tight_layout()
plt.show()



# Load the tournament data
m_tourney_results = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv")
w_tourney_results = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv")

# Display the first few rows
m_tourney_results.head(), w_tourney_results.head()




# Function to map DayNum to tournament round
def get_tournament_round(day_num):
    if day_num in [134, 135]:
        return "Play-in"
    elif day_num in [136, 137]:
        return "Round of 64"
    elif day_num in [138, 139]:
        return "Round of 32"
    elif day_num in [143, 144]:
        return "Sweet 16"
    elif day_num in [145, 146]:
        return "Elite Eight"
    elif day_num == 152:
        return "Final Four"
    elif day_num == 154:
        return "Championship"
    else:
        return "Other"

# Apply to datasets
m_tourney_results["Round"] = m_tourney_results["DayNum"].apply(get_tournament_round)
w_tourney_results["Round"] = w_tourney_results["DayNum"].apply(get_tournament_round)

# Display the updated data
m_tourney_results.head(), w_tourney_results.head()





plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(y=m_tourney_results["Round"], order=["Play-in", "Round of 64", "Round of 32", "Sweet 16", "Elite Eight", "Final Four", "Championship"], palette="Blues")
plt.title("NCAA Tournament Rounds (Men)")
plt.xlabel("Game Count")

plt.subplot(1, 2, 2)
sns.countplot(y=w_tourney_results["Round"], order=["Play-in", "Round of 64", "Round of 32", "Sweet 16", "Elite Eight", "Final Four", "Championship"], palette="Reds")
plt.title("NCAA Tournament Rounds (Women)")
plt.xlabel("Game Count")

plt.tight_layout()
plt.show()





# Load the sample submission file
submission = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv")

# Display the first few rows
print("Sample Submission Format:")
print(submission.head())

# Ensure all predictions are set to 50% as a baseline
submission["Pred"] = 0.50

# Save the updated submission file (optional)
submission.to_csv("submission.csv", index=False)

print("\nUpdated submission file saved as 'submission.csv' with all predictions set to 50%.")



import pandas as pd
from sklearn.metrics import brier_score_loss

# Load the sample submission file
submission_df = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv")

# Create a dataframe of ground truth values (assuming all should be 1 for testing purposes)
solution_df = submission_df.copy()
solution_df['Pred'] = 1  # Setting ground truth to 1 (for example)

# Extract true and predicted values
y_true = solution_df['Pred']  # Ground truth (all set to 1 here)
y_pred = submission_df['Pred']  # Model predictions

# Calculate the Brier score
brier_score = brier_score_loss(y_true, y_pred)
print(f"Brier Score: {brier_score}")



import pandas as pd
from sklearn.metrics import brier_score_loss

# Load the sample submission file
submission_df = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv")

# Create a dataframe of ground truth values (assuming all should be 1 for testing purposes)
solution_df = submission_df.copy()
solution_df['Pred'] = 1  # Setting ground truth to 1 (for example)

# Extract true and predicted values
y_true = solution_df['Pred']  # Ground truth
y_pred = submission_df['Pred']  # Model predictions

# Calculate the Brier score
brier_score = brier_score_loss(y_true, y_pred)
print(f"Brier Score: {brier_score}")

# Save the solution file for Kaggle submission
solution_df.to_csv("submission.csv", index=False)

print("Submission file saved: submission.csv")





