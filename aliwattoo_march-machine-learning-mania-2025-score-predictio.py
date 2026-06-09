





# Required Libraries

# Data Manipulation
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score

# Feature Engineering & Data Processing
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Tournament Simulation & Bracket Modeling
import itertools 
import networkx as nx 

# Warnings Handling
import warnings
warnings.filterwarnings('ignore')


# Data Path 
data_path = '/kaggle/input/march-machine-learning-mania-2025/'
# Game Results (Regular Season & Tournament)

df_regular_season_compact = pd.read_csv(data_path + "MRegularSeasonCompactResults.csv")
df_regular_season_detailed = pd.read_csv(data_path + "MRegularSeasonDetailedResults.csv")
df_tourney_compact = pd.read_csv(data_path + "MNCAATourneyCompactResults.csv")
df_tourney_detailed = pd.read_csv(data_path + "MNCAATourneyDetailedResults.csv")

# Tournament Seeding
df_seeds = pd.read_csv(data_path + "MNCAATourneySeeds.csv")

# Team Strength & Rankings
df_rankings = pd.read_csv(data_path + "MMasseyOrdinals.csv")

# Coaching Data
df_coaches = pd.read_csv(data_path + "MTeamCoaches.csv")

# Team Metadata
df_teams = pd.read_csv(data_path + "MTeams.csv")
df_team_spellings = pd.read_csv(data_path + "MTeamSpellings.csv", encoding="ISO-8859-1")

# Tournament Bracket Structure
df_tourney_slots = pd.read_csv(data_path + "MNCAATourneySlots.csv")
df_tourney_seed_round_slots = pd.read_csv(data_path + "MNCAATourneySeedRoundSlots.csv")

# Display basic info about datasets
print("Regular Season Compact Results:\n", df_regular_season_compact.head(), "\n")
print("NCAA Tournament Compact Results:\n", df_tourney_compact.head(), "\n")
print("Tournament Seeds:\n", df_seeds.head(), "\n")
print("Rankings:\n", df_rankings.head(), "\n")
print("Coaches:\n", df_coaches.head(), "\n")
print("Teams:\n", df_teams.head(), "\n")
print("Tournament Slots:\n", df_tourney_slots.head(), "\n")


# Basic Info of all the dataset
print("Regular Season Compact Results:\n", df_regular_season_compact.isnull().sum(), "\n")
print("NCAA Tournament Compact Results:\n", df_tourney_compact.isnull().sum(), "\n")
print("Tournament Seeds:\n", df_seeds.isnull().sum(), "\n")
print("Rankings:\n", df_rankings.isnull().sum(), "\n")
print("Coaches:\n", df_coaches.isnull().sum(), "\n")
print("Teams:\n", df_teams.isnull().sum(), "\n")
print("Tournament Slots:\n", df_tourney_slots.isnull().sum(), "\n")

# There is no null values in these files


# Data Cleaning Process


# Check missing values in all datasets
datasets = {
    "Regular Season Compact": df_regular_season_compact,
    "Regular Season Detailed": df_regular_season_detailed,
    "Tournament Compact": df_tourney_compact,
    "Tournament Detailed": df_tourney_detailed,
    "Tournament Seeds": df_seeds,
    "Rankings": df_rankings,
    "Coaches": df_coaches,
    "Teams": df_teams,
    "Team Spellings": df_team_spellings,
    "Tournament Slots": df_tourney_slots,
    "Seed Round Slots": df_tourney_seed_round_slots
}

# Display missing Values in each dataset
for name, df in datasets.items():
    print(f"Missing Values in {name}:\n", df.isnull().sum(), "\n")


# Feature Engineering for NCAA Tournament Prediction




# Load the dataset
regular_season_detailed = df_regular_season_detailed

# Compute team-level statistics
winning_stats = regular_season_detailed.groupby("WTeamID").agg({
    "WScore": ["mean", "sum"],
    "WFGM": "mean", "WFGA": "mean", "WFGM3": "mean", "WFGA3": "mean",
    "WFTM": "mean", "WFTA": "mean", "WOR": "mean", "WDR": "mean",
    "WAst": "mean", "WTO": "mean", "WStl": "mean", "WBlk": "mean", "WPF": "mean"
}).reset_index()

# Rename columns for clarity
winning_stats.columns = ["TeamID", "AvgPointsScored", "TotalPointsScored",
                         "AvgFGM", "AvgFGA", "AvgFGM3", "AvgFGA3", 
                         "AvgFTM", "AvgFTA", "AvgOReb", "AvgDReb", 
                         "AvgAssists", "AvgTurnovers", "AvgSteals", 
                         "AvgBlocks", "AvgFouls"]

# Compute losing team stats similarly
losing_stats = regular_season_detailed.groupby("LTeamID").agg({
    "LScore": ["mean", "sum"],
    "LFGM": "mean", "LFGA": "mean", "LFGM3": "mean", "LFGA3": "mean",
    "LFTM": "mean", "LFTA": "mean", "LOR": "mean", "LDR": "mean",
    "LAst": "mean", "LTO": "mean", "LStl": "mean", "LBlk": "mean", "LPF": "mean"
}).reset_index()

# Rename columns for clarity
losing_stats.columns = ["TeamID", "AvgPointsAllowed", "TotalPointsAllowed",
                        "AvgFGM_Allowed", "AvgFGA_Allowed", "AvgFGM3_Allowed", "AvgFGA3_Allowed", 
                        "AvgFTM_Allowed", "AvgFTA_Allowed", "AvgOReb_Allowed", "AvgDReb_Allowed", 
                        "AvgAssists_Allowed", "AvgTurnovers_Allowed", "AvgSteals_Allowed", 
                        "AvgBlocks_Allowed", "AvgFouls_Allowed"]

# Merge winning and losing stats to get full team stats
team_stats = pd.merge(winning_stats, losing_stats, on="TeamID", how="outer").fillna(0)

# Compute Win Percentage
team_stats["WinRate"] = team_stats["TotalPointsScored"] / (team_stats["TotalPointsScored"] + team_stats["TotalPointsAllowed"])

# Save the processed features
team_stats.to_csv("Team_Stats.csv", index=False)

print("✅ Team-level features created successfully!")



# Load tournament seed data
tourney_seeds = df_seeds

# Extract numerical seed
tourney_seeds["SeedValue"] = tourney_seeds["Seed"].apply(lambda x: int(x[1:3]))

# Save processed seeds
tourney_seeds.to_csv("Processed_Seeds.csv", index=False)

print("✅ Tournament seeds extracted!")


# Load rankings dataset
rankings = df_rankings

# Compute average ranking for each team per season
avg_rankings = rankings.groupby(["Season", "TeamID"]).agg({"OrdinalRank": "mean"}).reset_index()
avg_rankings.rename(columns={"OrdinalRank": "AvgRank"}, inplace=True)

# Save processed rankings
avg_rankings.to_csv("Processed_Rankings.csv", index=False)

print("✅ Team rankings processed!")


# Merge All Features


# Merge team stats with seeds
final_data = pd.merge(team_stats, tourney_seeds, on="TeamID", how="left")

# Merge with rankings
final_data = pd.merge(final_data, avg_rankings, on=["Season", "TeamID"], how="left")

# Fill missing values
final_data.fillna(0, inplace=True)

# Save the final feature dataset
final_data.to_csv("Final_Feature_Dataset.csv", index=False)

print("✅ Feature Engineering Completed!")



#EDA


#Basic Statistics and Overview

# Load Dataset
regular_season = df_regular_season_detailed
tournament_results = df_tourney_detailed
team_info = df_teams

# Overview of Dataset
print(regular_season.info())
print(regular_season.describe())


# Distribution of Team Scores
plt.figure(figsize=(12, 6))
sns.histplot(regular_season['WScore'], bins=30, kde=True, label='Winning Team Score')
sns.histplot(regular_season['LScore'], bins=30, kde=True, color='red', label='Losing Team Score')
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Distribution of Winning & Losing Scores")
plt.legend()
plt.show()


# Win/Loss Trends (Checking which teams win more often)

# Count wins per team
winning_teams = regular_season['WTeamID'].value_counts()
losing_teams = regular_season['LTeamID'].value_counts()

# Plot top 10 winning teams
plt.figure(figsize=(10, 5))
sns.barplot(x=winning_teams[:10].index, y=winning_teams[:10].values, palette='viridis')
plt.xlabel("Team ID")
plt.ylabel("Wins")
plt.title("Top 10 Winning Teams")
plt.show()





