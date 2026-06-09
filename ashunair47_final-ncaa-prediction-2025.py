import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


!ls -GFlash ../input/march-machine-learning-mania-2025


import pandas as pd
import os
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV


base_path = "../input/march-machine-learning-mania-2025/"  # Change this path as needed


# Load datasets
tourney_results = pd.read_csv(os.path.join(base_path, "MNCAATourneyDetailedResults.csv"))
tourney_seeds = pd.read_csv(os.path.join(base_path, "MNCAATourneySeeds.csv"))
tourney_slots = pd.read_csv(os.path.join(base_path, "MNCAATourneySlots.csv"))
tourney_seed_round_slots = pd.read_csv(os.path.join(base_path, "MNCAATourneySeedRoundSlots.csv"))
seasons = pd.read_csv(os.path.join(base_path, "MSeasons.csv"))

wtourney_results = pd.read_csv(os.path.join(base_path, "WNCAATourneyDetailedResults.csv"))
wtourney_seeds = pd.read_csv(os.path.join(base_path, "WNCAATourneySeeds.csv"))
wtourney_slots = pd.read_csv(os.path.join(base_path, "WNCAATourneySlots.csv"))
# wtourney_seed_round_slots = pd.read_csv(os.path.join(base_path, "WNCAATourneySeedRoundSlots.csv"))
wseasons = pd.read_csv(os.path.join(base_path, "WSeasons.csv"))





# Merge winner and loser seeds
tourney_results = tourney_results.merge(
    tourney_seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how="left"
).rename(columns={'Seed': 'WSeed'}).drop(columns=['TeamID'])

tourney_results = tourney_results.merge(
    tourney_seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how="left"
).rename(columns={'Seed': 'LSeed'}).drop(columns=['TeamID'])


#woman Merge winner and loser seeds
wtourney_results = wtourney_results.merge(
    wtourney_seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how="left"
).rename(columns={'Seed': 'WSeed'}).drop(columns=['TeamID'])

wtourney_results = wtourney_results.merge(
    wtourney_seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how="left"
).rename(columns={'Seed': 'LSeed'}).drop(columns=['TeamID'])


# Merge slots (outer join to prevent data loss)
tourney_results = tourney_results.merge(tourney_slots, on=["Season"], how="outer")
tourney_results = tourney_results.merge(tourney_seed_round_slots, left_on="WSeed", right_on="Seed", how="left").drop(columns=["Seed"])
tourney_results = tourney_results.merge(tourney_seed_round_slots, left_on="LSeed", right_on="Seed", how="left").drop(columns=["Seed"])


#woman Merge slots (outer join to prevent data loss)
wtourney_results = wtourney_results.merge(wtourney_slots, on=["Season"], how="outer")



print("Columns in tourney_results:", tourney_results.columns)


#woman
print("Columns in tourney_results:", wtourney_results.columns)


print(tourney_results["StrongSeed"].unique())


#woman
print(wtourney_results["StrongSeed"].unique())


print(tourney_results["WLoc"].unique())



print(wtourney_results["WLoc"].unique())


tourney_results_cleaned = tourney_results.merge(seasons, on="Season", how="left")


wtourney_results_cleaned = wtourney_results.merge(seasons, on="Season", how="left")


# Drop unnecessary columns
drop_columns = ["DayZero", "RegionW", "RegionX", "RegionY", "RegionZ"]
tourney_results_cleaned.drop(columns=drop_columns, inplace=True, errors="ignore")


# woman Drop unnecessary columns
wdrop_columns = ["DayZero", "RegionW", "RegionX", "RegionY", "RegionZ"]
wtourney_results_cleaned.drop(columns=drop_columns, inplace=True, errors="ignore")


print("Columns in tourney_results:", tourney_results.columns)


#woman
print("Columns in wtourney_results:", wtourney_results.columns)


# Extract Region and Seed Number
tourney_results_cleaned["WRegion"] = tourney_results_cleaned["WSeed"].str[0]
tourney_results_cleaned["LRegion"] = tourney_results_cleaned["LSeed"].str[0]


# Extract Region and Seed Number
wtourney_results_cleaned["WRegion"] = tourney_results_cleaned["WSeed"].str[0]
wtourney_results_cleaned["LRegion"] = tourney_results_cleaned["LSeed"].str[0]


print(tourney_results_cleaned["WRegion"].unique())


print(tourney_results_cleaned["LRegion"].unique())


print(wtourney_results_cleaned["WRegion"].unique())
print(wtourney_results_cleaned["LRegion"].unique())


# Extract play-in flag before removing a/b
tourney_results_cleaned["WPlayIn"] = tourney_results_cleaned["WSeed"].str[-1].isin(["a", "b"]).astype(int)
tourney_results_cleaned["LPlayIn"] = tourney_results_cleaned["LSeed"].str[-1].isin(["a", "b"]).astype(int)


# Woman Extract play-in flag before removing a/b
wtourney_results_cleaned["WPlayIn"] = wtourney_results_cleaned["WSeed"].str[-1].isin(["a", "b"]).astype(int)
wtourney_results_cleaned["LPlayIn"] = wtourney_results_cleaned["LSeed"].str[-1].isin(["a", "b"]).astype(int)


print(tourney_results_cleaned["WPlayIn"].unique())


print(wtourney_results_cleaned["WPlayIn"].unique())


# Remove 'a'/'b' before converting to integer
tourney_results_cleaned["WSeedNum"] = tourney_results_cleaned["WSeed"].str[1:3].str.replace(r"[ab]", "", regex=True).astype(float)
tourney_results_cleaned["LSeedNum"] = tourney_results_cleaned["LSeed"].str[1:3].str.replace(r"[ab]", "", regex=True).astype(float)


# Remove 'a'/'b' before converting to integer
wtourney_results_cleaned["WSeedNum"] = wtourney_results_cleaned["WSeed"].str[1:3].str.replace(r"[ab]", "", regex=True).astype(float)
wtourney_results_cleaned["LSeedNum"] = wtourney_results_cleaned["LSeed"].str[1:3].str.replace(r"[ab]", "", regex=True).astype(float)


print(tourney_results_cleaned["WSeedNum"].unique())


print(wtourney_results_cleaned["WSeedNum"].unique())


# Drop old seed columns if no longer needed
tourney_results_cleaned.drop(columns=["WSeed", "LSeed"], inplace=True)


# Drop old seed columns if no longer needed
wtourney_results_cleaned.drop(columns=["WSeed", "LSeed"], inplace=True)


columns_to_drop = ['StrongSeed', 'WeakSeed', 'GameRound_x', 'GameSlot_x', 'EarlyDayNum_x', 'LateDayNum_x',
                   'GameRound_y', 'GameSlot_y', 'EarlyDayNum_y', 'LateDayNum_y']
tourney_results_cleaned.drop(columns=columns_to_drop, inplace=True, errors="ignore")


wcolumns_to_drop = ['StrongSeed', 'WeakSeed', 'GameRound_x', 'GameSlot_x', 'EarlyDayNum_x', 'LateDayNum_x',
                   'GameRound_y', 'GameSlot_y', 'EarlyDayNum_y', 'LateDayNum_y']
wtourney_results_cleaned.drop(columns=columns_to_drop, inplace=True, errors="ignore")


# Display cleaned columns
print("Updated Columns:", tourney_results_cleaned.columns)


# Display cleaned columns
print("Updated Columns:", wtourney_results_cleaned.columns)


print(tourney_results_cleaned.head())


print(wtourney_results_cleaned.head())


# Load Compact Results
tourney_compact = pd.read_csv(os.path.join(base_path, "MNCAATourneyCompactResults.csv"))


# Load Compact Results
wtourney_compact = pd.read_csv(os.path.join(base_path, "WNCAATourneyCompactResults.csv"))


# Merge with existing results (fill in missing older data)
tourney_results = pd.concat([tourney_compact, tourney_results_cleaned], ignore_index=True)


# Woman Merge with existing results (fill in missing older data)
wtourney_results = pd.concat([wtourney_compact, wtourney_results_cleaned], ignore_index=True)


# Display updated dataset
print(tourney_results.head())  # Check first few rows
print(tourney_results['Season'].min(), "to", tourney_results['Season'].max())  # Check season range


# Display updated dataset
print(wtourney_results.head())  # Check first few rows
print(wtourney_results['Season'].min(), "to", wtourney_results['Season'].max())  # Check season range


# List of numeric columns to fill
numeric_columns = ["WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR",
                   "WAst", "WTO", "WStl", "WBlk", "WPF", "LFGM", "LFGA", "LFGM3",
                   "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF"]


#Woman List of numeric columns to fill
wnumeric_columns = ["WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR",
                   "WAst", "WTO", "WStl", "WBlk", "WPF", "LFGM", "LFGA", "LFGM3",
                   "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF"]


# Compute average values using only 2003+ data
avg_stats = tourney_results[tourney_results["Season"] >= 2003][numeric_columns].mean()


#Woman Compute average values using only 2003+ data
wavg_stats = wtourney_results[wtourney_results["Season"] >= 2003][wnumeric_columns].mean()


# Fill missing values with computed averages
tourney_results[numeric_columns] = tourney_results[numeric_columns].fillna(avg_stats)


# Fill missing values with computed averages
wtourney_results[wnumeric_columns] = wtourney_results[wnumeric_columns].fillna(wavg_stats)


# Fill categorical columns with the most common value
categorical_columns = ["WLoc", "WRegion", "LRegion"]
for col in categorical_columns:
    mode_value = tourney_results[col].mode()[0]  # Get the most common value
    tourney_results[col] = tourney_results[col].fillna(mode_value)


# Fill categorical columns with the most common value
wcategorical_columns = ["WLoc", "WRegion", "LRegion"]
for col in wcategorical_columns:
    wmode_value = wtourney_results[col].mode()[0]  # Get the most common value
    wtourney_results[col] = wtourney_results[col].fillna(wmode_value)


#Fill Missing Slot, WPlayIn, and LPlayIn with 0
tourney_results["Slot"] = tourney_results["Slot"].fillna("Unknown")
tourney_results["WPlayIn"] = tourney_results["WPlayIn"].fillna(0).astype(int)
tourney_results["LPlayIn"] = tourney_results["LPlayIn"].fillna(0).astype(int)


#Woman Fill Missing Slot, WPlayIn, and LPlayIn with 0
wtourney_results["Slot"] = wtourney_results["Slot"].fillna("Unknown")
wtourney_results["WPlayIn"] = wtourney_results["WPlayIn"].fillna(0).astype(int)
wtourney_results["LPlayIn"] = wtourney_results["LPlayIn"].fillna(0).astype(int)


# Check if there are still missing values
missing_values_after = tourney_results.isnull().sum()
print(missing_values_after[missing_values_after > 0])  # Should return empty if all missing values are handled


# Woman Check if there are still missing values
wmissing_values_after = wtourney_results.isnull().sum()
print(wmissing_values_after[wmissing_values_after > 0])  # Should return empty if all missing values are handled


# Fill missing values using tournament compact results (only if available)
for col in ["DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "NumOT"]:
    tourney_results[col] = tourney_results[col].fillna(method="ffill")  # Forward fill


# Fill missing values using tournament compact results (only if available)
for col in ["DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "NumOT"]:
    wtourney_results[col] = wtourney_results[col].fillna(method="ffill")  # Forward fill


# Fill missing seeds with the most common value per team
tourney_results["WSeedNum"] = tourney_results.groupby("WTeamID")["WSeedNum"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 16))
tourney_results["LSeedNum"] = tourney_results.groupby("LTeamID")["LSeedNum"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 16))


# Fill missing seeds with the most common value per team
wtourney_results["WSeedNum"] = wtourney_results.groupby("WTeamID")["WSeedNum"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 16))
wtourney_results["LSeedNum"] = wtourney_results.groupby("LTeamID")["LSeedNum"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 16))


missing_values_after = tourney_results.isnull().sum()
print(missing_values_after[missing_values_after > 0])


wmissing_values_after = wtourney_results.isnull().sum()
print(wmissing_values_after[wmissing_values_after > 0])


print(tourney_results.head())


print(wtourney_results.head())


# Load the team conference data
team_conferences = pd.read_csv(os.path.join(base_path, "MTeamConferences.csv"))


# Load the team conference data
wteam_conferences = pd.read_csv(os.path.join(base_path, "WTeamConferences.csv"))


# Merge Winner Team's Conference (WConf)
tourney_results = tourney_results.merge(
    team_conferences,
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID'],
    how="left"
).rename(columns={'ConfAbbrev': 'WConf'}).drop(columns=['TeamID'])


#Woman Merge Winner Team's Conference (WConf)
wtourney_results = wtourney_results.merge(
    wteam_conferences,
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID'],
    how="left"
).rename(columns={'ConfAbbrev': 'WConf'}).drop(columns=['TeamID'])


# Merge Loser Team's Conference (LConf)
tourney_results = tourney_results.merge(
    team_conferences,
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID'],
    how="left"
).rename(columns={'ConfAbbrev': 'LConf'}).drop(columns=['TeamID'])


#Woman Merge Loser Team's Conference (LConf)
wtourney_results = wtourney_results.merge(
    wteam_conferences,
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID'],
    how="left"
).rename(columns={'ConfAbbrev': 'LConf'}).drop(columns=['TeamID'])


# Load the conference tournament games data
conference_tourney_games = pd.read_csv(os.path.join(base_path,"MConferenceTourneyGames.csv"))


# Woman Load the conference tournament games data
wconference_tourney_games = pd.read_csv(os.path.join(base_path,"WConferenceTourneyGames.csv"))


# Rename columns to avoid confusion
conference_tourney_games = conference_tourney_games.rename(columns={
    "WTeamID": "Conf_WTeamID",
    "LTeamID": "Conf_LTeamID"
})


#Woman Rename columns to avoid confusion
wconference_tourney_games = wconference_tourney_games.rename(columns={
    "WTeamID": "Conf_WTeamID",
    "LTeamID": "Conf_LTeamID"
})


# Merge to check if the game was a conference tournament game
tourney_results = tourney_results.merge(
    conference_tourney_games[['Season', 'ConfAbbrev', 'DayNum', 'Conf_WTeamID', 'Conf_LTeamID']],
    left_on=['Season', 'WTeamID', 'LTeamID'],
    right_on=['Season', 'Conf_WTeamID', 'Conf_LTeamID'],
    how="left"
).rename(columns={'ConfAbbrev': 'ConfTourney'})


# Woman Merge to check if the game was a conference tournament game
wtourney_results = wtourney_results.merge(
    wconference_tourney_games[['Season', 'ConfAbbrev', 'DayNum', 'Conf_WTeamID', 'Conf_LTeamID']],
    left_on=['Season', 'WTeamID', 'LTeamID'],
    right_on=['Season', 'Conf_WTeamID', 'Conf_LTeamID'],
    how="left"
).rename(columns={'ConfAbbrev': 'ConfTourney'})


print(tourney_results.columns)


print(tourney_results[['Season', 'WTeamID', 'WConf', 'LTeamID', 'LConf', 'ConfTourney']].head())


print(tourney_results["ConfTourney"].unique())


print(wtourney_results.columns)
print(wtourney_results[['Season', 'WTeamID', 'WConf', 'LTeamID', 'LConf', 'ConfTourney']].head())
print(wtourney_results["ConfTourney"].unique())


 #Drop unnecessary columns
tourney_results.drop(columns=['DayNum_y', 'DayNum_x'], inplace=True)


 #Woman Drop unnecessary columns
wtourney_results.drop(columns=['DayNum_y', 'DayNum_x'], inplace=True)


print(tourney_results[['Season', 'WTeamID', 'WConf', 'LTeamID', 'LConf', 'ConfTourney','Conf_WTeamID','Conf_LTeamID']].head())


print(wtourney_results[['Season', 'WTeamID', 'WConf', 'LTeamID', 'LConf', 'ConfTourney','Conf_WTeamID','Conf_LTeamID']].head())


print(tourney_results["Conf_WTeamID"].unique())


print(wtourney_results["Conf_WTeamID"].unique())


print(tourney_results["Conf_LTeamID"].unique())


print(wtourney_results["Conf_LTeamID"].unique())


print(tourney_results["LConf"].unique())


print(wtourney_results["LConf"].unique())


# Drop unnecessary columns
tourney_results.drop(columns=['Conf_WTeamID', 'Conf_LTeamID'], inplace=True, errors="ignore")


# Drop unnecessary columns
wtourney_results.drop(columns=['Conf_WTeamID', 'Conf_LTeamID'], inplace=True, errors="ignore")


# Check final columns
print(tourney_results.columns)


# Check final columns
print(wtourney_results.columns)


# Check for duplicate rows
duplicate_count = tourney_results.duplicated().sum()
print(f"Total Duplicate Rows: {duplicate_count}")


# Check for duplicate rows
wduplicate_count = wtourney_results.duplicated().sum()
print(f"Total Duplicate Rows: {wduplicate_count}")


# Check for missing values in each column
missing_values = tourney_results.isnull().sum()
print("\nMissing Values in Each Column:")
print(missing_values[missing_values > 0])  # Show only columns with NaN values


# Check for missing values in each column
wmissing_values = wtourney_results.isnull().sum()
print("\nMissing Values in Each Column:")
print(wmissing_values[wmissing_values > 0])  # Show only columns with NaN values


# Remove duplicate rows
tourney_results.drop_duplicates(inplace=True)

# Verify duplicates are removed
print(f"Total Duplicate Rows after removal: {tourney_results.duplicated().sum()}")


# Remove duplicate rows
wtourney_results.drop_duplicates(inplace=True)

# Verify duplicates are removed
print(f"Total Duplicate Rows after removal: {wtourney_results.duplicated().sum()}")


# Fill missing values with "Unknown" and assign back to the column
tourney_results["ConfTourney"] = tourney_results["ConfTourney"].fillna("Unknown")

# Verify missing values are handled
print(f"Missing Values after filling: {tourney_results['ConfTourney'].isnull().sum()}")


# Fill missing values with "Unknown" and assign back to the column
wtourney_results["ConfTourney"] = wtourney_results["ConfTourney"].fillna("Unknown")

# Verify missing values are handled
print(f"Missing Values after filling: {wtourney_results['ConfTourney'].isnull().sum()}")


print(tourney_results.duplicated().sum())
tourney_results.isna().sum()


print(wtourney_results.duplicated().sum())
wtourney_results.isna().sum()


print("Final Dataset Shape:", tourney_results.shape)


print("Final Dataset Shape:", wtourney_results.shape)


print("Final Columns:", tourney_results.columns)


print("Final Columns:", wtourney_results.columns)


# Drop unnecessary columns
tourney_results.drop(columns=['NumOT'], inplace=True, errors="ignore")


# Drop unnecessary columns
wtourney_results.drop(columns=['NumOT'], inplace=True, errors="ignore")


# Load MTeams.csv
mteams = pd.read_csv(os.path.join(base_path,"MTeams.csv"))


# Load MTeams.csv
wteams = pd.read_csv(os.path.join(base_path,"WTeams.csv"))


# Merge Winning Team Names with Winning Team ID
tourney_results = tourney_results.merge(
    mteams[['TeamID', 'TeamName']],
    left_on="WTeamID",
    right_on="TeamID",
    how="left"
).rename(columns={'TeamName': 'WTeamName'}).drop(columns=['TeamID'])


# Merge Winning Team Names with Winning Team ID
wtourney_results = wtourney_results.merge(
    wteams[['TeamID', 'TeamName']],
    left_on="WTeamID",
    right_on="TeamID",
    how="left"
).rename(columns={'TeamName': 'WTeamName'}).drop(columns=['TeamID'])


# Merge Losing Team Names with Loosing Team ID
tourney_results = tourney_results.merge(
    mteams[['TeamID', 'TeamName']],
    left_on="LTeamID",
    right_on="TeamID",
    how="left"
).rename(columns={'TeamName': 'LTeamName'}).drop(columns=['TeamID'])


# Merge Losing Team Names with Loosing Team ID
wtourney_results = wtourney_results.merge(
    wteams[['TeamID', 'TeamName']],
    left_on="LTeamID",
    right_on="TeamID",
    how="left"
).rename(columns={'TeamName': 'LTeamName'}).drop(columns=['TeamID'])


print("Final Columns after MTeams merge:", tourney_results.columns)


print("Final Columns after WTeams merge:", wtourney_results.columns)


print(tourney_results.head())


print(wtourney_results.head())


# Filter the dataset for Season 2024
season_2024_results = tourney_results[tourney_results["Season"] == 2024]

# Display Winning and Losing Team Names
print(season_2024_results[["Season", "WTeamName", "LTeamName"]])


# Filter the dataset for Season 2024
wseason_2024_results = wtourney_results[wtourney_results["Season"] == 2024]

# Display Winning and Losing Team Names
print(wseason_2024_results[["Season", "WTeamName", "LTeamName"]])


# Count the number of wins per team in 2024
most_winning_team = season_2024_results["WTeamName"].value_counts().idxmax()
print("Most Winning Team in 2024:", most_winning_team)


# Count the number of wins per team in 2024
wmost_winning_team = wseason_2024_results["WTeamName"].value_counts().idxmax()
print("Most Winning Team in 2024:", wmost_winning_team)


# Filter data for 2024 season
season_2024_results = tourney_results[tourney_results["Season"] == 2024]

# Display the last recorded game (final game of the tournament)
final_game = season_2024_results.tail(1)  # Last row is usually the final game
print(final_game[["Season", "WTeamName", "LTeamName"]])


# Filter data for 2024 season
wseason_2024_results = wtourney_results[wtourney_results["Season"] == 2024]

# Display the last recorded game (final game of the tournament)
wfinal_game = wseason_2024_results.tail(1)  # Last row is usually the final game
print(wfinal_game[["Season", "WTeamName", "LTeamName"]])


print(tourney_results.duplicated().sum())
tourney_results.isna().sum()


print(wtourney_results.duplicated().sum())
wtourney_results.isna().sum()


print(tourney_results["ConfTourney"].unique())


print(wtourney_results["ConfTourney"].unique())


print(tourney_results["WConf"].unique())


print(wtourney_results["WConf"].unique())


tourney_results.drop(columns=['ConfTourney'], inplace=True, errors="ignore")


wtourney_results.drop(columns=['ConfTourney'], inplace=True, errors="ignore")


print("Final Columns:", tourney_results.columns)


print("Final Columns:", wtourney_results.columns)


print(tourney_results["WLoc"].unique())


print(wtourney_results["WLoc"].unique())


print(len(tourney_results))


print(len(wtourney_results))


# 1. Winning Margin
tourney_results["WinMargin"] = tourney_results["WScore"] - tourney_results["LScore"]

# 2. Shooting Efficiency
tourney_results["WFG_Pct"] = tourney_results["WFGM"] / tourney_results["WFGA"]
tourney_results["LFG_Pct"] = tourney_results["LFGM"] / tourney_results["LFGA"]

tourney_results["W3P_Pct"] = tourney_results["WFGM3"] / tourney_results["WFGA3"]
tourney_results["L3P_Pct"] = tourney_results["LFGM3"] / tourney_results["LFGA3"]

tourney_results["WFT_Pct"] = tourney_results["WFTM"] / tourney_results["WFTA"]
tourney_results["LFT_Pct"] = tourney_results["LFTM"] / tourney_results["LFTA"]

# 3. Seed-Based Features
tourney_results["Seed_Diff"] = tourney_results["WSeedNum"] - tourney_results["LSeedNum"]
tourney_results["PlayInGame"] = ((tourney_results["WPlayIn"] == 1) | (tourney_results["LPlayIn"] == 1)).astype(int)

# 4. Location Features
tourney_results["HomeAdvantage"] = tourney_results["WLoc"].map({"H": 1, "A": -1, "N": 0})


# 1. Winning Margin
wtourney_results["WinMargin"] = wtourney_results["WScore"] - wtourney_results["LScore"]

# 2. Shooting Efficiency
wtourney_results["WFG_Pct"] = wtourney_results["WFGM"] / wtourney_results["WFGA"]
wtourney_results["LFG_Pct"] = wtourney_results["LFGM"] / wtourney_results["LFGA"]

wtourney_results["W3P_Pct"] = wtourney_results["WFGM3"] / wtourney_results["WFGA3"]
wtourney_results["L3P_Pct"] = wtourney_results["LFGM3"] / wtourney_results["LFGA3"]

wtourney_results["WFT_Pct"] = wtourney_results["WFTM"] / wtourney_results["WFTA"]
wtourney_results["LFT_Pct"] = wtourney_results["LFTM"] / wtourney_results["LFTA"]

# 3. Seed-Based Features
wtourney_results["Seed_Diff"] = wtourney_results["WSeedNum"] - wtourney_results["LSeedNum"]
wtourney_results["PlayInGame"] = ((wtourney_results["WPlayIn"] == 1) | (wtourney_results["LPlayIn"] == 1)).astype(int)

# 4. Location Features
wtourney_results["HomeAdvantage"] = wtourney_results["WLoc"].map({"H": 1, "A": -1, "N": 0})


# Check the updated dataset
print("New Features Added:", tourney_results.columns)
print(tourney_results[["WinMargin", "WFG_Pct", "LFG_Pct", "Seed_Diff", "HomeAdvantage"]].head())
print(len(tourney_results))


# Check the updated dataset
print("New Features Added:", wtourney_results.columns)
print(wtourney_results[["WinMargin", "WFG_Pct", "LFG_Pct", "Seed_Diff", "HomeAdvantage"]].head())
print(len(wtourney_results))


print(tourney_results["HomeAdvantage"].unique())


print(wtourney_results["HomeAdvantage"].unique())


# Check for duplicate rows
duplicate_count = tourney_results.duplicated().sum()
print(f"Total Duplicate Rows: {duplicate_count}")


# Check for duplicate rows
wduplicate_count = wtourney_results.duplicated().sum()
print(f"Total Duplicate Rows: {wduplicate_count}")


# Check for missing values in each column
missing_values = tourney_results.isnull().sum()
print("\nMissing Values in Each Column:")
print(missing_values[missing_values > 0])  # Show only columns with NaN values


# Check for missing values in each column
wmissing_values = wtourney_results.isnull().sum()
print("\nMissing Values in Each Column:")
print(wmissing_values[wmissing_values > 0])  # Show only columns with NaN values


tourney_results = tourney_results.assign(
    WFG_Pct=tourney_results["WFG_Pct"].fillna(0),
    W3P_Pct=tourney_results["W3P_Pct"].fillna(0),
    WFT_Pct=tourney_results["WFT_Pct"].fillna(0),
)
print("Missing Values After Filling:")
print(tourney_results[["WFG_Pct", "W3P_Pct", "WFT_Pct"]].isnull().sum())


wtourney_results = wtourney_results.assign(
    WFG_Pct=wtourney_results["WFG_Pct"].fillna(0),
    W3P_Pct=wtourney_results["W3P_Pct"].fillna(0),
    WFT_Pct=wtourney_results["WFT_Pct"].fillna(0),
)
print("Missing Values After Filling:")
print(wtourney_results[["WFG_Pct", "W3P_Pct", "WFT_Pct"]].isnull().sum())


print(tourney_results["HomeAdvantage"].value_counts())


print(wtourney_results["HomeAdvantage"].value_counts())


import matplotlib.pyplot as plt
tourney_results["WinMargin"].hist(bins=20)
plt.xlabel("Winning Margin")
plt.ylabel("Number of Games")
plt.title("Distribution of Winning Margins")
plt.show()


import matplotlib.pyplot as plt
wtourney_results["WinMargin"].hist(bins=20)
plt.xlabel("Winning Margin")
plt.ylabel("Number of Games")
plt.title("Distribution of Winning Margins")
plt.show()


print(tourney_results["WLoc"].value_counts())


print(wtourney_results["WLoc"].value_counts())


plt.hist(tourney_results["WFG_Pct"], bins=20, alpha=0.5, label="Winning Team FG%")
plt.hist(tourney_results["LFG_Pct"], bins=20, alpha=0.5, label="Losing Team FG%")
plt.xlabel("Field Goal Percentage")
plt.ylabel("Number of Games")
plt.title("Distribution of FG% for Winning & Losing Teams")
plt.legend()
plt.show()


plt.hist(wtourney_results["WFG_Pct"], bins=20, alpha=0.5, label="Winning Team FG%")
plt.hist(wtourney_results["LFG_Pct"], bins=20, alpha=0.5, label="Losing Team FG%")
plt.xlabel("Field Goal Percentage")
plt.ylabel("Number of Games")
plt.title("Distribution of FG% for Winning & Losing Teams")
plt.legend()
plt.show()


plt.hist(tourney_results["WFT_Pct"], bins=20, alpha=0.5, label="Winning Team FT%")
plt.hist(tourney_results["LFT_Pct"], bins=20, alpha=0.5, label="Losing Team FT%")
plt.xlabel("Free Throw Percentage")
plt.ylabel("Number of Games")
plt.title("Distribution of FT% for Winning & Losing Teams")
plt.legend()
plt.show()


plt.hist(wtourney_results["WFT_Pct"], bins=20, alpha=0.5, label="Winning Team FT%")
plt.hist(wtourney_results["LFT_Pct"], bins=20, alpha=0.5, label="Losing Team FT%")
plt.xlabel("Free Throw Percentage")
plt.ylabel("Number of Games")
plt.title("Distribution of FT% for Winning & Losing Teams")
plt.legend()
plt.show()



# Create a copy for losing teams with inverted features
losses = tourney_results.copy()

# Swap columns to balance the dataset
losses["Win"] = 0  # Losing team
losses["Seed_Diff"] = -losses["Seed_Diff"]
losses["WinMargin"] = -losses["WinMargin"]
# Rename columns for consistency
swap_cols = {
    "WTeamID": "LTeamID", "LTeamID": "WTeamID",
    "WSeedNum": "LSeedNum", "LSeedNum": "WSeedNum",
    "WFG_Pct": "LFG_Pct", "LFG_Pct": "WFG_Pct",
    "W3P_Pct": "L3P_Pct", "L3P_Pct": "W3P_Pct",
    "WFT_Pct": "LFT_Pct", "LFT_Pct": "WFT_Pct"
}
losses = losses.rename(columns=swap_cols)

# Original dataset: winners labeled as 1
tourney_results["Win"] = 1

# Combine the two datasets
final_data = pd.concat([tourney_results, losses], ignore_index=True)

# Define features and target variable
features = ["Seed_Diff", "WFG_Pct", "LFG_Pct", "W3P_Pct", "L3P_Pct", "WFT_Pct", "LFT_Pct"]
X = final_data[features]
y = final_data["Win"]



# Create a copy for losing teams with inverted features
wlosses = wtourney_results.copy()

# Swap columns to balance the dataset
wlosses["Win"] = 0  # Losing team
wlosses["Seed_Diff"] = -wlosses["Seed_Diff"]
wlosses["WinMargin"] = -wlosses["WinMargin"]
# Rename columns for consistency
wswap_cols = {
    "WTeamID": "LTeamID", "LTeamID": "WTeamID",
    "WSeedNum": "LSeedNum", "LSeedNum": "WSeedNum",
    "WFG_Pct": "LFG_Pct", "LFG_Pct": "WFG_Pct",
    "W3P_Pct": "L3P_Pct", "L3P_Pct": "W3P_Pct",
    "WFT_Pct": "LFT_Pct", "LFT_Pct": "WFT_Pct"
}
wlosses = wlosses.rename(columns=wswap_cols)

# Original dataset: winners labeled as 1
wtourney_results["Win"] = 1

# Combine the two datasets
wfinal_data = pd.concat([wtourney_results, wlosses], ignore_index=True)

# Define features and target variable
wfeatures = ["Seed_Diff", "WFG_Pct", "LFG_Pct", "W3P_Pct", "L3P_Pct", "WFT_Pct", "LFT_Pct"]
A = wfinal_data[wfeatures]
b = wfinal_data["Win"]


from sklearn.model_selection import train_test_split

# 80-20 Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.model_selection import train_test_split

# 80-20 Train-Test split
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=42)


# Convert datasets into XGBoost DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define model parameters
params = {
    "objective": "binary:logistic",
    "learning_rate": 0.05,  # Reduced learning rate
    "max_depth": 4,
    "eval_metric": "logloss"
}

# Train model with early stopping
xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=300,  # Increased number of trees
    evals=[(dtest, "test")],  # Validation set for early stopping
    early_stopping_rounds=30  # Stop if no improvement in 20 rounds
)


# Convert datasets into XGBoost DMatrix format
wdtrain = xgb.DMatrix(A_train, label=b_train)
wdtest = xgb.DMatrix(A_test, label=b_test)

# Define model parameters
params = {
    "objective": "binary:logistic",
    "learning_rate": 0.05,  # Reduced learning rate
    "max_depth": 4,
    "eval_metric": "logloss"
}

# Train model with early stopping
xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=300,  # Increased number of trees
    evals=[(wdtest, "test")],  # Validation set for early stopping
    early_stopping_rounds=30  # Stop if no improvement in 20 rounds
)


# Make predictions
y_pred_proba = xgb_model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

# Print results
print(f"Accuracy: {accuracy}")
print(f"Log Loss: {logloss}")
print(f"AUC Score: {auc}")


# Make predictions
b_pred_proba = xgb_model.predict(wdtest)
b_pred = (b_pred_proba > 0.5).astype(int)

# Evaluate performance
accuracy = accuracy_score(b_test, b_pred)
logloss = log_loss(b_test, b_pred_proba)
auc = roc_auc_score(b_test, b_pred_proba)

# Print results
print(f"Accuracy: {accuracy}")
print(f"Log Loss: {logloss}")
print(f"AUC Score: {auc}")


train_preds = xgb_model.predict(dtrain)
train_labels = (train_preds > 0.5).astype(int)

train_acc = accuracy_score(y_train, train_labels)
print(f"Training Accuracy: {train_acc}")


wtrain_preds = xgb_model.predict(wdtrain)
wtrain_labels = (wtrain_preds > 0.5).astype(int)

wtrain_acc = accuracy_score(b_train, wtrain_labels)
print(f"Training Accuracy: {wtrain_acc}")


from sklearn.metrics import brier_score_loss

# Make predictions
y_pred_proba = xgb_model.predict(dtest)  # Probabilities
y_pred = (y_pred_proba > 0.5).astype(int)  # Convert to binary predictions

# Compute Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
brier_score = brier_score_loss(y_test, y_pred_proba)  # <-- Added Brier Score Calculation

# Print All Evaluation Results
print(f"Accuracy: {accuracy}")
print(f"Log Loss: {logloss}")
print(f"AUC Score: {auc}")
print(f"Brier Score: {brier_score}")  # <-- Added Brier Score Output


from sklearn.metrics import brier_score_loss

# Make predictions
b_pred_proba = xgb_model.predict(wdtest)  # Probabilities
b_pred = (b_pred_proba > 0.5).astype(int)  # Convert to binary predictions

# Compute Performance Metrics
accuracy = accuracy_score(b_test, b_pred)
logloss = log_loss(b_test, b_pred_proba)
auc = roc_auc_score(b_test, b_pred_proba)
brier_score = brier_score_loss(b_test, b_pred_proba)  # <-- Added Brier Score Calculation

# Woman Print All Evaluation Results
print(f"Accuracy: {accuracy}")
print(f"Log Loss: {logloss}")
print(f"AUC Score: {auc}")
print(f"Womens Champion Brier Score: {brier_score}")  # <-- Added Brier Score Output


# Combine both winning and losing teams
teams_2025 = set(tourney_results["WTeamID"]).union(set(tourney_results["LTeamID"]))


# Combine both winning and losing teams
wteams_2025 = set(wtourney_results["WTeamID"]).union(set(wtourney_results["LTeamID"]))


# Convert back to list
teams_2025 = list(teams_2025)


# Convert back to list
wteams_2025 = list(wteams_2025)


# Generate all possible matchups
import itertools
matchup_list = list(itertools.combinations(teams_2025, 2))


# Generate all possible matchups
import itertools
wmatchup_list = list(itertools.combinations(wteams_2025, 2))


matchups_2025 = pd.DataFrame(matchup_list, columns=["Team1", "Team2"])


wmatchups_2025 = pd.DataFrame(wmatchup_list, columns=["Team1", "Team2"])


print(f"Total possible matchups : {len(matchups_2025)}")
print(matchups_2025.head())


print(f"Total possible matchups women's: {len(wmatchups_2025)}")
print(wmatchups_2025.head())


# Combine stats from both winning and losing teams
team_stats_win = tourney_results.groupby("WTeamID").agg({
    "WFG_Pct": "mean",
    "W3P_Pct": "mean",
    "WFT_Pct": "mean",
    "WSeedNum": "mean"
}).reset_index()
team_stats_win.rename(columns={"WTeamID": "TeamID"}, inplace=True)

team_stats_loss = tourney_results.groupby("LTeamID").agg({
    "LFG_Pct": "mean",
    "L3P_Pct": "mean",
    "LFT_Pct": "mean",
    "LSeedNum": "mean"
}).reset_index()
team_stats_loss.rename(columns={
    "LTeamID": "TeamID",
    "LFG_Pct": "WFG_Pct",  # Rename to match winning stats format
    "L3P_Pct": "W3P_Pct",
    "LFT_Pct": "WFT_Pct",
    "LSeedNum": "WSeedNum"
}, inplace=True)


# Combine stats from both winning and losing teams
wteam_stats_win = wtourney_results.groupby("WTeamID").agg({
    "WFG_Pct": "mean",
    "W3P_Pct": "mean",
    "WFT_Pct": "mean",
    "WSeedNum": "mean"
}).reset_index()
wteam_stats_win.rename(columns={"WTeamID": "TeamID"}, inplace=True)

wteam_stats_loss = wtourney_results.groupby("LTeamID").agg({
    "LFG_Pct": "mean",
    "L3P_Pct": "mean",
    "LFT_Pct": "mean",
    "LSeedNum": "mean"
}).reset_index()
wteam_stats_loss.rename(columns={
    "LTeamID": "TeamID",
    "LFG_Pct": "WFG_Pct",  # Rename to match winning stats format
    "L3P_Pct": "W3P_Pct",
    "LFT_Pct": "WFT_Pct",
    "LSeedNum": "WSeedNum"
}, inplace=True)


# Combine winning and losing team stats into one dataframe
team_stats = pd.concat([team_stats_win, team_stats_loss]).groupby("TeamID").mean().reset_index()

# Display team stats to confirm it looks correct
print(team_stats.head())


# Combine winning and losing team stats into one dataframe
wteam_stats = pd.concat([wteam_stats_win, wteam_stats_loss]).groupby("TeamID").mean().reset_index()

# Display team stats to confirm it looks correct
print(wteam_stats.head())


print("Unique teams in tournament data:", len(set(tourney_results["WTeamID"]).union(set(tourney_results["LTeamID"]))))
print("Teams in final stats:", len(team_stats))


print("Unique teams in tournament data for Women's:", len(set(wtourney_results["WTeamID"]).union(set(wtourney_results["LTeamID"]))))
print("Teams in final stats:", len(wteam_stats))


# Merge stats for Team1
matchups_2025 = matchups_2025.merge(team_stats, left_on="Team1", right_on="TeamID", how="left").drop(columns=["TeamID"])

# Merge stats for Team2
matchups_2025 = matchups_2025.merge(team_stats, left_on="Team2", right_on="TeamID", how="left", suffixes=("_T1", "_T2")).drop(columns=["TeamID"])


# Merge stats for Team1
wmatchups_2025 = wmatchups_2025.merge(wteam_stats, left_on="Team1", right_on="TeamID", how="left").drop(columns=["TeamID"])

# Merge stats for Team2
wmatchups_2025 = wmatchups_2025.merge(wteam_stats, left_on="Team2", right_on="TeamID", how="left", suffixes=("_T1", "_T2")).drop(columns=["TeamID"])


# Check merged data
print(matchups_2025.head())


# Check merged data
print(wmatchups_2025.head())


# Compute feature differences (Team1 - Team2)
matchups_2025["Seed_Diff"] = matchups_2025["WSeedNum_T1"] - matchups_2025["WSeedNum_T2"]
matchups_2025["WFG_Pct"] = matchups_2025["WFG_Pct_T1"] - matchups_2025["WFG_Pct_T2"]
matchups_2025["LFG_Pct"] = matchups_2025["WFG_Pct_T2"] - matchups_2025["WFG_Pct_T1"]
matchups_2025["W3P_Pct"] = matchups_2025["W3P_Pct_T1"] - matchups_2025["W3P_Pct_T2"]
matchups_2025["L3P_Pct"] = matchups_2025["W3P_Pct_T2"] - matchups_2025["W3P_Pct_T1"]
matchups_2025["WFT_Pct"] = matchups_2025["WFT_Pct_T1"] - matchups_2025["WFT_Pct_T2"]
matchups_2025["LFT_Pct"] = matchups_2025["WFT_Pct_T2"] - matchups_2025["WFT_Pct_T1"]

# Keep only relevant columns for prediction
X_matchups = matchups_2025[["Seed_Diff", "WFG_Pct", "LFG_Pct", "W3P_Pct", "L3P_Pct", "WFT_Pct", "LFT_Pct"]]


# Compute feature differences (Team1 - Team2)
wmatchups_2025["Seed_Diff"] = wmatchups_2025["WSeedNum_T1"] - wmatchups_2025["WSeedNum_T2"]
wmatchups_2025["WFG_Pct"] = wmatchups_2025["WFG_Pct_T1"] - wmatchups_2025["WFG_Pct_T2"]
wmatchups_2025["LFG_Pct"] = wmatchups_2025["WFG_Pct_T2"] - wmatchups_2025["WFG_Pct_T1"]
wmatchups_2025["W3P_Pct"] = wmatchups_2025["W3P_Pct_T1"] - wmatchups_2025["W3P_Pct_T2"]
wmatchups_2025["L3P_Pct"] = wmatchups_2025["W3P_Pct_T2"] - wmatchups_2025["W3P_Pct_T1"]
wmatchups_2025["WFT_Pct"] = wmatchups_2025["WFT_Pct_T1"] - wmatchups_2025["WFT_Pct_T2"]
wmatchups_2025["LFT_Pct"] = wmatchups_2025["WFT_Pct_T2"] - wmatchups_2025["WFT_Pct_T1"]

# Keep only relevant columns for prediction
WX_matchups = wmatchups_2025[["Seed_Diff", "WFG_Pct", "LFG_Pct", "W3P_Pct", "L3P_Pct", "WFT_Pct", "LFT_Pct"]]


# Check feature differences
print(X_matchups.head())


# Check feature differences
print(WX_matchups.head())


import xgboost as xgb

# Convert to DMatrix (XGBoost format)
dmatch_2025 = xgb.DMatrix(X_matchups)


import xgboost as xgb

# Convert to DMatrix (XGBoost format)
wdmatch_2025 = xgb.DMatrix(WX_matchups)


# Predict win probabilities
matchups_2025["Win_Prob_Team1"] = xgb_model.predict(dmatch_2025)


# Predict win probabilities
wmatchups_2025["Win_Prob_Team1"] = xgb_model.predict(wdmatch_2025)


# Predict the winner
matchups_2025["Predicted_Winner"] = matchups_2025.apply(
    lambda row: row["Team1"] if row["Win_Prob_Team1"] > 0.5 else row["Team2"], axis=1
)

# Display predictions
print("Predicted matchups for 2025:\n", matchups_2025[["Team1", "Team2", "Win_Prob_Team1", "Predicted_Winner"]].head())


# Predict the winner
wmatchups_2025["Predicted_Winner"] = wmatchups_2025.apply(
    lambda row: row["Team1"] if row["Win_Prob_Team1"] > 0.5 else row["Team2"], axis=1
)

# Display predictions
print("Predicted matchups for 2025:\n", wmatchups_2025[["Team1", "Team2", "Win_Prob_Team1", "Predicted_Winner"]].head())


# Ensure Team1 is always the predicted winner
matchups_2025["Losing_Team"] = matchups_2025.apply(
    lambda row: row["Team2"] if row["Predicted_Winner"] == row["Team1"] else row["Team1"], axis=1
)

matchups_2025["Team1"] = matchups_2025["Predicted_Winner"]
matchups_2025["Team2"] = matchups_2025["Losing_Team"]

# Drop the temporary column
matchups_2025.drop(columns=["Losing_Team"], inplace=True)

# Display sorted matchups
print("Sorted matchups for 2025:\n", matchups_2025[["Team1", "Team2", "Win_Prob_Team1", "Predicted_Winner"]].head())


# Ensure Team1 is always the predicted winner
wmatchups_2025["Losing_Team"] = wmatchups_2025.apply(
    lambda row: row["Team2"] if row["Predicted_Winner"] == row["Team1"] else row["Team1"], axis=1
)

wmatchups_2025["Team1"] = wmatchups_2025["Predicted_Winner"]
wmatchups_2025["Team2"] = wmatchups_2025["Losing_Team"]

# Drop the temporary column
wmatchups_2025.drop(columns=["Losing_Team"], inplace=True)

# Display sorted matchups
print("Sorted matchups for 2025:\n", wmatchups_2025[["Team1", "Team2", "Win_Prob_Team1", "Predicted_Winner"]].head())


import pandas as pd

# Create a new column in the format '2025_Team1_Team2'
matchups_2025["ID"] = matchups_2025.apply(
    lambda row: f"2025_{int(row['Team1'])}_{int(row['Team2'])}", axis=1
)

# Create final_prediction DataFrame with the desired columns
final_prediction = matchups_2025[["ID", "Win_Prob_Team1"]].copy()
final_prediction.rename(columns={"Win_Prob_Team1": "Pred"}, inplace=True)

# Display final predictions
print(final_prediction.head())


import pandas as pd

# Create a new column in the format '2025_Team1_Team2'
wmatchups_2025["ID"] = wmatchups_2025.apply(
    lambda row: f"2025_{int(row['Team1'])}_{int(row['Team2'])}", axis=1
)

# Create final_prediction DataFrame with the desired columns
wfinal_prediction = wmatchups_2025[["ID", "Win_Prob_Team1"]].copy()
wfinal_prediction.rename(columns={"Win_Prob_Team1": "Pred"}, inplace=True)

# Display final predictions
print(wfinal_prediction.head())


print(len(final_prediction))


print(len(wfinal_prediction))


final_prediction.to_csv("SampleSubmissionStage2.csv", index=False)


wfinal_prediction.to_csv("WSampleSubmissionStage2.csv", index=False)


# Step 1: Start with all teams that won at least once
qualified_teams = set(matchups_2025["Predicted_Winner"])
team_wins = matchups_2025["Predicted_Winner"].value_counts().to_dict()  # Track wins per team

# Simulate rounds until only 2 teams remain
while len(qualified_teams) > 2:
    new_round_winners = set()

    for team in qualified_teams:
        # Get matchups involving this team
        potential_matchups = matchups_2025[
            (matchups_2025["Team1"].isin(qualified_teams)) &
            (matchups_2025["Team2"].isin(qualified_teams))
        ]

        # Ensure valid matchups exist
        if not potential_matchups.empty:
            # Count wins for each remaining team
            winners = potential_matchups.groupby("Predicted_Winner").size().reset_index(name="Wins")

            # Select the top teams based on the highest win counts
            top_winners = winners.sort_values(by="Wins", ascending=False)["Predicted_Winner"].tolist()

            # Select top winners for the next round
            new_round_winners.update(top_winners[:len(qualified_teams) // 2])  # Keep strongest half

    # Prevent infinite loop: If no progress, pick top teams manually
    if len(new_round_winners) >= len(qualified_teams):
        print("Tournament stuck, selecting top teams based on total wins.")
        qualified_teams = set(sorted(team_wins, key=team_wins.get, reverse=True)[:2])
        break

    qualified_teams = new_round_winners


# Step 1: Start with all teams that won at least once
wqualified_teams = set(wmatchups_2025["Predicted_Winner"])
wteam_wins = wmatchups_2025["Predicted_Winner"].value_counts().to_dict()  # Track wins per team

# Simulate rounds until only 2 teams remain
while len(wqualified_teams) > 2:
    wnew_round_winners = set()

    for team in wqualified_teams:
        # Get matchups involving this team
        wpotential_matchups = wmatchups_2025[
            (wmatchups_2025["Team1"].isin(wqualified_teams)) &
            (wmatchups_2025["Team2"].isin(wqualified_teams))
        ]

        # Ensure valid matchups exist
        if not wpotential_matchups.empty:
            # Count wins for each remaining team
            wwinners = wpotential_matchups.groupby("Predicted_Winner").size().reset_index(name="Wins")

            # Select the top teams based on the highest win counts
            wtop_winners = wwinners.sort_values(by="Wins", ascending=False)["Predicted_Winner"].tolist()

            # Select top winners for the next round
            wnew_round_winners.update(wtop_winners[:len(wqualified_teams) // 2])  # Keep strongest half

    # Prevent infinite loop: If no progress, pick top teams manually
    if len(wnew_round_winners) >= len(wqualified_teams):
        print("Tournament stuck, selecting top teams based on total wins.")
        wqualified_teams = set(sorted(wteam_wins, key=wteam_wins.get, reverse=True)[:2])
        break

    wqualified_teams = wnew_round_winners


# Step 2: Identify the final two teams
final_teams = list(qualified_teams)
print(f"The final teams are: {final_teams[0]} vs. {final_teams[1]}")


# Step 2: Identify the final two teams
wfinal_teams = list(wqualified_teams)
print(f"The final teams are: {wfinal_teams[0]} vs. {wfinal_teams[1]}")


# Debugging: Check if a final matchup exists
print(f"Checking final matchup for: {final_teams[0]} vs {final_teams[1]}")

# Ensure `final_matchup` is assigned before printing
final_matchup = matchups_2025[
    ((matchups_2025["Team1"] == final_teams[0]) & (matchups_2025["Team2"] == final_teams[1])) |
    ((matchups_2025["Team1"] == final_teams[1]) & (matchups_2025["Team2"] == final_teams[0]))
]

print(final_matchup)  # Now it's correctly defined

# If the matchup exists, use the predicted winner
if not final_matchup.empty:
    final_winner = final_matchup.iloc[0]["Predicted_Winner"]
    print(f"The 2025 NCAA Tournament Champion is: {final_winner}!")
else:
    print("⚠ No final matchup found. Predicting manually...")

    # Get team stats
    team1_stats = team_stats[team_stats["TeamID"] == final_teams[0]]
    team2_stats = team_stats[team_stats["TeamID"] == final_teams[1]]

    if team1_stats.empty or team2_stats.empty:
        print("⚠ Missing team stats. Cannot predict the final game.")
    else:
        # Compute feature differences
        final_matchup_data = pd.DataFrame({
            "Seed_Diff": team1_stats["WSeedNum"].values[0] - team2_stats["WSeedNum"].values[0],
            "WFG_Pct": team1_stats["WFG_Pct"].values[0] - team2_stats["WFG_Pct"].values[0],
            "LFG_Pct": team2_stats["WFG_Pct"].values[0] - team1_stats["WFG_Pct"].values[0],
            "W3P_Pct": team1_stats["W3P_Pct"].values[0] - team2_stats["W3P_Pct"].values[0],
            "L3P_Pct": team2_stats["W3P_Pct"].values[0] - team1_stats["W3P_Pct"].values[0],
            "WFT_Pct": team1_stats["WFT_Pct"].values[0] - team2_stats["WFT_Pct"].values[0],
            "LFT_Pct": team2_stats["WFT_Pct"].values[0] - team1_stats["WFT_Pct"].values[0]
        }, index=[0])

        print("Final matchup data for prediction:", final_matchup_data)

        # Convert to DMatrix
        final_matchup_dmatrix = xgb.DMatrix(final_matchup_data)

        # Predict the final winner
        win_prob = xgb_model.predict(final_matchup_dmatrix)[0]
        final_winner = final_teams[0] if win_prob > 0.5 else final_teams[1]
        print(f"The manually predicted 2025 NCAA Champion is: {final_winner}!")


# Debugging: Check if a final matchup exists
print(f"Checking final matchup for: {wfinal_teams[0]} vs {wfinal_teams[1]}")

# Ensure `final_matchup` is assigned before printing
wfinal_matchup = wmatchups_2025[
    ((wmatchups_2025["Team1"] == wfinal_teams[0]) & (wmatchups_2025["Team2"] == wfinal_teams[1])) |
    ((wmatchups_2025["Team1"] == wfinal_teams[1]) & (wmatchups_2025["Team2"] == wfinal_teams[0]))
]

print(wfinal_matchup)  # Now it's correctly defined

# If the matchup exists, use the predicted winner
if not wfinal_matchup.empty:
    wfinal_winner = wfinal_matchup.iloc[0]["Predicted_Winner"]
    print(f"The 2025 NCAA Tournament Champion is: {wfinal_winner}!")
else:
    print("⚠ No final matchup found. Predicting manually...")

    # Get team stats
    wteam1_stats = wteam_stats[wteam_stats["TeamID"] == wfinal_teams[0]]
    wteam2_stats = wteam_stats[wteam_stats["TeamID"] == wfinal_teams[1]]

    if wteam1_stats.empty or wteam2_stats.empty:
        print("⚠ Missing team stats. Cannot predict the final game.")
    else:
        # Compute feature differences
        wfinal_matchup_data = pd.DataFrame({
            "Seed_Diff": wteam1_stats["WSeedNum"].values[0] - wteam2_stats["WSeedNum"].values[0],
            "WFG_Pct": wteam1_stats["WFG_Pct"].values[0] - wteam2_stats["WFG_Pct"].values[0],
            "LFG_Pct": wteam2_stats["WFG_Pct"].values[0] - wteam1_stats["WFG_Pct"].values[0],
            "W3P_Pct": wteam1_stats["W3P_Pct"].values[0] - wteam2_stats["W3P_Pct"].values[0],
            "L3P_Pct": wteam2_stats["W3P_Pct"].values[0] - wteam1_stats["W3P_Pct"].values[0],
            "WFT_Pct": wteam1_stats["WFT_Pct"].values[0] - wteam2_stats["WFT_Pct"].values[0],
            "LFT_Pct": wteam2_stats["WFT_Pct"].values[0] - wteam1_stats["WFT_Pct"].values[0]
        }, index=[0])

        print("Final matchup data for prediction:", wfinal_matchup_data)

        # Convert to DMatrix
        wfinal_matchup_dmatrix = xgb.DMatrix(wfinal_matchup_data)

        # Predict the final winner
        wwin_prob = xgb_model.predict(wfinal_matchup_dmatrix)[0]
        wfinal_winner = wfinal_teams[0] if wwin_prob > 0.5 else wfinal_teams[1]
        print(f"The manually predicted 2025 NCAA Champion is: {wfinal_winner}!")


# Create a dictionary to map TeamID to TeamName
team_name_mapping = dict(zip(tourney_results["WTeamID"], tourney_results["WTeamName"]))  # Adjust column names if needed


# Check the name of the champion
final_winner_name = team_name_mapping.get(final_winner, f"Unknown Team ({final_winner})")


print(f"The 2025 NCAA Tournament Champion is: {final_winner_name}!")


# Create a dictionary to map TeamID to TeamName
wteam_name_mapping = dict(zip(wtourney_results["WTeamID"], wtourney_results["WTeamName"]))  # Adjust column names if needed


# Check the name of the champion
wfinal_winner_name = wteam_name_mapping.get(wfinal_winner, f"Unknown Team ({wfinal_winner})")


print(f"The 2025 Women's NCAA Tournament Champion is: {wfinal_winner_name}!")







