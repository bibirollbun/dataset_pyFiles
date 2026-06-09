import os

# List files in the dataset directory
dataset_path = "/kaggle/input/march-machine-learning-mania-2025"
print(os.listdir(dataset_path))






import pandas as pd
import os

# Define dataset path
data_path = "/kaggle/input/march-machine-learning-mania-2025/"

# Load key datasets
teams_men = pd.read_csv(data_path + "MTeams.csv")
teams_women = pd.read_csv(data_path + "WTeams.csv")
seasons_men = pd.read_csv(data_path + "MSeasons.csv")
seasons_women = pd.read_csv(data_path + "WSeasons.csv")
tourney_seeds_men = pd.read_csv(data_path + "MNCAATourneySeeds.csv")
tourney_seeds_women = pd.read_csv(data_path + "WNCAATourneySeeds.csv")
games_men = pd.read_csv(data_path + "MRegularSeasonCompactResults.csv")
games_women = pd.read_csv(data_path + "WRegularSeasonCompactResults.csv")
tourney_results_men = pd.read_csv(data_path + "MNCAATourneyCompactResults.csv")
tourney_results_women = pd.read_csv(data_path + "WNCAATourneyCompactResults.csv")


datasets = {
    "teams_men": teams_men,
    "teams_women": teams_women,
    "seasons_men": seasons_men,
    "seasons_women": seasons_women,
    "tourney_seeds_men": tourney_seeds_men,
    "tourney_seeds_women": tourney_seeds_women,
    "games_men": games_men,
    "games_women": games_women,
    "tourney_results_men": tourney_results_men,
    "tourney_results_women": tourney_results_women
}

for name, df in datasets.items():
    print(f"{name}: {df.shape}")


for name, df in datasets.items():
    print(f"\nColumns in {name}:")
    print(df.columns)


for name, df in datasets.items():
    print(f"\n{name} Data Types:")
    print(df.dtypes)



# Display first few rows of key datasets
print("Men's Teams:")
display(teams_men.head())

print("\nWomen's Teams:")
display(teams_women.head())

print("\nMen's Seasons:")
display(seasons_men.head())

print("\nMen's Regular Season Game Results:")
display(games_men.head())

print("\nMen's Tournament Seeds:")
display(tourney_seeds_men.head())

print("\nWomen's Tournament Seeds:")
display(tourney_seeds_women.head())


print("\nMen's NCAA Tournament Results:")
display(tourney_results_men.head())

print("\nWomen's NCAA Tournament Results:")
display(tourney_results_women.head())


# Check for missing values and data types
print("\nMen's Teams Dataset Info:")
print(teams_men.info())

print("\nMen's Regular Season Results Info:")
print(games_men.info())

print("\nMen's NCAA Tournament Results Info:")
print(tourney_results_men.info())


# Check for missing values in all datasets


for name, df in datasets.items():
    missing = df.isnull().sum()
    print(f"\nMissing values in {name}:")
    print(missing[missing > 0])


for name, df in datasets.items():
    duplicate_rows = df.duplicated().sum()
    print(f"{name} has {duplicate_rows} duplicate rows")


datasets["games_men"].drop_duplicates(inplace=True)
datasets["games_women"].drop_duplicates(inplace=True)


print("Unique teams in Men's Data:", teams_men["TeamID"].nunique())
print("Unique teams in Women's Data:", teams_women["TeamID"].nunique())


print("Men's seasons covered:", datasets["seasons_men"]["Season"].unique())
print()
print("Women's seasons covered:", datasets["seasons_women"]["Season"].unique())


# Display basic statistics for the men's regular season dataset
print("\nSummary Statistics for Men's Games Data:")
display(games_men.describe())



# Loop through all datasets and display summary statistics
for name, df in datasets.items():
    print(f"\nðŸ”¹ Summary Statistics for {name.replace('_', ' ').title()}:\n")
    display(df.describe())


import matplotlib.pyplot as plt
import seaborn as sns

# Count number of games per season for Men
games_per_season_men = datasets["games_men"].groupby("Season")["DayNum"].count()

# Count number of games per season for Women
games_per_season_women = datasets["games_women"].groupby("Season")["DayNum"].count()

# Plot the number of games per season
plt.figure(figsize=(12, 5))
sns.lineplot(x=games_per_season_men.index, y=games_per_season_men.values, marker="o", label="Men")
sns.lineplot(x=games_per_season_women.index, y=games_per_season_women.values, marker="s", label="Women")
plt.title("Number of Games Played Per Season (Men & Women)")
plt.xlabel("Season")
plt.ylabel("Number of Games")
plt.legend()
plt.show()



# Calculate average winning score per season (Men)
avg_score_per_season_men = datasets["games_men"].groupby("Season")["WScore"].mean()

# Calculate average winning score per season (Women)
avg_score_per_season_women = datasets["games_women"].groupby("Season")["WScore"].mean()

# Plot trends
plt.figure(figsize=(12, 5))
sns.lineplot(x=avg_score_per_season_men.index, y=avg_score_per_season_men.values, marker="o", label="Men")
sns.lineplot(x=avg_score_per_season_women.index, y=avg_score_per_season_women.values, marker="s", label="Women")
plt.title("Average Winning Scores Per Season (Men & Women)")
plt.xlabel("Season")
plt.ylabel("Average Winning Score")
plt.legend()
plt.show()


# Count total regular-season games (Men & Women)
total_regular_games_men = datasets["games_men"].shape[0]
total_regular_games_women = datasets["games_women"].shape[0]

# Count total tournament games (Men & Women)
total_tourney_games_men = datasets["tourney_results_men"].shape[0]
total_tourney_games_women = datasets["tourney_results_women"].shape[0]

# Create a bar chart
plt.figure(figsize=(10, 5))
plt.bar(["Regular Season (Men)", "Tournament (Men)", "Regular Season (Women)", "Tournament (Women)"], 
        [total_regular_games_men, total_tourney_games_men, total_regular_games_women, total_tourney_games_women], 
        color=["blue", "darkblue", "red", "darkred"])
plt.title("Regular Season vs Tournament Games (Men & Women)")
plt.ylabel("Number of Games")
plt.show()


# Count home, away, and neutral games (Men)
home_wins_men = datasets["games_men"][datasets["games_men"]["WLoc"] == "H"].shape[0]
away_wins_men = datasets["games_men"][datasets["games_men"]["WLoc"] == "A"].shape[0]
neutral_wins_men = datasets["games_men"][datasets["games_men"]["WLoc"] == "N"].shape[0]

# Count home, away, and neutral games (Women)
home_wins_women = datasets["games_women"][datasets["games_women"]["WLoc"] == "H"].shape[0]
away_wins_women = datasets["games_women"][datasets["games_women"]["WLoc"] == "A"].shape[0]
neutral_wins_women = datasets["games_women"][datasets["games_women"]["WLoc"] == "N"].shape[0]

# Create Pie Charts
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].pie([home_wins_men, away_wins_men, neutral_wins_men], labels=["Home", "Away", "Neutral"], autopct="%1.1f%%", colors=["blue", "red", "gray"])
ax[0].set_title("Men's Home vs. Away vs. Neutral Wins")

ax[1].pie([home_wins_women, away_wins_women, neutral_wins_women], labels=["Home", "Away", "Neutral"], autopct="%1.1f%%", colors=["pink", "purple", "gray"])
ax[1].set_title("Women's Home vs. Away vs. Neutral Wins")

plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create temporary copies of data (DOES NOT MODIFY ORIGINAL DATASET)
temp_tourney_men = tourney_results_men.copy()
temp_tourney_women = tourney_results_women.copy()
temp_seeds_men = tourney_seeds_men.copy()
temp_seeds_women = tourney_seeds_women.copy()


# Check column names before merging
print("\nMen's Tournament Results Columns:", temp_tourney_men.columns)
print("\nMen's Tournament Seeds Columns:", temp_seeds_men.columns)
print("\nWomen's Tournament Results Columns:", temp_tourney_women.columns)
print("\nWomen's Tournament Seeds Columns:", temp_seeds_women.columns)

# Merge tournament results with seeds (Men) - Using WTeamID instead of TeamID
temp_tourney_men = temp_tourney_men.merge(temp_seeds_men, left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"], how="left")

# Merge tournament results with seeds (Women) - Using WTeamID instead of TeamID
temp_tourney_women = temp_tourney_women.merge(temp_seeds_women, left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"], how="left")

# Drop extra 'TeamID' column after merging
temp_tourney_men.drop(columns=["TeamID"], inplace=True)
temp_tourney_women.drop(columns=["TeamID"], inplace=True)

# Extract the numeric seed value (Seeds are stored as 'W01', 'X16', etc.)
temp_tourney_men["Seed_Num"] = temp_tourney_men["Seed"].str.extract(r'(\d+)').astype(float)
temp_tourney_women["Seed_Num"] = temp_tourney_women["Seed"].str.extract(r'(\d+)').astype(float)

# Count number of wins per seed
temp_seed_wins_men = temp_tourney_men["Seed_Num"].value_counts().sort_index()
temp_seed_wins_women = temp_tourney_women["Seed_Num"].value_counts().sort_index()

# Plot results
plt.figure(figsize=(12, 5))
sns.barplot(x=temp_seed_wins_men.index, y=temp_seed_wins_men.values, color="blue", label="Men")
sns.barplot(x=temp_seed_wins_women.index, y=temp_seed_wins_women.values, color="red", label="Women", alpha=0.7)
plt.title("Tournament Wins by Seed Ranking (Men & Women)")
plt.xlabel("Seed Number")
plt.ylabel("Number of Wins")
plt.legend()
plt.xticks(rotation=90)
plt.show()


for name, df in datasets.items():
    if "Season" in df.columns:
        print(f"Checking {name} for future data leaks...")
        print(df.groupby("Season").size().tail(10))  # Check if later seasons appear in training data


import seaborn as sns
import matplotlib.pyplot as plt

# Select relevant numeric features
heatmap_features = ["WScore", "LScore", "NumOT"]

# Compute correlation matrix (Men's Regular Season)
corr_matrix_men = games_men[heatmap_features].corr()

# Compute correlation matrix (Women's Regular Season)
corr_matrix_women = games_women[heatmap_features].corr()

# Plot heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(corr_matrix_men, annot=True, cmap="coolwarm", ax=axes[0])
axes[0].set_title("Correlation Heatmap - Men's Regular Season")

sns.heatmap(corr_matrix_women, annot=True, cmap="coolwarm", ax=axes[1])
axes[1].set_title("Correlation Heatmap - Women's Regular Season")

plt.show()



plt.figure(figsize=(14, 6))
sns.boxplot(x="Season", y="WScore", data=games_men)
plt.title("Distribution of Winning Scores Per Season (Men)")
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(14, 6))
sns.boxplot(x="Season", y="WScore", data=games_women)
plt.title("Distribution of Winning Scores Per Season (Women)")
plt.xticks(rotation=90)
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Extract numeric seed values from the temporary dataset
temp_tourney_men["Seed_Num"] = temp_tourney_men["Seed"].str.extract(r'(\d+)').astype(float)
temp_tourney_women["Seed_Num"] = temp_tourney_women["Seed"].str.extract(r'(\d+)').astype(float)

# Scatter plot - Seed vs. Score (Men)
plt.figure(figsize=(10, 5))
sns.scatterplot(x=temp_tourney_men["Seed_Num"], y=temp_tourney_men["WScore"], alpha=0.6)
plt.title("Winning Scores vs. Tournament Seed (Men)")
plt.xlabel("Seed Number (Lower is Better)")
plt.ylabel("Winning Score")
plt.show()

# Scatter plot - Seed vs. Score (Women)
plt.figure(figsize=(10, 5))
sns.scatterplot(x=temp_tourney_women["Seed_Num"], y=temp_tourney_women["WScore"], alpha=0.6, color="red")
plt.title("Winning Scores vs. Tournament Seed (Women)")
plt.xlabel("Seed Number (Lower is Better)")
plt.ylabel("Winning Score")
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Aggregate mean losing score by seed
losing_score_men = temp_tourney_men.groupby("Seed_Num")["LScore"].mean()
losing_score_women = temp_tourney_women.groupby("Seed_Num")["LScore"].mean()

# Bar Chart - Men's Losing Score by Seed
plt.figure(figsize=(12, 5))
sns.barplot(x=losing_score_men.index, y=losing_score_men.values, color="blue", label="Men")
sns.barplot(x=losing_score_women.index, y=losing_score_women.values, color="red", alpha=0.6, label="Women")
plt.title("Average Losing Score by Seed (Men & Women)")
plt.xlabel("Seed Number (Lower is Better)")
plt.ylabel("Average Losing Score")
plt.legend()
plt.xticks(rotation=90)
plt.show()


'''# Box Plot - Winning Margin by Seed (Men)
plt.figure(figsize=(12, 5))
sns.boxplot(x=temp_tourney_men["Seed_Num"], y=temp_tourney_men["WinMargin"])
plt.title("Winning Margin by Seed (Men)")
plt.xlabel("Seed Number (Lower is Better)")
plt.ylabel("Winning Margin")
plt.xticks(rotation=90)
plt.show()

# Box Plot - Winning Margin by Seed (Women)
plt.figure(figsize=(12, 5))
sns.boxplot(x=temp_tourney_women["Seed_Num"], y=temp_tourney_women["WinMargin"], color="red")
plt.title("Winning Margin by Seed (Women)")
plt.xlabel("Seed Number (Lower is Better)")
plt.ylabel("Winning Margin")
plt.xticks(rotation=90)
plt.show()
'''


# Aggregate average overtime games per seed
overtime_men = temp_tourney_men.groupby("Seed_Num")["NumOT"].mean()
overtime_women = temp_tourney_women.groupby("Seed_Num")["NumOT"].mean()

# Line Plot - Overtimes by Seed
plt.figure(figsize=(12, 5))
sns.lineplot(x=overtime_men.index, y=overtime_men.values, marker="o", label="Men", color="blue")
sns.lineplot(x=overtime_women.index, y=overtime_women.values, marker="s", label="Women", color="red")
plt.title("Average Number of Overtimes by Seed (Men & Women)")
plt.xlabel("Seed Number (Lower is Better)")
plt.ylabel("Average Number of Overtimes")
plt.legend()
plt.xticks(rotation=90)
plt.show()



plt.figure(figsize=(12, 5))
sns.histplot(temp_tourney_men["WScore"], kde=True, color="blue", label="Winning Score (Men)", bins=30)
sns.histplot(temp_tourney_men["LScore"], kde=True, color="red", label="Losing Score (Men)", bins=30, alpha=0.6)
plt.title("Distribution of Winning vs. Losing Scores (Men)")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
sns.histplot(temp_tourney_women["WScore"], kde=True, color="blue", label="Winning Score (Women)", bins=30)
sns.histplot(temp_tourney_women["LScore"], kde=True, color="red", label="Losing Score (Women)", bins=30, alpha=0.6)
plt.title("Distribution of Winning vs. Losing Scores (Women)")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# Create dataset with labels
games_men["Type"] = "Regular Season"
temp_tourney_men["Type"] = "Tournament"
combined_men = pd.concat([games_men[["WScore", "Type"]], temp_tourney_men[["WScore", "Type"]]])

games_women["Type"] = "Regular Season"
temp_tourney_women["Type"] = "Tournament"
combined_women = pd.concat([games_women[["WScore", "Type"]], temp_tourney_women[["WScore", "Type"]]])

# Violin Plot - Regular Season vs. Tournament Scores (Men)
plt.figure(figsize=(12, 5))
sns.violinplot(x="Type", y="WScore", data=combined_men)
plt.title("Regular Season vs. Tournament Winning Scores (Men)")
plt.xlabel("Game Type")
plt.ylabel("Winning Score")
plt.show()

# Violin Plot - Regular Season vs. Tournament Scores (Women)
plt.figure(figsize=(12, 5))
sns.violinplot(x="Type", y="WScore", data=combined_women, color="red")
plt.title("Regular Season vs. Tournament Winning Scores (Women)")
plt.xlabel("Game Type")
plt.ylabel("Winning Score")
plt.show()


print("Win rate by seed ranking (Men):")
print(temp_tourney_men.groupby("Seed_Num")["WTeamID"].count() / len(temp_tourney_men))

print("\nWin rate by seed ranking (Women):")
print(temp_tourney_women.groupby("Seed_Num")["WTeamID"].count() / len(temp_tourney_women))


seed_win_rates_men = temp_tourney_men.groupby("Seed_Num")["WTeamID"].count() / temp_tourney_men["Seed_Num"].value_counts()
seed_win_rates_women = temp_tourney_women.groupby("Seed_Num")["WTeamID"].count() / temp_tourney_women["Seed_Num"].value_counts()

plt.figure(figsize=(12, 5))
sns.lineplot(x=seed_win_rates_men.index, y=seed_win_rates_men.values, marker="o", label="Men")
sns.lineplot(x=seed_win_rates_women.index, y=seed_win_rates_women.values, marker="s", label="Women")
plt.title("Tournament Win Rate by Seed (Men vs. Women)")
plt.xlabel("Seed Number")
plt.ylabel("Win Rate")
plt.legend()
plt.show()


# Check for missing values in all datasets
missing_values = {name: df.isnull().sum().sum() for name, df in datasets.items()}
missing_df = pd.DataFrame.from_dict(missing_values, orient="index", columns=["Missing Values"])
missing_df = missing_df[missing_df["Missing Values"] > 0]  # Only show datasets with missing values

# Display missing values overview
print("Missing Values Overview:")
print(missing_df)


# Define expected data types for key datasets
data_types_corrections = {
    "Season": "int",
    "DayNum": "int",
    "WTeamID": "int",
    "LTeamID": "int",
    "WScore": "int",
    "LScore": "int",
    "NumOT": "int"
}

# Apply data type corrections to all relevant datasets
for name, df in datasets.items():
    for col, dtype in data_types_corrections.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)



# Check for duplicate rows in each dataset
duplicate_counts = {name: df.duplicated().sum() for name, df in datasets.items()}
duplicates_df = pd.DataFrame.from_dict(duplicate_counts, orient="index", columns=["Duplicate Rows"])

# Display duplicate rows overview
print("Duplicate Rows Overview:")
print(duplicates_df)

# Remove duplicates if found
for name, df in datasets.items():
    if df.duplicated().sum() > 0:
        datasets[name] = df.drop_duplicates()

print("âœ… Duplicates removed (if any).")


# Check unique values in key categorical columns
categorical_columns = ["WLoc"]  # Example categorical columns
inconsistencies = {}

for name, df in datasets.items():
    for col in categorical_columns:
        if col in df.columns:
            inconsistencies[name + " - " + col] = df[col].unique()

# Display categorical inconsistencies
print("Categorical Data Inconsistencies:")
for key, value in inconsistencies.items():
    print(f"{key}: {value}")


# Count occurrences of each WLoc value in women's tournament data
print("Women's Tournament WLoc Value Counts:")
print(datasets["tourney_results_women"]["WLoc"].value_counts())

# Show sample rows where WLoc is not 'N'
print("\nExamples of Non-Neutral Women's Tournament Games:")
print(datasets["tourney_results_women"][datasets["tourney_results_women"]["WLoc"] != "N"].head(10))


# Compute win rate per team
def compute_win_rate(games_df):
    team_wins = games_df.groupby(["Season", "WTeamID"]).size().reset_index(name="Wins")
    team_losses = games_df.groupby(["Season", "LTeamID"]).size().reset_index(name="Losses")

    # Merge wins and losses
    team_stats = pd.merge(team_wins, team_losses, left_on=["Season", "WTeamID"], right_on=["Season", "LTeamID"], how="outer").fillna(0)

    # Compute total games and win rate
    team_stats["TotalGames"] = team_stats["Wins"] + team_stats["Losses"]
    team_stats["WinRate"] = team_stats["Wins"] / team_stats["TotalGames"]

    # Rename columns
    team_stats = team_stats.rename(columns={"WTeamID": "TeamID"})
    team_stats = team_stats[["Season", "TeamID", "Wins", "Losses", "TotalGames", "WinRate"]]

    return team_stats

# Compute win rates for men and women
win_rate_men = compute_win_rate(games_men)
win_rate_women = compute_win_rate(games_women)

# Display Results
import pandas as pd
print("Men's Win Rate Sample:")
print(win_rate_men.head())
print("\nWomen's Win Rate Sample:")
print(win_rate_women.head())


# Compute average points scored per game
def compute_avg_points_scored(games_df):
    team_points = games_df.groupby(["Season", "WTeamID"])["WScore"].sum().reset_index(name="TotalPointsScored")
    team_games = games_df.groupby(["Season", "WTeamID"]).size().reset_index(name="TotalGames")

    avg_points = pd.merge(team_points, team_games, on=["Season", "WTeamID"])
    avg_points["AvgPointsScored"] = avg_points["TotalPointsScored"] / avg_points["TotalGames"]
    
    return avg_points

# Compute for both genders
avg_points_men = compute_avg_points_scored(games_men)
avg_points_women = compute_avg_points_scored(games_women)

# Display Results
print("Men's Avg Points Scored Sample:")
print(avg_points_men.head())
print("\nWomen's Avg Points Scored Sample:")
print(avg_points_women.head())


# Compute average points allowed per game
def compute_avg_points_allowed(games_df):
    team_points_allowed = games_df.groupby(["Season", "LTeamID"])["WScore"].sum().reset_index(name="TotalPointsAllowed")
    team_games = games_df.groupby(["Season", "LTeamID"]).size().reset_index(name="TotalGames")

    avg_points_allowed = pd.merge(team_points_allowed, team_games, on=["Season", "LTeamID"])
    avg_points_allowed["AvgPointsAllowed"] = avg_points_allowed["TotalPointsAllowed"] / avg_points_allowed["TotalGames"]
    
    return avg_points_allowed

# Compute for both genders
avg_points_allowed_men = compute_avg_points_allowed(games_men)
avg_points_allowed_women = compute_avg_points_allowed(games_women)

# Display Results
print("Men's Avg Points Allowed Sample:")
print(avg_points_allowed_men.head())
print("\nWomen's Avg Points Allowed Sample:")
print(avg_points_allowed_women.head())


print("Columns in avg_points_scored:", avg_points_men.columns)
print("Columns in avg_points_allowed:", avg_points_allowed_men.columns)



# Rename columns for merging
avg_points_men = avg_points_men.rename(columns={"WTeamID": "TeamID"})
avg_points_allowed_men = avg_points_allowed_men.rename(columns={"LTeamID": "TeamID"})
avg_points_women = avg_points_women.rename(columns={"WTeamID": "TeamID"})
avg_points_allowed_women = avg_points_allowed_women.rename(columns={"LTeamID": "TeamID"})


# Function to compute point differential
def compute_point_differential(avg_points_scored, avg_points_allowed):
    point_diff = pd.merge(avg_points_scored, avg_points_allowed, on=["Season", "TeamID"])
    point_diff["PointDifferential"] = point_diff["AvgPointsScored"] - point_diff["AvgPointsAllowed"]
    return point_diff

# Compute for men's and women's teams
point_differential_men = compute_point_differential(avg_points_men, avg_points_allowed_men)
point_differential_women = compute_point_differential(avg_points_women, avg_points_allowed_women)


print("Men's Point Differential Sample:")
print(point_differential_men.head())

print("\nWomen's Point Differential Sample:")
print(point_differential_women.head())


# Load detailed stats
detailed_men = pd.read_csv(data_path + "MRegularSeasonDetailedResults.csv")
detailed_women = pd.read_csv(data_path + "WRegularSeasonDetailedResults.csv")


# Function to compute offensive/defensive efficiency & tempo
def compute_efficiency_stats(detailed_df):
    # Calculate possessions
    detailed_df["W_Possessions"] = detailed_df["WFGA"] - detailed_df["WOR"] + detailed_df["WTO"] + (0.44 * detailed_df["WFTA"])
    detailed_df["L_Possessions"] = detailed_df["LFGA"] - detailed_df["LOR"] + detailed_df["LTO"] + (0.44 * detailed_df["LFTA"])

    # Compute efficiency metrics
    detailed_df["W_OffensiveEff"] = detailed_df["WScore"] / detailed_df["W_Possessions"]
    detailed_df["L_OffensiveEff"] = detailed_df["LScore"] / detailed_df["L_Possessions"]
    detailed_df["W_DefensiveEff"] = detailed_df["LScore"] / detailed_df["W_Possessions"]
    detailed_df["L_DefensiveEff"] = detailed_df["WScore"] / detailed_df["L_Possessions"]

    # Compute tempo (pace of play)
    detailed_df["W_Tempo"] = detailed_df["W_Possessions"] / 2
    detailed_df["L_Tempo"] = detailed_df["L_Possessions"] / 2

    return detailed_df

# Compute metrics for men & women
advanced_stats_men = compute_efficiency_stats(detailed_men)
advanced_stats_women = compute_efficiency_stats(detailed_women)


print("Men's Advanced Stats Sample:")
print(advanced_stats_men[["Season", "WTeamID", "W_OffensiveEff", "W_DefensiveEff", "W_Tempo"]].head())

print("\nWomen's Advanced Stats Sample:")
print(advanced_stats_women[["Season", "WTeamID", "W_OffensiveEff", "W_DefensiveEff", "W_Tempo"]].head())


# Load conference affiliations
team_conferences_men = pd.read_csv(data_path + "MTeamConferences.csv")
team_conferences_women = pd.read_csv(data_path + "WTeamConferences.csv")

# Check structure
print("Men's Conference Data Sample:")
print(team_conferences_men.head())

print("\nWomen's Conference Data Sample:")
print(team_conferences_women.head())



# Recompute team performance metrics (if lost)
def compute_team_stats(games_df):
    team_stats = {}

    for _, row in games_df.iterrows():
        season = row["Season"]
        w_team, l_team = row["WTeamID"], row["LTeamID"]
        w_score, l_score = row["WScore"], row["LScore"]

        # Initialize team stats
        if (season, w_team) not in team_stats:
            team_stats[(season, w_team)] = {"Wins": 0, "Losses": 0, "TotalPointsScored": 0, "TotalPointsAllowed": 0}
        if (season, l_team) not in team_stats:
            team_stats[(season, l_team)] = {"Wins": 0, "Losses": 0, "TotalPointsScored": 0, "TotalPointsAllowed": 0}

        # Update stats
        team_stats[(season, w_team)]["Wins"] += 1
        team_stats[(season, l_team)]["Losses"] += 1
        team_stats[(season, w_team)]["TotalPointsScored"] += w_score
        team_stats[(season, w_team)]["TotalPointsAllowed"] += l_score
        team_stats[(season, l_team)]["TotalPointsScored"] += l_score
        team_stats[(season, l_team)]["TotalPointsAllowed"] += w_score

    # Convert to DataFrame
    team_stats_df = pd.DataFrame.from_dict(team_stats, orient="index")
    team_stats_df.index = pd.MultiIndex.from_tuples(team_stats_df.index, names=["Season", "TeamID"])

    # Compute additional metrics
    team_stats_df["GamesPlayed"] = team_stats_df["Wins"] + team_stats_df["Losses"]
    team_stats_df["WinRate"] = team_stats_df["Wins"] / team_stats_df["GamesPlayed"]
    team_stats_df["AvgPointsScored"] = team_stats_df["TotalPointsScored"] / team_stats_df["GamesPlayed"]
    team_stats_df["AvgPointsAllowed"] = team_stats_df["TotalPointsAllowed"] / team_stats_df["GamesPlayed"]
    team_stats_df["PointDifferential"] = team_stats_df["AvgPointsScored"] - team_stats_df["AvgPointsAllowed"]

    return team_stats_df.reset_index()

# Recompute stats for men and women
team_stats_men = compute_team_stats(games_men)
team_stats_women = compute_team_stats(games_women)


# Merge team win rate data with conference info
team_stats_men = team_stats_men.merge(team_conferences_men, on=["Season", "TeamID"], how="left")
team_stats_women = team_stats_women.merge(team_conferences_women, on=["Season", "TeamID"], how="left")


# Compute average win rate per conference
conference_strength_men = team_stats_men.groupby(["Season", "ConfAbbrev"])["WinRate"].mean().reset_index()
conference_strength_women = team_stats_women.groupby(["Season", "ConfAbbrev"])["WinRate"].mean().reset_index()


print("Men's Conference Strength Sample:")
print(conference_strength_men.head())

print("\nWomen's Conference Strength Sample:")
print(conference_strength_women.head())



# Function to calculate win rates by location
def compute_location_win_rates(games_df):
    location_stats = games_df.groupby("WLoc")["WTeamID"].count()
    total_games = len(games_df)
    win_rates = location_stats / total_games
    return win_rates

# Compute for Men & Women
home_court_advantage_men = compute_location_win_rates(games_men)
home_court_advantage_women = compute_location_win_rates(games_women)

# Display Results
print("ðŸ“Š Home Court Win Rates (Men):")
print(home_court_advantage_men)

print("\nðŸ“Š Home Court Win Rates (Women):")
print(home_court_advantage_women)


import matplotlib.pyplot as plt

# Plot Men's Home Court Advantage
plt.figure(figsize=(8, 5))
home_court_advantage_men.plot(kind="bar", color=["blue", "red", "gray"])
plt.title("Men's Home Court Advantage")
plt.xlabel("Game Location (WLoc)")
plt.ylabel("Win Rate")
plt.xticks(rotation=0)
plt.show()

# Plot Women's Home Court Advantage
plt.figure(figsize=(8, 5))
home_court_advantage_women.plot(kind="bar", color=["blue", "red", "gray"])
plt.title("Women's Home Court Advantage")
plt.xlabel("Game Location (WLoc)")
plt.ylabel("Win Rate")
plt.xticks(rotation=0)
plt.show()


def compute_head_to_head_win_percentage(games_df):
    head_to_head = games_df.groupby(["Season", "WTeamID", "LTeamID"]).size().reset_index(name="Wins")

    # Compute total games played between each team pair
    total_games = games_df.groupby(["Season", "WTeamID", "LTeamID"]).size().reset_index(name="TotalGames")

    # Merge total games with wins
    head_to_head = head_to_head.merge(total_games, on=["Season", "WTeamID", "LTeamID"])
    head_to_head["WinPercentage"] = head_to_head["Wins"] / head_to_head["TotalGames"]

    return head_to_head

# Compute for both menâ€™s and womenâ€™s datasets
head_to_head_men = compute_head_to_head_win_percentage(games_men)
head_to_head_women = compute_head_to_head_win_percentage(games_women)

print("ðŸ“Š Head-to-Head Win Percentage (Men's Teams)")
display(head_to_head_men.head())

print("\nðŸ“Š Head-to-Head Win Percentage (Women's Teams)")
display(head_to_head_women.head())


import pandas as pd
from IPython.display import display

# Function to compute head-to-head score difference
def compute_head_to_head_score_diff(games_df):
    # Compute score difference for each game
    games_df["ScoreDifference"] = games_df["WScore"] - games_df["LScore"]

    # Group by season and team pair
    head_to_head_diff = games_df.groupby(["Season", "WTeamID", "LTeamID"])["ScoreDifference"].mean().reset_index()

    return head_to_head_diff

# Compute for menâ€™s and womenâ€™s datasets
head_to_head_score_men = compute_head_to_head_score_diff(games_men)
head_to_head_score_women = compute_head_to_head_score_diff(games_women)

# Display results using Pandas
print("ðŸ“Š Men's Head-to-Head Score Difference")
display(head_to_head_score_men.head())

print("\nðŸ“Š Women's Head-to-Head Score Difference")
display(head_to_head_score_women.head())


from IPython.display import display

# Function to compute most recent matchup outcome
def compute_recent_matchup_outcome(games_df):
    # Sort by Season and DayNum (most recent games first)
    games_df = games_df.sort_values(by=["Season", "DayNum"], ascending=[False, False])

    # Drop duplicates to keep only the most recent game between two teams
    recent_matchups = games_df.drop_duplicates(subset=["WTeamID", "LTeamID"], keep="first")

    # Select relevant columns
    recent_matchups = recent_matchups[["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]]

    return recent_matchups

# Compute for menâ€™s and womenâ€™s datasets
recent_matchups_men = compute_recent_matchup_outcome(games_men)
recent_matchups_women = compute_recent_matchup_outcome(games_women)

# Display results using Pandas
print("ðŸ“Š Men's Most Recent Matchups")
display(recent_matchups_men.head())

print("\nðŸ“Š Women's Most Recent Matchups")
display(recent_matchups_women.head())


# Load all datasets
team_stats_combined_men = pd.read_csv(data_path + "MRegularSeasonCompactResults.csv")
team_stats_combined_women = pd.read_csv(data_path + "WRegularSeasonCompactResults.csv")
conference_strength_men = pd.read_csv(data_path + "MTeamConferences.csv")
conference_strength_women = pd.read_csv(data_path + "WTeamConferences.csv")
head_to_head_men = pd.read_csv(data_path + "MNCAATourneyCompactResults.csv")
head_to_head_women = pd.read_csv(data_path + "WNCAATourneyCompactResults.csv")
head_to_head_score_men = pd.read_csv(data_path + "MNCAATourneyDetailedResults.csv")
head_to_head_score_women = pd.read_csv(data_path + "WNCAATourneyDetailedResults.csv")
recent_matchups_men = pd.read_csv(data_path + "MRegularSeasonDetailedResults.csv")
recent_matchups_women = pd.read_csv(data_path + "WRegularSeasonDetailedResults.csv")
tourney_seeds_men = pd.read_csv(data_path + "MNCAATourneySeeds.csv")
tourney_seeds_women = pd.read_csv(data_path + "WNCAATourneySeeds.csv")

# Function to print column names
def print_column_names(df, name):
    print(f"Columns in {name}:")
    print(df.columns)
    print("-" * 80)

# Check column names for all relevant datasets
print_column_names(team_stats_combined_men, "team_stats_combined_men")
print_column_names(team_stats_combined_women, "team_stats_combined_women")
print_column_names(conference_strength_men, "conference_strength_men")
print_column_names(conference_strength_women, "conference_strength_women")
print_column_names(head_to_head_men, "head_to_head_men")
print_column_names(head_to_head_women, "head_to_head_women")
print_column_names(head_to_head_score_men, "head_to_head_score_men")
print_column_names(head_to_head_score_women, "head_to_head_score_women")
print_column_names(recent_matchups_men, "recent_matchups_men")
print_column_names(recent_matchups_women, "recent_matchups_women")


# Function to calculate team statistics including both wins and losses
def calculate_team_stats(games):
    # Total Wins for winning teams
    wins = games.groupby(['Season', 'WTeamID']).size().reset_index(name='Wins')
    
    # Total Losses for losing teams
    losses = games.groupby(['Season', 'LTeamID']).size().reset_index(name='Losses')
    
    # Total Points Scored by Winning Teams
    points_scored_winners = games.groupby(['Season', 'WTeamID'])['WScore'].sum().reset_index(name='TotalPointsScored_W')
    
    # Total Points Scored by Losing Teams
    points_scored_losers = games.groupby(['Season', 'LTeamID'])['LScore'].sum().reset_index(name='TotalPointsScored_L')

    # Rename columns for merging consistency
    losses = losses.rename(columns={'LTeamID': 'TeamID'})
    points_scored_losers = points_scored_losers.rename(columns={'LTeamID': 'TeamID'})
    wins = wins.rename(columns={'WTeamID': 'TeamID'})
    points_scored_winners = points_scored_winners.rename(columns={'WTeamID': 'TeamID'})

    # Merge wins and losses
    team_stats = pd.merge(wins, losses, on=['Season', 'TeamID'], how='outer').fillna(0)

    # Merge points scored (by winners and losers)
    team_stats = pd.merge(team_stats, points_scored_winners, on=['Season', 'TeamID'], how='left')
    team_stats = pd.merge(team_stats, points_scored_losers, on=['Season', 'TeamID'], how='left')

    # Compute Total Games Played
    team_stats['GamesPlayed'] = team_stats['Wins'] + team_stats['Losses']

    # Compute Win Rate
    team_stats['WinRate'] = team_stats['Wins'] / team_stats['GamesPlayed']

    # Compute Point Differential (Points Scored - Points Allowed)
    team_stats['TotalPointsScored_W'] = team_stats['TotalPointsScored_W'].fillna(0)
    team_stats['TotalPointsScored_L'] = team_stats['TotalPointsScored_L'].fillna(0)
    team_stats['PointDifferential'] = team_stats['TotalPointsScored_W'] - team_stats['TotalPointsScored_L']

    # Select only necessary columns
    return team_stats[['Season', 'TeamID', 'Wins', 'Losses', 'GamesPlayed', 'WinRate', 'PointDifferential']]

# âœ… Compute Team Statistics for Men and Women
team_stats_men = calculate_team_stats(team_stats_combined_men)
team_stats_women = calculate_team_stats(team_stats_combined_women)

# âœ… Display the first few rows
print("âœ… Computed Team Stats for Men:")
print(team_stats_men.head())

print("âœ… Computed Team Stats for Women:")
print(team_stats_women.head())


# Merge conference strength with team stats
def merge_conference_strength(team_stats, conference_strength):
    """
    Merges the conference strength information with the team statistics dataset.
    """
    # Merge on 'Season' and 'TeamID'
    merged_data = pd.merge(team_stats, conference_strength, on=['Season', 'TeamID'], how='left')
    
    # Display merged data
    print("âœ… Merged Conference Strength Successfully!")
    return merged_data

# âœ… Merge for Men and Women
team_stats_men = merge_conference_strength(team_stats_men, conference_strength_men)
team_stats_women = merge_conference_strength(team_stats_women, conference_strength_women)

# âœ… Display the first few rows
print("âœ… Final Team Stats with Conference Strength (Men):")
print(team_stats_men.head())

print("âœ… Final Team Stats with Conference Strength (Women):")
print(team_stats_women.head())


# Merge tournament seeds with team statistics
def merge_tournament_seeds(team_stats, tourney_seeds):
    """
    Merges the tournament seed information with the team statistics dataset.
    """
    # Merge on 'Season' and 'TeamID'
    merged_data = pd.merge(team_stats, tourney_seeds, on=['Season', 'TeamID'], how='left')

    # Display merged data
    print("âœ… Merged Tournament Seeds Successfully!")
    return merged_data

# âœ… Merge for Men and Women
team_stats_men = merge_tournament_seeds(team_stats_men, tourney_seeds_men)
team_stats_women = merge_tournament_seeds(team_stats_women, tourney_seeds_women)

# âœ… Display the first few rows
print("âœ… Final Team Stats with Tournament Seeds (Men):")
print(team_stats_men.head())

print("âœ… Final Team Stats with Tournament Seeds (Women):")
print(team_stats_women.head())


print("Columns in team_stats_men:", team_stats_men.columns)
print("Columns in conference_strength_men:", conference_strength_men.columns)


# âœ… Compute Conference Strength with Fix
def compute_conference_strength(team_stats, conference_strength):
    """
    Computes conference strength as the average WinRate of all teams in a conference.
    """
    # ðŸ”¹ Debugging: Print columns before merging
    print("Before merge - team_stats columns:", team_stats.columns)
    print("Before merge - conference_strength columns:", conference_strength.columns)

    # ðŸ”¹ Merge conference info
    team_stats = team_stats.merge(conference_strength, on=['Season', 'TeamID'], how='left')

    # ðŸ”¹ Debugging: Print columns after merge
    print("After merge - team_stats columns:", team_stats.columns)

    # âœ… Fix: Use the correct ConfAbbrev column after merging
    if 'ConfAbbrev_y' in team_stats.columns:
        team_stats.rename(columns={'ConfAbbrev_y': 'ConfAbbrev'}, inplace=True)
        team_stats.drop(columns=['ConfAbbrev_x'], errors='ignore', inplace=True)

    # âœ… Fill missing conference values if any
    team_stats['ConfAbbrev'].fillna("UNKNOWN", inplace=True)

    # âœ… Compute conference strength
    conf_strength = team_stats.groupby(['Season', 'ConfAbbrev'])['WinRate'].mean().reset_index()
    conf_strength.rename(columns={'WinRate': 'ConfStrength'}, inplace=True)

    # âœ… Merge computed conference strength back
    team_stats = team_stats.merge(conf_strength, on=['Season', 'ConfAbbrev'], how='left')

    # ðŸ”¹ Debugging: Check final columns
    print("Final team_stats columns:", team_stats.columns)
    
    return team_stats

# âœ… Apply Fix
team_stats_men = compute_conference_strength(team_stats_men, conference_strength_men)
team_stats_women = compute_conference_strength(team_stats_women, conference_strength_women)

print("âœ… Conference Strength Computed & Merged Successfully!")


# Function to extract numeric seed values
def extract_seed_number(df):
    df = df.copy()
    df['Seed'] = df['Seed'].str.extract('(\d+)')  # Extract only the numeric part
    df['Seed'] = pd.to_numeric(df['Seed'])  # Convert to integer
    return df

# âœ… Apply extraction
tourney_seeds_men = extract_seed_number(tourney_seeds_men)
tourney_seeds_women = extract_seed_number(tourney_seeds_women)

# âœ… Display sample
print("âœ… Extracted Seed Numbers Successfully!")
print(tourney_seeds_men.head())
print(tourney_seeds_women.head())


print("Columns in tourney_seeds_men:", tourney_seeds_men.columns)
print("Columns in tourney_seeds_women:", tourney_seeds_women.columns)
print("Columns in team_stats_men before merge:", team_stats_men.columns)
print("Columns in team_stats_women before merge:", team_stats_women.columns)


# âœ… Merge tournament seeds into team stats
team_stats_men = team_stats_men.merge(tourney_seeds_men, on=['Season', 'TeamID'], how='left')
team_stats_women = team_stats_women.merge(tourney_seeds_women, on=['Season', 'TeamID'], how='left')

# âœ… Resolve duplicate "Seed" columns
team_stats_men['Seed'] = team_stats_men['Seed_y'].fillna(team_stats_men['Seed_x'])
team_stats_women['Seed'] = team_stats_women['Seed_y'].fillna(team_stats_women['Seed_x'])

# âœ… Drop unnecessary columns
team_stats_men.drop(columns=['Seed_x', 'Seed_y'], inplace=True)
team_stats_women.drop(columns=['Seed_x', 'Seed_y'], inplace=True)

# âœ… Print columns to verify
print("Columns in team_stats_men after fixing seeds:", team_stats_men.columns)
print("Columns in team_stats_women after fixing seeds:", team_stats_women.columns)

# âœ… Fill missing values
team_stats_men['Seed'].fillna(99, inplace=True)
team_stats_women['Seed'].fillna(99, inplace=True)

# âœ… Convert to integer
team_stats_men['Seed'] = team_stats_men['Seed'].astype(int)
team_stats_women['Seed'] = team_stats_women['Seed'].astype(int)

print("âœ… Tournament Seeds Merged Successfully!")
print(team_stats_men.head())
print(team_stats_women.head())


def generate_matchup_features(team_stats, match_results):
    """Generate matchup-based features for each game"""
    matchups = []

    for season in match_results['Season'].unique():
        season_stats = team_stats[team_stats['Season'] == season]
        season_matches = match_results[match_results['Season'] == season]

        for _, row in season_matches.iterrows():
            teamA = row['WTeamID']
            teamB = row['LTeamID']

            # Ensure both teams exist in the stats dataset
            if teamA in season_stats['TeamID'].values and teamB in season_stats['TeamID'].values:
                row_data = {
                    'Season': season,
                    'TeamA': teamA,
                    'TeamB': teamB,
                    'TeamA_WinRate': season_stats.loc[season_stats['TeamID'] == teamA, 'WinRate'].values[0],
                    'TeamB_WinRate': season_stats.loc[season_stats['TeamID'] == teamB, 'WinRate'].values[0],
                    'TeamA_PointDifferential': season_stats.loc[season_stats['TeamID'] == teamA, 'PointDifferential'].values[0],
                    'TeamB_PointDifferential': season_stats.loc[season_stats['TeamID'] == teamB, 'PointDifferential'].values[0],
                    'TeamA_ConfStrength': season_stats.loc[season_stats['TeamID'] == teamA, 'ConfStrength'].values[0],
                    'TeamB_ConfStrength': season_stats.loc[season_stats['TeamID'] == teamB, 'ConfStrength'].values[0],
                    'TeamA_Seed': season_stats.loc[season_stats['TeamID'] == teamA, 'Seed'].values[0],
                    'TeamB_Seed': season_stats.loc[season_stats['TeamID'] == teamB, 'Seed'].values[0],
                    'Label_A': 1,  # âœ… Team A Wins
                    'Label_B': 0   # âœ… Team B Loses
                }

                matchups.append(row_data)

    return pd.DataFrame(matchups)

# âœ… Generate matchups for Men's and Women's tournaments
matchups_men = generate_matchup_features(team_stats_men, head_to_head_men)
matchups_women = generate_matchup_features(team_stats_women, head_to_head_women)

# âœ… Display results
print("âœ… Matchup Features Generated Successfully!")
print(matchups_men.head())
print(matchups_women.head())


print("Columns in matchups_men:", matchups_men.columns)
print("Columns in matchups_women:", matchups_women.columns)


def compute_win_loss_distribution(matchups_df):
    """Compute wins and losses for both TeamA and TeamB."""
    
    # Count Wins and Losses for Team A
    teamA_wins = matchups_df.groupby('TeamA')['Label_A'].sum().reset_index(name='Wins_A')
    teamA_losses = matchups_df.groupby('TeamA')['Label_A'].apply(lambda x: (x == 0).sum()).reset_index(name='Losses_A')

    # Count Wins and Losses for Team B
    teamB_wins = matchups_df.groupby('TeamB')['Label_B'].sum().reset_index(name='Wins_B')
    teamB_losses = matchups_df.groupby('TeamB')['Label_B'].apply(lambda x: (x == 0).sum()).reset_index(name='Losses_B')

    # Merge All Results Together
    team_win_loss = (
        teamA_wins
        .merge(teamA_losses, on='TeamA', how='outer')
        .merge(teamB_wins, left_on='TeamA', right_on='TeamB', how='outer')
        .merge(teamB_losses, left_on='TeamA', right_on='TeamB', how='outer')
        .fillna(0)
    )

    # Drop duplicate TeamB column
    team_win_loss.drop(columns=['TeamB_x', 'TeamB_y'], inplace=True)

    return team_win_loss

# âœ… Compute Wins & Losses for Men and Women
team_win_loss_men = compute_win_loss_distribution(matchups_men)
team_win_loss_women = compute_win_loss_distribution(matchups_women)

# âœ… Display Results
print("Men's Team Win/Loss Distribution (First 10 Rows):")
print(team_win_loss_men.head(10))

print("\nWomen's Team Win/Loss Distribution (First 10 Rows):")
print(team_win_loss_women.head(10))


# âœ… Aggregate Wins/Losses Per Team
team_wins_men = matchups_men.groupby('TeamA')['Label_A'].sum().reset_index(name='Total Wins')
team_losses_men = matchups_men.groupby('TeamA')['Label_B'].sum().reset_index(name='Total Losses')

team_wins_women = matchups_women.groupby('TeamA')['Label_A'].sum().reset_index(name='Total Wins')
team_losses_women = matchups_women.groupby('TeamA')['Label_B'].sum().reset_index(name='Total Losses')

# âœ… Merge to Get a Full Win/Loss Table
team_record_men = pd.merge(team_wins_men, team_losses_men, on='TeamA', how='outer').fillna(0)
team_record_women = pd.merge(team_wins_women, team_losses_women, on='TeamA', how='outer').fillna(0)

# âœ… Display Results
print("âœ… Men's Team Win/Loss Summary:")
print(team_record_men.head(10))

print("\nâœ… Women's Team Win/Loss Summary:")
print(team_record_women.head(10))


from sklearn.model_selection import train_test_split

# Redo train-test split
train_men, val_men = train_test_split(matchups_men, test_size=0.2, random_state=42)
train_women, val_women = train_test_split(matchups_women, test_size=0.2, random_state=42)




# Compute Feature Differences for Men's Data
train_men['WinRateDiff'] = train_men['TeamA_WinRate'] - train_men['TeamB_WinRate']
train_men['PointDifferentialDiff'] = train_men['TeamA_PointDifferential'] - train_men['TeamB_PointDifferential']
train_men['ConfStrengthDiff'] = train_men['TeamA_ConfStrength'] - train_men['TeamB_ConfStrength']
train_men['SeedDiff'] = train_men['TeamA_Seed'] - train_men['TeamB_Seed']

val_men['WinRateDiff'] = val_men['TeamA_WinRate'] - val_men['TeamB_WinRate']
val_men['PointDifferentialDiff'] = val_men['TeamA_PointDifferential'] - val_men['TeamB_PointDifferential']
val_men['ConfStrengthDiff'] = val_men['TeamA_ConfStrength'] - val_men['TeamB_ConfStrength']
val_men['SeedDiff'] = val_men['TeamA_Seed'] - val_men['TeamB_Seed']

# Compute Feature Differences for Women's Data
train_women['WinRateDiff'] = train_women['TeamA_WinRate'] - train_women['TeamB_WinRate']
train_women['PointDifferentialDiff'] = train_women['TeamA_PointDifferential'] - train_women['TeamB_PointDifferential']
train_women['ConfStrengthDiff'] = train_women['TeamA_ConfStrength'] - train_women['TeamB_ConfStrength']
train_women['SeedDiff'] = train_women['TeamA_Seed'] - train_women['TeamB_Seed']

val_women['WinRateDiff'] = val_women['TeamA_WinRate'] - val_women['TeamB_WinRate']
val_women['PointDifferentialDiff'] = val_women['TeamA_PointDifferential'] - val_women['TeamB_PointDifferential']
val_women['ConfStrengthDiff'] = val_women['TeamA_ConfStrength'] - val_women['TeamB_ConfStrength']
val_women['SeedDiff'] = val_women['TeamA_Seed'] - val_women['TeamB_Seed']


print("Columns in train_men:", train_men.columns)
print("Columns in val_men:", val_men.columns)
print("Columns in train_women:", train_women.columns)
print("Columns in val_women:", val_women.columns)


print("ðŸ”¹ Label Distribution in train_men:")
print(train_men[['Label_A', 'Label_B']].sum())

print("\nðŸ”¹ Label Distribution in val_men:")
print(val_men[['Label_A', 'Label_B']].sum())

print("\nðŸ”¹ Label Distribution in train_women:")
print(train_women[['Label_A', 'Label_B']].sum())

print("\nðŸ”¹ Label Distribution in val_women:")
print(val_women[['Label_A', 'Label_B']].sum())


# Fix training labels by ensuring each matchup has both perspectives
train_men_fixed = train_men.melt(id_vars=['Season', 'TeamA', 'TeamB', 'WinRateDiff', 'PointDifferentialDiff',
                                           'ConfStrengthDiff', 'SeedDiff'],
                                  value_vars=['Label_A', 'Label_B'],
                                  var_name='Label_Type', value_name='Label')

train_women_fixed = train_women.melt(id_vars=['Season', 'TeamA', 'TeamB', 'WinRateDiff', 'PointDifferentialDiff',
                                              'ConfStrengthDiff', 'SeedDiff'],
                                     value_vars=['Label_A', 'Label_B'],
                                     var_name='Label_Type', value_name='Label')

val_men_fixed = val_men.melt(id_vars=['Season', 'TeamA', 'TeamB', 'WinRateDiff', 'PointDifferentialDiff',
                                       'ConfStrengthDiff', 'SeedDiff'],
                              value_vars=['Label_A', 'Label_B'],
                              var_name='Label_Type', value_name='Label')

val_women_fixed = val_women.melt(id_vars=['Season', 'TeamA', 'TeamB', 'WinRateDiff', 'PointDifferentialDiff',
                                          'ConfStrengthDiff', 'SeedDiff'],
                                 value_vars=['Label_A', 'Label_B'],
                                 var_name='Label_Type', value_name='Label')

# Drop Label_Type column (not needed)
train_men_fixed.drop(columns=['Label_Type'], inplace=True)
train_women_fixed.drop(columns=['Label_Type'], inplace=True)
val_men_fixed.drop(columns=['Label_Type'], inplace=True)
val_women_fixed.drop(columns=['Label_Type'], inplace=True)

# Confirm the fix
print("âœ… Train-Men Label Distribution:\n", train_men_fixed['Label'].value_counts())
print("âœ… Train-Women Label Distribution:\n", train_women_fixed['Label'].value_counts())


import pandas as pd
import numpy as np

# Define dataset path
data_path = "/kaggle/input/march-machine-learning-mania-2025/"

# Load datasets
teams_men = pd.read_csv(data_path + "MTeams.csv")
teams_women = pd.read_csv(data_path + "WTeams.csv")
seasons = pd.read_csv(data_path + "MSeasons.csv")  # Season structure is the same for men & women
tourney_seeds_men = pd.read_csv(data_path + "MNCAATourneySeeds.csv")
tourney_seeds_women = pd.read_csv(data_path + "WNCAATourneySeeds.csv")
games_men = pd.read_csv(data_path + "MRegularSeasonDetailedResults.csv")
games_women = pd.read_csv(data_path + "WRegularSeasonDetailedResults.csv")
rankings_men = pd.read_csv(data_path + "MMasseyOrdinals.csv")  # No equivalent for women

# Combine Men and Women Games
games = pd.concat([games_men, games_women], ignore_index=True)
tourney_seeds = pd.concat([tourney_seeds_men, tourney_seeds_women], ignore_index=True)

# Create a win ratio feature
def calculate_team_win_ratio(games):
    """
    Computes the win ratio for each team across all seasons.
    """
    win_counts = games.groupby('WTeamID').size().reset_index(name='Wins')
    loss_counts = games.groupby('LTeamID').size().reset_index(name='Losses')

    team_stats = win_counts.merge(loss_counts, left_on='WTeamID', right_on='LTeamID', how='outer').fillna(0)
    team_stats['TeamID'] = team_stats['WTeamID'].combine_first(team_stats['LTeamID'])
    team_stats['WinRatio'] = team_stats['Wins'] / (team_stats['Wins'] + team_stats['Losses'])

    return team_stats[['TeamID', 'WinRatio']]

team_win_ratios = calculate_team_win_ratio(games)

# Merge win ratio into tournament seeds
tourney_seeds = tourney_seeds.merge(team_win_ratios, on="TeamID", how="left")

# Construct all possible matchups
teams = tourney_seeds["TeamID"].unique()
seasons = tourney_seeds["Season"].unique()

matchups = []
for season in seasons:
    season_teams = tourney_seeds[tourney_seeds["Season"] == season]["TeamID"].values
    for i in range(len(season_teams)):
        for j in range(i + 1, len(season_teams)):
            t1, t2 = season_teams[i], season_teams[j]
            matchups.append((season, min(t1, t2), max(t1, t2)))

matchups_df = pd.DataFrame(matchups, columns=["Season", "TeamA", "TeamB"])

# Merge win ratios
matchups_df = matchups_df.merge(team_win_ratios, left_on="TeamA", right_on="TeamID", how="left").rename(columns={"WinRatio": "TeamA_WinRatio"})
matchups_df = matchups_df.merge(team_win_ratios, left_on="TeamB", right_on="TeamID", how="left").rename(columns={"WinRatio": "TeamB_WinRatio"})

# Fill missing ratios with a default value (assume 50% if no history)
matchups_df["TeamA_WinRatio"].fillna(0.5, inplace=True)
matchups_df["TeamB_WinRatio"].fillna(0.5, inplace=True)

# Calculate probability using a simple logistic model
matchups_df["Pred"] = 1 / (1 + np.exp(-(matchups_df["TeamA_WinRatio"] - matchups_df["TeamB_WinRatio"])))

# Create submission file
matchups_df["ID"] = matchups_df["Season"].astype(str) + "_" + \
                    matchups_df["TeamA"].astype(str) + "_" + \
                    matchups_df["TeamB"].astype(str)

submission_df = matchups_df[["ID", "Pred"]]

# Save submission file
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"âœ… Submission file saved as {submission_path}")

from IPython.display import FileLink

# Display a download link for submission.csv
FileLink("submission.csv")



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Define dataset path
data_path = "/kaggle/input/march-machine-learning-mania-2025/"

# Load datasets
teams_men = pd.read_csv(data_path + "MTeams.csv")
teams_women = pd.read_csv(data_path + "WTeams.csv")
seasons_men = pd.read_csv(data_path + "MSeasons.csv")
seasons_women = pd.read_csv(data_path + "WSeasons.csv")
games_men = pd.read_csv(data_path + "MRegularSeasonCompactResults.csv")
games_women = pd.read_csv(data_path + "WRegularSeasonCompactResults.csv")

# Combine men's and women's games into one dataset
games = pd.concat([games_men, games_women], ignore_index=True)

# Feature Engineering: Calculate historical win ratios
team_wins = games.groupby("WTeamID").size().reset_index(name="Wins")
team_losses = games.groupby("LTeamID").size().reset_index(name="Losses")

# Merge wins and losses
team_stats = pd.merge(team_wins, team_losses, left_on="WTeamID", right_on="LTeamID", how="outer").fillna(0)
team_stats["TotalGames"] = team_stats["Wins"] + team_stats["Losses"]
team_stats["WinRatio"] = team_stats["Wins"] / team_stats["TotalGames"]

# Rename columns
team_stats = team_stats.rename(columns={"WTeamID": "TeamID"}).drop(columns=["LTeamID"])

# Prepare matchups for 2025 (generate all possible pairs)
teams = pd.concat([teams_men, teams_women], ignore_index=True)
all_teams = teams["TeamID"].unique()
matchups = [(2025, teamA, teamB) for i, teamA in enumerate(all_teams) for teamB in all_teams[i+1:]]

# Create DataFrame for matchups
matchups_df = pd.DataFrame(matchups, columns=["Season", "TeamA", "TeamB"])

# Merge team win ratios
matchups_df = matchups_df.merge(team_stats, left_on="TeamA", right_on="TeamID", how="left").rename(columns={"WinRatio": "WinRatioA"})
matchups_df = matchups_df.merge(team_stats, left_on="TeamB", right_on="TeamID", how="left").rename(columns={"WinRatio": "WinRatioB"})

# Fill missing win ratios with 0.5 (new teams with no history)
matchups_df["WinRatioA"].fillna(0.5, inplace=True)
matchups_df["WinRatioB"].fillna(0.5, inplace=True)

# Create labels for training (include both perspectives)
games["Result"] = 1  # Winning team = 1

# Flip games to balance labels (Team A wins = 1, Team B wins = 0)
flipped_games = games.rename(columns={"WTeamID": "TeamB", "LTeamID": "TeamA"})
flipped_games["Result"] = 0  # Losing team perspective

# Combine original and flipped games
historical_data = pd.concat([
    games.rename(columns={"WTeamID": "TeamA", "LTeamID": "TeamB"}), flipped_games
], ignore_index=True)

# Merge historical win ratios
historical_data = historical_data.merge(team_stats, left_on="TeamA", right_on="TeamID", how="left").rename(columns={"WinRatio": "WinRatioA"})
historical_data = historical_data.merge(team_stats, left_on="TeamB", right_on="TeamID", how="left").rename(columns={"WinRatio": "WinRatioB"})

# Fill missing values
historical_data["WinRatioA"].fillna(0.5, inplace=True)
historical_data["WinRatioB"].fillna(0.5, inplace=True)

# Define feature and target variables
X = historical_data[["WinRatioA", "WinRatioB"]]
y = historical_data["Result"]

# Check class balance
print(y.value_counts())  # Ensure we have both 1s and 0s

# Train a simple Logistic Regression model
model = LogisticRegression()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Generate predictions for 2025 matchups
matchups_df["Pred"] = model.predict_proba(matchups_df[["WinRatioA", "WinRatioB"]])[:, 1]

# Ensure only 2025 predictions are included
matchups_2025 = matchups_df[matchups_df["Season"] == 2025].copy()

# Create submission ID format (2025_TeamA_TeamB)
matchups_2025["ID"] = matchups_2025["Season"].astype(str) + "_" + \
                      matchups_2025["TeamA"].astype(str) + "_" + \
                      matchups_2025["TeamB"].astype(str)

# Select required columns
submission_df = matchups_2025[["ID", "Pred"]]

# Save to CSV
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"âœ… Submission file saved as {submission_path}, containing only 2025 matchups.")
from IPython.display import FileLink

# Display a download link for submission.csv
FileLink("submission.csv")



import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class MarchMadnessPredictor:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
        self.scaler = StandardScaler()
        self.team_stats = {}

    def load_data(self, base_path):
        """
        Load all necessary data files
        """
        # Load basic team data
        self.men_teams = pd.read_csv(f"{base_path}/MTeams.csv")
        self.women_teams = pd.read_csv(f"{base_path}/WTeams.csv")

        # Load regular season and tournament results
        self.men_regular = pd.read_csv(f"{base_path}/MRegularSeasonDetailedResults.csv")
        self.women_regular = pd.read_csv(f"{base_path}/WRegularSeasonDetailedResults.csv")
        self.men_tourney = pd.read_csv(f"{base_path}/MNCAATourneyDetailedResults.csv")
        self.women_tourney = pd.read_csv(f"{base_path}/WNCAATourneyDetailedResults.csv")

        # Load team rankings for men's only (no rankings for women's teams)
        self.massey = pd.read_csv(f"{base_path}/MMasseyOrdinals.csv")

    def calculate_team_stats(self, games_df, season):
        """
        Calculate per-game average statistics for each team in a given season.
        """
        stats = {}

        for _, game in games_df[games_df['Season'] == season].iterrows():
            for team_id in [game['WTeamID'], game['LTeamID']]:
                if team_id not in stats:
                    stats[team_id] = {'games': 0, 'wins': 0, 'points_scored': 0, 'points_allowed': 0,
                                      'fg_pct': 0, 'fg3_pct': 0, 'ft_pct': 0, 'rebounds': 0, 'assists': 0,
                                      'steals': 0, 'blocks': 0, 'turnovers': 0}
            
            # Update winner stats
            w_stats = stats[game['WTeamID']]
            w_stats['games'] += 1
            w_stats['wins'] += 1
            w_stats['points_scored'] += game['WScore']
            w_stats['points_allowed'] += game['LScore']
            w_stats['fg_pct'] += game['WFGM'] / game['WFGA'] if game['WFGA'] > 0 else 0
            w_stats['fg3_pct'] += game['WFGM3'] / game['WFGA3'] if game['WFGA3'] > 0 else 0
            w_stats['ft_pct'] += game['WFTM'] / game['WFTA'] if game['WFTA'] > 0 else 0
            w_stats['rebounds'] += game['WOR'] + game['WDR']
            w_stats['assists'] += game['WAst']
            w_stats['steals'] += game['WStl']
            w_stats['blocks'] += game['WBlk']
            w_stats['turnovers'] += game['WTO']

            # Update loser stats
            l_stats = stats[game['LTeamID']]
            l_stats['games'] += 1
            l_stats['points_scored'] += game['LScore']
            l_stats['points_allowed'] += game['WScore']
            l_stats['fg_pct'] += game['LFGM'] / game['LFGA'] if game['LFGA'] > 0 else 0
            l_stats['fg3_pct'] += game['LFGM3'] / game['LFGA3'] if game['LFGA3'] > 0 else 0
            l_stats['ft_pct'] += game['LFTM'] / game['LFTA'] if game['LFTA'] > 0 else 0
            l_stats['rebounds'] += game['LOR'] + game['LDR']
            l_stats['assists'] += game['LAst']
            l_stats['steals'] += game['LStl']
            l_stats['blocks'] += game['LBlk']
            l_stats['turnovers'] += game['LTO']

        # Compute per-game averages
        for team_id, team_stats in stats.items():
            games = team_stats['games']
            if games > 0:
                for key in ['points_scored', 'points_allowed', 'fg_pct', 'fg3_pct', 'ft_pct',
                            'rebounds', 'assists', 'steals', 'blocks', 'turnovers']:
                    team_stats[key] /= games
                team_stats['win_pct'] = team_stats['wins'] / games

        return stats

    def prepare_training_data(self, start_season, end_season):
        """
        Prepare training data from historical games.
        """
        X, y = [], []

        for season in range(start_season, end_season + 1):
            # Compute team statistics
            men_stats = self.calculate_team_stats(self.men_regular, season)
            women_stats = self.calculate_team_stats(self.women_regular, season)
            self.team_stats[season] = {**men_stats, **women_stats}

            # Process historical tournament games
            for tourney_df in [self.men_tourney, self.women_tourney]:
                season_games = tourney_df[tourney_df['Season'] == season]
                for _, game in season_games.iterrows():
                    teamA, teamB = game['WTeamID'], game['LTeamID']

                    if teamA in self.team_stats[season] and teamB in self.team_stats[season]:
                        features = self.get_matchup_features(teamA, teamB, season)
                        X.append(features)
                        y.append(1)  # Winner is first team

                        # Reverse matchup
                        features_reversed = self.get_matchup_features(teamB, teamA, season)
                        X.append(features_reversed)
                        y.append(0)  # Winner is second team

        return np.array(X), np.array(y)

    def get_matchup_features(self, teamA, teamB, season):
        """
        Create feature vector for a matchup.
        """
        if season not in self.team_stats:
            return None
        
        stats = self.team_stats[season]
        if teamA not in stats or teamB not in stats:
            return None

        tA, tB = stats[teamA], stats[teamB]

        return [
            tA['win_pct'] - tB['win_pct'],
            tA['points_scored'] - tB['points_scored'],
            tA['points_allowed'] - tB['points_allowed'],
            tA['fg_pct'] - tB['fg_pct'],
            tA['fg3_pct'] - tB['fg3_pct'],
            tA['ft_pct'] - tB['ft_pct'],
            tA['rebounds'] - tB['rebounds'],
            tA['assists'] - tB['assists'],
            tA['steals'] - tB['steals'],
            tA['blocks'] - tB['blocks'],
            tA['turnovers'] - tB['turnovers']
        ]

    def train_model(self, start_season=2003, end_season=2024):
        """
        Train model on historical data.
        """
        X, y = self.prepare_training_data(start_season, end_season)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_2025(self):
        """
        Generate 2025 predictions for all possible matchups.
        """
        predictions = []
        all_teams = pd.concat([self.men_teams['TeamID'], self.women_teams['TeamID']]).unique()

        for teamA, teamB in combinations(sorted(all_teams), 2):
            features = self.get_matchup_features(teamA, teamB, 2024)  # Use most recent stats
            if features:
                pred = self.model.predict_proba(self.scaler.transform([features]))[0][1]
                predictions.append({'ID': f"2025_{teamA}_{teamB}", 'Pred': pred})

        return pd.DataFrame(predictions)

# Run everything
predictor = MarchMadnessPredictor()
predictor.load_data("/kaggle/input/march-machine-learning-mania-2025")
predictor.train_model()
submission = predictor.predict_2025()
print(submission.head())



# Save to CSV
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"âœ… Submission file saved as {submission_path}, containing only 2025 matchups.")
FileLink("submission.csv")

