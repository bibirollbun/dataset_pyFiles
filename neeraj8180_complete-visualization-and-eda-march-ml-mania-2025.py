import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings

warnings.filterwarnings('ignore') # To not display warnings


#files path
data_path = "/kaggle/input/march-machine-learning-mania-2025"


#All the datafiles
csv_files = [
    "Cities.csv", "Conferences.csv", "MConferenceTourneyGames.csv", "MGameCities.csv",
    "MMasseyOrdinals.csv", "MNCAATourneyCompactResults.csv", "MNCAATourneyDetailedResults.csv",
    "MNCAATourneySeedRoundSlots.csv", "MNCAATourneySeeds.csv", "MNCAATourneySlots.csv",
    "MRegularSeasonCompactResults.csv", "MRegularSeasonDetailedResults.csv", "MSeasons.csv",
    "MSecondaryTourneyCompactResults.csv", "MSecondaryTourneyTeams.csv", "MTeamCoaches.csv",
    "MTeamConferences.csv", "MTeamSpellings.csv", "MTeams.csv", "SampleSubmissionStage1.csv",
    "SeedBenchmarkStage1.csv", "WConferenceTourneyGames.csv", "WGameCities.csv",
    "WNCAATourneyCompactResults.csv", "WNCAATourneyDetailedResults.csv", "WNCAATourneySeeds.csv",
    "WNCAATourneySlots.csv", "WRegularSeasonCompactResults.csv", "WRegularSeasonDetailedResults.csv",
    "WSeasons.csv", "WSecondaryTourneyCompactResults.csv", "WSecondaryTourneyTeams.csv",
    "WTeamConferences.csv", "WTeamSpellings.csv", "WTeams.csv"
]


#Number of Files verification
len(csv_files)


# Load all datasets into a dictionary
data = {}

for file in csv_files:
    file_path = os.path.join(data_path, file)
    if os.path.exists(file_path):
        try:
            data[file] = pd.read_csv(file_path, encoding="utf-8")  # Try UTF-8 first
        except UnicodeDecodeError:
            try:
                data[file] = pd.read_csv(file_path, encoding="ISO-8859-1")  # Try ISO-8859-1
            except Exception as e:
                print(f"Error reading {file}: {e}")
    else:
        print(f"File not found: {file}")

print("Data loaded successfully!")


# Display basic info and missing values
for name, df in data.items():
    print(f"\nDataset: {name}")
    print(df.info())
    print("Missing Values:\n", df.isnull().sum())
    print("-" * 50)


# Quick check for duplicates
for name, df in data.items():
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"{name} has {duplicate_count} duplicate rows")


# Summary statistics of numerical features
for name, df in data.items():
    print(f"\nSummary Statistics for {name}:")
    print(df.describe())


#Set visualization style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# 1. Score Distributions - Regular Season & Tournament for both mens and womens
def plot_score_distribution(df, title):
    plt.figure(figsize=(10, 5))
    sns.histplot(df["WScore"], bins=30, kde=True, color='blue', label='Winning Score')
    sns.histplot(df["LScore"], bins=30, kde=True, color='red', label='Losing Score')
    plt.title(title)
    plt.xlabel("Score")
    plt.legend()
    plt.show()

# Men's Tournaments
plot_score_distribution(data["MNCAATourneyCompactResults.csv"], "Men's NCAA Tournament Compact Score Distribution")
plot_score_distribution(data["MNCAATourneyDetailedResults.csv"], "Men's NCAA Tournament Detailed Score Distribution")
plot_score_distribution(data["MRegularSeasonCompactResults.csv"], "Men's Regular Tournament Compact Score Distribution")
plot_score_distribution(data["MRegularSeasonDetailedResults.csv"], "Men's Regular Tournament Detailed Score Distribution")
plot_score_distribution(data["MSecondaryTourneyCompactResults.csv"], "Men's Secondary Tournament Compact Score Distribution")


# Women's Tournaments
plot_score_distribution(data["WNCAATourneyCompactResults.csv"], "Women's NCAA Tournament Compact Score Distribution")
plot_score_distribution(data["WNCAATourneyDetailedResults.csv"], "Women's NCAA Tournament Detailed Score Distribution")
plot_score_distribution(data["WRegularSeasonCompactResults.csv"], "Women's Regular Tournament Compact Score Distribution")
plot_score_distribution(data["WRegularSeasonDetailedResults.csv"], "Women's Regular Tournament Detailed Score Distribution")
plot_score_distribution(data["WSecondaryTourneyCompactResults.csv"], "Women's Secondary Tournament Compact Score Distribution")


# 2. Team Performance - Win Frequency
def plot_team_wins(df, title, team_col="WTeamID"):
    team_wins = df[team_col].value_counts().head(20)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=team_wins.index, y=team_wins.values, palette="viridis")
    plt.title(title)
    plt.xlabel("Team ID")
    plt.ylabel("Wins")
    plt.xticks(rotation=45)
    plt.show()


# Men's Tournaments
plot_team_wins(data["MNCAATourneyCompactResults.csv"], "Top 20 Teams by Wins - Men's NCAA Tourney Compcat")
plot_team_wins(data["MNCAATourneyDetailedResults.csv"], "Top 20 Teams by Wins - Men's NCAA Tourney Detailed")
plot_team_wins(data["MRegularSeasonCompactResults.csv"], "Top 20 Teams by Wins - Men's Regular Season Compact")
plot_team_wins(data["MRegularSeasonDetailedResults.csv"], "Top 20 Teams by Wins - Men's Regular Season Detailed")
plot_team_wins(data["MSecondaryTourneyCompactResults.csv"], "Top 20 Teams by Wins - Men's Secondary Tourney Compact")


# Women's Tournaments
plot_team_wins(data["WNCAATourneyCompactResults.csv"], "Top 20 Teams by Wins - Women's NCAA Tourney Compcat")
plot_team_wins(data["WNCAATourneyDetailedResults.csv"], "Top 20 Teams by Wins - Women's NCAA Tourney Detailed")
plot_team_wins(data["WRegularSeasonCompactResults.csv"], "Top 20 Teams by Wins - Women's Regular Season Compact")
plot_team_wins(data["WRegularSeasonDetailedResults.csv"], "Top 20 Teams by Wins - Women's Regular Season Detailed")
plot_team_wins(data["WSecondaryTourneyCompactResults.csv"], "Top 20 Teams by Wins - Women's Secondary Tourney Compact")


# 3. Seed Analysis - Performance vs. Seed
import re

def extract_seed_number(seed):
    """ Extracts the numeric part of the seed, ignoring letters like 'a' or 'b'. """
    match = re.search(r'\d+', seed)  # Find first numeric part
    return int(match.group()) if match else None  # Convert to integer

def plot_seed_performance(df, results_df, title):
    """ Plots seed performance based on number of wins. """
    df["SeedNum"] = df["Seed"].apply(extract_seed_number)  # Extract seed number
    merged_df = df.merge(results_df, left_on="TeamID", right_on="WTeamID")  # Merge with wins data
    avg_wins = merged_df.groupby("SeedNum")["WTeamID"].count()  # Count wins per seed

    plt.figure(figsize=(10, 5))
    sns.barplot(x=avg_wins.index, y=avg_wins.values, palette="coolwarm")
    plt.xlabel("Seed")
    plt.ylabel("Win Count")
    plt.title(title)
    plt.show()

# Men's Seed Analysis
plot_seed_performance(data["MNCAATourneySeeds.csv"], data["MNCAATourneyCompactResults.csv"], "Men's NCAA Tournament Seeds vs. Wins")

# Women's Seed Analysis
plot_seed_performance(data["WNCAATourneySeeds.csv"], data["WNCAATourneyCompactResults.csv"], "Women's NCAA Tournament Seeds vs. Wins")



# 4. Ranking Trends Over Time
def plot_ranking_trends(df, system_name="POM"):
    df_filtered = df[df["SystemName"] == system_name]
    plt.figure(figsize=(12, 6))
    for team_id in df_filtered["TeamID"].unique()[:5]:  # Plot for first 5 teams
        team_data = df_filtered[df_filtered["TeamID"] == team_id]
        plt.plot(team_data["RankingDayNum"], team_data["OrdinalRank"], label=f"Team {team_id}")
    
    plt.gca().invert_yaxis()  # Lower rank is better
    plt.title(f"Ranking Trends ({system_name} System)")
    plt.xlabel("Ranking Day")
    plt.ylabel("Rank (Lower is Better)")
    plt.legend()
    plt.show()

plot_ranking_trends(data["MMasseyOrdinals.csv"], system_name="POM")


# Define a function to plot trends over years
def plot_trend(df, year_col, value_col, title, xlabel, ylabel):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=year_col, y=value_col, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.show()

# 1ï¸�âƒ£ Game Trends Over the Years (Men & Women)
m_games = data["MRegularSeasonCompactResults.csv"]
w_games = data["WRegularSeasonCompactResults.csv"]

m_games['Season'] = m_games['Season'].astype(int)
w_games['Season'] = w_games['Season'].astype(int)

m_games_per_season = m_games.groupby("Season").size().reset_index(name="NumGames")
w_games_per_season = w_games.groupby("Season").size().reset_index(name="NumGames")

plot_trend(m_games_per_season, "Season", "NumGames", "Men's NCAA Games Over the Years", "Season", "Number of Games")
plot_trend(w_games_per_season, "Season", "NumGames", "Women's NCAA Games Over the Years", "Season", "Number of Games")



# 2ï¸�âƒ£ Home-Court Advantage Analysis
m_game_cities = data["MGameCities.csv"]
w_game_cities = data["WGameCities.csv"]

m_home_wins = m_games[m_games['WLoc'] == 'H'].shape[0] / m_games.shape[0] * 100
w_home_wins = w_games[w_games['WLoc'] == 'H'].shape[0] / w_games.shape[0] * 100

print(f"ğŸ�  Home Win Percentage (Men): {m_home_wins:.2f}%")
print(f"ğŸ�  Home Win Percentage (Women): {w_home_wins:.2f}%")


# 3ï¸�âƒ£ Seed Performance Over the Years
m_seeds = data["MNCAATourneySeeds.csv"]
w_seeds = data["WNCAATourneySeeds.csv"]

# Extracting numeric seed values
m_seeds["SeedNum"] = m_seeds["Seed"].str.extract(r'(\d+)').astype(int)
w_seeds["SeedNum"] = w_seeds["Seed"].str.extract(r'(\d+)').astype(int)

# Visualize seed distribution
plt.figure(figsize=(12, 6))
sns.histplot(m_seeds["SeedNum"], bins=16, kde=True, color="blue", label="Men")
sns.histplot(w_seeds["SeedNum"], bins=16, kde=True, color="red", label="Women")
plt.title("Distribution of Tournament Seeds (Men & Women)")
plt.xlabel("Seed Number")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# 5ï¸�âƒ£ Conference Dominance Analysis
m_team_conferences = data["MTeamConferences.csv"]
w_team_conferences = data["WTeamConferences.csv"]

m_conf_counts = m_team_conferences['ConfAbbrev'].value_counts().head(10)
w_conf_counts = w_team_conferences['ConfAbbrev'].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=m_conf_counts.index, y=m_conf_counts.values, palette='coolwarm')
plt.title("Top 10 Conferences by Number of Teams (Men)")
plt.xlabel("Conference")
plt.ylabel("Number of Teams")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x=w_conf_counts.index, y=w_conf_counts.values, palette='coolwarm')
plt.title("Top 10 Conferences by Number of Teams (Women)")
plt.xlabel("Conference")
plt.ylabel("Number of Teams")
plt.xticks(rotation=45)
plt.show()


# 1ï¸�âƒ£ Win Rate Analysis Over Seasons
win_counts = data["MRegularSeasonCompactResults.csv"].groupby(["Season", "WTeamID"]).size().reset_index(name='Wins')
total_games = data["MRegularSeasonCompactResults.csv"].groupby(["Season", "WTeamID"]).size().add(
                data["MRegularSeasonCompactResults.csv"].groupby(["Season", "LTeamID"]).size(), fill_value=0).reset_index(name='TotalGames')
win_rate = pd.merge(win_counts, total_games, on=["Season", "WTeamID"])
win_rate["WinRate"] = win_rate["Wins"] / win_rate["TotalGames"]

plt.figure(figsize=(12, 6))
sns.lineplot(x='Season', y='WinRate', data=win_rate, ci=None)
plt.title("Win Rate Trends Over Seasons(Men's)")
plt.xlabel("Season")
plt.ylabel("Win Rate")
plt.show()

win_counts_w = data["WRegularSeasonCompactResults.csv"].groupby(["Season", "WTeamID"]).size().reset_index(name='Wins')
total_games_w = data["WRegularSeasonCompactResults.csv"].groupby(["Season", "WTeamID"]).size().add(
                data["WRegularSeasonCompactResults.csv"].groupby(["Season", "LTeamID"]).size(), fill_value=0).reset_index(name='TotalGames')
win_rate_w = pd.merge(win_counts_w, total_games_w, on=["Season", "WTeamID"])
win_rate_w["WinRate"] = win_rate_w["Wins"] / win_rate_w["TotalGames"]

plt.figure(figsize=(12, 6))
sns.lineplot(x='Season', y='WinRate', data=win_rate_w, ci=None)
plt.title("Win Rate Trends Over Seasons (Women's)")
plt.xlabel("Season")
plt.ylabel("Win Rate")
plt.show()


# 2ï¸�âƒ£ Home vs. Away Performance
home_wins = data["MRegularSeasonCompactResults.csv"].groupby(["Season"]).apply(lambda x: (x['WLoc'] == 'H').mean()).reset_index(name='HomeWinRate')
plt.figure(figsize=(12, 6))
sns.lineplot(x='Season', y='HomeWinRate', data=home_wins, marker='o')
plt.title("Home Win Rate Over Seasons(Men's)")
plt.xlabel("Season")
plt.ylabel("Home Win Rate")
plt.show()

# Home vs. Away Performance for Women's Games
home_wins_women = data["WRegularSeasonCompactResults.csv"].groupby(["Season"]).apply(lambda x: (x['WLoc'] == 'H').mean()).reset_index(name='HomeWinRate')

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(x='Season', y='HomeWinRate', data=home_wins_women, marker='o', color='purple', label="Women's Home Win Rate")
plt.title("Women's Home Win Rate Over Seasons")
plt.xlabel("Season")
plt.ylabel("Home Win Rate")
plt.show()



# 3ï¸�âƒ£ Conference Performance Trends (Men's)
conference_wins_men = data["MTeamConferences.csv"].merge(
    data["MRegularSeasonCompactResults.csv"], left_on=["Season", "TeamID"], right_on=["Season", "WTeamID"])

conference_win_rate_men = conference_wins_men.groupby(["Season", "ConfAbbrev"]).size().reset_index(name="Wins")

plt.figure(figsize=(14, 6))
sns.lineplot(x='Season', y='Wins', hue='ConfAbbrev', data=conference_win_rate_men, legend=False)
plt.title("Men's Conference Performance Over Time")
plt.xlabel("Season")
plt.ylabel("Total Wins")
plt.show()

# 3ï¸�âƒ£ Conference Performance Trends (Women's)
conference_wins_women = data["WTeamConferences.csv"].merge(
    data["WRegularSeasonCompactResults.csv"], left_on=["Season", "TeamID"], right_on=["Season", "WTeamID"])

conference_win_rate_women = conference_wins_women.groupby(["Season", "ConfAbbrev"]).size().reset_index(name="Wins")

plt.figure(figsize=(14, 6))
sns.lineplot(x='Season', y='Wins', hue='ConfAbbrev', data=conference_win_rate_women, legend=False)
plt.title("Women's Conference Performance Over Time")
plt.xlabel("Season")
plt.ylabel("Total Wins")
plt.show()



# 4ï¸�âƒ£ Seed Performance in NCAA Tournament (Men)
seeds_men = data["MNCAATourneySeeds.csv"]
tournament_results_men = data["MNCAATourneyCompactResults.csv"]
tournament_seeds_men = tournament_results_men.merge(seeds_men, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])

plt.figure(figsize=(14, 6))
sns.boxplot(x='Seed', y='Season', data=tournament_seeds_men)
plt.title("Men's Tournament Performance by Seed")
plt.xlabel("Seed")
plt.ylabel("Season")
plt.xticks(rotation=90)
plt.show()

# 4ï¸�âƒ£ Seed Performance in NCAA Tournament (Women)
seeds_women = data["WNCAATourneySeeds.csv"]
tournament_results_women = data["WNCAATourneyCompactResults.csv"]
tournament_seeds_women = tournament_results_women.merge(seeds_women, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])

plt.figure(figsize=(14, 6))
sns.boxplot(x='Seed', y='Season', data=tournament_seeds_women)
plt.title("Women's Tournament Performance by Seed")
plt.xlabel("Seed")
plt.ylabel("Season")
plt.xticks(rotation=90)
plt.show()



# Ensure the dataset is correctly assigned
tournament_results = data["MNCAATourneyCompactResults.csv"]
seeds = data["MNCAATourneySeeds.csv"]

# Merge tournament results with seeds for men's teams
cinderella_teams = tournament_results.merge(seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'])

# Group by seed and count the number of times a lower seed lost
cinderella_teams = cinderella_teams.groupby("Seed").size().reset_index(name='Losses')

# Plot Cinderella teams for men's tournament
plt.figure(figsize=(12, 6))
sns.barplot(x='Seed', y='Losses', data=cinderella_teams, palette="viridis")
plt.title("Cinderella Teams - Lower Seeds Losing in Men's Tournament")
plt.xlabel("Seed")
plt.ylabel("Loss Count")
plt.show()


# Ensure women's dataset is correctly assigned
w_tournament_results = data["WNCAATourneyCompactResults.csv"]
w_seeds = data["WNCAATourneySeeds.csv"]

# Merge tournament results with seeds for women's teams
w_cinderella_teams = w_tournament_results.merge(w_seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'])

# Group by seed and count the number of times a lower seed lost
w_cinderella_teams = w_cinderella_teams.groupby("Seed").size().reset_index(name='Losses')

# Plot Cinderella teams for women's tournament
plt.figure(figsize=(12, 6))
sns.barplot(x='Seed', y='Losses', data=w_cinderella_teams, palette="magma")
plt.title("Cinderella Teams - Lower Seeds Losing in Women's Tournament")
plt.xlabel("Seed")
plt.ylabel("Loss Count")
plt.show()



# Score Distributions & Game Margins
data["MRegularSeasonCompactResults.csv"]["PointDiff"] = data["MRegularSeasonCompactResults.csv"]["WScore"] - data["MRegularSeasonCompactResults.csv"]["LScore"]
plt.figure(figsize=(12, 6))
sns.histplot(data["MRegularSeasonCompactResults.csv"], x="PointDiff", bins=30, kde=True)
plt.title("Distribution of Game Margins")
plt.xlabel("Winning Margin")
plt.ylabel("Frequency")
plt.show()

# Compute point difference for Women's Regular Season games
data["WRegularSeasonCompactResults.csv"]["PointDiff"] = (
    data["WRegularSeasonCompactResults.csv"]["WScore"] - data["WRegularSeasonCompactResults.csv"]["LScore"]
)

# Plot the distribution of game margins
plt.figure(figsize=(12, 6))
sns.histplot(data["WRegularSeasonCompactResults.csv"], x="PointDiff", bins=30, kde=True, color="purple")
plt.title("Distribution of Game Margins (Women's Regular Season)")
plt.xlabel("Winning Margin")
plt.ylabel("Frequency")
plt.show()



# Load necessary tournament results and seeds
m_tourney_results = data["MNCAATourneyCompactResults.csv"].copy()
w_tourney_results = data["WNCAATourneyCompactResults.csv"].copy()

m_seeds = data["MNCAATourneySeeds.csv"].copy()
w_seeds = data["WNCAATourneySeeds.csv"].copy()

# Extract numeric seed value (e.g., "W01" -> 1, "X16" -> 16)
m_seeds["SeedNum"] = m_seeds["Seed"].str.extract("(\d+)").astype(int)
w_seeds["SeedNum"] = w_seeds["Seed"].str.extract("(\d+)").astype(int)

# Merge seeds with tournament results to get seed information for both teams
m_tourney_results = m_tourney_results.merge(m_seeds, left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"], how="left").rename(columns={"SeedNum": "WSeed"})
m_tourney_results = m_tourney_results.merge(m_seeds, left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"], how="left").rename(columns={"SeedNum": "LSeed"})

w_tourney_results = w_tourney_results.merge(w_seeds, left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"], how="left").rename(columns={"SeedNum": "WSeed"})
w_tourney_results = w_tourney_results.merge(w_seeds, left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"], how="left").rename(columns={"SeedNum": "LSeed"})

# Define upsets as cases where the winning team had a worse (higher) seed than the losing team
m_upsets = m_tourney_results[m_tourney_results["WSeed"] > m_tourney_results["LSeed"]]
w_upsets = w_tourney_results[w_tourney_results["WSeed"] > w_tourney_results["LSeed"]]

# Calculate upset percentages
m_upset_percentage = (len(m_upsets) / len(m_tourney_results)) * 100
w_upset_percentage = (len(w_upsets) / len(w_tourney_results)) * 100

print(f"ğŸ”¥ Upset Percentage (Men): {m_upset_percentage:.2f}%")
print(f"ğŸ”¥ Upset Percentage (Women): {w_upset_percentage:.2f}%")

# Visualization of upset distributions
plt.figure(figsize=(10, 5))
sns.histplot(m_upsets["WSeed"], bins=16, kde=True, label="Men", color="blue")
sns.histplot(w_upsets["WSeed"], bins=16, kde=True, label="Women", color="red")
plt.xlabel("Seed Number of Upset Winner")
plt.ylabel("Frequency")
plt.title("Upsets Distribution by Seed (Men vs Women)")
plt.legend()
plt.show()



# Extract necessary columns for men's and women's regular season results
m_games = data["MRegularSeasonCompactResults.csv"]
w_games = data["WRegularSeasonCompactResults.csv"]

# Compute clutch games (margin <= 5 points)
m_clutch_games = m_games[abs(m_games['WScore'] - m_games['LScore']) <= 5]
w_clutch_games = w_games[abs(w_games['WScore'] - w_games['LScore']) <= 5]

# Calculate clutch game percentage
m_clutch_rate = len(m_clutch_games) / len(m_games) * 100
w_clutch_rate = len(w_clutch_games) / len(w_games) * 100

print(f"â�³ Clutch Games Percentage (Men): {m_clutch_rate:.2f}%")
print(f"â�³ Clutch Games Percentage (Women): {w_clutch_rate:.2f}%")

# Add margin column for visualization
m_games["Margin"] = m_games["WScore"] - m_games["LScore"]
w_games["Margin"] = w_games["WScore"] - w_games["LScore"]

# Filter close games
m_close_games = m_games[m_games["Margin"] <= 5]
w_close_games = w_games[w_games["Margin"] <= 5]

# Plot close games for men and women
plt.figure(figsize=(8, 4))
sns.barplot(x=['Men', 'Women'], y=[len(m_close_games), len(w_close_games)], palette='coolwarm')
plt.ylabel('Number of Close Games')
plt.title('Close Games (Margin â‰¤ 5 Points)')
plt.show()



