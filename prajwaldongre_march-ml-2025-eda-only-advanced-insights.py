import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.simplefilter('ignore')


def read_csv_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df_name = "df_" + os.path.splitext(filename)[0]  

            try:
                df = pd.read_csv(file_path)
                globals()[df_name] = df
                print(f"Read and created DataFrame: {df_name} from {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

folder_path = '/kaggle/input/march-machine-learning-mania-2025' 
read_csv_files_in_folder(folder_path)


def load_all_csv_to_dict(folder_path):
    dataframes = {}
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return dataframes

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df_name = os.path.splitext(filename)[0]

            try:
                df = pd.read_csv(file_path)
                dataframes[df_name] = df
                print(f"Loaded {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return dataframes

def check_missing_values(dataframes):
    for df_name, df in dataframes.items():
        print(f"\nDataFrame: {df_name}")
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("Missing Values:")
            print(missing_values[missing_values > 0])
        else:
            print("No missing values found.")

folder_path = '/kaggle/input/march-machine-learning-mania-2025' 
dataframes = load_all_csv_to_dict(folder_path)

if dataframes:
    check_missing_values(dataframes)


dfs = {
    "Cities": df_Cities,
    "Conferences": df_Conferences,
    "MConferenceTourneyGames": df_MConferenceTourneyGames,
    "MGameCities": df_MGameCities,
    "MMasseyOrdinals": df_MMasseyOrdinals,
    "MNCAATourneyCompactResults": df_MNCAATourneyCompactResults,
    "MNCAATourneyDetailedResults": df_MNCAATourneyDetailedResults,
    "MNCAATourneySeedRoundSlots": df_MNCAATourneySeedRoundSlots,
    "MNCAATourneySeeds": df_MNCAATourneySeeds,
    "MNCAATourneySlots": df_MNCAATourneySlots,
    "MRegularSeasonCompactResults": df_MRegularSeasonCompactResults,
    "MRegularSeasonDetailedResults": df_MRegularSeasonDetailedResults,
    "MSeasons": df_MSeasons,
    "MSecondaryTourneyCompactResults": df_MSecondaryTourneyCompactResults,
    "MSecondaryTourneyTeams": df_MSecondaryTourneyTeams,
    "MTeamCoaches": df_MTeamCoaches,
    "MTeamConferences": df_MTeamConferences,
    "MTeamSpellings": df_MTeamSpellings,
    "MTeams": df_MTeams,
    "SampleSubmissionStage1": df_SampleSubmissionStage1,
    "SampleSubmissionStage2": df_SampleSubmissionStage2,
    "SeedBenchmarkStage1": df_SeedBenchmarkStage1,
    "WConferenceTourneyGames": df_WConferenceTourneyGames,
    "WGameCities": df_WGameCities,
    "WNCAATourneyCompactResults": df_WNCAATourneyCompactResults,
    "WNCAATourneyDetailedResults": df_WNCAATourneyDetailedResults,
    "WNCAATourneySeeds": df_WNCAATourneySeeds,
    "WNCAATourneySlots": df_WNCAATourneySlots,
    "WRegularSeasonCompactResults": df_WRegularSeasonCompactResults,
    "WRegularSeasonDetailedResults": df_WRegularSeasonDetailedResults,
    "WSeasons": df_WSeasons,
    "WSecondaryTourneyCompactResults": df_WSecondaryTourneyCompactResults,
    "WSecondaryTourneyTeams": df_WSecondaryTourneyTeams,
    "WTeamConferences": df_WTeamConferences,
    "WTeamSpellings": df_WTeamSpellings,
    "WTeams": df_WTeams
}


for name, df in dfs.items():
    print(f"\n{name}: {df.shape} (Rows, Columns)")


df_men = df_MRegularSeasonCompactResults.copy()
df_women = df_WRegularSeasonCompactResults.copy()


# Calculating win margins
df_men["WinMargin"] = df_men["WScore"] - df_men["LScore"]
df_women["WinMargin"] = df_women["WScore"] - df_women["LScore"]

print("Men's Regular Season Stats:")
display(df_men.describe())

print("\nWomen's Regular Season Stats:")
display(df_women.describe())


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

axes[0].hist(df_men["WinMargin"], bins=30, color="blue", alpha=0.7)
axes[0].set_title("Men's Win Margin Distribution")
axes[0].set_xlabel("Win Margin")
axes[0].set_ylabel("Frequency")

axes[1].hist(df_women["WinMargin"], bins=30, color="red", alpha=0.7)
axes[1].set_title("Women's Win Margin Distribution")
axes[1].set_xlabel("Win Margin")

plt.tight_layout()
plt.show()


df_MRegularSeasonCompactResults["WinMargin"] = df_MRegularSeasonCompactResults["WScore"] - df_MRegularSeasonCompactResults["LScore"]
df_WRegularSeasonCompactResults["WinMargin"] = df_WRegularSeasonCompactResults["WScore"] - df_WRegularSeasonCompactResults["LScore"]

men_seasonal_win_margin = df_MRegularSeasonCompactResults.groupby("Season")["WinMargin"].mean()
women_seasonal_win_margin = df_WRegularSeasonCompactResults.groupby("Season")["WinMargin"].mean()

plt.figure(figsize=(12, 6))
plt.plot(men_seasonal_win_margin.index, men_seasonal_win_margin, label="Men's Win Margin", marker='o')
plt.plot(women_seasonal_win_margin.index, women_seasonal_win_margin, label="Women's Win Margin", marker='s')
plt.title("Average Win Margin Over Seasons", fontsize=14)
plt.xlabel("Season", fontsize=12)
plt.ylabel("Average Win Margin", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

fig = px.line(x=men_seasonal_win_margin.index, y=men_seasonal_win_margin, labels={'x': 'Season', 'y': 'Win Margin'},
              title="Average Win Margin Over Seasons (Men vs Women)", line_shape='linear')
fig.add_scatter(x=women_seasonal_win_margin.index, y=women_seasonal_win_margin, mode='lines', name="Women's Win Margin")
fig.show()


fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

sns.lineplot(x="Season", y="WScore", data=df_men, ax=ax[0], label="Men", color="blue")
ax[0].set_title("Men's Average Winning Score per Season")
ax[0].set_ylabel("Winning Score")
ax[0].set_xlabel("Season")

sns.lineplot(x="Season", y="WScore", data=df_women, ax=ax[1], label="Women", color="red")
ax[1].set_title("Women's Average Winning Score per Season")
ax[1].set_ylabel("")
ax[1].set_xlabel("Season")

plt.tight_layout()
plt.show()


home_wins_men = df_MRegularSeasonCompactResults[df_MRegularSeasonCompactResults["WLoc"] == "H"].shape[0]
away_wins_men = df_MRegularSeasonCompactResults[df_MRegularSeasonCompactResults["WLoc"] == "A"].shape[0]
neutral_games_men = df_MRegularSeasonCompactResults[df_MRegularSeasonCompactResults["WLoc"] == "N"].shape[0]

home_win_rate_men = home_wins_men / (home_wins_men + away_wins_men) * 100

print(f"ğŸ�€ Men's Home Win Rate: {home_win_rate_men:.2f}%")


home_wins_women = df_WRegularSeasonCompactResults[df_WRegularSeasonCompactResults["WLoc"] == "H"].shape[0]
away_wins_women = df_WRegularSeasonCompactResults[df_WRegularSeasonCompactResults["WLoc"] == "A"].shape[0]
neutral_games_women = df_WRegularSeasonCompactResults[df_WRegularSeasonCompactResults["WLoc"] == "N"].shape[0]

home_win_rate_women = home_wins_women / (home_wins_women + away_wins_women) * 100

print(f"ğŸ�€ Women's Home Win Rate: {home_win_rate_women:.2f}%")


# Win Margin for Men & Women
df_MRegularSeasonCompactResults["WinMargin"] = df_MRegularSeasonCompactResults["WScore"] - df_MRegularSeasonCompactResults["LScore"]
df_WRegularSeasonCompactResults["WinMargin"] = df_WRegularSeasonCompactResults["WScore"] - df_WRegularSeasonCompactResults["LScore"]

df_men_home = df_MRegularSeasonCompactResults[df_MRegularSeasonCompactResults["WLoc"] == "H"]
df_men_away = df_MRegularSeasonCompactResults[df_MRegularSeasonCompactResults["WLoc"] == "A"]

df_women_home = df_WRegularSeasonCompactResults[df_WRegularSeasonCompactResults["WLoc"] == "H"]
df_women_away = df_WRegularSeasonCompactResults[df_WRegularSeasonCompactResults["WLoc"] == "A"]

df_home_away = pd.DataFrame({
    "Category": ["Men - Home", "Men - Away", "Women - Home", "Women - Away"],
    "Win Margin": [
        df_men_home["WinMargin"].mean(), df_men_away["WinMargin"].mean(),
        df_women_home["WinMargin"].mean(), df_women_away["WinMargin"].mean()
    ]
})

plt.figure(figsize=(8, 5))
sns.barplot(x="Category", y="Win Margin", data=df_home_away, palette=["blue", "blue", "red", "red"])
plt.title("Home vs. Away Win Margins")
plt.ylabel("Average Win Margin")
plt.xlabel("")
plt.xticks(rotation=15)
plt.show()


df_men_tourney = df_MNCAATourneyCompactResults.copy()
df_women_tourney = df_WNCAATourneyCompactResults.copy()

# Win Margin for Mens & Womens Tournament Games 
df_men_tourney["WinMargin"] = df_men_tourney["WScore"] - df_men_tourney["LScore"]
df_women_tourney["WinMargin"] = df_women_tourney["WScore"] - df_women_tourney["LScore"]

# Grouping by Season to get the average Win Margin
men_win_margin_season = df_men_tourney.groupby("Season")["WinMargin"].mean()
women_win_margin_season = df_women_tourney.groupby("Season")["WinMargin"].mean()

plt.figure(figsize=(12, 6))
sns.lineplot(x=men_win_margin_season.index, y=men_win_margin_season.values, label="Men's Tournament", marker="o")
sns.lineplot(x=women_win_margin_season.index, y=women_win_margin_season.values, label="Women's Tournament", marker="o", linestyle="dashed")

plt.xlabel("Season", fontsize=12)
plt.ylabel("Average Win Margin", fontsize=12)
plt.title("Trend of Win Margins in NCAA Tournaments", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()


# Extracting numeric seed values for Men
df_MNCAATourneySeeds["SeedValue"] = df_MNCAATourneySeeds["Seed"].str.extract('(\d+)').astype(int)

# Extracting numeric seed values for Women
df_WNCAATourneySeeds["SeedValue"] = df_WNCAATourneySeeds["Seed"].str.extract('(\d+)').astype(int)


# Here merging Mens Tournament Results with Seeds
df_men_tourney = df_MNCAATourneyCompactResults.merge(
    df_MNCAATourneySeeds[["Season", "TeamID", "SeedValue"]],
    left_on=["Season", "WTeamID"],
    right_on=["Season", "TeamID"],
    how="left"
).rename(columns={"SeedValue": "WSeed"}).drop(columns=["TeamID"])

df_men_tourney = df_men_tourney.merge(
    df_MNCAATourneySeeds[["Season", "TeamID", "SeedValue"]],
    left_on=["Season", "LTeamID"],
    right_on=["Season", "TeamID"],
    how="left"
).rename(columns={"SeedValue": "LSeed"}).drop(columns=["TeamID"])

df_men_tourney["WinMargin"] = df_men_tourney["WScore"] - df_men_tourney["LScore"]

# Mergeing Womens Tournament Results with Seeds
df_women_tourney = df_WNCAATourneyCompactResults.merge(
    df_WNCAATourneySeeds[["Season", "TeamID", "SeedValue"]],
    left_on=["Season", "WTeamID"],
    right_on=["Season", "TeamID"],
    how="left"
).rename(columns={"SeedValue": "WSeed"}).drop(columns=["TeamID"])

df_women_tourney = df_women_tourney.merge(
    df_WNCAATourneySeeds[["Season", "TeamID", "SeedValue"]],
    left_on=["Season", "LTeamID"],
    right_on=["Season", "TeamID"],
    how="left"
).rename(columns={"SeedValue": "LSeed"}).drop(columns=["TeamID"])

df_women_tourney["WinMargin"] = df_women_tourney["WScore"] - df_women_tourney["LScore"]

display(df_men_tourney.head(), df_women_tourney.head(3))


df_men_tourney["WSeed"] = df_men_tourney["WSeed"].astype(str).str.extract("(\d+)").astype(float)
df_men_tourney["LSeed"] = df_men_tourney["LSeed"].astype(str).str.extract("(\d+)").astype(float)
df_men_tourney["Seed_Diff"] = df_men_tourney["LSeed"] - df_men_tourney["WSeed"]

df_women_tourney["WSeed"] = df_women_tourney["WSeed"].astype(str).str.extract("(\d+)").astype(float)
df_women_tourney["LSeed"] = df_women_tourney["LSeed"].astype(str).str.extract("(\d+)").astype(float)
df_women_tourney["Seed_Diff"] = df_women_tourney["LSeed"] - df_women_tourney["WSeed"]


plt.figure(figsize=(14, 6))

# Men's Tournament
plt.subplot(1, 2, 1)
sns.scatterplot(x=df_men_tourney["Seed_Diff"], y=df_men_tourney["WinMargin"], alpha=0.5)
sns.regplot(x=df_men_tourney["Seed_Diff"], y=df_men_tourney["WinMargin"], scatter=False, color="red")
plt.axhline(0, color="black", linestyle="dashed", alpha=0.7)
plt.xlabel("Seed Difference (LSeed - WSeed)")
plt.ylabel("Win Margin")
plt.title("Men's Tournament: Seed Difference vs. Win Margin")
plt.grid(True)

# Women's Tournament
plt.subplot(1, 2, 2)
sns.scatterplot(x=df_women_tourney["Seed_Diff"], y=df_women_tourney["WinMargin"], alpha=0.5)
sns.regplot(x=df_women_tourney["Seed_Diff"], y=df_women_tourney["WinMargin"], scatter=False, color="red")
plt.axhline(0, color="black", linestyle="dashed", alpha=0.7)
plt.xlabel("Seed Difference (LSeed - WSeed)")
plt.ylabel("Win Margin")
plt.title("Women's Tournament: Seed Difference vs. Win Margin")
plt.grid(True)

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))

# Men's Tournament
sns.scatterplot(x=df_men_tourney["Seed_Diff"], y=df_men_tourney["WinMargin"], alpha=0.6, label="Men's Tournament", color="blue")

# Women's Tournament
sns.scatterplot(x=df_women_tourney["Seed_Diff"], y=df_women_tourney["WinMargin"], alpha=0.6, label="Women's Tournament", color="red")

plt.axvline(x=0, color="gray", linestyle="--", linewidth=1)  # Reference line at Seed Difference = 0
plt.xlabel("Seed Difference (Lower Seed - Higher Seed)", fontsize=12)
plt.ylabel("Win Margin", fontsize=12)
plt.title("Seed Difference vs. Win Margin in NCAA Tournaments", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()


corr_men = df_men_tourney["Seed_Diff"].corr(df_men_tourney["WinMargin"])
corr_women = df_women_tourney["Seed_Diff"].corr(df_women_tourney["WinMargin"])

print(f"ğŸ“Œ Pearson Correlation (Men's Tournament): {corr_men:.4f}")
print(f"ğŸ“Œ Pearson Correlation (Women's Tournament): {corr_women:.4f}")


men_team_wins = df_MNCAATourneyCompactResults["WTeamID"].value_counts().reset_index()
men_team_wins.columns = ["TeamID", "Total Wins"]

women_team_wins = df_WNCAATourneyCompactResults["WTeamID"].value_counts().reset_index()
women_team_wins.columns = ["TeamID", "Total Wins"]

men_team_wins = men_team_wins.merge(df_MTeams, on="TeamID", how="left")
women_team_wins = women_team_wins.merge(df_WTeams, on="TeamID", how="left")

# Sorting by most wins
top_men_teams = men_team_wins.sort_values(by="Total Wins", ascending=False).head(10)
top_women_teams = women_team_wins.sort_values(by="Total Wins", ascending=False).head(10)

display(top_men_teams)
display(top_women_teams)


plt.figure(figsize=(14, 6))

# Men's Top Teams
plt.subplot(1, 2, 1)
sns.barplot(y=top_men_teams["TeamName"], x=top_men_teams["Total Wins"], palette="Blues_r")
plt.xlabel("Total NCAA Tournament Wins")
plt.ylabel("Team")
plt.title("Top 10 Most Successful Men's Teams")

# Women's Top Teams
plt.subplot(1, 2, 2)
sns.barplot(y=top_women_teams["TeamName"], x=top_women_teams["Total Wins"], palette="Purples_r")
plt.xlabel("Total NCAA Tournament Wins")
plt.ylabel("Team")
plt.title("Top 10 Most Successful Women's Teams")

plt.tight_layout()
plt.show()


# Categorizing seasons into decades
def get_decade(year):
    return f"{(year // 10) * 10}s"  # Lets say for example: 1985  into 1980s

df_MNCAATourneyCompactResults["Decade"] = df_MNCAATourneyCompactResults["Season"].apply(get_decade)
df_WNCAATourneyCompactResults["Decade"] = df_WNCAATourneyCompactResults["Season"].apply(get_decade)

men_decade_wins = df_MNCAATourneyCompactResults.groupby(["Decade", "WTeamID"]).size().reset_index(name="Total Wins")
women_decade_wins = df_WNCAATourneyCompactResults.groupby(["Decade", "WTeamID"]).size().reset_index(name="Total Wins")

men_decade_wins = men_decade_wins.merge(df_MTeams, left_on="WTeamID", right_on="TeamID", how="left")
women_decade_wins = women_decade_wins.merge(df_WTeams, left_on="WTeamID", right_on="TeamID", how="left")


top_men_teams_decade = men_decade_wins.sort_values(by=["Decade", "Total Wins"], ascending=[True, False]).groupby("Decade").head(5)
top_women_teams_decade = women_decade_wins.sort_values(by=["Decade", "Total Wins"], ascending=[True, False]).groupby("Decade").head(5)

display(top_men_teams_decade)
display(top_women_teams_decade)


plt.figure(figsize=(16, 8))

# Men's Plot
plt.subplot(1, 2, 1)
sns.barplot(data=top_men_teams_decade, x="Decade", y="Total Wins", hue="TeamName", palette="Blues_r")
plt.xlabel("Decade")
plt.ylabel("Total Wins")
plt.title("Top 5 Most Dominant Men's Teams per Decade")
plt.xticks(rotation=45)
plt.legend(title="Team", bbox_to_anchor=(1, 1))

# Women's Plot
plt.subplot(1, 2, 2)
sns.barplot(data=top_women_teams_decade, x="Decade", y="Total Wins", hue="TeamName", palette="Purples_r")
plt.xlabel("Decade")
plt.ylabel("Total Wins")
plt.title("Top 5 Most Dominant Women's Teams per Decade")
plt.xticks(rotation=45)
plt.legend(title="Team", bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()


df_men_cinderella = df_MNCAATourneyCompactResults.merge(df_MNCAATourneySeeds, left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"], how="left")
df_women_cinderella = df_WNCAATourneyCompactResults.merge(df_WNCAATourneySeeds, left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"], how="left")

df_men_cinderella["Seed"] = df_men_cinderella["Seed"].str.extract("(\d+)").astype(float)
df_women_cinderella["Seed"] = df_women_cinderella["Seed"].str.extract("(\d+)").astype(float)

men_wins = df_men_cinderella.groupby(["Season", "WTeamID", "Seed"]).size().reset_index(name="Wins")
women_wins = df_women_cinderella.groupby(["Season", "WTeamID", "Seed"]).size().reset_index(name="Wins")

# Cinderella Teams means basically a (low-seeded teams with multiple wins)
cinderella_men = men_wins[(men_wins["Seed"] >= 10) & (men_wins["Wins"] >= 3)]
cinderella_women = women_wins[(women_wins["Seed"] >= 10) & (women_wins["Wins"] >= 3)]

cinderella_men = cinderella_men.merge(df_MTeams, left_on="WTeamID", right_on="TeamID", how="left")
cinderella_women = cinderella_women.merge(df_WTeams, left_on="WTeamID", right_on="TeamID", how="left")

display(cinderella_men)
display(cinderella_women)


plt.figure(figsize=(16, 8))

# Men's Cinderella Plot
plt.subplot(1, 2, 1)
sns.barplot(data=cinderella_men, x="Season", y="Wins", hue="TeamName", palette="Greens_r", dodge=True, width=8)  # Increased bar width
plt.xlabel("Season")
plt.ylabel("Tournament Wins")
plt.title("Men's Cinderella Teams (Seed â‰¥10, Wins â‰¥3)")
plt.xticks(rotation=45)
plt.legend(title="Team", bbox_to_anchor=(1.02, 1), loc='upper left')  # Adjusted legend position for clarity

# Women's Cinderella Plot
plt.subplot(1, 2, 2)
sns.barplot(data=cinderella_women, x="Season", y="Wins", hue="TeamName", palette="Oranges_r")
plt.xlabel("Season")
plt.ylabel("Tournament Wins")
plt.title("Women's Cinderella Teams (Seed â‰¥10, Wins â‰¥3)")
plt.xticks(rotation=45)
plt.legend(title="Team", bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()


CLOSE_GAME_MARGIN = 5

# close games for Mens Regular Season
df_men_clutch_regular = df_MRegularSeasonCompactResults[
    (df_MRegularSeasonCompactResults["WScore"] - df_MRegularSeasonCompactResults["LScore"]) <= CLOSE_GAME_MARGIN
]

# close games for Mens Tournament
df_men_clutch_tourney = df_MNCAATourneyCompactResults[
    (df_MNCAATourneyCompactResults["WScore"] - df_MNCAATourneyCompactResults["LScore"]) <= CLOSE_GAME_MARGIN
]

# Same for Womens Regular Season
df_women_clutch_regular = df_WRegularSeasonCompactResults[
    (df_WRegularSeasonCompactResults["WScore"] - df_WRegularSeasonCompactResults["LScore"]) <= CLOSE_GAME_MARGIN
]

# Same for Womens Tournament
df_women_clutch_tourney = df_WNCAATourneyCompactResults[
    (df_WNCAATourneyCompactResults["WScore"] - df_WNCAATourneyCompactResults["LScore"]) <= CLOSE_GAME_MARGIN
]

print("Men's Regular Season Close Games:", df_men_clutch_regular.shape[0])
print("Men's Tournament Close Games:", df_men_clutch_tourney.shape[0])
print("Women's Regular Season Close Games:", df_women_clutch_regular.shape[0])
print("Women's Tournament Close Games:", df_women_clutch_tourney.shape[0])


df_MRegularSeasonCompactResults["WinMargin"] = df_MRegularSeasonCompactResults["WScore"] - df_MRegularSeasonCompactResults["LScore"]
df_MNCAATourneyCompactResults["WinMargin"] = df_MNCAATourneyCompactResults["WScore"] - df_MNCAATourneyCompactResults["LScore"]

df_WRegularSeasonCompactResults["WinMargin"] = df_WRegularSeasonCompactResults["WScore"] - df_WRegularSeasonCompactResults["LScore"]
df_WNCAATourneyCompactResults["WinMargin"] = df_WNCAATourneyCompactResults["WScore"] - df_WNCAATourneyCompactResults["LScore"]


def compute_close_game_win_pct(df, season_type):
    close_wins = df[df["WinMargin"] <= 5].groupby("WTeamID").size().reset_index(name="CloseWins")
    
    # Total games played per team (as winner)
    total_games = df.groupby("WTeamID").size().reset_index(name="TotalWins")
    
    close_win_pct = total_games.merge(close_wins, on="WTeamID", how="left").fillna(0)
    close_win_pct["CloseWinPct"] = close_win_pct["CloseWins"] / close_win_pct["TotalWins"]
    
    # Adding season type for reference
    close_win_pct["SeasonType"] = season_type
    return close_win_pct

men_close_win_pct_reg = compute_close_game_win_pct(df_MRegularSeasonCompactResults, "Regular Season")
men_close_win_pct_tourney = compute_close_game_win_pct(df_MNCAATourneyCompactResults, "Tournament")

women_close_win_pct_reg = compute_close_game_win_pct(df_WRegularSeasonCompactResults, "Regular Season")
women_close_win_pct_tourney = compute_close_game_win_pct(df_WNCAATourneyCompactResults, "Tournament")

display(men_close_win_pct_reg.head(), men_close_win_pct_tourney.head())
display(women_close_win_pct_reg.head(), women_close_win_pct_tourney.head())


# Merging Mens Data
men_close_win_compare = men_close_win_pct_reg.merge(
    men_close_win_pct_tourney, on="WTeamID", suffixes=("_Reg", "_Tourney")
)

# Merging Womens Data
women_close_win_compare = women_close_win_pct_reg.merge(
    women_close_win_pct_tourney, on="WTeamID", suffixes=("_Reg", "_Tourney")
)

display(men_close_win_compare.head(), women_close_win_compare.head())


plt.figure(figsize=(14, 6))

# Mens Tournament Close Win Percentage vs. Regular Season
plt.subplot(1, 2, 1)
sns.scatterplot(x=men_close_win_compare["CloseWinPct_Reg"], 
                y=men_close_win_compare["CloseWinPct_Tourney"], alpha=0.7)
plt.plot([0, 1], [0, 1], 'r--', label="1:1 Line")  
plt.xlabel("Regular Season Close Win %")
plt.ylabel("Tournament Close Win %")
plt.title("Men's Close Game Win %: Regular Season vs. Tournament")
plt.legend()

# Womens Tournament Close Win Percentage vs. Regular Season
plt.subplot(1, 2, 2)
sns.scatterplot(x=women_close_win_compare["CloseWinPct_Reg"], 
                y=women_close_win_compare["CloseWinPct_Tourney"], alpha=0.7)
plt.plot([0, 1], [0, 1], 'r--', label="1:1 Line")  
plt.xlabel("Regular Season Close Win %")
plt.ylabel("Tournament Close Win %")
plt.title("Women's Close Game Win %: Regular Season vs. Tournament")
plt.legend()

# Show plots
plt.tight_layout()
plt.show()




