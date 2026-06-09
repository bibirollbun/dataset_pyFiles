import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 01. Cities. csv

# There is a lot of files in as data for this competition
# Let's try to get to know them in a more organic way by looking over them one by one
cities = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/Cities.csv")
cities


# It's just a list of id of cities and their corresponding States


# 02. Conferences.csv
conferences = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/Conferences.csv")
conferences


# List of all conferences, with their abbreviation name and full name
# There are 51 conferences included, some seems location specific (Western, Southwest...)
# and some aren't

# btw: If you don't know about conference
# Conference: A set of teams that play against each other. 
# - The winner for conferences are selected for March Madness
# March Madness (college basketball): 68 teams plays against each other 
# - 32 is automatic winners of Conferences (so not all 51 conferences above count)
# - 36 others is hand selected
# - Single elimination: If you lose, you're out


# 03. MConferenceTourneyGames
mConfTourneyGames = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MConferenceTourneyGames.csv")
# len(mConfTourneyGames['ConfAbbrev'].unique())
# mConfTourneyGames.groupby('ConfAbbrev')['DayNum'].max()
mConfTourneyGames


# Question: What is the team that consisitently plays in a tournament 
# last 3 days for all year?
# (I assume last 3 days result are more important to determine team's strength)

# Load the data
df = mConfTourneyGames.copy()

# Get the last 3 days of each conference tournament for every season
df_last_days = df.groupby(["Season", "ConfAbbrev"])["DayNum"].nlargest(3).reset_index()

# Filter only games that happened on these last 3 days
df_filtered = df.merge(df_last_days, on=["Season", "ConfAbbrev", "DayNum"], how="inner")

# Get all teams (winning and losing) that played in those games
df_filtered["TeamID"] = df_filtered[["WTeamID", "LTeamID"]].values.tolist()
df_filtered = df_filtered.explode("TeamID")  # Split winners and losers into separate rows

# Count appearances of each team per season
team_season_counts = df_filtered.groupby(["TeamID", "Season"]).size().unstack(fill_value=0)

# Keep only teams that appeared in all seasons
consistent_teams = team_season_counts[(team_season_counts > 0).all(axis=1)].index

# Filter dataset to only include those teams
df_final = df_filtered[df_filtered["TeamID"].isin(consistent_teams)]
consistent_teams


# We found that only team 1211 is consistent.
# Let's see how good this team do by plotting a heat map with it
team_season_counts.reset_index()[team_season_counts.reset_index()['TeamID'] == 1211].set_index(['TeamID'])

# We can see that team 1211 appear to play in the finals of all year. 
# What a team


# Question: With the same motif, let's try to investing more broadly.
# What are the teams that appear in the last 3 days (games) of the tournament
# in the last 5 years?

# Load the data
df = mConfTourneyGames.copy()

# Get the last 3 days of each conference tournament for every season
df_last_days = df.groupby(["Season", "ConfAbbrev"])["DayNum"].nlargest(3).reset_index()

# Filter only games that happened on these last 3 days
df_filtered = df.merge(df_last_days, on=["Season", "ConfAbbrev", "DayNum"], how="inner")

# Get all teams (winning and losing) that played in those games
df_filtered["TeamID"] = df_filtered[["WTeamID", "LTeamID"]].values.tolist()
df_filtered = df_filtered.explode("TeamID")  # Split winners and losers into separate rows

# Identify relevant seasons for analysis
max_season = df["Season"].max()
recent_seasons = list(range(max_season - 2, max_season + 1))  # Last 3 seasons

# Filter to only include recent seasons
df_recent = df_filtered[df_filtered["Season"].isin(recent_seasons)]

# Count appearances of each team per season 
team_season_counts = df_recent.groupby(["TeamID", "Season"]).size().unstack(fill_value=0)

# Keep only teams that appeared in all recent seasons
consistent_teams = team_season_counts[(team_season_counts > 0).all(axis=1)].index.tolist()

# Filter original dataset to only include games with these teams
df_final = df.loc[
    (df["Season"].isin(recent_seasons)) & 
    ((df["WTeamID"].isin(consistent_teams)) | (df["LTeamID"].isin(consistent_teams)))
]

# Show results
print(f"Number of consistent tournament teams: {len(consistent_teams)}")
print(team_season_counts[team_season_counts.index.isin(consistent_teams)])


# There are 37 teams that make this list
# Notice that theter is no 1 values in the list. 
# This is probably due to the game finding 4 last teams and plays to rank
# So In the last 3 days those 4 teams will play 2 days at least 


def compute_tournament_strength(df, team_id, n_year_window=5, n_dates_window=3, w_win=0.7, w_game=0.3):
    """Computes Tournament Consistency Score for a team based on recent tournament performance."""
    # Basic validation
    if n_year_window <= 0 or n_dates_window <= 0 or not np.isclose(w_win + w_game, 1.0):
        raise ValueError("Invalid parameters")
        
    # Filter to recent seasons and get last days of each tournament
    max_season = df["Season"].max()
    recent_seasons = range(max_season - n_year_window + 1, max_season + 1)
    df = df[df["Season"].isin(recent_seasons)].copy()
    
    # Get tournament last days and filter to those games
    last_days = df.groupby(["Season", "ConfAbbrev"])["DayNum"].nlargest(n_dates_window)
    df = df.merge(last_days.reset_index()[["Season", "ConfAbbrev", "DayNum"]], 
                  on=["Season", "ConfAbbrev", "DayNum"])
    
    # Get team games and calculate stats
    team_games = df[(df["WTeamID"] == team_id) | (df["LTeamID"] == team_id)]
    team_games["Wins"] = (team_games["WTeamID"] == team_id).astype(int)
    
    # Aggregate by season and calculate weighted score
    stats = team_games.groupby("Season").agg(Wins=("Wins", "sum"), Games=("WTeamID", "count"))
    stats["weight"] = 1 + (stats.index - min(recent_seasons)) / n_year_window
    stats["score"] = stats["weight"] * (w_win * stats["Wins"] + w_game * stats["Games"])
    
    # Handle missing seasons and calculate final score
    all_seasons = pd.DataFrame(index=recent_seasons)
    stats = all_seasons.join(stats).fillna(0)
    
    return min(stats["score"].sum() / n_year_window, 1)


# Loop-hole: We didn't consider the tournament quality so an easy tournament 
# can a a problem. Other than that, we're also not talking in account the strength 
# of the opponent. 


# 04. MGameCities.cvs
mGamesCities = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MGameCities.csv")
mGamesCities




