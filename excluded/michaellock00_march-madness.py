import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings
from IPython.display import display
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import optuna

# Configure Pandas display options
pd.set_option("display.max_columns", None)  # Show all columns when displaying DataFrames
pd.options.mode.chained_assignment = None  # Suppress unnecessary warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)  # Ignore performance warnings

# ---------------------------------------
# Key Definitions: LV vs. HV
# ---------------------------------------
# - LV (Lower Value) refers to the lower-ranked team in a matchup.
# - HV (Higher Value) refers to the higher-ranked team in a matchup.
# 
# This naming helps when tracking game outcomes. For example, `LVWin = 1` means the underdog (lower-ranked team) won, while `LVWin = 0` means the favorite (higher-ranked team) won.
# ---------------------------------------

# Columns tracking game outcomes
lv_win_column = "LVWin"  # Did the lower-ranked team win?
hv_win_column = "HVWin"  # Did the higher-ranked team win?

# Columns tracking scores
lv_score_column = "LVScore"  # Points scored by lower-ranked team
hv_score_column = "HVScore"  # Points scored by higher-ranked team

# Team identifiers
lv_team_column = "LVTeamID"  # Lower-ranked team ID
hv_team_column = "HVTeamID"  # Higher-ranked team ID

# ---------------------------------------
# Tournament Rounds Mapping
# ---------------------------------------
# The dataset uses numerical codes to represent different tournament rounds.
# This dictionary maps those codes to human-readable names.
# ---------------------------------------
tourney_dict = {
    134: "Playin",
    135: "Playin",
    136: "Round1",
    137: "Round1",
    138: "Round2",
    139: "Round2",
    143: "Last16",
    144: "Last16",
    145: "Last8",
    146: "Last8",
    152: "SemiFinals",
    154: "Finals",
}

# Root folder for dataset (specific to Kaggle environment)
root_folder = '/kaggle/input/march-machine-learning-mania-2025'

# List of both team IDs for easier DataFrame operations
both_team_column = ["LVTeamID", "HVTeamID"]

# ---------------------------------------
# Detailed Team Performance Stats
# ---------------------------------------
# These are the key game statistics used for feature engineering.
# ---------------------------------------
detailed_cols = [
    "FGM",   # Field Goals Made
    "FGA",   # Field Goals Attempted
    "FGM3",  # 3-Point Field Goals Made
    "FGA3",  # 3-Point Field Goals Attempted
    "FTM",   # Free Throws Made
    "FTA",   # Free Throws Attempted
    "OR",    # Offensive Rebounds
    "DR",    # Defensive Rebounds
    "Ast",   # Assists
    "TO",    # Turnovers
    "Stl",   # Steals
    "Blk",   # Blocks
    "PF",    # Personal Fouls
    "Score", # Total Points Scored
]

# Create win/loss versions of each stat (i.e., tracking stats for both winners and losers)
win_cols = ["W" + col for col in detailed_cols]  # Stats for winning teams
lose_cols = ["L" + col for col in detailed_cols]  # Stats for losing teams

# Placeholder lists for feature engineering (these get filled later)
hv_detailed_cols = []  # Stats for higher-ranked teams
lv_detailed_cols = []  # Stats for lower-ranked teams

all_hv_sum_cols = []  # Aggregated sum features for HV teams
all_hv_mean_cols = []  # Aggregated mean features for HV teams

all_both_sum_cols = []  # Sum features for both teams
all_both_mean_cols = []  # Mean features for both teams

all_lv_sum_cols = []  # Aggregated sum features for LV teams
all_lv_mean_cols = []  # Aggregated mean features for LV teams

all_prev_season_columns = []  # Features tracking previous season performance
all_cumsum_columns = []  # Cumulative sum features (for tracking momentum)

all_momentum_columns = []  # Features tracking recent performance trends
shift_values = [2, 10]  # Used for momentum tracking (e.g., last 2 games, last 10 games)

# ---------------------------------------
# Columns Used for Feature Engineering
# ---------------------------------------
season_columns_both_teams = ["Season", lv_team_column, hv_team_column]  # Season + both teams
season_columns_lv_team = ["Season", lv_team_column]  # Season + LV team only
season_columns_hv_team = ["Season", hv_team_column]  # Season + HV team only

# Features tracking per-season stats for teams
per_season_columns = [
    "TimesPlayedPerSeason",          # How many times the teams have played that season
    "LVWinCountPerSeason",           # Wins by LV team in the season
    "HVWinCountPerSeason",           # Wins by HV team in the season
    "LVWinCountPerSeasonTotal",      # Cumulative wins by LV team in previous seasons
    "HVWinCountPerSeasonTotal",      # Cumulative wins by HV team in previous seasons
    "LVScorePerSeason",              # Average LV team score for the season
    "HVScorePerSeason",              # Average HV team score for the season
]

# ---------------------------------------
# Placeholder for Raw Data Storage
# ---------------------------------------
all_games = []  # Full dataset of all games played
all_detailed_results = []  # More detailed game results
all_teams = []  # List of all teams in the dataset

# Gender mapping (for potential filtering by men's/women's tournaments)
gender_dict = {"M": "men", "W": "women"}

# Tracking coaches for both teams
coach_columns = ["HVCoach", "LVCoach"]

# Target variable: Did the lower-ranked team win?
target_col = "LVWin"


# ---------------------------------------
# Get Teams Info: Merge team metadata
# ---------------------------------------
# - Loads team names and alternative spellings
# - Identifies each team's home city and state
# - Merges all relevant information for a complete team dataset
# ---------------------------------------
def get_teams_info(team_games_df, gender):
    teams = pd.read_csv(f"{root_folder}/{gender}Teams.csv")
    teams_spellings = pd.read_csv(f"{root_folder}/{gender}TeamSpellings.csv")
    
    # Some teams have multiple spellings, so we consolidate them
    teams_spellings = teams_spellings.groupby(by="TeamID")["TeamNameSpelling"].unique().reset_index()
    teams_info = teams.merge(teams_spellings, on="TeamID")

    # Identify each team's home city/state based on where they play home games
    teams_home_state_city = (
        team_games_df[team_games_df["WLoc"] == "H"][["Season", "WTeamID", "City", "State"]]
        .rename(columns={"WTeamID": "TeamID"})
        .drop_duplicates()
    )

    # Merge with team info
    teams_info = teams_home_state_city.merge(teams_info, on="TeamID")

    return teams_info




# ---------------------------------------
# Load Game Data: Regular season & tournament
# ---------------------------------------
# - Loads historical games from regular season and NCAA tournament
# - Merges with city locations for additional context
# - Adds a flag to distinguish tournament games from regular-season matchups
# ---------------------------------------
def get_historic_games(gender, city_df):
    compact_tourney_results = pd.read_csv(f"{root_folder}/{gender}NCAATourneyCompactResults.csv")
    tourney_results_d = pd.read_csv(f"{root_folder}/{gender}NCAATourneyDetailedResults.csv")

    compact_season_results = pd.read_csv(f"{root_folder}/{gender}RegularSeasonCompactResults.csv")
    reg_season_results_d = pd.read_csv(f"{root_folder}/{gender}RegularSeasonDetailedResults.csv")

    games_played = pd.read_csv(f"{root_folder}/{gender}GameCities.csv")

    # Filter out "Secondary" games to focus on primary matchups
    regular_season_and_ncaa = games_played[games_played["CRType"] != "Secondary"]

    # Add tournament flag to differentiate between regular season and NCAA tournament games
    compact_tourney_results["Tournament"] = True
    compact_season_results["Tournament"] = False

    # Merge tournament and regular season data
    compact_tour_reg = pd.concat([compact_tourney_results, compact_season_results], axis=0).sort_values(
        by=["Season", "DayNum"], ascending=True
    )

    detailed_tour_reg = pd.concat([tourney_results_d, reg_season_results_d], axis=0).sort_values(
        by=["Season", "DayNum"], ascending=True
    )

    # Merge city data to enrich the dataset
    regular_season_and_ncaa = regular_season_and_ncaa.merge(
        compact_tour_reg, on=["Season", "DayNum", "WTeamID", "LTeamID"], how="left"
    ).sort_values(by=["Season", "DayNum"], ascending=True)

    regular_season_and_ncaa = regular_season_and_ncaa.merge(city_df, on=["CityID"], how="left")

    # Assign gender label to the dataset
    regular_season_and_ncaa["Gender"] = gender

    return regular_season_and_ncaa, detailed_tour_reg


# ---------------------------------------
# Standardize Data: Assign LV/HV team roles
# ---------------------------------------
# - Reorganizes matchups so lower-ranked teams (LV) and higher-ranked teams (HV) are consistent
# - Adds win/loss labels for easier model training
# ---------------------------------------    
def add_detail_cols(
    regular_matches, all_detailed_results, all_teams, lv_team_column, hv_team_column, lv_win_column, hv_win_column
):

    # Ensure the lower-ranked team (LV) is always listed first
    regular_matches[lv_team_column] = regular_matches[["WTeamID", "LTeamID"]].min(axis=1)
    regular_matches[hv_team_column] = regular_matches[["WTeamID", "LTeamID"]].max(axis=1)

    # Assign win/loss indicators based on the LV/HV team
    regular_matches[lv_win_column] = np.where(regular_matches["WTeamID"] == regular_matches[lv_team_column], 1, 0)
    regular_matches[hv_win_column] = np.where(regular_matches["WTeamID"] == regular_matches[hv_team_column], 1, 0)

    # Merge detailed game statistics
    regular_matches = regular_matches.merge(
        all_detailed_results,
        on=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc", "NumOT"],
        how="left",
    )

    # Merge LV and HV team data into the main dataset
    regular_matches = regular_matches.merge(
        all_teams.drop_duplicates(subset=["TeamID", "Season"]).add_prefix("LV"),
        left_on=["LVTeamID", "Season"],
        right_on=["LVTeamID", "LVSeason"],
        how="left",
    ).drop(columns=["LVSeason"])
    regular_matches = regular_matches.merge(
        all_teams.drop_duplicates(subset=["TeamID", "Season"]).add_prefix("HV"),
        left_on=["HVTeamID", "Season"],
        right_on=["HVTeamID", "HVSeason"],
        how="left",
    ).drop(columns=["HVSeason"])

    return regular_matches


# ---------------------------------------
# Convert Win/Loss Stats into LV/HV Format
# ---------------------------------------
# - Transforms raw win/loss statistics into LV/HV specific columns
# - Ensures that the dataset tracks performance relative to each team's ranking
# ---------------------------------------

def get_detailed_hv_lv_cols(regular_matches, win_cols, lose_cols, hv_win_column, lv_win_column):

    for col_idx in range(len(win_cols)):
        win_col = win_cols[col_idx]
        lose_col = lose_cols[col_idx]

        hv_col = "HV" + win_col[1:] # High-ranked team stat
        lv_col = "LV" + win_col[1:] # Low-ranked team stat

        # Assign win/loss stats to LV/HV teams
        regular_matches[hv_col] = (regular_matches[hv_win_column] * regular_matches[win_col]) + (
            regular_matches[lv_win_column] * regular_matches[lose_col]
        )
        regular_matches[lv_col] = (regular_matches[lv_win_column] * regular_matches[win_col]) + (
            regular_matches[hv_win_column] * regular_matches[lose_col]
        )

        # Drop original win/loss columns since we now have LV/HV specific versions
        regular_matches = regular_matches.drop(columns=[win_col, lose_col])
        
        # Append new columns to feature lists
        lv_detailed_cols.append(lv_col)
        hv_detailed_cols.append(hv_col)

    return regular_matches, lv_detailed_cols, hv_detailed_cols



# ---------------------------------------
# Compute Rolling Averages for Momentum Tracking
# ---------------------------------------
# - Calculates recent performance trends over the last few games
# - Uses rolling averages for key statistics to track team momentum
# ---------------------------------------

def get_lv_rolling_scores(
    regular_matches, lv_team_column, hv_team_column, lv_deet_col, hv_deet_col, deet_col, shift_values
):
    all_lv_data = []

    for lv_id, lv_data_orinal in regular_matches[
        ["Season", "DayNum", lv_team_column, hv_team_column, lv_deet_col, hv_deet_col]
    ].groupby(by=[lv_team_column]):

        lv_data = lv_data_orinal.copy()

        # Compute the stat difference between LV and HV teams
        lv_data[f"Diff{deet_col}"] = lv_data[lv_deet_col] - lv_data[hv_deet_col]

        # Compute rolling averages for past performance trends
        for i in shift_values:
            lv_data[f"{lv_deet_col}_R{i}S"] = lv_data[lv_deet_col].rolling(window=i).mean().shift(1)
            lv_data[f"{hv_deet_col}AgainstLV_R{i}S"] = lv_data[hv_deet_col].rolling(window=i).mean().shift(1)
            lv_data[f"Diff{deet_col}_R{i}S"] = lv_data[f"Diff{deet_col}"].rolling(window=i).mean().shift(1)

        lv_data = lv_data.drop(columns=[lv_deet_col, hv_deet_col, f"Diff{deet_col}"])
        all_lv_data.append(lv_data)

    return pd.concat(all_lv_data)


def get_hv_rolling_scores(regular_matches, lv_team_column, hv_team_column, lv_deet_col, hv_deet_col, shift_values):
    all_hv_data = []

    for hv_id, hv_data_orinal in regular_matches[
        ["Season", "DayNum", lv_team_column, hv_team_column, lv_deet_col, hv_deet_col]
    ].groupby(by=[hv_team_column]):

        hv_data = hv_data_orinal.copy()

        # Compute rolling averages for past performance trends
        for i in shift_values:
            hv_data[f"{hv_deet_col}_R{i}S"] = hv_data[hv_deet_col].rolling(window=i).mean().shift(1)
            hv_data[f"{lv_deet_col}AgainstHV_R{i}S"] = hv_data[lv_deet_col].rolling(window=i).mean().shift(1)

        hv_data = hv_data.drop(columns=[lv_deet_col, hv_deet_col])
        all_hv_data.append(hv_data)

    return pd.concat(all_hv_data)


# ---------------------------------------
# Track Momentum-Based Features
# ---------------------------------------
# - Stores rolling averages and differentials for momentum tracking
# - Helps model understand recent team performance rather than just season averages
# ---------------------------------------

def get_momentum_cols(all_momentum_columns, hv_deet_col, lv_deet_col, deet_col, shift_values):

    all_momentum_columns += [f"{hv_deet_col}_R{i}S" for i in shift_values]  # HV rolling stats
    all_momentum_columns += [f"{lv_deet_col}_R{i}S" for i in shift_values]  # LV rolling stats
    all_momentum_columns += [f"{lv_deet_col}AgainstHV_R{i}S" for i in shift_values]  # LV stats against HV
    all_momentum_columns += [f"{hv_deet_col}AgainstLV_R{i}S" for i in shift_values]  # HV stats against LV
    all_momentum_columns += [f"Diff{deet_col}_R{i}S" for i in shift_values]  # Performance difference

    return all_momentum_columns


# ---------------------------------------
# Compute All Rolling Scores
# ---------------------------------------
# - Uses LV/HV rolling stats to generate meaningful momentum features
# - Merges these features into the main dataset
# ---------------------------------------
def get_all_rolling_scores(
    regular_matches, lv_team_column, hv_team_column, shift_values, detailed_cols, all_momentum_columns
):

    for deet_col in detailed_cols:

        lv_deet_col = "LV" + deet_col
        hv_deet_col = "HV" + deet_col

        all_lv_data = get_lv_rolling_scores(
            regular_matches, lv_team_column, hv_team_column, lv_deet_col, hv_deet_col, deet_col, shift_values
        )

        regular_matches = regular_matches.merge(
            all_lv_data,
            how="inner",
            on=["Season", "DayNum", lv_team_column, hv_team_column],
        )

        all_hv_data = get_hv_rolling_scores(
            regular_matches, lv_team_column, hv_team_column, lv_deet_col, hv_deet_col, shift_values
        )

        regular_matches = regular_matches.merge(
            all_hv_data,
            how="inner",
            on=["Season", "DayNum", lv_team_column, hv_team_column],
        )

        all_momentum_columns = get_momentum_cols(all_momentum_columns, hv_deet_col, lv_deet_col, deet_col, shift_values)

        # Compute difference in rolling scores (is one team improving faster?)
        for i in shift_values:
            regular_matches[f"{deet_col}_diff_R{i}S"] = (
                regular_matches[f"{hv_deet_col}_R{i}S"] - regular_matches[f"{lv_deet_col}_R{i}S"]
            )
            all_momentum_columns.append(f"{deet_col}_diff_R{i}S")

    return regular_matches, all_momentum_columns

# ---------------------------------------
# Compute Per-Season Averages and Totals
# ---------------------------------------
# - Calculates season-long stat sums and means for LV/HV teams
# - Helps capture long-term team performance rather than just recent form
# ---------------------------------------

def get_sum_and_mean_detailed_cols(
    regular_matches,
    lv_detailed_cols,
    all_lv_sum_cols,
    all_lv_mean_cols,
    season_columns_lv_team,
    hv_detailed_cols,
    all_hv_sum_cols,
    all_hv_mean_cols,
    season_columns_hv_team,
):

    # This get the low value teams sum and mean value for each metric of the game ✓ could add std, count, min, max, median, ect.
    for col in lv_detailed_cols:

        lv_sum_col = col + "PerSeasonSum"
        lv_mean_col = col + "PerSeasonMean"

        regular_matches = regular_matches.merge(
            regular_matches.groupby(by=season_columns_lv_team)[col].sum().reset_index(name=lv_sum_col),
            on=season_columns_lv_team,
        )
        regular_matches = regular_matches.merge(
            regular_matches.groupby(by=season_columns_lv_team)[col].mean().reset_index(name=lv_mean_col),
            on=season_columns_lv_team,
        )

        all_lv_sum_cols.append(lv_sum_col)
        all_lv_mean_cols.append(lv_mean_col)

    # This get the high value teams sum and mean value for each metric of the game ✓ could add std, count, min, max, median, ect.
    for col in hv_detailed_cols:

        hv_sum_col = col + "PerSeasonSum"
        hv_mean_col = col + "PerSeasonMean"

        regular_matches = regular_matches.merge(
            regular_matches.groupby(by=season_columns_hv_team)[col].sum().reset_index(name=hv_sum_col),
            on=season_columns_hv_team,
        )
        regular_matches = regular_matches.merge(
            regular_matches.groupby(by=season_columns_hv_team)[col].mean().reset_index(name=hv_mean_col),
            on=season_columns_hv_team,
        )

        all_hv_sum_cols.append(hv_sum_col)
        all_hv_mean_cols.append(hv_mean_col)

    return regular_matches, all_lv_sum_cols, all_lv_mean_cols, all_hv_sum_cols, all_hv_mean_cols


# ---------------------------------------
# Compute Head-to-Head Stats
# ---------------------------------------
# - Tracks performance of two teams when they've played each other before
# - Helps capture past matchup dynamics
# ---------------------------------------

def get_head_to_head_cols(
    regular_matches,
    hv_detailed_cols,
    lv_detailed_cols,
    season_columns_both_teams,
    all_both_sum_cols,
    all_both_mean_cols,
):
    # This get the high value and low value head to head info

    for col in hv_detailed_cols + lv_detailed_cols:

        both_sum_col = col + "BothPerSeasonSum"
        both_mean_col = col + "BothPerSeasonMean"

        regular_matches = regular_matches.merge(
            regular_matches.groupby(by=season_columns_both_teams)[col].sum().reset_index(name=both_sum_col),
            on=season_columns_both_teams,
        )
        regular_matches = regular_matches.merge(
            regular_matches.groupby(by=season_columns_both_teams)[col].mean().reset_index(name=both_mean_col),
            on=season_columns_both_teams,
        )

        all_both_sum_cols.append(both_sum_col)
        all_both_mean_cols.append(both_mean_col)

    return regular_matches, all_both_sum_cols, all_both_mean_cols

# ---------------------------------------
# Track Tournament Performance (LV Teams)
# ---------------------------------------
# - Counts how far LV teams advanced in past tournaments
# - Helps track Cinderella teams or consistent underdog success
# ---------------------------------------

def get_lv_tourney_cols(regular_matches):
    lv_team_results = []

    for (season, lv_team_id), group in regular_matches[regular_matches["LVWin"] == 1].groupby(
        by=["Season", "LVTeamID"]
    ):

        single_teams_data = group["TourneyRound"].value_counts()

        for round_num in list(set(tourney_dict.values())):

            if round_num in single_teams_data.index:
                lv_team_results.append([season, lv_team_id, round_num, single_teams_data[round_num]])
            else:
                lv_team_results.append([season, lv_team_id, round_num, 0])

    lv_team_results = pd.DataFrame(lv_team_results, columns=["Season", "LVTeamID", "Round", "Count"])
    lv_team_results = lv_team_results.pivot(index=["Season", "LVTeamID"], columns="Round", values="Count").reset_index()
    lv_team_results[["CumSum" + col for col in lv_team_results.columns[2:]]] = lv_team_results.groupby(by=["LVTeamID"])[
        lv_team_results.columns[2:]
    ].cumsum()

    drop_cols = lv_team_results.columns[2:]

    lv_team_results[["LVPrev" + col for col in lv_team_results.columns[2:]]] = lv_team_results.groupby(by=["LVTeamID"])[
        lv_team_results.columns[2:]
    ].shift(1)

    return lv_team_results.drop(columns=drop_cols).fillna(0)

# ---------------------------------------
# Track Tournament Performance (HV Teams)
# ---------------------------------------
# - Similar to LV teams but tracks HV teams instead
# - Useful for identifying powerhouse teams that dominate the tournament
# ---------------------------------------

def get_hv_tourney_cols(regular_matches):
    hv_team_results = []

    for (season, hv_team_id), group in regular_matches[regular_matches["HVWin"] == 1].groupby(
        by=["Season", "HVTeamID"]
    ):

        single_teams_data = group["TourneyRound"].value_counts()

        for round_num in list(set(tourney_dict.values())):

            if round_num in single_teams_data.index:
                hv_team_results.append([season, hv_team_id, round_num, single_teams_data[round_num]])
            else:
                hv_team_results.append([season, hv_team_id, round_num, 0])

    hv_team_results = pd.DataFrame(hv_team_results, columns=["Season", "HVTeamID", "Round", "Count"])
    hv_team_results = hv_team_results.pivot(index=["Season", "HVTeamID"], columns="Round", values="Count").reset_index()
    hv_team_results[["CumSum" + col for col in hv_team_results.columns[2:]]] = hv_team_results.groupby(by=["HVTeamID"])[
        hv_team_results.columns[2:]
    ].cumsum()

    drop_cols = hv_team_results.columns[2:]

    hv_team_results[["HVPrev" + col for col in hv_team_results.columns[2:]]] = hv_team_results.groupby(by=["HVTeamID"])[
        hv_team_results.columns[2:]
    ].shift(1)
    return hv_team_results.drop(columns=drop_cols).fillna(0)

# ---------------------------------------
# Process and Track Tournament Seeds
# ---------------------------------------
# - Extracts and standardizes seed numbers and regions
# - Adds previous season's seed info for trend tracking
# - Helps the model understand how a team's seed compares year-over-year
# ---------------------------------------

def get_seed_data(seeds, regular_matches):
    # Split the 'Seed' column
    all_seed_cols = []

    # Extract region letters (e.g., "W", "E", "S", "MW") and numeric seed values
    seeds["Region"] = seeds["Seed"].str.extract(r"([A-Za-z]+)")  # Extract letters
    seeds["Seed"] = seeds["Seed"].str.extract(r"(\d+)").astype(int)  # Extract numbers

    # ---------------------------------------
    # Merge Seed Info for LV (Lower-Ranked) Teams
    # ---------------------------------------
    regular_matches = (
        regular_matches.merge(seeds, left_on=["Season", "LVTeamID"], right_on=["Season", "TeamID"], how="left")
        .drop(columns=["TeamID"])
        .rename(columns={"Seed": "LVSeed", "Region": "LVSeedRegion"})
    )

    # If a team doesn't have a seed (not in tournament), default to 17 (lower than any real seed)
    regular_matches["LVSeed"] = regular_matches["LVSeed"].fillna(17)
    regular_matches["LVSeedRegion"] = regular_matches["LVSeedRegion"].fillna("U")

    # ---------------------------------------
    # Merge Seed Info for HV (Higher-Ranked) Teams
    # ---------------------------------------
    regular_matches = (
        regular_matches.merge(seeds, left_on=["Season", "HVTeamID"], right_on=["Season", "TeamID"], how="left")
        .drop(columns=["TeamID"])
        .rename(columns={"Seed": "HVSeed", "Region": "HVSeedRegion"})
    )

    # ---------------------------------------
    # Create Previous Season Seed Tracking
    # ---------------------------------------
    # This maps previous season's seeds to the current season for both LV and HV teams
    regular_matches["HVSeed"] = regular_matches["HVSeed"].fillna(17)
    regular_matches["HVSeedRegion"] = regular_matches["HVSeedRegion"].fillna("U")

    # Create a mapping of TeamID to HVSeed and LVSeed for each season
    prev_season_seeds = regular_matches[
        [
            "Season",
            "HVTeamID",
            "LVTeamID",
            "HVSeed",
            "LVSeed",
            "LVSeedRegion",
            "HVSeedRegion",
        ]
    ].copy()
    prev_season_seeds["Season"] += 1  # Shift the season forward

    for col in ["Seed", "SeedRegion"]:

        lv_col = "LV" + col
        hv_col = "HV" + col

        lv_prev_col = lv_col + "Prev"
        hv_prev_col = hv_col + "Prev"

        # Convert previous season's seed data into dictionaries for quick lookup
        lv_col_map = prev_season_seeds.set_index(["Season", "LVTeamID"])[lv_col].to_dict()
        hv_col_map = prev_season_seeds.set_index(["Season", "HVTeamID"])[hv_col].to_dict()

        # Add previous season's seeds to the current dataset
        regular_matches[hv_prev_col] = regular_matches.apply(
            lambda row: hv_col_map.get((row["Season"], row["HVTeamID"])), axis=1
        )
        regular_matches[lv_prev_col] = regular_matches.apply(
            lambda row: lv_col_map.get((row["Season"], row["LVTeamID"])), axis=1
        )

        all_seed_cols.append(lv_prev_col)
        all_seed_cols.append(hv_prev_col)

    # ---------------------------------------
    # Handle Missing Seed Data from Previous Season
    # ---------------------------------------
    # If a team wasn't in the tournament last year, assume they were unseeded (default to 17)
    regular_matches[["HVSeedPrev", "LVSeedPrev"]] = regular_matches[["HVSeedPrev", "LVSeedPrev"]].fillna(17)
    
    # Compute the difference in previous year's seed (was a team improving or dropping?)
    regular_matches["SeedDiff"] = regular_matches["HVSeedPrev"] - regular_matches["LVSeedPrev"]
    
    # Fill missing region data with "U" (Unknown)
    regular_matches[["HVSeedRegionPrev", "LVSeedRegionPrev"]] = regular_matches[
        ["HVSeedRegionPrev", "LVSeedRegionPrev"]
    ].fillna("U")

    # Create a combined feature for seed region matchups (e.g., "MW_E" for Midwest vs. East)
    regular_matches["SeedRegionGroup"] = regular_matches["HVSeedRegionPrev"] + regular_matches["LVSeedRegionPrev"]

    all_seed_cols.append("SeedDiff")
    all_seed_cols.append("SeedRegionGroup")

    return regular_matches, all_seed_cols


# ---------------------------------------
# Load City Data
# ---------------------------------------
# - Contains location info for games, which might help track home/away effects.
# ---------------------------------------
city_df = pd.read_csv(f"{root_folder}/Cities.csv")

# ---------------------------------------
# Load Game Data (Regular Season & NCAA Tournament)
# ---------------------------------------
# - Pulls in both men's and women's games.
# - Merges team info and detailed tournament results.
# - Stores all games, teams, and detailed results in separate lists.
# ---------------------------------------
for gender in ["W", "M"]:
    regular_season_and_ncaa, detailed_tour_reg = get_historic_games(gender, city_df)
    team_info = get_teams_info(regular_season_and_ncaa, gender)

    all_games.append(regular_season_and_ncaa)
    all_teams.append(team_info)
    all_detailed_results.append(detailed_tour_reg)

# Convert lists into DataFrames for easier processing
all_games = pd.concat(all_games).reset_index(drop=True).sort_values(by=["Season", "DayNum"], ascending=True)
all_teams = pd.concat(all_teams).reset_index(drop=True)
all_detailed_results = (
    pd.concat(all_detailed_results).reset_index(drop=True).sort_values(by=["Season", "DayNum"], ascending=True)
)

# ---------------------------------------
# Standardize Match Data: Assign LV & HV Teams
# ---------------------------------------
# - Ensures all matchups follow the same format (lower-ranked vs. higher-ranked).
# - Merges detailed stats into a structured format.
# ---------------------------------------
regular_matches = add_detail_cols(
    all_games.copy(), all_detailed_results, all_teams, lv_team_column, hv_team_column, lv_win_column, hv_win_column
)

# ---------------------------------------
# Convert Win/Loss Stats to LV/HV Columns
# ---------------------------------------
# - Moves raw win/loss stats into LV/HV format.
# - This ensures that every row compares a lower-ranked team to a higher-ranked team.
# ---------------------------------------
regular_matches, lv_detailed_cols, hv_detailed_cols = get_detailed_hv_lv_cols(
    regular_matches, win_cols, lose_cols, hv_win_column, lv_win_column
)

# ---------------------------------------
# Generate Rolling Averages (Momentum Tracking)
# ---------------------------------------
# - Computes rolling averages of key stats over the last few games.
# - Helps capture whether a team is getting better or worse as the season progresses.
# ---------------------------------------
regular_matches, all_momentum_columns = get_all_rolling_scores(
    regular_matches, lv_team_column, hv_team_column, shift_values, detailed_cols, all_momentum_columns
)

# ---------------------------------------
# Compute Season-Long Performance Stats
# ---------------------------------------
# - Aggregates sum and mean stats for each team per season.
# - Helps the model compare teams based on full-season performance.
# ---------------------------------------
regular_matches, all_lv_sum_cols, all_lv_mean_cols, all_hv_sum_cols, all_hv_mean_cols = get_sum_and_mean_detailed_cols(
    regular_matches,
    lv_detailed_cols,
    all_lv_sum_cols,
    all_lv_mean_cols,
    season_columns_lv_team,
    hv_detailed_cols,
    all_hv_sum_cols,
    all_hv_mean_cols,
    season_columns_hv_team,
)

# ---------------------------------------
# Track Head-to-Head Performance
# ---------------------------------------
# - Captures stats from previous matchups between two teams.
# - Helps identify trends where one team consistently beats another.
# ---------------------------------------
regular_matches, all_both_sum_cols, all_both_mean_cols = get_head_to_head_cols(
    regular_matches,
    hv_detailed_cols,
    lv_detailed_cols,
    season_columns_both_teams,
    all_both_sum_cols,
    all_both_mean_cols,
)

# ---------------------------------------
# Load NCAA Tournament Seeds
# ---------------------------------------
# - NCAA assigns seeds to tournament teams, which is a key predictor of success.
# - Merges seed data for both men’s and women’s tournaments.
# ---------------------------------------
seeds = []
for gender in ["M", "W"]:
    seed = pd.read_csv(f"{root_folder}/{gender}NCAATourneySeeds.csv")
    seeds.append(seed)

# Combine seeds into a single DataFrame
seeds = pd.concat(seeds)

# Merge seed data into match dataset
regular_matches, all_seed_cols = get_seed_data(seeds, regular_matches)

# ---------------------------------------
# Merge Coach Data
# ---------------------------------------
# - Some coaches consistently build strong teams, while others struggle.
# - Mapping coaches to teams might help track coaching impact.
# ---------------------------------------
coaches = pd.read_csv(f"{root_folder}/MTeamCoaches.csv")
coaches_map = coaches.set_index(["Season", "TeamID"])["CoachName"].to_dict()

# Assign coach names to teams in the dataset
regular_matches["HVCoach"] = regular_matches.apply(
    lambda row: coaches_map.get((row["Season"], row["HVTeamID"])), axis=1
)
regular_matches["LVCoach"] = regular_matches.apply(
    lambda row: coaches_map.get((row["Season"], row["LVTeamID"])), axis=1
)

# ---------------------------------------
# Track Tournament Round (Pressure Context)
# ---------------------------------------
# - Maps tournament games to specific rounds (e.g., Sweet 16, Final Four).
# - Helps identify how teams perform under high-pressure situations.
# ---------------------------------------
regular_matches["TourneyRound"] = regular_matches["DayNum"].map(tourney_dict)

# ---------------------------------------
# Track Past Tournament Performance
# ---------------------------------------
# - Counts how far each team advanced in past tournaments.
# - Helps spot teams with a history of deep runs (e.g., Cinderella stories).
# ---------------------------------------
lv_team_results = get_lv_tourney_cols(regular_matches)
hv_team_results = get_hv_tourney_cols(regular_matches)

# Merge tournament performance into dataset
regular_matches = regular_matches.merge(lv_team_results, on=season_columns_lv_team, how="left")
regular_matches = regular_matches.merge(hv_team_results, on=season_columns_hv_team, how="left")

# ---------------------------------------
# Split Tournament Features into Regular and Cumulative
# ---------------------------------------
# - "prev_tourney_cols": Tracks past tournament performance at each round.
# - "prev_tourney_cumsum_cols": Tracks cumulative past tournament performance.
# ---------------------------------------
tourney_columns = regular_matches.columns[-28:]  # Last 28 columns are tournament-related

# Features tracking tournament results from previous seasons
prev_tourney_cols = [col for col in tourney_columns if 'CumSum' not in col]

# Features tracking cumulative past tournament performance
prev_tourney_cumsum_cols = [col for col in tourney_columns if 'CumSum' in col]

# Display available tournament columns
tourney_columns


# ---------------------------------------
# Track Previous Season & Cumulative Performance
# ---------------------------------------
# - For each team, track their stats from the previous season.
# - Compute cumulative season-long performance trends.
# - Helps model understand if a team is improving or declining over time.
# ---------------------------------------

for col in (
    all_hv_mean_cols + all_lv_mean_cols + all_hv_sum_cols + all_lv_sum_cols + all_both_mean_cols + all_both_sum_cols
):

    prev_col = col + "PrevSeason"  # Previous season's version of this stat
    cum_sum_col = prev_col + "Cumsum"  # Cumulative sum of the previous season's stat

    # ---------------------------------------
    # Handle LV Team Stats (Lower-Ranked Teams)
    # ---------------------------------------
    if col in all_lv_sum_cols + all_lv_mean_cols:
        lv_per_season_per_team = regular_matches.drop_duplicates(subset=season_columns_lv_team)  # One entry per season/team

        # Shift previous season's stat forward so it aligns with current season
        lv_per_season_per_team[prev_col] = lv_per_season_per_team.groupby(by=[lv_team_column])[col].shift(1)

        # Compute cumulative sum to track long-term team performance
        lv_per_season_per_team[cum_sum_col] = lv_per_season_per_team.groupby(by=[lv_team_column])[prev_col].cumsum()

        # Keep only relevant columns for merging
        lv_per_season_per_team = lv_per_season_per_team[season_columns_lv_team + [prev_col, cum_sum_col]]

        # Merge back into main dataset
        regular_matches = regular_matches.merge(lv_per_season_per_team, on=season_columns_lv_team, how="left")

    # ---------------------------------------
    # Handle HV Team Stats (Higher-Ranked Teams)
    # ---------------------------------------
    elif col in all_hv_sum_cols + all_hv_mean_cols:
        hv_per_season_per_team = regular_matches.drop_duplicates(subset=season_columns_hv_team)  # One entry per season/team

        # Shift previous season's stat forward so it aligns with current season
        hv_per_season_per_team[prev_col] = hv_per_season_per_team.groupby(by=[hv_team_column])[col].shift(1)

        # Compute cumulative sum to track long-term team performance
        hv_per_season_per_team[cum_sum_col] = hv_per_season_per_team.groupby(by=[hv_team_column])[prev_col].cumsum()

        # Keep only relevant columns for merging
        hv_per_season_per_team = hv_per_season_per_team[season_columns_hv_team + [prev_col, cum_sum_col]]

        # Merge back into main dataset
        regular_matches = regular_matches.merge(hv_per_season_per_team, on=season_columns_hv_team, how="left")

    # ---------------------------------------
    # Handle Head-to-Head Performance Stats
    # ---------------------------------------
    # - Looks at previous season’s matchups between two teams.
    # - Helps track if a team consistently beats another team.
    # ---------------------------------------
    else:
        head_to_head_per_season_per_team = regular_matches.drop_duplicates(subset=season_columns_lv_team)  # One entry per season/team

        # Shift previous season's head-to-head stat forward
        head_to_head_per_season_per_team[prev_col] = head_to_head_per_season_per_team.groupby(by=both_team_column)[col].shift(1)

        # Compute cumulative sum of past head-to-head matchups
        head_to_head_per_season_per_team[cum_sum_col] = head_to_head_per_season_per_team.groupby(by=both_team_column)[prev_col].cumsum()

        # Keep only relevant columns for merging
        head_to_head_per_season_per_team = head_to_head_per_season_per_team[
            season_columns_both_teams + [prev_col, cum_sum_col]
        ]

        # Merge back into main dataset
        regular_matches = regular_matches.merge(
            head_to_head_per_season_per_team, on=season_columns_both_teams, how="left"
        )

    # ---------------------------------------
    # Store These Features for Later Use
    # ---------------------------------------
    all_prev_season_columns.append(prev_col)
    all_cumsum_columns.append(cum_sum_col)

    # Optionally drop the original column if needed
    # regular_matches = regular_matches.drop(columns=[col])


# Track previous season's stats for each per-season feature
prev_per_season_columns = [col + "PrevSeason" for col in per_season_columns]

# Track cumulative sum of previous season's stats
prev_per_season_cumsum_columns = [col + "Cumsum" for col in prev_per_season_columns]

# ---------------------------------------
# Mean-Based Features (Averages)
# ---------------------------------------
# - We track previous season's mean stats for HV (higher-ranked) and LV (lower-ranked) teams.
# - Cumulative sum versions capture season-long trends in performance.
# ---------------------------------------

# HV team mean stats from last season
prev_all_hv_mean_cols = [col + "PrevSeason" for col in all_hv_mean_cols]

# LV team mean stats from last season
prev_all_lv_mean_cols = [col + "PrevSeason" for col in all_lv_mean_cols]

# HV team cumulative mean stats over seasons
prev_all_hv_mean_cumsum_cols = [col + "Cumsum" for col in prev_all_hv_mean_cols]

# LV team cumulative mean stats over seasons
prev_all_lv_mean_cumsum_cols = [col + "Cumsum" for col in prev_all_lv_mean_cols]

# ---------------------------------------
# Sum-Based Features (Total Counts)
# ---------------------------------------
# - We track total stats from last season for both HV and LV teams.
# - These help identify teams that consistently perform at a high level.
# ---------------------------------------

# HV team total stats from last season
prev_all_hv_sum_cols = [col + "PrevSeason" for col in all_hv_sum_cols]

# LV team total stats from last season
prev_all_lv_sum_cols = [col + "PrevSeason" for col in all_lv_sum_cols]

# HV team cumulative sum of stats over seasons
prev_all_hv_sum_cumsum_cols = [col + "Cumsum" for col in prev_all_hv_sum_cols]

# LV team cumulative sum of stats over seasons
prev_all_lv_sum_cumsum_cols = [col + "Cumsum" for col in prev_all_lv_sum_cols]

# ---------------------------------------
# Head-to-Head Matchup Features
# ---------------------------------------
# - Tracks how teams performed against each other in past seasons.
# - Helps determine if an underdog has a history of beating certain teams.
# ---------------------------------------

# Mean stats for both teams from previous matchups
prev_all_both_mean_cols = [col + "PrevSeason" for col in all_both_mean_cols]

# Cumulative sum of previous mean stats for both teams
prev_all_both_mean_cumsum_cols = [col + "Cumsum" for col in prev_all_both_mean_cols]

# Total stats from previous matchups
prev_all_both_sum_cols = [col + "PrevSeason" for col in all_both_sum_cols]

# Cumulative total stats from previous matchups
prev_all_both_sum_cumsum_cols = [col + "Cumsum" for col in prev_all_both_sum_cols]

# ---------------------------------------
# Final Lists of All Feature Groups
# ---------------------------------------
# These combine everything above into larger feature groups that will
# be used in model training.
# ---------------------------------------

# All cumulative mean features
prev_all_mean_cumsum_cols = prev_all_hv_mean_cumsum_cols + prev_all_lv_mean_cumsum_cols

# All previous season mean features
prev_all_mean_cols = prev_all_hv_mean_cols + prev_all_lv_mean_cols

# All previous season sum features
prev_all_sum_cols = prev_all_hv_sum_cols + prev_all_lv_sum_cols

# All cumulative sum features
prev_all_sum_cumsum_cols = prev_all_hv_sum_cumsum_cols + prev_all_lv_sum_cumsum_cols


from sklearn.preprocessing import LabelEncoder

# ---------------------------------------
# Feature Selection & Hyperparameter Tuning
# ---------------------------------------
# - Uses Optuna to determine the best combination of features.
# - Optimizes for lowest Brier Score (better probabilistic predictions).
# - Dynamically selects different feature groups to see which improve the model.
# ---------------------------------------

df = regular_matches.copy()  # Work with a copy to avoid modifying the original dataset

# ---------------------------------------
# Encode Categorical Features
# ---------------------------------------
# - Converts categorical columns into numerical labels for model training.
# - This ensures ML models can process team/coach names, regions, etc.
# ---------------------------------------
def encode_categorical_features(df, categorical_columns):
    df = df.copy()  # Avoid modifying the original DataFrame
    
    for col in categorical_columns:
        try:
            if df[col].dtype == 'object':  # Only encode string-based categorical columns
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))    
        except:
            continue  # Skip problematic columns
    return df


# ---------------------------------------
# Define Optuna's Objective Function
# ---------------------------------------
# - Randomly selects different feature groups and evaluates their impact.
# - Minimizes Brier Score, which measures prediction accuracy.
# - Helps identify which features contribute most to better predictions.
# ---------------------------------------
def objective(trial):
    # Randomly enable/disable different feature groups
    feature_flags = {
        'prev_tourney_cols': trial.suggest_categorical('prev_tourney_cols', [True, False]),
        'prev_tourney_cumsum_cols': trial.suggest_categorical('prev_tourney_cumsum_cols', [True, False]),
        'prev_all_mean_cols': trial.suggest_categorical('prev_all_mean_cols', [True, False]),
        'all_seed_cols': trial.suggest_categorical('all_seed_cols', [True, False]),
        'coach_columns': trial.suggest_categorical('coach_columns', [True, False]),
        'all_momentum_2_columns': trial.suggest_categorical('all_momentum_2_columns', [True, False]),
        'all_momentum_10_columns': trial.suggest_categorical('all_momentum_10_columns', [True, False]),
    }

    # Map feature flags to actual column lists
    feature_groups = {
        'prev_tourney_cols': prev_tourney_cols,
        'prev_tourney_cumsum_cols': prev_tourney_cumsum_cols,
        'prev_all_mean_cols': prev_all_mean_cols,
        'all_seed_cols': list(all_seed_cols),
        'coach_columns': list(coach_columns),
        'all_momentum_2_columns': [col for col in regular_matches.columns if '_R2S' in col],  # Features tracking recent momentum (2-game avg)
        'all_momentum_10_columns': [col for col in regular_matches.columns if '_R10S' in col],  # Features tracking longer momentum (10-game avg)
    }

    # Combine selected feature groups into a single list
    total_features = [col for key, cols in feature_groups.items() if feature_flags[key] for col in cols]

    try:
        # Prepare training data
        X = df[total_features].copy()
        y = df[target_col].copy()
    
        # Encode categorical features before feeding into the model
        categorical_columns = ['LVSeedRegionPrev', 'HVSeedRegionPrev', 'SeedRegionGroup', 'HVCoach', 'LVCoach']
        X = encode_categorical_features(X, categorical_columns)
    
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
        # Train a LightGBM model
        model = lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1)
        model.fit(X_train, y_train)
    
        # Predict probability for LVWin
        y_pred_proba = model.predict_proba(X_test)[:, 1]  
        brier_score = brier_score_loss(y_test, y_pred_proba)  # Lower is better

    except:
        return np.inf  # If anything goes wrong, return a bad score to discard the trial

    return brier_score


# ---------------------------------------
# Run Hyperparameter Tuning with Optuna
# ---------------------------------------
# - Runs 300 trials to find the best feature combination.
# - Uses NSGA-II (a genetic algorithm) to efficiently search feature space.
# - The best feature set will be used for final model training.
# ---------------------------------------
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.NSGAIISampler())
study.optimize(objective, n_trials=300)

# Output best result
print("Best trial:", study.best_trial)


submission_file = pd.read_csv(f"{root_folder}/SampleSubmissionStage2.csv")
submission_file[["Season", "LVTeamID", "HVTeamID"]] = submission_file["ID"].str.split("_", expand=True).astype(int)
submission_file['CRType'] = 'NCAA'
testing = regular_matches[
    [
        "Season",
        "DayNum",
        # "WTeamID",
        # "LTeamID",
        "CRType",
        "Gender",
        "LVTeamID",
        "HVTeamID",
        "LVWin",
        "HVWin",
        "LVCity",
        "LVState",
        "LVTeamName",
        "LVFirstD1Season",
        "HVCity",
        "HVState",
        "HVTeamName",
        "HVFirstD1Season",
        "HVCoach",
        "LVCoach",
        "LVSeed",
        "HVSeed",
        "LVSeedRegion",
        "HVSeedRegion",
        'LVFGA_R2S',
        'HVFGA_R2S',
        'LVScore_R10S',
        'HVScore_R10S',
        'HVScorePerSeasonMeanPrevSeason',
        'LVScorePerSeasonMeanPrevSeason',
        # "HVFGM",
        # "LVFGM",
        # "HVFGA",
        # "LVFGA",
        # "HVFGM3",
        # "LVFGM3",
        # "HVFGA3",
        # "LVFGA3",
        # "HVFTM",
        # "LVFTM",
        # "HVFTA",
        # "LVFTA",
        # "HVOR",
        # "LVOR",
        # "HVDR",
        # "LVDR",
        # "HVAst",
        # "LVAst",
        # "HVTO",
        # "LVTO",
        # "HVStl",
        # "LVStl",
        # "HVBlk",
        # "LVBlk",
        # "HVPF",
        # "LVPF",
        # "HVScore",
        # "LVScore",
        "LVPrevFinals",
        "LVPrevLast16",
        "LVPrevLast8",
        "LVPrevPlayin",
        "LVPrevRound1",
        "LVPrevRound2",
        "LVPrevSemiFinals",
        "HVPrevFinals",
        "HVPrevLast16",
        "HVPrevLast8",
        "HVPrevPlayin",
        "HVPrevRound1",
        "HVPrevRound2",
        "HVPrevSemiFinals",
        "LVPrevCumSumFinals",
        "LVPrevCumSumLast16",
        "LVPrevCumSumLast8",
        "LVPrevCumSumPlayin",
        "LVPrevCumSumRound1",
        "LVPrevCumSumRound2",
        "LVPrevCumSumSemiFinals",
        "HVPrevCumSumFinals",
        "HVPrevCumSumLast16",
        "HVPrevCumSumLast8",
        "HVPrevCumSumPlayin",
        "HVPrevCumSumRound1",
        "HVPrevCumSumRound2",
        "HVPrevCumSumSemiFinals",
    ]
]


all_results_dict = {}
for season in testing.Season.unique()[1:]:
    print(f'--------------------{season}---------------------')

    if season == 2020:
        continue

    single_season = testing[testing["Season"] <= season]

    training_data = single_season[single_season['Season'] <= season].drop(columns='HVWin')
    
    testing_data = single_season[(single_season['CRType'] == 'NCAA')&(single_season['Season'] == season)]
    print('Testing Data:', testing_data.shape)
    print('Training Data:', training_data.shape)
    if season == 2025:
        real_testing_data = submission_file[['ID', 'Season', 'LVTeamID', 'HVTeamID', 'CRType']]
    else:
        real_testing_data = testing_data[['Season', 'LVTeamID', 'HVTeamID', 'LVWin', 'CRType']]

    lv_training_merge = training_data.groupby(by=['LVTeamID']).tail(1).drop(columns=['HVTeamID', 'LVWin', 'HVCity', 'HVState', 'HVTeamName', 'HVFirstD1Season', 'HVCoach', 'HVSeed', 'HVSeedRegion', 'HVPrevFinals', 'HVPrevLast16', 'HVPrevLast8', 'HVPrevPlayin', 'HVPrevRound1', 'HVPrevRound2', 'HVPrevSemiFinals', 'CRType', 'DayNum', 'Gender', 'HVFGA_R2S', 'HVScore_R10S', "HVPrevCumSumFinals", "HVPrevCumSumLast16","HVPrevCumSumLast8","HVPrevCumSumPlayin","HVPrevCumSumRound1","HVPrevCumSumRound2","HVPrevCumSumSemiFinals", 'HVScorePerSeasonMeanPrevSeason'])

    hv_training_merge = training_data.groupby(by=['HVTeamID']).tail(1).drop(columns=['LVTeamID', 'LVWin', 'LVCity', 'LVState', 'LVTeamName', 'LVFirstD1Season', 'LVCoach', 'LVSeed', 'LVSeedRegion', 'LVPrevFinals', 'LVPrevLast16', 'LVPrevLast8', 'LVPrevPlayin', 'LVPrevRound1', 'LVPrevRound2', 'LVPrevSemiFinals', 'CRType', 'DayNum', 'Gender', 'LVFGA_R2S', 'LVScore_R10S', "LVPrevCumSumFinals", "LVPrevCumSumLast16", "LVPrevCumSumLast8", "LVPrevCumSumPlayin", "LVPrevCumSumRound1", "LVPrevCumSumRound2", "LVPrevCumSumSemiFinals", 'LVScorePerSeasonMeanPrevSeason'])

    real_testing_data = real_testing_data.merge(lv_training_merge, on=['Season', 'LVTeamID'], how='left')
    real_testing_data = real_testing_data.merge(hv_training_merge, on=['Season', 'HVTeamID'], how='left')
    gender_dict = pd.concat([training_data[['LVTeamID', 'Gender']].rename(columns={'LVTeamID' : 'TeamID'}), training_data[['HVTeamID', 'Gender']].rename(columns={'HVTeamID' : 'TeamID'})]).drop_duplicates().set_index('TeamID')['Gender'].to_dict()
    real_testing_data['Gender'] = real_testing_data['LVTeamID'].replace(gender_dict)

    # Identify categorical features
    categorical_cols = training_data.select_dtypes(include='object').columns.tolist()
    real_testing_data[categorical_cols[:-2]] = real_testing_data[categorical_cols[:-2]].fillna('Unknown')
    training_data[categorical_cols[:-2]] = training_data[categorical_cols[:-2]].fillna('Unknown')

    real_testing_data[categorical_cols[-2:]] = real_testing_data[categorical_cols[-2:]].fillna('U')
    training_data[categorical_cols[-2:]] = training_data[categorical_cols[-2:]].fillna('U')


    # Label Encode categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        training_data[col] = le.fit_transform(training_data[col].astype(str))
        real_testing_data[col] = le.transform(real_testing_data[col].astype(str))
        label_encoders[col] = le  # Store encoders for future use if needed

    training_data['SeedDiff'] = training_data['HVSeed'] - training_data['LVSeed']
    real_testing_data['SeedDiff'] = real_testing_data['HVSeed'] - real_testing_data['LVSeed']
    
    # Define features and target
    target_col = 'LVWin'
    features = list(set(training_data.drop(columns=['DayNum', 'LVWin']).columns) & set(real_testing_data.columns))

    X = training_data[features]
    y = training_data[target_col]

    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # LightGBM Model
    lgb_model = lgb.LGBMClassifier(**{'verbose':-1, 'learning_rate': 0.23703169748956787, 'num_leaves': 31, 'max_depth': 15, 'min_child_samples': 65, 'subsample': 0.5961344179199886, 'colsample_bytree': 0.7837635648550635})
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict_proba(X_val)[:, 1]

    # XGBoost Model
    xgb_model = xgb.XGBClassifier(**{'learning_rate': 0.23703169748956787, 'num_leaves': 31, 'max_depth': 15, 'min_child_samples': 65, 'subsample': 0.5961344179199886, 'colsample_bytree': 0.7837635648550635, 'eval_metric':'logloss', 'enable_categorical':True})
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict_proba(X_val)[:, 1]

    # Ensemble Predictions
    ensemble_preds = (lgb_preds + xgb_preds) / 2

    ensemble_binary = (ensemble_preds > 0.5).astype(int)
    lgb_binary = (lgb_preds > 0.5).astype(int)
    xgb_binary = (xgb_preds > 0.5).astype(int)

    # Evaluate Brier Score
    lgb_brier_score = brier_score_loss(y_val, lgb_preds)
    lgb_accuracy = accuracy_score(y_val, lgb_binary)
    
    xgb_brier_score = brier_score_loss(y_val, xgb_preds)
    xgb_accuracy = accuracy_score(y_val, xgb_binary)
    
    ensemble_brier_score = brier_score_loss(y_val, ensemble_preds)
    ensemble_accuracy = accuracy_score(y_val, ensemble_binary)

    print(f"LightGBM Brier Score: {lgb_brier_score:.4f}, Accuracy: {lgb_accuracy:.4f}")
    print(f"XGBoost Brier Score: {xgb_brier_score:.4f}, Accuracy: {xgb_accuracy:.4f}")
    print(f"Ensemble Brier Score: {ensemble_brier_score:.4f}, Accuracy: {ensemble_accuracy:.4f}")


    # Predicting on the real_testing_data
    real_testing_data['LGB_Pred'] = lgb_model.predict_proba(real_testing_data[features])[:, 1]
    real_testing_data['XGB_Pred'] = xgb_model.predict_proba(real_testing_data[features])[:, 1]
    real_testing_data['Ensemble_Pred'] = (real_testing_data['LGB_Pred'] + real_testing_data['XGB_Pred']) / 2

    if season == 2025:
        continue
    # Evaluate Brier Score on real_testing_data
    real_lgb_brier_score = brier_score_loss(real_testing_data[target_col], real_testing_data['LGB_Pred'])
    real_xgb_brier_score = brier_score_loss(real_testing_data[target_col], real_testing_data['XGB_Pred'])
    real_ensemble_brier_score = brier_score_loss(real_testing_data[target_col], real_testing_data['Ensemble_Pred'])

    real_lgb_accuracy = accuracy_score(real_testing_data[target_col], (real_testing_data['LGB_Pred'] > 0.5).astype(int))
    real_xgb_accuracy = accuracy_score(real_testing_data[target_col], (real_testing_data['XGB_Pred'] > 0.5).astype(int))
    real_ensemble_accuracy = accuracy_score(real_testing_data[target_col], (real_testing_data['Ensemble_Pred'] > 0.5).astype(int))
    

    print(f"Real Testing Data LightGBM Brier Score: {real_lgb_brier_score:.4f}, Accuracy: {real_lgb_accuracy:.4f}")
    print(f"Real Testing Data XGBoost Brier Score: {real_xgb_brier_score:.4f}, Accuracy: {real_xgb_accuracy:.4f}")
    print(f"Real Testing Data Ensemble Brier Score: {real_ensemble_brier_score:.4f}, Accuracy: {real_ensemble_accuracy:.4f}")

    # Multi-Figure Feature Importance Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(features, lgb_model.feature_importances_)
    axes[0].set_title('LightGBM Feature Importance')

    axes[1].barh(features, xgb_model.feature_importances_)
    axes[1].set_title('XGBoost Feature Importance')

    plt.tight_layout()
    plt.show()

    all_results_dict[season] = {'real_lgb_brier_score': real_lgb_brier_score, 'real_xgb_brier_score': real_xgb_brier_score, 'real_ensemble_brier_score': real_ensemble_brier_score}


real_testing_data[['ID', 'LGB_Pred', 'XGB_Pred']]

# real_testing_data['Pred'] = (real_testing_data['LGB_Pred'] + real_testing_data['XGB_Pred']) / 2
real_testing_data['Pred'] = real_testing_data['XGB_Pred']
real_testing_data[['ID', 'Pred']].to_csv('submission.csv', index=False)

pd.DataFrame(all_results_dict).mean(axis=1)

