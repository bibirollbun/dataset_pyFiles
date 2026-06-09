import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings
from IPython.display import display
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

pd.set_option("display.max_columns", None)
pd.options.mode.chained_assignment = None
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


# Gets all the season columns needed ✓
lv_win_column = "LVWin"
hv_win_column = "HVWin"

lv_score_column = "LVScore"
hv_score_column = "HVScore"

lv_team_column = "LVTeamID"
hv_team_column = "HVTeamID"

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
root_folder = '/kaggle/input/march-machine-learning-mania-2025'
both_team_column = ["LVTeamID", "HVTeamID"]

detailed_cols = [
    "FGM",
    "FGA",
    "FGM3",
    "FGA3",
    "FTM",
    "FTA",
    "OR",
    "DR",
    "Ast",
    "TO",
    "Stl",
    "Blk",
    "PF",
    "Score",
]

win_cols = ["W" + col for col in detailed_cols]

lose_cols = ["L" + col for col in detailed_cols]

hv_detailed_cols = []
lv_detailed_cols = []

all_hv_sum_cols = []
all_hv_mean_cols = []

all_both_sum_cols = []
all_both_mean_cols = []

all_lv_sum_cols = []
all_lv_mean_cols = []

all_prev_season_columns = []
all_cumsum_columns = []

all_momentum_columns = []
shift_values = [5, 10, 15]

season_columns_both_teams = ["Season", lv_team_column, hv_team_column]
season_columns_lv_team = ["Season", lv_team_column]
season_columns_hv_team = ["Season", hv_team_column]
per_season_columns = [
    "TimesPlayedPerSeason",
    "LVWinCountPerSeason",
    "HVWinCountPerSeason",
    "LVWinCountPerSeasonTotal",
    "HVWinCountPerSeasonTotal",
    "LVScorePerSeason",
    "HVScorePerSeason",
]


# prev_ranking_cols = ["Prev" + col for col in ranking_cols]
# lv_prev_ranking_cols = ["LV" + col for col in prev_ranking_cols]
# hv_prev_ranking_cols = ["HV" + col for col in prev_ranking_cols]


all_games = []
all_detailed_results = []
all_teams = []

gender_dict = {"M": "men", "W": "women"}

coach_columns = ["HVCoach", "LVCoach"]

target_col = 'LVWin'


def get_teams_info(team_games_df, gender):
    teams = pd.read_csv(f"{root_folder}/{gender}Teams.csv")
    teams_spellings = pd.read_csv(f"{root_folder}/{gender}TeamSpellings.csv")

    teams_spellings = teams_spellings.groupby(by="TeamID")["TeamNameSpelling"].unique().reset_index()
    teams_info = teams.merge(teams_spellings, on="TeamID")

    teams_home_state_city = (
        team_games_df[team_games_df["WLoc"] == "H"][["Season", "WTeamID", "City", "State"]]
        .rename(columns={"WTeamID": "TeamID"})
        .drop_duplicates()
    )
    teams_info = teams_home_state_city.merge(teams_info, on="TeamID")

    return teams_info


def get_historic_games(gender, city_df):
    compact_tourney_results = pd.read_csv(f"{root_folder}/{gender}NCAATourneyCompactResults.csv")
    tourney_results_d = pd.read_csv(f"{root_folder}/{gender}NCAATourneyDetailedResults.csv")

    compact_season_results = pd.read_csv(f"{root_folder}/{gender}RegularSeasonCompactResults.csv")
    reg_season_results_d = pd.read_csv(f"{root_folder}/{gender}RegularSeasonDetailedResults.csv")

    games_played = pd.read_csv(f"{root_folder}/{gender}GameCities.csv")

    regular_season_and_ncaa = games_played[games_played["CRType"] != "Secondary"]

    compact_tourney_results["Tournament"] = True
    compact_season_results["Tournament"] = False

    compact_tour_reg = pd.concat([compact_tourney_results, compact_season_results], axis=0).sort_values(
        by=["Season", "DayNum"], ascending=True
    )

    detailed_tour_reg = pd.concat([tourney_results_d, reg_season_results_d], axis=0).sort_values(
        by=["Season", "DayNum"], ascending=True
    )

    regular_season_and_ncaa = regular_season_and_ncaa.merge(
        compact_tour_reg, on=["Season", "DayNum", "WTeamID", "LTeamID"], how="left"
    ).sort_values(by=["Season", "DayNum"], ascending=True)

    regular_season_and_ncaa = regular_season_and_ncaa.merge(city_df, on=["CityID"], how="left")

    regular_season_and_ncaa["Gender"] = gender

    return regular_season_and_ncaa, detailed_tour_reg


def add_detail_cols(
    regular_matches, all_detailed_results, all_teams, lv_team_column, hv_team_column, lv_win_column, hv_win_column
):

    regular_matches[lv_team_column] = regular_matches[["WTeamID", "LTeamID"]].min(axis=1)
    regular_matches[hv_team_column] = regular_matches[["WTeamID", "LTeamID"]].max(axis=1)

    regular_matches[lv_win_column] = np.where(regular_matches["WTeamID"] == regular_matches[lv_team_column], 1, 0)
    regular_matches[hv_win_column] = np.where(regular_matches["WTeamID"] == regular_matches[hv_team_column], 1, 0)

    regular_matches = regular_matches.merge(
        all_detailed_results,
        on=["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc", "NumOT"],
        how="left",
    )

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


# This converts the Win and Lose columns to High and Low value columns


def get_detailed_hv_lv_cols(regular_matches, win_cols, lose_cols, hv_win_column, lv_win_column):

    for col_idx in range(len(win_cols)):
        win_col = win_cols[col_idx]
        lose_col = lose_cols[col_idx]

        hv_col = "HV" + win_col[1:]
        lv_col = "LV" + win_col[1:]

        regular_matches[hv_col] = (regular_matches[hv_win_column] * regular_matches[win_col]) + (
            regular_matches[lv_win_column] * regular_matches[lose_col]
        )
        regular_matches[lv_col] = (regular_matches[lv_win_column] * regular_matches[win_col]) + (
            regular_matches[hv_win_column] * regular_matches[lose_col]
        )

        regular_matches = regular_matches.drop(columns=[win_col, lose_col])
        lv_detailed_cols.append(lv_col)
        hv_detailed_cols.append(hv_col)

    return regular_matches, lv_detailed_cols, hv_detailed_cols


def get_lv_rolling_scores(
    regular_matches, lv_team_column, hv_team_column, lv_deet_col, hv_deet_col, deet_col, shift_values
):
    all_lv_data = []

    for lv_id, lv_data_orinal in regular_matches[
        ["Season", "DayNum", lv_team_column, hv_team_column, lv_deet_col, hv_deet_col]
    ].groupby(by=[lv_team_column]):

        lv_data = lv_data_orinal.copy()

        lv_data[f"Diff{deet_col}"] = lv_data[lv_deet_col] - lv_data[hv_deet_col]

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

        for i in shift_values:
            hv_data[f"{hv_deet_col}_R{i}S"] = hv_data[hv_deet_col].rolling(window=i).mean().shift(1)
            hv_data[f"{lv_deet_col}AgainstHV_R{i}S"] = hv_data[lv_deet_col].rolling(window=i).mean().shift(1)

        hv_data = hv_data.drop(columns=[lv_deet_col, hv_deet_col])
        all_hv_data.append(hv_data)

    return pd.concat(all_hv_data)


def get_momentum_cols(all_momentum_columns, hv_deet_col, lv_deet_col, deet_col, shift_values):

    all_momentum_columns += [f"{hv_deet_col}_R{i}S" for i in shift_values]
    all_momentum_columns += [f"{lv_deet_col}_R{i}S" for i in shift_values]
    all_momentum_columns += [f"{lv_deet_col}AgainstHV_R{i}S" for i in shift_values]
    all_momentum_columns += [f"{hv_deet_col}AgainstLV_R{i}S" for i in shift_values]
    all_momentum_columns += [f"Diff{deet_col}_R{i}S" for i in shift_values]

    return all_momentum_columns


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

        for i in shift_values:
            regular_matches[f"{deet_col}_diff_R{i}S"] = (
                regular_matches[f"{hv_deet_col}_R{i}S"] - regular_matches[f"{lv_deet_col}_R{i}S"]
            )
            all_momentum_columns.append(f"{deet_col}_diff_R{i}S")

    return regular_matches, all_momentum_columns


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


def get_seed_data(seeds, regular_matches):
    # Split the 'Seed' column
    all_seed_cols = []

    seeds["Region"] = seeds["Seed"].str.extract(r"([A-Za-z]+)")  # Extract letters
    seeds["Seed"] = seeds["Seed"].str.extract(r"(\d+)").astype(int)  # Extract numbers

    regular_matches = (
        regular_matches.merge(seeds, left_on=["Season", "LVTeamID"], right_on=["Season", "TeamID"], how="left")
        .drop(columns=["TeamID"])
        .rename(columns={"Seed": "LVSeed", "Region": "LVSeedRegion"})
    )

    regular_matches["LVSeed"] = regular_matches["LVSeed"].fillna(17)
    regular_matches["LVSeedRegion"] = regular_matches["LVSeedRegion"].fillna("U")

    regular_matches = (
        regular_matches.merge(seeds, left_on=["Season", "HVTeamID"], right_on=["Season", "TeamID"], how="left")
        .drop(columns=["TeamID"])
        .rename(columns={"Seed": "HVSeed", "Region": "HVSeedRegion"})
    )

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
    prev_season_seeds["Season"] += 1  # Shift the season forward to align previous season data

    for col in ["Seed", "SeedRegion"]:

        lv_col = "LV" + col
        hv_col = "HV" + col

        lv_prev_col = lv_col + "Prev"
        hv_prev_col = hv_col + "Prev"

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

    regular_matches[["HVSeedPrev", "LVSeedPrev"]] = regular_matches[["HVSeedPrev", "LVSeedPrev"]].fillna(17)
    regular_matches["SeedDiff"] = regular_matches["HVSeedPrev"] - regular_matches["LVSeedPrev"]
    regular_matches[["HVSeedRegionPrev", "LVSeedRegionPrev"]] = regular_matches[
        ["HVSeedRegionPrev", "LVSeedRegionPrev"]
    ].fillna("U")

    regular_matches["SeedRegionGroup"] = regular_matches["HVSeedRegionPrev"] + regular_matches["LVSeedRegionPrev"]

    all_seed_cols.append("SeedDiff")
    all_seed_cols.append("SeedRegionGroup")

    return regular_matches, all_seed_cols


city_df = pd.read_csv(f"{root_folder}/Cities.csv")

for gender in ["W", "M"]:

    regular_season_and_ncaa, detailed_tour_reg = get_historic_games(gender, city_df)
    team_info = get_teams_info(regular_season_and_ncaa, gender)

    all_games.append(regular_season_and_ncaa)
    all_teams.append(team_info)
    all_detailed_results.append(detailed_tour_reg)

all_games = pd.concat(all_games).reset_index(drop=True).sort_values(by=["Season", "DayNum"], ascending=True)
all_teams = pd.concat(all_teams).reset_index(drop=True)
all_detailed_results = (
    pd.concat(all_detailed_results).reset_index(drop=True).sort_values(by=["Season", "DayNum"], ascending=True)
)

regular_matches = add_detail_cols(
    all_games.copy(), all_detailed_results, all_teams, lv_team_column, hv_team_column, lv_win_column, hv_win_column
)
regular_matches, lv_detailed_cols, hv_detailed_cols = get_detailed_hv_lv_cols(
    regular_matches, win_cols, lose_cols, hv_win_column, lv_win_column
)

regular_matches, all_momentum_columns = get_all_rolling_scores(
    regular_matches, lv_team_column, hv_team_column, shift_values, detailed_cols, all_momentum_columns
)

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

regular_matches, all_both_sum_cols, all_both_mean_cols = get_head_to_head_cols(
    regular_matches,
    hv_detailed_cols,
    lv_detailed_cols,
    season_columns_both_teams,
    all_both_sum_cols,
    all_both_mean_cols,
)

# NCAA tournament seeds
seeds = []
for gender in ["M", "W"]:
    seed = pd.read_csv(f"{root_folder}/{gender}NCAATourneySeeds.csv")
    seeds.append(seed)

seeds = pd.concat(seeds)

regular_matches, all_seed_cols = get_seed_data(seeds, regular_matches)

# Coaches info
coaches = pd.read_csv(f"{root_folder}/MTeamCoaches.csv")
coaches_map = coaches.set_index(["Season", "TeamID"])["CoachName"].to_dict()

regular_matches["HVCoach"] = regular_matches.apply(
    lambda row: coaches_map.get((row["Season"], row["HVTeamID"])), axis=1
)
regular_matches["LVCoach"] = regular_matches.apply(
    lambda row: coaches_map.get((row["Season"], row["LVTeamID"])), axis=1
)

# Tournament round columns
regular_matches["TourneyRound"] = regular_matches["DayNum"].map(tourney_dict)

lv_team_results = get_lv_tourney_cols(regular_matches)

hv_team_results = get_hv_tourney_cols(regular_matches)

regular_matches = regular_matches.merge(lv_team_results, on=season_columns_lv_team, how="left")
regular_matches = regular_matches.merge(hv_team_results, on=season_columns_hv_team, how="left")

tourney_columns = regular_matches.columns[-28:]

prev_tourney_cols = [col for col in tourney_columns if 'CumSum' not in col]
prev_tourney_cumsum_cols = [col for col in tourney_columns if 'CumSum' in col]

tourney_columns


for col in (
    all_hv_mean_cols + all_lv_mean_cols + all_hv_sum_cols + all_lv_sum_cols + all_both_mean_cols + all_both_sum_cols
):

    prev_col = col + "PrevSeason"
    cum_sum_col = prev_col + "Cumsum"

    if col in all_lv_sum_cols + all_lv_mean_cols:
        lv_per_season_per_team = regular_matches.drop_duplicates(subset=season_columns_lv_team)

        lv_per_season_per_team[prev_col] = lv_per_season_per_team.groupby(by=[lv_team_column])[col].shift(1)
        lv_per_season_per_team[cum_sum_col] = lv_per_season_per_team.groupby(by=[lv_team_column])[prev_col].cumsum()
        lv_per_season_per_team = lv_per_season_per_team[season_columns_lv_team + [prev_col, cum_sum_col]]
        regular_matches = regular_matches.merge(lv_per_season_per_team, on=season_columns_lv_team, how="left")

    elif col in all_hv_sum_cols + all_hv_mean_cols:
        hv_per_season_per_team = regular_matches.drop_duplicates(subset=season_columns_hv_team)

        hv_per_season_per_team[prev_col] = hv_per_season_per_team.groupby(by=[hv_team_column])[col].shift(1)
        hv_per_season_per_team[cum_sum_col] = hv_per_season_per_team.groupby(by=[hv_team_column])[prev_col].cumsum()
        hv_per_season_per_team = hv_per_season_per_team[season_columns_hv_team + [prev_col, cum_sum_col]]

        regular_matches = regular_matches.merge(hv_per_season_per_team, on=season_columns_hv_team, how="left")

    else:
        head_to_head_per_season_per_team = regular_matches.drop_duplicates(subset=season_columns_lv_team)

        head_to_head_per_season_per_team[prev_col] = head_to_head_per_season_per_team.groupby(by=both_team_column)[
            col
        ].shift(1)
        head_to_head_per_season_per_team[cum_sum_col] = head_to_head_per_season_per_team.groupby(by=both_team_column)[
            prev_col
        ].cumsum()
        head_to_head_per_season_per_team = head_to_head_per_season_per_team[
            season_columns_both_teams + [prev_col, cum_sum_col]
        ]

        regular_matches = regular_matches.merge(
            head_to_head_per_season_per_team, on=season_columns_both_teams, how="left"
        )

    all_prev_season_columns.append(prev_col)
    all_cumsum_columns.append(cum_sum_col)

    # regular_matches = regular_matches.drop(columns=[col])


"""
Ranking Data for men only but could use past performance to predict ranking?
"""
# # This data includes a Season a teamID and the various ranking systems
# ranking_data = pd.read_csv("data/men/MMasseyOrdinals.csv")

# pivoted_data = ranking_data.pivot(
#     index=["Season", "RankingDayNum", "TeamID"],
#     columns=["SystemName"],
#     values=["OrdinalRank"],
# ).reset_index()
# ranking_data = pd.concat(
#     [pivoted_data.drop(columns=["OrdinalRank"]), pivoted_data["OrdinalRank"]], axis=1
# )

# ranking_data.columns.values[:3] = ["Season", "RankingDayNum", "TeamID"]

# ranking_data = (
#     ranking_data.groupby(by=["Season", "TeamID"])
#     .agg(["mean", "median", "std", "min", "max"])
#     .drop(columns=["RankingDayNum"])
# )

# ranking_data.columns = [
#     "_".join(col) if isinstance(col, tuple) else col for col in ranking_data.columns
# ]
# ranking_data = ranking_data.reset_index()

# ranking_data = men_teams_info.merge(ranking_data, on=["Season", "TeamID"]).drop(
#     columns=["TeamNameSpelling", "TeamName"]
# )

# for col in ["State", "City"]:
#     ranking_data[col] = ranking_data[col].astype("category")

# mean_ranking_cols = [col for col in ranking_data.columns if "mean" in col]

# median_ranking_cols = [col for col in ranking_data.columns if "median" in col]

# min_ranking_cols = [col for col in ranking_data.columns if "min" in col]

# max_ranking_cols = [col for col in ranking_data.columns if "max" in col]

# ranking_cols = [
#     "MeanMeanRanking",
#     "MeanMedianRanking",
#     "MedianMeanRanking",
#     "MedianMedianRanking",
#     "MinMeanRanking",
#     "MinMedianRanking",
#     "MaxMeanRanking",
#     "MaxMedianRanking",
# ]

# ranking_data[ranking_cols[0]] = ranking_data.loc[:, mean_ranking_cols].mean(axis=1)
# ranking_data[ranking_cols[1]] = ranking_data.loc[:, mean_ranking_cols].median(axis=1)

# ranking_data[ranking_cols[2]] = ranking_data.loc[:, median_ranking_cols].mean(axis=1)
# ranking_data[ranking_cols[3]] = ranking_data.loc[:, median_ranking_cols].median(axis=1)

# ranking_data[ranking_cols[4]] = ranking_data.loc[:, min_ranking_cols].mean(axis=1)
# ranking_data[ranking_cols[5]] = ranking_data.loc[:, min_ranking_cols].median(axis=1)


# ranking_data[ranking_cols[6]] = ranking_data.loc[:, max_ranking_cols].mean(axis=1)
# ranking_data[ranking_cols[7]] = ranking_data.loc[:, max_ranking_cols].median(axis=1)

# ranking_data = ranking_data.drop(columns=ranking_data.iloc[:, 6:-8].columns)

# # Sort the data by TeamID and Season
# ranking_data = ranking_data.sort_values(["TeamID", "Season"])

# # For each team, shift the ranking values by one row to get the previous season's ranking
# ranking_data["PrevMeanMeanRanking"] = ranking_data.groupby("TeamID")[
#     ranking_cols[0]
# ].shift(1)
# ranking_data["PrevMeanMedianRanking"] = ranking_data.groupby("TeamID")[
#     ranking_cols[1]
# ].shift(1)

# ranking_data["PrevMedianMeanRanking"] = ranking_data.groupby("TeamID")[
#     ranking_cols[2]
# ].shift(1)
# ranking_data["PrevMedianMedianRanking"] = ranking_data.groupby("TeamID")[
#     ranking_cols[3]
# ].shift(1)

# ranking_data["PrevMinMeanRanking"] = ranking_data.groupby("TeamID")[
#     ranking_cols[4]
# ].shift(1)
# ranking_data["PrevMinMedianRanking"] = ranking_data.groupby("TeamID")[
#     ranking_cols[5]
# ].shift(1)

# ranking_data["PrevMaxMeanRanking"] = ranking_data.groupby("TeamID")[
#     ranking_cols[6]
# ].shift(1)
# ranking_data["PrevMaxMedianRanking"] = ranking_data.groupby("TeamID")[
#     ranking_cols[7]
# ].shift(1)

# ranking_data = ranking_data.drop(columns=["LastD1Season"] + ranking_cols)
# ranking_data = ranking_data.drop_duplicates(subset=["Season", "TeamID"])

# ranking_data.groupby(by=["Season"]).size().plot.bar()
# plt.title("Number of Mens Teams per Season")
# plt.ylabel("Mens Team Count")
# plt.ylabel("Season")
# plt.ylim(340, 380)
# plt.show()

# regular_matches = regular_matches.merge(
#     ranking_data.add_prefix("LV"),
#     left_on=season_columns_lv_team,
#     right_on=["LVSeason", "LVTeamID"],
#     how="left",
# )
# regular_matches = regular_matches.merge(
#     ranking_data.add_prefix("HV"),
#     left_on=season_columns_hv_team,
#     right_on=["HVSeason", "HVTeamID"],
#     how="left",
# )


prev_per_season_columns = [col + "PrevSeason" for col in per_season_columns]
prev_per_season_cumsum_columns = [col + "Cumsum" for col in prev_per_season_columns]

prev_all_hv_mean_cols = [col + "PrevSeason" for col in all_hv_mean_cols]
prev_all_lv_mean_cols = [col + "PrevSeason" for col in all_lv_mean_cols]

prev_all_hv_mean_cumsum_cols = [col + "Cumsum" for col in prev_all_hv_mean_cols]
prev_all_lv_mean_cumsum_cols = [col + "Cumsum" for col in prev_all_lv_mean_cols]

prev_all_hv_sum_cols = [col + "PrevSeason" for col in all_hv_sum_cols]
prev_all_lv_sum_cols = [col + "PrevSeason" for col in all_lv_sum_cols]

prev_all_hv_sum_cumsum_cols = [col + "Cumsum" for col in prev_all_hv_sum_cols]
prev_all_lv_sum_cumsum_cols = [col + "Cumsum" for col in prev_all_lv_sum_cols]

prev_all_both_mean_cols = [col + "PrevSeason" for col in all_both_mean_cols]
prev_all_both_mean_cumsum_cols = [col + "Cumsum" for col in prev_all_both_mean_cols]

prev_all_both_sum_cols = [col + "PrevSeason" for col in all_both_sum_cols]
prev_all_both_sum_cumsum_cols = [col + "Cumsum" for col in prev_all_both_sum_cols]

prev_all_mean_cumsum_cols = prev_all_hv_mean_cumsum_cols + prev_all_lv_mean_cumsum_cols
prev_all_mean_cols = prev_all_hv_mean_cols + prev_all_lv_mean_cols
prev_all_sum_cols = prev_all_hv_sum_cols + prev_all_lv_sum_cols
prev_all_sum_cumsum_cols = prev_all_hv_sum_cumsum_cols + prev_all_lv_sum_cumsum_cols


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
        'LVScore_R15S',
        'HVScore_R15S',
        'LVScore_R10S',
        'HVScore_R10S',
        'LVScore_R5S',
        'HVScore_R5S',
        # 'LVScoreAgainstHV_R10S',
        # 'HVScoreAgainstLV_R10S',
        # 'HVScorePerSeasonMeanPrevSeason',
        # 'LVScorePerSeasonMeanPrevSeason',
        # 'LVFGM3_R10S',
        # 'HVFGM3_R10S',
        # 'LVPF_R10S',
        # 'HVPF_R10S',
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
        # "LVPrevFinals",
        # "LVPrevLast16",
        # "LVPrevLast8",
        # "LVPrevPlayin",
        "LVPrevRound1",
        # "LVPrevRound2",
        # "LVPrevSemiFinals",
        # "HVPrevFinals",
        # "HVPrevLast16",
        # "HVPrevLast8",
        # "HVPrevPlayin",
        "HVPrevRound1",
        # "HVPrevRound2",
        # "HVPrevSemiFinals",
        # "LVPrevCumSumFinals",
        # "LVPrevCumSumLast16",
        # "LVPrevCumSumLast8",
        # "LVPrevCumSumPlayin",
        # "LVPrevCumSumRound1",
        # "LVPrevCumSumRound2",
        # "LVPrevCumSumSemiFinals",
        # "HVPrevCumSumFinals",
        # "HVPrevCumSumLast16",
        # "HVPrevCumSumLast8",
        # "HVPrevCumSumPlayin",
        # "HVPrevCumSumRound1",
        # "HVPrevCumSumRound2",
        # "HVPrevCumSumSemiFinals",
    ]
]


all_results_dict = {}
for season in testing.Season.unique()[1:]:
    print(f'--------------------{season}---------------------')

    if season == 2020:
        continue

    single_season = testing[(testing["Season"] <= season)] # & (testing['Season'] >= (season - 4))]

    training_data = single_season[single_season['Season'] <= season].drop(columns='HVWin')
    training_data = training_data[~((training_data['Season'] == season) & (training_data['CRType'] == 'NCAA'))]
    
    testing_data = single_season[(single_season['CRType'] == 'NCAA')&(single_season['Season'] == season)]
    
    print('Testing Data:', testing_data.shape)
    print('Training Data:', training_data.shape)
    if season == 2025:
        real_testing_data = submission_file[['ID', 'Season', 'LVTeamID', 'HVTeamID', 'CRType']]
    else:
        real_testing_data = testing_data[['Season', 'LVTeamID', 'HVTeamID', 'LVWin', 'CRType']]

    lv_training_merge = training_data.groupby(by=['LVTeamID']).tail(1).drop(columns=['HVTeamID', 'LVWin', 'HVCity', 'HVState', 'HVTeamName',
                                                                                     'HVFirstD1Season', 'HVCoach', 'HVSeed', 'HVSeedRegion',
                                                                                     # 'HVPrevFinals', 'HVPrevLast16', 'HVPrevLast8', 'HVPrevPlayin',
                                                                                     'HVPrevRound1', # 'HVPrevRound2', 'HVPrevSemiFinals'
                                                                                     'CRType', 'DayNum', 'Gender', 'HVScore_R5S', 'HVScore_R10S', 'HVScore_R15S',
                                                                                     #  "HVPrevCumSumFinals", "HVPrevCumSumLast16","HVPrevCumSumLast8","HVPrevCumSumPlayin",
                                                                                     #  "HVPrevCumSumRound1","HVPrevCumSumRound2","HVPrevCumSumSemiFinals",
                                                                                     # 'HVScorePerSeasonMeanPrevSeason', # 'LVScoreAgainstHV_R10S'
                                                                                    ])

    hv_training_merge = training_data.groupby(by=['HVTeamID']).tail(1).drop(columns=['LVTeamID', 'LVWin', 'LVCity', 'LVState', 'LVTeamName',
                                                                                     'LVFirstD1Season', 'LVCoach', 'LVSeed', 'LVSeedRegion',
                                                                                     # 'LVPrevFinals', 'LVPrevLast16', 'LVPrevLast8', 'LVPrevPlayin',
                                                                                     'LVPrevRound1', # 'LVPrevRound2', 'LVPrevSemiFinals',
                                                                                     'CRType','DayNum', 'Gender', 'LVScore_R5S', 'LVScore_R10S', 'LVScore_R15S',
                                                                                     # "LVPrevCumSumFinals",  "LVPrevCumSumLast16", "LVPrevCumSumLast8", "LVPrevCumSumPlayin",
                                                                                     #  "LVPrevCumSumRound1", "LVPrevCumSumRound2", "LVPrevCumSumSemiFinals",
                                                                                     # 'LVScorePerSeasonMeanPrevSeason', #  'HVScoreAgainstLV_R10S'
                                                                                    ])

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
    lgb_model = lgb.LGBMClassifier(**{'verbose':-1, 'linear_tree':True})
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict_proba(X_val)[:, 1]

    # XGBoost Model
    xgb_model = xgb.XGBClassifier(**{'eval_metric':'logloss', 'enable_categorical':True, 'gamma':3, 'reg_lambda': 0.05})
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))

    axes[0].barh(features, lgb_model.feature_importances_)
    axes[0].set_title('LightGBM Feature Importance')

    axes[1].barh(features, xgb_model.feature_importances_)
    axes[1].set_title('XGBoost Feature Importance')

    plt.tight_layout()
    plt.show()

    all_results_dict[season] = {'real_lgb_brier_score': real_lgb_brier_score, 'real_xgb_brier_score': real_xgb_brier_score, 'real_ensemble_brier_score': real_ensemble_brier_score}


real_testing_data[['ID', 'LGB_Pred', 'XGB_Pred']]

real_testing_data['Pred'] = (real_testing_data['LGB_Pred'] + real_testing_data['XGB_Pred']) / 2
# real_testing_data['Pred'] = real_testing_data['XGB_Pred']
real_testing_data[['ID', 'Pred']].to_csv('submission.csv', index=False)

submission_metric_data = pd.DataFrame(all_results_dict)


submission_metric_data = pd.DataFrame(all_results_dict)
submission_metric_data = submission_metric_data.T.reset_index(names=['Season'])

final_results = submission_metric_data[submission_metric_data['Season']>=2021].drop(columns=['Season'])


final_results.mean(), final_results.median(), final_results.std()


'''
EXPERIMENT 1:
            - 0.187270
            - 0.204689
            - 0.188157
  
EXPERIMENT 2:
            - 0.192026
            - 0.203805
            - 0.191522

EXPERIMENT 3:
            - 0.185000
            - 0.231366
            - 0.200471

EXPERIMENT 4:
            - 0.188162
            - 0.216139
            - 0.194134

EXPERIMENT 5:
            - 0.187598
            - 0.220176
            - 0.196973

EXPERIMENT 6:
            - 0.193900
            - 0.223097
            - 0.198163

EXPERIMENT 7:
            - 0.189005
            - 0.194639
            - 0.189971

EXPERIMENT 8:
            - 0.192505
            - 0.194639
            - 0.191545

EXPERIMENT 9:
            - 0.191438
            - 0.196068
            - 0.191782

EXPERIMENT 10:
            - 0.189169
            - 0.208013
            - 0.195998

EXPERIMENT 11:
            - 0.189632
            - 0.189692
            - 0.186606

EXPERIMENT 12:
            - 0.192263
            - 0.198007
            - 0.193821

EXPERIMENT 13:
            - 0.192263
            - 0.187615
            - 0.188678





'''

