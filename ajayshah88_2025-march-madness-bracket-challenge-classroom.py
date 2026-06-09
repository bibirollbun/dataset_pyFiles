# Remove all conda packages
!find /opt/conda \( -name "cudf*" -o -name "libcudf*" -o -name "cuml*" -o -name "libcuml*" \
                   -o -name "cugraph*" -o -name "libcugraph*" -o -name "raft*" -o -name "libraft*" \
                   -o -name "pylibraft*" -o -name "libkvikio*" -o -name "*dask*" -o -name "rmm*"\
                   -o -name "librmm*" \) -exec rm -rf {} \; 2>/dev/null

# pip uninstall, just incase there are packages lying around
!pip uninstall -q cudf cuml dask xgboost dask-cudf cuml cugraph cupy cupy-cuda12x --y


!pip install \
    -q \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.2.* \
    dask-cudf-cu12==24.2.* \
    cuml-cu12==24.2.* \
    cugraph-cu12==24.2.*


!pip install -q jupyter-black
!pip install -q --upgrade xgboost[dask]





import cudf  # this should work without any errors

%load_ext cudf.pandas


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle, Arc
import seaborn as sns

from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
from sklearn.model_selection import GroupKFold

mypal = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # Grab the color pal


!ls -GFlash ../input/march-machine-learning-mania-2024/


DATA_PATH = "../input/march-machine-learning-mania-2024/"


df_seeds = pd.concat(
    [
        pd.read_csv(DATA_PATH + "MNCAATourneySeeds.csv").assign(League="M"),
        pd.read_csv(DATA_PATH + "WNCAATourneySeeds.csv").assign(League="W"),
    ],
).reset_index(drop=True)

df_season_results = pd.concat(
    [
        pd.read_csv(DATA_PATH + "MRegularSeasonCompactResults.csv").assign(League="M"),
        pd.read_csv(DATA_PATH + "WRegularSeasonCompactResults.csv").assign(League="W"),
    ]
).reset_index(drop=True)

df_tourney_results = pd.concat(
    [
        pd.read_csv(DATA_PATH + "MNCAATourneyCompactResults.csv").assign(League="M"),
        pd.read_csv(DATA_PATH + "WNCAATourneyCompactResults.csv").assign(League="W"),
    ]
).reset_index(drop=True)


df_team_season_results = pd.concat(
    [
        df_season_results[["Season", "League", "WTeamID", "DayNum", "WScore", "LScore"]]
        .assign(GameResult="W")
        .rename(
            columns={"WTeamID": "TeamID", "WScore": "TeamScore", "LScore": "OppScore"}
        ),
        df_season_results[["Season", "League", "LTeamID", "DayNum", "WScore", "LScore"]]
        .assign(GameResult="L")
        .rename(
            columns={"LTeamID": "TeamID", "LScore": "TeamScore", "WScore": "OppScore"}
        ),
    ]
).reset_index(drop=True)


# Score Differential
df_team_season_results["ScoreDiff"] = (
    df_team_season_results["TeamScore"] - df_team_season_results["OppScore"]
)
df_team_season_results["Win"] = (df_team_season_results["GameResult"] == "W").astype(
    "int"
)


df_team_season_results.sample(10, random_state=529)


# Aggregate the data
team_season_agg = (
    df_team_season_results.groupby(["Season", "TeamID", "League"])
    .agg(
        AvgScoreDiff=("ScoreDiff", "mean"),
        MedianScoreDiff=("ScoreDiff", "median"),
        MinScoreDiff=("ScoreDiff", "min"),
        MaxScoreDiff=("ScoreDiff", "max"),
        Wins=("Win", "sum"),
        Losses=("GameResult", lambda x: (x == "L").sum()),
        WinPercentage=("Win", "mean"),
    )
    .reset_index()
)


team_season_agg.head()


df_seeds["ChalkSeed"] = (
    df_seeds["Seed"].str.replace("a", "").str.replace("b", "").str[1:].astype("int")
)

team_season_agg = team_season_agg.merge(
    df_seeds, on=["Season", "TeamID", "League"], how="left"
)


team_season_agg.shape, df_seeds.shape


df_team_tourney_results = pd.concat(
    [
        df_tourney_results[
            ["Season", "League", "WTeamID", "LTeamID", "WScore", "LScore"]
        ]
        .assign(GameResult="W")
        .rename(
            columns={
                "WTeamID": "TeamID",
                "LTeamID": "OppTeamID",
                "WScore": "TeamScore",
                "LScore": "OppScore",
            }
        ),
        df_tourney_results[
            ["Season", "League", "LTeamID", "WTeamID", "LScore", "WScore"]
        ]
        .assign(GameResult="L")
        .rename(
            columns={
                "LTeamID": "TeamID",
                "WTeamID": "OppTeamID",
                "LScore": "TeamScore",
                "WScore": "OppScore",
            }
        ),
    ]
).reset_index(drop=True)

df_team_tourney_results["Win"] = (df_team_tourney_results["GameResult"] == "W").astype(
    "int"
)


df_team_tourney_results.head()


df_historic_tourney_features = df_team_tourney_results.merge(
    team_season_agg[
        ["Season", "League", "TeamID", "WinPercentage", "MedianScoreDiff", "ChalkSeed"]
    ],
    on=["Season", "League", "TeamID"],
    how="left",
).merge(
    team_season_agg[
        ["Season", "League", "TeamID", "WinPercentage", "MedianScoreDiff", "ChalkSeed"]
    ].rename(
        columns={
            "TeamID": "OppTeamID",
            "WinPercentage": "OppWinPercentage",
            "MedianScoreDiff": "OppMedianScoreDiff",
            "ChalkSeed": "OppChalkSeed",
        }
    ),
    on=["Season", "League", "OppTeamID"],
)


df_historic_tourney_features.head()


df_historic_tourney_features["WinPctDiff"] = (
    df_historic_tourney_features["WinPercentage"]
    - df_historic_tourney_features["OppWinPercentage"]
)

df_historic_tourney_features["ChalkSeedDiff"] = (
    df_historic_tourney_features["ChalkSeed"]
    - df_historic_tourney_features["OppChalkSeed"]
)

df_historic_tourney_features["MedianScoreDiffDiff"] = (
    df_historic_tourney_features["MedianScoreDiff"]
    - df_historic_tourney_features["OppMedianScoreDiff"]
)


df_historic_tourney_features.columns


df_historic_tourney_features.sample(5, random_state=529)


df_historic_tourney_features[
    ["Season", "TeamID"]
].head()


df_historic_tourney_features["BaselinePred"] = (
    df_historic_tourney_features["ChalkSeed"]
    < df_historic_tourney_features["OppChalkSeed"]
)

df_historic_tourney_features.loc[
    df_historic_tourney_features["ChalkSeed"]
    == df_historic_tourney_features["OppChalkSeed"],
    "BaselinePred",
] = (
    df_historic_tourney_features["WinPercentage"]
    > df_historic_tourney_features["OppWinPercentage"]
)


cv_scores_baseline = []
for season in df_historic_tourney_features["Season"].unique():
    pred = df_historic_tourney_features.query("Season == @season")[
        "BaselinePred"
    ].astype("int")
    y = df_historic_tourney_features.query("Season == @season")["Win"]
    score = accuracy_score(y, pred)
    score_ll = log_loss(y, pred)
    cv_scores_baseline.append(score)
    print(f"Holdout season {season} - Accuracy {score:0.4f} Log Loss {score_ll:0.4f}")

print(f"Baseline accuracy {np.mean(cv_scores_baseline):0.4f}")


FEATURES = [
    #     "WinPercentage",
    #     "MedianScoreDiff",
    #     "ChalkSeed",
    #     "OppWinPercentage",
    #     "OppMedianScoreDiff",
    #     "OppChalkSeed",
    "WinPctDiff",
    "ChalkSeedDiff"
]
TARGET = "Win"


X = df_historic_tourney_features[FEATURES]
y = df_historic_tourney_features[TARGET]
groups = df_historic_tourney_features["Season"]
seasons = df_historic_tourney_features["Season"].unique()

# Setup cross-validation
gkf = GroupKFold(n_splits=df_historic_tourney_features["Season"].nunique())
cv_results = []
models = []

season_idx = 0
for train_index, test_index in gkf.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Prepare the model
    model = xgb.XGBRegressor(
        eval_metric="logloss",
        n_estimators=1_000,
        learning_rate=0.001,
    )
    holdout_season = seasons[season_idx]
    print(f"Holdout Season: {holdout_season}")
    # Train the model
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    # Predict on the test set
    y_pred = model.predict(X_test)
    score_ll = log_loss(y_test, y_pred)
    y_pred = y_pred > 0.5
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cv_results.append(accuracy)
    season_idx += 1
    print(f"Season {holdout_season}: {accuracy} {score_ll}")
    models.append(model)
# Print the average accuracy across all folds
print("Average CV Accuracy:", np.mean(cv_results))


TEST_SEASON = 2024  # Change to 2025 when it comes out!

# Need to change file once data becomes available
seeds_2024 = pd.read_csv(DATA_PATH + "2024_tourney_seeds.csv")

seeds_2024["ChalkSeed"] = (
    seeds_2024["Seed"].str.replace("a", "").str.replace("b", "").str[1:].astype("int")
)


tourney_pairs = (
    seeds_2024.merge(seeds_2024, on=["Tournament"], suffixes=("", "Opp"))
    .assign(Season=TEST_SEASON)
    .query("TeamID != TeamIDOpp")
    .rename(columns={"Tournament": "League"})
)

tourney_pairs = (
    tourney_pairs.merge(
        team_season_agg[
            ["Season", "League", "TeamID", "WinPercentage", "MedianScoreDiff"]
        ],
        on=["Season", "League", "TeamID"],
        how="left",
    )
    .merge(
        team_season_agg[
            ["Season", "League", "TeamID", "WinPercentage", "MedianScoreDiff"]
        ].rename(
            columns={
                "TeamID": "TeamIDOpp",
                "WinPercentage": "OppWinPercentage",
                "MedianScoreDiff": "OppMedianScoreDiff",
            }
        ),
        on=["Season", "League", "TeamIDOpp"],
    )
    .reset_index(drop=True)
)

tourney_pairs["OppChalkSeed"] = (
    tourney_pairs["SeedOpp"]
    .str.replace("a", "")
    .str.replace("b", "")
    .str[1:]
    .astype("int")
)


tourney_pairs = tourney_pairs.merge(
    fivethiryeight_scores.drop("TeamName", axis=1),
    on=["Season", "League", "TeamID"],
    how="left",
)

tourney_pairs = tourney_pairs.merge(
    fivethiryeight_scores.drop("TeamName", axis=1).rename(
        columns={"TeamID": "TeamIDOpp"}
    ),
    on=["Season", "League", "TeamIDOpp"],
    how="left",
    suffixes=("", "Opp"),
)
'''
# Diff features
tourney_pairs["538rating_diff"] = (
    tourney_pairs["538rating"] - tourney_pairs["538ratingOpp"]
)
'''
tourney_pairs["BaselinePred"] = (
    tourney_pairs["ChalkSeed"] < tourney_pairs["OppChalkSeed"]
)

tourney_pairs.loc[
    tourney_pairs["ChalkSeed"] == tourney_pairs["OppChalkSeed"],
    "BaselinePred",
] = (
    tourney_pairs["WinPercentage"] > tourney_pairs["OppWinPercentage"]
)

tourney_pairs["WinPctDiff"] = (
    tourney_pairs["WinPercentage"] - tourney_pairs["OppWinPercentage"]
)

tourney_pairs["ChalkSeedDiff"] = (
    tourney_pairs["ChalkSeed"] - tourney_pairs["OppChalkSeed"]
)

tourney_pairs["MedianScoreDiffDiff"] = (
    tourney_pairs["MedianScoreDiff"] - tourney_pairs["OppMedianScoreDiff"]
)


tourney_pairs.head(100)


for i, model in enumerate(models):
    tourney_pairs[f"pred_model{i}"] = model.predict(tourney_pairs[FEATURES])


tourney_pairs["Pred"] = tourney_pairs[
    [f for f in tourney_pairs.columns if "model" in f]
].mean(axis=1)

tourney_pairs["ID"] = (
    tourney_pairs["Season"].astype("str")
    + "_"
    + tourney_pairs["TeamID"].astype("str")
    + "_"
    + tourney_pairs["TeamIDOpp"].astype("str")
)

preds = tourney_pairs.copy()


from tqdm import tqdm

# Load and filter data
# Check info again here
round_slots = pd.read_csv(
    "/kaggle/input/march-machine-learning-mania-2024/MNCAATourneySlots.csv"
)
round_slots = round_slots[round_slots["Season"] == 2024]
round_slots = round_slots[
    round_slots["Slot"].str.contains("R")
]  # Filter out First Four

seeds = pd.read_csv(
    "/kaggle/input/march-machine-learning-mania-2024/2024_tourney_seeds.csv"
)
seeds_m = seeds[seeds["Tournament"] == "M"]
seeds_w = seeds[seeds["Tournament"] == "W"]

preds["ID"] = preds["ID"].str.split("_")


def prepare_data(seeds, preds):
    # Function preparing the data for the simulation
    seed_dict = seeds.set_index("Seed")["TeamID"].to_dict()
    inverted_seed_dict = {value: key for key, value in seed_dict.items()}
    probas_dict = {}

    for teams, proba in zip(preds["ID"], preds["Pred"]):
        team1, team2 = teams[1], teams[2]

        probas_dict.setdefault(team1, {})[team2] = proba
        probas_dict.setdefault(team2, {})[team1] = 1 - proba

    return seed_dict, inverted_seed_dict, probas_dict


def simulate(round_slots, seeds, inverted_seeds, probas, sim=True):
    """
    Simulates each round of the tournament.

    Parameters:
    - round_slots: DataFrame containing information on who is playing in each round.
    - seeds (dict): Dictionary mapping seed values to team IDs.
    - inverted_seeds (dict): Dictionary mapping team IDs to seed values.
    - probas (dict): Dictionary containing matchup probabilities.
    - sim (boolean): Simulates match if True. Chooses team with higher probability as winner otherwise.

    Returns:
    - list: List with winning team IDs for each match.
    - list: List with corresponding slot names for each match.
    """
    winners = []
    slots = []

    for slot, strong, weak in zip(
        round_slots.Slot, round_slots.StrongSeed, round_slots.WeakSeed
    ):
        team_1, team_2 = seeds[strong], seeds[weak]

        # Get the probability of team_1 winning
        proba = probas[str(team_1)][str(team_2)]

        if sim:
            # Randomly determine the winner based on the probability
            winner = np.random.choice([team_1, team_2], p=[proba, 1 - proba])
        else:
            # Determine the winner based on the higher probability
            winner = [team_1, team_2][np.argmax([proba, 1 - proba])]

        # Append the winner and corresponding slot to the lists
        winners.append(winner)
        slots.append(slot)

        seeds[slot] = winner

    # Convert winners to original seeds using the inverted_seeds dictionary
    return [inverted_seeds[w] for w in winners], slots


def run_simulation(brackets=1, seeds=None, preds=None, round_slots=None, sim=True):
    """
    Runs a simulation of bracket tournaments.

    Parameters:
    - brackets (int): Number of brackets to simulate.
    - seeds (pd.DataFrame): DataFrame containing seed information.
    - preds (pd.DataFrame): DataFrame containing prediction information for each match-up.
    - round_slots (pd.DataFrame): DataFrame containing information about the tournament rounds.
    - sim (boolean): Simulates matches if True. Chooses team with higher probability as winner otherwise.

    Returns:
    - pd.DataFrame: DataFrame with simulation results.
    """
    # Get relevant data for the simulation
    seed_dict, inverted_seed_dict, probas_dict = prepare_data(seeds, preds)
    # Lists to store simulation results
    results = []
    bracket = []
    slots = []

    # Iterate through the specified number of brackets
    for b in tqdm(range(1, brackets + 1)):
        # Run single simulation
        r, s = simulate(round_slots, seed_dict, inverted_seed_dict, probas_dict, sim)

        # Update results
        results.extend(r)
        bracket.extend([b] * len(r))
        slots.extend(s)

    # Create final DataFrame
    result_df = pd.DataFrame({"Bracket": bracket, "Slot": slots, "Team": results})

    return result_df


n_brackets = 1
result_m = run_simulation(
    brackets=n_brackets, seeds=seeds_m, preds=preds, round_slots=round_slots, sim=False
)
result_m["Tournament"] = "M"
result_w = run_simulation(
    brackets=n_brackets, seeds=seeds_w, preds=preds, round_slots=round_slots, sim=False
)
result_w["Tournament"] = "W"
submission = pd.concat([result_m, result_w])
submission = submission.reset_index(drop=True)
submission.index.names = ["RowId"]
submission = submission.reset_index()


ss = pd.read_csv(DATA_PATH + "sample_submission.csv")
submission[ss.columns] = submission[ss.columns]
submission[ss.columns].to_csv("submission.csv", index=False)


submission_with_names = submission.rename(columns={"Team": "Seed"}).merge(
    seeds, on=["Seed", "Tournament"], how="left"
)

teams = pd.concat(
    [
        pd.read_csv(DATA_PATH + "MTeams.csv").assign(Tournament="M"),
        pd.read_csv(DATA_PATH + "WTeams.csv").assign(Tournament="W"),
    ]
)

submission_with_names = submission_with_names.merge(
    teams[["Tournament", "TeamID", "TeamName"]], how="left"
)


submission_with_names.to_csv("submission_with_names.csv")


submission_with_names

