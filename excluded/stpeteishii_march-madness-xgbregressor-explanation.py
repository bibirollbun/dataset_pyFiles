import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle, Arc
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
from sklearn.model_selection import GroupKFold


DATA_PATH = "../input/march-machine-learning-mania-2024/"


df_seeds = pd.concat([
    pd.read_csv(DATA_PATH + "MNCAATourneySeeds.csv").assign(League = "M"),
    pd.read_csv(DATA_PATH + "WNCAATourneySeeds.csv").assign(League = "W"),
]).reset_index(drop=True)

df_season_results = pd.concat([
    pd.read_csv(DATA_PATH + "MRegularSeasonDetailedResults.csv").assign(League = "M"),
    pd.read_csv(DATA_PATH + "WRegularSeasonDetailedResults.csv").assign(League = "W"),
]).reset_index(drop=True)

df_tourney_results = pd.concat([
    pd.read_csv(DATA_PATH + "MNCAATourneyDetailedResults.csv").assign(League = "M"),
    pd.read_csv(DATA_PATH + "WNCAATourneyDetailedResults.csv").assign(League = "W"),
]).reset_index(drop=True)


def calculate_efficiency(df, prefix):
    points = 2 * (df[f'{prefix}FGM'] - df[f'{prefix}FGM3']) + df[f'{prefix}FTM'] + 3 * df[f'{prefix}FGM3']
    rebounds = df[f'{prefix}OR'] + df[f'{prefix}DR']
    missed_2pointers = df[f'{prefix}FGA'] - df[f'{prefix}FGA3'] - (df[f'{prefix}FGM'] - df[f'{prefix}FGM3'])
    missed_3pointers = df[f'{prefix}FGA3'] - df[f'{prefix}FGM3']
    missed_field_goals = missed_2pointers + missed_3pointers
    missed_free_throws = df[f'{prefix}FTA'] - df[f'{prefix}FTM']

    efficiency = (points + rebounds + df[f'{prefix}Ast'] +
                  df[f'{prefix}Stl'] + df[f'{prefix}Blk'] -
                  (missed_field_goals + missed_free_throws + df[f'{prefix}PF']))

    return efficiency


df_season_results['WEff'] = calculate_efficiency(df_season_results,'W')
df_season_results['LEff'] = calculate_efficiency(df_season_results,'L')
df_tourney_results['WEff'] = calculate_efficiency(df_tourney_results,'W')
df_tourney_results['LEff'] = calculate_efficiency(df_tourney_results,'L')

df_team_season_results = pd.concat([
    df_season_results[["Season", "League", "WTeamID","DayNum", "WScore", "LScore","WEff","LEff"]]
    .assign(GameResult="W")
    .rename(
        columns={"WTeamID":"TeamID","WScore": "TeamScore", "LScore": "OppScore","WEff":"TeamEff","LEff":"OppEff"}
    ),
    df_season_results[["Season", "League", "LTeamID", "DayNum", "WScore", "LScore","WEff","LEff"]]
    .assign(GameResult = "L")
    .rename(
        columns = {"LTeamID": "TeamID", "LScore": "TeamScore", "WScore": "OppScore","LEff":"TeamEff","WEff":"OppEff"}
    ),
]).reset_index(drop = True)

display(df_team_season_results)


K = 20
HOME_ADVANTAGE = 100

df_season_results_old = pd.concat([
    pd.read_csv(DATA_PATH + "MRegularSeasonCompactResults.csv").assign(League = "M"),
    pd.read_csv(DATA_PATH + "WRegularSeasonCompactResults.csv").assign(League = "W"),
]).reset_index(drop=True)
team_ids = set(df_season_results_old.WTeamID).union(df_season_results_old.LTeamID)
len(team_ids)
elo_dict = dict(zip(list(team_ids), [1500] * len(team_ids)))

df_season_results_old['margin'] = df_season_results_old.WScore - df_season_results_old.LScore


def elo_pred(elo1, elo2):
    return(1. / (10. ** (-(elo1 - elo2) / 400.) + 1.))

def expected_margin(elo_diff):
    return((7.5 + 0.006 * elo_diff))

def elo_update(w_elo, l_elo, margin):
    elo_diff = w_elo - l_elo
    pred = elo_pred(w_elo, l_elo)
    mult = ((margin + 3.) ** 0.8) / expected_margin(elo_diff)
    update = K * mult * (1 - pred)
    return(pred, update)


preds = []
w_elo = []
l_elo = []

for row in df_season_results_old.itertuples():

    w = row.WTeamID
    l = row.LTeamID
    margin = row.margin
    wloc = row.WLoc
    
    # Does either team get a home-court advantage?
    w_ad, l_ad, = 0., 0.
    if wloc == "H":
        w_ad += HOME_ADVANTAGE
    elif wloc == "A":
        l_ad += HOME_ADVANTAGE
    
    # Get elo updates as a result of the game
    pred, update = elo_update(elo_dict[w] + w_ad,
                              elo_dict[l] + l_ad, 
                              margin)
    elo_dict[w] += update
    elo_dict[l] -= update
    preds.append(pred)
    w_elo.append(elo_dict[w])
    l_elo.append(elo_dict[l])

df_season_results_old['w_elo'] = w_elo
df_season_results_old['l_elo'] = l_elo


def final_elo_per_season(df, team_id):
    d = df.copy()
    d = d.loc[(d.WTeamID == team_id) | (d.LTeamID == team_id), :]
    d.sort_values(['Season', 'DayNum'], inplace=True)
    d.drop_duplicates(['Season'], keep='last', inplace=True)
    w_mask = d.WTeamID == team_id
    l_mask = d.LTeamID == team_id
    d['season_elo'] = None
    d.loc[w_mask, 'season_elo'] = d.loc[w_mask, 'w_elo']
    d.loc[l_mask, 'season_elo'] = d.loc[l_mask, 'l_elo']
    out = pd.DataFrame({
        'team_id': team_id,
        'season': d.Season,
        'season_elo': d.season_elo
    })
    return(out)
    
df_list = [final_elo_per_season(df_season_results_old, id) for id in team_ids]
season_elos = pd.concat(df_list)


df_team_season_results["ScoreDiff"] = (
    df_team_season_results["TeamScore"]-df_team_season_results["OppScore"]
)
df_team_season_results["Win"] = (df_team_season_results["GameResult"] == "W").astype("int")

df_team_season_results["EffDiff"] = (df_team_season_results["TeamEff"]-df_team_season_results["OppEff"])



team_season_agg = (
    df_team_season_results.groupby(["Season","TeamID","League"])
    .agg(
        AVGScoreDiff = ("ScoreDiff", "mean"),
        MedianScoreDiff = ("ScoreDiff", "median"),
        MinScoreDiff = ("ScoreDiff","min"),
        MaxScoreDiff = ("ScoreDiff","max"),
        MedianEffDiff = ("EffDiff","median"),
        Wins = ("Win","sum"),
        Losses = ("Win", lambda x: (x==0).sum()),
        WinPercentage = ("Win","mean"),
    ).reset_index()
)


df_seeds["ChalkSeed"] = (
    df_seeds["Seed"].str.replace("a","").str.replace("b", "").str[1:].astype("int")
)

team_season_agg = team_season_agg.merge(
    df_seeds, on = ["Season", "TeamID", "League"], how ="left"
)

#missing values: teams didn't make it to the tournament


df_team_tourney_results = pd.concat([
    df_tourney_results[
        ['Season','DayNum','WTeamID','WScore','LTeamID','LScore','League']
    ]
    .assign(GameResult = "W")
.rename(
    columns = {
        "WTeamID": "TeamID",
        "LTeamID": "OppTeamID",
        "WScore": "TeamScore",
        "LScore": "OppScore",
    }
),
df_tourney_results[
    ['Season','DayNum','WTeamID','WScore','LTeamID','LScore','League']
]
.assign(GameResult = "L")
.rename(
    columns ={
        "LTeamID": "TeamID",
        "WTeamID": "OppTeamID",
        "LScore": "TeamScore",
        "WScore": "OppScore",
    }
),
]
).reset_index(drop = True)

df_team_tourney_results["Win"] = (df_team_tourney_results["GameResult"] == "W").astype("int")

df_team_tourney_results = df_team_tourney_results.drop("GameResult",axis=1)

team_season_agg = team_season_agg.merge(season_elos,left_on =["TeamID","Season"],right_on = ["team_id","season"])

#"MedianFGMDiff","MedianFGADiff","MedianFGM3Diff","MedianFGA3Diff","MedianFTMDiff","MedianFTADiff","MedianORDiff","MedianDRDiff","MedianAstDiff","MedianTODiff","MedianStlDiff","MedianBlkDiff" ,"MedianPFDiff"  

display(team_season_agg)

# Add the tournament results for each season to the aggregated Features


df_historic_tourney_features = df_team_tourney_results.merge(
    team_season_agg[
        ["Season","League","TeamID","WinPercentage","MedianScoreDiff","ChalkSeed",'MedianEffDiff','season_elo']
    ],
    on=["Season","League","TeamID"],
    how="left",
).merge(
    team_season_agg[
        ["Season","League","TeamID","WinPercentage","MedianScoreDiff","ChalkSeed",'MedianEffDiff','season_elo']
    ].rename(
    columns = {
        "TeamID":"OppTeamID",
        "WinPercentage": "OppWinPercentage",
        "MedianScoreDiff": "OppMedianScoreDiff",
        "ChalkSeed": "OppChalkSeed",
        'MedianEffDiff':'OppMedianEffDiff',
        'season_elo':'Oppseason_elo'
    }),
    on=["Season","League","OppTeamID"]
)

df_historic_tourney_features["WinPctDiff"] = (
    df_historic_tourney_features["WinPercentage"]-df_historic_tourney_features["OppWinPercentage"])

df_historic_tourney_features["ChalkSeedDiff"]=(
    df_historic_tourney_features["ChalkSeed"]-df_historic_tourney_features["OppChalkSeed"])

df_historic_tourney_features["MedianScoreDiffDiff"] = (
    df_historic_tourney_features["MedianScoreDiff"]-df_historic_tourney_features["OppMedianScoreDiff"])

df_historic_tourney_features["seasonEloDiff"] = pd.to_numeric(
    df_historic_tourney_features["season_elo"]-df_historic_tourney_features["Oppseason_elo"])

df_historic_tourney_features["MedianEffDiff"] = pd.to_numeric(
    df_historic_tourney_features["MedianEffDiff"]-df_historic_tourney_features["OppMedianEffDiff"])

display(df_historic_tourney_features)


FEATURES = [
    "WinPctDiff",
    "ChalkSeedDiff",
    "seasonEloDiff",
    "MedianEffDiff",
]
TARGET = "Win"

X = df_historic_tourney_features[FEATURES]
y = df_historic_tourney_features[TARGET]
groups = df_historic_tourney_features["Season"]
seasons = df_historic_tourney_features["Season"].unique()

gkf = GroupKFold(n_splits=df_historic_tourney_features["Season"].nunique())
cv_results = []
models = []
season_idx = 0


for train_index, test_index in gkf.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = xgb.XGBRegressor(
        eval_metric="logloss",
        n_estimators=100,
        learning_rate=0.02,
        subsample = 0.35,
        colsample_bytree = 0.7,
        num_parallel_tree = 10,
        min_child_weight = 40,
        gamma = 10,
        max_depth = 3,
    )
    holdout_season = seasons[season_idx]
    print(f"Holdout Season: {holdout_season}")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    y_pred = model.predict(X_test)
    score_ll = log_loss(y_test, y_pred)
    y_pred = y_pred > 0.5
    
    accuracy = accuracy_score(y_test, y_pred)
    cv_results.append(accuracy)
    season_idx += 1
    print(f"Season {holdout_season}: {accuracy} {score_ll}")
    models.append(model)

print("Average CV Accuracy:", np.mean(cv_results))


TEST_SEASON = 2024  # Change to 2024 when it comes out!

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

display(team_season_agg)

tourney_pairs = (
    tourney_pairs.merge(
        team_season_agg[
            ["Season", "League", "TeamID", "WinPercentage", "MedianScoreDiff",'MedianEffDiff','season_elo']
        ],
        on=["Season", "League", "TeamID"],
        how="left",
    )
    .merge(
        team_season_agg[
            ["Season", "League", "TeamID", "WinPercentage", "MedianScoreDiff",'MedianEffDiff','season_elo']
        ].rename(
            columns={
                "TeamID": "TeamIDOpp",
                "WinPercentage": "OppWinPercentage",
                "MedianScoreDiff": "OppMedianScoreDiff",
                "MedianEffDiff":"OppMedianEffDiff",
                "season_elo":"Oppseason_elo"
            }
        ),
        on=["Season", "League", "TeamIDOpp"],
    )
    .reset_index(drop=True)
)

display(tourney_pairs)

tourney_pairs["OppChalkSeed"] = (
    tourney_pairs["SeedOpp"]
    .str.replace("a", "")
    .str.replace("b", "")
    .str[1:]
    .astype("int")
)

display(season_elos)

tourney_pairs = tourney_pairs.merge(
    season_elos,
    right_on=["season", "team_id"],left_on=["Season","TeamID"],
    how="left",
)

tourney_pairs = tourney_pairs.merge(
    season_elos.rename(
        columns={"team_id": "TeamIDOpp"}
    ),
    right_on=["season","TeamIDOpp"],left_on=["Season","TeamID"],
    how="left",
    suffixes=("", "Opp"),
)

display(tourney_pairs)

tourney_pairs["seasonEloDiff"] = pd.to_numeric(
    tourney_pairs["season_elo"] - tourney_pairs["Oppseason_elo"]
)


tourney_pairs["WinPctDiff"] = pd.to_numeric(
    tourney_pairs["WinPercentage"] - tourney_pairs["OppWinPercentage"]
)

tourney_pairs["ChalkSeedDiff"] = pd.to_numeric(
    tourney_pairs["ChalkSeed"] - tourney_pairs["ChalkSeedOpp"]
)

tourney_pairs["MedianScoreDiffDiff"] = pd.to_numeric(
    tourney_pairs["MedianScoreDiff"] - tourney_pairs["OppMedianScoreDiff"])

tourney_pairs['MedianEffDiff'] = tourney_pairs["MedianEffDiff"] - tourney_pairs["OppMedianEffDiff"]


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

import numpy as np 
import pandas as pd 
from tqdm import tqdm

# Load and filter data
round_slots = pd.read_csv('/kaggle/input/march-machine-learning-mania-2024/MNCAATourneySlots.csv')
round_slots = round_slots[round_slots['Season'] == 2023]
round_slots = round_slots[round_slots['Slot'].str.contains('R')] # Filter out First Four

seeds = pd.read_csv('/kaggle/input/march-machine-learning-mania-2024/2024_tourney_seeds.csv')
seeds_m = seeds[seeds['Tournament'] == 'M']
seeds_w = seeds[seeds['Tournament'] == 'W']

preds['ID'] = preds['ID'].str.split('_')


def prepare_data(seeds, preds):
    # Function preparing the data for the simulation
    seed_dict = seeds.set_index('Seed')['TeamID'].to_dict()
    inverted_seed_dict = {value: key for key, value in seed_dict.items()}
    probas_dict = {}
    
    for teams, proba in zip(preds['ID'], preds['Pred']):
        team1, team2 = teams[1], teams[2]

        probas_dict.setdefault(team1, {})[team2] = proba
        probas_dict.setdefault(team2, {})[team1] = 1 - proba

    return seed_dict, inverted_seed_dict, probas_dict

def simulate(round_slots, seeds, inverted_seeds, probas, random_values, sim=True):
    winners = []
    slots = []

    for slot, strong, weak, random_val in zip(round_slots.Slot, round_slots.StrongSeed, round_slots.WeakSeed, random_values):
        team1, team2 = seeds[strong], seeds[weak]

        # Get the probability of team_1 winning
        proba = probas[str(team1)][str(team2)]
            
        if sim:
            # Randomly determine the winner based on the probability
            winner = team1 if random_val < proba else team2
        else:
            # Determine the winner based on the higher probability
            winner = [team1, team2][np.argmax([proba, 1-proba])]
            
        # Append the winner and corresponding slot to the lists
        winners.append(winner)
        slots.append(slot)

        seeds[slot] = winner

    # Convert winners to original seeds using the inverted_seeds dictionary
    return [inverted_seeds[w] for w in winners], slots



def run_simulation(brackets=1, seeds=None, preds=None, round_slots=None, sim=True):

    # Get relevant data for the simulation
    seed_dict, inverted_seed_dict, probas_dict = prepare_data(seeds, preds)
    # Lists to store simulation results
    results = []
    bracket = []
    slots = []
    
    random_values = np.random.random(size=(brackets, len(round_slots)))

    # Iterate through the specified number of brackets
    for b in tqdm(range(1, brackets+1)):
        # Run single simulation
        r, s = simulate(round_slots, seed_dict, inverted_seed_dict, probas_dict, random_values[b-1], sim)
        
        # Update results
        results.extend(r)
        bracket.extend([b] * len(r))
        slots.extend(s)

    # Create final DataFrame
    result_df = pd.DataFrame({'Bracket': bracket, 'Slot': slots, 'Team': results})

    return result_df


n_brackets = 100000
result_m = run_simulation(
    brackets=n_brackets, seeds=seeds_m, preds=preds, round_slots=round_slots, sim=True
)
result_m["Tournament"] = "M"
result_w = run_simulation(
    brackets=n_brackets, seeds=seeds_w, preds=preds, round_slots=round_slots, sim=True
)
result_w["Tournament"] = "W"
submission = pd.concat([result_m, result_w])
submission = submission.reset_index(drop=True)
submission.index.names = ["RowId"]
submission = submission.reset_index()
display(submission)

ss = pd.read_csv(DATA_PATH + "sample_submission.csv")
submission[ss.columns] = submission[ss.columns]
submission[ss.columns].to_csv("submission.csv", index=False)
display(submission[ss.columns])




