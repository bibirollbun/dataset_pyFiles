###############################################
# Kaggle Notebook: NCAA Prediction with
# Detailed Results + Isotonic Calibration
###############################################
import os
import re
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss

###############################################
# 1. Load Data
###############################################
DATA_PATH = "/kaggle/input/march-machine-learning-mania-2025/"

# Seeds
df_seeds_m = pd.read_csv(os.path.join(DATA_PATH, "MNCAATourneySeeds.csv"))
df_seeds_w = pd.read_csv(os.path.join(DATA_PATH, "WNCAATourneySeeds.csv"))
df_seeds = pd.concat([df_seeds_m, df_seeds_w], ignore_index=True)

# Teams (optional, for reference)
df_teams_m = pd.read_csv(os.path.join(DATA_PATH, "MTeams.csv"))
df_teams_w = pd.read_csv(os.path.join(DATA_PATH, "WTeams.csv"))
df_teams = pd.concat([df_teams_m, df_teams_w], ignore_index=True)

# Regular Season Detailed Results
df_season_m = pd.read_csv(os.path.join(DATA_PATH, "MRegularSeasonDetailedResults.csv"))
df_season_w = pd.read_csv(os.path.join(DATA_PATH, "WRegularSeasonDetailedResults.csv"))
df_season = pd.concat([df_season_m, df_season_w], ignore_index=True)

# Tourney Detailed Results
df_tourney_m = pd.read_csv(os.path.join(DATA_PATH, "MNCAATourneyDetailedResults.csv"))
df_tourney_w = pd.read_csv(os.path.join(DATA_PATH, "WNCAATourneyDetailedResults.csv"))
df_tourney = pd.concat([df_tourney_m, df_tourney_w], ignore_index=True)

# Sample Submission for 2025
df_sub = pd.read_csv(os.path.join(DATA_PATH, "SampleSubmissionStage1.csv"))

# cleanup
del df_seeds_m, df_seeds_w, df_teams_m, df_teams_w
del df_season_m, df_season_w, df_tourney_m, df_tourney_w

###############################################
# 2. Helper Functions
###############################################
def treat_seed(seed_str):
    """Convert 'W01', 'X16a', etc. -> integer (1,16,...). If missing, return 100."""
    if not isinstance(seed_str, str):
        return 100
    digits = re.sub("[^0-9]", "", seed_str)  # keep only digits
    if digits == "":
        return 100
    return int(digits)

df_seeds["SeedValue"] = df_seeds["Seed"].apply(treat_seed)

def create_team_stats(df):
    """
    Convert Detailed Results into a long format of (Season, TeamID)'s stats:
    - PF, PA, FGM, FGA, etc. from the perspective of that TeamID
    """
    # Winners
    w_cols = [
        "Season","WTeamID","WScore","LScore",
        "WFGM","WFGA","WFGM3","WFGA3","WFTM","WFTA",
        "WOR","WDR","WAst","WTO","WStl","WBlk","WPF"
    ]
    winners = df[w_cols].copy()
    winners.rename(columns={
        "WTeamID":"TeamID","WScore":"PF","LScore":"PA",
        "WFGM":"FGM","WFGA":"FGA","WFGM3":"FGM3","WFGA3":"FGA3",
        "WFTM":"FTM","WFTA":"FTA","WOR":"OR","WDR":"DR","WAst":"Ast",
        "WTO":"TO","WStl":"Stl","WBlk":"Blk","WPF":"PFoul"
    }, inplace=True)
    winners["Result"] = 1

    # Losers
    l_cols = [
        "Season","LTeamID","WScore","LScore",
        "LFGM","LFGA","LFGM3","LFGA3","LFTM","LFTA",
        "LOR","LDR","LAst","LTO","LStl","LBlk","LPF"
    ]
    losers = df[l_cols].copy()
    losers.rename(columns={
        "LTeamID":"TeamID","LScore":"PF","WScore":"PA",
        "LFGM":"FGM","LFGA":"FGA","LFGM3":"FGM3","LFGA3":"FGA3",
        "LFTM":"FTM","LFTA":"FTA","LOR":"OR","LDR":"DR","LAst":"Ast",
        "LTO":"TO","LStl":"Stl","LBlk":"Blk","LPF":"PFoul"
    }, inplace=True)
    losers["Result"] = 0

    return pd.concat([winners, losers], axis=0, ignore_index=True)

df_season_long = create_team_stats(df_season)

# Aggregate by (Season, TeamID)
stats_agg = df_season_long.groupby(["Season","TeamID"]).agg({
    "PF":["mean","sum"],
    "PA":["mean","sum"],
    "FGM":["mean"],
    "FGA":["mean"],
    "FGM3":["mean"],
    "FGA3":["mean"],
    "FTM":["mean"],
    "FTA":["mean"],
    "OR":["mean"],
    "DR":["mean"],
    "Ast":["mean"],
    "TO":["mean"],
    "Stl":["mean"],
    "Blk":["mean"],
    "PFoul":["mean"],
    "Result":["mean","count"]
})
stats_agg.columns = ["_".join(x) for x in stats_agg.columns]  # flatten
stats_agg.reset_index(inplace=True)

# rename for clarity
stats_agg.rename(columns={
    "PF_mean":"MeanPF","PF_sum":"TotalPointsFor",
    "PA_mean":"MeanPA","PA_sum":"TotalPointsAgainst",
    "Result_mean":"WinRate","Result_count":"Games"
}, inplace=True)

# Additional derived stats
stats_agg["FGPct"] = stats_agg["FGM_mean"] / (stats_agg["FGA_mean"] + 1e-9)
stats_agg["FG3Pct"] = stats_agg["FGM3_mean"] / (stats_agg["FGA3_mean"] + 1e-9)
stats_agg["FTPct"] = stats_agg["FTM_mean"] / (stats_agg["FTA_mean"] + 1e-9)
stats_agg["MeanMargin"] = stats_agg["MeanPF"] - stats_agg["MeanPA"]

stats_agg.fillna(0, inplace=True)
df_team_stats = stats_agg.copy()

def add_binary_rows(df):
    """
    For a Detailed Tourney DF with WTeamID, LTeamID, etc.,
    create two rows per game:
      1) TeamA=winner, label=1
      2) TeamA=loser, label=0
    """
    rename_winner = {
        "WTeamID":"TeamA","WScore":"ScoreA",
        "LTeamID":"TeamB","LScore":"ScoreB"
    }
    rename_loser = {
        "WTeamID":"TeamB","WScore":"ScoreB",
        "LTeamID":"TeamA","LScore":"ScoreA"
    }

    df_win = df.rename(columns=rename_winner).copy()
    df_lose = df.rename(columns=rename_loser).copy()

    df_win["Label"] = 1
    df_lose["Label"] = 0

    return pd.concat([df_win, df_lose], ignore_index=True)

###############################################
# 3. Build Tourney Training Set
###############################################
df_tourney_mod = add_binary_rows(df_tourney)

# Merge Seeds
df_seeds_mod = df_seeds[["Season","TeamID","SeedValue"]].copy()
df_tourney_mod = df_tourney_mod.merge(
    df_seeds_mod, left_on=["Season","TeamA"], right_on=["Season","TeamID"], how="left"
).drop("TeamID", axis=1).rename(columns={"SeedValue":"SeedA"})

df_tourney_mod = df_tourney_mod.merge(
    df_seeds_mod, left_on=["Season","TeamB"], right_on=["Season","TeamID"], how="left"
).drop("TeamID", axis=1).rename(columns={"SeedValue":"SeedB"})

# Merge Season Stats
df_tourney_mod = df_tourney_mod.merge(
    df_team_stats, left_on=["Season","TeamA"], right_on=["Season","TeamID"], how="left"
).drop("TeamID", axis=1)
colsA = {}
for c in df_team_stats.columns:
    if c not in ["Season","TeamID"]:
        colsA[c] = f"A_{c}"
df_tourney_mod.rename(columns=colsA, inplace=True)

df_tourney_mod = df_tourney_mod.merge(
    df_team_stats, left_on=["Season","TeamB"], right_on=["Season","TeamID"], how="left"
).drop("TeamID", axis=1)
colsB = {}
for c in df_team_stats.columns:
    if c not in ["Season","TeamID"]:
        colsB[c] = f"B_{c}"
df_tourney_mod.rename(columns=colsB, inplace=True)

# Differences
df_tourney_mod["SeedDiff"] = df_tourney_mod["SeedA"] - df_tourney_mod["SeedB"]
df_tourney_mod["WinRateDiff"] = df_tourney_mod["A_WinRate"] - df_tourney_mod["B_WinRate"]
df_tourney_mod["MeanMarginDiff"] = df_tourney_mod["A_MeanMargin"] - df_tourney_mod["B_MeanMargin"]

df_tourney_mod.fillna(0, inplace=True)
df_tourney_mod.sort_values(["Season"], inplace=True)

###############################################
# 4. Define Features
###############################################
feature_cols = [
    "SeedA","SeedB","SeedDiff",
    "A_WinRate","B_WinRate","WinRateDiff",
    "A_MeanMargin","B_MeanMargin","MeanMarginDiff",
    "A_FGPct","B_FGPct","A_FG3Pct","B_FG3Pct","A_FTPct","B_FTPct",
    "A_Games","B_Games"
]

###############################################
# 5. Season-by-Season Cross-Validation
###############################################
unique_seasons = sorted(df_tourney_mod["Season"].unique())
scores = []

# We'll store predictions for diagnosing, but it's optional
df_tourney_mod["pred_proba"] = np.nan

for i, test_season in enumerate(unique_seasons):
    # Need at least 1 prior season to train on
    if i == 0:
        continue
    train_seasons = [s for s in unique_seasons if s < test_season]

    df_train = df_tourney_mod[df_tourney_mod["Season"].isin(train_seasons)].copy()
    df_test  = df_tourney_mod[df_tourney_mod["Season"] == test_season].copy()

    X_train = df_train[feature_cols]
    y_train = df_train["Label"]
    X_test  = df_test[feature_cols]
    y_test  = df_test["Label"]

    # Build Pipeline
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42))
    ])

    # Calibrate with 5-fold CV
    calibrator = CalibratedClassifierCV(
        estimator=pipeline,
        method='isotonic',
        cv=5  # uses 5-fold internally
    )
    calibrator.fit(X_train, y_train)

    # Evaluate
    preds_test = calibrator.predict_proba(X_test)[:,1]
    ll = log_loss(y_test, preds_test)
    scores.append(ll)
    df_tourney_mod.loc[df_tourney_mod["Season"] == test_season, "pred_proba"] = preds_test

    print(f"Season {test_season}, LogLoss = {ll:.4f}")

mean_cv = np.mean(scores)
print(f"\nOverall CV LogLoss across seasons = {mean_cv:.4f}")

###############################################
# 6. Final Model (Train on all seasons < 2025)
###############################################
df_train_final = df_tourney_mod[df_tourney_mod["Season"] < 2025].copy()
X_final = df_train_final[feature_cols]
y_final = df_train_final["Label"]

pipeline_final = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("clf", ExtraTreesClassifier(n_estimators=300, max_depth=12, random_state=42))
])

calibrator_final = CalibratedClassifierCV(
    estimator=pipeline_final,
    method='isotonic',
    cv=5
)
calibrator_final.fit(X_final, y_final)

###############################################
# 7. Prepare Submission Data (2025 matchups)
###############################################
df_sub["Season"] = df_sub["ID"].apply(lambda x: int(x.split("_")[0]))
df_sub["TeamA"]  = df_sub["ID"].apply(lambda x: int(x.split("_")[1]))
df_sub["TeamB"]  = df_sub["ID"].apply(lambda x: int(x.split("_")[2]))

df_seeds_mod = df_seeds[["Season","TeamID","SeedValue"]].copy()

# Merge seeds
df_sub = pd.merge(df_sub, df_seeds_mod, left_on=["Season","TeamA"], right_on=["Season","TeamID"], how="left")\
    .drop("TeamID", axis=1).rename(columns={"SeedValue":"SeedA"})
df_sub = pd.merge(df_sub, df_seeds_mod, left_on=["Season","TeamB"], right_on=["Season","TeamID"], how="left")\
    .drop("TeamID", axis=1).rename(columns={"SeedValue":"SeedB"})

# Merge team stats
df_sub = df_sub.merge(df_team_stats, left_on=["Season","TeamA"], right_on=["Season","TeamID"], how="left")
for c in df_team_stats.columns:
    if c not in ["Season","TeamID"]:
        df_sub.rename(columns={c: f"A_{c}"}, inplace=True)
df_sub.drop("TeamID", axis=1, inplace=True)

df_sub = df_sub.merge(df_team_stats, left_on=["Season","TeamB"], right_on=["Season","TeamID"], how="left")
for c in df_team_stats.columns:
    if c not in ["Season","TeamID"]:
        df_sub.rename(columns={c: f"B_{c}"}, inplace=True)
df_sub.drop("TeamID", axis=1, inplace=True)

df_sub["SeedDiff"] = df_sub["SeedA"] - df_sub["SeedB"]
df_sub["WinRateDiff"] = df_sub["A_WinRate"] - df_sub["B_WinRate"]
df_sub["MeanMarginDiff"] = df_sub["A_MeanMargin"] - df_sub["B_MeanMargin"]

df_sub.fillna(0, inplace=True)

###############################################
# 8. Predict and Save Submission
###############################################
X_sub = df_sub[feature_cols].copy()
preds_sub = calibrator_final.predict_proba(X_sub)[:,1]
# clip extremes
preds_sub = np.clip(preds_sub, 0.001, 0.999)

df_sub["Pred"] = preds_sub
submission = df_sub[["ID","Pred"]].copy()
submission.to_csv("submission.csv", index=False)

print(submission.head(10))
sns.histplot(submission["Pred"], bins=20)
plt.title("Final Probability Distribution")
plt.show()


