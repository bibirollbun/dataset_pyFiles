#############################################
# Fast March Machine Learning Mania Pipeline
#############################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

#############################################
# 1. IMPORT LIBRARIES & SETUP
#############################################
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from IPython.display import FileLink

# Define the data path where all CSV files reside
DATA_PATH = "/kaggle/input/march-machine-learning-mania-2025/"

#############################################
# 2. LOAD ALL CSV FILES (Key Files Only)
#############################################
# For speed, we load only key files needed for a fast pipeline.
filenames = [
    "MTeams.csv", "WTeams.csv",
    "MNCAATourneySeeds.csv", "WNCAATourneySeeds.csv",
    "MRegularSeasonDetailedResults.csv", "WRegularSeasonDetailedResults.csv",
    "Conferences.csv", "MTeamConferences.csv", "WTeamConferences.csv",
    "MMasseyOrdinals.csv",
    "SampleSubmissionStage2.csv"
]

data = {}
for fname in filenames:
    fpath = os.path.join(DATA_PATH, fname)
    if os.path.exists(fpath):
        data[fname] = pd.read_csv(fpath)
        print(f"Loaded {fname} with shape {data[fname].shape}")
    else:
        print(f"File {fname} not found.")

#############################################
# 3. FEATURE ENGINEERING – TEAM AGGREGATES
#############################################
# Combine detailed regular season results from men and women.
m_reg = data.get("MRegularSeasonDetailedResults.csv")
w_reg = data.get("WRegularSeasonDetailedResults.csv")
m_reg["Gender"] = "M"
w_reg["Gender"] = "W"
reg_detailed = pd.concat([m_reg, w_reg], ignore_index=True)
reg_detailed["ScoreDiff"] = reg_detailed["WScore"] - reg_detailed["LScore"]

# Use base stats from detailed results.
base_stats = ["Score", "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]
reg_detailed["Weight"] = reg_detailed["DayNum"] + 1

from collections import defaultdict
simple_agg = {}
weighted_agg = {}
margin_dict = {}
for idx, row in reg_detailed.iterrows():
    season = row["Season"]
    key_w = (season, row["WTeamID"])
    if key_w not in simple_agg:
        simple_agg[key_w] = {stat: 0 for stat in base_stats}
        simple_agg[key_w]["count"] = 0
        weighted_agg[key_w] = {stat: 0 for stat in base_stats}
        weighted_agg[key_w]["w_sum"] = 0
        margin_dict[key_w] = []
    for stat in base_stats:
        simple_agg[key_w][stat] += row["W"+stat]
        weighted_agg[key_w][stat] += row["W"+stat] * row["Weight"]
    simple_agg[key_w]["count"] += 1
    weighted_agg[key_w]["w_sum"] += row["Weight"]
    margin_dict[key_w].append(row["ScoreDiff"])
    
    key_l = (season, row["LTeamID"])
    if key_l not in simple_agg:
        simple_agg[key_l] = {stat: 0 for stat in base_stats}
        simple_agg[key_l]["count"] = 0
        weighted_agg[key_l] = {stat: 0 for stat in base_stats}
        weighted_agg[key_l]["w_sum"] = 0
        margin_dict[key_l] = []
    for stat in base_stats:
        simple_agg[key_l][stat] += row["L"+stat]
        weighted_agg[key_l][stat] += row["L"+stat] * row["Weight"]
    simple_agg[key_l]["count"] += 1
    weighted_agg[key_l]["w_sum"] += row["Weight"]
    margin_dict[key_l].append(-row["ScoreDiff"])

team_feats = []
for key, agg in simple_agg.items():
    season, team_id = key
    count = agg["count"]
    avgs = {f"Avg{stat}": agg[stat] / count for stat in base_stats}
    weighted = weighted_agg[key]
    w_avgs = {f"WAvg{stat}": weighted[stat] / weighted["w_sum"] if weighted["w_sum"]>0 else 0 for stat in base_stats}
    std_margin = np.std(margin_dict[key]) if len(margin_dict[key]) > 1 else 0
    feat = {"Season": season, "TeamID": team_id, "StdMargin": std_margin}
    feat.update(avgs)
    feat.update(w_avgs)
    team_feats.append(feat)
team_stats_df = pd.DataFrame(team_feats)
print("Team features shape:", team_stats_df.shape)

# Efficiency Metrics:
team_stats_df["EFgPct"] = team_stats_df.apply(lambda r: (r["AvgFGM"] + 0.5*r["AvgFGM3"]) / r["AvgFGA"] if r["AvgFGA"]>0 else 0, axis=1)
team_stats_df["FTPct"] = team_stats_df.apply(lambda r: r["AvgFTM"] / r["AvgFTA"] if r["AvgFTA"]>0 else 0, axis=1)
team_stats_df["Possessions"] = team_stats_df.apply(lambda r: r["AvgFGA"] + 0.44*r["AvgFTA"] - r["AvgOR"] + r["AvgDR"], axis=1)
team_stats_df["OffEff"] = team_stats_df.apply(lambda r: r["AvgScore"] / r["Possessions"] if r["Possessions"]>0 else 0, axis=1)

#############################################
# Merge Additional Signals
#############################################
# Conferences:
if "MTeamConferences.csv" in data and "WTeamConferences.csv" in data:
    m_conf = data["MTeamConferences.csv"]
    w_conf = data["WTeamConferences.csv"]
    m_conf["Gender"] = "M"
    w_conf["Gender"] = "W"
    all_conf = pd.concat([m_conf, w_conf], ignore_index=True)
    conf_latest = all_conf.sort_values("Season").groupby(["Season", "TeamID"], as_index=False).last()[["Season", "TeamID", "ConfAbbrev"]]
    team_stats_df = team_stats_df.merge(conf_latest, how="left", on=["Season", "TeamID"])
else:
    team_stats_df["ConfAbbrev"] = "UNKNOWN"

# Seeds:
seed_extractor = lambda s: int("".join(ch for ch in str(s) if ch.isdigit()))
if "MNCAATourneySeeds.csv" in data:
    m_seeds = data["MNCAATourneySeeds.csv"]
    m_seeds["SeedValue"] = m_seeds["Seed"].apply(seed_extractor)
if "WNCAATourneySeeds.csv" in data:
    w_seeds = data["WNCAATourneySeeds.csv"]
    w_seeds["Seed"].fillna("0", inplace=True)
    w_seeds["SeedValue"] = w_seeds["Seed"].apply(seed_extractor)
all_seeds = pd.concat([m_seeds[["Season", "TeamID", "SeedValue"]],
                       w_seeds[["Season", "TeamID", "SeedValue"]]], ignore_index=True).drop_duplicates()
team_stats_df = team_stats_df.merge(all_seeds, how="left", on=["Season", "TeamID"])
team_stats_df["SeedValue"] = team_stats_df["SeedValue"].fillna(25)

# Massey Ordinals:
m_massey = data.get("MMasseyOrdinals.csv")
if m_massey is not None:
    m_massey["OrdinalRank"] = m_massey["OrdinalRank"].astype(float)
    def final_massey_ratings(df):
        df = df.sort_values("RankingDayNum")
        last_rows = df.groupby(["Season", "TeamID", "SystemName"], as_index=False).last()
        mean_ranks = last_rows.groupby(["Season", "TeamID"], as_index=False)["OrdinalRank"].mean()
        mean_ranks.rename(columns={"OrdinalRank": "MasseyRating"}, inplace=True)
        return mean_ranks
    massey_final = final_massey_ratings(m_massey)
else:
    massey_final = pd.DataFrame(columns=["Season", "TeamID", "MasseyRating"])
team_stats_df = team_stats_df.merge(massey_final, how="left", on=["Season", "TeamID"])
team_stats_df["MasseyRating"] = team_stats_df["MasseyRating"].fillna(0)

# Coaches (optional):
if "MTeamCoaches.csv" in data:
    coaches = data["MTeamCoaches.csv"]
    coach_changes = coaches.groupby(["Season", "TeamID"]).size().reset_index(name="CoachChanges")
    team_stats_df = team_stats_df.merge(coach_changes, how="left", on=["Season", "TeamID"])
    team_stats_df["CoachChanges"] = team_stats_df["CoachChanges"].fillna(0)
else:
    team_stats_df["CoachChanges"] = 0

#############################################
# 4. BUILD TRAINING DATA – MIRRORED GAMES & INTERACTION FEATURES
#############################################
train_rows = []
train_labels = []
for idx, row in reg_detailed.iterrows():
    season = row["Season"]
    wteam = row["WTeamID"]
    lteam = row["LTeamID"]
    diff = row["ScoreDiff"]
    train_rows.append([season, wteam, lteam, diff])
    train_labels.append(1)
    train_rows.append([season, lteam, wteam, -diff])
    train_labels.append(0)
train_df = pd.DataFrame(train_rows, columns=["Season", "Team1", "Team2", "ScoreDiff"])
train_labels = np.array(train_labels)

# Merge team features for Team1:
train_df = train_df.merge(team_stats_df, how="left", left_on=["Season", "Team1"], right_on=["Season", "TeamID"])
train_df = train_df.rename(columns={col: "T1_" + col for col in team_stats_df.columns if col not in ["Season", "TeamID"]})
train_df.drop("TeamID", axis=1, inplace=True)
# Merge team features for Team2:
train_df = train_df.merge(team_stats_df, how="left", left_on=["Season", "Team2"], right_on=["Season", "TeamID"])
train_df = train_df.rename(columns={col: "T2_" + col for col in team_stats_df.columns if col not in ["Season", "TeamID"]})
train_df.drop("TeamID", axis=1, inplace=True)

# Define feature sets.
# Exclude string features like "ConfAbbrev" from difference computations.
basic_feats = ["AvgScore", "AvgFGM", "AvgFGA", "AvgFGM3", "AvgFGA3", "AvgFTM", "AvgFTA",
               "AvgOR", "AvgDR", "AvgAst", "AvgTO", "AvgStl", "AvgBlk", "AvgPF",
               "EFgPct", "FTPct", "Possessions", "OffEff", "StdMargin",
               "SeedValue", "MasseyRating", "CoachChanges"]
w_feats = ["WAvg" + stat for stat in base_stats]
all_feats = basic_feats + w_feats

for feat in all_feats:
    train_df["Diff_" + feat] = train_df["T1_" + feat].fillna(0) - train_df["T2_" + feat].fillna(0)

# Additional interaction features:
train_df["SameConf"] = (train_df["T1_ConfAbbrev"] == train_df["T2_ConfAbbrev"]).astype(int)
# Use OffEff (Offensive Efficiency) from advanced metrics:
train_df["Int_Seed_OffEff"] = (train_df["T1_SeedValue"] - train_df["T2_SeedValue"]) * (train_df["T1_OffEff"] - train_df["T2_OffEff"])

# Final training feature set:
diff_cols = [col for col in train_df.columns if col.startswith("Diff_")]
final_train_features = diff_cols + ["ScoreDiff", "SameConf", "Int_Seed_OffEff"]

X = train_df[final_train_features].fillna(0).values
y = train_labels

#############################################
# 5. MODEL TRAINING – FAST STACKING ENSEMBLE WITH CALIBRATION
#############################################
# To speed up runtime, we use a lighter ensemble: LR, RF, and XGB only.

xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=4, learning_rate=0.03)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)

estimators = [
    ('lr', lr_model),
    ('rf', rf_model),
    ('xgb', xgb_model)
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=3,  # use fewer folds to speed up
    stack_method='predict_proba',
    passthrough=True
)

calibrated_model = CalibratedClassifierCV(stack_model, cv=3, method='isotonic')

# --- Split and Scale Data ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# --- Train the Ensemble ---
calibrated_model.fit(X_train_scaled, y_train)
val_preds = calibrated_model.predict_proba(X_val_scaled)[:, 1]
print("Validation LogLoss:", log_loss(y_val, val_preds))

# Retrain on full training data for final submission:
X_full_scaled = scaler.fit_transform(X)
calibrated_model.fit(X_full_scaled, y)

#############################################
# 6. FINAL SUBMISSION GENERATION
#############################################
sample_submission = data.get("SampleSubmissionStage2.csv")
if sample_submission is None:
    raise ValueError("SampleSubmissionStage2.csv not found!")

def parse_submission_id(id_str):
    parts = id_str.split("_")
    return int(parts[0]), int(parts[1]), int(parts[2])
sample_submission["Season"], sample_submission["Team1"], sample_submission["Team2"] = zip(*sample_submission["ID"].apply(parse_submission_id))

test_df = sample_submission.copy()
# For Team1:
test_df = test_df.merge(team_stats_df, how="left", left_on=["Season", "Team1"], right_on=["Season", "TeamID"])
test_df = test_df.rename(columns={col: "T1_" + col for col in team_stats_df.columns if col not in ["Season", "TeamID"]})
test_df.drop("TeamID", axis=1, inplace=True)
# For Team2:
test_df = test_df.merge(team_stats_df, how="left", left_on=["Season", "Team2"], right_on=["Season", "TeamID"])
test_df = test_df.rename(columns={col: "T2_" + col for col in team_stats_df.columns if col not in ["Season", "TeamID"]})
test_df.drop("TeamID", axis=1, inplace=True)

for feat in all_feats:
    test_df["Diff_" + feat] = test_df["T1_" + feat].fillna(0) - test_df["T2_" + feat].fillna(0)
test_df["SameConf"] = (test_df["T1_ConfAbbrev"] == test_df["T2_ConfAbbrev"]).astype(int)
test_df["Int_Seed_OffEff"] = (test_df["T1_SeedValue"] - test_df["T2_SeedValue"]) * (test_df["T1_OffEff"] - test_df["T2_OffEff"])
test_df["ScoreDiff"] = 0

test_feature_cols = [ "Diff_" + feat for feat in all_feats] + ["SameConf", "Int_Seed_OffEff", "ScoreDiff"]
X_test = test_df[test_feature_cols].fillna(0).values
X_test_scaled = scaler.transform(X_test)

preds = calibrated_model.predict_proba(X_test_scaled)[:, 1]
preds = np.clip(preds, 0.025, 0.975)

submission = sample_submission.copy()
submission["Pred"] = preds
submission = submission[["ID", "Pred"]]  # Keep only these two columns
submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created successfully!")

FileLink("submission.csv")





