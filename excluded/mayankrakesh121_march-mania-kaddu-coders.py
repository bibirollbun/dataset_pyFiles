import os, re, pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# RAPIDS cuDF import for GPU preprocessing
import cudf

# Import CatBoost (GPU‑enabled)
from catboost import CatBoostClassifier

# Set pandas options and data path
pd.set_option('display.max_columns', None)
DATA_PATH = '/kaggle/input/march-machine-learning-mania-2025/'

#############################################
# Helper: treat_seed function
#############################################
def treat_seed(seed):
    return int(re.sub("[^0-9]", "", seed))

#############################################
# 1. READ & PREPARE THE DATA (Pandas)
#############################################
print("Files in DATA_PATH:")
for filename in sorted(os.listdir(DATA_PATH)):
    print(filename)

# --- Tournament Seeds (Men's and Women's) ---
df_seeds = pd.concat([
    pd.read_csv(os.path.join(DATA_PATH, "MNCAATourneySeeds.csv")),
    pd.read_csv(os.path.join(DATA_PATH, "WNCAATourneySeeds.csv"))
], ignore_index=True)

# --- Regular Season Results (Men's and Women's) ---
df_season_results = pd.concat([
    pd.read_csv(os.path.join(DATA_PATH, "MRegularSeasonCompactResults.csv")),
    pd.read_csv(os.path.join(DATA_PATH, "WRegularSeasonCompactResults.csv"))
], ignore_index=True)
df_season_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)
df_season_results['ScoreGap'] = df_season_results['WScore'] - df_season_results['LScore']

# ----------------------------
# NEW: Compute a Pace Proxy & Opponent Quality
# ----------------------------
# (1) Compute total points per game as a proxy for pace.
df_season_results['TotalScore'] = df_season_results['WScore'] + df_season_results['LScore']

# (2) For each team, compute the average total points (i.e. average game scoring environment)
avg_total_win = df_season_results.groupby(['Season', 'WTeamID'])['TotalScore'].mean().reset_index().rename(
    columns={'WTeamID':'TeamID', 'TotalScore':'AvgTotalPoints'}
)
avg_total_loss = df_season_results.groupby(['Season', 'LTeamID'])['TotalScore'].mean().reset_index().rename(
    columns={'LTeamID':'TeamID', 'TotalScore':'AvgTotalPoints'}
)
df_avg_total = pd.concat([avg_total_win, avg_total_loss]).groupby(['Season','TeamID']).mean().reset_index()

# (3) Compute opponent average score as a proxy for strength of schedule.
opp_score_win = df_season_results.groupby(['Season', 'WTeamID'])['LScore'].mean().reset_index().rename(
    columns={'WTeamID':'TeamID', 'LScore':'OppAvgScore'}
)
opp_score_loss = df_season_results.groupby(['Season', 'LTeamID'])['WScore'].mean().reset_index().rename(
    columns={'LTeamID':'TeamID', 'WScore':'OppAvgScore'}
)
df_opp_avg = pd.concat([opp_score_win, opp_score_loss]).groupby(['Season','TeamID']).mean().reset_index()

#############################################
# 1A. Compute Overall Season-Level Features
#############################################
# Overall aggregates: wins, losses, and average score gap.
num_win = (df_season_results.groupby(['Season', 'WTeamID'])
           .count().reset_index()[['Season', 'WTeamID', 'DayNum']]
           .rename(columns={"DayNum": "NumWins", "WTeamID": "TeamID"}))
num_loss = (df_season_results.groupby(['Season', 'LTeamID'])
            .count().reset_index()[['Season', 'LTeamID', 'DayNum']]
            .rename(columns={"DayNum": "NumLosses", "LTeamID": "TeamID"}))
gap_win = (df_season_results.groupby(['Season', 'WTeamID'])
           .mean().reset_index()[['Season', 'WTeamID', 'ScoreGap']]
           .rename(columns={"ScoreGap": "GapWins", "WTeamID": "TeamID"}))
gap_loss = (df_season_results.groupby(['Season', 'LTeamID'])
            .mean().reset_index()[['Season', 'LTeamID', 'ScoreGap']]
            .rename(columns={"ScoreGap": "GapLosses", "LTeamID": "TeamID"}))
df_features_season_w = (df_season_results.groupby(['Season', 'WTeamID'])
                        .count().reset_index()[['Season', 'WTeamID']]
                        .rename(columns={"WTeamID": "TeamID"}))
df_features_season_l = (df_season_results.groupby(['Season', 'LTeamID'])
                        .count().reset_index()[['Season', 'LTeamID']]
                        .rename(columns={"LTeamID": "TeamID"}))
df_features_season = pd.concat([df_features_season_w, df_features_season_l], axis=0)\
                       .drop_duplicates()\
                       .sort_values(['Season', 'TeamID'])\
                       .reset_index(drop=True)
df_features_season = df_features_season.merge(num_win, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(num_loss, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(gap_win, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(gap_loss, on=['Season', 'TeamID'], how='left')
df_features_season.fillna(0, inplace=True)
df_features_season['WinRatio'] = df_features_season['NumWins'] / (df_features_season['NumWins'] + df_features_season['NumLosses'])
df_features_season['GapAvg'] = ((df_features_season['NumWins'] * df_features_season['GapWins'] - 
                                 df_features_season['NumLosses'] * df_features_season['GapLosses']) /
                                (df_features_season['NumWins'] + df_features_season['NumLosses']))
df_features_season.drop(['NumWins', 'NumLosses', 'GapWins', 'GapLosses'], axis=1, inplace=True)

# Merge in our pace proxy and SoS (opponent quality) features.
df_features_season = df_features_season.merge(df_avg_total, on=['Season','TeamID'], how='left')
df_features_season = df_features_season.merge(df_opp_avg, on=['Season','TeamID'], how='left')

#############################################
# 1B. Compute Recent Season-Level Features
#############################################
# Define a threshold for "recent" games (e.g., DayNum >= 110)
recent_threshold = 110
df_recent = df_season_results[df_season_results['DayNum'] >= recent_threshold]

num_win_recent = (df_recent.groupby(['Season', 'WTeamID'])
           .count().reset_index()[['Season', 'WTeamID', 'DayNum']]
           .rename(columns={"DayNum": "NumWinsRecent", "WTeamID": "TeamID"}))
num_loss_recent = (df_recent.groupby(['Season', 'LTeamID'])
            .count().reset_index()[['Season', 'LTeamID', 'DayNum']]
            .rename(columns={"DayNum": "NumLossesRecent", "LTeamID": "TeamID"}))
gap_win_recent = (df_recent.groupby(['Season', 'WTeamID'])
           .mean().reset_index()[['Season', 'WTeamID', 'ScoreGap']]
           .rename(columns={"ScoreGap": "GapWinsRecent", "WTeamID": "TeamID"}))
gap_loss_recent = (df_recent.groupby(['Season', 'LTeamID'])
            .mean().reset_index()[['Season', 'LTeamID', 'ScoreGap']]
            .rename(columns={"ScoreGap": "GapLossesRecent", "LTeamID": "TeamID"}))
df_features_season_recent_w = (df_recent.groupby(['Season', 'WTeamID'])
                        .count().reset_index()[['Season', 'WTeamID']]
                        .rename(columns={"WTeamID": "TeamID"}))
df_features_season_recent_l = (df_recent.groupby(['Season', 'LTeamID'])
                        .count().reset_index()[['Season', 'LTeamID']]
                        .rename(columns={"LTeamID": "TeamID"}))
df_features_season_recent = pd.concat([df_features_season_recent_w, df_features_season_recent_l], axis=0)\
                       .drop_duplicates()\
                       .sort_values(['Season', 'TeamID'])\
                       .reset_index(drop=True)
df_features_season_recent = df_features_season_recent.merge(num_win_recent, on=['Season', 'TeamID'], how='left')
df_features_season_recent = df_features_season_recent.merge(num_loss_recent, on=['Season', 'TeamID'], how='left')
df_features_season_recent = df_features_season_recent.merge(gap_win_recent, on=['Season', 'TeamID'], how='left')
df_features_season_recent = df_features_season_recent.merge(gap_loss_recent, on=['Season', 'TeamID'], how='left')
df_features_season_recent.fillna(0, inplace=True)
df_features_season_recent['WinRatioRecent'] = df_features_season_recent['NumWinsRecent'] / (df_features_season_recent['NumWinsRecent'] + df_features_season_recent['NumLossesRecent'])
df_features_season_recent['GapAvgRecent'] = ((df_features_season_recent['NumWinsRecent'] * df_features_season_recent['GapWinsRecent'] - 
                                 df_features_season_recent['NumLossesRecent'] * df_features_season_recent['GapLossesRecent']) /
                                (df_features_season_recent['NumWinsRecent'] + df_features_season_recent['NumLossesRecent']))
df_features_season_recent.drop(['NumWinsRecent', 'NumLossesRecent', 'GapWinsRecent', 'GapLossesRecent'], axis=1, inplace=True)

# ----------------------------
# NEW: Add Temporal Momentum Features
# ----------------------------
# Delta metrics: change from overall to recent performance.
df_features_season['DeltaWinRatio'] = df_features_season_recent['WinRatioRecent'] - df_features_season['WinRatio']
df_features_season['DeltaGapAvg'] = df_features_season_recent['GapAvgRecent'] - df_features_season['GapAvg']

# ----------------------------
# NEW: Compute Adjusted Score Gap (AdjScoreGap)
# ----------------------------
# Avoid division by zero by replacing zeros in AvgTotalPoints with a small value.
df_features_season['AvgTotalPoints'] = df_features_season['AvgTotalPoints'].replace(0, 1e-5)
df_features_season['AdjScoreGap'] = df_features_season['GapAvg'] / df_features_season['AvgTotalPoints']

# --- Merge overall and recent season-level features ---
# (Merge the recent metrics into the overall features by key: Season and TeamID)
df_features_season = pd.merge(df_features_season, 
                              df_features_season_recent[['Season','TeamID','WinRatioRecent','GapAvgRecent']],
                              on=['Season','TeamID'], 
                              how='left')
df_features_season.fillna(0, inplace=True)

#############################################
# 2. Prepare Tournament Data & Merge Features
#############################################
# --- Tournament Results (we use seasons >= 2016) ---
df_tourney_results = pd.concat([
    pd.read_csv(os.path.join(DATA_PATH, "WNCAATourneyCompactResults.csv")),
    pd.read_csv(os.path.join(DATA_PATH, "MNCAATourneyCompactResults.csv"))
], ignore_index=True)
df_tourney_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)
df = df_tourney_results.copy()
df = df[df['Season'] >= 2016].reset_index(drop=True)

# --- Merge Seeds for Winner and Loser ---
df = pd.merge(
    df, df_seeds,
    how='left',
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedW'})
df = pd.merge(
    df, df_seeds,
    how='left',
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedL'})

df['SeedW'] = df['SeedW'].apply(treat_seed)
df['SeedL'] = df['SeedL'].apply(treat_seed)

# --- Merge Overall & Recent Season-Level Features for both Winner and Loser ---
# For winners:
df = pd.merge(
    df, df_features_season,
    how='left',
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'WinRatio': 'WinRatioW', 
    'GapAvg': 'GapAvgW',
    'WinRatioRecent': 'WinRatioRecentW',
    'GapAvgRecent': 'GapAvgRecentW',
    'AvgTotalPoints': 'AvgTotalPointsW',
    'OppAvgScore': 'OppAvgScoreW',
    'DeltaWinRatio': 'DeltaWinRatioW',
    'DeltaGapAvg': 'DeltaGapAvgW',
    'AdjScoreGap': 'AdjScoreGapW'
}).drop('TeamID', axis=1)
# For losers:
df = pd.merge(
    df, df_features_season,
    how='left',
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'WinRatio': 'WinRatioL', 
    'GapAvg': 'GapAvgL',
    'WinRatioRecent': 'WinRatioRecentL',
    'GapAvgRecent': 'GapAvgRecentL',
    'AvgTotalPoints': 'AvgTotalPointsL',
    'OppAvgScore': 'OppAvgScoreL',
    'DeltaWinRatio': 'DeltaWinRatioL',
    'DeltaGapAvg': 'DeltaGapAvgL',
    'AdjScoreGap': 'AdjScoreGapL'
}).drop('TeamID', axis=1)

# --- Create Mirrored Matches ---
def add_loosing_matches(df):
    # Rename columns for winning records (Team A wins, Team B loses)
    win_rename = {
        "WTeamID": "TeamIdA", 
        "WScore": "ScoreA", 
        "LTeamID": "TeamIdB",
        "LScore": "ScoreB",
    }
    win_rename.update({c: c[:-1] + "A" for c in df.columns if c.endswith('W')})
    win_rename.update({c: c[:-1] + "A" for c in ['SeedW', 'WinRatioW', 'GapAvgW', 'WinRatioRecentW', 'GapAvgRecentW',
                                                  'AvgTotalPointsW', 'OppAvgScoreW', 'DeltaWinRatioW', 'DeltaGapAvgW', 'AdjScoreGapW']})
    win_rename.update({c: c[:-1] + "B" for c in df.columns if c.endswith('L')})
    
    lose_rename = {
        "WTeamID": "TeamIdB", 
        "WScore": "ScoreB", 
        "LTeamID": "TeamIdA",
        "LScore": "ScoreA",
    }
    lose_rename.update({c: c[:-1] + "B" for c in df.columns if c.endswith('W')})
    lose_rename.update({c: c[:-1] + "A" for c in df.columns if c.endswith('L')})
    
    win_df = df.copy().rename(columns=win_rename)
    lose_df = df.copy().rename(columns=lose_rename)
    return pd.concat([win_df, lose_df], axis=0, sort=False)

df = add_loosing_matches(df)

# --- Compute Difference Features for Overall, Recent & New Metrics ---
# Already computed for Seed, WinRatio, GapAvg, WinRatioRecent, GapAvgRecent.
cols_to_diff = ['Seed', 'WinRatio', 'GapAvg', 'WinRatioRecent', 'GapAvgRecent']
for col in cols_to_diff:
    df[col + 'Diff'] = df[col + 'A'] - df[col + 'B']

# New features:
df['AvgTotalPointsDiff'] = df['AvgTotalPointsA'] - df['AvgTotalPointsB']
df['OppAvgScoreDiff'] = df['OppAvgScoreA'] - df['OppAvgScoreB']
df['DeltaWinRatioDiff'] = df['DeltaWinRatioA'] - df['DeltaWinRatioB']
df['DeltaGapAvgDiff'] = df['DeltaGapAvgA'] - df['DeltaGapAvgB']
df['AdjScoreGapDiff'] = df['AdjScoreGapA'] - df['AdjScoreGapB']

# Matchup-specific interaction features:
df['SeedWinDiff'] = (df['SeedA'] * df['WinRatioA']) - (df['SeedB'] * df['WinRatioB'])
df['SeedWinRecentDiff'] = (df['SeedA'] * df['WinRatioRecentA']) - (df['SeedB'] * df['WinRatioRecentB'])

df['ScoreDiff'] = df['ScoreA'] - df['ScoreB']
df['WinA'] = (df['ScoreDiff'] > 0).astype(int)

#############################################
# 3. Prepare the Test Data
#############################################
df_test = pd.read_csv(os.path.join(DATA_PATH, "SampleSubmissionStage1.csv"))
df_test['Season'] = df_test['ID'].apply(lambda x: int(x.split('_')[0]))
df_test['TeamIdA'] = df_test['ID'].apply(lambda x: int(x.split('_')[1]))
df_test['TeamIdB'] = df_test['ID'].apply(lambda x: int(x.split('_')[2]))

# --- Merge Seeds for Test Data ---
df_test = pd.merge(
    df_test, df_seeds,
    how='left',
    left_on=['Season', 'TeamIdA'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedA'}).fillna('W01')
df_test = pd.merge(
    df_test, df_seeds,
    how='left',
    left_on=['Season', 'TeamIdB'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedB'}).fillna('W01')
df_test['SeedA'] = df_test['SeedA'].apply(treat_seed)
df_test['SeedB'] = df_test['SeedB'].apply(treat_seed)

# --- Merge Overall & Recent Season-Level Features for TeamIdA ---
df_test = pd.merge(
    df_test, df_features_season,
    how='left',
    left_on=['Season', 'TeamIdA'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'WinRatio': 'WinRatioA', 
    'GapAvg': 'GapAvgA',
    'WinRatioRecent': 'WinRatioRecentA',
    'GapAvgRecent': 'GapAvgRecentA',
    'AvgTotalPoints': 'AvgTotalPointsA',
    'OppAvgScore': 'OppAvgScoreA',
    'DeltaWinRatio': 'DeltaWinRatioA',
    'DeltaGapAvg': 'DeltaGapAvgA',
    'AdjScoreGap': 'AdjScoreGapA'
}).drop('TeamID', axis=1)

# --- Merge Overall & Recent Season-Level Features for TeamIdB ---
df_test = pd.merge(
    df_test, df_features_season,
    how='left',
    left_on=['Season', 'TeamIdB'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'WinRatio': 'WinRatioB', 
    'GapAvg': 'GapAvgB',
    'WinRatioRecent': 'WinRatioRecentB',
    'GapAvgRecent': 'GapAvgRecentB',
    'AvgTotalPoints': 'AvgTotalPointsB',
    'OppAvgScore': 'OppAvgScoreB',
    'DeltaWinRatio': 'DeltaWinRatioB',
    'DeltaGapAvg': 'DeltaGapAvgB',
    'AdjScoreGap': 'AdjScoreGapB'
}).drop('TeamID', axis=1)

# --- Create Difference Features for Test Data ---
df_test["SeedDiff"] = df_test["SeedA"] - df_test["SeedB"]
df_test["WinRatioDiff"] = df_test["WinRatioA"] - df_test["WinRatioB"]
df_test["GapAvgDiff"] = df_test["GapAvgA"] - df_test["GapAvgB"]
df_test["WinRatioRecentDiff"] = df_test["WinRatioRecentA"] - df_test["WinRatioRecentB"]
df_test["GapAvgRecentDiff"] = df_test["GapAvgRecentA"] - df_test["GapAvgRecentB"]
df_test["AvgTotalPointsDiff"] = df_test["AvgTotalPointsA"] - df_test["AvgTotalPointsB"]
df_test["OppAvgScoreDiff"] = df_test["OppAvgScoreA"] - df_test["OppAvgScoreB"]
df_test["DeltaWinRatioDiff"] = df_test["DeltaWinRatioA"] - df_test["DeltaWinRatioB"]
df_test["DeltaGapAvgDiff"] = df_test["DeltaGapAvgA"] - df_test["DeltaGapAvgB"]
df_test["AdjScoreGapDiff"] = df_test["AdjScoreGapA"] - df_test["AdjScoreGapB"]

df_test["SeedWinDiff"] = (df_test["SeedA"] * df_test["WinRatioA"]) - (df_test["SeedB"] * df_test["WinRatioB"])
df_test["SeedWinRecentDiff"] = (df_test["SeedA"] * df_test["WinRatioRecentA"]) - (df_test["SeedB"] * df_test["WinRatioRecentB"])

#############################################
# 4. Define Features & Convert to GPU (cuDF)
#############################################
# Numeric features: include overall, recent and our new engineered metrics.
features_numeric = [
    'SeedA', 'SeedB', 
    'WinRatioA', 'GapAvgA', 'WinRatioRecentA', 'GapAvgRecentA',
    'AvgTotalPointsA', 'OppAvgScoreA', 'DeltaWinRatioA', 'DeltaGapAvgA', 'AdjScoreGapA',
    'WinRatioB', 'GapAvgB', 'WinRatioRecentB', 'GapAvgRecentB',
    'AvgTotalPointsB', 'OppAvgScoreB', 'DeltaWinRatioB', 'DeltaGapAvgB', 'AdjScoreGapB',
    'SeedDiff', 'WinRatioDiff', 'GapAvgDiff', 'WinRatioRecentDiff', 'GapAvgRecentDiff',
    'AvgTotalPointsDiff', 'OppAvgScoreDiff', 'DeltaWinRatioDiff', 'DeltaGapAvgDiff', 'AdjScoreGapDiff',
    'SeedWinDiff', 'SeedWinRecentDiff'
]
# Categorical columns (Team IDs) – we will target encode these.
cat_cols = ['TeamIdA', 'TeamIdB']
features_gpu = features_numeric + [col + '_target' for col in cat_cols]

#############################################
# 5. GPU Target Encoding using cuDF
#############################################
def gpu_target_encode_train(df, cat_cols, target):
    for col in cat_cols:
        df[col] = df[col].astype(str)
    global_mean = df[target].mean()
    mappings = {}
    for col in cat_cols:
        mapping = df.groupby(col)[target].mean().reset_index()
        mapping = mapping.rename(columns={target: col + '_target'})
        mappings[col] = mapping
        df = df.merge(mapping, on=col, how='left')
        df[col + '_target'] = df[col + '_target'].fillna(global_mean)
    return df, mappings

def gpu_target_encode_test(df, mappings, cat_cols):
    for col in cat_cols:
        df[col] = df[col].astype(str)
        mapping = mappings[col]
        df = df.merge(mapping, on=col, how='left')
        global_mean = mapping[col + '_target'].mean()
        df[col + '_target'] = df[col + '_target'].fillna(global_mean)
    return df

#############################################
# Convert Pandas DataFrames to cuDF DataFrames
#############################################
gdf = cudf.DataFrame.from_pandas(df)
gdf_test = cudf.DataFrame.from_pandas(df_test)

gdf, mappings = gpu_target_encode_train(gdf, cat_cols, 'WinA')
gdf_test = gpu_target_encode_test(gdf_test, mappings, cat_cols)

gdf = gdf.reset_index(drop=True)
gdf_test = gdf_test.reset_index(drop=True)

#############################################
# 6. GPU Rescaling: median imputation + min–max scaling using cuDF
#############################################
def gpu_rescale(df, features):
    for col in features:
        df[col] = df[col].fillna(0)
    mins = df[features].min()
    maxs = df[features].max()
    for col in features:
        diff = maxs[col] - mins[col]
        if diff == 0:
            df[col] = 0.0
        else:
            df[col] = (df[col] - mins[col]) / diff
    return df

gdf = gpu_rescale(gdf, features_gpu)
gdf_test = gpu_rescale(gdf_test, features_gpu)

#############################################
# 7. Train a GPU-Enabled CatBoost Model on 50% of Training Data
#############################################
# Convert GPU preprocessed training data back to Pandas.
X_full = gdf[features_gpu].to_pandas()
y_full = gdf['WinA'].to_pandas()

# Use 50% of the training data for model fitting.
X_train_half, X_val_half, y_train_half, y_val_half = train_test_split(
    X_full, y_full, train_size=0.5, random_state=42, stratify=y_full
)



# Import additional libraries for stacking
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# --- Create Base Models ---
# GPU-enabled CatBoost (with parameters similar to those used earlier)
base_cat = CatBoostClassifier(**{'iterations': 864,
 'learning_rate': 0.09842363449008831,
 'depth': 3,
 'l2_leaf_reg': 1.387525124100511,
 'verbose': 100,
 # 'task_type': 'GPU',
 'devices': '1'})

# LightGBM model
base_lgb = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6
)

# XGBoost model
base_xgb = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Define the list of base estimators.
estimators = [
    ('cat', base_cat),
    ('lgb', base_lgb),
    ('xgb', base_xgb)
]

# --- Create the Meta-Model ---
# We use a simple logistic regression as the meta-model.
meta_model = LogisticRegression(max_iter=10000)

# --- Build the Stacking Classifier ---
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    cv=5,            # Use 5-fold cross-validation to generate out-of-fold predictions
    passthrough=True,  # Optionally pass original features along with base predictions to the meta-model
    n_jobs=-1
)

# --- Train the Stacking Classifier on the Full Training Data ---
# X_full and y_full are obtained from your GPU-rescaled data:
X_full = gdf[features_gpu].to_pandas()
y_full = gdf['WinA'].to_pandas()
stacking_clf.fit(X_full, y_full)

# --- Generate Predictions on the Test Set ---
# X_test is obtained similarly from your GPU-rescaled test data:
X_test = gdf_test[features_gpu].to_pandas()
stacked_preds = stacking_clf.predict_proba(X_test)[:, 1]

# Save predictions to the test cuDF DataFrame and then to a CSV file.
gdf_test['pred'] = stacked_preds
final_sub = gdf_test[['ID', 'pred']].to_pandas()
final_sub.to_csv('submission.csv', index=False)
print(final_sub.head())

# Plot the distribution of the stacked classifier's predictions.
sns.displot(final_sub['pred'], kde=True)
plt.title("Distribution of Stacked Classifier Predictions")
plt.show()





