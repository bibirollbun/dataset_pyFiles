# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import pandas as pd

# Update DATA_PATH to the correct folder where your CSV files are located.
DATA_PATH = "/kaggle/input/march-machine-learning-mania-2025/"

def load_data(filepath, **kwargs):
    return pd.read_csv(filepath, **kwargs)

# Now load your files using the updated DATA_PATH
cities_df = load_data(os.path.join(DATA_PATH, "Cities.csv"))
conferences_df = load_data(os.path.join(DATA_PATH, "Conferences.csv"))
mens_conferences_df = load_data(os.path.join(DATA_PATH, "MConferenceTourneyGames.csv"))
mens_game_cities_df = load_data(os.path.join(DATA_PATH, "MGameCities.csv"))
mens_massey_ordinals_df = load_data(os.path.join(DATA_PATH, "MMasseyOrdinals.csv"))
mens_ncaa_tourney_compactResults_df = load_data(os.path.join(DATA_PATH, "MNCAATourneyCompactResults.csv"))
mens_ncaa_tourney_detailResults_df = load_data(os.path.join(DATA_PATH, "MNCAATourneyDetailedResults.csv"))
mens_ncaa_tourney_seedRoundSlots_df = load_data(os.path.join(DATA_PATH, "MNCAATourneySeedRoundSlots.csv"))
mens_ncaa_tourney_seeds_df = load_data(os.path.join(DATA_PATH, "MNCAATourneySeeds.csv"))
mens_ncaa_tourney_slots_df = load_data(os.path.join(DATA_PATH, "MNCAATourneySlots.csv"))
mens_regular_season_compactResults_df = load_data(os.path.join(DATA_PATH, "MRegularSeasonCompactResults.csv"))
mens_regular_season_detailResults_df = load_data(os.path.join(DATA_PATH, "MRegularSeasonDetailedResults.csv"))
mens_seasons_df = load_data(os.path.join(DATA_PATH, "MSeasons.csv"))
mens_secondary_tourneyCompactResults_df = load_data(os.path.join(DATA_PATH, "MSecondaryTourneyCompactResults.csv"))
mens_secondary_tourneyTeams_df = load_data(os.path.join(DATA_PATH, "MSecondaryTourneyTeams.csv"))
mens_TeamCoaches_df = load_data(os.path.join(DATA_PATH, "MTeamCoaches.csv"))
mens_TeamConferences_df = load_data(os.path.join(DATA_PATH, "MTeamConferences.csv"))
mens_Teamspellings_df = load_data(os.path.join(DATA_PATH, "MTeamSpellings.csv"), encoding='ISO-8859-1')
mens_teams_df = load_data(os.path.join(DATA_PATH, "MTeams.csv"))
sampleSubmission_df = load_data(os.path.join(DATA_PATH, "SampleSubmissionStage1.csv"))
seedBenchmark_df = load_data(os.path.join(DATA_PATH, "SeedBenchmarkStage1.csv"))
womans_conferences_df = load_data(os.path.join(DATA_PATH, "WConferenceTourneyGames.csv"))
womans_game_cities_df = load_data(os.path.join(DATA_PATH, "WGameCities.csv"))
womans_ncaa_tourney_compactResults_df = load_data(os.path.join(DATA_PATH, "WNCAATourneyCompactResults.csv"))
womans_ncaa_tourney_detailResults_df = load_data(os.path.join(DATA_PATH, "WNCAATourneyDetailedResults.csv"))
womans_ncaa_tourney_seeds_df = load_data(os.path.join(DATA_PATH, "WNCAATourneySeeds.csv"))
womans_ncaa_tourney_slots_df = load_data(os.path.join(DATA_PATH, "WNCAATourneySlots.csv"))
womans_regular_season_compactResults_df = load_data(os.path.join(DATA_PATH, "WRegularSeasonCompactResults.csv"))
womans_regular_season_detailedResults_df = load_data(os.path.join(DATA_PATH, "WRegularSeasonDetailedResults.csv"))
womans_seasons_df = load_data(os.path.join(DATA_PATH, "WSeasons.csv"))
womans_secondary_tourneyCompactResults_df = load_data(os.path.join(DATA_PATH, "WSecondaryTourneyCompactResults.csv"))
womans_secondary_tourney_teams_df = load_data(os.path.join(DATA_PATH, "WSecondaryTourneyTeams.csv"))
womans_teamConferences_df = load_data(os.path.join(DATA_PATH, "WTeamConferences.csv"))
womans_teamSpellings_df = load_data(os.path.join(DATA_PATH, "WTeamSpellings.csv"), encoding='ISO-8859-1')
womans_teams_df = load_data(os.path.join(DATA_PATH, "WTeams.csv"))



import os
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn ML
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import log_loss, roc_auc_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# XGBoost / LightGBM (Boosting)
import xgboost as xgb

# Optional: PyTorch or Keras for neural networks
import torch
import torch.nn as nn
import torch.optim as optim

# If you want a deep learning library: from tensorflow import keras

import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "/kaggle/input/march-machine-learning-mania-2025/"

# Random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# --------------------------------------------------------------------
# (A) Load Basic CSVs: Teams, Seeds, Seasons, Regular/Tourney (Compact)
# --------------------------------------------------------------------
m_teams = pd.read_csv(os.path.join(DATA_PATH, "MTeams.csv"))
w_teams = pd.read_csv(os.path.join(DATA_PATH, "WTeams.csv"))

m_seeds = pd.read_csv(os.path.join(DATA_PATH, "MNCAATourneySeeds.csv"))
w_seeds = pd.read_csv(os.path.join(DATA_PATH, "WNCAATourneySeeds.csv"))

m_reg_compact = pd.read_csv(os.path.join(DATA_PATH, "MRegularSeasonCompactResults.csv"))
w_reg_compact = pd.read_csv(os.path.join(DATA_PATH, "WRegularSeasonCompactResults.csv"))

m_tourn_compact = pd.read_csv(os.path.join(DATA_PATH, "MNCAATourneyCompactResults.csv"))
w_tourn_compact = pd.read_csv(os.path.join(DATA_PATH, "WNCAATourneyCompactResults.csv"))

# --------------------------------------------------------------------
# (B) Load Detailed Box Scores (since 2003 men, 2010 women)
# --------------------------------------------------------------------
m_reg_detailed = pd.read_csv(os.path.join(DATA_PATH, "MRegularSeasonDetailedResults.csv"))
w_reg_detailed = pd.read_csv(os.path.join(DATA_PATH, "WRegularSeasonDetailedResults.csv"))
m_tourn_detailed = pd.read_csv(os.path.join(DATA_PATH, "MNCAATourneyDetailedResults.csv"))
w_tourn_detailed = pd.read_csv(os.path.join(DATA_PATH, "WNCAATourneyDetailedResults.csv"))

# --------------------------------------------------------------------
# (C) Load Conferences, Cities, Massey, etc.
# --------------------------------------------------------------------
conferences_df = pd.read_csv(os.path.join(DATA_PATH, "Conferences.csv"))
m_team_conf = pd.read_csv(os.path.join(DATA_PATH, "MTeamConferences.csv"))
w_team_conf = pd.read_csv(os.path.join(DATA_PATH, "WTeamConferences.csv"))

cities_df = pd.read_csv(os.path.join(DATA_PATH, "Cities.csv"))
m_game_cities = pd.read_csv(os.path.join(DATA_PATH, "MGameCities.csv"))
w_game_cities = pd.read_csv(os.path.join(DATA_PATH, "WGameCities.csv"))

m_massey = pd.read_csv(os.path.join(DATA_PATH, "MMasseyOrdinals.csv"))
w_massey_path = os.path.join(DATA_PATH, "WMasseyOrdinals.csv")
have_w_massey = os.path.exists(w_massey_path)
if have_w_massey:
    w_massey = pd.read_csv(w_massey_path)
else:
    w_massey = None

# Stage 2 sample submission
sample_stage2 = pd.read_csv(os.path.join(DATA_PATH, "SampleSubmissionStage2.csv"))

print("Data loaded successfully.")
print(sample_stage2.head())
print(sample_stage2.shape)


# Example: fill seeds with 25 if missing
def extract_seed_val(seed_str):
    if not isinstance(seed_str, str):
        return np.nan
    digits = "".join(ch for ch in seed_str[1:] if ch.isdigit())
    return int(digits) if digits else np.nan

m_seeds["SeedValue"] = m_seeds["Seed"].apply(extract_seed_val).fillna(25)
w_seeds["SeedValue"] = w_seeds["Seed"].apply(extract_seed_val).fillna(25)
print(m_seeds["SeedValue"].head())


print("MSeeds duplicates:", m_seeds.duplicated().sum())


# Combine men+women detailed for a single pipeline
m_reg_detailed["Gender"]   = "M"
m_tourn_detailed["Gender"] = "M"
w_reg_detailed["Gender"]   = "W"
w_tourn_detailed["Gender"] = "W"

all_detailed = pd.concat([
    m_reg_detailed, m_tourn_detailed,
    w_reg_detailed, w_tourn_detailed
], ignore_index=True)

# Example advanced stats from WFGM, WFGA, etc.
all_detailed["ScoreDiff"] = all_detailed["WScore"] - all_detailed["LScore"]
print(all_detailed["ScoreDiff"].head())


# Merge men’s + women’s conferences
m_team_conf["Gender"] = "M"
w_team_conf["Gender"] = "W"
all_team_conf = pd.concat([m_team_conf, w_team_conf], ignore_index=True)
print(all_team_conf.head())


stats_cols = ["Score","FGM","FGA","FGM3","FGA3","FTM","FTA","OR","DR","Ast","TO","Stl","Blk","PF"]

from collections import defaultdict

team_sum = defaultdict(lambda: {c:0 for c in stats_cols})
team_count = defaultdict(int)

for idx, row in all_detailed.iterrows():
    season = row["Season"]
    wteam = row["WTeamID"]
    lteam = row["LTeamID"]
    
    for c in stats_cols:
        team_sum[(season,wteam)][c]+= row["W"+c]
        team_sum[(season,lteam)][c]+= row["L"+c]
    team_count[(season,wteam)] += 1
    team_count[(season,lteam)] += 1

team_stats = []
for (season, team_id), sums in team_sum.items():
    games = team_count[(season, team_id)]
    row_data = {
        "Season": season,
        "TeamID": team_id
    }
    for c in stats_cols:
        row_data["Avg"+c] = sums[c]/games if games>0 else 0
    team_stats.append(row_data)

team_stats_df = pd.DataFrame(team_stats)
print("team_stats_df shape:", team_stats_df.shape)
print(team_stats_df.head())


m_seeds_df = m_seeds[["Season","TeamID","SeedValue"]].drop_duplicates()
w_seeds_df = w_seeds[["Season","TeamID","SeedValue"]].drop_duplicates()
all_seeds_df = pd.concat([m_seeds_df, w_seeds_df], ignore_index=True)

team_stats_df = team_stats_df.merge(all_seeds_df, how="left", on=["Season","TeamID"])
team_stats_df["SeedValue"] = team_stats_df["SeedValue"].fillna(25)

team_stats_df = team_stats_df.merge(all_team_conf[["Season","TeamID","ConfAbbrev"]], how="left", on=["Season","TeamID"])
team_stats_df["ConfAbbrev"] = team_stats_df["ConfAbbrev"].fillna("NO_CONF")
print(team_stats_df.head())
print(team_stats_df["ConfAbbrev"].head())


m_massey["OrdinalRank"] = m_massey["OrdinalRank"].astype(float)
if have_w_massey:
    w_massey["OrdinalRank"] = w_massey["OrdinalRank"].astype(float)
    combined_massey = pd.concat([m_massey, w_massey], ignore_index=True)
else:
    combined_massey = m_massey.copy()

def final_massey_ratings(df):
    df = df.sort_values("RankingDayNum")
    last_rows = df.groupby(["Season","TeamID","SystemName"], as_index=False).last()
    mean_ranks = last_rows.groupby(["Season","TeamID"], as_index=False)["OrdinalRank"].mean()
    mean_ranks.rename(columns={"OrdinalRank":"MasseyRating"}, inplace=True)
    return mean_ranks

massey_final = final_massey_ratings(combined_massey)
team_stats_df = team_stats_df.merge(massey_final, how="left", on=["Season","TeamID"])
team_stats_df["MasseyRating"] = team_stats_df["MasseyRating"].fillna(0)
print(massey_final.head())
print()
print(team_stats_df.head())
print()
print(team_stats_df["MasseyRating"].head())


plt.figure(figsize=(12,6))
sns.histplot(team_stats_df["AvgScore"], bins=30, kde=True)
plt.title("Distribution of Average Score per Team/Season")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x="SeedValue", y="AvgScore", data=team_stats_df)
plt.title("SeedValue vs. AvgScore (Boxplot)")
plt.show()


features_for_cluster = ["AvgScore","AvgOR","AvgDR","AvgAst","AvgTO","MasseyRating"]
kmeans_data = team_stats_df[features_for_cluster].fillna(0).values

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(kmeans_data)
team_stats_df["ClusterLabel"] = clusters

# Visualize in 2D using PCA
pca = PCA(n_components=2, random_state=42)
reduced = pca.fit_transform(kmeans_data)
plt.figure(figsize=(8,6))
plt.scatter(reduced[:,0], reduced[:,1], c=clusters, cmap="viridis", alpha=0.6)
plt.title("Team Clusters by K-Means (5 clusters)")
plt.show()


def _plot_series(series, series_name, series_index=0):
  palette = list(sns.palettes.mpl_palette('Dark2'))
  counted = (series['Season']
                .value_counts()
              .reset_index(name='counts')
              .rename({'index': 'Season'}, axis=1)
              .sort_values('Season', ascending=True))
  xs = counted['Season']
  ys = counted['counts']
  plt.plot(xs, ys, label=series_name, color=palette[series_index % len(palette)])

fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
df_sorted = mens_conferences_df.sort_values('Season', ascending=True)
_plot_series(df_sorted, '')
sns.despine(fig=fig, ax=ax)
plt.xlabel('Season')
_ = plt.ylabel('count()')


mens_avg_scores = mens_ncaa_tourney_detailResults_df.groupby('Season')[['WScore', 'LScore']].mean().mean(axis=1)

# Calculate average scores for women's teams per season
womans_avg_scores = womans_ncaa_tourney_detailResults_df.groupby('Season')[['WScore', 'LScore']].mean().mean(axis=1)

# Create the line plot
plt.figure(figsize=(10, 6))
sns.lineplot(x=mens_avg_scores.index, y=mens_avg_scores.values, label='Men')
sns.lineplot(x=womans_avg_scores.index, y=womans_avg_scores.values, label='Women')
plt.title('Average Scores of Men\'s and Women\'s Teams Over Time')
plt.xlabel('Season')
plt.ylabel('Average Score')
plt.legend()
plt.show()


mens_scores = mens_ncaa_tourney_detailResults_df[['WScore', 'LScore']].melt(var_name='Team', value_name='Score')
mens_scores['Gender'] = 'Men'
womans_scores = womans_ncaa_tourney_detailResults_df[['WScore', 'LScore']].melt(var_name='Team', value_name='Score')
womans_scores['Gender'] = 'Women'
all_scores = pd.concat([mens_scores, womans_scores])

# Create the plot
plt.figure(figsize=(10, 6))
sns.histplot(data=all_scores, x='Score', hue='Gender', kde=True, element='step')
plt.title('Distribution of Men\'s and Women\'s Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()


def display_heads(dfs_dict):

    for name, df in dfs_dict.items():
        print(f"DataFrame: {name}")
        print(df.head(), "\n")
dfs = {
    'cities_df': cities_df,
    'conferences_df': conferences_df,
    'mens_conferences_df': mens_conferences_df,
    'mens_game_cities_df': mens_game_cities_df,
    'mens_massey_ordinals_df': mens_massey_ordinals_df,
    'mens_ncaa_tourney_compactResults_df': mens_ncaa_tourney_compactResults_df,
    'mens_ncaa_tourney_detailResults_df': mens_ncaa_tourney_detailResults_df,
    'mens_ncaa_tourney_seedRoundSlots_df': mens_ncaa_tourney_seedRoundSlots_df,
    'mens_ncaa_tourney_seeds_df': mens_ncaa_tourney_seeds_df,
    'mens_ncaa_tourney_slots_df': mens_ncaa_tourney_slots_df,
    'mens_regular_season_compactResults_df': mens_regular_season_compactResults_df,
    'mens_regular_season_detailResults_df': mens_regular_season_detailResults_df,
    'mens_seasons_df': mens_seasons_df,
    'mens_secondary_tourneyCompactResults_df': mens_secondary_tourneyCompactResults_df,
    'mens_secondary_tourneyTeams_df': mens_secondary_tourneyTeams_df,
    'mens_TeamCoaches_df': mens_TeamCoaches_df,
    'mens_TeamConferences_df': mens_TeamConferences_df,
    'mens_Teamspellings_df': mens_Teamspellings_df,
    'mens_teams_df': mens_teams_df,
    'sampleSubmission_df': sampleSubmission_df,
    'seedBenchmark_df': seedBenchmark_df,
    'womans_conferences_df': womans_conferences_df,
    'womans_game_cities_df': womans_game_cities_df,
    'womans_ncaa_tourney_compactResults_df': womans_ncaa_tourney_compactResults_df,
    'womans_ncaa_tourney_detailResults_df': womans_ncaa_tourney_detailResults_df,
    'womans_ncaa_tourney_seeds_df': womans_ncaa_tourney_seeds_df,
    'womans_ncaa_tourney_slots_df': womans_ncaa_tourney_slots_df,
    'womans_regular_season_compactResults_df': womans_regular_season_compactResults_df,
    'womans_regular_season_detailedResults_df': womans_regular_season_detailedResults_df,
    'womans_seasons_df': womans_seasons_df,
    'womans_secondary_tourneyCompactResults_df': womans_secondary_tourneyCompactResults_df,
    'womans_secondary_tourney_teams_df': womans_secondary_tourney_teams_df,
    'womans_teamConferences_df': womans_teamConferences_df,
    'womans_teamSpellings_df': womans_teamSpellings_df,
    'womans_teams_df': womans_teams_df
}
display_heads(dfs)


def display_numeric_descriptive_stats(dataframes):
    """
    Displays descriptive statistics for numeric columns in a dictionary of DataFrames.

    Args:
        dataframes (dict): A dictionary where keys are names and values are pandas DataFrames.
    """
    for key, df in dataframes.items():
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if numeric_cols:  # Equivalent to numeric_cols.size > 0
            print(f"\n=== Descriptive Statistics for {key} (Numeric Columns) ===")
            display(df[numeric_cols].describe())

    print("\nDescriptive statistics generated for all numeric columns.")
# Call the function with your dictionary of DataFrames
display_numeric_descriptive_stats(dfs)


train_rows = []
labels = []

for idx, row in all_detailed.iterrows():
    season = row["Season"]
    wteam = row["WTeamID"]
    lteam = row["LTeamID"]
    diff  = row["ScoreDiff"]
    
    # label=1 => T1=winner
    train_rows.append([season, wteam, lteam, diff])
    labels.append(1)
    # mirror => label=0
    train_rows.append([season, lteam, wteam, -diff])
    labels.append(0)

train_df = pd.DataFrame(train_rows, columns=["Season","Team1","Team2","ScoreDiff"])
labels = np.array(labels)
print(train_df.head)


# Merge T1 stats
train_df = train_df.merge(team_stats_df, how="left", 
                          left_on=["Season","Team1"], 
                          right_on=["Season","TeamID"]).drop(columns=["TeamID"])
train_df.rename(columns={
    c: f"T1_{c}" for c in team_stats_df.columns if c not in ["Season","TeamID"]
}, inplace=True)

# Merge T2 stats
train_df = train_df.merge(team_stats_df, how="left", 
                          left_on=["Season","Team2"], 
                          right_on=["Season","TeamID"]).drop(columns=["TeamID"])
train_df.rename(columns={
    c: f"T2_{c}" for c in team_stats_df.columns if c not in ["Season","TeamID"]
}, inplace=True)

for c in stats_cols:
    train_df[f"Diff_Avg{c}"] = train_df[f"T1_Avg{c}"].fillna(0) - train_df[f"T2_Avg{c}"].fillna(0)

train_df["Diff_SeedVal"] = train_df["T1_SeedValue"] - train_df["T2_SeedValue"]
train_df["Diff_Massey"]  = train_df["T1_MasseyRating"] - train_df["T2_MasseyRating"]

# >>> NEW: Create an interaction feature combining seed difference and Massey rating difference.
train_df["Diff_Seed_Massey"] = train_df["Diff_SeedVal"] * train_df["Diff_Massey"]

# For future matchups, ScoreDiff remains unknown so it will be set to 0 later.
feature_cols = [col for col in train_df.columns if col.startswith("Diff_")] + ["ScoreDiff"]

X_all = train_df[feature_cols].fillna(0).values
y_all = labels

# -----------------------------
# 5.3. Train-Test Split, Scale, and Model Training
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.1, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

def compute_logloss(y_true, y_prob):
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

# 1) Logistic Regression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train_scaled, y_train)
val_preds_lr = model_lr.predict_proba(X_val_scaled)[:, 1]
lr_logloss = compute_logloss(y_val, val_preds_lr)

# 2) Random Forest
model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model_rf.fit(X_train_scaled, y_train)
val_preds_rf = model_rf.predict_proba(X_val_scaled)[:, 1]
rf_logloss = compute_logloss(y_val, val_preds_rf)

# 3) XGBoost with updated hyperparameters (learning_rate=0.1)
model_xgb = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, n_jobs=-1)
model_xgb.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], 
              eval_metric="logloss", early_stopping_rounds=10, verbose=False)
val_preds_xgb = model_xgb.predict_proba(X_val_scaled)[:, 1]
xgb_logloss = compute_logloss(y_val, val_preds_xgb)

print("Validation Log Losses:")
print(f"  LogisticRegression: {lr_logloss:.4f}")
print(f"  RandomForest:       {rf_logloss:.4f}")
print(f"  XGBoost:            {xgb_logloss:.4f}")

# -----------------------------
# 5.3.2. Ensemble – Re-weighted Average
# -----------------------------
# Since XGBoost is performing best (lowest logloss), assign it a higher weight.
# Adjust these weights based on your validation experiments.
preds_lr  = model_lr.predict_proba(X_val_scaled)[:, 1]
preds_rf  = model_rf.predict_proba(X_val_scaled)[:, 1]
preds_xgb = model_xgb.predict_proba(X_val_scaled)[:, 1]

ensemble_val = 0.25 * preds_lr + 0.15 * preds_rf + 0.60 * preds_xgb
ensemble_logloss = compute_logloss(y_val, ensemble_val)
print(f"Ensemble Validation LogLoss (re-weighted): {ensemble_logloss:.4f}")

# -----------------------------
# 6. Final Submission: Applying Changes to Test Data
# -----------------------------
# (Assuming sample_stage2 has been loaded and merged as in your code)

# (1) Parse Season, T1, and T2 from submission IDs (same as before)
def parse_id(id_str):
    parts = id_str.split("_")
    season = int(parts[0])
    t1 = int(parts[1])
    t2 = int(parts[2])
    return season, t1, t2

sample_stage2["Season"], sample_stage2["T1"], sample_stage2["T2"] = zip(*sample_stage2["ID"].apply(parse_id))

# (2) Merge T1 stats and rename columns (as before)
sample_stage2 = sample_stage2.merge(team_stats_df, how="left", left_on=["Season", "T1"], right_on=["Season", "TeamID"])
cols = [col for col in team_stats_df.columns if col not in ["Season", "TeamID"]]
for col in cols:
    sample_stage2.rename(columns={col: "T1_" + col}, inplace=True)
if "TeamID" in sample_stage2.columns:
    sample_stage2.drop("TeamID", axis=1, inplace=True)

# (3) Merge T2 stats and rename columns (as before)
sample_stage2 = sample_stage2.merge(team_stats_df, how="left", left_on=["Season", "T2"], right_on=["Season", "TeamID"])
cols = [col for col in team_stats_df.columns if col not in ["Season", "TeamID"]]
for col in cols:
    sample_stage2.rename(columns={col: "T2_" + col}, inplace=True)
if "TeamID" in sample_stage2.columns:
    sample_stage2.drop("TeamID", axis=1, inplace=True)

# (4) Build difference features (including the new interaction feature)
for stat in stats_cols:
    t1_stat = sample_stage2[f"T1_Avg{stat}"]
    if isinstance(t1_stat, pd.DataFrame):
         t1_stat = t1_stat.iloc[:, 0]
    t2_stat = sample_stage2[f"T2_Avg{stat}"]
    if isinstance(t2_stat, pd.DataFrame):
         t2_stat = t2_stat.iloc[:, 0]
    sample_stage2[f"Diff_Avg{stat}"] = t1_stat.fillna(0) - t2_stat.fillna(0)

t1_seed = sample_stage2["T1_SeedValue"]
if isinstance(t1_seed, pd.DataFrame):
    t1_seed = t1_seed.iloc[:, 0]
t2_seed = sample_stage2["T2_SeedValue"]
if isinstance(t2_seed, pd.DataFrame):
    t2_seed = t2_seed.iloc[:, 0]
sample_stage2["Diff_SeedVal"] = t1_seed.fillna(25) - t2_seed.fillna(25)

t1_massey = sample_stage2["T1_MasseyRating"]
if isinstance(t1_massey, pd.DataFrame):
    t1_massey = t1_massey.iloc[:, 0]
t2_massey = sample_stage2["T2_MasseyRating"]
if isinstance(t2_massey, pd.DataFrame):
    t2_massey = t2_massey.iloc[:, 0]
sample_stage2["Diff_Massey"] = t1_massey.fillna(0) - t2_massey.fillna(0)

# >>> NEW: Create the interaction term in test data as well.
sample_stage2["Diff_Seed_Massey"] = sample_stage2["Diff_SeedVal"] * sample_stage2["Diff_Massey"]

# For ScoreDiff, which is unknown for future matchups, set to 0
sample_stage2["ScoreDiff"] = 0

final_feature_cols = [col for col in train_df.columns if col.startswith("Diff_")] + ["ScoreDiff"]
X_test = sample_stage2[final_feature_cols].fillna(0).values
X_test_scaled = scaler.transform(X_test)

# (5) Generate predictions using the trained models (with the same weights as in validation)
preds_lr_test  = model_lr.predict_proba(X_test_scaled)[:, 1]
preds_rf_test  = model_rf.predict_proba(X_test_scaled)[:, 1]
preds_xgb_test = model_xgb.predict_proba(X_test_scaled)[:, 1]

ensemble_preds = 0.25 * preds_lr_test + 0.15 * preds_rf_test + 0.60 * preds_xgb_test
ensemble_preds = np.clip(ensemble_preds, 0.025, 0.975)

final_submission = sample_stage2[["ID"]].copy()
final_submission["Pred"] = ensemble_preds
final_submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created successfully!")
print("Final submission shape:", final_submission.shape)
print(final_submission.head(10))




