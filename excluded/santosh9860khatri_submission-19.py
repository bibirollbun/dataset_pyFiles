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
# Similarly for other data frames...


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

# Build difference features
for c in stats_cols:
    train_df[f"Diff_Avg{c}"] = train_df[f"T1_Avg{c}"].fillna(0) - train_df[f"T2_Avg{c}"].fillna(0)

train_df["Diff_SeedVal"] = train_df["T1_SeedValue"] - train_df["T2_SeedValue"]
train_df["Diff_Massey"]  = train_df["T1_MasseyRating"] - train_df["T2_MasseyRating"]

# ScoreDiff is from the actual game. For future matchups, it is unknown => 0
feature_cols = [col for col in train_df.columns if col.startswith("Diff_")] + ["ScoreDiff"]

X_all = train_df[feature_cols].fillna(0).values
y_all = labels


X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, 
                                                  test_size=0.1, 
                                                  random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

def compute_logloss(y_true, y_prob):
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1-eps)
    return -np.mean(y_true*np.log(y_prob) + (1-y_true)*np.log(1-y_prob))




# 1) Logistic Regression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train_scaled, y_train)
val_preds_lr = model_lr.predict_proba(X_val_scaled)[:,1]
lr_logloss = compute_logloss(y_val, val_preds_lr)




# 2) Random Forest
model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model_rf.fit(X_train_scaled, y_train)
val_preds_rf = model_rf.predict_proba(X_val_scaled)[:,1]
rf_logloss = compute_logloss(y_val, val_preds_rf)




#3) SVM (with probability=True to get predict_proba)
#model_svm = SVC(probability=True, kernel="rbf", random_state=42)
#model_svm.fit(X_train_scaled, y_train)
#val_preds_svm = model_svm.predict_proba(X_val_scaled)[:,1]
#svm_logloss = compute_logloss(y_val, val_preds_svm)




# 4) XGBoost
model_xgb = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, 
                              max_depth=6, subsample=0.8, colsample_bytree=0.8,
                              random_state=42, n_jobs=-1)
model_xgb.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled,y_val)],
              eval_metric="logloss", early_stopping_rounds=10, verbose=False)
val_preds_xgb = model_xgb.predict_proba(X_val_scaled)[:,1]
xgb_logloss = compute_logloss(y_val, val_preds_xgb)




print("Validation Log Losses:")
print(f"  LogisticRegression: {lr_logloss:.4f}")
print(f"  RandomForest:       {rf_logloss:.4f}")
#print(f"  SVM:                {svm_logloss:.4f}")
print(f"  XGBoost:            {xgb_logloss:.4f}")



models = ["LR","RF","XGB"]
scores = [lr_logloss, rf_logloss, xgb_logloss]

plt.figure(figsize=(6,4))
sns.barplot(x=models, y=scores)
plt.title("Comparison of Validation LogLoss")
plt.ylabel("LogLoss")
plt.show()


# Example: use ScoreDiff as a numeric target
# Then you'd do something like a RandomForestRegressor or LinearRegression
model_linreg = LinearRegression()
model_linreg.fit(X_train_scaled, y_train)  # But here y_train are 0/1. 
# Actually you'd set y_train=ScoreDiff in a different pipeline


# We'll create a small dataset wrapper & simple feedforward
class NCAAData(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = NCAAData(X_train_scaled, y_train)
val_dataset   = NCAAData(X_val_scaled, y_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

# Simple feedforward net
class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

net = SimpleNet(input_dim=X_train_scaled.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train loop
EPOCHS=5
for epoch in range(EPOCHS):
    net.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        preds = net(batch_x).squeeze()
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
    # Evaluate on val
    net.eval()
    val_losses=[]
    with torch.no_grad():
        for vx, vy in val_loader:
            vpred = net(vx).squeeze()
            vloss = criterion(vpred, vy)
            val_losses.append(vloss.item())
    print(f"Epoch {epoch+1}/{EPOCHS}, val_loss={np.mean(val_losses):.4f}")



val_probs_lr  = model_lr.predict_proba(X_val_scaled)[:,1]
val_probs_rf  = model_rf.predict_proba(X_val_scaled)[:,1]
#val_probs_svm = model_svm.predict_proba(X_val_scaled)[:,1]
val_probs_xgb = model_xgb.predict_proba(X_val_scaled)[:,1]

ensemble_probs = (val_probs_lr + val_probs_rf + val_probs_xgb )/3
ensemble_logloss = compute_logloss(y_val, ensemble_probs)
print(f"Ensemble LogLoss (avg): {ensemble_logloss:.4f}")


pca = PCA(n_components=2)
pca_feats = pca.fit_transform(X_val_scaled)
plt.figure(figsize=(8,6))
plt.scatter(pca_feats[:,0], pca_feats[:,1], c=y_val, cmap="coolwarm", alpha=0.4)
plt.title("PCA visualization of validation matchups")
plt.show()


tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_feats = tsne.fit_transform(X_val_scaled)
plt.figure(figsize=(8,6))
plt.scatter(tsne_feats[:,0], tsne_feats[:,1], c=y_val, cmap="coolwarm", alpha=0.4)
plt.title("t-SNE of validation matchups")
plt.show()


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_logloss = []

for train_idx, test_idx in skf.split(X_all, y_all):
    X_tr, X_ts = X_all[train_idx], X_all[test_idx]
    y_tr, y_ts = y_all[train_idx], y_all[test_idx]
    
    # scale each fold
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_ts_sc = sc.transform(X_ts)
    
    model_cv = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05,
                                 max_depth=6, random_state=42)
    model_cv.fit(X_tr_sc, y_tr)
    preds_ts = model_cv.predict_proba(X_ts_sc)[:,1]
    fold_logloss = compute_logloss(y_ts, preds_ts)
    cv_logloss.append(fold_logloss)

print("5-Fold CV LogLoss:", np.mean(cv_logloss), "+/-", np.std(cv_logloss))


from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "n_estimators": [100,200,300],
    "learning_rate": [0.01,0.05,0.1],
    "max_depth": [4,6,8],
    "subsample": [0.8,1.0]
}
model_xgb_cv = xgb.XGBClassifier(random_state=42)
search = RandomizedSearchCV(model_xgb_cv, param_distributions=param_grid,
                            n_iter=10, scoring="neg_log_loss",
                            cv=3, verbose=1, random_state=42)
search.fit(X_train_scaled, y_train)
print("Best params:", search.best_params_)
print("Best score:", search.best_score_)


# Example SHAP usage for XGBoost
import shap
explainer = shap.TreeExplainer(model_xgb)
shap_vals = explainer.shap_values(X_val_scaled)
shap.summary_plot(shap_vals, X_val_scaled, feature_names=feature_cols)


# -------------------------------
# FINAL SUBMISSION CODE (Remaining Steps Only)
# -------------------------------

# (1) Helper to parse submission IDs (if not already defined)
def parse_id(id_str):
    parts = id_str.split("_")
    season = int(parts[0])
    t1 = int(parts[1])
    t2 = int(parts[2])
    return season, t1, t2

# (2) Parse Season, T1, and T2 from the sample submission file
sample_stage2["Season"], sample_stage2["T1"], sample_stage2["T2"] = zip(*sample_stage2["ID"].apply(parse_id))

# (3) Merge team stats for T1 (if not already merged)
sample_stage2 = sample_stage2.merge(
    team_stats_df, how="left", left_on=["Season", "T1"], right_on=["Season", "TeamID"]
)
# Rename merged columns for T1 (except "Season" and "TeamID")
cols = [col for col in team_stats_df.columns if col not in ["Season", "TeamID"]]
for col in cols:
    sample_stage2.rename(columns={col: "T1_" + col}, inplace=True)
if "TeamID" in sample_stage2.columns:
    sample_stage2.drop("TeamID", axis=1, inplace=True)

# (4) Merge team stats for T2
sample_stage2 = sample_stage2.merge(
    team_stats_df, how="left", left_on=["Season", "T2"], right_on=["Season", "TeamID"]
)
cols = [col for col in team_stats_df.columns if col not in ["Season", "TeamID"]]
for col in cols:
    sample_stage2.rename(columns={col: "T2_" + col}, inplace=True)
if "TeamID" in sample_stage2.columns:
    sample_stage2.drop("TeamID", axis=1, inplace=True)

# (5) Create difference features
# For each stat in stats_cols, ensure the merged columns are Series before subtracting
for stat in stats_cols:
    t1_stat = sample_stage2[f"T1_Avg{stat}"]
    if isinstance(t1_stat, pd.DataFrame):
         t1_stat = t1_stat.iloc[:, 0]
    t2_stat = sample_stage2[f"T2_Avg{stat}"]
    if isinstance(t2_stat, pd.DataFrame):
         t2_stat = t2_stat.iloc[:, 0]
    sample_stage2[f"Diff_Avg{stat}"] = t1_stat.fillna(0) - t2_stat.fillna(0)

# For seed values:
t1_seed = sample_stage2["T1_SeedValue"]
if isinstance(t1_seed, pd.DataFrame):
    t1_seed = t1_seed.iloc[:, 0]
t2_seed = sample_stage2["T2_SeedValue"]
if isinstance(t2_seed, pd.DataFrame):
    t2_seed = t2_seed.iloc[:, 0]
sample_stage2["Diff_SeedVal"] = t1_seed.fillna(25) - t2_seed.fillna(25)

# For Massey ratings:
t1_massey = sample_stage2["T1_MasseyRating"]
if isinstance(t1_massey, pd.DataFrame):
    t1_massey = t1_massey.iloc[:, 0]
t2_massey = sample_stage2["T2_MasseyRating"]
if isinstance(t2_massey, pd.DataFrame):
    t2_massey = t2_massey.iloc[:, 0]
sample_stage2["Diff_Massey"] = t1_massey.fillna(0) - t2_massey.fillna(0)

# Set ScoreDiff to 0 (unknown for future matchups)
sample_stage2["ScoreDiff"] = 0

# (6) Build test feature matrix.
# Use the same order as in training. In your training you used:
# feature_cols = [col for col in train_df.columns if col.startswith("Diff_")] + ["ScoreDiff"]
final_feature_cols = [col for col in train_df.columns if col.startswith("Diff_")] + ["ScoreDiff"]
X_test = sample_stage2[final_feature_cols].fillna(0).values

# (7) Scale test features using the same scaler fitted during training
X_test_scaled = scaler.transform(X_test)

# (8) Generate predictions using your trained models (ensemble of LR, RF, and XGBoost)
preds_lr  = model_lr.predict_proba(X_test_scaled)[:, 1]
preds_rf  = model_rf.predict_proba(X_test_scaled)[:, 1]
preds_xgb = model_xgb.predict_proba(X_test_scaled)[:, 1]
ensemble_preds = (preds_lr + preds_rf + preds_xgb) / 3.0
ensemble_preds = np.clip(ensemble_preds, 0.025, 0.975)  # optional clipping

# (9) Create and save the final submission file
final_submission = sample_stage2[["ID"]].copy()
final_submission["Pred"] = ensemble_preds
final_submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created successfully!")



print("Final submission shape:", final_submission.shape)
print(final_submission.head(10))


from IPython.display import FileLink
# Display a link to download the submission file
FileLink("submission.csv")


