import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_absolute_error, brier_score_loss
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")



# 1. Load Dataset
path = "/kaggle/input/march-machine-learning-mania-2025/**"
data = {p.split("/")[-1].split(".")[0]: pd.read_csv(p, encoding="latin-1") for p in glob.glob(path)}

# Merge Menâ€™s & Womenâ€™s Data
teams = pd.concat([data["MTeams"], data["WTeams"]])
season_dresults = pd.concat([data["MRegularSeasonDetailedResults"], data["WRegularSeasonDetailedResults"]])
tourney_dresults = pd.concat([data["MNCAATourneyDetailedResults"], data["WNCAATourneyDetailedResults"]])
seeds = pd.concat([data["MNCAATourneySeeds"], data["WNCAATourneySeeds"]])

# Create a mapping for seeds
seeds_dict = {"_".join(map(str, [int(k1), k2])): int(v[1:3]) for k1, v, k2 in seeds[["Season", "Seed", "TeamID"]].values}




# 2. Feature Engineering
season_dresults["ID"] = season_dresults.apply(lambda r: "_".join(map(str, [r["Season"]] + sorted([r["WTeamID"], r["LTeamID"]]))), axis=1)
season_dresults["SeedDiff"] = season_dresults.apply(lambda r: seeds_dict.get(f"{r['Season']}_{r['WTeamID']}", 0) - seeds_dict.get(f"{r['Season']}_{r['LTeamID']}", 0), axis=1)
season_dresults["ScoreDiff"] = season_dresults["WScore"] - season_dresults["LScore"]
season_dresults["WinFlag"] = season_dresults.apply(lambda r: 1 if r["WTeamID"] < r["LTeamID"] else 0, axis=1)

# NEW: Aggregating Past Team Performance
game_stats = season_dresults.groupby("ID").agg(
    AvgScoreDiff=("ScoreDiff", "mean"),
    MaxScoreDiff=("ScoreDiff", "max"),
    MinScoreDiff=("ScoreDiff", "min"),
    WinRate=("WinFlag", "mean"),  # How often the lower-seed team wins
).reset_index()

# Compact Results for Additional Team Statistics
season_cresults = pd.concat([data["MRegularSeasonCompactResults"], data["WRegularSeasonCompactResults"]])
team_wins = season_cresults.groupby("WTeamID").size().reset_index(name="TotalWins")
teams = teams.merge(team_wins, left_on="TeamID", right_on="WTeamID", how="left").fillna(0)

# Merge Compact Results (Team Wins) into Main Dataset
season_dresults = season_dresults.merge(game_stats, on="ID", how="left")
season_dresults = season_dresults.merge(teams[["TeamID", "TotalWins"]], left_on="WTeamID", right_on="TeamID", how="left").fillna(0)

print(teams.columns)



# Final Feature List (Including TotalWins)**
features = ["SeedDiff", "AvgScoreDiff", "MaxScoreDiff", "MinScoreDiff", "WinRate", "TotalWins"]

# Select Features & Target
X = season_dresults[features].fillna(0)
y = season_dresults["WinFlag"]


# 3. Data Preprocessing
imputer = SimpleImputer(strategy="mean")
scaler = MinMaxScaler()

X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)


print("Target variable distribution:\n", y.value_counts(normalize=True))


from sklearn.model_selection import train_test_split

# Split Data into Training & Validation Set
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)



xgb = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    device="gpu",
    random_state=42
)
xgb.fit(X_scaled, y)



# Make predictions on the training set
train_pred = xgb.predict(X_scaled).clip(0.001, 0.999)

# Calculate evaluation metrics
print(f'Log Loss: {log_loss(y, train_pred)}')
print(f'Mean Absolute Error: {mean_absolute_error(y, train_pred)}')
print(f'Brier Score: {brier_score_loss(y, train_pred)}')

cv_mse_scores = cross_val_score(xgb, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
print(f'Cross-validated MSE: {-cv_mse_scores.mean()}')


# Load Submission File
sub = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv")

# Extract Season & Teams from ID
sub["Season"] = sub["ID"].apply(lambda x: int(x.split("_")[0]))
sub["Team1"] = sub["ID"].apply(lambda x: int(x.split("_")[1]))
sub["Team2"] = sub["ID"].apply(lambda x: int(x.split("_")[2]))

# Compute Seed Difference
sub["SeedDiff"] = sub.apply(
    lambda r: seeds_dict.get(f"{r['Season']}_{r['Team1']}", 0) - seeds_dict.get(f"{r['Season']}_{r['Team2']}", 0),
    axis=1
)

# Merge Total Wins for Both Teams
sub = sub.merge(teams[["TeamID", "TotalWins"]].rename(columns={"TotalWins": "TotalWins_T1"}), 
                left_on="Team1", right_on="TeamID", how="left").fillna(0)

sub = sub.merge(teams[["TeamID", "TotalWins"]].rename(columns={"TotalWins": "TotalWins_T2"}), 
                left_on="Team2", right_on="TeamID", how="left").fillna(0)

# Drop Duplicate TeamID Columns (if exist)
sub = sub.drop(columns=["TeamID_x", "TeamID_y"], errors="ignore")

# Merge Historical Game Stats (Ensure correct ID format)
sub = sub.merge(game_stats, on="ID", how="left").fillna(0)

# Create a single TotalWins column to match training data
sub["TotalWins"] = sub.apply(lambda r: r["TotalWins_T1"] if r["Team1"] < r["Team2"] else r["TotalWins_T2"], axis=1)

# Ensure Feature Set Matches Training Data
model_features = ['SeedDiff', 'AvgScoreDiff', 'MaxScoreDiff', 'MinScoreDiff', 'WinRate', 'TotalWins']

# DEBUGGING: Print feature alignment check
print("ğŸ”� Features used during training:", model_features)
print("ğŸ”� Features available in submission data:", list(sub.columns))
print(f"ğŸ”� Expected Feature Count: {len(model_features)}, Found: {len(sub[model_features].columns)}")

# Select the Exact Features for Prediction
sub_X = sub[model_features].fillna(0)

# Transform Submission Data using pre-fitted imputer & scaler
sub_X_imputed = imputer.transform(sub_X)
sub_X_scaled = scaler.transform(sub_X_imputed)

# Make Predictions Using Trained Model
sub["Pred"] = xgb.predict(sub_X_scaled).clip(0.001, 0.999)

# Save Final Submission File
sub[["ID", "Pred"]].to_csv("submission.csv", index=False)

# Display First Few Predictions
print("âœ… Submission File Created Successfully!")
print(sub[["ID", "Pred"]].head())



submission = pd.read_csv("/kaggle/working/submission.csv")
submission.head()


submission['Pred'].unique()













