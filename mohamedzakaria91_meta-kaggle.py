import kagglehub

# Download latest version
path = kagglehub.dataset_download("kaggle/meta-kaggle")

print("Path to dataset files:", path)
import kagglehub

# Download latest version
path = kagglehub.dataset_download("kaggle/meta-kaggle-code")

print("Path to dataset files:", path)



# ğŸ“Š What Makes a Kaggle Notebook Popular?
# ----------------------------------------------------------
# Meta-Kaggle Hackathon 2025 | Mohamed Zakaria

# ğŸ“¦ 1. Imports & Data Loading
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
import lightgbm as lgb
import shap
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

# âœ… Efficient, professional loading (updated with correct filenames)
DATA_DIR = "/kaggle/input/meta-kaggle"
REQUIRED_FILES = {
    "Notebooks": "Kernels.csv",
    "NotebookVersions": "KernelVersions.csv",
    "NotebookTags": "KernelTags.csv",
    "Users": "Users.csv",
    "Competitions": "Competitions.csv",
    "Submissions": "Submissions.csv"
}

data = {}
for name, filename in REQUIRED_FILES.items():
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"âš ï¸�  {filename} exists but is empty.")
            else:
                data[name] = df
                print(f"âœ… Loaded {name} ({df.shape[0]:,} rows)")
        except pd.errors.EmptyDataError:
            print(f"âš ï¸�  {filename} is empty or malformed.")
    else:
        print(f"â�Œ {filename} not found. Please add it from the 'Add Data' panel.")

# ğŸŒŸ Unpack DataFrames
notebooks = data["Notebooks"]
users = data.get("Users")
submissions = data.get("Submissions")
notebook_versions = data.get("NotebookVersions")

# ğŸ§¹ Feature Engineering + Merging
users['PerformanceTier'] = users['PerformanceTier'].fillna(0)
users['RegisterDate'] = pd.to_datetime(users['RegisterDate'])
notebooks['CreationDate'] = pd.to_datetime(notebooks['CreationDate'])

merged_df = notebooks.merge(users, left_on='AuthorUserId', right_on='Id', how='left', suffixes=('', '_User'))
merged_df['TenureDays'] = (merged_df['CreationDate'] - merged_df['RegisterDate']).dt.days
merged_df['TotalActivities'] = 1
merged_df['CreationHour'] = merged_df['CreationDate'].dt.hour
merged_df['WeekendCreation'] = merged_df['CreationDate'].dt.dayofweek >= 5
merged_df['ActivityRate'] = merged_df['TotalActivities'] / (merged_df['TenureDays'] + 30)
merged_df['HighPerformer'] = (merged_df['PerformanceTier'] >= 4).astype(int)
merged_df['RecentActivity'] = (merged_df['TenureDays'] < 7).astype(int)
merged_df['PeakHour'] = merged_df['CreationHour'].between(12, 18).astype(int)
merged_df['EliteUser'] = (merged_df['PerformanceTier'] >= 5).astype(int)

# ğŸ“ˆ Target Variable from Submissions (via KernelVersionId â†’ KernelId)
if submissions is not None and "PublicScoreLeaderboardDisplay" in submissions.columns and notebook_versions is not None:
    if "KernelCurrentVersionId" in notebooks.columns:
        version_map = notebooks[["Id", "KernelCurrentVersionId"]].rename(columns={"Id": "NotebookId", "KernelCurrentVersionId": "KernelVersionId"})
        notebook_versions = notebook_versions.rename(columns={"Id": "KernelVersionId"})
        version_map = version_map.merge(notebook_versions[["KernelVersionId", "KernelId"]], on="KernelVersionId", how="left")
        submissions = submissions.merge(version_map, on="KernelVersionId", how="left")
        valid_scores = submissions.dropna(subset=["PublicScoreLeaderboardDisplay", "KernelId"])
        valid_scores = valid_scores[valid_scores["PublicScoreLeaderboardDisplay"] > 0]
        leaderboard_scores = valid_scores.groupby("KernelId")["PublicScoreLeaderboardDisplay"].mean().reset_index()
        leaderboard_scores.columns = ["KernelId", "LeaderboardScore"]
        merged_df = merged_df.merge(leaderboard_scores, left_on="Id", right_on="KernelId", how="left")
        merged_df = merged_df.dropna(subset=["LeaderboardScore"])
        merged_df['LogScore'] = np.log1p(merged_df['LeaderboardScore'])
        y = merged_df["LogScore"]
        print("âœ… Using log leaderboard score as target")
    else:
        y = np.random.rand(len(merged_df)) * 10
        print("âš ï¸� Missing KernelCurrentVersionId â€” using fallback target")
else:
    y = np.random.rand(len(merged_df)) * 10
    print("âš ï¸� No valid leaderboard score found, using random target")

# ğŸ§  Feature Selection
features = [
    'TenureDays', 'PerformanceTier', 'TotalActivities',
    'CreationHour', 'WeekendCreation', 'ActivityRate',
    'HighPerformer', 'RecentActivity', 'PeakHour', 'EliteUser'
]
X = merged_df[features].fillna(0)

# ğŸ§ª Time-Based Validation
latest_date = merged_df['CreationDate'].max()
val_cutoff = latest_date - pd.Timedelta(days=60)
train_mask = merged_df['CreationDate'] < val_cutoff
val_mask = ~train_mask

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]

# ğŸ”� Model Ensembling (LightGBM + RandomForest + Ridge + GradientBoosting)
lgb_model = lgb.LGBMRegressor(**{
    'objective': 'regression',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbosity': -1,
    'random_state': 42
}, n_estimators=1000)
lgb_model.fit(X_train, y_train)
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.01, random_state=42)
gb_model.fit(X_train, y_train)

# ğŸ§® Averaging Predictions
y_pred_ensemble = (lgb_model.predict(X_val) + rf_model.predict(X_val) + ridge_model.predict(X_val) + gb_model.predict(X_val)) / 4

# ğŸ“‰ Metrics
rmse = mean_squared_error(y_val, y_pred_ensemble, squared=False)
mae = mean_absolute_error(y_val, y_pred_ensemble)
r2 = r2_score(y_val, y_pred_ensemble)

print(f"\nâœ… RMSE: {rmse:.4f} (Log Leaderboard Score)")
print(f"âœ… MAE: {mae:.4f}")
print(f"âœ… RÂ²: {r2:.4f}")

# ğŸ”� SHAP Explainability
explainer = shap.Explainer(lgb_model)
shap_values = explainer(X_val)
shap.plots.beeswarm(shap_values)

# ğŸ“ˆ Simulated Leaderboard Ranking Distribution
merged_df['PredictedLogScore'] = lgb_model.predict(X)
merged_df['PredictedScore'] = np.expm1(merged_df['PredictedLogScore'])
merged_df['SimulatedRank'] = merged_df['PredictedScore'].rank(ascending=False)

plt.figure(figsize=(10, 5))
sns.histplot(merged_df['SimulatedRank'], bins=50, kde=True)
plt.title("ğŸ�� Simulated Leaderboard Rank Distribution")
plt.xlabel("Rank")
plt.ylabel("Notebook Count")
plt.show()

# ğŸ”� Interaction Analysis
interaction_matrix = pd.DataFrame(shap_values.values, columns=X_val.columns)
correlation = interaction_matrix.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("ğŸ“Œ SHAP Interaction Feature Correlations")
plt.show()

# ğŸ’¾ Export Results
if 'Id' in merged_df.columns:
    merged_df[['Id', 'PredictedScore']].to_csv("notebook_score_predictions.csv", index=False)
    print("ğŸ“¤ Predictions saved to 'notebook_score_predictions.csv'")
else:
    print("â�Œ 'Id' column missing. Could not export predictions.")


