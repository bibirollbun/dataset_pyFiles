import pandas as pd
import gc
import os
import gc
import numpy as np
import pandas as pd
project_dir = '/kaggle/input/pre-processed-nfl2025'
weekly_files = [f"week_{w}_final.parquet" for w in range(1, 10)]
# Let's pairwise combine them:
temp_files = []
batch_size = 1  # read 3 files at a time

for i in range(0, len(weekly_files), batch_size):
    subset = weekly_files[i : i + batch_size]
    print(f"\nCombining subset: {subset}")
    
    dfs = []
    for f in subset:
        print(f"  Reading {f}")
        #df = pd.read_parquet(f)
        #os.path.join(project_dir, "plays.csv")
        df = pd.read_parquet(os.path.join(project_dir, f))
        dfs.append(df)
        del df
        gc.collect()
    
    merged = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    
    # Optionally sample if needed
    if len(merged) > 500_000:
        merged = merged.sample(n=100_000, random_state=42)
    
    out_name = f"merged_batch_{i//batch_size+1}.parquet"
    merged.to_parquet(out_name, index=False)
    temp_files.append(out_name)
    del merged
    gc.collect()

print("\nNow combine the temp_files if needed, in a second pass:")
final_dfs = []
for tf in temp_files:
    print(f"Reading {tf}")
    df2 = pd.read_parquet(tf)
    final_dfs.append(df2)
    del df2
    gc.collect()

df_all = pd.concat(final_dfs, ignore_index=True)
del final_dfs
gc.collect()

print("Final df_all shape:", df_all.shape)



#########################################################
# 1. Imports & Setup
#########################################################
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score,
    roc_curve  # <-- NEW: we need roc_curve for plotting
)
import xgboost as xgb

# For optional animation
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Optionally suppress warnings
warnings_filter = False
if warnings_filter:
    import warnings
    warnings.filterwarnings("ignore")

#########################################################
# 2. Main Script
#########################################################
def main():
    global team_analysis
    #########################################################
    # A) Load or assume df_all in memory
    #########################################################
    global df_all  # We assume df_all is loaded prior to this script
    # e.g. df_all = pd.read_parquet("my_data.parquet")
    #print("df_all shape:", df_all.shape)

    # Suppose df_all includes:
    #   [gameId, playId, nflId, teamAbbr, position, offenseFormation,
    #    play_label, plus feature columns, etc.]

    #########################################################
    # B) Train-Test Split by playId
    #########################################################
    unique_plays = df_all["playId"].unique()
    train_plays, test_plays = train_test_split(unique_plays, test_size=0.2, random_state=42)

    train_data = df_all[df_all["playId"].isin(train_plays)].copy().dropna()
    test_data  = df_all[df_all["playId"].isin(test_plays)].copy().dropna()

    print("Train data shape:", train_data.shape)
    print("Test data shape: ", test_data.shape)

    # Label-encode your classification label (e.g. "play_label")
    label_encoder = LabelEncoder()
    train_data["play_label"] = train_data["play_label"].astype(str)
    test_data["play_label"]  = test_data["play_label"].astype(str)

    train_data["play_label_encoded"] = label_encoder.fit_transform(train_data["play_label"])
    test_data["play_label_encoded"]  = label_encoder.transform(test_data["play_label"])

    # ID columns we want to keep for merging into predictions
    id_cols = ["gameId","playId","nflId","teamAbbr"]

    # We'll keep them in separate DataFrames so we can reattach after predicting
    train_ids = train_data[id_cols].reset_index(drop=True)
    test_ids  = test_data[id_cols].reset_index(drop=True)

    # Non-feature columns we do NOT want in X
    non_feature_cols = id_cols + ["frameId", "play_label", "play_label_encoded"]
    feature_cols = [c for c in train_data.columns if c not in non_feature_cols]

    # Build X, y
    X_train = train_data[feature_cols].reset_index(drop=True)
    y_train = train_data["play_label_encoded"].reset_index(drop=True)

    X_test  = test_data[feature_cols].reset_index(drop=True)
    y_test  = test_data["play_label_encoded"].reset_index(drop=True)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:",  X_test.shape)
    print("y_test shape:",  y_test.shape)

    # Convert columns to categorical for XGBoost if needed
    cat_cols = ["position","teamAbbr","offenseFormation"]
    for c in cat_cols:
        if c in X_train.columns:
            X_train[c] = X_train[c].astype("category")
        if c in X_test.columns:
            X_test[c] = X_test[c].astype("category")

    # XGBoost DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  enable_categorical=True)

    # Training params
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "seed": 42,
        "max_depth": 2
    }
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=200,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=10,
        verbose_eval=2
    )

    # Generate predictions (test set)
    y_pred_proba = model.predict(dtest)
    #print("Sample predictions:", y_pred_proba[:10])

    #########################################################
    # NEW: ROC Curve Plot for Train & Test
    #########################################################
    # We also get train predictions so we can plot Train ROC
    y_pred_proba_train = model.predict(dtrain)

    # Compute AUCs
    train_auc = roc_auc_score(y_train, y_pred_proba_train)
    test_auc  = roc_auc_score(y_test,  y_pred_proba)

    # Build ROC curves
    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
    fpr_test, tpr_test, _   = roc_curve(y_test,  y_pred_proba)

    plt.figure(figsize=(8,6))
    plt.plot(fpr_train, tpr_train, label=f"Train (AUC={train_auc:.3f})")
    plt.plot(fpr_test,  tpr_test,  label=f"Test (AUC={test_auc:.3f})")
    plt.plot([0,1],[0,1],'r--', label="Random Guess")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Train vs. Test)")
    plt.legend(loc="lower right")
    plt.show()

    #########################################################
    # C) Feature Importance
    #########################################################
    importance_dict = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        "Feature": list(importance_dict.keys()),
        "Importance": list(importance_dict.values())
    }).sort_values("Importance", ascending=False)

    #print("\nFeature Importances (top 10):")
    #print(importance_df.head(10))

    # Plot top 10
    plt.figure(figsize=(7,5))
    plt.barh(importance_df["Feature"][:20], importance_df["Importance"][:20])
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importances (XGBoost)")
    plt.xlabel("Importance (Gain)")
    plt.show()

    #########################################################
    # D) Build a Predictions DataFrame with IDs
    #########################################################
    predictions_df = pd.DataFrame({
        "y_pred_proba": y_pred_proba,
        "y_test":       y_test.reset_index(drop=True)
    })

    # Reattach the ID columns
    predictions_df = pd.concat([predictions_df, test_ids], axis=1)
    predictions_df.to_csv("predictions_df.csv", index=False)

    #print("\npredictions_df shape:", predictions_df.shape)
    #print(predictions_df.head(5))

    #########################################################
    # E) Merge with PFF Stats & Team-Level Analysis
    #########################################################
    # e.g., to get OFF,PASS,RECV,RUN,RBLK columns
    pff_stats_path = "/kaggle/input/nfl-bigdata-pffscores-2022/merged_player_offensive_pff_stats_corrected.csv"
    merged_pff_stats = pd.read_csv(pff_stats_path, usecols=[
        "gameId","playId","nflId","teamAbbr","OFF","PASS","RECV","RUN","RBLK"
    ])

    # Merge
    predictions_with_pff = predictions_df.merge(
        merged_pff_stats,
        how="left",
        on=["gameId","playId","nflId","teamAbbr"]
    )
   # print("\nAfter merging with PFF, shape:", predictions_with_pff.shape)

    # Filter out teams with <5 plays
    counts_by_team = (
        predictions_with_pff
        .groupby("teamAbbr")["playId"]
        .nunique()
        .reset_index(name="num_plays")
    )
    teams_with_5_plays = counts_by_team.loc[counts_by_team["num_plays"] >= 5, "teamAbbr"].tolist()

    predictions_with_pff = predictions_with_pff[
        predictions_with_pff["teamAbbr"].isin(teams_with_5_plays)
    ].copy()
  #  print(f"Kept {len(teams_with_5_plays)} teams with >=5 plays. Now shape:", predictions_with_pff.shape)

    # Compute team-level metrics
    def compute_team_metrics(team_df):
        y_true = team_df["y_test"]
        y_pred_proba = team_df["y_pred_proba"]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        try:
            team_auc = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            team_auc = float("nan")

        team_accuracy = accuracy_score(y_true, y_pred)
        team_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        team_precision = precision_score(y_true, y_pred, zero_division=0)
        team_recall = recall_score(y_true, y_pred, zero_division=0)
        team_f1 = f1_score(y_true, y_pred, zero_division=0)

        return pd.Series({
            "AUC": team_auc,
            "Accuracy": team_accuracy,
            "BalancedAcc": team_balanced_accuracy,
            "Precision": team_precision,
            "Recall": team_recall,
            "F1": team_f1
        })

    team_metrics = predictions_with_pff.groupby("teamAbbr").apply(compute_team_metrics).reset_index()
    #print("\nTeam-Level Metrics:")
    #print(team_metrics.head())

    # Aggregating PFF by team
    team_level_pff = (
        merged_pff_stats
        .groupby("teamAbbr")[["OFF","PASS","RECV","RUN","RBLK"]]
        .mean(numeric_only=True)
        .reset_index()
    )

    team_analysis = team_metrics.merge(team_level_pff, on="teamAbbr", how="left")
   # print("\nteam_analysis sample:")
   # print(team_analysis.head())

    # Correlation heatmap
    cols_for_corr = ["AUC","Accuracy","BalancedAcc","Precision","Recall","F1","OFF","PASS","RECV","RUN","RBLK"]
    corr_matrix = team_analysis[cols_for_corr].corr()

  #  print("\nCorrelation Matrix between Team-Level Metrics & PFF Stats:")
  #  print(corr_matrix)

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix: Team-Level Metrics vs. PFF Stats")
    plt.show()

    #########################################################
    # F) (Optional) Example Animation
    #########################################################
    # If you have a separate DataFrame (like `test_data_anim`) that stores
    # x,y positions for frames, you can animate them. We'll do a minimal example.

    # Suppose test_data_anim has columns: [frameId, gameId, nflId, playId, x, y, position]
    # We'll define a placeholder function to animate one play
    """
    Example:
    test_data_anim = pd.DataFrame(...) # must be loaded/created
    """

    def animate_play(example_gameId, example_playId):
        # Filter one play's frames from test_data_anim
        df_play = test_data_anim[
            (test_data_anim["gameId"] == example_gameId) &
            (test_data_anim["playId"] == example_playId)
        ].copy()
        df_play = df_play.sort_values("frameId")

        fig, ax = plt.subplots(figsize=(6,6))

        def update(frame_idx):
            ax.clear()
            frame_df = df_play[df_play["frameId"] == frame_idx]
            ax.scatter(frame_df["x"], frame_df["y"], c='blue')
            ax.set_xlim(0,120)
            ax.set_ylim(0,53.3)
            ax.set_title(f"gameId={example_gameId}, playId={example_playId}, frame={frame_idx}")

        frames = sorted(df_play["frameId"].unique())
        ani = FuncAnimation(fig, update, frames=frames, interval=300)
        return ani

    # Uncomment to run an example:
    """
    example_game = 2017090700
    example_play = 75
    animation_obj = animate_play(example_game, example_play)
    HTML(animation_obj.to_jshtml())
    """

 #   print("\nDone! End-to-end pipeline complete.")

if __name__ == "__main__":
    # Load df_all or define it globally, e.g.:
    # df_all = pd.read_parquet("my_final_data.parquet")
    # Optionally load test_data_anim if you want animations
    main()



#!/usr/bin/env python
# -*- coding: utf-8 -*-

#########################################################
# 1) Imports & Installs
#########################################################
!pip install kagglehub --quiet

import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np
from scipy.stats import pearsonr

#########################################################
# 2) Download the NFL Logos Dataset from Kaggle
#########################################################
path = kagglehub.dataset_download("anzhemeng/nfl-team-logos")
print("Logos dataset downloaded to:", path)
print("Directory listing:", os.listdir(path))

#########################################################
# 3) Map Team Abbreviations to PNG Filenames
#########################################################
TEAM_LOGOS = {
    "ATL": "ATL.png",
    "BUF": "BUF.png",
    "CAR": "CAR.png",
    "CHI": "CHI.png",
    "CIN": "CIN.png",
    "CLE": "CLE.png",
    "DAL": "DAL.png",
    "DEN": "DEN.png",
    "DET": "DET.png",
    "GB":  "GB.png",
    "HOU": "HOU.png",
    "IND": "IND.png",
    "JAX": "JAX.png",
    "KC":  "KC.png",
    "LAC": "LAC.png",
    "LAR": "LAR.png",
    "LV":  "LV.png",    # You may get a warning if LV.png is missing in the dataset
    "MIA": "MIA.png",
    "MIN": "MIN.png",
    "NE":  "NE.png",
    "NO":  "NO.png",
    "NYG": "NYG.png",
    "NYJ": "NYJ.png",
    "PHI": "PHI.png",
    "PIT": "PIT.png",
    "SEA": "SEA.png",
    "SF":  "SF.png",
    "TB":  "TB.png",
    "TEN": "TEN.png",
    "WAS": "WAS.png",
    # Add more if needed
}

#########################################################
# 4) Pre-Load Each Team Logo as an OffsetImage (Bigger Zoom)
#########################################################
def load_logo_image(path_to_folder, filename, zoom=0.25):
    """
    Given the folder path from kagglehub and a PNG filename,
    read the image with Pillow, convert to NumPy, and build OffsetImage.
    Increase 'zoom' to enlarge the logos.
    """
    full_path = os.path.join(path_to_folder, filename)
    im_pil = Image.open(full_path)
    im_np = np.array(im_pil)
    return OffsetImage(im_np, zoom=zoom)

team_logo_images = {}
for abbr, filename in TEAM_LOGOS.items():
    try:
        team_logo_images[abbr] = load_logo_image(path, filename, zoom=0.25)
    except Exception as e:
        print(f"Warning: could not load logo for {abbr}, file={filename}, error={e}")
        team_logo_images[abbr] = None

#########################################################
# 5) Example 'team_analysis' DataFrame (Replace or Load Yours)
#########################################################


#########################################################
# 6) Scatter Plot Function
#########################################################
def scatter_logos(ax, df, xcol, ycol, draw_corr=True):
    """
    For each row in df, place a team logo at (xcol, ycol).
    Optionally computes & displays Pearson correlation.
    """
    # Draw an invisible scatter to set axis limits
    ax.scatter(df[xcol], df[ycol], alpha=0)

    for _, row in df.iterrows():
        abbr = row["teamAbbr"]
        x_val = row[xcol]
        y_val = row[ycol]
        logo_image = team_logo_images.get(abbr, None)
        if logo_image is not None:
            ab = AnnotationBbox(logo_image, (x_val, y_val), frameon=False)
            ax.add_artist(ab)
        else:
            # fallback if no logo
            ax.plot(x_val, y_val, "ro")

    ax.set_xlabel("2022 Offense Predictability (accuracy)") #xcol
    ax.set_ylabel("2022 Team Run Blocking Grade") #ycol
    ax.set_title("Team Run Blocking Grade vs Offense Predictability (accuracy)" ) #f"{xcol} vs. {ycol}")

    if draw_corr:
        # Compute and display Pearson correlation
        subset = df.dropna(subset=[xcol, ycol])
        if len(subset) >= 2:
            r, p = pearsonr(subset[xcol], subset[ycol])
            ax.text(
                0.05, 0.90,
                f"r={r:.2f}, p={p:.2g}",
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7)
            )

#########################################################
# 7) Single Plot: Accuracy vs RBLK (with Larger Logos)
#########################################################
plot_df = team_analysis.dropna(subset=["Accuracy","RBLK"]).copy()
fig, ax = plt.subplots(figsize=(8, 6))
scatter_logos(ax, plot_df, "Accuracy", "RBLK")
plt.tight_layout()
plt.show()



#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
#########################################################
# 0. Imports
#########################################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score
)

#########################################################
# 1. Load predictions_df & Compute Per-Player Metrics
#########################################################
predictions_df = pd.read_csv("predictions_df.csv")

def compute_player_metrics(df):
    """
    Compute classification metrics for one player's subset of rows.
    """
    y_true = df["y_test"].values
    y_pred_proba = df["y_pred_proba"].values
    y_pred = (y_pred_proba >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = float("nan")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    return pd.Series({
        "AUC": auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "BalancedAcc": balanced_acc
    })

player_metrics = (
    predictions_df
    .groupby("nflId", as_index=False)
    .apply(compute_player_metrics)
)

#print("\nplayer_metrics (head):\n", player_metrics.head())

#########################################################
# 2. Load & Prepare Players
#########################################################
players_df = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/players.csv")
players_df["name_clean"] = players_df["displayName"].str.strip().str.lower()
players_df_unique = players_df.drop_duplicates(subset="name_clean")

def load_and_filter_position_data(
    csv_path, 
    name_col="player",
    position_filter=None, 
    min_games=10, 
    position_col="position"
):
    """
    Loads the given CSV, standardizes the name column, maps nflId,
    filters by position & min games if columns exist, 
    and returns the DataFrame.
    """
    df = pd.read_csv(csv_path)
    df["name_clean"] = df[name_col].str.strip().str.lower()

    # Map to nflId
    df["nflId"] = df["name_clean"].map(
        players_df_unique.set_index("name_clean")["nflId"]
    )
    df.dropna(subset=["nflId"], inplace=True)

    # Position filter
    if position_filter and position_col in df.columns:
        df = df[df[position_col].isin(position_filter)]

    # player_game_count filter
    if "player_game_count" in df.columns:
        df = df[df["player_game_count"] > min_games]

    return df

#########################################################
# 3. Load Each Position Group
#########################################################

# Offensive Line (C, G, T)
df_ol = load_and_filter_position_data(
    csv_path="/kaggle/input/pff-offenseblocking-2022/offense_blocking_2022.csv",
    name_col="player",
    position_filter=["C","G","T"],
    min_games=10
)
# Ensure 'grades_offense' is present. If not, rename or define it:
# Example: df_ol["grades_offense"] = df_ol["some_offense_col"]

# Rushing (RB, FB, HB)
df_rb = load_and_filter_position_data(
    csv_path="/kaggle/input/pff-data/rushing_summary_2022.csv",
    name_col="player",
    position_filter=["RB","FB","HB"],
    min_games=10
)
# E.g. df_rb might already have 'grades_offense' or define:
# df_rb["grades_offense"] = df_rb["whatever_col_you_use"]

# Passing (QB)
df_qb = load_and_filter_position_data(
    csv_path="/kaggle/input/pff-data/passing_summary_2022.csv",
    name_col="player",
    position_filter=["QB"],
    min_games=10
)
# Similarly, ensure there's a 'grades_offense' column

# Receiving (WR, TE)
df_wr = load_and_filter_position_data(
    csv_path="/kaggle/input/pff-data/receiving_summary_2022.csv",
    name_col="player",
    position_filter=["WR","TE"],
    min_games=10
)
# Ensure 'grades_offense' is present here as well.

#########################################################
# 4. Combine (Stack) All Position Groups
#########################################################
all_positions_df = pd.concat([df_ol, df_rb, df_qb, df_wr], ignore_index=True)
#print("\nall_positions_df shape:", all_positions_df.shape)
#print(all_positions_df.head(5))

#########################################################
# 5. Group by nflId to get a single grades_offense per player
#########################################################
# If each row is a partial game, let's average 'grades_offense' across them.
if "grades_offense" in all_positions_df.columns:
    pff_grouped = (
        all_positions_df
        .groupby(["nflId", 'position', 'player'], as_index=False)["grades_offense"]
        .mean(numeric_only=True)
    )
else:
    raise ValueError("No 'grades_offense' column found in your position CSVs!")



#########################################################
# 6. Merge player_metrics with pff_grouped (grades_offense)
#########################################################
merged_df = player_metrics.merge(
    pff_grouped, 
    on="nflId",
    how="left"
)

# Drop any rows missing the key columns
merged_df = merged_df.dropna(subset=[
    "AUC","Accuracy","Precision","Recall","F1","BalancedAcc","grades_offense"
])

#print("\nmerged_df shape:", merged_df.shape)
#print(merged_df.head())

#########################################################
# 7. Correlation: Predictive Metrics vs. grades_offense
#########################################################
cols_for_corr = ["AUC","Accuracy","BalancedAcc","Precision","Recall","F1","grades_offense"]
corr_df = merged_df[cols_for_corr]

corr_matrix = corr_df.corr()
#print("\nCorrelation Matrix (predictive metrics vs. grades_offense):")
#print(corr_matrix)

#########################################################
# 8. Optional: Heatmap
#########################################################
#plt.figure(figsize=(8,6))
#sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
#plt.title("Predictive Metrics vs. grades_offense Correlation")
#plt.show()



import pandas as pd

# Define the metrics we care about
metrics = ["AUC","Accuracy","Precision","Recall","F1","BalancedAcc"]

def compute_means_and_corrs(df_group):
    """
    Given one position group (DataFrame subset),
    compute the mean of each metric,
    plus the correlation of each metric with grades_offense.
    Returns a Series of these values.
    """
    # We'll store results in a dict
    result = {}

    # 1) Compute means for each metric
    for m in metrics:
        result[f"{m}_mean"] = df_group[m].mean()

    # 2) Compute correlation of each metric with grades_offense
    #    We'll handle small sample or all-NaN edge cases by returning None.
    for m in metrics:
        if df_group["grades_offense"].notna().sum() > 1:
            corr_val = df_group[[m,"grades_offense"]].corr().iloc[0,1]
        else:
            corr_val = None
        result[f"{m}_corr"] = corr_val

    return pd.Series(result)

# Apply groupby, dropping unneeded columns first
# (You can drop or keep whichever columns you want.)
grouped_results = (
    merged_df
    .drop(columns=["player","nflId"])  # or any columns you don’t need
    .groupby("position")
    .apply(compute_means_and_corrs)
    .reset_index()
)

# Sort the resulting DataFrame by AUC_mean in descending order
grouped_results.sort_values("AUC_mean", ascending=False, inplace=True)

# Reset index for a clean final look
grouped_results.reset_index(drop=True, inplace=True)

#print(grouped_results)
#grouped_results


import seaborn as sns
import matplotlib.pyplot as plt

# Suppose your final DataFrame is called grouped_results, 
# with columns like ["position", "AUC_mean", "AUC_corr", ..., "F1_corr"].

# 1) Identify the correlation columns
corr_cols = [c for c in grouped_results.columns if c.endswith("_corr")]

# 2) Create a subset DataFrame with just 'position' + corr columns,
#    setting 'position' as the index so we can plot a matrix of 
#    positions (rows) vs. metrics (columns).
corr_df = grouped_results[["position"] + corr_cols].set_index("position")

# 3) Plot a heatmap
plt.figure(figsize=(7,5))
sns.heatmap(
    corr_df, 
    annot=True,  # show the correlation values
    cmap="coolwarm", 
    center=0,    # 0 correlation is the midpoint color
    vmin=-1, vmax=1
)
plt.title("Correlation with grades_offense by Position")
plt.show()



# Filter for position 'G'
guards_df = merged_df[merged_df["position"] == "G"]

# Sort by AUC to find the least and most predictable players
least_predictable_guard = guards_df.loc[guards_df["AUC"].idxmin()]
most_predictable_guard = guards_df.loc[guards_df["AUC"].idxmax()]


# Optional: Create a summary DataFrame for better visualization
summary_df = pd.DataFrame({
    "Metric": ["Least Predictable", "Most Predictable"],
    "Player": [least_predictable_guard["player"], most_predictable_guard["player"]],
    "AUC": [least_predictable_guard["AUC"], most_predictable_guard["AUC"]]
})





# Filter for position 'G'
guards_df = merged_df[merged_df["position"] == "G"]

# Top 5 least predictable guards
least_predictable_guards = guards_df.nsmallest(5, "AUC").assign(Predictability="Least Predictable")

# Top 5 most predictable guards
most_predictable_guards = guards_df.nlargest(5, "AUC").assign(Predictability="Most Predictable")

# Combine into a single DataFrame
summary_df = pd.concat([least_predictable_guards, most_predictable_guards]).reset_index(drop=True)


summary_df


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os

#########################################################
# A) Helper to Load & Map 'nflId'
#########################################################
def load_and_map_nflId(csv_path, players_df_unique, position_filter=None, original_grade_col="grades_offense"):
    """
    1) Reads the CSV at `csv_path`.
    2) Creates 'name_clean' from 'player'.
    3) Maps to players_df_unique['nflId'] via name_clean.
    4) Optionally filters by position if position_filter is not None.
    5) Renames `original_grade_col` -> 'grades_offense'.
    6) Returns a DataFrame with columns [nflId, player, grades_offense].
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    df = pd.read_csv(csv_path)

    # 1) Ensure there's a 'player' column
    if "player" not in df.columns:
        raise ValueError(f"No 'player' column in {csv_path}. Adjust code?")

    # 2) Clean the name
    df["name_clean"] = df["player"].str.strip().str.lower()

    # 3) Map to nflId (we assume players_df_unique has unique name_clean -> nflId)
    df["nflId"] = df["name_clean"].map(
        players_df_unique.set_index("name_clean")["nflId"]
    )

    # Drop rows missing nflId
    df.dropna(subset=["nflId"], inplace=True)

    # 4) Position filter if requested
    if position_filter and "position" in df.columns:
        df = df[df["position"].isin(position_filter)].copy()

    # 5) Rename the user’s grade column -> 'grades_offense'
    if original_grade_col != "grades_offense":
        df.rename(columns={original_grade_col: "grades_offense"}, inplace=True)

    # 6) Keep columns [nflId, player, grades_offense]
    df = df[["nflId","player","grades_offense"]].dropna(subset=["nflId","grades_offense"])
    return df

#########################################################
# B) Build pff_players_2022.csv
#########################################################
def create_pff_players_2022():
    """
    Loads players.csv for name->nflId mapping, deduplicates by name_clean.
    Loads 2022 Rushing, Passing, Receiving CSVs, merges to get [nflId, player, grades_offense].
    Groups by nflId (averaging grades_offense, taking first player) and saves pff_players_2022.csv.
    """
    # 1) Load players.csv
    players_df = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/players.csv")
    players_df["name_clean"] = players_df["displayName"].str.strip().str.lower()
    players_df_unique = players_df.drop_duplicates(subset="name_clean")

    # 2) File paths (adjust if needed)
    rb_2022_path = "/kaggle/input/pff-data/rushing_summary_2022.csv"
    qb_2022_path = "/kaggle/input/pff-data/passing_summary_2022.csv"
    wr_2022_path = "/kaggle/input/pff-data/receiving_summary_2022.csv"

    # 3) Load each subset
    df_rb_2022 = load_and_map_nflId(
        csv_path=rb_2022_path,
        players_df_unique=players_df_unique,
        position_filter=["RB","FB","HB"],
        original_grade_col="grades_offense"
    )
    df_qb_2022 = load_and_map_nflId(
        csv_path=qb_2022_path,
        players_df_unique=players_df_unique,
        position_filter=["QB"],
        original_grade_col="grades_offense"
    )
    df_wr_2022 = load_and_map_nflId(
        csv_path=wr_2022_path,
        players_df_unique=players_df_unique,
        position_filter=["WR","TE"],
        original_grade_col="grades_offense"
    )

    # 4) Combine
    all_positions_2022 = pd.concat([df_rb_2022, df_qb_2022, df_wr_2022], ignore_index=True)

    # 5) Group by nflId -> average grades_offense, take first 'player'
    pff_players_2022 = (
        all_positions_2022
        .groupby("nflId", as_index=False)
        .agg({
            "grades_offense": "mean",
            "player": "first"
        })
    )

    # 6) Reorder columns if desired: [nflId, player, grades_offense]
    pff_players_2022 = pff_players_2022[["nflId","player","grades_offense"]]

    # 7) Save
    pff_players_2022.to_csv("pff_players_2022.csv", index=False)
    print(f"Saved pff_players_2022.csv with shape {pff_players_2022.shape}")

#########################################################
# C) Build pff_players_2023.csv
#########################################################
def create_pff_players_2023():
    """
    Same approach for 2023 data.
    """
    players_df = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2025/players.csv")
    players_df["name_clean"] = players_df["displayName"].str.strip().str.lower()
    players_df_unique = players_df.drop_duplicates(subset="name_clean")

    rb_2023_path = "/kaggle/input/pff-data/rushing_summary_2023.csv"
    qb_2023_path = "/kaggle/input/pff-data/passing_summary_2023.csv"
    wr_2023_path = "/kaggle/input/pff-data/receiving_summary_2023.csv"

    df_rb_2023 = load_and_map_nflId(
        csv_path=rb_2023_path,
        players_df_unique=players_df_unique,
        position_filter=["RB","FB","HB"],
        original_grade_col="grades_offense"
    )
    df_qb_2023 = load_and_map_nflId(
        csv_path=qb_2023_path,
        players_df_unique=players_df_unique,
        position_filter=["QB"],
        original_grade_col="grades_offense"
    )
    df_wr_2023 = load_and_map_nflId(
        csv_path=wr_2023_path,
        players_df_unique=players_df_unique,
        position_filter=["WR","TE"],
        original_grade_col="grades_offense"
    )

    all_positions_2023 = pd.concat([df_rb_2023, df_qb_2023, df_wr_2023], ignore_index=True)

    pff_players_2023 = (
        all_positions_2023
        .groupby("nflId", as_index=False)
        .agg({
            "grades_offense": "mean",
            "player": "first"
        })
    )
    pff_players_2023 = pff_players_2023[["nflId","player","grades_offense"]]

    pff_players_2023.to_csv("pff_players_2023.csv", index=False)
    print(f"Saved pff_players_2023.csv with shape {pff_players_2023.shape}")

#########################################################
# D) Main Execution
#########################################################
if __name__ == "__main__":
    create_pff_players_2022()
    create_pff_players_2023()
    print("\nDone creating pff_players_2022.csv and pff_players_2023.csv!")



#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################
# 0) Imports
###############################################################
import pandas as pd

###############################################################
# 1) Load Switchers: 2022 -> 2023
###############################################################
# This CSV has columns like:
# ["Player Name","playerId","2022 Team","2023 Team"]
switchers_df = pd.read_csv("/kaggle/input/pff-data/nfl_switchers_2022_2023.csv")

# Let's rename columns for clarity
switchers_df.rename(columns={
    "2022 Team": "teamAbbr_2022",
    "2023 Team": "teamAbbr_2023"
}, inplace=True)

# We'll keep only those who definitely switched from one team to another
switchers_df = switchers_df[
    switchers_df["teamAbbr_2022"] != switchers_df["teamAbbr_2023"]
].copy()

###############################################################
# 2) Load Team-Level Data (2022 & 2023)
###############################################################
# Suppose each CSV has columns: 
# teamAbbr, AUC, Accuracy, BalancedAcc, Precision, Recall, F1, OFF, PASS, RECV, RUN, RBLK
# (like your 'team_analysis' sample).

team_analysis_2022 = team_analysis.copy() #pd.read_csv("team_analysis_2022.csv")  # or loaded from memory
team_analysis_2023 = team_analysis.copy() #pd.read_csv("team_analysis_2023.csv")  # similarly

# We'll rename columns for clarity: AUC_2022, Accuracy_2022, etc.
rename_2022 = {col: f"{col}_2022" for col in ["AUC","Accuracy"]}
team_analysis_2022.rename(columns=rename_2022, inplace=True)

rename_2023 = {col: f"{col}_2023" for col in ["AUC","Accuracy"]}
team_analysis_2023.rename(columns=rename_2023, inplace=True)

# Merge them to switchers_df by matching teamAbbr
# For 2022:
switchers_df = switchers_df.merge(
    team_analysis_2022[["teamAbbr","AUC_2022","Accuracy_2022"]],
    how="left",
    left_on="teamAbbr_2022",
    right_on="teamAbbr"
).drop(columns=["teamAbbr"])

# For 2023:
switchers_df = switchers_df.merge(
    team_analysis_2023[["teamAbbr","AUC_2023","Accuracy_2023"]],
    how="left",
    left_on="teamAbbr_2023",
    right_on="teamAbbr"
).drop(columns=["teamAbbr"])







#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################
# 0) Imports
###############################################################
import pandas as pd

###############################################################
# 1) Load Switchers: 2022 -> 2023
###############################################################
switchers_df = pd.read_csv("/kaggle/input/pff-data/nfl_switchers_2022_2023.csv")

# Define a dictionary from full name -> short abbr (extend as needed)
TEAM_NAME_TO_ABBR = {
    "Atlanta Falcons": "ATL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",     # or LAR if you prefer
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    # Add or adjust if you have older names or missing entries
}

# Rename columns
switchers_df.rename(columns={
    "2022 Team": "teamAbbr_2022",
    "2023 Team": "teamAbbr_2023"
}, inplace=True)

# Filter out players who didn't actually switch teams
switchers_df = switchers_df[
    switchers_df["teamAbbr_2022"] != switchers_df["teamAbbr_2023"]
].copy()

# Map full names -> 3-letter codes
switchers_df["teamAbbr_2022"] = switchers_df["teamAbbr_2022"].map(TEAM_NAME_TO_ABBR)
switchers_df["teamAbbr_2023"] = switchers_df["teamAbbr_2023"].map(TEAM_NAME_TO_ABBR)

# If any rows didn't map, they become NaN. Optionally drop them:
switchers_df.dropna(subset=["teamAbbr_2022","teamAbbr_2023"], inplace=True)

###############################################################
# 2) Load Team-Level Data (2022 & 2023)
###############################################################
# For demonstration, we'll assume team_analysis_2022 and _2023
# each has columns like [teamAbbr, AUC, Accuracy, BalancedAcc, Precision, Recall, F1, ...]

team_analysis_2022 = team_analysis.copy()
team_analysis_2023 = team_analysis.copy()

# Rename columns to differentiate 2022 vs 2023
rename_2022 = {col: f"{col}_2022" for col in ["AUC","Accuracy"]}
team_analysis_2022.rename(columns=rename_2022, inplace=True)

rename_2023 = {col: f"{col}_2023" for col in ["AUC","Accuracy"]}
team_analysis_2023.rename(columns=rename_2023, inplace=True)

###############################################################
# 3) Merge 2022 metrics
###############################################################
switchers_df = switchers_df.merge(
    team_analysis_2022[["teamAbbr","AUC_2022","Accuracy_2022"]],
    how="left",
    left_on="teamAbbr_2022",
    right_on="teamAbbr"
).drop(columns=["teamAbbr"])

###############################################################
# 4) Merge 2023 metrics
###############################################################
switchers_df = switchers_df.merge(
    team_analysis_2023[["teamAbbr","AUC_2023","Accuracy_2023"]],
    how="left",
    left_on="teamAbbr_2023",
    right_on="teamAbbr"
).drop(columns=["teamAbbr"])

###############################################################
# 5) Inspect the result
###############################################################
#print(switchers_df.head(20))



###############################################################
# 3) Load Player-Level PFF Data (2022 & 2023) with grades_offense
###############################################################
# Example: you might unify your rushing/passing/receiving data into one DataFrame
# for each season. We assume columns: ["playerId","grades_offense"] plus any needed filters.

pff_players_2022 = pd.read_csv("/kaggle/working/pff_players_2022.csv")  # Must have [playerId, grades_offense_2022]
pff_players_2023 = pd.read_csv("/kaggle/working/pff_players_2023.csv")  # Must have [playerId, grades_offense_2023]

# Make sure columns align:
pff_players_2022.rename(columns={"grades_offense": "grades_offense_2022"}, inplace=True)
pff_players_2023.rename(columns={"grades_offense": "grades_offense_2023"}, inplace=True)
pff_players_2023['nflId'] = pff_players_2023['nflId'].astype(int)
pff_players_2022['nflId'] = pff_players_2022['nflId'].astype(int)

switchers_df['playerId'] = switchers_df['playerId'].astype(int)
###############################################################
# 4) Merge Player PFF data into Switchers
###############################################################
# Join on playerId to get each player's 2022 & 2023 PFF grade

switchers_df = switchers_df.merge(
    pff_players_2022[["nflId","grades_offense_2022", 'player']],
    how="left",
    right_on = 'player',
    left_on="Player Name"
)
switchers_df = switchers_df.merge(
    pff_players_2023[["nflId","grades_offense_2023", 'player']],
    how="left",
    
    right_on = 'player',
    left_on="Player Name"
   # on="nflId"
)
# Only keep rows where we found PFF data for both years
switchers_df.dropna(subset=["grades_offense_2022","grades_offense_2023"], inplace=True)

switchers_df = switchers_df.dropna()



###############################################################
# 5) Compute the Delta for PFF Grades & Team Predictability
###############################################################
switchers_df["delta_grades_offense"] = (
    switchers_df["grades_offense_2023"] - switchers_df["grades_offense_2022"]
)

# For AUC
switchers_df["delta_AUC"] = switchers_df["AUC_2023"] - switchers_df["AUC_2022"]

# For Accuracy
switchers_df["delta_Accuracy"] = (
    switchers_df["Accuracy_2023"] - switchers_df["Accuracy_2022"]
)

switchers_df = switchers_df[(switchers_df['delta_grades_offense'] < 10) & (switchers_df['delta_grades_offense'] > -10) ]
switchers_df

###############################################################
# 6) (Optional) Explore correlation or regression
###############################################################
# Example: correlation of change in PFF grade vs. change in AUC
corr_auc = switchers_df[["delta_grades_offense","delta_AUC"]].corr().iloc[0,1]
print(f"Correlation between Δgrades_offense & ΔAUC = {corr_auc:.3f}")

# Or do the same for Accuracy
#corr_acc = switchers_df[["delta_grades_offense","delta_Accuracy"]].corr().iloc[0,1]
#print(f"Correlation between Δgrades_offense & ΔAccuracy = {corr_acc:.3f}")

# You could also do a quick scatter:
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.scatter(
    switchers_df["delta_AUC"], 
    switchers_df["delta_grades_offense"],
    alpha=0.7
)
plt.axhline(y=0, color="gray", linestyle="--")
plt.axvline(x=0, color="gray", linestyle="--")
plt.xlabel("ΔAUC (2023 - 2022)")
plt.ylabel("Δgrades_offense (2023 - 2022)")
plt.title("Delta Offense Grade vs. Delta Team AUC for Switchers")
plt.show()


###############################################################
# Done! You now have a DataFrame that can be used 
# to investigate difference-in-differences for switchers.
###############################################################





path = "/kaggle/input/nfl-team-logos"
switchers_df


TEAM_LOGOS = {
    "TB":  "TB.png",
    "WAS": "WAS.png",
    "NYJ": "NYJ.png",
    "CIN": "CIN.png",
    "ARI": "ARI.png",
    "LAC": "LAC.png",
    "ATL": "ATL.png",
    "MIN": "MIN.png",
    "CHI": "CHI.png",
    "LV":  "OAK.png",   # If your dataset has 'OAK.png' for the Raiders
    "DAL": "DAL.png",
    "SEA": "SEA.png",
    "LAR": "LA.png",    # If your dataset calls Rams "LA.png"
    "NYG": "NYG.png",
    "CLE": "CLE.png",
    "DET": "DET.png",
    "DEN": "DEN.png",
    "PHI": "PHI.png",
    "MIA": "MIA.png",
    "CAR": "CAR.png",
    "BUF": "BUF.png",
    "IND": "IND.png",
    "KC":  "KC.png",
    "SF":  "SF.png",
    "GB":  "GB.png",
    "NE":  "NE.png",
    "PIT": "PIT.png",
    "JAX": "JAX.png",
    "NO":  "NO.png",
    "TEN": "TEN.png",
    "BAL": "BAL.png",
    "HOU": "HOU.png",
    # ... add more if needed ...
}


path_to_folder = "/kaggle/input/nfl-team-logos"


#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################
# 1) Imports
###############################################################
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from adjustText import adjust_text
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.stats import pearsonr

###############################################################
# 2) Resampling Fallback for Old Pillow Versions
###############################################################
try:
    LANCZOS = Image.Resampling.LANCZOS
except (AttributeError, ImportError):
    try:
        LANCZOS = Image.LANCZOS
    except AttributeError:
        LANCZOS = Image.BICUBIC

###############################################################
# 3) Dual Logo Creator (Slightly Larger Zoom, but No Arrow)
###############################################################
def create_dual_logo_image(abbr_from, abbr_to, path_to_folder, zoom=0.17, TEAM_LOGOS=None):
    """
    Side-by-side logos with no arrow in between, to keep it simpler.
    """
    # Make sure TEAM_LOGOS is passed
    if TEAM_LOGOS is None:
        return None

    from_filename = TEAM_LOGOS.get(abbr_from, None)
    to_filename   = TEAM_LOGOS.get(abbr_to, None)
    if not from_filename or not to_filename:
        return None

    try:
        from_img = Image.open(os.path.join(path_to_folder, from_filename)).convert("RGBA")
        to_img   = Image.open(os.path.join(path_to_folder, to_filename)).convert("RGBA")
    except:
        return None

    def pil_rescale(img, scale=zoom):
        w, h = img.size
        return img.resize((int(w*scale), int(h*scale)), resample=LANCZOS)

    from_img = pil_rescale(from_img, zoom)
    to_img   = pil_rescale(to_img,   zoom)

    # Combine side-by-side (no arrow)
    total_w = from_img.size[0] + to_img.size[0]
    max_h   = max(from_img.size[1], to_img.size[1])
    combined = Image.new("RGBA", (total_w, max_h), (0,0,0,0))

    x_offset = 0
    combined.paste(from_img, (x_offset, 0), from_img)
    x_offset += from_img.size[0]
    combined.paste(to_img, (x_offset, 0), to_img)

    combined_np = np.array(combined)
    return OffsetImage(combined_np, zoom=1.0)

###############################################################
# 4) Final Plot Function
###############################################################
def plot_switchers_for_linkedin(
    switchers_df,
    path_to_folder,
    TEAM_LOGOS,
    title="Delta Offense Grade vs. Delta Team Offense Predictability AUC"
):
    """
    Creates a visually pleasing scatter plot with larger fonts,
    bigger figure size, and minimal overlap, suitable for a
    LinkedIn post or similar social media share.
    """
    # 1) Figure & Basic Scatter
    fig, ax = plt.subplots(figsize=(12, 8))  # Larger figure
    ax.scatter(
        switchers_df["delta_AUC"], 
        switchers_df["delta_grades_offense"], 
        alpha=0
    )

    # 2) Place Logos & Collect Label Texts
    texts = []
    for idx, row in switchers_df.iterrows():
        x_val = row["delta_AUC"]
        y_val = row["delta_grades_offense"]

        abbr_from   = row["teamAbbr_2022"]
        abbr_to     = row["teamAbbr_2023"]
        player_name = row["Player Name"]

        # Slightly bigger zoom for the logos
        dual_logo = create_dual_logo_image(abbr_from, abbr_to, path_to_folder, zoom=0.17, TEAM_LOGOS=TEAM_LOGOS)
        if dual_logo is not None:
            # Slight alpha to help text stand out behind/around logos
            ab = AnnotationBbox(dual_logo, (x_val, y_val), frameon=False, alpha=0.95)
            ax.add_artist(ab)
        else:
            ax.plot(x_val, y_val, "ro")

        # 3) Larger text for LinkedIn
        label_text = f"{player_name} ({abbr_from}->{abbr_to})"
        t = ax.text(
            x_val, 
            y_val, 
            label_text,
            fontsize=12,        # Larger font
            fontweight="regular",
            ha="left",
            va="bottom"
        )
        texts.append(t)

    # 4) Lines, Axes, Title
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("ΔTeam Offense Predictability (AUC)", fontsize=14, fontweight="bold")
    ax.set_ylabel("ΔPFF Offense Grade", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")

    # 5) Adjust Overlaps & Layout
    adjust_text(
        texts, ax=ax,
        only_move={'text': 'xy', 'objects': 'xy'},
        arrowprops=None,  # no lines from text to point for a cleaner look
        force_text=(0.5, 0.5),
        expand_text=(1.2, 1.2),
        expand_points=(1.2, 1.2),
    )

    plt.tight_layout()
    plt.show()

###############################################################
# Usage Example
###############################################################
# Suppose you have:
# TEAM_LOGOS = { "SF":"SF.png", "LV":"OAK.png", ... }  # etc.
# switchers_df columns = ["Player Name","teamAbbr_2022","teamAbbr_2023",
#                         "delta_AUC","delta_grades_offense"]

# Then run:
plot_switchers_for_linkedin(switchers_df, path_to_folder, TEAM_LOGOS)


