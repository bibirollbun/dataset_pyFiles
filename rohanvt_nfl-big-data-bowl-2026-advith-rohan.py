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





# FULL PIPELINE: train XGBoost on targeted receivers, predict for ALL offensive route-runners
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
import joblib
import math
import glob

# ---------------------------
# 1) File paths (all weeks)
# ---------------------------
tracking_files = glob.glob("/kaggle/input/nfl-big-data-bowl-2026-analytics/*/train/input_2023_w*.csv")
supp_fp = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv"

# ---------------------------
# 2) Load files
# ---------------------------
tracking_list = [pd.read_csv(f) for f in tracking_files]
tracking = pd.concat(tracking_list, ignore_index=True)
supp = pd.read_csv(supp_fp, low_memory=False)

# Merge play-level metadata into tracking (pass_length, pass_result, possession_team, etc.)
meta_cols = ['game_id','play_id','pass_length','pass_result','possession_team']
df = tracking.merge(supp[meta_cols].drop_duplicates(), on=['game_id','play_id'], how='left')

# ---------------------------
# 3) Keep only the final pre-throw snapshot (last input frame) for each play
# ---------------------------
last_frame_by_play = df.groupby(['game_id','play_id'])['frame_id'].transform('max')
df_last = df[df['frame_id'] == last_frame_by_play].copy()
df_last.reset_index(drop=True, inplace=True)

# ---------------------------
# 4) Identify offensive route-runners and targeted players
# ---------------------------
df_last['player_role'] = df_last['player_role'].astype(str)
df_last['player_side'] = df_last['player_side'].astype(str)
df_last['player_position'] = df_last['player_position'].astype(str)

route_roles = ['Targeted Receiver','Other Route Runner','Receiver']
df_last['is_off_route_runner'] = (
    (df_last['player_side'].str.lower() == 'offense') &
    (
        df_last['player_role'].isin(route_roles) |
        df_last['player_position'].isin(['WR','TE','RB'])
    )
)
df_last['is_targeted'] = df_last['player_role'] == 'Targeted Receiver'

# ---------------------------
# 5) Feature engineering per play
# ---------------------------
def compute_velocity_components(df_in):
    theta = np.deg2rad(df_in['dir'].fillna(0.0).astype(float))
    vx = df_in['s'].fillna(0.0).astype(float) * np.cos(theta)
    vy = df_in['s'].fillna(0.0).astype(float) * np.sin(theta)
    return vx, vy

df_last['vx'], df_last['vy'] = compute_velocity_components(df_last)

# QB position per play
qb_pos = df_last[df_last['player_role'] == 'Passer'][['game_id','play_id','nfl_id','x','y']].copy()
qb_pos.rename(columns={'nfl_id':'qb_nfl_id','x':'qb_x','y':'qb_y'}, inplace=True)
df_last = df_last.merge(qb_pos[['game_id','play_id','qb_x','qb_y']], on=['game_id','play_id'], how='left')

# Prepare features
feat_rows = []

for (g, p), play_df in df_last.groupby(['game_id','play_id']):
    defenders = play_df[play_df['player_side'].str.lower() == 'defense']
    def_x = defenders['x'].values
    def_y = defenders['y'].values
    qb_x = play_df['qb_x'].iloc[0] if 'qb_x' in play_df.columns else np.nan
    qb_y = play_df['qb_y'].iloc[0] if 'qb_y' in play_df.columns else np.nan

    for _, row in play_df.iterrows():
        if not row['is_off_route_runner']:
            continue

        wr_x, wr_y = row['x'], row['y']
        is_target = bool(row['is_targeted'])

        land_x = row.get('ball_land_x', wr_x) if is_target else wr_x
        land_y = row.get('ball_land_y', wr_y) if is_target else wr_y

        receiver_to_land = math.hypot(wr_x - land_x, wr_y - land_y) if pd.notna(land_x) and pd.notna(land_y) else np.nan
        nearest_def = float(np.min(np.hypot(def_x - wr_x, def_y - wr_y))) if len(def_x) > 0 else np.nan

        if (land_x == wr_x and land_y == wr_y) or pd.isna(land_x) or pd.isna(land_y):
            closing_speed = 0.0
        else:
            tx, ty = land_x - wr_x, land_y - wr_y
            norm = math.hypot(tx, ty)
            txn, tyn = tx/norm, ty/norm if norm != 0 else (0.0, 0.0)
            closing_speed = float(row.get('vx',0.0)*txn + row.get('vy',0.0)*tyn)

        if is_target and pd.notna(row.get('pass_length', np.nan)):
            pass_length_val = float(row['pass_length'])
        else:
            pass_length_val = math.hypot(qb_x - wr_x, qb_y - wr_y) if pd.notna(qb_x) and pd.notna(qb_y) else float(row.get('pass_length', np.nan))

        feat_rows.append({
            'game_id': g,
            'play_id': p,
            'nfl_id': row['nfl_id'],
            'player_name': row.get('player_name',''),
            'is_targeted': is_target,
            'receiver_to_land': receiver_to_land,
            'nearest_defender_dist': nearest_def,
            'closing_speed': closing_speed,
            'pass_length': pass_length_val,
            'pass_result': row.get('pass_result', np.nan)
        })

features_all = pd.DataFrame(feat_rows)

# ---------------------------
# 6) Training data (ONLY targeted)
# ---------------------------
targets = features_all[features_all['is_targeted']==True].copy()
targets['completed'] = targets['pass_result'].apply(lambda x: 1 if str(x).strip().upper()=='C' else 0)
core_feats = ['receiver_to_land','nearest_defender_dist','closing_speed','pass_length']
targets = targets.dropna(subset=core_feats).reset_index(drop=True)

X = targets[core_feats].values
y = targets['completed'].values

# ---------------------------
# 7) Train / test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# ---------------------------
# 8) Train XGBoost classifier
# ---------------------------
xgb = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb.fit(X_train, y_train)
joblib.dump(xgb, "xgb_catchprob_model.joblib")

# ---------------------------
# 9) Evaluate model
# ---------------------------
y_proba = xgb.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_proba)
brier = brier_score_loss(y_test, y_proba)
print(f"XGBoost evaluation on test set: AUC = {auc:.4f}  |  Brier score = {brier:.4f}")

# ---------------------------
# 10) Predict for ALL offensive route-runners
# ---------------------------
pred_df = features_all.copy()
for c in core_feats:
    if pred_df[c].isna().any():
        pred_df[c].fillna(pred_df[c].median(), inplace=True)

X_all = pred_df[core_feats].values
pred_df['catch_prob'] = xgb.predict_proba(X_all)[:,1]

catch_prob_table = pred_df[['game_id','play_id','nfl_id','player_name','is_targeted','catch_prob']].copy()

# ---------------------------
# 11) Save output CSV
# ---------------------------
out_fp = "/kaggle/working/catch_prob_for_all_receivers.csv"
catch_prob_table.to_csv(out_fp, index=False)
print("Saved:", out_fp)
print("\nSample rows:")
#print(catch_prob_table.head(20000).to_string(index=False))



import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from IPython.display import HTML

# ---------------------------
# Load play and catch_prob data
# ---------------------------
df = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w01.csv") 
catch_prob_table = pd.read_csv("/kaggle/working/catch_prob_for_all_receivers.csv")  

# ---------------------------
# Pick a single play
# ---------------------------
game_id = 2023090700
play_id = 194
play_df = df[(df['game_id'] == game_id) & (df['play_id'] == play_id)].copy()
play_df = play_df.merge(
    catch_prob_table[['game_id','play_id','nfl_id','catch_prob']],
    on=['game_id','play_id','nfl_id'],
    how='left'
)

# Identify targeted receiver
targeted = play_df[play_df['player_role'] == 'Targeted Receiver'].iloc[0]
target_id = targeted['nfl_id']

# ---------------------------
# Field setup
# ---------------------------
fig, ax = plt.subplots(figsize=(12,6))
ax.set_xlim(0, 120)
ax.set_ylim(0, 53.3)
ax.set_xlabel("Field Length (yds)")
ax.set_ylabel("Field Width (yds)")
ax.set_title(f"Game {game_id} Play {play_id}")

# Correct End zones
ax.add_patch(patches.Rectangle((0,0), 10, 53.3, color='green', alpha=0.2))    # left end zone
ax.add_patch(patches.Rectangle((110,0), 10, 53.3, color='green', alpha=0.2))  # right end zone

# Yard lines every 10 yards (field coordinates from 10 to 110)
for i in range(10, 110, 10):
    ax.vlines(i, 0, 53.3, color='gray', linewidth=0.5, alpha=0.5)
    
    # Calculate field label (distance from nearest goal line)
    if i <= 60:
        label = i - 10
    else:
        label = 120 - i - 10

    # Place numbers at top and bottom
    ax.text(i, 2, str(label), color='gray', fontsize=8, ha='center', alpha=0.7)
    ax.text(i, 53.3-2, str(label), color='gray', fontsize=8, ha='center', alpha=0.7)
# ---------------------------
# Player scatter objects
# ---------------------------
players = play_df['nfl_id'].unique()
scatters = {}
colors = {'Offense':'blue', 'Defense':'red'}
for pid in players:
    player = play_df[play_df['nfl_id'] == pid].iloc[0]
    if pid == target_id:
        scatters[pid] = ax.scatter(player['x'], player['y'], s=200, color='yellow', label=player['player_name'], edgecolors='black', linewidths=1.5)
    else:
        scatters[pid] = ax.scatter(player['x'], player['y'], s=100, color=colors.get(player['player_side'], 'gray'), label=player['player_name'])

# Legend
ax.legend(loc='upper right', fontsize=8, ncol=2)

# ---------------------------
# Animation function
# ---------------------------
frames = play_df['frame_id'].unique()
num_frames = len(frames)

def update(frame_idx):
    current_frame = frames[frame_idx]
    current = play_df[play_df['frame_id'] == current_frame]
    
    # Update player positions
    for pid in players:
        player_row = current[current['nfl_id'] == pid]
        if not player_row.empty:
            scatters[pid].set_offsets([player_row['x'].values[0], player_row['y'].values[0]])
    
    # At the last frame, show catch probabilities for offensive players
    if frame_idx == num_frames-1:
        offensive = current  # at final frame
        for _, row in offensive[offensive['player_side'] == 'Offense'].iterrows():
            prob = row.get('catch_prob', None)
            if prob is not None:
                ax.text(row['x']+0.5, row['y']+0.5, f"{prob:.2f}", color='black', fontsize=10, fontweight='bold')
    
    return list(scatters.values())

# ---------------------------
# Animate
# ---------------------------
ani = FuncAnimation(fig, update, frames=num_frames, interval=200, blit=False)
HTML(ani.to_jshtml())



import pandas as pd

# ---------------------------
# Load data
# ---------------------------
catch_prob = pd.read_csv("/kaggle/working/catch_prob_for_all_receivers.csv")
epa = pd.read_csv("/kaggle/input/epa-prediction/epa_predictions.csv")

# ---------------------------
# Merge on player + play
# ---------------------------
merged = catch_prob.merge(
    epa,
    on=["game_id", "play_id", "nfl_id", "player_name"],
    how="inner"
)

# ---------------------------
# Compute Reception Score
# ---------------------------
merged["Rec_score"] = merged["catch_prob"] * merged["epa_if_complete"]

# ---------------------------
# Compute max Rec_score per play
# ---------------------------
merged["max_Rec_score_play"] = merged.groupby(
    ["game_id", "play_id"]
)["Rec_score"].transform("max")

# ---------------------------
# Compute QBDE (ONLY for targeted receiver)
# ---------------------------
merged["QBDE"] = None

mask = (merged["is_targeted"] == True) & (merged["max_Rec_score_play"] > 0)
merged.loc[mask, "QBDE"] = (
    merged.loc[mask, "Rec_score"] /
    merged.loc[mask, "max_Rec_score_play"]
)

# ---------------------------
# Final table
# ---------------------------
qbde_table = merged[
    [
        "game_id",
        "play_id",
        "qb_name",
        "nfl_id",
        "player_name",
        "is_targeted",
        "catch_prob",
        "epa_if_complete",
        "Rec_score",
        "QBDE"
    ]
].copy()

# ---------------------------
# Save output
# ---------------------------
out_fp = "/kaggle/working/qbde_table.csv"
qbde_table.to_csv(out_fp, index=False)

print("Saved:", out_fp)
print(qbde_table.head(10).to_string(index=False))



import pandas as pd

# ---------------------------
# Load QBDE table
# ---------------------------
qbde = pd.read_csv("/kaggle/working/qbde_table.csv")

# ---------------------------
# Keep only targeted throws with valid QBDE
# ---------------------------
qb_throws = qbde[
    (qbde["is_targeted"] == True) &
    (qbde["QBDE"].notna())
].copy()

# ---------------------------
# Aggregate by QB
# ---------------------------
qb_summary = (
    qb_throws
    .groupby("qb_name")
    .agg(
        mean_QBDE=("QBDE", "mean"),
        total_throws=("QBDE", "count"),
        games_played=("game_id", "nunique")
    )
    .reset_index()
)

# ---------------------------
# Filter: QBs with 3+ games
# ---------------------------
qb_summary = qb_summary[qb_summary["games_played"] >= 10]

# ---------------------------
# Rank QBs
# ---------------------------
top_10 = qb_summary.sort_values("mean_QBDE", ascending=False).head(10)
bottom_10 = qb_summary.sort_values("mean_QBDE", ascending=True).head(10)

# ---------------------------
# Display results
# ---------------------------
print("\nTOP 10 QBs by QBDE (min 10 games)\n")
print(top_10.to_string(index=False, formatters={"mean_QBDE": "{:.3f}".format}))

print("\nBOTTOM 10 QBs by QBDE (min 10 games)\n")
print(bottom_10.to_string(index=False, formatters={"mean_QBDE": "{:.3f}".format}))



import pandas as pd

# ---------------------------
# SELECT QB (change this)
# ---------------------------
QB_NAME = "Patrick Mahomes"   # <-- change to any QB

# ---------------------------
# Load data
# ---------------------------
qbde = pd.read_csv("/kaggle/working/qbde_table.csv")

supp = pd.read_csv(
    "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv",
    low_memory=False
)

# ---------------------------
# Merge coverage
# ---------------------------
coverage_df = (
    supp[["game_id", "play_id", "team_coverage_type"]]
    .drop_duplicates()
)

qbde_cov = qbde.merge(
    coverage_df,
    on=["game_id", "play_id"],
    how="left"
)

# ---------------------------
# Filter to QB decision rows
# ---------------------------
qbde_qb = qbde_cov[
    (qbde_cov["qb_name"] == QB_NAME) &
    (qbde_cov["is_targeted"] == True) &
    (qbde_cov["QBDE"].notna()) &
    (qbde_cov["team_coverage_type"].notna())
].copy()

# ---------------------------
# Aggregate by coverage
# ---------------------------
qb_by_coverage = (
    qbde_qb
    .groupby("team_coverage_type")
    .agg(
        mean_QBDE=("QBDE", "mean"),
        std_QBDE=("QBDE", "std"),
        attempts=("QBDE", "count"),
        games_played=("game_id", "nunique")
    )
    .reset_index()
    .sort_values("mean_QBDE", ascending=False)
)

# ---------------------------
# Reliability filter
# ---------------------------
qb_by_coverage = qb_by_coverage[
    (qb_by_coverage["attempts"] >= 5) &
    (qb_by_coverage["games_played"] >= 2)
]

# ---------------------------
# Display
# ---------------------------
print(f"\nQBDE by Coverage for {QB_NAME}:\n")
print(qb_by_coverage.to_string(index=False, formatters={"mean_QBDE": "{:.3f}".format}))



import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ---------------------------
# USER INPUT: PICK QB
# ---------------------------
QB_NAME = "Patrick Mahomes"   # <-- change QB here

# ---------------------------
# FILE PATHS
# ---------------------------
QBDE_FP = "/kaggle/working/qbde_table.csv"
SUPP_FP = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv"

EPA_COL = "expected_points_added"

MIN_TARGETS_ELITE = 60
MIN_TARGETS_AVG = 20

# ---------------------------
# Load data
# ---------------------------
qbde = pd.read_csv(QBDE_FP)
supp = pd.read_csv(SUPP_FP)

supp = supp[["game_id", "play_id", EPA_COL]].drop_duplicates()

# ---------------------------
# Merge EPA
# ---------------------------
df = qbde.merge(
    supp,
    on=["game_id", "play_id"],
    how="left"
)

# Keep only targeted throws for selected QB
df = df[(df["qb_name"] == QB_NAME) & (df["is_targeted"] == True)].copy()

print(f"\nQB analyzed: {QB_NAME}")
print(f"Targeted throws: {len(df)}")

# ---------------------------
# (1) QBDE vs EPA
# ---------------------------
valid = df[["QBDE", EPA_COL]].dropna()

if len(valid) >= 3:
    pearson_r, pearson_p = pearsonr(valid["QBDE"], valid[EPA_COL])
    spearman_r, spearman_p = spearmanr(valid["QBDE"], valid[EPA_COL])

    print("\n==============================")
    print("QBDE vs Expected Points Added")
    print("==============================")
    print(f"Plays analyzed: {len(valid)}")
    print(f"Pearson r   : {pearson_r:.4f}  (p = {pearson_p:.2e})")
    print(f"Spearman ρ  : {spearman_r:.4f}  (p = {spearman_p:.2e})")
else:
    print("\nNot enough plays for QBDE vs EPA correlation")

# ---------------------------
# (7) QBDE vs Receiver Tier
# ---------------------------
receiver_counts = (
    df.groupby("player_name")
      .size()
      .rename("num_targets")
      .reset_index()
)

def assign_tier(n):
    if n >= MIN_TARGETS_ELITE:
        return "Elite"
    elif n >= MIN_TARGETS_AVG:
        return "Average"
    else:
        return "Low-Usage"

receiver_counts["receiver_tier"] = receiver_counts["num_targets"].apply(assign_tier)

df = df.merge(
    receiver_counts[["player_name", "receiver_tier"]],
    on="player_name",
    how="left"
)

tier_map = {"Low-Usage": 0, "Average": 1, "Elite": 2}
df["receiver_tier_num"] = df["receiver_tier"].map(tier_map)

tier_valid = df[["QBDE", "receiver_tier_num"]].dropna()

if len(tier_valid) >= 3:
    tier_p_r, tier_p_p = pearsonr(
        tier_valid["QBDE"], tier_valid["receiver_tier_num"]
    )
    tier_s_r, tier_s_p = spearmanr(
        tier_valid["QBDE"], tier_valid["receiver_tier_num"]
    )

    print("\n==============================")
    print("QBDE vs Receiver Tier")
    print("==============================")
    print(f"Plays analyzed: {len(tier_valid)}")
    print(f"Pearson r   : {tier_p_r:.4f}  (p = {tier_p_p:.2e})")
    print(f"Spearman ρ  : {tier_s_r:.4f}  (p = {tier_s_p:.2e})")
else:
    print("\nNot enough plays for QBDE vs Receiver Tier correlation")





