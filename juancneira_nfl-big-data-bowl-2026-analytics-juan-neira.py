import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


base_path = "/kaggle/input/nfl-big-data-bowl-2026-analytics"
train_path = os.path.join(base_path, os.listdir(base_path)[0], "train")  # pick the main folder, then train

# Example: load one week of tracking input and output + plays metadata
input_files = sorted(glob.glob(os.path.join(train_path, "input_2023_w*.csv")))
output_files = sorted(glob.glob(os.path.join(train_path, "output_2023_w*.csv")))
plays_file = os.path.join(train_path, "plays.csv")
games_file = os.path.join(train_path, "games.csv")        

print("Input files:", input_files[:3])
print("Output files:", output_files[:3])

df_input = pd.read_csv(input_files[0])
df_output = pd.read_csv(output_files[0])

print(df_input.columns)
print(df_output.columns)




targets = (
    df_input[df_input["player_to_predict"] == 1]
    [["game_id","play_id","nfl_id","player_side","player_position","player_role"]]
    .drop_duplicates(subset=["game_id","play_id","nfl_id"])
    .rename(columns={
        "nfl_id":"target_nfl_id",
        "player_side":"target_side",
        "player_position":"target_position",
        "player_role":"target_role"
    })
)

print("Number of (play, target) pairs:", len(targets))
print("Targets per play (distribution):")
print(targets.groupby(["game_id","play_id"]).size().value_counts().sort_index())
targets.head()


# Target trajectories from output
traj = (
    df_output
    .rename(columns={
        "nfl_id": "target_nfl_id",
        "x": "x_t",
        "y": "y_t"
    })
    .merge(
        targets[["game_id","play_id","target_nfl_id","target_side"]],
        on=["game_id","play_id","target_nfl_id"],
        how="inner"
    )
)

print("Target trajectory rows:", len(traj))
print("Unique (play, target) pairs in traj:",
      traj[["game_id","play_id","target_nfl_id"]].drop_duplicates().shape[0])

traj.head()


# All players from input (potential defenders)
others = df_input[[
    "game_id","play_id","frame_id",
    "nfl_id","player_side","x","y"
]]

# Join target trajectories with all players at the same frame
pairs = traj.merge(
    others,
    on=["game_id","play_id","frame_id"],
    how="inner"
)

# Keep only defenders and exclude the target itself
pairs = pairs[
    (pairs["player_side"] != pairs["target_side"]) &
    (pairs["nfl_id"] != pairs["target_nfl_id"])
].copy()

# Compute distance
pairs["dist"] = (
    (pairs["x"] - pairs["x_t"])**2 +
    (pairs["y"] - pairs["y_t"])**2
) ** 0.5

# Nearest defender per target per frame
nearest = (
    pairs
    .groupby(["game_id","play_id","target_nfl_id","frame_id"])["dist"]
    .min()
    .reset_index(name="nearest_defender_dist")
)

print("Nearest rows:", len(nearest))
print("Unique (play, target) pairs:",
      nearest[["game_id","play_id","target_nfl_id"]].drop_duplicates().shape[0])
nearest.head()


play_target_metrics = (
    nearest
    .groupby(["game_id","play_id","target_nfl_id"])["nearest_defender_dist"]
    .agg(
        sep_min="min",
        sep_mean="mean",
        sep_max="max",
        sep_p10=lambda x: x.quantile(0.10),
        sep_p90=lambda x: x.quantile(0.90),
        frac_sep_gt_3=lambda x: (x > 3.0).mean(),
        frac_sep_gt_5=lambda x: (x > 5.0).mean(),
        n_frames="count"
    )
    .reset_index()
)

print("Rows in play_target_metrics:", len(play_target_metrics))
print("Unique (play, target) pairs:",
      play_target_metrics[["game_id","play_id","target_nfl_id"]].drop_duplicates().shape[0])

play_target_metrics.head()


target_meta = targets[[
    "game_id","play_id","target_nfl_id",
    "target_side","target_position","target_role"
]]

# Ball landing + play context (one row per play)
ball_land = (
    df_input[[
        "game_id","play_id",
        "ball_land_x","ball_land_y",
        "absolute_yardline_number","play_direction"
    ]]
    .drop_duplicates(subset=["game_id","play_id"])
)

# Merge everything
final_df = (
    play_target_metrics
    .merge(target_meta, on=["game_id","play_id","target_nfl_id"], how="left")
    .merge(ball_land, on=["game_id","play_id"], how="left")
)

print("Rows in final_df:", len(final_df))
print("Missing target_side:",
      final_df["target_side"].isna().mean())
print("Missing ball_land_x:",
      final_df["ball_land_x"].isna().mean())

final_df.head()


# Restrict to offensive targets
off_df = final_df[final_df["target_side"] == "Offense"].copy()

# --- Get final frame position for each (play, target) ---
# We use the last available frame in traj for that target
final_pos = (
    traj
    .sort_values("frame_id")
    .groupby(["game_id","play_id","target_nfl_id"], as_index=False)
    .last()[["game_id","play_id","target_nfl_id","x_t","y_t"]]
    .rename(columns={"x_t":"x_final", "y_t":"y_final"})
)

# Merge final positions into offensive dataframe
off_df = off_df.merge(
    final_pos,
    on=["game_id","play_id","target_nfl_id"],
    how="left"
)

# --- Compute final distance to ball landing point ---
off_df["final_ball_dist"] = (
    (off_df["x_final"] - off_df["ball_land_x"])**2 +
    (off_df["y_final"] - off_df["ball_land_y"])**2
) ** 0.5

# --- Define pass success (1.0-yard threshold) ---
off_df["pass_success"] = (off_df["final_ball_dist"] <= 1.0).astype(int)

# --- Sanity checks ---
print("Offensive targets:", len(off_df))
print("Pass success rate:", off_df["pass_success"].mean())
print("Final distance (yards) summary:")
print(off_df["final_ball_dist"].describe())

off_df.head()


vars_to_test = [
    "sep_min",
    "sep_mean",
    "sep_max",
    "frac_sep_gt_3"
]

results = []

for v in vars_to_test:
    r, p = pointbiserialr(off_df["pass_success"], off_df[v])
    results.append({
        "variable": v,
        "correlation_r": r,
        "p_value": p
    })

corr_df = pd.DataFrame(results).sort_values("correlation_r", ascending=False)
corr_df


KEYS = ["game_id", "play_id", "frame_id"]
KEYS_PLAY_TARGET = ["game_id", "play_id", "target_nfl_id"]

# -----------------------------
# 0) Ensure clean state (prevents dist_to_qb_x / dist_to_qb_y confusion)
# -----------------------------
if "dist_to_qb" in off_df.columns:
    off_df = off_df.drop(columns=["dist_to_qb"])
if "dist_to_qb_x" in off_df.columns or "dist_to_qb_y" in off_df.columns:
    off_df = off_df.drop(columns=[c for c in ["dist_to_qb_x", "dist_to_qb_y"] if c in off_df.columns])

# -----------------------------
# 1) QB positions per play-frame
# -----------------------------
qb_pos = (
    df_input.loc[df_input["player_position"].eq("QB"), ["game_id","play_id","frame_id","x","y"]]
    .rename(columns={"x": "x_qb", "y": "y_qb"})
)

print("QB rows in df_input:", len(qb_pos))
print("Plays with QB present:", qb_pos[["game_id","play_id"]].drop_duplicates().shape[0])

# -----------------------------
# 2) Target positions per play-target-frame
# -----------------------------
target_pos = traj.loc[:, ["game_id","play_id","target_nfl_id","frame_id","x_t","y_t"]]

# -----------------------------
# 3) Merge target and QB by play-frame
# -----------------------------
target_qb = target_pos.merge(qb_pos, on=KEYS, how="inner")

# -----------------------------
# 4) First available frame per (play, target)
# -----------------------------
first_frame = (
    target_qb.groupby(KEYS_PLAY_TARGET, as_index=False)["frame_id"]
    .min()
    .rename(columns={"frame_id": "first_frame_id"})
)

target_qb_first = target_qb.merge(first_frame, on=KEYS_PLAY_TARGET, how="inner")
target_qb_first = target_qb_first.loc[
    target_qb_first["frame_id"].eq(target_qb_first["first_frame_id"]),
    KEYS_PLAY_TARGET + ["x_t", "y_t", "x_qb", "y_qb"]
].drop_duplicates(subset=KEYS_PLAY_TARGET)

# -----------------------------
# 5) Compute distance to QB
# -----------------------------
dx = target_qb_first["x_t"] - target_qb_first["x_qb"]
dy = target_qb_first["y_t"] - target_qb_first["y_qb"]
target_qb_first["dist_to_qb"] = np.sqrt(dx*dx + dy*dy)

# Sanity: ensure the column exists before merging
assert "dist_to_qb" in target_qb_first.columns, "dist_to_qb was not created."

# -----------------------------
# 6) Merge into off_df (overwrite-safe)
# -----------------------------
off_df = off_df.merge(
    target_qb_first[KEYS_PLAY_TARGET + ["dist_to_qb"]],
    on=KEYS_PLAY_TARGET,
    how="left",
    validate="one_to_one"
)

# -----------------------------
# 7) Diagnostics + correlation on valid rows only
# -----------------------------
print("Columns containing 'dist_to_qb':", [c for c in off_df.columns if "dist_to_qb" in c])

missing = off_df["dist_to_qb"].isna()
print("Missing dist_to_qb rate:", float(missing.mean()))
print("Missing dist_to_qb count:", int(missing.sum()))

valid = off_df.dropna(subset=["dist_to_qb", "pass_success"])
print("Rows used for correlation:", len(valid))

r, p = pointbiserialr(valid["pass_success"], valid["dist_to_qb"])
print("Correlation r:", r)
print("p-value:", p)

print("\nDistance to QB summary (valid rows):")
print(valid["dist_to_qb"].describe())


# --- Restrict target trajectories to offensive targets only ---
traj_off = traj.merge(
    off_df[["game_id","play_id","target_nfl_id"]],
    on=["game_id","play_id","target_nfl_id"],
    how="inner"
)

# --- Attach ball landing point to each frame ---
traj_off = traj_off.merge(
    off_df[[
        "game_id","play_id","target_nfl_id",
        "ball_land_x","ball_land_y","pass_success"
    ]],
    on=["game_id","play_id","target_nfl_id"],
    how="left"
)

# --- Distance to ball landing point at each frame ---
traj_off["ball_dist"] = (
    (traj_off["x_t"] - traj_off["ball_land_x"])**2 +
    (traj_off["y_t"] - traj_off["ball_land_y"])**2
) ** 0.5

# --- Minimum distance during ball flight ---
ball_min = (
    traj_off
    .groupby(["game_id","play_id","target_nfl_id"], as_index=False)
    .agg(
        ball_dist_min=("ball_dist", "min"),
        pass_success=("pass_success", "first")
    )
)

print("Rows in ball_min:", len(ball_min))

# --- Correlation with pass success ---
r, p = pointbiserialr(ball_min["pass_success"], ball_min["ball_dist_min"])

print("Ball–receiver min distance:")
print("Correlation r:", r)
print("p-value:", p)
print()
print("ball_dist_min summary:")
print(ball_min["ball_dist_min"].describe())

ball_min.head()


KEYS_PLAY_TARGET = ["game_id", "play_id", "target_nfl_id"]

# 0) Clean any prior sep_pre_min columns (prevents _x/_y suffix issues)
for c in [col for col in off_df.columns if col.startswith("sep_pre_min")]:
    off_df = off_df.drop(columns=[c])

# 1) Frame count per (play, target)
frame_counts = (
    nearest.groupby(KEYS_PLAY_TARGET)["frame_id"]
    .nunique()
    .rename("n_frames_total")
    .reset_index()
)

nearest_ext = nearest.merge(frame_counts, on=KEYS_PLAY_TARGET, how="left")

# 2) Early window = first ceil(25%) frames, at least 1 frame
nearest_ext["n_early"] = np.maximum(1, np.ceil(0.25 * nearest_ext["n_frames_total"]).astype(int))
nearest_ext["early_frame"] = nearest_ext["frame_id"] <= nearest_ext["n_early"]

# 3) Pre-throw minimum separation over early frames
sep_pre = (
    nearest_ext.loc[nearest_ext["early_frame"]]
    .groupby(KEYS_PLAY_TARGET, as_index=False)
    .agg(sep_pre_min=("nearest_defender_dist", "min"))
)

# 4) Merge into off_df
off_df = off_df.merge(sep_pre, on=KEYS_PLAY_TARGET, how="left", validate="one_to_one")

# 5) Verify column exists (prevents silent failure)
print("Columns containing sep_pre_min:", [c for c in off_df.columns if "sep_pre_min" in c])

# 6) Correlation (valid rows only)
valid = off_df.dropna(subset=["sep_pre_min", "pass_success"])
print("Rows used:", len(valid), "out of", len(off_df))

r, p = pointbiserialr(valid["pass_success"], valid["sep_pre_min"])
print("Correlation r:", r)
print("p-value:", p)

print("\nsep_pre_min summary:")
print(valid["sep_pre_min"].describe())


# Correlation values that were computed
corr_data = pd.DataFrame({
    "Metric": [
        "Pre-throw openness (sep_pre_min)",
        "Openness during flight (sep_min)",
        "Distance to QB (dist_to_qb)",
        "Ball–receiver alignment (ball_dist_min)"
    ],
    "Correlation": [
        -0.06036,
        -0.06583,
        -0.10855,
        -0.45399
    ]
})

# Sort by absolute correlation strength
corr_data["abs_corr"] = corr_data["Correlation"].abs()
corr_data = corr_data.sort_values("abs_corr")

plt.figure(figsize=(7,4))
plt.barh(corr_data["Metric"], corr_data["Correlation"])
plt.axvline(0, linestyle="--", linewidth=1)
plt.xlabel("Correlation with Pass Success")
plt.title("Relative Importance of Spatial Factors for Pass Success")
plt.tight_layout()
plt.show()


# Remove any previous versions (prevents _x/_y suffix issues on reruns)
for c in [col for col in off_df.columns if col.startswith("ball_dist_min")]:
    off_df = off_df.drop(columns=[c])

# Merge ball_dist_min from ball_min into off_df
off_df = off_df.merge(
    ball_min[["game_id","play_id","target_nfl_id","ball_dist_min"]],
    on=["game_id","play_id","target_nfl_id"],
    how="left",
    validate="one_to_one"
)

print("Missing ball_dist_min rate:", off_df["ball_dist_min"].isna().mean())
print("Columns check:", [c for c in off_df.columns if "ball_dist_min" in c])


traj_off = traj.merge(
    off_df[["game_id","play_id","target_nfl_id"]],
    on=["game_id","play_id","target_nfl_id"],
    how="inner"
).merge(
    off_df[["game_id","play_id","target_nfl_id","ball_land_x","ball_land_y","pass_success"]],
    on=["game_id","play_id","target_nfl_id"],
    how="left"
)

traj_off["ball_dist"] = ((traj_off["x_t"] - traj_off["ball_land_x"])**2 +
                         (traj_off["y_t"] - traj_off["ball_land_y"])**2) ** 0.5

ball_min = (traj_off.groupby(["game_id","play_id","target_nfl_id"], as_index=False)
            .agg(ball_dist_min=("ball_dist","min"),
                 pass_success=("pass_success","first")))


features = ["ball_dist_min", "dist_to_qb", "sep_pre_min", "sep_min"]

# Confirm columns exist before dropping NaNs
missing_cols = [c for c in features + ["pass_success"] if c not in off_df.columns]
print("Missing columns:", missing_cols)

model_df = off_df.dropna(subset=features + ["pass_success"]).copy()

X = model_df[features].values
y = model_df["pass_success"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_scaled, y)

coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": clf.coef_[0]
}).sort_values("Coefficient")

print("Rows used:", len(model_df))
coef_df


plt.figure(figsize=(6,4))
plt.barh(coef_df["Feature"], coef_df["Coefficient"])
plt.axvline(0, linestyle="--")
plt.xlabel("Standardized Logistic Regression Coefficient")
plt.title("Logistic Regression: Factors Associated with Pass Success")
plt.tight_layout()
plt.show()

