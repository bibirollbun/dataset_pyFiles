# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob, os



train_path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
root_path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final"

# --- Load only Week 1 and Week 2 input/output files ---
input_files = [
    os.path.join(train_path, "input_2023_w01.csv"),
    os.path.join(train_path, "input_2023_w02.csv")
]

output_files = [
    os.path.join(train_path, "output_2023_w01.csv"),
    os.path.join(train_path, "output_2023_w02.csv")
]




# --- Load supplementary from parent folder ---
supplementary_path = os.path.join(root_path, "supplementary_data.csv")

# --- Load dataframes ---
input_df = pd.concat([pd.read_csv(f, low_memory=False) for f in input_files], ignore_index=True)
output_df = pd.concat([pd.read_csv(f, low_memory=False) for f in output_files], ignore_index=True)
supp_df = pd.read_csv(supplementary_path, low_memory=False)



input_df.head()


supp_df.head()


output_df.head()


output_df["game_id"] = output_df["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)
output_df["play_id"] = output_df["play_id"].astype(str).str.replace(r"\.0$", "", regex=True) 


supp_df["game_id"] = supp_df["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)
supp_df["play_id"] = supp_df["play_id"].astype(str).str.replace(r"\.0$", "", regex=True)
print("Output sample IDs:", output_df[["game_id", "play_id"]].head(2).values)
print("Supp sample IDs:", supp_df[["game_id", "play_id"]].head(2).values)


merge_keys = ["game_id", "play_id"]
merged_df = output_df.merge(supp_df, on=merge_keys, how="left")
print("Merged shape:", merged_df.shape)


input_df.columns


merged_df.columns


player_info_cols = [
    "game_id", "play_id", "nfl_id",
    "player_name", "player_side", "player_position", "player_role"
]

player_info = input_df[player_info_cols].drop_duplicates()
for df in [player_info, merged_df]:
    df["game_id"] = df["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["play_id"] = df["play_id"].astype(str).str.replace(r"\.0$", "", regex=True)



merged_df = merged_df.merge(
    player_info,
    on=["game_id", "play_id", "nfl_id"],
    how="left"
)
print("New merged_df shape:", merged_df.shape)
print("Sample columns:", [col for col in merged_df.columns if 'player' in col or 'role' in col])



roles_of_interest = ["Targeted Receiver", "Defensive Coverage"]
filtered_df = merged_df[merged_df["player_role"].isin(roles_of_interest)].copy()
print(f"Filtered data shape: {filtered_df.shape}")
print("Unique roles:", filtered_df['player_role'].dropna().unique())


receivers_df = filtered_df[filtered_df["player_role"] == "Targeted Receiver"].copy()
defenders_df = filtered_df[filtered_df["player_role"] == "Defensive Coverage"].copy()

print("Receivers:", receivers_df.shape, " | Defenders:", defenders_df.shape)


pairs_df = receivers_df.merge(
    defenders_df,
    on=["game_id", "play_id", "frame_id"],
    suffixes=("_rec", "_def")
)

pairs_df["distance"] = np.sqrt(
    (pairs_df["x_rec"] - pairs_df["x_def"])**2 +
    (pairs_df["y_rec"] - pairs_df["y_def"])**2
)


separation_df = (
    pairs_df.groupby(["game_id", "play_id", "frame_id"], as_index=False)["distance"]
    .min()
    .rename(columns={"distance": "min_separation"})
)

receivers_df = receivers_df.merge(
    separation_df,
    on=["game_id", "play_id", "frame_id"],
    how="left"
)


print("Receiver dataset shape:", receivers_df.shape)
print(receivers_df[["game_id", "play_id", "frame_id", "player_name", "min_separation"]].head())


receivers = receivers_df.copy()

receivers_clean = receivers.dropna(
    subset=["min_separation", "route_of_targeted_receiver", "team_coverage_man_zone", "pass_result"]
)
receivers_clean.shape


route_sep = (
    receivers_clean.groupby("route_of_targeted_receiver")["min_separation"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
)



plt.figure(figsize=(10,5))
sns.barplot(x=route_sep.values, y=route_sep.index, palette="viridis")
plt.title("Average Receiver Separation by Route Type", fontsize=14)
plt.xlabel("Average Minimum Separation (yards)")
plt.ylabel("Route Type")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
sns.boxplot(
    data=receivers_clean,
    x="team_coverage_man_zone",
    y="min_separation",
    palette="coolwarm"
)
plt.title("Separation Distribution by Coverage Type", fontsize=13)
plt.xlabel("Coverage Type")
plt.ylabel("Min Separation (yards)")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
sns.boxplot(
    data=receivers_clean,
    x="pass_result",
    y="min_separation",
    order=["C", "I"],
    palette="Set2"
)
plt.title("Receiver Separation vs Pass Result", fontsize=13)
plt.xlabel("Pass Result (C = Complete, I = Incomplete)")
plt.ylabel("Min Separation (yards)")
plt.tight_layout()
plt.show()


summary = (
    receivers_clean.groupby(["team_coverage_man_zone", "pass_result"])["min_separation"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .sort_values("mean", ascending=False)
)

display(summary.head(10))


route_difficulty_map = {
    "SCREEN": 0.8, "FLAT": 0.8, "ANGLE": 1.0, "WHEEL": 1.0,
    "CROSS": 1.0, "OUT": 1.0, "HITCH": 0.9, "CORNER": 1.2,
    "POST": 1.2, "SLANT": 1.1, "IN": 1.1, "GO": 1.3
}

coverage_map = {
    "MAN_COVERAGE": 1.2,
    "ZONE_COVERAGE": 1.0
}



receivers_esi = receivers_df.copy()

receivers_esi["route_difficulty"] = receivers_esi["route_of_targeted_receiver"].map(route_difficulty_map).fillna(1.0)
receivers_esi["coverage_tightness"] = receivers_esi["team_coverage_man_zone"].map(coverage_map).fillna(1.0)



receivers_esi["ESI"] = receivers_esi["min_separation"] / (receivers_esi["route_difficulty"] * receivers_esi["coverage_tightness"])
print(receivers_esi[["route_of_targeted_receiver", "team_coverage_man_zone", "min_separation", "ESI"]].head())
esi_summary = (
    receivers_esi.groupby("route_of_targeted_receiver")["ESI"]
    .mean()
    .sort_values(ascending=False)
)
display(esi_summary.head(10))


plt.figure(figsize=(6,4))
sns.boxplot(data=receivers_esi, x="team_coverage_man_zone", y="ESI", palette="mako")
plt.title("Effective Separation Index by Coverage Type", fontsize=13)
plt.xlabel("Coverage Type")
plt.ylabel("ESI (Normalized Separation)")
plt.tight_layout()
plt.show()


player_esi = receivers_esi.groupby("player_name")["ESI"].mean().sort_values(ascending=False).head(10)
display(player_esi)



esi_df = receivers_esi.dropna(subset=["ESI", "pass_result"])
top_players = (
    esi_df.groupby("player_name")["ESI"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)


plt.figure(figsize=(8,5))
sns.barplot(x=top_players.values, y=top_players.index, palette="crest")
plt.title("Top 10 Players by Effective Separation Index (ESI)", fontsize=14)
plt.xlabel("Average ESI (Normalized Separation)")
plt.ylabel("Player Name")
plt.tight_layout()
plt.show()


top_routes = (
    esi_df.groupby("route_of_targeted_receiver")["ESI"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(8,5))
sns.barplot(x=top_routes.values, y=top_routes.index, palette="viridis")
plt.title("Top 10 Routes by Effective Separation Index (ESI)", fontsize=14)
plt.xlabel("Average ESI")
plt.ylabel("Route Type")
plt.tight_layout()
plt.show()


esi_df["is_complete"] = esi_df["pass_result"].apply(lambda x: 1 if x == "C" else 0)
corr = esi_df[["ESI", "is_complete"]].corr().iloc[0,1]

plt.figure(figsize=(6,4))
sns.boxplot(data=esi_df, x="pass_result", y="ESI", order=["C","I"], palette="coolwarm")
plt.title(f"ESI Distribution by Pass Result (corr = {corr:.2f})", fontsize=13)
plt.xlabel("Pass Result (C = Complete, I = Incomplete)")
plt.ylabel("Effective Separation Index (ESI)")
plt.tight_layout()
plt.show()


summary = (
    esi_df.groupby("pass_result")["ESI"]
    .agg(["mean","std","count"])
    .reset_index()
)
display(summary)

