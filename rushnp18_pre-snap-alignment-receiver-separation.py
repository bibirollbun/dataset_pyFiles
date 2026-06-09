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


# NFL Big Data Bowl 2026
# Pre-snap alignment and receiver separation
# Rushil Patel - GWU (University track)

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind, pearsonr

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)

# viz settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

print("NFL Big Data Bowl 2026 – pre-snap alignment project")
print("Loading data...\n")

# load data
DATA_DIR = "/kaggle/input/nfl-big-data-bowl-2026-analytics"
supp_paths = glob.glob(os.path.join(DATA_DIR, "**", "supplementary_data.csv"), recursive=True)
train_input_paths = sorted(glob.glob(os.path.join(DATA_DIR, "**", "input_2023_w*.csv"), recursive=True))

print(f"Found {len(train_input_paths)} weeks of tracking data")

# load supplementary data with coverage info
supp = pd.read_csv(supp_paths[0], low_memory=False)
supp_small = supp[[
    "game_id", "week", "play_id",
    "defenders_in_the_box", "team_coverage_man_zone", "team_coverage_type",
    "receiver_alignment", "route_of_targeted_receiver", "offense_formation",
    "pass_result", "pass_length", "yards_gained",
    "expected_points", "expected_points_added"
]].copy()

print(f"Loaded supplementary data: {supp_small.shape[0]:,} plays\n")

# pre-snap defensive features
print("Building pre-snap features...\n")

def extract_presnap_defensive_features(play_df):
    # grab how the defense is lined up at the snap
    snap_frame = play_df[play_df["frame_id"] == 1].copy()
    if snap_frame.empty:
        return None
    
    defense = snap_frame[snap_frame["player_side"] == "Defense"]
    receivers = snap_frame[snap_frame["player_role"].isin(["Targeted Receiver", "Other Route Runner"])]
    
    if defense.empty or receivers.empty:
        return None
    
    los = snap_frame["absolute_yardline_number"].iloc[0] if "absolute_yardline_number" in snap_frame.columns else 50
    
    # safety depth - how deep are the safeties?
    defense_sorted = defense.sort_values("x", ascending=False)
    if len(defense_sorted) >= 2:
        safety_depth = defense_sorted.iloc[:2]["x"].mean() - los
    else:
        safety_depth = 0
    
    # defender proximity - how close are defenders to receivers?
    if len(receivers) > 0 and len(defense) > 0:
        rec_positions = receivers[["x", "y"]].values
        def_positions = defense[["x", "y"]].values
        distances = cdist(rec_positions, def_positions, metric='euclidean')
        
        nearest_3_dists = []
        for rec_idx in range(len(rec_positions)):
            rec_dists = np.sort(distances[rec_idx])[:min(3, len(def_positions))]
            nearest_3_dists.extend(rec_dists)
        
        avg_defender_proximity = np.mean(nearest_3_dists) if nearest_3_dists else 0
        min_defender_proximity = np.min(nearest_3_dists) if nearest_3_dists else 0
    else:
        avg_defender_proximity = 0
        min_defender_proximity = 0
    
    # defensive shell - 0-high, 1-high, or 2-high
    deep_threshold = los + 12
    medium_threshold = los + 5
    
    defenders_deep = (defense["x"] > deep_threshold).sum()
    defenders_medium = ((defense["x"] > medium_threshold) & (defense["x"] <= deep_threshold)).sum()
    defenders_box = (defense["x"] <= medium_threshold).sum()
    
    if defenders_deep == 0:
        defensive_shell = "0-high"
    elif defenders_deep == 1:
        defensive_shell = "1-high"
    elif defenders_deep >= 2:
        defensive_shell = "2-high"
    else:
        defensive_shell = "unknown"
    
    # defensive spread
    def_spread_x = defense["x"].std() if len(defense) > 1 else 0
    def_spread_y = defense["y"].std() if len(defense) > 1 else 0
    defensive_compactness = def_spread_x + def_spread_y
    
    # tight coverage - defenders within 5 yards
    tight_coverage_count = 0
    if len(receivers) > 0 and len(defense) > 0:
        for _, rec in receivers.iterrows():
            rec_pos = np.array([rec["x"], rec["y"]])
            for _, defender in defense.iterrows():
                def_pos = np.array([defender["x"], defender["y"]])
                if np.linalg.norm(rec_pos - def_pos) < 5:
                    tight_coverage_count += 1
    
    return {
        "safety_depth": safety_depth,
        "avg_defender_proximity": avg_defender_proximity,
        "min_defender_proximity": min_defender_proximity,
        "defenders_deep": defenders_deep,
        "defenders_medium": defenders_medium,
        "defenders_box_presnap": defenders_box,
        "defensive_shell": defensive_shell,
        "defensive_compactness": defensive_compactness,
        "tight_coverage_count": tight_coverage_count
    }


# separation metrics
def compute_separation_timeseries(play_df):
    # how the space between target and nearest defender changes while ball is in the air
    rec_rows = play_df[play_df["player_role"] == "Targeted Receiver"][["nfl_id"]]
    if rec_rows.empty:
        return None
    rec_id = rec_rows["nfl_id"].iloc[0]

    rec_traj = play_df[play_df["nfl_id"] == rec_id][["frame_id", "x", "y"]].copy()
    cov_traj = play_df[play_df["player_role"] == "Defensive Coverage"][["frame_id", "x", "y"]].copy()
    
    if rec_traj.empty or cov_traj.empty:
        return None

    merged = cov_traj.merge(rec_traj, on="frame_id", suffixes=("_def", "_rec"))
    merged["dist"] = np.sqrt((merged["x_def"] - merged["x_rec"])**2 + 
                              (merged["y_def"] - merged["y_rec"])**2)
    
    frame_min = merged.groupby("frame_id", as_index=False)["dist"].min().rename(columns={"dist": "sep"})
    frame_min = frame_min.sort_values("frame_id")

    if frame_min.empty:
        return None

    seps = frame_min["sep"].values
    frames = frame_min["frame_id"].values

    sep_first = float(seps[0])
    sep_last = float(seps[-1])
    sep_mean = float(seps.mean())
    sep_max = float(seps.max())
    sep_min = float(seps.min())
    sep_std = float(seps.std()) if len(seps) > 1 else 0
    
    tight_window_frac = float((seps < 1.0).mean())
    open_window_frac = float((seps > 3.0).mean())
    
    if len(seps) > 1 and frames[-1] != frames[0]:
        sep_slope = float((seps[-1] - seps[0]) / (frames[-1] - frames[0]))
    else:
        sep_slope = 0.0
    
    max_sep_frame = int(frames[np.argmax(seps)])
    min_sep_frame = int(frames[np.argmin(seps)])
    
    return {
        "initial_separation": sep_first,
        "final_separation": sep_last,
        "mean_separation": sep_mean,
        "max_separation": sep_max,
        "min_separation": sep_min,
        "sep_std": sep_std,
        "tight_window_frac": tight_window_frac,
        "open_window_frac": open_window_frac,
        "sep_slope": sep_slope,
        "max_sep_frame": max_sep_frame,
        "min_sep_frame": min_sep_frame,
        "num_frames": len(seps)
    }


# process all plays
print("Processing plays...\n")

all_rows = []
WEEKS_TO_PROCESS = len(train_input_paths)  # all 18 weeks

for input_path in train_input_paths[:WEEKS_TO_PROCESS]:
    week_num = int(os.path.basename(input_path).split("_")[-1].replace("w", "").replace(".csv", ""))
    print(f"  Week {week_num:02d}...")

    df = pd.read_csv(input_path)
    grouped = df.groupby(["game_id", "play_id"], sort=False)

    for (game_id, play_id), play_df in grouped:
        presnap_features = extract_presnap_defensive_features(play_df)
        sep_stats = compute_separation_timeseries(play_df)
        
        if presnap_features is None or sep_stats is None:
            continue
        
        row = {"game_id": game_id, "play_id": play_id, "week": week_num}
        row.update(presnap_features)
        row.update(sep_stats)
        all_rows.append(row)

play_features_df = pd.DataFrame(all_rows)
print(f"\nExtracted features for {len(play_features_df):,} plays")

# merge with coverage data
print("Merging with coverage data...")

full_df = play_features_df.merge(supp_small, on=["game_id", "play_id"], how="left")
full_df = full_df.dropna(subset=["mean_separation", "team_coverage_man_zone"]).copy()

full_df["defenders_in_the_box"] = full_df["defenders_in_the_box"].fillna(-1).astype(int)
full_df["expected_points_added"] = full_df["expected_points_added"].fillna(0.0)

print(f"Final dataset: {full_df.shape[0]:,} plays with {full_df.shape[1]} features")

# coverage stress index
print("Creating coverage stress index...")

def zscore(x):
    x = x.astype(float)
    return (x - x.mean()) / (x.std() + 1e-6)

cs_components = pd.DataFrame({
    "z_min_sep": zscore(-full_df["min_separation"]),
    "z_tight_frac": zscore(full_df["tight_window_frac"]),
    "z_defender_prox": zscore(-full_df["min_defender_proximity"]),
    "z_epa": zscore(-full_df["expected_points_added"])
})

full_df["coverage_stress_index"] = cs_components.mean(axis=1)

# save
full_df.to_csv("play_level_features.csv", index=False)
print(f"Saved: play_level_features.csv")

print(f"\nDone – {len(full_df):,} plays ready for analysis")


# analysis & visualizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

full_df = pd.read_csv("play_level_features.csv")
print(f"Loaded {len(full_df):,} plays\n")

# --- man vs zone: first look ---
print("\n--- Man vs Zone ---")

man_sep = full_df[full_df["team_coverage_man_zone"] == "MAN_COVERAGE"]["mean_separation"]
zone_sep = full_df[full_df["team_coverage_man_zone"] == "ZONE_COVERAGE"]["mean_separation"]

t_stat, p_value = ttest_ind(zone_sep, man_sep)

print(f"Zone (n={len(zone_sep):,}): {zone_sep.mean():.3f} yards")
print(f"Man (n={len(man_sep):,}): {man_sep.mean():.3f} yards")
print(f"Difference: {zone_sep.mean() - man_sep.mean():.3f} yards")
print(f"p-value: {p_value:.4f}")

# visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# bar chart
ax1 = axes[0]
man_zone_data = full_df.groupby("team_coverage_man_zone").agg({
    "mean_separation": ["mean", "std", "count"]
}).reset_index()
man_zone_data.columns = ["Coverage", "Mean", "Std", "Count"]

colors = ["#ff7f0e", "#1f77b4"]
bars = ax1.bar(["Man", "Zone"], man_zone_data["Mean"], 
               yerr=man_zone_data["Std"]/np.sqrt(man_zone_data["Count"]),
               color=colors, alpha=0.8, capsize=5)
ax1.set_ylabel("Mean Separation (yards)")
ax1.set_title("Average separation by coverage")
ax1.grid(axis="y", alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontsize=12)

# distributions
ax2 = axes[1]
full_df[full_df["team_coverage_man_zone"] == "MAN_COVERAGE"]["mean_separation"].hist(
    bins=50, alpha=0.6, label="Man", color=colors[0], ax=ax2)
full_df[full_df["team_coverage_man_zone"] == "ZONE_COVERAGE"]["mean_separation"].hist(
    bins=50, alpha=0.6, label="Zone", color=colors[1], ax=ax2)
ax2.set_xlabel("Mean Separation (yards)")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution comparison")
ax2.legend()
ax2.grid(axis="y", alpha=0.3)

# tight windows
ax3 = axes[2]
tight_window_comp = full_df.groupby("team_coverage_man_zone")["tight_window_frac"].mean() * 100
bars3 = ax3.bar(["Man", "Zone"], tight_window_comp.values, color=colors, alpha=0.8)
ax3.set_ylabel("% Time <1 Yard Separation")
ax3.set_title("Tight windows")
ax3.grid(axis="y", alpha=0.3)

for i, bar in enumerate(bars3):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig("finding_1_man_vs_zone.png", dpi=300, bbox_inches="tight")
print("Saved: finding_1_man_vs_zone.png\n")

# --- defensive shells: 0/1/2-high ---
print("\n--- Defensive Shells ---")

shell_analysis = full_df.groupby("defensive_shell").agg({
    "mean_separation": ["count", "mean"],
    "tight_window_frac": "mean"
})
print(shell_analysis)

# visualize
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

common_shells = ["0-high", "1-high", "2-high"]
shell_df = full_df[full_df["defensive_shell"].isin(common_shells)]

# mean separation
ax1 = axes[0, 0]
shell_sep = shell_df.groupby("defensive_shell")["mean_separation"].mean().reindex(common_shells)
bars = ax1.bar(common_shells, shell_sep.values, color=["#d62728", "#2ca02c", "#1f77b4"], alpha=0.8)
ax1.set_ylabel("Mean Separation (yards)")
ax1.set_title("Separation by shell")
ax1.grid(axis="y", alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom')

# box plots
ax2 = axes[0, 1]
shell_df.boxplot(column="mean_separation", by="defensive_shell", ax=ax2, 
                 patch_artist=True, positions=[0, 1, 2])
ax2.set_xlabel("Defensive Shell")
ax2.set_ylabel("Mean Separation (yards)")
ax2.set_title("Distribution by shell")
ax2.set_xticklabels(common_shells)

# coverage stress
ax3 = axes[1, 0]
shell_stress = shell_df.groupby("defensive_shell")["coverage_stress_index"].mean().reindex(common_shells)
bars = ax3.bar(common_shells, shell_stress.values, color=["#d62728", "#2ca02c", "#1f77b4"], alpha=0.8)
ax3.set_ylabel("Coverage Stress Index")
ax3.set_title("Defensive pressure by shell")
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax3.grid(axis="y", alpha=0.3)

# tight windows
ax4 = axes[1, 1]
shell_tight = shell_df.groupby("defensive_shell")["tight_window_frac"].mean().reindex(common_shells) * 100
bars = ax4.bar(common_shells, shell_tight.values, color=["#d62728", "#2ca02c", "#1f77b4"], alpha=0.8)
ax4.set_ylabel("% Time <1 Yard")
ax4.set_title("Tight coverage by shell")
ax4.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("finding_2_defensive_shells.png", dpi=300, bbox_inches="tight")
print("Saved: finding_2_defensive_shells.png\n")

# --- pre-snap proximity vs separation ---
print("\n--- Pre-Snap Proximity ---")

corr, p_val = pearsonr(full_df["min_defender_proximity"], full_df["mean_separation"])
print(f"Correlation: r = {corr:.3f}, p = {p_val:.4f}")

# visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# scatter with trend
ax1 = axes[0]
sample = full_df.sample(min(5000, len(full_df)), random_state=42)
scatter = ax1.scatter(sample["min_defender_proximity"], sample["mean_separation"],
                     c=sample["coverage_stress_index"], cmap="RdYlGn_r",
                     alpha=0.5, s=20, edgecolors='none')
ax1.set_xlabel("Pre-Snap Defender Proximity (yards)")
ax1.set_ylabel("Mean Separation (yards)")
ax1.set_title("Pre-snap predicts post-snap")
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label("Coverage Stress")
ax1.grid(alpha=0.3)

# trend line
z = np.polyfit(full_df["min_defender_proximity"], full_df["mean_separation"], 1)
p = np.poly1d(z)
x_line = np.linspace(full_df["min_defender_proximity"].min(), 
                     full_df["min_defender_proximity"].max(), 100)
ax1.plot(x_line, p(x_line), "r--", linewidth=2, label=f"r={corr:.3f}")
ax1.legend()

# binned analysis
ax2 = axes[1]
proximity_bins = pd.cut(full_df["min_defender_proximity"], bins=10)
binned_sep = full_df.groupby(proximity_bins)["mean_separation"].agg(["mean", "count", "std"])
binned_sep = binned_sep[binned_sep["count"] >= 50]

bin_centers = [interval.mid for interval in binned_sep.index]
ax2.errorbar(bin_centers, binned_sep["mean"], 
            yerr=binned_sep["std"]/np.sqrt(binned_sep["count"]),
            marker='o', linestyle='-', linewidth=2, markersize=8, capsize=5)
ax2.set_xlabel("Pre-Snap Proximity (yards)")
ax2.set_ylabel("Mean Separation (yards)")
ax2.set_title("Binned analysis")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("finding_3_proximity_vs_separation.png", dpi=300, bbox_inches="tight")
print("Saved: finding_3_proximity_vs_separation.png\n")

# --- coverage schemes breakdown ---
print("\n--- Coverage Schemes ---")

scheme_analysis = full_df.groupby("team_coverage_type").agg({
    "mean_separation": ["count", "mean"],
    "coverage_stress_index": "mean",
    "tight_window_frac": "mean"
})

scheme_analysis.columns = ["plays", "avg_separation", "stress", "tight_frac"]
scheme_analysis = scheme_analysis[scheme_analysis["plays"] >= 100]
scheme_analysis = scheme_analysis.sort_values("stress", ascending=False)

print(scheme_analysis.head(10))

# visualize
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

top_schemes = scheme_analysis.head(8).index

# separation by scheme
ax1 = axes[0, 0]
x_pos = np.arange(len(top_schemes))
bars = ax1.barh(x_pos, scheme_analysis.loc[top_schemes, "avg_separation"], 
                color="steelblue", alpha=0.8)
ax1.set_yticks(x_pos)
ax1.set_yticklabels(top_schemes)
ax1.set_xlabel("Average Separation (yards)")
ax1.set_title("Separation by scheme")
ax1.grid(axis="x", alpha=0.3)

# stress by scheme
ax2 = axes[0, 1]
bars = ax2.barh(x_pos, scheme_analysis.loc[top_schemes, "stress"], 
                color="darkred", alpha=0.8)
ax2.set_yticks(x_pos)
ax2.set_yticklabels(top_schemes)
ax2.set_xlabel("Coverage Stress Index")
ax2.set_title("Pressure by scheme")
ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax2.grid(axis="x", alpha=0.3)

# separation vs EPA
ax3 = axes[1, 0]
epa_by_scheme = full_df.groupby("team_coverage_type")["expected_points_added"].mean()
common_schemes_for_plot = scheme_analysis.index.intersection(epa_by_scheme.index)

ax3.scatter(scheme_analysis.loc[common_schemes_for_plot, "avg_separation"], 
           epa_by_scheme.loc[common_schemes_for_plot],
           s=scheme_analysis.loc[common_schemes_for_plot, "plays"]/10, 
           alpha=0.6, edgecolors='black')

for idx in list(top_schemes[:5]):
    if idx in common_schemes_for_plot:
        ax3.annotate(idx, 
                    (scheme_analysis.loc[idx, "avg_separation"], 
                     epa_by_scheme.loc[idx]),
                    fontsize=9, alpha=0.7)

ax3.set_xlabel("Average Separation (yards)")
ax3.set_ylabel("Average EPA")
ax3.set_title("Separation vs success")
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax3.grid(alpha=0.3)

# tight windows by scheme
ax4 = axes[1, 1]
bars = ax4.barh(x_pos, scheme_analysis.loc[top_schemes, "tight_frac"] * 100, 
                color="darkgreen", alpha=0.8)
ax4.set_yticks(x_pos)
ax4.set_yticklabels(top_schemes)
ax4.set_xlabel("% Time <1 Yard")
ax4.set_title("Tight windows by scheme")
ax4.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("finding_4_coverage_schemes.png", dpi=300, bbox_inches="tight")
print("Saved: finding_4_coverage_schemes.png\n")

print("\nDone – generated 4 figures")

