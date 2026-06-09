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


!pip install --upgrade scikit-learn



# =====================================================
# KAGGLE NOTEBOOK â€” GLOBAL SETUP (RUN ONCE)
# =====================================================

# ---------------------
# Core Libraries
# ---------------------

# ---------------------
# Visualization
# ---------------------
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.family"] = "sans-serif"

# ---------------------
# Modeling / Statistics
# ---------------------
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

import statsmodels.api as sm
import statsmodels.formula.api as smf

import joblib

# ---------------------
# Kaggle Paths
# ---------------------
# CHANGE THIS if your dataset folder name differs
DATASET_SLUG = "input-and-results"

INPUT_DIR = f"/kaggle/input/{DATASET_SLUG}"
OUTPUT_DIR = "/kaggle/working"

METRICS_PATH_BASELINE = os.path.join(INPUT_DIR, "metrics_playlevel_baseline.parquet")
METRICS_PATH_SUPERVISED = os.path.join(INPUT_DIR, "metrics_playlevel_supervised (3).parquet")
SUPP_PATH = os.path.join(INPUT_DIR, "supplementary_data.csv")
METRICS_PATH_SUPERVISED = "/kaggle/input/input-and-results/metrics_playlevel_supervised (3).parquet"



# Output folders
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------
# Sanity Checks
# ---------------------
print(" Kaggle INPUT_DIR contents:")
if os.path.exists(INPUT_DIR):
    for f in os.listdir(INPUT_DIR):
        print("  -", f)
else:
    raise FileNotFoundError(f"INPUT_DIR not found: {INPUT_DIR}")

print("\n OUTPUT_DIR:", OUTPUT_DIR)
print(" MODELS_DIR:", MODELS_DIR)
print(" FIGURES_DIR:", FIGURES_DIR)

# ---------------------
# Global Constants
# ---------------------
SEED = 42
np.random.seed(SEED)

print("\n Setup complete. Ready for analysis.")


import warnings

warnings.filterwarnings("ignore")

# =========================
# KAGGLE PATHS (EDIT THESE)
# =========================
# 1) Put your CSV inside a Kaggle Dataset and attach it to the notebook.
# 2) Replace DATASET_SLUG with the folder name under /kaggle/input/.
DATASET_SLUG = "input-and-results"  # e.g. "my-dataset-name" as it appears in /kaggle/input/
INPUT_FILENAME = "animation_data_2023090700_101.csv"

INPUT_CSV = f"/kaggle/input/{DATASET_SLUG}/{INPUT_FILENAME}"

# Writable output dir
OUTPUT_DIR = "/kaggle/working/analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEAM_THEME = "seahawks"  # or "vikings"

# Analysis toggles
GENERATE_POSITION_HEATMAPS = True
GENERATE_STRESS_CORRELATIONS = True
GENERATE_MOVEMENT_ANALYSIS = True
GENERATE_FORMATION_ANALYSIS = True
GENERATE_TIME_SERIES = True

print("INPUT_CSV =", INPUT_CSV)
print("OUTPUT_DIR =", OUTPUT_DIR)


print("sklearn version:", sklearn.__version__)


warnings.filterwarnings("ignore")

# =========================
# KAGGLE PATHS (EDIT THESE)
# =========================
DATASET_SLUG = "input-and-results"   # folder name under /kaggle/input/
INPUT_FILENAME = "animation_data_2023090700_101.csv"

INPUT_CSV = f"/kaggle/input/{DATASET_SLUG}/{INPUT_FILENAME}"

OUTPUT_DIR = "/kaggle/working/player_analysis"
TEAM_THEME = "seahawks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("INPUT_CSV =", INPUT_CSV)
print("OUTPUT_DIR =", OUTPUT_DIR)



DATASET_SLUG = "input-and-results"      # folder name under /kaggle/input/
INPUT_CSV_NAME = "vikings_play_79_data.csv"

INPUT_CSV = f"/kaggle/input/{DATASET_SLUG}/{INPUT_CSV_NAME}"
OUTPUT_DIR = "/kaggle/working/team_animations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("INPUT_CSV =", INPUT_CSV)
print("OUTPUT_DIR =", OUTPUT_DIR)
print("Available /kaggle/input folders:", os.listdir("/kaggle/input"))


"""
Phase 5 â€” Supervised DCI Calibration 
====================================================
Improvements:
1.  Adds Contextual Features (Down, Distance, Defenders in Box).
2.  Adds Feature Interactions (Ratios).
3.  Uses HistGradientBoostingClassifier.
4.  Proper Categorical Handling for Cluster IDs.
"""

# -----------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------

print("[INFO] Loading baseline metrics...")
df = pd.read_parquet(METRICS_PATH_BASELINE)

print("[INFO] Loading ground truth labels...")
supp = pd.read_csv(SUPP_PATH, low_memory=False)

# --- FIX: ROBUST RENAMING & EXTRA COLUMNS ---
cols_map = {
    "gameId": "game_id", 
    "playId": "play_id", 
    "passResult": "pass_result",
    "expectedPointsAdded": "epa",
    "expected_points_added": "epa",
    "down": "down",
    "yardsToGo": "yards_to_go",
    "defendersInTheBox": "defenders_in_the_box"
}
supp.rename(columns=cols_map, inplace=True)

# Merge
merged = df.merge(supp, on=["game_id", "play_id"], how="inner")

# Filter Pass Plays
valid_pass_types = ['C', 'I', 'S', 'IN']
pass_df = merged[merged['pass_result'].isin(valid_pass_types)].copy()

print(f"[INFO] Dataset filtered. Analyzing {len(pass_df)} valid pass plays.")

# Define Target (1 = Good Defense)(Good Defense is any play where EPA <= 0)
pass_df['defensive_success'] = (pass_df['epa'] <= 0).astype(int)

# -----------------------------------------------------------
# FEATURE ENGINEERING (THE UPGRADE)
# -----------------------------------------------------------

print("[INFO] Engineering contextual features...")

# 1. Fill NaNs in Context Features
pass_df['down'] = pass_df['down'].fillna(1).astype(int)
pass_df['yards_to_go'] = pass_df['yards_to_go'].fillna(10).astype(int)
pass_df['defenders_in_the_box'] = pass_df['defenders_in_the_box'].fillna(6).astype(int)

# 2. Feature Interactions (Ratios)
# "Integrity per Distance Unit": Does strict integrity compensate for being far away?
pass_df['integrity_dist_ratio'] = pass_df['integrity_proxy'] / (pass_df['distance_to_ideal'] + 1e-6)

# 3. Categorical Handling
pass_df['cluster_id'] = pass_df['cluster_id'].astype('category')

features = [
    'distance_to_ideal', 
    'distance_to_second', 
    'spacing_proxy', 
    'integrity_proxy',
    'integrity_dist_ratio',  
    'down',                  
    'yards_to_go',           
    'defenders_in_the_box',         
    'cluster_id'
]

X = pass_df[features]
y = pass_df['defensive_success'].values

# -----------------------------------------------------------
# MODEL TRAINING 
# -----------------------------------------------------------

print("[INFO] Training HistGradientBoostingClassifier...")

clf = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=200,
    max_depth=6,
    l2_regularization=0.1,
    random_state=SEED,
    categorical_features='from_dtype'
)

# Cross-Validation Predictions
print("[INFO] Generating calibrated DCI scores via 5-Fold CV...")
dci_probs = cross_val_predict(
    clf, X, y, cv=5, method='predict_proba'
)[:, 1]

# Fit final model
clf.fit(X, y)

# -----------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------

auc_score = roc_auc_score(y, dci_probs)
print(f"\n MODEL PERFORMANCE (UPGRADED):")
print(f"   AUC: {auc_score:.4f}")

# Correlation
pass_df['dci_supervised'] = dci_probs
corr_epa = pass_df[['dci_supervised', 'epa']].corr().iloc[0, 1]

print(f"\n DCI vs EPA CORRELATION: {corr_epa:.4f}")

# -----------------------------------------------------------
# EXPORT
# -----------------------------------------------------------

pass_df['dis_final'] = pass_df['integrity_proxy']

output_cols = [
    'game_id', 'play_id', 
    'dci_supervised', 
    'dis_final', 
    'epa', 
    'pass_result', 
    'cluster_id'
]

OUT_PATH = os.path.join(OUTPUT_DIR, "metrics_playlevel_supervised_v3.parquet")
pass_df[output_cols].to_parquet(OUT_PATH, index=False)
print(f"\n[INFO] Final Optimized Metrics saved to: {OUT_PATH}")

MODEL_OUT = os.path.join(MODELS_DIR, "dci_calibrator.pkl")
joblib.dump(clf, MODEL_OUT)
print(f"[INFO] Saved DCI Calibrator model to: {MODEL_OUT}")


"""
Advanced Defensive Performance Map (Isoquants + Pareto + Hulls)
==============================================================

INPUTS
------
1. metrics_playlevel_supervised.parquet
2. supplementary_data.csv

OUTPUTS
-------
1. defensive_performance_advanced.png
"""

# -------------------------------------------------------
# 1. CONFIGURATION (KAGGLE)
# -------------------------------------------------------
METRICS_PATH = os.path.join(INPUT_DIR, "metrics_playlevel_supervised (3).parquet")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "defensive_performance_advanced.png")

# Analysis Constants
TOP_N_LABELS = 10
WEIGHT_DCI = 0.5
WEIGHT_DIS = 0.5

# -------------------------------------------------------
# 2. DATA LOADING & PROCESSING
# -------------------------------------------------------
print("Loading data...")
df_metrics = pd.read_parquet(METRICS_PATH)
df_supp = pd.read_csv(SUPP_PATH, low_memory=False)

# Standardize Columns
df_supp.rename(columns={
    "gameId": "game_id", "playId": "play_id", "defensiveTeam": "defensive_team"
}, inplace=True)

# Merge
print("Merging datasets...")
merged_df = df_metrics.merge(
    df_supp[["game_id", "play_id", "defensive_team"]],
    on=["game_id", "play_id"],
    how="inner"
)

# Aggregation: Mean metrics AND Count (for sizing)
team_stats = merged_df.groupby("defensive_team").agg(
    dci_supervised=("dci_supervised", "mean"),
    dis_final=("dis_final", "mean"),
    play_count=("play_id", "count")
).reset_index()

# -------------------------------------------------------
# 3. SCORING & Z-SCORES
# -------------------------------------------------------
team_stats["dci_z"] = (team_stats["dci_supervised"] - team_stats["dci_supervised"].mean()) / team_stats["dci_supervised"].std()
team_stats["dis_z"] = (team_stats["dis_final"] - team_stats["dis_final"].mean()) / team_stats["dis_final"].std()

team_stats["elite_score"] = (team_stats["dci_z"] * WEIGHT_DCI) + (team_stats["dis_z"] * WEIGHT_DIS)

team_stats = team_stats.sort_values("elite_score", ascending=False).reset_index(drop=True)
team_stats["rank"] = team_stats.index + 1

# -------------------------------------------------------
# 4. PARETO FRONTIER CALCULATION
# -------------------------------------------------------
def get_pareto_frontier(df, x_col, y_col):
    sorted_df = df.sort_values(x_col, ascending=False)
    pareto_front = []
    max_y = -np.inf

    for row in sorted_df.itertuples():
        if getattr(row, y_col) >= max_y:
            pareto_front.append((getattr(row, x_col), getattr(row, y_col)))
            max_y = getattr(row, y_col)

    return sorted(pareto_front, key=lambda x: x[0])

pareto_points = get_pareto_frontier(team_stats, "dci_supervised", "dis_final")
pareto_x, pareto_y = zip(*pareto_points) if pareto_points else ([], [])

# -------------------------------------------------------
# 5. PLOTTING
# -------------------------------------------------------
plt.figure(figsize=(14, 11))
ax = plt.gca()

# --- A. Isoquants (Contour Lines) ---
x_min, x_max = team_stats["dci_supervised"].min(), team_stats["dci_supervised"].max()
y_min, y_max = team_stats["dis_final"].min(), team_stats["dis_final"].max()
pad_x, pad_y = (x_max - x_min)*0.1, (y_max - y_min)*0.1

xi = np.linspace(x_min - pad_x, x_max + pad_x, 100)
yi = np.linspace(y_min - pad_y, y_max + pad_y, 100)
X, Y = np.meshgrid(xi, yi)

mu_dci, sig_dci = team_stats["dci_supervised"].mean(), team_stats["dci_supervised"].std()
mu_dis, sig_dis = team_stats["dis_final"].mean(), team_stats["dis_final"].std()

Z_dci = (X - mu_dci) / sig_dci
Z_dis = (Y - mu_dis) / sig_dis
Z_score = (Z_dci * WEIGHT_DCI) + (Z_dis * WEIGHT_DIS)

levels = np.linspace(Z_score.min(), Z_score.max(), 8)
cntr = plt.contour(X, Y, Z_score, levels=levels, colors='gray', alpha=0.2, linestyles='dashed')
plt.clabel(cntr, inline=True, fontsize=8, fmt='Score: %.1f')

# --- B. Convex Hull (Elite Tier) ---
top_tier = team_stats.head(8)[["dci_supervised", "dis_final"]].values
if len(top_tier) >= 3:
    hull = ConvexHull(top_tier)
    hull_points = top_tier[hull.vertices]
    hull_points = np.append(hull_points, [hull_points[0]], axis=0)

    plt.fill(hull_points[:,0], hull_points[:,1], color='gold', alpha=0.1, label='Elite Tier')
    plt.plot(hull_points[:,0], hull_points[:,1], color='gold', alpha=0.4, linestyle='--')

# --- C. Main Scatter Plot ---
sizes = (team_stats["play_count"] / team_stats["play_count"].max()) * 400 + 100

sc = plt.scatter(
    team_stats["dci_supervised"],
    team_stats["dis_final"],
    c=team_stats["elite_score"],
    s=sizes,
    cmap="viridis",
    alpha=0.85,
    edgecolors="white",
    linewidth=2,
    zorder=3
)

cbar = plt.colorbar(sc, pad=0.02)
cbar.set_label("Composite Elite Score (z-DCI + z-DIS)", rotation=270, labelpad=15, fontweight='bold')

# --- D. Pareto Frontier Line ---
plt.plot(pareto_x, pareto_y, color='crimson', linestyle='-', linewidth=2.5, alpha=0.6, zorder=2, label='Pareto Frontier')

# --- E. Labels & Reference Lines ---
plt.axvline(mu_dci, color="black", linestyle=":", alpha=0.4)
plt.axhline(mu_dis, color="black", linestyle=":", alpha=0.4)
plt.text(x_max, mu_dis, "League Avg DIS", ha="right", va="bottom", fontsize=9, alpha=0.5)
plt.text(mu_dci, y_max, "League Avg DCI", ha="left", va="top", fontsize=9, alpha=0.5, rotation=90)

top_teams_list = team_stats.head(TOP_N_LABELS)
offset_cycle = [(15, 15), (15, -15), (-15, 15), (-15, -15), (0, 20), (0, -20)]

for i, row in top_teams_list.iterrows():
    dx, dy = offset_cycle[i % len(offset_cycle)]
    label_txt = f"{row['defensive_team']}\n#{row['rank']}"

    plt.annotate(
        label_txt,
        (row["dci_supervised"], row["dis_final"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        ha='center',
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9, lw=0.5),
        arrowprops=dict(arrowstyle="-", color="black", alpha=0.3)
    )

# -------------------------------------------------------
# 6. FINAL LAYOUT
# -------------------------------------------------------
plt.title("Defensive Landscape: The Elite Frontier", fontsize=20, fontweight="bold", pad=15)
plt.xlabel("Defensive Coverage Index (DCI) â†’ Higher is Tighter", fontsize=13, fontweight='bold')
plt.ylabel("Defensive Integrity Score (DIS) â†’ Higher is Better", fontsize=13, fontweight='bold')

info_text = (
    "â—‹ Point Size: Play Volume (Sample Size)\n"
    "-- Dashed Lines: Efficiency Isoquants\n"
    "â–¬ Red Line: Pareto Frontier (Unbeatable Tradeoffs)\n"
    "Yellow Zone: Elite Defense Cluster"
)
plt.text(x_min, y_max, info_text, fontsize=9, va='top', ha='left',
         bbox=dict(boxstyle="round", fc="whitesmoke", ec="none", alpha=0.8))

plt.grid(True, linestyle='--', alpha=0.15)
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"[SUCCESS] Plot saved to: {OUTPUT_FILE}")

# plt.show()



# ==========================================
# 1. CONFIGURATION & DATA LOADING (KAGGLE)
# ==========================================

# Put the dataset folder name shown under /kaggle/input/ here:
# Example: "input-and-results"
DATASET_SLUG = "input-and-results"

INPUT_DIR = f"/kaggle/input/{DATASET_SLUG}"
OUTPUT_DIR = "/kaggle/working"  # writable + persisted in Kaggle outputs

METRICS_PATH = os.path.join(INPUT_DIR, "metrics_playlevel_supervised (3).parquet")
SUPP_PATH = os.path.join(INPUT_DIR, "supplementary_data.csv")

print("[INFO] Loading datasets...")
try:
    df_metrics = pd.read_parquet(METRICS_PATH)
    df_supp = pd.read_csv(SUPP_PATH, low_memory=False)
    print(f"   -> Metrics loaded: {len(df_metrics)} rows")
    print(f"   -> Supplement loaded: {len(df_supp)} rows")
except Exception as e:
    print(f"[ERROR] Could not load files: {e}")
    print("[DEBUG] Files in INPUT_DIR:")
    try:
        for f in os.listdir(INPUT_DIR):
            print(" -", f)
    except Exception as e2:
        print("   (Could not list INPUT_DIR)", e2)
    raise

# ==========================================
# 1b. SUPPLEMENTARY CLEANING / RENAME (MATCH YOUR CSV)
# ==========================================

rename_map = {
    'gameId': 'game_id',
    'playId': 'play_id',

    'possession_team': 'posteam',
    'defensive_team': 'defteam',

    'team_coverage_type': 'team_coverage_type',
    'pre_snap_home_team_win_probability': 'pre_snap_home_team_win_probability',
    'home_team_win_probability_added': 'home_team_win_probability_added',

    # IMPORTANT: misspelled in your CSV
    'visitor_team_win_probility_added': 'visitor_team_win_probability_added',

    # Avoid conflicts with parquet (we will use parquet's EPA/pass_result when available)
    'expected_points_added': 'epa_supp',
    'pass_result': 'pass_result_supp',

    'home_team_abbr': 'homeTeamAbbr',
    'visitor_team_abbr': 'visitorTeamAbbr',

    'yards_to_go': 'yardsToGo'
}
df_supp.rename(columns=rename_map, inplace=True)

print("[DEBUG] posteam/defteam present:",
      'posteam' in df_supp.columns, 'defteam' in df_supp.columns)
print("[DEBUG] visitor WP added present:",
      'visitor_team_win_probability_added' in df_supp.columns)

# ==========================================
# 1c. SELECT ONLY USEFUL COLUMNS FROM SUPP
# ==========================================

cols_to_keep = [
    'game_id', 'play_id', 'posteam', 'defteam',
    'team_coverage_type',
    'pre_snap_home_team_win_probability',
    'home_team_win_probability_added',
    'visitor_team_win_probability_added',
    'homeTeamAbbr', 'visitorTeamAbbr',
    'quarter', 'down', 'yardsToGo', 'play_description'
]

existing_cols = [c for c in cols_to_keep if c in df_supp.columns]
df_supp = df_supp[existing_cols].copy()

missing_keys = [c for c in ['game_id', 'play_id'] if c not in df_supp.columns]
if missing_keys:
    raise KeyError(f"supplementary_data.csv missing required merge keys: {missing_keys}")

# ==========================================
# 1d. MERGE
# ==========================================
print("[INFO] Merging data...")
df = df_metrics.merge(df_supp, on=['game_id', 'play_id'], how='inner')
print(f"   -> Merged dataset size: {len(df)} rows")
print("[DEBUG] merged columns contain posteam?", 'posteam' in df.columns)

if 'yardsToGo' in df.columns:
    df['ydstogo'] = df['yardsToGo']

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
print("[INFO] Engineering features...")

if 'epa' not in df.columns:
    if 'epa_supp' in df.columns:
        df['epa'] = df['epa_supp']
    else:
        df['epa'] = 0.0

df['is_explosive'] = (df['epa'] >= 2.0).astype(int)

if 'dci_supervised' not in df.columns:
    raise KeyError("Column 'dci_supervised' not found in metrics parquet. Verify METRICS_PATH contents.")
df['dci_quartile'] = pd.qcut(df['dci_supervised'], 4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])

df['dci_z'] = (df['dci_supervised'] - df['dci_supervised'].mean()) / df['dci_supervised'].std()

if 'dis_final' not in df.columns:
    for alt in ['dis', 'distance', 'displacement', 'dis_mean']:
        if alt in df.columns:
            df['dis_final'] = df[alt]
            print(f"[WARN] Using '{alt}' as fallback for dis_final.")
            break
    else:
        df['dis_final'] = 0.0
        print("[WARN] No displacement column found. Setting dis_final = 0.0")

df['dis_z'] = (df['dis_final'] - df['dis_final'].mean()) / df['dis_final'].std() if df['dis_final'].std() != 0 else 0.0

if 'pre_snap_home_team_win_probability' not in df.columns:
    df['pre_snap_home_team_win_probability'] = 0.5

df['pre_snap_win_probability'] = df['pre_snap_home_team_win_probability']

if 'quarter' not in df.columns:
    df['quarter'] = 1

df['is_high_leverage'] = (
    (df['pre_snap_win_probability'].between(0.35, 0.65)) &
    (df['quarter'] >= 3)
)

if 'play_description' in df.columns:
    df['is_deep'] = df['play_description'].astype(str).str.contains("deep", case=False, na=False)
else:
    df['is_deep'] = False

if all(c in df.columns for c in ['posteam', 'homeTeamAbbr', 'home_team_win_probability_added']):
    df['wp_damage'] = np.where(
        df['posteam'] == df['homeTeamAbbr'],
        df['home_team_win_probability_added'],
        -df['home_team_win_probability_added']
    )
else:
    df['wp_damage'] = 0.0
    print("[WARN] Missing posteam/homeTeamAbbr/home_team_win_probability_added; wp_damage set to 0.")

# ==========================================
# 3. THE MONEY FIGURE (4 PANELS)
# ==========================================
print("[INFO] Generating 'The Money Figure'...")
sns.set_theme(style="whitegrid", context="talk")
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# --- PANEL A: Explosive Rate by Coverage & DCI ---
if 'team_coverage_type' in df.columns:
    top_coverages = df['team_coverage_type'].value_counts().nlargest(4).index
    df_cov = df[df['team_coverage_type'].isin(top_coverages)].copy()

    sns.barplot(
        data=df_cov, x='dci_quartile', y='is_explosive', hue='team_coverage_type',
        ax=axes[0, 0], palette="viridis", errorbar=('ci', 95)
    )
    axes[0, 0].set_title("Panel A: Explosive Rate by DCI (Within Coverage)", fontweight='bold')
    axes[0, 0].set_ylabel("Explosive Probability (EPA >= 2)")
    axes[0, 0].legend(title="Coverage")
else:
    sns.barplot(
        data=df, x='dci_quartile', y='is_explosive',
        ax=axes[0, 0], palette="viridis", errorbar=('ci', 95)
    )
    axes[0, 0].set_title("Panel A: Explosive Rate by DCI (All Coverages)", fontweight='bold')
    axes[0, 0].set_ylabel("Explosive Probability (EPA >= 2)")

# --- PANEL B: Win Probability Impact ---
df_explosive = df[df['is_explosive'] == 1].copy()
if not df_explosive.empty:
    sns.lineplot(
        data=df_explosive, x='dci_quartile', y='wp_damage',
        ax=axes[0, 1], marker='o', linewidth=3
    )
else:
    axes[0, 1].text(0.5, 0.5, "No Explosive Plays to Analyze", ha='center', va='center')

axes[0, 1].set_title("Panel B: WP Damage of Explosive Plays", fontweight='bold')
axes[0, 1].set_ylabel("Avg Win Prob Added by Offense")
axes[0, 1].set_xlabel("DCI Quartile")

# --- PANEL C: High-Leverage Situations ---
df_leverage = df[df['is_high_leverage']].copy()
if not df_leverage.empty:
    sns.barplot(
        data=df_leverage, x='dci_quartile', y='is_explosive',
        ax=axes[1, 0], palette="Blues_d", errorbar=('ci', 90)
    )
    axes[1, 0].set_title("Panel C: Explosive Rate in High Leverage", fontweight='bold')
    axes[1, 0].set_ylabel("Explosive Probability")
    axes[1, 0].set_xlabel("DCI Quartile (Q4 = Tightest)")
else:
    axes[1, 0].text(0.5, 0.5, "No High Leverage Plays Found", ha='center', va='center')

# --- PANEL D: Ceiling Suppression (Deep vs Short) ---
if 'is_deep' in df.columns:
    res_d = df.groupby(['is_deep', 'dci_quartile'])['epa'].quantile(0.90).reset_index()
    res_d['type'] = np.where(res_d['is_deep'], "Deep Pass", "Short Pass")

    sns.lineplot(
        data=res_d, x='dci_quartile', y='epa', hue='type',
        ax=axes[1, 1], marker='s', linewidth=3
    )
    axes[1, 1].set_title("Panel D: 90th Percentile EPA (Ceiling Suppression)", fontweight='bold')
    axes[1, 1].set_ylabel("90th Percentile EPA")
    axes[1, 1].set_xlabel("DCI Quartile")
else:
    axes[1, 1].text(0.5, 0.5, "Deep Pass data not available", ha='center', va='center')

plt.tight_layout()

# Save to Kaggle working directory
out_fig = os.path.join(OUTPUT_DIR, "competition_winning_figures.png")
plt.savefig(out_fig, dpi=300)
print(f"[SUCCESS] Saved figure to: {out_fig}")
# plt.show()

# ==========================================
# 4. STATISTICAL MODEL UPGRADE
# ==========================================
print("\n" + "=" * 60)
print("STATISTICAL MODEL: Explosive Play Reduction")
print("=" * 60)

formula = "is_explosive ~ dci_z + dis_z + C(quarter)"
if 'ydstogo' in df.columns:
    formula += " + ydstogo"
if 'pre_snap_win_probability' in df.columns:
    formula += " + pre_snap_win_probability"
if 'team_coverage_type' in df.columns:
    formula += " + C(team_coverage_type) + dci_z:C(team_coverage_type)"

try:
    model = smf.logit(formula, data=df).fit(disp=0)
    print(model.summary())

    print("\n--- KEY INSIGHTS (Odds Ratios) ---")
    odds = np.exp(model.params)
    dci_effect = odds.get('dci_z', np.nan)
    print(f"Effect of 1 SD increase in DCI: {dci_effect:.3f}")

    if np.isfinite(dci_effect) and dci_effect < 1.0:
        print(" SUCCESS: Value < 1.0 confirms DCI reduces explosive risk (Negative Correlation).")
    else:
        print(" NOTE: Value >= 1.0 implies DCI correlates with risk (Paradox Scenario) or is not estimated.")
except Exception as e:
    print(f"[ERROR] Model failed: {e}")


# ==========================================
# METRIC VALIDATION BOXPLOTS (KAGGLE CELL)
# Assumes you already defined:
#   - METRICS_PATH (e.g., /kaggle/input/<slug>/metrics_playlevel_supervised.parquet)
#   - OUTPUT_DIR   (e.g., /kaggle/working)
#   - sns theme / plt rcParams (optional)
# ==========================================

print("[INFO] Loading metrics parquet...")
df = pd.read_parquet(METRICS_PATH)

# Filter for relevant pass outcomes (adjust if your dataset differs)
target_outcomes = ['C', 'I', 'S', 'IN']
df_clean = df[df['pass_result'].isin(target_outcomes)].copy()

# Map abbreviations to readable labels
label_map = {
    'C': 'Complete',
    'I': 'Incomplete',
    'S': 'Sack',
    'IN': 'Interception'
}
df_clean['Outcome'] = df_clean['pass_result'].map(label_map)

# Order outcomes to tell a story (offense good -> offense bad)
order = ['Complete', 'Incomplete', 'Sack', 'Interception']

# Basic sanity checks (fail early with a clear message)
required_cols = ['pass_result', 'dci_supervised', 'dis_final']
missing = [c for c in required_cols if c not in df_clean.columns]
if missing:
    raise KeyError(f"Missing required columns in METRICS_PATH parquet: {missing}")

# ==========================================
# PLOTTING
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)

# Panel 1: DCI vs Outcome
sns.boxplot(
    data=df_clean,
    x='Outcome',
    y='dci_supervised',
    order=order,
    palette="Blues_d",
    ax=axes[0],
    showfliers=False
)
axes[0].set_title('Defensive Coverage Index (DCI) by Outcome', fontweight='bold', pad=15)
axes[0].set_ylabel('DCI Score (Higher = Tighter Coverage)')
axes[0].set_xlabel('')

# Panel 2: DIS vs Outcome
sns.boxplot(
    data=df_clean,
    x='Outcome',
    y='dis_final',
    order=order,
    palette="Reds_d",
    ax=axes[1],
    showfliers=False
)
axes[1].set_title('Defensive Integrity Score (DIS) by Outcome', fontweight='bold', pad=15)
axes[1].set_ylabel('DIS Score (Higher = Better Integrity)')
axes[1].set_xlabel('')

plt.suptitle('Validation of Defensive Metrics against Play Outcomes', fontsize=20, y=1.02)
plt.tight_layout()

# Save to Kaggle working directory
output_file = os.path.join(OUTPUT_DIR, "metric_validation_boxplot.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"[SUCCESS] Plot saved to: {output_file}")

plt.show()


# ==========================================
# EPA CORRELATION REGRESSION PLOTS (KAGGLE CELL)
# Assumes you already defined:
#   - METRICS_PATH
#   - OUTPUT_DIR
# ==========================================

print("[INFO] Loading metrics parquet...")
df = pd.read_parquet(METRICS_PATH)

# Clean data: drop rows where key metrics are missing
required = ['dci_supervised', 'dis_final', 'epa']
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns in METRICS_PATH parquet: {missing}")

df_clean = df.dropna(subset=required).copy()

# Optional: trim extreme EPA outliers for clearer visualization
df_clean = df_clean[(df_clean['epa'] > -5) & (df_clean['epa'] < 5)].copy()

def plot_regression(ax, x_data, y_data, title, xlabel, scatter_kwargs=None, line_kwargs=None):
    """
    Scatter + OLS trend line + Pearson correlation summary.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes to draw on.
    x_data, y_data : pd.Series
        Data vectors for plot.
    title : str
        Panel title.
    xlabel : str
        X-axis label.
    scatter_kwargs : dict
        Passed to ax.scatter.
    line_kwargs : dict
        Passed to ax.plot.
    """
    scatter_kwargs = scatter_kwargs or {}
    line_kwargs = line_kwargs or {}

    # Scatter (low alpha due to many points)
    ax.scatter(x_data, y_data, alpha=0.15, s=15, edgecolors='none', label='Individual Plays', **scatter_kwargs)

    # Regression line
    slope, intercept = np.polyfit(x_data, y_data, 1)
    x_range = np.linspace(x_data.min(), x_data.max(), 100)
    y_pred = slope * x_range + intercept
    ax.plot(x_range, y_pred, linewidth=3, linestyle='--',
            label=f'Trend (Slope: {slope:.3f})', **line_kwargs)

    # Pearson correlation
    corr = x_data.corr(y_data)

    # Styling
    ax.set_title(f"{title}\nCorrelation (r): {corr:.3f}", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Offensive EPA (Expected Points Added)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)

fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

# Panel 1: DCI vs EPA
plot_regression(
    axes[0],
    df_clean['dci_supervised'],
    df_clean['epa'],
    title="Defensive Coverage (DCI) Impact on EPA",
    xlabel="DCI Score (Higher = Tighter Coverage)",
    scatter_kwargs={"c": "#4c72b0"},
    line_kwargs={"color": "darkblue"}
)

# Panel 2: DIS vs EPA
plot_regression(
    axes[1],
    df_clean['dis_final'],
    df_clean['epa'],
    title="Defensive Integrity (DIS) Impact on EPA",
    xlabel="DIS Score (Higher = Better Integrity)",
    scatter_kwargs={"c": "#c44e52"},
    line_kwargs={"color": "darkred"}
)

plt.suptitle('Statistical Validation: Does Better Defense Lower Offensive EPA?', fontsize=18, y=1.02)
plt.tight_layout()

output_file = os.path.join(OUTPUT_DIR, "epa_correlation_regplot.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"[SUCCESS] Plot saved to: {output_file}")

plt.show()


# ==========================================
# EXPERIMENTS: LOGIT (Explosives) + QUANTREG (Ceiling)  (KAGGLE CELL)
# Assumes you already defined:
#   - METRICS_PATH
#   - SUPP_PATH
#   - OUTPUT_DIR
# And you already imported:
#   - pandas as pd, numpy as np
#   - statsmodels.formula.api as smf
#   - seaborn as sns, matplotlib.pyplot as plt
# ==========================================

print("[INFO] Loading data...")
df_metrics = pd.read_parquet(METRICS_PATH)
df_supp = pd.read_csv(SUPP_PATH, low_memory=False)

# ==========================================
# COLUMN MAPPING & CLEANING
# ==========================================
rename_map = {
    'gameId': 'game_id',
    'playId': 'play_id',
    'defensiveTeam': 'defensive_team',
    'possessionTeam': 'possession_team',
    'down': 'down',
    'quarter': 'qtr',
    'preSnapHomeScore': 'home_score',
    'preSnapVisitorScore': 'away_score'
}
df_supp.rename(columns=rename_map, inplace=True)

# Find "Yards To Go" dynamically
possible_yd_cols = ['yardsToGo', 'YardsToGo', 'yards_to_go', 'ydstogo']
found_yd_col = next((c for c in possible_yd_cols if c in df_supp.columns), None)

if found_yd_col:
    df_supp.rename(columns={found_yd_col: 'ydstogo'}, inplace=True)
    print(f"[INFO] Found yards column: '{found_yd_col}' -> renamed to 'ydstogo'")
else:
    print("[WARN] Could not find yards-to-go column; creating default ydstogo=10.")
    df_supp['ydstogo'] = 10

# Build control columns only if present
control_cols = ['game_id', 'play_id', 'defensive_team', 'possession_team', 'ydstogo']
if 'down' in df_supp.columns:
    control_cols.append('down')

if 'home_score' in df_supp.columns and 'away_score' in df_supp.columns:
    df_supp['score_diff'] = df_supp['home_score'] - df_supp['away_score']
    control_cols.append('score_diff')

# Merge
df = df_metrics.merge(df_supp[control_cols], on=['game_id', 'play_id'], how='inner')

# Filter passes if pass_result exists
valid_pass_codes = ['C', 'I', 'S', 'IN', 'COMPLETE', 'INCOMPLETE', 'INTERCEPTION', 'SACK']
if 'pass_result' in df.columns:
    df = df[df['pass_result'].isin(valid_pass_codes)].copy()

print(f"[INFO] Analysis set: {len(df)} plays.")

# ==========================================
# FEATURE ENGINEERING
# ==========================================
required_cols = ['epa', 'dci_supervised', 'dis_final', 'ydstogo']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns after merge: {missing}")

df = df.dropna(subset=required_cols).copy()

df['is_explosive'] = (df['epa'] >= 2.0).astype(int)

df['dci_z'] = (df['dci_supervised'] - df['dci_supervised'].mean()) / df['dci_supervised'].std()
df['dis_z'] = (df['dis_final'] - df['dis_final'].mean()) / df['dis_final'].std()

# ==========================================
# EXPERIMENT 1: LOGISTIC REGRESSION
# ==========================================
print("\n" + "="*60)
print("EXPERIMENT 1: LOGISTIC REGRESSION (Explosives)")
print("="*60)

formula_logit = "is_explosive ~ dci_z + dis_z + ydstogo"
if 'down' in df.columns:
    formula_logit += " + C(down)"
if 'score_diff' in df.columns:
    formula_logit += " + score_diff"

try:
    model_logit = smf.logit(formula_logit, data=df).fit(disp=0)
    print(model_logit.summary())

    # Odds Ratios for key variables
    params = model_logit.params
    conf = model_logit.conf_int()
    conf['Odds_Ratio'] = np.exp(params)
    conf.columns = ['2.5%', '97.5%', 'Odds_Ratio']
    print("\n--- EFFECT SIZES (Odds Ratios) ---")
    print(conf.loc[['dci_z', 'dis_z']])
    print("(Odds_Ratio < 1.0 means the metric is associated with LOWER explosive risk.)")
except Exception as e:
    print(f"[ERROR] Logistic Regression failed: {e}")

# ==========================================
# EXPERIMENT 2: QUANTILE REGRESSION (90th percentile)
# ==========================================
print("\n" + "="*60)
print("EXPERIMENT 2: QUANTILE REGRESSION (Offensive Ceiling)")
print("="*60)

formula_quant = "epa ~ dci_z + dis_z + ydstogo"
if 'down' in df.columns:
    formula_quant += " + C(down)"
if 'score_diff' in df.columns:
    formula_quant += " + score_diff"

try:
    mod_quant = smf.quantreg(formula_quant, df)
    res_quant = mod_quant.fit(q=0.9)
    print(res_quant.summary())
    print("\n[INTERPRETATION] Focus on 'dci_z' coefficient.")
    print("If negative: higher DCI is associated with lowering the offense's 90th percentile EPA (ceiling suppression).")
except Exception as e:
    print(f"[ERROR] Quantile Regression failed: {e}")

# ==========================================
# VISUALIZATION
# ==========================================
print("\n[INFO] Generating plot...")
try:
    df['dci_quartile'] = pd.qcut(df['dci_supervised'], 4, labels=["Q1 (Loose)", "Q2", "Q3", "Q4 (Tight)"])

    plt.figure(figsize=(10, 6))
    sns.barplot(x='dci_quartile', y='is_explosive', data=df, palette="Blues", errorbar=('ci', 95))
    plt.title("Probability of Explosive Play (EPA >= 2.0) by Coverage Quality")
    plt.ylabel("Explosive Play Probability")
    plt.xlabel("Defensive Coverage Index (DCI)")

    output_file = os.path.join(OUTPUT_DIR, "explosive_play_reduction.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Plot saved to {output_file}")
    plt.show()
except Exception as e:
    print(f"[WARN] Could not plot: {e}")


"""
NFL Big Data Bowl - Enhanced Animation with DCI/DIS Gauge (Kaggle-Safe)
- Auto-finds INPUT_CSV under /kaggle/input/**
- Writes outputs to /kaggle/working/ (only writable location on Kaggle)
- Avoids exit() (won't kill the notebook)
- Saves GIF reliably; saves MP4 only if ffmpeg exists
- Uses blit=False for stability (you add/remove many artists each frame)
"""

import matplotlib
matplotlib.use("Agg")  # file-based rendering (Kaggle-safe)

import os
import glob
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Circle, Rectangle

warnings.filterwarnings("ignore")

# ============================================================================
# TEAM COLOR SCHEMES
# ============================================================================
TEAM_COLORS = {
    "vikings": {
        "name": "Minnesota Vikings",
        "primary": "#4F2683",
        "secondary": "#FFC62F",
        "background": "#4F2683",
        "field": "#2d5016",
        "offense_color": "#FFC62F",
        "defense_color": "#FFFFFF",
        "field_stripe": "#3d3166",
    },
    "seahawks": {
        "name": "Seattle Seahawks",
        "primary": "#002244",
        "secondary": "#69BE28",
        "background": "#002244",
        "field": "#2d5016",
        "offense_color": "#69BE28",
        "defense_color": "#A5ACAF",
        "field_stripe": "#001933",
    },
    "default": {
        "name": "NFL Default",
        "primary": "#013369",
        "secondary": "#D50A0A",
        "background": "#0a0e1a",
        "field": "#2d5016",
        "offense_color": "#1E90FF",
        "defense_color": "#DC143C",
        "field_stripe": "#1d3010",
    },
}

# ============================================================================
# CONFIGURATION (Kaggle-safe)
# ============================================================================
INPUT_CSV_NAME = "vikings_play_79_data.csv"  # filename only; auto-found on Kaggle
OUTPUT_DIR = "/kaggle/working/team_animations"
TEAM_THEME = "vikings"

GAME_ID = None  # auto-select if present and not provided
PLAY_ID = None  # auto-select if present and not provided

FPS = 10
VIDEO_FORMAT = "gif"  # "mp4", "gif", or "both"

SHOW_PLAYER_NAMES = True
SHOW_PLAYER_NUMBERS = False
SHOW_POSITIONS = False  # keep False if you only want names
SHOW_TEAM_LOGO_AREA = True
SHOW_NODE_STRESS = True
SHOW_DCI_DIS_GAUGE = True

METRIC_MODE = "geometric"  # "precomputed" or "geometric"
DCI_COLUMN = "dci_score"
DIS_COLUMN = "dis_score"

# ============================================================================
# PLAYER NAME LOOKUPS
# ============================================================================
VIKINGS_49ERS_PLAYER_NAMES = {
    38632: "Jordan Addison",
    47791: "Kirk Cousins",
    47852: "Justin Jefferson",
    47885: "T.J. Hockenson",
    52584: "Alexander Mattison",
    55887: "K.J. Osborn",
    38868: "Nick Bosa",
    46139: "Fred Warner",
    46157: "Dre Greenlaw",
    46757: "Javon Hargrave",
    47931: "Charvarius Ward",
    53601: "Talanoa Hufanga",
    53609: "Deommodore Lenoir",
}

SEAHAWKS_GIANTS_PLAYER_NAMES = {
    38577: "Bobby Wagner",
    39987: "Drew Lock",
    42412: "Jake Bobo",
    42543: "Quandre Diggs",
    42547: "Will Dissly",
    43329: "Tyler Lockett",
    43333: "Uchenna Nwosu",
    44818: "Boye Mafe",
    44830: "Riq Woolen",
    45186: "Kenneth Walker III",
    46117: "Jarran Reed",
    46189: "Darren Waller",
    47789: "Geno Smith",
    47793: "Bobby Okereke",
    47803: "Noah Fant",
    47825: "Daniel Jones",
    47842: "Darius Slayton",
    47847: "DK Metcalf",
    47872: "Jordyn Brooks",
    47891: "Jamal Adams",
    47941: "Devon Witherspoon",
    47954: "Jaxon Smith-Njigba",
    48266: "Sterling Shepard",
    52416: "Leonard Williams",
    52435: "Dre'Mont Jones",
    52444: "Xavier McKinney",
    52541: "Colby Parkinson",
    52552: "Saquon Barkley",
    52615: "Dareke Young",
    53604: "Julian Love",
    53625: "Zach Charbonnet",
    54014: "Tre Brown",
    54470: "Johnathan Abram",
    54506: "DeeJay Dallas",
    54508: "Wan'Dale Robinson",
    54546: "Deonte Banks",
    54577: "Pharaoh Brown",
    54579: "Isaiah Simmons",
    54611: "Micah McFadden",
    54618: "Adoree' Jackson",
    55869: "Nick McCloud",
    55884: "Jalin Hyatt",
    55888: "Cor'Dale Flott",
    55902: "Dane Belton",
    55917: "Matt Breida",
    55938: "Parris Campbell",
    56063: "Bobby McCain",
    56471: "Isaiah Hodgins",
}

GAME_MATCHUP = "vikings_49ers"  # "vikings_49ers" or "seahawks_giants"
if GAME_MATCHUP == "vikings_49ers":
    PLAYER_NAMES = VIKINGS_49ERS_PLAYER_NAMES
elif GAME_MATCHUP == "seahawks_giants":
    PLAYER_NAMES = SEAHAWKS_GIANTS_PLAYER_NAMES
else:
    PLAYER_NAMES = {}

# ============================================================================
# KAGGLE: auto-find input CSV + ensure output dir exists
# ============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

candidates = glob.glob(f"/kaggle/input/**/{INPUT_CSV_NAME}", recursive=True) + glob.glob(INPUT_CSV_NAME)
if not candidates:
    raise FileNotFoundError(
        f"Could not find '{INPUT_CSV_NAME}'.\n"
        f"- Add the dataset containing this file to the notebook, or\n"
        f"- Upload the CSV, or\n"
        f"- Rename INPUT_CSV_NAME to match the file.\n"
        f"Searched: /kaggle/input/**/{INPUT_CSV_NAME} and ./{INPUT_CSV_NAME}"
    )
INPUT_CSV = candidates[0]

colors = TEAM_COLORS.get(TEAM_THEME, TEAM_COLORS["default"])

print("\n" + "=" * 60)
print(f"NFL ANIMATION - {colors['name']} Theme")
print("=" * 60 + "\n")
print(f"Loading data from: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
print(f"Total rows: {len(df):,}")
print(f"Available columns: {list(df.columns)}")

# ============================================================================
# Column normalization / validation
# ============================================================================
# Allow common variants
if "playerSide" in df.columns and "player_side" not in df.columns:
    df = df.rename(columns={"playerSide": "player_side"})
if "nflId" in df.columns and "nfl_id" not in df.columns:
    df = df.rename(columns={"nflId": "nfl_id"})
if "frameId" in df.columns and "frame_id" not in df.columns:
    df = df.rename(columns={"frameId": "frame_id"})

required = ["frame_id", "nfl_id", "x", "y"]
missing_required = [c for c in required if c not in df.columns]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}. Present columns: {list(df.columns)}")

if "player_side" not in df.columns:
    raise ValueError(
        "Missing 'player_side' column. This animation requires Offense/Defense separation.\n"
        "If your file uses a different column name, rename it to 'player_side' (values: 'Offense'/'Defense').\n"
        f"Present columns: {list(df.columns)}"
    )

# ============================================================================
# Play selection logic
# ============================================================================
if "game_id" not in df.columns or "play_id" not in df.columns:
    print("New animation data format detected (no game_id/play_id columns). Using entire file as one play.")
    play_data = df.copy()
    GAME_ID = "animation_data"
    PLAY_ID = "101"
else:
    play_summary = (
        df.groupby(["game_id", "play_id"])
        .agg(frame_id=("frame_id", "nunique"), nfl_id=("nfl_id", "nunique"))
        .reset_index()
        .rename(columns={"frame_id": "frames", "nfl_id": "players"})
        .sort_values(["players", "frames"], ascending=[False, False])
    )

    if GAME_ID is None or PLAY_ID is None:
        print("\n" + "=" * 60)
        print("AVAILABLE PLAYS IN YOUR CSV")
        print("=" * 60 + "\n")
        print("Top 20 plays by player count:\n")
        print(play_summary.head(20).to_string(index=False))

        # Kaggle-friendly behavior: auto-select best candidate (most players, most frames)
        GAME_ID = int(play_summary.iloc[0]["game_id"])
        PLAY_ID = int(play_summary.iloc[0]["play_id"])
        print(f"\nAuto-selected GAME_ID={GAME_ID}, PLAY_ID={PLAY_ID} (top by players/frames).")

    play_data = df[(df["game_id"] == GAME_ID) & (df["play_id"] == PLAY_ID)].copy()
    if len(play_data) == 0:
        raise ValueError(f"No data found for Game {GAME_ID}, Play {PLAY_ID}.")

# ============================================================================
# Metric availability checks
# ============================================================================
has_dci = DCI_COLUMN in df.columns
has_dis = DIS_COLUMN in df.columns

if METRIC_MODE == "precomputed" and not (has_dci and has_dis):
    print("\nWARNING: METRIC_MODE='precomputed' but metric columns not found.")
    print(f"Looking for: {DCI_COLUMN}, {DIS_COLUMN}")
    print("Switching to METRIC_MODE='geometric'.")
    METRIC_MODE = "geometric"

print(f"\nGame ID: {GAME_ID}")
print(f"Play ID: {PLAY_ID}")
print(f"Players: {play_data['nfl_id'].nunique()}")
print(f"Frames: {play_data['frame_id'].nunique()}")
print(f"Theme: {colors['name']}")
print(f"Metric Mode: {METRIC_MODE.upper()}")

# Separate actual positions from projections (if 's' exists)
if "s" in play_data.columns:
    play_data["is_projection"] = play_data["s"].isna()
else:
    play_data["is_projection"] = False

# Precompute frame ids
frame_ids = np.sort(play_data["frame_id"].unique())
num_frames = len(frame_ids)

# ============================================================================
# Metric functions
# ============================================================================
def calculate_geometric_dci(defense_positions: np.ndarray) -> float:
    """Geometric proxy for DCI (coverage tightness): tighter defender spacing -> higher DCI."""
    if defense_positions is None or len(defense_positions) < 2:
        return 0.5

    distances = []
    for i, pos in enumerate(defense_positions):
        other_pos = np.delete(defense_positions, i, axis=0)
        if len(other_pos) > 0:
            dists = np.linalg.norm(other_pos - pos, axis=1)
            distances.append(np.min(dists))

    if not distances:
        return 0.5

    avg_spacing = float(np.mean(distances))
    # Typical spacing: 3-15 yards (smaller spacing => higher score)
    return float(np.clip(1.0 - (avg_spacing - 3.0) / 12.0, 0.0, 1.0))


def calculate_geometric_dis(defense_positions: np.ndarray, prev_defense_positions=None) -> float:
    """Geometric proxy for DIS (structural integrity): consistent spacing -> higher DIS."""
    if defense_positions is None or len(defense_positions) < 3:
        return 0.5

    distances = []
    for i in range(len(defense_positions)):
        for j in range(i + 1, len(defense_positions)):
            distances.append(float(np.linalg.norm(defense_positions[i] - defense_positions[j])))

    if not distances:
        return 0.5

    std_spacing = float(np.std(distances))
    mean_spacing = float(np.mean(distances))
    if mean_spacing <= 0:
        return 0.5

    cv = std_spacing / mean_spacing
    return float(np.clip(1.0 - cv, 0.0, 1.0))

# ============================================================================
# Drawing functions
# ============================================================================
def draw_field(ax, colors_):
    # Yard lines
    for yard in range(10, 111, 5):
        linewidth = 2 if yard % 10 == 0 else 1
        c = "white" if yard % 10 == 0 else colors_["secondary"]
        alpha = 0.7 if yard % 10 == 0 else 0.4
        ax.plot([yard, yard], [0, 53.3], color=c, linewidth=linewidth, alpha=alpha)

    # Goal lines
    ax.plot([10, 10], [0, 53.3], color=colors_["secondary"], linewidth=4, alpha=0.9)
    ax.plot([110, 110], [0, 53.3], color=colors_["secondary"], linewidth=4, alpha=0.9)

    # Sidelines
    ax.plot([0, 120], [0, 0], color="white", linewidth=2, alpha=0.8)
    ax.plot([0, 120], [53.3, 53.3], color="white", linewidth=2, alpha=0.8)
    ax.plot([0, 0], [0, 53.3], color="white", linewidth=2, alpha=0.8)
    ax.plot([120, 120], [0, 53.3], color="white", linewidth=2, alpha=0.8)

    # Hash marks
    for yard in range(10, 111):
        ax.plot([yard, yard], [23.36, 23.36], color="white", marker=".", markersize=2, alpha=0.6)
        ax.plot([yard, yard], [29.94, 29.94], color="white", marker=".", markersize=2, alpha=0.6)

    # Stripes
    for yard in range(0, 120, 10):
        rect = patches.Rectangle(
            (yard, 0), 5, 53.3, linewidth=0, edgecolor="none", facecolor=colors_["field_stripe"], alpha=0.15
        )
        ax.add_patch(rect)

    # Branding corners
    if SHOW_TEAM_LOGO_AREA:
        ax.add_patch(patches.Rectangle((0, 48), 8, 5.3, linewidth=0, facecolor=colors_["primary"], alpha=0.3))
        ax.add_patch(patches.Rectangle((112, 48), 8, 5.3, linewidth=0, facecolor=colors_["secondary"], alpha=0.3))


def draw_dci_dis_gauge(ax, dci_value, dis_value, colors_):
    artists = []

    gauge_x, gauge_y = 102, 43
    gauge_width, gauge_height = 16, 9

    bg = Rectangle(
        (gauge_x, gauge_y),
        gauge_width,
        gauge_height,
        facecolor=colors_["primary"],
        edgecolor="white",
        linewidth=2,
        alpha=0.9,
        zorder=100,
    )
    ax.add_patch(bg)
    artists.append(bg)

    title = ax.text(
        gauge_x + gauge_width / 2,
        gauge_y + gauge_height - 1,
        "DEFENSIVE METRICS",
        ha="center",
        va="top",
        fontsize=10,
        color="white",
        fontweight="bold",
        zorder=101,
    )
    artists.append(title)

    bar_height = 1.2
    bar_width = gauge_width - 3

    # DCI
    bar_y_dci = gauge_y + gauge_height - 3.5
    dci_bg = Rectangle((gauge_x + 1.5, bar_y_dci), bar_width, bar_height, facecolor="#333333",
                       edgecolor="white", linewidth=1, alpha=0.5, zorder=101)
    ax.add_patch(dci_bg); artists.append(dci_bg)

    dci_color = plt.cm.RdYlGn(dci_value)
    dci_fill = Rectangle((gauge_x + 1.5, bar_y_dci), bar_width * dci_value, bar_height,
                         facecolor=dci_color, edgecolor="none", alpha=0.8, zorder=102)
    ax.add_patch(dci_fill); artists.append(dci_fill)

    dci_label = ax.text(gauge_x + 1.5, bar_y_dci - 0.4, "DCI (Coverage)",
                        ha="left", va="top", fontsize=8, color="white",
                        fontweight="bold", zorder=103)
    artists.append(dci_label)

    dci_text = ax.text(gauge_x + gauge_width - 1.5, bar_y_dci + bar_height / 2, f"{dci_value:.3f}",
                       ha="right", va="center", fontsize=9, color="white", fontweight="bold", zorder=103)
    artists.append(dci_text)

    # DIS
    bar_y_dis = bar_y_dci - 2.5
    dis_bg = Rectangle((gauge_x + 1.5, bar_y_dis), bar_width, bar_height, facecolor="#333333",
                       edgecolor="white", linewidth=1, alpha=0.5, zorder=101)
    ax.add_patch(dis_bg); artists.append(dis_bg)

    dis_color = plt.cm.RdYlGn(dis_value)
    dis_fill = Rectangle((gauge_x + 1.5, bar_y_dis), bar_width * dis_value, bar_height,
                         facecolor=dis_color, edgecolor="none", alpha=0.8, zorder=102)
    ax.add_patch(dis_fill); artists.append(dis_fill)

    dis_label = ax.text(gauge_x + 1.5, bar_y_dis - 0.4, "DIS (Integrity)",
                        ha="left", va="top", fontsize=8, color="white",
                        fontweight="bold", zorder=103)
    artists.append(dis_label)

    dis_text = ax.text(gauge_x + gauge_width - 1.5, bar_y_dis + bar_height / 2, f"{dis_value:.3f}",
                       ha="right", va="center", fontsize=9, color="white", fontweight="bold", zorder=103)
    artists.append(dis_text)

    return artists

# ============================================================================
# Figure / axes
# ============================================================================
fig = plt.figure(figsize=(16, 10), facecolor=colors["background"])
ax = fig.add_subplot(111, facecolor=colors["field"])
ax.set_xlim(0, 120)
ax.set_ylim(0, 53.3)
ax.set_aspect("equal")
draw_field(ax, colors)

offense_scatter = ax.scatter([], [], c=colors["offense_color"], s=350,
                             edgecolors="white", linewidths=2.5, zorder=5, label="Offense", alpha=0.95)
defense_scatter = ax.scatter([], [], c=colors["defense_color"], s=350,
                             edgecolors=colors["primary"], linewidths=2.5, zorder=5, label="Defense", alpha=0.95)

player_labels = []
stress_circles = []
gauge_artists = []

stats_text = ax.text(
    0.02, 0.98, "",
    transform=ax.transAxes,
    fontsize=14,
    verticalalignment="top",
    color="white",
    fontweight="bold",
    bbox=dict(boxstyle="round", facecolor=colors["primary"], alpha=0.85,
              edgecolor=colors["secondary"], linewidth=2),
)

title = ax.text(
    0.5, 0.98, "NFL Big Data Bowl - Animation",
    transform=ax.transAxes,
    fontsize=18,
    verticalalignment="top",
    horizontalalignment="center",
    color="white",
    fontweight="bold",
    bbox=dict(boxstyle="round", facecolor=colors["primary"], alpha=0.85,
              edgecolor=colors["secondary"], linewidth=3),
)

ax.set_xlabel("Yards", color="white", fontsize=12, fontweight="bold")
ax.set_ylabel("Field Width (yards)", color="white", fontsize=12, fontweight="bold")
ax.tick_params(colors="white")

prev_defense_positions = None

def init():
    offense_scatter.set_offsets(np.empty((0, 2)))
    defense_scatter.set_offsets(np.empty((0, 2)))
    stats_text.set_text("")
    return [offense_scatter, defense_scatter, stats_text]

def update(frame_num: int):
    global prev_defense_positions

    frame_id = frame_ids[frame_num]
    current_frame = play_data[play_data["frame_id"] == frame_id]

    actual_positions = current_frame[~current_frame["is_projection"]]

    # Clear dynamic artists
    for label in player_labels:
        label.remove()
    player_labels.clear()

    for circle in stress_circles:
        circle.remove()
    stress_circles.clear()

    for artist in gauge_artists:
        artist.remove()
    gauge_artists.clear()

    offense_data = actual_positions[actual_positions["player_side"] == "Offense"]
    defense_data = actual_positions[actual_positions["player_side"] == "Defense"]

    offense_scatter.set_offsets(offense_data[["x", "y"]].values if len(offense_data) else np.empty((0, 2)))
    defense_scatter.set_offsets(defense_data[["x", "y"]].values if len(defense_data) else np.empty((0, 2)))

    # DCI/DIS
    if METRIC_MODE == "precomputed" and has_dci and has_dis:
        dci_value = float(current_frame[DCI_COLUMN].mean())
        dis_value = float(current_frame[DIS_COLUMN].mean())
    else:
        if len(defense_data) > 0:
            defense_positions = defense_data[["x", "y"]].values
            dci_value = calculate_geometric_dci(defense_positions)
            dis_value = calculate_geometric_dis(defense_positions, prev_defense_positions)
            prev_defense_positions = defense_positions
        else:
            dci_value, dis_value = 0.5, 0.5

    if SHOW_DCI_DIS_GAUGE:
        gauge_artists.extend(draw_dci_dis_gauge(ax, dci_value, dis_value, colors))

    # Node stress halos
    if SHOW_NODE_STRESS and "node_stress" in actual_positions.columns:
        for _, player in actual_positions.iterrows():
            if pd.notna(player.get("node_stress", np.nan)):
                stress = float(player["node_stress"])
                if stress < 0.20:
                    c = "#00FF80"
                elif stress < 0.35:
                    c = "#FFD700"
                else:
                    c = "#FF4500"

                radius = 2.5 + (stress * 4.0)
                alpha = 0.3 + (stress * 0.4)

                circle = Circle((float(player["x"]), float(player["y"])), radius,
                                color=c, alpha=alpha, zorder=1, linewidth=3, edgecolor="white")
                ax.add_patch(circle)
                stress_circles.append(circle)

    # Labels
    if SHOW_POSITIONS or SHOW_PLAYER_NAMES or SHOW_PLAYER_NUMBERS:
        for _, player in actual_positions.iterrows():
            nfl_id = int(player["nfl_id"]) if pd.notna(player["nfl_id"]) else None

            if SHOW_POSITIONS and "player_position" in actual_positions.columns and pd.notna(player.get("player_position", np.nan)):
                label_text = str(player["player_position"])
                fontsize = 9
            elif SHOW_PLAYER_NAMES and nfl_id in PLAYER_NAMES:
                label_text = PLAYER_NAMES[nfl_id]
                fontsize = 9
            elif SHOW_PLAYER_NUMBERS and "jersey_number" in actual_positions.columns and pd.notna(player.get("jersey_number", np.nan)):
                label_text = f"#{int(player['jersey_number'])}"
                fontsize = 9
            else:
                label_text = str(nfl_id) if nfl_id is not None else "NA"
                fontsize = 9

            side = player.get("player_side", "")
            if side == "Offense":
                bg_color = colors["offense_color"]
                txt_color = colors["primary"]
            else:
                bg_color = colors["defense_color"]
                txt_color = colors["primary"]

            label = ax.text(
                float(player["x"]), float(player["y"]) - 1.5, label_text,
                ha="center", va="top", fontsize=fontsize, color=txt_color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, alpha=0.85, edgecolor="white", linewidth=1),
            )
            player_labels.append(label)

    # Stats box
    offense_count = len(offense_data)
    defense_count = len(defense_data)

    stats = f"Frame: {frame_num + 1}/{num_frames}\n"
    stats += f"OFF: {offense_count} | DEF: {defense_count}\n"

    if "node_stress" in actual_positions.columns:
        avg_stress = actual_positions["node_stress"].mean()
        if pd.notna(avg_stress):
            stats += f"Avg Stress: {float(avg_stress):.3f}\n"

    stats += f"\nDCI: {dci_value:.3f}\nDIS: {dis_value:.3f}"

    stats_text.set_text(stats)

    elements = [offense_scatter, defense_scatter, stats_text]
    elements.extend(player_labels)
    elements.extend(stress_circles)
    elements.extend(gauge_artists)
    return elements

print(f"\nGenerating animation with {num_frames} frames...")
anim = FuncAnimation(fig, update, init_func=init, frames=num_frames, interval=100, blit=False)

output_base = f"{TEAM_THEME}_game{GAME_ID}_play{PLAY_ID}_metrics"
ffmpeg_ok = shutil.which("ffmpeg") is not None

if VIDEO_FORMAT in ["mp4", "both"] and not ffmpeg_ok:
    print("ffmpeg not found in this runtime; cannot save MP4. Switching to GIF only.")
    VIDEO_FORMAT = "gif"

if VIDEO_FORMAT in ["mp4", "both"]:
    output_path_mp4 = os.path.join(OUTPUT_DIR, f"{output_base}.mp4")
    print(f"Saving MP4: {output_path_mp4}")
    writer_mp4 = FFMpegWriter(fps=FPS, bitrate=2000)
    anim.save(output_path_mp4, writer=writer_mp4)
    print("MP4 saved.")

if VIDEO_FORMAT in ["gif", "both"]:
    output_path_gif = os.path.join(OUTPUT_DIR, f"{output_base}.gif")
    print(f"Saving GIF: {output_path_gif}")
    writer_gif = PillowWriter(fps=FPS)
    anim.save(output_path_gif, writer=writer_gif)
    print("GIF saved.")

plt.close(fig)

print("\n" + "=" * 60)
print("ANIMATION COMPLETE")
print("=" * 60)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Base name: {output_base}")
print(f"Metric Mode: {METRIC_MODE}")
print("DCI: Defensive Coverage Index (0=loose, 1=tight)")
print("DIS: Defensive Integrity Score (0=chaotic, 1=disciplined)")



"""
NFL Individual Player Analysis
Generate detailed heatmaps and tracking for individual players
"""

import matplotlib.patches as patches

# ============================================================================
# TEAM COLORS
# ============================================================================
TEAM_COLORS = {
    'seahawks': {
        'primary': '#002244',
        'secondary': '#69BE28',
        'offense': '#69BE28',
        'defense': '#A5ACAF'
    }
}

colors = TEAM_COLORS[TEAM_THEME]

print(f"\n{'='*60}")
print("NFL INDIVIDUAL PLAYER ANALYSIS")
print(f"{'='*60}\n")

# Load and prepare data
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"INPUT_CSV not found.\nExpected: {INPUT_CSV}\n"
        f"Tip: check /kaggle/input/ for your dataset folder name."
    )

df = pd.read_csv(INPUT_CSV)
df_clean = df[~df['s'].isna() & ~df['x'].isna() & ~df['y'].isna()].copy()

print(f"Players available: {df_clean['nfl_id'].nunique() if 'nfl_id' in df_clean.columns else 'n/a'}")
print(f"Frames: {df_clean['frame_id'].nunique() if 'frame_id' in df_clean.columns else 'n/a'}")

# Player lookup
PLAYER_NAMES = {
    43290: "QB Wilson",
    44930: "RB Lynch",
    53541: "WR Metcalf",
    53959: "TE Olsen",
    46137: "CB Sherman",
    52546: "LB Wagner",
    53487: "SS Thomas",
    54486: "CB Griffin",
    54527: "FS Adams"
}

def draw_field_background(ax):
    """Draw NFL field background"""
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_aspect('equal')
    ax.set_facecolor('#2d5016')

    # Yard lines
    for yard in range(10, 111, 10):
        ax.axvline(yard, color='white', alpha=0.4, linewidth=1)
        ax.text(yard, 2, str(yard), ha='center', color='white', fontsize=8)

    # Goal lines
    ax.axvline(10, color='white', linewidth=3, alpha=0.8)
    ax.axvline(110, color='white', linewidth=3, alpha=0.8)

    # Sidelines
    for y in [0, 53.3]:
        ax.axhline(y, color='white', linewidth=2)

    # Hash marks
    for yard in range(10, 111):
        ax.plot([yard, yard], [23.36, 23.36], color='white', marker='|', markersize=4, alpha=0.6)
        ax.plot([yard, yard], [29.94, 29.94], color='white', marker='|', markersize=4, alpha=0.6)

def generate_individual_player_analysis():
    """Generate analysis for each individual player"""

    unique_players = df_clean['nfl_id'].unique()

    for player_id in unique_players:
        player_data = df_clean[df_clean['nfl_id'] == player_id].copy()

        if len(player_data) < 3:  # Skip players with too little data
            continue

        player_name = PLAYER_NAMES.get(player_id, f"Player_{player_id}")
        player_side = player_data['player_side'].iloc[0] if 'player_side' in player_data.columns else "Unknown"

        print(f"  Analyzing {player_name} ({player_side})...")

        # Create 2x2 subplot for this player
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{player_name} - Individual Analysis', fontsize=16, fontweight='bold')

        # 1. Movement trail with stress coloring
        draw_field_background(ax1)

        if 'node_stress' in player_data.columns:
            scatter = ax1.scatter(player_data['x'], player_data['y'],
                                  c=player_data['node_stress'],
                                  s=100, cmap='Reds', alpha=0.7,
                                  vmin=0, vmax=1, zorder=5)
            plt.colorbar(scatter, ax=ax1, shrink=0.6)

            # Connect positions with lines
            ax1.plot(player_data['x'], player_data['y'],
                     color='black', alpha=0.5, linewidth=2, zorder=3)
        else:
            ax1.scatter(player_data['x'], player_data['y'], s=80, alpha=0.7, zorder=5)
            ax1.plot(player_data['x'], player_data['y'], color='black', alpha=0.5, linewidth=2, zorder=3)

        # Add frame numbers
        if 'frame_id' in player_data.columns:
            for _, row in player_data.sample(min(8, len(player_data)), random_state=1).iterrows():
                ax1.annotate(f"{int(row['frame_id'])}",
                             (row['x'], row['y']),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8, alpha=0.7)

        ax1.set_title('Movement Trail (Colored by Stress)')
        ax1.set_xlabel('Field Position (yards)')
        ax1.set_ylabel('Field Width (yards)')

        # 2. Position heatmap
        draw_field_background(ax2)

        try:
            ax2.hexbin(player_data['x'], player_data['y'],
                       gridsize=15, cmap='Blues', alpha=0.7)
            ax2.set_title('Position Density')
        except:
            ax2.scatter(player_data['x'], player_data['y'],
                        alpha=0.6, s=50, color='blue')
            ax2.set_title('Position History')

        # 3. Metrics over time
        if 'frame_id' in player_data.columns:
            if 'node_stress' in player_data.columns:
                ax3.plot(player_data['frame_id'], player_data['node_stress'],
                         'r-', linewidth=2, label='Node Stress')
                ax3.set_ylabel('Node Stress', color='red')
                ax3.tick_params(axis='y', labelcolor='red')

                ax3_twin = ax3.twinx()
                if 's' in player_data.columns:
                    speed_mag = np.sqrt(player_data['s']**2)
                    ax3_twin.plot(player_data['frame_id'], speed_mag,
                                  'b--', linewidth=2, label='Speed')
                    ax3_twin.set_ylabel('Speed (yards/sec)', color='blue')
                    ax3_twin.tick_params(axis='y', labelcolor='blue')

                ax3.set_xlabel('Frame ID')
                ax3.set_title('Metrics Over Time')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.plot(player_data['frame_id'], np.sqrt(player_data['s']**2),
                         'b-', linewidth=2, label='Speed')
                ax3.set_xlabel('Frame ID')
                ax3.set_ylabel('Speed (yards/sec)')
                ax3.set_title('Speed Over Time')
                ax3.grid(True, alpha=0.3)
        else:
            ax3.axis('off')

        # 4. Statistics summary
        ax4.axis('off')

        stats_text = f"""
PLAYER STATISTICS

Name: {player_name}
Side: {player_side}
Total Frames: {len(player_data)}

POSITION:
â€¢ Avg X: {player_data['x'].mean():.1f} yards
â€¢ Avg Y: {player_data['y'].mean():.1f} yards
â€¢ X Range: {player_data['x'].max() - player_data['x'].min():.1f} yards
â€¢ Y Range: {player_data['y'].max() - player_data['y'].min():.1f} yards

MOVEMENT:
â€¢ Avg Speed: {np.sqrt(player_data['s']**2).mean():.2f} yards/sec
â€¢ Max Speed: {np.sqrt(player_data['s']**2).max():.2f} yards/sec"""

        if 'node_stress' in player_data.columns:
            stats_text += f"""

STRESS:
â€¢ Avg Stress: {player_data['node_stress'].mean():.3f}
â€¢ Max Stress: {player_data['node_stress'].max():.3f}
â€¢ Min Stress: {player_data['node_stress'].min():.3f}"""

        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        out_path = f"{OUTPUT_DIR}/{player_name.replace(' ', '_')}_analysis.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

def generate_player_comparison():
    """Generate comparative analysis between offense and defense"""

    print("  Generating player comparison...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Offense vs Defense Comparison', fontsize=16, fontweight='bold')

    offense_data = df_clean[df_clean['player_side'] == 'Offense'] if 'player_side' in df_clean.columns else df_clean.iloc[0:0]
    defense_data = df_clean[df_clean['player_side'] == 'Defense'] if 'player_side' in df_clean.columns else df_clean.iloc[0:0]

    # 1. All positions overlay
    draw_field_background(ax1)
    if len(offense_data) > 0:
        ax1.scatter(offense_data['x'], offense_data['y'],
                    color=colors['offense'], alpha=0.6, s=50, label='Offense')
    if len(defense_data) > 0:
        ax1.scatter(defense_data['x'], defense_data['y'],
                    color=colors['defense'], alpha=0.6, s=50, label='Defense')
    ax1.legend()
    ax1.set_title('All Player Positions')

    # 2. Average positions by frame
    draw_field_background(ax2)
    if 'frame_id' in df_clean.columns and len(offense_data) > 0 and len(defense_data) > 0:
        frame_avg_off = offense_data.groupby('frame_id')[['x', 'y']].mean()
        frame_avg_def = defense_data.groupby('frame_id')[['x', 'y']].mean()

        ax2.plot(frame_avg_off['x'], frame_avg_off['y'],
                 color=colors['offense'], linewidth=3, label='Offense Center', alpha=0.8)
        ax2.plot(frame_avg_def['x'], frame_avg_def['y'],
                 color=colors['defense'], linewidth=3, label='Defense Center', alpha=0.8)
        ax2.legend()
    ax2.set_title('Formation Centers Over Time')

    # 3. Stress comparison
    if 'node_stress' in df_clean.columns and 'frame_id' in df_clean.columns and len(offense_data) > 0 and len(defense_data) > 0:
        stress_by_frame_off = offense_data.groupby('frame_id')['node_stress'].mean()
        stress_by_frame_def = defense_data.groupby('frame_id')['node_stress'].mean()

        ax3.plot(stress_by_frame_off.index, stress_by_frame_off.values,
                 color=colors['offense'], linewidth=2, label='Offense')
        ax3.plot(stress_by_frame_def.index, stress_by_frame_def.values,
                 color=colors['defense'], linewidth=2, label='Defense')
        ax3.set_xlabel('Frame ID')
        ax3.set_ylabel('Average Node Stress')
        ax3.set_title('Team Stress Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.axis('off')

    # 4. Speed comparison
    speed_off = np.sqrt(offense_data['s']**2) if len(offense_data) > 0 else np.array([])
    speed_def = np.sqrt(defense_data['s']**2) if len(defense_data) > 0 else np.array([])

    ax4.hist(speed_off, bins=15, alpha=0.7, color=colors['offense'],
             label='Offense', density=True)
    ax4.hist(speed_def, bins=15, alpha=0.7, color=colors['defense'],
             label='Defense', density=True)
    ax4.set_xlabel('Speed (yards/sec)')
    ax4.set_ylabel('Density')
    ax4.set_title('Speed Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/team_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def generate_player_network():
    """Generate a network-style visualization showing player relationships"""

    print("  Generating player network analysis...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Player Interaction Network', fontsize=16, fontweight='bold')

    # Calculate average positions for each player
    player_avg_pos = df_clean.groupby('nfl_id')[['x', 'y', 'player_side']].agg({
        'x': 'mean',
        'y': 'mean',
        'player_side': 'first'
    }).reset_index()

    # 1. Formation diagram
    draw_field_background(ax1)

    for _, player in player_avg_pos.iterrows():
        c = colors['offense'] if player['player_side'] == 'Offense' else colors['defense']
        player_name = PLAYER_NAMES.get(player['nfl_id'], f"P{player['nfl_id']}")

        circle = plt.Circle((player['x'], player['y']), 2,
                            color=c, alpha=0.7, zorder=5)
        ax1.add_patch(circle)

        ax1.annotate(player_name.split()[-1],
                     (player['x'], player['y']),
                     ha='center', va='center',
                     fontsize=8, fontweight='bold', zorder=6)

        teammates = player_avg_pos[
            (player_avg_pos['player_side'] == player['player_side']) &
            (player_avg_pos['nfl_id'] != player['nfl_id'])
        ]

        for _, teammate in teammates.iterrows():
            distance = np.sqrt((player['x'] - teammate['x'])**2 +
                               (player['y'] - teammate['y'])**2)
            if distance < 15:
                ax1.plot([player['x'], teammate['x']],
                         [player['y'], teammate['y']],
                         color=c, alpha=0.3, linewidth=1, zorder=1)

    ax1.set_title('Average Formation with Connections')

    # 2. Distance matrix heatmap

    n_players = len(player_avg_pos)
    distance_matrix = np.zeros((n_players, n_players))

    for i, player1 in player_avg_pos.iterrows():
        for j, player2 in player_avg_pos.iterrows():
            if i != j:
                distance = np.sqrt((player1['x'] - player2['x'])**2 +
                                   (player1['y'] - player2['y'])**2)
                distance_matrix[i, j] = distance

    player_labels = [PLAYER_NAMES.get(pid, f"P{pid}") for pid in player_avg_pos['nfl_id']]

    sns.heatmap(distance_matrix,
                xticklabels=player_labels,
                yticklabels=player_labels,
                annot=True, fmt='.1f',
                cmap='viridis_r', ax=ax2,
                cbar_kws={'label': 'Distance (yards)'})
    ax2.set_title('Player Distance Matrix')

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/player_network.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Run all analyses
print(" Generating Individual Player Analysis...")
generate_individual_player_analysis()

print(" Generating Team Comparison...")
generate_player_comparison()

print(" Generating Player Network...")
generate_player_network()

print(f"\n{'='*60}")
print("INDIVIDUAL PLAYER ANALYSIS COMPLETE!")
print(f"{'='*60}")
print("Output directory:", OUTPUT_DIR)
print("Generated files:")

analysis_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
for file in sorted(analysis_files):
    print(f"  âœ“ {file}")

print("\nğŸ“ˆ This analysis provides:")
print("  â€¢ Individual movement trails for each player")
print("  â€¢ Position density heatmaps")
print("  â€¢ Stress and speed metrics over time")
print("  â€¢ Team formation comparisons")
print("  â€¢ Player interaction networks")
print("  â€¢ Distance relationships between players")


"""
NFL Tactical Analysis - Working with Limited Player Coverage
Generate strategic insights from partial player tracking data
"""

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION (KAGGLE)
# ============================================================================
# INPUT_CSV, OUTPUT_DIR, TEAM_THEME are expected to be defined in the setup cell.

colors = {'offense': '#69BE28', 'defense': '#A5ACAF', 'primary': '#002244'}
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print("NFL TACTICAL ANALYSIS")
print("Limited Player Coverage Optimization")
print(f"{'='*60}\n")

# Load data
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"INPUT_CSV not found.\nExpected: {INPUT_CSV}\n"
        f"Tip: verify DATASET_SLUG and INPUT_FILENAME in the setup cell."
    )

df = pd.read_csv(INPUT_CSV)
df_clean = df[~df['s'].isna() & ~df['x'].isna() & ~df['y'].isna()].copy()

print(f"Coverage: {df_clean['nfl_id'].nunique()}/22 players ({df_clean['nfl_id'].nunique()/22*100:.1f}%)")
print(f"Frames: {df_clean['frame_id'].nunique()}")

def draw_field_background(ax, show_zones=True):
    """Draw field with tactical zones"""
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_aspect('equal')
    ax.set_facecolor('#2d5016')

    # Basic field markings
    for yard in range(10, 111, 10):
        ax.axvline(yard, color='white', alpha=0.4, linewidth=1)

    ax.axvline(10, color='white', linewidth=3)   # Goal line
    ax.axvline(110, color='white', linewidth=3)  # Goal line
    ax.axhline(0, color='white', linewidth=2)    # Sideline
    ax.axhline(53.3, color='white', linewidth=2) # Sideline

    if show_zones:
        # Add tactical zones
        ax.axvspan(0, 20, alpha=0.1, color='red', label='Red Zone')
        ax.axvspan(100, 120, alpha=0.1, color='red')

        ax.axvspan(40, 80, alpha=0.05, color='yellow', label='Middle Field')

        ax.axhline(23.36, color='white', alpha=0.3, linestyle='--')
        ax.axhline(29.94, color='white', alpha=0.3, linestyle='--')

def analyze_coverage_gaps():
    """Identify areas with limited or missing coverage"""
    print("ğŸ“� Analyzing Coverage Gaps...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Coverage Gap Analysis', fontsize=16, fontweight='bold')

    # 1. Coverage density heatmap
    draw_field_background(ax1, show_zones=False)

    x_bins = np.linspace(0, 120, 25)
    y_bins = np.linspace(0, 53.3, 15)

    coverage_counts = np.zeros((len(y_bins)-1, len(x_bins)-1))

    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            x_mask = (df_clean['x'] >= x_bins[i]) & (df_clean['x'] < x_bins[i+1])
            y_mask = (df_clean['y'] >= y_bins[j]) & (df_clean['y'] < y_bins[j+1])
            coverage_counts[j, i] = len(df_clean[x_mask & y_mask])

    im1 = ax1.imshow(coverage_counts, extent=[0, 120, 0, 53.3],
                     cmap='Reds', alpha=0.7, aspect='auto', origin='lower')
    ax1.set_title('Player Position Density')
    plt.colorbar(im1, ax=ax1, label='Observation Count')

    # 2. Missing positions (low coverage areas)
    draw_field_background(ax2, show_zones=False)

    low_coverage_mask = coverage_counts < np.percentile(coverage_counts, 25)

    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            if low_coverage_mask[j, i] and coverage_counts[j, i] > 0:
                rect = patches.Rectangle((x_bins[i], y_bins[j]),
                                         x_bins[i+1] - x_bins[i],
                                         y_bins[j+1] - y_bins[j],
                                         linewidth=1, edgecolor='red',
                                         facecolor='red', alpha=0.3)
                ax2.add_patch(rect)

    ax2.set_title('Low Coverage Areas (Bottom 25%)')

    # 3. Player distribution by side
    offense_data = df_clean[df_clean['player_side'] == 'Offense']
    defense_data = df_clean[df_clean['player_side'] == 'Defense']

    ax3.hist2d(offense_data['x'], offense_data['y'], bins=15,
               alpha=0.6, cmap='Greens')
    ax3.set_title(f'Offense Coverage ({len(offense_data["nfl_id"].unique())} players)')
    ax3.set_xlim(0, 120)
    ax3.set_ylim(0, 53.3)

    ax4.hist2d(defense_data['x'], defense_data['y'], bins=15,
               alpha=0.6, cmap='Blues')
    ax4.set_title(f'Defense Coverage ({len(defense_data["nfl_id"].unique())} players)')
    ax4.set_xlim(0, 120)
    ax4.set_ylim(0, 53.3)

    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/coverage_analysis.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return coverage_counts

def analyze_formation_changes():
    """Analyze how formations change over time with limited data"""
    print("ğŸ“� Analyzing Formation Evolution...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Formation Analysis (Partial Coverage)', fontsize=16, fontweight='bold')

    formation_metrics = []

    for frame in sorted(df_clean['frame_id'].unique()):
        frame_data = df_clean[df_clean['frame_id'] == frame]

        if len(frame_data) > 0:
            center_x = frame_data['x'].mean()
            center_y = frame_data['y'].mean()

            spread_x = frame_data['x'].std()
            spread_y = frame_data['y'].std()

            offense = frame_data[frame_data['player_side'] == 'Offense']
            defense = frame_data[frame_data['player_side'] == 'Defense']

            metrics = {
                'frame': frame,
                'center_x': center_x,
                'center_y': center_y,
                'spread_x': spread_x,
                'spread_y': spread_y,
                'total_players': len(frame_data),
                'offense_players': len(offense),
                'defense_players': len(defense)
            }

            if len(offense) > 0:
                metrics['offense_center_x'] = offense['x'].mean()
                metrics['offense_center_y'] = offense['y'].mean()
                metrics['offense_spread_y'] = offense['y'].std()

            if len(defense) > 0:
                metrics['defense_center_x'] = defense['x'].mean()
                metrics['defense_center_y'] = defense['y'].mean()
                metrics['defense_spread_y'] = defense['y'].std()

            if 'node_stress' in frame_data.columns:
                metrics['avg_stress'] = frame_data['node_stress'].mean()
                if len(offense) > 0:
                    metrics['offense_stress'] = offense['node_stress'].mean()
                if len(defense) > 0:
                    metrics['defense_stress'] = defense['node_stress'].mean()

            formation_metrics.append(metrics)

    formation_df = pd.DataFrame(formation_metrics)

    if len(formation_df) > 0:
        draw_field_background(ax1)
        ax1.plot(formation_df['center_x'], formation_df['center_y'],
                 'ko-', linewidth=2, markersize=8, alpha=0.7, label='Formation Center')

        for i in range(len(formation_df)-1):
            dx = formation_df.iloc[i+1]['center_x'] - formation_df.iloc[i]['center_x']
            dy = formation_df.iloc[i+1]['center_y'] - formation_df.iloc[i]['center_y']
            ax1.arrow(formation_df.iloc[i]['center_x'], formation_df.iloc[i]['center_y'],
                      dx, dy, head_width=1, head_length=1, fc='black', ec='black', alpha=0.5)

        ax1.set_title('Formation Center Movement')
        ax1.legend()

        ax2.plot(formation_df['frame'], formation_df['spread_y'],
                 'b-', linewidth=2, label='Overall Width')

        if 'offense_spread_y' in formation_df.columns:
            ax2.plot(formation_df['frame'], formation_df['offense_spread_y'],
                     color=colors['offense'], linewidth=2, label='Offense Width')
        if 'defense_spread_y' in formation_df.columns:
            ax2.plot(formation_df['frame'], formation_df['defense_spread_y'],
                     color=colors['defense'], linewidth=2, label='Defense Width')

        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Formation Width (yards)')
        ax2.set_title('Formation Width Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3.plot(formation_df['frame'], formation_df['total_players'],
                 'k-', linewidth=2, label='Total Tracked')
        ax3.plot(formation_df['frame'], formation_df['offense_players'],
                 color=colors['offense'], linewidth=2, label='Offense')
        ax3.plot(formation_df['frame'], formation_df['defense_players'],
                 color=colors['defense'], linewidth=2, label='Defense')

        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Players Tracked')
        ax3.set_title('Player Coverage Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        if 'avg_stress' in formation_df.columns:
            scatter = ax4.scatter(formation_df['spread_y'], formation_df['avg_stress'],
                                  c=formation_df['frame'], cmap='viridis', s=60, alpha=0.7)
            ax4.set_xlabel('Formation Width')
            ax4.set_ylabel('Average Stress')
            ax4.set_title('Formation Width vs Stress')
            plt.colorbar(scatter, ax=ax4, label='Frame')
            ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/formation_evolution.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return formation_df

def analyze_tactical_zones():
    """Analyze player activity in different tactical zones"""
    print(" Analyzing Tactical Zones...")

    def get_tactical_zone(x, y):
        if x <= 20:
            v_zone = "Own_End"
        elif x <= 40:
            v_zone = "Own_Side"
        elif x <= 60:
            v_zone = "Midfield"
        elif x <= 80:
            v_zone = "Opp_Side"
        else:
            v_zone = "Red_Zone"

        if y <= 17.77:
            h_zone = "Left"
        elif y <= 35.53:
            h_zone = "Center"
        else:
            h_zone = "Right"

        return f"{v_zone}_{h_zone}"

    df_clean['tactical_zone'] = df_clean.apply(lambda row: get_tactical_zone(row['x'], row['y']), axis=1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Tactical Zone Analysis', fontsize=16, fontweight='bold')

    zone_counts = df_clean.groupby(['tactical_zone', 'player_side']).size().unstack(fill_value=0)
    zone_counts.plot(kind='bar', ax=ax1, color=[colors['defense'], colors['offense']])
    ax1.set_title('Zone Occupation by Side')
    ax1.set_xlabel('Tactical Zone')
    ax1.set_ylabel('Observation Count')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Side')

    if 'node_stress' in df_clean.columns:
        zone_stress = df_clean.groupby(['tactical_zone', 'player_side'])['node_stress'].mean().unstack()
        zone_stress.plot(kind='bar', ax=ax2, color=[colors['defense'], colors['offense']])
        ax2.set_title('Average Stress by Zone')
        ax2.set_xlabel('Tactical Zone')
        ax2.set_ylabel('Average Node Stress')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(title='Side')
    else:
        ax2.axis('off')

    transitions = {}
    for player in df_clean['nfl_id'].unique():
        player_data = df_clean[df_clean['nfl_id'] == player].sort_values('frame_id')
        for i in range(len(player_data) - 1):
            current_zone = player_data.iloc[i]['tactical_zone']
            next_zone = player_data.iloc[i+1]['tactical_zone']
            if current_zone != next_zone:
                transition = f"{current_zone} -> {next_zone}"
                transitions[transition] = transitions.get(transition, 0) + 1

    if transitions:
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]
        transition_names = [t[0] for t in sorted_transitions]
        transition_counts = [t[1] for t in sorted_transitions]

        ax3.barh(range(len(transition_names)), transition_counts)
        ax3.set_yticks(range(len(transition_names)))
        ax3.set_yticklabels(transition_names, fontsize=8)
        ax3.set_xlabel('Transition Count')
        ax3.set_title('Top Zone Transitions')
    else:
        ax3.axis('off')

    all_zones = [f"{v}_{h}" for v in ["Own_End", "Own_Side", "Midfield", "Opp_Side", "Red_Zone"]
                 for h in ["Left", "Center", "Right"]]

    covered_zones = set(df_clean['tactical_zone'].unique())
    zone_coverage = []
    for zone in all_zones:
        if zone in covered_zones:
            zone_coverage.append(len(df_clean[df_clean['tactical_zone'] == zone]))
        else:
            zone_coverage.append(0)

    bars = ax4.bar(range(len(all_zones)), zone_coverage)
    ax4.set_xticks(range(len(all_zones)))
    ax4.set_xticklabels(all_zones, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Observation Count')
    ax4.set_title(f'Zone Coverage ({len(covered_zones)}/{len(all_zones)} zones covered)')

    for bar, count in zip(bars, zone_coverage):
        if count == 0:
            bar.set_color('red'); bar.set_alpha(0.3)
        elif count < 5:
            bar.set_color('orange'); bar.set_alpha(0.6)
        else:
            bar.set_color('green'); bar.set_alpha(0.8)

    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/tactical_zones.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def generate_tactical_insights_summary(coverage_counts, formation_df):
    """Generate summary insights for tactical analysis"""
    print("ğŸ“Š Generating Tactical Summary...")

    insights = {
        'coverage_percentage': (df_clean['nfl_id'].nunique() / 22) * 100,
        'total_observations': len(df_clean),
        'frames_analyzed': df_clean['frame_id'].nunique(),
        'offense_players_tracked': len(df_clean[df_clean['player_side'] == 'Offense']['nfl_id'].unique()),
        'defense_players_tracked': len(df_clean[df_clean['player_side'] == 'Defense']['nfl_id'].unique()),
    }

    if 'node_stress' in df_clean.columns:
        insights['avg_stress'] = df_clean['node_stress'].mean()
        insights['max_stress'] = df_clean['node_stress'].max()
        insights['stress_variance'] = df_clean['node_stress'].var()

    field_x_coverage = (df_clean['x'].max() - df_clean['x'].min()) / 120 * 100
    field_y_coverage = (df_clean['y'].max() - df_clean['y'].min()) / 53.3 * 100

    insights['field_x_coverage'] = field_x_coverage
    insights['field_y_coverage'] = field_y_coverage

    if len(formation_df) > 0:
        insights['formation_stability_x'] = formation_df['spread_x'].std()
        insights['formation_stability_y'] = formation_df['spread_y'].std()
        insights['avg_formation_width'] = formation_df['spread_y'].mean()

    with open(f'{OUTPUT_DIR}/tactical_insights.txt', 'w') as f:
        f.write("NFL TACTICAL ANALYSIS - KEY INSIGHTS\n")
        f.write("="*50 + "\n\n")

        f.write("COVERAGE ANALYSIS:\n")
        f.write(f"  â€¢ Player Coverage: {insights['coverage_percentage']:.1f}% ({df_clean['nfl_id'].nunique()}/22 players)\n")
        f.write(f"  â€¢ Offense Players Tracked: {insights['offense_players_tracked']}\n")
        f.write(f"  â€¢ Defense Players Tracked: {insights['defense_players_tracked']}\n")
        f.write(f"  â€¢ Total Observations: {insights['total_observations']:,}\n")
        f.write(f"  â€¢ Frames Analyzed: {insights['frames_analyzed']}\n\n")

        f.write("FIELD COVERAGE:\n")
        f.write(f"  â€¢ X-axis Coverage: {insights['field_x_coverage']:.1f}% of field length\n")
        f.write(f"  â€¢ Y-axis Coverage: {insights['field_y_coverage']:.1f}% of field width\n\n")

        if 'avg_stress' in insights:
            f.write("STRESS ANALYSIS:\n")
            f.write(f"  â€¢ Average Node Stress: {insights['avg_stress']:.3f}\n")
            f.write(f"  â€¢ Maximum Stress Recorded: {insights['max_stress']:.3f}\n")
            f.write(f"  â€¢ Stress Variance: {insights['stress_variance']:.3f}\n\n")

        if len(formation_df) > 0:
            f.write("FORMATION ANALYSIS:\n")
            f.write(f"  â€¢ Average Formation Width: {insights['avg_formation_width']:.2f} yards\n")
            f.write(f"  â€¢ Formation X Stability: {insights['formation_stability_x']:.2f}\n")
            f.write(f"  â€¢ Formation Y Stability: {insights['formation_stability_y']:.2f}\n\n")

        f.write("RECOMMENDATIONS FOR LIMITED COVERAGE:\n")
        f.write("  â€¢ Focus on relative positioning rather than absolute formations\n")
        f.write("  â€¢ Use stress patterns to identify key moments\n")
        f.write("  â€¢ Analyze zone transitions for tactical insights\n")
        f.write("  â€¢ Consider formation stability as a measure of discipline\n")
        f.write("  â€¢ Use partial coverage to understand player roles\n")

# Run all tactical analyses
coverage_data = analyze_coverage_gaps()
formation_data = analyze_formation_changes()
analyze_tactical_zones()
generate_tactical_insights_summary(coverage_data, formation_data)

print(f"\n{'='*60}")
print("TACTICAL ANALYSIS COMPLETE!")
print(f"{'='*60}")
print("Output directory:", OUTPUT_DIR)
print("Generated files:")

tactical_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.png', '.txt'))]
for file in sorted(tactical_files):
    print(f"  âœ“ {file}")

print("\n Tactical Insights Generated:")
print("  â€¢ Coverage gap analysis")
print("  â€¢ Formation evolution tracking")
print("  â€¢ Tactical zone occupation")
print("  â€¢ Player transition patterns")
print("  â€¢ Strategic recommendations for partial coverage")


"""
NFL Big Data Bowl - Enhanced Animation with DCI/DIS Gauge
Includes real-time defensive metrics visualization
"""

from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Circle, Rectangle
warnings.filterwarnings('ignore')

# ============================================================================
# TEAM COLOR SCHEMES
# ============================================================================
TEAM_COLORS = {
    'vikings': {
        'name': 'Minnesota Vikings',
        'primary': '#4F2683',
        'secondary': '#FFC62F',
        'background': '#4F2683',
        'field': '#2d5016',
        'offense_color': '#FFC62F',
        'defense_color': '#FFFFFF',
        'field_stripe': '#3d3166'
    },
    'seahawks': {
        'name': 'Seattle Seahawks',
        'primary': '#002244',
        'secondary': '#69BE28',
        'background': '#002244',
        'field': '#2d5016',
        'offense_color': '#69BE28',
        'defense_color': '#A5ACAF',
        'field_stripe': '#001933'
    },
    'default': {
        'name': 'NFL Default',
        'primary': '#013369',
        'secondary': '#D50A0A',
        'background': '#0a0e1a',
        'field': '#2d5016',
        'offense_color': '#1E90FF',
        'defense_color': '#DC143C',
        'field_stripe': '#1d3010'
    }
}

# ============================================================================
# CONFIGURATION (KAGGLE)
# ============================================================================
# INPUT_CSV and OUTPUT_DIR are expected to be defined in the setup cell.
TEAM_THEME = 'vikings'

GAME_ID = None
PLAY_ID = None

# Animation settings
FPS = 10
VIDEO_FORMAT = 'gif'   # 'mp4', 'gif', or 'both'  (use 'gif' if ffmpeg not available)
SHOW_POSITIONS = False
SHOW_TEAM_LOGO_AREA = True
SHOW_NODE_STRESS = True
SHOW_DCI_DIS_GAUGE = True

# Metric calculation mode
METRIC_MODE = 'geometric'   # 'precomputed' or 'geometric'
DCI_COLUMN = 'dci_score'
DIS_COLUMN = 'dis_score'

# ============================================================================
# PLAYER NAME LOOKUPS (unchanged)
# ============================================================================
VIKINGS_49ERS_PLAYER_NAMES = {
    38632: 'Jordan Addison',
    47791: 'Kirk Cousins',
    47852: 'Justin Jefferson',
    47885: 'T.J. Hockenson',
    52584: 'Alexander Mattison',
    55887: 'K.J. Osborn',
    38868: 'Nick Bosa',
    46139: 'Fred Warner',
    46157: 'Dre Greenlaw',
    46757: 'Javon Hargrave',
    47931: 'Charvarius Ward',
    53601: 'Talanoa Hufanga',
    53609: 'Deommodore Lenoir',
}

SEAHAWKS_GIANTS_PLAYER_NAMES = {
    38577: 'Bobby Wagner',
    39987: 'Drew Lock',
    42412: 'Jake Bobo',
    42543: 'Quandre Diggs',
    42547: 'Will Dissly',
    43329: 'Tyler Lockett',
    43333: 'Uchenna Nwosu',
    44818: 'Boye Mafe',
    44830: 'Riq Woolen',
    45186: 'Kenneth Walker III',
    46117: 'Jarran Reed',
    46189: 'Darren Waller',
    47789: 'Geno Smith',
    47793: 'Bobby Okereke',
    47803: 'Noah Fant',
    47825: 'Daniel Jones',
    47842: 'Darius Slayton',
    47847: 'DK Metcalf',
    47872: 'Jordyn Brooks',
    47891: 'Jamal Adams',
    47941: 'Devon Witherspoon',
    47954: 'Jaxon Smith-Njigba',
    48266: 'Sterling Shepard',
    52416: 'Leonard Williams',
    52435: "Dre'Mont Jones",
    52444: 'Xavier McKinney',
    52541: 'Colby Parkinson',
    52552: 'Saquon Barkley',
    52615: 'Dareke Young',
    53604: 'Julian Love',
    53625: 'Zach Charbonnet',
    54014: 'Tre Brown',
    54470: 'Johnathan Abram',
    54506: 'DeeJay Dallas',
    54508: "Wan'Dale Robinson",
    54546: 'Deonte Banks',
    54577: 'Pharaoh Brown',
    54579: 'Isaiah Simmons',
    54611: 'Micah McFadden',
    54618: "Adoree' Jackson",
    55869: 'Nick McCloud',
    55884: 'Jalin Hyatt',
    55888: "Cor'Dale Flott",
    55902: 'Dane Belton',
    55917: 'Matt Breida',
    55938: 'Parris Campbell',
    56063: 'Bobby McCain',
    56471: 'Isaiah Hodgins',
}

GAME_MATCHUP = 'vikings_49ers'  # 'vikings_49ers' or 'seahawks_giants'

if GAME_MATCHUP == 'vikings_49ers':
    PLAYER_NAMES = VIKINGS_49ERS_PLAYER_NAMES
elif GAME_MATCHUP == 'seahawks_giants':
    PLAYER_NAMES = SEAHAWKS_GIANTS_PLAYER_NAMES
else:
    PLAYER_NAMES = {}
    print(f"Warning: Unknown GAME_MATCHUP '{GAME_MATCHUP}'. Using empty player database.")

SHOW_PLAYER_NAMES = True
SHOW_PLAYER_NUMBERS = False
SHOW_POSITIONS = False

# ============================================================================

colors = TEAM_COLORS.get(TEAM_THEME, TEAM_COLORS['default'])
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"NFL ANIMATION - {colors['name']} Theme")
print(f"{'='*60}\n")

# Load data
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"INPUT_CSV not found.\nExpected: {INPUT_CSV}\n"
        f"Tip: check DATASET_SLUG / INPUT_CSV_NAME in the setup cell."
    )

print(f"Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

print(f"Total rows: {len(df):,}")
print(f"Available columns: {list(df.columns)}")

# For new animation data format without game_id/play_id
if 'game_id' not in df.columns or 'play_id' not in df.columns:
    print("New animation data format detected (no game_id/play_id columns)")
    print(f"Players: {df['nfl_id'].nunique()}")
    print(f"Frames: {df['frame_id'].nunique()}")

    play_data = df.copy()
    GAME_ID = "animation_data"
    PLAY_ID = "101"
else:
    print(f"Total games: {df['game_id'].nunique()}")
    print(f"Total plays: {len(df.groupby(['game_id', 'play_id']))}")

    play_summary = df.groupby(['game_id', 'play_id']).agg({
        'frame_id': 'nunique',
        'nfl_id': 'nunique'
    }).reset_index()
    play_summary.columns = ['game_id', 'play_id', 'frames', 'players']
    play_summary = play_summary.sort_values(['players', 'frames'], ascending=[False, False])

    if GAME_ID is None or PLAY_ID is None:
        print(f"\n{'='*60}")
        print("AVAILABLE PLAYS IN YOUR CSV")
        print(f"{'='*60}\n")
        print("Top 20 plays by player count:\n")
        print(play_summary.head(20).to_string(index=False))
        raise SystemExit("Set GAME_ID and PLAY_ID near the top and rerun.")

    play_data = df[(df['game_id'] == GAME_ID) & (df['play_id'] == PLAY_ID)].copy()
    if len(play_data) == 0:
        raise ValueError(f"No data found for Game {GAME_ID}, Play {PLAY_ID}.")

has_dci = DCI_COLUMN in df.columns
has_dis = DIS_COLUMN in df.columns

if METRIC_MODE == 'precomputed' and not (has_dci and has_dis):
    print("\nâš ï¸�  WARNING: METRIC_MODE='precomputed' but columns not found.")
    print(f"Looking for: {DCI_COLUMN}, {DIS_COLUMN}")
    print("Switching to 'geometric' mode...")
    METRIC_MODE = 'geometric'

print(f"\nGame ID: {GAME_ID}")
print(f"Play ID: {PLAY_ID}")
print(f"Players: {play_data['nfl_id'].nunique()}")
print(f"Frames: {play_data['frame_id'].nunique()}")
print(f"Theme: {colors['name']}")
print(f"Metric Mode: {METRIC_MODE.upper()}")

# Separate actual positions from projections
play_data['is_projection'] = play_data['s'].isna()

def calculate_geometric_dci(defense_positions):
    if len(defense_positions) < 2:
        return 0.5
    distances = []
    for i, pos in enumerate(defense_positions):
        other_pos = np.delete(defense_positions, i, axis=0)
        if len(other_pos) > 0:
            dists = np.linalg.norm(other_pos - pos, axis=1)
            distances.append(np.min(dists))
    if not distances:
        return 0.5
    avg_spacing = np.mean(distances)
    dci = np.clip(1.0 - (avg_spacing - 3.0) / 12.0, 0.0, 1.0)
    return float(dci)

def calculate_geometric_dis(defense_positions, prev_defense_positions=None):
    if len(defense_positions) < 3:
        return 0.5
    distances = []
    for i in range(len(defense_positions)):
        for j in range(i + 1, len(defense_positions)):
            dist = np.linalg.norm(defense_positions[i] - defense_positions[j])
            distances.append(dist)
    if not distances:
        return 0.5
    std_spacing = np.std(distances)
    mean_spacing = np.mean(distances)
    if mean_spacing > 0:
        cv = std_spacing / mean_spacing
        dis = np.clip(1.0 - cv, 0.0, 1.0)
    else:
        dis = 0.5
    return float(dis)

def draw_field(ax, colors):
    for yard in range(10, 111, 5):
        linewidth = 2 if yard % 10 == 0 else 1
        color = 'white' if yard % 10 == 0 else colors['secondary']
        alpha = 0.7 if yard % 10 == 0 else 0.4
        ax.plot([yard, yard], [0, 53.3], color=color, linewidth=linewidth, alpha=alpha)

    ax.plot([10, 10], [0, 53.3], color=colors['secondary'], linewidth=4, alpha=0.9)
    ax.plot([110, 110], [0, 53.3], color=colors['secondary'], linewidth=4, alpha=0.9)

    ax.plot([0, 120], [0, 0], color='white', linewidth=2, alpha=0.8)
    ax.plot([0, 120], [53.3, 53.3], color='white', linewidth=2, alpha=0.8)
    ax.plot([0, 0], [0, 53.3], color='white', linewidth=2, alpha=0.8)
    ax.plot([120, 120], [0, 53.3], color='white', linewidth=2, alpha=0.8)

    for yard in range(10, 111):
        ax.plot([yard, yard], [23.36, 23.36], color='white', marker='.', markersize=2, alpha=0.6)
        ax.plot([yard, yard], [29.94, 29.94], color='white', marker='.', markersize=2, alpha=0.6)

    for yard in range(0, 120, 10):
        rect = patches.Rectangle((yard, 0), 5, 53.3, linewidth=0,
                                 edgecolor='none', facecolor=colors['field_stripe'], alpha=0.15)
        ax.add_patch(rect)

    if SHOW_TEAM_LOGO_AREA:
        top_left = patches.Rectangle((0, 48), 8, 5.3, linewidth=0,
                                     facecolor=colors['primary'], alpha=0.3)
        top_right = patches.Rectangle((112, 48), 8, 5.3, linewidth=0,
                                      facecolor=colors['secondary'], alpha=0.3)
        ax.add_patch(top_left)
        ax.add_patch(top_right)

def draw_dci_dis_gauge(ax, dci_value, dis_value, colors):
    artists = []

    gauge_x, gauge_y = 102, 43
    gauge_width, gauge_height = 16, 9

    bg = Rectangle((gauge_x, gauge_y), gauge_width, gauge_height,
                   facecolor=colors['primary'], edgecolor='white',
                   linewidth=2, alpha=0.9, zorder=100)
    ax.add_patch(bg); artists.append(bg)

    title = ax.text(gauge_x + gauge_width/2, gauge_y + gauge_height - 1,
                    'DEFENSIVE METRICS', ha='center', va='top',
                    fontsize=10, color='white', fontweight='bold', zorder=101)
    artists.append(title)

    bar_y_dci = gauge_y + gauge_height - 3.5
    bar_height = 1.2
    bar_width = gauge_width - 3

    dci_bg = Rectangle((gauge_x + 1.5, bar_y_dci), bar_width, bar_height,
                       facecolor='#333333', edgecolor='white',
                       linewidth=1, alpha=0.5, zorder=101)
    ax.add_patch(dci_bg); artists.append(dci_bg)

    dci_color = plt.cm.RdYlGn(dci_value)
    dci_fill = Rectangle((gauge_x + 1.5, bar_y_dci), bar_width * dci_value, bar_height,
                         facecolor=dci_color, edgecolor='none', alpha=0.8, zorder=102)
    ax.add_patch(dci_fill); artists.append(dci_fill)

    dci_label = ax.text(gauge_x + 1.5, bar_y_dci - 0.4, 'DCI (Coverage)',
                        ha='left', va='top', fontsize=8, color='white',
                        fontweight='bold', zorder=103)
    artists.append(dci_label)

    dci_text = ax.text(gauge_x + gauge_width - 1.5, bar_y_dci + bar_height/2,
                       f'{dci_value:.3f}', ha='right', va='center',
                       fontsize=9, color='white', fontweight='bold', zorder=103)
    artists.append(dci_text)

    bar_y_dis = bar_y_dci - 2.5

    dis_bg = Rectangle((gauge_x + 1.5, bar_y_dis), bar_width, bar_height,
                       facecolor='#333333', edgecolor='white',
                       linewidth=1, alpha=0.5, zorder=101)
    ax.add_patch(dis_bg); artists.append(dis_bg)

    dis_color = plt.cm.RdYlGn(dis_value)
    dis_fill = Rectangle((gauge_x + 1.5, bar_y_dis), bar_width * dis_value, bar_height,
                         facecolor=dis_color, edgecolor='none', alpha=0.8, zorder=102)
    ax.add_patch(dis_fill); artists.append(dis_fill)

    dis_label = ax.text(gauge_x + 1.5, bar_y_dis - 0.4, 'DIS (Integrity)',
                        ha='left', va='top', fontsize=8, color='white',
                        fontweight='bold', zorder=103)
    artists.append(dis_label)

    dis_text = ax.text(gauge_x + gauge_width - 1.5, bar_y_dis + bar_height/2,
                       f'{dis_value:.3f}', ha='right', va='center',
                       fontsize=9, color='white', fontweight='bold', zorder=103)
    artists.append(dis_text)

    return artists

# Create figure with team background
fig = plt.figure(figsize=(16, 10), facecolor=colors['background'])
ax = fig.add_subplot(111, facecolor=colors['field'])
ax.set_xlim(0, 120)
ax.set_ylim(0, 53.3)
ax.set_aspect('equal')
draw_field(ax, colors)

offense_scatter = ax.scatter([], [], c=colors['offense_color'], s=350,
                             edgecolors='white', linewidths=2.5, zorder=5,
                             label='Offense', alpha=0.95)
defense_scatter = ax.scatter([], [], c=colors['defense_color'], s=350,
                             edgecolors=colors['primary'], linewidths=2.5,
                             zorder=5, label='Defense', alpha=0.95)

player_labels = []
stress_circles = []
gauge_artists = []

stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                     fontsize=14, verticalalignment='top',
                     color='white', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor=colors['primary'],
                               alpha=0.85, edgecolor=colors['secondary'], linewidth=2))

title = ax.text(0.5, 0.98, 'Minnesota Vikings vs San Francisco 49ers - Play 79',
                transform=ax.transAxes, fontsize=18,
                verticalalignment='top', horizontalalignment='center',
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=colors['primary'],
                          alpha=0.85, edgecolor=colors['secondary'], linewidth=3))

ax.set_xlabel('Yards', color='white', fontsize=12, fontweight='bold')
ax.set_ylabel('Field Width (yards)', color='white', fontsize=12, fontweight='bold')
ax.tick_params(colors='white')

prev_defense_positions = None

def init():
    offense_scatter.set_offsets(np.empty((0, 2)))
    defense_scatter.set_offsets(np.empty((0, 2)))
    stats_text.set_text('')
    return [offense_scatter, defense_scatter, stats_text]

def update(frame_num):
    global prev_defense_positions

    frame_ids = sorted(play_data['frame_id'].unique())
    frame_id = frame_ids[frame_num]
    current_frame = play_data[play_data['frame_id'] == frame_id]

    actual_positions = current_frame[~current_frame['is_projection']]

    for label in player_labels:
        label.remove()
    player_labels.clear()

    for circle in stress_circles:
        circle.remove()
    stress_circles.clear()

    for artist in gauge_artists:
        artist.remove()
    gauge_artists.clear()

    offense_data = actual_positions[actual_positions['player_side'] == 'Offense']
    defense_data = actual_positions[actual_positions['player_side'] == 'Defense']

    offense_scatter.set_offsets(offense_data[['x', 'y']].values if len(offense_data) else np.empty((0, 2)))
    defense_scatter.set_offsets(defense_data[['x', 'y']].values if len(defense_data) else np.empty((0, 2)))

    if METRIC_MODE == 'precomputed' and has_dci and has_dis:
        dci_value = float(current_frame[DCI_COLUMN].mean())
        dis_value = float(current_frame[DIS_COLUMN].mean())
    else:
        if len(defense_data) > 0:
            defense_positions = defense_data[['x', 'y']].values
            dci_value = calculate_geometric_dci(defense_positions)
            dis_value = calculate_geometric_dis(defense_positions, prev_defense_positions)
            prev_defense_positions = defense_positions
        else:
            dci_value, dis_value = 0.5, 0.5

    if SHOW_DCI_DIS_GAUGE:
        gauge_artists.extend(draw_dci_dis_gauge(ax, dci_value, dis_value, colors))

    if SHOW_NODE_STRESS and 'node_stress' in actual_positions.columns:
        for _, player in actual_positions.iterrows():
            if pd.notna(player.get('node_stress', np.nan)):
                stress = float(player['node_stress'])

                if stress < 0.20:
                    color = '#00FF80'
                elif stress < 0.35:
                    color = '#FFD700'
                else:
                    color = '#FF4500'

                radius = 2.5 + (stress * 4.0)
                alpha = 0.3 + (stress * 0.4)

                circle = Circle((player['x'], player['y']), radius,
                                color=color, alpha=alpha,
                                zorder=1, linewidth=3, edgecolor='white')
                ax.add_patch(circle)
                stress_circles.append(circle)

    if SHOW_POSITIONS or SHOW_PLAYER_NAMES or SHOW_PLAYER_NUMBERS:
        for _, player in actual_positions.iterrows():
            if SHOW_POSITIONS and 'player_position' in actual_positions.columns and pd.notna(player.get('player_position', np.nan)):
                label_text = str(player['player_position'])
                fontsize = 9
            elif SHOW_PLAYER_NAMES and int(player['nfl_id']) in PLAYER_NAMES:
                label_text = PLAYER_NAMES[int(player['nfl_id'])]
                fontsize = 9
            elif SHOW_PLAYER_NUMBERS and 'jersey_number' in actual_positions.columns and pd.notna(player.get('jersey_number', np.nan)):
                label_text = f"#{int(player['jersey_number'])}"
                fontsize = 9
            else:
                label_text = str(int(player['nfl_id']))
                fontsize = 9

            if player['player_side'] == 'Offense':
                bg_color = colors['offense_color']
                txt_color = colors['primary']
            else:
                bg_color = colors['defense_color']
                txt_color = colors['primary']

            label = ax.text(player['x'], player['y'] - 1.5, label_text,
                            ha='center', va='top', fontsize=fontsize, color=txt_color,
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor=bg_color, alpha=0.85,
                                      edgecolor='white', linewidth=1))
            player_labels.append(label)

    offense_count = len(offense_data)
    defense_count = len(defense_data)
    num_frames = play_data['frame_id'].nunique()

    stats = f"Frame: {frame_num + 1}/{num_frames}\n"
    stats += f"OFF: {offense_count} | DEF: {defense_count}\n"

    if 'node_stress' in actual_positions.columns:
        avg_stress = actual_positions['node_stress'].mean()
        if pd.notna(avg_stress):
            stats += f"Avg Stress: {avg_stress:.3f}\n"

    stats += f"\nDCI: {dci_value:.3f}\nDIS: {dis_value:.3f}"

    if SHOW_NODE_STRESS:
        stats += "\n\nSTRESS LEGEND:"
        stats += "\nğŸŸ¢ Safe (< 0.20)"
        stats += "\nğŸŸ¡ At Risk (0.20-0.35)"
        stats += "\nğŸŸ  Breaking (> 0.35)"

    stats_text.set_text(stats)

    elements = [offense_scatter, defense_scatter, stats_text]
    elements.extend(player_labels)
    elements.extend(stress_circles)
    elements.extend(gauge_artists)
    return elements

num_frames = play_data['frame_id'].nunique()
print(f"\nGenerating animation with {num_frames} frames...")

anim = FuncAnimation(fig, update, init_func=init,
                     frames=num_frames, interval=100, blit=True)

output_base = f"{TEAM_THEME}_game{GAME_ID}_play{PLAY_ID}_metrics"

if VIDEO_FORMAT in ['mp4', 'both']:
    output_path_mp4 = f'{OUTPUT_DIR}/{output_base}.mp4'
    print(f"Saving MP4: {output_path_mp4} ...")
    writer_mp4 = FFMpegWriter(fps=FPS, bitrate=2000)
    anim.save(output_path_mp4, writer=writer_mp4)
    print("âœ“ MP4 saved")

if VIDEO_FORMAT in ['gif', 'both']:
    output_path_gif = f'{OUTPUT_DIR}/{output_base}.gif'
    print(f"Saving GIF: {output_path_gif} ...")
    writer_gif = PillowWriter(fps=FPS)
    anim.save(output_path_gif, writer=writer_gif)
    print("âœ“ GIF saved")

plt.close(fig)

print(f"\n{'='*60}")
print("ANIMATION COMPLETE!")
print(f"{'='*60}")
print("Output directory:", OUTPUT_DIR)
print("Base name:", output_base)
print("Metric Mode:", METRIC_MODE)
print("\nDCI: Defensive Coverage Index (0=loose, 1=tight)")
print("DIS: Defensive Integrity Score (0=chaotic, 1=disciplined)")


"""
Enhanced Animation Data Generator with DCI Timeline (Kaggle-Safe)

Fixes / Improvements:
1) Kaggle pathing: reads plays_processed.parquet from /kaggle/input/** (by filename) or local dir
2) Writes output CSV to /kaggle/working/
3) Robust to notebook environment (no __file__ dependency)
4) Defensive stress computed ONLY among defensive players (case-insensitive; supports Defense/defense)
5) Keeps player_position in output (falls back to UNK)
6) Model loading is optional (kept as a stub; does not break if train_ssl / model is missing)

NOTE:
- This script expects your filtered play data to exist in a parquet named plays_processed.parquet
  with at least: game_id, play_id, frame_id, x, y, s, a, o, dir, nfl_id, player_side, player_position
"""

import os
import csv
import glob
import numpy as np
import torch

# Optional: pyarrow
try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except Exception:
    PYARROW_AVAILABLE = False

# Optional: your model code
DynamicEncoder = None
HIDDEN_DIM = 128
IN_DIM = 6

try:
    from train_ssl import DynamicEncoder, HIDDEN_DIM, IN_DIM
except Exception:
    try:
        # If you have train_ssl in a Kaggle dataset or working dir, you can add paths here if needed
        # For now, we keep it optional and non-fatal.
        print("[WARN] train_ssl not found. Running without model support (stress + CSV export only).")
    except Exception:
        pass

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
TARGET_GAME_ID = 2023090700
TARGET_PLAY_ID = 101

PARQUET_NAME = "plays_processed.parquet"
MODEL_NAME = "backbone_ssl_final.pth"  # optional, only if you have it available

OUT_DIR = "/kaggle/working"
OUT_CSV = os.path.join(OUT_DIR, f"animation_data_{TARGET_GAME_ID}_{TARGET_PLAY_ID}.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------
# HELPERS
# -----------------------------------------------------------
def find_file_in_kaggle_inputs(filename: str):
    """Find a file by name under /kaggle/input/**, else None."""
    matches = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    return matches[0] if matches else None

def safe_float(x, default=0.0):
    try:
        if x in [None, ""]:
            return default
        return float(x)
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        if x in [None, ""]:
            return default
        return int(float(x))
    except Exception:
        return default

def normalize_side(val):
    """Normalize player_side strings to 'offense'/'defense' when possible."""
    if val is None:
        return "unknown"
    s = str(val).strip().lower()
    if s in ["offense", "off", "o"]:
        return "offense"
    if s in ["defense", "def", "d"]:
        return "defense"
    return s if s else "unknown"

def calculate_defensive_stress(player_pos, defensive_teammate_pos):
    """
    Stress ONLY among defensive players.
    High stress = defender is isolated from defensive help.
    """
    if defensive_teammate_pos is None or len(defensive_teammate_pos) == 0:
        return 0.0

    dists = np.linalg.norm(defensive_teammate_pos - player_pos, axis=1)
    if len(dists) == 0:
        return 0.0

    nearest_dist = float(np.min(dists))
    # Normalize: >10 yards = high stress
    return float(np.clip(nearest_dist / 10.0, 0.0, 1.0))

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    print(f"[INFO] Generating enhanced animation data for Game {TARGET_GAME_ID} Play {TARGET_PLAY_ID}...")

    # Locate parquet
    parquet_path = find_file_in_kaggle_inputs(PARQUET_NAME)
    if parquet_path is None and os.path.exists(PARQUET_NAME):
        parquet_path = PARQUET_NAME

    if parquet_path is None:
        raise FileNotFoundError(
            f"Could not find {PARQUET_NAME}.\n"
            f"- Add the dataset containing it in Kaggle (Data panel), or\n"
            f"- Upload it to the notebook, or\n"
            f"- Place it in the working directory.\n"
            f"Searched: /kaggle/input/**/{PARQUET_NAME} and ./{PARQUET_NAME}"
        )

    print(f"[INFO] Using parquet: {parquet_path}")

    # Optional model loading (kept non-fatal)
    encoder = None
    model_path = find_file_in_kaggle_inputs(MODEL_NAME)
    if model_path is None and os.path.exists(MODEL_NAME):
        model_path = MODEL_NAME

    if DynamicEncoder is not None and model_path is not None:
        try:
            encoder = DynamicEncoder(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
            state = torch.load(model_path, map_location=DEVICE)

            # Some checkpoints store encoder.* keys
            new_state = {k.replace("encoder.", ""): v for k, v in state.items() if "encoder." in k}
            if not new_state:
                new_state = state

            encoder.load_state_dict(new_state, strict=False)
            encoder.eval()
            print(f"[INFO] Backbone model loaded from: {model_path}")
        except Exception as e:
            print(f"[WARN] Model loading skipped: {e}")
            encoder = None
    else:
        print("[INFO] Running without model (stress + CSV export only).")

    if not PYARROW_AVAILABLE:
        raise RuntimeError("pyarrow is not available in this environment; cannot read parquet.")

    # Load filtered play with parquet filters
    print("[INFO] Loading raw play data (filtered)...")
    try:
        table = pq.read_table(
            parquet_path,
            filters=[("game_id", "=", TARGET_GAME_ID), ("play_id", "=", TARGET_PLAY_ID)],
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet with filters: {e}")

    if table.num_rows == 0:
        raise ValueError(f"Play not found: game_id={TARGET_GAME_ID}, play_id={TARGET_PLAY_ID} (0 rows).")

    # Convert to column dict lists
    full_df = {c: table.column(c).to_pylist() for c in table.schema.names}

    # Ensure required columns exist
    required = ["frame_id", "x", "y", "s", "a", "o", "dir", "nfl_id", "player_side", "player_position"]
    n = len(full_df.get("frame_id", []))
    for col in required:
        if col not in full_df:
            full_df[col] = [None] * n

    # Typed vectors
    frames_list = [safe_int(v) for v in full_df.get("frame_id", [])]
    x_list      = [safe_float(v) for v in full_df.get("x", [])]
    y_list      = [safe_float(v) for v in full_df.get("y", [])]
    s_list      = [safe_float(v) for v in full_df.get("s", [])]
    a_list      = [safe_float(v) for v in full_df.get("a", [])]
    o_list      = [safe_float(v) for v in full_df.get("o", [])]
    dir_list    = [safe_float(v) for v in full_df.get("dir", [])]

    nfl_list    = full_df.get("nfl_id", [])
    side_list   = [normalize_side(v) for v in full_df.get("player_side", [])]
    pos_list    = full_df.get("player_position", [])

    if len(frames_list) == 0:
        raise ValueError("Empty frame_id after load.")

    # Group indices by frame
    frame_to_idx = {}
    for i, fid in enumerate(frames_list):
        frame_to_idx.setdefault(fid, []).append(i)

    frames = sorted(frame_to_idx.keys())
    print(f"[INFO] Processing {len(frames)} frames...")

    output_rows = []

    # Frame-by-frame
    with torch.no_grad():
        for frame in frames:
            idxs = frame_to_idx[frame]

            # Coordinates for this frame (N x 2)
            coords = np.stack(
                [[x_list[i] for i in idxs], [y_list[i] for i in idxs]],
                axis=1
            ).astype(np.float32)
            coords = np.nan_to_num(coords)

            # Identify defensive players in this frame
            defensive_local_idxs = [local_i for local_i, src_i in enumerate(idxs) if side_list[src_i] == "defense"]
            defensive_coords = coords[defensive_local_idxs] if defensive_local_idxs else np.empty((0, 2), dtype=np.float32)

            # Stress per player
            for local_i, src_i in enumerate(idxs):
                side = side_list[src_i] if src_i < len(side_list) else "unknown"

                if side == "defense" and len(defensive_coords) > 1:
                    # Remove current defender from defensive_coords
                    try:
                        def_pos_idx = defensive_local_idxs.index(local_i)
                        other_def_coords = np.delete(defensive_coords, def_pos_idx, axis=0)
                    except ValueError:
                        other_def_coords = defensive_coords
                    stress = calculate_defensive_stress(coords[local_i], other_def_coords)
                else:
                    stress = 0.0

                nfl_val = nfl_list[src_i] if src_i < len(nfl_list) else 0
                nfl_val = safe_int(nfl_val, default=0)

                position = pos_list[src_i] if (src_i < len(pos_list) and pos_list[src_i]) else "UNK"

                output_rows.append(
                    {
                        "frame_id": int(frame),
                        "nfl_id": int(nfl_val),
                        "x": float(x_list[src_i]),
                        "y": float(y_list[src_i]),
                        "s": float(s_list[src_i]),
                        "a": float(a_list[src_i]),
                        "dir": float(dir_list[src_i]),
                        "o": float(o_list[src_i]),
                        "node_stress": float(stress),
                        "player_side": side,
                        "player_position": str(position),
                    }
                )

    if not output_rows:
        raise RuntimeError("No rows produced; nothing to export.")

    # Export CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"âœ“ Animation data saved: {OUT_CSV}")
    print("  - node_stress: Defensive isolation metric (0=tight, 1=isolated)")
    print("  - player_position: Position labels for visualization")
    print("\nIn Kaggle: open the right panel -> Output -> refresh to download the CSV.")

if __name__ == "__main__":
    main()



"""
NFL Tactical Analysis - Working with Limited Player Coverage (Kaggle-Safe)
Generate strategic insights from partial player tracking data

Kaggle adjustments:
- Auto-finds the CSV under /kaggle/input/** by filename
- Writes outputs to /kaggle/working/ (writable)
- Avoids Windows paths and notebook-breaking assumptions
"""

import matplotlib
matplotlib.use("Agg")

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION (Kaggle-safe)
# ============================================================================
INPUT_CSV_NAME = "/kaggle/working/animation_data_2023090700_101.csv"
OUTPUT_DIR = "/kaggle/working/tactical_analysis"
TEAM_THEME = "seahawks"

colors = {"offense": "#69BE28", "defense": "#A5ACAF", "primary": "#002244"}
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "=" * 60)
print("NFL TACTICAL ANALYSIS")
print("Limited Player Coverage Optimization")
print("=" * 60 + "\n")

# ============================================================================
# Locate input CSV on Kaggle
# ============================================================================
candidates = glob.glob(f"/kaggle/input/**/{INPUT_CSV_NAME}", recursive=True) + glob.glob(INPUT_CSV_NAME)
if not candidates:
    raise FileNotFoundError(
        f"Could not find '{INPUT_CSV_NAME}'.\n"
        f"- Add the dataset containing it (Kaggle Data panel), or\n"
        f"- Upload it, or\n"
        f"- Set INPUT_CSV_NAME to the correct filename.\n"
        f"Searched: /kaggle/input/**/{INPUT_CSV_NAME} and ./{INPUT_CSV_NAME}"
    )
INPUT_CSV = candidates[0]
print(f"Loading data from: {INPUT_CSV}")

# ============================================================================
# Load + normalize columns
# ============================================================================
df = pd.read_csv(INPUT_CSV)

# Normalize common naming variants
rename_map = {}
if "nflId" in df.columns and "nfl_id" not in df.columns:
    rename_map["nflId"] = "nfl_id"
if "frameId" in df.columns and "frame_id" not in df.columns:
    rename_map["frameId"] = "frame_id"
if "playerSide" in df.columns and "player_side" not in df.columns:
    rename_map["playerSide"] = "player_side"
if rename_map:
    df = df.rename(columns=rename_map)

required = ["nfl_id", "frame_id", "x", "y", "player_side"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Present columns: {list(df.columns)}\n"
        f"Note: this script expects 'player_side' with values 'Offense'/'Defense'."
    )

# Clean rows
if "s" in df.columns:
    df_clean = df[~df["s"].isna() & ~df["x"].isna() & ~df["y"].isna()].copy()
else:
    df_clean = df[~df["x"].isna() & ~df["y"].isna()].copy()

# Basic sanity: keep only known sides if noisy
df_clean = df_clean[df_clean["player_side"].isin(["Offense", "Defense"])].copy()

tracked_players = df_clean["nfl_id"].nunique()
print(f"Coverage: {tracked_players}/22 players ({tracked_players/22*100:.1f}%)")
print(f"Frames: {df_clean['frame_id'].nunique()}")

# ============================================================================
# Helpers
# ============================================================================
def draw_field_background(ax, show_zones=True):
    """Draw field with tactical zones"""
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_aspect("equal")
    ax.set_facecolor("#2d5016")

    # Basic field markings
    for yard in range(10, 111, 10):
        ax.axvline(yard, color="white", alpha=0.4, linewidth=1)

    ax.axvline(10, color="white", linewidth=3)   # Goal line
    ax.axvline(110, color="white", linewidth=3)  # Goal line
    ax.axhline(0, color="white", linewidth=2)    # Sideline
    ax.axhline(53.3, color="white", linewidth=2) # Sideline

    if show_zones:
        ax.axvspan(0, 20, alpha=0.1, color="red", label="Red Zone")
        ax.axvspan(100, 120, alpha=0.1, color="red")
        ax.axvspan(40, 80, alpha=0.05, color="yellow", label="Middle Field")

        # Hash marks (approx)
        ax.axhline(23.36, color="white", alpha=0.3, linestyle="--")
        ax.axhline(29.94, color="white", alpha=0.3, linestyle="--")


def analyze_coverage_gaps(df_clean_):
    """Identify areas with limited or missing coverage"""
    print("Analyzing Coverage Gaps...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Coverage Gap Analysis", fontsize=16, fontweight="bold")

    # 1. Coverage density heatmap
    draw_field_background(ax1, show_zones=False)

    x_bins = np.linspace(0, 120, 25)
    y_bins = np.linspace(0, 53.3, 15)
    coverage_counts = np.zeros((len(y_bins) - 1, len(x_bins) - 1), dtype=float)

    # Vectorized binning (faster than nested loops)
    x_idx = np.clip(np.digitize(df_clean_["x"].values, x_bins) - 1, 0, len(x_bins) - 2)
    y_idx = np.clip(np.digitize(df_clean_["y"].values, y_bins) - 1, 0, len(y_bins) - 2)

    for xi, yi in zip(x_idx, y_idx):
        coverage_counts[yi, xi] += 1

    im1 = ax1.imshow(
        coverage_counts,
        extent=[0, 120, 0, 53.3],
        cmap="Reds",
        alpha=0.7,
        aspect="auto",
        origin="lower",
    )
    ax1.set_title("Player Position Density")
    plt.colorbar(im1, ax=ax1, label="Observation Count")

    # 2. Low coverage areas
    draw_field_background(ax2, show_zones=False)

    # consider nonzero cells only to avoid labeling empty field as â€œlow coverageâ€�
    nonzero = coverage_counts[coverage_counts > 0]
    if nonzero.size > 0:
        thresh = np.percentile(nonzero, 25)
        low_coverage_mask = (coverage_counts > 0) & (coverage_counts <= thresh)
    else:
        low_coverage_mask = np.zeros_like(coverage_counts, dtype=bool)

    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            if low_coverage_mask[j, i]:
                rect = patches.Rectangle(
                    (x_bins[i], y_bins[j]),
                    x_bins[i + 1] - x_bins[i],
                    y_bins[j + 1] - y_bins[j],
                    linewidth=1,
                    edgecolor="red",
                    facecolor="red",
                    alpha=0.3,
                )
                ax2.add_patch(rect)

    ax2.set_title("Low Coverage Areas (Bottom 25% of nonzero bins)")

    # 3/4. Side coverage heatmaps
    offense_data = df_clean_[df_clean_["player_side"] == "Offense"]
    defense_data = df_clean_[df_clean_["player_side"] == "Defense"]

    ax3.hist2d(offense_data["x"], offense_data["y"], bins=15, alpha=0.6, cmap="Greens")
    ax3.set_title(f"Offense Coverage ({offense_data['nfl_id'].nunique()} players)")
    ax3.set_xlim(0, 120)
    ax3.set_ylim(0, 53.3)

    ax4.hist2d(defense_data["x"], defense_data["y"], bins=15, alpha=0.6, cmap="Blues")
    ax4.set_title(f"Defense Coverage ({defense_data['nfl_id'].nunique()} players)")
    ax4.set_xlim(0, 120)
    ax4.set_ylim(0, 53.3)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "coverage_analysis.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    return coverage_counts


def analyze_formation_changes(df_clean_):
    """Analyze how formations change over time with limited data"""
    print("Analyzing Formation Evolution...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Formation Analysis (Partial Coverage)", fontsize=16, fontweight="bold")

    formation_metrics = []
    for frame in np.sort(df_clean_["frame_id"].unique()):
        frame_data = df_clean_[df_clean_["frame_id"] == frame]
        if frame_data.empty:
            continue

        center_x = frame_data["x"].mean()
        center_y = frame_data["y"].mean()
        spread_x = frame_data["x"].std()
        spread_y = frame_data["y"].std()

        offense = frame_data[frame_data["player_side"] == "Offense"]
        defense = frame_data[frame_data["player_side"] == "Defense"]

        metrics = {
            "frame": frame,
            "center_x": center_x,
            "center_y": center_y,
            "spread_x": spread_x,
            "spread_y": spread_y,
            "total_players": len(frame_data),
            "offense_players": len(offense),
            "defense_players": len(defense),
        }

        if not offense.empty:
            metrics["offense_center_x"] = offense["x"].mean()
            metrics["offense_center_y"] = offense["y"].mean()
            metrics["offense_spread_y"] = offense["y"].std()

        if not defense.empty:
            metrics["defense_center_x"] = defense["x"].mean()
            metrics["defense_center_y"] = defense["y"].mean()
            metrics["defense_spread_y"] = defense["y"].std()

        if "node_stress" in frame_data.columns:
            metrics["avg_stress"] = frame_data["node_stress"].mean()
            if not offense.empty:
                metrics["offense_stress"] = offense["node_stress"].mean()
            if not defense.empty:
                metrics["defense_stress"] = defense["node_stress"].mean()

        formation_metrics.append(metrics)

    formation_df = pd.DataFrame(formation_metrics)

    if not formation_df.empty:
        # 1. Center movement on field
        draw_field_background(ax1)
        ax1.plot(
            formation_df["center_x"],
            formation_df["center_y"],
            "ko-",
            linewidth=2,
            markersize=6,
            alpha=0.7,
            label="Formation Center",
        )

        for i in range(len(formation_df) - 1):
            dx = formation_df.iloc[i + 1]["center_x"] - formation_df.iloc[i]["center_x"]
            dy = formation_df.iloc[i + 1]["center_y"] - formation_df.iloc[i]["center_y"]
            ax1.arrow(
                formation_df.iloc[i]["center_x"],
                formation_df.iloc[i]["center_y"],
                dx,
                dy,
                head_width=1,
                head_length=1,
                fc="black",
                ec="black",
                alpha=0.5,
            )

        ax1.set_title("Formation Center Movement")
        ax1.legend()

        # 2. Width over time
        ax2.plot(formation_df["frame"], formation_df["spread_y"], linewidth=2, label="Overall Width")
        if "offense_spread_y" in formation_df.columns:
            ax2.plot(formation_df["frame"], formation_df["offense_spread_y"], linewidth=2, color=colors["offense"], label="Offense Width")
        if "defense_spread_y" in formation_df.columns:
            ax2.plot(formation_df["frame"], formation_df["defense_spread_y"], linewidth=2, color=colors["defense"], label="Defense Width")

        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Formation Width (yards)")
        ax2.set_title("Formation Width Evolution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Player count over time
        ax3.plot(formation_df["frame"], formation_df["total_players"], linewidth=2, label="Total Tracked")
        ax3.plot(formation_df["frame"], formation_df["offense_players"], linewidth=2, color=colors["offense"], label="Offense")
        ax3.plot(formation_df["frame"], formation_df["defense_players"], linewidth=2, color=colors["defense"], label="Defense")

        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Players Tracked")
        ax3.set_title("Player Coverage Over Time")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Stress vs formation width
        if "avg_stress" in formation_df.columns:
            sc = ax4.scatter(formation_df["spread_y"], formation_df["avg_stress"], c=formation_df["frame"], cmap="viridis", s=60, alpha=0.7)
            ax4.set_xlabel("Formation Width")
            ax4.set_ylabel("Average Stress")
            ax4.set_title("Formation Width vs Stress")
            plt.colorbar(sc, ax=ax4, label="Frame")
            ax4.grid(True, alpha=0.3)
        else:
            ax4.axis("off")
            ax4.set_title("Stress not available")

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "formation_evolution.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    return formation_df


def analyze_tactical_zones(df_clean_):
    """Analyze player activity in different tactical zones"""
    print("Analyzing Tactical Zones...")

    def get_tactical_zone(x, y):
        # Vertical zones
        if x <= 20:
            v_zone = "Own_End"
        elif x <= 40:
            v_zone = "Own_Side"
        elif x <= 60:
            v_zone = "Midfield"
        elif x <= 80:
            v_zone = "Opp_Side"
        else:
            v_zone = "Red_Zone"

        # Horizontal zones
        if y <= 17.77:
            h_zone = "Left"
        elif y <= 35.53:
            h_zone = "Center"
        else:
            h_zone = "Right"

        return f"{v_zone}_{h_zone}"

    dfz = df_clean_.copy()
    dfz["tactical_zone"] = dfz.apply(lambda r: get_tactical_zone(r["x"], r["y"]), axis=1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Tactical Zone Analysis", fontsize=16, fontweight="bold")

    # 1. Zone occupation by side
    zone_counts = dfz.groupby(["tactical_zone", "player_side"]).size().unstack(fill_value=0)

    # Ensure consistent ordering if both columns exist
    cols = [c for c in ["Defense", "Offense"] if c in zone_counts.columns]
    zone_counts[cols].plot(kind="bar", ax=ax1, color=[colors["defense"], colors["offense"]][:len(cols)])
    ax1.set_title("Zone Occupation by Side")
    ax1.set_xlabel("Tactical Zone")
    ax1.set_ylabel("Observation Count")
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(title="Side")

    # 2. Avg stress by zone
    if "node_stress" in dfz.columns:
        zone_stress = dfz.groupby(["tactical_zone", "player_side"])["node_stress"].mean().unstack()
        cols2 = [c for c in ["Defense", "Offense"] if c in zone_stress.columns]
        zone_stress[cols2].plot(kind="bar", ax=ax2, color=[colors["defense"], colors["offense"]][:len(cols2)])
        ax2.set_title("Average Stress by Zone")
        ax2.set_xlabel("Tactical Zone")
        ax2.set_ylabel("Average Node Stress")
        ax2.tick_params(axis="x", rotation=45)
        ax2.legend(title="Side")
    else:
        ax2.axis("off")
        ax2.set_title("Stress not available")

    # 3. Zone transitions
    transitions = {}
    for player in dfz["nfl_id"].unique():
        player_data = dfz[dfz["nfl_id"] == player].sort_values("frame_id")
        zones = player_data["tactical_zone"].values
        for i in range(len(zones) - 1):
            if zones[i] != zones[i + 1]:
                tr = f"{zones[i]} -> {zones[i+1]}"
                transitions[tr] = transitions.get(tr, 0) + 1

    if transitions:
        top = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]
        names = [t[0] for t in top]
        counts = [t[1] for t in top]
        ax3.barh(range(len(names)), counts)
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(names, fontsize=8)
        ax3.set_xlabel("Transition Count")
        ax3.set_title("Top Zone Transitions")
    else:
        ax3.axis("off")
        ax3.set_title("No zone transitions detected")

    # 4. Coverage completeness
    all_zones = [f"{v}_{h}" for v in ["Own_End", "Own_Side", "Midfield", "Opp_Side", "Red_Zone"] for h in ["Left", "Center", "Right"]]
    covered = set(dfz["tactical_zone"].unique())

    zone_coverage = [int((dfz["tactical_zone"] == z).sum()) if z in covered else 0 for z in all_zones]
    bars = ax4.bar(range(len(all_zones)), zone_coverage)
    ax4.set_xticks(range(len(all_zones)))
    ax4.set_xticklabels(all_zones, rotation=45, ha="right", fontsize=8)
    ax4.set_ylabel("Observation Count")
    ax4.set_title(f"Zone Coverage ({len(covered)}/{len(all_zones)} zones covered)")

    for bar, count in zip(bars, zone_coverage):
        if count == 0:
            bar.set_color("red"); bar.set_alpha(0.3)
        elif count < 5:
            bar.set_color("orange"); bar.set_alpha(0.6)
        else:
            bar.set_color("green"); bar.set_alpha(0.8)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "tactical_zones.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    return dfz


def generate_tactical_insights_summary(df_clean_, formation_df):
    """Generate summary insights for tactical analysis"""
    print("Generating Tactical Summary...")

    insights = {
        "coverage_percentage": (df_clean_["nfl_id"].nunique() / 22) * 100,
        "total_observations": len(df_clean_),
        "frames_analyzed": df_clean_["frame_id"].nunique(),
        "offense_players_tracked": df_clean_[df_clean_["player_side"] == "Offense"]["nfl_id"].nunique(),
        "defense_players_tracked": df_clean_[df_clean_["player_side"] == "Defense"]["nfl_id"].nunique(),
        "field_x_coverage": (df_clean_["x"].max() - df_clean_["x"].min()) / 120 * 100,
        "field_y_coverage": (df_clean_["y"].max() - df_clean_["y"].min()) / 53.3 * 100,
    }

    if "node_stress" in df_clean_.columns:
        insights["avg_stress"] = float(df_clean_["node_stress"].mean())
        insights["max_stress"] = float(df_clean_["node_stress"].max())
        insights["stress_variance"] = float(df_clean_["node_stress"].var())

    if formation_df is not None and not formation_df.empty:
        insights["formation_stability_x"] = float(formation_df["spread_x"].std())
        insights["formation_stability_y"] = float(formation_df["spread_y"].std())
        insights["avg_formation_width"] = float(formation_df["spread_y"].mean())

    outpath = os.path.join(OUTPUT_DIR, "tactical_insights.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("NFL TACTICAL ANALYSIS - KEY INSIGHTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("COVERAGE ANALYSIS:\n")
        f.write(f"  â€¢ Player Coverage: {insights['coverage_percentage']:.1f}% ({df_clean_['nfl_id'].nunique()}/22 players)\n")
        f.write(f"  â€¢ Offense Players Tracked: {insights['offense_players_tracked']}\n")
        f.write(f"  â€¢ Defense Players Tracked: {insights['defense_players_tracked']}\n")
        f.write(f"  â€¢ Total Observations: {insights['total_observations']:,}\n")
        f.write(f"  â€¢ Frames Analyzed: {insights['frames_analyzed']}\n\n")

        f.write("FIELD COVERAGE:\n")
        f.write(f"  â€¢ X-axis Coverage: {insights['field_x_coverage']:.1f}% of field length\n")
        f.write(f"  â€¢ Y-axis Coverage: {insights['field_y_coverage']:.1f}% of field width\n\n")

        if "avg_stress" in insights:
            f.write("STRESS ANALYSIS:\n")
            f.write(f"  â€¢ Average Node Stress: {insights['avg_stress']:.3f}\n")
            f.write(f"  â€¢ Maximum Stress Recorded: {insights['max_stress']:.3f}\n")
            f.write(f"  â€¢ Stress Variance: {insights['stress_variance']:.3f}\n\n")

        if formation_df is not None and not formation_df.empty:
            f.write("FORMATION ANALYSIS:\n")
            f.write(f"  â€¢ Average Formation Width: {insights['avg_formation_width']:.2f} yards\n")
            f.write(f"  â€¢ Formation X Stability (std of spread_x): {insights['formation_stability_x']:.2f}\n")
            f.write(f"  â€¢ Formation Y Stability (std of spread_y): {insights['formation_stability_y']:.2f}\n\n")

        f.write("RECOMMENDATIONS FOR LIMITED COVERAGE:\n")
        f.write("  â€¢ Focus on relative positioning rather than full 22-man formations\n")
        f.write("  â€¢ Use stress patterns (if available) to identify key tactical moments\n")
        f.write("  â€¢ Analyze zone transitions to infer assignments and leverage shifts\n")
        f.write("  â€¢ Use formation stability as a proxy for discipline/communication\n")
        f.write("  â€¢ Treat these outputs as partial evidence; validate with more tracking when possible\n")

    return outpath

# ============================================================================
# Run analyses
# ============================================================================
coverage_data = analyze_coverage_gaps(df_clean)
formation_data = analyze_formation_changes(df_clean)
_ = analyze_tactical_zones(df_clean)
_ = generate_tactical_insights_summary(df_clean, formation_data)

print("\n" + "=" * 60)
print("TACTICAL ANALYSIS COMPLETE!")
print("=" * 60)
print(f"Output directory: {OUTPUT_DIR}")
print("Generated files:")

tactical_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith((".png", ".txt"))])
for file in tactical_files:
    print(f"  âœ“ {file}")

print("\nTactical Insights Generated:")
print("  â€¢ Coverage gap analysis")
print("  â€¢ Formation evolution tracking")
print("  â€¢ Tactical zone occupation")
print("  â€¢ Player transition patterns")
print("  â€¢ Strategic recommendations for partial coverage")
print("\nIn Kaggle: open the right panel -> Output -> refresh to download the files.")



"""
NFL Tactical Analysis - Working with Limited Player Coverage (Kaggle-Safe)
Generate strategic insights from partial player tracking data

Kaggle adjustments:
- Auto-finds the CSV under /kaggle/input/** by filename (or local ./)
- Writes outputs to /kaggle/working/ (writable)
- Robust player_side normalization (offense/defense casing variants)
- Guards against empty/non-numeric plot inputs
"""

import matplotlib
matplotlib.use("Agg")

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION (Kaggle-safe)
# ============================================================================
INPUT_CSV_NAME = "animation_data_2023090700_101.csv"  # change if your file name differs
OUTPUT_DIR = "/kaggle/working/tactical_analysis"
TEAM_THEME = "seahawks"

colors = {"offense": "#69BE28", "defense": "#A5ACAF", "primary": "#002244"}
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "=" * 60)
print("NFL TACTICAL ANALYSIS")
print("Limited Player Coverage Optimization")
print("=" * 60 + "\n")

# ============================================================================
# Locate input CSV on Kaggle
# ============================================================================
candidates = glob.glob(f"/kaggle/input/**/{INPUT_CSV_NAME}", recursive=True) + glob.glob(INPUT_CSV_NAME)
if not candidates:
    raise FileNotFoundError(
        f"Could not find '{INPUT_CSV_NAME}'.\n"
        f"- Add the dataset containing it (Kaggle Data panel), or\n"
        f"- Upload it, or\n"
        f"- Set INPUT_CSV_NAME to the correct filename.\n"
        f"Searched: /kaggle/input/**/{INPUT_CSV_NAME} and ./{INPUT_CSV_NAME}"
    )
INPUT_CSV = candidates[0]
print(f"Loading data from: {INPUT_CSV}")

# ============================================================================
# Load + normalize columns
# ============================================================================
df = pd.read_csv(INPUT_CSV)

# Normalize common naming variants
rename_map = {}
if "nflId" in df.columns and "nfl_id" not in df.columns:
    rename_map["nflId"] = "nfl_id"
if "frameId" in df.columns and "frame_id" not in df.columns:
    rename_map["frameId"] = "frame_id"
if "playerSide" in df.columns and "player_side" not in df.columns:
    rename_map["playerSide"] = "player_side"
if rename_map:
    df = df.rename(columns=rename_map)

required = ["nfl_id", "frame_id", "x", "y", "player_side"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Present columns: {list(df.columns)}\n"
        f"Note: this script expects 'player_side' indicating offense/defense."
    )

print("Raw player_side value counts (top 20):")
print(df["player_side"].value_counts(dropna=False).head(20).to_string())
print()

# Clean rows
if "s" in df.columns:
    df_clean = df[~df["s"].isna() & ~df["x"].isna() & ~df["y"].isna()].copy()
else:
    df_clean = df[~df["x"].isna() & ~df["y"].isna()].copy()

# Normalize player_side to Title Case Offense/Defense
df_clean["player_side"] = (
    df_clean["player_side"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "offense": "Offense",
        "off": "Offense",
        "o": "Offense",
        "defense": "Defense",
        "def": "Defense",
        "d": "Defense",
    })
)

# Keep only analyzable rows
df_clean = df_clean[df_clean["player_side"].isin(["Offense", "Defense"])].copy()

tracked_players = df_clean["nfl_id"].nunique()
print(f"Coverage: {tracked_players}/22 players ({tracked_players/22*100:.1f}%)")
print(f"Frames: {df_clean['frame_id'].nunique()}")
print("player_side counts after normalization:")
print(df_clean["player_side"].value_counts(dropna=False).to_string())
print()

# ============================================================================
# Helpers
# ============================================================================
def draw_field_background(ax, show_zones=True):
    """Draw field with tactical zones"""
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_aspect("equal")
    ax.set_facecolor("#2d5016")

    # Basic field markings
    for yard in range(10, 111, 10):
        ax.axvline(yard, color="white", alpha=0.4, linewidth=1)

    ax.axvline(10, color="white", linewidth=3)   # Goal line
    ax.axvline(110, color="white", linewidth=3)  # Goal line
    ax.axhline(0, color="white", linewidth=2)    # Sideline
    ax.axhline(53.3, color="white", linewidth=2) # Sideline

    if show_zones:
        ax.axvspan(0, 20, alpha=0.1, color="red", label="Red Zone")
        ax.axvspan(100, 120, alpha=0.1, color="red")
        ax.axvspan(40, 80, alpha=0.05, color="yellow", label="Middle Field")

        # Hash marks (approx)
        ax.axhline(23.36, color="white", alpha=0.3, linestyle="--")
        ax.axhline(29.94, color="white", alpha=0.3, linestyle="--")


def analyze_coverage_gaps(df_clean_):
    """Identify areas with limited or missing coverage"""
    print("Analyzing Coverage Gaps...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Coverage Gap Analysis", fontsize=16, fontweight="bold")

    # 1. Coverage density heatmap
    draw_field_background(ax1, show_zones=False)

    x_bins = np.linspace(0, 120, 25)
    y_bins = np.linspace(0, 53.3, 15)
    coverage_counts = np.zeros((len(y_bins) - 1, len(x_bins) - 1), dtype=float)

    # Vectorized binning
    x_idx = np.clip(np.digitize(df_clean_["x"].values, x_bins) - 1, 0, len(x_bins) - 2)
    y_idx = np.clip(np.digitize(df_clean_["y"].values, y_bins) - 1, 0, len(y_bins) - 2)

    for xi, yi in zip(x_idx, y_idx):
        coverage_counts[yi, xi] += 1

    im1 = ax1.imshow(
        coverage_counts,
        extent=[0, 120, 0, 53.3],
        cmap="Reds",
        alpha=0.7,
        aspect="auto",
        origin="lower",
    )
    ax1.set_title("Player Position Density")
    plt.colorbar(im1, ax=ax1, label="Observation Count")

    # 2. Low coverage areas
    draw_field_background(ax2, show_zones=False)

    nonzero = coverage_counts[coverage_counts > 0]
    if nonzero.size > 0:
        thresh = np.percentile(nonzero, 25)
        low_coverage_mask = (coverage_counts > 0) & (coverage_counts <= thresh)
    else:
        low_coverage_mask = np.zeros_like(coverage_counts, dtype=bool)

    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            if low_coverage_mask[j, i]:
                rect = patches.Rectangle(
                    (x_bins[i], y_bins[j]),
                    x_bins[i + 1] - x_bins[i],
                    y_bins[j + 1] - y_bins[j],
                    linewidth=1,
                    edgecolor="red",
                    facecolor="red",
                    alpha=0.3,
                )
                ax2.add_patch(rect)

    ax2.set_title("Low Coverage Areas (Bottom 25% of nonzero bins)")

    # 3/4. Side coverage heatmaps
    offense_data = df_clean_[df_clean_["player_side"] == "Offense"]
    defense_data = df_clean_[df_clean_["player_side"] == "Defense"]

    ax3.hist2d(offense_data["x"], offense_data["y"], bins=15, alpha=0.6, cmap="Greens")
    ax3.set_title(f"Offense Coverage ({offense_data['nfl_id'].nunique()} players)")
    ax3.set_xlim(0, 120)
    ax3.set_ylim(0, 53.3)

    ax4.hist2d(defense_data["x"], defense_data["y"], bins=15, alpha=0.6, cmap="Blues")
    ax4.set_title(f"Defense Coverage ({defense_data['nfl_id'].nunique()} players)")
    ax4.set_xlim(0, 120)
    ax4.set_ylim(0, 53.3)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "coverage_analysis.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    return coverage_counts


def analyze_formation_changes(df_clean_):
    """Analyze how formations change over time with limited data"""
    print("Analyzing Formation Evolution...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Formation Analysis (Partial Coverage)", fontsize=16, fontweight="bold")

    formation_metrics = []
    for frame in np.sort(df_clean_["frame_id"].unique()):
        frame_data = df_clean_[df_clean_["frame_id"] == frame]
        if frame_data.empty:
            continue

        center_x = frame_data["x"].mean()
        center_y = frame_data["y"].mean()
        spread_x = frame_data["x"].std()
        spread_y = frame_data["y"].std()

        offense = frame_data[frame_data["player_side"] == "Offense"]
        defense = frame_data[frame_data["player_side"] == "Defense"]

        metrics = {
            "frame": frame,
            "center_x": center_x,
            "center_y": center_y,
            "spread_x": spread_x,
            "spread_y": spread_y,
            "total_players": len(frame_data),
            "offense_players": len(offense),
            "defense_players": len(defense),
        }

        if not offense.empty:
            metrics["offense_center_x"] = offense["x"].mean()
            metrics["offense_center_y"] = offense["y"].mean()
            metrics["offense_spread_y"] = offense["y"].std()

        if not defense.empty:
            metrics["defense_center_x"] = defense["x"].mean()
            metrics["defense_center_y"] = defense["y"].mean()
            metrics["defense_spread_y"] = defense["y"].std()

        if "node_stress" in frame_data.columns:
            metrics["avg_stress"] = frame_data["node_stress"].mean()
            if not offense.empty:
                metrics["offense_stress"] = offense["node_stress"].mean()
            if not defense.empty:
                metrics["defense_stress"] = defense["node_stress"].mean()

        formation_metrics.append(metrics)

    formation_df = pd.DataFrame(formation_metrics)

    if not formation_df.empty:
        # 1. Center movement on field
        draw_field_background(ax1)
        ax1.plot(
            formation_df["center_x"],
            formation_df["center_y"],
            "ko-",
            linewidth=2,
            markersize=6,
            alpha=0.7,
            label="Formation Center",
        )

        for i in range(len(formation_df) - 1):
            dx = formation_df.iloc[i + 1]["center_x"] - formation_df.iloc[i]["center_x"]
            dy = formation_df.iloc[i + 1]["center_y"] - formation_df.iloc[i]["center_y"]
            ax1.arrow(
                formation_df.iloc[i]["center_x"],
                formation_df.iloc[i]["center_y"],
                dx,
                dy,
                head_width=1,
                head_length=1,
                fc="black",
                ec="black",
                alpha=0.5,
            )

        ax1.set_title("Formation Center Movement")
        ax1.legend()

        # 2. Width over time
        ax2.plot(formation_df["frame"], formation_df["spread_y"], linewidth=2, label="Overall Width")
        if "offense_spread_y" in formation_df.columns:
            ax2.plot(
                formation_df["frame"],
                formation_df["offense_spread_y"],
                linewidth=2,
                color=colors["offense"],
                label="Offense Width",
            )
        if "defense_spread_y" in formation_df.columns:
            ax2.plot(
                formation_df["frame"],
                formation_df["defense_spread_y"],
                linewidth=2,
                color=colors["defense"],
                label="Defense Width",
            )

        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Formation Width (yards)")
        ax2.set_title("Formation Width Evolution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Player count over time
        ax3.plot(formation_df["frame"], formation_df["total_players"], linewidth=2, label="Total Tracked")
        ax3.plot(formation_df["frame"], formation_df["offense_players"], linewidth=2, color=colors["offense"], label="Offense")
        ax3.plot(formation_df["frame"], formation_df["defense_players"], linewidth=2, color=colors["defense"], label="Defense")

        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Players Tracked")
        ax3.set_title("Player Coverage Over Time")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Stress vs formation width
        if "avg_stress" in formation_df.columns:
            sc = ax4.scatter(
                formation_df["spread_y"],
                formation_df["avg_stress"],
                c=formation_df["frame"],
                cmap="viridis",
                s=60,
                alpha=0.7,
            )
            ax4.set_xlabel("Formation Width")
            ax4.set_ylabel("Average Stress")
            ax4.set_title("Formation Width vs Stress")
            plt.colorbar(sc, ax=ax4, label="Frame")
            ax4.grid(True, alpha=0.3)
        else:
            ax4.axis("off")
            ax4.set_title("Stress not available")
    else:
        for ax in (ax1, ax2, ax3, ax4):
            ax.axis("off")
        fig.suptitle("Formation Analysis (no data after cleaning)", fontsize=16, fontweight="bold")

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "formation_evolution.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    return formation_df


def analyze_tactical_zones(df_clean_):
    """Analyze player activity in different tactical zones"""
    print("Analyzing Tactical Zones...")

    def get_tactical_zone(x, y):
        # Vertical zones
        if x <= 20:
            v_zone = "Own_End"
        elif x <= 40:
            v_zone = "Own_Side"
        elif x <= 60:
            v_zone = "Midfield"
        elif x <= 80:
            v_zone = "Opp_Side"
        else:
            v_zone = "Red_Zone"

        # Horizontal zones
        if y <= 17.77:
            h_zone = "Left"
        elif y <= 35.53:
            h_zone = "Center"
        else:
            h_zone = "Right"

        return f"{v_zone}_{h_zone}"

    dfz = df_clean_.copy()
    dfz["tactical_zone"] = dfz.apply(lambda r: get_tactical_zone(r["x"], r["y"]), axis=1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Tactical Zone Analysis", fontsize=16, fontweight="bold")

    # 1. Zone occupation by side
    zone_counts = dfz.groupby(["tactical_zone", "player_side"]).size().unstack(fill_value=0)
    zone_counts = zone_counts.apply(pd.to_numeric, errors="coerce").fillna(0)

    cols = [c for c in ["Defense", "Offense"] if c in zone_counts.columns]
    if len(cols) == 0 or zone_counts[cols].to_numpy().sum() == 0:
        ax1.axis("off")
        ax1.set_title("Zone Occupation by Side (no valid Offense/Defense data to plot)")
    else:
        zone_counts[cols].plot(
            kind="bar",
            ax=ax1,
            color=[colors["defense"], colors["offense"]][:len(cols)],
        )
        ax1.set_title("Zone Occupation by Side")
        ax1.set_xlabel("Tactical Zone")
        ax1.set_ylabel("Observation Count")
        ax1.tick_params(axis="x", rotation=45)
        ax1.legend(title="Side")

    # 2. Avg stress by zone
    if "node_stress" in dfz.columns:
        zone_stress = dfz.groupby(["tactical_zone", "player_side"])["node_stress"].mean().unstack()
        zone_stress = zone_stress.apply(pd.to_numeric, errors="coerce").fillna(0)
        cols2 = [c for c in ["Defense", "Offense"] if c in zone_stress.columns]

        if len(cols2) == 0 or zone_stress[cols2].to_numpy().sum() == 0:
            ax2.axis("off")
            ax2.set_title("Average Stress by Zone (no valid Offense/Defense data)")
        else:
            zone_stress[cols2].plot(
                kind="bar",
                ax=ax2,
                color=[colors["defense"], colors["offense"]][:len(cols2)],
            )
            ax2.set_title("Average Stress by Zone")
            ax2.set_xlabel("Tactical Zone")
            ax2.set_ylabel("Average Node Stress")
            ax2.tick_params(axis="x", rotation=45)
            ax2.legend(title="Side")
    else:
        ax2.axis("off")
        ax2.set_title("Stress not available")

    # 3. Zone transitions
    transitions = {}
    for player in dfz["nfl_id"].unique():
        player_data = dfz[dfz["nfl_id"] == player].sort_values("frame_id")
        zones = player_data["tactical_zone"].values
        for i in range(len(zones) - 1):
            if zones[i] != zones[i + 1]:
                tr = f"{zones[i]} -> {zones[i+1]}"
                transitions[tr] = transitions.get(tr, 0) + 1

    if transitions:
        top = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]
        names = [t[0] for t in top]
        counts = [t[1] for t in top]
        ax3.barh(range(len(names)), counts)
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(names, fontsize=8)
        ax3.set_xlabel("Transition Count")
        ax3.set_title("Top Zone Transitions")
    else:
        ax3.axis("off")
        ax3.set_title("No zone transitions detected")

    # 4. Coverage completeness
    all_zones = [f"{v}_{h}" for v in ["Own_End", "Own_Side", "Midfield", "Opp_Side", "Red_Zone"] for h in ["Left", "Center", "Right"]]
    covered = set(dfz["tactical_zone"].unique())

    zone_coverage = [int((dfz["tactical_zone"] == z).sum()) if z in covered else 0 for z in all_zones]
    bars = ax4.bar(range(len(all_zones)), zone_coverage)
    ax4.set_xticks(range(len(all_zones)))
    ax4.set_xticklabels(all_zones, rotation=45, ha="right", fontsize=8)
    ax4.set_ylabel("Observation Count")
    ax4.set_title(f"Zone Coverage ({len(covered)}/{len(all_zones)} zones covered)")

    for bar, count in zip(bars, zone_coverage):
        if count == 0:
            bar.set_color("red"); bar.set_alpha(0.3)
        elif count < 5:
            bar.set_color("orange"); bar.set_alpha(0.6)
        else:
            bar.set_color("green"); bar.set_alpha(0.8)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "tactical_zones.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    return dfz


def generate_tactical_insights_summary(df_clean_, formation_df):
    """Generate summary insights for tactical analysis"""
    print("Generating Tactical Summary...")

    insights = {
        "coverage_percentage": (df_clean_["nfl_id"].nunique() / 22) * 100,
        "total_observations": len(df_clean_),
        "frames_analyzed": df_clean_["frame_id"].nunique(),
        "offense_players_tracked": df_clean_[df_clean_["player_side"] == "Offense"]["nfl_id"].nunique(),
        "defense_players_tracked": df_clean_[df_clean_["player_side"] == "Defense"]["nfl_id"].nunique(),
        "field_x_coverage": (df_clean_["x"].max() - df_clean_["x"].min()) / 120 * 100,
        "field_y_coverage": (df_clean_["y"].max() - df_clean_["y"].min()) / 53.3 * 100,
    }

    if "node_stress" in df_clean_.columns:
        insights["avg_stress"] = float(df_clean_["node_stress"].mean())
        insights["max_stress"] = float(df_clean_["node_stress"].max())
        insights["stress_variance"] = float(df_clean_["node_stress"].var())

    if formation_df is not None and not formation_df.empty:
        insights["formation_stability_x"] = float(formation_df["spread_x"].std())
        insights["formation_stability_y"] = float(formation_df["spread_y"].std())
        insights["avg_formation_width"] = float(formation_df["spread_y"].mean())

    outpath = os.path.join(OUTPUT_DIR, "tactical_insights.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("NFL TACTICAL ANALYSIS - KEY INSIGHTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("COVERAGE ANALYSIS:\n")
        f.write(f"  â€¢ Player Coverage: {insights['coverage_percentage']:.1f}% ({df_clean_['nfl_id'].nunique()}/22 players)\n")
        f.write(f"  â€¢ Offense Players Tracked: {insights['offense_players_tracked']}\n")
        f.write(f"  â€¢ Defense Players Tracked: {insights['defense_players_tracked']}\n")
        f.write(f"  â€¢ Total Observations: {insights['total_observations']:,}\n")
        f.write(f"  â€¢ Frames Analyzed: {insights['frames_analyzed']}\n\n")

        f.write("FIELD COVERAGE:\n")
        f.write(f"  â€¢ X-axis Coverage: {insights['field_x_coverage']:.1f}% of field length\n")
        f.write(f"  â€¢ Y-axis Coverage: {insights['field_y_coverage']:.1f}% of field width\n\n")

        if "avg_stress" in insights:
            f.write("STRESS ANALYSIS:\n")
            f.write(f"  â€¢ Average Node Stress: {insights['avg_stress']:.3f}\n")
            f.write(f"  â€¢ Maximum Stress Recorded: {insights['max_stress']:.3f}\n")
            f.write(f"  â€¢ Stress Variance: {insights['stress_variance']:.3f}\n\n")

        if formation_df is not None and not formation_df.empty:
            f.write("FORMATION ANALYSIS:\n")
            f.write(f"  â€¢ Average Formation Width: {insights['avg_formation_width']:.2f} yards\n")
            f.write(f"  â€¢ Formation X Stability (std of spread_x): {insights['formation_stability_x']:.2f}\n")
            f.write(f"  â€¢ Formation Y Stability (std of spread_y): {insights['formation_stability_y']:.2f}\n\n")

        f.write("RECOMMENDATIONS FOR LIMITED COVERAGE:\n")
        f.write("  â€¢ Focus on relative positioning rather than full 22-man formations\n")
        f.write("  â€¢ Use stress patterns (if available) to identify key tactical moments\n")
        f.write("  â€¢ Analyze zone transitions to infer assignments and leverage shifts\n")
        f.write("  â€¢ Use formation stability as a proxy for discipline/communication\n")
        f.write("  â€¢ Treat these outputs as partial evidence; validate with more tracking when possible\n")

    return outpath


# ============================================================================
# Run analyses
# ============================================================================
if df_clean.empty:
    raise ValueError(
        "df_clean is empty after cleaning/normalization.\n"
        "This usually means player_side values do not map to offense/defense.\n"
        "Check the printed raw value_counts above and adjust the mapping."
    )

coverage_data = analyze_coverage_gaps(df_clean)
formation_data = analyze_formation_changes(df_clean)
_ = analyze_tactical_zones(df_clean)
_ = generate_tactical_insights_summary(df_clean, formation_data)

print("\n" + "=" * 60)
print("TACTICAL ANALYSIS COMPLETE!")
print("=" * 60)
print(f"Output directory: {OUTPUT_DIR}")
print("Generated files:")

tactical_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith((".png", ".txt"))])
for file in tactical_files:
    print(f"  âœ“ {file}")

print("\nTactical Insights Generated:")
print("  â€¢ Coverage gap analysis")
print("  â€¢ Formation evolution tracking")
print("  â€¢ Tactical zone occupation")
print("  â€¢ Player transition patterns")
print("  â€¢ Strategic recommendations for partial coverage")
print("\nIn Kaggle: open the right panel -> Output -> refresh to download the files.")



"""
NFL Big Data Bowl - Defensive Coverage Animation (Kaggle-Safe)
Visualizes player tracking data during pass plays with node stress indicators

Kaggle adjustments:
- Auto-finds the CSV under /kaggle/input/** by filename (or local ./)
- Robust column renaming (nflId/frameId/playerSide -> nfl_id/frame_id/player_side)
- Robust player_side normalization (offense/defense casing variants)
- Writes outputs to /kaggle/working/ (writable)
- Avoids hard-coded Windows paths and backend issues
"""

import os
import glob
import warnings

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION (Kaggle-safe)
# ============================================================================
INPUT_CSV_NAME = "animation_data_2023090700_101.csv"  # change if needed
OUTPUT_DIR = "/kaggle/working"
OUTPUT_GIF = os.path.join(OUTPUT_DIR, "nfl_tracking_animation.gif")
OUTPUT_MP4 = os.path.join(OUTPUT_DIR, "nfl_tracking_animation.mp4")

FIELD_LENGTH = 120  # yards
FIELD_WIDTH = 53.3  # yards
FPS = 10            # frames per second for animation

# Color scheme
OFFENSE_COLOR = "#0066ff"
DEFENSE_COLOR = "#ff3333"
PROJECTION_COLOR = "#ffaa00"

# ============================================================================
# Locate input CSV on Kaggle
# ============================================================================
candidates = glob.glob(f"/kaggle/input/**/{INPUT_CSV_NAME}", recursive=True) + glob.glob(INPUT_CSV_NAME)
if not candidates:
    raise FileNotFoundError(
        f"Could not find '{INPUT_CSV_NAME}'.\n"
        f"- Add the dataset containing it (Kaggle Data panel), or\n"
        f"- Upload it, or\n"
        f"- Set INPUT_CSV_NAME to the correct filename.\n"
        f"Searched: /kaggle/input/**/{INPUT_CSV_NAME} and ./{INPUT_CSV_NAME}"
    )

INPUT_CSV = candidates[0]
print(f"Loading data from: {INPUT_CSV}")

# ============================================================================
# Load + normalize columns
# ============================================================================
df = pd.read_csv(INPUT_CSV)

rename_map = {}
if "nflId" in df.columns and "nfl_id" not in df.columns:
    rename_map["nflId"] = "nfl_id"
if "frameId" in df.columns and "frame_id" not in df.columns:
    rename_map["frameId"] = "frame_id"
if "playerSide" in df.columns and "player_side" not in df.columns:
    rename_map["playerSide"] = "player_side"
if rename_map:
    df = df.rename(columns=rename_map)

required = ["nfl_id", "frame_id", "x", "y", "player_side"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Present columns: {list(df.columns)}"
    )

# node_stress is optional but recommended
if "node_stress" not in df.columns:
    df["node_stress"] = 0.0

# s is optional; used to split actual vs projected
if "s" not in df.columns:
    df["s"] = np.nan

# Normalize player_side to Offense/Defense
df["player_side"] = (
    df["player_side"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "offense": "Offense",
        "off": "Offense",
        "o": "Offense",
        "defense": "Defense",
        "def": "Defense",
        "d": "Defense",
    })
)

# Keep only analyzable rows (Offense/Defense)
df = df[df["player_side"].isin(["Offense", "Defense"])].copy()

# Ensure numeric types where needed
for col in ["x", "y", "s", "node_stress"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce").astype("Int64")
df["nfl_id"] = pd.to_numeric(df["nfl_id"], errors="coerce").astype("Int64")

df = df.dropna(subset=["frame_id", "nfl_id", "x", "y"]).copy()

# ============================================================================
# Separate actual positions from projected positions
# Players with speed data are actual positions, NaN speed are projections
# ============================================================================
df_actual = df[df["s"].notna()].copy()
df_projected = df[df["s"].isna()].copy()

frames = sorted(df_actual["frame_id"].dropna().unique().tolist())
if not frames:
    raise ValueError(
        "No frames found in df_actual (rows where s is not NaN).\n"
        "If your file contains only actual positions without 's', set df['s'] to 0 for all rows, "
        "or remove the actual/projected split."
    )

print(f"Creating animation with {len(frames)} frames")
print(f"Tracking {df_actual['nfl_id'].nunique()} players")

# ============================================================================
# Figure setup
# ============================================================================
fig = plt.figure(figsize=(16, 10), facecolor="#0a0e1a")
ax = fig.add_subplot(111, facecolor="#2d5016")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]  # DejaVu exists on Kaggle

def draw_field(ax_):
    """Draw an NFL field with yard lines"""
    ax_.set_xlim(0, FIELD_LENGTH)
    ax_.set_ylim(0, FIELD_WIDTH)
    ax_.set_aspect("equal")

    # Field background with stripe pattern
    for i in range(0, int(FIELD_LENGTH), 10):
        rect = patches.Rectangle(
            (i, 0), 5, FIELD_WIDTH,
            linewidth=0, edgecolor="none",
            facecolor="#3a6b1e", alpha=0.3
        )
        ax_.add_patch(rect)

    # Yard lines
    for yard in range(10, 110, 10):
        ax_.axvline(x=yard, color="white", linewidth=2, alpha=0.6, zorder=1)

    # Goal lines
    ax_.axvline(x=10, color="white", linewidth=3, zorder=1)
    ax_.axvline(x=110, color="white", linewidth=3, zorder=1)

    # Sidelines
    ax_.axhline(y=0, color="white", linewidth=3, zorder=1)
    ax_.axhline(y=FIELD_WIDTH, color="white", linewidth=3, zorder=1)

    # Hash marks (simplified)
    for yard in range(10, 110, 1):
        ax_.plot([yard, yard], [FIELD_WIDTH / 2 - 0.3, FIELD_WIDTH / 2 + 0.3],
                 color="white", linewidth=1, alpha=0.5, zorder=1)

    ax_.set_xticks([])
    ax_.set_yticks([])
    ax_.set_xlabel("")
    ax_.set_ylabel("")

draw_field(ax)

def get_stress_color(stress_value):
    """Map node stress to color (low=green, high=red)"""
    if pd.isna(stress_value):
        return "#888888"
    norm_stress = float(np.clip(stress_value, 0.0, 1.0))
    r = int(norm_stress * 255)
    g = int((1.0 - norm_stress) * 200)
    b = 50
    return f"#{r:02x}{g:02x}{b:02x}"

# Containers for artists
player_dots = {}
player_labels = {}
projection_lines = {}
stress_circles = {}

# Title and info text
ax.text(
    0.5, 1.08, "NFL Big Data Bowl - Defensive Coverage Analysis",
    transform=ax.transAxes, ha="center", va="top",
    fontsize=20, fontweight="bold", color="white"
)

frame_text = ax.text(
    0.02, 1.03, "", transform=ax.transAxes,
    ha="left", va="top", fontsize=14,
    color="#00d4ff", fontweight="bold"
)

stats_text = ax.text(
    0.98, 1.03, "", transform=ax.transAxes,
    ha="right", va="top", fontsize=11,
    color="#a0aec0"
)

# Legend
legend_x = 0.02
legend_y = 0.05
ax.text(legend_x, legend_y + 0.12, "LEGEND", transform=ax.transAxes,
        fontsize=10, fontweight="bold", color="white")

ax.plot([0.02], [0.09], "o", color=OFFENSE_COLOR, markersize=12,
        transform=ax.transAxes, markeredgecolor="white", markeredgewidth=1.5)
ax.text(legend_x + 0.04, legend_y + 0.08, "Offense", transform=ax.transAxes,
        fontsize=9, color="white", va="center")

ax.plot([0.02], [0.06], "o", color=DEFENSE_COLOR, markersize=12,
        transform=ax.transAxes, markeredgecolor="white", markeredgewidth=1.5)
ax.text(legend_x + 0.04, legend_y + 0.05, "Defense", transform=ax.transAxes,
        fontsize=9, color="white", va="center")

ax.plot([0.02], [0.03], "o", color=PROJECTION_COLOR, markersize=8,
        transform=ax.transAxes, marker="x", markeredgewidth=2)
ax.text(legend_x + 0.04, legend_y + 0.02, "Projection", transform=ax.transAxes,
        fontsize=9, color="white", va="center")

# Node stress legend
stress_legend_x = 0.88
ax.text(stress_legend_x, legend_y + 0.12, "NODE STRESS", transform=ax.transAxes,
        fontsize=10, fontweight="bold", color="white")

stress_positions = [0.09, 0.06, 0.03]
stress_colors = ["#00c850", "#ffaa00", "#ff3333"]
stress_labels = ["Low", "Med", "High"]

for pos, col, label in zip(stress_positions, stress_colors, stress_labels):
    circ = Circle((stress_legend_x, legend_y + pos), 0.008, transform=ax.transAxes, color=col, alpha=0.6)
    ax.add_patch(circ)
    ax.text(stress_legend_x + 0.025, legend_y + pos, label,
            transform=ax.transAxes, fontsize=9, color="white", va="center")

def init():
    return []

def animate(frame_idx):
    current_frame = frames[frame_idx]

    frame_data = df_actual[df_actual["frame_id"] == current_frame]
    frame_proj = df_projected[df_projected["frame_id"] == current_frame]

    frame_text.set_text(f"Frame: {frame_idx + 1:02d}/{len(frames):02d}")

    offense_count = int((frame_data["player_side"] == "Offense").sum())
    defense_count = int((frame_data["player_side"] == "Defense").sum())
    avg_stress = float(pd.to_numeric(frame_data["node_stress"], errors="coerce").mean())

    if np.isnan(avg_stress):
        avg_stress = 0.0

    stats_text.set_text(
        f"Offense: {offense_count} | Defense: {defense_count} | Avg Stress: {avg_stress:.3f}"
    )

    # Clear previous artists
    for pid in list(player_dots.keys()):
        if player_dots.get(pid) is not None:
            player_dots[pid].remove()
        if stress_circles.get(pid) is not None:
            stress_circles[pid].remove()
        if player_labels.get(pid) is not None:
            player_labels[pid].remove()
        if projection_lines.get(pid) is not None:
            for artist in projection_lines[pid]:
                artist.remove()

    player_dots.clear()
    stress_circles.clear()
    player_labels.clear()
    projection_lines.clear()

    # Draw players and projections
    for _, player in frame_data.iterrows():
        pid = int(player["nfl_id"])
        x, y = float(player["x"]), float(player["y"])
        side = player["player_side"]
        stress = player.get("node_stress", 0.0)

        color = OFFENSE_COLOR if side == "Offense" else DEFENSE_COLOR

        # Stress halo
        stress_color = get_stress_color(stress)
        stress_val = 0.0 if pd.isna(stress) else float(stress)
        stress_size = 300 + (np.clip(stress_val, 0.0, 1.0) * 400.0)
        halo = ax.scatter([x], [y], s=stress_size, c=stress_color, alpha=0.3, zorder=2, edgecolors="none")
        stress_circles[pid] = halo

        # Player dot
        dot = ax.scatter([x], [y], s=200, c=color, edgecolors="white", linewidths=2, zorder=5, alpha=0.95)
        player_dots[pid] = dot

        # Label: last 3 digits of nfl_id
        lab = ax.text(
            x, y - 1.5, str(pid)[-3:],
            ha="center", va="top", fontsize=7, color="white", fontweight="bold",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8, edgecolor="none")
        )
        player_labels[pid] = lab

        # Projection (if exists)
        proj = frame_proj[frame_proj["nfl_id"] == pid]
        if not proj.empty:
            proj_x = float(proj.iloc[0]["x"])
            proj_y = float(proj.iloc[0]["y"])

            line1 = ax.plot([x, proj_x], [y, proj_y],
                            color=PROJECTION_COLOR, linewidth=2,
                            linestyle="--", alpha=0.6, zorder=3)[0]
            line2 = ax.scatter([proj_x], [proj_y], s=100,
                               marker="x", c=PROJECTION_COLOR,
                               linewidths=3, zorder=4, alpha=0.8)
            projection_lines[pid] = [line1, line2]

    elements = (
        list(player_dots.values())
        + list(stress_circles.values())
        + list(player_labels.values())
        + [item for sublist in projection_lines.values() for item in sublist]
    )
    return elements

# ============================================================================
# Create + save animation
# ============================================================================
print("Generating animation...")
anim = FuncAnimation(
    fig, animate, init_func=init,
    frames=len(frames), interval=1000 / FPS,
    blit=False, repeat=True
)

print(f"Saving GIF to {OUTPUT_GIF}...")
writer = PillowWriter(fps=FPS)
anim.save(OUTPUT_GIF, writer=writer, dpi=100)
print("GIF saved successfully!")

# MP4 export (optional; ffmpeg availability varies by Kaggle image)
try:
    from matplotlib.animation import FFMpegWriter
    print(f"Saving MP4 to {OUTPUT_MP4}...")
    writer_mp4 = FFMpegWriter(fps=FPS, bitrate=2000)
    anim.save(OUTPUT_MP4, writer=writer_mp4, dpi=150)
    print("MP4 saved successfully!")
except Exception as e:
    print(f"FFMpeg not available (skipping MP4 export). Reason: {e}")

plt.close()
print("Done!")



"""
NFL Big Data Bowl - Ravens Theme Animation with DCI/DIS Gauge (Kaggle-Safe)
Uses: /kaggle/input/input-and-results/ravens_play_1531_data.csv

What this version fixes/does:
- Reads the CSV from the exact Kaggle path you provided
- Normalizes column names (nflId/frameId/playerSide -> nfl_id/frame_id/player_side)
- Normalizes player_side values (Defense/Offense, defense/offense, etc.)
- Handles presence/absence of: s, node_stress, player_position, jersey_number
- Uses Ravens branding (purple/gold) + darker broadcast-style background
- Writes outputs to /kaggle/working/team_animations (writable)
"""

import matplotlib
matplotlib.use("Agg")

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Circle, Rectangle

warnings.filterwarnings("ignore")

# ============================================================================
# TEAM COLOR SCHEMES
# ============================================================================
TEAM_COLORS = {
    "ravens": {
        "name": "Baltimore Ravens",
        "primary": "#241773",     # Ravens Purple
        "secondary": "#9E7C0C",   # Ravens Gold
        "accent": "#C60C30",      # subtle accent (optional)
        "background": "#0b061a",  # deep broadcast purple/black
        "field": "#214a21",       # slightly darker green
        "offense_color": "#9E7C0C",
        "defense_color": "#E6E6E6",
        "field_stripe": "#173517",
    },
    "default": {
        "name": "NFL Default",
        "primary": "#013369",
        "secondary": "#D50A0A",
        "accent": "#00d4ff",
        "background": "#0a0e1a",
        "field": "#2d5016",
        "offense_color": "#1E90FF",
        "defense_color": "#DC143C",
        "field_stripe": "#1d3010",
    },
}

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_CSV = "/kaggle/input/input-and-results/ravens_play_1531_data.csv"
OUTPUT_DIR = "/kaggle/working/team_animations"
TEAM_THEME = "ravens"

# Animation settings
FPS = 10
VIDEO_FORMAT = "gif"          # "mp4", "gif", "both"
SHOW_TEAM_LOGO_AREA = True
SHOW_NODE_STRESS = True
SHOW_DCI_DIS_GAUGE = True

# Labels
SHOW_PLAYER_NAMES = True
SHOW_PLAYER_NUMBERS = False
SHOW_POSITIONS = False        # set True if you want player_position labels

# Metric mode
METRIC_MODE = "geometric"     # "precomputed" or "geometric"
DCI_COLUMN = "dci_score"
DIS_COLUMN = "dis_score"

# Output naming
GAME_ID = "2023122502"
PLAY_ID = "1531"

# ============================================================================
# PLAYER NAME LOOKUPS (optional/partial)
# ============================================================================
RAVENS_49ERS_PLAYER_NAMES = {
    54727: "Lamar Jackson",
    44959: "Isaiah Likely",
    44820: "Gus Edwards",
    47819: "Zay Flowers",
    42419: "Nelson Agholor",
    52433: "Rashod Bateman",
    46077: "Fred Warner",
    52436: "Dre Greenlaw",
    44828: "Charvarius Ward",
    53533: "Deommodore Lenoir",
    44854: "Ji'Ayir Brown",
    52627: "Tashaun Gipson Sr.",
    41269: "Nick Bosa",
}
PLAYER_NAMES = RAVENS_49ERS_PLAYER_NAMES

# ============================================================================
# LOAD + NORMALIZE
# ============================================================================
colors = TEAM_COLORS.get(TEAM_THEME, TEAM_COLORS["default"])
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "=" * 60)
print(f"NFL ANIMATION - {colors['name']} Theme")
print("=" * 60 + "\n")
print(f"Loading: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
print(f"Rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")

# Normalize common naming variants
rename_map = {}
if "nflId" in df.columns and "nfl_id" not in df.columns:
    rename_map["nflId"] = "nfl_id"
if "frameId" in df.columns and "frame_id" not in df.columns:
    rename_map["frameId"] = "frame_id"
if "playerSide" in df.columns and "player_side" not in df.columns:
    rename_map["playerSide"] = "player_side"
if rename_map:
    df = df.rename(columns=rename_map)

required = ["nfl_id", "frame_id", "x", "y", "player_side"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}\nPresent: {list(df.columns)}")

# Ensure optional cols exist
if "s" not in df.columns:
    df["s"] = np.nan
if "node_stress" not in df.columns:
    df["node_stress"] = np.nan
if "player_position" not in df.columns:
    df["player_position"] = np.nan
# jersey_number is optional; leave as-is if missing

# Coerce numeric
for col in ["x", "y", "s", "node_stress"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce").astype("Int64")
df["nfl_id"] = pd.to_numeric(df["nfl_id"], errors="coerce").astype("Int64")

# Normalize side values
df["player_side"] = (
    df["player_side"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "offense": "Offense", "off": "Offense", "o": "Offense",
        "defense": "Defense", "def": "Defense", "d": "Defense",
    })
)
df = df[df["player_side"].isin(["Offense", "Defense"])].copy()

# Drop unusable rows
df = df.dropna(subset=["frame_id", "nfl_id", "x", "y"]).copy()

# New format handling: no game_id/play_id expected
print("Treating file as a single play (new animation format).")
play_data = df.copy()
play_data["is_projection"] = play_data["s"].isna()

has_dci = DCI_COLUMN in play_data.columns
has_dis = DIS_COLUMN in play_data.columns
if METRIC_MODE == "precomputed" and not (has_dci and has_dis):
    print("\n[WARN] METRIC_MODE='precomputed' but DCI/DIS columns not found. Switching to geometric.")
    METRIC_MODE = "geometric"

frames_sorted = sorted(play_data["frame_id"].unique().tolist())
num_frames = len(frames_sorted)

print(f"\nGame ID: {GAME_ID}")
print(f"Play ID: {PLAY_ID}")
print(f"Players: {play_data['nfl_id'].nunique()}")
print(f"Frames: {num_frames}")
print(f"Metric Mode: {METRIC_MODE.upper()}")

# ============================================================================
# METRICS
# ============================================================================
def calculate_geometric_dci(defense_positions: np.ndarray) -> float:
    """Geometric proxy: inverse of avg nearest-neighbor spacing among defenders."""
    if defense_positions is None or len(defense_positions) < 2:
        return 0.5

    nn = []
    for i, pos in enumerate(defense_positions):
        other = np.delete(defense_positions, i, axis=0)
        if len(other):
            d = np.linalg.norm(other - pos, axis=1)
            nn.append(np.min(d))
    if not nn:
        return 0.5

    avg_spacing = float(np.mean(nn))
    return float(np.clip(1.0 - (avg_spacing - 3.0) / 12.0, 0.0, 1.0))


def calculate_geometric_dis(defense_positions: np.ndarray, prev_defense_positions=None) -> float:
    """Geometric proxy: inverse coefficient of variation of defender pairwise distances."""
    if defense_positions is None or len(defense_positions) < 3:
        return 0.5

    dists = []
    for i in range(len(defense_positions)):
        for j in range(i + 1, len(defense_positions)):
            dists.append(np.linalg.norm(defense_positions[i] - defense_positions[j]))
    if not dists:
        return 0.5

    dists = np.array(dists, dtype=float)
    mean = float(np.mean(dists))
    std = float(np.std(dists))
    if mean <= 0:
        return 0.5

    cv = std / mean
    return float(np.clip(1.0 - cv, 0.0, 1.0))

# ============================================================================
# DRAWING
# ============================================================================
def draw_field(ax_, colors_):
    """Team-themed field with stronger Ravens look (purple/gold accents)."""
    ax_.set_xlim(0, 120)
    ax_.set_ylim(0, 53.3)
    ax_.set_aspect("equal")
    ax_.set_xticks([])
    ax_.set_yticks([])

    # Stripes
    for yard in range(0, 120, 10):
        ax_.add_patch(
            patches.Rectangle(
                (yard, 0), 5, 53.3,
                linewidth=0, edgecolor="none",
                facecolor=colors_["field_stripe"], alpha=0.18, zorder=0
            )
        )

    # Yard lines (gold-ish minor, white major)
    for yard in range(10, 111, 5):
        major = (yard % 10 == 0)
        ax_.plot(
            [yard, yard], [0, 53.3],
            color=("white" if major else colors_["secondary"]),
            linewidth=(2.2 if major else 1.0),
            alpha=(0.75 if major else 0.35),
            zorder=1
        )

    # Goal lines
    ax_.plot([10, 10], [0, 53.3], color=colors_["secondary"], linewidth=4, alpha=0.9, zorder=2)
    ax_.plot([110, 110], [0, 53.3], color=colors_["secondary"], linewidth=4, alpha=0.9, zorder=2)

    # Sidelines + border
    ax_.plot([0, 120], [0, 0], color="white", linewidth=2.5, alpha=0.85, zorder=2)
    ax_.plot([0, 120], [53.3, 53.3], color="white", linewidth=2.5, alpha=0.85, zorder=2)
    ax_.plot([0, 0], [0, 53.3], color="white", linewidth=2.5, alpha=0.85, zorder=2)
    ax_.plot([120, 120], [0, 53.3], color="white", linewidth=2.5, alpha=0.85, zorder=2)

    # Hash marks
    for yard in range(10, 111):
        ax_.plot([yard, yard], [23.36, 23.36], color="white", marker=".", markersize=2, alpha=0.55, zorder=2)
        ax_.plot([yard, yard], [29.94, 29.94], color="white", marker=".", markersize=2, alpha=0.55, zorder=2)

    # Branding blocks
    if SHOW_TEAM_LOGO_AREA:
        ax_.add_patch(patches.Rectangle((0, 48), 10, 5.3, linewidth=0, facecolor=colors_["primary"], alpha=0.35, zorder=0))
        ax_.add_patch(patches.Rectangle((110, 48), 10, 5.3, linewidth=0, facecolor=colors_["secondary"], alpha=0.25, zorder=0))
        ax_.text(5, 50.6, "RAVENS", ha="center", va="center", fontsize=10, color="white", fontweight="bold", alpha=0.9, zorder=3)
        ax_.text(115, 50.6, "BALTIMORE", ha="center", va="center", fontsize=10, color="white", fontweight="bold", alpha=0.9, zorder=3)


def draw_dci_dis_gauge(ax_, dci_value, dis_value, colors_):
    """Gauge box in upper-right (data coordinates)."""
    artists = []
    gx, gy = 102, 43
    gw, gh = 16, 9

    bg = Rectangle((gx, gy), gw, gh, facecolor=colors_["primary"], edgecolor="white",
                   linewidth=2, alpha=0.92, zorder=100)
    ax_.add_patch(bg); artists.append(bg)

    t = ax_.text(gx + gw/2, gy + gh - 1, "DEFENSIVE METRICS",
                 ha="center", va="top", fontsize=10, color="white",
                 fontweight="bold", zorder=101)
    artists.append(t)

    bar_h = 1.2
    bar_w = gw - 3

    # DCI
    y_dci = gy + gh - 3.5
    dci_bg = Rectangle((gx + 1.5, y_dci), bar_w, bar_h, facecolor="#333333",
                       edgecolor="white", linewidth=1, alpha=0.55, zorder=101)
    ax_.add_patch(dci_bg); artists.append(dci_bg)

    dci_fill = Rectangle((gx + 1.5, y_dci), bar_w * float(np.clip(dci_value, 0, 1)), bar_h,
                         facecolor=plt.cm.RdYlGn(float(np.clip(dci_value, 0, 1))),
                         edgecolor="none", alpha=0.85, zorder=102)
    ax_.add_patch(dci_fill); artists.append(dci_fill)

    artists.append(ax_.text(gx + 1.5, y_dci - 0.4, "DCI (Coverage)",
                            ha="left", va="top", fontsize=8, color="white",
                            fontweight="bold", zorder=103))
    artists.append(ax_.text(gx + gw - 1.5, y_dci + bar_h/2, f"{float(dci_value):.3f}",
                            ha="right", va="center", fontsize=9, color="white",
                            fontweight="bold", zorder=103))

    # DIS
    y_dis = y_dci - 2.5
    dis_bg = Rectangle((gx + 1.5, y_dis), bar_w, bar_h, facecolor="#333333",
                       edgecolor="white", linewidth=1, alpha=0.55, zorder=101)
    ax_.add_patch(dis_bg); artists.append(dis_bg)

    dis_fill = Rectangle((gx + 1.5, y_dis), bar_w * float(np.clip(dis_value, 0, 1)), bar_h,
                         facecolor=plt.cm.RdYlGn(float(np.clip(dis_value, 0, 1))),
                         edgecolor="none", alpha=0.85, zorder=102)
    ax_.add_patch(dis_fill); artists.append(dis_fill)

    artists.append(ax_.text(gx + 1.5, y_dis - 0.4, "DIS (Integrity)",
                            ha="left", va="top", fontsize=8, color="white",
                            fontweight="bold", zorder=103))
    artists.append(ax_.text(gx + gw - 1.5, y_dis + bar_h/2, f"{float(dis_value):.3f}",
                            ha="right", va="center", fontsize=9, color="white",
                            fontweight="bold", zorder=103))
    return artists

# ============================================================================
# FIGURE SETUP
# ============================================================================
fig = plt.figure(figsize=(16, 10), facecolor=colors["background"])
ax = fig.add_subplot(111, facecolor=colors["field"])
draw_field(ax, colors)

offense_scatter = ax.scatter([], [], c=colors["offense_color"], s=360,
                             edgecolors="white", linewidths=2.5, zorder=5, alpha=0.95)
defense_scatter = ax.scatter([], [], c=colors["defense_color"], s=360,
                             edgecolors=colors["secondary"], linewidths=2.5, zorder=5, alpha=0.95)

player_labels = []
stress_circles = []
gauge_artists = []

# Stats text (top-left)
stats_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes,
    fontsize=13, va="top", color="white", fontweight="bold",
    bbox=dict(boxstyle="round", facecolor=colors["primary"], alpha=0.86,
              edgecolor=colors["secondary"], linewidth=2),
)

# Title (top-center)
ax.text(
    0.5, 0.98, "Baltimore Ravens vs San Francisco 49ers â€” Christmas Day 2023",
    transform=ax.transAxes, fontsize=14, va="top", ha="center",
    color="white", fontweight="bold",
    bbox=dict(boxstyle="round", facecolor=colors["primary"], alpha=0.86,
              edgecolor=colors["secondary"], linewidth=3),
)

# Keep prior defense positions for DIS (optional)
prev_defense_positions = None

def init():
    offense_scatter.set_offsets(np.empty((0, 2)))
    defense_scatter.set_offsets(np.empty((0, 2)))
    stats_text.set_text("")
    return [offense_scatter, defense_scatter, stats_text]

def update(frame_num):
    global prev_defense_positions

    frame_id = frames_sorted[frame_num]
    current_frame = play_data[play_data["frame_id"] == frame_id]
    actual_positions = current_frame[~current_frame["is_projection"]].copy()

    # Clear previous artists
    for t in player_labels:
        t.remove()
    player_labels.clear()

    for c in stress_circles:
        c.remove()
    stress_circles.clear()

    for a in gauge_artists:
        a.remove()
    gauge_artists.clear()

    # Split sides
    offense_data = actual_positions[actual_positions["player_side"] == "Offense"]
    defense_data = actual_positions[actual_positions["player_side"] == "Defense"]

    offense_scatter.set_offsets(offense_data[["x", "y"]].values if not offense_data.empty else np.empty((0, 2)))
    defense_scatter.set_offsets(defense_data[["x", "y"]].values if not defense_data.empty else np.empty((0, 2)))

    # DCI/DIS
    if METRIC_MODE == "precomputed" and has_dci and has_dis:
        dci_value = float(pd.to_numeric(current_frame[DCI_COLUMN], errors="coerce").mean())
        dis_value = float(pd.to_numeric(current_frame[DIS_COLUMN], errors="coerce").mean())
        if np.isnan(dci_value): dci_value = 0.5
        if np.isnan(dis_value): dis_value = 0.5
    else:
        if not defense_data.empty:
            defense_positions = defense_data[["x", "y"]].values.astype(float)
            dci_value = calculate_geometric_dci(defense_positions)
            dis_value = calculate_geometric_dis(defense_positions, prev_defense_positions)
            prev_defense_positions = defense_positions
        else:
            dci_value, dis_value = 0.5, 0.5

    if SHOW_DCI_DIS_GAUGE:
        gauge_artists.extend(draw_dci_dis_gauge(ax, dci_value, dis_value, colors))

    # Node stress halos (tuned for typical stress ranges; robust to any range)
    if SHOW_NODE_STRESS and "node_stress" in actual_positions.columns:
        stress_vals = pd.to_numeric(actual_positions["node_stress"], errors="coerce")
        # Adaptive thresholds by play distribution (fallback to defaults)
        q1 = float(np.nanpercentile(stress_vals.values, 33)) if np.isfinite(stress_vals).any() else 0.20
        q2 = float(np.nanpercentile(stress_vals.values, 66)) if np.isfinite(stress_vals).any() else 0.35

        for _, p in actual_positions.iterrows():
            s = p.get("node_stress", np.nan)
            if pd.isna(s):
                continue
            s = float(s)
            s_clip = float(np.clip(s, 0.0, 1.0))

            if s <= q1:
                col = "#00FF80"
            elif s <= q2:
                col = "#FFD700"
            else:
                col = "#FF4500"

            radius = 2.4 + (s_clip * 4.2)
            alpha = 0.28 + (s_clip * 0.45)

            circ = Circle((float(p["x"]), float(p["y"])), radius,
                          color=col, alpha=alpha, zorder=1,
                          linewidth=2, edgecolor="white")
            ax.add_patch(circ)
            stress_circles.append(circ)

    # Player labels
    if SHOW_POSITIONS or SHOW_PLAYER_NAMES or SHOW_PLAYER_NUMBERS:
        for _, p in actual_positions.iterrows():
            pid = int(p["nfl_id"])
            pos = p.get("player_position", np.nan)

            if SHOW_POSITIONS and pd.notna(pos):
                label_text = str(pos)
            elif SHOW_PLAYER_NAMES and pid in PLAYER_NAMES:
                label_text = PLAYER_NAMES[pid]
            elif SHOW_PLAYER_NUMBERS and ("jersey_number" in actual_positions.columns) and pd.notna(p.get("jersey_number", np.nan)):
                label_text = f"#{int(p['jersey_number'])}"
            else:
                label_text = str(pid)

            bg = colors["offense_color"] if p["player_side"] == "Offense" else colors["defense_color"]
            txt = colors["primary"]

            t = ax.text(
                float(p["x"]), float(p["y"]) - 1.5, label_text,
                ha="center", va="top", fontsize=9, color=txt,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=bg, alpha=0.85,
                          edgecolor="white", linewidth=1),
                zorder=10
            )
            player_labels.append(t)

    # Stats panel
    offense_count = len(offense_data)
    defense_count = len(defense_data)
    avg_stress = float(pd.to_numeric(actual_positions["node_stress"], errors="coerce").mean())

    stats = (
        f"Frame: {frame_num + 1}/{num_frames}\n"
        f"OFF: {offense_count} | DEF: {defense_count}\n"
    )
    if np.isfinite(avg_stress):
        stats += f"Avg Stress: {avg_stress:.3f}\n"

    stats += f"\nDCI: {dci_value:.3f}\nDIS: {dis_value:.3f}"

    if SHOW_NODE_STRESS and "node_stress" in actual_positions.columns:
        stats += "\n\nSTRESS (adaptive):"
        stats += f"\nLow (â‰¤ {q1:.3f})"
        stats += f"\nMed (â‰¤ {q2:.3f})"
        stats += f"\nHigh (> {q2:.3f})"

    stats_text.set_text(stats)

    elements = [offense_scatter, defense_scatter, stats_text]
    elements.extend(player_labels)
    elements.extend(stress_circles)
    elements.extend(gauge_artists)
    return elements

print(f"\nGenerating animation with {num_frames} frames...")

anim = FuncAnimation(fig, update, init_func=init, frames=num_frames, interval=100, blit=True)

output_base = f"{TEAM_THEME}_game{GAME_ID}_play{PLAY_ID}_metrics"
out_gif = os.path.join(OUTPUT_DIR, f"{output_base}.gif")
out_mp4 = os.path.join(OUTPUT_DIR, f"{output_base}.mp4")

if VIDEO_FORMAT in ["gif", "both"]:
    print(f"Saving GIF -> {out_gif}")
    anim.save(out_gif, writer=PillowWriter(fps=FPS))
    print("âœ“ GIF saved")

if VIDEO_FORMAT in ["mp4", "both"]:
    try:
        print(f"Saving MP4 -> {out_mp4}")
        anim.save(out_mp4, writer=FFMpegWriter(fps=FPS, bitrate=2000))
        print("âœ“ MP4 saved")
    except Exception as e:
        print(f"[WARN] MP4 export failed (ffmpeg not available). Reason: {e}")

plt.close()

print("\n" + "=" * 60)
print("ANIMATION COMPLETE!")
print("=" * 60)
print(f"Output dir: {OUTPUT_DIR}")
print(f"Base name: {output_base}")
print(f"Metric Mode: {METRIC_MODE.upper()}")
print("DCI: Defensive Coverage Index (0=loose, 1=tight)")
print("DIS: Defensive Integrity Score (0=chaotic, 1=disciplined)")



"""
NFL Big Data Bowl - Seahawks Theme Animation with DCI/DIS Gauge (Kaggle-Safe)
Uses: /kaggle/input/input-and-results/seahawks_play_101_data.csv

What this version does:
- Reads the CSV from the exact Kaggle path you provided
- Normalizes column names (nflId/frameId/playerSide -> nfl_id/frame_id/player_side)
- Normalizes player_side values (Defense/Offense, defense/offense, etc.)
- Handles presence/absence of: s, node_stress, player_position, jersey_number
- Uses Seahawks branding (navy/green/silver) with broadcast-style background
- Writes outputs to /kaggle/working/team_animations (writable)
"""

import matplotlib
matplotlib.use("Agg")

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Circle, Rectangle

warnings.filterwarnings("ignore")

# ============================================================================
# TEAM COLOR SCHEMES
# ============================================================================
TEAM_COLORS = {
    "seahawks": {
        "name": "Seattle Seahawks",
        "primary": "#002244",     # Navy
        "secondary": "#69BE28",   # Action Green
        "accent": "#A5ACAF",      # Wolf Gray
        "background": "#070b12",  # broadcast dark navy/black
        "field": "#214a21",       # slightly darker green
        "offense_color": "#69BE28",
        "defense_color": "#A5ACAF",
        "field_stripe": "#173517",
    },
    "default": {
        "name": "NFL Default",
        "primary": "#013369",
        "secondary": "#D50A0A",
        "accent": "#00d4ff",
        "background": "#0a0e1a",
        "field": "#2d5016",
        "offense_color": "#1E90FF",
        "defense_color": "#DC143C",
        "field_stripe": "#1d3010",
    },
}

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_CSV = "/kaggle/input/input-and-results/seahawks_play_101_data.csv"
OUTPUT_DIR = "/kaggle/working/team_animations"
TEAM_THEME = "seahawks"

# Animation settings
FPS = 10
VIDEO_FORMAT = "gif"          # "mp4", "gif", "both"
SHOW_TEAM_LOGO_AREA = True
SHOW_NODE_STRESS = True
SHOW_DCI_DIS_GAUGE = True

# Labels
SHOW_PLAYER_NAMES = True
SHOW_PLAYER_NUMBERS = False
SHOW_POSITIONS = False        # set True if you want player_position labels

# Metric mode
METRIC_MODE = "geometric"     # "precomputed" or "geometric"
DCI_COLUMN = "dci_score"
DIS_COLUMN = "dis_score"

# Output naming
GAME_ID = "2023090700"
PLAY_ID = "101"

# ============================================================================
# PLAYER NAME LOOKUPS (optional/partial)
# ============================================================================
SEAHAWKS_GIANTS_PLAYER_NAMES = {
    38577: "Bobby Wagner",
    39987: "Drew Lock",
    42412: "Jake Bobo",
    42543: "Quandre Diggs",
    42547: "Will Dissly",
    43329: "Tyler Lockett",
    43333: "Uchenna Nwosu",
    44818: "Boye Mafe",
    44830: "Riq Woolen",
    45186: "Kenneth Walker III",
    47789: "Geno Smith",
    47803: "Noah Fant",
    47825: "Daniel Jones",
    47847: "DK Metcalf",
    47872: "Jordyn Brooks",
    47891: "Jamal Adams",
    47941: "Devon Witherspoon",
    47954: "Jaxon Smith-Njigba",
}
PLAYER_NAMES = SEAHAWKS_GIANTS_PLAYER_NAMES

# ============================================================================
# LOAD + NORMALIZE
# ============================================================================
colors = TEAM_COLORS.get(TEAM_THEME, TEAM_COLORS["default"])
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "=" * 60)
print(f"NFL ANIMATION - {colors['name']} Theme")
print("=" * 60 + "\n")
print(f"Loading: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
print(f"Rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")

# Normalize common naming variants
rename_map = {}
if "nflId" in df.columns and "nfl_id" not in df.columns:
    rename_map["nflId"] = "nfl_id"
if "frameId" in df.columns and "frame_id" not in df.columns:
    rename_map["frameId"] = "frame_id"
if "playerSide" in df.columns and "player_side" not in df.columns:
    rename_map["playerSide"] = "player_side"
if rename_map:
    df = df.rename(columns=rename_map)

required = ["nfl_id", "frame_id", "x", "y", "player_side"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}\nPresent: {list(df.columns)}")

# Ensure optional cols exist
if "s" not in df.columns:
    df["s"] = np.nan
if "node_stress" not in df.columns:
    df["node_stress"] = np.nan
if "player_position" not in df.columns:
    df["player_position"] = np.nan
# jersey_number is optional; leave as-is if missing

# Coerce numeric
for col in ["x", "y", "s", "node_stress"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce").astype("Int64")
df["nfl_id"] = pd.to_numeric(df["nfl_id"], errors="coerce").astype("Int64")

# Normalize side values
df["player_side"] = (
    df["player_side"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "offense": "Offense", "off": "Offense", "o": "Offense",
        "defense": "Defense", "def": "Defense", "d": "Defense",
    })
)
df = df[df["player_side"].isin(["Offense", "Defense"])].copy()

# Drop unusable rows
df = df.dropna(subset=["frame_id", "nfl_id", "x", "y"]).copy()

# Treat file as a single play (new animation format)
print("Treating file as a single play (new animation format).")
play_data = df.copy()
play_data["is_projection"] = play_data["s"].isna()

has_dci = DCI_COLUMN in play_data.columns
has_dis = DIS_COLUMN in play_data.columns
if METRIC_MODE == "precomputed" and not (has_dci and has_dis):
    print("\n[WARN] METRIC_MODE='precomputed' but DCI/DIS columns not found. Switching to geometric.")
    METRIC_MODE = "geometric"

frames_sorted = sorted(play_data["frame_id"].unique().tolist())
num_frames = len(frames_sorted)

print(f"\nGame ID: {GAME_ID}")
print(f"Play ID: {PLAY_ID}")
print(f"Players: {play_data['nfl_id'].nunique()}")
print(f"Frames: {num_frames}")
print(f"Metric Mode: {METRIC_MODE.upper()}")

# ============================================================================
# METRICS
# ============================================================================
def calculate_geometric_dci(defense_positions: np.ndarray) -> float:
    """Geometric proxy: inverse of avg nearest-neighbor spacing among defenders."""
    if defense_positions is None or len(defense_positions) < 2:
        return 0.5

    nn = []
    for i, pos in enumerate(defense_positions):
        other = np.delete(defense_positions, i, axis=0)
        if len(other):
            d = np.linalg.norm(other - pos, axis=1)
            nn.append(np.min(d))
    if not nn:
        return 0.5

    avg_spacing = float(np.mean(nn))
    return float(np.clip(1.0 - (avg_spacing - 3.0) / 12.0, 0.0, 1.0))


def calculate_geometric_dis(defense_positions: np.ndarray, prev_defense_positions=None) -> float:
    """Geometric proxy: inverse coefficient of variation of defender pairwise distances."""
    if defense_positions is None or len(defense_positions) < 3:
        return 0.5

    dists = []
    for i in range(len(defense_positions)):
        for j in range(i + 1, len(defense_positions)):
            dists.append(np.linalg.norm(defense_positions[i] - defense_positions[j]))
    if not dists:
        return 0.5

    dists = np.array(dists, dtype=float)
    mean = float(np.mean(dists))
    std = float(np.std(dists))
    if mean <= 0:
        return 0.5

    cv = std / mean
    return float(np.clip(1.0 - cv, 0.0, 1.0))

# ============================================================================
# DRAWING
# ============================================================================
def draw_field(ax_, colors_):
    """Team-themed field with Seahawks look (navy/green/silver accents)."""
    ax_.set_xlim(0, 120)
    ax_.set_ylim(0, 53.3)
    ax_.set_aspect("equal")
    ax_.set_xticks([])
    ax_.set_yticks([])

    # Stripes
    for yard in range(0, 120, 10):
        ax_.add_patch(
            patches.Rectangle(
                (yard, 0), 5, 53.3,
                linewidth=0, edgecolor="none",
                facecolor=colors_["field_stripe"], alpha=0.18, zorder=0
            )
        )

    # Yard lines: major white, minor greenish
    for yard in range(10, 111, 5):
        major = (yard % 10 == 0)
        ax_.plot(
            [yard, yard], [0, 53.3],
            color=("white" if major else colors_["secondary"]),
            linewidth=(2.2 if major else 1.0),
            alpha=(0.75 if major else 0.35),
            zorder=1
        )

    # Goal lines (Seahawks green)
    ax_.plot([10, 10], [0, 53.3], color=colors_["secondary"], linewidth=4, alpha=0.9, zorder=2)
    ax_.plot([110, 110], [0, 53.3], color=colors_["secondary"], linewidth=4, alpha=0.9, zorder=2)

    # Sidelines + border
    ax_.plot([0, 120], [0, 0], color="white", linewidth=2.5, alpha=0.85, zorder=2)
    ax_.plot([0, 120], [53.3, 53.3], color="white", linewidth=2.5, alpha=0.85, zorder=2)
    ax_.plot([0, 0], [0, 53.3], color="white", linewidth=2.5, alpha=0.85, zorder=2)
    ax_.plot([120, 120], [0, 53.3], color="white", linewidth=2.5, alpha=0.85, zorder=2)

    # Hash marks
    for yard in range(10, 111):
        ax_.plot([yard, yard], [23.36, 23.36], color="white", marker=".", markersize=2, alpha=0.55, zorder=2)
        ax_.plot([yard, yard], [29.94, 29.94], color="white", marker=".", markersize=2, alpha=0.55, zorder=2)

    # Branding blocks
    if SHOW_TEAM_LOGO_AREA:
        ax_.add_patch(patches.Rectangle((0, 48), 10, 5.3, linewidth=0, facecolor=colors_["primary"], alpha=0.40, zorder=0))
        ax_.add_patch(patches.Rectangle((110, 48), 10, 5.3, linewidth=0, facecolor=colors_["secondary"], alpha=0.25, zorder=0))
        ax_.text(5, 50.6, "SEAHAWKS", ha="center", va="center", fontsize=10, color="white", fontweight="bold", alpha=0.92, zorder=3)
        ax_.text(115, 50.6, "SEATTLE", ha="center", va="center", fontsize=10, color="white", fontweight="bold", alpha=0.92, zorder=3)


def draw_dci_dis_gauge(ax_, dci_value, dis_value, colors_):
    """Gauge box in upper-right (data coordinates)."""
    artists = []
    gx, gy = 102, 43
    gw, gh = 16, 9

    bg = Rectangle((gx, gy), gw, gh, facecolor=colors_["primary"], edgecolor="white",
                   linewidth=2, alpha=0.92, zorder=100)
    ax_.add_patch(bg); artists.append(bg)

    t = ax_.text(gx + gw/2, gy + gh - 1, "DEFENSIVE METRICS",
                 ha="center", va="top", fontsize=10, color="white",
                 fontweight="bold", zorder=101)
    artists.append(t)

    bar_h = 1.2
    bar_w = gw - 3

    # DCI
    y_dci = gy + gh - 3.5
    dci_bg = Rectangle((gx + 1.5, y_dci), bar_w, bar_h, facecolor="#333333",
                       edgecolor="white", linewidth=1, alpha=0.55, zorder=101)
    ax_.add_patch(dci_bg); artists.append(dci_bg)

    dci_fill = Rectangle((gx + 1.5, y_dci), bar_w * float(np.clip(dci_value, 0, 1)), bar_h,
                         facecolor=plt.cm.RdYlGn(float(np.clip(dci_value, 0, 1))),
                         edgecolor="none", alpha=0.85, zorder=102)
    ax_.add_patch(dci_fill); artists.append(dci_fill)

    artists.append(ax_.text(gx + 1.5, y_dci - 0.4, "DCI (Coverage)",
                            ha="left", va="top", fontsize=8, color="white",
                            fontweight="bold", zorder=103))
    artists.append(ax_.text(gx + gw - 1.5, y_dci + bar_h/2, f"{float(dci_value):.3f}",
                            ha="right", va="center", fontsize=9, color="white",
                            fontweight="bold", zorder=103))

    # DIS
    y_dis = y_dci - 2.5
    dis_bg = Rectangle((gx + 1.5, y_dis), bar_w, bar_h, facecolor="#333333",
                       edgecolor="white", linewidth=1, alpha=0.55, zorder=101)
    ax_.add_patch(dis_bg); artists.append(dis_bg)

    dis_fill = Rectangle((gx + 1.5, y_dis), bar_w * float(np.clip(dis_value, 0, 1)), bar_h,
                         facecolor=plt.cm.RdYlGn(float(np.clip(dis_value, 0, 1))),
                         edgecolor="none", alpha=0.85, zorder=102)
    ax_.add_patch(dis_fill); artists.append(dis_fill)

    artists.append(ax_.text(gx + 1.5, y_dis - 0.4, "DIS (Integrity)",
                            ha="left", va="top", fontsize=8, color="white",
                            fontweight="bold", zorder=103))
    artists.append(ax_.text(gx + gw - 1.5, y_dis + bar_h/2, f"{float(dis_value):.3f}",
                            ha="right", va="center", fontsize=9, color="white",
                            fontweight="bold", zorder=103))
    return artists

# ============================================================================
# FIGURE SETUP
# ============================================================================
fig = plt.figure(figsize=(16, 10), facecolor=colors["background"])
ax = fig.add_subplot(111, facecolor=colors["field"])
draw_field(ax, colors)

offense_scatter = ax.scatter([], [], c=colors["offense_color"], s=360,
                             edgecolors="white", linewidths=2.5, zorder=5, alpha=0.95)
defense_scatter = ax.scatter([], [], c=colors["defense_color"], s=360,
                             edgecolors=colors["primary"], linewidths=2.5, zorder=5, alpha=0.95)

player_labels = []
stress_circles = []
gauge_artists = []

# Stats text (top-left)
stats_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes,
    fontsize=13, va="top", color="white", fontweight="bold",
    bbox=dict(boxstyle="round", facecolor=colors["primary"], alpha=0.86,
              edgecolor=colors["secondary"], linewidth=2),
)

# Title (top-center)
ax.text(
    0.5, 0.98, "Seattle Seahawks â€” Play 101 (Tracking + DCI/DIS Gauge)",
    transform=ax.transAxes, fontsize=14, va="top", ha="center",
    color="white", fontweight="bold",
    bbox=dict(boxstyle="round", facecolor=colors["primary"], alpha=0.86,
              edgecolor=colors["secondary"], linewidth=3),
)

prev_defense_positions = None

def init():
    offense_scatter.set_offsets(np.empty((0, 2)))
    defense_scatter.set_offsets(np.empty((0, 2)))
    stats_text.set_text("")
    return [offense_scatter, defense_scatter, stats_text]

def update(frame_num):
    global prev_defense_positions

    frame_id = frames_sorted[frame_num]
    current_frame = play_data[play_data["frame_id"] == frame_id]
    actual_positions = current_frame[~current_frame["is_projection"]].copy()

    # Clear previous artists
    for t in player_labels:
        t.remove()
    player_labels.clear()

    for c in stress_circles:
        c.remove()
    stress_circles.clear()

    for a in gauge_artists:
        a.remove()
    gauge_artists.clear()

    offense_data = actual_positions[actual_positions["player_side"] == "Offense"]
    defense_data = actual_positions[actual_positions["player_side"] == "Defense"]

    offense_scatter.set_offsets(offense_data[["x", "y"]].values if not offense_data.empty else np.empty((0, 2)))
    defense_scatter.set_offsets(defense_data[["x", "y"]].values if not defense_data.empty else np.empty((0, 2)))

    # DCI/DIS
    if METRIC_MODE == "precomputed" and has_dci and has_dis:
        dci_value = float(pd.to_numeric(current_frame[DCI_COLUMN], errors="coerce").mean())
        dis_value = float(pd.to_numeric(current_frame[DIS_COLUMN], errors="coerce").mean())
        if np.isnan(dci_value): dci_value = 0.5
        if np.isnan(dis_value): dis_value = 0.5
    else:
        if not defense_data.empty:
            defense_positions = defense_data[["x", "y"]].values.astype(float)
            dci_value = calculate_geometric_dci(defense_positions)
            dis_value = calculate_geometric_dis(defense_positions, prev_defense_positions)
            prev_defense_positions = defense_positions
        else:
            dci_value, dis_value = 0.5, 0.5

    if SHOW_DCI_DIS_GAUGE:
        gauge_artists.extend(draw_dci_dis_gauge(ax, dci_value, dis_value, colors))

    # Node stress halos (adaptive thresholds)
    if SHOW_NODE_STRESS and "node_stress" in actual_positions.columns:
        stress_vals = pd.to_numeric(actual_positions["node_stress"], errors="coerce")
        q1 = float(np.nanpercentile(stress_vals.values, 33)) if np.isfinite(stress_vals).any() else 0.20
        q2 = float(np.nanpercentile(stress_vals.values, 66)) if np.isfinite(stress_vals).any() else 0.35

        for _, p in actual_positions.iterrows():
            s = p.get("node_stress", np.nan)
            if pd.isna(s):
                continue
            s = float(s)
            s_clip = float(np.clip(s, 0.0, 1.0))

            if s <= q1:
                col = "#00FF80"
            elif s <= q2:
                col = "#FFD700"
            else:
                col = "#FF4500"

            radius = 2.4 + (s_clip * 4.2)
            alpha = 0.28 + (s_clip * 0.45)

            circ = Circle((float(p["x"]), float(p["y"])), radius,
                          color=col, alpha=alpha, zorder=1,
                          linewidth=2, edgecolor="white")
            ax.add_patch(circ)
            stress_circles.append(circ)

    # Labels
    if SHOW_POSITIONS or SHOW_PLAYER_NAMES or SHOW_PLAYER_NUMBERS:
        for _, p in actual_positions.iterrows():
            pid = int(p["nfl_id"])
            pos = p.get("player_position", np.nan)

            if SHOW_POSITIONS and pd.notna(pos):
                label_text = str(pos)
            elif SHOW_PLAYER_NAMES and pid in PLAYER_NAMES:
                label_text = PLAYER_NAMES[pid]
            elif SHOW_PLAYER_NUMBERS and ("jersey_number" in actual_positions.columns) and pd.notna(p.get("jersey_number", np.nan)):
                label_text = f"#{int(p['jersey_number'])}"
            else:
                label_text = str(pid)

            bg = colors["offense_color"] if p["player_side"] == "Offense" else colors["defense_color"]
            txt = colors["primary"]

            t = ax.text(
                float(p["x"]), float(p["y"]) - 1.5, label_text,
                ha="center", va="top", fontsize=9, color=txt,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=bg, alpha=0.85,
                          edgecolor="white", linewidth=1),
                zorder=10
            )
            player_labels.append(t)

    # Stats
    offense_count = len(offense_data)
    defense_count = len(defense_data)
    avg_stress = float(pd.to_numeric(actual_positions["node_stress"], errors="coerce").mean())

    stats = (
        f"Frame: {frame_num + 1}/{num_frames}\n"
        f"OFF: {offense_count} | DEF: {defense_count}\n"
    )
    if np.isfinite(avg_stress):
        stats += f"Avg Stress: {avg_stress:.3f}\n"
    stats += f"\nDCI: {dci_value:.3f}\nDIS: {dis_value:.3f}"

    if SHOW_NODE_STRESS and "node_stress" in actual_positions.columns:
        stats += "\n\nSTRESS (adaptive):"
        stats += f"\nLow (â‰¤ {q1:.3f})"
        stats += f"\nMed (â‰¤ {q2:.3f})"
        stats += f"\nHigh (> {q2:.3f})"

    stats_text.set_text(stats)

    elements = [offense_scatter, defense_scatter, stats_text]
    elements.extend(player_labels)
    elements.extend(stress_circles)
    elements.extend(gauge_artists)
    return elements

print(f"\nGenerating animation with {num_frames} frames...")

anim = FuncAnimation(fig, update, init_func=init, frames=num_frames, interval=100, blit=True)

output_base = f"{TEAM_THEME}_game{GAME_ID}_play{PLAY_ID}_metrics"
out_gif = os.path.join(OUTPUT_DIR, f"{output_base}.gif")
out_mp4 = os.path.join(OUTPUT_DIR, f"{output_base}.mp4")

if VIDEO_FORMAT in ["gif", "both"]:
    print(f"Saving GIF -> {out_gif}")
    anim.save(out_gif, writer=PillowWriter(fps=FPS))
    print("âœ“ GIF saved")

if VIDEO_FORMAT in ["mp4", "both"]:
    try:
        print(f"Saving MP4 -> {out_mp4}")
        anim.save(out_mp4, writer=FFMpegWriter(fps=FPS, bitrate=2000))
        print("âœ“ MP4 saved")
    except Exception as e:
        print(f"[WARN] MP4 export failed (ffmpeg not available). Reason: {e}")

plt.close()

print("\n" + "=" * 60)
print("ANIMATION COMPLETE!")
print("=" * 60)
print(f"Output dir: {OUTPUT_DIR}")
print(f"Base name: {output_base}")
print(f"Metric Mode: {METRIC_MODE.upper()}")
print("DCI: Defensive Coverage Index (0=loose, 1=tight)")
print("DIS: Defensive Integrity Score (0=chaotic, 1=disciplined)")





