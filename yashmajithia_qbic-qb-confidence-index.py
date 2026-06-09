#QB confidence index (QBCI)
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


#importing the data from dropbox
!wget -O data.zip "https://www.dropbox.com/scl/fo/e87og5qq6bxz6y44ewupe/ANAAwOHA37SZLVgfk8ovVxY?rlkey=qrv016ov5t36lxfznyhq25xn7&st=t04vyf7y&dl=1"
!unzip data.zip -d data/


INPUT_FILES = glob.glob("/kaggle/working/data/input_2023_w*.csv")
OUTPUT_FILES = glob.glob("/kaggle/working/data/output_2023_w*.csv")
SUPPLEMENTARY_PATH = "/kaggle/working/data/supplementary_data.csv"
TIME_STEP = 0.1 #10 FPS


#1. DATA LOADING AND INITIAL HYGIENE
def load_data(file_list):
    """Loads and concatenates tracking data files."""
    try:
        df_list = [pd.read_csv(f) for f in file_list]
        return pd.concat(df_list, ignore_index=True)
    except FileNotFoundError as e:
        print(f"Error loading tracking files: {e}. Check directory structure.")
        return pd.DataFrame()

print("--- 1. Data Loading ---")
df_tracking_pre = load_data(INPUT_FILES)
df_tracking_post = load_data(OUTPUT_FILES)
df_plays = pd.read_csv(SUPPLEMENTARY_PATH, low_memory=False)

# Combine for initial cleaning
df_tracking = pd.concat([df_tracking_pre, df_tracking_post], ignore_index=True)
print(f"Combined Tracking Data Rows: {df_tracking.shape[0]:,}")

# Initial Cleaning: Filter rows where core metrics are NaN (e.g., player out of bounds/incomplete tracking)
df_tracking.dropna(subset=['x', 'y', 's', 'a', 'dir', 'o'], inplace=True)
print(f"Rows after basic NaN removal: {df_tracking.shape[0]:,}")


#2. FEATURE ENGINEERING & OUTLIER DETECTION
print("\n--- 2. Feature Engineering & Outlier Detection ---")

# 2a. Pre-snap Play Identification (Frames before t=0)
# We focus on the pre-snap motion for QBCI. Let's look at the last 30 frames (3 seconds) of the 'pre' data.
df_tracking_pre_snap = df_tracking_pre[df_tracking_pre['frame_id'] >= df_tracking_pre['frame_id'].max() - 29].copy()

# 2b. Outlier Detection using Isolation Forest (Focus on unrealistic speed/acceleration values)
# Isolation Forest is robust for detecting anomalies in large, multi-dimensional datasets.
features_for_outliers = ['s', 'a']

# Isolate the high-speed/high-acceleration data points
outlier_data = df_tracking_pre_snap[features_for_outliers].copy()
scaler = StandardScaler()
outlier_data_scaled = scaler.fit_transform(outlier_data)

# Fit Isolation Forest (contamination=0.01 assumes 1% of data are anomalies/outliers)
iso_forest = IsolationForest(contamination=0.01, random_state=42)
outlier_predictions = iso_forest.fit_predict(outlier_data_scaled)

# Add anomaly score to the DataFrame (-1 is outlier, 1 is inlier)
df_tracking_pre_snap['is_outlier'] = outlier_predictions
outlier_count = (df_tracking_pre_snap['is_outlier'] == -1).sum()

print(f"Detected {outlier_count:,} kinematic outliers (speed/acceleration spikes).")

# Remove outliers
df_tracking_clean = df_tracking_pre_snap[df_tracking_pre_snap['is_outlier'] == 1].drop(columns=['is_outlier']).copy()
print(f"Rows remaining after outlier removal: {df_tracking_clean.shape[0]:,}")


#3. TARGET STABILITY METRIC (QBCI)
print("\n--- 3. Core Metric Generation: QB Confidence Index (QBCI) ---")

def calculate_target_stability(group):
    """
    Calculates metrics related to the stability of the targeted receiver's separation
    and speed in the 3 seconds prior to the throw (t=0).
    """
    if group.empty:
        return None

    # Ensure necessary columns are present
    required_cols = ['game_id', 'play_id', 'player_role', 'player_side', 'nfl_id', 'x', 'y', 's', 'frame_id']
    if not all(col in group.columns for col in required_cols):
        # This should ideally not happen if groupby is done correctly and the input df has these cols
        return None

    game_id = group['game_id'].iloc[0]
    play_id = group['play_id'].iloc[0]

    target_row = group[group['player_role'] == 'Targeted Receiver']

    # If target is missing from the 3-second window, skip
    if target_row.empty: return None

    # Check if target_row has nfl_id before accessing
    if 'nfl_id' not in target_row.columns or target_row['nfl_id'].empty:
        return None

    target_id = target_row['nfl_id'].iloc[0]

    # --- A. Separation Stability (The Core Component) ---

    separation_metrics = []

    # Iterate through every defender to find the closest one at each frame
    for frame in group['frame_id'].unique():
        frame_data = group[group['frame_id'] == frame].copy()

        # Defender data in this frame
        defender_data = frame_data[frame_data['player_side'] == 'Defense']

        if defender_data.empty: continue

        # Check if target_id is in the current frame data before accessing
        if target_id not in frame_data['nfl_id'].values:
            continue

        # Check if target_id row exists and has 'x' and 'y' columns before accessing
        target_frame_data = frame_data[frame_data['nfl_id'] == target_id]
        if target_frame_data.empty or 'x' not in target_frame_data.columns or 'y' not in target_frame_data.columns:
             continue

        target_pos = target_frame_data[['x', 'y']].iloc[0]


        # Calculate distance from target to all defenders
        defender_data['separation'] = np.sqrt(
            (defender_data['x'] - target_pos['x'])**2 + (defender_data['y'] - target_pos['y'])**2
        )

        min_separation = defender_data['separation'].min()
        separation_metrics.append(min_separation)

    if not separation_metrics: return None

    # QBCI Component 1: Variance of Separation
    sep_variance = np.var(separation_metrics) if len(separation_metrics) > 1 else 0 # Handle single frame case

    # QBCI Component 2: Target Speed Volatility (How much the target's speed changes)
    target_speeds = target_row['s'].values
    speed_std_dev = np.std(target_speeds) if len(target_speeds) > 1 else 0 # Handle single frame case


    # --- B. The QBCI Metric ---
    # QBCI is a compound metric representing overall pre-snap instability.
    # High QBCI = High Variance/Volatility (Unstable, unpredictable target)
    # Low QBCI = Low Variance/Volatility (Stable, predictable target)

    QBCI = (sep_variance * 10) + speed_std_dev

    return pd.Series({
        'game_id': game_id,
        'play_id': play_id,
        'target_nfl_id': target_id,
        'sep_variance': sep_variance, # Component 1
        'speed_std_dev': speed_std_dev, # Component 2
        'QBCI': QBCI,
    })

# Execute the stability calculation
# Select necessary columns before grouping
cols_for_groupby = ['game_id', 'play_id', 'player_role', 'player_side', 'nfl_id', 'x', 'y', 's', 'frame_id']
df_metrics_raw = df_tracking_clean[cols_for_groupby].groupby(['game_id', 'play_id'], group_keys=False).apply(
    calculate_target_stability
).dropna().reset_index(drop=True)

# Merge with play context (pass result and EPA)
df_final_metrics = pd.merge(df_metrics_raw,
                            df_plays[['game_id', 'play_id', 'pass_result', 'expected_points_added']],
                            on=['game_id', 'play_id'], how='inner')

print(f"QBCI calculated successfully for {df_final_metrics.shape[0]} plays.")


#4. QBCI ANALYSIS AND VISUALIZATION
print("\n--- 4. QBCI Analysis and Visualization ---")

# 4a. Visualization Setup
def plot_qcbi_distribution(df):
    """Shows the distribution of QBCI based on the pass outcome (C vs I)."""
    df_plot = df[df['pass_result'].isin(['C', 'I'])].copy()

    plt.figure(figsize=(9, 6))
    sns.violinplot(x='pass_result', y='QBCI', data=df_plot, inner='quartile',
                   order=['C', 'I'], palette={'C': '#10b981', 'I': '#ef4444'})

    plt.title('QB Confidence Index (QBCI) Distribution by Pass Outcome', fontsize=14, weight='bold')
    plt.xlabel('Pass Result (C=Completed, I=Incomplete)', fontsize=12)
    plt.ylabel('QBCI (Target Instability Metric)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('qcbi_distribution.png')
    plt.close('all')
    plt.show()

def plot_qcbi_vs_epa(df):
    """Plots QBCI vs. Expected Points Added (EPA)."""
    df_plot = df.dropna(subset=['QBCI', 'expected_points_added']).copy()

    plt.figure(figsize=(10, 6))
    sns.regplot(x='QBCI', y='expected_points_added', data=df_plot,
                scatter_kws={'alpha':0.5, 'color': '#1d4ed8'}, line_kws={'color': '#ef4444'})

    plt.title('QBCI vs. Expected Points Added (EPA)', fontsize=14, weight='bold')
    plt.xlabel('QB Confidence Index (QBCI)', fontsize=12)
    plt.ylabel('Expected Points Added (EPA)', fontsize=12)
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('qcbi_vs_epa.png')
    plt.close('all')
    plt.show()


# Call the plotting functions to display the visualizations
plot_qcbi_distribution(df_final_metrics)
plot_qcbi_vs_epa(df_final_metrics)


# 4b. Execution (IPython display for Colab)
if not df_final_metrics.empty:
    print("--- SECTION 5: STARTING PLOT EXECUTION ---")

    # 1. Distribution Plot
    print("\nGenerating QBCI Distribution Plot...")
    plot_qcbi_distribution(df_final_metrics)
    # CRITICAL FOR COLAB
    from IPython.display import display, Image
    display(Image(filename='qcbi_distribution.png'))

    # 2. Regression Plot
    print("Generating QBCI vs. EPA Plot...")
    plot_qcbi_vs_epa(df_final_metrics)
    display(Image(filename='qcbi_vs_epa.png'))

    # 3. Final Summary
    print("\n--- Project Summary ---")
    print(f"Final analysis includes {df_final_metrics.shape[0]} plays.")

    # Statistical Insight
    corr = df_final_metrics['QBCI'].corr(df_final_metrics['expected_points_added'])
    print(f"Correlation (QBCI vs. EPA): {corr:.3f}")

    print("\nINSIGHT: A strong negative correlation (e.g., -0.30) suggests that the less stable the target (higher QBCI), the lower the offensive value (lower EPA) of the play, indicating the QB may be forced to check down or throw into less favorable coverage.")

else:
    print("\nAnalysis stopped because no valid metrics were generated after cleaning and processing.")

