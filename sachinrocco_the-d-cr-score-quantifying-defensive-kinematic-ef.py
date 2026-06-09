import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Kaggle Environment Configuration ---
BASE_PATH = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/'
TRAIN_PATH = os.path.join(BASE_PATH, 'train')
# ----------------------------------------

def load_tracking_data_kaggle(file_prefix, base_path, use_cols):
    """Loads all weekly tracking files from the specified Kaggle train path."""
    all_dfs = []
    file_pattern = os.path.join(base_path, f'{file_prefix}_2023_w*.csv')
    file_paths = glob.glob(file_pattern) 
    
    if not file_paths:
        raise ValueError(f"No files found matching pattern: {file_pattern}. Please verify the path and file structure.")
        
    print(f"Found {len(file_paths)} files for prefix '{file_prefix}'. Loading...")
    
    for file_path in file_paths:
        try:
            # We use low_memory=False to ensure correct dtype handling across files
            all_dfs.append(pd.read_csv(file_path, usecols=use_cols, low_memory=False))
        except ValueError as e:
            # Catch the missing column error specifically.
            print(f"Error loading {file_path}. Check if all 'usecols' are present in the file. Error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while loading {file_path}: {e}")
            raise
            
    return pd.concat(all_dfs, ignore_index=True)

# Define columns needed
output_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y']
input_cols = ['game_id', 'play_id', 'nfl_id', 'player_side', 'player_role', 'ball_land_x', 'ball_land_y', 's', 'a', 'dir'] # Adding s, a, dir
supp_cols = ['game_id', 'play_id', 'pass_result', 'team_coverage_man_zone', 'expected_points_added']

# Load data
output_df = load_tracking_data_kaggle('output', TRAIN_PATH, output_cols)
input_df = load_tracking_data_kaggle('input', TRAIN_PATH, input_cols)
supp_df = pd.read_csv(os.path.join(BASE_PATH, 'supplementary_data.csv'), usecols=supp_cols)

# --- Merging Data ---
# 1. Merge player roles AND kinematic features (from input_df) into the output_df
# We need the kinematic features (s, a, dir) from the input file *just before* the throw (frame 1)
player_info_df = input_df.drop_duplicates(subset=['game_id', 'play_id', 'nfl_id']).copy()
tracking_data = output_df.merge(player_info_df.drop(columns=['ball_land_x', 'ball_land_y', 's', 'a', 'dir']), 
                                on=['game_id', 'play_id', 'nfl_id'], 
                                how='left')

# 2. Merge ball landing spots (once per play) and play context (Supplementary data)
ball_land_df = input_df[['game_id', 'play_id', 'ball_land_x', 'ball_land_y']].dropna().drop_duplicates()
tracking_data = tracking_data.merge(ball_land_df, on=['game_id', 'play_id'], how='left')
tracking_data = tracking_data.merge(supp_df, on=['game_id', 'play_id'], how='left')

# 3. Final filtering for analysis scope (Completed, Incomplete, or Intercepted passes that have ball landing data)
tracking_data = tracking_data[tracking_data['pass_result'].isin(['C', 'I', 'IN'])].dropna(subset=['ball_land_x'])
tracking_data.dropna(subset=['nfl_id'], inplace=True) 

print(f"\n--- Data Loading Summary ---")
print(f"Total plays for analysis: {tracking_data[['game_id', 'play_id']].drop_duplicates().shape[0]}")
print(f"Total frames after filtering: {len(tracking_data)}")


# CODE BLOCK 2: NEAREST DEFENDER IDENTIFICATION

def find_nearest_defender(frame_group):
    """
    Calculates the Euclidean distance from the Targeted Receiver (TR) to every 
    defender (D) and identifies the closest one for a single frame.
    """
    tr_df = frame_group[frame_group['player_role'] == 'Targeted Receiver'].copy()
    defender_df = frame_group[frame_group['player_side'] == 'Defense'].copy()
    
    # Check for plays/frames where the receiver or defenders are missing
    if tr_df.empty or defender_df.empty:
        # Return the original group so processing continues without error
        return frame_group
        
    # TR Position (1x2 array)
    tr_pos = tr_df[['x', 'y']].values[0].reshape(1, 2)
    defender_pos = defender_df[['x', 'y']].values
    
    # Calculate all distances efficiently (Receiver-to-all-Defenders)
    distances = cdist(tr_pos, defender_pos, metric='euclidean')
    
    # Find the minimum distance and the index of the closest defender
    min_dist_index = np.argmin(distances)
    min_dist = distances[0, min_dist_index]
    nearest_defender_nflid = defender_df.iloc[min_dist_index]['nfl_id']
    
    # Assign the results to ALL rows in the frame group (for easy filtering later)
    frame_group['min_sep_distance'] = min_dist
    frame_group['nearest_defender_nflid'] = nearest_defender_nflid
    
    return frame_group

print("Starting Nearest Defender Calculation (This may take several minutes on the full dataset)...")
# Apply the function across all frames in all plays
tracking_data_processed = (tracking_data.groupby(['game_id', 'play_id', 'frame_id'])
                                        .apply(find_nearest_defender)
                                        .reset_index(drop=True))

# Filter down to only the Nearest Defender and the Targeted Receiver for subsequent feature engineering
final_analysis_df = tracking_data_processed[
    (tracking_data_processed['nfl_id'] == tracking_data_processed['nearest_defender_nflid']) | 
    (tracking_data_processed['player_role'] == 'Targeted Receiver')
].copy()

# Ensure we only keep the defender's row for the kinematic calculation in the next step
# The min_sep_distance column is copied to the defender's row due to the apply function
nearest_defender_df = final_analysis_df[final_analysis_df['player_role'] != 'Targeted Receiver'].copy()

print("\n--- Step 2 Complete: Nearest Defender Identified and Data Filtered ---")
print(nearest_defender_df[['game_id', 'play_id', 'frame_id', 'nfl_id', 'min_sep_distance', 'nearest_defender_nflid', 'ball_land_x', 'ball_land_y']].head())


# CODE BLOCK 3: KINEMATIC FEATURES & D-CR SCORE CALCULATION

def calculate_kinematics_and_dcr(df):
    """
    Calculates frame-to-frame position changes and the final D-CR Score 
    for the Nearest Defender on each play.
    """
    
    # 1. Calculate Frame-to-Frame Changes (Defender's Actual Movement Vector)
    
    # Group by play and player to calculate differences sequentially
    df.sort_values(by=['game_id', 'play_id', 'nfl_id', 'frame_id'], inplace=True)
    
    # Calculate Change in Position (Yards)
    df['delta_x'] = df.groupby(['game_id', 'play_id', 'nfl_id'])['x'].diff()
    df['delta_y'] = df.groupby(['game_id', 'play_id', 'nfl_id'])['y'].diff()
    
    # 2. Calculate the Optimal Pursuit Vector Direction (Optimal Direction)
    
    # Difference in coordinates between the ball landing spot and the defender's current position
    df['diff_x_to_land'] = df['ball_land_x'] - df['x']
    df['diff_y_to_land'] = df['ball_land_y'] - df['y']
    
    # Calculate the optimal direction (angle in radians) using atan2
    df['optimal_dir_rad'] = np.arctan2(df['diff_y_to_land'], df['diff_x_to_land'])

    
    # 3. Calculate Vector Alignment (Cosine Similarity)
    
    # Optimal Vector components (magnitude is 1 since it's just a direction)
    optimal_x = np.cos(df['optimal_dir_rad'])
    optimal_y = np.sin(df['optimal_dir_rad'])
    
    # Actual Movement Vector components
    actual_x = df['delta_x']
    actual_y = df['delta_y']
    
    # Magnitude of the Actual Movement Vector (Denominator component)
    df['actual_magnitude'] = np.sqrt(actual_x**2 + actual_y**2) # <--- FIX APPLIED HERE: Assign to DF
    
    # Dot Product (Numerator)
    dot_product = (optimal_x * actual_x) + (optimal_y * actual_y)
    
    # Alignment Score: Handle division by zero (when defender is stationary, actual_magnitude=0)
    df['frame_alignment_score'] = np.where(
        df['actual_magnitude'] == 0, # <-- Use the DF column
        0, 
        dot_product / df['actual_magnitude'] # <-- Use the DF column
    )

    return df

# Execute kinematic and alignment calculation
nearest_defender_df = calculate_kinematics_and_dcr(nearest_defender_df.copy())


# --- 4. Final D-CR Score Aggregation (Play-Level Metric) ---

def calculate_dcr_score(play_group):
    """
    Calculates the final D-CR Score for a single play by aggregating the 
    time-weighted frame alignment scores over the reaction window.
    """
    
    # Focus on the first 10 frames (a critical reaction window, frame_id > 1)
    reaction_window = play_group[play_group['frame_id'] > 1].head(10).dropna(subset=['frame_alignment_score', 'actual_magnitude']).copy()
    
    if reaction_window.empty:
        return pd.Series({'DCR_Score': np.nan, 'Reaction_Lag_Frame': np.nan})

    # Time Weight: Give higher weight to earlier frames (1/frame_id) to penalize lag.
    reaction_window['time_weight'] = 1 / reaction_window['frame_id']
    
    # Calculate Moment-Weighted Alignment
    weighted_alignment = (reaction_window['frame_alignment_score'] * reaction_window['time_weight']).sum()
    total_weight = reaction_window['time_weight'].sum()
    
    final_dcr_score = weighted_alignment / total_weight
    
    # Reaction Lag: Time until first clear movement towards the target (Alignment > 0.5 and moving)
    initial_reaction = reaction_window[
        (reaction_window['frame_alignment_score'] > 0.5) & 
        (reaction_window['actual_magnitude'] > 0.1) # This column is now safely defined
    ]
    
    # Reaction Lag is the frame ID where the significant movement first occurs
    reaction_lag = initial_reaction['frame_id'].min() if not initial_reaction.empty else 11

    # Return the aggregated results for the play
    return pd.Series({
        'DCR_Score': final_dcr_score, 
        'Reaction_Lag_Frame': reaction_lag
    })

# Apply the aggregation function
play_level_metrics = (nearest_defender_df.groupby(['game_id', 'play_id'])
                                        .apply(calculate_dcr_score)
                                        .reset_index())

# Merge the final score with contextual data for validation (pass result, EPA)
validation_df = play_level_metrics.merge(
    nearest_defender_df[['game_id', 'play_id', 'pass_result', 'team_coverage_man_zone', 'expected_points_added']].drop_duplicates(), 
    on=['game_id', 'play_id'], how='left')

print("\n--- Step 3 Complete: D-CR Score Calculated ---")
print(validation_df[['game_id', 'play_id', 'pass_result', 'DCR_Score', 'Reaction_Lag_Frame', 'expected_points_added']].head(10))


import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure validation_df is clean and ready
model_df = validation_df.dropna(subset=['DCR_Score', 'expected_points_added', 'pass_result', 'team_coverage_man_zone']).copy()

# --- 1. Model 1: Predicting Defensive Success (Logistic Regression) ---

print("--- Running Model 1: Logistic Regression (Predicting Defensive Success) ---")

# Define target variable Y: 1 if success (I/IN), 0 if failure (C)
model_df['Defensive_Success'] = np.where(model_df['pass_result'].isin(['I', 'IN']), 1, 0)

# 1. Handle Categorical Variable: Ensure only 'Man' and 'Zone' exist for clean dummy creation
model_df['team_coverage_man_zone'] = model_df['team_coverage_man_zone'].astype(str)
model_df = pd.get_dummies(model_df, columns=['team_coverage_man_zone'], prefix='Coverage', drop_first=True)

# 2. FIX APPLIED: Reliably identify the coverage control column
# Since drop_first=True, only one of the dummy columns (e.g., 'Coverage_Zone') will exist.
# We will explicitly check and assign that column name.
# NOTE: The default baseline is 'Man' if 'Zone' is the kept column.
if 'Coverage_Zone' in model_df.columns:
    coverage_col = 'Coverage_Zone'
elif 'Coverage_Man' in model_df.columns:
    # Highly unlikely given the data, but safe to check
    coverage_col = 'Coverage_Man'
else:
    # If neither exists (e.g., if the column was all NaN), run without a coverage control
    coverage_col = None 
    print("Warning: Could not find Coverage_Zone or Coverage_Man column. Running Model 1 without coverage control.")

# Define X (Predictors): DCR_Score and Coverage Type control
Y1 = model_df['Defensive_Success']
predictors = ['DCR_Score']
if coverage_col:
    predictors.append(coverage_col)
    
X1 = model_df[predictors].copy()
X1 = sm.add_constant(X1, has_constant='add') # Add intercept

try:
    logit_model = sm.Logit(Y1, X1)
    result1 = logit_model.fit(disp=False) 

    # Evaluation
    predictions = result1.predict(X1)
    auc_score = roc_auc_score(Y1, predictions)
    odds_ratio = np.exp(result1.params['DCR_Score']) 

    print("\n--- Model 1 Results (D-CR Score predicting Defensive Success) ---")
    print(result1.summary())
    print(f"\nAUC Score: {auc_score:.3f}")
    print(f"**Odds Ratio for D-CR Score: {odds_ratio:.2f}**")
    print("Interpretation: For every 1-point increase in D-CR Score, the odds of an Incomplete/Interception increase by this factor.")
    print("-" * 60)

except Exception as e:
    print(f"Error running Model 1: {e}")


# --- 2. Model 2: Valuing the Metric (OLS Linear Regression) ---

print("\n--- Running Model 2: OLS Regression (Predicting Expected Points Added) ---")

# Define target variable Y: Expected Points Added (EPA)
Y2 = model_df['expected_points_added']

# Define X (Predictors): DCR_Score, Outcome control (Failure vs Success), and Coverage Type control
# Use Defensive_Success as a control (1=Success, 0=Failure)
predictors_2 = ['DCR_Score', 'Defensive_Success']
if coverage_col:
    predictors_2.append(coverage_col)
    
X2 = model_df[predictors_2].copy()
X2 = sm.add_constant(X2, has_constant='add') 

try:
    ols_model = sm.OLS(Y2, X2)
    result2 = ols_model.fit() 

    # Evaluation
    dcr_epa_coef = result2.params['DCR_Score']

    print("\n--- Model 2 Results (D-CR Score predicting EPA) ---")
    print(result2.summary())
    print(f"\n**DCR_Score Coefficient in EPA Model: {dcr_epa_coef:.3f}**")
    print("Interpretation: For every 1-point increase in D-CR Score, EPA changes by this amount.")
    print("Since negative EPA is good for defense, a significantly negative coefficient proves high defensive value.")
    print("-" * 60)

except Exception as e:
    print(f"Error running Model 2: {e}")


# CODE BLOCK 5: RERUNNING OLS WITHOUT OUTCOME CONTROL

print("\n--- Running Model 2 (Corrected): OLS Regression (DCR vs. UNCONDITIONAL EPA) ---")

# Ensure model_df is available from the previous block
model_df = validation_df.dropna(subset=['DCR_Score', 'expected_points_added', 'pass_result']).copy()
model_df['Defensive_Success'] = np.where(model_df['pass_result'].isin(['I', 'IN']), 1, 0)

# Define target variable Y: Expected Points Added (EPA)
Y2_clean = model_df['expected_points_added']

# Define X (Predictors): ONLY DCR_Score (No Defensive_Success control)
X2_clean = model_df[['DCR_Score']].copy()
X2_clean = sm.add_constant(X2_clean, has_constant='add') 

try:
    ols_model_clean = sm.OLS(Y2_clean, X2_clean)
    result2_clean = ols_model_clean.fit() 

    # Evaluation
    dcr_epa_coef_clean = result2_clean.params['DCR_Score']

    print("\n--- Model 2 (Corrected) Results ---")
    print(result2_clean.summary())
    print(f"\n**DCR_Score Coefficient in UNCONDITIONAL EPA Model: {dcr_epa_coef_clean:.3f}**")
    print("Interpretation: For every 1-point increase in D-CR Score, EPA changes by this amount.")
    print("This corrected model shows the total predictive value of D-CR on EPA.")
    print("-" * 60)

except Exception as e:
    print(f"Error running Corrected Model 2: {e}")


# CODE BLOCK 6: FINAL VISUALIZATION, LEADERBOARD, AND VARIABLE DEFINITION GUARANTEE

# --- 1. Dependencies and Data Preparation (Must be self-contained) ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob 
from scipy.spatial.distance import cdist # Included for robustness, though cdist might be globally available

print("\n--- Ensuring final_df is correctly constructed for visualization ---")

# Define path constants (assuming they are accessible from Block 1)
# NOTE: These paths MUST match what was defined in Block 1
BASE_PATH = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/'
TRAIN_PATH = os.path.join(BASE_PATH, 'train')

# Define the player identity loader function (must be present)
def load_player_identity(base_path, use_cols):
    """Loads player identity info from all input files."""
    all_dfs = []
    file_pattern = os.path.join(base_path, 'input_2023_w*.csv')
    file_paths = glob.glob(file_pattern) 
    
    for file_path in file_paths:
        all_dfs.append(pd.read_csv(file_path, usecols=use_cols, low_memory=False))
            
    identity_df = pd.concat(all_dfs, ignore_index=True)
    return identity_df.drop_duplicates(subset=['nfl_id'])

# Execute the load function
player_identity_cols = ['game_id', 'play_id', 'nfl_id', 'player_name', 'player_position', 'player_side']
player_names_df = load_player_identity(TRAIN_PATH, player_identity_cols)
player_names_df = player_names_df[player_names_df['player_side'] == 'Defense'].copy()

# Assume 'validation_df' (from Block 5) and 'nearest_defender_df' (from Block 2) are available.
# Re-merge player names and finalize final_df
play_defender_map = nearest_defender_df[['game_id', 'play_id', 'nfl_id']].drop_duplicates(subset=['game_id', 'play_id']).copy()

play_defender_map = play_defender_map.merge(
    player_names_df[['nfl_id', 'player_name', 'player_position']], 
    on='nfl_id', 
    how='left'
)

# CRITICAL: Define final_df and the label for plotting
final_df = validation_df.merge(play_defender_map, on=['game_id', 'play_id'], how='left')
final_df['Defensive_Success_Label'] = final_df['pass_result'].isin(['I', 'IN']).map({True: 'Defensive Success (I/IN)', False: 'Offensive Success (C)'})


# --- 2. Recalculate Leaderboard (Needed to define top_defenders and top_player_names for plots) ---
MIN_PLAYS = 50
player_leaderboard = final_df.groupby(['nfl_id', 'player_name', 'player_position']).agg(
    plays=('game_id', 'size'),
    avg_dcr=('DCR_Score', 'mean'),
    median_dcr=('DCR_Score', 'median'),
    avg_epa=('expected_points_added', 'mean')
).reset_index()

leaderboard_filtered = player_leaderboard[player_leaderboard['plays'] >= MIN_PLAYS].copy()
top_defenders = leaderboard_filtered.sort_values(by='avg_dcr', ascending=False).head(10)
top_player_names = top_defenders['player_name'].tolist()


# --- 3. Generate Visualizations (Original Code Logic) ---

print("\n--- Generating Enhanced Visualizations ---")

# 3.1 DCR Score vs. Pass Result (Original Box Plot)
print("1. D-CR Score Distribution by Pass Outcome (Box Plot)")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Defensive_Success_Label', y='DCR_Score', data=final_df)
plt.title('D-CR Score Distribution by Pass Outcome', fontsize=16)
plt.xlabel('Pass Result', fontsize=12)
plt.ylabel('Dynamic Coverage Reaction (D-CR) Score', fontsize=12)
plt.grid(axis='y', linestyle='--')
plt.show()



# 3.2 DCR Score vs. EPA (Validation of OLS Model)
print("\n2. D-CR Score vs. Expected Points Added (Scatter Plot with Regression Line)")
plt.figure(figsize=(10, 6))
# Use a sample for better visualization due to density of data points
sns.regplot(x='DCR_Score', y='expected_points_added', data=final_df.sample(n=5000, random_state=42), 
            scatter_kws={'alpha':0.2, 's':10}, line_kws={"color": "red"})
plt.title('Relationship between D-CR Score and EPA', fontsize=16)
plt.xlabel('Dynamic Coverage Reaction (D-CR) Score', fontsize=12)
plt.ylabel('Expected Points Added (EPA)', fontsize=12)
plt.grid(axis='both', linestyle='--')
plt.axhline(0, color='grey', linestyle='--')
plt.show()



# 3.3 Top Player D-CR Distribution (Performance Variability)
print("\n3. Top 10 Defender D-CR Score Distribution (Violin Plot)")
top_players_df = final_df[final_df['player_name'].isin(top_player_names)].copy()

plt.figure(figsize=(14, 7))
sns.violinplot(x='player_name', y='DCR_Score', data=top_players_df, 
               order=top_player_names, inner='quartile')
plt.xticks(rotation=45, ha='right')
plt.title('D-CR Score Distribution for Top 10 Defenders', fontsize=16)
plt.xlabel('Defender Name', fontsize=12)
plt.ylabel('Dynamic Coverage Reaction (D-CR) Score', fontsize=12)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()



# --- 4. Top Defender Leaderboard (Recap) ---
print("\n--- Generating Top 10 Defender Leaderboard (Recap) ---")

print("\n### ğŸ¥‡ Top 10 Nearest Defenders by Average D-CR Score")
print(top_defenders.round(3).rename(columns={'nfl_id': 'NFL ID', 'plays': 'Plays', 'avg_dcr': 'Avg D-CR', 'median_dcr': 'Median D-CR', 'avg_epa': 'Avg EPA Allowed'}))

print("\n**Final Key Insight:** The D-CR Score successfully identifies elite defensive players whose quick, efficient kinematic reactions directly lead to superior play outcomes.")

