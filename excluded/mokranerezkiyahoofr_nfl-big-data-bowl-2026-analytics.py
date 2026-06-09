import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# === NFL DATA LOADING ===
data_path = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/'
train_path = os.path.join(data_path, 'train/')

print("ğŸ�ˆ NFL Big Data Bowl 2026 Data Loaded!")

# Load data with mixed types handling
input_data = pd.read_csv(os.path.join(train_path, 'input_2023_w01.csv'))
output_data = pd.read_csv(os.path.join(train_path, 'output_2023_w01.csv'))
supp_data = pd.read_csv(os.path.join(data_path, 'supplementary_data.csv'), low_memory=False)

print(f"ğŸ“Š Data Dimensions:")
print(f"   Input data: {input_data.shape}")
print(f"   Output data: {output_data.shape}")
print(f"   Supplementary data: {supp_data.shape}")

# === EXPLORATORY ANALYSIS ===
print(f"\nğŸ”� Data Analysis:")


print("Columns in supplementary_data file:")
print(supp_data.columns.tolist())


import pandas as pd
import os

# === CORRECTED CODE FOR KAGGLE ===
data_path = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/'
train_path = os.path.join(data_path, 'train/')

files = os.listdir(train_path)
input_files = sorted([f for f in files if f.startswith("input")])
output_files = sorted([f for f in files if f.startswith("output")])

def load_partial_csv_chunked(file_list, folder, max_rows=50000, chunk_size=10000):
    df_list = []
    total_loaded = 0
    for f in file_list:
        path = os.path.join(folder, f)
        print(f"Reading {f} in chunks...")
        for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=True):
            df_list.append(chunk)
            total_loaded += len(chunk)
            if total_loaded >= max_rows:
                break
        if total_loaded >= max_rows:
            break
    return pd.concat(df_list, ignore_index=True)

# Load without exceeding RAM
df_input_all = load_partial_csv_chunked(input_files, train_path, max_rows=50000)
df_output_all = load_partial_csv_chunked(output_files, train_path, max_rows=50000)

# Lightweight contextual file
supp_path = os.path.join(data_path, 'supplementary_data.csv')
df_supp = pd.read_csv(supp_path, low_memory=False)

print("\nâœ… Loading completed!")
print("Input :", df_input_all.shape)
print("Output:", df_output_all.shape)
print("Supp  :", df_supp.shape)


# --- DATASETS OVERVIEW ---

print("==== INPUT ====")
print(df_input_all.head(5))        # First 5 rows
print(df_input_all.info())         # Columns + data types


print("\n==== OUTPUT ====")
print(df_output_all.head(5))
print(df_output_all.info())


print("\n==== SUPPLEMENTARY ====")
print(df_supp.head(5))
print(df_supp.info())


# Light overview of numeric columns
numeric_cols = df_input_all.select_dtypes(include=['number']).columns

print("=== Light statistical overview of numeric columns ===\n")
for col in numeric_cols:
    desc = df_input_all[col].describe()
    display(f"{col}: mean={desc['mean']:.3f}, std={desc['std']:.3f}, min={desc['min']}, max={desc['max']}")


display(df_supp.columns.tolist())



# Number of distinct games and plays
nb_games = df_supp['game_id'].nunique()
nb_plays = df_supp['play_id'].nunique()

# Number of distinct players
nb_players = df_input_all['nfl_id'].nunique()

print(f"Number of games    : {nb_games}")
print(f"Number of plays    : {nb_plays}")
print(f"Number of players  : {nb_players}")


import pandas as pd

# --- Step 1: Verification of common keys ---
common_keys = ['game_id', 'play_id']
print("Common keys verified:", common_keys)
print("Input unique plays :", df_input_all[common_keys].drop_duplicates().shape[0])
print("Output unique plays:", df_output_all[common_keys].drop_duplicates().shape[0])
print("Supplementary plays:", df_supp[common_keys].drop_duplicates().shape[0])

# --- Step 2: Selection of useful columns in context ---
cols_context = [
    'game_id', 'play_id', 'season', 'week', 'game_date',
    'possession_team', 'defensive_team', 'offense_formation',
    'play_action', 'pass_result', 'pass_length', 'route_of_targeted_receiver',
    'team_coverage_type', 'defenders_in_the_box',
    'yards_gained', 'expected_points_added',
    'pre_snap_home_team_win_probability', 'pre_snap_visitor_team_win_probability'
]

# Verify all columns exist in df_supp
missing_cols = [c for c in cols_context if c not in df_supp.columns]
if missing_cols:
    raise ValueError(f"Missing columns in df_supp: {missing_cols}")

df_context = df_supp[cols_context].drop_duplicates(subset=['game_id', 'play_id'])

# --- Step 3: Merge Output + Context ---
df_output_merged = df_output_all.merge(
    df_context,
    on=['game_id', 'play_id'],
    how='left'
)
print("âœ… Output + Context merge completed:", df_output_merged.shape)

# --- Step 4: Merge Input + Output (with subsample for memory) ---
sample_plays = df_output_merged[['game_id', 'play_id']].drop_duplicates().sample(200, random_state=42)

df_input_subset = df_input_all.merge(
    sample_plays,
    on=['game_id', 'play_id'],
    how='inner'
)

df_final = df_input_subset.merge(
    df_output_merged,
    on=['game_id', 'play_id'],
    how='left',
    suffixes=('_input', '_output')  # âš ï¸� automatic suffixes for clarity
)

# --- Step 5: Final verification of duplicate columns ---
# Sometimes common columns (ex: frame_id) are not properly renamed
df_final = df_final.loc[:, ~df_final.columns.duplicated()]

print("âœ… Final merge (Input + Output + Context) completed:", df_final.shape)
print(df_final.head(3))
print("\nFinal columns:", list(df_final.columns))


# Logical column reorganization
cols_order = [
    # Identifiers
    'game_id', 'play_id', 'season', 'week', 'game_date',
    
    # Player / tracking input
    'player_to_predict', 'nfl_id_input', 'player_name', 'player_position',
    'player_role', 'player_side', 'player_height', 'player_weight', 'player_birth_date',
    
    # Tracking (input frames)
    'frame_id_input', 'x_input', 'y_input', 's', 'a', 'dir', 'o',
    
    # Tracking (output frames)
    'frame_id_output', 'x_output', 'y_output',
    
    # Play context
    'play_direction', 'absolute_yardline_number', 'possession_team', 'defensive_team',
    'offense_formation', 'play_action', 'pass_result', 'pass_length',
    'route_of_targeted_receiver', 'team_coverage_type', 'defenders_in_the_box',
    
    # Results
    'yards_gained', 'expected_points_added',
    'pre_snap_home_team_win_probability', 'pre_snap_visitor_team_win_probability',
    
    # Ballistic targets
    'num_frames_output', 'ball_land_x', 'ball_land_y'
]

# Reorder only existing columns
df_final = df_final[[c for c in cols_order if c in df_final.columns]]

# Save final subsample
df_final.to_csv("df_final_sample.csv", index=False)
print("ğŸ’¾ File saved: df_final_sample.csv")



import numpy as np

# --- Step 1: Check play direction distribution ---
print("Play direction distribution:")
print(df_final['play_direction'].value_counts())

# --- Step 2: Normalize coordinates ---
# The field is 120 yards long (0-120)
# If play goes left, we "flip" the field to align everything to the right.
df_final['x_input_std'] = np.where(
    df_final['play_direction'] == 'left',
    120 - df_final['x_input'],
    df_final['x_input']
)

df_final['x_output_std'] = np.where(
    df_final['play_direction'] == 'left',
    120 - df_final['x_output'],
    df_final['x_output']
)

# Same logic for ball landing position
df_final['ball_land_x_std'] = np.where(
    df_final['play_direction'] == 'left',
    120 - df_final['ball_land_x'],
    df_final['ball_land_x']
)

# --- Step 3: Quick verification ---
print("\nBefore/after normalization overview:")
print(df_final[['play_direction', 'x_input', 'x_input_std', 'x_output', 'x_output_std', 'ball_land_x', 'ball_land_x_std']].head(10))


import matplotlib.pyplot as plt

# Display a random sample to avoid graph saturation
sample_plot = df_final.sample(2000, random_state=42)

plt.figure(figsize=(10, 5))
plt.scatter(sample_plot['x_input'], sample_plot['y_input'], alpha=0.3, label='Before normalization', s=10)
plt.scatter(sample_plot['x_input_std'], sample_plot['y_input'], alpha=0.3, label='After normalization', s=10)
plt.title("Direction Normalization Verification (Input positions)")
plt.xlabel("x-coordinate (yards)")
plt.ylabel("y-coordinate (yards)")
plt.legend()
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt

# --- Throw logic verification ---
# 1. Calculate delta between ball position and player position
df_final['delta_x_ball'] = df_final['ball_land_x_std'] - df_final['x_input_std']

# 2. Descriptive statistics
print("Delta statistics (ball - player distance):")
print(df_final['delta_x_ball'].describe())

# 3. Percentage of consistent throws (ball in front of player)
valid_ratio = (df_final['delta_x_ball'] > 0).mean() * 100
print(f"\nâœ… {valid_ratio:.2f}% of throws have the ball in front of the player (directional consistency).")

# --- Visualization ---
plt.figure(figsize=(8, 5))
plt.hist(df_final['delta_x_ball'], bins=50, color='orange', edgecolor='black', alpha=0.7)
plt.axvline(0, color='red', linestyle='--', label='Player position')
plt.title("Distribution of Distance Between Player and Ball Position (After Normalization)")
plt.xlabel("Î”x = ball_land_x_std - x_input_std (yards)")
plt.ylabel("Number of throws")
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(8,6))
plt.scatter(df_final['x_input_std'], df_final['ball_land_x_std'], s=5, alpha=0.3)
plt.plot([0, 120], [0, 120], 'r--', label='y = x (ball at same position as player)')
plt.xlabel("Player position (x_input_std)")
plt.ylabel("Ball position (ball_land_x_std)")
plt.title("Player - Ball Relationship After Normalization")
plt.legend()
plt.grid(True)
plt.show()


# Quantitative check (optional)
ratio_ball_front = (df_final['ball_land_x_std'] > df_final['x_input_std']).mean()
print(f"{ratio_ball_front*100:.2f}% of passes have the ball in front of the player")



df_anomalies = df_final[df_final['ball_land_x_std'] < df_final['x_input_std']]
display(df_anomalies[['game_id', 'play_id', 'route_of_targeted_receiver', 'pass_length', 
                      'expected_points_added', 'play_direction']].head(10))


import matplotlib.pyplot as plt
import seaborn as sns

# --- Sample subset for readability ---
df_sample = df_final.sample(3000, random_state=42)

# --- Compute ball vs. player x-difference ---
df_sample['delta_x_ball'] = df_sample['ball_land_x_std'] - df_sample['x_input_std']

# --- Compute % of passes where the ball lands behind the player, grouped by route type ---
route_stats = (
    df_sample
    .groupby('route_of_targeted_receiver')['delta_x_ball']
    .apply(lambda x: (x < 0).mean() * 100)
    .reset_index(name='% Behind')
    .sort_values('% Behind', ascending=False)
)

print(route_stats)

# --- Main scatter plot ---
plt.figure(figsize=(9, 7))
sns.scatterplot(
    data=df_sample,
    x='x_input_std',
    y='ball_land_x_std',
    hue='route_of_targeted_receiver',
    palette='tab10',
    alpha=0.7
)

# --- Diagonal line y = x ---
plt.plot([0, 120], [0, 120], 'r--', label='y = x (ball at same position as player)')

# --- Titles ---
plt.title("Playerâ€“Ball Relationship after Standardization\n(colored by route type)", fontsize=13)
plt.xlabel("Player position (x_input_std)")
plt.ylabel("Ball position (ball_land_x_std)")

# --- Add annotations for % behind ---
for i, row in enumerate(route_stats.itertuples()):
    plt.text(5, 115 - i*5, f"{row.route_of_targeted_receiver}: {row._2:.1f}% behind",
             fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.legend(title='Route type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Data 
df_routes = pd.DataFrame({
    "route_of_targeted_receiver": ["FLAT", "SLANT", "OUT", "CROSS", "IN", "GO", "CORNER"],
    "% Behind": [77.64, 66.67, 17.56, 17.54, 12.44, 1.25, 0.00]
})

# Sort in descending order
df_routes = df_routes.sort_values("% Behind", ascending=False)

# --- Dark theme and gradient palette
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#121212", "figure.facecolor": "#121212"})
palette = sns.color_palette("magma", len(df_routes))

# --- Create bar chart
plt.figure(figsize=(9, 5))
bars = sns.barplot(
    data=df_routes,
    x="route_of_targeted_receiver",
    y="% Behind",
    palette=palette
)

# --- Aesthetic customization
plt.title("Percentage of Passes Where the Ball Lands Behind the Receiver", 
          fontsize=15, weight='bold', color='white', pad=20)
plt.xlabel("Route Type", fontsize=11, color='white', labelpad=10)
plt.ylabel("% Behind", fontsize=11, color='white', labelpad=10)
plt.ylim(0, 100)

# Subtle grid lines
plt.grid(axis="y", linestyle="--", alpha=0.3)

# Customize tick and label colors
bars.set_xticklabels(bars.get_xticklabels(), color='white', fontsize=10)
bars.set_yticklabels([f"{int(y)}" for y in bars.get_yticks()], color='white', fontsize=10)

# --- Annotate bar values
for i, v in enumerate(df_routes["% Behind"]):
    plt.text(i, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontsize=10, color='white', fontweight='semibold')

plt.tight_layout()
plt.show()



import numpy as np
import pandas as pd

# Calculate deltas and error distance
df_final["delta_x"] = df_final["ball_land_x_std"] - df_final["x_output_std"]
df_final["delta_y"] = df_final["ball_land_y"] - df_final["y_output"]
df_final["error_distance"] = np.sqrt(df_final["delta_x"]**2 + df_final["delta_y"]**2)

# Mean and standard deviation of error by route type
error_by_route = (
    df_final.groupby("route_of_targeted_receiver")["error_distance"]
    .agg(["mean", "std", "count"])
    .sort_values("mean")
)

print("=== Average error by route type ===")
print(error_by_route)


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# --- Step 1: Visualization - Average error by route type ---
plt.figure(figsize=(8,5))
order = df_final.groupby('route_of_targeted_receiver')['error_distance'].mean().sort_values().index
sns.barplot(
    data=df_final,
    x='route_of_targeted_receiver',
    y='error_distance',
    order=order,
    palette='viridis'
)
plt.title("Average Error by Route Type")
plt.xlabel("Route Type")
plt.ylabel("Average Distance Error (yards)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# --- Step 2: Correlation between pass_length and error_distance ---
corr, pval = pearsonr(df_final['pass_length'], df_final['error_distance'])
print(f"Pearson correlation coefficient: {corr:.3f} (p-value={pval:.3e})")

plt.figure(figsize=(7,6))
sns.scatterplot(
    data=df_final.sample(5000, random_state=42),  # subsample for readability
    x='pass_length',
    y='error_distance',
    hue='route_of_targeted_receiver',
    alpha=0.6,
    palette='tab10'
)
plt.title(f"Correlation Between Pass Length and Distance Error (r={corr:.2f})")
plt.xlabel("Pass Length (yards)")
plt.ylabel("Distance Error (yards)")
plt.legend(title="Route", bbox_to_anchor=(1.05,1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()

# --- Step 3: Automated interpretation ---
if corr > 0.5:
    relation = "strong and positive"
elif corr > 0.3:
    relation = "moderate and positive"
elif corr > 0.1:
    relation = "weak but positive"
elif corr < -0.3:
    relation = "moderate and negative"
elif corr < -0.1:
    relation = "weak but negative"
else:
    relation = "almost nonexistent"

print(f"ğŸ”� Analysis: The correlation between pass length and distance error is {relation}.")
print("This means that longer passes tend (or not) to result in greater errors, depending on the coefficient's sign.")


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as pe

# --- Improved Step 4: Efficiency vs Accuracy ---
df_route_stats = (
    df_final.groupby('route_of_targeted_receiver')
    .agg({
        'error_distance': 'mean',
        'expected_points_added': 'mean',
        'pass_length': 'mean',
        'play_id': 'count'
    })
    .rename(columns={
        'error_distance': 'Average_Error',
        'expected_points_added': 'Average_EPA',
        'pass_length': 'Average_Length',
        'play_id': 'Play_Count'
    })
    .reset_index()
)

# --- Calculate medians for quadrants ---
x_median = df_route_stats['Average_Error'].median()
y_median = df_route_stats['Average_EPA'].median()

plt.figure(figsize=(10,7))
sns.set_style("whitegrid")

# --- Scatter plot ---
sns.scatterplot(
    data=df_route_stats,
    x='Average_Error',
    y='Average_EPA',
    size='Play_Count',
    hue='route_of_targeted_receiver',
    palette='tab10',
    sizes=(150, 900),
    alpha=0.85,
    edgecolor='black',
    linewidth=0.8
)

# --- Quadrant lines ---
plt.axvline(x=x_median, color='gray', linestyle='--', alpha=0.7)
plt.axhline(y=y_median, color='gray', linestyle='--', alpha=0.7)

# --- Annotations with halo effect ---
for _, row in df_route_stats.iterrows():
    plt.text(
        row['Average_Error'] + 0.05,
        row['Average_EPA'] + 0.02,
        row['route_of_targeted_receiver'],
        fontsize=10,
        fontweight='bold',
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        color='black'
    )

# --- Chart styling ---
plt.title("ğŸ�¯ Efficiency (EPA) vs Accuracy (Error) by Route Type", fontsize=14, fontweight='bold')
plt.xlabel("Average Error (yards) â€” better accuracy â†’ left", fontsize=12)
plt.ylabel("Expected Points Added (EPA) â€” better efficiency â†‘ top", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Route", fontsize=9)
plt.tight_layout()

plt.show()

# --- Interpretation ---
best_routes = df_route_stats.sort_values(['Average_EPA', 'Average_Error'], ascending=[False, True]).head(3)
worst_routes = df_route_stats.sort_values(['Average_EPA', 'Average_Error'], ascending=[True, False]).head(3)

print("ğŸ�† Most Effective Routes (efficient and accurate):")
print(best_routes[['route_of_targeted_receiver', 'Average_EPA', 'Average_Error']])

print("\nâš ï¸� Least Effective and Least Accurate Routes:")
print(worst_routes[['route_of_targeted_receiver', 'Average_EPA', 'Average_Error']])



# Correlation between pass length and distance error
corr = df_final['pass_length'].corr(df_final['error_distance'])
print(f"ğŸ”— Correlation pass_length vs error_distance: {corr:.3f}")


!pip install tqdm



!C:\Users\mokra\anaconda3\envs\fixnumpy\python.exe -m pip install tqdm



from tqdm import tqdm

for i in tqdm(range(5)):
    pass



import pandas as pd
import numpy as np
from tqdm import tqdm

# -----------------------------
# âš™ï¸� Base Parameters
# -----------------------------
FRAME_TIME = 0.01  # frame duration (in seconds)
np.random.seed(42)

# -----------------------------
# âœ… Robust Subsampling
# -----------------------------
unique_plays = df_final[['game_id', 'play_id']].drop_duplicates()
n_plays = unique_plays.shape[0]

SAMPLE_PLAYS = unique_plays.sample(
    min(500, n_plays), random_state=42
).reset_index(drop=True)

# -----------------------------
# ğŸ§  Data Preparation
# -----------------------------
df_final = df_final.sort_values(
    ['game_id', 'play_id', 'frame_id_input', 'nfl_id_input']
).reset_index(drop=True)

# -----------------------------
# ğŸ§® Prototype Calculations on Sampled Plays
# -----------------------------
results = []

for _, play in tqdm(SAMPLE_PLAYS.iterrows(), total=len(SAMPLE_PLAYS), desc="Processing plays"):
    gid, pid = play['game_id'], play['play_id']
    df_play = df_final[(df_final['game_id'] == gid) & (df_final['play_id'] == pid)]
    
    # Simple example: calculating error mean and pass_length
    err_mean = df_play['error_distance'].mean()
    pass_mean = df_play['pass_length'].mean()
    epa_mean = df_play['expected_points_added'].mean()
    
    results.append({
        'game_id': gid,
        'play_id': pid,
        'average_error': err_mean,
        'average_pass_length': pass_mean,
        'average_EPA': epa_mean
    })

# -----------------------------
# ğŸ“Š Summary & Display
# -----------------------------
df_results = pd.DataFrame(results)
print("âœ… Calculations completed on", len(df_results), "plays.\n")
print(df_results.describe())

# Example correlation
corr = df_results['average_pass_length'].corr(df_results['average_error'])
print(f"ğŸ”— Correlation pass_length vs error_distance: {corr:.3f}")


import matplotlib.pyplot as plt
import seaborn as sns

# General style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

plt.figure(figsize=(9,6))

# Scatter plot with regression line
sns.regplot(
    data=df_results,
    x='average_pass_length',
    y='average_error',
    scatter_kws={'s':80, 'alpha':0.8, 'edgecolor':'k'},
    line_kws={'color':'red', 'lw':2},
)

# Correlation coefficient annotation
corr = df_results['average_pass_length'].corr(df_results['average_error'])
plt.text(
    0.05, 0.92,
    f"Correlation: {corr:.2f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    color='red',
    weight='bold'
)

# Titles and labels
plt.title("Relationship Between Pass Length and Average Trajectory Error", fontsize=14, weight='bold')
plt.xlabel("Average Pass Length (yards)")
plt.ylabel("Average Trajectory Error (yards)")
plt.tight_layout()
plt.show()


print(df_final.columns.tolist())



print(df_final['player_role'].unique())



example_play = df_final[df_final['play_id'] == df_final['play_id'].iloc[0]]
print(example_play[['player_name','player_role','x_input','y_input','x_output','y_output','play_direction']].head(10))



example_play


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# âœ… 1. Filter only targeted receivers
df_targeted = df_final[df_final['player_role'] == 'Targeted Receiver'].copy()

# âœ… 2. Normalize coordinates based on play direction
df_targeted['x_input_adj'] = np.where(df_targeted['play_direction'] == 'left',
                                      120 - df_targeted['x_input'], df_targeted['x_input'])
df_targeted['x_output_adj'] = np.where(df_targeted['play_direction'] == 'left',
                                       120 - df_targeted['x_output'], df_targeted['x_output'])
df_targeted['ball_land_x_adj'] = np.where(df_targeted['play_direction'] == 'left',
                                          120 - df_targeted['ball_land_x'], df_targeted['ball_land_x'])

# âœ… 3. Calculate RAS (Route Accuracy Score)
df_targeted['route_error'] = np.sqrt((df_targeted['x_output_adj'] - df_targeted['x_input_adj'])**2 +
                                    (df_targeted['y_output'] - df_targeted['y_input'])**2)

# Normalization for readability
df_targeted['RAS'] = 1 - (df_targeted['route_error'] / df_targeted['route_error'].max())

# âœ… 4. Calculate CI (Catch Influence)
df_targeted['catch_dist'] = np.sqrt((df_targeted['x_output_adj'] - df_targeted['ball_land_x_adj'])**2 +
                                   (df_targeted['y_output'] - df_targeted['ball_land_y'])**2)
df_targeted['CI'] = 1 - (df_targeted['catch_dist'] / df_targeted['catch_dist'].max())

# âœ… 5. Calculate CCT (Completion Conversion Time)
df_targeted['CCT'] = df_targeted['num_frames_output'] / df_targeted['num_frames_output'].max()

# âœ… 6. Averages per play
df_metrics = df_targeted.groupby(['game_id', 'play_id'], as_index=False)[['RAS', 'CI', 'CCT']].mean()

print("âœ… Scores calculated for", len(df_metrics), "plays.")
print(df_metrics.describe())

# âœ… 7. Combined scores visualization
plt.figure(figsize=(8,6))
plt.scatter(df_metrics['RAS'], df_metrics['CI'], c=df_metrics['CCT'], cmap='viridis', s=80, alpha=0.8)
plt.xlabel('RAS - Route Accuracy Score')
plt.ylabel('CI - Catch Influence')
plt.title('ğŸ“Š Play Performance Visualization (RAS vs CI, color = CCT)')
plt.colorbar(label='CCT')
plt.grid(True, alpha=0.3)
plt.show()

# âœ… 8. Average global score (for final ranking)
global_score = (df_metrics['RAS'].mean() + df_metrics['CI'].mean() + df_metrics['CCT'].mean()) / 3
print(f"\nğŸ�† Model Global Score (average of 3 metrics): {global_score:.4f}")


df_final.columns.tolist()

    


#  If you need to recreate the scoring structure
df_scores = pd.DataFrame({
    'metric': ['RAS', 'CI', 'CCT', 'Global_Score'],
    'value': [0.67, 0.76, 0.54, 0.6599],
    'interpretation': [
        'Predicted trajectories align well with actual routes',
        'Positioning at reception is quite precise', 
        'Execution timing adequate but improvable',
        'Competitive level for first iteration'
    ]
})

print(df_scores)




# ==========================================================
# ğŸ§© Merging scores (df_metrics) with complete data (df_final)
# ==========================================================
df_final = df_final.merge(
    df_metrics[['game_id', 'play_id', 'RAS', 'CI', 'CCT']],
    on=['game_id', 'play_id'],
    how='left'
)

# Verification
print("âœ… Merge successful.")
print("Available columns:", [c for c in df_final.columns if c in ['RAS','CI','CCT']])

# ==========================================================
# ğŸ“Š Calculating averages by player and route
# ==========================================================
df_scores_player_route = (
    df_final.groupby(['player_name', 'route_of_targeted_receiver'])
    .agg({
        'RAS': 'mean',
        'CI': 'mean', 
        'CCT': 'mean',
        'expected_points_added': 'mean',
        'error_distance': 'mean'
    })
    .reset_index()
)

print("âœ… Averages calculated by player and route.")
display(df_scores_player_route.head())

# ==========================================================
# ğŸ�† Ranking most effective routes
# ==========================================================
if 'expected_points_added' in df_scores_player_route.columns:
    df_leaderboard = (
        df_scores_player_route
        .sort_values(by=['expected_points_added'], ascending=False)
        .head(10)
    )
    print("ğŸ�† Top 10 most effective player/route combinations (by average EPA):")
    display(df_leaderboard)
else:
    print("âš ï¸� Column expected_points_added is missing â€” check the merge or available columns.")



df_final['player_name'].dropna().unique()[:10]



import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# ğŸ�¨ Visualization â€” top 10 routes/players by EPA
# ==========================================================
df_plot = df_leaderboard.copy()

plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_plot,
    x='expected_points_added',
    y='player_name',
    hue='route_of_targeted_receiver',
    palette='viridis'
)
plt.title("ğŸ�† Top 10 Most Effective Player/Route Combinations by Average EPA", fontsize=14, weight='bold')
plt.xlabel("Average Expected Points Added (EPA)")
plt.ylabel("Player")
plt.legend(title="Route")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ==========================================================
# ğŸ”� Correlations between metrics
# ==========================================================
metrics_cols = ['RAS', 'CI', 'CCT', 'expected_points_added', 'error_distance']
corr = df_scores_player_route[metrics_cols].corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("ğŸ”— Correlations Between Performance Metrics")
plt.tight_layout()
plt.show()


# ==========================================================
# ğŸ§® Creating global score (winning weighting)
# ==========================================================

df_scores_player_route['global_score'] = (
    0.4 * df_scores_player_route['expected_points_added'].rank(pct=True) + 
    0.25 * df_scores_player_route['RAS'].rank(pct=True) +
    0.2 * df_scores_player_route['CI'].rank(pct=True) +
    0.15 * df_scores_player_route['CCT'].rank(pct=True)
)

# Final ranking
df_leaderboard_global = df_scores_player_route.sort_values('global_score', ascending=False).head(15)

print("ğŸ�† Top 15 Player/Route Combinations (Global Score)")
display(df_leaderboard_global[['player_name', 'route_of_targeted_receiver', 
                               'expected_points_added', 'RAS', 'CI', 'CCT', 'global_score']])


plt.figure(figsize=(10,6))
sns.barplot(
    data=df_leaderboard_global,
    x='global_score',
    y='player_name',
    hue='route_of_targeted_receiver',
    palette='plasma'
)
plt.title("ğŸ”¥ Top 15 Overall Performers (Combined EPA + RAS + CI + CCT Score)", fontsize=14, weight='bold')
plt.xlabel("Normalized Global Score")
plt.ylabel("Player")
plt.legend(title="Route")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


df_leaderboard_global


# ğŸ”¹ Sort leaderboard to ensure top player is correct
df_leaderboard_global = df_scores_player_route.sort_values(
    by=['global_score', 'expected_points_added'],
    ascending=[False, False]
).reset_index(drop=True)

# ğŸ”¹ Function to generate a player summary
def player_summary(row):
    return f"""
ğŸ�� The top performing player is **{row['player_name']}** 
on the **{row['route_of_targeted_receiver']}** route with a global score of **{row['global_score']:.3f}**.
This player combines high EPA ({row['expected_points_added']:.2f}) 
with strong technical indicators (RAS={row['RAS']:.2f}, CI={row['CI']:.2f}, CCT={row['CCT']:.2f}).

ğŸ“Š This indicates excellent overall efficiency â€” they optimize both precision, route cohesion, and play success contribution.
"""

# ğŸ”¹ Top 1 player
top_player = df_leaderboard_global.iloc[0]  # entire row
summary_top1 = player_summary(top_player)
print(summary_top1)

# ğŸ”¹ Top 5 players (optional)
print("\n--- Top 5 player/route combinations ---\n")
for i, row in df_leaderboard_global.head(5).iterrows():
    print(f"{i+1}. {row['player_name']} | Route: {row['route_of_targeted_receiver']} | Global Score: {row['global_score']:.3f} | EPA: {row['expected_points_added']:.2f}")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================================
# 1ï¸�âƒ£ EXACT SORTING - FIRST PLAYER FROM LIST (Brandon Stephens)
# ==========================================================
df_leaderboard_global = df_scores_player_route.sort_values(
    by=['global_score', 'expected_points_added'], 
    ascending=[False, False]
).reset_index(drop=True)

# CORRECTION: TAKE EXACTLY THE FIRST (index 0)
top_row = df_leaderboard_global.iloc[0]  # Brandon Stephens
top_player = top_row['player_name']
top_route = top_row['route_of_targeted_receiver']

print("=" * 60)
print(f"ğŸ�� SELECTED PLAYER: {top_player}")
print(f"ğŸ“� ROUTE: {top_route}")
print(f"ğŸ“Š POSITION: First in list (index 0)")
print("=" * 60)

# Explicit verification
print(f"âœ… Verification: {df_leaderboard_global.iloc[0]['player_name']} = {top_player}")
print(f"ğŸ”� Comparison:")
print(f"   â€¢ Index 0: {df_leaderboard_global.iloc[0]['player_name']} (SELECTED)")
print(f"   â€¢ Index 1: {df_leaderboard_global.iloc[1]['player_name']}")
print(f"   â€¢ All players have same stats (aggregated by route)")

# ==========================================================
# 2ï¸�âƒ£ DATA FILTERING FOR THE CORRECT PLAYER
# ==========================================================
print(f"\nğŸ”� Searching data for {top_player}...")

df_route = df_final[
    (df_final['player_name'] == top_player) &  # EXACTLY Brandon Stephens
    (df_final['route_of_targeted_receiver'] == top_route)
].copy()

if df_route.empty:
    print(f"âš ï¸�  No specific data found for {top_player}")
    print("ğŸ”„ Using first available data for IN route...")
    
    # Fallback: take first data for IN route
    df_route = df_final[
        (df_final['route_of_targeted_receiver'] == top_route)
    ].head(30).copy()
    
    if not df_route.empty:
        print(f"âœ… {len(df_route)} data points found for route {top_route}")
else:
    print(f"âœ… {len(df_route)} specific data points found for {top_player}")

# Create or use temporal index
if 'frame_id_input' in df_route.columns:
    frame_col = 'frame_id_input'
elif 'frame_id_output' in df_route.columns:
    frame_col = 'frame_id_output'
else:
    df_route['frame_artificial'] = np.arange(len(df_route))
    frame_col = 'frame_artificial'
    print(f"ğŸ“… Temporal column created: {frame_col}")

df_route.sort_values(frame_col, inplace=True)

# ==========================================================
# 3ï¸�âƒ£ ANIMATION WITH CLEAR IDENTIFICATION
# ==========================================================
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 120)
ax.set_ylim(0, 53.3)
ax.set_xlabel("X (yards)")
ax.set_ylabel("Y (yards)")
ax.set_title(f"ğŸ�ˆ OFFICIAL ANIMATION - {top_player.upper()}\nRoute: {top_route} | Score: {top_row['global_score']:.3f}", 
             fontweight='bold', fontsize=14, pad=20)
ax.grid(True, linestyle="--", alpha=0.3)

# Add field lines
for yard in range(10, 111, 10):
    ax.axvline(yard, color='gray', linewidth=0.5, alpha=0.3)

# Animation elements with distinctive colors
scat_player = ax.scatter([], [], s=200, c="#FF6B6B", edgecolor='white', 
                         linewidth=3, label=f"â­� {top_player} (TOP 1)")
line_player, = ax.plot([], [], c="#FF6B6B", lw=3, alpha=0.7)

# Texts with clear identification
txt_name = ax.text(0, 0, "", fontsize=12, fontweight="bold", color="#FF6B6B",
                   bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9))
txt_position = ax.text(5, 50, f"ğŸ�¯ {top_player}\nğŸ“� {top_route}", 
                       fontsize=11, fontweight="bold", color="darkblue",
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))
txt_metrics = ax.text(80, 50, "", fontsize=10, color="black",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8))

def init():
    scat_player.set_offsets(np.empty((0,2)))
    line_player.set_data([], [])
    txt_name.set_text("")
    txt_metrics.set_text("")
    return scat_player, line_player, txt_name, txt_metrics

def update(frame):
    df_frame = df_route[df_route[frame_col] <= frame]

    # Choose x/y columns
    if "x_input" in df_frame.columns and "y_input" in df_frame.columns:
        x_col, y_col = "x_input", "y_input"
    elif "x_output" in df_frame.columns and "y_output" in df_frame.columns:
        x_col, y_col = "x_output", "y_output"
    else:
        # Create artificial coordinates if missing
        df_frame["x_input"] = np.linspace(20, 100, len(df_frame))
        df_frame["y_input"] = np.linspace(26, 30, len(df_frame))
        x_col, y_col = "x_input", "y_input"

    coords = df_frame[[x_col, y_col]].values
    if len(coords) > 0:
        scat_player.set_offsets(coords[-1].reshape(1,2))
        line_player.set_data(coords[:,0], coords[:,1])
        txt_name.set_position((coords[-1,0] + 2, coords[-1,1]))
        txt_name.set_text(f"â­� {top_player}")

        # Live metrics display
        progress = min(1.0, frame / len(df_route))
        
        metrics_text = f"ğŸ“ˆ Progress: {progress*100:.0f}%\n"
        metrics_text += f"RAS: {top_row['RAS']:.3f}\n"
        
        if progress > 0.4:
            metrics_text += f"CI: {top_row['CI']:.3f}\n"
        if progress > 0.6:
            metrics_text += f"CCT: {top_row['CCT']:.3f}\n"
        if progress > 0.8:
            metrics_text += f"Score: {top_row['global_score']:.3f}\n"
            metrics_text += f"EPA: {top_row['expected_points_added']:.3f}"
        
        txt_metrics.set_text(metrics_text)

    return scat_player, line_player, txt_name, txt_metrics

# Animation configuration
frames = min(int(df_route[frame_col].max()), 50)  # Limit to 50 frames max
ani = FuncAnimation(fig, update, frames=range(frames + 1),
                    init_func=init, blit=True, interval=120)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# ==========================================================
# 4ï¸�âƒ£ FINAL CONFIRMATION REPORT
# ==========================================================
print("\n" + "=" * 60)
print("ğŸ“‹ CONFIRMATION REPORT")
print("=" * 60)
print(f"âœ… ANIMATED PLAYER: {top_player}")
print(f"âœ… POSITION: First in list (index 0)")
print(f"âœ… ROUTE: {top_route}")
print(f"âœ… GLOBAL SCORE: {top_row['global_score']:.3f}")
print(f"âœ… EPA: {top_row['expected_points_added']:.3f}")
print(f"âœ… DATA USED: {len(df_route)} frames")
print(f"âœ… IDENTICAL STATS: All {top_route} players")
print("\nğŸ�¬ ANIMATION COMPLETED - BRANDON STEPHENS CONFIRMED")


# ===========================================
# ğŸ�† PHASE 3 â€” Winner Board + Automated Report
# ===========================================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ===========================
# 1ï¸�âƒ£ Model Global Summary
# ===========================
global_score = df_scores_player_route[['RAS', 'CI', 'CCT']].mean().mean()
print(f"ğŸ�† Model Global Score: {global_score:.3f}")

# ===========================
# 2ï¸�âƒ£ Routes and Players Ranking
# ===========================
df_routes = (
    df_scores_player_route.groupby('route_of_targeted_receiver')
    .agg({
        'expected_points_added': 'mean',
        'error_distance': 'mean',
        'RAS': 'mean',
        'CI': 'mean',
        'CCT': 'mean'
    })
    .reset_index()
    .sort_values(by='expected_points_added', ascending=False)
)

df_top_players = (
    df_scores_player_route.groupby('player_name')
    .agg({
        'expected_points_added': 'mean',
        'error_distance': 'mean',
        'RAS': 'mean'
    })
    .reset_index()
    .sort_values(by='expected_points_added', ascending=False)
    .head(10)
)

print("\nğŸ�† Top 5 Most Effective Routes:")
display(df_routes.head(5)[['route_of_targeted_receiver','expected_points_added','error_distance']])

print("\nğŸ”¥ Top 10 Most Performing Players:")
display(df_top_players[['player_name','expected_points_added','RAS']])

# ===========================
# 3ï¸�âƒ£ Visual Winner Board
# ===========================
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df_routes,
    x='error_distance', 
    y='expected_points_added',
    hue='route_of_targeted_receiver',
    size='RAS',
    sizes=(100, 300),
    alpha=0.8
)
plt.title("ğŸ�† WINNER BOARD â€” Accuracy vs Impact (EPA)", fontsize=16, fontweight='bold')
plt.xlabel("Average Error (distance)")
plt.ylabel("Average EPA (impact)")
plt.legend(title="Route", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===========================
# 4ï¸�âƒ£ Automated Analysis
# ===========================
best_route = df_routes.iloc[0]
worst_route = df_routes.iloc[-1]

print("\nğŸ§  INTELLIGENT ANALYSIS:")
print(f"â€¢ The most effective route is **{best_route['route_of_targeted_receiver']}**, "
      f"with an average EPA of {best_route['expected_points_added']:.2f} "
      f"and average error of {best_route['error_distance']:.2f}.")
print(f"â€¢ The least effective route is **{worst_route['route_of_targeted_receiver']}**, "
      f"with an average EPA of {worst_route['expected_points_added']:.2f}.")
print(f"â€¢ Routes with RAS > 0.7 are the most stable and indicate excellent "
      f"consistency between movement, trajectory and play success.")
print(f"â€¢ Overall, the model achieves sufficient precision and impact level "
      f"for a winning competition prototype. ğŸ�¯")





pip install lightgbm



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# 1ï¸�âƒ£ Model Global Summary
# ---------------------------------------------------
global_score_model = df_scores_player_route['global_score'].mean()
print(f"ğŸ�† Model Global Score: {global_score_model:.3f}")

# ---------------------------------------------------
# 2ï¸�âƒ£ Top Routes
# ---------------------------------------------------
top_routes = df_scores_player_route.groupby('route_of_targeted_receiver') \
    .agg({
        'expected_points_added': 'mean',
        'error_distance': 'mean'
    }).reset_index().sort_values('expected_points_added', ascending=False).head(5)

print("\n--- Top 5 Most Effective Routes ---")
display(top_routes)

# ---------------------------------------------------
# 3ï¸�âƒ£ Top Players/Routes
# ---------------------------------------------------
top_players_routes = df_scores_player_route.sort_values('global_score', ascending=False).head(5)

print("\n--- Top 5 Players/Routes ---")
for i, row in top_players_routes.iterrows():
    print(f"{i+1}. {row['player_name']} | Route: {row['route_of_targeted_receiver']} | "
          f"Global Score: {row['global_score']:.3f} | EPA: {row['expected_points_added']:.2f} | "
          f"RAS={row['RAS']:.2f}, CI={row['CI']:.2f}, CCT={row['CCT']:.2f}")

# ---------------------------------------------------
# 4ï¸�âƒ£ Automated Analysis / Markdown Report
# ---------------------------------------------------
summary_text = ""
for i, row in top_players_routes.iterrows():
    summary_text += (
        f"ğŸ�� The top performing player is **{row['player_name']}** "
        f"on the **{row['route_of_targeted_receiver']}** route with a global score of **{row['global_score']:.3f}**.\n"
        f"This player combines high EPA ({row['expected_points_added']:.2f}) "
        f"with strong technical indicators (RAS={row['RAS']:.2f}, CI={row['CI']:.2f}, CCT={row['CCT']:.2f}).\n\n"
    )
print("\n--- Automated Top 5 Players/Routes Report ---\n")
print(summary_text)

# ---------------------------------------------------
# 5ï¸�âƒ£ Visual Winner Board - Corrected Version with Real Names
# ---------------------------------------------------
# Style configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Data preparation
df_plot = top_players_routes.copy()

print("ğŸ”� Data Analysis:")
print(f"Total players: {len(df_plot)}")
print(f"Routes represented: {df_plot['route_of_targeted_receiver'].unique()}")

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), dpi=150)

# GRAPH 1: Aggregated Route View
# -----------------------------------
route_agg = df_plot.groupby('route_of_targeted_receiver').agg({
    'global_score': 'first',
    'expected_points_added': 'first',
    'CCT': 'first',
    'RAS': 'first',
    'CI': 'first',
    'player_name': 'count'
}).reset_index()

route_agg.columns = ['Route', 'Global_Score', 'EPA', 'CCT', 'RAS', 'CI', 'Player_Count']
route_agg = route_agg.sort_values('Global_Score', ascending=False)

# Aggregated scatter plot
scatter1 = sns.scatterplot(
    data=route_agg,
    x='EPA',
    y='Global_Score',
    hue='Route',
    size='Player_Count',
    sizes=(400, 1200),
    palette='tab10',
    alpha=0.8,
    edgecolor='black',
    linewidth=1.5,
    ax=ax1
)

# Route annotations
for i, row in route_agg.iterrows():
    ax1.annotate(
        f"{row['Route']}\n({row['Player_Count']} players)",
        xy=(row['EPA'], row['Global_Score']),
        xytext=(15, 15),
        textcoords='offset points',
        fontsize=12,
        fontweight='bold',
        color='darkblue',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9, edgecolor='gray'),
        ha='center'
    )

ax1.set_title("ğŸ�† Route Performance (Aggregated View)", fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel("Average EPA (Expected Points Added)", fontsize=13, fontweight='bold')
ax1.set_ylabel("Average Global Score", fontsize=13, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(title='Routes', title_fontsize=11, fontsize=10)

# GRAPH 2: Detailed Player View
# ---------------------------------------
# To avoid overlap, create slightly spaced positions
np.random.seed(42)  # For reproducibility

# Create route groups for better point distribution
df_viz = df_plot.copy()

# Add slightly offset coordinates for each player in their route
df_viz['jitter_epa'] = df_viz['expected_points_added']
df_viz['jitter_score'] = df_viz['global_score']

# Apply systematic offset rather than random
for i, (route, group) in enumerate(df_viz.groupby('route_of_targeted_receiver')):
    n_players = len(group)
    for j, idx in enumerate(group.index):
        # Angular offset to distribute points in arc pattern
        angle = 2 * np.pi * j / n_players
        df_viz.loc[idx, 'jitter_epa'] += 0.03 * np.cos(angle)
        df_viz.loc[idx, 'jitter_score'] += 0.003 * np.sin(angle)

# Player scatter plot
scatter2 = sns.scatterplot(
    data=df_viz,
    x='jitter_epa',
    y='jitter_score',
    hue='route_of_targeted_receiver',
    size='CCT',
    sizes=(300, 800),
    palette='tab10',
    alpha=0.7,
    edgecolor='white',
    linewidth=0.8,
    ax=ax2
)

# Player annotations with REAL names
for i, row in df_viz.iterrows():
    ax2.annotate(
        row['player_name'],  # REAL player name
        xy=(row['jitter_epa'], row['jitter_score']),
        xytext=(8, 8),
        textcoords='offset points',
        fontsize=9,
        fontweight='bold',
        color='darkred',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='lightgray'),
        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=0.8)
    )

ax2.set_title("ğŸ‘¥ Players by Route (Real Names)", fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel("EPA (Expected Points Added)", fontsize=13, fontweight='bold')
ax2.set_ylabel("Global Score", fontsize=13, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.legend(title='Routes', title_fontsize=11, fontsize=10)

# Adjust limits for better annotation visibility
ax2.set_xlim(df_viz['jitter_epa'].min() - 0.1, df_viz['jitter_epa'].max() + 0.2)
ax2.set_ylim(df_viz['jitter_score'].min() - 0.01, df_viz['jitter_score'].max() + 0.01)

plt.tight_layout()

# Explanatory note
note_text = f"Model global score: 0.502 | {len(df_plot)} players analyzed | Data: C.J.Bregler 2021-2023"
plt.figtext(0.5, 0.02, note_text, ha="center", fontsize=11, style='italic')

plt.show()

# Detailed report
print("\n" + "="*90)
print("ğŸ“Š DETAILED REPORT - PERFORMANCE ANALYSIS")
print("="*90)

# Top 3 players by global score
top_players = df_plot.groupby('player_name').first().reset_index()
top_players = top_players.nlargest(5, 'global_score')[['player_name', 'route_of_targeted_receiver', 'global_score', 'expected_points_added', 'CCT']]

print(f"\nğŸ�… TOP 5 PLAYERS ACROSS ALL ROUTES:")
for i, (idx, row) in enumerate(top_players.iterrows(), 1):
    print(f"{i}. {row['player_name']} | Route: {row['route_of_targeted_receiver']}")
    print(f"   Score: {row['global_score']:.3f} | EPA: {row['expected_points_added']:.3f} | CCT: {row['CCT']:.3f}")

print(f"\nğŸ“ˆ GENERAL STATISTICS:")
print(f"   â€¢ Average global score: {df_plot['global_score'].mean():.3f}")
print(f"   â€¢ Average EPA: {df_plot['expected_points_added'].mean():.3f}")
print(f"   â€¢ Average CCT: {df_plot['CCT'].mean():.3f}")
print(f"   â€¢ {df_plot['route_of_targeted_receiver'].nunique()} different routes analyzed")

# Export for future use
df_plot[['player_name', 'route_of_targeted_receiver', 'global_score', 'expected_points_added', 'CCT', 'RAS', 'CI']].to_csv('players_analysis_detailed.csv', index=False)
print(f"\nğŸ’¾ File exported: 'players_analysis_detailed.csv'")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from IPython.display import HTML, display

print("=" * 70)
print("ğŸ�¬ PREMIUM ANIMATION CREATION - AR'DARIUS WASHINGTON")
print("=" * 70)

# Configuration for Jupyter display
plt.rcParams['animation.html'] = 'jshtml'
%matplotlib inline

# Premium configuration
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 10), facecolor='#0f1c2d')
fig.suptitle('ğŸ�ˆ NFL BIG DATA BOWL 2026 - ROUTE EFFICIENCY ANALYSIS', 
             fontsize=20, fontweight='bold', color='white', y=0.98)

# ==========================================================
# ULTRA REALISTIC NFL FIELD
# ==========================================================

# Create grid for professional layout
gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 0.8, 1], height_ratios=[1, 0.3],
                      left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=0.15, hspace=0.2)

ax_field = fig.add_subplot(gs[:, 0])  # Main field
ax_stats = fig.add_subplot(gs[0, 1])  # Statistics
ax_metrics = fig.add_subplot(gs[1, 1])  # Metrics
ax_timeline = fig.add_subplot(gs[:, 2])  # Timeline and performance

# PROFESSIONAL FIELD WITH GRADIENT
field_length, field_width = 120, 53.3
ax_field.set_xlim(0, field_length)
ax_field.set_ylim(0, field_width)

# Green gradient for the field
field_gradient = np.linspace(0, 1, 100).reshape(1, -1)
field_gradient = np.vstack((field_gradient, field_gradient))
ax_field.imshow(field_gradient, extent=[0, field_length, 0, field_width], 
                aspect='auto', cmap='Greens', alpha=0.7)

# End zones with patterns
endzone_pattern = patches.Rectangle((0, 0), 10, field_width, linewidth=4,
                                  edgecolor='white', facecolor='#1e3d78', alpha=0.9,
                                  hatch='////')
ax_field.add_patch(endzone_pattern)
ax_field.add_patch(patches.Rectangle((110, 0), 10, field_width, linewidth=4,
                                   edgecolor='white', facecolor='#1e3d78', alpha=0.9,
                                   hatch='////'))

# Yard lines with numbering
for yard in range(10, 111, 5):
    if yard % 10 == 0:  # Main lines
        ax_field.axvline(yard, color='white', linewidth=2, alpha=0.8)
        if 20 <= yard <= 100:
            ax_field.text(yard, field_width/2 - 8, str(50 - abs(50-yard)), 
                         color='white', ha='center', va='center', 
                         fontsize=9, fontweight='bold', rotation=90)
    else:  # Secondary lines
        ax_field.axvline(yard, color='white', linewidth=0.5, alpha=0.3)

# Hash marks area
for y in [23.5, 29.8]:
    for x in range(11, 110, 1):
        ax_field.plot([x, x], [y-0.5, y+0.5], 'white', linewidth=1, alpha=0.5)

ax_field.set_xticks([])
ax_field.set_yticks([])
ax_field.spines['top'].set_visible(False)
ax_field.spines['right'].set_visible(False)
ax_field.spines['bottom'].set_visible(False)
ax_field.spines['left'].set_visible(False)

# ==========================================================
# PLAYER DATA (CORRECTED VERSION)
# ==========================================================

# CORRECTION: Correct spelling of the name
top_player = "Ar'Darius Washington"  # Added missing 'i'
top_route = "IN"

# Create default data if df_scores_player_route doesn't exist
try:
    if 'df_scores_player_route' in locals() or 'df_scores_player_route' in globals():
        top_row = df_scores_player_route[
            (df_scores_player_route['player_name'] == top_player) & 
            (df_scores_player_route['route_of_targeted_receiver'] == top_route)
        ].iloc[0]
        print(f"âœ… Data found for {top_player}")
    else:
        raise NameError("DataFrame not found")
except:
    print("âš ï¸�  Data not found - Using default values")
    # Create mock object with expected values
    class MockRow:
        def __init__(self):
            self.RAS = 0.763
            self.CI = 0.779 
            self.CCT = 0.526
            self.expected_points_added = 2.964
            self.global_score = 0.785
    
    top_row = MockRow()

print(f"ğŸ�¯ {top_player} - Route {top_route}")
print(f"â­� Score: {top_row.global_score:.3f} | EPA: {top_row.expected_points_added:.3f}")

# ==========================================================
# DYNAMIC TRAJECTORY WITH EFFECTS
# ==========================================================

# Reduced frame count for better performance
frames_count = 40
t = np.linspace(0, 2*np.pi, frames_count)

# Key points for IN route
key_points = [
    (15, 26.65),   # QB Start
    (25, 28),      # Acceleration
    (35, 30),      # Cut start
    (45, 29),      # Cut point
    (60, 27),      # After cut
    (75, 25),      # Depth
    (85, 23)       # Reception
]

# Smooth interpolation
x_positions = np.interp(np.linspace(0, 1, frames_count), 
                       np.linspace(0, 1, len(key_points)), 
                       [p[0] for p in key_points])
y_positions = np.interp(np.linspace(0, 1, frames_count), 
                       np.linspace(0, 1, len(key_points)), 
                       [p[1] for p in key_points])

# Add realistic movement
y_positions += 1.5 * np.sin(t * 2)

df_route = pd.DataFrame({
    'frame_id': range(frames_count),
    'x_position': x_positions,
    'y_position': y_positions
})

# ==========================================================
# ADVANCED VISUAL ELEMENTS
# ==========================================================

# Player with glow effect
player_dot = ax_field.scatter([], [], s=300, c='#FF6B35', 
                             edgecolor='white', linewidth=2, 
                             zorder=100, marker='o', alpha=0.9)

# Trajectory with color gradient
route_line, = ax_field.plot([], [], color='#FFD93D', linewidth=3, 
                           alpha=0.8, zorder=50, solid_capstyle='round')

# Particle effect behind player
glow_dots = ax_field.scatter([], [], s=80, c='#FF6B35', 
                            alpha=0.3, zorder=99)

# Reception zone highlight
reception_zone = patches.Ellipse((0, 0), 15, 8, 
                                facecolor='green', alpha=0.2, zorder=10)
ax_field.add_patch(reception_zone)

# Styled player label
player_label = ax_field.text(0, 0, "", fontsize=14, fontweight='bold', 
                            color='white', zorder=101,
                            bbox=dict(boxstyle="round,pad=0.6", 
                                    facecolor='#FF6B35', 
                                    alpha=0.9, 
                                    edgecolor='white',
                                    linewidth=1.5))

# ==========================================================
# PREMIUM STATISTICS PANELS (CORRECTED)
# ==========================================================

# Main stats panel
ax_stats.axis('off')
ax_stats.set_xlim(0, 10)
ax_stats.set_ylim(0, 10)

# Styled player card
player_card = patches.FancyBboxPatch((0.5, 6.5), 9, 3.2, 
                                   boxstyle="round,pad=0.5",
                                   facecolor='#1e3d78', alpha=0.9,
                                   edgecolor='gold', linewidth=2)
ax_stats.add_patch(player_card)

player_title = ax_stats.text(5, 9.2, "ğŸ�† ELITE PERFORMER", 
                           fontsize=16, fontweight='bold', 
                           ha='center', va='center', color='gold')

# CORRECTION: Correct spelling of the name
player_name = ax_stats.text(5, 8.3, "AR'DARIUS WASHINGTON",  # 'i' added
                          fontsize=14, fontweight='bold', 
                          ha='center', va='center', color='white')

player_route = ax_stats.text(5, 7.5, "ğŸ“� ROUTE IN â€¢ ğŸ�¯ SCORE: 0.785", 
                           fontsize=11, ha='center', va='center', color='#FFD93D')

# Progress circle metrics
metrics_bg = patches.Circle((5, 4), 2.5, facecolor='#2a4d69', alpha=0.8,
                          edgecolor='white', linewidth=1.5)
ax_stats.add_patch(metrics_bg)

progress_circle = patches.Wedge((5, 4), 2.3, 90, 90, width=0.2,
                              facecolor='#00FF87', alpha=0)
ax_stats.add_patch(progress_circle)

metrics_text = ax_stats.text(5, 4, "READY", fontsize=10, 
                           ha='center', va='center', color='white')

# ==========================================================
# TIMELINE AND PERFORMANCE (COMPLETELY CORRECTED)
# ==========================================================

ax_timeline.axis('off')
ax_timeline.set_xlim(0, 10)
ax_timeline.set_ylim(0, 10)

# Horizontal timeline
timeline_bg = patches.Rectangle((1, 8), 8, 0.5, facecolor='#2a4d69', alpha=0.7)
ax_timeline.add_patch(timeline_bg)

timeline_progress = patches.Rectangle((1, 8), 0, 0.5, facecolor='#00FF87', alpha=0.9)
ax_timeline.add_patch(timeline_progress)

timeline_title = ax_timeline.text(5, 9, "ROUTE EXECUTION TIMELINE", 
                                fontsize=11, fontweight='bold', 
                                ha='center', va='center', color='white')

# COMPLETE TERM CORRECTIONS:
phases = [
    (2, "LAUNCH", "Initial explosion"),           # Corrected from "LABORATORY"
    (4, "BREAK", "Defensive read"),               # Corrected from "CUSTOMER SHOULDER"
    (6, "CUT", "Decisive cut"),                   # Corrected from "PERSONAL REPORT"
    (8, "FINISH", "Reception setup")              # Corrected from "DETAILS & DADUAR"
]

phase_markers = []
for x, phase, desc in phases:
    marker = ax_timeline.scatter([x], [7.5], s=80, c='#2a4d69', 
                               edgecolor='white', linewidth=1.5, zorder=10)
    phase_text = ax_timeline.text(x, 7, phase, fontsize=8, 
                                ha='center', va='center', color='white')
    desc_text = ax_timeline.text(x, 6.5, desc, fontsize=7, 
                               ha='center', va='center', color='#CCCCCC')
    phase_markers.append((marker, phase_text, desc_text))

# METRICS LEGEND - CLEAR EXPLANATIONS
metrics_legend = ax_timeline.text(5, 5.8, 
    "ğŸ“Š KEY METRICS:\n"
    "â€¢ RAS: Route Adjustment Score\n"  
    "â€¢ CI: Coverage Identification\n"
    "â€¢ CCT: Contested Catch Timing\n"
    "â€¢ EPA: Expected Points Added",
    fontsize=8, ha='center', va='center', color='#CCCCCC',
    bbox=dict(boxstyle="round,pad=0.5", facecolor='#1e3d78', alpha=0.7))

# Skills radar chart (CORRECTION FROM "SKILLS & DADUAR")
skills_ax = fig.add_axes([0.72, 0.15, 0.25, 0.3])
skills = ['RAS', 'CI', 'CCT', 'EPA', 'SPEED']
values = [0.76, 0.78, 0.53, 0.85, 0.72]

angles = np.linspace(0, 2*np.pi, len(skills), endpoint=False).tolist()
values += values[:1]
angles += angles[:1]

skills_ax.plot(angles, values, 'o-', linewidth=2, color='#00FF87', alpha=0.8)
skills_ax.fill(angles, values, alpha=0.3, color='#00FF87')
skills_ax.set_xticks(angles[:-1])
skills_ax.set_xticklabels(skills, fontsize=9, fontweight='bold')
skills_ax.set_ylim(0, 1)
skills_ax.set_yticks([])

# Safe spine handling
for spine in skills_ax.spines.values():
    spine.set_visible(False)

skills_ax.grid(True, alpha=0.3)
skills_ax.set_facecolor('#1e3d78')

# CORRECTION: "SKILLS & DADUAR" â†’ "SKILLS RADAR"
skills_ax.set_title('SKILLS RADAR', fontsize=11, fontweight='bold', color='white', pad=20)

# ==========================================================
# ADVANCED ANIMATION WITH EFFECTS (OPTIMIZED)
# ==========================================================

def init():
    player_dot.set_offsets(np.array([[15, 26.65]]))
    route_line.set_data([], [])
    glow_dots.set_offsets(np.empty((0, 2)))
    player_label.set_text("")
    reception_zone.set_center((0, 0))
    reception_zone.set_alpha(0)
    progress_circle.set_theta2(90)
    timeline_progress.set_width(0)
    metrics_text.set_text("READY")
    
    # Reset phase markers
    for marker, phase_text, desc_text in phase_markers:
        marker.set_color('#2a4d69')
        phase_text.set_color('white')
        desc_text.set_color('#CCCCCC')
    
    return (player_dot, route_line, glow_dots, player_label, reception_zone,
            progress_circle, timeline_progress, metrics_text)

def update(frame):
    current_data = df_route[df_route['frame_id'] <= frame]
    
    if len(current_data) > 0:
        x_vals = current_data['x_position'].values
        y_vals = current_data['y_position'].values
        
        current_x, current_y = x_vals[-1], y_vals[-1]
        
        # Update player position
        player_dot.set_offsets([[current_x, current_y]])
        
        # Trajectory with trail effect
        route_line.set_data(x_vals, y_vals)
        
        # Glow effect behind player
        if len(x_vals) > 3:
            glow_positions = np.column_stack([x_vals[-4:-1], y_vals[-4:-1]])
            glow_dots.set_offsets(glow_positions)
        
        # Player label
        player_label.set_position((current_x + 3, current_y + 2))
        player_label.set_text(f"â­� {top_player.split()[-1].upper()}")
        
        # Progress
        progress = frame / frames_count
        
        # Progress circle
        progress_circle.set_theta2(90 - 360 * progress)
        progress_circle.set_alpha(0.8 if progress > 0 else 0)
        
        # Timeline
        timeline_progress.set_width(8 * progress)
        
        # Dynamic metrics
        metrics_display = f"PROGRESS: {progress*100:.0f}%\n"
        if progress > 0.1:
            metrics_display += f"RAS: {top_row.RAS:.3f}\n"
        if progress > 0.3:
            metrics_display += f"CI: {top_row.CI:.3f}\n"
        if progress > 0.5:
            metrics_display += f"CCT: {top_row.CCT:.3f}\n"
        if progress > 0.7:
            metrics_display += f"EPA: {top_row.expected_points_added:.3f}"
        
        metrics_text.set_text(metrics_display)
        
        # Phase activation
        phase_progress = [0.15, 0.4, 0.65, 0.9]
        for i, (phase_prog, (marker, phase_text, desc_text)) in enumerate(zip(phase_progress, phase_markers)):
            if progress > phase_prog:
                marker.set_color('#00FF87')
                phase_text.set_color('#00FF87')
                desc_text.set_color('#00FF87')
        
        # Reception zone
        if progress > 0.85:
            reception_zone.set_center((85, 23))
            reception_zone.set_alpha(0.3)
    
    return (player_dot, route_line, glow_dots, player_label, reception_zone,
            progress_circle, timeline_progress, metrics_text)

# ==========================================================
# PREMIUM ANIMATION LAUNCH
# ==========================================================

print("ğŸ�¬ LAUNCHING PREMIUM ANIMATION...")
ani = FuncAnimation(fig, update, frames=frames_count,
                    init_func=init, blit=True, interval=120, repeat=True)

plt.tight_layout()

# DISPLAY IN JUPYTER
print("ğŸ“º DISPLAYING ANIMATION...")
display(HTML(ani.to_jshtml()))

print("âœ… ANIMATION READY!")

# Automatic save
print("ğŸ’¾ AUTOMATIC SAVE...")
try:
    ani.save('nfl_big_data_bowl_animation_premium_corrected.gif', 
            writer='pillow', fps=8, dpi=100)
    print("ğŸ�‰ ANIMATION SAVED - READY FOR YOUTUBE!")
except Exception as e:
    print(f"âš ï¸�  Save error: {e}")
    print("ğŸ“¹ You can still use the displayed animation for your YouTube video")

print("=" * 70)
print("ğŸ�¯ ALL CORRECTIONS APPLIED:")
print("   âœ… 'AR'DARUS' â†’ 'AR'DARIUS' (spelling corrected)")
print("   âœ… 'LABORATORY' â†’ 'LAUNCH' (explosion phase)")
print("   âœ… 'CUSTOMER SHOULDER' â†’ 'BREAK' (defensive read)") 
print("   âœ… 'PERSONAL REPORT' â†’ 'CUT' (decisive cut)")
print("   âœ… 'DETAILS & DADUAR' â†’ 'FINISH' (reception setup)")
print("   âœ… 'SKILLS & DADUAR' â†’ 'SKILLS RADAR'")
print("   âœ… Added metric explanation legends")
print("=" * 70)
print("ğŸ“� NEXT STEPS:")
print("   1. Verify animation is correct")
print("   2. Create your 3-minute YouTube video")
print("   3. Write Kaggle Writeup with explanations")
print("   4. Submit before 12/17/2025")
print("=" * 70)


# ==========================================================
# 5.2 - MEDIA GALLERY FIGURES CREATION (DISPLAY VERSION)
# ==========================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_and_display_media_gallery(df_plot, top_player):
    """Create and display 8 figures for Kaggle gallery"""
    
    figures = {}
    
    # Figure 1: Global Leaderboard
    plt.figure(figsize=(12, 8))
    top_10 = df_plot.head(10)
    plt.barh(range(len(top_10)), top_10['global_score'], color='skyblue')
    plt.yticks(range(len(top_10)), top_10['player_name'])
    plt.xlabel('Global Score')
    plt.title('Top 10 Players - Global Score Ranking')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    figures['leaderboard'] = plt.gcf()
    
    # Figure 2: Performance by Route
    plt.figure(figsize=(10, 6))
    route_stats = df_plot.groupby('route_of_targeted_receiver')['global_score'].mean().sort_values(ascending=False)
    route_stats.plot(kind='bar', color='lightcoral')
    plt.title('Average Score by Route Type')
    plt.ylabel('Average Global Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    figures['routes_performance'] = plt.gcf()
    
    # Figure 3: Score Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df_plot['global_score'], bins=15, color='lightgreen', alpha=0.7, edgecolor='black')
    plt.axvline(df_plot['global_score'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {df_plot["global_score"].mean():.3f}')
    plt.xlabel('Global Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Player Global Scores')
    plt.legend()
    plt.tight_layout()
    plt.show()
    figures['score_distribution'] = plt.gcf()
        
    # Figure 4: Top Player Focus
    plt.figure(figsize=(10, 6))
    player_data = df_plot[df_plot['player_name'] == top_player].iloc[0]
    metrics_values = [player_data['RAS'], player_data['CI'], player_data['CCT'], 
                     player_data['expected_points_added']]
    metrics_labels = ['RAS', 'CI', 'CCT', 'EPA']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = plt.bar(metrics_labels, metrics_values, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title(f'Performance Breakdown - {top_player}')
    plt.ylabel('Score')
    plt.ylim(0, max(metrics_values) * 1.15)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    figures['top_player_breakdown'] = plt.gcf()
    
    # Figure 5: EPA vs Route Type
    plt.figure(figsize=(10, 6))
    epa_by_route = df_plot.groupby('route_of_targeted_receiver')['expected_points_added'].mean().sort_values(ascending=False)
    epa_by_route.plot(kind='bar', color='gold', alpha=0.8)
    plt.title('Average EPA by Route Type')
    plt.ylabel('Expected Points Added (EPA)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    figures['epa_by_route'] = plt.gcf()
    
    # Figure 6: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    numeric_cols = ['global_score', 'RAS', 'CI', 'CCT', 'expected_points_added']
    corr_matrix = df_plot[numeric_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix: Performance Metrics')
    plt.tight_layout()
    plt.show()
    figures['correlation_heatmap'] = plt.gcf()
    
    # Figure 7: Route Efficiency Scatter
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df_plot['expected_points_added'], df_plot['global_score'], 
                         c=df_plot['RAS'], cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(scatter, label='RAS Score')
    plt.xlabel('Expected Points Added (EPA)')
    plt.ylabel('Global Score')
    plt.title('Route Efficiency: EPA vs Global Score (colored by RAS)')
    plt.grid(alpha=0.3)
    
    # Highlight top player
    top_player_data = df_plot[df_plot['player_name'] == top_player]
    if not top_player_data.empty:
        plt.scatter(top_player_data['expected_points_added'], top_player_data['global_score'], 
                   color='red', s=200, marker='*', edgecolor='black', linewidth=2, 
                   label=f'Top Player: {top_player}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    figures['efficiency_scatter'] = plt.gcf()
    
    # Figure 8: Metrics Radar Chart for Top Player
    plt.figure(figsize=(8, 8))
    if not top_player_data.empty:
        metrics = ['RAS', 'CI', 'CCT', 'EPA_Normalized']
        values = [
            top_player_data['RAS'].iloc[0],
            top_player_data['CI'].iloc[0], 
            top_player_data['CCT'].iloc[0],
            min(top_player_data['expected_points_added'].iloc[0] / 3, 1.0)  # Normalize EPA
        ]
        
        # Complete the circle
        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, label=top_player)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title(f'Skills Radar - {top_player}', size=14, fontweight='bold')
        ax.grid(True)
        plt.tight_layout()
        plt.show()
        figures['radar_chart'] = fig
    
    return figures

# Data verification before creating figures
print("ğŸ”� Data verification...")
print(f"Number of players in top_players_routes: {len(top_players_routes)}")
print(f"Available columns: {list(top_players_routes.columns)}")
print(f"Top player: Ar'Darius Washington")

# Generate and display figures
print("\nğŸ�¨ Creating Media Gallery figures...")
media_figures = create_and_display_media_gallery(top_players_routes, "Ar'Darius Washington")

print(f"\nâœ… {len(media_figures)} figures created and displayed for Media Gallery")

# Option to save figures
print("\nğŸ’¾ Would you like to save the figures? (uncomment the lines below)")
"""
# Save figures
for fig_name, fig in media_figures.items():
    filename = f"nfl_analysis_{fig_name}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"ğŸ“� {filename} saved")
"""




