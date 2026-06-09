import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Define the directory and file names for week 1
input_file = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w01.csv'
output_file = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/output_2023_w01.csv'

# Load the data into pandas DataFrames
df_input = pd.read_csv(input_file)
df_output = pd.read_csv(output_file)

# Display the first few rows
print("Input DataFrame Head:")
print(df_input.head())

print("\nOutput DataFrame Head:")
print(df_output.head())

# Display summaries
print("\nInput DataFrame Info:")
df_input.info()

print("\nOutput DataFrame Info:")
df_output.info()

# Generate descriptive statistics
print("\nInput DataFrame Description:")
print(df_input.describe())

print("\nOutput DataFrame Description:")
print(df_output.describe())

# Check for missing values
print("\nInput DataFrame Missing Values:")
print(df_input.isnull().sum())

print("\nOutput DataFrame Missing Values:")
print(df_output.isnull().sum())


def load_and_merge_week_data(week_num):
    """
    Loads and merges the input and output data for a given week number.

    Args:
        week_num: The week number (integer) to load.

    Returns:
        A pandas DataFrame containing the merged data, or None if files are not found.
    """
    base_dir = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train'
    week_str = f"{week_num:02d}"
    input_file = os.path.join(base_dir, f'input_2023_w{week_str}.csv')
    output_file = os.path.join(base_dir, f'output_2023_w{week_str}.csv')

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return None
    if not os.path.exists(output_file):
        print(f"Output file not found: {output_file}")
        return None

    df_input = pd.read_csv(input_file)
    df_output = pd.read_csv(output_file)

    # Merge the dataframes on common identifier columns
    merged_df = pd.merge(df_input, df_output, on=['game_id', 'play_id', 'nfl_id', 'frame_id'], how='inner')

    return merged_df

# Example: Load and display data for week 1
df_week1 = load_and_merge_week_data(1)
if df_week1 is not None:
    print("\nMerged DataFrame Head (Week 1):")
    print(df_week1.head())
    print("\nMerged DataFrame Info (Week 1):")
    df_week1.info()



# Initialize an empty list to store the dfs for each week
weekly_dfs = []

# Iterate through the week numbers
# Inside the loop, call the load_and_merge_week_data function for the current week number
# If the function returns a valid df, append it to the list created
for week in range(1, 19):
    print(f"Loading and merging data for week {week}...")
    df_week = load_and_merge_week_data(week)
    if df_week is not None:
        weekly_dfs.append(df_week)
    else:
        print(f"Skipping week {week} due to missing files.")

# After the loop, concatenate all the dfs in the list into a single df
# Store the resulting combined df in a variable
if weekly_dfs:
    df_combined = pd.concat(weekly_dfs, ignore_index=True)


# Examine the df_combined df to understand its columns and data types
print("Combined DataFrame Columns and Data Types:")
df_combined.info()

# Calculate descriptive statistics for relevant numerical columns
print("\nDescriptive Statistics for Numerical Columns:")
print(df_combined.describe())


# Explore the distribution of categorical variables
print("\nValue Counts for Categorical Columns:")
for col in ['play_direction', 'player_position', 'player_side', 'player_role']:
    print(f"\n{df_combined[col].value_counts()}\n")

# Visualize the correlation matrix using a heatmap
print("\nCorrelation Matrix for Numerical Columns:\n")
numerical_cols = df_combined.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df_combined[numerical_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Columns')
plt.show()


# Calculate the distance from each player to the ball landing location
df_combined['distance_to_ball_land'] = np.sqrt(
    (df_combined['x_x'] - df_combined['ball_land_x'])**2 +
    (df_combined['y_x'] - df_combined['ball_land_y'])**2
)

# Calculate the distance covered by each player within each play
# This requires sorting by frame_id within each game_id and play_id for each player
df_combined = df_combined.sort_values(by=['game_id', 'play_id', 'nfl_id', 'frame_id'])

# Calculate the displacement between consecutive frames for each player in each play
df_combined['dx'] = df_combined.groupby(['game_id', 'play_id', 'nfl_id'])['x_x'].diff().fillna(0)
df_combined['dy'] = df_combined.groupby(['game_id', 'play_id', 'nfl_id'])['y_x'].diff().fillna(0)

# Calculate the distance covered in each frame
df_combined['frame_distance'] = np.sqrt(df_combined['dx']**2 + df_combined['dy']**2)

# Calculate the cumulative distance covered within each play for each player
df_combined['distance_covered'] = df_combined.groupby(['game_id', 'play_id', 'nfl_id'])['frame_distance'].cumsum()

# Display descriptive statistics for the new features
print("\nDescriptive Statistics for New Features:")
print(f"{df_combined[['distance_to_ball_land', 'distance_covered']].describe()}\n")

# Explore the distribution of distance_to_ball_land
df_combined['distance_to_ball_land'].hist(bins=50)
plt.title('Distribution of Distance to Ball Land:')
plt.grid(False)
plt.show()


# Visualize the distribution of speed (s) and acceleration (a)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_combined['s'], bins=50, kde=True)
plt.title('Distribution of Player Speed (s)')
plt.xlabel('Speed (yards/second)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(df_combined['a'], bins=50, kde=True)
plt.title('Distribution of Player Acceleration (a)')
plt.xlabel('Acceleration (yards/second^2)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Visualize the relationship between speed and distance covered (optional, might be too dense)
# Consider sampling or aggregating if plotting all points
# plt.figure(figsize=(8, 6))
# sns.scatterplot(data=df_combined.sample(10000), x='s', y='distance_covered', alpha=0.5)
# plt.title('Speed vs Distance Covered (Sample)')
# plt.xlabel('Speed (yards/second)')
# plt.ylabel('Distance Covered (yards)')
# plt.show()


# Analyze player-specific statistics, e.g., average speed or total distance covered per player
# Group by player_name and calculate aggregate statistics
player_stats = df_combined.groupby('player_name').agg(
    average_speed=('s', 'mean'),
    max_speed=('s', 'max'),
    average_acceleration=('a', 'mean'),
    max_acceleration=('a', 'max'),
    total_distance_covered=('distance_covered', 'max') # Max distance covered within each play for each player
).reset_index()

print("\nPlayer Statistics (Average Speed, Max Speed, etc.):")
print(f"{player_stats.sort_values(by='total_distance_covered', ascending=False).head()}\n")

# Visualize player movement for a specific play (example for game_id=2023090700, play_id=101)
# Filter data for a specific play
sample_play_df = df_combined[(df_combined['game_id'] == 2023090700) & (df_combined['play_id'] == 101)]

if not sample_play_df.empty:
    plt.figure(figsize=(15, 8))
    # Plot player trajectories, colored by player or team
    for nfl_id in sample_play_df['nfl_id'].unique():
        player_df = sample_play_df[sample_play_df['nfl_id'] == nfl_id]
        plt.plot(player_df['x_x'], player_df['y_x'], label=f'NFL ID: {nfl_id}', alpha=0.7)

    # Add ball landing location
    ball_land_x = sample_play_df['ball_land_x'].iloc[0]
    ball_land_y = sample_play_df['ball_land_y'].iloc[0]
    plt.scatter(ball_land_x, ball_land_y, color='red', marker='X', s=200, label='Ball Land Location')

    plt.title('Player Movement for Game ID 2023090700, Play ID 101')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(False)
    plt.axis('equal') # Ensure equal scaling for x and y axes
    plt.show()
else:
    print("\nSample play data not found.")


# Create a scatter plot of average speed vs. total distance covered
plt.figure(figsize=(10, 6))
sns.scatterplot(data=player_stats, x='average_speed', y='total_distance_covered', alpha=0.6)
plt.title('Average Speed vs. Total Distance Covered per Player')
plt.xlabel('Average Speed (yards/second)')
plt.ylabel('Total Distance Covered (yards)')
plt.grid(False)
plt.show()


# Filter the combined DataFrame for players to predict
players_to_predict_df = df_combined[df_combined['player_to_predict'] == True]

# Create a histogram of the distance_to_ball_land for players to predict
plt.figure(figsize=(10, 6))
sns.histplot(players_to_predict_df['distance_to_ball_land'], bins=50, kde=True)
plt.title('Distribution of Distance to Ball Land for Players to Predict')
plt.xlabel('Distance to Ball Land (yards)')
plt.ylabel('Frequency')
plt.show()


# Get the top N most common player positions
top_n_positions = df_combined['player_position'].value_counts().nlargest(10).index.tolist()

# Filter the combined DataFrame to include only the top N positions
df_top_positions = df_combined[df_combined['player_position'].isin(top_n_positions)]

# Create a box plot of speed across the top N player positions
plt.figure(figsize=(14, 7))
sns.boxplot(data=df_top_positions, x='player_position', y='s')
plt.title('Distribution of Speed (s) Across Top 10 Player Positions')
plt.xlabel('Player Position')
plt.ylabel('Speed (yards/second)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

