# Import necessary libraries
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import concurrent.futures
import ipywidgets as widgets
from IPython.display import display, Javascript, clear_output
import pandas as pd
import pyspark.sql as spark
import logging
import sys
from sklearn.ensemble import RandomForestRegressor


# Initialize global dictionary to store all dataframes
dataframes = {}

def load_all_data():
    """
    Load all CSV files from the data directory into a global dictionary.
    
    This function:
    1. Looks for CSV files in the 'data' directory
    2. Uses parallel processing to load files efficiently
    3. For each file, prints diagnostic information including:
       - Shape of the dataframe
       - First 5 rows preview
       - Count of null values
       - Column names
    4. Stores each dataframe in the global 'dataframes' dictionary
    
    Returns:
        bool: True if loading succeeds, False if there are any errors
    """
    global dataframes
    data_directory = 'data'
    
    # Check if data directory exists
    if not os.path.exists(data_directory):
        print(f"Error: Directory '{data_directory}' not found!")
        return False
        
    try:
        # Get list of CSV files and their full paths
        csv_files = [filename for filename in os.listdir(data_directory) if filename.endswith('.csv')]
        filepaths = [os.path.join(data_directory, filename) for filename in csv_files]

        def load_file(filepath):
            """
            Helper function to load a single CSV file and print diagnostic information
            
            Args:
                filepath: Path to the CSV file
                
            Returns:
                pandas.DataFrame: Loaded dataframe
            """
            df = pd.read_csv(filepath)
            # Print diagnostic information about the loaded file
            print(f"Loaded {os.path.basename(filepath)} with shape {df.shape}")
            print(f"\nFirst 5 rows of {os.path.basename(filepath)}:")
            print(df.head())
            print(f"\nNumber of nulls in {os.path.basename(filepath)}:")
            print(df.isnull().sum())
            print(f"\nColumns in {os.path.basename(filepath)}:")
            print(df.columns)
            return df

        # Use ThreadPoolExecutor for parallel file loading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(load_file, filepaths))
            # Store each dataframe in the global dictionary
            for filename, df in zip(csv_files, results):
                dataframes[filename] = df

        print("\nData loading complete!")
        return True
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False



load_all_data()


# Function to analyze files and provide recommendations for data cleaning
def analyze_files(dataframes):
    # Iterate through each dataframe in the provided dictionary
    for name, df in dataframes.items():
        print(f"\n{'='*50}\nAnalyzing {name}...\n{'='*50}")
        
        print(f'{name}')
        
        # Analyze missing values in detail
        missing_values = df.isnull().sum()  # Count missing values for each column
        missing_pct = (df.isnull().sum() / len(df)) * 100  # Calculate percentage of missing values
        
        # Check if there are any missing values in the dataframe
        if missing_values.any():
            print("\nMissing Values Analysis:")
            for col in df.columns:
                if missing_values[col] > 0:  # If the column has missing values
                    print(f"\nColumn: {col}")
                    print(f"Missing values: {missing_values[col]} ({missing_pct[col]:.2f}%)")
                    
                    # Display one row with a missing value and one with data (showing only a few columns)
                    print("\nExample Rows:")
                    missing_row = df[df[col].isnull()].iloc[0] if not df[df[col].isnull()].empty else None
                    data_row = df[df[col].notnull()].iloc[0] if not df[df[col].notnull()].empty else None
                    
                    if missing_row is not None:
                        print("Row with missing value:")
                        print(missing_row[:4])  # Show first 4 columns of the row with missing value
                    
                    if data_row is not None:
                        print("Row with data:")
                        print(data_row[:4])  # Show first 4 columns of the row with data
                    
                    # Analyze patterns in missing data
                    print("\nPattern Analysis:")
                    
                    # Check if missing values are concentrated in specific games
                    if 'gameId' in df.columns:
                        missing_by_game = df[df[col].isnull()]['gameId'].value_counts()
                        if len(missing_by_game) < len(df['gameId'].unique()) * 0.5:
                            print("- Missing values appear to be concentrated in specific games")
                    
                    # Check if missing values are related to specific teams
                    if 'possessionTeam' in df.columns:
                        missing_by_team = df[df[col].isnull()]['possessionTeam'].value_counts()
                        if len(missing_by_team) < len(df['possessionTeam'].unique()) * 0.5:
                            print("- Missing values appear to be team-specific")
                    
                    # Analyze data type and suggest handling strategy
                    dtype = df[col].dtype  # Get the data type of the column
                    non_null_values = df[col].dropna()  # Get non-null values for further analysis
                    
                    print("\nRecommended Action:")
                    
                    # Assess the percentage of missing values and provide recommendations
                    if missing_pct[col] < 5:
                        print("- Low Missingness (<5%): Generally safe to impute missing values.")
                    elif 5 <= missing_pct[col] <= 30:
                        print("- Moderate Missingness (5%-30%): Imputation is possible but requires careful consideration.")
                    else:
                        print("- High Missingness (>30%): Imputation may introduce significant bias; consider dropping the column or using advanced imputation methods.")
                    
                    # Recommendations based on data type
                    if dtype == 'object':
                        # Categorical data recommendations
                        unique_vals = non_null_values.nunique()  # Count unique non-null values
                        if unique_vals < 10:
                            print(f"- This appears to be a categorical column with {unique_vals} unique values")
                            mode_val = non_null_values.mode()[0]  # Get the mode of the column
                            print(f"- Consider filling missing values with mode '{mode_val}' or 'Unknown'")
                        else:
                            print("- This appears to be a text column with many unique values")
                            if missing_pct[col] > 50:
                                print("- Consider dropping this column due to high percentage of missing values")
                            else:
                                print("- Consider filling missing values with 'Unknown'")
                    
                    elif np.issubdtype(dtype, np.number):
                        # Numerical data recommendations
                        print(f"Statistical Summary of non-null values:")
                        print(non_null_values.describe())  # Display statistical summary
                        
                        if missing_pct[col] < 5:
                            print("- Low percentage of missing values")
                            print("- Consider filling with median")
                        elif missing_pct[col] > 50:
                            print("- High percentage of missing values")
                            print("- Consider dropping this column")
                        else:
                            print("Recommended options:")
                            print("1. Fill with mean")
                            print("2. Fill with median")
                            print("3. Fill with 0")
                            print("4. Drop column")
                    
                    elif dtype == 'bool':
                        print("- Boolean column")
                        print("- Consider filling missing values with False")
                    
                    elif dtype == 'datetime64[ns]':
                        print("- DateTime column")
                        if 'gameId' in df.columns:
                            print("- Consider filling missing dates based on gameId using forward fill")
        
        # Check for outliers in numerical columns
        print("\nOutlier Analysis:")
        numerical_cols = df.select_dtypes(include=[np.number]).columns  # Select numerical columns
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)  # Calculate the first quartile
            Q3 = df[col].quantile(0.75)  # Calculate the third quartile
            IQR = Q3 - Q1  # Calculate the interquartile range
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]  # Identify outliers
            if len(outliers) > 0:
                print(f"\nFound {len(outliers)} outliers in {col}")
                print("- Consider capping outliers at 1.5 IQR")  # Suggest capping outliers

# Call the function to analyze the dataframes and get recommendations
analyze_files(dataframes)



# Assuming 'dataframes' is a dictionary containing the loaded dataframes
games_df = dataframes['games.csv']  # Load the games dataframe from the dataframes dictionary

# Option to set a team filter or no filter
team_filter = 'KC'  # Specify the team abbreviation to filter by (e.g., 'KC' for Kansas City)

use_filter = False  # Set to False if you want to include all games; True to filter by team

if use_filter:
    # If filtering is enabled, keep only the rows for the specified team
    games_df = games_df[(games_df['homeTeamAbbr'] == team_filter) | (games_df['visitorTeamAbbr'] == team_filter)]


# Drop unnecessary columns to reduce data size
games_df = games_df[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']]  # Retain only essential columns

# Print count of games
print(f"Total number of games: {len(games_df)}")  # Display the total number of games in the dataframe

# Show the head of the filtered games dataframe
print(games_df.head())  # Display the first few rows of the games dataframe

# Show the description of the filtered games dataframe
print(games_df.describe())  # Provide summary statistics of the dataframe

# Show the data types of the filtered games dataframe
print(games_df.dtypes)  # Display the data types of each column in the dataframe

# Print missing values in the filtered games dataframe
missing_values = games_df.isnull().sum()  # Calculate the number of missing values in each column
print("Missing values in each column:")  # Inform about missing values
print(missing_values[missing_values > 0])  # Display columns with missing values

game_ids = games_df['gameId'].unique().tolist()  # Extract unique game IDs from the dataframe
print("Unique game IDs:", game_ids)  # Display the unique game IDs

# Save the edited dataframe back into the dataframes dictionary
dataframes['games.csv'] = games_df  # Update the dataframes dictionary with the modified games dataframe

# Count the number of unique gameIds
print(f"Number of unique gameIds: {len(game_ids)}")


# Assuming 'dataframes' is a dictionary containing the loaded dataframes
players_df = dataframes['player_play.csv']  # Load the player-play dataframe from the dataframes dictionary

# Filter players_df to keep only the rows where gameId corresponds to the selected game IDs
players_df = players_df[players_df['gameId'].isin(game_ids)]  # Retain only the relevant games based on game IDs

# Keep only the specified columns that are necessary for analysis
players_df = players_df[['hadRushAttempt', 'rushingYards', 'hadDropback', 'passingYards', 
                          'wasRunningRoute', 'routeRan', 'pff_defensiveCoverageAssignment', 
                          'gameId', 'playId', 'nflId']]  # Select essential columns for further processing

# Fill missing values as per instructions
players_df['wasRunningRoute'] = players_df['wasRunningRoute'].fillna(0)  # Fill NaN with 0
players_df['routeRan'] = players_df['routeRan'].fillna('Unknown')  # Fill NaN with 'unknown'
players_df['pff_defensiveCoverageAssignment'] = players_df['pff_defensiveCoverageAssignment'].fillna('Unknown')  # Fill NaN with 'unknown'

# Show the head of the dataframe to get a glimpse of the data
print(players_df.head())  # Display the first few rows of the players dataframe

# Show the description of the dataframe to understand its statistical properties
print(players_df.describe())  # Provide summary statistics of the dataframe

# Show the data types of the dataframe to check the format of each column
print(players_df.dtypes)  # Display the data types of each column in the dataframe

# Print unique values of each column to figure out what to fill null values with
for column in players_df.columns:
    print(f"Unique values in '{column}': {players_df[column].unique()}")  # Display unique values for each column

# Print missing values in the players_df dataframe to identify any data quality issues
missing_values = players_df.isnull().sum()  # Calculate the number of missing values in each column
print("Missing values in each column:")  # Inform about missing values
print(missing_values[missing_values > 0])  # Display columns with missing values

# Save the edited dataframe back into the dataframes dictionary for future use
dataframes['players.csv'] = players_df  # Update the dataframes dictionary with the modified players dataframe

# Count the number of rows in the player play dataframe
print(f"Number of rows in the player play dataframe: {len(players_df)}")



# Load the plays dataframe from the dataframes dictionary
plays_df = dataframes['plays.csv']

# Filter the dataframe to retain only the rows corresponding to the selected game IDs
plays_df = plays_df[plays_df['gameId'].isin(game_ids)]

# Apply a redzone filter to keep only plays where yardsToGo is less than or equal to 20
plays_df = plays_df[plays_df['yardsToGo'] <= 20]

# Check if a specific team filter is to be applied
if use_filter:
    # Filter the dataframe for plays where the defensive team and yardline side match the specified team
    plays_df = plays_df[(plays_df['defensiveTeam'] == team_filter) & (plays_df['yardlineSide'] == team_filter)]

# Display the first few rows of the filtered dataframe for inspection
print(plays_df.head())

# Check if the 'passTippedAtLine' column exists in the dataframe
if 'passTippedAtLine' in plays_df.columns:
    # Print unique values of the 'passTippedAtLine' column for understanding its contents
    print("Unique values of passTippedAtLine:", plays_df['passTippedAtLine'].unique())
    # Fill NaN values in 'passTippedAtLine' with False and convert the column to boolean type
    plays_df['passTippedAtLine'] = plays_df['passTippedAtLine'].fillna(False).astype(bool)
else:
    # If the column does not exist, initialize it to False
    print("passTippedAtLine column does not exist.")
    plays_df['passTippedAtLine'] = False  

# Define the essential columns needed for further analysis
essential_columns = [
    'gameId', 'playId', 'down', 'yardsToGo', 'possessionTeam', 
    'defensiveTeam', 'absoluteYardlineNumber', 'offenseFormation', 
    'receiverAlignment', 'passResult', 'passLength', 'targetX', 
    'targetY', 'unblockedPressure', 
    'yardsGained', 'isDropback', 'pff_runConceptPrimary', 
    'pff_passCoverage', 'pff_manZone', 'expectedPointsAdded'
]

# Filter the dataframe to keep only the essential columns
plays_df = plays_df[essential_columns]

# Set the 'pff_runConceptPrimary' column to 'UNDEFINED' for all rows
plays_df['pff_runConceptPrimary'] = 'UNDEFINED'

# Create boolean masks to filter out specific play types that are not of interest
plays_df = plays_df[
    (plays_df.get('qbSpike', pd.Series(0, index=plays_df.index)).fillna(0).astype(bool) == False) &
    (plays_df.get('qbKneel', pd.Series(0, index=plays_df.index)).fillna(0).astype(bool) == False) &
    (plays_df.get('qbSneak', pd.Series(0, index=plays_df.index)).fillna(0).astype(bool) == False) &
    (plays_df.get('passTippedAtLine', pd.Series(False, index=plays_df.index)).fillna(False).astype(bool) == False) &
    (plays_df.get('playNullifiedByPenalty', pd.Series(False, index=plays_df.index)).fillna(False).astype(bool) == False)
]

# Drop rows with missing values in 'targetX' and 'targetY' as they are critical for analysis
plays_df = plays_df.dropna(subset=['targetX', 'targetY'])

# Define a dictionary to fill missing values in essential columns with appropriate values
fills = {
    'offenseFormation': 'Dynamic',  # Fill with 'Dynamic' for unknown formations
    'receiverAlignment': 'Unknown',  # Default value for unknowns
    'passResult': 'Unknown',  # Default value for unknowns
    'passLength': plays_df['passLength'].median(),  # Fill with the median value for numerical columns
    'targetX': plays_df['targetX'].median(),
    'targetY': plays_df['targetY'].median(),
    'yardsGained': plays_df['yardsGained'].median(),
    'expectedPointsAdded': plays_df['expectedPointsAdded'].median()
}

# Count the number of rows with missing values in the essential columns
missing_values = plays_df.isnull().sum()
print("Missing values in essential columns:")
print(missing_values[missing_values > 0])

# Print unique values of each column for further inspection
for column in plays_df.columns:
    print(f"Unique values in '{column}': {plays_df[column].unique()}")

# Print the head, description, and missing values of the cleaned dataframe for review
print(plays_df.head(), plays_df.describe(), plays_df.isnull().sum(), sep='\n')
print(f"Rows after dropping those with missing critical data: {len(plays_df)}")

# Save the edited dataframe back into the dataframes dictionary for future use
dataframes['plays.csv'] = plays_df


import pandas as pd
import glob
import re

# Get a list of all tracking data files using regex
tracking_files = [file for file in dataframes.keys() if re.match(r'tracking_.*\.csv', file)]

# Initialize an empty list to hold DataFrames
tracking_dfs = []

# Define relevant events
relevant_events = [
    'pass_outcome_touchdown', 
    'touchdown', 
    'pass_outcome_caught', 
    'pass_outcome_incomplete', 
    'pass_outcome_interception', 
    'run', 
    'handoff', 
    'fumble', 
    'safety', 
    'field_goal_play',
    'qb_sack',  # This can also be relevant as it indicates a failed play
    'dropped_pass'  # Indicates a missed opportunity
]

# Loop through each file and process it
for file in tracking_files:
    # Filter tracking_df to keep only the rows where gameId corresponds to the selected game IDs
    tracking_df = dataframes[file]
    tracking_df = tracking_df[tracking_df['gameId'].isin(game_ids)]

    tracking_df = tracking_df.dropna(subset=['x', 'y'])
    
    # Select relevant columns
    tracking_df = tracking_df[['gameId', 'playId', 'nflId', 'frameType', 'frameId', 
                               'club', 'x', 'y', 'event']]
    
    # Fill null values in 'event' with 'No Event'
    tracking_df['event'] = tracking_df['event'].fillna('No Event')
    
    # Drop rows where 'club' is 'football'
    tracking_df = tracking_df[tracking_df['club'] != 'football']
    
    # Append the processed DataFrame to the list
    tracking_dfs.append(tracking_df)

# Concatenate all DataFrames into a single DataFrame
dataframes['all_tracking_data'] = pd.concat(tracking_dfs, ignore_index=True)

# Print unique values of each column for further inspection
for column in dataframes['all_tracking_data'].columns:
    print(f"Unique values in '{column}': {dataframes['all_tracking_data'][column].unique()}")


# Delete individual tracking DataFrames from the dictionary
for file in tracking_files:
    del dataframes[file]

# Print the rows with null values
print(dataframes['all_tracking_data'][dataframes['all_tracking_data'].isnull().any(axis=1)])

# Print the head and description of the combined tracking data
print(dataframes['all_tracking_data'].head(), dataframes['all_tracking_data'].describe(), sep='\n')



# Count the number of rows in the combined tracking data
print(f"Number of rows in the combined tracking data: {len(dataframes['all_tracking_data'])}")



# Since we already have a dataframe with all tracking data, we can simplify the merge process.
def merge_datasets(dataframes, game_ids=None):
    """
    Efficiently merge plays and player_play data with existing tracking data
    
    Args:
        dataframes (dict): Dictionary containing all dataframes
        game_ids (list, optional): List of specific game IDs to filter by
    
    Returns:
        pd.DataFrame: Merged dataset with relevant features
    """
    logging.info("Starting dataset merge process...")
    
    try:
        # Get required dataframes
        plays_df = dataframes['plays.csv']
        player_play_df = dataframes['player_play.csv']
        tracking_df = dataframes['all_tracking_data']  # Use the pre-loaded tracking data
        
        # Filter by game_ids if provided
        if game_ids:
            plays_df = plays_df[plays_df['gameId'].isin(game_ids)]
            player_play_df = player_play_df[player_play_df['gameId'].isin(game_ids)]
            tracking_df = tracking_df[tracking_df['gameId'].isin(game_ids)]
        
        # Select only necessary columns from each dataset
        plays_columns = [
            'gameId', 'playId', 'down', 'yardsToGo', 'absoluteYardlineNumber',
            'offenseFormation', 'isDropback', 'passLength', 'targetX', 'targetY',
            'expectedPointsAdded'
        ]
        
        player_play_columns = [
            'gameId', 'playId', 'nflId', 'hadRushAttempt', 'rushingYards',
            'hadDropback', 'passingYards', 'wasRunningRoute', 'routeRan'
        ]
        
        # Filter columns
        plays_df = plays_df[plays_columns]
        player_play_df = player_play_df[player_play_columns]
        
        # First merge plays with player_play
        logging.info("Merging plays with player_play data...")
        merged_df = pd.merge(
            plays_df,
            player_play_df,
            on=['gameId', 'playId'],
            how='left'
        )
        
        # Merge with tracking data
        logging.info("Merging with tracking data...")
        merged_df = pd.merge(
            merged_df,
            tracking_df,
            on=['gameId', 'playId', 'nflId'],
            how='inner'  # Use inner join to keep only matching records
        )
        
        # Optimize memory usage
        logging.info("Optimizing memory usage...")
        for col in merged_df.columns:
            if merged_df[col].dtype == 'object':
                merged_df[col] = merged_df[col].astype('category')
            elif merged_df[col].dtype == 'float64':
                merged_df[col] = merged_df[col].astype('float32')
            elif merged_df[col].dtype == 'int64':
                merged_df[col] = merged_df[col].astype('int32')
        
        logging.info(f"Final merged dataset shape: {merged_df.shape}")
        return merged_df
    
    except Exception as e:
        logging.error(f"Error during merge process: {str(e)}")
        raise

# Use the function
try:
    # You can optionally provide game_ids to filter
    # game_ids = plays_df['gameId'].unique().tolist()[:10]  # Example: first 10 games
    
    merged_df = merge_datasets(dataframes)
    
    # Print some information about the merged dataset
    print("\nMerged Dataset Info:")
    print(f"Shape: {merged_df.shape}")
    print("\nMemory usage:")
    print(merged_df.memory_usage(deep=True).sum() / 1024**2, "MB")
    print("\nSample of merged data:")
    print(merged_df.head())
    
except Exception as e:
    print(f"Error: {str(e)}")


# Fill wasRunningRoute with 0
merged_df['wasRunningRoute'] = merged_df['wasRunningRoute'].fillna(0)

# Drop all rows with no route
merged_df = merged_df[merged_df['routeRan'].notna()]

# Print all columns in merged_df that have null values
print(merged_df.columns[merged_df.isnull().any()])

# Print all unique values in the columns that have null values
for column in merged_df.columns[merged_df.isnull().any()]:
    print(f"Unique values in '{column}': {merged_df[column].unique()}")



# First, let's see what columns we have
print("Available columns in merged_df:")
print(merged_df.columns.tolist())
# Print unique values for each column in plays_df
print("Unique values in each column:")
for column in merged_df.columns:
    print(f"\n{column}:")
    print(merged_df[column].unique())


def categorize_play(row):
    """Create detailed play categories with hierarchical structure"""
    if row['isDropback']:
        # Base category
        if pd.notna(row['passLength']):
            pass_length = float(row['passLength'])
            if pass_length <= 5:
                base_type = 'QUICK_PASS'
            elif pass_length <= 15:
                base_type = 'MEDIUM_PASS'
            else:
                base_type = 'DEEP_PASS'
        else:
            base_type = 'MEDIUM_PASS'  # Default
            
        # Add formation with more options
        formation = row['offenseFormation'] if row['offenseFormation'] in ['SHOTGUN', 'UNDER_CENTER', 'I_FORM', 'PISTOL', 'WILDCAT'] else 'OTHER'
        
        # Add direction based on targetX
        if pd.notna(row['targetX']):
            if row['targetX'] < 26.65:  # Left side of the field
                direction = 'LEFT'
            elif row['targetX'] > 26.65:  # Right side of the field
                direction = 'RIGHT'
            else:  # Center
                direction = 'CENTER'
        
        # Include routeRan as an identifier
        route_ran = row['routeRan'] if pd.notna(row['routeRan']) else 'UNKNOWN_ROUTE'
        
        return f"{base_type}_{direction}_{formation}_{route_ran}"
    else:
        # Run plays
        formation = row['offenseFormation']
        if formation == 'SHOTGUN':
            return 'RUN_SHOTGUN'
        elif formation == 'I_FORM':
            return 'RUN_I_FORM'
        else:
            return 'RUN_STANDARD'


# The HierarchicalPlayPredictor class helps predict the success of different football plays
# by analyzing both basic and detailed play types, defensive formations, and historical success rates.

# The HierarchicalPlayPredictor uses a multi-level approach to predict play success:

# 1. Base Level - Analyzes fundamental play types (run vs pass) based on:
#    - Down and distance
#    - Field position 
#    - Defensive formation/personnel
#    - Historical success rates

# 2. Detailed Level - Evaluates specific variations within each play type:
#    - Route combinations
#    - Formation specifics
#    - Target locations
#    - Defensive matchups

# 3. Success Prediction:
#    - Uses ensemble of ML models (Random Forests, XGBoost)
#    - Weighs historical success rates
#    - Considers defensive vulnerabilities
#    - Accounts for game situation

# Key metrics for scoring probability:
# - Expected Points Added (EPA)
# - Success Rate
# - Yards gained probability distribution
# - Touchdown/Field Goal probability based on field position

# The predictor helps optimize play calling by:
# 1. Identifying defensive weaknesses (e.g. mismatches, formation vulnerabilities)
# 2. Recommending plays with highest expected success rate
# 3. Adapting to in-game defensive adjustments
# 4. Balancing risk/reward based on game situation

# Example usage:
# predictor = HierarchicalPlayPredictor()
# predictor.train(game_data)  # Train on historical play data
# recommendations = predictor.predict(current_situation)  # Get top play recommendations

# Features used for predictions
base_features = [
    'down', 'yardsToGo', 'absoluteYardlineNumber', # Game situation
    'x', 'y'  # Defender positions
]

# Additional features for more detailed predictions
detailed_features = base_features + ['targetX', 'targetY'] # Add pass target location

class HierarchicalPlayPredictor:
    def __init__(self, min_samples=5):
        self.base_models = {}        # Models for basic play types (run/pass)
        self.detailed_models = {}    # Models for specific play variations
        self.success_rates = {}      # Historical success rates for each play type
        self.min_samples = min_samples
        
    def get_defensive_features(self, play_data):
        """Analyzes how the defense is lined up before the snap"""
        defense_snapshot = play_data[
            (play_data['frameType'] == 'BEFORE_SNAP') & 
            (play_data['club'] != play_data['club'].iloc[0])
        ]
    
        # Return default features if no defensive data found
        if len(defense_snapshot) == 0:
            return {
                'def_spread': 0,
                'def_depth': 0,
                'def_players_in_box': 0,
                'def_players_deep': 0
            }
        
        # Rest of the function remains the same
        features = {   
            'def_spread': defense_snapshot['x'].max() - defense_snapshot['x'].min(),
            'def_depth': defense_snapshot['y'].max() - defense_snapshot['y'].min(),
            'def_players_in_box': len(defense_snapshot[defense_snapshot['y'] < 5]),
            'def_players_deep': len(defense_snapshot[defense_snapshot['y'] > 10])
        }
    
        return features
        
    def train(self, dataframe):
        # Train models on basic play types (e.g., RUN vs PASS)
        for play_type in dataframe['play_type_basic'].unique():
            play_data = dataframe[dataframe['play_type_basic'] == play_type]
            
            if len(play_data) >= self.min_samples:
                # Get defensive setup for each play
                play_features = []
                for play_id in play_data['playId'].unique():
                    play_snapshot = play_data[play_data['playId'] == play_id]
                    def_features = self.get_defensive_features(play_snapshot)
                    if def_features:
                        play_features.append({
                            'playId': play_id,
                            **def_features
                        })
                
                if play_features:
                    features_df = pd.DataFrame(play_features)
                    play_data = play_data.merge(features_df, on='playId')
                    
                    X = play_data[base_features + list(def_features.keys())]
                    y = play_data['expectedPointsAdded']
                    
                    model = RandomForestRegressor(
                        n_estimators=100,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=42
                    )
                    model.fit(X, y)
                    
                    self.base_models[play_type] = {
                        'model': model,
                        'success_rate': (play_data['expectedPointsAdded'] > 0).mean()
                    }
        
        # Train models on specific play variations (e.g., DEEP_PASS_RIGHT)
        for play_type in dataframe['play_type_detailed'].unique():
            play_data = dataframe[dataframe['play_type_detailed'] == play_type]
            
            if len(play_data) >= self.min_samples:
                play_features = []
                for play_id in play_data['playId'].unique():
                    play_snapshot = play_data[play_data['playId'] == play_id]
                    def_features = self.get_defensive_features(play_snapshot)
                    if def_features:
                        play_features.append({
                            'playId': play_id,
                            **def_features
                        })
                        
                if play_features:
                    features_df = pd.DataFrame(play_features)
                    play_data = play_data.merge(features_df, on='playId')
                    
                    X = play_data[detailed_features + list(def_features.keys())]
                    y = play_data['expectedPointsAdded']
                    
                    model = RandomForestRegressor(
                        n_estimators=100,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=42
                    )
                    model.fit(X, y)
                    
                    self.detailed_models[play_type] = {
                        'model': model,
                        'success_rate': (play_data['expectedPointsAdded'] > 0).mean(),
                        'avg_epa': play_data['expectedPointsAdded'].mean(),
                        'sample_size': len(play_data)
                    }
    
    def predict(self, situation, top_n=5):
        """Recommends the best plays for the current game situation"""
        predictions = []
        
        # Analyze current defensive setup
        def_features = self.get_defensive_features(situation)
        if not def_features:
            return []
            
        # Make predictions using basic play type models
        X_base = pd.concat([
            situation[base_features].to_frame().T,
            pd.DataFrame([def_features])
        ], axis=1)
        
        for play_type, model_dict in self.base_models.items():
            try:
                pred_epa = model_dict['model'].predict(X_base)[0]
                predictions.append({
                    'play_type': play_type,
                    'expected_points': pred_epa,
                    'success_rate': model_dict['success_rate'],
                    'confidence': 'HIGH',
                    'recommendation_score': pred_epa * model_dict['success_rate']
                })
            except Exception as e:
                print(f"Error predicting base type {play_type}: {str(e)}")
        
        # Make predictions using detailed play type models
        X_detailed = pd.concat([
            situation[detailed_features].to_frame().T,
            pd.DataFrame([def_features])
        ], axis=1)
        
        for play_type, model_dict in self.detailed_models.items():
            try:
                pred_epa = model_dict['model'].predict(X_detailed)[0]
                predictions.append({
                    'play_type': play_type,
                    'expected_points': pred_epa,
                    'success_rate': model_dict['success_rate'],
                    'confidence': 'MEDIUM' if model_dict['sample_size'] >= 10 else 'LOW',
                    'sample_size': model_dict['sample_size'],
                    'recommendation_score': pred_epa * model_dict['success_rate'] * (model_dict['sample_size'] / 100)
                })
            except Exception as e:
                print(f"Error predicting detailed type {play_type}: {str(e)}")
        
        # Return top N recommendations sorted by score
        return sorted(predictions, key=lambda x: x['recommendation_score'], reverse=True)[:top_n]
    
    


# Sample 5000 rows from merged_df based on unique playIDs
sampled_df = merged_df.groupby('playId').apply(lambda x: x.sample(min(len(x), 5000 // len(merged_df['playId'].unique())), random_state=42)).reset_index(drop=True)

# Create hierarchical play categories for the sampled data
sampled_df['play_type_detailed'] = sampled_df.apply(categorize_play, axis=1)
sampled_df['play_type_basic'] = sampled_df['play_type_detailed'].apply(lambda x: x.split('_')[0])
sampled_df['formation'] = sampled_df['play_type_detailed'].apply(lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'STANDARD')

# Train the hierarchical model on the sampled data
predictor = HierarchicalPlayPredictor(min_samples=5)
predictor.train(sampled_df)

# Get recommendations for a sample situation from the sampled data
sample_situation = sampled_df.iloc[[0]]  # Use double brackets to keep it as a DataFrame
recommendations = predictor.predict(sample_situation)

# Print recommendations
print("\nPlay Recommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"\nRecommendation {i}:")
    print(f"Play Type: {rec['play_type']}")
    print(f"Expected Points: {rec['expected_points']:.2f}")
    print(f"Success Rate: {rec['success_rate']:.1%}")
    print(f"Confidence: {rec['confidence']}")
    if 'sample_size' in rec:
        print(f"Sample Size: {rec['sample_size']}")
    print(f"Recommendation Score: {rec['recommendation_score']:.2f}")


print(merged_df.head( ))
print(merged_df.isnull().sum())

# print columns
print(merged_df.columns)



# Create detailed play categories based on the play type
# This line applies the categorize_play function to each row in merged_df to create a new column 'play_type_detailed'.
merged_df['play_type_detailed'] = merged_df.apply(categorize_play, axis=1)

# This line extracts the basic play type from 'play_type_detailed' by taking the first part before the underscore.
merged_df['play_type_basic'] = merged_df['play_type_detailed'].apply(lambda x: x.split('_')[0])

# This line determines the formation type from 'play_type_detailed'. 
# If there is a second part after the underscore, it uses that; otherwise, it defaults to 'STANDARD'.
merged_df['formation'] = merged_df['play_type_detailed'].apply(lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'STANDARD')

# Train the hierarchical model
predictor = HierarchicalPlayPredictor(min_samples=5)
predictor.train(merged_df)

# Get recommendations for sample situation
sample_situation = merged_df.iloc[0]
recommendations = predictor.predict(sample_situation)

# Print recommendations
print("\nPlay Recommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"\nRecommendation {i}:")
    print(f"Play Type: {rec['play_type']}")
    print(f"Expected Points: {rec['expected_points']:.2f}")
    print(f"Success Rate: {rec['success_rate']:.1%}")
    print(f"Confidence: {rec['confidence']}")
    if 'sample_size' in rec:
        print(f"Sample Size: {rec['sample_size']}")
    print(f"Recommendation Score: {rec['recommendation_score']:.2f}")


# Generate recommendations for 10 different sample situations
for i in range(10):
    sample_situation = sampled_df.sample().iloc[0]  # Randomly sample a situation
    recommendations = predictor.predict(sample_situation)

    # Print recommendations for the sampled situation
    print(f"\nPlay Recommendations for Scenario {i + 1}:")
    for j, rec in enumerate(recommendations, 1):
        print(f"\nRecommendation {j}:")
        print(f"Play Type: {rec['play_type']}")
        print(f"Expected Points: {rec['expected_points']:.2f}")
        print(f"Success Rate: {rec['success_rate']:.1%}")
        print(f"Confidence: {rec['confidence']}")
        if 'sample_size' in rec:
            print(f"Sample Size: {rec['sample_size']}")
        print(f"Recommendation Score: {rec['recommendation_score']:.2f}")




import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_play_visualization(recommendation, ax):
    # Parse play components
    components = recommendation['play_type'].split('_')
    pass_depth = components[0]
    formation = components[1] 
    coverage = components[2] + ' ' + components[3]
    direction = components[-1]
    
    # Draw football field with grass texture
    ax.set_facecolor('#2E7D32')  # Dark green base
    
    # Create alternating grass pattern
    for i in range(20, 81, 5):
        rect = patches.Rectangle((i, 20), 5, 13.3, 
                               facecolor='#357a38' if i % 10 == 0 else '#2E7D32',
                               alpha=0.3)
        ax.add_patch(rect)
    
    # Draw yard lines
    for i in range(20, 81, 5):
        alpha = 1.0 if i % 10 == 0 else 0.3
        ax.axvline(x=i, color='white', alpha=alpha, linewidth=2)
        if i % 10 == 0:  # Add yard numbers
            ax.text(i, 21, str(i-50), color='white', ha='center', va='top', alpha=0.7)
            ax.text(i, 32.3, str(i-50), color='white', ha='center', va='bottom', alpha=0.7)
    
    # Draw hash marks
    for i in range(20, 81):
        ax.plot([i, i], [23.36, 24.36], color='white', alpha=0.3)
        ax.plot([i, i], [29.94, 28.94], color='white', alpha=0.3)
    
    # Draw line of scrimmage
    los = 50
    ax.axvline(x=los, color='yellow', linestyle='--', alpha=0.8)
    
    # Draw offensive formation
    if formation == 'SHOTGUN':
        qb_x = los - 5
        # Add RB
        ax.plot(qb_x + 1, 25.65, 'bo', markersize=8, label='RB')
        rb_pos = (qb_x + 1, 25.65)
    else:  # UNDER_CENTER
        qb_x = los - 1
        # Add RB for I-formation
        ax.plot(qb_x - 3, 26.65, 'bo', markersize=8, label='RB')
        rb_pos = (qb_x - 3, 26.65)
    
    # Draw QB
    ax.plot(qb_x, 26.65, 'ro', markersize=10, label='QB')
    qb_pos = (qb_x, 26.65)
    
    # Draw offensive line
    for i in range(-2, 3):
        ax.plot(los - 0.5, 26.65 + i*0.6, 'wo', markersize=6)
    
    # Draw receivers based on play type
    if direction == 'RIGHT':
        wr_x = los + 0.5
        wr_positions = [(wr_x, 29.5), (wr_x, 28.5)]
    else:
        wr_x = los + 0.5
        wr_positions = [(wr_x, 23.8), (wr_x, 24.8)]
    
    for pos in wr_positions:
        ax.plot(pos[0], pos[1], 'go', markersize=8, label='WR')
    
    # Calculate target position based on pass depth and direction
    if direction == 'RIGHT':
        target_x = los + {'QUICK': 5, 'MEDIUM': 15, 'DEEP': 25}[pass_depth]
        target_y = 29
    elif direction == 'LEFT':
        target_x = los - {'QUICK': 5, 'MEDIUM': 15, 'DEEP': 25}[pass_depth]
        target_y = 24
    else:  # CENTER
        target_x = los + {'QUICK': 5, 'MEDIUM': 15, 'DEEP': 25}[pass_depth]
        target_y = 26.65
    
    # Draw QB pass trajectory with arrow
    ax.arrow(qb_pos[0], qb_pos[1], 
            target_x-qb_pos[0], target_y-qb_pos[1],
            head_width=0.5, head_length=1, fc='yellow', ec='yellow',
            length_includes_head=True, alpha=0.6)
    
    # Draw receiver routes with dotted lines
    for pos in wr_positions:
        ax.plot([pos[0], target_x], [pos[1], target_y], 
                'w--', alpha=0.4, linewidth=1)
    
    # Draw RB blocking or route path
    if formation == 'SHOTGUN':
        ax.plot([rb_pos[0], rb_pos[0] + 3], [rb_pos[1], rb_pos[1]], 
                'b--', alpha=0.4, linewidth=1)
    
    # Add defensive players (basic coverage indication)
    if 'ZONE' in coverage:
        # Draw zone defenders
        for i in range(-2, 3):
            ax.plot(los + 3, 26.65 + i*1.5, 'rx', markersize=8, alpha=0.5)
    else:  # Man coverage
        # Draw man defenders matching offensive players
        for pos in wr_positions:
            ax.plot(pos[0] + 1, pos[1], 'rx', markersize=8, alpha=0.5)
    
    # Add play information in a box
    info_text = (f"Formation: {formation}\n"
                f"Pass: {pass_depth} to {direction}\n"
                f"Coverage: {coverage}\n"
                f"EPA: {recommendation['expected_points']:.2f}\n"
                f"Success: {recommendation['success_rate']:.1%}\n"
                f"Confidence: {recommendation['confidence']}")
    
    ax.text(82, 27, info_text, bbox=dict(facecolor='black', alpha=0.7),
            color='white', fontsize=8)
    
    # Set field boundaries
    ax.set_xlim(20, 90)
    ax.set_ylim(20, 33.3)
    
    # Remove axes
    ax.axis('off')

# Create figure with subplots
num_plays = len(recommendations)
fig, axes = plt.subplots(num_plays, 1, figsize=(12, 8*num_plays))

# Create visualizations for top recommendations
for i, (rec, ax) in enumerate(zip(recommendations, axes)):
    ax.set_title(f'Play Recommendation #{i+1}', color='white', pad=20)
    create_play_visualization(rec, ax)

plt.tight_layout()
plt.show()








