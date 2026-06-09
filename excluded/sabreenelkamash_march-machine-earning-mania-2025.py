import numpy as np 
import pandas as pd 
import os
import warnings
warnings.filterwarnings('ignore')


# Load conference tournament match data
conference_games = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MConferenceTourneyGames.csv')

# Convert match results into a format where each team has its own record
# Create a DataFrame for winning teams with a win label
conference_games_wins = conference_games[['WTeamID']].copy()
conference_games_wins['Win'] = 1
conference_games_wins.rename(columns={'WTeamID': 'TeamID'}, inplace=True)

# Create a DataFrame for losing teams with a loss label
conference_games_losses = conference_games[['LTeamID']].copy()
conference_games_losses['Win'] = 0
conference_games_losses.rename(columns={'LTeamID': 'TeamID'}, inplace=True)

# Combine both DataFrames to create a unified match record per team
conference_team_stats = pd.concat([conference_games_wins, conference_games_losses], ignore_index=True)

# Calculate total matches played and win rate for each team in conference tournaments
conference_summary = (
    conference_team_stats.groupby('TeamID')
    .agg(TotalGames=('Win', 'count'), ConferenceWins=('Win', 'sum'))
    .reset_index()
)

# Compute win rate in conference tournaments
conference_summary['TeamConferenceWinRate'] = (
    conference_summary['ConferenceWins'] / conference_summary['TotalGames']
)

# Add an indicator feature to specify that this data is from a conference tournament
conference_summary['IsConferenceTourney'] = 1

# Save the processed conference tournament feature data
conference_summary.to_csv('/kaggle/working/conference_summary.csv', index=False)

# Display a preview of the processed data
print("Conference tournament summary preview:")
print(conference_summary.head())



# Load tournament match data
tourney_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')

# Calculate tournament wins and losses for each team
tourney_wins = tourney_results.groupby('WTeamID').size().reset_index(name='TournamentWins')
tourney_losses = tourney_results.groupby('LTeamID').size().reset_index(name='TournamentLosses')

# Rename columns for consistency before merging
tourney_wins.rename(columns={'WTeamID': 'TeamID'}, inplace=True)
tourney_losses.rename(columns={'LTeamID': 'TeamID'}, inplace=True)

# Merge win and loss data for each team
tourney_stats = pd.merge(tourney_wins, tourney_losses, on='TeamID', how='outer')

# Replace missing values with zero and ensure integer data type
tourney_stats.fillna(0, inplace=True)
tourney_stats = tourney_stats.astype({'TournamentWins': 'int', 'TournamentLosses': 'int'})

# Calculate tournament win rate while handling cases where no games were played
tourney_stats['TournamentWinRate'] = tourney_stats.apply(
    lambda row: row['TournamentWins'] / (row['TournamentWins'] + row['TournamentLosses']) 
    if (row['TournamentWins'] + row['TournamentLosses']) > 0 else 0, axis=1
)

# Calculate average score difference in tournament games
tourney_results['ScoreDiff'] = tourney_results['WScore'] - tourney_results['LScore']
score_diff = tourney_results.groupby('WTeamID')['ScoreDiff'].mean().reset_index()
score_diff.rename(columns={'WTeamID': 'TeamID', 'ScoreDiff': 'AvgTournamentScoreDiff'}, inplace=True)

# Merge calculated features
tourney_features = pd.merge(
    tourney_stats[['TeamID', 'TournamentWinRate']], 
    score_diff, 
    on='TeamID', 
    how='left'
)

# Replace missing values in score difference with zero
tourney_features['AvgTournamentScoreDiff'] = tourney_features['AvgTournamentScoreDiff'].fillna(0)

# Save the processed tournament features
tourney_features.to_csv('/kaggle/working/tourney_features.csv', index=False)

# Display a preview of the results
print("Tournament features preview:")
print(tourney_features[['TeamID', 'TournamentWinRate', 'AvgTournamentScoreDiff']].head())



# Load detailed tournament results data
tourney_detailed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyDetailedResults.csv')

# Extract relevant columns for winning teams: TeamID, Field Goals Made (FGM), Field Goals Attempted (FGA), and Overtime count
winners = tourney_detailed[['WTeamID', 'WFGM', 'WFGA', 'NumOT']].copy()
winners.rename(columns={'WTeamID': 'TeamID', 'WFGM': 'FGM', 'WFGA': 'FGA'}, inplace=True)

# Extract relevant columns for losing teams: TeamID, Field Goals Made (FGM), Field Goals Attempted (FGA), and Overtime count
losers = tourney_detailed[['LTeamID', 'LFGM', 'LFGA', 'NumOT']].copy()
losers.rename(columns={'LTeamID': 'TeamID', 'LFGM': 'FGM', 'LFGA': 'FGA'}, inplace=True)

# Combine winners and losers into a single dataset for shot statistics
teams_shots = pd.concat([winners, losers])

# Calculate the field goal percentage for each team
teams_shots['FG%'] = np.where(teams_shots['FGA'] > 0, teams_shots['FGM'] / teams_shots['FGA'], 0)

# Compute the average field goal percentage for each team across all tournament games
fg_percentage = teams_shots.groupby('TeamID')['FG%'].mean().reset_index()
fg_percentage.rename(columns={'FG%': 'AvgFieldGoalPercentage'}, inplace=True)

# Compute the average number of overtime games per team
ot_games = teams_shots.groupby('TeamID')['NumOT'].mean().reset_index()
ot_games.rename(columns={'NumOT': 'AvgOTGames'}, inplace=True)

# Merge the computed features into a single dataset
detailed_features = fg_percentage.merge(ot_games, on='TeamID', how='left')

# Save the final dataset
detailed_features.to_csv('/kaggle/working/detailed_features.csv', index=False)

# Display the first few rows of the final dataset
print("Detailed tournament features preview:")
print(detailed_features.head())



# Load Regular Season Results Data
season_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')

# Compute the number of wins for each team
season_wins = season_results.groupby('WTeamID').size().reset_index(name='SeasonWins')
season_wins.rename(columns={'WTeamID': 'TeamID'}, inplace=True)

# Compute the number of losses for each team
season_losses = season_results.groupby('LTeamID').size().reset_index(name='SeasonLosses')
season_losses.rename(columns={'LTeamID': 'TeamID'}, inplace=True)

# Merge wins and losses into a single dataset (ensuring all teams are included)
season_stats = pd.merge(season_wins, season_losses, on='TeamID', how='outer').fillna(0)

# Ensure numerical data types
season_stats[['SeasonWins', 'SeasonLosses']] = season_stats[['SeasonWins', 'SeasonLosses']].astype(int)

# Compute the win rate for each team
season_stats['SeasonWinRate'] = season_stats['SeasonWins'] / (season_stats['SeasonWins'] + season_stats['SeasonLosses'])

# Calculate the average points difference in the regular season
season_results['ScoreDiff'] = season_results['WScore'] - season_results['LScore']
avg_score_diff = season_results.groupby('WTeamID')['ScoreDiff'].mean().reset_index()
avg_score_diff.rename(columns={'WTeamID': 'TeamID', 'ScoreDiff': 'AvgSeasonScoreDiff'}, inplace=True)

# Merge the computed features into a single dataset
season_features = season_stats.merge(avg_score_diff, on='TeamID', how='left')

# Fill missing values for score difference with 0
season_features['AvgSeasonScoreDiff'] = season_features['AvgSeasonScoreDiff'].fillna(0)

# Save the final dataset
season_features.to_csv('/kaggle/working/season_features.csv', index=False)

# Display a preview of the dataset
print("Regular Season Features Preview:")
print(season_features.head())



# Load detailed results for the regular season
season_detailed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv')

# Extract relevant stats for the winning team (field goals made/attempted, points scored, and points allowed)
winners = season_detailed[['WTeamID', 'WFGM', 'WFGA', 'LScore', 'WScore']].copy()
winners.rename(columns={'WTeamID': 'TeamID', 'WFGM': 'FGM', 'WFGA': 'FGA', 
                          'LScore': 'PointsAllowed', 'WScore': 'PointsScored'}, inplace=True)

# Extract relevant stats for the losing team (same stats as winners)
losers = season_detailed[['LTeamID', 'LScore', 'WFGM', 'WFGA', 'WScore']].copy()
losers.rename(columns={'LTeamID': 'TeamID', 'LScore': 'PointsScored', 'WFGM': 'FGM', 
                         'WFGA': 'FGA', 'WScore': 'PointsAllowed'}, inplace=True)

# Combine winners and losers into a single dataset
teams_stats = pd.concat([winners, losers], ignore_index=True)

# Ensure FGA values are non-negative
teams_stats['FGA'] = teams_stats['FGA'].clip(lower=0)

# Calculate field goal percentage (avoiding division by zero)
teams_stats['FG%'] = np.where(teams_stats['FGA'] > 0, teams_stats['FGM'] / teams_stats['FGA'], 0)

# Compute the average field goal percentage for each team
fg_percentage_season = teams_stats.groupby('TeamID')['FG%'].mean().reset_index().rename(
    columns={'FG%': 'AvgFieldGoalPercentageSeason'}
)

# Compute the average points allowed (defensive strength) for each team
defensive_strength = teams_stats.groupby('TeamID')['PointsAllowed'].mean().reset_index().rename(
    columns={'PointsAllowed': 'AvgPointsAllowed'}
)

# Compute the average points scored (offensive strength) for each team
offensive_strength= teams_stats.groupby('TeamID')['PointsScored'].mean().reset_index().rename(
    columns={'PointsScored': 'AvgPointsScored'}
)

# Merge all extracted features into a single dataset
season_detailed_features = fg_percentage_season.merge(defensive_strength, on='TeamID', how='left')
season_detailed_features = season_detailed_features.merge(offensive_strength, on='TeamID', how='left')

# Fill NaN values with 0
season_detailed_features = season_detailed_features.fillna(0)

# Save the feature dataset
season_detailed_features.to_csv('/kaggle/working/season_detailed_features.csv', index=False)

# Display the first few rows of the final dataset
print(season_detailed_features.head())


# Load the secondary tournament results
secondary_tourney_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MSecondaryTourneyCompactResults.csv')

# Calculate the number of wins and losses in the secondary tournament
secondary_tourney_wins = secondary_tourney_results.groupby('WTeamID').size().reset_index(name='SecondaryTourneyWins')
secondary_tourney_losses = secondary_tourney_results.groupby('LTeamID').size().reset_index(name='SecondaryTourneyLosses')

# Merge win and loss data
secondary_tourney_stats = pd.merge(
    secondary_tourney_wins, secondary_tourney_losses, 
    left_on='WTeamID', right_on='LTeamID', how='outer'
).fillna(0)

# Assign TeamID correctly
secondary_tourney_stats['TeamID'] = secondary_tourney_stats['WTeamID'].fillna(secondary_tourney_stats['LTeamID']).astype(int)

# Compute win rate in the secondary tournament, handling division by zero
secondary_tourney_stats['SecondaryTourneyWinRate'] = np.where(
    (secondary_tourney_stats['SecondaryTourneyWins'] + secondary_tourney_stats['SecondaryTourneyLosses']) > 0,
    secondary_tourney_stats['SecondaryTourneyWins'] / (secondary_tourney_stats['SecondaryTourneyWins'] + secondary_tourney_stats['SecondaryTourneyLosses']),
    0
)

# Define games played under pressure:
# Conditions: overtime (NumOT > 0) or close games (point difference <= 5)
secondary_tourney_results['ScoreDiff'] = secondary_tourney_results['WScore'] - secondary_tourney_results['LScore']
pressure_games = secondary_tourney_results[
    (secondary_tourney_results['NumOT'] > 0) | (secondary_tourney_results['ScoreDiff'].abs() <= 5)
]

# Compute win and loss counts under pressure
pressure_wins = pressure_games.groupby('WTeamID').size().reset_index(name='PressureWins')
pressure_losses = pressure_games.groupby('LTeamID').size().reset_index(name='PressureLosses')

# Merge pressure-based performance data
pressure_stats = pd.merge(
    pressure_wins, pressure_losses, 
    left_on='WTeamID', right_on='LTeamID', how='outer'
).fillna(0)

# Assign TeamID correctly
pressure_stats['TeamID'] = pressure_stats['WTeamID'].fillna(pressure_stats['LTeamID']).astype(int)

# Compute win rate under pressure, handling division by zero
pressure_stats['SecondaryTourneyPressureWinRate'] = np.where(
    (pressure_stats['PressureWins'] + pressure_stats['PressureLosses']) > 0,
    pressure_stats['PressureWins'] / (pressure_stats['PressureWins'] + pressure_stats['PressureLosses']),
    0
)

# Merge extracted features
secondary_tourney_features = pd.merge(
    secondary_tourney_stats[['TeamID', 'SecondaryTourneyWinRate']],
    pressure_stats[['TeamID', 'SecondaryTourneyPressureWinRate']],
    on='TeamID',
    how='left'
).fillna(0)

# Save the feature file
secondary_tourney_features.to_csv('/kaggle/working/secondary_tourney_features.csv', index=False)

# Preview results
print("Secondary Tournament Features Preview:")
print(secondary_tourney_features.head())



# Load men's team data
teams = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')

# Check if required columns exist
required_columns = {'TeamID', 'FirstD1Season', 'LastD1Season'}
if not required_columns.issubset(teams.columns):
    missing_columns = required_columns - set(teams.columns)
    raise KeyError(f"The MTeams file is missing the following columns: {missing_columns}")

# Calculate each team's experience in the first division (D1)
teams['D1Experience'] = teams['LastD1Season'] - teams['FirstD1Season'] + 1

# Save extracted feature file
teams.to_csv('/kaggle/working/teams.csv', index=False)

# Preview final results (check if 'TeamName' exists before using it)
preview_columns = ['TeamID', 'D1Experience']
if 'TeamName' in teams.columns:
    preview_columns.insert(1, 'TeamName')

print("Teams Data Preview:")
print(teams[preview_columns].head())



# Load all feature files
conference_summary = pd.read_csv('conference_summary.csv', low_memory=False)
tourney_features = pd.read_csv('tourney_features.csv', low_memory=False)
detailed_features = pd.read_csv('detailed_features.csv', low_memory=False)
season_features = pd.read_csv('season_features.csv', low_memory=False)
season_detailed_features = pd.read_csv('season_detailed_features.csv', low_memory=False)
secondary_tourney_features = pd.read_csv('secondary_tourney_features.csv', low_memory=False)
teams = pd.read_csv('teams.csv', low_memory=False)

# Merge all files based on TeamID
merged_df = (
    conference_summary[['TeamID', 'TeamConferenceWinRate']]
    .merge(tourney_features[['TeamID', 'TournamentWinRate', 'AvgTournamentScoreDiff']], on='TeamID', how='left')
    .merge(detailed_features[['TeamID', 'AvgFieldGoalPercentage', 'AvgOTGames']], on='TeamID', how='left')
    .merge(season_features[['TeamID', 'SeasonWinRate', 'AvgSeasonScoreDiff']], on='TeamID', how='left')
    .merge(season_detailed_features[['TeamID', 'AvgFieldGoalPercentageSeason', 'AvgPointsAllowed', 'AvgPointsScored']], on='TeamID', how='left')  # âœ… Ø¥Ø¶Ø§Ù�Ø© AvgPointsScored
    .merge(secondary_tourney_features[['TeamID', 'SecondaryTourneyWinRate', 'SecondaryTourneyPressureWinRate']], on='TeamID', how='left')
    .merge(teams[['TeamID', 'D1Experience']], on='TeamID', how='left')
)

# Fill missing values with 0 (important to avoid NaN issues)
merged_df.fillna(0, inplace=True)

# Save the merged file
merged_df.to_csv('/kaggle/working/merged_team_features.csv', index=False)

# Show preliminary results
print("Merged Data Preview:")
print(merged_df.head())

# Check for missing values
print("\nData Overview:")
print(merged_df.info())



from itertools import combinations

# Load the dataset containing team features
file_path = "merged_team_features.csv"
df = pd.read_csv(file_path)

# Generate all possible team matchups (unique pairs of teams)
team_pairs = list(combinations(df['TeamID'], 2))

# Define column names for the new DataFrame
columns = ['TeamID1', 'TeamID2'] + [f'{col}_diff' for col in df.columns if col != 'TeamID'] + ['WinProbability']

# Initialize a list to store the processed matchups
data = []

# Iterate through all possible team pairs
for team1, team2 in team_pairs:
    # Retrieve data for both teams
    team1_data = df[df['TeamID'] == team1].squeeze()
    team2_data = df[df['TeamID'] == team2].squeeze()
    
    # Skip if data for any team is missing
    if team1_data.empty or team2_data.empty:
        continue  
    
    # Store team IDs
    row = {'TeamID1': team1, 'TeamID2': team2}
    
    # Compute feature differences between the two teams
    feature_diffs = []
    for col in df.columns:
        if col != 'TeamID':
            diff = team1_data[col] - team2_data[col]
            if pd.isna(diff):  # Handle missing values
                diff = 0
            row[f'{col}_diff'] = diff
            feature_diffs.append(diff)
    
    # Compute the estimated win probability using the logistic function
    feature_sum = sum(feature_diffs)
    
    if np.isfinite(feature_sum):  # Ensure the value is valid before applying the function
        row['WinProbability'] = round(1 / (1 + np.exp(-feature_sum)), 2)
    else:
        row['WinProbability'] = 0.5  # Assign a neutral probability in case of invalid values
    
    # Append the computed row to the data list
    data.append(row)

# Convert the list into a DataFrame
pairs_df = pd.DataFrame(data, columns=columns)

# Save the results to a CSV file
pairs_df.to_csv("/kaggle/working/team_pairs_diff.csv", index=False)

# Print success message
print(" The file 'team_pairs_diff.csv' was created successfully!")



team_pairs_diff = pd.read_csv("/kaggle/working/team_pairs_diff.csv")
team_pairs_diff


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import optuna

# Define feature columns (all '_diff' columns) and target variable ('WinProbability')
feature_columns = [col for col in pairs_df.columns if col.endswith('_diff')]
X = pairs_df[feature_columns]
y = pairs_df['WinProbability']

# Split the dataset into training (80%) and testing (20%) ensuring reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values using SimpleImputer (replace NaNs with the median)
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

print(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

# Define the Optuna optimization function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),  # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),  # L2 regularization
        'random_state': 42
    }
    
    # Train the model with early stopping to prevent overfitting
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=False)
    
    # Make predictions and clip values between 0 and 1
    y_pred = np.clip(model.predict(X_test), 0, 1)
    
    # Return Mean Squared Error (MSE) for evaluation
    return mean_squared_error(y_test, y_pred)

# Run Optuna to optimize hyperparameters
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=75)  # Increased trials for better optimization

# Retrieve the best hyperparameters found by Optuna
best_params = study.best_params
print("Best Hyperparameters: ", best_params)

# Train the final model with optimized hyperparameters and early stopping
best_model = XGBRegressor(**best_params)
best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=False)

# Generate final predictions and ensure values are between 0 and 1
y_pred_full = np.clip(best_model.predict(X), 0, 1)
pairs_df['WinProbability'] = y_pred_full

# Format output to match the required submission format
season = 2025  # Adjust the season as needed
pairs_df['ID'] = pairs_df.apply(lambda row: f"{season}_{int(row['TeamID1'])}_{int(row['TeamID2'])}", axis=1)
pairs_df['Pred'] = pairs_df['WinProbability'].round(1)  # Round probabilities to one decimal place
submission_df = pairs_df[['ID', 'Pred']]

# Save the submission file
submission_df.to_csv("/kaggle/working/submission_m.csv", index=False)
print("The file 'submission_m.csv' was created successfully!")

# Calculate and display MSE on both training and test sets
y_pred_train = np.clip(best_model.predict(X_train), 0, 1)
y_pred_test = np.clip(best_model.predict(X_test), 0, 1)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
print(f"Mean Squared Error on Training Set: {train_mse:.4f}")
print(f"Mean Squared Error on Test Set: {test_mse:.4f}")


submission_m = pd.read_csv("/kaggle/working/submission_m.csv") 
submission_m


# Download the file
submission_m = pd.read_csv('/kaggle/working/submission_m.csv')

# Check for blank values in ID column
missing_id = submission_m['ID'].isna().sum()
print(f"Number of empty values in ID: {missing_id}")

# Extract TeamID1 and TeamID2 from ID
submission_m[['Season', 'TeamID1', 'TeamID2']] = submission_m['ID'].str.split('_', expand=True)

# Check for empty values in TeamID1 and TeamID2
missing_teamid1 = submission_m['TeamID1'].isna().sum()
missing_teamid2 = submission_m['TeamID2'].isna().sum()

print(f"Number of empty values â€‹â€‹in TeamID1: {missing_teamid1}")
print(f"Number of empty values â€‹â€‹in TeamID2: {missing_teamid2}")

# If there are no empty values
if missing_id == 0 and missing_teamid1 == 0 and missing_teamid2 == 0:
    print("\nThere are no empty values in ID, TeamID1, or TeamID2.")
else:
    print("\nThere are blank values in some columns, check the data.")


# Load conference tournament game data
conference_games_w = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WConferenceTourneyGames.csv')

# Extract winning teams and mark them with "Win = 1"
conference_games_wins_w = conference_games_w[['WTeamID']].copy()
conference_games_wins_w['Win'] = 1
conference_games_wins_w.rename(columns={'WTeamID': 'TeamID'}, inplace=True)

# Extract losing teams and mark them with "Win = 0"
conference_games_losses_w = conference_games_w[['LTeamID']].copy()
conference_games_losses_w['Win'] = 0
conference_games_losses_w.rename(columns={'LTeamID': 'TeamID'}, inplace=True)

# Combine winning and losing teams into one dataset, ensuring each team has its match record
conference_team_stats_w = pd.concat([conference_games_wins_w, conference_games_losses_w])

# Compute total games played and number of wins per team
conference_summary_w = (
    conference_team_stats_w.groupby('TeamID')
    .agg(TotalGames=('Win', 'count'), ConferenceWins=('Win', 'sum'))
    .reset_index()
)

# Ensure there are no division errors by replacing NaN values with 0
conference_summary_w['TotalGames'] = conference_summary_w['TotalGames'].fillna(0)
conference_summary_w['ConferenceWins'] = conference_summary_w['ConferenceWins'].fillna(0)

# Compute win rate while avoiding division by zero
conference_summary_w['TeamConferenceWinRate'] = (
    conference_summary_w['ConferenceWins'] / conference_summary_w['TotalGames']
).fillna(0)  # In case any division by zero occurs, replace NaN with 0

# Add a feature that specifies that this data is for a conference tournament
conference_summary_w['IsConferenceTourney'] = 1

# Save the extracted conference statistics
conference_summary_w.to_csv('/kaggle/working/conference_summary_w.csv', index=False)

# Show preliminary results
conference_summary_w.head()


# Download tournament match data under pressure
tourney_results_w = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv')

# Calculate tournament wins and losses for each team
tourney_wins_w = tourney_results_w.groupby('WTeamID').size().reset_index(name='TournamentWins')
tourney_losses_w = tourney_results_w.groupby('LTeamID').size().reset_index(name='TournamentLosses')

# Rename columns for consistency before merging
tourney_wins_w.rename(columns={'WTeamID': 'TeamID'}, inplace=True)
tourney_losses_w.rename(columns={'LTeamID': 'TeamID'}, inplace=True)

# Merge win and loss data for each team
tourney_stats_w = pd.merge(tourney_wins_w, tourney_losses_w, on='TeamID', how='outer')

# Replace missing values with zero and ensure integer data type
tourney_stats_w.fillna(0, inplace=True)
tourney_stats_w = tourney_stats_w.astype({'TournamentWins': 'int', 'TournamentLosses': 'int'})

# Calculate tournament win rate while handling cases where no games were played
tourney_stats_w['TournamentWinRate'] = tourney_stats_w.apply(
    lambda row: row['TournamentWins'] / (row['TournamentWins'] + row['TournamentLosses'])
    if (row['TournamentWins'] + row['TournamentLosses']) > 0 else 0, axis=1
)

# Calculate average score difference in tournament games
tourney_results_w['ScoreDiff'] = tourney_results_w['WScore'] - tourney_results_w['LScore']
score_diff_w = tourney_results_w.groupby('WTeamID')['ScoreDiff'].mean().reset_index()
score_diff_w.rename(columns={'WTeamID': 'TeamID', 'ScoreDiff': 'AvgTournamentScoreDiff'}, inplace=True)

# Merge calculated features
tourney_features_w = pd.merge(
    tourney_stats_w[['TeamID', 'TournamentWinRate']], 
    score_diff_w, 
    on='TeamID', 
    how='left'
)

# Replace missing values in score difference with zero
tourney_features_w['AvgTournamentScoreDiff'] = tourney_features_w['AvgTournamentScoreDiff'].fillna(0)

# Save the processed tournament features
tourney_features_w.to_csv('/kaggle/working/tourney_features_w.csv', index=False)

# Display a preview of the results
print("Tournament features preview:")
print(tourney_features_w[['TeamID', 'TournamentWinRate', 'AvgTournamentScoreDiff']].head())



# Load detailed tournament results data
tourney_detailed_w = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyDetailedResults.csv')

# Create a copy for the winning teams containing team ID, field goals made (FGM), field goals attempted (FGA), and number of overtime games (NumOT)
winners_w = tourney_detailed_w[['WTeamID', 'WFGM', 'WFGA', 'NumOT']].copy()
winners_w.rename(columns={'WTeamID': 'TeamID', 'WFGM': 'FGM', 'WFGA': 'FGA'}, inplace=True)

# Create a copy for the losing teams containing team ID, points scored (used as FGM approximation), and number of overtime games
losers_w = tourney_detailed_w[['LTeamID', 'LScore', 'NumOT']].copy()
losers_w.rename(columns={'LTeamID': 'TeamID', 'LScore': 'FGM'}, inplace=True)

# Estimate field goal attempts for losing teams using an assumed 45% shooting percentage
losers_w['FGA'] = losers_w['FGM'] / 0.45  # Estimated based on general shooting accuracy

# Merge winners and losers into a single dataset
teams_shots_w = pd.concat([winners_w, losers_w], ignore_index=True)

# Handle missing or invalid values in FGA before calculating FG%
teams_shots_w['FGA'] = teams_shots_w['FGA'].fillna(1)  # Replace NaN with 1 to avoid division errors
teams_shots_w['FGA'] = teams_shots_w['FGA'].clip(lower=1)  # Ensure FGA is at least 1 to prevent division issues

# Compute field goal percentage (FG%)
teams_shots_w['FG%'] = teams_shots_w['FGM'] / teams_shots_w['FGA']

# Fill missing values in NumOT
teams_shots_w['NumOT'] = teams_shots_w['NumOT'].fillna(0)

# Calculate the average field goal percentage per team
fg_percentage_w = teams_shots_w.groupby('TeamID')['FG%'].mean().reset_index().rename(columns={'FG%': 'AvgFieldGoalPercentage'})

# Calculate the average number of overtime games per team
ot_games_w = teams_shots_w.groupby('TeamID')['NumOT'].mean().reset_index().rename(columns={'NumOT': 'AvgOTGames'})

# Merge the extracted features
detailed_features_w = fg_percentage_w.merge(ot_games_w, on='TeamID', how='left')

# Save the features to a CSV file
detailed_features_w.to_csv('/kaggle/working/detailed_features_w.csv', index=False)

# Display the first few rows of the final dataset
print(detailed_features_w.head())



# Load Regular Season Results Data
season_results_w = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv')

# Calculate Regular Season Wins and Losses
season_wins_w = season_results_w.groupby('WTeamID').size().reset_index(name='SeasonWins')
season_losses_w = season_results_w.groupby('LTeamID').size().reset_index(name='SeasonLosses')

# Rename columns to unify TeamID
season_wins_w.rename(columns={'WTeamID': 'TeamID'}, inplace=True)
season_losses_w.rename(columns={'LTeamID': 'TeamID'}, inplace=True)

# Merge wins and losses, replacing NaN with 0
season_stats_w = pd.merge(season_wins_w, season_losses_w, on='TeamID', how='outer').fillna(0)

# Convert columns to integer type after filling NaN values
season_stats_w[['SeasonWins', 'SeasonLosses']] = season_stats_w[['SeasonWins', 'SeasonLosses']].astype(int)

# Calculate Season Win Rate (avoid division by zero)
season_stats_w['SeasonWinRate'] = np.where(
    (season_stats_w['SeasonWins'] + season_stats_w['SeasonLosses']) > 0,
    season_stats_w['SeasonWins'] / (season_stats_w['SeasonWins'] + season_stats_w['SeasonLosses']),
    0
)

# Calculate average points difference for both winning and losing teams
season_results_w['ScoreDiff'] = season_results_w['WScore'] - season_results_w['LScore']
season_results_w['LScoreDiff'] = -season_results_w['ScoreDiff']

# Compute mean score difference for winners and losers
avg_score_diff_win = season_results_w.groupby('WTeamID')['ScoreDiff'].mean().reset_index()
avg_score_diff_loss = season_results_w.groupby('LTeamID')['LScoreDiff'].mean().reset_index()

# Rename columns
avg_score_diff_win.rename(columns={'WTeamID': 'TeamID', 'ScoreDiff': 'AvgSeasonScoreDiff'}, inplace=True)
avg_score_diff_loss.rename(columns={'LTeamID': 'TeamID', 'LScoreDiff': 'AvgSeasonScoreDiff'}, inplace=True)

# Combine winners and losers score differences and take the mean per team
avg_score_diff_w = pd.concat([avg_score_diff_win, avg_score_diff_loss]).groupby('TeamID')['AvgSeasonScoreDiff'].mean().reset_index()

# Merge extracted features
season_features_w = season_stats_w.merge(avg_score_diff_w, on='TeamID', how='left')

# Replace NaN values in score difference with zero
season_features_w['AvgSeasonScoreDiff'] = season_features_w['AvgSeasonScoreDiff'].fillna(0)

# Save feature file
season_features_w.to_csv('/kaggle/working/season_features_w.csv', index=False)

# Data Preview
print(season_features_w.head())


# Load detailed results for the regular season
season_detailed_w = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonDetailedResults.csv')

# Extract relevant stats for the winning team (field goals made/attempted, points scored, and points allowed)
winners_w = season_detailed_w[['WTeamID', 'WFGM', 'WFGA', 'LScore', 'WScore']].copy()
winners_w.rename(columns={'WTeamID': 'TeamID', 'WFGM': 'FGM', 'WFGA': 'FGA', 
                          'LScore': 'PointsAllowed', 'WScore': 'PointsScored'}, inplace=True)

# Extract relevant stats for the losing team (same stats as winners)
losers_w = season_detailed_w[['LTeamID', 'LScore', 'WFGM', 'WFGA', 'WScore']].copy()
losers_w.rename(columns={'LTeamID': 'TeamID', 'LScore': 'PointsScored', 'WFGM': 'FGM', 
                         'WFGA': 'FGA', 'WScore': 'PointsAllowed'}, inplace=True)

# Combine winners and losers into a single dataset
teams_stats_w = pd.concat([winners_w, losers_w], ignore_index=True)

# Ensure FGA values are non-negative
teams_stats_w['FGA'] = teams_stats_w['FGA'].clip(lower=0)

# Calculate field goal percentage (avoiding division by zero)
teams_stats_w['FG%'] = np.where(teams_stats_w['FGA'] > 0, teams_stats_w['FGM'] / teams_stats_w['FGA'], 0)

# Compute the average field goal percentage for each team
fg_percentage_season_w = teams_stats_w.groupby('TeamID')['FG%'].mean().reset_index().rename(
    columns={'FG%': 'AvgFieldGoalPercentageSeason'}
)

# Compute the average points allowed (defensive strength) for each team
defensive_strength_w = teams_stats_w.groupby('TeamID')['PointsAllowed'].mean().reset_index().rename(
    columns={'PointsAllowed': 'AvgPointsAllowed'}
)

# Compute the average points scored (offensive strength) for each team
offensive_strength_w = teams_stats_w.groupby('TeamID')['PointsScored'].mean().reset_index().rename(
    columns={'PointsScored': 'AvgPointsScored'}
)

# Merge all extracted features into a single dataset
season_detailed_features_w = fg_percentage_season_w.merge(defensive_strength_w, on='TeamID', how='left')
season_detailed_features_w = season_detailed_features_w.merge(offensive_strength_w, on='TeamID', how='left')

# Fill NaN values with 0
season_detailed_features_w = season_detailed_features_w.fillna(0)

# Save the feature dataset
season_detailed_features_w.to_csv('/kaggle/working/season_detailed_features_w.csv', index=False)

# Display the first few rows of the final dataset
print(season_detailed_features_w.head())



# Load secondary tournament results
file_path = "/kaggle/input/march-machine-learning-mania-2025/WSecondaryTourneyCompactResults.csv"
secondary_tourney_results_w = pd.read_csv(file_path)

# Calculate the number of wins and losses in the secondary tournament
secondary_tourney_wins_w = secondary_tourney_results_w.groupby('WTeamID').size().reset_index(name='SecondaryTourneyWins')
secondary_tourney_losses_w = secondary_tourney_results_w.groupby('LTeamID').size().reset_index(name='SecondaryTourneyLosses')

# Rename columns to standardize TeamID across wins and losses
secondary_tourney_wins_w.rename(columns={'WTeamID': 'TeamID'}, inplace=True)
secondary_tourney_losses_w.rename(columns={'LTeamID': 'TeamID'}, inplace=True)

# Merge win and loss statistics for each team
secondary_tourney_stats_w = pd.merge(secondary_tourney_wins_w, secondary_tourney_losses_w, on='TeamID', how='outer').fillna(0)

# Calculate win rate in the secondary tournament, avoiding division by zero
secondary_tourney_stats_w['SecondaryTourneyWinRate'] = np.where(
    (secondary_tourney_stats_w['SecondaryTourneyWins'] + secondary_tourney_stats_w['SecondaryTourneyLosses']) > 0,
    secondary_tourney_stats_w['SecondaryTourneyWins'] / (secondary_tourney_stats_w['SecondaryTourneyWins'] + secondary_tourney_stats_w['SecondaryTourneyLosses']),
    0
)

# Identify games played under pressure (either in overtime or with a score difference of 5 points or less)
secondary_tourney_results_w['ScoreDiff'] = secondary_tourney_results_w['WScore'] - secondary_tourney_results_w['LScore']
pressure_games_w = secondary_tourney_results_w[
    (secondary_tourney_results_w['NumOT'] > 0) | (secondary_tourney_results_w['ScoreDiff'].abs() <= 5)
]

# Calculate the number of wins and losses in pressure situations
pressure_wins_w = pressure_games_w.groupby('WTeamID').size().reset_index(name='PressureWins')
pressure_losses_w = pressure_games_w.groupby('LTeamID').size().reset_index(name='PressureLosses')

# Rename columns to standardize TeamID across wins and losses in pressure games
pressure_wins_w.rename(columns={'WTeamID': 'TeamID'}, inplace=True)
pressure_losses_w.rename(columns={'LTeamID': 'TeamID'}, inplace=True)

# Merge pressure game statistics
pressure_stats_w = pd.merge(pressure_wins_w, pressure_losses_w, on='TeamID', how='outer').fillna(0)

# Calculate win rate under pressure, avoiding division by zero
pressure_stats_w['SecondaryTourneyPressureWinRate'] = np.where(
    (pressure_stats_w['PressureWins'] + pressure_stats_w['PressureLosses']) > 0,
    pressure_stats_w['PressureWins'] / (pressure_stats_w['PressureWins'] + pressure_stats_w['PressureLosses']),
    0
)

# Merge secondary tournament statistics with pressure game statistics
secondary_tourney_features_w = pd.merge(
    secondary_tourney_stats_w[['TeamID', 'SecondaryTourneyWinRate']],
    pressure_stats_w[['TeamID', 'SecondaryTourneyPressureWinRate']],
    on='TeamID',
    how='left'
).fillna(0)

# Save the final dataset with extracted features
secondary_tourney_features_w.to_csv('/kaggle/working/secondary_tourney_features_w.csv', index=False)

# Display the first few rows of the final dataset
print(secondary_tourney_features_w.head())



# Load extracted feature files efficiently
def load_feature_file(filename, columns):
    return pd.read_csv(filename, usecols=columns)

# Define required columns for each file
feature_files = {
    'conference_summary_w.csv': ['TeamID', 'TeamConferenceWinRate'],
    'tourney_features_w.csv': ['TeamID', 'TournamentWinRate', 'AvgTournamentScoreDiff'],
    'detailed_features_w.csv': ['TeamID', 'AvgFieldGoalPercentage', 'AvgOTGames'],
    'season_features_w.csv': ['TeamID', 'SeasonWinRate', 'AvgSeasonScoreDiff'],
    'season_detailed_features_w.csv': ['TeamID', 'AvgFieldGoalPercentageSeason', 'AvgPointsAllowed', 'AvgPointsScored'],
    'secondary_tourney_features_w.csv': ['TeamID', 'SecondaryTourneyWinRate', 'SecondaryTourneyPressureWinRate']
}

# Load and merge feature data
merged_df_w = None
for file, cols in feature_files.items():
    df = load_feature_file(file, cols)
    merged_df_w = df if merged_df_w is None else merged_df_w.merge(df, on='TeamID', how='left')

# Check the number of unique teams in each file
for file_name, cols in feature_files.items():
    df = load_feature_file(file_name, cols)
    print(f"{file_name}: {df['TeamID'].nunique()} unique teams")

# Print missing values before filling
print("Missing values before filling:", merged_df_w.isna().sum())

# Fill missing values with column mean, then replace any remaining NaN with 0
merged_df_w.fillna(merged_df_w.mean(numeric_only=True), inplace=True)
merged_df_w.fillna(0, inplace=True)

# Print missing values after filling
print("Missing values after filling:", merged_df_w.isna().sum())

# Save the merged file
merged_df_w.to_csv('/kaggle/working/merged_team_features_w.csv', index=False)

# Show preliminary results
print(merged_df_w.head())



from itertools import combinations

# Load merged features
file_path = "/kaggle/working/merged_team_features_w.csv"
df = pd.read_csv(file_path)

# Extract all possible team pairs efficiently
team_pairs_w = combinations(df['TeamID'], 2)

# Create a DataFrame to store the differences
columns_w = ['TeamID1', 'TeamID2'] + [f'{col}_diff' for col in df.columns if col != 'TeamID'] + ['WinProbability']
pairs_df_w = []

# Convert df to a dictionary with TeamID as index for faster lookups
df_dict = df.set_index('TeamID').to_dict(orient='index')

# Process team pairs
for team1, team2 in team_pairs_w:
    team1_data_w = df_dict[team1]
    team2_data_w = df_dict[team2]

    row = {
        'TeamID1': team1,
        'TeamID2': team2,
    }

    feature_diffs_w = []
    for col in df.columns:
        if col != 'TeamID':
            diff = team1_data_w[col] - team2_data_w[col]
            row[f'{col}_diff'] = diff
            feature_diffs_w.append(diff)

    # Compute win probability using sigmoid function
    row['WinProbability'] = round(1 / (1 + np.exp(-sum(feature_diffs_w))), 2)

    pairs_df_w.append(row)

# Convert to DataFrame
pairs_df_w = pd.DataFrame(pairs_df_w, columns=columns_w)

# Save to CSV
pairs_df_w.to_csv("/kaggle/working/team_pairs_diff_w.csv", index=False)

print("The file team_pairs_diff_w.csv was created successfully!")



team_pairs_diff_w = pd.read_csv("/kaggle/working/team_pairs_diff_w.csv")
team_pairs_diff_w


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import optuna

# Define feature columns (all '_diff' columns) and target variable ('WinProbability')
feature_columns = [col for col in pairs_df_w.columns if col.endswith('_diff')]
XW = pairs_df_w[feature_columns]
yw = pairs_df_w['WinProbability']

# Split the dataset into training (80%) and testing (20%) ensuring reproducibility
XW_train, XW_test, yw_train, yw_test = train_test_split(XW, yw, test_size=0.2, random_state=42)

# Handle missing values using SimpleImputer (replace NaNs with the median)
imputer = SimpleImputer(strategy='median')
XW_train = imputer.fit_transform(XW_train)
XW_test = imputer.transform(XW_test)

print(f"Training set size: {XW_train.shape}, Testing set size: {XW_test.shape}")

# Define the Optuna optimization function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15),  # Reduced for stability
        'max_depth': trial.suggest_int('max_depth', 5, 6),  # Reduced complexity
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),  # Increased for better generalization
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),  # Increased for better generalization
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),  # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 7.0, 15.0),  # Increased L2 regularization to reduce overfitting
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  # Added to prevent overfitting
        'random_state': 42
    }
    
    # Train the model with early stopping to prevent overfitting
    model = XGBRegressor(**params)
    model.fit(XW_train, yw_train, eval_set=[(XW_test, yw_test)], early_stopping_rounds=30, verbose=False)
    
    # Make predictions and clip values between 0 and 1
    yw_pred = np.clip(model.predict(XW_test), 0, 1)
    
    # Return Mean Squared Error (MSE) for evaluation
    return mean_squared_error(yw_test, yw_pred)

# Run Optuna to optimize hyperparameters
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=75)  # Increased trials for better optimization

# Retrieve the best hyperparameters found by Optuna
best_params = study.best_params
print("Best Hyperparameters: ", best_params)

# Train the final model with optimized hyperparameters and early stopping
best_model = XGBRegressor(**best_params)
best_model.fit(XW_train, yw_train, eval_set=[(XW_test, yw_test)], early_stopping_rounds=30, verbose=False)

# Generate final predictions and ensure values are between 0 and 1
yw_pred_full = np.clip(best_model.predict(XW), 0, 1)
pairs_df_w['WinProbability'] = yw_pred_full

# Format output to match the required submission format
season = 2025  # Adjust the season as needed
pairs_df_w['ID'] = pairs_df_w.apply(lambda row: f"{season}_{int(row['TeamID1'])}_{int(row['TeamID2'])}", axis=1)
pairs_df_w['Pred'] = pairs_df_w['WinProbability'].round(1)  # Round probabilities to one decimal place
submission_df_w = pairs_df_w[['ID', 'Pred']]

# Save the submission file
submission_df_w.to_csv("/kaggle/working/submission_w.csv", index=False)
print("The file 'submission_w.csv' was created successfully!")

# Calculate and display MSE on both training and test sets
yw_pred_train = np.clip(best_model.predict(XW_train), 0, 1)
yw_pred_test = np.clip(best_model.predict(XW_test), 0, 1)
train_mse = mean_squared_error(yw_train, yw_pred_train)
test_mse = mean_squared_error(yw_test, yw_pred_test)
print(f"Mean Squared Error on Training Set: {train_mse:.4f}")
print(f"Mean Squared Error on Test Set: {test_mse:.4f}")



submission_w = pd.read_csv("/kaggle/working/submission_w.csv") 
submission_w


# Download the file
submission_w = pd.read_csv('submission_w.csv')

# Check for blank values â€‹â€‹in ID column
missing_id = submission_w['ID'].isna().sum()
print(f"Number of empty values â€‹â€‹in ID: {missing_id}")

# Extract TeamID1 and TeamID2 from ID
submission_w[['Season', 'TeamID1', 'TeamID2']] = submission_w['ID'].str.split('_', expand=True)

# Check for empty values â€‹â€‹in TeamID1 and TeamID2
missing_teamid1 = submission_w['TeamID1'].isna().sum()
missing_teamid2 = submission_w['TeamID2'].isna().sum()

print(f"Number of empty values â€‹â€‹in TeamID1: {missing_teamid1}")
print(f"Number of empty values â€‹â€‹in TeamID2: {missing_teamid2}")

# If there are no empty values
if missing_id == 0 and missing_teamid1 == 0 and missing_teamid2 == 0:
    print("\nThere are no empty values â€‹â€‹in ID, TeamID1, or TeamID2.")
else:
    print("\nThere are blank values â€‹â€‹in some columns, check the data.")


# Load the reference submission file (to maintain the original order)
submission_stage2_path = "/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv"
submission_m_path = "submission_m.csv"
submission_w_path = "submission_w.csv"

submission_stage2 = pd.read_csv(submission_stage2_path)
submission_m = pd.read_csv(submission_m_path)
submission_w = pd.read_csv(submission_w_path)

# Convert predictions into dictionaries for fast lookup
predictions_m = dict(zip(submission_m["ID"], submission_m["Pred"]))
predictions_w = dict(zip(submission_w["ID"], submission_w["Pred"]))

# Update predictions in the original file, keeping the same order
submission_stage2["Pred"] = submission_stage2["ID"].map(lambda x: predictions_m.get(x, predictions_w.get(x, 0.5)))

# Keep only the required columns: 'ID' and 'Pred'
submission_stage2 = submission_stage2[['ID', 'Pred']]

# Save the final merged submission file
submission = "/kaggle/working/submission.csv"
submission_stage2.to_csv(submission, index=False)

print(f"Submission file saved at: {submission}")



# Download the resulting file
submission = pd.read_csv('/kaggle/working/submission.csv')

# Ensure that the ID column exists.
if 'ID' in submission.columns:
    # Split ID column to extract TeamID1 and TeamID2
    submission[['Season', 'TeamID1', 'TeamID2']] = submission['ID'].str.split('_', expand=True)

    # Convert TeamID1 and TeamID2 to integers
    submission['TeamID1'] = pd.to_numeric(submission['TeamID1'], errors='coerce')
    submission['TeamID2'] = pd.to_numeric(submission['TeamID2'], errors='coerce')

    # Find any empty values.
    missing_values = submission[['TeamID1', 'TeamID2']].isna().sum()

    print("Empty values â€‹â€‹in columns:")
    print(missing_values)

    # Check if there are any rows with empty values.
    if missing_values.sum() == 0:
        print("\nThere are no empty values â€‹â€‹in TeamID1 or TeamID2.")
    else:
        print("\nThere are empty values â€‹â€‹in TeamID1 or TeamID2, please check the data.")

else:
    print("Column 'ID' is missing from the file.")


# Read the saved submission file
submission = pd.read_csv("/kaggle/working/submission.csv")

# Check the info of the submission DataFrame
submission.info()



SampleSubmissionStage2 = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv")
SampleSubmissionStage2.info()


submission

