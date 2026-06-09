import os
import joblib
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

mens_files = []
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        if filename[0] == "M":
            mens_files.append(os.path.join(dirname, filename))


mens_files.remove('/kaggle/input/march-machine-learning-mania-2025/MTeamSpellings.csv')
mens_files


# Path where all CSV files are stored
dir_path = "/kaggle/input/march-machine-learning-mania-2025/"

# Get a list of all CSV files
csv_files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]

# Dictionary to store column names for each file
files_and_cols = {}

# Loop through each file, read it, and store column names
for file in csv_files:
    file_path = os.path.join(dir_path, file)
    df = pd.read_csv(file_path, nrows=5)  # Read only the first few rows for efficiency
    files_and_cols[file] = df.columns.values
    del df  # Free memory

# Print extracted column names
for file, cols in files_and_cols.items():
    print(f"'{file}': {list(cols)}")


# Define file paths based on your Kaggle directory
dir_path = "/kaggle/input/march-machine-learning-mania-2025/"

file_paths = {
    "regular_season_detailed": os.path.join(dir_path, "MRegularSeasonDetailedResults.csv"),
    "teams": os.path.join(dir_path, "MTeams.csv"),
    "team_conferences": os.path.join(dir_path, "MTeamConferences.csv"),
    "team_rankings": os.path.join(dir_path, "MMasseyOrdinals.csv"),
    "tournament_seeds": os.path.join(dir_path, "MNCAATourneySeeds.csv"),
    "tournament_results": os.path.join(dir_path, "MNCAATourneyCompactResults.csv"),
    "game_cities": os.path.join(dir_path, "MGameCities.csv"),
}

# Load datasets
df_regular_season = pd.read_csv(file_paths["regular_season_detailed"])
df_teams = pd.read_csv(file_paths["teams"])
df_team_conferences = pd.read_csv(file_paths["team_conferences"])
df_team_rankings = pd.read_csv(file_paths["team_rankings"])
df_tournament_seeds = pd.read_csv(file_paths["tournament_seeds"])
df_tournament_results = pd.read_csv(file_paths["tournament_results"])
df_game_cities = pd.read_csv(file_paths["game_cities"])

# Merge team names into regular season results
df_regular_season = df_regular_season.merge(df_teams, left_on="WTeamID", right_on="TeamID", how= "left").rename(columns={"Teamname": "WTeamName"})
df_regular_season = df_regular_season.merge(df_teams, left_on="LTeamID", right_on="TeamID", how="left").rename(columns={"TeamName": "LTeamName"})
df_regular_season.drop(columns=["TeamID_x", "TeamID_y"], inplace=True)  # Remove redundant columns

# Merge team conferences
df_regular_season = df_regular_season.merge(df_team_conferences, left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"], how="left").rename(columns={"ConfAbbrev": "WConf"})
df_regular_season = df_regular_season.merge(df_team_conferences, left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"], how="left").rename(columns={"ConfAbbrev": "LConf"})
df_regular_season.drop(columns=["TeamID_x", "TeamID_y"], inplace=True)

# Merge rankings (Using median rank for each season/team to avoid multiple entries per day)
df_team_rankings = df_team_rankings.groupby(["Season", "TeamID"])["OrdinalRank"].median().reset_index()
df_regular_season = df_regular_season.merge(df_team_rankings, left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"], how="left").rename(columns={"OrdinalRank": "WTeamRank"})
df_regular_season = df_regular_season.merge(df_team_rankings, left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"], how="left").rename(columns={"OrdinalRank": "LTeamRank"})
df_regular_season.drop(columns=["TeamID_x", "TeamID_y"], inplace=True)

# Merge tournament seedings
df_regular_season = df_regular_season.merge(df_tournament_seeds, left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"], how="left").rename(columns={"Seed": "WSeed"})
df_regular_season = df_regular_season.merge(df_tournament_seeds, left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"], how="left").rename(columns={"Seed": "LSeed"})
df_regular_season.drop(columns=["TeamID_x", "TeamID_y"], inplace=True)

# Merge tournament results
df_regular_season = df_regular_season.merge(df_tournament_results, on=["Season", "WTeamID", "LTeamID", "DayNum"], how="left", suffixes=("", "_tourney"))

# Merge game cities
df_regular_season = df_regular_season.merge(df_game_cities, on=["Season", "DayNum", "WTeamID", "LTeamID"], how="left")

# Save the merged dataset
output_file = "/kaggle/working/merged_dataset.csv"
df_regular_season.to_csv(output_file, index=False)

print(f"✅ Merged dataset saved to: {output_file}")
df_regular_season.info()  # Display summary of merged dataset



drop_cols = ["WScore_tourney","LScore_tourney","NumOT_tourney","CityID","WScore", "CRType", "WSeed", "LSeed"]
X = df_regular_season.drop(drop_cols, axis =1)
y = df_regular_season[["WScore"]]

for colname in X.select_dtypes("object").columns:
    X[colname], _ = X[colname].factorize()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


X.head(10)


y.head(10)


discrete_features = X.dtypes == int
def make_mi_scores(X, y, discrete_features):
 mi_scores = mutual_info_regression(X, y.values, discrete_features=discrete_features)
 mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
 mi_scores = mi_scores.sort_values(ascending=False)
 return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores


def plot_mi_scores(scores):
 scores = scores.sort_values(ascending=True)
 width = np.arange(len(scores))
 ticks = list(scores.index)
 plt.barh(width, scores)
 plt.yticks(width, ticks)
 plt.title("Mutual Information Scores")

plt.figure(dpi=100, figsize=(12, 8))
plot_mi_scores(mi_scores)


# Model Initialisation
forest = RandomForestRegressor()

param_grid = {
    'n_estimators': [50,100],
    'max_depth': [10, 20],
    'min_samples_split': [5, 10]
}

# Grid Search initialisation
grid_search = GridSearchCV(
    estimator=forest,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)


print("Best parameters found:\n", grid_search.best_params_, end="\n")
print("Best cross-validation accuracy:\n", grid_search.best_score_)

# Reinitialising the best estimator from grid search
best_estimator = grid_search.best_estimator_

# Testing
test_accuracy = best_estimator.score(X_test, y_test)
y_pred = best_estimator.predict(X_test)

print("Test set accuracy: ", test_accuracy)
print("Test MAE: ", mean_absolute_error(y_test, y_pred))
print("Test RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

# Saving the trained estimator
if joblib.dump(best_estimator, "baseline_estimator.pkl"):
    print("The model was saved")


sns.jointplot(
    x=df_regular_season["WFGM"], y=y.iloc[:, 0]
)


sns.jointplot(
    x=df_regular_season["LScore"], y=y.iloc[:, 0]
)


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Define base path
# base_path = "/kaggle/input/march-machine-learning-mania-2025/"

# # Load core datasets
# regular_season = pd.read_csv(f'{base_path}MRegularSeasonDetailedResults.csv')
# tourney_results = pd.read_csv(f'{base_path}MNCAATourneyDetailedResults.csv')
# teams = pd.read_csv(f'{base_path}MTeams.csv')
# team_conferences = pd.read_csv(f'{base_path}MTeamConferences.csv')
# tourney_seeds = pd.read_csv(f'{base_path}MNCAATourneySeeds.csv')
# massey_rankings = pd.read_csv(f'{base_path}MMasseyOrdinals.csv')

# # Print basic info about datasets
# print(f"Regular season games: {len(regular_season)}")
# print(f"Tournament games: {len(tourney_results)}")
# print(f"Teams: {len(teams)}")
# print(f"Unique seasons in regular season data: {regular_season['Season'].nunique()}")


# def calculate_team_season_stats(games_df, season):
#     """Calculate team statistics for a given season."""
#     season_games = games_df[games_df['Season'] == season]
    
#     # Initialize dictionary to store team stats
#     team_stats = {}
    
#     # Get unique team IDs for this season
#     team_ids = set(season_games['WTeamID'].unique()) | set(season_games['LTeamID'].unique())
    
#     for team_id in team_ids:
#         # Games where team won
#         w_games = season_games[season_games['WTeamID'] == team_id]
#         # Games where team lost
#         l_games = season_games[season_games['LTeamID'] == team_id]
        
#         # Calculate win-loss record
#         wins = len(w_games)
#         losses = len(l_games)
#         win_percentage = wins / (wins + losses) if (wins + losses) > 0 else 0
        
#         # Calculate offensive stats
#         points_scored = w_games['WScore'].sum() + l_games['LScore'].sum()
#         games_played = wins + losses
#         ppg = points_scored / games_played if games_played > 0 else 0
        
#         # Calculate defensive stats
#         points_allowed = w_games['LScore'].sum() + l_games['WScore'].sum()
#         papg = points_allowed / games_played if games_played > 0 else 0
        
#         # Calculate shooting percentages
#         fg_made = w_games['WFGM'].sum() + l_games['LFGM'].sum()
#         fg_attempts = w_games['WFGA'].sum() + l_games['LFGA'].sum()
#         fg_pct = fg_made / fg_attempts if fg_attempts > 0 else 0
        
#         fg3_made = w_games['WFGM3'].sum() + l_games['LFGM3'].sum()
#         fg3_attempts = w_games['WFGA3'].sum() + l_games['LFGA3'].sum()
#         fg3_pct = fg3_made / fg3_attempts if fg3_attempts > 0 else 0
        
#         ft_made = w_games['WFTM'].sum() + l_games['LFTM'].sum()
#         ft_attempts = w_games['WFTA'].sum() + l_games['LFTA'].sum()
#         ft_pct = ft_made / ft_attempts if ft_attempts > 0 else 0
        
#         # Calculate rebounding
#         off_rebounds = w_games['WOR'].sum() + l_games['LOR'].sum()
#         def_rebounds = w_games['WDR'].sum() + l_games['LDR'].sum()
#         total_rebounds = off_rebounds + def_rebounds
#         rebounds_per_game = total_rebounds / games_played if games_played > 0 else 0
        
#         # Calculate assist and turnover stats
#         assists = w_games['WAst'].sum() + l_games['LAst'].sum()
#         turnovers = w_games['WTO'].sum() + l_games['LTO'].sum()
#         assist_to_turnover = assists / turnovers if turnovers > 0 else 0
        
#         # Calculate possession-based stats
#         w_possessions = w_games['WFGA'].sum() - w_games['WOR'].sum() + w_games['WTO'].sum() + 0.475 * w_games['WFTA'].sum()
#         l_possessions = l_games['LFGA'].sum() - l_games['LOR'].sum() + l_games['LTO'].sum() + 0.475 * l_games['LFTA'].sum()
#         total_possessions = w_possessions + l_possessions
        
#         # Offensive efficiency (points per 100 possessions)
#         w_points = w_games['WScore'].sum()
#         l_points = l_games['LScore'].sum()
#         offensive_efficiency = 100 * (w_points + l_points) / total_possessions if total_possessions > 0 else 0
        
#         # Defensive efficiency (points allowed per 100 possessions)
#         w_points_allowed = w_games['LScore'].sum()
#         l_points_allowed = l_games['WScore'].sum()
#         defensive_efficiency = 100 * (w_points_allowed + l_points_allowed) / total_possessions if total_possessions > 0 else 0
        
#         # Net efficiency
#         net_efficiency = offensive_efficiency - defensive_efficiency
        
#         # Store stats in dictionary
#         team_stats[team_id] = {
#             'Season': season,
#             'TeamID': team_id,
#             'Wins': wins,
#             'Losses': losses,
#             'WinPct': win_percentage,
#             'PPG': ppg,
#             'PAPG': papg,
#             'FGPct': fg_pct,
#             'FG3Pct': fg3_pct,
#             'FTPct': ft_pct,
#             'ReboundsPerGame': rebounds_per_game,
#             'AssistToTurnover': assist_to_turnover,
#             'OffensiveEfficiency': offensive_efficiency,
#             'DefensiveEfficiency': defensive_efficiency,
#             'NetEfficiency': net_efficiency
#         }
    
#     # Convert to DataFrame
#     return pd.DataFrame.from_dict(team_stats, orient='index')

# # Calculate team stats for each season
# seasons = sorted(regular_season['Season'].unique())
# all_team_stats = pd.DataFrame()

# for season in seasons:
#     season_stats = calculate_team_season_stats(regular_season, season)
#     all_team_stats = pd.concat([all_team_stats, season_stats])

# # Reset index
# all_team_stats = all_team_stats.reset_index(drop=True)

# # Check the stats
# print(all_team_stats.head())




