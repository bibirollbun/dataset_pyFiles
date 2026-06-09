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


# import pandas as pd

# def load_datasets(base_path, dataset_files):
#     """
#     Load multiple datasets from CSV files into a dictionary of DataFrames, limiting each to 100 rows and reducing memory usage.

#     Parameters:
#     - base_path (str): The base directory path where the CSV files are located.
#     - dataset_files (dict): A dictionary where keys are dataset names and values are the CSV file names.

#     Returns:
#     - dict: A dictionary where keys are dataset names and values are the loaded DataFrames.
#     """
#     datasets = {}
#     for name, filename in dataset_files.items():
#         file_path = f'{base_path}{filename}'
#         df = pd.read_csv(file_path, nrows=100)  # Limit to 100 rows
#         datasets[name] = optimize_memory_usage(df)
#     return datasets

# def optimize_memory_usage(df):
#     """
#     Optimize memory usage by converting columns to more memory-efficient data types.

#     Parameters:
#     - df (pd.DataFrame): The DataFrame to optimize.

#     Returns:
#     - pd.DataFrame: The optimized DataFrame.
#     """
#     # Convert object columns to category type
#     for col in df.select_dtypes(include=['object']).columns:
#         df[col] = df[col].astype('category')
    
#     # Convert integer columns to smaller integer types if possible
#     for col in df.select_dtypes(include=['int64']).columns:
#         df[col] = pd.to_numeric(df[col], downcast='integer')
    
#     # Convert float columns to smaller float types if possible
#     for col in df.select_dtypes(include=['float64']).columns:
#         df[col] = pd.to_numeric(df[col], downcast='float')
    
#     return df

# # Define the base path
# base_path = '/kaggle/input/march-machine-learning-mania-2025/'

# # Define the dataset files
# dataset_files = {
#     'regular_season_men': 'MRegularSeasonDetailedResults.csv',
#     'tourney_men': 'MNCAATourneyDetailedResults.csv',
#     'regular_season_women': 'WRegularSeasonDetailedResults.csv',
#     'tourney_women': 'WNCAATourneyDetailedResults.csv',
#     'seeds_men': 'MNCAATourneySeeds.csv',
#     'seeds_women': 'WNCAATourneySeeds.csv',
#     'team_conferences_men': 'MTeamConferences.csv',
#     'team_conferences_women': 'WTeamConferences.csv',
#     'massey_ordinal_men': 'MMasseyOrdinals.csv',
#     'team_coaches_men': 'MTeamCoaches.csv'
# }

# # Load the datasets
# datasets = load_datasets(base_path, dataset_files)

# # Access the loaded datasets
# regular_season_men = datasets['regular_season_men']
# tourney_men = datasets['tourney_men']
# regular_season_women = datasets['regular_season_women']
# tourney_women = datasets['tourney_women']
# seeds_men = datasets['seeds_men']
# seeds_women = datasets['seeds_women']
# team_conferences_men = datasets['team_conferences_men']
# team_conferences_women = datasets['team_conferences_women']
# massey_ordinal_men = datasets['massey_ordinal_men']
# team_coaches_men = datasets['team_coaches_men']


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# def show_datasets_grouped_by_function_with_head(data_dir):
#     datasets = {
#         "Cities and Locations": [
#             "Cities.csv",
#             "WGameCities.csv",
#             "MGameCities.csv"
#         ],
#         "Conferences": [
#             "Conferences.csv",
#             "MTeamConferences.csv",
#             "WTeamConferences.csv"
#         ],
#         "Teams": [
#             "MTeams.csv",
#             "WTeams.csv",
#             "MTeamSpellings.csv",
#             "WTeamSpellings.csv"
#         ],
#         "Tournament Seeds and Slots": [
#             "MNCAATourneySeeds.csv",
#             "WNCAATourneySeeds.csv",
#             "MNCAATourneySlots.csv",
#             "WNCAATourneySlots.csv",
#             "MNCAATourneySeedRoundSlots.csv"
#         ],
#         "Tournament Games": [
#             "MConferenceTourneyGames.csv",
#             "WConferenceTourneyGames.csv",
#             "WNCAATourneyCompactResults.csv",
#             "WNCAATourneyDetailedResults.csv",
#             "MNCAATourneyCompactResults.csv",
#             "MNCAATourneyDetailedResults.csv"
#         ],
#         "Regular Season Games": [
#             "MRegularSeasonCompactResults.csv",
#             "MRegularSeasonDetailedResults.csv",
#             "WRegularSeasonCompactResults.csv",
#             "WRegularSeasonDetailedResults.csv"
#         ],
#         "Secondary Tournaments": [
#             "MSecondaryTourneyCompactResults.csv",
#             "MSecondaryTourneyTeams.csv",
#             "WSecondaryTourneyCompactResults.csv",
#             "WSecondaryTourneyTeams.csv"
#         ],
#         "Coaches": [
#             "MTeamCoaches.csv"
#         ],
#         "Seasons": [
#             "MSeasons.csv",
#             "WSeasons.csv"
#         ],
#         "Massey Ordinals": [
#             "MMasseyOrdinals.csv"
#         ],
#         "Sample Submissions and Benchmarks": [
#             "SampleSubmissionStage1.csv",
#             "SeedBenchmarkStage1.csv"
#         ]
#     }

#     for role, datasets_list in datasets.items():
#         print(f"### {role}\n")
#         for dataset in datasets_list:
#             try:
#                 file_path = f"{data_dir}/{dataset}"
#                 df = pd.read_csv(file_path)
#                 print(f"#### {dataset}\n")
#                 print(df.head())
#                 print("\n")

#               # Plot histograms for numeric columns
#                 numeric_columns = df.select_dtypes(include=['number']).columns
#                 if not numeric_columns.empty:
#                     df[numeric_columns].hist(bins=20, figsize=(15, 10))
#                     plt.suptitle(f"Histograms for {dataset}")
#                     plt.show()
#                 else:
#                     print(f"No numeric columns to plot in {dataset}\n")
#             except FileNotFoundError:
#                 print(f"#### {dataset} - File not found\n")
#             except Exception as e:
#                 print(f"#### {dataset} - Error: {e}\n")
#         print("\n")

# # Example usage
# data_directory = '/kaggle/input/march-machine-learning-mania-2025/'
# show_datasets_grouped_by_function_with_head(data_directory)


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# def show_datasets_grouped_by_function_with_head(datasets):
#     dataset_groups = {
#         "Cities and Locations": ['cities', 'w_game_cities', 'm_game_cities'],
#         "Conferences": ['conferences', 'team_conferences_men', 'team_conferences_women'],
#         "Teams": ['m_teams', 'w_teams', 'm_team_spellings', 'w_team_spellings'],
#         "Tournament Seeds and Slots": ['seeds_men', 'seeds_women', 'm_tourney_slots', 'w_tourney_slots', 'm_seed_round_slots'],
#         "Tournament Games": ['m_conf_tourney_games', 'w_conf_tourney_games', 'tourney_women', 'tourney_men'],
#         "Regular Season Games": ['regular_season_men', 'regular_season_women'],
#         "Secondary Tournaments": ['m_secondary_tourney', 'm_secondary_teams', 'w_secondary_tourney', 'w_secondary_teams'],
#         "Coaches": ['team_coaches_men'],
#         "Seasons": ['m_seasons', 'w_seasons'],
#         "Massey Ordinals": ['massey_ordinal_men'],
#         "Sample Submissions and Benchmarks": ['sample_submission', 'seed_benchmark']
#     }

#     category_colors = {
#         "Cities and Locations": "skyblue",
#         "Conferences": "lightgreen",
#         "Teams": "lightcoral",
#         "Tournament Seeds and Slots": "lightyellow",
#         "Tournament Games": "lightpink",
#         "Regular Season Games": "lightsalmon",
#         "Secondary Tournaments": "lightblue",
#         "Coaches": "lightgray",
#         "Seasons": "lightcyan",
#         "Massey Ordinals": "lightgoldenrodyellow",
#         "Sample Submissions and Benchmarks": "lightsteelblue"
#     }

#     gender_colors = {'m': 'blue', 'w': 'red'}

#     for role, dataset_keys in dataset_groups.items():
#         print(f"### {role}\n")
#         for key in dataset_keys:
#             try:
#                 df = datasets[key]
#                 print(f"#### {key}\n")
#                 print(df.head(100))
#                 print("\n")

#                 first_letter = key[0].lower()
#                 plot_color = gender_colors.get(first_letter, category_colors[role])

#                 numeric_columns = df.select_dtypes(include=['number']).columns
#                 if not numeric_columns.empty:
#                     plt.figure(figsize=(15, 10))
#                     for col in numeric_columns:
#                         sns.histplot(df[col], bins=20, color=plot_color, kde=True, label=col)
#                     plt.title(f"Histograms for {key}")
#                     plt.xlabel("Value")
#                     plt.ylabel("Frequency")
#                     plt.legend()
#                     plt.show()
#                 else:
#                     print(f"No numeric columns to plot in {key}\n")
#             except KeyError:
#                 print(f"#### {key} - Dataset not loaded\n")
#             except Exception as e:
#                 print(f"#### {key} - Error: {e}\n")
#         print("\n")

# # Example usage (assuming datasets are already loaded)
# show_datasets_grouped_by_function_with_head(datasets)



# %%time
# import pandas as pd
# import itertools

# # Define data directory
# data_directory = '/kaggle/input/march-machine-learning-mania-2025/'

# # Correct file paths using actual competition filenames
# regular_season_results = pd.read_csv(data_directory + 'MRegularSeasonDetailedResults.csv')  # Correct men's data
# seeds = pd.read_csv(data_directory + 'MNCAATourneySeeds.csv')  # Correct seeds file

# # Ensure data types are consistent
# seeds["TeamID"] = seeds["TeamID"].astype(int)
# regular_season_results["WTeamID"] = regular_season_results["WTeamID"].astype(int)
# regular_season_results["LTeamID"] = regular_season_results["LTeamID"].astype(int)

# # ... rest of your feature engineering code remains the same ...

# # VERIFICATION STEP - USE TOURNAMENT TEAMS ONLY
# # Get valid tournament teams from seeds data
# tournament_teams = seeds.groupby('Season')['TeamID'].apply(set)

# all_combinations = []
# for season in tournament_teams.index:
#     season_teams = list(tournament_teams[season])
#     combinations = list(itertools.combinations(season_teams, 2))
#     all_combinations.extend([(season, t1, t2) for t1, t2 in combinations])

# all_games = pd.DataFrame(all_combinations, columns=['Season', 'Team1', 'Team2'])
# all_games['ID'] = all_games['Season'].astype(str) + '_' + \
#                   all_games['Team1'].astype(str) + '_' + \
#                   all_games['Team2'].astype(str)

# print(f"Generated {len(all_games):,} combinations")
# assert len(all_games) == 131407, "Row count mismatch!"



# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import log_loss

# # Define features and target
# X = features.drop(columns=["Season", "TeamID"])  # Drop non-numeric columns
# y = (features["WinLossRatio"] > 0.5).astype(int)  # Target: Binary classification (Win rate > 50%)

# # Train-Test Split
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Scale the data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)
# ''''
# # Train XGBoost Model
# xgb_model = xgb.XGBClassifier(
#     n_estimators=1000,
#     learning_rate=0.05,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     eval_metric="logloss",
#     use_label_encoder=False
# )
# '''
# # Train XGBoost Model
# model = xgb.XGBClassifier(n_estimators=1000, 
#                           learning_rate=0.05, 
#                           max_depth=5, 
#                           subsample=0.7, 
#                           colsample_bytree=0.8, 
#                           use_label_encoder=False, 
#                           eval_metric="logloss")

# #model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=True)
# model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# # Convert dataset into DMatrix (for better optimization)
# dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
# dval = xgb.DMatrix(X_val_scaled, label=y_val)

# # Train the model with early stopping
# evals = [(dtrain, 'train'), (dval, 'eval')]
# xgb_model = xgb.train(
#     params=model.get_params(),
#     dtrain=dtrain,
#     num_boost_round=1000,
#     evals=evals,
#     early_stopping_rounds=50,
#     verbose_eval=50
# )

# # Evaluate Model
# val_predictions = xgb_model.predict(dval)
# val_logloss = log_loss(y_val, val_predictions)
# print(f"Validation Log Loss: {val_logloss:.5f}")

# print("Model Training Completed!")


# import itertools

# # Load all unique seasons and teams from the ORIGINAL DATA (not features)
# # This ensures we get all possible teams, even those excluded during feature engineering
# all_seasons = regular_season_results['Season'].unique()
# all_teams = pd.concat([
#     regular_season_results['WTeamID'],
#     regular_season_results['LTeamID']
# ]).unique()

# # Generate all possible team combinations
# all_combinations = []
# for season in all_seasons:
#     season_teams = regular_season_results[
#         (regular_season_results['Season'] == season)
#     ][['WTeamID', 'LTeamID']].stack().unique()
#     combinations = list(itertools.combinations(season_teams, 2))
#     all_combinations.extend([(season, t1, t2) for t1, t2 in combinations])

# # Create DataFrame with 132K+ rows
# all_games = pd.DataFrame(all_combinations, columns=['Season', 'Team1', 'Team2'])
# all_games['ID'] = all_games['Season'].astype(str) + '_' + \
#                    all_games['Team1'].astype(str) + '_' + \
#                    all_games['Team2'].astype(str)

# # --- Keep this from your original code ---
# # Merge feature data and predict
# prediction_data = all_games.merge(features, left_on=["Season", "Team1"], right_on=["Season", "TeamID"], how="left")
# prediction_data = prediction_data.merge(features, left_on=["Season", "Team2"], right_on=["Season", "TeamID"], how="left", suffixes=("_1", "_2"))
# prediction_data.drop(columns=["TeamID_1", "TeamID_2"], inplace=True)
# prediction_data = prediction_data.reindex(columns=X.columns, fill_value=0)
# prediction_scaled = scaler.transform(prediction_data)
# all_games["Pred"] = model.predict_proba(prediction_scaled)[:, 1]

# # Merge with sample submission
# final_submission = pd.read_csv(data_directory + "SampleSubmissionStage1.csv")\
#                      .merge(all_games[['ID', 'Pred']], on='ID', how='left')\
#                      .fillna(0.5)

# final_submission.to_csv("submission.csv", index=False)
# print(f"Final submission size: {len(final_submission):,} rows")



# import pandas as pd
# import numpy as np
# import itertools
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import log_loss
# import optuna
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Enable CUDA for XGBoost
# # Ensure you have CUDA installed and a compatible GPU
# params = {
#     'tree_method': 'hist',  # Tree building algorithm
#     'device': 'cuda',       # Use GPU
#     'predictor': 'gpu_predictor',  # Use GPU for prediction (you can keep this)
#     'eval_metric': 'logloss'
# }

# # Load datasets
# data_directory = '/kaggle/input/march-machine-learning-mania-2025/'
# regular_season_results = pd.read_csv(data_directory + 'MRegularSeasonDetailedResults.csv')
# seeds = pd.read_csv(data_directory + 'MNCAATourneySeeds.csv')

# # Ensure data types are consistent
# seeds["TeamID"] = seeds["TeamID"].astype(int)
# regular_season_results["WTeamID"] = regular_season_results["WTeamID"].astype(int)
# regular_season_results["LTeamID"] = regular_season_results["LTeamID"].astype(int)

# # Feature engineering: Calculate team performance metrics
# def calculate_team_performance(results):
#     team_stats = results.groupby(['Season', 'WTeamID']).agg(
#         Wins=('WTeamID', 'size'),
#         AvgScore=('WScore', 'mean'),
#         AvgOpponentScore=('LScore', 'mean')
#     ).reset_index()
#     team_stats.rename(columns={'WTeamID': 'TeamID'}, inplace=True)

#     loss_stats = results.groupby(['Season', 'LTeamID']).agg(
#         Losses=('LTeamID', 'size'),
#         AvgLossScore=('LScore', 'mean'),
#         AvgOpponentWinScore=('WScore', 'mean')
#     ).reset_index()
#     loss_stats.rename(columns={'LTeamID': 'TeamID'}, inplace=True)

#     # Merge win and loss stats
#     team_performance = pd.merge(team_stats, loss_stats, on=['Season', 'TeamID'], how='outer').fillna(0)
#     team_performance['WinLossRatio'] = team_performance['Wins'] / (team_performance['Wins'] + team_performance['Losses'])
#     team_performance['ScoreDiff'] = team_performance['AvgScore'] - team_performance['AvgOpponentScore']
#     return team_performance

# team_performance = calculate_team_performance(regular_season_results)

# # Generate all possible team combinations for the tournament
# def generate_tournament_combinations(seeds):
#     all_combinations = []
#     for season in seeds['Season'].unique():
#         season_teams = seeds[seeds['Season'] == season]['TeamID'].unique()
#         combinations = list(itertools.combinations(season_teams, 2))
#         all_combinations.extend([(season, t1, t2) for t1, t2 in combinations])
#     return pd.DataFrame(all_combinations, columns=['Season', 'Team1', 'Team2'])

# all_games = generate_tournament_combinations(seeds)
# all_games['ID'] = all_games['Season'].astype(str) + '_' + \
#                   all_games['Team1'].astype(str) + '_' + \
#                   all_games['Team2'].astype(str)

# # Merge team performance data for Team1 and Team2
# features = all_games.merge(team_performance, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')
# features = features.merge(team_performance, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left', suffixes=('_1', '_2'))
# features.drop(columns=['TeamID_1', 'TeamID_2'], inplace=True)

# # Define features and target
# X = features[['WinLossRatio_1', 'ScoreDiff_1', 'WinLossRatio_2', 'ScoreDiff_2']]
# y = (features['WinLossRatio_1'] > features['WinLossRatio_2']).astype(int)

# # Train-Test Split
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Scale the data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)

# # Optuna objective function for hyperparameter tuning
# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'gamma': trial.suggest_float('gamma', 0, 1),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'tree_method': 'hist',   # Changed from 'gpu_hist'
#         'device': 'cuda',        # Added this line
#         'predictor': 'gpu_predictor',
#         'eval_metric': 'logloss'
#     }

#     model = xgb.XGBClassifier(**params)
#     model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], early_stopping_rounds=50, verbose=False)
#     val_predictions = model.predict_proba(X_val_scaled)[:, 1]
#     return log_loss(y_val, val_predictions)

# # Run Optuna optimization
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)

# # Best hyperparameters
# best_params = study.best_params
# best_params.update({
#     'tree_method': 'hist',    # Changed from 'gpu_hist'
#     'device': 'cuda',         # Added this line
#     'predictor': 'gpu_predictor',
#     'eval_metric': 'logloss'
# })
# print("Best Hyperparameters:", best_params)

# # Train the final model with the best hyperparameters
# final_model = xgb.XGBClassifier(**best_params)
# final_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], early_stopping_rounds=50, verbose=True)

# # Evaluate the final model
# val_predictions = final_model.predict_proba(X_val_scaled)[:, 1]
# val_logloss = log_loss(y_val, val_predictions)
# print(f"Validation Log Loss: {val_logloss:.5f}")

# # Generate predictions for all tournament matchups
# prediction_data = all_games.merge(team_performance, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')
# prediction_data = prediction_data.merge(team_performance, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left', suffixes=('_1', '_2'))
# prediction_data.drop(columns=['TeamID_1', 'TeamID_2'], inplace=True)
# prediction_data = prediction_data.reindex(columns=X.columns, fill_value=0)
# prediction_scaled = scaler.transform(prediction_data)
# all_games["Pred"] = final_model.predict_proba(prediction_scaled)[:, 1]

# # If the submission is too large, reduce it to exactly 134500 rows
# if len(final_submission) > 134500:
#     # Keep all rows that are in the sample submission
#     sample_ids = pd.read_csv(data_directory + "SampleSubmissionStage1.csv")['ID']
#     must_keep = final_submission[final_submission['ID'].isin(sample_ids)]
    
#     # Randomly sample the rest to reach exactly 134500 rows
#     remaining_rows = 134500 - len(must_keep)
#     if remaining_rows > 0:
#         can_drop = final_submission[~final_submission['ID'].isin(sample_ids)]
#         sampled = can_drop.sample(n=remaining_rows, random_state=42)
#         final_submission = pd.concat([must_keep, sampled])
#     else:
#         final_submission = must_keep.head(134500)

# # Ensure exactly 134500 rows
# final_submission = final_submission.head(134500)

# final_submission.to_csv("submission.csv", index=False)
# print(f"Final submission size: {len(final_submission):,} rows")


# final_submission.head()


# final_submission.tail()


# import pandas as pd
# import numpy as np
# import itertools
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import log_loss
# import optuna
# import warnings

# # Suppress warnings
# warnings.filterwarnings('ignore')

# # Load datasets
# data_directory = '/kaggle/input/march-machine-learning-mania-2025/'
# regular_season_results = pd.read_csv(data_directory + 'MRegularSeasonDetailedResults.csv')
# seeds = pd.read_csv(data_directory + 'MNCAATourneySeeds.csv')

# # Ensure data types are consistent
# seeds["TeamID"] = seeds["TeamID"].astype(int)
# regular_season_results["WTeamID"] = regular_season_results["WTeamID"].astype(int)
# regular_season_results["LTeamID"] = regular_season_results["LTeamID"].astype(int)

# # Feature engineering: Calculate team performance metrics
# def calculate_team_performance(results):
#     team_stats = results.groupby(['Season', 'WTeamID']).agg(
#         Wins=('WTeamID', 'size'),
#         AvgScore=('WScore', 'mean'),
#         AvgOpponentScore=('LScore', 'mean')
#     ).reset_index()
#     team_stats.rename(columns={'WTeamID': 'TeamID'}, inplace=True)

#     loss_stats = results.groupby(['Season', 'LTeamID']).agg(
#         Losses=('LTeamID', 'size'),
#         AvgLossScore=('LScore', 'mean'),
#         AvgOpponentWinScore=('WScore', 'mean')
#     ).reset_index()
#     loss_stats.rename(columns={'LTeamID': 'TeamID'}, inplace=True)

#     # Merge win and loss stats
#     team_performance = pd.merge(team_stats, loss_stats, on=['Season', 'TeamID'], how='outer').fillna(0)
#     team_performance['WinLossRatio'] = team_performance['Wins'] / (team_performance['Wins'] + team_performance['Losses'])
#     team_performance['ScoreDiff'] = team_performance['AvgScore'] - team_performance['AvgOpponentScore']
#     return team_performance

# team_performance = calculate_team_performance(regular_season_results)

# # Generate all possible team combinations for the tournament
# def generate_tournament_combinations(seeds):
#     all_combinations = []
#     for season in seeds['Season'].unique():
#         season_teams = seeds[seeds['Season'] == season]['TeamID'].unique()
#         combinations = list(itertools.combinations(season_teams, 2))
#         all_combinations.extend([(season, t1, t2) for t1, t2 in combinations])
#     return pd.DataFrame(all_combinations, columns=['Season', 'Team1', 'Team2'])

# all_games = generate_tournament_combinations(seeds)
# all_games['ID'] = all_games['Season'].astype(str) + '_' + \
#                   all_games['Team1'].astype(str) + '_' + \
#                   all_games['Team2'].astype(str)

# # Merge team performance data for Team1 and Team2
# features = all_games.merge(team_performance, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')
# features = features.merge(team_performance, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left', suffixes=('_1', '_2'))
# features.drop(columns=['TeamID_1', 'TeamID_2'], inplace=True)

# # Define features and target
# X = features[['WinLossRatio_1', 'ScoreDiff_1', 'WinLossRatio_2', 'ScoreDiff_2']]
# y = (features['WinLossRatio_1'] > features['WinLossRatio_2']).astype(int)

# # Train-Test Split
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Scale the data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)

# # Optuna objective function for hyperparameter tuning
# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'gamma': trial.suggest_float('gamma', 0, 1),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'tree_method': 'hist',  # Use histogram-based method
#         'device': 'cuda',  # Use GPU
#         'eval_metric': 'logloss',
#         'early_stopping_rounds': 50  # Set early stopping here
#     }

#     model = xgb.XGBClassifier(**params)
#     model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
#     val_predictions = model.predict_proba(X_val_scaled)[:, 1]
#     return log_loss(y_val, val_predictions)

# # Run Optuna optimization
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)

# # Best hyperparameters
# best_params = study.best_params
# best_params.update({
#     'tree_method': 'hist',  # Use histogram-based method
#     'device': 'cuda',  # Use GPU
#     'eval_metric': 'logloss',
#     'early_stopping_rounds': 50
# })
# print("Best Hyperparameters:", best_params)

# # Train the final model with the best hyperparameters
# final_model = xgb.XGBClassifier(**best_params)
# final_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=True)

# # Evaluate the final model
# val_predictions = final_model.predict_proba(X_val_scaled)[:, 1]
# val_logloss = log_loss(y_val, val_predictions)
# print(f"Validation Log Loss: {val_logloss:.5f}")

# # Generate predictions for all tournament matchups
# prediction_data = all_games.merge(team_performance, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')
# prediction_data = prediction_data.merge(team_performance, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left', suffixes=('_1', '_2'))
# prediction_data.drop(columns=['TeamID_1', 'TeamID_2'], inplace=True)
# prediction_data = prediction_data.reindex(columns=X.columns, fill_value=0)
# prediction_scaled = scaler.transform(prediction_data)
# all_games["Pred"] = final_model.predict_proba(prediction_scaled)[:, 1]

# # Prepare final submission
# final_submission = pd.read_csv(data_directory + "SampleSubmissionStage1.csv") \
#                      .merge(all_games[['ID', 'Pred']], on='ID', how='left') \
#                      .fillna(0.5)

# final_submission.to_csv("submission.csv", index=False)
# print(f"Final submission size: {len(final_submission):,} rows")


import torch
print(torch.cuda.is_available())  # Should return True if GPU is available


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import optuna
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load datasets
data_directory = '/kaggle/input/march-machine-learning-mania-2025/'
regular_season_results = pd.read_csv(data_directory + 'MRegularSeasonCompactResults.csv')
seeds = pd.read_csv(data_directory + 'MNCAATourneySeeds.csv')

# Ensure data types are consistent
seeds["TeamID"] = seeds["TeamID"].astype(int)
regular_season_results["WTeamID"] = regular_season_results["WTeamID"].astype(int)
regular_season_results["LTeamID"] = regular_season_results["LTeamID"].astype(int)

# Feature engineering: Calculate team performance metrics
def calculate_team_performance(results):
    team_stats = results.groupby(['Season', 'WTeamID']).agg(
        Wins=('WTeamID', 'size'),
        AvgScore=('WScore', 'mean'),
        AvgOpponentScore=('LScore', 'mean')
    ).reset_index()
    team_stats.rename(columns={'WTeamID': 'TeamID'}, inplace=True)

    loss_stats = results.groupby(['Season', 'LTeamID']).agg(
        Losses=('LTeamID', 'size'),
        AvgLossScore=('LScore', 'mean'),
        AvgOpponentWinScore=('WScore', 'mean')
    ).reset_index()
    loss_stats.rename(columns={'LTeamID': 'TeamID'}, inplace=True)

    # Merge win and loss stats
    team_performance = pd.merge(team_stats, loss_stats, on=['Season', 'TeamID'], how='outer').fillna(0)
    team_performance['WinLossRatio'] = team_performance['Wins'] / (team_performance['Wins'] + team_performance['Losses'])
    team_performance['ScoreDiff'] = team_performance['AvgScore'] - team_performance['AvgOpponentScore']
    return team_performance

team_performance = calculate_team_performance(regular_season_results)

# Generate all possible team combinations for the tournament
def generate_tournament_combinations(seeds):
    all_combinations = []
    for season in seeds['Season'].unique():
        season_teams = seeds[seeds['Season'] == season]['TeamID'].unique()
        combinations = list(itertools.combinations(season_teams, 2))
        all_combinations.extend([(season, t1, t2) for t1, t2 in combinations])
    return pd.DataFrame(all_combinations, columns=['Season', 'Team1', 'Team2'])

all_games = generate_tournament_combinations(seeds)
all_games['ID'] = all_games['Season'].astype(str) + '_' + \
                  all_games['Team1'].astype(str) + '_' + \
                  all_games['Team2'].astype(str)

# Merge team performance data for Team1 and Team2
features = all_games.merge(team_performance, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')
features = features.merge(team_performance, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left', suffixes=('_1', '_2'))
features.drop(columns=['TeamID_1', 'TeamID_2'], inplace=True)

# Define features and target
X = features[['WinLossRatio_1', 'ScoreDiff_1', 'WinLossRatio_2', 'ScoreDiff_2']]
y = (features['WinLossRatio_1'] > features['WinLossRatio_2']).astype(int)

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Optuna objective function for hyperparameter tuning
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'tree_method': 'hist',  # Use histogram-based method
        'device': 'cuda',  # Use GPU
        'eval_metric': 'logloss',
        'early_stopping_rounds': 50  # Set early stopping here
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
    val_predictions = model.predict_proba(X_val_scaled)[:, 1]
    return log_loss(y_val, val_predictions)

# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Best hyperparameters
best_params = study.best_params
best_params.update({
    'tree_method': 'hist',  # Use histogram-based method
    'device': 'cuda',  # Use GPU
    'eval_metric': 'logloss',
    'early_stopping_rounds': 50
})
print("Best Hyperparameters:", best_params)

# Train the final model with the best hyperparameters
final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=True)

# Evaluate the final model
val_predictions = final_model.predict_proba(X_val_scaled)[:, 1]
val_logloss = log_loss(y_val, val_predictions)
print(f"Validation Log Loss: {val_logloss:.5f}")

# Generate predictions for all tournament matchups
prediction_data = all_games.merge(team_performance, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')
prediction_data = prediction_data.merge(team_performance, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left', suffixes=('_1', '_2'))
prediction_data.drop(columns=['TeamID_1', 'TeamID_2'], inplace=True)
prediction_data = prediction_data.reindex(columns=X.columns, fill_value=0)
prediction_scaled = scaler.transform(prediction_data)
all_games["Pred"] = final_model.predict_proba(prediction_scaled)[:, 1]

# Prepare final submission
final_submission = pd.read_csv(data_directory + "SampleSubmissionStage1.csv") \
                     .merge(all_games[['ID', 'Pred']], on='ID', how='left') \
                     .fillna(0.5)

final_submission.to_csv("submission.csv", index=False)
print(f"Final submission size: {len(final_submission):,} rows")

