# # import numpy as np
# # import pandas as pd
# # from tqdm import tqdm
# # import os
# # import xgboost as xgb
# # from sklearn.model_selection import KFold
# # from sklearn.metrics import log_loss
# # from scipy.interpolate import UnivariateSpline
# # import statsmodels.api as sm
# # import warnings

# # warnings.filterwarnings('ignore')
# # pd.set_option("display.max_column", 999)

# # class MarchManiaPredictor:
# #     def __init__(self, data_path):
# #         self.data_path = data_path
# #         self.tourney_results = None
# #         self.seeds = None
# #         self.regular_results = None
# #         self.regular_data = None
# #         self.tourney_data = None
# #         self.model = None
# #         self.features = None
# #         self.elo = {}  # Elo ratings for teams

# #     def load_data(self):
# #         """Load and preprocess data."""
# #         def get_pd(dir1):
# #             w = pd.read_csv(f"{self.data_path}/{dir1}")
# #             return w

# #         self.tourney_results = pd.concat([
# #             get_pd("MNCAATourneyDetailedResults.csv"),
# #             get_pd("WNCAATourneyDetailedResults.csv")
# #         ], ignore_index=True)

# #         self.seeds = pd.concat([
# #             get_pd("MNCAATourneySeeds.csv"),
# #             get_pd("WNCAATourneySeeds.csv")
# #         ], ignore_index=True)

# #         self.regular_results = pd.concat([
# #             get_pd("MRegularSeasonDetailedResults.csv"),
# #             get_pd("WRegularSeasonDetailedResults.csv")
# #         ], ignore_index=True)

# #         print("Data loading completed.")

# #     def compute_elo(self, season, team1, team2, margin, k=20):
# #         """Compute Elo ratings for teams."""
# #         if (season, team1) not in self.elo:
# #             self.elo[(season, team1)] = 2000
# #         if (season, team2) not in self.elo:
# #             self.elo[(season, team2)] = 2000

# #         elo1, elo2 = self.elo[(season, team1)], self.elo[(season, team2)]
# #         expected_score = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
# #         actual_score = 1 if margin > 0 else 0
# #         update = k * (actual_score - expected_score)
# #         self.elo[(season, team1)] += update
# #         self.elo[(season, team2)] -= update

# #     def prepare_data(self, df):
# #         """Prepare and transform data for modeling."""
# #         dfswap = df[[
# #             'Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT', 
# #             'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 
# #             'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF'
# #         ]]

# #         dfswap.loc[df['WLoc'] == 'H', 'WLoc'] = 'A'
# #         dfswap.loc[df['WLoc'] == 'A', 'WLoc'] = 'H'
# #         df.columns.values[6] = 'location'
# #         dfswap.columns.values[6] = 'location'

# #         df.columns = [x.replace('W', 'T1_').replace('L', 'T2_') for x in list(df.columns)]
# #         dfswap.columns = [x.replace('L', 'T1_').replace('W', 'T2_') for x in list(dfswap.columns)]

# #         output = pd.concat([df, dfswap]).reset_index(drop=True)
# #         output.loc[output.location == 'N', 'location'] = '0'
# #         output.loc[output.location == 'H', 'location'] = '1'
# #         output.loc[output.location == 'A', 'location'] = '-1'
# #         output.location = output.location.astype(int)

# #         output['PointDiff'] = output['T1_Score'] - output['T2_Score']

# #         # Compute Elo ratings
# #         for idx, row in output.iterrows():
# #             self.compute_elo(row['Season'], row['T1_TeamID'], row['T2_TeamID'], row['PointDiff'])

# #         output['T1_Elo'] = output.apply(lambda row: self.elo[(row['Season'], row['T1_TeamID'])], axis=1)
# #         output['T2_Elo'] = output.apply(lambda row: self.elo[(row['Season'], row['T2_TeamID'])], axis=1)

# #         return output

# #     def feature_engineering(self):
# #         """Perform feature engineering."""
# #         self.regular_data = self.prepare_data(self.regular_results)
# #         self.tourney_data = self.prepare_data(self.tourney_results)

# #         # Basic boxscore stats
# #         boxscore_cols = [
# #             'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_FTM', 'T1_FTA', 'T1_OR', 'T1_DR', 'T1_Ast', 
# #             'T1_TO', 'T1_Stl', 'T1_Blk', 'T1_PF', 'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_FTM', 
# #             'T2_FTA', 'T2_OR', 'T2_DR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk', 'T2_PF', 'PointDiff'
# #         ]

# #         funcs = [np.mean, np.std]  # Add standard deviation for more variability
# #         season_stats = self.regular_data.groupby(["Season", 'T1_TeamID'])[boxscore_cols].agg(funcs).reset_index()
# #         season_stats.columns = [''.join(col).strip() for col in season_stats.columns.values]

# #         # Split for T1 and T2
# #         season_stats_T1 = season_stats.copy()
# #         season_stats_T2 = season_stats.copy()
# #         season_stats_T1.columns = ["T1_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(season_stats_T1.columns)]
# #         season_stats_T2.columns = ["T2_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(season_stats_T2.columns)]
# #         season_stats_T1.columns.values[0] = "Season"
# #         season_stats_T2.columns.values[0] = "Season"

# #         # Merge stats
# #         self.tourney_data = pd.merge(self.tourney_data, season_stats_T1, on=['Season', 'T1_TeamID'], how='left')
# #         self.tourney_data = pd.merge(self.tourney_data, season_stats_T2, on=['Season', 'T2_TeamID'], how='left')

# #         # Add win ratio for last 14 days
# #         last14days_stats_T1 = self.regular_data.loc[self.regular_data.DayNum > 118].reset_index(drop=True)
# #         last14days_stats_T1['win'] = np.where(last14days_stats_T1['PointDiff'] > 0, 1, 0)
# #         last14days_stats_T1 = last14days_stats_T1.groupby(['Season', 'T1_TeamID'])['win'].mean().reset_index(name='T1_win_ratio_14d')

# #         last14days_stats_T2 = self.regular_data.loc[self.regular_data.DayNum > 118].reset_index(drop=True)
# #         last14days_stats_T2['win'] = np.where(last14days_stats_T2['PointDiff'] < 0, 1, 0)
# #         last14days_stats_T2 = last14days_stats_T2.groupby(['Season', 'T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio_14d')

# #         self.tourney_data = pd.merge(self.tourney_data, last14days_stats_T1, on=['Season', 'T1_TeamID'], how='left')
# #         self.tourney_data = pd.merge(self.tourney_data, last14days_stats_T2, on=['Season', 'T2_TeamID'], how='left')

# #         # Add seed information
# #         self.seeds['Seed'] = self.seeds['Seed'].apply(lambda x: int(x[1:3]) if len(x) > 2 else int(x[1:]))
# #         seeds_T1 = self.seeds.rename(columns={'TeamID': 'T1_TeamID', 'Seed': 'T1_Seed'})
# #         seeds_T2 = self.seeds.rename(columns={'TeamID': 'T2_TeamID', 'Seed': 'T2_Seed'})
# #         self.tourney_data = pd.merge(self.tourney_data, seeds_T1, on=['Season', 'T1_TeamID'], how='left')
# #         self.tourney_data = pd.merge(self.tourney_data, seeds_T2, on=['Season', 'T2_TeamID'], how='left')

# #         # Define features
# #         self.features = list(season_stats_T1.columns[2:]) + \
# #                         list(season_stats_T2.columns[2:]) + \
# #                         ["T1_win_ratio_14d", "T2_win_ratio_14d", "T1_Elo", "T2_Elo", "T1_Seed", "T2_Seed"]

# #         # Fill missing values
# #         self.tourney_data[self.features] = self.tourney_data[self.features].fillna(self.tourney_data[self.features].mean())

# #     def train_model(self):
# #         """Train the XGBoost model."""
# #         # Use binary classification (win/loss) instead of point differential
# #         y = (self.tourney_data['T1_Score'] > self.tourney_data['T2_Score']).astype(int)
# #         X = self.tourney_data[self.features].values
# #         dtrain = xgb.DMatrix(X, label=y)

# #         param = {
# #             'objective': 'binary:logistic',  # Predict win probability
# #             'eval_metric': 'logloss',       # Optimize for log loss
# #             'eta': 0.01,                    # Lower learning rate for better convergence
# #             'subsample': 0.8,               # Increase subsample
# #             'colsample_bytree': 0.8,        # Increase feature sampling
# #             'max_depth': 5,                 # Slightly deeper trees
# #             'min_child_weight': 10,         # Reduce overfitting
# #             'gamma': 1,                     # Regularization
# #             'seed': 42
# #         }

# #         xgb_cv = []
# #         repeat_cv = 5  # Increase CV repeats for stability
# #         for i in range(repeat_cv):
# #             xgb_cv.append(
# #                 xgb.cv(
# #                     params=param,
# #                     dtrain=dtrain,
# #                     num_boost_round=5000,
# #                     folds=KFold(n_splits=10, shuffle=True, random_state=i),  # 10-fold CV
# #                     early_stopping_rounds=50,
# #                     verbose_eval=100
# #                 )
# #             )

# #         iteration_counts = [np.argmin(x['test-logloss-mean'].values) for x in xgb_cv]
# #         val_logloss = [np.min(x['test-logloss-mean'].values) for x in xgb_cv]
# #         print(f"Iteration counts: {iteration_counts}, Validation Log Loss: {val_logloss}")

# #         # Train final model
# #         self.model = xgb.train(params=param, dtrain=dtrain, num_boost_round=int(np.mean(iteration_counts)))

# #     def generate_submission(self, season=2024):
# #         """Generate submission file."""
# #         if self.model is None:
# #             raise ValueError("Model has not been trained. Call `train_model` first.")

# #         # Prepare submission data
# #         submission_file = f"{self.data_path}/SampleSubmissionStage1.csv"
# #         if os.path.exists(submission_file):
# #             submission = pd.read_csv(submission_file)
# #         else:
# #             print(f"Warning: {submission_file} not found. Creating a new submission template.")
# #             submission = pd.DataFrame({
# #                 'ID': [f"{season}_{t1}_{t2}" for t1, t2 in zip(self.tourney_data['T1_TeamID'], self.tourney_data['T2_TeamID'])],
# #                 'Pred': [0.5] * len(self.tourney_data)
# #             })

# #         # Merge features
# #         submission['Season'] = submission['ID'].apply(lambda x: int(x.split('_')[0]))
# #         submission['T1_TeamID'] = submission['ID'].apply(lambda x: int(x.split('_')[1]))
# #         submission['T2_TeamID'] = submission['ID'].apply(lambda x: int(x.split('_')[2]))

# #         submission = pd.merge(submission, self.tourney_data, on=['Season', 'T1_TeamID', 'T2_TeamID'], how='left')

# #         # Predict probabilities
# #         X_sub = submission[self.features].values
# #         dtest = xgb.DMatrix(X_sub)
# #         submission['Pred'] = self.model.predict(dtest)

# #         # Clip predictions to [0.05, 0.95]
# #         submission['Pred'] = submission['Pred'].clip(0.05, 0.95)

# #         # Save submission file
# #         submission[['ID', 'Pred']].to_csv('submission.csv', index=False)
# #         print("Submission file created successfully.")

# #     def run_all(self):
# #         """Run the entire pipeline."""
# #         self.load_data()
# #         self.feature_engineering()
# #         self.train_model()
# #         self.generate_submission()
# import pandas as pd
# from tqdm import tqdm
# import os
# import xgboost as xgb
# from sklearn.model_selection import KFold
# from sklearn.metrics import log_loss
# import warnings

# warnings.filterwarnings('ignore')
# pd.set_option("display.max_columns", None)

# class MarchManiaPredictor:
#     def __init__(self, data_path):
#         self.data_path = data_path
#         self.tourney_results = None
#         self.seeds = None
#         self.regular_results = None
#         self.regular_data = None
#         self.tourney_data = None
#         self.model = None
#         self.features = []
#         self.elo = {}  # Elo ratings for teams

#     def load_data(self):
#         def get_pd(filename):
#             return pd.read_csv(f"{self.data_path}/{filename}")

#         self.tourney_results = pd.concat([
#             get_pd("MNCAATourneyDetailedResults.csv"),
#             get_pd("WNCAATourneyDetailedResults.csv")
#         ], ignore_index=True)

#         self.seeds = pd.concat([
#             get_pd("MNCAATourneySeeds.csv"),
#             get_pd("WNCAATourneySeeds.csv")
#         ], ignore_index=True)

#         self.regular_results = pd.concat([
#             get_pd("MRegularSeasonDetailedResults.csv"),
#             get_pd("WRegularSeasonDetailedResults.csv")
#         ], ignore_index=True)
        
#         print("Data loading completed.")

#     def compute_elo(self, season, team1, team2, margin, k=20):
#         if (season, team1) not in self.elo:
#             self.elo[(season, team1)] = 2000
#         if (season, team2) not in self.elo:
#             self.elo[(season, team2)] = 2000

#         elo1, elo2 = self.elo[(season, team1)], self.elo[(season, team2)]
#         expected_score = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
#         actual_score = 1 if margin > 0 else 0
#         update = k * (actual_score - expected_score)
#         self.elo[(season, team1)] += update
#         self.elo[(season, team2)] -= update

#     def prepare_data(self, df):
#         """Prepare and transform data for modeling."""
#         df['PointDiff'] = df['WScore'] - df['LScore']
#         for idx, row in df.iterrows():
#             self.compute_elo(row['Season'], row['WTeamID'], row['LTeamID'], row['PointDiff'])
        
#         df['W_Elo'] = df.apply(lambda row: self.elo[(row['Season'], row['WTeamID'])], axis=1)
#         df['L_Elo'] = df.apply(lambda row: self.elo[(row['Season'], row['LTeamID'])], axis=1)

#         return df

#     def feature_engineering(self):
#         """Perform feature engineering."""
#         self.regular_data = self.prepare_data(self.regular_results)
#         self.tourney_data = self.prepare_data(self.tourney_results)

#         # Load and merge location data
#         game_cities = pd.concat([
#             pd.read_csv(f"{self.data_path}/MGameCities.csv"),
#             pd.read_csv(f"{self.data_path}/WGameCities.csv")
#         ])
#         cities = pd.read_csv(f"{self.data_path}/Cities.csv")
#         game_cities = game_cities.merge(cities, on="CityID")
        
#         self.tourney_data = self.tourney_data.merge(
#             game_cities[['Season', 'DayNum', 'WTeamID', 'LTeamID', 'CityID', 'State']],
#             left_on=['Season', 'DayNum', 'WTeamID', 'LTeamID'],
#             right_on=['Season', 'DayNum', 'WTeamID', 'LTeamID'],
#             how='left'
#         )

#         # Add home state feature
#         state_wins = self.regular_data.groupby(['Season', 'WTeamID'])['State'].agg(lambda x: x.mode()[0] if not x.empty else 'Unknown').reset_index()
#         state_wins.columns = ['Season', 'WTeamID', 'HomeState']
#         self.tourney_data = self.tourney_data.merge(state_wins, left_on=['Season', 'WTeamID'], right_on=['Season', 'WTeamID'], how='left')
        
#         self.features.extend(['CityID', 'HomeState'])
        
#         # Fill missing values
#         self.tourney_data['HomeState'] = self.tourney_data['HomeState'].fillna('Unknown')
#         self.tourney_data['CityID'] = self.tourney_data['CityID'].fillna(-1)

#     def train_model(self):
#         """Train the XGBoost model."""
#         y = (self.tourney_data['WScore'] > self.tourney_data['LScore']).astype(int)
#         X = self.tourney_data[self.features].values
#         dtrain = xgb.DMatrix(X, label=y)

#         param = {
#             'objective': 'binary:logistic',
#             'eval_metric': 'logloss',
#             'eta': 0.01,
#             'subsample': 0.8,
#             'colsample_bytree': 0.8,
#             'max_depth': 5,
#             'min_child_weight': 10,
#             'gamma': 1,
#             'seed': 42
#         }
        
#         self.model = xgb.train(params=param, dtrain=dtrain, num_boost_round=500)

#     def generate_submission(self, season=2025):
#         """Generate submission file."""
#         if self.model is None:
#             raise ValueError("Model has not been trained. Call `train_model` first.")

#         submission_file = f"{self.data_path}/SampleSubmissionStage1.csv"
#         submission = pd.read_csv(submission_file)

#         submission[['Season', 'T1_TeamID', 'T2_TeamID']] = submission['ID'].str.split('_', expand=True).astype(int)
#         submission = submission.merge(self.tourney_data, on=['Season', 'T1_TeamID', 'T2_TeamID'], how='left')
        
#         X_sub = submission[self.features].values
#         dtest = xgb.DMatrix(X_sub)
#         submission['Pred'] = self.model.predict(dtest)
#         submission['Pred'] = submission['Pred'].clip(0.05, 0.95)

#         submission[['ID', 'Pred']].to_csv('submission.csv', index=False)
#         print("Submission file created successfully.")

#     def run_all(self):
#         """Run the entire pipeline."""
#         self.load_data()
#         self.feature_engineering()
#         self.train_model()
#         self.generate_submission()



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# import os
# print(os.listdir('/kaggle/input/march-machine-learning-mania-2025/'))



# import numpy as np
# import pandas as pd
# import xgboost as xgb
# import plotly.express as px
# import os
# from sklearn.model_selection import KFold
# from sklearn.metrics import log_loss

# class Percent(float):
#     def __str__(self):
#         return '{:.1%}'.format(self)

# def get_heatMap(output_df, rounds_list, title):
#     output_df = output_df.sort_values(by=output_df.columns.tolist()[-1], ascending=False)
#     players_list = output_df.index.tolist()
#     probabilities_npMatrix = [[Percent(e) for e in l] for l in output_df.values.tolist()]
#     probabilitiesText_npMatrix = np.array(probabilities_npMatrix, dtype='str_')

#     fig = px.imshow(probabilities_npMatrix, x=rounds_list, y=players_list, color_continuous_scale='Greens', aspect="auto", width=800, height=1600)
#     fig.update_traces(text=probabilitiesText_npMatrix, texttemplate="%{text}")
#     fig.update_xaxes(side="top")
#     fig.update(layout_coloraxis_showscale=False)
#     fig.update_layout(title=title)
#     fig.show(renderer='iframe')

# def compute_elo_ratings(regular_results, k=20):
#     team_ratings = {}
#     for season in regular_results['Season'].unique():
#         team_ratings[season] = {}
#         all_teams = set(regular_results['WTeamID'].unique()) | set(regular_results['LTeamID'].unique())
#         for team_id in all_teams:
#             team_ratings[season][team_id] = 1500
        
#         season_data = regular_results[regular_results['Season'] == season].sort_values('DayNum')
#         for _, row in season_data.iterrows():
#             team1, team2 = row['WTeamID'], row['LTeamID']
#             rating1, rating2 = team_ratings[season][team1], team_ratings[season][team2]
#             expected_score1 = 1 / (1 + 10**((rating2 - rating1) / 400))
#             team_ratings[season][team1] += k * (1 - expected_score1)
#             team_ratings[season][team2] += k * (0 - expected_score1)
#     return team_ratings

# def get_team_game_counts(regular_results):
#     wins = regular_results.groupby(['Season', 'WTeamID']).size().reset_index(name='W_Games').rename(columns={'WTeamID': 'TeamID'})
#     losses = regular_results.groupby(['Season', 'LTeamID']).size().reset_index(name='L_Games').rename(columns={'LTeamID': 'TeamID'})
#     game_counts = wins.merge(losses, on=['Season', 'TeamID'], how='outer')
#     game_counts['Total_Games'] = game_counts['W_Games'].fillna(0) + game_counts['L_Games'].fillna(0)
#     game_counts = game_counts[['Season', 'TeamID', 'Total_Games']]
#     return game_counts

# def get_team_avg_score_diff(regular_results):
#     team_score_diff = {}
#     all_teams = set(regular_results['WTeamID'].unique()) | set(regular_results['LTeamID'].unique())
#     for season in regular_results['Season'].unique():
#         season_data = regular_results[regular_results['Season'] == season]
#         team_score_diff[season] = {}
#         for team_id in all_teams:
#             wins = season_data[season_data['WTeamID'] == team_id]
#             losses = season_data[season_data['LTeamID'] == team_id]
#             total_diff = sum(wins['WScore'] - wins['LScore']) + sum(losses['LScore'] - losses['WScore'])
#             total_games = len(wins) + len(losses)
#             team_score_diff[season][team_id] = total_diff / total_games if total_games > 0 else 0
#     return team_score_diff

# class MarchManiaPredictor:
#     def __init__(self, data_path, fivethirtyeight_path):
#         self.data_path = data_path
#         self.fivethirtyeight_path = fivethirtyeight_path
#         self.features = []
#         self.model = None
    
#     def load_data(self):
#         self.regular_results = pd.concat([
#             pd.read_csv(f"{self.data_path}/MRegularSeasonCompactResults.csv"),
#             pd.read_csv(f"{self.data_path}/WRegularSeasonCompactResults.csv")
#         ], ignore_index=True)
        
#         self.tourney_results = pd.concat([
#             pd.read_csv(f"{self.data_path}/MNCAATourneyCompactResults.csv"),
#             pd.read_csv(f"{self.data_path}/WNCAATourneyCompactResults.csv")
#         ], ignore_index=True)
        
#         self.seeds = pd.concat([
#             pd.read_csv(f"{self.data_path}/MNCAATourneySeeds.csv"),
#             pd.read_csv(f"{self.data_path}/WNCAATourneySeeds.csv")
#         ], ignore_index=True)
        
#         self.massey_ordinals = pd.read_csv(f"{self.data_path}/MMasseyOrdinals.csv")
#         self.cities = pd.read_csv(f"{self.data_path}/Cities.csv")
#         self.game_cities = pd.concat([
#             pd.read_csv(f"{self.data_path}/MGameCities.csv"),
#             pd.read_csv(f"{self.data_path}/WGameCities.csv")
#         ], ignore_index=True)

#         self.mens_probabilities = pd.read_csv(f"{self.fivethirtyeight_path}/mensProbabilitiesTable.csv", index_col='player')
#         self.mens_probabilities = self.mens_probabilities.drop('Elo_Rating', axis=1)
#         self.womens_probabilities = pd.read_csv(f"{self.fivethirtyeight_path}/womensProbabilitiesTable.csv", index_col='player')
#         self.womens_probabilities = self.womens_probabilities.drop('Elo_Rating', axis=1)

#     def prepare_data(self, df):
#         # Use .loc to avoid SettingWithCopyWarning
#         df = df.copy()  # Create a copy to avoid modifying the original
#         df.loc[:, 'T1_TeamID'] = np.minimum(df['WTeamID'], df['LTeamID'])
#         df.loc[:, 'T2_TeamID'] = np.maximum(df['WTeamID'], df['LTeamID'])
#         df.loc[:, 'T1_Win'] = (df['WTeamID'] == df['T1_TeamID']).astype(int)
#         return df[['Season', 'DayNum', 'T1_TeamID', 'T2_TeamID', 'T1_Win']]

#     def feature_engineering(self):
#         self.elo_ratings = compute_elo_ratings(self.regular_results)
#         self.game_counts = get_team_game_counts(self.regular_results)
#         self.avg_score_diff = get_team_avg_score_diff(self.regular_results)
        
#         self.tourney_data = self.prepare_data(self.tourney_results)
#         self.tourney_data['T1_Elo'] = self.tourney_data.apply(lambda row: self.elo_ratings[row['Season']][row['T1_TeamID']], axis=1)
#         self.tourney_data['T2_Elo'] = self.tourney_data.apply(lambda row: self.elo_ratings[row['Season']][row['T2_TeamID']], axis=1)
#         self.tourney_data['Elo_Diff'] = self.tourney_data['T1_Elo'] - self.tourney_data['T2_Elo']
        
#         self.seeds['Seed'] = self.seeds['Seed'].apply(lambda x: int(x[1:3]) if len(x) > 2 else int(x[1:]))
#         self.tourney_data = pd.merge(self.tourney_data, self.seeds, left_on=['Season', 'T1_TeamID'], right_on=['Season', 'TeamID'], how='left')
#         self.tourney_data = self.tourney_data.drop('TeamID', axis=1).rename(columns={'Seed': 'T1_Seed'})
#         self.tourney_data = pd.merge(self.tourney_data, self.seeds, left_on=['Season', 'T2_TeamID'], right_on=['Season', 'TeamID'], how='left')
#         self.tourney_data = self.tourney_data.drop('TeamID', axis=1).rename(columns={'Seed': 'T2_Seed'})
#         self.tourney_data['Seed_Diff'] = self.tourney_data['T1_Seed'].fillna(16) - self.tourney_data['T2_Seed'].fillna(16)
        
#         self.tourney_data = pd.merge(self.tourney_data, self.game_counts, left_on=['Season', 'T1_TeamID'], right_on=['Season', 'TeamID'], how='left')
#         self.tourney_data = self.tourney_data.drop('TeamID', axis=1).rename(columns={'Total_Games': 'T1_Games'})
#         self.tourney_data = pd.merge(self.tourney_data, self.game_counts, left_on=['Season', 'T2_TeamID'], right_on=['Season', 'TeamID'], how='left')
#         self.tourney_data = self.tourney_data.drop('TeamID', axis=1).rename(columns={'Total_Games': 'T2_Games'})
#         self.tourney_data['T1_Games'] = self.tourney_data['T1_Games'].fillna(0)
#         self.tourney_data['T2_Games'] = self.tourney_data['T2_Games'].fillna(0)
        
#         self.tourney_data['T1_Avg_Score_Diff'] = self.tourney_data.apply(lambda row: self.avg_score_diff[row['Season']][row['T1_TeamID']], axis=1)
#         self.tourney_data['T2_Avg_Score_Diff'] = self.tourney_data.apply(lambda row: self.avg_score_diff[row['Season']][row['T2_TeamID']], axis=1)
        
#         massey_latest = self.massey_ordinals[self.massey_ordinals['RankingDayNum'] == 133].groupby(['Season', 'TeamID'])['OrdinalRank'].mean().reset_index()
#         self.tourney_data = pd.merge(self.tourney_data, massey_latest, left_on=['Season', 'T1_TeamID'], right_on=['Season', 'TeamID'], how='left')
#         self.tourney_data = self.tourney_data.drop('TeamID', axis=1).rename(columns={'OrdinalRank': 'T1_Massey_Rank'})
#         self.tourney_data = pd.merge(self.tourney_data, massey_latest, left_on=['Season', 'T2_TeamID'], right_on=['Season', 'TeamID'], how='left')
#         self.tourney_data = self.tourney_data.drop('TeamID', axis=1).rename(columns={'OrdinalRank': 'T2_Massey_Rank'})
#         self.tourney_data['Massey_Rank_Diff'] = self.tourney_data['T1_Massey_Rank'].fillna(351) - self.tourney_data['T2_Massey_Rank'].fillna(351)
        
#         self.tourney_data = pd.merge(self.tourney_data, self.game_cities, left_on=['Season', 'DayNum', 'T1_TeamID', 'T2_TeamID'], 
#                                      right_on=['Season', 'DayNum', 'WTeamID', 'LTeamID'], how='left')
#         self.tourney_data = pd.merge(self.tourney_data, self.cities[['CityID', 'State']], on='CityID', how='left')
#         self.tourney_data['State'] = self.tourney_data['State'].fillna('Unknown')
        
#         self.features = ['Elo_Diff', 'Seed_Diff', 'T1_Games', 'T2_Games', 'T1_Avg_Score_Diff', 'T2_Avg_Score_Diff', 'Massey_Rank_Diff']
    
#     def train_model(self):
#         X = self.tourney_data[self.features]
#         y = self.tourney_data['T1_Win']
        
#         kfold = KFold(n_splits=10, shuffle=True, random_state=42)
#         log_losses = []
#         for train_idx, val_idx in kfold.split(X):
#             X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#             y_train, y_val = y[train_idx], y[val_idx]
            
#             model = xgb.XGBClassifier(
#                 use_label_encoder=False,
#                 eval_metric='logloss',
#                 n_estimators=1000,
#                 learning_rate=0.01,
#                 max_depth=6,
#                 subsample=0.8,
#                 colsample_bytree=0.8,
#                 min_child_weight=1
#             )
#             model.fit(X_train, y_train)
            
#             preds = model.predict_proba(X_val)[:, 1]
#             log_loss_val = log_loss(y_val, preds)
#             log_losses.append(log_loss_val)
        
#         print("Average Log Loss:", np.mean(log_losses))
        
#         self.model = xgb.XGBClassifier(
#             use_label_encoder=False,
#             eval_metric='logloss',
#             n_estimators=1000,
#             learning_rate=0.01,
#             max_depth=6,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             min_child_weight=1
#         )
#         self.model.fit(X, y)
    
#     def generate_submission(self, season=2025):
#         if self.model is None:
#             raise ValueError("Model has not been trained. Call `train_model` first.")
        
#         # Load the correct sample submission expecting 131,407 rows
#         submission_file = f"{self.data_path}/SampleSubmissionStage2.csv"  # Switch to Stage2
#         if os.path.exists(submission_file):
#             submission_df = pd.read_csv(submission_file)
#         else:
#             raise FileNotFoundError(f"Sample submission file not found at {submission_file}")
        
#         # Parse IDs to extract team IDs
#         submission_df['Season'] = submission_df['ID'].apply(lambda x: int(x.split('_')[0]))
#         submission_df['T1_TeamID'] = submission_df['ID'].apply(lambda x: int(x.split('_')[1]))
#         submission_df['T2_TeamID'] = submission_df['ID'].apply(lambda x: int(x.split('_')[2]))
        
#         # Ensure exactly 131,407 rows
#         if len(submission_df) != 131407:
#             raise ValueError(f"Sample submission has {len(submission_df)} rows, expected 131,407")
        
#         # Get features for 2025 using latest season (2024)
#         latest_season = max(self.regular_results['Season'].unique())
#         latest_elo = self.elo_ratings[latest_season]
#         latest_game_counts = self.game_counts[self.game_counts['Season'] == latest_season]
#         latest_avg_score_diff = self.avg_score_diff[latest_season]
#         latest_seeds = self.seeds[self.seeds['Season'] == latest_season].set_index('TeamID')['Seed'].to_dict()
#         latest_massey = self.massey_ordinals[(self.massey_ordinals['Season'] == latest_season) & 
#                                              (self.massey_ordinals['RankingDayNum'] == 133)].set_index('TeamID')['OrdinalRank'].to_dict()
        
#         def get_rating(team_id, ratings, default):
#             return ratings.get(team_id, default)
        
#         submission_df['T1_Elo'] = submission_df['T1_TeamID'].apply(lambda x: get_rating(x, latest_elo, 1500))
#         submission_df['T2_Elo'] = submission_df['T2_TeamID'].apply(lambda x: get_rating(x, latest_elo, 1500))
#         submission_df['Elo_Diff'] = submission_df['T1_Elo'] - submission_df['T2_Elo']
        
#         submission_df['T1_Seed'] = submission_df['T1_TeamID'].apply(lambda x: get_rating(x, latest_seeds, 16))
#         submission_df['T2_Seed'] = submission_df['T2_TeamID'].apply(lambda x: get_rating(x, latest_seeds, 16))
#         submission_df['Seed_Diff'] = submission_df['T1_Seed'] - submission_df['T2_Seed']
        
#         submission_df = pd.merge(submission_df, latest_game_counts, left_on='T1_TeamID', right_on='TeamID', how='left')
#         submission_df = submission_df.drop('TeamID', axis=1).rename(columns={'Total_Games': 'T1_Games'})
#         submission_df = pd.merge(submission_df, latest_game_counts, left_on='T2_TeamID', right_on='TeamID', how='left')
#         submission_df = submission_df.drop('TeamID', axis=1).rename(columns={'Total_Games': 'T2_Games'})
#         submission_df['T1_Games'] = submission_df['T1_Games'].fillna(0)
#         submission_df['T2_Games'] = submission_df['T2_Games'].fillna(0)
        
#         submission_df['T1_Avg_Score_Diff'] = submission_df['T1_TeamID'].apply(lambda x: get_rating(x, latest_avg_score_diff, 0))
#         submission_df['T2_Avg_Score_Diff'] = submission_df['T2_TeamID'].apply(lambda x: get_rating(x, latest_avg_score_diff, 0))
        
#         submission_df['T1_Massey_Rank'] = submission_df['T1_TeamID'].apply(lambda x: get_rating(x, latest_massey, 351))
#         submission_df['T2_Massey_Rank'] = submission_df['T2_TeamID'].apply(lambda x: get_rating(x, latest_massey, 351))
#         submission_df['Massey_Rank_Diff'] = submission_df['T1_Massey_Rank'] - submission_df['T2_Massey_Rank']
        
#         X_sub = submission_df[self.features].values
#         submission_df['Pred'] = self.model.predict_proba(X_sub)[:, 1]
#         submission_df['Pred'] = submission_df['Pred'].clip(0.05, 0.95)
        
#         submission_df[['ID', 'Pred']].to_csv('submission.csv', index=False)
#         print(f"Submission file created successfully with {len(submission_df)} rows.")
    
#     def run_all(self):
#         self.load_data()
#         self.feature_engineering()
#         self.train_model()
#         self.generate_submission()
        
#         get_heatMap(self.mens_probabilities, rounds_list=['Reach R2', 'Reach S16', 'Reach E8', 'Reach F4', 'Reach CG', 'Champion'], title='Mens March Madness 2024')
#         get_heatMap(self.womens_probabilities, rounds_list=['Reach R2', 'Reach S16', 'Reach E8', 'Reach F4', 'Reach CG', 'Champion'], title='Womens March Madness 2024')


import numpy as np
import pandas as pd
import xgboost as xgb
import plotly.express as px
import os
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import optuna  # Added for hyperparameter tuning

class Percent(float):
    def __str__(self):
        return '{:.1%}'.format(self)

def get_heatMap(output_df, rounds_list, title):
    output_df = output_df.sort_values(by=output_df.columns.tolist()[-1], ascending=False)
    players_list = output_df.index.tolist()
    probabilities_npMatrix = [[Percent(e) for e in l] for l in output_df.values.tolist()]
    probabilitiesText_npMatrix = np.array(probabilities_npMatrix, dtype='str_')

    fig = px.imshow(probabilities_npMatrix, x=rounds_list, y=players_list, color_continuous_scale='Greens', aspect="auto", width=800, height=1600)
    fig.update_traces(text=probabilitiesText_npMatrix, texttemplate="%{text}")
    fig.update_xaxes(side="top")
    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(title=title)
    fig.show(renderer='iframe')

def compute_elo_ratings(regular_results, k=20):
    team_ratings = {}
    for season in regular_results['Season'].unique():
        team_ratings[season] = {}
        all_teams = set(regular_results['WTeamID'].unique()) | set(regular_results['LTeamID'].unique())
        for team_id in all_teams:
            team_ratings[season][team_id] = 1500
        
        season_data = regular_results[regular_results['Season'] == season].sort_values('DayNum')
        for _, row in season_data.iterrows():
            team1, team2 = row['WTeamID'], row['LTeamID']
            rating1, rating2 = team_ratings[season][team1], team_ratings[season][team2]
            expected_score1 = 1 / (1 + 10**((rating2 - rating1) / 400))
            team_ratings[season][team1] += k * (1 - expected_score1)
            team_ratings[season][team2] += k * (0 - expected_score1)
    return team_ratings

def get_team_game_counts(regular_results):
    wins = regular_results.groupby(['Season', 'WTeamID']).size().reset_index(name='W_Games').rename(columns={'WTeamID': 'TeamID'})
    losses = regular_results.groupby(['Season', 'LTeamID']).size().reset_index(name='L_Games').rename(columns={'LTeamID': 'TeamID'})
    game_counts = wins.merge(losses, on=['Season', 'TeamID'], how='outer')
    game_counts['Total_Games'] = game_counts['W_Games'].fillna(0) + game_counts['L_Games'].fillna(0)
    game_counts = game_counts[['Season', 'TeamID', 'Total_Games']]
    return game_counts

def get_team_avg_score_diff(regular_results):
    team_score_diff = {}
    all_teams = set(regular_results['WTeamID'].unique()) | set(regular_results['LTeamID'].unique())
    for season in regular_results['Season'].unique():
        season_data = regular_results[regular_results['Season'] == season]
        team_score_diff[season] = {}
        for team_id in all_teams:
            wins = season_data[season_data['WTeamID'] == team_id]
            losses = season_data[season_data['LTeamID'] == team_id]
            total_diff = sum(wins['WScore'] - wins['LScore']) + sum(losses['LScore'] - losses['WScore'])
            total_games = len(wins) + len(losses)
            team_score_diff[season][team_id] = total_diff / total_games if total_games > 0 else 0
    return team_score_diff

class MarchManiaPredictor:
    def __init__(self, data_path, fivethirtyeight_path):
        self.data_path = data_path
        self.fivethirtyeight_path = fivethirtyeight_path
        self.features = []
        self.model = None
    
    def load_data(self):
        self.regular_results = pd.concat([
            pd.read_csv(f"{self.data_path}/MRegularSeasonCompactResults.csv"),
            pd.read_csv(f"{self.data_path}/WRegularSeasonCompactResults.csv")
        ], ignore_index=True)
        
        self.tourney_results = pd.concat([
            pd.read_csv(f"{self.data_path}/MNCAATourneyCompactResults.csv"),
            pd.read_csv(f"{self.data_path}/WNCAATourneyCompactResults.csv")
        ], ignore_index=True)
        
        self.seeds = pd.concat([
            pd.read_csv(f"{self.data_path}/MNCAATourneySeeds.csv"),
            pd.read_csv(f"{self.data_path}/WNCAATourneySeeds.csv")
        ], ignore_index=True)
        
        self.massey_ordinals = pd.read_csv(f"{self.data_path}/MMasseyOrdinals.csv")
        self.cities = pd.read_csv(f"{self.data_path}/Cities.csv")
        self.game_cities = pd.concat([
            pd.read_csv(f"{self.data_path}/MGameCities.csv"),
            pd.read_csv(f"{self.data_path}/WGameCities.csv")
        ], ignore_index=True)

        self.mens_probabilities = pd.read_csv(f"{self.fivethirtyeight_path}/mensProbabilitiesTable.csv", index_col='player')
        self.mens_probabilities = self.mens_probabilities.drop('Elo_Rating', axis=1)
        self.womens_probabilities = pd.read_csv(f"{self.fivethirtyeight_path}/womensProbabilitiesTable.csv", index_col='player')
        self.womens_probabilities = self.womens_probabilities.drop('Elo_Rating', axis=1)

    def prepare_data(self, df):
        df = df.copy()
        df.loc[:, 'T1_TeamID'] = np.minimum(df['WTeamID'], df['LTeamID'])
        df.loc[:, 'T2_TeamID'] = np.maximum(df['WTeamID'], df['LTeamID'])
        df.loc[:, 'T1_Win'] = (df['WTeamID'] == df['T1_TeamID']).astype(int)
        return df[['Season', 'DayNum', 'T1_TeamID', 'T2_TeamID', 'T1_Win']]

    def feature_engineering(self):
        self.elo_ratings = compute_elo_ratings(self.regular_results)
        self.game_counts = get_team_game_counts(self.regular_results)
        self.avg_score_diff = get_team_avg_score_diff(self.regular_results)
        
        self.tourney_data = self.prepare_data(self.tourney_results)
        self.tourney_data['T1_Elo'] = self.tourney_data.apply(lambda row: self.elo_ratings[row['Season']][row['T1_TeamID']], axis=1)
        self.tourney_data['T2_Elo'] = self.tourney_data.apply(lambda row: self.elo_ratings[row['Season']][row['T2_TeamID']], axis=1)
        self.tourney_data['Elo_Diff'] = self.tourney_data['T1_Elo'] - self.tourney_data['T2_Elo']
        
        self.seeds['Seed'] = self.seeds['Seed'].apply(lambda x: int(x[1:3]) if len(x) > 2 else int(x[1:]))
        self.tourney_data = pd.merge(self.tourney_data, self.seeds, left_on=['Season', 'T1_TeamID'], right_on=['Season', 'TeamID'], how='left')
        self.tourney_data = self.tourney_data.drop('TeamID', axis=1).rename(columns={'Seed': 'T1_Seed'})
        self.tourney_data = pd.merge(self.tourney_data, self.seeds, left_on=['Season', 'T2_TeamID'], right_on=['Season', 'TeamID'], how='left')
        self.tourney_data = self.tourney_data.drop('TeamID', axis=1).rename(columns={'Seed': 'T2_Seed'})
        self.tourney_data['Seed_Diff'] = self.tourney_data['T1_Seed'].fillna(16) - self.tourney_data['T2_Seed'].fillna(16)
        
        self.tourney_data = pd.merge(self.tourney_data, self.game_counts, left_on=['Season', 'T1_TeamID'], right_on=['Season', 'TeamID'], how='left')
        self.tourney_data = self.tourney_data.drop('TeamID', axis=1).rename(columns={'Total_Games': 'T1_Games'})
        self.tourney_data = pd.merge(self.tourney_data, self.game_counts, left_on=['Season', 'T2_TeamID'], right_on=['Season', 'TeamID'], how='left')
        self.tourney_data = self.tourney_data.drop('TeamID', axis=1).rename(columns={'Total_Games': 'T2_Games'})
        self.tourney_data['T1_Games'] = self.tourney_data['T1_Games'].fillna(0)
        self.tourney_data['T2_Games'] = self.tourney_data['T2_Games'].fillna(0)
        
        self.tourney_data['T1_Avg_Score_Diff'] = self.tourney_data.apply(lambda row: self.avg_score_diff[row['Season']][row['T1_TeamID']], axis=1)
        self.tourney_data['T2_Avg_Score_Diff'] = self.tourney_data.apply(lambda row: self.avg_score_diff[row['Season']][row['T2_TeamID']], axis=1)
        
        massey_latest = self.massey_ordinals[self.massey_ordinals['RankingDayNum'] == 133].groupby(['Season', 'TeamID'])['OrdinalRank'].mean().reset_index()
        self.tourney_data = pd.merge(self.tourney_data, massey_latest, left_on=['Season', 'T1_TeamID'], right_on=['Season', 'TeamID'], how='left')
        self.tourney_data = self.tourney_data.drop('TeamID', axis=1).rename(columns={'OrdinalRank': 'T1_Massey_Rank'})
        self.tourney_data = pd.merge(self.tourney_data, massey_latest, left_on=['Season', 'T2_TeamID'], right_on=['Season', 'TeamID'], how='left')
        self.tourney_data = self.tourney_data.drop('TeamID', axis=1).rename(columns={'OrdinalRank': 'T2_Massey_Rank'})
        self.tourney_data['Massey_Rank_Diff'] = self.tourney_data['T1_Massey_Rank'].fillna(351) - self.tourney_data['T2_Massey_Rank'].fillna(351)
        
        self.tourney_data = pd.merge(self.tourney_data, self.game_cities, left_on=['Season', 'DayNum', 'T1_TeamID', 'T2_TeamID'], 
                                     right_on=['Season', 'DayNum', 'WTeamID', 'LTeamID'], how='left')
        self.tourney_data = pd.merge(self.tourney_data, self.cities[['CityID', 'State']], on='CityID', how='left')
        self.tourney_data['State'] = self.tourney_data['State'].fillna('Unknown')
        
        self.features = ['Elo_Diff', 'Seed_Diff', 'T1_Games', 'T2_Games', 'T1_Avg_Score_Diff', 'T2_Avg_Score_Diff', 'Massey_Rank_Diff']
    
    def tune_hyperparameters(self):
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            
            X = self.tourney_data[self.features]
            y = self.tourney_data['T1_Win']
            
            kfold = KFold(n_splits=10, shuffle=True, random_state=42)
            log_losses = []
            for train_idx, val_idx in kfold.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train)
                preds = model.predict_proba(X_val)[:, 1]
                log_loss_val = log_loss(y_val, preds)
                log_losses.append(log_loss_val)
            
            return np.mean(log_losses)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        print("Best hyperparameters found:", best_params)
        return best_params

    def train_model(self):
        best_params = self.tune_hyperparameters()
        X = self.tourney_data[self.features]
        y = self.tourney_data['T1_Win']
        
        self.model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X, y)
        print("Final model trained with optimized hyperparameters.")

    def generate_submission(self, season=2025):
        if self.model is None:
            raise ValueError("Model has not been trained. Call `train_model` first.")
        
        submission_file = f"{self.data_path}/SampleSubmissionStage2.csv"
        if os.path.exists(submission_file):
            submission_df = pd.read_csv(submission_file)
        else:
            raise FileNotFoundError(f"Sample submission file not found at {submission_file}")
        
        submission_df['Season'] = submission_df['ID'].apply(lambda x: int(x.split('_')[0]))
        submission_df['T1_TeamID'] = submission_df['ID'].apply(lambda x: int(x.split('_')[1]))
        submission_df['T2_TeamID'] = submission_df['ID'].apply(lambda x: int(x.split('_')[2]))
        
        if len(submission_df) != 131407:
            raise ValueError(f"Sample submission has {len(submission_df)} rows, expected approximately 130,000 (specifically 131,407)")
        
        latest_season = max(self.regular_results['Season'].unique())
        latest_elo = self.elo_ratings[latest_season]
        latest_game_counts = self.game_counts[self.game_counts['Season'] == latest_season]
        latest_avg_score_diff = self.avg_score_diff[latest_season]
        latest_seeds = self.seeds[self.seeds['Season'] == latest_season].set_index('TeamID')['Seed'].to_dict()
        latest_massey = self.massey_ordinals[(self.massey_ordinals['Season'] == latest_season) & 
                                             (self.massey_ordinals['RankingDayNum'] == 133)].set_index('TeamID')['OrdinalRank'].to_dict()
        
        def get_rating(team_id, ratings, default):
            return ratings.get(team_id, default)
        
        submission_df['T1_Elo'] = submission_df['T1_TeamID'].apply(lambda x: get_rating(x, latest_elo, 1500))
        submission_df['T2_Elo'] = submission_df['T2_TeamID'].apply(lambda x: get_rating(x, latest_elo, 1500))
        submission_df['Elo_Diff'] = submission_df['T1_Elo'] - submission_df['T2_Elo']
        
        submission_df['T1_Seed'] = submission_df['T1_TeamID'].apply(lambda x: get_rating(x, latest_seeds, 16))
        submission_df['T2_Seed'] = submission_df['T2_TeamID'].apply(lambda x: get_rating(x, latest_seeds, 16))
        submission_df['Seed_Diff'] = submission_df['T1_Seed'] - submission_df['T2_Seed']
        
        submission_df = pd.merge(submission_df, latest_game_counts, left_on='T1_TeamID', right_on='TeamID', how='left')
        submission_df = submission_df.drop('TeamID', axis=1).rename(columns={'Total_Games': 'T1_Games'})
        submission_df = pd.merge(submission_df, latest_game_counts, left_on='T2_TeamID', right_on='TeamID', how='left')
        submission_df = submission_df.drop('TeamID', axis=1).rename(columns={'Total_Games': 'T2_Games'})
        submission_df['T1_Games'] = submission_df['T1_Games'].fillna(0)
        submission_df['T2_Games'] = submission_df['T2_Games'].fillna(0)
        
        submission_df['T1_Avg_Score_Diff'] = submission_df['T1_TeamID'].apply(lambda x: get_rating(x, latest_avg_score_diff, 0))
        submission_df['T2_Avg_Score_Diff'] = submission_df['T2_TeamID'].apply(lambda x: get_rating(x, latest_avg_score_diff, 0))
        
        submission_df['T1_Massey_Rank'] = submission_df['T1_TeamID'].apply(lambda x: get_rating(x, latest_massey, 351))
        submission_df['T2_Massey_Rank'] = submission_df['T2_TeamID'].apply(lambda x: get_rating(x, latest_massey, 351))
        submission_df['Massey_Rank_Diff'] = submission_df['T1_Massey_Rank'] - submission_df['T2_Massey_Rank']
        
        X_sub = submission_df[self.features].values
        submission_df['Pred'] = self.model.predict_proba(X_sub)[:, 1]
        submission_df['Pred'] = submission_df['Pred'].clip(0.05, 0.95)
        
        submission_df[['ID', 'Pred']].to_csv('submission.csv', index=False)
        print(f"Submission file created successfully with {len(submission_df)} rows.")
    
    def run_all(self):
        self.load_data()
        self.feature_engineering()
        self.train_model()
        self.generate_submission()
        
        get_heatMap(self.mens_probabilities, rounds_list=['Reach R2', 'Reach S16', 'Reach E8', 'Reach F4', 'Reach CG', 'Champion'], title='Mens March Madness 2024')
        get_heatMap(self.womens_probabilities, rounds_list=['Reach R2', 'Reach S16', 'Reach E8', 'Reach F4', 'Reach CG', 'Champion'], title='Womens March Madness 2024')


if __name__ == "__main__":
    data_path = "/kaggle/input/march-machine-learning-mania-2025"
    fivethirtyeight_path = "/kaggle/input/538data"
    predictor = MarchManiaPredictor(data_path, fivethirtyeight_path)
    predictor.run_all()


import pandas as pd

df = pd.read_csv('/kaggle/working/submission.csv')
print(df.head())  # Display first few rows
print(df.tail())
print(df.shape)




