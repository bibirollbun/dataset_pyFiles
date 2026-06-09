# import glob
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import log_loss, brier_score_loss
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.calibration import CalibratedClassifierCV
# import warnings

# warnings.filterwarnings("ignore")

# def compute_elo(rankings, winner, loser, k=32):
#     expected_win_w = 1 / (1 + 10 ** ((rankings[loser] - rankings[winner]) / 400))
#     expected_win_l = 1 - expected_win_w
#     rankings[winner] += k * (1 - expected_win_w)
#     rankings[loser] += k * (0 - expected_win_l)
#     return rankings

# class TournamentPredictor:
#     def __init__(self, data_path):
#         self.data_path = data_path
#         self.data = None
#         self.teams = None
#         self.seeds = None
#         self.games = None
#         self.elo = {}
#         self.model = CalibratedClassifierCV(RandomForestClassifier(
#             n_estimators=500,  
#             max_depth=30, 
#             min_samples_split=3, 
#             max_features='sqrt',
#             random_state=42
#         ))

#     def load_data(self):
#         files = glob.glob(self.data_path)
#         self.data = {p.split('/')[-1].split('.')[0]: pd.read_csv(p, encoding='latin-1') for p in files}
#         tourney_cresults = pd.concat([self.data['MNCAATourneyCompactResults'], self.data['WNCAATourneyCompactResults']])
#         self.games = tourney_cresults[tourney_cresults['Season'] < 2025]

#         seeds_df = pd.concat([self.data['MNCAATourneySeeds'], self.data['WNCAATourneySeeds']])
#         self.seeds = seeds_df.set_index(['Season', 'TeamID'])['Seed'].str.extract(r'(\d+)').astype(float).to_dict()[0]

#         self.elo = {team: 1500 for team in set(self.games['WTeamID']).union(set(self.games['LTeamID']))}
#         for _, row in self.games.iterrows():
#             self.elo = compute_elo(self.elo, row['WTeamID'], row['LTeamID'])
        
#         self.games['Elo_W'] = self.games['WTeamID'].map(self.elo)
#         self.games['Elo_L'] = self.games['LTeamID'].map(self.elo)
#         self.games['EloDiff'] = self.games['Elo_W'] - self.games['Elo_L']

#         self.games['SeedDiff'] = self.games.apply(lambda row: self.seeds.get((row['Season'], row['WTeamID']), 16) - self.seeds.get((row['Season'], row['LTeamID']), 16), axis=1)
#         self.games['Pred'] = 1
#         games_losing = self.games.copy()
#         games_losing.rename(columns={'WTeamID': 'LTeamID', 'LTeamID': 'WTeamID'}, inplace=True)
#         games_losing['Pred'] = 0
#         self.games = pd.concat([self.games, games_losing], ignore_index=True)
#         print("Data loading and preprocessing completed.")

#     def train_model(self):
#         X = self.games[['SeedDiff', 'EloDiff']]
#         y = self.games['Pred']
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
#         if len(set(y_train)) > 1:
#             self.model.fit(X_train, y_train)
#             pred_cal = self.model.predict_proba(X_test)[:, 1]
#             print(f'Log Loss: {log_loss(y_test, pred_cal):.4f}')
#             print(f'Brier Score: {brier_score_loss(y_test, pred_cal):.4f}')
#         else:
#             print("Training data has only one class. Adjust stratification.")

    # def generate_submission(self, season=2025):
    #     matchups = [(row['Season'], row['WTeamID'], row['LTeamID']) for _, row in self.games.iterrows() if row['Season'] == season]
    #     submission = []
    #     for season, wteam, lteam in matchups:
    #         seed_diff = self.seeds.get((season, wteam), 16) - self.seeds.get((season, lteam), 16)
    #         elo_diff = self.elo.get(wteam, 1500) - self.elo.get(lteam, 1500)
    #         pred = self.model.predict_proba([[seed_diff, elo_diff]])[:, 1][0]
    #         pred = max(0.05, min(0.95, pred))
    #         submission.append([f"{season}_{wteam}_{lteam}", pred])
    #     pd.DataFrame(submission, columns=['ID', 'Pred']).to_csv('submission.csv', index=False)
    #     print("Submission file created successfully.")

#     def run_all(self):
#         self.load_data()
#         self.train_model()
#         self.generate_submission()



import os

data_path = "/kaggle/input/march-machine-learning-mania-2025"
print("Files in dataset:")
print(os.listdir(data_path))



import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from scipy.interpolate import UnivariateSpline
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')
pd.set_option("display.max_column", 999)

class MarchManiaPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.tourney_results = None
        self.seeds = None
        self.regular_results = None
        self.regular_data = None
        self.tourney_data = None
        self.model = None
        self.features = None
        self.elo = {}  # Initialize elo attribute

    def load_data(self):
        """Load and preprocess data."""
        def get_pd(dir1):
            w = pd.read_csv(f"{self.data_path}/{dir1}")
            return w

        self.tourney_results = pd.concat([
            get_pd("MNCAATourneyDetailedResults.csv"),
            get_pd("WNCAATourneyDetailedResults.csv")
        ], ignore_index=True)

        self.seeds = pd.concat([
            get_pd("MNCAATourneySeeds.csv"),
            get_pd("WNCAATourneySeeds.csv")
        ], ignore_index=True)

        self.regular_results = pd.concat([
            get_pd("MRegularSeasonDetailedResults.csv"),
            get_pd("WRegularSeasonDetailedResults.csv")
        ], ignore_index=True)

        print("Data loading completed.")

    def prepare_data(self, df):
        """Prepare and transform data for modeling."""
        dfswap = df[[
            'Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT', 
            'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 
            'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF'
        ]]

        dfswap.loc[df['WLoc'] == 'H', 'WLoc'] = 'A'
        dfswap.loc[df['WLoc'] == 'A', 'WLoc'] = 'H'
        df.columns.values[6] = 'location'
        dfswap.columns.values[6] = 'location'

        df.columns = [x.replace('W', 'T1_').replace('L', 'T2_') for x in list(df.columns)]
        dfswap.columns = [x.replace('L', 'T1_').replace('W', 'T2_') for x in list(dfswap.columns)]

        output = pd.concat([df, dfswap]).reset_index(drop=True)
        output.loc[output.location == 'N', 'location'] = '0'
        output.loc[output.location == 'H', 'location'] = '1'
        output.loc[output.location == 'A', 'location'] = '-1'
        output.location = output.location.astype(int)

        output['PointDiff'] = output['T1_Score'] - output['T2_Score']
        return output

    def feature_engineering(self):
        """Perform feature engineering."""
        self.regular_data = self.prepare_data(self.regular_results)
        self.tourney_data = self.prepare_data(self.tourney_results)

        boxscore_cols = [
            'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_OR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_PF', 
            'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_OR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk',  
            'PointDiff'
        ]

        funcs = [np.mean]
        season_statistics = self.regular_data.groupby(["Season", 'T1_TeamID'])[boxscore_cols].agg(funcs).reset_index()
        season_statistics.columns = [''.join(col).strip() for col in season_statistics.columns.values]

        season_statistics_T1 = season_statistics.copy()
        season_statistics_T2 = season_statistics.copy()

        season_statistics_T1.columns = ["T1_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(season_statistics_T1.columns)]
        season_statistics_T2.columns = ["T2_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(season_statistics_T2.columns)]
        season_statistics_T1.columns.values[0] = "Season"
        season_statistics_T2.columns.values[0] = "Season"

        self.tourney_data = pd.merge(self.tourney_data, season_statistics_T1, on=['Season', 'T1_TeamID'], how='left')
        self.tourney_data = pd.merge(self.tourney_data, season_statistics_T2, on=['Season', 'T2_TeamID'], how='left')

        last14days_stats_T1 = self.regular_data.loc[self.regular_data.DayNum > 118].reset_index(drop=True)
        last14days_stats_T1['win'] = np.where(last14days_stats_T1['PointDiff'] > 0, 1, 0)
        last14days_stats_T1 = last14days_stats_T1.groupby(['Season', 'T1_TeamID'])['win'].mean().reset_index(name='T1_win_ratio_14d')

        last14days_stats_T2 = self.regular_data.loc[self.regular_data.DayNum > 118].reset_index(drop=True)
        last14days_stats_T2['win'] = np.where(last14days_stats_T2['PointDiff'] < 0, 1, 0)
        last14days_stats_T2 = last14days_stats_T2.groupby(['Season', 'T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio_14d')

        self.tourney_data = pd.merge(self.tourney_data, last14days_stats_T1, on=['Season', 'T1_TeamID'], how='left')
        self.tourney_data = pd.merge(self.tourney_data, last14days_stats_T2, on=['Season', 'T2_TeamID'], how='left')

        self.features = list(season_statistics_T1.columns[2:]) + \
                        list(season_statistics_T2.columns[2:]) + \
                        ["T1_win_ratio_14d", "T2_win_ratio_14d"]

    def train_model(self):
        """Train the XGBoost model."""
        y = self.tourney_data['T1_Score'] - self.tourney_data['T2_Score']
        X = self.tourney_data[self.features].values
        dtrain = xgb.DMatrix(X, label=y)

        param = {
            'eval_metric': 'mae',
            'booster': 'gbtree',
            'eta': 0.05,
            'subsample': 0.35,
            'colsample_bytree': 0.7,
            'num_parallel_tree': 3,
            'min_child_weight': 40,
            'gamma': 10,
            'max_depth': 3,
            'silent': 1
        }

        xgb_cv = []
        repeat_cv = 3
        for i in range(repeat_cv):
            xgb_cv.append(
                xgb.cv(
                    params=param,
                    dtrain=dtrain,
                    num_boost_round=3000,
                    folds=KFold(n_splits=5, shuffle=True, random_state=i),
                    early_stopping_rounds=25,
                    verbose_eval=50
                )
            )

        iteration_counts = [np.argmin(x['test-mae-mean'].values) for x in xgb_cv]
        val_mae = [np.min(x['test-mae-mean'].values) for x in xgb_cv]
        print(f"Iteration counts: {iteration_counts}, Validation MAE: {val_mae}")

        # Train final model
        self.model = xgb.train(params=param, dtrain=dtrain, num_boost_round=int(np.mean(iteration_counts)))

    def generate_submission(self, season=2024):
        """Generate submission file."""
        if self.model is None:
            raise ValueError("Model has not been trained. Call `train_model` first.")

        # Prepare submission data
        submission_file = f"{self.data_path}/SampleSubmissionStage1.csv"
        if os.path.exists(submission_file):
            submission = pd.read_csv(submission_file)
        else:
            print(f"Warning: {submission_file} not found. Creating a new submission template.")
            submission = pd.DataFrame({
                'ID': [f"{season}_{t1}_{t2}" for t1, t2 in zip(self.tourney_data['T1_TeamID'], self.tourney_data['T2_TeamID'])],
                'Pred': [0.5] * len(self.tourney_data)
            })

        # Merge features
        submission['Season'] = submission['ID'].apply(lambda x: int(x.split('_')[0]))
        submission['T1_TeamID'] = submission['ID'].apply(lambda x: int(x.split('_')[1]))
        submission['T2_TeamID'] = submission['ID'].apply(lambda x: int(x.split('_')[2]))

        submission = pd.merge(submission, self.tourney_data, on=['Season', 'T1_TeamID', 'T2_TeamID'], how='left')

        # Predict probabilities
        X_sub = submission[self.features].values
        dtest = xgb.DMatrix(X_sub)
        submission['Pred'] = self.model.predict(dtest)

        # Clip predictions to [0.05, 0.95]
        submission['Pred'] = submission['Pred'].clip(0.05, 0.95)

        # Save submission file
        submission[['ID', 'Pred']].to_csv('submission.csv', index=False)
        print("Submission file created successfully.")

    def run_all(self):
        """Run the entire pipeline."""
        self.load_data()
        self.feature_engineering()
        self.train_model()
        self.generate_submission()



# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import os
# import xgboost as xgb
# from sklearn.model_selection import KFold
# from sklearn.metrics import log_loss
# from scipy.interpolate import UnivariateSpline
# import statsmodels.api as sm
# import warnings

# warnings.filterwarnings('ignore')
# pd.set_option("display.max_column", 999)

# class MarchManiaPredictor:
#     def __init__(self, data_path):
#         """Initialize predictor with data path."""
#         self.data_path = data_path
#         self.tourney_results = None
#         self.seeds = None
#         self.regular_results = None
#         self.regular_data = None
#         self.tourney_data = None
#         self.model = None
#         self.features = None
#         self.elo = {}  # For potential Elo rating implementation

#     def load_data(self):
#         """Load and preprocess data with error handling."""
#         try:
#             def get_pd(file_name):
#                 file_path = os.path.join(self.data_path, file_name)
#                 if not os.path.exists(file_path):
#                     raise FileNotFoundError(f"Missing file: {file_path}")
#                 return pd.read_csv(file_path)

#             self.tourney_results = pd.concat([
#                 get_pd("MNCAATourneyDetailedResults.csv"),
#                 get_pd("WNCAATourneyDetailedResults.csv")
#             ], ignore_index=True)

#             self.seeds = pd.concat([
#                 get_pd("MNCAATourneySeeds.csv"),
#                 get_pd("WNCAATourneySeeds.csv")
#             ], ignore_index=True)

#             self.regular_results = pd.concat([
#                 get_pd("MRegularSeasonDetailedResults.csv"),
#                 get_pd("WRegularSeasonDetailedResults.csv")
#             ], ignore_index=True)

#             print("Data loading completed successfully.")
#             print(f"Tourney shape: {self.tourney_results.shape}")
#             print(f"Seeds shape: {self.seeds.shape}")
#             print(f"Regular shape: {self.regular_results.shape}")

#         except Exception as e:
#             print(f"Error loading data: {str(e)}")
#             raise

#     def prepare_data(self, df):
#         """Prepare and transform data for modeling with additional features."""
#         # Create swapped version with consistent column naming
#         dfswap = df.rename(columns={
#             'WTeamID': 'T2_TeamID', 'LTeamID': 'T1_TeamID',
#             'WScore': 'T2_Score', 'LScore': 'T1_Score'
#         })[df.columns]  # Maintain original column order
        
#         # Adjust location for swapped games
#         location_map = {'H': 'A', 'A': 'H', 'N': 'N'}
#         dfswap['WLoc'] = dfswap['WLoc'].map(location_map)
        
#         # Standardize column names
#         df.columns = [x.replace('W', 'T1_').replace('L', 'T2_') for x in df.columns]
#         dfswap.columns = [x.replace('W', 'T1_').replace('L', 'T2_') for x in dfswap.columns]
        
#         # Combine original and swapped data
#         output = pd.concat([df, dfswap]).reset_index(drop=True)
        
#         # Convert location to numeric
#         output['location'] = output['T1_Loc'].map({'H': 1, 'A': -1, 'N': 0})
        
#         # Calculate point differential and additional features
#         output['PointDiff'] = output['T1_Score'] - output['T2_Score']
#         output['T1_FG_Pct'] = output['T1_FGM'] / output['T1_FGA']
#         output['T2_FG_Pct'] = output['T2_FGM'] / output['T2_FGA']
        
#         return output

#     def feature_engineering(self):
#         """Enhanced feature engineering with additional metrics."""
#         self.regular_data = self.prepare_data(self.regular_results)
#         self.tourney_data = self.prepare_data(self.tourney_results)

#         # Define boxscore columns for aggregation
#         boxscore_cols = [
#             'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_FG_Pct',
#             'T1_OR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_PF',
#             'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_FG_Pct',
#             'T2_OR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk',
#             'PointDiff'
#         ]

#         # Calculate season statistics
#         season_stats = self.regular_data.groupby(["Season", 'T1_TeamID'])[boxscore_cols].agg([np.mean, np.median]).reset_index()
#         season_stats.columns = ['_'.join(col).strip() for col in season_stats.columns.values]

#         # Prepare T1 and T2 statistics
#         stats_t1 = season_stats.copy()
#         stats_t2 = season_stats.copy()
#         stats_t1.columns = ["T1_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in stats_t1.columns]
#         stats_t2.columns = ["T2_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in stats_t2.columns]
#         stats_t1.columns.values[0] = "Season"
#         stats_t2.columns.values[0] = "Season"

#         # Merge statistics with tourney data
#         self.tourney_data = pd.merge(self.tourney_data, stats_t1, on=['Season', 'T1_TeamID'], how='left')
#         self.tourney_data = pd.merge(self.tourney_data, stats_t2, on=['Season', 'T2_TeamID'], how='left')

#         # Add recent performance metrics
#         recent_games = self.regular_data[self.regular_data.DayNum > 118].copy()
#         recent_games['win'] = (recent_games['PointDiff'] > 0).astype(int)
        
#         win_ratio_t1 = recent_games.groupby(['Season', 'T1_TeamID'])['win'].mean().reset_index(name='T1_win_ratio_14d')
#         win_ratio_t2 = recent_games.groupby(['Season', 'T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio_14d')
        
#         self.tourney_data = pd.merge(self.tourney_data, win_ratio_t1, on=['Season', 'T1_TeamID'], how='left')
#         self.tourney_data = pd.merge(self.tourney_data, win_ratio_t2, on=['Season', 'T2_TeamID'], how='left')

#         # Update feature list
#         self.features = [col for col in self.tourney_data.columns if col.startswith(('T1_', 'T2_')) and col not in ['T1_TeamID', 'T2_TeamID']]

#     def train_model(self):
#         """Train XGBoost model with improved parameters."""
#         y = (self.tourney_data['PointDiff'] > 0).astype(int)  # Convert to binary classification
#         X = self.tourney_data[self.features].fillna(0)  # Handle missing values
        
#         dtrain = xgb.DMatrix(X, label=y)

#         param = {
#             'objective': 'binary:logistic',
#             'eval_metric': 'logloss',
#             'booster': 'gbtree',
#             'eta': 0.05,
#             'subsample': 0.8,
#             'colsample_bytree': 0.8,
#             'max_depth': 4,
#             'min_child_weight': 10,
#             'gamma': 1,
#             'seed': 42
#         }

#         # Cross-validation
#         cv_results = xgb.cv(
#             params=param,
#             dtrain=dtrain,
#             num_boost_round=1000,
#             nfold=5,
#             metrics=['logloss'],
#             early_stopping_rounds=50,
#             verbose_eval=50,
#             seed=42
#         )

#         best_iteration = cv_results['test-logloss-mean'].argmin()
#         print(f"Best iteration: {best_iteration}, Best logloss: {cv_results['test-logloss-mean'].min()}")

#         # Train final model
#         self.model = xgb.train(
#             params=param,
#             dtrain=dtrain,
#             num_boost_round=best_iteration
#         )

#     def generate_submission(self, season=2024):
#         """Generate submission file with validation."""
#         if self.model is None:
#             raise ValueError("Model not trained. Call train_model() first.")

#         # Load or create submission template
#         submission_file = os.path.join(self.data_path, "SampleSubmissionStage1.csv")
#         if os.path.exists(submission_file):
#             submission = pd.read_csv(submission_file)
#         else:
#             submission = pd.DataFrame(columns=['ID', 'Pred'])

#         # Parse IDs
#         submission[['Season', 'T1_TeamID', 'T2_TeamID']] = submission['ID'].str.split('_', expand=True).astype(int)
        
#         # Merge with features
#         submission = pd.merge(submission[['ID', 'Season', 'T1_TeamID', 'T2_TeamID']], 
#                             self.tourney_data[self.features + ['Season', 'T1_TeamID', 'T2_TeamID']],
#                             on=['Season', 'T1_TeamID', 'T2_TeamID'], 
#                             how='left')

#         # Generate predictions
#         X_sub = submission[self.features].fillna(0)
#         dtest = xgb.DMatrix(X_sub)
#         submission['Pred'] = self.model.predict(dtest)
#         submission['Pred'] = submission['Pred'].clip(0.05, 0.95)

#         # Save results
#         submission[['ID', 'Pred']].to_csv('submission.csv', index=False)
#         print(f"Submission file created with {len(submission)} predictions.")

#     def run_all(self):
#         """Execute complete prediction pipeline."""
#         print("Starting prediction pipeline...")
#         self.load_data()
#         self.feature_engineering()
#         self.train_model()
#         self.generate_submission()
#         print("Pipeline completed.")

# if __name__ == "__main__":
#     # Example usage
#     predictor = MarchManiaPredictor(data_path="/path/to/data")
#     predictor.run_all()


if __name__ == "__main__":
    data_path = "/kaggle/input/march-machine-learning-mania-2025"
    predictor = MarchManiaPredictor(data_path)
    predictor.run_all()


import pandas as pd

# Load the submission file
submission_df = pd.read_csv('submission.csv')

# Display the first 5 rows
print(submission_df.head())

#Display the last 5 rows
print(submission_df.tail())


print(submission_df.tail())




