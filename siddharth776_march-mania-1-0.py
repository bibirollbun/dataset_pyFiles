import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_absolute_error, brier_score_loss, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings("ignore")

# Custom scorer that uses predict instead of predict_proba
def custom_log_loss_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    # Ensure predictions are within valid probability bounds
    y_pred = np.clip(y_pred, 0.001, 0.999)
    # We return the negative log loss (since higher is better for scoring in RandomizedSearchCV)
    return -log_loss(y, y_pred)

custom_scorer = make_scorer(custom_log_loss_scorer, greater_is_better=True)

class TournamentPredictor:
    def __init__(self, data_path):
        self.data_path = data_path  # e.g. '/kaggle/input/march-machine-learning-mania-2025/**'
        self.data = None
        self.teams = None
        self.seeds = None
        self.games = None
        self.sub = None
        self.gb = None
        self.col = None
        self.calibrator = None
        
        # Build a pipeline for preprocessing and modeling
        self.model_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(
                n_estimators=235, 
                random_state=42, 
                max_depth=15,          # limit depth to prevent overfitting
                min_samples_split=2,   # require more samples to split
                max_features='auto',   # use sqrt(n_features) for better randomness
                n_jobs=-1              # parallel processing
            ))
        ])

    def load_data(self):
        # Load CSV files into a dictionary
        files = glob.glob(self.data_path)
        self.data = {
            p.split('/')[-1].split('.')[0]: pd.read_csv(p, encoding='latin-1')
            for p in files
        }
        
        # Process teams and team spellings
        teams = pd.concat([self.data['MTeams'], self.data['WTeams']])
        teams_spelling = pd.concat([self.data['MTeamSpellings'], self.data['WTeamSpellings']])
        teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()
        teams_spelling.columns = ['TeamID', 'TeamNameCount']
        self.teams = pd.merge(teams, teams_spelling, how='left', on=['TeamID'])
        del teams_spelling
        
        # Concatenate season and tournament results
        season_cresults = pd.concat([self.data['MRegularSeasonCompactResults'], self.data['WRegularSeasonCompactResults']])
        season_dresults = pd.concat([self.data['MRegularSeasonDetailedResults'], self.data['WRegularSeasonDetailedResults']])
        tourney_cresults = pd.concat([self.data['MNCAATourneyCompactResults'], self.data['WNCAATourneyCompactResults']])
        tourney_dresults = pd.concat([self.data['MNCAATourneyDetailedResults'], self.data['WNCAATourneyDetailedResults']])
        
        # Load seeds and build a lookup dictionary
        seeds_df = pd.concat([self.data['MNCAATourneySeeds'], self.data['WNCAATourneySeeds']])
        self.seeds = {
            '_'.join(map(str, [int(season), team_id])): int(seed[1:3])
            for season, seed, team_id in seeds_df[['Season', 'Seed', 'TeamID']].values
        }
        
        self.sub = self.data['SampleSubmissionStage2']
        
        # Mark results as season (S) or tournament (T)
        season_cresults['ST'] = 'S'
        season_dresults['ST'] = 'S'
        tourney_cresults['ST'] = 'T'
        tourney_dresults['ST'] = 'T'
        
        # Use detailed results for processing
        self.games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
        self.games.reset_index(drop=True, inplace=True)
        self.games['WLoc'] = self.games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})
        
        # Create IDs and team-related features
        self.games['ID'] = self.games.apply(
            lambda r: '_'.join(map(str, [r['Season']] + sorted([r['WTeamID'], r['LTeamID']])))
            , axis=1
        )
        self.games['IDTeams'] = self.games.apply(
            lambda r: '_'.join(map(str, sorted([r['WTeamID'], r['LTeamID']])))
            , axis=1
        )
        self.games['Team1'] = self.games.apply(
            lambda r: sorted([r['WTeamID'], r['LTeamID']])[0], axis=1
        )
        self.games['Team2'] = self.games.apply(
            lambda r: sorted([r['WTeamID'], r['LTeamID']])[1], axis=1
        )
        self.games['IDTeam1'] = self.games.apply(
            lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1
        )
        self.games['IDTeam2'] = self.games.apply(
            lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1
        )
        self.games['Team1Seed'] = self.games['IDTeam1'].map(self.seeds).fillna(0)
        self.games['Team2Seed'] = self.games['IDTeam2'].map(self.seeds).fillna(0)
        
        # Create additional features
        self.games['ScoreDiff'] = self.games['WScore'] - self.games['LScore']
        self.games['Pred'] = self.games.apply(
            lambda r: 1.0 if sorted([r['WTeamID'], r['LTeamID']])[0] == r['WTeamID'] else 0.0, axis=1
        )
        self.games['ScoreDiffNorm'] = self.games.apply(
            lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0.0 else r['ScoreDiff'], axis=1
        )
        self.games['SeedDiff'] = self.games['Team1Seed'] - self.games['Team2Seed']
        self.games = self.games.fillna(-1)
        
        # Aggregate game statistics by team pairing
        c_score_col = [
            'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 
            'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 
            'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF'
        ]
        c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
        self.gb = self.games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
        self.gb.columns = [''.join(c) + '_c_score' for c in self.gb.columns]
        
        # Keep only tournament games
        self.games = self.games[self.games['ST'] == 'T']
        
        # Process submission data
        self.sub['WLoc'] = 3
        self.sub['Season'] = self.sub['ID'].map(lambda x: int(x.split('_')[0]))
        self.sub['Team1'] = self.sub['ID'].map(lambda x: x.split('_')[1])
        self.sub['Team2'] = self.sub['ID'].map(lambda x: x.split('_')[2])
        self.sub['IDTeams'] = self.sub.apply(
            lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1
        )
        self.sub['IDTeam1'] = self.sub.apply(
            lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1
        )
        self.sub['IDTeam2'] = self.sub.apply(
            lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1
        )
        self.sub['Team1Seed'] = self.sub['IDTeam1'].map(self.seeds).fillna(0)
        self.sub['Team2Seed'] = self.sub['IDTeam2'].map(self.seeds).fillna(0)
        self.sub['SeedDiff'] = self.sub['Team1Seed'] - self.sub['Team2Seed']
        self.sub = self.sub.fillna(-1)
        
        # Merge aggregated stats into games and submission datasets
        self.games = pd.merge(self.games, self.gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
        self.sub = pd.merge(self.sub, self.gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
        
        # Define feature columns (exclude identifiers and raw score columns)
        exclude_cols = [
            'ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2',
            'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 
            'ScoreDiffNorm', 'WLoc'
        ] + c_score_col
        self.col = [c for c in self.games.columns if c not in exclude_cols]
        print("Data loading and preprocessing completed.")

    def tune_hyperparameters(self, X, y):
        # Define parameter space for the RandomForestRegressor within our pipeline
        param_distributions = {
            'model__n_estimators': [200, 235, 250, 300],
            'model__max_depth': [10, 15, 20, None],
            'model__min_samples_split': [2, 4, 6, 8],
            'model__max_features': ['auto', 'sqrt', 'log2']
        }
        search = RandomizedSearchCV(
            self.model_pipeline, 
            param_distributions=param_distributions, 
            n_iter=10, 
            cv=5, 
            scoring=custom_scorer, 
            random_state=42, 
            n_jobs=-1
        )
        search.fit(X, y)
        self.model_pipeline = search.best_estimator_
        print("Best hyperparameters found:", search.best_params_)

    def train_model(self):
        # Prepare training features and target
        X = self.games[self.col].fillna(-1)
        y = self.games['Pred']
        
        # Split into training and calibration sets
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Hyperparameter tuning on the training set
        self.tune_hyperparameters(X_train, y_train)
        
        # Refit the optimized pipeline on the training set
        self.model_pipeline.fit(X_train, y_train)
        
        # Generate predictions on training and calibration sets
        preds_train = self.model_pipeline.predict(X_train).clip(0.001, 0.999)
        preds_calib = self.model_pipeline.predict(X_calib).clip(0.001, 0.999)
        
        # Calibrate predictions using isotonic regression on the calibration set
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(preds_calib, y_calib)
        preds_train_cal = self.calibrator.transform(preds_train)
        
        # Output evaluation metrics on the training set
        print(f'Log Loss: {log_loss(y_train, preds_train_cal):.4f}')
        print(f'Mean Absolute Error: {mean_absolute_error(y_train, preds_train_cal):.4f}')
        print(f'Brier Score: {brier_score_loss(y_train, preds_train_cal):.4f}')
        cv_scores = cross_val_score(self.model_pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
        print(f'Cross-validated MSE: {-cv_scores.mean():.4f}')

    def predict_submission(self, output_file='submission.csv'):
        # Prepare submission features
        sub_X = self.sub[self.col].fillna(-1)
        preds = self.model_pipeline.predict(sub_X).clip(0.01, 0.99)
        
        # Calibrate predictions using the stored calibrator
        if self.calibrator:
            preds_cal = self.calibrator.transform(preds)
        else:
            preds_cal = preds
        
        self.sub['Pred'] = preds_cal
        self.sub[['ID', 'Pred']].to_csv(output_file, index=False)
        print(f"Submission file saved to {output_file}")

    def run_all(self):
        self.load_data()
        self.train_model()
        self.predict_submission()

# Example usage:
if __name__ == "__main__":
    data_path = '/kaggle/input/march-machine-learning-mania-2025/**'
    predictor = TournamentPredictor(data_path)
    predictor.run_all()


