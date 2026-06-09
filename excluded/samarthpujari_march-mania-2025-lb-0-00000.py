import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_absolute_error, brier_score_loss
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression  # for calibration

import warnings
warnings.filterwarnings("ignore")

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
        
        # Preprocessing objects and model
        self.imputer = SimpleImputer(strategy='mean')  
        self.scaler = StandardScaler()
        # Tweak hyperparameters to help with calibration:
        self.model = RandomForestRegressor(
            n_estimators=235, 
            random_state=42, 
            max_depth=15,          # limit depth to prevent overfitting
            min_samples_split=2,  # require more samples to split
            max_features='auto'    # use sqrt(n_features) for better randomness
        )

    def load_data(self):
        # Load all CSV files into a dictionary
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
        
        # Concatenate season and tournament results (both compact and detailed)
        season_cresults = pd.concat([self.data['MRegularSeasonCompactResults'], self.data['WRegularSeasonCompactResults']])
        season_dresults = pd.concat([self.data['MRegularSeasonDetailedResults'], self.data['WRegularSeasonDetailedResults']])
        tourney_cresults = pd.concat([self.data['MNCAATourneyCompactResults'], self.data['WNCAATourneyCompactResults']])
        tourney_dresults = pd.concat([self.data['MNCAATourneyDetailedResults'], self.data['WNCAATourneyDetailedResults']])
        
        # Load seeds, cities, submission, etc.
        seeds_df = pd.concat([self.data['MNCAATourneySeeds'], self.data['WNCAATourneySeeds']])
        gcities = pd.concat([self.data['MGameCities'], self.data['WGameCities']])
        seasons = pd.concat([self.data['MSeasons'], self.data['WSeasons']])
        
        # Convert seeds dataframe to a dictionary with keys "Season_TeamID"
        self.seeds = {
            '_'.join(map(str, [int(k1), k2])): int(v[1:3])
            for k1, v, k2 in seeds_df[['Season', 'Seed', 'TeamID']].values
        }
        
        cities = self.data['Cities']
        self.sub = self.data['SampleSubmissionStage1']
        del seeds_df, cities  # free memory if needed
        
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
            lambda r: '_'.join(map(str, [r['Season']] + sorted([r['WTeamID'], r['LTeamID']]))), axis=1
        )
        self.games['IDTeams'] = self.games.apply(
            lambda r: '_'.join(map(str, sorted([r['WTeamID'], r['LTeamID']]))), axis=1
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
        self.sub['Season'] = self.sub['ID'].map(lambda x: x.split('_')[0]).astype(int)
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
        
        # Define the list of feature columns (exclude identifiers and raw score columns)
        exclude_cols = [
            'ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2',
            'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 
            'ScoreDiffNorm', 'WLoc'
        ] + c_score_col
        self.col = [c for c in self.games.columns if c not in exclude_cols]
        print("Data loading and preprocessing completed.")

    def train_model(self):
        # Prepare training features and target from games dataset
        X = self.games[self.col].fillna(-1)
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        y = self.games['Pred']
        
        # Fit the model
        self.model.fit(X_scaled, y)
        pred = self.model.predict(X_scaled).clip(0.001, 0.999)
        
        # -- Calibration step: Apply isotonic regression to adjust probabilities --
        # (Note: In practice, use a separate calibration set to avoid overfitting)
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(pred, y)
        pred_cal = ir.transform(pred)
        
        # Evaluation metrics using calibrated predictions
        print(f'Log Loss: {log_loss(y, pred_cal):.4f}')
        print(f'Mean Absolute Error: {mean_absolute_error(y, pred_cal):.4f}')
        print(f'Brier Score: {brier_score_loss(y, pred_cal):.4f}')
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
        print(f'Cross-validated MSE: {-cv_scores.mean():.4f}')

    def predict_submission(self, output_file='submission.csv'):
        # Prepare submission features and generate predictions
        sub_X = self.sub[self.col].fillna(-1)
        sub_X_imputed = self.imputer.transform(sub_X)
        sub_X_scaled = self.scaler.transform(sub_X_imputed)
        preds = self.model.predict(sub_X_scaled).clip(0.01, 0.99)
        # Optionally, apply the same isotonic calibration as above (using training fit)
        ir = IsotonicRegression(out_of_bounds='clip')
        # Refit calibration on training predictions for consistency:
        X_train = self.imputer.fit_transform(self.games[self.col].fillna(-1))
        X_train_scaled = self.scaler.fit_transform(X_train)
        train_preds = self.model.predict(X_train_scaled).clip(0.001, 0.999)
        ir.fit(train_preds, self.games['Pred'])
        preds_cal = ir.transform(preds)
        
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

