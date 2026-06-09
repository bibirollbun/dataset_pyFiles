import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_absolute_error, brier_score_loss
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression


class MatchOutcomeEstimator:
    def __init__(self, path_pattern):
        self.path_pattern = path_pattern
        self.raw = {}
        self.meta = {}
        self.games = None
        self.test_data = None
        self.features = None

        self.fill_na = SimpleImputer(strategy='mean')
        self.normalize = StandardScaler()
        self.clf = RandomForestRegressor(
            n_estimators=250,
            random_state=42,
            max_depth=20,
            min_samples_split=3,
            max_features='sqrt'
        )

    def ingest(self):
        file_paths = glob.glob(self.path_pattern)
        self.raw = {
            p.split('/')[-1].split('.')[0]: pd.read_csv(p, encoding='latin-1') for p in file_paths
        }

        teams = pd.concat([self.raw['MTeams'], self.raw['WTeams']])
        alt_names = pd.concat([self.raw['MTeamSpellings'], self.raw['WTeamSpellings']])
        alt_names = alt_names.groupby('TeamID', as_index=False)['TeamNameSpelling'].count()
        alt_names.columns = ['TeamID', 'TeamNameCount']
        self.meta['teams'] = teams.merge(alt_names, on='TeamID', how='left')

        reg_simple = pd.concat([self.raw['MRegularSeasonCompactResults'], self.raw['WRegularSeasonCompactResults']])
        reg_detailed = pd.concat([self.raw['MRegularSeasonDetailedResults'], self.raw['WRegularSeasonDetailedResults']])
        tourney_simple = pd.concat([self.raw['MNCAATourneyCompactResults'], self.raw['WNCAATourneyCompactResults']])
        tourney_detailed = pd.concat([self.raw['MNCAATourneyDetailedResults'], self.raw['WNCAATourneyDetailedResults']])

        seeds_data = pd.concat([self.raw['MNCAATourneySeeds'], self.raw['WNCAATourneySeeds']])
        self.meta['seeds'] = {
            f"{int(season)}_{tid}": int(seed[1:3])
            for season, seed, tid in seeds_data[['Season', 'Seed', 'TeamID']].values
        }

        reg_detailed['ST'] = 'S'
        tourney_detailed['ST'] = 'T'
        combined_results = pd.concat([reg_detailed, tourney_detailed], ignore_index=True)
        combined_results['WLoc'] = combined_results['WLoc'].map({'A': 1, 'H': 2, 'N': 3})

        combined_results['ID'] = combined_results.apply(
            lambda row: f"{row['Season']}_{min(row['WTeamID'], row['LTeamID'])}_{max(row['WTeamID'], row['LTeamID'])}",
            axis=1
        )
        combined_results['IDTeams'] = combined_results.apply(
            lambda row: f"{min(row['WTeamID'], row['LTeamID'])}_{max(row['WTeamID'], row['LTeamID'])}",
            axis=1
        )
        combined_results['Team1'] = combined_results[['WTeamID', 'LTeamID']].min(axis=1)
        combined_results['Team2'] = combined_results[['WTeamID', 'LTeamID']].max(axis=1)

        combined_results['IDTeam1'] = combined_results.apply(
            lambda r: f"{r['Season']}_{r['Team1']}", axis=1
        )
        combined_results['IDTeam2'] = combined_results.apply(
            lambda r: f"{r['Season']}_{r['Team2']}", axis=1
        )

        combined_results['Team1Seed'] = combined_results['IDTeam1'].map(self.meta['seeds']).fillna(0)
        combined_results['Team2Seed'] = combined_results['IDTeam2'].map(self.meta['seeds']).fillna(0)

        combined_results['Margin'] = combined_results['WScore'] - combined_results['LScore']
        combined_results['WinLabel'] = combined_results.apply(
            lambda r: 1.0 if r['Team1'] == r['WTeamID'] else 0.0, axis=1
        )
        combined_results['NormDiff'] = combined_results.apply(
            lambda r: r['Margin'] if r['WinLabel'] == 1 else -r['Margin'], axis=1
        )
        combined_results['SeedDelta'] = combined_results['Team1Seed'] - combined_results['Team2Seed']
        combined_results.fillna(-1, inplace=True)

        score_fields = [
            'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst',
            'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
            'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF'
        ]
        agg_funcs = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
        stats_agg = combined_results.groupby('IDTeams').agg({col: agg_funcs for col in score_fields}).reset_index()
        stats_agg.columns = [''.join(col) + '_agg' for col in stats_agg.columns]

        combined_results = combined_results[combined_results['ST'] == 'T']

        self.test_data = self.raw['SampleSubmissionStage1'].copy()
        self.test_data['WLoc'] = 3
        self.test_data['Season'] = self.test_data['ID'].apply(lambda x: int(x.split('_')[0]))
        self.test_data['Team1'] = self.test_data['ID'].apply(lambda x: x.split('_')[1])
        self.test_data['Team2'] = self.test_data['ID'].apply(lambda x: x.split('_')[2])
        self.test_data['IDTeams'] = self.test_data.apply(lambda r: f"{r['Team1']}_{r['Team2']}", axis=1)
        self.test_data['IDTeam1'] = self.test_data.apply(lambda r: f"{r['Season']}_{r['Team1']}", axis=1)
        self.test_data['IDTeam2'] = self.test_data.apply(lambda r: f"{r['Season']}_{r['Team2']}", axis=1)
        self.test_data['Team1Seed'] = self.test_data['IDTeam1'].map(self.meta['seeds']).fillna(0)
        self.test_data['Team2Seed'] = self.test_data['IDTeam2'].map(self.meta['seeds']).fillna(0)
        self.test_data['SeedDelta'] = self.test_data['Team1Seed'] - self.test_data['Team2Seed']
        self.test_data.fillna(-1, inplace=True)

        combined_results = combined_results.merge(stats_agg, left_on='IDTeams', right_on='IDTeams_agg', how='left')
        self.test_data = self.test_data.merge(stats_agg, left_on='IDTeams', right_on='IDTeams_agg', how='left')

        skip_cols = [
            'ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2',
            'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'WinLabel', 'Margin',
            'NormDiff', 'WLoc'
        ] + score_fields

        self.games = combined_results
        self.features = [f for f in self.games.columns if f not in skip_cols]
        print("âœ”ï¸� Data ingestion and transformation done.")

    def fit(self):
        X_raw = self.games[self.features].fillna(-1)
        y_vals = self.games['WinLabel']

        X_prep = self.normalize.fit_transform(self.fill_na.fit_transform(X_raw))
        self.clf.fit(X_prep, y_vals)
        base_preds = self.clf.predict(X_prep).clip(0.001, 0.999)

        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(base_preds, y_vals)
        calibrated = calibrator.transform(base_preds)

        print(f'LogLoss: {log_loss(y_vals, calibrated):.4f}')
        print(f'MAE: {mean_absolute_error(y_vals, calibrated):.4f}')
        print(f'Brier Score: {brier_score_loss(y_vals, calibrated):.4f}')

        cv_score = cross_val_score(self.clf, X_prep, y_vals, scoring='neg_mean_squared_error', cv=5)
        print(f'Cross-validated MSE: {-cv_score.mean():.4f}')

        self.calibrator = calibrator

    def make_predictions(self, output='submission.csv'):
        test_features = self.test_data[self.features].fillna(-1)
        test_ready = self.normalize.transform(self.fill_na.transform(test_features))
        raw_preds = self.clf.predict(test_ready).clip(0.01, 0.99)
        final_preds = self.calibrator.transform(raw_preds)

        self.test_data['Pred'] = final_preds
        self.test_data[['ID', 'Pred']].to_csv(output, index=False)
        print(f"ğŸ“� Predictions saved to {output}")

    def execute_pipeline(self):
        self.ingest()
        self.fit()
        self.make_predictions()


if __name__ == "__main__":
    data_path = '/kaggle/input/march-machine-learning-mania-2025/**'
    estimator = MatchOutcomeEstimator(data_path)
    estimator.execute_pipeline()

