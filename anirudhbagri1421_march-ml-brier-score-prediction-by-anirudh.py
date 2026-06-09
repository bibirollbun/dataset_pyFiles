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


import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, brier_score_loss, mean_squared_error, roc_curve, auc
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

class TournamentPredictor:
    def __init__(self, data_dir):
        self.data_path = data_dir
        self.data = None
        self.teams = None
        self.seeds = None
        self.games = None
        self.sub = None
        self.gb = None
        self.col = None
        self.model = None # declare model here
        self.calibration_model = None # declare calibration model here
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

    def load_data(self):
        files = glob.glob(self.data_path)
        self.data = {p.split('/')[-1].split('.')[0]: pd.read_csv(p, encoding='latin-1') for p in files}

        teams = pd.concat([self.data['MTeams'], self.data['WTeams']])
        teams_spelling = pd.concat([self.data['MTeamSpellings'], self.data['WTeamSpellings']])
        teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()
        teams_spelling.columns = ['TeamID', 'TeamNameCount']
        self.teams = pd.merge(teams, teams_spelling, how='left', on=['TeamID'])

        season_cresults = pd.concat([self.data['MRegularSeasonCompactResults'], self.data['WRegularSeasonCompactResults']])
        season_dresults = pd.concat([self.data['MRegularSeasonDetailedResults'], self.data['WRegularSeasonDetailedResults']])
        tourney_cresults = pd.concat([self.data['MNCAATourneyCompactResults'], self.data['WNCAATourneyCompactResults']])
        tourney_dresults = pd.concat([self.data['MNCAATourneyDetailedResults'], self.data['WNCAATourneyDetailedResults']])

        seeds_df = pd.concat([self.data['MNCAATourneySeeds'], self.data['WNCAATourneySeeds']])
        self.seeds = {'_'.join(map(str, [int(k1), k2])): int(v[1:3]) for k1, v, k2 in seeds_df[['Season', 'Seed', 'TeamID']].values}

        self.sub = self.data['SampleSubmissionStage1']

        season_cresults['ST'] = 'S'
        season_dresults['ST'] = 'S'
        tourney_cresults['ST'] = 'T'
        tourney_dresults['ST'] = 'T'

        self.games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
        self.games['WLoc'] = self.games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})

        self.games['ID'] = self.games.apply(lambda r: '_'.join(map(str, [r['Season']] + sorted([r['WTeamID'], r['LTeamID']]))), axis=1)
        self.games['IDTeams'] = self.games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'], r['LTeamID']]))), axis=1)
        self.games['Team1'] = self.games.apply(lambda r: sorted([r['WTeamID'], r['LTeamID']])[0], axis=1)
        self.games['Team2'] = self.games.apply(lambda r: sorted([r['WTeamID'], r['LTeamID']])[1], axis=1)
        self.games['IDTeam1'] = self.games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
        self.games['IDTeam2'] = self.games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
        self.games['Team1Seed'] = self.games['IDTeam1'].map(self.seeds).fillna(0)
        self.games['Team2Seed'] = self.games['IDTeam2'].map(self.seeds).fillna(0)
        self.games['ScoreDiff'] = self.games['WScore'] - self.games['LScore']
        self.games['Pred'] = self.games.apply(lambda r: 1.0 if sorted([r['WTeamID'], r['LTeamID']])[0] == r['WTeamID'] else 0.0, axis=1)
        self.games['ScoreDiffNorm'] = self.games.apply(lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0.0 else r['ScoreDiff'], axis=1)
        self.games['SeedDiff'] = self.games['Team1Seed'] - self.games['Team2Seed']
        self.games = self.games.fillna(-1)

        c_score_col = ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
        c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
        self.gb = self.games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
        self.gb.columns = [''.join(c) + '_c_score' for c in self.gb.columns]

        self.games = self.games[self.games['ST'] == 'T']
        self.sub['WLoc'] = 3
        self.sub['Season'] = self.sub['ID'].map(lambda x: x.split('_')[0]).astype(int)
        self.sub['Team1'] = self.sub['ID'].map(lambda x: x.split('_')[1])
        self.sub['Team2'] = self.sub['ID'].map(lambda x: x.split('_')[2])
        self.sub['IDTeams'] = self.sub.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)
        self.sub['IDTeam1'] = self.sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
        self.sub['IDTeam2'] = self.sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
        self.sub['Team1Seed'] = self.sub['IDTeam1'].map(self.seeds).fillna(0)
        self.sub['Team2Seed'] = self.sub['IDTeam2'].map(self.seeds).fillna(0)
        self.sub['SeedDiff'] = self.sub['Team1Seed'] - self.sub['Team2Seed']
        self.sub = self.sub.fillna(-1)

        self.games = pd.merge(self.games, self.gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
        self.sub = pd.merge(self.sub, self.gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')

        exclude_cols = ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 'ScoreDiffNorm', 'WLoc'] + c_score_col
        self.col = [c for c in self.games.columns if c not in exclude_cols]
        print("Data loading and preprocessing completed.")

    def create_models(self):
        self.model = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.calibration_model = XGBClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42, n_jobs=-1)

    def train_model(self):
        X = self.games[self.col].fillna(-1)
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        y = self.games['Pred']

        X_train, X_cal, y_train, y_cal = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        train_preds = self.model.predict_proba(X_train)[:, 1].clip(0.001, 0.999)

        cal_preds = self.model.predict_proba(X_cal)[:, 1].clip(0.001, 0.999)
        self.calibration_model.fit(cal_preds.reshape(-1, 1), y_cal)

        train_preds_calibrated = self.calibration_model.predict_proba(train_preds.reshape(-1, 1))[:, 1].clip(0.001, 0.999)

        print(f'Log Loss (Train): {log_loss(y_train, train_preds_calibrated):.4f}')
        print(f'Brier Score (Train): {brier_score_loss(y_train, train_preds_calibrated):.4f}')
        print(f'MSE (Train): {mean_squared_error(y_train, train_preds_calibrated):.4f}')

        # Plot ROC Curve for the calibration set.
        self.plot_roc_curve(y_cal, cal_preds, "Calibration Set ROC Curve")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_mse_scores = []
        cv_logloss_scores = []
        for train_index, val_index in kf.split(X_scaled):
            X_train, X_val = X_scaled[train_index], X_scaled[val_index]
            y_train, y_val = y[train_index], y[val_index]

            self.model.fit(X_train, y_train)

            val_preds = self.model.predict_proba(X_val)[:, 1].clip(0.001, 0.999)

            self.calibration_model.fit(val_preds.reshape(-1, 1), y_val)
            val_preds_calibrated = self.calibration_model.predict_proba(val_preds.reshape(-1, 1))[:, 1].clip(0.001, 0.999)

            mse = mean_squared_error(y_val, val_preds_calibrated)
            logloss = log_loss(y_val, val_preds_calibrated)

            cv_mse_scores.append(mse)
            cv_logloss_scores.append(logloss)

        print(f'Cross-validated MSE: {np.mean(cv_mse_scores):.4f}')
        print(f'Cross-validated LogLoss: {np.mean(cv_logloss_scores):.4f}')

        feature_importances = self.model.feature_importances_
        feature_names = self.col
        self.plot_feature_importance(feature_importances, feature_names)

        self.plot_calibration_curve(y_cal, cal_preds)

        # Plot the distribution of calibrated predictions.
        self.plot_prediction_distribution(train_preds_calibrated, "Distribution of Calibrated Training Predictions")

    def predict_submission(self, output_file='submission.csv'):
        sub_X = self.sub[self.col].fillna(-1)
        sub_X_imputed = self.imputer.transform(sub_X)
        sub_X_scaled = self.scaler.transform(sub_X_imputed)
        preds = self.model.predict_proba(sub_X_scaled)[:, 1].clip(0.001, 0.999)
        preds_calibrated = self.calibration_model.predict_proba(preds.reshape(-1, 1))[:, 1].clip(0.001, 0.999)

        self.sub['Pred'] = preds_calibrated
        self.sub[['ID', 'Pred']].to_csv(output_file, index=False)
        print(f"Submission file saved to {output_file}")

    def plot_feature_importance(self, importances, feature_names, top_n=20):
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(top_n)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
        plt.title('Top {} Feature Importances'.format(top_n))
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    def plot_calibration_curve(self, y_true, y_proba, n_bins=10):
        combined = np.stack([y_proba, y_true], axis=-1)
        combined = combined[np.argsort(combined[:, 0])]
        sorted_probas = combined[:, 0]
        sorted_true = combined[:, 1]

        bins = np.linspace(0, 1, n_bins + 1)
        bin_midpoints = bins[:-1] + (bins[1] - bins[0]) / 2
        bin_assignments = np.digitize(sorted_probas, bins) - 1
        bin_sums = np.bincount(bin_assignments, weights=sorted_probas, minlength=n_bins)
        bin_true = np.bincount(bin_assignments, weights=sorted_true, minlength=n_bins)
        bin_total = np.bincount(bin_assignments, minlength=n_bins)

        fraction_of_positives = bin_true / bin_total
        fraction_of_positives[np.isnan(fraction_of_positives)] = 0

        plt.figure(figsize=(8, 6))
        plt.plot(bin_midpoints, fraction_of_positives, marker='o', label='Calibration Curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')

        plt.xlabel('Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_prediction_distribution(self, predictions, title="Distribution of Predictions"):
        plt.figure(figsize=(8, 6))
        sns.histplot(predictions, kde=True, color='skyblue')
        plt.title(title)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, y_true, y_proba, title="ROC Curve"):
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    def run_all(self):
        self.load_data()
        self.create_models()
        self.train_model()
        self.predict_submission()

# Example usage:
if __name__ == "__main__":
    data_dir = '/kaggle/input/march-machine-learning-mania-2025/**'  # Or a local dir
    predictor = TournamentPredictor(data_dir)
    predictor.run_all()

