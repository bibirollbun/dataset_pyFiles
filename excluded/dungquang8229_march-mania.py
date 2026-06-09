import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, brier_score_loss, mean_squared_error, roc_curve, auc
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
import optuna

import warnings
warnings.filterwarnings("ignore")


data_dir = '/kaggle/input/march-machine-learning-mania-2025/**'


# Parameters 
XGB_params = {'tree_mothod': 'gpu_hist', 'random_state': 42, 'n_estimators': 868, 'learning_rate': 0.010393149998639654, 'max_depth': 3, 'subsample': 0.9498234342455986, 'colsample_bytree': 0.8264271927329041, 'gamma': 0.9605760745782711, 'min_child_weight': 6, 'lambda': 0.05065692390862945, 'alpha': 0.9352163839243699}
LGB_params = {'device': 'gpu', 'verbose': -1, 'random_state': 42, 'n_estimators': 546, 'learning_rate': 0.005122294431470118, 'max_depth': 2, 'num_leaves': 185, 'subsample': 0.9973576609124172, 'colsample_bytree': 0.8360991232769169, 'min_child_samples': 44, 'lambda_l1': 0.5390328427385757, 'lambda_l2': 2.1068143900093554}
Cat_params = {'task_type': "GPU", 'verbose': False, "random_seed": 42, 'iterations': 300, 'learning_rate': 0.006436761924136745, 'depth': 8, 'l2_leaf_reg': 7.260145406582964, 'random_strength': 3.823150758602715, 'bagging_temperature': 0.9780053259230543, 'min_data_in_leaf': 48}


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
        self.model = None
        self.calibration_models = None
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

        self.sub = self.data['SampleSubmissionStage2']

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

        exclude_cols = ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 'ScoreDiffNorm', 'WLoc', 'IDTeams_c_score'] + c_score_col
        self.col = [c for c in self.games.columns if c not in exclude_cols]
        print("Data loading and preprocessing completed.")

    def create_models(self):
        self.models = []
        self.calibration_models = []

        # Parameters 
        XGB_params = {'tree_method': 'gpu_hist', 'random_state': 42, 'n_estimators': 578, 'learning_rate': 0.010393149998639654, 'max_depth': 3, 'subsample': 0.9498234342455986, 'colsample_bytree': 0.8264271927329041, 'gamma': 0.9605760745782711, 'min_child_weight': 6, 'lambda': 0.05065692390862945, 'alpha': 0.9352163839243699}
        LGB_params = {'device': 'gpu', 'verbose': -1, 'random_state': 42, 'n_estimators': 546, 'learning_rate': 0.005122294431470118, 'max_depth': 2, 'num_leaves': 185, 'subsample': 0.9973576609124172, 'colsample_bytree': 0.8360991232769169, 'min_child_samples': 44, 'lambda_l1': 0.5390328427385757, 'lambda_l2': 2.1068143900093554}
        Cat_params = {'task_type': "GPU", 'verbose': False, "random_seed": 42, 'iterations': 300, 'learning_rate': 0.006436761924136745, 'depth': 6, 'l2_leaf_reg': 7.260145406582964, 'random_strength': 3.823150758602715, 'bagging_temperature': 0.9780053259230543, 'min_data_in_leaf': 48}
    
        for _ in range(5):
            xgb = XGBRegressor(**XGB_params)
            lgb = LGBMRegressor(**LGB_params)
            cat = CatBoostRegressor(**Cat_params)

            calibration_xgb = IsotonicRegression(out_of_bounds='clip')
            calibration_lgb = IsotonicRegression(out_of_bounds='clip')
            calibration_cat = IsotonicRegression(out_of_bounds='clip')

            self.models.append({"xgb": xgb, "lgb": lgb, "cat": cat})
            self.calibration_models.append({
                "xgb": calibration_xgb,
                "lgb": calibration_lgb,
                "cat": calibration_cat
            })
    
        print("Models creation completed.")

    def fit(self, i, X_train, y_train, X_cal, y_cal):
        """Huấn luyện mô hình và calibration model."""
        xgb, lgb, cat = self.models[i]["xgb"], self.models[i]["lgb"], self.models[i]["cat"]
        calibration_xgb, calibration_lgb, calibration_cat = (
            self.calibration_models[i]["xgb"],
            self.calibration_models[i]["lgb"],
            self.calibration_models[i]["cat"],
        )

        xgb.fit(X_train, y_train)
        lgb.fit(X_train, y_train)
        cat.fit(X_train, y_train)

        y_cal_xgb = xgb.predict(X_cal).clip(0.001, 0.999)
        y_cal_lgb = lgb.predict(X_cal).clip(0.001, 0.999)
        y_cal_cat = cat.predict(X_cal).clip(0.001, 0.999)

        calibration_xgb.fit(y_cal_xgb, y_cal)
        calibration_lgb.fit(y_cal_lgb, y_cal)
        calibration_cat.fit(y_cal_cat, y_cal)

    def predict(self, i, X_test):
        """Dự đoán bằng mô hình ensemble với calibration."""
        predictions = []
        xgb, lgb, cat = self.models[i]["xgb"], self.models[i]["lgb"], self.models[i]["cat"]
        calibration_xgb, calibration_lgb, calibration_cat = (
            self.calibration_models[i]["xgb"],
            self.calibration_models[i]["lgb"],
            self.calibration_models[i]["cat"],
        )

        y_pred_xgb = xgb.predict(X_test).clip(0.001, 0.999)
        y_pred_lgb = lgb.predict(X_test).clip(0.001, 0.999)
        y_pred_cat = cat.predict(X_test).clip(0.001, 0.999)

        y_pred_xgb = calibration_xgb.predict(y_pred_xgb)
        y_pred_lgb = calibration_lgb.predict(y_pred_lgb)
        y_pred_cat = calibration_cat.predict(y_pred_cat)

        y_pred_ensemble = (0.4 * y_pred_xgb +
                           0.2 * y_pred_lgb +
                           0.4 * y_pred_cat).clip(0.001, 0.999)

        return y_pred_ensemble

    def train_model(self):
        X = self.games[self.col].fillna(-1)
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        y = self.games['Pred']

        # kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # for i, (train_idx, cal_idx) in enumerate(kf.split(X_scaled)):
        #     X_train, X_cal = X_scaled[train_idx], X_scaled[cal_idx]
        #     y_train, y_cal = y.iloc[train_idx], y.iloc[cal_idx]
            
        #     self.fit(i, X_train, y_train, X_cal, y_cal)
            
        kf = KFold(n_splits=5, shuffle=True, random_state=43)
        cv_mse_scores = []
        cv_logloss_scores = []
        cv_brier_scores = []
    
        y_val_all = []
        val_preds_all = []
    
        for i, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            X_train_full, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train_full, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
            X_train, X_cal, y_train, y_cal = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
    
            self.fit(i, X_train, y_train, X_cal, y_cal)
    
            val_preds = self.predict(i, X_val)
    
            y_val_all.extend(y_val)
            val_preds_all.extend(val_preds)
    
            mse = mean_squared_error(y_val, val_preds)
            logloss = log_loss(y_val, val_preds)
    
            print(f"Fold {i+1} results: MSE {mse:.4f}, LogLoss {logloss:.4f}")
    
            cv_mse_scores.append(mse)
            cv_logloss_scores.append(logloss)
            
        print(f'Cross-validated MSE: {np.mean(cv_mse_scores):.4f}')
        print(f'Cross-validated LogLoss: {np.mean(cv_logloss_scores):.4f}')

        self.plot_roc_curve(y_val_all, val_preds_all, "ROC Curve after Calibration")
        self.plot_calibration_curve(y_val_all, val_preds_all)
        # feature_importances = self.models[-1].feature_importances_
        # feature_names = self.col
        # self.plot_feature_importance(feature_importances, feature_names)
        self.plot_prediction_distribution(val_preds_all, "Distribution of Calibrated Predictions")

    def predict_submission(self, output_file='submission.csv'):
        sub_X = self.sub[self.col].fillna(-1)
        sub_X_imputed = self.imputer.transform(sub_X)
        sub_X_scaled = self.scaler.transform(sub_X_imputed)
    
        preds = []
    
        for i in range(5):
            pred = self.predict(i, sub_X_scaled)
            preds.append(pred)

        # weights = np.array([0.13333, 0.3, 0.13333, 0.3, 0.13333])
        final_pred = np.average(np.array(preds), axis=0)
        print("Final prediction shape:", final_pred.shape)
        self.sub['Pred'] = final_pred
        self.sub[['ID', 'Pred']].to_csv(output_file, index=False)
        self.plot_prediction_distribution(final_pred, "Final Preds Distribution")
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
        """Plots the distribution of model predictions."""
        plt.figure(figsize=(8, 6))
        sns.histplot(predictions, kde=True, color='skyblue')
        plt.title(title)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, y_true, y_proba, title="ROC Curve"):
      """Plots the Receiver Operating Characteristic (ROC) curve."""
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


data_dir = '/kaggle/input/march-machine-learning-mania-2025/**'
predictor = TournamentPredictor(data_dir)
# predictor.run_all()
predictor.load_data()
predictor.create_models()


predictor.train_model() 


predictor.predict_submission()

